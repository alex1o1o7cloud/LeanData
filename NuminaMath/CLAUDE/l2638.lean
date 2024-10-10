import Mathlib

namespace existence_of_binomial_solution_l2638_263813

theorem existence_of_binomial_solution (a b : ℕ+) :
  ∃ (x y : ℕ+), Nat.choose (x + y) 2 = a * x + b * y := by
  sorry

end existence_of_binomial_solution_l2638_263813


namespace seventh_term_is_four_l2638_263840

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 2) / a (n + 1) = a (n + 1) / a n
  a₁_eq_one : a 1 = 1
  a₃_a₅_eq_4a₄_minus_4 : a 3 * a 5 = 4 * (a 4 - 1)

/-- The 7th term of the geometric sequence is 4 -/
theorem seventh_term_is_four (seq : GeometricSequence) : seq.a 7 = 4 := by
  sorry

end seventh_term_is_four_l2638_263840


namespace gcd_problem_l2638_263865

theorem gcd_problem (h : Nat.Prime 97) : Nat.gcd (97^9 + 1) (97^9 + 97^2 + 1) = 1 := by
  sorry

end gcd_problem_l2638_263865


namespace total_savings_theorem_l2638_263837

def weekday_savings : ℝ := 24
def weekend_savings : ℝ := 30
def monthly_subscription : ℝ := 45
def annual_interest_rate : ℝ := 0.03
def weeks_in_year : ℕ := 52
def days_in_year : ℕ := 365

def total_savings : ℝ :=
  let weekday_count : ℕ := days_in_year - 2 * weeks_in_year
  let weekend_count : ℕ := 2 * weeks_in_year
  let total_savings_before_interest : ℝ :=
    weekday_count * weekday_savings + weekend_count * weekend_savings - 12 * monthly_subscription
  total_savings_before_interest * (1 + annual_interest_rate)

theorem total_savings_theorem :
  total_savings = 9109.32 := by
  sorry

end total_savings_theorem_l2638_263837


namespace larry_stickers_l2638_263852

theorem larry_stickers (initial_stickers lost_stickers : ℕ) 
  (h1 : initial_stickers = 93)
  (h2 : lost_stickers = 6) :
  initial_stickers - lost_stickers = 87 := by
  sorry

end larry_stickers_l2638_263852


namespace correct_average_marks_l2638_263870

/-- Calculates the correct average marks for a class given the following conditions:
  * There are 40 students in the class
  * The reported average marks are 65
  * Three students' marks were wrongly noted:
    - First student: 100 instead of 20
    - Second student: 85 instead of 50
    - Third student: 15 instead of 55
-/
theorem correct_average_marks (num_students : ℕ) (reported_average : ℚ)
  (incorrect_mark1 incorrect_mark2 incorrect_mark3 : ℕ)
  (correct_mark1 correct_mark2 correct_mark3 : ℕ) :
  num_students = 40 →
  reported_average = 65 →
  incorrect_mark1 = 100 →
  incorrect_mark2 = 85 →
  incorrect_mark3 = 15 →
  correct_mark1 = 20 →
  correct_mark2 = 50 →
  correct_mark3 = 55 →
  (num_students * reported_average - (incorrect_mark1 + incorrect_mark2 + incorrect_mark3) +
    (correct_mark1 + correct_mark2 + correct_mark3)) / num_students = 63125 / 1000 := by
  sorry

end correct_average_marks_l2638_263870


namespace contest_end_time_l2638_263896

-- Define a custom time type
structure Time where
  hours : Nat
  minutes : Nat

-- Define a function to add minutes to a time
def addMinutes (t : Time) (m : Nat) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + m
  { hours := totalMinutes / 60 % 24, minutes := totalMinutes % 60 }

-- Define the start time (3:00 p.m.)
def startTime : Time := { hours := 15, minutes := 0 }

-- Define the duration in minutes
def duration : Nat := 720

-- Theorem to prove
theorem contest_end_time :
  addMinutes startTime duration = { hours := 3, minutes := 0 } := by
  sorry

end contest_end_time_l2638_263896


namespace divisibility_condition_l2638_263836

theorem divisibility_condition (a b : ℕ+) :
  (a * b^2 + b + 7) ∣ (a^2 * b + a + b) ↔
  (∃ k : ℕ+, a = 7 * k^2 ∧ b = 7 * k) ∨ (a = 11 ∧ b = 1) ∨ (a = 49 ∧ b = 1) :=
sorry

end divisibility_condition_l2638_263836


namespace fraction_product_result_l2638_263863

def fraction_product (n : ℕ) : ℚ :=
  if n < 6 then 1
  else (n : ℚ) / (n + 5) * fraction_product (n - 1)

theorem fraction_product_result : fraction_product 95 = 1 / 75287520 := by
  sorry

end fraction_product_result_l2638_263863


namespace evaluate_expression_l2638_263818

theorem evaluate_expression : (728 * 728) - (727 * 729) = 1 := by
  sorry

end evaluate_expression_l2638_263818


namespace train_speed_train_speed_approx_66_l2638_263847

/-- The speed of a train given its length, the time it takes to pass a man running in the opposite direction, and the man's speed. -/
theorem train_speed (train_length : ℝ) (passing_time : ℝ) (man_speed_kmh : ℝ) : ℝ :=
  let man_speed_ms := man_speed_kmh * (1000 / 3600)
  let relative_speed := train_length / passing_time
  let train_speed_ms := relative_speed - man_speed_ms
  let train_speed_kmh := train_speed_ms * (3600 / 1000)
  train_speed_kmh

/-- The speed of the train is approximately 66 km/h given the specified conditions. -/
theorem train_speed_approx_66 :
  ∃ ε > 0, abs (train_speed 120 6 6 - 66) < ε :=
sorry

end train_speed_train_speed_approx_66_l2638_263847


namespace quadratic_real_solutions_l2638_263833

theorem quadratic_real_solutions (x y : ℝ) :
  (9 * y^2 + 6 * x * y + 2 * x + 10 = 0) ↔ (x ≤ -10/3 ∨ x ≥ 6) := by
  sorry

end quadratic_real_solutions_l2638_263833


namespace potion_price_l2638_263880

theorem potion_price (discounted_price original_price : ℝ) : 
  discounted_price = 8 → 
  discounted_price = (1 / 5) * original_price → 
  original_price = 40 := by
  sorry

end potion_price_l2638_263880


namespace rockham_soccer_league_members_l2638_263835

theorem rockham_soccer_league_members :
  let sock_cost : ℕ := 6
  let tshirt_cost : ℕ := sock_cost + 10
  let cap_cost : ℕ := 3
  let items_per_member : ℕ := 2  -- for both home and away games
  let cost_per_member : ℕ := items_per_member * (sock_cost + tshirt_cost + cap_cost)
  let total_expenditure : ℕ := 4620
  total_expenditure / cost_per_member = 92 :=
by sorry

end rockham_soccer_league_members_l2638_263835


namespace sarah_snack_purchase_l2638_263848

/-- The number of dimes used by Sarah to buy a $2 snack -/
def num_dimes : ℕ := 10

theorem sarah_snack_purchase :
  ∃ (n : ℕ),
    num_dimes + n = 50 ∧
    10 * num_dimes + 5 * n = 200 :=
by sorry

end sarah_snack_purchase_l2638_263848


namespace infimum_of_function_over_D_l2638_263875

-- Define the set D
def D : Set (ℝ × ℝ) := {p | p.1 > 0 ∧ p.2 > 0 ∧ p.1 ≠ p.2 ∧ p.1 ^ p.2 = p.2 ^ p.1}

-- State the theorem
theorem infimum_of_function_over_D (α β : ℝ) (hα : α > 0) (hβ : β > 0) (hαβ : α ≤ β) :
  ∃ (inf : ℝ), inf = Real.exp (α + β) ∧
    ∀ (x y : ℝ), (x, y) ∈ D → inf ≤ x^α * y^β :=
sorry

end infimum_of_function_over_D_l2638_263875


namespace tuesday_bags_count_l2638_263829

/-- The number of bags of leaves raked on Tuesday -/
def bags_on_tuesday (price_per_bag : ℕ) (bags_monday : ℕ) (bags_other_day : ℕ) (total_money : ℕ) : ℕ :=
  (total_money - price_per_bag * (bags_monday + bags_other_day)) / price_per_bag

/-- Theorem stating that given the conditions, the number of bags raked on Tuesday is 3 -/
theorem tuesday_bags_count :
  bags_on_tuesday 4 5 9 68 = 3 := by
  sorry

end tuesday_bags_count_l2638_263829


namespace smallest_a_value_l2638_263884

theorem smallest_a_value (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) 
  (h : ∀ x : ℤ, Real.cos (a * ↑x + b) = Real.cos (31 * ↑x)) :
  ∀ a' ≥ 0, (∀ x : ℤ, Real.cos (a' * ↑x + b) = Real.cos (31 * ↑x)) → a' ≥ 31 :=
sorry

end smallest_a_value_l2638_263884


namespace stream_speed_l2638_263830

/-- Proves that the speed of the stream is 8 kmph given the conditions of the problem -/
theorem stream_speed (rowing_speed : ℝ) (distance : ℝ) (time : ℝ) :
  rowing_speed = 10 →
  distance = 90 →
  time = 5 →
  ∃ (stream_speed : ℝ), 
    distance = (rowing_speed + stream_speed) * time ∧
    stream_speed = 8 := by
  sorry

end stream_speed_l2638_263830


namespace quadrilateral_area_is_2015029_l2638_263891

/-- The area of a quadrilateral with vertices at (2, 4), (2, 2), (3, 2), and (2010, 2011) -/
def quadrilateralArea : ℝ := 2015029

/-- The vertices of the quadrilateral -/
def vertices : List (ℝ × ℝ) := [(2, 4), (2, 2), (3, 2), (2010, 2011)]

/-- Theorem stating that the area of the quadrilateral with the given vertices is 2015029 square units -/
theorem quadrilateral_area_is_2015029 :
  let computeArea : List (ℝ × ℝ) → ℝ := sorry -- Function to compute area from vertices
  computeArea vertices = quadrilateralArea := by sorry

end quadrilateral_area_is_2015029_l2638_263891


namespace no_positive_integers_satisfying_equation_l2638_263824

theorem no_positive_integers_satisfying_equation : 
  ¬∃ (m n : ℕ), m > 0 ∧ n > 0 ∧ m * (m + 2) = n * (n + 1) := by
  sorry

end no_positive_integers_satisfying_equation_l2638_263824


namespace birdhouse_to_lawn_chair_ratio_l2638_263850

def car_distance : ℝ := 200
def lawn_chair_distance : ℝ := 2 * car_distance
def birdhouse_distance : ℝ := 1200

theorem birdhouse_to_lawn_chair_ratio :
  birdhouse_distance / lawn_chair_distance = 3 := by sorry

end birdhouse_to_lawn_chair_ratio_l2638_263850


namespace inequality_chain_l2638_263809

theorem inequality_chain (x y : ℝ) (h1 : y > x) (h2 : x > 0) (h3 : x + y = 1) :
  x < 2*x*y ∧ 2*x*y < (x + y)/2 ∧ (x + y)/2 < y := by
  sorry

end inequality_chain_l2638_263809


namespace function_extrema_l2638_263878

theorem function_extrema (x : ℝ) (hx : x ∈ Set.Icc (-π/3) (π/4)) :
  let y := (1 / (Real.cos x)^2) + 2 * Real.tan x + 1
  ∃ (min_y max_y : ℝ),
    (∀ z ∈ Set.Icc (-π/3) (π/4), y ≤ max_y ∧ min_y ≤ ((1 / (Real.cos z)^2) + 2 * Real.tan z + 1)) ∧
    y = min_y ↔ x = -π/4 ∧
    y = max_y ↔ x = π/4 ∧
    min_y = 1 ∧
    max_y = 5 :=
by sorry

end function_extrema_l2638_263878


namespace scout_troop_girls_l2638_263871

theorem scout_troop_girls (initial_total : ℕ) : 
  let initial_girls : ℕ := (6 * initial_total) / 10
  let final_total : ℕ := initial_total
  let final_girls : ℕ := initial_girls - 4
  (initial_girls : ℚ) / initial_total = 6 / 10 →
  (final_girls : ℚ) / final_total = 1 / 2 →
  initial_girls = 24 := by
sorry

end scout_troop_girls_l2638_263871


namespace identity_function_proof_l2638_263857

theorem identity_function_proof (f : ℝ → ℝ) 
  (h_increasing : ∀ x y, x < y → f x < f y) 
  (h_inverse : ∀ x, f (f x) = x) : 
  ∀ x, f x = x := by
sorry

end identity_function_proof_l2638_263857


namespace factorization_proof_l2638_263814

theorem factorization_proof (z : ℝ) : 
  88 * z^19 + 176 * z^38 + 264 * z^57 = 88 * z^19 * (1 + 2 * z^19 + 3 * z^38) := by
sorry

end factorization_proof_l2638_263814


namespace seven_balls_three_boxes_l2638_263804

/-- The number of ways to put n distinguishable balls into k distinguishable boxes -/
def ways_to_put_balls_in_boxes (n : ℕ) (k : ℕ) : ℕ := k^n

/-- Theorem: There are 3^7 ways to put 7 distinguishable balls into 3 distinguishable boxes -/
theorem seven_balls_three_boxes : ways_to_put_balls_in_boxes 7 3 = 3^7 := by
  sorry

end seven_balls_three_boxes_l2638_263804


namespace walk_distance_proof_l2638_263859

/-- Calculates the total distance walked given a constant speed and two walking durations -/
def total_distance (speed : ℝ) (duration1 : ℝ) (duration2 : ℝ) : ℝ :=
  speed * (duration1 + duration2)

/-- Proves that walking at 4 miles per hour for 2 hours and then 0.5 hours results in 10 miles -/
theorem walk_distance_proof :
  let speed := 4
  let duration1 := 2
  let duration2 := 0.5
  total_distance speed duration1 duration2 = 10 := by
  sorry

end walk_distance_proof_l2638_263859


namespace total_shoes_l2638_263845

/-- The number of shoes owned by each person -/
structure ShoeCount where
  daniel : ℕ
  christopher : ℕ
  brian : ℕ
  edward : ℕ
  jacob : ℕ

/-- The conditions of the shoe ownership problem -/
def shoe_conditions (s : ShoeCount) : Prop :=
  s.daniel = 15 ∧
  s.christopher = 37 ∧
  s.brian = s.christopher + 5 ∧
  s.edward = (7 * s.brian) / 2 ∧
  s.jacob = (2 * s.edward) / 3

/-- The theorem stating the total number of shoes -/
theorem total_shoes (s : ShoeCount) (h : shoe_conditions s) :
  s.daniel + s.christopher + s.brian + s.edward + s.jacob = 339 := by
  sorry

end total_shoes_l2638_263845


namespace symmetric_points_sum_l2638_263851

/-- Given two points P and Q symmetric with respect to the y-axis, prove that the sum of their x-coordinates is -8 --/
theorem symmetric_points_sum (a b : ℝ) : 
  (∃ (P Q : ℝ × ℝ), P = (a, -3) ∧ Q = (4, b) ∧ 
   (P.1 = -Q.1) ∧ (P.2 = Q.2)) → 
  a + b = -7 :=
by sorry

end symmetric_points_sum_l2638_263851


namespace shopping_ratio_l2638_263866

theorem shopping_ratio (emma_spent elsa_spent elizabeth_spent total_spent : ℚ) : 
  emma_spent = 58 →
  elizabeth_spent = 4 * elsa_spent →
  total_spent = 638 →
  emma_spent + elsa_spent + elizabeth_spent = total_spent →
  elsa_spent / emma_spent = 2 / 1 := by
sorry

end shopping_ratio_l2638_263866


namespace exponent_relationship_l2638_263883

theorem exponent_relationship (x y z a b : ℝ) 
  (h1 : 4^x = a) 
  (h2 : 2^y = b) 
  (h3 : 8^z = a * b) : 
  3 * z = 2 * x + y := by
sorry

end exponent_relationship_l2638_263883


namespace cake_brownie_calorie_difference_l2638_263838

/-- Represents the number of slices in the cake -/
def cake_slices : ℕ := 8

/-- Represents the number of calories in each cake slice -/
def calories_per_cake_slice : ℕ := 347

/-- Represents the number of brownies in a pan -/
def brownies_count : ℕ := 6

/-- Represents the number of calories in each brownie -/
def calories_per_brownie : ℕ := 375

/-- Theorem stating the difference in total calories between the cake and the brownies -/
theorem cake_brownie_calorie_difference :
  cake_slices * calories_per_cake_slice - brownies_count * calories_per_brownie = 526 := by
  sorry


end cake_brownie_calorie_difference_l2638_263838


namespace inequality_necessary_not_sufficient_l2638_263849

-- Define what it means for the equation to represent an ellipse
def represents_ellipse (m : ℝ) : Prop :=
  5 - m > 0 ∧ m + 3 > 0 ∧ 5 - m ≠ m + 3

-- Define the inequality condition
def inequality_condition (m : ℝ) : Prop :=
  -3 < m ∧ m < 5

-- Theorem statement
theorem inequality_necessary_not_sufficient :
  (∀ m : ℝ, represents_ellipse m → inequality_condition m) ∧
  (∃ m : ℝ, inequality_condition m ∧ ¬represents_ellipse m) :=
sorry

end inequality_necessary_not_sufficient_l2638_263849


namespace least_subtrahend_for_divisibility_l2638_263820

theorem least_subtrahend_for_divisibility (n : ℕ) (a b c : ℕ) (h_n : n = 157632) (h_a : a = 12) (h_b : b = 18) (h_c : c = 24) :
  ∃ (k : ℕ), k = 24 ∧
  (∀ (m : ℕ), m < k → ¬(∃ (q : ℕ), n - m = q * a ∧ n - m = q * b ∧ n - m = q * c)) ∧
  (∃ (q : ℕ), n - k = q * a ∧ n - k = q * b ∧ n - k = q * c) :=
by sorry

end least_subtrahend_for_divisibility_l2638_263820


namespace eddys_climbing_rate_l2638_263882

/-- Proves that Eddy's climbing rate is 500 ft/hr given the conditions of the problem --/
theorem eddys_climbing_rate (hillary_climb_rate : ℝ) (hillary_descent_rate : ℝ) 
  (start_time : ℝ) (pass_time : ℝ) (base_camp_distance : ℝ) (hillary_stop_distance : ℝ) :
  hillary_climb_rate = 800 →
  hillary_descent_rate = 1000 →
  start_time = 6 →
  pass_time = 12 →
  base_camp_distance = 5000 →
  hillary_stop_distance = 1000 →
  ∃ (eddy_climb_rate : ℝ), eddy_climb_rate = 500 :=
by
  sorry

end eddys_climbing_rate_l2638_263882


namespace dot_product_specific_vectors_l2638_263800

theorem dot_product_specific_vectors (α : ℝ) : 
  let a : ℝ × ℝ := (Real.cos α, Real.sin α)
  let b : ℝ × ℝ := (Real.cos (π/3 + α), Real.sin (π/3 + α))
  (a.1 * b.1 + a.2 * b.2) = 1/2 := by
sorry

end dot_product_specific_vectors_l2638_263800


namespace calculate_expression_l2638_263807

theorem calculate_expression : 500 * 996 * 0.0996 * 20 + 5000 = 997016 := by
  sorry

end calculate_expression_l2638_263807


namespace intersection_slope_l2638_263821

/-- Given two lines that intersect at a point, find the slope of one line -/
theorem intersection_slope (m : ℝ) : 
  (∃ (x y : ℝ), y = 2*x + 3 ∧ y = m*x + 1 ∧ x = 1 ∧ y = 5) → m = 4 := by
  sorry

end intersection_slope_l2638_263821


namespace square_area_equals_perimeter_l2638_263815

theorem square_area_equals_perimeter (s : ℝ) (h : s > 0) : s^2 = 4*s → s = 4 := by
  sorry

end square_area_equals_perimeter_l2638_263815


namespace ellipse_foci_product_l2638_263823

-- Define the ellipse
def is_on_ellipse (P : ℝ × ℝ) : Prop :=
  P.1^2 / 16 + P.2^2 / 12 = 1

-- Define the foci
def F1 : ℝ × ℝ := (-2, 0)
def F2 : ℝ × ℝ := (2, 0)

-- Define the dot product condition
def satisfies_dot_product (P : ℝ × ℝ) : Prop :=
  let PF1 := (P.1 - F1.1, P.2 - F1.2)
  let PF2 := (P.1 - F2.1, P.2 - F2.2)
  PF1.1 * PF2.1 + PF1.2 * PF2.2 = 9

-- Theorem statement
theorem ellipse_foci_product (P : ℝ × ℝ) :
  is_on_ellipse P → satisfies_dot_product P →
  let PF1 := (P.1 - F1.1, P.2 - F1.2)
  let PF2 := (P.1 - F2.1, P.2 - F2.2)
  (Real.sqrt (PF1.1^2 + PF1.2^2)) * (Real.sqrt (PF2.1^2 + PF2.2^2)) = 15 :=
by
  sorry

end ellipse_foci_product_l2638_263823


namespace domain_of_f_2x_minus_1_l2638_263841

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f(x+1)
def domain_f_x_plus_1 : Set ℝ := Set.Icc (-2) 3

-- State the theorem
theorem domain_of_f_2x_minus_1 :
  (∀ x ∈ domain_f_x_plus_1, f (x + 1) = f (x + 1)) →
  {x : ℝ | f (2 * x - 1) = f (2 * x - 1)} = Set.Icc 0 (5/2) := by sorry

end domain_of_f_2x_minus_1_l2638_263841


namespace unoccupied_volume_correct_l2638_263831

/-- Represents the dimensions of a cube in inches -/
structure CubeDimensions where
  side : ℝ

/-- Calculates the volume of a cube given its dimensions -/
def cubeVolume (d : CubeDimensions) : ℝ := d.side ^ 3

/-- Represents the container and its contents -/
structure Container where
  dimensions : CubeDimensions
  waterFillRatio : ℝ
  iceCubes : ℕ
  iceCubeDimensions : CubeDimensions

/-- Calculates the unoccupied volume in the container -/
def unoccupiedVolume (c : Container) : ℝ :=
  let containerVolume := cubeVolume c.dimensions
  let waterVolume := c.waterFillRatio * containerVolume
  let iceCubeVolume := cubeVolume c.iceCubeDimensions
  let totalIceVolume := c.iceCubes * iceCubeVolume
  containerVolume - waterVolume - totalIceVolume

/-- The main theorem to prove -/
theorem unoccupied_volume_correct (c : Container) : 
  c.dimensions.side = 12 ∧ 
  c.waterFillRatio = 3/4 ∧ 
  c.iceCubes = 6 ∧ 
  c.iceCubeDimensions.side = 1.5 → 
  unoccupiedVolume c = 411.75 := by
  sorry

end unoccupied_volume_correct_l2638_263831


namespace r_value_when_n_is_3_l2638_263802

theorem r_value_when_n_is_3 :
  let n : ℕ := 3
  let s : ℕ := 2^n - 1
  let r : ℕ := 4^s - s
  r = 16377 := by
sorry

end r_value_when_n_is_3_l2638_263802


namespace negation_of_proposition_l2638_263862

theorem negation_of_proposition (a b x : ℝ) :
  (¬(x ≥ a^2 + b^2 → x ≥ 2*a*b)) ↔ (x < a^2 + b^2 → x < 2*a*b) :=
by sorry

end negation_of_proposition_l2638_263862


namespace geometric_sequence_product_l2638_263887

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_product (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n, a n > 0) →
  a 4 * a 8 = 4 →
  a 5 * a 6 * a 7 = 8 := by
  sorry

end geometric_sequence_product_l2638_263887


namespace equation_solution_l2638_263898

theorem equation_solution : ∃ (x : ℚ), (3/4 : ℚ) + 1/x = 7/8 ∧ x = 8 := by
  sorry

end equation_solution_l2638_263898


namespace multiple_of_72_digits_l2638_263895

theorem multiple_of_72_digits (n : ℕ) (x y : Fin 10) :
  (n = 320000000 + x * 10000000 + 35717 * 10 + y) →
  (n % 72 = 0) →
  (x * y = 12) :=
by sorry

end multiple_of_72_digits_l2638_263895


namespace hiking_speeds_l2638_263899

-- Define the hiking speeds and relationships
def lucas_speed : ℚ := 5
def mia_speed_ratio : ℚ := 3/4
def grace_speed_ratio : ℚ := 6/7
def liam_speed_ratio : ℚ := 4/3

-- Define the hiking speeds of Mia, Grace, and Liam
def mia_speed : ℚ := lucas_speed * mia_speed_ratio
def grace_speed : ℚ := mia_speed * grace_speed_ratio
def liam_speed : ℚ := grace_speed * liam_speed_ratio

-- Theorem to prove Grace's and Liam's hiking speeds
theorem hiking_speeds :
  grace_speed = 45/14 ∧ liam_speed = 30/7 := by
  sorry

end hiking_speeds_l2638_263899


namespace two_less_than_negative_one_l2638_263872

theorem two_less_than_negative_one : -1 - 2 = -3 := by
  sorry

end two_less_than_negative_one_l2638_263872


namespace monotonic_quadratic_l2638_263877

def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x - 3

def monotonic_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y ∨ (∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x > f y)

theorem monotonic_quadratic (a : ℝ) :
  monotonic_on (f a) 1 2 ↔ a ≤ 1 ∨ a ≥ 2 :=
sorry

end monotonic_quadratic_l2638_263877


namespace carla_cooking_time_l2638_263817

/-- Represents the cooking time for each item in minutes -/
structure CookingTime where
  waffle : ℕ
  steak : ℕ
  chili : ℕ

/-- Represents the number of items to be cooked -/
structure CookingItems where
  waffle : ℕ
  steak : ℕ
  chili : ℕ

/-- Calculates the total cooking time given the cooking times and items to be cooked -/
def totalCookingTime (time : CookingTime) (items : CookingItems) : ℕ :=
  time.waffle * items.waffle + time.steak * items.steak + time.chili * items.chili

/-- Theorem stating that Carla's total cooking time is 100 minutes -/
theorem carla_cooking_time :
  let time := CookingTime.mk 10 6 20
  let items := CookingItems.mk 3 5 2
  totalCookingTime time items = 100 := by sorry

end carla_cooking_time_l2638_263817


namespace isosceles_right_triangle_property_l2638_263819

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x + a

-- State the theorem
theorem isosceles_right_triangle_property 
  (a : ℝ) (x₁ x₂ : ℝ) (t : ℝ) : 
  x₁ < x₂ →
  f a x₁ = 0 →
  f a x₂ = 0 →
  (∃ x₀, x₁ < x₀ ∧ x₀ < x₂ ∧ 
    (x₂ - x₁) / 2 = -f a x₀ ∧
    (x₂ - x₀) = (x₀ - x₁)) →
  Real.sqrt ((x₂ - 1) / (x₁ - 1)) = t →
  a * t - (a + t) = 1 := by
sorry

end isosceles_right_triangle_property_l2638_263819


namespace twelve_point_zero_six_million_scientific_notation_l2638_263844

-- Define 12.06 million
def twelve_point_zero_six_million : ℝ := 12.06 * 1000000

-- Define the scientific notation representation
def scientific_notation : ℝ := 1.206 * 10^7

-- Theorem statement
theorem twelve_point_zero_six_million_scientific_notation :
  twelve_point_zero_six_million = scientific_notation :=
by sorry

end twelve_point_zero_six_million_scientific_notation_l2638_263844


namespace binomial_coeff_x8_eq_10_l2638_263892

-- Define the binomial coefficient function
def binomial_coeff (n k : ℕ) : ℕ := sorry

-- Define the function to get the exponent of x in the general term
def x_exponent (r : ℕ) : ℚ := 15 - (7 * r) / 2

-- Define the function to find the binomial coefficient for x^8
def binomial_coeff_x8 (n : ℕ) : ℕ :=
  let r := 2 -- r is 2 when x_exponent(r) = 8
  binomial_coeff n r

-- Theorem statement
theorem binomial_coeff_x8_eq_10 :
  binomial_coeff_x8 5 = 10 := by sorry

end binomial_coeff_x8_eq_10_l2638_263892


namespace equation_solution_l2638_263810

theorem equation_solution (x : ℝ) :
  Real.sqrt (9 + Real.sqrt (27 + 3*x)) + Real.sqrt (3 + Real.sqrt (1 + x)) = 3 + 3 * Real.sqrt 3 →
  x = 10 + 8 * Real.sqrt 3 :=
by sorry

end equation_solution_l2638_263810


namespace dice_game_probability_l2638_263846

theorem dice_game_probability (n : ℕ) (max_score : ℕ) (num_dice : ℕ) (num_faces : ℕ) :
  let p_max_score := (1 / num_faces : ℚ) ^ num_dice
  let p_not_max_score := 1 - p_max_score
  n = 23 ∧ max_score = 18 ∧ num_dice = 3 ∧ num_faces = 6 →
  p_max_score * p_not_max_score ^ (n - 1) = (1 / 216 : ℚ) * (1 - 1 / 216 : ℚ) ^ 22 :=
by sorry

end dice_game_probability_l2638_263846


namespace conference_handshakes_l2638_263860

/-- Conference attendees are divided into three groups -/
structure ConferenceGroups where
  total : ℕ
  group1 : ℕ
  group2 : ℕ
  group3 : ℕ

/-- Calculate the number of handshakes in the conference -/
def calculate_handshakes (groups : ConferenceGroups) : ℕ :=
  let group3_handshakes := groups.group3 * (groups.total - groups.group3)
  let group2_handshakes := groups.group2 * (groups.group1 + groups.group3)
  (group3_handshakes + group2_handshakes) / 2

/-- Theorem stating that the number of handshakes is 237 -/
theorem conference_handshakes :
  ∃ (groups : ConferenceGroups),
    groups.total = 40 ∧
    groups.group1 = 25 ∧
    groups.group2 = 10 ∧
    groups.group3 = 5 ∧
    calculate_handshakes groups = 237 := by
  sorry


end conference_handshakes_l2638_263860


namespace f_root_condition_and_inequality_l2638_263839

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a / x

theorem f_root_condition_and_inequality (a : ℝ) (b : ℝ) :
  (a > 0 ∧ (∃ x > 0, f a x = 0) ↔ 0 < a ∧ a ≤ 1 / Real.exp 1) ∧
  (a ≥ 2 / Real.exp 1 ∧ b > 1 → f a (Real.log b) > 1 / b) := by
  sorry

end f_root_condition_and_inequality_l2638_263839


namespace negation_of_universal_statement_l2638_263822

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^2 + x + 1 ≥ 0) ↔ (∃ x : ℝ, x^2 + x + 1 < 0) := by
  sorry

end negation_of_universal_statement_l2638_263822


namespace three_numbers_average_l2638_263812

theorem three_numbers_average (a b c : ℝ) 
  (h1 : a + (b + c)/2 = 65)
  (h2 : b + (a + c)/2 = 69)
  (h3 : c + (a + b)/2 = 76) :
  (a + b + c)/3 = 35 := by
  sorry

end three_numbers_average_l2638_263812


namespace starfish_arms_l2638_263890

theorem starfish_arms (num_starfish : ℕ) (seastar_arms : ℕ) (total_arms : ℕ) :
  num_starfish = 7 →
  seastar_arms = 14 →
  total_arms = 49 →
  ∃ (starfish_arms : ℕ), num_starfish * starfish_arms + seastar_arms = total_arms ∧ starfish_arms = 5 :=
by sorry

end starfish_arms_l2638_263890


namespace floor_sqrt_inequality_l2638_263879

theorem floor_sqrt_inequality (x : ℝ) : 
  150 ≤ x ∧ x ≤ 300 ∧ ⌊Real.sqrt x⌋ = 16 → ⌊Real.sqrt (10 * x)⌋ ≠ 160 := by
  sorry

end floor_sqrt_inequality_l2638_263879


namespace grandmother_age_problem_l2638_263828

theorem grandmother_age_problem (yuna_initial_age grandmother_initial_age : ℕ) 
  (h1 : yuna_initial_age = 12)
  (h2 : grandmother_initial_age = 72) :
  ∃ (years_passed : ℕ), 
    grandmother_initial_age + years_passed = 5 * (yuna_initial_age + years_passed) ∧
    grandmother_initial_age + years_passed = 75 := by
  sorry

end grandmother_age_problem_l2638_263828


namespace pet_store_cages_used_l2638_263834

def pet_store_problem (initial_puppies sold_puppies puppies_per_cage : ℕ) : ℕ :=
  (initial_puppies - sold_puppies) / puppies_per_cage

theorem pet_store_cages_used :
  pet_store_problem 78 30 8 = 6 := by
  sorry

end pet_store_cages_used_l2638_263834


namespace regular_polygon_with_12_degree_exterior_angles_has_30_sides_l2638_263868

/-- A regular polygon with exterior angles measuring 12 degrees has 30 sides. -/
theorem regular_polygon_with_12_degree_exterior_angles_has_30_sides :
  ∀ n : ℕ, 
  n > 0 →
  (360 : ℝ) / n = 12 →
  n = 30 := by
  sorry

end regular_polygon_with_12_degree_exterior_angles_has_30_sides_l2638_263868


namespace theresa_final_count_l2638_263873

/-- Represents the number of crayons each person has -/
structure CrayonCount where
  theresa : ℕ
  janice : ℕ
  nancy : ℕ
  mark : ℕ

/-- Represents the initial state and actions taken -/
def initial_state : CrayonCount := {
  theresa := 32,
  janice := 12,
  nancy := 0,
  mark := 0
}

/-- Janice shares half of her crayons with Nancy and gives 3 to Mark -/
def share_crayons (state : CrayonCount) : CrayonCount := {
  theresa := state.theresa,
  janice := state.janice - (state.janice / 2) - 3,
  nancy := state.nancy + (state.janice / 2),
  mark := state.mark + 3
}

/-- Nancy gives 8 crayons to Theresa -/
def give_to_theresa (state : CrayonCount) : CrayonCount := {
  theresa := state.theresa + 8,
  janice := state.janice,
  nancy := state.nancy - 8,
  mark := state.mark
}

/-- The final state after all actions -/
def final_state : CrayonCount := give_to_theresa (share_crayons initial_state)

theorem theresa_final_count : final_state.theresa = 40 := by
  sorry

end theresa_final_count_l2638_263873


namespace orange_apple_weight_equivalence_l2638_263894

/-- Given that 8 oranges weigh as much as 6 apples, prove that 32 oranges weigh as much as 24 apples. -/
theorem orange_apple_weight_equivalence :
  ∀ (orange_weight apple_weight : ℝ),
  orange_weight > 0 →
  apple_weight > 0 →
  8 * orange_weight = 6 * apple_weight →
  32 * orange_weight = 24 * apple_weight :=
by
  sorry

end orange_apple_weight_equivalence_l2638_263894


namespace geometric_sequence_a12_l2638_263858

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_a12 (a : ℕ → ℝ) :
  geometric_sequence a →
  a 6 * a 10 = 16 →
  a 4 = 1 →
  a 12 = 16 := by
sorry

end geometric_sequence_a12_l2638_263858


namespace large_envelopes_count_l2638_263832

theorem large_envelopes_count (total_letters : ℕ) (small_envelope_letters : ℕ) (letters_per_large_envelope : ℕ) : 
  total_letters = 80 →
  small_envelope_letters = 20 →
  letters_per_large_envelope = 2 →
  (total_letters - small_envelope_letters) / letters_per_large_envelope = 30 := by
sorry

end large_envelopes_count_l2638_263832


namespace glucose_solution_volume_l2638_263808

/-- The volume of glucose solution containing 15 grams of glucose -/
def volume_15g : ℝ := 100

/-- The volume of glucose solution used in the given condition -/
def volume_given : ℝ := 65

/-- The mass of glucose in the given volume -/
def mass_given : ℝ := 9.75

/-- The target mass of glucose -/
def mass_target : ℝ := 15

theorem glucose_solution_volume :
  (mass_given / volume_given) * volume_15g = mass_target :=
by sorry

end glucose_solution_volume_l2638_263808


namespace new_conveyor_belt_time_l2638_263803

theorem new_conveyor_belt_time (old_time new_time combined_time : ℝ) 
  (h1 : old_time = 21)
  (h2 : combined_time = 8.75)
  (h3 : 1 / old_time + 1 / new_time = 1 / combined_time) : 
  new_time = 15 := by
sorry

end new_conveyor_belt_time_l2638_263803


namespace smallest_gcd_bc_l2638_263826

theorem smallest_gcd_bc (a b c : ℕ+) (h1 : Nat.gcd a b = 72) (h2 : Nat.gcd a c = 240) :
  (∃ (x y z : ℕ+), x = a ∧ y = b ∧ z = c ∧ Nat.gcd y z = 24) ∧
  (∀ (p q : ℕ+), Nat.gcd p q < 24 → ¬(∃ (r : ℕ+), Nat.gcd r p = 72 ∧ Nat.gcd r q = 240)) :=
by sorry

end smallest_gcd_bc_l2638_263826


namespace no_real_roots_quadratic_l2638_263867

theorem no_real_roots_quadratic (m : ℝ) :
  (∀ x : ℝ, x^2 + m*x + 1 ≠ 0) ↔ -2 < m ∧ m < 2 := by
  sorry

end no_real_roots_quadratic_l2638_263867


namespace x_range_for_decreasing_sequence_l2638_263842

def decreasing_sequence (x : ℝ) : ℕ → ℝ
  | 0 => 1 - x
  | n + 1 => (1 - x) ^ (n + 2)

theorem x_range_for_decreasing_sequence (x : ℝ) :
  (∀ n : ℕ, decreasing_sequence x n > decreasing_sequence x (n + 1)) ↔ 0 < x ∧ x < 1 := by
  sorry

end x_range_for_decreasing_sequence_l2638_263842


namespace mirella_purple_books_l2638_263886

/-- The number of pages in each purple book -/
def purple_book_pages : ℕ := 230

/-- The number of pages in each orange book -/
def orange_book_pages : ℕ := 510

/-- The number of orange books Mirella read -/
def orange_books_read : ℕ := 4

/-- The difference between orange and purple pages read -/
def page_difference : ℕ := 890

/-- The number of purple books Mirella read -/
def purple_books_read : ℕ := 5

theorem mirella_purple_books :
  orange_books_read * orange_book_pages - purple_books_read * purple_book_pages = page_difference :=
by sorry

end mirella_purple_books_l2638_263886


namespace sum_of_squared_coefficients_l2638_263806

/-- Given the expression 5(x^3 - 3x^2 + 4) - 8(2x^3 - x^2 - 2), 
    the sum of the squares of its coefficients when fully simplified is 1466. -/
theorem sum_of_squared_coefficients : 
  let expr := fun (x : ℝ) => 5 * (x^3 - 3*x^2 + 4) - 8 * (2*x^3 - x^2 - 2)
  let simplified := fun (x : ℝ) => -11*x^3 - 7*x^2 + 36
  (∀ x, expr x = simplified x) → 
  (-11)^2 + (-7)^2 + 36^2 = 1466 := by
  sorry

end sum_of_squared_coefficients_l2638_263806


namespace parabola_b_value_l2638_263805

/-- A parabola passing through three given points has a specific b value -/
theorem parabola_b_value (b c : ℚ) :
  ((-1)^2 + b*(-1) + c = -11) →
  (3^2 + b*3 + c = 17) →
  (2^2 + b*2 + c = 6) →
  b = 14/3 := by
  sorry

end parabola_b_value_l2638_263805


namespace locus_is_finite_l2638_263843

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance squared between two points -/
def distanceSquared (p1 p2 : Point) : ℝ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

/-- Definition of the right triangle -/
def rightTriangle (c : ℝ) : Set Point :=
  {p : Point | 0 ≤ p.x ∧ p.x ≤ c ∧ 0 ≤ p.y ∧ p.y ≤ c ∧ p.x + p.y ≤ c}

/-- The set of points satisfying the given conditions -/
def locusSet (c : ℝ) : Set Point :=
  {p ∈ rightTriangle c |
    distanceSquared p ⟨0, 0⟩ + distanceSquared p ⟨c, 0⟩ = 2 * c^2 ∧
    distanceSquared p ⟨0, c⟩ = c^2}

theorem locus_is_finite (c : ℝ) (h : c > 0) : Set.Finite (locusSet c) :=
  sorry

end locus_is_finite_l2638_263843


namespace circle1_properties_circle2_properties_l2638_263855

-- Define the circles and lines
def circle1 (x y : ℝ) : Prop := (x - 4)^2 + (y - 5)^2 = 10
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 2*y - 11 = 0
def line1 (x y : ℝ) : Prop := y = 2*x - 3
def line2 (x y : ℝ) : Prop := 3*x + 4*y - 1 = 0
def circle3 (x y : ℝ) : Prop := x^2 + y^2 - x + y - 2 = 0
def circle4 (x y : ℝ) : Prop := x^2 + y^2 = 5

-- Theorem for the first circle
theorem circle1_properties :
  (∀ x y : ℝ, circle1 x y → ((x = 5 ∧ y = 2) ∨ (x = 3 ∧ y = 2))) ∧
  (∃ x y : ℝ, circle1 x y ∧ line1 x y) :=
sorry

-- Theorem for the second circle
theorem circle2_properties :
  (∀ x y : ℝ, (circle3 x y ∧ circle4 x y) → circle2 x y) ∧
  (∃ x y : ℝ, circle2 x y ∧ line2 x y) :=
sorry

end circle1_properties_circle2_properties_l2638_263855


namespace sandy_marks_calculation_l2638_263861

theorem sandy_marks_calculation :
  ∀ (total_attempts : ℕ) (correct_attempts : ℕ) (marks_per_correct : ℕ) (marks_per_incorrect : ℕ),
    total_attempts = 30 →
    correct_attempts = 24 →
    marks_per_correct = 3 →
    marks_per_incorrect = 2 →
    (correct_attempts * marks_per_correct) - ((total_attempts - correct_attempts) * marks_per_incorrect) = 60 :=
by sorry

end sandy_marks_calculation_l2638_263861


namespace floor_sqrt_120_l2638_263864

theorem floor_sqrt_120 : ⌊Real.sqrt 120⌋ = 10 := by
  sorry

end floor_sqrt_120_l2638_263864


namespace yellow_beads_proof_l2638_263854

theorem yellow_beads_proof (green_beads : ℕ) (yellow_fraction : ℚ) : 
  green_beads = 4 → 
  yellow_fraction = 4/5 → 
  (yellow_fraction * (green_beads + 16 : ℚ)).num = 16 := by
  sorry

end yellow_beads_proof_l2638_263854


namespace part_one_part_two_l2638_263874

-- Define the sets A and B
def A : Set ℝ := {x | -3 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | x < 2*m - 3}

-- Statement for part 1
theorem part_one (m : ℝ) (h : m = 5) : 
  A ∩ B m = A ∧ (Aᶜ ∪ B m) = Set.univ := by sorry

-- Statement for part 2
theorem part_two (m : ℝ) : 
  A ⊆ B m ↔ m > 4 := by sorry

end part_one_part_two_l2638_263874


namespace no_positive_sheep_solution_l2638_263888

theorem no_positive_sheep_solution : ¬∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ y = 3 * x + 15 ∧ x = y - y / 3 := by
  sorry

end no_positive_sheep_solution_l2638_263888


namespace parallel_vectors_x_value_l2638_263811

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2

theorem parallel_vectors_x_value (x : ℝ) :
  let a : ℝ × ℝ := (2, x)
  let b : ℝ × ℝ := (1, 2)
  parallel a b → x = 4 := by
sorry

end parallel_vectors_x_value_l2638_263811


namespace fractional_equation_solution_l2638_263801

theorem fractional_equation_solution :
  ∃ (x : ℝ), (x + 2 ≠ 0 ∧ x - 2 ≠ 0) →
  (1 / (x + 2) + 4 * x / (x^2 - 4) = 1 / (x - 2)) ∧ x = 1 := by
sorry

end fractional_equation_solution_l2638_263801


namespace min_illuminated_points_l2638_263881

/-- Represents the number of illuminated points for a laser at angle θ --/
def illuminatedPoints (θ : ℕ) : ℕ := 180 / Nat.gcd 180 θ

/-- The problem statement --/
theorem min_illuminated_points :
  ∃ (n : ℕ), n < 90 ∧ 
  (∀ (m : ℕ), m < 90 → 
    illuminatedPoints n + illuminatedPoints (n + 1) - 1 ≤ 
    illuminatedPoints m + illuminatedPoints (m + 1) - 1) ∧
  illuminatedPoints n + illuminatedPoints (n + 1) - 1 = 28 :=
sorry

end min_illuminated_points_l2638_263881


namespace jorges_total_goals_l2638_263825

/-- The total number of goals Jorge scored over two seasons -/
def total_goals (last_season_goals this_season_goals : ℕ) : ℕ :=
  last_season_goals + this_season_goals

/-- Theorem stating that Jorge's total goals over two seasons is 343 -/
theorem jorges_total_goals :
  total_goals 156 187 = 343 := by
  sorry

end jorges_total_goals_l2638_263825


namespace min_value_zero_iff_c_eq_four_l2638_263897

/-- The quadratic expression in x and y with parameter c -/
def f (c x y : ℝ) : ℝ :=
  5 * x^2 - 8 * c * x * y + (4 * c^2 + 3) * y^2 - 5 * x - 5 * y + 7

/-- The theorem stating that c = 4 is the unique value for which the minimum of f is 0 -/
theorem min_value_zero_iff_c_eq_four :
  (∃ (c : ℝ), ∀ (x y : ℝ), f c x y ≥ 0 ∧ (∃ (x₀ y₀ : ℝ), f c x₀ y₀ = 0)) ↔ c = 4 :=
sorry

end min_value_zero_iff_c_eq_four_l2638_263897


namespace square_perimeter_problem_l2638_263869

/-- Given squares A and B with perimeters 16 and 32 respectively, 
    when placed side by side to form square C, the perimeter of C is 48. -/
theorem square_perimeter_problem (A B C : ℝ → ℝ → Prop) :
  (∀ x, A x x → 4 * x = 16) →  -- Square A has perimeter 16
  (∀ y, B y y → 4 * y = 32) →  -- Square B has perimeter 32
  (∀ z, C z z → ∃ x y, A x x ∧ B y y ∧ z = x + y) →  -- C is formed by A and B side by side
  (∀ z, C z z → 4 * z = 48) :=  -- The perimeter of C is 48
by sorry

end square_perimeter_problem_l2638_263869


namespace log_product_equals_four_l2638_263876

theorem log_product_equals_four : (Real.log 9 / Real.log 2) * (Real.log 4 / Real.log 3) = 4 := by
  sorry

end log_product_equals_four_l2638_263876


namespace average_sum_difference_l2638_263893

theorem average_sum_difference (x₁ x₂ x₃ y₁ y₂ y₃ z₁ z₂ z₃ a b c : ℝ) 
  (hx : (x₁ + x₂ + x₃) / 3 = a)
  (hy : (y₁ + y₂ + y₃) / 3 = b)
  (hz : (z₁ + z₂ + z₃) / 3 = c) :
  ((x₁ + y₁ - z₁) + (x₂ + y₂ - z₂) + (x₃ + y₃ - z₃)) / 3 = a + b - c := by
  sorry

end average_sum_difference_l2638_263893


namespace thermal_equilibrium_problem_l2638_263816

/-- Represents the thermal equilibrium in a system of water and metal bars -/
structure ThermalSystem where
  initialWaterTemp : ℝ
  initialBarTemp : ℝ
  firstEquilibriumTemp : ℝ
  finalEquilibriumTemp : ℝ

/-- The thermal equilibrium problem -/
theorem thermal_equilibrium_problem (system : ThermalSystem)
  (h1 : system.initialWaterTemp = 100)
  (h2 : system.initialBarTemp = 20)
  (h3 : system.firstEquilibriumTemp = 80)
  : system.finalEquilibriumTemp = 68 := by
  sorry

end thermal_equilibrium_problem_l2638_263816


namespace ac_over_b_squared_range_l2638_263853

/-- Given an obtuse triangle ABC with sides a, b, c satisfying a < b < c
    and internal angles forming an arithmetic sequence,
    the value of ac/b^2 is strictly between 0 and 2/3. -/
theorem ac_over_b_squared_range (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  a < b → b < c →
  0 < A → A < π → 0 < B → B < π → 0 < C → C < π →
  A + B + C = π →
  C > π / 2 →
  ∃ (k : ℝ), B - A = C - B ∧ B = k * A ∧ C = (k + 1) * A →
  0 < a * c / (b * b) ∧ a * c / (b * b) < 2 / 3 := by
  sorry

end ac_over_b_squared_range_l2638_263853


namespace mod_seven_equality_l2638_263885

theorem mod_seven_equality : (45^1234 - 25^1234) % 7 = 5 := by
  sorry

end mod_seven_equality_l2638_263885


namespace february_roses_l2638_263889

def rose_sequence (october november december january : ℕ) : Prop :=
  november - october = december - november ∧
  december - november = january - december ∧
  november > october ∧ december > november ∧ january > december

theorem february_roses 
  (october november december january : ℕ) 
  (h : rose_sequence october november december january) 
  (oct_val : october = 108) 
  (nov_val : november = 120) 
  (dec_val : december = 132) 
  (jan_val : january = 144) : 
  january + (january - december) = 156 := by
sorry

end february_roses_l2638_263889


namespace randy_biscuits_l2638_263827

/-- The number of biscuits Randy is left with after receiving and losing some -/
def biscuits_left (initial : ℕ) (father_gift : ℕ) (mother_gift : ℕ) (brother_ate : ℕ) : ℕ :=
  initial + father_gift + mother_gift - brother_ate

/-- Theorem stating that Randy is left with 40 biscuits -/
theorem randy_biscuits :
  biscuits_left 32 13 15 20 = 40 := by
  sorry

end randy_biscuits_l2638_263827


namespace fourth_pentagon_dots_l2638_263856

/-- Calculates the number of dots in a pentagon given its layer number -/
def dots_in_pentagon (n : ℕ) : ℕ :=
  if n = 0 then 1
  else dots_in_pentagon (n - 1) + 5 * n

theorem fourth_pentagon_dots :
  dots_in_pentagon 3 = 31 := by
  sorry

#eval dots_in_pentagon 3

end fourth_pentagon_dots_l2638_263856

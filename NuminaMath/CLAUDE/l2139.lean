import Mathlib

namespace correct_answers_statistics_probability_two_multiple_choice_A_l2139_213959

-- Define the data for schools A and B
def school_A_students : ℕ := 12
def school_A_mean : ℚ := 1
def school_A_variance : ℚ := 1

def school_B_students : ℕ := 8
def school_B_mean : ℚ := 3/2
def school_B_variance : ℚ := 1/4

-- Define the boxes
def box_A_multiple_choice : ℕ := 4
def box_A_fill_blank : ℕ := 2
def box_B_multiple_choice : ℕ := 3
def box_B_fill_blank : ℕ := 3

-- Part 1: Mean and Variance Calculation
def total_students : ℕ := school_A_students + school_B_students

theorem correct_answers_statistics : 
  let total_mean : ℚ := (school_A_students * school_A_mean + school_B_students * school_B_mean) / total_students
  let total_variance : ℚ := (school_A_students * (school_A_variance + (school_A_mean - total_mean)^2) + 
                             school_B_students * (school_B_variance + (school_B_mean - total_mean)^2)) / total_students
  total_mean = 6/5 ∧ total_variance = 19/25 := by sorry

-- Part 2: Probability Calculation
def prob_two_multiple_choice_A : ℚ := 2/5
def prob_one_multiple_one_fill_A : ℚ := 8/15
def prob_two_fill_A : ℚ := 1/15

def prob_B_multiple_given_A_two_multiple : ℚ := 5/8
def prob_B_multiple_given_A_one_each : ℚ := 8/15
def prob_B_multiple_given_A_two_fill : ℚ := 3/8

theorem probability_two_multiple_choice_A : 
  let prob_B_multiple : ℚ := prob_two_multiple_choice_A * prob_B_multiple_given_A_two_multiple + 
                              prob_one_multiple_one_fill_A * prob_B_multiple_given_A_one_each + 
                              prob_two_fill_A * prob_B_multiple_given_A_two_fill
  let prob_A_two_multiple_given_B_multiple : ℚ := (prob_two_multiple_choice_A * prob_B_multiple_given_A_two_multiple) / prob_B_multiple
  prob_A_two_multiple_given_B_multiple = 6/13 := by sorry

end correct_answers_statistics_probability_two_multiple_choice_A_l2139_213959


namespace quadratic_inequality_solution_l2139_213972

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x, ax^2 + bx + 1 > 0 ↔ -1 < x ∧ x < 1/3) → a + b = -5 := by
  sorry

end quadratic_inequality_solution_l2139_213972


namespace expand_expression_l2139_213922

theorem expand_expression (y : ℝ) : 5 * (y + 6) * (y - 3) = 5 * y^2 + 15 * y - 90 := by
  sorry

end expand_expression_l2139_213922


namespace total_cost_is_correct_l2139_213971

-- Define the package details
def package_A_price : ℝ := 10
def package_A_months : ℕ := 6
def package_A_discount : ℝ := 0.10

def package_B_price : ℝ := 12
def package_B_months : ℕ := 9
def package_B_discount : ℝ := 0.15

-- Define the tax rate
def sales_tax_rate : ℝ := 0.08

-- Define the function to calculate the total cost
def total_cost (package_A_price package_A_months package_A_discount
                package_B_price package_B_months package_B_discount
                sales_tax_rate : ℝ) : ℝ :=
  let package_A_total := package_A_price * package_A_months
  let package_B_total := package_B_price * package_B_months
  let package_A_discounted := package_A_total * (1 - package_A_discount)
  let package_B_discounted := package_B_total * (1 - package_B_discount)
  let package_A_tax := package_A_total * sales_tax_rate
  let package_B_tax := package_B_total * sales_tax_rate
  package_A_discounted + package_A_tax + package_B_discounted + package_B_tax

-- Theorem statement
theorem total_cost_is_correct :
  total_cost package_A_price package_A_months package_A_discount
             package_B_price package_B_months package_B_discount
             sales_tax_rate = 159.24 := by
  sorry

end total_cost_is_correct_l2139_213971


namespace xyz_product_l2139_213904

theorem xyz_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x * (y + z) = 120)
  (eq2 : y * (z + x) = 156)
  (eq3 : z * (x + y) = 144) :
  x * y * z = 360 := by
sorry

end xyz_product_l2139_213904


namespace fourth_number_l2139_213965

def sequence_property (a : ℕ → ℕ) : Prop :=
  ∀ n, n ≥ 3 → n ≤ 10 → a n = a (n - 1) + a (n - 2)

theorem fourth_number (a : ℕ → ℕ) (h : sequence_property a) 
  (h7 : a 7 = 42) (h9 : a 9 = 110) : a 4 = 10 := by
  sorry

end fourth_number_l2139_213965


namespace complex_fraction_simplification_l2139_213982

theorem complex_fraction_simplification :
  (Complex.I * 3 - 1) / (1 + Complex.I * 3) = Complex.mk (4/5) (3/5) := by sorry

end complex_fraction_simplification_l2139_213982


namespace jose_play_time_l2139_213987

/-- Calculates the total hours played given the minutes spent on football and basketball -/
def total_hours_played (football_minutes : ℕ) (basketball_minutes : ℕ) : ℚ :=
  (football_minutes + basketball_minutes : ℚ) / 60

/-- Proves that given Jose played football for 30 minutes and basketball for 60 minutes, 
    the total time he played is equal to 1.5 hours -/
theorem jose_play_time : total_hours_played 30 60 = 3/2 := by
  sorry

end jose_play_time_l2139_213987


namespace remainder_problem_l2139_213933

theorem remainder_problem (k : ℕ+) (h : 90 % k.val^2 = 18) : 130 % k.val = 4 := by
  sorry

end remainder_problem_l2139_213933


namespace fraction_value_theorem_l2139_213950

theorem fraction_value_theorem (x : ℝ) :
  2 / (x - 3) = 2 → x = 4 := by
  sorry

end fraction_value_theorem_l2139_213950


namespace new_average_weight_l2139_213915

def original_team_size : ℕ := 7
def original_team_avg_weight : ℚ := 94
def first_team_size : ℕ := 5
def first_team_avg_weight : ℚ := 100
def second_team_size : ℕ := 8
def second_team_avg_weight : ℚ := 90
def third_team_size : ℕ := 4
def third_team_avg_weight : ℚ := 120

theorem new_average_weight :
  let total_players := original_team_size + first_team_size + second_team_size + third_team_size
  let total_weight := original_team_size * original_team_avg_weight +
                      first_team_size * first_team_avg_weight +
                      second_team_size * second_team_avg_weight +
                      third_team_size * third_team_avg_weight
  (total_weight / total_players : ℚ) = 98.25 := by
  sorry

end new_average_weight_l2139_213915


namespace chess_tournament_games_l2139_213920

/-- The number of games played in a chess tournament --/
def num_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a chess group of 30 players, where each player plays every other player exactly once,
    and each game involves two players, the total number of games played is 435. --/
theorem chess_tournament_games :
  num_games 30 = 435 := by
  sorry

end chess_tournament_games_l2139_213920


namespace point_p_coordinates_and_b_range_l2139_213923

/-- The system of equations defining point P -/
def system_of_equations (x y a b : ℝ) : Prop :=
  x + y = 2*a - b - 4 ∧ x - y = b - 4

/-- Point P is in the second quadrant -/
def second_quadrant (x y : ℝ) : Prop :=
  x < 0 ∧ y > 0

/-- There are only three integers that satisfy the requirements for a -/
def three_integers_for_a (a : ℝ) : Prop :=
  a = 1 ∨ a = 2 ∨ a = 3

theorem point_p_coordinates_and_b_range :
  (∀ x y : ℝ, system_of_equations x y 1 1 → x = -3 ∧ y = 0) ∧
  (∀ a b : ℝ, (∃ x y : ℝ, system_of_equations x y a b ∧ second_quadrant x y) →
    three_integers_for_a a → 0 ≤ b ∧ b < 1) :=
sorry

end point_p_coordinates_and_b_range_l2139_213923


namespace sequence_periodicity_l2139_213927

def units_digit (n : ℕ) : ℕ := n % 10

def a (n : ℕ) : ℕ := units_digit (n^n)

theorem sequence_periodicity : ∀ n : ℕ, a (n + 20) = a n := by
  sorry

end sequence_periodicity_l2139_213927


namespace divisibility_property_l2139_213976

theorem divisibility_property (n : ℕ) (a : Fin n → ℕ+) 
  (h_n : n ≥ 3)
  (h_gcd : Nat.gcd (Finset.univ.prod (fun i => (a i).val)) = 1)
  (h_div : ∀ i : Fin n, (a i).val ∣ (Finset.univ.sum (fun j => (a j).val))) :
  (Finset.univ.prod (fun i => (a i).val)) ∣ (Finset.univ.sum (fun i => (a i).val))^(n-2) := by
  sorry

end divisibility_property_l2139_213976


namespace lettuce_purchase_proof_l2139_213956

/-- Calculates the total pounds of lettuce bought given the costs of green and red lettuce and the price per pound. -/
def total_lettuce_pounds (green_cost red_cost price_per_pound : ℚ) : ℚ :=
  (green_cost + red_cost) / price_per_pound

/-- Proves that given the specified costs and price per pound, the total pounds of lettuce is 7. -/
theorem lettuce_purchase_proof :
  let green_cost : ℚ := 8
  let red_cost : ℚ := 6
  let price_per_pound : ℚ := 2
  total_lettuce_pounds green_cost red_cost price_per_pound = 7 := by
sorry

#eval total_lettuce_pounds 8 6 2

end lettuce_purchase_proof_l2139_213956


namespace donut_selection_problem_l2139_213995

theorem donut_selection_problem :
  let n : ℕ := 5  -- number of donuts to select
  let k : ℕ := 4  -- number of donut types
  Nat.choose (n + k - 1) (k - 1) = 56 :=
by sorry

end donut_selection_problem_l2139_213995


namespace estimated_value_reasonable_l2139_213961

/-- The lower bound of the scale -/
def lower_bound : ℝ := 9.80

/-- The upper bound of the scale -/
def upper_bound : ℝ := 10.0

/-- The estimated value -/
def estimated_value : ℝ := 9.95

/-- Theorem stating that the estimated value is a reasonable approximation -/
theorem estimated_value_reasonable :
  lower_bound < estimated_value ∧
  estimated_value < upper_bound ∧
  (estimated_value - lower_bound) > (upper_bound - estimated_value) :=
by sorry

end estimated_value_reasonable_l2139_213961


namespace min_value_reciprocal_sum_l2139_213986

theorem min_value_reciprocal_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hsum : x + y + z = 4) : 
  (1 / x + 4 / y + 9 / z) ≥ 9 ∧ ∃ x y z, x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = 4 ∧ 1 / x + 4 / y + 9 / z = 9 :=
by sorry

end min_value_reciprocal_sum_l2139_213986


namespace max_brownies_l2139_213974

theorem max_brownies (m n : ℕ) (h1 : m > 0) (h2 : n > 0) : 
  (2 * (m - 2) * (n - 2) = 2 * m + 2 * n - 4) → m * n ≤ 84 := by
  sorry

end max_brownies_l2139_213974


namespace wheel_speed_problem_l2139_213944

theorem wheel_speed_problem (circumference : ℝ) (time_decrease : ℝ) (speed_increase : ℝ) :
  circumference = 15 →
  time_decrease = 1 / 3 →
  speed_increase = 10 →
  ∃ (original_speed : ℝ),
    original_speed * (circumference / 5280) = circumference / 5280 ∧
    (original_speed + speed_increase) * ((circumference / (5280 * original_speed)) - time_decrease / 3600) = circumference / 5280 ∧
    original_speed = 15 :=
by sorry

end wheel_speed_problem_l2139_213944


namespace both_systematic_sampling_l2139_213975

/-- Represents a sampling method --/
inductive SamplingMethod
| Systematic
| SimpleRandom
| Stratified

/-- Represents a reporter conducting interviews --/
structure Reporter where
  name : String
  interval : Nat
  intervalType : String

/-- Represents the interview setup at the train station --/
structure InterviewSetup where
  reporterA : Reporter
  reporterB : Reporter
  constantFlow : Bool

/-- Determines the sampling method based on the interview setup --/
def determineSamplingMethod (reporter : Reporter) (setup : InterviewSetup) : SamplingMethod :=
  if setup.constantFlow && (reporter.intervalType = "time" || reporter.intervalType = "people") then
    SamplingMethod.Systematic
  else
    SamplingMethod.SimpleRandom

/-- Theorem: Both reporters are using systematic sampling --/
theorem both_systematic_sampling (setup : InterviewSetup) 
  (h1 : setup.reporterA = { name := "A", interval := 10, intervalType := "time" })
  (h2 : setup.reporterB = { name := "B", interval := 1000, intervalType := "people" })
  (h3 : setup.constantFlow = true) :
  determineSamplingMethod setup.reporterA setup = SamplingMethod.Systematic ∧
  determineSamplingMethod setup.reporterB setup = SamplingMethod.Systematic := by
  sorry


end both_systematic_sampling_l2139_213975


namespace polygon_sides_doubled_l2139_213935

/-- The number of diagonals in a polygon with n sides -/
def diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: If doubling the sides of a polygon increases the diagonals by 45, the polygon has 6 sides -/
theorem polygon_sides_doubled (n : ℕ) (h : n > 3) :
  diagonals (2 * n) - diagonals n = 45 → n = 6 := by sorry

end polygon_sides_doubled_l2139_213935


namespace geometric_sequence_property_l2139_213906

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_property (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (a 1 * a 2 * a 3 = 3) →
  (a 10 * a 11 * a 12 = 24) →
  a 13 * a 14 * a 15 = 48 := by
  sorry

end geometric_sequence_property_l2139_213906


namespace card_sum_theorem_l2139_213930

theorem card_sum_theorem (n : ℕ) (m : ℕ) (h1 : n ≥ 3) (h2 : m = n * (n - 1) / 2) (h3 : Odd m) :
  ∃ k : ℕ, n - 2 = k^2 := by
  sorry

end card_sum_theorem_l2139_213930


namespace student_number_exists_l2139_213983

theorem student_number_exists : ∃ x : ℝ, Real.sqrt (2 * x^2 - 138) = 9 := by
  sorry

end student_number_exists_l2139_213983


namespace intersection_empty_implies_a_values_l2139_213985

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.2 - 3) / (p.1 - 2) = 3 ∧ p.1 ≠ 2}
def N (a : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | a * p.1 + 2 * p.2 + a = 0}

-- State the theorem
theorem intersection_empty_implies_a_values :
  ∀ a : ℝ, (M ∩ N a = ∅) → (a = -6 ∨ a = -2) :=
by sorry

end intersection_empty_implies_a_values_l2139_213985


namespace quartic_root_product_l2139_213984

theorem quartic_root_product (k : ℝ) : 
  (∃ a b c d : ℝ, 
    (a^4 - 18*a^3 + k*a^2 + 200*a - 1984 = 0) ∧
    (b^4 - 18*b^3 + k*b^2 + 200*b - 1984 = 0) ∧
    (c^4 - 18*c^3 + k*c^2 + 200*c - 1984 = 0) ∧
    (d^4 - 18*d^3 + k*d^2 + 200*d - 1984 = 0) ∧
    (a * b = -32 ∨ a * c = -32 ∨ a * d = -32 ∨ b * c = -32 ∨ b * d = -32 ∨ c * d = -32)) →
  k = 86 := by
sorry

end quartic_root_product_l2139_213984


namespace abs_c_value_l2139_213980

def f (a b c : ℤ) (x : ℂ) : ℂ := a * x^4 + b * x^3 + c * x^2 + b * x + a

theorem abs_c_value (a b c : ℤ) (h1 : Int.gcd a (Int.gcd b c) = 1) 
  (h2 : f a b c (2 + Complex.I) = 0) : 
  Int.natAbs c = 42 := by
  sorry

end abs_c_value_l2139_213980


namespace gcd_factorial_problem_l2139_213946

theorem gcd_factorial_problem : Nat.gcd (Nat.factorial 7) ((Nat.factorial 10) / (Nat.factorial 4)) = 5040 := by
  sorry

end gcd_factorial_problem_l2139_213946


namespace fishing_competition_result_l2139_213939

/-- The number of days in the fishing competition -/
def competition_days : ℕ := 5

/-- The number of fishes caught per day by the first person -/
def fishes_per_day_1 : ℕ := 6

/-- The number of fishes caught per day by the second person -/
def fishes_per_day_2 : ℕ := 4

/-- The number of fishes caught per day by the third person -/
def fishes_per_day_3 : ℕ := 8

/-- The total number of fishes caught by the team throughout the competition -/
def total_fishes : ℕ := competition_days * (fishes_per_day_1 + fishes_per_day_2 + fishes_per_day_3)

theorem fishing_competition_result : total_fishes = 90 := by
  sorry

end fishing_competition_result_l2139_213939


namespace sugar_amount_in_recipe_l2139_213955

/-- A recipe with specified amounts of ingredients -/
structure Recipe where
  flour : ℕ
  salt : ℕ
  sugar : ℕ

/-- The condition that sugar is one more cup than salt -/
def sugar_salt_relation (r : Recipe) : Prop :=
  r.sugar = r.salt + 1

theorem sugar_amount_in_recipe (r : Recipe) 
  (h1 : r.flour = 6)
  (h2 : r.salt = 7)
  (h3 : sugar_salt_relation r) :
  r.sugar = 8 := by
sorry

end sugar_amount_in_recipe_l2139_213955


namespace infinite_omega_increasing_sequence_l2139_213977

/-- The number of distinct prime divisors of a positive integer -/
def omega (n : ℕ) : ℕ := sorry

/-- The set of integers n > 1 satisfying ω(n) < ω(n+1) < ω(n+2) is infinite -/
theorem infinite_omega_increasing_sequence :
  Set.Infinite {n : ℕ | n > 1 ∧ omega n < omega (n + 1) ∧ omega (n + 1) < omega (n + 2)} :=
sorry

end infinite_omega_increasing_sequence_l2139_213977


namespace congruent_triangles_exist_l2139_213911

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  (n_ge_4 : n ≥ 4)

/-- A subset of vertices of a regular polygon -/
structure VertexSubset (n : ℕ) where
  (polygon : RegularPolygon n)
  (r : ℕ)
  (vertices : Finset (Fin n))
  (subset_size : vertices.card = r)

/-- Two triangles in a regular polygon -/
structure PolygonTrianglePair (n : ℕ) where
  (polygon : RegularPolygon n)
  (t1 t2 : Fin n → Fin n → Fin n → Prop)

/-- Congruence of two triangles in a regular polygon -/
def CongruentTriangles (n : ℕ) (pair : PolygonTrianglePair n) : Prop :=
  sorry

/-- The main theorem -/
theorem congruent_triangles_exist (n : ℕ) (V : VertexSubset n) 
  (h : V.r * (V.r - 3) ≥ n) : 
  ∃ (pair : PolygonTrianglePair n), 
    (∀ i j k, pair.t1 i j k → i ∈ V.vertices ∧ j ∈ V.vertices ∧ k ∈ V.vertices) ∧
    (∀ i j k, pair.t2 i j k → i ∈ V.vertices ∧ j ∈ V.vertices ∧ k ∈ V.vertices) ∧
    CongruentTriangles n pair :=
sorry

end congruent_triangles_exist_l2139_213911


namespace quadratic_roots_to_coefficients_l2139_213917

/-- The quadratic equation x^2 - bx + c = 0 with roots 1 and -2 has b = -1 and c = -2 -/
theorem quadratic_roots_to_coefficients :
  ∀ (b c : ℝ),
  (∀ x : ℝ, x^2 - b*x + c = 0 ↔ x = 1 ∨ x = -2) →
  b = -1 ∧ c = -2 := by
sorry

end quadratic_roots_to_coefficients_l2139_213917


namespace train_speed_crossing_bridge_l2139_213951

/-- The speed of a train crossing a bridge -/
theorem train_speed_crossing_bridge (train_length bridge_length : ℝ) (crossing_time : ℝ) :
  train_length = 165 →
  bridge_length = 850 →
  crossing_time = 67.66125376636536 →
  ∃ (speed : ℝ), (abs (speed - 54.018) < 0.001 ∧ 
    speed * crossing_time / 3.6 = train_length + bridge_length) := by
  sorry

end train_speed_crossing_bridge_l2139_213951


namespace total_candy_count_l2139_213900

theorem total_candy_count (chocolate_boxes : ℕ) (caramel_boxes : ℕ) (mint_boxes : ℕ) (berry_boxes : ℕ)
  (chocolate_caramel_pieces_per_box : ℕ) (mint_pieces_per_box : ℕ) (berry_pieces_per_box : ℕ)
  (h1 : chocolate_boxes = 7)
  (h2 : caramel_boxes = 3)
  (h3 : mint_boxes = 5)
  (h4 : berry_boxes = 4)
  (h5 : chocolate_caramel_pieces_per_box = 8)
  (h6 : mint_pieces_per_box = 10)
  (h7 : berry_pieces_per_box = 12) :
  chocolate_boxes * chocolate_caramel_pieces_per_box +
  caramel_boxes * chocolate_caramel_pieces_per_box +
  mint_boxes * mint_pieces_per_box +
  berry_boxes * berry_pieces_per_box = 178 := by
sorry

end total_candy_count_l2139_213900


namespace circles_externally_tangent_l2139_213998

/-- Two circles are externally tangent when the distance between their centers
    equals the sum of their radii -/
def externally_tangent (r₁ r₂ d : ℝ) : Prop := d = r₁ + r₂

/-- The problem statement -/
theorem circles_externally_tangent :
  let r₁ : ℝ := 1
  let r₂ : ℝ := 3
  let d : ℝ := 4
  externally_tangent r₁ r₂ d :=
by
  sorry

end circles_externally_tangent_l2139_213998


namespace stone_heap_theorem_l2139_213943

/-- 
Given k ≥ 3 heaps of stones with 1, 2, ..., k stones respectively,
after merging heaps, the final number of stones p is given by
p = (k + 1) * (3k - 1) / 8.
This function returns p given k.
-/
def final_stones (k : ℕ) : ℚ :=
  (k + 1) * (3 * k - 1) / 8

/-- 
This theorem states that for k ≥ 3, the final number of stones p
is a perfect square if and only if both 2k + 2 and 3k + 1 are perfect squares,
and that the least k satisfying this condition is 161.
-/
theorem stone_heap_theorem (k : ℕ) (h : k ≥ 3) :
  (∃ n : ℕ, final_stones k = n^2) ↔ 
  (∃ x y : ℕ, 2*k + 2 = x^2 ∧ 3*k + 1 = y^2) ∧
  (∀ m : ℕ, m < 161 → ¬(∃ x y : ℕ, 2*m + 2 = x^2 ∧ 3*m + 1 = y^2)) :=
sorry

end stone_heap_theorem_l2139_213943


namespace total_decrease_percentage_l2139_213994

-- Define the percentage decreases
def first_year_decrease : ℝ := 0.4
def second_year_decrease : ℝ := 0.1

-- Define the theorem
theorem total_decrease_percentage :
  ∀ (initial_value : ℝ), initial_value > 0 →
  let value_after_first_year := initial_value * (1 - first_year_decrease)
  let final_value := value_after_first_year * (1 - second_year_decrease)
  let total_decrease := (initial_value - final_value) / initial_value
  total_decrease = 0.46 := by
  sorry

end total_decrease_percentage_l2139_213994


namespace units_digit_of_42_cubed_plus_27_squared_l2139_213967

theorem units_digit_of_42_cubed_plus_27_squared : ∃ n : ℕ, 42^3 + 27^2 = 10 * n + 7 := by
  sorry

end units_digit_of_42_cubed_plus_27_squared_l2139_213967


namespace modulus_of_specific_complex_l2139_213919

theorem modulus_of_specific_complex : let z : ℂ := 1 + Complex.I * Real.sqrt 3
  ‖z‖ = 2 := by sorry

end modulus_of_specific_complex_l2139_213919


namespace equal_even_odd_probability_l2139_213929

/-- The number of dice being rolled -/
def numDice : ℕ := 6

/-- The number of sides on each die -/
def numSides : ℕ := 6

/-- The probability of rolling an even number on a single die -/
def probEven : ℚ := 1/2

/-- The probability of rolling an odd number on a single die -/
def probOdd : ℚ := 1/2

/-- The number of dice that need to show even (and odd) numbers for the desired outcome -/
def numEven : ℕ := numDice / 2

theorem equal_even_odd_probability :
  (Nat.choose numDice numEven : ℚ) * probEven ^ numDice = 5/16 := by
  sorry

end equal_even_odd_probability_l2139_213929


namespace ellipse_equation_l2139_213989

/-- Given an ellipse with the following properties:
  1. The axes of symmetry lie on the coordinate axes
  2. One endpoint of the minor axis and the two foci form an equilateral triangle
  3. The distance from the foci to the same vertex is √3
  Then the standard equation of the ellipse is x²/12 + y²/9 = 1 or y²/12 + x²/9 = 1 -/
theorem ellipse_equation (a c : ℝ) (h1 : a = 2 * c) (h2 : a - c = Real.sqrt 3) :
  ∃ (x y : ℝ), (x^2 / 12 + y^2 / 9 = 1) ∨ (y^2 / 12 + x^2 / 9 = 1) :=
sorry

end ellipse_equation_l2139_213989


namespace positive_root_of_cubic_equation_l2139_213947

theorem positive_root_of_cubic_equation :
  ∃ x : ℝ, x > 0 ∧ x^3 - 5*x^2 + 2*x - Real.sqrt 3 = 0 :=
by
  use 3 + Real.sqrt 3
  sorry

end positive_root_of_cubic_equation_l2139_213947


namespace todays_production_l2139_213979

def average_production (total_production : ℕ) (days : ℕ) : ℚ :=
  (total_production : ℚ) / (days : ℚ)

theorem todays_production
  (h1 : average_production (9 * 50) 9 = 50)
  (h2 : average_production ((9 * 50) + x) 10 = 55)
  : x = 100 := by
  sorry

end todays_production_l2139_213979


namespace distance_to_line_l2139_213918

/-- The distance from a point on the line y = ax - 2a + 5 to the line x - 2y + 3 = 0 is √5 -/
theorem distance_to_line : ∀ (a : ℝ), ∃ (A : ℝ × ℝ),
  (A.2 = a * A.1 - 2 * a + 5) ∧ 
  (|A.1 - 2 * A.2 + 3| / Real.sqrt (1 + 4) = Real.sqrt 5) :=
by sorry

end distance_to_line_l2139_213918


namespace course_selection_schemes_l2139_213991

theorem course_selection_schemes (pe art : ℕ) (total_courses : ℕ) : 
  pe = 4 → art = 4 → total_courses = pe + art →
  (Nat.choose pe 1 * Nat.choose art 1) + 
  (Nat.choose pe 2 * Nat.choose art 1 + Nat.choose pe 1 * Nat.choose art 2) = 64 :=
by sorry

end course_selection_schemes_l2139_213991


namespace largest_common_divisor_of_consecutive_squares_l2139_213909

theorem largest_common_divisor_of_consecutive_squares (n : ℤ) (h : Even n) :
  (∃ (k : ℤ), k > 1 ∧ ∀ (b : ℤ), k ∣ ((n + 1)^2 - n^2)) → False :=
by sorry

end largest_common_divisor_of_consecutive_squares_l2139_213909


namespace unique_triple_solution_l2139_213910

theorem unique_triple_solution (a b p : ℕ) (h_prime : Nat.Prime p) :
  (a + b)^p = p^a + p^b ↔ a = 1 ∧ b = 1 ∧ p = 2 :=
by sorry

end unique_triple_solution_l2139_213910


namespace parallelogram_theorem_l2139_213916

/-- A point in a 2D plane -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A quadrilateral defined by four points -/
structure Quadrilateral :=
  (A B C D : Point)

/-- Checks if a quadrilateral is convex -/
def is_convex (q : Quadrilateral) : Prop := sorry

/-- Checks if a point is the midpoint of a line segment -/
def is_midpoint (M : Point) (A B : Point) : Prop := sorry

/-- Checks if a point is inside a quadrilateral -/
def is_inside (M : Point) (q : Quadrilateral) : Prop := sorry

/-- Checks if four points form a parallelogram -/
def is_parallelogram (A B C D : Point) : Prop := sorry

/-- The main theorem -/
theorem parallelogram_theorem (ABCD : Quadrilateral) (P Q R S M : Point) :
  is_convex ABCD →
  is_midpoint P ABCD.A ABCD.B →
  is_midpoint Q ABCD.B ABCD.C →
  is_midpoint R ABCD.C ABCD.D →
  is_midpoint S ABCD.D ABCD.A →
  is_inside M ABCD →
  is_parallelogram ABCD.A P M S →
  is_parallelogram ABCD.C R M Q := by
  sorry

end parallelogram_theorem_l2139_213916


namespace next_base3_number_l2139_213997

/-- Converts a base 3 number represented as a list of digits to its decimal equivalent -/
def base3ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ i)) 0

/-- Converts a decimal number to its base 3 representation as a list of digits -/
def decimalToBase3 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 3) ((m % 3) :: acc)
    aux n []

/-- The base 3 representation of M -/
def M : List Nat := [0, 2, 0, 1]

theorem next_base3_number (h : base3ToDecimal M = base3ToDecimal [0, 2, 0, 1]) :
  decimalToBase3 (base3ToDecimal M + 1) = [1, 2, 0, 1] := by
  sorry

end next_base3_number_l2139_213997


namespace combine_expression_l2139_213949

theorem combine_expression (a b : ℝ) : 3 * (2 * a - 3 * b) - 6 * (a - b) = -3 * b := by
  sorry

end combine_expression_l2139_213949


namespace right_triangle_consecutive_legs_l2139_213925

theorem right_triangle_consecutive_legs (a : ℕ) :
  let b := a + 1
  let c := Real.sqrt (a^2 + b^2)
  c^2 = 2*a^2 + 2*a + 1 :=
by sorry

end right_triangle_consecutive_legs_l2139_213925


namespace bicycle_discount_proof_l2139_213996

theorem bicycle_discount_proof (original_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) :
  original_price = 200 ∧ discount1 = 0.4 ∧ discount2 = 0.25 →
  original_price * (1 - discount1) * (1 - discount2) = 90 := by
  sorry

end bicycle_discount_proof_l2139_213996


namespace three_students_same_group_probability_l2139_213970

theorem three_students_same_group_probability 
  (total_groups : ℕ) 
  (student_count : ℕ) 
  (h1 : total_groups = 4) 
  (h2 : student_count ≥ 3) 
  (h3 : student_count % total_groups = 0) :
  (1 : ℚ) / (total_groups ^ 2) = 1 / 16 :=
sorry

end three_students_same_group_probability_l2139_213970


namespace cherry_pitting_time_l2139_213913

/-- Proves that it takes 2 hours to pit cherries for a pie given the specified conditions -/
theorem cherry_pitting_time :
  ∀ (pounds_needed : ℕ) 
    (cherries_per_pound : ℕ) 
    (cherries_per_set : ℕ) 
    (minutes_per_set : ℕ),
  pounds_needed = 3 →
  cherries_per_pound = 80 →
  cherries_per_set = 20 →
  minutes_per_set = 10 →
  (pounds_needed * cherries_per_pound * minutes_per_set) / 
  (cherries_per_set * 60) = 2 := by
sorry

end cherry_pitting_time_l2139_213913


namespace trailing_zeros_of_square_l2139_213903

/-- The number of trailing zeros in (10^12 - 5)^2 is 12 -/
theorem trailing_zeros_of_square : ∃ n : ℕ, (10^12 - 5)^2 = n * 10^12 ∧ n % 10 ≠ 0 :=
sorry

end trailing_zeros_of_square_l2139_213903


namespace open_sets_l2139_213914

-- Define the concept of an open set in a plane
def is_open_set (A : Set (ℝ × ℝ)) : Prop :=
  ∀ (x₀ y₀ : ℝ), (x₀, y₀) ∈ A → 
    ∃ (r : ℝ), r > 0 ∧ {(x, y) | (x - x₀)^2 + (y - y₀)^2 < r^2} ⊆ A

-- Define the four sets
def set1 : Set (ℝ × ℝ) := {(x, y) | x^2 + y^2 = 1}
def set2 : Set (ℝ × ℝ) := {(x, y) | |x + y + 2| ≥ 1}
def set3 : Set (ℝ × ℝ) := {(x, y) | |x| + |y| < 1}
def set4 : Set (ℝ × ℝ) := {(x, y) | 0 < x^2 + (y - 1)^2 ∧ x^2 + (y - 1)^2 < 1}

-- State the theorem
theorem open_sets : 
  ¬(is_open_set set1) ∧ 
  ¬(is_open_set set2) ∧ 
  (is_open_set set3) ∧ 
  (is_open_set set4) := by
  sorry

end open_sets_l2139_213914


namespace number_problem_l2139_213940

theorem number_problem (x : ℝ) : (x / 4 + 15 = 27) → x = 48 := by
  sorry

end number_problem_l2139_213940


namespace solve_for_x_l2139_213926

theorem solve_for_x (x y : ℝ) (h1 : x + 2*y = 100) (h2 : y = 25) : x = 50 := by
  sorry

end solve_for_x_l2139_213926


namespace everett_work_weeks_l2139_213924

/-- Given that Everett worked 5 hours every day and a total of 140 hours,
    prove that he worked for 4 weeks. -/
theorem everett_work_weeks :
  let hours_per_day : ℕ := 5
  let total_hours : ℕ := 140
  let days_per_week : ℕ := 7
  let hours_per_week : ℕ := hours_per_day * days_per_week
  total_hours / hours_per_week = 4 := by
sorry

end everett_work_weeks_l2139_213924


namespace complement_N_star_in_N_l2139_213928

def N : Set ℕ := {n : ℕ | True}
def N_star : Set ℕ := {n : ℕ | n > 0}

theorem complement_N_star_in_N : N \ N_star = {0} := by sorry

end complement_N_star_in_N_l2139_213928


namespace arithmetic_calculation_l2139_213963

theorem arithmetic_calculation : 3127 + 240 / 60 * 5 - 227 = 2920 := by
  sorry

end arithmetic_calculation_l2139_213963


namespace function_value_at_negative_one_l2139_213962

/-- Given a function f(x) = a*sin(x) + b*tan(x) + 3 where a and b are real numbers,
    if f(1) = 1, then f(-1) = 5. -/
theorem function_value_at_negative_one 
  (a b : ℝ) 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = a * Real.sin x + b * Real.tan x + 3) 
  (h2 : f 1 = 1) : 
  f (-1) = 5 := by
  sorry

end function_value_at_negative_one_l2139_213962


namespace hyperbola_equation_l2139_213964

theorem hyperbola_equation (x y : ℝ) :
  (∃ (f : ℝ × ℝ), (f.1^2 / 16 - f.2^2 / 4 = 1) ∧
   ((x^2 / 15 - y^2 / 5 = 1) → (f = (x, y) ∨ f = (-x, y)))) →
  (x^2 / 15 - y^2 / 5 = 1) →
  ((3 * Real.sqrt 2)^2 / 15 - 2^2 / 5 = 1) :=
sorry

end hyperbola_equation_l2139_213964


namespace odd_not_even_function_implication_l2139_213957

def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| - |x - a|

theorem odd_not_even_function_implication (a : ℝ) :
  (∀ x, f a x = -f a (-x)) →
  (∃ x, f a x ≠ f a (-x)) →
  (∃ x, f a x ≠ 0) →
  (a + 1)^2016 = 0 := by
  sorry

end odd_not_even_function_implication_l2139_213957


namespace log_ratio_identity_l2139_213932

theorem log_ratio_identity (a b x : ℝ) (ha : a > 0) (ha' : a ≠ 1) (hb : b > 0) (hx : x > 0) :
  (Real.log x / Real.log a) / (Real.log x / Real.log (a * b)) = 1 + Real.log b / Real.log a := by
  sorry

end log_ratio_identity_l2139_213932


namespace hyperbola_m_value_l2139_213942

/-- Represents a hyperbola with parameter m -/
structure Hyperbola (m : ℝ) where
  eq : ∀ x y : ℝ, 3 * m * x^2 - m * y^2 = 3

/-- The distance from the center to a focus of the hyperbola -/
def focal_distance (h : Hyperbola m) : ℝ := 2

theorem hyperbola_m_value (h : Hyperbola m) 
  (focus : focal_distance h = 2) : m = -1 := by
  sorry

end hyperbola_m_value_l2139_213942


namespace probability_x_plus_y_less_than_4_l2139_213902

/-- A square in the 2D plane --/
structure Square where
  bottomLeft : ℝ × ℝ
  sideLength : ℝ

/-- The probability that a randomly chosen point in the square satisfies a condition --/
def probability (s : Square) (condition : ℝ × ℝ → Prop) : ℝ :=
  sorry

/-- The square with vertices (0, 0), (0, 3), (3, 3), and (3, 0) --/
def givenSquare : Square :=
  { bottomLeft := (0, 0), sideLength := 3 }

/-- The condition x + y < 4 --/
def condition (p : ℝ × ℝ) : Prop :=
  p.1 + p.2 < 4

theorem probability_x_plus_y_less_than_4 :
  probability givenSquare condition = 7 / 9 := by
  sorry

end probability_x_plus_y_less_than_4_l2139_213902


namespace escalator_length_is_126_l2139_213937

/-- Calculates the length of an escalator given its speed, a person's walking speed on it, and the time taken to cover the entire length. -/
def escalator_length (escalator_speed : ℝ) (person_speed : ℝ) (time : ℝ) : ℝ :=
  (escalator_speed + person_speed) * time

/-- Proves that the length of the escalator is 126 feet under the given conditions. -/
theorem escalator_length_is_126 :
  escalator_length 11 3 9 = 126 := by
  sorry

#eval escalator_length 11 3 9

end escalator_length_is_126_l2139_213937


namespace min_packs_for_120_cans_l2139_213912

/-- Represents the available pack sizes for soda cans -/
inductive PackSize
  | small : PackSize
  | medium : PackSize
  | large : PackSize

/-- Returns the number of cans in a given pack size -/
def cansInPack (size : PackSize) : ℕ :=
  match size with
  | .small => 8
  | .medium => 16
  | .large => 32

/-- Represents a combination of packs -/
structure PackCombination where
  small : ℕ
  medium : ℕ
  large : ℕ

/-- Calculates the total number of cans in a pack combination -/
def totalCans (combo : PackCombination) : ℕ :=
  combo.small * cansInPack PackSize.small +
  combo.medium * cansInPack PackSize.medium +
  combo.large * cansInPack PackSize.large

/-- Calculates the total number of packs in a pack combination -/
def totalPacks (combo : PackCombination) : ℕ :=
  combo.small + combo.medium + combo.large

/-- Checks if a pack combination is valid for the given total cans -/
def isValidCombination (combo : PackCombination) (totalCansNeeded : ℕ) : Prop :=
  totalCans combo = totalCansNeeded

/-- Theorem: The minimum number of packs needed to buy exactly 120 cans of soda is 5 -/
theorem min_packs_for_120_cans :
  ∃ (minCombo : PackCombination),
    isValidCombination minCombo 120 ∧
    totalPacks minCombo = 5 ∧
    ∀ (combo : PackCombination),
      isValidCombination combo 120 → totalPacks combo ≥ totalPacks minCombo := by
  sorry

end min_packs_for_120_cans_l2139_213912


namespace honey_servings_calculation_l2139_213981

/-- The number of servings in a container of honey -/
def number_of_servings (container_volume : ℚ) (serving_size : ℚ) : ℚ :=
  container_volume / serving_size

/-- Proof that a container with 37 2/3 tablespoons of honey contains 25 1/9 servings when each serving is 1 1/2 tablespoons -/
theorem honey_servings_calculation :
  let container_volume : ℚ := 113/3  -- 37 2/3 as an improper fraction
  let serving_size : ℚ := 3/2        -- 1 1/2 as an improper fraction
  number_of_servings container_volume serving_size = 226/9
  := by sorry

end honey_servings_calculation_l2139_213981


namespace candy_ratio_l2139_213993

/-- Given:
  - There were 22 sweets on the table initially.
  - Jack took some portion of all the candies and 4 more candies.
  - Paul took the remaining 7 sweets.
Prove that the ratio of candies Jack took (excluding the 4 additional candies) 
to the total number of candies is 1/2. -/
theorem candy_ratio : 
  ∀ (jack_portion : ℕ),
  jack_portion + 4 + 7 = 22 →
  (jack_portion : ℚ) / 22 = 1 / 2 := by
sorry

end candy_ratio_l2139_213993


namespace some_board_game_masters_enjoy_logic_puzzles_l2139_213960

-- Define the universe
variable (U : Type)

-- Define predicates
variable (M : U → Prop)  -- M x means x is a mathematics enthusiast
variable (B : U → Prop)  -- B x means x is a board game master
variable (L : U → Prop)  -- L x means x enjoys logic puzzles

-- State the theorem
theorem some_board_game_masters_enjoy_logic_puzzles
  (h1 : ∀ x, M x → L x)  -- All mathematics enthusiasts enjoy logic puzzles
  (h2 : ∃ x, B x ∧ M x)  -- Some board game masters are mathematics enthusiasts
  : ∃ x, B x ∧ L x :=    -- Some board game masters enjoy logic puzzles
by
  sorry


end some_board_game_masters_enjoy_logic_puzzles_l2139_213960


namespace retail_price_calculation_l2139_213953

theorem retail_price_calculation (wholesale_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) : 
  wholesale_price = 90 ∧ 
  discount_rate = 0.1 ∧ 
  profit_rate = 0.2 →
  ∃ retail_price : ℝ, 
    retail_price * (1 - discount_rate) = wholesale_price * (1 + profit_rate) ∧
    retail_price = 120 := by
sorry

end retail_price_calculation_l2139_213953


namespace square_to_rectangle_area_ratio_l2139_213954

/-- The ratio of the area of a square with side length 30 cm to the area of a rectangle with dimensions 28 cm by 45 cm is 5/7. -/
theorem square_to_rectangle_area_ratio : 
  let square_side : ℝ := 30
  let rect_length : ℝ := 28
  let rect_width : ℝ := 45
  let square_area := square_side ^ 2
  let rect_area := rect_length * rect_width
  square_area / rect_area = 5 / 7 := by sorry

end square_to_rectangle_area_ratio_l2139_213954


namespace geometry_theorem_l2139_213968

-- Define the types for planes and lines
variable (α β : Plane) (m n : Line)

-- Define the perpendicular relation between a line and a plane
def perpendicularToPlane (l : Line) (p : Plane) : Prop := sorry

-- Define parallel relation between lines
def parallelLines (l1 l2 : Line) : Prop := sorry

-- Define skew relation between lines
def skewLines (l1 l2 : Line) : Prop := sorry

-- Define parallel relation between planes
def parallelPlanes (p1 p2 : Plane) : Prop := sorry

-- Define intersection relation between planes
def planesIntersect (p1 p2 : Plane) : Prop := sorry

-- Define perpendicular relation between planes
def perpendicularPlanes (p1 p2 : Plane) : Prop := sorry

-- Define perpendicular relation between lines
def perpendicularLines (l1 l2 : Line) : Prop := sorry

-- State the theorem
theorem geometry_theorem 
  (h1 : perpendicularToPlane m α) 
  (h2 : perpendicularToPlane n β) :
  (parallelLines m n → parallelPlanes α β) ∧ 
  (skewLines m n → planesIntersect α β) ∧
  (perpendicularPlanes α β → perpendicularLines m n) := by
  sorry

end geometry_theorem_l2139_213968


namespace imaginary_part_of_z_l2139_213945

theorem imaginary_part_of_z (z : ℂ) (h : z * (2 + Complex.I) = 1) :
  z.im = -1/5 := by
  sorry

end imaginary_part_of_z_l2139_213945


namespace marie_keeps_remainder_l2139_213908

/-- The number of lollipops Marie keeps for herself -/
def lollipops_kept (total_lollipops : ℕ) (num_friends : ℕ) : ℕ :=
  total_lollipops % num_friends

/-- The total number of lollipops Marie has -/
def total_lollipops : ℕ := 75 + 132 + 9 + 315

/-- The number of friends Marie has -/
def num_friends : ℕ := 13

theorem marie_keeps_remainder :
  lollipops_kept total_lollipops num_friends = 11 := by
  sorry

end marie_keeps_remainder_l2139_213908


namespace divisor_problem_l2139_213969

theorem divisor_problem (x : ℕ) : 
  (95 / x = 6 ∧ 95 % x = 5) → x = 15 := by
  sorry

end divisor_problem_l2139_213969


namespace sum_of_squares_of_coefficients_l2139_213999

def polynomial (x : ℝ) : ℝ := 3 * (x^5 + 5*x^3 + 2)

theorem sum_of_squares_of_coefficients :
  (3^2 : ℝ) + 0^2 + 15^2 + 0^2 + 0^2 + 6^2 = 270 :=
by sorry

end sum_of_squares_of_coefficients_l2139_213999


namespace square_area_error_l2139_213941

theorem square_area_error (x : ℝ) (h : x > 0) :
  let measured_side := x * 1.12
  let actual_area := x^2
  let calculated_area := measured_side^2
  let error_percentage := (calculated_area - actual_area) / actual_area * 100
  error_percentage = 25.44 := by sorry

end square_area_error_l2139_213941


namespace quadratic_symmetry_l2139_213988

-- Define the quadratic function
def p (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Theorem statement
theorem quadratic_symmetry (a b c : ℝ) :
  (∀ x : ℝ, p a b c x = p a b c (21 - x)) →  -- Axis of symmetry at x = 10.5
  p a b c 0 = -4 →                           -- p(0) = -4
  p a b c 21 = -4 :=                         -- Conclusion: p(21) = -4
by
  sorry


end quadratic_symmetry_l2139_213988


namespace square_completion_l2139_213938

theorem square_completion (x : ℝ) : x^2 + 6*x - 5 = 0 ↔ (x + 3)^2 = 14 := by
  sorry

end square_completion_l2139_213938


namespace jasons_textbooks_l2139_213958

/-- Represents the problem of determining the number of textbooks Jason has. -/
theorem jasons_textbooks :
  let bookcase_limit : ℕ := 80  -- Maximum weight the bookcase can hold in pounds
  let hardcover_count : ℕ := 70  -- Number of hardcover books
  let hardcover_weight : ℚ := 1/2  -- Weight of each hardcover book in pounds
  let textbook_weight : ℕ := 2  -- Weight of each textbook in pounds
  let knickknack_count : ℕ := 3  -- Number of knick-knacks
  let knickknack_weight : ℕ := 6  -- Weight of each knick-knack in pounds
  let over_limit : ℕ := 33  -- Amount the total collection is over the weight limit in pounds

  let total_weight := bookcase_limit + over_limit
  let hardcover_total_weight := hardcover_count * hardcover_weight
  let knickknack_total_weight := knickknack_count * knickknack_weight
  let textbook_total_weight := total_weight - (hardcover_total_weight + knickknack_total_weight)

  textbook_total_weight / textbook_weight = 30 := by
  sorry

end jasons_textbooks_l2139_213958


namespace f_sum_property_l2139_213921

noncomputable def f (x : ℝ) : ℝ := 1 / (3^x + Real.sqrt 3)

theorem f_sum_property (x : ℝ) : f x + f (1 - x) = Real.sqrt 3 / 3 := by
  sorry

end f_sum_property_l2139_213921


namespace angle_complement_quadrant_l2139_213992

/-- An angle is in the fourth quadrant if it's between 270° and 360° (exclusive) -/
def is_fourth_quadrant (α : Real) : Prop :=
  270 < α ∧ α < 360

/-- An angle is in the third quadrant if it's between 180° and 270° (exclusive) -/
def is_third_quadrant (α : Real) : Prop :=
  180 < α ∧ α < 270

theorem angle_complement_quadrant (α : Real) :
  is_fourth_quadrant α → is_third_quadrant (180 - α) := by
  sorry

end angle_complement_quadrant_l2139_213992


namespace quadratic_equation_solution_l2139_213934

theorem quadratic_equation_solution :
  ∃! y : ℝ, y^2 + 6*y + 8 = -(y + 4)*(y + 6) :=
by
  -- The unique solution is y = -4
  use -4
  sorry

end quadratic_equation_solution_l2139_213934


namespace binary_1101100_equals_108_l2139_213966

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_1101100_equals_108 :
  binary_to_decimal [false, false, true, true, false, true, true] = 108 := by
  sorry

end binary_1101100_equals_108_l2139_213966


namespace other_number_proof_l2139_213948

theorem other_number_proof (x : Float) : 
  (0.5 : Float) = x + 0.33333333333333337 → x = 0.16666666666666663 := by
  sorry

end other_number_proof_l2139_213948


namespace fourth_root_of_x_sqrt_x_squared_l2139_213978

theorem fourth_root_of_x_sqrt_x_squared (x : ℝ) (hx : x > 0) : 
  (((x * Real.sqrt x) ^ 2) ^ (1/4 : ℝ)) = x ^ (3/4 : ℝ) :=
sorry

end fourth_root_of_x_sqrt_x_squared_l2139_213978


namespace eighth_term_is_128_l2139_213936

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, n > 0 → ∃ q : ℝ, a (n + 1) = q * a n
  second_term : a 2 = 2
  product_condition : a 3 * a 4 = 32

/-- The 8th term of the geometric sequence is 128 -/
theorem eighth_term_is_128 (seq : GeometricSequence) : seq.a 8 = 128 := by
  sorry

end eighth_term_is_128_l2139_213936


namespace shane_leftover_bread_l2139_213973

/-- The number of slices of bread leftover after making sandwiches --/
def bread_leftover (bread_packages : ℕ) (slices_per_bread_package : ℕ) 
                   (ham_packages : ℕ) (slices_per_ham_package : ℕ) 
                   (bread_per_sandwich : ℕ) : ℕ :=
  let total_bread := bread_packages * slices_per_bread_package
  let total_ham := ham_packages * slices_per_ham_package
  let sandwiches := total_ham
  let bread_used := sandwiches * bread_per_sandwich
  total_bread - bread_used

/-- Theorem stating that Shane will have 8 slices of bread leftover --/
theorem shane_leftover_bread : 
  bread_leftover 2 20 2 8 2 = 8 := by
  sorry

end shane_leftover_bread_l2139_213973


namespace election_percentages_correct_l2139_213931

def votes : List Nat := [1136, 7636, 10628, 8562, 6490]

def total_votes : Nat := votes.sum

def percentage (votes : Nat) (total : Nat) : Float :=
  (votes.toFloat / total.toFloat) * 100

def percentages : List Float :=
  votes.map (λ v => percentage v total_votes)

theorem election_percentages_correct :
  percentages ≈ [3.20, 21.54, 29.98, 24.15, 18.30] := by
  sorry

end election_percentages_correct_l2139_213931


namespace Q_has_exactly_one_negative_root_l2139_213907

def Q (x : ℝ) : ℝ := x^7 + 5*x^5 + 5*x^4 - 6*x^3 - 2*x^2 - 10*x + 12

theorem Q_has_exactly_one_negative_root :
  ∃! x : ℝ, x < 0 ∧ Q x = 0 :=
sorry

end Q_has_exactly_one_negative_root_l2139_213907


namespace total_chairs_count_l2139_213990

theorem total_chairs_count : ℕ := by
  -- Define the number of rows and chairs per row for each section
  let first_section_rows : ℕ := 5
  let first_section_chairs_per_row : ℕ := 10
  let second_section_rows : ℕ := 8
  let second_section_chairs_per_row : ℕ := 12

  -- Define the number of late arrivals and extra chairs per late arrival
  let late_arrivals : ℕ := 20
  let extra_chairs_per_late_arrival : ℕ := 3

  -- Calculate the total number of chairs
  let total_chairs := 
    (first_section_rows * first_section_chairs_per_row) +
    (second_section_rows * second_section_chairs_per_row) +
    (late_arrivals * extra_chairs_per_late_arrival)

  -- Prove that the total number of chairs is 206
  have h : total_chairs = 206 := by sorry

  exact 206


end total_chairs_count_l2139_213990


namespace triangle_side_length_l2139_213905

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  b = Real.sqrt 3 →
  A = π / 4 →
  B = π / 3 →
  (Real.sin A) / a = (Real.sin B) / b →
  a = Real.sqrt 2 := by
  sorry

end triangle_side_length_l2139_213905


namespace smallest_x_value_l2139_213901

theorem smallest_x_value (x y z : ℝ) 
  (sum_condition : x + y + z = 6)
  (product_condition : x * y + x * z + y * z = 10) :
  ∀ x' : ℝ, (∃ y' z' : ℝ, x' + y' + z' = 6 ∧ x' * y' + x' * z' + y' * z' = 10) → x' ≥ 2/3 :=
by sorry

end smallest_x_value_l2139_213901


namespace angle_terminal_side_l2139_213952

/-- Given an angle α whose terminal side passes through the point (m, -3) 
    and whose cosine is -4/5, prove that m = -4. -/
theorem angle_terminal_side (α : Real) (m : Real) : 
  (∃ (x y : Real), x = m ∧ y = -3 ∧ x = Real.cos α * Real.sqrt (x^2 + y^2) ∧ y = Real.sin α * Real.sqrt (x^2 + y^2)) →
  Real.cos α = -4/5 →
  m = -4 :=
by sorry

end angle_terminal_side_l2139_213952

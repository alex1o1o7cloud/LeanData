import Mathlib

namespace NUMINAMATH_CALUDE_only_height_weight_correlated_l1955_195586

-- Define the concept of correlation
def correlated (X Y : Type) (relation : X → Y → Prop) : Prop :=
  ∃ (x₁ x₂ : X) (y₁ y₂ : Y), x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧ relation x₁ y₁ ∧ relation x₂ y₂

-- Define the concept of a definite functional relationship
def definite_functional (X Y : Type) (f : X → Y) : Prop :=
  ∀ (x₁ x₂ : X), x₁ ≠ x₂ → f x₁ ≠ f x₂

-- Variables
variable (TaxiFare Distance HouseSize HousePrice Height Weight BlockSize BlockMass : Type)

-- Relationships
variable (taxi_relation : TaxiFare → Distance → Prop)
variable (house_relation : HouseSize → HousePrice → Prop)
variable (height_weight_relation : Height → Weight → Prop)
variable (block_relation : BlockSize → BlockMass → Prop)

-- Hypotheses
variable (h1 : ∃ f : TaxiFare → Distance, definite_functional TaxiFare Distance f)
variable (h2 : ∃ f : HouseSize → HousePrice, definite_functional HouseSize HousePrice f)
variable (h3 : ¬(∃ f : Height → Weight, definite_functional Height Weight f))
variable (h4 : ∃ f : BlockSize → BlockMass, definite_functional BlockSize BlockMass f)

-- Theorem
theorem only_height_weight_correlated :
  ¬(correlated TaxiFare Distance taxi_relation) ∧
  ¬(correlated HouseSize HousePrice house_relation) ∧
  (correlated Height Weight height_weight_relation) ∧
  ¬(correlated BlockSize BlockMass block_relation) :=
sorry

end NUMINAMATH_CALUDE_only_height_weight_correlated_l1955_195586


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l1955_195542

theorem consecutive_integers_sum (x : ℤ) :
  x * (x + 1) * (x + 2) = 2730 → x + (x + 1) + (x + 2) = 42 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l1955_195542


namespace NUMINAMATH_CALUDE_linear_equation_solution_l1955_195517

theorem linear_equation_solution (a b m : ℝ) : 
  (∀ y, (a + b) * y^2 - y^((1/3)*a + 2) + 5 = 0 → (a + b = 0 ∧ (1/3)*a + 2 = 1)) →
  ((a + 2)/6 - (a - 1)/2 + 3 = a - (2*a - m)/6) →
  |a - b| - |b - m| = -32 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l1955_195517


namespace NUMINAMATH_CALUDE_roots_custom_op_result_l1955_195514

-- Define the custom operation
def customOp (a b : ℝ) : ℝ := a * b - a - b

-- State the theorem
theorem roots_custom_op_result :
  ∀ x₁ x₂ : ℝ,
  (x₁^2 + x₁ - 1 = 0) →
  (x₂^2 + x₂ - 1 = 0) →
  customOp x₁ x₂ = 1 := by
sorry

end NUMINAMATH_CALUDE_roots_custom_op_result_l1955_195514


namespace NUMINAMATH_CALUDE_total_parts_calculation_l1955_195532

theorem total_parts_calculation (sample_size : ℕ) (probability : ℚ) (N : ℕ) : 
  sample_size = 30 → probability = 1/4 → N * probability = sample_size → N = 120 := by
  sorry

end NUMINAMATH_CALUDE_total_parts_calculation_l1955_195532


namespace NUMINAMATH_CALUDE_system_solution_l1955_195580

theorem system_solution (x y : ℝ) : 
  x^3 - x + 1 = y^2 ∧ y^3 - y + 1 = x^2 → 
  (x = 1 ∧ y = 1) ∨ (x = 1 ∧ y = -1) ∨ (x = -1 ∧ y = 1) ∨ (x = -1 ∧ y = -1) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1955_195580


namespace NUMINAMATH_CALUDE_polygon_sides_from_angle_sum_l1955_195503

theorem polygon_sides_from_angle_sum (n : ℕ) (angle_sum : ℝ) : 
  angle_sum = 720 → (n - 2) * 180 = angle_sum → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_from_angle_sum_l1955_195503


namespace NUMINAMATH_CALUDE_range_of_m_l1955_195556

theorem range_of_m (x y m : ℝ) : 
  x > 0 → y > 0 → x + y = 3 → 
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 3 → 
    (4 / (x + 1)) + (16 / y) > m^2 - 3*m + 5) → 
  -1 < m ∧ m < 4 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l1955_195556


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l1955_195531

/-- Given a geometric sequence {a_n} where a_1 = 1/9 and a_4 = 3, 
    the product of the first five terms is equal to 1 -/
theorem geometric_sequence_product (a : ℕ → ℝ) : 
  (∀ n, a (n + 1) = a n * (a 4 / a 1)^(1/3)) → -- Geometric sequence condition
  a 1 = 1/9 →                                  -- First term condition
  a 4 = 3 →                                    -- Fourth term condition
  a 1 * a 2 * a 3 * a 4 * a 5 = 1 :=            -- Product of first five terms
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l1955_195531


namespace NUMINAMATH_CALUDE_new_person_weight_is_68_l1955_195565

/-- The weight of the new person given the conditions of the problem -/
def new_person_weight (initial_count : ℕ) (average_increase : ℝ) (replaced_weight : ℝ) : ℝ :=
  replaced_weight + initial_count * average_increase

/-- Theorem stating that the weight of the new person is 68 kg -/
theorem new_person_weight_is_68 :
  new_person_weight 6 3.5 47 = 68 := by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_is_68_l1955_195565


namespace NUMINAMATH_CALUDE_correct_sums_l1955_195538

theorem correct_sums (R W : ℕ) : W = 5 * R → R + W = 180 → R = 30 := by
  sorry

end NUMINAMATH_CALUDE_correct_sums_l1955_195538


namespace NUMINAMATH_CALUDE_marble_selection_problem_l1955_195596

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem marble_selection_problem :
  let total_marbles : ℕ := 15
  let required_marbles : ℕ := 2
  let marbles_to_choose : ℕ := 5
  let remaining_marbles : ℕ := total_marbles - required_marbles
  let additional_marbles : ℕ := marbles_to_choose - required_marbles
  choose remaining_marbles additional_marbles = 286 :=
by sorry

end NUMINAMATH_CALUDE_marble_selection_problem_l1955_195596


namespace NUMINAMATH_CALUDE_divisibility_implies_seven_divides_l1955_195507

theorem divisibility_implies_seven_divides (n : ℕ) : 
  n ≥ 2 → (n ∣ 3^n + 4^n) → (7 ∣ n) := by sorry

end NUMINAMATH_CALUDE_divisibility_implies_seven_divides_l1955_195507


namespace NUMINAMATH_CALUDE_quadratic_minimum_l1955_195571

/-- 
Given a quadratic function y = ax² + px + q where a ≠ 0,
if the minimum value of y is m, then q = m + p²/(4a)
-/
theorem quadratic_minimum (a p q m : ℝ) (ha : a ≠ 0) :
  (∀ x, a * x^2 + p * x + q ≥ m) →
  (∃ x₀, a * x₀^2 + p * x₀ + q = m) →
  q = m + p^2 / (4 * a) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l1955_195571


namespace NUMINAMATH_CALUDE_homework_time_reduction_l1955_195579

theorem homework_time_reduction (initial_time final_time : ℝ) (x : ℝ) :
  initial_time = 100 →
  final_time = 70 →
  0 < x →
  x < 1 →
  initial_time * (1 - x)^2 = final_time :=
by
  sorry

end NUMINAMATH_CALUDE_homework_time_reduction_l1955_195579


namespace NUMINAMATH_CALUDE_cost_price_is_36_l1955_195589

/-- Given the total cloth length, total selling price, and loss per metre, 
    calculate the cost price for one metre of cloth. -/
def cost_price_per_metre (total_length : ℕ) (total_selling_price : ℕ) (loss_per_metre : ℕ) : ℕ :=
  (total_selling_price + total_length * loss_per_metre) / total_length

/-- Theorem: The cost price for one metre of cloth is Rs. 36 given the problem conditions. -/
theorem cost_price_is_36 :
  cost_price_per_metre 300 9000 6 = 36 := by
  sorry

#eval cost_price_per_metre 300 9000 6

end NUMINAMATH_CALUDE_cost_price_is_36_l1955_195589


namespace NUMINAMATH_CALUDE_third_term_is_twenty_l1955_195553

/-- A geometric sequence with positive integer terms -/
structure GeometricSequence where
  terms : ℕ → ℕ
  is_geometric : ∀ n : ℕ, n > 0 → terms (n + 1) * terms (n - 1) = (terms n) ^ 2

/-- Our specific geometric sequence -/
def our_sequence : GeometricSequence where
  terms := sorry
  is_geometric := sorry

theorem third_term_is_twenty 
  (h1 : our_sequence.terms 1 = 5)
  (h5 : our_sequence.terms 5 = 320) : 
  our_sequence.terms 3 = 20 := by
  sorry

end NUMINAMATH_CALUDE_third_term_is_twenty_l1955_195553


namespace NUMINAMATH_CALUDE_road_repair_equation_l1955_195555

theorem road_repair_equation (x : ℝ) (h : x > 0) : 
  (150 / x - 150 / (x + 5) = 5) ↔ 
  (∃ (original_days actual_days : ℝ), 
    original_days > 0 ∧ 
    actual_days > 0 ∧ 
    original_days = 150 / x ∧ 
    actual_days = 150 / (x + 5) ∧ 
    original_days - actual_days = 5) :=
by sorry

end NUMINAMATH_CALUDE_road_repair_equation_l1955_195555


namespace NUMINAMATH_CALUDE_game_score_product_l1955_195587

def score_function (n : ℕ) : ℕ :=
  if n % 3 = 0 then 9
  else if n % 2 = 0 then 3
  else if n % 3 ≠ 0 then 1
  else 0

def allie_rolls : List ℕ := [5, 2, 6, 1, 3]
def betty_rolls : List ℕ := [6, 4, 1, 2, 5]

def total_score (rolls : List ℕ) : ℕ :=
  (rolls.map score_function).sum

theorem game_score_product : 
  (total_score allie_rolls) * (total_score betty_rolls) = 391 := by
  sorry

end NUMINAMATH_CALUDE_game_score_product_l1955_195587


namespace NUMINAMATH_CALUDE_river_depth_calculation_l1955_195530

/-- Proves that given a river with specified width, flow rate, and discharge,
    the depth of the river is as calculated. -/
theorem river_depth_calculation
  (width : ℝ)
  (flow_rate_kmph : ℝ)
  (discharge_per_minute : ℝ)
  (h1 : width = 25)
  (h2 : flow_rate_kmph = 8)
  (h3 : discharge_per_minute = 26666.666666666668) :
  let flow_rate_mpm := flow_rate_kmph * 1000 / 60
  let depth := discharge_per_minute / (width * flow_rate_mpm)
  depth = 8 := by sorry

end NUMINAMATH_CALUDE_river_depth_calculation_l1955_195530


namespace NUMINAMATH_CALUDE_town_population_l1955_195550

theorem town_population (pet_owners_percentage : Real) 
  (dog_owners_fraction : Real) (cat_owners : ℕ) :
  pet_owners_percentage = 0.6 →
  dog_owners_fraction = 0.5 →
  cat_owners = 30 →
  (cat_owners : Real) / (1 - dog_owners_fraction) / pet_owners_percentage = 100 :=
by sorry

end NUMINAMATH_CALUDE_town_population_l1955_195550


namespace NUMINAMATH_CALUDE_sum4_is_27857_l1955_195573

/-- A geometric sequence with specific properties -/
structure GeometricSequence where
  a : ℝ  -- first term
  r : ℝ  -- common ratio
  sum3 : a + a*r + a*r^2 = 13
  sum5 : a + a*r + a*r^2 + a*r^3 + a*r^4 = 121

/-- The sum of the first 4 terms of the geometric sequence -/
def sum4 (seq : GeometricSequence) : ℝ :=
  seq.a + seq.a * seq.r + seq.a * seq.r^2 + seq.a * seq.r^3

/-- Theorem stating that the sum of the first 4 terms is 27.857 -/
theorem sum4_is_27857 (seq : GeometricSequence) : sum4 seq = 27.857 := by
  sorry

end NUMINAMATH_CALUDE_sum4_is_27857_l1955_195573


namespace NUMINAMATH_CALUDE_product_sequence_sum_l1955_195516

theorem product_sequence_sum (a b : ℕ) (h1 : a / 3 = 16) (h2 : b = a - 1) : a + b = 95 := by
  sorry

end NUMINAMATH_CALUDE_product_sequence_sum_l1955_195516


namespace NUMINAMATH_CALUDE_items_distribution_count_l1955_195541

-- Define the number of items and bags
def num_items : ℕ := 5
def num_bags : ℕ := 4

-- Define a function to calculate the number of ways to distribute items
def distribute_items (items : ℕ) (bags : ℕ) : ℕ :=
  sorry

-- Theorem statement
theorem items_distribution_count :
  distribute_items num_items num_bags = 52 := by
  sorry

end NUMINAMATH_CALUDE_items_distribution_count_l1955_195541


namespace NUMINAMATH_CALUDE_divisors_of_2_pow_48_minus_1_l1955_195578

theorem divisors_of_2_pow_48_minus_1 :
  ∃! (a b : ℕ), 60 < a ∧ a < 70 ∧ 60 < b ∧ b < 70 ∧
  (2^48 - 1) % a = 0 ∧ (2^48 - 1) % b = 0 ∧
  a = 63 ∧ b = 65 := by sorry

end NUMINAMATH_CALUDE_divisors_of_2_pow_48_minus_1_l1955_195578


namespace NUMINAMATH_CALUDE_regular_polygon_144_degrees_has_10_sides_l1955_195533

/-- A regular polygon with interior angles of 144 degrees has 10 sides -/
theorem regular_polygon_144_degrees_has_10_sides :
  ∀ n : ℕ,
  n > 2 →
  (n - 2 : ℝ) * 180 / n = 144 →
  n = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_144_degrees_has_10_sides_l1955_195533


namespace NUMINAMATH_CALUDE_shortest_path_length_l1955_195594

-- Define the circles
def circle1 (x y : ℝ) : Prop := (x - 7.5)^2 + (y - 10)^2 = 36
def circle2 (x y : ℝ) : Prop := (x - 15)^2 + (y - 5)^2 = 16

-- Define a path that avoids the circles
def valid_path (p : ℝ → ℝ × ℝ) : Prop :=
  (p 0 = (0, 0)) ∧ 
  (p 1 = (15, 20)) ∧ 
  ∀ t ∈ (Set.Icc 0 1), ¬(circle1 (p t).1 (p t).2) ∧ ¬(circle2 (p t).1 (p t).2)

-- Define the length of a path
def path_length (p : ℝ → ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem shortest_path_length :
  ∃ p, valid_path p ∧ 
    path_length p = 30.6 + 5 * Real.pi / 3 ∧
    ∀ q, valid_path q → path_length p ≤ path_length q :=
sorry

end NUMINAMATH_CALUDE_shortest_path_length_l1955_195594


namespace NUMINAMATH_CALUDE_sum_of_squares_l1955_195535

theorem sum_of_squares (x y z a b c : ℝ) 
  (h1 : x/a + y/b + z/c = 5)
  (h2 : a/x + b/y + c/z = 3) :
  x^2/a^2 + y^2/b^2 + z^2/c^2 = 13 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l1955_195535


namespace NUMINAMATH_CALUDE_comparison_of_powers_l1955_195515

theorem comparison_of_powers (a b c : ℝ) : 
  a = 10 ∧ b = -49 ∧ c = -50 → 
  a^b - 2 * a^c = 8 * a^c := by
  sorry

end NUMINAMATH_CALUDE_comparison_of_powers_l1955_195515


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1955_195592

theorem sufficient_not_necessary (a : ℝ) :
  (∀ a, a > 1 → 1/a < 1) ∧ (∃ a, 1/a < 1 ∧ a ≤ 1) := by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1955_195592


namespace NUMINAMATH_CALUDE_root_relationship_l1955_195547

/-- Given two functions f and g, and their respective roots x₁ and x₂, prove that x₁ < x₂ -/
theorem root_relationship (f g : ℝ → ℝ) (x₁ x₂ : ℝ) :
  (f = λ x => x + 2^x) →
  (g = λ x => x + Real.log x) →
  f x₁ = 0 →
  g x₂ = 0 →
  x₁ < x₂ := by
  sorry

end NUMINAMATH_CALUDE_root_relationship_l1955_195547


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l1955_195577

theorem sum_of_a_and_b (a b : ℚ) 
  (eq1 : a + 3 * b = 27) 
  (eq2 : 5 * a + 2 * b = 40) : 
  a + b = 161 / 13 := by
sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l1955_195577


namespace NUMINAMATH_CALUDE_geometric_sequence_a9_l1955_195548

def is_geometric_sequence (a : ℕ → ℤ) : Prop :=
  ∃ q : ℤ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_a9 (a : ℕ → ℤ) :
  is_geometric_sequence a →
  a 2 * a 5 = -32 →
  a 3 + a 4 = 4 →
  (∃ q : ℤ, ∀ n : ℕ, a (n + 1) = a n * q) →
  a 9 = -256 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a9_l1955_195548


namespace NUMINAMATH_CALUDE_triangle_inequality_theorem_l1955_195564

-- Define a structure for a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  positive_a : 0 < a
  positive_b : 0 < b
  positive_c : 0 < c
  triangle_inequality_ab : a + b > c
  triangle_inequality_bc : b + c > a
  triangle_inequality_ca : c + a > b

-- State the theorem
theorem triangle_inequality_theorem (t : Triangle) :
  t.a^2 * t.c * (t.a - t.b) + t.b^2 * t.a * (t.b - t.c) + t.c^2 * t.b * (t.c - t.a) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_theorem_l1955_195564


namespace NUMINAMATH_CALUDE_vehicles_separation_time_l1955_195527

/-- Given two vehicles moving in opposite directions, calculate the time taken to reach a specific distance apart. -/
theorem vehicles_separation_time
  (initial_distance : ℝ)
  (speed1 speed2 : ℝ)
  (final_distance : ℝ)
  (h1 : initial_distance = 5)
  (h2 : speed1 = 60)
  (h3 : speed2 = 40)
  (h4 : final_distance = 85) :
  (final_distance - initial_distance) / (speed1 + speed2) = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_vehicles_separation_time_l1955_195527


namespace NUMINAMATH_CALUDE_no_positive_lower_bound_l1955_195543

/-- The number of positive integers not containing the digit 9 that are less than or equal to n -/
def f (n : ℕ+) : ℕ := sorry

/-- For any positive real number p, there exists a positive integer n such that f(n)/n < p -/
theorem no_positive_lower_bound :
  ∀ p : ℝ, p > 0 → ∃ n : ℕ+, (f n : ℝ) / n < p :=
sorry

end NUMINAMATH_CALUDE_no_positive_lower_bound_l1955_195543


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l1955_195501

/-- Given a quadratic equation x^2 - 3x + k - 2 = 0 with two real roots x1 and x2 -/
theorem quadratic_equation_properties (k : ℝ) (x1 x2 : ℝ) 
  (h1 : x1^2 - 3*x1 + k - 2 = 0)
  (h2 : x2^2 - 3*x2 + k - 2 = 0)
  (h3 : x1 ≠ x2) :
  (k ≤ 17/4) ∧ 
  (x1 + x2 - x1*x2 = 1 → k = 4) := by
  sorry


end NUMINAMATH_CALUDE_quadratic_equation_properties_l1955_195501


namespace NUMINAMATH_CALUDE_solution_approximation_l1955_195599

/-- The solution to the equation (0.0077 * 4.5) / (x * 0.1 * 0.007) = 990 is approximately 28571.42 -/
theorem solution_approximation : ∃ x : ℝ, 
  (0.0077 * 4.5) / (x * 0.1 * 0.007) = 990 ∧ 
  abs (x - 28571.42) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_solution_approximation_l1955_195599


namespace NUMINAMATH_CALUDE_linear_equation_solution_l1955_195590

theorem linear_equation_solution : ∃ (x y : ℤ), x + y = 5 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l1955_195590


namespace NUMINAMATH_CALUDE_black_balls_count_l1955_195561

theorem black_balls_count (total_balls : ℕ) (white_balls : ℕ → ℕ) (black_balls : ℕ) :
  total_balls = 56 →
  white_balls black_balls = 6 * black_balls →
  total_balls = white_balls black_balls + black_balls →
  black_balls = 8 := by
sorry

end NUMINAMATH_CALUDE_black_balls_count_l1955_195561


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l1955_195544

theorem geometric_sequence_first_term
  (a : ℝ)  -- first term of the sequence
  (r : ℝ)  -- common ratio
  (h1 : a * r^2 = 27)  -- third term is 27
  (h2 : a * r^3 = 81)  -- fourth term is 81
  : a = 3 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l1955_195544


namespace NUMINAMATH_CALUDE_ben_needs_14_eggs_l1955_195545

/-- Represents the weekly egg requirements for a community -/
structure EggRequirements where
  saly : ℕ
  ben : ℕ
  ked : ℕ
  total_monthly : ℕ

/-- The number of weeks in a month -/
def weeks_in_month : ℕ := 4

/-- Checks if the given egg requirements are valid -/
def is_valid_requirements (req : EggRequirements) : Prop :=
  req.saly = 10 ∧
  req.ked = req.ben / 2 ∧
  req.total_monthly = weeks_in_month * (req.saly + req.ben + req.ked)

/-- Theorem stating that Ben needs 14 eggs per week -/
theorem ben_needs_14_eggs (req : EggRequirements) 
  (h : is_valid_requirements req) (h_total : req.total_monthly = 124) : 
  req.ben = 14 := by
  sorry


end NUMINAMATH_CALUDE_ben_needs_14_eggs_l1955_195545


namespace NUMINAMATH_CALUDE_parabola_equation_l1955_195552

/-- A parabola with vertex at the origin and directrix x = -2 has the equation y^2 = 8x -/
theorem parabola_equation (p : ℝ × ℝ → Prop) : 
  (∀ x y, p (x, y) ↔ y^2 = 8*x) ↔ 
  (∀ x y, p (x, y) → (x, y) ≠ (0, 0)) ∧ 
  (∀ x, x = -2 → ∀ y, ¬p (x, y)) := by
sorry

end NUMINAMATH_CALUDE_parabola_equation_l1955_195552


namespace NUMINAMATH_CALUDE_system_solution_l1955_195505

theorem system_solution : ∃ (x y : ℚ), (4 * x - 3 * y = -13) ∧ (5 * x + 3 * y = -14) ∧ (x = -3) ∧ (y = 1/3) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1955_195505


namespace NUMINAMATH_CALUDE_coefficient_x4_is_4374_l1955_195576

/-- The coefficient of x^4 in the expansion of ((4x^2 + 6x + 9/4)^4) -/
def coefficient_x4 : ℕ :=
  let a := 4  -- coefficient of x^2
  let b := 6  -- coefficient of x
  let c := 9/4  -- constant term
  let n := 4  -- power of the binomial
  -- The actual calculation of the coefficient would go here
  4374  -- This is the result we want to prove

/-- The expansion of ((4x^2 + 6x + 9/4)^4) has 4374 as the coefficient of x^4 -/
theorem coefficient_x4_is_4374 : coefficient_x4 = 4374 := by
  sorry


end NUMINAMATH_CALUDE_coefficient_x4_is_4374_l1955_195576


namespace NUMINAMATH_CALUDE_wall_passing_art_l1955_195570

theorem wall_passing_art (k : ℕ) (n : ℝ) (h1 : k = 8) (h2 : k > 0) (h3 : n > 0) :
  k * Real.sqrt (k / n) = Real.sqrt (k + k / n) → n = 63 := by
  sorry

end NUMINAMATH_CALUDE_wall_passing_art_l1955_195570


namespace NUMINAMATH_CALUDE_arc_length_sixty_degree_l1955_195593

/-- Given a circle with circumference 60 feet and an arc subtended by a central angle of 60°,
    the length of the arc is 10 feet. -/
theorem arc_length_sixty_degree (circle : Real → Real → Prop) 
  (center : Real × Real) (radius : Real) :
  (2 * Real.pi * radius = 60) →  -- Circumference is 60 feet
  (∀ (θ : Real), 0 ≤ θ ∧ θ ≤ 2 * Real.pi → 
    circle (center.1 + radius * Real.cos θ) (center.2 + radius * Real.sin θ)) →
  (10 : Real) = (60 / 6) := by sorry

end NUMINAMATH_CALUDE_arc_length_sixty_degree_l1955_195593


namespace NUMINAMATH_CALUDE_range_of_f_l1955_195598

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 2*x + 3

-- State the theorem
theorem range_of_f :
  {y : ℝ | ∃ x ≥ 0, f x = y} = {y : ℝ | y ≥ 3} := by
  sorry

end NUMINAMATH_CALUDE_range_of_f_l1955_195598


namespace NUMINAMATH_CALUDE_E_equals_F_l1955_195511

def E : Set ℝ := { x | ∃ n : ℤ, x = Real.cos (n * Real.pi / 3) }

def F : Set ℝ := { x | ∃ m : ℤ, x = Real.sin ((2 * m - 3) * Real.pi / 6) }

theorem E_equals_F : E = F := by
  sorry

end NUMINAMATH_CALUDE_E_equals_F_l1955_195511


namespace NUMINAMATH_CALUDE_square_sum_of_xy_l1955_195526

theorem square_sum_of_xy (x y : ℕ+) 
  (h1 : x * y + x + y = 35)
  (h2 : x^2 * y + x * y^2 = 306) : 
  x^2 + y^2 = 290 := by
sorry

end NUMINAMATH_CALUDE_square_sum_of_xy_l1955_195526


namespace NUMINAMATH_CALUDE_balanced_quadruple_inequality_l1955_195567

def balanced (a b c d : ℝ) : Prop :=
  a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ a + b + c + d = a^2 + b^2 + c^2 + d^2

theorem balanced_quadruple_inequality (x : ℝ) :
  (∀ a b c d : ℝ, balanced a b c d → (x - a) * (x - b) * (x - c) * (x - d) ≥ 0) ↔
  x ≥ 3/2 := by sorry

end NUMINAMATH_CALUDE_balanced_quadruple_inequality_l1955_195567


namespace NUMINAMATH_CALUDE_negative_f_m_plus_one_l1955_195572

theorem negative_f_m_plus_one 
  (f : ℝ → ℝ) 
  (a m : ℝ) 
  (h1 : ∀ x, f x = x^2 - x + a) 
  (h2 : f (-m) < 0) : 
  f (m + 1) < 0 := by
sorry

end NUMINAMATH_CALUDE_negative_f_m_plus_one_l1955_195572


namespace NUMINAMATH_CALUDE_square_area_l1955_195510

/-- The area of a square with side length 13 cm is 169 square centimeters. -/
theorem square_area (side_length : ℝ) (h : side_length = 13) : side_length ^ 2 = 169 := by
  sorry

end NUMINAMATH_CALUDE_square_area_l1955_195510


namespace NUMINAMATH_CALUDE_two_oak_trees_cut_down_l1955_195560

/-- The number of oak trees cut down in the park --/
def oak_trees_cut_down (initial : ℕ) (final : ℕ) : ℕ :=
  initial - final

/-- Theorem: Given the initial and final number of oak trees, prove that 2 trees were cut down --/
theorem two_oak_trees_cut_down :
  oak_trees_cut_down 9 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_oak_trees_cut_down_l1955_195560


namespace NUMINAMATH_CALUDE_quadratic_function_unique_l1955_195518

def quadratic_function (a b c : ℝ) : ℝ → ℝ := λ x => a * x^2 + b * x + c

theorem quadratic_function_unique 
  (f : ℝ → ℝ) 
  (h1 : ∃ a b c : ℝ, f = quadratic_function a b c)
  (h2 : f (-2) = 0)
  (h3 : f 4 = 0)
  (h4 : ∀ x : ℝ, f x ≤ 9)
  (h5 : ∃ x : ℝ, f x = 9) :
  f = quadratic_function (-1) 2 8 := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_unique_l1955_195518


namespace NUMINAMATH_CALUDE_mangoes_in_basket_l1955_195557

/-- The number of mangoes in a basket of fruits -/
def mangoes_count (total_fruits : ℕ) (pears : ℕ) (pawpaws : ℕ) (lemons : ℕ) : ℕ :=
  total_fruits - (pears + pawpaws + lemons + lemons)

theorem mangoes_in_basket :
  mangoes_count 58 10 12 9 = 18 :=
by sorry

end NUMINAMATH_CALUDE_mangoes_in_basket_l1955_195557


namespace NUMINAMATH_CALUDE_total_trees_equals_sum_our_park_total_is_55_l1955_195502

/-- Represents the number of walnut trees in a park -/
structure WalnutPark where
  initial : Nat  -- Initial number of walnut trees
  planted : Nat  -- Number of walnut trees planted

/-- Calculates the total number of walnut trees after planting -/
def total_trees (park : WalnutPark) : Nat :=
  park.initial + park.planted

/-- Theorem: The total number of walnut trees after planting is the sum of initial and planted trees -/
theorem total_trees_equals_sum (park : WalnutPark) : 
  total_trees park = park.initial + park.planted := by
  sorry

/-- The specific park instance from the problem -/
def our_park : WalnutPark := { initial := 22, planted := 33 }

/-- Theorem: The total number of walnut trees in our park after planting is 55 -/
theorem our_park_total_is_55 : total_trees our_park = 55 := by
  sorry

end NUMINAMATH_CALUDE_total_trees_equals_sum_our_park_total_is_55_l1955_195502


namespace NUMINAMATH_CALUDE_matrix_is_own_inverse_l1955_195559

def A (a b c : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  !![a, b, c; 2, -1, 0; 0, 0, 1]

theorem matrix_is_own_inverse (a b c : ℝ) :
  A a b c * A a b c = 1 → a = 1 ∧ b = 0 ∧ c = 0 := by
  sorry

end NUMINAMATH_CALUDE_matrix_is_own_inverse_l1955_195559


namespace NUMINAMATH_CALUDE_dinner_attendees_l1955_195563

theorem dinner_attendees (total_clinks : ℕ) : total_clinks = 45 → ∃ x : ℕ, x = 10 ∧ x * (x - 1) / 2 = total_clinks := by
  sorry

end NUMINAMATH_CALUDE_dinner_attendees_l1955_195563


namespace NUMINAMATH_CALUDE_problem_solution_l1955_195595

def arithmetic_sum (a₁ n d : ℕ) : ℕ := n * (2 * a₁ + (n - 1) * d) / 2

theorem problem_solution : 
  ∃ x : ℚ, 
    let n : ℕ := (196 - 2) / 2 + 1
    let S : ℕ := arithmetic_sum 2 n 2
    (S + x) / (n + 1 : ℚ) = 50 * x ∧ x = 2 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l1955_195595


namespace NUMINAMATH_CALUDE_xavier_probability_of_success_l1955_195597

theorem xavier_probability_of_success 
  (p_yvonne : ℝ) 
  (p_zelda : ℝ) 
  (p_xavier_and_yvonne_not_zelda : ℝ) 
  (h1 : p_yvonne = 2/3) 
  (h2 : p_zelda = 5/8) 
  (h3 : p_xavier_and_yvonne_not_zelda = 0.0625) :
  ∃ p_xavier : ℝ, 
    p_xavier * p_yvonne * (1 - p_zelda) = p_xavier_and_yvonne_not_zelda ∧ 
    p_xavier = 0.25 :=
by sorry

end NUMINAMATH_CALUDE_xavier_probability_of_success_l1955_195597


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1955_195529

/-- Represents a hyperbola with its asymptote equation coefficient -/
structure Hyperbola where
  k : ℝ
  asymptote_eq : ∀ (x y : ℝ), y = k * x ∨ y = -k * x

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ := sorry

/-- Theorem stating the eccentricity of a hyperbola with asymptote equations y = ± x/2 -/
theorem hyperbola_eccentricity (h : Hyperbola) (h_asymptote : h.k = 1/2) :
  eccentricity h = Real.sqrt 5 / 2 ∨ eccentricity h = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1955_195529


namespace NUMINAMATH_CALUDE_caco3_decomposition_spontaneity_l1955_195566

/-- Represents the thermodynamic properties of a chemical reaction -/
structure ThermodynamicProperties where
  ΔH : ℝ  -- Enthalpy change
  ΔS : ℝ  -- Entropy change

/-- Calculates the Gibbs free energy change for a given temperature -/
def gibbsFreeEnergyChange (props : ThermodynamicProperties) (T : ℝ) : ℝ :=
  props.ΔH - T * props.ΔS

/-- Theorem: For the CaCO₃ decomposition reaction, there exists a temperature
    above which the reaction becomes spontaneous -/
theorem caco3_decomposition_spontaneity 
    (props : ThermodynamicProperties) 
    (h_endothermic : props.ΔH > 0) 
    (h_disorder_increase : props.ΔS > 0) : 
    ∃ T₀ : ℝ, ∀ T > T₀, gibbsFreeEnergyChange props T < 0 := by
  sorry

end NUMINAMATH_CALUDE_caco3_decomposition_spontaneity_l1955_195566


namespace NUMINAMATH_CALUDE_lost_to_remaining_ratio_l1955_195506

def initial_amount : ℕ := 5000
def motorcycle_cost : ℕ := 2800
def final_amount : ℕ := 825

def amount_after_motorcycle : ℕ := initial_amount - motorcycle_cost
def concert_ticket_cost : ℕ := amount_after_motorcycle / 2
def amount_after_concert : ℕ := amount_after_motorcycle - concert_ticket_cost
def amount_lost : ℕ := amount_after_concert - final_amount

theorem lost_to_remaining_ratio :
  (amount_lost : ℚ) / (amount_after_concert : ℚ) = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_lost_to_remaining_ratio_l1955_195506


namespace NUMINAMATH_CALUDE_molecule_count_l1955_195584

-- Define Avogadro's constant
def avogadro_constant : ℝ := 6.022e23

-- Define the number of molecules
def number_of_molecules : ℝ := 3e26

-- Theorem to prove
theorem molecule_count : number_of_molecules = 3e26 := by
  sorry

end NUMINAMATH_CALUDE_molecule_count_l1955_195584


namespace NUMINAMATH_CALUDE_parabola_zeros_difference_l1955_195523

/-- Represents a parabola of the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The y-coordinate for a given x on the parabola -/
def Parabola.y (p : Parabola) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

/-- Checks if a point (x, y) is on the parabola -/
def Parabola.containsPoint (p : Parabola) (x y : ℝ) : Prop :=
  p.y x = y

/-- The zeros of the parabola -/
def Parabola.zeros (p : Parabola) : Set ℝ :=
  {x : ℝ | p.y x = 0}

theorem parabola_zeros_difference (p : Parabola) :
  p.containsPoint 3 (-9) →
  p.containsPoint 5 7 →
  ∃ m n : ℝ, m ∈ p.zeros ∧ n ∈ p.zeros ∧ m > n ∧ m - n = 3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_zeros_difference_l1955_195523


namespace NUMINAMATH_CALUDE_x_range_when_ln_x_less_than_neg_one_l1955_195591

theorem x_range_when_ln_x_less_than_neg_one (x : ℝ) (h : Real.log x < -1) : 0 < x ∧ x < Real.exp (-1) :=
sorry

end NUMINAMATH_CALUDE_x_range_when_ln_x_less_than_neg_one_l1955_195591


namespace NUMINAMATH_CALUDE_protege_zero_implies_two_and_five_l1955_195520

/-- A digit is a protégé of a natural number if it is the units digit of some divisor of that number. -/
def isProtege (d : Nat) (n : Nat) : Prop :=
  ∃ k : Nat, k ∣ n ∧ k % 10 = d

/-- Theorem: If 0 is a protégé of a natural number, then 2 and 5 are also protégés of that number. -/
theorem protege_zero_implies_two_and_five (n : Nat) :
  isProtege 0 n → isProtege 2 n ∧ isProtege 5 n := by
  sorry


end NUMINAMATH_CALUDE_protege_zero_implies_two_and_five_l1955_195520


namespace NUMINAMATH_CALUDE_monic_quadratic_with_complex_root_l1955_195536

theorem monic_quadratic_with_complex_root :
  ∃! (a b : ℝ), ∀ (x : ℂ), x^2 + a*x + b = 0 ↔ x = 2 - I ∨ x = 2 + I :=
by sorry

end NUMINAMATH_CALUDE_monic_quadratic_with_complex_root_l1955_195536


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1955_195551

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x) - 2 * x

theorem inequality_equivalence :
  ∀ x : ℝ, f (x^2 - 4) + f (3 * x) > 0 ↔ x > 1 ∨ x < -4 :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1955_195551


namespace NUMINAMATH_CALUDE_exponential_inequality_implies_upper_bound_l1955_195534

theorem exponential_inequality_implies_upper_bound (a : ℝ) :
  (∀ x : ℝ, Real.exp (2 * x) - (a - 3) * Real.exp x + 4 - 3 * a > 0) →
  a < 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_exponential_inequality_implies_upper_bound_l1955_195534


namespace NUMINAMATH_CALUDE_functions_equal_at_three_l1955_195575

-- Define the interval (2, 4)
def OpenInterval := {x : ℝ | 2 < x ∧ x < 4}

-- Define the properties of functions f and g
def FunctionProperties (f g : ℝ → ℝ) : Prop :=
  (∀ x ∈ OpenInterval, 2 < f x ∧ f x < 4) ∧
  (∀ x ∈ OpenInterval, 2 < g x ∧ g x < 4) ∧
  (∀ x ∈ OpenInterval, f (g x) = x) ∧
  (∀ x ∈ OpenInterval, g (f x) = x) ∧
  (∀ x ∈ OpenInterval, f x * g x = x^2)

-- Theorem statement
theorem functions_equal_at_three 
  (f g : ℝ → ℝ) 
  (h : FunctionProperties f g) : 
  f 3 = g 3 := by
  sorry


end NUMINAMATH_CALUDE_functions_equal_at_three_l1955_195575


namespace NUMINAMATH_CALUDE_discounted_cd_cost_l1955_195525

/-- The cost of five CDs with a 10% discount, given the cost of two CDs and the discount condition -/
theorem discounted_cd_cost (cost_of_two : ℝ) (discount_rate : ℝ) : 
  cost_of_two = 40 →
  discount_rate = 0.1 →
  (5 : ℝ) * (cost_of_two / 2) * (1 - discount_rate) = 90 := by
sorry

end NUMINAMATH_CALUDE_discounted_cd_cost_l1955_195525


namespace NUMINAMATH_CALUDE_lives_lost_l1955_195540

/-- Represents the number of lives Kaleb had initially -/
def initial_lives : ℕ := 98

/-- Represents the number of lives Kaleb had remaining -/
def remaining_lives : ℕ := 73

/-- Theorem stating that the number of lives Kaleb lost is 25 -/
theorem lives_lost : initial_lives - remaining_lives = 25 := by
  sorry

end NUMINAMATH_CALUDE_lives_lost_l1955_195540


namespace NUMINAMATH_CALUDE_rectangle_quadrilateral_inequality_l1955_195524

theorem rectangle_quadrilateral_inequality 
  (m n a b c d : ℝ) 
  (hm : m > 0) 
  (hn : n > 0) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hc : c > 0) 
  (hd : d > 0) 
  (h_rectangle : ∃ (x y z s t u v w : ℝ), 
    x + w = m ∧ y + z = n ∧ s + t = n ∧ u + v = m ∧
    a^2 = x^2 + y^2 ∧ b^2 = z^2 + s^2 ∧ c^2 = t^2 + u^2 ∧ d^2 = v^2 + w^2) :
  1 ≤ (a^2 + b^2 + c^2 + d^2) / (m^2 + n^2) ∧ (a^2 + b^2 + c^2 + d^2) / (m^2 + n^2) ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_quadrilateral_inequality_l1955_195524


namespace NUMINAMATH_CALUDE_function_root_implies_parameter_range_l1955_195537

theorem function_root_implies_parameter_range (a : ℝ) :
  (∃ x : ℝ, x ∈ Set.Ioo 0 1 ∧ a^2 * x^2 - 2*a*x + 1 = 0) →
  a > 1 := by sorry

end NUMINAMATH_CALUDE_function_root_implies_parameter_range_l1955_195537


namespace NUMINAMATH_CALUDE_complex_quadrant_l1955_195585

theorem complex_quadrant (z : ℂ) (h : -2 * I * z = 1 - I) : 
  0 < z.re ∧ 0 < z.im :=
sorry

end NUMINAMATH_CALUDE_complex_quadrant_l1955_195585


namespace NUMINAMATH_CALUDE_vectors_form_basis_l1955_195568

def v1 : Fin 2 → ℝ := ![2, 3]
def v2 : Fin 2 → ℝ := ![-4, 6]

theorem vectors_form_basis : LinearIndependent ℝ ![v1, v2] :=
sorry

end NUMINAMATH_CALUDE_vectors_form_basis_l1955_195568


namespace NUMINAMATH_CALUDE_simplified_expression_value_l1955_195539

theorem simplified_expression_value (a b : ℤ) (ha : a = 2) (hb : b = -3) :
  10 * a^2 * b - (2 * a * b^2 - 2 * (a * b - 5 * a^2 * b)) = -48 := by
  sorry

end NUMINAMATH_CALUDE_simplified_expression_value_l1955_195539


namespace NUMINAMATH_CALUDE_complex_expression_simplification_l1955_195558

theorem complex_expression_simplification :
  ∀ (i : ℂ), i^2 = -1 → 7 * (4 - 2*i) + 4*i * (6 - 3*i) = 40 + 10*i := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_simplification_l1955_195558


namespace NUMINAMATH_CALUDE_minimum_bailing_rate_l1955_195500

/-- Represents the problem of determining the minimum bailing rate for a leaking boat --/
theorem minimum_bailing_rate 
  (distance_to_shore : ℝ) 
  (water_intake_rate : ℝ) 
  (max_water_capacity : ℝ) 
  (rowing_speed : ℝ) 
  (h1 : distance_to_shore = 2) 
  (h2 : water_intake_rate = 15) 
  (h3 : max_water_capacity = 50) 
  (h4 : rowing_speed = 3) : 
  ∃ (min_bailing_rate : ℝ), 
    13 < min_bailing_rate ∧ 
    min_bailing_rate ≤ 14 ∧ 
    (distance_to_shore / rowing_speed) * water_intake_rate - 
      (distance_to_shore / rowing_speed) * min_bailing_rate ≤ max_water_capacity :=
by sorry

end NUMINAMATH_CALUDE_minimum_bailing_rate_l1955_195500


namespace NUMINAMATH_CALUDE_thursday_miles_proof_l1955_195549

def fixed_cost : ℝ := 150
def per_mile_cost : ℝ := 0.50
def monday_miles : ℝ := 620
def total_spent : ℝ := 832

theorem thursday_miles_proof :
  ∃ (thursday_miles : ℝ),
    fixed_cost + per_mile_cost * (monday_miles + thursday_miles) = total_spent ∧
    thursday_miles = 744 := by
  sorry

end NUMINAMATH_CALUDE_thursday_miles_proof_l1955_195549


namespace NUMINAMATH_CALUDE_fraction_simplification_l1955_195504

theorem fraction_simplification : (27 : ℚ) / 25 * 20 / 33 * 55 / 54 = 25 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1955_195504


namespace NUMINAMATH_CALUDE_fraction_product_square_l1955_195574

theorem fraction_product_square : (8 / 9) ^ 2 * (1 / 3) ^ 2 = 64 / 729 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_square_l1955_195574


namespace NUMINAMATH_CALUDE_matthew_hotdogs_l1955_195519

/-- The number of hotdogs each sister wants -/
def sisters_hotdogs : ℕ := 2

/-- The total number of hotdogs both sisters want -/
def total_sisters_hotdogs : ℕ := 2 * sisters_hotdogs

/-- The number of hotdogs Luke wants -/
def luke_hotdogs : ℕ := 2 * total_sisters_hotdogs

/-- The number of hotdogs Hunter wants -/
def hunter_hotdogs : ℕ := (3 * total_sisters_hotdogs) / 2

/-- The total number of hotdogs Matthew needs to cook -/
def total_hotdogs : ℕ := total_sisters_hotdogs + luke_hotdogs + hunter_hotdogs

theorem matthew_hotdogs : total_hotdogs = 18 := by
  sorry

end NUMINAMATH_CALUDE_matthew_hotdogs_l1955_195519


namespace NUMINAMATH_CALUDE_sequence_properties_l1955_195554

def a (n : ℕ) : ℚ := (2/3)^(n-1) * ((2/3)^(n-1) - 1)

theorem sequence_properties :
  (∀ n : ℕ, a n ≤ a 1) ∧
  (∀ n : ℕ, a n ≥ a 3) ∧
  (∀ n : ℕ, n ≥ 3 → a n > a (n+1)) ∧
  (a 1 = 0) ∧
  (a 3 = -20/81) := by
  sorry

end NUMINAMATH_CALUDE_sequence_properties_l1955_195554


namespace NUMINAMATH_CALUDE_daisy_toys_count_l1955_195528

/-- The total number of dog toys Daisy would have if all lost toys were found -/
def total_toys (initial : ℕ) (bought_tuesday : ℕ) (bought_wednesday : ℕ) : ℕ :=
  initial + bought_tuesday + bought_wednesday

/-- Theorem stating the total number of Daisy's toys if all were found -/
theorem daisy_toys_count :
  total_toys 5 3 5 = 13 :=
by sorry

end NUMINAMATH_CALUDE_daisy_toys_count_l1955_195528


namespace NUMINAMATH_CALUDE_tan_negative_1140_degrees_l1955_195508

theorem tan_negative_1140_degrees : Real.tan (-(1140 * π / 180)) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_negative_1140_degrees_l1955_195508


namespace NUMINAMATH_CALUDE_number_difference_l1955_195562

theorem number_difference (a b : ℕ) 
  (sum_eq : a + b = 23540)
  (b_div_16 : b % 16 = 0)
  (b_eq_100a : b = 100 * a) : 
  b - a = 23067 := by sorry

end NUMINAMATH_CALUDE_number_difference_l1955_195562


namespace NUMINAMATH_CALUDE_quadratic_inequality_always_positive_l1955_195582

theorem quadratic_inequality_always_positive (a : ℝ) :
  (∀ x : ℝ, x^2 - 2*x + a > 0) → a > 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_always_positive_l1955_195582


namespace NUMINAMATH_CALUDE_cori_age_relation_cori_current_age_l1955_195583

/-- Cori's current age -/
def cori_age : ℕ := sorry

/-- Cori's aunt's current age -/
def aunt_age : ℕ := 19

/-- In 5 years, Cori will be one-third the age of her aunt -/
theorem cori_age_relation : cori_age + 5 = (aunt_age + 5) / 3 := sorry

theorem cori_current_age : cori_age = 3 := by sorry

end NUMINAMATH_CALUDE_cori_age_relation_cori_current_age_l1955_195583


namespace NUMINAMATH_CALUDE_candy_cost_l1955_195569

/-- The cost of candy given initial amounts and final amount after transaction -/
theorem candy_cost (michael_initial : ℕ) (brother_initial : ℕ) (brother_final : ℕ) 
    (h1 : michael_initial = 42)
    (h2 : brother_initial = 17)
    (h3 : brother_final = 35) :
    michael_initial / 2 + brother_initial - brother_final = 3 :=
by sorry

end NUMINAMATH_CALUDE_candy_cost_l1955_195569


namespace NUMINAMATH_CALUDE_min_sum_xy_l1955_195588

theorem min_sum_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 4*y - x*y = 0) :
  x + y ≥ 9 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 4*y₀ - x₀*y₀ = 0 ∧ x₀ + y₀ = 9 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_xy_l1955_195588


namespace NUMINAMATH_CALUDE_prob_sum_less_than_one_l1955_195512

theorem prob_sum_less_than_one (x y z : ℝ) 
  (hx : 0 < x ∧ x < 1) (hy : 0 < y ∧ y < 1) (hz : 0 < z ∧ z < 1) : 
  x * (1 - y) * (1 - z) + (1 - x) * y * (1 - z) + (1 - x) * (1 - y) * z < 1 := by
  sorry

end NUMINAMATH_CALUDE_prob_sum_less_than_one_l1955_195512


namespace NUMINAMATH_CALUDE_number_of_boys_l1955_195513

/-- Proves that the number of boys is 15 given the problem conditions -/
theorem number_of_boys (men women boys : ℕ) (total_earnings men_wage : ℕ) : 
  (5 * men = women) → 
  (women = boys) → 
  (total_earnings = 180) → 
  (men_wage = 12) → 
  (5 * men * men_wage + women * (total_earnings - 5 * men * men_wage) / (women + boys) + 
   boys * (total_earnings - 5 * men * men_wage) / (women + boys) = total_earnings) →
  boys = 15 := by
sorry

end NUMINAMATH_CALUDE_number_of_boys_l1955_195513


namespace NUMINAMATH_CALUDE_hyperbola_from_ellipse_foci_l1955_195581

/-- Given an ellipse with equation x²/4 + y² = 1, prove that the hyperbola 
    with equation x²/2 - y² = 1 shares the same foci as the ellipse and 
    passes through the point (2,1) -/
theorem hyperbola_from_ellipse_foci (x y : ℝ) : 
  (∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
   (x^2 / (4 : ℝ) + y^2 = 1) ∧ 
   (c^2 = a^2 + b^2) ∧
   (a^2 = 2) ∧ 
   (b^2 = 1) ∧ 
   (c^2 = 3)) →
  (x^2 / (2 : ℝ) - y^2 = 1) ∧ 
  ((2 : ℝ)^2 / (2 : ℝ) - 1^2 = 1) :=
by sorry


end NUMINAMATH_CALUDE_hyperbola_from_ellipse_foci_l1955_195581


namespace NUMINAMATH_CALUDE_probability_non_defective_pencils_l1955_195546

theorem probability_non_defective_pencils :
  let total_pencils : ℕ := 8
  let defective_pencils : ℕ := 2
  let selected_pencils : ℕ := 3
  let non_defective_pencils : ℕ := total_pencils - defective_pencils
  let total_combinations : ℕ := Nat.choose total_pencils selected_pencils
  let non_defective_combinations : ℕ := Nat.choose non_defective_pencils selected_pencils
  (non_defective_combinations : ℚ) / total_combinations = 5 / 14 :=
by sorry

end NUMINAMATH_CALUDE_probability_non_defective_pencils_l1955_195546


namespace NUMINAMATH_CALUDE_original_number_of_people_l1955_195522

theorem original_number_of_people (n : ℕ) : 
  (n / 3 : ℚ) = 18 → n = 54 := by sorry

end NUMINAMATH_CALUDE_original_number_of_people_l1955_195522


namespace NUMINAMATH_CALUDE_different_color_chips_probability_l1955_195521

theorem different_color_chips_probability :
  let total_chips := 6 + 5 + 4 + 3
  let blue_chips := 6
  let red_chips := 5
  let yellow_chips := 4
  let green_chips := 3
  let prob_different_colors := 
    (blue_chips * (total_chips - blue_chips) +
     red_chips * (total_chips - red_chips) +
     yellow_chips * (total_chips - yellow_chips) +
     green_chips * (total_chips - green_chips)) / (total_chips * total_chips)
  prob_different_colors = 119 / 162 := by
  sorry

end NUMINAMATH_CALUDE_different_color_chips_probability_l1955_195521


namespace NUMINAMATH_CALUDE_optimal_gasoline_percentage_l1955_195509

/-- Calculates the optimal gasoline percentage for a car's fuel mixture --/
theorem optimal_gasoline_percentage
  (initial_volume : ℝ)
  (initial_ethanol_percentage : ℝ)
  (initial_gasoline_percentage : ℝ)
  (added_ethanol : ℝ)
  (optimal_ethanol_percentage : ℝ)
  (h1 : initial_volume = 36)
  (h2 : initial_ethanol_percentage = 5)
  (h3 : initial_gasoline_percentage = 95)
  (h4 : added_ethanol = 2)
  (h5 : optimal_ethanol_percentage = 10)
  (h6 : initial_ethanol_percentage + initial_gasoline_percentage = 100) :
  let final_volume := initial_volume + added_ethanol
  let final_ethanol := initial_volume * (initial_ethanol_percentage / 100) + added_ethanol
  let final_ethanol_percentage := (final_ethanol / final_volume) * 100
  100 - optimal_ethanol_percentage = 90 ∧ final_ethanol_percentage = optimal_ethanol_percentage :=
by sorry

end NUMINAMATH_CALUDE_optimal_gasoline_percentage_l1955_195509

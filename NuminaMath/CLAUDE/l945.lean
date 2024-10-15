import Mathlib

namespace NUMINAMATH_CALUDE_no_negative_roots_l945_94519

theorem no_negative_roots (x : ℝ) (h : x < 0) : x^4 - 4*x^3 - 6*x^2 - 3*x + 9 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_negative_roots_l945_94519


namespace NUMINAMATH_CALUDE_power_function_increasing_l945_94570

theorem power_function_increasing (m : ℝ) : 
  (∀ x > 0, Monotone (fun x => (m^2 - 4*m + 1) * x^(m^2 - 2*m - 3))) → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_power_function_increasing_l945_94570


namespace NUMINAMATH_CALUDE_parabola_shift_l945_94584

-- Define the initial parabola
def initial_parabola (x y : ℝ) : Prop :=
  y = -1/3 * (x - 2)^2

-- Define the shift
def shift_right : ℝ := 1
def shift_down : ℝ := 2

-- Define the resulting parabola
def resulting_parabola (x y : ℝ) : Prop :=
  y = -1/3 * (x - 3)^2 - 2

-- Theorem statement
theorem parabola_shift :
  ∀ x y : ℝ, initial_parabola x y →
  resulting_parabola (x + shift_right) (y - shift_down) :=
by sorry

end NUMINAMATH_CALUDE_parabola_shift_l945_94584


namespace NUMINAMATH_CALUDE_loaded_cartons_l945_94550

/-- Given information about cartons of canned juice, prove the number of loaded cartons. -/
theorem loaded_cartons (total_cartons : ℕ) (cans_per_carton : ℕ) (cans_left : ℕ) : 
  total_cartons = 50 →
  cans_per_carton = 20 →
  cans_left = 200 →
  total_cartons - (cans_left / cans_per_carton) = 40 :=
by sorry

end NUMINAMATH_CALUDE_loaded_cartons_l945_94550


namespace NUMINAMATH_CALUDE_max_yellow_apples_max_total_apples_l945_94524

/-- Represents the number of apples of each color in the basket -/
structure Basket :=
  (green : ℕ)
  (yellow : ℕ)
  (red : ℕ)

/-- Represents the number of apples Alyona has taken -/
structure TakenApples :=
  (green : ℕ)
  (yellow : ℕ)
  (red : ℕ)

/-- Checks if Alyona should stop taking apples -/
def shouldStop (taken : TakenApples) : Prop :=
  taken.green < taken.yellow ∧ taken.yellow < taken.red

/-- The initial state of the basket -/
def initialBasket : Basket :=
  { green := 10, yellow := 13, red := 18 }

/-- Theorem stating the maximum number of yellow apples Alyona can take -/
theorem max_yellow_apples :
  ∃ (taken : TakenApples),
    taken.yellow = initialBasket.yellow ∧
    taken.yellow ≤ initialBasket.yellow ∧
    ¬(shouldStop taken) ∧
    ∀ (other : TakenApples),
      other.yellow > taken.yellow →
      shouldStop other ∨ other.yellow > initialBasket.yellow :=
sorry

/-- Theorem stating the maximum total number of apples Alyona can take -/
theorem max_total_apples :
  ∃ (taken : TakenApples),
    taken.green + taken.yellow + taken.red = 39 ∧
    taken.green ≤ initialBasket.green ∧
    taken.yellow ≤ initialBasket.yellow ∧
    taken.red ≤ initialBasket.red ∧
    ¬(shouldStop taken) ∧
    ∀ (other : TakenApples),
      other.green + other.yellow + other.red > 39 →
      shouldStop other ∨
      other.green > initialBasket.green ∨
      other.yellow > initialBasket.yellow ∨
      other.red > initialBasket.red :=
sorry

end NUMINAMATH_CALUDE_max_yellow_apples_max_total_apples_l945_94524


namespace NUMINAMATH_CALUDE_trigonometric_expression_simplification_l945_94581

theorem trigonometric_expression_simplification :
  (Real.tan (20 * π / 180) + Real.tan (30 * π / 180) + Real.tan (60 * π / 180) + Real.tan (70 * π / 180)) / Real.cos (10 * π / 180) = 8 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_simplification_l945_94581


namespace NUMINAMATH_CALUDE_population_increase_in_one_day_l945_94539

/-- Represents the number of seconds in a day -/
def seconds_per_day : ℕ := 24 * 60 * 60

/-- Represents the birth rate (people per 2 seconds) -/
def birth_rate : ℕ := 10

/-- Represents the death rate (people per 2 seconds) -/
def death_rate : ℕ := 2

/-- Calculates the net population increase over one day -/
def net_population_increase : ℕ :=
  (seconds_per_day / 2) * birth_rate - (seconds_per_day / 2) * death_rate

theorem population_increase_in_one_day :
  net_population_increase = 345600 := by sorry

end NUMINAMATH_CALUDE_population_increase_in_one_day_l945_94539


namespace NUMINAMATH_CALUDE_parabola_coefficient_l945_94543

/-- Theorem: For a parabola y = ax² + bx + c passing through points (4, 0), (t/3, 0), and (0, 60), the value of a is 45/t. -/
theorem parabola_coefficient (a b c t : ℝ) : 
  (∀ x, a * x^2 + b * x + c = 0 ↔ (x = 4 ∨ x = t/3)) → 
  a * 0^2 + b * 0 + c = 60 →
  a = 45 / t :=
by sorry

end NUMINAMATH_CALUDE_parabola_coefficient_l945_94543


namespace NUMINAMATH_CALUDE_smallest_non_factor_product_of_48_l945_94595

def is_factor (a b : ℕ) : Prop := b % a = 0

theorem smallest_non_factor_product_of_48 (x y : ℕ) :
  x ≠ y →
  x > 0 →
  y > 0 →
  is_factor x 48 →
  is_factor y 48 →
  ¬ is_factor (x * y) 48 →
  ∀ a b : ℕ, a ≠ b → a > 0 → b > 0 → is_factor a 48 → is_factor b 48 → ¬ is_factor (a * b) 48 → x * y ≤ a * b →
  x * y = 18 :=
sorry

end NUMINAMATH_CALUDE_smallest_non_factor_product_of_48_l945_94595


namespace NUMINAMATH_CALUDE_large_bucket_capacity_l945_94577

theorem large_bucket_capacity (small_bucket : ℝ) (large_bucket : ℝ) : 
  (large_bucket = 2 * small_bucket + 3) →
  (2 * small_bucket + 5 * large_bucket = 63) →
  large_bucket = 11 := by
sorry

end NUMINAMATH_CALUDE_large_bucket_capacity_l945_94577


namespace NUMINAMATH_CALUDE_distinct_prime_factors_of_divisor_sum_800_l945_94514

-- Define the sum of positive divisors function
def sum_of_divisors (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem distinct_prime_factors_of_divisor_sum_800 :
  (Nat.factors (sum_of_divisors 800)).length = 4 := by sorry

end NUMINAMATH_CALUDE_distinct_prime_factors_of_divisor_sum_800_l945_94514


namespace NUMINAMATH_CALUDE_radio_price_rank_l945_94590

theorem radio_price_rank (n : ℕ) (prices : Finset ℕ) (radio_price : ℕ) :
  n = 58 →
  prices.card = n + 1 →
  radio_price ∈ prices →
  (∀ p ∈ prices, p ≤ radio_price) →
  (prices.filter (λ p => p < radio_price)).card = 41 →
  (prices.filter (λ p => p ≤ radio_price)).card = n + 1 :=
by
  sorry

end NUMINAMATH_CALUDE_radio_price_rank_l945_94590


namespace NUMINAMATH_CALUDE_price_decrease_l945_94515

theorem price_decrease (original_price : ℝ) (decreased_price : ℝ) : 
  decreased_price = original_price * (1 - 0.24) ∧ decreased_price = 608 → 
  original_price = 800 := by
sorry

end NUMINAMATH_CALUDE_price_decrease_l945_94515


namespace NUMINAMATH_CALUDE_tangent_line_passes_through_fixed_point_l945_94510

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the line l
def line_l (x : ℝ) : Prop := x = 2

-- Define a point P on line l
def point_P (t : ℝ) : ℝ × ℝ := (2, t)

-- Define the equation of the common chord AB
def common_chord (t x y : ℝ) : Prop := 2*x + t*y = 1

-- Theorem statement
theorem tangent_line_passes_through_fixed_point :
  ∀ t : ℝ, ∃ A B : ℝ × ℝ,
    circle_C A.1 A.2 ∧
    circle_C B.1 B.2 ∧
    common_chord t A.1 A.2 ∧
    common_chord t B.1 B.2 ∧
    common_chord t (1/2) 0 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_passes_through_fixed_point_l945_94510


namespace NUMINAMATH_CALUDE_product_of_roots_equation_l945_94571

theorem product_of_roots_equation (y : ℝ) (h1 : y > 0) 
  (h2 : Real.sqrt (5 * y) * Real.sqrt (15 * y) * Real.sqrt (2 * y) * Real.sqrt (6 * y) = 6) : 
  y = 1 / Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_product_of_roots_equation_l945_94571


namespace NUMINAMATH_CALUDE_jerry_max_throws_l945_94500

/-- Represents the point system in Mrs. Carlton's class -/
structure PointSystem where
  interrupt_points : ℕ
  insult_points : ℕ
  throw_points : ℕ
  office_threshold : ℕ

/-- Represents Jerry's current misbehavior record -/
structure MisbehaviorRecord where
  interrupts : ℕ
  insults : ℕ

/-- Calculates the maximum number of times Jerry can throw things before reaching the office threshold -/
def max_throws (ps : PointSystem) (record : MisbehaviorRecord) : ℕ :=
  let current_points := record.interrupts * ps.interrupt_points + record.insults * ps.insult_points
  let remaining_points := ps.office_threshold - current_points
  remaining_points / ps.throw_points

/-- Theorem stating that Jerry can throw things twice before being sent to the office -/
theorem jerry_max_throws :
  let ps : PointSystem := {
    interrupt_points := 5,
    insult_points := 10,
    throw_points := 25,
    office_threshold := 100
  }
  let record : MisbehaviorRecord := {
    interrupts := 2,
    insults := 4
  }
  max_throws ps record = 2 := by
  sorry

end NUMINAMATH_CALUDE_jerry_max_throws_l945_94500


namespace NUMINAMATH_CALUDE_cos_sum_seventh_roots_l945_94587

theorem cos_sum_seventh_roots : Real.cos (2 * Real.pi / 7) + Real.cos (4 * Real.pi / 7) + Real.cos (8 * Real.pi / 7) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_sum_seventh_roots_l945_94587


namespace NUMINAMATH_CALUDE_inequality_conversions_l945_94540

theorem inequality_conversions (x : ℝ) : 
  ((5 * x > 4 * x - 1) ↔ (x > -1)) ∧ 
  ((-x - 2 < 7) ↔ (x > -9)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_conversions_l945_94540


namespace NUMINAMATH_CALUDE_carls_open_house_l945_94575

/-- Carl's open house problem -/
theorem carls_open_house 
  (definite_attendees : ℕ) 
  (potential_attendees : ℕ)
  (extravagant_bags : ℕ)
  (average_bags : ℕ)
  (h1 : definite_attendees = 50)
  (h2 : potential_attendees = 40)
  (h3 : extravagant_bags = 10)
  (h4 : average_bags = 20) :
  definite_attendees + potential_attendees - (extravagant_bags + average_bags) = 60 :=
by sorry

end NUMINAMATH_CALUDE_carls_open_house_l945_94575


namespace NUMINAMATH_CALUDE_girls_average_weight_is_27_l945_94547

/-- Given a class with boys and girls, calculates the average weight of girls -/
def average_weight_of_girls (total_students : ℕ) (num_boys : ℕ) (boys_avg_weight : ℚ) (class_avg_weight : ℚ) : ℚ :=
  let total_weight := class_avg_weight * total_students
  let boys_total_weight := boys_avg_weight * num_boys
  let girls_total_weight := total_weight - boys_total_weight
  let num_girls := total_students - num_boys
  girls_total_weight / num_girls

/-- Theorem stating that the average weight of girls is 27 kgs given the problem conditions -/
theorem girls_average_weight_is_27 : 
  average_weight_of_girls 25 15 48 45 = 27 := by
  sorry

end NUMINAMATH_CALUDE_girls_average_weight_is_27_l945_94547


namespace NUMINAMATH_CALUDE_equation_solution_l945_94599

theorem equation_solution : 
  ∃ (x₁ x₂ : ℝ), x₁ = 1 ∧ x₂ = (3/2 : ℝ) ∧ 
  (∀ x : ℝ, 2 * (x - 1)^2 = x - 1 ↔ (x = x₁ ∨ x = x₂)) := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l945_94599


namespace NUMINAMATH_CALUDE_systematic_sampling_distribution_l945_94594

/-- Represents a building in the school -/
inductive Building
| A
| B
| C

/-- Represents the systematic sampling method -/
def systematicSampling (total : ℕ) (sampleSize : ℕ) (start : ℕ) : List ℕ :=
  let interval := total / sampleSize
  List.range (total - start + 1)
    |> List.filter (fun i => (i + start - 1) % interval == 0)
    |> List.map (fun i => i + start - 1)

/-- Assigns a student number to a building -/
def assignBuilding (studentNumber : ℕ) : Building :=
  if studentNumber ≤ 200 then Building.A
  else if studentNumber ≤ 295 then Building.B
  else Building.C

/-- Counts the number of students selected for each building -/
def countSelectedStudents (selectedStudents : List ℕ) : ℕ × ℕ × ℕ :=
  selectedStudents.foldl
    (fun (a, b, c) student =>
      match assignBuilding student with
      | Building.A => (a + 1, b, c)
      | Building.B => (a, b + 1, c)
      | Building.C => (a, b, c + 1))
    (0, 0, 0)

theorem systematic_sampling_distribution :
  let totalStudents := 400
  let sampleSize := 50
  let firstNumber := 3
  let selectedStudents := systematicSampling totalStudents sampleSize firstNumber
  let (buildingA, buildingB, buildingC) := countSelectedStudents selectedStudents
  buildingA = 25 ∧ buildingB = 12 ∧ buildingC = 13 := by
  sorry


end NUMINAMATH_CALUDE_systematic_sampling_distribution_l945_94594


namespace NUMINAMATH_CALUDE_people_per_car_l945_94526

theorem people_per_car (total_people : ℕ) (num_cars : ℕ) (h1 : total_people = 63) (h2 : num_cars = 3) :
  total_people / num_cars = 21 :=
by sorry

end NUMINAMATH_CALUDE_people_per_car_l945_94526


namespace NUMINAMATH_CALUDE_work_completion_time_l945_94530

/-- Proves that given two people who can complete a task in 4 days together, 
    and one of them can complete it in 12 days alone, 
    the other person can complete the task in 24 days alone. -/
theorem work_completion_time 
  (joint_time : ℝ) 
  (person1_time : ℝ) 
  (h1 : joint_time = 4) 
  (h2 : person1_time = 12) : 
  ∃ person2_time : ℝ, 
    person2_time = 24 ∧ 
    1 / joint_time = 1 / person1_time + 1 / person2_time :=
by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l945_94530


namespace NUMINAMATH_CALUDE_complex_equation_solution_l945_94511

theorem complex_equation_solution (a : ℝ) : 
  (Complex.mk 2 a) * (Complex.mk a (-2)) = Complex.I * (-4) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l945_94511


namespace NUMINAMATH_CALUDE_unique_number_satisfying_conditions_l945_94580

def contains_digit (n : ℕ) (d : ℕ) : Prop :=
  ∃ k m : ℕ, n = 10 * k + d + 10 * m

theorem unique_number_satisfying_conditions :
  ∃! n : ℕ,
    Odd n ∧
    contains_digit n 5 ∧
    3 ∣ n ∧
    12^2 < n ∧
    n < 13^2 :=
by sorry

end NUMINAMATH_CALUDE_unique_number_satisfying_conditions_l945_94580


namespace NUMINAMATH_CALUDE_cricket_runs_total_l945_94561

theorem cricket_runs_total (a b c : ℕ) : 
  (a : ℚ) / b = 1 / 3 →
  (b : ℚ) / c = 1 / 5 →
  c = 75 →
  a + b + c = 95 := by
  sorry

end NUMINAMATH_CALUDE_cricket_runs_total_l945_94561


namespace NUMINAMATH_CALUDE_decimal_difference_l945_94589

/- Define the repeating decimal 0.̅72 -/
def repeating_decimal : ℚ := 72 / 99

/- Define the terminating decimal 0.72 -/
def terminating_decimal : ℚ := 72 / 100

/- Theorem statement -/
theorem decimal_difference :
  repeating_decimal - terminating_decimal = 2 / 275 := by
  sorry

end NUMINAMATH_CALUDE_decimal_difference_l945_94589


namespace NUMINAMATH_CALUDE_geometric_series_problem_l945_94517

theorem geometric_series_problem (a r : ℝ) (h1 : r ≠ 1) (h2 : r > 0) : 
  (a / (1 - r) = 15) → (a / (1 - r^4) = 9) → r = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_problem_l945_94517


namespace NUMINAMATH_CALUDE_cloth_coloring_problem_l945_94538

/-- Represents the work done by a group of men coloring cloth -/
structure ClothColoring where
  men : ℕ
  days : ℝ
  length : ℝ

/-- The problem statement -/
theorem cloth_coloring_problem (group1 group2 : ClothColoring) :
  group1.men = 4 ∧
  group1.days = 2 ∧
  group2.men = 5 ∧
  group2.days = 1.2 ∧
  group2.length = 36 ∧
  group1.men * group1.days * group1.length = group2.men * group2.days * group2.length →
  group1.length = 27 := by
  sorry

end NUMINAMATH_CALUDE_cloth_coloring_problem_l945_94538


namespace NUMINAMATH_CALUDE_direct_variation_problem_l945_94552

theorem direct_variation_problem (k : ℝ) :
  (∀ x y : ℝ, 5 * y = k * x^2) →
  (5 * 8 = k * 2^2) →
  (5 * 32 = k * 4^2) :=
by
  sorry

end NUMINAMATH_CALUDE_direct_variation_problem_l945_94552


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_2_range_of_a_when_f_geq_4_l945_94574

-- Define the function f
def f (x a : ℝ) : ℝ := |x - a^2| + |x - 2*a + 1|

-- Part 1
theorem solution_set_when_a_is_2 :
  {x : ℝ | f x 2 ≥ 4} = {x : ℝ | x ≤ 3/2 ∨ x ≥ 11/2} := by sorry

-- Part 2
theorem range_of_a_when_f_geq_4 :
  ∀ x a : ℝ, f x a ≥ 4 → a ≤ -1 ∨ a ≥ 3 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_2_range_of_a_when_f_geq_4_l945_94574


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a20_l945_94582

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a20 (a : ℕ → ℝ) :
  ArithmeticSequence a →
  a 1 + a 3 + a 5 = 105 →
  a 2 + a 4 + a 6 = 99 →
  a 20 = 1 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a20_l945_94582


namespace NUMINAMATH_CALUDE_radio_loss_percentage_l945_94502

/-- Given the cost price and selling price of a radio, prove the loss percentage. -/
theorem radio_loss_percentage
  (cost_price : ℝ)
  (selling_price : ℝ)
  (h1 : cost_price = 2400)
  (h2 : selling_price = 2100) :
  (cost_price - selling_price) / cost_price * 100 = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_radio_loss_percentage_l945_94502


namespace NUMINAMATH_CALUDE_gcd_power_two_l945_94592

theorem gcd_power_two : 
  Nat.gcd (2^2100 - 1) (2^2091 + 31) = Nat.gcd (2^2091 + 31) 511 := by
  sorry

end NUMINAMATH_CALUDE_gcd_power_two_l945_94592


namespace NUMINAMATH_CALUDE_quadratic_roots_imply_c_value_l945_94512

theorem quadratic_roots_imply_c_value (c : ℝ) :
  (∀ x : ℝ, x^2 + 5*x + c = 0 ↔ x = (-5 + Real.sqrt c) / 2 ∨ x = (-5 - Real.sqrt c) / 2) →
  c = 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_imply_c_value_l945_94512


namespace NUMINAMATH_CALUDE_nine_candies_four_bags_l945_94535

/-- The number of ways to distribute distinct candies among bags --/
def distribute_candies (num_candies : ℕ) (num_bags : ℕ) : ℕ :=
  num_bags ^ (num_candies - num_bags)

/-- Theorem stating the number of ways to distribute 9 distinct candies among 4 bags --/
theorem nine_candies_four_bags : 
  distribute_candies 9 4 = 1024 :=
sorry

end NUMINAMATH_CALUDE_nine_candies_four_bags_l945_94535


namespace NUMINAMATH_CALUDE_max_projection_area_specific_tetrahedron_l945_94533

/-- Represents a tetrahedron with two adjacent equilateral triangular faces --/
structure Tetrahedron where
  side_length : ℝ
  dihedral_angle : ℝ

/-- Calculates the maximum projection area of a rotating tetrahedron --/
def max_projection_area (t : Tetrahedron) : ℝ :=
  sorry

/-- The theorem stating the maximum projection area of the specific tetrahedron --/
theorem max_projection_area_specific_tetrahedron :
  let t : Tetrahedron := { side_length := 1, dihedral_angle := π / 3 }
  max_projection_area t = Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_max_projection_area_specific_tetrahedron_l945_94533


namespace NUMINAMATH_CALUDE_same_heads_probability_l945_94518

/-- Represents the outcome of a coin toss -/
inductive CoinToss
| Heads
| Tails

/-- Represents the result of tossing two coins -/
def TwoCoins := (CoinToss × CoinToss)

/-- The sample space of all possible outcomes when two people each toss two coins -/
def SampleSpace := (TwoCoins × TwoCoins)

/-- Counts the number of heads in a two-coin toss -/
def countHeads : TwoCoins → Nat
| (CoinToss.Heads, CoinToss.Heads) => 2
| (CoinToss.Heads, CoinToss.Tails) => 1
| (CoinToss.Tails, CoinToss.Heads) => 1
| (CoinToss.Tails, CoinToss.Tails) => 0

/-- Checks if two two-coin tosses have the same number of heads -/
def sameHeads (t1 t2 : TwoCoins) : Bool :=
  countHeads t1 = countHeads t2

/-- The number of elements in the sample space -/
def totalOutcomes : Nat := 16

/-- The number of favorable outcomes (same number of heads) -/
def favorableOutcomes : Nat := 6

/-- The probability of getting the same number of heads -/
def probability : Rat := favorableOutcomes / totalOutcomes

theorem same_heads_probability : probability = 3 / 8 := by
  sorry


end NUMINAMATH_CALUDE_same_heads_probability_l945_94518


namespace NUMINAMATH_CALUDE_baron_weights_partition_l945_94501

/-- A set of weights satisfying the Baron's conditions -/
def BaronWeights : Type := 
  { s : Finset ℕ // s.card = 50 ∧ ∀ x ∈ s, x ≤ 100 ∧ Even (s.sum id) }

/-- The proposition that the weights can be partitioned into two subsets with equal sums -/
def CanPartition (weights : BaronWeights) : Prop :=
  ∃ (s₁ s₂ : Finset ℕ), s₁ ∪ s₂ = weights.val ∧ s₁ ∩ s₂ = ∅ ∧ s₁.sum id = s₂.sum id

/-- The theorem stating that any set of weights satisfying the Baron's conditions can be partitioned -/
theorem baron_weights_partition (weights : BaronWeights) : CanPartition weights := by
  sorry


end NUMINAMATH_CALUDE_baron_weights_partition_l945_94501


namespace NUMINAMATH_CALUDE_complementary_angle_adjustment_l945_94528

/-- Two angles are complementary if their sum is 90 degrees -/
def complementary (a b : ℝ) : Prop := a + b = 90

/-- The ratio of two real numbers is 4:5 -/
def ratio_4_to_5 (a b : ℝ) : Prop := 5 * a = 4 * b

theorem complementary_angle_adjustment (a b : ℝ) 
  (h1 : complementary a b) 
  (h2 : ratio_4_to_5 a b) :
  complementary (1.1 * a) (0.92 * b) := by
  sorry

end NUMINAMATH_CALUDE_complementary_angle_adjustment_l945_94528


namespace NUMINAMATH_CALUDE_intersection_A_B_l945_94568

def A : Set ℝ := {x | x * (x - 2) < 0}
def B : Set ℝ := {-1, 0, 1, 2}

theorem intersection_A_B : A ∩ B = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l945_94568


namespace NUMINAMATH_CALUDE_girls_multiple_of_five_l945_94521

/-- Represents the number of students in a group -/
structure GroupComposition :=
  (boys : ℕ)
  (girls : ℕ)

/-- Checks if a given number of boys and girls can be divided into the specified number of groups -/
def canDivideIntoGroups (totalBoys totalGirls groups : ℕ) : Prop :=
  ∃ (composition : GroupComposition),
    composition.boys * groups = totalBoys ∧
    composition.girls * groups = totalGirls

theorem girls_multiple_of_five (totalBoys totalGirls : ℕ) :
  totalBoys = 10 →
  canDivideIntoGroups totalBoys totalGirls 5 →
  ∃ (k : ℕ), totalGirls = 5 * k ∧ k ≥ 1 :=
by sorry

end NUMINAMATH_CALUDE_girls_multiple_of_five_l945_94521


namespace NUMINAMATH_CALUDE_expression_value_l945_94554

theorem expression_value (x y : ℝ) (hx : x = 3) (hy : y = 2) : 3 * x - 4 * y = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l945_94554


namespace NUMINAMATH_CALUDE_power_division_eight_sixtyfour_l945_94548

theorem power_division_eight_sixtyfour : 8^15 / 64^7 = 8 := by
  sorry

end NUMINAMATH_CALUDE_power_division_eight_sixtyfour_l945_94548


namespace NUMINAMATH_CALUDE_sarah_reading_time_l945_94573

/-- Calculates the reading time in hours for a given number of books -/
def reading_time (words_per_minute : ℕ) (words_per_page : ℕ) (pages_per_book : ℕ) (num_books : ℕ) : ℕ :=
  let total_words := words_per_page * pages_per_book * num_books
  let total_minutes := total_words / words_per_minute
  total_minutes / 60

/-- Theorem stating that Sarah's reading time for 6 books is 20 hours -/
theorem sarah_reading_time :
  reading_time 40 100 80 6 = 20 := by
  sorry

end NUMINAMATH_CALUDE_sarah_reading_time_l945_94573


namespace NUMINAMATH_CALUDE_dexter_card_count_l945_94525

/-- The number of boxes filled with basketball cards -/
def basketball_boxes : ℕ := 9

/-- The number of cards in each basketball box -/
def cards_per_basketball_box : ℕ := 15

/-- The number of cards in each football box -/
def cards_per_football_box : ℕ := 20

/-- The difference in number of boxes between basketball and football cards -/
def box_difference : ℕ := 3

/-- The total number of cards Dexter has -/
def total_cards : ℕ := 
  (basketball_boxes * cards_per_basketball_box) + 
  ((basketball_boxes - box_difference) * cards_per_football_box)

theorem dexter_card_count : total_cards = 255 := by
  sorry

end NUMINAMATH_CALUDE_dexter_card_count_l945_94525


namespace NUMINAMATH_CALUDE_path_1310_to_1315_l945_94569

/-- Represents a point in the cyclic path --/
def CyclicPoint := ℕ

/-- The length of one cycle in the path --/
def cycleLength : ℕ := 6

/-- Converts a given point to its equivalent position within a cycle --/
def toCyclicPosition (n : ℕ) : CyclicPoint :=
  n % cycleLength

/-- Checks if two points are equivalent in the cyclic representation --/
def areEquivalentPoints (a b : ℕ) : Prop :=
  toCyclicPosition a = toCyclicPosition b

theorem path_1310_to_1315 :
  areEquivalentPoints 1310 2 ∧ 
  areEquivalentPoints 1315 3 ∧
  (1315 - 1310 = cycleLength + 3) := by
  sorry

#check path_1310_to_1315

end NUMINAMATH_CALUDE_path_1310_to_1315_l945_94569


namespace NUMINAMATH_CALUDE_tire_rotation_mileage_l945_94534

theorem tire_rotation_mileage (total_tires : ℕ) (simultaneous_tires : ℕ) (total_miles : ℕ) :
  total_tires = 7 →
  simultaneous_tires = 6 →
  total_miles = 42000 →
  (total_miles * simultaneous_tires) / total_tires = 36000 :=
by sorry

end NUMINAMATH_CALUDE_tire_rotation_mileage_l945_94534


namespace NUMINAMATH_CALUDE_allison_wins_probability_l945_94529

def allison_cube : Fin 6 → ℕ := λ _ => 6

def brian_cube : Fin 6 → ℕ := λ i => i.val + 1

def noah_cube : Fin 6 → ℕ := λ i => if i.val < 3 then 3 else 5

def prob_brian_less_than_6 : ℚ := 5 / 6

def prob_noah_less_than_6 : ℚ := 1

theorem allison_wins_probability :
  (prob_brian_less_than_6 * prob_noah_less_than_6 : ℚ) = 5 / 6 := by sorry

end NUMINAMATH_CALUDE_allison_wins_probability_l945_94529


namespace NUMINAMATH_CALUDE_least_n_factorial_divisible_by_9450_l945_94596

def is_factor (a b : ℕ) : Prop := b % a = 0

theorem least_n_factorial_divisible_by_9450 :
  ∃ (n : ℕ), n > 0 ∧ is_factor 9450 (Nat.factorial n) ∧
  ∀ (m : ℕ), m > 0 ∧ m < n → ¬is_factor 9450 (Nat.factorial m) :=
by
  use 10
  sorry

end NUMINAMATH_CALUDE_least_n_factorial_divisible_by_9450_l945_94596


namespace NUMINAMATH_CALUDE_distance_between_AB_l945_94541

/-- The distance between two points A and B, where two motorcyclists meet twice -/
def distance_AB : ℝ := 125

/-- The distance of the first meeting point from B -/
def distance_first_meeting : ℝ := 50

/-- The distance of the second meeting point from A -/
def distance_second_meeting : ℝ := 25

/-- Theorem stating that the distance between A and B is 125 km -/
theorem distance_between_AB : 
  distance_AB = distance_first_meeting + distance_second_meeting :=
by sorry

end NUMINAMATH_CALUDE_distance_between_AB_l945_94541


namespace NUMINAMATH_CALUDE_additive_function_characterization_l945_94563

def is_additive (f : ℚ → ℚ) : Prop :=
  ∀ x y : ℚ, f (x + y) = f x + f y

theorem additive_function_characterization (f : ℚ → ℚ) (h : is_additive f) :
  ∃ k : ℚ, ∀ x : ℚ, f x = k * x := by
  sorry

end NUMINAMATH_CALUDE_additive_function_characterization_l945_94563


namespace NUMINAMATH_CALUDE_sum_of_non_visible_numbers_l945_94509

/-- Represents a standard six-sided die -/
def StandardDie : Type := Fin 6

/-- The sum of numbers on a standard six-sided die -/
def sumOfDie : ℕ := 21

/-- The total number of faces on four dice -/
def totalFaces : ℕ := 24

/-- The number of visible faces -/
def visibleFaces : ℕ := 9

/-- The list of visible numbers -/
def visibleNumbers : List ℕ := [1, 2, 3, 3, 4, 5, 5, 6, 6]

/-- The theorem stating the sum of non-visible numbers -/
theorem sum_of_non_visible_numbers :
  (4 * sumOfDie) - (visibleNumbers.sum) = 49 := by sorry

end NUMINAMATH_CALUDE_sum_of_non_visible_numbers_l945_94509


namespace NUMINAMATH_CALUDE_coprime_sequence_solution_l945_94551

/-- Represents the sequence of ones and twos constructed from multiples of a, b, and c -/
def constructSequence (a b c : ℕ) : List ℕ := sorry

/-- Checks if two natural numbers are coprime -/
def areCoprime (x y : ℕ) : Prop := Nat.gcd x y = 1

theorem coprime_sequence_solution :
  ∀ a b c : ℕ,
    a > 0 ∧ b > 0 ∧ c > 0 →
    areCoprime a b ∧ areCoprime b c ∧ areCoprime a c →
    (let seq := constructSequence a b c
     seq.count 1 = 356 ∧ 
     seq.count 2 = 36 ∧
     seq.take 16 = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2]) →
    a = 7 ∧ b = 9 ∧ c = 23 := by
  sorry

end NUMINAMATH_CALUDE_coprime_sequence_solution_l945_94551


namespace NUMINAMATH_CALUDE_gas_mixture_ratio_l945_94553

-- Define the gases and elements
inductive Gas : Type
| A : Gas  -- CO2
| B : Gas  -- O2

inductive Element : Type
| C : Element
| O : Element

-- Define the molar mass function
def molarMass : Gas → ℝ
| Gas.A => 44  -- Molar mass of CO2
| Gas.B => 32  -- Molar mass of O2

-- Define the number of atoms of each element in each gas molecule
def atomCount : Gas → Element → ℕ
| Gas.A, Element.C => 1
| Gas.A, Element.O => 2
| Gas.B, Element.C => 0
| Gas.B, Element.O => 2

-- Define the mass ratio of C to O in the mixed gas
def massRatio (x y : ℝ) : Prop :=
  (12 * x) / (16 * (2 * x + 2 * y)) = 1 / 8

-- Define the volume ratio of A to B
def volumeRatio (x y : ℝ) : Prop :=
  x / y = 1 / 2

-- The theorem to prove
theorem gas_mixture_ratio : 
  ∀ (x y : ℝ), x > 0 → y > 0 → massRatio x y → volumeRatio x y :=
sorry

end NUMINAMATH_CALUDE_gas_mixture_ratio_l945_94553


namespace NUMINAMATH_CALUDE_m_range_l945_94559

-- Define the propositions p and q
def p (x : ℝ) : Prop := x ≤ 2

def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

-- Define the condition that m is positive
def m_positive (m : ℝ) : Prop := m > 0

-- Define the condition about the relationship between p and q
def condition (m : ℝ) : Prop := 
  ∀ x, (¬(p x) → ¬(q x m)) ∧ ∃ x, (¬(p x) ∧ q x m)

-- State the theorem
theorem m_range (m : ℝ) : 
  m_positive m → condition m → m ≥ 9 :=
sorry

end NUMINAMATH_CALUDE_m_range_l945_94559


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l945_94565

theorem cube_root_equation_solution :
  ∃ x : ℝ, (x ≠ 0 ∧ (5 - 2/x)^(1/3) = -3) → x = 1/16 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l945_94565


namespace NUMINAMATH_CALUDE_chair_price_l945_94544

theorem chair_price (total_cost : ℕ) (num_desks : ℕ) (num_chairs : ℕ) (desk_price : ℕ) :
  total_cost = 1236 →
  num_desks = 5 →
  num_chairs = 8 →
  desk_price = 180 →
  (total_cost - num_desks * desk_price) / num_chairs = 42 := by
  sorry

end NUMINAMATH_CALUDE_chair_price_l945_94544


namespace NUMINAMATH_CALUDE_student_arrangement_theorem_l945_94508

/-- Represents the number of students in the group -/
def total_students : ℕ := 9

/-- Represents the probability of selecting at least one girl when choosing 3 students -/
def prob_at_least_one_girl : ℚ := 16/21

/-- Calculates the number of ways to select k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

/-- Calculates the number of permutations of k items from n items -/
def permute (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial (n - k))

/-- Theorem stating the main result -/
theorem student_arrangement_theorem :
  ∃ (male_count female_count : ℕ),
    male_count + female_count = total_students ∧
    (choose total_students 3 - choose male_count 3) / (choose total_students 3) = prob_at_least_one_girl ∧
    male_count = 6 ∧
    female_count = 3 ∧
    (choose male_count 2 * choose (male_count - 2) 2 * choose (male_count - 4) 2) / (permute 3 3) *
    (permute female_count female_count) * (permute (female_count + 1) 3) *
    ((permute 2 2) ^ 3) = 17280 :=
by sorry

end NUMINAMATH_CALUDE_student_arrangement_theorem_l945_94508


namespace NUMINAMATH_CALUDE_roots_modulus_one_preserved_l945_94506

theorem roots_modulus_one_preserved (a b c : ℂ) :
  (∀ z : ℂ, z^3 + a*z^2 + b*z + c = 0 → Complex.abs z = 1) →
  (∀ w : ℂ, w^3 + Complex.abs a * w^2 + Complex.abs b * w + Complex.abs c = 0 → Complex.abs w = 1) :=
by sorry

end NUMINAMATH_CALUDE_roots_modulus_one_preserved_l945_94506


namespace NUMINAMATH_CALUDE_original_denominator_problem_l945_94549

theorem original_denominator_problem (d : ℝ) : 
  (3 : ℝ) / d ≠ 0 →
  (3 + 3) / (d + 3) = (1 : ℝ) / 3 →
  d = 15 := by
  sorry

end NUMINAMATH_CALUDE_original_denominator_problem_l945_94549


namespace NUMINAMATH_CALUDE_sixth_grade_total_l945_94560

theorem sixth_grade_total (girls boys : ℕ) : 
  girls = boys + 2 →
  girls / 11 + 22 = (girls - girls / 11) / 2 + 22 →
  girls + boys = 86 :=
by sorry

end NUMINAMATH_CALUDE_sixth_grade_total_l945_94560


namespace NUMINAMATH_CALUDE_power_mod_seventeen_l945_94536

theorem power_mod_seventeen : 5^2023 ≡ 11 [ZMOD 17] := by sorry

end NUMINAMATH_CALUDE_power_mod_seventeen_l945_94536


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l945_94503

theorem consecutive_integers_sum (x : ℕ) (h1 : x > 0) (h2 : x * (x + 1) = 506) : 
  x + (x + 1) = 45 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l945_94503


namespace NUMINAMATH_CALUDE_certain_number_value_l945_94597

theorem certain_number_value : 
  ∀ x : ℝ,
  (28 + x + 42 + 78 + 104) / 5 = 62 →
  ∃ y : ℝ,
  (y + 62 + 98 + 124 + x) / 5 = 78 ∧
  y = 106 :=
by
  sorry

end NUMINAMATH_CALUDE_certain_number_value_l945_94597


namespace NUMINAMATH_CALUDE_power_twelve_half_l945_94567

theorem power_twelve_half : (12 : ℕ) ^ ((12 : ℕ) / 2) = 2985984 := by sorry

end NUMINAMATH_CALUDE_power_twelve_half_l945_94567


namespace NUMINAMATH_CALUDE_unknown_bill_value_is_five_l945_94545

/-- Represents the value of a US dollar bill -/
inductive USBill
| One
| Two
| Five
| Ten
| Twenty
| Fifty
| Hundred

/-- The wallet contents before purchase -/
structure Wallet where
  twenties : Nat
  unknown_bills : Nat
  unknown_bill_value : USBill
  loose_coins : Rat

def Wallet.total_value (w : Wallet) : Rat :=
  20 * w.twenties + 
  (match w.unknown_bill_value with
   | USBill.One => 1
   | USBill.Two => 2
   | USBill.Five => 5
   | USBill.Ten => 10
   | USBill.Twenty => 20
   | USBill.Fifty => 50
   | USBill.Hundred => 100) * w.unknown_bills +
  w.loose_coins

theorem unknown_bill_value_is_five (w : Wallet) (h1 : w.twenties = 2) 
  (h2 : w.loose_coins = 9/2) (h3 : Wallet.total_value w - 35/2 = 42) :
  w.unknown_bill_value = USBill.Five := by
  sorry

end NUMINAMATH_CALUDE_unknown_bill_value_is_five_l945_94545


namespace NUMINAMATH_CALUDE_value_of_k_l945_94579

theorem value_of_k (a b k : ℝ) (h1 : 2 * a = k) (h2 : 3 * b = k) (h3 : k ≠ 1) (h4 : 2 * a + b = a * b) : k = 18 := by
  sorry

end NUMINAMATH_CALUDE_value_of_k_l945_94579


namespace NUMINAMATH_CALUDE_second_year_probability_l945_94542

/-- Represents the academic year of a student -/
inductive AcademicYear
| FirstYear
| SecondYear
| ThirdYear
| Postgraduate

/-- Represents the department of a student -/
inductive Department
| Science
| Arts
| Engineering

/-- Represents the number of students in each academic year and department -/
def studentCount : AcademicYear → Department → ℕ
| AcademicYear.FirstYear, Department.Science => 300
| AcademicYear.FirstYear, Department.Arts => 200
| AcademicYear.FirstYear, Department.Engineering => 100
| AcademicYear.SecondYear, Department.Science => 250
| AcademicYear.SecondYear, Department.Arts => 150
| AcademicYear.SecondYear, Department.Engineering => 50
| AcademicYear.ThirdYear, Department.Science => 300
| AcademicYear.ThirdYear, Department.Arts => 200
| AcademicYear.ThirdYear, Department.Engineering => 50
| AcademicYear.Postgraduate, Department.Science => 200
| AcademicYear.Postgraduate, Department.Arts => 100
| AcademicYear.Postgraduate, Department.Engineering => 100

/-- The total number of students in the sample -/
def totalStudents : ℕ := 2000

/-- Theorem: The probability of selecting a second-year student from the group of students
    who are not third-year and not in the Science department is 2/7 -/
theorem second_year_probability :
  let nonThirdYearNonScience := (studentCount AcademicYear.FirstYear Department.Arts
                                + studentCount AcademicYear.FirstYear Department.Engineering
                                + studentCount AcademicYear.SecondYear Department.Arts
                                + studentCount AcademicYear.SecondYear Department.Engineering
                                + studentCount AcademicYear.Postgraduate Department.Arts
                                + studentCount AcademicYear.Postgraduate Department.Engineering)
  let secondYearNonScience := (studentCount AcademicYear.SecondYear Department.Arts
                              + studentCount AcademicYear.SecondYear Department.Engineering)
  (secondYearNonScience : ℚ) / nonThirdYearNonScience = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_second_year_probability_l945_94542


namespace NUMINAMATH_CALUDE_exists_invariant_point_l945_94586

/-- A set of non-constant functions with specific properties -/
def FunctionSet (G : Set (ℝ → ℝ)) : Prop :=
  ∀ f ∈ G, ∃ a b : ℝ, a ≠ 0 ∧ (∀ x, f x = a * x + b) ∧
  (∀ g ∈ G, (g ∘ f) ∈ G) ∧
  (Function.Bijective f → Function.invFun f ∈ G) ∧
  (∃ xₑ : ℝ, f xₑ = xₑ)

/-- The main theorem -/
theorem exists_invariant_point {G : Set (ℝ → ℝ)} (hG : FunctionSet G) :
  ∃ k : ℝ, ∀ f ∈ G, f k = k := by sorry

end NUMINAMATH_CALUDE_exists_invariant_point_l945_94586


namespace NUMINAMATH_CALUDE_fiona_final_piles_count_l945_94564

/-- Represents the number of distinct final pile configurations in Fiona's card arranging process. -/
def fiona_final_piles (n : ℕ) : ℕ :=
  if n ≥ 2 then 2^(n-2) else 1

/-- The theorem stating the number of distinct final pile configurations in Fiona's card arranging process. -/
theorem fiona_final_piles_count (n : ℕ) :
  (∀ k : ℕ, k < n → ∃ (m : ℕ), m ≤ n ∧ fiona_final_piles k = fiona_final_piles m) →
  fiona_final_piles n = if n ≥ 2 then 2^(n-2) else 1 :=
by sorry

end NUMINAMATH_CALUDE_fiona_final_piles_count_l945_94564


namespace NUMINAMATH_CALUDE_parallel_lines_angle_l945_94572

/-- Two lines are parallel -/
def parallel (l m : Set (ℝ × ℝ)) : Prop := sorry

/-- A point lies on a line -/
def on_line (P : ℝ × ℝ) (l : Set (ℝ × ℝ)) : Prop := sorry

/-- Angle measure in degrees -/
def angle_measure (A B C : ℝ × ℝ) : ℝ := sorry

theorem parallel_lines_angle (l m t : Set (ℝ × ℝ)) (P Q C : ℝ × ℝ) :
  parallel l m →
  on_line P l →
  on_line Q m →
  on_line P t →
  on_line Q t →
  on_line C m →
  angle_measure P Q C = 50 →
  angle_measure A P Q = 130 →
  true := by sorry

end NUMINAMATH_CALUDE_parallel_lines_angle_l945_94572


namespace NUMINAMATH_CALUDE_rectangle_perimeter_reduction_l945_94558

theorem rectangle_perimeter_reduction (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  2 * (0.9 * a + 0.8 * b) = 0.88 * 2 * (a + b) → 
  2 * (0.8 * a + 0.9 * b) = 0.82 * 2 * (a + b) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_reduction_l945_94558


namespace NUMINAMATH_CALUDE_all_divisors_of_30240_l945_94537

theorem all_divisors_of_30240 : ∀ n : ℕ, 1 ≤ n ∧ n ≤ 9 → 30240 % n = 0 := by
  sorry

end NUMINAMATH_CALUDE_all_divisors_of_30240_l945_94537


namespace NUMINAMATH_CALUDE_smallest_positive_solution_l945_94591

theorem smallest_positive_solution :
  ∃ (x : ℝ), x > 0 ∧ Real.sqrt (3 * x) = 5 * x + 1 ∧
  ∀ (y : ℝ), y > 0 ∧ Real.sqrt (3 * y) = 5 * y + 1 → x ≤ y ∧
  x = (-7 - Real.sqrt 349) / 50 :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_solution_l945_94591


namespace NUMINAMATH_CALUDE_stream_speed_l945_94555

/-- Proves that given a man's swimming speed in still water and the relationship
    between upstream and downstream swimming times, the speed of the stream is 0.5 km/h. -/
theorem stream_speed (swimming_speed : ℝ) (upstream_time_ratio : ℝ) :
  swimming_speed = 1.5 →
  upstream_time_ratio = 2 →
  ∃ (stream_speed : ℝ),
    (swimming_speed + stream_speed) * 1 = (swimming_speed - stream_speed) * upstream_time_ratio ∧
    stream_speed = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_l945_94555


namespace NUMINAMATH_CALUDE_smallest_b_for_factorization_l945_94513

theorem smallest_b_for_factorization : 
  ∃ (b : ℕ), b > 0 ∧ 
  (∀ (x : ℤ), ∃ (p q : ℤ), x^2 + b*x + 1764 = (x + p) * (x + q)) ∧
  (∀ (b' : ℕ), 0 < b' ∧ b' < b → 
    ¬(∀ (x : ℤ), ∃ (p q : ℤ), x^2 + b'*x + 1764 = (x + p) * (x + q))) ∧
  b = 84 :=
by sorry

end NUMINAMATH_CALUDE_smallest_b_for_factorization_l945_94513


namespace NUMINAMATH_CALUDE_total_points_in_game_l945_94516

def rounds : ℕ := 177
def points_per_round : ℕ := 46

theorem total_points_in_game : rounds * points_per_round = 8142 := by
  sorry

end NUMINAMATH_CALUDE_total_points_in_game_l945_94516


namespace NUMINAMATH_CALUDE_optimal_feeding_program_l945_94546

/-- Represents a feeding program for animals -/
structure FeedingProgram where
  x : ℝ  -- Amount of first feed in kg
  y : ℝ  -- Amount of second feed in kg

/-- Nutrient requirements for each animal per day -/
def nutrientRequirements : ℝ × ℝ × ℝ := (45, 60, 5)

/-- Nutrient content of first feed per kg -/
def firstFeedContent : ℝ × ℝ := (10, 10)

/-- Nutrient content of second feed per kg -/
def secondFeedContent : ℝ × ℝ × ℝ := (10, 20, 5)

/-- Cost of feeds in Ft/q -/
def feedCosts : ℝ × ℝ := (30, 120)

/-- Feeding loss percentages -/
def feedingLoss : ℝ × ℝ := (0.1, 0.2)

/-- Check if a feeding program satisfies nutrient requirements -/
def satisfiesRequirements (fp : FeedingProgram) : Prop :=
  let (reqA, reqB, reqC) := nutrientRequirements
  let (firstA, firstB) := firstFeedContent
  let (secondA, secondB, secondC) := secondFeedContent
  firstA * fp.x + secondA * fp.y ≥ reqA ∧
  firstB * fp.x + secondB * fp.y ≥ reqB ∧
  secondC * fp.y ≥ reqC

/-- Calculate the cost of a feeding program -/
def calculateCost (fp : FeedingProgram) : ℝ :=
  let (costFirst, costSecond) := feedCosts
  costFirst * fp.x + costSecond * fp.y

/-- Calculate the feeding loss of a feeding program -/
def calculateLoss (fp : FeedingProgram) : ℝ :=
  let (lossFirst, lossSecond) := feedingLoss
  lossFirst * fp.x + lossSecond * fp.y

/-- Theorem stating that (4, 1) is the optimal feeding program -/
theorem optimal_feeding_program :
  let optimalProgram := FeedingProgram.mk 4 1
  satisfiesRequirements optimalProgram ∧
  ∀ fp : FeedingProgram, satisfiesRequirements fp →
    calculateCost optimalProgram ≤ calculateCost fp ∧
    calculateLoss optimalProgram ≤ calculateLoss fp :=
by sorry

end NUMINAMATH_CALUDE_optimal_feeding_program_l945_94546


namespace NUMINAMATH_CALUDE_smaller_number_proof_l945_94531

theorem smaller_number_proof (x y : ℝ) 
  (sum_eq : x + y = 16)
  (diff_eq : x - y = 4)
  (prod_eq : x * y = 60) :
  min x y = 6 := by
sorry

end NUMINAMATH_CALUDE_smaller_number_proof_l945_94531


namespace NUMINAMATH_CALUDE_population_increase_l945_94588

theorem population_increase (a b c : ℝ) :
  let increase_0_to_1 := 1 + a / 100
  let increase_1_to_2 := 1 + b / 100
  let increase_2_to_3 := 1 + c / 100
  let total_increase := increase_0_to_1 * increase_1_to_2 * increase_2_to_3 - 1
  total_increase * 100 = a + b + c + (a * b + b * c + a * c) / 100 + a * b * c / 10000 :=
by sorry

end NUMINAMATH_CALUDE_population_increase_l945_94588


namespace NUMINAMATH_CALUDE_rope_cutting_problem_l945_94523

theorem rope_cutting_problem (a b c : ℕ) 
  (ha : a = 45) (hb : b = 60) (hc : c = 75) :
  Nat.gcd a (Nat.gcd b c) = 15 := by
  sorry

end NUMINAMATH_CALUDE_rope_cutting_problem_l945_94523


namespace NUMINAMATH_CALUDE_projection_difference_l945_94527

/-- Represents a projection type -/
inductive ProjectionType
| Parallel
| Central

/-- Represents the behavior of projection lines -/
inductive ProjectionLineBehavior
| Parallel
| Converging

/-- Defines the projection line behavior for a given projection type -/
def projectionLineBehavior (p : ProjectionType) : ProjectionLineBehavior :=
  match p with
  | ProjectionType.Parallel => ProjectionLineBehavior.Parallel
  | ProjectionType.Central => ProjectionLineBehavior.Converging

/-- Theorem stating the difference between parallel and central projections -/
theorem projection_difference :
  ∀ (p : ProjectionType),
    (p = ProjectionType.Parallel ∧ projectionLineBehavior p = ProjectionLineBehavior.Parallel) ∨
    (p = ProjectionType.Central ∧ projectionLineBehavior p = ProjectionLineBehavior.Converging) :=
by sorry

end NUMINAMATH_CALUDE_projection_difference_l945_94527


namespace NUMINAMATH_CALUDE_arithmetic_sequence_slope_l945_94557

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  sum : ℕ → ℝ -- Sum function
  sum_def : ∀ n, sum n = n * (2 * a 1 + (n - 1) * d) / 2
  a_def : ∀ n, a n = a 1 + (n - 1) * d

/-- The slope of the line passing through P(n, a_n) and Q(n+2, a_{n+2}) is 4 -/
theorem arithmetic_sequence_slope (seq : ArithmeticSequence)
  (h1 : seq.sum 2 = 10)
  (h2 : seq.sum 5 = 55) :
  ∀ n : ℕ, n ≥ 1 → (seq.a (n + 2) - seq.a n) / 2 = 4 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_slope_l945_94557


namespace NUMINAMATH_CALUDE_dilute_herbal_essence_l945_94593

/-- Proves that adding 7.5 ounces of water to a 15-ounce solution containing 60% essence
    results in a new solution with 40% essence -/
theorem dilute_herbal_essence :
  let initial_weight : ℝ := 15
  let initial_concentration : ℝ := 0.6
  let final_concentration : ℝ := 0.4
  let water_added : ℝ := 7.5
  let essence_amount : ℝ := initial_weight * initial_concentration
  let final_weight : ℝ := initial_weight + water_added
  essence_amount / final_weight = final_concentration := by sorry

end NUMINAMATH_CALUDE_dilute_herbal_essence_l945_94593


namespace NUMINAMATH_CALUDE_age_difference_equals_first_ratio_l945_94562

/-- Represents the age ratio of four siblings -/
structure AgeRatio :=
  (a b c d : ℕ)

/-- Calculates the age difference between the first two siblings given their age ratio and total future age -/
def ageDifference (ratio : AgeRatio) (totalFutureAge : ℕ) : ℚ :=
  let x : ℚ := (totalFutureAge - 20 : ℚ) / (ratio.a + ratio.b + ratio.c + ratio.d : ℚ)
  ratio.a * x - ratio.b * x

/-- Theorem: The age difference between the first two siblings is equal to the first number in the ratio -/
theorem age_difference_equals_first_ratio 
  (ratio : AgeRatio) 
  (totalFutureAge : ℕ) 
  (h1 : ratio.a = 4) 
  (h2 : ratio.b = 3) 
  (h3 : ratio.c = 7) 
  (h4 : ratio.d = 5) 
  (h5 : totalFutureAge = 230) : 
  ageDifference ratio totalFutureAge = ratio.a := by
  sorry

#eval ageDifference ⟨4, 3, 7, 5⟩ 230

end NUMINAMATH_CALUDE_age_difference_equals_first_ratio_l945_94562


namespace NUMINAMATH_CALUDE_tournament_dominating_set_exists_l945_94532

/-- Represents a directed graph where vertices are players and edges represent wins. -/
structure TournamentGraph where
  players : Finset ℕ
  wins : players → players → Prop

/-- A tournament graph is complete if every player has played against every other player exactly once. -/
def IsCompleteTournament (g : TournamentGraph) : Prop :=
  ∀ p q : g.players, p ≠ q → (g.wins p q ∨ g.wins q p) ∧ ¬(g.wins p q ∧ g.wins q p)

/-- A set of players dominates the rest if every other player has lost to at least one player in the set. -/
def DominatingSet (g : TournamentGraph) (s : Finset g.players) : Prop :=
  ∀ p : g.players, p ∉ s → ∃ q ∈ s, g.wins q p

theorem tournament_dominating_set_exists (g : TournamentGraph) 
  (h_complete : IsCompleteTournament g) (h_size : g.players.card = 14) :
  ∃ s : Finset g.players, s.card = 3 ∧ DominatingSet g s := by sorry

end NUMINAMATH_CALUDE_tournament_dominating_set_exists_l945_94532


namespace NUMINAMATH_CALUDE_water_consumption_problem_l945_94556

/-- The water consumption problem -/
theorem water_consumption_problem 
  (total_water : ℝ) 
  (initial_people : ℕ) 
  (initial_days : ℕ) 
  (later_people : ℕ) 
  (later_days : ℕ) 
  (h1 : total_water = 18.9)
  (h2 : initial_people = 6)
  (h3 : initial_days = 4)
  (h4 : later_people = 7)
  (h5 : later_days = 2) :
  ∃ (x : ℝ), 
    x = 6 ∧ 
    (initial_people * (total_water / (initial_people * initial_days)) * later_days + 
     x * (total_water / (initial_people * initial_days)) * later_days = total_water) := by
  sorry

end NUMINAMATH_CALUDE_water_consumption_problem_l945_94556


namespace NUMINAMATH_CALUDE_wrapper_cap_difference_l945_94504

/-- Represents Danny's collection of bottle caps and wrappers -/
structure Collection where
  caps : ℕ
  wrappers : ℕ

/-- The number of bottle caps and wrappers Danny found at the park -/
def park_find : Collection :=
  { caps := 15, wrappers := 18 }

/-- Danny's current collection -/
def current_collection : Collection :=
  { caps := 35, wrappers := 67 }

/-- The theorem stating the difference between wrappers and bottle caps in Danny's collection -/
theorem wrapper_cap_difference :
  current_collection.wrappers - current_collection.caps = 32 :=
by sorry

end NUMINAMATH_CALUDE_wrapper_cap_difference_l945_94504


namespace NUMINAMATH_CALUDE_class_selection_ways_l945_94576

def total_classes : ℕ := 10
def advanced_classes : ℕ := 3
def classes_to_select : ℕ := 5
def min_advanced : ℕ := 2

theorem class_selection_ways : 
  (Nat.choose advanced_classes min_advanced) * 
  (Nat.choose (total_classes - advanced_classes) (classes_to_select - min_advanced)) = 105 := by
  sorry

end NUMINAMATH_CALUDE_class_selection_ways_l945_94576


namespace NUMINAMATH_CALUDE_carrot_juice_distribution_l945_94566

-- Define the set of glass volumes
def glassVolumes : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9]

-- Define a type for a distribution of glasses
def Distribution := List (List ℕ)

-- Define the property of a valid distribution
def isValidDistribution (d : Distribution) : Prop :=
  d.length = 3 ∧
  d.all (fun l => l.length = 3) ∧
  d.all (fun l => l.sum = 15) ∧
  d.join.toFinset = glassVolumes.toFinset

-- State the theorem
theorem carrot_juice_distribution :
  ∃ (d1 d2 : Distribution),
    isValidDistribution d1 ∧
    isValidDistribution d2 ∧
    d1 ≠ d2 ∧
    ∀ (d : Distribution), isValidDistribution d → (d = d1 ∨ d = d2) :=
  sorry

end NUMINAMATH_CALUDE_carrot_juice_distribution_l945_94566


namespace NUMINAMATH_CALUDE_last_i_becomes_w_l945_94505

/-- Represents a letter in the alphabet --/
def Letter := Fin 26

/-- The encryption shift for the nth occurrence of a letter --/
def shift (n : Nat) : Nat := n^2

/-- The message to be encrypted --/
def message : String := "Mathematics is meticulous"

/-- Count occurrences of a character in a string --/
def countOccurrences (c : Char) (s : String) : Nat :=
  s.toList.filter (· = c) |>.length

/-- Apply the shift to a letter --/
def applyShift (l : Letter) (s : Nat) : Letter :=
  ⟨(l.val + s) % 26, by sorry⟩

/-- The theorem to be proved --/
theorem last_i_becomes_w :
  let iCount := countOccurrences 'i' message
  let totalShift := (List.range iCount).map shift |>.sum
  let iLetter : Letter := ⟨8, by sorry⟩  -- 'i' is the 9th letter (0-indexed)
  applyShift iLetter totalShift = ⟨22, by sorry⟩  -- 'w' is the 23rd letter (0-indexed)
  := by sorry

end NUMINAMATH_CALUDE_last_i_becomes_w_l945_94505


namespace NUMINAMATH_CALUDE_texas_integrated_school_students_l945_94578

theorem texas_integrated_school_students (original_classes : ℕ) (students_per_class : ℕ) (new_classes : ℕ) : 
  original_classes = 15 → 
  students_per_class = 20 → 
  new_classes = 5 → 
  (original_classes + new_classes) * students_per_class = 400 := by
sorry

end NUMINAMATH_CALUDE_texas_integrated_school_students_l945_94578


namespace NUMINAMATH_CALUDE_f_monotone_decreasing_l945_94507

-- Define the function f(x)
def f (x : ℝ) : ℝ := x - x^2 - x

-- State the theorem
theorem f_monotone_decreasing :
  ∀ x₁ x₂ : ℝ, -1/3 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 → f x₂ < f x₁ := by
  sorry

end NUMINAMATH_CALUDE_f_monotone_decreasing_l945_94507


namespace NUMINAMATH_CALUDE_goose_eggs_theorem_l945_94598

theorem goose_eggs_theorem (total_eggs : ℕ) : 
  (1 : ℚ) / 4 * (4 : ℚ) / 5 * (3 : ℚ) / 5 * total_eggs = 120 →
  total_eggs = 1000 := by
  sorry

end NUMINAMATH_CALUDE_goose_eggs_theorem_l945_94598


namespace NUMINAMATH_CALUDE_sector_central_angle_l945_94522

/-- Given a sector with arc length 4 cm and area 4 cm², prove that its central angle is 2 radians. -/
theorem sector_central_angle (r : ℝ) (θ : ℝ) : 
  r * θ = 4 → (1/2) * r^2 * θ = 4 → θ = 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_central_angle_l945_94522


namespace NUMINAMATH_CALUDE_jills_lavender_candles_l945_94585

/-- Represents the number of candles made with each scent -/
structure CandleCounts where
  lavender : ℕ
  coconut : ℕ
  almond : ℕ
  jasmine : ℕ

/-- Represents the amount of scent required for each candle -/
structure ScentRequirements where
  lavender : ℕ
  coconut : ℕ
  almond : ℕ
  jasmine : ℕ

/-- Theorem stating the number of lavender candles Jill made -/
theorem jills_lavender_candles 
  (req : ScentRequirements)
  (counts : CandleCounts)
  (h1 : req.lavender = 10)
  (h2 : req.coconut = 8)
  (h3 : req.almond = 12)
  (h4 : req.jasmine = 14)
  (h5 : counts.lavender = 3 * counts.coconut)
  (h6 : counts.almond = 2 * counts.jasmine)
  (h7 : counts.almond = 10)
  (h8 : req.coconut * counts.coconut = (5/2) * req.almond * counts.almond)
  : counts.lavender = 111 := by
  sorry

#check jills_lavender_candles

end NUMINAMATH_CALUDE_jills_lavender_candles_l945_94585


namespace NUMINAMATH_CALUDE_range_of_function_l945_94520

theorem range_of_function (x : ℝ) : 
  1/3 ≤ (2 - Real.cos x) / (2 + Real.cos x) ∧ (2 - Real.cos x) / (2 + Real.cos x) ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_function_l945_94520


namespace NUMINAMATH_CALUDE_circle_omega_area_l945_94583

/-- Circle ω with points A and B, and tangent lines intersecting on x-axis -/
structure Circle_omega where
  /-- Point A on the circle -/
  A : ℝ × ℝ
  /-- Point B on the circle -/
  B : ℝ × ℝ
  /-- The tangent lines at A and B intersect on the x-axis -/
  tangent_intersection_on_x_axis : Prop

/-- Theorem: Area of circle ω is 120375π/9600 -/
theorem circle_omega_area (ω : Circle_omega) 
  (h1 : ω.A = (5, 15)) 
  (h2 : ω.B = (13, 9)) : 
  ∃ (r : ℝ), r > 0 ∧ π * r^2 = 120375 * π / 9600 := by
  sorry

end NUMINAMATH_CALUDE_circle_omega_area_l945_94583

import Mathlib

namespace sqrt_meaningful_condition_l3681_368182

theorem sqrt_meaningful_condition (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 4) ↔ x ≥ 4 := by
  sorry

end sqrt_meaningful_condition_l3681_368182


namespace base_number_power_remainder_l3681_368183

theorem base_number_power_remainder (base : ℕ) : base = 1 → base ^ 8 % 100 = 1 := by
  sorry

end base_number_power_remainder_l3681_368183


namespace fixed_point_of_exponential_function_l3681_368179

/-- For all a > 0 and a ≠ 1, the function f(x) = a^(x-1) + 4 passes through (1, 5) -/
theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x-1) + 4
  f 1 = 5 := by
  sorry

end fixed_point_of_exponential_function_l3681_368179


namespace square_root_of_nine_l3681_368170

-- Define the square root function
def square_root (x : ℝ) : Set ℝ := {y : ℝ | y * y = x}

-- State the theorem
theorem square_root_of_nine : square_root 9 = {3, -3} := by sorry

end square_root_of_nine_l3681_368170


namespace data_analytics_course_hours_l3681_368172

/-- Calculates the total hours spent on a course given the course duration and weekly schedule. -/
def total_course_hours (weeks : ℕ) (three_hour_classes : ℕ) (four_hour_classes : ℕ) (homework_hours : ℕ) : ℕ :=
  weeks * (three_hour_classes * 3 + four_hour_classes * 4 + homework_hours)

/-- Proves that the total hours spent on the given course is 336. -/
theorem data_analytics_course_hours : 
  total_course_hours 24 2 1 4 = 336 := by
  sorry

end data_analytics_course_hours_l3681_368172


namespace not_p_and_q_implies_at_most_one_true_l3681_368120

theorem not_p_and_q_implies_at_most_one_true (p q : Prop) :
  ¬(p ∧ q) → (¬p ∨ ¬q) :=
by
  sorry

end not_p_and_q_implies_at_most_one_true_l3681_368120


namespace distribution_five_to_three_l3681_368196

/-- The number of ways to distribute n distinct objects into k distinct groups,
    where each group must contain at least min_per_group objects and
    at most max_per_group objects. -/
def distribution_count (n k min_per_group max_per_group : ℕ) : ℕ := sorry

/-- Theorem: There are 30 ways to distribute 5 distinct objects into 3 distinct groups,
    where each group must contain at least 1 object and at most 2 objects. -/
theorem distribution_five_to_three : distribution_count 5 3 1 2 = 30 := by sorry

end distribution_five_to_three_l3681_368196


namespace simple_compound_interest_equivalence_l3681_368137

theorem simple_compound_interest_equivalence (P : ℝ) : 
  (P * 0.04 * 2 = 0.5 * (4000 * ((1 + 0.10)^2 - 1))) → P = 5250 :=
by sorry

end simple_compound_interest_equivalence_l3681_368137


namespace constant_term_binomial_expansion_l3681_368166

theorem constant_term_binomial_expansion :
  let n : ℕ := 8
  let a : ℝ := 1
  let b : ℝ := 1/3
  (Finset.sum (Finset.range (n + 1)) (λ k => Nat.choose n k * a^k * b^(n - k) * (if k = n/2 then 1 else 0))) = 28 :=
by sorry

end constant_term_binomial_expansion_l3681_368166


namespace smallest_five_digit_mod_9_5_l3681_368103

theorem smallest_five_digit_mod_9_5 : ∃ n : ℕ, 
  (n ≥ 10000 ∧ n < 100000) ∧  -- five-digit integer
  n % 9 = 5 ∧                 -- equivalent to 5 mod 9
  (∀ m : ℕ, (m ≥ 10000 ∧ m < 100000 ∧ m % 9 = 5) → n ≤ m) ∧ 
  n = 10004 := by
sorry

end smallest_five_digit_mod_9_5_l3681_368103


namespace bag_equals_two_balls_l3681_368110

/-- Represents the weight of an object -/
structure Weight : Type :=
  (value : ℝ)

/-- Represents a balanced scale -/
structure BalancedScale : Type :=
  (left_bags : ℕ)
  (left_balls : ℕ)
  (right_bags : ℕ)
  (right_balls : ℕ)
  (bag_weight : Weight)
  (ball_weight : Weight)

/-- Predicate to check if a scale is balanced -/
def is_balanced (s : BalancedScale) : Prop :=
  s.left_bags * s.bag_weight.value + s.left_balls * s.ball_weight.value =
  s.right_bags * s.bag_weight.value + s.right_balls * s.ball_weight.value

theorem bag_equals_two_balls (s : BalancedScale) :
  s.left_bags = 5 ∧ s.left_balls = 4 ∧ s.right_bags = 2 ∧ s.right_balls = 10 ∧
  is_balanced s →
  s.bag_weight.value = 2 * s.ball_weight.value :=
sorry

end bag_equals_two_balls_l3681_368110


namespace imaginary_unit_sum_l3681_368188

theorem imaginary_unit_sum (i : ℂ) (hi : i^2 = -1) : i + i^2 + i^3 = -1 := by
  sorry

end imaginary_unit_sum_l3681_368188


namespace unique_integer_value_l3681_368129

theorem unique_integer_value : ∃! x : ℤ, 
  1 < x ∧ x < 9 ∧ 
  2 < x ∧ x < 15 ∧ 
  -1 < x ∧ x < 7 ∧ 
  0 < x ∧ x < 4 ∧ 
  x + 1 < 5 :=
by
  sorry

end unique_integer_value_l3681_368129


namespace quadratic_equation_roots_l3681_368175

theorem quadratic_equation_roots (k : ℚ) :
  (∃ x, 5 * x^2 + k * x - 6 = 0 ∧ x = 2) →
  (∃ y, 5 * y^2 + k * y - 6 = 0 ∧ y = -3/5) ∧
  k = -7 := by
  sorry

end quadratic_equation_roots_l3681_368175


namespace gnomes_distribution_l3681_368127

/-- Given a street with houses and gnomes, calculates the number of gnomes in each of the first few houses -/
def gnomes_per_house (total_houses : ℕ) (total_gnomes : ℕ) (last_house_gnomes : ℕ) : ℕ :=
  (total_gnomes - last_house_gnomes) / (total_houses - 1)

/-- Theorem stating that under given conditions, each of the first few houses has 3 gnomes -/
theorem gnomes_distribution (total_houses : ℕ) (total_gnomes : ℕ) (last_house_gnomes : ℕ)
  (h1 : total_houses = 5)
  (h2 : total_gnomes = 20)
  (h3 : last_house_gnomes = 8) :
  gnomes_per_house total_houses total_gnomes last_house_gnomes = 3 := by
  sorry

end gnomes_distribution_l3681_368127


namespace anna_candy_count_l3681_368113

theorem anna_candy_count (initial_candies : ℕ) (received_candies : ℕ) :
  initial_candies = 5 →
  received_candies = 86 →
  initial_candies + received_candies = 91 := by
  sorry

end anna_candy_count_l3681_368113


namespace quadratic_root_relation_l3681_368134

theorem quadratic_root_relation : 
  ∀ x₁ x₂ : ℝ, 
  x₁^2 - 2*x₁ - 8 = 0 → 
  x₂^2 - 2*x₂ - 8 = 0 → 
  (x₁ + x₂) / (x₁ * x₂) = -1/4 := by
sorry

end quadratic_root_relation_l3681_368134


namespace correct_propositions_l3681_368158

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (subset : Line → Plane → Prop)
variable (perp : Line → Plane → Prop)
variable (parallel_lines : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)

-- State the theorem
theorem correct_propositions
  (m n : Line) (α β : Plane) 
  (h_distinct_lines : m ≠ n)
  (h_distinct_planes : α ≠ β) :
  -- Proposition 2
  (parallel_planes α β ∧ subset m α → parallel_lines m β) ∧
  -- Proposition 3
  (perp n α ∧ perp n β ∧ perp m α → perp m β) :=
by sorry

end correct_propositions_l3681_368158


namespace vector_simplification_1_vector_simplification_2_vector_simplification_3_l3681_368124

variable {V : Type*} [AddCommGroup V]

-- Define vector between two points
def vec (A B : V) : V := B - A

-- Theorem 1
theorem vector_simplification_1 (A B C D : V) :
  vec A B + vec B C - vec A D = vec D C := by sorry

-- Theorem 2
theorem vector_simplification_2 (A B C D : V) :
  (vec A B - vec C D) - (vec A C - vec B D) = 0 := by sorry

-- Theorem 3
theorem vector_simplification_3 (A B C D O : V) :
  (vec A C + vec B O + vec O A) - (vec D C - vec D O - vec O B) = 0 := by sorry

end vector_simplification_1_vector_simplification_2_vector_simplification_3_l3681_368124


namespace complex_statements_l3681_368131

open Complex

theorem complex_statements :
  (∃ z : ℂ, z = 1 - I ∧ Complex.abs (2 / z + z^2) = Real.sqrt 2) ∧
  (∃ z : ℂ, z = 1 / I ∧ (z^5 + 1).re > 0 ∧ (z^5 + 1).im < 0) :=
by sorry

end complex_statements_l3681_368131


namespace base_seven_54321_to_decimal_l3681_368168

def base_seven_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (7 ^ i)) 0

theorem base_seven_54321_to_decimal :
  base_seven_to_decimal [1, 2, 3, 4, 5] = 13539 :=
by sorry

end base_seven_54321_to_decimal_l3681_368168


namespace fifth_power_fourth_decomposition_l3681_368141

/-- 
For a natural number m ≥ 2, m^4 can be decomposed into a sum of m consecutive odd numbers.
This function returns the starting odd number for this decomposition.
-/
def startingOddNumber (m : ℕ) : ℕ := 
  if m = 2 then 7 else 2 * (((m - 1) + 2) * (m - 2) / 2) + 1

/-- 
This function returns the nth odd number in the sequence starting from a given odd number.
-/
def nthOddNumber (start : ℕ) (n : ℕ) : ℕ := start + 2 * (n - 1)

theorem fifth_power_fourth_decomposition : 
  nthOddNumber (startingOddNumber 5) 3 = 125 := by sorry

end fifth_power_fourth_decomposition_l3681_368141


namespace remaining_truck_capacity_l3681_368115

theorem remaining_truck_capacity
  (max_load : ℕ)
  (bag_mass : ℕ)
  (num_bags : ℕ)
  (h1 : max_load = 900)
  (h2 : bag_mass = 8)
  (h3 : num_bags = 100) :
  max_load - (bag_mass * num_bags) = 100 :=
by
  sorry

end remaining_truck_capacity_l3681_368115


namespace total_hats_bought_l3681_368164

theorem total_hats_bought (blue_hat_price green_hat_price total_price green_hats : ℕ)
  (h1 : blue_hat_price = 6)
  (h2 : green_hat_price = 7)
  (h3 : total_price = 540)
  (h4 : green_hats = 30) :
  ∃ (blue_hats : ℕ), blue_hats * blue_hat_price + green_hats * green_hat_price = total_price ∧
                     blue_hats + green_hats = 85 := by
  sorry

end total_hats_bought_l3681_368164


namespace tyrone_quarters_l3681_368199

/-- Represents the count of each type of coin or bill --/
structure CoinCount where
  dollars_1 : ℕ
  dollars_5 : ℕ
  dimes : ℕ
  nickels : ℕ
  pennies : ℕ

/-- Calculates the total value in dollars of a given coin count, excluding quarters --/
def value_without_quarters (c : CoinCount) : ℚ :=
  c.dollars_1 + 5 * c.dollars_5 + 0.1 * c.dimes + 0.05 * c.nickels + 0.01 * c.pennies

/-- The value of a quarter in dollars --/
def quarter_value : ℚ := 0.25

theorem tyrone_quarters : 
  ∀ (c : CoinCount) (total : ℚ),
    c.dollars_1 = 2 →
    c.dollars_5 = 1 →
    c.dimes = 20 →
    c.nickels = 8 →
    c.pennies = 35 →
    total = 13 →
    (total - value_without_quarters c) / quarter_value = 13 := by
  sorry

end tyrone_quarters_l3681_368199


namespace min_sum_reciprocals_l3681_368153

theorem min_sum_reciprocals (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) 
  (sum_eq_3 : x + y + z = 3) : 
  (1 / (x + y) + 1 / (y + z) + 1 / (z + x)) ≥ 3 / 2 := by
  sorry


end min_sum_reciprocals_l3681_368153


namespace triangle_side_length_theorem_l3681_368122

def triangle_side_length (a : ℝ) : Set ℝ :=
  if a < Real.sqrt 3 / 2 then
    ∅
  else if a = Real.sqrt 3 / 2 then
    {1/2}
  else if a < 1 then
    {(1 + Real.sqrt (4 * a^2 - 3)) / 2, (1 - Real.sqrt (4 * a^2 - 3)) / 2}
  else
    {(1 + Real.sqrt (4 * a^2 - 3)) / 2}

theorem triangle_side_length_theorem (a : ℝ) :
  let A : ℝ := 60 * π / 180
  let AB : ℝ := 1
  let BC : ℝ := a
  ∀ AC ∈ triangle_side_length a,
    AC^2 = AB^2 + BC^2 - 2 * AB * BC * Real.cos A :=
by sorry

end triangle_side_length_theorem_l3681_368122


namespace rachel_math_homework_l3681_368130

theorem rachel_math_homework (total_math_bio : ℕ) (bio_pages : ℕ) (h1 : total_math_bio = 11) (h2 : bio_pages = 3) :
  total_math_bio - bio_pages = 8 :=
by sorry

end rachel_math_homework_l3681_368130


namespace cards_left_calculation_l3681_368163

def initial_cards : ℕ := 455
def cards_given_away : ℕ := 301

theorem cards_left_calculation : initial_cards - cards_given_away = 154 := by
  sorry

end cards_left_calculation_l3681_368163


namespace complement_of_M_l3681_368180

-- Define the universal set U
def U : Set ℕ := {1, 2, 3}

-- Define the set M
def M : Set ℕ := {x ∈ U | x^2 - 3*x + 2 = 0}

-- State the theorem
theorem complement_of_M (x : ℕ) : x ∈ (U \ M) ↔ x = 3 := by
  sorry

end complement_of_M_l3681_368180


namespace train_length_l3681_368157

/-- The length of a train given specific conditions --/
theorem train_length (jogger_speed : ℝ) (train_speed : ℝ) (initial_distance : ℝ) (passing_time : ℝ) :
  jogger_speed = 9 * (1000 / 3600) →
  train_speed = 45 * (1000 / 3600) →
  initial_distance = 280 →
  passing_time = 40 →
  (train_speed - jogger_speed) * passing_time + initial_distance = 680 := by
  sorry

end train_length_l3681_368157


namespace symmetric_points_sum_l3681_368111

/-- Two points are symmetric with respect to the origin if their coordinates sum to zero -/
def symmetric_wrt_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ + x₂ = 0 ∧ y₁ + y₂ = 0

/-- The theorem stating that if points A(m-1, -3) and B(2, n) are symmetric
    with respect to the origin, then m + n = 2 -/
theorem symmetric_points_sum (m n : ℝ) :
  symmetric_wrt_origin (m - 1) (-3) 2 n → m + n = 2 := by
  sorry

end symmetric_points_sum_l3681_368111


namespace park_population_l3681_368100

/-- Calculates the total population of lions, leopards, and elephants in a park. -/
theorem park_population (num_lions : ℕ) (num_leopards : ℕ) (num_elephants : ℕ) : 
  num_lions = 200 →
  num_lions = 2 * num_leopards →
  num_elephants = (num_lions + num_leopards) / 2 →
  num_lions + num_leopards + num_elephants = 450 := by
  sorry

#check park_population

end park_population_l3681_368100


namespace gas_volume_calculation_l3681_368177

/-- Calculate the volume of gas using the Mendeleev-Clapeyron equation -/
theorem gas_volume_calculation (m R T p M : ℝ) (h_m : m = 140) (h_R : R = 8.314) 
  (h_T : T = 305) (h_p : p = 283710) (h_M : M = 28) :
  let V := (m * R * T * 1000) / (p * M)
  ∃ ε > 0, |V - 44.7| < ε :=
sorry

end gas_volume_calculation_l3681_368177


namespace angle_between_vectors_l3681_368189

theorem angle_between_vectors (a b : ℝ × ℝ) :
  (a.1 * (a.1 - b.1) + a.2 * (a.2 - b.2) = 2) →
  (Real.sqrt (a.1^2 + a.2^2) = 1) →
  (Real.sqrt (b.1^2 + b.2^2) = 2) →
  Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))) = 2 * π / 3 := by
  sorry

end angle_between_vectors_l3681_368189


namespace walking_equations_correct_l3681_368150

/-- Represents the speeds and distances of two people walking --/
structure WalkingScenario where
  distance : ℝ
  catchup_time : ℝ
  meet_time : ℝ
  speed_a : ℝ
  speed_b : ℝ

/-- The system of equations correctly represents the walking scenario --/
def correct_equations (s : WalkingScenario) : Prop :=
  (10 * s.speed_b - 10 * s.speed_a = s.distance) ∧
  (2 * s.speed_a + 2 * s.speed_b = s.distance)

/-- The given scenario satisfies the conditions --/
def satisfies_conditions (s : WalkingScenario) : Prop :=
  s.distance = 50 ∧
  s.catchup_time = 10 ∧
  s.meet_time = 2 ∧
  s.speed_a > 0 ∧
  s.speed_b > 0

theorem walking_equations_correct (s : WalkingScenario) 
  (h : satisfies_conditions s) : correct_equations s := by
  sorry


end walking_equations_correct_l3681_368150


namespace prob_at_least_one_3_or_5_correct_l3681_368109

/-- The probability of at least one die showing either a 3 or a 5 when rolling two fair 6-sided dice -/
def prob_at_least_one_3_or_5 : ℚ :=
  5 / 9

/-- The number of sides on each die -/
def num_sides : ℕ := 6

/-- The set of outcomes for a single die roll -/
def die_outcomes : Finset ℕ := Finset.range num_sides

/-- The set of favorable outcomes for a single die (3 or 5) -/
def favorable_single : Finset ℕ := {3, 5}

/-- The sample space for rolling two dice -/
def sample_space : Finset (ℕ × ℕ) := die_outcomes.product die_outcomes

/-- The event where at least one die shows 3 or 5 -/
def event : Finset (ℕ × ℕ) :=
  sample_space.filter (fun p => p.1 ∈ favorable_single ∨ p.2 ∈ favorable_single)

theorem prob_at_least_one_3_or_5_correct :
  (event.card : ℚ) / sample_space.card = prob_at_least_one_3_or_5 := by
  sorry

end prob_at_least_one_3_or_5_correct_l3681_368109


namespace eating_competition_time_l3681_368149

/-- Represents the number of minutes it takes to eat everything -/
def total_time : ℕ := 48

/-- Represents the number of jars of honey eaten by Carlson -/
def carlson_honey : ℕ := 8

/-- Represents the number of jars of jam eaten by Carlson -/
def carlson_jam : ℕ := 4

/-- The time it takes Carlson to eat a jar of jam -/
def carlson_jam_time : ℕ := 2

/-- The time it takes Winnie the Pooh to eat a jar of jam -/
def pooh_jam_time : ℕ := 7

/-- The time it takes Winnie the Pooh to eat a pot of honey -/
def pooh_honey_time : ℕ := 3

/-- The time it takes Carlson to eat a pot of honey -/
def carlson_honey_time : ℕ := 5

/-- The total number of jars of jam and pots of honey -/
def total_jars : ℕ := 10

theorem eating_competition_time :
  carlson_honey * carlson_honey_time + carlson_jam * carlson_jam_time = total_time ∧
  (total_jars - carlson_honey) * pooh_honey_time + (total_jars - carlson_jam) * pooh_jam_time = total_time ∧
  carlson_honey + carlson_jam ≤ total_jars :=
by sorry

end eating_competition_time_l3681_368149


namespace sum_of_cubes_l3681_368176

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 := by
  sorry

end sum_of_cubes_l3681_368176


namespace simplify_expression_l3681_368139

theorem simplify_expression (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (sum : a + b + c = 1) :
  (1 / (b^2 + c^2 - a^2)) + (1 / (a^2 + c^2 - b^2)) + (1 / (a^2 + b^2 - c^2)) = -1 := by
  sorry

end simplify_expression_l3681_368139


namespace first_machine_rate_is_35_l3681_368105

/-- The number of copies the first machine makes per minute -/
def first_machine_rate : ℝ := sorry

/-- The number of copies the second machine makes per minute -/
def second_machine_rate : ℝ := 75

/-- The total number of copies both machines make in 30 minutes -/
def total_copies : ℝ := 3300

/-- The time period in minutes -/
def time_period : ℝ := 30

theorem first_machine_rate_is_35 :
  first_machine_rate = 35 :=
by
  have h1 : first_machine_rate * time_period + second_machine_rate * time_period = total_copies :=
    sorry
  sorry

#check first_machine_rate_is_35

end first_machine_rate_is_35_l3681_368105


namespace triangle_third_side_count_l3681_368112

theorem triangle_third_side_count : 
  let side1 : ℕ := 8
  let side2 : ℕ := 12
  let valid_third_side (x : ℕ) : Prop := 
    x + side1 > side2 ∧ 
    x + side2 > side1 ∧ 
    side1 + side2 > x
  (∃! (n : ℕ), (∀ (x : ℕ), valid_third_side x ↔ x ∈ Finset.range n) ∧ n = 15) :=
by sorry

end triangle_third_side_count_l3681_368112


namespace solve_candy_store_problem_l3681_368181

/-- Represents the candy store problem --/
def candy_store_problem (caramel_price toffee_price chocolate_price : ℕ)
  (initial_quantity : ℕ) (initial_money : ℕ) : Prop :=
  let chocolate_promo := initial_quantity / 3
  let toffee_to_buy := initial_quantity - chocolate_promo
  let caramel_promo := toffee_to_buy / 3
  let caramel_to_buy := initial_quantity - caramel_promo
  let total_cost := chocolate_price * initial_quantity +
                    toffee_price * toffee_to_buy +
                    caramel_price * caramel_to_buy
  initial_money - total_cost = 72

/-- Theorem stating the solution to the candy store problem --/
theorem solve_candy_store_problem :
  candy_store_problem 3 5 10 8 200 :=
sorry


end solve_candy_store_problem_l3681_368181


namespace union_of_A_and_B_l3681_368192

def set_A : Set ℝ := {x | x^2 + x - 2 < 0}
def set_B : Set ℝ := {x | x > 0}

theorem union_of_A_and_B : set_A ∪ set_B = {x : ℝ | x > -2} := by sorry

end union_of_A_and_B_l3681_368192


namespace distribute_five_balls_three_boxes_l3681_368107

/-- The number of ways to distribute n distinguishable objects into k distinguishable containers -/
def distribute (n k : ℕ) : ℕ := k^n

/-- Theorem: There are 243 ways to distribute 5 distinguishable balls into 3 distinguishable boxes -/
theorem distribute_five_balls_three_boxes : distribute 5 3 = 243 := by
  sorry

end distribute_five_balls_three_boxes_l3681_368107


namespace percentage_not_sold_approx_l3681_368132

def initial_stock : ℕ := 1400
def monday_sales : ℕ := 75
def tuesday_sales : ℕ := 50
def wednesday_sales : ℕ := 64
def thursday_sales : ℕ := 78
def friday_sales : ℕ := 135

theorem percentage_not_sold_approx (ε : ℝ) (ε_pos : ε > 0) :
  ∃ (p : ℝ), abs (p - ((initial_stock - (monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales)) / initial_stock * 100)) < ε ∧
             abs (p - 71.29) < ε := by
  sorry

end percentage_not_sold_approx_l3681_368132


namespace not_right_triangle_l3681_368173

theorem not_right_triangle (a b c : ℕ) (h1 : a = 5) (h2 : b = 11) (h3 : c = 12) :
  ¬(a^2 + b^2 = c^2) := by
  sorry

end not_right_triangle_l3681_368173


namespace product_of_fractions_squared_l3681_368143

theorem product_of_fractions_squared :
  (8 / 9) ^ 2 * (1 / 3) ^ 2 * (1 / 4) ^ 2 = 4 / 729 := by
  sorry

end product_of_fractions_squared_l3681_368143


namespace sandbox_width_l3681_368193

/-- A sandbox is a rectangle with a specific perimeter and length-width relationship -/
structure Sandbox where
  width : ℝ
  length : ℝ
  perimeter_eq : width * 2 + length * 2 = 30
  length_eq : length = 2 * width

theorem sandbox_width (s : Sandbox) : s.width = 5 := by
  sorry

end sandbox_width_l3681_368193


namespace food_problem_l3681_368190

/-- The number of days food lasts for a group of men -/
def food_duration (initial_men : ℕ) (additional_men : ℕ) (initial_days : ℕ) (additional_days : ℕ) : Prop :=
  initial_men * initial_days = 
  initial_men * 2 + (initial_men + additional_men) * additional_days

theorem food_problem : 
  ∃ (D : ℕ), food_duration 760 760 D 10 ∧ D = 22 := by
  sorry

end food_problem_l3681_368190


namespace mean_score_all_students_l3681_368136

/-- The mean score of all students given specific class conditions --/
theorem mean_score_all_students
  (morning_mean : ℝ)
  (afternoon_mean : ℝ)
  (class_ratio : ℚ)
  (additional_group_score : ℝ)
  (additional_group_ratio : ℚ)
  (h1 : morning_mean = 85)
  (h2 : afternoon_mean = 72)
  (h3 : class_ratio = 4/5)
  (h4 : additional_group_score = 68)
  (h5 : additional_group_ratio = 1/4) :
  ∃ (total_mean : ℝ), total_mean = 87 ∧
    total_mean = (morning_mean * class_ratio + 
                  afternoon_mean * (1 - additional_group_ratio) +
                  additional_group_score * additional_group_ratio) /
                 (class_ratio + 1) := by
  sorry

end mean_score_all_students_l3681_368136


namespace terminating_decimal_of_19_80_l3681_368128

theorem terminating_decimal_of_19_80 : ∃ (n : ℕ), (19 : ℚ) / 80 = (2375 : ℚ) / 10^n :=
sorry

end terminating_decimal_of_19_80_l3681_368128


namespace x_intercepts_count_l3681_368154

theorem x_intercepts_count (x : ℝ) : 
  (∃! x, (x - 4) * (x^2 + 4*x + 8) = 0) := by
  sorry

end x_intercepts_count_l3681_368154


namespace probability_of_matching_pair_l3681_368160

def blue_socks : ℕ := 12
def gray_socks : ℕ := 10
def white_socks : ℕ := 8

def total_socks : ℕ := blue_socks + gray_socks + white_socks

def ways_to_pick_two : ℕ := total_socks.choose 2

def matching_blue_pairs : ℕ := blue_socks.choose 2
def matching_gray_pairs : ℕ := gray_socks.choose 2
def matching_white_pairs : ℕ := white_socks.choose 2

def total_matching_pairs : ℕ := matching_blue_pairs + matching_gray_pairs + matching_white_pairs

theorem probability_of_matching_pair :
  (total_matching_pairs : ℚ) / ways_to_pick_two = 139 / 435 := by sorry

end probability_of_matching_pair_l3681_368160


namespace book_cost_in_rubles_l3681_368156

/-- Represents the exchange rate between US dollars and Namibian dollars -/
def usd_to_namibian : ℚ := 10

/-- Represents the exchange rate between US dollars and Russian rubles -/
def usd_to_rubles : ℚ := 8

/-- Represents the cost of the book in Namibian dollars -/
def book_cost_namibian : ℚ := 200

/-- Theorem stating that the cost of the book in Russian rubles is 160 -/
theorem book_cost_in_rubles :
  (book_cost_namibian / usd_to_namibian) * usd_to_rubles = 160 := by
  sorry


end book_cost_in_rubles_l3681_368156


namespace namjoon_lowest_height_l3681_368184

/-- Heights of planks in centimeters -/
def height_A : ℝ := 2.4
def height_B : ℝ := 3.2
def height_C : ℝ := 2.8

/-- Number of planks each person stands on -/
def num_A : ℕ := 8
def num_B : ℕ := 4
def num_C : ℕ := 5

/-- Total heights for each person -/
def height_Eunji : ℝ := height_A * num_A
def height_Namjoon : ℝ := height_B * num_B
def height_Hoseok : ℝ := height_C * num_C

theorem namjoon_lowest_height :
  height_Namjoon < height_Eunji ∧ height_Namjoon < height_Hoseok :=
by sorry

end namjoon_lowest_height_l3681_368184


namespace floor_width_proof_l3681_368152

/-- Proves that the width of a rectangular floor is 120 cm given specific conditions --/
theorem floor_width_proof (floor_length tile_length tile_width max_tiles : ℕ) 
  (h1 : floor_length = 180)
  (h2 : tile_length = 25)
  (h3 : tile_width = 16)
  (h4 : max_tiles = 54)
  (h5 : floor_length % tile_width = 0)
  (h6 : floor_length / tile_width * (floor_length / tile_width) ≤ max_tiles) :
  ∃ (floor_width : ℕ), floor_width = 120 ∧ 
    floor_length * floor_width = max_tiles * tile_length * tile_width :=
by sorry

end floor_width_proof_l3681_368152


namespace inequality_proof_l3681_368161

theorem inequality_proof (a b c : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : a + b + c + 3 * a * b * c ≥ (a * b)^2 + (b * c)^2 + (c * a)^2 + 3) :
  (a^3 + b^3 + c^3) / 3 ≥ (a * b * c + 2021) / 2022 := by
  sorry

end inequality_proof_l3681_368161


namespace abc_relationship_l3681_368118

open Real

noncomputable def a : ℝ := (1 / Real.sqrt 2) * (cos (34 * π / 180) - sin (34 * π / 180))

noncomputable def b : ℝ := cos (50 * π / 180) * cos (128 * π / 180) + cos (40 * π / 180) * cos (38 * π / 180)

noncomputable def c : ℝ := (1 / 2) * (cos (80 * π / 180) - 2 * (cos (50 * π / 180))^2 + 1)

theorem abc_relationship : b > a ∧ a > c := by sorry

end abc_relationship_l3681_368118


namespace max_segment_length_squared_l3681_368108

/-- Circle with center O and radius r -/
structure Circle where
  O : ℝ × ℝ
  r : ℝ

/-- Line defined by two points -/
structure Line where
  P : ℝ × ℝ
  Q : ℝ × ℝ

/-- Point on a circle -/
def PointOnCircle (ω : Circle) (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  let (cx, cy) := ω.O
  (x - cx)^2 + (y - cy)^2 = ω.r^2

/-- Tangent line to a circle at a point -/
def TangentLine (ω : Circle) (T : ℝ × ℝ) (l : Line) : Prop :=
  PointOnCircle ω T ∧ 
  ∃ (P : ℝ × ℝ), P ≠ T ∧ PointOnCircle ω P ∧ 
    ((P.1 - T.1) * (l.Q.1 - l.P.1) + (P.2 - T.2) * (l.Q.2 - l.P.2) = 0)

/-- Perpendicular foot from a point to a line -/
def PerpendicularFoot (A : ℝ × ℝ) (l : Line) (P : ℝ × ℝ) : Prop :=
  (P.1 - A.1) * (l.Q.1 - l.P.1) + (P.2 - A.2) * (l.Q.2 - l.P.2) = 0 ∧
  ∃ (t : ℝ), P = (l.P.1 + t * (l.Q.1 - l.P.1), l.P.2 + t * (l.Q.2 - l.P.2))

/-- The main theorem -/
theorem max_segment_length_squared 
  (ω : Circle) 
  (A B C T : ℝ × ℝ) 
  (l : Line) 
  (P : ℝ × ℝ) :
  PointOnCircle ω A ∧ 
  PointOnCircle ω B ∧
  ω.r = 12 ∧
  (A.1 - ω.O.1)^2 + (A.2 - ω.O.2)^2 = (B.1 - ω.O.1)^2 + (B.2 - ω.O.2)^2 ∧
  (C.1 - A.1) * (B.1 - A.1) + (C.2 - A.2) * (B.2 - A.2) < 0 ∧
  TangentLine ω T l ∧
  PerpendicularFoot A l P →
  ∃ (m : ℝ), m^2 = 612 ∧ 
    ∀ (X : ℝ × ℝ), PointOnCircle ω X → 
      (X.1 - B.1)^2 + (X.2 - B.2)^2 ≤ m^2 := by
  sorry

end max_segment_length_squared_l3681_368108


namespace sandy_correct_sums_l3681_368114

theorem sandy_correct_sums : ℕ :=
  let total_sums : ℕ := 40
  let marks_per_correct : ℕ := 4
  let marks_lost_per_incorrect : ℕ := 3
  let total_marks : ℕ := 72
  let correct_sums : ℕ := 27
  let incorrect_sums : ℕ := total_sums - correct_sums

  have h1 : correct_sums + incorrect_sums = total_sums := by sorry
  have h2 : marks_per_correct * correct_sums - marks_lost_per_incorrect * incorrect_sums = total_marks := by sorry

  correct_sums

-- The proof is omitted

end sandy_correct_sums_l3681_368114


namespace sin_cos_sum_13_17_l3681_368106

theorem sin_cos_sum_13_17 : 
  Real.sin (13 * π / 180) * Real.cos (17 * π / 180) + 
  Real.cos (13 * π / 180) * Real.sin (17 * π / 180) = 1 / 2 := by
  sorry

end sin_cos_sum_13_17_l3681_368106


namespace james_marbles_distribution_l3681_368191

theorem james_marbles_distribution (initial_marbles : ℕ) (remaining_marbles : ℕ) 
  (h1 : initial_marbles = 28)
  (h2 : remaining_marbles = 21)
  (h3 : initial_marbles > remaining_marbles) :
  ∃ (num_bags : ℕ), 
    num_bags > 1 ∧ 
    (initial_marbles - remaining_marbles) * num_bags = initial_marbles ∧
    num_bags = 4 := by
  sorry

end james_marbles_distribution_l3681_368191


namespace expression_value_l3681_368142

theorem expression_value (a : ℝ) (h1 : 1 ≤ a) (h2 : a < 2) :
  Real.sqrt (a + 2 * Real.sqrt (a - 1)) + Real.sqrt (a - 2 * Real.sqrt (a - 1)) = 2 :=
by sorry

end expression_value_l3681_368142


namespace solve_system_of_equations_l3681_368174

theorem solve_system_of_equations (b : ℝ) : 
  (∃ x : ℝ, 2 * x + 7 = 3 ∧ b * x - 10 = -2) → b = -4 := by
  sorry

end solve_system_of_equations_l3681_368174


namespace twenty_fifth_in_base5_l3681_368102

/-- Converts a natural number to its base 5 representation -/
def toBase5 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 5) ((m % 5) :: acc)
    aux n []

/-- Checks if a list of digits is a valid base 5 number -/
def isValidBase5 (l : List ℕ) : Prop :=
  l.all (· < 5)

theorem twenty_fifth_in_base5 :
  let base5Repr := toBase5 25
  isValidBase5 base5Repr ∧ base5Repr = [1, 0, 0] := by sorry

end twenty_fifth_in_base5_l3681_368102


namespace distance_walked_l3681_368125

theorem distance_walked (x t d : ℝ) 
  (h1 : d = x * t) 
  (h2 : d = (x + 1/2) * (4/5 * t))
  (h3 : d = (x - 1/2) * (t + 5/2)) :
  d = 15 := by
  sorry

end distance_walked_l3681_368125


namespace num_organizations_in_foundation_l3681_368185

/-- The number of organizations in a public foundation --/
def num_organizations (total_raised : ℚ) (donation_percentage : ℚ) (amount_per_org : ℚ) : ℚ :=
  (total_raised * donation_percentage) / amount_per_org

/-- Theorem stating the number of organizations in the public foundation --/
theorem num_organizations_in_foundation : 
  num_organizations 2500 0.8 250 = 8 := by
  sorry

end num_organizations_in_foundation_l3681_368185


namespace min_value_a_l3681_368147

theorem min_value_a (a : ℝ) (h1 : a > 1) :
  (∀ x : ℝ, x ≥ 1/3 → (1/(3*x) - 2*x + Real.log (3*x) ≤ 1/(a*(Real.exp (2*x))) + Real.log a)) →
  a ≥ 3/(2*(Real.exp 1)) := by
  sorry

end min_value_a_l3681_368147


namespace limit_of_sequence_a_l3681_368151

def a (n : ℕ) : ℚ := (3 - n^2) / (4 + 2*n^2)

theorem limit_of_sequence_a :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |a n - (-1/2)| < ε :=
by sorry

end limit_of_sequence_a_l3681_368151


namespace school_report_mistake_l3681_368187

theorem school_report_mistake :
  ¬ ∃ (girls : ℕ), 
    let boys := girls + 373
    let total := girls + boys
    total = 3688 :=
by
  sorry

end school_report_mistake_l3681_368187


namespace rhombus_diagonal_l3681_368171

/-- Given a rhombus with area 432 sq m and one diagonal 24 m, prove the other diagonal is 36 m -/
theorem rhombus_diagonal (area : ℝ) (diagonal2 : ℝ) (diagonal1 : ℝ) : 
  area = 432 → diagonal2 = 24 → area = (diagonal1 * diagonal2) / 2 → diagonal1 = 36 :=
by sorry

end rhombus_diagonal_l3681_368171


namespace inverse_of_complex_l3681_368194

theorem inverse_of_complex (z : ℂ) : z = 1 - 2 * I → z⁻¹ = (1 / 5 : ℂ) + (2 / 5 : ℂ) * I := by
  sorry

end inverse_of_complex_l3681_368194


namespace inscribed_box_radius_l3681_368178

/-- A rectangular box inscribed in a sphere -/
structure InscribedBox where
  s : ℝ  -- radius of the sphere
  a : ℝ  -- length of the box
  b : ℝ  -- width of the box
  c : ℝ  -- height of the box

/-- The sum of the lengths of the 12 edges of the box -/
def edge_sum (box : InscribedBox) : ℝ := 4 * (box.a + box.b + box.c)

/-- The surface area of the box -/
def surface_area (box : InscribedBox) : ℝ := 2 * (box.a * box.b + box.b * box.c + box.c * box.a)

/-- The main theorem -/
theorem inscribed_box_radius (box : InscribedBox) 
  (h1 : edge_sum box = 72)
  (h2 : surface_area box = 216) :
  box.s = 3 * Real.sqrt 3 := by
  sorry

end inscribed_box_radius_l3681_368178


namespace james_weekly_earnings_l3681_368144

/-- Calculates the weekly earnings from car rental -/
def weekly_earnings (rate : ℝ) (hours_per_day : ℝ) (days_per_week : ℝ) : ℝ :=
  rate * hours_per_day * days_per_week

/-- Proof that James' weekly earnings from car rental are $640 -/
theorem james_weekly_earnings :
  weekly_earnings 20 8 4 = 640 := by
  sorry

end james_weekly_earnings_l3681_368144


namespace fraction_problem_l3681_368135

theorem fraction_problem (n : ℤ) : 
  (n : ℚ) / (4 * n - 5 : ℚ) = 3 / 7 → n = 3 := by
  sorry

end fraction_problem_l3681_368135


namespace fraction_equality_l3681_368133

theorem fraction_equality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + b)^2 / (a^2 + 2*a*b + b^2) = 1 := by
  sorry

end fraction_equality_l3681_368133


namespace building_shadow_length_l3681_368104

/-- Given a flagpole and a building under similar lighting conditions, 
    this theorem proves the length of the building's shadow. -/
theorem building_shadow_length 
  (flagpole_height : ℝ) 
  (flagpole_shadow : ℝ) 
  (building_height : ℝ) 
  (h1 : flagpole_height = 18) 
  (h2 : flagpole_shadow = 45) 
  (h3 : building_height = 28) : 
  ∃ (building_shadow : ℝ), building_shadow = 70 ∧ 
  flagpole_height / flagpole_shadow = building_height / building_shadow :=
sorry

end building_shadow_length_l3681_368104


namespace grandpa_grandchildren_ages_l3681_368198

theorem grandpa_grandchildren_ages (grandpa_age : ℕ) (gc1_age gc2_age gc3_age : ℕ) (years : ℕ) :
  grandpa_age = 75 →
  gc1_age = 13 →
  gc2_age = 15 →
  gc3_age = 17 →
  years = 15 →
  grandpa_age + years = (gc1_age + years) + (gc2_age + years) + (gc3_age + years) :=
by sorry

end grandpa_grandchildren_ages_l3681_368198


namespace sqrt_450_simplified_l3681_368117

theorem sqrt_450_simplified : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end sqrt_450_simplified_l3681_368117


namespace square_area_l3681_368145

/-- The parabola function -/
def parabola (x : ℝ) : ℝ := x^2 - 4*x + 3

/-- The line function -/
def line : ℝ := 8

theorem square_area : ∃ (x₁ x₂ : ℝ), 
  parabola x₁ = line ∧ 
  parabola x₂ = line ∧ 
  (x₂ - x₁)^2 = 36 := by
  sorry

end square_area_l3681_368145


namespace triangle_side_length_l3681_368197

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) (S : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  S = (1/2) * b * c * Real.sin A ∧
  S = Real.sqrt 3 ∧
  b = 1 ∧
  A = π/3 →
  a = Real.sqrt 13 := by
sorry

end triangle_side_length_l3681_368197


namespace horner_method_v3_horner_method_correct_l3681_368148

def horner_polynomial (x : ℝ) : ℝ := 12 + 35*x - 8*x^2 + 79*x^3 + 6*x^4 + 5*x^5 + 3*x^6

def horner_v3 (x : ℝ) : ℝ :=
  let v0 := 3
  let v1 := v0 * x + 5
  let v2 := v1 * x + 6
  v2 * x + 79

theorem horner_method_v3 :
  horner_v3 (-4) = -57 :=
by sorry

theorem horner_method_correct :
  horner_v3 (-4) = horner_polynomial (-4) :=
by sorry

end horner_method_v3_horner_method_correct_l3681_368148


namespace plant_branches_l3681_368162

theorem plant_branches : ∃ (x : ℕ), x > 0 ∧ 1 + x + x * x = 57 := by
  sorry

end plant_branches_l3681_368162


namespace correct_mean_after_error_fix_l3681_368159

/-- Given a set of values with an incorrect mean due to a misrecorded value,
    calculate the correct mean after fixing the error. -/
theorem correct_mean_after_error_fix (n : ℕ) (incorrect_mean : ℚ) (wrong_value correct_value : ℚ) 
    (h1 : n = 30)
    (h2 : incorrect_mean = 140)
    (h3 : wrong_value = 135)
    (h4 : correct_value = 145) :
    let total_sum := n * incorrect_mean
    let difference := correct_value - wrong_value
    let corrected_sum := total_sum + difference
    corrected_sum / n = 140333 / 1000 := by
  sorry

#eval (140333 : ℚ) / 1000  -- To verify the result is indeed 140.333

end correct_mean_after_error_fix_l3681_368159


namespace min_value_problem_l3681_368126

theorem min_value_problem (a b c d e f g h : ℝ) 
  (h1 : a * b * c * d = 9) 
  (h2 : e * f * g * h = 4) : 
  (a * e)^2 + (b * f)^2 + (c * g)^2 + (d * h)^2 ≥ 24 := by
  sorry

end min_value_problem_l3681_368126


namespace greatest_divisible_integer_l3681_368119

theorem greatest_divisible_integer (m : ℕ+) :
  (∃ (n : ℕ), n > 0 ∧ (m^2 + n) ∣ (n^2 + m)) ∧
  (∀ (k : ℕ), k > (m^4 - m^2 + m) → ¬((m^2 + k) ∣ (k^2 + m))) :=
by sorry

end greatest_divisible_integer_l3681_368119


namespace inscribed_squares_ratio_l3681_368155

theorem inscribed_squares_ratio : 
  let triangle1 : ℝ × ℝ × ℝ := (5, 12, 13)
  let triangle2 : ℝ × ℝ × ℝ := (5, 12, 13)
  let a := (60 : ℝ) / 17  -- side length of square in triangle1
  let b := (65 : ℝ) / 17  -- side length of square in triangle2
  (a^2) / (b^2) = 3600 / 4225 :=
by sorry

end inscribed_squares_ratio_l3681_368155


namespace max_water_bottles_proof_l3681_368167

/-- Given a total number of water bottles and athletes, with each athlete receiving at least one water bottle,
    calculate the maximum number of water bottles one athlete could have received. -/
def max_water_bottles (total_bottles : ℕ) (total_athletes : ℕ) : ℕ :=
  total_bottles - (total_athletes - 1)

/-- Prove that given 40 water bottles distributed among 25 athletes, with each athlete receiving at least one water bottle,
    the maximum number of water bottles one athlete could have received is 16. -/
theorem max_water_bottles_proof :
  max_water_bottles 40 25 = 16 := by
  sorry

#eval max_water_bottles 40 25

end max_water_bottles_proof_l3681_368167


namespace parsley_sprigs_remaining_l3681_368101

/-- Calculates the number of parsley sprigs remaining after decorating plates. -/
theorem parsley_sprigs_remaining 
  (initial_sprigs : ℕ) 
  (whole_sprig_plates : ℕ) 
  (half_sprig_plates : ℕ) : 
  initial_sprigs = 25 → 
  whole_sprig_plates = 8 → 
  half_sprig_plates = 12 → 
  initial_sprigs - (whole_sprig_plates + half_sprig_plates / 2) = 11 := by
  sorry

end parsley_sprigs_remaining_l3681_368101


namespace total_chicken_pieces_is_74_l3681_368140

/-- The number of chicken pieces needed for all orders at Clucks Delux -/
def total_chicken_pieces : ℕ :=
  let chicken_pasta_pieces : ℕ := 2
  let barbecue_chicken_pieces : ℕ := 4
  let fried_chicken_dinner_pieces : ℕ := 8
  let grilled_chicken_salad_pieces : ℕ := 1
  
  let fried_chicken_dinner_orders : ℕ := 4
  let chicken_pasta_orders : ℕ := 8
  let barbecue_chicken_orders : ℕ := 5
  let grilled_chicken_salad_orders : ℕ := 6

  (fried_chicken_dinner_pieces * fried_chicken_dinner_orders) +
  (chicken_pasta_pieces * chicken_pasta_orders) +
  (barbecue_chicken_pieces * barbecue_chicken_orders) +
  (grilled_chicken_salad_pieces * grilled_chicken_salad_orders)

theorem total_chicken_pieces_is_74 : total_chicken_pieces = 74 := by
  sorry

end total_chicken_pieces_is_74_l3681_368140


namespace library_visitors_theorem_l3681_368123

/-- Calculates the average number of visitors per day in a library for a 30-day month -/
def average_visitors (sundays : Nat) (sunday_visitors : Nat) (regular_visitors : Nat) (holiday_visitors : Nat) (holidays : Nat) : Nat :=
  let regular_days := 30 - sundays - holidays
  let total_visitors := sundays * sunday_visitors + regular_days * regular_visitors + holidays * holiday_visitors
  total_visitors / 30

/-- Theorem stating the average number of visitors for different scenarios -/
theorem library_visitors_theorem (sundays : Nat) (sunday_visitors : Nat) (regular_visitors : Nat) (holiday_visitors : Nat) (holidays : Nat) :
  (sundays = 4 ∨ sundays = 5) →
  sunday_visitors = 510 →
  regular_visitors = 240 →
  holiday_visitors = 375 →
  holidays = 2 →
  (average_visitors sundays sunday_visitors regular_visitors holiday_visitors holidays = 
    if sundays = 4 then 285 else 294) :=
by
  sorry

#eval average_visitors 4 510 240 375 2
#eval average_visitors 5 510 240 375 2

end library_visitors_theorem_l3681_368123


namespace sum_of_first_12_mod_9_l3681_368165

def sum_of_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

theorem sum_of_first_12_mod_9 : sum_of_first_n 12 % 9 = 6 := by
  sorry

end sum_of_first_12_mod_9_l3681_368165


namespace two_cones_cost_l3681_368169

/-- The cost of a single ice cream cone in cents -/
def single_cone_cost : ℕ := 99

/-- The number of ice cream cones -/
def num_cones : ℕ := 2

/-- Theorem: The cost of 2 ice cream cones is 198 cents -/
theorem two_cones_cost : single_cone_cost * num_cones = 198 := by
  sorry

end two_cones_cost_l3681_368169


namespace partition_6_5_l3681_368195

/-- The number of ways to partition n into at most k non-negative integer parts -/
def num_partitions (n k : ℕ) : ℕ := sorry

/-- The number of ways to partition 6 into at most 5 non-negative integer parts -/
theorem partition_6_5 : num_partitions 6 5 = 11 := by sorry

end partition_6_5_l3681_368195


namespace gold_alloy_calculation_l3681_368116

theorem gold_alloy_calculation (initial_weight : ℝ) (initial_gold_percentage : ℝ) 
  (target_gold_percentage : ℝ) (added_gold : ℝ) : 
  initial_weight = 16 →
  initial_gold_percentage = 0.5 →
  target_gold_percentage = 0.8 →
  added_gold = 24 →
  (initial_weight * initial_gold_percentage + added_gold) / (initial_weight + added_gold) = target_gold_percentage :=
by
  sorry

end gold_alloy_calculation_l3681_368116


namespace square_remainder_l3681_368186

theorem square_remainder (N : ℤ) : 
  (N % 5 = 3) → (N^2 % 5 = 4) := by
sorry

end square_remainder_l3681_368186


namespace fruit_purchase_price_l3681_368146

/-- The price of an orange in cents -/
def orange_price : ℕ := 3000

/-- The price of a pear in cents -/
def pear_price : ℕ := 9000

/-- The price of a banana in cents -/
def banana_price : ℕ := pear_price - orange_price

/-- The total cost of an orange and a pear in cents -/
def orange_pear_total : ℕ := orange_price + pear_price

/-- The total cost of 50 oranges and 25 bananas in cents -/
def fifty_orange_twentyfive_banana : ℕ := 50 * orange_price + 25 * banana_price

/-- The number of items purchased -/
def total_items : ℕ := 200 + 400

/-- The discount rate as a rational number -/
def discount_rate : ℚ := 1 / 10

theorem fruit_purchase_price :
  orange_pear_total = 12000 ∧
  fifty_orange_twentyfive_banana % 700 = 0 ∧
  total_items > 300 →
  (200 * banana_price + 400 * orange_price) * (1 - discount_rate) = 2160000 := by
  sorry

end fruit_purchase_price_l3681_368146


namespace arithmetic_sequence_problem_l3681_368121

theorem arithmetic_sequence_problem (n : ℕ) (min max sum : ℚ) (h_n : n = 150) 
  (h_min : min = 20) (h_max : max = 90) (h_sum : sum = 9000) :
  let avg := sum / n
  let d := (max - min) / (2 * (n - 1))
  let L := avg - (29 * d)
  let G := avg + (29 * d)
  G - L = 7140 / 149 := by
  sorry

end arithmetic_sequence_problem_l3681_368121


namespace car_both_ways_time_l3681_368138

/-- Represents the time in hours for different travel scenarios -/
structure TravelTime where
  mixedTrip : ℝ  -- Time for walking one way and taking a car back
  walkingBothWays : ℝ  -- Time for walking both ways
  carBothWays : ℝ  -- Time for taking a car both ways

/-- Proves that given the conditions, the time taken if taking a car both ways is 30 minutes -/
theorem car_both_ways_time (t : TravelTime) 
  (h1 : t.mixedTrip = 1.5)
  (h2 : t.walkingBothWays = 2.5) : 
  t.carBothWays * 60 = 30 := by
  sorry

end car_both_ways_time_l3681_368138

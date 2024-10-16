import Mathlib

namespace NUMINAMATH_CALUDE_harvest_duration_l1421_142164

/-- The number of weeks of harvest given total earnings and weekly earnings -/
def harvest_weeks (total_earnings weekly_earnings : ℕ) : ℕ :=
  total_earnings / weekly_earnings

/-- Theorem: The harvest lasted for 76 weeks -/
theorem harvest_duration :
  harvest_weeks 1216 16 = 76 := by
  sorry

end NUMINAMATH_CALUDE_harvest_duration_l1421_142164


namespace NUMINAMATH_CALUDE_M_mod_1000_eq_9_l1421_142179

/-- The number of 8-digit positive integers with strictly increasing digits -/
def M : ℕ := Nat.choose 9 8

/-- The theorem stating that M modulo 1000 equals 9 -/
theorem M_mod_1000_eq_9 : M % 1000 = 9 := by
  sorry

end NUMINAMATH_CALUDE_M_mod_1000_eq_9_l1421_142179


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1421_142128

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  (∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n) →  -- geometric sequence condition
  (a 1 + a 2 + a 3 = 8) →                    -- first condition
  (a 4 + a 5 + a 6 = -4) →                   -- second condition
  (a 7 + a 8 + a 9 = 2) :=                   -- conclusion to prove
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1421_142128


namespace NUMINAMATH_CALUDE_shekars_weighted_average_sum_of_weightages_is_one_l1421_142190

/-- Represents the subjects and their corresponding scores and weightages -/
structure Subject where
  name : String
  score : ℝ
  weightage : ℝ

/-- Calculates the weighted average of a list of subjects -/
def weightedAverage (subjects : List Subject) : ℝ :=
  (subjects.map (fun s => s.score * s.weightage)).sum

/-- Shekar's subjects with their scores and weightages -/
def shekarsSubjects : List Subject := [
  ⟨"Mathematics", 76, 0.15⟩,
  ⟨"Science", 65, 0.15⟩,
  ⟨"Social Studies", 82, 0.20⟩,
  ⟨"English", 67, 0.20⟩,
  ⟨"Biology", 75, 0.10⟩,
  ⟨"Computer Science", 89, 0.10⟩,
  ⟨"History", 71, 0.10⟩
]

/-- Theorem stating that Shekar's weighted average marks is 74.45 -/
theorem shekars_weighted_average :
  weightedAverage shekarsSubjects = 74.45 := by
  sorry

/-- Proof that the sum of weightages is 1 -/
theorem sum_of_weightages_is_one :
  (shekarsSubjects.map (fun s => s.weightage)).sum = 1 := by
  sorry

end NUMINAMATH_CALUDE_shekars_weighted_average_sum_of_weightages_is_one_l1421_142190


namespace NUMINAMATH_CALUDE_unique_stamp_denomination_l1421_142166

/-- A function that determines if a postage value can be formed with given stamp denominations -/
def can_form_postage (n : ℕ) (postage : ℕ) : Prop :=
  ∃ (a b c : ℕ), postage = 10 * a + n * b + (n + 1) * c

/-- The main theorem stating that 16 is the unique positive integer satisfying the conditions -/
theorem unique_stamp_denomination : 
  ∃! (n : ℕ), n > 0 ∧ 
    (¬ can_form_postage n 120) ∧ 
    (∀ m > 120, can_form_postage n m) ∧
    n = 16 :=
sorry

end NUMINAMATH_CALUDE_unique_stamp_denomination_l1421_142166


namespace NUMINAMATH_CALUDE_polynomial_B_value_l1421_142187

def polynomial (z A B C D : ℝ) : ℝ := z^6 - 12*z^5 + A*z^4 + B*z^3 + C*z^2 + D*z + 36

theorem polynomial_B_value (A B C D : ℝ) :
  (∃ (r₁ r₂ r₃ r₄ r₅ r₆ : ℕ+), 
    (∀ z : ℝ, polynomial z A B C D = (z - r₁) * (z - r₂) * (z - r₃) * (z - r₄) * (z - r₅) * (z - r₆)) ∧
    r₁ + r₂ + r₃ + r₄ + r₅ + r₆ = 12) →
  B = -122 := by
sorry

end NUMINAMATH_CALUDE_polynomial_B_value_l1421_142187


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l1421_142135

/-- The speed of a boat in still water, given downstream travel information and current speed. -/
theorem boat_speed_in_still_water
  (current_speed : ℝ)
  (downstream_distance : ℝ)
  (downstream_time : ℝ)
  (h1 : current_speed = 5)
  (h2 : downstream_distance = 5)
  (h3 : downstream_time = 1/5) :
  let downstream_speed := (boat_speed : ℝ) + current_speed
  downstream_distance = downstream_speed * downstream_time →
  boat_speed = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l1421_142135


namespace NUMINAMATH_CALUDE_unique_solution_diophantine_equation_l1421_142154

theorem unique_solution_diophantine_equation :
  ∀ a b c d : ℕ+,
    4^(a:ℕ) * 5^(b:ℕ) - 3^(c:ℕ) * 11^(d:ℕ) = 1 →
    a = 1 ∧ b = 2 ∧ c = 2 ∧ d = 1 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_diophantine_equation_l1421_142154


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l1421_142107

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (81 - 27*x - 3*x^2 = 0) → 
  (∃ r s : ℝ, (81 - 27*r - 3*r^2 = 0) ∧ (81 - 27*s - 3*s^2 = 0) ∧ (r + s = -9)) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l1421_142107


namespace NUMINAMATH_CALUDE_two_and_half_in_one_and_three_fourths_l1421_142124

theorem two_and_half_in_one_and_three_fourths : 
  (1 + 3/4) / (2 + 1/2) = 7/10 := by sorry

end NUMINAMATH_CALUDE_two_and_half_in_one_and_three_fourths_l1421_142124


namespace NUMINAMATH_CALUDE_extremum_point_implies_a_max_min_on_interval_not_monotone_increasing_l1421_142118

-- Define the function f(x) with parameter a
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 - 9*x - 1

-- Part I
theorem extremum_point_implies_a (a : ℝ) :
  (∃ ε > 0, ∀ x ∈ Set.Ioo (-1-ε) (-1+ε), f a x ≤ f a (-1) ∨ f a x ≥ f a (-1)) →
  a = -3 :=
sorry

-- Define f with a = -3
def f_fixed (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x - 1

theorem max_min_on_interval :
  (∀ x ∈ Set.Icc (-2) 5, f_fixed x ≤ 4) ∧
  (∃ x ∈ Set.Icc (-2) 5, f_fixed x = 4) ∧
  (∀ x ∈ Set.Icc (-2) 5, f_fixed x ≥ -28) ∧
  (∃ x ∈ Set.Icc (-2) 5, f_fixed x = -28) :=
sorry

-- Part II
theorem not_monotone_increasing (a : ℝ) :
  ¬(∀ x y : ℝ, x < y → f a x < f a y) :=
sorry

end NUMINAMATH_CALUDE_extremum_point_implies_a_max_min_on_interval_not_monotone_increasing_l1421_142118


namespace NUMINAMATH_CALUDE_sum_of_complex_unit_magnitude_l1421_142141

theorem sum_of_complex_unit_magnitude (a b c : ℂ) 
  (h1 : Complex.abs a = 1) 
  (h2 : Complex.abs b = 1) 
  (h3 : Complex.abs c = 1)
  (h4 : a^2 / (b*c) + b^2 / (a*c) + c^2 / (a*b) = 3)
  (h5 : a + b + c ≠ 0) :
  Complex.abs (a + b + c) = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_complex_unit_magnitude_l1421_142141


namespace NUMINAMATH_CALUDE_volume_of_specific_box_l1421_142130

/-- The volume of an open box constructed from a rectangular metal sheet. -/
def boxVolume (sheetLength sheetWidth y : ℝ) : ℝ :=
  (sheetLength - 2*y) * (sheetWidth - 2*y) * y

/-- Theorem stating the volume of the box for the given dimensions -/
theorem volume_of_specific_box (y : ℝ) :
  boxVolume 18 12 y = 4*y^3 - 60*y^2 + 216*y :=
by sorry

end NUMINAMATH_CALUDE_volume_of_specific_box_l1421_142130


namespace NUMINAMATH_CALUDE_complete_square_with_integer_l1421_142122

theorem complete_square_with_integer : 
  ∃ (k : ℤ) (b : ℝ), ∀ (x : ℝ), x^2 + 8*x + 20 = (x + b)^2 + k := by
  sorry

end NUMINAMATH_CALUDE_complete_square_with_integer_l1421_142122


namespace NUMINAMATH_CALUDE_probability_of_correct_answer_l1421_142104

theorem probability_of_correct_answer (options : Nat) (correct_options : Nat) : 
  options = 4 → correct_options = 1 → (correct_options : ℚ) / options = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_correct_answer_l1421_142104


namespace NUMINAMATH_CALUDE_log_inequality_implies_order_l1421_142158

theorem log_inequality_implies_order (x y : ℝ) :
  (Real.log x / Real.log (1/2)) < (Real.log y / Real.log (1/2)) ∧
  (Real.log y / Real.log (1/2)) < 0 →
  1 < y ∧ y < x :=
by sorry

end NUMINAMATH_CALUDE_log_inequality_implies_order_l1421_142158


namespace NUMINAMATH_CALUDE_equation_solution_l1421_142175

/-- The function f(x) = x + 4 -/
def f (x : ℝ) : ℝ := x + 4

/-- The equation to be solved -/
def equation (x : ℝ) : Prop :=
  (3 * f (x - 2)) / f 0 + 4 = f (2 * x + 1)

theorem equation_solution :
  ∃ x : ℝ, equation x ∧ x = 2 / 5 :=
sorry

end NUMINAMATH_CALUDE_equation_solution_l1421_142175


namespace NUMINAMATH_CALUDE_m_range_l1421_142121

-- Define the set A
def A (m : ℝ) : Set ℝ := {x | x^2 - 2*m*x + m^2 - 1 < 0}

-- Theorem statement
theorem m_range (m : ℝ) : 1 ∈ A m ∧ 3 ∉ A m → 0 < m ∧ m < 2 :=
by
  sorry

end NUMINAMATH_CALUDE_m_range_l1421_142121


namespace NUMINAMATH_CALUDE_unique_function_satisfying_conditions_l1421_142156

theorem unique_function_satisfying_conditions :
  ∃! f : ℝ → ℝ, 
    (∀ x : ℝ, f (x + 1) ≥ f x + 1) ∧ 
    (∀ x y : ℝ, f (x * y) ≥ f x * f y) ∧
    (∀ x : ℝ, f x = x) := by
  sorry

end NUMINAMATH_CALUDE_unique_function_satisfying_conditions_l1421_142156


namespace NUMINAMATH_CALUDE_kayla_apples_l1421_142117

theorem kayla_apples (total : ℕ) (kayla kylie : ℕ) : 
  total = 200 →
  kayla + kylie = total →
  kayla = kylie / 4 →
  kayla = 40 := by
sorry

end NUMINAMATH_CALUDE_kayla_apples_l1421_142117


namespace NUMINAMATH_CALUDE_pizza_payment_difference_l1421_142131

theorem pizza_payment_difference :
  -- Define the total number of slices
  let total_slices : ℕ := 12
  -- Define the cost of a plain pizza
  let plain_pizza_cost : ℚ := 12
  -- Define the additional cost for extra cheese
  let extra_cheese_cost : ℚ := 3
  -- Define the number of slices with extra cheese (one-third of the pizza)
  let extra_cheese_slices : ℕ := total_slices / 3
  -- Define the number of plain slices
  let plain_slices : ℕ := total_slices - extra_cheese_slices
  -- Define the total cost of the pizza
  let total_cost : ℚ := plain_pizza_cost + extra_cheese_cost
  -- Define the cost per slice
  let cost_per_slice : ℚ := total_cost / total_slices
  -- Define the number of slices Nancy ate
  let nancy_slices : ℕ := extra_cheese_slices + 3
  -- Define the number of slices Carol ate
  let carol_slices : ℕ := total_slices - nancy_slices
  -- Define Nancy's payment
  let nancy_payment : ℚ := cost_per_slice * nancy_slices
  -- Define Carol's payment
  let carol_payment : ℚ := cost_per_slice * carol_slices
  -- The theorem to prove
  nancy_payment - carol_payment = (5/2 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_pizza_payment_difference_l1421_142131


namespace NUMINAMATH_CALUDE_multiplicative_additive_function_theorem_l1421_142198

theorem multiplicative_additive_function_theorem :
  (¬ ∃ f : ℕ → ℕ, Function.Injective f ∧ ∀ m n : ℕ, f (m * n) = f m + f n) ∧
  (∀ k : ℕ+, ∃ f : Fin k → ℕ, Function.Injective f ∧
    ∀ m n : Fin k, (m.val * n.val : ℕ) ≤ k → f (⟨m.val * n.val, sorry⟩) = f m + f n) :=
by sorry

end NUMINAMATH_CALUDE_multiplicative_additive_function_theorem_l1421_142198


namespace NUMINAMATH_CALUDE_multiplication_commutative_l1421_142192

theorem multiplication_commutative (a b : ℝ) : a * b = b * a := by
  sorry

end NUMINAMATH_CALUDE_multiplication_commutative_l1421_142192


namespace NUMINAMATH_CALUDE_complement_of_union_equals_five_l1421_142147

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5}

-- Define set M
def M : Set Nat := {1, 2}

-- Define set N
def N : Set Nat := {3, 4}

-- Theorem statement
theorem complement_of_union_equals_five : 
  (U \ (M ∪ N)) = {5} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_equals_five_l1421_142147


namespace NUMINAMATH_CALUDE_peter_pizza_fraction_l1421_142127

/-- Given a pizza with 16 slices, calculate the fraction eaten by Peter -/
theorem peter_pizza_fraction :
  let total_slices : ℕ := 16
  let whole_slices_eaten : ℕ := 2
  let shared_slice : ℚ := 1/2
  (whole_slices_eaten : ℚ) / total_slices + shared_slice / total_slices = 5/32 := by
  sorry

end NUMINAMATH_CALUDE_peter_pizza_fraction_l1421_142127


namespace NUMINAMATH_CALUDE_completing_square_transformation_l1421_142137

theorem completing_square_transformation (x : ℝ) : 
  (x^2 - 2*x - 5 = 0) ↔ ((x - 1)^2 = 6) := by
  sorry

end NUMINAMATH_CALUDE_completing_square_transformation_l1421_142137


namespace NUMINAMATH_CALUDE_triangle_shortest_side_l1421_142136

theorem triangle_shortest_side 
  (a b c : ℕ) 
  (h : ℕ) 
  (area : ℕ) 
  (ha : a = 24) 
  (hperim : a + b + c = 55) 
  (harea : area = a * h / 2) 
  (hherons : area^2 = ((a + b + c) / 2) * (((a + b + c) / 2) - a) * (((a + b + c) / 2) - b) * (((a + b + c) / 2) - c)) : 
  min b c = 14 := by
sorry

end NUMINAMATH_CALUDE_triangle_shortest_side_l1421_142136


namespace NUMINAMATH_CALUDE_infinitely_many_pairs_smallest_pair_l1421_142176

/-- Predicate defining the conditions for x and y -/
def satisfies_conditions (x y : ℕ+) : Prop :=
  (x * (x + 1) ∣ y * (y + 1)) ∧
  ¬(x ∣ y) ∧
  ¬((x + 1) ∣ y) ∧
  ¬(x ∣ (y + 1)) ∧
  ¬((x + 1) ∣ (y + 1))

/-- There exist infinitely many pairs of positive integers satisfying the conditions -/
theorem infinitely_many_pairs :
  ∀ n : ℕ, ∃ x y : ℕ+, x > n ∧ y > n ∧ satisfies_conditions x y :=
sorry

/-- The smallest pair satisfying the conditions is (14, 20) -/
theorem smallest_pair :
  satisfies_conditions 14 20 ∧
  ∀ x y : ℕ+, satisfies_conditions x y → x ≥ 14 ∧ y ≥ 20 :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_pairs_smallest_pair_l1421_142176


namespace NUMINAMATH_CALUDE_chris_dana_distance_difference_l1421_142178

/-- The difference in distance traveled between two bikers after a given time -/
def distance_difference (speed1 : ℝ) (speed2 : ℝ) (time : ℝ) : ℝ :=
  (speed1 * time) - (speed2 * time)

/-- Theorem stating the difference in distance traveled between Chris and Dana -/
theorem chris_dana_distance_difference :
  distance_difference 17 12 6 = 30 := by
  sorry

end NUMINAMATH_CALUDE_chris_dana_distance_difference_l1421_142178


namespace NUMINAMATH_CALUDE_tournament_games_theorem_l1421_142129

/-- Represents a single-elimination tournament. -/
structure Tournament where
  num_teams : ℕ
  single_elimination : Bool
  no_ties : Bool

/-- Calculates the number of games needed to determine a winner in a tournament. -/
def games_to_winner (t : Tournament) : ℕ :=
  t.num_teams - 1

/-- The theorem stating that a single-elimination tournament with 23 teams and no ties requires 22 games to determine a winner. -/
theorem tournament_games_theorem (t : Tournament) 
  (h1 : t.num_teams = 23) 
  (h2 : t.single_elimination = true) 
  (h3 : t.no_ties = true) : 
  games_to_winner t = 22 := by
  sorry


end NUMINAMATH_CALUDE_tournament_games_theorem_l1421_142129


namespace NUMINAMATH_CALUDE_complex_power_sum_l1421_142138

theorem complex_power_sum (z : ℂ) (h : z + (1 / z) = 2 * Real.cos (5 * π / 180)) :
  z^1000 + (1 / z^1000) = 2 * Real.cos (20 * π / 180) :=
by sorry

end NUMINAMATH_CALUDE_complex_power_sum_l1421_142138


namespace NUMINAMATH_CALUDE_sum_of_specific_numbers_l1421_142194

theorem sum_of_specific_numbers : 
  15.58 + 21.32 + 642.51 + 51.51 = 730.92 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_specific_numbers_l1421_142194


namespace NUMINAMATH_CALUDE_tree_planting_schedule_l1421_142197

theorem tree_planting_schedule (total_trees : ℕ) (days_saved : ℕ) : 
  total_trees = 960 →
  days_saved = 4 →
  ∃ (original_plan : ℕ),
    original_plan = 120 ∧
    (total_trees / original_plan) - (total_trees / (2 * original_plan)) = days_saved :=
by
  sorry

end NUMINAMATH_CALUDE_tree_planting_schedule_l1421_142197


namespace NUMINAMATH_CALUDE_problem_solution_l1421_142144

theorem problem_solution (a b : ℚ) 
  (eq1 : 8*a + 3*b = -1)
  (eq2 : a = b - 3) : 
  5*b = 115/11 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1421_142144


namespace NUMINAMATH_CALUDE_power_function_classification_l1421_142170

/-- Definition of a power function -/
def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ (a : ℝ), ∀ (x : ℝ), f x = x ^ a

/-- The given functions -/
def f1 (x : ℝ) : ℝ := x^2 + 2
def f2 (x : ℝ) : ℝ := x^(1/2)
def f3 (x : ℝ) : ℝ := 2 * x^3
def f4 (x : ℝ) : ℝ := x^(3/4)
def f5 (x : ℝ) : ℝ := x^(1/3) + 1

/-- The theorem stating which functions are power functions -/
theorem power_function_classification :
  ¬(is_power_function f1) ∧
  (is_power_function f2) ∧
  ¬(is_power_function f3) ∧
  (is_power_function f4) ∧
  ¬(is_power_function f5) :=
sorry

end NUMINAMATH_CALUDE_power_function_classification_l1421_142170


namespace NUMINAMATH_CALUDE_other_divisor_problem_l1421_142134

theorem other_divisor_problem (n : Nat) (d1 d2 : Nat) : 
  (n = 386) →
  (d1 = 35) →
  (n % d1 = 1) →
  (n % d2 = 1) →
  (∀ m : Nat, m < n → (m % d1 = 1 ∧ m % d2 = 1) → False) →
  (d2 = 11) := by
  sorry

end NUMINAMATH_CALUDE_other_divisor_problem_l1421_142134


namespace NUMINAMATH_CALUDE_red_button_probability_main_theorem_l1421_142160

/-- Represents a jar containing buttons of different colors -/
structure Jar :=
  (red : ℕ)
  (blue : ℕ)

/-- Calculates the total number of buttons in a jar -/
def Jar.total (j : Jar) : ℕ := j.red + j.blue

/-- Represents the state of the jars after Carla's action -/
structure JarState :=
  (jarA : Jar)
  (jarB : Jar)

/-- The probability of selecting a red button from a jar -/
def redProbability (j : Jar) : ℚ :=
  j.red / j.total

/-- The initial state of Jar A -/
def initialJarA : Jar := ⟨6, 10⟩

/-- Theorem stating the probability of selecting red buttons from both jars -/
theorem red_button_probability (state : JarState) : 
  redProbability state.jarA * redProbability state.jarB = 1/6 :=
sorry

/-- Main theorem combining all conditions and the result -/
theorem main_theorem (state : JarState) :
  initialJarA.total = 16 →
  state.jarA.total = (3/4 : ℚ) * initialJarA.total →
  state.jarB.total = initialJarA.total - state.jarA.total →
  state.jarB.red = state.jarB.blue →
  state.jarA.red + state.jarB.red = initialJarA.red →
  state.jarA.blue + state.jarB.blue = initialJarA.blue →
  redProbability state.jarA * redProbability state.jarB = 1/6 :=
sorry

end NUMINAMATH_CALUDE_red_button_probability_main_theorem_l1421_142160


namespace NUMINAMATH_CALUDE_cloth_selling_price_l1421_142191

theorem cloth_selling_price 
  (meters : ℕ) 
  (profit_per_meter : ℕ) 
  (cost_price_per_meter : ℕ) 
  (h1 : meters = 75)
  (h2 : profit_per_meter = 15)
  (h3 : cost_price_per_meter = 51) :
  meters * (cost_price_per_meter + profit_per_meter) = 4950 := by
  sorry

end NUMINAMATH_CALUDE_cloth_selling_price_l1421_142191


namespace NUMINAMATH_CALUDE_snail_max_distance_l1421_142149

/-- Represents the movement of a snail over time -/
structure SnailMovement where
  /-- The total observation time in hours -/
  total_time : ℝ
  /-- The observation duration of each observer in hours -/
  observer_duration : ℝ
  /-- The distance traveled during each observation in meters -/
  distance_per_observation : ℝ
  /-- Ensures there is always at least one observer -/
  always_observed : Prop

/-- The maximum distance the snail can travel given the conditions -/
def max_distance (sm : SnailMovement) : ℝ :=
  18

/-- Theorem stating the maximum distance the snail can travel is 18 meters -/
theorem snail_max_distance (sm : SnailMovement) 
    (h1 : sm.total_time = 10)
    (h2 : sm.observer_duration = 1)
    (h3 : sm.distance_per_observation = 1)
    (h4 : sm.always_observed) : 
  max_distance sm = 18 := by
  sorry

end NUMINAMATH_CALUDE_snail_max_distance_l1421_142149


namespace NUMINAMATH_CALUDE_contrapositive_true_l1421_142139

theorem contrapositive_true : 
  (∀ x : ℝ, (x^2 ≤ 0 → x ≥ 0)) := by sorry

end NUMINAMATH_CALUDE_contrapositive_true_l1421_142139


namespace NUMINAMATH_CALUDE_base_seven_subtraction_l1421_142113

/-- Represents a number in base 7 as a list of digits (least significant first) -/
def BaseSevenNum := List Nat

/-- Converts a base 7 number to its decimal representation -/
def to_decimal (n : BaseSevenNum) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- Subtracts two base 7 numbers -/
def base_seven_sub (a b : BaseSevenNum) : BaseSevenNum :=
  sorry -- Implementation details omitted

theorem base_seven_subtraction :
  let a : BaseSevenNum := [3, 3, 3, 2]  -- 2333 in base 7
  let b : BaseSevenNum := [1, 1, 1, 1]  -- 1111 in base 7
  let result : BaseSevenNum := [2, 2, 2, 1]  -- 1222 in base 7
  base_seven_sub a b = result :=
by sorry

end NUMINAMATH_CALUDE_base_seven_subtraction_l1421_142113


namespace NUMINAMATH_CALUDE_smallest_of_five_consecutive_sum_100_l1421_142145

theorem smallest_of_five_consecutive_sum_100 (n : ℕ) :
  (n + (n + 1) + (n + 2) + (n + 3) + (n + 4) = 100) → n = 18 := by
  sorry

end NUMINAMATH_CALUDE_smallest_of_five_consecutive_sum_100_l1421_142145


namespace NUMINAMATH_CALUDE_pencils_bought_l1421_142193

-- Define the cost of a single pencil and notebook
variable (P N : ℝ)

-- Define the number of pencils in the second case
variable (X : ℝ)

-- Conditions from the problem
axiom cost_condition1 : 96 * P + 24 * N = 520
axiom cost_condition2 : X * P + 4 * N = 60
axiom sum_condition : P + N = 15.512820512820513

-- Theorem to prove
theorem pencils_bought : X = 3 := by
  sorry

end NUMINAMATH_CALUDE_pencils_bought_l1421_142193


namespace NUMINAMATH_CALUDE_g_difference_l1421_142169

/-- The function g defined as g(x) = 6x^2 + 3x - 4 -/
def g (x : ℝ) : ℝ := 6 * x^2 + 3 * x - 4

/-- Theorem stating that g(x+h) - g(x) = h(12x + 6h + 3) for all real x and h -/
theorem g_difference (x h : ℝ) : g (x + h) - g x = h * (12 * x + 6 * h + 3) := by
  sorry

end NUMINAMATH_CALUDE_g_difference_l1421_142169


namespace NUMINAMATH_CALUDE_polynomial_non_real_root_l1421_142110

theorem polynomial_non_real_root (q : ℝ) : 
  ∃ (z : ℂ), z.im ≠ 0 ∧ z^4 - 2*q*z^3 - z^2 - 2*q*z + 1 = 0 := by sorry

end NUMINAMATH_CALUDE_polynomial_non_real_root_l1421_142110


namespace NUMINAMATH_CALUDE_inequality_proof_l1421_142119

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (h_abcd : a * b * c * d = 1) :
  (a * b + 1) / (a + 1) + (b * c + 1) / (b + 1) + (c * d + 1) / (c + 1) + (d * a + 1) / (d + 1) ≥ 4 ∧
  ((a * b + 1) / (a + 1) + (b * c + 1) / (b + 1) + (c * d + 1) / (c + 1) + (d * a + 1) / (d + 1) = 4 ↔ a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1421_142119


namespace NUMINAMATH_CALUDE_infinitely_many_composites_l1421_142142

def last_digit (n : ℕ) : ℕ := n % 10

def remove_last_digit (n : ℕ) : ℕ := n / 10

def is_composite (n : ℕ) : Prop := n > 1 ∧ ∃ m, 1 < m ∧ m < n ∧ n % m = 0

def sequence_property (p : ℕ → ℕ) : Prop :=
  ∀ i : ℕ, last_digit (p (i + 1)) ≠ 9 ∧ remove_last_digit (p (i + 1)) = p i

theorem infinitely_many_composites (p : ℕ → ℕ) (h : sequence_property p) :
  ∀ N : ℕ, ∃ n : ℕ, n > N ∧ is_composite (p n) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_composites_l1421_142142


namespace NUMINAMATH_CALUDE_negation_of_inequality_negation_of_specific_inequality_l1421_142180

theorem negation_of_inequality (P : ℝ → Prop) :
  (¬ ∀ x > 0, P x) ↔ (∃ x > 0, ¬ P x) :=
by
  sorry

theorem negation_of_specific_inequality :
  (¬ ∀ x > 0, x + 1/x ≥ 2) ↔ (∃ x > 0, x + 1/x < 2) :=
by
  sorry

end NUMINAMATH_CALUDE_negation_of_inequality_negation_of_specific_inequality_l1421_142180


namespace NUMINAMATH_CALUDE_exam_average_l1421_142151

theorem exam_average (successful_count unsuccessful_count : ℕ)
                     (successful_avg unsuccessful_avg : ℚ)
                     (h1 : successful_count = 20)
                     (h2 : unsuccessful_count = 20)
                     (h3 : successful_avg = 42)
                     (h4 : unsuccessful_avg = 38) :
  let total_count := successful_count + unsuccessful_count
  let total_points := successful_count * successful_avg + unsuccessful_count * unsuccessful_avg
  total_points / total_count = 40 := by
sorry

end NUMINAMATH_CALUDE_exam_average_l1421_142151


namespace NUMINAMATH_CALUDE_new_person_weight_is_105_l1421_142105

/-- The weight of the new person given the conditions of the problem -/
def weight_of_new_person (initial_count : ℕ) (leaving_weight : ℝ) (average_increase : ℝ) : ℝ :=
  leaving_weight + initial_count * average_increase

/-- Theorem stating that under the given conditions, the weight of the new person is 105 kg -/
theorem new_person_weight_is_105 :
  weight_of_new_person 8 85 2.5 = 105 := by
  sorry

#eval weight_of_new_person 8 85 2.5

end NUMINAMATH_CALUDE_new_person_weight_is_105_l1421_142105


namespace NUMINAMATH_CALUDE_additional_dividend_calculation_l1421_142155

/-- Calculates the additional dividend per share given expected and actual earnings -/
def additional_dividend (expected_earnings : ℚ) (actual_earnings : ℚ) : ℚ :=
  let earnings_difference := actual_earnings - expected_earnings
  let additional_earnings := max earnings_difference 0
  additional_earnings / 2

/-- Proves that the additional dividend is $0.15 per share given the problem conditions -/
theorem additional_dividend_calculation :
  let expected_earnings : ℚ := 80 / 100
  let actual_earnings : ℚ := 110 / 100
  additional_dividend expected_earnings actual_earnings = 15 / 100 := by
  sorry


end NUMINAMATH_CALUDE_additional_dividend_calculation_l1421_142155


namespace NUMINAMATH_CALUDE_skyler_song_composition_l1421_142100

/-- Represents the success levels of songs --/
inductive SuccessLevel
  | ExtremelySuccessful
  | Successful
  | ModeratelySuccessful
  | LessSuccessful
  | Unreleased

/-- Represents Skyler's song composition --/
structure SongComposition where
  hitSongs : Nat
  top100Songs : Nat
  unreleasedSongs : Nat
  duetsTop20 : Nat
  duetsBelow200 : Nat
  soundtracksExtremely : Nat
  soundtracksModerate : Nat
  soundtracksLukewarm : Nat
  internationalGlobal : Nat
  internationalRegional : Nat
  internationalOverlooked : Nat

/-- Calculates the total number of songs --/
def totalSongs (composition : SongComposition) : Nat :=
  composition.hitSongs + composition.top100Songs + composition.unreleasedSongs +
  composition.duetsTop20 + composition.duetsBelow200 +
  composition.soundtracksExtremely + composition.soundtracksModerate + composition.soundtracksLukewarm +
  composition.internationalGlobal + composition.internationalRegional + composition.internationalOverlooked

/-- Calculates the number of songs for each success level --/
def songsBySuccessLevel (composition : SongComposition) : SuccessLevel → Nat
  | SuccessLevel.ExtremelySuccessful => composition.hitSongs + composition.internationalGlobal
  | SuccessLevel.Successful => composition.top100Songs + composition.duetsTop20 + composition.soundtracksExtremely
  | SuccessLevel.ModeratelySuccessful => composition.soundtracksModerate + composition.internationalRegional
  | SuccessLevel.LessSuccessful => composition.soundtracksLukewarm + composition.internationalOverlooked + composition.duetsBelow200
  | SuccessLevel.Unreleased => composition.unreleasedSongs

/-- Theorem stating the total number of songs and their success level breakdown --/
theorem skyler_song_composition :
  ∃ (composition : SongComposition),
    composition.hitSongs = 25 ∧
    composition.top100Songs = composition.hitSongs + 10 ∧
    composition.unreleasedSongs = composition.hitSongs - 5 ∧
    composition.duetsTop20 = 6 ∧
    composition.duetsBelow200 = 6 ∧
    composition.soundtracksExtremely = 3 ∧
    composition.soundtracksModerate = 8 ∧
    composition.soundtracksLukewarm = 7 ∧
    composition.internationalGlobal = 1 ∧
    composition.internationalRegional = 7 ∧
    composition.internationalOverlooked = 14 ∧
    totalSongs composition = 132 ∧
    songsBySuccessLevel composition SuccessLevel.ExtremelySuccessful = 26 ∧
    songsBySuccessLevel composition SuccessLevel.Successful = 44 ∧
    songsBySuccessLevel composition SuccessLevel.ModeratelySuccessful = 15 ∧
    songsBySuccessLevel composition SuccessLevel.LessSuccessful = 27 ∧
    songsBySuccessLevel composition SuccessLevel.Unreleased = 20 := by
  sorry

end NUMINAMATH_CALUDE_skyler_song_composition_l1421_142100


namespace NUMINAMATH_CALUDE_arithmetic_equations_correctness_l1421_142185

theorem arithmetic_equations_correctness : 
  (-2 + 8 ≠ 10) ∧ 
  (-1 - 3 = -4) ∧ 
  (-2 * 2 ≠ 4) ∧ 
  (-8 / -1 ≠ -1/8) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_equations_correctness_l1421_142185


namespace NUMINAMATH_CALUDE_square_plus_minus_one_divisible_by_five_l1421_142184

theorem square_plus_minus_one_divisible_by_five (a : ℤ) : 
  ¬(5 ∣ a) → (5 ∣ (a^2 + 1)) ∨ (5 ∣ (a^2 - 1)) :=
by sorry

end NUMINAMATH_CALUDE_square_plus_minus_one_divisible_by_five_l1421_142184


namespace NUMINAMATH_CALUDE_tangent_line_triangle_area_l1421_142125

-- Define the function f(x) = x^3 - x + 1
def f (x : ℝ) : ℝ := x^3 - x + 1

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3 * x^2 - 1

-- Theorem statement
theorem tangent_line_triangle_area :
  let x₀ : ℝ := 0
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  let tangent_line (x : ℝ) : ℝ := m * (x - x₀) + y₀
  let x_intercept : ℝ := -y₀ / m + x₀
  let y_intercept : ℝ := tangent_line 0
  (1/2) * x_intercept * y_intercept = 1/2 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_triangle_area_l1421_142125


namespace NUMINAMATH_CALUDE_flea_jump_l1421_142109

def jump_sequence (start : ℤ) : ℕ → ℤ
  | 0 => start
  | n + 1 => 
    if n % 2 = 0 then
      jump_sequence start n - (n + 1)
    else
      jump_sequence start n + (n + 1)

theorem flea_jump (start : ℤ) : 
  jump_sequence start 100 = 20 → start = -30 := by
  sorry

end NUMINAMATH_CALUDE_flea_jump_l1421_142109


namespace NUMINAMATH_CALUDE_probability_at_least_one_red_l1421_142103

-- Define the contents of each box
def box_contents : ℕ := 3

-- Define the number of red balls in each box
def red_balls : ℕ := 2

-- Define the number of white balls in each box
def white_balls : ℕ := 1

-- Define the total number of possible outcomes
def total_outcomes : ℕ := box_contents * box_contents

-- Define the number of outcomes with no red balls
def no_red_outcomes : ℕ := white_balls * white_balls

-- State the theorem
theorem probability_at_least_one_red :
  (1 : ℚ) - (no_red_outcomes : ℚ) / (total_outcomes : ℚ) = 8 / 9 :=
sorry

end NUMINAMATH_CALUDE_probability_at_least_one_red_l1421_142103


namespace NUMINAMATH_CALUDE_complex_equation_product_l1421_142126

theorem complex_equation_product (a b : ℝ) (i : ℂ) :
  i * i = -1 →
  a + b * i = 5 / (1 + 2 * i) →
  a * b = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_product_l1421_142126


namespace NUMINAMATH_CALUDE_scientific_notation_million_l1421_142150

theorem scientific_notation_million (x : ℝ) (h : x = 1464.3) :
  x * (10 : ℝ)^6 = 1.4643 * (10 : ℝ)^7 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_million_l1421_142150


namespace NUMINAMATH_CALUDE_mary_baseball_cards_l1421_142148

theorem mary_baseball_cards :
  ∀ (initial_cards torn_cards fred_cards bought_cards : ℕ),
    initial_cards = 18 →
    torn_cards = 8 →
    fred_cards = 26 →
    bought_cards = 40 →
    initial_cards - torn_cards + fred_cards + bought_cards = 76 :=
by
  sorry

end NUMINAMATH_CALUDE_mary_baseball_cards_l1421_142148


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainder_one_l1421_142120

theorem smallest_integer_with_remainder_one : ∃ n : ℕ, 
  n > 1 ∧ 
  n % 4 = 1 ∧ 
  n % 5 = 1 ∧ 
  n % 6 = 1 ∧ 
  (∀ m : ℕ, m > 1 → m % 4 = 1 → m % 5 = 1 → m % 6 = 1 → n ≤ m) ∧
  n = 61 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainder_one_l1421_142120


namespace NUMINAMATH_CALUDE_michaels_house_paint_area_l1421_142143

/-- Represents the dimensions of a room -/
structure RoomDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the total area to be painted in a house -/
def totalPaintArea (numRooms : ℕ) (dimensions : RoomDimensions) (windowDoorArea : ℝ) : ℝ :=
  let wallArea := 2 * (dimensions.length * dimensions.height + dimensions.width * dimensions.height)
  let paintableArea := wallArea - windowDoorArea
  numRooms * paintableArea

/-- Theorem: The total area to be painted in Michael's house is 1600 square feet -/
theorem michaels_house_paint_area :
  let dimensions : RoomDimensions := ⟨14, 11, 9⟩
  totalPaintArea 4 dimensions 50 = 1600 := by sorry

end NUMINAMATH_CALUDE_michaels_house_paint_area_l1421_142143


namespace NUMINAMATH_CALUDE_tenRowTrianglePieces_l1421_142171

/-- Calculates the sum of an arithmetic sequence -/
def arithmeticSum (a1 n : ℕ) : ℕ := n * (2 * a1 + (n - 1)) / 2

/-- Represents a triangle structure with rods and connectors -/
structure Triangle where
  rows : ℕ
  rodSequence : ℕ → ℕ
  connectorSequence : ℕ → ℕ

/-- Calculates the total number of pieces in the triangle -/
def totalPieces (t : Triangle) : ℕ :=
  (arithmeticSum (t.rodSequence 1) t.rows) + (arithmeticSum (t.connectorSequence 1) (t.rows + 1))

/-- The specific 10-row triangle described in the problem -/
def tenRowTriangle : Triangle :=
  { rows := 10
  , rodSequence := fun n => 3 * n
  , connectorSequence := fun n => n }

/-- Theorem stating that the total number of pieces in the 10-row triangle is 231 -/
theorem tenRowTrianglePieces : totalPieces tenRowTriangle = 231 := by
  sorry

end NUMINAMATH_CALUDE_tenRowTrianglePieces_l1421_142171


namespace NUMINAMATH_CALUDE_elevator_movement_l1421_142183

/-- Represents the number of floors in a building --/
def TotalFloors : ℕ := 13

/-- Represents the initial floor of the elevator --/
def InitialFloor : ℕ := 9

/-- Represents the first upward movement of the elevator --/
def FirstUpwardMovement : ℕ := 3

/-- Represents the second upward movement of the elevator --/
def SecondUpwardMovement : ℕ := 8

/-- Represents the final floor of the elevator (top floor) --/
def FinalFloor : ℕ := 13

theorem elevator_movement (x : ℕ) : 
  InitialFloor - x + FirstUpwardMovement + SecondUpwardMovement = FinalFloor → 
  x = 7 := by
sorry

end NUMINAMATH_CALUDE_elevator_movement_l1421_142183


namespace NUMINAMATH_CALUDE_max_value_5x_3y_l1421_142167

theorem max_value_5x_3y (x y : ℝ) (h : x^2 + y^2 = 10*x + 8*y + 10) :
  ∃ (M : ℝ), M = 105 ∧ 5*x + 3*y ≤ M ∧ ∃ (x₀ y₀ : ℝ), x₀^2 + y₀^2 = 10*x₀ + 8*y₀ + 10 ∧ 5*x₀ + 3*y₀ = M :=
sorry

end NUMINAMATH_CALUDE_max_value_5x_3y_l1421_142167


namespace NUMINAMATH_CALUDE_square_root_fraction_simplification_l1421_142114

theorem square_root_fraction_simplification :
  (Real.sqrt (7^2 + 24^2)) / (Real.sqrt (49 + 16)) = 5 * Real.sqrt 65 / 13 := by
  sorry

end NUMINAMATH_CALUDE_square_root_fraction_simplification_l1421_142114


namespace NUMINAMATH_CALUDE_subtraction_of_like_terms_l1421_142132

theorem subtraction_of_like_terms (a : ℝ) : 3 * a - a = 2 * a := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_like_terms_l1421_142132


namespace NUMINAMATH_CALUDE_even_function_range_l1421_142123

def f (a b x : ℝ) : ℝ := a * x^2 + b * x + 2

theorem even_function_range (a b : ℝ) :
  (∀ x ∈ Set.Icc (1 + a) 2, f a b x = f a b (-x)) →
  (∃ y ∈ Set.Icc (1 + a) 2, -y ∈ Set.Icc (1 + a) 2) →
  (Set.range (f a b) = Set.Icc (-10) 2) :=
sorry

end NUMINAMATH_CALUDE_even_function_range_l1421_142123


namespace NUMINAMATH_CALUDE_phil_final_quarters_l1421_142152

/-- Calculates the number of quarters Phil has after four years of collecting and losing some. -/
def phil_quarters : ℕ :=
  let initial := 50
  let after_first_year := initial * 2
  let second_year_collection := 3 * 12
  let third_year_collection := 12 / 3
  let total_before_loss := after_first_year + second_year_collection + third_year_collection
  let quarters_lost := total_before_loss / 4
  total_before_loss - quarters_lost

/-- Theorem stating that Phil ends up with 105 quarters after four years. -/
theorem phil_final_quarters : phil_quarters = 105 := by
  sorry

end NUMINAMATH_CALUDE_phil_final_quarters_l1421_142152


namespace NUMINAMATH_CALUDE_balls_placement_count_l1421_142111

-- Define the number of balls and boxes
def num_balls : ℕ := 4
def num_boxes : ℕ := 4

-- Define the function to calculate the number of ways to place the balls
def place_balls : ℕ := sorry

-- Theorem statement
theorem balls_placement_count :
  place_balls = 144 := by sorry

end NUMINAMATH_CALUDE_balls_placement_count_l1421_142111


namespace NUMINAMATH_CALUDE_square_field_area_l1421_142182

/-- Proves that a square field with specific barbed wire conditions has an area of 27889 square meters -/
theorem square_field_area (wire_cost_per_meter : ℝ) (total_cost : ℝ) (gate_width : ℝ) (num_gates : ℕ) :
  wire_cost_per_meter = 2.0 →
  total_cost = 1332 →
  gate_width = 1 →
  num_gates = 2 →
  ∃ (side_length : ℝ),
    side_length > 0 ∧
    total_cost = wire_cost_per_meter * (4 * side_length - (↑num_gates * gate_width)) ∧
    side_length^2 = 27889 :=
by sorry

end NUMINAMATH_CALUDE_square_field_area_l1421_142182


namespace NUMINAMATH_CALUDE_sum_and_one_known_l1421_142102

theorem sum_and_one_known (x y : ℤ) : x + y = -26 ∧ x = 11 → y = -37 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_one_known_l1421_142102


namespace NUMINAMATH_CALUDE_number_ratio_l1421_142140

/-- Given three numbers satisfying specific conditions, prove their ratio -/
theorem number_ratio (A B C : ℚ) : 
  A + B + C = 98 →
  B = 30 →
  8 * B = 5 * C →
  A * 3 = B * 2 := by
sorry

end NUMINAMATH_CALUDE_number_ratio_l1421_142140


namespace NUMINAMATH_CALUDE_largest_gcd_sum_780_l1421_142188

theorem largest_gcd_sum_780 :
  ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ x + y = 780 ∧
  (∀ (a b : ℕ), a > 0 → b > 0 → a + b = 780 → Nat.gcd a b ≤ Nat.gcd x y) ∧
  Nat.gcd x y = 390 := by
sorry

end NUMINAMATH_CALUDE_largest_gcd_sum_780_l1421_142188


namespace NUMINAMATH_CALUDE_solution_set_inequality_l1421_142153

theorem solution_set_inequality (x : ℝ) : 
  (2*x - 1) / x < 0 ↔ 0 < x ∧ x < 1/2 :=
by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l1421_142153


namespace NUMINAMATH_CALUDE_average_weight_problem_l1421_142196

theorem average_weight_problem (a b c : ℝ) :
  (a + b + c) / 3 = 45 →
  (b + c) / 2 = 46 →
  b = 37 →
  (a + b) / 2 = 40 := by
sorry

end NUMINAMATH_CALUDE_average_weight_problem_l1421_142196


namespace NUMINAMATH_CALUDE_sasha_purchase_l1421_142106

/-- The total number of items (pencils and pens) purchased by Sasha -/
def total_items : ℕ := 23

/-- The cost of a single pencil in rubles -/
def pencil_cost : ℕ := 13

/-- The cost of a single pen in rubles -/
def pen_cost : ℕ := 20

/-- The total amount spent in rubles -/
def total_spent : ℕ := 350

/-- Theorem stating that given the costs and total spent, the total number of items purchased is 23 -/
theorem sasha_purchase :
  ∃ (pencils pens : ℕ),
    pencils * pencil_cost + pens * pen_cost = total_spent ∧
    pencils + pens = total_items :=
by sorry

end NUMINAMATH_CALUDE_sasha_purchase_l1421_142106


namespace NUMINAMATH_CALUDE_last_digit_of_seven_to_seventh_l1421_142186

theorem last_digit_of_seven_to_seventh : 7^7 % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_seven_to_seventh_l1421_142186


namespace NUMINAMATH_CALUDE_nineteen_power_nineteen_not_sum_of_cube_and_fourth_power_l1421_142159

theorem nineteen_power_nineteen_not_sum_of_cube_and_fourth_power :
  ¬ ∃ (x y : ℤ), 19^19 = x^3 + y^4 := by
  sorry

end NUMINAMATH_CALUDE_nineteen_power_nineteen_not_sum_of_cube_and_fourth_power_l1421_142159


namespace NUMINAMATH_CALUDE_parabola_point_distance_l1421_142174

/-- The x-coordinate of point B on a parabola y^2 = 2px (p > 0) with point A(1, 2),
    where the distance from A to B(x, 0) equals the distance from A to the line x = -1 -/
theorem parabola_point_distance (p : ℝ) (x : ℝ) (h1 : p > 0) : 
  2 * p = 4 →  -- A(1, 2) lies on the parabola
  (x - 1)^2 + 2^2 = (1 - (-1))^2 →  -- Distance equality condition
  x = 1 :=
by sorry

end NUMINAMATH_CALUDE_parabola_point_distance_l1421_142174


namespace NUMINAMATH_CALUDE_sum_first_four_is_sixty_l1421_142163

/-- A geometric sequence with specific properties -/
structure GeometricSequence where
  a : ℝ  -- first term
  r : ℝ  -- common ratio
  sum_first_two : a + a * r = 15
  sum_first_six : a * (1 - r^6) / (1 - r) = 93

/-- The sum of the first 4 terms of the geometric sequence is 60 -/
theorem sum_first_four_is_sixty (seq : GeometricSequence) :
  seq.a * (1 - seq.r^4) / (1 - seq.r) = 60 := by
  sorry


end NUMINAMATH_CALUDE_sum_first_four_is_sixty_l1421_142163


namespace NUMINAMATH_CALUDE_cookie_ratio_l1421_142116

theorem cookie_ratio (monday tuesday wednesday : ℕ) : 
  monday = 32 →
  tuesday = monday / 2 →
  monday + tuesday + (wednesday - 4) = 92 →
  wednesday / tuesday = 3 := by
  sorry

end NUMINAMATH_CALUDE_cookie_ratio_l1421_142116


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1421_142161

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x, (1 - 2*x)^7 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = -2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l1421_142161


namespace NUMINAMATH_CALUDE_belongs_to_group_45_l1421_142199

/-- Defines the group number for a given positive integer -/
def groupNumber (n : ℕ+) : ℕ :=
  (n.val.sqrt : ℕ) + 1

/-- Defines the cumulative total of elements up to group n -/
def cumulativeTotal (n : ℕ) : ℕ := n^2

/-- States that 2009 belongs to group 45 -/
theorem belongs_to_group_45 : groupNumber 2009 = 45 := by sorry

end NUMINAMATH_CALUDE_belongs_to_group_45_l1421_142199


namespace NUMINAMATH_CALUDE_shopping_money_l1421_142101

theorem shopping_money (initial_amount : ℝ) : 
  0.7 * initial_amount = 3500 → initial_amount = 5000 := by
  sorry

end NUMINAMATH_CALUDE_shopping_money_l1421_142101


namespace NUMINAMATH_CALUDE_continuity_at_zero_l1421_142146

noncomputable def f (x : ℝ) : ℝ := 
  (Real.rpow (1 + x) (1/3) - 1) / (Real.sqrt (4 + x) - 2)

theorem continuity_at_zero : 
  Filter.Tendsto f (nhds 0) (nhds (4/3)) := by sorry

end NUMINAMATH_CALUDE_continuity_at_zero_l1421_142146


namespace NUMINAMATH_CALUDE_egg_laying_hens_l1421_142173

/-- Calculates the number of egg-laying hens on Mr. Curtis's farm -/
theorem egg_laying_hens (total_chickens roosters non_laying_hens : ℕ) 
  (h1 : total_chickens = 325)
  (h2 : roosters = 28)
  (h3 : non_laying_hens = 20) :
  total_chickens - roosters - non_laying_hens = 277 := by
  sorry

#check egg_laying_hens

end NUMINAMATH_CALUDE_egg_laying_hens_l1421_142173


namespace NUMINAMATH_CALUDE_stratified_sampling_sophomores_l1421_142157

theorem stratified_sampling_sophomores 
  (total_students : ℕ) 
  (sophomores : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_students = 1000)
  (h2 : sophomores = 320)
  (h3 : sample_size = 50) : 
  (sophomores * sample_size) / total_students = 16 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_sophomores_l1421_142157


namespace NUMINAMATH_CALUDE_antique_shop_glass_price_l1421_142168

theorem antique_shop_glass_price :
  let num_dolls : ℕ := 3
  let num_clocks : ℕ := 2
  let num_glasses : ℕ := 5
  let doll_price : ℕ := 5
  let clock_price : ℕ := 15
  let total_cost : ℕ := 40
  let profit : ℕ := 25
  let total_revenue : ℕ := total_cost + profit
  let doll_revenue : ℕ := num_dolls * doll_price
  let clock_revenue : ℕ := num_clocks * clock_price
  let glass_revenue : ℕ := total_revenue - doll_revenue - clock_revenue
  glass_revenue / num_glasses = 4
  := by sorry

end NUMINAMATH_CALUDE_antique_shop_glass_price_l1421_142168


namespace NUMINAMATH_CALUDE_value_preserving_2x_squared_value_preserving_x_squared_minus_2x_plus_m_l1421_142133

/-- A function is value-preserving on an interval [a, b] if it is monotonic
and its range on [a, b] is exactly [a, b] -/
def is_value_preserving (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  a < b ∧ 
  (∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → (f x < f y ∨ f y < f x)) ∧
  (∀ y, a ≤ y ∧ y ≤ b → ∃ x, a ≤ x ∧ x ≤ b ∧ f x = y)

/-- The function f(x) = 2x² has a unique value-preserving interval [0, 1/2] -/
theorem value_preserving_2x_squared :
  ∃! (a b : ℝ), is_value_preserving (fun x ↦ 2 * x^2) a b ∧ a = 0 ∧ b = 1/2 :=
sorry

/-- The function g(x) = x² - 2x + m has value-preserving intervals
if and only if m ∈ [1, 5/4) ∪ [2, 9/4) -/
theorem value_preserving_x_squared_minus_2x_plus_m (m : ℝ) :
  (∃ a b, is_value_preserving (fun x ↦ x^2 - 2*x + m) a b) ↔ 
  (1 ≤ m ∧ m < 5/4) ∨ (2 ≤ m ∧ m < 9/4) :=
sorry

end NUMINAMATH_CALUDE_value_preserving_2x_squared_value_preserving_x_squared_minus_2x_plus_m_l1421_142133


namespace NUMINAMATH_CALUDE_imaginary_part_of_one_plus_i_squared_l1421_142108

theorem imaginary_part_of_one_plus_i_squared (i : ℂ) : 
  i^2 = -1 → Complex.im ((1 + i)^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_one_plus_i_squared_l1421_142108


namespace NUMINAMATH_CALUDE_chess_tournament_games_l1421_142177

theorem chess_tournament_games (total_games : ℕ) (participants : ℕ) 
  (h1 : total_games = 120) (h2 : participants = 16) :
  (participants - 1 : ℕ) = 15 ∧ total_games = participants * (participants - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l1421_142177


namespace NUMINAMATH_CALUDE_trig_identity_l1421_142181

theorem trig_identity (α : Real) (h : Real.sin α + 2 * Real.cos α = 0) :
  2 * Real.sin α * Real.cos α - Real.cos α ^ 2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1421_142181


namespace NUMINAMATH_CALUDE_percentage_sold_first_day_l1421_142112

-- Define the initial number of watermelons
def initial_watermelons : ℕ := 10 * 12

-- Define the number of watermelons left after two days of selling
def remaining_watermelons : ℕ := 54

-- Define the percentage sold on the second day
def second_day_percentage : ℚ := 1 / 4

-- Theorem to prove
theorem percentage_sold_first_day :
  ∃ (p : ℚ), 0 ≤ p ∧ p ≤ 1 ∧
  (1 - second_day_percentage) * ((1 - p) * initial_watermelons) = remaining_watermelons ∧
  p = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_percentage_sold_first_day_l1421_142112


namespace NUMINAMATH_CALUDE_steps_to_top_floor_l1421_142115

/-- The number of steps between each floor in the building -/
def steps_between_floors : ℕ := 13

/-- The total number of floors in the building -/
def total_floors : ℕ := 7

/-- The number of intervals between floors when going from ground to top floor -/
def floor_intervals : ℕ := total_floors - 1

/-- The total number of steps from ground floor to the top floor -/
def total_steps : ℕ := steps_between_floors * floor_intervals

theorem steps_to_top_floor :
  total_steps = 78 :=
sorry

end NUMINAMATH_CALUDE_steps_to_top_floor_l1421_142115


namespace NUMINAMATH_CALUDE_chess_tournament_schedules_l1421_142165

/-- Represents a chess tournament between two schools -/
structure ChessTournament where
  players_per_school : Nat
  games_per_player : Nat
  games_per_round : Nat

/-- Calculates the total number of games in the tournament -/
def total_games (t : ChessTournament) : Nat :=
  t.players_per_school * t.players_per_school * t.games_per_player

/-- Calculates the number of rounds in the tournament -/
def num_rounds (t : ChessTournament) : Nat :=
  total_games t / t.games_per_round

/-- Theorem stating the number of ways to schedule the tournament -/
theorem chess_tournament_schedules (t : ChessTournament) 
  (h1 : t.players_per_school = 4)
  (h2 : t.games_per_player = 2)
  (h3 : t.games_per_round = 4) :
  Nat.factorial (num_rounds t) = 40320 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_schedules_l1421_142165


namespace NUMINAMATH_CALUDE_a_values_theorem_l1421_142189

theorem a_values_theorem (a b x : ℝ) (h1 : a - b = x) (h2 : x ≠ 0) (h3 : a^3 - b^3 = 19*x^3) :
  a = 3*x ∨ a = -2*x :=
by sorry

end NUMINAMATH_CALUDE_a_values_theorem_l1421_142189


namespace NUMINAMATH_CALUDE_distribute_5_3_l1421_142162

/-- The number of ways to distribute n distinct items into k identical bags -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 5 distinct items into 3 identical bags -/
theorem distribute_5_3 : distribute 5 3 = 36 := by sorry

end NUMINAMATH_CALUDE_distribute_5_3_l1421_142162


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l1421_142172

/-- Given a line L1 with equation 2x-y+3=0 and a point P(1,1), 
    the line L2 passing through P and perpendicular to L1 
    has the equation x+2y-3=0 -/
theorem perpendicular_line_equation : 
  let L1 : ℝ → ℝ → Prop := λ x y ↦ 2*x - y + 3 = 0
  let P : ℝ × ℝ := (1, 1)
  let L2 : ℝ → ℝ → Prop := λ x y ↦ x + 2*y - 3 = 0
  (∀ x y, L2 x y ↔ (y - P.2 = -(1/2) * (x - P.1))) ∧ 
  (∀ x₁ y₁ x₂ y₂, L1 x₁ y₁ → L1 x₂ y₂ → L2 x₁ y₁ → L2 x₂ y₂ → 
    (x₂ - x₁) * (x₂ - x₁) + (y₂ - y₁) * (y₂ - y₁) = 0) ∧
  L2 P.1 P.2 := by
  sorry


end NUMINAMATH_CALUDE_perpendicular_line_equation_l1421_142172


namespace NUMINAMATH_CALUDE_cone_surface_area_l1421_142195

/-- The surface area of a cone, given its lateral surface properties -/
theorem cone_surface_area (r : ℝ) (h : ℝ) : 
  (r * r * π + r * h * π = 16 * π / 9) →
  (h * h + r * r = 2 * 2) →
  (2 * π * r = 4 * π / 3) →
  (r * h * π = 4 * π / 3) →
  (r * r * π + r * h * π = 16 * π / 9) :=
by sorry

end NUMINAMATH_CALUDE_cone_surface_area_l1421_142195

import Mathlib

namespace NUMINAMATH_CALUDE_boat_rental_problem_l2240_224044

theorem boat_rental_problem (total_students : ℕ) 
  (large_boat_capacity small_boat_capacity : ℕ) :
  total_students = 104 →
  large_boat_capacity = 12 →
  small_boat_capacity = 5 →
  ∃ (num_large_boats num_small_boats : ℕ),
    num_large_boats * large_boat_capacity + 
    num_small_boats * small_boat_capacity = total_students ∧
    (num_large_boats = 2 ∨ num_large_boats = 7) :=
by sorry

end NUMINAMATH_CALUDE_boat_rental_problem_l2240_224044


namespace NUMINAMATH_CALUDE_f_negative_two_is_zero_l2240_224064

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2007 + b * x + 1

-- State the theorem
theorem f_negative_two_is_zero (a b : ℝ) :
  f a b 2 = 2 → f a b (-2) = 0 := by sorry

end NUMINAMATH_CALUDE_f_negative_two_is_zero_l2240_224064


namespace NUMINAMATH_CALUDE_sum_of_fractions_equals_sum_of_roots_l2240_224082

theorem sum_of_fractions_equals_sum_of_roots : 
  let T := 1 / (Real.sqrt 10 - Real.sqrt 8) + 
           1 / (Real.sqrt 8 - Real.sqrt 6) + 
           1 / (Real.sqrt 6 - Real.sqrt 4)
  T = (Real.sqrt 10 + 2 * Real.sqrt 8 + 2 * Real.sqrt 6 + 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_equals_sum_of_roots_l2240_224082


namespace NUMINAMATH_CALUDE_nth_equation_proof_l2240_224020

theorem nth_equation_proof (n : ℕ) : 
  n^2 + (n+1)^2 = (n*(n+1)+1)^2 - (n*(n+1))^2 := by
  sorry

end NUMINAMATH_CALUDE_nth_equation_proof_l2240_224020


namespace NUMINAMATH_CALUDE_parallel_vectors_imply_x_value_l2240_224032

/-- Two 2D vectors are parallel if their cross product is zero -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

/-- Given vectors a and b, prove that if they are parallel, then x = -2 -/
theorem parallel_vectors_imply_x_value :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (x, -4)
  are_parallel a b → x = -2 :=
by
  sorry


end NUMINAMATH_CALUDE_parallel_vectors_imply_x_value_l2240_224032


namespace NUMINAMATH_CALUDE_geometric_series_ratio_l2240_224068

theorem geometric_series_ratio (a r : ℝ) (hr : r ≠ 1) (ha : a ≠ 0) :
  (a / (1 - r) = 81 * (a * r^4) / (1 - r)) → r = 1/3 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_ratio_l2240_224068


namespace NUMINAMATH_CALUDE_race_time_theorem_l2240_224099

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ
  time : ℝ

/-- Represents the race scenario -/
structure Race where
  distance : ℝ
  runner_a : Runner
  runner_b : Runner

/-- The race conditions -/
def race_conditions (r : Race) : Prop :=
  r.distance = 1000 ∧
  (r.runner_a.speed * r.runner_a.time = r.distance) ∧
  (r.runner_b.speed * r.runner_b.time = r.distance - 40 ∨
   r.runner_b.speed * (r.runner_a.time + 10) = r.distance)

/-- The theorem to prove -/
theorem race_time_theorem (r : Race) :
  race_conditions r → r.runner_a.time = 240 := by
  sorry

end NUMINAMATH_CALUDE_race_time_theorem_l2240_224099


namespace NUMINAMATH_CALUDE_theresa_final_week_hours_l2240_224036

def hours_worked : List Nat := [9, 12, 6, 13, 11]
def total_weeks : Nat := 6
def required_average : Nat := 9

theorem theresa_final_week_hours :
  ∃ x : Nat, 
    (hours_worked.sum + x) / total_weeks = required_average ∧ 
    x = 3 := by
  sorry

end NUMINAMATH_CALUDE_theresa_final_week_hours_l2240_224036


namespace NUMINAMATH_CALUDE_monic_polynomial_square_decomposition_l2240_224016

theorem monic_polynomial_square_decomposition
  (P : Polynomial ℤ)
  (h_monic : P.Monic)
  (h_even_degree : Even P.degree)
  (h_infinite_squares : ∃ S : Set ℤ, Infinite S ∧ ∀ x ∈ S, ∃ y : ℤ, 0 < y ∧ P.eval x = y^2) :
  ∃ Q : Polynomial ℤ, P = Q^2 :=
sorry

end NUMINAMATH_CALUDE_monic_polynomial_square_decomposition_l2240_224016


namespace NUMINAMATH_CALUDE_system_solution_l2240_224034

theorem system_solution (x y : ℝ) 
  (h1 : x * y = -8)
  (h2 : x^2 * y + x * y^2 + 3*x + 3*y = 100) :
  x^2 + y^2 = 416 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l2240_224034


namespace NUMINAMATH_CALUDE_mass_of_man_is_60kg_l2240_224005

/-- The mass of a man who causes a boat to sink by a certain amount. -/
def mass_of_man (boat_length boat_breadth sink_depth water_density : ℝ) : ℝ :=
  boat_length * boat_breadth * sink_depth * water_density

/-- Theorem stating that the mass of the man is 60 kg under the given conditions. -/
theorem mass_of_man_is_60kg :
  mass_of_man 3 2 0.01 1000 = 60 := by sorry

end NUMINAMATH_CALUDE_mass_of_man_is_60kg_l2240_224005


namespace NUMINAMATH_CALUDE_other_number_problem_l2240_224084

theorem other_number_problem (a b : ℕ) : 
  a + b = 96 → 
  (a = b + 12 ∨ b = a + 12) → 
  (a = 42 ∨ b = 42) → 
  (a = 54 ∨ b = 54) :=
by sorry

end NUMINAMATH_CALUDE_other_number_problem_l2240_224084


namespace NUMINAMATH_CALUDE_arithmetic_progression_probability_l2240_224079

/-- The number of faces on each die -/
def num_faces : ℕ := 6

/-- The total number of possible outcomes when tossing three dice -/
def total_outcomes : ℕ := num_faces ^ 3

/-- A function that checks if three numbers form an arithmetic progression with common difference 2 -/
def is_arithmetic_progression (a b c : ℕ) : Prop :=
  (b = a + 2 ∧ c = b + 2) ∨ (b = a - 2 ∧ c = b - 2) ∨
  (a = b + 2 ∧ c = a + 2) ∨ (c = b + 2 ∧ a = c + 2) ∨
  (a = b - 2 ∧ c = a - 2) ∨ (c = b - 2 ∧ a = c - 2)

/-- The number of favorable outcomes (i.e., outcomes that form an arithmetic progression) -/
def favorable_outcomes : ℕ := 12

/-- The theorem stating the probability of getting an arithmetic progression -/
theorem arithmetic_progression_probability :
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 18 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_probability_l2240_224079


namespace NUMINAMATH_CALUDE_solve_for_A_l2240_224037

theorem solve_for_A (A : ℤ) (h : A - 10 = 15) : A = 25 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_A_l2240_224037


namespace NUMINAMATH_CALUDE_negation_existence_real_l2240_224041

theorem negation_existence_real : 
  (¬ ∃ x : ℝ, x > 1) ↔ (∀ x : ℝ, x ≤ 1) := by sorry

end NUMINAMATH_CALUDE_negation_existence_real_l2240_224041


namespace NUMINAMATH_CALUDE_complement_A_eq_three_four_l2240_224015

-- Define the set A
def A : Set ℕ := {x : ℕ | x^2 - 7*x + 10 ≥ 0}

-- Define the complement of A with respect to ℕ
def complement_A : Set ℕ := {x : ℕ | x ∉ A}

-- Theorem statement
theorem complement_A_eq_three_four : complement_A = {3, 4} := by sorry

end NUMINAMATH_CALUDE_complement_A_eq_three_four_l2240_224015


namespace NUMINAMATH_CALUDE_trapezoid_area_l2240_224071

/-- The area of a trapezoid with height x, one base 4x, and the other base 3x is 7x²/2 -/
theorem trapezoid_area (x : ℝ) : 
  x * ((4 * x + 3 * x) / 2) = 7 * x^2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_l2240_224071


namespace NUMINAMATH_CALUDE_art_museum_pictures_l2240_224030

theorem art_museum_pictures : ∃ (P : ℕ), P > 0 ∧ P % 2 = 1 ∧ (P + 1) % 2 = 0 ∧ ∀ (Q : ℕ), (Q > 0 ∧ Q % 2 = 1 ∧ (Q + 1) % 2 = 0) → P ≤ Q :=
by sorry

end NUMINAMATH_CALUDE_art_museum_pictures_l2240_224030


namespace NUMINAMATH_CALUDE_xiaoxia_exceeds_xiaoming_l2240_224008

/-- Represents the savings of a person over time -/
structure Savings where
  initial : ℕ  -- Initial savings
  monthly : ℕ  -- Monthly savings rate
  months : ℕ   -- Number of months passed

/-- Calculates the total savings after a given number of months -/
def totalSavings (s : Savings) : ℕ :=
  s.initial + s.monthly * s.months

/-- Xiaoxia's savings parameters -/
def xiaoxia : Savings :=
  { initial := 52, monthly := 15, months := 0 }

/-- Xiaoming's savings parameters -/
def xiaoming : Savings :=
  { initial := 70, monthly := 12, months := 0 }

/-- Theorem stating when Xiaoxia's savings exceed Xiaoming's -/
theorem xiaoxia_exceeds_xiaoming (n : ℕ) :
  totalSavings { xiaoxia with months := n } > totalSavings { xiaoming with months := n } ↔
  52 + 15 * n > 70 + 12 * n :=
sorry

end NUMINAMATH_CALUDE_xiaoxia_exceeds_xiaoming_l2240_224008


namespace NUMINAMATH_CALUDE_ben_remaining_amount_l2240_224000

/-- Calculates the remaining amount after a series of transactions -/
def remaining_amount (initial: Int) (supplier_payment: Int) (debtor_payment: Int) (maintenance_cost: Int) : Int :=
  initial - supplier_payment + debtor_payment - maintenance_cost

/-- Proves that given the specified transactions, the remaining amount is $1000 -/
theorem ben_remaining_amount :
  remaining_amount 2000 600 800 1200 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_ben_remaining_amount_l2240_224000


namespace NUMINAMATH_CALUDE_next_number_with_property_l2240_224093

/-- A function that splits a four-digit number into its hundreds and tens-ones parts -/
def split_number (n : ℕ) : ℕ × ℕ :=
  (n / 100, n % 100)

/-- A function that checks if a number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

/-- The property we're looking for in the number -/
def has_property (n : ℕ) : Prop :=
  let (a, b) := split_number n
  is_perfect_square (a * b)

theorem next_number_with_property :
  ∀ n : ℕ, 1818 < n → n < 1832 → ¬(has_property n) ∧ has_property 1832 := by
  sorry

#check next_number_with_property

end NUMINAMATH_CALUDE_next_number_with_property_l2240_224093


namespace NUMINAMATH_CALUDE_book_pages_l2240_224049

/-- The number of pages Charlie read in the book -/
def total_pages : ℕ :=
  let first_four_days : ℕ := 4 * 45
  let next_three_days : ℕ := 3 * 52
  let last_day : ℕ := 15
  first_four_days + next_three_days + last_day

/-- Theorem stating that the total number of pages in the book is 351 -/
theorem book_pages : total_pages = 351 := by
  sorry

end NUMINAMATH_CALUDE_book_pages_l2240_224049


namespace NUMINAMATH_CALUDE_frog_arrangement_count_l2240_224025

/-- Represents the number of ways to arrange frogs with given constraints -/
def frog_arrangements (n : ℕ) (green red : ℕ) (blue yellow : ℕ) : ℕ :=
  sorry

/-- The main theorem stating the number of valid frog arrangements -/
theorem frog_arrangement_count :
  frog_arrangements 7 2 3 1 1 = 120 := by
  sorry

end NUMINAMATH_CALUDE_frog_arrangement_count_l2240_224025


namespace NUMINAMATH_CALUDE_alien_abduction_l2240_224021

theorem alien_abduction (P : ℕ) : 
  (80 : ℚ) / 100 * P + 40 = P → P = 200 := by
  sorry

end NUMINAMATH_CALUDE_alien_abduction_l2240_224021


namespace NUMINAMATH_CALUDE_unique_solution_implies_a_greater_than_one_l2240_224055

-- Define the function f(x) = 2ax^2 - x - 1
def f (a : ℝ) (x : ℝ) : ℝ := 2 * a * x^2 - x - 1

-- State the theorem
theorem unique_solution_implies_a_greater_than_one :
  ∀ a : ℝ, (∃! x : ℝ, x ∈ (Set.Ioo 0 1) ∧ f a x = 0) → a > 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_implies_a_greater_than_one_l2240_224055


namespace NUMINAMATH_CALUDE_eighty_nine_degrees_is_acute_l2240_224051

-- Define what an acute angle is
def is_acute_angle (angle : ℝ) : Prop := 0 < angle ∧ angle < 90

-- State the theorem
theorem eighty_nine_degrees_is_acute : is_acute_angle 89 := by
  sorry

end NUMINAMATH_CALUDE_eighty_nine_degrees_is_acute_l2240_224051


namespace NUMINAMATH_CALUDE_locus_of_Q_l2240_224019

/-- The locus of point Q given an ellipse with specific properties -/
theorem locus_of_Q (a b : ℝ) (P : ℝ × ℝ) (E : ℝ × ℝ) (Q : ℝ × ℝ) :
  a > b → b > 0 →
  (P.1^2 / a^2) + (P.2^2 / b^2) = 1 →
  P ≠ (-2, 0) → P ≠ (2, 0) →
  a = 2 →
  (1 : ℝ) / 2 = Real.sqrt (1 - b^2 / a^2) →
  E.1 - (-4) = (3 / 5) * (P.1 - (-4)) →
  E.2 = (3 / 5) * P.2 →
  (Q.2 + 2) / (Q.1 + 2) = P.2 / (P.1 + 2) →
  (Q.2 - 0) / (Q.1 - 2) = E.2 / (E.1 - 2) →
  Q.2 ≠ 0 →
  (Q.1 + 1)^2 + (4 * Q.2^2) / 3 = 1 := by
sorry

end NUMINAMATH_CALUDE_locus_of_Q_l2240_224019


namespace NUMINAMATH_CALUDE_task_completion_proof_l2240_224091

def task_completion (x : ℝ) : Prop :=
  let a := x
  let b := x + 6
  let c := x + 9
  (3 / a + 4 / b = 9 / c) ∧ (a = 18) ∧ (b = 24) ∧ (c = 27)

theorem task_completion_proof : ∃ x : ℝ, task_completion x := by
  sorry

end NUMINAMATH_CALUDE_task_completion_proof_l2240_224091


namespace NUMINAMATH_CALUDE_water_intake_glasses_l2240_224056

/-- Calculates the number of glasses of water needed to meet a daily water intake goal -/
theorem water_intake_glasses (daily_goal : ℝ) (glass_capacity : ℝ) : 
  daily_goal = 1.5 → glass_capacity = 0.250 → (daily_goal * 1000) / glass_capacity = 6 := by
  sorry

end NUMINAMATH_CALUDE_water_intake_glasses_l2240_224056


namespace NUMINAMATH_CALUDE_crabby_squido_ratio_l2240_224090

def squido_oysters : ℕ := 200
def total_oysters : ℕ := 600

def crabby_oysters : ℕ := total_oysters - squido_oysters

theorem crabby_squido_ratio : 
  (crabby_oysters : ℚ) / squido_oysters = 2 := by sorry

end NUMINAMATH_CALUDE_crabby_squido_ratio_l2240_224090


namespace NUMINAMATH_CALUDE_june_first_is_friday_l2240_224060

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a month with given properties -/
structure Month where
  days : Nat
  firstDay : DayOfWeek
  mondayCount : Nat
  thursdayCount : Nat

/-- Function to determine if a month satisfies the given conditions -/
def satisfiesConditions (m : Month) : Prop :=
  m.days = 30 ∧ m.mondayCount = 3 ∧ m.thursdayCount = 3

/-- Theorem stating that a month satisfying the conditions must start on a Friday -/
theorem june_first_is_friday (m : Month) :
  satisfiesConditions m → m.firstDay = DayOfWeek.Friday :=
by
  sorry


end NUMINAMATH_CALUDE_june_first_is_friday_l2240_224060


namespace NUMINAMATH_CALUDE_quadratic_has_real_roots_rhombus_area_when_m_neg_seven_l2240_224006

-- Define the quadratic equation
def quadratic_equation (x m : ℝ) : ℝ := 2 * x^2 + (m - 2) * x - m

-- Theorem 1: The equation always has real roots
theorem quadratic_has_real_roots (m : ℝ) :
  ∃ x : ℝ, quadratic_equation x m = 0 :=
sorry

-- Theorem 2: Area of rhombus when m = -7
theorem rhombus_area_when_m_neg_seven :
  let m : ℝ := -7
  let root1 : ℝ := (9 + Real.sqrt 25) / 4
  let root2 : ℝ := (9 - Real.sqrt 25) / 4
  quadratic_equation root1 m = 0 ∧
  quadratic_equation root2 m = 0 →
  (1 / 2) * root1 * root2 = 7 / 4 :=
sorry

end NUMINAMATH_CALUDE_quadratic_has_real_roots_rhombus_area_when_m_neg_seven_l2240_224006


namespace NUMINAMATH_CALUDE_integer_solution_l2240_224022

theorem integer_solution (x : ℤ) : 
  x + 15 ≥ 16 ∧ -3*x ≥ -15 → x ∈ ({1, 2, 3, 4, 5} : Set ℤ) := by
sorry

end NUMINAMATH_CALUDE_integer_solution_l2240_224022


namespace NUMINAMATH_CALUDE_minimum_amount_is_1000_l2240_224043

/-- The minimum amount of the sell to get a discount -/
def minimum_amount_for_discount (
  item_count : ℕ) 
  (item_cost : ℚ) 
  (discounted_total : ℚ) 
  (discount_rate : ℚ) : ℚ :=
  item_count * item_cost - (item_count * item_cost - discounted_total) / discount_rate

/-- Theorem stating the minimum amount for discount is $1000 -/
theorem minimum_amount_is_1000 : 
  minimum_amount_for_discount 7 200 1360 (1/10) = 1000 := by
  sorry

end NUMINAMATH_CALUDE_minimum_amount_is_1000_l2240_224043


namespace NUMINAMATH_CALUDE_vector_c_determination_l2240_224075

/-- Given vectors a and b, if vector c satisfies the conditions, then c = (2, 1) -/
theorem vector_c_determination (a b c : ℝ × ℝ) 
  (ha : a = (1, -1)) 
  (hb : b = (1, 2)) 
  (hperp : (c.1 + b.1, c.2 + b.2) • a = 0)  -- (c + b) ⊥ a
  (hpar : ∃ k : ℝ, (c.1 - a.1, c.2 - a.2) = (k * b.1, k * b.2))  -- (c - a) ∥ b
  : c = (2, 1) := by
  sorry

end NUMINAMATH_CALUDE_vector_c_determination_l2240_224075


namespace NUMINAMATH_CALUDE_z_max_min_difference_l2240_224059

theorem z_max_min_difference (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x ≠ y) : 
  let z := fun (a b : ℝ) => |a^2 - b^2| / (|a^2| + |b^2|)
  ∃ (max min : ℝ), 
    (∀ a b, a ≠ 0 → b ≠ 0 → a ≠ b → z a b ≤ max) ∧
    (∃ a b, a ≠ 0 ∧ b ≠ 0 ∧ a ≠ b ∧ z a b = max) ∧
    (∀ a b, a ≠ 0 → b ≠ 0 → a ≠ b → min ≤ z a b) ∧
    (∃ a b, a ≠ 0 ∧ b ≠ 0 ∧ a ≠ b ∧ z a b = min) ∧
    max = 1 ∧ min = 0 ∧ max - min = 1 :=
by sorry

end NUMINAMATH_CALUDE_z_max_min_difference_l2240_224059


namespace NUMINAMATH_CALUDE_decimal_multiplication_l2240_224083

theorem decimal_multiplication : (0.5 : ℝ) * 0.7 = 0.35 := by sorry

end NUMINAMATH_CALUDE_decimal_multiplication_l2240_224083


namespace NUMINAMATH_CALUDE_sum_of_fraction_parts_l2240_224031

/-- The decimal representation of the number we're considering -/
def repeating_decimal : ℚ := 0.45454545

/-- Expresses the repeating decimal as a fraction -/
def as_fraction (x : ℚ) : ℚ := (100 * x - x) / 99

/-- Reduces a fraction to its lowest terms -/
def reduce_fraction (x : ℚ) : ℚ := x

theorem sum_of_fraction_parts : 
  (reduce_fraction (as_fraction repeating_decimal)).num +
  (reduce_fraction (as_fraction repeating_decimal)).den = 16 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fraction_parts_l2240_224031


namespace NUMINAMATH_CALUDE_candidate_vote_percentage_l2240_224003

theorem candidate_vote_percentage 
  (total_votes : ℕ) 
  (loss_margin : ℕ) 
  (candidate_percentage : ℚ) :
  total_votes = 4400 →
  loss_margin = 1760 →
  candidate_percentage = total_votes.cast⁻¹ * (total_votes - loss_margin) / 2 →
  candidate_percentage = 30 / 100 :=
by sorry

end NUMINAMATH_CALUDE_candidate_vote_percentage_l2240_224003


namespace NUMINAMATH_CALUDE_reservoir_after_storm_l2240_224057

/-- Represents the capacity of the reservoir in billion gallons -/
def reservoir_capacity : ℝ := 400

/-- Represents the initial amount of water in the reservoir in billion gallons -/
def initial_water : ℝ := 200

/-- Represents the amount of water added by the storm in billion gallons -/
def storm_water : ℝ := 120

/-- Theorem stating that the reservoir is 80% full after the storm -/
theorem reservoir_after_storm :
  (initial_water + storm_water) / reservoir_capacity = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_reservoir_after_storm_l2240_224057


namespace NUMINAMATH_CALUDE_matrix_equation_solution_l2240_224095

theorem matrix_equation_solution :
  let A : Matrix (Fin 2) (Fin 2) ℚ := !![2, -3; 5, -1]
  let B : Matrix (Fin 2) (Fin 2) ℚ := !![-30, -9; 11, 1]
  let N : Matrix (Fin 2) (Fin 2) ℚ := !![5, -8; 7/13, 35/13]
  N * A = B := by sorry

end NUMINAMATH_CALUDE_matrix_equation_solution_l2240_224095


namespace NUMINAMATH_CALUDE_stem_and_leaf_plot_preserves_information_l2240_224092

-- Define the different types of charts
inductive ChartType
  | BarChart
  | PieChart
  | LineChart
  | StemAndLeafPlot

-- Define a property for information preservation
def preserves_all_information (chart : ChartType) : Prop :=
  match chart with
  | ChartType.StemAndLeafPlot => True
  | _ => False

-- Theorem statement
theorem stem_and_leaf_plot_preserves_information :
  ∀ (chart : ChartType), preserves_all_information chart ↔ chart = ChartType.StemAndLeafPlot :=
by
  sorry


end NUMINAMATH_CALUDE_stem_and_leaf_plot_preserves_information_l2240_224092


namespace NUMINAMATH_CALUDE_impossible_arrangement_l2240_224027

theorem impossible_arrangement : ¬ ∃ (a b : Fin 2005 → Fin 4010),
  (∀ i : Fin 2005, a i < b i) ∧
  (∀ i : Fin 2005, b i - a i = i.val + 1) ∧
  (∀ k : Fin 4010, ∃! i : Fin 2005, a i = k ∨ b i = k) :=
by sorry

end NUMINAMATH_CALUDE_impossible_arrangement_l2240_224027


namespace NUMINAMATH_CALUDE_box_height_proof_l2240_224045

/-- Given a box with specified dimensions and cube requirements, prove its height --/
theorem box_height_proof (length width : ℝ) (cube_volume : ℝ) (min_cubes : ℕ) 
  (h_length : length = 9)
  (h_width : width = 12)
  (h_cube_volume : cube_volume = 3)
  (h_min_cubes : min_cubes = 108) :
  (cube_volume * min_cubes) / (length * width) = 3 := by
  sorry

end NUMINAMATH_CALUDE_box_height_proof_l2240_224045


namespace NUMINAMATH_CALUDE_sum_of_three_consecutive_cubes_divisible_by_nine_l2240_224047

theorem sum_of_three_consecutive_cubes_divisible_by_nine (n : ℕ) (h : n > 1) :
  ∃ k : ℤ, (n - 1)^3 + n^3 + (n + 1)^3 = 9 * k := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_consecutive_cubes_divisible_by_nine_l2240_224047


namespace NUMINAMATH_CALUDE_theater_attendance_l2240_224094

theorem theater_attendance
  (adult_ticket_price : ℕ)
  (child_ticket_price : ℕ)
  (total_revenue : ℕ)
  (num_children : ℕ)
  (h1 : adult_ticket_price = 8)
  (h2 : child_ticket_price = 1)
  (h3 : total_revenue = 50)
  (h4 : num_children = 18) :
  adult_ticket_price * (total_revenue - child_ticket_price * num_children) / adult_ticket_price + num_children = 22 :=
by
  sorry

end NUMINAMATH_CALUDE_theater_attendance_l2240_224094


namespace NUMINAMATH_CALUDE_twin_brothers_age_l2240_224028

/-- Theorem: Age of twin brothers
  Given that the product of their ages today is 13 less than the product of their ages a year from today,
  prove that the age of twin brothers today is 6 years old.
-/
theorem twin_brothers_age (x : ℕ) : x * x + 13 = (x + 1) * (x + 1) → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_twin_brothers_age_l2240_224028


namespace NUMINAMATH_CALUDE_arithmetic_sequence_12th_term_l2240_224018

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence satisfying certain conditions, its 12th term equals 14. -/
theorem arithmetic_sequence_12th_term
  (a : ℕ → ℝ)
  (h_arithmetic : is_arithmetic_sequence a)
  (h_sum : a 7 + a 9 = 16)
  (h_4th : a 4 = 2) :
  a 12 = 14 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_12th_term_l2240_224018


namespace NUMINAMATH_CALUDE_first_day_exceeding_150_fungi_l2240_224062

def fungi_growth (n : ℕ) : ℕ := 4 * 2^n

theorem first_day_exceeding_150_fungi : 
  (∃ n : ℕ, fungi_growth n > 150) ∧ 
  (∀ m : ℕ, m < 6 → fungi_growth m ≤ 150) ∧
  (fungi_growth 6 > 150) :=
by sorry

end NUMINAMATH_CALUDE_first_day_exceeding_150_fungi_l2240_224062


namespace NUMINAMATH_CALUDE_ant_movement_probability_l2240_224069

structure Octahedron where
  middleVertices : Finset Nat
  topVertex : Nat
  bottomVertex : Nat

def moveToMiddle (o : Octahedron) (start : Nat) : Finset Nat :=
  o.middleVertices.filter (λ v => v ≠ start)

def moveFromMiddle (o : Octahedron) (middle : Nat) : Finset Nat :=
  insert o.bottomVertex (insert o.topVertex (o.middleVertices.filter (λ v => v ≠ middle)))

theorem ant_movement_probability (o : Octahedron) (start : Nat) :
  start ∈ o.middleVertices →
  (1 : ℚ) / 4 = (moveToMiddle o start).sum (λ a =>
    (1 : ℚ) / (moveToMiddle o start).card *
    (1 : ℚ) / (moveFromMiddle o a).card *
    if o.bottomVertex ∈ moveFromMiddle o a then 1 else 0) :=
sorry

end NUMINAMATH_CALUDE_ant_movement_probability_l2240_224069


namespace NUMINAMATH_CALUDE_cricket_team_average_age_l2240_224017

/-- The average age of a cricket team given specific conditions -/
theorem cricket_team_average_age :
  ∀ (team_size : ℕ) (captain_age : ℕ) (wicket_keeper_age_diff : ℕ) (average_age : ℝ),
  team_size = 11 →
  captain_age = 24 →
  wicket_keeper_age_diff = 7 →
  (team_size : ℝ) * average_age = 
    (captain_age : ℝ) + (captain_age + wicket_keeper_age_diff : ℝ) + 
    ((team_size - 2 : ℝ) * (average_age - 1)) →
  average_age = 23 := by
sorry

end NUMINAMATH_CALUDE_cricket_team_average_age_l2240_224017


namespace NUMINAMATH_CALUDE_square_of_negative_two_a_squared_l2240_224024

theorem square_of_negative_two_a_squared (a : ℝ) : (-2 * a^2)^2 = 4 * a^4 := by
  sorry

end NUMINAMATH_CALUDE_square_of_negative_two_a_squared_l2240_224024


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_numbers_l2240_224007

def numbers : List ℝ := [18, 27, 45]

theorem arithmetic_mean_of_numbers : 
  (List.sum numbers) / (List.length numbers) = 30 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_numbers_l2240_224007


namespace NUMINAMATH_CALUDE_amy_started_with_101_seeds_l2240_224004

/-- The number of seeds Amy planted in her garden -/
def amy_garden_problem (big_garden_seeds : ℕ) (small_gardens : ℕ) (seeds_per_small_garden : ℕ) : ℕ :=
  big_garden_seeds + small_gardens * seeds_per_small_garden

/-- Theorem stating that Amy started with 101 seeds -/
theorem amy_started_with_101_seeds :
  amy_garden_problem 47 9 6 = 101 := by
  sorry

end NUMINAMATH_CALUDE_amy_started_with_101_seeds_l2240_224004


namespace NUMINAMATH_CALUDE_least_integer_x_l2240_224011

theorem least_integer_x : ∃ x : ℤ, (∀ z : ℤ, |3*z + 5 - 4| ≤ 25 → x ≤ z) ∧ |3*x + 5 - 4| ≤ 25 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_least_integer_x_l2240_224011


namespace NUMINAMATH_CALUDE_fixed_point_theorem_l2240_224086

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2 * p * x

-- Define a point on the parabola
def point_on_parabola (p : ℝ) (x y : ℝ) : Prop := parabola p x y

-- Define the fixed points A and B
def point_A (a b : ℝ) : ℝ × ℝ := (a, b)
def point_B (a : ℝ) : ℝ × ℝ := (-a, 0)

-- Define a line passing through two points
def line_through_points (x1 y1 x2 y2 : ℝ) (x y : ℝ) : Prop :=
  (y - y1) * (x2 - x1) = (y2 - y1) * (x - x1)

-- Define the theorem
theorem fixed_point_theorem (p a b : ℝ) (M M1 M2 : ℝ × ℝ) 
  (h1 : a * b ≠ 0)
  (h2 : b^2 ≠ 2 * p * a)
  (h3 : point_on_parabola p M.1 M.2)
  (h4 : point_on_parabola p M1.1 M1.2)
  (h5 : point_on_parabola p M2.1 M2.2)
  (h6 : line_through_points a b M.1 M.2 M1.1 M1.2)
  (h7 : line_through_points (-a) 0 M.1 M.2 M2.1 M2.2)
  (h8 : M1 ≠ M2) :
  line_through_points M1.1 M1.2 M2.1 M2.2 a (2 * p * a / b) :=
sorry

end NUMINAMATH_CALUDE_fixed_point_theorem_l2240_224086


namespace NUMINAMATH_CALUDE_value_calculation_l2240_224053

theorem value_calculation : 0.833 * (-72.0) = -59.976 := by
  sorry

end NUMINAMATH_CALUDE_value_calculation_l2240_224053


namespace NUMINAMATH_CALUDE_davis_oldest_child_age_l2240_224061

/-- The age of the oldest Davis child given the conditions -/
def oldest_child_age (avg_age : ℕ) (younger_child1 : ℕ) (younger_child2 : ℕ) : ℕ :=
  3 * avg_age - younger_child1 - younger_child2

/-- Theorem stating the age of the oldest Davis child -/
theorem davis_oldest_child_age :
  oldest_child_age 10 7 9 = 14 := by
  sorry

end NUMINAMATH_CALUDE_davis_oldest_child_age_l2240_224061


namespace NUMINAMATH_CALUDE_inverse_37_mod_53_l2240_224096

theorem inverse_37_mod_53 : ∃ x : ℤ, 37 * x ≡ 1 [ZMOD 53] :=
by
  use 10
  sorry

end NUMINAMATH_CALUDE_inverse_37_mod_53_l2240_224096


namespace NUMINAMATH_CALUDE_unique_prime_with_square_free_remainders_l2240_224013

theorem unique_prime_with_square_free_remainders : ∃! p : ℕ, 
  Nat.Prime p ∧ 
  (∀ q : ℕ, Nat.Prime q → q < p → 
    ∀ k r : ℕ, p = k * q + r → 0 ≤ r → r < q → 
      ∀ a : ℕ, a > 1 → ¬(a * a ∣ r)) ∧
  p = 13 := by
sorry

end NUMINAMATH_CALUDE_unique_prime_with_square_free_remainders_l2240_224013


namespace NUMINAMATH_CALUDE_consecutive_negative_integers_product_2850_l2240_224066

theorem consecutive_negative_integers_product_2850 :
  ∃ (n : ℤ), n < 0 ∧ n * (n + 1) = 2850 → (n + (n + 1)) = -107 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_negative_integers_product_2850_l2240_224066


namespace NUMINAMATH_CALUDE_log_equation_solution_l2240_224039

theorem log_equation_solution (b x : ℝ) (hb_pos : b > 0) (hb_neq_one : b ≠ 1) (hx_neq_one : x ≠ 1) :
  (Real.log x) / (Real.log (b^3)) + 3 * (Real.log b) / (Real.log x) = 2 → x = b^3 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l2240_224039


namespace NUMINAMATH_CALUDE_min_value_and_inequality_l2240_224014

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| + |x - 2|

-- State the theorem
theorem min_value_and_inequality :
  (∃ (a : ℝ), ∀ (x : ℝ), f x ≥ a ∧ ∃ (x₀ : ℝ), f x₀ = a) ∧
  (∀ (p q r : ℝ), p > 0 → q > 0 → r > 0 → p + q + r = 3 → p^2 + q^2 + r^2 ≥ 3) :=
by sorry

end NUMINAMATH_CALUDE_min_value_and_inequality_l2240_224014


namespace NUMINAMATH_CALUDE_twenty_is_forty_percent_l2240_224012

theorem twenty_is_forty_percent : ∃ x : ℝ, x = 55 ∧ 20 / (x - 5) = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_twenty_is_forty_percent_l2240_224012


namespace NUMINAMATH_CALUDE_alpha_plus_beta_equals_two_l2240_224058

theorem alpha_plus_beta_equals_two (α β : ℝ) 
  (h1 : α^3 - 3*α^2 + 5*α = 1) 
  (h2 : β^3 - 3*β^2 + 5*β = 5) : 
  α + β = 2 := by
sorry

end NUMINAMATH_CALUDE_alpha_plus_beta_equals_two_l2240_224058


namespace NUMINAMATH_CALUDE_probability_adjacent_circular_probability_two_adjacent_in_six_l2240_224042

def num_people : ℕ := 6

def total_arrangements (n : ℕ) : ℕ := Nat.factorial (n - 1)

def adjacent_arrangements (n : ℕ) : ℕ := 2 * Nat.factorial (n - 2)

theorem probability_adjacent_circular (n : ℕ) (h : n ≥ 3) :
  (adjacent_arrangements n : ℚ) / (total_arrangements n : ℚ) = 2 / (n - 1 : ℚ) :=
sorry

theorem probability_two_adjacent_in_six :
  (adjacent_arrangements num_people : ℚ) / (total_arrangements num_people : ℚ) = 2 / 5 :=
sorry

end NUMINAMATH_CALUDE_probability_adjacent_circular_probability_two_adjacent_in_six_l2240_224042


namespace NUMINAMATH_CALUDE_paul_total_crayons_l2240_224081

/-- The number of crayons Paul initially had -/
def initial_crayons : ℝ := 479.0

/-- The number of additional crayons Paul received -/
def additional_crayons : ℝ := 134.0

/-- The total number of crayons Paul now has -/
def total_crayons : ℝ := initial_crayons + additional_crayons

/-- Theorem stating that Paul now has 613.0 crayons -/
theorem paul_total_crayons : total_crayons = 613.0 := by
  sorry

end NUMINAMATH_CALUDE_paul_total_crayons_l2240_224081


namespace NUMINAMATH_CALUDE_fastest_student_survey_method_l2240_224046

/-- Represents a survey method -/
inductive SurveyMethod
| Comprehensive
| Sample

/-- Represents a scenario requiring a survey -/
structure Scenario where
  description : String
  requiredMethod : SurveyMethod

/-- Represents the selection of the fastest student in a school's short-distance race -/
def fastestStudentSelection : Scenario :=
  { description := "Selecting the fastest student in a school's short-distance race",
    requiredMethod := SurveyMethod.Comprehensive }

/-- Theorem: The appropriate survey method for selecting the fastest student
    in a school's short-distance race is a comprehensive survey -/
theorem fastest_student_survey_method :
  fastestStudentSelection.requiredMethod = SurveyMethod.Comprehensive :=
by sorry


end NUMINAMATH_CALUDE_fastest_student_survey_method_l2240_224046


namespace NUMINAMATH_CALUDE_hike_length_is_four_l2240_224033

/-- Represents the hike details -/
structure Hike where
  initial_water : ℝ
  duration : ℝ
  remaining_water : ℝ
  leak_rate : ℝ
  last_mile_consumption : ℝ
  first_part_consumption_rate : ℝ

/-- Calculates the length of the hike in miles -/
def hike_length (h : Hike) : ℝ :=
  sorry

/-- Theorem stating that given the conditions, the hike length is 4 miles -/
theorem hike_length_is_four (h : Hike) 
  (h_initial : h.initial_water = 10)
  (h_duration : h.duration = 2)
  (h_remaining : h.remaining_water = 2)
  (h_leak : h.leak_rate = 1)
  (h_last_mile : h.last_mile_consumption = 3)
  (h_first_part : h.first_part_consumption_rate = 1) :
  hike_length h = 4 := by
  sorry

end NUMINAMATH_CALUDE_hike_length_is_four_l2240_224033


namespace NUMINAMATH_CALUDE_problem_solution_l2240_224063

/-- A polynomial of degree 4 with integer coefficients -/
def f (c₀ c₁ c₂ c₃ c₄ : ℤ) (x : ℤ) : ℤ := c₄ * x^4 + c₃ * x^3 + c₂ * x^2 + c₁ * x + c₀

/-- There exists a unique solution to the problem -/
theorem problem_solution (c₀ c₁ c₂ c₃ c₄ : ℤ) :
  ∃! A : ℤ, ∃ B : ℤ, 
    A > B ∧ B > 7 ∧
    f c₀ c₁ c₂ c₃ c₄ A = 0 ∧
    f c₀ c₁ c₂ c₃ c₄ 7 = 77 ∧
    f c₀ c₁ c₂ c₃ c₄ B = 85 ∧
    A = 14 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2240_224063


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2240_224050

/-- Triangle ABC with sides a, b, c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A hyperbola with foci at the vertices of a triangle -/
structure Hyperbola (t : Triangle) where
  /-- The hyperbola passes through point A of the triangle -/
  passes_through_A : Bool

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola t) : ℝ := sorry

/-- The main theorem -/
theorem hyperbola_eccentricity (t : Triangle) (h : Hyperbola t) :
  t.a = 4 ∧ t.b = 5 ∧ t.c = Real.sqrt 21 ∧ h.passes_through_A = true →
  eccentricity h = 5 + Real.sqrt 21 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2240_224050


namespace NUMINAMATH_CALUDE_unique_solution_iff_k_eq_neg_three_fourths_l2240_224029

/-- The equation (x + 3) / (kx - 2) = x has exactly one solution if and only if k = -3/4 -/
theorem unique_solution_iff_k_eq_neg_three_fourths (k : ℝ) : 
  (∃! x : ℝ, (x + 3) / (k * x - 2) = x) ↔ k = -3/4 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_iff_k_eq_neg_three_fourths_l2240_224029


namespace NUMINAMATH_CALUDE_not_decreasing_everywhere_l2240_224038

theorem not_decreasing_everywhere (f : ℝ → ℝ) (h : f 1 < f 2) :
  ¬(∀ x y : ℝ, x < y → f x ≥ f y) :=
sorry

end NUMINAMATH_CALUDE_not_decreasing_everywhere_l2240_224038


namespace NUMINAMATH_CALUDE_perpendicular_tangents_ratio_l2240_224098

-- Define the curve
def curve (x : ℝ) : ℝ := x^3 + x

-- Define the derivative of the curve
def curve_derivative (x : ℝ) : ℝ := 3 * x^2 + 1

-- Define the point P
def P : ℝ × ℝ := (1, 2)

-- Theorem statement
theorem perpendicular_tangents_ratio (a b : ℝ) :
  (a * P.1 - b * P.2 - 2 = 0) →  -- Line equation at point P
  (curve P.1 = P.2) →            -- Curve passes through P
  (curve_derivative P.1 * (a / b) = -1) →  -- Perpendicular tangents condition
  a / b = -1/4 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_tangents_ratio_l2240_224098


namespace NUMINAMATH_CALUDE_alans_current_rate_prove_alans_current_rate_l2240_224009

/-- Alan's attempt to beat Kevin's hot wings eating record -/
theorem alans_current_rate (kevin_wings : ℕ) (kevin_time : ℕ) (alan_additional : ℕ) : ℕ :=
  let kevin_rate := kevin_wings / kevin_time
  let alan_target_rate := kevin_rate + 1
  alan_target_rate - alan_additional

/-- Proof of Alan's current rate of eating hot wings -/
theorem prove_alans_current_rate :
  alans_current_rate 64 8 4 = 4 := by
  sorry

end NUMINAMATH_CALUDE_alans_current_rate_prove_alans_current_rate_l2240_224009


namespace NUMINAMATH_CALUDE_max_soap_boxes_in_carton_l2240_224085

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a rectangular object given its dimensions -/
def volume (d : Dimensions) : ℕ := d.length * d.width * d.height

/-- The dimensions of the carton -/
def carton_dimensions : Dimensions := ⟨30, 42, 60⟩

/-- The dimensions of a soap box -/
def soap_box_dimensions : Dimensions := ⟨7, 6, 5⟩

/-- Theorem stating the maximum number of soap boxes that can fit in the carton -/
theorem max_soap_boxes_in_carton :
  (volume carton_dimensions) / (volume soap_box_dimensions) = 360 := by
  sorry

end NUMINAMATH_CALUDE_max_soap_boxes_in_carton_l2240_224085


namespace NUMINAMATH_CALUDE_range_of_f_l2240_224048

def f (x : ℝ) : ℝ := x^2 - 4*x + 1

theorem range_of_f :
  let S := {y | ∃ x ∈ Set.Icc 2 5, f x = y}
  S = Set.Icc (-3) 6 := by sorry

end NUMINAMATH_CALUDE_range_of_f_l2240_224048


namespace NUMINAMATH_CALUDE_function_property_Z_function_property_Q_l2240_224080

-- For integers
theorem function_property_Z (f : ℤ → ℤ) :
  (∀ a b : ℤ, f (2 * a) + 2 * f b = f (f (a + b))) →
  (∀ x : ℤ, f x = 2 * x ∨ f x = 0) :=
sorry

-- For rationals (bonus)
theorem function_property_Q (f : ℚ → ℚ) :
  (∀ a b : ℚ, f (2 * a) + 2 * f b = f (f (a + b))) →
  (∀ x : ℚ, f x = 2 * x ∨ f x = 0) :=
sorry

end NUMINAMATH_CALUDE_function_property_Z_function_property_Q_l2240_224080


namespace NUMINAMATH_CALUDE_crosswalk_stripe_distance_l2240_224026

theorem crosswalk_stripe_distance 
  (curb_distance : ℝ) 
  (curb_length : ℝ) 
  (stripe_length : ℝ) 
  (h1 : curb_distance = 30) 
  (h2 : curb_length = 10) 
  (h3 : stripe_length = 60) : 
  ∃ (stripe_distance : ℝ), 
    stripe_distance * stripe_length = curb_length * curb_distance ∧ 
    stripe_distance = 5 := by
  sorry

end NUMINAMATH_CALUDE_crosswalk_stripe_distance_l2240_224026


namespace NUMINAMATH_CALUDE_squirrel_travel_time_l2240_224077

/-- Proves that a squirrel traveling 3 miles at 6 miles per hour takes 30 minutes -/
theorem squirrel_travel_time :
  let speed : ℝ := 6 -- miles per hour
  let distance : ℝ := 3 -- miles
  let time_hours : ℝ := distance / speed
  let time_minutes : ℝ := time_hours * 60
  time_minutes = 30 := by
  sorry

end NUMINAMATH_CALUDE_squirrel_travel_time_l2240_224077


namespace NUMINAMATH_CALUDE_greatest_possible_k_l2240_224097

theorem greatest_possible_k (k : ℝ) : 
  (∃ x y : ℝ, x^2 + k*x + 8 = 0 ∧ y^2 + k*y + 8 = 0 ∧ |x - y| = Real.sqrt 73) →
  k ≤ Real.sqrt 105 :=
by sorry

end NUMINAMATH_CALUDE_greatest_possible_k_l2240_224097


namespace NUMINAMATH_CALUDE_blanket_donation_ratio_l2240_224087

/-- The ratio of blankets collected on the second day to the first day -/
def blanket_ratio (team_size : ℕ) (first_day_per_person : ℕ) (last_day_total : ℕ) (total_blankets : ℕ) : ℚ :=
  let first_day := team_size * first_day_per_person
  let second_day := total_blankets - first_day - last_day_total
  (second_day : ℚ) / first_day

/-- Proves that the ratio of blankets collected on the second day to the first day is 3 -/
theorem blanket_donation_ratio :
  blanket_ratio 15 2 22 142 = 3 := by
  sorry

end NUMINAMATH_CALUDE_blanket_donation_ratio_l2240_224087


namespace NUMINAMATH_CALUDE_gcd_of_360_and_504_l2240_224023

theorem gcd_of_360_and_504 : Nat.gcd 360 504 = 72 := by sorry

end NUMINAMATH_CALUDE_gcd_of_360_and_504_l2240_224023


namespace NUMINAMATH_CALUDE_min_workers_for_profit_is_16_l2240_224002

/-- Represents the minimum number of workers required for a manufacturing plant to make a profit -/
def min_workers_for_profit (
  maintenance_cost : ℕ)  -- Daily maintenance cost in dollars
  (hourly_wage : ℕ)      -- Hourly wage per worker in dollars
  (widgets_per_hour : ℕ) -- Number of widgets produced per worker per hour
  (widget_price : ℕ)     -- Selling price of each widget in dollars
  (work_hours : ℕ)       -- Number of work hours per day
  : ℕ :=
  16

/-- Theorem stating that given the specific conditions, the minimum number of workers for profit is 16 -/
theorem min_workers_for_profit_is_16 :
  min_workers_for_profit 600 20 4 4 10 = 16 := by
  sorry

#eval min_workers_for_profit 600 20 4 4 10

end NUMINAMATH_CALUDE_min_workers_for_profit_is_16_l2240_224002


namespace NUMINAMATH_CALUDE_polynomial_equality_l2240_224078

theorem polynomial_equality (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (3 - 2*x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  (a₀ + a₂ + a₄)^2 - (a₁ + a₃ + a₅)^2 = 3125 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l2240_224078


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2240_224072

theorem polynomial_factorization (x : ℝ) : 
  x^6 + 2*x^4 - x^2 - 2 = (x - 1) * (x + 1) * (x^2 + 1) * (x^2 + 2) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2240_224072


namespace NUMINAMATH_CALUDE_smallest_sum_of_two_distinct_primes_greater_than_500_l2240_224035

def is_prime (n : ℕ) : Prop := sorry

def sum_of_two_distinct_primes_greater_than_500 (n : ℕ) : Prop :=
  ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p > 500 ∧ q > 500 ∧ p ≠ q ∧ n = p + q

theorem smallest_sum_of_two_distinct_primes_greater_than_500 :
  (∀ m : ℕ, sum_of_two_distinct_primes_greater_than_500 m → m ≥ 1012) ∧
  sum_of_two_distinct_primes_greater_than_500 1012 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_two_distinct_primes_greater_than_500_l2240_224035


namespace NUMINAMATH_CALUDE_equilateral_triangle_sticks_l2240_224074

def canFormEquilateralTriangle (n : ℕ) : Prop :=
  ∃ (side : ℕ), 3 * side = n * (n + 1) / 2

theorem equilateral_triangle_sticks (n : ℕ) :
  canFormEquilateralTriangle n ↔ n ≥ 5 ∧ (n % 6 = 0 ∨ n % 6 = 2 ∨ n % 6 = 3 ∨ n % 6 = 5) :=
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_sticks_l2240_224074


namespace NUMINAMATH_CALUDE_distance_to_big_rock_l2240_224054

/-- The distance to Big Rock given the rower's speed, river current, and round trip time -/
theorem distance_to_big_rock 
  (rower_speed : ℝ) 
  (river_current : ℝ) 
  (round_trip_time : ℝ) 
  (h1 : rower_speed = 7)
  (h2 : river_current = 1)
  (h3 : round_trip_time = 1) : 
  ∃ (distance : ℝ), distance = 24 / 7 ∧ 
    (distance / (rower_speed - river_current) + 
     distance / (rower_speed + river_current) = round_trip_time) :=
by sorry

end NUMINAMATH_CALUDE_distance_to_big_rock_l2240_224054


namespace NUMINAMATH_CALUDE_total_sum_is_71_rupees_l2240_224067

/-- Calculates the total sum of money in rupees given the number of 20 paise and 25 paise coins -/
def total_sum_rupees (total_coins : ℕ) (coins_20_paise : ℕ) : ℚ :=
  let coins_25_paise := total_coins - coins_20_paise
  let sum_20_paise := (coins_20_paise : ℚ) * (20 : ℚ) / 100
  let sum_25_paise := (coins_25_paise : ℚ) * (25 : ℚ) / 100
  sum_20_paise + sum_25_paise

/-- Theorem stating that given 324 total coins with 200 coins of 20 paise, the total sum is 71 rupees -/
theorem total_sum_is_71_rupees :
  total_sum_rupees 324 200 = 71 := by
  sorry

end NUMINAMATH_CALUDE_total_sum_is_71_rupees_l2240_224067


namespace NUMINAMATH_CALUDE_binomial_cube_expansion_problem_solution_l2240_224065

theorem binomial_cube_expansion (n : ℕ) : n^3 + 3*(n^2) + 3*n + 1 = (n+1)^3 := by
  sorry

theorem problem_solution : 98^3 + 3*(98^2) + 3*98 + 1 = 99^3 := by
  sorry

end NUMINAMATH_CALUDE_binomial_cube_expansion_problem_solution_l2240_224065


namespace NUMINAMATH_CALUDE_unique_congruence_in_range_l2240_224052

theorem unique_congruence_in_range : ∃! n : ℤ,
  5 ≤ n ∧ n ≤ 10 ∧ n ≡ 10543 [ZMOD 7] ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_unique_congruence_in_range_l2240_224052


namespace NUMINAMATH_CALUDE_permutations_6_3_l2240_224001

/-- The number of permutations of k elements chosen from a set of n elements -/
def permutations (n k : ℕ) : ℕ := 
  Nat.factorial n / Nat.factorial (n - k)

/-- Theorem: The number of permutations of 3 elements chosen from a set of 6 elements is 120 -/
theorem permutations_6_3 : permutations 6 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_permutations_6_3_l2240_224001


namespace NUMINAMATH_CALUDE_max_reciprocal_sum_l2240_224088

theorem max_reciprocal_sum (x y a b : ℝ) 
  (ha : a > 1) (hb : b > 1)
  (hax : a^x = 3) (hby : b^y = 3) 
  (hab : a + b = 2 * Real.sqrt 3) : 
  (∀ z w : ℝ, a^z = 3 → b^w = 3 → 1/z + 1/w ≤ 1) :=
sorry

end NUMINAMATH_CALUDE_max_reciprocal_sum_l2240_224088


namespace NUMINAMATH_CALUDE_students_neither_art_nor_music_l2240_224040

theorem students_neither_art_nor_music 
  (total : ℕ) (art : ℕ) (music : ℕ) (both : ℕ) :
  total = 75 →
  art = 45 →
  music = 50 →
  both = 30 →
  total - (art + music - both) = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_students_neither_art_nor_music_l2240_224040


namespace NUMINAMATH_CALUDE_max_subsets_l2240_224073

-- Define the set S
def S : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

-- Define the property for subsets A₁, A₂, ..., Aₖ
def valid_subsets (A : Finset (Finset ℕ)) : Prop :=
  ∀ X ∈ A, X ⊆ S ∧ X.card = 5 ∧ ∀ Y ∈ A, X ≠ Y → (X ∩ Y).card ≤ 2

-- Theorem statement
theorem max_subsets :
  ∀ A : Finset (Finset ℕ), valid_subsets A → A.card ≤ 6 :=
sorry

end NUMINAMATH_CALUDE_max_subsets_l2240_224073


namespace NUMINAMATH_CALUDE_equation_is_linear_l2240_224089

/-- A linear equation with two variables is of the form ax + by = c, where a and b are not both zero -/
def is_linear_equation_two_vars (f : ℝ → ℝ → Prop) : Prop :=
  ∃ (a b c : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧ ∀ x y, f x y ↔ a * x + b * y = c

/-- The equation x + y = 2 -/
def equation (x y : ℝ) : Prop := x + y = 2

theorem equation_is_linear : is_linear_equation_two_vars equation :=
sorry

end NUMINAMATH_CALUDE_equation_is_linear_l2240_224089


namespace NUMINAMATH_CALUDE_revenue_is_288_l2240_224076

/-- Represents the rental business with canoes and kayaks -/
structure RentalBusiness where
  canoe_price : ℕ
  kayak_price : ℕ
  canoe_kayak_ratio : ℚ
  canoe_kayak_difference : ℕ

/-- Calculates the total revenue for a day given the rental business conditions -/
def calculate_revenue (business : RentalBusiness) : ℕ :=
  let kayaks := business.canoe_kayak_difference * 2
  let canoes := kayaks + business.canoe_kayak_difference
  kayaks * business.kayak_price + canoes * business.canoe_price

/-- Theorem stating that the total revenue for the day is $288 -/
theorem revenue_is_288 (business : RentalBusiness) 
    (h1 : business.canoe_price = 14)
    (h2 : business.kayak_price = 15)
    (h3 : business.canoe_kayak_ratio = 3 / 2)
    (h4 : business.canoe_kayak_difference = 4) :
  calculate_revenue business = 288 := by
  sorry

#eval calculate_revenue { 
  canoe_price := 14, 
  kayak_price := 15, 
  canoe_kayak_ratio := 3 / 2, 
  canoe_kayak_difference := 4 
}

end NUMINAMATH_CALUDE_revenue_is_288_l2240_224076


namespace NUMINAMATH_CALUDE_circle_area_with_diameter_10_l2240_224010

/-- The area of a circle with diameter 10 meters is 25π square meters. -/
theorem circle_area_with_diameter_10 :
  let diameter : ℝ := 10
  let radius : ℝ := diameter / 2
  let area : ℝ := π * radius ^ 2
  area = 25 * π :=
by sorry

end NUMINAMATH_CALUDE_circle_area_with_diameter_10_l2240_224010


namespace NUMINAMATH_CALUDE_increasing_f_range_l2240_224070

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (2 - a) * x + 1 else a^x

theorem increasing_f_range (a : ℝ) 
  (h1 : a > 0) (h2 : a ≠ 1) 
  (h3 : ∀ x y : ℝ, x < y → f a x < f a y) : 
  a ∈ Set.Icc (3/2) 2 ∧ a < 2 :=
sorry

end NUMINAMATH_CALUDE_increasing_f_range_l2240_224070

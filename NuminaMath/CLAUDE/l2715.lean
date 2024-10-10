import Mathlib

namespace smallest_mn_for_almost_shaded_square_l2715_271505

theorem smallest_mn_for_almost_shaded_square (m n : ℕ+) 
  (h_bound : 2 * n < m ∧ m < 3 * n) 
  (h_exists : ∃ (p q : ℕ) (k : ℤ), 
    p < m ∧ q < n ∧ 
    0 < (m * q - n * p) * (m * q - n * p) ∧ 
    (m * q - n * p) * (m * q - n * p) < 2 * m * n / 1000) :
  506 ≤ m * n ∧ m * n ≤ 510 := by
sorry

end smallest_mn_for_almost_shaded_square_l2715_271505


namespace quadratic_equation_solution_l2715_271599

theorem quadratic_equation_solution : 
  let f : ℝ → ℝ := λ x ↦ x^2 - 2*x - 1
  ∃ x₁ x₂ : ℝ, x₁ = 1 + Real.sqrt 2 ∧ x₂ = 1 - Real.sqrt 2 ∧ f x₁ = 0 ∧ f x₂ = 0 :=
by sorry

end quadratic_equation_solution_l2715_271599


namespace necessary_and_sufficient_condition_l2715_271587

theorem necessary_and_sufficient_condition : 
  (∀ x : ℝ, x = 1 ↔ x^2 - 2*x + 1 = 0) := by sorry

end necessary_and_sufficient_condition_l2715_271587


namespace bijective_function_exists_l2715_271579

/-- A function that maps elements of ℤm × ℤn to itself -/
def bijective_function (m n : ℕ+) : (Fin m × Fin n) → (Fin m × Fin n) := sorry

/-- Predicate to check if all f(v) + v are pairwise distinct -/
def all_distinct (m n : ℕ+) (f : (Fin m × Fin n) → (Fin m × Fin n)) : Prop := sorry

/-- Main theorem statement -/
theorem bijective_function_exists (m n : ℕ+) :
  (∃ f : (Fin m × Fin n) → (Fin m × Fin n), Function.Bijective f ∧ all_distinct m n f) ↔
  (m.val % 2 = n.val % 2) := by sorry

end bijective_function_exists_l2715_271579


namespace simplify_expression_l2715_271516

theorem simplify_expression (z : ℝ) : z - 2*z + 4*z - 6 + 3 + 7 - 2 = 3*z + 2 := by
  sorry

end simplify_expression_l2715_271516


namespace rafael_hourly_rate_l2715_271517

theorem rafael_hourly_rate (monday_hours : ℕ) (tuesday_hours : ℕ) (remaining_hours : ℕ) (total_earnings : ℕ) :
  monday_hours = 10 →
  tuesday_hours = 8 →
  remaining_hours = 20 →
  total_earnings = 760 →
  (total_earnings : ℚ) / (monday_hours + tuesday_hours + remaining_hours : ℚ) = 20 := by
  sorry

end rafael_hourly_rate_l2715_271517


namespace logarithmic_equation_solution_l2715_271530

theorem logarithmic_equation_solution (a : ℝ) (ha : a > 0) :
  ∃ x : ℝ, x > 1 ∧ Real.log (a * x) = 2 * Real.log (x - 1) ↔
  ∃ x : ℝ, x = (2 + a + Real.sqrt (a^2 + 4*a)) / 2 :=
by sorry

end logarithmic_equation_solution_l2715_271530


namespace monotone_decreasing_implies_a_leq_neg_four_l2715_271532

/-- A quadratic function f(x) = x^2 + 2ax + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x + 2

/-- The property of f being monotonically decreasing on (-∞, 4] -/
def is_monotone_decreasing_on_interval (a : ℝ) : Prop :=
  ∀ x y, x < y → y ≤ 4 → f a x > f a y

/-- The main theorem: if f is monotonically decreasing on (-∞, 4], then a ≤ -4 -/
theorem monotone_decreasing_implies_a_leq_neg_four (a : ℝ) :
  is_monotone_decreasing_on_interval a → a ≤ -4 :=
by
  sorry

end monotone_decreasing_implies_a_leq_neg_four_l2715_271532


namespace expression_evaluation_l2715_271500

theorem expression_evaluation :
  let x : ℤ := -2
  (x^2 + 7*x - 8) = -18 := by sorry

end expression_evaluation_l2715_271500


namespace frequency_problem_l2715_271575

theorem frequency_problem (sample_size : ℕ) (num_groups : ℕ) 
  (common_diff : ℚ) (last_seven_sum : ℚ) : 
  sample_size = 1000 →
  num_groups = 10 →
  common_diff = 0.05 →
  last_seven_sum = 0.79 →
  ∃ (x : ℚ), 
    x > 0 ∧ 
    x + common_diff > 0 ∧ 
    x + 2 * common_diff > 0 ∧
    x + (x + common_diff) + (x + 2 * common_diff) + last_seven_sum = 1 →
    (x * sample_size : ℚ) = 20 :=
by sorry

end frequency_problem_l2715_271575


namespace sand_weight_difference_l2715_271501

/-- Proves that the sand in the box is heavier than the sand in the barrel by 260 grams --/
theorem sand_weight_difference 
  (barrel_weight : ℕ) 
  (barrel_with_sand_weight : ℕ) 
  (box_weight : ℕ) 
  (box_with_sand_weight : ℕ) 
  (h1 : barrel_weight = 250)
  (h2 : barrel_with_sand_weight = 1780)
  (h3 : box_weight = 460)
  (h4 : box_with_sand_weight = 2250) :
  (box_with_sand_weight - box_weight) - (barrel_with_sand_weight - barrel_weight) = 260 := by
  sorry

#check sand_weight_difference

end sand_weight_difference_l2715_271501


namespace A_intersect_B_eq_open_interval_l2715_271595

-- Define set A
def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}

-- Define set B
def B : Set ℝ := {y | ∃ x, y = Real.exp x}

-- Theorem statement
theorem A_intersect_B_eq_open_interval : A ∩ B = Set.Ioo 0 3 := by sorry

end A_intersect_B_eq_open_interval_l2715_271595


namespace problem_solution_l2715_271540

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 4| + |x - 4|

-- Theorem statement
theorem problem_solution :
  -- Part 1: Solution set of f(x) ≥ 10
  (∀ x, f x ≥ 10 ↔ x ∈ Set.Iic (-10/3) ∪ Set.Ici 2) ∧
  -- Part 2: Minimum value of f(x) is 6
  (∃ x, f x = 6 ∧ ∀ y, f y ≥ f x) ∧
  -- Part 3: Inequality for positive real numbers a, b, c
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a + b + c = 6 →
    1 / (a + b) + 1 / (b + c) + 1 / (c + a) ≥ 3 / 4) :=
by sorry

end problem_solution_l2715_271540


namespace smallest_value_w_cube_plus_z_cube_l2715_271521

theorem smallest_value_w_cube_plus_z_cube (w z : ℂ) 
  (h1 : Complex.abs (w + z) = 2)
  (h2 : Complex.abs (w^2 + z^2) = 18) :
  Complex.abs (w^3 + z^3) = 50 := by
sorry

end smallest_value_w_cube_plus_z_cube_l2715_271521


namespace pizza_payment_difference_l2715_271537

/-- Represents the pizza sharing scenario between Doug and Dave -/
structure PizzaSharing where
  total_slices : ℕ
  plain_cost : ℚ
  topping_cost : ℚ
  topped_slices : ℕ
  dave_plain_slices : ℕ

/-- Calculates the cost per slice given the total cost and number of slices -/
def cost_per_slice (total_cost : ℚ) (total_slices : ℕ) : ℚ :=
  total_cost / total_slices

/-- Calculates the payment difference between Dave and Doug -/
def payment_difference (ps : PizzaSharing) : ℚ :=
  let total_cost := ps.plain_cost + ps.topping_cost
  let per_slice_cost := cost_per_slice total_cost ps.total_slices
  let dave_slices := ps.topped_slices + ps.dave_plain_slices
  let doug_slices := ps.total_slices - dave_slices
  dave_slices * per_slice_cost - doug_slices * per_slice_cost

/-- Theorem stating that the payment difference is 2.8 under the given conditions -/
theorem pizza_payment_difference :
  let ps : PizzaSharing := {
    total_slices := 10,
    plain_cost := 10,
    topping_cost := 4,
    topped_slices := 4,
    dave_plain_slices := 2
  }
  payment_difference ps = 2.8 := by
  sorry


end pizza_payment_difference_l2715_271537


namespace function_equivalence_l2715_271573

theorem function_equivalence (x : ℝ) : (x^3 + x) / (x^2 + 1) = x := by
  sorry

end function_equivalence_l2715_271573


namespace equation_solution_l2715_271584

theorem equation_solution (x y : ℕ) : 
  (x^2 + 1)^y - (x^2 - 1)^y = 2*x^y ↔ 
  (x = 1 ∧ y = 1) ∨ (x = 0 ∧ ∃ k : ℕ, y = 2*k + 2) :=
by sorry

end equation_solution_l2715_271584


namespace john_vacation_expenses_l2715_271592

def octal_to_decimal (n : Nat) : Nat :=
  let digits := n.digits 8
  (List.range digits.length).foldl (fun acc i => acc + digits[i]! * (8 ^ i)) 0

theorem john_vacation_expenses :
  octal_to_decimal 5555 - 1500 = 1425 := by
  sorry

end john_vacation_expenses_l2715_271592


namespace least_sum_of_exponents_for_520_l2715_271576

theorem least_sum_of_exponents_for_520 (n : ℕ) (h1 : n = 520) :
  ∃ (a b : ℕ), 
    n = 2^a + 2^b ∧ 
    a ≠ b ∧ 
    (a = 3 ∨ b = 3) ∧ 
    ∀ (c d : ℕ), (n = 2^c + 2^d ∧ c ≠ d ∧ (c = 3 ∨ d = 3)) → a + b ≤ c + d :=
by sorry

end least_sum_of_exponents_for_520_l2715_271576


namespace pipeA_rate_correct_l2715_271566

/-- Represents the rate at which Pipe A fills the tank -/
def pipeA_rate : ℝ := 40

/-- The capacity of the tank in liters -/
def tank_capacity : ℝ := 750

/-- The rate at which Pipe B fills the tank in liters per minute -/
def pipeB_rate : ℝ := 30

/-- The rate at which Pipe C drains the tank in liters per minute -/
def pipeC_rate : ℝ := 20

/-- The time in minutes it takes to fill the tank -/
def fill_time : ℝ := 45

/-- The duration of one cycle in minutes -/
def cycle_duration : ℝ := 3

/-- Theorem stating that the rate of Pipe A is correct given the conditions -/
theorem pipeA_rate_correct : 
  tank_capacity = (fill_time / cycle_duration) * (pipeA_rate + pipeB_rate - pipeC_rate) :=
by sorry

end pipeA_rate_correct_l2715_271566


namespace cube_root_difference_theorem_l2715_271560

theorem cube_root_difference_theorem (x : ℝ) 
  (h1 : x > 0) 
  (h2 : (1 - x^3)^(1/3) - (1 + x^3)^(1/3) = 1) : 
  x^3 = (x^2 * (28^(1/9))) / 3 := by
  sorry

end cube_root_difference_theorem_l2715_271560


namespace stock_price_fluctuation_l2715_271502

theorem stock_price_fluctuation (original_price : ℝ) (increase_percent : ℝ) (decrease_percent : ℝ) :
  increase_percent = 0.40 →
  decrease_percent = 2 / 7 →
  original_price * (1 + increase_percent) * (1 - decrease_percent) = original_price :=
by sorry

end stock_price_fluctuation_l2715_271502


namespace fish_tank_ratio_l2715_271510

/-- The number of fish in the first tank -/
def first_tank : ℕ := 7 + 8

/-- The number of fish in the second tank -/
def second_tank : ℕ := 2 * first_tank

/-- The number of fish in the third tank -/
def third_tank : ℕ := 10

theorem fish_tank_ratio : 
  (third_tank : ℚ) / second_tank = 1 / 3 := by sorry

end fish_tank_ratio_l2715_271510


namespace ellipse_circle_centers_distance_l2715_271547

/-- Given an ellipse with center O and semi-axes a and b, and a circle with radius r 
    and center C on the major semi-axis of the ellipse that touches the ellipse at two points, 
    prove that the square of the distance between the centers of the ellipse and the circle 
    is equal to ((a^2 - b^2) * (b^2 - r^2)) / b^2. -/
theorem ellipse_circle_centers_distance 
  (O : ℝ × ℝ) (C : ℝ × ℝ) (a b r : ℝ) : 
  (a > 0) → (b > 0) → (r > 0) → (a ≥ b) →
  (∃ (P Q : ℝ × ℝ), 
    (P.1 - O.1)^2 / a^2 + (P.2 - O.2)^2 / b^2 = 1 ∧
    (Q.1 - O.1)^2 / a^2 + (Q.2 - O.2)^2 / b^2 = 1 ∧
    (P.1 - C.1)^2 + (P.2 - C.2)^2 = r^2 ∧
    (Q.1 - C.1)^2 + (Q.2 - C.2)^2 = r^2 ∧
    C.2 = O.2) →
  (C.1 - O.1)^2 = ((a^2 - b^2) * (b^2 - r^2)) / b^2 := by
  sorry

end ellipse_circle_centers_distance_l2715_271547


namespace shorter_base_length_l2715_271504

/-- A trapezoid with given properties -/
structure Trapezoid where
  long_base : ℝ
  short_base : ℝ
  midpoint_segment : ℝ

/-- The trapezoid satisfies the given conditions -/
def satisfies_conditions (t : Trapezoid) : Prop :=
  t.long_base = 85 ∧ t.midpoint_segment = 5

/-- Theorem: In a trapezoid satisfying the given conditions, the shorter base is 75 -/
theorem shorter_base_length (t : Trapezoid) (h : satisfies_conditions t) : 
  t.short_base = 75 := by
  sorry

end shorter_base_length_l2715_271504


namespace normal_distribution_probability_l2715_271554

/-- A random variable following a normal distribution with mean 2 and standard deviation 1. -/
def ξ : Real → Real := sorry

/-- The probability density function of the standard normal distribution. -/
noncomputable def φ : Real → Real := sorry

/-- The cumulative distribution function of the standard normal distribution. -/
noncomputable def Φ : Real → Real := sorry

/-- The probability that ξ is greater than 3. -/
def P_gt_3 : Real := 0.023

theorem normal_distribution_probability (h : P_gt_3 = 1 - Φ 1) : 
  Φ 1 - Φ (-1) = 0.954 := by sorry

end normal_distribution_probability_l2715_271554


namespace smallest_band_size_l2715_271541

theorem smallest_band_size : ∃ n : ℕ, 
  n > 0 ∧ 
  n % 6 = 5 ∧ 
  n % 5 = 4 ∧ 
  n % 7 = 6 ∧ 
  (∀ m : ℕ, m > 0 → m % 6 = 5 → m % 5 = 4 → m % 7 = 6 → m ≥ n) ∧
  n = 119 :=
by
  sorry

end smallest_band_size_l2715_271541


namespace determinant_max_value_l2715_271544

theorem determinant_max_value (θ : ℝ) :
  (∀ θ', -Real.sin (4 * θ') / 2 ≤ -Real.sin (4 * θ) / 2) →
  -Real.sin (4 * θ) / 2 = 1/2 := by
sorry

end determinant_max_value_l2715_271544


namespace cost_price_per_meter_l2715_271507

theorem cost_price_per_meter (cloth_length : ℕ) (selling_price : ℕ) (profit_per_meter : ℕ) 
  (h1 : cloth_length = 85)
  (h2 : selling_price = 8925)
  (h3 : profit_per_meter = 25) : 
  (selling_price - cloth_length * profit_per_meter) / cloth_length = 80 :=
by sorry

end cost_price_per_meter_l2715_271507


namespace paint_theorem_l2715_271558

def paint_problem (initial_paint : ℚ) (first_day_fraction : ℚ) (second_day_fraction : ℚ) : Prop :=
  let remaining_after_first_day := initial_paint - (first_day_fraction * initial_paint)
  let used_second_day := second_day_fraction * remaining_after_first_day
  let remaining_after_second_day := remaining_after_first_day - used_second_day
  remaining_after_second_day = (4 : ℚ) / 9 * initial_paint

theorem paint_theorem : 
  paint_problem 1 (1/3) (1/3) := by sorry

end paint_theorem_l2715_271558


namespace restaurant_bill_fraction_l2715_271590

theorem restaurant_bill_fraction (akshitha veena lasya total : ℚ) : 
  akshitha = (3 / 4) * veena →
  veena = (1 / 2) * lasya →
  total = akshitha + veena + lasya →
  veena / total = 4 / 15 := by
sorry

end restaurant_bill_fraction_l2715_271590


namespace direction_vector_k_l2715_271568

/-- The direction vector of the line passing through points A(0,2) and B(-1,0) is (1,k). -/
theorem direction_vector_k (k : ℝ) : 
  let A : ℝ × ℝ := (0, 2)
  let B : ℝ × ℝ := (-1, 0)
  let direction_vector : ℝ × ℝ := (1, k)
  (direction_vector.1 * (B.1 - A.1) = direction_vector.2 * (B.2 - A.2)) → k = 2 :=
by
  sorry


end direction_vector_k_l2715_271568


namespace min_students_four_correct_is_eight_l2715_271578

-- Define the total number of students
def total_students : ℕ := 100

-- Define the number of students who performed each spell correctly
def spell1_correct : ℕ := 95
def spell2_correct : ℕ := 75
def spell3_correct : ℕ := 97
def spell4_correct : ℕ := 95
def spell5_correct : ℕ := 96

-- Define the function to calculate the minimum number of students who performed exactly 4 out of 5 spells correctly
def min_students_four_correct : ℕ :=
  total_students - spell2_correct - (total_students - spell1_correct) - (total_students - spell3_correct) - (total_students - spell4_correct) - (total_students - spell5_correct)

-- Theorem statement
theorem min_students_four_correct_is_eight :
  min_students_four_correct = 8 :=
sorry

end min_students_four_correct_is_eight_l2715_271578


namespace systematic_sampling_result_l2715_271562

/-- Represents a systematic sampling setup -/
structure SystematicSampling where
  totalPopulation : Nat
  sampleSize : Nat
  interval : Nat
  startingNumber : Nat

/-- Calculates the selected number within a given range for a systematic sampling -/
def selectedNumber (s : SystematicSampling) (rangeStart rangeEnd : Nat) : Nat :=
  let adjustedStart := (rangeStart - s.startingNumber) / s.interval * s.interval + s.startingNumber
  if adjustedStart < rangeStart then
    adjustedStart + s.interval
  else
    adjustedStart

/-- Theorem stating that for the given systematic sampling, the selected number in the range 033 to 048 is 039 -/
theorem systematic_sampling_result :
  let s : SystematicSampling := {
    totalPopulation := 800,
    sampleSize := 50,
    interval := 16,
    startingNumber := 7
  }
  selectedNumber s 33 48 = 39 := by
  sorry


end systematic_sampling_result_l2715_271562


namespace sum_of_vectors_is_zero_l2715_271596

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

/-- Given vectors a, b, c in a real vector space V, prove that their sum is zero
    under the given conditions. -/
theorem sum_of_vectors_is_zero (a b c : V)
  (not_collinear_ab : ¬ Collinear ℝ ({0, a, b} : Set V))
  (not_collinear_bc : ¬ Collinear ℝ ({0, b, c} : Set V))
  (not_collinear_ca : ¬ Collinear ℝ ({0, c, a} : Set V))
  (collinear_ab_c : Collinear ℝ ({0, a + b, c} : Set V))
  (collinear_bc_a : Collinear ℝ ({0, b + c, a} : Set V)) :
  a + b + c = (0 : V) := by
sorry

end sum_of_vectors_is_zero_l2715_271596


namespace combinations_permutations_relation_l2715_271520

/-- The number of combinations of n elements taken k at a time -/
def C (n k : ℕ) : ℕ := sorry

/-- The number of permutations of k elements from an n-element set -/
def A (n k : ℕ) : ℕ := sorry

/-- Theorem: The number of combinations is equal to the number of permutations divided by k factorial -/
theorem combinations_permutations_relation (n k : ℕ) : C n k = A n k / k! := by
  sorry

end combinations_permutations_relation_l2715_271520


namespace fold_points_area_l2715_271556

-- Define the triangle DEF
def triangle_DEF (D E F : ℝ × ℝ) : Prop :=
  let de := Real.sqrt ((E.1 - D.1)^2 + (E.2 - D.2)^2)
  let df := Real.sqrt ((F.1 - D.1)^2 + (F.2 - D.2)^2)
  let angle_e := Real.arccos ((de^2 + df^2 - (F.1 - E.1)^2 - (F.2 - E.2)^2) / (2 * de * df))
  de = 48 ∧ df = 96 ∧ angle_e = Real.pi / 2

-- Define the area of fold points
def area_fold_points (D E F : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem fold_points_area (D E F : ℝ × ℝ) :
  triangle_DEF D E F →
  area_fold_points D E F = 432 * Real.pi - 518 * Real.sqrt 3 :=
sorry

end fold_points_area_l2715_271556


namespace function_inequality_l2715_271524

open Set

theorem function_inequality (f g : ℝ → ℝ) (a b x : ℝ) :
  DifferentiableOn ℝ f (Icc a b) →
  DifferentiableOn ℝ g (Icc a b) →
  (∀ y ∈ Icc a b, deriv f y > deriv g y) →
  a < x →
  x < b →
  f x + g a > g x + f a :=
by sorry

end function_inequality_l2715_271524


namespace min_sum_with_gcd_conditions_l2715_271534

theorem min_sum_with_gcd_conditions :
  ∃ (a b c : ℕ+),
    (Nat.gcd a b > 1) ∧
    (Nat.gcd b c > 1) ∧
    (Nat.gcd c a > 1) ∧
    (Nat.gcd a (Nat.gcd b c) = 1) ∧
    (a + b + c = 31) ∧
    (∀ (x y z : ℕ+),
      (Nat.gcd x y > 1) →
      (Nat.gcd y z > 1) →
      (Nat.gcd z x > 1) →
      (Nat.gcd x (Nat.gcd y z) = 1) →
      (x + y + z ≥ 31)) :=
by sorry

end min_sum_with_gcd_conditions_l2715_271534


namespace xiaoming_estimate_larger_l2715_271508

/-- Rounds a number up to the nearest ten -/
def roundUp (n : ℤ) : ℤ := sorry

/-- Rounds a number down to the nearest ten -/
def roundDown (n : ℤ) : ℤ := sorry

theorem xiaoming_estimate_larger (x y : ℤ) (hx : x > 0) (hy : y > 0) (hxy : x > y) :
  roundUp x - roundDown y > x - y := by sorry

end xiaoming_estimate_larger_l2715_271508


namespace cost_of_three_batches_l2715_271597

/-- Represents the cost and quantity of ingredients for yogurt production -/
structure YogurtProduction where
  milk_price : ℝ
  fruit_price : ℝ
  milk_per_batch : ℝ
  fruit_per_batch : ℝ

/-- Calculates the cost of producing a given number of yogurt batches -/
def cost_of_batches (y : YogurtProduction) (num_batches : ℝ) : ℝ :=
  num_batches * (y.milk_price * y.milk_per_batch + y.fruit_price * y.fruit_per_batch)

/-- Theorem: The cost of producing three batches of yogurt is $63 -/
theorem cost_of_three_batches :
  ∃ (y : YogurtProduction),
    y.milk_price = 1.5 ∧
    y.fruit_price = 2 ∧
    y.milk_per_batch = 10 ∧
    y.fruit_per_batch = 3 ∧
    cost_of_batches y 3 = 63 := by
  sorry

end cost_of_three_batches_l2715_271597


namespace simplify_expression_l2715_271593

theorem simplify_expression (a : ℝ) : (-2 * a) * a - (-2 * a)^2 = -6 * a^2 := by
  sorry

end simplify_expression_l2715_271593


namespace complex_square_simplification_l2715_271522

theorem complex_square_simplification : 
  let i : ℂ := Complex.I
  (4 - 3*i)^2 = 7 - 24*i :=
by sorry

end complex_square_simplification_l2715_271522


namespace max_sum_abc_l2715_271559

theorem max_sum_abc (a b c : ℝ) (ha : a ≥ 1) (hb : b ≥ 1) (hc : c ≥ 1)
  (h : a * b * c + 2 * a^2 + 2 * b^2 + 2 * c^2 + c * a - c * b - 4 * a + 4 * b - c = 28) :
  a + b + c ≤ 6 :=
sorry

end max_sum_abc_l2715_271559


namespace rectangular_field_area_l2715_271518

theorem rectangular_field_area (w : ℝ) (d : ℝ) (h1 : w = 15) (h2 : d = 17) :
  ∃ l : ℝ, w * l = 120 ∧ d^2 = w^2 + l^2 := by
  sorry

end rectangular_field_area_l2715_271518


namespace baguettes_left_at_end_of_day_l2715_271545

/-- The number of baguettes left at the end of the day in a bakery --/
def baguettes_left (batches_per_day : ℕ) (baguettes_per_batch : ℕ) 
  (sold_after_first : ℕ) (sold_after_second : ℕ) (sold_after_third : ℕ) : ℕ :=
  let total_baguettes := batches_per_day * baguettes_per_batch
  let left_after_first := baguettes_per_batch - sold_after_first
  let left_after_second := (baguettes_per_batch + left_after_first) - sold_after_second
  let left_after_third := (baguettes_per_batch + left_after_second) - sold_after_third
  left_after_third

/-- Theorem stating the number of baguettes left at the end of the day --/
theorem baguettes_left_at_end_of_day :
  baguettes_left 3 48 37 52 49 = 6 := by
  sorry

end baguettes_left_at_end_of_day_l2715_271545


namespace total_price_is_23_l2715_271525

/-- The price of cucumbers in dollars per kilogram -/
def cucumber_price : ℝ := 5

/-- The price of tomatoes in dollars per kilogram -/
def tomato_price : ℝ := cucumber_price * (1 - 0.2)

/-- The total price of tomatoes and cucumbers -/
def total_price : ℝ := 2 * tomato_price + 3 * cucumber_price

theorem total_price_is_23 : total_price = 23 := by
  sorry

end total_price_is_23_l2715_271525


namespace exact_two_females_one_male_probability_l2715_271574

def total_contestants : ℕ := 8
def female_contestants : ℕ := 5
def male_contestants : ℕ := 3
def selected_contestants : ℕ := 3

theorem exact_two_females_one_male_probability :
  (Nat.choose female_contestants 2 * Nat.choose male_contestants 1) / 
  Nat.choose total_contestants selected_contestants = 15 / 28 := by
  sorry

end exact_two_females_one_male_probability_l2715_271574


namespace ratio_of_sum_and_difference_l2715_271542

theorem ratio_of_sum_and_difference (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) (h : a + b = 7 * (a - b)) : a / b = 4 / 3 := by
  sorry

end ratio_of_sum_and_difference_l2715_271542


namespace pitcher_juice_distribution_l2715_271570

theorem pitcher_juice_distribution :
  ∀ (C : ℝ),
  C > 0 →
  let pineapple_juice := (1/2 : ℝ) * C
  let orange_juice := (1/4 : ℝ) * C
  let total_juice := pineapple_juice + orange_juice
  let cups := 4
  let juice_per_cup := total_juice / cups
  (juice_per_cup / C) * 100 = 18.75 :=
by
  sorry

end pitcher_juice_distribution_l2715_271570


namespace sqrt_equation_solution_l2715_271591

theorem sqrt_equation_solution (a b : ℕ+) (h1 : a < b) :
  Real.sqrt (1 + Real.sqrt (25 + 20 * Real.sqrt 2)) = Real.sqrt a + Real.sqrt b →
  a = 2 ∧ b = 8 := by
sorry

end sqrt_equation_solution_l2715_271591


namespace annual_mischief_convention_handshakes_l2715_271582

/-- The number of handshakes at the Annual Mischief Convention -/
theorem annual_mischief_convention_handshakes (n_gremlins : ℕ) (n_imps : ℕ) : 
  n_gremlins = 30 → n_imps = 15 → 
  (n_gremlins * (n_gremlins - 1)) / 2 + n_imps * (n_gremlins / 2) = 660 := by
  sorry

end annual_mischief_convention_handshakes_l2715_271582


namespace composite_shape_area_l2715_271523

-- Define the dimensions of the rectangles
def rect1_width : ℕ := 6
def rect1_height : ℕ := 7
def rect2_width : ℕ := 3
def rect2_height : ℕ := 5
def rect3_width : ℕ := 5
def rect3_height : ℕ := 6

-- Define the function to calculate the area of a rectangle
def rectangle_area (width height : ℕ) : ℕ := width * height

-- Theorem statement
theorem composite_shape_area :
  rectangle_area rect1_width rect1_height +
  rectangle_area rect2_width rect2_height +
  rectangle_area rect3_width rect3_height = 87 := by
  sorry


end composite_shape_area_l2715_271523


namespace missing_root_l2715_271546

theorem missing_root (x : ℝ) : x^2 - 2*x = 0 → (x = 2 ∨ x = 0) := by
  sorry

end missing_root_l2715_271546


namespace solution_count_implies_n_l2715_271586

/-- The number of solutions to the equation 3x + 2y + 4z = n in positive integers x, y, and z -/
def num_solutions (n : ℕ+) : ℕ :=
  (Finset.filter (fun (x, y, z) => 3 * x + 2 * y + 4 * z = n) (Finset.product (Finset.range n) (Finset.product (Finset.range n) (Finset.range n)))).card

/-- Theorem stating that if the equation 3x + 2y + 4z = n has exactly 30 solutions in positive integers,
    then n must be either 22 or 23 -/
theorem solution_count_implies_n (n : ℕ+) :
  num_solutions n = 30 → n = 22 ∨ n = 23 := by
  sorry

end solution_count_implies_n_l2715_271586


namespace unique_solution_to_equation_l2715_271531

theorem unique_solution_to_equation (x : ℝ) :
  x + 2 ≠ 0 →
  ((16 * x - x^2) / (x + 2) * (x + (16 - x) / (x + 2)) = 60) ↔ x = 4 := by
sorry

end unique_solution_to_equation_l2715_271531


namespace necessary_not_sufficient_condition_l2715_271571

/-- A quadratic equation x^2 + x + c = 0 has two real roots of opposite signs -/
def has_two_real_roots_opposite_signs (c : ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ x * y < 0 ∧ x^2 + x + c = 0 ∧ y^2 + y + c = 0

theorem necessary_not_sufficient_condition :
  (∀ c : ℝ, has_two_real_roots_opposite_signs c → c < 0) ∧
  (∃ c : ℝ, c < 0 ∧ ¬has_two_real_roots_opposite_signs c) :=
by sorry

end necessary_not_sufficient_condition_l2715_271571


namespace joans_cat_kittens_l2715_271539

/-- The number of kittens Joan has now -/
def total_kittens : ℕ := 10

/-- The number of kittens Joan got from her friends -/
def kittens_from_friends : ℕ := 2

/-- The number of kittens Joan's cat had -/
def cat_kittens : ℕ := total_kittens - kittens_from_friends

theorem joans_cat_kittens : cat_kittens = 8 := by
  sorry

end joans_cat_kittens_l2715_271539


namespace total_donation_equals_854_l2715_271589

/-- Represents a fundraising event with earnings and donation percentage -/
structure FundraisingEvent where
  earnings : ℝ
  donationPercentage : ℝ

/-- Calculates the donation amount for a fundraising event -/
def donationAmount (event : FundraisingEvent) : ℝ :=
  event.earnings * event.donationPercentage

/-- Theorem: The total donation from five fundraising events equals $854 -/
theorem total_donation_equals_854 
  (carWash : FundraisingEvent)
  (bakeSale : FundraisingEvent)
  (mowingLawns : FundraisingEvent)
  (handmadeCrafts : FundraisingEvent)
  (charityConcert : FundraisingEvent)
  (h1 : carWash.earnings = 200 ∧ carWash.donationPercentage = 0.9)
  (h2 : bakeSale.earnings = 160 ∧ bakeSale.donationPercentage = 0.8)
  (h3 : mowingLawns.earnings = 120 ∧ mowingLawns.donationPercentage = 1)
  (h4 : handmadeCrafts.earnings = 180 ∧ handmadeCrafts.donationPercentage = 0.7)
  (h5 : charityConcert.earnings = 500 ∧ charityConcert.donationPercentage = 0.6)
  : donationAmount carWash + donationAmount bakeSale + donationAmount mowingLawns + 
    donationAmount handmadeCrafts + donationAmount charityConcert = 854 := by
  sorry


end total_donation_equals_854_l2715_271589


namespace permutation_calculation_l2715_271583

-- Define the permutation function
def A (n : ℕ) (r : ℕ) : ℚ :=
  if r ≤ n then (Nat.factorial n) / (Nat.factorial (n - r)) else 0

-- State the theorem
theorem permutation_calculation :
  (4 * A 8 4 + 2 * A 8 5) / (A 8 6 - A 9 5) * Nat.factorial 0 = 2.4 := by
  sorry

end permutation_calculation_l2715_271583


namespace fraction_equals_d_minus_one_l2715_271552

theorem fraction_equals_d_minus_one (n d : ℕ) (h : d ∣ n) :
  ∃ i : ℕ, i < n ∧ (i : ℚ) / (n - i : ℚ) = d - 1 := by
  sorry

end fraction_equals_d_minus_one_l2715_271552


namespace arithmetic_mean_problem_l2715_271549

theorem arithmetic_mean_problem (x : ℝ) : 
  (12 + 18 + 24 + 36 + 6 + x) / 6 = 16 → x = 0 := by
  sorry

end arithmetic_mean_problem_l2715_271549


namespace first_reading_takes_15_days_l2715_271557

/-- The number of pages in the book -/
def total_pages : ℕ := 480

/-- The additional pages read per day in the second reading -/
def additional_pages_per_day : ℕ := 16

/-- The number of days saved in the second reading -/
def days_saved : ℕ := 5

/-- The number of days taken for the first reading -/
def first_reading_days : ℕ := 15

/-- The number of pages read per day in the first reading -/
def pages_per_day_first : ℕ := total_pages / first_reading_days

/-- The number of pages read per day in the second reading -/
def pages_per_day_second : ℕ := pages_per_day_first + additional_pages_per_day

/-- Theorem stating that the given conditions result in 15 days for the first reading -/
theorem first_reading_takes_15_days :
  (total_pages / pages_per_day_first = first_reading_days) ∧
  (total_pages / pages_per_day_second = first_reading_days - days_saved) :=
by sorry

end first_reading_takes_15_days_l2715_271557


namespace cos_180_degrees_l2715_271515

theorem cos_180_degrees : Real.cos (π) = -1 := by
  sorry

end cos_180_degrees_l2715_271515


namespace quadratic_tangent_line_l2715_271598

/-- Given a quadratic function f(x) = x^2 + ax + b, prove that if its tangent line
    at (0, b) has the equation x - y + 1 = 0, then a = 1 and b = 1. -/
theorem quadratic_tangent_line (a b : ℝ) :
  let f : ℝ → ℝ := λ x ↦ x^2 + a*x + b
  let f' : ℝ → ℝ := λ x ↦ 2*x + a
  (∀ x y, y = f x → x - y + 1 = 0 → x = 0) →
  f' 0 = 1 →
  a = 1 ∧ b = 1 := by
sorry

end quadratic_tangent_line_l2715_271598


namespace cos_negative_twentythree_fourths_pi_l2715_271585

theorem cos_negative_twentythree_fourths_pi :
  Real.cos (-23 / 4 * Real.pi) = Real.sqrt 2 / 2 := by
  sorry

end cos_negative_twentythree_fourths_pi_l2715_271585


namespace mike_ride_distance_l2715_271527

/-- Represents the taxi fare structure and ride details -/
structure TaxiRide where
  base_fare : ℝ
  per_mile_rate : ℝ
  additional_fee : ℝ
  distance : ℝ

/-- Calculates the total fare for a taxi ride -/
def total_fare (ride : TaxiRide) : ℝ :=
  ride.base_fare + ride.per_mile_rate * ride.distance + ride.additional_fee

/-- Proves that Mike's ride was 42 miles long given the conditions -/
theorem mike_ride_distance (mike annie : TaxiRide) 
    (h1 : mike.base_fare = 2.5)
    (h2 : mike.per_mile_rate = 0.25)
    (h3 : mike.additional_fee = 0)
    (h4 : annie.base_fare = 2.5)
    (h5 : annie.per_mile_rate = 0.25)
    (h6 : annie.additional_fee = 5)
    (h7 : annie.distance = 22)
    (h8 : total_fare mike = total_fare annie) : mike.distance = 42 := by
  sorry

#check mike_ride_distance

end mike_ride_distance_l2715_271527


namespace smallest_inverse_domain_l2715_271528

def g (x : ℝ) : ℝ := -3 * (x - 1)^2 + 4

theorem smallest_inverse_domain (c : ℝ) : 
  (∀ x y, x ≥ c → y ≥ c → g x = g y → x = y) ↔ c ≥ 1 :=
sorry

end smallest_inverse_domain_l2715_271528


namespace quadratic_equation_equivalence_l2715_271561

theorem quadratic_equation_equivalence :
  ∀ (x : ℝ), 3 * x^2 + 1 = 6 * x ↔ 3 * x^2 - 6 * x + 1 = 0 :=
by sorry

end quadratic_equation_equivalence_l2715_271561


namespace arithmetic_sequence_count_l2715_271580

theorem arithmetic_sequence_count :
  ∀ (a₁ last d : ℕ) (n : ℕ),
    a₁ = 1 →
    last = 2025 →
    d = 4 →
    last = a₁ + d * (n - 1) →
    n = 507 :=
by
  sorry

end arithmetic_sequence_count_l2715_271580


namespace amanda_drawer_pulls_l2715_271567

/-- Proves that the number of drawer pulls Amanda is replacing is 8 --/
theorem amanda_drawer_pulls (num_cabinet_knobs : ℕ) (cost_cabinet_knob : ℚ) 
  (cost_drawer_pull : ℚ) (total_cost : ℚ) 
  (h1 : num_cabinet_knobs = 18)
  (h2 : cost_cabinet_knob = 5/2)
  (h3 : cost_drawer_pull = 4)
  (h4 : total_cost = 77) :
  (total_cost - num_cabinet_knobs * cost_cabinet_knob) / cost_drawer_pull = 8 := by
  sorry

#eval (77 : ℚ) - 18 * (5/2 : ℚ)

end amanda_drawer_pulls_l2715_271567


namespace six_sufficient_not_necessary_l2715_271548

-- Define the binomial expansion term
def binomialTerm (n : ℕ) (r : ℕ) : ℚ → ℚ := λ x => x^(2*n - 3*r)

-- Define the condition for a constant term
def hasConstantTerm (n : ℕ) : Prop := ∃ r : ℕ, 2*n = 3*r

-- Theorem stating that n=6 is sufficient but not necessary
theorem six_sufficient_not_necessary :
  (hasConstantTerm 6) ∧ (∃ m : ℕ, m ≠ 6 ∧ hasConstantTerm m) :=
sorry

end six_sufficient_not_necessary_l2715_271548


namespace exists_m_iff_n_power_of_two_l2715_271503

theorem exists_m_iff_n_power_of_two (n : ℕ+) :
  (∃ m : ℤ, (2^n.val - 1) ∣ (m^2 + 9)) ↔ ∃ k : ℕ, n.val = 2^k :=
sorry

end exists_m_iff_n_power_of_two_l2715_271503


namespace intersection_range_l2715_271526

-- Define the curve
def curve (x y : ℝ) : Prop := x^2 + y^2 - 6*x = 0 ∧ y > 0

-- Define the line
def line (k x y : ℝ) : Prop := y = k*(x + 2)

-- Define the intersection condition
def intersects (k : ℝ) : Prop :=
  ∃ x y, curve x y ∧ line k x y

-- State the theorem
theorem intersection_range :
  ∀ k, intersects k ↔ k > 0 ∧ k ≤ 3/4 :=
sorry

end intersection_range_l2715_271526


namespace concentric_circles_area_ratio_l2715_271553

theorem concentric_circles_area_ratio :
  let d₁ : ℝ := 2  -- diameter of smallest circle
  let d₂ : ℝ := 4  -- diameter of middle circle
  let d₃ : ℝ := 6  -- diameter of largest circle
  let r₁ : ℝ := d₁ / 2  -- radius of smallest circle
  let r₂ : ℝ := d₂ / 2  -- radius of middle circle
  let r₃ : ℝ := d₃ / 2  -- radius of largest circle
  let A₁ : ℝ := π * r₁^2  -- area of smallest circle
  let A₂ : ℝ := π * r₂^2  -- area of middle circle
  let A₃ : ℝ := π * r₃^2  -- area of largest circle
  (A₃ - A₂) / A₁ = 5 :=
by
  sorry

end concentric_circles_area_ratio_l2715_271553


namespace line_slope_45_degrees_l2715_271594

/-- Given a line passing through points P(-2, m) and Q(m, 4) with a slope angle of 45°, the value of m is 1. -/
theorem line_slope_45_degrees (m : ℝ) : 
  (∃ (P Q : ℝ × ℝ), 
    P = (-2, m) ∧ 
    Q = (m, 4) ∧ 
    (Q.2 - P.2) / (Q.1 - P.1) = Real.tan (π / 4)) → 
  m = 1 := by
sorry

end line_slope_45_degrees_l2715_271594


namespace fraction_equality_sum_l2715_271533

theorem fraction_equality_sum (P Q : ℚ) : 
  (5 : ℚ) / 7 = P / 63 ∧ (5 : ℚ) / 7 = 140 / Q → P + Q = 241 := by
  sorry

end fraction_equality_sum_l2715_271533


namespace right_triangle_from_sine_condition_l2715_271555

theorem right_triangle_from_sine_condition (A B C : ℝ) :
  (0 < A) → (A < π) →
  (0 < B) → (B < π) →
  (0 < C) → (C < π) →
  A + B + C = π →
  (Real.sin A + Real.sin B) * (Real.sin A - Real.sin B) = (Real.sin C)^2 →
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 = c^2 :=
by sorry

end right_triangle_from_sine_condition_l2715_271555


namespace constant_function_theorem_l2715_271572

theorem constant_function_theorem (f : ℝ → ℝ) (hf : Differentiable ℝ f) :
  (∀ x, f x * (deriv f x) = 0) → ∃ C, ∀ x, f x = C := by
  sorry

end constant_function_theorem_l2715_271572


namespace range_of_f_l2715_271569

-- Define the function f
def f (x : ℝ) : ℝ := x - x^3

-- State the theorem
theorem range_of_f :
  ∃ (a b : ℝ), a = -6 ∧ b = 2 * Real.sqrt 3 / 9 ∧
  (∀ y, (∃ x ∈ Set.Icc 0 2, f x = y) ↔ a ≤ y ∧ y ≤ b) :=
by sorry

end range_of_f_l2715_271569


namespace expression1_equals_4_expression2_equals_neg6_l2715_271514

-- Define the expressions
def expression1 : ℚ := (-36) * (1/3 - 1/2) + 16 / ((-2)^3)
def expression2 : ℚ := (-5 + 2) * (1/3) + 5^2 / (-5)

-- Theorem statements
theorem expression1_equals_4 : expression1 = 4 := by sorry

theorem expression2_equals_neg6 : expression2 = -6 := by sorry

end expression1_equals_4_expression2_equals_neg6_l2715_271514


namespace seeds_in_second_plot_is_200_l2715_271543

/-- The number of seeds planted in the second plot -/
def seeds_in_second_plot : ℕ := 200

/-- The number of seeds planted in the first plot -/
def seeds_in_first_plot : ℕ := 500

/-- The germination rate of seeds in the first plot -/
def germination_rate_first : ℚ := 30 / 100

/-- The germination rate of seeds in the second plot -/
def germination_rate_second : ℚ := 50 / 100

/-- The total germination rate of all seeds -/
def total_germination_rate : ℚ := 35714285714285715 / 100000000000000000

theorem seeds_in_second_plot_is_200 : 
  (germination_rate_first * seeds_in_first_plot + 
   germination_rate_second * seeds_in_second_plot) / 
  (seeds_in_first_plot + seeds_in_second_plot) = total_germination_rate :=
sorry

end seeds_in_second_plot_is_200_l2715_271543


namespace triangle_side_length_range_l2715_271509

theorem triangle_side_length_range : ∃ (min max : ℤ),
  (∀ x : ℤ, (x + 8 > 10 ∧ x + 10 > 8 ∧ 8 + 10 > x) → min ≤ x ∧ x ≤ max) ∧
  min = 3 ∧ max = 17 ∧ max - min = 14 := by
  sorry

end triangle_side_length_range_l2715_271509


namespace bertha_family_without_daughters_l2715_271565

/-- Represents a family tree starting from Bertha -/
structure BerthaFamily where
  daughters : Nat
  daughters_with_children : Nat
  total_descendants : Nat

/-- The conditions of Bertha's family -/
def bertha_family : BerthaFamily := {
  daughters := 6,
  daughters_with_children := 4,
  total_descendants := 30
}

/-- Theorem: The number of Bertha's daughters and granddaughters who have no daughters is 26 -/
theorem bertha_family_without_daughters : 
  (bertha_family.total_descendants - bertha_family.daughters_with_children * bertha_family.daughters) + 
  (bertha_family.daughters - bertha_family.daughters_with_children) = 26 := by
  sorry

#check bertha_family_without_daughters

end bertha_family_without_daughters_l2715_271565


namespace sphere_volume_radius_3_l2715_271538

/-- The volume of a sphere with radius 3 is 36π. -/
theorem sphere_volume_radius_3 :
  let r : ℝ := 3
  let volume := (4 / 3) * Real.pi * r ^ 3
  volume = 36 * Real.pi := by sorry

end sphere_volume_radius_3_l2715_271538


namespace division_remainder_l2715_271512

theorem division_remainder (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (h1 : dividend = 3086) (h2 : divisor = 85) (h3 : quotient = 36) :
  dividend - divisor * quotient = 26 := by
  sorry

end division_remainder_l2715_271512


namespace cubic_equation_solutions_l2715_271519

theorem cubic_equation_solutions :
  let f (x : ℝ) := (10 * x - 1) ^ (1/3) + (20 * x + 1) ^ (1/3) - 3 * (5 * x) ^ (1/3)
  ∀ x : ℝ, f x = 0 ↔ x = 0 ∨ x = 1/10 ∨ x = -45/973 := by
  sorry

end cubic_equation_solutions_l2715_271519


namespace yahs_to_bahs_conversion_l2715_271581

/-- Represents the number of bahs equivalent to 36 rahs -/
def bahs_per_36_rahs : ℕ := 24

/-- Represents the number of rahs equivalent to 18 yahs -/
def rahs_per_18_yahs : ℕ := 12

/-- Represents the number of yahs we want to convert to bahs -/
def yahs_to_convert : ℕ := 1500

/-- Theorem stating the equivalence between 1500 yahs and 667 bahs -/
theorem yahs_to_bahs_conversion :
  ∃ (bahs : ℕ), bahs = 667 ∧
  (bahs * bahs_per_36_rahs * rahs_per_18_yahs : ℚ) / 36 / 18 = yahs_to_convert / 1 :=
sorry

end yahs_to_bahs_conversion_l2715_271581


namespace complex_magnitude_l2715_271564

theorem complex_magnitude (z : ℂ) (i : ℂ) (h1 : i * i = -1) (h2 : (1 - z) * i = 2) : Complex.abs z = Real.sqrt 5 := by
  sorry

end complex_magnitude_l2715_271564


namespace puppy_weight_l2715_271535

theorem puppy_weight (puppy smaller_cat larger_cat bird : ℝ) 
  (total_weight : puppy + smaller_cat + larger_cat + bird = 34)
  (larger_cat_weight : puppy + larger_cat = 3 * bird)
  (smaller_cat_weight : puppy + smaller_cat = 2 * bird) :
  puppy = 17 := by
  sorry

end puppy_weight_l2715_271535


namespace rectangle_area_l2715_271588

theorem rectangle_area (a b : ℝ) (h1 : a = 10) (h2 : 2*a + 2*b = 40) : a * b = 100 := by
  sorry

end rectangle_area_l2715_271588


namespace equal_division_of_sweets_and_candies_l2715_271506

theorem equal_division_of_sweets_and_candies :
  let num_sweets : ℕ := 72
  let num_candies : ℕ := 56
  let num_people : ℕ := 4
  let sweets_per_person : ℕ := num_sweets / num_people
  let candies_per_person : ℕ := num_candies / num_people
  let total_per_person : ℕ := sweets_per_person + candies_per_person
  total_per_person = 32 := by
  sorry

end equal_division_of_sweets_and_candies_l2715_271506


namespace company_demographics_l2715_271550

theorem company_demographics (total : ℕ) (total_pos : 0 < total) :
  let men_percent : ℚ := 48 / 100
  let union_percent : ℚ := 60 / 100
  let union_men_percent : ℚ := 70 / 100
  let men := (men_percent * total).floor
  let union := (union_percent * total).floor
  let union_men := (union_men_percent * union).floor
  let non_union := total - union
  let non_union_men := men - union_men
  let non_union_women := non_union - non_union_men
  (non_union_women : ℚ) / non_union = 85 / 100 :=
by sorry

end company_demographics_l2715_271550


namespace L_quotient_property_l2715_271563

/-- L(a,b) is defined as the exponent c such that a^c = b, for positive numbers a and b -/
noncomputable def L (a b : ℝ) : ℝ :=
  Real.log b / Real.log a

/-- Theorem: For positive real numbers a, m, and n, L(a, m/n) = L(a,m) - L(a,n) -/
theorem L_quotient_property (a m n : ℝ) (ha : 0 < a) (hm : 0 < m) (hn : 0 < n) :
  L a (m/n) = L a m - L a n := by
  sorry

end L_quotient_property_l2715_271563


namespace weight_selection_theorem_l2715_271536

theorem weight_selection_theorem (N : ℕ) :
  (∃ (S : Finset ℕ) (k : ℕ), 
    1 < k ∧ 
    k ≤ N ∧
    (∀ i ∈ S, 1 ≤ i ∧ i ≤ N) ∧
    S.card = k ∧
    (S.sum id) * (N - k + 1) = (N * (N + 1)) / 2) ↔ 
  (∃ m : ℕ, N + 1 = m^2) :=
by sorry

end weight_selection_theorem_l2715_271536


namespace percentage_increase_decrease_l2715_271577

theorem percentage_increase_decrease (p q M : ℝ) 
  (hp : p > 0) (hq : q > 0) (hq_bound : q < 200) (hM : M > 0) :
  M * (1 + p / 100) * (1 - q / 100) > M ↔ p > 100 * q / (100 - q) := by
  sorry

end percentage_increase_decrease_l2715_271577


namespace jenny_travel_distance_l2715_271513

/-- The distance from Jenny's home to her friend's place in miles -/
def total_distance : ℝ := 155

/-- Jenny's initial speed in miles per hour -/
def initial_speed : ℝ := 45

/-- Jenny's increased speed in miles per hour -/
def increased_speed : ℝ := 65

/-- The time Jenny stops at the store in hours -/
def stop_time : ℝ := 0.25

/-- The total travel time in hours -/
def total_time : ℝ := 3.4375

theorem jenny_travel_distance :
  (initial_speed * (total_time + 1) = total_distance) ∧
  (total_distance - initial_speed = increased_speed * (total_time - stop_time - 1)) ∧
  (total_distance = initial_speed * (total_time + 1)) :=
sorry

end jenny_travel_distance_l2715_271513


namespace colonization_combinations_l2715_271511

def total_planets : ℕ := 15
def earth_like_planets : ℕ := 6
def mars_like_planets : ℕ := 9
def total_units : ℕ := 16

def valid_combination (a b : ℕ) : Prop :=
  a ≤ earth_like_planets ∧ 
  b ≤ mars_like_planets ∧ 
  2 * a + b = total_units

def combinations_count : ℕ := 
  (Nat.choose earth_like_planets 6 * Nat.choose mars_like_planets 4) +
  (Nat.choose earth_like_planets 5 * Nat.choose mars_like_planets 6) +
  (Nat.choose earth_like_planets 4 * Nat.choose mars_like_planets 8)

theorem colonization_combinations : 
  combinations_count = 765 :=
sorry

end colonization_combinations_l2715_271511


namespace max_exterior_sum_is_34_l2715_271529

/-- Represents a rectangular prism with a pyramid added to one face -/
structure PrismWithPyramid where
  prism_faces : Nat
  prism_edges : Nat
  prism_vertices : Nat
  pyramid_new_faces : Nat
  pyramid_new_edges : Nat
  pyramid_new_vertices : Nat

/-- Calculates the total number of exterior elements (faces, edges, vertices) -/
def totalExteriorElements (shape : PrismWithPyramid) : Nat :=
  shape.prism_faces - 1 + shape.pyramid_new_faces +
  shape.prism_edges + shape.pyramid_new_edges +
  shape.prism_vertices + shape.pyramid_new_vertices

/-- The maximum sum of exterior faces, vertices, and edges -/
def maxExteriorSum : Nat := 34

/-- Theorem stating that the maximum sum of exterior elements is 34 -/
theorem max_exterior_sum_is_34 :
  ∀ shape : PrismWithPyramid,
    shape.prism_faces = 6 ∧
    shape.prism_edges = 12 ∧
    shape.prism_vertices = 8 ∧
    shape.pyramid_new_faces ≤ 4 ∧
    shape.pyramid_new_edges ≤ 4 ∧
    shape.pyramid_new_vertices = 1 →
    totalExteriorElements shape ≤ maxExteriorSum :=
by
  sorry


end max_exterior_sum_is_34_l2715_271529


namespace problem_solution_l2715_271551

theorem problem_solution : (12 : ℝ) ^ 1 * 6 ^ 4 / 432 = 36 := by
  sorry

end problem_solution_l2715_271551

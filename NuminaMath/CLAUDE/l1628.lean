import Mathlib

namespace NUMINAMATH_CALUDE_optimal_allocation_l1628_162896

/-- Represents the allocation of workers to different parts -/
structure WorkerAllocation where
  partA : ℕ
  partB : ℕ
  partC : ℕ

/-- Checks if the given allocation satisfies the total worker constraint -/
def satisfiesTotalWorkers (allocation : WorkerAllocation) : Prop :=
  allocation.partA + allocation.partB + allocation.partC = 45

/-- Checks if the given allocation produces parts in the required ratio -/
def satisfiesProductionRatio (allocation : WorkerAllocation) : Prop :=
  30 * allocation.partA = 25 * allocation.partB * 3 / 5 ∧
  30 * allocation.partA = 20 * allocation.partC * 3 / 4

/-- The main theorem stating that the given allocation satisfies all constraints -/
theorem optimal_allocation :
  let allocation := WorkerAllocation.mk 9 18 18
  satisfiesTotalWorkers allocation ∧ satisfiesProductionRatio allocation :=
by sorry

end NUMINAMATH_CALUDE_optimal_allocation_l1628_162896


namespace NUMINAMATH_CALUDE_problem_solution_l1628_162898

theorem problem_solution (a b c : ℝ) (h1 : a < 0) (h2 : a < b) (h3 : b < 0) (h4 : 0 < c) :
  (a * b < b * c) ∧ (a * c < b * c) ∧ (a + b < b + c) ∧ (c / a < 1) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1628_162898


namespace NUMINAMATH_CALUDE_monic_polynomial_value_theorem_l1628_162883

theorem monic_polynomial_value_theorem (p : ℤ → ℤ) (a b c d : ℤ) :
  (∀ x, p x = p (x + 1) - p x) →  -- p is monic
  (∀ x, ∃ k, p x = k) →  -- p has integer coefficients
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →  -- distinct values
  p a = 5 ∧ p b = 5 ∧ p c = 5 ∧ p d = 5 →  -- p takes value 5 at four distinct integers
  ∀ x : ℤ, p x ≠ 8 :=
by
  sorry

#check monic_polynomial_value_theorem

end NUMINAMATH_CALUDE_monic_polynomial_value_theorem_l1628_162883


namespace NUMINAMATH_CALUDE_matrix_multiplication_and_scalar_l1628_162889

theorem matrix_multiplication_and_scalar : 
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![3, 1; 4, -2]
  let B : Matrix (Fin 2) (Fin 2) ℤ := !![5, -3; 2, 2]
  2 • (A * B) = !![34, -14; 32, -32] := by sorry

end NUMINAMATH_CALUDE_matrix_multiplication_and_scalar_l1628_162889


namespace NUMINAMATH_CALUDE_july_birth_percentage_l1628_162806

theorem july_birth_percentage (total_scientists : ℕ) (july_births : ℕ) : 
  total_scientists = 150 → july_births = 15 → 
  (july_births : ℚ) / (total_scientists : ℚ) * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_july_birth_percentage_l1628_162806


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l1628_162828

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Checks if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

/-- Checks if a point lies on a line -/
def pointOnLine (l : Line) (x y : ℝ) : Prop :=
  y = l.slope * x + l.intercept

/-- The given line 4x + 2y = 8 -/
def givenLine : Line :=
  { slope := -2, intercept := 4 }

/-- The line we need to prove -/
def parallelLine : Line :=
  { slope := -2, intercept := 1 }

theorem parallel_line_through_point :
  parallel parallelLine givenLine ∧
  pointOnLine parallelLine 0 1 :=
sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l1628_162828


namespace NUMINAMATH_CALUDE_battery_current_l1628_162829

/-- Given a battery with voltage 48V, prove that when the resistance R is 12Ω, 
    the current I is 4A, where I is related to R by the function I = 48 / R. -/
theorem battery_current (R : ℝ) (I : ℝ) : 
  (I = 48 / R) → (R = 12) → (I = 4) := by
  sorry

end NUMINAMATH_CALUDE_battery_current_l1628_162829


namespace NUMINAMATH_CALUDE_cylinder_radius_comparison_l1628_162810

theorem cylinder_radius_comparison (h : ℝ) (r : ℝ) (h_pos : h > 0) (r_pos : r > 0) : 
  let original_volume := (6/7) * π * r^2 * h
  let new_height := (7/10) * h
  let new_volume := original_volume
  let new_radius := Real.sqrt ((5/3) * new_volume / (π * new_height))
  (new_radius - r) / r = 3/7 := by
sorry

end NUMINAMATH_CALUDE_cylinder_radius_comparison_l1628_162810


namespace NUMINAMATH_CALUDE_batsman_average_l1628_162868

theorem batsman_average (total_innings : ℕ) (last_innings_score : ℕ) (average_increase : ℚ) : 
  total_innings = 25 →
  last_innings_score = 95 →
  average_increase = 5/2 →
  (∃ (previous_average : ℚ), 
    (previous_average * (total_innings - 1) + last_innings_score) / total_innings = previous_average + average_increase) →
  (∃ (final_average : ℚ), final_average = 35) := by
sorry

end NUMINAMATH_CALUDE_batsman_average_l1628_162868


namespace NUMINAMATH_CALUDE_wall_painting_problem_l1628_162803

theorem wall_painting_problem (heidi_rate peter_rate : ℚ) 
  (heidi_time peter_time painting_time : ℕ) :
  heidi_rate = 1 / 60 →
  peter_rate = 1 / 75 →
  heidi_time = 60 →
  peter_time = 75 →
  painting_time = 15 →
  (heidi_rate + peter_rate) * painting_time = 9 / 20 := by
  sorry

end NUMINAMATH_CALUDE_wall_painting_problem_l1628_162803


namespace NUMINAMATH_CALUDE_function_depends_on_one_arg_l1628_162846

/-- A function that depends only on one of its arguments -/
def DependsOnOneArg {α : Type*} {k : ℕ} (f : (Fin k → α) → α) : Prop :=
  ∃ i : Fin k, ∀ x y : Fin k → α, (x i = y i) → (f x = f y)

/-- The main theorem -/
theorem function_depends_on_one_arg
  {n : ℕ} (h_n : n ≥ 3) (k : ℕ) (f : (Fin k → Fin n) → Fin n)
  (h_f : ∀ x y : Fin k → Fin n, (∀ i, x i ≠ y i) → f x ≠ f y) :
  DependsOnOneArg f := by
  sorry

end NUMINAMATH_CALUDE_function_depends_on_one_arg_l1628_162846


namespace NUMINAMATH_CALUDE_angle_expression_value_l1628_162831

theorem angle_expression_value (θ : Real) 
  (h1 : π/2 < θ ∧ θ < π) -- θ is in the second quadrant
  (h2 : Real.tan (θ - π) = -1/2) :
  Real.sqrt ((1 + Real.cos θ) / (1 - Real.sin (π/2 - θ))) - 
  Real.sqrt ((1 - Real.cos θ) / (1 + Real.sin (θ - 3*π/2))) = -4 := by
  sorry

end NUMINAMATH_CALUDE_angle_expression_value_l1628_162831


namespace NUMINAMATH_CALUDE_intersection_singleton_iff_a_in_range_l1628_162850

def set_A (a : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = a * |p.1|}

def set_B (a : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1 + a}

theorem intersection_singleton_iff_a_in_range (a : ℝ) :
  (∃! p : ℝ × ℝ, p ∈ set_A a ∩ set_B a) ↔ a ∈ Set.Icc (-1 : ℝ) 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_singleton_iff_a_in_range_l1628_162850


namespace NUMINAMATH_CALUDE_largest_sum_and_simplification_l1628_162808

theorem largest_sum_and_simplification : 
  let sums := [2/5 + 1/6, 2/5 + 1/3, 2/5 + 1/7, 2/5 + 1/8, 2/5 + 1/9]
  (∀ x ∈ sums, x ≤ 2/5 + 1/3) ∧ (2/5 + 1/3 = 11/15) := by
  sorry

end NUMINAMATH_CALUDE_largest_sum_and_simplification_l1628_162808


namespace NUMINAMATH_CALUDE_percentage_problem_l1628_162879

theorem percentage_problem (x : ℝ) (p : ℝ) : 
  x = 230 → 
  p / 100 * x = 20 / 100 * 747.50 → 
  p = 65 := by
sorry

end NUMINAMATH_CALUDE_percentage_problem_l1628_162879


namespace NUMINAMATH_CALUDE_probability_not_jim_pictures_l1628_162800

/-- Given a set of pictures, calculate the probability of picking two pictures
    that are not among those bought by Jim. -/
theorem probability_not_jim_pictures
  (total_pictures : ℕ)
  (jim_bought : ℕ)
  (pick_count : ℕ)
  (h_total : total_pictures = 10)
  (h_jim : jim_bought = 3)
  (h_pick : pick_count = 2) :
  (pick_count.choose (total_pictures - jim_bought)) / (pick_count.choose total_pictures) = 7 / 15 := by
  sorry

end NUMINAMATH_CALUDE_probability_not_jim_pictures_l1628_162800


namespace NUMINAMATH_CALUDE_cats_eating_mice_l1628_162858

/-- If n cats eat n mice in n hours, then p cats eat (p^2 / n) mice in p hours -/
theorem cats_eating_mice (n p : ℕ) (h : n ≠ 0) : 
  (n : ℚ) * (n : ℚ) / (n : ℚ) = n → (p : ℚ) * (p : ℚ) / (n : ℚ) = p^2 / n := by
  sorry

end NUMINAMATH_CALUDE_cats_eating_mice_l1628_162858


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1628_162856

theorem sqrt_equation_solution : ∃ x : ℝ, x = 1225 / 36 ∧ Real.sqrt x + Real.sqrt (x + 4) = 12 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1628_162856


namespace NUMINAMATH_CALUDE_unique_solution_l1628_162834

theorem unique_solution : 
  ∃! x : ℝ, -1 < x ∧ x ≤ 2 ∧ 
  Real.sqrt (2 - x) + Real.sqrt (2 + 2*x) = 
  Real.sqrt ((x^4 + 1) / (x^2 + 1)) + (x + 3) / (x + 1) ∧ 
  x = 1 := by sorry

end NUMINAMATH_CALUDE_unique_solution_l1628_162834


namespace NUMINAMATH_CALUDE_oregano_basil_difference_l1628_162840

theorem oregano_basil_difference (basil : ℕ) (total : ℕ) (oregano : ℕ) :
  basil = 5 →
  total = 17 →
  oregano > 2 * basil →
  total = basil + oregano →
  oregano - 2 * basil = 2 := by
  sorry

end NUMINAMATH_CALUDE_oregano_basil_difference_l1628_162840


namespace NUMINAMATH_CALUDE_tom_fishing_probability_l1628_162843

-- Define the weather conditions
inductive Weather
  | Sunny
  | Rainy
  | Cloudy

-- Define the probability of Tom going fishing for each weather condition
def fishing_prob (w : Weather) : ℝ :=
  match w with
  | Weather.Sunny => 0.7
  | Weather.Rainy => 0.3
  | Weather.Cloudy => 0.5

-- Define the probability of each weather condition
def weather_prob (w : Weather) : ℝ :=
  match w with
  | Weather.Sunny => 0.3
  | Weather.Rainy => 0.5
  | Weather.Cloudy => 0.2

-- Theorem stating the probability of Tom going fishing
theorem tom_fishing_probability :
  (fishing_prob Weather.Sunny * weather_prob Weather.Sunny +
   fishing_prob Weather.Rainy * weather_prob Weather.Rainy +
   fishing_prob Weather.Cloudy * weather_prob Weather.Cloudy) = 0.46 := by
  sorry


end NUMINAMATH_CALUDE_tom_fishing_probability_l1628_162843


namespace NUMINAMATH_CALUDE_f_max_value_l1628_162805

/-- The function f(x) = -x^2 + 4x + a -/
def f (a : ℝ) (x : ℝ) : ℝ := -x^2 + 4*x + a

/-- Theorem: If f(x) has a minimum value of -2 on [0, 1], then its maximum value on [0, 1] is 1 -/
theorem f_max_value (a : ℝ) :
  (∃ x ∈ Set.Icc 0 1, ∀ y ∈ Set.Icc 0 1, f a x ≤ f a y) →
  (∃ x ∈ Set.Icc 0 1, f a x = -2) →
  (∃ x ∈ Set.Icc 0 1, ∀ y ∈ Set.Icc 0 1, f a y ≤ f a x) ∧
  (∃ x ∈ Set.Icc 0 1, f a x = 1) :=
by sorry

#check f_max_value

end NUMINAMATH_CALUDE_f_max_value_l1628_162805


namespace NUMINAMATH_CALUDE_union_M_N_l1628_162882

-- Define the universe set U
def U : Set ℝ := {x : ℝ | -3 ≤ x ∧ x < 2}

-- Define set M
def M : Set ℝ := {x : ℝ | -1 < x ∧ x < 1}

-- Define the complement of N in U
def complement_N_in_U : Set ℝ := {x : ℝ | 0 < x ∧ x < 2}

-- Define set N
def N : Set ℝ := U \ complement_N_in_U

-- Theorem to prove
theorem union_M_N : M ∪ N = {x : ℝ | -3 ≤ x ∧ x < 1} := by
  sorry

end NUMINAMATH_CALUDE_union_M_N_l1628_162882


namespace NUMINAMATH_CALUDE_exam_time_allocation_l1628_162878

theorem exam_time_allocation (total_time : ℕ) (total_questions : ℕ) (type_a_questions : ℕ) :
  total_time = 180 →
  total_questions = 200 →
  type_a_questions = 50 →
  let type_b_questions := total_questions - type_a_questions
  let time_ratio := 2
  let total_time_units := type_a_questions * time_ratio + type_b_questions
  let time_per_unit := total_time / total_time_units
  let time_for_type_a := type_a_questions * time_ratio * time_per_unit
  time_for_type_a = 72 :=
by
  sorry

#check exam_time_allocation

end NUMINAMATH_CALUDE_exam_time_allocation_l1628_162878


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_35_l1628_162823

theorem smallest_four_digit_divisible_by_35 :
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 35 ∣ n → 1015 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_35_l1628_162823


namespace NUMINAMATH_CALUDE_modulo_congruence_problem_l1628_162830

theorem modulo_congruence_problem : ∃! n : ℤ, 0 ≤ n ∧ n < 31 ∧ 49325 % 31 = n % 31 ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_modulo_congruence_problem_l1628_162830


namespace NUMINAMATH_CALUDE_quadratic_symmetry_and_point_l1628_162863

def f (x : ℝ) := (x - 2)^2 - 3

theorem quadratic_symmetry_and_point :
  (∀ x : ℝ, f (2 + x) = f (2 - x)) ∧ f 0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_symmetry_and_point_l1628_162863


namespace NUMINAMATH_CALUDE_min_sum_of_parallel_vectors_l1628_162801

-- Define the vectors
def a (x : ℝ) : Fin 2 → ℝ := ![1, x - 1]
def b (y : ℝ) : Fin 2 → ℝ := ![y, 2]

-- Theorem statement
theorem min_sum_of_parallel_vectors (x y : ℝ) 
  (h1 : x - 1 ≥ 0) 
  (h2 : y ≥ 0) 
  (h3 : ∃ (k : ℝ), k ≠ 0 ∧ a x = k • b y) : 
  x + y ≥ 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_of_parallel_vectors_l1628_162801


namespace NUMINAMATH_CALUDE_height_difference_proof_l1628_162813

/-- Proves that the height difference between Vlad and his sister is 104.14 cm -/
theorem height_difference_proof (vlad_height_m : ℝ) (sister_height_cm : ℝ) 
  (h1 : vlad_height_m = 1.905) (h2 : sister_height_cm = 86.36) : 
  vlad_height_m * 100 - sister_height_cm = 104.14 := by
  sorry

end NUMINAMATH_CALUDE_height_difference_proof_l1628_162813


namespace NUMINAMATH_CALUDE_perpendicular_lines_theorem_l1628_162899

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between planes
variable (perp_planes : Plane → Plane → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perp_line_plane : Line → Plane → Prop)

-- Define the parallel relation between a line and a plane
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between lines
variable (perp_lines : Line → Line → Prop)

-- Define the intersection operation between planes
variable (intersection : Plane → Plane → Line)

-- State the theorem
theorem perpendicular_lines_theorem 
  (α β : Plane) (a b l : Line) 
  (h1 : perp_planes α β)
  (h2 : intersection α β = l)
  (h3 : parallel_line_plane a α)
  (h4 : perp_line_plane b β) :
  perp_lines b l :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_theorem_l1628_162899


namespace NUMINAMATH_CALUDE_toys_sold_l1628_162842

theorem toys_sold (selling_price : ℕ) (cost_price : ℕ) (gain : ℕ) :
  selling_price = 27300 →
  gain = 3 * cost_price →
  cost_price = 1300 →
  selling_price = (selling_price - gain) / cost_price * cost_price + gain →
  (selling_price - gain) / cost_price = 18 :=
by sorry

end NUMINAMATH_CALUDE_toys_sold_l1628_162842


namespace NUMINAMATH_CALUDE_option_A_is_incorrect_l1628_162872

-- Define the set of angles whose terminal sides lie on y=x
def AnglesOnYEqualsX : Set ℝ := {β | ∃ n : ℤ, β = 45 + n * 180}

-- Define the set given in option A
def OptionASet : Set ℝ := {β | ∃ k : ℤ, β = 45 + k * 360 ∨ β = -45 + k * 360}

-- Theorem statement
theorem option_A_is_incorrect : OptionASet ≠ AnglesOnYEqualsX := by
  sorry

end NUMINAMATH_CALUDE_option_A_is_incorrect_l1628_162872


namespace NUMINAMATH_CALUDE_infinitely_many_pairs_exist_l1628_162812

/-- A function that checks if all digits in the decimal representation of a natural number are greater than or equal to 7. -/
def allDigitsAtLeastSeven (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d ≥ 7

/-- A function that generates a pair of natural numbers based on the input n. -/
noncomputable def f (n : ℕ) : ℕ × ℕ :=
  let a := (887 : ℕ).pow n
  let b := 10^(3*n) - 123
  (a, b)

/-- Theorem stating that there exist infinitely many pairs of integers satisfying the given conditions. -/
theorem infinitely_many_pairs_exist :
  ∀ n : ℕ, 
    let (a, b) := f n
    allDigitsAtLeastSeven a ∧
    allDigitsAtLeastSeven b ∧
    allDigitsAtLeastSeven (a * b) :=
by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_pairs_exist_l1628_162812


namespace NUMINAMATH_CALUDE_not_all_nonnegative_l1628_162827

theorem not_all_nonnegative (a b c d : ℝ) 
  (sum_ab : a + b = 1) 
  (sum_cd : c + d = 1) 
  (prod_sum : a * c + b * d > 1) : 
  ¬(0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d) := by
  sorry

end NUMINAMATH_CALUDE_not_all_nonnegative_l1628_162827


namespace NUMINAMATH_CALUDE_distance_between_cities_l1628_162817

theorem distance_between_cities (t : ℝ) : ∃ x : ℝ,
  x / 50 = t - 1 ∧ x / 35 = t + 2 → x = 350 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_cities_l1628_162817


namespace NUMINAMATH_CALUDE_odd_factors_of_420_l1628_162818

-- Define 420 as a natural number
def n : ℕ := 420

-- Define a function to count odd factors
def count_odd_factors (m : ℕ) : ℕ := sorry

-- Theorem statement
theorem odd_factors_of_420 : count_odd_factors n = 8 := by sorry

end NUMINAMATH_CALUDE_odd_factors_of_420_l1628_162818


namespace NUMINAMATH_CALUDE_point_movement_l1628_162841

def move_point (start : ℤ) (distance : ℤ) : ℤ := start + distance

theorem point_movement (A B : ℤ) :
  A = -3 →
  move_point A 4 = B →
  B = 1 := by sorry

end NUMINAMATH_CALUDE_point_movement_l1628_162841


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_problem_l1628_162833

theorem gcd_lcm_sum_problem : Nat.gcd 40 60 + 2 * Nat.lcm 20 15 = 140 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_problem_l1628_162833


namespace NUMINAMATH_CALUDE_geometric_sequence_third_term_l1628_162848

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ+ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ+, a (n + 1) = a n * r

theorem geometric_sequence_third_term
  (a : ℕ+ → ℝ)
  (h_geom : GeometricSequence a)
  (h_pos : ∀ n : ℕ+, a n > 0)
  (h_prod : a 2 * a 4 = 9) :
  a 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_third_term_l1628_162848


namespace NUMINAMATH_CALUDE_age_height_not_function_l1628_162853

-- Define a type for people
structure Person where
  age : ℕ
  height : ℝ

-- Define what it means for a relation to be a function
def is_function (R : α → β → Prop) : Prop :=
  ∀ a : α, ∃! b : β, R a b

-- State the theorem
theorem age_height_not_function :
  ¬ is_function (λ (p : Person) (h : ℝ) => p.height = h) :=
sorry

end NUMINAMATH_CALUDE_age_height_not_function_l1628_162853


namespace NUMINAMATH_CALUDE_consecutive_sum_product_l1628_162844

theorem consecutive_sum_product (start : ℕ) :
  (start + (start + 1) + (start + 2) + (start + 3) + (start + 4) + (start + 5) = 33) →
  (start * (start + 1) * (start + 2) * (start + 3) * (start + 4) * (start + 5) = 20160) :=
by
  sorry

end NUMINAMATH_CALUDE_consecutive_sum_product_l1628_162844


namespace NUMINAMATH_CALUDE_trip_time_proof_l1628_162877

-- Define the distances
def freeway_distance : ℝ := 60
def mountain_distance : ℝ := 20

-- Define the time spent on mountain pass
def mountain_time : ℝ := 40

-- Define the speed ratio
def speed_ratio : ℝ := 4

-- Define the total trip time
def total_trip_time : ℝ := 70

-- Theorem statement
theorem trip_time_proof :
  let mountain_speed := mountain_distance / mountain_time
  let freeway_speed := speed_ratio * mountain_speed
  let freeway_time := freeway_distance / freeway_speed
  mountain_time + freeway_time = total_trip_time := by sorry

end NUMINAMATH_CALUDE_trip_time_proof_l1628_162877


namespace NUMINAMATH_CALUDE_sum_of_fractions_l1628_162864

theorem sum_of_fractions : (1 : ℚ) / 4 + (3 : ℚ) / 9 = (7 : ℚ) / 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l1628_162864


namespace NUMINAMATH_CALUDE_order_of_operations_l1628_162849

theorem order_of_operations (a b c : ℕ) : a - b * c = a - (b * c) := by
  sorry

end NUMINAMATH_CALUDE_order_of_operations_l1628_162849


namespace NUMINAMATH_CALUDE_emily_holidays_l1628_162824

/-- The number of holidays Emily takes in a year -/
def holidays_per_year (days_off_per_month : ℕ) (months_in_year : ℕ) : ℕ :=
  days_off_per_month * months_in_year

/-- Theorem: Emily takes 24 holidays in a year -/
theorem emily_holidays :
  holidays_per_year 2 12 = 24 := by
  sorry

end NUMINAMATH_CALUDE_emily_holidays_l1628_162824


namespace NUMINAMATH_CALUDE_distance_between_points_l1628_162875

/-- The distance between two points A and B, given the travel time and average speed -/
theorem distance_between_points (time : ℝ) (speed : ℝ) (h1 : time = 4.5) (h2 : speed = 80) :
  time * speed = 360 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l1628_162875


namespace NUMINAMATH_CALUDE_wednesday_sales_l1628_162845

def initial_stock : ℕ := 1400
def monday_sales : ℕ := 62
def tuesday_sales : ℕ := 62
def thursday_sales : ℕ := 48
def friday_sales : ℕ := 40
def unsold_percentage : ℚ := 80.57142857142857 / 100

theorem wednesday_sales :
  let unsold := (initial_stock : ℚ) * unsold_percentage
  let other_days_sales := monday_sales + tuesday_sales + thursday_sales + friday_sales
  initial_stock - (unsold.floor + other_days_sales) = 60 := by
  sorry

end NUMINAMATH_CALUDE_wednesday_sales_l1628_162845


namespace NUMINAMATH_CALUDE_apps_files_difference_l1628_162861

/-- Represents the state of Dave's phone --/
structure PhoneState where
  apps : ℕ
  files : ℕ

/-- The initial state of Dave's phone --/
def initial_state : PhoneState := { apps := 15, files := 24 }

/-- The final state of Dave's phone --/
def final_state : PhoneState := { apps := 21, files := 4 }

/-- Theorem stating the difference between apps and files in the final state --/
theorem apps_files_difference :
  final_state.apps - final_state.files = 17 := by
  sorry

end NUMINAMATH_CALUDE_apps_files_difference_l1628_162861


namespace NUMINAMATH_CALUDE_trivia_team_score_l1628_162862

theorem trivia_team_score : 
  let total_members : ℕ := 30
  let absent_members : ℕ := 8
  let points_per_member : ℕ := 4
  let deduction_per_incorrect : ℕ := 2
  let total_incorrect : ℕ := 6
  let bonus_multiplier : ℚ := 3/2

  let present_members : ℕ := total_members - absent_members
  let initial_points : ℕ := present_members * points_per_member
  let total_deductions : ℕ := total_incorrect * deduction_per_incorrect
  let points_after_deductions : ℕ := initial_points - total_deductions
  let final_score : ℚ := (points_after_deductions : ℚ) * bonus_multiplier

  final_score = 114 := by sorry

end NUMINAMATH_CALUDE_trivia_team_score_l1628_162862


namespace NUMINAMATH_CALUDE_quotient_remainder_difference_l1628_162832

theorem quotient_remainder_difference (N : ℕ) : 
  N ≥ 75 → 
  N % 5 = 0 → 
  (∀ m : ℕ, m ≥ 75 ∧ m % 5 = 0 → m ≥ N) →
  (N / 5) - (N % 34) = 8 := by
  sorry

end NUMINAMATH_CALUDE_quotient_remainder_difference_l1628_162832


namespace NUMINAMATH_CALUDE_simple_interest_calculation_l1628_162869

/-- Calculate the total amount owed after one year with simple interest. -/
theorem simple_interest_calculation 
  (principal : ℝ) 
  (rate : ℝ) 
  (time : ℝ) 
  (h1 : principal = 75) 
  (h2 : rate = 0.07) 
  (h3 : time = 1) : 
  principal * (1 + rate * time) = 80.25 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_calculation_l1628_162869


namespace NUMINAMATH_CALUDE_tangent_product_l1628_162815

theorem tangent_product (A B : ℝ) (hA : A = 30 * π / 180) (hB : B = 60 * π / 180) :
  (1 + Real.tan A) * (1 + Real.tan B) = 2 + 4 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_product_l1628_162815


namespace NUMINAMATH_CALUDE_smallest_n_less_than_one_hundredth_l1628_162891

/-- The probability of stopping after drawing exactly n marbles -/
def Q (n : ℕ+) : ℚ := 1 / (n * (n + 1))

/-- The number of boxes -/
def num_boxes : ℕ := 100

theorem smallest_n_less_than_one_hundredth :
  (∀ k : ℕ+, k < 10 → Q k ≥ 1/100) ∧
  (Q 10 < 1/100) ∧
  (∀ n : ℕ+, n ≤ num_boxes → Q n < 1/100 → n ≥ 10) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_less_than_one_hundredth_l1628_162891


namespace NUMINAMATH_CALUDE_club_size_after_five_years_l1628_162814

/-- Calculates the number of people in the club after a given number of years -/
def club_size (initial_size : ℕ) (years : ℕ) : ℕ :=
  match years with
  | 0 => initial_size
  | n + 1 => 4 * (club_size initial_size n - 7) + 7

theorem club_size_after_five_years :
  club_size 21 5 = 14343 := by
  sorry

#eval club_size 21 5

end NUMINAMATH_CALUDE_club_size_after_five_years_l1628_162814


namespace NUMINAMATH_CALUDE_target_breaking_orders_l1628_162894

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def multinomial (n : ℕ) (ks : List ℕ) : ℕ :=
  factorial n / (ks.map factorial).prod

theorem target_breaking_orders : 
  let n : ℕ := 9
  let ks : List ℕ := [4, 3, 2]
  multinomial n ks = 1260 := by sorry

end NUMINAMATH_CALUDE_target_breaking_orders_l1628_162894


namespace NUMINAMATH_CALUDE_quadratic_form_only_trivial_solution_l1628_162826

theorem quadratic_form_only_trivial_solution (a b c d : ℤ) :
  a^2 + 5*b^2 - 2*c^2 - 2*c*d - 3*d^2 = 0 → a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_only_trivial_solution_l1628_162826


namespace NUMINAMATH_CALUDE_derivative_of_exp_ax_l1628_162871

theorem derivative_of_exp_ax (a : ℝ) (x : ℝ) :
  deriv (fun x => Real.exp (a * x)) x = a * Real.exp (a * x) := by
  sorry

end NUMINAMATH_CALUDE_derivative_of_exp_ax_l1628_162871


namespace NUMINAMATH_CALUDE_new_car_distance_l1628_162857

theorem new_car_distance (old_car_speed : ℝ) (old_car_distance : ℝ) (speed_increase : ℝ) :
  old_car_distance = 150 →
  speed_increase = 0.3 →
  old_car_speed * (1 + speed_increase) * (old_car_distance / old_car_speed) = 195 :=
by sorry

end NUMINAMATH_CALUDE_new_car_distance_l1628_162857


namespace NUMINAMATH_CALUDE_shopping_remainder_l1628_162867

def initial_amount : ℕ := 109
def shirt_cost : ℕ := 11
def num_shirts : ℕ := 2
def pants_cost : ℕ := 13

theorem shopping_remainder :
  initial_amount - (shirt_cost * num_shirts + pants_cost) = 74 := by
  sorry

end NUMINAMATH_CALUDE_shopping_remainder_l1628_162867


namespace NUMINAMATH_CALUDE_function_composition_ratio_l1628_162807

def f (x : ℝ) : ℝ := 3 * x + 2

def g (x : ℝ) : ℝ := 2 * x - 3

theorem function_composition_ratio :
  f (g (f 2)) / g (f (g 2)) = 41 / 7 := by
  sorry

end NUMINAMATH_CALUDE_function_composition_ratio_l1628_162807


namespace NUMINAMATH_CALUDE_new_average_weight_l1628_162887

def original_team_size : ℕ := 7
def original_average_weight : ℝ := 121
def new_player1_weight : ℝ := 110
def new_player2_weight : ℝ := 60

theorem new_average_weight :
  let total_original_weight : ℝ := original_team_size * original_average_weight
  let new_total_weight : ℝ := total_original_weight + new_player1_weight + new_player2_weight
  let new_team_size : ℕ := original_team_size + 2
  (new_total_weight / new_team_size : ℝ) = 113 := by sorry

end NUMINAMATH_CALUDE_new_average_weight_l1628_162887


namespace NUMINAMATH_CALUDE_spermatogenesis_experiment_verification_l1628_162855

-- Define the available materials and tools
inductive Material
| MouseLiver
| Testes
| Kidneys

inductive Stain
| SudanIII
| AceticOrcein
| JanusGreen

inductive Tool
| DissociationFixative

-- Define the experiment steps
structure ExperimentSteps where
  material : Material
  fixative : Tool
  stain : Stain

-- Define the experiment result
structure ExperimentResult where
  cellTypesObserved : Nat

-- Define the correct experiment setup and result
def correctExperiment : ExperimentSteps := {
  material := Material.Testes,
  fixative := Tool.DissociationFixative,
  stain := Stain.AceticOrcein
}

def correctResult : ExperimentResult := {
  cellTypesObserved := 3
}

-- Theorem statement
theorem spermatogenesis_experiment_verification :
  ∀ (setup : ExperimentSteps) (result : ExperimentResult),
  setup = correctExperiment ∧ result = correctResult →
  setup.material = Material.Testes ∧
  setup.fixative = Tool.DissociationFixative ∧
  setup.stain = Stain.AceticOrcein ∧
  result.cellTypesObserved = 3 :=
by sorry

end NUMINAMATH_CALUDE_spermatogenesis_experiment_verification_l1628_162855


namespace NUMINAMATH_CALUDE_square_minus_one_l1628_162816

theorem square_minus_one (x : ℤ) (h : x^2 = 1521) : (x + 1) * (x - 1) = 1520 := by
  sorry

end NUMINAMATH_CALUDE_square_minus_one_l1628_162816


namespace NUMINAMATH_CALUDE_circle_radius_increase_l1628_162804

/-- If a circle's radius r is increased by n, and its new area is twice the original area,
    then r = n(√2 + 1) -/
theorem circle_radius_increase (n : ℝ) (h : n > 0) :
  ∃ (r : ℝ), r > 0 ∧ π * (r + n)^2 = 2 * π * r^2 → r = n * (Real.sqrt 2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_increase_l1628_162804


namespace NUMINAMATH_CALUDE_congruence_solution_l1628_162836

theorem congruence_solution (n : ℤ) : 
  4 ≤ n ∧ n ≤ 10 ∧ n ≡ 11783 [ZMOD 7] → n = 5 := by sorry

end NUMINAMATH_CALUDE_congruence_solution_l1628_162836


namespace NUMINAMATH_CALUDE_rotation_sum_65_l1628_162838

/-- Triangle in 2D space defined by three points -/
structure Triangle where
  x : ℝ × ℝ
  y : ℝ × ℝ
  z : ℝ × ℝ

/-- Rotation in 2D space defined by an angle and a center point -/
structure Rotation where
  angle : ℝ
  center : ℝ × ℝ

/-- Check if two triangles are congruent under rotation -/
def isCongruentUnderRotation (t1 t2 : Triangle) (r : Rotation) : Prop :=
  sorry

theorem rotation_sum_65 (xyz x'y'z' : Triangle) (r : Rotation) :
  xyz.x = (0, 0) →
  xyz.y = (0, 15) →
  xyz.z = (20, 0) →
  x'y'z'.x = (30, 10) →
  x'y'z'.y = (40, 10) →
  x'y'z'.z = (30, 0) →
  isCongruentUnderRotation xyz x'y'z' r →
  r.angle ≤ r'.angle → isCongruentUnderRotation xyz x'y'z' r' →
  r.angle + r.center.1 + r.center.2 = 65 := by
  sorry

end NUMINAMATH_CALUDE_rotation_sum_65_l1628_162838


namespace NUMINAMATH_CALUDE_divisibility_by_nine_l1628_162888

theorem divisibility_by_nine (n : ℕ) (h : 900 ≤ n ∧ n ≤ 999) : 
  (n % 9 = 0) ↔ ((n / 100 + (n / 10) % 10 + n % 10) % 9 = 0) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_nine_l1628_162888


namespace NUMINAMATH_CALUDE_circle_center_not_constructible_with_straightedge_l1628_162847

-- Define a circle on a plane
def Circle : Type := sorry

-- Define a straightedge
def Straightedge : Type := sorry

-- Define a point on a plane
def Point : Type := sorry

-- Define the concept of constructing a point using a straightedge
def constructible (p : Point) (s : Straightedge) : Prop := sorry

-- Define the center of a circle
def center (c : Circle) : Point := sorry

-- Theorem statement
theorem circle_center_not_constructible_with_straightedge (c : Circle) (s : Straightedge) :
  ¬(constructible (center c) s) := by sorry

end NUMINAMATH_CALUDE_circle_center_not_constructible_with_straightedge_l1628_162847


namespace NUMINAMATH_CALUDE_number_added_before_division_l1628_162822

theorem number_added_before_division (x n : ℤ) : 
  (∃ k : ℤ, x = 82 * k + 5) →
  (∃ m : ℤ, x + n = 41 * m + 22) →
  n = 5 := by
sorry

end NUMINAMATH_CALUDE_number_added_before_division_l1628_162822


namespace NUMINAMATH_CALUDE_y_is_75_percent_of_x_l1628_162892

-- Define variables
variable (x y z p : ℝ)

-- Define the theorem
theorem y_is_75_percent_of_x
  (h1 : 0.45 * z = 0.9 * y)
  (h2 : z = 1.5 * x)
  (h3 : y = p * x)
  : y = 0.75 * x :=
by sorry

end NUMINAMATH_CALUDE_y_is_75_percent_of_x_l1628_162892


namespace NUMINAMATH_CALUDE_charlie_has_largest_answer_l1628_162860

def starting_number : ℕ := 15

def alice_operation (n : ℕ) : ℕ := ((n - 2)^2 + 3)

def bob_operation (n : ℕ) : ℕ := (n^2 - 2 + 3)

def charlie_operation (n : ℕ) : ℕ := ((n - 2 + 3)^2)

theorem charlie_has_largest_answer :
  charlie_operation starting_number > alice_operation starting_number ∧
  charlie_operation starting_number > bob_operation starting_number := by
  sorry

end NUMINAMATH_CALUDE_charlie_has_largest_answer_l1628_162860


namespace NUMINAMATH_CALUDE_rectangle_to_square_l1628_162876

/-- Represents a rectangle with integer dimensions -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- Represents a square with integer side length -/
structure Square where
  side : ℕ

/-- Represents the area of a shape -/
def area : Rectangle → ℕ
  | ⟨l, w⟩ => l * w

/-- Theorem stating that a 9x4 rectangle can be cut and rearranged into a 6x6 square -/
theorem rectangle_to_square : 
  ∃ (r : Rectangle) (s : Square), 
    r.length = 9 ∧ 
    r.width = 4 ∧ 
    s.side = 6 ∧ 
    area r = s.side * s.side := by
  sorry

end NUMINAMATH_CALUDE_rectangle_to_square_l1628_162876


namespace NUMINAMATH_CALUDE_mountain_distance_l1628_162839

/-- Represents the mountain climbing scenario -/
structure MountainClimb where
  /-- Distance from bottom to top of the mountain in meters -/
  total_distance : ℝ
  /-- A's ascending speed in meters per hour -/
  speed_a_up : ℝ
  /-- B's ascending speed in meters per hour -/
  speed_b_up : ℝ
  /-- Distance from top where A and B meet in meters -/
  meeting_point : ℝ
  /-- Assumption that descending speed is 3 times ascending speed -/
  descent_speed_multiplier : ℝ
  /-- Assumption that A reaches bottom when B is halfway down -/
  b_halfway_when_a_bottom : Bool

/-- Main theorem: The distance from bottom to top is 1550 meters -/
theorem mountain_distance (climb : MountainClimb) 
  (h1 : climb.meeting_point = 150)
  (h2 : climb.descent_speed_multiplier = 3)
  (h3 : climb.b_halfway_when_a_bottom = true) :
  climb.total_distance = 1550 := by
  sorry

end NUMINAMATH_CALUDE_mountain_distance_l1628_162839


namespace NUMINAMATH_CALUDE_wire_cutting_l1628_162893

theorem wire_cutting (total_length : ℝ) (used_parts : ℕ) (unused_length : ℝ) (n : ℕ) :
  total_length = 50 →
  used_parts = 3 →
  unused_length = 20 →
  total_length = n * (total_length - unused_length) / used_parts →
  n = 5 := by
  sorry

end NUMINAMATH_CALUDE_wire_cutting_l1628_162893


namespace NUMINAMATH_CALUDE_jim_juice_consumption_l1628_162880

theorem jim_juice_consumption (susan_juice : ℚ) (jim_fraction : ℚ) :
  susan_juice = 3/8 →
  jim_fraction = 5/6 →
  jim_fraction * susan_juice = 5/16 := by
  sorry

end NUMINAMATH_CALUDE_jim_juice_consumption_l1628_162880


namespace NUMINAMATH_CALUDE_quadratic_root_range_l1628_162897

theorem quadratic_root_range (a : ℝ) :
  (∃ x y : ℝ, x > 0 ∧ y < 0 ∧ x^2 + a*x + a^2 - 1 = 0 ∧ y^2 + a*y + a^2 - 1 = 0) →
  -1 < a ∧ a < 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_range_l1628_162897


namespace NUMINAMATH_CALUDE_no_solution_for_part_a_unique_solution_for_part_b_l1628_162886

-- Define S(x) as the sum of digits of a natural number
def S (x : ℕ) : ℕ := sorry

-- Theorem for part (a)
theorem no_solution_for_part_a :
  ¬ ∃ x : ℕ, x + S x + S (S x) = 1993 := by sorry

-- Theorem for part (b)
theorem unique_solution_for_part_b :
  ∃! x : ℕ, x + S x + S (S x) + S (S (S x)) = 1993 ∧ x = 1963 := by sorry

end NUMINAMATH_CALUDE_no_solution_for_part_a_unique_solution_for_part_b_l1628_162886


namespace NUMINAMATH_CALUDE_unique_prime_n_l1628_162885

def f (n : ℕ+) : ℤ := -n^4 + n^3 - 4*n^2 + 18*n - 19

theorem unique_prime_n : ∃! (n : ℕ+), Nat.Prime (Int.natAbs (f n)) :=
sorry

end NUMINAMATH_CALUDE_unique_prime_n_l1628_162885


namespace NUMINAMATH_CALUDE_golden_rabbit_cards_count_l1628_162851

/-- The total number of possible four-digit combinations -/
def total_combinations : ℕ := 10000

/-- The number of digits that are not 6 or 8 -/
def available_digits : ℕ := 8

/-- The number of digits in the combination -/
def combination_length : ℕ := 4

/-- The number of combinations without 6 or 8 -/
def combinations_without_6_or_8 : ℕ := available_digits ^ combination_length

/-- The number of "Golden Rabbit Cards" -/
def golden_rabbit_cards : ℕ := total_combinations - combinations_without_6_or_8

theorem golden_rabbit_cards_count : golden_rabbit_cards = 5904 := by
  sorry

end NUMINAMATH_CALUDE_golden_rabbit_cards_count_l1628_162851


namespace NUMINAMATH_CALUDE_arithmetic_sequence_nth_term_l1628_162874

def arithmetic_sequence (a b c : ℝ) : Prop :=
  b - a = c - b

def nth_term (a d : ℝ) (n : ℕ) : ℝ :=
  a + (n - 1) * d

theorem arithmetic_sequence_nth_term 
  (x : ℝ) (n : ℕ) 
  (h1 : arithmetic_sequence (3*x - 4) (7*x - 14) (4*x + 5))
  (h2 : ∃ (a d : ℝ), nth_term a d n = 4013 ∧ a = 3*x - 4 ∧ d = (7*x - 14) - (3*x - 4)) :
  n = 610 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_nth_term_l1628_162874


namespace NUMINAMATH_CALUDE_cake_muffin_probability_l1628_162866

/-- The probability of selecting a buyer who purchases neither cake mix nor muffin mix -/
theorem cake_muffin_probability (total : ℕ) (cake : ℕ) (muffin : ℕ) (both : ℕ)
  (h_total : total = 100)
  (h_cake : cake = 50)
  (h_muffin : muffin = 40)
  (h_both : both = 17) :
  (total - (cake + muffin - both)) / total = 27 / 100 :=
by sorry

end NUMINAMATH_CALUDE_cake_muffin_probability_l1628_162866


namespace NUMINAMATH_CALUDE_tangent_circles_m_value_l1628_162811

/-- The value of m for which the circle x² + y² = 1 is tangent to the circle x² + y² + 6x - 8y + m = 0 -/
theorem tangent_circles_m_value : ∃ m : ℝ, 
  (∀ x y : ℝ, x^2 + y^2 = 1 → x^2 + y^2 + 6*x - 8*y + m = 0 → 
    (x + 3)^2 + (y - 4)^2 = 5^2 ∨ (x + 3)^2 + (y - 4)^2 = 4^2) ∧
  (m = -11 ∨ m = 9) := by
  sorry

end NUMINAMATH_CALUDE_tangent_circles_m_value_l1628_162811


namespace NUMINAMATH_CALUDE_fraction_equality_l1628_162881

theorem fraction_equality : (3/7 + 5/8) / (5/12 + 2/15) = 295/154 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1628_162881


namespace NUMINAMATH_CALUDE_rectangular_plot_breadth_l1628_162854

/-- Properties of a rectangular plot -/
structure RectangularPlot where
  breadth : ℝ
  length : ℝ
  area : ℝ
  length_is_thrice_breadth : length = 3 * breadth
  area_formula : area = length * breadth

/-- Theorem: The breadth of a rectangular plot with given properties is 11 meters -/
theorem rectangular_plot_breadth (plot : RectangularPlot) 
  (h : plot.area = 363) : plot.breadth = 11 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_plot_breadth_l1628_162854


namespace NUMINAMATH_CALUDE_position_of_2010_l1628_162835

/-- The sum of the first n terms of the arithmetic sequence representing the number of integers in each row -/
def rowSum (n : ℕ) : ℕ := n^2

/-- The first number in the nth row -/
def firstInRow (n : ℕ) : ℕ := rowSum (n - 1) + 1

/-- The position of a number in the table -/
structure Position where
  row : ℕ
  column : ℕ

/-- Find the position of a number in the table -/
def findPosition (num : ℕ) : Position :=
  let row := (Nat.sqrt (num - 1) + 1)
  let column := num - firstInRow row + 1
  ⟨row, column⟩

theorem position_of_2010 : findPosition 2010 = ⟨45, 74⟩ := by
  sorry

end NUMINAMATH_CALUDE_position_of_2010_l1628_162835


namespace NUMINAMATH_CALUDE_pentagon_enclosure_percentage_l1628_162895

/-- Represents a tiling pattern on a plane -/
structure TilingPattern where
  smallSquaresPerLargeSquare : ℕ
  pentagonsPerLargeSquare : ℕ

/-- Calculates the percentage of the plane enclosed by pentagons -/
def percentEnclosedByPentagons (pattern : TilingPattern) : ℚ :=
  (pattern.pentagonsPerLargeSquare : ℚ) / (pattern.smallSquaresPerLargeSquare : ℚ) * 100

/-- The specific tiling pattern described in the problem -/
def problemPattern : TilingPattern :=
  { smallSquaresPerLargeSquare := 16
  , pentagonsPerLargeSquare := 5 }

theorem pentagon_enclosure_percentage :
  percentEnclosedByPentagons problemPattern = 31.25 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_enclosure_percentage_l1628_162895


namespace NUMINAMATH_CALUDE_simple_interest_increase_l1628_162859

/-- Given that the simple interest on $2000 increases by $40 when the time increases by x years,
    and the rate percent per annum is 0.5, prove that x = 4. -/
theorem simple_interest_increase (x : ℝ) : 
  (2000 * 0.5 * x) / 100 = 40 → x = 4 := by sorry

end NUMINAMATH_CALUDE_simple_interest_increase_l1628_162859


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l1628_162865

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The domain of f is an infinite set -/
def InfiniteDomain (f : ℝ → ℝ) : Prop :=
  ∀ n : ℕ, ∃ (S : Finset ℝ), S.card = n ∧ ∀ x ∈ S, f x ≠ 0

/-- There exist infinitely many real numbers x in the domain such that f(-x) = f(x) -/
def InfinitelyManySymmetricPoints (f : ℝ → ℝ) : Prop :=
  ∀ n : ℕ, ∃ (S : Finset ℝ), S.card = n ∧ ∀ x ∈ S, f (-x) = f x

theorem necessary_not_sufficient_condition (f : ℝ → ℝ) :
  InfiniteDomain f →
  (IsEven f → InfinitelyManySymmetricPoints f) ∧
  ¬(InfinitelyManySymmetricPoints f → IsEven f) :=
by sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l1628_162865


namespace NUMINAMATH_CALUDE_geometric_series_sum_special_series_sum_l1628_162884

theorem geometric_series_sum (a : ℝ) (r : ℝ) (h : |r| < 1) :
  ∑' n, a * r^n = a / (1 - r) :=
sorry

/-- The sum of the infinite series 5 + 6(1/1000) + 7(1/1000)^2 + 8(1/1000)^3 + ... is 4995005/998001 -/
theorem special_series_sum :
  ∑' n : ℕ, (n + 5 : ℝ) * (1/1000)^(n-1) = 4995005 / 998001 :=
sorry

end NUMINAMATH_CALUDE_geometric_series_sum_special_series_sum_l1628_162884


namespace NUMINAMATH_CALUDE_nine_keys_required_l1628_162821

/-- Represents the warehouse setup and retrieval task -/
structure WarehouseSetup where
  total_cabinets : ℕ
  boxes_per_cabinet : ℕ
  phones_per_box : ℕ
  phones_to_retrieve : ℕ

/-- Calculates the minimum number of keys required for the given warehouse setup -/
def min_keys_required (setup : WarehouseSetup) : ℕ :=
  let boxes_needed := (setup.phones_to_retrieve + setup.phones_per_box - 1) / setup.phones_per_box
  let cabinets_needed := (boxes_needed + setup.boxes_per_cabinet - 1) / setup.boxes_per_cabinet
  boxes_needed + cabinets_needed + 1

/-- Theorem stating that for the given warehouse setup, 9 keys are required -/
theorem nine_keys_required : 
  let setup : WarehouseSetup := {
    total_cabinets := 8,
    boxes_per_cabinet := 4,
    phones_per_box := 10,
    phones_to_retrieve := 52
  }
  min_keys_required setup = 9 := by
  sorry

end NUMINAMATH_CALUDE_nine_keys_required_l1628_162821


namespace NUMINAMATH_CALUDE_solve_system_l1628_162890

theorem solve_system (x y z : ℝ) 
  (eq1 : x + 2*y = 10)
  (eq2 : y = 3)
  (eq3 : x - 3*y + z = 7) :
  z = 12 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l1628_162890


namespace NUMINAMATH_CALUDE_pounds_in_ton_l1628_162870

theorem pounds_in_ton (ounces_per_pound : ℕ) (num_packets : ℕ) (packet_weight_pounds : ℕ) 
  (packet_weight_ounces : ℕ) (bag_capacity_tons : ℕ) :
  ounces_per_pound = 16 →
  num_packets = 1680 →
  packet_weight_pounds = 16 →
  packet_weight_ounces = 4 →
  bag_capacity_tons = 13 →
  ∃ (pounds_per_ton : ℕ), pounds_per_ton = 2100 :=
by
  sorry

#check pounds_in_ton

end NUMINAMATH_CALUDE_pounds_in_ton_l1628_162870


namespace NUMINAMATH_CALUDE_third_term_value_l1628_162852

-- Define the sequence sum function
def S (n : ℕ) : ℚ := (n + 1 : ℚ) / (n + 2 : ℚ)

-- Define the sequence term function
def a (n : ℕ) : ℚ :=
  if n = 1 then S 1
  else S n - S (n - 1)

-- Theorem statement
theorem third_term_value : a 3 = 1 / 20 := by
  sorry

end NUMINAMATH_CALUDE_third_term_value_l1628_162852


namespace NUMINAMATH_CALUDE_given_number_scientific_notation_l1628_162873

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_normalized : 1 ≤ coefficient ∧ coefficient < 10

/-- The given number in meters -/
def given_number : ℝ := 0.0000084

/-- The expected scientific notation representation -/
def expected_representation : ScientificNotation := {
  coefficient := 8.4
  exponent := -6
  is_normalized := by sorry
}

/-- Theorem stating that the given number is equal to its scientific notation representation -/
theorem given_number_scientific_notation : 
  given_number = expected_representation.coefficient * (10 : ℝ) ^ expected_representation.exponent := by
  sorry

end NUMINAMATH_CALUDE_given_number_scientific_notation_l1628_162873


namespace NUMINAMATH_CALUDE_exist_consecutive_amazing_numbers_l1628_162825

/-- Definition of an amazing number -/
def is_amazing (n : ℕ) : Prop :=
  ∃ a b c : ℕ, 
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    n = (Nat.gcd b c) * (Nat.gcd a (b*c)) + 
        (Nat.gcd c a) * (Nat.gcd b (c*a)) + 
        (Nat.gcd a b) * (Nat.gcd c (a*b))

/-- Theorem: There exist 2011 consecutive amazing numbers -/
theorem exist_consecutive_amazing_numbers : 
  ∃ start : ℕ, ∀ i : ℕ, i < 2011 → is_amazing (start + i) := by
  sorry

end NUMINAMATH_CALUDE_exist_consecutive_amazing_numbers_l1628_162825


namespace NUMINAMATH_CALUDE_max_consecutive_working_days_l1628_162819

/-- Represents a day of the week -/
inductive DayOfWeek
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday
| Sunday

/-- Represents a date in a month -/
structure Date where
  day : Nat
  dayOfWeek : DayOfWeek

/-- Checks if a given date is a working day for the guard -/
def isWorkingDay (d : Date) : Prop :=
  d.dayOfWeek = DayOfWeek.Tuesday ∨ 
  d.dayOfWeek = DayOfWeek.Friday ∨ 
  d.day % 2 = 1

/-- Represents a sequence of consecutive dates -/
def ConsecutiveDates (n : Nat) := Fin n → Date

/-- Checks if all dates in a sequence are working days -/
def allWorkingDays (dates : ConsecutiveDates n) : Prop :=
  ∀ i, isWorkingDay (dates i)

/-- The main theorem: The maximum number of consecutive working days is 6 -/
theorem max_consecutive_working_days :
  (∃ (dates : ConsecutiveDates 6), allWorkingDays dates) ∧
  (∀ (dates : ConsecutiveDates 7), ¬ allWorkingDays dates) :=
sorry

end NUMINAMATH_CALUDE_max_consecutive_working_days_l1628_162819


namespace NUMINAMATH_CALUDE_continued_fraction_equality_l1628_162820

/-- Defines the continued fraction [2; 2, ..., 2] with n occurrences of 2 -/
def continued_fraction (n : ℕ) : ℚ :=
  if n = 0 then 2
  else 2 + 1 / continued_fraction (n - 1)

/-- The main theorem stating the equality of the continued fraction and the algebraic expression -/
theorem continued_fraction_equality (n : ℕ) :
  continued_fraction n = (((1 + Real.sqrt 2) ^ (n + 1) - (1 - Real.sqrt 2) ^ (n + 1)) /
                          ((1 + Real.sqrt 2) ^ n - (1 - Real.sqrt 2) ^ n)) := by
  sorry


end NUMINAMATH_CALUDE_continued_fraction_equality_l1628_162820


namespace NUMINAMATH_CALUDE_permutation_sum_consecutive_l1628_162837

theorem permutation_sum_consecutive (n : ℕ) (h : n ≥ 2) :
  (∃ (a b : Fin n → Fin n),
    Function.Bijective a ∧ Function.Bijective b ∧
    ∃ (k : ℕ), ∀ i : Fin n, (a i).val + (b i).val = k + i.val) ↔
  Odd n :=
sorry

end NUMINAMATH_CALUDE_permutation_sum_consecutive_l1628_162837


namespace NUMINAMATH_CALUDE_valid_schedules_l1628_162802

/-- Represents the number of periods in a day -/
def total_periods : ℕ := 9

/-- Represents the number of morning periods -/
def morning_periods : ℕ := 5

/-- Represents the number of afternoon periods -/
def afternoon_periods : ℕ := 4

/-- Represents the number of classes to be taught -/
def classes_to_teach : ℕ := 3

/-- Calculates the number of ways to arrange n items taken k at a time -/
def arrangement (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

/-- Calculates the number of prohibited arrangements in the morning -/
def morning_prohibited : ℕ := 3 * Nat.factorial classes_to_teach

/-- Calculates the number of prohibited arrangements in the afternoon -/
def afternoon_prohibited : ℕ := 2 * Nat.factorial classes_to_teach

/-- The main theorem stating the number of valid schedules -/
theorem valid_schedules : 
  arrangement total_periods classes_to_teach - morning_prohibited - afternoon_prohibited = 474 := by
  sorry


end NUMINAMATH_CALUDE_valid_schedules_l1628_162802


namespace NUMINAMATH_CALUDE_convex_polygon_interior_angles_l1628_162809

theorem convex_polygon_interior_angles (n : ℕ) : 
  n ≥ 3 →  -- Convex polygon has at least 3 sides
  (∀ k, k ∈ Finset.range n → 
    100 + k * 10 < 180) →  -- All interior angles are less than 180°
  (100 + (n - 1) * 10 ≥ 180) →  -- The largest angle is at least 180°
  n = 8 := by
  sorry

end NUMINAMATH_CALUDE_convex_polygon_interior_angles_l1628_162809

import Mathlib

namespace NUMINAMATH_CALUDE_hoseok_number_problem_l1664_166406

theorem hoseok_number_problem (x : ℝ) : 15 * x = 45 → x - 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_hoseok_number_problem_l1664_166406


namespace NUMINAMATH_CALUDE_f_is_quadratic_l1664_166436

/-- Definition of a quadratic equation in terms of x -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing 3x² + 1 -/
def f (x : ℝ) : ℝ := 3 * x^2 + 1

/-- Theorem stating that f is a quadratic equation -/
theorem f_is_quadratic : is_quadratic_equation f := by
  sorry


end NUMINAMATH_CALUDE_f_is_quadratic_l1664_166436


namespace NUMINAMATH_CALUDE_intersection_A_B_l1664_166474

-- Define set A
def A : Set ℝ := {x | (x - 2) / (x + 1) ≥ 0}

-- Define set B
def B : Set ℝ := {y | 0 ≤ y ∧ y < 4}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {x | 2 ≤ x ∧ x < 4} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l1664_166474


namespace NUMINAMATH_CALUDE_zeros_after_one_in_8000_to_50_l1664_166405

theorem zeros_after_one_in_8000_to_50 :
  let n : ℕ := 8000
  let k : ℕ := 50
  let base_ten_factor : ℕ := 3
  n = 8 * (10 ^ base_ten_factor) →
  (∃ m : ℕ, n^k = m * 10^(base_ten_factor * k) ∧ m % 10 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_zeros_after_one_in_8000_to_50_l1664_166405


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l1664_166414

theorem sum_of_roots_quadratic (b : ℝ) (x₁ x₂ : ℝ) : 
  (x₁^2 - 2*x₁ + b = 0) → (x₂^2 - 2*x₂ + b = 0) → x₁ + x₂ = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l1664_166414


namespace NUMINAMATH_CALUDE_base_five_product_l1664_166488

/-- Converts a base 5 number to decimal --/
def baseToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- Converts a decimal number to base 5 --/
def decimalToBase (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
  aux n []

theorem base_five_product :
  let a := [2, 3, 1] -- represents 132₅ in reverse order
  let b := [2, 1]    -- represents 12₅ in reverse order
  let product := [4, 3, 1, 2] -- represents 2134₅ in reverse order
  (baseToDecimal a) * (baseToDecimal b) = baseToDecimal product ∧
  decimalToBase ((baseToDecimal a) * (baseToDecimal b)) = product.reverse :=
sorry

end NUMINAMATH_CALUDE_base_five_product_l1664_166488


namespace NUMINAMATH_CALUDE_complementary_angles_ratio_l1664_166499

theorem complementary_angles_ratio (a b : ℝ) : 
  a + b = 90 →  -- angles are complementary
  b / a = 2 / 7 →  -- ratio of angles is 2:7
  a = 110 :=  -- complement of larger angle is 110°
by sorry

end NUMINAMATH_CALUDE_complementary_angles_ratio_l1664_166499


namespace NUMINAMATH_CALUDE_polynomial_intersection_l1664_166422

-- Define the polynomials f and g
def f (a b x : ℝ) : ℝ := x^2 + a*x + b
def g (c d x : ℝ) : ℝ := x^2 + c*x + d

-- State the theorem
theorem polynomial_intersection (a b c d : ℝ) : 
  -- f and g are distinct
  f a b ≠ g c d →
  -- The x-coordinate of the vertex of f is a root of g
  g c d (-a/2) = 0 →
  -- The x-coordinate of the vertex of g is a root of f
  f a b (-c/2) = 0 →
  -- Both f and g have the same minimum value
  (∃ (k : ℝ), ∀ (x : ℝ), f a b x ≥ k ∧ g c d x ≥ k) →
  -- The graphs of f and g intersect at (200, -200)
  f a b 200 = -200 ∧ g c d 200 = -200 →
  -- Conclusion: a + c = -800
  a + c = -800 := by
sorry

end NUMINAMATH_CALUDE_polynomial_intersection_l1664_166422


namespace NUMINAMATH_CALUDE_grid_paths_6x5_10_l1664_166460

/-- The number of different paths on a grid --/
def grid_paths (width height path_length : ℕ) : ℕ :=
  Nat.choose path_length height

/-- Theorem: The number of different paths on a 6x5 grid with path length 10 is 210 --/
theorem grid_paths_6x5_10 :
  grid_paths 6 5 10 = 210 := by
  sorry

end NUMINAMATH_CALUDE_grid_paths_6x5_10_l1664_166460


namespace NUMINAMATH_CALUDE_train_length_l1664_166445

/-- Calculates the length of a train given its speed, the speed of a motorbike it overtakes, and the time it takes to overtake. -/
theorem train_length (train_speed : ℝ) (motorbike_speed : ℝ) (overtake_time : ℝ) : 
  train_speed = 100 → 
  motorbike_speed = 64 → 
  overtake_time = 20 → 
  (train_speed - motorbike_speed) * overtake_time * (1000 / 3600) = 200 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l1664_166445


namespace NUMINAMATH_CALUDE_complex_multiplication_l1664_166479

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem complex_multiplication :
  i * (2 - i) = 1 + 2*i :=
by sorry

end NUMINAMATH_CALUDE_complex_multiplication_l1664_166479


namespace NUMINAMATH_CALUDE_smallest_four_digit_congruent_to_one_mod_seventeen_l1664_166441

theorem smallest_four_digit_congruent_to_one_mod_seventeen :
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 17 = 1 → n ≥ 1003 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_congruent_to_one_mod_seventeen_l1664_166441


namespace NUMINAMATH_CALUDE_largest_k_value_l1664_166469

theorem largest_k_value (x y k : ℝ) : 
  (2 * x + y = k) →
  (3 * x + y = 3) →
  (x - 2 * y ≥ 1) →
  (∀ m : ℤ, m > k → ¬(∃ x' y' : ℝ, 2 * x' + y' = m ∧ 3 * x' + y' = 3 ∧ x' - 2 * y' ≥ 1)) →
  k ≤ 2 ∧ (∃ x' y' : ℝ, 2 * x' + y' = 2 ∧ 3 * x' + y' = 3 ∧ x' - 2 * y' ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_largest_k_value_l1664_166469


namespace NUMINAMATH_CALUDE_digit_150_is_3_l1664_166437

/-- The decimal representation of 1/13 -/
def decimal_rep_1_13 : ℕ → Fin 10
  | n => Fin.ofNat ((10 * (10^n % 13)) / 13)

/-- The length of the repeating block in the decimal representation of 1/13 -/
def rep_length : ℕ := 6

/-- The 150th digit after the decimal point in the decimal representation of 1/13 -/
def digit_150 : Fin 10 := decimal_rep_1_13 149

theorem digit_150_is_3 : digit_150 = 3 := by sorry

end NUMINAMATH_CALUDE_digit_150_is_3_l1664_166437


namespace NUMINAMATH_CALUDE_article_original_price_l1664_166439

/-- Given an article sold with an 18% profit resulting in a profit of 542.8,
    prove that the original price of the article was 3016. -/
theorem article_original_price (profit_percentage : ℝ) (profit : ℝ) (original_price : ℝ) :
  profit_percentage = 18 →
  profit = 542.8 →
  profit = original_price * (profit_percentage / 100) →
  original_price = 3016 := by
  sorry

end NUMINAMATH_CALUDE_article_original_price_l1664_166439


namespace NUMINAMATH_CALUDE_mother_triple_age_l1664_166491

/-- Represents the age difference between Serena and her mother -/
def age_difference : ℕ := 30

/-- Represents Serena's current age -/
def serena_age : ℕ := 9

/-- Represents the number of years until Serena's mother is three times as old as Serena -/
def years_until_triple : ℕ := 6

/-- Theorem stating that after 'years_until_triple' years, Serena's mother will be three times as old as Serena -/
theorem mother_triple_age :
  serena_age + years_until_triple = (serena_age + age_difference + years_until_triple) / 3 :=
sorry

end NUMINAMATH_CALUDE_mother_triple_age_l1664_166491


namespace NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l1664_166497

/-- An arithmetic sequence with common difference d -/
def arithmetic_sequence (a : ℕ → ℚ) (d : ℚ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

/-- Three terms form a geometric sequence -/
def geometric_sequence (x y z : ℚ) : Prop :=
  y * y = x * z

/-- The common ratio of a geometric sequence -/
def common_ratio (x y : ℚ) : ℚ :=
  y / x

theorem arithmetic_geometric_ratio 
  (a : ℕ → ℚ) (d : ℚ) 
  (h_arith : arithmetic_sequence a d)
  (h_geom : geometric_sequence (a 1) (a 3) (a 4)) :
  (common_ratio (a 1) (a 3) = 1/2) ∨ (common_ratio (a 1) (a 3) = 1) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l1664_166497


namespace NUMINAMATH_CALUDE_some_number_in_formula_l1664_166426

/-- The toll formula for a truck using a certain bridge -/
def toll_formula (x : ℕ) (n : ℚ) : ℚ := 2.5 + 0.5 * (x - n)

/-- The number of axles on an 18-wheel truck -/
def axles_18_wheel_truck : ℕ := 5

/-- The toll for an 18-wheel truck -/
def toll_18_wheel_truck : ℚ := 4

theorem some_number_in_formula : 
  ∃ n : ℚ, toll_formula axles_18_wheel_truck n = toll_18_wheel_truck ∧ n = 2 :=
by sorry

end NUMINAMATH_CALUDE_some_number_in_formula_l1664_166426


namespace NUMINAMATH_CALUDE_sum_of_penultimate_terms_l1664_166480

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_of_penultimate_terms 
  (a : ℕ → ℚ) 
  (h_arithmetic : arithmetic_sequence a) 
  (h_first : a 0 = 3) 
  (h_last : ∃ n : ℕ, a n = 33 ∧ a (n - 1) + a (n - 2) = x + y) : 
  x + y = 51 := by
sorry

end NUMINAMATH_CALUDE_sum_of_penultimate_terms_l1664_166480


namespace NUMINAMATH_CALUDE_supplement_of_half_angle_l1664_166430

-- Define the angle α
def α : ℝ := 90 - 50

-- Theorem statement
theorem supplement_of_half_angle (h : α = 90 - 50) : 
  180 - (α / 2) = 160 := by sorry

end NUMINAMATH_CALUDE_supplement_of_half_angle_l1664_166430


namespace NUMINAMATH_CALUDE_rectangular_prism_sum_l1664_166433

/-- A rectangular prism is a three-dimensional shape with 6 rectangular faces. -/
structure RectangularPrism where
  faces : Nat
  edges : Nat
  vertices : Nat

/-- Properties of a rectangular prism -/
axiom rectangular_prism_properties (rp : RectangularPrism) : 
  rp.faces = 6 ∧ rp.edges = 12 ∧ rp.vertices = 8

/-- Theorem: The sum of faces, edges, and vertices of a rectangular prism is 26 -/
theorem rectangular_prism_sum (rp : RectangularPrism) : 
  rp.faces + rp.edges + rp.vertices = 26 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_sum_l1664_166433


namespace NUMINAMATH_CALUDE_four_stamps_cost_l1664_166456

/-- The cost of a single stamp in dollars -/
def stamp_cost : ℚ := 34/100

/-- The cost of two stamps in dollars -/
def two_stamps_cost : ℚ := 68/100

/-- The cost of three stamps in dollars -/
def three_stamps_cost : ℚ := 102/100

/-- Proves that the cost of four stamps is $1.36 -/
theorem four_stamps_cost :
  stamp_cost * 4 = 136/100 :=
by sorry

end NUMINAMATH_CALUDE_four_stamps_cost_l1664_166456


namespace NUMINAMATH_CALUDE_batsman_average_after_17th_inning_l1664_166446

/-- Represents a batsman's score statistics -/
structure BatsmanStats where
  inningsPlayed : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the new average after an innings -/
def newAverage (stats : BatsmanStats) (runsScored : ℕ) : ℚ :=
  (stats.totalRuns + runsScored) / (stats.inningsPlayed + 1)

/-- Theorem: A batsman's average after 17th inning is 39, given the conditions -/
theorem batsman_average_after_17th_inning
  (stats : BatsmanStats)
  (h1 : stats.inningsPlayed = 16)
  (h2 : newAverage stats 87 = stats.average + 3)
  : newAverage stats 87 = 39 := by
  sorry


end NUMINAMATH_CALUDE_batsman_average_after_17th_inning_l1664_166446


namespace NUMINAMATH_CALUDE_manufacturing_department_percentage_l1664_166421

theorem manufacturing_department_percentage (total_degrees : ℝ) (manufacturing_degrees : ℝ) :
  total_degrees = 360 →
  manufacturing_degrees = 216 →
  (manufacturing_degrees / total_degrees) * 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_manufacturing_department_percentage_l1664_166421


namespace NUMINAMATH_CALUDE_continuous_function_property_l1664_166411

theorem continuous_function_property (f : ℝ → ℝ) 
  (h_cont : Continuous f) 
  (h_prop : ∀ x y : ℝ, f (x + y) * f (x - y) = f x ^ 2) :
  (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_continuous_function_property_l1664_166411


namespace NUMINAMATH_CALUDE_sin_2alpha_proof_l1664_166419

theorem sin_2alpha_proof (α : ℝ) (h : Real.cos (π / 4 - α) = 3 / 5) : 
  Real.sin (2 * α) = -7 / 25 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_proof_l1664_166419


namespace NUMINAMATH_CALUDE_parking_savings_yearly_parking_savings_l1664_166409

theorem parking_savings : ℕ → ℕ → ℕ → ℕ → ℕ → Prop :=
  fun weekly_rate monthly_rate weeks_per_year months_per_year savings =>
    weekly_rate * weeks_per_year - monthly_rate * months_per_year = savings

/-- Proof of yearly savings when renting monthly instead of weekly --/
theorem yearly_parking_savings : parking_savings 10 40 52 12 40 := by
  sorry

end NUMINAMATH_CALUDE_parking_savings_yearly_parking_savings_l1664_166409


namespace NUMINAMATH_CALUDE_sum_of_distances_l1664_166440

/-- Parabola with equation y^2 = 4x -/
structure Parabola where
  eq : ∀ x y : ℝ, y^2 = 4*x

/-- Line with equation 2x + y - 4 = 0 -/
structure Line where
  eq : ∀ x y : ℝ, 2*x + y - 4 = 0

/-- Point A with coordinates (1, 2) -/
def A : ℝ × ℝ := (1, 2)

/-- Point B, the other intersection of the parabola and line -/
def B : ℝ × ℝ := sorry

/-- F is the focus of the parabola -/
def F : ℝ × ℝ := sorry

/-- |FA| is the distance between F and A -/
def FA : ℝ := sorry

/-- |FB| is the distance between F and B -/
def FB : ℝ := sorry

/-- Theorem stating that |FA| + |FB| = 7 -/
theorem sum_of_distances (p : Parabola) (l : Line) : FA + FB = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_distances_l1664_166440


namespace NUMINAMATH_CALUDE_johns_tax_rate_johns_tax_rate_approx_30_percent_l1664_166465

/-- Calculates John's tax rate given the incomes and tax rates of John and Ingrid --/
theorem johns_tax_rate (john_income ingrid_income : ℝ) 
                       (ingrid_tax_rate combined_tax_rate : ℝ) : ℝ :=
  let total_income := john_income + ingrid_income
  let total_tax := combined_tax_rate * total_income
  let ingrid_tax := ingrid_tax_rate * ingrid_income
  let john_tax := total_tax - ingrid_tax
  john_tax / john_income

/-- John's tax rate is approximately 30.00% --/
theorem johns_tax_rate_approx_30_percent : 
  ∃ ε > 0, 
    |johns_tax_rate 58000 72000 0.40 0.3554 - 0.30| < ε ∧ 
    ε < 0.0001 :=
sorry

end NUMINAMATH_CALUDE_johns_tax_rate_johns_tax_rate_approx_30_percent_l1664_166465


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1664_166442

def solution_set (x : ℝ) : Prop := x ∈ Set.Ici 0 ∩ Set.Iio 2

theorem inequality_solution_set :
  ∀ x : ℝ, x ≠ 2 → (x / (x - 2) ≤ 0 ↔ solution_set x) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1664_166442


namespace NUMINAMATH_CALUDE_descent_problem_l1664_166464

/-- The number of floors Austin and Jake descended. -/
def floors : ℕ := sorry

/-- The number of steps Jake descends per second. -/
def steps_per_second : ℕ := 3

/-- The number of steps per floor. -/
def steps_per_floor : ℕ := 30

/-- The time (in seconds) it takes Austin to reach the ground floor using the elevator. -/
def austin_time : ℕ := 60

/-- The time (in seconds) it takes Jake to reach the ground floor using the stairs. -/
def jake_time : ℕ := 90

theorem descent_problem :
  floors = (jake_time * steps_per_second) / steps_per_floor := by
  sorry

end NUMINAMATH_CALUDE_descent_problem_l1664_166464


namespace NUMINAMATH_CALUDE_exists_complementary_not_acute_not_obtuse_l1664_166431

-- Define complementary angles
def complementary (a b : ℝ) : Prop := a + b = 180

-- Define acute angle
def acute (a : ℝ) : Prop := 0 < a ∧ a < 90

-- Define obtuse angle
def obtuse (a : ℝ) : Prop := 90 < a ∧ a < 180

-- Theorem statement
theorem exists_complementary_not_acute_not_obtuse :
  ∃ (a b : ℝ), complementary a b ∧ ¬(acute a ∨ obtuse a) ∧ ¬(acute b ∨ obtuse b) :=
sorry

end NUMINAMATH_CALUDE_exists_complementary_not_acute_not_obtuse_l1664_166431


namespace NUMINAMATH_CALUDE_f_monotone_increasing_f_monotone_decreasing_f_not_always_above_a_l1664_166450

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x - 1

-- Theorem 1: f(x) is monotonically increasing on ℝ iff a ≤ 0
theorem f_monotone_increasing (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ a ≤ 0 :=
sorry

-- Theorem 2: f(x) is monotonically decreasing on (-1, 1) iff a ≥ 3
theorem f_monotone_decreasing (a : ℝ) :
  (∀ x y : ℝ, -1 < x ∧ x < y ∧ y < 1 → f a x > f a y) ↔ a ≥ 3 :=
sorry

-- Theorem 3: ∃x ∈ ℝ, f(x) < a
theorem f_not_always_above_a (a : ℝ) :
  ∃ x : ℝ, f a x < a :=
sorry

end NUMINAMATH_CALUDE_f_monotone_increasing_f_monotone_decreasing_f_not_always_above_a_l1664_166450


namespace NUMINAMATH_CALUDE_simplest_quadratic_radical_l1664_166418

def is_simplest_quadratic_radical (x : ℝ) : Prop :=
  ∃ (n : ℕ), x = Real.sqrt n ∧ 
  (∀ (m : ℕ), m ^ 2 ∣ n → m = 1) ∧
  (∀ (a b : ℕ), n = a / b → b = 1)

theorem simplest_quadratic_radical :
  ¬ is_simplest_quadratic_radical (Real.sqrt (1 / 2)) ∧
  is_simplest_quadratic_radical (Real.sqrt 3) ∧
  ¬ is_simplest_quadratic_radical (Real.sqrt 8) ∧
  ¬ is_simplest_quadratic_radical (Real.sqrt 0.1) :=
by sorry

end NUMINAMATH_CALUDE_simplest_quadratic_radical_l1664_166418


namespace NUMINAMATH_CALUDE_community_center_ticket_sales_l1664_166408

/-- Calculates the total amount collected from ticket sales given the ticket prices and quantities sold. -/
def total_amount_collected (adult_price child_price : ℕ) (total_tickets adult_tickets : ℕ) : ℕ :=
  adult_price * adult_tickets + child_price * (total_tickets - adult_tickets)

/-- Theorem stating that given the specific conditions of the problem, the total amount collected is $275. -/
theorem community_center_ticket_sales :
  let adult_price : ℕ := 5
  let child_price : ℕ := 2
  let total_tickets : ℕ := 85
  let adult_tickets : ℕ := 35
  total_amount_collected adult_price child_price total_tickets adult_tickets = 275 := by
sorry

end NUMINAMATH_CALUDE_community_center_ticket_sales_l1664_166408


namespace NUMINAMATH_CALUDE_y1_greater_than_y2_l1664_166472

/-- A linear function f(x) = -x + 1 -/
def f (x : ℝ) : ℝ := -x + 1

theorem y1_greater_than_y2 (y₁ y₂ : ℝ) 
  (h₁ : f (-1) = y₁) 
  (h₂ : f 2 = y₂) : 
  y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_y1_greater_than_y2_l1664_166472


namespace NUMINAMATH_CALUDE_equation_solution_l1664_166401

theorem equation_solution (b c : ℝ) (θ : ℝ) :
  let x := (b^2 - c^2 * Real.sin θ^2) / (2 * b)
  x^2 + c^2 * Real.sin θ^2 = (b - x)^2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1664_166401


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l1664_166494

/-- Given a geometric sequence with common ratio 2 and all positive terms,
    if the product of the 4th and 12th terms is 64, then the 7th term is 4. -/
theorem geometric_sequence_problem (a : ℕ → ℝ) : 
  (∀ n, a (n + 1) = 2 * a n) →  -- Common ratio is 2
  (∀ n, a n > 0) →              -- All terms are positive
  a 4 * a 12 = 64 →             -- Product of 4th and 12th terms is 64
  a 7 = 4 := by                 -- The 7th term is 4
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l1664_166494


namespace NUMINAMATH_CALUDE_cement_mixture_weight_l1664_166478

theorem cement_mixture_weight :
  ∀ (W : ℝ),
    (1/3 : ℝ) * W + (1/4 : ℝ) * W + 10 = W →
    W = 24 := by
  sorry

end NUMINAMATH_CALUDE_cement_mixture_weight_l1664_166478


namespace NUMINAMATH_CALUDE_consecutive_integers_product_sum_l1664_166483

theorem consecutive_integers_product_sum (x : ℕ) : 
  x > 0 ∧ x * (x + 1) = 812 → x + (x + 1) = 57 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_sum_l1664_166483


namespace NUMINAMATH_CALUDE_expected_moves_is_six_l1664_166486

/-- Represents the state of the glasses --/
inductive GlassState
| Full
| Empty

/-- Represents the configuration of the 4 glasses --/
structure GlassConfig :=
(glass1 : GlassState)
(glass2 : GlassState)
(glass3 : GlassState)
(glass4 : GlassState)

/-- The initial configuration --/
def initialConfig : GlassConfig :=
{ glass1 := GlassState.Full,
  glass2 := GlassState.Empty,
  glass3 := GlassState.Full,
  glass4 := GlassState.Empty }

/-- The target configuration --/
def targetConfig : GlassConfig :=
{ glass1 := GlassState.Empty,
  glass2 := GlassState.Full,
  glass3 := GlassState.Empty,
  glass4 := GlassState.Full }

/-- Represents a valid move (pouring from a full glass to an empty one) --/
inductive ValidMove : GlassConfig → GlassConfig → Prop

/-- The expected number of moves to reach the target configuration --/
noncomputable def expectedMoves : ℝ := 6

/-- Main theorem: The expected number of moves from initial to target config is 6 --/
theorem expected_moves_is_six :
  expectedMoves = 6 :=
sorry

end NUMINAMATH_CALUDE_expected_moves_is_six_l1664_166486


namespace NUMINAMATH_CALUDE_isosceles_triangle_m_value_l1664_166420

/-- An isosceles triangle with side length 8 and other sides as roots of x^2 - 10x + m = 0 -/
structure IsoscelesTriangle where
  m : ℝ
  BC : ℝ
  AB_AC_eq : x^2 - 10*x + m = 0 → x = AB ∨ x = AC
  BC_eq : BC = 8
  isosceles : AB = AC

/-- The value of m in the isosceles triangle is either 25 or 16 -/
theorem isosceles_triangle_m_value (t : IsoscelesTriangle) : t.m = 25 ∨ t.m = 16 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_m_value_l1664_166420


namespace NUMINAMATH_CALUDE_negative_three_to_fourth_power_l1664_166416

theorem negative_three_to_fourth_power :
  -3^4 = -(3 * 3 * 3 * 3) := by sorry

end NUMINAMATH_CALUDE_negative_three_to_fourth_power_l1664_166416


namespace NUMINAMATH_CALUDE_sum_even_factors_1176_l1664_166498

def sum_even_factors (n : ℕ) : ℕ := sorry

theorem sum_even_factors_1176 : sum_even_factors 1176 = 3192 := by sorry

end NUMINAMATH_CALUDE_sum_even_factors_1176_l1664_166498


namespace NUMINAMATH_CALUDE_simplify_expression_l1664_166487

theorem simplify_expression : 0.2 * 0.4 + 0.6 * 0.8 = 0.56 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1664_166487


namespace NUMINAMATH_CALUDE_min_k_value_k_range_l1664_166496

/-- Given that for all a ∈ (-∞, 0) and all x ∈ (0, +∞), 
    the inequality x^2 + (3-a)x + 3 - 2a^2 < ke^x holds,
    prove that the minimum value of k is 3. -/
theorem min_k_value (k : ℝ) : 
  (∀ a < 0, ∀ x > 0, x^2 + (3-a)*x + 3 - 2*a^2 < k * Real.exp x) → 
  k ≥ 3 := by
  sorry

/-- The range of k is [3, +∞) -/
theorem k_range (k : ℝ) : 
  (∀ a < 0, ∀ x > 0, x^2 + (3-a)*x + 3 - 2*a^2 < k * Real.exp x) ↔ 
  k ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_min_k_value_k_range_l1664_166496


namespace NUMINAMATH_CALUDE_decimal_56_to_binary_binary_to_decimal_56_decimal_56_binary_equivalence_l1664_166403

/-- Converts a natural number to its binary representation as a list of bits -/
def to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec aux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
  aux n

/-- Converts a list of bits to its decimal representation -/
def from_binary (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

theorem decimal_56_to_binary :
  to_binary 56 = [false, false, false, true, true, true] :=
by sorry

theorem binary_to_decimal_56 :
  from_binary [false, false, false, true, true, true] = 56 :=
by sorry

theorem decimal_56_binary_equivalence :
  to_binary 56 = [false, false, false, true, true, true] ∧
  from_binary [false, false, false, true, true, true] = 56 :=
by sorry

end NUMINAMATH_CALUDE_decimal_56_to_binary_binary_to_decimal_56_decimal_56_binary_equivalence_l1664_166403


namespace NUMINAMATH_CALUDE_bowling_ball_weight_proof_l1664_166410

/-- The weight of one bowling ball in pounds -/
def bowling_ball_weight : ℚ := 56 / 5

/-- The weight of one canoe in pounds -/
def canoe_weight : ℚ := 28

theorem bowling_ball_weight_proof :
  (5 : ℚ) * bowling_ball_weight = 2 * canoe_weight ∧
  (3 : ℚ) * canoe_weight = 84 →
  bowling_ball_weight = 56 / 5 := by
sorry

end NUMINAMATH_CALUDE_bowling_ball_weight_proof_l1664_166410


namespace NUMINAMATH_CALUDE_sum_of_integers_l1664_166476

theorem sum_of_integers (a b : ℕ+) (h1 : a.val^2 - b.val^2 = 44) (h2 : a.val * b.val = 120) : 
  a.val + b.val = 22 := by
sorry

end NUMINAMATH_CALUDE_sum_of_integers_l1664_166476


namespace NUMINAMATH_CALUDE_log_cutting_problem_l1664_166493

/-- Represents the number of cuts needed to split a log into 1-meter pieces -/
def cuts_needed (length : ℕ) : ℕ := length - 1

/-- Represents the total number of logs -/
def total_logs : ℕ := 30

/-- Represents the total length of all logs in meters -/
def total_length : ℕ := 100

/-- Represents the possible lengths of logs in meters -/
inductive LogLength
| short : LogLength  -- 3 meters
| long : LogLength   -- 4 meters

/-- Calculates the minimum number of cuts needed for the given log configuration -/
def min_cuts (x y : ℕ) : Prop :=
  x + y = total_logs ∧
  3 * x + 4 * y = total_length ∧
  x * cuts_needed 3 + y * cuts_needed 4 = 70

theorem log_cutting_problem :
  ∃ x y : ℕ, min_cuts x y :=
sorry

end NUMINAMATH_CALUDE_log_cutting_problem_l1664_166493


namespace NUMINAMATH_CALUDE_max_chickens_and_chicks_optimal_chicken_count_l1664_166495

/-- Represents the chicken coop problem -/
structure ChickenCoop where
  area : ℝ
  chicken_space : ℝ
  chick_space : ℝ
  chicken_feed : ℝ
  chick_feed : ℝ
  max_feed : ℝ

/-- Defines the conditions of the problem -/
def problem_conditions : ChickenCoop :=
  { area := 240
  , chicken_space := 4
  , chick_space := 2
  , chicken_feed := 160
  , chick_feed := 40
  , max_feed := 8000
  }

/-- Checks if a given number of chickens and chicks satisfies the constraints -/
def satisfies_constraints (coop : ChickenCoop) (chickens : ℕ) (chicks : ℕ) : Prop :=
  (chickens : ℝ) * coop.chicken_space + (chicks : ℝ) * coop.chick_space ≤ coop.area ∧
  (chickens : ℝ) * coop.chicken_feed + (chicks : ℝ) * coop.chick_feed ≤ coop.max_feed

/-- Theorem stating the maximum number of chickens and chicks -/
theorem max_chickens_and_chicks (coop : ChickenCoop := problem_conditions) :
  satisfies_constraints coop 40 40 ∧
  (∀ c : ℕ, c > 40 → ¬satisfies_constraints coop c 40) ∧
  satisfies_constraints coop 0 120 ∧
  (∀ k : ℕ, k > 120 → ¬satisfies_constraints coop 0 k) := by
  sorry

/-- Theorem stating that 40 chickens and 40 chicks is optimal when maximizing chickens -/
theorem optimal_chicken_count (coop : ChickenCoop := problem_conditions) :
  ∀ c k : ℕ, satisfies_constraints coop c k →
    c ≤ 40 ∧ (c = 40 → k ≤ 40) := by
  sorry

end NUMINAMATH_CALUDE_max_chickens_and_chicks_optimal_chicken_count_l1664_166495


namespace NUMINAMATH_CALUDE_quadratic_minimum_l1664_166457

theorem quadratic_minimum (b : ℝ) : 
  ∃ (min : ℝ), (∀ x : ℝ, (1/2) * x^2 + 5*x - 3 ≥ (1/2) * min^2 + 5*min - 3) ∧ min = -5 :=
sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l1664_166457


namespace NUMINAMATH_CALUDE_orthographic_projection_area_l1664_166448

/-- The area of the orthographic projection of an equilateral triangle -/
theorem orthographic_projection_area (side_length : ℝ) (h : side_length = 2) :
  let original_area := (Real.sqrt 3 / 4) * side_length ^ 2
  let projection_area := (Real.sqrt 2 / 4) * original_area
  projection_area = Real.sqrt 6 / 4 := by
  sorry

end NUMINAMATH_CALUDE_orthographic_projection_area_l1664_166448


namespace NUMINAMATH_CALUDE_smallest_root_quadratic_l1664_166470

theorem smallest_root_quadratic (x : ℝ) :
  (9 * x^2 - 45 * x + 50 = 0) →
  (∀ y : ℝ, 9 * y^2 - 45 * y + 50 = 0 → x ≤ y) →
  x = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_root_quadratic_l1664_166470


namespace NUMINAMATH_CALUDE_cos_4050_degrees_l1664_166459

theorem cos_4050_degrees : Real.cos (4050 * π / 180) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_4050_degrees_l1664_166459


namespace NUMINAMATH_CALUDE_benjamin_walks_158_miles_l1664_166400

/-- Calculates the total miles Benjamin walks in a week -/
def total_miles_walked : ℕ :=
  let work_distance := 8
  let work_days := 5
  let dog_walk_distance := 3
  let dog_walks_per_day := 2
  let days_in_week := 7
  let friend_distance := 5
  let friend_visits := 1
  let store_distance := 4
  let store_visits := 2
  let hike_distance := 10

  let work_miles := work_distance * 2 * work_days
  let dog_miles := dog_walk_distance * dog_walks_per_day * days_in_week
  let friend_miles := friend_distance * 2 * friend_visits
  let store_miles := store_distance * 2 * store_visits
  let hike_miles := hike_distance

  work_miles + dog_miles + friend_miles + store_miles + hike_miles

theorem benjamin_walks_158_miles : total_miles_walked = 158 := by
  sorry

end NUMINAMATH_CALUDE_benjamin_walks_158_miles_l1664_166400


namespace NUMINAMATH_CALUDE_initial_milk_percentage_l1664_166463

/-- Given a mixture of milk and water, prove that the initial milk percentage is 84% -/
theorem initial_milk_percentage 
  (initial_volume : ℝ) 
  (added_water : ℝ) 
  (final_milk_percentage : ℝ) :
  initial_volume = 60 →
  added_water = 14.117647058823536 →
  final_milk_percentage = 68 →
  (initial_volume * (84 / 100)) / (initial_volume + added_water) = final_milk_percentage / 100 :=
by sorry

end NUMINAMATH_CALUDE_initial_milk_percentage_l1664_166463


namespace NUMINAMATH_CALUDE_binomial_12_9_l1664_166475

theorem binomial_12_9 : Nat.choose 12 9 = 220 := by
  sorry

end NUMINAMATH_CALUDE_binomial_12_9_l1664_166475


namespace NUMINAMATH_CALUDE_berry_exchange_theorem_l1664_166434

/-- The number of blueberries in each blue box -/
def B : ℕ := 35

/-- The number of strawberries in each red box -/
def S : ℕ := 100 + B

/-- The change in total berries when exchanging one blue box for one red box -/
def ΔT : ℤ := S - B

theorem berry_exchange_theorem : ΔT = 65 := by
  sorry

end NUMINAMATH_CALUDE_berry_exchange_theorem_l1664_166434


namespace NUMINAMATH_CALUDE_smallest_number_range_l1664_166492

theorem smallest_number_range 
  (a b c d e : ℝ) 
  (h_distinct : a < b ∧ b < c ∧ c < d ∧ d < e) 
  (h_sum1 : a + b = 20) 
  (h_sum2 : a + c = 200) 
  (h_sum3 : d + e = 2014) 
  (h_sum4 : c + e = 2000) : 
  -793 < a ∧ a < 10 := by
sorry

end NUMINAMATH_CALUDE_smallest_number_range_l1664_166492


namespace NUMINAMATH_CALUDE_two_roots_implies_c_values_l1664_166432

-- Define the function f(x) = x³ - 3x + c
def f (c : ℝ) (x : ℝ) : ℝ := x^3 - 3*x + c

-- Define the property of having exactly two roots
def has_exactly_two_roots (f : ℝ → ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧
  ∀ (x : ℝ), f x = 0 → x = x₁ ∨ x = x₂

-- Theorem statement
theorem two_roots_implies_c_values (c : ℝ) :
  has_exactly_two_roots (f c) → c = -2 ∨ c = 2 :=
sorry

end NUMINAMATH_CALUDE_two_roots_implies_c_values_l1664_166432


namespace NUMINAMATH_CALUDE_at_least_one_solution_l1664_166402

open Complex

-- Define the equation
def satisfies_equation (z : ℂ) : Prop := exp z = z^2 + 1

-- Define the constraint
def within_bound (z : ℂ) : Prop := abs z < 20

-- Theorem statement
theorem at_least_one_solution :
  ∃ z : ℂ, satisfies_equation z ∧ within_bound z :=
sorry

end NUMINAMATH_CALUDE_at_least_one_solution_l1664_166402


namespace NUMINAMATH_CALUDE_complex_square_sum_l1664_166417

theorem complex_square_sum (a b : ℝ) (i : ℂ) (h : i * i = -1) :
  (a + b * i) ^ 2 = 3 + 4 * i → a ^ 2 + b ^ 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_square_sum_l1664_166417


namespace NUMINAMATH_CALUDE_person_a_work_time_l1664_166484

theorem person_a_work_time (b : ℝ) (combined_rate : ℝ) (combined_time : ℝ) 
  (hb : b = 45)
  (hcombined : combined_rate * combined_time = 1 / 9)
  (htime : combined_time = 2) :
  ∃ a : ℝ, a = 30 ∧ combined_rate = 1 / a + 1 / b := by
  sorry

end NUMINAMATH_CALUDE_person_a_work_time_l1664_166484


namespace NUMINAMATH_CALUDE_parabola_ellipse_focus_coincidence_l1664_166449

/-- The value of p for which the focus of the parabola y² = 2px coincides with 
    the right focus of the ellipse x²/5 + y² = 1 -/
theorem parabola_ellipse_focus_coincidence : ∃ p : ℝ, 
  (∀ x y : ℝ, y^2 = 2*p*x → x^2/5 + y^2 = 1 → x = 2) → p = 4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_ellipse_focus_coincidence_l1664_166449


namespace NUMINAMATH_CALUDE_jacket_price_proof_l1664_166467

/-- The original price of the jacket -/
def original_price : ℝ := 250

/-- The regular discount percentage -/
def regular_discount : ℝ := 0.4

/-- The weekend additional discount percentage -/
def weekend_discount : ℝ := 0.1

/-- The final price after both discounts -/
def final_price : ℝ := original_price * (1 - regular_discount) * (1 - weekend_discount)

theorem jacket_price_proof : final_price = 135 := by
  sorry

end NUMINAMATH_CALUDE_jacket_price_proof_l1664_166467


namespace NUMINAMATH_CALUDE_min_bottles_to_fill_l1664_166444

/-- The capacity of a small bottle in milliliters -/
def small_bottle_capacity : ℝ := 35

/-- The capacity of a large bottle in milliliters -/
def large_bottle_capacity : ℝ := 500

/-- The minimum number of small bottles needed to completely fill a large bottle -/
def min_bottles : ℕ := 15

theorem min_bottles_to_fill :
  ∃ (n : ℕ), n * small_bottle_capacity ≥ large_bottle_capacity ∧
  ∀ (m : ℕ), m * small_bottle_capacity ≥ large_bottle_capacity → n ≤ m ∧
  n = min_bottles :=
by sorry

end NUMINAMATH_CALUDE_min_bottles_to_fill_l1664_166444


namespace NUMINAMATH_CALUDE_min_c_value_l1664_166481

theorem min_c_value (a b c : ℕ) (h1 : a < b) (h2 : b < c) 
  (h3 : ∃! (x y : ℝ), 2*x + y = 2029 ∧ y = |x - a| + |x - b| + |x - c|) : 
  c ≥ 1015 :=
by sorry

end NUMINAMATH_CALUDE_min_c_value_l1664_166481


namespace NUMINAMATH_CALUDE_activity_participation_l1664_166471

def total_sample : ℕ := 100
def male_participants : ℕ := 60
def willing_to_participate : ℕ := 70
def males_willing : ℕ := 48
def females_not_willing : ℕ := 18

def chi_square (a b c d : ℕ) : ℚ :=
  let n := a + b + c + d
  (n * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

def critical_value : ℚ := 6635 / 1000

theorem activity_participation :
  let females_willing := willing_to_participate - males_willing
  let males_not_willing := male_participants - males_willing
  let female_participants := total_sample - male_participants
  let chi_sq := chi_square males_willing females_willing males_not_willing females_not_willing
  let male_proportion := (males_willing : ℚ) / male_participants
  let female_proportion := (females_willing : ℚ) / female_participants
  (chi_sq > critical_value) ∧
  (male_proportion > female_proportion) ∧
  (12 / 7 : ℚ) = (4 * 0 + 3 * 1 + 2 * 2 + 1 * 3 : ℚ) / (Nat.choose 7 3) := by sorry

end NUMINAMATH_CALUDE_activity_participation_l1664_166471


namespace NUMINAMATH_CALUDE_pentagon_angle_sequences_l1664_166435

def is_valid_sequence (x d : ℕ) : Prop :=
  x > 0 ∧ d > 0 ∧
  x + (x+d) + (x+2*d) + (x+3*d) + (x+4*d) = 540 ∧
  x + 4*d < 120

theorem pentagon_angle_sequences :
  ∃! n : ℕ, n > 0 ∧ 
  (∃ s : Finset (ℕ × ℕ), s.card = n ∧ 
    (∀ p ∈ s, is_valid_sequence p.1 p.2) ∧
    (∀ x d : ℕ, is_valid_sequence x d → (x, d) ∈ s)) :=
sorry

end NUMINAMATH_CALUDE_pentagon_angle_sequences_l1664_166435


namespace NUMINAMATH_CALUDE_coefficient_x6_is_180_l1664_166423

/-- The coefficient of x^6 in the binomial expansion of (x - 2/x)^10 -/
def coefficient_x6 : ℤ := 
  let n : ℕ := 10
  let k : ℕ := (n - 6) / 2
  (n.choose k) * (-2)^k

theorem coefficient_x6_is_180 : coefficient_x6 = 180 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x6_is_180_l1664_166423


namespace NUMINAMATH_CALUDE_sector_area_l1664_166453

theorem sector_area (θ : Real) (L : Real) (A : Real) :
  θ = π / 6 →
  L = 2 * π / 3 →
  A = 4 * π / 3 :=
by sorry

end NUMINAMATH_CALUDE_sector_area_l1664_166453


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1664_166490

/-- A hyperbola with eccentricity √6/2 has the equation x²/4 - y²/2 = 1 -/
theorem hyperbola_equation (e : ℝ) (h : e = Real.sqrt 6 / 2) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  (∀ (x y : ℝ), x^2 / (a^2) - y^2 / (b^2) = 1 ↔ 
    x^2 / 4 - y^2 / 2 = 1) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1664_166490


namespace NUMINAMATH_CALUDE_original_loaf_size_l1664_166427

def slices_per_sandwich : ℕ := 2
def days_with_one_sandwich : ℕ := 5
def sandwiches_on_saturday : ℕ := 2
def slices_left : ℕ := 6

theorem original_loaf_size :
  slices_per_sandwich * days_with_one_sandwich +
  slices_per_sandwich * sandwiches_on_saturday +
  slices_left = 20 := by
  sorry

end NUMINAMATH_CALUDE_original_loaf_size_l1664_166427


namespace NUMINAMATH_CALUDE_reciprocal_sum_theorem_l1664_166462

theorem reciprocal_sum_theorem (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x + y = 3 * x * y) : 
  1 / x + 1 / y = 3 := by
sorry

end NUMINAMATH_CALUDE_reciprocal_sum_theorem_l1664_166462


namespace NUMINAMATH_CALUDE_circle_placement_l1664_166452

theorem circle_placement (rectangle_width : ℝ) (rectangle_height : ℝ) 
  (num_squares : ℕ) (square_size : ℝ) (circle_diameter : ℝ) :
  rectangle_width = 20 ∧ 
  rectangle_height = 25 ∧ 
  num_squares = 120 ∧ 
  square_size = 1 ∧ 
  circle_diameter = 1 →
  ∃ (x y : ℝ), 
    0 ≤ x ∧ x ≤ rectangle_width ∧ 
    0 ≤ y ∧ y ≤ rectangle_height ∧ 
    ∀ (i : ℕ), i < num_squares →
      ∃ (sx sy : ℝ), 
        0 ≤ sx ∧ sx + square_size ≤ rectangle_width ∧
        0 ≤ sy ∧ sy + square_size ≤ rectangle_height ∧
        (x - sx)^2 + (y - sy)^2 ≥ (circle_diameter / 2 + square_size / 2)^2 :=
by
  sorry

end NUMINAMATH_CALUDE_circle_placement_l1664_166452


namespace NUMINAMATH_CALUDE_matrix_inverse_and_transformation_l1664_166407

def A : Matrix (Fin 2) (Fin 2) ℝ := !![2, 3; 1, 2]

theorem matrix_inverse_and_transformation :
  let A_inv : Matrix (Fin 2) (Fin 2) ℝ := !![2, -3; -1, 2]
  let P : Fin 2 → ℝ := ![3, -1]
  (A⁻¹ = A_inv) ∧ (A.mulVec P = ![3, 1]) := by sorry

end NUMINAMATH_CALUDE_matrix_inverse_and_transformation_l1664_166407


namespace NUMINAMATH_CALUDE_project_hours_difference_l1664_166466

/-- Given a project where three people charged time, prove that one person charged 100 more hours than another. -/
theorem project_hours_difference (total_hours kate_hours pat_hours mark_hours : ℕ) : 
  total_hours = 180 ∧ 
  pat_hours = 2 * kate_hours ∧ 
  pat_hours * 3 = mark_hours ∧
  total_hours = kate_hours + pat_hours + mark_hours →
  mark_hours = kate_hours + 100 := by
  sorry

end NUMINAMATH_CALUDE_project_hours_difference_l1664_166466


namespace NUMINAMATH_CALUDE_inverse_contrapositive_l1664_166415

theorem inverse_contrapositive (x y : ℝ) : x = 0 ∧ y = 2 → x + y = 2 := by
  sorry

end NUMINAMATH_CALUDE_inverse_contrapositive_l1664_166415


namespace NUMINAMATH_CALUDE_min_value_theorem_equality_condition_l1664_166482

theorem min_value_theorem (x : ℝ) (h : x > 0) : 4 * x + 1 / x^4 ≥ 5 := by
  sorry

theorem equality_condition : 4 * 1 + 1 / 1^4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_equality_condition_l1664_166482


namespace NUMINAMATH_CALUDE_open_box_volume_l1664_166477

/-- The volume of an open box formed by cutting squares from the corners of a rectangular sheet. -/
theorem open_box_volume 
  (sheet_length : ℝ) 
  (sheet_width : ℝ) 
  (cut_length : ℝ) 
  (h1 : sheet_length = 48)
  (h2 : sheet_width = 36)
  (h3 : cut_length = 7) : 
  (sheet_length - 2 * cut_length) * (sheet_width - 2 * cut_length) * cut_length = 5236 := by
  sorry

end NUMINAMATH_CALUDE_open_box_volume_l1664_166477


namespace NUMINAMATH_CALUDE_f_derivative_condition_implies_a_range_g_minimum_value_l1664_166413

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 + ((a-3)/2) * x^2 + (a^2-3*a) * x - 2*a

-- Define the derivative of f
def f_derivative (a : ℝ) (x : ℝ) : ℝ := x^2 + (a-3)*x + a^2 - 3*a

-- Define the function g
def g (a x₁ x₂ : ℝ) : ℝ := x₁^3 + x₂^3 + a^3

theorem f_derivative_condition_implies_a_range (a : ℝ) :
  (∀ x ∈ Set.Icc 1 2, f_derivative a x > a^2) → a ∈ Set.Ioi (-2) :=
sorry

theorem g_minimum_value (a x₁ x₂ : ℝ) :
  a ∈ Set.Ioo (-1) 3 →
  x₁ ≠ x₂ →
  f_derivative a x₁ = 0 →
  f_derivative a x₂ = 0 →
  g a x₁ x₂ ≥ 15 :=
sorry

end

end NUMINAMATH_CALUDE_f_derivative_condition_implies_a_range_g_minimum_value_l1664_166413


namespace NUMINAMATH_CALUDE_seventh_term_is_4374_l1664_166485

/-- A geometric sequence of positive integers with first term 6 and fifth term 486 -/
def GeometricSequence : ℕ → ℕ :=
  fun n => 6 * (486 / 6) ^ ((n - 1) / 4)

/-- The seventh term of the geometric sequence is 4374 -/
theorem seventh_term_is_4374 : GeometricSequence 7 = 4374 := by
  sorry

end NUMINAMATH_CALUDE_seventh_term_is_4374_l1664_166485


namespace NUMINAMATH_CALUDE_a_fourth_plus_reciprocal_l1664_166404

theorem a_fourth_plus_reciprocal (a : ℝ) (h : (a + 1/a)^2 = 5) : a^4 + 1/a^4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_a_fourth_plus_reciprocal_l1664_166404


namespace NUMINAMATH_CALUDE_total_wallpaper_removal_time_l1664_166451

-- Define the structure for a room
structure Room where
  name : String
  walls : Nat
  time_per_wall : List Float

-- Define the rooms
def dining_room : Room := { name := "Dining Room", walls := 3, time_per_wall := [1.5, 1.5, 1.5] }
def living_room : Room := { name := "Living Room", walls := 4, time_per_wall := [1, 1, 2.5, 2.5] }
def bedroom : Room := { name := "Bedroom", walls := 3, time_per_wall := [3, 3, 3] }
def hallway : Room := { name := "Hallway", walls := 5, time_per_wall := [4, 2, 2, 2, 2] }
def kitchen : Room := { name := "Kitchen", walls := 4, time_per_wall := [3, 1.5, 1.5, 2] }
def bathroom : Room := { name := "Bathroom", walls := 2, time_per_wall := [2, 3] }

-- Define the list of all rooms
def all_rooms : List Room := [dining_room, living_room, bedroom, hallway, kitchen, bathroom]

-- Function to calculate total time for a room
def room_time (room : Room) : Float :=
  room.time_per_wall.sum

-- Theorem: The total time to remove wallpaper from all rooms is 45.5 hours
theorem total_wallpaper_removal_time :
  (all_rooms.map room_time).sum = 45.5 := by
  sorry


end NUMINAMATH_CALUDE_total_wallpaper_removal_time_l1664_166451


namespace NUMINAMATH_CALUDE_homer_candy_crush_score_l1664_166458

theorem homer_candy_crush_score (first_try : ℕ) (second_try : ℕ) (third_try : ℕ) 
  (h1 : first_try = 400)
  (h2 : second_try < first_try)
  (h3 : third_try = 2 * second_try)
  (h4 : first_try + second_try + third_try = 1390) :
  second_try = 330 := by
  sorry

end NUMINAMATH_CALUDE_homer_candy_crush_score_l1664_166458


namespace NUMINAMATH_CALUDE_union_cardinality_of_subset_count_l1664_166473

/-- Given two finite sets A and B, if the number of sets which are subsets of A or subsets of B is 144, then the cardinality of their union is 8. -/
theorem union_cardinality_of_subset_count (A B : Finset ℕ) : 
  (Finset.powerset A).card + (Finset.powerset B).card - (Finset.powerset (A ∩ B)).card = 144 →
  (A ∪ B).card = 8 := by
  sorry

end NUMINAMATH_CALUDE_union_cardinality_of_subset_count_l1664_166473


namespace NUMINAMATH_CALUDE_min_group_size_l1664_166455

theorem min_group_size (adult_group_size children_group_size : ℕ) 
  (h1 : adult_group_size = 17)
  (h2 : children_group_size = 15)
  (h3 : ∃ n : ℕ, n > 0 ∧ n % adult_group_size = 0 ∧ n % children_group_size = 0) :
  (Nat.lcm adult_group_size children_group_size = 255) := by
  sorry

end NUMINAMATH_CALUDE_min_group_size_l1664_166455


namespace NUMINAMATH_CALUDE_negation_of_at_most_one_obtuse_l1664_166443

/-- Represents a triangle -/
structure Triangle where
  angles : Fin 3 → ℝ
  sum_180 : angles 0 + angles 1 + angles 2 = 180
  all_positive : ∀ i, angles i > 0

/-- An angle is obtuse if it's greater than 90 degrees -/
def is_obtuse (angle : ℝ) : Prop := angle > 90

/-- At most one interior angle is obtuse -/
def at_most_one_obtuse (t : Triangle) : Prop :=
  (is_obtuse (t.angles 0) → ¬is_obtuse (t.angles 1) ∧ ¬is_obtuse (t.angles 2)) ∧
  (is_obtuse (t.angles 1) → ¬is_obtuse (t.angles 0) ∧ ¬is_obtuse (t.angles 2)) ∧
  (is_obtuse (t.angles 2) → ¬is_obtuse (t.angles 0) ∧ ¬is_obtuse (t.angles 1))

/-- At least two interior angles are obtuse -/
def at_least_two_obtuse (t : Triangle) : Prop :=
  (is_obtuse (t.angles 0) ∧ is_obtuse (t.angles 1)) ∨
  (is_obtuse (t.angles 1) ∧ is_obtuse (t.angles 2)) ∨
  (is_obtuse (t.angles 0) ∧ is_obtuse (t.angles 2))

/-- The main theorem: the negation of "at most one obtuse" is "at least two obtuse" -/
theorem negation_of_at_most_one_obtuse (t : Triangle) :
  ¬(at_most_one_obtuse t) ↔ at_least_two_obtuse t :=
sorry

end NUMINAMATH_CALUDE_negation_of_at_most_one_obtuse_l1664_166443


namespace NUMINAMATH_CALUDE_tickets_sold_and_given_away_l1664_166428

theorem tickets_sold_and_given_away (initial_tickets : ℕ) (h : initial_tickets = 5760) :
  let sold_tickets := initial_tickets / 2
  let remaining_tickets := initial_tickets - sold_tickets
  let given_away_tickets := remaining_tickets / 4
  sold_tickets + given_away_tickets = 3600 :=
by sorry

end NUMINAMATH_CALUDE_tickets_sold_and_given_away_l1664_166428


namespace NUMINAMATH_CALUDE_p_minus_q_empty_iff_a_nonneg_l1664_166468

/-- The set P as defined in the problem -/
def P : Set ℝ :=
  {y | ∃ x, 1 - Real.sqrt 2 / 2 < x ∧ x < 3/2 ∧ y = -x^2 + 2*x - 1/2}

/-- The set Q as defined in the problem -/
def Q (a : ℝ) : Set ℝ :=
  {x | x^2 + (a-1)*x - a < 0}

/-- The main theorem stating the equivalence between P - Q being empty and a being in [0, +∞) -/
theorem p_minus_q_empty_iff_a_nonneg (a : ℝ) :
  P \ Q a = ∅ ↔ a ∈ Set.Ici 0 := by sorry

end NUMINAMATH_CALUDE_p_minus_q_empty_iff_a_nonneg_l1664_166468


namespace NUMINAMATH_CALUDE_lawn_mowing_l1664_166425

theorem lawn_mowing (total_time : ℝ) (worked_time : ℝ) :
  total_time = 6 →
  worked_time = 3 →
  1 - (worked_time / total_time) = (1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_lawn_mowing_l1664_166425


namespace NUMINAMATH_CALUDE_max_log_sum_l1664_166454

theorem max_log_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hsum : a + 2*b = 6) :
  ∃ (max : ℝ), max = 3 * Real.log 2 ∧ 
  ∀ (x y : ℝ), x > 0 → y > 0 → x + 2*y = 6 → Real.log x + 2 * Real.log y ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_log_sum_l1664_166454


namespace NUMINAMATH_CALUDE_distance_to_nearest_city_l1664_166429

theorem distance_to_nearest_city (d : ℝ) : 
  (¬(d ≥ 8)) ∧ (¬(d ≤ 7)) ∧ (¬(d ≤ 6)) ∧ (d ≠ 10) → 7 < d ∧ d < 8 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_nearest_city_l1664_166429


namespace NUMINAMATH_CALUDE_trigonometric_identity_l1664_166461

theorem trigonometric_identity (α : Real) 
  (h : (Real.sin (11 * Real.pi - α) - Real.cos (-α)) / Real.cos ((7 * Real.pi / 2) + α) = 3) : 
  (Real.tan α = -1/2) ∧ (Real.sin (2*α) + Real.cos (2*α) = -1/5) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l1664_166461


namespace NUMINAMATH_CALUDE_rectangle_width_l1664_166447

/-- Given a rectangle with area 1638 square inches, where ten such rectangles
    would have a total length of 390 inches, prove that its width is 42 inches. -/
theorem rectangle_width (area : ℝ) (total_length : ℝ) (h1 : area = 1638) 
    (h2 : total_length = 390) : ∃ (width : ℝ), width = 42 ∧ 
    ∃ (length : ℝ), area = length * width ∧ total_length = 10 * length :=
sorry

end NUMINAMATH_CALUDE_rectangle_width_l1664_166447


namespace NUMINAMATH_CALUDE_equation_solution_l1664_166438

theorem equation_solution : ∃ x : ℤ, 45 - (28 - (x - (15 - 18))) = 57 ∧ x = 37 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1664_166438


namespace NUMINAMATH_CALUDE_max_x0_value_l1664_166424

def max_x0 (x : Fin 1997 → ℝ) : Prop :=
  (∀ i, x i > 0) ∧
  x 0 = x 1995 ∧
  (∀ i ∈ Finset.range 1995, x i + 2 / x i = 2 * x (i + 1) + 1 / x (i + 1))

theorem max_x0_value (x : Fin 1997 → ℝ) (h : max_x0 x) :
  x 0 ≤ 2^997 ∧ ∃ y : Fin 1997 → ℝ, max_x0 y ∧ y 0 = 2^997 :=
sorry

end NUMINAMATH_CALUDE_max_x0_value_l1664_166424


namespace NUMINAMATH_CALUDE_cosine_product_bounds_l1664_166412

theorem cosine_product_bounds : 
  1/8 < Real.cos (20 * π / 180) * Real.cos (40 * π / 180) * Real.cos (70 * π / 180) ∧ 
  Real.cos (20 * π / 180) * Real.cos (40 * π / 180) * Real.cos (70 * π / 180) < 1/4 :=
by sorry

end NUMINAMATH_CALUDE_cosine_product_bounds_l1664_166412


namespace NUMINAMATH_CALUDE_pokemon_cards_distribution_l1664_166489

theorem pokemon_cards_distribution (total_cards : ℕ) (num_friends : ℕ) 
  (h1 : total_cards = 432)
  (h2 : num_friends = 4)
  (h3 : total_cards % num_friends = 0)
  (h4 : (total_cards / num_friends) % 12 = 0) :
  (total_cards / num_friends) / 12 = 9 := by
  sorry

end NUMINAMATH_CALUDE_pokemon_cards_distribution_l1664_166489

import Mathlib

namespace NUMINAMATH_CALUDE_coin_division_problem_l572_57253

theorem coin_division_problem (n : ℕ) : 
  (∀ m : ℕ, m > 0 ∧ m < n → ¬(m % 7 = 3 ∧ m % 4 = 2)) →
  n % 7 = 3 →
  n % 4 = 2 →
  n % 8 = 2 :=
by sorry

end NUMINAMATH_CALUDE_coin_division_problem_l572_57253


namespace NUMINAMATH_CALUDE_salary_increase_percentage_l572_57211

theorem salary_increase_percentage (original_salary : ℝ) (h : original_salary > 0) : 
  let decreased_salary := 0.5 * original_salary
  let final_salary := 0.75 * original_salary
  ∃ P : ℝ, decreased_salary * (1 + P) = final_salary ∧ P = 0.5 :=
by sorry

end NUMINAMATH_CALUDE_salary_increase_percentage_l572_57211


namespace NUMINAMATH_CALUDE_smallest_abs_value_not_one_l572_57224

theorem smallest_abs_value_not_one : ¬(∀ q : ℚ, q ≠ 0 → |q| ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_smallest_abs_value_not_one_l572_57224


namespace NUMINAMATH_CALUDE_f_value_at_pi_sixth_f_monotone_increasing_intervals_l572_57213

noncomputable def f (x : ℝ) := Real.sqrt 3 * Real.sin (2 * x) - 2 * (Real.cos x) ^ 2

theorem f_value_at_pi_sixth : f (π / 6) = 0 := by sorry

theorem f_monotone_increasing_intervals (k : ℤ) :
  StrictMonoOn f (Set.Icc (-(π / 6) + k * π) ((π / 3) + k * π)) := by sorry

end NUMINAMATH_CALUDE_f_value_at_pi_sixth_f_monotone_increasing_intervals_l572_57213


namespace NUMINAMATH_CALUDE_vector_operation_proof_l572_57267

def vector1 : ℝ × ℝ := (4, -5)
def vector2 : ℝ × ℝ := (-2, 8)

theorem vector_operation_proof :
  2 • (vector1 + vector2) = (4, 6) := by
  sorry

end NUMINAMATH_CALUDE_vector_operation_proof_l572_57267


namespace NUMINAMATH_CALUDE_eggs_given_by_marie_l572_57263

/-- Given that Joyce initially had 8 eggs and ended up with 14 eggs in total,
    prove that Marie gave Joyce 6 eggs. -/
theorem eggs_given_by_marie 
  (initial_eggs : ℕ) 
  (total_eggs : ℕ) 
  (h1 : initial_eggs = 8) 
  (h2 : total_eggs = 14) : 
  total_eggs - initial_eggs = 6 := by
  sorry

end NUMINAMATH_CALUDE_eggs_given_by_marie_l572_57263


namespace NUMINAMATH_CALUDE_two_abs_plus_x_nonnegative_l572_57294

theorem two_abs_plus_x_nonnegative (x : ℚ) : 2 * |x| + x ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_two_abs_plus_x_nonnegative_l572_57294


namespace NUMINAMATH_CALUDE_solve_linear_equation_l572_57217

theorem solve_linear_equation :
  ∀ x : ℚ, 3 * x + 4 = -6 * x - 11 → x = -5/3 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l572_57217


namespace NUMINAMATH_CALUDE_coin_count_l572_57203

/-- The total value of coins in cents -/
def total_value : ℕ := 240

/-- The number of nickels -/
def num_nickels : ℕ := 12

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The total number of coins -/
def total_coins : ℕ := num_nickels + (total_value - num_nickels * nickel_value) / dime_value

theorem coin_count : total_coins = 30 := by
  sorry

end NUMINAMATH_CALUDE_coin_count_l572_57203


namespace NUMINAMATH_CALUDE_twelfth_day_is_monday_l572_57283

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a month with its properties -/
structure Month where
  firstDay : DayOfWeek
  lastDay : DayOfWeek
  fridayCount : Nat
  dayCount : Nat

/-- The theorem to be proved -/
theorem twelfth_day_is_monday (m : Month) : 
  m.fridayCount = 5 ∧ 
  m.firstDay ≠ DayOfWeek.Friday ∧ 
  m.lastDay ≠ DayOfWeek.Friday ∧
  m.dayCount ≥ 28 ∧ m.dayCount ≤ 31 →
  (DayOfWeek.Monday : DayOfWeek) = 
    match (m.firstDay, 11) with
    | (DayOfWeek.Sunday, n) => DayOfWeek.Wednesday
    | (DayOfWeek.Monday, n) => DayOfWeek.Thursday
    | (DayOfWeek.Tuesday, n) => DayOfWeek.Friday
    | (DayOfWeek.Wednesday, n) => DayOfWeek.Saturday
    | (DayOfWeek.Thursday, n) => DayOfWeek.Sunday
    | (DayOfWeek.Friday, n) => DayOfWeek.Monday
    | (DayOfWeek.Saturday, n) => DayOfWeek.Tuesday
  := by sorry

end NUMINAMATH_CALUDE_twelfth_day_is_monday_l572_57283


namespace NUMINAMATH_CALUDE_aunt_gemma_feeding_times_l572_57234

/-- Calculates the number of times Aunt Gemma feeds her dogs per day -/
def feeding_times_per_day (num_dogs : ℕ) (food_per_meal : ℕ) (num_sacks : ℕ) (sack_weight : ℕ) (days : ℕ) : ℕ :=
  let total_food := num_sacks * sack_weight * 1000
  let food_per_day := total_food / days
  let food_per_dog_per_day := food_per_day / num_dogs
  food_per_dog_per_day / food_per_meal

theorem aunt_gemma_feeding_times : 
  feeding_times_per_day 4 250 2 50 50 = 2 := by sorry

end NUMINAMATH_CALUDE_aunt_gemma_feeding_times_l572_57234


namespace NUMINAMATH_CALUDE_ratio_equality_l572_57247

theorem ratio_equality (p q r u v w : ℝ) 
  (h_pos : p > 0 ∧ q > 0 ∧ r > 0 ∧ u > 0 ∧ v > 0 ∧ w > 0)
  (h_pqr : p^2 + q^2 + r^2 = 49)
  (h_uvw : u^2 + v^2 + w^2 = 64)
  (h_sum : p*u + q*v + r*w = 56) :
  (p + q + r) / (u + v + w) = 7/8 := by
sorry

end NUMINAMATH_CALUDE_ratio_equality_l572_57247


namespace NUMINAMATH_CALUDE_probability_sum_six_is_three_sixteenths_l572_57230

/-- A uniform tetrahedral die with faces numbered 1, 2, 3, 4 -/
def TetrahedralDie : Finset ℕ := {1, 2, 3, 4}

/-- The sample space of throwing the die twice -/
def SampleSpace : Finset (ℕ × ℕ) := TetrahedralDie.product TetrahedralDie

/-- The event where the sum of two throws equals 6 -/
def SumSixEvent : Finset (ℕ × ℕ) := SampleSpace.filter (fun p => p.1 + p.2 = 6)

/-- The probability of the sum being 6 when throwing the die twice -/
def probability_sum_six : ℚ := (SumSixEvent.card : ℚ) / (SampleSpace.card : ℚ)

theorem probability_sum_six_is_three_sixteenths : 
  probability_sum_six = 3 / 16 := by
  sorry

end NUMINAMATH_CALUDE_probability_sum_six_is_three_sixteenths_l572_57230


namespace NUMINAMATH_CALUDE_cos_sin_fifteen_degrees_l572_57205

theorem cos_sin_fifteen_degrees : 
  Real.cos (15 * π / 180) ^ 4 - Real.sin (15 * π / 180) ^ 4 = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_sin_fifteen_degrees_l572_57205


namespace NUMINAMATH_CALUDE_genevieve_cherry_shortage_l572_57258

/-- The amount Genevieve was short when buying cherries -/
def amount_short (cost_per_kg : ℕ) (amount_had : ℕ) (kg_bought : ℕ) : ℕ :=
  cost_per_kg * kg_bought - amount_had

/-- Proof that Genevieve was short $400 -/
theorem genevieve_cherry_shortage : amount_short 8 1600 250 = 400 := by
  sorry

end NUMINAMATH_CALUDE_genevieve_cherry_shortage_l572_57258


namespace NUMINAMATH_CALUDE_function_composition_l572_57200

theorem function_composition (f : ℝ → ℝ) (x : ℝ) : 
  (∀ y, f y = y^2 + 2*y - 1) → f (x - 1) = x^2 - 2 := by
  sorry

end NUMINAMATH_CALUDE_function_composition_l572_57200


namespace NUMINAMATH_CALUDE_symmetric_point_coordinates_l572_57260

/-- Given a line l: 2x + y - 5 = 0 and a point M(-1, 2), 
    this function returns the coordinates of the symmetric point Q with respect to l. -/
def symmetricPoint (l : ℝ → ℝ → Prop) (M : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := M
  -- Define the symmetric point Q
  let Q := (3, 4)
  Q

/-- The theorem states that the symmetric point of M(-1, 2) with respect to
    the line 2x + y - 5 = 0 is (3, 4). -/
theorem symmetric_point_coordinates :
  let l : ℝ → ℝ → Prop := fun x y ↦ 2 * x + y - 5 = 0
  let M : ℝ × ℝ := (-1, 2)
  symmetricPoint l M = (3, 4) := by
  sorry


end NUMINAMATH_CALUDE_symmetric_point_coordinates_l572_57260


namespace NUMINAMATH_CALUDE_S_pq_equation_l572_57285

/-- S(n) is the sum of squares of positive integers less than and coprime to n -/
def S (n : ℕ) : ℕ := sorry

/-- p is a prime number equal to 2^7 - 1 -/
def p : ℕ := 127

/-- q is a prime number equal to 2^5 - 1 -/
def q : ℕ := 31

/-- a is a positive integer -/
def a : ℕ := 7561

theorem S_pq_equation : 
  ∃ (b c : ℕ), 
    b < c ∧ 
    Nat.Coprime b c ∧
    S (p * q) = (p^2 * q^2 / 6) * (a - b / c) := by sorry

end NUMINAMATH_CALUDE_S_pq_equation_l572_57285


namespace NUMINAMATH_CALUDE_circle_diameter_l572_57209

theorem circle_diameter (A : ℝ) (r : ℝ) (d : ℝ) : 
  A = π * r^2 → A = 196 * π → d = 2 * r → d = 28 := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_l572_57209


namespace NUMINAMATH_CALUDE_min_value_on_negative_interval_l572_57280

/-- A function is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- The function F defined in terms of f and g -/
def F (f g : ℝ → ℝ) (a b : ℝ) (x : ℝ) : ℝ := a * f x + b * g x + 3

theorem min_value_on_negative_interval
  (f g : ℝ → ℝ) (a b : ℝ)
  (hf : IsOdd f) (hg : IsOdd g)
  (hmax : ∀ x > 0, F f g a b x ≤ 10) :
  ∀ x < 0, F f g a b x ≥ -4 :=
sorry

end NUMINAMATH_CALUDE_min_value_on_negative_interval_l572_57280


namespace NUMINAMATH_CALUDE_expression_value_l572_57240

theorem expression_value : (10 : ℝ) * 0.5 * 3 / (1/6) = 90 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l572_57240


namespace NUMINAMATH_CALUDE_partition_iff_even_l572_57257

def is_valid_partition (n : ℕ) (partition : List (List ℕ)) : Prop :=
  partition.length = n ∧
  partition.all (λ l => l.length = 4) ∧
  (partition.join.toFinset : Finset ℕ) = Finset.range (4 * n + 1) \ {0} ∧
  ∀ l ∈ partition, ∃ x ∈ l, 3 * x = (l.sum - x)

theorem partition_iff_even (n : ℕ) :
  (∃ partition : List (List ℕ), is_valid_partition n partition) ↔ Even n :=
sorry

end NUMINAMATH_CALUDE_partition_iff_even_l572_57257


namespace NUMINAMATH_CALUDE_rowing_problem_l572_57288

/-- Proves that given the conditions of the rowing problem, the downstream distance is 60 km -/
theorem rowing_problem (upstream_distance : ℝ) (upstream_time : ℝ) (downstream_time : ℝ) (stream_speed : ℝ)
  (h1 : upstream_distance = 30)
  (h2 : upstream_time = 3)
  (h3 : downstream_time = 3)
  (h4 : stream_speed = 5) :
  let boat_speed := upstream_distance / upstream_time + stream_speed
  let downstream_speed := boat_speed + stream_speed
  downstream_speed * downstream_time = 60 := by
  sorry

end NUMINAMATH_CALUDE_rowing_problem_l572_57288


namespace NUMINAMATH_CALUDE_f_has_minimum_value_neg_twelve_l572_57281

def f (x : ℝ) : ℝ := 3 * x^2 + 6 * x - 9

theorem f_has_minimum_value_neg_twelve :
  ∃ (x₀ : ℝ), ∀ (x : ℝ), f x ≥ f x₀ ∧ f x₀ = -12 := by
  sorry

end NUMINAMATH_CALUDE_f_has_minimum_value_neg_twelve_l572_57281


namespace NUMINAMATH_CALUDE_solution_satisfies_equations_l572_57218

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := (x^2 + 11) * Real.sqrt (21 + y^2) = 180
def equation2 (y z : ℝ) : Prop := (y^2 + 21) * Real.sqrt (z^2 - 33) = 100
def equation3 (z x : ℝ) : Prop := (z^2 - 33) * Real.sqrt (11 + x^2) = 96

-- Define the solution set
def solutionSet : Set (ℝ × ℝ × ℝ) :=
  {(5, 2, 7), (5, 2, -7), (5, -2, 7), (5, -2, -7),
   (-5, 2, 7), (-5, 2, -7), (-5, -2, 7), (-5, -2, -7)}

-- Theorem stating that all elements in the solution set satisfy the system of equations
theorem solution_satisfies_equations :
  ∀ (x y z : ℝ), (x, y, z) ∈ solutionSet →
    equation1 x y ∧ equation2 y z ∧ equation3 z x :=
by sorry

end NUMINAMATH_CALUDE_solution_satisfies_equations_l572_57218


namespace NUMINAMATH_CALUDE_second_to_first_l572_57243

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Predicate for a point being in the second quadrant -/
def inSecondQuadrant (p : Point) : Prop := p.x < 0 ∧ p.y > 0

/-- Predicate for a point being in the first quadrant -/
def inFirstQuadrant (p : Point) : Prop := p.x > 0 ∧ p.y > 0

/-- Theorem: If A(m,n) is in the second quadrant, then B(-m,|n|) is in the first quadrant -/
theorem second_to_first (m n : ℝ) :
  inSecondQuadrant ⟨m, n⟩ → inFirstQuadrant ⟨-m, |n|⟩ := by
  sorry

end NUMINAMATH_CALUDE_second_to_first_l572_57243


namespace NUMINAMATH_CALUDE_imaginary_iff_m_neq_zero_pure_imaginary_iff_m_eq_two_or_three_not_in_second_quadrant_l572_57229

-- Define the complex number z as a function of real number m
def z (m : ℝ) : ℂ := (m^2 - 5*m + 6) - 3*m*Complex.I

-- Theorem 1: z is an imaginary number iff m ≠ 0
theorem imaginary_iff_m_neq_zero (m : ℝ) :
  (z m).re = 0 ∧ (z m).im ≠ 0 ↔ m ≠ 0 :=
sorry

-- Theorem 2: z is a pure imaginary number iff m = 2 or m = 3
theorem pure_imaginary_iff_m_eq_two_or_three (m : ℝ) :
  (z m).re = 0 ↔ m = 2 ∨ m = 3 :=
sorry

-- Theorem 3: z cannot be in the second quadrant for any real m
theorem not_in_second_quadrant (m : ℝ) :
  ¬((z m).re < 0 ∧ (z m).im > 0) :=
sorry

end NUMINAMATH_CALUDE_imaginary_iff_m_neq_zero_pure_imaginary_iff_m_eq_two_or_three_not_in_second_quadrant_l572_57229


namespace NUMINAMATH_CALUDE_fraction_of_product_l572_57271

theorem fraction_of_product (total : ℝ) (result : ℝ) : 
  total = 5020 →
  (3/4 : ℝ) * (1/2 : ℝ) * total = (3/4 : ℝ) * (1/2 : ℝ) * 5020 →
  result = 753.0000000000001 →
  (result / ((3/4 : ℝ) * (1/2 : ℝ) * total) : ℝ) = 0.4 :=
by sorry

end NUMINAMATH_CALUDE_fraction_of_product_l572_57271


namespace NUMINAMATH_CALUDE_roots_expression_value_l572_57265

theorem roots_expression_value (x₁ x₂ : ℝ) : 
  x₁^2 - 3*x₁ + 1 = 0 → x₂^2 - 3*x₂ + 1 = 0 → 
  (x₁ + x₂) / (1 + x₁ * x₂) = 3/2 := by sorry

end NUMINAMATH_CALUDE_roots_expression_value_l572_57265


namespace NUMINAMATH_CALUDE_yellow_opposite_blue_l572_57250

-- Define the colors
inductive Color
  | Red
  | Blue
  | Orange
  | Yellow
  | Green
  | White

-- Define a square with colors on both sides
structure Square where
  front : Color
  back : Color

-- Define the cube
structure Cube where
  squares : Vector Square 6

-- Define the function to get the opposite face
def oppositeFace (c : Cube) (face : Color) : Color :=
  sorry

-- Theorem statement
theorem yellow_opposite_blue (c : Cube) :
  (∃ (s : Square), s ∈ c.squares.toList ∧ s.front = Color.Yellow) →
  oppositeFace c Color.Yellow = Color.Blue :=
sorry

end NUMINAMATH_CALUDE_yellow_opposite_blue_l572_57250


namespace NUMINAMATH_CALUDE_shopping_cart_fruit_ratio_l572_57212

theorem shopping_cart_fruit_ratio :
  ∀ (apples oranges pears : ℕ),
    oranges = 3 * apples →
    apples = (pears : ℚ) * (83333333333333333 : ℚ) / (1000000000000000000 : ℚ) →
    (pears : ℚ) / (oranges : ℚ) = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_shopping_cart_fruit_ratio_l572_57212


namespace NUMINAMATH_CALUDE_floor_tiles_l572_57277

theorem floor_tiles (n : ℕ) (h1 : n % 3 = 0) 
  (h2 : 2 * (2 * n / 3) - 1 = 49) : 
  n^2 - (n / 3)^2 = 1352 := by
  sorry

end NUMINAMATH_CALUDE_floor_tiles_l572_57277


namespace NUMINAMATH_CALUDE_triangle_nth_root_l572_57249

theorem triangle_nth_root (a b c : ℝ) (n : ℕ) (h_triangle : a + b > c ∧ b + c > a ∧ a + c > b) (h_n : n ≥ 2) :
  (a^(1/n) : ℝ) + (b^(1/n) : ℝ) > (c^(1/n) : ℝ) ∧
  (b^(1/n) : ℝ) + (c^(1/n) : ℝ) > (a^(1/n) : ℝ) ∧
  (a^(1/n) : ℝ) + (c^(1/n) : ℝ) > (b^(1/n) : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_triangle_nth_root_l572_57249


namespace NUMINAMATH_CALUDE_quadrilateral_area_l572_57231

/-- The area of a quadrilateral with vertices at (4,0), (0,5), (3,4), and (10,10) is 22.5 square units. -/
theorem quadrilateral_area : 
  let vertices : List (ℝ × ℝ) := [(4,0), (0,5), (3,4), (10,10)]
  ∃ (area : ℝ), area = 22.5 ∧ 
  area = (1/2) * abs (
    (4 * 5 + 0 * 4 + 3 * 10 + 10 * 0) - 
    (0 * 0 + 5 * 3 + 4 * 10 + 10 * 4)
  ) := by sorry

end NUMINAMATH_CALUDE_quadrilateral_area_l572_57231


namespace NUMINAMATH_CALUDE_rachels_homework_l572_57210

/-- 
Given that Rachel has 5 pages of math homework and 3 more pages of math homework 
than reading homework, prove that she has 2 pages of reading homework.
-/
theorem rachels_homework (math_pages reading_pages : ℕ) : 
  math_pages = 5 → 
  math_pages = reading_pages + 3 → 
  reading_pages = 2 := by
sorry

end NUMINAMATH_CALUDE_rachels_homework_l572_57210


namespace NUMINAMATH_CALUDE_storks_on_fence_l572_57228

/-- The number of storks that joined the birds on the fence -/
def num_storks_joined : ℕ := 4

theorem storks_on_fence :
  let initial_birds : ℕ := 2
  let additional_birds : ℕ := 5
  let total_birds : ℕ := initial_birds + additional_birds
  let bird_stork_difference : ℕ := 3
  num_storks_joined = total_birds - bird_stork_difference := by
  sorry

end NUMINAMATH_CALUDE_storks_on_fence_l572_57228


namespace NUMINAMATH_CALUDE_product_difference_equals_two_tenths_l572_57296

theorem product_difference_equals_two_tenths : 0.5 * 0.8 - 0.2 = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_product_difference_equals_two_tenths_l572_57296


namespace NUMINAMATH_CALUDE_units_digit_of_G_1000_l572_57242

-- Define G_n
def G (n : ℕ) : ℕ := 3^(3^n) + 1

-- Define a function to get the units digit
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Theorem statement
theorem units_digit_of_G_1000 : unitsDigit (G 1000) = 4 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_G_1000_l572_57242


namespace NUMINAMATH_CALUDE_christen_peeled_18_potatoes_l572_57227

/-- Represents the potato peeling scenario -/
structure PotatoPeeling where
  total_potatoes : ℕ
  homer_rate : ℕ
  christen_rate : ℕ
  christen_join_time : ℕ

/-- Calculates the number of potatoes Christen peeled -/
def potatoes_peeled_by_christen (scenario : PotatoPeeling) : ℕ :=
  sorry

/-- Theorem stating that Christen peeled 18 potatoes in the given scenario -/
theorem christen_peeled_18_potatoes :
  let scenario : PotatoPeeling := {
    total_potatoes := 50,
    homer_rate := 4,
    christen_rate := 6,
    christen_join_time := 5
  }
  potatoes_peeled_by_christen scenario = 18 := by
  sorry

end NUMINAMATH_CALUDE_christen_peeled_18_potatoes_l572_57227


namespace NUMINAMATH_CALUDE_factorization_proof_l572_57241

variables (a x y : ℝ)

theorem factorization_proof :
  (ax^2 - 7*a*x + 6*a = a*(x-6)*(x-1)) ∧
  (x*y^2 - 9*x = x*(y+3)*(y-3)) ∧
  (1 - x^2 + 2*x*y - y^2 = (1+x-y)*(1-x+y)) ∧
  (8*(x^2 - 2*y^2) - x*(7*x+y) + x*y = (x+4*y)*(x-4*y)) :=
by sorry

end NUMINAMATH_CALUDE_factorization_proof_l572_57241


namespace NUMINAMATH_CALUDE_inverse_function_theorem_l572_57204

noncomputable def g (p q r s : ℝ) (x : ℝ) : ℝ := (p * x + q) / (r * x + s)

theorem inverse_function_theorem (p q r s : ℝ) :
  p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0 ∧ s ≠ 0 →
  (∀ x, g (g p q r s x) p q r s = x) →
  p + s = 2 * q →
  p + s = 0 := by sorry

end NUMINAMATH_CALUDE_inverse_function_theorem_l572_57204


namespace NUMINAMATH_CALUDE_no_valid_formation_l572_57221

/-- Represents a rectangular formation of musicians. -/
structure Formation where
  rows : ℕ
  musicians_per_row : ℕ

/-- Checks if a formation is valid according to the given conditions. -/
def is_valid_formation (f : Formation) : Prop :=
  f.rows * f.musicians_per_row = 400 ∧
  f.musicians_per_row % 4 = 0 ∧
  10 ≤ f.musicians_per_row ∧
  f.musicians_per_row ≤ 50

/-- Represents the constraint of having a triangle formation for brass section. -/
def has_triangle_brass_formation (f : Formation) : Prop :=
  f.rows ≥ 3 ∧
  ∃ (a b c : ℕ), a < b ∧ b < c ∧ a + b + c = 100 ∧
  a % (f.musicians_per_row / 4) = 0 ∧
  b % (f.musicians_per_row / 4) = 0 ∧
  c % (f.musicians_per_row / 4) = 0

/-- The main theorem stating that no valid formation exists. -/
theorem no_valid_formation :
  ¬∃ (f : Formation), is_valid_formation f ∧ has_triangle_brass_formation f :=
sorry

end NUMINAMATH_CALUDE_no_valid_formation_l572_57221


namespace NUMINAMATH_CALUDE_alphanumeric_puzzle_l572_57292

theorem alphanumeric_puzzle :
  ∃! (A B C D E F H J K L : Nat),
    (A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧ E < 10 ∧
     F < 10 ∧ H < 10 ∧ J < 10 ∧ K < 10 ∧ L < 10) ∧
    (A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧ A ≠ H ∧ A ≠ J ∧ A ≠ K ∧ A ≠ L ∧
     B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧ B ≠ H ∧ B ≠ J ∧ B ≠ K ∧ B ≠ L ∧
     C ≠ D ∧ C ≠ E ∧ C ≠ F ∧ C ≠ H ∧ C ≠ J ∧ C ≠ K ∧ C ≠ L ∧
     D ≠ E ∧ D ≠ F ∧ D ≠ H ∧ D ≠ J ∧ D ≠ K ∧ D ≠ L ∧
     E ≠ F ∧ E ≠ H ∧ E ≠ J ∧ E ≠ K ∧ E ≠ L ∧
     F ≠ H ∧ F ≠ J ∧ F ≠ K ∧ F ≠ L ∧
     H ≠ J ∧ H ≠ K ∧ H ≠ L ∧
     J ≠ K ∧ J ≠ L ∧
     K ≠ L) ∧
    (A * B = B) ∧
    (B * C = 10 * A + C) ∧
    (C * D = 10 * B + C) ∧
    (D * E = 10 * C + H) ∧
    (E * F = 10 * D + K) ∧
    (F * H = 10 * C + J) ∧
    (H * J = 10 * K + J) ∧
    (J * K = E) ∧
    (K * L = L) ∧
    (A * L = L) ∧
    (A = 1 ∧ B = 3 ∧ C = 5 ∧ D = 7 ∧ E = 8 ∧ F = 9 ∧ H = 6 ∧ J = 4 ∧ K = 2 ∧ L = 0) :=
by sorry

end NUMINAMATH_CALUDE_alphanumeric_puzzle_l572_57292


namespace NUMINAMATH_CALUDE_second_half_duration_percentage_l572_57215

/-- Proves that the second half of a trip takes 200% longer than the first half
    given specific conditions about distance and speed. -/
theorem second_half_duration_percentage (total_distance : ℝ) (first_half_speed : ℝ) (average_speed : ℝ) :
  total_distance = 640 →
  first_half_speed = 80 →
  average_speed = 40 →
  let first_half_time := (total_distance / 2) / first_half_speed
  let total_time := total_distance / average_speed
  let second_half_time := total_time - first_half_time
  (second_half_time - first_half_time) / first_half_time * 100 = 200 := by
  sorry

end NUMINAMATH_CALUDE_second_half_duration_percentage_l572_57215


namespace NUMINAMATH_CALUDE_num_machines_proof_l572_57291

/-- The number of machines that complete a job in 6 hours, 
    given that 2 machines complete the same job in 24 hours -/
def num_machines : ℕ :=
  let time_many : ℕ := 6  -- time taken by multiple machines
  let time_two : ℕ := 24   -- time taken by 2 machines
  let machines_two : ℕ := 2  -- number of machines in second scenario
  8  -- to be proved

theorem num_machines_proof : 
  num_machines * time_many = machines_two * time_two :=
by sorry

end NUMINAMATH_CALUDE_num_machines_proof_l572_57291


namespace NUMINAMATH_CALUDE_first_terrific_tuesday_l572_57284

/-- Represents a day of the week -/
inductive DayOfWeek
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday
| Sunday

/-- Represents a date in October -/
structure OctoberDate where
  day : Nat
  dayOfWeek : DayOfWeek

/-- Definition of a Terrific Tuesday -/
def isTerrificTuesday (date : OctoberDate) : Prop :=
  date.dayOfWeek = DayOfWeek.Tuesday ∧ 
  (∃ n : Nat, n = 5 ∧ date.day = n * 7 - 4)

/-- The company's start date -/
def startDate : OctoberDate :=
  { day := 2, dayOfWeek := DayOfWeek.Monday }

/-- The number of days in October -/
def octoberDays : Nat := 31

/-- Theorem: The first Terrific Tuesday after operations begin is October 31 -/
theorem first_terrific_tuesday : 
  ∃ (date : OctoberDate), 
    date.day = 31 ∧ 
    isTerrificTuesday date ∧ 
    ∀ (earlier : OctoberDate), 
      earlier.day > startDate.day ∧ 
      earlier.day < date.day → 
      ¬isTerrificTuesday earlier :=
by sorry

end NUMINAMATH_CALUDE_first_terrific_tuesday_l572_57284


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l572_57251

theorem diophantine_equation_solution (n : ℕ+) (a b c : ℕ+) 
  (ha : a ≤ 3 * n ^ 2 + 4 * n) 
  (hb : b ≤ 3 * n ^ 2 + 4 * n) 
  (hc : c ≤ 3 * n ^ 2 + 4 * n) : 
  ∃ (x y z : ℤ), 
    (abs x ≤ 2 * n ∧ abs y ≤ 2 * n ∧ abs z ≤ 2 * n) ∧ 
    (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0) ∧
    (a * x + b * y + c * z = 0) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l572_57251


namespace NUMINAMATH_CALUDE_percentage_difference_l572_57232

theorem percentage_difference (n : ℝ) (h : n = 160) : 0.5 * n - 0.35 * n = 24 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l572_57232


namespace NUMINAMATH_CALUDE_stream_speed_l572_57262

/-- Proves that given a man's downstream speed of 18 km/h, upstream speed of 6 km/h,
    and still water speed of 12 km/h, the speed of the stream is 6 km/h. -/
theorem stream_speed (v_downstream v_upstream v_stillwater : ℝ)
    (h_downstream : v_downstream = 18)
    (h_upstream : v_upstream = 6)
    (h_stillwater : v_stillwater = 12)
    (h_downstream_eq : v_downstream = v_stillwater + (v_downstream - v_upstream) / 2)
    (h_upstream_eq : v_upstream = v_stillwater - (v_downstream - v_upstream) / 2) :
    (v_downstream - v_upstream) / 2 = 6 :=
by sorry

end NUMINAMATH_CALUDE_stream_speed_l572_57262


namespace NUMINAMATH_CALUDE_matrix_power_2019_l572_57299

def A : Matrix (Fin 2) (Fin 2) ℤ := !![1, 0; 2, 1]

theorem matrix_power_2019 :
  A ^ 2019 = !![1, 0; 4038, 1] := by sorry

end NUMINAMATH_CALUDE_matrix_power_2019_l572_57299


namespace NUMINAMATH_CALUDE_complex_equation_implies_difference_l572_57246

theorem complex_equation_implies_difference (x y : ℝ) :
  (x * Complex.I + 2 = y - Complex.I) → (x - y = -3) := by sorry

end NUMINAMATH_CALUDE_complex_equation_implies_difference_l572_57246


namespace NUMINAMATH_CALUDE_least_multiple_24_greater_450_l572_57219

theorem least_multiple_24_greater_450 : ∃ n : ℕ, 24 * n = 456 ∧ 456 > 450 ∧ ∀ m : ℕ, 24 * m > 450 → 24 * m ≥ 456 :=
sorry

end NUMINAMATH_CALUDE_least_multiple_24_greater_450_l572_57219


namespace NUMINAMATH_CALUDE_square_of_binomial_formula_l572_57269

theorem square_of_binomial_formula (a b : ℝ) :
  (a - b) * (b + a) = a^2 - b^2 ∧
  (4*a + b) * (4*a - 2*b) ≠ (2*a + b)^2 - (2*a - b)^2 ∧
  (a - 2*b) * (2*b - a) ≠ (a + b)^2 - (a - b)^2 ∧
  (2*a - b) * (-2*a + b) ≠ (a + b)^2 - (a - b)^2 :=
by sorry

#check square_of_binomial_formula

end NUMINAMATH_CALUDE_square_of_binomial_formula_l572_57269


namespace NUMINAMATH_CALUDE_box_ball_count_l572_57233

theorem box_ball_count (red_balls : ℕ) (red_prob : ℚ) (total_balls : ℕ) : 
  red_balls = 12 → red_prob = 3/5 → (red_balls : ℚ) / total_balls = red_prob → total_balls = 20 := by
  sorry

end NUMINAMATH_CALUDE_box_ball_count_l572_57233


namespace NUMINAMATH_CALUDE_solution_of_system_l572_57214

def system_of_equations (x y z : ℝ) : Prop :=
  1 / x = y + z ∧ 1 / y = z + x ∧ 1 / z = x + y

theorem solution_of_system :
  ∃ (x y z : ℝ), system_of_equations x y z ∧
    ((x = Real.sqrt 2 / 2 ∧ y = Real.sqrt 2 / 2 ∧ z = Real.sqrt 2 / 2) ∨
     (x = -Real.sqrt 2 / 2 ∧ y = -Real.sqrt 2 / 2 ∧ z = -Real.sqrt 2 / 2)) :=
by sorry

end NUMINAMATH_CALUDE_solution_of_system_l572_57214


namespace NUMINAMATH_CALUDE_largest_divisor_of_expression_l572_57236

theorem largest_divisor_of_expression (x : ℤ) (h : Odd x) :
  (∃ (k : ℤ), (8*x + 4) * (8*x + 8) * (4*x + 2) = 384 * k) ∧
  (∀ (d : ℤ), d > 384 → ¬(∀ (y : ℤ), Odd y → ∃ (m : ℤ), (8*y + 4) * (8*y + 8) * (4*y + 2) = d * m)) :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_expression_l572_57236


namespace NUMINAMATH_CALUDE_three_circles_cross_ratio_invariance_l572_57245

-- Define a circle in 2D space
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define a line by two points
structure Line where
  p1 : Point
  p2 : Point

-- Define the cross-ratio of four points on a line
def cross_ratio (p1 p2 p3 p4 : Point) : ℝ := sorry

-- Define a function to check if a point is on a circle
def point_on_circle (p : Point) (c : Circle) : Prop := sorry

-- Define a function to check if a point is on a line
def point_on_line (p : Point) (l : Line) : Prop := sorry

-- Define a function to find the intersection points of a line and a circle
def line_circle_intersection (l : Line) (c : Circle) : Set Point := sorry

theorem three_circles_cross_ratio_invariance 
  (c1 c2 c3 : Circle) 
  (A B : Point) 
  (h1 : point_on_circle A c1 ∧ point_on_circle A c2 ∧ point_on_circle A c3)
  (h2 : point_on_circle B c1 ∧ point_on_circle B c2 ∧ point_on_circle B c3)
  (h3 : A ≠ B) :
  ∀ (l1 l2 : Line), 
  (point_on_line A l1 ∧ point_on_line A l2) →
  ∃ (P1 Q1 R1 P2 Q2 R2 : Point),
  (P1 ∈ line_circle_intersection l1 c1 ∧ 
   Q1 ∈ line_circle_intersection l1 c2 ∧ 
   R1 ∈ line_circle_intersection l1 c3 ∧
   P2 ∈ line_circle_intersection l2 c1 ∧ 
   Q2 ∈ line_circle_intersection l2 c2 ∧ 
   R2 ∈ line_circle_intersection l2 c3) →
  cross_ratio A P1 Q1 R1 = cross_ratio A P2 Q2 R2 :=
sorry

end NUMINAMATH_CALUDE_three_circles_cross_ratio_invariance_l572_57245


namespace NUMINAMATH_CALUDE_max_area_constrained_rectangle_l572_57252

/-- Represents a rectangular garden with given length and width. -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangle. -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

/-- Calculates the area of a rectangle. -/
def area (r : Rectangle) : ℝ := r.length * r.width

/-- Checks if a rectangle satisfies the given constraints. -/
def satisfiesConstraints (r : Rectangle) : Prop :=
  perimeter r = 400 ∧ r.length ≥ 100 ∧ r.width ≥ 50

/-- States that the maximum area of a constrained rectangle is 7500. -/
theorem max_area_constrained_rectangle :
  ∀ r : Rectangle, satisfiesConstraints r → area r ≤ 7500 :=
by sorry

end NUMINAMATH_CALUDE_max_area_constrained_rectangle_l572_57252


namespace NUMINAMATH_CALUDE_min_fraction_value_l572_57274

/-- A function that checks if a natural number contains the digit string "11235" -/
def contains_11235 (n : ℕ) : Prop := sorry

/-- The main theorem -/
theorem min_fraction_value (N k : ℕ) (h1 : N > 0) (h2 : k > 0) (h3 : contains_11235 N) (h4 : 10^k > N) :
  (∀ N' k' : ℕ, N' > 0 → k' > 0 → contains_11235 N' → 10^k' > N' →
    (10^k' - 1) / Nat.gcd (10^k' - 1) N' ≥ 89) ∧
  (∃ N' k' : ℕ, N' > 0 ∧ k' > 0 ∧ contains_11235 N' ∧ 10^k' > N' ∧
    (10^k' - 1) / Nat.gcd (10^k' - 1) N' = 89) :=
by sorry

end NUMINAMATH_CALUDE_min_fraction_value_l572_57274


namespace NUMINAMATH_CALUDE_shortest_major_axis_ellipse_l572_57222

-- Define the line l
def line_l (x y : ℝ) : Prop := y = x + 2

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := 12 * x^2 - 4 * y^2 = 3

-- Define a general ellipse
def ellipse (a b x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the condition for an ellipse to share foci with the hyperbola
def shared_foci (a b : ℝ) : Prop := a^2 - b^2 = 1

-- Define the tangency condition
def is_tangent (a b : ℝ) : Prop := ∃ x y : ℝ, line_l x y ∧ ellipse a b x y

-- Theorem statement
theorem shortest_major_axis_ellipse :
  ∀ a b : ℝ, a > 0 → b > 0 →
  shared_foci a b →
  is_tangent a b →
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → shared_foci a' b' → is_tangent a' b' → a ≤ a') →
  a^2 = 5 ∧ b^2 = 4 :=
sorry

end NUMINAMATH_CALUDE_shortest_major_axis_ellipse_l572_57222


namespace NUMINAMATH_CALUDE_log_equation_solution_l572_57297

theorem log_equation_solution : 
  ∃! x : ℝ, x > 0 ∧ 2 * Real.log x = Real.log 192 + Real.log 3 - Real.log 4 :=
by
  -- The unique solution is x = 12
  use 12
  constructor
  · -- Prove that x = 12 satisfies the equation
    sorry
  · -- Prove uniqueness
    sorry

#check log_equation_solution

end NUMINAMATH_CALUDE_log_equation_solution_l572_57297


namespace NUMINAMATH_CALUDE_sin_sixty_degrees_l572_57256

theorem sin_sixty_degrees : Real.sin (π / 3) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_sixty_degrees_l572_57256


namespace NUMINAMATH_CALUDE_units_digit_pow_two_cycle_units_digit_pow_two_2015_l572_57237

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_pow_two_cycle (n : ℕ) (h : n ≥ 1) : 
  units_digit (2^n) = units_digit (2^((n - 1) % 4 + 1)) :=
sorry

theorem units_digit_pow_two_2015 : units_digit (2^2015) = 8 :=
sorry

end NUMINAMATH_CALUDE_units_digit_pow_two_cycle_units_digit_pow_two_2015_l572_57237


namespace NUMINAMATH_CALUDE_odd_function_negative_x_l572_57207

/-- A function f is odd if f(-x) = -f(x) for all x -/
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_negative_x
  (f : ℝ → ℝ)
  (odd : OddFunction f)
  (pos : ∀ x > 0, f x = x * (1 - x)) :
  ∀ x < 0, f x = x * (1 + x) := by
sorry

end NUMINAMATH_CALUDE_odd_function_negative_x_l572_57207


namespace NUMINAMATH_CALUDE_product_abcd_l572_57223

theorem product_abcd (a b c d : ℚ) 
  (eq1 : 4*a + 5*b + 7*c + 9*d = 82)
  (eq2 : d + c = 2*b)
  (eq3 : 2*b + 2*c = 3*a)
  (eq4 : c - 2 = d) :
  a * b * c * d = 276264960 / 14747943 := by
sorry

end NUMINAMATH_CALUDE_product_abcd_l572_57223


namespace NUMINAMATH_CALUDE_integer_linear_combination_sqrt2_sqrt3_l572_57235

theorem integer_linear_combination_sqrt2_sqrt3 (a b c : ℤ) :
  a * Real.sqrt 2 + b * Real.sqrt 3 + c = 0 → a = 0 ∧ b = 0 ∧ c = 0 := by
sorry

end NUMINAMATH_CALUDE_integer_linear_combination_sqrt2_sqrt3_l572_57235


namespace NUMINAMATH_CALUDE_xy_expression_value_l572_57270

theorem xy_expression_value (x y m : ℝ) 
  (eq1 : x + y + m = 6) 
  (eq2 : 3 * x - y + m = 4) : 
  -2 * x * y + 1 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_xy_expression_value_l572_57270


namespace NUMINAMATH_CALUDE_inequality_range_l572_57239

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, 0 < x ∧ x < 4 → x^2 - 2*x + 1 - a^2 < 0) ↔ 
  (a > 3 ∨ a < -3) := by
sorry

end NUMINAMATH_CALUDE_inequality_range_l572_57239


namespace NUMINAMATH_CALUDE_no_integer_cube_equals_3n2_plus_3n_plus_7_l572_57295

theorem no_integer_cube_equals_3n2_plus_3n_plus_7 :
  ¬ ∃ (x n : ℤ), x^3 = 3*n^2 + 3*n + 7 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_cube_equals_3n2_plus_3n_plus_7_l572_57295


namespace NUMINAMATH_CALUDE_lilies_bought_l572_57244

/-- Given the cost of roses and lilies, the total paid, and the change received,
    prove that the number of lilies bought is 6. -/
theorem lilies_bought (rose_cost : ℕ) (lily_cost : ℕ) (total_paid : ℕ) (change : ℕ) : 
  rose_cost = 3000 →
  lily_cost = 2800 →
  total_paid = 25000 →
  change = 2200 →
  (total_paid - change - 2 * rose_cost) / lily_cost = 6 := by
  sorry

end NUMINAMATH_CALUDE_lilies_bought_l572_57244


namespace NUMINAMATH_CALUDE_greatest_divisor_four_consecutive_integers_l572_57216

theorem greatest_divisor_four_consecutive_integers :
  ∀ n : ℕ, n > 0 →
  ∃ m : ℕ, m > 0 ∧ m ∣ (n * (n + 1) * (n + 2) * (n + 3)) ∧
  ∀ k : ℕ, k > m → ¬(∀ i : ℕ, i > 0 → k ∣ (i * (i + 1) * (i + 2) * (i + 3))) →
  m = 24 :=
by sorry

end NUMINAMATH_CALUDE_greatest_divisor_four_consecutive_integers_l572_57216


namespace NUMINAMATH_CALUDE_particular_innings_number_l572_57276

/-- Represents the statistics of a cricket player -/
structure CricketStats where
  innings : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the new average after adding runs -/
def newAverage (stats : CricketStats) (newRuns : ℕ) : ℚ :=
  (stats.totalRuns + newRuns) / (stats.innings + 1)

theorem particular_innings_number
  (initialStats : CricketStats)
  (h1 : initialStats.innings = 16)
  (h2 : newAverage initialStats 112 = initialStats.average + 6)
  (h3 : newAverage initialStats 112 = 16) :
  initialStats.innings + 1 = 17 := by
  sorry

end NUMINAMATH_CALUDE_particular_innings_number_l572_57276


namespace NUMINAMATH_CALUDE_quadrilateral_symmetry_theorem_l572_57208

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : Point)

-- Define a symmetry operation
def symmetryOperation (Q : Quadrilateral) : Quadrilateral := sorry

-- Define a cyclic quadrilateral
def isCyclic (Q : Quadrilateral) : Prop := sorry

-- Define a permissible quadrilateral
def isPermissible (Q : Quadrilateral) : Prop := sorry

-- Define equality of quadrilaterals
def equalQuadrilaterals (Q1 Q2 : Quadrilateral) : Prop := sorry

-- Define the application of n symmetry operations
def applyNOperations (Q : Quadrilateral) (n : ℕ) : Quadrilateral := sorry

theorem quadrilateral_symmetry_theorem (Q : Quadrilateral) :
  (isCyclic Q → equalQuadrilaterals Q (applyNOperations Q 3)) ∧
  (isPermissible Q → equalQuadrilaterals Q (applyNOperations Q 6)) := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_symmetry_theorem_l572_57208


namespace NUMINAMATH_CALUDE_science_club_enrollment_l572_57226

theorem science_club_enrollment (total : ℕ) (math physics chem : ℕ) 
  (math_physics math_chem physics_chem : ℕ) (all_three : ℕ) 
  (h_total : total = 150)
  (h_math : math = 90)
  (h_physics : physics = 70)
  (h_chem : chem = 40)
  (h_math_physics : math_physics = 20)
  (h_math_chem : math_chem = 15)
  (h_physics_chem : physics_chem = 10)
  (h_all_three : all_three = 5) :
  total - (math + physics + chem - math_physics - math_chem - physics_chem + all_three) = 5 := by
  sorry

end NUMINAMATH_CALUDE_science_club_enrollment_l572_57226


namespace NUMINAMATH_CALUDE_sequence_sum_theorem_l572_57289

theorem sequence_sum_theorem (a : ℕ+ → ℚ) (S : ℕ+ → ℚ) 
  (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ+, S n = n^2 * a n) :
  (∀ n : ℕ+, S n = 2 * n / (n + 1)) ∧
  (∀ n : ℕ+, a n = 2 / (n * (n + 1))) := by
sorry

end NUMINAMATH_CALUDE_sequence_sum_theorem_l572_57289


namespace NUMINAMATH_CALUDE_expected_red_balls_l572_57220

/-- The expected number of red balls selected when choosing 2 balls from a box containing 4 black, 3 red, and 2 white balls -/
theorem expected_red_balls (total_balls : ℕ) (red_balls : ℕ) (selected_balls : ℕ) 
  (h_total : total_balls = 9)
  (h_red : red_balls = 3)
  (h_selected : selected_balls = 2) :
  (red_balls : ℚ) * selected_balls / total_balls = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_expected_red_balls_l572_57220


namespace NUMINAMATH_CALUDE_sports_club_membership_l572_57272

theorem sports_club_membership (total : ℕ) (badminton : ℕ) (tennis : ℕ) (neither : ℕ) :
  total = 80 →
  badminton = 48 →
  tennis = 46 →
  neither = 7 →
  badminton + tennis - (total - neither) = 21 :=
by
  sorry

end NUMINAMATH_CALUDE_sports_club_membership_l572_57272


namespace NUMINAMATH_CALUDE_max_intersections_circle_quadrilateral_l572_57201

/-- A circle in a 2D plane -/
structure Circle where
  -- We don't need to define the specifics of a circle for this problem

/-- A quadrilateral in a 2D plane -/
structure Quadrilateral where
  -- We don't need to define the specifics of a quadrilateral for this problem

/-- The number of sides in a quadrilateral -/
def quadrilateral_sides : ℕ := 4

/-- The maximum number of intersections between a line segment and a circle -/
def max_intersections_line_circle : ℕ := 2

/-- Theorem: The maximum number of intersection points between a circle and a quadrilateral is 8 -/
theorem max_intersections_circle_quadrilateral (c : Circle) (q : Quadrilateral) :
  (quadrilateral_sides * max_intersections_line_circle) = 8 := by
  sorry

#check max_intersections_circle_quadrilateral

end NUMINAMATH_CALUDE_max_intersections_circle_quadrilateral_l572_57201


namespace NUMINAMATH_CALUDE_centroid_satisfies_conditions_l572_57287

/-- A point in a 2D plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A triangle defined by three points -/
structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)

/-- Check if a point is inside a triangle -/
def isInside (p : Point) (t : Triangle) : Prop := sorry

/-- Check if a point is on a line segment between two other points -/
def isOnSegment (p : Point) (a : Point) (b : Point) : Prop := sorry

/-- Check if two line segments are parallel -/
def areParallel (p1 : Point) (p2 : Point) (q1 : Point) (q2 : Point) : Prop := sorry

/-- Calculate the area of a triangle given three points -/
def triangleArea (p1 : Point) (p2 : Point) (p3 : Point) : ℝ := sorry

/-- The centroid of a triangle -/
def centroid (t : Triangle) : Point := sorry

theorem centroid_satisfies_conditions (t : Triangle) :
  ∃ (O L M N : Point),
    isInside O t ∧
    isOnSegment L t.A t.B ∧
    isOnSegment M t.B t.C ∧
    isOnSegment N t.C t.A ∧
    areParallel O L t.B t.C ∧
    areParallel O M t.A t.C ∧
    areParallel O N t.A t.B ∧
    triangleArea O t.B L = triangleArea O t.C M ∧
    triangleArea O t.C M = triangleArea O t.A N ∧
    O = centroid t :=
  sorry

end NUMINAMATH_CALUDE_centroid_satisfies_conditions_l572_57287


namespace NUMINAMATH_CALUDE_factorization_equality_l572_57298

theorem factorization_equality (m n : ℝ) : 
  2*m^2 - m*n + 2*m + n - n^2 = (2*m + n)*(m - n + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l572_57298


namespace NUMINAMATH_CALUDE_qiqi_mistake_xiaoming_jiajia_relation_l572_57261

-- Define the polynomials A and B
def A (x : ℝ) : ℝ := -x^2 + 4*x
def B (x : ℝ) : ℝ := 2*x^2 + 5*x - 4

-- Define Jiajia's correct answer
def correct_answer : ℝ := -18

-- Define Qiqi's mistaken coefficient
def qiqi_coefficient : ℝ := 3

-- Define the value of x
def x_value : ℝ := -2

-- Theorem 1: Qiqi's mistaken coefficient
theorem qiqi_mistake :
  A x_value + (2*x_value^2 + qiqi_coefficient*x_value - 4) = correct_answer + 16 :=
sorry

-- Theorem 2: Relationship between Xiaoming's and Jiajia's results
theorem xiaoming_jiajia_relation :
  A (-x_value) + B (-x_value) = -(A x_value + B x_value) :=
sorry

end NUMINAMATH_CALUDE_qiqi_mistake_xiaoming_jiajia_relation_l572_57261


namespace NUMINAMATH_CALUDE_roses_recipients_l572_57206

/-- Given Ricky's initial number of roses, the number of roses stolen, and the number of roses
    per person, calculate the number of people who will receive roses. -/
def number_of_recipients (initial_roses : ℕ) (stolen_roses : ℕ) (roses_per_person : ℕ) : ℕ :=
  (initial_roses - stolen_roses) / roses_per_person

/-- Theorem stating that given the specific values in the problem, 
    the number of people who will receive roses is 9. -/
theorem roses_recipients : 
  number_of_recipients 40 4 4 = 9 := by
  sorry

end NUMINAMATH_CALUDE_roses_recipients_l572_57206


namespace NUMINAMATH_CALUDE_overlook_distance_proof_l572_57255

/-- The distance to Mount Overlook in miles -/
def distance_to_overlook : ℝ := 12

/-- Jeannie's hiking speed to Mount Overlook in miles per hour -/
def speed_to_overlook : ℝ := 4

/-- Jeannie's hiking speed from Mount Overlook in miles per hour -/
def speed_from_overlook : ℝ := 6

/-- Total time of the hike in hours -/
def total_time : ℝ := 5

theorem overlook_distance_proof :
  distance_to_overlook = 12 ∧
  (distance_to_overlook / speed_to_overlook + distance_to_overlook / speed_from_overlook = total_time) :=
by sorry

end NUMINAMATH_CALUDE_overlook_distance_proof_l572_57255


namespace NUMINAMATH_CALUDE_max_prob_with_highest_prob_second_l572_57286

/-- Represents a chess player's probability of winning against an opponent -/
structure PlayerProb where
  prob : ℝ
  pos : prob > 0

/-- Represents the probabilities of winning against three players -/
structure ThreePlayerProbs where
  p₁ : PlayerProb
  p₂ : PlayerProb
  p₃ : PlayerProb
  p₃_gt_p₂ : p₃.prob > p₂.prob
  p₂_gt_p₁ : p₂.prob > p₁.prob

/-- Calculates the probability of winning two consecutive games given the order of opponents -/
def prob_two_consecutive_wins (probs : ThreePlayerProbs) (second_player : ℕ) : ℝ :=
  match second_player with
  | 1 => 2 * (probs.p₁.prob * (probs.p₂.prob + probs.p₃.prob) - 2 * probs.p₁.prob * probs.p₂.prob * probs.p₃.prob)
  | 2 => 2 * (probs.p₂.prob * (probs.p₁.prob + probs.p₃.prob) - 2 * probs.p₁.prob * probs.p₂.prob * probs.p₃.prob)
  | _ => 2 * (probs.p₁.prob * probs.p₃.prob + probs.p₂.prob * probs.p₃.prob - 2 * probs.p₁.prob * probs.p₂.prob * probs.p₃.prob)

theorem max_prob_with_highest_prob_second (probs : ThreePlayerProbs) :
  ∀ i, prob_two_consecutive_wins probs 3 ≥ prob_two_consecutive_wins probs i :=
sorry

end NUMINAMATH_CALUDE_max_prob_with_highest_prob_second_l572_57286


namespace NUMINAMATH_CALUDE_mr_A_net_gain_l572_57275

def initial_cash_A : ℕ := 20000
def initial_house_value : ℕ := 20000
def initial_car_value : ℕ := 5000
def initial_cash_B : ℕ := 25000

def house_sale_price : ℕ := 21000
def car_sale_price : ℕ := 4500

def house_buyback_price : ℕ := 19000
def car_depreciation_rate : ℚ := 1/10
def car_buyback_price : ℕ := 4050

theorem mr_A_net_gain :
  let first_transaction_cash_A := initial_cash_A + house_sale_price + car_sale_price
  let second_transaction_cash_A := first_transaction_cash_A - house_buyback_price - car_buyback_price
  second_transaction_cash_A - initial_cash_A = 2000 := by sorry

end NUMINAMATH_CALUDE_mr_A_net_gain_l572_57275


namespace NUMINAMATH_CALUDE_negation_of_implication_l572_57293

theorem negation_of_implication (x : ℝ) :
  ¬(x > 2 → x^2 - 3*x + 2 > 0) ↔ (x ≤ 2 → x^2 - 3*x + 2 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l572_57293


namespace NUMINAMATH_CALUDE_triangle_area_l572_57268

/-- Given a triangle with perimeter 36 and inradius 2.5, its area is 45 -/
theorem triangle_area (p : ℝ) (r : ℝ) (area : ℝ) 
    (h1 : p = 36) (h2 : r = 2.5) (h3 : area = r * p / 2) : area = 45 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l572_57268


namespace NUMINAMATH_CALUDE_cashew_price_satisfies_conditions_l572_57248

/-- The price per pound of cashews that satisfies the mixture conditions -/
def cashew_price : ℝ := 6.75

/-- The total weight of the mixture in pounds -/
def total_mixture : ℝ := 50

/-- The selling price of the mixture per pound -/
def mixture_price : ℝ := 5.70

/-- The weight of cashews used in the mixture in pounds -/
def cashew_weight : ℝ := 20

/-- The price of Brazil nuts per pound -/
def brazil_nut_price : ℝ := 5.00

/-- Theorem stating that the calculated cashew price satisfies the mixture conditions -/
theorem cashew_price_satisfies_conditions : 
  cashew_weight * cashew_price + (total_mixture - cashew_weight) * brazil_nut_price = 
  total_mixture * mixture_price :=
sorry

end NUMINAMATH_CALUDE_cashew_price_satisfies_conditions_l572_57248


namespace NUMINAMATH_CALUDE_angle_SPQ_is_20_degrees_l572_57225

/-- A geometric configuration with specific angle measures -/
structure GeometricConfiguration where
  -- Point Q lies on PR and Point S lies on QT (implied by the existence of these angles)
  angle_QST : ℝ  -- Measure of angle QST
  angle_TSP : ℝ  -- Measure of angle TSP
  angle_RQS : ℝ  -- Measure of angle RQS

/-- Theorem stating that under the given conditions, angle SPQ measures 20 degrees -/
theorem angle_SPQ_is_20_degrees (config : GeometricConfiguration)
  (h1 : config.angle_QST = 180)
  (h2 : config.angle_TSP = 50)
  (h3 : config.angle_RQS = 150) :
  config.angle_TSP + 20 = config.angle_RQS :=
sorry

end NUMINAMATH_CALUDE_angle_SPQ_is_20_degrees_l572_57225


namespace NUMINAMATH_CALUDE_probability_sum_binary_digits_not_exceed_eight_l572_57273

/-- The maximum number in the set of possible values -/
def max_num : ℕ := 2016

/-- Function to calculate the sum of binary digits of a natural number -/
def sum_binary_digits (n : ℕ) : ℕ := sorry

/-- The count of numbers from 1 to max_num with sum of binary digits not exceeding 8 -/
def count_valid_numbers : ℕ := sorry

/-- Theorem stating the probability of a randomly chosen number from 1 to max_num 
    having a sum of binary digits not exceeding 8 -/
theorem probability_sum_binary_digits_not_exceed_eight :
  (count_valid_numbers : ℚ) / max_num = 655 / 672 := by sorry

end NUMINAMATH_CALUDE_probability_sum_binary_digits_not_exceed_eight_l572_57273


namespace NUMINAMATH_CALUDE_point_B_coordinates_l572_57282

/-- Given two points A and B in a 2D plane, this theorem proves that
    if the vector from A to B is (3, 4) and A has coordinates (-2, -1),
    then B has coordinates (1, 3). -/
theorem point_B_coordinates
  (A B : ℝ × ℝ)
  (h1 : A = (-2, -1))
  (h2 : B.1 - A.1 = 3 ∧ B.2 - A.2 = 4) :
  B = (1, 3) := by
  sorry

end NUMINAMATH_CALUDE_point_B_coordinates_l572_57282


namespace NUMINAMATH_CALUDE_rhombus_area_l572_57259

/-- Given a rhombus with perimeter 48 and sum of diagonals 26, its area is 25 -/
theorem rhombus_area (perimeter : ℝ) (diagonal_sum : ℝ) (area : ℝ) : 
  perimeter = 48 → diagonal_sum = 26 → area = 25 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_l572_57259


namespace NUMINAMATH_CALUDE_union_of_P_and_Q_l572_57254

def P : Set ℕ := {1, 2}
def Q : Set ℕ := {y | ∃ a ∈ P, y = 2*a - 1}

theorem union_of_P_and_Q : P ∪ Q = {1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_union_of_P_and_Q_l572_57254


namespace NUMINAMATH_CALUDE_line_tangent_to_parabola_l572_57290

/-- A line is tangent to a parabola if and only if the discriminant of the resulting quadratic equation is zero -/
axiom tangent_condition (a b c : ℝ) : 
  b^2 - 4*a*c = 0 ↔ ∃ k, (∀ x y, a*y^2 + b*y + c = 0 ∧ y^2 = 16*x ∧ 6*x - 4*y + k = 0)

/-- The value of k for which the line 6x - 4y + k = 0 is tangent to the parabola y^2 = 16x -/
theorem line_tangent_to_parabola : 
  ∃! k, ∀ x y, (y^2 = 16*x ∧ 6*x - 4*y + k = 0) → k = 32/3 := by
  sorry

end NUMINAMATH_CALUDE_line_tangent_to_parabola_l572_57290


namespace NUMINAMATH_CALUDE_parabola_tangent_line_l572_57278

theorem parabola_tangent_line (b c : ℝ) : 
  (∀ x, x^2 + b*x + c = 2*x → x = 2) ∧ 
  (2*2 = 2^2 + b*2 + c) ∧
  (∀ x, 2*x + b = 2) →
  b = -2 ∧ c = 4 := by
sorry

end NUMINAMATH_CALUDE_parabola_tangent_line_l572_57278


namespace NUMINAMATH_CALUDE_soccer_ball_white_patches_l572_57238

/-- Represents a soccer ball with hexagonal and pentagonal patches -/
structure SoccerBall where
  total_patches : ℕ
  white_patches : ℕ
  black_patches : ℕ
  white_black_borders : ℕ

/-- Conditions for a valid soccer ball configuration -/
def is_valid_soccer_ball (ball : SoccerBall) : Prop :=
  ball.total_patches = 32 ∧
  ball.white_patches + ball.black_patches = ball.total_patches ∧
  ball.white_black_borders = 3 * ball.white_patches ∧
  ball.white_black_borders = 5 * ball.black_patches

/-- Theorem stating that a valid soccer ball has 20 white patches -/
theorem soccer_ball_white_patches (ball : SoccerBall) 
  (h : is_valid_soccer_ball ball) : ball.white_patches = 20 := by
  sorry

#check soccer_ball_white_patches

end NUMINAMATH_CALUDE_soccer_ball_white_patches_l572_57238


namespace NUMINAMATH_CALUDE_student_count_l572_57266

/-- In a class, given a student who is both the 30th best and 30th worst, 
    the total number of students in the class is 59. -/
theorem student_count (n : ℕ) (rob : ℕ) 
  (h1 : rob = 30)  -- Rob's position from the top
  (h2 : rob = n - 29) : -- Rob's position from the bottom
  n = 59 := by
  sorry

end NUMINAMATH_CALUDE_student_count_l572_57266


namespace NUMINAMATH_CALUDE_number_divided_by_expression_equals_one_l572_57202

theorem number_divided_by_expression_equals_one :
  ∃ x : ℝ, x / (5 + 3 / 0.75) = 1 ∧ x = 9 := by
  sorry

end NUMINAMATH_CALUDE_number_divided_by_expression_equals_one_l572_57202


namespace NUMINAMATH_CALUDE_mary_score_unique_l572_57279

/-- Represents the scoring system for the AHSME -/
structure AHSMEScore where
  correct : ℕ
  wrong : ℕ
  score : ℕ
  total_problems : ℕ := 30
  score_formula : score = 35 + 5 * correct - wrong
  valid_answers : correct + wrong ≤ total_problems

/-- Represents the condition for John to uniquely determine Mary's score -/
def uniquely_determinable (s : AHSMEScore) : Prop :=
  ∀ s' : AHSMEScore, s'.score > 90 → s'.score ≤ s.score → s' = s

/-- Mary's AHSME score satisfies all conditions and is uniquely determinable -/
theorem mary_score_unique : 
  ∃! s : AHSMEScore, s.score > 90 ∧ uniquely_determinable s ∧ 
  s.correct = 12 ∧ s.wrong = 0 ∧ s.score = 95 := by
  sorry


end NUMINAMATH_CALUDE_mary_score_unique_l572_57279


namespace NUMINAMATH_CALUDE_t_shape_perimeter_l572_57264

/-- A figure consisting of six identical squares arranged in a "T" shape -/
structure TShapeFigure where
  /-- The side length of each square in the figure -/
  square_side : ℝ
  /-- The total area of the figure is 150 cm² -/
  total_area_eq : 6 * square_side^2 = 150

/-- The perimeter of the T-shaped figure -/
def perimeter (fig : TShapeFigure) : ℝ :=
  9 * fig.square_side

/-- Theorem stating that the perimeter of the T-shaped figure is 45 cm -/
theorem t_shape_perimeter (fig : TShapeFigure) : perimeter fig = 45 := by
  sorry

end NUMINAMATH_CALUDE_t_shape_perimeter_l572_57264

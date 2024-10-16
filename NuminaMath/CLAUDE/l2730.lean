import Mathlib

namespace NUMINAMATH_CALUDE_pure_imaginary_equation_l2730_273037

theorem pure_imaginary_equation (z : ℂ) (b : ℝ) : 
  (∃ (a : ℝ), z = a * Complex.I) → 
  (2 - Complex.I) * z = 4 - b * Complex.I → 
  b = -8 := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_equation_l2730_273037


namespace NUMINAMATH_CALUDE_negative_two_squared_l2730_273059

theorem negative_two_squared : -2^2 = -4 := by
  sorry

end NUMINAMATH_CALUDE_negative_two_squared_l2730_273059


namespace NUMINAMATH_CALUDE_square_distance_l2730_273080

theorem square_distance (small_perimeter : ℝ) (large_area : ℝ) :
  small_perimeter = 8 →
  large_area = 36 →
  let small_side := small_perimeter / 4
  let large_side := Real.sqrt large_area
  let leg1 := large_side
  let leg2 := large_side - 2 * small_side
  Real.sqrt (leg1^2 + leg2^2) = 2 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_square_distance_l2730_273080


namespace NUMINAMATH_CALUDE_bowls_lost_l2730_273007

/-- Proves that the number of lost bowls is 26 given the problem conditions --/
theorem bowls_lost (total_bowls : ℕ) (fee : ℕ) (safe_payment : ℕ) (penalty : ℕ) 
  (broken_bowls : ℕ) (total_payment : ℕ) :
  total_bowls = 638 →
  fee = 100 →
  safe_payment = 3 →
  penalty = 4 →
  broken_bowls = 15 →
  total_payment = 1825 →
  ∃ (lost_bowls : ℕ), 
    fee + safe_payment * (total_bowls - lost_bowls - broken_bowls) - 
    penalty * (lost_bowls + broken_bowls) = total_payment ∧
    lost_bowls = 26 :=
by sorry

end NUMINAMATH_CALUDE_bowls_lost_l2730_273007


namespace NUMINAMATH_CALUDE_first_equation_is_double_root_second_equation_values_l2730_273096

/-- Definition of a double root equation -/
def is_double_root_equation (a b c : ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 ∧ y = 2 * x

/-- The equation x^2 - 3x + 2 = 0 is a double root equation -/
theorem first_equation_is_double_root : is_double_root_equation 1 (-3) 2 :=
sorry

/-- For ax^2 + bx - 6 = 0, if it's a double root equation with one root as 2,
    then a and b have specific values -/
theorem second_equation_values (a b : ℝ) :
  is_double_root_equation a b (-6) ∧ (∃ x : ℝ, a * x^2 + b * x - 6 = 0 ∧ x = 2) →
  ((a = -3/4 ∧ b = 9/2) ∨ (a = -3 ∧ b = 9)) :=
sorry

end NUMINAMATH_CALUDE_first_equation_is_double_root_second_equation_values_l2730_273096


namespace NUMINAMATH_CALUDE_f_properties_l2730_273070

noncomputable def f (x : ℝ) : ℝ := Real.cos (x - Real.pi / 3) - Real.sin (Real.pi / 2 - x)

theorem f_properties (α : ℝ) (h1 : 0 < α) (h2 : α < Real.pi / 2) (h3 : f (α + Real.pi / 6) = 3 / 5) :
  (∃ T : ℝ, T > 0 ∧ (∀ x : ℝ, f (x + T) = f x) ∧ (∀ S : ℝ, S > 0 ∧ (∀ x : ℝ, f (x + S) = f x) → T ≤ S)) ∧
  f (2 * α) = (24 * Real.sqrt 3 - 7) / 50 :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l2730_273070


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l2730_273024

-- Define the universal set U
def U : Finset Nat := {1, 2, 3, 4, 5}

-- Define set A
def A : Finset Nat := {1, 5}

-- Define set B
def B : Finset Nat := {2, 4}

-- Theorem statement
theorem complement_intersection_theorem :
  (U \ A) ∩ B = {2, 4} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l2730_273024


namespace NUMINAMATH_CALUDE_board_number_after_hour_l2730_273001

def digit_product (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) * digit_product (n / 10)

def next_number (n : ℕ) : ℕ :=
  digit_product n + 15

def iterate_operation (n : ℕ) (iterations : ℕ) : ℕ :=
  match iterations with
  | 0 => n
  | k + 1 => iterate_operation (next_number n) k

theorem board_number_after_hour (initial : ℕ) (h : initial = 98) :
  iterate_operation initial 60 = 24 :=
sorry

end NUMINAMATH_CALUDE_board_number_after_hour_l2730_273001


namespace NUMINAMATH_CALUDE_largest_digit_divisible_by_six_l2730_273012

theorem largest_digit_divisible_by_six :
  ∃ (N : ℕ), N ≤ 9 ∧ (7218 * N) % 6 = 0 ∧ ∀ (M : ℕ), M ≤ 9 ∧ (7218 * M) % 6 = 0 → M ≤ N :=
by sorry

end NUMINAMATH_CALUDE_largest_digit_divisible_by_six_l2730_273012


namespace NUMINAMATH_CALUDE_bread_making_time_is_375_l2730_273097

/-- Represents the duration of each step in Mark's bread-making process -/
def bread_making_steps : List ℕ := [30, 120, 20, 120, 10, 30, 30, 15]

/-- The total time Mark spends making bread -/
def total_bread_making_time : ℕ := bread_making_steps.sum

/-- Theorem stating that the total time Mark spends making bread is 375 minutes -/
theorem bread_making_time_is_375 : total_bread_making_time = 375 := by
  sorry

#eval total_bread_making_time

end NUMINAMATH_CALUDE_bread_making_time_is_375_l2730_273097


namespace NUMINAMATH_CALUDE_max_sum_abc_l2730_273023

/-- Definition of A_n -/
def A_n (a n : ℕ) : ℕ := a * (10^(3*n) - 1) / 9

/-- Definition of B_n -/
def B_n (b n : ℕ) : ℕ := b * (10^(2*n) - 1) / 9

/-- Definition of C_n -/
def C_n (c n : ℕ) : ℕ := c * (10^(2*n) - 1) / 9

/-- The main theorem -/
theorem max_sum_abc :
  ∃ (a b c : ℕ), 1 ≤ a ∧ a ≤ 9 ∧
                 1 ≤ b ∧ b ≤ 9 ∧
                 1 ≤ c ∧ c ≤ 9 ∧
                 (∃ (n : ℕ), n > 0 ∧ C_n c n - B_n b n = (A_n a n)^2) ∧
                 a + b + c = 18 ∧
                 ∀ (a' b' c' : ℕ), 1 ≤ a' ∧ a' ≤ 9 ∧
                                   1 ≤ b' ∧ b' ≤ 9 ∧
                                   1 ≤ c' ∧ c' ≤ 9 ∧
                                   (∃ (n : ℕ), n > 0 ∧ C_n c' n - B_n b' n = (A_n a' n)^2) →
                                   a' + b' + c' ≤ 18 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_abc_l2730_273023


namespace NUMINAMATH_CALUDE_complex_power_4_30_degrees_l2730_273092

theorem complex_power_4_30_degrees : 
  (2 * Complex.cos (π / 6) + 2 * Complex.I * Complex.sin (π / 6)) ^ 4 = -8 + 8 * Complex.I * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_4_30_degrees_l2730_273092


namespace NUMINAMATH_CALUDE_sum_of_xyz_l2730_273058

theorem sum_of_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x * y = 30) (h2 : x * z = 60) (h3 : y * z = 90) :
  x + y + z = 11 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_sum_of_xyz_l2730_273058


namespace NUMINAMATH_CALUDE_grasshopper_movement_l2730_273044

/-- Represents the possible jump distances of the grasshopper -/
inductive Jump
| large : Jump  -- 36 cm jump
| small : Jump  -- 14 cm jump

/-- Represents the direction of the jump -/
inductive Direction
| left : Direction
| right : Direction

/-- Represents a single jump of the grasshopper -/
structure GrasshopperJump :=
  (distance : Jump)
  (direction : Direction)

/-- The distance covered by a single jump -/
def jumpDistance (j : GrasshopperJump) : ℤ :=
  match j.distance, j.direction with
  | Jump.large, Direction.right => 36
  | Jump.large, Direction.left  => -36
  | Jump.small, Direction.right => 14
  | Jump.small, Direction.left  => -14

/-- The total distance covered by a sequence of jumps -/
def totalDistance (jumps : List GrasshopperJump) : ℤ :=
  jumps.foldl (fun acc j => acc + jumpDistance j) 0

/-- Predicate to check if a distance is reachable by the grasshopper -/
def isReachable (d : ℤ) : Prop :=
  ∃ (jumps : List GrasshopperJump), totalDistance jumps = d

theorem grasshopper_movement :
  (¬ isReachable 3) ∧ (isReachable 2) ∧ (isReachable 1234) := by sorry

end NUMINAMATH_CALUDE_grasshopper_movement_l2730_273044


namespace NUMINAMATH_CALUDE_irrational_number_existence_l2730_273055

theorem irrational_number_existence : ∃ α : ℝ, (α > 1) ∧ (Irrational α) ∧
  (∀ n : ℕ, n ≥ 1 → (⌊α^n⌋ : ℤ) % 2017 = 0) := by
  sorry

end NUMINAMATH_CALUDE_irrational_number_existence_l2730_273055


namespace NUMINAMATH_CALUDE_division_problem_l2730_273046

theorem division_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) 
  (h1 : dividend = 181)
  (h2 : divisor = 20)
  (h3 : remainder = 1)
  (h4 : dividend = divisor * quotient + remainder) :
  quotient = 9 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l2730_273046


namespace NUMINAMATH_CALUDE_inequality_proof_l2730_273026

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  x / Real.sqrt y + y / Real.sqrt x ≥ Real.sqrt x + Real.sqrt y := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2730_273026


namespace NUMINAMATH_CALUDE_discount_rates_sum_l2730_273014

theorem discount_rates_sum : 
  let fox_price : ℝ := 15
  let pony_price : ℝ := 20
  let fox_quantity : ℕ := 3
  let pony_quantity : ℕ := 2
  let total_savings : ℝ := 9
  let pony_discount_rate : ℝ := 18.000000000000014
  let fox_discount_rate : ℝ := (total_savings - pony_discount_rate / 100 * pony_price * pony_quantity) / (fox_price * fox_quantity) * 100
  fox_discount_rate + pony_discount_rate = 22.000000000000014 := by
  sorry

end NUMINAMATH_CALUDE_discount_rates_sum_l2730_273014


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2730_273015

theorem sqrt_equation_solution :
  ∃! x : ℝ, Real.sqrt (x + 4) + Real.sqrt (x + 6) = 12 ∧ x = 4465 / 144 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2730_273015


namespace NUMINAMATH_CALUDE_probability_theorem_l2730_273008

/-- Represents a unit cube with a certain number of painted faces -/
structure UnitCube where
  painted_faces : Nat

/-- Represents the large cube composed of unit cubes -/
def LargeCube : Type := List UnitCube

/-- Creates a large cube with the given specifications -/
def create_large_cube : LargeCube :=
  -- 8 cubes with 3 painted faces
  (List.replicate 8 ⟨3⟩) ++
  -- 18 cubes with 2 painted faces
  (List.replicate 18 ⟨2⟩) ++
  -- 27 cubes with 1 painted face
  (List.replicate 27 ⟨1⟩) ++
  -- Remaining cubes with 0 painted faces
  (List.replicate 72 ⟨0⟩)

/-- Calculates the probability of selecting one cube with 3 painted faces
    and one cube with 1 painted face when choosing 2 cubes at random -/
def probability_3_and_1 (cube : LargeCube) : Rat :=
  let total_combinations := (List.length cube).choose 2
  let favorable_outcomes := (cube.filter (λ c => c.painted_faces = 3)).length *
                            (cube.filter (λ c => c.painted_faces = 1)).length
  favorable_outcomes / total_combinations

/-- The main theorem to prove -/
theorem probability_theorem :
  probability_3_and_1 create_large_cube = 216 / 7750 := by
  sorry

end NUMINAMATH_CALUDE_probability_theorem_l2730_273008


namespace NUMINAMATH_CALUDE_new_men_average_age_l2730_273000

/-- Given a group of 12 men, where replacing two men aged 21 and 23 with two new men
    increases the average age by 1 year, prove that the average age of the two new men is 28 years. -/
theorem new_men_average_age
  (n : ℕ) -- number of men
  (old_age1 old_age2 : ℕ) -- ages of the two replaced men
  (avg_increase : ℚ) -- increase in average age
  (h1 : n = 12)
  (h2 : old_age1 = 21)
  (h3 : old_age2 = 23)
  (h4 : avg_increase = 1) :
  (old_age1 + old_age2 + n * avg_increase) / 2 = 28 :=
by sorry

end NUMINAMATH_CALUDE_new_men_average_age_l2730_273000


namespace NUMINAMATH_CALUDE_min_value_theorem_l2730_273020

-- Define the function f(x) = ax^2 - 4x + c
def f (a c x : ℝ) : ℝ := a * x^2 - 4 * x + c

-- State the theorem
theorem min_value_theorem (a c : ℝ) (h1 : a > 0) 
  (h2 : Set.range (f a c) = Set.Ici 1) :
  ∃ (m : ℝ), m = 3 ∧ ∀ (x : ℝ), (1 / (c - 1)) + (9 / a) ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2730_273020


namespace NUMINAMATH_CALUDE_smallest_divisible_by_14_15_16_l2730_273027

theorem smallest_divisible_by_14_15_16 : ∃ n : ℕ+, 
  (∀ m : ℕ+, 14 ∣ m ∧ 15 ∣ m ∧ 16 ∣ m → n ≤ m) ∧
  14 ∣ n ∧ 15 ∣ n ∧ 16 ∣ n :=
by
  use 1680
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_14_15_16_l2730_273027


namespace NUMINAMATH_CALUDE_sugar_concentration_of_second_solution_l2730_273065

/-- Given two solutions A and B, where:
    - A is 10% sugar by weight
    - B has an unknown sugar concentration
    - 3/4 of A is mixed with 1/4 of B
    - The resulting mixture is 16% sugar by weight
    This theorem proves that B must be 34% sugar by weight -/
theorem sugar_concentration_of_second_solution
  (W : ℝ) -- Total weight of the original solution
  (h_W_pos : W > 0) -- Assumption that W is positive
  : let A := 0.10 -- Sugar concentration of solution A (10%)
    let final_concentration := 0.16 -- Sugar concentration of final mixture (16%)
    let B := (4 * final_concentration - 3 * A) -- Sugar concentration of solution B
    B = 0.34 -- B is 34% sugar by weight
  := by sorry

end NUMINAMATH_CALUDE_sugar_concentration_of_second_solution_l2730_273065


namespace NUMINAMATH_CALUDE_adults_average_age_is_22_l2730_273095

/-- Represents the programming bootcamp group -/
structure BootcampGroup where
  totalMembers : ℕ
  averageAge : ℕ
  girlsCount : ℕ
  boysCount : ℕ
  adultsCount : ℕ
  girlsAverageAge : ℕ
  boysAverageAge : ℕ

/-- Calculates the average age of adults in the bootcamp group -/
def adultsAverageAge (group : BootcampGroup) : ℕ :=
  ((group.totalMembers * group.averageAge) - 
   (group.girlsCount * group.girlsAverageAge) - 
   (group.boysCount * group.boysAverageAge)) / group.adultsCount

/-- Theorem stating that the average age of adults is 22 years -/
theorem adults_average_age_is_22 (group : BootcampGroup) 
  (h1 : group.totalMembers = 50)
  (h2 : group.averageAge = 20)
  (h3 : group.girlsCount = 25)
  (h4 : group.boysCount = 20)
  (h5 : group.adultsCount = 5)
  (h6 : group.girlsAverageAge = 18)
  (h7 : group.boysAverageAge = 22) :
  adultsAverageAge group = 22 := by
  sorry


end NUMINAMATH_CALUDE_adults_average_age_is_22_l2730_273095


namespace NUMINAMATH_CALUDE_females_with_advanced_degrees_l2730_273093

/-- Proves the number of females with advanced degrees in a company --/
theorem females_with_advanced_degrees 
  (total_employees : ℕ)
  (total_females : ℕ)
  (employees_with_advanced_degrees : ℕ)
  (males_with_college_only : ℕ)
  (h1 : total_employees = 180)
  (h2 : total_females = 110)
  (h3 : employees_with_advanced_degrees = 90)
  (h4 : males_with_college_only = 35) :
  total_females - (total_employees - employees_with_advanced_degrees - males_with_college_only) = 55 :=
by sorry

end NUMINAMATH_CALUDE_females_with_advanced_degrees_l2730_273093


namespace NUMINAMATH_CALUDE_point_transformation_l2730_273047

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the second quadrant -/
def SecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Definition of the third quadrant -/
def ThirdQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- The theorem stating that if P is in the second quadrant, then P' is in the third quadrant -/
theorem point_transformation (m n : ℝ) :
  let P : Point := ⟨m, n⟩
  let P' : Point := ⟨-m^2, -n⟩
  SecondQuadrant P → ThirdQuadrant P' := by
  sorry


end NUMINAMATH_CALUDE_point_transformation_l2730_273047


namespace NUMINAMATH_CALUDE_reciprocal_and_opposite_l2730_273064

-- Define the reciprocal function
def reciprocal (x : ℚ) : ℚ := 1 / x

-- Define the opposite function
def opposite (x : ℝ) : ℝ := -x

-- Theorem statement
theorem reciprocal_and_opposite :
  (reciprocal (2 / 3) = 3 / 2) ∧ (opposite (-2.5) = 2.5) := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_and_opposite_l2730_273064


namespace NUMINAMATH_CALUDE_pen_sales_revenue_pen_sales_revenue_proof_l2730_273033

theorem pen_sales_revenue : ℝ → Prop :=
  fun total_revenue =>
    ∀ (total_pens : ℕ) (displayed_pens : ℕ) (storeroom_pens : ℕ),
      (displayed_pens : ℝ) = 0.3 * total_pens ∧
      (storeroom_pens : ℝ) = 0.7 * total_pens ∧
      storeroom_pens = 210 ∧
      total_revenue = (displayed_pens : ℝ) * 2 →
      total_revenue = 180

-- The proof is omitted
theorem pen_sales_revenue_proof : pen_sales_revenue 180 := by
  sorry

end NUMINAMATH_CALUDE_pen_sales_revenue_pen_sales_revenue_proof_l2730_273033


namespace NUMINAMATH_CALUDE_power_function_positive_l2730_273045

theorem power_function_positive (α : ℚ) (x : ℝ) (h : x > 0) : x ^ (α : ℝ) > 0 := by
  sorry

end NUMINAMATH_CALUDE_power_function_positive_l2730_273045


namespace NUMINAMATH_CALUDE_total_baseball_fans_l2730_273006

theorem total_baseball_fans (yankees mets redsox : ℕ) : 
  yankees * 2 = mets * 3 →
  mets * 5 = redsox * 4 →
  mets = 88 →
  yankees + mets + redsox = 330 :=
by sorry

end NUMINAMATH_CALUDE_total_baseball_fans_l2730_273006


namespace NUMINAMATH_CALUDE_a_between_3_and_5_necessary_not_sufficient_l2730_273016

/-- The equation of a potential ellipse -/
def ellipse_equation (a x y : ℝ) : Prop :=
  x^2 / (a - 3) + y^2 / (5 - a) = 1

/-- The condition that a is between 3 and 5 -/
def a_between_3_and_5 (a : ℝ) : Prop :=
  3 < a ∧ a < 5

/-- The statement that the equation represents an ellipse -/
def is_ellipse (a : ℝ) : Prop :=
  ∃ (x y : ℝ), ellipse_equation a x y ∧ (x ≠ 0 ∨ y ≠ 0)

/-- The main theorem: a_between_3_and_5 is necessary but not sufficient for is_ellipse -/
theorem a_between_3_and_5_necessary_not_sufficient :
  (∀ a : ℝ, is_ellipse a → a_between_3_and_5 a) ∧
  ¬(∀ a : ℝ, a_between_3_and_5 a → is_ellipse a) :=
sorry

end NUMINAMATH_CALUDE_a_between_3_and_5_necessary_not_sufficient_l2730_273016


namespace NUMINAMATH_CALUDE_factorization_difference_l2730_273032

theorem factorization_difference (a b : ℤ) : 
  (∀ y : ℝ, 5 * y^2 + 3 * y - 44 = (5 * y + a) * (y + b)) → 
  a - b = -15 := by
sorry

end NUMINAMATH_CALUDE_factorization_difference_l2730_273032


namespace NUMINAMATH_CALUDE_binary_sum_proof_l2730_273067

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldr (λ ⟨i, bit⟩ acc => acc + if bit then 2^i else 0) 0

theorem binary_sum_proof :
  let b1 := [true, true, false, true]  -- 1101₂
  let b2 := [true, false, true]        -- 101₂
  let b3 := [true, true, true]         -- 111₂
  let b4 := [true, false, false, false, true]  -- 10001₂
  let result := [false, true, false, true, false, true]  -- 101010₂
  binary_to_decimal b1 + binary_to_decimal b2 + binary_to_decimal b3 + binary_to_decimal b4 = binary_to_decimal result :=
by sorry

end NUMINAMATH_CALUDE_binary_sum_proof_l2730_273067


namespace NUMINAMATH_CALUDE_cannot_eat_all_except_central_l2730_273075

/-- Represents a 3D coordinate within the cheese cube -/
structure Coordinate where
  x : Fin 3
  y : Fin 3
  z : Fin 3

/-- Represents the color of a unit cube -/
inductive Color
  | White
  | Black

/-- The cheese cube -/
def CheeseCube := Fin 3 → Fin 3 → Fin 3 → Color

/-- Determines if two coordinates are adjacent (share a face) -/
def isAdjacent (c1 c2 : Coordinate) : Prop :=
  (c1.x = c2.x ∧ c1.y = c2.y ∧ (c1.z = c2.z + 1 ∨ c1.z + 1 = c2.z)) ∨
  (c1.x = c2.x ∧ c1.z = c2.z ∧ (c1.y = c2.y + 1 ∨ c1.y + 1 = c2.y)) ∨
  (c1.y = c2.y ∧ c1.z = c2.z ∧ (c1.x = c2.x + 1 ∨ c1.x + 1 = c2.x))

/-- Assigns a color to each coordinate based on the sum of its components -/
def colorCube : CheeseCube :=
  fun x y z => if (x.val + y.val + z.val) % 2 = 0 then Color.White else Color.Black

/-- The central cube coordinate -/
def centralCube : Coordinate := ⟨1, 1, 1⟩

/-- Theorem stating that it's impossible to eat all cubes except the central one -/
theorem cannot_eat_all_except_central :
  ¬∃ (path : List Coordinate),
    path.Nodup ∧
    path.length = 26 ∧
    (∀ i, i ∈ path → i ≠ centralCube) ∧
    (∀ i j, i ∈ path → j ∈ path → i ≠ j → isAdjacent i j) :=
  sorry

end NUMINAMATH_CALUDE_cannot_eat_all_except_central_l2730_273075


namespace NUMINAMATH_CALUDE_prime_divides_all_f_l2730_273025

def f (n x : ℕ) : ℕ := Nat.choose n x

theorem prime_divides_all_f (p : ℕ) (hp : Prime p) (n : ℕ) (hn : n > 1) :
  (∀ x : ℕ, 1 ≤ x ∧ x < n → p ∣ f n x) ↔ ∃ m : ℕ, n = p ^ m :=
sorry

end NUMINAMATH_CALUDE_prime_divides_all_f_l2730_273025


namespace NUMINAMATH_CALUDE_hockey_team_selection_l2730_273089

def number_of_players : ℕ := 18
def players_to_select : ℕ := 8

theorem hockey_team_selection :
  Nat.choose number_of_players players_to_select = 43758 := by
  sorry

end NUMINAMATH_CALUDE_hockey_team_selection_l2730_273089


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2730_273048

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x => x^2 + 4*x - 1
  ∃ x₁ x₂ : ℝ, (x₁ = -2 + Real.sqrt 5 ∧ x₂ = -2 - Real.sqrt 5) ∧ 
              (f x₁ = 0 ∧ f x₂ = 0) ∧
              (∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2730_273048


namespace NUMINAMATH_CALUDE_goose_eggs_count_l2730_273043

theorem goose_eggs_count (total_eggs : ℕ) : 
  (total_eggs : ℚ) * (1/2) * (3/4) * (2/5) = 120 →
  total_eggs = 400 := by
  sorry

end NUMINAMATH_CALUDE_goose_eggs_count_l2730_273043


namespace NUMINAMATH_CALUDE_seed_germination_problem_l2730_273060

theorem seed_germination_problem (seeds_plot1 seeds_plot2 : ℕ) 
  (germination_rate_plot1 germination_rate_total : ℚ) :
  seeds_plot1 = 300 →
  seeds_plot2 = 200 →
  germination_rate_plot1 = 25 / 100 →
  germination_rate_total = 29 / 100 →
  ∃ (germination_rate_plot2 : ℚ),
    germination_rate_plot2 = 35 / 100 ∧
    (seeds_plot1 : ℚ) * germination_rate_plot1 + (seeds_plot2 : ℚ) * germination_rate_plot2 = 
    ((seeds_plot1 + seeds_plot2) : ℚ) * germination_rate_total :=
by sorry

end NUMINAMATH_CALUDE_seed_germination_problem_l2730_273060


namespace NUMINAMATH_CALUDE_nicole_fish_tanks_water_needed_l2730_273021

theorem nicole_fish_tanks_water_needed :
  -- Define the number of tanks
  let total_tanks : ℕ := 4
  let first_group_tanks : ℕ := 2
  let second_group_tanks : ℕ := total_tanks - first_group_tanks

  -- Define water needed for each group
  let first_group_water : ℕ := 8
  let second_group_water : ℕ := first_group_water - 2

  -- Define the number of weeks
  let weeks : ℕ := 4

  -- Calculate total water needed per week
  let water_per_week : ℕ := first_group_tanks * first_group_water + second_group_tanks * second_group_water

  -- Calculate total water needed for four weeks
  let total_water : ℕ := water_per_week * weeks

  -- Prove that the total water needed is 112 gallons
  total_water = 112 := by sorry

end NUMINAMATH_CALUDE_nicole_fish_tanks_water_needed_l2730_273021


namespace NUMINAMATH_CALUDE_isosceles_triangle_properties_l2730_273084

/-- A triangle with sides 13, 13, and 10 units -/
structure IsoscelesTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a = 13
  hb : b = 13
  hc : c = 10

/-- The sum of squares of median lengths for the given triangle -/
def sumSquaresMedians (t : IsoscelesTriangle) : ℝ := 278.5

/-- The area of the given triangle -/
def triangleArea (t : IsoscelesTriangle) : ℝ := 60

theorem isosceles_triangle_properties (t : IsoscelesTriangle) :
  sumSquaresMedians t = 278.5 ∧ triangleArea t = 60 := by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_properties_l2730_273084


namespace NUMINAMATH_CALUDE_hyperbola_min_value_l2730_273078

theorem hyperbola_min_value (x y : ℝ) : 
  x^2 / 4 - y^2 = 1 → (∀ z w : ℝ, z^2 / 4 - w^2 = 1 → 3*x^2 - 2*y ≤ 3*z^2 - 2*w) ∧ (∃ a b : ℝ, a^2 / 4 - b^2 = 1 ∧ 3*a^2 - 2*b = 143/12) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_min_value_l2730_273078


namespace NUMINAMATH_CALUDE_glorias_turtle_time_l2730_273036

/-- The time it takes for Gloria's turtle to finish the race -/
def glorias_time (gretas_time georges_time : ℕ) : ℕ :=
  2 * georges_time

/-- Theorem stating that Gloria's turtle finished in 8 minutes -/
theorem glorias_turtle_time :
  let gretas_time := 6
  let georges_time := gretas_time - 2
  glorias_time gretas_time georges_time = 8 := by sorry

end NUMINAMATH_CALUDE_glorias_turtle_time_l2730_273036


namespace NUMINAMATH_CALUDE_quadratic_equation_distinct_roots_l2730_273035

theorem quadratic_equation_distinct_roots (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 4*x₁ - m = 0 ∧ x₂^2 + 4*x₂ - m = 0) ↔ m > -4 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_distinct_roots_l2730_273035


namespace NUMINAMATH_CALUDE_equal_intercept_line_equation_l2730_273052

/-- A line passing through the point (3, -4) with equal intercepts on the coordinate axes -/
structure EqualInterceptLine where
  -- The slope-intercept form of the line: y = mx + b
  m : ℝ
  b : ℝ
  -- The line passes through (3, -4)
  point_condition : -4 = m * 3 + b
  -- The line has equal intercepts on both axes
  equal_intercepts : m ≠ -1 → b / (1 + m) = -b / m

/-- The equation of an EqualInterceptLine is either 4x + 3y = 0 or x + y + 1 = 0 -/
theorem equal_intercept_line_equation (l : EqualInterceptLine) :
  (4 * l.m + 3 = 0 ∧ l.b = 0) ∨ (l.m = -1 ∧ l.b = -1) :=
sorry

end NUMINAMATH_CALUDE_equal_intercept_line_equation_l2730_273052


namespace NUMINAMATH_CALUDE_line_plane_relationship_l2730_273086

/-- A line in 3D space -/
structure Line3D where
  -- Add necessary fields

/-- A plane in 3D space -/
structure Plane3D where
  -- Add necessary fields

/-- Perpendicular relation between a line and a plane -/
def perpendicular_line_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Perpendicular relation between two lines -/
def perpendicular_lines (l1 l2 : Line3D) : Prop :=
  sorry

/-- Parallel relation between a line and a plane -/
def parallel_line_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Containment relation of a line in a plane -/
def line_in_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

theorem line_plane_relationship (m n : Line3D) (α : Plane3D)
    (h1 : perpendicular_line_plane m α)
    (h2 : perpendicular_lines m n) :
    parallel_line_plane n α ∨ line_in_plane n α :=
  sorry

end NUMINAMATH_CALUDE_line_plane_relationship_l2730_273086


namespace NUMINAMATH_CALUDE_seating_uncertainty_l2730_273062

-- Define the types for people and seats
inductive Person : Type
| Abby : Person
| Bret : Person
| Carl : Person
| Dana : Person

inductive Seat : Type
| One : Seat
| Two : Seat
| Three : Seat
| Four : Seat

-- Define the seating arrangement
def Seating := Person → Seat

-- Define the "next to" relation
def next_to (s : Seating) (p1 p2 : Person) : Prop :=
  (s p1 = Seat.One ∧ s p2 = Seat.Two) ∨
  (s p1 = Seat.Two ∧ s p2 = Seat.Three) ∨
  (s p1 = Seat.Three ∧ s p2 = Seat.Four) ∨
  (s p2 = Seat.One ∧ s p1 = Seat.Two) ∨
  (s p2 = Seat.Two ∧ s p1 = Seat.Three) ∨
  (s p2 = Seat.Three ∧ s p1 = Seat.Four)

-- Define the "between" relation
def between (s : Seating) (p1 p2 p3 : Person) : Prop :=
  (s p1 = Seat.One ∧ s p2 = Seat.Two ∧ s p3 = Seat.Three) ∨
  (s p1 = Seat.Two ∧ s p2 = Seat.Three ∧ s p3 = Seat.Four) ∨
  (s p3 = Seat.One ∧ s p2 = Seat.Two ∧ s p1 = Seat.Three) ∨
  (s p3 = Seat.Two ∧ s p2 = Seat.Three ∧ s p1 = Seat.Four)

theorem seating_uncertainty (s : Seating) :
  (next_to s Person.Dana Person.Bret) ∧
  (¬ between s Person.Abby Person.Bret Person.Carl) ∧
  (s Person.Bret = Seat.One) →
  ¬ (∀ p : Person, s p = Seat.Three → (p = Person.Abby ∨ p = Person.Carl)) :=
by sorry

end NUMINAMATH_CALUDE_seating_uncertainty_l2730_273062


namespace NUMINAMATH_CALUDE_correct_calculation_l2730_273071

theorem correct_calculation : (36 - 12) / (3 / 2) = 16 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l2730_273071


namespace NUMINAMATH_CALUDE_total_fruits_l2730_273063

theorem total_fruits (cucumbers watermelons : ℕ) : 
  cucumbers = 18 → 
  watermelons = cucumbers + 8 → 
  cucumbers + watermelons = 44 :=
by
  sorry

end NUMINAMATH_CALUDE_total_fruits_l2730_273063


namespace NUMINAMATH_CALUDE_parabola_directrix_tangent_to_circle_l2730_273017

/-- The value of p for a parabola y^2 = 2px (p > 0) whose directrix is tangent to the circle (x-3)^2 + y^2 = 16 -/
theorem parabola_directrix_tangent_to_circle : 
  ∃ (p : ℝ), p > 0 ∧ 
  (∃ (x y : ℝ), y^2 = 2*p*x) ∧ 
  (∃ (x y : ℝ), (x-3)^2 + y^2 = 16) ∧
  (∃ (x : ℝ), x = -p/2 ∧ (x-3)^2 = 16) →
  p = 2 :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_tangent_to_circle_l2730_273017


namespace NUMINAMATH_CALUDE_principal_value_integral_equals_zero_l2730_273085

open Real MeasureTheory

/-- The principal value of the improper integral from 1/e to e of 1/(x ln(x)) dx is 0 -/
theorem principal_value_integral_equals_zero :
  ∃ (I : ℝ), I = (∫ (x : ℝ) in Set.Icc (1/Real.exp 1) (Real.exp 1), 1 / (x * Real.log x)) ∧ I = 0 := by
  sorry

end NUMINAMATH_CALUDE_principal_value_integral_equals_zero_l2730_273085


namespace NUMINAMATH_CALUDE_fraction_simplification_l2730_273040

theorem fraction_simplification : (3 : ℚ) / (2 - 2 / 5) = 15 / 8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2730_273040


namespace NUMINAMATH_CALUDE_subtraction_and_divisibility_implies_sum_l2730_273066

theorem subtraction_and_divisibility_implies_sum (a b : Nat) : 
  (741 - (300 + 10*a + 4) = 400 + 10*b + 7) → 
  ((400 + 10*b + 7) % 11 = 0) → 
  (a + b = 3) := by
sorry

end NUMINAMATH_CALUDE_subtraction_and_divisibility_implies_sum_l2730_273066


namespace NUMINAMATH_CALUDE_vaccine_cost_reduction_formula_correct_l2730_273076

/-- Given an initial cost and an annual decrease rate, calculates the cost reduction of producing vaccines after two years. -/
def vaccine_cost_reduction (initial_cost : ℝ) (annual_decrease_rate : ℝ) : ℝ :=
  let cost_last_year := initial_cost * (1 - annual_decrease_rate)
  let cost_this_year := initial_cost * (1 - annual_decrease_rate)^2
  cost_last_year - cost_this_year

/-- Theorem stating that the vaccine cost reduction formula is correct for the given initial cost. -/
theorem vaccine_cost_reduction_formula_correct :
  ∀ (x : ℝ), vaccine_cost_reduction 5000 x = 5000 * x - 5000 * x^2 :=
by
  sorry

#eval vaccine_cost_reduction 5000 0.1

end NUMINAMATH_CALUDE_vaccine_cost_reduction_formula_correct_l2730_273076


namespace NUMINAMATH_CALUDE_focus_of_specific_ellipse_l2730_273049

/-- An ellipse with given major and minor axis endpoints -/
structure Ellipse where
  major_axis_start : ℝ × ℝ
  major_axis_end : ℝ × ℝ
  minor_axis_start : ℝ × ℝ
  minor_axis_end : ℝ × ℝ

/-- The focus of an ellipse with the greater x-coordinate -/
def focus_with_greater_x (e : Ellipse) : ℝ × ℝ :=
  sorry

/-- Theorem stating that for the given ellipse, the focus with greater x-coordinate is at (3, -2) -/
theorem focus_of_specific_ellipse :
  let e : Ellipse := {
    major_axis_start := (0, -2),
    major_axis_end := (6, -2),
    minor_axis_start := (3, 1),
    minor_axis_end := (3, -5)
  }
  focus_with_greater_x e = (3, -2) := by
  sorry

end NUMINAMATH_CALUDE_focus_of_specific_ellipse_l2730_273049


namespace NUMINAMATH_CALUDE_cubs_series_win_probability_l2730_273013

def probability_cubs_win_game : ℚ := 2/3

def probability_cubs_win_series : ℚ :=
  (1 * probability_cubs_win_game^3 * (1 - probability_cubs_win_game)^0) +
  (3 * probability_cubs_win_game^3 * (1 - probability_cubs_win_game)^1) +
  (6 * probability_cubs_win_game^3 * (1 - probability_cubs_win_game)^2)

theorem cubs_series_win_probability :
  probability_cubs_win_series = 64/81 :=
by sorry

end NUMINAMATH_CALUDE_cubs_series_win_probability_l2730_273013


namespace NUMINAMATH_CALUDE_triangles_count_l2730_273082

/-- The number of triangles that can be made from a wire -/
def triangles_from_wire (original_length : ℕ) (remaining_length : ℕ) (triangle_wire_length : ℕ) : ℕ :=
  (original_length - remaining_length) / triangle_wire_length

/-- Theorem: Given the specified wire lengths, 24 triangles can be made -/
theorem triangles_count : triangles_from_wire 84 12 3 = 24 := by
  sorry

end NUMINAMATH_CALUDE_triangles_count_l2730_273082


namespace NUMINAMATH_CALUDE_simplify_fraction_l2730_273099

theorem simplify_fraction : (120 : ℚ) / 180 = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2730_273099


namespace NUMINAMATH_CALUDE_min_value_of_trig_function_l2730_273087

theorem min_value_of_trig_function :
  ∃ (min : ℝ), min = -Real.sqrt 2 - 1 ∧
  ∀ (x : ℝ), 2 * Real.sin x * Real.cos x - 2 * (Real.sin x)^2 ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_of_trig_function_l2730_273087


namespace NUMINAMATH_CALUDE_two_point_zero_six_recurring_l2730_273073

def recurring_decimal_02 : ℚ := 2 / 99

theorem two_point_zero_six_recurring (h : recurring_decimal_02 = 2 / 99) :
  2 + 3 * recurring_decimal_02 = 68 / 33 := by
  sorry

end NUMINAMATH_CALUDE_two_point_zero_six_recurring_l2730_273073


namespace NUMINAMATH_CALUDE_two_trains_problem_l2730_273022

/-- Two trains problem -/
theorem two_trains_problem (train_length : ℝ) (time_first_train : ℝ) (time_crossing : ℝ) :
  train_length = 120 →
  time_first_train = 12 →
  time_crossing = 16 →
  ∃ time_second_train : ℝ,
    time_second_train = 24 ∧
    train_length / time_first_train + train_length / time_second_train = 2 * train_length / time_crossing :=
by sorry

end NUMINAMATH_CALUDE_two_trains_problem_l2730_273022


namespace NUMINAMATH_CALUDE_wedge_volume_l2730_273081

/-- The volume of a wedge cut from a cylindrical log --/
theorem wedge_volume (d : ℝ) (θ : ℝ) (h : θ = 60) :
  let r := d / 2
  let cylinder_volume := π * r^2 * d
  let wedge_volume := cylinder_volume * θ / 360
  d = 16 → wedge_volume = 341 * π :=
by sorry

end NUMINAMATH_CALUDE_wedge_volume_l2730_273081


namespace NUMINAMATH_CALUDE_store_shirts_sold_l2730_273083

theorem store_shirts_sold (num_jeans : ℕ) (shirt_price : ℕ) (total_earnings : ℕ) :
  num_jeans = 10 ∧ 
  shirt_price = 10 ∧ 
  total_earnings = 400 →
  ∃ (num_shirts : ℕ), 
    num_shirts * shirt_price + num_jeans * (2 * shirt_price) = total_earnings ∧
    num_shirts = 20 :=
by sorry

end NUMINAMATH_CALUDE_store_shirts_sold_l2730_273083


namespace NUMINAMATH_CALUDE_max_M_is_five_l2730_273079

/-- Definition of I_k -/
def I (k : ℕ) : ℕ := 10^(k+2) + 25

/-- Definition of M(k) -/
def M (k : ℕ) : ℕ := (I k).factors.count 2

/-- Theorem: The maximum value of M(k) for k > 0 is 5 -/
theorem max_M_is_five : ∃ (k : ℕ), k > 0 ∧ M k = 5 ∧ ∀ (j : ℕ), j > 0 → M j ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_max_M_is_five_l2730_273079


namespace NUMINAMATH_CALUDE_function_value_at_ten_l2730_273002

/-- Given a function f satisfying the recursive relation
    f(x+1) = f(x) / (1 + f(x)) for all x, and f(1) = 1,
    prove that f(10) = 1/10 -/
theorem function_value_at_ten
  (f : ℝ → ℝ)
  (h1 : ∀ x, f (x + 1) = f x / (1 + f x))
  (h2 : f 1 = 1) :
  f 10 = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_ten_l2730_273002


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2730_273090

theorem quadratic_inequality_solution_set : 
  {x : ℝ | 3 * x^2 - 5 * x - 2 < 0} = {x : ℝ | -1/3 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2730_273090


namespace NUMINAMATH_CALUDE_salary_unspent_fraction_l2730_273005

theorem salary_unspent_fraction (salary : ℝ) (salary_positive : salary > 0) :
  let first_week_spent := (1 / 4 : ℝ) * salary
  let each_other_week_spent := (1 / 5 : ℝ) * salary
  let total_spent := first_week_spent + 3 * each_other_week_spent
  (salary - total_spent) / salary = 3 / 20 := by
  sorry

end NUMINAMATH_CALUDE_salary_unspent_fraction_l2730_273005


namespace NUMINAMATH_CALUDE_cube_sum_from_sum_and_product_l2730_273068

theorem cube_sum_from_sum_and_product (x y : ℝ) 
  (h1 : x + y = 10) (h2 : x * y = 14) : x^3 + y^3 = 580 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_from_sum_and_product_l2730_273068


namespace NUMINAMATH_CALUDE_cattle_milk_production_l2730_273074

/-- Represents the total milk production of a group of cows over a given number of days -/
def total_milk_production (num_cows : ℕ) (milk_per_cow : ℕ) (num_days : ℕ) : ℕ :=
  num_cows * milk_per_cow * num_days

/-- Proves that the total milk production of 150 cows over 12 days is 2655000 oz -/
theorem cattle_milk_production : 
  let total_cows : ℕ := 150
  let group1_cows : ℕ := 75
  let group2_cows : ℕ := 75
  let group1_milk_per_cow : ℕ := 1300
  let group2_milk_per_cow : ℕ := 1650
  let num_days : ℕ := 12
  total_milk_production group1_cows group1_milk_per_cow num_days + 
  total_milk_production group2_cows group2_milk_per_cow num_days = 2655000 :=
by
  sorry

end NUMINAMATH_CALUDE_cattle_milk_production_l2730_273074


namespace NUMINAMATH_CALUDE_not_all_datasets_have_regression_equation_l2730_273041

-- Define a type for datasets
def Dataset : Type := Set (ℝ × ℝ)

-- Define a predicate for whether a dataset has a regression equation
def has_regression_equation (d : Dataset) : Prop := sorry

-- Theorem stating that not every dataset has a regression equation
theorem not_all_datasets_have_regression_equation : 
  ¬ (∀ d : Dataset, has_regression_equation d) := by sorry

end NUMINAMATH_CALUDE_not_all_datasets_have_regression_equation_l2730_273041


namespace NUMINAMATH_CALUDE_cos_is_valid_g_l2730_273053

def a (x : ℝ) : ℝ × ℝ := (1, x)
def b (f : ℝ → ℝ) (x : ℝ) : ℝ × ℝ := (f x, -x)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem cos_is_valid_g (f : ℝ → ℝ) (g : ℝ → ℝ) :
  (∀ x, dot_product (a x) (b f x) = g x) →
  is_even f →
  g = cos :=
by sorry

end NUMINAMATH_CALUDE_cos_is_valid_g_l2730_273053


namespace NUMINAMATH_CALUDE_shifted_sine_function_proof_l2730_273019

open Real

theorem shifted_sine_function_proof (φ : ℝ) (h1 : 0 < φ) (h2 : φ < π/2) : 
  (∃ x₁ x₂ : ℝ, |sin (2*x₁) - sin (2*(x₂ - φ))| = 2 ∧ 
   (∀ y₁ y₂ : ℝ, |sin (2*y₁) - sin (2*(y₂ - φ))| = 2 → |x₁ - x₂| ≤ |y₁ - y₂|) ∧
   |x₁ - x₂| = π/3) →
  φ = π/6 := by
sorry

end NUMINAMATH_CALUDE_shifted_sine_function_proof_l2730_273019


namespace NUMINAMATH_CALUDE_problem_solution_l2730_273009

def f (x : ℝ) : ℝ := |x - 1| - |x|

theorem problem_solution :
  let u := f (5/16)
  let v := f u
  let s := f v
  s = 1/2 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2730_273009


namespace NUMINAMATH_CALUDE_house_height_difference_l2730_273030

/-- Given three house heights, proves that the difference between the average height and 80 feet is 3 feet -/
theorem house_height_difference (h1 h2 h3 : ℕ) (h1_eq : h1 = 80) (h2_eq : h2 = 70) (h3_eq : h3 = 99) :
  (h1 + h2 + h3) / 3 - h1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_house_height_difference_l2730_273030


namespace NUMINAMATH_CALUDE_coefficient_value_l2730_273054

-- Define the derivative operation
noncomputable def derivative (f : ℝ → ℝ) : ℝ → ℝ := sorry

-- Define our function q
noncomputable def q : ℝ → ℝ := sorry

-- Define the coefficient a
def a : ℝ := sorry

-- State the theorem
theorem coefficient_value :
  (∀ x, derivative q x = a * q x - 3) →
  derivative (derivative q) 5 = 132 →
  a = 11.25 := by sorry

end NUMINAMATH_CALUDE_coefficient_value_l2730_273054


namespace NUMINAMATH_CALUDE_sin_cos_product_l2730_273094

theorem sin_cos_product (α : Real) (h : Real.tan α = 3) : 
  Real.sin α * Real.cos α = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_product_l2730_273094


namespace NUMINAMATH_CALUDE_ice_problem_solution_l2730_273051

def ice_problem (tray_a_initial tray_a_added : ℕ) : ℕ :=
  let tray_a := tray_a_initial + tray_a_added
  let tray_b := tray_a / 3
  let tray_c := 2 * tray_a
  tray_a + tray_b + tray_c

theorem ice_problem_solution :
  ice_problem 2 7 = 30 := by
  sorry

end NUMINAMATH_CALUDE_ice_problem_solution_l2730_273051


namespace NUMINAMATH_CALUDE_smallest_four_digit_arithmetic_sequence_l2730_273056

def is_arithmetic_sequence (a b c d : ℕ) : Prop :=
  ∃ r : ℤ, b = a + r ∧ c = b + r ∧ d = c + r

def digits_are_distinct (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.length = 4 ∧ digits.toFinset.card = 4

theorem smallest_four_digit_arithmetic_sequence :
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 
    digits_are_distinct n ∧
    is_arithmetic_sequence (n / 1000 % 10) (n / 100 % 10) (n / 10 % 10) (n % 10) →
  1234 ≤ n :=
sorry

end NUMINAMATH_CALUDE_smallest_four_digit_arithmetic_sequence_l2730_273056


namespace NUMINAMATH_CALUDE_part_one_part_two_l2730_273031

/-- Given a point M(2m+1, m-4) and N(5, 2) in the Cartesian coordinate system -/
def M (m : ℝ) : ℝ × ℝ := (2*m + 1, m - 4)
def N : ℝ × ℝ := (5, 2)

/-- Part 1: If MN is parallel to the x-axis, then M(13, 2) -/
theorem part_one (m : ℝ) : 
  (M m).2 = N.2 → M m = (13, 2) := by sorry

/-- Part 2: If M is 3 units to the right of the y-axis, then M(3, -3) -/
theorem part_two (m : ℝ) :
  (M m).1 = 3 → M m = (3, -3) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2730_273031


namespace NUMINAMATH_CALUDE_train_distance_difference_l2730_273061

/-- Proves the difference in distance traveled by two trains meeting each other --/
theorem train_distance_difference (v1 v2 total_distance : ℝ) 
  (h1 : v1 = 16)
  (h2 : v2 = 21)
  (h3 : total_distance = 444)
  (h4 : v1 > 0)
  (h5 : v2 > 0) :
  let t := total_distance / (v1 + v2)
  let d1 := v1 * t
  let d2 := v2 * t
  d2 - d1 = 60 := by sorry

end NUMINAMATH_CALUDE_train_distance_difference_l2730_273061


namespace NUMINAMATH_CALUDE_circle_equation_radius_l2730_273072

theorem circle_equation_radius (x y d : ℝ) : 
  (∀ x y, x^2 + 8*x + y^2 + 2*y + d = 0) → 
  (∃ h k : ℝ, ∀ x y, (x - h)^2 + (y - k)^2 = 5^2) →
  d = -8 := by
sorry

end NUMINAMATH_CALUDE_circle_equation_radius_l2730_273072


namespace NUMINAMATH_CALUDE_race_head_start_l2730_273010

theorem race_head_start (L : ℝ) (Va Vb : ℝ) (h : Va = (30 / 17) * Vb) :
  ∃ H : ℝ, H = (13 / 30) * L ∧ L / Va = (L - H) / Vb :=
by sorry

end NUMINAMATH_CALUDE_race_head_start_l2730_273010


namespace NUMINAMATH_CALUDE_standard_deviation_transform_l2730_273011

/-- Given a sample of 10 data points, this function represents their standard deviation. -/
def standard_deviation (x : Fin 10 → ℝ) : ℝ := sorry

/-- This function represents the transformation applied to each data point. -/
def transform (x : ℝ) : ℝ := 3 * x - 1

theorem standard_deviation_transform (x : Fin 10 → ℝ) :
  standard_deviation x = 8 →
  standard_deviation (λ i => transform (x i)) = 24 := by
  sorry

end NUMINAMATH_CALUDE_standard_deviation_transform_l2730_273011


namespace NUMINAMATH_CALUDE_poster_purchase_l2730_273028

theorem poster_purchase (regular_price : ℕ) (budget : ℕ) : 
  budget = 24 * regular_price → 
  (∃ (num_posters : ℕ), 
    num_posters * regular_price + (num_posters / 2) * (regular_price / 2) = budget ∧ 
    num_posters = 32) :=
by sorry

end NUMINAMATH_CALUDE_poster_purchase_l2730_273028


namespace NUMINAMATH_CALUDE_new_person_weight_is_75_l2730_273091

/-- The weight of the new person given the conditions of the problem -/
def new_person_weight (initial_count : ℕ) (average_increase : ℚ) (replaced_weight : ℚ) : ℚ :=
  replaced_weight + (initial_count : ℚ) * average_increase

/-- Theorem stating that the weight of the new person is 75 kg -/
theorem new_person_weight_is_75 :
  new_person_weight 8 (5/2) 55 = 75 := by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_is_75_l2730_273091


namespace NUMINAMATH_CALUDE_group_transfer_equation_l2730_273004

/-- 
Given two groups of people, with 22 in the first group and 26 in the second group,
this theorem proves the equation for the number of people that should be transferred
from the second group to the first group so that the first group has twice the number
of people as the second group.
-/
theorem group_transfer_equation (x : ℤ) : (22 + x = 2 * (26 - x)) ↔ 
  (22 + x = 2 * (26 - x) ∧ 
   22 + x > 0 ∧ 
   26 - x > 0) := by
  sorry

end NUMINAMATH_CALUDE_group_transfer_equation_l2730_273004


namespace NUMINAMATH_CALUDE_train_length_l2730_273057

/-- The length of a train given its speed, time to cross a platform, and the platform's length. -/
theorem train_length (train_speed : ℝ) (cross_time : ℝ) (platform_length : ℝ) : 
  train_speed = 72 * (1000 / 3600) → 
  cross_time = 25 → 
  platform_length = 300.04 →
  (train_speed * cross_time - platform_length) = 199.96 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l2730_273057


namespace NUMINAMATH_CALUDE_explanatory_variable_is_fertilizer_amount_l2730_273042

/-- A study on crop yield prediction -/
structure CropStudy where
  fertilizer_amount : ℝ
  crop_yield : ℝ

/-- The explanatory variable in a regression analysis -/
inductive ExplanatoryVariable
  | CropYield
  | FertilizerAmount
  | Experimenter
  | OtherFactors

/-- The study aims to predict crop yield based on fertilizer amount -/
def study_aim (s : CropStudy) : Prop :=
  ∃ f : ℝ → ℝ, s.crop_yield = f s.fertilizer_amount

/-- The correct explanatory variable for the given study -/
def correct_explanatory_variable : ExplanatoryVariable :=
  ExplanatoryVariable.FertilizerAmount

/-- Theorem: The explanatory variable in the crop yield study is the fertilizer amount -/
theorem explanatory_variable_is_fertilizer_amount 
  (s : CropStudy) (aim : study_aim s) :
  correct_explanatory_variable = ExplanatoryVariable.FertilizerAmount :=
sorry

end NUMINAMATH_CALUDE_explanatory_variable_is_fertilizer_amount_l2730_273042


namespace NUMINAMATH_CALUDE_factor_expression_l2730_273029

theorem factor_expression (a : ℝ) : 37 * a^2 + 111 * a = 37 * a * (a + 3) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l2730_273029


namespace NUMINAMATH_CALUDE_function_form_from_inequality_l2730_273098

/-- A function satisfying the given inequality property. -/
def SatisfiesInequality (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, |f (x + y) - f (x - y) - y| ≤ y^2

/-- The main theorem stating that a function satisfying the inequality
    must be of the form f(x) = x/2 + c for some constant c. -/
theorem function_form_from_inequality (f : ℝ → ℝ) 
    (h : SatisfiesInequality f) : 
    ∃ c : ℝ, ∀ x : ℝ, f x = x / 2 + c := by
  sorry

end NUMINAMATH_CALUDE_function_form_from_inequality_l2730_273098


namespace NUMINAMATH_CALUDE_flower_pots_height_l2730_273003

/-- Calculates the total vertical distance of stacked flower pots --/
def total_vertical_distance (top_diameter : ℕ) (bottom_diameter : ℕ) (thickness : ℕ) : ℕ :=
  let num_pots := (top_diameter - bottom_diameter) / 2 + 1
  let inner_sum := num_pots * (top_diameter - thickness + bottom_diameter - thickness) / 2
  inner_sum + 2 * thickness

/-- Theorem stating the total vertical distance of the flower pots --/
theorem flower_pots_height : total_vertical_distance 16 4 1 = 65 := by
  sorry

end NUMINAMATH_CALUDE_flower_pots_height_l2730_273003


namespace NUMINAMATH_CALUDE_rope_length_theorem_l2730_273088

/-- Represents a rope that can be folded in a specific manner. -/
structure Rope where
  /-- The distance between points (2) and (3) in the final folding. -/
  distance_2_3 : ℝ
  /-- Assertion that the distance between points (2) and (3) is positive. -/
  distance_positive : distance_2_3 > 0

/-- Calculates the total length of the rope based on its properties. -/
def total_length (rope : Rope) : ℝ :=
  6 * rope.distance_2_3

/-- Theorem stating that for a rope with distance between points (2) and (3) equal to 20,
    the total length is 120. -/
theorem rope_length_theorem (rope : Rope) (h : rope.distance_2_3 = 20) :
  total_length rope = 120 := by
  sorry

end NUMINAMATH_CALUDE_rope_length_theorem_l2730_273088


namespace NUMINAMATH_CALUDE_triangle_properties_l2730_273018

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def condition1 (t : Triangle) : Prop :=
  1 + Real.tan t.C / Real.tan t.B = 2 * t.a / t.b

def condition2 (t : Triangle) : Prop :=
  (t.a + t.b)^2 - t.c^2 = 4

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : condition1 t) (h2 : condition2 t) : 
  t.C = Real.pi / 3 ∧ 
  ∃ (min : ℝ), min = -4 ∧ ∀ (x : ℝ), x ≥ min → 1 / t.b^2 - 3 * t.a ≥ x :=
sorry

end NUMINAMATH_CALUDE_triangle_properties_l2730_273018


namespace NUMINAMATH_CALUDE_solution_satisfies_equation_l2730_273069

-- Define the cube root function
noncomputable def cubeRoot (x : ℝ) : ℝ := Real.rpow x (1/3)

-- Define the equation
def equation (y : ℝ) : Prop :=
  cubeRoot (30 * y + cubeRoot (30 * y + 26)) = 26

-- Theorem statement
theorem solution_satisfies_equation : equation 585 := by
  sorry

end NUMINAMATH_CALUDE_solution_satisfies_equation_l2730_273069


namespace NUMINAMATH_CALUDE_rohans_salary_l2730_273039

/-- Rohan's monthly salary in Rupees -/
def monthly_salary : ℝ := 7500

/-- Percentage of salary spent on food -/
def food_expense_percent : ℝ := 40

/-- Percentage of salary spent on house rent -/
def rent_expense_percent : ℝ := 20

/-- Percentage of salary spent on entertainment -/
def entertainment_expense_percent : ℝ := 10

/-- Percentage of salary spent on conveyance -/
def conveyance_expense_percent : ℝ := 10

/-- Rohan's savings at the end of the month in Rupees -/
def savings : ℝ := 1500

/-- Theorem stating that Rohan's monthly salary is 7500 Rupees -/
theorem rohans_salary :
  monthly_salary = 7500 ∧
  food_expense_percent + rent_expense_percent + entertainment_expense_percent + conveyance_expense_percent = 80 ∧
  savings = monthly_salary * (100 - (food_expense_percent + rent_expense_percent + entertainment_expense_percent + conveyance_expense_percent)) / 100 :=
by sorry

end NUMINAMATH_CALUDE_rohans_salary_l2730_273039


namespace NUMINAMATH_CALUDE_distinct_arrangements_count_l2730_273038

/-- A regular six-pointed star -/
structure SixPointedStar :=
  (points : Fin 12)

/-- The number of symmetries of a regular six-pointed star -/
def star_symmetries : ℕ := 12

/-- The number of distinct arrangements of 12 different objects on a regular six-pointed star,
    where reflections and rotations are considered equivalent -/
def distinct_arrangements : ℕ := Nat.factorial 12 / star_symmetries

theorem distinct_arrangements_count :
  distinct_arrangements = 39916800 := by
  sorry

end NUMINAMATH_CALUDE_distinct_arrangements_count_l2730_273038


namespace NUMINAMATH_CALUDE_ratio_solution_l2730_273077

theorem ratio_solution (x y z a : ℤ) : 
  (∃ (k : ℚ), x = 3 * k ∧ y = 4 * k ∧ z = 7 * k) → 
  y = 24 * a - 12 → 
  a = 2 :=
by sorry

end NUMINAMATH_CALUDE_ratio_solution_l2730_273077


namespace NUMINAMATH_CALUDE_inequality_implies_zero_for_nonpositive_l2730_273050

/-- A function satisfying the given inequality for all real x and y -/
def SatisfiesInequality (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) ≤ y * f x + f (f x)

/-- The main theorem: if f satisfies the inequality, then f(x) = 0 for all x ≤ 0 -/
theorem inequality_implies_zero_for_nonpositive
  (f : ℝ → ℝ) (h : SatisfiesInequality f) :
  ∀ x : ℝ, x ≤ 0 → f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implies_zero_for_nonpositive_l2730_273050


namespace NUMINAMATH_CALUDE_final_price_is_correct_l2730_273034

-- Define the initial price, discounts, and conversion rate
def initial_price : ℝ := 150
def discount1 : ℝ := 0.20
def discount2 : ℝ := 0.15
def discount3 : ℝ := 0.05
def usd_to_inr : ℝ := 75

-- Define the function to calculate the final price
def final_price : ℝ :=
  let price1 := initial_price * (1 - discount1)
  let price2 := price1 * (1 - discount2)
  let price3 := price2 * (1 - discount3)
  price3 * usd_to_inr

-- Theorem statement
theorem final_price_is_correct : final_price = 7267.5 := by
  sorry

end NUMINAMATH_CALUDE_final_price_is_correct_l2730_273034

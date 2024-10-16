import Mathlib

namespace NUMINAMATH_CALUDE_pauls_lost_crayons_l1767_176729

/-- Paul's crayon problem -/
theorem pauls_lost_crayons (initial : ℕ) (given_away : ℕ) (remaining : ℕ) :
  initial = 1453 →
  given_away = 563 →
  remaining = 332 →
  initial - given_away - remaining = 558 := by
  sorry

end NUMINAMATH_CALUDE_pauls_lost_crayons_l1767_176729


namespace NUMINAMATH_CALUDE_exam_students_count_l1767_176762

theorem exam_students_count :
  ∀ (N : ℕ) (T : ℝ),
    N > 5 →
    T = N * 80 →
    (T - 250) / (N - 5 : ℝ) = 90 →
    N = 20 := by
  sorry

end NUMINAMATH_CALUDE_exam_students_count_l1767_176762


namespace NUMINAMATH_CALUDE_smallest_sum_of_consecutive_multiples_l1767_176725

theorem smallest_sum_of_consecutive_multiples : ∃ (a b c : ℕ),
  (b = a + 1) ∧
  (c = a + 2) ∧
  (a % 9 = 0) ∧
  (b % 8 = 0) ∧
  (c % 7 = 0) ∧
  (a + b + c = 1488) ∧
  (∀ (x y z : ℕ), (y = x + 1) ∧ (z = x + 2) ∧ (x % 9 = 0) ∧ (y % 8 = 0) ∧ (z % 7 = 0) → (x + y + z ≥ 1488)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_consecutive_multiples_l1767_176725


namespace NUMINAMATH_CALUDE_quadratic_equation_with_specific_discriminant_l1767_176770

/-- Represents a quadratic equation of the form ax² + bx + c = 0 -/
structure QuadraticEquation (α : Type*) [Field α] where
  a : α
  b : α
  c : α

/-- Calculates the discriminant of a quadratic equation -/
def discriminant {α : Type*} [Field α] (eq : QuadraticEquation α) : α :=
  eq.b ^ 2 - 4 * eq.a * eq.c

/-- Checks if the roots of a quadratic equation are real and unequal -/
def has_real_unequal_roots {α : Type*} [LinearOrderedField α] (eq : QuadraticEquation α) : Prop :=
  discriminant eq > 0

theorem quadratic_equation_with_specific_discriminant 
  (d : ℝ) (eq : QuadraticEquation ℝ) 
  (h1 : eq.a = 3)
  (h2 : eq.b = -6 * Real.sqrt 3)
  (h3 : eq.c = d)
  (h4 : discriminant eq = 12) :
  d = 8 ∧ has_real_unequal_roots eq :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_with_specific_discriminant_l1767_176770


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1767_176744

theorem inequality_solution_set (x : ℝ) : 
  (x - 1) / (x + 3) > (2 * x + 5) / (3 * x + 8) ↔ 
  (x > -3 ∧ x < -8/3) ∨ (x > (3 - Real.sqrt 69) / 2 ∧ x < (3 + Real.sqrt 69) / 2) := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1767_176744


namespace NUMINAMATH_CALUDE_inequality_system_solution_l1767_176771

theorem inequality_system_solution (m n : ℝ) : 
  (∀ x, -1 < x ∧ x < 3 ↔ (x - 3*m < 0 ∧ n - 2*x < 0)) →
  (m + n)^2023 = -1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1767_176771


namespace NUMINAMATH_CALUDE_tennis_ball_storage_l1767_176786

theorem tennis_ball_storage (n : ℕ) : n = 105 ↔ 
  (n % 25 = 5 ∧ n % 20 = 5 ∧ ∀ m : ℕ, m < n → (m % 25 ≠ 5 ∨ m % 20 ≠ 5)) :=
by sorry

end NUMINAMATH_CALUDE_tennis_ball_storage_l1767_176786


namespace NUMINAMATH_CALUDE_inverse_contrapositive_relationship_l1767_176760

theorem inverse_contrapositive_relationship (p q r : Prop) :
  (p → q) ↔ r →
  ((q → p) ↔ q) →
  ((¬q → ¬p) ↔ r) →
  (q ↔ ¬r) :=
by sorry

end NUMINAMATH_CALUDE_inverse_contrapositive_relationship_l1767_176760


namespace NUMINAMATH_CALUDE_exists_modular_inverse_l1767_176715

theorem exists_modular_inverse :
  ∃ n : ℤ, 21 * n ≡ 1 [ZMOD 74] := by
  sorry

end NUMINAMATH_CALUDE_exists_modular_inverse_l1767_176715


namespace NUMINAMATH_CALUDE_unique_age_sum_of_digits_l1767_176702

theorem unique_age_sum_of_digits : ∃! y : ℕ,
  1900 ≤ y ∧ y < 2000 ∧
  1988 - y = 22 ∧
  1988 - y = (y / 1000) + ((y / 100) % 10) + ((y / 10) % 10) + (y % 10) :=
by sorry

end NUMINAMATH_CALUDE_unique_age_sum_of_digits_l1767_176702


namespace NUMINAMATH_CALUDE_sequence_max_value_l1767_176746

theorem sequence_max_value (n : ℕ+) : 
  let a := λ (k : ℕ+) => (k : ℝ) / ((k : ℝ)^2 + 6)
  (∀ k : ℕ+, a k ≤ 1/5) ∧ (∃ k : ℕ+, a k = 1/5) :=
sorry

end NUMINAMATH_CALUDE_sequence_max_value_l1767_176746


namespace NUMINAMATH_CALUDE_quadratic_inequality_properties_l1767_176706

-- Define the quadratic function
def f (a b c x : ℝ) := a * x^2 + b * x + c

-- Define the solution set
def solution_set (a b c : ℝ) := {x : ℝ | f a b c x < 0}

-- State the theorem
theorem quadratic_inequality_properties
  (a b c : ℝ)
  (h : solution_set a b c = {x : ℝ | x < -1 ∨ x > 3}) :
  a < 0 ∧ 
  a + b + c > 0 ∧ 
  solution_set c (-b) a = {x : ℝ | -1/3 < x ∧ x < 1} :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_properties_l1767_176706


namespace NUMINAMATH_CALUDE_triangle_properties_l1767_176753

/-- Represents an acute triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure AcuteTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2
  law_of_sines : a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C

/-- The theorem to be proved -/
theorem triangle_properties (t : AcuteTriangle)
  (h1 : Real.sqrt 3 * t.a = 2 * t.c * Real.sin t.A)
  (h2 : t.c = Real.sqrt 7)
  (h3 : t.a = 2) :
  t.C = π/3 ∧ t.a * t.b * Real.sin t.C / 2 = 3 * Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l1767_176753


namespace NUMINAMATH_CALUDE_intersection_condition_l1767_176778

theorem intersection_condition (m : ℤ) : 
  let A : Set ℤ := {0, m}
  let B : Set ℤ := {n : ℤ | n^2 - 3*n < 0}
  (A ∩ B).Nonempty → m = 1 ∨ m = 2 := by
sorry

end NUMINAMATH_CALUDE_intersection_condition_l1767_176778


namespace NUMINAMATH_CALUDE_bob_age_proof_l1767_176717

theorem bob_age_proof (alice_age bob_age charlie_age : ℕ) : 
  (alice_age + 10 = 2 * (bob_age - 10)) →
  (alice_age = bob_age + 7) →
  (charlie_age = (alice_age + bob_age) / 2) →
  bob_age = 37 := by
  sorry

end NUMINAMATH_CALUDE_bob_age_proof_l1767_176717


namespace NUMINAMATH_CALUDE_sue_answer_for_ben_partner_answer_formula_l1767_176718

/-- Given an initial number, calculate the partner's final answer according to the instructions -/
def partnerAnswer (x : ℤ) : ℤ :=
  (((x + 2) * 3 - 2) * 3)

/-- Theorem stating that for Ben's initial number 6, Sue's answer should be 66 -/
theorem sue_answer_for_ben :
  partnerAnswer 6 = 66 := by sorry

/-- Theorem proving the general formula for the partner's answer -/
theorem partner_answer_formula (x : ℤ) :
  partnerAnswer x = (((x + 2) * 3 - 2) * 3) := by sorry

end NUMINAMATH_CALUDE_sue_answer_for_ben_partner_answer_formula_l1767_176718


namespace NUMINAMATH_CALUDE_triangle_inequality_for_powers_l1767_176773

theorem triangle_inequality_for_powers (a b c : ℝ) :
  (∀ n : ℕ, a^n + b^n > c^n ∧ a^n + c^n > b^n ∧ b^n + c^n > a^n) ↔ 
  ((a = b ∧ a > c) ∨ (a = b ∧ b = c)) :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_for_powers_l1767_176773


namespace NUMINAMATH_CALUDE_initial_men_count_l1767_176733

theorem initial_men_count (M : ℝ) : 
  M * 17 = (M + 320) * 14.010989010989011 → M = 1500 := by
  sorry

end NUMINAMATH_CALUDE_initial_men_count_l1767_176733


namespace NUMINAMATH_CALUDE_sum_reciprocals_bound_l1767_176748

theorem sum_reciprocals_bound (a b c : ℕ) (h : 1 / a + 1 / b + 1 / c < 1) :
  1 / a + 1 / b + 1 / c ≤ 41 / 42 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocals_bound_l1767_176748


namespace NUMINAMATH_CALUDE_hyperbola_inequality_l1767_176705

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 = 1

-- Define the asymptotes
def asymptote (x y : ℝ) : Prop := y = x / 2 ∨ y = -x / 2

-- Define points A and B
def A : ℝ × ℝ := (2, 1)
def B : ℝ × ℝ := (2, -1)

-- Define any point P on the hyperbola
def P (a b : ℝ) : ℝ × ℝ := (2*a + 2*b, a - b)

theorem hyperbola_inequality (a b : ℝ) :
  hyperbola (P a b).1 (P a b).2 →
  |a + b| ≥ 1 := by sorry

end NUMINAMATH_CALUDE_hyperbola_inequality_l1767_176705


namespace NUMINAMATH_CALUDE_beijing_winter_olympics_assignment_schemes_l1767_176736

/-- The number of ways to assign volunteers to events -/
def assignment_schemes (n m : ℕ) : ℕ :=
  (n.choose 2) * m.factorial

/-- Theorem stating the number of assignment schemes for 5 volunteers and 4 events -/
theorem beijing_winter_olympics_assignment_schemes :
  assignment_schemes 5 4 = 240 := by
  sorry

end NUMINAMATH_CALUDE_beijing_winter_olympics_assignment_schemes_l1767_176736


namespace NUMINAMATH_CALUDE_parallelogram_height_l1767_176782

/-- The height of a parallelogram given its area and base -/
theorem parallelogram_height (area : ℝ) (base : ℝ) (height : ℝ) : 
  area = 200 → base = 10 → area = base * height → height = 20 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_height_l1767_176782


namespace NUMINAMATH_CALUDE_a2_value_l1767_176722

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

def geometric_sequence (a b c : ℝ) :=
  b / a = c / b

theorem a2_value (a : ℕ → ℝ) :
  arithmetic_sequence a 2 →
  geometric_sequence (a 1) (a 3) (a 4) →
  a 2 = -6 :=
by
  sorry

end NUMINAMATH_CALUDE_a2_value_l1767_176722


namespace NUMINAMATH_CALUDE_lioness_hyena_age_ratio_l1767_176727

/-- The ratio of a lioness's age to a hyena's age in a park -/
theorem lioness_hyena_age_ratio :
  ∀ (hyena_age : ℕ) (k : ℕ+),
  k * hyena_age = 12 →
  (6 + 5) + (hyena_age / 2 + 5) = 19 →
  (12 : ℚ) / hyena_age = 2 := by
  sorry

end NUMINAMATH_CALUDE_lioness_hyena_age_ratio_l1767_176727


namespace NUMINAMATH_CALUDE_vector_magnitude_l1767_176747

theorem vector_magnitude (m : ℝ) : 
  let a : Fin 2 → ℝ := ![2, -1]
  let b : Fin 2 → ℝ := ![m, 2]
  (∃ (k : ℝ), b = k • a) → 
  ‖a + 2 • b‖ = 3 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_vector_magnitude_l1767_176747


namespace NUMINAMATH_CALUDE_difference_in_balls_l1767_176743

/-- The number of packs of red bouncy balls Jill bought -/
def red_packs : ℕ := 5

/-- The number of packs of yellow bouncy balls Jill bought -/
def yellow_packs : ℕ := 4

/-- The number of bouncy balls in each package -/
def balls_per_pack : ℕ := 18

/-- The total number of red bouncy balls Jill bought -/
def total_red_balls : ℕ := red_packs * balls_per_pack

/-- The total number of yellow bouncy balls Jill bought -/
def total_yellow_balls : ℕ := yellow_packs * balls_per_pack

theorem difference_in_balls : total_red_balls - total_yellow_balls = 18 := by
  sorry

end NUMINAMATH_CALUDE_difference_in_balls_l1767_176743


namespace NUMINAMATH_CALUDE_integral_equation_solution_l1767_176749

theorem integral_equation_solution (k : ℝ) : (∫ x in (0:ℝ)..1, (3 * x^2 + k)) = 10 ↔ k = 9 := by
  sorry

end NUMINAMATH_CALUDE_integral_equation_solution_l1767_176749


namespace NUMINAMATH_CALUDE_floor_of_4_8_l1767_176775

theorem floor_of_4_8 : ⌊(4.8 : ℝ)⌋ = 4 := by sorry

end NUMINAMATH_CALUDE_floor_of_4_8_l1767_176775


namespace NUMINAMATH_CALUDE_cube_edge_length_l1767_176712

theorem cube_edge_length (a : ℝ) :
  (6 * a^2 = a^3 → a = 6) ∧
  (6 * a^2 = (a^3)^2 → a = Real.rpow 6 (1/4)) ∧
  ((6 * a^2)^3 = a^3 → a = 1/36) :=
by sorry

end NUMINAMATH_CALUDE_cube_edge_length_l1767_176712


namespace NUMINAMATH_CALUDE_smallest_d_for_3150_perfect_square_l1767_176795

theorem smallest_d_for_3150_perfect_square : 
  ∃ (d : ℕ), d > 0 ∧ d = 14 ∧ 
  (∃ (n : ℕ), 3150 * d = n^2) ∧
  (∀ (k : ℕ), k > 0 → k < d → ¬∃ (m : ℕ), 3150 * k = m^2) := by
  sorry

end NUMINAMATH_CALUDE_smallest_d_for_3150_perfect_square_l1767_176795


namespace NUMINAMATH_CALUDE_gcd_182_98_l1767_176714

theorem gcd_182_98 : Nat.gcd 182 98 = 14 := by
  sorry

end NUMINAMATH_CALUDE_gcd_182_98_l1767_176714


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l1767_176728

theorem quadratic_equation_properties (a b c : ℝ) (h : a ≠ 0) :
  -- Statement ①
  (a + b + c = 0 → b^2 - 4*a*c ≥ 0) ∧
  -- Statement ②
  (∃ x y : ℝ, x ≠ y ∧ a*x^2 + c = 0 ∧ a*y^2 + c = 0 →
    ∃ u v : ℝ, u ≠ v ∧ a*u^2 + b*u + c = 0 ∧ a*v^2 + b*v + c = 0) ∧
  -- Statement ④
  (∀ x₀ : ℝ, a*x₀^2 + b*x₀ + c = 0 → b^2 - 4*a*c = (2*a*x₀ + b)^2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l1767_176728


namespace NUMINAMATH_CALUDE_smallest_integer_solution_l1767_176741

-- Define the custom operation
def custom_op (x y : ℚ) : ℚ := (x * y / 3) - 2 * y

-- Theorem statement
theorem smallest_integer_solution :
  ∀ a : ℤ, (custom_op 2 (↑a) ≤ 2) → (a ≥ -1) ∧ 
  ∀ b : ℤ, (b < -1) → (custom_op 2 (↑b) > 2) :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_solution_l1767_176741


namespace NUMINAMATH_CALUDE_fourteen_trucks_sufficient_l1767_176708

/-- Represents the number of packages of each size -/
structure PackageDistribution where
  small : ℕ
  medium : ℕ
  large : ℕ

/-- Represents the capacity of a Type B truck for each package size -/
structure TruckCapacity where
  small : ℕ
  medium : ℕ
  large : ℕ

/-- Calculates the number of Type B trucks needed given a package distribution and truck capacity -/
def trucksNeeded (packages : PackageDistribution) (capacity : TruckCapacity) : ℕ :=
  let smallTrucks := (packages.small + capacity.small - 1) / capacity.small
  let mediumTrucks := (packages.medium + capacity.medium - 1) / capacity.medium
  let largeTrucks := (packages.large + capacity.large - 1) / capacity.large
  smallTrucks + mediumTrucks + largeTrucks

/-- Theorem stating that 14 Type B trucks are sufficient for the given package distribution -/
theorem fourteen_trucks_sufficient 
  (packages : PackageDistribution)
  (capacity : TruckCapacity)
  (h1 : packages.small + packages.medium + packages.large = 1000)
  (h2 : packages.small = 2 * packages.medium)
  (h3 : packages.medium = 3 * packages.large)
  (h4 : capacity.small = 90)
  (h5 : capacity.medium = 60)
  (h6 : capacity.large = 50) :
  trucksNeeded packages capacity ≤ 14 :=
sorry

end NUMINAMATH_CALUDE_fourteen_trucks_sufficient_l1767_176708


namespace NUMINAMATH_CALUDE_arctan_sum_three_seven_l1767_176732

theorem arctan_sum_three_seven : Real.arctan (3/7) + Real.arctan (7/3) = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_three_seven_l1767_176732


namespace NUMINAMATH_CALUDE_white_ball_count_l1767_176769

theorem white_ball_count (total : ℕ) (white blue red : ℕ) : 
  total = 1000 →
  blue = white + 14 →
  red = 3 * (blue - white) →
  total = white + blue + red →
  white = 472 := by
  sorry

end NUMINAMATH_CALUDE_white_ball_count_l1767_176769


namespace NUMINAMATH_CALUDE_madeline_unused_crayons_l1767_176758

theorem madeline_unused_crayons :
  let box1to3 := 3 * 30 * (1/2 : ℚ)
  let box4to5 := 2 * 36 * (3/4 : ℚ)
  let box6to7 := 2 * 40 * (2/5 : ℚ)
  let box8 := 1 * 45 * (5/9 : ℚ)
  let box9to10 := 2 * 48 * (7/8 : ℚ)
  let box11 := 1 * 27 * (5/6 : ℚ)
  let box12 := 1 * 54 * (1/2 : ℚ)
  let total_unused := box1to3 + box4to5 + box6to7 + box8 + box9to10 + box11 + box12
  ⌊total_unused⌋ = 289 :=
by sorry

end NUMINAMATH_CALUDE_madeline_unused_crayons_l1767_176758


namespace NUMINAMATH_CALUDE_denominator_problem_l1767_176703

theorem denominator_problem (numerator denominator : ℤ) : 
  denominator = numerator - 4 →
  numerator + 6 = 3 * denominator →
  denominator = 5 := by
sorry

end NUMINAMATH_CALUDE_denominator_problem_l1767_176703


namespace NUMINAMATH_CALUDE_inverse_composition_equals_six_l1767_176704

-- Define the function f
def f : ℕ → ℕ
| 1 => 4
| 2 => 6
| 3 => 2
| 4 => 1
| 5 => 5
| 6 => 3
| _ => 0  -- Default case for other inputs

-- Define the inverse function f⁻¹
def f_inv : ℕ → ℕ
| 1 => 4
| 2 => 3
| 3 => 6
| 4 => 1
| 5 => 5
| 6 => 2
| _ => 0  -- Default case for other inputs

-- State the theorem
theorem inverse_composition_equals_six :
  f_inv (f_inv (f_inv 6)) = 6 := by sorry

end NUMINAMATH_CALUDE_inverse_composition_equals_six_l1767_176704


namespace NUMINAMATH_CALUDE_quadratic_root_transformation_l1767_176791

theorem quadratic_root_transformation (a b c r s : ℝ) : 
  (∀ x, a * x^2 + b * x + c = 0 ↔ x = r ∨ x = s) →
  (∀ y, y^2 - b * y + 4 * a * c = 0 ↔ y = 2 * a * r + b ∨ y = 2 * a * s + b) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_transformation_l1767_176791


namespace NUMINAMATH_CALUDE_memory_card_picture_size_l1767_176720

/-- Represents a memory card with a given capacity and picture storage capabilities. -/
structure MemoryCard where
  capacity : ℕ  -- Total capacity in megabytes
  large_pics : ℕ  -- Number of large pictures it can hold
  small_pics : ℕ  -- Number of small pictures it can hold
  small_pic_size : ℕ  -- Size of small pictures in megabytes

/-- Calculates the size of pictures when the card is filled with large pictures. -/
def large_pic_size (card : MemoryCard) : ℕ :=
  card.capacity / card.large_pics

theorem memory_card_picture_size (card : MemoryCard) 
  (h1 : card.small_pics = 3000)
  (h2 : card.large_pics = 4000)
  (h3 : large_pic_size card = 6) :
  card.small_pic_size = 8 := by
  sorry

#check memory_card_picture_size

end NUMINAMATH_CALUDE_memory_card_picture_size_l1767_176720


namespace NUMINAMATH_CALUDE_game_cost_proof_l1767_176745

/-- The cost of a video game that Ronald and Max want to buy --/
def game_cost : ℕ := 60

/-- The price of each ice cream --/
def ice_cream_price : ℕ := 5

/-- The number of ice creams they need to sell to afford the game --/
def ice_creams_needed : ℕ := 24

/-- The number of people splitting the cost of the game --/
def people_splitting_cost : ℕ := 2

/-- Theorem stating that the game cost is correct given the conditions --/
theorem game_cost_proof : 
  game_cost = (ice_cream_price * ice_creams_needed) / people_splitting_cost :=
by sorry

end NUMINAMATH_CALUDE_game_cost_proof_l1767_176745


namespace NUMINAMATH_CALUDE_martin_ticket_count_l1767_176761

/-- The number of tickets Martin bought at full price -/
def full_price_tickets : ℕ := sorry

/-- The price of a full-price ticket in cents -/
def full_price : ℕ := 200

/-- The number of discounted tickets Martin bought -/
def discounted_tickets : ℕ := 4

/-- The price of a discounted ticket in cents -/
def discounted_price : ℕ := 160

/-- The total amount Martin spent in cents -/
def total_spent : ℕ := 1840

theorem martin_ticket_count :
  full_price_tickets * full_price + discounted_tickets * discounted_price = total_spent ∧
  full_price_tickets + discounted_tickets = 10 :=
sorry

end NUMINAMATH_CALUDE_martin_ticket_count_l1767_176761


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l1767_176716

theorem polynomial_coefficient_sum : 
  ∀ A B C D : ℝ, 
  (∀ x : ℝ, (x - 3) * (4 * x^2 + 2 * x - 7) = A * x^3 + B * x^2 + C * x + D) →
  A + B + C + D = 2 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l1767_176716


namespace NUMINAMATH_CALUDE_strawberry_picker_l1767_176737

/-- Given three people picking strawberries, proves that one person picked 200 strawberries -/
theorem strawberry_picker (total jonathan_matthew matthew_zac : ℕ) 
  (h_total : total = 550)
  (h_jonathan_matthew : jonathan_matthew = 350)
  (h_matthew_zac : matthew_zac = 250) :
  ∃ (jonathan matthew zac : ℕ),
    jonathan + matthew + zac = total ∧
    jonathan + matthew = jonathan_matthew ∧
    matthew + zac = matthew_zac ∧
    zac = 200 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_picker_l1767_176737


namespace NUMINAMATH_CALUDE_annulus_chord_circle_area_equality_l1767_176735

theorem annulus_chord_circle_area_equality (R r x : ℝ) (h1 : 0 < r) (h2 : r < R) (h3 : R^2 = r^2 + x^2) :
  π * x^2 = π * (R^2 - r^2) :=
by sorry

end NUMINAMATH_CALUDE_annulus_chord_circle_area_equality_l1767_176735


namespace NUMINAMATH_CALUDE_pizza_fraction_l1767_176765

theorem pizza_fraction (total_slices : ℕ) (whole_slice : ℚ) (shared_slice : ℚ) :
  total_slices = 12 →
  whole_slice = 1 →
  shared_slice = 1 / 2 →
  (whole_slice + shared_slice) / total_slices = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_pizza_fraction_l1767_176765


namespace NUMINAMATH_CALUDE_weekend_sleep_calculation_l1767_176759

/-- Calculates the number of hours slept during weekends per day, given the total weekly sleep and weekday sleep hours. -/
def weekend_sleep_hours (total_weekly_sleep : ℕ) (weekday_sleep : ℕ) : ℕ :=
  (total_weekly_sleep - 5 * weekday_sleep) / 2

/-- Proves that given 7 hours of sleep each weekday and 51 hours of total weekly sleep, 
    the person sleeps 8 hours each day during the weekend. -/
theorem weekend_sleep_calculation :
  weekend_sleep_hours 51 7 = 8 := by
  sorry

end NUMINAMATH_CALUDE_weekend_sleep_calculation_l1767_176759


namespace NUMINAMATH_CALUDE_security_guard_schedule_l1767_176701

structure Guard where
  id : Nat
  hours : Nat

def valid_schedule (g2 g3 g4 g5 : Guard) : Prop :=
  g2.id = 2 ∧ g3.id = 3 ∧ g4.id = 4 ∧ g5.id = 5 ∧
  g2.hours + g3.hours + g4.hours + g5.hours = 6 ∧
  g2.hours ≤ 2 ∧
  g3.hours ≤ 3 ∧
  g4.hours = g5.hours + 1 ∧
  g5.hours > 0

theorem security_guard_schedule :
  ∃ (g2 g3 g4 g5 : Guard), valid_schedule g2 g3 g4 g5 :=
sorry

end NUMINAMATH_CALUDE_security_guard_schedule_l1767_176701


namespace NUMINAMATH_CALUDE_workshop_average_salary_l1767_176756

theorem workshop_average_salary 
  (total_workers : ℕ) 
  (num_technicians : ℕ) 
  (avg_salary_technicians : ℕ) 
  (avg_salary_others : ℕ) 
  (h1 : total_workers = 28) 
  (h2 : num_technicians = 7) 
  (h3 : avg_salary_technicians = 14000) 
  (h4 : avg_salary_others = 6000) :
  (num_technicians * avg_salary_technicians + 
   (total_workers - num_technicians) * avg_salary_others) / total_workers = 8000 :=
by sorry

end NUMINAMATH_CALUDE_workshop_average_salary_l1767_176756


namespace NUMINAMATH_CALUDE_quadratic_solution_and_max_product_l1767_176793

-- Define the quadratic inequality
def quadratic_inequality (x m : ℝ) : Prop := x^2 - 3*x + m < 0

-- Define the solution set
def solution_set (x n : ℝ) : Prop := 1 < x ∧ x < n

-- Define the constraint for a and b
def constraint (m n a b : ℝ) : Prop := m*a + 2*n*b = 3

-- Theorem statement
theorem quadratic_solution_and_max_product :
  ∃ (m n : ℝ),
    (∀ x, quadratic_inequality x m ↔ solution_set x n) ∧
    (m = 2 ∧ n = 2) ∧
    (∀ a b : ℝ, a > 0 → b > 0 → constraint m n a b →
      a * b ≤ 9/32 ∧ ∃ a₀ b₀, a₀ > 0 ∧ b₀ > 0 ∧ constraint m n a₀ b₀ ∧ a₀ * b₀ = 9/32) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_solution_and_max_product_l1767_176793


namespace NUMINAMATH_CALUDE_train_passing_time_train_passing_time_proof_l1767_176755

/-- Calculates the time taken for two trains to pass each other --/
theorem train_passing_time (train_length : ℝ) (speed_fast : ℝ) (speed_slow : ℝ) : ℝ :=
  let speed_fast_ms := speed_fast * 1000 / 3600
  let speed_slow_ms := speed_slow * 1000 / 3600
  let relative_speed := speed_fast_ms + speed_slow_ms
  train_length / relative_speed

/-- Proves that the time taken for the slower train to pass the driver of the faster train is approximately 18 seconds --/
theorem train_passing_time_proof :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |train_passing_time 475 55 40 - 18| < ε :=
sorry

end NUMINAMATH_CALUDE_train_passing_time_train_passing_time_proof_l1767_176755


namespace NUMINAMATH_CALUDE_smallest_multiple_of_42_and_56_not_18_l1767_176780

theorem smallest_multiple_of_42_and_56_not_18 : 
  ∃ (n : ℕ), n > 0 ∧ 42 ∣ n ∧ 56 ∣ n ∧ ¬(18 ∣ n) ∧
  ∀ (m : ℕ), m > 0 → 42 ∣ m → 56 ∣ m → ¬(18 ∣ m) → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_42_and_56_not_18_l1767_176780


namespace NUMINAMATH_CALUDE_frog_jump_probability_l1767_176751

/-- Represents a point on the grid -/
structure Point where
  x : ℕ
  y : ℕ

/-- Represents the rectangle boundary -/
def Rectangle := {p : Point | p.x = 0 ∨ p.x = 5 ∨ p.y = 0 ∨ p.y = 5}

/-- Represents a vertical side of the rectangle -/
def VerticalSide := {p : Point | p.x = 0 ∨ p.x = 5}

/-- The probability of ending on a vertical side starting from a given point -/
def probabilityVerticalSide (p : Point) : ℚ := sorry

/-- The frog's starting point -/
def startPoint : Point := ⟨2, 3⟩

/-- Theorem stating the probability of ending on a vertical side -/
theorem frog_jump_probability : probabilityVerticalSide startPoint = 2/3 := by sorry

end NUMINAMATH_CALUDE_frog_jump_probability_l1767_176751


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_and_necessary_not_sufficient_l1767_176783

theorem sufficient_not_necessary_and_necessary_not_sufficient :
  (∀ a : ℝ, a > 1 → 1 / a < 1) ∧
  (∃ a : ℝ, 1 / a < 1 ∧ a ≤ 1) ∧
  (∀ a b : ℝ, a * b ≠ 0 → a ≠ 0) ∧
  (∃ a b : ℝ, a ≠ 0 ∧ a * b = 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_and_necessary_not_sufficient_l1767_176783


namespace NUMINAMATH_CALUDE_largest_multiple_of_9_less_than_120_l1767_176789

theorem largest_multiple_of_9_less_than_120 : 
  ∀ n : ℕ, n % 9 = 0 → n < 120 → n ≤ 117 :=
sorry

end NUMINAMATH_CALUDE_largest_multiple_of_9_less_than_120_l1767_176789


namespace NUMINAMATH_CALUDE_rational_equation_solution_l1767_176768

theorem rational_equation_solution (A B : ℚ) :
  (∀ x : ℚ, x ≠ 3 ∧ x ≠ 5 → (B * x - 13) / (x^2 - 8*x + 15) = A / (x - 3) + 4 / (x - 5)) →
  A + B = 22 / 5 := by
sorry

end NUMINAMATH_CALUDE_rational_equation_solution_l1767_176768


namespace NUMINAMATH_CALUDE_random_events_l1767_176777

-- Define a type for the events
inductive Event
  | addition
  | subtraction
  | multiplication
  | division

-- Define a function to check if an event is random
def is_random (e : Event) : Prop :=
  match e with
  | Event.addition => ∃ (a b : ℝ), a * b < 0 ∧ a + b < 0
  | Event.subtraction => ∃ (a b : ℝ), a * b < 0 ∧ a - b > 0
  | Event.multiplication => false
  | Event.division => true

-- Theorem stating which events are random
theorem random_events :
  (is_random Event.addition) ∧
  (is_random Event.subtraction) ∧
  (¬ is_random Event.multiplication) ∧
  (¬ is_random Event.division) := by
  sorry

end NUMINAMATH_CALUDE_random_events_l1767_176777


namespace NUMINAMATH_CALUDE_min_solutions_in_interval_l1767_176730

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem min_solutions_in_interval 
  (f : ℝ → ℝ) 
  (h_even : is_even f) 
  (h_period : has_period f 3) 
  (h_root : f 2 = 0) : 
  ∃ (a b c d : ℝ), 0 < a ∧ a < b ∧ b < c ∧ c < d ∧ d < 6 ∧ 
    f a = 0 ∧ f b = 0 ∧ f c = 0 ∧ f d = 0 :=
sorry

end NUMINAMATH_CALUDE_min_solutions_in_interval_l1767_176730


namespace NUMINAMATH_CALUDE_major_axis_length_is_6_l1767_176710

/-- An ellipse with given properties -/
structure Ellipse where
  /-- The ellipse is tangent to the y-axis -/
  tangent_y_axis : Bool
  /-- The ellipse is tangent to the line x = 4 -/
  tangent_x_4 : Bool
  /-- The x-coordinate of both foci -/
  foci_x : ℝ
  /-- The y-coordinates of the foci -/
  foci_y1 : ℝ
  foci_y2 : ℝ

/-- The length of the major axis of the ellipse -/
def majorAxisLength (e : Ellipse) : ℝ := sorry

/-- Theorem stating that the length of the major axis is 6 -/
theorem major_axis_length_is_6 (e : Ellipse) 
  (h1 : e.tangent_y_axis = true) 
  (h2 : e.tangent_x_4 = true)
  (h3 : e.foci_x = 3)
  (h4 : e.foci_y1 = 1 + Real.sqrt 3)
  (h5 : e.foci_y2 = 1 - Real.sqrt 3) : 
  majorAxisLength e = 6 := by sorry

end NUMINAMATH_CALUDE_major_axis_length_is_6_l1767_176710


namespace NUMINAMATH_CALUDE_quadratic_minimum_l1767_176792

theorem quadratic_minimum (m : ℝ) (hm : m ≠ 0) :
  (∀ x ∈ Set.Icc (-2) 2, m * x^2 - 2 * m * x + 2 ≥ -2) ∧
  (∃ x ∈ Set.Ioc (-2) 2, m * x^2 - 2 * m * x + 2 = -2) →
  m = 4 ∨ m = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l1767_176792


namespace NUMINAMATH_CALUDE_not_closed_set_1_not_closed_set_2_closed_set_3_exist_closed_sets_union_not_closed_l1767_176784

-- Definition of a closed set
def is_closed_set (M : Set Int) : Prop :=
  ∀ a b : Int, a ∈ M → b ∈ M → (a + b) ∈ M ∧ (a - b) ∈ M

-- Statement 1
theorem not_closed_set_1 : ¬ is_closed_set {-4, -2, 0, 2, 4} := by sorry

-- Statement 2
def positive_integers : Set Int := {n | n > 0}

theorem not_closed_set_2 : ¬ is_closed_set positive_integers := by sorry

-- Statement 3
def multiples_of_three : Set Int := {n | ∃ k : Int, n = 3 * k}

theorem closed_set_3 : is_closed_set multiples_of_three := by sorry

-- Statement 4
theorem exist_closed_sets_union_not_closed :
  ∃ A₁ A₂ : Set Int, is_closed_set A₁ ∧ is_closed_set A₂ ∧ ¬ is_closed_set (A₁ ∪ A₂) := by sorry

end NUMINAMATH_CALUDE_not_closed_set_1_not_closed_set_2_closed_set_3_exist_closed_sets_union_not_closed_l1767_176784


namespace NUMINAMATH_CALUDE_circumcircle_radius_of_triangle_l1767_176750

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin x * Real.cos x + Real.cos x ^ 2 + 1

theorem circumcircle_radius_of_triangle (A B C : ℝ) (a b c : ℝ) :
  (∀ x, f x = Real.sqrt 3 * Real.sin x * Real.cos x + Real.cos x ^ 2 + 1) →
  (a > 0 ∧ b > 0 ∧ c > 0) →
  (f C = 2) →
  (a + b = 4) →
  (1/2 * a * b * Real.sin C = Real.sqrt 3 / 3) →
  (∃ R : ℝ, R = c / (2 * Real.sin C) ∧ R = 2) :=
by sorry

end NUMINAMATH_CALUDE_circumcircle_radius_of_triangle_l1767_176750


namespace NUMINAMATH_CALUDE_vitya_wins_l1767_176731

/-- Represents a point on the infinite grid --/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- Represents the game state --/
structure GameState where
  marked_points : List GridPoint
  current_player : Bool  -- true for Kolya, false for Vitya

/-- Checks if a list of points forms a convex polygon --/
def is_convex_polygon (points : List GridPoint) : Prop :=
  sorry

/-- Checks if a move is valid according to the game rules --/
def is_valid_move (state : GameState) (new_point : GridPoint) : Prop :=
  is_convex_polygon (new_point :: state.marked_points)

/-- Represents a strategy for playing the game --/
def Strategy := GameState → Option GridPoint

/-- Checks if a strategy is winning for a player --/
def is_winning_strategy (strategy : Strategy) (player : Bool) : Prop :=
  sorry

theorem vitya_wins :
  ∃ (strategy : Strategy), is_winning_strategy strategy false :=
sorry

end NUMINAMATH_CALUDE_vitya_wins_l1767_176731


namespace NUMINAMATH_CALUDE_five_balls_four_boxes_l1767_176724

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 68 ways to distribute 5 indistinguishable balls into 4 distinguishable boxes -/
theorem five_balls_four_boxes : distribute_balls 5 4 = 68 := by sorry

end NUMINAMATH_CALUDE_five_balls_four_boxes_l1767_176724


namespace NUMINAMATH_CALUDE_lemonade_sales_calculation_l1767_176742

/-- Calculates the total sales for lemonade glasses sold over two days -/
theorem lemonade_sales_calculation (price_per_glass : ℚ) (saturday_sales sunday_sales : ℕ) :
  price_per_glass = 25 / 100 →
  saturday_sales = 41 →
  sunday_sales = 53 →
  (saturday_sales + sunday_sales : ℚ) * price_per_glass = 2350 / 100 := by
  sorry

#eval (41 + 53 : ℚ) * (25 / 100) -- Optional: to verify the result

end NUMINAMATH_CALUDE_lemonade_sales_calculation_l1767_176742


namespace NUMINAMATH_CALUDE_unknown_number_proof_l1767_176779

theorem unknown_number_proof (X : ℕ) : 
  1000 + X + 1000 + 30 + 1000 + 40 + 1000 + 10 = 4100 → X = 20 := by
  sorry

end NUMINAMATH_CALUDE_unknown_number_proof_l1767_176779


namespace NUMINAMATH_CALUDE_problem_solution_l1767_176740

theorem problem_solution : (150 * (150 - 4)) / (150 * 150 * 2 - 4) = 21900 / 44996 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1767_176740


namespace NUMINAMATH_CALUDE_min_value_quadratic_l1767_176738

theorem min_value_quadratic (x : ℝ) : 
  ∃ (min_y : ℝ), ∀ (y : ℝ), y = 2 * x^2 - 8 * x + 10 → y ≥ min_y ∧ min_y = 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l1767_176738


namespace NUMINAMATH_CALUDE_sqrt_3_times_sqrt_12_l1767_176721

theorem sqrt_3_times_sqrt_12 : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_times_sqrt_12_l1767_176721


namespace NUMINAMATH_CALUDE_case_one_solutions_case_two_no_solution_l1767_176794

-- Case 1
theorem case_one_solutions (a b : ℝ) (A : ℝ) (ha : a = 14) (hb : b = 16) (hA : A = 45 * π / 180) :
  ∃! (B C : ℝ), 0 < B ∧ 0 < C ∧ A + B + C = π ∧ 
  a / Real.sin A = b / Real.sin B ∧
  a / Real.sin A = Real.sqrt (a^2 + b^2 - 2*a*b*Real.cos C) / Real.sin C :=
sorry

-- Case 2
theorem case_two_no_solution (a b : ℝ) (B : ℝ) (ha : a = 60) (hb : b = 48) (hB : B = 60 * π / 180) :
  ¬ ∃ (A C : ℝ), 0 < A ∧ 0 < C ∧ A + B + C = π ∧
  a / Real.sin A = b / Real.sin B ∧
  a / Real.sin A = Real.sqrt (a^2 + b^2 - 2*a*b*Real.cos C) / Real.sin C :=
sorry

end NUMINAMATH_CALUDE_case_one_solutions_case_two_no_solution_l1767_176794


namespace NUMINAMATH_CALUDE_seokjin_class_size_l1767_176719

/-- The number of students in Taehyung's class -/
def taehyung_class : ℕ := 35

/-- The number of students in Jimin's class -/
def jimin_class : ℕ := taehyung_class - 3

/-- The number of students in Seokjin's class -/
def seokjin_class : ℕ := jimin_class + 2

theorem seokjin_class_size : seokjin_class = 34 := by
  sorry

end NUMINAMATH_CALUDE_seokjin_class_size_l1767_176719


namespace NUMINAMATH_CALUDE_snack_pack_suckers_l1767_176752

/-- The number of suckers needed for snack packs --/
def suckers_needed (pretzels : ℕ) (goldfish_multiplier : ℕ) (kids : ℕ) (items_per_baggie : ℕ) : ℕ :=
  kids * items_per_baggie - (pretzels + goldfish_multiplier * pretzels)

theorem snack_pack_suckers :
  suckers_needed 64 4 16 22 = 32 := by
  sorry

end NUMINAMATH_CALUDE_snack_pack_suckers_l1767_176752


namespace NUMINAMATH_CALUDE_z_is_negative_intercept_l1767_176754

/-- Represents a linear equation in two variables -/
structure LinearEquation where
  slope : ℝ
  intercept : ℝ

/-- Converts an objective function z = ax - y to a linear equation y = ax - z -/
def objectiveFunctionToLinearEquation (a : ℝ) (z : ℝ) : LinearEquation :=
  { slope := a, intercept := -z }

/-- Theorem: In the equation z = 3x - y, z represents the negative of the vertical intercept -/
theorem z_is_negative_intercept (z : ℝ) :
  let eq := objectiveFunctionToLinearEquation 3 z
  eq.intercept = -z := by sorry

end NUMINAMATH_CALUDE_z_is_negative_intercept_l1767_176754


namespace NUMINAMATH_CALUDE_rectangle_max_area_l1767_176757

theorem rectangle_max_area (x y P D : ℝ) (h1 : P = 2*x + 2*y) (h2 : D^2 = x^2 + y^2) 
  (h3 : P = 14) (h4 : D = 5) :
  ∃ (A : ℝ), A = x * y ∧ A ≤ 49/4 ∧ ∃ (x₀ y₀ : ℝ), x₀ * y₀ = 49/4 := by
  sorry

#check rectangle_max_area

end NUMINAMATH_CALUDE_rectangle_max_area_l1767_176757


namespace NUMINAMATH_CALUDE_expression_value_l1767_176709

theorem expression_value (a x : ℝ) (h : a^(2*x) = Real.sqrt 2 - 1) :
  (a^(3*x) + a^(-3*x)) / (a^x + a^(-x)) = 2 * Real.sqrt 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1767_176709


namespace NUMINAMATH_CALUDE_simplify_expression_l1767_176707

theorem simplify_expression (y : ℝ) : 2 - (2 * (1 - (3 - (2 * (2 - y))))) = -2 + 4 * y := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1767_176707


namespace NUMINAMATH_CALUDE_lap_time_improvement_is_two_thirds_l1767_176796

/-- Represents running data with number of laps and total time in minutes -/
structure RunningData where
  laps : ℕ
  time : ℚ

/-- Calculates the lap time in minutes for given running data -/
def lapTime (data : RunningData) : ℚ :=
  data.time / data.laps

/-- The initial running data -/
def initialData : RunningData :=
  { laps := 15, time := 45 }

/-- The final running data after training -/
def finalData : RunningData :=
  { laps := 18, time := 42 }

/-- The improvement in lap time -/
def lapTimeImprovement : ℚ :=
  lapTime initialData - lapTime finalData

theorem lap_time_improvement_is_two_thirds :
  lapTimeImprovement = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_lap_time_improvement_is_two_thirds_l1767_176796


namespace NUMINAMATH_CALUDE_cube_root_opposite_zero_l1767_176763

theorem cube_root_opposite_zero :
  ∀ x : ℝ, (x^(1/3) = -x) ↔ (x = 0) :=
sorry

end NUMINAMATH_CALUDE_cube_root_opposite_zero_l1767_176763


namespace NUMINAMATH_CALUDE_perfect_square_quadratic_l1767_176700

theorem perfect_square_quadratic (m : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, x^2 + m*x + 16 = y^2) → (m = 8 ∨ m = -8) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_quadratic_l1767_176700


namespace NUMINAMATH_CALUDE_harold_marbles_l1767_176726

/-- Given that Harold had 100 marbles, shared them evenly among 5 friends,
    and each friend received 16 marbles, prove that Harold kept 20 marbles. -/
theorem harold_marbles :
  ∀ (total_marbles friends_count marbles_per_friend marbles_kept : ℕ),
    total_marbles = 100 →
    friends_count = 5 →
    marbles_per_friend = 16 →
    marbles_kept + (friends_count * marbles_per_friend) = total_marbles →
    marbles_kept = 20 :=
by sorry

end NUMINAMATH_CALUDE_harold_marbles_l1767_176726


namespace NUMINAMATH_CALUDE_sugar_solution_percentage_l1767_176711

/-- Proves that a replacing solution must be 40% sugar by weight given the conditions of the problem. -/
theorem sugar_solution_percentage (original_percentage : ℝ) (replaced_fraction : ℝ) (final_percentage : ℝ) :
  original_percentage = 8 →
  replaced_fraction = 1 / 4 →
  final_percentage = 16 →
  (1 - replaced_fraction) * original_percentage + replaced_fraction * (100 : ℝ) * final_percentage / 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_sugar_solution_percentage_l1767_176711


namespace NUMINAMATH_CALUDE_at_least_one_fraction_less_than_one_l1767_176788

theorem at_least_one_fraction_less_than_one 
  (x y : ℝ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (hxy : y - x > 1) : 
  (1 - y) / x < 1 ∨ (1 + 3 * x) / y < 1 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_fraction_less_than_one_l1767_176788


namespace NUMINAMATH_CALUDE_symmetric_points_line_equation_l1767_176723

/-- Given two points A and B that are symmetric with respect to a line l,
    prove that the equation of line l is 3x + y + 4 = 0 --/
theorem symmetric_points_line_equation (A B : ℝ × ℝ) (l : Set (ℝ × ℝ)) : 
  A = (1, 3) →
  B = (-5, 1) →
  (∀ (P : ℝ × ℝ), P ∈ l ↔ dist P A = dist P B) →
  (∀ (x y : ℝ), (x, y) ∈ l ↔ 3 * x + y + 4 = 0) :=
by sorry


end NUMINAMATH_CALUDE_symmetric_points_line_equation_l1767_176723


namespace NUMINAMATH_CALUDE_one_way_ticket_cost_l1767_176790

-- Define the cost of a 30-day pass
def pass_cost : ℝ := 50

-- Define the minimum number of rides for the pass to be cheaper
def min_rides : ℕ := 26

-- Define the cost of a one-way ticket
def ticket_cost : ℝ := 2

-- Theorem statement
theorem one_way_ticket_cost :
  (pass_cost / min_rides < ticket_cost) ∧
  (∀ x : ℝ, x > 0 ∧ x < ticket_cost → pass_cost / min_rides ≥ x) := by
  sorry

end NUMINAMATH_CALUDE_one_way_ticket_cost_l1767_176790


namespace NUMINAMATH_CALUDE_outfits_count_l1767_176799

/-- The number of shirts available. -/
def num_shirts : ℕ := 6

/-- The number of ties available. -/
def num_ties : ℕ := 5

/-- The number of pants available. -/
def num_pants : ℕ := 4

/-- The total number of possible outfits. -/
def total_outfits : ℕ := num_shirts * num_ties * num_pants

/-- Theorem stating that the total number of outfits is 120. -/
theorem outfits_count : total_outfits = 120 := by
  sorry

end NUMINAMATH_CALUDE_outfits_count_l1767_176799


namespace NUMINAMATH_CALUDE_sum_of_digits_of_power_l1767_176766

-- Define a function to get the tens digit of a natural number
def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

-- Define a function to get the ones digit of a natural number
def ones_digit (n : ℕ) : ℕ := n % 10

-- State the theorem
theorem sum_of_digits_of_power : 
  tens_digit ((3 + 4)^17) + ones_digit ((3 + 4)^17) = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_power_l1767_176766


namespace NUMINAMATH_CALUDE_ball_cost_price_l1767_176764

theorem ball_cost_price (cost : ℕ → ℝ) (h1 : cost 11 - 720 = cost 5) : cost 1 = 120 := by
  sorry

end NUMINAMATH_CALUDE_ball_cost_price_l1767_176764


namespace NUMINAMATH_CALUDE_product_xyz_is_zero_l1767_176774

theorem product_xyz_is_zero 
  (x y z : ℝ) 
  (h1 : x + 2/y = 2) 
  (h2 : y + 2/z = 2) : 
  x * y * z = 0 := by
sorry

end NUMINAMATH_CALUDE_product_xyz_is_zero_l1767_176774


namespace NUMINAMATH_CALUDE_card_position_retained_l1767_176713

theorem card_position_retained (n : ℕ) : 
  (∃ (total_cards : ℕ), 
    total_cards = 2 * n ∧ 
    201 ≤ n ∧ 
    (∀ (card : ℕ), card ≤ total_cards → 
      (card ≤ n → (card + n).mod 2 = 1) ∧ 
      (n < card → card.mod 2 = 0))) →
  201 = n :=
by sorry

end NUMINAMATH_CALUDE_card_position_retained_l1767_176713


namespace NUMINAMATH_CALUDE_monthly_expenses_ratio_l1767_176781

theorem monthly_expenses_ratio (E : ℝ) (rent_percentage : ℝ) (rent_amount : ℝ) (savings : ℝ)
  (h1 : rent_percentage = 0.07)
  (h2 : rent_amount = 133)
  (h3 : savings = 817)
  (h4 : rent_amount = E * rent_percentage) :
  (E - rent_amount - savings) / E = 0.5 := by
sorry

end NUMINAMATH_CALUDE_monthly_expenses_ratio_l1767_176781


namespace NUMINAMATH_CALUDE_sum_of_remainders_consecutive_integers_l1767_176776

theorem sum_of_remainders_consecutive_integers (n : ℕ) : 
  (n % 4 + (n + 1) % 4 + (n + 2) % 4 + (n + 3) % 4 = 6) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_remainders_consecutive_integers_l1767_176776


namespace NUMINAMATH_CALUDE_ram_birthday_is_19th_l1767_176798

/-- Represents the number of languages learned per day -/
def languages_per_day : ℕ := sorry

/-- Represents the number of languages known on the first day of the month -/
def languages_first_day : ℕ := 820

/-- Represents the number of languages known on the last day of the month -/
def languages_last_day : ℕ := 1100

/-- Represents the number of languages known on the birthday -/
def languages_birthday : ℕ := 1000

/-- Represents the day of the month on which the birthday falls -/
def birthday : ℕ := sorry

theorem ram_birthday_is_19th : 
  birthday = 19 ∧
  languages_per_day * (birthday - 1) + languages_first_day = languages_birthday ∧
  languages_per_day * (31 - 1) + languages_first_day = languages_last_day :=
sorry

end NUMINAMATH_CALUDE_ram_birthday_is_19th_l1767_176798


namespace NUMINAMATH_CALUDE_equidistant_function_b_squared_l1767_176797

/-- A complex function that is equidistant from its input and the origin -/
def equidistant_function (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) : ℂ → ℂ := 
  fun z ↦ (a + b * Complex.I) * z

/-- The main theorem -/
theorem equidistant_function_b_squared 
  (a b : ℝ) 
  (h₁ : 0 < a) 
  (h₂ : 0 < b) 
  (h₃ : ∀ z : ℂ, Complex.abs (equidistant_function a b h₁ h₂ z - z) = Complex.abs (equidistant_function a b h₁ h₂ z))
  (h₄ : Complex.abs (a + b * Complex.I) = 10) :
  b^2 = 99.75 := by
sorry

end NUMINAMATH_CALUDE_equidistant_function_b_squared_l1767_176797


namespace NUMINAMATH_CALUDE_x_value_l1767_176785

theorem x_value (x : ℝ) : x = 80 * (1 + 13 / 100) → x = 90.4 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l1767_176785


namespace NUMINAMATH_CALUDE_solution_set_f_gt_2_min_value_f_l1767_176772

-- Define the function f(x)
def f (x : ℝ) : ℝ := |2*x + 1| - |x - 4|

-- Theorem for the solution set of f(x) > 2
theorem solution_set_f_gt_2 :
  {x : ℝ | f x > 2} = {x : ℝ | x < -7 ∨ x > 5/3} := by sorry

-- Theorem for the minimum value of f(x)
theorem min_value_f :
  ∃ (x : ℝ), f x = -9/2 ∧ ∀ (y : ℝ), f y ≥ -9/2 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_gt_2_min_value_f_l1767_176772


namespace NUMINAMATH_CALUDE_volume_ratio_equal_surface_area_l1767_176734

/-- Given an equilateral cone, an equilateral cylinder, and a sphere, all with equal surface area F,
    their volumes are in the ratio 2 : √6 : 3. -/
theorem volume_ratio_equal_surface_area (F : ℝ) (F_pos : F > 0) :
  ∃ (K₁ K₂ K₃ : ℝ),
    (K₁ > 0 ∧ K₂ > 0 ∧ K₃ > 0) ∧
    (K₁ = F * Real.sqrt F / (9 * Real.sqrt Real.pi)) ∧  -- Volume of equilateral cone
    (K₂ = F * Real.sqrt F * Real.sqrt 6 / (18 * Real.sqrt Real.pi)) ∧  -- Volume of equilateral cylinder
    (K₃ = F * Real.sqrt F / (6 * Real.sqrt Real.pi)) ∧  -- Volume of sphere
    (K₁ / 2 = K₂ / Real.sqrt 6 ∧ K₁ / 2 = K₃ / 3) :=
by sorry

end NUMINAMATH_CALUDE_volume_ratio_equal_surface_area_l1767_176734


namespace NUMINAMATH_CALUDE_triangle_angle_problem_l1767_176767

theorem triangle_angle_problem (x z : ℝ) : 
  (2*x + 3*x + x = 180) → 
  (x + z = 180) → 
  z = 150 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_problem_l1767_176767


namespace NUMINAMATH_CALUDE_circle_diameter_endpoint_l1767_176739

/-- Given a circle with center (3.5, -2) and one endpoint of a diameter at (1, -6),
    prove that the other endpoint of the diameter is at (6, 2). -/
theorem circle_diameter_endpoint (center : ℝ × ℝ) (endpoint1 : ℝ × ℝ) (endpoint2 : ℝ × ℝ) : 
  center = (3.5, -2) →
  endpoint1 = (1, -6) →
  endpoint2 = (6, 2) →
  (center.1 - endpoint1.1 = endpoint2.1 - center.1) ∧
  (center.2 - endpoint1.2 = endpoint2.2 - center.2) := by
  sorry


end NUMINAMATH_CALUDE_circle_diameter_endpoint_l1767_176739


namespace NUMINAMATH_CALUDE_additive_inverse_solution_l1767_176787

theorem additive_inverse_solution (x : ℝ) : (2*x - 12) + (x + 3) = 0 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_additive_inverse_solution_l1767_176787

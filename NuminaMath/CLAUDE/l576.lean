import Mathlib

namespace NUMINAMATH_CALUDE_average_glasses_per_box_l576_57654

/-- Proves that the average number of glasses per box is 15 given the specified conditions -/
theorem average_glasses_per_box (small_box_count : ℕ) (large_box_count : ℕ) : 
  large_box_count = small_box_count + 16 →
  12 * small_box_count + 16 * large_box_count = 480 →
  (480 : ℚ) / (small_box_count + large_box_count) = 15 := by
sorry

end NUMINAMATH_CALUDE_average_glasses_per_box_l576_57654


namespace NUMINAMATH_CALUDE_power_of_product_squared_l576_57697

theorem power_of_product_squared (x y : ℝ) : (x * y^2)^2 = x^2 * y^4 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_squared_l576_57697


namespace NUMINAMATH_CALUDE_not_divisible_by_seven_divisible_by_others_l576_57602

theorem not_divisible_by_seven (n : ℤ) : ¬(7 ∣ (n^2225 - n^2005)) :=
sorry

theorem divisible_by_others (n : ℤ) : 
  (3 ∣ (n^2225 - n^2005)) ∧ 
  (5 ∣ (n^2225 - n^2005)) ∧ 
  (11 ∣ (n^2225 - n^2005)) ∧ 
  (23 ∣ (n^2225 - n^2005)) :=
sorry

end NUMINAMATH_CALUDE_not_divisible_by_seven_divisible_by_others_l576_57602


namespace NUMINAMATH_CALUDE_inverse_proportion_relation_l576_57628

theorem inverse_proportion_relation :
  ∀ (y₁ y₂ y₃ : ℝ),
    y₁ = 1 / (-1) →
    y₂ = 1 / (-2) →
    y₃ = 1 / 3 →
    y₃ > y₂ ∧ y₂ > y₁ :=
by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_relation_l576_57628


namespace NUMINAMATH_CALUDE_angle_in_full_rotation_l576_57656

theorem angle_in_full_rotation (y : ℝ) : y + 90 = 360 → y = 270 := by
  sorry

end NUMINAMATH_CALUDE_angle_in_full_rotation_l576_57656


namespace NUMINAMATH_CALUDE_sum_of_squares_unique_l576_57601

theorem sum_of_squares_unique (p q r : ℕ+) : 
  p + q + r = 33 → 
  Nat.gcd p.val q.val + Nat.gcd q.val r.val + Nat.gcd r.val p.val = 11 → 
  p^2 + q^2 + r^2 = 419 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_unique_l576_57601


namespace NUMINAMATH_CALUDE_pentagon_coverage_theorem_l576_57696

/-- Represents the tiling of a plane with squares and pentagons -/
structure PlaneTiling where
  /-- The number of smaller squares in each large square -/
  total_squares : ℕ
  /-- The number of smaller squares used to form pentagons in each large square -/
  pentagon_squares : ℕ

/-- Calculates the percentage of the plane covered by pentagons -/
def pentagon_coverage_percentage (tiling : PlaneTiling) : ℚ :=
  (tiling.pentagon_squares : ℚ) / (tiling.total_squares : ℚ) * 100

/-- Rounds a rational number to the nearest integer -/
def round_to_nearest_integer (q : ℚ) : ℤ :=
  ⌊q + 1/2⌋

/-- Theorem stating that the percentage of the plane covered by pentagons
    in the given tiling is 56% when rounded to the nearest integer -/
theorem pentagon_coverage_theorem (tiling : PlaneTiling) 
  (h1 : tiling.total_squares = 9)
  (h2 : tiling.pentagon_squares = 5) : 
  round_to_nearest_integer (pentagon_coverage_percentage tiling) = 56 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_coverage_theorem_l576_57696


namespace NUMINAMATH_CALUDE_angle_between_vectors_l576_57679

def tangent_of_angle_between_vectors (a b : ℝ × ℝ) : ℝ := sorry

theorem angle_between_vectors
  (a b : ℝ × ℝ)
  (h1 : a.1 * (a.1 + b.1) + a.2 * (a.2 + b.2) = 5)
  (h2 : Real.sqrt (a.1^2 + a.2^2) = 2)
  (h3 : Real.sqrt (b.1^2 + b.2^2) = 1) :
  tangent_of_angle_between_vectors a b = Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_angle_between_vectors_l576_57679


namespace NUMINAMATH_CALUDE_convex_n_gon_division_possible_values_l576_57630

/-- A convex n-gon divided into three convex polygons -/
structure ConvexNGonDivision (n : ℕ) where
  (polygon1 : ℕ)  -- Number of sides of the first polygon
  (polygon2 : ℕ)  -- Number of sides of the second polygon
  (polygon3 : ℕ)  -- Number of sides of the third polygon
  (h1 : polygon1 = n)  -- First polygon has n sides
  (h2 : polygon2 > n)  -- Second polygon has more than n sides
  (h3 : polygon3 < n)  -- Third polygon has fewer than n sides

/-- The theorem stating the possible values of n -/
theorem convex_n_gon_division_possible_values :
  ∀ n : ℕ, (∃ d : ConvexNGonDivision n, True) → n = 4 ∨ n = 5 :=
sorry

end NUMINAMATH_CALUDE_convex_n_gon_division_possible_values_l576_57630


namespace NUMINAMATH_CALUDE_geometric_sequence_a6_l576_57606

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_a6 (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n, a n > 0) →
  a 2 = 1 →
  a 7 = a 5 + 2 * a 3 →
  a 6 = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_a6_l576_57606


namespace NUMINAMATH_CALUDE_sixth_power_sum_of_roots_l576_57686

theorem sixth_power_sum_of_roots (r s : ℝ) : 
  r^2 - 2*r*Real.sqrt 5 + 2 = 0 → 
  s^2 - 2*s*Real.sqrt 5 + 2 = 0 → 
  r^6 + s^6 = 3904 := by
  sorry

end NUMINAMATH_CALUDE_sixth_power_sum_of_roots_l576_57686


namespace NUMINAMATH_CALUDE_square_roots_problem_l576_57684

theorem square_roots_problem (x : ℝ) :
  (x + 1 > 0) ∧ (4 - 2*x > 0) ∧ (x + 1)^2 = (4 - 2*x)^2 →
  (x + 1)^2 = 36 :=
by sorry

end NUMINAMATH_CALUDE_square_roots_problem_l576_57684


namespace NUMINAMATH_CALUDE_rare_integer_existence_and_uniqueness_l576_57621

/-- A function f: ℤ → ℤ satisfying the given functional equation -/
def FunctionalEquation (f : ℤ → ℤ) : Prop :=
  ∀ x y : ℤ, f (f (x + y) + y) = f (f x + y)

/-- An integer v is f-rare if the set {x ∈ ℤ : f(x) = v} is finite and nonempty -/
def IsRare (f : ℤ → ℤ) (v : ℤ) : Prop :=
  let X_v := {x : ℤ | f x = v}
  Set.Finite X_v ∧ Set.Nonempty X_v

theorem rare_integer_existence_and_uniqueness :
  (∃ f : ℤ → ℤ, FunctionalEquation f ∧ ∃ v : ℤ, IsRare f v) ∧
  (∀ f : ℤ → ℤ, FunctionalEquation f → ∀ v w : ℤ, IsRare f v → IsRare f w → v = w) :=
by sorry

end NUMINAMATH_CALUDE_rare_integer_existence_and_uniqueness_l576_57621


namespace NUMINAMATH_CALUDE_m_range_theorem_l576_57641

-- Define the quadratic function p
def p (x : ℝ) : Prop := x^2 - 8*x - 20 > 0

-- Define the function q
def q (x m : ℝ) : Prop := (x - (1 - m)) * (x - (1 + m)) > 0

-- State the theorem
theorem m_range_theorem (h_sufficient : ∀ x m : ℝ, p x → q x m) 
                        (h_not_necessary : ∃ x m : ℝ, q x m ∧ ¬(p x))
                        (h_m_positive : m > 0) :
  0 < m ∧ m ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_m_range_theorem_l576_57641


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l576_57689

/-- Given a geometric sequence {a_n} where a_5 = -16 and a_8 = 8, prove that a_11 = -4 -/
theorem geometric_sequence_problem (a : ℕ → ℝ) :
  (∀ n m : ℕ, a (n + m) = a n * (a (n + 1) / a n)^m) →  -- geometric sequence property
  a 5 = -16 →
  a 8 = 8 →
  a 11 = -4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l576_57689


namespace NUMINAMATH_CALUDE_corresponding_angles_equal_l576_57627

/-- Given two triangles ABC and A₁B₁C₁, where for each pair of corresponding angles,
    either the angles are equal or their sum is 180°, all corresponding angles are equal. -/
theorem corresponding_angles_equal 
  (α β γ α₁ β₁ γ₁ : ℝ) 
  (triangle_ABC : α + β + γ = 180)
  (triangle_A₁B₁C₁ : α₁ + β₁ + γ₁ = 180)
  (h1 : α = α₁ ∨ α + α₁ = 180)
  (h2 : β = β₁ ∨ β + β₁ = 180)
  (h3 : γ = γ₁ ∨ γ + γ₁ = 180) :
  α = α₁ ∧ β = β₁ ∧ γ = γ₁ := by
  sorry

end NUMINAMATH_CALUDE_corresponding_angles_equal_l576_57627


namespace NUMINAMATH_CALUDE_buses_needed_l576_57612

theorem buses_needed (num_students : ℕ) (seats_per_bus : ℕ) (h1 : num_students = 14) (h2 : seats_per_bus = 2) :
  (num_students + seats_per_bus - 1) / seats_per_bus = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_buses_needed_l576_57612


namespace NUMINAMATH_CALUDE_exist_tetrahedra_volume_area_paradox_l576_57639

/-- A tetrahedron in 3D space -/
structure Tetrahedron where
  vertices : Fin 4 → ℝ × ℝ × ℝ

/-- Calculate the volume of a tetrahedron -/
def volume (t : Tetrahedron) : ℝ := sorry

/-- Calculate the area of a face of a tetrahedron -/
def face_area (t : Tetrahedron) (face : Fin 4) : ℝ := sorry

/-- Theorem: There exist two tetrahedra such that one has greater volume
    but smaller or equal face areas compared to the other -/
theorem exist_tetrahedra_volume_area_paradox :
  ∃ (t₁ t₂ : Tetrahedron),
    volume t₁ > volume t₂ ∧
    ∀ (face₁ : Fin 4), ∃ (face₂ : Fin 4),
      face_area t₁ face₁ ≤ face_area t₂ face₂ :=
sorry

end NUMINAMATH_CALUDE_exist_tetrahedra_volume_area_paradox_l576_57639


namespace NUMINAMATH_CALUDE_root_sum_squares_plus_product_l576_57671

theorem root_sum_squares_plus_product (a b : ℝ) (x₁ x₂ : ℝ) : 
  (x₁^2 + b*x₁ + b^2 + a = 0) → 
  (x₂^2 + b*x₂ + b^2 + a = 0) → 
  x₁^2 + x₁*x₂ + x₂^2 + a = 0 := by
sorry

end NUMINAMATH_CALUDE_root_sum_squares_plus_product_l576_57671


namespace NUMINAMATH_CALUDE_ellipse_area_l576_57699

/-- The area of an ellipse defined by the equation 9x^2 + 16y^2 = 144 is 12π. -/
theorem ellipse_area (x y : ℝ) : 
  (9 * x^2 + 16 * y^2 = 144) → (π * 4 * 3 : ℝ) = 12 * π := by
  sorry

end NUMINAMATH_CALUDE_ellipse_area_l576_57699


namespace NUMINAMATH_CALUDE_chess_club_election_proof_l576_57634

def total_candidates : ℕ := 20
def previous_board_members : ℕ := 10
def board_positions : ℕ := 6

theorem chess_club_election_proof :
  (Nat.choose total_candidates board_positions) - 
  (Nat.choose (total_candidates - previous_board_members) board_positions) = 38550 := by
  sorry

end NUMINAMATH_CALUDE_chess_club_election_proof_l576_57634


namespace NUMINAMATH_CALUDE_train_length_l576_57622

/-- Given a train traveling at a certain speed that crosses a bridge of known length in a specific time,
    this theorem proves the length of the train. -/
theorem train_length
  (train_speed : ℝ)
  (bridge_length : ℝ)
  (crossing_time : ℝ)
  (h1 : train_speed = 45)  -- km/hr
  (h2 : bridge_length = 235)  -- meters
  (h3 : crossing_time = 30)  -- seconds
  : ∃ (train_length : ℝ), train_length = 140 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l576_57622


namespace NUMINAMATH_CALUDE_class_size_l576_57623

theorem class_size (s : ℕ) (r : ℕ) : 
  (0 * 2 + 1 * 12 + 2 * 10 + 3 * r) / s = 2 →
  s = 2 + 12 + 10 + r →
  s = 40 :=
by sorry

end NUMINAMATH_CALUDE_class_size_l576_57623


namespace NUMINAMATH_CALUDE_quadratic_properties_l576_57662

/-- A quadratic function f(x) = ax² + bx + c with specific properties -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_properties (a b c d : ℝ) :
  (∀ x, QuadraticFunction a b c x = a * x^2 + b * x + c) →
  QuadraticFunction a b c 0 = 3 →
  QuadraticFunction a b c (-1/2) = 0 →
  QuadraticFunction a b c 3 = 0 →
  (∃ x, QuadraticFunction a b c x = x + d ∧ 
        ∀ y, y ≠ x → QuadraticFunction a b c y > y + d) →
  a = -2 ∧ b = 5 ∧ c = 3 ∧ d = 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_properties_l576_57662


namespace NUMINAMATH_CALUDE_bank_interest_calculation_l576_57632

def initial_deposit : ℝ := 5600
def interest_rate : ℝ := 0.07
def time_period : ℕ := 2

theorem bank_interest_calculation :
  let interest_per_year := initial_deposit * interest_rate
  let total_interest := interest_per_year * time_period
  initial_deposit + total_interest = 6384 := by
  sorry

end NUMINAMATH_CALUDE_bank_interest_calculation_l576_57632


namespace NUMINAMATH_CALUDE_polynomial_simplification_l576_57626

theorem polynomial_simplification (x : ℝ) : 
  (3 * x^2 + 4 * x - 5) - (2 * x^2 - 3 * x + 8) = x^2 + 7 * x - 13 := by
sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l576_57626


namespace NUMINAMATH_CALUDE_cubic_root_ratio_l576_57647

theorem cubic_root_ratio (r : ℝ) (h : r > 1) : 
  (∃ a b c : ℝ, 
    (81 * a^3 - 243 * a^2 + 216 * a - 64 = 0) ∧ 
    (81 * b^3 - 243 * b^2 + 216 * b - 64 = 0) ∧ 
    (81 * c^3 - 243 * c^2 + 216 * c - 64 = 0) ∧ 
    (b = a * r) ∧ 
    (c = b * r)) → 
  (c / a = r^2) :=
by sorry

end NUMINAMATH_CALUDE_cubic_root_ratio_l576_57647


namespace NUMINAMATH_CALUDE_arrange_digits_eq_96_l576_57633

/-- The number of ways to arrange the digits of 60,402 to form a 5-digit number, 
    ensuring numbers do not begin with 0 -/
def arrange_digits : ℕ :=
  let digits : List ℕ := [6, 0, 4, 0, 2]
  let non_zero_digits : List ℕ := digits.filter (· ≠ 0)
  let zero_count : ℕ := digits.count 0
  let digit_count : ℕ := digits.length
  (digit_count - 1) * (non_zero_digits.length).factorial

theorem arrange_digits_eq_96 : arrange_digits = 96 := by
  sorry

end NUMINAMATH_CALUDE_arrange_digits_eq_96_l576_57633


namespace NUMINAMATH_CALUDE_volume_ratio_theorem_l576_57658

/-- A right rectangular prism with edge lengths 2, 3, and 5 -/
structure RectPrism where
  length : ℝ := 2
  width : ℝ := 3
  height : ℝ := 5

/-- The set of points within distance r of the prism -/
def S (B : RectPrism) (r : ℝ) : Set (ℝ × ℝ × ℝ) := sorry

/-- The volume of S(r) -/
noncomputable def volume_S (B : RectPrism) (r : ℝ) : ℝ := sorry

/-- Coefficients of the volume function -/
structure VolumeCoeffs where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  all_positive : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d

theorem volume_ratio_theorem (B : RectPrism) (coeffs : VolumeCoeffs) 
  (h : ∀ r : ℝ, volume_S B r = coeffs.a * r^3 + coeffs.b * r^2 + coeffs.c * r + coeffs.d) :
  coeffs.b * coeffs.c / (coeffs.a * coeffs.d) = 15.5 := by sorry

end NUMINAMATH_CALUDE_volume_ratio_theorem_l576_57658


namespace NUMINAMATH_CALUDE_a_is_most_suitable_l576_57675

-- Define the participants
inductive Participant
  | A
  | B
  | C
  | D

-- Define the variance for each participant
def variance (p : Participant) : ℝ :=
  match p with
  | Participant.A => 0.15
  | Participant.B => 0.2
  | Participant.C => 0.4
  | Participant.D => 0.35

-- Define the function to find the most suitable participant
def most_suitable : Participant :=
  Participant.A

-- Theorem to prove A is the most suitable
theorem a_is_most_suitable :
  ∀ p : Participant, variance most_suitable ≤ variance p :=
by sorry

end NUMINAMATH_CALUDE_a_is_most_suitable_l576_57675


namespace NUMINAMATH_CALUDE_orange_harvest_theorem_l576_57680

theorem orange_harvest_theorem (daily_harvest : ℕ) (harvest_days : ℕ) 
  (h1 : daily_harvest = 76) (h2 : harvest_days = 63) :
  daily_harvest * harvest_days = 4788 := by
  sorry

end NUMINAMATH_CALUDE_orange_harvest_theorem_l576_57680


namespace NUMINAMATH_CALUDE_max_area_rectangular_fence_l576_57615

/-- Represents a rectangular fence with given constraints -/
structure RectangularFence where
  length : ℝ
  width : ℝ
  perimeter_constraint : length + width = 200
  length_constraint : length ≥ 100
  width_constraint : width ≥ 50

/-- Calculates the area of a rectangular fence -/
def area (fence : RectangularFence) : ℝ :=
  fence.length * fence.width

/-- Theorem stating the maximum area of the rectangular fence -/
theorem max_area_rectangular_fence :
  ∃ (fence : RectangularFence), ∀ (other : RectangularFence), area fence ≥ area other ∧ area fence = 10000 := by
  sorry

end NUMINAMATH_CALUDE_max_area_rectangular_fence_l576_57615


namespace NUMINAMATH_CALUDE_fruit_bowl_problem_l576_57616

theorem fruit_bowl_problem (initial_oranges : ℕ) : 
  (14 : ℝ) / ((14 : ℝ) + (initial_oranges - 19)) = 0.7 → 
  initial_oranges = 25 := by
  sorry

end NUMINAMATH_CALUDE_fruit_bowl_problem_l576_57616


namespace NUMINAMATH_CALUDE_product_with_floor_l576_57649

theorem product_with_floor (x : ℝ) : 
  x > 0 → x * ⌊x⌋ = 48 → x = 8 := by sorry

end NUMINAMATH_CALUDE_product_with_floor_l576_57649


namespace NUMINAMATH_CALUDE_power_multiplication_equals_512_l576_57637

theorem power_multiplication_equals_512 : 2^3 * 2^6 = 512 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_equals_512_l576_57637


namespace NUMINAMATH_CALUDE_problem_solution_l576_57614

theorem problem_solution (a b : ℤ) (h1 : a > b) (h2 : b > 0) (h3 : a + b + a * b = 152) : a = 50 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l576_57614


namespace NUMINAMATH_CALUDE_largest_integer_x_l576_57682

theorem largest_integer_x : ∀ x : ℤ, (2 * x : ℚ) / 7 + 3 / 4 < 8 / 7 ↔ x ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_largest_integer_x_l576_57682


namespace NUMINAMATH_CALUDE_no_function_satisfies_condition_l576_57620

theorem no_function_satisfies_condition :
  ¬ ∃ f : ℝ → ℝ, ∀ x y z : ℝ, f (x * y) + f (x * z) - f x * f (y * z) > 1 := by
  sorry

end NUMINAMATH_CALUDE_no_function_satisfies_condition_l576_57620


namespace NUMINAMATH_CALUDE_amy_final_money_l576_57613

-- Define the initial conditions
def initial_money : ℕ := 2
def num_neighbors : ℕ := 5
def chore_pay : ℕ := 13
def birthday_money : ℕ := 3
def toy_cost : ℕ := 12

-- Define the calculation steps
def money_after_chores : ℕ := initial_money + num_neighbors * chore_pay
def money_after_birthday : ℕ := money_after_chores + birthday_money
def money_after_toy : ℕ := money_after_birthday - toy_cost
def grandparents_gift : ℕ := 2 * money_after_toy

-- Theorem to prove
theorem amy_final_money :
  money_after_toy + grandparents_gift = 174 := by
  sorry


end NUMINAMATH_CALUDE_amy_final_money_l576_57613


namespace NUMINAMATH_CALUDE_bees_in_hive_l576_57674

/-- The total number of bees in a hive after more bees fly in -/
def total_bees (initial : ℕ) (additional : ℕ) : ℕ :=
  initial + additional

/-- Theorem: The total number of bees is 24 when there are initially 16 bees and 8 more fly in -/
theorem bees_in_hive : total_bees 16 8 = 24 := by
  sorry

end NUMINAMATH_CALUDE_bees_in_hive_l576_57674


namespace NUMINAMATH_CALUDE_intersection_of_sets_l576_57650

theorem intersection_of_sets : 
  let A : Set ℕ := {1, 2, 3}
  let B : Set ℕ := {3, 4}
  A ∩ B = {3} := by
sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l576_57650


namespace NUMINAMATH_CALUDE_optimal_sampling_theorem_l576_57685

/-- Represents the different blood types --/
inductive BloodType
| O
| A
| B
| AB

/-- Represents the available sampling methods --/
inductive SamplingMethod
| Random
| Systematic
| Stratified

structure School :=
  (total_students : Nat)
  (blood_type_counts : BloodType → Nat)
  (sample_size : Nat)
  (soccer_team_size : Nat)
  (soccer_sample_size : Nat)

def optimal_sampling_method (school : School) (is_blood_type_study : Bool) : SamplingMethod :=
  if is_blood_type_study then
    SamplingMethod.Stratified
  else
    SamplingMethod.Random

theorem optimal_sampling_theorem (school : School) :
  (school.total_students = 500) →
  (school.blood_type_counts BloodType.O = 200) →
  (school.blood_type_counts BloodType.A = 125) →
  (school.blood_type_counts BloodType.B = 125) →
  (school.blood_type_counts BloodType.AB = 50) →
  (school.sample_size = 20) →
  (school.soccer_team_size = 11) →
  (school.soccer_sample_size = 2) →
  (optimal_sampling_method school true = SamplingMethod.Stratified) ∧
  (optimal_sampling_method school false = SamplingMethod.Random) :=
by
  sorry

end NUMINAMATH_CALUDE_optimal_sampling_theorem_l576_57685


namespace NUMINAMATH_CALUDE_sqrt_equation_solutions_l576_57677

theorem sqrt_equation_solutions (x : ℝ) : 
  Real.sqrt ((3 + 2 * Real.sqrt 2) ^ x) + Real.sqrt ((3 - 2 * Real.sqrt 2) ^ x) = 6 ↔ x = 2 ∨ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solutions_l576_57677


namespace NUMINAMATH_CALUDE_train_speed_l576_57690

/-- The speed of a train given its length and time to cross a fixed point -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 120) (h2 : time = 16) :
  length / time = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l576_57690


namespace NUMINAMATH_CALUDE_intersection_with_complement_of_B_l576_57645

open Set

theorem intersection_with_complement_of_B (U A B : Set ℕ) : 
  U = {1, 2, 3, 4, 5, 6} →
  A = {2, 4, 6} →
  B = {1, 3} →
  A ∩ (U \ B) = {2, 4, 6} := by
sorry

end NUMINAMATH_CALUDE_intersection_with_complement_of_B_l576_57645


namespace NUMINAMATH_CALUDE_mary_final_cards_l576_57635

def initial_cards : ℕ := 18
def torn_cards : ℕ := 8
def cards_from_fred : ℕ := 26
def cards_bought : ℕ := 40
def cards_exchanged : ℕ := 10
def cards_lost : ℕ := 5

theorem mary_final_cards : 
  initial_cards - torn_cards + cards_from_fred + cards_bought - cards_lost = 71 := by
  sorry

end NUMINAMATH_CALUDE_mary_final_cards_l576_57635


namespace NUMINAMATH_CALUDE_solution_set_f_less_than_8_range_of_m_for_solution_existence_l576_57693

-- Define the function f
def f (x : ℝ) : ℝ := 45 * abs (2 * x + 3) + abs (2 * x - 1)

-- Theorem for part I
theorem solution_set_f_less_than_8 :
  {x : ℝ | f x < 8} = {x : ℝ | -5/2 < x ∧ x < 3/2} :=
sorry

-- Theorem for part II
theorem range_of_m_for_solution_existence :
  {m : ℝ | ∃ x, f x ≤ |3 * m + 1|} = {m : ℝ | m ≤ -5/3 ∨ m ≥ 1} :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_less_than_8_range_of_m_for_solution_existence_l576_57693


namespace NUMINAMATH_CALUDE_stratified_sampling_first_year_l576_57692

theorem stratified_sampling_first_year
  (total_sample : ℕ)
  (first_year_ratio second_year_ratio third_year_ratio : ℕ)
  (h_total_sample : total_sample = 56)
  (h_ratios : first_year_ratio = 7 ∧ second_year_ratio = 3 ∧ third_year_ratio = 4) :
  (total_sample * first_year_ratio) / (first_year_ratio + second_year_ratio + third_year_ratio) = 28 := by
  sorry

#check stratified_sampling_first_year

end NUMINAMATH_CALUDE_stratified_sampling_first_year_l576_57692


namespace NUMINAMATH_CALUDE_common_solutions_iff_y_values_l576_57625

theorem common_solutions_iff_y_values (x y : ℝ) : 
  (∃ x : ℝ, x^2 + y^2 - 16 = 0 ∧ x^2 - 3*y - 12 = 0) ↔ (y = -4 ∨ y = 1) :=
by sorry

end NUMINAMATH_CALUDE_common_solutions_iff_y_values_l576_57625


namespace NUMINAMATH_CALUDE_circle_center_perpendicular_line_l576_57603

-- Define the circle C
def circle_C : Set (ℝ × ℝ) := sorry

-- Define the center of the circle
def center : ℝ × ℝ := sorry

-- Define the line l
def line_l (x y : ℝ) : Prop := y = x - 1

-- Define the perpendicular line passing through the center
def perpendicular_line (x y : ℝ) : Prop := x + y - 3 = 0

theorem circle_center_perpendicular_line :
  (1, 0) ∈ circle_C ∧
  center.1 > 0 ∧
  center.2 = 0 ∧
  (∃ (a b : ℝ), (a, b) ∈ circle_C ∧ line_l a b ∧
    (a - center.1)^2 + (b - center.2)^2 = 8) →
  ∀ x y, perpendicular_line x y ↔ 
    (x - center.1) * 1 + (y - center.2) * 1 = 0 ∧
    center ∈ ({p : ℝ × ℝ | perpendicular_line p.1 p.2} : Set (ℝ × ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_circle_center_perpendicular_line_l576_57603


namespace NUMINAMATH_CALUDE_least_zogs_for_dropping_advantage_l576_57687

/-- Score for dropping n zogs -/
def drop_score (n : ℕ) : ℕ := n * (n + 1)

/-- Score for eating n zogs -/
def eat_score (n : ℕ) : ℕ := 8 * n

/-- Predicate for when dropping earns more points than eating -/
def dropping_beats_eating (n : ℕ) : Prop := drop_score n > eat_score n

theorem least_zogs_for_dropping_advantage : 
  (∀ k < 8, ¬dropping_beats_eating k) ∧ dropping_beats_eating 8 := by sorry

end NUMINAMATH_CALUDE_least_zogs_for_dropping_advantage_l576_57687


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l576_57669

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ a₁₂ a₁₃ : ℝ) :
  (∀ x : ℝ, (x^2 + 1) * (x - 3)^11 = a + a₁*(x - 2) + a₂*(x - 2)^2 + a₃*(x - 2)^3 + 
    a₄*(x - 2)^4 + a₅*(x - 2)^5 + a₆*(x - 2)^6 + a₇*(x - 2)^7 + a₈*(x - 2)^8 + 
    a₉*(x - 2)^9 + a₁₀*(x - 2)^10 + a₁₁*(x - 2)^11 + a₁₂*(x - 2)^12 + a₁₃*(x - 2)^13) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ + a₁₁ + a₁₂ = 5 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l576_57669


namespace NUMINAMATH_CALUDE_count_valid_numbers_l576_57688

def is_odd (n : ℕ) : Prop := n % 2 = 1

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + digit_sum (n / 10)

def digit_product (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) * digit_product (n / 10)

def valid_number (n : ℕ) : Prop :=
  n > 0 ∧ n < 1000000 ∧ is_odd n ∧ is_odd (digit_sum n) ∧ is_odd (digit_product n)

theorem count_valid_numbers :
  ∃ (S : Finset ℕ), (∀ n ∈ S, valid_number n) ∧ S.card = 39 ∧
  ∀ m, valid_number m → m ∈ S :=
sorry

end NUMINAMATH_CALUDE_count_valid_numbers_l576_57688


namespace NUMINAMATH_CALUDE_angles_with_same_terminal_side_as_45_l576_57610

def same_terminal_side (α β : ℝ) : Prop :=
  ∃ k : ℤ, β = α + k * 360

theorem angles_with_same_terminal_side_as_45 :
  ∀ θ : ℝ, -720 ≤ θ ∧ θ < 0 ∧ same_terminal_side 45 θ →
    θ = -675 ∨ θ = -315 := by sorry

end NUMINAMATH_CALUDE_angles_with_same_terminal_side_as_45_l576_57610


namespace NUMINAMATH_CALUDE_cos_neg_nineteen_pi_sixths_l576_57681

theorem cos_neg_nineteen_pi_sixths :
  Real.cos (-19 * π / 6) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_neg_nineteen_pi_sixths_l576_57681


namespace NUMINAMATH_CALUDE_platform_length_l576_57683

/-- Given a train of length 450 meters that takes 39 seconds to cross a platform
    and 18 seconds to cross a signal pole, the length of the platform is 525 meters. -/
theorem platform_length (train_length : ℝ) (time_platform : ℝ) (time_pole : ℝ) :
  train_length = 450 →
  time_platform = 39 →
  time_pole = 18 →
  (train_length / time_pole) * time_platform - train_length = 525 :=
by sorry

end NUMINAMATH_CALUDE_platform_length_l576_57683


namespace NUMINAMATH_CALUDE_sum_of_fractions_equals_one_l576_57644

theorem sum_of_fractions_equals_one (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h : a + b + c = -a*b*c) :
  (a^2*b^2)/((a^2+b*c)*(b^2+a*c)) + (a^2*c^2)/((a^2+b*c)*(c^2+a*b)) +
  (b^2*c^2)/((b^2+a*c)*(c^2+a*b)) = 1 := by
sorry

end NUMINAMATH_CALUDE_sum_of_fractions_equals_one_l576_57644


namespace NUMINAMATH_CALUDE_multiply_by_special_number_l576_57691

theorem multiply_by_special_number : ∃ x : ℝ, x * (1/1000) = 0.735 ∧ 10 * x = 7350 := by
  sorry

end NUMINAMATH_CALUDE_multiply_by_special_number_l576_57691


namespace NUMINAMATH_CALUDE_hyperbola_equation_y_axis_l576_57668

/-- Given a hyperbola with foci on the y-axis, ratio of real to imaginary axis 2:3, 
    and passing through (√6, 2), prove its equation is y²/1 - x²/3 = 3 -/
theorem hyperbola_equation_y_axis (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : 3 * a = 2 * b) (h4 : 4 / a^2 - 6 / b^2 = 1) :
  ∃ (k : ℝ), k * (y^2 / 1 - x^2 / 3) = 3 := by sorry


end NUMINAMATH_CALUDE_hyperbola_equation_y_axis_l576_57668


namespace NUMINAMATH_CALUDE_arithmetic_progression_product_power_l576_57664

theorem arithmetic_progression_product_power : ∃ (a b : ℕ), 
  a > 0 ∧ 
  (a * (2*a) * (3*a) * (4*a) * (5*a) = b^2008) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_product_power_l576_57664


namespace NUMINAMATH_CALUDE_range_of_a_l576_57604

theorem range_of_a (p q : Prop) (h_p : ∀ x ∈ Set.Icc 1 2, 2 * x^2 - a ≥ 0)
  (h_q : ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0) (h_pq : p ∧ q) :
  a ≤ -2 ∨ (1 ≤ a ∧ a ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l576_57604


namespace NUMINAMATH_CALUDE_initial_marbles_l576_57673

theorem initial_marbles (M : ℚ) : 
  (2 / 5 : ℚ) * M = 30 →
  (1 / 2 : ℚ) * ((2 / 5 : ℚ) * M) = 15 →
  M = 75 := by
  sorry

end NUMINAMATH_CALUDE_initial_marbles_l576_57673


namespace NUMINAMATH_CALUDE_one_third_percent_of_200_plus_50_l576_57651

/-- Calculates the result of taking a percentage of a number and adding a constant to it. -/
def percentageOfPlusConstant (percentage : ℚ) (number : ℚ) (constant : ℚ) : ℚ :=
  percentage / 100 * number + constant

/-- The main theorem stating that 1/3% of 200 plus 50 is approximately 50.6667 -/
theorem one_third_percent_of_200_plus_50 :
  ∃ (result : ℚ), abs (percentageOfPlusConstant (1/3) 200 50 - result) < 0.00005 ∧ result = 50.6667 := by
  sorry

#eval percentageOfPlusConstant (1/3) 200 50

end NUMINAMATH_CALUDE_one_third_percent_of_200_plus_50_l576_57651


namespace NUMINAMATH_CALUDE_cannot_transform_to_target_l576_57617

/-- Represents a natural number with its digits. -/
structure DigitNumber where
  digits : List Nat
  first_nonzero : digits.head? ≠ some 0

/-- Represents the allowed operations on the number. -/
inductive Operation
  | multiply_by_five
  | rearrange_digits

/-- Defines the target 150-digit number 5222...2223. -/
def target_number : DigitNumber := {
  digits := 5 :: List.replicate 148 2 ++ [2, 3]
  first_nonzero := by simp
}

/-- Applies an operation to a DigitNumber. -/
def apply_operation (n : DigitNumber) (op : Operation) : DigitNumber :=
  sorry

/-- Checks if a DigitNumber can be transformed into the target number using the allowed operations. -/
def can_transform (n : DigitNumber) : Prop :=
  ∃ (ops : List Operation), (ops.foldl apply_operation n) = target_number

/-- The initial number 1. -/
def initial_number : DigitNumber := {
  digits := [1]
  first_nonzero := by simp
}

/-- The main theorem stating that it's impossible to transform 1 into the target number. -/
theorem cannot_transform_to_target : ¬(can_transform initial_number) :=
  sorry

end NUMINAMATH_CALUDE_cannot_transform_to_target_l576_57617


namespace NUMINAMATH_CALUDE_stairs_climbed_together_l576_57646

theorem stairs_climbed_together (jonny_stairs : ℕ) (julia_stairs : ℕ) : 
  jonny_stairs = 1269 →
  julia_stairs = jonny_stairs / 3 - 7 →
  jonny_stairs + julia_stairs = 1685 := by
sorry

end NUMINAMATH_CALUDE_stairs_climbed_together_l576_57646


namespace NUMINAMATH_CALUDE_flower_prices_l576_57636

theorem flower_prices (x y z : ℚ) 
  (eq1 : 3 * x + 7 * y + z = 14)
  (eq2 : 4 * x + 10 * y + z = 16) :
  3 * (x + y + z) = 30 := by
sorry

end NUMINAMATH_CALUDE_flower_prices_l576_57636


namespace NUMINAMATH_CALUDE_base_6_representation_of_231_base_6_to_decimal_l576_57655

/-- Converts a natural number to its base 6 representation --/
def toBase6 (n : ℕ) : List ℕ :=
  if n < 6 then [n]
  else (n % 6) :: toBase6 (n / 6)

/-- Converts a list of digits in base 6 to a natural number --/
def fromBase6 (digits : List ℕ) : ℕ :=
  digits.foldr (fun d acc => d + 6 * acc) 0

theorem base_6_representation_of_231 :
  toBase6 231 = [3, 2, 0, 1] :=
sorry

theorem base_6_to_decimal :
  fromBase6 [3, 2, 0, 1] = 231 :=
sorry

end NUMINAMATH_CALUDE_base_6_representation_of_231_base_6_to_decimal_l576_57655


namespace NUMINAMATH_CALUDE_find_a₂_l576_57672

-- Define the equation as a function
def f (x a₀ a₁ a₂ a₃ : ℝ) : ℝ := a₀ + a₁ * (x - 2) + a₂ * (x - 2)^2 + a₃ * (x - 2)^3

-- State the theorem
theorem find_a₂ (a₀ a₁ a₂ a₃ : ℝ) : 
  (∀ x : ℝ, x^3 = f x a₀ a₁ a₂ a₃) → a₂ = 6 := by
  sorry

end NUMINAMATH_CALUDE_find_a₂_l576_57672


namespace NUMINAMATH_CALUDE_cube_surface_area_from_prisms_l576_57629

-- Define the dimensions of a single prism
def prism_length : ℝ := 10
def prism_width : ℝ := 3
def prism_height : ℝ := 30

-- Define the number of prisms
def num_prisms : ℕ := 2

-- Theorem statement
theorem cube_surface_area_from_prisms :
  let prism_volume := prism_length * prism_width * prism_height
  let total_volume := num_prisms * prism_volume
  let cube_edge := total_volume ^ (1/3 : ℝ)
  let cube_surface_area := 6 * cube_edge^2
  cube_surface_area = 600 := by sorry

end NUMINAMATH_CALUDE_cube_surface_area_from_prisms_l576_57629


namespace NUMINAMATH_CALUDE_rhombus_perimeter_rhombus_perimeter_is_20_l576_57660

/-- The perimeter of a rhombus whose diagonals are the roots of x^2 - 14x + 48 = 0 -/
theorem rhombus_perimeter : ℝ → Prop :=
  fun p =>
    ∀ (x₁ x₂ : ℝ),
      x₁^2 - 14*x₁ + 48 = 0 →
      x₂^2 - 14*x₂ + 48 = 0 →
      x₁ ≠ x₂ →
      let s := Real.sqrt ((x₁^2 + x₂^2) / 4)
      p = 4 * s

/-- The perimeter of the rhombus is 20 -/
theorem rhombus_perimeter_is_20 : rhombus_perimeter 20 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_rhombus_perimeter_is_20_l576_57660


namespace NUMINAMATH_CALUDE_square_diagonal_shorter_path_l576_57657

theorem square_diagonal_shorter_path (ε : Real) (h : ε > 0) : 
  ∃ (diff : Real), 
    abs (diff - 0.3) < ε ∧ 
    (2 - Real.sqrt 2) / 2 = diff :=
by sorry

end NUMINAMATH_CALUDE_square_diagonal_shorter_path_l576_57657


namespace NUMINAMATH_CALUDE_triangle_circumcircle_radius_l576_57659

theorem triangle_circumcircle_radius (a b c : ℝ) (h1 : a = 3) (h2 : b = 5) (h3 : c = 7) :
  let R := c / (2 * Real.sqrt (1 - ((a^2 + b^2 - c^2) / (2 * a * b))^2))
  R = 7 * Real.sqrt 3 / 3 := by sorry

end NUMINAMATH_CALUDE_triangle_circumcircle_radius_l576_57659


namespace NUMINAMATH_CALUDE_tower_divisibility_l576_57624

/-- Represents the number of towers that can be built with cubes up to edge-length n -/
def S : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | 2 => 2
  | n + 1 => S n * (min 5 (n + 1))

/-- The problem statement -/
theorem tower_divisibility : S 9 % 1000 = 0 := by
  sorry

end NUMINAMATH_CALUDE_tower_divisibility_l576_57624


namespace NUMINAMATH_CALUDE_simplify_expressions_l576_57665

variable (a b t : ℝ)

theorem simplify_expressions :
  (6 * a^2 - 2 * a * b - 2 * (3 * a^2 - (1/2) * a * b) = -a * b) ∧
  (-(t^2 - t - 1) + (2 * t^2 - 3 * t + 1) = t^2 - 2 * t + 2) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expressions_l576_57665


namespace NUMINAMATH_CALUDE_largest_five_digit_sum_20_l576_57663

def is_five_digit (n : ℕ) : Prop := n ≥ 10000 ∧ n < 100000

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  (List.sum digits)

theorem largest_five_digit_sum_20 :
  ∀ n : ℕ, is_five_digit n → digit_sum n = 20 → n ≤ 99200 :=
by sorry

end NUMINAMATH_CALUDE_largest_five_digit_sum_20_l576_57663


namespace NUMINAMATH_CALUDE_acid_mixture_problem_l576_57619

/-- Represents the contents of a jar --/
structure Jar where
  volume : ℚ
  acid_concentration : ℚ

/-- Represents the problem setup --/
structure ProblemSetup where
  jar_a : Jar
  jar_b : Jar
  jar_c : Jar
  m : ℕ
  n : ℕ

/-- The initial setup of the problem --/
def initial_setup (k : ℚ) : ProblemSetup where
  jar_a := { volume := 4, acid_concentration := 45/100 }
  jar_b := { volume := 5, acid_concentration := 48/100 }
  jar_c := { volume := 1, acid_concentration := k/100 }
  m := 2
  n := 3

/-- The final state after mixing --/
def final_state (setup : ProblemSetup) : Prop :=
  let new_jar_a_volume := setup.jar_a.volume + setup.m / setup.n
  let new_jar_b_volume := setup.jar_b.volume + (1 - setup.m / setup.n)
  let new_jar_a_acid := setup.jar_a.volume * setup.jar_a.acid_concentration + 
                        setup.jar_c.volume * setup.jar_c.acid_concentration * (setup.m / setup.n)
  let new_jar_b_acid := setup.jar_b.volume * setup.jar_b.acid_concentration + 
                        setup.jar_c.volume * setup.jar_c.acid_concentration * (1 - setup.m / setup.n)
  (new_jar_a_acid / new_jar_a_volume = 1/2) ∧ (new_jar_b_acid / new_jar_b_volume = 1/2)

/-- The main theorem --/
theorem acid_mixture_problem (k : ℚ) :
  final_state (initial_setup k) → k + 2 + 3 = 85 := by
  sorry


end NUMINAMATH_CALUDE_acid_mixture_problem_l576_57619


namespace NUMINAMATH_CALUDE_quadratic_sum_l576_57631

/-- A quadratic function with specific properties -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_sum (a b c : ℝ) :
  (∃ (x_max : ℝ), ∀ x, QuadraticFunction a b c x ≤ QuadraticFunction a b c x_max ∧
    QuadraticFunction a b c x_max = 72) →
  QuadraticFunction a b c 0 = -1 →
  QuadraticFunction a b c 6 = -1 →
  a + b + c = 356 / 9 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l576_57631


namespace NUMINAMATH_CALUDE_determinant_in_terms_of_coefficients_l576_57643

theorem determinant_in_terms_of_coefficients 
  (s p q : ℝ) (a b c : ℝ) 
  (h1 : a^3 + s*a^2 + p*a + q = 0)
  (h2 : b^3 + s*b^2 + p*b + q = 0)
  (h3 : c^3 + s*c^2 + p*c + q = 0) :
  let M : Matrix (Fin 3) (Fin 3) ℝ := !![1+a, 1, 1; 1, 1+b, 1; 1, 1, 1+c]
  Matrix.det M = -q + p - s := by
  sorry

end NUMINAMATH_CALUDE_determinant_in_terms_of_coefficients_l576_57643


namespace NUMINAMATH_CALUDE_f_2x_l576_57600

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 1

-- State the theorem
theorem f_2x (x : ℝ) : f (2 * x) = 4 * x^2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_f_2x_l576_57600


namespace NUMINAMATH_CALUDE_sum_of_selection_l576_57640

/-- Represents a selection of numbers from an 8x8 grid -/
def Selection := Fin 8 → Fin 8

/-- The sum of numbers in a selection -/
def sum_selection (s : Selection) : ℕ :=
  Finset.sum Finset.univ (λ i => s i + 1 + 8 * i)

/-- Theorem: The sum of any valid selection is 260 -/
theorem sum_of_selection (s : Selection) (h : Function.Injective s) : sum_selection s = 260 := by
  sorry

#eval sum_selection (λ i => i)  -- Should output 260

end NUMINAMATH_CALUDE_sum_of_selection_l576_57640


namespace NUMINAMATH_CALUDE_carrot_theorem_l576_57618

def carrot_problem (initial_carrots additional_carrots final_total : ℕ) : Prop :=
  initial_carrots + additional_carrots - final_total = 4

theorem carrot_theorem : carrot_problem 19 46 61 := by
  sorry

end NUMINAMATH_CALUDE_carrot_theorem_l576_57618


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l576_57652

-- Define the set M
def M : Set ℝ := {α | ∃ k : ℤ, α = k * 90 - 36}

-- Define the set N
def N : Set ℝ := {α | -180 < α ∧ α < 180}

-- Define the intersection set
def intersection : Set ℝ := {-36, 54, -126, 144}

-- Theorem statement
theorem intersection_of_M_and_N : M ∩ N = intersection := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l576_57652


namespace NUMINAMATH_CALUDE_arithmetic_mean_example_l576_57667

theorem arithmetic_mean_example : 
  let numbers : List ℕ := [12, 24, 36, 48]
  (numbers.sum / numbers.length : ℚ) = 30 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_example_l576_57667


namespace NUMINAMATH_CALUDE_taya_jenna_meet_l576_57661

/-- The floor where Taya and Jenna meet -/
def meeting_floor : ℕ := 32

/-- The starting floor -/
def start_floor : ℕ := 22

/-- Time Jenna waits for the elevator (in seconds) -/
def wait_time : ℕ := 120

/-- Time Taya takes to go up one floor (in seconds) -/
def taya_time_per_floor : ℕ := 15

/-- Time the elevator takes to go up one floor (in seconds) -/
def elevator_time_per_floor : ℕ := 3

/-- Theorem stating that Taya and Jenna arrive at the meeting floor at the same time -/
theorem taya_jenna_meet :
  taya_time_per_floor * (meeting_floor - start_floor) =
  wait_time + elevator_time_per_floor * (meeting_floor - start_floor) :=
by sorry

end NUMINAMATH_CALUDE_taya_jenna_meet_l576_57661


namespace NUMINAMATH_CALUDE_absolute_value_equation_l576_57666

theorem absolute_value_equation : 
  {x : ℤ | |(-5 + x)| = 11} = {16, -6} := by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_l576_57666


namespace NUMINAMATH_CALUDE_inequality_and_range_l576_57605

-- Define the function representing the left side of the inequality
def f (x : ℝ) : ℝ := |x - 3| + |x + 1|

-- State the theorem
theorem inequality_and_range : 
  (∀ x : ℝ, f x ≥ 4) ∧ 
  (∀ x : ℝ, f x = 4 ↔ -1 ≤ x ∧ x ≤ 3) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_range_l576_57605


namespace NUMINAMATH_CALUDE_no_primes_in_range_l576_57695

theorem no_primes_in_range (n : ℕ) (h : n > 2) :
  ∀ k ∈ Set.Ioo (n! + 2) (n! + n + 1), ¬ Nat.Prime k := by
  sorry

end NUMINAMATH_CALUDE_no_primes_in_range_l576_57695


namespace NUMINAMATH_CALUDE_tagged_fish_in_second_catch_l576_57670

-- Define the parameters of the problem
def initial_tagged : ℕ := 30
def second_catch : ℕ := 50
def total_fish : ℕ := 750

-- Define the theorem
theorem tagged_fish_in_second_catch :
  ∃ (T : ℕ), (T : ℚ) / second_catch = initial_tagged / total_fish ∧ T = 2 := by
  sorry

end NUMINAMATH_CALUDE_tagged_fish_in_second_catch_l576_57670


namespace NUMINAMATH_CALUDE_number_difference_l576_57607

theorem number_difference (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) :
  |x - y| = 4 := by
sorry

end NUMINAMATH_CALUDE_number_difference_l576_57607


namespace NUMINAMATH_CALUDE_principal_is_15000_l576_57653

/-- Calculates the principal amount given simple interest, rate, and time -/
def calculate_principal (interest : ℚ) (rate : ℚ) (time : ℚ) : ℚ :=
  (interest * 100) / (rate * time)

/-- Theorem: Given the specified conditions, the principal sum is 15000 -/
theorem principal_is_15000 :
  let interest : ℚ := 2700
  let rate : ℚ := 6
  let time : ℚ := 3
  calculate_principal interest rate time = 15000 := by
  sorry

end NUMINAMATH_CALUDE_principal_is_15000_l576_57653


namespace NUMINAMATH_CALUDE_octagon_diagonal_length_l576_57694

/-- The length of a diagonal in a regular octagon inscribed in a circle -/
theorem octagon_diagonal_length (r : ℝ) (h : r = 12) :
  let diagonal_length := Real.sqrt (288 + 144 * Real.sqrt 2)
  ∃ (AC : ℝ), AC = diagonal_length := by sorry

end NUMINAMATH_CALUDE_octagon_diagonal_length_l576_57694


namespace NUMINAMATH_CALUDE_cost_of_500_pencils_l576_57642

/-- The cost of 500 pencils in dollars, given that 1 pencil costs 3 cents -/
theorem cost_of_500_pencils :
  let cost_of_one_pencil_cents : ℕ := 3
  let number_of_pencils : ℕ := 500
  let cents_per_dollar : ℕ := 100
  (cost_of_one_pencil_cents * number_of_pencils) / cents_per_dollar = 15 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_500_pencils_l576_57642


namespace NUMINAMATH_CALUDE_watson_class_composition_l576_57698

/-- The number of kindergartners in Ms. Watson's class -/
def num_kindergartners : ℕ := 42 - (24 + 4)

theorem watson_class_composition :
  num_kindergartners = 14 :=
by sorry

end NUMINAMATH_CALUDE_watson_class_composition_l576_57698


namespace NUMINAMATH_CALUDE_least_froods_to_drop_l576_57609

def score_dropping (n : ℕ) : ℕ := n * (n + 1) / 2

def score_eating (n : ℕ) : ℕ := 15 * n

theorem least_froods_to_drop : 
  ∀ k < 30, score_dropping k ≤ score_eating k ∧ 
  score_dropping 30 > score_eating 30 := by
  sorry

end NUMINAMATH_CALUDE_least_froods_to_drop_l576_57609


namespace NUMINAMATH_CALUDE_john_index_cards_purchase_l576_57678

/-- Calculates the total number of index card packs bought for all students -/
def total_packs_bought (num_classes : ℕ) (students_per_class : ℕ) (packs_per_student : ℕ) : ℕ :=
  num_classes * students_per_class * packs_per_student

/-- Proves that given 6 classes with 30 students each, and 2 packs per student, the total packs bought is 360 -/
theorem john_index_cards_purchase :
  total_packs_bought 6 30 2 = 360 := by
  sorry

end NUMINAMATH_CALUDE_john_index_cards_purchase_l576_57678


namespace NUMINAMATH_CALUDE_a_geq_one_l576_57611

-- Define the conditions
def p (x : ℝ) : Prop := |x + 1| > 2
def q (x a : ℝ) : Prop := x > a

-- Define the relationship between p and q
axiom not_p_sufficient_for_not_q : ∀ x a : ℝ, (¬p x → ¬q x a) ∧ ∃ x a : ℝ, ¬p x ∧ q x a

-- Theorem to prove
theorem a_geq_one : ∀ a : ℝ, (∀ x : ℝ, q x a → p x) → a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_a_geq_one_l576_57611


namespace NUMINAMATH_CALUDE_min_value_f_l576_57648

def f (a x : ℝ) : ℝ := x^2 - 2*a*x - 1

theorem min_value_f (a : ℝ) :
  (∀ x ∈ Set.Icc (-1) 1, f a x ≥ 
    (if a < -1 then 2*a
     else if a ≤ 1 then -1 - a^2
     else -2*a)) ∧
  (∃ x ∈ Set.Icc (-1) 1, f a x = 
    (if a < -1 then 2*a
     else if a ≤ 1 then -1 - a^2
     else -2*a)) := by
  sorry

end NUMINAMATH_CALUDE_min_value_f_l576_57648


namespace NUMINAMATH_CALUDE_rectangular_hall_area_l576_57608

theorem rectangular_hall_area (length width : ℝ) : 
  width = (1/2) * length →
  length - width = 10 →
  length * width = 200 := by
sorry

end NUMINAMATH_CALUDE_rectangular_hall_area_l576_57608


namespace NUMINAMATH_CALUDE_urn_problem_l576_57676

def urn1_green : ℚ := 5
def urn1_blue : ℚ := 7
def urn2_green : ℚ := 20
def urn1_total : ℚ := urn1_green + urn1_blue
def same_color_prob : ℚ := 62/100

theorem urn_problem (M : ℚ) :
  (urn1_green / urn1_total) * (urn2_green / (urn2_green + M)) +
  (urn1_blue / urn1_total) * (M / (urn2_green + M)) = same_color_prob →
  M = 610/1657 := by
sorry

end NUMINAMATH_CALUDE_urn_problem_l576_57676


namespace NUMINAMATH_CALUDE_problem_statement_l576_57638

theorem problem_statement (x y : ℝ) (hx : x = 12) (hy : y = 7) :
  (x - y) * (2 * x + y) = 155 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l576_57638

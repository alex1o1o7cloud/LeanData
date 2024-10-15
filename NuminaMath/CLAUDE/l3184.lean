import Mathlib

namespace NUMINAMATH_CALUDE_A_inter_B_equals_open_interval_l3184_318492

def A : Set ℝ := {x | x^2 + 2*x - 3 < 0}
def B : Set ℝ := {x | |x - 1| < 2}

theorem A_inter_B_equals_open_interval : A ∩ B = {x : ℝ | -1 < x ∧ x < 1} := by
  sorry

end NUMINAMATH_CALUDE_A_inter_B_equals_open_interval_l3184_318492


namespace NUMINAMATH_CALUDE_largest_mersenne_prime_factor_of_1000_l3184_318419

def is_mersenne_prime (n : ℕ) : Prop :=
  ∃ p : ℕ, Prime p ∧ n = 2^p - 1 ∧ Prime n

theorem largest_mersenne_prime_factor_of_1000 :
  ∃ n : ℕ, is_mersenne_prime n ∧ n < 500 ∧ n ∣ 1000 ∧
  ∀ m : ℕ, is_mersenne_prime m ∧ m < 500 ∧ m ∣ 1000 → m ≤ n :=
by
  use 3
  sorry

end NUMINAMATH_CALUDE_largest_mersenne_prime_factor_of_1000_l3184_318419


namespace NUMINAMATH_CALUDE_f_min_value_l3184_318477

/-- The function to be minimized -/
def f (x y : ℝ) : ℝ := 3 * x^2 + 4 * x * y + 4 * y^2 - 12 * x - 8 * y

/-- The theorem stating the minimum value and where it occurs -/
theorem f_min_value :
  (∀ x y : ℝ, f x y ≥ -28) ∧
  f (8/3) (-1) = -28 :=
sorry

end NUMINAMATH_CALUDE_f_min_value_l3184_318477


namespace NUMINAMATH_CALUDE_tshirt_sale_problem_l3184_318459

theorem tshirt_sale_problem (sale_duration : ℕ) (black_price white_price : ℚ) 
  (revenue_per_minute : ℚ) (h1 : sale_duration = 25) 
  (h2 : black_price = 30) (h3 : white_price = 25) (h4 : revenue_per_minute = 220) :
  ∃ (total_shirts : ℕ), 
    (total_shirts : ℚ) / 2 * black_price + (total_shirts : ℚ) / 2 * white_price = 
      sale_duration * revenue_per_minute ∧ total_shirts = 200 := by
  sorry

end NUMINAMATH_CALUDE_tshirt_sale_problem_l3184_318459


namespace NUMINAMATH_CALUDE_equation_solution_l3184_318473

theorem equation_solution :
  ∃ x : ℝ, (1 / x + (2 / x) / (4 / x) = 3 / 4) ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3184_318473


namespace NUMINAMATH_CALUDE_vector_magnitude_l3184_318449

/-- Given vectors a and b, if c is parallel to a and perpendicular to b + c, 
    then the magnitude of c is 3√2. -/
theorem vector_magnitude (a b c : ℝ × ℝ) : 
  a = (-1, 1) → 
  b = (-2, 4) → 
  (∃ k : ℝ, c = k • a) →  -- parallel condition
  (a.1 * (b.1 + c.1) + a.2 * (b.2 + c.2) = 0) →  -- perpendicular condition
  ‖c‖ = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_l3184_318449


namespace NUMINAMATH_CALUDE_complement_of_M_in_U_l3184_318490

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define the set M
def M : Set ℝ := {x : ℝ | x ≥ 1}

-- State the theorem
theorem complement_of_M_in_U : 
  (U \ M) = {x : ℝ | x < 1} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_in_U_l3184_318490


namespace NUMINAMATH_CALUDE_train_length_proof_l3184_318493

/-- Given a train that crosses an electric pole in 30 seconds at a speed of 43.2 km/h,
    prove that its length is 360 meters. -/
theorem train_length_proof (crossing_time : ℝ) (speed_kmh : ℝ) (length : ℝ) : 
  crossing_time = 30 →
  speed_kmh = 43.2 →
  length = speed_kmh * 1000 / 3600 * crossing_time →
  length = 360 := by
  sorry

#check train_length_proof

end NUMINAMATH_CALUDE_train_length_proof_l3184_318493


namespace NUMINAMATH_CALUDE_product_divisibility_l3184_318471

/-- Given two lists of positive integers of equal length, where the number of multiples
    of any d > 1 in the first list is no less than that in the second list,
    prove that the product of the first list is divisible by the product of the second list. -/
theorem product_divisibility
  (r : ℕ)
  (m n : List ℕ)
  (h_length : m.length = r ∧ n.length = r)
  (h_positive : ∀ x ∈ m, x > 0) (h_positive' : ∀ y ∈ n, y > 0)
  (h_multiples : ∀ d > 1, (m.filter (· % d = 0)).length ≥ (n.filter (· % d = 0)).length) :
  (m.prod % n.prod = 0) :=
sorry

end NUMINAMATH_CALUDE_product_divisibility_l3184_318471


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l3184_318455

theorem simplify_and_rationalize (x : ℝ) (h : x = Real.sqrt 5) :
  1 / (2 + 2 / (x + 3)) = (7 + x) / 22 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l3184_318455


namespace NUMINAMATH_CALUDE_midpoint_barycentric_coords_l3184_318467

/-- Barycentric coordinates of a point -/
structure BarycentricCoord where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Given two points in barycentric coordinates, compute the midpoint -/
def midpoint_barycentric (M N : BarycentricCoord) : Prop :=
  let m := M.x + M.y + M.z
  let n := N.x + N.y + N.z
  ∃ (k : ℝ) (S : BarycentricCoord), k ≠ 0 ∧
    S.x = k * (M.x / (2 * m) + N.x / (2 * n)) ∧
    S.y = k * (M.y / (2 * m) + N.y / (2 * n)) ∧
    S.z = k * (M.z / (2 * m) + N.z / (2 * n))

theorem midpoint_barycentric_coords (M N : BarycentricCoord) : 
  midpoint_barycentric M N := by sorry

end NUMINAMATH_CALUDE_midpoint_barycentric_coords_l3184_318467


namespace NUMINAMATH_CALUDE_chelsea_needs_52_bullseyes_l3184_318410

/-- Represents the archery contest scenario -/
structure ArcheryContest where
  total_shots : Nat
  chelsea_lead : Nat
  chelsea_min_score : Nat
  opponent_min_score : Nat
  bullseye_score : Nat

/-- Calculates the minimum number of bullseyes needed for Chelsea to guarantee a win -/
def min_bullseyes_needed (contest : ArcheryContest) : Nat :=
  let remaining_shots := contest.total_shots / 2
  let max_opponent_gain := remaining_shots * contest.bullseye_score
  let chelsea_gain_per_bullseye := contest.bullseye_score - contest.chelsea_min_score
  ((max_opponent_gain - contest.chelsea_lead) / chelsea_gain_per_bullseye) + 1

/-- Theorem stating that Chelsea needs at least 52 bullseyes to guarantee a win -/
theorem chelsea_needs_52_bullseyes (contest : ArcheryContest) 
  (h1 : contest.total_shots = 120)
  (h2 : contest.chelsea_lead = 60)
  (h3 : contest.chelsea_min_score = 3)
  (h4 : contest.opponent_min_score = 1)
  (h5 : contest.bullseye_score = 10) :
  min_bullseyes_needed contest ≥ 52 := by
  sorry

#eval min_bullseyes_needed { total_shots := 120, chelsea_lead := 60, chelsea_min_score := 3, opponent_min_score := 1, bullseye_score := 10 }

end NUMINAMATH_CALUDE_chelsea_needs_52_bullseyes_l3184_318410


namespace NUMINAMATH_CALUDE_election_votes_l3184_318478

theorem election_votes (total_votes : ℕ) 
  (winning_percentage : ℚ) (vote_majority : ℕ) :
  winning_percentage = 70 / 100 →
  vote_majority = 192 →
  (winning_percentage * total_votes - (1 - winning_percentage) * total_votes : ℚ) = vote_majority →
  total_votes = 480 :=
by sorry

end NUMINAMATH_CALUDE_election_votes_l3184_318478


namespace NUMINAMATH_CALUDE_lisas_number_l3184_318431

theorem lisas_number (n : ℕ) : 
  (∃ k : ℕ, n = 150 * k) ∧ 
  (∃ m : ℕ, n = 45 * m) ∧ 
  1000 ≤ n ∧ n < 3000 ∧
  (∀ x : ℕ, (∃ i : ℕ, x = 150 * i) ∧ (∃ j : ℕ, x = 45 * j) ∧ 1000 ≤ x ∧ x < 3000 → n ≤ x) →
  n = 1350 := by
sorry

end NUMINAMATH_CALUDE_lisas_number_l3184_318431


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3184_318496

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

/-- The theorem statement -/
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  a 2 + a 6 = 3 →
  a 6 + a 10 = 12 →
  a 8 + a 12 = 24 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3184_318496


namespace NUMINAMATH_CALUDE_expression_simplification_l3184_318463

theorem expression_simplification (m : ℝ) (h : m = Real.sqrt 3) :
  (m - (m + 9) / (m + 1)) / ((m^2 + 3*m) / (m + 1)) = 1 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3184_318463


namespace NUMINAMATH_CALUDE_horner_method_for_f_l3184_318403

/-- Horner's method representation of a polynomial -/
def horner_rep (a : List ℝ) (x : ℝ) : ℝ :=
  a.foldl (fun acc coeff => acc * x + coeff) 0

/-- The original polynomial function -/
def f (x : ℝ) : ℝ := x^5 + 2*x^4 + x^3 - x^2 + 3*x - 5

theorem horner_method_for_f :
  f 5 = horner_rep [1, 2, 1, -1, 3, -5] 5 ∧ 
  horner_rep [1, 2, 1, -1, 3, -5] 5 = 4485 :=
sorry

end NUMINAMATH_CALUDE_horner_method_for_f_l3184_318403


namespace NUMINAMATH_CALUDE_cube_surface_area_l3184_318446

theorem cube_surface_area (edge_length : ℝ) (h : edge_length = 1) : 
  6 * edge_length^2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_l3184_318446


namespace NUMINAMATH_CALUDE_hexagon_diagonal_small_triangle_l3184_318422

/-- A convex hexagon in the plane -/
structure ConvexHexagon where
  -- We don't need to define the specific properties of a convex hexagon for this statement
  area : ℝ
  area_pos : area > 0

/-- A diagonal of a hexagon -/
structure Diagonal (h : ConvexHexagon) where
  -- We don't need to define the specific properties of a diagonal for this statement

/-- The area of the triangle cut off by a diagonal -/
noncomputable def triangle_area (h : ConvexHexagon) (d : Diagonal h) : ℝ :=
  sorry -- Definition not provided, as it's not part of the original conditions

theorem hexagon_diagonal_small_triangle (h : ConvexHexagon) :
  ∃ (d : Diagonal h), triangle_area h d ≤ h.area / 6 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_diagonal_small_triangle_l3184_318422


namespace NUMINAMATH_CALUDE_prob_at_least_one_head_correct_l3184_318448

/-- The probability of getting at least one head when tossing 3 coins simultaneously -/
def prob_at_least_one_head : ℚ :=
  7/8

/-- The number of coins being tossed simultaneously -/
def num_coins : ℕ := 3

/-- The probability of getting heads on a single coin toss -/
def prob_heads : ℚ := 1/2

theorem prob_at_least_one_head_correct :
  prob_at_least_one_head = 1 - (1 - prob_heads) ^ num_coins :=
by sorry


end NUMINAMATH_CALUDE_prob_at_least_one_head_correct_l3184_318448


namespace NUMINAMATH_CALUDE_negation_equivalence_l3184_318434

theorem negation_equivalence :
  (¬ ∀ a b : ℝ, a = b → a^2 = a*b) ↔ (∀ a b : ℝ, a ≠ b → a^2 ≠ a*b) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3184_318434


namespace NUMINAMATH_CALUDE_barn_paint_area_l3184_318497

/-- Represents the dimensions of a rectangular barn -/
structure BarnDimensions where
  width : ℝ
  length : ℝ
  height : ℝ

/-- Calculates the total area to be painted for a rectangular barn -/
def totalPaintArea (dims : BarnDimensions) : ℝ :=
  2 * (2 * dims.width * dims.height + 2 * dims.length * dims.height) + dims.width * dims.length

/-- Theorem stating that the total area to be painted for the given barn is 654 sq yd -/
theorem barn_paint_area :
  let dims : BarnDimensions := { width := 11, length := 14, height := 6 }
  totalPaintArea dims = 654 := by sorry

end NUMINAMATH_CALUDE_barn_paint_area_l3184_318497


namespace NUMINAMATH_CALUDE_complement_of_angle_l3184_318484

theorem complement_of_angle (A : ℝ) : A = 35 → 180 - A = 145 := by
  sorry

end NUMINAMATH_CALUDE_complement_of_angle_l3184_318484


namespace NUMINAMATH_CALUDE_always_negative_l3184_318456

-- Define the chessboard as a function from positions to integers
def Chessboard := Fin 8 → Fin 8 → Int

-- Initial configuration of the chessboard
def initial_board : Chessboard :=
  fun row col => if row = 1 ∧ col = 1 then -1 else 1

-- Define a single operation (flipping signs in a row or column)
def flip_row_or_col (board : Chessboard) (is_row : Bool) (index : Fin 8) : Chessboard :=
  fun row col => 
    if (is_row ∧ row = index) ∨ (¬is_row ∧ col = index) then
      -board row col
    else
      board row col

-- Define a sequence of operations
def apply_operations (board : Chessboard) (ops : List (Bool × Fin 8)) : Chessboard :=
  ops.foldl (fun b (is_row, index) => flip_row_or_col b is_row index) board

-- Theorem statement
theorem always_negative (ops : List (Bool × Fin 8)) :
  ∃ row col, (apply_operations initial_board ops) row col < 0 := by
  sorry

end NUMINAMATH_CALUDE_always_negative_l3184_318456


namespace NUMINAMATH_CALUDE_machine_does_not_require_repair_l3184_318453

/-- Represents a weighing machine for food portions --/
structure WeighingMachine where
  max_deviation : ℝ
  nominal_mass : ℝ
  unreadable_deviation_bound : ℝ

/-- Determines if a weighing machine requires repair --/
def requires_repair (m : WeighingMachine) : Prop :=
  m.max_deviation > 0.1 * m.nominal_mass ∨ 
  m.max_deviation < m.unreadable_deviation_bound

/-- Theorem: The weighing machine does not require repair --/
theorem machine_does_not_require_repair (m : WeighingMachine) 
  (h1 : m.max_deviation = 37)
  (h2 : m.max_deviation ≤ 0.1 * m.nominal_mass)
  (h3 : m.unreadable_deviation_bound < m.max_deviation) :
  ¬(requires_repair m) := by
  sorry

#check machine_does_not_require_repair

end NUMINAMATH_CALUDE_machine_does_not_require_repair_l3184_318453


namespace NUMINAMATH_CALUDE_property_set_characterization_l3184_318436

/-- A number is a prime power if it's of the form p^k where p is prime and k ≥ 1 -/
def IsPrimePower (n : Nat) : Prop :=
  ∃ (p k : Nat), Prime p ∧ k ≥ 1 ∧ n = p^k

/-- A perfect square n satisfies the property if for all its divisors a ≥ 15, a + 15 is a prime power -/
def SatisfiesProperty (n : Nat) : Prop :=
  ∃ m : Nat, n = m^2 ∧ ∀ a : Nat, a ≥ 15 → a ∣ n → IsPrimePower (a + 15)

/-- The set of all perfect squares satisfying the property -/
def PropertySet : Set Nat :=
  {n : Nat | SatisfiesProperty n}

/-- The theorem stating that the set of perfect squares satisfying the property
    is exactly {1, 4, 9, 16, 49, 64, 196} -/
theorem property_set_characterization :
  PropertySet = {1, 4, 9, 16, 49, 64, 196} := by
  sorry


end NUMINAMATH_CALUDE_property_set_characterization_l3184_318436


namespace NUMINAMATH_CALUDE_distance_is_1000_l3184_318499

/-- The distance between Liang Liang's home and school in meters. -/
def distance : ℝ := sorry

/-- The time taken (in minutes) when walking at 40 meters per minute. -/
def time_at_40 : ℝ := sorry

/-- Assertion that distance equals speed multiplied by time for 40 m/min speed. -/
axiom distance_eq_40_times_time : distance = 40 * time_at_40

/-- Assertion that distance equals speed multiplied by time for 50 m/min speed. -/
axiom distance_eq_50_times_time_minus_5 : distance = 50 * (time_at_40 - 5)

theorem distance_is_1000 : distance = 1000 := by sorry

end NUMINAMATH_CALUDE_distance_is_1000_l3184_318499


namespace NUMINAMATH_CALUDE_first_cross_fraction_solution_second_cross_fraction_solution_third_cross_fraction_solution_l3184_318476

/-- Definition of a cross fraction equation -/
def is_cross_fraction_equation (m n x : ℝ) : Prop :=
  m ≠ 0 ∧ n ≠ 0 ∧ x + m * n / x = m + n

/-- Theorem for the first cross fraction equation -/
theorem first_cross_fraction_solution :
  ∀ x₁ x₂ : ℝ, is_cross_fraction_equation (-3) (-4) x₁ ∧ is_cross_fraction_equation (-3) (-4) x₂ →
  (x₁ = -3 ∧ x₂ = -4) ∨ (x₁ = -4 ∧ x₂ = -3) :=
sorry

/-- Theorem for the second cross fraction equation -/
theorem second_cross_fraction_solution :
  ∀ a b : ℝ, is_cross_fraction_equation a b a ∧ is_cross_fraction_equation a b b →
  b / a + a / b + 1 = -31 / 6 :=
sorry

/-- Theorem for the third cross fraction equation -/
theorem third_cross_fraction_solution :
  ∀ k x₁ x₂ : ℝ, k > 2 → x₁ > x₂ →
  is_cross_fraction_equation (2023 * k - 2022) 1 x₁ ∧ is_cross_fraction_equation (2023 * k - 2022) 1 x₂ →
  (x₁ + 4044) / x₂ = 2022 :=
sorry

end NUMINAMATH_CALUDE_first_cross_fraction_solution_second_cross_fraction_solution_third_cross_fraction_solution_l3184_318476


namespace NUMINAMATH_CALUDE_rational_sqrt_property_l3184_318486

theorem rational_sqrt_property (A : Set ℝ) : 
  (∃ a b c d : ℝ, a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ d ∈ A ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) →
  (∀ a b c : ℝ, a ∈ A → b ∈ A → c ∈ A → a ≠ b → a ≠ c → b ≠ c → ∃ q : ℚ, (a^2 + b*c : ℝ) = q) →
  ∃ M : ℕ, ∀ a : ℝ, a ∈ A → ∃ q : ℚ, a * Real.sqrt M = q :=
by sorry

end NUMINAMATH_CALUDE_rational_sqrt_property_l3184_318486


namespace NUMINAMATH_CALUDE_min_cubes_for_box_l3184_318408

theorem min_cubes_for_box (box_length box_width box_height cube_volume : ℕ) 
  (h1 : box_length = 10)
  (h2 : box_width = 13)
  (h3 : box_height = 5)
  (h4 : cube_volume = 5) :
  (box_length * box_width * box_height) / cube_volume = 130 := by
  sorry

end NUMINAMATH_CALUDE_min_cubes_for_box_l3184_318408


namespace NUMINAMATH_CALUDE_exponent_multiplication_l3184_318404

theorem exponent_multiplication (a : ℝ) (m n : ℕ) : a ^ m * a ^ n = a ^ (m + n) := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l3184_318404


namespace NUMINAMATH_CALUDE_fraction_simplification_l3184_318435

theorem fraction_simplification (x : ℝ) : 
  (2*x - 3)/4 + (3*x + 5)/5 - (x - 1)/2 = (12*x + 15)/20 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3184_318435


namespace NUMINAMATH_CALUDE_girls_combined_average_score_l3184_318491

theorem girls_combined_average_score 
  (f1 l1 f2 l2 : ℕ) 
  (h1 : (71 * f1 + 76 * l1) / (f1 + l1) = 74)
  (h2 : (81 * f2 + 90 * l2) / (f2 + l2) = 84)
  (h3 : (71 * f1 + 81 * f2) / (f1 + f2) = 79)
  : (76 * l1 + 90 * l2) / (l1 + l2) = 84 := by
  sorry


end NUMINAMATH_CALUDE_girls_combined_average_score_l3184_318491


namespace NUMINAMATH_CALUDE_initial_nurses_count_l3184_318440

/-- Proves that the initial number of nurses is 18 given the conditions of the problem -/
theorem initial_nurses_count (initial_doctors : ℕ) (quit_doctors quit_nurses remaining_staff : ℕ) 
  (h1 : initial_doctors = 11)
  (h2 : quit_doctors = 5)
  (h3 : quit_nurses = 2)
  (h4 : remaining_staff = 22)
  (h5 : initial_doctors - quit_doctors + (initial_nurses - quit_nurses) = remaining_staff) :
  initial_nurses = 18 :=
by
  sorry
where
  initial_nurses : ℕ := by sorry

end NUMINAMATH_CALUDE_initial_nurses_count_l3184_318440


namespace NUMINAMATH_CALUDE_intersection_point_is_unique_l3184_318450

/-- The point of intersection of two lines -/
def intersection_point : ℚ × ℚ := (-15/8, 13/4)

/-- First line equation -/
def line1 (x y : ℚ) : Prop := 3 * y = -2 * x + 6

/-- Second line equation -/
def line2 (x y : ℚ) : Prop := 4 * y = -8 * x - 2

theorem intersection_point_is_unique :
  (line1 intersection_point.1 intersection_point.2) ∧
  (line2 intersection_point.1 intersection_point.2) ∧
  (∀ x y : ℚ, line1 x y ∧ line2 x y → (x, y) = intersection_point) :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_is_unique_l3184_318450


namespace NUMINAMATH_CALUDE_simplify_expression_l3184_318413

theorem simplify_expression (p : ℝ) : ((6*p+2)-3*p*5)^2 + (5-2/4)*(8*p-12) = 81*p^2 - 50 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3184_318413


namespace NUMINAMATH_CALUDE_max_value_on_interval_solution_set_inequality_l3184_318447

-- Define the function f
def f (x : ℝ) : ℝ := (x + 2) * |x - 2|

-- Theorem for the maximum value of f on [-3, 1]
theorem max_value_on_interval :
  ∃ (m : ℝ), m = 4 ∧ ∀ x ∈ Set.Icc (-3) 1, f x ≤ m :=
sorry

-- Theorem for the solution set of f(x) > 3x
theorem solution_set_inequality :
  {x : ℝ | f x > 3 * x} = {x : ℝ | x > 4 ∨ (-4 < x ∧ x < 1)} :=
sorry

end NUMINAMATH_CALUDE_max_value_on_interval_solution_set_inequality_l3184_318447


namespace NUMINAMATH_CALUDE_books_per_shelf_l3184_318458

theorem books_per_shelf (total_books : ℕ) (num_shelves : ℕ) 
  (h1 : total_books = 504) (h2 : num_shelves = 9) :
  total_books / num_shelves = 56 := by
sorry

end NUMINAMATH_CALUDE_books_per_shelf_l3184_318458


namespace NUMINAMATH_CALUDE_magnitude_of_complex_square_l3184_318479

theorem magnitude_of_complex_square : 
  let z : ℂ := (3 + Complex.I) ^ 2
  ‖z‖ = 10 := by sorry

end NUMINAMATH_CALUDE_magnitude_of_complex_square_l3184_318479


namespace NUMINAMATH_CALUDE_second_investment_amount_l3184_318451

/-- Proves that the amount of the second investment is $1500 given the conditions of the problem -/
theorem second_investment_amount
  (total_return : ℝ → ℝ → ℝ)
  (first_investment : ℝ)
  (first_return_rate : ℝ)
  (second_return_rate : ℝ)
  (total_return_rate : ℝ)
  (h1 : first_investment = 500)
  (h2 : first_return_rate = 0.07)
  (h3 : second_return_rate = 0.09)
  (h4 : total_return_rate = 0.085)
  (h5 : ∀ x, total_return first_investment x = total_return_rate * (first_investment + x))
  (h6 : ∀ x, total_return first_investment x = first_return_rate * first_investment + second_return_rate * x) :
  ∃ x, x = 1500 ∧ total_return first_investment x = total_return_rate * (first_investment + x) :=
by sorry

end NUMINAMATH_CALUDE_second_investment_amount_l3184_318451


namespace NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l3184_318405

/-- An arithmetic sequence is increasing if its common difference is positive -/
def IsIncreasingArithmeticSequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  (∀ n, a (n + 1) = a n + d) ∧ d > 0

/-- The sum of the first three terms of an arithmetic sequence -/
def SumFirstThree (a : ℕ → ℝ) : ℝ :=
  a 1 + a 2 + a 3

/-- The product of the first three terms of an arithmetic sequence -/
def ProductFirstThree (a : ℕ → ℝ) : ℝ :=
  a 1 * a 2 * a 3

theorem arithmetic_sequence_first_term
  (a : ℕ → ℝ) (d : ℝ)
  (h_increasing : IsIncreasingArithmeticSequence a d)
  (h_sum : SumFirstThree a = 12)
  (h_product : ProductFirstThree a = 48) :
  a 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l3184_318405


namespace NUMINAMATH_CALUDE_expand_polynomial_l3184_318482

theorem expand_polynomial (x : ℝ) : (5*x^2 + 3*x - 7) * 4*x^3 = 20*x^5 + 12*x^4 - 28*x^3 := by
  sorry

end NUMINAMATH_CALUDE_expand_polynomial_l3184_318482


namespace NUMINAMATH_CALUDE_files_per_folder_l3184_318427

theorem files_per_folder (initial_files : ℕ) (deleted_files : ℕ) (num_folders : ℕ) :
  initial_files = 43 →
  deleted_files = 31 →
  num_folders = 2 →
  num_folders > 0 →
  ∃ (files_per_folder : ℕ),
    files_per_folder * num_folders = initial_files - deleted_files ∧
    files_per_folder = 6 :=
by sorry

end NUMINAMATH_CALUDE_files_per_folder_l3184_318427


namespace NUMINAMATH_CALUDE_parallel_vectors_k_eq_two_l3184_318441

/-- Two vectors in ℝ² are parallel if and only if their components are proportional -/
axiom vector_parallel_iff_proportional {a b : ℝ × ℝ} :
  (∃ (t : ℝ), a = (t * b.1, t * b.2)) ↔ ∃ (s : ℝ), a.1 * b.2 = s * a.2 * b.1

/-- Given vectors a = (k, 2) and b = (1, 1), if a is parallel to b, then k = 2 -/
theorem parallel_vectors_k_eq_two (k : ℝ) :
  let a : ℝ × ℝ := (k, 2)
  let b : ℝ × ℝ := (1, 1)
  (∃ (t : ℝ), a = (t * b.1, t * b.2)) → k = 2 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_k_eq_two_l3184_318441


namespace NUMINAMATH_CALUDE_geometric_to_arithmetic_sequence_l3184_318420

theorem geometric_to_arithmetic_sequence :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
  (∃ (q : ℝ), q ≠ 0 ∧ b = a * q ∧ c = b * q) ∧
  a + b + c = 19 ∧
  b - a = (c - 1) - b :=
by sorry

end NUMINAMATH_CALUDE_geometric_to_arithmetic_sequence_l3184_318420


namespace NUMINAMATH_CALUDE_partitioned_rectangle_is_square_l3184_318400

-- Define the structure for a rectangle
structure Rectangle where
  width : ℝ
  height : ℝ

-- Define the structure for the partitioned rectangle
structure PartitionedRectangle where
  main : Rectangle
  p1 : Rectangle
  p2 : Rectangle
  p3 : Rectangle
  p4 : Rectangle
  p5 : Rectangle

-- Define the property of being a square
def isSquare (r : Rectangle) : Prop :=
  r.width = r.height

-- Define the property of having equal areas
def equalAreas (r1 r2 r3 r4 : Rectangle) : Prop :=
  r1.width * r1.height = r2.width * r2.height ∧
  r2.width * r2.height = r3.width * r3.height ∧
  r3.width * r3.height = r4.width * r4.height

-- Theorem statement
theorem partitioned_rectangle_is_square 
  (pr : PartitionedRectangle) 
  (h1 : isSquare pr.p5)
  (h2 : equalAreas pr.p1 pr.p2 pr.p3 pr.p4) :
  isSquare pr.main :=
sorry

end NUMINAMATH_CALUDE_partitioned_rectangle_is_square_l3184_318400


namespace NUMINAMATH_CALUDE_screen_height_is_100_l3184_318452

/-- The height of a computer screen given the side length of a square paper and the difference between the screen height and the paper's perimeter. -/
def screen_height (square_side : ℝ) (perimeter_difference : ℝ) : ℝ :=
  4 * square_side + perimeter_difference

/-- Theorem stating that the height of the computer screen is 100 cm. -/
theorem screen_height_is_100 :
  screen_height 20 20 = 100 := by
  sorry

end NUMINAMATH_CALUDE_screen_height_is_100_l3184_318452


namespace NUMINAMATH_CALUDE_gcd_power_two_minus_one_l3184_318474

theorem gcd_power_two_minus_one : 
  Nat.gcd (2^2100 - 1) (2^2000 - 1) = 2^100 - 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_power_two_minus_one_l3184_318474


namespace NUMINAMATH_CALUDE_chicken_crossing_ratio_l3184_318411

theorem chicken_crossing_ratio (initial_feathers final_feathers cars_dodged : ℕ) 
  (h1 : initial_feathers = 5263)
  (h2 : final_feathers = 5217)
  (h3 : cars_dodged = 23) :
  (initial_feathers - final_feathers) / cars_dodged = 2 := by
sorry

end NUMINAMATH_CALUDE_chicken_crossing_ratio_l3184_318411


namespace NUMINAMATH_CALUDE_consecutive_page_numbers_sum_l3184_318429

theorem consecutive_page_numbers_sum (n : ℕ) : 
  n > 0 ∧ n * (n + 1) = 20412 → n + (n + 1) = 287 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_page_numbers_sum_l3184_318429


namespace NUMINAMATH_CALUDE_num_lines_formula_l3184_318430

/-- The number of lines through n points in a plane, where n ≥ 3 and no three points are collinear -/
def num_lines (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: The number of lines through n points in a plane, where n ≥ 3 and no three points are collinear, is n(n-1)/2 -/
theorem num_lines_formula (n : ℕ) (h : n ≥ 3) :
  num_lines n = n * (n - 1) / 2 := by sorry

end NUMINAMATH_CALUDE_num_lines_formula_l3184_318430


namespace NUMINAMATH_CALUDE_reduced_speed_percentage_l3184_318428

theorem reduced_speed_percentage (usual_time : ℝ) (additional_time : ℝ) : 
  usual_time = 24 → additional_time = 24 → 
  (usual_time / (usual_time + additional_time)) * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_reduced_speed_percentage_l3184_318428


namespace NUMINAMATH_CALUDE_irina_square_area_l3184_318421

/-- Given a square with side length 12 cm, if another square has a perimeter 8 cm larger,
    then the area of the second square is 196 cm². -/
theorem irina_square_area (original_side : ℝ) (irina_side : ℝ) : 
  original_side = 12 →
  4 * irina_side = 4 * original_side + 8 →
  irina_side * irina_side = 196 :=
by
  sorry

#check irina_square_area

end NUMINAMATH_CALUDE_irina_square_area_l3184_318421


namespace NUMINAMATH_CALUDE_new_person_weight_l3184_318444

theorem new_person_weight (initial_count : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 8 →
  weight_increase = 4.5 →
  replaced_weight = 65 →
  (initial_count : ℝ) * weight_increase + replaced_weight = 101 :=
by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_l3184_318444


namespace NUMINAMATH_CALUDE_rms_geq_cube_root_avg_product_l3184_318485

theorem rms_geq_cube_root_avg_product (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  Real.sqrt ((a^2 + b^2 + c^2 + d^2) / 4) ≥ (((a*b*c + a*b*d + a*c*d + b*c*d) / 4) ^ (1/3 : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_rms_geq_cube_root_avg_product_l3184_318485


namespace NUMINAMATH_CALUDE_floor_sum_equals_140_l3184_318418

theorem floor_sum_equals_140 
  (p q r s : ℝ) 
  (pos_p : 0 < p) (pos_q : 0 < q) (pos_r : 0 < r) (pos_s : 0 < s)
  (sum_squares : p^2 + q^2 = 2512 ∧ r^2 + s^2 = 2512)
  (products : p * r = 1225 ∧ q * s = 1225) : 
  ⌊p + q + r + s⌋ = 140 := by
sorry

end NUMINAMATH_CALUDE_floor_sum_equals_140_l3184_318418


namespace NUMINAMATH_CALUDE_cube_fraction_equals_150_l3184_318445

theorem cube_fraction_equals_150 :
  (68^3 - 65^3) * (32^3 + 18^3) / ((32^2 - 32 * 18 + 18^2) * (68^2 + 68 * 65 + 65^2)) = 150 := by
  sorry

end NUMINAMATH_CALUDE_cube_fraction_equals_150_l3184_318445


namespace NUMINAMATH_CALUDE_max_sum_of_factors_l3184_318423

theorem max_sum_of_factors (x y : ℕ+) : 
  x.val * y.val = 48 → 
  4 ∣ x.val → 
  ∀ (a b : ℕ+), a.val * b.val = 48 → 4 ∣ a.val → a + b ≤ x + y → 
  x + y = 49 := by sorry

end NUMINAMATH_CALUDE_max_sum_of_factors_l3184_318423


namespace NUMINAMATH_CALUDE_parallel_implies_x_half_perpendicular_implies_x_two_or_neg_two_l3184_318466

-- Define the vectors
def a : ℝ × ℝ := (1, 2)
def b (x : ℝ) : ℝ × ℝ := (x, 1)

-- Define u and v
def u (x : ℝ) : ℝ × ℝ := a + b x
def v (x : ℝ) : ℝ × ℝ := a - b x

-- Helper function to check if two vectors are parallel
def isParallel (v1 v2 : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v1.1 * k = v2.1 ∧ v1.2 * k = v2.2

-- Helper function to check if two vectors are perpendicular
def isPerpendicular (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 = 0

-- Theorem for part I
theorem parallel_implies_x_half :
  ∀ x : ℝ, isParallel (u x) (v x) → x = 1/2 := by sorry

-- Theorem for part II
theorem perpendicular_implies_x_two_or_neg_two :
  ∀ x : ℝ, isPerpendicular (u x) (v x) → x = 2 ∨ x = -2 := by sorry

end NUMINAMATH_CALUDE_parallel_implies_x_half_perpendicular_implies_x_two_or_neg_two_l3184_318466


namespace NUMINAMATH_CALUDE_acute_angle_x_l3184_318475

theorem acute_angle_x (x : Real) (h : 0 < x ∧ x < π / 2) 
  (eq : Real.sin (3 * π / 5) * Real.cos x + Real.cos (2 * π / 5) * Real.sin x = Real.sqrt 3 / 2) : 
  x = 4 * π / 15 := by
  sorry

end NUMINAMATH_CALUDE_acute_angle_x_l3184_318475


namespace NUMINAMATH_CALUDE_circle_condition_intersection_condition_l3184_318472

-- Define the equation C
def C (x y m : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y + m = 0

-- Define the line l
def l (x y : ℝ) : Prop := x + 2*y - 4 = 0

-- Theorem 1: Range of m for C to represent a circle
theorem circle_condition (m : ℝ) :
  (∃ x y, C x y m) ∧ (∀ x y, C x y m → (x - 1)^2 + (y - 2)^2 = 5 - m) →
  m < 5 :=
sorry

-- Theorem 2: Value of m when C intersects l with |MN| = 4√5/5
theorem intersection_condition (m : ℝ) :
  (∃ x₁ y₁ x₂ y₂, C x₁ y₁ m ∧ C x₂ y₂ m ∧ l x₁ y₁ ∧ l x₂ y₂ ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = (4*Real.sqrt 5 / 5)^2) →
  m = 4 :=
sorry

end NUMINAMATH_CALUDE_circle_condition_intersection_condition_l3184_318472


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3184_318402

theorem quadratic_inequality_solution (p : ℝ) : 
  (∀ x, x^2 + p*x - 6 < 0 ↔ -3 < x ∧ x < 2) → p = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3184_318402


namespace NUMINAMATH_CALUDE_probability_of_humanities_course_l3184_318409

/-- Represents a course --/
inductive Course
| Mathematics
| Chinese
| Politics
| Geography
| English
| History
| PhysicalEducation

/-- Represents the time of day --/
inductive TimeOfDay
| Morning
| Afternoon

/-- Defines whether a course is in humanities and social sciences --/
def isHumanities (c : Course) : Bool :=
  match c with
  | Course.Politics | Course.History | Course.Geography => true
  | _ => false

/-- Defines the courses available in each time slot --/
def availableCourses (t : TimeOfDay) : List Course :=
  match t with
  | TimeOfDay.Morning => [Course.Mathematics, Course.Chinese, Course.Politics, Course.Geography]
  | TimeOfDay.Afternoon => [Course.English, Course.History, Course.PhysicalEducation]

theorem probability_of_humanities_course :
  let totalChoices := (availableCourses TimeOfDay.Morning).length * (availableCourses TimeOfDay.Afternoon).length
  let humanitiesChoices := totalChoices - ((availableCourses TimeOfDay.Morning).filter (fun c => !isHumanities c)).length *
                                          ((availableCourses TimeOfDay.Afternoon).filter (fun c => !isHumanities c)).length
  (humanitiesChoices : ℚ) / totalChoices = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_humanities_course_l3184_318409


namespace NUMINAMATH_CALUDE_total_pets_is_108_l3184_318489

/-- The total number of pets owned by Teddy, Ben, and Dave -/
def totalPets : ℕ :=
  let teddy_initial_dogs : ℕ := 7
  let teddy_initial_cats : ℕ := 8
  let teddy_initial_rabbits : ℕ := 6
  let teddy_adopted_dogs : ℕ := 2
  let teddy_adopted_rabbits : ℕ := 4
  
  let teddy_final_dogs : ℕ := teddy_initial_dogs + teddy_adopted_dogs
  let teddy_final_cats : ℕ := teddy_initial_cats
  let teddy_final_rabbits : ℕ := teddy_initial_rabbits + teddy_adopted_rabbits
  
  let ben_dogs : ℕ := 3 * teddy_initial_dogs
  let ben_cats : ℕ := 2 * teddy_final_cats
  
  let dave_dogs : ℕ := teddy_final_dogs - 4
  let dave_cats : ℕ := teddy_final_cats + 13
  let dave_rabbits : ℕ := 3 * teddy_initial_rabbits
  
  let teddy_total : ℕ := teddy_final_dogs + teddy_final_cats + teddy_final_rabbits
  let ben_total : ℕ := ben_dogs + ben_cats
  let dave_total : ℕ := dave_dogs + dave_cats + dave_rabbits
  
  teddy_total + ben_total + dave_total

theorem total_pets_is_108 : totalPets = 108 := by
  sorry

end NUMINAMATH_CALUDE_total_pets_is_108_l3184_318489


namespace NUMINAMATH_CALUDE_identify_brother_l3184_318468

-- Define the brothers
inductive Brother : Type
| Tweedledum : Brother
| Tweedledee : Brother

-- Define the card suits
inductive Suit : Type
| Red : Suit
| Black : Suit

-- Define the statement made by one of the brothers
def statement (b : Brother) (s : Suit) : Prop :=
  b = Brother.Tweedledum ∨ s = Suit.Black

-- Define the rule that someone with a black card cannot make a true statement
axiom black_card_rule : ∀ (b : Brother) (s : Suit), 
  s = Suit.Black → ¬(statement b s)

-- Theorem to prove
theorem identify_brother : 
  ∃ (b : Brother) (s : Suit), statement b s ∧ b = Brother.Tweedledum ∧ s = Suit.Red :=
sorry

end NUMINAMATH_CALUDE_identify_brother_l3184_318468


namespace NUMINAMATH_CALUDE_zoe_dolphin_show_pictures_l3184_318470

/-- Represents the number of pictures Zoe took in different scenarios -/
structure ZoePictures where
  before_dolphin_show : ℕ
  total : ℕ
  remaining_film : ℕ

/-- Calculates the number of pictures Zoe took at the dolphin show -/
def pictures_at_dolphin_show (z : ZoePictures) : ℕ :=
  z.total - z.before_dolphin_show

/-- Theorem stating that for Zoe's specific scenario, she took 16 pictures at the dolphin show -/
theorem zoe_dolphin_show_pictures (z : ZoePictures) 
  (h1 : z.before_dolphin_show = 28)
  (h2 : z.remaining_film = 32)
  (h3 : z.total = 44) : 
  pictures_at_dolphin_show z = 16 := by
  sorry

end NUMINAMATH_CALUDE_zoe_dolphin_show_pictures_l3184_318470


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_is_two_l3184_318424

-- Define the complex number (2+i)i
def z : ℂ := (2 + Complex.I) * Complex.I

-- Theorem statement
theorem imaginary_part_of_z_is_two : Complex.im z = 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_is_two_l3184_318424


namespace NUMINAMATH_CALUDE_largest_number_l3184_318462

theorem largest_number (a b c : ℝ) : 
  a = (1 : ℝ) / 2 →
  b = Real.log 3 / Real.log 4 →
  c = Real.sin (π / 8) →
  b ≥ a ∧ b ≥ c :=
by sorry

end NUMINAMATH_CALUDE_largest_number_l3184_318462


namespace NUMINAMATH_CALUDE_average_income_B_C_l3184_318494

def average_income (x y : ℕ) : ℚ := (x + y) / 2

theorem average_income_B_C 
  (h1 : average_income A_income B_income = 4050)
  (h2 : average_income A_income C_income = 4200)
  (h3 : A_income = 3000) :
  average_income B_income C_income = 5250 :=
by
  sorry


end NUMINAMATH_CALUDE_average_income_B_C_l3184_318494


namespace NUMINAMATH_CALUDE_metallic_sheet_width_l3184_318465

/-- Given a rectangular metallic sheet, this theorem proves that if the length is 48 meters,
    a 3-meter square is cut from each corner, and the resulting open box has a volume of 3780 m³,
    then the original width of the sheet must be 36 meters. -/
theorem metallic_sheet_width (length : ℝ) (width : ℝ) (cut_size : ℝ) (volume : ℝ) :
  length = 48 →
  cut_size = 3 →
  volume = 3780 →
  volume = (length - 2 * cut_size) * (width - 2 * cut_size) * cut_size →
  width = 36 := by
  sorry

#check metallic_sheet_width

end NUMINAMATH_CALUDE_metallic_sheet_width_l3184_318465


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l3184_318495

theorem complex_fraction_equality : 
  (Complex.I : ℂ) ^ 2 = -1 → 
  (1 + Complex.I) ^ 3 / (1 - Complex.I) ^ 2 = -1 - Complex.I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l3184_318495


namespace NUMINAMATH_CALUDE_sum_of_segments_eq_165_l3184_318437

/-- The sum of lengths of all possible line segments formed by dividing a line segment of length 9 into 9 equal parts -/
def sum_of_segments : ℕ :=
  let n : ℕ := 9  -- number of divisions
  (n * (n + 1) * (n + 2)) / 6

/-- Theorem: The sum of lengths of all possible line segments formed by dividing a line segment of length 9 into 9 equal parts is equal to 165 -/
theorem sum_of_segments_eq_165 : sum_of_segments = 165 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_segments_eq_165_l3184_318437


namespace NUMINAMATH_CALUDE_house_height_difference_l3184_318469

/-- Given three house heights, proves that the difference between the average height and 80 feet is 3 feet -/
theorem house_height_difference (h1 h2 h3 : ℕ) (h1_eq : h1 = 80) (h2_eq : h2 = 70) (h3_eq : h3 = 99) :
  (h1 + h2 + h3) / 3 - h1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_house_height_difference_l3184_318469


namespace NUMINAMATH_CALUDE_part_one_part_two_l3184_318438

/-- Definition of the sequence sum -/
def S (n : ℕ) (a : ℝ) : ℝ := a * 2^n - 1

/-- Definition of the sequence terms -/
def a (n : ℕ) (a : ℝ) : ℝ := S n a - S (n-1) a

/-- Part 1: Prove the values of a_1 and a_4 when a = 3 -/
theorem part_one :
  a 1 3 = 5 ∧ a 4 3 = 24 :=
sorry

/-- Definition of geometric sequence -/
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n+1) = r * a n

/-- Part 2: Prove the value of a when {a_n} is a geometric sequence -/
theorem part_two :
  ∃ f : ℕ → ℝ, is_geometric_sequence f ∧ (∀ n : ℕ, S n 1 = f n - f 0) :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3184_318438


namespace NUMINAMATH_CALUDE_cesar_watched_fraction_l3184_318432

theorem cesar_watched_fraction (total_seasons : ℕ) (episodes_per_season : ℕ) (remaining_episodes : ℕ) :
  total_seasons = 12 →
  episodes_per_season = 20 →
  remaining_episodes = 160 →
  (total_seasons * episodes_per_season - remaining_episodes) / (total_seasons * episodes_per_season) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cesar_watched_fraction_l3184_318432


namespace NUMINAMATH_CALUDE_union_equals_A_implies_m_zero_or_three_l3184_318401

-- Define the sets A and B
def A (m : ℝ) : Set ℝ := {1, 3, Real.sqrt m}
def B (m : ℝ) : Set ℝ := {1, m}

-- State the theorem
theorem union_equals_A_implies_m_zero_or_three (m : ℝ) :
  A m ∪ B m = A m → m = 0 ∨ m = 3 := by
  sorry

end NUMINAMATH_CALUDE_union_equals_A_implies_m_zero_or_three_l3184_318401


namespace NUMINAMATH_CALUDE_domain_intersection_l3184_318415

def A : Set ℝ := {x : ℝ | x > -1}
def B : Set ℝ := {-1, 0, 1, 2}

theorem domain_intersection : A ∩ B = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_domain_intersection_l3184_318415


namespace NUMINAMATH_CALUDE_problem_statement_l3184_318483

theorem problem_statement : ((18^18 / 18^17)^3 * 9^3) / 3^6 = 5832 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3184_318483


namespace NUMINAMATH_CALUDE_parallelepiped_intersection_length_l3184_318488

/-- A parallelepiped with points A, B, C, D, A₁, B₁, C₁, D₁ -/
structure Parallelepiped (V : Type*) [NormedAddCommGroup V] :=
  (A B C D A₁ B₁ C₁ D₁ : V)

/-- Point X on edge A₁D₁ -/
def X {V : Type*} [NormedAddCommGroup V] (p : Parallelepiped V) : V :=
  p.A₁ + 5 • (p.D₁ - p.A₁)

/-- Point Y on edge BC -/
def Y {V : Type*} [NormedAddCommGroup V] (p : Parallelepiped V) : V :=
  p.B + 3 • (p.C - p.B)

/-- Intersection point Z of plane C₁XY and ray DA -/
noncomputable def Z {V : Type*} [NormedAddCommGroup V] (p : Parallelepiped V) : V :=
  sorry

/-- Theorem stating that DZ = 20 -/
theorem parallelepiped_intersection_length
  {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V] (p : Parallelepiped V) :
  ‖p.D - Z p‖ = 20 ∧ ‖p.B₁ - p.C₁‖ = 14 :=
sorry

end NUMINAMATH_CALUDE_parallelepiped_intersection_length_l3184_318488


namespace NUMINAMATH_CALUDE_m_range_theorem_l3184_318426

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem m_range_theorem (f : ℝ → ℝ) (m : ℝ) 
  (h_odd : is_odd_function f)
  (h_prop : ∀ x, f (3/4 * x) = f (3/4 * x))
  (h_lower_bound : ∀ x, f x > -2)
  (h_f_1 : f 1 = -3/m) :
  (0 < m ∧ m < 3) ∨ m < -1 := by
sorry

end NUMINAMATH_CALUDE_m_range_theorem_l3184_318426


namespace NUMINAMATH_CALUDE_ones_digit_of_13_power_l3184_318414

-- Define a function to get the ones digit of a natural number
def ones_digit (n : ℕ) : ℕ := n % 10

-- Define the exponent
def exponent : ℕ := 13 * (7^7)

-- Theorem statement
theorem ones_digit_of_13_power : ones_digit (13^exponent) = 7 := by
  sorry

end NUMINAMATH_CALUDE_ones_digit_of_13_power_l3184_318414


namespace NUMINAMATH_CALUDE_line_through_intersection_and_origin_l3184_318433

-- Define the lines l1 and l2
def l1 (x y : ℝ) : Prop := 2*x - y + 7 = 0
def l2 (x y : ℝ) : Prop := y = 1 - x

-- Define the intersection point of l1 and l2
def intersection : ℝ × ℝ := ((-2 : ℝ), (3 : ℝ))

-- Define the origin
def origin : ℝ × ℝ := ((0 : ℝ), (0 : ℝ))

-- Theorem statement
theorem line_through_intersection_and_origin :
  ∀ (x y : ℝ), l1 (intersection.1) (intersection.2) ∧ 
               l2 (intersection.1) (intersection.2) ∧ 
               (3*x + 2*y = 0 ↔ ∃ t : ℝ, x = t * (intersection.1 - origin.1) ∧ 
                                        y = t * (intersection.2 - origin.2)) :=
sorry

end NUMINAMATH_CALUDE_line_through_intersection_and_origin_l3184_318433


namespace NUMINAMATH_CALUDE_quadratic_function_positive_l3184_318460

/-- The quadratic function y = ax² - 2ax + 3 -/
def f (a x : ℝ) : ℝ := a * x^2 - 2 * a * x + 3

/-- The set of x values we're interested in -/
def X : Set ℝ := {x | 0 < x ∧ x < 3}

/-- The set of a values that satisfy the condition -/
def A : Set ℝ := {a | -1 ≤ a ∧ a < 0} ∪ {a | 0 < a ∧ a < 3}

theorem quadratic_function_positive (a : ℝ) :
  (∀ x ∈ X, f a x > 0) ↔ a ∈ A :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_positive_l3184_318460


namespace NUMINAMATH_CALUDE_no_linear_term_implies_m_value_l3184_318442

theorem no_linear_term_implies_m_value (m : ℝ) : 
  (∀ x : ℝ, ∃ a b c : ℝ, (x^2 - x + m) * (x - 8) = a * x^3 + b * x^2 + c) → m = -8 :=
sorry

end NUMINAMATH_CALUDE_no_linear_term_implies_m_value_l3184_318442


namespace NUMINAMATH_CALUDE_complex_modulus_of_fraction_l3184_318481

theorem complex_modulus_of_fraction (z : ℂ) : z = (2 - I) / (2 + I) → Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_of_fraction_l3184_318481


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3184_318412

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (x - 1)^4 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  a + a₂ + a₄ = 8 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3184_318412


namespace NUMINAMATH_CALUDE_negative_a_range_l3184_318480

theorem negative_a_range (a : ℝ) :
  a < 0 →
  (∀ x : ℝ, Real.sin x ^ 2 + a * Real.cos x + a ^ 2 ≥ 1 + Real.cos x) →
  a ≤ -2 := by
  sorry

end NUMINAMATH_CALUDE_negative_a_range_l3184_318480


namespace NUMINAMATH_CALUDE_pencil_exchange_coloring_l3184_318457

-- Define a permutation as a bijective function from ℕ to ℕ
def Permutation (n : ℕ) := {f : ℕ → ℕ // Function.Bijective f ∧ ∀ i, i ≥ n → f i = i}

-- Define a coloring as a function from ℕ to a three-element type
def Coloring (n : ℕ) := ℕ → Fin 3

-- The main theorem
theorem pencil_exchange_coloring (n : ℕ) (p : Permutation n) :
  ∃ c : Coloring n, ∀ i < n, c i ≠ c (p.val i) :=
sorry

end NUMINAMATH_CALUDE_pencil_exchange_coloring_l3184_318457


namespace NUMINAMATH_CALUDE_square_diagonal_l3184_318487

theorem square_diagonal (perimeter : ℝ) (h : perimeter = 40) :
  let side := perimeter / 4
  let diagonal := side * Real.sqrt 2
  diagonal = 10 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_square_diagonal_l3184_318487


namespace NUMINAMATH_CALUDE_quartet_performance_count_l3184_318464

/-- Represents the number of songs sung by each friend -/
structure SongCounts where
  lucy : ℕ
  sarah : ℕ
  beth : ℕ
  jane : ℕ

/-- The total number of songs performed by the quartets -/
def total_songs (counts : SongCounts) : ℕ :=
  (counts.lucy + counts.sarah + counts.beth + counts.jane) / 3

theorem quartet_performance_count (counts : SongCounts) :
  counts.lucy = 8 →
  counts.sarah = 5 →
  counts.beth > counts.sarah →
  counts.beth < counts.lucy →
  counts.jane > counts.sarah →
  counts.jane < counts.lucy →
  total_songs counts = 9 := by
  sorry

#eval total_songs ⟨8, 5, 7, 7⟩

end NUMINAMATH_CALUDE_quartet_performance_count_l3184_318464


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l3184_318407

-- Define the sets A and B
def A : Set ℝ := {x | x > 1}
def B : Set ℝ := {y | -1 < y ∧ y < 2}

-- State the theorem
theorem intersection_A_complement_B :
  A ∩ Bᶜ = {x : ℝ | x ≥ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l3184_318407


namespace NUMINAMATH_CALUDE_resistor_value_l3184_318425

/-- Given two identical resistors R₀ connected in series, with a voltmeter reading U across one resistor
    and an ammeter reading I when replacing the voltmeter, prove that R₀ = 9 Ω. -/
theorem resistor_value (R₀ : ℝ) (U I : ℝ) : 
  U = 9 → I = 2 → R₀ = 9 := by
  sorry

end NUMINAMATH_CALUDE_resistor_value_l3184_318425


namespace NUMINAMATH_CALUDE_root_of_equation_l3184_318443

theorem root_of_equation (a b c d : ℝ) (h1 : a + d = 2015) (h2 : b + c = 2015) (h3 : a ≠ c) :
  ∀ x : ℝ, (x - a) * (x - b) = (x - c) * (x - d) → x = 0 := by
  sorry

end NUMINAMATH_CALUDE_root_of_equation_l3184_318443


namespace NUMINAMATH_CALUDE_boys_playing_marbles_l3184_318461

theorem boys_playing_marbles (total_marbles : ℕ) (marbles_per_boy : ℕ) (h1 : total_marbles = 35) (h2 : marbles_per_boy = 7) :
  total_marbles / marbles_per_boy = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_boys_playing_marbles_l3184_318461


namespace NUMINAMATH_CALUDE_petya_has_five_five_ruble_coins_l3184_318406

/-- Represents the coin denominations --/
inductive Denomination
  | One
  | Two
  | Five
  | Ten

/-- Represents Petya's coin collection --/
structure CoinCollection where
  total : Nat
  not_two : Nat
  not_ten : Nat
  not_one : Nat

/-- Calculates the number of five-ruble coins in the collection --/
def count_five_ruble_coins (c : CoinCollection) : Nat :=
  c.total - ((c.total - c.not_two) + (c.total - c.not_ten) + (c.total - c.not_one))

/-- Theorem stating that Petya has 5 five-ruble coins --/
theorem petya_has_five_five_ruble_coins :
  let petya_coins : CoinCollection := {
    total := 25,
    not_two := 19,
    not_ten := 20,
    not_one := 16
  }
  count_five_ruble_coins petya_coins = 5 := by
  sorry

#eval count_five_ruble_coins {
  total := 25,
  not_two := 19,
  not_ten := 20,
  not_one := 16
}

end NUMINAMATH_CALUDE_petya_has_five_five_ruble_coins_l3184_318406


namespace NUMINAMATH_CALUDE_arithmetic_sequence_log_implies_square_product_square_product_not_sufficient_for_arithmetic_sequence_log_l3184_318416

def is_arithmetic_sequence (a b c : ℝ) : Prop :=
  2 * b = a + c

theorem arithmetic_sequence_log_implies_square_product
  (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : is_arithmetic_sequence (Real.log x) (Real.log y) (Real.log z)) :
  y^2 = x * z :=
sorry

theorem square_product_not_sufficient_for_arithmetic_sequence_log :
  ∃ x y z : ℝ, y^2 = x * z ∧ ¬is_arithmetic_sequence (Real.log x) (Real.log y) (Real.log z) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_log_implies_square_product_square_product_not_sufficient_for_arithmetic_sequence_log_l3184_318416


namespace NUMINAMATH_CALUDE_simplify_expression_l3184_318439

theorem simplify_expression : 
  (((Real.sqrt 2 - 1) ^ (-(Real.sqrt 3) + Real.sqrt 5)) / 
   ((Real.sqrt 2 + 1) ^ (Real.sqrt 5 - Real.sqrt 3))) = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3184_318439


namespace NUMINAMATH_CALUDE_kindergarten_tissues_l3184_318498

/-- The number of tissues in a mini tissue box -/
def tissues_per_box : ℕ := 40

/-- The number of students in the first kindergartner group -/
def group1_size : ℕ := 9

/-- The number of students in the second kindergartner group -/
def group2_size : ℕ := 10

/-- The number of students in the third kindergartner group -/
def group3_size : ℕ := 11

/-- The total number of tissues brought by all kindergartner groups -/
def total_tissues : ℕ := (group1_size + group2_size + group3_size) * tissues_per_box

theorem kindergarten_tissues : total_tissues = 1200 := by
  sorry

end NUMINAMATH_CALUDE_kindergarten_tissues_l3184_318498


namespace NUMINAMATH_CALUDE_richards_score_l3184_318454

/-- Richard and Bruno's miniature golf scores -/
def miniature_golf (richard_score bruno_score : ℕ) : Prop :=
  bruno_score = richard_score - 14 ∧ bruno_score = 48

theorem richards_score : ∃ (richard_score : ℕ), miniature_golf richard_score 48 ∧ richard_score = 62 := by
  sorry

end NUMINAMATH_CALUDE_richards_score_l3184_318454


namespace NUMINAMATH_CALUDE_expression_evaluation_l3184_318417

theorem expression_evaluation (a b : ℤ) (ha : a = -1) (hb : b = 1) :
  (a^2 * b - 4 * a * b^2 - 1) - 3 * (a * b^2 - 2 * a^2 * b + 1) = 10 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3184_318417

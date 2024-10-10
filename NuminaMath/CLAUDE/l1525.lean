import Mathlib

namespace square_sum_given_product_and_sum_of_squares_l1525_152546

theorem square_sum_given_product_and_sum_of_squares (a b : ℝ) 
  (h1 : a * b = 3) 
  (h2 : a^2 * b + a * b^2 = 15) : 
  a^2 + b^2 = 19 := by sorry

end square_sum_given_product_and_sum_of_squares_l1525_152546


namespace square_area_is_26_l1525_152519

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The square of the distance between two points -/
def squaredDistance (p q : Point) : ℝ :=
  (p.x - q.x)^2 + (p.y - q.y)^2

/-- The area of a square given its four vertices -/
def squareArea (p q r s : Point) : ℝ :=
  squaredDistance p q

theorem square_area_is_26 : 
  let p : Point := ⟨1, 2⟩
  let q : Point := ⟨-4, 3⟩
  let r : Point := ⟨-3, -2⟩
  let s : Point := ⟨2, -3⟩
  squareArea p q r s = 26 := by
  sorry

end square_area_is_26_l1525_152519


namespace triangle_problem_l1525_152524

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  (0 < A) → (A < π) →
  (0 < B) → (B < π) →
  (0 < C) → (C < π) →
  (A + B + C = π) →
  (2 * c = Real.sqrt 3 * a + 2 * b * Real.cos A) →
  (c = 7) →
  (b * Real.sin A = Real.sqrt 3) →
  (B = π / 6 ∧ b = Real.sqrt 19) := by
sorry

end triangle_problem_l1525_152524


namespace divisible_by_thirteen_l1525_152530

theorem divisible_by_thirteen (n : ℕ) : ∃ k : ℤ, (7^(2*n) + 10^(n+1) + 2 * 10^n) = 13 * k := by
  sorry

end divisible_by_thirteen_l1525_152530


namespace max_distance_to_origin_l1525_152531

open Complex

theorem max_distance_to_origin (z : ℂ) (h_norm : abs z = 1) : 
  let w := 2*z - Complex.I*z
  ∀ ε > 0, abs w ≤ 3 + ε :=
by sorry

end max_distance_to_origin_l1525_152531


namespace coins_missing_l1525_152578

theorem coins_missing (total : ℚ) : 
  let lost := (1 : ℚ) / 3 * total
  let found := (3 : ℚ) / 4 * lost
  let remaining := total - lost + found
  total - remaining = (1 : ℚ) / 12 * total := by
sorry

end coins_missing_l1525_152578


namespace buoy_radius_l1525_152564

/-- The radius of a spherical buoy given the dimensions of the hole it leaves in ice --/
theorem buoy_radius (hole_diameter : ℝ) (hole_depth : ℝ) (buoy_radius : ℝ) : 
  hole_diameter = 30 → hole_depth = 12 → buoy_radius = 15.375 := by
  sorry

#check buoy_radius

end buoy_radius_l1525_152564


namespace room_length_from_carpet_cost_room_length_is_208_l1525_152545

/-- The length of a room given carpet and cost information -/
theorem room_length_from_carpet_cost (room_width : ℝ) (carpet_width : ℝ) 
  (carpet_cost_per_sqm : ℝ) (total_cost : ℝ) : ℝ :=
  let total_area := total_cost / carpet_cost_per_sqm
  let carpet_width_m := carpet_width / 100
  total_area / carpet_width_m

/-- Proof that the room length is 208 meters given specific conditions -/
theorem room_length_is_208 :
  room_length_from_carpet_cost 9 75 12 1872 = 208 := by
  sorry

end room_length_from_carpet_cost_room_length_is_208_l1525_152545


namespace polyhedron_distance_equation_l1525_152517

/-- A convex polyhedron with 12 regular triangular faces -/
structure Polyhedron :=
  (e : ℝ)  -- Common edge length
  (t : ℝ)  -- Additional length in the distance between non-adjacent five-edged vertices

/-- The distance between two non-adjacent five-edged vertices is (e+t) -/
def distance (p : Polyhedron) : ℝ := p.e + p.t

/-- Theorem: For the given polyhedron, t³ - 7et² + 2e³ = 0 -/
theorem polyhedron_distance_equation (p : Polyhedron) : 
  p.t^3 - 7 * p.e * p.t^2 + 2 * p.e^3 = 0 :=
sorry

end polyhedron_distance_equation_l1525_152517


namespace box_height_is_five_l1525_152567

-- Define the box dimensions and cube properties
def box_length : ℝ := 8
def box_width : ℝ := 15
def cube_volume : ℝ := 10
def min_cubes : ℕ := 60

-- Define the theorem
theorem box_height_is_five :
  let total_volume := (min_cubes : ℝ) * cube_volume
  let height := total_volume / (box_length * box_width)
  height = 5 := by sorry

end box_height_is_five_l1525_152567


namespace shortest_distance_to_E_l1525_152509

/-- Represents a point on the grid -/
structure GridPoint where
  x : Nat
  y : Nat

/-- Calculate the distance between two points on the grid -/
def gridDistance (p1 p2 : GridPoint) : Nat :=
  (p2.x - p1.x) + (p2.y - p1.y)

theorem shortest_distance_to_E :
  let P : GridPoint := ⟨0, 0⟩
  let A : GridPoint := ⟨5, 4⟩
  let B : GridPoint := ⟨6, 2⟩
  let C : GridPoint := ⟨3, 3⟩
  let D : GridPoint := ⟨5, 1⟩
  let E : GridPoint := ⟨1, 4⟩
  (gridDistance P E ≤ gridDistance P A) ∧
  (gridDistance P E ≤ gridDistance P B) ∧
  (gridDistance P E ≤ gridDistance P C) ∧
  (gridDistance P E ≤ gridDistance P D) :=
by sorry

end shortest_distance_to_E_l1525_152509


namespace right_triangle_conditions_l1525_152548

-- Define a structure for a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define what it means for a triangle to be right-angled
def is_right_triangle (t : Triangle) : Prop :=
  t.a^2 + t.b^2 = t.c^2

-- Define the conditions from the problem
def condition_A (t : Triangle) : Prop :=
  ∃ (x : ℝ), x > 0 ∧ t.a = 5*x ∧ t.b = 12*x ∧ t.c = 13*x

def condition_B (t : Triangle) : Prop :=
  ∃ (x : ℝ), x > 0 ∧ t.a = 2*x ∧ t.b = 3*x ∧ t.c = 5*x

def condition_C (t : Triangle) : Prop :=
  ∃ (k : ℝ), k > 0 ∧ t.a = 9*k ∧ t.b = 40*k ∧ t.c = 41*k

def condition_D (t : Triangle) : Prop :=
  t.a = 3^2 ∧ t.b = 4^2 ∧ t.c = 5^2

-- Theorem statement
theorem right_triangle_conditions :
  (∀ t : Triangle, condition_A t → is_right_triangle t) ∧
  (∀ t : Triangle, condition_B t → is_right_triangle t) ∧
  (∀ t : Triangle, condition_C t → is_right_triangle t) ∧
  (∃ t : Triangle, condition_D t ∧ ¬is_right_triangle t) :=
by sorry

end right_triangle_conditions_l1525_152548


namespace min_value_of_a_l1525_152559

theorem min_value_of_a (a b c : ℝ) : 
  a + b + c = 3 → 
  a ≥ b → 
  b ≥ c → 
  ∃ x : ℝ, a * x^2 + b * x + c = 0 → 
  a ≥ 4/3 ∧ ∀ a' : ℝ, (∃ b' c' : ℝ, 
    a' + b' + c' = 3 ∧ 
    a' ≥ b' ∧ 
    b' ≥ c' ∧ 
    (∃ x : ℝ, a' * x^2 + b' * x + c' = 0)) → 
  a' ≥ 4/3 := by
  sorry

end min_value_of_a_l1525_152559


namespace alice_minimum_speed_l1525_152532

-- Define the problem parameters
def distance : ℝ := 180
def bob_speed : ℝ := 40
def alice_delay : ℝ := 0.5

-- Define the theorem
theorem alice_minimum_speed :
  ∀ (alice_speed : ℝ),
  alice_speed > distance / (distance / bob_speed - alice_delay) →
  alice_speed * (distance / bob_speed - alice_delay) > distance :=
by sorry

end alice_minimum_speed_l1525_152532


namespace sqrt_x_div_sqrt_y_l1525_152584

theorem sqrt_x_div_sqrt_y (x y : ℝ) :
  (1/3)^2 + (1/4)^2 + (1/6)^2 = (37 * x / 85) * ((1/5)^2 + (1/7)^2 + (1/8)^2) * y →
  Real.sqrt x / Real.sqrt y = 1737 / 857 := by
  sorry

end sqrt_x_div_sqrt_y_l1525_152584


namespace summer_performs_1300_salutations_l1525_152587

/-- The number of sun salutations Summer performs throughout an entire year. -/
def summer_sun_salutations : ℕ :=
  let poses_per_day : ℕ := 5
  let weekdays_per_week : ℕ := 5
  let weeks_per_year : ℕ := 52
  poses_per_day * weekdays_per_week * weeks_per_year

/-- Theorem stating that Summer performs 1300 sun salutations throughout an entire year. -/
theorem summer_performs_1300_salutations : summer_sun_salutations = 1300 := by
  sorry

end summer_performs_1300_salutations_l1525_152587


namespace count_non_consecutive_digits_999999_l1525_152562

/-- Counts integers from 0 to n without consecutive identical digits -/
def countNonConsecutiveDigits (n : ℕ) : ℕ :=
  sorry

/-- The sum of geometric series 9^1 + 9^2 + ... + 9^6 -/
def geometricSum : ℕ :=
  sorry

theorem count_non_consecutive_digits_999999 :
  countNonConsecutiveDigits 999999 = 597880 := by
  sorry

end count_non_consecutive_digits_999999_l1525_152562


namespace unique_consecutive_odd_primes_l1525_152526

theorem unique_consecutive_odd_primes :
  ∀ p q r : ℕ,
  Prime p ∧ Prime q ∧ Prime r →
  p < q ∧ q < r →
  Odd p ∧ Odd q ∧ Odd r →
  q = p + 2 ∧ r = q + 2 →
  p = 3 ∧ q = 5 ∧ r = 7 := by
sorry

end unique_consecutive_odd_primes_l1525_152526


namespace vector_problem_l1525_152553

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = (k * b.1, k * b.2)

theorem vector_problem (a : ℝ × ℝ) :
  collinear a (1, -2) →
  a.1 * 1 + a.2 * (-2) = -10 →
  a = (-2, 4) ∧ Real.sqrt ((a.1 + 6)^2 + (a.2 - 7)^2) = 5 := by
  sorry

end vector_problem_l1525_152553


namespace characterize_f_l1525_152508

def is_valid_f (f : ℕ+ → ℝ) : Prop :=
  (∀ n : ℕ+, f (n + 1) ≥ f n) ∧
  (∀ m n : ℕ+, Nat.gcd m.val n.val = 1 → f (m * n) = f m * f n)

theorem characterize_f (f : ℕ+ → ℝ) (hf : is_valid_f f) :
  (∃ a : ℝ, a ≥ 0 ∧ ∀ n : ℕ+, f n = (n : ℝ) ^ a) ∨ (∀ n : ℕ+, f n = 0) :=
sorry

end characterize_f_l1525_152508


namespace percentage_of_total_l1525_152513

theorem percentage_of_total (N F M : ℝ) 
  (h1 : N = 0.05 * F) 
  (h2 : N = 0.20 * M) : 
  N / (F + M) = 0.04 := by
sorry

end percentage_of_total_l1525_152513


namespace vector_magnitude_range_l1525_152557

open Real
open InnerProductSpace

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem vector_magnitude_range (a b : V) 
  (h1 : ‖b‖ = 2) 
  (h2 : ‖a‖ = 2 * ‖b - a‖) : 
  4/3 ≤ ‖a‖ ∧ ‖a‖ ≤ 4 := by
sorry

end vector_magnitude_range_l1525_152557


namespace union_of_M_and_Q_l1525_152522

def M : Set ℕ := {0, 2, 4, 6}
def Q : Set ℕ := {0, 1, 3, 5}

theorem union_of_M_and_Q : M ∪ Q = {0, 1, 2, 3, 4, 5, 6} := by
  sorry

end union_of_M_and_Q_l1525_152522


namespace only_negative_three_less_than_negative_two_l1525_152533

theorem only_negative_three_less_than_negative_two :
  ((-3 : ℝ) < -2) ∧
  ((-1 : ℝ) > -2) ∧
  ((-Real.sqrt 2 : ℝ) > -2) ∧
  ((-Real.pi / 2 : ℝ) > -2) :=
by sorry

end only_negative_three_less_than_negative_two_l1525_152533


namespace min_value_sqrt_sum_squares_l1525_152588

theorem min_value_sqrt_sum_squares (a b m n : ℝ) 
  (h1 : a^2 + b^2 = 3) 
  (h2 : m*a + n*b = 3) : 
  Real.sqrt (m^2 + n^2) ≥ Real.sqrt 3 := by
  sorry

end min_value_sqrt_sum_squares_l1525_152588


namespace min_sum_mutually_exclusive_events_l1525_152598

theorem min_sum_mutually_exclusive_events (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (hA : ℝ) (hB : ℝ) (h_mutually_exclusive : hA + hB = 1) 
  (h_prob_A : hA = 1 / y) (h_prob_B : hB = 4 / x) : 
  x + y ≥ 9 ∧ ∃ x y, x + y = 9 := by
sorry

end min_sum_mutually_exclusive_events_l1525_152598


namespace vertex_of_quadratic_l1525_152534

/-- The quadratic function f(x) = x^2 - 2x + 3 -/
def f (x : ℝ) : ℝ := x^2 - 2*x + 3

/-- The x-coordinate of the vertex of f -/
def vertex_x : ℝ := 1

/-- The y-coordinate of the vertex of f -/
def vertex_y : ℝ := 2

/-- Theorem: The vertex of the quadratic function f(x) = x^2 - 2x + 3 is at (1, 2) -/
theorem vertex_of_quadratic :
  (∀ x : ℝ, f x ≥ f vertex_x) ∧ f vertex_x = vertex_y :=
sorry

end vertex_of_quadratic_l1525_152534


namespace coat_price_proof_l1525_152501

/-- Proves that the original price of a coat is $500 given the specified conditions -/
theorem coat_price_proof (P : ℝ) 
  (h1 : 0.70 * P = 350) : P = 500 := by
  sorry

end coat_price_proof_l1525_152501


namespace probability_sum_six_two_dice_l1525_152511

/-- A fair die has 6 sides -/
def fairDieSides : ℕ := 6

/-- The probability of an event is the number of favorable outcomes divided by the total number of possible outcomes -/
def probability (favorableOutcomes totalOutcomes : ℕ) : ℚ :=
  favorableOutcomes / totalOutcomes

/-- The total number of possible outcomes when throwing a die twice is the square of the number of sides -/
def totalOutcomes (sides : ℕ) : ℕ :=
  sides * sides

/-- The favorable outcomes are the pairs of numbers that sum to 6 -/
def favorableOutcomes : ℕ := 5

theorem probability_sum_six_two_dice :
  probability favorableOutcomes (totalOutcomes fairDieSides) = 5 / 36 := by
  sorry

end probability_sum_six_two_dice_l1525_152511


namespace subcommittee_formation_count_l1525_152505

theorem subcommittee_formation_count :
  let total_republicans : ℕ := 10
  let total_democrats : ℕ := 4
  let subcommittee_republicans : ℕ := 4
  let subcommittee_democrats : ℕ := 2
  let ways_to_choose_republicans : ℕ := (total_republicans.choose subcommittee_republicans)
  let ways_to_choose_democrats : ℕ := (total_democrats.choose subcommittee_democrats)
  ways_to_choose_republicans * ways_to_choose_democrats = 1260 :=
by sorry

end subcommittee_formation_count_l1525_152505


namespace probability_three_girls_l1525_152516

theorem probability_three_girls (total : ℕ) (girls : ℕ) (chosen : ℕ) : 
  total = 15 → girls = 9 → chosen = 3 →
  (Nat.choose girls chosen : ℚ) / (Nat.choose total chosen : ℚ) = 12 / 65 := by
  sorry

end probability_three_girls_l1525_152516


namespace area_of_overlapping_squares_l1525_152582

/-- Represents a square in a 2D plane -/
structure Square where
  sideLength : ℝ
  center : ℝ × ℝ

/-- Calculates the area of overlap between two squares -/
def overlapArea (s1 s2 : Square) : ℝ :=
  sorry

/-- Calculates the total area covered by two squares -/
def totalCoveredArea (s1 s2 : Square) : ℝ :=
  sorry

theorem area_of_overlapping_squares :
  let s1 : Square := { sideLength := 20, center := (0, 0) }
  let s2 : Square := { sideLength := 20, center := (10, 0) }
  totalCoveredArea s1 s2 = 600 := by
  sorry

end area_of_overlapping_squares_l1525_152582


namespace complex_multiplication_division_l1525_152554

theorem complex_multiplication_division (P F G : ℂ) :
  P = 3 + 4 * Complex.I ∧
  F = -Complex.I ∧
  G = 3 - 4 * Complex.I →
  (P * F * G) / (-3 * Complex.I) = 25 / 3 := by
  sorry

end complex_multiplication_division_l1525_152554


namespace quadratic_inequality_l1525_152560

theorem quadratic_inequality (x : ℝ) : 3 * x^2 - 8 * x - 3 > 0 ↔ x < -1/3 ∨ x > 3 := by
  sorry

end quadratic_inequality_l1525_152560


namespace school_pairing_fraction_l1525_152590

theorem school_pairing_fraction :
  ∀ (s n : ℕ), 
    s > 0 → n > 0 →
    (n : ℚ) / 4 = (s : ℚ) / 3 →
    ((s : ℚ) / 3 + (n : ℚ) / 4) / ((s : ℚ) + (n : ℚ)) = 2 / 7 := by
  sorry

end school_pairing_fraction_l1525_152590


namespace set_intersection_example_l1525_152535

theorem set_intersection_example : 
  ({3, 5, 6, 8} : Set ℕ) ∩ ({4, 5, 8} : Set ℕ) = {5, 8} := by
sorry

end set_intersection_example_l1525_152535


namespace tangent_line_equation_l1525_152586

/-- The function f(x) = x³ - x + 3 -/
def f (x : ℝ) : ℝ := x^3 - x + 3

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3 * x^2 - 1

theorem tangent_line_equation :
  let P : ℝ × ℝ := (1, 3)
  let m : ℝ := f' P.1
  let tangent_eq (x y : ℝ) : Prop := 2 * x - y + 1 = 0
  tangent_eq P.1 P.2 ∧ ∀ x y, tangent_eq x y ↔ y - P.2 = m * (x - P.1) :=
by sorry

end tangent_line_equation_l1525_152586


namespace inequality_proof_l1525_152510

theorem inequality_proof (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_prod : (a + b) * (b + c) * (c + a) = 1) : 
  a^2 / (1 + Real.sqrt (b * c)) + 
  b^2 / (1 + Real.sqrt (c * a)) + 
  c^2 / (1 + Real.sqrt (a * b)) ≥ 1/2 := by
sorry

end inequality_proof_l1525_152510


namespace perfect_square_binomial_l1525_152525

theorem perfect_square_binomial : ∃ a : ℝ, ∀ x : ℝ, x^2 - 20*x + 100 = (x - a)^2 := by
  sorry

end perfect_square_binomial_l1525_152525


namespace eighteen_wheeler_axles_l1525_152514

/-- Represents the toll calculation for a truck on a bridge -/
def toll_formula (num_axles : ℕ) : ℚ :=
  2.5 + 0.5 * (num_axles - 2)

theorem eighteen_wheeler_axles :
  ∃ (num_axles : ℕ),
    (18 = 2 + 4 * (num_axles - 1)) ∧
    (toll_formula num_axles = 4) ∧
    (num_axles = 5) := by
  sorry

end eighteen_wheeler_axles_l1525_152514


namespace sum_power_inequality_l1525_152539

theorem sum_power_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  a^a * b^b + a^b * b^a ≤ 1 := by
  sorry

end sum_power_inequality_l1525_152539


namespace base_conversion_problem_l1525_152577

theorem base_conversion_problem (a b c : ℕ) (h1 : a ≤ 6) (h2 : b ≤ 6) (h3 : c ≤ 6) 
  (h4 : a ≤ 8) (h5 : b ≤ 8) (h6 : c ≤ 8) :
  (49 * a + 7 * b + c = 81 * c + 9 * b + a) → (49 * a + 7 * b + c = 248) :=
by sorry

end base_conversion_problem_l1525_152577


namespace decoration_nail_count_l1525_152593

theorem decoration_nail_count :
  ∀ D : ℕ,
  (D : ℚ) * (21/80) = 20 →
  ⌊(D : ℚ) * (5/8)⌋ = 47 :=
by
  sorry

end decoration_nail_count_l1525_152593


namespace min_value_of_sum_of_squares_l1525_152572

theorem min_value_of_sum_of_squares (p q r s t u v w : ℝ) 
  (h1 : p * q * r * s = 8) 
  (h2 : t * u * v * w = 16) : 
  (p * t)^2 + (q * u)^2 + (r * v)^2 + (s * w)^2 ≥ 64 := by
  sorry

end min_value_of_sum_of_squares_l1525_152572


namespace gcd_of_180_270_450_l1525_152558

theorem gcd_of_180_270_450 : Nat.gcd 180 (Nat.gcd 270 450) = 90 := by
  sorry

end gcd_of_180_270_450_l1525_152558


namespace simple_interest_calculation_l1525_152574

/-- Simple interest calculation -/
theorem simple_interest_calculation
  (principal : ℝ)
  (rate : ℝ)
  (time : ℝ)
  (h1 : principal = 10000)
  (h2 : rate = 0.09)
  (h3 : time = 1) :
  principal * rate * time = 900 :=
by sorry

end simple_interest_calculation_l1525_152574


namespace sequence_formula_correct_l1525_152542

def sequence_term (n : ℕ) : ℚ := (-1)^n * (n^2 : ℚ) / (2*n - 1)

theorem sequence_formula_correct : 
  (sequence_term 1 = -1) ∧ 
  (sequence_term 2 = 4/3) ∧ 
  (sequence_term 3 = -9/5) ∧ 
  (sequence_term 4 = 16/7) := by
  sorry

end sequence_formula_correct_l1525_152542


namespace ratio_proof_l1525_152585

theorem ratio_proof (a b : ℕ+) (h1 : a.val * 4 = b.val * 3) (h2 : Nat.gcd a.val b.val = 5) (h3 : Nat.lcm a.val b.val = 60) : a.val * 4 = b.val * 3 := by
  sorry

end ratio_proof_l1525_152585


namespace bridge_length_l1525_152570

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 120 →
  train_speed_kmh = 45 →
  crossing_time = 30 →
  (train_speed_kmh * 1000 / 3600 * crossing_time) - train_length = 255 := by
  sorry

end bridge_length_l1525_152570


namespace projectile_motion_time_l1525_152507

/-- The equation of motion for a projectile launched from the ground -/
def equation_of_motion (v : ℝ) (t : ℝ) : ℝ := -16 * t^2 + v * t

/-- The initial velocity of the projectile in feet per second -/
def initial_velocity : ℝ := 80

/-- The height reached by the projectile in feet -/
def height_reached : ℝ := 100

/-- The time taken to reach the specified height -/
def time_to_reach_height : ℝ := 2.5

theorem projectile_motion_time :
  equation_of_motion initial_velocity time_to_reach_height = height_reached :=
by sorry

end projectile_motion_time_l1525_152507


namespace man_to_son_age_ratio_l1525_152592

def son_age : ℕ := 20
def age_difference : ℕ := 22

def man_age : ℕ := son_age + age_difference

def son_age_in_two_years : ℕ := son_age + 2
def man_age_in_two_years : ℕ := man_age + 2

theorem man_to_son_age_ratio :
  man_age_in_two_years / son_age_in_two_years = 2 ∧
  man_age_in_two_years % son_age_in_two_years = 0 := by
  sorry

#eval man_age_in_two_years / son_age_in_two_years

end man_to_son_age_ratio_l1525_152592


namespace arithmetic_geometric_sequence_minimum_l1525_152580

theorem arithmetic_geometric_sequence_minimum (n : ℕ) (d : ℝ) (a : ℕ → ℝ) (S : ℕ → ℝ) :
  d > 0 →
  (∀ k, a k = a 1 + (k - 1) * d) →
  a 1 = 5 →
  (a 5 - 1)^2 = a 2 * a 10 →
  (∀ k, S k = (k / 2) * (2 * a 1 + (k - 1) * d)) →
  (∀ k, (2 * S k + k + 32) / (a k + 1) ≥ 20 / 3) ∧
  (∃ k, (2 * S k + k + 32) / (a k + 1) = 20 / 3) :=
by sorry

end arithmetic_geometric_sequence_minimum_l1525_152580


namespace distinct_roots_sum_abs_gt_six_l1525_152596

theorem distinct_roots_sum_abs_gt_six (p : ℝ) (r₁ r₂ : ℝ) : 
  r₁ ≠ r₂ → 
  r₁^2 + p*r₁ + 9 = 0 → 
  r₂^2 + p*r₂ + 9 = 0 → 
  |r₁ + r₂| > 6 := by
sorry

end distinct_roots_sum_abs_gt_six_l1525_152596


namespace show_episodes_per_week_l1525_152583

/-- Calculates the number of episodes shown per week given the episode length,
    filming time multiplier, and total filming time for a certain number of weeks. -/
def episodes_per_week (episode_length : ℕ) (filming_multiplier : ℚ) (total_filming_time : ℕ) (num_weeks : ℕ) : ℚ :=
  let filming_time_per_episode : ℚ := episode_length * filming_multiplier
  let total_minutes : ℕ := total_filming_time * 60
  let total_episodes : ℚ := total_minutes / filming_time_per_episode
  total_episodes / num_weeks

/-- Proves that the number of episodes shown each week is 5 under the given conditions. -/
theorem show_episodes_per_week :
  episodes_per_week 20 (3/2) 10 4 = 5 := by
  sorry

end show_episodes_per_week_l1525_152583


namespace remaining_numbers_l1525_152528

theorem remaining_numbers (total : ℕ) (total_avg : ℚ) (subset : ℕ) (subset_avg : ℚ) (remaining_avg : ℚ) :
  total = 9 →
  total_avg = 18 →
  subset = 4 →
  subset_avg = 8 →
  remaining_avg = 26 →
  total - subset = (total * total_avg - subset * subset_avg) / remaining_avg :=
by
  sorry

#eval (9 : ℕ) - 4  -- Expected output: 5

end remaining_numbers_l1525_152528


namespace monkey_climb_theorem_l1525_152543

/-- Calculates the time taken for a monkey to climb a tree given the tree height and climbing behavior -/
def monkey_climb_time (tree_height : ℕ) (hop_up : ℕ) (slip_back : ℕ) : ℕ := sorry

/-- Theorem stating that for the given conditions, the monkey takes 15 hours to reach the top -/
theorem monkey_climb_theorem :
  monkey_climb_time 51 7 4 = 15 := by sorry

end monkey_climb_theorem_l1525_152543


namespace future_ratio_years_l1525_152538

/-- Represents the ages and time in the problem -/
structure AgeData where
  vimal_initial : ℕ  -- Vimal's age 6 years ago
  saroj_initial : ℕ  -- Saroj's age 6 years ago
  years_passed : ℕ   -- Years passed since the initial ratio

/-- The conditions of the problem -/
def problem_conditions (data : AgeData) : Prop :=
  data.vimal_initial * 5 = data.saroj_initial * 6 ∧  -- Initial ratio 6:5
  data.saroj_initial + 6 = 16 ∧                      -- Saroj's current age is 16
  (data.vimal_initial + 6 + 4) * 10 = (data.saroj_initial + 6 + 4) * 11  -- Future ratio 11:10

/-- The theorem to be proved -/
theorem future_ratio_years (data : AgeData) :
  problem_conditions data → data.years_passed = 4 := by
  sorry


end future_ratio_years_l1525_152538


namespace second_reader_average_pages_per_day_l1525_152536

/-- Calculates the average pages read per day by the second-place reader -/
def average_pages_per_day (break_days : ℕ) (deshaun_books : ℕ) (avg_pages_per_book : ℕ) (second_reader_percentage : ℚ) : ℚ :=
  let deshaun_total_pages := deshaun_books * avg_pages_per_book
  let second_reader_pages := second_reader_percentage * deshaun_total_pages
  second_reader_pages / break_days

/-- Theorem stating that the second-place reader averaged 180 pages per day -/
theorem second_reader_average_pages_per_day :
  average_pages_per_day 80 60 320 (3/4) = 180 := by
  sorry

end second_reader_average_pages_per_day_l1525_152536


namespace only_zero_solution_for_diophantine_equation_l1525_152503

theorem only_zero_solution_for_diophantine_equation :
  ∀ x y : ℤ, x^4 + y^4 = 3*x^3*y → x = 0 ∧ y = 0 := by
  sorry

end only_zero_solution_for_diophantine_equation_l1525_152503


namespace polygon_sides_count_l1525_152506

theorem polygon_sides_count (n : ℕ) : n > 2 → (n - 2) * 180 = 3 * 360 → n = 8 := by
  sorry

end polygon_sides_count_l1525_152506


namespace convex_separation_equivalence_l1525_152569

-- Define the type for a compact convex set in ℝ²
def CompactConvexSet : Type := Set (Real × Real)

-- Define the property of a set being compact and convex
def is_compact_convex (S : CompactConvexSet) : Prop := sorry

-- Define the convex hull operation
def conv_hull (S T : CompactConvexSet) : CompactConvexSet := sorry

-- Define the property of a line separating two sets
def separates (L : Set (Real × Real)) (S T : CompactConvexSet) : Prop := sorry

-- Define the property of a line intersecting a set
def intersects (L : Set (Real × Real)) (S : CompactConvexSet) : Prop := sorry

-- The main theorem
theorem convex_separation_equivalence 
  (A B C : CompactConvexSet) 
  (hA : is_compact_convex A) 
  (hB : is_compact_convex B) 
  (hC : is_compact_convex C) : 
  (∀ L : Set (Real × Real), ¬(intersects L A ∧ intersects L B ∧ intersects L C)) ↔ 
  (∃ LA LB LC : Set (Real × Real), 
    separates LA A (conv_hull B C) ∧ 
    separates LB B (conv_hull A C) ∧ 
    separates LC C (conv_hull A B)) := by
  sorry

end convex_separation_equivalence_l1525_152569


namespace system_solution_unique_l1525_152555

theorem system_solution_unique (x y : ℝ) : 
  2 * x - 5 * y = 2 ∧ x + 3 * y = 12 ↔ x = 6 ∧ y = 2 := by
  sorry

end system_solution_unique_l1525_152555


namespace sqrt_equation_solution_l1525_152568

theorem sqrt_equation_solution (s : ℝ) : 
  Real.sqrt (3 * Real.sqrt (s - 3)) = (8 - s) ^ (1/4) → s = 3.5 := by
  sorry

end sqrt_equation_solution_l1525_152568


namespace complex_equation_solution_l1525_152544

theorem complex_equation_solution (z : ℂ) : (1 - Complex.I) * z = -1 - Complex.I → z = -Complex.I := by
  sorry

end complex_equation_solution_l1525_152544


namespace walking_time_difference_l1525_152561

/-- Proof of the walking time difference between Cara and Don --/
theorem walking_time_difference 
  (total_distance : ℝ) 
  (cara_speed : ℝ) 
  (don_speed : ℝ) 
  (cara_distance : ℝ) 
  (h1 : total_distance = 45) 
  (h2 : cara_speed = 6) 
  (h3 : don_speed = 5) 
  (h4 : cara_distance = 30) : 
  (cara_distance / cara_speed) - ((total_distance - cara_distance) / don_speed) = 2 := by
  sorry

end walking_time_difference_l1525_152561


namespace small_cakes_per_hour_l1525_152599

-- Define the variables
def helpers : ℕ := 10
def hours : ℕ := 3
def large_cakes_needed : ℕ := 20
def small_cakes_needed : ℕ := 700
def large_cakes_per_hour : ℕ := 2

-- Define the theorem
theorem small_cakes_per_hour :
  ∃ (s : ℕ), 
    s * helpers * (hours - (large_cakes_needed / large_cakes_per_hour)) = small_cakes_needed ∧
    s = 35 := by
  sorry

end small_cakes_per_hour_l1525_152599


namespace toy_car_spending_l1525_152576

theorem toy_car_spending
  (A B C D E F G H : ℝ)
  (last_month : ℝ := A + B + C + D + E)
  (this_month_new : ℝ := F + G + H)
  (discount : ℝ := 0.2)
  (total_before_discount : ℝ := 2 * last_month + this_month_new)
  (total_after_discount : ℝ := (1 - discount) * total_before_discount) :
  total_after_discount = 1.6 * A + 1.6 * B + 1.6 * C + 1.6 * D + 1.6 * E + 0.8 * F + 0.8 * G + 0.8 * H :=
by sorry

end toy_car_spending_l1525_152576


namespace product_equals_64_l1525_152515

theorem product_equals_64 : 
  (1/2 : ℚ) * 4 * (1/8) * 16 * (1/32) * 64 * (1/128) * 256 * (1/512) * 1024 * (1/2048) * 4096 = 64 := by
  sorry

end product_equals_64_l1525_152515


namespace probability_of_six_l1525_152581

/-- A fair die with 6 faces -/
structure FairDie :=
  (faces : Nat)
  (is_fair : Bool)
  (h_faces : faces = 6)
  (h_fair : is_fair = true)

/-- The probability of getting a specific face on a fair die -/
def probability_of_face (d : FairDie) : ℚ :=
  1 / d.faces

/-- Theorem: The probability of getting any specific face on a fair 6-faced die is 1/6 -/
theorem probability_of_six (d : FairDie) : probability_of_face d = 1 / 6 := by
  sorry

#eval (1 : ℚ) / 6  -- To show that 1/6 ≈ 0.17

end probability_of_six_l1525_152581


namespace inequality_proof_l1525_152518

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 8) :
  (x + 1/y)^2 + (y + 1/x)^2 ≥ 289/8 := by
  sorry

end inequality_proof_l1525_152518


namespace day_of_week_previous_year_l1525_152504

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a year -/
structure Year where
  value : ℕ
  isLeap : Bool

/-- Returns the day of the week given a day number and a year -/
def dayOfWeek (day : ℕ) (year : Year) : DayOfWeek :=
  sorry

/-- Advances the day of the week by a given number of days -/
def advanceDays (start : DayOfWeek) (days : ℕ) : DayOfWeek :=
  sorry

theorem day_of_week_previous_year 
  (N : Year)
  (h1 : N.isLeap = true)
  (h2 : dayOfWeek 250 N = DayOfWeek.Wednesday)
  (h3 : dayOfWeek 150 ⟨N.value + 1, false⟩ = DayOfWeek.Wednesday) :
  dayOfWeek 100 ⟨N.value - 1, false⟩ = DayOfWeek.Saturday :=
by sorry

end day_of_week_previous_year_l1525_152504


namespace no_difference_of_primes_in_S_l1525_152550

/-- The set of numbers we're considering -/
def S : Set ℕ := {n : ℕ | ∃ k : ℕ, n = 10 * k + 7}

/-- A function that checks if a natural number is prime -/
def is_prime (n : ℕ) : Prop := Nat.Prime n

/-- A function that checks if a number can be expressed as the difference of two primes -/
def is_difference_of_primes (n : ℕ) : Prop :=
  ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p - q = n

/-- The main theorem: no number in S can be expressed as the difference of two primes -/
theorem no_difference_of_primes_in_S : ∀ n ∈ S, ¬(is_difference_of_primes n) := by
  sorry

end no_difference_of_primes_in_S_l1525_152550


namespace max_side_length_of_triangle_l1525_152521

theorem max_side_length_of_triangle (a b c : ℕ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →  -- three different side lengths
  a + b + c = 30 →        -- perimeter is 30
  a < 15 ∧ b < 15 ∧ c < 15 →  -- each side is less than 15
  a ≤ 14 ∧ b ≤ 14 ∧ c ≤ 14 →  -- each side is at most 14
  ∃ (x y z : ℕ), x + y + z = 30 ∧ x = 14 ∧ y < x ∧ z < x ∧ y ≠ z →  -- there exists a triangle with max side 14
  (∀ (s : ℕ), s ≤ a ∨ s ≤ b ∨ s ≤ c) →  -- s is not greater than all sides
  14 = max a (max b c)  -- 14 is the maximum side length
  := by sorry

end max_side_length_of_triangle_l1525_152521


namespace right_angled_triangle_x_values_l1525_152565

def triangle_ABC (x : ℝ) : Prop :=
  ∃ (A B C : ℝ × ℝ),
    let AB := (2, -1)
    let AC := (x, 3)
    let BC := (x - 2, 4)
    (AB.1 * AC.1 + AB.2 * AC.2 = 0) ∨ 
    (AB.1 * BC.1 + AB.2 * BC.2 = 0) ∨
    (AC.1 * BC.1 + AC.2 * BC.2 = 0)

theorem right_angled_triangle_x_values :
  ∀ x : ℝ, triangle_ABC x → x = 3/2 ∨ x = 4 :=
sorry

end right_angled_triangle_x_values_l1525_152565


namespace mn_length_is_two_l1525_152537

-- Define the line l
def line_l (k : ℝ) (x : ℝ) : ℝ := k * x + 1

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + (y - 3)^2 = 1

-- Define the intersection points M and N
def intersection_points (k : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
    y₁ = line_l k x₁ ∧ y₂ = line_l k x₂ ∧
    x₁ ≠ x₂

-- Define the dot product condition
def dot_product_condition (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ * x₂ + y₁ * y₂ = 12

-- Main theorem
theorem mn_length_is_two (k : ℝ) :
  intersection_points k →
  (∃ x₁ y₁ x₂ y₂ : ℝ, circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
                      y₁ = line_l k x₁ ∧ y₂ = line_l k x₂ ∧
                      dot_product_condition x₁ y₁ x₂ y₂) →
  ∃ x₁ y₁ x₂ y₂ : ℝ, circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
                     y₁ = line_l k x₁ ∧ y₂ = line_l k x₂ ∧
                     (x₁ - x₂)^2 + (y₁ - y₂)^2 = 4 :=
by sorry

end mn_length_is_two_l1525_152537


namespace surface_area_of_cube_with_holes_l1525_152597

/-- Represents a cube with square holes cut through each face -/
structure CubeWithHoles where
  edge_length : ℝ
  hole_side_length : ℝ

/-- Calculates the surface area of a cube with square holes cut through each face -/
def surface_area (cube : CubeWithHoles) : ℝ :=
  let original_surface_area := 6 * cube.edge_length^2
  let hole_area := 6 * cube.hole_side_length^2
  let inner_surface_area := 6 * 4 * cube.hole_side_length^2
  original_surface_area - hole_area + inner_surface_area

/-- Theorem stating that the surface area of the specified cube with holes is 168 square meters -/
theorem surface_area_of_cube_with_holes :
  let cube := CubeWithHoles.mk 4 2
  surface_area cube = 168 := by
  sorry


end surface_area_of_cube_with_holes_l1525_152597


namespace simplify_sqrt_difference_l1525_152541

theorem simplify_sqrt_difference : Real.sqrt 8 - Real.sqrt 2 = Real.sqrt 2 := by
  sorry

end simplify_sqrt_difference_l1525_152541


namespace total_homework_pages_l1525_152500

def math_homework_pages : ℕ := 10
def reading_homework_difference : ℕ := 3

def total_pages : ℕ := math_homework_pages + (math_homework_pages + reading_homework_difference)

theorem total_homework_pages : total_pages = 23 := by
  sorry

end total_homework_pages_l1525_152500


namespace equilateral_triangle_complex_plane_l1525_152591

theorem equilateral_triangle_complex_plane (z : ℂ) (μ : ℝ) : 
  Complex.abs z = 3 →
  μ > 2 →
  (Complex.abs (z^3 - z) = Complex.abs (μ • z - z) ∧
   Complex.abs (z^3 - μ • z) = Complex.abs (μ • z - z) ∧
   Complex.abs (z^3 - μ • z) = Complex.abs (z^3 - z)) →
  μ = 1 + Real.sqrt 82 := by
sorry

end equilateral_triangle_complex_plane_l1525_152591


namespace joan_balloons_l1525_152520

theorem joan_balloons (initial : ℕ) : initial + 2 = 10 → initial = 8 := by
  sorry

end joan_balloons_l1525_152520


namespace derivative_f_at_zero_l1525_152540

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then x + Real.arcsin (x^2 * Real.sin (6 / x))
  else 0

-- State the theorem
theorem derivative_f_at_zero :
  deriv f 0 = 1 := by sorry

end derivative_f_at_zero_l1525_152540


namespace stationery_cost_l1525_152529

def total_spent : ℝ := 32
def backpack_cost : ℝ := 15
def notebook_cost : ℝ := 3
def notebook_count : ℕ := 5

theorem stationery_cost :
  total_spent - (backpack_cost + notebook_cost * notebook_count) = 2 := by
  sorry

end stationery_cost_l1525_152529


namespace problem_equivalence_l1525_152579

theorem problem_equivalence :
  (3 * (Real.sqrt 3 + Real.sqrt 2) - 2 * (Real.sqrt 3 - Real.sqrt 2) = Real.sqrt 3 + 5 * Real.sqrt 2) ∧
  (|Real.sqrt 3 - Real.sqrt 2| + |Real.sqrt 3 - 2| + Real.sqrt ((-2)^2) = 4 - Real.sqrt 2) := by
  sorry

end problem_equivalence_l1525_152579


namespace sequence_relation_l1525_152563

def a : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 4 * a (n + 1) - a n

def b : ℕ → ℤ
  | 0 => 1
  | 1 => 2
  | (n + 2) => 4 * b (n + 1) - b n

theorem sequence_relation (n : ℕ) : (b n)^2 = 3 * (a n)^2 + 1 := by
  sorry

end sequence_relation_l1525_152563


namespace line_intersection_x_axis_l1525_152556

/-- A point in the 2D plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A line defined by two points -/
structure Line :=
  (p1 : Point)
  (p2 : Point)

/-- The x-axis -/
def x_axis : Line :=
  { p1 := ⟨0, 0⟩, p2 := ⟨1, 0⟩ }

/-- Function to determine if a point lies on a given line -/
def point_on_line (p : Point) (l : Line) : Prop :=
  (p.y - l.p1.y) * (l.p2.x - l.p1.x) = (p.x - l.p1.x) * (l.p2.y - l.p1.y)

/-- Function to determine if a point lies on the x-axis -/
def point_on_x_axis (p : Point) : Prop :=
  p.y = 0

/-- The main theorem -/
theorem line_intersection_x_axis :
  let l : Line := { p1 := ⟨7, 3⟩, p2 := ⟨3, 7⟩ }
  let intersection_point : Point := ⟨10, 0⟩
  point_on_line intersection_point l ∧ point_on_x_axis intersection_point :=
by
  sorry

end line_intersection_x_axis_l1525_152556


namespace parabola_equation_and_max_area_l1525_152573

-- Define the parabola
structure Parabola where
  p : ℝ
  focus : ℝ × ℝ
  equation : ℝ → ℝ → Prop

-- Define a point on the parabola
structure PointOnParabola (c : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : c.equation x y

-- Define the vector from focus to a point
def vector_from_focus (c : Parabola) (p : PointOnParabola c) : ℝ × ℝ :=
  (p.x - c.focus.1, p.y - c.focus.2)

theorem parabola_equation_and_max_area 
  (c : Parabola)
  (h_focus : c.focus = (0, 1))
  (h_equation : ∀ x y, c.equation x y ↔ x^2 = 2 * c.p * y)
  (A B C : PointOnParabola c)
  (h_vector_sum : vector_from_focus c A + vector_from_focus c B + vector_from_focus c C = (0, 0)) :
  (∀ x y, c.equation x y ↔ x^2 = 4 * y) ∧
  (∃ (max_area : ℝ), max_area = (3 * Real.sqrt 6) / 2 ∧
    ∀ (area : ℝ), area = abs (A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y)) / 2 →
      area ≤ max_area) :=
sorry

end parabola_equation_and_max_area_l1525_152573


namespace largest_binomial_coefficient_in_expansion_fourth_term_has_largest_coefficient_l1525_152552

theorem largest_binomial_coefficient_in_expansion :
  ∀ k : ℕ, k ≤ 6 → Nat.choose 6 3 ≥ Nat.choose 6 k :=
by sorry

theorem fourth_term_has_largest_coefficient :
  ∃ k : ℕ, k = 4 ∧
  ∀ j : ℕ, j ≤ 6 → Nat.choose 6 (k - 1) ≥ Nat.choose 6 j :=
by sorry

end largest_binomial_coefficient_in_expansion_fourth_term_has_largest_coefficient_l1525_152552


namespace seed_placement_count_l1525_152523

/-- The number of ways to select k items from n distinct items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of ways to arrange k items from n distinct items -/
def arrange (n k : ℕ) : ℕ := sorry

/-- The total number of seed placement methods -/
def totalPlacements : ℕ := sorry

theorem seed_placement_count :
  let totalSeeds : ℕ := 10
  let bottleCount : ℕ := 6
  let seedsNotForBottleOne : ℕ := 2
  totalPlacements = choose (totalSeeds - seedsNotForBottleOne) 1 * arrange (totalSeeds - 1) (bottleCount - 1) := by
  sorry

end seed_placement_count_l1525_152523


namespace harmonic_subsequence_existence_l1525_152566

/-- The harmonic sequence -/
def harmonic_seq : ℕ → ℚ
  | n => 1 / n

/-- A subsequence of the harmonic sequence -/
def subseq (f : ℕ → ℕ) : ℕ → ℚ := λ n => harmonic_seq (f n)

/-- The property that each term, starting from the third, is the difference of the two preceding terms -/
def has_difference_property (s : ℕ → ℚ) : Prop :=
  ∀ n ≥ 3, s n = s (n - 1) - s (n - 2)

theorem harmonic_subsequence_existence :
  ∃ f : ℕ → ℕ, (∀ n m, n < m → f n < f m) ∧ 
              has_difference_property (subseq f) ∧
              (∃ N : ℕ, N ≥ 100) :=
sorry

end harmonic_subsequence_existence_l1525_152566


namespace equation_solution_l1525_152594

theorem equation_solution : 
  ∃ (x₁ x₂ : ℝ), x₁ = -3/2 ∧ x₂ = 7/2 ∧ 
  (∀ x : ℝ, 4 * (1 - x)^2 = 25 ↔ (x = x₁ ∨ x = x₂)) :=
by sorry

end equation_solution_l1525_152594


namespace boat_speed_in_still_water_l1525_152547

/-- Proves that the speed of a boat in still water is 30 km/hr given specific conditions -/
theorem boat_speed_in_still_water 
  (current_speed : ℝ) 
  (downstream_distance : ℝ) 
  (downstream_time : ℝ) 
  (h1 : current_speed = 7)
  (h2 : downstream_distance = 22.2)
  (h3 : downstream_time = 0.6) :
  ∃ (boat_speed : ℝ), 
    boat_speed = 30 ∧ 
    downstream_distance = (boat_speed + current_speed) * downstream_time :=
by sorry

end boat_speed_in_still_water_l1525_152547


namespace basketball_team_squads_l1525_152549

/-- The number of ways to choose k items from n items without replacement and where order doesn't matter. -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The number of different team squads that can be formed from a given number of players,
    selecting a captain and a specified number of additional players. -/
def teamSquads (totalPlayers captains additionalPlayers : ℕ) : ℕ :=
  totalPlayers * binomial (totalPlayers - 1) additionalPlayers

theorem basketball_team_squads :
  teamSquads 12 1 5 = 5544 := by sorry

end basketball_team_squads_l1525_152549


namespace paper_strip_dimensions_l1525_152589

theorem paper_strip_dimensions (a b c : ℕ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : c > 0) 
  (h4 : a * b + a * c + a * (b - a) + a^2 + a * (c - a) = 43) : 
  (a = 1 ∧ b + c = 22) ∨ (a = 22 ∧ b + c = 1) :=
sorry

end paper_strip_dimensions_l1525_152589


namespace mangoes_purchased_is_nine_l1525_152502

/-- The amount of mangoes purchased, given the conditions of the problem -/
def mangoes_purchased (apple_kg : ℕ) (apple_rate : ℕ) (mango_rate : ℕ) (total_paid : ℕ) : ℕ :=
  (total_paid - apple_kg * apple_rate) / mango_rate

/-- Theorem stating that the amount of mangoes purchased is 9 kg -/
theorem mangoes_purchased_is_nine :
  mangoes_purchased 8 70 45 965 = 9 := by sorry

end mangoes_purchased_is_nine_l1525_152502


namespace unique_solution_quadratic_l1525_152527

theorem unique_solution_quadratic (m : ℝ) : 
  (∃! x : ℝ, (x + 6) * (x + 2) = m + 3 * x) ↔ m = 23 / 4 := by
  sorry

end unique_solution_quadratic_l1525_152527


namespace max_at_2_implies_c_6_l1525_152512

/-- The function f(x) = x(x-c)² has a maximum value at x=2 -/
def has_max_at_2 (c : ℝ) : Prop :=
  let f := fun x => x * (x - c)^2
  ∀ x, f x ≤ f 2

/-- Theorem: If f(x) = x(x-c)² has a maximum value at x=2, then c = 6 -/
theorem max_at_2_implies_c_6 : 
  ∀ c : ℝ, has_max_at_2 c → c = 6 := by
  sorry

end max_at_2_implies_c_6_l1525_152512


namespace animal_shelter_problem_l1525_152571

/-- Represents the animal shelter problem --/
theorem animal_shelter_problem
  (initial_dogs : ℕ) (initial_cats : ℕ) (new_pets : ℕ) (total_after_month : ℕ)
  (dog_adoption_rate : ℚ) (cat_adoption_rate : ℚ) (lizard_adoption_rate : ℚ)
  (h1 : initial_dogs = 30)
  (h2 : initial_cats = 28)
  (h3 : new_pets = 13)
  (h4 : total_after_month = 65)
  (h5 : dog_adoption_rate = 1/2)
  (h6 : cat_adoption_rate = 1/4)
  (h7 : lizard_adoption_rate = 1/5) :
  ∃ (initial_lizards : ℕ),
    initial_lizards = 20 ∧
    (↑initial_dogs * (1 - dog_adoption_rate) +
     ↑initial_cats * (1 - cat_adoption_rate) +
     ↑initial_lizards * (1 - lizard_adoption_rate) +
     ↑new_pets : ℚ) = total_after_month :=
by sorry

end animal_shelter_problem_l1525_152571


namespace sum_of_squares_divisible_by_five_l1525_152575

theorem sum_of_squares_divisible_by_five (x y : ℤ) :
  (∃ n : ℤ, (x^2 + y^2) = 5*n) →
  ∃ a b : ℤ, (x^2 + y^2) / 5 = a^2 + b^2 := by
sorry

end sum_of_squares_divisible_by_five_l1525_152575


namespace mildred_total_oranges_l1525_152595

/-- The number of oranges Mildred initially collected -/
def initial_oranges : ℕ := 77

/-- The number of oranges Mildred's father gave her -/
def additional_oranges : ℕ := 2

/-- Theorem: Mildred's total number of oranges is 79 -/
theorem mildred_total_oranges : 
  initial_oranges + additional_oranges = 79 := by
  sorry

end mildred_total_oranges_l1525_152595


namespace necessary_not_sufficient_condition_l1525_152551

theorem necessary_not_sufficient_condition (a b : ℝ) :
  (∀ a b : ℝ, a < b → a < b + 1) ∧
  (∃ a b : ℝ, a < b + 1 ∧ ¬(a < b)) := by
  sorry

end necessary_not_sufficient_condition_l1525_152551

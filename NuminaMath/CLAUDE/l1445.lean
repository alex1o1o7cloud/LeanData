import Mathlib

namespace NUMINAMATH_CALUDE_gold_bars_problem_l1445_144541

theorem gold_bars_problem (initial : ℕ) : 
  (initial : ℚ) * (1 - 0.1) * 0.5 = 27 → initial = 60 := by
  sorry

end NUMINAMATH_CALUDE_gold_bars_problem_l1445_144541


namespace NUMINAMATH_CALUDE_white_coincide_pairs_l1445_144588

-- Define the structure of our figure
structure Figure where
  red_triangles : ℕ
  blue_triangles : ℕ
  white_triangles : ℕ
  red_coincide : ℕ
  blue_coincide : ℕ
  red_white_pairs : ℕ

-- Define our specific figure
def our_figure : Figure :=
  { red_triangles := 4
  , blue_triangles := 6
  , white_triangles := 10
  , red_coincide := 3
  , blue_coincide := 4
  , red_white_pairs := 3 }

-- Theorem statement
theorem white_coincide_pairs (f : Figure) (h : f = our_figure) : 
  ∃ (white_coincide : ℕ), white_coincide = 3 := by
  sorry

end NUMINAMATH_CALUDE_white_coincide_pairs_l1445_144588


namespace NUMINAMATH_CALUDE_expression_equality_l1445_144579

theorem expression_equality : 
  |Real.sqrt 3 - 3| - Real.sqrt 16 + Real.cos (30 * π / 180) + (1/3)^0 = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l1445_144579


namespace NUMINAMATH_CALUDE_sqrt_square_not_always_equal_to_a_l1445_144562

theorem sqrt_square_not_always_equal_to_a : ¬ ∀ a : ℝ, Real.sqrt (a^2) = a := by
  sorry

end NUMINAMATH_CALUDE_sqrt_square_not_always_equal_to_a_l1445_144562


namespace NUMINAMATH_CALUDE_smallest_divisible_by_2000_l1445_144586

def sequence_a (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → (n - 1 : ℤ) * a (n + 1) = (n + 1 : ℤ) * a n - 2 * (n - 1 : ℤ)

theorem smallest_divisible_by_2000 (a : ℕ → ℤ) (h : sequence_a a) (h2000 : 2000 ∣ a 1999) :
  (∃ n : ℕ, n ≥ 2 ∧ 2000 ∣ a n) ∧ (∀ m : ℕ, m ≥ 2 ∧ m < 249 → ¬(2000 ∣ a m)) ∧ 2000 ∣ a 249 :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_2000_l1445_144586


namespace NUMINAMATH_CALUDE_factorization_of_4x_cubed_minus_16x_l1445_144518

theorem factorization_of_4x_cubed_minus_16x (x : ℝ) :
  4 * x^3 - 16 * x = 4 * x * (x + 2) * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_4x_cubed_minus_16x_l1445_144518


namespace NUMINAMATH_CALUDE_square_sum_divisible_by_product_l1445_144559

def is_valid_triple (k : ℤ) (x y z : ℤ) : Prop :=
  x > 0 ∧ y > 0 ∧ z > 0 ∧ x ≤ 1000 ∧ y ≤ 1000 ∧ z ≤ 1000 ∧
  x^2 + y^2 + z^2 = k * x * y * z

def valid_triples_for_k (k : ℤ) : List (ℤ × ℤ × ℤ) :=
  if k = 3 then
    [(1, 1, 1), (1, 1, 2), (1, 2, 5), (1, 5, 13), (2, 5, 29), (5, 29, 169)]
  else if k = 1 then
    [(3, 3, 3), (3, 3, 6), (3, 6, 15), (6, 15, 39), (6, 15, 87)]
  else
    []

theorem square_sum_divisible_by_product :
  ∀ k x y z : ℤ,
    is_valid_triple k x y z ↔
      (k = 1 ∨ k = 3) ∧
      (x, y, z) ∈ valid_triples_for_k k :=
sorry

end NUMINAMATH_CALUDE_square_sum_divisible_by_product_l1445_144559


namespace NUMINAMATH_CALUDE_factorization_proof_l1445_144544

theorem factorization_proof (x y : ℝ) : 9*y - 25*x^2*y = y*(3+5*x)*(3-5*x) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l1445_144544


namespace NUMINAMATH_CALUDE_tetrahedron_volume_is_4_sqrt_6_div_3_l1445_144534

/-- Tetrahedron with specific face angles and areas -/
structure Tetrahedron where
  /-- Face angle APB -/
  angle_APB : ℝ
  /-- Face angle BPC -/
  angle_BPC : ℝ
  /-- Face angle CPA -/
  angle_CPA : ℝ
  /-- Area of face PAB -/
  area_PAB : ℝ
  /-- Area of face PBC -/
  area_PBC : ℝ
  /-- Area of face PCA -/
  area_PCA : ℝ
  /-- All face angles are 60° -/
  angle_constraint : angle_APB = 60 ∧ angle_BPC = 60 ∧ angle_CPA = 60
  /-- Areas of faces are √3/2, 2, and 1 -/
  area_constraint : area_PAB = Real.sqrt 3 / 2 ∧ area_PBC = 2 ∧ area_PCA = 1

/-- Volume of a tetrahedron -/
noncomputable def tetrahedronVolume (t : Tetrahedron) : ℝ := sorry

/-- Theorem: The volume of the specified tetrahedron is 4√6/3 -/
theorem tetrahedron_volume_is_4_sqrt_6_div_3 (t : Tetrahedron) :
  tetrahedronVolume t = 4 * Real.sqrt 6 / 3 := by
  sorry


end NUMINAMATH_CALUDE_tetrahedron_volume_is_4_sqrt_6_div_3_l1445_144534


namespace NUMINAMATH_CALUDE_blue_face_prob_half_l1445_144512

/-- A rectangular prism with colored faces -/
structure ColoredPrism where
  green_faces : ℕ
  yellow_faces : ℕ
  blue_faces : ℕ

/-- The probability of rolling a blue face on a colored prism -/
def blue_face_probability (prism : ColoredPrism) : ℚ :=
  prism.blue_faces / (prism.green_faces + prism.yellow_faces + prism.blue_faces)

/-- Theorem: The probability of rolling a blue face on the given prism is 1/2 -/
theorem blue_face_prob_half :
  let prism : ColoredPrism := ⟨4, 2, 6⟩
  blue_face_probability prism = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_blue_face_prob_half_l1445_144512


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l1445_144598

/-- Proves that the speed of a boat in still water is 16 km/hr given specific downstream conditions. -/
theorem boat_speed_in_still_water 
  (stream_speed : ℝ) 
  (downstream_time : ℝ) 
  (downstream_distance : ℝ) 
  (h1 : stream_speed = 5)
  (h2 : downstream_time = 6)
  (h3 : downstream_distance = 126) :
  downstream_distance = (boat_speed + stream_speed) * downstream_time → 
  boat_speed = 16 :=
by
  sorry

#check boat_speed_in_still_water

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l1445_144598


namespace NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l1445_144528

theorem smallest_prime_divisor_of_sum (p : ℕ → ℕ → Prop) :
  (∀ x y, p x y ↔ x ∣ y) →
  p 2 (3^19 + 11^13) ∧ 
  (∀ q, q < 2 → q.Prime → ¬p q (3^19 + 11^13)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l1445_144528


namespace NUMINAMATH_CALUDE_incorrect_step_l1445_144561

theorem incorrect_step (a b : ℝ) (h : a < b) : ¬(2 * (a - b)^2 < (a - b)^2) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_step_l1445_144561


namespace NUMINAMATH_CALUDE_even_multiples_of_45_l1445_144535

theorem even_multiples_of_45 :
  let lower_bound := 449
  let upper_bound := 990
  let count_even_multiples := (upper_bound - lower_bound) / (45 * 2)
  count_even_multiples = 6.022222222222222 := by
  sorry

end NUMINAMATH_CALUDE_even_multiples_of_45_l1445_144535


namespace NUMINAMATH_CALUDE_smallest_two_digit_prime_with_reversed_composite_div_five_l1445_144530

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def reverse_digits (n : ℕ) : ℕ :=
  let tens := n / 10
  let ones := n % 10
  ones * 10 + tens

def is_composite (n : ℕ) : Prop := ∃ m : ℕ, m > 1 ∧ m < n ∧ m ∣ n

theorem smallest_two_digit_prime_with_reversed_composite_div_five :
  ∃ (n : ℕ),
    n ≥ 20 ∧ n < 30 ∧
    is_prime n ∧
    is_composite (reverse_digits n) ∧
    (reverse_digits n) % 5 = 0 ∧
    ∀ (m : ℕ), m ≥ 20 ∧ m < n →
      ¬(is_prime m ∧ is_composite (reverse_digits m) ∧ (reverse_digits m) % 5 = 0) :=
  by sorry

end NUMINAMATH_CALUDE_smallest_two_digit_prime_with_reversed_composite_div_five_l1445_144530


namespace NUMINAMATH_CALUDE_sachins_age_l1445_144587

theorem sachins_age (rahuls_age : ℝ) : 
  (rahuls_age + 7) / rahuls_age = 11 / 9 → rahuls_age + 7 = 38.5 := by
  sorry

end NUMINAMATH_CALUDE_sachins_age_l1445_144587


namespace NUMINAMATH_CALUDE_apartment_office_sale_net_effect_l1445_144573

theorem apartment_office_sale_net_effect :
  ∀ (apartment_cost office_cost : ℝ),
  apartment_cost * (1 - 0.25) = 15000 →
  office_cost * (1 + 0.25) = 15000 →
  apartment_cost + office_cost - 2 * 15000 = 2000 :=
by
  sorry

end NUMINAMATH_CALUDE_apartment_office_sale_net_effect_l1445_144573


namespace NUMINAMATH_CALUDE_lucy_fish_purchase_l1445_144533

theorem lucy_fish_purchase (current : ℕ) (desired : ℕ) (h1 : current = 212) (h2 : desired = 280) :
  desired - current = 68 := by
  sorry

end NUMINAMATH_CALUDE_lucy_fish_purchase_l1445_144533


namespace NUMINAMATH_CALUDE_difference_value_l1445_144545

theorem difference_value (n : ℚ) : n = 45 → (n / 3 - 5 : ℚ) = 10 := by
  sorry

end NUMINAMATH_CALUDE_difference_value_l1445_144545


namespace NUMINAMATH_CALUDE_smallest_n_for_P_less_than_threshold_l1445_144581

/-- Represents the number of boxes in the game --/
def total_boxes : ℕ := 2023

/-- Represents the number of boxes with 2 red marbles --/
def boxes_with_two_red : ℕ := 1012

/-- Calculates the probability of drawing a red marble from a box --/
def prob_red (box_number : ℕ) : ℚ :=
  if box_number ≤ boxes_with_two_red then
    2 / (box_number + 2)
  else
    1 / (box_number + 1)

/-- Calculates the probability of drawing a white marble from a box --/
def prob_white (box_number : ℕ) : ℚ :=
  1 - prob_red box_number

/-- Represents the probability of Isabella stopping after drawing exactly n marbles --/
noncomputable def P (n : ℕ) : ℚ :=
  sorry -- Definition of P(n) based on the game rules

/-- Theorem stating that 51 is the smallest n for which P(n) < 1/2023 --/
theorem smallest_n_for_P_less_than_threshold :
  (∀ k < 51, P k ≥ 1 / total_boxes) ∧
  P 51 < 1 / total_boxes :=
sorry

#check smallest_n_for_P_less_than_threshold

end NUMINAMATH_CALUDE_smallest_n_for_P_less_than_threshold_l1445_144581


namespace NUMINAMATH_CALUDE_quadratic_radical_equivalence_l1445_144538

theorem quadratic_radical_equivalence (x : ℝ) :
  (∃ (y : ℝ), y > 0 ∧ y * y = x - 1 ∧ (∀ (z : ℝ), z > 0 → z * z = 8 → ∃ (k : ℚ), y = k * z)) →
  x = 9 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_radical_equivalence_l1445_144538


namespace NUMINAMATH_CALUDE_staplers_left_after_stapling_l1445_144575

/-- The number of staplers left after stapling some reports -/
def staplers_left (initial_staplers : ℕ) (dozen_reports_stapled : ℕ) : ℕ :=
  initial_staplers - dozen_reports_stapled * 12

/-- Theorem: Given 50 initial staplers and 3 dozen reports stapled, 14 staplers are left -/
theorem staplers_left_after_stapling :
  staplers_left 50 3 = 14 := by
  sorry

end NUMINAMATH_CALUDE_staplers_left_after_stapling_l1445_144575


namespace NUMINAMATH_CALUDE_not_perfect_square_l1445_144529

theorem not_perfect_square (n : ℕ) (d : ℕ) (h : d > 0) (h' : d ∣ 2 * n^2) :
  ¬∃ (x : ℕ), x^2 = n^2 + d := by
sorry

end NUMINAMATH_CALUDE_not_perfect_square_l1445_144529


namespace NUMINAMATH_CALUDE_prop_q_indeterminate_l1445_144510

theorem prop_q_indeterminate (h1 : p ∨ q) (h2 : ¬(¬p)) : 
  (q ∨ ¬q) ∧ (∃ (v : Prop), v = q) :=
by sorry

end NUMINAMATH_CALUDE_prop_q_indeterminate_l1445_144510


namespace NUMINAMATH_CALUDE_oak_trees_after_planting_l1445_144501

/-- The number of oak trees in the park after planting -/
def total_oak_trees (initial : ℕ) (planted : ℕ) : ℕ :=
  initial + planted

theorem oak_trees_after_planting :
  total_oak_trees 5 4 = 9 := by
  sorry

end NUMINAMATH_CALUDE_oak_trees_after_planting_l1445_144501


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l1445_144590

theorem sum_of_a_and_b (a b : ℝ) : 
  (a + Real.sqrt b + (a - Real.sqrt b) = -6) →
  ((a + Real.sqrt b) * (a - Real.sqrt b) = 4) →
  a + b = 2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l1445_144590


namespace NUMINAMATH_CALUDE_blue_ball_count_l1445_144549

/-- The number of balls of each color in a box --/
structure BallCounts where
  blue : ℕ
  red : ℕ
  green : ℕ
  yellow : ℕ

/-- The conditions of the ball counting problem --/
def ballProblem (counts : BallCounts) : Prop :=
  counts.red = 4 ∧
  counts.green = 3 * counts.blue ∧
  counts.yellow = 2 * counts.red ∧
  counts.blue + counts.red + counts.green + counts.yellow = 36

theorem blue_ball_count :
  ∃ (counts : BallCounts), ballProblem counts ∧ counts.blue = 6 := by
  sorry

end NUMINAMATH_CALUDE_blue_ball_count_l1445_144549


namespace NUMINAMATH_CALUDE_rectangular_solid_surface_area_l1445_144589

/-- The surface area of a rectangular solid. -/
def surface_area (length width depth : ℝ) : ℝ :=
  2 * (length * width + length * depth + width * depth)

/-- Theorem: The surface area of a rectangular solid with length 10 meters, width 9 meters, 
    and depth 6 meters is 408 square meters. -/
theorem rectangular_solid_surface_area :
  surface_area 10 9 6 = 408 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_surface_area_l1445_144589


namespace NUMINAMATH_CALUDE_probability_of_no_mismatch_l1445_144597

/-- The number of red socks -/
def num_red_socks : ℕ := 4

/-- The number of blue socks -/
def num_blue_socks : ℕ := 4

/-- The total number of socks -/
def total_socks : ℕ := num_red_socks + num_blue_socks

/-- The number of pairs to be formed -/
def num_pairs : ℕ := total_socks / 2

/-- The number of ways to divide red socks into pairs -/
def red_pairings : ℕ := (Nat.choose num_red_socks 2) / 2

/-- The number of ways to divide blue socks into pairs -/
def blue_pairings : ℕ := (Nat.choose num_blue_socks 2) / 2

/-- The total number of favorable pairings -/
def favorable_pairings : ℕ := red_pairings * blue_pairings

/-- The total number of possible pairings -/
def total_pairings : ℕ := (Nat.factorial total_socks) / ((Nat.factorial 2)^num_pairs * Nat.factorial num_pairs)

/-- The probability of no mismatched pairs -/
def probability_no_mismatch : ℚ := favorable_pairings / total_pairings

theorem probability_of_no_mismatch : probability_no_mismatch = 3 / 35 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_no_mismatch_l1445_144597


namespace NUMINAMATH_CALUDE_max_value_of_expression_l1445_144593

theorem max_value_of_expression (a b c d e f g h k : Int) 
  (ha : a = 1 ∨ a = -1) (hb : b = 1 ∨ b = -1) (hc : c = 1 ∨ c = -1)
  (hd : d = 1 ∨ d = -1) (he : e = 1 ∨ e = -1) (hf : f = 1 ∨ f = -1)
  (hg : g = 1 ∨ g = -1) (hh : h = 1 ∨ h = -1) (hk : k = 1 ∨ k = -1) :
  (∃ (a' b' c' d' e' f' g' h' k' : Int),
    (a' = 1 ∨ a' = -1) ∧ (b' = 1 ∨ b' = -1) ∧ (c' = 1 ∨ c' = -1) ∧
    (d' = 1 ∨ d' = -1) ∧ (e' = 1 ∨ e' = -1) ∧ (f' = 1 ∨ f' = -1) ∧
    (g' = 1 ∨ g' = -1) ∧ (h' = 1 ∨ h' = -1) ∧ (k' = 1 ∨ k' = -1) ∧
    a'*e'*k' - a'*f'*h' + b'*f'*g' - b'*d'*k' + c'*d'*h' - c'*e'*g' = 4) ∧
  (∀ (a' b' c' d' e' f' g' h' k' : Int),
    (a' = 1 ∨ a' = -1) → (b' = 1 ∨ b' = -1) → (c' = 1 ∨ c' = -1) →
    (d' = 1 ∨ d' = -1) → (e' = 1 ∨ e' = -1) → (f' = 1 ∨ f' = -1) →
    (g' = 1 ∨ g' = -1) → (h' = 1 ∨ h' = -1) → (k' = 1 ∨ k' = -1) →
    a'*e'*k' - a'*f'*h' + b'*f'*g' - b'*d'*k' + c'*d'*h' - c'*e'*g' ≤ 4) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l1445_144593


namespace NUMINAMATH_CALUDE_battleship_theorem_l1445_144516

/-- Represents a rectangular grid -/
structure Grid :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a ship on the grid -/
structure Ship :=
  (length : ℕ)
  (width : ℕ)

/-- Represents a set of connected cells -/
structure ConnectedCells :=
  (num_cells : ℕ)

/-- The minimum number of shots needed to guarantee hitting a ship on a grid -/
def min_shots_to_hit_ship (g : Grid) (s : Ship) : ℕ := sorry

/-- The minimum number of shots needed to guarantee hitting connected cells on a grid -/
def min_shots_to_hit_connected_cells (g : Grid) (c : ConnectedCells) : ℕ := sorry

/-- The main theorem for the Battleship problem -/
theorem battleship_theorem (g : Grid) (s : Ship) (c : ConnectedCells) :
  g.rows = 7 ∧ g.cols = 7 ∧ 
  ((s.length = 1 ∧ s.width = 4) ∨ (s.length = 4 ∧ s.width = 1)) ∧
  c.num_cells = 4 →
  (min_shots_to_hit_ship g s = 12) ∧
  (min_shots_to_hit_connected_cells g c = 20) := by sorry

end NUMINAMATH_CALUDE_battleship_theorem_l1445_144516


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l1445_144557

theorem complex_modulus_problem (z : ℂ) (h : z * (2 + Complex.I) = 5 * Complex.I - 10) : 
  Complex.abs z = 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l1445_144557


namespace NUMINAMATH_CALUDE_unique_a_value_l1445_144570

def A (a : ℝ) : Set ℝ := {-4, 2*a-1, a^2}
def B (a : ℝ) : Set ℝ := {a-5, 1-a, 9}

theorem unique_a_value : ∃! a : ℝ, (9 ∈ (A a ∩ B a)) ∧ ({9} = A a ∩ B a) := by
  sorry

end NUMINAMATH_CALUDE_unique_a_value_l1445_144570


namespace NUMINAMATH_CALUDE_tangent_perpendicular_l1445_144519

-- Define the curve f(x) = x²
def f (x : ℝ) : ℝ := x^2

-- Define the line perpendicular to the tangent
def perp_line (x y : ℝ) : Prop := x + 4*y - 8 = 0

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := 4*x - y - 4 = 0

-- Theorem statement
theorem tangent_perpendicular :
  ∃ (x₀ y₀ : ℝ),
    -- (x₀, y₀) is on the curve
    f x₀ = y₀ ∧
    -- The tangent at (x₀, y₀) is perpendicular to the given line
    (∀ (x y : ℝ), perp_line x y → (y - y₀) = -(1/4) * (x - x₀)) ∧
    -- The tangent line equation
    tangent_line x₀ y₀ :=
sorry

end NUMINAMATH_CALUDE_tangent_perpendicular_l1445_144519


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l1445_144594

theorem partial_fraction_decomposition :
  ∃ (A B : ℚ), A = 73/15 ∧ B = 17/15 ∧
  ∀ (x : ℚ), x ≠ 12 → x ≠ -3 →
    (6*x + 1) / (x^2 - 9*x - 36) = A / (x - 12) + B / (x + 3) :=
by sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l1445_144594


namespace NUMINAMATH_CALUDE_cherry_pies_count_l1445_144552

theorem cherry_pies_count (total_pies : ℕ) (apple_ratio blueberry_ratio cherry_ratio : ℕ) 
  (h_total : total_pies = 36)
  (h_ratio : apple_ratio = 2 ∧ blueberry_ratio = 5 ∧ cherry_ratio = 4) :
  (cherry_ratio * total_pies) / (apple_ratio + blueberry_ratio + cherry_ratio) = 13 := by
  sorry

end NUMINAMATH_CALUDE_cherry_pies_count_l1445_144552


namespace NUMINAMATH_CALUDE_urn_problem_l1445_144513

theorem urn_problem (N : ℝ) : 
  (5 / 10 * 20 / (20 + N) + 5 / 10 * N / (20 + N) = 0.6) → N = 20 := by
  sorry

end NUMINAMATH_CALUDE_urn_problem_l1445_144513


namespace NUMINAMATH_CALUDE_certain_number_problem_l1445_144572

theorem certain_number_problem : ∃ x : ℕ, 
  220040 = (x + 445) * (2 * (x - 445)) + 40 ∧ x = 555 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l1445_144572


namespace NUMINAMATH_CALUDE_range_of_f_l1445_144543

noncomputable def f (x : ℝ) : ℝ := Real.arctan x + Real.arctan ((2 - x) / (2 + x))

theorem range_of_f :
  Set.range f = {π/4, Real.arctan 2} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l1445_144543


namespace NUMINAMATH_CALUDE_always_odd_l1445_144580

theorem always_odd (n : ℤ) : ∃ k : ℤ, 2017 + 2*n = 2*k + 1 := by
  sorry

end NUMINAMATH_CALUDE_always_odd_l1445_144580


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1445_144525

theorem quadratic_inequality_solution_set :
  let f : ℝ → ℝ := λ x => x^2 - 50*x + 601
  let solution_set := {x : ℝ | f x ≤ 9}
  let lower_bound := (50 - Real.sqrt 132) / 2
  let upper_bound := (50 + Real.sqrt 132) / 2
  solution_set = Set.Icc lower_bound upper_bound :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1445_144525


namespace NUMINAMATH_CALUDE_option_b_is_best_l1445_144592

-- Define the problem parameters
def total_metal_needed : ℝ := 635
def metal_in_storage : ℝ := 276
def aluminum_percentage : ℝ := 0.60
def steel_percentage : ℝ := 0.40

-- Define supplier options
structure Supplier :=
  (aluminum_price : ℝ)
  (steel_price : ℝ)

def option_a : Supplier := ⟨1.30, 0.90⟩
def option_b : Supplier := ⟨1.10, 1.00⟩
def option_c : Supplier := ⟨1.25, 0.95⟩

-- Calculate additional metal needed
def additional_metal_needed : ℝ := total_metal_needed - metal_in_storage

-- Calculate cost for a supplier
def calculate_cost (s : Supplier) : ℝ :=
  (additional_metal_needed * aluminum_percentage * s.aluminum_price) +
  (additional_metal_needed * steel_percentage * s.steel_price)

-- Theorem to prove
theorem option_b_is_best :
  calculate_cost option_b < calculate_cost option_a ∧
  calculate_cost option_b < calculate_cost option_c :=
by sorry

end NUMINAMATH_CALUDE_option_b_is_best_l1445_144592


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1445_144569

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

/-- The theorem statement -/
theorem geometric_sequence_property (a : ℕ → ℝ) :
  GeometricSequence a →
  a 3 * a 5 * a 7 * a 9 * a 11 = 243 →
  a 9 ^ 2 / a 11 = 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1445_144569


namespace NUMINAMATH_CALUDE_investment_principal_calculation_l1445_144523

/-- Proves that given an investment with a monthly interest payment of $228 and a simple annual interest rate of 9%, the principal amount of the investment is $30,400. -/
theorem investment_principal_calculation (monthly_interest : ℝ) (annual_rate : ℝ) :
  monthly_interest = 228 →
  annual_rate = 0.09 →
  ∃ principal : ℝ, principal = 30400 ∧ monthly_interest = principal * (annual_rate / 12) :=
by sorry

end NUMINAMATH_CALUDE_investment_principal_calculation_l1445_144523


namespace NUMINAMATH_CALUDE_problem_solution_l1445_144596

-- Define the conditions
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0
def q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0 ∧ x^2 + 3*x - 10 > 0

-- Define the theorem
theorem problem_solution :
  (∀ a : ℝ, a > 0) →
  (∀ x : ℝ, p x 1 ∧ q x → 2 < x ∧ x < 3) ∧
  (∀ a : ℝ, (∀ x : ℝ, ¬(p x a) → ¬(q x)) ∧ (∃ x : ℝ, ¬(q x) ∧ p x a) → 1 < a ∧ a ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l1445_144596


namespace NUMINAMATH_CALUDE_smallest_number_l1445_144542

def A : ℕ := 36

def B : ℕ := 27 + 5

def C : ℕ := 3 * 10

def D : ℕ := 40 - 3

theorem smallest_number (h : A = 36 ∧ B = 27 + 5 ∧ C = 3 * 10 ∧ D = 40 - 3) :
  C ≤ A ∧ C ≤ B ∧ C ≤ D :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l1445_144542


namespace NUMINAMATH_CALUDE_cn_length_l1445_144595

/-- Right-angled triangle with squares on legs -/
structure RightTriangleWithSquares where
  -- Points
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ
  F : ℝ × ℝ
  G : ℝ × ℝ
  M : ℝ × ℝ
  N : ℝ × ℝ
  -- Conditions
  right_angle : (A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2) = 0
  ac_length : Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2) = 4
  bc_length : Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) = 1
  square_acde : (D.1 - A.1) = (E.1 - C.1) ∧ (D.2 - A.2) = (E.2 - C.2) ∧
                 (D.1 - A.1) * (E.1 - C.1) + (D.2 - A.2) * (E.2 - C.2) = 0
  square_bcfg : (F.1 - B.1) = (G.1 - C.1) ∧ (F.2 - B.2) = (G.2 - C.2) ∧
                 (F.1 - B.1) * (G.1 - C.1) + (F.2 - B.2) * (G.2 - C.2) = 0
  m_midpoint : M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  n_on_cm : (N.2 - C.2) * (M.1 - C.1) = (N.1 - C.1) * (M.2 - C.2)
  n_on_df : (N.2 - D.2) * (F.1 - D.1) = (N.1 - D.1) * (F.2 - D.2)

/-- The length of CN is √17 -/
theorem cn_length (t : RightTriangleWithSquares) : 
  Real.sqrt ((t.N.1 - t.C.1)^2 + (t.N.2 - t.C.2)^2) = Real.sqrt 17 := by
  sorry

end NUMINAMATH_CALUDE_cn_length_l1445_144595


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l1445_144517

-- Define the universe set U
def U : Set ℝ := {x : ℝ | 0 ≤ x ∧ x < 10}

-- Define subset A
def A : Set ℝ := {x : ℝ | 2 < x ∧ x ≤ 4}

-- Define subset B
def B : Set ℝ := {x : ℝ | 3 < x ∧ x ≤ 5}

-- State the theorem
theorem complement_intersection_theorem :
  (U \ A) ∩ (U \ B) = {x : ℝ | (0 ≤ x ∧ x ≤ 2) ∨ (5 < x ∧ x < 10)} := by
  sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l1445_144517


namespace NUMINAMATH_CALUDE_jeff_saturday_laps_l1445_144599

theorem jeff_saturday_laps (total_laps : ℕ) (sunday_morning_laps : ℕ) (remaining_laps : ℕ) 
  (h1 : total_laps = 98)
  (h2 : sunday_morning_laps = 15)
  (h3 : remaining_laps = 56) :
  total_laps - (sunday_morning_laps + remaining_laps) = 27 := by
  sorry

end NUMINAMATH_CALUDE_jeff_saturday_laps_l1445_144599


namespace NUMINAMATH_CALUDE_expression_evaluation_l1445_144576

theorem expression_evaluation : (4 * 6) / (12 * 14) * (8 * 12 * 14) / (4 * 6 * 8) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1445_144576


namespace NUMINAMATH_CALUDE_alex_peeled_22_potatoes_l1445_144531

/-- The number of potatoes Alex peeled -/
def alexPotatoes (totalPotatoes : ℕ) (homerRate alextRate : ℕ) (alexJoinTime : ℕ) : ℕ :=
  let homerPotatoes := homerRate * alexJoinTime
  let remainingPotatoes := totalPotatoes - homerPotatoes
  let combinedRate := homerRate + alextRate
  let remainingTime := remainingPotatoes / combinedRate
  alextRate * remainingTime

theorem alex_peeled_22_potatoes :
  alexPotatoes 60 4 6 6 = 22 := by
  sorry

end NUMINAMATH_CALUDE_alex_peeled_22_potatoes_l1445_144531


namespace NUMINAMATH_CALUDE_complex_power_four_l1445_144514

theorem complex_power_four : 
  (3 * (Complex.cos (π / 6) + Complex.I * Complex.sin (π / 6)))^4 = 
  Complex.mk (-40.5) (40.5 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_complex_power_four_l1445_144514


namespace NUMINAMATH_CALUDE_f_seven_equals_f_nine_l1445_144546

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the property of f being decreasing on (8, +∞)
def DecreasingAfterEight (f : ℝ → ℝ) : Prop :=
  ∀ x y, x > 8 ∧ y > x → f y < f x

-- Define the property of f(x+8) being an even function
def EvenShiftedByEight (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 8) = f (-x + 8)

-- State the theorem
theorem f_seven_equals_f_nine
  (h1 : DecreasingAfterEight f)
  (h2 : EvenShiftedByEight f) :
  f 7 = f 9 := by
  sorry

end NUMINAMATH_CALUDE_f_seven_equals_f_nine_l1445_144546


namespace NUMINAMATH_CALUDE_two_digit_swap_l1445_144564

/-- 
Given a two-digit number with 1 in the tens place and x in the ones place,
if swapping these digits results in a number 18 greater than the original,
then the equation 10x + 1 - (10 + x) = 18 holds.
-/
theorem two_digit_swap (x : ℕ) : 
  (x < 10) →  -- Ensure x is a single digit
  (10 * x + 1) - (10 + x) = 18 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_swap_l1445_144564


namespace NUMINAMATH_CALUDE_path_count_equals_binomial_coefficient_l1445_144584

/-- The number of paths composed of n rises and n descents of the same amplitude -/
def pathCount (n : ℕ) : ℕ := Nat.choose (2 * n) n

/-- Theorem: The number of paths composed of n rises and n descents of the same amplitude
    is equal to the binomial coefficient (2n choose n) -/
theorem path_count_equals_binomial_coefficient (n : ℕ) :
  pathCount n = Nat.choose (2 * n) n := by sorry

end NUMINAMATH_CALUDE_path_count_equals_binomial_coefficient_l1445_144584


namespace NUMINAMATH_CALUDE_sum_of_sixth_powers_of_roots_l1445_144524

theorem sum_of_sixth_powers_of_roots (p q : ℝ) : 
  p^2 - 3*p*Real.sqrt 3 + 3 = 0 → 
  q^2 - 3*q*Real.sqrt 3 + 3 = 0 → 
  p^6 + q^6 = 99171 := by
sorry

end NUMINAMATH_CALUDE_sum_of_sixth_powers_of_roots_l1445_144524


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1445_144555

theorem complex_equation_solution (m : ℝ) : 
  (m - 1 : ℂ) + 2*m*Complex.I = 1 + 4*Complex.I → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1445_144555


namespace NUMINAMATH_CALUDE_min_dot_product_op_ab_l1445_144502

open Real

/-- The minimum dot product of OP and AB -/
theorem min_dot_product_op_ab :
  ∀ x : ℝ,
  let A : ℝ × ℝ := (1, 0)
  let B : ℝ × ℝ := (0, 1)
  let P : ℝ × ℝ := (x, exp x)
  let OP : ℝ × ℝ := P
  let AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
  (OP.1 * AB.1 + OP.2 * AB.2) ≥ 1 :=
by sorry

#check min_dot_product_op_ab

end NUMINAMATH_CALUDE_min_dot_product_op_ab_l1445_144502


namespace NUMINAMATH_CALUDE_no_x_squared_term_l1445_144550

theorem no_x_squared_term (p : ℚ) : 
  (∀ x, (x^2 + p*x) * (x^2 - 3*x + 1) = x^4 + (p-3)*x^3 + 0*x^2 + p*x) → p = 1/3 := by
sorry

end NUMINAMATH_CALUDE_no_x_squared_term_l1445_144550


namespace NUMINAMATH_CALUDE_carp_classification_l1445_144515

-- Define the characteristics of an individual
structure IndividualCharacteristics where
  birth : Bool
  death : Bool
  gender : Bool
  age : ℕ

-- Define the characteristics of a population
structure PopulationCharacteristics where
  birthRate : ℝ
  deathRate : ℝ
  genderRatio : ℝ
  ageComposition : List ℝ

-- Define the types
inductive EntityType
  | Carp
  | CarpPopulation

-- Define the main theorem
theorem carp_classification 
  (a : IndividualCharacteristics) 
  (b : PopulationCharacteristics) : 
  (EntityType.Carp, EntityType.CarpPopulation) = 
  (
    match a with
    | { birth := _, death := _, gender := _, age := _ } => EntityType.Carp,
    match b with
    | { birthRate := _, deathRate := _, genderRatio := _, ageComposition := _ } => EntityType.CarpPopulation
  ) := by
  sorry

end NUMINAMATH_CALUDE_carp_classification_l1445_144515


namespace NUMINAMATH_CALUDE_number_of_trucks_filled_l1445_144503

/-- Prove that the number of trucks filled up is 2, given the specified conditions. -/
theorem number_of_trucks_filled (service_cost : ℚ) (fuel_cost_per_liter : ℚ) (total_cost : ℚ)
  (num_minivans : ℕ) (minivan_capacity : ℚ) (truck_capacity_factor : ℚ) :
  service_cost = 23/10 →
  fuel_cost_per_liter = 7/10 →
  total_cost = 396 →
  num_minivans = 4 →
  minivan_capacity = 65 →
  truck_capacity_factor = 22/10 →
  ∃ (num_trucks : ℕ), num_trucks = 2 ∧
    total_cost = (num_minivans * (service_cost + fuel_cost_per_liter * minivan_capacity)) +
                 (num_trucks * (service_cost + fuel_cost_per_liter * (minivan_capacity * truck_capacity_factor))) :=
by sorry


end NUMINAMATH_CALUDE_number_of_trucks_filled_l1445_144503


namespace NUMINAMATH_CALUDE_pedestrian_cyclist_speeds_l1445_144522

/-- Proves that given the conditions of the problem, the pedestrian's speed is 5 km/h and the cyclist's speed is 11 km/h -/
theorem pedestrian_cyclist_speeds :
  ∀ (v₁ v₂ : ℝ),
    (27 : ℝ) > 0 →  -- Distance from A to B is 27 km
    (12 / 5 * v₁ - v₂ = 1) →  -- After 1 hour of cyclist's travel, they were 1 km behind the pedestrian
    (27 - 17 / 5 * v₁ = 2 * (27 - 2 * v₂)) →  -- After 2 hours of cyclist's travel, the cyclist had half the distance to B remaining compared to the pedestrian
    v₁ = 5 ∧ v₂ = 11 := by
  sorry

#check pedestrian_cyclist_speeds

end NUMINAMATH_CALUDE_pedestrian_cyclist_speeds_l1445_144522


namespace NUMINAMATH_CALUDE_product_sum_7293_l1445_144504

theorem product_sum_7293 : ∃ (a b : ℕ), 
  10 ≤ a ∧ a ≤ 99 ∧ 
  10 ≤ b ∧ b ≤ 99 ∧ 
  a * b = 7293 ∧ 
  a + b = 114 := by
sorry

end NUMINAMATH_CALUDE_product_sum_7293_l1445_144504


namespace NUMINAMATH_CALUDE_not_perfect_square_l1445_144547

theorem not_perfect_square (n d : ℕ+) (h : d ∣ (2 * n^2)) :
  ¬ ∃ (x : ℕ), (n : ℕ)^2 + (d : ℕ) = x^2 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_l1445_144547


namespace NUMINAMATH_CALUDE_melanie_balloons_l1445_144577

theorem melanie_balloons (joan_balloons total_balloons : ℕ) 
  (h1 : joan_balloons = 40)
  (h2 : total_balloons = 81) :
  total_balloons - joan_balloons = 41 :=
by
  sorry

end NUMINAMATH_CALUDE_melanie_balloons_l1445_144577


namespace NUMINAMATH_CALUDE_chris_savings_l1445_144554

theorem chris_savings (x : ℝ) 
  (grandmother : ℝ) (aunt_uncle : ℝ) (parents : ℝ) (total : ℝ)
  (h1 : grandmother = 25)
  (h2 : aunt_uncle = 20)
  (h3 : parents = 75)
  (h4 : total = 279)
  (h5 : x + grandmother + aunt_uncle + parents = total) :
  x = 159 := by
sorry

end NUMINAMATH_CALUDE_chris_savings_l1445_144554


namespace NUMINAMATH_CALUDE_eight_people_arrangements_l1445_144560

/-- The number of ways to arrange n distinct objects in a line -/
def linearArrangements (n : ℕ) : ℕ := Nat.factorial n

/-- There are 8! ways to arrange 8 people in a line -/
theorem eight_people_arrangements : linearArrangements 8 = 40320 := by
  sorry

end NUMINAMATH_CALUDE_eight_people_arrangements_l1445_144560


namespace NUMINAMATH_CALUDE_complete_square_sum_l1445_144585

theorem complete_square_sum (x : ℝ) : ∃ (a b c : ℤ), 
  (25 * x^2 + 30 * x - 75 = 0 ↔ (a * x + b)^2 = c) ∧ 
  a > 0 ∧ 
  a + b + c = -58 := by
  sorry

end NUMINAMATH_CALUDE_complete_square_sum_l1445_144585


namespace NUMINAMATH_CALUDE_power_8_2048_mod_50_l1445_144556

theorem power_8_2048_mod_50 : 8^2048 % 50 = 38 := by sorry

end NUMINAMATH_CALUDE_power_8_2048_mod_50_l1445_144556


namespace NUMINAMATH_CALUDE_mother_carrots_count_l1445_144506

/-- The number of carrots Vanessa picked -/
def vanessa_carrots : ℕ := 17

/-- The number of good carrots -/
def good_carrots : ℕ := 24

/-- The number of bad carrots -/
def bad_carrots : ℕ := 7

/-- The number of carrots Vanessa's mother picked -/
def mother_carrots : ℕ := (good_carrots + bad_carrots) - vanessa_carrots

theorem mother_carrots_count : mother_carrots = 14 := by
  sorry

end NUMINAMATH_CALUDE_mother_carrots_count_l1445_144506


namespace NUMINAMATH_CALUDE_dans_candy_purchase_l1445_144553

def candy_problem (initial_money remaining_money candy_cost : ℕ) : Prop :=
  let spent_money := initial_money - remaining_money
  let num_candy_bars := spent_money / candy_cost
  num_candy_bars = 1

theorem dans_candy_purchase :
  candy_problem 4 1 3 := by sorry

end NUMINAMATH_CALUDE_dans_candy_purchase_l1445_144553


namespace NUMINAMATH_CALUDE_black_cars_count_l1445_144507

theorem black_cars_count (total : ℕ) (blue_fraction red_fraction green_fraction : ℚ) :
  total = 1824 →
  blue_fraction = 2 / 5 →
  red_fraction = 1 / 3 →
  green_fraction = 1 / 8 →
  ∃ (blue red green black : ℕ),
    blue + red + green + black = total ∧
    blue = ⌊blue_fraction * total⌋ ∧
    red = red_fraction * total ∧
    green = green_fraction * total ∧
    black = 259 :=
by sorry

end NUMINAMATH_CALUDE_black_cars_count_l1445_144507


namespace NUMINAMATH_CALUDE_square_area_increase_l1445_144520

theorem square_area_increase (x y : ℝ) : 
  (∀ s : ℝ, s = 3 → (s + x)^2 - s^2 = y) → 
  y = x^2 + 6*x := by sorry

end NUMINAMATH_CALUDE_square_area_increase_l1445_144520


namespace NUMINAMATH_CALUDE_nine_rooks_on_checkerboard_l1445_144500

/-- Represents a chessboard -/
structure Chessboard :=
  (size : Nat)
  (is_checkerboard : Bool)

/-- Represents a rook placement on a chessboard -/
structure RookPlacement :=
  (board : Chessboard)
  (num_rooks : Nat)
  (same_color : Bool)
  (non_attacking : Bool)

/-- Counts the number of valid rook placements -/
def count_rook_placements (placement : RookPlacement) : Nat :=
  sorry

/-- Theorem: The number of ways to place 9 non-attacking rooks on cells of the same color on a 9x9 checkerboard is 2880 -/
theorem nine_rooks_on_checkerboard :
  ∀ (board : Chessboard) (placement : RookPlacement),
    board.size = 9 ∧
    board.is_checkerboard = true ∧
    placement.board = board ∧
    placement.num_rooks = 9 ∧
    placement.same_color = true ∧
    placement.non_attacking = true →
    count_rook_placements placement = 2880 :=
  sorry

end NUMINAMATH_CALUDE_nine_rooks_on_checkerboard_l1445_144500


namespace NUMINAMATH_CALUDE_opposite_of_negative_fraction_l1445_144567

theorem opposite_of_negative_fraction :
  ∀ (x : ℚ), x = -6/7 → (x + 6/7 = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_fraction_l1445_144567


namespace NUMINAMATH_CALUDE_calculate_second_month_sale_second_month_sale_l1445_144558

/-- Given sales figures for 5 out of 6 months and the average sale, 
    prove the sales figure for the remaining month. -/
theorem calculate_second_month_sale 
  (sale1 sale3 sale4 sale5 sale6 average_sale : ℕ) : ℕ :=
  let total_sales := average_sale * 6
  let known_sales := sale1 + sale3 + sale4 + sale5 + sale6
  let sale2 := total_sales - known_sales
  sale2

/-- The sales figure for the second month is 9000. -/
theorem second_month_sale : 
  calculate_second_month_sale 5400 6300 7200 4500 1200 5600 = 9000 := by
  sorry

end NUMINAMATH_CALUDE_calculate_second_month_sale_second_month_sale_l1445_144558


namespace NUMINAMATH_CALUDE_garret_age_proof_l1445_144566

/-- Garret's current age -/
def garret_age : ℕ := 12

/-- Shane's current age -/
def shane_current_age : ℕ := 44

theorem garret_age_proof :
  (shane_current_age - 20 = 2 * garret_age) →
  garret_age = 12 := by
sorry

end NUMINAMATH_CALUDE_garret_age_proof_l1445_144566


namespace NUMINAMATH_CALUDE_closed_triangular_path_steps_divisible_by_three_l1445_144583

/-- A closed path on a triangular lattice -/
structure TriangularPath where
  steps : ℕ
  is_closed : Bool

/-- Theorem: The number of steps in a closed path on a triangular lattice is divisible by 3 -/
theorem closed_triangular_path_steps_divisible_by_three (path : TriangularPath) 
  (h : path.is_closed = true) : 
  ∃ k : ℕ, path.steps = 3 * k := by
  sorry

end NUMINAMATH_CALUDE_closed_triangular_path_steps_divisible_by_three_l1445_144583


namespace NUMINAMATH_CALUDE_parabola_and_line_properties_l1445_144526

-- Define the parabola
structure Parabola where
  vertex : ℝ × ℝ
  focus : ℝ × ℝ
  a : ℝ

-- Define a point on the parabola
def PointOnParabola (p : Parabola) (x y : ℝ) : Prop :=
  y = p.a * x^2

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a line
structure Line where
  slope : ℝ
  y_intercept : ℝ

-- Main theorem
theorem parabola_and_line_properties
  (p : Parabola)
  (A : ℝ × ℝ)
  (c : Circle)
  (m : Line) :
  p.vertex = (0, 0) →
  p.focus.1 = 0 →
  p.focus.2 > 0 →
  PointOnParabola p A.1 A.2 →
  c.center = A →
  c.radius = 2 →
  c.center.2 - c.radius = p.focus.2 →
  m.y_intercept = 6 →
  ∃ (P Q : ℝ × ℝ),
    PointOnParabola p P.1 P.2 ∧
    PointOnParabola p Q.1 Q.2 ∧
    P.2 = m.slope * P.1 + m.y_intercept ∧
    Q.2 = m.slope * Q.1 + m.y_intercept →
  (∀ (x y : ℝ), y = p.a * x^2 ↔ y = (1/4) * x^2) ∧
  (m.slope = 1/2 ∨ m.slope = -1/2) :=
sorry

end NUMINAMATH_CALUDE_parabola_and_line_properties_l1445_144526


namespace NUMINAMATH_CALUDE_symmetric_point_theorem_l1445_144574

/-- The coordinates of a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Given a point P, returns its symmetric point with respect to the origin -/
def symmetricPoint (p : Point3D) : Point3D :=
  { x := -p.x, y := -p.y, z := -p.z }

/-- Theorem: The symmetric point of P(1, 3, -5) with respect to the origin is (-1, -3, 5) -/
theorem symmetric_point_theorem :
  let P : Point3D := { x := 1, y := 3, z := -5 }
  symmetricPoint P = { x := -1, y := -3, z := 5 } := by
  sorry


end NUMINAMATH_CALUDE_symmetric_point_theorem_l1445_144574


namespace NUMINAMATH_CALUDE_power_and_division_equality_l1445_144565

theorem power_and_division_equality : (12 : ℕ)^3 * 6^4 / 432 = 5184 := by sorry

end NUMINAMATH_CALUDE_power_and_division_equality_l1445_144565


namespace NUMINAMATH_CALUDE_existence_of_special_integers_l1445_144508

theorem existence_of_special_integers : ∃ (a b : ℕ+), 
  (¬ (7 ∣ (a.val * b.val * (a.val + b.val)))) ∧ 
  (7 ∣ ((a.val + b.val)^7 - a.val^7 - b.val^7)) ∧
  (a.val = 18 ∧ b.val = 1) := by
sorry

end NUMINAMATH_CALUDE_existence_of_special_integers_l1445_144508


namespace NUMINAMATH_CALUDE_total_cost_calculation_l1445_144548

def muffin_cost : ℚ := 0.75
def juice_cost : ℚ := 1.45
def muffin_count : ℕ := 3

theorem total_cost_calculation : 
  (muffin_count : ℚ) * muffin_cost + juice_cost = 3.70 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_calculation_l1445_144548


namespace NUMINAMATH_CALUDE_binomial_18_10_l1445_144537

theorem binomial_18_10 (h1 : Nat.choose 16 7 = 11440) (h2 : Nat.choose 16 9 = 11440) :
  Nat.choose 18 10 = 47190 := by
  sorry

end NUMINAMATH_CALUDE_binomial_18_10_l1445_144537


namespace NUMINAMATH_CALUDE_triangle_inequality_l1445_144582

/-- Given a triangle with side lengths a, b, c, and semiperimeter p, 
    prove that 2√((p-b)(p-c)) ≤ a. -/
theorem triangle_inequality (a b c p : ℝ) 
    (h_positive : 0 < a ∧ 0 < b ∧ 0 < c)
    (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
    (h_semiperimeter : p = (a + b + c) / 2) : 
  2 * Real.sqrt ((p - b) * (p - c)) ≤ a := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1445_144582


namespace NUMINAMATH_CALUDE_convention_handshakes_l1445_144578

/-- Represents the convention of twins and triplets --/
structure Convention where
  twin_sets : ℕ
  triplet_sets : ℕ

/-- Calculates the total number of handshakes in the convention --/
def total_handshakes (c : Convention) : ℕ :=
  let twin_count := c.twin_sets * 2
  let triplet_count := c.triplet_sets * 3
  let twin_handshakes := (twin_count * (twin_count - 2)) / 2
  let triplet_handshakes := (triplet_count * (triplet_count - 3)) / 2
  let twin_to_triplet := twin_count * (triplet_count / 2)
  twin_handshakes + triplet_handshakes + twin_to_triplet

/-- The theorem stating that the total number of handshakes in the given convention is 354 --/
theorem convention_handshakes :
  total_handshakes ⟨10, 4⟩ = 354 := by sorry

end NUMINAMATH_CALUDE_convention_handshakes_l1445_144578


namespace NUMINAMATH_CALUDE_apps_after_deletion_l1445_144568

/-- Represents the number of apps on Faye's phone. -/
structure PhoneApps where
  total : ℕ
  gaming : ℕ
  utility : ℕ
  gaming_deleted : ℕ
  utility_deleted : ℕ

/-- Calculates the number of remaining apps after deletion. -/
def remaining_apps (apps : PhoneApps) : ℕ :=
  apps.total - (apps.gaming_deleted + apps.utility_deleted)

/-- Theorem stating the number of remaining apps after deletion. -/
theorem apps_after_deletion (apps : PhoneApps)
  (h1 : apps.total = 12)
  (h2 : apps.gaming = 5)
  (h3 : apps.utility = apps.total - apps.gaming)
  (h4 : apps.gaming_deleted = 4)
  (h5 : apps.utility_deleted = 3)
  (h6 : apps.gaming - apps.gaming_deleted ≥ 1)
  (h7 : apps.utility - apps.utility_deleted ≥ 1) :
  remaining_apps apps = 5 := by
  sorry


end NUMINAMATH_CALUDE_apps_after_deletion_l1445_144568


namespace NUMINAMATH_CALUDE_average_bull_weight_l1445_144571

/-- Represents a section of the farm with a ratio of cows to bulls -/
structure FarmSection where
  cows : ℕ
  bulls : ℕ

/-- Represents the farm with its sections and total cattle -/
structure Farm where
  sectionA : FarmSection
  sectionB : FarmSection
  sectionC : FarmSection
  totalCattle : ℕ
  totalBullWeight : ℕ

def farm : Farm := {
  sectionA := { cows := 7, bulls := 21 },
  sectionB := { cows := 5, bulls := 15 },
  sectionC := { cows := 3, bulls := 9 },
  totalCattle := 1220,
  totalBullWeight := 200000
}

theorem average_bull_weight (f : Farm) :
  f = farm →
  (f.totalBullWeight : ℚ) / (((f.sectionA.bulls + f.sectionB.bulls + f.sectionC.bulls) * f.totalCattle) / (f.sectionA.cows + f.sectionA.bulls + f.sectionB.cows + f.sectionB.bulls + f.sectionC.cows + f.sectionC.bulls)) = 218579 / 1000 := by
  sorry

end NUMINAMATH_CALUDE_average_bull_weight_l1445_144571


namespace NUMINAMATH_CALUDE_parabola_directrix_l1445_144527

/-- The equation of a parabola -/
def parabola_equation (x y : ℝ) : Prop := y = 8 * x^2

/-- The equation of the directrix -/
def directrix_equation (y : ℝ) : Prop := y = -2

/-- Theorem stating that the given directrix equation is correct for the given parabola -/
theorem parabola_directrix : 
  ∀ (x y : ℝ), parabola_equation x y → (∃ (d : ℝ), directrix_equation d ∧ 
    -- Additional conditions to relate the parabola and directrix
    (x^2 + (y - 2)^2 = (y + 2)^2)) := by
  sorry

end NUMINAMATH_CALUDE_parabola_directrix_l1445_144527


namespace NUMINAMATH_CALUDE_max_min_m_l1445_144509

theorem max_min_m (a b c : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0)
  (h4 : 3*a + 2*b + c = 5) (h5 : 2*a + b - 3*c = 1) :
  let m := 3*a + b - 7*c
  ∃ (m_max m_min : ℝ), 
    (∀ x, x = m → x ≤ m_max) ∧ 
    (∀ x, x = m → x ≥ m_min) ∧ 
    m_max = -1/11 ∧ 
    m_min = -5/7 :=
sorry

end NUMINAMATH_CALUDE_max_min_m_l1445_144509


namespace NUMINAMATH_CALUDE_sandy_molly_age_ratio_l1445_144536

/-- Proves that the ratio of Sandy's current age to Molly's current age is 4:3 -/
theorem sandy_molly_age_ratio :
  let sandy_future_age : ℕ := 34
  let years_to_future : ℕ := 6
  let molly_current_age : ℕ := 21
  let sandy_current_age : ℕ := sandy_future_age - years_to_future
  (sandy_current_age : ℚ) / (molly_current_age : ℚ) = 4 / 3 := by
  sorry

#check sandy_molly_age_ratio

end NUMINAMATH_CALUDE_sandy_molly_age_ratio_l1445_144536


namespace NUMINAMATH_CALUDE_cos_225_degrees_l1445_144540

theorem cos_225_degrees : Real.cos (225 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_225_degrees_l1445_144540


namespace NUMINAMATH_CALUDE_lcm_of_8_12_15_l1445_144505

theorem lcm_of_8_12_15 : Nat.lcm (Nat.lcm 8 12) 15 = 120 := by sorry

end NUMINAMATH_CALUDE_lcm_of_8_12_15_l1445_144505


namespace NUMINAMATH_CALUDE_dannys_english_marks_l1445_144511

theorem dannys_english_marks 
  (math_marks : ℕ) 
  (physics_marks : ℕ) 
  (chemistry_marks : ℕ) 
  (biology_marks : ℕ) 
  (average_marks : ℕ) 
  (total_subjects : ℕ) 
  (h1 : math_marks = 65) 
  (h2 : physics_marks = 82) 
  (h3 : chemistry_marks = 67) 
  (h4 : biology_marks = 75) 
  (h5 : average_marks = 73) 
  (h6 : total_subjects = 5) : 
  ∃ (english_marks : ℕ), english_marks = 76 ∧ 
    (english_marks + math_marks + physics_marks + chemistry_marks + biology_marks) / total_subjects = average_marks :=
by
  sorry

end NUMINAMATH_CALUDE_dannys_english_marks_l1445_144511


namespace NUMINAMATH_CALUDE_line_up_count_l1445_144591

/-- The number of ways to arrange n distinct objects --/
def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

/-- The number of ways to arrange k distinct objects from n objects --/
def permutations (n k : ℕ) : ℕ := 
  if k > n then 0
  else factorial n / factorial (n - k)

/-- The number of boys in the group --/
def num_boys : ℕ := 2

/-- The number of girls in the group --/
def num_girls : ℕ := 3

/-- The total number of people in the group --/
def total_people : ℕ := num_boys + num_girls

theorem line_up_count : 
  factorial total_people - factorial (total_people - 1) * factorial num_boys = 72 := by
  sorry

end NUMINAMATH_CALUDE_line_up_count_l1445_144591


namespace NUMINAMATH_CALUDE_cardinal_transitivity_l1445_144563

-- Define the theorem
theorem cardinal_transitivity (α β γ : Cardinal) 
  (h1 : α < β) (h2 : β < γ) : α < γ := by
  sorry

end NUMINAMATH_CALUDE_cardinal_transitivity_l1445_144563


namespace NUMINAMATH_CALUDE_solutions_to_equation_all_solutions_l1445_144551

def solutions : Set ℂ := {1 + Complex.I * (2 : ℂ)^(1/3 : ℂ), 
                          1 - Complex.I * (2 : ℂ)^(1/3 : ℂ), 
                          -1 + Complex.I * (2 : ℂ)^(1/3 : ℂ), 
                          -1 - Complex.I * (2 : ℂ)^(1/3 : ℂ), 
                          Complex.I * (2 : ℂ)^(1/3 : ℂ), 
                          -Complex.I * (2 : ℂ)^(1/3 : ℂ)}

theorem solutions_to_equation : ∀ z ∈ solutions, z^6 = -8 :=
by sorry

theorem all_solutions : ∀ z : ℂ, z^6 = -8 → z ∈ solutions :=
by sorry

end NUMINAMATH_CALUDE_solutions_to_equation_all_solutions_l1445_144551


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1445_144521

theorem complex_equation_solution (z : ℂ) : (Complex.I * z = 4 + 3 * Complex.I) → z = 3 - 4 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1445_144521


namespace NUMINAMATH_CALUDE_total_boys_l1445_144539

theorem total_boys (total_children happy_children sad_children neutral_children : ℕ)
  (girls happy_boys sad_girls neutral_boys : ℕ)
  (h1 : total_children = 60)
  (h2 : happy_children = 30)
  (h3 : sad_children = 10)
  (h4 : neutral_children = 20)
  (h5 : girls = 41)
  (h6 : happy_boys = 6)
  (h7 : sad_girls = 4)
  (h8 : neutral_boys = 7)
  (h9 : total_children = happy_children + sad_children + neutral_children)
  (h10 : total_children = girls + (happy_boys + (sad_children - sad_girls) + neutral_boys)) :
  happy_boys + (sad_children - sad_girls) + neutral_boys = 19 := by
  sorry

#check total_boys

end NUMINAMATH_CALUDE_total_boys_l1445_144539


namespace NUMINAMATH_CALUDE_unique_valid_number_l1445_144532

def is_valid_number (n : ℕ) : Prop :=
  350000 ≤ n ∧ n ≤ 359992 ∧ n % 100 = 2 ∧ n % 6 = 0

theorem unique_valid_number :
  ∃! n : ℕ, is_valid_number n ∧ n = 351152 := by sorry

end NUMINAMATH_CALUDE_unique_valid_number_l1445_144532

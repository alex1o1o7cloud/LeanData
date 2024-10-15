import Mathlib

namespace NUMINAMATH_CALUDE_range_of_m_l1824_182467

theorem range_of_m (m : ℝ) : 
  (∀ x y : ℝ, x > 0 → y > 0 → (2 * y / x + 8 * x / y > m^2 + 2 * m)) → 
  -4 < m ∧ m < 2 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l1824_182467


namespace NUMINAMATH_CALUDE_other_players_score_l1824_182472

theorem other_players_score (total_score : ℕ) (faye_score : ℕ) (total_players : ℕ) :
  total_score = 68 →
  faye_score = 28 →
  total_players = 5 →
  ∃ (other_player_score : ℕ),
    other_player_score * (total_players - 1) = total_score - faye_score ∧
    other_player_score = 10 := by
  sorry

end NUMINAMATH_CALUDE_other_players_score_l1824_182472


namespace NUMINAMATH_CALUDE_largest_x_floor_ratio_l1824_182427

theorem largest_x_floor_ratio : ∃ (x : ℝ), x = 63/8 ∧ 
  (∀ (y : ℝ), y > x → (⌊y⌋ : ℝ) / y ≠ 8/9) ∧ 
  (⌊x⌋ : ℝ) / x = 8/9 := by
  sorry

end NUMINAMATH_CALUDE_largest_x_floor_ratio_l1824_182427


namespace NUMINAMATH_CALUDE_rectangular_hall_dimensions_l1824_182421

theorem rectangular_hall_dimensions (length width : ℝ) (area : ℝ) : 
  width = length / 2 →
  area = length * width →
  area = 288 →
  length - width = 12 := by
sorry

end NUMINAMATH_CALUDE_rectangular_hall_dimensions_l1824_182421


namespace NUMINAMATH_CALUDE_raffle_ticket_sales_difference_l1824_182465

/-- Proves that the difference between Saturday's and Sunday's raffle ticket sales is 284 -/
theorem raffle_ticket_sales_difference (friday_sales : ℕ) (sunday_sales : ℕ) 
  (h1 : friday_sales = 181)
  (h2 : sunday_sales = 78) : 
  2 * friday_sales - sunday_sales = 284 := by
  sorry

#check raffle_ticket_sales_difference

end NUMINAMATH_CALUDE_raffle_ticket_sales_difference_l1824_182465


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l1824_182475

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + d * (n - 1)

theorem arithmetic_sequence_formula :
  ∀ n : ℕ, arithmetic_sequence (-1) 4 n = 4 * n - 5 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l1824_182475


namespace NUMINAMATH_CALUDE_melanie_cats_count_l1824_182433

/-- Given that Jacob has 90 cats, Annie has three times fewer cats than Jacob,
    and Melanie has twice as many cats as Annie, prove that Melanie has 60 cats. -/
theorem melanie_cats_count :
  ∀ (jacob_cats annie_cats melanie_cats : ℕ),
    jacob_cats = 90 →
    annie_cats * 3 = jacob_cats →
    melanie_cats = annie_cats * 2 →
    melanie_cats = 60 := by
  sorry

end NUMINAMATH_CALUDE_melanie_cats_count_l1824_182433


namespace NUMINAMATH_CALUDE_base_n_representation_of_b_l1824_182403

/-- Represents a number in base n -/
def BaseN (n : ℕ) (x : ℕ) : Prop :=
  ∃ (d₁ d₀ : ℕ), x = d₁ * n + d₀ ∧ d₀ < n

theorem base_n_representation_of_b
  (n : ℕ) (a b : ℕ) 
  (h_n : n > 9)
  (h_root : n^2 - a*n + b = 0)
  (h_a : BaseN n 19) :
  BaseN n 90 := by
  sorry

end NUMINAMATH_CALUDE_base_n_representation_of_b_l1824_182403


namespace NUMINAMATH_CALUDE_exists_same_color_rectangle_l1824_182479

/-- A color representation --/
inductive Color
| Black
| White

/-- A 3 × 7 grid where each cell is colored either black or white --/
def Grid := Fin 3 → Fin 7 → Color

/-- A rectangle in the grid, represented by its top-left and bottom-right corners --/
structure Rectangle where
  top_left : Fin 3 × Fin 7
  bottom_right : Fin 3 × Fin 7

/-- Check if a rectangle has all corners of the same color --/
def has_same_color_corners (g : Grid) (r : Rectangle) : Prop :=
  let (t, l) := r.top_left
  let (b, r) := r.bottom_right
  g t l = g t r ∧ g t l = g b l ∧ g t l = g b r

/-- Main theorem: There exists a rectangle with all corners of the same color --/
theorem exists_same_color_rectangle (g : Grid) : 
  ∃ r : Rectangle, has_same_color_corners g r := by sorry

end NUMINAMATH_CALUDE_exists_same_color_rectangle_l1824_182479


namespace NUMINAMATH_CALUDE_password_count_l1824_182494

/-- The number of digits in the password -/
def password_length : ℕ := 4

/-- The number of available digits (0-9 excluding 7) -/
def available_digits : ℕ := 9

/-- The total number of possible passwords without restrictions -/
def total_passwords : ℕ := available_digits ^ password_length

/-- The number of ways to choose digits for a password with all different digits -/
def ways_to_choose_digits : ℕ := Nat.choose available_digits password_length

/-- The number of ways to arrange the chosen digits -/
def ways_to_arrange_digits : ℕ := Nat.factorial password_length

/-- The number of passwords with all different digits -/
def passwords_with_different_digits : ℕ := ways_to_choose_digits * ways_to_arrange_digits

/-- The number of passwords with at least two identical digits -/
def passwords_with_identical_digits : ℕ := total_passwords - passwords_with_different_digits

theorem password_count : passwords_with_identical_digits = 3537 := by
  sorry

end NUMINAMATH_CALUDE_password_count_l1824_182494


namespace NUMINAMATH_CALUDE_negation_exists_not_eq_forall_eq_l1824_182471

theorem negation_exists_not_eq_forall_eq :
  (¬ ∃ x : ℝ, x^2 ≠ 1) ↔ (∀ x : ℝ, x^2 = 1) := by sorry

end NUMINAMATH_CALUDE_negation_exists_not_eq_forall_eq_l1824_182471


namespace NUMINAMATH_CALUDE_jogger_distance_ahead_l1824_182406

/-- Calculates the distance a jogger is ahead of a train given their speeds, the train's length, and the time it takes for the train to pass the jogger. -/
theorem jogger_distance_ahead (jogger_speed : ℝ) (train_speed : ℝ) (train_length : ℝ) (passing_time : ℝ) :
  jogger_speed = 9 * (1000 / 3600) →
  train_speed = 45 * (1000 / 3600) →
  train_length = 210 →
  passing_time = 45 →
  (train_speed - jogger_speed) * passing_time - train_length = 240 :=
by sorry

end NUMINAMATH_CALUDE_jogger_distance_ahead_l1824_182406


namespace NUMINAMATH_CALUDE_equal_expressions_l1824_182499

theorem equal_expressions (x : ℝ) (hx : x > 0) :
  (x^(x+1) + x^(x+1) = 2*x^(x+1)) ∧
  (x^(x+1) + x^(x+1) ≠ x^(2*x+2)) ∧
  (x^(x+1) + x^(x+1) ≠ (x+1)^(x+1)) ∧
  (x^(x+1) + x^(x+1) ≠ (2*x)^(x+1)) :=
by sorry

end NUMINAMATH_CALUDE_equal_expressions_l1824_182499


namespace NUMINAMATH_CALUDE_smallest_perimeter_l1824_182422

-- Define the triangle ABC and point D
structure Triangle :=
  (A B C D : ℝ × ℝ)

-- Define the conditions
def isIsosceles (t : Triangle) : Prop :=
  ‖t.A - t.B‖ = ‖t.A - t.C‖

def DOnAC (t : Triangle) : Prop :=
  ∃ k : ℝ, 0 ≤ k ∧ k ≤ 1 ∧ t.D = k • t.A + (1 - k) • t.C

def BDPerpAC (t : Triangle) : Prop :=
  (t.B.1 - t.D.1) * (t.A.1 - t.C.1) + (t.B.2 - t.D.2) * (t.A.2 - t.C.2) = 0

def ACCDEven (t : Triangle) : Prop :=
  ∃ m n : ℕ, ‖t.A - t.C‖ = 2 * m ∧ ‖t.C - t.D‖ = 2 * n

def BDSquared36 (t : Triangle) : Prop :=
  ‖t.B - t.D‖^2 = 36

def perimeter (t : Triangle) : ℝ :=
  ‖t.A - t.B‖ + ‖t.B - t.C‖ + ‖t.C - t.A‖

theorem smallest_perimeter (t : Triangle) 
  (h1 : isIsosceles t) 
  (h2 : DOnAC t) 
  (h3 : BDPerpAC t) 
  (h4 : ACCDEven t) 
  (h5 : BDSquared36 t) : 
  ∀ t' : Triangle, 
    isIsosceles t' → DOnAC t' → BDPerpAC t' → ACCDEven t' → BDSquared36 t' → 
    perimeter t ≤ perimeter t' ∧ perimeter t = 24 :=
sorry

end NUMINAMATH_CALUDE_smallest_perimeter_l1824_182422


namespace NUMINAMATH_CALUDE_two_digit_reverse_sum_square_l1824_182454

/-- A function that reverses the digits of a two-digit number -/
def reverse_digits (n : ℕ) : ℕ :=
  10 * (n % 10) + (n / 10)

/-- Predicate for a number being a perfect square -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

/-- The main theorem -/
theorem two_digit_reverse_sum_square :
  (∃ (S : Finset ℕ), S.card = 8 ∧
    ∀ n ∈ S, 10 ≤ n ∧ n < 100 ∧
    is_perfect_square (n + reverse_digits n)) ∧
  ¬∃ (S : Finset ℕ), S.card > 8 ∧
    ∀ n ∈ S, 10 ≤ n ∧ n < 100 ∧
    is_perfect_square (n + reverse_digits n) := by
  sorry


end NUMINAMATH_CALUDE_two_digit_reverse_sum_square_l1824_182454


namespace NUMINAMATH_CALUDE_random_variable_iff_preimage_singleton_l1824_182439

variable {Ω : Type*} [MeasurableSpace Ω]
variable {E : Set ℝ} (hE : Countable E)
variable (ξ : Ω → ℝ) (hξ : ∀ ω, ξ ω ∈ E)

theorem random_variable_iff_preimage_singleton :
  Measurable ξ ↔ ∀ x ∈ E, MeasurableSet {ω | ξ ω = x} := by
  sorry

end NUMINAMATH_CALUDE_random_variable_iff_preimage_singleton_l1824_182439


namespace NUMINAMATH_CALUDE_ellie_distance_after_six_steps_l1824_182481

/-- The distance Ellie walks after n steps, starting from 0 and aiming for a target 5 meters away,
    walking 1/4 of the remaining distance with each step. -/
def ellieDistance (n : ℕ) : ℚ :=
  5 * (1 - (3/4)^n)

/-- Theorem stating that after 6 steps, Ellie has walked 16835/4096 meters. -/
theorem ellie_distance_after_six_steps :
  ellieDistance 6 = 16835 / 4096 := by
  sorry


end NUMINAMATH_CALUDE_ellie_distance_after_six_steps_l1824_182481


namespace NUMINAMATH_CALUDE_complex_reciprocal_l1824_182464

theorem complex_reciprocal (i : ℂ) : i * i = -1 → (1 : ℂ) / (1 - i) = (1 : ℂ) / 2 + i / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_reciprocal_l1824_182464


namespace NUMINAMATH_CALUDE_solution_set_ln_inequality_l1824_182402

theorem solution_set_ln_inequality :
  {x : ℝ | Real.log (x - Real.exp 1) < 1} = {x | Real.exp 1 < x ∧ x < 2 * Real.exp 1} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_ln_inequality_l1824_182402


namespace NUMINAMATH_CALUDE_ratio_not_always_constant_l1824_182485

theorem ratio_not_always_constant : ∃ (f g : ℝ → ℝ), ¬(∀ x : ℝ, ∃ c : ℝ, f x = c * g x) :=
sorry

end NUMINAMATH_CALUDE_ratio_not_always_constant_l1824_182485


namespace NUMINAMATH_CALUDE_smallest_possible_b_l1824_182413

theorem smallest_possible_b : 
  ∃ (b : ℝ), ∀ (a : ℝ), 
    (2 < a ∧ a < b) → 
    (2 + a ≤ b) → 
    (1 / a + 1 / b ≤ 1 / 2) → 
    (b = 3 + Real.sqrt 5) ∧
    (∀ (b' : ℝ), 
      (∃ (a' : ℝ), 
        (2 < a' ∧ a' < b') ∧ 
        (2 + a' ≤ b') ∧ 
        (1 / a' + 1 / b' ≤ 1 / 2)) → 
      b ≤ b') :=
sorry

end NUMINAMATH_CALUDE_smallest_possible_b_l1824_182413


namespace NUMINAMATH_CALUDE_gambler_max_return_l1824_182493

/-- Represents the maximum amount a gambler can receive back after losing chips at a casino. -/
def max_amount_received (initial_value : ℕ) (chip_20_value : ℕ) (chip_100_value : ℕ) 
  (total_chips_lost : ℕ) : ℕ :=
  let chip_20_lost := (total_chips_lost + 2) / 2
  let chip_100_lost := total_chips_lost - chip_20_lost
  let value_lost := chip_20_lost * chip_20_value + chip_100_lost * chip_100_value
  initial_value - value_lost

/-- Theorem stating the maximum amount a gambler can receive back under specific conditions. -/
theorem gambler_max_return :
  max_amount_received 3000 20 100 16 = 2120 := by
  sorry

end NUMINAMATH_CALUDE_gambler_max_return_l1824_182493


namespace NUMINAMATH_CALUDE_nougat_caramel_ratio_l1824_182428

def chocolate_problem (total caramels truffles peanut_clusters nougats : ℕ) : Prop :=
  total = 50 ∧
  caramels = 3 ∧
  truffles = caramels + 6 ∧
  peanut_clusters = (64 * total) / 100 ∧
  nougats = total - caramels - truffles - peanut_clusters ∧
  nougats = 2 * caramels

theorem nougat_caramel_ratio :
  ∀ total caramels truffles peanut_clusters nougats : ℕ,
  chocolate_problem total caramels truffles peanut_clusters nougats →
  nougats = 2 * caramels :=
by
  sorry

#check nougat_caramel_ratio

end NUMINAMATH_CALUDE_nougat_caramel_ratio_l1824_182428


namespace NUMINAMATH_CALUDE_inequality_solution_l1824_182477

def solution_set (a : ℝ) : Set ℝ :=
  if a = 0 then { x | x ≥ 3 }
  else if a < 0 then { x | x ≥ 3 ∨ x ≤ 2/a }
  else if 0 < a ∧ a < 2/3 then { x | 3 ≤ x ∧ x ≤ 2/a }
  else if a = 2/3 then { x | x = 3 }
  else { x | 2/a ≤ x ∧ x ≤ 3 }

theorem inequality_solution (a : ℝ) (x : ℝ) :
  x ∈ solution_set a ↔ a * x^2 - (3*a + 2) * x + 6 ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1824_182477


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_attained_l1824_182480

theorem min_value_expression (x y : ℝ) (hx : x > 1) (hy : y > 1) :
  (x^2 / (y - 1)) + (y^2 / (x - 1)) + (x + y) ≥ 12 :=
by sorry

theorem min_value_attained (ε : ℝ) (hε : ε > 0) :
  ∃ x y : ℝ, x > 1 ∧ y > 1 ∧
  (x^2 / (y - 1)) + (y^2 / (x - 1)) + (x + y) < 12 + ε :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_attained_l1824_182480


namespace NUMINAMATH_CALUDE_concert_ticket_cost_haleys_concert_cost_l1824_182408

/-- Calculate the total amount spent on concert tickets --/
theorem concert_ticket_cost (ticket_price : ℝ) (num_tickets : ℕ) 
  (discount_rate : ℝ) (discount_threshold : ℕ) (service_fee : ℝ) : ℝ :=
  let base_cost := ticket_price * num_tickets
  let discount := if num_tickets > discount_threshold then discount_rate * base_cost else 0
  let discounted_cost := base_cost - discount
  let total_service_fee := service_fee * num_tickets
  let total_cost := discounted_cost + total_service_fee
  by
    -- Proof goes here
    sorry

/-- Haley's concert ticket purchase --/
theorem haleys_concert_cost : 
  concert_ticket_cost 4 8 0.1 5 2 = 44.8 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_concert_ticket_cost_haleys_concert_cost_l1824_182408


namespace NUMINAMATH_CALUDE_even_painted_faces_count_l1824_182491

/-- Represents a rectangular block -/
structure Block where
  length : Nat
  width : Nat
  height : Nat

/-- Counts the number of cubes with even number of painted faces in a painted block -/
def countEvenPaintedFaces (b : Block) : Nat :=
  sorry

/-- The theorem to be proved -/
theorem even_painted_faces_count (b : Block) :
  b.length = 6 → b.width = 4 → b.height = 2 →
  countEvenPaintedFaces b = 32 := by
  sorry

end NUMINAMATH_CALUDE_even_painted_faces_count_l1824_182491


namespace NUMINAMATH_CALUDE_problem_statement_l1824_182497

theorem problem_statement (x y : ℝ) (h1 : x - y = 3) (h2 : x * y = 2) :
  3 * x - 5 * x * y - 3 * y = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1824_182497


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l1824_182442

theorem geometric_sequence_ratio (a : ℕ → ℝ) :
  (∀ n, a n > 0) →  -- positive terms
  (∃ q : ℝ, q > 0 ∧ ∀ n, a (n + 1) = q * a n) →  -- geometric sequence
  (a 2 - (1/2) * a 3 = (1/2) * a 3 - a 1) →  -- arithmetic sequence condition
  (a 3 + a 4) / (a 4 + a 5) = (Real.sqrt 5 - 1) / 2 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l1824_182442


namespace NUMINAMATH_CALUDE_find_A_value_l1824_182400

theorem find_A_value (A B : Nat) (h1 : A < 10) (h2 : B < 10) 
  (h3 : 500 + 10 * A + 8 - (100 * B + 14) = 364) : A = 7 := by
  sorry

end NUMINAMATH_CALUDE_find_A_value_l1824_182400


namespace NUMINAMATH_CALUDE_paper_crane_ratio_l1824_182436

/-- Represents the number of paper cranes Alice wants in total -/
def total_cranes : ℕ := 1000

/-- Represents the number of paper cranes Alice still needs to fold -/
def remaining_cranes : ℕ := 400

/-- Represents the ratio of cranes folded by Alice's friend to remaining cranes after Alice folded half -/
def friend_to_remaining_ratio : Rat := 1 / 5

theorem paper_crane_ratio :
  let alice_folded := total_cranes / 2
  let remaining_after_alice := total_cranes - alice_folded
  let friend_folded := remaining_after_alice - remaining_cranes
  friend_folded / remaining_after_alice = friend_to_remaining_ratio := by
sorry

end NUMINAMATH_CALUDE_paper_crane_ratio_l1824_182436


namespace NUMINAMATH_CALUDE_min_values_ab_and_a_plus_2b_l1824_182441

theorem min_values_ab_and_a_plus_2b 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (h : 1/a + 2/b = 2) : 
  a * b ≥ 2 ∧ 
  a + 2*b ≥ 9/2 ∧ 
  (a + 2*b = 9/2 ↔ a = 3/2 ∧ b = 3/2) :=
by sorry

end NUMINAMATH_CALUDE_min_values_ab_and_a_plus_2b_l1824_182441


namespace NUMINAMATH_CALUDE_quadratic_two_real_roots_l1824_182457

theorem quadratic_two_real_roots (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 4*x₁ + k = 0 ∧ x₂^2 + 4*x₂ + k = 0) ↔ k ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_quadratic_two_real_roots_l1824_182457


namespace NUMINAMATH_CALUDE_opposite_of_negative_four_l1824_182438

theorem opposite_of_negative_four :
  ∀ x : ℤ, x + (-4) = 0 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_four_l1824_182438


namespace NUMINAMATH_CALUDE_B_subset_A_iff_A_disjoint_B_iff_l1824_182415

/-- Set A defined as {x | -3 < 2x-1 < 7} -/
def A : Set ℝ := {x | -3 < 2*x-1 ∧ 2*x-1 < 7}

/-- Set B defined as {x | 2a ≤ x ≤ a+3} -/
def B (a : ℝ) : Set ℝ := {x | 2*a ≤ x ∧ x ≤ a+3}

/-- Theorem stating the conditions for B to be a subset of A -/
theorem B_subset_A_iff (a : ℝ) : B a ⊆ A ↔ -1/2 < a ∧ a < 1 := by sorry

/-- Theorem stating the conditions for A and B to be disjoint -/
theorem A_disjoint_B_iff (a : ℝ) : A ∩ B a = ∅ ↔ a ≤ -4 ∨ a ≥ 2 := by sorry

end NUMINAMATH_CALUDE_B_subset_A_iff_A_disjoint_B_iff_l1824_182415


namespace NUMINAMATH_CALUDE_sector_forms_cylinder_l1824_182474

-- Define the sector
def sector_angle : ℝ := 300
def sector_radius : ℝ := 12

-- Define the cylinder
def cylinder_base_radius : ℝ := 10
def cylinder_height : ℝ := 12

-- Theorem statement
theorem sector_forms_cylinder :
  2 * Real.pi * cylinder_base_radius = (sector_angle / 360) * 2 * Real.pi * sector_radius ∧
  cylinder_height = sector_radius :=
by sorry

end NUMINAMATH_CALUDE_sector_forms_cylinder_l1824_182474


namespace NUMINAMATH_CALUDE_triangle_properties_l1824_182446

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Given conditions for the triangle --/
def TriangleConditions (t : Triangle) : Prop :=
  t.a * Real.cos t.B = 3 ∧
  t.b * Real.sin t.A = 4 ∧
  (1/2) * t.a * t.b * Real.sin t.C = 10

/-- Theorem stating the length of side a and the perimeter of the triangle --/
theorem triangle_properties (t : Triangle) (h : TriangleConditions t) :
  t.a = 5 ∧ t.a + t.b + t.c = 10 + 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l1824_182446


namespace NUMINAMATH_CALUDE_bus_tour_tickets_l1824_182418

/-- Proves that the number of regular tickets sold is 41 -/
theorem bus_tour_tickets (total_tickets : ℕ) (senior_price regular_price : ℚ) (total_sales : ℚ) 
  (h1 : total_tickets = 65)
  (h2 : senior_price = 10)
  (h3 : regular_price = 15)
  (h4 : total_sales = 855) :
  ∃ (senior_tickets regular_tickets : ℕ),
    senior_tickets + regular_tickets = total_tickets ∧
    senior_price * senior_tickets + regular_price * regular_tickets = total_sales ∧
    regular_tickets = 41 := by
  sorry

end NUMINAMATH_CALUDE_bus_tour_tickets_l1824_182418


namespace NUMINAMATH_CALUDE_next_perfect_cube_l1824_182404

theorem next_perfect_cube (x : ℕ) (h : ∃ k : ℕ, x = k^2) :
  ∃ y : ℕ, y > x ∧ (∃ m : ℕ, y = m^3) ∧
  y = x * (x : ℝ).sqrt + 3 * x + 3 * (x : ℝ).sqrt + 1 :=
sorry

end NUMINAMATH_CALUDE_next_perfect_cube_l1824_182404


namespace NUMINAMATH_CALUDE_initial_number_of_persons_l1824_182450

theorem initial_number_of_persons (avg_weight_increase : ℝ) 
  (old_person_weight new_person_weight : ℝ) :
  avg_weight_increase = 2.5 →
  old_person_weight = 75 →
  new_person_weight = 95 →
  (new_person_weight - old_person_weight) / avg_weight_increase = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_initial_number_of_persons_l1824_182450


namespace NUMINAMATH_CALUDE_limit_equals_derivative_l1824_182448

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem limit_equals_derivative
  (h1 : Differentiable ℝ f)  -- f is differentiable
  (h2 : deriv f 1 = 1) :     -- f'(1) = 1
  Filter.Tendsto
    (fun Δx => (f (1 - Δx) - f 1) / (-Δx))
    (nhds 0)
    (nhds 1) :=
by
  sorry

end NUMINAMATH_CALUDE_limit_equals_derivative_l1824_182448


namespace NUMINAMATH_CALUDE_brothers_initial_money_l1824_182459

theorem brothers_initial_money 
  (michael_initial : ℝ) 
  (brother_final : ℝ) 
  (candy_cost : ℝ) :
  michael_initial = 42 →
  brother_final = 35 →
  candy_cost = 3 →
  ∃ (brother_initial : ℝ),
    brother_initial + michael_initial / 2 - candy_cost = brother_final ∧
    brother_initial = 17 := by
  sorry

end NUMINAMATH_CALUDE_brothers_initial_money_l1824_182459


namespace NUMINAMATH_CALUDE_negate_negative_twenty_l1824_182482

theorem negate_negative_twenty : -(-20) = 20 := by
  sorry

end NUMINAMATH_CALUDE_negate_negative_twenty_l1824_182482


namespace NUMINAMATH_CALUDE_roger_trays_second_table_l1824_182453

/-- Represents the number of trays Roger can carry in one trip -/
def trays_per_trip : ℕ := 4

/-- Represents the number of trips Roger made -/
def num_trips : ℕ := 3

/-- Represents the number of trays Roger picked up from the first table -/
def trays_first_table : ℕ := 10

/-- Calculates the number of trays Roger picked up from the second table -/
def trays_second_table : ℕ := trays_per_trip * num_trips - trays_first_table

theorem roger_trays_second_table : trays_second_table = 2 := by
  sorry

end NUMINAMATH_CALUDE_roger_trays_second_table_l1824_182453


namespace NUMINAMATH_CALUDE_pizza_topping_cost_l1824_182411

/-- Represents the cost of a pizza with toppings -/
def pizza_cost (base_cost : ℚ) (first_topping_cost : ℚ) (next_two_toppings_cost : ℚ) 
  (num_slices : ℕ) (cost_per_slice : ℚ) (num_toppings : ℕ) : Prop :=
  let total_cost := cost_per_slice * num_slices
  let known_cost := base_cost + first_topping_cost + 2 * next_two_toppings_cost
  let remaining_toppings_cost := total_cost - known_cost
  let num_remaining_toppings := num_toppings - 3
  remaining_toppings_cost / num_remaining_toppings = 0.5

theorem pizza_topping_cost : 
  pizza_cost 10 2 1 8 2 7 :=
by sorry

end NUMINAMATH_CALUDE_pizza_topping_cost_l1824_182411


namespace NUMINAMATH_CALUDE_no_solution_exists_l1824_182451

theorem no_solution_exists : ¬∃ (a b c : ℝ), 
  (a > 0 ∧ b > 0 ∧ c > 0) ∧
  (∃ (n : ℕ), 
    ((b + c - a) / a = n) ∧
    ((a + c - b) / b = n) ∧
    ((a + b - c) / c = n)) ∧
  ((a + b) * (b + c) * (a + c)) / (a * b * c) = 12 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l1824_182451


namespace NUMINAMATH_CALUDE_boys_girls_difference_l1824_182490

theorem boys_girls_difference (x y : ℕ) (a b : ℚ) : 
  x > y → 
  x * a + y * b = x * b + y * a - 1 → 
  x = y + 1 := by
sorry

end NUMINAMATH_CALUDE_boys_girls_difference_l1824_182490


namespace NUMINAMATH_CALUDE_investment_ratio_is_seven_to_five_l1824_182484

/-- Represents the investment and profit information for two partners -/
structure PartnerInvestment where
  profit_ratio : Rat
  p_investment_time : ℕ
  q_investment_time : ℕ

/-- Calculates the investment ratio given the profit ratio and investment times -/
def investment_ratio (info : PartnerInvestment) : Rat :=
  (info.profit_ratio * info.q_investment_time) / info.p_investment_time

/-- Theorem stating that given the specified conditions, the investment ratio is 7:5 -/
theorem investment_ratio_is_seven_to_five (info : PartnerInvestment) 
  (h1 : info.profit_ratio = 7 / 10)
  (h2 : info.p_investment_time = 2)
  (h3 : info.q_investment_time = 4) :
  investment_ratio info = 7 / 5 := by
  sorry

#eval investment_ratio { profit_ratio := 7 / 10, p_investment_time := 2, q_investment_time := 4 }

end NUMINAMATH_CALUDE_investment_ratio_is_seven_to_five_l1824_182484


namespace NUMINAMATH_CALUDE_subtraction_is_perfect_square_l1824_182476

def A : ℕ := (10^1001 - 1) / 9
def B : ℕ := (10^2002 - 1) / 9
def C : ℕ := 2 * A

theorem subtraction_is_perfect_square : B - C = (3 * A)^2 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_is_perfect_square_l1824_182476


namespace NUMINAMATH_CALUDE_bus_stop_walking_time_bus_stop_walking_time_proof_l1824_182452

/-- The time to walk to the bus stop at the usual speed, given that walking at 4/5 of the usual speed results in arriving 7 minutes later than normal, is 28 minutes. -/
theorem bus_stop_walking_time : ℝ → Prop :=
  fun T : ℝ =>
    (4 / 5 * T + 7 = T) → T = 28

/-- Proof of the bus_stop_walking_time theorem -/
theorem bus_stop_walking_time_proof : ∃ T : ℝ, bus_stop_walking_time T :=
  sorry

end NUMINAMATH_CALUDE_bus_stop_walking_time_bus_stop_walking_time_proof_l1824_182452


namespace NUMINAMATH_CALUDE_binomial_expansion_ratio_l1824_182466

theorem binomial_expansion_ratio (a₀ a₁ a₂ a₃ a₄ a₅ : ℚ) :
  (∀ x, (2 - x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  (a₀ + a₂ + a₄) / (a₁ + a₃) = -61 / 60 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_ratio_l1824_182466


namespace NUMINAMATH_CALUDE_no_real_solution_l1824_182495

theorem no_real_solution :
  ¬∃ (a b c d : ℝ), 
    a^3 + c^3 = 2 ∧
    a^2 * b + c^2 * d = 0 ∧
    b^3 + d^3 = 1 ∧
    a * b^2 + c * d^2 = -6 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solution_l1824_182495


namespace NUMINAMATH_CALUDE_senior_junior_ratio_l1824_182456

theorem senior_junior_ratio (j k : ℕ) (hj : j > 0) (hk : k > 0)
  (h_junior_contestants : (3 * j) / 5 = (j * 3) / 5)
  (h_senior_contestants : k / 5 = (k * 1) / 5)
  (h_equal_contestants : (3 * j) / 5 = k / 5) :
  k = 3 * j :=
sorry

end NUMINAMATH_CALUDE_senior_junior_ratio_l1824_182456


namespace NUMINAMATH_CALUDE_prime_count_200_to_220_l1824_182473

theorem prime_count_200_to_220 : ∃! p, Nat.Prime p ∧ 200 < p ∧ p < 220 := by
  sorry

end NUMINAMATH_CALUDE_prime_count_200_to_220_l1824_182473


namespace NUMINAMATH_CALUDE_largest_number_l1824_182419

theorem largest_number (a b c d e : ℝ) : 
  a = 24680 + 1/1357 →
  b = 24680 - 1/1357 →
  c = 24680 * (1/1357) →
  d = 24680 / (1/1357) →
  e = 24680.1357 →
  d > a ∧ d > b ∧ d > c ∧ d > e :=
by sorry

end NUMINAMATH_CALUDE_largest_number_l1824_182419


namespace NUMINAMATH_CALUDE_heptagon_diagonals_l1824_182405

/-- The number of diagonals in a convex n-gon --/
def numDiagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A convex heptagon has 14 diagonals --/
theorem heptagon_diagonals : numDiagonals 7 = 14 := by
  sorry

end NUMINAMATH_CALUDE_heptagon_diagonals_l1824_182405


namespace NUMINAMATH_CALUDE_dodecagon_interior_angles_sum_l1824_182489

/-- The sum of interior angles of a polygon with n sides is (n - 2) * 180° --/
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- A dodecagon is a polygon with 12 sides --/
def is_dodecagon (n : ℕ) : Prop := n = 12

theorem dodecagon_interior_angles_sum :
  ∀ n : ℕ, is_dodecagon n → sum_interior_angles n = 1800 :=
by sorry

end NUMINAMATH_CALUDE_dodecagon_interior_angles_sum_l1824_182489


namespace NUMINAMATH_CALUDE_meaningful_fraction_l1824_182414

theorem meaningful_fraction (x : ℝ) : 
  (∃ y : ℝ, y = 3 / (x - 2)) ↔ x ≠ 2 := by sorry

end NUMINAMATH_CALUDE_meaningful_fraction_l1824_182414


namespace NUMINAMATH_CALUDE_regular_polygon_perimeter_l1824_182423

/-- A regular polygon with side length 8 units and exterior angle 45 degrees has a perimeter of 64 units. -/
theorem regular_polygon_perimeter (n : ℕ) (side_length : ℝ) (exterior_angle : ℝ) :
  n > 0 ∧ 
  side_length = 8 ∧ 
  exterior_angle = 45 ∧ 
  exterior_angle = 360 / n →
  n * side_length = 64 := by
sorry


end NUMINAMATH_CALUDE_regular_polygon_perimeter_l1824_182423


namespace NUMINAMATH_CALUDE_seating_arrangements_count_l1824_182487

/-- The number of seats in a row -/
def total_seats : ℕ := 22

/-- The number of candidates to be seated -/
def num_candidates : ℕ := 4

/-- The minimum number of empty seats required between any two candidates -/
def min_empty_seats : ℕ := 5

/-- Calculate the number of ways to arrange the candidates -/
def seating_arrangements : ℕ := sorry

/-- Theorem stating that the number of seating arrangements is 840 -/
theorem seating_arrangements_count : seating_arrangements = 840 := by sorry

end NUMINAMATH_CALUDE_seating_arrangements_count_l1824_182487


namespace NUMINAMATH_CALUDE_expression_equals_25_l1824_182461

theorem expression_equals_25 : 
  (5^1010)^2 - (5^1008)^2 / (5^1009)^2 - (5^1007)^2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_25_l1824_182461


namespace NUMINAMATH_CALUDE_candy_canes_count_l1824_182478

-- Define the problem parameters
def num_kids : ℕ := 3
def beanie_babies_per_stocking : ℕ := 2
def books_per_stocking : ℕ := 1
def total_stuffers : ℕ := 21

-- Define the function to calculate candy canes per stocking
def candy_canes_per_stocking : ℕ :=
  let non_candy_items_per_stocking := beanie_babies_per_stocking + books_per_stocking
  let total_non_candy_items := non_candy_items_per_stocking * num_kids
  let total_candy_canes := total_stuffers - total_non_candy_items
  total_candy_canes / num_kids

-- Theorem statement
theorem candy_canes_count : candy_canes_per_stocking = 4 := by
  sorry

end NUMINAMATH_CALUDE_candy_canes_count_l1824_182478


namespace NUMINAMATH_CALUDE_yellow_highlighters_l1824_182462

theorem yellow_highlighters (total : ℕ) (pink : ℕ) (blue : ℕ) 
  (h1 : total = 15) 
  (h2 : pink = 3) 
  (h3 : blue = 5) : 
  total - pink - blue = 7 := by
  sorry

end NUMINAMATH_CALUDE_yellow_highlighters_l1824_182462


namespace NUMINAMATH_CALUDE_borrowed_sum_l1824_182412

/-- Given a principal P borrowed at 5% simple interest per annum,
    if after 5 years the interest is Rs. 750 less than P,
    then P must be Rs. 1000. -/
theorem borrowed_sum (P : ℝ) : 
  (P * 0.05 * 5 = P - 750) → P = 1000 := by
  sorry

end NUMINAMATH_CALUDE_borrowed_sum_l1824_182412


namespace NUMINAMATH_CALUDE_uniform_cost_is_355_l1824_182432

/-- Calculates the total cost of uniforms for a student given the costs of individual items -/
def uniform_cost (pants_cost shirt_cost tie_cost socks_cost : ℚ) : ℚ :=
  5 * (pants_cost + shirt_cost + tie_cost + socks_cost)

/-- Proves that the total cost of uniforms for a student is $355 -/
theorem uniform_cost_is_355 :
  uniform_cost 20 40 8 3 = 355 := by
  sorry

#eval uniform_cost 20 40 8 3

end NUMINAMATH_CALUDE_uniform_cost_is_355_l1824_182432


namespace NUMINAMATH_CALUDE_pillsbury_sugar_needed_l1824_182496

/-- Chef Pillsbury's recipe ratios -/
structure RecipeRatios where
  eggs_to_flour : ℚ
  milk_to_eggs : ℚ
  sugar_to_milk : ℚ

/-- Calculate the number of tablespoons of sugar needed for a given amount of flour -/
def sugar_needed (ratios : RecipeRatios) (flour_cups : ℚ) : ℚ :=
  let eggs := flour_cups * ratios.eggs_to_flour
  let milk := eggs * ratios.milk_to_eggs
  milk * ratios.sugar_to_milk

/-- Theorem: For 24 cups of flour, Chef Pillsbury needs 90 tablespoons of sugar -/
theorem pillsbury_sugar_needed :
  let ratios : RecipeRatios := {
    eggs_to_flour := 7 / 2,
    milk_to_eggs := 5 / 14,
    sugar_to_milk := 3 / 1
  }
  sugar_needed ratios 24 = 90 := by
  sorry

end NUMINAMATH_CALUDE_pillsbury_sugar_needed_l1824_182496


namespace NUMINAMATH_CALUDE_intersection_M_N_l1824_182460

-- Define the sets M and N
def M : Set ℝ := {x | x / (x - 1) > 0}
def N : Set ℝ := {x | ∃ y, y * y = x}

-- State the theorem
theorem intersection_M_N : M ∩ N = {x | x > 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1824_182460


namespace NUMINAMATH_CALUDE_moon_arrangements_l1824_182435

/-- The number of distinct arrangements of letters in a word -/
def distinctArrangements (totalLetters : ℕ) (repeatedLetters : List ℕ) : ℕ :=
  Nat.factorial totalLetters / (repeatedLetters.map Nat.factorial).prod

/-- The word "MOON" has 4 letters with one letter repeated twice -/
def moonWord : (ℕ × List ℕ) := (4, [2])

theorem moon_arrangements :
  distinctArrangements moonWord.fst moonWord.snd = 12 := by
  sorry

end NUMINAMATH_CALUDE_moon_arrangements_l1824_182435


namespace NUMINAMATH_CALUDE_triangle_sine_sides_l1824_182426

theorem triangle_sine_sides (a b c : Real) 
  (h1 : 0 < a ∧ 0 < b ∧ 0 < c)
  (h2 : a + b > c ∧ b + c > a ∧ c + a > b)
  (h3 : a + b + c ≤ 2 * Real.pi) :
  Real.sin a + Real.sin b > Real.sin c ∧ 
  Real.sin b + Real.sin c > Real.sin a ∧ 
  Real.sin c + Real.sin a > Real.sin b := by
sorry

end NUMINAMATH_CALUDE_triangle_sine_sides_l1824_182426


namespace NUMINAMATH_CALUDE_kylie_daisies_l1824_182420

def initial_daisies : ℕ := 5
def sister_daisies : ℕ := 9

def total_daisies : ℕ := initial_daisies + sister_daisies

def remaining_daisies : ℕ := total_daisies / 2

theorem kylie_daisies : remaining_daisies = 7 := by sorry

end NUMINAMATH_CALUDE_kylie_daisies_l1824_182420


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1824_182469

/-- A hyperbola with foci on the x-axis -/
structure Hyperbola where
  /-- The ratio of y to x for the asymptotes -/
  asymptote_slope : ℝ
  /-- The hyperbola has foci on the x-axis -/
  foci_on_x_axis : Bool

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ := sorry

/-- Theorem stating the eccentricity of a specific hyperbola -/
theorem hyperbola_eccentricity (h : Hyperbola) 
  (h_slope : h.asymptote_slope = 2/3) 
  (h_foci : h.foci_on_x_axis = true) : 
  eccentricity h = Real.sqrt 13 / 3 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1824_182469


namespace NUMINAMATH_CALUDE_aquarium_fish_count_l1824_182416

theorem aquarium_fish_count (stingrays sharks eels : ℕ) : 
  stingrays = 28 →
  sharks = 2 * stingrays →
  eels = 3 * stingrays →
  stingrays + sharks + eels = 168 := by
  sorry

end NUMINAMATH_CALUDE_aquarium_fish_count_l1824_182416


namespace NUMINAMATH_CALUDE_sqrt_product_quotient_equals_twelve_l1824_182434

theorem sqrt_product_quotient_equals_twelve :
  Real.sqrt 27 * Real.sqrt (8/3) / Real.sqrt (1/2) = 12 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_quotient_equals_twelve_l1824_182434


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l1824_182458

theorem min_reciprocal_sum (x y z : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0) (h_sum : x + y + z = 2) :
  (1 / x + 1 / y + 1 / z) ≥ 4.5 ∧ ∃ x y z, x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = 2 ∧ 1 / x + 1 / y + 1 / z = 4.5 :=
sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l1824_182458


namespace NUMINAMATH_CALUDE_a_100_equals_116_l1824_182447

/-- Sequence of positive integers not divisible by 7 -/
def a : ℕ → ℕ :=
  λ n => (n + (n - 1) / 6) + 1

theorem a_100_equals_116 : a 100 = 116 := by
  sorry

end NUMINAMATH_CALUDE_a_100_equals_116_l1824_182447


namespace NUMINAMATH_CALUDE_inequality_proof_l1824_182488

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  Real.sqrt ((a^2 + b^2 + c^2) / 3) ≥ (a + b + c) / 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1824_182488


namespace NUMINAMATH_CALUDE_isosceles_with_120_exterior_is_equilateral_equal_exterior_angles_is_equilateral_l1824_182440

-- Define an isosceles triangle
structure IsoscelesTriangle :=
  (a b c : ℝ)
  (ab_eq_ac : a = c)

-- Define an equilateral triangle
structure EquilateralTriangle :=
  (a b c : ℝ)
  (all_sides_equal : a = b ∧ b = c)

-- Define exterior angle
def exterior_angle (interior_angle : ℝ) : ℝ := 180 - interior_angle

theorem isosceles_with_120_exterior_is_equilateral 
  (t : IsoscelesTriangle) 
  (h : ∃ (angle : ℝ), exterior_angle angle = 120) : 
  EquilateralTriangle :=
sorry

theorem equal_exterior_angles_is_equilateral 
  (t : IsoscelesTriangle) 
  (h : ∃ (angle : ℝ), 
    exterior_angle angle = exterior_angle (exterior_angle angle) ∧ 
    exterior_angle angle = exterior_angle (exterior_angle (exterior_angle angle))) : 
  EquilateralTriangle :=
sorry

end NUMINAMATH_CALUDE_isosceles_with_120_exterior_is_equilateral_equal_exterior_angles_is_equilateral_l1824_182440


namespace NUMINAMATH_CALUDE_constant_killing_time_l1824_182443

/-- The time it takes for lions to kill deers -/
def killing_time (n : ℕ) : ℝ :=
  14

/-- Given conditions -/
axiom condition_14 : killing_time 14 = 14
axiom condition_100 : killing_time 100 = 14

/-- Theorem: For any positive number of lions, it takes 14 minutes to kill the same number of deers -/
theorem constant_killing_time (n : ℕ) (h : n > 0) : killing_time n = 14 := by
  sorry

end NUMINAMATH_CALUDE_constant_killing_time_l1824_182443


namespace NUMINAMATH_CALUDE_fraction_simplification_l1824_182492

theorem fraction_simplification :
  ((3^12)^2 - (3^10)^2) / ((3^11)^2 - (3^9)^2) = 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1824_182492


namespace NUMINAMATH_CALUDE_power_division_l1824_182437

theorem power_division (a : ℝ) : a^6 / a^3 = a^3 := by
  sorry

end NUMINAMATH_CALUDE_power_division_l1824_182437


namespace NUMINAMATH_CALUDE_concave_integral_inequality_l1824_182445

open Set Function MeasureTheory

variable {m : ℕ}
variable (P : Set (EuclideanSpace ℝ (Fin m)))
variable (f : EuclideanSpace ℝ (Fin m) → ℝ)
variable (ξ : EuclideanSpace ℝ (Fin m))

theorem concave_integral_inequality
  (h_nonempty : Set.Nonempty P)
  (h_compact : IsCompact P)
  (h_convex : Convex ℝ P)
  (h_concave : ConcaveOn ℝ P f)
  (h_nonneg : ∀ x ∈ P, 0 ≤ f x) :
  ∫ x in P, ⟪ξ, x⟫_ℝ * f x ≤ 
    ((m + 1 : ℝ) / (m + 2 : ℝ) * ⨆ (x : EuclideanSpace ℝ (Fin m)) (h : x ∈ P), ⟪ξ, x⟫_ℝ + 
     (1 : ℝ) / (m + 2 : ℝ) * ⨅ (x : EuclideanSpace ℝ (Fin m)) (h : x ∈ P), ⟪ξ, x⟫_ℝ) * 
    ∫ x in P, f x :=
sorry

end NUMINAMATH_CALUDE_concave_integral_inequality_l1824_182445


namespace NUMINAMATH_CALUDE_peter_pizza_fraction_l1824_182417

/-- Given a pizza with 16 slices, calculate the fraction eaten by Peter -/
theorem peter_pizza_fraction :
  let total_slices : ℕ := 16
  let whole_slices_eaten : ℕ := 2
  let shared_slice : ℚ := 1/2
  (whole_slices_eaten : ℚ) / total_slices + shared_slice / total_slices = 5/32 := by
  sorry

end NUMINAMATH_CALUDE_peter_pizza_fraction_l1824_182417


namespace NUMINAMATH_CALUDE_min_students_is_fifteen_l1824_182468

/-- Represents the attendance for each day of the week -/
structure WeeklyAttendance where
  monday : Nat
  tuesday : Nat
  wednesday : Nat
  thursday : Nat
  friday : Nat

/-- Calculates the minimum number of students given weekly attendance -/
def minStudents (attendance : WeeklyAttendance) : Nat :=
  max (attendance.monday + attendance.wednesday + attendance.friday)
      (attendance.tuesday + attendance.thursday)

/-- Theorem: The minimum number of students who visited the library during the week is 15 -/
theorem min_students_is_fifteen (attendance : WeeklyAttendance)
  (h1 : attendance.monday = 5)
  (h2 : attendance.tuesday = 6)
  (h3 : attendance.wednesday = 4)
  (h4 : attendance.thursday = 8)
  (h5 : attendance.friday = 7) :
  minStudents attendance = 15 := by
  sorry

#eval minStudents ⟨5, 6, 4, 8, 7⟩

end NUMINAMATH_CALUDE_min_students_is_fifteen_l1824_182468


namespace NUMINAMATH_CALUDE_line_opposite_sides_range_l1824_182430

/-- Given that the points (1, 1) and (0, 1) are on opposite sides of the line 3x - 2y + a = 0,
    the range of values for a is (-1, 2). -/
theorem line_opposite_sides_range (a : ℝ) : 
  (∃ (x y : ℝ), (x = 1 ∧ y = 1) ∨ (x = 0 ∧ y = 1)) →
  ((3 * 1 - 2 * 1 + a) * (3 * 0 - 2 * 1 + a) < 0) →
  a ∈ Set.Ioo (-1 : ℝ) 2 :=
by sorry

end NUMINAMATH_CALUDE_line_opposite_sides_range_l1824_182430


namespace NUMINAMATH_CALUDE_sum_of_i_powers_l1824_182483

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem sum_of_i_powers : i^14 + i^19 + i^24 + i^29 + i^34 = -1 := by
  sorry

-- Define the property of i
axiom i_squared : i^2 = -1

end NUMINAMATH_CALUDE_sum_of_i_powers_l1824_182483


namespace NUMINAMATH_CALUDE_magpie_porridge_l1824_182449

/-- Represents the amount of porridge each chick received -/
structure ChickPorridge where
  x1 : ℝ
  x2 : ℝ
  x3 : ℝ
  x4 : ℝ
  x5 : ℝ
  x6 : ℝ

/-- The conditions of porridge distribution -/
def porridge_conditions (p : ChickPorridge) : Prop :=
  p.x3 = p.x1 + p.x2 ∧
  p.x4 = p.x2 + p.x3 ∧
  p.x5 = p.x3 + p.x4 ∧
  p.x6 = p.x4 + p.x5 ∧
  p.x5 = 10

/-- The total amount of porridge cooked by the magpie -/
def total_porridge (p : ChickPorridge) : ℝ :=
  p.x1 + p.x2 + p.x3 + p.x4 + p.x5 + p.x6

/-- Theorem stating that the total amount of porridge is 40 grams -/
theorem magpie_porridge (p : ChickPorridge) :
  porridge_conditions p → total_porridge p = 40 := by
  sorry

end NUMINAMATH_CALUDE_magpie_porridge_l1824_182449


namespace NUMINAMATH_CALUDE_lemonade_ratio_l1824_182407

/-- Given that 30 lemons make 25 gallons of lemonade, prove that 12 lemons make 10 gallons -/
theorem lemonade_ratio (lemons : ℕ) (gallons : ℕ) 
  (h : (30 : ℚ) / 25 = lemons / gallons) (h10 : gallons = 10) : lemons = 12 := by
  sorry

end NUMINAMATH_CALUDE_lemonade_ratio_l1824_182407


namespace NUMINAMATH_CALUDE_divisible_by_37_l1824_182455

theorem divisible_by_37 (n d : ℕ) (h : d ≤ 9) : 
  ∃ k : ℕ, (d * (10^(3*n) - 1) / 9) = 37 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_37_l1824_182455


namespace NUMINAMATH_CALUDE_polynomial_remainder_theorem_l1824_182444

theorem polynomial_remainder_theorem (f : ℝ → ℝ) (a b c l m n p q r : ℝ) :
  a ≠ 0 → b ≠ 0 → c ≠ 0 →
  (∃ Q₁ : ℝ → ℝ, ∀ x, f x = (x - a) * (x - b) * Q₁ x + (p * x + l)) →
  (∃ Q₂ : ℝ → ℝ, ∀ x, f x = (x - b) * (x - c) * Q₂ x + (q * x + m)) →
  (∃ Q₃ : ℝ → ℝ, ∀ x, f x = (x - c) * (x - a) * Q₃ x + (r * x + n)) →
  l * (1 / a - 1 / b) + m * (1 / b - 1 / c) + n * (1 / c - 1 / a) = 0 := by
sorry

end NUMINAMATH_CALUDE_polynomial_remainder_theorem_l1824_182444


namespace NUMINAMATH_CALUDE_combined_tax_rate_l1824_182463

/-- Represents the combined tax rate problem for Mork, Mindy, and Julie -/
theorem combined_tax_rate 
  (mork_rate : ℚ) 
  (mindy_rate : ℚ) 
  (julie_rate : ℚ) 
  (mork_income : ℚ) 
  (mindy_income : ℚ) 
  (julie_income : ℚ) :
  mork_rate = 45/100 →
  mindy_rate = 25/100 →
  julie_rate = 35/100 →
  mindy_income = 4 * mork_income →
  julie_income = 2 * mork_income →
  julie_income = (1/2) * mindy_income →
  (mork_rate * mork_income + mindy_rate * mindy_income + julie_rate * julie_income) / 
  (mork_income + mindy_income + julie_income) = 215/700 :=
by sorry

end NUMINAMATH_CALUDE_combined_tax_rate_l1824_182463


namespace NUMINAMATH_CALUDE_tangent_curves_n_value_l1824_182431

-- Define the ellipse equation
def ellipse (x y : ℝ) : Prop := 4 * x^2 + y^2 = 4

-- Define the hyperbola equation
def hyperbola (x y n : ℝ) : Prop := x^2 - n * (y - 1)^2 = 1

-- Define the tangency condition
def are_tangent (n : ℝ) : Prop :=
  ∃ x y : ℝ, ellipse x y ∧ hyperbola x y n ∧
  ∀ x' y' : ℝ, ellipse x' y' ∧ hyperbola x' y' n → (x', y') = (x, y)

-- State the theorem
theorem tangent_curves_n_value :
  ∀ n : ℝ, are_tangent n → n = 3 :=
by sorry

end NUMINAMATH_CALUDE_tangent_curves_n_value_l1824_182431


namespace NUMINAMATH_CALUDE_lisa_caffeine_consumption_l1824_182409

/-- Represents the number of each beverage Lisa consumed --/
structure BeverageConsumption where
  coffee : ℕ
  soda : ℕ
  tea : ℕ
  energyDrink : ℕ

/-- Represents the caffeine content of each beverage in milligrams --/
structure CaffeineContent where
  coffee : ℕ
  soda : ℕ
  tea : ℕ
  energyDrink : ℕ

def totalCaffeine (consumption : BeverageConsumption) (content : CaffeineContent) : ℕ :=
  consumption.coffee * content.coffee +
  consumption.soda * content.soda +
  consumption.tea * content.tea +
  consumption.energyDrink * content.energyDrink

theorem lisa_caffeine_consumption
  (consumption : BeverageConsumption)
  (content : CaffeineContent)
  (daily_goal : ℕ)
  (h_consumption : consumption = { coffee := 3, soda := 1, tea := 2, energyDrink := 1 })
  (h_content : content = { coffee := 95, soda := 45, tea := 55, energyDrink := 120 })
  (h_goal : daily_goal = 200) :
  totalCaffeine consumption content = 560 ∧ totalCaffeine consumption content - daily_goal = 360 := by
  sorry


end NUMINAMATH_CALUDE_lisa_caffeine_consumption_l1824_182409


namespace NUMINAMATH_CALUDE_inequality_theorem_l1824_182425

theorem inequality_theorem (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, x * (deriv f x) > -f x) (a b : ℝ) (hab : a > b) : 
  a * f a > b * f b := by
  sorry

end NUMINAMATH_CALUDE_inequality_theorem_l1824_182425


namespace NUMINAMATH_CALUDE_test_has_hundred_questions_l1824_182470

/-- Represents a test with a specific scoring system -/
structure Test where
  total_questions : ℕ
  correct_responses : ℕ
  incorrect_responses : ℕ
  score : ℤ
  score_calculation : score = correct_responses - 2 * incorrect_responses
  total_questions_sum : total_questions = correct_responses + incorrect_responses

/-- Theorem stating that given the conditions, the test has 100 questions -/
theorem test_has_hundred_questions (t : Test) 
  (h1 : t.score = 79) 
  (h2 : t.correct_responses = 93) : 
  t.total_questions = 100 := by
  sorry


end NUMINAMATH_CALUDE_test_has_hundred_questions_l1824_182470


namespace NUMINAMATH_CALUDE_merchant_pricing_strategy_l1824_182401

/-- Represents the pricing strategy of a merchant -/
structure MerchantPricing where
  list_price : ℝ
  purchase_discount : ℝ
  marked_price : ℝ
  selling_discount : ℝ
  profit_margin : ℝ

/-- Calculates the purchase price based on the list price and purchase discount -/
def purchase_price (m : MerchantPricing) : ℝ :=
  m.list_price * (1 - m.purchase_discount)

/-- Calculates the selling price based on the marked price and selling discount -/
def selling_price (m : MerchantPricing) : ℝ :=
  m.marked_price * (1 - m.selling_discount)

/-- Calculates the profit based on the selling price and purchase price -/
def profit (m : MerchantPricing) : ℝ :=
  selling_price m - purchase_price m

/-- Theorem: The merchant must mark the goods at 125% of the list price -/
theorem merchant_pricing_strategy (m : MerchantPricing) 
  (h1 : m.purchase_discount = 0.3)
  (h2 : m.selling_discount = 0.2)
  (h3 : m.profit_margin = 0.3)
  (h4 : profit m = m.profit_margin * selling_price m) :
  m.marked_price = 1.25 * m.list_price := by
  sorry

end NUMINAMATH_CALUDE_merchant_pricing_strategy_l1824_182401


namespace NUMINAMATH_CALUDE_tim_sleep_total_l1824_182498

theorem tim_sleep_total (sleep_first_two_days sleep_next_two_days : ℕ) 
  (h1 : sleep_first_two_days = 6 * 2)
  (h2 : sleep_next_two_days = 10 * 2) :
  sleep_first_two_days + sleep_next_two_days = 32 := by
  sorry

end NUMINAMATH_CALUDE_tim_sleep_total_l1824_182498


namespace NUMINAMATH_CALUDE_inscribed_cube_properties_l1824_182424

/-- A cube inscribed in a hemisphere -/
structure InscribedCube (R : ℝ) where
  -- The edge length of the cube
  a : ℝ
  -- The distance from the center of the hemisphere base to a vertex of the square face
  r : ℝ
  -- Four vertices of the cube are on the surface of the hemisphere
  vertices_on_surface : a ^ 2 + r ^ 2 = R ^ 2
  -- Four vertices of the cube are on the circular boundary of the hemisphere's base
  vertices_on_base : r = a * (Real.sqrt 2) / 2

/-- The edge length and distance properties of a cube inscribed in a hemisphere -/
theorem inscribed_cube_properties (R : ℝ) (h : R > 0) :
  ∃ (cube : InscribedCube R),
    cube.a = R * Real.sqrt (2/3) ∧
    cube.r = R / Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_cube_properties_l1824_182424


namespace NUMINAMATH_CALUDE_three_four_five_pythagorean_triple_l1824_182410

/-- Definition of a Pythagorean triple -/
def isPythagoreanTriple (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

/-- Theorem: (3,4,5) is a Pythagorean triple -/
theorem three_four_five_pythagorean_triple :
  isPythagoreanTriple 3 4 5 := by
  sorry

end NUMINAMATH_CALUDE_three_four_five_pythagorean_triple_l1824_182410


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1824_182429

-- Define set M
def M : Set ℝ := {y | ∃ x, y = x^2 + 2*x - 3}

-- Define set N
def N : Set ℝ := {x | |x - 2| ≤ 3}

-- Theorem statement
theorem intersection_of_M_and_N :
  M ∩ N = {y | -1 ≤ y ∧ y ≤ 5} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1824_182429


namespace NUMINAMATH_CALUDE_top_coat_drying_time_l1824_182486

/-- Given nail polish drying times, prove the top coat drying time -/
theorem top_coat_drying_time 
  (base_coat_time : ℕ) 
  (color_coat_time : ℕ) 
  (num_color_coats : ℕ) 
  (total_drying_time : ℕ) 
  (h1 : base_coat_time = 2)
  (h2 : color_coat_time = 3)
  (h3 : num_color_coats = 2)
  (h4 : total_drying_time = 13) :
  total_drying_time - (base_coat_time + num_color_coats * color_coat_time) = 5 := by
  sorry

end NUMINAMATH_CALUDE_top_coat_drying_time_l1824_182486

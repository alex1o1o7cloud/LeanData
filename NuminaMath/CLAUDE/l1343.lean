import Mathlib

namespace NUMINAMATH_CALUDE_prime_divides_sum_of_squares_l1343_134310

theorem prime_divides_sum_of_squares (p a b : ℤ) : 
  Prime p → p % 4 = 3 → (a^2 + b^2) % p = 0 → p ∣ a ∧ p ∣ b := by
  sorry

end NUMINAMATH_CALUDE_prime_divides_sum_of_squares_l1343_134310


namespace NUMINAMATH_CALUDE_fishes_from_ontario_erie_l1343_134360

/-- The number of fishes taken from Lake Huron and Michigan -/
def huron_michigan : ℕ := 30

/-- The number of fishes taken from Lake Superior -/
def superior : ℕ := 44

/-- The total number of fishes brought home -/
def total : ℕ := 97

/-- The number of fishes taken from Lake Ontario and Erie -/
def ontario_erie : ℕ := total - (huron_michigan + superior)

theorem fishes_from_ontario_erie : ontario_erie = 23 := by
  sorry

end NUMINAMATH_CALUDE_fishes_from_ontario_erie_l1343_134360


namespace NUMINAMATH_CALUDE_watch_sale_gain_percentage_l1343_134326

/-- Prove the gain percentage for a watch sale --/
theorem watch_sale_gain_percentage 
  (cost_price : ℝ) 
  (initial_loss_percentage : ℝ) 
  (price_increase : ℝ) : 
  cost_price = 875 → 
  initial_loss_percentage = 12 → 
  price_increase = 140 → 
  let initial_selling_price := cost_price * (1 - initial_loss_percentage / 100)
  let new_selling_price := initial_selling_price + price_increase
  let gain := new_selling_price - cost_price
  let gain_percentage := (gain / cost_price) * 100
  gain_percentage = 4 := by
sorry


end NUMINAMATH_CALUDE_watch_sale_gain_percentage_l1343_134326


namespace NUMINAMATH_CALUDE_window_area_ratio_l1343_134332

theorem window_area_ratio (AB : ℝ) (h1 : AB = 40) : 
  let AD : ℝ := 3 / 2 * AB
  let rectangle_area : ℝ := AD * AB
  let semicircle_area : ℝ := π * (AB / 2) ^ 2
  rectangle_area / semicircle_area = 6 / π := by sorry

end NUMINAMATH_CALUDE_window_area_ratio_l1343_134332


namespace NUMINAMATH_CALUDE_unit_square_quadrilateral_bounds_l1343_134315

/-- A quadrilateral formed by selecting one point on each side of a unit square -/
structure UnitSquareQuadrilateral where
  a : ℝ  -- Length of side a
  b : ℝ  -- Length of side b
  c : ℝ  -- Length of side c
  d : ℝ  -- Length of side d
  ha : 0 ≤ a ∧ a ≤ 1  -- a is between 0 and 1
  hb : 0 ≤ b ∧ b ≤ 1  -- b is between 0 and 1
  hc : 0 ≤ c ∧ c ≤ 1  -- c is between 0 and 1
  hd : 0 ≤ d ∧ d ≤ 1  -- d is between 0 and 1

theorem unit_square_quadrilateral_bounds (q : UnitSquareQuadrilateral) :
  2 ≤ q.a^2 + q.b^2 + q.c^2 + q.d^2 ∧ q.a^2 + q.b^2 + q.c^2 + q.d^2 ≤ 4 ∧
  2 * Real.sqrt 2 ≤ q.a + q.b + q.c + q.d ∧ q.a + q.b + q.c + q.d ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_unit_square_quadrilateral_bounds_l1343_134315


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1343_134321

/-- Given a hyperbola with a = 5 and c = 7, prove its standard equation. -/
theorem hyperbola_equation (a c : ℝ) (ha : a = 5) (hc : c = 7) :
  ∃ (x y : ℝ → ℝ), 
    (∀ t, x t ^ 2 / 25 - y t ^ 2 / 24 = 1) ∨
    (∀ t, y t ^ 2 / 25 - x t ^ 2 / 24 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1343_134321


namespace NUMINAMATH_CALUDE_last_duck_bread_pieces_l1343_134370

theorem last_duck_bread_pieces (total : ℕ) (left : ℕ) (first_duck : ℕ) (second_duck : ℕ) :
  total = 100 →
  left = 30 →
  first_duck = total / 2 →
  second_duck = 13 →
  total - left - first_duck - second_duck = 7 :=
by sorry

end NUMINAMATH_CALUDE_last_duck_bread_pieces_l1343_134370


namespace NUMINAMATH_CALUDE_janet_waiting_time_l1343_134301

/-- Proves that the waiting time for Janet is 3 hours given the conditions of the problem -/
theorem janet_waiting_time (lake_width : ℝ) (speedboat_speed : ℝ) (sailboat_speed : ℝ)
  (h1 : lake_width = 60)
  (h2 : speedboat_speed = 30)
  (h3 : sailboat_speed = 12) :
  sailboat_speed * (lake_width / speedboat_speed) - lake_width = 3 * speedboat_speed := by
  sorry


end NUMINAMATH_CALUDE_janet_waiting_time_l1343_134301


namespace NUMINAMATH_CALUDE_range_of_m_l1343_134350

theorem range_of_m : 
  (∀ x : ℝ, |2009 * x + 1| ≥ |m - 1| - 2) ↔ m ∈ Set.Icc (-1 : ℝ) 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l1343_134350


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1343_134323

theorem quadratic_equation_solution : 
  let f : ℝ → ℝ := fun x ↦ 5 * x^2 - 2 * x
  ∃ x₁ x₂ : ℝ, x₁ = 0 ∧ x₂ = 2/5 ∧ 
    (∀ x : ℝ, f x = 0 ↔ x = x₁ ∨ x = x₂) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1343_134323


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1343_134380

theorem quadratic_equation_roots (m : ℝ) : 
  let f : ℝ → ℝ := λ x => x^2 - (2*m - 1)*x + m^2
  ∃ x₁ x₂ : ℝ, (∀ x : ℝ, f x = 0 ↔ x = x₁ ∨ x = x₂) ∧ 
  (x₁ ≠ x₂) ∧
  ((x₁ + 1) * (x₂ + 1) = 3) →
  m = -3 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1343_134380


namespace NUMINAMATH_CALUDE_tan_sum_identity_l1343_134305

theorem tan_sum_identity (x y z : Real) 
  (hx : x = 20 * π / 180)
  (hy : y = 30 * π / 180)
  (hz : z = 40 * π / 180)
  (h1 : Real.tan (60 * π / 180) = Real.sqrt 3)
  (h2 : Real.tan (30 * π / 180) = 1 / Real.sqrt 3) :
  Real.tan x * Real.tan y + Real.tan y * Real.tan z + Real.tan z * Real.tan x = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_identity_l1343_134305


namespace NUMINAMATH_CALUDE_expression_value_l1343_134345

theorem expression_value (b : ℝ) (h : b = 1/3) : 
  (3 * b⁻¹ + b⁻¹ / 3) / b = 30 := by sorry

end NUMINAMATH_CALUDE_expression_value_l1343_134345


namespace NUMINAMATH_CALUDE_sesame_seed_weight_scientific_notation_l1343_134384

theorem sesame_seed_weight_scientific_notation : 
  ∃ (a : ℝ) (n : ℤ), 0.00000201 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 2.01 ∧ n = -6 := by
  sorry

end NUMINAMATH_CALUDE_sesame_seed_weight_scientific_notation_l1343_134384


namespace NUMINAMATH_CALUDE_domain_of_composite_function_l1343_134334

-- Define the function f with domain (1,3)
def f : Set ℝ := Set.Ioo 1 3

-- Define the composite function g(x) = f(2x-1)
def g (x : ℝ) : ℝ := 2 * x - 1

-- Theorem statement
theorem domain_of_composite_function :
  {x : ℝ | g x ∈ f} = Set.Ioo 1 2 := by sorry

end NUMINAMATH_CALUDE_domain_of_composite_function_l1343_134334


namespace NUMINAMATH_CALUDE_systematic_sampling_probability_l1343_134372

theorem systematic_sampling_probability (total_students : ℕ) (sample_size : ℕ) 
  (h1 : total_students = 121) (h2 : sample_size = 20) :
  (sample_size : ℚ) / total_students = 20 / 121 :=
by sorry

end NUMINAMATH_CALUDE_systematic_sampling_probability_l1343_134372


namespace NUMINAMATH_CALUDE_updated_mean_after_decrement_l1343_134331

def original_mean : ℝ := 250
def num_observations : ℕ := 100
def decrement : ℝ := 20

theorem updated_mean_after_decrement :
  (original_mean * num_observations - decrement * num_observations) / num_observations = 230 := by
  sorry

end NUMINAMATH_CALUDE_updated_mean_after_decrement_l1343_134331


namespace NUMINAMATH_CALUDE_tangent_line_minimum_value_l1343_134354

/-- Given a function f(x) = ax² + b/x where a > 0 and b > 0, 
    and its tangent line at x = 1 passes through (3/2, 1/2),
    prove that the minimum value of 1/a + 1/b is 9 -/
theorem tangent_line_minimum_value (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let f := fun x => a * x^2 + b / x
  let f' := fun x => 2 * a * x - b / x^2
  let tangent_slope := f' 1
  let tangent_point := (1, f 1)
  (tangent_slope * (3/2 - 1) = 1/2 - f 1) →
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → 1/a' + 1/b' ≥ 9) ∧ 
  (∃ a' b' : ℝ, a' > 0 ∧ b' > 0 ∧ 1/a' + 1/b' = 9) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_minimum_value_l1343_134354


namespace NUMINAMATH_CALUDE_solution_exists_l1343_134302

theorem solution_exists (x y b : ℝ) : 
  (4 * x + 2 * y = b) →
  (3 * x + 7 * y = 3 * b) →
  (x = -1) →
  b = -22 :=
by sorry

end NUMINAMATH_CALUDE_solution_exists_l1343_134302


namespace NUMINAMATH_CALUDE_integer_roots_of_polynomial_l1343_134303

def polynomial (x a₂ a₁ : ℤ) : ℤ := x^3 + a₂*x^2 + a₁*x + 24

def possible_roots : Set ℤ := {-24, -12, -8, -6, -4, -3, -2, -1, 1, 2, 3, 4, 6, 8, 12, 24}

theorem integer_roots_of_polynomial (a₂ a₁ : ℤ) :
  {x : ℤ | polynomial x a₂ a₁ = 0} ⊆ possible_roots :=
by sorry

end NUMINAMATH_CALUDE_integer_roots_of_polynomial_l1343_134303


namespace NUMINAMATH_CALUDE_jakes_initial_money_l1343_134353

theorem jakes_initial_money (M : ℝ) : 
  (M - 2800 - (M - 2800) / 2) * 3 / 4 = 825 → M = 5000 := by
  sorry

end NUMINAMATH_CALUDE_jakes_initial_money_l1343_134353


namespace NUMINAMATH_CALUDE_seven_balls_two_boxes_l1343_134328

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  n + 1

theorem seven_balls_two_boxes :
  distribute_balls 7 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_seven_balls_two_boxes_l1343_134328


namespace NUMINAMATH_CALUDE_cube_side_length_l1343_134395

theorem cube_side_length (C R T : ℝ) (h1 : C = 36.50) (h2 : R = 16) (h3 : T = 876) :
  ∃ L : ℝ, L = 8 ∧ T = (6 * L^2) * (C / R) :=
sorry

end NUMINAMATH_CALUDE_cube_side_length_l1343_134395


namespace NUMINAMATH_CALUDE_determinant_transformation_l1343_134365

theorem determinant_transformation (p q r s : ℝ) :
  Matrix.det !![p, q; r, s] = 3 →
  Matrix.det !![p, 5*p + 4*q; r, 5*r + 4*s] = 12 := by
sorry

end NUMINAMATH_CALUDE_determinant_transformation_l1343_134365


namespace NUMINAMATH_CALUDE_alloy_mixture_specific_alloy_mixture_l1343_134314

/-- Given two alloys with different chromium percentages, prove the amount of the second alloy
    needed to create a new alloy with a specific chromium percentage. -/
theorem alloy_mixture (first_alloy_chromium_percent : ℝ) 
                      (second_alloy_chromium_percent : ℝ)
                      (new_alloy_chromium_percent : ℝ)
                      (first_alloy_amount : ℝ) : ℝ :=
  let second_alloy_amount := 
    (new_alloy_chromium_percent * first_alloy_amount - first_alloy_chromium_percent * first_alloy_amount) /
    (second_alloy_chromium_percent - new_alloy_chromium_percent)
  second_alloy_amount

/-- Prove that 35 kg of the second alloy is needed to create the new alloy with 8.6% chromium. -/
theorem specific_alloy_mixture : 
  alloy_mixture 0.10 0.08 0.086 15 = 35 := by
  sorry

end NUMINAMATH_CALUDE_alloy_mixture_specific_alloy_mixture_l1343_134314


namespace NUMINAMATH_CALUDE_other_number_from_hcf_lcm_l1343_134342

theorem other_number_from_hcf_lcm (A B : ℕ+) : 
  Nat.gcd A B = 12 → 
  Nat.lcm A B = 396 → 
  A = 24 → 
  B = 198 := by
sorry

end NUMINAMATH_CALUDE_other_number_from_hcf_lcm_l1343_134342


namespace NUMINAMATH_CALUDE_root_value_theorem_l1343_134369

theorem root_value_theorem (m : ℝ) (h : m^2 - 2*m - 3 = 0) : 2026 - m^2 + 2*m = 2023 := by
  sorry

end NUMINAMATH_CALUDE_root_value_theorem_l1343_134369


namespace NUMINAMATH_CALUDE_line_separate_from_circle_l1343_134304

/-- The line x₀x + y₀y - a² = 0 is separate from the circle x² + y² = a² (a > 0),
    given that point M(x₀, y₀) is inside the circle and different from its center. -/
theorem line_separate_from_circle
  (a : ℝ) (x₀ y₀ : ℝ) 
  (h_a_pos : a > 0)
  (h_inside : x₀^2 + y₀^2 < a^2)
  (h_not_center : x₀ ≠ 0 ∨ y₀ ≠ 0) :
  let d := a^2 / Real.sqrt (x₀^2 + y₀^2)
  d > a :=
by sorry

end NUMINAMATH_CALUDE_line_separate_from_circle_l1343_134304


namespace NUMINAMATH_CALUDE_stratified_sampling_l1343_134375

theorem stratified_sampling (total_capacity : ℕ) (sample_capacity : ℕ) 
  (ratio_A ratio_B ratio_C : ℕ) :
  total_capacity = 56 →
  sample_capacity = 14 →
  ratio_A = 1 →
  ratio_B = 2 →
  ratio_C = 4 →
  ∃ (sample_A sample_B sample_C : ℕ),
    sample_A = 2 ∧
    sample_B = 4 ∧
    sample_C = 8 ∧
    sample_A + sample_B + sample_C = sample_capacity ∧
    sample_A * (ratio_A + ratio_B + ratio_C) = sample_capacity * ratio_A ∧
    sample_B * (ratio_A + ratio_B + ratio_C) = sample_capacity * ratio_B ∧
    sample_C * (ratio_A + ratio_B + ratio_C) = sample_capacity * ratio_C :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_l1343_134375


namespace NUMINAMATH_CALUDE_evaluate_expression_l1343_134397

theorem evaluate_expression : (-3)^7 / 3^5 + 2^5 - 7^2 = -26 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1343_134397


namespace NUMINAMATH_CALUDE_special_equation_solution_l1343_134357

theorem special_equation_solution :
  ∃ x : ℝ, 9 - 8 / 7 * x + 10 = 13.285714285714286 ∧ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_special_equation_solution_l1343_134357


namespace NUMINAMATH_CALUDE_house_application_proof_l1343_134324

/-- The number of houses available -/
def num_houses : ℕ := 3

/-- The number of persons applying for houses -/
def num_persons : ℕ := 3

/-- The probability that all persons apply for the same house -/
def prob_same_house : ℚ := 1 / 9

/-- The number of houses each person applies for -/
def houses_per_person : ℕ := 1

theorem house_application_proof :
  (prob_same_house = (houses_per_person : ℚ)^2 / num_houses^2) →
  houses_per_person = 1 :=
by sorry

end NUMINAMATH_CALUDE_house_application_proof_l1343_134324


namespace NUMINAMATH_CALUDE_dual_polyhedra_equal_radii_l1343_134386

/-- Represents a regular polyhedron -/
structure RegularPolyhedron where
  inscribed_radius : ℝ
  circumscribed_radius : ℝ
  face_circumscribed_radius : ℝ

/-- Represents a pair of dual regular polyhedra -/
structure DualRegularPolyhedra where
  original : RegularPolyhedron
  dual : RegularPolyhedron

/-- Theorem: For dual regular polyhedra with equal inscribed sphere radii,
    their circumscribed sphere radii and face circumscribed circle radii are equal -/
theorem dual_polyhedra_equal_radii (p : DualRegularPolyhedra) 
    (h : p.original.inscribed_radius = p.dual.inscribed_radius) : 
    p.original.circumscribed_radius = p.dual.circumscribed_radius ∧ 
    p.original.face_circumscribed_radius = p.dual.face_circumscribed_radius := by
  sorry


end NUMINAMATH_CALUDE_dual_polyhedra_equal_radii_l1343_134386


namespace NUMINAMATH_CALUDE_parallelogram_perimeter_plus_area_l1343_134340

/-- A parallelogram with integer coordinates -/
structure Parallelogram where
  v1 : ℤ × ℤ
  v2 : ℤ × ℤ
  v3 : ℤ × ℤ
  v4 : ℤ × ℤ

/-- The perimeter of a parallelogram -/
def perimeter (p : Parallelogram) : ℝ :=
  sorry

/-- The area of a parallelogram -/
def area (p : Parallelogram) : ℝ :=
  sorry

theorem parallelogram_perimeter_plus_area :
  let p : Parallelogram := ⟨(1,1), (6,3), (9,3), (4,1)⟩
  perimeter p + area p = 2 * Real.sqrt 29 + 12 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_perimeter_plus_area_l1343_134340


namespace NUMINAMATH_CALUDE_mike_seeds_left_l1343_134374

/-- The number of seeds Mike has left after feeding the birds -/
def seeds_left (total : ℕ) (left : ℕ) (right_multiplier : ℕ) (late : ℕ) : ℕ :=
  total - (left + right_multiplier * left + late)

/-- Theorem stating that Mike has 30 seeds left -/
theorem mike_seeds_left :
  seeds_left 120 20 2 30 = 30 := by
  sorry

end NUMINAMATH_CALUDE_mike_seeds_left_l1343_134374


namespace NUMINAMATH_CALUDE_solution_set_is_singleton_l1343_134389

def solution_set : Set (ℝ × ℝ) := {(x, y) | 2*x + y = 0 ∧ x - y + 3 = 0}

theorem solution_set_is_singleton : solution_set = {(-1, 2)} := by sorry

end NUMINAMATH_CALUDE_solution_set_is_singleton_l1343_134389


namespace NUMINAMATH_CALUDE_problem_statement_l1343_134311

theorem problem_statement (x y : ℝ) (h : |x - 5| + (x - y - 1)^2 = 0) : 
  (x - y)^2023 = 1 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l1343_134311


namespace NUMINAMATH_CALUDE_percentage_sum_proof_l1343_134366

theorem percentage_sum_proof : 
  ∃ (x : ℝ), x * 400 + 0.45 * 250 = 224.5 ∧ x = 0.28 := by
  sorry

end NUMINAMATH_CALUDE_percentage_sum_proof_l1343_134366


namespace NUMINAMATH_CALUDE_min_moves_to_win_l1343_134347

/-- Represents a box containing balls -/
structure Box where
  white : Nat
  black : Nat

/-- Represents the state of the game -/
structure GameState where
  round : Box
  square : Box

/-- Checks if two boxes have identical contents -/
def boxesIdentical (b1 b2 : Box) : Bool :=
  b1.white = b2.white ∧ b1.black = b2.black

/-- Defines a single move in the game -/
inductive Move
  | discard : Box → Move
  | transfer : Box → Box → Move

/-- Applies a move to the game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  sorry

/-- Checks if the game is in a winning state -/
def isWinningState (state : GameState) : Bool :=
  boxesIdentical state.round state.square

/-- The initial state of the game -/
def initialState : GameState :=
  { round := { white := 3, black := 10 }
  , square := { white := 0, black := 8 } }

/-- Theorem: The minimum number of moves to reach a winning state is 17 -/
theorem min_moves_to_win :
  ∃ (moves : List Move),
    moves.length = 17 ∧
    isWinningState (moves.foldl applyMove initialState) ∧
    ∀ (shorter_moves : List Move),
      shorter_moves.length < 17 →
      ¬isWinningState (shorter_moves.foldl applyMove initialState) :=
  sorry

end NUMINAMATH_CALUDE_min_moves_to_win_l1343_134347


namespace NUMINAMATH_CALUDE_girls_without_pets_girls_without_pets_proof_l1343_134359

theorem girls_without_pets (total_students : ℕ) (boys_fraction : ℚ) 
  (girls_with_dogs : ℚ) (girls_with_cats : ℚ) : ℕ :=
  let girls_fraction := 1 - boys_fraction
  let total_girls := (total_students : ℚ) * girls_fraction
  let girls_without_pets_fraction := 1 - girls_with_dogs - girls_with_cats
  let girls_without_pets := total_girls * girls_without_pets_fraction
  8

theorem girls_without_pets_proof :
  girls_without_pets 30 (1/3) (2/5) (1/5) = 8 := by
  sorry

end NUMINAMATH_CALUDE_girls_without_pets_girls_without_pets_proof_l1343_134359


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1343_134358

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_property
  (a : ℕ → ℝ)
  (h_geom : is_geometric_sequence a)
  (h_positive : ∀ n, a n > 0)
  (h_a3 : a 3 = 4)
  (h_a5 : a 5 = 16) :
  a 3^2 + 2 * a 2 * a 6 + a 3 * a 7 = 400 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1343_134358


namespace NUMINAMATH_CALUDE_probability_red_from_B_probability_red_from_B_is_correct_l1343_134398

/-- Represents the number of red balls in Box A -/
def red_balls_A : ℕ := 5

/-- Represents the number of white balls in Box A -/
def white_balls_A : ℕ := 2

/-- Represents the number of red balls in Box B -/
def red_balls_B : ℕ := 4

/-- Represents the number of white balls in Box B -/
def white_balls_B : ℕ := 3

/-- Represents the total number of balls in Box A -/
def total_balls_A : ℕ := red_balls_A + white_balls_A

/-- Represents the total number of balls in Box B -/
def total_balls_B : ℕ := red_balls_B + white_balls_B

/-- The probability of drawing a red ball from Box B after the process -/
theorem probability_red_from_B : ℚ :=
  33 / 56

theorem probability_red_from_B_is_correct :
  probability_red_from_B = 33 / 56 := by sorry

end NUMINAMATH_CALUDE_probability_red_from_B_probability_red_from_B_is_correct_l1343_134398


namespace NUMINAMATH_CALUDE_total_campers_rowing_l1343_134313

theorem total_campers_rowing (morning_campers afternoon_campers : ℕ) 
  (h1 : morning_campers = 53)
  (h2 : afternoon_campers = 7) :
  morning_campers + afternoon_campers = 60 := by
  sorry

end NUMINAMATH_CALUDE_total_campers_rowing_l1343_134313


namespace NUMINAMATH_CALUDE_eel_length_ratio_l1343_134316

theorem eel_length_ratio (total_length : ℝ) (jenna_length : ℝ) :
  total_length = 64 →
  jenna_length = 16 →
  (jenna_length / (total_length - jenna_length) = 1 / 3) :=
by
  sorry

end NUMINAMATH_CALUDE_eel_length_ratio_l1343_134316


namespace NUMINAMATH_CALUDE_arithmetic_sequence_third_term_l1343_134393

/-- An arithmetic sequence is a sequence where the difference between
    consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_third_term
  (a : ℕ → ℝ)
  (h_arithmetic : ArithmeticSequence a)
  (h_20th : a 20 = 15)
  (h_21st : a 21 = 18) :
  a 3 = -36 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_third_term_l1343_134393


namespace NUMINAMATH_CALUDE_transformed_stddev_l1343_134343

variable {n : ℕ}
variable (a : Fin n → ℝ)
variable (S : ℝ)

def variance (x : Fin n → ℝ) : ℝ := sorry

def stdDev (x : Fin n → ℝ) : ℝ := sorry

theorem transformed_stddev 
  (h : variance a = S^2) : 
  stdDev (fun i => 2 * a i - 3) = 2 * S := by sorry

end NUMINAMATH_CALUDE_transformed_stddev_l1343_134343


namespace NUMINAMATH_CALUDE_increasing_power_function_m_l1343_134317

/-- A power function f(x) = (m^2 - 3)x^(m+1) is increasing on (0, +∞) -/
def is_increasing_power_function (m : ℝ) : Prop :=
  ∀ x : ℝ, x > 0 → Monotone (fun x => (m^2 - 3) * x^(m+1))

/-- The value of m for which the power function is increasing -/
theorem increasing_power_function_m : 
  ∃ m : ℝ, is_increasing_power_function m ∧ m = 2 :=
sorry

end NUMINAMATH_CALUDE_increasing_power_function_m_l1343_134317


namespace NUMINAMATH_CALUDE_equal_roots_implies_c_value_l1343_134383

-- Define the quadratic equation
def quadratic (x c : ℝ) : ℝ := x^2 + 6*x - c

-- Define the discriminant of the quadratic equation
def discriminant (c : ℝ) : ℝ := 6^2 - 4*(1)*(-c)

-- Theorem statement
theorem equal_roots_implies_c_value :
  (∃ x : ℝ, quadratic x c = 0 ∧ 
    ∀ y : ℝ, quadratic y c = 0 → y = x) →
  c = -9 := by sorry

end NUMINAMATH_CALUDE_equal_roots_implies_c_value_l1343_134383


namespace NUMINAMATH_CALUDE_scientific_notation_proof_l1343_134306

/-- Proves that 470,000,000 is equal to 4.7 × 10^8 in scientific notation -/
theorem scientific_notation_proof :
  (470000000 : ℝ) = 4.7 * (10 ^ 8) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_proof_l1343_134306


namespace NUMINAMATH_CALUDE_trees_in_garden_l1343_134382

/-- The number of trees in a yard with given length and spacing -/
def number_of_trees (yard_length : ℕ) (tree_spacing : ℕ) : ℕ :=
  (yard_length / tree_spacing) + 1

/-- Theorem: There are 26 trees in a 300-meter yard with 12-meter spacing -/
theorem trees_in_garden : number_of_trees 300 12 = 26 := by
  sorry

end NUMINAMATH_CALUDE_trees_in_garden_l1343_134382


namespace NUMINAMATH_CALUDE_sum_largest_smallest_prime_factors_546_l1343_134338

def largest_prime_factor (n : ℕ) : ℕ := sorry

def smallest_prime_factor (n : ℕ) : ℕ := sorry

theorem sum_largest_smallest_prime_factors_546 :
  largest_prime_factor 546 + smallest_prime_factor 546 = 15 := by sorry

end NUMINAMATH_CALUDE_sum_largest_smallest_prime_factors_546_l1343_134338


namespace NUMINAMATH_CALUDE_negation_equivalence_l1343_134363

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x > 0 ∧ x^3 - x + 1 > 0) ↔ (∀ x : ℝ, x > 0 → x^3 - x + 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1343_134363


namespace NUMINAMATH_CALUDE_altitude_and_angle_bisector_equations_l1343_134322

/-- Triangle ABC with vertices A(1,-1), B(-1,3), C(3,0) -/
structure Triangle :=
  (A : ℝ × ℝ)
  (B : ℝ × ℝ)
  (C : ℝ × ℝ)

/-- The given triangle -/
def ABC : Triangle :=
  { A := (1, -1),
    B := (-1, 3),
    C := (3, 0) }

/-- Altitude from A to BC -/
def altitude (t : Triangle) : ℝ × ℝ → Prop :=
  λ p => 4 * p.1 - 3 * p.2 - 7 = 0

/-- Angle bisector of ∠BAC -/
def angle_bisector (t : Triangle) : ℝ × ℝ → Prop :=
  λ p => p.1 - p.2 - 4 = 0

/-- Main theorem -/
theorem altitude_and_angle_bisector_equations :
  (∀ p, altitude ABC p ↔ 4 * p.1 - 3 * p.2 - 7 = 0) ∧
  (∀ p, angle_bisector ABC p ↔ p.1 - p.2 - 4 = 0) := by
  sorry

end NUMINAMATH_CALUDE_altitude_and_angle_bisector_equations_l1343_134322


namespace NUMINAMATH_CALUDE_quadratic_minimum_l1343_134344

theorem quadratic_minimum (x : ℝ) : 
  ∃ (min_x : ℝ), ∀ (y : ℝ), x^2 + 8*x + 7 ≥ min_x^2 + 8*min_x + 7 ∧ min_x = -4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l1343_134344


namespace NUMINAMATH_CALUDE_missed_number_l1343_134341

theorem missed_number (n : ℕ) (incorrect_sum correct_sum missed_number : ℕ) :
  n > 0 →
  incorrect_sum = 575 →
  correct_sum = n * (n + 1) / 2 →
  correct_sum = 595 →
  incorrect_sum + missed_number = correct_sum →
  missed_number = 20 := by
  sorry

end NUMINAMATH_CALUDE_missed_number_l1343_134341


namespace NUMINAMATH_CALUDE_bethany_portraits_l1343_134309

/-- The number of portraits Bethany saw at the museum -/
def num_portraits : ℕ := 16

/-- The number of still lifes Bethany saw at the museum -/
def num_still_lifes : ℕ := 4 * num_portraits

/-- The total number of paintings Bethany saw at the museum -/
def total_paintings : ℕ := 80

theorem bethany_portraits :
  num_portraits + num_still_lifes = total_paintings ∧
  num_still_lifes = 4 * num_portraits →
  num_portraits = 16 := by sorry

end NUMINAMATH_CALUDE_bethany_portraits_l1343_134309


namespace NUMINAMATH_CALUDE_base_b_not_divisible_by_five_l1343_134348

theorem base_b_not_divisible_by_five (b : ℤ) (h : b ∈ ({3, 5, 7, 10, 12} : Set ℤ)) : 
  ¬ (5 ∣ ((b - 1)^2)) := by
sorry

end NUMINAMATH_CALUDE_base_b_not_divisible_by_five_l1343_134348


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l1343_134381

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |x - 1| ≤ 2} = Set.Icc (-1) 3 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l1343_134381


namespace NUMINAMATH_CALUDE_inequality_addition_l1343_134308

theorem inequality_addition (a b c : ℝ) (h1 : a > b) (h2 : b > c) : a + b > b + c := by
  sorry

end NUMINAMATH_CALUDE_inequality_addition_l1343_134308


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l1343_134373

/-- Given two lines in a plane, this theorem states that if one line passes through 
    a specific point and is parallel to the other line, then it has a specific equation. -/
theorem parallel_line_through_point (x y : ℝ) : 
  (x - 2*y + 3 = 0) →  -- Given line
  (2 = x ∧ 0 = y) →    -- Point (2, 0)
  (2*y - x + 2 = 0) →  -- Equation to prove
  ∃ (m b : ℝ), (y = m*x + b ∧ 2*y - x + 2 = 0) ∧ 
               (∃ (c : ℝ), x - 2*y + c = 0) :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l1343_134373


namespace NUMINAMATH_CALUDE_sum_of_roots_is_eight_l1343_134330

/-- A function f is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- A function g is odd if g(-x) = -g(x) for all x -/
def IsOdd (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = -g x

/-- The number of real roots of an equation -/
noncomputable def numRealRoots (f : ℝ → ℝ) : ℕ :=
  sorry

/-- Theorem: For an even function f and an odd function g, 
    the sum of the number of real roots of f(f(x)) = 0, f(g(x)) = 0, 
    g(g(x)) = 0, and g(f(x)) = 0 is equal to 8 -/
theorem sum_of_roots_is_eight 
    (f : ℝ → ℝ) (g : ℝ → ℝ) 
    (hf : IsEven f) (hg : IsOdd g) : 
  numRealRoots (λ x => f (f x)) + 
  numRealRoots (λ x => f (g x)) + 
  numRealRoots (λ x => g (g x)) + 
  numRealRoots (λ x => g (f x)) = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_is_eight_l1343_134330


namespace NUMINAMATH_CALUDE_birthday_month_l1343_134379

def is_valid_day (d : ℕ) : Prop := 1 ≤ d ∧ d ≤ 31

def is_valid_month (m : ℕ) : Prop := 1 ≤ m ∧ m ≤ 12

theorem birthday_month (d m : ℕ) (h1 : is_valid_day d) (h2 : is_valid_month m) 
  (h3 : d * m = 248) : m = 8 := by
  sorry

end NUMINAMATH_CALUDE_birthday_month_l1343_134379


namespace NUMINAMATH_CALUDE_heat_bulls_difference_l1343_134392

/-- The number of games won by the Chicago Bulls -/
def bulls_games : ℕ := 70

/-- The total number of games won by both the Chicago Bulls and the Miami Heat -/
def total_games : ℕ := 145

/-- The number of games won by the Miami Heat -/
def heat_games : ℕ := total_games - bulls_games

/-- Theorem stating the difference in games won between the Miami Heat and the Chicago Bulls -/
theorem heat_bulls_difference : heat_games - bulls_games = 5 := by
  sorry

end NUMINAMATH_CALUDE_heat_bulls_difference_l1343_134392


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_common_difference_l1343_134325

/-- An arithmetic sequence with the given properties has a common difference of -1/5 -/
theorem arithmetic_geometric_sequence_common_difference :
  ∀ (a : ℕ → ℚ) (d : ℚ),
  d ≠ 0 →
  (∀ n, a (n + 1) = a n + d) →
  a 1 = 1 →
  (a 2) * (a 5) = (a 4)^2 →
  d = -1/5 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_common_difference_l1343_134325


namespace NUMINAMATH_CALUDE_april_initial_roses_l1343_134351

/-- The number of roses April started with, given the price per rose, 
    the number of roses left, and the total earnings from selling roses. -/
def initial_roses (price_per_rose : ℕ) (roses_left : ℕ) (total_earnings : ℕ) : ℕ :=
  (total_earnings / price_per_rose) + roses_left

/-- Theorem stating that April started with 13 roses -/
theorem april_initial_roses : 
  initial_roses 4 4 36 = 13 := by
  sorry

end NUMINAMATH_CALUDE_april_initial_roses_l1343_134351


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l1343_134378

/-- Calculate the interest rate given principal, final amount, and time -/
theorem interest_rate_calculation (P A t : ℝ) (h1 : P = 1200) (h2 : A = 1344) (h3 : t = 2.4) :
  (A - P) / (P * t) = 0.05 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l1343_134378


namespace NUMINAMATH_CALUDE_lower_bound_of_set_A_l1343_134396

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def set_A : Set ℕ := {n : ℕ | is_prime n ∧ n ≥ 17 ∧ n ≤ 36}

theorem lower_bound_of_set_A :
  (∃ (max_A : ℕ), max_A ∈ set_A ∧ ∀ n ∈ set_A, n ≤ max_A) ∧
  (∃ (min_A : ℕ), min_A ∈ set_A ∧ ∀ n ∈ set_A, n ≥ min_A) ∧
  (∃ (max_A min_A : ℕ), max_A - min_A = 14) →
  (∃ (min_A : ℕ), min_A = 17 ∧ min_A ∈ set_A ∧ ∀ n ∈ set_A, n ≥ min_A) :=
by sorry

end NUMINAMATH_CALUDE_lower_bound_of_set_A_l1343_134396


namespace NUMINAMATH_CALUDE_unique_value_sum_l1343_134319

/-- Given that {a, b, c} = {0, 1, 2} and exactly one of (a ≠ 2), (b = 2), (c ≠ 0) is true,
    prove that a + 2b + 5c = 7 -/
theorem unique_value_sum (a b c : ℤ) : 
  ({a, b, c} : Set ℤ) = {0, 1, 2} →
  ((a ≠ 2) ∨ (b = 2) ∨ (c ≠ 0)) ∧
  (¬((a ≠ 2) ∧ (b = 2)) ∧ ¬((a ≠ 2) ∧ (c ≠ 0)) ∧ ¬((b = 2) ∧ (c ≠ 0))) →
  a + 2*b + 5*c = 7 := by
  sorry

end NUMINAMATH_CALUDE_unique_value_sum_l1343_134319


namespace NUMINAMATH_CALUDE_remainder_2519_div_6_l1343_134390

theorem remainder_2519_div_6 : 2519 % 6 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_2519_div_6_l1343_134390


namespace NUMINAMATH_CALUDE_z_in_third_quadrant_l1343_134335

/-- The complex number z defined as i(-2 + i) -/
def z : ℂ := Complex.I * (Complex.mk (-2) 1)

/-- Predicate to check if a complex number is in the third quadrant -/
def in_third_quadrant (w : ℂ) : Prop :=
  w.re < 0 ∧ w.im < 0

/-- Theorem stating that z is in the third quadrant -/
theorem z_in_third_quadrant : in_third_quadrant z := by sorry

end NUMINAMATH_CALUDE_z_in_third_quadrant_l1343_134335


namespace NUMINAMATH_CALUDE_factor_sum_l1343_134329

theorem factor_sum (R S : ℝ) : 
  (∃ b c : ℝ, (X^2 + 3*X + 7) * (X^2 + b*X + c) = X^4 + R*X^2 + S) → 
  R + S = 54 := by
sorry

end NUMINAMATH_CALUDE_factor_sum_l1343_134329


namespace NUMINAMATH_CALUDE_g_of_3_l1343_134307

def g (x : ℝ) : ℝ := 7 * x^3 - 5 * x^2 + 3 * x - 6

theorem g_of_3 : g 3 = 147 := by
  sorry

end NUMINAMATH_CALUDE_g_of_3_l1343_134307


namespace NUMINAMATH_CALUDE_sin_value_given_sum_and_tan_condition_l1343_134349

theorem sin_value_given_sum_and_tan_condition (θ : Real) 
  (h1 : Real.sin θ + Real.cos θ = 7/5)
  (h2 : Real.tan θ < 1) : 
  Real.sin θ = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_value_given_sum_and_tan_condition_l1343_134349


namespace NUMINAMATH_CALUDE_frustum_smaller_cone_altitude_l1343_134368

-- Define the frustum
structure Frustum where
  altitude : ℝ
  lowerBaseArea : ℝ
  upperBaseArea : ℝ

-- Define the theorem
theorem frustum_smaller_cone_altitude (f : Frustum) 
  (h1 : f.altitude = 30)
  (h2 : f.lowerBaseArea = 400 * Real.pi)
  (h3 : f.upperBaseArea = 100 * Real.pi) :
  ∃ (smallerConeAltitude : ℝ), smallerConeAltitude = f.altitude := by
  sorry

end NUMINAMATH_CALUDE_frustum_smaller_cone_altitude_l1343_134368


namespace NUMINAMATH_CALUDE_maplewood_elementary_difference_l1343_134312

theorem maplewood_elementary_difference (students_per_class : ℕ) (guinea_pigs_per_class : ℕ) (num_classes : ℕ) :
  students_per_class = 20 →
  guinea_pigs_per_class = 3 →
  num_classes = 4 →
  students_per_class * num_classes - guinea_pigs_per_class * num_classes = 68 :=
by
  sorry

end NUMINAMATH_CALUDE_maplewood_elementary_difference_l1343_134312


namespace NUMINAMATH_CALUDE_rectangle_toothpicks_l1343_134361

/-- Calculate the number of toothpicks needed for a rectangular grid -/
def toothpicks_in_rectangle (length width : ℕ) : ℕ :=
  (length + 1) * width + (width + 1) * length

/-- Theorem: A rectangular grid with length 20 and width 10 requires 430 toothpicks -/
theorem rectangle_toothpicks :
  toothpicks_in_rectangle 20 10 = 430 := by
  sorry

#eval toothpicks_in_rectangle 20 10

end NUMINAMATH_CALUDE_rectangle_toothpicks_l1343_134361


namespace NUMINAMATH_CALUDE_min_value_trig_expression_l1343_134318

theorem min_value_trig_expression (θ : Real) (h : 0 < θ ∧ θ < π/2) :
  ∃ (min_val : Real), min_val = 5 * Real.sqrt 5 ∧
  ∀ ϕ, 0 < ϕ ∧ ϕ < π/2 → (8 / Real.cos ϕ + 1 / Real.sin ϕ) ≥ min_val :=
by sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_l1343_134318


namespace NUMINAMATH_CALUDE_polynomial_root_problem_l1343_134337

theorem polynomial_root_problem (d : ℚ) :
  (∃ (x : ℝ), x^3 + 4*x + d = 0 ∧ x = 2 + Real.sqrt 5) →
  (∃ (n : ℤ), n^3 + 4*n + d = 0) →
  (∃ (n : ℤ), n^3 + 4*n + d = 0 ∧ n = -4) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_root_problem_l1343_134337


namespace NUMINAMATH_CALUDE_sequence_property_l1343_134327

/-- The function generating the sequence -/
def f (n : ℕ) : ℕ := 2 * (n + 1)^2 * (n + 2)^2

/-- Predicate to check if a number is the sum of two square integers -/
def isSumOfTwoSquares (m : ℕ) : Prop := ∃ a b : ℕ, m = a^2 + b^2

theorem sequence_property :
  (∀ n : ℕ, f n < f (n + 1)) ∧
  (∀ n : ℕ, isSumOfTwoSquares (f n)) ∧
  f 1 = 72 ∧ f 2 = 288 ∧ f 3 = 800 :=
sorry

end NUMINAMATH_CALUDE_sequence_property_l1343_134327


namespace NUMINAMATH_CALUDE_B_is_closed_l1343_134394

def ClosedSet (A : Set ℤ) : Prop :=
  ∀ a b : ℤ, a ∈ A → b ∈ A → (a + b) ∈ A ∧ (a - b) ∈ A

def B : Set ℤ := {n : ℤ | ∃ k : ℤ, n = 3 * k}

theorem B_is_closed : ClosedSet B := by
  sorry

end NUMINAMATH_CALUDE_B_is_closed_l1343_134394


namespace NUMINAMATH_CALUDE_polyhedral_angle_sum_lt_360_l1343_134385

/-- A polyhedral angle is represented by a list of planar angles (in degrees) -/
def PolyhedralAngle := List Float

/-- The sum of planar angles in a polyhedral angle is less than 360° -/
theorem polyhedral_angle_sum_lt_360 (pa : PolyhedralAngle) : 
  pa.sum < 360 := by sorry

end NUMINAMATH_CALUDE_polyhedral_angle_sum_lt_360_l1343_134385


namespace NUMINAMATH_CALUDE_trolley_passengers_count_l1343_134387

/-- Calculates the number of people on a trolley after three stops -/
def trolley_passengers : ℕ :=
  let initial := 1  -- driver
  let first_stop := initial + 10
  let second_stop := first_stop - 3 + (2 * 10)
  let third_stop := second_stop - 18 + 2
  third_stop

theorem trolley_passengers_count : trolley_passengers = 12 := by
  sorry

end NUMINAMATH_CALUDE_trolley_passengers_count_l1343_134387


namespace NUMINAMATH_CALUDE_car_average_speed_l1343_134320

/-- Given a car's speed for two hours, calculate its average speed. -/
theorem car_average_speed (speed1 speed2 : ℝ) (h1 : speed1 = 85) (h2 : speed2 = 45) :
  (speed1 + speed2) / 2 = 65 := by
  sorry

#check car_average_speed

end NUMINAMATH_CALUDE_car_average_speed_l1343_134320


namespace NUMINAMATH_CALUDE_rectangle_area_similarity_l1343_134364

theorem rectangle_area_similarity (R1_side : ℝ) (R1_area : ℝ) (R2_diagonal : ℝ) :
  R1_side = 3 →
  R1_area = 24 →
  R2_diagonal = 20 →
  ∃ (R2_area : ℝ), R2_area = 3200 / 73 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_similarity_l1343_134364


namespace NUMINAMATH_CALUDE_initial_maple_trees_l1343_134346

/-- The number of maple trees in the park after planting -/
def total_trees : ℕ := 64

/-- The number of maple trees planted today -/
def planted_trees : ℕ := 11

/-- The initial number of maple trees in the park -/
def initial_trees : ℕ := total_trees - planted_trees

theorem initial_maple_trees : initial_trees = 53 := by
  sorry

end NUMINAMATH_CALUDE_initial_maple_trees_l1343_134346


namespace NUMINAMATH_CALUDE_marys_oranges_l1343_134391

theorem marys_oranges :
  ∀ (oranges : ℕ),
    (14 + oranges + 6 - 3 = 26) →
    oranges = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_marys_oranges_l1343_134391


namespace NUMINAMATH_CALUDE_symmetry_proof_l1343_134377

/-- A point in the 2D Cartesian coordinate system -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Symmetry with respect to the y-axis -/
def symmetric_wrt_y_axis (p q : Point) : Prop :=
  p.x = -q.x ∧ p.y = q.y

/-- The given point -/
def given_point : Point :=
  { x := -1, y := -2 }

/-- The symmetric point to be proved -/
def symmetric_point : Point :=
  { x := 1, y := -2 }

theorem symmetry_proof : symmetric_wrt_y_axis given_point symmetric_point := by
  sorry

end NUMINAMATH_CALUDE_symmetry_proof_l1343_134377


namespace NUMINAMATH_CALUDE_krakozyabr_population_is_32_l1343_134362

structure Krakozyabr where
  hasHorns : Bool
  hasWings : Bool

def totalKrakozyabrs (population : List Krakozyabr) : Nat :=
  population.length

theorem krakozyabr_population_is_32 
  (population : List Krakozyabr) 
  (all_have_horns_or_wings : ∀ k ∈ population, k.hasHorns ∨ k.hasWings)
  (horns_with_wings_ratio : (population.filter (λ k => k.hasHorns ∧ k.hasWings)).length = 
    (population.filter (λ k => k.hasHorns)).length / 5)
  (wings_with_horns_ratio : (population.filter (λ k => k.hasHorns ∧ k.hasWings)).length = 
    (population.filter (λ k => k.hasWings)).length / 4)
  (population_range : 25 < totalKrakozyabrs population ∧ totalKrakozyabrs population < 35) :
  totalKrakozyabrs population = 32 := by
  sorry

end NUMINAMATH_CALUDE_krakozyabr_population_is_32_l1343_134362


namespace NUMINAMATH_CALUDE_roses_cut_l1343_134336

theorem roses_cut (initial_roses final_roses : ℕ) (h1 : initial_roses = 3) (h2 : final_roses = 14) :
  final_roses - initial_roses = 11 := by
  sorry

end NUMINAMATH_CALUDE_roses_cut_l1343_134336


namespace NUMINAMATH_CALUDE_second_to_last_term_l1343_134355

-- Define the sequence type
def Sequence := Fin 201 → ℕ

-- Define the properties of the sequence
def ValidSequence (a : Sequence) : Prop :=
  (a 0 = 19999) ∧ 
  (a 200 = 19999) ∧
  (∃ t : ℕ+, ∀ n : Fin 199, 
    a (n + 1) + t = (a n + a (n + 2)) / 2)

-- Theorem statement
theorem second_to_last_term (a : Sequence) 
  (h : ValidSequence a) : a 199 = 19800 := by
  sorry

end NUMINAMATH_CALUDE_second_to_last_term_l1343_134355


namespace NUMINAMATH_CALUDE_new_car_cost_proof_l1343_134371

/-- The monthly cost of renting a car -/
def rental_cost : ℕ := 20

/-- The number of months in a year -/
def months_in_year : ℕ := 12

/-- The total difference in cost over a year between renting and buying -/
def total_difference : ℕ := 120

/-- The monthly cost of the new car -/
def new_car_cost : ℕ := 30

theorem new_car_cost_proof : 
  new_car_cost * months_in_year - rental_cost * months_in_year = total_difference := by
  sorry

end NUMINAMATH_CALUDE_new_car_cost_proof_l1343_134371


namespace NUMINAMATH_CALUDE_rearranged_digits_subtraction_l1343_134376

theorem rearranged_digits_subtraction :
  ∀ h t u : ℕ,
  h ≠ t → h ≠ u → t ≠ u →
  h > 0 → t > 0 → u > 0 →
  h > u →
  h * 100 + t * 10 + u - (t * 100 + u * 10 + h) = 179 →
  h = 8 ∧ t = 7 ∧ u = 9 :=
by sorry

end NUMINAMATH_CALUDE_rearranged_digits_subtraction_l1343_134376


namespace NUMINAMATH_CALUDE_min_jumps_to_cover_race_l1343_134352

/-- A cricket can jump either 9 meters or 8 meters. -/
inductive JumpDistance : Type
  | long : JumpDistance  -- 9 meters
  | short : JumpDistance -- 8 meters

/-- The race distance is 100 meters. -/
def raceDistance : ℕ := 100

/-- The distance covered by a jump. -/
def jumpLength (j : JumpDistance) : ℕ :=
  match j with
  | JumpDistance.long => 9
  | JumpDistance.short => 8

/-- The total distance covered by a sequence of jumps. -/
def totalDistance (jumps : List JumpDistance) : ℕ :=
  jumps.foldl (fun acc j => acc + jumpLength j) 0

/-- A valid jump sequence covers exactly the race distance. -/
def isValidJumpSequence (jumps : List JumpDistance) : Prop :=
  totalDistance jumps = raceDistance

/-- The theorem stating that the minimum number of jumps to cover the race distance is 12. -/
theorem min_jumps_to_cover_race : 
  ∃ (jumps : List JumpDistance), isValidJumpSequence jumps ∧ 
    (∀ (other_jumps : List JumpDistance), isValidJumpSequence other_jumps → 
      jumps.length ≤ other_jumps.length) ∧
    jumps.length = 12 := by
  sorry

end NUMINAMATH_CALUDE_min_jumps_to_cover_race_l1343_134352


namespace NUMINAMATH_CALUDE_sequence_properties_l1343_134333

def arithmetic_seq (a b n : ℕ) : ℕ := a + (n - 1) * b

def geometric_seq (b a n : ℕ) : ℕ := b * a^(n - 1)

def c_seq (a b n : ℕ) : ℚ := (arithmetic_seq a b n - 8) / (geometric_seq b a n)

theorem sequence_properties (a b : ℕ) :
  (a > 0) →
  (b > 0) →
  (arithmetic_seq a b 1 < geometric_seq b a 1) →
  (geometric_seq b a 1 < arithmetic_seq a b 2) →
  (arithmetic_seq a b 2 < geometric_seq b a 2) →
  (geometric_seq b a 2 < arithmetic_seq a b 3) →
  (∃ m n : ℕ, m > 0 ∧ n > 0 ∧ arithmetic_seq a b m + 1 = geometric_seq b a n) →
  (a = 2 ∧ b = 3 ∧ ∃ k : ℕ, ∀ n : ℕ, n > 0 → c_seq a b n ≤ c_seq a b k ∧ c_seq a b k = 1/8) :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l1343_134333


namespace NUMINAMATH_CALUDE_circle_equation_k_range_l1343_134300

theorem circle_equation_k_range (x y k : ℝ) :
  (∃ r : ℝ, r > 0 ∧ ∀ x y, x^2 + y^2 - 2*x + y + k = 0 ↔ (x - 1)^2 + (y + 1/2)^2 = r^2) →
  k < 5/4 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_k_range_l1343_134300


namespace NUMINAMATH_CALUDE_expression_evaluation_l1343_134339

theorem expression_evaluation : -2^3 + 36 / 3^2 * (-1/2) + |(-5)| = -5 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1343_134339


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l1343_134388

/-- The distance between the foci of the ellipse 9x^2 + y^2 = 36 is 8√2 -/
theorem ellipse_foci_distance (x y : ℝ) :
  (9 * x^2 + y^2 = 36) → (∃ f₁ f₂ : ℝ × ℝ, 
    (f₁.1 - f₂.1)^2 + (f₁.2 - f₂.2)^2 = 128) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l1343_134388


namespace NUMINAMATH_CALUDE_tuesday_wednesday_most_available_l1343_134367

-- Define the days of the week
inductive Day
| monday
| tuesday
| wednesday
| thursday
| friday
| saturday

-- Define the students
inductive Student
| alice
| bob
| cindy
| david
| eva

-- Define the availability function
def availability (s : Student) (d : Day) : Bool :=
  match s, d with
  | Student.alice, Day.monday => false
  | Student.alice, Day.tuesday => true
  | Student.alice, Day.wednesday => false
  | Student.alice, Day.thursday => true
  | Student.alice, Day.friday => true
  | Student.alice, Day.saturday => false
  | Student.bob, Day.monday => true
  | Student.bob, Day.tuesday => false
  | Student.bob, Day.wednesday => true
  | Student.bob, Day.thursday => false
  | Student.bob, Day.friday => false
  | Student.bob, Day.saturday => true
  | Student.cindy, Day.monday => false
  | Student.cindy, Day.tuesday => false
  | Student.cindy, Day.wednesday => true
  | Student.cindy, Day.thursday => false
  | Student.cindy, Day.friday => false
  | Student.cindy, Day.saturday => true
  | Student.david, Day.monday => true
  | Student.david, Day.tuesday => true
  | Student.david, Day.wednesday => false
  | Student.david, Day.thursday => false
  | Student.david, Day.friday => true
  | Student.david, Day.saturday => false
  | Student.eva, Day.monday => false
  | Student.eva, Day.tuesday => true
  | Student.eva, Day.wednesday => true
  | Student.eva, Day.thursday => true
  | Student.eva, Day.friday => false
  | Student.eva, Day.saturday => false

-- Count available students for a given day
def availableStudents (d : Day) : Nat :=
  (Student.alice :: Student.bob :: Student.cindy :: Student.david :: Student.eva :: []).filter (fun s => availability s d) |>.length

-- Theorem stating that Tuesday and Wednesday have the most available students
theorem tuesday_wednesday_most_available :
  (availableStudents Day.tuesday = availableStudents Day.wednesday) ∧
  (∀ d : Day, availableStudents d ≤ availableStudents Day.tuesday) :=
by sorry

end NUMINAMATH_CALUDE_tuesday_wednesday_most_available_l1343_134367


namespace NUMINAMATH_CALUDE_square_perimeter_problem_l1343_134399

/-- Given squares I, II, and III, prove that the perimeter of III is 16√2 + 32 -/
theorem square_perimeter_problem (I II III : ℝ → ℝ → Prop) : 
  (∀ x, I x x → 4 * x = 16) →  -- Square I has perimeter 16
  (∀ y, II y y → 4 * y = 32) →  -- Square II has perimeter 32
  (∀ x y z, I x x → II y y → III z z → z = x * Real.sqrt 2 + y) →  -- Side of III is diagonal of I plus side of II
  (∃ z, III z z ∧ 4 * z = 16 * Real.sqrt 2 + 32) :=  -- Perimeter of III is 16√2 + 32
by sorry

end NUMINAMATH_CALUDE_square_perimeter_problem_l1343_134399


namespace NUMINAMATH_CALUDE_bird_nest_area_scientific_notation_l1343_134356

/-- The construction area of the National Stadium "Bird's Nest" in square meters -/
def bird_nest_area : ℝ := 258000

/-- The scientific notation representation of the bird_nest_area -/
def bird_nest_scientific : ℝ := 2.58 * (10 ^ 5)

theorem bird_nest_area_scientific_notation :
  bird_nest_area = bird_nest_scientific := by
  sorry

end NUMINAMATH_CALUDE_bird_nest_area_scientific_notation_l1343_134356

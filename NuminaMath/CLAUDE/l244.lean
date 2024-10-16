import Mathlib

namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l244_24424

theorem imaginary_part_of_z (z : ℂ) (h : z * (1 - Complex.I) = Complex.abs (Complex.I + 1)) :
  z.im = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l244_24424


namespace NUMINAMATH_CALUDE_crayons_in_drawer_l244_24428

theorem crayons_in_drawer (initial_crayons final_crayons benny_crayons : ℕ) : 
  initial_crayons = 9 → 
  final_crayons = 12 → 
  benny_crayons = final_crayons - initial_crayons →
  benny_crayons = 3 := by
sorry

end NUMINAMATH_CALUDE_crayons_in_drawer_l244_24428


namespace NUMINAMATH_CALUDE_age_difference_l244_24464

/-- Given three people a, b, and c, prove that a is 2 years older than b -/
theorem age_difference (a b c : ℕ) : 
  (∃ k, a = b + k) →  -- a is some years older than b
  (b = 2 * c) →       -- b is twice as old as c
  (a + b + c = 27) →  -- The total of the ages of a, b, and c is 27
  (b = 10) →          -- b is 10 years old
  (a = b + 2)         -- a is 2 years older than b
  := by sorry

end NUMINAMATH_CALUDE_age_difference_l244_24464


namespace NUMINAMATH_CALUDE_quiz_ranking_l244_24481

theorem quiz_ranking (F E H G : ℝ) 
  (nonneg : F ≥ 0 ∧ E ≥ 0 ∧ H ≥ 0 ∧ G ≥ 0)
  (sum_equal : E + G = F + H)
  (sum_equal_swap : F + E = H + G)
  (george_higher : G > E + F) :
  G > E ∧ G > H ∧ E = H ∧ E > F ∧ H > F := by
  sorry

end NUMINAMATH_CALUDE_quiz_ranking_l244_24481


namespace NUMINAMATH_CALUDE_circle_radius_l244_24491

theorem circle_radius (P Q : ℝ) (h : P / Q = 15) : 
  ∃ r : ℝ, r > 0 ∧ P = π * r^2 ∧ Q = 2 * π * r ∧ r = 30 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_l244_24491


namespace NUMINAMATH_CALUDE_percent_relation_l244_24472

theorem percent_relation (a b : ℝ) (h : a = 1.25 * b) : 4 * b = 3.2 * a := by
  sorry

end NUMINAMATH_CALUDE_percent_relation_l244_24472


namespace NUMINAMATH_CALUDE_surface_area_ratio_of_cubes_l244_24462

theorem surface_area_ratio_of_cubes (a b : ℝ) (h : a / b = 7) :
  (6 * a^2) / (6 * b^2) = 49 := by
  sorry

end NUMINAMATH_CALUDE_surface_area_ratio_of_cubes_l244_24462


namespace NUMINAMATH_CALUDE_negation_equivalence_l244_24443

theorem negation_equivalence :
  (¬ ∀ a : ℝ, a ∈ Set.Icc 0 1 → a^4 + a^2 > 1) ↔
  (∃ a : ℝ, a ∈ Set.Icc 0 1 ∧ a^4 + a^2 ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l244_24443


namespace NUMINAMATH_CALUDE_only_two_works_l244_24490

/-- A move that can be applied to a table --/
inductive Move
  | MultiplyRow (row : Nat) : Move
  | SubtractColumn (col : Nat) : Move

/-- Definition of a rectangular table with positive integer entries --/
def Table := List (List Nat)

/-- Apply a move to a table --/
def applyMove (n : Nat) (t : Table) (m : Move) : Table :=
  sorry

/-- Check if all entries in a table are zero --/
def allZero (t : Table) : Prop :=
  sorry

/-- The main theorem --/
theorem only_two_works (n : Nat) : 
  (n > 0) → 
  (∀ t : Table, ∃ moves : List Move, allZero (moves.foldl (applyMove n) t)) ↔ 
  n = 2 := by
  sorry

end NUMINAMATH_CALUDE_only_two_works_l244_24490


namespace NUMINAMATH_CALUDE_more_karabases_than_barabases_l244_24452

/-- Represents the inhabitants of Perra-Terra -/
inductive Inhabitant
| Karabas
| Barabas

/-- The number of acquaintances each type of inhabitant has -/
def acquaintances (i : Inhabitant) : Nat × Nat :=
  match i with
  | Inhabitant.Karabas => (6, 9)  -- (Other Karabases, Barabases)
  | Inhabitant.Barabas => (10, 7) -- (Karabases, Other Barabases)

theorem more_karabases_than_barabases (K B : Nat) 
  (hK : K > 0) (hB : B > 0) 
  (h_acquaintances : K * (acquaintances Inhabitant.Karabas).2 = B * (acquaintances Inhabitant.Barabas).1) :
  K > B := by
  sorry

#check more_karabases_than_barabases

end NUMINAMATH_CALUDE_more_karabases_than_barabases_l244_24452


namespace NUMINAMATH_CALUDE_octal_7421_to_decimal_l244_24422

def octal_to_decimal (octal : ℕ) : ℕ :=
  let digits := [7, 4, 2, 1]
  (List.zipWith (λ (d : ℕ) (p : ℕ) => d * (8 ^ p)) digits (List.range 4)).sum

theorem octal_7421_to_decimal :
  octal_to_decimal 7421 = 1937 := by
  sorry

end NUMINAMATH_CALUDE_octal_7421_to_decimal_l244_24422


namespace NUMINAMATH_CALUDE_parabola_directrix_coefficient_l244_24415

/-- For a parabola with equation y = ax² and directrix y = 2, prove that a = -1/8 -/
theorem parabola_directrix_coefficient (a : ℝ) : 
  (∀ x y : ℝ, y = a * x^2) →  -- Parabola equation
  (∃ y : ℝ, y = 2 ∧ ∀ x : ℝ, y = -1 / (4 * a)) →  -- Directrix equation
  a = -1/8 := by
sorry

end NUMINAMATH_CALUDE_parabola_directrix_coefficient_l244_24415


namespace NUMINAMATH_CALUDE_total_cards_l244_24496

theorem total_cards (initial_cards added_cards : ℕ) 
  (h1 : initial_cards = 4)
  (h2 : added_cards = 3) :
  initial_cards + added_cards = 7 := by
sorry

end NUMINAMATH_CALUDE_total_cards_l244_24496


namespace NUMINAMATH_CALUDE_original_cost_of_mixed_nuts_l244_24418

/-- Calculates the original cost of a bag of mixed nuts -/
theorem original_cost_of_mixed_nuts
  (bag_size : ℕ)
  (serving_size : ℕ)
  (cost_per_serving_after_coupon : ℚ)
  (coupon_value : ℚ)
  (h1 : bag_size = 40)
  (h2 : serving_size = 1)
  (h3 : cost_per_serving_after_coupon = 1/2)
  (h4 : coupon_value = 5) :
  bag_size * cost_per_serving_after_coupon + coupon_value = 25 :=
sorry

end NUMINAMATH_CALUDE_original_cost_of_mixed_nuts_l244_24418


namespace NUMINAMATH_CALUDE_gcd_102_238_l244_24488

theorem gcd_102_238 : Nat.gcd 102 238 = 34 := by sorry

end NUMINAMATH_CALUDE_gcd_102_238_l244_24488


namespace NUMINAMATH_CALUDE_billiard_angle_range_l244_24400

/-- A regular hexagon with vertices A, B, C, D, E, F -/
structure RegularHexagon where
  vertices : Fin 6 → ℝ × ℝ
  is_regular : sorry

/-- A billiard ball trajectory on a regular hexagon -/
structure BilliardTrajectory (hex : RegularHexagon) where
  start_point : ℝ × ℝ
  is_midpoint_AB : sorry
  hit_points : Fin 6 → ℝ × ℝ
  hits_sides_in_order : sorry
  angle_of_incidence_equals_reflection : sorry

/-- The theorem stating the range of possible values for the initial angle θ -/
theorem billiard_angle_range (hex : RegularHexagon) (traj : BilliardTrajectory hex) :
  let θ := sorry -- angle between BP and BQ
  Real.arctan (3 * Real.sqrt 3 / 10) < θ ∧ θ < Real.arctan (3 * Real.sqrt 3 / 8) := by
  sorry

end NUMINAMATH_CALUDE_billiard_angle_range_l244_24400


namespace NUMINAMATH_CALUDE_complex_number_problem_l244_24493

theorem complex_number_problem (m : ℝ) (z z₁ : ℂ) :
  z₁ = m * (m - 1) + (m - 1) * Complex.I ∧
  z₁.re = 0 ∧
  z₁.im ≠ 0 ∧
  (3 + z₁) * z = 4 + 2 * Complex.I →
  m = 0 ∧ z = 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_number_problem_l244_24493


namespace NUMINAMATH_CALUDE_sin_shift_l244_24497

theorem sin_shift (x : ℝ) : 
  Real.sin (2 * x - π / 3) = Real.sin (2 * (x - π / 4) + π / 6) := by
  sorry

end NUMINAMATH_CALUDE_sin_shift_l244_24497


namespace NUMINAMATH_CALUDE_congruence_definition_l244_24409

-- Define a type for geometric figures
def Figure : Type := sorry

-- Define a relation for figures that can completely overlap
def can_overlap (f1 f2 : Figure) : Prop := sorry

-- Define congruence for figures
def congruent (f1 f2 : Figure) : Prop := sorry

-- Theorem stating the definition of congruent figures
theorem congruence_definition :
  ∀ (f1 f2 : Figure), congruent f1 f2 ↔ can_overlap f1 f2 := by sorry

end NUMINAMATH_CALUDE_congruence_definition_l244_24409


namespace NUMINAMATH_CALUDE_remainder_6n_mod_4_l244_24480

theorem remainder_6n_mod_4 (n : ℤ) (h : n ≡ 3 [ZMOD 4]) : 6 * n ≡ 2 [ZMOD 4] := by
  sorry

end NUMINAMATH_CALUDE_remainder_6n_mod_4_l244_24480


namespace NUMINAMATH_CALUDE_opposite_of_negative_two_l244_24425

theorem opposite_of_negative_two : (- (-2)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_two_l244_24425


namespace NUMINAMATH_CALUDE_tangent_line_parallel_points_l244_24427

def f (x : ℝ) : ℝ := x^3 - 2

theorem tangent_line_parallel_points :
  ∀ x y : ℝ, f x = y →
  (∃ k : ℝ, k * (x - 1) = y + 1 ∧ k = 3) ↔ 
  ((x = 1 ∧ y = -1) ∨ (x = -1 ∧ y = -3)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_parallel_points_l244_24427


namespace NUMINAMATH_CALUDE_quadratic_problem_l244_24433

def quadratic_function (a b x : ℝ) : ℝ := a * x^2 - 4 * a * x + 3 + b

theorem quadratic_problem (a b : ℤ) (h1 : a ≠ 0) (h2 : a > 0) 
  (h3 : 4 < a + |b| ∧ a + |b| < 9) 
  (h4 : quadratic_function a b 1 = 3) :
  (∃ (x : ℝ), x = 2 ∧ ∀ (y : ℝ), quadratic_function a b (x - y) = quadratic_function a b (x + y)) ∧
  (a = 2 ∧ b = 6) ∧
  (∃ (t : ℝ), (t = 1/2 ∨ t = 5/2) ∧
    ∀ (x : ℝ), t ≤ x ∧ x ≤ t + 1 → quadratic_function a b x ≥ 3/2 ∧
    ∃ (x₀ : ℝ), t ≤ x₀ ∧ x₀ ≤ t + 1 ∧ quadratic_function a b x₀ = 3/2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_problem_l244_24433


namespace NUMINAMATH_CALUDE_bottle_caps_eaten_l244_24447

theorem bottle_caps_eaten (initial : ℕ) (final : ℕ) (eaten : ℕ) : 
  initial = 65 → final = 61 → initial - final = eaten → eaten = 4 := by
sorry

end NUMINAMATH_CALUDE_bottle_caps_eaten_l244_24447


namespace NUMINAMATH_CALUDE_infinite_sum_equals_9_320_l244_24450

/-- The sum of the infinite series n / (n^4 + 16) from n=1 to infinity equals 9/320 -/
theorem infinite_sum_equals_9_320 :
  (∑' n : ℕ, n / (n^4 + 16 : ℝ)) = 9 / 320 := by
  sorry

end NUMINAMATH_CALUDE_infinite_sum_equals_9_320_l244_24450


namespace NUMINAMATH_CALUDE_wall_thickness_l244_24423

/-- Calculates the thickness of a wall given its dimensions and the number of bricks used. -/
theorem wall_thickness
  (wall_length : Real)
  (wall_height : Real)
  (brick_length : Real)
  (brick_width : Real)
  (brick_height : Real)
  (num_bricks : Nat)
  (h_wall_length : wall_length = 9)
  (h_wall_height : wall_height = 6)
  (h_brick_length : brick_length = 0.25)
  (h_brick_width : brick_width = 0.1125)
  (h_brick_height : brick_height = 0.06)
  (h_num_bricks : num_bricks = 7200) :
  ∃ (wall_thickness : Real),
    wall_thickness = 0.225 ∧
    wall_length * wall_height * wall_thickness =
      num_bricks * brick_length * brick_width * brick_height :=
by sorry

end NUMINAMATH_CALUDE_wall_thickness_l244_24423


namespace NUMINAMATH_CALUDE_rope_length_l244_24454

/-- Calculates the length of a rope in centimeters given specific conditions -/
theorem rope_length : 
  let total_pieces : ℕ := 154
  let equal_pieces : ℕ := 150
  let equal_piece_length : ℕ := 75  -- in millimeters
  let remaining_piece_length : ℕ := 100  -- in millimeters
  let total_length : ℕ := equal_pieces * equal_piece_length + 
                          (total_pieces - equal_pieces) * remaining_piece_length
  total_length / 10 = 1165  -- length in centimeters
  := by sorry

end NUMINAMATH_CALUDE_rope_length_l244_24454


namespace NUMINAMATH_CALUDE_tim_water_consumption_l244_24417

/-- The number of ounces in a quart -/
def ounces_per_quart : ℕ := 32

/-- The number of quarts in each bottle Tim drinks -/
def quarts_per_bottle : ℚ := 3/2

/-- The number of bottles Tim drinks per day -/
def bottles_per_day : ℕ := 2

/-- The additional ounces Tim drinks per day -/
def additional_ounces_per_day : ℕ := 20

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The total amount of water Tim drinks in a week, in ounces -/
def water_per_week : ℕ := 812

theorem tim_water_consumption :
  (bottles_per_day * (quarts_per_bottle * ounces_per_quart).floor + additional_ounces_per_day) * days_per_week = water_per_week :=
sorry

end NUMINAMATH_CALUDE_tim_water_consumption_l244_24417


namespace NUMINAMATH_CALUDE_trigonometric_expression_simplification_l244_24413

theorem trigonometric_expression_simplification :
  (Real.tan (20 * π / 180) + Real.tan (30 * π / 180) + Real.tan (60 * π / 180) + Real.tan (70 * π / 180)) / Real.cos (10 * π / 180) = 8 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_simplification_l244_24413


namespace NUMINAMATH_CALUDE_golden_ratio_expressions_l244_24403

theorem golden_ratio_expressions (θ : Real) (h : θ = 18 * π / 180) :
  let φ := (Real.sqrt 5 - 1) / 4
  φ = Real.sin θ ∧
  φ = Real.cos (10 * π / 180) * Real.cos (82 * π / 180) + Real.sin (10 * π / 180) * Real.sin (82 * π / 180) ∧
  φ = Real.sin (173 * π / 180) * Real.cos (11 * π / 180) - Real.sin (83 * π / 180) * Real.cos (101 * π / 180) ∧
  φ = Real.sqrt ((1 - Real.sin (54 * π / 180)) / 2) :=
by sorry

end NUMINAMATH_CALUDE_golden_ratio_expressions_l244_24403


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_product_l244_24407

theorem partial_fraction_decomposition_product (x : ℝ) 
  (A B C : ℝ) : 
  (x^2 - 25) / ((x - 2) * (x + 3) * (x - 4)) = 
    A / (x - 2) + B / (x + 3) + C / (x - 4) → 
  A * B * C = 1 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_product_l244_24407


namespace NUMINAMATH_CALUDE_min_value_theorem_l244_24458

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 5/b = 1) :
  a + 5*b ≥ 36 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 1/a₀ + 5/b₀ = 1 ∧ a₀ + 5*b₀ = 36 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l244_24458


namespace NUMINAMATH_CALUDE_brocard_and_steiner_coordinates_l244_24402

/-- Given a triangle with side lengths a, b, and c, this theorem states the trilinear coordinates
    of vertex A1 of the Brocard triangle and the Steiner point. -/
theorem brocard_and_steiner_coordinates (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) :
  ∃ (k₁ k₂ : ℝ),
    k₁ > 0 ∧ k₂ > 0 ∧
    (k₁ * (a * b * c), k₁ * c^3, k₁ * b^3) = (1, 1, 1) ∧
    (k₂ / (a * (b^2 - c^2)), k₂ / (b * (c^2 - a^2)), k₂ / (c * (a^2 - b^2))) = (1, 1, 1) :=
by sorry

end NUMINAMATH_CALUDE_brocard_and_steiner_coordinates_l244_24402


namespace NUMINAMATH_CALUDE_f_inequality_range_l244_24479

noncomputable def f (x : ℝ) : ℝ := Real.log (1 + |x|) - 1 / (1 + x^2)

theorem f_inequality_range (x : ℝ) : 
  f (2 * x) > f (x - 1) ↔ x < -1 ∨ x > 1/3 :=
sorry

end NUMINAMATH_CALUDE_f_inequality_range_l244_24479


namespace NUMINAMATH_CALUDE_smallest_nine_digit_divisible_by_11_l244_24498

theorem smallest_nine_digit_divisible_by_11 : ℕ :=
  let n := 100000010
  have h1 : n ≥ 100000000 ∧ n < 1000000000 := by sorry
  have h2 : n % 11 = 0 := by sorry
  have h3 : ∀ m : ℕ, m ≥ 100000000 ∧ m < n → m % 11 ≠ 0 := by sorry
  n

end NUMINAMATH_CALUDE_smallest_nine_digit_divisible_by_11_l244_24498


namespace NUMINAMATH_CALUDE_smaller_solution_quadratic_l244_24440

theorem smaller_solution_quadratic (x : ℝ) : 
  (x^2 + 17*x - 72 = 0) → (x = -24 ∨ x = 3) → x = min (-24) 3 := by
  sorry

end NUMINAMATH_CALUDE_smaller_solution_quadratic_l244_24440


namespace NUMINAMATH_CALUDE_complex_number_properties_l244_24476

/-- Given a complex number z = (a+i)(1-i)+bi where a and b are real, and the point
    corresponding to z in the complex plane lies on the graph of y = x - 3 -/
theorem complex_number_properties (a b : ℝ) (z : ℂ) 
  (h1 : z = (a + Complex.I) * (1 - Complex.I) + b * Complex.I)
  (h2 : z.im = z.re - 3) : 
  (2 * a > b) ∧ (Complex.abs z ≥ 3 * Real.sqrt 2 / 2) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_properties_l244_24476


namespace NUMINAMATH_CALUDE_cube_volume_from_space_diagonal_l244_24419

theorem cube_volume_from_space_diagonal (d : ℝ) (h : d = 6 * Real.sqrt 3) :
  let s := d / Real.sqrt 3
  s ^ 3 = 216 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_space_diagonal_l244_24419


namespace NUMINAMATH_CALUDE_plot_length_is_75_l244_24412

/-- Proves that the length of a rectangular plot is 75 meters given the specified conditions -/
theorem plot_length_is_75 (breadth : ℝ) (length : ℝ) (perimeter : ℝ) (cost_per_meter : ℝ) (total_cost : ℝ) :
  length = breadth + 50 →
  perimeter = 2 * length + 2 * breadth →
  cost_per_meter = 26.50 →
  total_cost = 5300 →
  total_cost = perimeter * cost_per_meter →
  length = 75 := by
sorry

end NUMINAMATH_CALUDE_plot_length_is_75_l244_24412


namespace NUMINAMATH_CALUDE_chair_price_l244_24432

theorem chair_price (total_cost : ℕ) (num_desks : ℕ) (num_chairs : ℕ) (desk_price : ℕ) :
  total_cost = 1236 →
  num_desks = 5 →
  num_chairs = 8 →
  desk_price = 180 →
  (total_cost - num_desks * desk_price) / num_chairs = 42 := by
  sorry

end NUMINAMATH_CALUDE_chair_price_l244_24432


namespace NUMINAMATH_CALUDE_probability_five_odd_in_six_rolls_l244_24430

theorem probability_five_odd_in_six_rolls : 
  let n : ℕ := 6  -- number of rolls
  let k : ℕ := 5  -- number of desired odd rolls
  let p : ℚ := 1/2  -- probability of rolling an odd number on a single roll
  Nat.choose n k * p^k * (1-p)^(n-k) = 3/32 := by
sorry

end NUMINAMATH_CALUDE_probability_five_odd_in_six_rolls_l244_24430


namespace NUMINAMATH_CALUDE_N_mod_100_l244_24416

/-- The number of ways to select a group of singers satisfying given conditions -/
def N : ℕ := sorry

/-- The total number of tenors available -/
def num_tenors : ℕ := 8

/-- The total number of basses available -/
def num_basses : ℕ := 10

/-- The total number of singers to be selected -/
def total_singers : ℕ := 6

/-- Predicate to check if a group satisfies the conditions -/
def valid_group (tenors basses : ℕ) : Prop :=
  tenors + basses = total_singers ∧ 
  ∃ k : ℤ, tenors - basses = 4 * k

theorem N_mod_100 : N % 100 = 96 := by sorry

end NUMINAMATH_CALUDE_N_mod_100_l244_24416


namespace NUMINAMATH_CALUDE_birds_beetles_per_day_l244_24483

-- Define the constants
def birds_per_snake : ℕ := 3
def snakes_per_jaguar : ℕ := 5
def num_jaguars : ℕ := 6
def total_beetles : ℕ := 1080

-- Define the theorem
theorem birds_beetles_per_day :
  ∀ (beetles_per_bird : ℕ),
    beetles_per_bird * (birds_per_snake * snakes_per_jaguar * num_jaguars) = total_beetles →
    beetles_per_bird = 12 := by
  sorry

end NUMINAMATH_CALUDE_birds_beetles_per_day_l244_24483


namespace NUMINAMATH_CALUDE_necessary_condition_for_inequality_l244_24466

theorem necessary_condition_for_inequality (a b : ℝ) :
  a * Real.sqrt a + b * Real.sqrt b > a * Real.sqrt b + b * Real.sqrt a →
  a ≥ 0 ∧ b ≥ 0 ∧ a ≠ b :=
by sorry

end NUMINAMATH_CALUDE_necessary_condition_for_inequality_l244_24466


namespace NUMINAMATH_CALUDE_min_snack_cost_l244_24437

/-- Calculates the minimum number of packs/bags needed given the number of items per pack/bag and the total number of items required -/
def min_packs_needed (items_per_pack : ℕ) (total_items_needed : ℕ) : ℕ :=
  (total_items_needed + items_per_pack - 1) / items_per_pack

/-- Represents the problem of buying snacks for soccer players -/
def snack_problem (num_players : ℕ) (juice_per_pack : ℕ) (juice_pack_cost : ℚ) 
                  (apples_per_bag : ℕ) (apple_bag_cost : ℚ) : ℚ :=
  let juice_packs := min_packs_needed juice_per_pack num_players
  let apple_bags := min_packs_needed apples_per_bag num_players
  juice_packs * juice_pack_cost + apple_bags * apple_bag_cost

/-- The theorem stating the minimum amount Danny spends -/
theorem min_snack_cost : 
  snack_problem 17 3 2 5 4 = 28 :=
sorry

end NUMINAMATH_CALUDE_min_snack_cost_l244_24437


namespace NUMINAMATH_CALUDE_angelina_walking_speed_l244_24465

/-- Angelina's walking problem -/
theorem angelina_walking_speed 
  (distance_home_grocery : ℝ) 
  (distance_grocery_gym : ℝ) 
  (speed_home_grocery : ℝ) 
  (speed_grocery_gym : ℝ) 
  (time_difference : ℝ) :
  distance_home_grocery = 150 →
  distance_grocery_gym = 200 →
  speed_grocery_gym = 2 * speed_home_grocery →
  distance_home_grocery / speed_home_grocery - 
    distance_grocery_gym / speed_grocery_gym = time_difference →
  time_difference = 10 →
  speed_grocery_gym = 10 := by
sorry


end NUMINAMATH_CALUDE_angelina_walking_speed_l244_24465


namespace NUMINAMATH_CALUDE_functional_equation_solution_l244_24401

-- Define the property that the function f must satisfy
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * y - 1) + f x * f y = 2 * x * y - 1

-- State the theorem
theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, SatisfiesFunctionalEquation f →
  (∀ x : ℝ, f x = x) ∨ (∀ x : ℝ, f x = -x^2) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l244_24401


namespace NUMINAMATH_CALUDE_initial_water_amount_l244_24434

/-- The amount of water initially in the bucket, in gallons. -/
def initial_water : ℝ := sorry

/-- The amount of water remaining in the bucket, in gallons. -/
def remaining_water : ℝ := 0.5

/-- The amount of water that leaked out of the bucket, in gallons. -/
def leaked_water : ℝ := 0.25

/-- Theorem stating that the initial amount of water is equal to 0.75 gallon. -/
theorem initial_water_amount : initial_water = 0.75 := by sorry

end NUMINAMATH_CALUDE_initial_water_amount_l244_24434


namespace NUMINAMATH_CALUDE_square_sum_geq_double_product_l244_24475

theorem square_sum_geq_double_product (a b : ℝ) : a^2 + b^2 ≥ 2*a*b := by
  sorry

end NUMINAMATH_CALUDE_square_sum_geq_double_product_l244_24475


namespace NUMINAMATH_CALUDE_scaled_prism_volume_scaled_54_cubic_feet_prism_l244_24478

/-- Theorem: Scaling a rectangular prism's volume -/
theorem scaled_prism_volume 
  (V : ℝ) 
  (a b c : ℝ) 
  (hV : V > 0) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hc : c > 0) : 
  a * b * c * V = (a * b * c) * V := by sorry

/-- Corollary: Specific case of scaling a 54 cubic feet prism -/
theorem scaled_54_cubic_feet_prism :
  let V : ℝ := 54
  let a : ℝ := 2
  let b : ℝ := 3
  let c : ℝ := 1.5
  a * b * c * V = 486 := by sorry

end NUMINAMATH_CALUDE_scaled_prism_volume_scaled_54_cubic_feet_prism_l244_24478


namespace NUMINAMATH_CALUDE_last_letter_151st_permutation_l244_24406

-- Define the word and its permutations
def word : String := "JOKING"
def num_permutations : Nat := 720

-- Define the dictionary order function (not implemented, just declared)
def dictionary_order (s1 s2 : String) : Bool :=
  sorry

-- Define a function to get the nth permutation in dictionary order
def nth_permutation (n : Nat) : String :=
  sorry

-- Define a function to get the last letter of a string
def last_letter (s : String) : Char :=
  sorry

-- The theorem to prove
theorem last_letter_151st_permutation :
  last_letter (nth_permutation 151) = 'O' :=
sorry

end NUMINAMATH_CALUDE_last_letter_151st_permutation_l244_24406


namespace NUMINAMATH_CALUDE_sum_f_at_one_equals_exp_e_l244_24444

noncomputable def f : ℕ → (ℝ → ℝ)
| 0 => fun x => Real.exp x
| (n + 1) => fun x => x * (deriv (f n)) x

theorem sum_f_at_one_equals_exp_e :
  (∑' n, (f n 1) / n.factorial) = Real.exp (Real.exp 1) := by sorry

end NUMINAMATH_CALUDE_sum_f_at_one_equals_exp_e_l244_24444


namespace NUMINAMATH_CALUDE_completing_square_quadratic_l244_24438

theorem completing_square_quadratic : 
  ∀ x : ℝ, 2 * x^2 - 3 * x - 1 = 0 ↔ (x - 3/4)^2 = 17/16 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_quadratic_l244_24438


namespace NUMINAMATH_CALUDE_perpendicular_lines_from_perpendicular_planes_l244_24426

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines and planes
variable (perp_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between planes
variable (perp_plane : Plane → Plane → Prop)

-- Define the perpendicular relation between lines
variable (perp_line : Line → Line → Prop)

-- Theorem statement
theorem perpendicular_lines_from_perpendicular_planes 
  (m n : Line) (α β : Plane) :
  perp_line_plane m α → 
  perp_line_plane n β → 
  perp_plane α β → 
  perp_line m n :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_from_perpendicular_planes_l244_24426


namespace NUMINAMATH_CALUDE_handshakes_count_l244_24485

/-- Represents a social gathering with specific group interactions -/
structure SocialGathering where
  total_people : Nat
  group1_size : Nat
  subgroup_size : Nat
  group2_size : Nat
  outsiders : Nat

/-- Calculates the number of handshakes in a social gathering -/
def handshakes (sg : SocialGathering) : Nat :=
  sg.subgroup_size * (sg.group2_size + sg.outsiders) +
  (sg.group1_size - sg.subgroup_size) * sg.outsiders +
  sg.group2_size * sg.outsiders

/-- Theorem stating the number of handshakes in the specific social gathering -/
theorem handshakes_count :
  let sg : SocialGathering := {
    total_people := 36,
    group1_size := 25,
    subgroup_size := 15,
    group2_size := 6,
    outsiders := 5
  }
  handshakes sg = 245 := by sorry

end NUMINAMATH_CALUDE_handshakes_count_l244_24485


namespace NUMINAMATH_CALUDE_quadratic_condition_l244_24439

theorem quadratic_condition (a b c : ℝ) :
  (∀ x : ℝ, a * x^2 + b * x + c > 0) →
  (a > 0 ∧ b^2 - 4*a*c < 0) ∧
  ¬(a > 0 ∧ b^2 - 4*a*c < 0 → ∀ x : ℝ, a * x^2 + b * x + c > 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_condition_l244_24439


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l244_24414

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (x + 3) = 7 → x = 46 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l244_24414


namespace NUMINAMATH_CALUDE_units_digit_37_pow_37_l244_24459

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The units digit of 37^37 is 7 -/
theorem units_digit_37_pow_37 : unitsDigit (37^37) = 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_37_pow_37_l244_24459


namespace NUMINAMATH_CALUDE_two_out_of_three_win_probability_l244_24477

/-- The probability that exactly two out of three players win a game, given their individual probabilities of success. -/
theorem two_out_of_three_win_probability
  (p_alice : ℚ) (p_benjamin : ℚ) (p_carol : ℚ)
  (h_alice : p_alice = 1/5)
  (h_benjamin : p_benjamin = 3/8)
  (h_carol : p_carol = 2/7) :
  (p_alice * p_benjamin * (1 - p_carol)) +
  (p_alice * p_carol * (1 - p_benjamin)) +
  (p_benjamin * p_carol * (1 - p_alice)) = 49/280 := by
  sorry

end NUMINAMATH_CALUDE_two_out_of_three_win_probability_l244_24477


namespace NUMINAMATH_CALUDE_z_eighth_power_equals_one_l244_24404

theorem z_eighth_power_equals_one :
  let z : ℂ := (-Real.sqrt 3 - I) / 2
  z^8 = 1 := by sorry

end NUMINAMATH_CALUDE_z_eighth_power_equals_one_l244_24404


namespace NUMINAMATH_CALUDE_min_tiles_needed_is_260_l244_24469

/-- Represents the dimensions of a rectangle in inches -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangle given its dimensions -/
def area (d : Dimensions) : ℕ := d.length * d.width

/-- Converts feet to inches -/
def feetToInches (feet : ℕ) : ℕ := feet * 12

/-- The dimensions of the tile -/
def tileDimensions : Dimensions := ⟨2, 5⟩

/-- The dimensions of the floor in feet -/
def floorDimensionsFeet : Dimensions := ⟨3, 6⟩

/-- The dimensions of the floor in inches -/
def floorDimensionsInches : Dimensions :=
  ⟨feetToInches floorDimensionsFeet.length, feetToInches floorDimensionsFeet.width⟩

/-- Calculates the minimum number of tiles needed to cover the floor -/
def minTilesNeeded : ℕ :=
  (area floorDimensionsInches + area tileDimensions - 1) / area tileDimensions

theorem min_tiles_needed_is_260 : minTilesNeeded = 260 := by
  sorry

end NUMINAMATH_CALUDE_min_tiles_needed_is_260_l244_24469


namespace NUMINAMATH_CALUDE_cersei_cotton_candies_l244_24486

/-- The number of cotton candies Cersei initially bought -/
def initial_candies : ℕ := 40

/-- The number of cotton candies given to brother and sister -/
def given_to_siblings : ℕ := 10

/-- The fraction of remaining candies given to cousin -/
def fraction_to_cousin : ℚ := 1/4

/-- The number of cotton candies Cersei ate -/
def eaten_candies : ℕ := 12

/-- The number of cotton candies left at the end -/
def remaining_candies : ℕ := 18

theorem cersei_cotton_candies : 
  initial_candies = 40 ∧
  (initial_candies - given_to_siblings) * (1 - fraction_to_cousin) - eaten_candies = remaining_candies :=
by sorry

end NUMINAMATH_CALUDE_cersei_cotton_candies_l244_24486


namespace NUMINAMATH_CALUDE_suitable_pairs_solution_l244_24457

def suitable_pair (a b : ℕ+) : Prop := (a + b) ∣ (a * b)

def pairs : List (ℕ+ × ℕ+) := [
  (3, 6), (4, 12), (5, 20), (6, 30), (7, 42), (8, 56),
  (9, 72), (10, 90), (11, 110), (12, 132), (13, 156), (14, 168)
]

theorem suitable_pairs_solution :
  (∀ (p : ℕ+ × ℕ+), p ∈ pairs → suitable_pair p.1 p.2) ∧
  (pairs.length = 12) ∧
  (∀ (n : ℕ+), (n ∈ pairs.map Prod.fst ∨ n ∈ pairs.map Prod.snd) →
    (pairs.map Prod.fst ++ pairs.map Prod.snd).count n = 1) ∧
  (∀ (p : ℕ+ × ℕ+), p ∉ pairs → p.1 ≤ 168 ∧ p.2 ≤ 168 → ¬suitable_pair p.1 p.2) :=
by sorry

end NUMINAMATH_CALUDE_suitable_pairs_solution_l244_24457


namespace NUMINAMATH_CALUDE_triangle_equilateral_iff_area_condition_l244_24487

/-- Triangle with vertices A₁, A₂, A₃ -/
structure Triangle where
  A₁ : ℝ × ℝ
  A₂ : ℝ × ℝ
  A₃ : ℝ × ℝ

/-- Altitude of a triangle from a vertex to the opposite side -/
def altitude (T : Triangle) (i : Fin 3) : ℝ := sorry

/-- Area of a triangle -/
def area (T : Triangle) : ℝ := sorry

/-- Length of a side of a triangle -/
def sideLength (T : Triangle) (i j : Fin 3) : ℝ := sorry

/-- A triangle is equilateral if all its sides have equal length -/
def isEquilateral (T : Triangle) : Prop :=
  sideLength T 0 1 = sideLength T 1 2 ∧ sideLength T 1 2 = sideLength T 2 0

/-- Main theorem: A triangle is equilateral iff its area satisfies the given condition -/
theorem triangle_equilateral_iff_area_condition (T : Triangle) :
  isEquilateral T ↔ 
    area T = (1/6) * (sideLength T 0 1 * altitude T 0 + 
                      sideLength T 1 2 * altitude T 1 + 
                      sideLength T 2 0 * altitude T 2) :=
sorry

end NUMINAMATH_CALUDE_triangle_equilateral_iff_area_condition_l244_24487


namespace NUMINAMATH_CALUDE_part_to_whole_ratio_l244_24429

theorem part_to_whole_ratio (N : ℝ) (x : ℝ) (h1 : N = 160) (h2 : x + 4 = (N/4) - 4) : x / N = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_part_to_whole_ratio_l244_24429


namespace NUMINAMATH_CALUDE_max_k_no_intersection_l244_24494

noncomputable def f (x : ℝ) : ℝ := x - 1 + (Real.exp x)⁻¹

theorem max_k_no_intersection : 
  (∃ k : ℝ, ∀ x : ℝ, f x ≠ k * x - 1) ∧ 
  (∀ k : ℝ, k > 1 → ∃ x : ℝ, f x = k * x - 1) :=
sorry

end NUMINAMATH_CALUDE_max_k_no_intersection_l244_24494


namespace NUMINAMATH_CALUDE_line_and_symmetric_line_equations_l244_24473

-- Define the lines
def line1 (x y : ℝ) := 3*x + 4*y - 2 = 0
def line2 (x y : ℝ) := 2*x + y + 2 = 0
def line3 (x y : ℝ) := x - 2*y - 1 = 0

-- Define the intersection point P
def P : ℝ × ℝ := (-2, 2)

-- Define line l
def l (x y : ℝ) := 2*x + y + 2 = 0

-- Define the symmetric line
def sym_l (x y : ℝ) := 2*x + y - 2 = 0

-- Theorem statement
theorem line_and_symmetric_line_equations :
  (∀ x y : ℝ, line1 x y ∧ line2 x y → (x, y) = P) ∧  -- P is the intersection of line1 and line2
  (∀ x y : ℝ, l x y → line3 (y - P.2) (P.1 - x)) ∧   -- l is perpendicular to line3
  (l P.1 P.2) ∧                                      -- l passes through P
  (∀ x y : ℝ, sym_l x y ↔ l (-x) (-y)) :=            -- sym_l is symmetric to l with respect to origin
by sorry

end NUMINAMATH_CALUDE_line_and_symmetric_line_equations_l244_24473


namespace NUMINAMATH_CALUDE_opposite_numbers_and_cube_root_l244_24408

theorem opposite_numbers_and_cube_root (a b c : ℝ) : 
  (a + b = 0) → (c^3 = 8) → (2*a + 2*b - c = -2) := by
sorry

end NUMINAMATH_CALUDE_opposite_numbers_and_cube_root_l244_24408


namespace NUMINAMATH_CALUDE_no_duplicates_on_diagonal_l244_24463

/-- Represents a symmetric table with specific properties -/
structure SymmetricTable :=
  (size : Nat)
  (values : Fin size → Fin size → Fin size)
  (symmetric : ∀ i j, values i j = values j i)
  (distinct_rows : ∀ i j k, j ≠ k → values i j ≠ values i k)

/-- The main theorem stating that there are no duplicate numbers on the diagonal of symmetry -/
theorem no_duplicates_on_diagonal (t : SymmetricTable) (h : t.size = 101) :
  ∀ i j, i ≠ j → t.values i i ≠ t.values j j := by
  sorry

end NUMINAMATH_CALUDE_no_duplicates_on_diagonal_l244_24463


namespace NUMINAMATH_CALUDE_range_of_f_l244_24492

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 4*x + 5

-- Define the domain
def domain : Set ℝ := {x | 1 ≤ x ∧ x ≤ 4}

-- Theorem statement
theorem range_of_f :
  {y | ∃ x ∈ domain, f x = y} = {y | 1 ≤ y ∧ y ≤ 5} :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l244_24492


namespace NUMINAMATH_CALUDE_sum_of_products_of_roots_l244_24451

theorem sum_of_products_of_roots (p q r : ℂ) : 
  (2 * p^3 + p^2 - 7*p + 2 = 0) → 
  (2 * q^3 + q^2 - 7*q + 2 = 0) → 
  (2 * r^3 + r^2 - 7*r + 2 = 0) → 
  p * q + q * r + r * p = -7/2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_products_of_roots_l244_24451


namespace NUMINAMATH_CALUDE_largest_n_value_l244_24449

/-- Represents a digit in base 5 -/
def Base5Digit := Fin 5

/-- Represents a digit in base 9 -/
def Base9Digit := Fin 9

/-- Converts a three-digit number in base 5 to base 10 -/
def base5ToBase10 (x y z : Base5Digit) : ℕ :=
  25 * x.val + 5 * y.val + z.val

/-- Converts a three-digit number in base 9 to base 10 -/
def base9ToBase10 (z y x : Base9Digit) : ℕ :=
  81 * z.val + 9 * y.val + x.val

theorem largest_n_value (n : ℕ) 
  (h1 : ∃ (x y z : Base5Digit), n = base5ToBase10 x y z)
  (h2 : ∃ (x y z : Base9Digit), n = base9ToBase10 z y x) :
  n ≤ 121 ∧ ∃ (x y z : Base5Digit), 121 = base5ToBase10 x y z ∧ 
    ∃ (x y z : Base9Digit), 121 = base9ToBase10 z y x :=
by sorry

end NUMINAMATH_CALUDE_largest_n_value_l244_24449


namespace NUMINAMATH_CALUDE_direction_vector_b_value_l244_24461

/-- Given a line passing through points (-1, 3) and (2, 7) with direction vector (2, b), prove that b = 8/3 -/
theorem direction_vector_b_value (b : ℚ) : 
  let p1 : ℚ × ℚ := (-1, 3)
  let p2 : ℚ × ℚ := (2, 7)
  let direction_vector : ℚ × ℚ := (2, b)
  (∃ (k : ℚ), k • (p2.1 - p1.1, p2.2 - p1.2) = direction_vector) →
  b = 8/3 := by
sorry

end NUMINAMATH_CALUDE_direction_vector_b_value_l244_24461


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l244_24455

/-- Two vectors are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_m_value :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (-1, m)
  parallel a b → m = -2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l244_24455


namespace NUMINAMATH_CALUDE_fractional_equation_solution_range_l244_24411

theorem fractional_equation_solution_range (m : ℝ) : 
  (∃ x : ℝ, x ≥ 0 ∧ x ≠ 2 ∧ (m / (x - 2) + 1 = x / (2 - x))) ↔ 
  (m ≤ 2 ∧ m ≠ -2) :=
by sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_range_l244_24411


namespace NUMINAMATH_CALUDE_solution_set_inequality_l244_24474

theorem solution_set_inequality (x : ℝ) : 
  (x - 2) * (3 - x) > 0 ↔ x ∈ Set.Ioo 2 3 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l244_24474


namespace NUMINAMATH_CALUDE_expression_value_l244_24499

def numerator : ℤ := 20 - 19 + 18 - 17 + 16 - 15 + 14 - 13 + 12 - 11 + 10 - 9 + 8 - 7 + 6 - 5 + 4 - 3 + 2 - 1

def denominator : ℤ := 1 - 2 + 3 - 4 + 5 - 6 + 7 - 8 + 9 - 10 + 11 - 12 + 13 - 14 + 15 - 16 + 17 - 18 + 19 - 20

theorem expression_value : (numerator : ℚ) / denominator = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l244_24499


namespace NUMINAMATH_CALUDE_toys_per_day_l244_24448

/-- Given a factory that produces toys, this theorem proves the number of toys produced each day. -/
theorem toys_per_day 
  (total_toys : ℕ)           -- Total number of toys produced per week
  (work_days : ℕ)            -- Number of work days per week
  (h1 : total_toys = 4560)   -- The factory produces 4560 toys per week
  (h2 : work_days = 4)       -- Workers work 4 days a week
  (h3 : total_toys % work_days = 0)  -- The number of toys produced is the same each day
  : total_toys / work_days = 1140 := by
  sorry

#check toys_per_day

end NUMINAMATH_CALUDE_toys_per_day_l244_24448


namespace NUMINAMATH_CALUDE_odd_digits_base4_157_l244_24441

/-- Converts a natural number to its base-4 representation -/
def toBase4 (n : ℕ) : List ℕ :=
  sorry

/-- Counts the number of odd digits in a list of digits -/
def countOddDigits (digits : List ℕ) : ℕ :=
  sorry

/-- The number of odd digits in the base-4 representation of 157 is 3 -/
theorem odd_digits_base4_157 : countOddDigits (toBase4 157) = 3 := by
  sorry

end NUMINAMATH_CALUDE_odd_digits_base4_157_l244_24441


namespace NUMINAMATH_CALUDE_smallest_n_multiple_of_13_l244_24489

theorem smallest_n_multiple_of_13 (x y : ℤ) 
  (h1 : (2 * x - 3) % 13 = 0) 
  (h2 : (3 * y + 4) % 13 = 0) : 
  ∃ n : ℕ+, (x^2 - x*y + y^2 + n) % 13 = 0 ∧ 
  ∀ m : ℕ+, m < n → (x^2 - x*y + y^2 + m) % 13 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_multiple_of_13_l244_24489


namespace NUMINAMATH_CALUDE_necklace_arrangement_count_l244_24484

/-- The number of distinct circular arrangements of balls in a necklace -/
def necklace_arrangements (red : ℕ) (green : ℕ) (yellow : ℕ) : ℕ :=
  let total := red + green + yellow
  let linear_arrangements := Nat.choose (total - 1) red * Nat.choose (total - 1 - red) yellow
  (linear_arrangements - Nat.choose (total / 2) (red / 2)) / 2 + Nat.choose (total / 2) (red / 2)

/-- Theorem stating the number of distinct arrangements for the given problem -/
theorem necklace_arrangement_count :
  necklace_arrangements 6 1 8 = 1519 := by
  sorry

#eval necklace_arrangements 6 1 8

end NUMINAMATH_CALUDE_necklace_arrangement_count_l244_24484


namespace NUMINAMATH_CALUDE_eight_by_eight_unfolds_to_nine_l244_24495

/-- Represents a square grid -/
structure Grid :=
  (size : ℕ)

/-- Represents a folded and cut grid -/
structure FoldedCutGrid :=
  (original : Grid)
  (folded_size : ℕ)
  (cut : Bool)

/-- Counts the number of parts after unfolding a cut grid -/
def count_parts (g : FoldedCutGrid) : ℕ :=
  sorry

/-- Theorem stating that an 8x8 grid folded to 1x1 and cut unfolds into 9 parts -/
theorem eight_by_eight_unfolds_to_nine :
  ∀ (g : FoldedCutGrid),
    g.original.size = 8 →
    g.folded_size = 1 →
    g.cut = true →
    count_parts g = 9 :=
  sorry

end NUMINAMATH_CALUDE_eight_by_eight_unfolds_to_nine_l244_24495


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l244_24470

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 - a*x + 2*a > 0) ↔ (0 < a ∧ a < 8) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l244_24470


namespace NUMINAMATH_CALUDE_complex_on_line_with_magnitude_l244_24421

theorem complex_on_line_with_magnitude (z : ℂ) :
  (z.im = 2 * z.re) → (Complex.abs z = Real.sqrt 5) →
  (z = Complex.mk 1 2 ∨ z = Complex.mk (-1) (-2)) := by
  sorry

end NUMINAMATH_CALUDE_complex_on_line_with_magnitude_l244_24421


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l244_24405

/-- The lateral surface area of a cone with base radius 3 and slant height 5 is 15π. -/
theorem cone_lateral_surface_area : 
  ∀ (r l : ℝ), r = 3 → l = 5 → π * r * l = 15 * π := by
  sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l244_24405


namespace NUMINAMATH_CALUDE_range_of_a_l244_24435

def A : Set ℝ := {x | x^2 + 4*x = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + 2*(a+1)*x + a^2 - 1 = 0}

theorem range_of_a (a : ℝ) : A ∩ B a = B a → a = 1 ∨ a ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l244_24435


namespace NUMINAMATH_CALUDE_sum_of_a_and_d_l244_24410

theorem sum_of_a_and_d (a b c d : ℤ) 
  (eq1 : a + b = 5)
  (eq2 : b + c = 6)
  (eq3 : c + d = 3) :
  a + d = 2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_a_and_d_l244_24410


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l244_24460

theorem sufficient_but_not_necessary :
  (∀ x : ℝ, 0 < x ∧ x < 5 → |x - 2| < 3) ∧
  (∃ x : ℝ, |x - 2| < 3 ∧ ¬(0 < x ∧ x < 5)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l244_24460


namespace NUMINAMATH_CALUDE_reciprocal_roots_quadratic_l244_24436

theorem reciprocal_roots_quadratic (a b : ℂ) : 
  (a^2 + 4*a + 8 = 0) ∧ (b^2 + 4*b + 8 = 0) → 
  (8*(1/a)^2 + 4*(1/a) + 1 = 0) ∧ (8*(1/b)^2 + 4*(1/b) + 1 = 0) := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_roots_quadratic_l244_24436


namespace NUMINAMATH_CALUDE_negation_of_existence_l244_24431

theorem negation_of_existence (Z : Type) [Ring Z] : 
  (¬ ∃ x : Z, x^2 = 2*x) ↔ (∀ x : Z, x^2 ≠ 2*x) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_l244_24431


namespace NUMINAMATH_CALUDE_factoring_a_squared_minus_nine_l244_24471

theorem factoring_a_squared_minus_nine (a : ℝ) : a^2 - 9 = (a + 3) * (a - 3) := by
  sorry

end NUMINAMATH_CALUDE_factoring_a_squared_minus_nine_l244_24471


namespace NUMINAMATH_CALUDE_arithmetic_progression_cosine_squared_l244_24467

open Real

theorem arithmetic_progression_cosine_squared (x y z : ℝ) (α : ℝ) :
  (∃ k : ℝ, y = x + α ∧ z = y + α) →  -- x, y, z form an arithmetic progression
  α = arccos (-1/3) →                 -- common difference
  (∃ m : ℝ, 3 / cos y = 1 / cos x + m ∧ 1 / cos z = 3 / cos y + m) →  -- 1/cos(x), 3/cos(y), 1/cos(z) form an AP
  cos y ^ 2 = 4/5 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_cosine_squared_l244_24467


namespace NUMINAMATH_CALUDE_students_play_both_football_and_cricket_l244_24442

/-- Represents the number of students who play both football and cricket -/
def students_play_both (total students_football students_cricket students_neither : ℕ) : ℕ :=
  students_football + students_cricket - (total - students_neither)

/-- Theorem stating that given the conditions, 140 students play both football and cricket -/
theorem students_play_both_football_and_cricket :
  students_play_both 410 325 175 50 = 140 := by
  sorry

end NUMINAMATH_CALUDE_students_play_both_football_and_cricket_l244_24442


namespace NUMINAMATH_CALUDE_square_difference_169_168_l244_24456

theorem square_difference_169_168 : (169 : ℕ)^2 - (168 : ℕ)^2 = 337 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_169_168_l244_24456


namespace NUMINAMATH_CALUDE_stock_price_problem_l244_24482

theorem stock_price_problem (initial_price : ℝ) : 
  let day1 := initial_price * (1 + 1/10)
  let day2 := day1 * (1 - 1/11)
  let day3 := day2 * (1 + 1/12)
  let day4 := day3 * (1 - 1/13)
  day4 = 5000 → initial_price = 5000 := by
sorry

end NUMINAMATH_CALUDE_stock_price_problem_l244_24482


namespace NUMINAMATH_CALUDE_geometric_series_common_ratio_l244_24453

/-- The common ratio of the geometric series 7/8 - 35/72 + 175/432 - ... is -5/9 -/
theorem geometric_series_common_ratio : 
  let a₁ : ℚ := 7/8
  let a₂ : ℚ := -35/72
  let a₃ : ℚ := 175/432
  let r := a₂ / a₁
  r = -5/9 := by sorry

end NUMINAMATH_CALUDE_geometric_series_common_ratio_l244_24453


namespace NUMINAMATH_CALUDE_defect_selection_probability_l244_24445

/-- Given a set of tubes with defects, calculate the probability of selecting specific defect types --/
theorem defect_selection_probability
  (total_tubes : ℕ)
  (type_a_defects : ℕ)
  (type_b_defects : ℕ)
  (h1 : total_tubes = 50)
  (h2 : type_a_defects = 5)
  (h3 : type_b_defects = 3)
  : ℚ :=
  3 / 490

#check defect_selection_probability

end NUMINAMATH_CALUDE_defect_selection_probability_l244_24445


namespace NUMINAMATH_CALUDE_inverse_mod_53_l244_24420

theorem inverse_mod_53 (h : (17⁻¹ : ZMod 53) = 11) : (36⁻¹ : ZMod 53) = 42 := by
  sorry

end NUMINAMATH_CALUDE_inverse_mod_53_l244_24420


namespace NUMINAMATH_CALUDE_collinear_points_d_values_l244_24446

/-- Four points in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Define the four points -/
def p1 (a : ℝ) : Point3D := ⟨2, 0, a⟩
def p2 (b : ℝ) : Point3D := ⟨b, 2, 0⟩
def p3 (c : ℝ) : Point3D := ⟨0, c, 2⟩
def p4 (d : ℝ) : Point3D := ⟨8*d, 8*d, -2*d⟩

/-- Define collinearity for four points -/
def collinear (p q r s : Point3D) : Prop :=
  ∃ (t₁ t₂ t₃ : ℝ), (q.x - p.x, q.y - p.y, q.z - p.z) = t₁ • (r.x - p.x, r.y - p.y, r.z - p.z)
                 ∧ (q.x - p.x, q.y - p.y, q.z - p.z) = t₂ • (s.x - p.x, s.y - p.y, s.z - p.z)
                 ∧ (r.x - p.x, r.y - p.y, r.z - p.z) = t₃ • (s.x - p.x, s.y - p.y, s.z - p.z)

/-- The main theorem -/
theorem collinear_points_d_values (a b c d : ℝ) :
  collinear (p1 a) (p2 b) (p3 c) (p4 d) → d = 1/8 ∨ d = -1/32 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_d_values_l244_24446


namespace NUMINAMATH_CALUDE_gcd_282_470_l244_24468

theorem gcd_282_470 : Nat.gcd 282 470 = 94 := by
  sorry

end NUMINAMATH_CALUDE_gcd_282_470_l244_24468

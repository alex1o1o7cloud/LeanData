import Mathlib

namespace NUMINAMATH_CALUDE_trig_expression_simplification_l284_28406

theorem trig_expression_simplification :
  (Real.tan (40 * π / 180) + Real.tan (50 * π / 180) + Real.tan (60 * π / 180) + Real.tan (70 * π / 180)) / Real.cos (30 * π / 180) =
  2 * (Real.cos (60 * π / 180) * Real.cos (70 * π / 180) + Real.sin (50 * π / 180) * Real.cos (40 * π / 180) * Real.cos (50 * π / 180)) /
  (Real.sqrt 3 * Real.cos (40 * π / 180) * Real.cos (50 * π / 180) * Real.cos (60 * π / 180) * Real.cos (70 * π / 180)) := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_simplification_l284_28406


namespace NUMINAMATH_CALUDE_point_movement_l284_28441

/-- Given a point A with coordinates (2, 1) in the Cartesian coordinate system,
    prove that moving it 3 units left and 1 unit up results in coordinates (-1, 2) for point A'. -/
theorem point_movement (A : ℝ × ℝ) (h1 : A = (2, 1)) :
  let A' := (A.1 - 3, A.2 + 1)
  A' = (-1, 2) := by
  sorry

end NUMINAMATH_CALUDE_point_movement_l284_28441


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l284_28408

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_ratio 
  (a : ℕ → ℝ) 
  (h_geo : geometric_sequence a) 
  (h_a1 : a 1 = 2) 
  (h_a4 : a 4 = 1/4) : 
  ∃ q : ℝ, (∀ n : ℕ, a (n + 1) = a n * q) ∧ q = 1/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l284_28408


namespace NUMINAMATH_CALUDE_complex_modulus_l284_28415

theorem complex_modulus (z : ℂ) (h : z * (1 + Complex.I) = Complex.I ^ 2016) : 
  Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l284_28415


namespace NUMINAMATH_CALUDE_vector_problem_l284_28472

/-- Define a 2D vector -/
def Vector2D := ℝ × ℝ

/-- Check if two vectors are collinear -/
def collinear (v w : Vector2D) : Prop :=
  ∃ (k : ℝ), v.1 = k * w.1 ∧ v.2 = k * w.2

/-- Dot product of two vectors -/
def dot_product (v w : Vector2D) : ℝ :=
  v.1 * w.1 + v.2 * w.2

/-- Check if two vectors are perpendicular -/
def perpendicular (v w : Vector2D) : Prop :=
  dot_product v w = 0

/-- The main theorem -/
theorem vector_problem :
  ∀ (m : ℝ),
  let a : Vector2D := (2, 1)
  let b : Vector2D := (3, -1)
  let c : Vector2D := (3, m)
  (collinear a c → m = 3/2) ∧
  (perpendicular (a.1 - 2*b.1, a.2 - 2*b.2) c → m = 4) :=
sorry

end NUMINAMATH_CALUDE_vector_problem_l284_28472


namespace NUMINAMATH_CALUDE_cross_section_properties_l284_28481

/-- Regular triangular prism with given dimensions -/
structure RegularTriangularPrism where
  base_side_length : ℝ
  height : ℝ

/-- Cross-section of the prism -/
structure CrossSection where
  area : ℝ
  angle_with_base : ℝ

/-- Theorem about the cross-section of a specific regular triangular prism -/
theorem cross_section_properties (prism : RegularTriangularPrism) 
  (h1 : prism.base_side_length = 6)
  (h2 : prism.height = (1/3) * Real.sqrt 7) :
  ∃ (cs : CrossSection), cs.area = 39/4 ∧ cs.angle_with_base = 30 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_cross_section_properties_l284_28481


namespace NUMINAMATH_CALUDE_stock_market_value_l284_28474

/-- Calculates the market value of a stock given its income, interest rate, and brokerage fee. -/
def market_value (income : ℚ) (interest_rate : ℚ) (brokerage_rate : ℚ) : ℚ :=
  let face_value := (income * 100) / interest_rate
  let brokerage_fee := (face_value / 100) * brokerage_rate
  face_value - brokerage_fee

/-- Theorem stating that the market value of the stock is 7182 given the specified conditions. -/
theorem stock_market_value :
  market_value 756 10.5 0.25 = 7182 :=
by sorry

end NUMINAMATH_CALUDE_stock_market_value_l284_28474


namespace NUMINAMATH_CALUDE_miranda_rearrangement_time_l284_28443

/-- Calculates the time in hours to write all rearrangements of a name -/
def time_to_write_rearrangements (num_letters : ℕ) (rearrangements_per_minute : ℕ) : ℚ :=
  (Nat.factorial num_letters : ℚ) / (rearrangements_per_minute : ℚ) / 60

/-- Proves that writing all rearrangements of a 6-letter name at 15 per minute takes 0.8 hours -/
theorem miranda_rearrangement_time :
  time_to_write_rearrangements 6 15 = 4/5 := by
  sorry

#eval time_to_write_rearrangements 6 15

end NUMINAMATH_CALUDE_miranda_rearrangement_time_l284_28443


namespace NUMINAMATH_CALUDE_cylinder_volume_l284_28483

/-- Given a cylinder with lateral surface area 100π cm² and an inscribed rectangular solid
    with diagonal length 10√2 cm, prove that the cylinder's volume is 250π cm³. -/
theorem cylinder_volume (r h : ℝ) (lateral_area : 2 * Real.pi * r * h = 100 * Real.pi)
    (diagonal_length : 4 * r^2 + h^2 = 200) : Real.pi * r^2 * h = 250 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_cylinder_volume_l284_28483


namespace NUMINAMATH_CALUDE_time_to_see_all_animals_after_import_l284_28400

/-- Calculates the time required to see all animal types after importing new species -/
def time_to_see_all_animals (initial_types : ℕ) (time_per_type : ℕ) (new_species : ℕ) : ℕ :=
  (initial_types + new_species) * time_per_type

/-- Proves that the time required to see all animal types after importing new species is 54 minutes -/
theorem time_to_see_all_animals_after_import :
  time_to_see_all_animals 5 6 4 = 54 := by
  sorry

end NUMINAMATH_CALUDE_time_to_see_all_animals_after_import_l284_28400


namespace NUMINAMATH_CALUDE_prime_root_pairs_classification_l284_28444

/-- A pair of positive primes (p,q) such that 3x^2 - px + q = 0 has two distinct rational roots -/
structure PrimeRootPair where
  p : ℕ
  q : ℕ
  p_prime : Nat.Prime p
  q_prime : Nat.Prime q
  has_distinct_rational_roots : ∃ (x y : ℚ), x ≠ y ∧ 3 * x^2 - p * x + q = 0 ∧ 3 * y^2 - p * y + q = 0

/-- The theorem stating that there are only two pairs of primes satisfying the condition -/
theorem prime_root_pairs_classification : 
  {pair : PrimeRootPair | True} = {⟨5, 2, sorry, sorry, sorry⟩, ⟨7, 2, sorry, sorry, sorry⟩} :=
by sorry

end NUMINAMATH_CALUDE_prime_root_pairs_classification_l284_28444


namespace NUMINAMATH_CALUDE_other_number_is_99_l284_28470

/-- Given two positive integers with specific HCF and LCM, prove one is 99 when the other is 48 -/
theorem other_number_is_99 (a b : ℕ+) (h1 : Nat.gcd a b = 12) (h2 : Nat.lcm a b = 396) (h3 : a = 48) :
  b = 99 := by
  sorry

end NUMINAMATH_CALUDE_other_number_is_99_l284_28470


namespace NUMINAMATH_CALUDE_least_integer_with_digit_removal_property_l284_28446

theorem least_integer_with_digit_removal_property : ∃ (n : ℕ), 
  n > 0 ∧ 
  (n % 10 = 5 ∧ n / 10 = 9) ∧
  n = 19 * (n % 10) ∧
  (∀ m : ℕ, m > 0 → m < n → 
    (m % 10 ≠ 19 * (m / 10) ∨ m / 10 = 0)) := by
  sorry

end NUMINAMATH_CALUDE_least_integer_with_digit_removal_property_l284_28446


namespace NUMINAMATH_CALUDE_tangent_line_through_origin_l284_28460

/-- The function f(x) = x³ + x - 16 -/
def f (x : ℝ) : ℝ := x^3 + x - 16

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3*x^2 + 1

theorem tangent_line_through_origin (x₀ : ℝ) :
  (f' x₀ = 13 ∧ f x₀ = -f' x₀ * x₀) →
  (x₀ = -2 ∧ f x₀ = -26 ∧ ∀ x, f' x₀ * x = f' x₀ * x₀ + f x₀) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_through_origin_l284_28460


namespace NUMINAMATH_CALUDE_simplify_trigonometric_expression_evaluate_trigonometric_fraction_l284_28491

-- Part 1
theorem simplify_trigonometric_expression :
  (Real.sqrt (1 - 2 * Real.sin (20 * π / 180) * Real.cos (20 * π / 180))) /
  (Real.sin (160 * π / 180) - Real.sqrt (1 - Real.sin (20 * π / 180) ^ 2)) = -1 := by
  sorry

-- Part 2
theorem evaluate_trigonometric_fraction (α : Real) (h : Real.tan α = 1 / 3) :
  1 / (4 * Real.cos α ^ 2 - 6 * Real.sin α * Real.cos α) = 5 / 9 := by
  sorry

end NUMINAMATH_CALUDE_simplify_trigonometric_expression_evaluate_trigonometric_fraction_l284_28491


namespace NUMINAMATH_CALUDE_max_candy_count_l284_28494

/-- Represents the state of the board and candy count -/
structure BoardState where
  numbers : List Nat
  candy_count : Nat

/-- Combines two numbers on the board and updates the candy count -/
def combine_numbers (state : BoardState) (i j : Nat) : BoardState :=
  { numbers := (state.numbers.removeNth i).removeNth j ++ [state.numbers[i]! + state.numbers[j]!],
    candy_count := state.candy_count + state.numbers[i]! * state.numbers[j]! }

/-- Theorem: The maximum number of candies Karlson can eat is 300 -/
theorem max_candy_count :
  ∃ (final_state : BoardState),
    (final_state.numbers.length = 1) ∧
    (final_state.candy_count = 300) ∧
    (∃ (initial_state : BoardState),
      (initial_state.numbers = List.replicate 25 1) ∧
      (∃ (moves : List (Nat × Nat)),
        moves.length = 24 ∧
        final_state = moves.foldl (fun state (i, j) => combine_numbers state i j) initial_state)) :=
by
  sorry

#check max_candy_count

end NUMINAMATH_CALUDE_max_candy_count_l284_28494


namespace NUMINAMATH_CALUDE_special_function_at_five_l284_28450

/-- A function satisfying the given properties -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, f (x * y) = f x * f y) ∧ 
  (f 0 ≠ 0) ∧ 
  (f 1 = 2)

/-- Theorem stating that f(5) = 0 for any function satisfying the special properties -/
theorem special_function_at_five {f : ℝ → ℝ} (hf : special_function f) : f 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_special_function_at_five_l284_28450


namespace NUMINAMATH_CALUDE_condition_A_neither_necessary_nor_sufficient_l284_28459

/-- Condition A: √(1 + sin θ) = a -/
def condition_A (θ : Real) (a : Real) : Prop :=
  Real.sqrt (1 + Real.sin θ) = a

/-- Condition B: sin(θ/2) + cos(θ/2) = a -/
def condition_B (θ : Real) (a : Real) : Prop :=
  Real.sin (θ / 2) + Real.cos (θ / 2) = a

/-- Theorem stating that condition A is neither necessary nor sufficient for condition B -/
theorem condition_A_neither_necessary_nor_sufficient :
  ¬(∀ θ a, condition_A θ a ↔ condition_B θ a) ∧
  ¬(∀ θ a, condition_A θ a → condition_B θ a) ∧
  ¬(∀ θ a, condition_B θ a → condition_A θ a) := by
  sorry

end NUMINAMATH_CALUDE_condition_A_neither_necessary_nor_sufficient_l284_28459


namespace NUMINAMATH_CALUDE_chord_length_squared_l284_28485

/-- The square of the length of a chord that is a common external tangent to two circles -/
theorem chord_length_squared (r₁ r₂ R : ℝ) (h₁ : r₁ > 0) (h₂ : r₂ > 0) (h₃ : R > 0)
  (h₄ : r₁ + r₂ < R) : 
  let d := R - (r₁ + r₂) + Real.sqrt (r₁ * r₂)
  4 * (R^2 - d^2) = 516 :=
by sorry

end NUMINAMATH_CALUDE_chord_length_squared_l284_28485


namespace NUMINAMATH_CALUDE_ruler_cost_l284_28414

theorem ruler_cost (total_students : ℕ) (total_expense : ℕ) :
  total_students = 42 →
  total_expense = 2310 →
  ∃ (num_buyers : ℕ) (rulers_per_student : ℕ) (cost_per_ruler : ℕ),
    num_buyers > total_students / 2 ∧
    cost_per_ruler > rulers_per_student ∧
    num_buyers * rulers_per_student * cost_per_ruler = total_expense ∧
    cost_per_ruler = 11 :=
by sorry


end NUMINAMATH_CALUDE_ruler_cost_l284_28414


namespace NUMINAMATH_CALUDE_correct_divisor_l284_28404

theorem correct_divisor (X D : ℕ) 
  (h1 : X % D = 0)
  (h2 : X / (D - 12) = 42)
  (h3 : X / D = 24) :
  D = 28 := by
  sorry

end NUMINAMATH_CALUDE_correct_divisor_l284_28404


namespace NUMINAMATH_CALUDE_recommended_apps_proof_l284_28439

/-- The recommended number of apps for Roger's phone -/
def recommended_apps : ℕ := 35

/-- The maximum number of apps for optimal function -/
def max_optimal_apps : ℕ := 50

/-- The number of apps Roger currently has -/
def rogers_current_apps : ℕ := 2 * recommended_apps

/-- The number of apps Roger needs to delete -/
def apps_to_delete : ℕ := 20

theorem recommended_apps_proof :
  (rogers_current_apps = max_optimal_apps + apps_to_delete) ∧
  (rogers_current_apps = 2 * recommended_apps) ∧
  (max_optimal_apps = 50) ∧
  (apps_to_delete = 20) →
  recommended_apps = 35 := by sorry

end NUMINAMATH_CALUDE_recommended_apps_proof_l284_28439


namespace NUMINAMATH_CALUDE_sum_not_prime_l284_28492

theorem sum_not_prime (a b c x y z : ℕ+) (h1 : a * x * y = b * y * z) (h2 : b * y * z = c * z * x) :
  ∃ (k m : ℕ+), a + b + c + x + y + z = k * m ∧ k ≠ 1 ∧ m ≠ 1 :=
sorry

end NUMINAMATH_CALUDE_sum_not_prime_l284_28492


namespace NUMINAMATH_CALUDE_max_age_on_aubrey_birthday_l284_28428

/-- The age difference between Luka and Aubrey -/
def age_difference : ℕ := 2

/-- Luka's age when Max was born -/
def luka_age_at_max_birth : ℕ := 4

/-- Aubrey's age for which we want to find Max's age -/
def aubrey_target_age : ℕ := 8

/-- Max's age when Aubrey reaches the target age -/
def max_age : ℕ := aubrey_target_age - age_difference

theorem max_age_on_aubrey_birthday :
  max_age = 6 := by sorry

end NUMINAMATH_CALUDE_max_age_on_aubrey_birthday_l284_28428


namespace NUMINAMATH_CALUDE_cubic_equation_root_l284_28431

theorem cubic_equation_root (h : ℚ) : 
  (3 : ℚ)^3 + h * 3 - 14 = 0 → h = -13/3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_root_l284_28431


namespace NUMINAMATH_CALUDE_floor_ceil_sum_l284_28468

theorem floor_ceil_sum : ⌊(0.998 : ℝ)⌋ + ⌈(3.002 : ℝ)⌉ = 4 := by
  sorry

end NUMINAMATH_CALUDE_floor_ceil_sum_l284_28468


namespace NUMINAMATH_CALUDE_inequalities_always_true_l284_28442

theorem inequalities_always_true (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a > |a - b| - b) ∧ (a * b + 2 / (a * b) > 2) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_always_true_l284_28442


namespace NUMINAMATH_CALUDE_morning_fliers_fraction_l284_28436

theorem morning_fliers_fraction (total : ℕ) (remaining : ℕ) : 
  total = 2500 → remaining = 1500 → 
  ∃ x : ℚ, x > 0 ∧ x < 1 ∧ 
  (1 - x) * total - (1 - x) * total / 4 = remaining ∧
  x = 1/5 := by
sorry

end NUMINAMATH_CALUDE_morning_fliers_fraction_l284_28436


namespace NUMINAMATH_CALUDE_rem_evaluation_l284_28484

def rem (x y : ℚ) : ℚ := x - y * ⌊x / y⌋

theorem rem_evaluation :
  rem (7/12 : ℚ) (-3/4 : ℚ) = -1/6 := by
  sorry

end NUMINAMATH_CALUDE_rem_evaluation_l284_28484


namespace NUMINAMATH_CALUDE_shortest_side_theorem_l284_28456

theorem shortest_side_theorem (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 → 
  a + b > c → b + c > a → a + c > b → 
  a^2 + b^2 > 5*c^2 → 
  c < a ∧ c < b :=
sorry

end NUMINAMATH_CALUDE_shortest_side_theorem_l284_28456


namespace NUMINAMATH_CALUDE_unique_solution_mod_37_l284_28454

theorem unique_solution_mod_37 :
  ∃! (a b c d : ℤ),
    (a^2 + b*c) % 37 = a % 37 ∧
    (b*(a + d)) % 37 = b % 37 ∧
    (c*(a + d)) % 37 = c % 37 ∧
    (b*c + d^2) % 37 = d % 37 ∧
    (a*d - b*c) % 37 = 1 % 37 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_mod_37_l284_28454


namespace NUMINAMATH_CALUDE_decimal_sum_to_fraction_l284_28416

theorem decimal_sum_to_fraction :
  (0.2 : ℚ) + 0.03 + 0.004 + 0.0005 + 0.00006 = 733 / 3125 := by
  sorry

end NUMINAMATH_CALUDE_decimal_sum_to_fraction_l284_28416


namespace NUMINAMATH_CALUDE_orthocenter_of_triangle_l284_28426

/-- The orthocenter of a triangle in 3D space. -/
def orthocenter (A B C : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  sorry

/-- Theorem: The orthocenter of triangle ABC with given coordinates is (3/2, 5/2, 6). -/
theorem orthocenter_of_triangle :
  let A : ℝ × ℝ × ℝ := (2, 3, 4)
  let B : ℝ × ℝ × ℝ := (4, 4, 2)
  let C : ℝ × ℝ × ℝ := (3, 5, 6)
  orthocenter A B C = (3/2, 5/2, 6) := by
  sorry

end NUMINAMATH_CALUDE_orthocenter_of_triangle_l284_28426


namespace NUMINAMATH_CALUDE_parabola_vector_max_value_l284_28496

/-- The parabola C: x^2 = 4y -/
def parabola (p : ℝ × ℝ) : Prop := p.1^2 = 4 * p.2

/-- The line l intersecting the parabola at points A and B -/
def line_intersects (l : Set (ℝ × ℝ)) (A B : ℝ × ℝ) : Prop :=
  A ∈ l ∧ B ∈ l ∧ parabola A ∧ parabola B

/-- Vector from origin to a point -/
def vec_from_origin (p : ℝ × ℝ) : ℝ × ℝ := p

/-- Vector between two points -/
def vec_between (p q : ℝ × ℝ) : ℝ × ℝ := (q.1 - p.1, q.2 - p.2)

/-- Scalar multiplication of a vector -/
def scalar_mul (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)

/-- Vector equality -/
def vec_eq (v w : ℝ × ℝ) : Prop := v = w

/-- Dot product of two vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- The main theorem -/
theorem parabola_vector_max_value 
  (l : Set (ℝ × ℝ)) (A B G : ℝ × ℝ) :
  line_intersects l A B →
  vec_eq (vec_between A B) (scalar_mul 2 (vec_between A G)) →
  (∃ (max : ℝ), 
    max = 16 ∧ 
    ∀ (X Y : ℝ × ℝ), parabola X → parabola Y → 
      (dot_product (vec_from_origin X) (vec_from_origin X) +
       dot_product (vec_from_origin Y) (vec_from_origin Y) -
       2 * dot_product (vec_from_origin X) (vec_from_origin Y) -
       4 * dot_product (vec_from_origin G) (vec_from_origin G)) ≤ max) :=
sorry

end NUMINAMATH_CALUDE_parabola_vector_max_value_l284_28496


namespace NUMINAMATH_CALUDE_min_value_expression_l284_28482

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a ≤ b + c) (hbc : b ≤ a + c) (hca : c ≤ a + b) :
  c / (a + b) + b / c ≥ Real.sqrt 2 - 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l284_28482


namespace NUMINAMATH_CALUDE_mileage_difference_l284_28495

/-- Calculates the difference between advertised and actual mileage -/
theorem mileage_difference (advertised_mpg : ℝ) (tank_capacity : ℝ) (total_miles : ℝ) :
  advertised_mpg = 35 →
  tank_capacity = 12 →
  total_miles = 372 →
  advertised_mpg - (total_miles / tank_capacity) = 4 := by
  sorry


end NUMINAMATH_CALUDE_mileage_difference_l284_28495


namespace NUMINAMATH_CALUDE_garret_age_proof_l284_28461

/-- Garret's current age -/
def garret_age : ℕ := 12

/-- Shane's current age -/
def shane_current_age : ℕ := 44

theorem garret_age_proof :
  (shane_current_age - 20 = 2 * garret_age) →
  garret_age = 12 := by
sorry

end NUMINAMATH_CALUDE_garret_age_proof_l284_28461


namespace NUMINAMATH_CALUDE_fraction_equality_implies_zero_l284_28473

theorem fraction_equality_implies_zero (x : ℝ) : 
  (4 + x) / (6 + x) = (2 + x) / (3 + x) → x = 0 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_zero_l284_28473


namespace NUMINAMATH_CALUDE_min_value_squared_sum_l284_28432

theorem min_value_squared_sum (a b c : ℝ) (h : a + 2*b + 3*c = 6) :
  ∃ m : ℝ, m = 12 ∧ ∀ x y z : ℝ, x + 2*y + 3*z = 6 → x^2 + 4*y^2 + 9*z^2 ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_squared_sum_l284_28432


namespace NUMINAMATH_CALUDE_negation_of_existence_l284_28480

theorem negation_of_existence (a : ℝ) :
  (¬ ∃ x : ℝ, x^2 + (a - 1) * x + 1 < 0) ↔ (∀ x : ℝ, x^2 + (a - 1) * x + 1 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_l284_28480


namespace NUMINAMATH_CALUDE_find_b_l284_28445

theorem find_b : ∃ b : ℝ,
  let p : ℝ → ℝ := λ x ↦ 2 * x - 3
  let q : ℝ → ℝ := λ x ↦ 5 * x - b
  p (q 3) = 13 → b = 7 := by
  sorry

end NUMINAMATH_CALUDE_find_b_l284_28445


namespace NUMINAMATH_CALUDE_area_perimeter_ratio_equal_l284_28471

/-- An isosceles trapezoid inscribed in a circle -/
structure InscribedIsoscelesTrapezoid where
  /-- Radius of the circle -/
  R : ℝ
  /-- Perimeter of the trapezoid -/
  P : ℝ
  /-- Radius is positive -/
  R_pos : R > 0
  /-- Perimeter is positive -/
  P_pos : P > 0

/-- Theorem: The ratio of the area of the trapezoid to the area of the circle
    is equal to the ratio of the perimeter of the trapezoid to the circumference of the circle -/
theorem area_perimeter_ratio_equal
  (trap : InscribedIsoscelesTrapezoid) :
  (trap.P * trap.R / 2) / (Real.pi * trap.R^2) = trap.P / (2 * Real.pi * trap.R) :=
sorry

end NUMINAMATH_CALUDE_area_perimeter_ratio_equal_l284_28471


namespace NUMINAMATH_CALUDE_unique_two_digit_number_l284_28457

/-- A function that returns true if a number is a two-digit number -/
def isTwoDigit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- A function that returns true if a number is odd -/
def isOdd (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k + 1

/-- A function that returns true if a number is a multiple of 9 -/
def isMultipleOf9 (n : ℕ) : Prop := ∃ k : ℕ, n = 9 * k

/-- A function that returns the tens digit of a two-digit number -/
def tensDigit (n : ℕ) : ℕ := n / 10

/-- A function that returns the ones digit of a two-digit number -/
def onesDigit (n : ℕ) : ℕ := n % 10

/-- A function that returns true if a number is a perfect square -/
def isPerfectSquare (n : ℕ) : Prop := ∃ k : ℕ, n = k * k

theorem unique_two_digit_number :
  ∃! n : ℕ, isTwoDigit n ∧ isOdd n ∧ isMultipleOf9 n ∧
    isPerfectSquare (tensDigit n * onesDigit n) ∧ n = 99 :=
sorry

end NUMINAMATH_CALUDE_unique_two_digit_number_l284_28457


namespace NUMINAMATH_CALUDE_max_integers_with_pairwise_common_divisor_and_coprime_triples_l284_28458

theorem max_integers_with_pairwise_common_divisor_and_coprime_triples :
  (∃ (n : ℕ) (a : Fin n → ℕ), n ≥ 3 ∧
    (∀ i, a i < 5000) ∧
    (∀ i j, i ≠ j → ∃ d > 1, d ∣ a i ∧ d ∣ a j) ∧
    (∀ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k → Nat.gcd (a i) (Nat.gcd (a j) (a k)) = 1)) →
  (∀ (n : ℕ) (a : Fin n → ℕ), n ≥ 3 →
    (∀ i, a i < 5000) →
    (∀ i j, i ≠ j → ∃ d > 1, d ∣ a i ∧ d ∣ a j) →
    (∀ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k → Nat.gcd (a i) (Nat.gcd (a j) (a k)) = 1) →
    n ≤ 4) :=
by sorry

end NUMINAMATH_CALUDE_max_integers_with_pairwise_common_divisor_and_coprime_triples_l284_28458


namespace NUMINAMATH_CALUDE_apples_remaining_l284_28419

theorem apples_remaining (total : ℕ) (eaten : ℕ) (h1 : total = 15) (h2 : eaten = 7) :
  total - eaten = 8 := by
  sorry

end NUMINAMATH_CALUDE_apples_remaining_l284_28419


namespace NUMINAMATH_CALUDE_b_value_proof_l284_28497

theorem b_value_proof (a b c m : ℝ) (h : m = (c * a * b) / (a - b)) : 
  b = (m * a) / (m + c * a) := by
  sorry

end NUMINAMATH_CALUDE_b_value_proof_l284_28497


namespace NUMINAMATH_CALUDE_complex_equation_solution_l284_28478

theorem complex_equation_solution :
  ∀ z : ℂ, (1 + Complex.I * Real.sqrt 3) * z = Complex.I * Real.sqrt 3 →
    z = (3 / 4 : ℂ) + Complex.I * (Real.sqrt 3 / 4) :=
by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l284_28478


namespace NUMINAMATH_CALUDE_additional_bags_needed_l284_28437

/-- The number of people guaranteed to show up -/
def guaranteed_visitors : ℕ := 50

/-- The number of additional people who might show up -/
def potential_visitors : ℕ := 40

/-- The number of extravagant gift bags already made -/
def extravagant_bags : ℕ := 10

/-- The number of average gift bags already made -/
def average_bags : ℕ := 20

/-- The total number of visitors Carl is preparing for -/
def total_visitors : ℕ := guaranteed_visitors + potential_visitors

/-- The total number of gift bags already made -/
def existing_bags : ℕ := extravagant_bags + average_bags

/-- Theorem stating the number of additional bags Carl needs to make -/
theorem additional_bags_needed : total_visitors - existing_bags = 60 := by
  sorry

end NUMINAMATH_CALUDE_additional_bags_needed_l284_28437


namespace NUMINAMATH_CALUDE_weight_problem_l284_28489

theorem weight_problem (c d e f : ℝ) 
  (h1 : c + d = 330)
  (h2 : d + e = 290)
  (h3 : e + f = 310) :
  c + f = 350 := by
sorry

end NUMINAMATH_CALUDE_weight_problem_l284_28489


namespace NUMINAMATH_CALUDE_shiela_drawings_l284_28462

theorem shiela_drawings (neighbors : ℕ) (drawings_per_neighbor : ℕ) 
  (h1 : neighbors = 6) 
  (h2 : drawings_per_neighbor = 9) : 
  neighbors * drawings_per_neighbor = 54 := by
  sorry

end NUMINAMATH_CALUDE_shiela_drawings_l284_28462


namespace NUMINAMATH_CALUDE_percentage_of_older_female_students_l284_28467

theorem percentage_of_older_female_students
  (total_students : ℝ)
  (h1 : total_students > 0)
  (h2 : 0.4 * total_students = male_students)
  (h3 : 0.5 * male_students = older_male_students)
  (h4 : 0.56 * total_students = younger_students)
  : 0.4 * (total_students - male_students) = older_female_students :=
by
  sorry

end NUMINAMATH_CALUDE_percentage_of_older_female_students_l284_28467


namespace NUMINAMATH_CALUDE_sqrt_72_div_sqrt_8_minus_abs_neg_2_equals_1_l284_28418

theorem sqrt_72_div_sqrt_8_minus_abs_neg_2_equals_1 :
  Real.sqrt 72 / Real.sqrt 8 - |(-2)| = 1 := by sorry

end NUMINAMATH_CALUDE_sqrt_72_div_sqrt_8_minus_abs_neg_2_equals_1_l284_28418


namespace NUMINAMATH_CALUDE_sixth_power_of_complex_number_l284_28452

theorem sixth_power_of_complex_number :
  let z : ℂ := (Real.sqrt 3 + Complex.I) / 2
  z^6 = -1 := by sorry

end NUMINAMATH_CALUDE_sixth_power_of_complex_number_l284_28452


namespace NUMINAMATH_CALUDE_paulson_income_increase_paulson_income_increase_percentage_proof_l284_28435

/-- Paulson's financial situation --/
structure PaulsonFinances where
  income : ℝ
  expenditure_ratio : ℝ
  income_increase_ratio : ℝ
  expenditure_increase_ratio : ℝ
  savings_increase_ratio : ℝ

/-- Theorem stating the relationship between Paulson's financial changes --/
theorem paulson_income_increase
  (p : PaulsonFinances)
  (h1 : p.expenditure_ratio = 0.75)
  (h2 : p.expenditure_increase_ratio = 0.1)
  (h3 : p.savings_increase_ratio = 0.4999999999999996)
  : p.income_increase_ratio = 0.2 := by
  sorry

/-- The main result: Paulson's income increase percentage --/
def paulson_income_increase_percentage : ℝ := 20

/-- Theorem proving the income increase percentage --/
theorem paulson_income_increase_percentage_proof
  (p : PaulsonFinances)
  (h1 : p.expenditure_ratio = 0.75)
  (h2 : p.expenditure_increase_ratio = 0.1)
  (h3 : p.savings_increase_ratio = 0.4999999999999996)
  : paulson_income_increase_percentage = 100 * p.income_increase_ratio := by
  sorry

end NUMINAMATH_CALUDE_paulson_income_increase_paulson_income_increase_percentage_proof_l284_28435


namespace NUMINAMATH_CALUDE_sphere_only_circular_cross_sections_l284_28476

-- Define the possible geometric shapes
inductive GeometricShape
  | Cylinder
  | Cone
  | Sphere
  | ConeWithCircularBase

-- Define a function to check if a shape has circular cross-sections for all plane intersections
def hasCircularCrossSections (shape : GeometricShape) : Prop :=
  match shape with
  | GeometricShape.Sphere => true
  | _ => false

-- Theorem statement
theorem sphere_only_circular_cross_sections :
  ∀ (shape : GeometricShape),
    hasCircularCrossSections shape ↔ shape = GeometricShape.Sphere :=
by sorry

end NUMINAMATH_CALUDE_sphere_only_circular_cross_sections_l284_28476


namespace NUMINAMATH_CALUDE_distribution_recurrence_l284_28453

/-- The number of ways to distribute n distinct items to k people,
    such that each person receives at least one item -/
def g (n k : ℕ) : ℕ := sorry

theorem distribution_recurrence (n k : ℕ) (h1 : n ≥ k) (h2 : k ≥ 2) :
  g (n + 1) k = k * g n (k - 1) + k * g n k :=
sorry

end NUMINAMATH_CALUDE_distribution_recurrence_l284_28453


namespace NUMINAMATH_CALUDE_smallest_number_with_remainders_l284_28447

theorem smallest_number_with_remainders : ∃ (a : ℕ), 
  (a % 3 = 2) ∧ 
  (a % 5 = 4) ∧ 
  (a % 7 = 4) ∧ 
  (∀ n : ℕ, n < a → (n % 3 ≠ 2 ∨ n % 5 ≠ 4 ∨ n % 7 ≠ 4)) ∧
  a = 74 := by
sorry

end NUMINAMATH_CALUDE_smallest_number_with_remainders_l284_28447


namespace NUMINAMATH_CALUDE_concert_ticket_price_l284_28438

theorem concert_ticket_price (student_price : ℕ) (total_tickets : ℕ) (total_revenue : ℕ) (student_tickets : ℕ) :
  student_price = 9 →
  total_tickets = 2000 →
  total_revenue = 20960 →
  student_tickets = 520 →
  ∃ (non_student_price : ℕ),
    non_student_price * (total_tickets - student_tickets) + student_price * student_tickets = total_revenue ∧
    non_student_price = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_concert_ticket_price_l284_28438


namespace NUMINAMATH_CALUDE_event_probability_l284_28411

theorem event_probability (n : ℕ) (k₀ : ℕ) (p : ℝ) 
  (h1 : n = 120) 
  (h2 : k₀ = 32) 
  (h3 : k₀ = Int.floor (n * p)) :
  32 / 121 ≤ p ∧ p ≤ 33 / 121 := by
  sorry

end NUMINAMATH_CALUDE_event_probability_l284_28411


namespace NUMINAMATH_CALUDE_sqrt_calculations_l284_28422

theorem sqrt_calculations :
  (2 * Real.sqrt 2 + Real.sqrt 27 - Real.sqrt 8 = 3 * Real.sqrt 3) ∧
  ((2 * Real.sqrt 12 - 3 * Real.sqrt (1/3)) * Real.sqrt 6 = 9 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_calculations_l284_28422


namespace NUMINAMATH_CALUDE_alex_dresses_theorem_l284_28466

/-- Calculates the maximum number of complete dresses Alex can make --/
def max_dresses (initial_silk initial_satin initial_chiffon : ℕ) 
                (silk_per_dress satin_per_dress chiffon_per_dress : ℕ) 
                (friends : ℕ) (silk_per_friend satin_per_friend chiffon_per_friend : ℕ) : ℕ :=
  let remaining_silk := initial_silk - friends * silk_per_friend
  let remaining_satin := initial_satin - friends * satin_per_friend
  let remaining_chiffon := initial_chiffon - friends * chiffon_per_friend
  min (remaining_silk / silk_per_dress) 
      (min (remaining_satin / satin_per_dress) (remaining_chiffon / chiffon_per_dress))

theorem alex_dresses_theorem : 
  max_dresses 600 400 350 5 3 2 8 15 10 5 = 96 := by
  sorry

end NUMINAMATH_CALUDE_alex_dresses_theorem_l284_28466


namespace NUMINAMATH_CALUDE_unique_cube_prime_factor_l284_28448

def greatest_prime_factor (n : ℕ) : ℕ := sorry

theorem unique_cube_prime_factor : 
  ∃! n : ℕ, n > 1 ∧ 
    (greatest_prime_factor n = n^(1/3)) ∧ 
    (greatest_prime_factor (n + 200) = (n + 200)^(1/3)) := by
  sorry

end NUMINAMATH_CALUDE_unique_cube_prime_factor_l284_28448


namespace NUMINAMATH_CALUDE_right_trapezoid_perimeter_l284_28402

/-- A right trapezoid with upper base a, lower base b, height h, and leg l. -/
structure RightTrapezoid where
  a : ℝ
  b : ℝ
  h : ℝ
  l : ℝ

/-- The perimeter of a right trapezoid. -/
def perimeter (t : RightTrapezoid) : ℝ := t.a + t.b + t.h + t.l

/-- The theorem stating the conditions and the result for the right trapezoid problem. -/
theorem right_trapezoid_perimeter (t : RightTrapezoid) :
  t.a < t.b →
  π * t.h^2 * t.a + (1/3) * π * t.h^2 * (t.b - t.a) = 80 * π →
  π * t.h^2 * t.b + (1/3) * π * t.h^2 * (t.b - t.a) = 112 * π →
  (1/3) * π * (t.a^2 + t.a * t.b + t.b^2) * t.h = 156 * π →
  perimeter t = 20 + 2 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_right_trapezoid_perimeter_l284_28402


namespace NUMINAMATH_CALUDE_original_price_correct_l284_28479

/-- The original price of a dish, given specific discount and tip conditions --/
def original_price : ℝ := 24

/-- John's total payment for the dish --/
def john_payment (price : ℝ) : ℝ := 0.9 * price + 0.15 * price

/-- Jane's total payment for the dish --/
def jane_payment (price : ℝ) : ℝ := 0.9 * price + 0.15 * (0.9 * price)

/-- Theorem stating that the original price satisfies the given conditions --/
theorem original_price_correct :
  john_payment original_price - jane_payment original_price = 0.36 :=
by sorry

end NUMINAMATH_CALUDE_original_price_correct_l284_28479


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_l284_28417

theorem geometric_arithmetic_sequence (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- geometric sequence condition
  q ≠ 1 →  -- q is not equal to 1
  2 * a 3 = a 1 + a 2 →  -- arithmetic sequence condition
  q = -1/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_l284_28417


namespace NUMINAMATH_CALUDE_master_percentage_is_76_l284_28464

/-- Represents a team of junior and master players -/
structure Team where
  juniors : ℕ
  masters : ℕ

/-- The average score of the entire team -/
def teamAverage (t : Team) (juniorAvg masterAvg : ℚ) : ℚ :=
  (juniorAvg * t.juniors + masterAvg * t.masters) / (t.juniors + t.masters)

/-- The percentage of masters in the team -/
def masterPercentage (t : Team) : ℚ :=
  t.masters * 100 / (t.juniors + t.masters)

theorem master_percentage_is_76 (t : Team) :
  teamAverage t 22 47 = 41 →
  masterPercentage t = 76 := by
  sorry

#check master_percentage_is_76

end NUMINAMATH_CALUDE_master_percentage_is_76_l284_28464


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_2_l284_28440

/-- The displacement function of the object -/
def h (t : ℝ) : ℝ := 14 * t - t^2

/-- The velocity function of the object -/
def v (t : ℝ) : ℝ := 14 - 2 * t

/-- Theorem: The instantaneous velocity at t = 2 seconds is 10 meters/second -/
theorem instantaneous_velocity_at_2 : v 2 = 10 := by sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_2_l284_28440


namespace NUMINAMATH_CALUDE_certain_number_proof_l284_28409

theorem certain_number_proof (x : ℝ) (h : x = 3) :
  ∃ y : ℝ, (x + y) / (x + y + 5) = (x + y + 5) / (x + y + 5 + 13) ∧ y = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l284_28409


namespace NUMINAMATH_CALUDE_union_and_intersection_when_a_is_two_range_of_a_for_necessary_but_not_sufficient_l284_28412

-- Define the sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 3}
def B (a : ℝ) : Set ℝ := {x | 2 - a < x ∧ x < 2 + a}

-- Theorem for part (1)
theorem union_and_intersection_when_a_is_two :
  (A ∪ B 2 = {x | -1 < x ∧ x < 4}) ∧ (A ∩ B 2 = {x | 0 < x ∧ x < 3}) := by sorry

-- Theorem for part (2)
theorem range_of_a_for_necessary_but_not_sufficient :
  {a : ℝ | ∀ x, x ∈ B a → x ∈ A} ∩ {a : ℝ | ∃ x, x ∈ B a ∧ x ∉ A} = Set.Iic 1 := by sorry

end NUMINAMATH_CALUDE_union_and_intersection_when_a_is_two_range_of_a_for_necessary_but_not_sufficient_l284_28412


namespace NUMINAMATH_CALUDE_polynomial_integer_solution_l284_28449

theorem polynomial_integer_solution (p : ℤ → ℤ) 
  (h_integer_coeff : ∀ x y : ℤ, x - y ∣ p x - p y)
  (h_p_15 : p 15 = 6)
  (h_p_22 : p 22 = 1196)
  (h_p_35 : p 35 = 26) :
  ∃ n : ℤ, p n = n + 82 ∧ n = 28 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_integer_solution_l284_28449


namespace NUMINAMATH_CALUDE_investment_rate_proof_l284_28455

/-- Calculates the final amount after compound interest -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Proves that the given investment scenario results in a 10% annual interest rate -/
theorem investment_rate_proof (principal : ℝ) (final_amount : ℝ) (time : ℕ) 
  (h1 : principal = 5000)
  (h2 : final_amount = 6050.000000000001)
  (h3 : time = 2) :
  ∃ (rate : ℝ), compound_interest principal rate time = final_amount ∧ rate = 0.1 := by
  sorry

#check investment_rate_proof

end NUMINAMATH_CALUDE_investment_rate_proof_l284_28455


namespace NUMINAMATH_CALUDE_foreign_language_score_foreign_language_score_is_98_l284_28423

theorem foreign_language_score (average_three : ℝ) (average_two : ℝ) 
  (h1 : average_three = 94) (h2 : average_two = 92) : ℝ :=
  3 * average_three - 2 * average_two

theorem foreign_language_score_is_98 (average_three : ℝ) (average_two : ℝ) 
  (h1 : average_three = 94) (h2 : average_two = 92) : 
  foreign_language_score average_three average_two h1 h2 = 98 := by
  sorry

end NUMINAMATH_CALUDE_foreign_language_score_foreign_language_score_is_98_l284_28423


namespace NUMINAMATH_CALUDE_integer_triples_theorem_l284_28413

def satisfies_conditions (a b c : ℤ) : Prop :=
  a + b + c = 24 ∧ a^2 + b^2 + c^2 = 210 ∧ a * b * c = 440

def solution_set : Set (ℤ × ℤ × ℤ) :=
  {(11, 8, 5), (8, 11, 5), (8, 5, 11), (5, 8, 11), (11, 5, 8), (5, 11, 8)}

theorem integer_triples_theorem :
  ∀ (a b c : ℤ), satisfies_conditions a b c ↔ (a, b, c) ∈ solution_set :=
by sorry

end NUMINAMATH_CALUDE_integer_triples_theorem_l284_28413


namespace NUMINAMATH_CALUDE_inverse_composition_l284_28490

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the inverse functions
variable (f_inv g_inv : ℝ → ℝ)

-- State the given condition
axiom condition : ∀ x, f_inv (g x) = 5 * x + 3

-- State the theorem to be proved
theorem inverse_composition : g_inv (f (-7)) = -2 := by sorry

end NUMINAMATH_CALUDE_inverse_composition_l284_28490


namespace NUMINAMATH_CALUDE_first_graders_count_l284_28425

/-- The number of Kindergarteners to be checked -/
def kindergarteners : ℕ := 26

/-- The number of second graders to be checked -/
def second_graders : ℕ := 20

/-- The number of third graders to be checked -/
def third_graders : ℕ := 25

/-- The time in minutes it takes to check one student -/
def check_time : ℕ := 2

/-- The total time in hours available for all checks -/
def total_time_hours : ℕ := 3

/-- Calculate the number of first graders that need to be checked -/
def first_graders_to_check : ℕ :=
  (total_time_hours * 60 - (kindergarteners + second_graders + third_graders) * check_time) / check_time

/-- Theorem stating that the number of first graders to be checked is 19 -/
theorem first_graders_count : first_graders_to_check = 19 := by
  sorry

end NUMINAMATH_CALUDE_first_graders_count_l284_28425


namespace NUMINAMATH_CALUDE_helen_baked_554_cookies_this_morning_l284_28475

/-- Given the total number of chocolate chip cookies and the number baked yesterday,
    calculate the number of chocolate chip cookies baked this morning. -/
def cookies_baked_this_morning (total : ℕ) (yesterday : ℕ) : ℕ :=
  total - yesterday

/-- Theorem stating that Helen baked 554 chocolate chip cookies this morning. -/
theorem helen_baked_554_cookies_this_morning :
  cookies_baked_this_morning 1081 527 = 554 := by
  sorry

end NUMINAMATH_CALUDE_helen_baked_554_cookies_this_morning_l284_28475


namespace NUMINAMATH_CALUDE_two_people_two_rooms_probability_prove_two_people_two_rooms_probability_l284_28427

/-- The probability of two individuals randomly choosing different rooms out of two available rooms -/
theorem two_people_two_rooms_probability : ℝ :=
  1 / 2

/-- Prove that the probability of two individuals randomly choosing different rooms out of two available rooms is 1/2 -/
theorem prove_two_people_two_rooms_probability :
  two_people_two_rooms_probability = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_two_people_two_rooms_probability_prove_two_people_two_rooms_probability_l284_28427


namespace NUMINAMATH_CALUDE_pizza_eaters_l284_28434

theorem pizza_eaters (total_slices : ℕ) (slices_left : ℕ) (slices_per_person : ℕ) : 
  total_slices = 16 →
  slices_left = 4 →
  slices_per_person = 2 →
  (total_slices - slices_left) / slices_per_person = 6 :=
by sorry

end NUMINAMATH_CALUDE_pizza_eaters_l284_28434


namespace NUMINAMATH_CALUDE_sum_equals_twelve_l284_28499

theorem sum_equals_twelve (a b c : ℕ) (h : 28 * a + 30 * b + 31 * c = 365) : a + b + c = 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_twelve_l284_28499


namespace NUMINAMATH_CALUDE_no_strictly_increasing_sequence_with_addition_property_l284_28469

theorem no_strictly_increasing_sequence_with_addition_property :
  ¬ ∃ (a : ℕ → ℕ), 
    (∀ n m : ℕ, a (n * m) = a n + a m) ∧ 
    (∀ n : ℕ, a n < a (n + 1)) :=
by sorry

end NUMINAMATH_CALUDE_no_strictly_increasing_sequence_with_addition_property_l284_28469


namespace NUMINAMATH_CALUDE_hyperbola_center_l284_28487

-- Define the hyperbola equation
def hyperbola_equation (x y : ℝ) : Prop :=
  (4 * x - 8)^2 / 8^2 - (5 * y + 10)^2 / 3^2 = 1

-- Theorem stating that the center of the hyperbola is at (2, -2)
theorem hyperbola_center :
  ∃ (h k : ℝ), h = 2 ∧ k = -2 ∧
  (∀ (x y : ℝ), hyperbola_equation x y ↔ hyperbola_equation (x - h) (y - k)) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_center_l284_28487


namespace NUMINAMATH_CALUDE_brothers_money_l284_28477

theorem brothers_money (michael_initial : ℕ) (brother_initial : ℕ) (candy_cost : ℕ) : 
  michael_initial = 42 →
  brother_initial = 17 →
  candy_cost = 3 →
  brother_initial + michael_initial / 2 - candy_cost = 35 :=
by sorry

end NUMINAMATH_CALUDE_brothers_money_l284_28477


namespace NUMINAMATH_CALUDE_difference_of_squares_l284_28421

theorem difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l284_28421


namespace NUMINAMATH_CALUDE_parabola_h_value_l284_28465

/-- Represents a parabola of the form y = a(x-h)^2 + c -/
structure Parabola where
  a : ℝ
  h : ℝ
  c : ℝ

/-- The y-intercept of a parabola -/
def y_intercept (p : Parabola) : ℝ := p.a * p.h^2 + p.c

/-- Checks if a parabola has two positive integer x-intercepts -/
def has_two_positive_integer_x_intercepts (p : Parabola) : Prop :=
  ∃ x1 x2 : ℤ, x1 > 0 ∧ x2 > 0 ∧ x1 ≠ x2 ∧ 
    p.a * (x1 - p.h)^2 + p.c = 0 ∧ 
    p.a * (x2 - p.h)^2 + p.c = 0

theorem parabola_h_value 
  (p1 p2 : Parabola)
  (h1 : p1.a = 4)
  (h2 : p2.a = 5)
  (h3 : p1.h = p2.h)
  (h4 : y_intercept p1 = 4027)
  (h5 : y_intercept p2 = 4028)
  (h6 : has_two_positive_integer_x_intercepts p1)
  (h7 : has_two_positive_integer_x_intercepts p2) :
  p1.h = 36 := by
  sorry

end NUMINAMATH_CALUDE_parabola_h_value_l284_28465


namespace NUMINAMATH_CALUDE_exponent_multiplication_l284_28488

theorem exponent_multiplication (x : ℝ) : x^3 * (2*x^4) = 2*x^7 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l284_28488


namespace NUMINAMATH_CALUDE_neither_necessary_nor_sufficient_l284_28430

theorem neither_necessary_nor_sufficient : 
  ¬(∀ x : ℝ, -1/2 < x ∧ x < 1 → 0 < x ∧ x < 2) ∧ 
  ¬(∀ x : ℝ, 0 < x ∧ x < 2 → -1/2 < x ∧ x < 1) := by
  sorry

end NUMINAMATH_CALUDE_neither_necessary_nor_sufficient_l284_28430


namespace NUMINAMATH_CALUDE_least_positive_integer_congruence_l284_28420

theorem least_positive_integer_congruence :
  ∃ (x : ℕ), x > 0 ∧ (x + 3490) % 15 = 2801 % 15 ∧
  ∀ (y : ℕ), y > 0 ∧ (y + 3490) % 15 = 2801 % 15 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_congruence_l284_28420


namespace NUMINAMATH_CALUDE_only_valid_solutions_l284_28486

/-- A structure representing a solution to the equation AB = B^V --/
structure Solution :=
  (a : Nat) (b : Nat) (v : Nat)
  (h1 : a ≠ b) -- Different letters correspond to different digits
  (h2 : a * 10 + b ≥ 10 ∧ a * 10 + b < 100) -- AB is a two-digit number
  (h3 : a * 10 + b = b ^ v) -- AB = B^V

/-- The set of all valid solutions --/
def validSolutions : Set Solution :=
  { s : Solution | s.a = 3 ∧ s.b = 2 ∧ s.v = 5 ∨
                   s.a = 3 ∧ s.b = 6 ∧ s.v = 2 ∨
                   s.a = 6 ∧ s.b = 4 ∧ s.v = 3 }

/-- Theorem stating that the only solutions are 32 = 2^5, 36 = 6^2, and 64 = 4^3 --/
theorem only_valid_solutions (s : Solution) : s ∈ validSolutions := by
  sorry

end NUMINAMATH_CALUDE_only_valid_solutions_l284_28486


namespace NUMINAMATH_CALUDE_ceiling_floor_difference_l284_28433

theorem ceiling_floor_difference : 
  ⌈(15 : ℝ) / 8 * (-34 : ℝ) / 4⌉ - ⌊(15 : ℝ) / 8 * ⌈(-34 : ℝ) / 4⌉⌋ = 0 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_difference_l284_28433


namespace NUMINAMATH_CALUDE_cube_side_length_l284_28424

theorem cube_side_length (surface_area : ℝ) (side_length : ℝ) : 
  surface_area = 864 → 
  surface_area = 6 * side_length^2 → 
  side_length = 12 := by
sorry

end NUMINAMATH_CALUDE_cube_side_length_l284_28424


namespace NUMINAMATH_CALUDE_smallest_perimeter_of_special_triangle_l284_28403

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- A function that checks if three numbers are consecutive odd primes -/
def areConsecutiveOddPrimes (a b c : ℕ) : Prop :=
  isPrime a ∧ isPrime b ∧ isPrime c ∧ 
  b = a + 2 ∧ c = b + 2 ∧
  a % 2 = 1

/-- The main theorem -/
theorem smallest_perimeter_of_special_triangle :
  ∀ a b c : ℕ,
    areConsecutiveOddPrimes a b c →
    isPrime (a + b + c) →
    a + b + c ≥ 41 :=
  sorry

end NUMINAMATH_CALUDE_smallest_perimeter_of_special_triangle_l284_28403


namespace NUMINAMATH_CALUDE_accounting_majors_count_l284_28407

theorem accounting_majors_count 
  (p q r s : ℕ+) 
  (h1 : p * q * r * s = 1365)
  (h2 : 1 < p) (h3 : p < q) (h4 : q < r) (h5 : r < s) :
  p = 3 := by
sorry

end NUMINAMATH_CALUDE_accounting_majors_count_l284_28407


namespace NUMINAMATH_CALUDE_smallest_representable_number_l284_28498

/-- Sum of decimal digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Checks if a natural number can be represented as the sum of k positive integers
    with the same sum of decimal digits -/
def representable (n k : ℕ) : Prop :=
  ∃ (d : ℕ), d > 0 ∧ n = k * d ∧ sum_of_digits d = sum_of_digits n / k

theorem smallest_representable_number :
  (∀ m : ℕ, m < 10010 → ¬(representable m 2002 ∧ representable m 2003)) ∧
  (representable 10010 2002 ∧ representable 10010 2003) := by sorry

end NUMINAMATH_CALUDE_smallest_representable_number_l284_28498


namespace NUMINAMATH_CALUDE_grazing_months_a_l284_28410

/-- The number of months a put his oxen for grazing -/
def months_a : ℕ := 7

/-- The number of oxen a put for grazing -/
def oxen_a : ℕ := 10

/-- The number of oxen b put for grazing -/
def oxen_b : ℕ := 12

/-- The number of months b put his oxen for grazing -/
def months_b : ℕ := 5

/-- The number of oxen c put for grazing -/
def oxen_c : ℕ := 15

/-- The number of months c put his oxen for grazing -/
def months_c : ℕ := 3

/-- The total rent of the pasture in rupees -/
def total_rent : ℚ := 245

/-- The share of rent c pays in rupees -/
def c_rent_share : ℚ := 62.99999999999999

theorem grazing_months_a : 
  months_a * oxen_a * total_rent = 
  c_rent_share * (months_a * oxen_a + months_b * oxen_b + months_c * oxen_c) := by
  sorry

end NUMINAMATH_CALUDE_grazing_months_a_l284_28410


namespace NUMINAMATH_CALUDE_probability_at_least_one_vowel_l284_28405

def set1 : Finset Char := {'a', 'b', 'c', 'd', 'e'}
def set2 : Finset Char := {'k', 'l', 'm', 'n', 'o', 'p'}

def vowels : Finset Char := {'a', 'e', 'i', 'o', 'u'}

def isVowel (c : Char) : Bool := c ∈ vowels

theorem probability_at_least_one_vowel :
  let prob_no_vowel_set1 := (set1.filter (λ c => ¬isVowel c)).card / set1.card
  let prob_no_vowel_set2 := (set2.filter (λ c => ¬isVowel c)).card / set2.card
  1 - (prob_no_vowel_set1 * prob_no_vowel_set2) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_vowel_l284_28405


namespace NUMINAMATH_CALUDE_counterexample_exists_l284_28401

theorem counterexample_exists : ∃ a b : ℝ, a > b ∧ a^2 ≤ b^2 := by sorry

end NUMINAMATH_CALUDE_counterexample_exists_l284_28401


namespace NUMINAMATH_CALUDE_equation_solution_l284_28429

theorem equation_solution : ∃! x : ℝ, (2 / (x - 3) = 3 / x) ∧ x = 9 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l284_28429


namespace NUMINAMATH_CALUDE_parabola_through_fixed_point_l284_28493

-- Define the line equation as a function of a
def line_equation (a x y : ℝ) : Prop := (a - 1) * x - y + 2 * a + 1 = 0

-- Define the fixed point P
def fixed_point : ℝ × ℝ := (-2, 3)

-- Define the two possible parabola equations
def parabola1 (x y : ℝ) : Prop := y^2 = -9/2 * x
def parabola2 (x y : ℝ) : Prop := x^2 = 4/3 * y

-- State the theorem
theorem parabola_through_fixed_point :
  (∀ a : ℝ, line_equation a (fixed_point.1) (fixed_point.2)) →
  (parabola1 (fixed_point.1) (fixed_point.2) ∨ parabola2 (fixed_point.1) (fixed_point.2)) :=
sorry

end NUMINAMATH_CALUDE_parabola_through_fixed_point_l284_28493


namespace NUMINAMATH_CALUDE_remainder_theorem_l284_28463

theorem remainder_theorem (x : ℤ) (h : x % 11 = 7) : (x^3 - (2*x)^2) % 11 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l284_28463


namespace NUMINAMATH_CALUDE_oranges_per_box_l284_28451

theorem oranges_per_box (total_oranges : ℕ) (num_boxes : ℕ) (h1 : total_oranges = 24) (h2 : num_boxes = 3) :
  total_oranges / num_boxes = 8 := by
  sorry

end NUMINAMATH_CALUDE_oranges_per_box_l284_28451

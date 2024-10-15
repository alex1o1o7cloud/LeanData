import Mathlib

namespace NUMINAMATH_CALUDE_card_events_l3374_337462

structure Card where
  suit : Fin 4
  number : Fin 10

def Deck : Finset Card := sorry

def isHeart (c : Card) : Prop := c.suit = 0
def isSpade (c : Card) : Prop := c.suit = 1
def isRed (c : Card) : Prop := c.suit = 0 ∨ c.suit = 2
def isBlack (c : Card) : Prop := c.suit = 1 ∨ c.suit = 3
def isMultipleOf5 (c : Card) : Prop := c.number % 5 = 0
def isGreaterThan9 (c : Card) : Prop := c.number = 9

def mutuallyExclusive (e1 e2 : Card → Prop) : Prop :=
  ∀ c : Card, ¬(e1 c ∧ e2 c)

def complementary (e1 e2 : Card → Prop) : Prop :=
  ∀ c : Card, e1 c ∨ e2 c

theorem card_events :
  (mutuallyExclusive isHeart isSpade ∧ ¬complementary isHeart isSpade) ∧
  (mutuallyExclusive isRed isBlack ∧ complementary isRed isBlack) ∧
  (¬mutuallyExclusive isMultipleOf5 isGreaterThan9 ∧ ¬complementary isMultipleOf5 isGreaterThan9) :=
by sorry

end NUMINAMATH_CALUDE_card_events_l3374_337462


namespace NUMINAMATH_CALUDE_factor_polynomial_l3374_337471

theorem factor_polynomial (x : ℝ) : 54 * x^4 - 135 * x^8 = -27 * x^4 * (5 * x^4 - 2) := by
  sorry

end NUMINAMATH_CALUDE_factor_polynomial_l3374_337471


namespace NUMINAMATH_CALUDE_sin_cos_sixth_power_inequality_l3374_337451

theorem sin_cos_sixth_power_inequality (x : ℝ) : Real.sin x ^ 6 + Real.cos x ^ 6 ≥ (1/4 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sixth_power_inequality_l3374_337451


namespace NUMINAMATH_CALUDE_subset_implies_a_values_l3374_337437

def A : Set ℝ := {x | x^2 - 8*x + 15 = 0}

def B (a : ℝ) : Set ℝ := {x | a*x - 1 = 0}

theorem subset_implies_a_values (a : ℝ) :
  B a ⊆ A → a = 1/3 ∨ a = 1/5 ∨ a = 0 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_values_l3374_337437


namespace NUMINAMATH_CALUDE_implication_equiv_contrapositive_l3374_337423

-- Define the propositions
variable (P Q : Prop)

-- Define the original implication
def original : Prop := P → Q

-- Define the contrapositive
def contrapositive : Prop := ¬Q → ¬P

-- Theorem stating the equivalence of the original implication and its contrapositive
theorem implication_equiv_contrapositive :
  original P Q ↔ contrapositive P Q :=
sorry

end NUMINAMATH_CALUDE_implication_equiv_contrapositive_l3374_337423


namespace NUMINAMATH_CALUDE_functional_equation_zero_solution_l3374_337470

theorem functional_equation_zero_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x + y) = f x - f y) : 
  ∀ x : ℝ, f x = 0 := by
sorry

end NUMINAMATH_CALUDE_functional_equation_zero_solution_l3374_337470


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3374_337413

def solution_set (x : ℝ) : Prop := -1 < x ∧ x < 2

theorem inequality_solution_set :
  ∀ x : ℝ, x * (x - 1) < 2 ↔ solution_set x :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3374_337413


namespace NUMINAMATH_CALUDE_equal_cupcake_distribution_l3374_337425

theorem equal_cupcake_distribution (total_cupcakes : ℕ) (num_children : ℕ) 
  (h1 : total_cupcakes = 96) (h2 : num_children = 8) :
  total_cupcakes / num_children = 12 := by
  sorry

end NUMINAMATH_CALUDE_equal_cupcake_distribution_l3374_337425


namespace NUMINAMATH_CALUDE_remainder_444_power_444_mod_13_l3374_337434

theorem remainder_444_power_444_mod_13 : 444^444 ≡ 1 [ZMOD 13] := by
  sorry

end NUMINAMATH_CALUDE_remainder_444_power_444_mod_13_l3374_337434


namespace NUMINAMATH_CALUDE_correct_stickers_calculation_l3374_337426

/-- The number of cat stickers each girl received from their grandparents -/
def stickers_from_grandparents (june_initial : ℕ) (bonnie_initial : ℕ) (total_after : ℕ) : ℕ :=
  (total_after - (june_initial + bonnie_initial)) / 2

theorem correct_stickers_calculation :
  stickers_from_grandparents 76 63 189 = 25 := by
  sorry

end NUMINAMATH_CALUDE_correct_stickers_calculation_l3374_337426


namespace NUMINAMATH_CALUDE_cycle_selling_price_l3374_337489

/-- Calculates the final selling price of a cycle given the original price and various discounts and losses. -/
def final_selling_price (original_price : ℝ) (initial_discount : ℝ) (loss_on_sale : ℝ) (exchange_discount : ℝ) : ℝ :=
  let price_after_initial_discount := original_price * (1 - initial_discount)
  let intended_selling_price := price_after_initial_discount * (1 - loss_on_sale)
  intended_selling_price * (1 - exchange_discount)

/-- Theorem stating that the final selling price of the cycle is 897.75 given the specified conditions. -/
theorem cycle_selling_price :
  final_selling_price 1400 0.05 0.25 0.10 = 897.75 := by
  sorry

end NUMINAMATH_CALUDE_cycle_selling_price_l3374_337489


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l3374_337479

def M : Set ℤ := {x | Real.log (x - 1) ≤ 0}
def N : Set ℤ := {x | Int.natAbs x < 2}

theorem union_of_M_and_N : M ∪ N = {-1, 0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l3374_337479


namespace NUMINAMATH_CALUDE_x_less_than_y_l3374_337403

theorem x_less_than_y :
  let x : ℝ := Real.sqrt 7 - Real.sqrt 3
  let y : ℝ := Real.sqrt 6 - Real.sqrt 2
  x < y := by sorry

end NUMINAMATH_CALUDE_x_less_than_y_l3374_337403


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l3374_337407

theorem simplify_trig_expression :
  Real.sqrt (1 - 2 * Real.sin (35 * π / 180) * Real.cos (35 * π / 180)) =
  Real.cos (35 * π / 180) - Real.sin (35 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l3374_337407


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3374_337454

def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {2, 4, 5}

theorem intersection_of_A_and_B :
  A ∩ B = {2, 4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3374_337454


namespace NUMINAMATH_CALUDE_y_relationship_l3374_337433

/-- A quadratic function of the form y = -(x+2)² + h -/
def f (x h : ℝ) : ℝ := -(x + 2)^2 + h

/-- The y-coordinate of point A -/
def y₁ (h : ℝ) : ℝ := f (-3) h

/-- The y-coordinate of point B -/
def y₂ (h : ℝ) : ℝ := f 2 h

/-- The y-coordinate of point C -/
def y₃ (h : ℝ) : ℝ := f 3 h

/-- Theorem stating the relationship between y₁, y₂, and y₃ -/
theorem y_relationship (h : ℝ) : y₃ h < y₂ h ∧ y₂ h < y₁ h := by
  sorry

end NUMINAMATH_CALUDE_y_relationship_l3374_337433


namespace NUMINAMATH_CALUDE_parabola_p_value_l3374_337466

/-- A parabola with focus distance 12 and y-axis distance 9 has p = 6 -/
theorem parabola_p_value (p : ℝ) (A : ℝ × ℝ) :
  p > 0 →
  A.2^2 = 2 * p * A.1 →
  dist A (p / 2, 0) = 12 →
  A.1 = 9 →
  p = 6 := by
  sorry

end NUMINAMATH_CALUDE_parabola_p_value_l3374_337466


namespace NUMINAMATH_CALUDE_sum_of_linear_equations_l3374_337478

theorem sum_of_linear_equations (x y : ℝ) 
  (h1 : 2*x - 1 = 5) 
  (h2 : 3*y + 2 = 17) : 
  2*x + 3*y = 21 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_linear_equations_l3374_337478


namespace NUMINAMATH_CALUDE_binomial_coefficient_self_l3374_337483

theorem binomial_coefficient_self : (510 : ℕ).choose 510 = 1 := by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_self_l3374_337483


namespace NUMINAMATH_CALUDE_not_sufficient_not_necessary_l3374_337431

theorem not_sufficient_not_necessary (a b : ℝ) : 
  ¬(∀ a b, (a ≠ 1 ∧ b ≠ 2) → (a + b ≠ 3)) ∧ 
  ¬(∀ a b, (a + b ≠ 3) → (a ≠ 1 ∧ b ≠ 2)) := by
  sorry

end NUMINAMATH_CALUDE_not_sufficient_not_necessary_l3374_337431


namespace NUMINAMATH_CALUDE_casey_calculation_l3374_337459

theorem casey_calculation (x : ℝ) : (x / 7) - 20 = 19 → (x * 7) + 20 = 1931 := by
  sorry

end NUMINAMATH_CALUDE_casey_calculation_l3374_337459


namespace NUMINAMATH_CALUDE_kangaroo_fraction_sum_l3374_337419

theorem kangaroo_fraction_sum (total : ℕ) (grey pink : ℕ) : 
  total = grey + pink ∧ 
  grey > 0 ∧ 
  pink > 0 ∧
  total = 2016 →
  (grey : ℝ) * (pink / grey) + (pink : ℝ) * (grey / pink) = total := by
  sorry

end NUMINAMATH_CALUDE_kangaroo_fraction_sum_l3374_337419


namespace NUMINAMATH_CALUDE_air_quality_probability_l3374_337405

theorem air_quality_probability (p_single : ℝ) (p_consecutive : ℝ) (p_next : ℝ) : 
  p_single = 0.75 → p_consecutive = 0.6 → p_next = p_consecutive / p_single → p_next = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_air_quality_probability_l3374_337405


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_1729_l3374_337401

theorem largest_prime_factor_of_1729 :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 1729 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 1729 → q ≤ p ∧ p = 19 :=
sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_1729_l3374_337401


namespace NUMINAMATH_CALUDE_mixed_number_calculation_l3374_337417

theorem mixed_number_calculation : 
  (4 + 2/7 : ℚ) * (5 + 1/2) - ((3 + 1/3) + (2 + 1/6)) = 18 + 1/14 := by
  sorry

end NUMINAMATH_CALUDE_mixed_number_calculation_l3374_337417


namespace NUMINAMATH_CALUDE_max_value_sum_l3374_337484

theorem max_value_sum (a b c d e : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (he : 0 < e)
  (h_sum_squares : a^2 + b^2 + c^2 + d^2 + e^2 = 504) :
  ∃ (N a_N b_N c_N d_N e_N : ℝ),
    (∀ x y z w v : ℝ, x > 0 → y > 0 → z > 0 → w > 0 → v > 0 →
      x^2 + y^2 + z^2 + w^2 + v^2 = 504 →
      x*z + 3*y*z + 4*z*w + 8*z*v ≤ N) ∧
    a_N > 0 ∧ b_N > 0 ∧ c_N > 0 ∧ d_N > 0 ∧ e_N > 0 ∧
    a_N^2 + b_N^2 + c_N^2 + d_N^2 + e_N^2 = 504 ∧
    a_N*c_N + 3*b_N*c_N + 4*c_N*d_N + 8*c_N*e_N = N ∧
    N + a_N + b_N + c_N + d_N + e_N = 96 * Real.sqrt 2 + 6 * Real.sqrt 21 + 756 * Real.sqrt 10 :=
by sorry

end NUMINAMATH_CALUDE_max_value_sum_l3374_337484


namespace NUMINAMATH_CALUDE_marshmallow_roasting_l3374_337404

/-- The number of marshmallows Joe's dad has -/
def dads_marshmallows : ℕ := 21

/-- The number of marshmallows Joe has -/
def joes_marshmallows : ℕ := 4 * dads_marshmallows

/-- The number of marshmallows Joe's dad roasts -/
def dads_roasted : ℕ := dads_marshmallows / 3

/-- The number of marshmallows Joe roasts -/
def joes_roasted : ℕ := joes_marshmallows / 2

/-- The total number of marshmallows roasted -/
def total_roasted : ℕ := dads_roasted + joes_roasted

theorem marshmallow_roasting :
  total_roasted = 49 := by sorry

end NUMINAMATH_CALUDE_marshmallow_roasting_l3374_337404


namespace NUMINAMATH_CALUDE_betty_initial_marbles_l3374_337432

/-- Proves that Betty initially had 60 marbles given the conditions of the problem -/
theorem betty_initial_marbles :
  ∀ (betty_initial : ℕ) (stuart_initial : ℕ) (stuart_final : ℕ),
    stuart_initial = 56 →
    stuart_final = 80 →
    stuart_final = stuart_initial + (betty_initial * 40 / 100) →
    betty_initial = 60 :=
by sorry

end NUMINAMATH_CALUDE_betty_initial_marbles_l3374_337432


namespace NUMINAMATH_CALUDE_probability_two_math_teachers_l3374_337449

def english_teachers : ℕ := 3
def math_teachers : ℕ := 4
def social_teachers : ℕ := 2
def total_teachers : ℕ := english_teachers + math_teachers + social_teachers
def selected_members : ℕ := 2

theorem probability_two_math_teachers :
  (Nat.choose math_teachers selected_members : ℚ) / (Nat.choose total_teachers selected_members) = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_math_teachers_l3374_337449


namespace NUMINAMATH_CALUDE_class_size_proof_l3374_337472

theorem class_size_proof (x : ℕ) 
  (h1 : x > 3)
  (h2 : (85 - 78) / (x - 3 : ℝ) = 0.75) : 
  x = 13 := by
sorry

end NUMINAMATH_CALUDE_class_size_proof_l3374_337472


namespace NUMINAMATH_CALUDE_apple_calculation_l3374_337435

/-- The number of apples Pinky, Danny, and Benny collectively have after accounting for Lucy's sales -/
def total_apples (pinky_apples danny_apples lucy_sales benny_apples : ℝ) : ℝ :=
  pinky_apples + danny_apples + benny_apples - lucy_sales

/-- Theorem stating the total number of apples after Lucy's sales -/
theorem apple_calculation :
  total_apples 36.5 73.2 15.7 48.8 = 142.8 := by
  sorry

end NUMINAMATH_CALUDE_apple_calculation_l3374_337435


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l3374_337445

def i : ℂ := Complex.I

theorem complex_fraction_equality : (2 * i) / (1 + i) = 1 + i := by sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l3374_337445


namespace NUMINAMATH_CALUDE_recycling_drive_target_l3374_337496

/-- The recycling drive problem -/
theorem recycling_drive_target (num_sections : ℕ) (kilos_per_section : ℕ) (kilos_needed : ℕ) : 
  num_sections = 6 → 
  kilos_per_section = 280 → 
  kilos_needed = 320 → 
  num_sections * kilos_per_section + kilos_needed = 2000 := by
sorry

end NUMINAMATH_CALUDE_recycling_drive_target_l3374_337496


namespace NUMINAMATH_CALUDE_lowest_number_three_probability_l3374_337463

-- Define a six-sided die
def die := Fin 6

-- Define the probability of rolling at least 3 on a single die
def prob_at_least_3 : ℚ := 4 / 6

-- Define the probability of rolling at least 4 on a single die
def prob_at_least_4 : ℚ := 3 / 6

-- Define the number of dice rolled
def num_dice : ℕ := 4

-- Theorem statement
theorem lowest_number_three_probability :
  (prob_at_least_3 ^ num_dice - prob_at_least_4 ^ num_dice) = 175 / 1296 :=
by sorry

end NUMINAMATH_CALUDE_lowest_number_three_probability_l3374_337463


namespace NUMINAMATH_CALUDE_lock_rings_count_l3374_337443

theorem lock_rings_count : ∃ (n : ℕ), n > 0 ∧ 6^n - 1 ≤ 215 ∧ ∀ (m : ℕ), m > 0 → 6^m - 1 ≤ 215 → m ≤ n := by
  sorry

end NUMINAMATH_CALUDE_lock_rings_count_l3374_337443


namespace NUMINAMATH_CALUDE_pythagorean_triple_divisibility_l3374_337492

theorem pythagorean_triple_divisibility (a b c : ℕ) (h : a^2 + b^2 = c^2) :
  (3 ∣ a ∨ 3 ∣ b) ∧ (4 ∣ a ∨ 4 ∣ b) ∧ (5 ∣ a ∨ 5 ∣ b ∨ 5 ∣ c) := by
  sorry

end NUMINAMATH_CALUDE_pythagorean_triple_divisibility_l3374_337492


namespace NUMINAMATH_CALUDE_quadratic_discriminant_nonnegative_l3374_337456

theorem quadratic_discriminant_nonnegative (x : ℤ) :
  x^2 * (25 - 24*x^2) ≥ 0 ↔ x = 0 ∨ x = 1 ∨ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_nonnegative_l3374_337456


namespace NUMINAMATH_CALUDE_math_books_count_l3374_337461

theorem math_books_count (total_books : ℕ) (math_price history_price total_price : ℕ) 
  (h1 : total_books = 80)
  (h2 : math_price = 4)
  (h3 : history_price = 5)
  (h4 : total_price = 390) :
  ∃ (math_books : ℕ), 
    math_books * math_price + (total_books - math_books) * history_price = total_price ∧ 
    math_books = 10 := by
  sorry

end NUMINAMATH_CALUDE_math_books_count_l3374_337461


namespace NUMINAMATH_CALUDE_library_book_count_l3374_337460

/-- Calculates the final number of books in a library given initial count and changes. -/
def finalBookCount (initial : ℕ) (takenTuesday : ℕ) (broughtThursday : ℕ) (takenFriday : ℕ) : ℕ :=
  initial - takenTuesday + broughtThursday - takenFriday

/-- Theorem stating that given the specific book counts and changes, the final count is 29. -/
theorem library_book_count :
  finalBookCount 235 227 56 35 = 29 := by
  sorry

end NUMINAMATH_CALUDE_library_book_count_l3374_337460


namespace NUMINAMATH_CALUDE_factor_81_minus_36x4_l3374_337499

theorem factor_81_minus_36x4 (x : ℝ) : 
  81 - 36 * x^4 = 9 * (Real.sqrt 3 - Real.sqrt 2 * x) * (Real.sqrt 3 + Real.sqrt 2 * x) * (3 + 2 * x^2) := by
  sorry

end NUMINAMATH_CALUDE_factor_81_minus_36x4_l3374_337499


namespace NUMINAMATH_CALUDE_parcel_cost_correct_l3374_337420

/-- The cost function for sending a parcel post package -/
def parcel_cost (P : ℕ) : ℕ :=
  10 + 3 * (P - 1)

/-- Theorem stating that the parcel_cost function correctly calculates the cost
    for a package of weight P pounds, where P is a positive integer -/
theorem parcel_cost_correct (P : ℕ) (h : P > 0) :
  parcel_cost P = 10 + 3 * (P - 1) ∧
  (P = 1 → parcel_cost P = 10) ∧
  (P > 1 → parcel_cost P = 10 + 3 * (P - 1)) :=
by sorry

end NUMINAMATH_CALUDE_parcel_cost_correct_l3374_337420


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l3374_337448

/-- The equation of a line perpendicular to 2x-y+1=0 and passing through (0,1) -/
theorem perpendicular_line_equation :
  let l₁ : ℝ → ℝ → Prop := λ x y => 2*x - y + 1 = 0
  let p : ℝ × ℝ := (0, 1)
  let l₂ : ℝ → ℝ → Prop := λ x y => x + 2*y - 2 = 0
  (∀ x y, l₂ x y ↔ (y - p.2 = -(1/(2:ℝ)) * (x - p.1))) ∧
  (∀ x y, l₁ x y → ∀ x' y', l₂ x' y' → (y - y') * (x - x') = -(1:ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l3374_337448


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l3374_337440

theorem quadratic_two_distinct_roots (k : ℝ) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
  (x₁^2 + (k-5)*x₁ - 3*k = 0) ∧ 
  (x₂^2 + (k-5)*x₂ - 3*k = 0) := by
sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l3374_337440


namespace NUMINAMATH_CALUDE_three_numbers_sum_l3374_337468

theorem three_numbers_sum (a b c : ℝ) : 
  a ≤ b ∧ b ≤ c →
  (a + b + c) / 3 = a + 8 →
  (a + b + c) / 3 = c - 9 →
  c - a = 26 →
  a + b + c = 81 := by
sorry

end NUMINAMATH_CALUDE_three_numbers_sum_l3374_337468


namespace NUMINAMATH_CALUDE_function_satisfies_equation_l3374_337428

theorem function_satisfies_equation (a b c : ℝ) (h : a ≠ b) :
  let f : ℝ → ℝ := λ x ↦ (c / (a - b)) * x
  ∀ x, a * f (x - 1) + b * f (1 - x) = c * x := by
  sorry

end NUMINAMATH_CALUDE_function_satisfies_equation_l3374_337428


namespace NUMINAMATH_CALUDE_total_distance_walked_l3374_337465

def distance_to_water_fountain : ℕ := 30
def distance_to_staff_lounge : ℕ := 45
def trips_to_water_fountain : ℕ := 4
def trips_to_staff_lounge : ℕ := 3

theorem total_distance_walked :
  2 * (distance_to_water_fountain * trips_to_water_fountain +
       distance_to_staff_lounge * trips_to_staff_lounge) = 510 :=
by sorry

end NUMINAMATH_CALUDE_total_distance_walked_l3374_337465


namespace NUMINAMATH_CALUDE_nonlinear_system_solutions_l3374_337457

theorem nonlinear_system_solutions :
  let f₁ (x y z : ℝ) := x + 4*y + 6*z - 16
  let f₂ (x y z : ℝ) := x + 6*y + 12*z - 24
  let f₃ (x y z : ℝ) := x^2 + 4*y^2 + 36*z^2 - 76
  ∀ (x y z : ℝ),
    f₁ x y z = 0 ∧ f₂ x y z = 0 ∧ f₃ x y z = 0 ↔
    (x = 6 ∧ y = 1 ∧ z = 1) ∨ (x = -2/3 ∧ y = 13/3 ∧ z = -1/9) :=
by
  sorry

#check nonlinear_system_solutions

end NUMINAMATH_CALUDE_nonlinear_system_solutions_l3374_337457


namespace NUMINAMATH_CALUDE_total_money_l3374_337497

theorem total_money (a b c : ℕ) 
  (h1 : a + c = 700)
  (h2 : b + c = 600)
  (h3 : c = 300) : 
  a + b + c = 1000 := by
sorry

end NUMINAMATH_CALUDE_total_money_l3374_337497


namespace NUMINAMATH_CALUDE_distance_traveled_l3374_337447

/-- Proves that given a constant speed and time, the distance traveled is equal to speed multiplied by time -/
theorem distance_traveled (speed : ℝ) (time : ℝ) : 
  speed = 6 → time = 16 → speed * time = 96 := by
  sorry

end NUMINAMATH_CALUDE_distance_traveled_l3374_337447


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l3374_337476

theorem polynomial_divisibility (p' q' : ℝ) : 
  (∀ x : ℝ, (x + 1) * (x - 2) = 0 → x^5 - x^4 + x^3 - p' * x^2 + q' * x - 6 = 0) →
  p' = 0 ∧ q' = -9 := by
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l3374_337476


namespace NUMINAMATH_CALUDE_absolute_value_minus_half_power_l3374_337477

theorem absolute_value_minus_half_power : |(-3 : ℝ)| - (1/2 : ℝ)^0 = 2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_minus_half_power_l3374_337477


namespace NUMINAMATH_CALUDE_integral_sqrt_minus_sin_l3374_337485

theorem integral_sqrt_minus_sin : ∫ x in (-1)..1, (2 * Real.sqrt (1 - x^2) - Real.sin x) = π := by sorry

end NUMINAMATH_CALUDE_integral_sqrt_minus_sin_l3374_337485


namespace NUMINAMATH_CALUDE_students_who_left_l3374_337442

/-- Proves the number of students who left given initial, new, and final student counts -/
theorem students_who_left (initial : ℕ) (new : ℕ) (final : ℕ) 
  (h_initial : initial = 10)
  (h_new : new = 42)
  (h_final : final = 48) :
  initial + new - final = 4 := by
  sorry

end NUMINAMATH_CALUDE_students_who_left_l3374_337442


namespace NUMINAMATH_CALUDE_watermelon_weight_theorem_l3374_337453

/-- Represents the weight of watermelons in a basket -/
structure WatermelonBasket where
  initialWeight : ℝ  -- Initial weight of basket with watermelons
  halfRemovedWeight : ℝ  -- Weight after removing half of watermelons
  basketWeight : ℝ  -- Weight of the empty basket

/-- Calculates the total weight of watermelons in the basket -/
def totalWatermelonWeight (basket : WatermelonBasket) : ℝ :=
  basket.initialWeight - basket.basketWeight

/-- Theorem stating the total weight of watermelons in the given scenario -/
theorem watermelon_weight_theorem (basket : WatermelonBasket) 
  (h1 : basket.initialWeight = 63)
  (h2 : basket.halfRemovedWeight = 34)
  (h3 : basket.basketWeight = basket.halfRemovedWeight - (basket.initialWeight - basket.basketWeight) / 2) :
  totalWatermelonWeight basket = 58 := by
  sorry

#check watermelon_weight_theorem

end NUMINAMATH_CALUDE_watermelon_weight_theorem_l3374_337453


namespace NUMINAMATH_CALUDE_expression_evaluation_l3374_337481

theorem expression_evaluation :
  let a : ℚ := -1/2
  (3*a + 2) * (a - 1) - 4*a * (a + 1) = 1/4 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3374_337481


namespace NUMINAMATH_CALUDE_no_n_satisfies_condition_l3374_337480

def T_n (n : ℕ+) : Set ℕ+ :=
  {a | ∃ k h : ℕ, 1 ≤ k ∧ k ≤ 10 ∧ 1 ≤ h ∧ h ≤ 10 ∧
    a = 11 * (k + h) + 10 * (n ^ k + n ^ h)}

theorem no_n_satisfies_condition :
  ∀ n : ℕ+, ∃ a b : ℕ+, a ∈ T_n n ∧ b ∈ T_n n ∧ a ≠ b ∧ a ≡ b [MOD 110] :=
sorry

end NUMINAMATH_CALUDE_no_n_satisfies_condition_l3374_337480


namespace NUMINAMATH_CALUDE_quadrilateral_reconstruction_l3374_337429

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a vector in 2D space -/
def Vector2D := Point2D

/-- Quadrilateral represented by four points -/
structure Quadrilateral where
  A : Point2D
  B : Point2D
  C : Point2D
  D : Point2D

/-- Extended quadrilateral with additional points -/
structure ExtendedQuadrilateral where
  Q : Quadrilateral
  A' : Point2D
  D' : Point2D

/-- Vector addition -/
def vectorAdd (v1 v2 : Vector2D) : Vector2D :=
  { x := v1.x + v2.x, y := v1.y + v2.y }

/-- Scalar multiplication of a vector -/
def scalarMul (s : ℝ) (v : Vector2D) : Vector2D :=
  { x := s * v.x, y := s * v.y }

/-- The main theorem to prove -/
theorem quadrilateral_reconstruction 
  (Q : ExtendedQuadrilateral) 
  (h1 : Q.A' = vectorAdd Q.Q.A (scalarMul 1 (vectorAdd Q.Q.B (scalarMul (-1) Q.Q.A))))
  (h2 : Q.D' = vectorAdd Q.Q.D (scalarMul 1 (vectorAdd Q.Q.C (scalarMul (-1) Q.Q.D))))
  : ∃ (p q r s : ℝ), 
    Q.Q.A = vectorAdd 
      (scalarMul p Q.A') 
      (vectorAdd 
        (scalarMul q Q.Q.B) 
        (vectorAdd 
          (scalarMul r Q.Q.C) 
          (scalarMul s Q.D')))
    ∧ p = 0 
    ∧ q = 0 
    ∧ r = 1/4 
    ∧ s = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_reconstruction_l3374_337429


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3374_337486

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - x + 1 > 0) ↔ (∃ x : ℝ, x^2 - x + 1 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3374_337486


namespace NUMINAMATH_CALUDE_least_positive_integer_to_multiple_of_five_l3374_337473

theorem least_positive_integer_to_multiple_of_five : 
  ∃ (n : ℕ), n > 0 ∧ (625 + n) % 5 = 0 ∧ ∀ (m : ℕ), m > 0 ∧ (625 + m) % 5 = 0 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_to_multiple_of_five_l3374_337473


namespace NUMINAMATH_CALUDE_abs_increasing_on_unit_interval_l3374_337452

/-- The function f(x) = |x| is increasing on the interval (0,1) -/
theorem abs_increasing_on_unit_interval :
  ∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → x₂ < 1 → |x₁| < |x₂| := by
  sorry

end NUMINAMATH_CALUDE_abs_increasing_on_unit_interval_l3374_337452


namespace NUMINAMATH_CALUDE_certain_number_proof_l3374_337474

theorem certain_number_proof (w : ℕ) (n : ℕ) : 
  w > 0 ∧ 
  168 ≤ w ∧
  ∃ (k : ℕ), k > 0 ∧ n * w = k * 2^5 ∧
  ∃ (l : ℕ), l > 0 ∧ n * w = l * 3^3 ∧
  ∃ (m : ℕ), m > 0 ∧ n * w = m * 14^2 →
  n = 1008 := by
sorry

end NUMINAMATH_CALUDE_certain_number_proof_l3374_337474


namespace NUMINAMATH_CALUDE_vasya_arrives_first_l3374_337414

/-- Represents the relative step length and step count of two people walking to school -/
structure WalkingData where
  vasya_step_length : ℝ
  petya_step_length : ℝ
  vasya_step_count : ℝ
  petya_step_count : ℝ

/-- The conditions of the problem -/
def walking_conditions (data : WalkingData) : Prop :=
  data.vasya_step_length > 0 ∧
  data.petya_step_length = 0.75 * data.vasya_step_length ∧
  data.petya_step_count = 1.25 * data.vasya_step_count

/-- Theorem stating that Vasya travels further in the same time -/
theorem vasya_arrives_first (data : WalkingData) 
  (h : walking_conditions data) : 
  data.vasya_step_length * data.vasya_step_count > 
  data.petya_step_length * data.petya_step_count := by
  sorry

#check vasya_arrives_first

end NUMINAMATH_CALUDE_vasya_arrives_first_l3374_337414


namespace NUMINAMATH_CALUDE_vova_gave_three_l3374_337427

/-- Represents the number of nuts Vova gave to Pavlik -/
def k : ℕ := sorry

/-- Represents Vova's initial number of nuts -/
def V : ℕ := sorry

/-- Represents Pavlik's initial number of nuts -/
def P : ℕ := sorry

/-- Vova has more nuts than Pavlik -/
axiom vova_more : V > P

/-- If Vova gave Pavlik as many nuts as Pavlik had, they would have the same number -/
axiom equal_after_giving : V - P = P + P

/-- Vova gave Pavlik no more than 5 nuts -/
axiom k_at_most_5 : k ≤ 5

/-- The remaining nuts were divided equally among 3 squirrels -/
axiom divisible_by_3 : (V - k) % 3 = 0

/-- The number of nuts Vova gave to Pavlik is 3 -/
theorem vova_gave_three : k = 3 := by sorry

end NUMINAMATH_CALUDE_vova_gave_three_l3374_337427


namespace NUMINAMATH_CALUDE_binomial_30_3_l3374_337408

theorem binomial_30_3 : Nat.choose 30 3 = 4060 := by
  sorry

end NUMINAMATH_CALUDE_binomial_30_3_l3374_337408


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l3374_337455

/-- Given two arithmetic sequences and their sum properties, prove the ratio of their 7th terms -/
theorem arithmetic_sequence_ratio (a b : ℕ → ℚ) (S T : ℕ → ℚ) :
  (∀ n : ℕ, n > 0 → S n / T n = (3 * n + 5 : ℚ) / (2 * n + 3)) →
  (∀ n : ℕ, S n = (n : ℚ) * (a 1 + a n) / 2) →
  (∀ n : ℕ, T n = (n : ℚ) * (b 1 + b n) / 2) →
  a 7 / b 7 = 44 / 29 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l3374_337455


namespace NUMINAMATH_CALUDE_sin_equality_implies_zero_l3374_337439

theorem sin_equality_implies_zero (n : ℤ) (h1 : -90 ≤ n) (h2 : n ≤ 90) 
  (h3 : Real.sin (n * π / 180) = Real.sin (720 * π / 180)) : n = 0 :=
by sorry

end NUMINAMATH_CALUDE_sin_equality_implies_zero_l3374_337439


namespace NUMINAMATH_CALUDE_sum_of_cubes_roots_l3374_337421

/-- For a quadratic equation x^2 + ax + a + 1 = 0, the sum of cubes of its roots equals 1 iff a = -1 -/
theorem sum_of_cubes_roots (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁^2 + a*x₁ + a + 1 = 0 ∧ x₂^2 + a*x₂ + a + 1 = 0 ∧ x₁^3 + x₂^3 = 1) ↔ a = -1 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_cubes_roots_l3374_337421


namespace NUMINAMATH_CALUDE_box_counting_l3374_337495

theorem box_counting (initial_boxes : ℕ) (boxes_per_filled : ℕ) (non_empty_boxes : ℕ) : 
  initial_boxes = 7 →
  boxes_per_filled = 7 →
  non_empty_boxes = 10 →
  initial_boxes + (boxes_per_filled * non_empty_boxes) = 77 :=
by sorry

end NUMINAMATH_CALUDE_box_counting_l3374_337495


namespace NUMINAMATH_CALUDE_chessboard_parallelogram_l3374_337418

/-- Represents a chess piece placement on an n×n board -/
structure Placement (n : ℕ) where
  pieces : Finset (Fin n × Fin n)

/-- Checks if four pieces form a parallelogram -/
def is_parallelogram (n : ℕ) (p1 p2 p3 p4 : Fin n × Fin n) : Prop :=
  (p1.1 + p3.1 = p2.1 + p4.1) ∧ (p1.2 + p3.2 = p2.2 + p4.2)

/-- The main theorem about chess piece placements -/
theorem chessboard_parallelogram (n : ℕ) (h : n > 1) :
  (∀ (p : Placement n), p.pieces.card = 2 * n →
    ∃ (p1 p2 p3 p4 : Fin n × Fin n),
      p1 ∈ p.pieces ∧ p2 ∈ p.pieces ∧ p3 ∈ p.pieces ∧ p4 ∈ p.pieces ∧
      is_parallelogram n p1 p2 p3 p4) ∧
  (∃ (p : Placement n), p.pieces.card = 2 * n - 1 ∧
    ∀ (p1 p2 p3 p4 : Fin n × Fin n),
      p1 ∈ p.pieces → p2 ∈ p.pieces → p3 ∈ p.pieces → p4 ∈ p.pieces →
      ¬is_parallelogram n p1 p2 p3 p4) :=
sorry

end NUMINAMATH_CALUDE_chessboard_parallelogram_l3374_337418


namespace NUMINAMATH_CALUDE_dianas_age_l3374_337438

theorem dianas_age (Carlos Diana Emily : ℚ) 
  (h1 : Carlos = 4 * Diana)
  (h2 : Emily = Diana + 5)
  (h3 : Carlos = Emily) : 
  Diana = 5 / 3 := by
sorry

end NUMINAMATH_CALUDE_dianas_age_l3374_337438


namespace NUMINAMATH_CALUDE_line_intersection_y_axis_l3374_337410

/-- A line passing through two points intersects the y-axis at a specific point -/
theorem line_intersection_y_axis (x₁ y₁ x₂ y₂ : ℝ) (hx : x₁ ≠ x₂) :
  let m := (y₂ - y₁) / (x₂ - x₁)
  let b := y₁ - m * x₁
  (x₁ = 2 ∧ y₁ = 9 ∧ x₂ = 4 ∧ y₂ = 17) →
  (0, m * 0 + b) = (0, 1) :=
sorry

end NUMINAMATH_CALUDE_line_intersection_y_axis_l3374_337410


namespace NUMINAMATH_CALUDE_triangle_isosceles_condition_l3374_337464

/-- A triangle with sides a, b, and c is isosceles if at least two of its sides are equal. -/
def IsIsosceles (a b c : ℝ) : Prop :=
  a = b ∨ b = c ∨ a = c

/-- If the three sides a, b, and c of a triangle ABC satisfy (a-b)(b²-2bc+c²) = 0,
    then the triangle ABC is isosceles. -/
theorem triangle_isosceles_condition (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) 
    (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
    (h_condition : (a - b) * (b^2 - 2*b*c + c^2) = 0) : 
    IsIsosceles a b c := by
  sorry


end NUMINAMATH_CALUDE_triangle_isosceles_condition_l3374_337464


namespace NUMINAMATH_CALUDE_prob_two_bags_theorem_l3374_337400

/-- Represents a bag of colored balls -/
structure Bag where
  red : Nat
  white : Nat

/-- Calculates the probability of drawing at least one white ball from two bags -/
def prob_at_least_one_white (bagA bagB : Bag) : Rat :=
  let total_outcomes := bagA.red * bagB.red + bagA.red * bagB.white + bagA.white * bagB.red + bagA.white * bagB.white
  let favorable_outcomes := bagA.white * bagB.red + bagA.red * bagB.white + bagA.white * bagB.white
  favorable_outcomes / total_outcomes

/-- The main theorem to prove -/
theorem prob_two_bags_theorem (bagA bagB : Bag) 
    (h1 : bagA.red = 3) (h2 : bagA.white = 2) 
    (h3 : bagB.red = 2) (h4 : bagB.white = 1) : 
    prob_at_least_one_white bagA bagB = 3/5 := by
  sorry

#eval prob_at_least_one_white ⟨3, 2⟩ ⟨2, 1⟩

end NUMINAMATH_CALUDE_prob_two_bags_theorem_l3374_337400


namespace NUMINAMATH_CALUDE_girls_in_school_l3374_337444

/-- The number of girls in a school with a given total number of pupils and boys -/
def number_of_girls (total_pupils : ℕ) (number_of_boys : ℕ) : ℕ :=
  total_pupils - number_of_boys

/-- Theorem stating that the number of girls in the school is 232 -/
theorem girls_in_school : number_of_girls 485 253 = 232 := by
  sorry

end NUMINAMATH_CALUDE_girls_in_school_l3374_337444


namespace NUMINAMATH_CALUDE_orthographic_projection_properties_l3374_337487

-- Define the basic structure for a view in orthographic projection
structure View where
  length : ℝ
  width : ℝ
  height : ℝ

-- Define the orthographic projection
structure OrthographicProjection where
  main_view : View
  top_view : View
  left_view : View

-- Define the properties of orthographic projection
def is_valid_orthographic_projection (op : OrthographicProjection) : Prop :=
  -- Main view and top view have aligned lengths
  op.main_view.length = op.top_view.length ∧
  -- Main view and left view are height level
  op.main_view.height = op.left_view.height ∧
  -- Left view and top view have equal widths
  op.left_view.width = op.top_view.width

-- Theorem statement
theorem orthographic_projection_properties (op : OrthographicProjection) 
  (h : is_valid_orthographic_projection op) :
  op.main_view.length = op.top_view.length ∧
  op.main_view.height = op.left_view.height ∧
  op.left_view.width = op.top_view.width := by
  sorry

end NUMINAMATH_CALUDE_orthographic_projection_properties_l3374_337487


namespace NUMINAMATH_CALUDE_pentagon_area_theorem_l3374_337415

theorem pentagon_area_theorem (u v : ℤ) 
  (h1 : 0 < v) (h2 : v < u) 
  (h3 : (2 * u * v : ℤ) + (8 * u * v : ℤ) = 902) : 
  2 * u + v = 29 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_area_theorem_l3374_337415


namespace NUMINAMATH_CALUDE_ellipse_max_value_l3374_337491

theorem ellipse_max_value (x y : ℝ) : 
  ((x - 4)^2 / 4 + y^2 / 9 = 1) → (x^2 / 4 + y^2 / 9 ≤ 9) ∧ (∃ x y : ℝ, ((x - 4)^2 / 4 + y^2 / 9 = 1) ∧ (x^2 / 4 + y^2 / 9 = 9)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_max_value_l3374_337491


namespace NUMINAMATH_CALUDE_parallel_vectors_k_value_l3374_337458

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), a.1 * b.2 = t * a.2 * b.1

/-- The given vectors -/
def a : ℝ × ℝ := (6, 2)
def b (k : ℝ) : ℝ × ℝ := (-3, k)

/-- The theorem stating that if the given vectors are parallel, then k = -1 -/
theorem parallel_vectors_k_value :
  parallel a (b k) → k = -1 := by sorry

end NUMINAMATH_CALUDE_parallel_vectors_k_value_l3374_337458


namespace NUMINAMATH_CALUDE_complement_of_A_range_of_a_l3374_337446

-- Define the sets A and B
def A : Set ℝ := {x | x^2 + 4*x > 0}
def B (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < a + 1}

-- Define the universal set U
def U : Set ℝ := Set.univ

-- Theorem for the complement of A
theorem complement_of_A : (Aᶜ : Set ℝ) = {x : ℝ | -4 ≤ x ∧ x ≤ 0} := by sorry

-- Theorem for the range of a
theorem range_of_a (a : ℝ) : B a ⊆ Aᶜ ↔ -3 ≤ a ∧ a ≤ -1 := by sorry

end NUMINAMATH_CALUDE_complement_of_A_range_of_a_l3374_337446


namespace NUMINAMATH_CALUDE_tan_2715_degrees_l3374_337490

theorem tan_2715_degrees : Real.tan (2715 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_2715_degrees_l3374_337490


namespace NUMINAMATH_CALUDE_circle_center_symmetry_l3374_337494

/-- Given an equation of the form x^2 + y^2 + 2ax - 2ay = 0 where a ≠ 0,
    representing a circle, prove that the center of this circle lies on the line x + y = 0 -/
theorem circle_center_symmetry (a : ℝ) (h : a ≠ 0) :
  let f : ℝ × ℝ → ℝ := fun (x, y) ↦ x^2 + y^2 + 2*a*x - 2*a*y
  let center : ℝ × ℝ := (-a, a)
  (∀ x y, f (x, y) = 0 → (x - center.1)^2 + (y - center.2)^2 = 2 * a^2) →
  center.1 + center.2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_symmetry_l3374_337494


namespace NUMINAMATH_CALUDE_fixed_point_theorem_l3374_337482

theorem fixed_point_theorem (f : ℝ → ℝ) :
  Continuous f →
  (∀ x ∈ Set.Icc 0 1, f x ∈ Set.Icc 0 1) →
  ∃ x ∈ Set.Icc 0 1, f x = x := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_theorem_l3374_337482


namespace NUMINAMATH_CALUDE_inverse_of_M_l3374_337416

/-- The line 2x - y = 3 -/
def line (x y : ℝ) : Prop := 2 * x - y = 3

/-- The matrix M -/
def M (a b : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![-1, a],
    ![b, 3]]

/-- M maps the line onto itself -/
def M_maps_line (a b : ℝ) : Prop :=
  ∀ x y : ℝ, line x y → line (-x + a*y) (b*x + 3*y)

/-- The inverse of M -/
def M_inv : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![3, -1],
    ![4, -1]]

theorem inverse_of_M (a b : ℝ) (h : M_maps_line a b) :
  M a b * M_inv = 1 ∧ M_inv * M a b = 1 := by
  sorry

end NUMINAMATH_CALUDE_inverse_of_M_l3374_337416


namespace NUMINAMATH_CALUDE_specific_league_games_l3374_337498

/-- Represents a sports league with the given conditions -/
structure SportsLeague where
  total_teams : Nat
  divisions : Nat
  teams_per_division : Nat
  intra_division_games : Nat
  inter_division_games : Nat

/-- Calculate the total number of games in a complete season -/
def total_games (league : SportsLeague) : Nat :=
  -- We'll implement the calculation here
  sorry

/-- Theorem stating the total number of games in the specific league configuration -/
theorem specific_league_games :
  let league : SportsLeague := {
    total_teams := 20,
    divisions := 4,
    teams_per_division := 5,
    intra_division_games := 3,
    inter_division_games := 1
  }
  total_games league = 270 := by
  sorry

end NUMINAMATH_CALUDE_specific_league_games_l3374_337498


namespace NUMINAMATH_CALUDE_circle_intersection_angle_equality_l3374_337409

-- Define the types for points and circles
variable (Point Circle : Type)
[MetricSpace Point]

-- Define the intersection function
variable (intersect : Circle → Circle → Set Point)

-- Define the center of a circle
variable (center : Circle → Point)

-- Define the angle between three points
variable (angle : Point → Point → Point → ℝ)

-- State the theorem
theorem circle_intersection_angle_equality
  (c₁ c₂ c₃ : Circle)
  (P Q A B C D : Point)
  (h₁ : P ∈ intersect c₁ c₂)
  (h₂ : Q ∈ intersect c₁ c₂)
  (h₃ : center c₃ = P)
  (h₄ : A ∈ intersect c₁ c₃)
  (h₅ : B ∈ intersect c₁ c₃)
  (h₆ : C ∈ intersect c₂ c₃)
  (h₇ : D ∈ intersect c₂ c₃) :
  angle A Q D = angle B Q C :=
sorry

end NUMINAMATH_CALUDE_circle_intersection_angle_equality_l3374_337409


namespace NUMINAMATH_CALUDE_reading_time_difference_problem_l3374_337411

/-- The difference in reading time between two people reading the same book -/
def reading_time_difference (ken_speed lisa_speed book_pages : ℕ) : ℕ :=
  let ken_time := book_pages / ken_speed
  let lisa_time := book_pages / lisa_speed
  (lisa_time - ken_time) * 60

/-- Theorem stating the difference in reading time for the given problem -/
theorem reading_time_difference_problem :
  reading_time_difference 75 60 360 = 72 := by
  sorry

end NUMINAMATH_CALUDE_reading_time_difference_problem_l3374_337411


namespace NUMINAMATH_CALUDE_machine_count_l3374_337493

theorem machine_count (x : ℝ) : 
  ∃ (N : ℝ) (r : ℝ),
    N * r * 8 = x ∧ 
    30 * r * 4 = 3 * x ∧ 
    N = 5 := by
  sorry

end NUMINAMATH_CALUDE_machine_count_l3374_337493


namespace NUMINAMATH_CALUDE_scarf_parity_l3374_337436

theorem scarf_parity (initial_count : ℕ) (actions : ℕ) (final_count : ℕ) : 
  initial_count % 2 = 0 → 
  actions % 2 = 1 → 
  (∃ (changes : List ℤ), 
    changes.length = actions ∧ 
    (∀ c ∈ changes, c = 1 ∨ c = -1) ∧
    final_count = initial_count + changes.sum) →
  final_count % 2 = 1 :=
by sorry

#check scarf_parity 20 17 10

end NUMINAMATH_CALUDE_scarf_parity_l3374_337436


namespace NUMINAMATH_CALUDE_volume_removed_is_two_l3374_337412

/-- Represents a cube with corner cuts -/
structure CutCube where
  side : ℝ
  cut_depth : ℝ
  face_square_side : ℝ

/-- Calculates the volume of material removed from a cut cube -/
def volume_removed (c : CutCube) : ℝ :=
  8 * (c.side - c.face_square_side) * (c.side - c.face_square_side) * c.cut_depth

/-- Theorem stating the volume removed from a 2x2x2 cube with specific cuts is 2 cubic units -/
theorem volume_removed_is_two :
  let c : CutCube := ⟨2, 1, 1⟩
  volume_removed c = 2 := by
  sorry


end NUMINAMATH_CALUDE_volume_removed_is_two_l3374_337412


namespace NUMINAMATH_CALUDE_only_statements_3_and_4_are_propositions_l3374_337441

-- Define a type for our statements
inductive Statement
  | equation : Statement
  | question : Statement
  | arithmeticFalse : Statement
  | universalFalse : Statement

-- Define a function to check if a statement is a proposition
def isProposition (s : Statement) : Prop :=
  match s with
  | Statement.equation => False
  | Statement.question => False
  | Statement.arithmeticFalse => True
  | Statement.universalFalse => True

-- Define our statements
def statement1 : Statement := Statement.equation
def statement2 : Statement := Statement.question
def statement3 : Statement := Statement.arithmeticFalse
def statement4 : Statement := Statement.universalFalse

-- Theorem to prove
theorem only_statements_3_and_4_are_propositions :
  (isProposition statement1 = False) ∧
  (isProposition statement2 = False) ∧
  (isProposition statement3 = True) ∧
  (isProposition statement4 = True) :=
sorry

end NUMINAMATH_CALUDE_only_statements_3_and_4_are_propositions_l3374_337441


namespace NUMINAMATH_CALUDE_pentagon_angle_sum_l3374_337430

theorem pentagon_angle_sum (a b c d q : ℝ) : 
  a = 118 → b = 105 → c = 87 → d = 135 →
  (a + b + c + d + q = 540) →
  q = 95 := by sorry

end NUMINAMATH_CALUDE_pentagon_angle_sum_l3374_337430


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3374_337402

theorem sqrt_equation_solution :
  ∃! x : ℝ, (Real.sqrt x + 2 * Real.sqrt (x^2 + 9*x) + Real.sqrt (x + 9) = 45 - 2*x) ∧ 
             (x = 729/144) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3374_337402


namespace NUMINAMATH_CALUDE_complement_of_intersection_AB_l3374_337424

open Set

-- Define the universal set U as the set of real numbers
def U : Set ℝ := univ

-- Define set A
def A : Set ℝ := {x | x ≥ 1}

-- Define set B
def B : Set ℝ := {x | -1 ≤ x ∧ x < 2}

-- State the theorem
theorem complement_of_intersection_AB : 
  (A ∩ B)ᶜ = {x : ℝ | x < 1 ∨ x ≥ 2} := by sorry

end NUMINAMATH_CALUDE_complement_of_intersection_AB_l3374_337424


namespace NUMINAMATH_CALUDE_real_part_of_complex_product_l3374_337469

theorem real_part_of_complex_product : 
  let z : ℂ := (1 + 2*Complex.I) * (3 - Complex.I)
  Complex.re z = 5 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_complex_product_l3374_337469


namespace NUMINAMATH_CALUDE_date_calculation_l3374_337467

/-- Given that December 1, 2015 was a Tuesday (day 2 of the week) and there are 31 days between
    December 1, 2015 and January 1, 2016, prove that January 1, 2016 was a Friday (day 5 of the week) -/
theorem date_calculation (start_day : Nat) (days_between : Nat) (end_day : Nat) : 
  start_day = 2 → days_between = 31 → end_day = (start_day + days_between) % 7 → end_day = 5 := by
  sorry

#check date_calculation

end NUMINAMATH_CALUDE_date_calculation_l3374_337467


namespace NUMINAMATH_CALUDE_incorrect_inequality_l3374_337475

theorem incorrect_inequality (a b : ℝ) (h : a > b) : ¬ (-2 * a > -2 * b) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_inequality_l3374_337475


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l3374_337450

/-- The perimeter of a rhombus with diagonals of 10 inches and 24 inches is 52 inches. -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) : 
  4 * Real.sqrt ((d1/2)^2 + (d2/2)^2) = 52 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l3374_337450


namespace NUMINAMATH_CALUDE_at_least_three_pass_six_students_l3374_337422

def exam_pass_probability : ℚ := 1/3

def at_least_three_pass (n : ℕ) (p : ℚ) : ℚ :=
  1 - (Nat.choose n 0 * (1 - p)^n +
       Nat.choose n 1 * p * (1 - p)^(n-1) +
       Nat.choose n 2 * p^2 * (1 - p)^(n-2))

theorem at_least_three_pass_six_students :
  at_least_three_pass 6 exam_pass_probability = 353/729 := by
  sorry

end NUMINAMATH_CALUDE_at_least_three_pass_six_students_l3374_337422


namespace NUMINAMATH_CALUDE_valid_arrangements_l3374_337406

/-- The number of ways to arrange 7 distinct digits with 1 to the left of 2 and 3 -/
def arrange_digits : ℕ :=
  (Nat.choose 7 3) * (Nat.factorial 4)

/-- Theorem stating that there are 840 valid arrangements -/
theorem valid_arrangements : arrange_digits = 840 := by
  sorry

end NUMINAMATH_CALUDE_valid_arrangements_l3374_337406


namespace NUMINAMATH_CALUDE_problem_solution_l3374_337488

theorem problem_solution (a b : ℝ) : 
  |a + 1| + (b - 2)^2 = 0 → 
  a = -1 ∧ b = 2 ∧ (a + b)^2020 + a^2019 = 0 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3374_337488

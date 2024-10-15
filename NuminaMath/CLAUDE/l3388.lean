import Mathlib

namespace NUMINAMATH_CALUDE_square_sum_geq_neg_double_product_l3388_338871

theorem square_sum_geq_neg_double_product (a b : ℝ) : a^2 + b^2 ≥ -2*a*b := by
  sorry

end NUMINAMATH_CALUDE_square_sum_geq_neg_double_product_l3388_338871


namespace NUMINAMATH_CALUDE_count_integer_solutions_l3388_338856

theorem count_integer_solutions : ∃! (s : Finset (ℕ × ℕ)), 
  (∀ (p : ℕ × ℕ), p ∈ s ↔ p.1 > 0 ∧ p.2 > 0 ∧ 8 / p.1 + 6 / p.2 = 1) ∧ 
  s.card = 5 := by
sorry

end NUMINAMATH_CALUDE_count_integer_solutions_l3388_338856


namespace NUMINAMATH_CALUDE_probability_all_genuine_proof_l3388_338817

/-- The total number of coins -/
def total_coins : ℕ := 15

/-- The number of genuine coins -/
def genuine_coins : ℕ := 12

/-- The number of counterfeit coins -/
def counterfeit_coins : ℕ := 3

/-- The number of pairs selected -/
def pairs_selected : ℕ := 3

/-- The number of coins in each pair -/
def coins_per_pair : ℕ := 2

/-- Predicate that the weight of counterfeit coins is different from genuine coins -/
axiom counterfeit_weight_different : True

/-- Predicate that the combined weight of all three pairs is the same -/
axiom combined_weight_same : True

/-- The probability of selecting all genuine coins given the conditions -/
def probability_all_genuine : ℚ := 264 / 443

/-- Theorem stating that the probability of selecting all genuine coins
    given the conditions is equal to 264/443 -/
theorem probability_all_genuine_proof :
  probability_all_genuine = 264 / 443 :=
by sorry

end NUMINAMATH_CALUDE_probability_all_genuine_proof_l3388_338817


namespace NUMINAMATH_CALUDE_matrix_sum_theorem_l3388_338816

theorem matrix_sum_theorem (a b c : ℝ) : 
  let M : Matrix (Fin 3) (Fin 3) ℝ := !![a^2, b^2, c^2; b^2, c^2, a^2; c^2, a^2, b^2]
  ¬(IsUnit (M.det)) →
  (a^2 / (b^2 + c^2) + b^2 / (a^2 + c^2) + c^2 / (a^2 + b^2) = 3/2) ∨
  (a^2 / (b^2 + c^2) + b^2 / (a^2 + c^2) + c^2 / (a^2 + b^2) = -3) :=
by sorry


end NUMINAMATH_CALUDE_matrix_sum_theorem_l3388_338816


namespace NUMINAMATH_CALUDE_vasya_problem_impossible_l3388_338887

theorem vasya_problem_impossible : ¬ ∃ (x₁ x₂ x₃ : ℕ), 
  x₁ + 3 * x₂ + 15 * x₃ = 100 ∧ 11 * x₁ + 8 * x₂ = 144 := by
  sorry

end NUMINAMATH_CALUDE_vasya_problem_impossible_l3388_338887


namespace NUMINAMATH_CALUDE_cos_2x_value_l3388_338895

theorem cos_2x_value (x : ℝ) (h : 2 * Real.sin (Real.pi - x) + 1 = 0) : 
  Real.cos (2 * x) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_2x_value_l3388_338895


namespace NUMINAMATH_CALUDE_room_width_is_four_meters_l3388_338813

/-- Proves that the width of a rectangular room is 4 meters given the specified conditions -/
theorem room_width_is_four_meters 
  (length : ℝ) 
  (cost_per_sqm : ℝ) 
  (total_cost : ℝ) 
  (h1 : length = 5.5)
  (h2 : cost_per_sqm = 700)
  (h3 : total_cost = 15400) :
  ∃ (width : ℝ), width = 4 ∧ length * width * cost_per_sqm = total_cost :=
by sorry

end NUMINAMATH_CALUDE_room_width_is_four_meters_l3388_338813


namespace NUMINAMATH_CALUDE_kenya_has_133_peanuts_l3388_338860

-- Define the number of peanuts Jose has
def jose_peanuts : ℕ := 85

-- Define the difference in peanuts between Kenya and Jose
def peanut_difference : ℕ := 48

-- Define Kenya's peanuts in terms of Jose's peanuts and the difference
def kenya_peanuts : ℕ := jose_peanuts + peanut_difference

-- Theorem stating that Kenya has 133 peanuts
theorem kenya_has_133_peanuts : kenya_peanuts = 133 := by
  sorry

end NUMINAMATH_CALUDE_kenya_has_133_peanuts_l3388_338860


namespace NUMINAMATH_CALUDE_tangent_product_equals_two_l3388_338857

theorem tangent_product_equals_two (x y : Real) 
  (h1 : x = 21 * π / 180) 
  (h2 : y = 24 * π / 180) 
  (h3 : Real.tan (π / 4) = 1) 
  (h4 : π / 4 = x + y) : 
  (1 + Real.tan x) * (1 + Real.tan y) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_product_equals_two_l3388_338857


namespace NUMINAMATH_CALUDE_intersection_equality_implies_range_l3388_338851

theorem intersection_equality_implies_range (a : ℝ) : 
  (∀ x : ℝ, (1 ≤ x ∧ x ≤ 2) ↔ (1 ≤ x ∧ x ≤ 2 ∧ 2 - a ≤ x ∧ x ≤ 1 + a)) →
  a ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_intersection_equality_implies_range_l3388_338851


namespace NUMINAMATH_CALUDE_spherical_coordinates_conversion_l3388_338891

/-- Converts non-standard spherical coordinates to standard form -/
def standardize_spherical (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  sorry

/-- Checks if spherical coordinates are in standard form -/
def is_standard_form (ρ θ φ : ℝ) : Prop :=
  ρ > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧ 0 ≤ φ ∧ φ ≤ Real.pi

theorem spherical_coordinates_conversion :
  let original := (5, 5 * Real.pi / 6, 9 * Real.pi / 4)
  let standard := (5, 11 * Real.pi / 6, 3 * Real.pi / 4)
  standardize_spherical original.1 original.2.1 original.2.2 = standard ∧
  is_standard_form standard.1 standard.2.1 standard.2.2 :=
sorry

end NUMINAMATH_CALUDE_spherical_coordinates_conversion_l3388_338891


namespace NUMINAMATH_CALUDE_inequality_proof_l3388_338866

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x + y + z ≥ 3) :
  1 / (x + y + z^2) + 1 / (y + z + x^2) + 1 / (z + x + y^2) ≤ 1 ∧
  (1 / (x + y + z^2) + 1 / (y + z + x^2) + 1 / (z + x + y^2) = 1 ↔ x = 1 ∧ y = 1 ∧ z = 1) :=
by sorry


end NUMINAMATH_CALUDE_inequality_proof_l3388_338866


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l3388_338807

theorem complex_modulus_problem (z : ℂ) : 
  ((1 + Complex.I) / (1 - Complex.I)) * z = 3 + 4 * Complex.I → Complex.abs z = 5 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l3388_338807


namespace NUMINAMATH_CALUDE_profit_and_sales_maximization_max_profit_with_constraints_l3388_338889

/-- Represents the daily sales quantity as a function of the selling price -/
def sales_quantity (x : ℝ) : ℝ := -10 * x + 400

/-- Represents the daily profit as a function of the selling price -/
def profit (x : ℝ) : ℝ := sales_quantity x * (x - 10)

/-- The cost price of the item -/
def cost_price : ℝ := 10

/-- The domain constraints for the selling price -/
def price_domain (x : ℝ) : Prop := 10 < x ∧ x ≤ 40

theorem profit_and_sales_maximization (x : ℝ) 
  (h : price_domain x) : 
  profit x = 1250 ∧ 
  (∀ y, price_domain y → sales_quantity x ≥ sales_quantity y) → 
  x = 15 :=
sorry

theorem max_profit_with_constraints (x : ℝ) :
  28 ≤ x ∧ x ≤ 35 →
  profit x ≤ 2160 :=
sorry

end NUMINAMATH_CALUDE_profit_and_sales_maximization_max_profit_with_constraints_l3388_338889


namespace NUMINAMATH_CALUDE_smallest_a_value_l3388_338874

/-- Given a polynomial x^3 - ax^2 + bx - 2550 with three positive integer roots,
    the smallest possible value of a is 62 -/
theorem smallest_a_value (a b : ℤ) (r s t : ℕ+) : 
  (∀ x, x^3 - a*x^2 + b*x - 2550 = (x - r.val)*(x - s.val)*(x - t.val)) →
  a = r.val + s.val + t.val →
  62 ≤ a :=
sorry

end NUMINAMATH_CALUDE_smallest_a_value_l3388_338874


namespace NUMINAMATH_CALUDE_vasyas_numbers_l3388_338832

theorem vasyas_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x + y = x / y) (h3 : y ≠ 0) :
  x = (1 : ℝ) / 2 ∧ y = -1 := by
  sorry

end NUMINAMATH_CALUDE_vasyas_numbers_l3388_338832


namespace NUMINAMATH_CALUDE_cost_of_500_cookies_l3388_338849

/-- The cost in dollars for buying a number of cookies -/
def cookie_cost (num_cookies : ℕ) : ℚ :=
  (num_cookies * 2) / 100

/-- Proof that buying 500 cookies costs 10 dollars -/
theorem cost_of_500_cookies : cookie_cost 500 = 10 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_500_cookies_l3388_338849


namespace NUMINAMATH_CALUDE_divisibility_equivalence_l3388_338859

theorem divisibility_equivalence (n : ℤ) : 
  let A := n % 1000
  let B := n / 1000
  let k := A - B
  (7 ∣ n ∨ 11 ∣ n ∨ 13 ∣ n) ↔ (7 ∣ k ∨ 11 ∣ k ∨ 13 ∣ k) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_equivalence_l3388_338859


namespace NUMINAMATH_CALUDE_equation_system_solution_l3388_338879

theorem equation_system_solution (a b : ℝ) : 
  ((a / 4 - 1) + 2 * (b / 3 + 2) = 4 ∧ 2 * (a / 4 - 1) + (b / 3 + 2) = 5) → 
  (a = 12 ∧ b = -3) := by sorry

end NUMINAMATH_CALUDE_equation_system_solution_l3388_338879


namespace NUMINAMATH_CALUDE_carolyn_practice_time_l3388_338806

/-- Calculates the total practice time for Carolyn in a month --/
def total_practice_time (piano_time : ℕ) (violin_multiplier : ℕ) (practice_days_per_week : ℕ) (weeks_in_month : ℕ) : ℕ :=
  let violin_time := piano_time * violin_multiplier
  let daily_practice_time := piano_time + violin_time
  let weekly_practice_time := daily_practice_time * practice_days_per_week
  weekly_practice_time * weeks_in_month

/-- Proves that Carolyn's total practice time in a month with 4 weeks is 1920 minutes --/
theorem carolyn_practice_time :
  total_practice_time 20 3 6 4 = 1920 := by
  sorry

end NUMINAMATH_CALUDE_carolyn_practice_time_l3388_338806


namespace NUMINAMATH_CALUDE_square_completion_l3388_338800

theorem square_completion (a h k : ℝ) : 
  (∀ x, x^2 - 6*x = a*(x - h)^2 + k) → k = -9 := by
  sorry

end NUMINAMATH_CALUDE_square_completion_l3388_338800


namespace NUMINAMATH_CALUDE_prob_neither_red_nor_green_l3388_338885

/-- Given a bag with green, black, and red pens, this theorem proves the probability
    of picking a pen that is neither red nor green. -/
theorem prob_neither_red_nor_green (green black red : ℕ) 
  (h_green : green = 5) 
  (h_black : black = 6) 
  (h_red : red = 7) : 
  (black : ℚ) / (green + black + red) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_prob_neither_red_nor_green_l3388_338885


namespace NUMINAMATH_CALUDE_pablo_puzzle_speed_l3388_338809

/-- Represents the number of pieces Pablo can put together per hour -/
def pieces_per_hour : ℕ := sorry

/-- The number of puzzles with 300 pieces -/
def puzzles_300 : ℕ := 8

/-- The number of puzzles with 500 pieces -/
def puzzles_500 : ℕ := 5

/-- The number of pieces in a 300-piece puzzle -/
def pieces_300 : ℕ := 300

/-- The number of pieces in a 500-piece puzzle -/
def pieces_500 : ℕ := 500

/-- The maximum number of hours Pablo works each day -/
def hours_per_day : ℕ := 7

/-- The number of days it takes Pablo to complete all puzzles -/
def days_to_complete : ℕ := 7

/-- The total number of pieces in all puzzles -/
def total_pieces : ℕ := puzzles_300 * pieces_300 + puzzles_500 * pieces_500

/-- The total number of hours Pablo spends on puzzles -/
def total_hours : ℕ := hours_per_day * days_to_complete

theorem pablo_puzzle_speed : pieces_per_hour = 100 := by
  sorry

end NUMINAMATH_CALUDE_pablo_puzzle_speed_l3388_338809


namespace NUMINAMATH_CALUDE_no_prime_solution_l3388_338865

theorem no_prime_solution : ¬ ∃ (p : ℕ), 
  Nat.Prime p ∧ 
  p > 8 ∧ 
  2 * p^3 + 7 * p^2 + 6 * p + 20 = 6 * p^2 + 19 * p + 10 := by
sorry

end NUMINAMATH_CALUDE_no_prime_solution_l3388_338865


namespace NUMINAMATH_CALUDE_difference_of_squares_l3388_338848

theorem difference_of_squares : 75^2 - 25^2 = 5000 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l3388_338848


namespace NUMINAMATH_CALUDE_subtract_negatives_example_l3388_338876

theorem subtract_negatives_example : (-3) - (-5) = 2 := by
  sorry

end NUMINAMATH_CALUDE_subtract_negatives_example_l3388_338876


namespace NUMINAMATH_CALUDE_additional_marbles_for_lisa_l3388_338801

/-- The minimum number of additional marbles needed -/
def min_additional_marbles (num_friends : ℕ) (initial_marbles : ℕ) : ℕ :=
  (num_friends * (num_friends + 1)) / 2 - initial_marbles

/-- Theorem stating the minimum number of additional marbles needed -/
theorem additional_marbles_for_lisa : 
  min_additional_marbles 12 40 = 38 := by
  sorry

end NUMINAMATH_CALUDE_additional_marbles_for_lisa_l3388_338801


namespace NUMINAMATH_CALUDE_gamma_value_l3388_338840

theorem gamma_value (γ δ : ℂ) 
  (h1 : (γ + δ).re > 0)
  (h2 : (Complex.I * (γ - δ)).re > 0)
  (h3 : δ = 2 + 3 * Complex.I) : 
  γ = 2 - 3 * Complex.I := by
sorry

end NUMINAMATH_CALUDE_gamma_value_l3388_338840


namespace NUMINAMATH_CALUDE_arctan_sum_equation_l3388_338803

theorem arctan_sum_equation (y : ℝ) :
  2 * Real.arctan (1/3) + Real.arctan (1/15) + Real.arctan (1/y) = π/4 →
  y = -43/3 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_equation_l3388_338803


namespace NUMINAMATH_CALUDE_house_transaction_profit_l3388_338883

def initial_value : ℝ := 15000
def profit_percentage : ℝ := 0.20
def loss_percentage : ℝ := 0.15

theorem house_transaction_profit : 
  let first_sale := initial_value * (1 + profit_percentage)
  let second_sale := first_sale * (1 - loss_percentage)
  first_sale - second_sale = 2700 := by sorry

end NUMINAMATH_CALUDE_house_transaction_profit_l3388_338883


namespace NUMINAMATH_CALUDE_expansion_has_four_nonzero_terms_l3388_338824

def expansion (x : ℝ) : ℝ := (x^2 + 2) * (3*x^3 - x^2 + 4) - 2 * (x^4 - 3*x^3 + x^2)

def count_nonzero_terms (p : ℝ → ℝ) : ℕ := sorry

theorem expansion_has_four_nonzero_terms :
  count_nonzero_terms expansion = 4 := by sorry

end NUMINAMATH_CALUDE_expansion_has_four_nonzero_terms_l3388_338824


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l3388_338875

/-- Represents the side lengths of squares in ascending order -/
structure SquareSides where
  b₁ : ℝ
  b₂ : ℝ
  b₃ : ℝ
  b₄ : ℝ
  b₅ : ℝ
  b₆ : ℝ
  h_order : 0 < b₁ ∧ b₁ < b₂ ∧ b₂ < b₃ ∧ b₃ < b₄ ∧ b₄ < b₅ ∧ b₅ < b₆

/-- Represents a rectangle partitioned into six squares -/
structure PartitionedRectangle where
  sides : SquareSides
  length : ℝ
  width : ℝ
  h_partition : length = sides.b₃ + sides.b₆ ∧ width = sides.b₁ + sides.b₅
  h_sum_smallest : sides.b₁ + sides.b₂ = sides.b₃
  h_longest_side : 2 * length = 3 * sides.b₆

theorem rectangle_perimeter (rect : PartitionedRectangle) :
    2 * (rect.length + rect.width) = 12 * rect.sides.b₆ := by
  sorry


end NUMINAMATH_CALUDE_rectangle_perimeter_l3388_338875


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l3388_338831

theorem fraction_sum_equality (a b c : ℝ) 
  (h : a / (30 - a) + b / (75 - b) + c / (55 - c) = 8) :
  6 / (30 - a) + 15 / (75 - b) + 11 / (55 - c) = 187 / 30 :=
by sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l3388_338831


namespace NUMINAMATH_CALUDE_geometric_harmonic_mean_inequality_l3388_338819

theorem geometric_harmonic_mean_inequality {a b : ℝ} (ha : a ≥ 0) (hb : b ≥ 0) :
  Real.sqrt (a * b) ≥ 2 / (1 / a + 1 / b) ∧
  (Real.sqrt (a * b) = 2 / (1 / a + 1 / b) ↔ a = b) :=
by sorry

end NUMINAMATH_CALUDE_geometric_harmonic_mean_inequality_l3388_338819


namespace NUMINAMATH_CALUDE_tangent_cubic_to_line_l3388_338821

/-- Given that the graph of y = ax³ + 1 is tangent to the line y = x, prove that a = 4/27 -/
theorem tangent_cubic_to_line (a : ℝ) : 
  (∃ x : ℝ, x = a * x^3 + 1 ∧ 3 * a * x^2 = 1) → a = 4/27 := by
  sorry

end NUMINAMATH_CALUDE_tangent_cubic_to_line_l3388_338821


namespace NUMINAMATH_CALUDE_grid_rectangle_division_l3388_338812

/-- A grid rectangle with cell side length 1 cm and area 2021 cm² -/
structure GridRectangle where
  width : ℕ
  height : ℕ
  area_eq : width * height = 2021

/-- A cut configuration for the grid rectangle -/
structure CutConfig where
  hor_cut : ℕ
  ver_cut : ℕ

/-- The four parts resulting from a cut configuration -/
def parts (rect : GridRectangle) (cut : CutConfig) : Fin 4 → ℕ
| ⟨0, _⟩ => cut.hor_cut * cut.ver_cut
| ⟨1, _⟩ => cut.hor_cut * (rect.width - cut.ver_cut)
| ⟨2, _⟩ => (rect.height - cut.hor_cut) * cut.ver_cut
| ⟨3, _⟩ => (rect.height - cut.hor_cut) * (rect.width - cut.ver_cut)
| _ => 0

/-- The theorem to be proved -/
theorem grid_rectangle_division (rect : GridRectangle) :
  ∀ (cut : CutConfig), cut.hor_cut < rect.height → cut.ver_cut < rect.width →
  ∃ (i : Fin 4), parts rect cut i ≥ 528 := by
  sorry

end NUMINAMATH_CALUDE_grid_rectangle_division_l3388_338812


namespace NUMINAMATH_CALUDE_ascending_order_abc_l3388_338808

theorem ascending_order_abc :
  let a := Real.sin (17 * π / 180) * Real.cos (45 * π / 180) + Real.cos (17 * π / 180) * Real.sin (45 * π / 180)
  let b := 2 * (Real.cos (13 * π / 180))^2 - 1
  let c := Real.sqrt 3 / 2
  c < a ∧ a < b := by
  sorry

end NUMINAMATH_CALUDE_ascending_order_abc_l3388_338808


namespace NUMINAMATH_CALUDE_problem_solution_l3388_338843

theorem problem_solution (p q : ℝ) 
  (h1 : 1 < p) 
  (h2 : p < q) 
  (h3 : 1 / p + 1 / q = 1) 
  (h4 : p * q = 8) : 
  q = 4 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3388_338843


namespace NUMINAMATH_CALUDE_forest_foxes_l3388_338896

theorem forest_foxes (total : ℕ) (deer_fraction : ℚ) (fox_fraction : ℚ) : 
  total = 160 →
  deer_fraction = 7 / 8 →
  fox_fraction = 1 - deer_fraction →
  (fox_fraction * total : ℚ) = 20 := by
  sorry

end NUMINAMATH_CALUDE_forest_foxes_l3388_338896


namespace NUMINAMATH_CALUDE_find_number_l3388_338853

theorem find_number : ∃ x : ℝ, 0.45 * x - 85 = 10 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l3388_338853


namespace NUMINAMATH_CALUDE_initial_men_correct_l3388_338837

/-- The number of days it takes to dig the entire tunnel with the initial workforce -/
def initial_days : ℝ := 30

/-- The number of days worked before adding more men -/
def days_before_addition : ℝ := 10

/-- The number of additional men added to the workforce -/
def additional_men : ℕ := 20

/-- The number of days it takes to complete the tunnel after adding more men -/
def remaining_days : ℝ := 10.000000000000002

/-- The initial number of men digging the tunnel -/
def initial_men : ℕ := 6

/-- Theorem stating that the initial number of men is correct given the conditions -/
theorem initial_men_correct :
  (initial_men : ℝ) * initial_days =
    (initial_men + additional_men) * remaining_days * (2/3) :=
by sorry

end NUMINAMATH_CALUDE_initial_men_correct_l3388_338837


namespace NUMINAMATH_CALUDE_bookstore_purchasing_plans_l3388_338825

theorem bookstore_purchasing_plans :
  let n : ℕ := 3 -- number of books
  let select_at_least_one (k : ℕ) : ℕ := 
    Finset.card (Finset.powerset (Finset.range k) \ {∅})
  select_at_least_one n = 7 := by
  sorry

end NUMINAMATH_CALUDE_bookstore_purchasing_plans_l3388_338825


namespace NUMINAMATH_CALUDE_smallest_number_l3388_338834

theorem smallest_number (S : Set ℤ) (h : S = {0, -1, 1, 2}) : 
  ∃ m ∈ S, ∀ n ∈ S, m ≤ n ∧ m = -1 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l3388_338834


namespace NUMINAMATH_CALUDE_intersection_point_k_value_l3388_338870

/-- Given two lines -3x + y = k and 2x + y = 10 that intersect at x = -5, prove that k = 35 -/
theorem intersection_point_k_value :
  ∀ (x y k : ℝ),
  (-3 * x + y = k) →
  (2 * x + y = 10) →
  (x = -5) →
  (k = 35) :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_k_value_l3388_338870


namespace NUMINAMATH_CALUDE_assignment_time_ratio_l3388_338869

theorem assignment_time_ratio : 
  let total_time : ℕ := 120
  let first_part : ℕ := 25
  let third_part : ℕ := 45
  let second_part : ℕ := total_time - (first_part + third_part)
  (second_part : ℚ) / first_part = 2 := by
  sorry

end NUMINAMATH_CALUDE_assignment_time_ratio_l3388_338869


namespace NUMINAMATH_CALUDE_compute_expression_l3388_338867

theorem compute_expression : 9 * (2/3)^4 = 16/9 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l3388_338867


namespace NUMINAMATH_CALUDE_jake_brought_six_balloons_l3388_338823

/-- The number of balloons Jake brought to the park -/
def jakes_balloons (allans_initial_balloons allans_bought_balloons : ℕ) : ℕ :=
  allans_initial_balloons + allans_bought_balloons + 1

/-- Theorem stating that Jake brought 6 balloons to the park -/
theorem jake_brought_six_balloons :
  jakes_balloons 2 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_jake_brought_six_balloons_l3388_338823


namespace NUMINAMATH_CALUDE_y_coordinate_product_l3388_338882

/-- The product of y-coordinates for points on x = -2 that are 12 units from (6, 3) -/
theorem y_coordinate_product : ∃ y₁ y₂ : ℝ,
  ((-2 - 6)^2 + (y₁ - 3)^2 = 12^2) ∧
  ((-2 - 6)^2 + (y₂ - 3)^2 = 12^2) ∧
  y₁ ≠ y₂ ∧
  y₁ * y₂ = -71 := by
sorry

end NUMINAMATH_CALUDE_y_coordinate_product_l3388_338882


namespace NUMINAMATH_CALUDE_triangle_ratio_theorem_l3388_338830

-- Define the triangle PQR
def Triangle (P Q R : ℝ) : Prop := 
  0 < P ∧ 0 < Q ∧ 0 < R ∧ P + Q > R ∧ P + R > Q ∧ Q + R > P

-- Define the sides of the triangle
def PQ : ℝ := 8
def PR : ℝ := 7
def QR : ℝ := 5

-- State the theorem
theorem triangle_ratio_theorem (P Q R : ℝ) 
  (h : Triangle P Q R) 
  (h_pq : PQ = 8) 
  (h_pr : PR = 7) 
  (h_qr : QR = 5) : 
  (Real.cos ((P - Q) / 2) / Real.sin (R / 2)) - 
  (Real.sin ((P - Q) / 2) / Real.cos (R / 2)) = 5 / 7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_ratio_theorem_l3388_338830


namespace NUMINAMATH_CALUDE_lagrange_interpolation_identity_l3388_338899

theorem lagrange_interpolation_identity 
  (a b c x : ℝ) 
  (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) : 
  c^2 * ((x-a)*(x-b))/((c-a)*(c-b)) + 
  b^2 * ((x-a)*(x-c))/((b-a)*(b-c)) + 
  a^2 * ((x-b)*(x-c))/((a-b)*(a-c)) = x^2 := by
  sorry

end NUMINAMATH_CALUDE_lagrange_interpolation_identity_l3388_338899


namespace NUMINAMATH_CALUDE_set_equality_l3388_338838

universe u

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def M : Set ℕ := {3, 4, 5}
def P : Set ℕ := {1, 3, 6}

theorem set_equality : (U \ M) ∩ (U \ P) = {2, 7, 8} := by sorry

end NUMINAMATH_CALUDE_set_equality_l3388_338838


namespace NUMINAMATH_CALUDE_likelihood_number_is_probability_l3388_338850

/-- A number representing the likelihood of a random event occurring -/
def likelihood_number : ℝ := sorry

/-- The term for the number representing the likelihood of a random event occurring -/
def probability_term : String := sorry

/-- The theorem stating that the term for the number representing the likelihood of a random event occurring is "probability" -/
theorem likelihood_number_is_probability : probability_term = "probability" := by
  sorry

end NUMINAMATH_CALUDE_likelihood_number_is_probability_l3388_338850


namespace NUMINAMATH_CALUDE_initial_velocity_is_three_l3388_338888

-- Define the displacement function
def displacement (t : ℝ) : ℝ := 3 * t - t^2

-- Define the velocity function as the derivative of displacement
def velocity (t : ℝ) : ℝ := 3 - 2 * t

-- Theorem statement
theorem initial_velocity_is_three :
  velocity 0 = 3 :=
sorry

end NUMINAMATH_CALUDE_initial_velocity_is_three_l3388_338888


namespace NUMINAMATH_CALUDE_xyz_values_l3388_338810

theorem xyz_values (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0)
  (eq1 : x * y = 30)
  (eq2 : x * z = 60)
  (eq3 : x + y + z = 27) :
  x = (27 + Real.sqrt 369) / 2 ∧
  y = 60 / ((27 + Real.sqrt 369) / 2) ∧
  z = 30 / ((27 + Real.sqrt 369) / 2) := by
sorry

end NUMINAMATH_CALUDE_xyz_values_l3388_338810


namespace NUMINAMATH_CALUDE_ski_trip_sponsorship_l3388_338893

/-- The ski trip sponsorship problem -/
theorem ski_trip_sponsorship 
  (total : ℝ) 
  (first_father : ℝ) 
  (second_father third_father fourth_father : ℝ) 
  (h1 : first_father = 11500)
  (h2 : second_father = (1/3) * (total - second_father))
  (h3 : third_father = (1/4) * (total - third_father))
  (h4 : fourth_father = (1/5) * (total - fourth_father))
  (h5 : total = first_father + second_father + third_father + fourth_father) :
  second_father = 7500 ∧ third_father = 6000 ∧ fourth_father = 5000 := by
  sorry

#eval Float.toString 7500
#eval Float.toString 6000
#eval Float.toString 5000

end NUMINAMATH_CALUDE_ski_trip_sponsorship_l3388_338893


namespace NUMINAMATH_CALUDE_alpha_value_l3388_338842

theorem alpha_value (m n : ℝ) (hm : m > 0) (hn : n > 0) (h_sum : m + n = 1)
  (h_min : ∀ x y : ℝ, x > 0 → y > 0 → x + y = 1 → 1/x + 9/y ≥ 1/m + 9/n)
  (α : ℝ) (h_curve : m^α = 2/3 * n) : α = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_alpha_value_l3388_338842


namespace NUMINAMATH_CALUDE_mom_bought_39_shirts_l3388_338811

/-- The number of t-shirts in each package -/
def shirts_per_package : ℕ := 13

/-- The number of packages mom bought -/
def packages_bought : ℕ := 3

/-- The total number of t-shirts mom bought -/
def total_shirts : ℕ := shirts_per_package * packages_bought

theorem mom_bought_39_shirts : total_shirts = 39 := by
  sorry

end NUMINAMATH_CALUDE_mom_bought_39_shirts_l3388_338811


namespace NUMINAMATH_CALUDE_jake_monday_sales_l3388_338844

/-- The number of candy pieces Jake sold on Monday -/
def monday_sales : ℕ := 15

/-- The initial number of candy pieces Jake had -/
def initial_candy : ℕ := 80

/-- The number of candy pieces Jake sold on Tuesday -/
def tuesday_sales : ℕ := 58

/-- The number of candy pieces Jake had left by Wednesday -/
def wednesday_leftover : ℕ := 7

/-- Theorem stating that the number of candy pieces Jake sold on Monday is 15 -/
theorem jake_monday_sales : 
  monday_sales = initial_candy - tuesday_sales - wednesday_leftover := by
  sorry

end NUMINAMATH_CALUDE_jake_monday_sales_l3388_338844


namespace NUMINAMATH_CALUDE_cats_sold_proof_l3388_338818

/-- Calculates the number of cats sold during a sale at a pet store. -/
def cats_sold (siamese : ℕ) (house : ℕ) (left : ℕ) : ℕ :=
  siamese + house - left

/-- Proves that the number of cats sold during the sale is 45. -/
theorem cats_sold_proof :
  cats_sold 38 25 18 = 45 := by
  sorry

end NUMINAMATH_CALUDE_cats_sold_proof_l3388_338818


namespace NUMINAMATH_CALUDE_scientific_notation_correct_l3388_338835

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- The number to be expressed in scientific notation -/
def original_number : ℕ := 682000000

/-- The proposed scientific notation representation -/
def proposed_notation : ScientificNotation :=
  { coefficient := 6.82
    exponent := 8
    is_valid := by sorry }

/-- Theorem stating that the proposed scientific notation correctly represents the original number -/
theorem scientific_notation_correct :
  (proposed_notation.coefficient * (10 : ℝ) ^ proposed_notation.exponent) = original_number := by sorry

end NUMINAMATH_CALUDE_scientific_notation_correct_l3388_338835


namespace NUMINAMATH_CALUDE_frank_can_buy_seven_candies_l3388_338846

-- Define the given conditions
def whack_a_mole_tickets : ℕ := 33
def skee_ball_tickets : ℕ := 9
def candy_cost : ℕ := 6

-- Define the total number of tickets
def total_tickets : ℕ := whack_a_mole_tickets + skee_ball_tickets

-- Define the number of candies Frank can buy
def candies_bought : ℕ := total_tickets / candy_cost

-- Theorem statement
theorem frank_can_buy_seven_candies : candies_bought = 7 := by
  sorry

end NUMINAMATH_CALUDE_frank_can_buy_seven_candies_l3388_338846


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l3388_338863

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 5*x - 6 ≤ 0}
def B : Set ℝ := {x | x > 5/2}

-- State the theorem
theorem complement_A_intersect_B :
  (Aᶜ ∩ B) = {x | x > 6} :=
sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l3388_338863


namespace NUMINAMATH_CALUDE_candy_distribution_l3388_338827

theorem candy_distribution (adam james rubert : ℕ) 
  (h1 : rubert = 4 * james) 
  (h2 : james = 3 * adam) 
  (h3 : adam + james + rubert = 96) : 
  adam = 6 := by
sorry

end NUMINAMATH_CALUDE_candy_distribution_l3388_338827


namespace NUMINAMATH_CALUDE_rabbit_apple_collection_l3388_338872

theorem rabbit_apple_collection (rabbit_apples_per_basket deer_apples_per_basket : ℕ)
  (rabbit_baskets deer_baskets total_apples : ℕ) :
  rabbit_apples_per_basket = 5 →
  deer_apples_per_basket = 6 →
  rabbit_baskets = deer_baskets + 3 →
  rabbit_apples_per_basket * rabbit_baskets = total_apples →
  deer_apples_per_basket * deer_baskets = total_apples →
  rabbit_apples_per_basket * rabbit_baskets = 90 :=
by sorry

end NUMINAMATH_CALUDE_rabbit_apple_collection_l3388_338872


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l3388_338877

theorem arithmetic_sequence_ratio (a : ℕ → ℝ) (d : ℝ) (h1 : d ≠ 0) 
  (h2 : ∀ n : ℕ, a (n + 1) = a n + d) (h3 : a 3 = 2 * a 1) :
  (a 1 + a 3) / (a 2 + a 4) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l3388_338877


namespace NUMINAMATH_CALUDE_delta_value_l3388_338805

theorem delta_value (Δ : ℤ) (h : 5 * (-3) = Δ - 3) : Δ = -12 := by
  sorry

end NUMINAMATH_CALUDE_delta_value_l3388_338805


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3388_338881

theorem complex_fraction_simplification :
  ∀ (i : ℂ), i^2 = -1 → (11 + 3*i) / (1 - 2*i) = 1 + 5*i := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3388_338881


namespace NUMINAMATH_CALUDE_unique_positive_solution_l3388_338820

theorem unique_positive_solution (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x + y^2 + z^3 = 3)
  (eq2 : y + z^2 + x^3 = 3)
  (eq3 : z + x^2 + y^3 = 3) :
  x = 1 ∧ y = 1 ∧ z = 1 := by
sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l3388_338820


namespace NUMINAMATH_CALUDE_solution_set_nonempty_l3388_338861

theorem solution_set_nonempty (a : ℝ) : 
  ∃ x : ℝ, a * x^2 - (a - 2) * x - 2 ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_nonempty_l3388_338861


namespace NUMINAMATH_CALUDE_endpoint_from_midpoint_and_one_endpoint_l3388_338858

/-- Given a line segment with midpoint (3, 4) and one endpoint at (0, -1), 
    the other endpoint is at (6, 9). -/
theorem endpoint_from_midpoint_and_one_endpoint :
  let midpoint : ℝ × ℝ := (3, 4)
  let endpoint1 : ℝ × ℝ := (0, -1)
  let endpoint2 : ℝ × ℝ := (6, 9)
  (midpoint.1 = (endpoint1.1 + endpoint2.1) / 2 ∧
   midpoint.2 = (endpoint1.2 + endpoint2.2) / 2) :=
by sorry

end NUMINAMATH_CALUDE_endpoint_from_midpoint_and_one_endpoint_l3388_338858


namespace NUMINAMATH_CALUDE_journey_speed_calculation_l3388_338841

theorem journey_speed_calculation (total_distance : ℝ) (total_time : ℝ) 
  (first_part_time : ℝ) (first_part_speed : ℝ) :
  total_distance = 24 →
  total_time = 8 →
  first_part_time = 4 →
  first_part_speed = 4 →
  (total_distance - first_part_time * first_part_speed) / (total_time - first_part_time) = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_journey_speed_calculation_l3388_338841


namespace NUMINAMATH_CALUDE_equal_distance_at_time_l3388_338845

/-- The time in minutes past 3 o'clock when the minute hand is at the same distance 
    to the left of 12 as the hour hand is to the right of 12 -/
def time_equal_distance : ℚ := 13 + 11/13

theorem equal_distance_at_time (t : ℚ) : 
  t = time_equal_distance →
  (180 - 6 * t = 90 + 0.5 * t) := by sorry


end NUMINAMATH_CALUDE_equal_distance_at_time_l3388_338845


namespace NUMINAMATH_CALUDE_complex_fraction_sum_l3388_338898

theorem complex_fraction_sum (z : ℂ) (a b : ℝ) : 
  z = (2 + I) / (1 - 2*I) → 
  z = Complex.mk a b → 
  a + b = 1 :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_sum_l3388_338898


namespace NUMINAMATH_CALUDE_square_congruent_neg_one_mod_prime_l3388_338873

theorem square_congruent_neg_one_mod_prime (p : ℕ) (hp : Nat.Prime p) :
  (∃ k : ℤ, k^2 ≡ -1 [ZMOD p]) ↔ p = 2 ∨ p ≡ 1 [ZMOD 4] :=
sorry

end NUMINAMATH_CALUDE_square_congruent_neg_one_mod_prime_l3388_338873


namespace NUMINAMATH_CALUDE_factorial_ratio_eq_120_l3388_338868

-- Define the factorial function
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- State the theorem
theorem factorial_ratio_eq_120 :
  factorial 10 / (factorial 7 * factorial 3) = 120 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_eq_120_l3388_338868


namespace NUMINAMATH_CALUDE_tape_length_theorem_l3388_338894

/-- Given 15 sheets of tape, each 25 cm long, overlapping by 0.5 cm,
    the total length of the attached tape is 3.68 meters. -/
theorem tape_length_theorem (num_sheets : ℕ) (sheet_length : ℝ) (overlap : ℝ) :
  num_sheets = 15 →
  sheet_length = 25 →
  overlap = 0.5 →
  (num_sheets * sheet_length - (num_sheets - 1) * overlap) / 100 = 3.68 := by
  sorry

end NUMINAMATH_CALUDE_tape_length_theorem_l3388_338894


namespace NUMINAMATH_CALUDE_man_swimming_speed_l3388_338862

/-- The speed of a man in still water, given his downstream and upstream swimming distances and times. -/
theorem man_swimming_speed 
  (downstream_distance : ℝ) 
  (upstream_distance : ℝ) 
  (time : ℝ) 
  (h1 : downstream_distance = 51) 
  (h2 : upstream_distance = 18) 
  (h3 : time = 3) :
  ∃ (man_speed stream_speed : ℝ), 
    downstream_distance = (man_speed + stream_speed) * time ∧ 
    upstream_distance = (man_speed - stream_speed) * time ∧ 
    man_speed = 11.5 := by
sorry

end NUMINAMATH_CALUDE_man_swimming_speed_l3388_338862


namespace NUMINAMATH_CALUDE_binary_11010_equals_octal_32_l3388_338890

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Converts a decimal number to its octal representation -/
def decimal_to_octal (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) :=
      if m = 0 then acc else aux (m / 8) ((m % 8) :: acc)
    aux n []

/-- The binary representation of the number 11010 -/
def binary_11010 : List Bool := [false, true, false, true, true]

/-- The octal representation of the number 32 -/
def octal_32 : List ℕ := [3, 2]

theorem binary_11010_equals_octal_32 :
  decimal_to_octal (binary_to_decimal binary_11010) = octal_32 := by
  sorry

end NUMINAMATH_CALUDE_binary_11010_equals_octal_32_l3388_338890


namespace NUMINAMATH_CALUDE_complex_square_one_minus_i_l3388_338815

-- Define the complex number i
def i : ℂ := Complex.I

-- Theorem statement
theorem complex_square_one_minus_i : (1 - i)^2 = -2*i := by sorry

end NUMINAMATH_CALUDE_complex_square_one_minus_i_l3388_338815


namespace NUMINAMATH_CALUDE_wage_cut_and_raise_l3388_338852

theorem wage_cut_and_raise (original_wage : ℝ) (h : original_wage > 0) :
  let reduced_wage := 0.75 * original_wage
  let raise_percentage := 1 / 3
  reduced_wage * (1 + raise_percentage) = original_wage := by sorry

end NUMINAMATH_CALUDE_wage_cut_and_raise_l3388_338852


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_boat_speed_in_still_water_proof_l3388_338828

/-- Given a boat that travels 15 km/hr along a stream and 5 km/hr against the same stream,
    its speed in still water is 10 km/hr. -/
theorem boat_speed_in_still_water : ℝ → ℝ → Prop :=
  fun (along_stream : ℝ) (against_stream : ℝ) =>
    along_stream = 15 ∧ against_stream = 5 →
    ∃ (boat_speed stream_speed : ℝ),
      boat_speed + stream_speed = along_stream ∧
      boat_speed - stream_speed = against_stream ∧
      boat_speed = 10

/-- Proof of the theorem -/
theorem boat_speed_in_still_water_proof :
  boat_speed_in_still_water 15 5 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_boat_speed_in_still_water_proof_l3388_338828


namespace NUMINAMATH_CALUDE_equation_solution_l3388_338804

theorem equation_solution : 
  ∀ y : ℝ, (12 - y)^2 = 4 * y^2 ↔ y = 4 ∨ y = -12 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3388_338804


namespace NUMINAMATH_CALUDE_computer_multiplications_l3388_338897

/-- Represents the number of multiplications a computer can perform per minute -/
def multiplications_per_minute : ℕ := 25000

/-- Represents the number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- Represents the number of hours we're calculating for -/
def hours : ℕ := 3

/-- Theorem stating that the computer will perform 4,500,000 multiplications in three hours -/
theorem computer_multiplications :
  multiplications_per_minute * minutes_per_hour * hours = 4500000 :=
by sorry

end NUMINAMATH_CALUDE_computer_multiplications_l3388_338897


namespace NUMINAMATH_CALUDE_light_glow_theorem_l3388_338892

def seconds_since_midnight (hours minutes seconds : ℕ) : ℕ :=
  hours * 3600 + minutes * 60 + seconds

def light_glow_count (start_time end_time glow_interval : ℕ) : ℕ :=
  (end_time - start_time) / glow_interval

theorem light_glow_theorem (start_a start_b start_c end_time : ℕ) 
  (interval_a interval_b interval_c : ℕ) : 
  let count_a := light_glow_count start_a end_time interval_a
  let count_b := light_glow_count start_b end_time interval_b
  let count_c := light_glow_count start_c end_time interval_c
  ∃ (x y z : ℕ), x = count_a ∧ y = count_b ∧ z = count_c := by
  sorry

#eval light_glow_count (seconds_since_midnight 1 57 58) (seconds_since_midnight 3 20 47) 14
#eval light_glow_count (seconds_since_midnight 2 0 25) (seconds_since_midnight 3 20 47) 21
#eval light_glow_count (seconds_since_midnight 2 10 15) (seconds_since_midnight 3 20 47) 10

end NUMINAMATH_CALUDE_light_glow_theorem_l3388_338892


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l3388_338829

/-- Given two hyperbolas with the same asymptotes, prove the value of T -/
theorem hyperbola_asymptotes (T : ℚ) : 
  (∀ x y, y^2 / 49 - x^2 / 25 = 1 → 
    ∃ k, y = k * x ∧ k^2 = 49 / 25) ∧
  (∀ x y, x^2 / T - y^2 / 18 = 1 → 
    ∃ k, y = k * x ∧ k^2 = 18 / T) →
  T = 450 / 49 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l3388_338829


namespace NUMINAMATH_CALUDE_linear_dependence_implies_k₁_plus_4k₃_eq_zero_l3388_338886

def a₁ : Fin 2 → ℝ := ![1, 0]
def a₂ : Fin 2 → ℝ := ![1, -1]
def a₃ : Fin 2 → ℝ := ![2, 2]

theorem linear_dependence_implies_k₁_plus_4k₃_eq_zero :
  ∃ (k₁ k₂ k₃ : ℝ), (k₁ ≠ 0 ∨ k₂ ≠ 0 ∨ k₃ ≠ 0) ∧
    (∀ i : Fin 2, k₁ * a₁ i + k₂ * a₂ i + k₃ * a₃ i = 0) →
  ∀ (k₁ k₂ k₃ : ℝ), (∀ i : Fin 2, k₁ * a₁ i + k₂ * a₂ i + k₃ * a₃ i = 0) →
  k₁ + 4 * k₃ = 0 :=
by sorry

end NUMINAMATH_CALUDE_linear_dependence_implies_k₁_plus_4k₃_eq_zero_l3388_338886


namespace NUMINAMATH_CALUDE_sin_80_cos_20_minus_cos_80_sin_20_l3388_338826

theorem sin_80_cos_20_minus_cos_80_sin_20 : 
  Real.sin (80 * π / 180) * Real.cos (20 * π / 180) - 
  Real.cos (80 * π / 180) * Real.sin (20 * π / 180) = 
  Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_sin_80_cos_20_minus_cos_80_sin_20_l3388_338826


namespace NUMINAMATH_CALUDE_system_solution_l3388_338833

theorem system_solution : 
  ∃ (x y : ℚ), (4 * x - 3 * y = -2) ∧ (9 * x + 5 * y = 9) ∧ (x = 17/47) ∧ (y = 54/47) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3388_338833


namespace NUMINAMATH_CALUDE_special_function_property_l3388_338878

/-- A function satisfying g(xy) = g(x)/y for all positive real numbers x and y -/
def SpecialFunction (g : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ), x > 0 → y > 0 → g (x * y) = g x / y

theorem special_function_property (g : ℝ → ℝ) 
    (h1 : SpecialFunction g) 
    (h2 : g 30 = 30) : 
    g 45 = 20 := by
  sorry

end NUMINAMATH_CALUDE_special_function_property_l3388_338878


namespace NUMINAMATH_CALUDE_floor_ceil_sqrt_50_sum_squares_l3388_338814

theorem floor_ceil_sqrt_50_sum_squares : ⌊Real.sqrt 50⌋^2 + ⌈Real.sqrt 50⌉^2 = 113 := by
  sorry

end NUMINAMATH_CALUDE_floor_ceil_sqrt_50_sum_squares_l3388_338814


namespace NUMINAMATH_CALUDE_P_roots_l3388_338836

def P : ℕ → ℝ → ℝ
  | 0, x => 1
  | n + 1, x => x^(5 * (n + 1)) - P n x

theorem P_roots (n : ℕ) :
  (n % 2 = 1 → P n 1 = 0 ∧ ∀ x : ℝ, x ≠ 1 → P n x ≠ 0) ∧
  (n % 2 = 0 → ∀ x : ℝ, P n x ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_P_roots_l3388_338836


namespace NUMINAMATH_CALUDE_total_production_proof_l3388_338822

def week1_production : ℕ := 320
def week2_production : ℕ := 400
def week3_production : ℕ := 300
def increase_percentage : ℚ := 20 / 100

theorem total_production_proof :
  let average := (week1_production + week2_production + week3_production) / 3
  let week4_production := average + (average * increase_percentage).floor
  week1_production + week2_production + week3_production + week4_production = 1428 := by
  sorry

end NUMINAMATH_CALUDE_total_production_proof_l3388_338822


namespace NUMINAMATH_CALUDE_percentage_non_swimmers_basketball_l3388_338880

/-- Represents the percentage of students who play basketball -/
def basketball_players : ℝ := 0.7

/-- Represents the percentage of students who swim -/
def swimmers : ℝ := 0.5

/-- Represents the percentage of basketball players who also swim -/
def basketball_and_swim : ℝ := 0.3

/-- Theorem: The percentage of non-swimmers who play basketball is 98% -/
theorem percentage_non_swimmers_basketball : 
  (basketball_players - basketball_players * basketball_and_swim) / (1 - swimmers) = 0.98 := by
  sorry

end NUMINAMATH_CALUDE_percentage_non_swimmers_basketball_l3388_338880


namespace NUMINAMATH_CALUDE_min_obtuse_angles_convex_octagon_min_obtuse_angles_convex_octagon_proof_l3388_338855

/-- The minimum number of obtuse interior angles in a convex octagon -/
theorem min_obtuse_angles_convex_octagon : ℕ :=
  let exterior_angles : ℕ := 8
  let sum_exterior_angles : ℕ := 360
  5

/-- Proof of the minimum number of obtuse interior angles in a convex octagon -/
theorem min_obtuse_angles_convex_octagon_proof :
  min_obtuse_angles_convex_octagon = 5 := by
  sorry

end NUMINAMATH_CALUDE_min_obtuse_angles_convex_octagon_min_obtuse_angles_convex_octagon_proof_l3388_338855


namespace NUMINAMATH_CALUDE_min_radius_circle_equation_l3388_338884

/-- The line on which points A and B move --/
def line (x y : ℝ) : Prop := 3 * x + y - 10 = 0

/-- The circle M with diameter AB --/
def circle_M (a b : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - (a.1 + b.1) / 2)^2 + (p.2 - (a.2 + b.2) / 2)^2 = ((a.1 - b.1)^2 + (a.2 - b.2)^2) / 4}

/-- The origin point --/
def origin : ℝ × ℝ := (0, 0)

/-- Theorem stating the standard equation of circle M when its radius is minimum --/
theorem min_radius_circle_equation :
  ∀ a b : ℝ × ℝ,
  line a.1 a.2 → line b.1 b.2 →
  origin ∈ circle_M a b →
  (∀ c d : ℝ × ℝ, line c.1 c.2 → line d.1 d.2 → origin ∈ circle_M c d →
    (a.1 - b.1)^2 + (a.2 - b.2)^2 ≤ (c.1 - d.1)^2 + (c.2 - d.2)^2) →
  circle_M a b = {p : ℝ × ℝ | (p.1 - 3)^2 + (p.2 - 1)^2 = 10} :=
sorry

end NUMINAMATH_CALUDE_min_radius_circle_equation_l3388_338884


namespace NUMINAMATH_CALUDE_anthony_painting_time_l3388_338802

/-- The time it takes Kathleen and Anthony to paint two rooms together -/
def joint_time : ℝ := 3.428571428571429

/-- The time it takes Kathleen to paint one room -/
def kathleen_time : ℝ := 3

/-- Anthony's painting time for one room -/
def anthony_time : ℝ := 4

/-- Theorem stating that given Kathleen's painting time and their joint time for two rooms, 
    Anthony's individual painting time for one room is 4 hours -/
theorem anthony_painting_time : 
  (1 / kathleen_time + 1 / anthony_time) * joint_time = 2 :=
sorry

end NUMINAMATH_CALUDE_anthony_painting_time_l3388_338802


namespace NUMINAMATH_CALUDE_last_digit_of_7_to_1032_l3388_338864

theorem last_digit_of_7_to_1032 : ∃ n : ℕ, 7^1032 ≡ 1 [ZMOD 10] := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_7_to_1032_l3388_338864


namespace NUMINAMATH_CALUDE_f_composition_eq_exp_l3388_338854

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 1 then 2^x else 3*x - 1

theorem f_composition_eq_exp (a : ℝ) :
  {a : ℝ | f (f a) = 2^(f a)} = Set.Ici (2/3) := by sorry

end NUMINAMATH_CALUDE_f_composition_eq_exp_l3388_338854


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3388_338839

/-- A sequence of real numbers -/
def Sequence := ℕ → ℝ

/-- Predicate for a sequence being arithmetic -/
def is_arithmetic (a : Sequence) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Predicate for a point being on the line y = 2x + 1 -/
def on_line (n : ℕ) (a : Sequence) : Prop :=
  a n = 2 * n + 1

theorem sufficient_not_necessary :
  (∀ a : Sequence, (∀ n : ℕ, n > 0 → on_line n a) → is_arithmetic a) ∧
  (∃ a : Sequence, is_arithmetic a ∧ ∃ n : ℕ, n > 0 ∧ ¬on_line n a) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3388_338839


namespace NUMINAMATH_CALUDE_smallest_divisor_with_remainder_l3388_338847

theorem smallest_divisor_with_remainder (d : ℕ) : d = 6 ↔ 
  d > 1 ∧ 
  (∀ n : ℤ, n % d = 1 → (5 * n) % d = 5) ∧
  (∀ d' : ℕ, d' < d → d' > 1 → ∃ n : ℤ, n % d' = 1 ∧ (5 * n) % d' ≠ 5) :=
by sorry

#check smallest_divisor_with_remainder

end NUMINAMATH_CALUDE_smallest_divisor_with_remainder_l3388_338847

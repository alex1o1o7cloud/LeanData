import Mathlib

namespace NUMINAMATH_CALUDE_equation_roots_l1460_146013

theorem equation_roots : 
  {x : ℝ | Real.sqrt (x^2) + 3 * x⁻¹ = 4} = {3, -3, 1, -1} :=
by sorry

end NUMINAMATH_CALUDE_equation_roots_l1460_146013


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_expression_l1460_146009

theorem simplify_and_evaluate_expression (x : ℝ) (h : x = -4) :
  (1 - (x + 1) / (x^2 - 2*x + 1)) / ((x - 3) / (x - 1)) = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_expression_l1460_146009


namespace NUMINAMATH_CALUDE_fraction_simplification_l1460_146027

theorem fraction_simplification (x y z : ℝ) (h : x + y + z = 3) :
  (x*y + y*z + z*x) / (x^2 + y^2 + z^2) = (x*y + y*z + z*x) / (9 - 2*(x*y + y*z + z*x)) :=
by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1460_146027


namespace NUMINAMATH_CALUDE_solve_system_of_equations_l1460_146039

theorem solve_system_of_equations (a b : ℝ) 
  (eq1 : 3 * a + 2 * b = 18) 
  (eq2 : 5 * a + 4 * b = 31) : 
  2 * a + b = 11.5 := by
sorry

end NUMINAMATH_CALUDE_solve_system_of_equations_l1460_146039


namespace NUMINAMATH_CALUDE_geometric_sequence_third_term_l1460_146091

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_third_term
  (a : ℕ → ℝ)
  (h_geometric : GeometricSequence a)
  (h_first : a 1 = 1024)
  (h_fifth : a 5 = 128) :
  a 3 = 256 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_third_term_l1460_146091


namespace NUMINAMATH_CALUDE_nails_per_plank_l1460_146051

/-- Given that John uses 11 nails in total, with 8 additional nails, and needs 1 plank,
    prove that each plank requires 3 nails to be secured. -/
theorem nails_per_plank (total_nails : ℕ) (additional_nails : ℕ) (num_planks : ℕ)
  (h1 : total_nails = 11)
  (h2 : additional_nails = 8)
  (h3 : num_planks = 1) :
  total_nails - additional_nails = 3 := by
sorry

end NUMINAMATH_CALUDE_nails_per_plank_l1460_146051


namespace NUMINAMATH_CALUDE_die_roll_probability_l1460_146094

def is_valid_roll (x : ℕ) : Prop := 1 ≤ x ∧ x ≤ 6

def angle_in_range (m n : ℕ) : Prop :=
  let a : ℝ × ℝ := (m, n)
  let b : ℝ × ℝ := (1, 0)
  let cos_alpha := (m : ℝ) / Real.sqrt ((m^2 : ℝ) + (n^2 : ℝ))
  Real.sqrt 2 / 2 < cos_alpha ∧ cos_alpha < 1

def count_favorable_outcomes : ℕ := 15

def total_outcomes : ℕ := 36

theorem die_roll_probability :
  (count_favorable_outcomes : ℚ) / total_outcomes = 5 / 12 :=
sorry

end NUMINAMATH_CALUDE_die_roll_probability_l1460_146094


namespace NUMINAMATH_CALUDE_unique_sums_count_l1460_146059

def bag_A : Finset ℕ := {1, 4, 5, 8}
def bag_B : Finset ℕ := {2, 3, 7, 9}

theorem unique_sums_count : 
  Finset.card ((bag_A.product bag_B).image (λ (p : ℕ × ℕ) => p.1 + p.2)) = 12 := by
  sorry

end NUMINAMATH_CALUDE_unique_sums_count_l1460_146059


namespace NUMINAMATH_CALUDE_hexagon_division_l1460_146066

/- Define a hexagon -/
def Hexagon : Type := Unit

/- Define a legal point in the hexagon -/
inductive LegalPoint : Type
| vertex : LegalPoint
| intersection : LegalPoint → LegalPoint → LegalPoint

/- Define a legal triangle in the hexagon -/
structure LegalTriangle :=
(p1 p2 p3 : LegalPoint)

/- Define a division of the hexagon -/
def Division := List LegalTriangle

/- The main theorem to prove -/
theorem hexagon_division (n : Nat) (h : n ≥ 6) : 
  ∃ (d : Division), d.length = n := by sorry

end NUMINAMATH_CALUDE_hexagon_division_l1460_146066


namespace NUMINAMATH_CALUDE_abs_equation_solution_l1460_146002

theorem abs_equation_solution :
  ∃! y : ℝ, |y - 6| + 3*y = 12 :=
by
  -- The unique solution is y = 3
  use 3
  sorry

end NUMINAMATH_CALUDE_abs_equation_solution_l1460_146002


namespace NUMINAMATH_CALUDE_simplify_expression_l1460_146033

theorem simplify_expression (a b c : ℝ) : 
  3*a - (4*a - 6*b - 3*c) - 5*(c - b) = -a + 11*b - 2*c := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1460_146033


namespace NUMINAMATH_CALUDE_race_distance_l1460_146073

theorem race_distance (total_length : Real) (part1 : Real) (part2 : Real) (part3 : Real)
  (h1 : total_length = 74.5)
  (h2 : part1 = 15.5)
  (h3 : part2 = 21.5)
  (h4 : part3 = 21.5) :
  total_length - (part1 + part2 + part3) = 16 := by
  sorry

end NUMINAMATH_CALUDE_race_distance_l1460_146073


namespace NUMINAMATH_CALUDE_triple_overlap_area_is_six_l1460_146028

/-- Represents a rectangular rug with width and height -/
structure Rug where
  width : ℝ
  height : ℝ

/-- Represents the hall and the rugs placed in it -/
structure HallWithRugs where
  hallSize : ℝ
  rug1 : Rug
  rug2 : Rug
  rug3 : Rug

/-- Calculates the area covered by all three rugs in the hall -/
def tripleOverlapArea (hall : HallWithRugs) : ℝ :=
  2 * 3

/-- Theorem stating that the area covered by all three rugs is 6 square meters -/
theorem triple_overlap_area_is_six (hall : HallWithRugs) 
  (h1 : hall.hallSize = 10)
  (h2 : hall.rug1 = ⟨6, 8⟩)
  (h3 : hall.rug2 = ⟨6, 6⟩)
  (h4 : hall.rug3 = ⟨5, 7⟩) :
  tripleOverlapArea hall = 6 := by
  sorry

end NUMINAMATH_CALUDE_triple_overlap_area_is_six_l1460_146028


namespace NUMINAMATH_CALUDE_cubic_expression_equal_sixty_times_ten_power_l1460_146071

theorem cubic_expression_equal_sixty_times_ten_power : 
  (2^1501 + 5^1502)^3 - (2^1501 - 5^1502)^3 = 60 * 10^1501 := by sorry

end NUMINAMATH_CALUDE_cubic_expression_equal_sixty_times_ten_power_l1460_146071


namespace NUMINAMATH_CALUDE_triangle_problem_l1460_146087

/-- Given a triangle ABC with sides a, b, c and angles A, B, C, prove that
    under certain conditions, angle A is π/4 and the area is 9/4. -/
theorem triangle_problem (a b c : ℝ) (A B C : ℝ) : 
  a = 3 →
  b^2 + c^2 - a^2 - Real.sqrt 2 * b * c = 0 →
  Real.sin B^2 + Real.sin C^2 = 2 * Real.sin A^2 →
  A = π / 4 ∧ 
  (1/2) * b * c * Real.sin A = 9/4 :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l1460_146087


namespace NUMINAMATH_CALUDE_geometric_sequence_in_arithmetic_progression_l1460_146085

theorem geometric_sequence_in_arithmetic_progression (x : ℚ) (hx : x > 0) :
  ∃ (i j k : ℕ), i < j ∧ j < k ∧ (x + i) * (x + k) = (x + j)^2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_in_arithmetic_progression_l1460_146085


namespace NUMINAMATH_CALUDE_sara_quarters_l1460_146042

/-- The number of quarters Sara had initially -/
def initial_quarters : ℕ := sorry

/-- The number of quarters Sara's dad gave her -/
def dad_gift : ℕ := 49

/-- The total number of quarters Sara has after her dad's gift -/
def total_quarters : ℕ := 70

theorem sara_quarters : initial_quarters + dad_gift = total_quarters := by sorry

end NUMINAMATH_CALUDE_sara_quarters_l1460_146042


namespace NUMINAMATH_CALUDE_basketball_score_possibilities_count_basketball_scores_l1460_146061

def basketball_scores (n : ℕ) : Finset ℕ :=
  Finset.image (λ k => 3 * k + 2 * (n - k)) (Finset.range (n + 1))

theorem basketball_score_possibilities :
  basketball_scores 5 = {10, 11, 12, 13, 14, 15} :=
sorry

theorem count_basketball_scores :
  (basketball_scores 5).card = 6 :=
sorry

end NUMINAMATH_CALUDE_basketball_score_possibilities_count_basketball_scores_l1460_146061


namespace NUMINAMATH_CALUDE_no_three_digit_multiple_base_l1460_146017

/-- Definition of a valid base for a number x -/
def valid_base (x : ℕ) (b : ℕ) : Prop :=
  b ≥ 2 ∧ b ≤ 10 ∧ (b - 1)^4 < x ∧ x < b^4

/-- Definition of a three-digit number in base b -/
def three_digit (x : ℕ) (b : ℕ) : Prop :=
  b^2 ≤ x ∧ x < b^3

/-- Main theorem: No three-digit number represents multiple values in different bases -/
theorem no_three_digit_multiple_base :
  ¬ ∃ (x : ℕ) (b1 b2 : ℕ), x < 10000 ∧ b1 < b2 ∧
    valid_base x b1 ∧ valid_base x b2 ∧
    three_digit x b1 ∧ three_digit x b2 :=
sorry

end NUMINAMATH_CALUDE_no_three_digit_multiple_base_l1460_146017


namespace NUMINAMATH_CALUDE_equal_roots_implies_a_equals_negative_one_l1460_146035

/-- The quadratic equation with parameter a -/
def quadratic_equation (a : ℝ) (x : ℝ) : ℝ := x * (x + 1) + a * x

/-- The discriminant of the quadratic equation -/
def discriminant (a : ℝ) : ℝ := (1 + a)^2

theorem equal_roots_implies_a_equals_negative_one :
  (∃ x : ℝ, quadratic_equation a x = 0 ∧ 
    ∀ y : ℝ, quadratic_equation a y = 0 → y = x) →
  discriminant a = 0 →
  a = -1 :=
sorry

end NUMINAMATH_CALUDE_equal_roots_implies_a_equals_negative_one_l1460_146035


namespace NUMINAMATH_CALUDE_rationalize_denominator_l1460_146096

theorem rationalize_denominator :
  ∃ (A B C : ℕ) (D : ℕ+),
    (1 : ℝ) / (Real.rpow 5 (1/3) - Real.rpow 4 (1/3)) =
    (Real.rpow A (1/3) + Real.rpow B (1/3) + Real.rpow C (1/3)) / D ∧
    A = 25 ∧ B = 20 ∧ C = 16 ∧ D = 1 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l1460_146096


namespace NUMINAMATH_CALUDE_percentage_problem_l1460_146075

theorem percentage_problem (x : ℝ) : 0.25 * x = 0.15 * 1500 - 30 → x = 780 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1460_146075


namespace NUMINAMATH_CALUDE_janes_blouses_l1460_146007

theorem janes_blouses (skirt_price : ℕ) (blouse_price : ℕ) (num_skirts : ℕ) (total_paid : ℕ) (change : ℕ) : 
  skirt_price = 13 →
  blouse_price = 6 →
  num_skirts = 2 →
  total_paid = 100 →
  change = 56 →
  (total_paid - change - (num_skirts * skirt_price)) / blouse_price = 3 :=
by sorry

end NUMINAMATH_CALUDE_janes_blouses_l1460_146007


namespace NUMINAMATH_CALUDE_quadruplet_babies_l1460_146004

theorem quadruplet_babies (total_babies : ℕ) 
  (h_total : total_babies = 1500)
  (h_triplets : ∃ c : ℕ, 5 * c = number_of_triplet_sets)
  (h_twins : number_of_twin_sets = 2 * number_of_triplet_sets)
  (h_quintuplets : number_of_quintuplet_sets = number_of_quadruplet_sets / 2)
  (h_sum : 2 * number_of_twin_sets + 3 * number_of_triplet_sets + 
           4 * number_of_quadruplet_sets + 5 * number_of_quintuplet_sets = total_babies) :
  4 * number_of_quadruplet_sets = 145 :=
by sorry

-- Define variables
variable (number_of_twin_sets number_of_triplet_sets number_of_quadruplet_sets number_of_quintuplet_sets : ℕ)

end NUMINAMATH_CALUDE_quadruplet_babies_l1460_146004


namespace NUMINAMATH_CALUDE_pauls_coupon_percentage_l1460_146000

def initial_cost : ℝ := 350
def store_discount_percent : ℝ := 20
def final_price : ℝ := 252

theorem pauls_coupon_percentage :
  ∃ (coupon_percent : ℝ),
    final_price = initial_cost * (1 - store_discount_percent / 100) * (1 - coupon_percent / 100) ∧
    coupon_percent = 10 := by
  sorry

end NUMINAMATH_CALUDE_pauls_coupon_percentage_l1460_146000


namespace NUMINAMATH_CALUDE_ceiling_minus_fractional_part_l1460_146090

theorem ceiling_minus_fractional_part (x : ℝ) : ⌈x⌉ - (x - ⌊x⌋) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_minus_fractional_part_l1460_146090


namespace NUMINAMATH_CALUDE_complex_fraction_equals_i_l1460_146015

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem complex_fraction_equals_i : (1 + i^2017) / (1 - i) = i := by sorry

end NUMINAMATH_CALUDE_complex_fraction_equals_i_l1460_146015


namespace NUMINAMATH_CALUDE_polynomial_root_sum_l1460_146024

theorem polynomial_root_sum (a b : ℝ) : 
  (Complex.I * Real.sqrt 3 + 1 : ℂ) ^ 3 + a * (Complex.I * Real.sqrt 3 + 1) + b = 0 ∧ 
  (-3 : ℂ) ^ 3 + a * (-3) + b = 0 → 
  a + b = 11 := by sorry

end NUMINAMATH_CALUDE_polynomial_root_sum_l1460_146024


namespace NUMINAMATH_CALUDE_never_equal_amounts_l1460_146060

/-- Represents the currencies in Dillie and Dallie -/
inductive Currency
| Diller
| Daller

/-- Represents the state of the financier's money -/
structure MoneyState :=
  (dillers : ℕ)
  (dallers : ℕ)

/-- Represents a currency exchange -/
inductive Exchange
| ToDallers
| ToDillers

/-- The exchange rate from dillers to dallers -/
def dillerToDallerRate : ℕ := 10

/-- The exchange rate from dallers to dillers -/
def dallerToDillerRate : ℕ := 10

/-- Perform a single exchange -/
def performExchange (state : MoneyState) (exchange : Exchange) : MoneyState :=
  match exchange with
  | Exchange.ToDallers => 
      { dillers := state.dillers / dillerToDallerRate,
        dallers := state.dallers + state.dillers * dillerToDallerRate }
  | Exchange.ToDillers => 
      { dillers := state.dillers + state.dallers * dallerToDillerRate,
        dallers := state.dallers / dallerToDillerRate }

/-- The initial state of the financier's money -/
def initialState : MoneyState := { dillers := 1, dallers := 0 }

/-- The main theorem to prove -/
theorem never_equal_amounts (exchanges : List Exchange) :
  let finalState := exchanges.foldl performExchange initialState
  finalState.dillers ≠ finalState.dallers :=
sorry

end NUMINAMATH_CALUDE_never_equal_amounts_l1460_146060


namespace NUMINAMATH_CALUDE_projection_a_onto_b_l1460_146038

def a : ℝ × ℝ := (2, 3)
def b : ℝ × ℝ := (-4, 7)

theorem projection_a_onto_b :
  let proj := (a.1 * b.1 + a.2 * b.2) / Real.sqrt (b.1^2 + b.2^2)
  proj = Real.sqrt 65 / 5 := by sorry

end NUMINAMATH_CALUDE_projection_a_onto_b_l1460_146038


namespace NUMINAMATH_CALUDE_jim_buicks_count_l1460_146019

/-- The number of model cars Jim has for each brand -/
structure ModelCars where
  ford : ℕ
  buick : ℕ
  chevy : ℕ

/-- Jim's collection of model cars satisfying the given conditions -/
def jim_collection : ModelCars → Prop
  | ⟨f, b, c⟩ => f + b + c = 301 ∧ b = 4 * f ∧ f = 2 * c + 3

theorem jim_buicks_count (cars : ModelCars) (h : jim_collection cars) : cars.buick = 220 := by
  sorry

end NUMINAMATH_CALUDE_jim_buicks_count_l1460_146019


namespace NUMINAMATH_CALUDE_integer_root_b_values_l1460_146072

def has_integer_root (b : ℤ) : Prop :=
  ∃ x : ℤ, x^3 + 4*x^2 + b*x + 12 = 0

def valid_b_values : Set ℤ :=
  {-193, -97, -62, -35, -25, -18, -17, -14, -3, -1, 2, 9}

theorem integer_root_b_values :
  ∀ b : ℤ, has_integer_root b ↔ b ∈ valid_b_values :=
sorry

end NUMINAMATH_CALUDE_integer_root_b_values_l1460_146072


namespace NUMINAMATH_CALUDE_smallest_invertible_domain_l1460_146074

def g (x : ℝ) : ℝ := (2*x - 3)^2 - 4

theorem smallest_invertible_domain (c : ℝ) : 
  (∀ x y, x ≥ c → y ≥ c → g x = g y → x = y) ∧ 
  (∀ c' : ℝ, c' < c → ∃ x y, x ≥ c' ∧ y ≥ c' ∧ x ≠ y ∧ g x = g y) → 
  c = 3/2 :=
sorry

end NUMINAMATH_CALUDE_smallest_invertible_domain_l1460_146074


namespace NUMINAMATH_CALUDE_area1_is_linear_area2_is_quadratic_l1460_146014

-- Define the rectangles
def rectangle1 (x : ℝ) : ℝ × ℝ := (10 - x, 5)
def rectangle2 (x : ℝ) : ℝ × ℝ := (30 + x, 20 + x)

-- Define the area functions
def area1 (x : ℝ) : ℝ := (rectangle1 x).1 * (rectangle1 x).2
def area2 (x : ℝ) : ℝ := (rectangle2 x).1 * (rectangle2 x).2

-- Theorem statements
theorem area1_is_linear : ∃ (m b : ℝ), ∀ x, area1 x = m * x + b := by sorry

theorem area2_is_quadratic : ∃ (a b c : ℝ), a ≠ 0 ∧ (∀ x, area2 x = a * x^2 + b * x + c) := by sorry

end NUMINAMATH_CALUDE_area1_is_linear_area2_is_quadratic_l1460_146014


namespace NUMINAMATH_CALUDE_shopkeeper_cloth_sale_l1460_146021

/-- Calculates the total selling amount for cloth given the length, cost price, and loss per metre -/
def total_selling_amount (cloth_length : ℕ) (cost_price_per_metre : ℕ) (loss_per_metre : ℕ) : ℕ :=
  let selling_price_per_metre := cost_price_per_metre - loss_per_metre
  cloth_length * selling_price_per_metre

/-- Proves that the total selling amount for 200 metres of cloth with a cost price of 95 Rs per metre 
    and a loss of 5 Rs per metre is 18000 Rs -/
theorem shopkeeper_cloth_sale : 
  total_selling_amount 200 95 5 = 18000 := by
  sorry

end NUMINAMATH_CALUDE_shopkeeper_cloth_sale_l1460_146021


namespace NUMINAMATH_CALUDE_first_coordinate_on_line_l1460_146093

theorem first_coordinate_on_line (n : ℝ) (a : ℝ) :
  (a = 4 * n + 5 ∧ a + 2 = 4 * (n + 0.5) + 5) → a = 4 * n + 5 :=
by sorry

end NUMINAMATH_CALUDE_first_coordinate_on_line_l1460_146093


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l1460_146037

/-- A positive geometric sequence -/
def IsPositiveGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), r > 0 ∧ ∀ n, a (n + 1) = r * a n ∧ a n > 0

theorem geometric_sequence_problem (a : ℕ → ℝ)
    (h_geom : IsPositiveGeometricSequence a)
    (h_sum : a 1 + 2/3 * a 2 = 3)
    (h_prod : a 4 ^ 2 = 1/9 * a 3 * a 7) :
    a 4 = 27 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l1460_146037


namespace NUMINAMATH_CALUDE_cube_roots_unity_sum_l1460_146083

theorem cube_roots_unity_sum (x y : ℂ) : 
  x = (-1 + Complex.I * Real.sqrt 3) / 2 →
  y = (-1 - Complex.I * Real.sqrt 3) / 2 →
  x^9 + y^9 ≠ -1 := by
  sorry

end NUMINAMATH_CALUDE_cube_roots_unity_sum_l1460_146083


namespace NUMINAMATH_CALUDE_shopkeeper_decks_l1460_146080

/-- Represents the number of face cards in a standard deck of playing cards. -/
def face_cards_per_deck : ℕ := 12

/-- Represents the total number of face cards the shopkeeper has. -/
def total_face_cards : ℕ := 60

/-- Calculates the number of complete decks given the total number of face cards. -/
def number_of_decks : ℕ := total_face_cards / face_cards_per_deck

theorem shopkeeper_decks : number_of_decks = 5 := by
  sorry

end NUMINAMATH_CALUDE_shopkeeper_decks_l1460_146080


namespace NUMINAMATH_CALUDE_Φ_is_connected_Φ_single_part_l1460_146099

/-- The set of points (x, y) in R^2 satisfying the given system of inequalities -/
def Φ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
               Real.sqrt (x^2 - 3*y^2 + 4*x + 4) ≤ 2*x + 1 ∧
               x^2 + y^2 ≤ 4}

/-- Theorem stating that Φ is a connected set -/
theorem Φ_is_connected : IsConnected Φ := by
  sorry

/-- Corollary stating that Φ consists of a single part -/
theorem Φ_single_part : ∃! (S : Set (ℝ × ℝ)), S = Φ ∧ IsConnected S := by
  sorry

end NUMINAMATH_CALUDE_Φ_is_connected_Φ_single_part_l1460_146099


namespace NUMINAMATH_CALUDE_consecutive_squares_sum_l1460_146063

theorem consecutive_squares_sum (n : ℤ) : 
  n^2 + (n + 1)^2 = 452 → n + 1 = 15 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_squares_sum_l1460_146063


namespace NUMINAMATH_CALUDE_commute_time_difference_l1460_146095

def commute_times (x y : ℝ) : List ℝ := [x, y, 8, 11, 9]

theorem commute_time_difference (x y : ℝ) :
  (List.sum (commute_times x y)) / 5 = 8 →
  (List.sum (List.map (λ t => (t - 8)^2) (commute_times x y))) / 5 = 4 →
  |x - y| = 2 := by
sorry

end NUMINAMATH_CALUDE_commute_time_difference_l1460_146095


namespace NUMINAMATH_CALUDE_quadratic_nature_l1460_146050

/-- A quadratic function g(x) with the condition c = a + b^2 -/
def g (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + (a + b^2)

theorem quadratic_nature (a b : ℝ) :
  (a < 0 → ∃ x₀, ∀ x, g a b x ≤ g a b x₀) ∧
  (a > 0 → ∃ x₀, ∀ x, g a b x ≥ g a b x₀) :=
sorry

end NUMINAMATH_CALUDE_quadratic_nature_l1460_146050


namespace NUMINAMATH_CALUDE_odd_function_geometric_sequence_l1460_146077

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then Real.log x + a / x
  else -(Real.log (-x) + a / (-x))

theorem odd_function_geometric_sequence (a : ℝ) :
  (∃ x₁ x₂ x₃ x₄ : ℝ,
    x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < x₄ ∧
    x₁ + x₄ = 0 ∧
    ∃ q : ℝ, q ≠ 0 ∧
      f a x₂ / f a x₁ = q ∧
      f a x₃ / f a x₂ = q ∧
      f a x₄ / f a x₃ = q) →
  a ≤ Real.sqrt 3 / (2 * Real.exp 1) :=
sorry

end NUMINAMATH_CALUDE_odd_function_geometric_sequence_l1460_146077


namespace NUMINAMATH_CALUDE_particular_number_addition_l1460_146018

theorem particular_number_addition : ∃ x : ℝ, 0.46 + x = 0.72 ∧ x = 0.26 := by
  sorry

end NUMINAMATH_CALUDE_particular_number_addition_l1460_146018


namespace NUMINAMATH_CALUDE_multiply_52_48_l1460_146040

theorem multiply_52_48 : 52 * 48 = 2496 := by
  sorry

end NUMINAMATH_CALUDE_multiply_52_48_l1460_146040


namespace NUMINAMATH_CALUDE_quadratic_properties_l1460_146001

/-- The quadratic function y = mx^2 - x - m + 1 where m ≠ 0 -/
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 - x - m + 1

theorem quadratic_properties (m : ℝ) (hm : m ≠ 0) :
  (∀ x, f m x = 0 → x = 1 ∨ x = (1 - m) / m) ∧
  (m < 0 → ∀ a b, f m a = 0 → f m b = 0 → a ≠ b → |a - b| > 2) ∧
  (m > 1 → ∀ x > 1, ∀ y > x, f m y > f m x) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_properties_l1460_146001


namespace NUMINAMATH_CALUDE_b_investment_is_8000_l1460_146022

/-- Represents the investment and profit distribution in a partnership business. -/
structure Partnership where
  a_investment : ℕ
  b_investment : ℕ
  c_investment : ℕ
  total_profit : ℕ
  c_profit : ℕ

/-- Calculates b's investment in the partnership. -/
def calculate_b_investment (p : Partnership) : ℕ :=
  (22 * p.c_investment - 9 * (p.a_investment + p.c_investment)) / 9

/-- Theorem stating that given the conditions, b's investment is 8000. -/
theorem b_investment_is_8000 (p : Partnership)
  (h1 : p.a_investment = 5000)
  (h2 : p.c_investment = 9000)
  (h3 : p.total_profit = 88000)
  (h4 : p.c_profit = 36000)
  : calculate_b_investment p = 8000 := by
  sorry

#eval calculate_b_investment ⟨5000, 0, 9000, 88000, 36000⟩

end NUMINAMATH_CALUDE_b_investment_is_8000_l1460_146022


namespace NUMINAMATH_CALUDE_equation_solution_l1460_146062

theorem equation_solution : ∃ x : ℝ, (2 / x = 1 / (x + 1)) ∧ (x = -2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1460_146062


namespace NUMINAMATH_CALUDE_car_distance_theorem_l1460_146097

/-- Calculates the distance between two cars on a main road -/
def distance_between_cars (initial_distance : ℝ) (car1_distance : ℝ) (car2_distance : ℝ) : ℝ :=
  initial_distance - car1_distance - car2_distance

theorem car_distance_theorem (initial_distance car1_distance car2_distance : ℝ) 
  (h1 : initial_distance = 150)
  (h2 : car1_distance = 50)
  (h3 : car2_distance = 62) :
  distance_between_cars initial_distance car1_distance car2_distance = 38 := by
  sorry

#eval distance_between_cars 150 50 62

end NUMINAMATH_CALUDE_car_distance_theorem_l1460_146097


namespace NUMINAMATH_CALUDE_box_volume_condition_unique_x_existence_l1460_146067

theorem box_volume_condition (x : ℕ) : Bool := 
  (x > 5) ∧ ((x + 5) * (x - 5) * (x^2 + 25) < 700)

theorem unique_x_existence : 
  ∃! x : ℕ, box_volume_condition x := by
  sorry

end NUMINAMATH_CALUDE_box_volume_condition_unique_x_existence_l1460_146067


namespace NUMINAMATH_CALUDE_select_four_with_one_girl_l1460_146029

/-- The number of ways to select 4 people from two groups with exactly 1 girl -/
def select_with_one_girl (boys_a boys_b girls_a girls_b : ℕ) : ℕ :=
  (girls_a.choose 1 * boys_a.choose 1 * boys_b.choose 2) +
  (boys_a.choose 2 * girls_b.choose 1 * boys_b.choose 1)

/-- Theorem stating the correct number of selections for the given group compositions -/
theorem select_four_with_one_girl :
  select_with_one_girl 5 6 3 2 = 345 := by
  sorry

end NUMINAMATH_CALUDE_select_four_with_one_girl_l1460_146029


namespace NUMINAMATH_CALUDE_ceiling_minus_x_bounds_l1460_146054

theorem ceiling_minus_x_bounds (x : ℝ) : 
  ⌈x⌉ - ⌊x⌋ = 1 → 0 < ⌈x⌉ - x ∧ ⌈x⌉ - x ≤ 1 := by sorry

end NUMINAMATH_CALUDE_ceiling_minus_x_bounds_l1460_146054


namespace NUMINAMATH_CALUDE_sin_cos_pi_over_12_l1460_146084

theorem sin_cos_pi_over_12 : 
  Real.sin (π / 12) * Real.cos (π / 12) = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_sin_cos_pi_over_12_l1460_146084


namespace NUMINAMATH_CALUDE_exist_valid_subgrid_l1460_146047

/-- Represents a grid of 0s and 1s -/
def Grid := Matrix (Fin 100) (Fin 2018) Bool

/-- A predicate that checks if a grid satisfies the condition of having at least 75 ones in each column -/
def ValidGrid (g : Grid) : Prop :=
  ∀ j : Fin 2018, (Finset.filter (fun i => g i j) Finset.univ).card ≥ 75

/-- A predicate that checks if a 5-row subgrid has at most one all-zero column -/
def ValidSubgrid (g : Grid) (rows : Finset (Fin 100)) : Prop :=
  rows.card = 5 ∧
  (Finset.filter (fun j : Fin 2018 => ∀ i ∈ rows, ¬g i j) Finset.univ).card ≤ 1

/-- The main theorem to be proved -/
theorem exist_valid_subgrid (g : Grid) (h : ValidGrid g) :
  ∃ rows : Finset (Fin 100), ValidSubgrid g rows := by
  sorry

end NUMINAMATH_CALUDE_exist_valid_subgrid_l1460_146047


namespace NUMINAMATH_CALUDE_apple_sales_proof_l1460_146023

/-- The number of kilograms of apples sold in the first hour -/
def first_hour_sales : ℝ := 10

/-- The number of kilograms of apples sold in the second hour -/
def second_hour_sales : ℝ := 2

/-- The average number of kilograms of apples sold per hour over two hours -/
def average_sales : ℝ := 6

theorem apple_sales_proof :
  first_hour_sales = 10 :=
by
  have h1 : average_sales = (first_hour_sales + second_hour_sales) / 2 :=
    sorry
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_apple_sales_proof_l1460_146023


namespace NUMINAMATH_CALUDE_smallest_n_for_prob_less_than_half_l1460_146025

def probability_red (n : ℕ) : ℚ :=
  9 / (11 - n)

theorem smallest_n_for_prob_less_than_half :
  ∀ n : ℕ, n > 0 → (∀ k : ℕ, 0 < k → k ≤ n → probability_red k < (1/2)) →
    n ≥ 8 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_prob_less_than_half_l1460_146025


namespace NUMINAMATH_CALUDE_dereks_current_dogs_l1460_146045

/-- Represents the number of dogs and cars Derek has at different ages -/
structure DereksPets where
  dogs_at_six : ℕ
  cars_at_six : ℕ
  cars_bought : ℕ
  current_dogs : ℕ

/-- Theorem stating the conditions and the result to be proven -/
theorem dereks_current_dogs (d : DereksPets) 
  (h1 : d.dogs_at_six = 3 * d.cars_at_six)
  (h2 : d.dogs_at_six = 90)
  (h3 : d.cars_bought = 210)
  (h4 : d.cars_at_six + d.cars_bought = 2 * d.current_dogs) :
  d.current_dogs = 120 := by
  sorry

end NUMINAMATH_CALUDE_dereks_current_dogs_l1460_146045


namespace NUMINAMATH_CALUDE_sum_of_four_consecutive_integers_divisible_by_two_l1460_146005

theorem sum_of_four_consecutive_integers_divisible_by_two :
  ∀ n : ℤ, ∃ k : ℤ, (n - 1) + n + (n + 1) + (n + 2) = 2 * k :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_consecutive_integers_divisible_by_two_l1460_146005


namespace NUMINAMATH_CALUDE_circle_symmetry_tangent_length_l1460_146034

/-- Circle C with equation x^2 + y^2 + 2x - 4y + 3 = 0 -/
def Circle (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y + 3 = 0

/-- Line of symmetry with equation 2ax + by + 6 = 0 -/
def SymmetryLine (a b x y : ℝ) : Prop := 2*a*x + b*y + 6 = 0

/-- Point (a, b) lies on the symmetry line -/
def PointOnSymmetryLine (a b : ℝ) : Prop := 2*a*a + b*b + 6 = 0

/-- Minimum length of tangent line segment from (a, b) to the circle -/
def MinTangentLength (a b : ℝ) : ℝ := 4

theorem circle_symmetry_tangent_length 
  (a b : ℝ) 
  (h1 : PointOnSymmetryLine a b) :
  MinTangentLength a b = 4 := by sorry

end NUMINAMATH_CALUDE_circle_symmetry_tangent_length_l1460_146034


namespace NUMINAMATH_CALUDE_sum_pascal_row_21st_triangular_l1460_146089

/-- The n-th triangular number -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The sum of entries in the n-th row of Pascal's triangle -/
def pascal_row_sum (n : ℕ) : ℕ := 2^n

theorem sum_pascal_row_21st_triangular : 
  pascal_row_sum (triangular_number 21 - 1) = 2^230 := by sorry

end NUMINAMATH_CALUDE_sum_pascal_row_21st_triangular_l1460_146089


namespace NUMINAMATH_CALUDE_probability_two_non_defective_pens_l1460_146055

theorem probability_two_non_defective_pens 
  (total_pens : ℕ) 
  (defective_pens : ℕ) 
  (selected_pens : ℕ) 
  (h1 : total_pens = 12) 
  (h2 : defective_pens = 4) 
  (h3 : selected_pens = 2) :
  (total_pens - defective_pens : ℚ) / total_pens * 
  ((total_pens - defective_pens - 1) : ℚ) / (total_pens - 1) = 14/33 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_non_defective_pens_l1460_146055


namespace NUMINAMATH_CALUDE_workshop_workers_l1460_146043

/-- The total number of workers in a workshop with specific salary conditions -/
theorem workshop_workers (total_avg : ℕ) (tech_count : ℕ) (tech_avg : ℕ) (non_tech_avg : ℕ) :
  total_avg = 8000 →
  tech_count = 7 →
  tech_avg = 18000 →
  non_tech_avg = 6000 →
  ∃ (total_workers : ℕ), total_workers = 42 ∧
    total_workers * total_avg = tech_count * tech_avg + (total_workers - tech_count) * non_tech_avg :=
by sorry

end NUMINAMATH_CALUDE_workshop_workers_l1460_146043


namespace NUMINAMATH_CALUDE_fraction_saved_is_one_third_l1460_146082

/-- Represents the fraction of take-home pay saved each month -/
def fraction_saved : ℝ := sorry

/-- Represents the monthly take-home pay -/
def monthly_pay : ℝ := sorry

/-- The total amount saved at the end of the year -/
def total_saved : ℝ := 12 * fraction_saved * monthly_pay

/-- The amount not saved in a month -/
def monthly_not_saved : ℝ := (1 - fraction_saved) * monthly_pay

/-- States that the total amount saved is 6 times the monthly amount not saved -/
axiom total_saved_eq_six_times_not_saved : total_saved = 6 * monthly_not_saved

/-- Theorem stating that the fraction saved each month is 1/3 -/
theorem fraction_saved_is_one_third : fraction_saved = 1/3 := by sorry

end NUMINAMATH_CALUDE_fraction_saved_is_one_third_l1460_146082


namespace NUMINAMATH_CALUDE_construction_company_stone_order_l1460_146068

/-- The weight of stone ordered by a construction company -/
theorem construction_company_stone_order
  (concrete : ℝ) (bricks : ℝ) (total : ℝ)
  (h1 : concrete = 0.16666666666666666)
  (h2 : bricks = 0.16666666666666666)
  (h3 : total = 0.8333333333333334) :
  total - (concrete + bricks) = 0.5 := by
sorry

end NUMINAMATH_CALUDE_construction_company_stone_order_l1460_146068


namespace NUMINAMATH_CALUDE_dumbbell_weight_problem_l1460_146008

theorem dumbbell_weight_problem (total_weight : ℝ) (first_pair_weight : ℝ) (third_pair_weight : ℝ) 
  (h1 : total_weight = 32)
  (h2 : first_pair_weight = 3)
  (h3 : third_pair_weight = 8) :
  total_weight - 2 * first_pair_weight - 2 * third_pair_weight = 10 := by
  sorry

end NUMINAMATH_CALUDE_dumbbell_weight_problem_l1460_146008


namespace NUMINAMATH_CALUDE_parabolas_intersection_l1460_146020

/-- The x-coordinates of the intersection points of two parabolas -/
def intersection_x : Set ℝ :=
  {x | x = 2 - Real.sqrt 26 ∨ x = 2 + Real.sqrt 26}

/-- The y-coordinate of the intersection points of two parabolas -/
def intersection_y : ℝ := 48

/-- The first parabola function -/
def f (x : ℝ) : ℝ := 3 * x^2 - 12 * x - 18

/-- The second parabola function -/
def g (x : ℝ) : ℝ := 2 * x^2 - 8 * x + 4

theorem parabolas_intersection :
  ∀ x y : ℝ, (f x = y ∧ g x = y) ↔ (x ∈ intersection_x ∧ y = intersection_y) :=
sorry

end NUMINAMATH_CALUDE_parabolas_intersection_l1460_146020


namespace NUMINAMATH_CALUDE_max_value_theorem_l1460_146046

/-- A function satisfying the given recurrence relation -/
def RecurrenceFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 1) = 1 + Real.sqrt (2 * f x - f x ^ 2)

/-- The theorem stating the maximum value of f(1) + f(2020) -/
theorem max_value_theorem (f : ℝ → ℝ) (h : RecurrenceFunction f) :
    ∃ M : ℝ, M = 2 + Real.sqrt 2 ∧ f 1 + f 2020 ≤ M ∧ 
    ∃ g : ℝ → ℝ, RecurrenceFunction g ∧ g 1 + g 2020 = M :=
  sorry

end NUMINAMATH_CALUDE_max_value_theorem_l1460_146046


namespace NUMINAMATH_CALUDE_purely_imaginary_iff_one_i_l1460_146065

/-- A complex number z is purely imaginary if and only if it has the form i (i.e., a = 1 and b = 0) -/
theorem purely_imaginary_iff_one_i (z : ℂ) : 
  (∃ (a b : ℝ), z = Complex.I * a + b) → 
  (z.re = 0 ∧ z.im ≠ 0) ↔ z = Complex.I :=
sorry

end NUMINAMATH_CALUDE_purely_imaginary_iff_one_i_l1460_146065


namespace NUMINAMATH_CALUDE_power_function_decreasing_m_l1460_146098

/-- A function f: ℝ → ℝ is a power function if it has the form f(x) = ax^b for some constants a and b, where a ≠ 0 -/
def IsPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, a ≠ 0 ∧ ∀ x : ℝ, x > 0 → f x = a * x ^ b

/-- A function f: ℝ → ℝ is decreasing on (0, +∞) if for any x₁, x₂ ∈ (0, +∞) with x₁ < x₂, we have f(x₁) > f(x₂) -/
def IsDecreasingOn (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → f x₁ > f x₂

/-- The main theorem -/
theorem power_function_decreasing_m (m : ℝ) :
  let f : ℝ → ℝ := fun x ↦ (m^2 - m - 1) * x^m
  IsPowerFunction f ∧ IsDecreasingOn f → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_power_function_decreasing_m_l1460_146098


namespace NUMINAMATH_CALUDE_two_std_dev_below_mean_l1460_146052

/-- For a normal distribution with mean μ and standard deviation σ,
    the value 2σ below the mean is μ - 2σ. -/
theorem two_std_dev_below_mean (μ σ : ℝ) (hμ : μ = 15.5) (hσ : σ = 1.5) :
  μ - 2 * σ = 12.5 := by
  sorry

#check two_std_dev_below_mean

end NUMINAMATH_CALUDE_two_std_dev_below_mean_l1460_146052


namespace NUMINAMATH_CALUDE_solution_set_and_range_l1460_146064

def f (a : ℝ) (x : ℝ) : ℝ := |2*x - a| + a

def g (x : ℝ) : ℝ := |2*x - 3|

theorem solution_set_and_range :
  (∀ x, x ∈ {y : ℝ | 0 ≤ y ∧ y ≤ 3} ↔ f 3 x ≤ 6) ∧
  (∀ a, (∀ x, f a x + g x ≥ 5) ↔ a ≥ 11/3) := by sorry

end NUMINAMATH_CALUDE_solution_set_and_range_l1460_146064


namespace NUMINAMATH_CALUDE_prob_at_least_one_box_same_color_l1460_146048

/-- The number of people placing blocks -/
def num_people : ℕ := 3

/-- The number of blocks each person has -/
def num_blocks : ℕ := 6

/-- The number of boxes -/
def num_boxes : ℕ := 6

/-- The probability of a specific color being placed in a specific box -/
def prob_specific_color : ℚ := 1 / num_blocks

/-- The probability that all three blocks in a specific box are the same color -/
def prob_same_color_in_box : ℚ := prob_specific_color ^ (num_people - 1)

/-- The probability that at least one box has all blocks of the same color -/
theorem prob_at_least_one_box_same_color :
  1 - (1 - prob_same_color_in_box) ^ num_boxes = 1 / 6 := by sorry

end NUMINAMATH_CALUDE_prob_at_least_one_box_same_color_l1460_146048


namespace NUMINAMATH_CALUDE_overall_profit_l1460_146079

def grinder_cost : ℚ := 15000
def mobile_cost : ℚ := 8000
def grinder_loss_percent : ℚ := 5
def mobile_profit_percent : ℚ := 10

def grinder_selling_price : ℚ := grinder_cost * (1 - grinder_loss_percent / 100)
def mobile_selling_price : ℚ := mobile_cost * (1 + mobile_profit_percent / 100)

def total_cost : ℚ := grinder_cost + mobile_cost
def total_selling_price : ℚ := grinder_selling_price + mobile_selling_price

theorem overall_profit : total_selling_price - total_cost = 50 := by
  sorry

end NUMINAMATH_CALUDE_overall_profit_l1460_146079


namespace NUMINAMATH_CALUDE_cryptarithm_solution_l1460_146044

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- The cryptarithm equations -/
def cryptarithm (A B C D E F G H J : Digit) : Prop :=
  (A.val * 10 + B.val) * (C.val * 10 + A.val) = D.val * 100 + E.val * 10 + B.val ∧
  F.val * 10 + C.val - (D.val * 10 + G.val) = D.val ∧
  E.val * 10 + G.val + H.val * 10 + J.val = A.val * 100 + A.val * 10 + G.val

/-- All digits are different -/
def all_different (A B C D E F G H J : Digit) : Prop :=
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧ A ≠ G ∧ A ≠ H ∧ A ≠ J ∧
  B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧ B ≠ G ∧ B ≠ H ∧ B ≠ J ∧
  C ≠ D ∧ C ≠ E ∧ C ≠ F ∧ C ≠ G ∧ C ≠ H ∧ C ≠ J ∧
  D ≠ E ∧ D ≠ F ∧ D ≠ G ∧ D ≠ H ∧ D ≠ J ∧
  E ≠ F ∧ E ≠ G ∧ E ≠ H ∧ E ≠ J ∧
  F ≠ G ∧ F ≠ H ∧ F ≠ J ∧
  G ≠ H ∧ G ≠ J ∧
  H ≠ J

theorem cryptarithm_solution :
  ∃! (A B C D E F G H J : Digit),
    cryptarithm A B C D E F G H J ∧
    all_different A B C D E F G H J ∧
    A.val = 1 ∧ B.val = 7 ∧ C.val = 2 ∧ D.val = 3 ∧
    E.val = 5 ∧ F.val = 4 ∧ G.val = 9 ∧ H.val = 6 ∧ J.val = 0 :=
by sorry

end NUMINAMATH_CALUDE_cryptarithm_solution_l1460_146044


namespace NUMINAMATH_CALUDE_units_digit_17_25_l1460_146006

theorem units_digit_17_25 : 17^25 % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_17_25_l1460_146006


namespace NUMINAMATH_CALUDE_fruit_basket_problem_l1460_146086

theorem fruit_basket_problem :
  Nat.gcd (Nat.gcd 15 9) 18 = 3 := by
  sorry

end NUMINAMATH_CALUDE_fruit_basket_problem_l1460_146086


namespace NUMINAMATH_CALUDE_butterfly_equal_roots_l1460_146070

/-- A quadratic equation ax^2 + bx + c = 0 is a "butterfly" equation if a - b + c = 0 -/
def is_butterfly_equation (a b c : ℝ) : Prop := a - b + c = 0

/-- The discriminant of a quadratic equation ax^2 + bx + c = 0 -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

theorem butterfly_equal_roots (a b c : ℝ) (ha : a ≠ 0) 
  (h_butterfly : is_butterfly_equation a b c) 
  (h_equal_roots : discriminant a b c = 0) : 
  a = c := by sorry

end NUMINAMATH_CALUDE_butterfly_equal_roots_l1460_146070


namespace NUMINAMATH_CALUDE_transformed_circle_equation_l1460_146036

-- Define the original circle
def original_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the scaling transformation
def scaling_transform (x y x' y' : ℝ) : Prop := x' = 5*x ∧ y' = 3*y

-- State the theorem
theorem transformed_circle_equation (x y x' y' : ℝ) :
  original_circle x y ∧ scaling_transform x y x' y' →
  x'^2 / 25 + y'^2 / 9 = 1 :=
by sorry

end NUMINAMATH_CALUDE_transformed_circle_equation_l1460_146036


namespace NUMINAMATH_CALUDE_pencil_price_l1460_146010

theorem pencil_price (joy_pencils colleen_pencils : ℕ) (price_difference : ℚ) :
  joy_pencils = 30 →
  colleen_pencils = 50 →
  price_difference = 80 →
  ∃ (price : ℚ), 
    colleen_pencils * price = joy_pencils * price + price_difference ∧
    price = 4 := by
  sorry

end NUMINAMATH_CALUDE_pencil_price_l1460_146010


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1460_146016

def A : Set ℤ := {-2, -1, 3, 4}
def B : Set ℤ := {-1, 2, 3}

theorem intersection_of_A_and_B : A ∩ B = {-1, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1460_146016


namespace NUMINAMATH_CALUDE_circle_radii_in_square_l1460_146058

theorem circle_radii_in_square (r : ℝ) : 
  r > 0 →  -- radius is positive
  r < 1/4 →  -- each circle fits in a corner
  (∀ (i j : Fin 4), i ≠ j → 
    (∃ (x y : ℝ), x^2 + y^2 = (2*r)^2 ∧ 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1)) →  -- circles touch
  (∃ (i j k : Fin 4), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ 
    (∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ), 
      (x₁ - x₂)^2 + (y₁ - y₂)^2 = (2*r)^2 ∧
      (x₂ - x₃)^2 + (y₂ - y₃)^2 = (2*r)^2 ∧
      (x₃ - x₁)^2 + (y₃ - y₁)^2 > (2*r)^2)) →  -- only two circles touch each other
  1 - Real.sqrt 2 / 2 < r ∧ r < 2 - Real.sqrt 2 / 2 - Real.sqrt (4 - 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_circle_radii_in_square_l1460_146058


namespace NUMINAMATH_CALUDE_unique_solution_l1460_146011

theorem unique_solution (x y z : ℝ) 
  (hx : x > 4) (hy : y > 4) (hz : z > 4)
  (h : (x + 3)^2 / (y + z - 3) + (y + 5)^2 / (z + x - 5) + (z + 7)^2 / (x + y - 7) = 45) :
  x = 12 ∧ y = 10 ∧ z = 8 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l1460_146011


namespace NUMINAMATH_CALUDE_horner_v₂_value_l1460_146057

def horner_polynomial (x : ℝ) : ℝ := x^6 - 5*x^5 + 6*x^4 - 3*x^3 + 1.8*x^2 + 0.35*x + 2

def v₀ : ℝ := 1

def v₁ (x : ℝ) : ℝ := v₀ * x - 5

def v₂ (x : ℝ) : ℝ := v₁ x * x + 6

theorem horner_v₂_value :
  v₂ (-1) = 12 :=
by sorry

end NUMINAMATH_CALUDE_horner_v₂_value_l1460_146057


namespace NUMINAMATH_CALUDE_radical_axes_intersect_l1460_146069

/-- A hexagon with vertices in a 2D plane -/
structure Hexagon :=
  (vertices : Fin 6 → ℝ × ℝ)

/-- A circle in a 2D plane -/
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

/-- The radical axis of two circles -/
def radical_axis (c1 c2 : Circle) : Set (ℝ × ℝ) := sorry

/-- A point lies on a circle -/
def on_circle (p : ℝ × ℝ) (c : Circle) : Prop := sorry

/-- The diagonals of a hexagon -/
def diagonals (h : Hexagon) : List (Set (ℝ × ℝ)) := sorry

/-- The intersection point of a list of sets -/
def intersection_point (sets : List (Set (ℝ × ℝ))) : Option (ℝ × ℝ) := sorry

/-- Main theorem -/
theorem radical_axes_intersect (h : Hexagon) : 
  (∀ (i j k l : Fin 6), i ≠ j ∧ j ≠ k ∧ k ≠ l ∧ l ≠ i → 
    ¬∃ (c : Circle), on_circle (h.vertices i) c ∧ on_circle (h.vertices j) c ∧ 
                     on_circle (h.vertices k) c ∧ on_circle (h.vertices l) c) →
  (∃! p, ∀ d ∈ diagonals h, p ∈ d) →
  ∃! p, ∀ (i j k : Fin 6), i ≠ j ∧ j ≠ k ∧ k ≠ i → 
    p ∈ radical_axis 
      (Circle.mk (h.vertices i) (dist (h.vertices i) (h.vertices j))) 
      (Circle.mk (h.vertices j) (dist (h.vertices j) (h.vertices k))) :=
by sorry

end NUMINAMATH_CALUDE_radical_axes_intersect_l1460_146069


namespace NUMINAMATH_CALUDE_team_b_four_wins_prob_l1460_146012

/-- Represents a team in the tournament -/
inductive Team
  | A
  | B
  | C

/-- The probability of one team beating another -/
def beat_prob (winner loser : Team) : ℝ :=
  match winner, loser with
  | Team.A, Team.B => 0.4
  | Team.B, Team.C => 0.5
  | Team.C, Team.A => 0.6
  | _, _ => 0 -- For other combinations, we set probability to 0

/-- The probability of Team B winning four consecutive matches -/
def team_b_four_wins : ℝ :=
  (1 - beat_prob Team.A Team.B) * (beat_prob Team.B Team.C) * 
  (1 - beat_prob Team.A Team.B) * (beat_prob Team.B Team.C)

theorem team_b_four_wins_prob : team_b_four_wins = 0.09 := by
  sorry

end NUMINAMATH_CALUDE_team_b_four_wins_prob_l1460_146012


namespace NUMINAMATH_CALUDE_point_on_line_l1460_146078

/-- The value of m that makes the point (3, -2) lie on the line 2m - my = 3x + 1 -/
theorem point_on_line (m : ℚ) : m = 5/2 ↔ 2*m - m*(-2) = 3*3 + 1 := by sorry

end NUMINAMATH_CALUDE_point_on_line_l1460_146078


namespace NUMINAMATH_CALUDE_triangle_formation_check_l1460_146003

-- Define a function to check if three lengths can form a triangle
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

-- Theorem statement
theorem triangle_formation_check :
  ¬(can_form_triangle 3 3 6) ∧
  ¬(can_form_triangle 2 3 6) ∧
  can_form_triangle 5 8 12 ∧
  ¬(can_form_triangle 4 7 11) :=
sorry

end NUMINAMATH_CALUDE_triangle_formation_check_l1460_146003


namespace NUMINAMATH_CALUDE_arctan_equation_solution_l1460_146053

theorem arctan_equation_solution (x : ℝ) :
  Real.arctan (1 / x) + Real.arctan (1 / x^3) = π / 4 → x = (1 + Real.sqrt 5) / 2 := by
sorry

end NUMINAMATH_CALUDE_arctan_equation_solution_l1460_146053


namespace NUMINAMATH_CALUDE_gasoline_tank_capacity_l1460_146056

theorem gasoline_tank_capacity : ∃ (capacity : ℝ), 
  capacity > 0 ∧
  (3/4 * capacity - 18 = 1/3 * capacity) ∧
  capacity = 43.2 := by
  sorry

end NUMINAMATH_CALUDE_gasoline_tank_capacity_l1460_146056


namespace NUMINAMATH_CALUDE_scientific_notation_exponent_l1460_146081

theorem scientific_notation_exponent (n : ℤ) :
  0.0000502 = 5.02 * (10 : ℝ) ^ n → n = -4 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_exponent_l1460_146081


namespace NUMINAMATH_CALUDE_other_number_proof_l1460_146049

theorem other_number_proof (a b : ℕ+) (h1 : Nat.lcm a b = 2310) (h2 : Nat.gcd a b = 26) (h3 : a = 210) : b = 286 := by
  sorry

end NUMINAMATH_CALUDE_other_number_proof_l1460_146049


namespace NUMINAMATH_CALUDE_tom_balloons_l1460_146026

theorem tom_balloons (initial : ℕ) (given : ℕ) (remaining : ℕ) : 
  initial = 30 → given = 16 → remaining = initial - given → remaining = 14 := by
  sorry

end NUMINAMATH_CALUDE_tom_balloons_l1460_146026


namespace NUMINAMATH_CALUDE_meat_price_proof_l1460_146032

/-- The price of meat per ounce in cents -/
def meat_price : ℝ := 6

theorem meat_price_proof :
  (∃ (paid_16 paid_8 : ℝ),
    16 * meat_price = paid_16 - 30 ∧
    8 * meat_price = paid_8 + 18) :=
by sorry

end NUMINAMATH_CALUDE_meat_price_proof_l1460_146032


namespace NUMINAMATH_CALUDE_product_of_six_integers_square_sum_l1460_146088

theorem product_of_six_integers_square_sum (ints : Finset ℕ) : 
  ints = {1, 2, 3, 4, 5, 6, 7, 8, 9} →
  ∃ (A B : Finset ℕ), 
    A ⊆ ints ∧ B ⊆ ints ∧
    A.card = 6 ∧ B.card = 6 ∧
    A ≠ B ∧
    (∃ p : ℕ, (A.prod id : ℕ) = p^2) ∧
    (∃ q : ℕ, (B.prod id : ℕ) = q^2) ∧
    ∃ (p q : ℕ), 
      (A.prod id : ℕ) = p^2 ∧
      (B.prod id : ℕ) = q^2 ∧
      p + q = 108 :=
by sorry

end NUMINAMATH_CALUDE_product_of_six_integers_square_sum_l1460_146088


namespace NUMINAMATH_CALUDE_spurs_team_size_l1460_146092

theorem spurs_team_size : 
  ∀ (num_players : ℕ) (basketballs_per_player : ℕ) (total_basketballs : ℕ),
    basketballs_per_player = 11 →
    total_basketballs = 242 →
    num_players * basketballs_per_player = total_basketballs →
    num_players = 22 := by
  sorry

end NUMINAMATH_CALUDE_spurs_team_size_l1460_146092


namespace NUMINAMATH_CALUDE_trig_identity_l1460_146076

theorem trig_identity (α : ℝ) : 
  -Real.sin α + Real.sqrt 3 * Real.cos α = 2 * Real.sin (α + 2 * Real.pi / 3) :=
by sorry

end NUMINAMATH_CALUDE_trig_identity_l1460_146076


namespace NUMINAMATH_CALUDE_binomial_1300_2_l1460_146031

theorem binomial_1300_2 : Nat.choose 1300 2 = 844350 := by sorry

end NUMINAMATH_CALUDE_binomial_1300_2_l1460_146031


namespace NUMINAMATH_CALUDE_cubic_equation_roots_difference_l1460_146041

theorem cubic_equation_roots_difference (p : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ 
    x^3 + 3*p*x^2 + (4*p - 1)*x + p = 0 ∧ 
    y^3 + 3*p*y^2 + (4*p - 1)*y + p = 0 ∧ 
    |x - y| = 1) ↔ 
  (p = 0 ∨ p = 6/5 ∨ p = 10/9) := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_difference_l1460_146041


namespace NUMINAMATH_CALUDE_orphanage_donation_percentage_l1460_146030

theorem orphanage_donation_percentage (total_income : ℝ) 
  (children_percentage : ℝ) (num_children : ℕ) (wife_percentage : ℝ) 
  (remaining_amount : ℝ) :
  total_income = 1200000 →
  children_percentage = 0.2 →
  num_children = 3 →
  wife_percentage = 0.3 →
  remaining_amount = 60000 →
  let distributed_percentage := children_percentage * num_children + wife_percentage
  let distributed_amount := distributed_percentage * total_income
  let amount_before_donation := total_income - distributed_amount
  let donation_amount := amount_before_donation - remaining_amount
  donation_amount / amount_before_donation = 0.5 := by sorry

end NUMINAMATH_CALUDE_orphanage_donation_percentage_l1460_146030

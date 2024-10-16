import Mathlib

namespace NUMINAMATH_CALUDE_license_plate_count_l1730_173088

/-- The number of consonants in the alphabet, including Y -/
def num_consonants : ℕ := 21

/-- The number of vowels in the alphabet -/
def num_vowels : ℕ := 5

/-- The number of digits (0-9) -/
def num_digits : ℕ := 10

/-- The total number of possible license plates -/
def total_license_plates : ℕ := num_consonants^2 * num_vowels^2 * num_digits^2

theorem license_plate_count : total_license_plates = 1102500 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l1730_173088


namespace NUMINAMATH_CALUDE_polynomial_roots_l1730_173023

theorem polynomial_roots : 
  let f : ℝ → ℝ := λ x => 3*x^4 + 2*x^3 - 7*x^2 + 2*x + 3
  let root1 : ℝ := ((-1 + 2*Real.sqrt 10)/3 + Real.sqrt (((-1 + 2*Real.sqrt 10)/3)^2 - 4))/2
  let root2 : ℝ := ((-1 + 2*Real.sqrt 10)/3 - Real.sqrt (((-1 + 2*Real.sqrt 10)/3)^2 - 4))/2
  let root3 : ℝ := ((-1 - 2*Real.sqrt 10)/3 + Real.sqrt (((-1 - 2*Real.sqrt 10)/3)^2 - 4))/2
  let root4 : ℝ := ((-1 - 2*Real.sqrt 10)/3 - Real.sqrt (((-1 - 2*Real.sqrt 10)/3)^2 - 4))/2
  (f root1 = 0) ∧ (f root2 = 0) ∧ (f root3 = 0) ∧ (f root4 = 0) ∧
  (∀ x : ℝ, f x = 0 → (x = root1 ∨ x = root2 ∨ x = root3 ∨ x = root4)) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_roots_l1730_173023


namespace NUMINAMATH_CALUDE_complex_product_QED_l1730_173046

theorem complex_product_QED : 
  let Q : ℂ := 5 + 3 * Complex.I
  let E : ℂ := 2 * Complex.I
  let D : ℂ := 5 - 3 * Complex.I
  Q * E * D = 68 * Complex.I := by
sorry

end NUMINAMATH_CALUDE_complex_product_QED_l1730_173046


namespace NUMINAMATH_CALUDE_pens_after_sale_l1730_173009

theorem pens_after_sale (initial_pens : ℕ) (sold_pens : ℕ) (h1 : initial_pens = 106) (h2 : sold_pens = 92) :
  initial_pens - sold_pens = 14 := by
  sorry

end NUMINAMATH_CALUDE_pens_after_sale_l1730_173009


namespace NUMINAMATH_CALUDE_initial_men_count_initial_men_count_correct_l1730_173054

/-- The number of days the initial food supply lasts for the initial group -/
def initial_days : ℕ := 22

/-- The number of days that pass before new men join -/
def days_before_joining : ℕ := 2

/-- The number of new men that join -/
def new_men : ℕ := 1140

/-- The number of additional days the food lasts after new men join -/
def additional_days : ℕ := 8

/-- Proves that the initial number of men is 760 -/
theorem initial_men_count : ℕ :=
  760

/-- Theorem stating that the initial_men_count satisfies the given conditions -/
theorem initial_men_count_correct :
  initial_men_count * initial_days =
  (initial_men_count + new_men) * additional_days +
  initial_men_count * days_before_joining :=
by
  sorry

end NUMINAMATH_CALUDE_initial_men_count_initial_men_count_correct_l1730_173054


namespace NUMINAMATH_CALUDE_ladies_walk_distance_l1730_173069

/-- Two ladies walk in Central Park. The first lady walks twice as far as the second lady,
    who walks 4 miles. Their combined total distance is 12 miles. -/
theorem ladies_walk_distance :
  ∀ (d1 d2 : ℝ),
    d1 = 2 * d2 →
    d2 = 4 →
    d1 + d2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_ladies_walk_distance_l1730_173069


namespace NUMINAMATH_CALUDE_compound_composition_l1730_173031

/-- Represents the number of atoms of each element in the compound -/
structure CompoundComposition where
  h : ℕ
  cl : ℕ
  o : ℕ

/-- Represents the atomic weights of elements -/
structure AtomicWeights where
  h : ℝ
  cl : ℝ
  o : ℝ

/-- Calculates the molecular weight of a compound given its composition and atomic weights -/
def molecularWeight (comp : CompoundComposition) (weights : AtomicWeights) : ℝ :=
  comp.h * weights.h + comp.cl * weights.cl + comp.o * weights.o

/-- The main theorem to prove -/
theorem compound_composition (weights : AtomicWeights) :
  let comp := CompoundComposition.mk 1 1 2
  weights.h = 1 ∧ weights.cl = 35.5 ∧ weights.o = 16 →
  molecularWeight comp weights = 68 := by
  sorry

end NUMINAMATH_CALUDE_compound_composition_l1730_173031


namespace NUMINAMATH_CALUDE_sin_inequality_l1730_173097

theorem sin_inequality (x : ℝ) : 
  let f : ℝ → ℝ := λ x => Real.sin (2 * x + π / 6)
  f x ≤ |f (π / 6)| := by
sorry

end NUMINAMATH_CALUDE_sin_inequality_l1730_173097


namespace NUMINAMATH_CALUDE_labor_cost_increase_l1730_173080

/-- Represents the cost components of manufacturing a car --/
structure CarCost where
  raw_material : ℝ
  labor : ℝ
  overhead : ℝ

/-- The initial cost ratio --/
def initial_ratio : CarCost := ⟨4, 3, 2⟩

/-- The percentage changes in costs --/
structure CostChanges where
  raw_material : ℝ := 0.10  -- 10% increase
  labor : ℝ                 -- Unknown, to be calculated
  overhead : ℝ := -0.05     -- 5% decrease
  total : ℝ := 0.06         -- 6% increase

/-- Calculates the new cost based on the initial cost and percentage change --/
def new_cost (initial : ℝ) (change : ℝ) : ℝ :=
  initial * (1 + change)

/-- Theorem stating that the labor cost increased by 8% --/
theorem labor_cost_increase (c : CostChanges) :
  c.labor = 0.08 := by sorry

end NUMINAMATH_CALUDE_labor_cost_increase_l1730_173080


namespace NUMINAMATH_CALUDE_cookies_theorem_l1730_173001

def cookies_problem (initial : ℕ) (first_friend : ℕ) (second_friend : ℕ) (eaten : ℕ) (bought : ℕ) (third_friend : ℕ) : Prop :=
  let remaining_after_first := initial - first_friend
  let remaining_after_second := remaining_after_first - second_friend
  let remaining_after_eating := remaining_after_second - eaten
  let remaining_after_buying := remaining_after_eating + bought
  let final_remaining := remaining_after_buying - third_friend
  final_remaining = 67

theorem cookies_theorem : cookies_problem 120 34 29 20 45 15 := by
  sorry

end NUMINAMATH_CALUDE_cookies_theorem_l1730_173001


namespace NUMINAMATH_CALUDE_append_digits_divisible_by_53_l1730_173065

theorem append_digits_divisible_by_53 (x y : Nat) : 
  x < 10 ∧ y < 10 ∧ (131300 + 10 * x + y) % 53 = 0 ↔ (x = 3 ∧ y = 4) ∨ (x = 8 ∧ y = 7) := by
  sorry

end NUMINAMATH_CALUDE_append_digits_divisible_by_53_l1730_173065


namespace NUMINAMATH_CALUDE_ringbinder_price_decrease_l1730_173011

def original_backpack_price : ℝ := 50
def original_ringbinder_price : ℝ := 20
def backpack_price_increase : ℝ := 5
def num_ringbinders : ℕ := 3
def total_spent : ℝ := 109

theorem ringbinder_price_decrease :
  ∃ (x : ℝ),
    x = 2 ∧
    (original_backpack_price + backpack_price_increase) +
    num_ringbinders * (original_ringbinder_price - x) = total_spent :=
by sorry

end NUMINAMATH_CALUDE_ringbinder_price_decrease_l1730_173011


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1730_173015

/-- An isosceles triangle with side lengths a and b satisfying a certain equation -/
structure IsoscelesTriangle where
  a : ℝ
  b : ℝ
  isosceles : True  -- We don't need to specify which sides are equal for this problem
  equation : Real.sqrt (2 * a - 3 * b + 5) + (2 * a + 3 * b - 13)^2 = 0

/-- The perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℝ :=
  t.a + t.a + t.b

/-- Theorem stating that the perimeter is either 7 or 8 -/
theorem isosceles_triangle_perimeter (t : IsoscelesTriangle) :
  perimeter t = 7 ∨ perimeter t = 8 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1730_173015


namespace NUMINAMATH_CALUDE_arithmetic_sequence_max_sum_l1730_173074

/-- An arithmetic sequence with its properties -/
structure ArithmeticSequence where
  a : ℕ+ → ℚ
  is_arithmetic : ∀ n m : ℕ+, a (n + 1) - a n = a (m + 1) - a m

/-- Sum of the first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ+) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

theorem arithmetic_sequence_max_sum (seq : ArithmeticSequence) :
  (seq.a 1 + seq.a 4 + seq.a 7 = 99) →
  (seq.a 2 + seq.a 5 + seq.a 8 = 93) →
  (∀ n : ℕ+, S seq n ≤ S seq 20) →
  (∀ k : ℕ+, (∀ n : ℕ+, S seq n ≤ S seq k) → k = 20) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_max_sum_l1730_173074


namespace NUMINAMATH_CALUDE_sum_to_n_equals_91_l1730_173059

theorem sum_to_n_equals_91 : ∃ n : ℕ, n * (n + 1) / 2 = 91 ∧ n = 13 := by
  sorry

end NUMINAMATH_CALUDE_sum_to_n_equals_91_l1730_173059


namespace NUMINAMATH_CALUDE_solve_system_l1730_173030

theorem solve_system (y z x : ℚ) 
  (h1 : (2 : ℚ) / 3 = y / 90)
  (h2 : (2 : ℚ) / 3 = (y + z) / 120)
  (h3 : (2 : ℚ) / 3 = (x - z) / 150) : 
  x = 120 := by sorry

end NUMINAMATH_CALUDE_solve_system_l1730_173030


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l1730_173051

theorem expression_simplification_and_evaluation (a b : ℝ) 
  (h1 : a = 1) (h2 : b = -1) : 
  (2*a^2*b - 2*a*b^2 - b^3) / b - (a + b)*(a - b) = 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l1730_173051


namespace NUMINAMATH_CALUDE_painting_price_change_l1730_173040

/-- Calculates the final price of a painting after a series of value changes and currency depreciation. -/
def final_price_percentage (initial_increase : ℝ) (first_decrease : ℝ) (second_decrease : ℝ) 
  (discount : ℝ) (currency_depreciation : ℝ) : ℝ :=
  let year1 := 1 + initial_increase
  let year2 := year1 * (1 - first_decrease)
  let year3 := year2 * (1 - second_decrease)
  let discounted := year3 * (1 - discount)
  discounted * (1 + currency_depreciation)

/-- Theorem stating that the final price of the painting is 113.373% of the original price -/
theorem painting_price_change : 
  ∀ (ε : ℝ), ε > 0 → 
  |final_price_percentage 0.30 0.15 0.10 0.05 0.20 - 1.13373| < ε :=
sorry

end NUMINAMATH_CALUDE_painting_price_change_l1730_173040


namespace NUMINAMATH_CALUDE_pie_distribution_l1730_173096

theorem pie_distribution (T R B S : ℕ) : 
  R = T / 2 →
  B = R - 14 →
  S = (R + B) / 2 →
  T = R + B + S →
  (T = 42 ∧ R = 21 ∧ B = 7 ∧ S = 14) :=
by sorry

end NUMINAMATH_CALUDE_pie_distribution_l1730_173096


namespace NUMINAMATH_CALUDE_juice_fraction_is_one_fourth_l1730_173004

/-- Represents the contents of a cup -/
structure CupContents where
  milk : ℚ
  juice : ℚ

/-- Represents the state of both cups -/
structure TwoCups where
  cup1 : CupContents
  cup2 : CupContents

def initial_state : TwoCups := {
  cup1 := { milk := 6, juice := 0 },
  cup2 := { milk := 0, juice := 6 }
}

def transfer_milk (state : TwoCups) : TwoCups := {
  cup1 := { milk := state.cup1.milk * 2/3, juice := state.cup1.juice },
  cup2 := { milk := state.cup2.milk + state.cup1.milk * 1/3, juice := state.cup2.juice }
}

def transfer_mixture (state : TwoCups) : TwoCups :=
  let total2 := state.cup2.milk + state.cup2.juice
  let transfer_amount := total2 * 1/4
  let milk_fraction := state.cup2.milk / total2
  let juice_fraction := state.cup2.juice / total2
  {
    cup1 := {
      milk := state.cup1.milk + transfer_amount * milk_fraction,
      juice := state.cup1.juice + transfer_amount * juice_fraction
    },
    cup2 := {
      milk := state.cup2.milk - transfer_amount * milk_fraction,
      juice := state.cup2.juice - transfer_amount * juice_fraction
    }
  }

def final_state : TwoCups :=
  transfer_mixture (transfer_milk initial_state)

theorem juice_fraction_is_one_fourth :
  (final_state.cup1.juice) / (final_state.cup1.milk + final_state.cup1.juice) = 1/4 := by
  sorry


end NUMINAMATH_CALUDE_juice_fraction_is_one_fourth_l1730_173004


namespace NUMINAMATH_CALUDE_binomial_distribution_problem_l1730_173087

/-- A random variable following a binomial distribution -/
structure BinomialVariable where
  n : ℕ
  p : ℝ
  ξ : ℝ

/-- The expectation of a binomial variable -/
def expectation (X : BinomialVariable) : ℝ := X.n * X.p

/-- The variance of a binomial variable -/
def variance (X : BinomialVariable) : ℝ := X.n * X.p * (1 - X.p)

theorem binomial_distribution_problem (X : BinomialVariable) 
  (h1 : expectation X = 300)
  (h2 : variance X = 200) :
  X.p = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_binomial_distribution_problem_l1730_173087


namespace NUMINAMATH_CALUDE_trigonometric_sequence_solution_l1730_173077

theorem trigonometric_sequence_solution (a : ℕ → ℝ) (d : ℝ) :
  (∀ n, 2 * (Real.cos (a n))^2 = Real.cos (a (n + 1))) →
  (∀ n, Real.cos (a (n + 1)) ≥ 0) →
  (∀ n, |Real.cos (a n)| ≤ 1 / Real.sqrt 2) →
  (∀ n, a (n + 1) = a n + d) →
  (∃ k : ℤ, d = 2 * Real.pi * ↑k ∧ k ≠ 0) →
  (∃ m : ℤ, a 1 = Real.pi / 2 + Real.pi * ↑m) ∨
  (∃ m : ℤ, a 1 = Real.pi / 3 + 2 * Real.pi * ↑m) ∨
  (∃ m : ℤ, a 1 = -Real.pi / 3 + 2 * Real.pi * ↑m) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_sequence_solution_l1730_173077


namespace NUMINAMATH_CALUDE_mixtape_song_length_l1730_173067

/-- Represents a mixtape with two sides -/
structure Mixtape where
  side1_songs : ℕ
  side2_songs : ℕ
  total_length : ℕ

/-- Theorem: Given a mixtape with 6 songs on side 1, 4 songs on side 2, 
    and a total length of 40 minutes, if all songs have the same length, 
    then each song is 4 minutes long. -/
theorem mixtape_song_length (m : Mixtape) 
    (h1 : m.side1_songs = 6)
    (h2 : m.side2_songs = 4)
    (h3 : m.total_length = 40) :
    m.total_length / (m.side1_songs + m.side2_songs) = 4 := by
  sorry

end NUMINAMATH_CALUDE_mixtape_song_length_l1730_173067


namespace NUMINAMATH_CALUDE_inscribed_hexagon_area_l1730_173058

/-- The area of a regular hexagon inscribed in a circle with area 324π -/
theorem inscribed_hexagon_area :
  ∀ (circle_area hexagon_area : ℝ),
  circle_area = 324 * Real.pi →
  hexagon_area = 6 * (((Real.sqrt (circle_area / Real.pi)) ^ 2 * Real.sqrt 3) / 4) →
  hexagon_area = 486 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_hexagon_area_l1730_173058


namespace NUMINAMATH_CALUDE_smallest_valid_number_l1730_173013

def is_valid (n : ℕ) : Prop :=
  n > 9 ∧
  ¬(n % 7 = 0) ∧
  ∀ (i : ℕ), i < (String.length (toString n)) →
    ((n.div (10^i) % 10) ≠ 7) ∧
    (((n - (n.div (10^i) % 10) * 10^i + 7 * 10^i) % 7 = 0))

theorem smallest_valid_number :
  is_valid 13264513 ∧ ∀ (m : ℕ), m < 13264513 → ¬(is_valid m) := by sorry

end NUMINAMATH_CALUDE_smallest_valid_number_l1730_173013


namespace NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l1730_173072

theorem coefficient_x_squared_in_expansion (x : ℝ) : 
  (Finset.range 7).sum (fun k => (Nat.choose 6 k) * ((-1)^k) * ((Real.sqrt 2)^(6-k)) * (x^k)) = 
  60 * x^2 + (Finset.range 7).sum (fun k => if k ≠ 2 then (Nat.choose 6 k) * ((-1)^k) * ((Real.sqrt 2)^(6-k)) * (x^k) else 0) :=
by sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l1730_173072


namespace NUMINAMATH_CALUDE_race_heartbeats_l1730_173039

/-- Calculates the total number of heartbeats during a race. -/
def total_heartbeats (heart_rate : ℕ) (pace : ℕ) (race_distance : ℕ) : ℕ :=
  (race_distance * pace * heart_rate)

/-- Proves that the total number of heartbeats during a 30-mile race is 28800,
    given the specified heart rate and pace. -/
theorem race_heartbeats :
  total_heartbeats 160 6 30 = 28800 := by
  sorry

end NUMINAMATH_CALUDE_race_heartbeats_l1730_173039


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l1730_173020

/-- Given that x and y are positive real numbers, x² and y² vary inversely,
    and y = 5 when x = 2, prove that x = 2/25 when y = 125. -/
theorem inverse_variation_problem (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h_inverse : ∃ (k : ℝ), ∀ x y, x^2 * y^2 = k)
  (h_initial : 2^2 * 5^2 = x^2 * 125^2) :
  y = 125 → x = 2/25 := by
sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l1730_173020


namespace NUMINAMATH_CALUDE_milk_water_ratio_in_combined_mixture_l1730_173062

/-- Represents a mixture of milk and water -/
structure Mixture where
  milk : ℚ
  water : ℚ

/-- Calculates the ratio of milk to water in a mixture -/
def ratioMilkToWater (m : Mixture) : ℚ × ℚ :=
  (m.milk, m.water)

/-- Combines multiple mixtures into a single mixture -/
def combineMixtures (mixtures : List Mixture) : Mixture :=
  { milk := mixtures.map (·.milk) |>.sum,
    water := mixtures.map (·.water) |>.sum }

theorem milk_water_ratio_in_combined_mixture :
  let m1 := Mixture.mk (7 : ℚ) (2 : ℚ)
  let m2 := Mixture.mk (8 : ℚ) (1 : ℚ)
  let m3 := Mixture.mk (9 : ℚ) (3 : ℚ)
  let combined := combineMixtures [m1, m2, m3]
  ratioMilkToWater combined = (29, 7) := by
  sorry

#check milk_water_ratio_in_combined_mixture

end NUMINAMATH_CALUDE_milk_water_ratio_in_combined_mixture_l1730_173062


namespace NUMINAMATH_CALUDE_garden_breadth_calculation_l1730_173014

/-- Represents a rectangular garden -/
structure RectangularGarden where
  length : ℝ
  breadth : ℝ

/-- Calculates the perimeter of a rectangular garden -/
def perimeter (garden : RectangularGarden) : ℝ :=
  2 * (garden.length + garden.breadth)

theorem garden_breadth_calculation (garden : RectangularGarden) 
  (h1 : perimeter garden = 480)
  (h2 : garden.length = 140) :
  garden.breadth = 100 := by
  sorry

end NUMINAMATH_CALUDE_garden_breadth_calculation_l1730_173014


namespace NUMINAMATH_CALUDE_complex_product_modulus_l1730_173090

theorem complex_product_modulus : Complex.abs (4 - 3*I) * Complex.abs (4 + 3*I) = 25 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_modulus_l1730_173090


namespace NUMINAMATH_CALUDE_intersection_segment_length_l1730_173024

noncomputable section

/-- Curve C in Cartesian coordinates -/
def C (x y : ℝ) : Prop := x^2 = 4*y

/-- Line l in Cartesian coordinates -/
def l (x y : ℝ) : Prop := y = x + 1

/-- Point on both curve C and line l -/
def intersection_point (p : ℝ × ℝ) : Prop :=
  C p.1 p.2 ∧ l p.1 p.2

/-- Distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem intersection_segment_length :
  ∃ (M N : ℝ × ℝ), intersection_point M ∧ intersection_point N ∧ distance M N = 8 :=
sorry

end

end NUMINAMATH_CALUDE_intersection_segment_length_l1730_173024


namespace NUMINAMATH_CALUDE_trig_identity_proof_l1730_173003

theorem trig_identity_proof :
  Real.cos (10 * π / 180) * Real.cos (20 * π / 180) - 
  Real.cos (80 * π / 180) * Real.sin (20 * π / 180) = 
  Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_proof_l1730_173003


namespace NUMINAMATH_CALUDE_discount_order_difference_l1730_173092

theorem discount_order_difference (initial_price : ℝ) (flat_discount : ℝ) (percentage_discount : ℝ) : 
  initial_price = 30 ∧ 
  flat_discount = 5 ∧ 
  percentage_discount = 0.25 →
  (initial_price - flat_discount) * (1 - percentage_discount) - 
  (initial_price * (1 - percentage_discount) - flat_discount) = 1.25 := by
sorry

end NUMINAMATH_CALUDE_discount_order_difference_l1730_173092


namespace NUMINAMATH_CALUDE_fixed_point_of_quadratic_l1730_173002

/-- The quadratic function y = -x^2 + (m-1)x + m passes through the point (-1, 0) for all real m. -/
theorem fixed_point_of_quadratic (m : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ -x^2 + (m-1)*x + m
  f (-1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_quadratic_l1730_173002


namespace NUMINAMATH_CALUDE_line_arrangements_with_restriction_l1730_173064

def num_students : Nat := 4

def total_arrangements : Nat := Nat.factorial num_students

def arrangements_with_restricted_pair : Nat :=
  (Nat.factorial (num_students - 1)) * (Nat.factorial 2)

theorem line_arrangements_with_restriction :
  total_arrangements - arrangements_with_restricted_pair = 12 := by
  sorry

end NUMINAMATH_CALUDE_line_arrangements_with_restriction_l1730_173064


namespace NUMINAMATH_CALUDE_equation_represents_two_hyperbolas_l1730_173005

-- Define the equation
def equation (x y : ℝ) : Prop :=
  y^4 - 6*x^4 = 3*y^2 - 2

-- Define what a hyperbola equation looks like
def is_hyperbola_equation (a b c : ℝ) (x y : ℝ) : Prop :=
  y^2 - a*x^2 = c ∧ b ≠ 0

-- Theorem statement
theorem equation_represents_two_hyperbolas :
  ∃ (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ),
    (∀ x y : ℝ, equation x y ↔ 
      (is_hyperbola_equation a₁ b₁ c₁ x y ∨ is_hyperbola_equation a₂ b₂ c₂ x y)) ∧
    b₁ ≠ 0 ∧ b₂ ≠ 0 ∧ (a₁ ≠ a₂ ∨ c₁ ≠ c₂) :=
sorry

end NUMINAMATH_CALUDE_equation_represents_two_hyperbolas_l1730_173005


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l1730_173035

theorem contrapositive_equivalence (a b x : ℝ) :
  (x ≥ a^2 + b^2 → x ≥ 2*a*b) ↔ (x < 2*a*b → x < a^2 + b^2) := by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l1730_173035


namespace NUMINAMATH_CALUDE_bike_price_l1730_173036

theorem bike_price (upfront_percentage : ℝ) (upfront_payment : ℝ) (total_price : ℝ) : 
  upfront_percentage = 0.20 → 
  upfront_payment = 200 → 
  upfront_percentage * total_price = upfront_payment → 
  total_price = 1000 := by
sorry

end NUMINAMATH_CALUDE_bike_price_l1730_173036


namespace NUMINAMATH_CALUDE_nested_fourth_root_l1730_173027

theorem nested_fourth_root (M : ℝ) (h : M > 1) :
  (M * (M * (M * M^(1/4))^(1/4))^(1/4))^(1/4) = M^(21/64) := by
  sorry

end NUMINAMATH_CALUDE_nested_fourth_root_l1730_173027


namespace NUMINAMATH_CALUDE_max_value_a_l1730_173095

theorem max_value_a : ∃ (a : ℝ) (b : ℤ), 
  (a * b^2) / (a + 2 * ↑b) = 2019 ∧ 
  ∀ (a' : ℝ) (b' : ℤ), (a' * b'^2) / (a' + 2 * ↑b') = 2019 → a' ≤ a :=
by sorry

end NUMINAMATH_CALUDE_max_value_a_l1730_173095


namespace NUMINAMATH_CALUDE_fraction_evaluation_l1730_173000

theorem fraction_evaluation : (1 - 1/4) / (1 - 1/5) = 15/16 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l1730_173000


namespace NUMINAMATH_CALUDE_power_of_64_l1730_173091

theorem power_of_64 : (64 : ℝ) ^ (5/3) = 1024 := by
  sorry

end NUMINAMATH_CALUDE_power_of_64_l1730_173091


namespace NUMINAMATH_CALUDE_number_in_seventh_group_l1730_173078

/-- Represents the systematic sampling method for a population of 100 individuals -/
structure SystematicSampling where
  population_size : Nat
  num_groups : Nat
  sample_size : Nat
  first_number : Nat

/-- The number drawn in the k-th group -/
def number_in_group (ss : SystematicSampling) (k : Nat) : Nat :=
  (ss.first_number + k - 1) % 10 + (k - 1) * 10

/-- Theorem stating that the number drawn in the 7th group is 63 -/
theorem number_in_seventh_group (ss : SystematicSampling) : 
  ss.population_size = 100 →
  ss.num_groups = 10 →
  ss.sample_size = 10 →
  ss.first_number = 6 →
  number_in_group ss 7 = 63 := by
  sorry

end NUMINAMATH_CALUDE_number_in_seventh_group_l1730_173078


namespace NUMINAMATH_CALUDE_cubic_polynomial_integer_root_l1730_173007

theorem cubic_polynomial_integer_root 
  (b c : ℚ) 
  (h1 : (3 - Real.sqrt 5)^3 + b*(3 - Real.sqrt 5) + c = 0) 
  (h2 : ∃ (n : ℤ), n^3 + b*n + c = 0) :
  ∃ (n : ℤ), n^3 + b*n + c = 0 ∧ n = -6 := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_integer_root_l1730_173007


namespace NUMINAMATH_CALUDE_permutation_formula_l1730_173047

def A (n k : ℕ) : ℕ :=
  (List.range k).foldl (fun acc i => acc * (n - i)) n

theorem permutation_formula (n k : ℕ) (h : k ≤ n) :
  A n k = (List.range k).foldl (fun acc i => acc * (n - i)) n :=
by sorry

end NUMINAMATH_CALUDE_permutation_formula_l1730_173047


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l1730_173045

theorem absolute_value_inequality (x : ℝ) :
  (3 ≤ |x + 3| ∧ |x + 3| ≤ 7) ↔ ((-10 ≤ x ∧ x ≤ -6) ∨ (0 ≤ x ∧ x ≤ 4)) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l1730_173045


namespace NUMINAMATH_CALUDE_root_negative_implies_inequality_l1730_173076

theorem root_negative_implies_inequality (a : ℝ) : 
  (∃ x : ℝ, x - 2*a + 4 = 0 ∧ x < 0) → (a-3)*(a-4) > 0 := by
  sorry

end NUMINAMATH_CALUDE_root_negative_implies_inequality_l1730_173076


namespace NUMINAMATH_CALUDE_square_root_and_abs_simplification_l1730_173028

theorem square_root_and_abs_simplification :
  Real.sqrt ((-2)^2) + |Real.sqrt 2 - Real.sqrt 3| - |Real.sqrt 3 - 1| = 3 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_square_root_and_abs_simplification_l1730_173028


namespace NUMINAMATH_CALUDE_smallest_regular_polygon_with_28_degree_extension_l1730_173008

/-- The angle (in degrees) at which two extended sides of a regular polygon meet -/
def extended_angle (n : ℕ) : ℚ :=
  180 / n

/-- Theorem stating that 45 is the smallest positive integer n for which
    a regular n-sided polygon has two extended sides meeting at an angle of 28 degrees -/
theorem smallest_regular_polygon_with_28_degree_extension :
  (∀ k : ℕ, k > 0 → k < 45 → extended_angle k ≠ 28) ∧ extended_angle 45 = 28 :=
sorry

end NUMINAMATH_CALUDE_smallest_regular_polygon_with_28_degree_extension_l1730_173008


namespace NUMINAMATH_CALUDE_logarithm_simplification_l1730_173089

theorem logarithm_simplification (x : Real) (h : 0 < x ∧ x < Real.pi / 2) :
  Real.log (Real.cos x * Real.tan x + 1 - 2 * Real.sin (x / 2) ^ 2) +
  Real.log (Real.sqrt 2 * Real.cos (x - Real.pi / 4)) -
  Real.log (1 + Real.sin (2 * x)) = 0 := by sorry

end NUMINAMATH_CALUDE_logarithm_simplification_l1730_173089


namespace NUMINAMATH_CALUDE_vanya_number_l1730_173038

theorem vanya_number : ∃! n : ℕ, 
  10 ≤ n ∧ n < 100 ∧ 
  let m := n / 10
  let d := n % 10
  (10 * d + m)^2 = 4 * n :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_vanya_number_l1730_173038


namespace NUMINAMATH_CALUDE_thirty_is_seventy_five_percent_of_forty_l1730_173018

theorem thirty_is_seventy_five_percent_of_forty :
  ∀ x : ℝ, (75 / 100) * x = 30 → x = 40 := by
  sorry

end NUMINAMATH_CALUDE_thirty_is_seventy_five_percent_of_forty_l1730_173018


namespace NUMINAMATH_CALUDE_unique_number_between_cube_roots_l1730_173034

theorem unique_number_between_cube_roots : ∃! (n : ℕ),
  n > 0 ∧ 24 ∣ n ∧ (9 : ℝ) < n ^ (1/3) ∧ n ^ (1/3) < (9.1 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_unique_number_between_cube_roots_l1730_173034


namespace NUMINAMATH_CALUDE_lamplighter_monkey_distance_l1730_173010

/-- Represents the speed and duration of a monkey's movement. -/
structure MonkeyMovement where
  speed : ℝ
  duration : ℝ

/-- Calculates the total distance traveled by a Lamplighter monkey. -/
def totalDistance (running : MonkeyMovement) (swinging : MonkeyMovement) : ℝ :=
  running.speed * running.duration + swinging.speed * swinging.duration

/-- Theorem: A Lamplighter monkey travels 175 feet given the specified conditions. -/
theorem lamplighter_monkey_distance :
  let running : MonkeyMovement := ⟨15, 5⟩
  let swinging : MonkeyMovement := ⟨10, 10⟩
  totalDistance running swinging = 175 := by
  sorry

#eval totalDistance ⟨15, 5⟩ ⟨10, 10⟩

end NUMINAMATH_CALUDE_lamplighter_monkey_distance_l1730_173010


namespace NUMINAMATH_CALUDE_tangent_line_smallest_slope_l1730_173081

-- Define the curve
def f (x : ℝ) : ℝ := x^3 + 3*x^2 + 6*x - 10

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 + 6*x + 6

-- Theorem statement
theorem tangent_line_smallest_slope :
  ∃ (x₀ y₀ : ℝ),
    (y₀ = f x₀) ∧
    (∀ x, f' x₀ ≤ f' x) ∧
    (3*x₀ - y₀ - 11 = 0) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_smallest_slope_l1730_173081


namespace NUMINAMATH_CALUDE_smallest_integer_solution_three_is_solution_three_is_smallest_solution_l1730_173037

theorem smallest_integer_solution (x : ℤ) : (6 - 3 * x < 0) → x ≥ 3 :=
by sorry

theorem three_is_solution : 6 - 3 * 3 < 0 :=
by sorry

theorem three_is_smallest_solution : ∀ y : ℤ, y < 3 → 6 - 3 * y ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_solution_three_is_solution_three_is_smallest_solution_l1730_173037


namespace NUMINAMATH_CALUDE_complex_computations_l1730_173017

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem complex_computations :
  (Complex.abs (3 - i) = Real.sqrt 10) ∧
  ((10 * i) / (3 - i) = -1 + 3 * i) :=
by sorry

end NUMINAMATH_CALUDE_complex_computations_l1730_173017


namespace NUMINAMATH_CALUDE_function_property_l1730_173029

def is_symmetric_about_one (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 1) = f (1 - x)

def is_increasing_on_right_of_one (f : ℝ → ℝ) : Prop :=
  ∀ x y, 1 ≤ x → x < y → f x < f y

def satisfies_inequality (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, x ∈ Set.Icc (1/2) 1 → f (a * x + 2) ≤ f (x - 1)

theorem function_property (f : ℝ → ℝ) (h1 : is_symmetric_about_one f)
    (h2 : is_increasing_on_right_of_one f) :
    {a : ℝ | satisfies_inequality f a} = Set.Icc (-2) 0 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l1730_173029


namespace NUMINAMATH_CALUDE_angle_measure_proof_l1730_173033

theorem angle_measure_proof (x : ℝ) : 
  (x + (3 * x + 10) = 90) → x = 20 := by
sorry

end NUMINAMATH_CALUDE_angle_measure_proof_l1730_173033


namespace NUMINAMATH_CALUDE_polynomial_division_l1730_173052

theorem polynomial_division (x : ℝ) :
  5*x^4 - 9*x^3 + 3*x^2 + 7*x - 6 = (x - 1)*(5*x^3 - 4*x^2 + 7*x + 7) := by sorry

end NUMINAMATH_CALUDE_polynomial_division_l1730_173052


namespace NUMINAMATH_CALUDE_only_set_B_forms_triangle_l1730_173042

-- Define a function to check if three lengths can form a triangle
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

-- Theorem statement
theorem only_set_B_forms_triangle :
  ¬(can_form_triangle 1 2 3) ∧
  can_form_triangle 2 3 4 ∧
  ¬(can_form_triangle 3 4 9) ∧
  ¬(can_form_triangle 2 2 4) :=
sorry

end NUMINAMATH_CALUDE_only_set_B_forms_triangle_l1730_173042


namespace NUMINAMATH_CALUDE_simon_stamps_l1730_173099

theorem simon_stamps (initial_stamps : ℕ) (friend1_stamps : ℕ) (friend2_stamps : ℕ) (friend3_stamps : ℕ) 
  (h1 : initial_stamps = 34)
  (h2 : friend1_stamps = 15)
  (h3 : friend2_stamps = 23)
  (h4 : initial_stamps + friend1_stamps + friend2_stamps + friend3_stamps = 61) :
  friend3_stamps = 23 ∧ friend1_stamps + friend2_stamps + friend3_stamps = 61 := by
  sorry

end NUMINAMATH_CALUDE_simon_stamps_l1730_173099


namespace NUMINAMATH_CALUDE_escalator_walking_speed_l1730_173043

theorem escalator_walking_speed 
  (escalator_speed : ℝ) 
  (escalator_length : ℝ) 
  (time_taken : ℝ) 
  (h1 : escalator_speed = 11) 
  (h2 : escalator_length = 140) 
  (h3 : time_taken = 10) : 
  ∃ (walking_speed : ℝ), 
    walking_speed = 3 ∧ 
    escalator_length = (walking_speed + escalator_speed) * time_taken := by
  sorry

end NUMINAMATH_CALUDE_escalator_walking_speed_l1730_173043


namespace NUMINAMATH_CALUDE_quadratic_equation_real_roots_l1730_173082

theorem quadratic_equation_real_roots (a b c : ℝ) : 
  ac < 0 → ∃ x : ℝ, a * x^2 + b * x + c = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_real_roots_l1730_173082


namespace NUMINAMATH_CALUDE_free_time_correct_l1730_173066

/-- The time required to free Hannah's younger son -/
def free_time (total_strands : ℕ) (hannah_rate : ℕ) (son_rate : ℕ) : ℕ :=
  (total_strands + (hannah_rate + son_rate) - 1) / (hannah_rate + son_rate)

theorem free_time_correct : free_time 78 5 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_free_time_correct_l1730_173066


namespace NUMINAMATH_CALUDE_two_real_roots_implies_m_geq_one_l1730_173056

theorem two_real_roots_implies_m_geq_one (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ (x + 5)^2 = m - 1 ∧ (y + 5)^2 = m - 1) →
  m ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_two_real_roots_implies_m_geq_one_l1730_173056


namespace NUMINAMATH_CALUDE_find_a_l1730_173071

-- Define the points A and B
def A (a : ℝ) : ℝ × ℝ := (-1, a)
def B (a : ℝ) : ℝ × ℝ := (a, 8)

-- Define the slope of the line 2x - y + 1 = 0
def slope_given_line : ℝ := 2

-- Define the theorem
theorem find_a : ∃ a : ℝ, 
  (B a).2 - (A a).2 = slope_given_line * ((B a).1 - (A a).1) :=
sorry

-- Note: (p.1) and (p.2) represent the x and y coordinates of a point p respectively

end NUMINAMATH_CALUDE_find_a_l1730_173071


namespace NUMINAMATH_CALUDE_x_range_l1730_173068

theorem x_range (x : ℝ) (h1 : x^2 - 2*x - 3 < 0) (h2 : 1/(x-2) < 0) : -1 < x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_x_range_l1730_173068


namespace NUMINAMATH_CALUDE_max_removable_edges_l1730_173022

/-- Represents a volleyball net grid with internal divisions -/
structure VolleyballNet where
  rows : Nat
  cols : Nat
  internalDivisions : Nat

/-- Calculates the total number of nodes in the volleyball net -/
def totalNodes (net : VolleyballNet) : Nat :=
  (net.rows + 1) * (net.cols + 1) + net.rows * net.cols

/-- Calculates the total number of edges in the volleyball net -/
def totalEdges (net : VolleyballNet) : Nat :=
  net.rows * (net.cols + 1) + net.cols * (net.rows + 1) + net.internalDivisions * net.rows * net.cols

/-- Theorem stating the maximum number of removable edges -/
theorem max_removable_edges (net : VolleyballNet) :
  net.rows = 10 → net.cols = 20 → net.internalDivisions = 4 →
  totalEdges net - (totalNodes net - 1) = 800 := by
  sorry


end NUMINAMATH_CALUDE_max_removable_edges_l1730_173022


namespace NUMINAMATH_CALUDE_f_even_and_decreasing_l1730_173021

def f (x : ℝ) := -x^2

theorem f_even_and_decreasing :
  (∀ x : ℝ, f x = f (-x)) ∧
  (∀ x y : ℝ, 0 < x → x < y → f y < f x) :=
by sorry

end NUMINAMATH_CALUDE_f_even_and_decreasing_l1730_173021


namespace NUMINAMATH_CALUDE_peter_chip_cost_l1730_173032

/-- Calculates the cost to consume a given number of calories from chips, given the calorie content per chip, chips per bag, and cost per bag. -/
def cost_for_calories (calories_per_chip : ℕ) (chips_per_bag : ℕ) (cost_per_bag : ℚ) (target_calories : ℕ) : ℚ :=
  let calories_per_bag := calories_per_chip * chips_per_bag
  let bags_needed := (target_calories + calories_per_bag - 1) / calories_per_bag
  bags_needed * cost_per_bag

/-- Theorem stating that Peter needs to spend $4 to consume 480 calories of chips. -/
theorem peter_chip_cost : cost_for_calories 10 24 2 480 = 4 := by
  sorry

end NUMINAMATH_CALUDE_peter_chip_cost_l1730_173032


namespace NUMINAMATH_CALUDE_triangle_longer_segment_l1730_173084

theorem triangle_longer_segment (a b c h x : ℝ) : 
  a = 40 → b = 90 → c = 100 → 
  x^2 + h^2 = a^2 → 
  (c - x)^2 + h^2 = b^2 → 
  c - x = 82.5 :=
by sorry

end NUMINAMATH_CALUDE_triangle_longer_segment_l1730_173084


namespace NUMINAMATH_CALUDE_unique_m_value_l1730_173053

theorem unique_m_value : ∃! m : ℝ, (abs m = 1) ∧ (m - 1 ≠ 0) := by sorry

end NUMINAMATH_CALUDE_unique_m_value_l1730_173053


namespace NUMINAMATH_CALUDE_h_one_value_l1730_173012

/-- A polynomial of degree 3 with constant coefficients -/
structure CubicPolynomial where
  p : ℝ
  q : ℝ
  r : ℝ
  h_order : p < q ∧ q < r

/-- The function f(x) = x^3 + px^2 + qx + r -/
def f (c : CubicPolynomial) (x : ℝ) : ℝ :=
  x^3 + c.p * x^2 + c.q * x + c.r

/-- A polynomial h(x) whose roots are the squares of the reciprocals of the roots of f(x) -/
def h (c : CubicPolynomial) (x : ℝ) : ℝ :=
  sorry  -- Definition of h(x) is not explicitly given in the problem

/-- Theorem stating the value of h(1) in terms of p, q, and r -/
theorem h_one_value (c : CubicPolynomial) :
  h c 1 = (1 - c.p + c.q - c.r) * (1 - c.q + c.p - c.r) * (1 - c.r + c.p - c.q) / c.r^2 :=
sorry

end NUMINAMATH_CALUDE_h_one_value_l1730_173012


namespace NUMINAMATH_CALUDE_pure_imaginary_product_l1730_173083

def complex_number (a b : ℝ) := a + b * Complex.I

theorem pure_imaginary_product (m : ℝ) :
  (complex_number 1 m * complex_number 2 (-1)).re = 0 →
  m = -2 :=
by sorry

end NUMINAMATH_CALUDE_pure_imaginary_product_l1730_173083


namespace NUMINAMATH_CALUDE_jack_payback_l1730_173063

/-- The amount borrowed by Jack -/
def principal : ℝ := 1200

/-- The interest rate as a decimal -/
def interestRate : ℝ := 0.1

/-- The total amount Jack will pay back -/
def totalAmount : ℝ := principal * (1 + interestRate)

/-- Theorem stating that the total amount Jack will pay back is $1320 -/
theorem jack_payback : totalAmount = 1320 := by
  sorry

end NUMINAMATH_CALUDE_jack_payback_l1730_173063


namespace NUMINAMATH_CALUDE_rectangle_minimum_width_l1730_173055

/-- A rectangle with length 1.5 times its width and area at least 450 square feet has a minimum width of 10√3 feet. -/
theorem rectangle_minimum_width (w : ℝ) (h_positive : w > 0) : 
  1.5 * w * w ≥ 450 → w ≥ 10 * Real.sqrt 3 :=
by
  sorry

#check rectangle_minimum_width

end NUMINAMATH_CALUDE_rectangle_minimum_width_l1730_173055


namespace NUMINAMATH_CALUDE_factorial_fraction_equals_four_l1730_173098

theorem factorial_fraction_equals_four :
  (4 * Nat.factorial 6 + 20 * Nat.factorial 5) / Nat.factorial 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_factorial_fraction_equals_four_l1730_173098


namespace NUMINAMATH_CALUDE_system_solution_l1730_173093

theorem system_solution (x y z t : ℤ) : 
  (x * y + z * t = 1 ∧ 
   x * z + y * t = 1 ∧ 
   x * t + y * z = 1) ↔ 
  ((x, y, z, t) = (0, 1, 1, 1) ∨
   (x, y, z, t) = (1, 0, 1, 1) ∨
   (x, y, z, t) = (1, 1, 0, 1) ∨
   (x, y, z, t) = (1, 1, 1, 0) ∨
   (x, y, z, t) = (0, -1, -1, -1) ∨
   (x, y, z, t) = (-1, 0, -1, -1) ∨
   (x, y, z, t) = (-1, -1, 0, -1) ∨
   (x, y, z, t) = (-1, -1, -1, 0)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1730_173093


namespace NUMINAMATH_CALUDE_last_three_digits_of_7_to_80_l1730_173061

theorem last_three_digits_of_7_to_80 : 7^80 ≡ 961 [ZMOD 1000] := by sorry

end NUMINAMATH_CALUDE_last_three_digits_of_7_to_80_l1730_173061


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l1730_173060

theorem fraction_sum_equality (a b : ℝ) (ha : a ≠ 0) :
  1 / (2 * a * b) + b / (4 * a) = (2 + b^2) / (4 * a * b) := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l1730_173060


namespace NUMINAMATH_CALUDE_opposite_seven_eighteen_implies_twentytwo_l1730_173041

/-- Represents a circular arrangement of people -/
structure CircularArrangement where
  total : ℕ
  is_valid : total > 0

/-- Defines the property of two positions being opposite in a circular arrangement -/
def are_opposite (c : CircularArrangement) (p1 p2 : ℕ) : Prop :=
  p1 ≤ c.total ∧ p2 ≤ c.total ∧ (2 * p1 - 1) % c.total = (2 * p2 - 1) % c.total

/-- Theorem: In a circular arrangement where the 7th person is opposite the 18th, there are 22 people -/
theorem opposite_seven_eighteen_implies_twentytwo :
  ∀ c : CircularArrangement, are_opposite c 7 18 → c.total = 22 :=
sorry

end NUMINAMATH_CALUDE_opposite_seven_eighteen_implies_twentytwo_l1730_173041


namespace NUMINAMATH_CALUDE_investment_calculation_l1730_173094

theorem investment_calculation (total : ℝ) (ratio : ℝ) (mutual_funds : ℝ) (bonds : ℝ) :
  total = 240000 ∧ 
  mutual_funds = ratio * bonds ∧ 
  ratio = 6 ∧ 
  total = mutual_funds + bonds →
  mutual_funds = 205714.29 := by
  sorry

end NUMINAMATH_CALUDE_investment_calculation_l1730_173094


namespace NUMINAMATH_CALUDE_speed_difference_l1730_173049

/-- Given distances and times for cycling and walking, prove the speed difference --/
theorem speed_difference (school_distance : ℝ) (cycle_time : ℝ) 
  (park_distance : ℝ) (walk_time : ℝ) 
  (h1 : school_distance = 9.3) 
  (h2 : cycle_time = 0.6)
  (h3 : park_distance = 0.9)
  (h4 : walk_time = 0.2) :
  (school_distance / cycle_time) - (park_distance / walk_time) = 11 := by
  sorry


end NUMINAMATH_CALUDE_speed_difference_l1730_173049


namespace NUMINAMATH_CALUDE_weavers_in_first_group_l1730_173025

/-- The number of weavers in the first group -/
def first_group_weavers : ℕ := 4

/-- The number of mats woven by the first group -/
def first_group_mats : ℕ := 4

/-- The number of days taken by the first group -/
def first_group_days : ℕ := 4

/-- The number of weavers in the second group -/
def second_group_weavers : ℕ := 12

/-- The number of mats woven by the second group -/
def second_group_mats : ℕ := 36

/-- The number of days taken by the second group -/
def second_group_days : ℕ := 12

/-- Theorem stating that the number of weavers in the first group is 4 -/
theorem weavers_in_first_group :
  first_group_weavers = 4 :=
by sorry

end NUMINAMATH_CALUDE_weavers_in_first_group_l1730_173025


namespace NUMINAMATH_CALUDE_subtracted_value_l1730_173086

theorem subtracted_value (chosen_number : ℕ) (x : ℚ) : 
  chosen_number = 120 → (chosen_number / 6 : ℚ) - x = 5 → x = 15 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_value_l1730_173086


namespace NUMINAMATH_CALUDE_laptop_final_price_l1730_173006

/-- Calculate the final price of a laptop given the original price, discount rate, tax rate, and commission rate. -/
def calculate_final_price (original_price discount_rate tax_rate commission_rate : ℝ) : ℝ :=
  let discounted_price := original_price * (1 - discount_rate)
  let price_after_tax := discounted_price * (1 + tax_rate)
  price_after_tax * (1 + commission_rate)

/-- Theorem stating that the final price of the laptop is 1199.52 dollars given the specified conditions. -/
theorem laptop_final_price :
  calculate_final_price 1200 0.15 0.12 0.05 = 1199.52 := by
  sorry

end NUMINAMATH_CALUDE_laptop_final_price_l1730_173006


namespace NUMINAMATH_CALUDE_min_value_of_f_l1730_173019

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := ∫ t, (2 * t - 4)

-- State the theorem
theorem min_value_of_f :
  ∃ (min : ℝ), min = -4 ∧ ∀ x ∈ Set.Icc (-1) 3, f x ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l1730_173019


namespace NUMINAMATH_CALUDE_point_Q_in_second_quadrant_l1730_173070

theorem point_Q_in_second_quadrant (a : ℝ) : 
  a < 0 → -a^2 - 1 < 0 ∧ -a + 1 > 0 := by
  sorry

end NUMINAMATH_CALUDE_point_Q_in_second_quadrant_l1730_173070


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_neg_one_a_range_when_f_bounded_l1730_173050

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| - |x - 4|

-- Theorem for part I
theorem solution_set_when_a_is_neg_one :
  {x : ℝ | f (-1) x ≥ 4} = {x : ℝ | x ≥ 7/2} := by sorry

-- Theorem for part II
theorem a_range_when_f_bounded :
  (∀ x : ℝ, |f a x| ≤ 2) → a ∈ Set.Icc 2 6 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_neg_one_a_range_when_f_bounded_l1730_173050


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l1730_173075

-- Problem 1
theorem problem_1 : -3 + (-2) * 5 - (-3) = -10 := by sorry

-- Problem 2
theorem problem_2 : -1^4 + ((-5)^2 - 3) / |(-2)| = 10 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l1730_173075


namespace NUMINAMATH_CALUDE_small_bottle_price_theorem_l1730_173073

/-- The price of a small bottle that results in the given average price -/
def price_small_bottle (large_quantity : ℕ) (small_quantity : ℕ) (large_price : ℚ) (average_price : ℚ) : ℚ :=
  ((large_quantity + small_quantity : ℚ) * average_price - large_quantity * large_price) / small_quantity

theorem small_bottle_price_theorem (large_quantity small_quantity : ℕ) (large_price average_price : ℚ) :
  large_quantity = 1365 →
  small_quantity = 720 →
  large_price = 189/100 →
  average_price = 173/100 →
  ∃ ε > 0, |price_small_bottle large_quantity small_quantity large_price average_price - 142/100| < ε :=
by sorry


end NUMINAMATH_CALUDE_small_bottle_price_theorem_l1730_173073


namespace NUMINAMATH_CALUDE_curves_intersection_l1730_173057

/-- The first curve equation -/
def curve1 (x y : ℝ) : Prop :=
  2 * x^2 + 3 * x * y - 2 * y^2 - 6 * x + 3 * y = 0

/-- The second curve equation -/
def curve2 (x y : ℝ) : Prop :=
  3 * x^2 + 7 * x * y + 2 * y^2 - 7 * x + y - 6 = 0

/-- The set of intersection points -/
def intersection_points : Set (ℝ × ℝ) :=
  {(-1, 2), (1, 1), (0, 3/2), (3, 0), (4, -1/2), (5, -1)}

/-- Theorem stating that the given points are the intersection points of the two curves -/
theorem curves_intersection :
  ∀ (p : ℝ × ℝ), p ∈ intersection_points ↔ (curve1 p.1 p.2 ∧ curve2 p.1 p.2) :=
sorry

end NUMINAMATH_CALUDE_curves_intersection_l1730_173057


namespace NUMINAMATH_CALUDE_det_special_matrix_l1730_173048

/-- The determinant of the matrix [[1, a, b], [1, a+b, b+c], [1, a, a+c]] is ab + b^2 + bc -/
theorem det_special_matrix (a b c : ℝ) : 
  Matrix.det ![![1, a, b], ![1, a+b, b+c], ![1, a, a+c]] = a*b + b^2 + b*c := by
  sorry

end NUMINAMATH_CALUDE_det_special_matrix_l1730_173048


namespace NUMINAMATH_CALUDE_volume_rotational_ellipsoid_l1730_173044

/-- The volume of a rotational ellipsoid -/
theorem volume_rotational_ellipsoid (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (∫ y in (-b)..b, π * a^2 * (1 - y^2 / b^2)) = (4 / 3) * π * a^2 * b :=
by sorry

end NUMINAMATH_CALUDE_volume_rotational_ellipsoid_l1730_173044


namespace NUMINAMATH_CALUDE_f_is_even_and_decreasing_l1730_173079

def f (x : ℝ) : ℝ := -x^2 + abs x

theorem f_is_even_and_decreasing :
  (∀ x, f x = f (-x)) ∧
  (∀ x y, 1 < x ∧ x < y → f y < f x) :=
by sorry

end NUMINAMATH_CALUDE_f_is_even_and_decreasing_l1730_173079


namespace NUMINAMATH_CALUDE_max_value_of_a_l1730_173016

/-- Given that "x^2 + 2x - 3 > 0" is a necessary but not sufficient condition for "x < a",
    prove that the maximum value of a is -3. -/
theorem max_value_of_a (a : ℝ) : 
  (∀ x : ℝ, x < a → x^2 + 2*x - 3 > 0) ∧ 
  (∃ x : ℝ, x^2 + 2*x - 3 > 0 ∧ x ≥ a) →
  a ≤ -3 ∧ ∀ b : ℝ, b > -3 → ¬((∀ x : ℝ, x < b → x^2 + 2*x - 3 > 0) ∧ 
                               (∃ x : ℝ, x^2 + 2*x - 3 > 0 ∧ x ≥ b)) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_a_l1730_173016


namespace NUMINAMATH_CALUDE_quadratic_equation_no_real_roots_l1730_173085

theorem quadratic_equation_no_real_roots 
  (a b c : ℝ) 
  (h : a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) : 
  ∀ x : ℝ, x^2 + (a + b + c) * x + a^2 + b^2 + c^2 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_no_real_roots_l1730_173085


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l1730_173026

-- Problem 1
theorem problem_1 (a b c : ℝ) : 
  (-8 * a^4 * b^5 * c / (4 * a * b^5)) * (3 * a^3 * b^2) = -6 * a^6 * b^2 * c :=
sorry

-- Problem 2
theorem problem_2 (a : ℝ) :
  (2*a + 1)^2 - (2*a + 1)*(2*a - 1) = 4*a + 2 :=
sorry

-- Problem 3
theorem problem_3 (x y : ℝ) :
  (x - y - 2) * (x - y + 2) - (x + 2*y) * (x - 3*y) = 7*y^2 - x*y - 4 :=
sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l1730_173026

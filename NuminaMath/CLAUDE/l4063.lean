import Mathlib

namespace NUMINAMATH_CALUDE_log_equation_sum_l4063_406345

theorem log_equation_sum (a b : ℤ) (h : a * Real.log 2 / Real.log 250 + b * Real.log 5 / Real.log 250 = 3) : 
  a + 2 * b = 21 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_sum_l4063_406345


namespace NUMINAMATH_CALUDE_greene_nursery_yellow_carnations_l4063_406306

theorem greene_nursery_yellow_carnations :
  let total_flowers : ℕ := 6284
  let red_roses : ℕ := 1491
  let white_roses : ℕ := 1768
  let yellow_carnations : ℕ := total_flowers - (red_roses + white_roses)
  yellow_carnations = 3025 := by
  sorry

end NUMINAMATH_CALUDE_greene_nursery_yellow_carnations_l4063_406306


namespace NUMINAMATH_CALUDE_max_guaranteed_guesses_l4063_406360

/- Define the deck of cards -/
def deck_size : Nat := 52

/- Define a function to represent the alternating arrangement of cards -/
def alternating_arrangement (i : Nat) : Bool :=
  i % 2 = 0

/- Define a function to represent the state of the deck after cutting and riffling -/
def riffled_deck (n : Nat) (i : Nat) : Bool :=
  if i < n then alternating_arrangement i else alternating_arrangement (i - n)

/- Theorem: The maximum number of guaranteed correct guesses is 26 -/
theorem max_guaranteed_guesses :
  ∀ n : Nat, n ≤ deck_size →
  ∃ strategy : Nat → Bool,
    (∀ i : Nat, i < deck_size → strategy i = riffled_deck n i) →
    (∃ correct_guesses : Nat, correct_guesses = deck_size / 2 ∧
      ∀ k : Nat, k < correct_guesses → strategy k = riffled_deck n k) :=
by sorry

end NUMINAMATH_CALUDE_max_guaranteed_guesses_l4063_406360


namespace NUMINAMATH_CALUDE_toaster_cost_is_72_l4063_406376

/-- Calculates the total cost of a toaster including insurance, fees, and taxes. -/
def toaster_total_cost (msrp : ℝ) (insurance_rate : ℝ) (premium_upgrade : ℝ) 
  (recycling_fee : ℝ) (tax_rate : ℝ) : ℝ :=
  let insurance_cost := msrp * insurance_rate
  let total_insurance := insurance_cost + premium_upgrade
  let cost_before_tax := msrp + total_insurance + recycling_fee
  let tax := cost_before_tax * tax_rate
  cost_before_tax + tax

/-- Theorem stating that the total cost of the toaster is $72 given the specified conditions. -/
theorem toaster_cost_is_72 : 
  toaster_total_cost 30 0.2 7 5 0.5 = 72 := by
  sorry

end NUMINAMATH_CALUDE_toaster_cost_is_72_l4063_406376


namespace NUMINAMATH_CALUDE_x_gt_3_sufficient_not_necessary_l4063_406354

theorem x_gt_3_sufficient_not_necessary :
  (∃ x : ℝ, x ≤ 3 ∧ 1 / x < 1 / 3) ∧
  (∀ x : ℝ, x > 3 → 1 / x < 1 / 3) :=
by sorry

end NUMINAMATH_CALUDE_x_gt_3_sufficient_not_necessary_l4063_406354


namespace NUMINAMATH_CALUDE_triangle_reconstruction_uniqueness_l4063_406334

/-- A triangle in a 2D plane --/
structure Triangle :=
  (A B C : EuclideanSpace ℝ (Fin 2))

/-- The circumcenter of a triangle --/
def circumcenter (t : Triangle) : EuclideanSpace ℝ (Fin 2) :=
  sorry

/-- Reflect a point across a line defined by two points --/
def reflect_point (p q r : EuclideanSpace ℝ (Fin 2)) : EuclideanSpace ℝ (Fin 2) :=
  sorry

/-- Given three points that are reflections of a fourth point across the sides of a triangle,
    reconstruct the original triangle --/
def reconstruct_triangle (O1 O2 O3 : EuclideanSpace ℝ (Fin 2)) : Triangle :=
  sorry

theorem triangle_reconstruction_uniqueness 
  (t : Triangle) 
  (O : EuclideanSpace ℝ (Fin 2)) 
  (hO : O = circumcenter t) 
  (O1 : EuclideanSpace ℝ (Fin 2)) 
  (hO1 : O1 = reflect_point O t.B t.C) 
  (O2 : EuclideanSpace ℝ (Fin 2)) 
  (hO2 : O2 = reflect_point O t.C t.A) 
  (O3 : EuclideanSpace ℝ (Fin 2)) 
  (hO3 : O3 = reflect_point O t.A t.B) :
  reconstruct_triangle O1 O2 O3 = t :=
sorry

end NUMINAMATH_CALUDE_triangle_reconstruction_uniqueness_l4063_406334


namespace NUMINAMATH_CALUDE_complement_intersection_equals_four_l4063_406373

-- Define the universe
def U : Set Nat := {1, 2, 3, 4, 5}

-- Define sets M and N
def M : Set Nat := {1, 3, 5}
def N : Set Nat := {3, 4, 5}

-- State the theorem
theorem complement_intersection_equals_four :
  (U \ M) ∩ N = {4} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_equals_four_l4063_406373


namespace NUMINAMATH_CALUDE_triple_composition_identity_implies_identity_l4063_406303

theorem triple_composition_identity_implies_identity 
  (f : ℝ → ℝ) (hf : Continuous f) (h : ∀ x, f (f (f x)) = x) : 
  ∀ x, f x = x := by
  sorry

end NUMINAMATH_CALUDE_triple_composition_identity_implies_identity_l4063_406303


namespace NUMINAMATH_CALUDE_ones_digit_of_power_l4063_406330

theorem ones_digit_of_power (x : ℕ) : (2^3)^x = 4096 → (3^(x^3)) % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ones_digit_of_power_l4063_406330


namespace NUMINAMATH_CALUDE_hyperbola_equation_correct_l4063_406314

/-- Represents a hyperbola with given asymptotes and passing through a specific point -/
def is_correct_hyperbola (a b : ℝ) : Prop :=
  -- The equation of the hyperbola
  (∀ x y : ℝ, (3 * y^2 / 4) - (x^2 / 3) = 1 ↔ b * y^2 - a * x^2 = a * b) ∧
  -- The asymptotes are y = ±(2/3)x
  (a / b = 3 / 2) ∧
  -- The hyperbola passes through the point (√6, 2)
  (3 * 2^2 / 4 - Real.sqrt 6^2 / 3 = 1)

/-- The standard equation of the hyperbola satisfies the given conditions -/
theorem hyperbola_equation_correct :
  ∃ a b : ℝ, is_correct_hyperbola a b :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_correct_l4063_406314


namespace NUMINAMATH_CALUDE_sqrt_negative_eight_a_cubed_l4063_406382

theorem sqrt_negative_eight_a_cubed (a : ℝ) (h : a ≤ 0) :
  Real.sqrt (-8 * a^3) = -2 * a * Real.sqrt (-2 * a) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_negative_eight_a_cubed_l4063_406382


namespace NUMINAMATH_CALUDE_sum_of_absolute_b_values_l4063_406351

-- Define the polynomials p and q
def p (a b x : ℝ) : ℝ := x^3 + a*x + b
def q (a b x : ℝ) : ℝ := x^3 + a*x + b + 240

-- Define the theorem
theorem sum_of_absolute_b_values (a b r s : ℝ) : 
  (p a b r = 0) → 
  (p a b s = 0) → 
  (q a b (r + 4) = 0) → 
  (q a b (s - 3) = 0) → 
  (∃ b₁ b₂ : ℝ, (b = b₁ ∨ b = b₂) ∧ (|b₁| + |b₂| = 62)) := by
sorry


end NUMINAMATH_CALUDE_sum_of_absolute_b_values_l4063_406351


namespace NUMINAMATH_CALUDE_ones_digit_73_pow_351_l4063_406342

theorem ones_digit_73_pow_351 : (73^351) % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_ones_digit_73_pow_351_l4063_406342


namespace NUMINAMATH_CALUDE_locus_is_circumcircle_l4063_406358

/-- Triangle represented by its vertices -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Distance from a point to a line segment -/
def distToSide (P : Point) (A B : Point) : ℝ := sorry

/-- Distance between two points -/
def dist (P Q : Point) : ℝ := sorry

/-- Circumcircle of a triangle -/
def circumcircle (t : Triangle) : Set Point := sorry

/-- A point lies on the circumcircle of a triangle -/
def onCircumcircle (P : Point) (t : Triangle) : Prop :=
  P ∈ circumcircle t

theorem locus_is_circumcircle (t : Triangle) (P : Point) :
  (distToSide P t.A t.B * dist P t.C = distToSide P t.A t.C * dist P t.B) →
  onCircumcircle P t := by
  sorry

end NUMINAMATH_CALUDE_locus_is_circumcircle_l4063_406358


namespace NUMINAMATH_CALUDE_parallel_iff_a_eq_neg_one_l4063_406326

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := x - y - 1 = 0
def l₂ (a x y : ℝ) : Prop := x + a*y - 2 = 0

-- Define parallelism for these lines
def parallel (a : ℝ) : Prop := ∀ x y, l₁ x y ↔ ∃ k, l₂ a (x + k) (y + k)

-- State the theorem
theorem parallel_iff_a_eq_neg_one :
  ∀ a : ℝ, parallel a ↔ a = -1 := by sorry

end NUMINAMATH_CALUDE_parallel_iff_a_eq_neg_one_l4063_406326


namespace NUMINAMATH_CALUDE_car_stop_once_probability_l4063_406337

/-- The probability of a car stopping once at three traffic lights. -/
theorem car_stop_once_probability
  (pA pB pC : ℝ)
  (hA : pA = 1/3)
  (hB : pB = 1/2)
  (hC : pC = 2/3)
  : (1 - pA) * pB * pC + pA * (1 - pB) * pC + pA * pB * (1 - pC) = 7/18 := by
  sorry

end NUMINAMATH_CALUDE_car_stop_once_probability_l4063_406337


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l4063_406362

theorem quadratic_equation_solution : 
  ∀ x : ℝ, x^2 + 2*x = 0 ↔ x = 0 ∨ x = -2 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l4063_406362


namespace NUMINAMATH_CALUDE_farm_plot_length_l4063_406392

/-- Proves that a rectangular plot with given width and area has a specific length -/
theorem farm_plot_length (width : ℝ) (area_acres : ℝ) (acre_sq_ft : ℝ) :
  width = 1210 →
  area_acres = 10 →
  acre_sq_ft = 43560 →
  (area_acres * acre_sq_ft) / width = 360 := by
  sorry

end NUMINAMATH_CALUDE_farm_plot_length_l4063_406392


namespace NUMINAMATH_CALUDE_percent_of_12356_l4063_406385

theorem percent_of_12356 (p : ℝ) : p * 12356 = 1.2356 → p * 100 = 0.01 := by
  sorry

end NUMINAMATH_CALUDE_percent_of_12356_l4063_406385


namespace NUMINAMATH_CALUDE_min_value_of_sum_l4063_406340

theorem min_value_of_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 1 / (a + 3) + 1 / (b + 3) = 1 / 4) : 
  a + 3 * b ≥ 4 + 8 * Real.sqrt 3 ∧ 
  (a + 3 * b = 4 + 8 * Real.sqrt 3 ↔ a = 1 + 4 * Real.sqrt 3 ∧ b = (3 + 4 * Real.sqrt 3) / 3) := by
sorry

end NUMINAMATH_CALUDE_min_value_of_sum_l4063_406340


namespace NUMINAMATH_CALUDE_equipment_cannot_fit_l4063_406346

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The L-shaped corridor width -/
def corridorWidth : ℝ := 3

/-- The center of the unit circle representing the corner of the L-shaped corridor -/
def circleCenter : Point := ⟨corridorWidth, corridorWidth⟩

/-- The radius of the circle representing the corner of the L-shaped corridor -/
def circleRadius : ℝ := 1

/-- The origin point -/
def origin : Point := ⟨0, 0⟩

/-- The maximum length of the equipment -/
def maxEquipmentLength : ℝ := 7

/-- A line passing through the origin -/
structure Line where
  a : ℝ
  b : ℝ

/-- The length of the line segment from the origin to its intersection with the circle -/
def lineSegmentLength (l : Line) : ℝ := sorry

/-- The minimum length of a line segment intersecting the circle and passing through the origin -/
def minLineSegmentLength : ℝ := sorry

/-- Theorem stating that the equipment cannot fit through the L-shaped corridor -/
theorem equipment_cannot_fit : minLineSegmentLength > maxEquipmentLength := by sorry

end NUMINAMATH_CALUDE_equipment_cannot_fit_l4063_406346


namespace NUMINAMATH_CALUDE_percent_difference_l4063_406370

theorem percent_difference (p q : ℝ) (h : p = 1.5 * q) : 
  (q / p) = 2/3 ∧ ((p - q) / q) = 1/2 := by sorry

end NUMINAMATH_CALUDE_percent_difference_l4063_406370


namespace NUMINAMATH_CALUDE_equation_solution_l4063_406304

theorem equation_solution : ∃ x : ℝ, 45 * x = 0.6 * 900 ∧ x = 12 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l4063_406304


namespace NUMINAMATH_CALUDE_existence_of_three_numbers_with_same_product_last_digit_l4063_406308

-- Define a function to get the last digit of a natural number
def lastDigit (n : ℕ) : ℕ := n % 10

-- Define the theorem
theorem existence_of_three_numbers_with_same_product_last_digit :
  ∃ (a b c : ℕ), 
    (lastDigit a ≠ lastDigit b) ∧ 
    (lastDigit b ≠ lastDigit c) ∧ 
    (lastDigit a ≠ lastDigit c) ∧
    (lastDigit (a * b) = lastDigit (b * c)) ∧
    (lastDigit (b * c) = lastDigit (a * c)) :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_existence_of_three_numbers_with_same_product_last_digit_l4063_406308


namespace NUMINAMATH_CALUDE_ramesh_refrigerator_cost_l4063_406369

theorem ramesh_refrigerator_cost 
  (P : ℝ)  -- Labeled price
  (discount_rate : ℝ)  -- Discount rate
  (transport_cost : ℝ)  -- Transport cost
  (installation_cost : ℝ)  -- Installation cost
  (profit_rate : ℝ)  -- Profit rate
  (selling_price : ℝ)  -- Selling price for profit
  (h1 : discount_rate = 0.2)
  (h2 : transport_cost = 125)
  (h3 : installation_cost = 250)
  (h4 : profit_rate = 0.18)
  (h5 : selling_price = 18880)
  (h6 : selling_price = P * (1 + profit_rate)) :
  P * (1 - discount_rate) + transport_cost + installation_cost = 13175 :=
by sorry

end NUMINAMATH_CALUDE_ramesh_refrigerator_cost_l4063_406369


namespace NUMINAMATH_CALUDE_strength_increase_percentage_l4063_406395

/-- Calculates the percentage increase in strength due to a magical bracer --/
theorem strength_increase_percentage 
  (original_weight : ℝ) 
  (training_increase : ℝ) 
  (final_weight : ℝ) : 
  original_weight = 135 →
  training_increase = 265 →
  final_weight = 2800 →
  ((final_weight - (original_weight + training_increase)) / (original_weight + training_increase)) * 100 = 600 := by
  sorry

end NUMINAMATH_CALUDE_strength_increase_percentage_l4063_406395


namespace NUMINAMATH_CALUDE_christmas_games_l4063_406394

theorem christmas_games (C B : ℕ) (h1 : B = 8) (h2 : C + B + (C + B) / 2 = 30) : C = 12 := by
  sorry

end NUMINAMATH_CALUDE_christmas_games_l4063_406394


namespace NUMINAMATH_CALUDE_smallest_sum_B_plus_b_l4063_406348

theorem smallest_sum_B_plus_b :
  ∀ (B b : ℕ),
  0 ≤ B ∧ B < 5 →
  b > 6 →
  31 * B = 4 * b + 4 →
  ∀ (B' b' : ℕ),
  0 ≤ B' ∧ B' < 5 →
  b' > 6 →
  31 * B' = 4 * b' + 4 →
  B + b ≤ B' + b' :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_B_plus_b_l4063_406348


namespace NUMINAMATH_CALUDE_max_distance_between_inscribed_squares_max_distance_is_5_sqrt_2_l4063_406300

/-- The maximum distance between vertices of two squares, where the smaller square
    (perimeter 24) is inscribed in the larger square (perimeter 32) and rotated such that
    one of its vertices lies on the midpoint of one side of the larger square. -/
theorem max_distance_between_inscribed_squares : ℝ :=
  let inner_perimeter : ℝ := 24
  let outer_perimeter : ℝ := 32
  let inner_side : ℝ := inner_perimeter / 4
  let outer_side : ℝ := outer_perimeter / 4
  5 * Real.sqrt 2

/-- Proof that the maximum distance between vertices of the inscribed squares is 5√2. -/
theorem max_distance_is_5_sqrt_2 (inner_perimeter outer_perimeter : ℝ)
  (h1 : inner_perimeter = 24)
  (h2 : outer_perimeter = 32)
  (h3 : ∃ (v : ℝ × ℝ), v.1 = outer_perimeter / 8 ∧ v.2 = 0) :
  max_distance_between_inscribed_squares = 5 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_CALUDE_max_distance_between_inscribed_squares_max_distance_is_5_sqrt_2_l4063_406300


namespace NUMINAMATH_CALUDE_square_and_add_l4063_406377

theorem square_and_add (x : ℝ) (h : x = 5) : 2 * x^2 + 5 = 55 := by
  sorry

end NUMINAMATH_CALUDE_square_and_add_l4063_406377


namespace NUMINAMATH_CALUDE_complex_on_imaginary_axis_l4063_406396

theorem complex_on_imaginary_axis (a : ℝ) : 
  let z : ℂ := (1 + Complex.I) * (1 + a * Complex.I)
  (z.re = 0) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_on_imaginary_axis_l4063_406396


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l4063_406352

theorem pure_imaginary_condition (a : ℝ) : 
  (Complex.I : ℂ).im ≠ 0 →
  (Complex.ofReal (a - 1) * Complex.ofReal (a + 1) + Complex.I) = 
    Complex.ofReal (a^2 - 1) + Complex.I * Complex.ofReal (a - 1) →
  (Complex.ofReal (a - 1) * Complex.ofReal (a + 1) + Complex.I).re = 0 →
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l4063_406352


namespace NUMINAMATH_CALUDE_probability_specific_selection_l4063_406317

/-- The number of shirts in the drawer -/
def num_shirts : ℕ := 3

/-- The number of pairs of shorts in the drawer -/
def num_shorts : ℕ := 6

/-- The number of pairs of socks in the drawer -/
def num_socks : ℕ := 7

/-- The total number of articles of clothing in the drawer -/
def total_articles : ℕ := num_shirts + num_shorts + num_socks

/-- The number of articles to be selected -/
def num_selected : ℕ := 4

/-- The probability of selecting one shirt, two pairs of shorts, and one pair of socks -/
theorem probability_specific_selection :
  (Nat.choose num_shirts 1 * Nat.choose num_shorts 2 * Nat.choose num_socks 1) /
  (Nat.choose total_articles num_selected) = 63 / 364 :=
sorry

end NUMINAMATH_CALUDE_probability_specific_selection_l4063_406317


namespace NUMINAMATH_CALUDE_f_one_zero_iff_l4063_406399

/-- A function f(x) = ax^2 - x - 1 where a is a real number -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - x - 1

/-- The property that f has exactly one zero -/
def has_exactly_one_zero (a : ℝ) : Prop :=
  ∃! x, f a x = 0

/-- Theorem stating that f has exactly one zero iff a = 0 or a = -1/4 -/
theorem f_one_zero_iff (a : ℝ) :
  has_exactly_one_zero a ↔ a = 0 ∨ a = -1/4 := by sorry

end NUMINAMATH_CALUDE_f_one_zero_iff_l4063_406399


namespace NUMINAMATH_CALUDE_words_with_vowels_count_l4063_406320

def alphabet : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'I'}
def vowels : Finset Char := {'A', 'E', 'I'}
def consonants : Finset Char := alphabet \ vowels

def word_length : Nat := 5

def total_words : Nat := alphabet.card ^ word_length
def words_without_vowels : Nat := consonants.card ^ word_length

theorem words_with_vowels_count :
  total_words - words_without_vowels = 29643 :=
sorry

end NUMINAMATH_CALUDE_words_with_vowels_count_l4063_406320


namespace NUMINAMATH_CALUDE_min_m_value_l4063_406379

noncomputable section

-- Define the functions f and g
def f (x : ℝ) := Real.exp x - Real.exp (-x)
def g (m : ℝ) (x : ℝ) := Real.log (m * x^2 - x + 1/4)

-- State the theorem
theorem min_m_value :
  (∀ x1 ∈ Set.Iic (0 : ℝ), ∃ x2 : ℝ, f x1 = g m x2) →
  (∀ m' : ℝ, m' < -1/3 → ¬(∀ x1 ∈ Set.Iic (0 : ℝ), ∃ x2 : ℝ, f x1 = g m' x2)) →
  (∃ x1 ∈ Set.Iic (0 : ℝ), ∃ x2 : ℝ, f x1 = g (-1/3) x2) →
  m = -1/3 :=
sorry

end

end NUMINAMATH_CALUDE_min_m_value_l4063_406379


namespace NUMINAMATH_CALUDE_reciprocal_sum_l4063_406366

theorem reciprocal_sum (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 6 * x * y) :
  1 / x + 1 / y = 6 := by
sorry

end NUMINAMATH_CALUDE_reciprocal_sum_l4063_406366


namespace NUMINAMATH_CALUDE_square_root_problem_l4063_406322

theorem square_root_problem (x y : ℝ) 
  (h1 : (x - 1) = 9) 
  (h2 : (2 * x + y + 7) = 8) : 
  (7 - x - y) = 16 := by
  sorry

end NUMINAMATH_CALUDE_square_root_problem_l4063_406322


namespace NUMINAMATH_CALUDE_sum_base4_equals_1332_l4063_406328

/-- Converts a base 4 number to base 10 --/
def base4ToBase10 (a b c : ℕ) : ℕ := a * 4^2 + b * 4 + c

/-- Converts a base 10 number to base 4 --/
def base10ToBase4 (n : ℕ) : ℕ × ℕ × ℕ × ℕ :=
  let d := n / (4^3)
  let r := n % (4^3)
  let c := r / (4^2)
  let r' := r % (4^2)
  let b := r' / 4
  let a := r' % 4
  (d, c, b, a)

theorem sum_base4_equals_1332 :
  let sum := base4ToBase10 2 1 3 + base4ToBase10 1 3 2 + base4ToBase10 3 2 1
  base10ToBase4 sum = (1, 3, 3, 2) := by sorry

end NUMINAMATH_CALUDE_sum_base4_equals_1332_l4063_406328


namespace NUMINAMATH_CALUDE_find_x_l4063_406359

theorem find_x : 
  ∀ x : ℝ, 
  (x * 0.48 * 2.50) / (0.12 * 0.09 * 0.5) = 2400.0000000000005 → 
  x = 10.8 := by
sorry

end NUMINAMATH_CALUDE_find_x_l4063_406359


namespace NUMINAMATH_CALUDE_inequality_implication_l4063_406384

theorem inequality_implication (a b c : ℝ) (hc : c ≠ 0) :
  a * c^2 > b * c^2 → a > b := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l4063_406384


namespace NUMINAMATH_CALUDE_max_second_term_arithmetic_sequence_l4063_406368

theorem max_second_term_arithmetic_sequence : ∀ (a d : ℕ),
  a > 0 ∧ d > 0 ∧ 
  a + (a + d) + (a + 2*d) + (a + 3*d) = 52 →
  a + d ≤ 10 :=
by sorry

end NUMINAMATH_CALUDE_max_second_term_arithmetic_sequence_l4063_406368


namespace NUMINAMATH_CALUDE_probability_divisible_by_15_l4063_406349

def digits : Finset ℕ := {1, 2, 3, 5, 5, 8}

def is_valid_arrangement (n : ℕ) : Prop :=
  n ≥ 100000 ∧ n < 1000000 ∧ ∀ d, d ∈ digits.toList.map Nat.digitChar → d ∈ n.repr.data

def is_divisible_by_15 (n : ℕ) : Prop := n % 15 = 0

def total_arrangements : ℕ := 6 * 5 * 4 * 3 * 2 * 1

def favorable_arrangements : ℕ := 2 * (5 * 4 * 3 * 2 * 1)

theorem probability_divisible_by_15 :
  (favorable_arrangements : ℚ) / total_arrangements = 1 / 3 :=
sorry

end NUMINAMATH_CALUDE_probability_divisible_by_15_l4063_406349


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l4063_406327

theorem quadratic_inequality_solution (a b : ℝ) :
  (∀ x, x^2 - a*x - b < 0 ↔ 2 < x ∧ x < 3) →
  a + b = -1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l4063_406327


namespace NUMINAMATH_CALUDE_remainder_preserving_operation_l4063_406302

theorem remainder_preserving_operation (N : ℤ) (f : ℤ → ℤ) :
  N % 6 = 3 → f N % 6 = 3 →
  ∃ k : ℤ, f N = N + 6 * k :=
sorry

end NUMINAMATH_CALUDE_remainder_preserving_operation_l4063_406302


namespace NUMINAMATH_CALUDE_odd_rolls_probability_l4063_406341

/-- The probability of getting exactly k successes in n independent trials,
    where each trial has a success probability of p -/
def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k : ℚ) * p^k * (1 - p)^(n - k)

/-- The number of rolls of the die -/
def num_rolls : ℕ := 7

/-- The number of odd outcomes we're interested in -/
def num_odd : ℕ := 5

/-- The probability of rolling an odd number on a fair 6-sided die -/
def prob_odd : ℚ := 1/2

theorem odd_rolls_probability :
  binomial_probability num_rolls num_odd prob_odd = 21/128 := by
  sorry

end NUMINAMATH_CALUDE_odd_rolls_probability_l4063_406341


namespace NUMINAMATH_CALUDE_quadratic_equation_root_l4063_406353

/-- Given a quadratic equation (m+2)x^2 - x + m^2 - 4 = 0 where one root is 0,
    prove that the other root is 1/4 -/
theorem quadratic_equation_root (m : ℝ) :
  (∃ x : ℝ, (m + 2) * x^2 - x + m^2 - 4 = 0 ∧ x = 0) →
  (∃ y : ℝ, (m + 2) * y^2 - y + m^2 - 4 = 0 ∧ y = 1/4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_root_l4063_406353


namespace NUMINAMATH_CALUDE_rectangle_area_l4063_406364

/-- Given a rectangle with perimeter 100 cm and diagonal x cm, its area is 1250 - (x^2 / 2) square cm -/
theorem rectangle_area (x : ℝ) :
  let perimeter : ℝ := 100
  let diagonal : ℝ := x
  let area : ℝ := 1250 - (x^2 / 2)
  (∃ (length width : ℝ),
    length > 0 ∧ 
    width > 0 ∧
    2 * (length + width) = perimeter ∧
    length^2 + width^2 = diagonal^2 ∧
    length * width = area) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_l4063_406364


namespace NUMINAMATH_CALUDE_rectangular_solid_diagonal_l4063_406311

theorem rectangular_solid_diagonal 
  (a b c : ℝ) 
  (h1 : 2 * (a * b + b * c + a * c) = 24) 
  (h2 : a + b + c = 6) : 
  Real.sqrt (a^2 + b^2 + c^2) = 2 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_rectangular_solid_diagonal_l4063_406311


namespace NUMINAMATH_CALUDE_round_05019_to_thousandth_l4063_406389

/-- Custom rounding function that rounds to the nearest thousandth as described in the problem -/
def roundToThousandth (x : ℚ) : ℚ :=
  (⌊x * 1000⌋ : ℚ) / 1000

/-- Theorem stating that rounding 0.05019 to the nearest thousandth results in 0.050 -/
theorem round_05019_to_thousandth :
  roundToThousandth (5019 / 100000) = 50 / 1000 := by
  sorry

end NUMINAMATH_CALUDE_round_05019_to_thousandth_l4063_406389


namespace NUMINAMATH_CALUDE_solve_inequality_find_range_of_a_l4063_406374

-- Define the functions f and g
def f (x a : ℝ) : ℝ := |x - 1| + |x + a|
def g (a : ℝ) : ℝ := a^2 - a - 2

-- Theorem for part (1)
theorem solve_inequality (x : ℝ) :
  let a : ℝ := 3
  f x a > g a + 2 ↔ x < -4 ∨ x > 2 := by sorry

-- Theorem for part (2)
theorem find_range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc (-a) 1, f x a ≤ g a) ↔ a ≥ 3 := by sorry

end NUMINAMATH_CALUDE_solve_inequality_find_range_of_a_l4063_406374


namespace NUMINAMATH_CALUDE_average_boxes_theorem_l4063_406339

def boxes_day1 : ℕ := 318
def boxes_day2 : ℕ := 312
def boxes_day3_part1 : ℕ := 180
def boxes_day3_part2 : ℕ := 162
def total_days : ℕ := 3

def average_boxes_per_day : ℚ :=
  (boxes_day1 + boxes_day2 + boxes_day3_part1 + boxes_day3_part2) / total_days

theorem average_boxes_theorem : average_boxes_per_day = 324 := by
  sorry

end NUMINAMATH_CALUDE_average_boxes_theorem_l4063_406339


namespace NUMINAMATH_CALUDE_total_stripes_is_34_l4063_406356

/-- The total number of stripes on Vaishali's hats -/
def total_stripes : ℕ :=
  let hats_with_3_stripes := 4
  let hats_with_4_stripes := 3
  let hats_with_0_stripes := 6
  let hats_with_5_stripes := 2
  hats_with_3_stripes * 3 +
  hats_with_4_stripes * 4 +
  hats_with_0_stripes * 0 +
  hats_with_5_stripes * 5

/-- Theorem stating that the total number of stripes on Vaishali's hats is 34 -/
theorem total_stripes_is_34 : total_stripes = 34 := by
  sorry

end NUMINAMATH_CALUDE_total_stripes_is_34_l4063_406356


namespace NUMINAMATH_CALUDE_refrigerator_profit_percentage_l4063_406305

/-- Calculates the profit percentage for a refrigerator sale --/
theorem refrigerator_profit_percentage 
  (discounted_price : ℝ) 
  (discount_rate : ℝ) 
  (transport_cost : ℝ) 
  (installation_cost : ℝ) 
  (selling_price : ℝ) 
  (h1 : discounted_price = 13500) 
  (h2 : discount_rate = 0.20) 
  (h3 : transport_cost = 125) 
  (h4 : installation_cost = 250) 
  (h5 : selling_price = 18975) : 
  ∃ (profit_percentage : ℝ), abs (profit_percentage - 36.73) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_refrigerator_profit_percentage_l4063_406305


namespace NUMINAMATH_CALUDE_intersection_M_N_l4063_406398

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - 3*x - 4 ≤ 0}
def N : Set ℝ := {x | Real.log x ≥ 0}

-- State the theorem
theorem intersection_M_N : M ∩ N = {x | 1 ≤ x ∧ x ≤ 4} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l4063_406398


namespace NUMINAMATH_CALUDE_banana_arrangements_l4063_406387

def word_length : ℕ := 6
def a_count : ℕ := 3
def n_count : ℕ := 2
def b_count : ℕ := 1

theorem banana_arrangements : 
  (word_length.factorial) / (a_count.factorial * n_count.factorial * b_count.factorial) = 60 := by
  sorry

end NUMINAMATH_CALUDE_banana_arrangements_l4063_406387


namespace NUMINAMATH_CALUDE_problem_solution_l4063_406323

theorem problem_solution (t s x : ℝ) : 
  t = 15 * s^2 → t = 3.75 → x = s / 2 → s = 0.5 ∧ x = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l4063_406323


namespace NUMINAMATH_CALUDE_cube_root_of_four_fifth_power_l4063_406347

theorem cube_root_of_four_fifth_power : 
  (5^7 + 5^7 + 5^7 + 5^7 : ℝ)^(1/3) = 5^(7/3) * 4^(1/3) :=
by sorry

end NUMINAMATH_CALUDE_cube_root_of_four_fifth_power_l4063_406347


namespace NUMINAMATH_CALUDE_triangle_similarity_theorem_l4063_406390

-- Define the triangle XYZ
structure Triangle :=
  (X Y Z : ℝ × ℝ)

-- Define the line segment MN
structure LineSegment :=
  (M N : ℝ × ℝ)

-- Define the parallel property
def isParallel (l1 l2 : LineSegment) : Prop := sorry

-- Define the length of a line segment
def length (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem triangle_similarity_theorem (XYZ : Triangle) (MN : LineSegment) :
  isParallel MN (LineSegment.mk XYZ.X XYZ.Y) →
  length XYZ.X MN.M = 5 →
  length MN.M XYZ.Y = 8 →
  length MN.N XYZ.Z = 9 →
  length XYZ.Y XYZ.Z = 23.4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_similarity_theorem_l4063_406390


namespace NUMINAMATH_CALUDE_smallest_n_and_b_over_a_l4063_406333

theorem smallest_n_and_b_over_a : ∃ (n : ℕ+) (a b : ℝ),
  (∀ m : ℕ+, m < n → ¬∃ x y : ℝ, x > 0 ∧ y > 0 ∧ (x + 3*y*Complex.I)^(m:ℕ) = (x - 3*y*Complex.I)^(m:ℕ)) ∧
  a > 0 ∧ b > 0 ∧
  (a + 3*b*Complex.I)^(n:ℕ) = (a - 3*b*Complex.I)^(n:ℕ) ∧
  b/a = Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_smallest_n_and_b_over_a_l4063_406333


namespace NUMINAMATH_CALUDE_complex_square_eq_neg100_minus_64i_l4063_406319

theorem complex_square_eq_neg100_minus_64i (z : ℂ) :
  z^2 = -100 - 64*I ↔ z = 4 - 8*I ∨ z = -4 + 8*I := by
  sorry

end NUMINAMATH_CALUDE_complex_square_eq_neg100_minus_64i_l4063_406319


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l4063_406355

def equation (a b : ℕ) : Prop := 4 * a + b = 6

theorem min_reciprocal_sum :
  ∀ a b : ℕ, equation a b →
  (a ≠ 0 ∧ b ≠ 0) →
  (1 : ℚ) / a + (1 : ℚ) / b ≥ (1 : ℚ) / 1 + (1 : ℚ) / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l4063_406355


namespace NUMINAMATH_CALUDE_juanita_contest_cost_l4063_406367

/-- Represents the drumming contest Juanita entered -/
structure DrummingContest where
  min_drums : ℕ  -- Minimum number of drums to hit before earning money
  earn_rate : ℚ  -- Amount earned per drum hit above min_drums
  time_limit : ℕ  -- Time limit in minutes

/-- Represents Juanita's performance in the contest -/
structure Performance where
  drums_hit : ℕ  -- Number of drums hit
  money_lost : ℚ  -- Amount of money lost (negative earnings)

def contest_entry_cost (contest : DrummingContest) (performance : Performance) : ℚ :=
  let earnings := max ((performance.drums_hit - contest.min_drums) * contest.earn_rate) 0
  earnings + performance.money_lost

theorem juanita_contest_cost :
  let contest := DrummingContest.mk 200 0.025 2
  let performance := Performance.mk 300 7.5
  contest_entry_cost contest performance = 10 := by
  sorry

end NUMINAMATH_CALUDE_juanita_contest_cost_l4063_406367


namespace NUMINAMATH_CALUDE_third_candidate_votes_l4063_406329

theorem third_candidate_votes (total_votes : ℕ) 
  (h1 : total_votes = 52500)
  (h2 : ∃ (c1 c2 c3 : ℕ), c1 + c2 + c3 = total_votes ∧ c1 = 2500 ∧ c2 = 15000)
  (h3 : ∃ (winner : ℕ), winner = (2 : ℚ) / 3 * total_votes) :
  ∃ (third : ℕ), third = 35000 := by
sorry

end NUMINAMATH_CALUDE_third_candidate_votes_l4063_406329


namespace NUMINAMATH_CALUDE_bears_in_stock_is_four_l4063_406365

/-- The number of bears in the new shipment -/
def new_shipment : ℕ := 10

/-- The number of bears on each shelf -/
def bears_per_shelf : ℕ := 7

/-- The number of shelves used -/
def shelves_used : ℕ := 2

/-- The number of bears in stock before the shipment -/
def bears_in_stock : ℕ := shelves_used * bears_per_shelf - new_shipment

theorem bears_in_stock_is_four : bears_in_stock = 4 := by
  sorry

end NUMINAMATH_CALUDE_bears_in_stock_is_four_l4063_406365


namespace NUMINAMATH_CALUDE_ratio_closest_to_ten_l4063_406318

theorem ratio_closest_to_ten : 
  let r := (10^3000 + 10^3004) / (10^3001 + 10^3003)
  ∀ n : ℤ, n ≠ 10 → |r - 10| < |r - n| := by
  sorry

end NUMINAMATH_CALUDE_ratio_closest_to_ten_l4063_406318


namespace NUMINAMATH_CALUDE_sin_cos_identity_l4063_406375

theorem sin_cos_identity : 
  Real.sin (18 * π / 180) * Real.sin (78 * π / 180) - 
  Real.cos (162 * π / 180) * Real.cos (78 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l4063_406375


namespace NUMINAMATH_CALUDE_leslie_garden_walkway_area_l4063_406344

/-- Represents Leslie's garden layout --/
structure GardenLayout where
  rows : Nat
  columns : Nat
  bed_width : Nat
  bed_height : Nat
  row_walkway_width : Nat
  column_walkway_width : Nat

/-- Calculates the total area of walkways in the garden --/
def walkway_area (garden : GardenLayout) : Nat :=
  let total_width := garden.columns * garden.bed_width + (garden.columns + 1) * garden.column_walkway_width
  let total_height := garden.rows * garden.bed_height + (garden.rows + 1) * garden.row_walkway_width
  let total_area := total_width * total_height
  let beds_area := garden.rows * garden.columns * garden.bed_width * garden.bed_height
  total_area - beds_area

/-- Leslie's garden layout --/
def leslie_garden : GardenLayout :=
  { rows := 4
  , columns := 3
  , bed_width := 8
  , bed_height := 3
  , row_walkway_width := 1
  , column_walkway_width := 2
  }

/-- Theorem stating that the walkway area in Leslie's garden is 256 square feet --/
theorem leslie_garden_walkway_area :
  walkway_area leslie_garden = 256 := by
  sorry

end NUMINAMATH_CALUDE_leslie_garden_walkway_area_l4063_406344


namespace NUMINAMATH_CALUDE_dinosaur_count_correct_l4063_406378

/-- Represents the number of dinosaurs in the flock -/
def num_dinosaurs : ℕ := 5

/-- Represents the number of legs each dinosaur has -/
def legs_per_dinosaur : ℕ := 3

/-- Represents the total number of heads and legs in the flock -/
def total_heads_and_legs : ℕ := 20

/-- Proves that the number of dinosaurs in the flock is correct -/
theorem dinosaur_count_correct :
  num_dinosaurs * (legs_per_dinosaur + 1) = total_heads_and_legs :=
by sorry

end NUMINAMATH_CALUDE_dinosaur_count_correct_l4063_406378


namespace NUMINAMATH_CALUDE_inequality_proof_l4063_406350

theorem inequality_proof (a b c d : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_pos_d : d > 0)
  (h_sum : a + b + c + d = 1) : 
  a^2 / (1 + a) + b^2 / (1 + b) + c^2 / (1 + c) + d^2 / (1 + d) ≥ 1/5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4063_406350


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l4063_406381

/-- Represents a cylinder with two spheres inside it -/
structure CylinderWithSpheres where
  cylinderRadius : ℝ
  sphereRadius : ℝ
  sphereCenterDistance : ℝ

/-- Represents the ellipse formed by the intersection of a plane with the cylinder -/
structure IntersectionEllipse where
  majorAxis : ℝ

/-- The length of the major axis of the ellipse formed by a plane tangent to both spheres 
    and intersecting the cylindrical surface is equal to the distance between sphere centers -/
theorem ellipse_major_axis_length 
  (c : CylinderWithSpheres) 
  (h1 : c.cylinderRadius = 6) 
  (h2 : c.sphereRadius = 6) 
  (h3 : c.sphereCenterDistance = 13) : 
  ∃ e : IntersectionEllipse, e.majorAxis = c.sphereCenterDistance :=
sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l4063_406381


namespace NUMINAMATH_CALUDE_value_of_a_l4063_406307

theorem value_of_a (a b c : ℤ) 
  (h1 : a + b = c) 
  (h2 : b + c = 7) 
  (h3 : c = 5) : 
  a = 3 := by
sorry

end NUMINAMATH_CALUDE_value_of_a_l4063_406307


namespace NUMINAMATH_CALUDE_sqrt_two_division_l4063_406363

theorem sqrt_two_division : 3 * Real.sqrt 2 / Real.sqrt 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_division_l4063_406363


namespace NUMINAMATH_CALUDE_union_of_sets_l4063_406372

theorem union_of_sets : 
  let A : Set ℕ := {1, 2}
  let B : Set ℕ := {2, 3}
  A ∪ B = {1, 2, 3} := by
sorry

end NUMINAMATH_CALUDE_union_of_sets_l4063_406372


namespace NUMINAMATH_CALUDE_walkers_meet_at_corner_d_l4063_406357

/-- Represents the corners of the rectangular area -/
inductive Corner
| A
| B
| C
| D

/-- Represents the rectangular area -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- Represents a person walking along the perimeter -/
structure Walker where
  speed : ℚ
  startCorner : Corner
  direction : Bool  -- True for clockwise, False for counterclockwise

/-- The meeting point of two walkers -/
def meetingPoint (rect : Rectangle) (walker1 walker2 : Walker) : Corner :=
  sorry

/-- The theorem to be proved -/
theorem walkers_meet_at_corner_d 
  (rect : Rectangle)
  (jane hector : Walker)
  (h_rect_dims : rect.length = 10 ∧ rect.width = 4)
  (h_start : jane.startCorner = Corner.A ∧ hector.startCorner = Corner.A)
  (h_directions : jane.direction = false ∧ hector.direction = true)
  (h_speeds : jane.speed = 2 * hector.speed) :
  meetingPoint rect jane hector = Corner.D :=
sorry

end NUMINAMATH_CALUDE_walkers_meet_at_corner_d_l4063_406357


namespace NUMINAMATH_CALUDE_smallest_surface_areas_100_cubes_l4063_406383

/-- Represents a polyhedron formed by unit cubes -/
structure Polyhedron :=
  (length width height : ℕ)
  (total_cubes : ℕ)
  (surface_area : ℕ)

/-- Calculates the surface area of a rectangular prism -/
def calculate_surface_area (l w h : ℕ) : ℕ :=
  2 * (l * w + l * h + w * h)

/-- Generates all possible polyhedra from 100 unit cubes -/
def generate_polyhedra (n : ℕ) : List Polyhedron :=
  sorry

/-- Finds the first 6 smallest surface areas -/
def first_6_surface_areas (polyhedra : List Polyhedron) : List ℕ :=
  sorry

theorem smallest_surface_areas_100_cubes :
  let polyhedra := generate_polyhedra 100
  let areas := first_6_surface_areas polyhedra
  areas = [130, 134, 136, 138, 140, 142] :=
sorry

end NUMINAMATH_CALUDE_smallest_surface_areas_100_cubes_l4063_406383


namespace NUMINAMATH_CALUDE_odd_integers_sum_21_to_65_l4063_406343

def arithmetic_sum (a₁ : ℕ) (aₙ : ℕ) (d : ℕ) : ℕ :=
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

theorem odd_integers_sum_21_to_65 :
  arithmetic_sum 21 65 2 = 989 := by
  sorry

end NUMINAMATH_CALUDE_odd_integers_sum_21_to_65_l4063_406343


namespace NUMINAMATH_CALUDE_vasya_shirt_day_l4063_406312

structure TennisTournament where
  participants : ℕ
  days : ℕ
  matches_per_day : ℕ
  petya_shirt_day : ℕ
  petya_shirt_rank : ℕ
  vasya_shirt_rank : ℕ

def tournament : TennisTournament :=
  { participants := 20
  , days := 19
  , matches_per_day := 10
  , petya_shirt_day := 11
  , petya_shirt_rank := 11
  , vasya_shirt_rank := 15
  }

theorem vasya_shirt_day (t : TennisTournament) (h1 : t = tournament) :
  t.petya_shirt_day + (t.vasya_shirt_rank - t.petya_shirt_rank) = 15 :=
by
  sorry

#check vasya_shirt_day

end NUMINAMATH_CALUDE_vasya_shirt_day_l4063_406312


namespace NUMINAMATH_CALUDE_turner_ticket_count_l4063_406338

/-- The number of times Turner wants to ride the rollercoaster -/
def rollercoaster_rides : ℕ := 3

/-- The number of times Turner wants to ride the Catapult -/
def catapult_rides : ℕ := 2

/-- The number of times Turner wants to ride the Ferris wheel -/
def ferris_wheel_rides : ℕ := 1

/-- The number of tickets required for one rollercoaster ride -/
def rollercoaster_cost : ℕ := 4

/-- The number of tickets required for one Catapult ride -/
def catapult_cost : ℕ := 4

/-- The number of tickets required for one Ferris wheel ride -/
def ferris_wheel_cost : ℕ := 1

/-- The total number of tickets Turner needs -/
def total_tickets : ℕ := 
  rollercoaster_rides * rollercoaster_cost + 
  catapult_rides * catapult_cost + 
  ferris_wheel_rides * ferris_wheel_cost

theorem turner_ticket_count : total_tickets = 21 := by
  sorry

end NUMINAMATH_CALUDE_turner_ticket_count_l4063_406338


namespace NUMINAMATH_CALUDE_combined_machine_time_order_completion_time_l4063_406388

theorem combined_machine_time (t1 t2 : ℝ) (h1 : t1 > 0) (h2 : t2 > 0) : 
  1 / (1 / t1 + 1 / t2) = (t1 * t2) / (t1 + t2) := by sorry

theorem order_completion_time (t1 t2 : ℝ) (h1 : t1 > 0) (h2 : t2 > 0) :
  t1 = 20 → t2 = 30 → 1 / (1 / t1 + 1 / t2) = 12 := by sorry

end NUMINAMATH_CALUDE_combined_machine_time_order_completion_time_l4063_406388


namespace NUMINAMATH_CALUDE_billy_wednesday_apples_l4063_406315

/-- The number of apples Billy ate in a week -/
def total_apples : ℕ := 20

/-- The number of apples Billy ate on Monday -/
def monday_apples : ℕ := 2

/-- The number of apples Billy ate on Tuesday -/
def tuesday_apples : ℕ := 2 * monday_apples

/-- The number of apples Billy ate on Friday -/
def friday_apples : ℕ := monday_apples / 2

/-- The number of apples Billy ate on Thursday -/
def thursday_apples : ℕ := 4 * friday_apples

/-- The number of apples Billy ate on Wednesday -/
def wednesday_apples : ℕ := total_apples - (monday_apples + tuesday_apples + thursday_apples + friday_apples)

theorem billy_wednesday_apples :
  wednesday_apples = 9 := by sorry

end NUMINAMATH_CALUDE_billy_wednesday_apples_l4063_406315


namespace NUMINAMATH_CALUDE_cos_identity_l4063_406332

theorem cos_identity (α : ℝ) (h : Real.cos (π / 6 - α) = 3 / 5) :
  Real.cos (5 * π / 6 + α) = -(3 / 5) := by
  sorry

end NUMINAMATH_CALUDE_cos_identity_l4063_406332


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l4063_406397

/-- Given a hyperbola with equation y^2 - x^2/4 = 1, its asymptotes have the equation y = ± x/2 -/
theorem hyperbola_asymptotes (x y : ℝ) :
  (y^2 - x^2/4 = 1) → (∃ (k : ℝ), k = x/2 ∧ (y = k ∨ y = -k)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l4063_406397


namespace NUMINAMATH_CALUDE_nested_sqrt_evaluation_l4063_406313

theorem nested_sqrt_evaluation :
  ∃ x : ℝ, x = Real.sqrt (3 - x) ∧ x = (-1 + Real.sqrt 13) / 2 := by sorry

end NUMINAMATH_CALUDE_nested_sqrt_evaluation_l4063_406313


namespace NUMINAMATH_CALUDE_matching_shoe_probability_l4063_406386

/-- Represents a shoe pair --/
structure ShoePair :=
  (left : Nat)
  (right : Nat)

/-- The probability of selecting a matching pair of shoes --/
def probability_matching_pair (n : Nat) : Rat :=
  if n > 0 then 1 / n else 0

theorem matching_shoe_probability (cabinet : Finset ShoePair) :
  cabinet.card = 3 →
  probability_matching_pair cabinet.card = 1 / 3 := by
  sorry

#eval probability_matching_pair 3

end NUMINAMATH_CALUDE_matching_shoe_probability_l4063_406386


namespace NUMINAMATH_CALUDE_f_has_unique_zero_in_interval_l4063_406301

noncomputable def f (x : ℝ) := Real.exp x + 4 * x - 3

theorem f_has_unique_zero_in_interval :
  ∃! x, x ∈ (Set.Ioo 0 (1/2)) ∧ f x = 0 :=
sorry

end NUMINAMATH_CALUDE_f_has_unique_zero_in_interval_l4063_406301


namespace NUMINAMATH_CALUDE_tan_alpha_2_implies_fraction_equals_3_5_l4063_406336

theorem tan_alpha_2_implies_fraction_equals_3_5 (α : Real) (h : Real.tan α = 2) :
  (2 * Real.sin α - Real.cos α) / (2 * Real.sin α + Real.cos α) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_2_implies_fraction_equals_3_5_l4063_406336


namespace NUMINAMATH_CALUDE_ticket_circle_circumference_l4063_406393

/-- The circumference of a circle formed by overlapping tickets -/
theorem ticket_circle_circumference
  (ticket_length : ℝ)
  (overlap : ℝ)
  (num_tickets : ℕ)
  (h1 : ticket_length = 10.4)
  (h2 : overlap = 3.5)
  (h3 : num_tickets = 16) :
  (ticket_length - overlap) * num_tickets = 110.4 :=
by sorry

end NUMINAMATH_CALUDE_ticket_circle_circumference_l4063_406393


namespace NUMINAMATH_CALUDE_triangle_angle_b_value_l4063_406325

/-- Given a triangle ABC with side lengths a and b, and angle A, proves that angle B has a specific value. -/
theorem triangle_angle_b_value 
  (a b : ℝ) 
  (A B : ℝ) 
  (h1 : a = 2 * Real.sqrt 3)
  (h2 : b = Real.sqrt 6)
  (h3 : A = π/4)  -- 45° in radians
  (h4 : 0 < A ∧ A < π)  -- A is a valid angle
  (h5 : 0 < B ∧ B < π)  -- B is a valid angle
  : B = π/6  -- 30° in radians
:= by sorry

end NUMINAMATH_CALUDE_triangle_angle_b_value_l4063_406325


namespace NUMINAMATH_CALUDE_fraction_1790_1799_l4063_406380

/-- The number of states that joined the union from 1790 to 1799 -/
def states_1790_1799 : ℕ := 10

/-- The total number of states in Sophie's collection -/
def total_states : ℕ := 25

/-- The fraction of states that joined from 1790 to 1799 among the first 25 states -/
theorem fraction_1790_1799 : 
  (states_1790_1799 : ℚ) / total_states = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_1790_1799_l4063_406380


namespace NUMINAMATH_CALUDE_dream_car_mileage_difference_l4063_406391

/-- Proves that the difference in miles driven between tomorrow and today is 200 miles -/
theorem dream_car_mileage_difference (consumption_rate : ℝ) (today_miles : ℝ) (total_consumption : ℝ)
  (h1 : consumption_rate = 4)
  (h2 : today_miles = 400)
  (h3 : total_consumption = 4000) :
  (total_consumption / consumption_rate) - today_miles = 200 := by
  sorry

end NUMINAMATH_CALUDE_dream_car_mileage_difference_l4063_406391


namespace NUMINAMATH_CALUDE_trig_expression_equals_zero_l4063_406371

theorem trig_expression_equals_zero :
  (Real.sin (15 * π / 180) * Real.cos (15 * π / 180) + 
   Real.cos (165 * π / 180) * Real.cos (105 * π / 180)) / 
  (Real.sin (19 * π / 180) * Real.cos (11 * π / 180) + 
   Real.cos (161 * π / 180) * Real.cos (101 * π / 180)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_zero_l4063_406371


namespace NUMINAMATH_CALUDE_prob_at_least_one_expired_l4063_406310

def total_bottles : ℕ := 10
def expired_bottles : ℕ := 3
def selected_bottles : ℕ := 3

def probability_at_least_one_expired : ℚ := 17/24

theorem prob_at_least_one_expired :
  (1 : ℚ) - (Nat.choose (total_bottles - expired_bottles) selected_bottles : ℚ) / 
  (Nat.choose total_bottles selected_bottles : ℚ) = probability_at_least_one_expired := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_expired_l4063_406310


namespace NUMINAMATH_CALUDE_sum_of_abc_l4063_406321

theorem sum_of_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a * b = 30) (hac : a * c = 60) (hbc : b * c = 90) :
  a + b + c = 11 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_abc_l4063_406321


namespace NUMINAMATH_CALUDE_quadratic_always_nonnegative_implies_a_in_range_l4063_406335

theorem quadratic_always_nonnegative_implies_a_in_range (a : ℝ) :
  (∀ x : ℝ, x^2 + a*x + a ≥ 0) → a ∈ Set.Icc 0 4 := by sorry

end NUMINAMATH_CALUDE_quadratic_always_nonnegative_implies_a_in_range_l4063_406335


namespace NUMINAMATH_CALUDE_square_difference_2019_l4063_406324

theorem square_difference_2019 (a b : ℕ) (h1 : 2019 = a^2 - b^2) (h2 : a < 1000) : a = 338 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_2019_l4063_406324


namespace NUMINAMATH_CALUDE_harold_bought_three_doughnuts_l4063_406309

def harold_doughnuts (harold_coffee : ℕ) (harold_total : ℚ) 
  (melinda_doughnuts : ℕ) (melinda_coffee : ℕ) (melinda_total : ℚ) 
  (doughnut_price : ℚ) : Prop :=
  ∃ (coffee_price : ℚ),
    (doughnut_price * 3 + coffee_price * harold_coffee = harold_total) ∧
    (doughnut_price * melinda_doughnuts + coffee_price * melinda_coffee = melinda_total)

theorem harold_bought_three_doughnuts :
  harold_doughnuts 4 4.91 5 6 7.59 0.45 :=
sorry

end NUMINAMATH_CALUDE_harold_bought_three_doughnuts_l4063_406309


namespace NUMINAMATH_CALUDE_negation_equivalence_l4063_406361

theorem negation_equivalence :
  (¬ ∃ x ∈ Set.Ioo (-1 : ℝ) 3, x^2 - 1 ≤ 2*x) ↔
  (∀ x ∈ Set.Ioo (-1 : ℝ) 3, x^2 - 1 > 2*x) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l4063_406361


namespace NUMINAMATH_CALUDE_recreation_spending_comparison_l4063_406331

theorem recreation_spending_comparison 
  (last_week_wages : ℝ) 
  (last_week_recreation_percent : ℝ) 
  (this_week_wage_reduction : ℝ) 
  (this_week_recreation_percent : ℝ) 
  (h1 : last_week_recreation_percent = 0.1)
  (h2 : this_week_wage_reduction = 0.1)
  (h3 : this_week_recreation_percent = 0.4) :
  (this_week_recreation_percent * (1 - this_week_wage_reduction) * last_week_wages) / 
  (last_week_recreation_percent * last_week_wages) * 100 = 360 := by
  sorry

end NUMINAMATH_CALUDE_recreation_spending_comparison_l4063_406331


namespace NUMINAMATH_CALUDE_f_decreasing_implies_a_range_l4063_406316

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 1 then x * Real.log x - a * x^2 else a^x

theorem f_decreasing_implies_a_range (a : ℝ) :
  (∀ x y, x < y → f a x > f a y) →
  a ∈ Set.Icc (1/2) 1 ∧ a ≠ 1 :=
sorry

end NUMINAMATH_CALUDE_f_decreasing_implies_a_range_l4063_406316

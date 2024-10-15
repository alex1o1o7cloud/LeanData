import Mathlib

namespace NUMINAMATH_CALUDE_max_band_members_l1864_186432

/-- Represents a band formation --/
structure BandFormation where
  rows : ℕ
  membersPerRow : ℕ
  totalMembers : ℕ

/-- Checks if a band formation is valid according to the problem conditions --/
def isValidFormation (bf : BandFormation) : Prop :=
  bf.totalMembers < 120 ∧
  bf.totalMembers = bf.rows * bf.membersPerRow + 3 ∧
  bf.totalMembers = (bf.rows + 1) * (bf.membersPerRow + 2)

/-- Theorem stating the maximum number of band members --/
theorem max_band_members :
  ∀ bf : BandFormation, isValidFormation bf → bf.totalMembers ≤ 119 :=
by sorry

end NUMINAMATH_CALUDE_max_band_members_l1864_186432


namespace NUMINAMATH_CALUDE_base_2_representation_of_84_l1864_186459

theorem base_2_representation_of_84 :
  ∃ (a b c d e f g : ℕ),
    (a = 1 ∧ b = 0 ∧ c = 1 ∧ d = 0 ∧ e = 1 ∧ f = 0 ∧ g = 0) ∧
    84 = a * 2^6 + b * 2^5 + c * 2^4 + d * 2^3 + e * 2^2 + f * 2^1 + g * 2^0 :=
by sorry

end NUMINAMATH_CALUDE_base_2_representation_of_84_l1864_186459


namespace NUMINAMATH_CALUDE_f_three_zeros_range_l1864_186440

/-- The function f(x) = x^2 * exp(x) - a -/
noncomputable def f (x a : ℝ) : ℝ := x^2 * Real.exp x - a

/-- The statement that f has exactly three zeros -/
def has_exactly_three_zeros (a : ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ : ℝ), (x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃) ∧ 
    (f x₁ a = 0 ∧ f x₂ a = 0 ∧ f x₃ a = 0) ∧
    (∀ x : ℝ, f x a = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃)

/-- The theorem stating the range of 'a' for which f has exactly three zeros -/
theorem f_three_zeros_range :
  ∀ a : ℝ, has_exactly_three_zeros a ↔ 0 < a ∧ a < 4 / Real.exp 2 := by
  sorry

end NUMINAMATH_CALUDE_f_three_zeros_range_l1864_186440


namespace NUMINAMATH_CALUDE_enterprise_tax_comparison_l1864_186420

theorem enterprise_tax_comparison (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a < b) : 
  let x := (b - a) / 2
  let y := Real.sqrt (b / a) - 1
  b * (1 + y) > b + x := by sorry

end NUMINAMATH_CALUDE_enterprise_tax_comparison_l1864_186420


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1864_186431

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (∀ a b, a < b ∧ b < 0 → 1/a > 1/b) ∧
  (∃ a b, 1/a > 1/b ∧ ¬(a < b ∧ b < 0)) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1864_186431


namespace NUMINAMATH_CALUDE_equation_equivalence_l1864_186451

theorem equation_equivalence (x y : ℝ) :
  (x - 60) / 3 = (4 - 3 * x) / 6 + y ↔ x = (124 + 6 * y) / 5 := by
  sorry

end NUMINAMATH_CALUDE_equation_equivalence_l1864_186451


namespace NUMINAMATH_CALUDE_fraction_product_theorem_l1864_186463

/-- The type of fractions in the sequence -/
def Fraction (n : ℕ) := { k : ℕ // 2 ≤ k ∧ k ≤ n }

/-- The sequence of fractions -/
def fractionSequence (n : ℕ) : List (Fraction n) :=
  List.range (n - 1) |>.map (fun i => ⟨i + 2, by sorry⟩)

/-- The product of the original sequence of fractions -/
def originalProduct (n : ℕ) : ℚ :=
  (fractionSequence n).foldl (fun acc f => acc * (f.val : ℚ) / ((f.val - 1) : ℚ)) 1

/-- A function that determines whether a fraction should be reciprocated -/
def reciprocate (n : ℕ) : Fraction n → Bool := sorry

/-- The product after reciprocating some fractions -/
def modifiedProduct (n : ℕ) : ℚ :=
  (fractionSequence n).foldl
    (fun acc f => 
      if reciprocate n f
      then acc * ((f.val - 1) : ℚ) / (f.val : ℚ)
      else acc * (f.val : ℚ) / ((f.val - 1) : ℚ))
    1

/-- The main theorem -/
theorem fraction_product_theorem (n : ℕ) (h : n > 2) :
  (∃ (reciprocate : Fraction n → Bool), modifiedProduct n = 1) ↔ ∃ (a : ℕ), n = a^2 ∧ a > 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_theorem_l1864_186463


namespace NUMINAMATH_CALUDE_fraction_inequality_l1864_186468

theorem fraction_inequality (a b c d : ℝ) 
  (h1 : a > b) (h2 : b > 0) 
  (h3 : c < d) (h4 : d < 0) : 
  a / d < b / c := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l1864_186468


namespace NUMINAMATH_CALUDE_mango_cost_theorem_l1864_186479

/-- The cost of mangoes in dollars per pound -/
def cost_per_pound (total_cost : ℚ) (total_pounds : ℚ) : ℚ :=
  total_cost / total_pounds

/-- The cost of a given weight of mangoes in dollars -/
def cost_of_weight (cost_per_pound : ℚ) (weight : ℚ) : ℚ :=
  cost_per_pound * weight

theorem mango_cost_theorem (total_cost : ℚ) (total_pounds : ℚ) 
  (h : total_cost = 12 ∧ total_pounds = 10) : 
  cost_of_weight (cost_per_pound total_cost total_pounds) (1/2) = 0.6 := by
  sorry

#eval cost_of_weight (cost_per_pound 12 10) (1/2)

end NUMINAMATH_CALUDE_mango_cost_theorem_l1864_186479


namespace NUMINAMATH_CALUDE_train_length_calculation_l1864_186435

theorem train_length_calculation (speed_kmh : ℝ) (crossing_time : ℝ) : 
  speed_kmh = 18 → crossing_time = 20 → speed_kmh * (1000 / 3600) * crossing_time = 100 := by
  sorry

end NUMINAMATH_CALUDE_train_length_calculation_l1864_186435


namespace NUMINAMATH_CALUDE_square_inequality_l1864_186474

theorem square_inequality (a b : ℝ) : a > |b| → a^2 > b^2 := by sorry

end NUMINAMATH_CALUDE_square_inequality_l1864_186474


namespace NUMINAMATH_CALUDE_y_in_terms_of_x_l1864_186436

theorem y_in_terms_of_x (p : ℝ) (x y : ℝ) 
  (hx : x = 1 + 3^p) (hy : y = 1 + 3^(-p)) : 
  y = x / (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_y_in_terms_of_x_l1864_186436


namespace NUMINAMATH_CALUDE_onion_harvest_weight_l1864_186481

/-- Calculates the total weight of onions harvested given the number of trips, 
    initial number of bags, increase in bags per trip, and weight per bag. -/
def totalOnionWeight (trips : ℕ) (initialBags : ℕ) (increase : ℕ) (weightPerBag : ℕ) : ℕ :=
  let finalBags := initialBags + (trips - 1) * increase
  let totalBags := trips * (initialBags + finalBags) / 2
  totalBags * weightPerBag

/-- Theorem stating that the total weight of onions harvested is 29,000 kilograms
    given the specific conditions of the problem. -/
theorem onion_harvest_weight :
  totalOnionWeight 20 10 2 50 = 29000 := by
  sorry

end NUMINAMATH_CALUDE_onion_harvest_weight_l1864_186481


namespace NUMINAMATH_CALUDE_largest_quotient_and_smallest_product_l1864_186434

def S : Set ℤ := {-25, -4, -1, 3, 5, 9}

theorem largest_quotient_and_smallest_product (a b : ℤ) (ha : a ∈ S) (hb : b ∈ S) (hb_nonzero : b ≠ 0) :
  (∃ (x y : ℤ) (hx : x ∈ S) (hy : y ∈ S) (hy_nonzero : y ≠ 0), (a / b : ℚ) ≤ (x / y : ℚ)) ∧
  (∃ (x y : ℤ) (hx : x ∈ S) (hy : y ∈ S), a * b ≥ x * y) ∧
  (∃ (x y : ℤ) (hx : x ∈ S) (hy : y ∈ S) (hy_nonzero : y ≠ 0), (x / y : ℚ) = 3) ∧
  (∃ (x y : ℤ) (hx : x ∈ S) (hy : y ∈ S), x * y = -225) := by
  sorry

end NUMINAMATH_CALUDE_largest_quotient_and_smallest_product_l1864_186434


namespace NUMINAMATH_CALUDE_equation_pattern_l1864_186492

theorem equation_pattern (n : ℕ) (hn : n ≥ 1) :
  Real.sqrt (n + 1 / (n + 2)) = (n + 1) * Real.sqrt (1 / (n + 2)) := by
  sorry

end NUMINAMATH_CALUDE_equation_pattern_l1864_186492


namespace NUMINAMATH_CALUDE_largest_common_term_l1864_186414

def isInFirstSequence (n : ℕ) : Prop := ∃ k : ℕ, n = 3 + 8 * k

def isInSecondSequence (n : ℕ) : Prop := ∃ m : ℕ, n = 5 + 9 * m

theorem largest_common_term : 
  (∀ n : ℕ, n > 59 ∧ n ≤ 90 → ¬(isInFirstSequence n ∧ isInSecondSequence n)) ∧ 
  isInFirstSequence 59 ∧ 
  isInSecondSequence 59 :=
sorry

end NUMINAMATH_CALUDE_largest_common_term_l1864_186414


namespace NUMINAMATH_CALUDE_chef_cakes_problem_l1864_186442

def chef_cakes (total_eggs : ℕ) (fridge_eggs : ℕ) (eggs_per_cake : ℕ) : ℕ :=
  (total_eggs - fridge_eggs) / eggs_per_cake

theorem chef_cakes_problem :
  chef_cakes 60 10 5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_chef_cakes_problem_l1864_186442


namespace NUMINAMATH_CALUDE_sum_b_formula_l1864_186407

def a (n : ℕ) : ℕ := 2 * n + 1

def b (n : ℕ) : ℚ :=
  (Finset.sum (Finset.range n) (fun i => a (i + 1))) / n

def sum_b (n : ℕ) : ℚ :=
  Finset.sum (Finset.range n) (fun i => b (i + 1))

theorem sum_b_formula (n : ℕ) : sum_b n = (n * (n + 5) : ℚ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_b_formula_l1864_186407


namespace NUMINAMATH_CALUDE_square_difference_l1864_186439

theorem square_difference (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) : x^2 - y^2 = 80 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l1864_186439


namespace NUMINAMATH_CALUDE_ceiling_floor_sum_l1864_186430

theorem ceiling_floor_sum : ⌈(7 : ℝ) / 3⌉ + ⌊-(7 : ℝ) / 3⌋ = 0 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_sum_l1864_186430


namespace NUMINAMATH_CALUDE_gum_cost_theorem_l1864_186454

/-- Calculates the discounted cost in dollars for a bulk purchase of gum -/
def discounted_gum_cost (quantity : ℕ) (price_per_piece : ℚ) (discount_rate : ℚ) (discount_threshold : ℕ) : ℚ :=
  let total_cost := quantity * price_per_piece
  let discount := if quantity > discount_threshold then discount_rate * total_cost else 0
  (total_cost - discount) / 100

theorem gum_cost_theorem :
  discounted_gum_cost 1500 2 (1/10) 1000 = 27 := by
  sorry

end NUMINAMATH_CALUDE_gum_cost_theorem_l1864_186454


namespace NUMINAMATH_CALUDE_piece_sequence_properties_l1864_186470

/-- Represents the number of small squares in the nth piece -/
def pieceSquares (n : ℕ) : ℕ := 2 * n - 1

/-- Represents the sum of small squares in pieces 1 to n -/
def totalSquares (n : ℕ) : ℕ := n * n

/-- Represents the sum of the first n even numbers -/
def sumFirstEvenNumbers (n : ℕ) : ℕ := n * (n + 1)

/-- Represents the sum of integers from 1 to n -/
def sumIntegers (n : ℕ) : ℕ := n * (n + 1) / 2

theorem piece_sequence_properties :
  (pieceSquares 50 = 99) ∧
  (totalSquares 50 = 2500) ∧
  (sumFirstEvenNumbers 50 = 2550) ∧
  (sumIntegers 100 = 5050) := by
  sorry

end NUMINAMATH_CALUDE_piece_sequence_properties_l1864_186470


namespace NUMINAMATH_CALUDE_temperature_conversion_l1864_186482

theorem temperature_conversion (t k : ℝ) : 
  t = 5 / 9 * (k - 32) → t = 35 → k = 95 := by
  sorry

end NUMINAMATH_CALUDE_temperature_conversion_l1864_186482


namespace NUMINAMATH_CALUDE_cyclist_journey_solution_l1864_186485

/-- Represents the cyclist's journey with flat, uphill, and downhill segments -/
structure CyclistJourney where
  flat : ℝ
  uphill : ℝ
  downhill : ℝ

/-- Checks if the given journey satisfies all conditions -/
def is_valid_journey (j : CyclistJourney) : Prop :=
  -- Total distance is 80 km
  j.flat + j.uphill + j.downhill = 80 ∧
  -- Forward journey time (47/12 hours)
  j.flat / 21 + j.uphill / 12 + j.downhill / 30 = 47 / 12 ∧
  -- Return journey time (14/3 hours)
  j.flat / 21 + j.uphill / 30 + j.downhill / 12 = 14 / 3

/-- The theorem stating the correct lengths of the journey segments -/
theorem cyclist_journey_solution :
  ∃ (j : CyclistJourney), is_valid_journey j ∧ j.flat = 35 ∧ j.uphill = 15 ∧ j.downhill = 30 :=
by
  sorry


end NUMINAMATH_CALUDE_cyclist_journey_solution_l1864_186485


namespace NUMINAMATH_CALUDE_seating_arrangements_with_restriction_l1864_186449

/-- The number of ways to arrange n people in a row -/
def totalArrangements (n : ℕ) : ℕ := n.factorial

/-- The number of ways to arrange n people in a row where 2 specific people must sit together -/
def restrictedArrangements (n : ℕ) : ℕ := (n - 1).factorial * 2

/-- The number of ways to arrange n people in a row where 2 specific people cannot sit together -/
def acceptableArrangements (n : ℕ) : ℕ := totalArrangements n - restrictedArrangements n

theorem seating_arrangements_with_restriction (n : ℕ) (hn : n = 9) :
  acceptableArrangements n = 282240 := by
  sorry

#eval acceptableArrangements 9

end NUMINAMATH_CALUDE_seating_arrangements_with_restriction_l1864_186449


namespace NUMINAMATH_CALUDE_calculation_difference_l1864_186441

theorem calculation_difference : 
  let correct_calc := 8 - (2 + 5)
  let incorrect_calc := 8 - 2 + 5
  correct_calc - incorrect_calc = -10 := by
sorry

end NUMINAMATH_CALUDE_calculation_difference_l1864_186441


namespace NUMINAMATH_CALUDE_intersection_point_sum_l1864_186405

/-- Given two lines that intersect at (3, 5), prove that a + b = 86/15 -/
theorem intersection_point_sum (a b : ℚ) : 
  (3 = (1/3) * 5 + a) → 
  (5 = (1/5) * 3 + b) → 
  a + b = 86/15 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_sum_l1864_186405


namespace NUMINAMATH_CALUDE_smallest_integer_in_ratio_l1864_186422

theorem smallest_integer_in_ratio (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- positive integers
  a + b + c = 90 →         -- sum is 90
  b = 3 * a ∧ c = 5 * a →  -- ratio 2:3:5
  a = 10 :=                -- smallest integer is 10
by sorry

end NUMINAMATH_CALUDE_smallest_integer_in_ratio_l1864_186422


namespace NUMINAMATH_CALUDE_no_rectangular_prism_exists_l1864_186423

theorem no_rectangular_prism_exists : ¬∃ (a b c : ℝ), 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  a + b + c = 12 ∧ 
  a * b + b * c + c * a = 1 ∧ 
  a * b * c = 12 := by
  sorry

end NUMINAMATH_CALUDE_no_rectangular_prism_exists_l1864_186423


namespace NUMINAMATH_CALUDE_books_for_sale_l1864_186476

/-- The total number of books for sale is the sum of initial books and additional books found. -/
theorem books_for_sale (initial_books additional_books : ℕ) :
  initial_books = 33 → additional_books = 26 →
  initial_books + additional_books = 59 := by
  sorry

end NUMINAMATH_CALUDE_books_for_sale_l1864_186476


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_l1864_186499

/-- The function f(x) = x - a --/
def f (a : ℝ) (x : ℝ) : ℝ := x - a

/-- The open interval (0, 1) --/
def open_unit_interval : Set ℝ := { x | 0 < x ∧ x < 1 }

/-- f has a zero in (0, 1) --/
def has_zero_in_unit_interval (a : ℝ) : Prop :=
  ∃ x ∈ open_unit_interval, f a x = 0

theorem necessary_not_sufficient :
  (∀ a : ℝ, has_zero_in_unit_interval a → a < 1) ∧
  (∃ a : ℝ, a < 1 ∧ ¬has_zero_in_unit_interval a) := by sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_l1864_186499


namespace NUMINAMATH_CALUDE_parabola_vertex_l1864_186488

/-- The vertex of the parabola y = 2x^2 - 4x - 7 is at the point (1, -9). -/
theorem parabola_vertex (x y : ℝ) : y = 2 * x^2 - 4 * x - 7 → (1, -9) = (x, y) := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_l1864_186488


namespace NUMINAMATH_CALUDE_probability_even_or_greater_than_4_l1864_186415

/-- A fair six-sided die. -/
structure Die :=
  (faces : Finset ℕ)
  (fair : faces.card = 6)
  (labeled : faces = {1, 2, 3, 4, 5, 6})

/-- The event "the number facing up is even or greater than 4". -/
def EventEvenOrGreaterThan4 (d : Die) : Finset ℕ :=
  d.faces.filter (λ x => x % 2 = 0 ∨ x > 4)

/-- The probability of an event for a fair die. -/
def Probability (d : Die) (event : Finset ℕ) : ℚ :=
  event.card / d.faces.card

theorem probability_even_or_greater_than_4 (d : Die) :
  Probability d (EventEvenOrGreaterThan4 d) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_probability_even_or_greater_than_4_l1864_186415


namespace NUMINAMATH_CALUDE_initial_dumbbell_count_l1864_186433

theorem initial_dumbbell_count (dumbbell_weight : ℕ) (added_dumbbells : ℕ) (total_weight : ℕ) : 
  dumbbell_weight = 20 →
  added_dumbbells = 2 →
  total_weight = 120 →
  (total_weight / dumbbell_weight) - added_dumbbells = 4 := by
  sorry

end NUMINAMATH_CALUDE_initial_dumbbell_count_l1864_186433


namespace NUMINAMATH_CALUDE_nicki_total_distance_l1864_186437

/-- Represents Nicki's exercise regime for a year --/
structure ExerciseRegime where
  running_miles_first_3_months : ℕ
  running_miles_next_3_months : ℕ
  running_miles_last_6_months : ℕ
  swimming_miles_first_6_months : ℕ
  hiking_miles_per_rest_week : ℕ
  weeks_in_year : ℕ
  weeks_per_month : ℕ

/-- Calculates the total distance covered in all exercises during the year --/
def totalDistance (regime : ExerciseRegime) : ℕ :=
  let running_weeks_per_month := regime.weeks_per_month - 1
  let running_miles := 
    (running_weeks_per_month * 3 * regime.running_miles_first_3_months) +
    (running_weeks_per_month * 3 * regime.running_miles_next_3_months) +
    (running_weeks_per_month * 6 * regime.running_miles_last_6_months)
  let swimming_miles := running_weeks_per_month * 6 * regime.swimming_miles_first_6_months
  let rest_weeks := regime.weeks_in_year / 4
  let hiking_miles := rest_weeks * regime.hiking_miles_per_rest_week
  running_miles + swimming_miles + hiking_miles

/-- Theorem stating that Nicki's total distance is 1095 miles --/
theorem nicki_total_distance :
  ∃ (regime : ExerciseRegime),
    regime.running_miles_first_3_months = 10 ∧
    regime.running_miles_next_3_months = 20 ∧
    regime.running_miles_last_6_months = 30 ∧
    regime.swimming_miles_first_6_months = 5 ∧
    regime.hiking_miles_per_rest_week = 15 ∧
    regime.weeks_in_year = 52 ∧
    regime.weeks_per_month = 4 ∧
    totalDistance regime = 1095 := by
  sorry

end NUMINAMATH_CALUDE_nicki_total_distance_l1864_186437


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l1864_186421

theorem sum_of_coefficients (a b : ℝ) : 
  (Nat.choose 6 4) * a^4 * b^2 = 135 →
  (Nat.choose 6 5) * a^5 * b = -18 →
  (a + b)^6 = 64 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l1864_186421


namespace NUMINAMATH_CALUDE_basketball_game_score_l1864_186473

theorem basketball_game_score (a r b d : ℕ) : 
  -- Raiders' scores form a geometric sequence
  0 < a ∧ 1 < r ∧ 
  -- Wildcats' scores form an arithmetic sequence
  0 < b ∧ 0 < d ∧ 
  -- Game tied at end of first quarter
  a = b ∧ 
  -- Raiders won by one point
  a * (1 + r + r^2 + r^3) = 4 * b + 6 * d + 1 ∧ 
  -- Neither team scored more than 100 points
  a * (1 + r + r^2 + r^3) ≤ 100 ∧ 
  4 * b + 6 * d ≤ 100 →
  -- Total points in first half
  a + a * r + b + (b + d) = 34 := by
sorry

end NUMINAMATH_CALUDE_basketball_game_score_l1864_186473


namespace NUMINAMATH_CALUDE_arithmetic_sequence_average_l1864_186444

/-- The average value of an arithmetic sequence with 5 terms, starting at 0 and with a common difference of 3x, is 6x. -/
theorem arithmetic_sequence_average (x : ℝ) : 
  let sequence := [0, 3*x, 6*x, 9*x, 12*x]
  (sequence.sum / sequence.length : ℝ) = 6*x := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_average_l1864_186444


namespace NUMINAMATH_CALUDE_black_coverage_probability_theorem_l1864_186469

/-- Represents the square with black regions -/
structure ColoredSquare where
  side_length : ℝ
  triangle_leg : ℝ
  diamond_side : ℝ

/-- Represents the circular coin -/
structure Coin where
  diameter : ℝ

/-- Calculates the probability of the coin covering part of the black region -/
def black_coverage_probability (square : ColoredSquare) (coin : Coin) : ℝ :=
  sorry

/-- The main theorem to prove -/
theorem black_coverage_probability_theorem (square : ColoredSquare) (coin : Coin) :
  square.side_length = 10 ∧
  square.triangle_leg = 3 ∧
  square.diamond_side = 3 * Real.sqrt 2 ∧
  coin.diameter = 2 →
  black_coverage_probability square coin = (48 + 12 * Real.sqrt 2 + 2 * Real.pi) / 100 :=
sorry

end NUMINAMATH_CALUDE_black_coverage_probability_theorem_l1864_186469


namespace NUMINAMATH_CALUDE_football_competition_kicks_l1864_186480

/-- Calculates the number of penalty kicks required for a football competition --/
def penalty_kicks (total_players : ℕ) (goalkeepers : ℕ) : ℕ :=
  goalkeepers * (total_players - 1)

/-- Theorem: Given 24 players with 4 goalkeepers, 92 penalty kicks are required --/
theorem football_competition_kicks : penalty_kicks 24 4 = 92 := by
  sorry

end NUMINAMATH_CALUDE_football_competition_kicks_l1864_186480


namespace NUMINAMATH_CALUDE_expression_evaluation_l1864_186417

theorem expression_evaluation (x y : ℚ) (hx : x = 1/3) (hy : y = -2) :
  (x * (x + y) - (x - y)^2) / y = 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1864_186417


namespace NUMINAMATH_CALUDE_equation_solution_l1864_186425

theorem equation_solution : 
  ∃ (x₁ x₂ : ℝ), (x₁ = 5 ∧ x₂ = -1) ∧ 
  (∀ x : ℝ, (x - 2)^2 = 9 ↔ (x = x₁ ∨ x = x₂)) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1864_186425


namespace NUMINAMATH_CALUDE_cindy_envelopes_left_l1864_186487

/-- Calculates the number of envelopes Cindy has left after giving some to her friends -/
def envelopes_left (initial : ℕ) (friends : ℕ) (per_friend : ℕ) : ℕ :=
  initial - friends * per_friend

/-- Proves that Cindy has 22 envelopes left -/
theorem cindy_envelopes_left : 
  envelopes_left 37 5 3 = 22 := by
  sorry

end NUMINAMATH_CALUDE_cindy_envelopes_left_l1864_186487


namespace NUMINAMATH_CALUDE_firetruck_reachable_area_l1864_186429

/-- Represents the speed of the firetruck in different terrains --/
structure FiretruckSpeed where
  road : ℝ
  field : ℝ

/-- Calculates the area reachable by a firetruck given its speed and time --/
def reachable_area (speed : FiretruckSpeed) (time : ℝ) : ℝ :=
  sorry

/-- Theorem stating the area reachable by the firetruck in 15 minutes --/
theorem firetruck_reachable_area :
  let speed := FiretruckSpeed.mk 60 18
  let time := 15 / 60  -- 15 minutes in hours
  reachable_area speed time = 1194.75 := by
  sorry

end NUMINAMATH_CALUDE_firetruck_reachable_area_l1864_186429


namespace NUMINAMATH_CALUDE_max_prob_with_highest_prob_second_l1864_186413

/-- Represents a chess player with a given win probability -/
structure Player where
  winProb : ℝ

/-- Represents the chess player's opponents -/
structure Opponents where
  A : Player
  B : Player
  C : Player

/-- Calculates the probability of winning two consecutive games given the order of opponents -/
def probTwoConsecutiveWins (opponents : Opponents) (first second : Player) : ℝ :=
  2 * (first.winProb * second.winProb - 2 * opponents.A.winProb * opponents.B.winProb * opponents.C.winProb)

/-- Theorem stating that playing against the opponent with the highest win probability in the second game maximizes the probability of winning two consecutive games -/
theorem max_prob_with_highest_prob_second (opponents : Opponents)
    (h1 : opponents.C.winProb > opponents.B.winProb)
    (h2 : opponents.B.winProb > opponents.A.winProb)
    (h3 : opponents.A.winProb > 0) :
    ∀ (first : Player),
      probTwoConsecutiveWins opponents first opponents.C ≥ probTwoConsecutiveWins opponents first opponents.B ∧
      probTwoConsecutiveWins opponents first opponents.C ≥ probTwoConsecutiveWins opponents first opponents.A :=
  sorry


end NUMINAMATH_CALUDE_max_prob_with_highest_prob_second_l1864_186413


namespace NUMINAMATH_CALUDE_tiles_in_row_l1864_186447

theorem tiles_in_row (area : ℝ) (length : ℝ) (tile_size : ℝ) : 
  area = 320 → length = 16 → tile_size = 1 → 
  (area / length) / tile_size = 20 := by sorry

end NUMINAMATH_CALUDE_tiles_in_row_l1864_186447


namespace NUMINAMATH_CALUDE_probability_white_ball_l1864_186403

/-- The probability of drawing a white ball from a bag containing 2 red balls and 1 white ball is 1/3. -/
theorem probability_white_ball (red_balls white_balls total_balls : ℕ) : 
  red_balls = 2 → white_balls = 1 → total_balls = red_balls + white_balls →
  (white_balls : ℚ) / total_balls = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_white_ball_l1864_186403


namespace NUMINAMATH_CALUDE_factorization_equality_l1864_186456

theorem factorization_equality (x : ℝ) : -3*x^3 + 12*x^2 - 12*x = -3*x*(x-2)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1864_186456


namespace NUMINAMATH_CALUDE_total_pets_l1864_186462

theorem total_pets (dogs : ℕ) (fish : ℕ) (cats : ℕ)
  (h1 : dogs = 43)
  (h2 : fish = 72)
  (h3 : cats = 34) :
  dogs + fish + cats = 149 := by
  sorry

end NUMINAMATH_CALUDE_total_pets_l1864_186462


namespace NUMINAMATH_CALUDE_smallest_positive_linear_combination_l1864_186416

theorem smallest_positive_linear_combination : 
  (∃ (k : ℕ+), k = Nat.gcd 3003 60606 ∧ 
   (∀ (x : ℕ+), (∃ (m n : ℤ), x.val = 3003 * m + 60606 * n) → k ≤ x) ∧
   (∃ (m n : ℤ), k.val = 3003 * m + 60606 * n)) ∧
  Nat.gcd 3003 60606 = 273 := by
sorry

end NUMINAMATH_CALUDE_smallest_positive_linear_combination_l1864_186416


namespace NUMINAMATH_CALUDE_triangle_is_right_angled_l1864_186483

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively,
    if b² = c² + a² - ca and sin A = 2 sin C, then the triangle is right-angled. -/
theorem triangle_is_right_angled 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h1 : b^2 = c^2 + a^2 - c*a) 
  (h2 : Real.sin A = 2 * Real.sin C) 
  (h3 : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h4 : 0 < A ∧ 0 < B ∧ 0 < C) 
  (h5 : A + B + C = Real.pi) : 
  ∃ (X : ℝ), (X = A ∨ X = B ∨ X = C) ∧ X = Real.pi / 2 := by
  sorry


end NUMINAMATH_CALUDE_triangle_is_right_angled_l1864_186483


namespace NUMINAMATH_CALUDE_prob_sum_gt_15_eq_5_108_l1864_186493

/-- The number of faces on each die -/
def num_faces : ℕ := 6

/-- The minimum possible sum when rolling three dice -/
def min_sum : ℕ := 3

/-- The maximum possible sum when rolling three dice -/
def max_sum : ℕ := 18

/-- The total number of possible outcomes when rolling three dice -/
def total_outcomes : ℕ := num_faces ^ 3

/-- The number of favorable outcomes (sum > 15) -/
def favorable_outcomes : ℕ := 10

/-- The probability of rolling three dice and getting a sum greater than 15 -/
def prob_sum_gt_15 : ℚ := favorable_outcomes / total_outcomes

theorem prob_sum_gt_15_eq_5_108 : prob_sum_gt_15 = 5 / 108 := by
  sorry

end NUMINAMATH_CALUDE_prob_sum_gt_15_eq_5_108_l1864_186493


namespace NUMINAMATH_CALUDE_shifted_sine_value_l1864_186457

theorem shifted_sine_value (g f : ℝ → ℝ) :
  (∀ x, g x = Real.sin (x - π/6)) →
  (∀ x, f x = g (x - π/6)) →
  f (π/6) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_shifted_sine_value_l1864_186457


namespace NUMINAMATH_CALUDE_even_mono_decreasing_relation_l1864_186411

/-- A function f is even if f(x) = f(-x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- A function f is monotonically decreasing on [0, +∞) if
    for all x, y ≥ 0, x < y implies f(x) > f(y) -/
def IsMonoDecreasingOnNonnegatives (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → 0 ≤ y → x < y → f x > f y

theorem even_mono_decreasing_relation
    (f : ℝ → ℝ)
    (h_even : IsEven f)
    (h_mono : IsMonoDecreasingOnNonnegatives f) :
    f 1 > f (-6) := by
  sorry

end NUMINAMATH_CALUDE_even_mono_decreasing_relation_l1864_186411


namespace NUMINAMATH_CALUDE_system_solution_ratio_l1864_186409

theorem system_solution_ratio (x y a b : ℝ) (h1 : 2 * x - y = a) (h2 : 3 * y - 6 * x = b) (h3 : b ≠ 0) :
  a / b = -1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_ratio_l1864_186409


namespace NUMINAMATH_CALUDE_fare_comparison_l1864_186428

/-- Fare calculation for city A -/
def fareA (x : ℝ) : ℝ := 10 + 2 * (x - 3)

/-- Fare calculation for city B -/
def fareB (x : ℝ) : ℝ := 8 + 2.5 * (x - 3)

/-- Theorem stating the condition for city A's fare to be higher than city B's -/
theorem fare_comparison (x : ℝ) :
  x > 3 → (fareA x > fareB x ↔ 3 < x ∧ x < 7) := by sorry

end NUMINAMATH_CALUDE_fare_comparison_l1864_186428


namespace NUMINAMATH_CALUDE_angle_conversion_l1864_186478

theorem angle_conversion (angle : Real) : ∃ (k : Int) (α : Real),
  angle * Real.pi / 180 = 2 * k * Real.pi + α ∧ 0 < α ∧ α < 2 * Real.pi :=
by
  -- The angle -1485° in radians is equal to -1485 * π / 180
  -- We need to prove that this is equal to -10π + 7π/4
  -- and that 7π/4 satisfies the conditions for α
  sorry

#check angle_conversion (-1485)

end NUMINAMATH_CALUDE_angle_conversion_l1864_186478


namespace NUMINAMATH_CALUDE_joseph_drives_one_mile_more_than_kyle_l1864_186427

/-- Calculates the distance traveled given speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Joseph's driving speed in mph -/
def joseph_speed : ℝ := 50

/-- Joseph's driving time in hours -/
def joseph_time : ℝ := 2.5

/-- Kyle's driving speed in mph -/
def kyle_speed : ℝ := 62

/-- Kyle's driving time in hours -/
def kyle_time : ℝ := 2

theorem joseph_drives_one_mile_more_than_kyle :
  distance joseph_speed joseph_time - distance kyle_speed kyle_time = 1 := by
  sorry

end NUMINAMATH_CALUDE_joseph_drives_one_mile_more_than_kyle_l1864_186427


namespace NUMINAMATH_CALUDE_largest_four_digit_perfect_cube_largest_four_digit_perfect_cube_is_9261_l1864_186408

theorem largest_four_digit_perfect_cube : ℕ → Prop :=
  fun n => (1000 ≤ n ∧ n ≤ 9999) ∧  -- n is a four-digit number
            (∃ m : ℕ, n = m^3) ∧    -- n is a perfect cube
            (∀ k : ℕ, (1000 ≤ k ∧ k ≤ 9999 ∧ ∃ m : ℕ, k = m^3) → k ≤ n)  -- n is the largest such number

theorem largest_four_digit_perfect_cube_is_9261 :
  largest_four_digit_perfect_cube 9261 := by
  sorry

end NUMINAMATH_CALUDE_largest_four_digit_perfect_cube_largest_four_digit_perfect_cube_is_9261_l1864_186408


namespace NUMINAMATH_CALUDE_a_squared_plus_b_minus_c_in_M_l1864_186410

def P : Set ℤ := {x | ∃ k, x = 3*k + 1}
def Q : Set ℤ := {x | ∃ k, x = 3*k - 1}
def M : Set ℤ := {x | ∃ k, x = 3*k}

theorem a_squared_plus_b_minus_c_in_M (a b c : ℤ) 
  (ha : a ∈ P) (hb : b ∈ Q) (hc : c ∈ M) : 
  a^2 + b - c ∈ M :=
by sorry

end NUMINAMATH_CALUDE_a_squared_plus_b_minus_c_in_M_l1864_186410


namespace NUMINAMATH_CALUDE_crossed_out_digit_l1864_186494

theorem crossed_out_digit (N : Nat) (x : Nat) : 
  (N % 9 = 3) → 
  (x ≤ 9) →
  (∃ a b : Nat, N = a * 10 + x + b ∧ b < 10^9) →
  ((N - x) % 9 = 7) →
  x = 5 := by
sorry

end NUMINAMATH_CALUDE_crossed_out_digit_l1864_186494


namespace NUMINAMATH_CALUDE_nested_f_evaluation_l1864_186495

def f (x : ℝ) : ℝ := x^2 + 1

theorem nested_f_evaluation : f (f (f (-1))) = 26 := by sorry

end NUMINAMATH_CALUDE_nested_f_evaluation_l1864_186495


namespace NUMINAMATH_CALUDE_calculation_proof_l1864_186496

theorem calculation_proof : (18 / (5 + 2 - 3)) * 4 = 18 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1864_186496


namespace NUMINAMATH_CALUDE_train_speed_calculation_l1864_186475

/-- Calculates the speed of a train crossing a bridge -/
theorem train_speed_calculation (train_length bridge_length : ℝ) (crossing_time : ℝ) :
  train_length = 110 →
  bridge_length = 265 →
  crossing_time = 30 →
  (train_length + bridge_length) / crossing_time * 3.6 = 45 := by
  sorry

#check train_speed_calculation

end NUMINAMATH_CALUDE_train_speed_calculation_l1864_186475


namespace NUMINAMATH_CALUDE_circle_center_is_two_one_l1864_186455

/-- A line passing through two points -/
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- A circle defined by its center and a point on its circumference -/
structure Circle where
  center : ℝ × ℝ
  point_on_circle : ℝ × ℝ

/-- The line l passing through (2, 1) and (6, 3) -/
def l : Line := { point1 := (2, 1), point2 := (6, 3) }

/-- The circle C with center on line l and tangent to x-axis at (2, 0) -/
noncomputable def C : Circle :=
  { center := sorry,  -- To be proved
    point_on_circle := (2, 0) }

theorem circle_center_is_two_one :
  C.center = (2, 1) := by sorry

end NUMINAMATH_CALUDE_circle_center_is_two_one_l1864_186455


namespace NUMINAMATH_CALUDE_simplify_fraction_l1864_186471

theorem simplify_fraction (a b : ℝ) (h1 : a + b ≠ 0) (h2 : a + 2*b ≠ 0) :
  (a - b) / (a + 2*b) / ((a^2 - b^2) / (a^2 + 4*a*b + 4*b^2)) - 2 = -a / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1864_186471


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l1864_186400

theorem absolute_value_equation_solution :
  ∃! x : ℚ, |x - 5| = 3*x + 6 :=
by
  -- The unique solution is x = -1/4
  use (-1/4 : ℚ)
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l1864_186400


namespace NUMINAMATH_CALUDE_balloon_difference_l1864_186406

theorem balloon_difference (your_balloons friend_balloons : ℕ) 
  (h1 : your_balloons = 7) 
  (h2 : friend_balloons = 5) : 
  your_balloons - friend_balloons = 2 := by
sorry

end NUMINAMATH_CALUDE_balloon_difference_l1864_186406


namespace NUMINAMATH_CALUDE_park_problem_solution_l1864_186461

/-- The problem setup -/
structure ParkProblem where
  distance_to_park : ℝ
  mother_speed_ratio : ℝ
  time_difference : ℝ
  distance_to_company : ℝ
  mother_run_speed : ℝ
  available_time : ℝ

/-- The solution to be proved -/
structure ParkSolution where
  mother_speed : ℝ
  min_run_time : ℝ

/-- The main theorem to be proved -/
theorem park_problem_solution (p : ParkProblem) 
  (h1 : p.distance_to_park = 4320)
  (h2 : p.mother_speed_ratio = 1.2)
  (h3 : p.time_difference = 12)
  (h4 : p.distance_to_company = 2940)
  (h5 : p.mother_run_speed = 150)
  (h6 : p.available_time = 30) :
  ∃ (s : ParkSolution), 
    s.mother_speed = 72 ∧ 
    s.min_run_time = 10 ∧
    (p.distance_to_park / s.mother_speed - p.distance_to_park / (s.mother_speed / p.mother_speed_ratio) = p.time_difference) ∧
    ((p.distance_to_company - p.mother_run_speed * s.min_run_time) / s.mother_speed + s.min_run_time ≤ p.available_time) := by
  sorry

end NUMINAMATH_CALUDE_park_problem_solution_l1864_186461


namespace NUMINAMATH_CALUDE_red_cars_count_l1864_186419

theorem red_cars_count (black_cars : ℕ) (ratio_red : ℕ) (ratio_black : ℕ) : 
  black_cars = 90 → ratio_red = 3 → ratio_black = 8 → 
  ∃ red_cars : ℕ, red_cars * ratio_black = black_cars * ratio_red ∧ red_cars = 33 :=
by
  sorry

end NUMINAMATH_CALUDE_red_cars_count_l1864_186419


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1864_186401

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = q * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  is_geometric_sequence a q →
  q > 1 →
  (4 * (a 2016)^2 - 8 * (a 2016) + 3 = 0) →
  (4 * (a 2017)^2 - 8 * (a 2017) + 3 = 0) →
  a 2018 + a 2019 = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1864_186401


namespace NUMINAMATH_CALUDE_line_perpendicular_sufficient_condition_l1864_186465

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines
variable (perp : Line → Line → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perpToPlane : Line → Plane → Prop)

-- Define the intersection of two planes
variable (intersect : Plane → Plane → Line)

-- Define the subset relation for a line in a plane
variable (subset : Line → Plane → Prop)

-- State the theorem
theorem line_perpendicular_sufficient_condition
  (l m : Line) (α β : Plane)
  (h1 : intersect α β = l)
  (h2 : subset m β)
  (h3 : perpToPlane m α) :
  perp l m :=
sorry

end NUMINAMATH_CALUDE_line_perpendicular_sufficient_condition_l1864_186465


namespace NUMINAMATH_CALUDE_range_of_function_l1864_186466

open Real

theorem range_of_function (f : ℝ → ℝ) (x : ℝ) :
  (f = fun x ↦ sin (x/2) * cos (x/2) + cos (x/2)^2) →
  (x ∈ Set.Ioo 0 (π/2)) →
  ∃ y, y ∈ Set.Ioc (1/2) ((Real.sqrt 2 + 1)/2) ∧ ∃ x, f x = y ∧
  ∀ z, (∃ x, f x = z) → z ∈ Set.Ioc (1/2) ((Real.sqrt 2 + 1)/2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_function_l1864_186466


namespace NUMINAMATH_CALUDE_tenth_stays_tenth_probability_sum_of_numerator_and_denominator_l1864_186450

/-- Represents a sequence of distinct numbers -/
def DistinctSequence (n : ℕ) := { s : Fin n → ℕ // Function.Injective s }

/-- The probability of the 10th element staying in the 10th position after one bubble pass -/
def probabilityTenthStaysTenth (n : ℕ) : ℚ :=
  if n < 12 then 0 else 1 / (12 * 11)

theorem tenth_stays_tenth_probability :
  probabilityTenthStaysTenth 20 = 1 / 132 := by sorry

#eval Nat.gcd 1 132  -- Should output 1, confirming 1/132 is in lowest terms

theorem sum_of_numerator_and_denominator :
  let p := 1
  let q := 132
  p + q = 133 := by sorry

end NUMINAMATH_CALUDE_tenth_stays_tenth_probability_sum_of_numerator_and_denominator_l1864_186450


namespace NUMINAMATH_CALUDE_vector_addition_l1864_186464

variable {V : Type*} [AddCommGroup V]

theorem vector_addition (A B C : V) (a b : V) 
  (h1 : B - A = a) (h2 : C - B = b) : C - A = a + b := by
  sorry

end NUMINAMATH_CALUDE_vector_addition_l1864_186464


namespace NUMINAMATH_CALUDE_jan_skipping_speed_ratio_jan_skipping_speed_ratio_is_two_l1864_186467

/-- The ratio of Jan's skipping speed after training to her speed before training -/
theorem jan_skipping_speed_ratio : ℝ :=
  let speed_before : ℝ := 70  -- skips per minute
  let total_skips_after : ℝ := 700
  let total_minutes_after : ℝ := 5
  let speed_after : ℝ := total_skips_after / total_minutes_after
  speed_after / speed_before

/-- Proof that the ratio of Jan's skipping speed after training to her speed before training is 2 -/
theorem jan_skipping_speed_ratio_is_two :
  jan_skipping_speed_ratio = 2 := by
  sorry

end NUMINAMATH_CALUDE_jan_skipping_speed_ratio_jan_skipping_speed_ratio_is_two_l1864_186467


namespace NUMINAMATH_CALUDE_company_shares_l1864_186438

theorem company_shares (K : ℝ) (P V S I : ℝ) 
  (h1 : P + V + S + I = K)
  (h2 : K + P = 1.25 * K)
  (h3 : K + V = 1.35 * K)
  (h4 : K + 2 * S = 1.4 * K)
  (h5 : I > 0) :
  ∃ x : ℝ, x > 2.5 ∧ x * I > 0.5 * K := by
  sorry

end NUMINAMATH_CALUDE_company_shares_l1864_186438


namespace NUMINAMATH_CALUDE_time_saved_weekly_l1864_186404

/-- Time saved weekly by eliminating a daily habit -/
theorem time_saved_weekly (search_time complain_time : ℕ) (days_per_week : ℕ) : 
  search_time = 8 → complain_time = 3 → days_per_week = 7 →
  (search_time + complain_time) * days_per_week = 77 :=
by sorry

end NUMINAMATH_CALUDE_time_saved_weekly_l1864_186404


namespace NUMINAMATH_CALUDE_comparison_of_expressions_l1864_186418

theorem comparison_of_expressions : 2 + Real.log 6 / Real.log 2 > 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_comparison_of_expressions_l1864_186418


namespace NUMINAMATH_CALUDE_circles_externally_tangent_l1864_186445

def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6*y = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 8*x + 12 = 0

def center1 : ℝ × ℝ := (0, 3)
def center2 : ℝ × ℝ := (4, 0)

def radius1 : ℝ := 3
def radius2 : ℝ := 2

theorem circles_externally_tangent :
  let d := Real.sqrt ((center1.1 - center2.1)^2 + (center1.2 - center2.2)^2)
  d = radius1 + radius2 := by sorry

end NUMINAMATH_CALUDE_circles_externally_tangent_l1864_186445


namespace NUMINAMATH_CALUDE_solve_equation_l1864_186443

theorem solve_equation : ∃ m : ℤ, 2^4 - 3 = 3^3 + m ∧ m = -14 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1864_186443


namespace NUMINAMATH_CALUDE_expression_simplification_l1864_186426

theorem expression_simplification :
  let a : ℝ := Real.sqrt 3
  let b : ℝ := Real.sqrt 2
  let c : ℝ := Real.sqrt 5
  6 * 37 * (a + b) ^ (2 * (Real.log c / Real.log (a - b))) = 1110 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l1864_186426


namespace NUMINAMATH_CALUDE_xy_divides_x2_plus_y2_plus_1_l1864_186484

theorem xy_divides_x2_plus_y2_plus_1 (x y : ℕ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : (x * y) ∣ (x^2 + y^2 + 1)) : 
  (x^2 + y^2 + 1) / (x * y) = 3 := by
sorry

end NUMINAMATH_CALUDE_xy_divides_x2_plus_y2_plus_1_l1864_186484


namespace NUMINAMATH_CALUDE_store_display_arrangement_l1864_186424

def stripe_arrangement (n : ℕ) : ℕ := 
  match n with
  | 0 => 0
  | 1 => 2
  | 2 => 2
  | (n + 3) => stripe_arrangement (n + 1) + stripe_arrangement (n + 2)

theorem store_display_arrangement : 
  stripe_arrangement 10 = 110 := by sorry

end NUMINAMATH_CALUDE_store_display_arrangement_l1864_186424


namespace NUMINAMATH_CALUDE_apple_tree_bearing_time_l1864_186491

def time_to_bear_fruit (age_planted : ℕ) (age_first_apple : ℕ) : ℕ :=
  age_first_apple - age_planted

theorem apple_tree_bearing_time :
  let age_planted : ℕ := 4
  let age_first_apple : ℕ := 11
  time_to_bear_fruit age_planted age_first_apple = 7 := by
sorry

end NUMINAMATH_CALUDE_apple_tree_bearing_time_l1864_186491


namespace NUMINAMATH_CALUDE_students_transferred_theorem_l1864_186490

/-- Calculates the number of students transferred to fifth grade -/
def students_transferred_to_fifth (initial_students : ℝ) (students_left : ℝ) (final_students : ℝ) : ℝ :=
  initial_students - students_left - final_students

/-- Proves that the number of students transferred to fifth grade is 10.0 -/
theorem students_transferred_theorem (initial_students : ℝ) (students_left : ℝ) (final_students : ℝ)
  (h1 : initial_students = 42.0)
  (h2 : students_left = 4.0)
  (h3 : final_students = 28.0) :
  students_transferred_to_fifth initial_students students_left final_students = 10.0 := by
  sorry

#eval students_transferred_to_fifth 42.0 4.0 28.0

end NUMINAMATH_CALUDE_students_transferred_theorem_l1864_186490


namespace NUMINAMATH_CALUDE_max_value_ab_l1864_186477

theorem max_value_ab (a b : ℝ) (h : ∀ x : ℝ, Real.exp (x + 1) ≥ a * x + b) :
  a * b ≤ Real.exp 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_ab_l1864_186477


namespace NUMINAMATH_CALUDE_valentines_to_cinco_l1864_186472

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a date in a year -/
structure Date where
  month : Nat
  day : Nat

def valentinesDay : Date := ⟨2, 14⟩
def cincoMayo : Date := ⟨5, 5⟩

/-- Given that February 14 is a Tuesday, calculate the day of the week for any date -/
def dayOfWeek (d : Date) : DayOfWeek := sorry

/-- Calculate the number of days between two dates, inclusive -/
def daysBetween (d1 d2 : Date) : Nat := sorry

theorem valentines_to_cinco : 
  dayOfWeek valentinesDay = DayOfWeek.Tuesday →
  (dayOfWeek cincoMayo = DayOfWeek.Friday ∧ 
   daysBetween valentinesDay cincoMayo = 81) := by
  sorry

end NUMINAMATH_CALUDE_valentines_to_cinco_l1864_186472


namespace NUMINAMATH_CALUDE_rectangle_area_rectangle_area_proof_l1864_186458

theorem rectangle_area (perimeter : ℝ) (h1 : perimeter = 56) : ℝ :=
  let side_length := perimeter / 8
  let square_area := side_length ^ 2
  3 * square_area

theorem rectangle_area_proof :
  rectangle_area 56 rfl = 147 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_rectangle_area_proof_l1864_186458


namespace NUMINAMATH_CALUDE_slope_divides_area_in_half_l1864_186460

/-- L-shaped region in the xy-plane -/
structure LShapedRegion where
  vertices : List (ℝ × ℝ)
  is_l_shaped : vertices = [(0,0), (0,4), (4,4), (4,2), (6,2), (6,0)]

/-- Line passing through the origin -/
structure LineFromOrigin where
  slope : ℝ

/-- Function to calculate the area of the L-shaped region -/
def area (r : LShapedRegion) : ℝ :=
  20 -- The total area of the L-shaped region

/-- Function to calculate the area divided by a line -/
def area_divided_by_line (r : LShapedRegion) (l : LineFromOrigin) : ℝ × ℝ :=
  sorry -- Returns a pair of areas divided by the line

/-- Theorem stating that the slope 1/2 divides the area in half -/
theorem slope_divides_area_in_half (r : LShapedRegion) :
  let l := LineFromOrigin.mk (1/2)
  let (area1, area2) := area_divided_by_line r l
  area1 = area2 ∧ area1 + area2 = area r :=
sorry

end NUMINAMATH_CALUDE_slope_divides_area_in_half_l1864_186460


namespace NUMINAMATH_CALUDE_z_in_fourth_quadrant_l1864_186446

/-- A complex number is in the fourth quadrant if its real part is positive and its imaginary part is negative. -/
def in_fourth_quadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im < 0

/-- Given complex number z -/
def z : ℂ := 1 - 2 * Complex.I

/-- Theorem: The complex number z = 1 - 2i is in the fourth quadrant -/
theorem z_in_fourth_quadrant : in_fourth_quadrant z := by
  sorry

end NUMINAMATH_CALUDE_z_in_fourth_quadrant_l1864_186446


namespace NUMINAMATH_CALUDE_solve_linear_equation_l1864_186453

theorem solve_linear_equation (x : ℚ) :
  3 * x - 5 * x + 6 * x = 150 → x = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l1864_186453


namespace NUMINAMATH_CALUDE_lcm_24_90_128_l1864_186452

theorem lcm_24_90_128 : Nat.lcm (Nat.lcm 24 90) 128 = 2880 := by
  sorry

end NUMINAMATH_CALUDE_lcm_24_90_128_l1864_186452


namespace NUMINAMATH_CALUDE_inscribed_semicircle_radius_l1864_186402

/-- An isosceles triangle with a semicircle inscribed in its base -/
structure IsoscelesTriangleWithSemicircle where
  /-- The base of the isosceles triangle -/
  base : ℝ
  /-- The height of the isosceles triangle -/
  height : ℝ
  /-- The radius of the inscribed semicircle -/
  radius : ℝ
  /-- The base is positive -/
  base_pos : 0 < base
  /-- The height is positive -/
  height_pos : 0 < height
  /-- The radius is positive -/
  radius_pos : 0 < radius
  /-- The diameter of the semicircle is equal to the base of the triangle -/
  diameter_eq_base : 2 * radius = base

/-- The radius of the inscribed semicircle in the given isosceles triangle -/
theorem inscribed_semicircle_radius 
    (t : IsoscelesTriangleWithSemicircle) 
    (h1 : t.base = 20) 
    (h2 : t.height = 21) : 
    t.radius = 210 / Real.sqrt 541 := by
  sorry


end NUMINAMATH_CALUDE_inscribed_semicircle_radius_l1864_186402


namespace NUMINAMATH_CALUDE_cupcakes_sold_l1864_186489

theorem cupcakes_sold (initial : ℕ) (additional : ℕ) (final : ℕ) : 
  initial = 19 → additional = 10 → final = 24 → 
  initial + additional - final = 5 := by
sorry

end NUMINAMATH_CALUDE_cupcakes_sold_l1864_186489


namespace NUMINAMATH_CALUDE_rational_expression_value_l1864_186448

theorem rational_expression_value (a b c d m : ℚ) : 
  a ≠ 0 ∧ 
  a + b = 0 ∧ 
  c * d = 1 ∧ 
  (m = -5 ∨ m = 1) → 
  |m| - a/b + (a+b)/2020 - c*d = 1 ∨ |m| - a/b + (a+b)/2020 - c*d = 5 :=
by sorry

end NUMINAMATH_CALUDE_rational_expression_value_l1864_186448


namespace NUMINAMATH_CALUDE_f_of_f_2_l1864_186412

def f (x : ℝ) : ℝ := 2 * x^3 - 4 * x^2 + 3 * x - 1

theorem f_of_f_2 : f (f 2) = 164 := by sorry

end NUMINAMATH_CALUDE_f_of_f_2_l1864_186412


namespace NUMINAMATH_CALUDE_sum_of_triangle_operations_l1864_186497

def triangle_operation (a b c : ℤ) : ℤ := 2*a + b - c

theorem sum_of_triangle_operations : 
  triangle_operation 1 2 3 + triangle_operation 4 6 5 + triangle_operation 2 7 1 = 20 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_triangle_operations_l1864_186497


namespace NUMINAMATH_CALUDE_display_rows_for_225_cans_l1864_186486

/-- Represents a pyramidal display of cans -/
structure CanDisplay where
  rows : ℕ

/-- The number of cans in the nth row of the display -/
def cans_in_row (n : ℕ) : ℕ := 4 * n - 3

/-- The total number of cans in the display -/
def total_cans (d : CanDisplay) : ℕ :=
  (d.rows * (cans_in_row 1 + cans_in_row d.rows)) / 2

/-- The theorem stating that a display with 225 cans has 11 rows -/
theorem display_rows_for_225_cans :
  ∃ (d : CanDisplay), total_cans d = 225 ∧ d.rows = 11 :=
by sorry

end NUMINAMATH_CALUDE_display_rows_for_225_cans_l1864_186486


namespace NUMINAMATH_CALUDE_smallest_b_value_l1864_186498

/-- Given real numbers a and b satisfying certain conditions, 
    the smallest possible value of b is 2. -/
theorem smallest_b_value (a b : ℝ) 
  (h1 : 2 < a) 
  (h2 : a < b) 
  (h3 : ¬ (2 + a > b ∧ 2 + b > a ∧ a + b > 2)) 
  (h4 : ¬ (1/b + 1/a > 2 ∧ 1/b + 2 > 1/a ∧ 1/a + 2 > 1/b)) : 
  ∀ ε > 0, b ≥ 2 - ε := by
sorry

end NUMINAMATH_CALUDE_smallest_b_value_l1864_186498

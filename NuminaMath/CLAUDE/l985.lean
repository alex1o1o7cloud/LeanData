import Mathlib

namespace NUMINAMATH_CALUDE_max_display_sum_l985_98559

def is_valid_hour (h : ℕ) : Prop := h ≥ 0 ∧ h ≤ 23

def is_valid_minute (m : ℕ) : Prop := m ≥ 0 ∧ m ≤ 59

def digit_sum (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

def display_sum (h m : ℕ) : ℕ :=
  digit_sum h + digit_sum m

theorem max_display_sum :
  ∃ (h m : ℕ), is_valid_hour h ∧ is_valid_minute m ∧
  ∀ (h' m' : ℕ), is_valid_hour h' → is_valid_minute m' →
  display_sum h' m' ≤ display_sum h m ∧
  display_sum h m = 24 :=
sorry

end NUMINAMATH_CALUDE_max_display_sum_l985_98559


namespace NUMINAMATH_CALUDE_count_solutions_correct_l985_98595

/-- The number of integer solutions to x^2 - y^2 = 45 -/
def count_solutions : ℕ := 12

/-- A pair of integers (x, y) is a solution if x^2 - y^2 = 45 -/
def is_solution (x y : ℤ) : Prop := x^2 - y^2 = 45

/-- The theorem stating that there are exactly 12 integer solutions to x^2 - y^2 = 45 -/
theorem count_solutions_correct :
  (∃ (s : Finset (ℤ × ℤ)), s.card = count_solutions ∧ 
    (∀ (p : ℤ × ℤ), p ∈ s ↔ is_solution p.1 p.2)) :=
sorry


end NUMINAMATH_CALUDE_count_solutions_correct_l985_98595


namespace NUMINAMATH_CALUDE_van_tire_usage_l985_98507

/-- Represents the number of miles each tire is used in a van with a tire rotation system -/
def miles_per_tire (total_miles : ℕ) (total_tires : ℕ) (simultaneous_tires : ℕ) : ℕ :=
  (total_miles * simultaneous_tires) / total_tires

/-- Theorem stating that in a van with 6 tires, where 4 are used simultaneously,
    traveling 40,000 miles results in each tire being used for approximately 26,667 miles -/
theorem van_tire_usage :
  miles_per_tire 40000 6 4 = 26667 := by
  sorry

end NUMINAMATH_CALUDE_van_tire_usage_l985_98507


namespace NUMINAMATH_CALUDE_min_value_of_f_l985_98503

-- Define the expression as a function of x
def f (x : ℝ) : ℝ := (15 - x) * (8 - x) * (15 + x) * (8 + x) + 200

-- State the theorem
theorem min_value_of_f :
  ∀ x : ℝ, f x ≥ -6290.25 ∧ ∃ y : ℝ, f y = -6290.25 := by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l985_98503


namespace NUMINAMATH_CALUDE_phone_prob_theorem_l985_98546

def phone_prob (p1 p2 p3 : ℝ) : Prop :=
  p1 = 0.5 ∧ p2 = 0.3 ∧ p3 = 0.2 →
  p1 + p2 = 0.8

theorem phone_prob_theorem :
  ∀ p1 p2 p3 : ℝ, phone_prob p1 p2 p3 :=
by
  sorry

end NUMINAMATH_CALUDE_phone_prob_theorem_l985_98546


namespace NUMINAMATH_CALUDE_f_composition_equals_result_l985_98520

noncomputable def f (z : ℂ) : ℂ :=
  if z.im = 0 then -z^3 else z^3

theorem f_composition_equals_result : 
  f (f (f (f (1 + 2*I)))) = (23882205 - 24212218*I)^3 := by sorry

end NUMINAMATH_CALUDE_f_composition_equals_result_l985_98520


namespace NUMINAMATH_CALUDE_middle_group_frequency_l985_98539

theorem middle_group_frequency 
  (sample_size : ℕ) 
  (num_rectangles : ℕ) 
  (middle_area_ratio : ℚ) : 
  sample_size = 300 →
  num_rectangles = 9 →
  middle_area_ratio = 1/5 →
  (middle_area_ratio * (1 - middle_area_ratio / (1 + middle_area_ratio))) * sample_size = 50 :=
by sorry

end NUMINAMATH_CALUDE_middle_group_frequency_l985_98539


namespace NUMINAMATH_CALUDE_bakery_storage_theorem_l985_98581

def bakery_storage_problem (sugar : ℕ) (flour : ℕ) (baking_soda : ℕ) (added_baking_soda : ℕ) : Prop :=
  sugar = 2400 ∧
  sugar = flour ∧
  10 * baking_soda = flour ∧
  added_baking_soda = 60 ∧
  8 * (baking_soda + added_baking_soda) = flour

theorem bakery_storage_theorem :
  ∃ (sugar flour baking_soda added_baking_soda : ℕ),
    bakery_storage_problem sugar flour baking_soda added_baking_soda :=
by
  sorry

end NUMINAMATH_CALUDE_bakery_storage_theorem_l985_98581


namespace NUMINAMATH_CALUDE_child_growth_l985_98569

theorem child_growth (current_height previous_height : ℝ) 
  (h1 : current_height = 41.5)
  (h2 : previous_height = 38.5) :
  current_height - previous_height = 3 := by
  sorry

end NUMINAMATH_CALUDE_child_growth_l985_98569


namespace NUMINAMATH_CALUDE_line_circle_intersection_a_values_l985_98597

/-- A line intersecting a circle -/
structure LineCircleIntersection where
  /-- The parameter of the line equation 4x + 3y + a = 0 -/
  a : ℝ
  /-- The line 4x + 3y + a = 0 intersects the circle (x-1)^2 + (y-2)^2 = 9 -/
  intersects : ∃ (x y : ℝ), 4*x + 3*y + a = 0 ∧ (x-1)^2 + (y-2)^2 = 9
  /-- The distance between intersection points is 4√2 -/
  chord_length : ∃ (A B : ℝ × ℝ), 
    (4*(A.1) + 3*(A.2) + a = 0) ∧ ((A.1-1)^2 + (A.2-2)^2 = 9) ∧
    (4*(B.1) + 3*(B.2) + a = 0) ∧ ((B.1-1)^2 + (B.2-2)^2 = 9) ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 32

/-- The theorem stating the possible values of a -/
theorem line_circle_intersection_a_values (lci : LineCircleIntersection) :
  lci.a = -5 ∨ lci.a = -15 := by sorry

end NUMINAMATH_CALUDE_line_circle_intersection_a_values_l985_98597


namespace NUMINAMATH_CALUDE_darias_savings_correct_l985_98575

/-- Calculates the weekly savings amount needed to reach a target --/
def weekly_savings (total_cost : ℕ) (initial_savings : ℕ) (weeks : ℕ) : ℕ :=
  (total_cost - initial_savings) / weeks

/-- Proves that Daria's weekly savings amount is correct --/
theorem darias_savings_correct (total_cost initial_savings weeks : ℕ)
  (h1 : total_cost = 120)
  (h2 : initial_savings = 20)
  (h3 : weeks = 10) :
  weekly_savings total_cost initial_savings weeks = 10 := by
  sorry

end NUMINAMATH_CALUDE_darias_savings_correct_l985_98575


namespace NUMINAMATH_CALUDE_composition_equality_l985_98573

-- Define the functions f and g
def f (x : ℝ) : ℝ := 2 * x + 4
def g (x : ℝ) : ℝ := 5 * x + 2

-- State the theorem
theorem composition_equality : f (g (f 3)) = 108 := by
  sorry

end NUMINAMATH_CALUDE_composition_equality_l985_98573


namespace NUMINAMATH_CALUDE_min_value_a_plus_b_l985_98536

theorem min_value_a_plus_b (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : (2 / (2 + a)) + (1 / (a + 2 * b)) = 1) :
  (a + b ≥ Real.sqrt 2 + 1/2) ∧ 
  (a + b = Real.sqrt 2 + 1/2 ↔ a = Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_min_value_a_plus_b_l985_98536


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l985_98549

theorem sum_of_reciprocals_of_roots (r₁ r₂ : ℝ) : 
  r₁^2 - 26*r₁ + 12 = 0 → 
  r₂^2 - 26*r₂ + 12 = 0 → 
  r₁ ≠ r₂ →
  (1/r₁ + 1/r₂) = 13/6 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l985_98549


namespace NUMINAMATH_CALUDE_inequality_pattern_l985_98554

theorem inequality_pattern (x : ℝ) (n : ℕ) (h : x > 0) : 
  x + (n^n : ℝ) / x^n ≥ (n : ℝ) + 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_pattern_l985_98554


namespace NUMINAMATH_CALUDE_greatest_base_eight_digit_sum_l985_98560

/-- Represents a positive integer in base 8 --/
def BaseEightRepresentation (n : ℕ+) : List ℕ := sorry

/-- Calculates the sum of digits in a base 8 representation --/
def sumOfDigits (digits : List ℕ) : ℕ := sorry

/-- Theorem stating that the greatest possible sum of digits in base 8 for numbers less than 1728 is 23 --/
theorem greatest_base_eight_digit_sum :
  ∃ (n : ℕ+), n < 1728 ∧
  sumOfDigits (BaseEightRepresentation n) = 23 ∧
  ∀ (m : ℕ+), m < 1728 →
    sumOfDigits (BaseEightRepresentation m) ≤ 23 := by
  sorry

end NUMINAMATH_CALUDE_greatest_base_eight_digit_sum_l985_98560


namespace NUMINAMATH_CALUDE_fourth_term_is_negative_eight_l985_98523

/-- A geometric sequence with a quadratic equation and specific conditions -/
structure GeometricSequence where
  -- The quadratic equation coefficients
  a : ℝ
  b : ℝ
  c : ℝ
  -- The condition that the quadratic equation holds for the sequence
  quad_eq : a * t^2 + b * t + c = 0
  -- The conditions given in the problem
  sum_condition : a1 + a2 = -1
  diff_condition : a1 - a3 = -3
  -- The general term of the sequence
  a_n : ℕ → ℝ

/-- The theorem stating that the fourth term of the sequence is -8 -/
theorem fourth_term_is_negative_eight (seq : GeometricSequence) :
  seq.a = 1 ∧ seq.b = -36 ∧ seq.c = 288 →
  seq.a_n 4 = -8 := by
  sorry

end NUMINAMATH_CALUDE_fourth_term_is_negative_eight_l985_98523


namespace NUMINAMATH_CALUDE_complex_equation_solution_l985_98526

theorem complex_equation_solution : ∃ z : ℂ, z * (1 + Complex.I) + Complex.I = 0 ∧ z = -1/2 - Complex.I/2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l985_98526


namespace NUMINAMATH_CALUDE_correct_calculation_l985_98510

theorem correct_calculation (x : ℝ) : (4 * x + 16 = 32) → (x / 4 + 16 = 17) := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l985_98510


namespace NUMINAMATH_CALUDE_multiplication_properties_l985_98548

theorem multiplication_properties :
  (∃ (p q : Nat), Prime p ∧ Prime q ∧ ¬(Prime (p * q))) ∧
  (∀ (a b : Int), ∃ (c : Int), (a^2) * (b^2) = c^2) ∧
  (∀ (m n : Int), Odd m → Odd n → Odd (m * n)) ∧
  (∀ (x y : Int), Even x → Even y → Even (x * y)) :=
by sorry

#check multiplication_properties

end NUMINAMATH_CALUDE_multiplication_properties_l985_98548


namespace NUMINAMATH_CALUDE_expected_heads_value_l985_98565

/-- The number of coins -/
def n : ℕ := 64

/-- The probability of getting heads on a single fair coin toss -/
def p : ℚ := 1/2

/-- The probability of getting heads after up to three tosses -/
def prob_heads : ℚ := p + (1 - p) * p + (1 - p) * (1 - p) * p

/-- The expected number of coins showing heads after the process -/
def expected_heads : ℚ := n * prob_heads

theorem expected_heads_value : expected_heads = 56 := by sorry

end NUMINAMATH_CALUDE_expected_heads_value_l985_98565


namespace NUMINAMATH_CALUDE_count_perfect_square_factors_7560_l985_98533

def prime_factorization (n : Nat) : List (Nat × Nat) :=
  [(2, 3), (3, 3), (5, 1), (7, 1)]

def is_perfect_square (factor : List (Nat × Nat)) : Bool :=
  factor.all (fun (p, e) => e % 2 = 0)

def count_perfect_square_factors (n : Nat) : Nat :=
  let factors := List.filter is_perfect_square 
    (List.map (fun l => List.map (fun (p, e) => (p, Nat.min e l.2)) (prime_factorization n)) 
      [(2, 0), (2, 2), (3, 0), (3, 2), (5, 0), (7, 0)])
  factors.length

theorem count_perfect_square_factors_7560 :
  count_perfect_square_factors 7560 = 4 := by
  sorry

end NUMINAMATH_CALUDE_count_perfect_square_factors_7560_l985_98533


namespace NUMINAMATH_CALUDE_lcm_of_20_45_75_l985_98502

theorem lcm_of_20_45_75 : Nat.lcm 20 (Nat.lcm 45 75) = 900 := by sorry

end NUMINAMATH_CALUDE_lcm_of_20_45_75_l985_98502


namespace NUMINAMATH_CALUDE_base_8_representation_of_512_l985_98538

/-- Converts a natural number to its base-8 representation as a list of digits (least significant first) -/
def to_base_8 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) : List ℕ :=
      if m = 0 then [] else (m % 8) :: aux (m / 8)
    aux n

theorem base_8_representation_of_512 :
  to_base_8 512 = [0, 0, 0, 1] := by
sorry

end NUMINAMATH_CALUDE_base_8_representation_of_512_l985_98538


namespace NUMINAMATH_CALUDE_standardDeviation_best_stability_measure_l985_98585

-- Define the type for crop yields
def CropYield := ℝ

-- Define a list of crop yields
def YieldList := List CropYield

-- Define statistical measures
def mean (yields : YieldList) : ℝ := sorry
def standardDeviation (yields : YieldList) : ℝ := sorry
def maximum (yields : YieldList) : ℝ := sorry
def median (yields : YieldList) : ℝ := sorry

-- Define a measure of stability
def stabilityMeasure : (YieldList → ℝ) → Prop := sorry

-- Theorem statement
theorem standardDeviation_best_stability_measure :
  ∀ (yields : YieldList),
    stabilityMeasure standardDeviation ∧
    ¬stabilityMeasure mean ∧
    ¬stabilityMeasure maximum ∧
    ¬stabilityMeasure median :=
  sorry

end NUMINAMATH_CALUDE_standardDeviation_best_stability_measure_l985_98585


namespace NUMINAMATH_CALUDE_product_equality_l985_98525

theorem product_equality (a b : ℤ) : 
  (∃ C : ℤ, a * (a - 5) = C ∧ b * (b - 8) = C) → 
  (a * (a - 5) = 0 ∨ a * (a - 5) = 84) :=
by sorry

end NUMINAMATH_CALUDE_product_equality_l985_98525


namespace NUMINAMATH_CALUDE_dance_team_new_members_l985_98598

/-- Calculates the number of new people who joined a dance team given the initial size, number of people who quit, and final size. -/
def new_members (initial_size quit_count final_size : ℕ) : ℕ :=
  final_size - (initial_size - quit_count)

/-- Proves that 13 new people joined the dance team given the specific conditions. -/
theorem dance_team_new_members :
  let initial_size := 25
  let quit_count := 8
  let final_size := 30
  new_members initial_size quit_count final_size = 13 := by
  sorry

end NUMINAMATH_CALUDE_dance_team_new_members_l985_98598


namespace NUMINAMATH_CALUDE_hole_large_enough_for_person_l985_98571

/-- Represents a two-dimensional shape --/
structure Shape :=
  (perimeter : ℝ)

/-- Represents a hole cut in a shape --/
structure Hole :=
  (opening_size : ℝ)

/-- Represents a person --/
structure Person :=
  (size : ℝ)

/-- Function to create a hole in a shape --/
def cut_hole (s : Shape) : Hole :=
  sorry

/-- Theorem stating that it's possible to cut a hole in a shape that a person can fit through --/
theorem hole_large_enough_for_person (s : Shape) (p : Person) :
  ∃ (h : Hole), h = cut_hole s ∧ h.opening_size > p.size :=
sorry

end NUMINAMATH_CALUDE_hole_large_enough_for_person_l985_98571


namespace NUMINAMATH_CALUDE_cubic_quadratic_inequality_l985_98576

theorem cubic_quadratic_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^3 / (a^2 + a*b + b^2)) + (b^3 / (b^2 + b*c + c^2)) + (c^3 / (c^2 + c*a + a^2)) ≥ (a + b + c) / 3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_quadratic_inequality_l985_98576


namespace NUMINAMATH_CALUDE_tangent_and_trig_identity_l985_98588

theorem tangent_and_trig_identity (α : Real) (h : Real.tan α = -2) : 
  Real.tan (α - 7 * Real.pi) = -2 ∧ 
  (2 * Real.sin (Real.pi - α) * Real.sin (α - Real.pi / 2)) / 
  (Real.sin α ^ 2 - 2 * Real.cos α ^ 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_and_trig_identity_l985_98588


namespace NUMINAMATH_CALUDE_angela_jacob_insect_ratio_l985_98572

/-- Proves that the ratio of Angela's insects to Jacob's insects is 1:2 -/
theorem angela_jacob_insect_ratio :
  let dean_insects : ℕ := 30
  let jacob_insects : ℕ := 5 * dean_insects
  let angela_insects : ℕ := 75
  (angela_insects : ℚ) / jacob_insects = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_angela_jacob_insect_ratio_l985_98572


namespace NUMINAMATH_CALUDE_sin_squared_plus_sin_double_l985_98552

theorem sin_squared_plus_sin_double (α : Real) (h : Real.tan α = 1/2) :
  Real.sin α ^ 2 + Real.sin (2 * α) = 1 := by sorry

end NUMINAMATH_CALUDE_sin_squared_plus_sin_double_l985_98552


namespace NUMINAMATH_CALUDE_tiger_catch_deer_distance_l985_98590

/-- The distance a tiger needs to run to catch a deer given their speeds and initial separation -/
theorem tiger_catch_deer_distance 
  (tiger_leaps_behind : ℕ)
  (tiger_leaps_per_minute : ℕ)
  (deer_leaps_per_minute : ℕ)
  (tiger_meters_per_leap : ℕ)
  (deer_meters_per_leap : ℕ)
  (h1 : tiger_leaps_behind = 50)
  (h2 : tiger_leaps_per_minute = 5)
  (h3 : deer_leaps_per_minute = 4)
  (h4 : tiger_meters_per_leap = 8)
  (h5 : deer_meters_per_leap = 5) :
  (tiger_leaps_behind * tiger_meters_per_leap * tiger_leaps_per_minute) /
  (tiger_leaps_per_minute * tiger_meters_per_leap - deer_leaps_per_minute * deer_meters_per_leap) = 800 :=
by sorry

end NUMINAMATH_CALUDE_tiger_catch_deer_distance_l985_98590


namespace NUMINAMATH_CALUDE_hypotenuse_increase_bound_l985_98515

theorem hypotenuse_increase_bound (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  Real.sqrt ((x + 1)^2 + (y + 1)^2) - Real.sqrt (x^2 + y^2) ≤ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_hypotenuse_increase_bound_l985_98515


namespace NUMINAMATH_CALUDE_perfume_price_change_l985_98535

-- Define the original price
def original_price : ℝ := 1200

-- Define the increase percentage
def increase_percent : ℝ := 10

-- Define the decrease percentage
def decrease_percent : ℝ := 15

-- Theorem statement
theorem perfume_price_change :
  let increased_price := original_price * (1 + increase_percent / 100)
  let final_price := increased_price * (1 - decrease_percent / 100)
  original_price - final_price = 78 := by
sorry

end NUMINAMATH_CALUDE_perfume_price_change_l985_98535


namespace NUMINAMATH_CALUDE_john_toy_store_spending_l985_98530

def weekly_allowance : ℚ := 9/4  -- $2.25 as a rational number

def arcade_fraction : ℚ := 3/5

def candy_store_spending : ℚ := 3/5  -- $0.60 as a rational number

theorem john_toy_store_spending :
  let remaining_after_arcade := weekly_allowance * (1 - arcade_fraction)
  let toy_store_spending := remaining_after_arcade - candy_store_spending
  toy_store_spending / remaining_after_arcade = 1/3 := by sorry

end NUMINAMATH_CALUDE_john_toy_store_spending_l985_98530


namespace NUMINAMATH_CALUDE_exp_greater_or_equal_e_l985_98532

theorem exp_greater_or_equal_e : ∀ x : ℝ, Real.exp x ≥ Real.exp 1 := by sorry

end NUMINAMATH_CALUDE_exp_greater_or_equal_e_l985_98532


namespace NUMINAMATH_CALUDE_oprah_car_collection_l985_98555

/-- The number of cars in Oprah's collection -/
def total_cars : ℕ := 3500

/-- The average number of cars Oprah gives away per year -/
def cars_given_per_year : ℕ := 50

/-- The number of years it takes to reduce the collection -/
def years_to_reduce : ℕ := 60

/-- The number of cars left after giving away -/
def cars_left : ℕ := 500

theorem oprah_car_collection :
  total_cars = cars_left + cars_given_per_year * years_to_reduce :=
by sorry

end NUMINAMATH_CALUDE_oprah_car_collection_l985_98555


namespace NUMINAMATH_CALUDE_inequality_solution_range_l985_98580

/-- The solution set of the inequality |x - 1| < kx contains exactly three integers -/
def has_three_integer_solutions (k : ℝ) : Prop :=
  ∃ (a b c : ℤ), a < b ∧ b < c ∧
  (∀ x : ℝ, |x - 1| < k * x ↔ (x > a ∧ x < c)) ∧
  (∀ n : ℤ, |n - 1| < k * n ↔ (n = a + 1 ∨ n = b ∨ n = c - 1))

/-- The main theorem -/
theorem inequality_solution_range (k : ℝ) :
  has_three_integer_solutions k → k ∈ Set.Ioo (2/3) (3/4) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l985_98580


namespace NUMINAMATH_CALUDE_polynomial_d_abs_l985_98589

/-- A polynomial with complex roots 3 + i and 3 - i -/
def polynomial (a b c d e : ℤ) : ℂ → ℂ := fun z ↦ 
  a * (z - (3 + Complex.I))^4 + b * (z - (3 + Complex.I))^3 + 
  c * (z - (3 + Complex.I))^2 + d * (z - (3 + Complex.I)) + e

/-- The coefficients have no common factors other than 1 -/
def coprime (a b c d e : ℤ) : Prop := 
  Nat.gcd (Nat.gcd (Nat.gcd (Nat.gcd a.natAbs b.natAbs) c.natAbs) d.natAbs) e.natAbs = 1

theorem polynomial_d_abs (a b c d e : ℤ) 
  (h1 : polynomial a b c d e (3 + Complex.I) = 0)
  (h2 : coprime a b c d e) : 
  Int.natAbs d = 40 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_d_abs_l985_98589


namespace NUMINAMATH_CALUDE_coin_fraction_missing_l985_98540

theorem coin_fraction_missing (x : ℚ) : x > 0 → 
  let lost := x / 2
  let found := (3 / 4) * lost
  x - (x - lost + found) = x / 8 := by
sorry

end NUMINAMATH_CALUDE_coin_fraction_missing_l985_98540


namespace NUMINAMATH_CALUDE_right_triangle_area_l985_98528

theorem right_triangle_area (a b c : ℝ) (h1 : a = 24) (h2 : c = 26) (h3 : a^2 + b^2 = c^2) :
  (1/2) * a * b = 120 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l985_98528


namespace NUMINAMATH_CALUDE_initial_red_marbles_l985_98517

theorem initial_red_marbles (r g : ℕ) : 
  (r : ℚ) / g = 5 / 3 →
  ((r - 15) : ℚ) / (g + 18) = 1 / 2 →
  r = 34 :=
by sorry

end NUMINAMATH_CALUDE_initial_red_marbles_l985_98517


namespace NUMINAMATH_CALUDE_triangle_area_l985_98509

theorem triangle_area (a b c : ℝ) (A B C : ℝ) :
  (2 * a * b * Real.sin C = Real.sqrt 3 * (b^2 + c^2 - a^2)) →
  (a = Real.sqrt 13) →
  (c = 3) →
  (1/2 * b * c * Real.sin A = 3 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l985_98509


namespace NUMINAMATH_CALUDE_student_a_score_l985_98592

def final_score (total_questions : ℕ) (correct_answers : ℕ) : ℤ :=
  correct_answers - 2 * (total_questions - correct_answers)

theorem student_a_score :
  final_score 100 93 = 79 := by
  sorry

end NUMINAMATH_CALUDE_student_a_score_l985_98592


namespace NUMINAMATH_CALUDE_lizard_comparison_l985_98593

/-- Represents a three-eyed lizard with wrinkles and spots -/
structure Lizard where
  eyes : Nat
  wrinkle_multiplier : Nat
  spot_multiplier : Nat

/-- Calculates the number of wrinkles for a lizard -/
def wrinkles (l : Lizard) : Nat :=
  l.eyes * l.wrinkle_multiplier

/-- Calculates the number of spots for a lizard -/
def spots (l : Lizard) : Nat :=
  l.spot_multiplier * (wrinkles l) ^ 2

/-- Calculates the total number of spots and wrinkles for a lizard -/
def total_spots_and_wrinkles (l : Lizard) : Nat :=
  spots l + wrinkles l

/-- The main theorem to prove -/
theorem lizard_comparison : 
  let jans_lizard : Lizard := { eyes := 3, wrinkle_multiplier := 3, spot_multiplier := 7 }
  let cousin_lizard : Lizard := { eyes := 3, wrinkle_multiplier := 2, spot_multiplier := 5 }
  total_spots_and_wrinkles jans_lizard + total_spots_and_wrinkles cousin_lizard - 
  (jans_lizard.eyes + cousin_lizard.eyes) = 756 := by
  sorry

end NUMINAMATH_CALUDE_lizard_comparison_l985_98593


namespace NUMINAMATH_CALUDE_diane_poker_debt_l985_98563

/-- Calculates the amount owed in a poker game scenario -/
def amount_owed (initial_amount winnings total_loss : ℕ) : ℕ :=
  total_loss - (initial_amount + winnings)

/-- Theorem: In Diane's poker game scenario, she owes $50 to her friends -/
theorem diane_poker_debt : amount_owed 100 65 215 = 50 := by
  sorry

end NUMINAMATH_CALUDE_diane_poker_debt_l985_98563


namespace NUMINAMATH_CALUDE_log_z_m_value_l985_98562

theorem log_z_m_value (x y z m : ℝ) 
  (hx : x > 1) (hy : y > 1) (hz : z > 1) (hm : m > 0)
  (hlogx : Real.log m / Real.log x = 24)
  (hlogy : Real.log m / Real.log y = 40)
  (hlogxyz : Real.log m / (Real.log x + Real.log y + Real.log z) = 12) :
  Real.log m / Real.log z = 60 := by
  sorry

end NUMINAMATH_CALUDE_log_z_m_value_l985_98562


namespace NUMINAMATH_CALUDE_arrangements_theorem_l985_98599

def num_men : ℕ := 5
def num_women : ℕ := 2
def positions_for_man_a : ℕ := 2

def arrangements_count : ℕ :=
  positions_for_man_a * Nat.factorial (num_men - 1 + 1) * Nat.factorial num_women

theorem arrangements_theorem : arrangements_count = 480 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_theorem_l985_98599


namespace NUMINAMATH_CALUDE_selection_problem_l985_98584

theorem selection_problem (total : Nat) (translation_capable : Nat) (software_capable : Nat) 
  (both_capable : Nat) (to_select : Nat) (for_translation : Nat) (for_software : Nat) :
  total = 8 →
  translation_capable = 5 →
  software_capable = 4 →
  both_capable = 1 →
  to_select = 5 →
  for_translation = 3 →
  for_software = 2 →
  (Nat.choose (translation_capable - 1) for_translation * 
   Nat.choose (software_capable - 1) for_software) +
  (Nat.choose (translation_capable - 1) (for_translation - 1) * 
   Nat.choose software_capable for_software) +
  (Nat.choose translation_capable for_translation * 
   Nat.choose (software_capable - 1) (for_software - 1)) = 42 := by
  sorry

#check selection_problem

end NUMINAMATH_CALUDE_selection_problem_l985_98584


namespace NUMINAMATH_CALUDE_mathematics_letter_probability_l985_98519

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The number of unique letters in 'MATHEMATICS' -/
def unique_letters : ℕ := 8

/-- The probability of selecting a letter from the alphabet that appears in 'MATHEMATICS' -/
def probability : ℚ := unique_letters / alphabet_size

theorem mathematics_letter_probability : probability = 4 / 13 := by
  sorry

end NUMINAMATH_CALUDE_mathematics_letter_probability_l985_98519


namespace NUMINAMATH_CALUDE_three_solutions_iff_a_values_l985_98506

-- Define the system of equations
def equation1 (x y : ℝ) : Prop :=
  (|y - 10| + |x + 3| - 2) * (x^2 + y^2 - 6) = 0

def equation2 (x y a : ℝ) : Prop :=
  (x + 3)^2 + (y - 5)^2 = a

-- Define the solution set
def solution_set (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | equation1 p.1 p.2 ∧ equation2 p.1 p.2 a}

-- Theorem statement
theorem three_solutions_iff_a_values (a : ℝ) :
  (solution_set a).ncard = 3 ↔ (a = 49 ∨ a = 40 - 4 * Real.sqrt 51) :=
sorry

end NUMINAMATH_CALUDE_three_solutions_iff_a_values_l985_98506


namespace NUMINAMATH_CALUDE_spanish_not_german_students_l985_98591

theorem spanish_not_german_students (total : ℕ) (both : ℕ) (spanish : ℕ) (german : ℕ) : 
  total = 30 →
  both = 2 →
  spanish = 3 * german →
  spanish + german - both = total →
  spanish - both = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_spanish_not_german_students_l985_98591


namespace NUMINAMATH_CALUDE_trumpington_band_max_size_l985_98511

theorem trumpington_band_max_size :
  ∃ m : ℕ,
    (∀ k : ℕ, 24 * k < 1000 → 24 * k ≤ 24 * m) ∧
    (24 * m < 1000) ∧
    (24 * m % 30 = 6) ∧
    (24 * m = 936) := by
  sorry

end NUMINAMATH_CALUDE_trumpington_band_max_size_l985_98511


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l985_98553

-- Define the universal set U as the real numbers
def U := ℝ

-- Define set A as the non-negative real numbers
def A := {x : ℝ | x ≥ 0}

-- State the theorem
theorem complement_of_A_in_U : Set.compl A = {x : ℝ | x < 0} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l985_98553


namespace NUMINAMATH_CALUDE_unique_solution_cubic_equation_l985_98504

theorem unique_solution_cubic_equation :
  ∀ x : ℝ, (1 + x^2) * (1 + x^4) = 4 * x^3 → x = 1 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_cubic_equation_l985_98504


namespace NUMINAMATH_CALUDE_lcm_problem_l985_98596

theorem lcm_problem (a b : ℕ+) (h1 : Nat.gcd a b = 16) (h2 : a * b = 2560) :
  Nat.lcm a b = 160 := by
  sorry

end NUMINAMATH_CALUDE_lcm_problem_l985_98596


namespace NUMINAMATH_CALUDE_soap_brand_usage_l985_98542

theorem soap_brand_usage (total : ℕ) (only_A : ℕ) (both : ℕ) (ratio_B_to_both : ℕ) : 
  total = 240 →
  only_A = 60 →
  both = 25 →
  ratio_B_to_both = 3 →
  total - (only_A + ratio_B_to_both * both + both) = 80 := by
sorry

end NUMINAMATH_CALUDE_soap_brand_usage_l985_98542


namespace NUMINAMATH_CALUDE_interior_diagonal_sum_for_specific_box_l985_98574

/-- Represents a rectangular box with given surface area and edge length sum -/
structure RectangularBox where
  surface_area : ℝ
  edge_length_sum : ℝ

/-- Calculates the sum of lengths of all interior diagonals of a rectangular box -/
def interior_diagonal_sum (box : RectangularBox) : ℝ :=
  sorry

/-- Theorem: For a rectangular box with surface area 130 and edge length sum 56,
    the sum of interior diagonal lengths is 4√66 -/
theorem interior_diagonal_sum_for_specific_box :
  let box : RectangularBox := { surface_area := 130, edge_length_sum := 56 }
  interior_diagonal_sum box = 4 * Real.sqrt 66 := by
  sorry

end NUMINAMATH_CALUDE_interior_diagonal_sum_for_specific_box_l985_98574


namespace NUMINAMATH_CALUDE_certain_number_problem_l985_98505

theorem certain_number_problem (N : ℚ) : 
  (5 / 6 : ℚ) * N - (5 / 16 : ℚ) * N = 300 → N = 576 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l985_98505


namespace NUMINAMATH_CALUDE_hydrochloric_acid_moles_required_l985_98522

/-- Represents a chemical substance with its coefficient in a chemical equation -/
structure Substance where
  name : String
  coefficient : ℕ

/-- Represents a chemical reaction with reactants and products -/
structure Reaction where
  reactants : List Substance
  products : List Substance

def sodium_bisulfite : Substance := ⟨"NaHSO3", 1⟩
def hydrochloric_acid : Substance := ⟨"HCl", 1⟩
def sodium_chloride : Substance := ⟨"NaCl", 1⟩
def water : Substance := ⟨"H2O", 1⟩
def sulfur_dioxide : Substance := ⟨"SO2", 1⟩

def reaction : Reaction :=
  ⟨[sodium_bisulfite, hydrochloric_acid], [sodium_chloride, water, sulfur_dioxide]⟩

/-- The number of moles of a substance required or produced in a reaction -/
def moles_required (s : Substance) (n : ℕ) : ℕ := s.coefficient * n

theorem hydrochloric_acid_moles_required :
  moles_required hydrochloric_acid 2 = 2 :=
sorry

end NUMINAMATH_CALUDE_hydrochloric_acid_moles_required_l985_98522


namespace NUMINAMATH_CALUDE_same_solution_implies_c_equals_6_l985_98566

theorem same_solution_implies_c_equals_6 (x : ℝ) (c : ℝ) : 
  (3 * x + 6 = 0) → (c * x + 15 = 3) → c = 6 := by
  sorry

end NUMINAMATH_CALUDE_same_solution_implies_c_equals_6_l985_98566


namespace NUMINAMATH_CALUDE_unique_solution_l985_98512

theorem unique_solution : ∃! x : ℝ, ((x / 8) + 8 - 30) * 6 = 12 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l985_98512


namespace NUMINAMATH_CALUDE_sum_of_common_ratios_is_two_l985_98500

/-- Given two nonconstant geometric sequences with different common ratios,
    prove that the sum of their common ratios is 2 if the difference of their
    third terms equals 3 times the difference of their second terms minus their first term. -/
theorem sum_of_common_ratios_is_two
  (k : ℝ) (p r : ℝ) (hp : p ≠ 1) (hr : r ≠ 1) (hpr : p ≠ r) (hk : k ≠ 0) :
  (k * p^2 - k * r^2 = 3 * (k * p - k * r) - k) →
  p + r = 2 :=
sorry

end NUMINAMATH_CALUDE_sum_of_common_ratios_is_two_l985_98500


namespace NUMINAMATH_CALUDE_relay_team_members_l985_98578

/-- Represents a cross-country relay team -/
structure RelayTeam where
  totalDistance : ℝ
  standardMemberDistance : ℝ
  ralphDistance : ℝ
  otherMembersCount : ℕ

/-- Conditions for the relay team -/
def validRelayTeam (team : RelayTeam) : Prop :=
  team.totalDistance = 18 ∧
  team.standardMemberDistance = 3 ∧
  team.ralphDistance = 2 * team.standardMemberDistance ∧
  team.totalDistance = team.ralphDistance + team.otherMembersCount * team.standardMemberDistance

/-- Theorem: The number of other team members is 4 -/
theorem relay_team_members (team : RelayTeam) (h : validRelayTeam team) : 
  team.otherMembersCount = 4 := by
  sorry

end NUMINAMATH_CALUDE_relay_team_members_l985_98578


namespace NUMINAMATH_CALUDE_count_two_digit_numbers_tens_less_than_ones_eq_36_l985_98551

/-- The count of two-digit numbers where the tens digit is less than the ones digit -/
def count_two_digit_numbers_tens_less_than_ones : ℕ :=
  (Finset.range 9).sum (λ t => (Finset.range (10 - t)).card)

/-- Theorem stating that the count of two-digit numbers where the tens digit is less than the ones digit is 36 -/
theorem count_two_digit_numbers_tens_less_than_ones_eq_36 :
  count_two_digit_numbers_tens_less_than_ones = 36 := by
  sorry

end NUMINAMATH_CALUDE_count_two_digit_numbers_tens_less_than_ones_eq_36_l985_98551


namespace NUMINAMATH_CALUDE_chantel_bracelets_l985_98543

/-- The number of bracelets Chantel gave away at soccer practice -/
def bracelets_given_at_soccer : ℕ := sorry

/-- The number of days Chantel makes 2 bracelets per day -/
def days_making_two : ℕ := 5

/-- The number of bracelets Chantel makes per day in the first period -/
def bracelets_per_day_first : ℕ := 2

/-- The number of bracelets Chantel gives away at school -/
def bracelets_given_at_school : ℕ := 3

/-- The number of days Chantel makes 3 bracelets per day -/
def days_making_three : ℕ := 4

/-- The number of bracelets Chantel makes per day in the second period -/
def bracelets_per_day_second : ℕ := 3

/-- The number of bracelets Chantel has at the end -/
def bracelets_at_end : ℕ := 13

theorem chantel_bracelets : 
  bracelets_given_at_soccer = 
    days_making_two * bracelets_per_day_first + 
    days_making_three * bracelets_per_day_second - 
    bracelets_given_at_school - 
    bracelets_at_end := by sorry

end NUMINAMATH_CALUDE_chantel_bracelets_l985_98543


namespace NUMINAMATH_CALUDE_rectangle_area_measurement_error_l985_98582

theorem rectangle_area_measurement_error (L W : ℝ) (hL : L > 0) (hW : W > 0) : 
  let true_area := L * W
  let measured_length := 1.20 * L
  let measured_width := 0.90 * W
  let calculated_area := measured_length * measured_width
  let error := calculated_area - true_area
  let error_percent := (error / true_area) * 100
  error_percent = 8 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_measurement_error_l985_98582


namespace NUMINAMATH_CALUDE_felix_brother_lift_multiple_l985_98594

theorem felix_brother_lift_multiple :
  ∀ (felix_weight brother_weight : ℝ),
  felix_weight > 0 →
  brother_weight > 0 →
  1.5 * felix_weight = 150 →
  brother_weight = 2 * felix_weight →
  600 / brother_weight = 3 := by
  sorry

end NUMINAMATH_CALUDE_felix_brother_lift_multiple_l985_98594


namespace NUMINAMATH_CALUDE_quiz_sum_l985_98516

theorem quiz_sum (x y : ℕ+) (h1 : x - y = 4) (h2 : x * y = 104) : x + y = 20 := by
  sorry

end NUMINAMATH_CALUDE_quiz_sum_l985_98516


namespace NUMINAMATH_CALUDE_inequality_solution_set_l985_98524

theorem inequality_solution_set (x : ℝ) : 
  (x - 1) * (x - 2) * (x - 3)^2 > 0 ↔ x < 1 ∨ 1 < x ∧ x < 2 ∨ 2 < x ∧ x < 3 ∨ x > 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l985_98524


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l985_98570

def U : Set Nat := {0, 1, 2, 3}
def A : Set Nat := {0, 1}
def B : Set Nat := {1, 2, 3}

theorem complement_intersection_theorem :
  (U \ A) ∩ B = {2, 3} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l985_98570


namespace NUMINAMATH_CALUDE_lcm_sequence_lower_bound_l985_98577

theorem lcm_sequence_lower_bound (a : Fin 2000 → ℕ+) 
  (h_distinct : ∀ i j, i ≠ j → a i ≠ a j)
  (h_order : ∀ i j, i < j → a i < a j)
  (h_upper_bound : ∀ i, a i < 4000)
  (h_lcm : ∀ i j, i ≠ j → Nat.lcm (a i) (a j) ≥ 4000) :
  a 0 ≥ 1334 := by
  sorry

end NUMINAMATH_CALUDE_lcm_sequence_lower_bound_l985_98577


namespace NUMINAMATH_CALUDE_X_inverse_of_A_l985_98529

def A : Matrix (Fin 3) (Fin 3) ℚ := !![2, -1, 0; -3, 5, 0; 0, 0, 2]

def X : Matrix (Fin 3) (Fin 3) ℚ := !![5/7, 1/7, 0; 3/7, 2/7, 0; 0, 0, 1/2]

theorem X_inverse_of_A : X * A = 1 := by sorry

end NUMINAMATH_CALUDE_X_inverse_of_A_l985_98529


namespace NUMINAMATH_CALUDE_total_heroes_l985_98568

/-- The number of heroes Will drew on the front of the paper. -/
def heroes_on_front : ℕ := 2

/-- The number of heroes Will drew on the back of the paper. -/
def heroes_on_back : ℕ := 7

/-- The total number of heroes Will drew is the sum of heroes on the front and back. -/
theorem total_heroes : heroes_on_front + heroes_on_back = 9 := by
  sorry

end NUMINAMATH_CALUDE_total_heroes_l985_98568


namespace NUMINAMATH_CALUDE_circle_slope_range_l985_98508

/-- The range of y/x for points on the circle x^2 + y^2 - 4x - 6y + 12 = 0 -/
theorem circle_slope_range :
  let circle := {p : ℝ × ℝ | p.1^2 + p.2^2 - 4*p.1 - 6*p.2 + 12 = 0}
  ∀ (x y : ℝ), (x, y) ∈ circle → x ≠ 0 →
    (6 - 2*Real.sqrt 3) / 3 ≤ y / x ∧ y / x ≤ (6 + 2*Real.sqrt 3) / 3 :=
by sorry


end NUMINAMATH_CALUDE_circle_slope_range_l985_98508


namespace NUMINAMATH_CALUDE_expected_vote_for_a_l985_98544

/-- Percentage of registered voters who are Democrats -/
def democrat_percentage : ℝ := 0.60

/-- Percentage of registered voters who are Republicans -/
def republican_percentage : ℝ := 1 - democrat_percentage

/-- Percentage of Democrats expected to vote for candidate A -/
def democrat_vote_a : ℝ := 0.70

/-- Percentage of Republicans expected to vote for candidate A -/
def republican_vote_a : ℝ := 0.20

/-- Theorem: The percentage of registered voters expected to vote for candidate A is 50% -/
theorem expected_vote_for_a :
  democrat_percentage * democrat_vote_a + republican_percentage * republican_vote_a = 0.50 := by
  sorry

end NUMINAMATH_CALUDE_expected_vote_for_a_l985_98544


namespace NUMINAMATH_CALUDE_smallest_cube_root_with_small_remainder_l985_98579

theorem smallest_cube_root_with_small_remainder (m n : ℕ) (r : ℝ) : 
  (∀ k < m, ¬∃ (j : ℕ) (s : ℝ), k^(1/3 : ℝ) = j + s ∧ 0 < s ∧ s < 1/2000) →
  (m : ℝ)^(1/3 : ℝ) = n + r →
  0 < r →
  r < 1/2000 →
  n = 26 := by
sorry

end NUMINAMATH_CALUDE_smallest_cube_root_with_small_remainder_l985_98579


namespace NUMINAMATH_CALUDE_first_number_value_l985_98501

theorem first_number_value (a b c d : ℝ) : 
  (a + b + c) / 3 = 20 →
  (b + c + d) / 3 = 15 →
  d = 18 →
  a = 33 := by
sorry

end NUMINAMATH_CALUDE_first_number_value_l985_98501


namespace NUMINAMATH_CALUDE_g_value_at_3_l985_98518

def g (x : ℝ) : ℝ := 2 * x^5 - 3 * x^4 + 4 * x^2 - 3 * x + 6

theorem g_value_at_3 (h : g (-3) = 2) : g 3 = -20 := by
  sorry

end NUMINAMATH_CALUDE_g_value_at_3_l985_98518


namespace NUMINAMATH_CALUDE_inverse_proportion_k_value_l985_98561

/-- Given an inverse proportion function y = k/x passing through the point (2, -1), 
    prove that k = -2 -/
theorem inverse_proportion_k_value : 
  ∀ k : ℝ, (∀ x : ℝ, x ≠ 0 → -1 = k / 2) → k = -2 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_k_value_l985_98561


namespace NUMINAMATH_CALUDE_circle_tangency_l985_98537

/-- Two circles are externally tangent if the distance between their centers
    equals the sum of their radii -/
def externally_tangent (c1_center : ℝ × ℝ) (c1_radius : ℝ) 
                       (c2_center : ℝ × ℝ) (c2_radius : ℝ) : Prop :=
  (c1_center.1 - c2_center.1)^2 + (c1_center.2 - c2_center.2)^2 = (c1_radius + c2_radius)^2

theorem circle_tangency (a : ℝ) (h : a > 0) :
  externally_tangent (a, 0) 2 (0, Real.sqrt 5) 3 → a = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_circle_tangency_l985_98537


namespace NUMINAMATH_CALUDE_max_value_a_l985_98547

theorem max_value_a (a b c d : ℕ+) (h1 : a < 3 * b) (h2 : b < 4 * c) (h3 : c < 5 * d) 
  (h4 : Even c) (h5 : d < 150) : a ≤ 8924 := by
  sorry

end NUMINAMATH_CALUDE_max_value_a_l985_98547


namespace NUMINAMATH_CALUDE_gumball_probability_l985_98557

/-- Given a jar with blue and pink gumballs, if the probability of drawing
    two blue gumballs with replacement is 25/49, then the probability of
    drawing a pink gumball is 2/7. -/
theorem gumball_probability (p_blue p_pink : ℝ) : 
  p_blue + p_pink = 1 →
  p_blue^2 = 25/49 →
  p_pink = 2/7 := by
sorry

end NUMINAMATH_CALUDE_gumball_probability_l985_98557


namespace NUMINAMATH_CALUDE_farm_field_correct_l985_98564

/-- Represents the farm field ploughing problem -/
structure FarmField where
  total_area : ℕ  -- Total area of the farm field in hectares
  planned_days : ℕ  -- Initially planned number of days
  daily_plan : ℕ  -- Hectares planned to be ploughed per day
  actual_daily : ℕ  -- Hectares actually ploughed per day
  extra_days : ℕ  -- Additional days worked
  remaining : ℕ  -- Hectares remaining to be ploughed

/-- The farm field problem solution -/
def farm_field_solution : FarmField :=
  { total_area := 720
  , planned_days := 6
  , daily_plan := 120
  , actual_daily := 85
  , extra_days := 2
  , remaining := 40 }

/-- Theorem stating the correctness of the farm field problem solution -/
theorem farm_field_correct (f : FarmField) : 
  f.daily_plan * f.planned_days = f.total_area ∧
  f.actual_daily * (f.planned_days + f.extra_days) + f.remaining = f.total_area ∧
  f = farm_field_solution := by
  sorry

#check farm_field_correct

end NUMINAMATH_CALUDE_farm_field_correct_l985_98564


namespace NUMINAMATH_CALUDE_same_color_probability_l985_98556

def total_marbles : ℕ := 9
def marbles_per_color : ℕ := 3
def num_draws : ℕ := 3

theorem same_color_probability :
  let prob_same_color := (marbles_per_color / total_marbles) ^ num_draws * 3
  prob_same_color = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l985_98556


namespace NUMINAMATH_CALUDE_walter_school_allocation_l985_98527

/-- Represents Walter's work and school allocation details -/
structure WalterFinances where
  days_per_week : ℕ
  hours_per_day : ℕ
  hourly_rate : ℚ
  school_allocation : ℚ

/-- Calculates the fraction of weekly earnings allocated for schooling -/
def school_allocation_fraction (w : WalterFinances) : ℚ :=
  w.school_allocation / (w.days_per_week * w.hours_per_day * w.hourly_rate)

/-- Theorem stating that Walter allocates 3/4 of his weekly earnings for schooling -/
theorem walter_school_allocation :
  let w : WalterFinances := {
    days_per_week := 5,
    hours_per_day := 4,
    hourly_rate := 5,
    school_allocation := 75
  }
  school_allocation_fraction w = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_walter_school_allocation_l985_98527


namespace NUMINAMATH_CALUDE_two_times_choose_six_two_l985_98558

theorem two_times_choose_six_two : 2 * (Nat.choose 6 2) = 30 := by
  sorry

end NUMINAMATH_CALUDE_two_times_choose_six_two_l985_98558


namespace NUMINAMATH_CALUDE_john_finishes_ahead_l985_98521

/-- The distance John finishes ahead of Steve in a race --/
def distance_john_ahead (john_speed steve_speed initial_distance push_time : ℝ) : ℝ :=
  (john_speed * push_time - initial_distance) - (steve_speed * push_time)

/-- Theorem stating that John finishes 2 meters ahead of Steve --/
theorem john_finishes_ahead :
  let john_speed : ℝ := 4.2
  let steve_speed : ℝ := 3.7
  let initial_distance : ℝ := 12
  let push_time : ℝ := 28
  distance_john_ahead john_speed steve_speed initial_distance push_time = 2 := by
sorry


end NUMINAMATH_CALUDE_john_finishes_ahead_l985_98521


namespace NUMINAMATH_CALUDE_estimate_sqrt_difference_l985_98586

theorem estimate_sqrt_difference (ε : Real) (h : ε > 0) : 
  |Real.sqrt 58 - Real.sqrt 55 - 0.20| < ε :=
sorry

end NUMINAMATH_CALUDE_estimate_sqrt_difference_l985_98586


namespace NUMINAMATH_CALUDE_rectangular_field_width_l985_98587

/-- 
Given a rectangular field where the length is 7/5 of its width and the perimeter is 360 meters,
prove that the width of the field is 75 meters.
-/
theorem rectangular_field_width (width length perimeter : ℝ) : 
  length = (7/5) * width → 
  perimeter = 2 * length + 2 * width → 
  perimeter = 360 → 
  width = 75 := by sorry

end NUMINAMATH_CALUDE_rectangular_field_width_l985_98587


namespace NUMINAMATH_CALUDE_estimate_pi_l985_98531

theorem estimate_pi (total_points : ℕ) (circle_points : ℕ) 
  (h1 : total_points = 1000) 
  (h2 : circle_points = 780) : 
  (circle_points : ℚ) / total_points * 4 = 78 / 25 := by
  sorry

end NUMINAMATH_CALUDE_estimate_pi_l985_98531


namespace NUMINAMATH_CALUDE_probability_more_heads_l985_98534

/-- 
Given two players A and B, where A flips a fair coin n+1 times and B flips a fair coin n times,
this theorem states that the probability of A having more heads than B is 1/2.
-/
theorem probability_more_heads (n : ℕ) : ℝ := by
  sorry

#check probability_more_heads

end NUMINAMATH_CALUDE_probability_more_heads_l985_98534


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l985_98550

def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

theorem intersection_of_M_and_N : M ∩ N = {2, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l985_98550


namespace NUMINAMATH_CALUDE_book_problem_solution_l985_98567

/-- Represents the cost and quantity relationships between two types of books -/
structure BookProblem where
  cost_diff : ℕ             -- Cost difference between type B and type A
  total_cost_A : ℕ          -- Total cost for type A books
  total_cost_B : ℕ          -- Total cost for type B books
  total_books : ℕ           -- Total number of books to purchase
  max_total_cost : ℕ        -- Maximum total cost allowed

/-- Calculates the cost of type A books given the problem parameters -/
def cost_A (p : BookProblem) : ℕ :=
  p.total_cost_A * p.total_cost_B / (p.total_cost_B - p.total_cost_A * p.cost_diff)

/-- Calculates the cost of type B books given the problem parameters -/
def cost_B (p : BookProblem) : ℕ :=
  cost_A p + p.cost_diff

/-- Calculates the minimum number of type A books to purchase -/
def min_books_A (p : BookProblem) : ℕ :=
  (p.total_books * cost_B p - p.max_total_cost) / (cost_B p - cost_A p)

/-- Theorem stating the solution to the book purchasing problem -/
theorem book_problem_solution (p : BookProblem) 
  (h : p = { cost_diff := 20, total_cost_A := 540, total_cost_B := 780, 
             total_books := 70, max_total_cost := 3550 }) : 
  cost_A p = 45 ∧ cost_B p = 65 ∧ min_books_A p = 50 := by
  sorry

end NUMINAMATH_CALUDE_book_problem_solution_l985_98567


namespace NUMINAMATH_CALUDE_birds_cannot_all_be_in_same_nest_l985_98541

/-- Represents the color of a nest -/
inductive NestColor
| Red
| Blue

/-- Represents the state of the nests -/
structure NestState :=
  (red_birds : Nat)
  (blue_birds : Nat)

/-- The initial state of the nests -/
def initial_state : NestState :=
  { red_birds := 3, blue_birds := 3 }

/-- The state transition function representing one night's movement -/
def night_transition (state : NestState) : NestState :=
  { red_birds := state.blue_birds, blue_birds := state.red_birds }

/-- Predicate to check if all birds are in the same nest -/
def all_birds_in_same_nest (state : NestState) : Prop :=
  state.red_birds = 6 ∨ state.blue_birds = 6

/-- Theorem stating that it's impossible for all birds to end up in the same nest -/
theorem birds_cannot_all_be_in_same_nest :
  ∀ (n : Nat), ¬(all_birds_in_same_nest ((night_transition^[n]) initial_state)) :=
sorry

end NUMINAMATH_CALUDE_birds_cannot_all_be_in_same_nest_l985_98541


namespace NUMINAMATH_CALUDE_product_last_two_digits_not_consecutive_l985_98545

theorem product_last_two_digits_not_consecutive (a b c : ℕ) : 
  ¬ (∃ n : ℕ, 
    (10 ≤ n ∧ n < 100) ∧
    (ab % 100 = n ∧ ac % 100 = n + 1 ∧ bc % 100 = n + 2) ∨
    (ab % 100 = n ∧ bc % 100 = n + 1 ∧ ac % 100 = n + 2) ∨
    (ac % 100 = n ∧ ab % 100 = n + 1 ∧ bc % 100 = n + 2) ∨
    (ac % 100 = n ∧ bc % 100 = n + 1 ∧ ab % 100 = n + 2) ∨
    (bc % 100 = n ∧ ab % 100 = n + 1 ∧ ac % 100 = n + 2) ∨
    (bc % 100 = n ∧ ac % 100 = n + 1 ∧ ab % 100 = n + 2)) :=
by
  sorry

end NUMINAMATH_CALUDE_product_last_two_digits_not_consecutive_l985_98545


namespace NUMINAMATH_CALUDE_total_fish_l985_98513

/-- The number of fish Billy has -/
def billy : ℕ := 10

/-- The number of fish Tony has -/
def tony : ℕ := 3 * billy

/-- The number of fish Sarah has -/
def sarah : ℕ := tony + 5

/-- The number of fish Bobby has -/
def bobby : ℕ := 2 * sarah

/-- The total number of fish all 4 people have -/
def total : ℕ := billy + tony + sarah + bobby

theorem total_fish : total = 145 := by sorry

end NUMINAMATH_CALUDE_total_fish_l985_98513


namespace NUMINAMATH_CALUDE_temperature_difference_l985_98514

def highest_temp : ℝ := 8
def lowest_temp : ℝ := -2

theorem temperature_difference : highest_temp - lowest_temp = 10 := by
  sorry

end NUMINAMATH_CALUDE_temperature_difference_l985_98514


namespace NUMINAMATH_CALUDE_angle_range_given_monotonic_function_l985_98583

/-- Given two vectors a and b in a real inner product space, 
    with |a| = 2√2|b| ≠ 0 and f(x) = 2x³ + 3|a|x² + 6(a · b)x + 7 
    monotonically increasing on ℝ, prove that the angle θ between 
    a and b satisfies 0 ≤ θ ≤ π/4 -/
theorem angle_range_given_monotonic_function 
  {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (a b : V) (h1 : ‖a‖ = 2 * Real.sqrt 2 * ‖b‖) 
  (h2 : ‖b‖ ≠ 0) 
  (h3 : Monotone (fun x : ℝ => 2 * x^3 + 3 * ‖a‖ * x^2 + 6 * inner a b * x + 7)) :
  let θ := Real.arccos (inner a b / (‖a‖ * ‖b‖))
  0 ≤ θ ∧ θ ≤ π/4 := by
  sorry

end NUMINAMATH_CALUDE_angle_range_given_monotonic_function_l985_98583

import Mathlib

namespace NUMINAMATH_CALUDE_dice_probability_l393_39328

/-- The number of faces on each die -/
def num_faces : ℕ := 6

/-- The number of dice rolled -/
def num_dice : ℕ := 6

/-- The total number of possible outcomes when rolling the dice -/
def total_outcomes : ℕ := num_faces ^ num_dice

/-- The number of favorable outcomes (at least one pair but not a three-of-a-kind) -/
def favorable_outcomes : ℕ := 27000

/-- The probability of rolling at least one pair but not a three-of-a-kind -/
def probability : ℚ := favorable_outcomes / total_outcomes

theorem dice_probability : probability = 625 / 1089 := by
  sorry

end NUMINAMATH_CALUDE_dice_probability_l393_39328


namespace NUMINAMATH_CALUDE_constant_function_from_parallel_tangent_l393_39313

-- Define a function f: ℝ → ℝ
variable (f : ℝ → ℝ)

-- Define the property that the tangent line is parallel to the x-axis at every point
def tangent_parallel_to_x_axis (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, deriv f x = 0

-- Theorem statement
theorem constant_function_from_parallel_tangent :
  tangent_parallel_to_x_axis f → ∃ c : ℝ, ∀ x : ℝ, f x = c :=
by
  sorry

end NUMINAMATH_CALUDE_constant_function_from_parallel_tangent_l393_39313


namespace NUMINAMATH_CALUDE_distance_A_to_y_axis_l393_39370

def point_A : ℝ × ℝ := (-2, 1)

def distance_to_y_axis (p : ℝ × ℝ) : ℝ := |p.1|

theorem distance_A_to_y_axis :
  distance_to_y_axis point_A = 2 := by sorry

end NUMINAMATH_CALUDE_distance_A_to_y_axis_l393_39370


namespace NUMINAMATH_CALUDE_dolphin_population_estimate_l393_39341

/-- Estimate the number of dolphins in a coastal area on January 1st -/
theorem dolphin_population_estimate (tagged_initial : ℕ) (captured_june : ℕ) (tagged_june : ℕ)
  (migration_rate : ℚ) (new_arrival_rate : ℚ) :
  tagged_initial = 100 →
  captured_june = 90 →
  tagged_june = 4 →
  migration_rate = 1/5 →
  new_arrival_rate = 1/2 →
  ∃ (initial_population : ℕ), initial_population = 1125 :=
by sorry

end NUMINAMATH_CALUDE_dolphin_population_estimate_l393_39341


namespace NUMINAMATH_CALUDE_yogurt_combinations_yogurt_shop_combinations_l393_39311

theorem yogurt_combinations (n : ℕ) (k : ℕ) : n ≥ k → (n.choose k) = n.factorial / (k.factorial * (n - k).factorial) := by sorry

theorem yogurt_shop_combinations : 
  (5 : ℕ) * ((7 : ℕ).choose 3) = 175 := by sorry

end NUMINAMATH_CALUDE_yogurt_combinations_yogurt_shop_combinations_l393_39311


namespace NUMINAMATH_CALUDE_gcd_factorial_problem_l393_39358

theorem gcd_factorial_problem : Nat.gcd (Nat.factorial 7) ((Nat.factorial 10) / (Nat.factorial 4)) = 5040 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_problem_l393_39358


namespace NUMINAMATH_CALUDE_spring_center_max_height_l393_39337

/-- The maximum height reached by the center of a spring connecting two identical masses -/
theorem spring_center_max_height 
  (m : ℝ) -- mass of each object
  (g : ℝ) -- acceleration due to gravity
  (V₁ V₂ : ℝ) -- initial velocities of upper and lower masses
  (α β : ℝ) -- angles of initial velocities with respect to horizontal
  (h : ℝ) -- maximum height reached by the center of the spring
  (h_pos : 0 < h) -- height is positive
  (m_pos : 0 < m) -- mass is positive
  (g_pos : 0 < g) -- gravity is positive
  : h = (1 / (2 * g)) * ((V₁ * Real.sin β + V₂ * Real.sin α) / 2)^2 :=
by sorry

end NUMINAMATH_CALUDE_spring_center_max_height_l393_39337


namespace NUMINAMATH_CALUDE_smallest_x_congruence_and_divisible_l393_39325

theorem smallest_x_congruence_and_divisible (x : ℕ) : x = 45 ↔ 
  (x > 0 ∧ 
   (x + 6721) % 12 = 3458 % 12 ∧ 
   x % 5 = 0 ∧
   ∀ y : ℕ, y > 0 → (y + 6721) % 12 = 3458 % 12 → y % 5 = 0 → y ≥ x) :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_congruence_and_divisible_l393_39325


namespace NUMINAMATH_CALUDE_distinct_triangles_count_l393_39386

/-- The number of vertices in our geometric solid -/
def n : ℕ := 10

/-- The number of vertices needed to form a triangle -/
def k : ℕ := 3

/-- Combination function -/
def combination (n k : ℕ) : ℕ := 
  Nat.choose n k

theorem distinct_triangles_count : combination n k = 120 := by
  sorry

end NUMINAMATH_CALUDE_distinct_triangles_count_l393_39386


namespace NUMINAMATH_CALUDE_harkamal_fruit_purchase_l393_39330

/-- Calculates the final amount paid after discount for a fruit purchase --/
def final_amount_paid (grapes_kg : ℝ) (grapes_price : ℝ) (mangoes_kg : ℝ) (mangoes_price : ℝ)
  (oranges_kg : ℝ) (oranges_price : ℝ) (bananas_kg : ℝ) (bananas_price : ℝ) (discount_rate : ℝ) : ℝ :=
  let total_cost := grapes_kg * grapes_price + mangoes_kg * mangoes_price +
                    oranges_kg * oranges_price + bananas_kg * bananas_price
  let discount := discount_rate * total_cost
  total_cost - discount

/-- Theorem stating the final amount paid for Harkamal's fruit purchase --/
theorem harkamal_fruit_purchase :
  final_amount_paid 3 70 9 55 5 40 7 20 0.1 = 940.5 := by
  sorry

end NUMINAMATH_CALUDE_harkamal_fruit_purchase_l393_39330


namespace NUMINAMATH_CALUDE_point_translation_point_translation_proof_l393_39354

/-- Given a point B with coordinates (-5, 1), moving it 4 units right and 2 units up
    results in a point B' with coordinates (-1, 3). -/
theorem point_translation : ℝ × ℝ → ℝ × ℝ → Prop :=
  fun B B' => B = (-5, 1) → B' = (B.1 + 4, B.2 + 2) → B' = (-1, 3)

/-- The proof of the theorem. -/
theorem point_translation_proof : point_translation (-5, 1) (-1, 3) := by
  sorry

end NUMINAMATH_CALUDE_point_translation_point_translation_proof_l393_39354


namespace NUMINAMATH_CALUDE_subsets_with_sum_2008_l393_39366

def set_63 : Finset ℕ := Finset.range 64 \ {0}

theorem subsets_with_sum_2008 : 
  (Finset.filter (fun S => S.sum id = 2008) (Finset.powerset set_63)).card = 6 := by
  sorry

end NUMINAMATH_CALUDE_subsets_with_sum_2008_l393_39366


namespace NUMINAMATH_CALUDE_jacket_price_restoration_l393_39303

theorem jacket_price_restoration (initial_price : ℝ) (h_pos : initial_price > 0) :
  let price_after_first_reduction := initial_price * (1 - 0.15)
  let price_after_second_reduction := price_after_first_reduction * (1 - 0.30)
  let required_increase := (initial_price / price_after_second_reduction) - 1
  abs (required_increase - 0.6807) < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_jacket_price_restoration_l393_39303


namespace NUMINAMATH_CALUDE_consecutive_card_picks_standard_deck_l393_39332

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (num_suits : ℕ)
  (cards_per_suit : ℕ)
  (face_cards_per_suit : ℕ)
  (number_cards_per_suit : ℕ)

/-- Calculates the number of ways to pick two consecutive cards from the same suit,
    where one is a face card and the other is a number card -/
def consecutive_card_picks (d : Deck) : ℕ :=
  d.num_suits * (d.face_cards_per_suit * d.number_cards_per_suit * 2)

/-- Theorem stating that for a standard deck, there are 240 ways to pick two consecutive
    cards from the same suit, where one is a face card and the other is a number card -/
theorem consecutive_card_picks_standard_deck :
  let d : Deck := {
    total_cards := 48,
    num_suits := 4,
    cards_per_suit := 12,
    face_cards_per_suit := 3,
    number_cards_per_suit := 10
  }
  consecutive_card_picks d = 240 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_card_picks_standard_deck_l393_39332


namespace NUMINAMATH_CALUDE_function_property_l393_39377

theorem function_property (a : ℝ) : 
  (∀ x ∈ Set.Icc (0 : ℝ) 1, 
    ∀ y ∈ Set.Icc (0 : ℝ) 1, 
    ∀ z ∈ Set.Icc (0 : ℝ) 1, 
    (1/2) * a * x^2 - (x - 1) * Real.exp x + 
    (1/2) * a * y^2 - (y - 1) * Real.exp y ≥ 
    (1/2) * a * z^2 - (z - 1) * Real.exp z) →
  a ∈ Set.Icc 1 4 := by
sorry

end NUMINAMATH_CALUDE_function_property_l393_39377


namespace NUMINAMATH_CALUDE_middle_digit_is_zero_l393_39352

/-- Represents a three-digit number in a given base -/
structure ThreeDigitNumber (base : ℕ) where
  hundreds : ℕ
  tens : ℕ
  ones : ℕ
  valid_digits : hundreds < base ∧ tens < base ∧ ones < base

/-- The value of a three-digit number in decimal -/
def ThreeDigitNumber.value {base : ℕ} (n : ThreeDigitNumber base) : ℕ :=
  n.hundreds * base^2 + n.tens * base + n.ones

/-- A number M that satisfies the problem conditions -/
structure M where
  base5 : ThreeDigitNumber 5
  base8 : ThreeDigitNumber 8
  reversed_in_base8 : base8.hundreds = base5.ones ∧
                      base8.tens = base5.tens ∧
                      base8.ones = base5.hundreds
  same_value : base5.value = base8.value

theorem middle_digit_is_zero (m : M) : m.base5.tens = 0 := by
  sorry

end NUMINAMATH_CALUDE_middle_digit_is_zero_l393_39352


namespace NUMINAMATH_CALUDE_four_of_a_kind_count_l393_39374

/-- Represents a standard deck of 52 playing cards --/
def Deck : Type := Fin 52

/-- Represents a 5-card hand --/
def Hand : Type := Finset Deck

/-- Returns true if a hand contains exactly four cards of the same value --/
def hasFourOfAKind (h : Hand) : Prop := sorry

/-- The number of 5-card hands containing exactly four cards of the same value --/
def numHandsWithFourOfAKind : ℕ := sorry

theorem four_of_a_kind_count : numHandsWithFourOfAKind = 624 := by sorry

end NUMINAMATH_CALUDE_four_of_a_kind_count_l393_39374


namespace NUMINAMATH_CALUDE_last_triangle_perimeter_l393_39315

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℚ
  b : ℚ
  c : ℚ

/-- Checks if the given side lengths form a valid triangle -/
def Triangle.isValid (t : Triangle) : Prop :=
  t.a + t.b > t.c ∧ t.b + t.c > t.a ∧ t.c + t.a > t.b

/-- Generates the next triangle in the sequence -/
def nextTriangle (t : Triangle) : Triangle :=
  { a := (t.b + t.c - t.a) / 2,
    b := (t.c + t.a - t.b) / 2,
    c := (t.a + t.b - t.c) / 2 }

/-- The initial triangle T₁ -/
def T₁ : Triangle := { a := 401, b := 403, c := 405 }

/-- The sequence of triangles -/
def triangleSequence : ℕ → Triangle
  | 0 => T₁
  | n + 1 => nextTriangle (triangleSequence n)

/-- The perimeter of a triangle -/
def Triangle.perimeter (t : Triangle) : ℚ := t.a + t.b + t.c

theorem last_triangle_perimeter :
  ∃ n : ℕ, 
    (Triangle.isValid (triangleSequence n)) ∧ 
    ¬(Triangle.isValid (triangleSequence (n + 1))) ∧
    (Triangle.perimeter (triangleSequence n) = 1209 / 512) := by
  sorry

#check last_triangle_perimeter

end NUMINAMATH_CALUDE_last_triangle_perimeter_l393_39315


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l393_39351

theorem quadratic_real_roots (a : ℝ) :
  (∃ x : ℝ, a * x^2 + 2 * x - 1 = 0) ↔ (a ≥ -1 ∧ a ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l393_39351


namespace NUMINAMATH_CALUDE_biased_coin_probability_l393_39393

def coin_prob (n : Nat) : ℚ :=
  match n with
  | 1 => 3/4
  | 2 => 1/2
  | 3 => 1/4
  | 4 => 1/3
  | 5 => 2/3
  | 6 => 3/5
  | 7 => 4/7
  | _ => 0

theorem biased_coin_probability :
  (coin_prob 1 * coin_prob 2 * (1 - coin_prob 3) * (1 - coin_prob 4) *
   (1 - coin_prob 5) * (1 - coin_prob 6) * (1 - coin_prob 7)) = 3/560 := by
  sorry

end NUMINAMATH_CALUDE_biased_coin_probability_l393_39393


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l393_39396

theorem sqrt_equation_solution (x : ℝ) : 
  Real.sqrt (x - 4) = 10 → x = 104 := by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l393_39396


namespace NUMINAMATH_CALUDE_sqrt_f_squared_2009_l393_39327

-- Define the function f with the given property
axiom f : ℝ → ℝ
axiom f_property : ∀ a b : ℝ, f (a * f b) = a * b

-- State the theorem to be proved
theorem sqrt_f_squared_2009 : Real.sqrt (f 2009 ^ 2) = 2009 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_f_squared_2009_l393_39327


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_inequality_l393_39305

theorem unique_solution_quadratic_inequality (a : ℝ) :
  (∃! x : ℝ, |x^2 + 3*a*x + 4*a| ≤ 3) ↔ (a = 8 + 2*Real.sqrt 13 ∨ a = 8 - 2*Real.sqrt 13) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_inequality_l393_39305


namespace NUMINAMATH_CALUDE_polygon_line_theorem_l393_39339

/-- A polygon is represented as a set of points in the plane. -/
def Polygon : Type := Set (ℝ × ℝ)

/-- A line in the plane. -/
def Line : Type := Set (ℝ × ℝ)

/-- The number of sides in a polygon. -/
def numSides (p : Polygon) : ℕ := sorry

/-- A side of a polygon is contained in a line. -/
def sideInLine (p : Polygon) (l : Line) : Prop := sorry

/-- A line contains exactly one side of a polygon. -/
def lineContainsExactlyOneSide (p : Polygon) (l : Line) : Prop := sorry

theorem polygon_line_theorem :
  (∀ p : Polygon, numSides p = 13 → ∃ l : Line, lineContainsExactlyOneSide p l) ∧
  (∀ n : ℕ, n > 13 → ∃ p : Polygon, numSides p = n ∧ 
    ∀ l : Line, sideInLine p l → ∃ l' : Line, l ≠ l' ∧ sideInLine p l') :=
sorry

end NUMINAMATH_CALUDE_polygon_line_theorem_l393_39339


namespace NUMINAMATH_CALUDE_largest_guaranteed_divisor_l393_39363

def die_numbers : Finset Nat := {1, 2, 3, 4, 5, 6}

def is_valid_product (P : Nat) : Prop :=
  ∃ (S : Finset Nat), S ⊆ die_numbers ∧ S.card = 5 ∧ P = S.prod id

theorem largest_guaranteed_divisor :
  ∀ P, is_valid_product P → (12 ∣ P) ∧ ∀ n, n > 12 → ¬∀ Q, is_valid_product Q → (n ∣ Q) :=
by sorry

end NUMINAMATH_CALUDE_largest_guaranteed_divisor_l393_39363


namespace NUMINAMATH_CALUDE_ball_color_equality_l393_39304

theorem ball_color_equality (r g b : ℕ) : 
  (r + g + b = 20) →
  (b ≥ 7) →
  (r ≥ 4) →
  (b = 2 * g) →
  (r = b ∨ r = g) :=
by sorry

end NUMINAMATH_CALUDE_ball_color_equality_l393_39304


namespace NUMINAMATH_CALUDE_intersection_with_complement_l393_39372

def U : Finset Nat := {0, 1, 2, 3, 4}
def A : Finset Nat := {0, 1, 2, 3}
def B : Finset Nat := {0, 2, 4}

theorem intersection_with_complement :
  A ∩ (U \ B) = {1, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l393_39372


namespace NUMINAMATH_CALUDE_largest_divisible_n_l393_39331

theorem largest_divisible_n : ∃ (n : ℕ), 
  (∀ (m : ℕ), m > n → ¬((m + 20) ∣ (m^3 - 100))) ∧ 
  ((n + 20) ∣ (n^3 - 100)) ∧ 
  n = 2080 := by
  sorry

end NUMINAMATH_CALUDE_largest_divisible_n_l393_39331


namespace NUMINAMATH_CALUDE_concert_attendance_l393_39387

theorem concert_attendance (total_students : ℕ) (total_attendees : ℕ)
  (h_total : total_students = 1500)
  (h_attendees : total_attendees = 900) :
  ∃ (girls boys girls_attended : ℕ),
    girls + boys = total_students ∧
    (3 * girls + 2 * boys = 5 * total_attendees) ∧
    girls_attended = 643 ∧
    4 * girls_attended = 3 * girls :=
by sorry

end NUMINAMATH_CALUDE_concert_attendance_l393_39387


namespace NUMINAMATH_CALUDE_adjacent_complementary_implies_complementary_l393_39367

/-- Two angles are adjacent if they share a common vertex and a common side. -/
def adjacent_angles (α β : Real) : Prop := sorry

/-- Two angles are complementary if their measures add up to 90 degrees. -/
def complementary_angles (α β : Real) : Prop := α + β = 90

theorem adjacent_complementary_implies_complementary 
  (α β : Real) (h1 : adjacent_angles α β) (h2 : complementary_angles α β) : 
  complementary_angles α β :=
sorry

end NUMINAMATH_CALUDE_adjacent_complementary_implies_complementary_l393_39367


namespace NUMINAMATH_CALUDE_stating_equal_probability_for_all_methods_l393_39300

/-- Represents a sampling method -/
inductive SamplingMethod
  | Random
  | Systematic
  | Stratified

/-- The total number of components -/
def total_components : ℕ := 100

/-- The number of items to be sampled -/
def sample_size : ℕ := 20

/-- The number of first-grade items -/
def first_grade : ℕ := 20

/-- The number of second-grade items -/
def second_grade : ℕ := 30

/-- The number of third-grade items -/
def third_grade : ℕ := 50

/-- The probability of selecting any individual component -/
def selection_probability : ℚ := 1 / 5

/-- 
  Theorem stating that for all sampling methods, 
  the probability of selecting any individual component is 1/5
-/
theorem equal_probability_for_all_methods (method : SamplingMethod) : 
  (selection_probability : ℚ) = 1 / 5 := by sorry

end NUMINAMATH_CALUDE_stating_equal_probability_for_all_methods_l393_39300


namespace NUMINAMATH_CALUDE_problem_one_l393_39362

theorem problem_one : 
  64.83 - 5 * (18/19 : ℚ) + 35.17 - 44 * (1/19 : ℚ) = 50 := by sorry

end NUMINAMATH_CALUDE_problem_one_l393_39362


namespace NUMINAMATH_CALUDE_unique_operation_equals_one_l393_39350

theorem unique_operation_equals_one :
  ((-3 + (-3) = 1) = False) ∧
  ((-3 - (-3) = 1) = False) ∧
  ((-3 / (-3) = 1) = True) ∧
  ((-3 * (-3) = 1) = False) := by
  sorry

end NUMINAMATH_CALUDE_unique_operation_equals_one_l393_39350


namespace NUMINAMATH_CALUDE_function_properties_l393_39316

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

def symmetric_about (f : ℝ → ℝ) (a : ℝ) : Prop := ∀ x, f (a + x) = f (a - x)

def increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

theorem function_properties (f : ℝ → ℝ) 
  (h1 : is_even f)
  (h2 : ∀ x, f (x + 1) = -f x)
  (h3 : increasing_on f (-1) 0) :
  (periodic f 2) ∧ 
  (symmetric_about f 1) ∧ 
  (f 2 = f 0) := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l393_39316


namespace NUMINAMATH_CALUDE_sector_area_l393_39344

/-- Given an arc length of 4 cm corresponding to a central angle of 2 radians,
    the area of the sector enclosed by this central angle is 4 cm². -/
theorem sector_area (arc_length : ℝ) (central_angle : ℝ) (h1 : arc_length = 4) (h2 : central_angle = 2) :
  let radius := arc_length / central_angle
  let sector_area := (1 / 2) * radius^2 * central_angle
  sector_area = 4 := by
sorry

end NUMINAMATH_CALUDE_sector_area_l393_39344


namespace NUMINAMATH_CALUDE_twins_age_problem_l393_39307

theorem twins_age_problem (age : ℕ) : 
  (age + 1) * (age + 1) = age * age + 5 → age = 2 := by
sorry

end NUMINAMATH_CALUDE_twins_age_problem_l393_39307


namespace NUMINAMATH_CALUDE_square_fold_perimeter_l393_39338

/-- Given a square ABCD with side length 2, where C is folded to meet AD at point C' 
    such that C'D = 2/3, and BC intersects AB at point E, 
    the perimeter of triangle AEC' is (17√10 + √37) / 12 -/
theorem square_fold_perimeter (A B C D C' E : ℝ × ℝ) : 
  let square_side : ℝ := 2
  let C'D : ℝ := 2/3
  -- Define the square
  square_side = ‖A - B‖ ∧ square_side = ‖B - C‖ ∧ 
  square_side = ‖C - D‖ ∧ square_side = ‖D - A‖ ∧
  -- C' is on AD
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ C' = t • A + (1 - t) • D ∧
  -- E is on AB and BC
  ∃ s r : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ 0 ≤ r ∧ r ≤ 1 ∧ 
  E = s • A + (1 - s) • B ∧ E = r • B + (1 - r) • C ∧
  -- C'D condition
  ‖C' - D‖ = C'D
  →
  ‖A - E‖ + ‖E - C'‖ + ‖C' - A‖ = (17 * Real.sqrt 10 + Real.sqrt 37) / 12 :=
by sorry


end NUMINAMATH_CALUDE_square_fold_perimeter_l393_39338


namespace NUMINAMATH_CALUDE_equal_cost_at_250_minutes_plan_a_cheaper_at_300_minutes_l393_39322

/-- Represents the monthly fee for a phone plan given the call duration -/
structure PhonePlan where
  fixed_rental : ℝ
  per_minute_rate : ℝ
  monthly_fee : ℝ → ℝ

/-- Plan A with 50 yuan fixed rental and 0.4 yuan per minute -/
def plan_a : PhonePlan :=
  { fixed_rental := 50
    per_minute_rate := 0.4
    monthly_fee := λ x => 50 + 0.4 * x }

/-- Plan B with no fixed rental and 0.6 yuan per minute -/
def plan_b : PhonePlan :=
  { fixed_rental := 0
    per_minute_rate := 0.6
    monthly_fee := λ x => 0.6 * x }

/-- The call duration at which both plans have the same cost -/
def equal_cost_duration : ℝ := 250

theorem equal_cost_at_250_minutes :
  plan_a.monthly_fee equal_cost_duration = plan_b.monthly_fee equal_cost_duration :=
by sorry

theorem plan_a_cheaper_at_300_minutes :
  plan_a.monthly_fee 300 < plan_b.monthly_fee 300 :=
by sorry

end NUMINAMATH_CALUDE_equal_cost_at_250_minutes_plan_a_cheaper_at_300_minutes_l393_39322


namespace NUMINAMATH_CALUDE_sasha_muffins_count_l393_39309

/-- The number of muffins Sasha made -/
def sasha_muffins : ℕ := 50

/-- The number of muffins Melissa made -/
def melissa_muffins : ℕ := 4 * sasha_muffins

/-- The number of muffins Tiffany made -/
def tiffany_muffins : ℕ := (sasha_muffins + melissa_muffins) / 2

/-- The total number of muffins made -/
def total_muffins : ℕ := sasha_muffins + melissa_muffins + tiffany_muffins

/-- The price of each muffin in cents -/
def muffin_price : ℕ := 400

/-- The total amount raised in cents -/
def total_raised : ℕ := 90000

theorem sasha_muffins_count : 
  sasha_muffins = 50 ∧ 
  melissa_muffins = 4 * sasha_muffins ∧
  tiffany_muffins = (sasha_muffins + melissa_muffins) / 2 ∧
  total_muffins * muffin_price = total_raised := by
  sorry

end NUMINAMATH_CALUDE_sasha_muffins_count_l393_39309


namespace NUMINAMATH_CALUDE_fraction_denominator_expression_l393_39356

theorem fraction_denominator_expression 
  (x y a b : ℝ) 
  (h1 : x / y = 3) 
  (h2 : (2 * a - x) / (3 * b - y) = 3) 
  (h3 : a / b = 4.5) : 
  ∃ (E : ℝ), (2 * a - x) / E = 3 ∧ E = 3 * b - y := by
  sorry

end NUMINAMATH_CALUDE_fraction_denominator_expression_l393_39356


namespace NUMINAMATH_CALUDE_ticket_cost_l393_39376

/-- The cost of a single ticket at the fair, given the initial number of tickets,
    remaining tickets, and total amount spent on the ferris wheel. -/
theorem ticket_cost (initial_tickets : ℕ) (remaining_tickets : ℕ) (total_spent : ℕ) :
  initial_tickets > remaining_tickets →
  total_spent % (initial_tickets - remaining_tickets) = 0 →
  total_spent / (initial_tickets - remaining_tickets) = 9 :=
by
  intro h_tickets h_divisible
  sorry

#check ticket_cost 13 4 81

end NUMINAMATH_CALUDE_ticket_cost_l393_39376


namespace NUMINAMATH_CALUDE_triangle_perimeter_l393_39382

def is_odd (n : ℕ) : Prop := ∃ k, n = 2 * k + 1

theorem triangle_perimeter (a b c : ℕ) :
  a = 2 → b = 5 → is_odd c → a + b > c → b + c > a → c + a > b →
  a + b + c = 12 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l393_39382


namespace NUMINAMATH_CALUDE_incorrect_propositions_l393_39398

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A line in 3D space -/
structure Line3D where
  point : Point3D
  direction : Point3D

/-- A plane in 3D space -/
structure Plane3D where
  point : Point3D
  normal : Point3D

/-- Two lines intersect -/
def intersect (l1 l2 : Line3D) : Prop := sorry

/-- A point lies on a line -/
def point_on_line (p : Point3D) (l : Line3D) : Prop := sorry

/-- A point lies on a plane -/
def point_on_plane (p : Point3D) (plane : Plane3D) : Prop := sorry

/-- Two lines are perpendicular -/
def perpendicular (l1 l2 : Line3D) : Prop := sorry

/-- Two lines are skew (not parallel and not intersecting) -/
def skew (l1 l2 : Line3D) : Prop := sorry

/-- A line lies on a plane -/
def line_on_plane (l : Line3D) (p : Plane3D) : Prop := sorry

/-- Three points determine a plane -/
def points_determine_plane (p1 p2 p3 : Point3D) : Prop := sorry

theorem incorrect_propositions :
  -- Proposition ③: Three points on two intersecting lines determine a plane
  ¬ (∀ (l1 l2 : Line3D) (p1 p2 p3 : Point3D),
    intersect l1 l2 →
    point_on_line p1 l1 →
    point_on_line p2 l1 →
    point_on_line p3 l2 →
    points_determine_plane p1 p2 p3) ∧
  -- Proposition ④: Two perpendicular lines are coplanar
  ¬ (∀ (l1 l2 : Line3D),
    perpendicular l1 l2 →
    ∃ (p : Plane3D), line_on_plane l1 p ∧ line_on_plane l2 p) :=
by sorry

end NUMINAMATH_CALUDE_incorrect_propositions_l393_39398


namespace NUMINAMATH_CALUDE_square_area_equals_36_l393_39383

theorem square_area_equals_36 (square_perimeter triangle_perimeter : ℝ) 
  (triangle_side1 triangle_side2 triangle_side3 : ℝ) :
  triangle_side1 = 5.5 →
  triangle_side2 = 7.5 →
  triangle_side3 = 11 →
  triangle_perimeter = triangle_side1 + triangle_side2 + triangle_side3 →
  square_perimeter = triangle_perimeter →
  (square_perimeter / 4) ^ 2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_square_area_equals_36_l393_39383


namespace NUMINAMATH_CALUDE_exists_positive_value_for_expression_l393_39395

theorem exists_positive_value_for_expression : ∃ n : ℕ+, n.val^2 - 8*n.val + 7 > 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_positive_value_for_expression_l393_39395


namespace NUMINAMATH_CALUDE_average_book_price_l393_39333

/-- The average price of books bought from two shops -/
theorem average_book_price (books1 books2 : ℕ) (price1 price2 : ℚ) :
  books1 = 27 →
  books2 = 20 →
  price1 = 581 →
  price2 = 594 →
  (price1 + price2) / (books1 + books2 : ℚ) = 25 := by
  sorry

end NUMINAMATH_CALUDE_average_book_price_l393_39333


namespace NUMINAMATH_CALUDE_flower_shop_problem_l393_39388

/-- Given information about flower purchases and sales, prove the cost price of the first batch and minimum selling price of the second batch -/
theorem flower_shop_problem (first_batch_cost second_batch_cost : ℝ) 
  (quantity_ratio : ℝ) (price_difference : ℝ) (min_total_profit : ℝ) 
  (first_batch_selling_price : ℝ) :
  first_batch_cost = 1000 →
  second_batch_cost = 2500 →
  quantity_ratio = 2 →
  price_difference = 0.5 →
  min_total_profit = 1500 →
  first_batch_selling_price = 3 →
  ∃ (first_batch_cost_price second_batch_min_selling_price : ℝ),
    first_batch_cost_price = 2 ∧
    second_batch_min_selling_price = 3.5 ∧
    (first_batch_cost / first_batch_cost_price) * quantity_ratio = 
      second_batch_cost / (first_batch_cost_price + price_difference) ∧
    (first_batch_cost / first_batch_cost_price) * 
      (first_batch_selling_price - first_batch_cost_price) +
    (second_batch_cost / (first_batch_cost_price + price_difference)) * 
      (second_batch_min_selling_price - (first_batch_cost_price + price_difference)) ≥ 
    min_total_profit := by
  sorry

end NUMINAMATH_CALUDE_flower_shop_problem_l393_39388


namespace NUMINAMATH_CALUDE_sum_of_all_expressions_l393_39365

/-- Represents an expression formed by replacing * with + or - in 1 * 2 * 3 * 4 * 5 * 6 -/
def Expression := List (Bool × ℕ)

/-- Generates all possible expressions -/
def generateExpressions : List Expression :=
  sorry

/-- Evaluates a single expression -/
def evaluateExpression (expr : Expression) : ℤ :=
  sorry

/-- Sums the results of all expressions -/
def sumAllExpressions : ℤ :=
  (generateExpressions.map evaluateExpression).sum

theorem sum_of_all_expressions :
  sumAllExpressions = 32 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_all_expressions_l393_39365


namespace NUMINAMATH_CALUDE_polynomial_expansion_l393_39384

theorem polynomial_expansion (x : ℝ) :
  (1 + x^3) * (1 - x^4) * (1 + x^5) = 1 + x^3 - x^4 + x^5 - x^7 + x^8 - x^9 - x^12 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l393_39384


namespace NUMINAMATH_CALUDE_equal_integers_from_cyclic_equation_l393_39326

theorem equal_integers_from_cyclic_equation (n : ℕ+) (p : ℕ) (a b c : ℤ)
  (hp : Nat.Prime p)
  (h1 : a^(n : ℕ) + p * b = b^(n : ℕ) + p * c)
  (h2 : b^(n : ℕ) + p * c = c^(n : ℕ) + p * a) :
  a = b ∧ b = c := by
  sorry

end NUMINAMATH_CALUDE_equal_integers_from_cyclic_equation_l393_39326


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_division_l393_39368

theorem imaginary_part_of_complex_division (Z : ℂ) (h : Z = 1 - 2*I) :
  (Complex.im ((1 : ℂ) + 3*I) / Z) = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_division_l393_39368


namespace NUMINAMATH_CALUDE_point_trajectory_l393_39347

-- Define the condition for point M(x,y)
def point_condition (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + (y+2)^2) + Real.sqrt (x^2 + (y-2)^2) = 8

-- Define the trajectory equation
def trajectory_equation (x y : ℝ) : Prop :=
  x^2 / 12 + y^2 / 16 = 1

-- Theorem statement
theorem point_trajectory : ∀ x y : ℝ, point_condition x y → trajectory_equation x y :=
by sorry

end NUMINAMATH_CALUDE_point_trajectory_l393_39347


namespace NUMINAMATH_CALUDE_not_much_different_from_2023_l393_39373

theorem not_much_different_from_2023 (x : ℝ) : 
  (x - 2023 ≤ 0) ↔ (x ≤ 2023) :=
by sorry

end NUMINAMATH_CALUDE_not_much_different_from_2023_l393_39373


namespace NUMINAMATH_CALUDE_quadratic_function_positive_range_l393_39394

theorem quadratic_function_positive_range (a : ℝ) : 
  (∀ x : ℝ, 0 < x → x < 3 → a * x^2 - 2 * a * x + 3 > 0) ↔ 
  (-1 ≤ a ∧ a < 0) ∨ (0 < a ∧ a < 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_positive_range_l393_39394


namespace NUMINAMATH_CALUDE_chess_tournament_matches_chess_tournament_problem_l393_39302

/-- Represents a single elimination chess tournament --/
structure ChessTournament where
  total_players : ℕ
  bye_players : ℕ
  matches_played : ℕ

/-- Theorem stating the number of matches in the given tournament --/
theorem chess_tournament_matches 
  (tournament : ChessTournament) 
  (h1 : tournament.total_players = 128) 
  (h2 : tournament.bye_players = 32) : 
  tournament.matches_played = 127 := by
  sorry

/-- Main theorem to be proved --/
theorem chess_tournament_problem : 
  ∃ (t : ChessTournament), t.total_players = 128 ∧ t.bye_players = 32 ∧ t.matches_played = 127 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_matches_chess_tournament_problem_l393_39302


namespace NUMINAMATH_CALUDE_valid_K_values_l393_39312

/-- The sum of the first n positive integers -/
def triangular_sum (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Predicate for K being a valid solution -/
def is_valid_K (K : ℕ) : Prop :=
  ∃ (N : ℕ), N < 50 ∧ triangular_sum K = N^2

theorem valid_K_values :
  {K : ℕ | is_valid_K K} = {1, 8, 49} := by sorry

end NUMINAMATH_CALUDE_valid_K_values_l393_39312


namespace NUMINAMATH_CALUDE_snowfall_probability_l393_39369

theorem snowfall_probability (p_A p_B : ℝ) (h1 : p_A = 0.4) (h2 : p_B = 0.3) :
  (1 - p_A) * (1 - p_B) = 0.42 := by
  sorry

end NUMINAMATH_CALUDE_snowfall_probability_l393_39369


namespace NUMINAMATH_CALUDE_integer_root_quadratic_count_l393_39334

theorem integer_root_quadratic_count :
  ∃! (S : Finset ℝ), (∀ a ∈ S, ∃ r s : ℤ, ∀ x : ℝ, x^2 + a*x + 9*a = 0 ↔ x = r ∨ x = s) ∧ Finset.card S = 10 :=
sorry

end NUMINAMATH_CALUDE_integer_root_quadratic_count_l393_39334


namespace NUMINAMATH_CALUDE_similar_quadratic_radicals_l393_39397

def are_similar_quadratic_radicals (a b : ℝ) : Prop :=
  ∃ (k : ℚ), a = k * b

theorem similar_quadratic_radicals :
  are_similar_quadratic_radicals (Real.sqrt 18) (Real.sqrt 72) ∧
  ¬ are_similar_quadratic_radicals (Real.sqrt 12) (Real.sqrt 18) ∧
  ¬ are_similar_quadratic_radicals (Real.sqrt 20) (Real.sqrt 50) ∧
  ¬ are_similar_quadratic_radicals (Real.sqrt 24) (Real.sqrt 32) :=
by sorry

end NUMINAMATH_CALUDE_similar_quadratic_radicals_l393_39397


namespace NUMINAMATH_CALUDE_choir_singing_problem_l393_39381

theorem choir_singing_problem (total_singers : ℕ) 
  (h1 : total_singers = 30)
  (first_verse : ℕ) 
  (h2 : first_verse = total_singers / 2)
  (second_verse : ℕ)
  (h3 : second_verse = (total_singers - first_verse) / 3)
  (final_verse : ℕ)
  (h4 : final_verse = total_singers - first_verse - second_verse) :
  final_verse = 10 := by
  sorry

end NUMINAMATH_CALUDE_choir_singing_problem_l393_39381


namespace NUMINAMATH_CALUDE_museum_ticket_cost_l393_39353

theorem museum_ticket_cost : 
  ∀ (num_students num_teachers : ℕ) 
    (student_ticket_cost teacher_ticket_cost : ℕ),
  num_students = 12 →
  num_teachers = 4 →
  student_ticket_cost = 1 →
  teacher_ticket_cost = 3 →
  num_students * student_ticket_cost + num_teachers * teacher_ticket_cost = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_museum_ticket_cost_l393_39353


namespace NUMINAMATH_CALUDE_umbrella_arrangements_seven_l393_39323

def umbrella_arrangements (n : ℕ) : ℕ := 
  if n % 2 = 0 then 0
  else Nat.choose (n - 1) ((n - 1) / 2)

theorem umbrella_arrangements_seven :
  umbrella_arrangements 7 = 20 := by
sorry

end NUMINAMATH_CALUDE_umbrella_arrangements_seven_l393_39323


namespace NUMINAMATH_CALUDE_number_of_boys_in_class_l393_39391

/-- Given the conditions of a class of boys with height measurements:
    - initial_average: The initially calculated average height
    - wrong_height: The wrongly recorded height of one boy
    - correct_height: The correct height of the boy with the wrong measurement
    - actual_average: The actual average height after correction
    
    Prove that the number of boys in the class is equal to the given value.
-/
theorem number_of_boys_in_class 
  (initial_average : ℝ) 
  (wrong_height : ℝ) 
  (correct_height : ℝ) 
  (actual_average : ℝ) 
  (h1 : initial_average = 180) 
  (h2 : wrong_height = 156) 
  (h3 : correct_height = 106) 
  (h4 : actual_average = 178) : 
  ∃ n : ℕ, n * actual_average = n * initial_average - (wrong_height - correct_height) ∧ n = 25 :=
by sorry

end NUMINAMATH_CALUDE_number_of_boys_in_class_l393_39391


namespace NUMINAMATH_CALUDE_P_in_fourth_quadrant_l393_39361

/-- A point in the Cartesian coordinate system -/
structure CartesianPoint where
  x : ℝ
  y : ℝ

/-- Definition of the fourth quadrant -/
def is_in_fourth_quadrant (p : CartesianPoint) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- The given point P -/
def P : CartesianPoint :=
  { x := 1, y := -4 }

/-- Theorem: P lies in the fourth quadrant -/
theorem P_in_fourth_quadrant : is_in_fourth_quadrant P := by
  sorry

end NUMINAMATH_CALUDE_P_in_fourth_quadrant_l393_39361


namespace NUMINAMATH_CALUDE_base8_to_base10_conversion_l393_39348

/-- Converts a base-8 number to base-10 --/
def base8ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (8 ^ i)) 0

/-- The base-8 representation of the number --/
def base8Number : List Nat := [3, 4, 6, 2, 5]

theorem base8_to_base10_conversion :
  base8ToBase10 base8Number = 21923 := by
  sorry

end NUMINAMATH_CALUDE_base8_to_base10_conversion_l393_39348


namespace NUMINAMATH_CALUDE_probability_red_ball_is_four_fifths_l393_39357

/-- Represents a container with red and green balls -/
structure Container where
  red : ℕ
  green : ℕ

/-- The experiment setup -/
def experiment : List Container :=
  [{ red := 5, green := 5 },  -- Container A
   { red := 7, green := 3 },  -- Container B
   { red := 7, green := 3 }]  -- Container C

/-- The probability of selecting a red ball in the described experiment -/
def probability_red_ball : ℚ :=
  (experiment.map (fun c => c.red / (c.red + c.green))).sum / experiment.length

theorem probability_red_ball_is_four_fifths :
  probability_red_ball = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_probability_red_ball_is_four_fifths_l393_39357


namespace NUMINAMATH_CALUDE_functions_continuous_and_equal_l393_39318

/-- Darboux property (intermediate value property) -/
def has_darboux_property (f : ℝ → ℝ) : Prop :=
  ∀ a b y, a < b → f a < y → y < f b → ∃ c, a < c ∧ c < b ∧ f c = y

/-- The problem statement -/
theorem functions_continuous_and_equal
  (f g : ℝ → ℝ)
  (h1 : ∀ a, ⨅ (x > a), f x = g a)
  (h2 : ∀ a, ⨆ (x < a), g x = f a)
  (h3 : has_darboux_property f) :
  Continuous f ∧ Continuous g ∧ f = g := by
  sorry

end NUMINAMATH_CALUDE_functions_continuous_and_equal_l393_39318


namespace NUMINAMATH_CALUDE_golus_journey_l393_39375

theorem golus_journey (a b c : ℝ) (h1 : a = 8) (h2 : c = 10) (h3 : a^2 + b^2 = c^2) : b = 6 := by
  sorry

end NUMINAMATH_CALUDE_golus_journey_l393_39375


namespace NUMINAMATH_CALUDE_divisibility_property_l393_39360

theorem divisibility_property (a b c d e n : ℤ) 
  (h_odd : Odd n)
  (h_sum_div : n ∣ (a + b + c + d + e))
  (h_sum_squares_div : n ∣ (a^2 + b^2 + c^2 + d^2 + e^2)) :
  n ∣ (a^5 + b^5 + c^5 + d^5 + e^5 - 5*a*b*c*d*e) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_property_l393_39360


namespace NUMINAMATH_CALUDE_student_ticket_cost_l393_39345

/-- Proves that the cost of each student ticket is $6 given the conditions of the problem -/
theorem student_ticket_cost (adult_ticket_cost : ℕ) (num_students : ℕ) (num_adults : ℕ) (total_revenue : ℕ) :
  adult_ticket_cost = 8 →
  num_students = 20 →
  num_adults = 12 →
  total_revenue = 216 →
  ∃ (student_ticket_cost : ℕ), 
    student_ticket_cost * num_students + adult_ticket_cost * num_adults = total_revenue ∧
    student_ticket_cost = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_student_ticket_cost_l393_39345


namespace NUMINAMATH_CALUDE_class_fraction_proof_l393_39364

theorem class_fraction_proof (G : ℚ) (B : ℚ) (T : ℚ) (h1 : B / G = 3 / 2) (h2 : T = B + G) :
  (G / 2) / T = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_class_fraction_proof_l393_39364


namespace NUMINAMATH_CALUDE_descent_route_length_l393_39379

/- Define the hiking trip parameters -/
def forest_speed : ℝ := 8
def rocky_speed : ℝ := 5
def snowy_speed : ℝ := 3
def forest_time : ℝ := 1
def rocky_time : ℝ := 1
def snowy_time : ℝ := 0.5
def speed_multiplier : ℝ := 1.5
def total_days : ℝ := 2

/- Define the theorem -/
theorem descent_route_length :
  let grassland_speed := forest_speed * speed_multiplier
  let sandy_speed := rocky_speed * speed_multiplier
  let descent_distance := grassland_speed * forest_time + sandy_speed * rocky_time
  descent_distance = 19.5 := by sorry

end NUMINAMATH_CALUDE_descent_route_length_l393_39379


namespace NUMINAMATH_CALUDE_first_five_terms_of_sequence_l393_39321

def a (n : ℕ) : ℤ := (-1: ℤ)^n + n

theorem first_five_terms_of_sequence :
  (a 1 = 0) ∧ (a 2 = 3) ∧ (a 3 = 2) ∧ (a 4 = 5) ∧ (a 5 = 4) :=
by sorry

end NUMINAMATH_CALUDE_first_five_terms_of_sequence_l393_39321


namespace NUMINAMATH_CALUDE_sixth_term_of_geometric_sequence_l393_39378

-- Define a geometric sequence
def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ := a₁ * r^(n - 1)

-- Theorem statement
theorem sixth_term_of_geometric_sequence :
  ∀ (r : ℝ),
  (geometric_sequence 16 r 8 = 11664) →
  (geometric_sequence 16 r 6 = 3888) :=
by
  sorry

end NUMINAMATH_CALUDE_sixth_term_of_geometric_sequence_l393_39378


namespace NUMINAMATH_CALUDE_howlers_lineup_count_l393_39355

def total_players : Nat := 15
def lineup_size : Nat := 6
def excluded_players : Nat := 3

theorem howlers_lineup_count :
  (Nat.choose (total_players - excluded_players) lineup_size) +
  (excluded_players * Nat.choose (total_players - excluded_players) (lineup_size - 1)) = 3300 :=
by sorry

end NUMINAMATH_CALUDE_howlers_lineup_count_l393_39355


namespace NUMINAMATH_CALUDE_ball_drawing_probabilities_l393_39301

/-- Represents the color of a ball -/
inductive BallColor
| Red
| White

/-- Represents a ball with its color and number -/
structure Ball :=
  (color : BallColor)
  (number : Nat)

/-- The box of balls -/
def box : List Ball := [
  ⟨BallColor.Red, 1⟩, ⟨BallColor.Red, 2⟩, ⟨BallColor.Red, 3⟩, ⟨BallColor.Red, 4⟩,
  ⟨BallColor.White, 3⟩, ⟨BallColor.White, 4⟩
]

/-- The number of balls to draw -/
def drawCount : Nat := 3

/-- Calculates the probability of drawing 3 balls with maximum number 3 -/
def probMaxThree : ℚ := 1 / 5

/-- Calculates the mathematical expectation of the maximum number among red balls drawn -/
def expectationMaxRed : ℚ := 13 / 4

theorem ball_drawing_probabilities :
  (probMaxThree = 1 / 5) ∧
  (expectationMaxRed = 13 / 4) := by
  sorry


end NUMINAMATH_CALUDE_ball_drawing_probabilities_l393_39301


namespace NUMINAMATH_CALUDE_cubic_sum_problem_l393_39329

theorem cubic_sum_problem (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_eq : (a^3 + 12) / a = (b^3 + 12) / b ∧ (b^3 + 12) / b = (c^3 + 12) / c) : 
  a^3 + b^3 + c^3 = -36 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_problem_l393_39329


namespace NUMINAMATH_CALUDE_find_x_l393_39335

theorem find_x : ∃ x : ℝ, (85 + x / 113) * 113 = 9637 ∧ x = 9552 := by
  sorry

end NUMINAMATH_CALUDE_find_x_l393_39335


namespace NUMINAMATH_CALUDE_smallest_n_for_factorization_l393_39359

theorem smallest_n_for_factorization : 
  let can_be_factored (n : ℤ) := ∃ (A B : ℤ), 
    (A * B = 60) ∧ 
    (6 * B + A = n) ∧ 
    (∀ x, 6 * x^2 + n * x + 60 = (6 * x + A) * (x + B))
  ∀ n : ℤ, can_be_factored n → n ≥ 66
  ∧ can_be_factored 66 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_factorization_l393_39359


namespace NUMINAMATH_CALUDE_root_implies_h_value_l393_39320

theorem root_implies_h_value (h : ℝ) : 
  (3 : ℝ)^3 + h * 3 + 5 = 0 → h = -32/3 := by
  sorry

end NUMINAMATH_CALUDE_root_implies_h_value_l393_39320


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l393_39399

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h1 : a 1 = 8)
  (h2 : ∀ n : ℕ, a (n + 1) = a n * (a 2 / a 1))
  (h3 : a 4 = a 3 * a 5) :
  a 2 / a 1 = 1/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l393_39399


namespace NUMINAMATH_CALUDE_employed_males_percentage_l393_39310

theorem employed_males_percentage (total_employed_percent : Real) 
  (employed_females_percent : Real) (h1 : total_employed_percent = 64) 
  (h2 : employed_females_percent = 28.125) : 
  (total_employed_percent / 100) * (100 - employed_females_percent) = 45.96 :=
sorry

end NUMINAMATH_CALUDE_employed_males_percentage_l393_39310


namespace NUMINAMATH_CALUDE_q_div_p_equals_168_l393_39343

/-- The number of slips in the hat -/
def total_slips : ℕ := 60

/-- The number of distinct numbers on the slips -/
def distinct_numbers : ℕ := 15

/-- The number of slips drawn -/
def drawn_slips : ℕ := 5

/-- The number of slips with each number -/
def slips_per_number : ℕ := 4

/-- The probability that all drawn slips bear the same number -/
def p : ℚ := (distinct_numbers : ℚ) / Nat.choose total_slips drawn_slips

/-- The probability that three slips bear one number and two bear a different number -/
def q : ℚ := (Nat.choose distinct_numbers 2 * Nat.choose slips_per_number 3 * Nat.choose slips_per_number 2 : ℚ) / Nat.choose total_slips drawn_slips

/-- The main theorem stating the ratio of q to p -/
theorem q_div_p_equals_168 : q / p = 168 := by sorry

end NUMINAMATH_CALUDE_q_div_p_equals_168_l393_39343


namespace NUMINAMATH_CALUDE_rebate_calculation_l393_39371

theorem rebate_calculation (polo_price necklace_price game_price : ℕ)
  (polo_count necklace_count : ℕ) (total_after_rebate : ℕ) :
  polo_price = 26 →
  necklace_price = 83 →
  game_price = 90 →
  polo_count = 3 →
  necklace_count = 2 →
  total_after_rebate = 322 →
  (polo_price * polo_count + necklace_price * necklace_count + game_price) - total_after_rebate = 12 := by
  sorry

end NUMINAMATH_CALUDE_rebate_calculation_l393_39371


namespace NUMINAMATH_CALUDE_expression_evaluation_l393_39385

theorem expression_evaluation :
  let x : ℚ := -1/4
  (x - 1)^2 - 3*x*(1 - x) - (2*x - 1)*(2*x + 1) = 13/4 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l393_39385


namespace NUMINAMATH_CALUDE_people_in_house_l393_39349

theorem people_in_house : 
  ∀ (initial_bedroom : ℕ) (entering_bedroom : ℕ) (living_room : ℕ),
    initial_bedroom = 2 →
    entering_bedroom = 5 →
    living_room = 8 →
    initial_bedroom + entering_bedroom + living_room = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_people_in_house_l393_39349


namespace NUMINAMATH_CALUDE_box_surface_area_l393_39306

/-- The surface area of a rectangular parallelepiped with dimensions a, b, c -/
def surfaceArea (a b c : ℕ) : ℕ := 2 * (a * b + b * c + c * a)

theorem box_surface_area :
  ∀ a b c : ℕ,
    0 < a ∧ a < 10 →
    0 < b ∧ b < 10 →
    0 < c ∧ c < 10 →
    a * b * c = 280 →
    surfaceArea a b c = 262 := by
  sorry

end NUMINAMATH_CALUDE_box_surface_area_l393_39306


namespace NUMINAMATH_CALUDE_distance_to_origin_l393_39340

/-- The distance from point P(3,4) to the origin in the Cartesian coordinate system is 5. -/
theorem distance_to_origin : Real.sqrt (3^2 + 4^2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_origin_l393_39340


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l393_39317

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  (∀ n : ℕ, a n > 0) →
  a 6 * a 10 + a 3 * a 5 = 26 →
  a 5 * a 7 = 5 →
  a 4 + a 8 = 6 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l393_39317


namespace NUMINAMATH_CALUDE_binary_addition_theorem_l393_39336

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : Nat :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

/-- Converts a decimal number to its binary representation as a list of bits -/
def decimal_to_binary (n : Nat) : List Bool :=
  if n = 0 then [false] else
  let rec to_binary_aux (m : Nat) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: to_binary_aux (m / 2)
  to_binary_aux n

theorem binary_addition_theorem :
  let a := [true, false, true]  -- 101₂
  let b := [true, true]         -- 11₂
  let c := [false, false, true, true]  -- 1100₂
  let d := [true, false, true, true, true]  -- 11101₂
  let result := [true, false, false, false, false, true, true]  -- 110001₂
  binary_to_decimal a + binary_to_decimal b + binary_to_decimal c + binary_to_decimal d =
  binary_to_decimal result := by
  sorry

end NUMINAMATH_CALUDE_binary_addition_theorem_l393_39336


namespace NUMINAMATH_CALUDE_prime_solution_equation_l393_39319

theorem prime_solution_equation : ∃! (p q : ℕ), 
  Prime p ∧ Prime q ∧ p^2 - 6*p*q + q^2 + 3*q - 1 = 0 ∧ p = 17 ∧ q = 3 := by
  sorry

end NUMINAMATH_CALUDE_prime_solution_equation_l393_39319


namespace NUMINAMATH_CALUDE_divisible_by_24_l393_39390

theorem divisible_by_24 (n : ℕ) : ∃ k : ℤ, (n^4 : ℤ) + 2*(n^3 : ℤ) + 11*(n^2 : ℤ) + 10*(n : ℤ) = 24*k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_24_l393_39390


namespace NUMINAMATH_CALUDE_max_k_inequality_l393_39389

theorem max_k_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (∀ k : ℝ, k ≤ 6 → (2 * (a^2 + k*a*b + b^2)) / ((k+2)*(a+b)) ≥ Real.sqrt (a*b)) ∧
  (∀ ε > 0, ∃ a b : ℝ, 0 < a ∧ 0 < b ∧ (2 * (a^2 + (6+ε)*a*b + b^2)) / ((6+ε+2)*(a+b)) < Real.sqrt (a*b)) :=
by sorry

end NUMINAMATH_CALUDE_max_k_inequality_l393_39389


namespace NUMINAMATH_CALUDE_choose_from_four_and_three_l393_39342

/-- The number of ways to choose one item from each of two sets -/
def choose_one_from_each (set1_size set2_size : ℕ) : ℕ :=
  set1_size * set2_size

/-- Theorem: Choosing one item from a set of 4 and one from a set of 3 results in 12 possibilities -/
theorem choose_from_four_and_three :
  choose_one_from_each 4 3 = 12 := by
  sorry

end NUMINAMATH_CALUDE_choose_from_four_and_three_l393_39342


namespace NUMINAMATH_CALUDE_blue_pens_removed_l393_39392

/-- Represents the number of pens in a jar -/
structure JarContents where
  blue : ℕ
  black : ℕ
  red : ℕ

/-- The initial contents of the jar -/
def initial_jar : JarContents := ⟨9, 21, 6⟩

/-- The number of black pens removed -/
def black_pens_removed : ℕ := 7

/-- The final number of pens in the jar after removals -/
def final_pens : ℕ := 25

/-- Theorem stating that 4 blue pens were removed -/
theorem blue_pens_removed :
  ∃ (x : ℕ),
    x = 4 ∧
    initial_jar.blue - x +
    (initial_jar.black - black_pens_removed) +
    initial_jar.red = final_pens :=
  sorry

end NUMINAMATH_CALUDE_blue_pens_removed_l393_39392


namespace NUMINAMATH_CALUDE_find_m_l393_39314

theorem find_m (a : ℝ) (n m : ℕ) (h1 : a^n = 2) (h2 : a^(m*n) = 16) : m = 4 := by
  sorry

end NUMINAMATH_CALUDE_find_m_l393_39314


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l393_39380

-- Define a positive geometric sequence
def is_positive_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n ∧ a n > 0

-- State the theorem
theorem geometric_sequence_problem (a : ℕ → ℝ) 
  (h_geom : is_positive_geometric_sequence a)
  (h_sum : a 2 + a 6 = 10)
  (h_prod : a 4 * a 8 = 64) :
  a 4 = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l393_39380


namespace NUMINAMATH_CALUDE_part_one_part_two_l393_39308

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | x - a < 0}
def B : Set ℝ := {x | x^2 - 2*x - 8 < 0}

-- Part 1
theorem part_one (a : ℝ) (h : a = 3) :
  let U := A a ∪ B
  B ∪ (U \ A a) = {x | x > -2} := by sorry

-- Part 2
theorem part_two :
  {a : ℝ | A a ∩ B = B} = {a : ℝ | a ≥ 4} := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l393_39308


namespace NUMINAMATH_CALUDE_connor_date_expense_l393_39346

/-- The total amount Connor spends on his movie date -/
def connor_total_spent (ticket_price : ℚ) (combo_price : ℚ) (candy_price : ℚ) (cup_price : ℚ) : ℚ :=
  2 * ticket_price + combo_price + 2 * candy_price + cup_price

/-- Theorem: Connor spends $49.00 on his movie date -/
theorem connor_date_expense :
  connor_total_spent 14 11 2.5 5 = 49 := by
  sorry

end NUMINAMATH_CALUDE_connor_date_expense_l393_39346


namespace NUMINAMATH_CALUDE_power_of_two_l393_39324

theorem power_of_two (n : ℕ) : 32 * (1/2)^2 = 2^n → 2^n = 8 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_l393_39324

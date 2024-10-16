import Mathlib

namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l259_25904

/-- A geometric sequence with common ratio 2 and specific sum condition -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a (n + 1) = 2 * a n) ∧ (a 1 + a 2 + a 3 = 21)

/-- The sum of the 3rd, 4th, and 5th terms equals 84 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (h : GeometricSequence a) :
    a 3 + a 4 + a 5 = 84 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l259_25904


namespace NUMINAMATH_CALUDE_king_can_equalize_l259_25933

/-- Represents a chessboard square --/
structure Square where
  row : Fin 8
  col : Fin 8

/-- Represents the state of the chessboard --/
def Chessboard := Square → ℤ

/-- Represents a sequence of king's moves --/
def KingPath := List Square

/-- Checks if a move between two squares is valid for a king --/
def isValidKingMove (s1 s2 : Square) : Prop :=
  (abs (s1.row - s2.row) ≤ 1) ∧ (abs (s1.col - s2.col) ≤ 1)

/-- Applies a sequence of king's moves to a chessboard --/
def applyMoves (board : Chessboard) (path : KingPath) : Chessboard :=
  sorry

/-- The main theorem --/
theorem king_can_equalize (initial : Chessboard) :
  ∃ (path : KingPath), ∀ (s1 s2 : Square), (applyMoves initial path s1) = (applyMoves initial path s2) :=
sorry

end NUMINAMATH_CALUDE_king_can_equalize_l259_25933


namespace NUMINAMATH_CALUDE_hendrix_class_size_l259_25915

theorem hendrix_class_size :
  ∀ (initial_students : ℕ),
    (initial_students + 20 : ℚ) * (2/3) = 120 →
    initial_students = 160 := by
  sorry

end NUMINAMATH_CALUDE_hendrix_class_size_l259_25915


namespace NUMINAMATH_CALUDE_total_books_on_shelves_l259_25987

theorem total_books_on_shelves (num_shelves : ℕ) (books_per_shelf : ℕ) 
  (h1 : num_shelves = 150) 
  (h2 : books_per_shelf = 15) : 
  num_shelves * books_per_shelf = 2250 := by
sorry

end NUMINAMATH_CALUDE_total_books_on_shelves_l259_25987


namespace NUMINAMATH_CALUDE_crafts_club_beads_l259_25993

/-- The number of beads needed for a group of people making necklaces -/
def total_beads (num_members : ℕ) (necklaces_per_member : ℕ) (beads_per_necklace : ℕ) : ℕ :=
  num_members * necklaces_per_member * beads_per_necklace

theorem crafts_club_beads : 
  total_beads 9 2 50 = 900 := by
  sorry

end NUMINAMATH_CALUDE_crafts_club_beads_l259_25993


namespace NUMINAMATH_CALUDE_legacy_gain_satisfies_conditions_l259_25942

/-- The legacy gain received by Ms. Emily Smith -/
def legacy_gain : ℝ := 46345

/-- The federal tax rate as a decimal -/
def federal_tax_rate : ℝ := 0.25

/-- The regional tax rate as a decimal -/
def regional_tax_rate : ℝ := 0.15

/-- The total amount of taxes paid -/
def total_taxes_paid : ℝ := 16800

/-- Theorem stating that the legacy gain satisfies the given conditions -/
theorem legacy_gain_satisfies_conditions :
  federal_tax_rate * legacy_gain + 
  regional_tax_rate * (legacy_gain - federal_tax_rate * legacy_gain) = 
  total_taxes_paid := by sorry

end NUMINAMATH_CALUDE_legacy_gain_satisfies_conditions_l259_25942


namespace NUMINAMATH_CALUDE_min_value_expression_l259_25908

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 16) :
  x^2 + 8*x*y + 16*y^2 + 4*z^2 ≥ 48 ∧ ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ * y₀ * z₀ = 16 ∧ x₀^2 + 8*x₀*y₀ + 16*y₀^2 + 4*z₀^2 = 48 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l259_25908


namespace NUMINAMATH_CALUDE_lance_workdays_per_week_l259_25996

/-- Given Lance's work schedule and earnings, prove the number of workdays per week -/
theorem lance_workdays_per_week 
  (total_weekly_hours : ℕ) 
  (hourly_wage : ℚ) 
  (daily_earnings : ℚ) 
  (h1 : total_weekly_hours = 35)
  (h2 : hourly_wage = 9)
  (h3 : daily_earnings = 63)
  (h4 : ∃ (daily_hours : ℚ), daily_hours * hourly_wage = daily_earnings ∧ 
        daily_hours * (total_weekly_hours / daily_hours) = total_weekly_hours) :
  total_weekly_hours / (daily_earnings / hourly_wage) = 5 := by
  sorry

end NUMINAMATH_CALUDE_lance_workdays_per_week_l259_25996


namespace NUMINAMATH_CALUDE_father_son_age_sum_l259_25930

theorem father_son_age_sum (father_age son_age : ℕ) : 
  father_age = 64 →
  son_age = 16 →
  father_age = 4 * son_age →
  father_age - 10 + (son_age - 10) = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_father_son_age_sum_l259_25930


namespace NUMINAMATH_CALUDE_distribute_objects_eq_144_l259_25914

-- Define the number of objects and containers
def n : ℕ := 4

-- Define the function to calculate the number of ways to distribute objects
def distribute_objects : ℕ := sorry

-- Theorem statement
theorem distribute_objects_eq_144 : distribute_objects = 144 := by sorry

end NUMINAMATH_CALUDE_distribute_objects_eq_144_l259_25914


namespace NUMINAMATH_CALUDE_isaac_pen_purchase_l259_25919

theorem isaac_pen_purchase : ∃ (pens : ℕ), 
  pens + (12 + 5 * pens) = 108 ∧ pens = 16 := by
  sorry

end NUMINAMATH_CALUDE_isaac_pen_purchase_l259_25919


namespace NUMINAMATH_CALUDE_distance_between_signs_l259_25909

theorem distance_between_signs (total_distance : ℕ) (distance_to_first_sign : ℕ) (distance_after_second_sign : ℕ)
  (h1 : total_distance = 1000)
  (h2 : distance_to_first_sign = 350)
  (h3 : distance_after_second_sign = 275) :
  total_distance - distance_to_first_sign - distance_after_second_sign = 375 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_signs_l259_25909


namespace NUMINAMATH_CALUDE_parallelogram_smallest_angle_l259_25955

theorem parallelogram_smallest_angle (a b c d : ℝ) : 
  -- Conditions
  a + b + c + d = 360 →  -- Sum of angles in a quadrilateral is 360°
  a = c →                -- Opposite angles are equal
  b = d →                -- Opposite angles are equal
  max a b - min a b = 100 →  -- Largest angle is 100° greater than smallest
  -- Conclusion
  min a b = 40 :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_smallest_angle_l259_25955


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l259_25950

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_product (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 7 * a 13 = 2) →
  (a 7 + a 13 = 3) →
  a 2 * a 18 = 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l259_25950


namespace NUMINAMATH_CALUDE_chord_length_unit_circle_l259_25901

/-- The length of the chord cut by a line on a unit circle -/
theorem chord_length_unit_circle (a b c : ℝ) (h : a ≠ 0 ∨ b ≠ 0) :
  let line := {p : ℝ × ℝ | a * p.1 + b * p.2 + c = 0}
  let circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}
  let d := |c| / Real.sqrt (a^2 + b^2)
  2 * Real.sqrt (1 - d^2) = 8/5 ↔ 
    a = 3 ∧ b = -4 ∧ c = 3 :=
by sorry

end NUMINAMATH_CALUDE_chord_length_unit_circle_l259_25901


namespace NUMINAMATH_CALUDE_smallest_n_for_unique_k_l259_25977

theorem smallest_n_for_unique_k : 
  ∃ (k : ℤ), (9 : ℚ)/16 < (1 : ℚ)/(1 + k) ∧ (1 : ℚ)/(1 + k) < 7/12 ∧ 
  (∀ (k' : ℤ), (9 : ℚ)/16 < (1 : ℚ)/(1 + k') ∧ (1 : ℚ)/(1 + k') < 7/12 → k' = k) ∧
  (∀ (n : ℕ), n > 0 → n < 1 → 
    ¬(∃! (k : ℤ), (9 : ℚ)/16 < (n : ℚ)/(n + k) ∧ (n : ℚ)/(n + k) < 7/12)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_unique_k_l259_25977


namespace NUMINAMATH_CALUDE_sallys_car_fuel_efficiency_l259_25961

/-- Calculates the fuel efficiency of Sally's car given her trip expenses and savings --/
theorem sallys_car_fuel_efficiency :
  ∀ (savings : ℝ) (parking : ℝ) (entry : ℝ) (meal : ℝ) (distance : ℝ) (gas_price : ℝ) (additional_savings : ℝ),
    savings = 28 →
    parking = 10 →
    entry = 55 →
    meal = 25 →
    distance = 165 →
    gas_price = 3 →
    additional_savings = 95 →
    (2 * distance) / ((savings + additional_savings - (parking + entry + meal)) / gas_price) = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_sallys_car_fuel_efficiency_l259_25961


namespace NUMINAMATH_CALUDE_fraction_problem_l259_25945

theorem fraction_problem (a b : ℤ) (ha : a > 0) (hb : b > 0) :
  (a : ℚ) / (b + 6) = 1 / 6 ∧ (a + 4 : ℚ) / b = 1 / 4 →
  (a : ℚ) / b = 11 / 60 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l259_25945


namespace NUMINAMATH_CALUDE_infinite_non_fractional_numbers_l259_25948

/-- A number is p-good if it cannot be expressed as p^x * (p^(yz) - 1) / (p^y - 1) for any nonnegative integers x, y, z -/
def IsPGood (n : ℕ) (p : ℕ) : Prop :=
  ∀ x y z : ℕ, n ≠ p^x * (p^(y*z) - 1) / (p^y - 1)

/-- The set of numbers that cannot be expressed as (p^a - p^b) / (p^c - p^d) for any prime p and integers a, b, c, d -/
def NonFractionalSet : Set ℕ :=
  {n : ℕ | ∀ p : ℕ, Prime p → IsPGood n p}

theorem infinite_non_fractional_numbers : Set.Infinite NonFractionalSet := by
  sorry

end NUMINAMATH_CALUDE_infinite_non_fractional_numbers_l259_25948


namespace NUMINAMATH_CALUDE_equal_area_line_equation_l259_25980

-- Define the circle arrangement
def circle_arrangement : List (ℝ × ℝ) :=
  [(1, 1), (3, 1), (5, 1), (7, 1), (1, 3), (3, 3), (5, 3), (1, 5), (3, 5), (5, 5)]

-- Define the line with slope 2
def line_slope : ℝ := 2

-- Define the function to check if a line divides the area equally
def divides_area_equally (a b c : ℤ) : Prop := sorry

-- Define the function to check if three integers are coprime
def are_coprime (a b c : ℤ) : Prop := sorry

-- Main theorem
theorem equal_area_line_equation :
  ∃ (a b c : ℤ),
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    are_coprime a b c ∧
    divides_area_equally a b c ∧
    a^2 + b^2 + c^2 = 86 :=
  sorry

end NUMINAMATH_CALUDE_equal_area_line_equation_l259_25980


namespace NUMINAMATH_CALUDE_bankers_discount_problem_l259_25947

/-- Proves that given a sum S where the banker's discount is 18 and the true discount is 15, S equals 75 -/
theorem bankers_discount_problem (S : ℝ) 
  (h1 : 18 = 15 + (15^2 / S)) : S = 75 := by
  sorry

end NUMINAMATH_CALUDE_bankers_discount_problem_l259_25947


namespace NUMINAMATH_CALUDE_specific_plate_probability_l259_25932

/-- Represents the set of vowels used in license plates -/
def Vowels : Finset Char := {'A', 'E', 'I', 'O', 'U'}

/-- Represents the set of non-vowel letters used in license plates -/
def NonVowels : Finset Char := {'B', 'C', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'X', 'Y', 'Z'}

/-- Represents the set of even digits used in license plates -/
def EvenDigits : Finset Char := {'0', '2', '4', '6', '8'}

/-- Represents a license plate -/
structure LicensePlate where
  first : Char
  second : Char
  third : Char
  fourth : Char
  fifth : Char
  h1 : first ∈ Vowels
  h2 : second ∈ Vowels
  h3 : first ≠ second
  h4 : third ∈ NonVowels
  h5 : fourth ∈ NonVowels
  h6 : third ≠ fourth
  h7 : fifth ∈ EvenDigits

/-- The probability of a specific license plate occurring -/
def licensePlateProbability (plate : LicensePlate) : ℚ :=
  1 / (Vowels.card * (Vowels.card - 1) * NonVowels.card * (NonVowels.card - 1) * EvenDigits.card)

theorem specific_plate_probability :
  ∃ (plate : LicensePlate), licensePlateProbability plate = 1 / 50600 :=
sorry

end NUMINAMATH_CALUDE_specific_plate_probability_l259_25932


namespace NUMINAMATH_CALUDE_symmetric_point_coordinates_l259_25921

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define symmetry with respect to x-axis
def symmetricXAxis (p q : Point2D) : Prop :=
  p.x = q.x ∧ p.y = -q.y

-- Theorem statement
theorem symmetric_point_coordinates :
  ∀ (B A : Point2D),
    B.x = 4 ∧ B.y = -1 →
    symmetricXAxis A B →
    A.x = 4 ∧ A.y = 1 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_coordinates_l259_25921


namespace NUMINAMATH_CALUDE_robin_bracelet_cost_l259_25968

/-- Represents the types of bracelets available --/
inductive BraceletType
| Plastic
| Metal
| Beaded

/-- Represents a friend and their bracelet preference --/
structure Friend where
  name : String
  preference : List BraceletType

/-- Calculates the cost of a single bracelet --/
def braceletCost (type : BraceletType) : ℚ :=
  match type with
  | BraceletType.Plastic => 2
  | BraceletType.Metal => 3
  | BraceletType.Beaded => 5

/-- Calculates the total cost for a friend's bracelets --/
def friendCost (friend : Friend) : ℚ :=
  let numBracelets := friend.name.length
  let preferredTypes := friend.preference
  let costs := preferredTypes.map braceletCost
  let totalCost := costs.sum * numBracelets / preferredTypes.length
  totalCost

/-- Applies discount if applicable --/
def applyDiscount (total : ℚ) (numBracelets : ℕ) : ℚ :=
  if numBracelets ≥ 10 then total * (1 - 0.1) else total

/-- Applies sales tax --/
def applySalesTax (total : ℚ) : ℚ :=
  total * (1 + 0.07)

/-- The main theorem to prove --/
theorem robin_bracelet_cost : 
  let friends : List Friend := [
    ⟨"Jessica", [BraceletType.Plastic]⟩,
    ⟨"Tori", [BraceletType.Metal]⟩,
    ⟨"Lily", [BraceletType.Beaded]⟩,
    ⟨"Patrice", [BraceletType.Metal, BraceletType.Beaded]⟩
  ]
  let totalCost := friends.map friendCost |>.sum
  let numBracelets := friends.map (fun f => f.name.length) |>.sum
  let discountedCost := applyDiscount totalCost numBracelets
  let finalCost := applySalesTax discountedCost
  finalCost = 7223/100 := by
  sorry

end NUMINAMATH_CALUDE_robin_bracelet_cost_l259_25968


namespace NUMINAMATH_CALUDE_option_A_is_deductive_l259_25929

/-- Represents a type of reasoning --/
inductive ReasoningType
| Deductive
| Inductive
| Analogical

/-- Represents a logical argument --/
structure Argument where
  premises : List String
  conclusion : String

/-- Determines if an argument follows deductive reasoning --/
def is_deductive (arg : Argument) : Prop :=
  arg.premises.length ≥ 2 ∧
  arg.conclusion ≠ "" ∧
  (∀ (p : String), p ∈ arg.premises → p ≠ arg.conclusion)

/-- The argument for option A --/
def option_A : Argument :=
  { premises := [
      "For a circle with radius r, the area S = π r^2",
      "A unit circle has radius 1"
    ],
    conclusion := "For a unit circle, the area S = π"
  }

/-- Theorem stating that option A is deductive reasoning --/
theorem option_A_is_deductive : is_deductive option_A := by
  sorry

#check option_A_is_deductive

end NUMINAMATH_CALUDE_option_A_is_deductive_l259_25929


namespace NUMINAMATH_CALUDE_geometric_series_constant_l259_25954

/-- A geometric series with sum of first n terms given by S_n = 3^(n+1) + a -/
def GeometricSeries (a : ℝ) : ℕ → ℝ := fun n ↦ 3^(n+1) + a

/-- The sum of the first n terms of the geometric series -/
def SeriesSum (a : ℝ) : ℕ → ℝ := fun n ↦ GeometricSeries a n

theorem geometric_series_constant (a : ℝ) : a = -3 :=
  sorry

end NUMINAMATH_CALUDE_geometric_series_constant_l259_25954


namespace NUMINAMATH_CALUDE_intersection_dot_product_l259_25988

/-- An ellipse with equation x²/25 + y²/16 = 1 -/
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 16 = 1

/-- A hyperbola with equation x²/4 - y²/5 = 1 -/
def is_on_hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 5 = 1

/-- The common foci of the ellipse and hyperbola -/
def F₁ : ℝ × ℝ := sorry
def F₂ : ℝ × ℝ := sorry

/-- A point P that lies on both the ellipse and the hyperbola -/
def P : ℝ × ℝ := sorry

/-- Vector from P to F₁ -/
def PF₁ : ℝ × ℝ := (F₁.1 - P.1, F₁.2 - P.2)

/-- Vector from P to F₂ -/
def PF₂ : ℝ × ℝ := (F₂.1 - P.1, F₂.2 - P.2)

/-- Dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem intersection_dot_product :
  is_on_ellipse P.1 P.2 ∧ is_on_hyperbola P.1 P.2 →
  dot_product PF₁ PF₂ = 11 := by sorry

end NUMINAMATH_CALUDE_intersection_dot_product_l259_25988


namespace NUMINAMATH_CALUDE_sequence_sum_l259_25943

theorem sequence_sum (a : ℕ → ℕ) : 
  (a 1 = 1) → 
  (∀ n : ℕ, n ≥ 1 → a (n + 1) = a n + 2^n) → 
  a 10 = 1023 := by
sorry

end NUMINAMATH_CALUDE_sequence_sum_l259_25943


namespace NUMINAMATH_CALUDE_area_of_parallelogram_EFGH_l259_25978

/-- Represents a 2D vector -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Calculates the magnitude of the cross product of two 2D vectors -/
def crossProductMagnitude (v1 v2 : Vector2D) : ℝ :=
  |v1.x * v2.y - v1.y * v2.x|

/-- Theorem: Area of parallelogram EFGH -/
theorem area_of_parallelogram_EFGH : 
  let EF : Vector2D := ⟨3, 1⟩
  let EG : Vector2D := ⟨1, 5⟩
  crossProductMagnitude EF EG = 14 := by
  sorry

#check area_of_parallelogram_EFGH

end NUMINAMATH_CALUDE_area_of_parallelogram_EFGH_l259_25978


namespace NUMINAMATH_CALUDE_inverse_g_at_43_16_l259_25981

/-- Given a function g(x) = (x^3 - 5) / 4, prove that g⁻¹(43/16) = 3 * ∛7 / 2 -/
theorem inverse_g_at_43_16 (g : ℝ → ℝ) (h : ∀ x, g x = (x^3 - 5) / 4) :
  g⁻¹ (43/16) = 3 * Real.rpow 7 (1/3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_inverse_g_at_43_16_l259_25981


namespace NUMINAMATH_CALUDE_AB_vector_l259_25972

def OA : ℝ × ℝ := (1, -2)
def OB : ℝ × ℝ := (-3, 1)

theorem AB_vector : (OB.1 - OA.1, OB.2 - OA.2) = (-4, 3) := by
  sorry

end NUMINAMATH_CALUDE_AB_vector_l259_25972


namespace NUMINAMATH_CALUDE_rhombus_inscribed_circle_radius_l259_25918

theorem rhombus_inscribed_circle_radius 
  (side_length : ℝ) 
  (acute_angle : ℝ) 
  (h : side_length = 8 ∧ acute_angle = 30 * π / 180) : 
  side_length * Real.sin (acute_angle) / 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_inscribed_circle_radius_l259_25918


namespace NUMINAMATH_CALUDE_floor_function_unique_l259_25916

-- Define the function type
def RealFunction := ℝ → ℝ

-- Define the conditions
def Condition1 (f : RealFunction) : Prop :=
  ∀ x y : ℝ, f x + f y + 1 ≥ f (x + y) ∧ f (x + y) ≥ f x + f y

def Condition2 (f : RealFunction) : Prop :=
  ∀ x : ℝ, 0 ≤ x ∧ x < 1 → f 0 ≥ f x

def Condition3 (f : RealFunction) : Prop :=
  f (-1) = -1 ∧ f 1 = 1

-- Main theorem
theorem floor_function_unique (f : RealFunction)
  (h1 : Condition1 f) (h2 : Condition2 f) (h3 : Condition3 f) :
  ∀ x : ℝ, f x = ⌊x⌋ := by sorry

end NUMINAMATH_CALUDE_floor_function_unique_l259_25916


namespace NUMINAMATH_CALUDE_conic_is_pair_of_lines_l259_25975

/-- The equation of the conic section -/
def conic_equation (x y : ℝ) : Prop := 9 * x^2 - 36 * y^2 = 0

/-- The first line of the pair -/
def line1 (x y : ℝ) : Prop := x = 2 * y

/-- The second line of the pair -/
def line2 (x y : ℝ) : Prop := x = -2 * y

/-- Theorem stating that the conic equation represents a pair of straight lines -/
theorem conic_is_pair_of_lines :
  ∀ x y : ℝ, conic_equation x y ↔ (line1 x y ∨ line2 x y) :=
sorry

end NUMINAMATH_CALUDE_conic_is_pair_of_lines_l259_25975


namespace NUMINAMATH_CALUDE_no_mem_is_veen_l259_25900

-- Define the sets
variable (U : Type) -- Universe set
variable (Mem En Veen : Set U) -- Subsets of U

-- State the theorem
theorem no_mem_is_veen 
  (h1 : Mem ⊆ En) -- All Mems are Ens
  (h2 : En ∩ Veen = ∅) -- No Ens are Veens
  : Mem ∩ Veen = ∅ := -- No Mem is a Veen
by
  sorry

end NUMINAMATH_CALUDE_no_mem_is_veen_l259_25900


namespace NUMINAMATH_CALUDE_quadratic_root_problem_l259_25952

theorem quadratic_root_problem (k : ℝ) : 
  (∃ x : ℝ, x^2 + 2*x - k = 0 ∧ x = 0) → 
  (∃ y : ℝ, y^2 + 2*y - k = 0 ∧ y = -2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_problem_l259_25952


namespace NUMINAMATH_CALUDE_min_distance_vectors_l259_25946

/-- Given planar vectors a and b with an angle of 120° between them and a dot product of -1,
    the minimum value of |a - b| is √6. -/
theorem min_distance_vectors (a b : ℝ × ℝ) : 
  (Real.cos (120 * π / 180) = -1/2) →
  (a.1 * b.1 + a.2 * b.2 = -1) →
  (∀ c d : ℝ × ℝ, c.1 * d.1 + c.2 * d.2 = -1 → 
    Real.sqrt ((c.1 - d.1)^2 + (c.2 - d.2)^2) ≥ Real.sqrt 6) ∧
  (∃ e f : ℝ × ℝ, e.1 * f.1 + e.2 * f.2 = -1 ∧ 
    Real.sqrt ((e.1 - f.1)^2 + (e.2 - f.2)^2) = Real.sqrt 6) := by
  sorry

end NUMINAMATH_CALUDE_min_distance_vectors_l259_25946


namespace NUMINAMATH_CALUDE_sum_of_digits_l259_25984

/-- Given two single-digit numbers x and y, prove that x + y = 6 under certain conditions. -/
theorem sum_of_digits (x y : ℕ) : 
  (0 ≤ x ∧ x ≤ 9) →
  (0 ≤ y ∧ y ≤ 9) →
  (200 + 10 * x + 3) + 326 = (500 + 10 * y + 9) →
  (500 + 10 * y + 9) % 9 = 0 →
  x + y = 6 := by
sorry

end NUMINAMATH_CALUDE_sum_of_digits_l259_25984


namespace NUMINAMATH_CALUDE_john_playing_time_l259_25994

theorem john_playing_time (beats_per_minute : ℕ) (total_days : ℕ) (total_beats : ℕ) :
  beats_per_minute = 200 →
  total_days = 3 →
  total_beats = 72000 →
  (total_beats / beats_per_minute / 60) / total_days = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_john_playing_time_l259_25994


namespace NUMINAMATH_CALUDE_base5_division_theorem_l259_25953

/-- Converts a base 5 number to base 10 --/
def base5ToBase10 (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => acc * 5 + d) 0

/-- Converts a base 10 number to base 5 --/
def base10ToBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
  aux n []

theorem base5_division_theorem :
  let dividend := [2, 1, 3, 4, 2]  -- 21342₅
  let divisor := [2, 3]            -- 23₅
  let quotient := [4, 0, 4, 3]     -- 4043₅
  (base5ToBase10 dividend) / (base5ToBase10 divisor) = base5ToBase10 quotient := by
  sorry

end NUMINAMATH_CALUDE_base5_division_theorem_l259_25953


namespace NUMINAMATH_CALUDE_example_is_quadratic_l259_25957

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation x² = 2 + 3x is a quadratic equation -/
theorem example_is_quadratic : is_quadratic_equation (λ x => x^2 - 3*x - 2) := by
  sorry

end NUMINAMATH_CALUDE_example_is_quadratic_l259_25957


namespace NUMINAMATH_CALUDE_annika_hiking_distance_l259_25982

/-- Annika's hiking problem -/
theorem annika_hiking_distance 
  (rate : ℝ) -- Hiking rate in minutes per kilometer
  (initial_distance : ℝ) -- Initial distance hiked east in kilometers
  (total_time : ℝ) -- Total time available in minutes
  (h_rate : rate = 10) -- Hiking rate is 10 minutes per kilometer
  (h_initial : initial_distance = 2.5) -- Initial distance is 2.5 kilometers
  (h_time : total_time = 45) -- Total available time is 45 minutes
  : ∃ (total_east : ℝ), total_east = 3.5 ∧ 
    2 * (total_east - initial_distance) * rate + initial_distance * rate = total_time :=
by sorry

end NUMINAMATH_CALUDE_annika_hiking_distance_l259_25982


namespace NUMINAMATH_CALUDE_barbara_butcher_cost_l259_25956

/-- The total cost of Barbara's purchase at the butcher's -/
def total_cost (steak_weight : ℝ) (steak_price : ℝ) (chicken_weight : ℝ) (chicken_price : ℝ) : ℝ :=
  steak_weight * steak_price + chicken_weight * chicken_price

/-- Theorem: Barbara's total cost at the butcher's is $79.50 -/
theorem barbara_butcher_cost : 
  total_cost 4.5 15 1.5 8 = 79.5 := by
  sorry

#eval total_cost 4.5 15 1.5 8

end NUMINAMATH_CALUDE_barbara_butcher_cost_l259_25956


namespace NUMINAMATH_CALUDE_lollipop_ratio_l259_25959

theorem lollipop_ratio : 
  ∀ (alison henry diane : ℕ),
    alison = 60 →
    henry = alison + 30 →
    alison + henry + diane = 45 * 6 →
    (alison : ℚ) / diane = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_lollipop_ratio_l259_25959


namespace NUMINAMATH_CALUDE_m_minus_n_value_l259_25963

theorem m_minus_n_value (m n : ℤ) 
  (h1 : |m| = 5)
  (h2 : n^2 = 36)
  (h3 : m * n < 0) :
  m - n = 11 ∨ m - n = -11 := by
  sorry

end NUMINAMATH_CALUDE_m_minus_n_value_l259_25963


namespace NUMINAMATH_CALUDE_solution_sum_l259_25928

theorem solution_sum (x y : ℝ) (h : x^2 + y^2 = 8*x - 4*y - 20) : x + y = 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_sum_l259_25928


namespace NUMINAMATH_CALUDE_distance_to_origin_l259_25964

/-- The distance from point P(1, 2, 2) to the origin (0, 0, 0) is 3. -/
theorem distance_to_origin : Real.sqrt (1^2 + 2^2 + 2^2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_origin_l259_25964


namespace NUMINAMATH_CALUDE_exactly_two_false_l259_25967

-- Define the basic concepts
def Line : Type := sorry
def perpendicular (l1 l2 : Line) : Prop := sorry
def parallel (l1 l2 : Line) : Prop := sorry
def skew (l1 l2 : Line) : Prop := sorry
def intersects (l1 l2 : Line) : Prop := sorry

-- Define the statements
def statement1 : Prop := ∀ l1 l2 l3 : Line, perpendicular l1 l3 → perpendicular l2 l3 → parallel l1 l2
def statement2 : Prop := ∀ l1 l2 l3 : Line, parallel l1 l3 → parallel l2 l3 → parallel l1 l2
def statement3 : Prop := ∀ a b c : Line, parallel a b → perpendicular b c → perpendicular a c
def statement4 : Prop := ∀ a b l1 l2 : Line, skew a b → intersects l1 a → intersects l1 b → intersects l2 a → intersects l2 b → skew l1 l2

-- The theorem to prove
theorem exactly_two_false : 
  (¬statement1 ∧ statement2 ∧ statement3 ∧ ¬statement4) ∨
  (¬statement1 ∧ statement2 ∧ ¬statement3 ∧ ¬statement4) ∨
  (¬statement1 ∧ ¬statement2 ∧ statement3 ∧ ¬statement4) ∨
  (statement1 ∧ ¬statement2 ∧ statement3 ∧ ¬statement4) ∨
  (statement1 ∧ statement2 ∧ ¬statement3 ∧ ¬statement4) ∨
  (¬statement1 ∧ ¬statement2 ∧ statement3 ∧ statement4) :=
sorry

end NUMINAMATH_CALUDE_exactly_two_false_l259_25967


namespace NUMINAMATH_CALUDE_arithmetic_conversion_problem_l259_25962

theorem arithmetic_conversion_problem : 
  (2468 / (1 + 5^2 + 0)) - (1 + 5*7 + 4*7^2 + 3*7^3) + 6791 = 7624 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_conversion_problem_l259_25962


namespace NUMINAMATH_CALUDE_functional_equation_solutions_l259_25902

/-- The functional equation problem -/
def FunctionalEquation (t : ℝ) (f : ℝ → ℝ) : Prop :=
  t ≠ -1 ∧ ∀ x y : ℝ, (t + 1) * f (1 + x * y) - f (x + y) = f (x + 1) * f (y + 1)

/-- The set of solutions to the functional equation -/
def Solutions (t : ℝ) (f : ℝ → ℝ) : Prop :=
  (∀ x, f x = 0) ∨ (∀ x, f x = t) ∨ (∀ x, f x = (t + 1) * x - (t + 2))

/-- The main theorem: all solutions to the functional equation -/
theorem functional_equation_solutions (t : ℝ) (f : ℝ → ℝ) :
  FunctionalEquation t f ↔ Solutions t f := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solutions_l259_25902


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_x_range_l259_25925

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the second quadrant -/
def second_quadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

theorem point_in_second_quadrant_x_range :
  ∀ x : ℝ, second_quadrant ⟨x - 2, x⟩ → 0 < x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_x_range_l259_25925


namespace NUMINAMATH_CALUDE_shortest_return_path_length_l259_25926

/-- Represents a truncated right circular cone with given properties --/
structure TruncatedCone where
  lowerBaseCircumference : ℝ
  upperBaseCircumference : ℝ
  slopeAngle : ℝ

/-- Represents the tourist's path on the cone --/
def touristPath (cone : TruncatedCone) (upperBaseTravel : ℝ) : ℝ := sorry

/-- Theorem stating the shortest return path length --/
theorem shortest_return_path_length 
  (cone : TruncatedCone) 
  (h1 : cone.lowerBaseCircumference = 10)
  (h2 : cone.upperBaseCircumference = 9)
  (h3 : cone.slopeAngle = π / 3) -- 60 degrees in radians
  (h4 : upperBaseTravel = 3) :
  touristPath cone upperBaseTravel = (5 * Real.sqrt 3) / π :=
sorry

end NUMINAMATH_CALUDE_shortest_return_path_length_l259_25926


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l259_25990

theorem necessary_but_not_sufficient :
  (∀ x : ℝ, x^2 + 5*x + 6 < 0 → |x| > 1) ∧
  (∃ x : ℝ, |x| > 1 ∧ x^2 + 5*x + 6 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l259_25990


namespace NUMINAMATH_CALUDE_class_b_more_uniform_l259_25983

/-- Represents a class of students participating in a gymnastics competition -/
structure GymClass where
  name : String
  num_students : Nat
  avg_height : Float
  height_variance : Float

/-- Determines which of two classes has more uniform heights based on their variances -/
def more_uniform_heights (class_a class_b : GymClass) : Prop :=
  class_a.height_variance < class_b.height_variance

/-- Theorem: Given the variances of Class A and Class B, Class B has more uniform heights -/
theorem class_b_more_uniform (class_a class_b : GymClass) 
  (h1 : class_a.name = "A" ∧ class_b.name = "B")
  (h2 : class_a.num_students = 18 ∧ class_b.num_students = 18)
  (h3 : class_a.avg_height = 1.72 ∧ class_b.avg_height = 1.72)
  (h4 : class_a.height_variance = 3.24)
  (h5 : class_b.height_variance = 1.63) :
  more_uniform_heights class_b class_a :=
by sorry

end NUMINAMATH_CALUDE_class_b_more_uniform_l259_25983


namespace NUMINAMATH_CALUDE_math_competition_unattempted_questions_l259_25924

theorem math_competition_unattempted_questions :
  ∀ (total_questions : ℕ) (correct_points incorrect_points : ℤ) (score : ℕ),
    total_questions = 20 →
    correct_points = 8 →
    incorrect_points = -5 →
    (∃ k : ℕ, score = 13 * k) →
    ∀ (correct attempted : ℕ),
      attempted ≤ total_questions →
      score = correct_points * correct + incorrect_points * (attempted - correct) →
      (total_questions - attempted = 20 ∨ total_questions - attempted = 7) :=
by sorry

end NUMINAMATH_CALUDE_math_competition_unattempted_questions_l259_25924


namespace NUMINAMATH_CALUDE_B_power_48_l259_25969

def B : Matrix (Fin 3) (Fin 3) ℤ := !![0, 0, 0; 0, 0, 2; 0, -2, 0]

theorem B_power_48 : 
  B^48 = !![0, 0, 0; 0, 16^12, 0; 0, 0, 16^12] := by sorry

end NUMINAMATH_CALUDE_B_power_48_l259_25969


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_ratio_l259_25905

theorem isosceles_right_triangle_ratio (a c : ℝ) (h1 : a > 0) (h2 : c > 0) : 
  (a^2 + a^2 = c^2) → (2 * a / c = Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_ratio_l259_25905


namespace NUMINAMATH_CALUDE_johny_journey_distance_johny_journey_specific_distance_l259_25998

/-- Calculates the total distance of Johny's journey given his travel pattern. -/
theorem johny_journey_distance : ℕ → ℕ → ℕ
  | south_distance, east_extra_distance =>
    let east_distance := south_distance + east_extra_distance
    let north_distance := 2 * east_distance
    south_distance + east_distance + north_distance

/-- Proves that Johny's journey distance is 220 miles given the specific conditions. -/
theorem johny_journey_specific_distance :
  johny_journey_distance 40 20 = 220 := by
  sorry

end NUMINAMATH_CALUDE_johny_journey_distance_johny_journey_specific_distance_l259_25998


namespace NUMINAMATH_CALUDE_parabola_directrix_l259_25936

/-- The directrix of a parabola with equation y = -1/8 * x^2 is y = 2 -/
theorem parabola_directrix (x y : ℝ) : 
  (y = -1/8 * x^2) → (∃ (k : ℝ), k = 2 ∧ k = y + 1/(4 * (1/8))) :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l259_25936


namespace NUMINAMATH_CALUDE_gathering_handshakes_l259_25966

/-- Calculates the number of handshakes in a gathering with specific rules -/
def handshakes (n : ℕ) : ℕ :=
  let couples := n
  let men := couples
  let women := couples
  let guest := 1
  let total_people := men + women + guest
  let handshakes_among_men := men * (men - 1) / 2
  let handshakes_men_women := men * (women - 1)
  let handshakes_with_guest := total_people - 1
  handshakes_among_men + handshakes_men_women + handshakes_with_guest

/-- Theorem stating that in a gathering of 15 married couples and 1 special guest,
    with specific handshake rules, the total number of handshakes is 345 -/
theorem gathering_handshakes : handshakes 15 = 345 := by
  sorry

end NUMINAMATH_CALUDE_gathering_handshakes_l259_25966


namespace NUMINAMATH_CALUDE_ten_caterpillars_left_l259_25910

/-- The number of caterpillars left on a tree after some changes -/
def caterpillars_left (initial : ℕ) (hatched : ℕ) (left : ℕ) : ℕ :=
  initial + hatched - left

/-- Theorem: Given the initial conditions, prove that 10 caterpillars are left on the tree -/
theorem ten_caterpillars_left : 
  caterpillars_left 14 4 8 = 10 := by
  sorry

end NUMINAMATH_CALUDE_ten_caterpillars_left_l259_25910


namespace NUMINAMATH_CALUDE_grid_problem_l259_25913

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  first : ℝ
  diff : ℝ

/-- Represents the grid of numbers -/
structure NumberGrid where
  row : ArithmeticSequence
  col1 : ArithmeticSequence
  col2 : ArithmeticSequence

/-- The main theorem -/
theorem grid_problem (g : NumberGrid) :
  g.row.first = 16 ∧
  g.col1.first + g.col1.diff = 10 ∧
  g.col1.first + 2 * g.col1.diff = 19 ∧
  g.col2.first + 4 * g.col2.diff = -13 ∧
  g.row.first + 6 * g.row.diff = g.col2.first + 4 * g.col2.diff →
  g.col2.first = -36.75 := by
  sorry

end NUMINAMATH_CALUDE_grid_problem_l259_25913


namespace NUMINAMATH_CALUDE_jelly_bean_division_l259_25939

theorem jelly_bean_division (initial_amount : ℕ) (eaten_amount : ℕ) (num_piles : ℕ) 
  (h1 : initial_amount = 36)
  (h2 : eaten_amount = 6)
  (h3 : num_piles = 3)
  (h4 : initial_amount > eaten_amount) :
  (initial_amount - eaten_amount) / num_piles = 10 := by
  sorry

end NUMINAMATH_CALUDE_jelly_bean_division_l259_25939


namespace NUMINAMATH_CALUDE_rational_root_of_cubic_l259_25970

theorem rational_root_of_cubic (b c : ℚ) :
  (∃ x : ℝ, x^3 - 4*x^2 + b*x + c = 0 ∧ x = 4 - Real.sqrt 11) →
  (∃ y : ℚ, y^3 - 4*y^2 + b*y + c = 0) →
  (∃ z : ℚ, z^3 - 4*z^2 + b*z + c = 0 ∧ z = -4) :=
by sorry

end NUMINAMATH_CALUDE_rational_root_of_cubic_l259_25970


namespace NUMINAMATH_CALUDE_tire_cost_l259_25944

theorem tire_cost (total_cost : ℝ) (num_tires : ℕ) (cost_per_tire : ℝ) :
  total_cost = 4 →
  num_tires = 8 →
  cost_per_tire = total_cost / num_tires →
  cost_per_tire = 0.50 := by
sorry

end NUMINAMATH_CALUDE_tire_cost_l259_25944


namespace NUMINAMATH_CALUDE_total_value_is_correct_l259_25960

/-- The number of £5 notes issued by the Bank of England -/
def num_notes : ℕ := 440000000

/-- The face value of each note in pounds -/
def face_value : ℕ := 5

/-- The total face value of all notes in pounds -/
def total_value : ℕ := num_notes * face_value

/-- Theorem: The total face value of all notes is £2,200,000,000 -/
theorem total_value_is_correct : total_value = 2200000000 := by
  sorry

end NUMINAMATH_CALUDE_total_value_is_correct_l259_25960


namespace NUMINAMATH_CALUDE_juggler_count_l259_25937

theorem juggler_count (balls_per_juggler : ℕ) (total_balls : ℕ) (h1 : balls_per_juggler = 6) (h2 : total_balls = 2268) :
  total_balls / balls_per_juggler = 378 := by
  sorry

end NUMINAMATH_CALUDE_juggler_count_l259_25937


namespace NUMINAMATH_CALUDE_unique_remainder_sum_equal_l259_25965

/-- The sum of distinct remainders when dividing a natural number by all smaller positive natural numbers -/
def sumDistinctRemainders (n : ℕ) : ℕ :=
  (Finset.range n).sum (λ k => n % (k + 1))

/-- Theorem stating that 3 is the only natural number equal to the sum of its distinct remainders -/
theorem unique_remainder_sum_equal : ∀ n : ℕ, n > 0 → (sumDistinctRemainders n = n ↔ n = 3) := by
  sorry

end NUMINAMATH_CALUDE_unique_remainder_sum_equal_l259_25965


namespace NUMINAMATH_CALUDE_gcd_lcm_product_l259_25951

theorem gcd_lcm_product (a b : ℕ) (ha : a = 90) (hb : b = 135) :
  (Nat.gcd a b) * (Nat.lcm a b) = 12150 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_l259_25951


namespace NUMINAMATH_CALUDE_sheila_hourly_wage_l259_25995

/-- Sheila's work schedule and earnings -/
structure WorkSchedule where
  hours_mwf : ℕ  -- Hours worked on Monday, Wednesday, Friday
  days_mwf : ℕ   -- Number of days worked with hours_mwf
  hours_tt : ℕ   -- Hours worked on Tuesday, Thursday
  days_tt : ℕ    -- Number of days worked with hours_tt
  weekly_earnings : ℕ  -- Weekly earnings in dollars

/-- Calculate Sheila's hourly wage -/
def hourly_wage (schedule : WorkSchedule) : ℚ :=
  let total_hours := schedule.hours_mwf * schedule.days_mwf + schedule.hours_tt * schedule.days_tt
  schedule.weekly_earnings / total_hours

/-- Theorem: Sheila's hourly wage is $12 -/
theorem sheila_hourly_wage :
  let sheila_schedule : WorkSchedule := {
    hours_mwf := 8,
    days_mwf := 3,
    hours_tt := 6,
    days_tt := 2,
    weekly_earnings := 432
  }
  hourly_wage sheila_schedule = 12 := by
  sorry


end NUMINAMATH_CALUDE_sheila_hourly_wage_l259_25995


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l259_25906

theorem polynomial_evaluation :
  let y : ℤ := -2
  y^3 - y^2 + y - 1 = -7 := by sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l259_25906


namespace NUMINAMATH_CALUDE_positive_multiple_of_seven_find_x_l259_25979

theorem positive_multiple_of_seven (x : ℕ) : Prop :=
  ∃ k : ℕ, x = 7 * k ∧ k > 0

theorem find_x : ∃ x : ℕ, 
  positive_multiple_of_seven x ∧ 
  x^2 > 50 ∧ 
  x < 30 ∧
  (x = 14 ∨ x = 21 ∨ x = 28) :=
by sorry

end NUMINAMATH_CALUDE_positive_multiple_of_seven_find_x_l259_25979


namespace NUMINAMATH_CALUDE_machine_purchase_price_l259_25989

def machine_value (purchase_price : ℝ) (years : ℕ) : ℝ :=
  purchase_price * (1 - 0.3) ^ years

theorem machine_purchase_price : 
  ∃ (purchase_price : ℝ), 
    purchase_price > 0 ∧ 
    machine_value purchase_price 2 = 3200 ∧
    purchase_price = 8000 := by
  sorry

end NUMINAMATH_CALUDE_machine_purchase_price_l259_25989


namespace NUMINAMATH_CALUDE_binomial_expansion_problem_l259_25958

theorem binomial_expansion_problem (n : ℕ) :
  (∀ k, 0 ≤ k ∧ k ≤ n → Nat.choose n 6 ≥ Nat.choose n k) ∧
  (∀ k, k ≠ 6 → Nat.choose n 6 > Nat.choose n k) →
  n = 12 ∧ 2^(n+4) % 7 = 2 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_problem_l259_25958


namespace NUMINAMATH_CALUDE_root_sum_reciprocals_l259_25931

theorem root_sum_reciprocals (p q r s : ℂ) : 
  p^4 + 6*p^3 + 11*p^2 + 6*p + 3 = 0 →
  q^4 + 6*q^3 + 11*q^2 + 6*q + 3 = 0 →
  r^4 + 6*r^3 + 11*r^2 + 6*r + 3 = 0 →
  s^4 + 6*s^3 + 11*s^2 + 6*s + 3 = 0 →
  p ≠ q → p ≠ r → p ≠ s → q ≠ r → q ≠ s → r ≠ s →
  1/(p*q) + 1/(p*r) + 1/(p*s) + 1/(q*r) + 1/(q*s) + 1/(r*s) = 11/3 := by
sorry

end NUMINAMATH_CALUDE_root_sum_reciprocals_l259_25931


namespace NUMINAMATH_CALUDE_binomial_expansion_ratio_l259_25991

theorem binomial_expansion_ratio (n : ℕ) (a b c : ℝ) :
  n ≥ 3 →
  (∀ x : ℝ, (x + 2)^n = x^n + a * x^3 + b * x^2 + c * x + 2^n) →
  a / b = 3 / 2 →
  n = 11 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_ratio_l259_25991


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l259_25976

theorem geometric_sequence_ratio (a₁ : ℝ) (q : ℝ) : 
  let a₂ := a₁ * q
  let a₃ := a₁ * q^2
  let S₃ := a₁ + a₂ + a₃
  (S₃ = 13 ∧ 2 * (a₂ + 2) = a₁ + a₃) → (q = 3 ∨ q = 1/3) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l259_25976


namespace NUMINAMATH_CALUDE_min_colors_100x100_board_l259_25923

/-- Represents a board with cells divided into triangles -/
structure Board :=
  (size : Nat)
  (cells_divided : Bool)

/-- Represents a coloring of the triangles on the board -/
def Coloring := Board → Nat → Nat → Bool → Nat

/-- Checks if a coloring is valid (no adjacent triangles have the same color) -/
def is_valid_coloring (b : Board) (c : Coloring) : Prop := sorry

/-- The minimum number of colors needed for a valid coloring -/
def min_colors (b : Board) : Nat := sorry

/-- Theorem stating the minimum number of colors for a 100x100 board -/
theorem min_colors_100x100_board :
  ∀ (b : Board),
    b.size = 100 ∧
    b.cells_divided →
    min_colors b = 8 := by sorry

end NUMINAMATH_CALUDE_min_colors_100x100_board_l259_25923


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l259_25985

theorem simplify_and_evaluate (x y : ℚ) (hx : x = 1/2) (hy : y = 2023) :
  (x + y)^2 + (x + y)*(x - y) - 2*x^2 = 2*x*y := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l259_25985


namespace NUMINAMATH_CALUDE_solution_product_l259_25907

theorem solution_product (p q : ℝ) : 
  (p - 6) * (2 * p + 8) + p^2 - 15 * p + 56 = 0 →
  (q - 6) * (2 * q + 8) + q^2 - 15 * q + 56 = 0 →
  p ≠ q →
  (p + 3) * (q + 3) = 92 / 3 := by
sorry

end NUMINAMATH_CALUDE_solution_product_l259_25907


namespace NUMINAMATH_CALUDE_quadratic_roots_and_integer_case_l259_25917

-- Define the quadratic equation
def quadratic_equation (m : ℕ+) (x : ℝ) : ℝ :=
  m * x^2 - (3 * m + 2) * x + 6

-- Theorem statement
theorem quadratic_roots_and_integer_case (m : ℕ+) :
  (∃ x y : ℝ, x ≠ y ∧ quadratic_equation m x = 0 ∧ quadratic_equation m y = 0) ∧
  ((∃ x y : ℤ, x ≠ y ∧ quadratic_equation m x = 0 ∧ quadratic_equation m y = 0) →
   (m = 1 ∨ m = 2)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_and_integer_case_l259_25917


namespace NUMINAMATH_CALUDE_chessboard_covering_impossibility_l259_25999

/-- Represents a chessboard -/
structure Chessboard :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a domino -/
inductive Domino
  | TwoByTwo
  | OneByFour

/-- Represents a set of dominoes -/
def DominoSet := List Domino

/-- A function to check if a set of dominoes can cover a chessboard -/
def can_cover (board : Chessboard) (dominoes : DominoSet) : Prop :=
  sorry

/-- A function to replace one 2x2 domino with a 1x4 domino in a set -/
def replace_one_domino (dominoes : DominoSet) : DominoSet :=
  sorry

theorem chessboard_covering_impossibility (board : Chessboard) (original_dominoes : DominoSet) :
  board.rows = 2007 →
  board.cols = 2008 →
  can_cover board original_dominoes →
  ¬(can_cover board (replace_one_domino original_dominoes)) :=
  sorry

end NUMINAMATH_CALUDE_chessboard_covering_impossibility_l259_25999


namespace NUMINAMATH_CALUDE_min_cubes_for_valid_config_l259_25971

/-- Represents a modified cube with two protruding snaps and four receptacle holes. -/
structure ModifiedCube :=
  (snaps : Fin 2)
  (holes : Fin 4)

/-- Represents a configuration of snapped-together cubes. -/
structure CubeConfiguration :=
  (cubes : List ModifiedCube)
  (all_snaps_covered : Bool)

/-- Returns true if all snaps are covered in the given configuration. -/
def all_snaps_covered (config : CubeConfiguration) : Bool :=
  config.all_snaps_covered

/-- The minimum number of cubes required for a valid configuration. -/
def min_cubes : Nat := 6

/-- Theorem stating that the minimum number of cubes for a valid configuration is 6. -/
theorem min_cubes_for_valid_config :
  ∀ (config : CubeConfiguration),
    all_snaps_covered config →
    config.cubes.length ≥ min_cubes :=
  sorry

end NUMINAMATH_CALUDE_min_cubes_for_valid_config_l259_25971


namespace NUMINAMATH_CALUDE_count_even_greater_than_20000_position_of_35214_count_divisible_by_6_l259_25941

/-- The set of available digits --/
def digits : Finset Nat := {0, 1, 2, 3, 4, 5}

/-- A five-digit number formed from the available digits --/
structure FiveDigitNumber where
  d1 : Nat
  d2 : Nat
  d3 : Nat
  d4 : Nat
  d5 : Nat
  h1 : d1 ∈ digits
  h2 : d2 ∈ digits
  h3 : d3 ∈ digits
  h4 : d4 ∈ digits
  h5 : d5 ∈ digits
  h6 : d1 ≠ 0  -- Ensures it's a five-digit number

/-- The value of a FiveDigitNumber --/
def FiveDigitNumber.value (n : FiveDigitNumber) : Nat :=
  10000 * n.d1 + 1000 * n.d2 + 100 * n.d3 + 10 * n.d4 + n.d5

/-- The set of all valid FiveDigitNumbers --/
def allFiveDigitNumbers : Finset FiveDigitNumber := sorry

theorem count_even_greater_than_20000 :
  (allFiveDigitNumbers.filter (λ n => n.value % 2 = 0 ∧ n.value > 20000)).card = 240 := by sorry

theorem position_of_35214 :
  (allFiveDigitNumbers.filter (λ n => n.value < 35214)).card + 1 = 351 := by sorry

theorem count_divisible_by_6 :
  (allFiveDigitNumbers.filter (λ n => n.value % 6 = 0)).card = 108 := by sorry

end NUMINAMATH_CALUDE_count_even_greater_than_20000_position_of_35214_count_divisible_by_6_l259_25941


namespace NUMINAMATH_CALUDE_intersection_k_value_l259_25940

/-- Given two lines that intersect at x = -15, prove the value of k -/
theorem intersection_k_value (k : ℝ) : 
  (∀ x y : ℝ, -3 * x + y = k ∧ 0.3 * x + y = 10) →
  (∃ y : ℝ, -3 * (-15) + y = k ∧ 0.3 * (-15) + y = 10) →
  k = 59.5 := by
sorry

end NUMINAMATH_CALUDE_intersection_k_value_l259_25940


namespace NUMINAMATH_CALUDE_credit_card_balance_proof_l259_25935

/-- Calculates the new credit card balance after transactions -/
def new_balance (initial_balance groceries_charge towels_return : ℚ) : ℚ :=
  initial_balance + groceries_charge + (groceries_charge / 2) - towels_return

/-- Proves that the new balance is correct given the transactions -/
theorem credit_card_balance_proof :
  new_balance 126 60 45 = 171 := by
  sorry

end NUMINAMATH_CALUDE_credit_card_balance_proof_l259_25935


namespace NUMINAMATH_CALUDE_connect_to_inaccessible_intersection_l259_25997

-- Define the basic types
variable (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]

-- Define a line as a point and a direction vector
structure Line (V : Type*) [NormedAddCommGroup V] where
  point : V
  direction : V

-- Define the problem setup
variable (l₁ l₂ : Line V) (M : V)

-- State the theorem
theorem connect_to_inaccessible_intersection :
  ∃ (L : Line V), L.point = M ∧ 
    ∃ (t : ℝ), M + t • L.direction ∈ {x | ∃ (s₁ s₂ : ℝ), 
      x = l₁.point + s₁ • l₁.direction ∧ 
      x = l₂.point + s₂ • l₂.direction} :=
sorry

end NUMINAMATH_CALUDE_connect_to_inaccessible_intersection_l259_25997


namespace NUMINAMATH_CALUDE_elliott_triangle_hypotenuse_l259_25920

-- Define a right-angle triangle
structure RightTriangle where
  base : ℝ
  height : ℝ
  hypotenuse : ℝ
  right_angle : base^2 + height^2 = hypotenuse^2

-- Theorem statement
theorem elliott_triangle_hypotenuse (t : RightTriangle) 
  (h1 : t.base = 4)
  (h2 : t.height = 3) : 
  t.hypotenuse = 5 := by
  sorry

end NUMINAMATH_CALUDE_elliott_triangle_hypotenuse_l259_25920


namespace NUMINAMATH_CALUDE_cinema_tickets_l259_25911

theorem cinema_tickets (x y : ℕ) : 
  x + y = 35 →
  24 * x + 18 * y = 750 →
  x = 20 ∧ y = 15 := by
sorry

end NUMINAMATH_CALUDE_cinema_tickets_l259_25911


namespace NUMINAMATH_CALUDE_g_of_neg_one_eq_neg_seven_l259_25973

/-- Given a function g(x) = 5x - 2, prove that g(-1) = -7 -/
theorem g_of_neg_one_eq_neg_seven :
  let g : ℝ → ℝ := fun x ↦ 5 * x - 2
  g (-1) = -7 := by sorry

end NUMINAMATH_CALUDE_g_of_neg_one_eq_neg_seven_l259_25973


namespace NUMINAMATH_CALUDE_r_fourth_plus_inverse_r_fourth_l259_25938

theorem r_fourth_plus_inverse_r_fourth (r : ℝ) (h : (r + 1/r)^2 = 5) :
  r^4 + 1/r^4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_r_fourth_plus_inverse_r_fourth_l259_25938


namespace NUMINAMATH_CALUDE_investment_interest_calculation_l259_25986

/-- Calculates the total interest earned on an investment -/
def total_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * ((1 + rate) ^ time - 1)

/-- The problem statement -/
theorem investment_interest_calculation :
  let principal : ℝ := 2000
  let rate : ℝ := 0.08
  let time : ℕ := 5
  abs (total_interest principal rate time - 938.66) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_investment_interest_calculation_l259_25986


namespace NUMINAMATH_CALUDE_original_number_proof_l259_25934

theorem original_number_proof (N : ℕ) : 
  (∃ k : ℕ, N - 33 = 87 * k) ∧ 
  (∀ m : ℕ, m < 33 → ¬∃ j : ℕ, N - m = 87 * j) → 
  N = 120 := by
sorry

end NUMINAMATH_CALUDE_original_number_proof_l259_25934


namespace NUMINAMATH_CALUDE_highlighter_count_l259_25903

/-- The number of pink highlighters in Kaya's teacher's desk -/
def pink_highlighters : ℕ := 10

/-- The number of yellow highlighters in Kaya's teacher's desk -/
def yellow_highlighters : ℕ := 15

/-- The number of blue highlighters in Kaya's teacher's desk -/
def blue_highlighters : ℕ := 8

/-- The total number of highlighters in Kaya's teacher's desk -/
def total_highlighters : ℕ := pink_highlighters + yellow_highlighters + blue_highlighters

theorem highlighter_count : total_highlighters = 33 := by
  sorry

end NUMINAMATH_CALUDE_highlighter_count_l259_25903


namespace NUMINAMATH_CALUDE_cubic_inequality_solution_l259_25922

theorem cubic_inequality_solution (x : ℝ) :
  x^3 - 12*x^2 + 36*x > 0 ↔ (x > 0 ∧ x < 6) ∨ x > 6 := by
  sorry

end NUMINAMATH_CALUDE_cubic_inequality_solution_l259_25922


namespace NUMINAMATH_CALUDE_line_perpendicular_range_l259_25912

/-- Given a line l: x - y + a = 0 and points A(-2,0) and B(2,0),
    if there exists a point P on line l such that AP ⊥ BP,
    then -2√2 ≤ a ≤ 2√2. -/
theorem line_perpendicular_range (a : ℝ) :
  (∃ (P : ℝ × ℝ), 
    (P.1 - P.2 + a = 0) ∧ 
    ((P.1 + 2) * (P.1 - 2) + P.2 * P.2 = 0)) →
  -2 * Real.sqrt 2 ≤ a ∧ a ≤ 2 * Real.sqrt 2 :=
by sorry


end NUMINAMATH_CALUDE_line_perpendicular_range_l259_25912


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l259_25949

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →  -- sides are positive
  a^2 + b^2 = c^2 →  -- right-angled triangle (Pythagorean theorem)
  a^2 + b^2 + c^2 = 1800 →  -- sum of squares of all sides
  c = 30 := by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l259_25949


namespace NUMINAMATH_CALUDE_zoey_reading_schedule_l259_25974

def days_to_read (n : ℕ) : ℕ := 2 * n - 1

def total_days (num_books : ℕ) : ℕ := num_books^2

def day_of_week (start_day : ℕ) (days_passed : ℕ) : ℕ :=
  (start_day + days_passed - 1) % 7 + 1

theorem zoey_reading_schedule :
  let num_books := 18
  let start_day := 1  -- Monday
  let total_reading_days := total_days num_books
  day_of_week start_day total_reading_days = 3  -- Wednesday
:= by sorry

end NUMINAMATH_CALUDE_zoey_reading_schedule_l259_25974


namespace NUMINAMATH_CALUDE_partner_c_profit_l259_25927

/-- Represents a business partner --/
inductive Partner
| A
| B
| C
| D

/-- Calculates the profit for a given partner in the first year of the business --/
def calculateProfit (totalProfit : ℕ) (partnerShares : Partner → ℕ) (dJoinTime : ℚ) (partner : Partner) : ℚ :=
  let fullYearShares := partnerShares Partner.A + partnerShares Partner.B + partnerShares Partner.C
  let dAdjustedShare := (partnerShares Partner.D : ℚ) * dJoinTime
  let totalAdjustedShares := (fullYearShares : ℚ) + dAdjustedShare
  let sharePerPart := (totalProfit : ℚ) / totalAdjustedShares
  sharePerPart * (partnerShares partner : ℚ)

/-- Theorem stating that partner C's profit is $20,250 given the problem conditions --/
theorem partner_c_profit :
  let totalProfit : ℕ := 56700
  let partnerShares : Partner → ℕ := fun
    | Partner.A => 7
    | Partner.B => 9
    | Partner.C => 10
    | Partner.D => 4
  let dJoinTime : ℚ := 1/2
  calculateProfit totalProfit partnerShares dJoinTime Partner.C = 20250 := by
  sorry


end NUMINAMATH_CALUDE_partner_c_profit_l259_25927


namespace NUMINAMATH_CALUDE_local_minimum_at_one_l259_25992

-- Define the function f
def f (x m : ℝ) : ℝ := x * (x - m)^2

-- State the theorem
theorem local_minimum_at_one (m : ℝ) :
  (∃ δ > 0, ∀ x ∈ Set.Ioo (1 - δ) (1 + δ), f x m ≥ f 1 m) → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_local_minimum_at_one_l259_25992

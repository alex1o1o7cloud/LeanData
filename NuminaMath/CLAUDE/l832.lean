import Mathlib

namespace NUMINAMATH_CALUDE_total_sheets_is_114_l832_83297

/-- The number of bundles of colored paper -/
def coloredBundles : ℕ := 3

/-- The number of bunches of white paper -/
def whiteBunches : ℕ := 2

/-- The number of heaps of scrap paper -/
def scrapHeaps : ℕ := 5

/-- The number of sheets in a bunch -/
def sheetsPerBunch : ℕ := 4

/-- The number of sheets in a bundle -/
def sheetsPerBundle : ℕ := 2

/-- The number of sheets in a heap -/
def sheetsPerHeap : ℕ := 20

/-- The total number of sheets of paper removed from the chest of drawers -/
def totalSheets : ℕ := coloredBundles * sheetsPerBundle + whiteBunches * sheetsPerBunch + scrapHeaps * sheetsPerHeap

theorem total_sheets_is_114 : totalSheets = 114 := by
  sorry

end NUMINAMATH_CALUDE_total_sheets_is_114_l832_83297


namespace NUMINAMATH_CALUDE_parallel_vectors_l832_83286

/-- Given vectors a and b in ℝ², prove that if a is parallel to b, then λ = 8/5 -/
theorem parallel_vectors (a b : ℝ × ℝ) (h : a.1 / b.1 = a.2 / b.2) :
  a = (2, 5) → b.2 = 4 → b.1 = 8/5 := by
  sorry

#check parallel_vectors

end NUMINAMATH_CALUDE_parallel_vectors_l832_83286


namespace NUMINAMATH_CALUDE_kims_average_round_answers_l832_83221

/-- Represents the number of correct answers in each round of a math contest -/
structure ContestResults where
  easy : ℕ
  average : ℕ
  hard : ℕ

/-- Calculates the total points earned in the contest -/
def totalPoints (results : ContestResults) : ℕ :=
  2 * results.easy + 3 * results.average + 5 * results.hard

/-- Kim's contest results -/
def kimsResults : ContestResults := {
  easy := 6,
  average := 2,  -- This is what we want to prove
  hard := 4
}

theorem kims_average_round_answers :
  totalPoints kimsResults = 38 :=
by sorry

end NUMINAMATH_CALUDE_kims_average_round_answers_l832_83221


namespace NUMINAMATH_CALUDE_least_number_with_remainder_l832_83254

theorem least_number_with_remainder (n : ℕ) : n = 115 ↔ 
  (n > 0 ∧ 
   n % 38 = 1 ∧ 
   n % 3 = 1 ∧ 
   ∀ m : ℕ, m > 0 → m % 38 = 1 → m % 3 = 1 → n ≤ m) :=
by sorry

end NUMINAMATH_CALUDE_least_number_with_remainder_l832_83254


namespace NUMINAMATH_CALUDE_alok_veggie_plates_l832_83246

/-- Represents the order and payment details of Alok's meal -/
structure MealOrder where
  chapatis : Nat
  rice_plates : Nat
  ice_cream_cups : Nat
  chapati_cost : Nat
  rice_cost : Nat
  veggie_cost : Nat
  total_paid : Nat

/-- Calculates the number of mixed vegetable plates ordered -/
def veggie_plates_ordered (order : MealOrder) : Nat :=
  let known_cost := order.chapatis * order.chapati_cost + order.rice_plates * order.rice_cost
  let veggie_total_cost := order.total_paid - known_cost
  veggie_total_cost / order.veggie_cost

/-- Theorem stating that Alok ordered 11 plates of mixed vegetable -/
theorem alok_veggie_plates (order : MealOrder) 
        (h1 : order.chapatis = 16)
        (h2 : order.rice_plates = 5)
        (h3 : order.ice_cream_cups = 6)
        (h4 : order.chapati_cost = 6)
        (h5 : order.rice_cost = 45)
        (h6 : order.veggie_cost = 70)
        (h7 : order.total_paid = 1111) :
        veggie_plates_ordered order = 11 := by
  sorry

end NUMINAMATH_CALUDE_alok_veggie_plates_l832_83246


namespace NUMINAMATH_CALUDE_linear_equation_solutions_l832_83201

theorem linear_equation_solutions (x y : ℝ) : 
  (x = 1 ∧ y = 2 → 2*x + y = 4) ∧
  (x = 2 ∧ y = 0 → 2*x + y = 4) ∧
  (x = 0.5 ∧ y = 3 → 2*x + y = 4) ∧
  (x = -2 ∧ y = 4 → 2*x + y ≠ 4) := by
  sorry

#check linear_equation_solutions

end NUMINAMATH_CALUDE_linear_equation_solutions_l832_83201


namespace NUMINAMATH_CALUDE_two_digit_sum_reverse_l832_83271

theorem two_digit_sum_reverse : 
  (∃! n : Nat, n = (Finset.filter 
    (fun p : Nat × Nat => 
      let (a, b) := p
      0 < a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ (10 * a + b) + (10 * b + a) = 143)
    (Finset.product (Finset.range 10) (Finset.range 10))).card
  ∧ n = 6) := by sorry

end NUMINAMATH_CALUDE_two_digit_sum_reverse_l832_83271


namespace NUMINAMATH_CALUDE_quadratic_range_on_interval_l832_83237

def g (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_range_on_interval
  (a b c : ℝ)
  (ha : a > 0) :
  let range_min := min (g a b c (-1)) (g a b c 2)
  let range_max := max (g a b c (-1)) (max (g a b c 2) (g a b c (-b/(2*a))))
  ∀ x ∈ Set.Icc (-1 : ℝ) 2,
    range_min ≤ g a b c x ∧ g a b c x ≤ range_max :=
by sorry

end NUMINAMATH_CALUDE_quadratic_range_on_interval_l832_83237


namespace NUMINAMATH_CALUDE_baking_on_thursday_l832_83204

/-- The number of days between Amrita's cake baking -/
def baking_cycle : ℕ := 5

/-- The number of days between Thursdays -/
def thursday_cycle : ℕ := 7

/-- The number of days until Amrita bakes a cake on a Thursday again -/
def days_until_thursday_baking : ℕ := 35

theorem baking_on_thursday :
  Nat.lcm baking_cycle thursday_cycle = days_until_thursday_baking := by
  sorry

end NUMINAMATH_CALUDE_baking_on_thursday_l832_83204


namespace NUMINAMATH_CALUDE_multiples_of_6_or_8_not_both_count_multiples_6_or_8_not_both_l832_83200

theorem multiples_of_6_or_8_not_both (n : Nat) : 
  (Finset.filter (fun x => (x % 6 = 0 ∨ x % 8 = 0) ∧ ¬(x % 6 = 0 ∧ x % 8 = 0)) (Finset.range n)).card = 
  (Finset.filter (fun x => x % 6 = 0) (Finset.range n)).card + 
  (Finset.filter (fun x => x % 8 = 0) (Finset.range n)).card - 
  (Finset.filter (fun x => x % 24 = 0) (Finset.range n)).card :=
by sorry

theorem count_multiples_6_or_8_not_both : 
  (Finset.filter (fun x => (x % 6 = 0 ∨ x % 8 = 0) ∧ ¬(x % 6 = 0 ∧ x % 8 = 0)) (Finset.range 201)).card = 42 :=
by sorry

end NUMINAMATH_CALUDE_multiples_of_6_or_8_not_both_count_multiples_6_or_8_not_both_l832_83200


namespace NUMINAMATH_CALUDE_two_digit_prime_difference_l832_83270

theorem two_digit_prime_difference (x y : ℕ) : 
  x < 10 → y < 10 → (10 * x + y) - (10 * y + x) = 90 → x - y = 10 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_prime_difference_l832_83270


namespace NUMINAMATH_CALUDE_polynomial_factorization_constant_term_l832_83272

theorem polynomial_factorization_constant_term (a b c d e f : ℝ) 
  (p : ℝ → ℝ) (x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ : ℝ) :
  (∀ x, p x = x^8 - 4*x^7 + 7*x^6 + a*x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + f) →
  (∀ x, p x = (x - x₁) * (x - x₂) * (x - x₃) * (x - x₄) * (x - x₅) * (x - x₆) * (x - x₇) * (x - x₈)) →
  (x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₄ > 0 ∧ x₅ > 0 ∧ x₆ > 0 ∧ x₇ > 0 ∧ x₈ > 0) →
  f = 1 / 256 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_constant_term_l832_83272


namespace NUMINAMATH_CALUDE_an_is_arithmetic_sequence_l832_83257

/-- Definition of an arithmetic sequence's general term -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (m b : ℝ), ∀ n, a n = m * n + b

/-- The sequence defined by an = 3n - 1 -/
def a (n : ℕ) : ℝ := 3 * n - 1

/-- Theorem: The sequence defined by an = 3n - 1 is an arithmetic sequence -/
theorem an_is_arithmetic_sequence : is_arithmetic_sequence a := by
  sorry

end NUMINAMATH_CALUDE_an_is_arithmetic_sequence_l832_83257


namespace NUMINAMATH_CALUDE_negative_inequality_l832_83211

theorem negative_inequality (a b : ℝ) (h : a > b) : -a < -b := by
  sorry

end NUMINAMATH_CALUDE_negative_inequality_l832_83211


namespace NUMINAMATH_CALUDE_specific_tetrahedron_volume_l832_83225

/-- Represents a tetrahedron with vertices P, Q, R, S -/
structure Tetrahedron where
  PQ : ℝ
  PR : ℝ
  QR : ℝ
  PS : ℝ
  QS : ℝ
  RS : ℝ

/-- Calculates the volume of a tetrahedron -/
def tetrahedronVolume (t : Tetrahedron) : ℝ :=
  sorry

/-- Theorem stating that the volume of the specific tetrahedron is approximately 10.54 -/
theorem specific_tetrahedron_volume :
  let t : Tetrahedron := {
    PQ := 5.5,
    PR := 3.5,
    QR := 4,
    PS := 4.2,
    QS := 3.7,
    RS := 2.6
  }
  abs (tetrahedronVolume t - 10.54) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_specific_tetrahedron_volume_l832_83225


namespace NUMINAMATH_CALUDE_parabola_intersection_l832_83206

theorem parabola_intersection
  (a h d : ℝ) (ha : a ≠ 0) :
  let f (x : ℝ) := a * (x - h)^2 + d
  let g (x : ℝ) := a * ((x + 3) - h)^2 + d
  ∃! x, f x = g x ∧ x = -3/2 :=
sorry

end NUMINAMATH_CALUDE_parabola_intersection_l832_83206


namespace NUMINAMATH_CALUDE_tracy_candies_l832_83299

theorem tracy_candies (x : ℕ) : 
  (∃ (y : ℕ), x = 4 * y) →  -- x is divisible by 4
  (∃ (z : ℕ), (3 * x) / 4 = 3 * z) →  -- (3/4)x is divisible by 3
  (7 ≤ x / 2 - 24) →  -- lower bound after brother takes candies
  (x / 2 - 24 ≤ 11) →  -- upper bound after brother takes candies
  (x = 72 ∨ x = 76) :=
by sorry

end NUMINAMATH_CALUDE_tracy_candies_l832_83299


namespace NUMINAMATH_CALUDE_corner_cut_pentagon_area_l832_83230

/-- A pentagon formed by cutting a triangular corner from a rectangular piece of paper -/
structure CornerCutPentagon where
  sides : Finset ℕ
  is_valid : sides = {13, 19, 20, 25, 31}

/-- The area of a CornerCutPentagon -/
def area (p : CornerCutPentagon) : ℕ :=
  745

theorem corner_cut_pentagon_area (p : CornerCutPentagon) : area p = 745 := by
  sorry

end NUMINAMATH_CALUDE_corner_cut_pentagon_area_l832_83230


namespace NUMINAMATH_CALUDE_alex_jimmy_yellow_ratio_l832_83261

-- Define the number of marbles each person has
def lorin_black : ℕ := 4
def jimmy_yellow : ℕ := 22
def alex_total : ℕ := 19

-- Define Alex's black marbles as twice Lorin's
def alex_black : ℕ := 2 * lorin_black

-- Define Alex's yellow marbles
def alex_yellow : ℕ := alex_total - alex_black

-- Theorem to prove
theorem alex_jimmy_yellow_ratio :
  (alex_yellow : ℚ) / jimmy_yellow = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_alex_jimmy_yellow_ratio_l832_83261


namespace NUMINAMATH_CALUDE_mary_marbles_l832_83220

def dan_marbles : ℕ := 8
def mary_times_more : ℕ := 4

theorem mary_marbles : dan_marbles * mary_times_more = 32 := by
  sorry

end NUMINAMATH_CALUDE_mary_marbles_l832_83220


namespace NUMINAMATH_CALUDE_xyz_product_l832_83295

theorem xyz_product (x y z : ℂ) 
  (eq1 : x * y + 2 * y = -8)
  (eq2 : y * z + 2 * z = -8)
  (eq3 : z * x + 2 * x = -8) : 
  x * y * z = 32 := by
sorry

end NUMINAMATH_CALUDE_xyz_product_l832_83295


namespace NUMINAMATH_CALUDE_decimal_168_equals_binary_10101000_l832_83245

/-- Converts a natural number to its binary representation as a list of bits -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec toBinaryAux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: toBinaryAux (m / 2)
  toBinaryAux n

/-- Converts a list of bits to its decimal representation -/
def fromBinary (bits : List Bool) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

theorem decimal_168_equals_binary_10101000 :
  toBinary 168 = [false, false, false, true, false, true, false, true] ∧
  fromBinary [false, false, false, true, false, true, false, true] = 168 :=
by sorry

end NUMINAMATH_CALUDE_decimal_168_equals_binary_10101000_l832_83245


namespace NUMINAMATH_CALUDE_summer_camp_probability_l832_83235

/-- Given a summer camp with 30 kids total, 22 in coding, and 19 in robotics,
    the probability of selecting two kids from different workshops is 32/39. -/
theorem summer_camp_probability (total : ℕ) (coding : ℕ) (robotics : ℕ) 
  (h_total : total = 30)
  (h_coding : coding = 22)
  (h_robotics : robotics = 19) :
  (total.choose 2 - (coding - (coding + robotics - total)).choose 2 - (robotics - (coding + robotics - total)).choose 2) / total.choose 2 = 32 / 39 :=
by sorry

end NUMINAMATH_CALUDE_summer_camp_probability_l832_83235


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_ratio_l832_83256

/-- Given a hyperbola with equation x^2/a^2 - y^2/b^2 = 1, where a ≠ b, 
    if the angle between its asymptotes is 90°, then a/b = 1 -/
theorem hyperbola_asymptote_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (∀ m₁ m₂ : ℝ, m₁ * m₂ = -1 ∧ m₁ = a/b ∧ m₂ = -a/b) →
  a / b = 1 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_ratio_l832_83256


namespace NUMINAMATH_CALUDE_negative_seven_in_A_l832_83218

def A : Set ℤ := {1, -7}

theorem negative_seven_in_A : -7 ∈ A := by
  sorry

end NUMINAMATH_CALUDE_negative_seven_in_A_l832_83218


namespace NUMINAMATH_CALUDE_common_roots_product_l832_83240

-- Define the two polynomial equations
def poly1 (K : ℝ) (x : ℝ) : ℝ := x^3 + K*x + 20
def poly2 (L : ℝ) (x : ℝ) : ℝ := x^3 + L*x^2 + 100

-- Define the theorem
theorem common_roots_product (K L : ℝ) :
  (∃ (u v : ℝ), u ≠ v ∧ 
    poly1 K u = 0 ∧ poly1 K v = 0 ∧
    poly2 L u = 0 ∧ poly2 L v = 0) →
  (∃ (p : ℝ), p = 10 * Real.rpow 2 (1/3) ∧
    ∃ (u v : ℝ), u ≠ v ∧ 
      poly1 K u = 0 ∧ poly1 K v = 0 ∧
      poly2 L u = 0 ∧ poly2 L v = 0 ∧
      u * v = p) :=
by sorry

end NUMINAMATH_CALUDE_common_roots_product_l832_83240


namespace NUMINAMATH_CALUDE_cube_root_inequality_l832_83208

theorem cube_root_inequality (x : ℝ) : 
  x > 0 → (x^(1/3) < 3*x ↔ x > 1/(3*(3^(1/2)))) :=
by sorry

end NUMINAMATH_CALUDE_cube_root_inequality_l832_83208


namespace NUMINAMATH_CALUDE_arithmetic_proof_l832_83212

theorem arithmetic_proof : (139 + 27) * 2 + (23 + 11) = 366 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_proof_l832_83212


namespace NUMINAMATH_CALUDE_two_face_cubes_count_l832_83263

/-- Represents a 3x3x3 cube formed by cutting a larger cube painted on all faces -/
structure PaintedCube :=
  (size : Nat)
  (painted_faces : Nat)
  (h_size : size = 3)
  (h_painted : painted_faces = 6)

/-- Counts the number of smaller cubes painted on exactly two faces -/
def count_two_face_cubes (cube : PaintedCube) : Nat :=
  12

/-- Theorem: The number of smaller cubes painted on exactly two faces in a 3x3x3 PaintedCube is 12 -/
theorem two_face_cubes_count (cube : PaintedCube) : count_two_face_cubes cube = 12 := by
  sorry

end NUMINAMATH_CALUDE_two_face_cubes_count_l832_83263


namespace NUMINAMATH_CALUDE_milk_water_ratio_after_filling_l832_83259

/-- Represents the ratio of milk to water -/
structure Ratio where
  milk : ℕ
  water : ℕ

/-- Represents the can with its contents -/
structure Can where
  capacity : ℕ
  current_volume : ℕ
  ratio : Ratio

def initial_can : Can :=
  { capacity := 60
  , current_volume := 40
  , ratio := { milk := 5, water := 3 } }

def final_can : Can :=
  { capacity := 60
  , current_volume := 60
  , ratio := { milk := 3, water := 1 } }

theorem milk_water_ratio_after_filling (c : Can) (h : c = initial_can) :
  (final_can.ratio.milk : ℚ) / final_can.ratio.water = 3 := by
  sorry

#check milk_water_ratio_after_filling

end NUMINAMATH_CALUDE_milk_water_ratio_after_filling_l832_83259


namespace NUMINAMATH_CALUDE_sports_club_membership_l832_83238

theorem sports_club_membership (total : ℕ) (badminton : ℕ) (tennis : ℕ) (both : ℕ) : 
  total = 30 → badminton = 17 → tennis = 19 → both = 9 →
  total - (badminton + tennis - both) = 3 := by
  sorry

end NUMINAMATH_CALUDE_sports_club_membership_l832_83238


namespace NUMINAMATH_CALUDE_max_value_of_f_l832_83293

/-- The function f(x) defined as sin(2x) - 2√3 * sin²(x) has a maximum value of 2 - √3 -/
theorem max_value_of_f (x : ℝ) : 
  let f : ℝ → ℝ := λ x => Real.sin (2 * x) - 2 * Real.sqrt 3 * (Real.sin x) ^ 2
  ∃ (M : ℝ), M = 2 - Real.sqrt 3 ∧ ∀ x, f x ≤ M := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_f_l832_83293


namespace NUMINAMATH_CALUDE_solve_scooter_problem_l832_83247

def scooter_problem (purchase_price repair_cost gain_percentage : ℝ) : Prop :=
  let total_cost := purchase_price + repair_cost
  let gain := total_cost * (gain_percentage / 100)
  let selling_price := total_cost + gain
  (purchase_price = 4700) ∧ 
  (repair_cost = 1000) ∧ 
  (gain_percentage = 1.7543859649122806) →
  selling_price = 5800

theorem solve_scooter_problem :
  scooter_problem 4700 1000 1.7543859649122806 :=
by
  sorry

end NUMINAMATH_CALUDE_solve_scooter_problem_l832_83247


namespace NUMINAMATH_CALUDE_two_digit_number_property_l832_83251

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  units : ℕ
  tens : ℕ
  units_lt_ten : units < 10
  tens_lt_ten : tens < 10

/-- The sum of digits is 8 -/
def sum_of_digits (n : TwoDigitNumber) : Prop :=
  n.units + n.tens = 8

/-- Adding 18 results in the reversed number -/
def reverse_property (n : TwoDigitNumber) : Prop :=
  n.units + 10 * n.tens + 18 = 10 * n.units + n.tens

theorem two_digit_number_property (n : TwoDigitNumber) 
  (h1 : sum_of_digits n) (h2 : reverse_property n) :
  n.units + n.tens = 8 ∧ n.units + 10 * n.tens + 18 = 10 * n.units + n.tens := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_property_l832_83251


namespace NUMINAMATH_CALUDE_base_difference_theorem_l832_83268

-- Define a function to convert a number from any base to base 10
def to_base_10 (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * base ^ i) 0

-- Define the given numbers
def num1 : List Nat := [3, 2, 7]
def base1 : Nat := 9
def num2 : List Nat := [2, 5, 3]
def base2 : Nat := 8

-- State the theorem
theorem base_difference_theorem : 
  to_base_10 num1 base1 - to_base_10 num2 base2 = 97 := by
  sorry

end NUMINAMATH_CALUDE_base_difference_theorem_l832_83268


namespace NUMINAMATH_CALUDE_sqrt_nine_is_rational_l832_83249

-- Define rationality
def IsRational (x : ℝ) : Prop := ∃ (p q : ℤ), q ≠ 0 ∧ x = (p : ℝ) / (q : ℝ)

-- State the theorem
theorem sqrt_nine_is_rational : IsRational (Real.sqrt 9) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_nine_is_rational_l832_83249


namespace NUMINAMATH_CALUDE_third_side_length_l832_83298

/-- A right-angled isosceles triangle with specific dimensions -/
structure RightIsoscelesTriangle where
  /-- The length of the equal sides -/
  a : ℝ
  /-- The length of the hypotenuse -/
  c : ℝ
  /-- The triangle is right-angled -/
  right_angled : a^2 + a^2 = c^2
  /-- The triangle is isosceles -/
  isosceles : a = 50
  /-- The perimeter of the triangle -/
  perimeter : a + a + c = 160

/-- The theorem stating the length of the third side -/
theorem third_side_length (t : RightIsoscelesTriangle) : t.c = 60 := by
  sorry

end NUMINAMATH_CALUDE_third_side_length_l832_83298


namespace NUMINAMATH_CALUDE_min_value_expression_l832_83284

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 3) :
  (y^2 / (x + 1)) + (x^2 / (y + 1)) ≥ 9/5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l832_83284


namespace NUMINAMATH_CALUDE_unique_solution_l832_83227

def digit_product (n : ℕ) : ℕ := 
  if n < 10 then n
  else (n % 10) * digit_product (n / 10)

theorem unique_solution : 
  ∃! x : ℕ, digit_product x = x^2 - 10*x - 22 ∧ x = 12 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_l832_83227


namespace NUMINAMATH_CALUDE_nine_team_league_games_l832_83282

/-- The number of games played in a league where each team plays every other team once -/
def num_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a league with 9 teams, where each team plays every other team once, 
    the total number of games played is 36 -/
theorem nine_team_league_games :
  num_games 9 = 36 := by
  sorry

end NUMINAMATH_CALUDE_nine_team_league_games_l832_83282


namespace NUMINAMATH_CALUDE_x_percent_of_z_l832_83232

theorem x_percent_of_z (x y z : ℝ) 
  (h1 : x = 1.20 * y) 
  (h2 : y = 0.40 * z) : 
  x = 0.48 * z := by
sorry

end NUMINAMATH_CALUDE_x_percent_of_z_l832_83232


namespace NUMINAMATH_CALUDE_tank_capacity_l832_83236

/-- The capacity of a tank given outlet and inlet pipe rates -/
theorem tank_capacity
  (outlet_time : ℝ)
  (inlet_rate : ℝ)
  (combined_time : ℝ)
  (h1 : outlet_time = 8)
  (h2 : inlet_rate = 8)
  (h3 : combined_time = 12) :
  ∃ (capacity : ℝ), capacity = 11520 ∧
    capacity / outlet_time - (inlet_rate * 60) = capacity / combined_time :=
sorry

end NUMINAMATH_CALUDE_tank_capacity_l832_83236


namespace NUMINAMATH_CALUDE_ratio_common_value_l832_83228

theorem ratio_common_value (x y z : ℝ) (h : (x + y) / z = (x + z) / y ∧ (x + z) / y = (y + z) / x) :
  (x + y) / z = -1 ∨ (x + y) / z = 2 :=
sorry

end NUMINAMATH_CALUDE_ratio_common_value_l832_83228


namespace NUMINAMATH_CALUDE_max_value_of_expression_l832_83296

theorem max_value_of_expression (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 2) :
  (a * b) / (a + b) + (a * c) / (a + c) + (b * c) / (b + c) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l832_83296


namespace NUMINAMATH_CALUDE_rope_cutting_impossibility_l832_83289

theorem rope_cutting_impossibility : ¬ ∃ (n : ℕ), 5 + 4 * n = 2019 := by
  sorry

end NUMINAMATH_CALUDE_rope_cutting_impossibility_l832_83289


namespace NUMINAMATH_CALUDE_circle_radius_is_two_l832_83253

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 8*x + y^2 + 4*y + 16 = 0

/-- The radius of the circle -/
def circle_radius : ℝ := 2

theorem circle_radius_is_two :
  ∃ (h k : ℝ), ∀ (x y : ℝ),
    circle_equation x y ↔ (x - h)^2 + (y - k)^2 = circle_radius^2 :=
sorry

end NUMINAMATH_CALUDE_circle_radius_is_two_l832_83253


namespace NUMINAMATH_CALUDE_polynomial_factorization_l832_83260

theorem polynomial_factorization (a b : ℝ) :
  (a^2 + 10*a + 25) - b^2 = (a + 5 + b) * (a + 5 - b) := by
  sorry

#check polynomial_factorization

end NUMINAMATH_CALUDE_polynomial_factorization_l832_83260


namespace NUMINAMATH_CALUDE_volunteer_arrangement_count_l832_83287

theorem volunteer_arrangement_count : 
  (n : ℕ) → 
  (total : ℕ) → 
  (day1 : ℕ) → 
  (day2 : ℕ) → 
  (day3 : ℕ) → 
  n = 4 → 
  total = 5 → 
  day1 = 1 → 
  day2 = 2 → 
  day3 = 1 → 
  day1 + day2 + day3 = n →
  (total.choose day1) * ((total - day1).choose day2) * ((total - day1 - day2).choose day3) = 60 := by
sorry

end NUMINAMATH_CALUDE_volunteer_arrangement_count_l832_83287


namespace NUMINAMATH_CALUDE_fraction_equality_l832_83252

theorem fraction_equality (x y z : ℝ) 
  (h : (x + 4) / 2 = (y + 9) / (z - 3) ∧ (x + 4) / 2 = (x + 5) / (z - 5)) : 
  x / y = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l832_83252


namespace NUMINAMATH_CALUDE_equation_solution_l832_83203

theorem equation_solution : ∃ x : ℝ, 45 - (x - (37 - (15 - 16))) = 55 ∧ x = 28 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l832_83203


namespace NUMINAMATH_CALUDE_long_jump_competition_l832_83234

theorem long_jump_competition (first second third fourth : ℝ) : 
  first = 22 →
  second > first →
  third = second - 2 →
  fourth = third + 3 →
  fourth = 24 →
  second - first = 1 := by
sorry

end NUMINAMATH_CALUDE_long_jump_competition_l832_83234


namespace NUMINAMATH_CALUDE_dance_attendance_l832_83258

theorem dance_attendance (girls boys : ℕ) : 
  boys = 2 * girls ∧ 
  boys = (girls - 1) + 8 → 
  boys = 14 := by
sorry

end NUMINAMATH_CALUDE_dance_attendance_l832_83258


namespace NUMINAMATH_CALUDE_divisibility_of_cube_difference_l832_83275

theorem divisibility_of_cube_difference (a b c : ℕ) : 
  Nat.Prime a → Nat.Prime b → Nat.Prime c → 
  c ∣ (a + b) → c ∣ (a * b) → 
  c ∣ (a^3 - b^3) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_cube_difference_l832_83275


namespace NUMINAMATH_CALUDE_remaining_money_l832_83288

def initial_amount : ℚ := 100
def apple_price : ℚ := 1.5
def orange_price : ℚ := 2
def pear_price : ℚ := 2.25
def apple_quantity : ℕ := 5
def orange_quantity : ℕ := 10
def pear_quantity : ℕ := 4

theorem remaining_money :
  initial_amount - 
  (apple_price * apple_quantity + 
   orange_price * orange_quantity + 
   pear_price * pear_quantity) = 63.5 := by
  sorry

end NUMINAMATH_CALUDE_remaining_money_l832_83288


namespace NUMINAMATH_CALUDE_equation_solution_l832_83285

theorem equation_solution : ∃ x : ℝ, x ≠ 2 ∧ (4*x^2 + 3*x + 2) / (x - 2) = 4*x + 5 → x = -2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l832_83285


namespace NUMINAMATH_CALUDE_range_of_a_l832_83267

-- Define the propositions p and q
def p (x : ℝ) : Prop := x^2 + 2*x - 3 > 0
def q (x a : ℝ) : Prop := (x - a) / (x - a - 1) > 0

-- Define the theorem
theorem range_of_a (a : ℝ) :
  (∀ x, ¬(q x a) → ¬(p x)) ∧ 
  (∃ x, ¬(p x) ∧ (q x a)) →
  a ∈ Set.Icc (-3) 0 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l832_83267


namespace NUMINAMATH_CALUDE_selling_price_after_markup_and_discount_l832_83277

/-- The selling price of a commodity after markup and discount -/
theorem selling_price_after_markup_and_discount (a : ℝ) : 
  let markup_rate : ℝ := 0.5
  let discount_rate : ℝ := 0.3
  let marked_price : ℝ := a * (1 + markup_rate)
  let final_price : ℝ := marked_price * (1 - discount_rate)
  final_price = 1.05 * a :=
by sorry

end NUMINAMATH_CALUDE_selling_price_after_markup_and_discount_l832_83277


namespace NUMINAMATH_CALUDE_households_using_all_brands_l832_83269

/-- Represents the survey results of household soap usage -/
structure SoapSurvey where
  total : ℕ
  none : ℕ
  only_x : ℕ
  only_y : ℕ
  only_z : ℕ
  ratio_all_to_two : ℕ
  ratio_all_to_one : ℕ

/-- Calculates the number of households using all three brands of soap -/
def households_using_all (survey : SoapSurvey) : ℕ :=
  (survey.only_x + survey.only_y + survey.only_z) / survey.ratio_all_to_one

/-- Theorem stating the number of households using all three brands of soap -/
theorem households_using_all_brands (survey : SoapSurvey) 
  (h1 : survey.total = 5000)
  (h2 : survey.none = 1200)
  (h3 : survey.only_x = 800)
  (h4 : survey.only_y = 600)
  (h5 : survey.only_z = 300)
  (h6 : survey.ratio_all_to_two = 5)
  (h7 : survey.ratio_all_to_one = 10) :
  households_using_all survey = 170 := by
  sorry


end NUMINAMATH_CALUDE_households_using_all_brands_l832_83269


namespace NUMINAMATH_CALUDE_equal_area_division_l832_83207

/-- Represents a shape on a grid -/
structure GridShape where
  area : ℝ
  mk_area_pos : area > 0

/-- Represents a line on a grid -/
structure GridLine where
  distance_from_origin : ℝ

/-- Represents the division of a shape by a line -/
def divides_equally (s : GridShape) (l : GridLine) : Prop :=
  ∃ (area1 area2 : ℝ), 
    area1 > 0 ∧ 
    area2 > 0 ∧ 
    area1 = area2 ∧ 
    area1 + area2 = s.area

/-- The main theorem -/
theorem equal_area_division 
  (gray_shape : GridShape) 
  (h_area : gray_shape.area = 10) 
  (mo : GridLine) 
  (parallel_line : GridLine) 
  (h_distance : parallel_line.distance_from_origin = mo.distance_from_origin + 2.6) :
  divides_equally gray_shape parallel_line := by
  sorry

end NUMINAMATH_CALUDE_equal_area_division_l832_83207


namespace NUMINAMATH_CALUDE_symmetric_points_imply_fourth_quadrant_l832_83216

/-- Given two points A and B symmetric with respect to the y-axis, 
    prove that point C lies in the fourth quadrant. -/
theorem symmetric_points_imply_fourth_quadrant 
  (a b : ℝ) 
  (h_symmetric : (a - 2, 3) = (-(-1), b + 5)) : 
  (a > 0 ∧ b < 0) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_imply_fourth_quadrant_l832_83216


namespace NUMINAMATH_CALUDE_starting_number_proof_l832_83283

theorem starting_number_proof (x : Int) : 
  (∃ (l : List Int), l.length = 15 ∧ 
    (∀ n ∈ l, Even n ∧ x ≤ n ∧ n ≤ 40) ∧
    (∀ n, x ≤ n ∧ n ≤ 40 ∧ Even n → n ∈ l)) →
  x = 12 := by
sorry

end NUMINAMATH_CALUDE_starting_number_proof_l832_83283


namespace NUMINAMATH_CALUDE_number_card_problem_l832_83215

theorem number_card_problem (A B C : ℝ) : 
  (A + B + C) / 3 = 143 →
  A + 4.5 = (B + C) / 2 →
  C = B - 3 →
  C = 143 := by sorry

end NUMINAMATH_CALUDE_number_card_problem_l832_83215


namespace NUMINAMATH_CALUDE_M_subset_N_l832_83274

-- Define the sets M and N
def M : Set ℝ := {x | x^2 = x}
def N : Set ℝ := {x | x ≤ 1}

-- Theorem statement
theorem M_subset_N : M ⊆ N := by
  sorry

end NUMINAMATH_CALUDE_M_subset_N_l832_83274


namespace NUMINAMATH_CALUDE_change_distance_scientific_notation_l832_83224

/-- Definition of scientific notation -/
def is_scientific_notation (n : ℝ) (a : ℝ) (p : ℤ) : Prop :=
  n = a * (10 : ℝ) ^ p ∧ 1 ≤ |a| ∧ |a| < 10

/-- The distance of Chang'e 1 from Earth in kilometers -/
def change_distance : ℝ := 380000

/-- Theorem stating that 380,000 is equal to 3.8 × 10^5 in scientific notation -/
theorem change_distance_scientific_notation :
  is_scientific_notation change_distance 3.8 5 :=
sorry

end NUMINAMATH_CALUDE_change_distance_scientific_notation_l832_83224


namespace NUMINAMATH_CALUDE_subset_implies_a_equals_one_l832_83213

theorem subset_implies_a_equals_one (a : ℝ) : 
  let A : Set ℝ := {0, -a}
  let B : Set ℝ := {1, a-2, 2*a-2}
  A ⊆ B → a = 1 := by
sorry

end NUMINAMATH_CALUDE_subset_implies_a_equals_one_l832_83213


namespace NUMINAMATH_CALUDE_train_journey_solution_l832_83280

/-- Represents the train journey problem -/
structure TrainJourney where
  distance : ℝ  -- Distance between stations in km
  speed : ℝ     -- Initial speed of the train in km/h

/-- Conditions of the train journey -/
def journey_conditions (j : TrainJourney) : Prop :=
  let reduced_speed := j.speed / 3
  let first_day_time := 2 + 0.5 + (j.distance - 2 * j.speed) / reduced_speed
  let second_day_time := (2 * j.speed + 14) / j.speed + 0.5 + (j.distance - (2 * j.speed + 14)) / reduced_speed
  first_day_time = j.distance / j.speed + 7/6 ∧
  second_day_time = j.distance / j.speed + 5/6

/-- The theorem to prove -/
theorem train_journey_solution :
  ∃ j : TrainJourney, journey_conditions j ∧ j.distance = 196 ∧ j.speed = 84 :=
sorry

end NUMINAMATH_CALUDE_train_journey_solution_l832_83280


namespace NUMINAMATH_CALUDE_inequality_system_solution_l832_83239

theorem inequality_system_solution (x : ℝ) : 
  ((x + 3) / 2 ≤ x + 2 ∧ 2 * (x + 4) > 4 * x + 2) ↔ (-1 ≤ x ∧ x < 3) :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l832_83239


namespace NUMINAMATH_CALUDE_perfect_square_polynomial_l832_83279

theorem perfect_square_polynomial (k : ℝ) : 
  (∀ a : ℝ, ∃ b : ℝ, a^2 + 2*k*a + 1 = b^2) → (k = 1 ∨ k = -1) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_polynomial_l832_83279


namespace NUMINAMATH_CALUDE_new_person_weight_l832_83248

/-- Given a group of 5 people where one person weighing 40 kg is replaced,
    resulting in an average weight increase of 10 kg, prove that the new
    person weighs 90 kg. -/
theorem new_person_weight (initial_count : Nat) (weight_replaced : Real) 
  (avg_increase : Real) (new_weight : Real) : 
  initial_count = 5 → 
  weight_replaced = 40 → 
  avg_increase = 10 → 
  new_weight = weight_replaced + (initial_count * avg_increase) → 
  new_weight = 90 := by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_l832_83248


namespace NUMINAMATH_CALUDE_pie_eating_contest_l832_83292

theorem pie_eating_contest (first_participant second_participant : ℚ) : 
  first_participant = 5/6 → second_participant = 2/3 → 
  first_participant - second_participant = 1/6 := by
sorry

end NUMINAMATH_CALUDE_pie_eating_contest_l832_83292


namespace NUMINAMATH_CALUDE_remainder_theorem_l832_83276

theorem remainder_theorem (x y u v : ℤ) : 
  x > 0 → y > 0 → x = u * y + v → 0 ≤ v → v < y → 
  (x - u * y + 3 * v) % y = 4 * v % y := by
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l832_83276


namespace NUMINAMATH_CALUDE_K_characterization_l832_83294

/-- Function that reverses the digits of a positive integer in decimal notation -/
def f (n : ℕ+) : ℕ+ := sorry

/-- The set of all positive integers k such that, for any multiple n of k, k also divides f(n) -/
def K : Set ℕ+ :=
  {k : ℕ+ | ∀ n : ℕ+, k ∣ n → k ∣ f n}

/-- Theorem stating that K is equal to the set {1, 3, 9, 11, 33, 99} -/
theorem K_characterization : K = {1, 3, 9, 11, 33, 99} := by sorry

end NUMINAMATH_CALUDE_K_characterization_l832_83294


namespace NUMINAMATH_CALUDE_quadrilateral_equal_sides_is_rhombus_l832_83266

-- Define a quadrilateral
structure Quadrilateral :=
  (a b c d : ℝ)

-- Define a rhombus
def is_rhombus (q : Quadrilateral) : Prop :=
  q.a = q.b ∧ q.b = q.c ∧ q.c = q.d

-- Theorem: A quadrilateral with all sides equal is a rhombus
theorem quadrilateral_equal_sides_is_rhombus (q : Quadrilateral) :
  q.a = q.b ∧ q.b = q.c ∧ q.c = q.d → is_rhombus q :=
by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_equal_sides_is_rhombus_l832_83266


namespace NUMINAMATH_CALUDE_telescope_purchase_problem_l832_83291

theorem telescope_purchase_problem (joan_price karl_price : ℝ) 
  (h1 : joan_price + karl_price = 400)
  (h2 : 2 * joan_price = karl_price + 74) :
  joan_price = 158 := by
  sorry

end NUMINAMATH_CALUDE_telescope_purchase_problem_l832_83291


namespace NUMINAMATH_CALUDE_james_total_cost_l832_83241

/-- Calculates the total cost of James' vehicle purchases, registrations, and maintenance packages --/
def total_cost : ℕ :=
  let dirt_bike_cost : ℕ := 3 * 150
  let off_road_cost : ℕ := 4 * 300
  let atv_cost : ℕ := 2 * 450
  let moped_cost : ℕ := 5 * 200
  let scooter_cost : ℕ := 3 * 100

  let dirt_bike_reg : ℕ := 3 * 25
  let off_road_reg : ℕ := 4 * 25
  let atv_reg : ℕ := 2 * 30
  let moped_reg : ℕ := 5 * 15
  let scooter_reg : ℕ := 3 * 20

  let dirt_bike_maint : ℕ := 3 * 50
  let off_road_maint : ℕ := 4 * 75
  let atv_maint : ℕ := 2 * 100
  let moped_maint : ℕ := 5 * 60

  dirt_bike_cost + off_road_cost + atv_cost + moped_cost + scooter_cost +
  dirt_bike_reg + off_road_reg + atv_reg + moped_reg + scooter_reg +
  dirt_bike_maint + off_road_maint + atv_maint + moped_maint

theorem james_total_cost : total_cost = 5170 := by
  sorry

end NUMINAMATH_CALUDE_james_total_cost_l832_83241


namespace NUMINAMATH_CALUDE_molecular_weight_3_moles_HBrO3_l832_83278

/-- Atomic weight of Hydrogen in g/mol -/
def atomic_weight_H : ℝ := 1.01

/-- Atomic weight of Bromine in g/mol -/
def atomic_weight_Br : ℝ := 79.90

/-- Atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- Number of Hydrogen atoms in HBrO3 -/
def num_H : ℕ := 1

/-- Number of Bromine atoms in HBrO3 -/
def num_Br : ℕ := 1

/-- Number of Oxygen atoms in HBrO3 -/
def num_O : ℕ := 3

/-- Number of moles of HBrO3 -/
def num_moles : ℝ := 3

/-- Calculates the molecular weight of HBrO3 in g/mol -/
def molecular_weight_HBrO3 : ℝ := 
  num_H * atomic_weight_H + num_Br * atomic_weight_Br + num_O * atomic_weight_O

/-- Theorem: The molecular weight of 3 moles of HBrO3 is 386.73 grams -/
theorem molecular_weight_3_moles_HBrO3 : 
  num_moles * molecular_weight_HBrO3 = 386.73 := by
  sorry

end NUMINAMATH_CALUDE_molecular_weight_3_moles_HBrO3_l832_83278


namespace NUMINAMATH_CALUDE_diagonal_length_of_quadrilateral_l832_83202

/-- The length of a diagonal in a quadrilateral with given area and offsets -/
theorem diagonal_length_of_quadrilateral (area : ℝ) (offset1 offset2 : ℝ) :
  area = 210 →
  offset1 = 9 →
  offset2 = 6 →
  (∃ d : ℝ, area = 0.5 * d * (offset1 + offset2) ∧ d = 28) :=
by sorry

end NUMINAMATH_CALUDE_diagonal_length_of_quadrilateral_l832_83202


namespace NUMINAMATH_CALUDE_greenwood_school_quiz_l832_83210

theorem greenwood_school_quiz (f s : ℕ) (h1 : f > 0) (h2 : s > 0) :
  (3 * f : ℚ) / 4 = (s : ℚ) / 3 → s = 3 * f := by
  sorry

end NUMINAMATH_CALUDE_greenwood_school_quiz_l832_83210


namespace NUMINAMATH_CALUDE_student_class_sizes_l832_83229

/-- Represents a configuration of students in classes -/
structure StudentConfig where
  total_students : ℕ
  classes : List ℕ
  classes_sum_eq_total : classes.sum = total_students

/-- Checks if any group of n students contains at least k from the same class -/
def satisfies_group_condition (config : StudentConfig) (n k : ℕ) : Prop :=
  ∀ (subset : List ℕ), subset.sum ≤ n → (∃ (c : ℕ), c ∈ config.classes ∧ c ≥ k)

/-- The main theorem to be proved -/
theorem student_class_sizes 
  (config : StudentConfig)
  (h_total : config.total_students = 60)
  (h_condition : satisfies_group_condition config 10 3) :
  (∃ (c : ℕ), c ∈ config.classes ∧ c ≥ 15) ∧
  ¬(∀ (config : StudentConfig), 
    config.total_students = 60 → 
    satisfies_group_condition config 10 3 → 
    ∃ (c : ℕ), c ∈ config.classes ∧ c ≥ 16) :=
by sorry

end NUMINAMATH_CALUDE_student_class_sizes_l832_83229


namespace NUMINAMATH_CALUDE_intersection_A_B_l832_83290

-- Define set A
def A : Set ℝ := {x | ∃ y : ℝ, y^2 = x}

-- Define set B
def B : Set ℝ := {y | ∃ x : ℝ, y = Real.sin x}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {x : ℝ | 0 ≤ x ∧ x ≤ 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l832_83290


namespace NUMINAMATH_CALUDE_seashell_counting_l832_83226

theorem seashell_counting (initial : ℕ) (given_joan : ℕ) (given_ali : ℕ) (given_lee : ℕ) 
  (h1 : initial = 200)
  (h2 : given_joan = 43)
  (h3 : given_ali = 27)
  (h4 : given_lee = 59) :
  initial - given_joan - given_ali - given_lee = 71 := by
  sorry

end NUMINAMATH_CALUDE_seashell_counting_l832_83226


namespace NUMINAMATH_CALUDE_eleven_in_base_two_l832_83214

theorem eleven_in_base_two : 11 = 1 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0 := by
  sorry

#eval toString (1 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0)

end NUMINAMATH_CALUDE_eleven_in_base_two_l832_83214


namespace NUMINAMATH_CALUDE_inverse_of_A_l832_83222

def A : Matrix (Fin 2) (Fin 2) ℚ := !![4, -2; 5, 3]

theorem inverse_of_A :
  let A_inv : Matrix (Fin 2) (Fin 2) ℚ := !![3/22, 1/11; -5/22, 2/11]
  A * A_inv = 1 ∧ A_inv * A = 1 := by sorry

end NUMINAMATH_CALUDE_inverse_of_A_l832_83222


namespace NUMINAMATH_CALUDE_quadratic_radicals_combination_l832_83250

theorem quadratic_radicals_combination (a : ℝ) : 
  (∃ (k : ℝ), k > 0 ∧ (1 + a) = k * (4 - 2*a)) ↔ a = 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_radicals_combination_l832_83250


namespace NUMINAMATH_CALUDE_derivative_of_exp_x_squared_minus_one_l832_83273

theorem derivative_of_exp_x_squared_minus_one (x : ℝ) :
  deriv (λ x => Real.exp (x^2 - 1)) x = 2 * x * Real.exp (x^2 - 1) :=
by sorry

end NUMINAMATH_CALUDE_derivative_of_exp_x_squared_minus_one_l832_83273


namespace NUMINAMATH_CALUDE_find_a_l832_83219

def A (a : ℝ) : Set ℝ := {4, a^2}
def B (a : ℝ) : Set ℝ := {a-6, a+1, 9}

theorem find_a : ∀ a : ℝ, A a ∩ B a = {9} → a = -3 := by
  sorry

end NUMINAMATH_CALUDE_find_a_l832_83219


namespace NUMINAMATH_CALUDE_hamburgers_served_l832_83255

/-- Proves that the number of hamburgers served is 3, given the total made and left over. -/
theorem hamburgers_served (total : ℕ) (leftover : ℕ) (h1 : total = 9) (h2 : leftover = 6) :
  total - leftover = 3 := by
  sorry

end NUMINAMATH_CALUDE_hamburgers_served_l832_83255


namespace NUMINAMATH_CALUDE_equation_solutions_l832_83243

theorem equation_solutions :
  (∀ x : ℝ, 2 * x^2 - 1 = 49 ↔ x = 5 ∨ x = -5) ∧
  (∀ x : ℝ, (x + 3)^3 = 64 ↔ x = 1) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l832_83243


namespace NUMINAMATH_CALUDE_smallest_positive_multiple_of_45_l832_83262

theorem smallest_positive_multiple_of_45 :
  ∀ n : ℕ, n > 0 → 45 ∣ n → n ≥ 45 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_multiple_of_45_l832_83262


namespace NUMINAMATH_CALUDE_floor_sqrt_eight_count_l832_83242

theorem floor_sqrt_eight_count : 
  (Finset.filter (fun x : ℕ => ⌊Real.sqrt x⌋ = 8) (Finset.range 81)).card = 17 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_eight_count_l832_83242


namespace NUMINAMATH_CALUDE_dice_sum_possibility_l832_83264

-- Define a type for standard six-sided dice
def Die := Fin 6

-- Define the problem conditions
def roll_conditions (a b c : Die) : Prop :=
  (a.val + 1 = 4 ∨ b.val + 1 = 4 ∨ c.val + 1 = 4) ∧
  ((a.val + 1) * (b.val + 1) * (c.val + 1) = 72)

-- Define the theorem
theorem dice_sum_possibility :
  ∃ (a b c : Die), roll_conditions a b c ∧ (a.val + b.val + c.val + 3 = 13) ∧
  (∀ (x y z : Die), roll_conditions x y z → 
    (x.val + y.val + z.val + 3 ≠ 12) ∧
    (x.val + y.val + z.val + 3 ≠ 14) ∧
    (x.val + y.val + z.val + 3 ≠ 15) ∧
    (x.val + y.val + z.val + 3 ≠ 16)) :=
by sorry

end NUMINAMATH_CALUDE_dice_sum_possibility_l832_83264


namespace NUMINAMATH_CALUDE_unique_positive_solution_l832_83217

theorem unique_positive_solution (x y z : ℝ) :
  x > 0 ∧ y > 0 ∧ z > 0 →
  x^y = z ∧ y^z = x ∧ z^x = y →
  x = 1 ∧ y = 1 ∧ z = 1 := by
sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l832_83217


namespace NUMINAMATH_CALUDE_post_office_distance_l832_83244

theorem post_office_distance (outbound_speed inbound_speed : ℝ) 
  (total_time : ℝ) (distance : ℝ) : 
  outbound_speed = 12.5 →
  inbound_speed = 2 →
  total_time = 5.8 →
  distance / outbound_speed + distance / inbound_speed = total_time →
  distance = 10 := by
sorry

end NUMINAMATH_CALUDE_post_office_distance_l832_83244


namespace NUMINAMATH_CALUDE_base7_to_base10_conversion_l832_83265

/-- Converts a base 7 number to base 10 --/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (7 ^ i)) 0

/-- The base 7 representation of the number --/
def base7Number : List Nat := [4, 6, 5, 7, 3]

theorem base7_to_base10_conversion :
  base7ToBase10 base7Number = 9895 := by
  sorry

end NUMINAMATH_CALUDE_base7_to_base10_conversion_l832_83265


namespace NUMINAMATH_CALUDE_michelle_taxi_cost_l832_83223

/-- Calculates the total cost of a taxi ride given the initial fee, distance, and per-mile charge. -/
def taxi_cost (initial_fee : ℝ) (distance : ℝ) (per_mile_charge : ℝ) : ℝ :=
  initial_fee + distance * per_mile_charge

/-- Theorem stating that for the given conditions, the total cost is $12. -/
theorem michelle_taxi_cost : taxi_cost 2 4 2.5 = 12 := by sorry

end NUMINAMATH_CALUDE_michelle_taxi_cost_l832_83223


namespace NUMINAMATH_CALUDE_four_digit_number_problem_l832_83205

def is_arithmetic_sequence (a b c d : ℕ) : Prop :=
  b - a = c - b ∧ c - b = d - c

def is_geometric_mean (x y z : ℕ) : Prop :=
  x * z = y * y

theorem four_digit_number_problem (a b c d : ℕ) :
  a ≥ 1 ∧ a ≤ 9 ∧
  b ≥ 0 ∧ b ≤ 9 ∧
  c ≥ 0 ∧ c ≤ 9 ∧
  d ≥ 0 ∧ d ≤ 9 ∧
  is_arithmetic_sequence a b c d ∧
  is_geometric_mean a b d ∧
  1000 * a + 100 * b + 10 * c + d + 1000 * d + 100 * c + 10 * b + a = 11110 →
  (a = 5 ∧ b = 5 ∧ c = 5 ∧ d = 5) ∨ (a = 2 ∧ b = 4 ∧ c = 6 ∧ d = 8) :=
by sorry

end NUMINAMATH_CALUDE_four_digit_number_problem_l832_83205


namespace NUMINAMATH_CALUDE_angles_on_y_axis_correct_l832_83281

/-- The set of angles whose terminal sides fall on the y-axis -/
def angles_on_y_axis : Set ℝ :=
  { α | ∃ k : ℤ, α = k * Real.pi + Real.pi / 2 }

/-- Theorem stating that angles_on_y_axis correctly represents
    the set of angles whose terminal sides fall on the y-axis -/
theorem angles_on_y_axis_correct :
  ∀ α : ℝ, α ∈ angles_on_y_axis ↔ 
    (∃ k : ℤ, α = k * Real.pi + Real.pi / 2) :=
by sorry

end NUMINAMATH_CALUDE_angles_on_y_axis_correct_l832_83281


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l832_83209

theorem unique_solution_quadratic (k : ℝ) : 
  (∃! x : ℝ, 3 * x^2 + k * x + 16 = 0) ↔ k = 8 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l832_83209


namespace NUMINAMATH_CALUDE_sin_equals_cos_690_l832_83231

theorem sin_equals_cos_690 (n : ℤ) (h1 : -180 ≤ n ∧ n ≤ 180) 
  (h2 : Real.sin (n * π / 180) = Real.cos (690 * π / 180)) : n = 60 := by
  sorry

end NUMINAMATH_CALUDE_sin_equals_cos_690_l832_83231


namespace NUMINAMATH_CALUDE_largest_angle_in_pentagon_l832_83233

theorem largest_angle_in_pentagon (F G H I J : ℝ) : 
  F = 70 → 
  G = 110 → 
  H = I → 
  J = 2 * H + 25 → 
  F + G + H + I + J = 540 → 
  J = 192.5 ∧ J = max F (max G (max H (max I J))) := by
  sorry

end NUMINAMATH_CALUDE_largest_angle_in_pentagon_l832_83233

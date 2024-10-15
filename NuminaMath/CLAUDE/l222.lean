import Mathlib

namespace NUMINAMATH_CALUDE_arithmetic_mean_after_removal_l222_22268

theorem arithmetic_mean_after_removal (s : Finset ℝ) (a b c : ℝ) :
  s.card = 80 →
  a = 50 ∧ b = 60 ∧ c = 70 →
  a ∈ s ∧ b ∈ s ∧ c ∈ s →
  (s.sum id) / s.card = 45 →
  ((s.sum id) - (a + b + c)) / (s.card - 3) = 3420 / 77 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_after_removal_l222_22268


namespace NUMINAMATH_CALUDE_fraction_equality_l222_22265

theorem fraction_equality (a b : ℕ) (h1 : a + b = 1210) (h2 : b = 484) :
  (4 / 15 : ℚ) * a = (2 / 5 : ℚ) * b :=
sorry

end NUMINAMATH_CALUDE_fraction_equality_l222_22265


namespace NUMINAMATH_CALUDE_floor_inequality_l222_22293

theorem floor_inequality (x : ℝ) : 
  ⌊5*x⌋ ≥ ⌊x⌋ + ⌊2*x⌋/2 + ⌊3*x⌋/3 + ⌊4*x⌋/4 + ⌊5*x⌋/5 := by
  sorry

end NUMINAMATH_CALUDE_floor_inequality_l222_22293


namespace NUMINAMATH_CALUDE_derricks_yard_length_l222_22239

theorem derricks_yard_length :
  ∀ (derrick_length alex_length brianne_length : ℝ),
    brianne_length = 30 →
    alex_length = derrick_length / 2 →
    brianne_length = 6 * alex_length →
    derrick_length = 10 := by
  sorry

end NUMINAMATH_CALUDE_derricks_yard_length_l222_22239


namespace NUMINAMATH_CALUDE_jacksons_decorations_l222_22281

/-- Given that Mrs. Jackson has 4 boxes of Christmas decorations with 15 decorations in each box
    and she used 35 decorations, prove that she gave 25 decorations to her neighbor. -/
theorem jacksons_decorations (num_boxes : ℕ) (decorations_per_box : ℕ) (used_decorations : ℕ)
    (h1 : num_boxes = 4)
    (h2 : decorations_per_box = 15)
    (h3 : used_decorations = 35) :
    num_boxes * decorations_per_box - used_decorations = 25 := by
  sorry

end NUMINAMATH_CALUDE_jacksons_decorations_l222_22281


namespace NUMINAMATH_CALUDE_rhombus_longer_diagonal_l222_22227

/-- A rhombus with side length 65 and shorter diagonal 72 has a longer diagonal of 108 -/
theorem rhombus_longer_diagonal (side : ℝ) (shorter_diag : ℝ) (longer_diag : ℝ) : 
  side = 65 → shorter_diag = 72 → longer_diag = 108 → 
  side^2 = (shorter_diag / 2)^2 + (longer_diag / 2)^2 := by
sorry

end NUMINAMATH_CALUDE_rhombus_longer_diagonal_l222_22227


namespace NUMINAMATH_CALUDE_krishans_money_krishan_has_4046_l222_22219

/-- Given the ratios of money between Ram, Gopal, and Krishan, and Ram's amount, 
    calculate Krishan's amount. -/
theorem krishans_money 
  (ram_gopal_ratio : ℚ) 
  (gopal_krishan_ratio : ℚ) 
  (ram_money : ℕ) : ℕ :=
  let gopal_money := (ram_money * 17) / 7
  let krishan_money := (gopal_money * 17) / 7
  krishan_money

/-- Prove that Krishan has Rs. 4046 given the problem conditions. -/
theorem krishan_has_4046 :
  krishans_money (7/17) (7/17) 686 = 4046 := by
  sorry

end NUMINAMATH_CALUDE_krishans_money_krishan_has_4046_l222_22219


namespace NUMINAMATH_CALUDE_shaded_cubes_count_l222_22217

/-- Represents a 4x4x4 cube with a specific shading pattern -/
structure ShadedCube where
  /-- Total number of smaller cubes -/
  total_cubes : Nat
  /-- Number of cubes per face -/
  cubes_per_face : Nat
  /-- Number of shaded cubes on one face -/
  shaded_per_face : Nat
  /-- Number of corner cubes -/
  corner_cubes : Nat
  /-- Number of edge cubes -/
  edge_cubes : Nat
  /-- Condition: The cube is 4x4x4 -/
  is_4x4x4 : total_cubes = 64 ∧ cubes_per_face = 16
  /-- Condition: Shading pattern on one face -/
  shading_pattern : shaded_per_face = 9
  /-- Condition: Number of corners and edges -/
  cube_structure : corner_cubes = 8 ∧ edge_cubes = 12

/-- Theorem: The number of shaded cubes in the given 4x4x4 cube is 33 -/
theorem shaded_cubes_count (c : ShadedCube) : 
  c.corner_cubes + c.edge_cubes + (3 * c.shaded_per_face - c.corner_cubes - c.edge_cubes) = 33 := by
  sorry

end NUMINAMATH_CALUDE_shaded_cubes_count_l222_22217


namespace NUMINAMATH_CALUDE_cubic_equation_transformation_l222_22244

theorem cubic_equation_transformation (A B C : ℝ) :
  ∃ (p q β : ℝ), ∀ (z x : ℝ),
    (z^3 + A * z^2 + B * z + C = 0) ↔
    (z = x + β ∧ x^3 + p * x + q = 0) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_transformation_l222_22244


namespace NUMINAMATH_CALUDE_simplify_expression_l222_22214

theorem simplify_expression (a : ℝ) : ((4 * a + 6) - 7 * a) / 3 = -a + 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l222_22214


namespace NUMINAMATH_CALUDE_farmer_cows_l222_22288

theorem farmer_cows (initial_cows : ℕ) (added_cows : ℕ) (sold_fraction : ℚ) 
  (h1 : initial_cows = 51)
  (h2 : added_cows = 5)
  (h3 : sold_fraction = 1/4) :
  initial_cows + added_cows - ⌊(initial_cows + added_cows : ℚ) * sold_fraction⌋ = 42 := by
  sorry

end NUMINAMATH_CALUDE_farmer_cows_l222_22288


namespace NUMINAMATH_CALUDE_find_X_l222_22250

theorem find_X : ∃ X : ℝ, 
  1.5 * ((3.6 * 0.48 * 2.50) / (X * 0.09 * 0.5)) = 1200.0000000000002 ∧ 
  X = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_find_X_l222_22250


namespace NUMINAMATH_CALUDE_height_order_l222_22278

-- Define the set of children
inductive Child : Type
  | A : Child
  | B : Child
  | C : Child
  | D : Child

-- Define the height relation
def taller_than (x y : Child) : Prop := sorry

-- Define the conditions
axiom A_taller_than_B : taller_than Child.A Child.B
axiom B_shorter_than_C : taller_than Child.C Child.B
axiom D_shorter_than_A : taller_than Child.A Child.D
axiom A_not_tallest : ∃ x : Child, taller_than x Child.A
axiom D_not_shortest : ∃ x : Child, taller_than Child.D x

-- Define the order relation
def in_order (w x y z : Child) : Prop :=
  taller_than w x ∧ taller_than x y ∧ taller_than y z

-- State the theorem
theorem height_order : in_order Child.C Child.A Child.D Child.B := by sorry

end NUMINAMATH_CALUDE_height_order_l222_22278


namespace NUMINAMATH_CALUDE_find_a_l222_22200

def U (a : ℤ) : Set ℤ := {2, 4, a^2 - a + 1}

def A (a : ℤ) : Set ℤ := {a+4, 4}

def complement_A (a : ℤ) : Set ℤ := {7}

theorem find_a : ∃ a : ℤ, 
  (U a = {2, 4, a^2 - a + 1}) ∧ 
  (A a = {a+4, 4}) ∧ 
  (complement_A a = {7}) ∧
  (Set.inter (A a) (complement_A a) = ∅) ∧
  (Set.union (A a) (complement_A a) = U a) ∧
  (a = -2) := by sorry

end NUMINAMATH_CALUDE_find_a_l222_22200


namespace NUMINAMATH_CALUDE_price_adjustment_l222_22267

theorem price_adjustment (original_price : ℝ) (original_price_pos : 0 < original_price) : 
  let increased_price := original_price * (1 + 0.25)
  let decrease_percentage := (increased_price - original_price) / increased_price
  decrease_percentage = 0.20 := by
  sorry

end NUMINAMATH_CALUDE_price_adjustment_l222_22267


namespace NUMINAMATH_CALUDE_unique_four_digit_square_l222_22243

/-- A four-digit number -/
def FourDigitNumber (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

/-- Each digit is less than 7 -/
def DigitsLessThan7 (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d < 7

/-- The number is a perfect square -/
def IsPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m^2

/-- The main theorem -/
theorem unique_four_digit_square (N : ℕ) : 
  FourDigitNumber N ∧ 
  DigitsLessThan7 N ∧ 
  IsPerfectSquare N ∧ 
  IsPerfectSquare (N + 3333) → 
  N = 1156 := by
  sorry

end NUMINAMATH_CALUDE_unique_four_digit_square_l222_22243


namespace NUMINAMATH_CALUDE_max_value_abc_l222_22231

theorem max_value_abc (a b c : ℕ+) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) (h4 : a * b * c = 16) :
  (∀ x y z : ℕ+, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x * y * z = 16 →
    x ^ y.val - y ^ z.val + z ^ x.val ≤ a ^ b.val - b ^ c.val + c ^ a.val) →
  a ^ b.val - b ^ c.val + c ^ a.val = 263 :=
by sorry

end NUMINAMATH_CALUDE_max_value_abc_l222_22231


namespace NUMINAMATH_CALUDE_sin_3theta_l222_22296

theorem sin_3theta (θ : ℝ) (h : Complex.exp (θ * Complex.I) = (1 + Complex.I * Real.sqrt 2) / 2) : 
  Real.sin (3 * θ) = Real.sqrt 2 / 8 := by
  sorry

end NUMINAMATH_CALUDE_sin_3theta_l222_22296


namespace NUMINAMATH_CALUDE_payment_calculation_l222_22257

/-- The payment for C given the work rates of A and B, total payment, and total work days -/
def payment_for_C (a_rate : ℚ) (b_rate : ℚ) (total_payment : ℚ) (total_days : ℚ) : ℚ :=
  let ab_rate := a_rate + b_rate
  let ab_work := ab_rate * total_days
  let c_work := 1 - ab_work
  c_work * total_payment

theorem payment_calculation (a_rate b_rate total_payment total_days : ℚ) 
  (ha : a_rate = 1/6)
  (hb : b_rate = 1/8)
  (hp : total_payment = 3360)
  (hd : total_days = 3) :
  payment_for_C a_rate b_rate total_payment total_days = 420 := by
  sorry

#eval payment_for_C (1/6) (1/8) 3360 3

end NUMINAMATH_CALUDE_payment_calculation_l222_22257


namespace NUMINAMATH_CALUDE_product_of_fraction_parts_l222_22226

/-- The decimal representation of the number we're considering -/
def repeating_decimal : ℚ := 0.018018018018018018018018018018018018018018018018018

/-- Express the repeating decimal as a fraction in lowest terms -/
def decimal_to_fraction (d : ℚ) : ℚ := d

/-- Calculate the product of numerator and denominator of a fraction -/
def numerator_denominator_product (q : ℚ) : ℕ :=
  (q.num.natAbs) * (q.den)

/-- Theorem stating that the product of numerator and denominator of 0.018̅ in lowest terms is 222 -/
theorem product_of_fraction_parts : 
  numerator_denominator_product (decimal_to_fraction repeating_decimal) = 222 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fraction_parts_l222_22226


namespace NUMINAMATH_CALUDE_smallest_number_with_remainders_l222_22282

theorem smallest_number_with_remainders : ∃ (a : ℕ), 
  (a % 3 = 2) ∧ (a % 5 = 3) ∧ (a % 7 = 3) ∧
  (∀ (b : ℕ), b < a → ¬((b % 3 = 2) ∧ (b % 5 = 3) ∧ (b % 7 = 3))) ∧
  a = 98 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_with_remainders_l222_22282


namespace NUMINAMATH_CALUDE_sum_of_cubes_l222_22205

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 2) (h2 : x * y = 3) : x^3 + y^3 = -10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l222_22205


namespace NUMINAMATH_CALUDE_ship_placement_theorem_l222_22289

/-- Represents a ship on the grid -/
structure Ship :=
  (length : Nat)
  (width : Nat)

/-- Represents the grid -/
def Grid := Fin 10 → Fin 10 → Bool

/-- Checks if a ship placement is valid -/
def isValidPlacement (grid : Grid) (ship : Ship) (x y : Fin 10) : Bool :=
  sorry

/-- Places a ship on the grid -/
def placeShip (grid : Grid) (ship : Ship) (x y : Fin 10) : Grid :=
  sorry

/-- List of ships to be placed -/
def ships : List Ship :=
  [⟨4, 1⟩, ⟨3, 1⟩, ⟨3, 1⟩, ⟨2, 1⟩, ⟨2, 1⟩, ⟨2, 1⟩, ⟨1, 1⟩, ⟨1, 1⟩, ⟨1, 1⟩, ⟨1, 1⟩]

/-- Attempts to place all ships on the grid -/
def placeAllShips (grid : Grid) (ships : List Ship) : Option Grid :=
  sorry

theorem ship_placement_theorem :
  (∃ (grid : Grid), placeAllShips grid ships = some grid) ∧
  (∃ (grid : Grid), placeAllShips grid (ships.reverse) = none) :=
by sorry

end NUMINAMATH_CALUDE_ship_placement_theorem_l222_22289


namespace NUMINAMATH_CALUDE_batsman_running_fraction_l222_22237

/-- Represents the score of a batsman in cricket --/
structure BatsmanScore where
  total_runs : ℕ
  boundaries : ℕ
  sixes : ℕ

/-- Calculates the fraction of runs made by running between wickets --/
def runningFraction (score : BatsmanScore) : ℚ :=
  let boundary_runs := 4 * score.boundaries
  let six_runs := 6 * score.sixes
  let running_runs := score.total_runs - (boundary_runs + six_runs)
  (running_runs : ℚ) / score.total_runs

theorem batsman_running_fraction :
  let score : BatsmanScore := ⟨250, 15, 10⟩
  runningFraction score = 13 / 25 := by
  sorry

end NUMINAMATH_CALUDE_batsman_running_fraction_l222_22237


namespace NUMINAMATH_CALUDE_root_square_minus_two_plus_2023_l222_22209

theorem root_square_minus_two_plus_2023 (m : ℝ) :
  m^2 - 2*m - 3 = 0 → m^2 - 2*m + 2023 = 2026 := by
  sorry

end NUMINAMATH_CALUDE_root_square_minus_two_plus_2023_l222_22209


namespace NUMINAMATH_CALUDE_yz_minus_zx_minus_xy_l222_22206

theorem yz_minus_zx_minus_xy (x y z : ℝ) 
  (h1 : x - y - z = 19) 
  (h2 : x^2 + y^2 + z^2 ≠ 19) : 
  y*z - z*x - x*y = 171 := by sorry

end NUMINAMATH_CALUDE_yz_minus_zx_minus_xy_l222_22206


namespace NUMINAMATH_CALUDE_female_grade_one_jiu_is_set_l222_22241

-- Define the universe of students
def Student : Type := sorry

-- Define the property of being female
def is_female : Student → Prop := sorry

-- Define the property of being in grade one of Jiu Middle School
def is_grade_one_jiu : Student → Prop := sorry

-- Define our set
def female_grade_one_jiu : Set Student :=
  {s : Student | is_female s ∧ is_grade_one_jiu s}

-- Theorem stating that female_grade_one_jiu is a well-defined set
theorem female_grade_one_jiu_is_set :
  ∀ (s : Student), Decidable (s ∈ female_grade_one_jiu) :=
sorry

end NUMINAMATH_CALUDE_female_grade_one_jiu_is_set_l222_22241


namespace NUMINAMATH_CALUDE_sum_of_i_powers_l222_22291

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem sum_of_i_powers : i^23 + i^28 + i^33 + i^38 + i^43 = -i := by
  sorry

end NUMINAMATH_CALUDE_sum_of_i_powers_l222_22291


namespace NUMINAMATH_CALUDE_factorization_mx_minus_my_l222_22253

theorem factorization_mx_minus_my (m x y : ℝ) : m * x - m * y = m * (x - y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_mx_minus_my_l222_22253


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l222_22249

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 9/y = 2) :
  ∃ (m : ℝ), (∀ a b : ℝ, a > 0 → b > 0 → 1/a + 9/b = 2 → x + y ≤ a + b) ∧ x + y = m ∧ m = 8 :=
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l222_22249


namespace NUMINAMATH_CALUDE_probability_at_least_one_vowel_l222_22270

structure LetterSet where
  letters : Finset Char
  vowels : Finset Char
  vowels_subset : vowels ⊆ letters

def probability_no_vowel (s : LetterSet) : ℚ :=
  (s.letters.card - s.vowels.card : ℚ) / s.letters.card

def set1 : LetterSet := {
  letters := {'a', 'b', 'c', 'd', 'e'},
  vowels := {'a', 'e'},
  vowels_subset := by simp
}

def set2 : LetterSet := {
  letters := {'k', 'l', 'm', 'n', 'o', 'p'},
  vowels := ∅,
  vowels_subset := by simp
}

def set3 : LetterSet := {
  letters := {'r', 's', 't', 'u', 'v'},
  vowels := ∅,
  vowels_subset := by simp
}

def set4 : LetterSet := {
  letters := {'w', 'x', 'y', 'z', 'i'},
  vowels := {'i'},
  vowels_subset := by simp
}

theorem probability_at_least_one_vowel :
  1 - (probability_no_vowel set1 * probability_no_vowel set2 * 
       probability_no_vowel set3 * probability_no_vowel set4) = 17 / 20 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_vowel_l222_22270


namespace NUMINAMATH_CALUDE_debate_tournament_participants_l222_22271

theorem debate_tournament_participants (initial_participants : ℕ) : 
  (initial_participants : ℝ) * 0.4 * 0.25 = 30 → initial_participants = 300 := by
  sorry

end NUMINAMATH_CALUDE_debate_tournament_participants_l222_22271


namespace NUMINAMATH_CALUDE_no_intersection_point_l222_22274

theorem no_intersection_point :
  ¬ ∃ (x y : ℝ), 
    (3 * x + 4 * y - 12 = 0) ∧ 
    (5 * x - 4 * y - 10 = 0) ∧ 
    (x = 3) ∧ 
    (y = -1/2) := by
  sorry

end NUMINAMATH_CALUDE_no_intersection_point_l222_22274


namespace NUMINAMATH_CALUDE_supplement_not_always_greater_l222_22201

/-- The supplement of an angle (in degrees) -/
def supplement (x : ℝ) : ℝ := 180 - x

/-- Theorem stating that the statement "The supplement of an angle is always greater than the angle itself" is false -/
theorem supplement_not_always_greater (x : ℝ) : ¬ (∀ x, supplement x > x) := by
  sorry

end NUMINAMATH_CALUDE_supplement_not_always_greater_l222_22201


namespace NUMINAMATH_CALUDE_oil_mixture_price_l222_22280

/-- Given two types of oil mixed together, calculate the price of the second oil. -/
theorem oil_mixture_price (volume1 volume2 total_volume : ℚ) (price1 mixture_price : ℚ) :
  volume1 = 10 →
  volume2 = 5 →
  total_volume = volume1 + volume2 →
  price1 = 54 →
  mixture_price = 58 →
  ∃ price2 : ℚ, 
    price2 = 66 ∧
    volume1 * price1 + volume2 * price2 = total_volume * mixture_price :=
by sorry

end NUMINAMATH_CALUDE_oil_mixture_price_l222_22280


namespace NUMINAMATH_CALUDE_specific_kite_area_l222_22232

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a kite shape -/
structure Kite where
  v1 : Point
  v2 : Point
  v3 : Point
  v4 : Point

/-- Represents a square -/
structure Square where
  center : Point
  sideLength : ℝ

/-- Calculates the area of a kite with an internal square -/
def kiteArea (k : Kite) (s : Square) : ℝ :=
  sorry

/-- The theorem stating the area of the specific kite -/
theorem specific_kite_area :
  let k : Kite := {
    v1 := {x := 1, y := 6},
    v2 := {x := 4, y := 7},
    v3 := {x := 7, y := 6},
    v4 := {x := 4, y := 0}
  }
  let s : Square := {
    center := {x := 4, y := 3},
    sideLength := 2
  }
  kiteArea k s = 10 := by
  sorry

end NUMINAMATH_CALUDE_specific_kite_area_l222_22232


namespace NUMINAMATH_CALUDE_xiaolis_estimate_l222_22234

theorem xiaolis_estimate (x y z w : ℝ) (hx : x > y) (hy : y > 0) (hz : z > 0) (hw : w > 0) :
  (x + z) - (y - w) > x - y := by
  sorry

end NUMINAMATH_CALUDE_xiaolis_estimate_l222_22234


namespace NUMINAMATH_CALUDE_floor_sqrt_50_l222_22245

theorem floor_sqrt_50 : ⌊Real.sqrt 50⌋ = 7 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_50_l222_22245


namespace NUMINAMATH_CALUDE_company_employees_l222_22212

/-- 
If a company had 15% more employees in December than in January,
and it had 450 employees in December, then it had 391 employees in January.
-/
theorem company_employees (december_employees : ℕ) (january_employees : ℕ) : 
  december_employees = 450 → 
  december_employees = january_employees + (january_employees * 15 / 100) →
  january_employees = 391 := by
sorry

end NUMINAMATH_CALUDE_company_employees_l222_22212


namespace NUMINAMATH_CALUDE_x_coordinate_range_l222_22247

-- Define the circle M
def circle_M (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 4

-- Define the line l
def line_l (x y : ℝ) : Prop := x + y = 6

-- Define a point on the circle
def point_on_circle (x y : ℝ) : Prop := circle_M x y

-- Define a point on the line
def point_on_line (x y : ℝ) : Prop := line_l x y

-- Define the angle between three points
def angle (A B C : ℝ × ℝ) : ℝ := sorry

-- Main theorem
theorem x_coordinate_range :
  ∀ (A B C : ℝ × ℝ),
  point_on_line A.1 A.2 →
  point_on_circle B.1 B.2 →
  point_on_circle C.1 C.2 →
  angle A B C = π/3 →
  1 ≤ A.1 ∧ A.1 ≤ 5 :=
by sorry

end NUMINAMATH_CALUDE_x_coordinate_range_l222_22247


namespace NUMINAMATH_CALUDE_no_intersection_l222_22230

/-- Definition of a parabola -/
def is_parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- Definition of a point inside the parabola -/
def is_inside_parabola (x₀ y₀ : ℝ) : Prop := y₀^2 < 4*x₀

/-- Definition of the line -/
def line_equation (x₀ y₀ x y : ℝ) : Prop := y₀*y = 2*(x + x₀)

/-- Theorem stating that a line passing through a point inside the parabola has no intersection with the parabola -/
theorem no_intersection (x₀ y₀ : ℝ) :
  is_inside_parabola x₀ y₀ →
  ∀ x y : ℝ, is_parabola x y ∧ line_equation x₀ y₀ x y → False :=
sorry

end NUMINAMATH_CALUDE_no_intersection_l222_22230


namespace NUMINAMATH_CALUDE_sin_derivative_bound_and_inequality_range_l222_22216

noncomputable def f (x : ℝ) : ℝ := Real.sin x

theorem sin_derivative_bound_and_inequality_range :
  (∀ x > 0, (deriv f) x > 1 - x^2 / 2) ∧
  (∀ a : ℝ, (∀ x ∈ Set.Ioo 0 (Real.pi / 2), f x + f x / (deriv f) x > a * x) → a ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_sin_derivative_bound_and_inequality_range_l222_22216


namespace NUMINAMATH_CALUDE_half_minus_quarter_equals_two_l222_22246

theorem half_minus_quarter_equals_two (n : ℝ) : n = 8 → (0.5 * n) - (0.25 * n) = 2 := by
  sorry

end NUMINAMATH_CALUDE_half_minus_quarter_equals_two_l222_22246


namespace NUMINAMATH_CALUDE_smallest_undefined_value_l222_22210

theorem smallest_undefined_value (x : ℝ) : 
  (∀ y < (1/4 : ℝ), (10*y^2 - 90*y + 20) ≠ 0) ∧ 
  (10*(1/4 : ℝ)^2 - 90*(1/4 : ℝ) + 20 = 0) := by
  sorry

end NUMINAMATH_CALUDE_smallest_undefined_value_l222_22210


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l222_22285

/-- Given a geometric sequence with common ratio 2, prove that (2a₁ + a₂) / (2a₃ + a₄) = 1/4 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) :
  (∀ n, a (n + 1) = 2 * a n) →
  (2 * a 1 + a 2) / (2 * a 3 + a 4) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l222_22285


namespace NUMINAMATH_CALUDE_marble_196_is_green_l222_22233

/-- Represents the color of a marble -/
inductive MarbleColor
  | Red
  | Green
  | Blue

/-- Returns the color of the nth marble in the sequence -/
def marbleColor (n : ℕ) : MarbleColor :=
  match n % 12 with
  | 0 | 1 | 2 => MarbleColor.Red
  | 3 | 4 | 5 | 6 | 7 => MarbleColor.Green
  | _ => MarbleColor.Blue

/-- Theorem stating that the 196th marble is green -/
theorem marble_196_is_green : marbleColor 196 = MarbleColor.Green := by
  sorry


end NUMINAMATH_CALUDE_marble_196_is_green_l222_22233


namespace NUMINAMATH_CALUDE_sum_of_two_squares_l222_22277

theorem sum_of_two_squares (a b : ℝ) : 2 * a^2 + 2 * b^2 = (a + b)^2 + (a - b)^2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_squares_l222_22277


namespace NUMINAMATH_CALUDE_deepak_age_l222_22255

theorem deepak_age (rahul_ratio : ℕ) (deepak_ratio : ℕ) (rahul_future_age : ℕ) (years_ahead : ℕ) :
  rahul_ratio = 4 →
  deepak_ratio = 2 →
  rahul_future_age = 26 →
  years_ahead = 10 →
  ∃ (x : ℕ), rahul_ratio * x + years_ahead = rahul_future_age ∧ deepak_ratio * x = 8 :=
by sorry

end NUMINAMATH_CALUDE_deepak_age_l222_22255


namespace NUMINAMATH_CALUDE_lawn_mowing_l222_22229

theorem lawn_mowing (mary_rate tom_rate : ℚ)
  (h1 : mary_rate = 1 / 3)
  (h2 : tom_rate = 1 / 6)
  (total_lawn : ℚ)
  (h3 : total_lawn = 1) :
  let combined_rate := mary_rate + tom_rate
  let mowed_together := combined_rate * 1
  let mowed_mary_alone := mary_rate * 1
  let total_mowed := mowed_together + mowed_mary_alone
  total_lawn - total_mowed = 1 / 6 := by
sorry

end NUMINAMATH_CALUDE_lawn_mowing_l222_22229


namespace NUMINAMATH_CALUDE_joan_sold_26_books_l222_22202

/-- The number of books Joan sold in the yard sale -/
def books_sold (initial_books : ℕ) (remaining_books : ℕ) : ℕ :=
  initial_books - remaining_books

/-- Proof that Joan sold 26 books -/
theorem joan_sold_26_books (initial_books : ℕ) (remaining_books : ℕ) 
  (h1 : initial_books = 33) (h2 : remaining_books = 7) : 
  books_sold initial_books remaining_books = 26 := by
  sorry

#eval books_sold 33 7

end NUMINAMATH_CALUDE_joan_sold_26_books_l222_22202


namespace NUMINAMATH_CALUDE_a_eq_b_sufficient_a_eq_b_not_necessary_l222_22276

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x : ℝ | x^2 - x + a ≤ 0}
def B (b : ℝ) : Set ℝ := {x : ℝ | x^2 - x + b ≤ 0}

-- Theorem stating that a = b is sufficient for A = B
theorem a_eq_b_sufficient (a b : ℝ) : a = b → A a = B b := by sorry

-- Theorem stating that a = b is not necessary for A = B
theorem a_eq_b_not_necessary : ∃ a b : ℝ, A a = B b ∧ a ≠ b := by sorry

end NUMINAMATH_CALUDE_a_eq_b_sufficient_a_eq_b_not_necessary_l222_22276


namespace NUMINAMATH_CALUDE_mrs_wilsborough_tickets_l222_22292

def prove_regular_tickets_bought : Prop :=
  let initial_savings : ℕ := 500
  let vip_ticket_cost : ℕ := 100
  let vip_tickets_bought : ℕ := 2
  let regular_ticket_cost : ℕ := 50
  let money_left : ℕ := 150
  let total_spent : ℕ := initial_savings - money_left
  let vip_tickets_total_cost : ℕ := vip_ticket_cost * vip_tickets_bought
  let regular_tickets_total_cost : ℕ := total_spent - vip_tickets_total_cost
  let regular_tickets_bought : ℕ := regular_tickets_total_cost / regular_ticket_cost
  regular_tickets_bought = 3

theorem mrs_wilsborough_tickets : prove_regular_tickets_bought := by
  sorry

end NUMINAMATH_CALUDE_mrs_wilsborough_tickets_l222_22292


namespace NUMINAMATH_CALUDE_sum_of_fractions_l222_22295

theorem sum_of_fractions : 
  (3 : ℚ) / 100 + (2 : ℚ) / 1000 + (8 : ℚ) / 10000 + (5 : ℚ) / 100000 = 0.03285 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l222_22295


namespace NUMINAMATH_CALUDE_banana_production_theorem_l222_22252

/-- The total banana production from two islands -/
def total_banana_production (jakies_production : ℕ) (nearby_production : ℕ) : ℕ :=
  jakies_production + nearby_production

/-- Theorem stating the total banana production from Jakies Island and a nearby island -/
theorem banana_production_theorem (nearby_production : ℕ) 
  (h1 : nearby_production = 9000)
  (h2 : ∃ (jakies_production : ℕ), jakies_production = 10 * nearby_production) :
  ∃ (total_production : ℕ), total_production = 99000 ∧ 
  total_production = total_banana_production (10 * nearby_production) nearby_production :=
by
  sorry


end NUMINAMATH_CALUDE_banana_production_theorem_l222_22252


namespace NUMINAMATH_CALUDE_froglet_is_sane_l222_22207

-- Define the servants
inductive Servant
| LackeyLecc
| Froglet

-- Define the sanity state
inductive SanityState
| Sane
| Insane

-- Define a function to represent the claim of Lackey-Lecc
def lackey_lecc_claim (lackey_state froglet_state : SanityState) : Prop :=
  (lackey_state = SanityState.Sane ∧ froglet_state = SanityState.Sane) ∨
  (lackey_state = SanityState.Insane ∧ froglet_state = SanityState.Insane)

-- Theorem stating that Froglet is sane
theorem froglet_is_sane :
  ∀ (lackey_state : SanityState),
    (lackey_lecc_claim lackey_state SanityState.Sane) →
    SanityState.Sane = SanityState.Sane :=
by
  sorry


end NUMINAMATH_CALUDE_froglet_is_sane_l222_22207


namespace NUMINAMATH_CALUDE_probability_of_double_l222_22272

/-- Represents a domino with two ends --/
structure Domino :=
  (end1 : Nat)
  (end2 : Nat)

/-- A standard set of dominoes with numbers from 0 to 6 --/
def StandardDominoSet : Set Domino :=
  {d : Domino | d.end1 ≤ 6 ∧ d.end2 ≤ 6}

/-- Predicate for a double domino --/
def IsDouble (d : Domino) : Prop :=
  d.end1 = d.end2

/-- The total number of dominoes in a standard set --/
def TotalDominoes : Nat := 28

/-- The number of doubles in a standard set --/
def NumberOfDoubles : Nat := 7

theorem probability_of_double :
  (NumberOfDoubles : ℚ) / (TotalDominoes : ℚ) = 1 / 4 := by
  sorry


end NUMINAMATH_CALUDE_probability_of_double_l222_22272


namespace NUMINAMATH_CALUDE_perpendicular_tangents_circles_l222_22238

/-- Two circles with perpendicular tangents at intersection points -/
theorem perpendicular_tangents_circles (a : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 + 4*y = 0 ∧ x^2 + y^2 + 2*(a-1)*x + 2*y + a^2 = 0 →
    ∃ m n : ℝ, 
      m^2 + n^2 + 4*n = 0 ∧
      2*(a-1)*m - 2*n + a^2 = 0 ∧
      (n + 2) / m * (n + 1) / (m - (1 - a)) = -1) →
  a = -2 :=
sorry

end NUMINAMATH_CALUDE_perpendicular_tangents_circles_l222_22238


namespace NUMINAMATH_CALUDE_set_equality_l222_22279

-- Define sets A and B
def A : Set ℝ := {x | x < 4}
def B : Set ℝ := {x | x^2 - 4*x + 3 > 0}

-- Define the set we want to prove equal to our result
def S : Set ℝ := {x | x ∈ A ∧ x ∉ A ∩ B}

-- State the theorem
theorem set_equality : S = {x : ℝ | 1 ≤ x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_set_equality_l222_22279


namespace NUMINAMATH_CALUDE_stream_speed_l222_22236

/-- Given a river with stream speed v and a rower with speed u in still water,
    if the rower travels 27 km upstream and 81 km downstream, each in 9 hours,
    then the speed of the stream v is 3 km/h. -/
theorem stream_speed (v u : ℝ) 
  (h1 : 27 / (u - v) = 9)  -- Upstream condition
  (h2 : 81 / (u + v) = 9)  -- Downstream condition
  : v = 3 := by
  sorry


end NUMINAMATH_CALUDE_stream_speed_l222_22236


namespace NUMINAMATH_CALUDE_min_roots_count_l222_22242

def is_symmetric_about (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (a + x) = f (a - x)

theorem min_roots_count
  (f : ℝ → ℝ)
  (h1 : is_symmetric_about f 2)
  (h2 : is_symmetric_about f 7)
  (h3 : f 0 = 0) :
  ∃ N : ℕ, N ≥ 401 ∧
  (∀ m : ℕ, (∃ S : Finset ℝ, S.card = m ∧
    (∀ x ∈ S, -1000 ≤ x ∧ x ≤ 1000 ∧ f x = 0)) →
    m ≤ N) :=
  sorry

end NUMINAMATH_CALUDE_min_roots_count_l222_22242


namespace NUMINAMATH_CALUDE_set_operations_l222_22262

open Set

-- Define the universe set U
def U : Set ℤ := {x | 0 < x ∧ x ≤ 10}

-- Define sets A, B, and C
def A : Set ℤ := {1, 2, 4, 5, 9}
def B : Set ℤ := {4, 6, 7, 8, 10}
def C : Set ℤ := {3, 5, 7}

theorem set_operations :
  (A ∩ B = {4}) ∧
  (A ∪ B = {1, 2, 4, 5, 6, 7, 8, 9, 10}) ∧
  ((U \ A) ∩ (U \ B) = {3}) ∧
  ((U \ A) ∪ (U \ B) = {1, 2, 3, 5, 6, 7, 8, 9, 10}) ∧
  ((A ∩ B) ∩ C = ∅) ∧
  ((A ∪ B) ∩ C = {5, 7}) := by
sorry


end NUMINAMATH_CALUDE_set_operations_l222_22262


namespace NUMINAMATH_CALUDE_ratio_satisfies_conditions_l222_22223

/-- Represents the number of people in each profession --/
structure ProfessionCount where
  doctors : ℕ
  lawyers : ℕ
  engineers : ℕ

/-- Checks if the given counts satisfy the average age conditions --/
def satisfiesAverageConditions (count : ProfessionCount) : Prop :=
  let totalPeople := count.doctors + count.lawyers + count.engineers
  let totalAge := 40 * count.doctors + 50 * count.lawyers + 60 * count.engineers
  totalAge / totalPeople = 45

/-- The theorem stating that the ratio 3:6:1 satisfies the conditions --/
theorem ratio_satisfies_conditions :
  ∃ (k : ℕ), k > 0 ∧ 
    let count : ProfessionCount := ⟨3*k, 6*k, k⟩
    satisfiesAverageConditions count :=
sorry

end NUMINAMATH_CALUDE_ratio_satisfies_conditions_l222_22223


namespace NUMINAMATH_CALUDE_cantor_is_founder_l222_22259

/-- Represents a mathematician -/
inductive Mathematician
  | Gauss
  | Dedekind
  | Weierstrass
  | Cantor

/-- Represents the founder of modern set theory -/
def founder_of_modern_set_theory : Mathematician := Mathematician.Cantor

/-- Theorem stating that Cantor is the founder of modern set theory -/
theorem cantor_is_founder : 
  founder_of_modern_set_theory = Mathematician.Cantor := by sorry

end NUMINAMATH_CALUDE_cantor_is_founder_l222_22259


namespace NUMINAMATH_CALUDE_expression_positivity_l222_22260

theorem expression_positivity (x : ℝ) : (x + 2) * (x - 3) > 0 ↔ x < -2 ∨ x > 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_positivity_l222_22260


namespace NUMINAMATH_CALUDE_tangerine_persimmon_ratio_l222_22218

theorem tangerine_persimmon_ratio :
  let apples : ℕ := 24
  let tangerines : ℕ := 6 * apples
  let persimmons : ℕ := 8
  tangerines = 18 * persimmons :=
by
  sorry

end NUMINAMATH_CALUDE_tangerine_persimmon_ratio_l222_22218


namespace NUMINAMATH_CALUDE_floor_abs_negative_l222_22228

theorem floor_abs_negative : ⌊|(-45.8 : ℝ)|⌋ = 45 := by
  sorry

end NUMINAMATH_CALUDE_floor_abs_negative_l222_22228


namespace NUMINAMATH_CALUDE_sqrt_sum_squares_eq_sum_l222_22275

theorem sqrt_sum_squares_eq_sum (a b c : ℝ) :
  Real.sqrt (a^2 + b^2 + c^2) = a + b + c ↔ a*b + a*c + b*c = 0 ∧ a + b + c ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_squares_eq_sum_l222_22275


namespace NUMINAMATH_CALUDE_exponent_multiplication_l222_22264

theorem exponent_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l222_22264


namespace NUMINAMATH_CALUDE_point_on_600_degree_angle_l222_22256

/-- If a point (-4, a) lies on the terminal side of an angle of 600°, then a = -4√3 -/
theorem point_on_600_degree_angle (a : ℝ) : 
  (∃ θ : ℝ, θ = 600 * π / 180 ∧ Real.tan θ = a / (-4)) → a = -4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_point_on_600_degree_angle_l222_22256


namespace NUMINAMATH_CALUDE_books_per_shelf_l222_22299

theorem books_per_shelf 
  (total_shelves : ℕ) 
  (total_books : ℕ) 
  (h1 : total_shelves = 14240)
  (h2 : total_books = 113920) :
  total_books / total_shelves = 8 :=
by sorry

end NUMINAMATH_CALUDE_books_per_shelf_l222_22299


namespace NUMINAMATH_CALUDE_solve_system_l222_22297

theorem solve_system (t : ℝ) (x y : ℝ) 
  (h1 : x = 3 - 2*t) 
  (h2 : y = 5*t + 6) 
  (h3 : x = 1) : 
  y = 11 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l222_22297


namespace NUMINAMATH_CALUDE_andy_solves_56_problems_l222_22286

/-- The number of problems Andy solves -/
def problems_solved (first last : ℕ) : ℕ := last - first + 1

/-- Theorem stating that Andy solves 56 problems -/
theorem andy_solves_56_problems : 
  problems_solved 70 125 = 56 := by sorry

end NUMINAMATH_CALUDE_andy_solves_56_problems_l222_22286


namespace NUMINAMATH_CALUDE_calculate_total_profit_total_profit_is_150000_l222_22273

/-- Calculates the total profit given investment ratios and B's profit -/
theorem calculate_total_profit (a_c_ratio : Rat) (a_b_ratio : Rat) (b_profit : ℕ) : ℕ :=
  let a_c_ratio := 2/1
  let a_b_ratio := 2/3
  let b_profit := 75000
  2 * b_profit

theorem total_profit_is_150000 : 
  calculate_total_profit (2/1) (2/3) 75000 = 150000 := by
  sorry

end NUMINAMATH_CALUDE_calculate_total_profit_total_profit_is_150000_l222_22273


namespace NUMINAMATH_CALUDE_absolute_value_equation_unique_solution_l222_22261

theorem absolute_value_equation_unique_solution :
  ∃! x : ℝ, |x - 10| = |x + 4| := by
sorry

end NUMINAMATH_CALUDE_absolute_value_equation_unique_solution_l222_22261


namespace NUMINAMATH_CALUDE_alfred_maize_storage_l222_22294

/-- Proves that Alfred stores 1 tonne of maize per month given the conditions -/
theorem alfred_maize_storage (x : ℝ) : 
  24 * x - 5 + 8 = 27 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_alfred_maize_storage_l222_22294


namespace NUMINAMATH_CALUDE_trailing_zeros_bound_l222_22290

theorem trailing_zeros_bound (n : ℕ) : ∃ (k : ℕ), k ≤ 2 ∧ (1^n + 2^n + 3^n + 4^n) % 10^(k+1) ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_trailing_zeros_bound_l222_22290


namespace NUMINAMATH_CALUDE_total_samplers_percentage_l222_22203

/-- Represents the percentage of customers for a specific candy type -/
structure CandyData where
  caught : ℝ
  notCaught : ℝ

/-- Represents the data for all candy types -/
structure CandyStore where
  A : CandyData
  B : CandyData
  C : CandyData
  D : CandyData

/-- Calculates the total percentage of customers who sample any type of candy -/
def totalSamplers (store : CandyStore) : ℝ :=
  store.A.caught + store.A.notCaught +
  store.B.caught + store.B.notCaught +
  store.C.caught + store.C.notCaught +
  store.D.caught + store.D.notCaught

/-- The candy store data -/
def candyStoreData : CandyStore :=
  { A := { caught := 12, notCaught := 7 }
    B := { caught := 5,  notCaught := 6 }
    C := { caught := 9,  notCaught := 3 }
    D := { caught := 4,  notCaught := 8 } }

theorem total_samplers_percentage :
  totalSamplers candyStoreData = 54 := by sorry

end NUMINAMATH_CALUDE_total_samplers_percentage_l222_22203


namespace NUMINAMATH_CALUDE_max_ratio_squared_l222_22266

theorem max_ratio_squared (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≥ b) :
  (∃ (ρ : ℝ), ∀ (x y : ℝ), 
    (0 ≤ x ∧ x < a) → 
    (0 ≤ y ∧ y < b) → 
    (a^2 + y^2 = b^2 + x^2 ∧ b^2 + x^2 = (a + x)^2 + (b - y)^2) →
    (a / b)^2 ≤ ρ^2 ∧
    ρ^2 = 4/3) :=
sorry

end NUMINAMATH_CALUDE_max_ratio_squared_l222_22266


namespace NUMINAMATH_CALUDE_property_necessary_not_sufficient_l222_22251

-- Define a real-valued function on ℝ
variable (f : ℝ → ℝ)

-- Define the property that f(x+1) > f(x) for all x ∈ ℝ
def property_f (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 1) > f x

-- Define what it means for a function to be increasing on ℝ
def increasing_on_reals (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

-- Theorem stating that the property is necessary but not sufficient
-- for the function to be increasing on ℝ
theorem property_necessary_not_sufficient :
  (∀ f : ℝ → ℝ, increasing_on_reals f → property_f f) ∧
  ¬(∀ f : ℝ → ℝ, property_f f → increasing_on_reals f) :=
sorry

end NUMINAMATH_CALUDE_property_necessary_not_sufficient_l222_22251


namespace NUMINAMATH_CALUDE_max_ab_value_l222_22269

theorem max_ab_value (a b c : ℝ) : 
  (∀ x : ℝ, 2*x + 2 ≤ a*x^2 + b*x + c ∧ a*x^2 + b*x + c ≤ 2*x^2 - 2*x + 4) →
  a*b ≤ (1/2 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_max_ab_value_l222_22269


namespace NUMINAMATH_CALUDE_rational_function_property_l222_22225

theorem rational_function_property (f : ℚ → ℝ) 
  (add_prop : ∀ x y : ℚ, f (x + y) = f x + f y)
  (mul_prop : ∀ x y : ℚ, f (x * y) = f x * f y) :
  (∀ x : ℚ, f x = x) ∨ (∀ x : ℚ, f x = 0) := by
sorry

end NUMINAMATH_CALUDE_rational_function_property_l222_22225


namespace NUMINAMATH_CALUDE_millipede_segment_ratio_l222_22284

/-- Proves that the ratio of segments of two unknown-length millipedes to a 60-segment millipede is 4:1 --/
theorem millipede_segment_ratio : 
  ∀ (x : ℕ), -- x represents the number of segments in each of the two unknown-length millipedes
  (2 * x + 60 + 500 = 800) → -- Total segments equation
  ((2 * x) : ℚ) / 60 = 4 / 1 := by
sorry

end NUMINAMATH_CALUDE_millipede_segment_ratio_l222_22284


namespace NUMINAMATH_CALUDE_greatest_base_nine_digit_sum_l222_22220

def base_nine_digit_sum (n : ℕ) : ℕ :=
  (n.digits 9).sum

theorem greatest_base_nine_digit_sum :
  ∃ (m : ℕ), m < 2500 ∧ base_nine_digit_sum m = 24 ∧
  ∀ (n : ℕ), n < 2500 → base_nine_digit_sum n ≤ 24 := by
  sorry

end NUMINAMATH_CALUDE_greatest_base_nine_digit_sum_l222_22220


namespace NUMINAMATH_CALUDE_reporter_earnings_l222_22298

/-- A reporter's earnings calculation --/
theorem reporter_earnings 
  (earnings_per_article : ℝ) 
  (articles : ℕ) 
  (total_hours : ℝ) 
  (words_per_minute : ℝ) 
  (earnings_per_hour : ℝ) 
  (h1 : earnings_per_article = 60) 
  (h2 : articles = 3) 
  (h3 : total_hours = 4) 
  (h4 : words_per_minute = 10) 
  (h5 : earnings_per_hour = 105) : 
  let total_earnings := earnings_per_hour * total_hours
  let total_words := words_per_minute * (total_hours * 60)
  let article_earnings := earnings_per_article * articles
  let word_earnings := total_earnings - article_earnings
  word_earnings / total_words = 0.1 := by
sorry

end NUMINAMATH_CALUDE_reporter_earnings_l222_22298


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l222_22235

theorem regular_polygon_sides (n : ℕ) (exterior_angle : ℝ) :
  n > 2 →
  exterior_angle = 40 * Real.pi / 180 →
  exterior_angle = (2 * Real.pi) / n →
  n = 9 := by
sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l222_22235


namespace NUMINAMATH_CALUDE_smallest_sqrt_x_minus_one_l222_22211

theorem smallest_sqrt_x_minus_one :
  ∀ x : ℝ, 
    (Real.sqrt (x - 1) ≥ 0) ∧ 
    (Real.sqrt (x - 1) = 0 ↔ x = 1) :=
by sorry

end NUMINAMATH_CALUDE_smallest_sqrt_x_minus_one_l222_22211


namespace NUMINAMATH_CALUDE_trapezoid_area_l222_22204

/-- The area of a trapezoid with given base lengths and leg lengths -/
theorem trapezoid_area (b1 b2 l1 l2 : ℝ) (h : ℝ) 
  (hb1 : b1 = 10) 
  (hb2 : b2 = 21) 
  (hl1 : l1 = Real.sqrt 34) 
  (hl2 : l2 = 3 * Real.sqrt 5) 
  (hh : h^2 + 5^2 = 34) : 
  (b1 + b2) * h / 2 = 93 / 2 := by
  sorry

#check trapezoid_area

end NUMINAMATH_CALUDE_trapezoid_area_l222_22204


namespace NUMINAMATH_CALUDE_number_and_square_sum_l222_22224

theorem number_and_square_sum (x : ℝ) : x + x^2 = 306 → x = 17 := by
  sorry

end NUMINAMATH_CALUDE_number_and_square_sum_l222_22224


namespace NUMINAMATH_CALUDE_valid_rectangles_count_l222_22258

/-- Represents a square array of dots -/
structure DotArray where
  size : ℕ

/-- Represents a rectangle in the dot array -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Returns true if the rectangle has an area greater than 1 -/
def Rectangle.areaGreaterThanOne (r : Rectangle) : Prop :=
  r.width * r.height > 1

/-- Returns the number of valid rectangles in the dot array -/
def countValidRectangles (arr : DotArray) : ℕ :=
  sorry

theorem valid_rectangles_count (arr : DotArray) :
  arr.size = 5 → countValidRectangles arr = 84 := by
  sorry

end NUMINAMATH_CALUDE_valid_rectangles_count_l222_22258


namespace NUMINAMATH_CALUDE_average_book_price_l222_22287

theorem average_book_price (books1 books2 : ℕ) (price1 price2 : ℚ) :
  books1 = 65 →
  books2 = 55 →
  price1 = 1280 →
  price2 = 880 →
  (price1 + price2) / (books1 + books2 : ℚ) = 18 := by
  sorry

end NUMINAMATH_CALUDE_average_book_price_l222_22287


namespace NUMINAMATH_CALUDE_expand_expression_l222_22208

theorem expand_expression (x y : ℝ) : (3*x - 15) * (4*y + 20) = 12*x*y + 60*x - 60*y - 300 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l222_22208


namespace NUMINAMATH_CALUDE_lawnmower_initial_price_l222_22240

/-- Proves that the initial price of a lawnmower was $100 given specific depreciation rates and final value -/
theorem lawnmower_initial_price (initial_price : ℝ) : 
  let price_after_six_months := initial_price * 0.75
  let final_price := price_after_six_months * 0.8
  final_price = 60 →
  initial_price = 100 := by
  sorry

end NUMINAMATH_CALUDE_lawnmower_initial_price_l222_22240


namespace NUMINAMATH_CALUDE_pascal_triangle_interior_sum_l222_22222

/-- Sum of interior numbers in the n-th row of Pascal's Triangle -/
def interior_sum (n : ℕ) : ℕ := 2^(n-1) - 2

/-- The problem statement -/
theorem pascal_triangle_interior_sum :
  interior_sum 6 = 30 →
  interior_sum 8 = 126 := by
sorry

end NUMINAMATH_CALUDE_pascal_triangle_interior_sum_l222_22222


namespace NUMINAMATH_CALUDE_sum_of_i_powers_l222_22215

/-- The imaginary unit i -/
noncomputable def i : ℂ := Complex.I

/-- Theorem stating that the sum of specific powers of i equals i -/
theorem sum_of_i_powers : i^13 + i^18 + i^23 + i^28 + i^33 = i := by sorry

end NUMINAMATH_CALUDE_sum_of_i_powers_l222_22215


namespace NUMINAMATH_CALUDE_exists_valid_grid_l222_22283

-- Define a 5x5 grid of integers (0 or 1)
def Grid := Fin 5 → Fin 5 → Fin 2

-- Function to check if a number is divisible by 3
def divisible_by_three (n : ℕ) : Prop := ∃ k, n = 3 * k

-- Function to sum a 2x2 subgrid
def sum_subgrid (g : Grid) (i j : Fin 4) : ℕ :=
  (g i j).val + (g i (j + 1)).val + (g (i + 1) j).val + (g (i + 1) (j + 1)).val

-- Theorem statement
theorem exists_valid_grid : ∃ (g : Grid),
  (∀ i j : Fin 4, divisible_by_three (sum_subgrid g i j)) ∧
  (∃ i j : Fin 5, g i j = 0) ∧
  (∃ i j : Fin 5, g i j = 1) :=
sorry

end NUMINAMATH_CALUDE_exists_valid_grid_l222_22283


namespace NUMINAMATH_CALUDE_sufficiency_not_necessity_l222_22254

theorem sufficiency_not_necessity (x₁ x₂ : ℝ) : 
  (∀ x₁ x₂ : ℝ, x₁ > 1 ∧ x₂ > 1 → x₁ + x₂ > 2 ∧ x₁ * x₂ > 1) ∧ 
  (∃ x₁ x₂ : ℝ, x₁ + x₂ > 2 ∧ x₁ * x₂ > 1 ∧ ¬(x₁ > 1 ∧ x₂ > 1)) :=
by sorry

end NUMINAMATH_CALUDE_sufficiency_not_necessity_l222_22254


namespace NUMINAMATH_CALUDE_average_gas_mileage_l222_22248

def total_distance : ℝ := 300
def sedan_efficiency : ℝ := 25
def truck_efficiency : ℝ := 15

theorem average_gas_mileage : 
  let sedan_distance := total_distance / 2
  let truck_distance := total_distance / 2
  let sedan_fuel := sedan_distance / sedan_efficiency
  let truck_fuel := truck_distance / truck_efficiency
  let total_fuel := sedan_fuel + truck_fuel
  (total_distance / total_fuel) = 18.75 := by sorry

end NUMINAMATH_CALUDE_average_gas_mileage_l222_22248


namespace NUMINAMATH_CALUDE_range_of_a_l222_22213

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x

def g (x : ℝ) : ℝ := -x^2 + 2*x + 1

theorem range_of_a (a : ℝ) :
  (∀ x₁ ∈ Set.Icc 1 (Real.exp 1), ∃ x₂ ∈ Set.Icc 0 3, f a x₁ = g x₂) →
  a ∈ Set.Icc (-1 / Real.exp 1) (3 / Real.exp 1) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l222_22213


namespace NUMINAMATH_CALUDE_class_average_theorem_l222_22263

theorem class_average_theorem (total_students : ℕ) (students_without_two : ℕ) 
  (avg_without_two : ℚ) (score1 : ℕ) (score2 : ℕ) :
  total_students = students_without_two + 2 →
  (students_without_two : ℚ) * avg_without_two + score1 + score2 = total_students * 80 :=
by
  sorry

#check class_average_theorem 40 38 79 98 100

end NUMINAMATH_CALUDE_class_average_theorem_l222_22263


namespace NUMINAMATH_CALUDE_profit_maximized_at_optimal_production_l222_22221

/-- Sales revenue as a function of production volume -/
def sales_revenue (x : ℝ) : ℝ := 17 * x^2

/-- Total production cost as a function of production volume -/
def total_cost (x : ℝ) : ℝ := 2 * x^3 - x^2

/-- Profit as a function of production volume -/
def profit (x : ℝ) : ℝ := sales_revenue x - total_cost x

/-- The production volume that maximizes profit -/
def optimal_production : ℝ := 6

theorem profit_maximized_at_optimal_production :
  ∀ x > 0, profit x ≤ profit optimal_production :=
by sorry

end NUMINAMATH_CALUDE_profit_maximized_at_optimal_production_l222_22221

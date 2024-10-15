import Mathlib

namespace NUMINAMATH_CALUDE_store_pricing_strategy_l3152_315207

/-- Calculates the sale price given the cost price and profit percentage -/
def calculateSalePrice (costPrice : ℚ) (profitPercentage : ℚ) : ℚ :=
  costPrice * (1 + profitPercentage / 100)

/-- Represents the store's pricing strategy -/
theorem store_pricing_strategy :
  let costA : ℚ := 320
  let costB : ℚ := 480
  let costC : ℚ := 600
  let profitA : ℚ := 50
  let profitB : ℚ := 70
  let profitC : ℚ := 40
  (calculateSalePrice costA profitA = 480) ∧
  (calculateSalePrice costB profitB = 816) ∧
  (calculateSalePrice costC profitC = 840) := by
  sorry

end NUMINAMATH_CALUDE_store_pricing_strategy_l3152_315207


namespace NUMINAMATH_CALUDE_extra_fruits_l3152_315257

def red_apples_ordered : ℕ := 60
def green_apples_ordered : ℕ := 34
def bananas_ordered : ℕ := 25
def oranges_ordered : ℕ := 45

def red_apple_students : ℕ := 3
def green_apple_students : ℕ := 2
def banana_students : ℕ := 5
def orange_students : ℕ := 10

def red_apples_per_student : ℕ := 2
def green_apples_per_student : ℕ := 2
def bananas_per_student : ℕ := 2
def oranges_per_student : ℕ := 1

theorem extra_fruits :
  red_apples_ordered - red_apple_students * red_apples_per_student +
  green_apples_ordered - green_apple_students * green_apples_per_student +
  bananas_ordered - banana_students * bananas_per_student +
  oranges_ordered - orange_students * oranges_per_student = 134 := by
  sorry

end NUMINAMATH_CALUDE_extra_fruits_l3152_315257


namespace NUMINAMATH_CALUDE_tan_45_degrees_l3152_315256

theorem tan_45_degrees : Real.tan (π / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_45_degrees_l3152_315256


namespace NUMINAMATH_CALUDE_perpendicular_vectors_k_value_l3152_315238

/-- Given vectors a, b, and c in ℝ², prove that if k*a + 2*b is perpendicular to c,
    then k = -17/3 -/
theorem perpendicular_vectors_k_value (a b c : ℝ × ℝ) (k : ℝ) 
    (h1 : a = (3, 4))
    (h2 : b = (-1, 5))
    (h3 : c = (2, -3))
    (h4 : (k * a.1 + 2 * b.1, k * a.2 + 2 * b.2) • c = 0) :
  k = -17/3 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_k_value_l3152_315238


namespace NUMINAMATH_CALUDE_hydrangea_spend_1989_to_2021_l3152_315267

/-- The amount spent on hydrangeas from a start year to an end year -/
def hydrangeaSpend (startYear endYear : ℕ) (pricePerPlant : ℚ) : ℚ :=
  (endYear - startYear + 1 : ℕ) * pricePerPlant

/-- Theorem stating the total spend on hydrangeas from 1989 to 2021 -/
theorem hydrangea_spend_1989_to_2021 :
  hydrangeaSpend 1989 2021 20 = 640 := by
  sorry

end NUMINAMATH_CALUDE_hydrangea_spend_1989_to_2021_l3152_315267


namespace NUMINAMATH_CALUDE_smallest_fraction_l3152_315276

theorem smallest_fraction : 
  let a := 7 / 15
  let b := 5 / 11
  let c := 16 / 33
  let d := 49 / 101
  let e := 89 / 183
  b ≤ a ∧ b ≤ c ∧ b ≤ d ∧ b ≤ e :=
by sorry

end NUMINAMATH_CALUDE_smallest_fraction_l3152_315276


namespace NUMINAMATH_CALUDE_closest_root_is_point_four_l3152_315279

/-- Quadratic function f(x) = 3x^2 - 6x + c -/
def f (c : ℝ) (x : ℝ) : ℝ := 3 * x^2 - 6 * x + c

/-- The constant c in the quadratic function -/
def c : ℝ := 2.24  -- f(0) = 2.24, so c = 2.24

theorem closest_root_is_point_four :
  let options : List ℝ := [0.2, 0.4, 0.6, 0.8]
  ∃ (root : ℝ), f c root = 0 ∧
    ∀ (x : ℝ), x ∈ options → |x - root| ≥ |0.4 - root| :=
by sorry

end NUMINAMATH_CALUDE_closest_root_is_point_four_l3152_315279


namespace NUMINAMATH_CALUDE_angle_trisection_l3152_315291

theorem angle_trisection (α : ℝ) (h : α = 54) :
  ∃ β : ℝ, β * 3 = α ∧ β = 18 := by
  sorry

end NUMINAMATH_CALUDE_angle_trisection_l3152_315291


namespace NUMINAMATH_CALUDE_fraction_equality_l3152_315225

theorem fraction_equality : ∃! (n m : ℕ) (d : ℚ), n > 0 ∧ m > 0 ∧ (n : ℚ) / m = d ∧ d = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3152_315225


namespace NUMINAMATH_CALUDE_tan_difference_l3152_315268

theorem tan_difference (α β : Real) 
  (h1 : Real.tan (α + π/3) = -3)
  (h2 : Real.tan (β - π/6) = 5) : 
  Real.tan (α - β) = -7/4 := by
sorry

end NUMINAMATH_CALUDE_tan_difference_l3152_315268


namespace NUMINAMATH_CALUDE_product_ratio_theorem_l3152_315282

theorem product_ratio_theorem (a b c d e f : ℝ) 
  (h1 : a * b * c = 195)
  (h2 : b * c * d = 65)
  (h3 : c * d * e = 1000)
  (h4 : d * e * f = 250)
  : (a * f) / (c * d) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_product_ratio_theorem_l3152_315282


namespace NUMINAMATH_CALUDE_negation_equivalence_l3152_315230

theorem negation_equivalence : 
  (¬(∀ x : ℝ, (x = 0 ∨ x = 1) → x^2 - x = 0)) ↔ 
  (∀ x : ℝ, (x ≠ 0 ∧ x ≠ 1) → x^2 - x ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3152_315230


namespace NUMINAMATH_CALUDE_square_side_length_l3152_315219

/-- Given a rectangle formed by three squares and two other rectangles, 
    prove that the middle square has a side length of 651 -/
theorem square_side_length (s₁ s₂ s₃ : ℕ) : 
  s₁ + s₂ + s₃ = 3322 →
  s₁ - s₂ + s₃ = 2020 →
  s₂ = 651 := by
sorry

end NUMINAMATH_CALUDE_square_side_length_l3152_315219


namespace NUMINAMATH_CALUDE_max_trailing_zeros_l3152_315210

def trailing_zeros (n : ℕ) : ℕ := sorry

def expr_a : ℕ := 2^5 * 3^4 * 5^6
def expr_b : ℕ := 2^4 * 3^4 * 5^5
def expr_c : ℕ := 4^3 * 5^6 * 6^5
def expr_d : ℕ := 4^2 * 5^4 * 6^3

theorem max_trailing_zeros :
  trailing_zeros expr_c > trailing_zeros expr_a ∧
  trailing_zeros expr_c > trailing_zeros expr_b ∧
  trailing_zeros expr_c > trailing_zeros expr_d :=
sorry

end NUMINAMATH_CALUDE_max_trailing_zeros_l3152_315210


namespace NUMINAMATH_CALUDE_tangent_square_area_l3152_315270

/-- Given a 6 by 6 square with semicircles on its sides, prove the area of the tangent square ABCD -/
theorem tangent_square_area :
  -- Original square side length
  let original_side : ℝ := 6
  -- Radius of semicircles (half of original side)
  let semicircle_radius : ℝ := original_side / 2
  -- Side length of square ABCD (original side + 2 * radius)
  let abcd_side : ℝ := original_side + 2 * semicircle_radius
  -- Area of square ABCD
  let abcd_area : ℝ := abcd_side ^ 2
  -- The area of square ABCD is 144
  abcd_area = 144 := by sorry

end NUMINAMATH_CALUDE_tangent_square_area_l3152_315270


namespace NUMINAMATH_CALUDE_periodic_sequence_characterization_l3152_315217

def is_periodic_sequence (x : ℕ → ℝ) : Prop :=
  ∃ T : ℕ, T > 0 ∧ ∀ n, x (n + T) = x n

theorem periodic_sequence_characterization
  (x : ℕ → ℝ)
  (h_pos : ∀ n, x n > 0)
  (h_periodic : is_periodic_sequence x)
  (h_recurrence : ∀ n, x (n + 2) = (1 / 2) * (1 / x (n + 1) + x n)) :
  ∃ a : ℝ, a > 0 ∧ ∀ n, x n = if n % 2 = 0 then a else 1 / a :=
sorry

end NUMINAMATH_CALUDE_periodic_sequence_characterization_l3152_315217


namespace NUMINAMATH_CALUDE_kids_joined_soccer_l3152_315214

theorem kids_joined_soccer (initial_kids final_kids : ℕ) (h1 : initial_kids = 14) (h2 : final_kids = 36) :
  final_kids - initial_kids = 22 := by
  sorry

end NUMINAMATH_CALUDE_kids_joined_soccer_l3152_315214


namespace NUMINAMATH_CALUDE_rectangle_area_with_squares_l3152_315209

/-- The area of a rectangle containing three non-overlapping squares -/
theorem rectangle_area_with_squares (s : ℝ) (h : s > 0) : 
  let small_square_area := s^2
  let large_square_area := (3*s)^2
  let total_area := 2 * small_square_area + large_square_area
  total_area = 11 * s^2 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_with_squares_l3152_315209


namespace NUMINAMATH_CALUDE_quadratic_real_roots_discriminant_nonnegative_l3152_315211

theorem quadratic_real_roots_discriminant_nonnegative
  (a b c : ℝ) (ha : a ≠ 0)
  (h_real_roots : ∃ x : ℝ, a * x^2 + b * x + c = 0) :
  b^2 - 4*a*c ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_discriminant_nonnegative_l3152_315211


namespace NUMINAMATH_CALUDE_polygonal_chain_existence_l3152_315280

-- Define a type for points in a plane
def Point := ℝ × ℝ

-- Define a type for lines in a plane
def Line := Point → Point → Prop

-- Define a type for a polygonal chain
def PolygonalChain (n : ℕ) := Fin (n + 1) → Point

-- Define the property of n lines in a plane
def LinesInPlane (n : ℕ) (lines : Fin n → Line) : Prop :=
  -- No two lines are parallel
  ∀ i j, i ≠ j → ¬ (∀ p q, lines i p q ↔ lines j p q) ∧
  -- No three lines intersect at a single point
  ∀ i j k, i ≠ j → j ≠ k → i ≠ k → 
    ¬ ∃ p, (lines i p p ∧ lines j p p ∧ lines k p p)

-- Define the property of a non-self-intersecting polygonal chain
def NonSelfIntersecting (chain : PolygonalChain n) : Prop :=
  ∀ i j k l, i < j → j < k → k < l → 
    ¬ (∃ p, (chain i = p ∧ chain j = p) ∨ (chain k = p ∧ chain l = p))

-- Define the property that each line contains exactly one segment of the chain
def EachLineOneSegment (n : ℕ) (lines : Fin n → Line) (chain : PolygonalChain n) : Prop :=
  ∀ i, ∃! j, lines i (chain j) (chain (j + 1))

-- The main theorem
theorem polygonal_chain_existence (n : ℕ) (lines : Fin n → Line) 
  (h : LinesInPlane n lines) :
  ∃ chain : PolygonalChain n, NonSelfIntersecting chain ∧ EachLineOneSegment n lines chain :=
sorry

end NUMINAMATH_CALUDE_polygonal_chain_existence_l3152_315280


namespace NUMINAMATH_CALUDE_dice_throw_outcomes_l3152_315224

/-- The number of possible outcomes for a single dice throw -/
def single_throw_outcomes : ℕ := 6

/-- The number of times the dice is thrown -/
def number_of_throws : ℕ := 2

/-- The total number of different outcomes when throwing a dice twice in succession -/
def total_outcomes : ℕ := single_throw_outcomes ^ number_of_throws

theorem dice_throw_outcomes : total_outcomes = 36 := by
  sorry

end NUMINAMATH_CALUDE_dice_throw_outcomes_l3152_315224


namespace NUMINAMATH_CALUDE_thickness_after_four_folds_l3152_315263

def blanket_thickness (initial_thickness : ℝ) (num_folds : ℕ) : ℝ :=
  initial_thickness * (2 ^ num_folds)

theorem thickness_after_four_folds :
  blanket_thickness 3 4 = 48 := by
  sorry

end NUMINAMATH_CALUDE_thickness_after_four_folds_l3152_315263


namespace NUMINAMATH_CALUDE_smithtown_handedness_ratio_l3152_315244

-- Define the population of Smithtown
structure Population where
  total : ℝ
  men : ℝ
  women : ℝ
  rightHanded : ℝ
  leftHanded : ℝ

-- Define the conditions
def smithtown_conditions (p : Population) : Prop :=
  p.men / p.women = 3 / 2 ∧
  p.men = p.rightHanded ∧
  p.leftHanded / p.total = 0.2500000000000001

-- Theorem statement
theorem smithtown_handedness_ratio (p : Population) :
  smithtown_conditions p →
  p.rightHanded / p.leftHanded = 3 / 1 :=
by sorry

end NUMINAMATH_CALUDE_smithtown_handedness_ratio_l3152_315244


namespace NUMINAMATH_CALUDE_relationship_abc_l3152_315281

theorem relationship_abc :
  let a := Real.tan (135 * π / 180)
  let b := Real.cos (Real.cos 0)
  let c := (fun x : ℝ => (x^2 + 1/2)^0) 0
  c > b ∧ b > a := by sorry

end NUMINAMATH_CALUDE_relationship_abc_l3152_315281


namespace NUMINAMATH_CALUDE_max_absolute_value_complex_l3152_315297

theorem max_absolute_value_complex (z : ℂ) (h : Complex.abs (z - 15) + Complex.abs (z - 8 * Complex.I) = 20) :
  Complex.abs z ≤ Real.sqrt 222 :=
sorry

end NUMINAMATH_CALUDE_max_absolute_value_complex_l3152_315297


namespace NUMINAMATH_CALUDE_expected_red_pairs_50_cards_l3152_315295

/-- Represents a deck of cards -/
structure Deck :=
  (total : ℕ)
  (red : ℕ)
  (black : ℕ)
  (h_total : total = red + black)
  (h_equal : red = black)

/-- The expected number of adjacent red pairs in a circular arrangement -/
def expected_red_pairs (d : Deck) : ℚ :=
  (d.red : ℚ) * ((d.red - 1) / (d.total - 1))

theorem expected_red_pairs_50_cards :
  ∃ d : Deck, d.total = 50 ∧ expected_red_pairs d = 600 / 49 := by
  sorry

end NUMINAMATH_CALUDE_expected_red_pairs_50_cards_l3152_315295


namespace NUMINAMATH_CALUDE_quadratic_roots_imaginary_l3152_315271

theorem quadratic_roots_imaginary (a b c a₁ b₁ c₁ : ℝ) : 
  let discriminant := 4 * ((a * a₁ + b * b₁ + c * c₁)^2 - (a^2 + b^2 + c^2) * (a₁^2 + b₁^2 + c₁^2))
  discriminant ≤ 0 ∧ 
  (discriminant = 0 ↔ ∃ (k : ℝ), k ≠ 0 ∧ a = k * a₁ ∧ b = k * b₁ ∧ c = k * c₁) := by
  sorry


end NUMINAMATH_CALUDE_quadratic_roots_imaginary_l3152_315271


namespace NUMINAMATH_CALUDE_necessary_condition_k_l3152_315285

theorem necessary_condition_k (k : ℝ) : 
  (∀ x : ℝ, -4 < x ∧ x < 1 → (x < k ∨ x > k + 2)) ∧
  (∃ x : ℝ, (x < k ∨ x > k + 2) ∧ ¬(-4 < x ∧ x < 1)) ↔
  k ≤ -6 ∨ k ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_necessary_condition_k_l3152_315285


namespace NUMINAMATH_CALUDE_sequence_is_arithmetic_l3152_315223

/-- Given a sequence {a_n} where the sum of its first n terms is S_n = 2n^2 - 3n,
    prove that {a_n} is an arithmetic sequence. -/
theorem sequence_is_arithmetic (a : ℕ → ℝ) (S : ℕ → ℝ) 
    (h : ∀ n : ℕ, S n = 2 * n^2 - 3 * n) :
    ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d :=
  sorry

end NUMINAMATH_CALUDE_sequence_is_arithmetic_l3152_315223


namespace NUMINAMATH_CALUDE_fraction_subtraction_l3152_315277

theorem fraction_subtraction (x : ℝ) (hx : x ≠ 0) : 1 / x - 2 / (3 * x) = 1 / (3 * x) := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l3152_315277


namespace NUMINAMATH_CALUDE_equal_max_attendance_l3152_315204

-- Define the days of the week
inductive Day
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday

-- Define the people
inductive Person
| Anna
| Bill
| Carl
| Dana

-- Define a function that returns whether a person can attend on a given day
def canAttend (p : Person) (d : Day) : Bool :=
  match p, d with
  | Person.Anna, Day.Monday => false
  | Person.Anna, Day.Wednesday => false
  | Person.Anna, Day.Friday => false
  | Person.Bill, Day.Tuesday => false
  | Person.Bill, Day.Thursday => false
  | Person.Carl, Day.Monday => false
  | Person.Carl, Day.Tuesday => false
  | Person.Carl, Day.Thursday => false
  | Person.Carl, Day.Friday => false
  | Person.Dana, Day.Wednesday => false
  | _, _ => true

-- Define a function that counts the number of people who can attend on a given day
def attendanceCount (d : Day) : Nat :=
  List.foldl (fun count p => count + if canAttend p d then 1 else 0) 0 [Person.Anna, Person.Bill, Person.Carl, Person.Dana]

-- Statement to prove
theorem equal_max_attendance :
  ∀ d1 d2 : Day, attendanceCount d1 = attendanceCount d2 ∧ attendanceCount d1 = 2 :=
sorry

end NUMINAMATH_CALUDE_equal_max_attendance_l3152_315204


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l3152_315262

theorem simplify_and_evaluate (x : ℝ) (h : x = -2) :
  1 / (x - 1) - 2 / (x^2 - 1) = -1 := by sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l3152_315262


namespace NUMINAMATH_CALUDE_jadens_estimate_l3152_315227

theorem jadens_estimate (p q δ γ : ℝ) 
  (h1 : p > q) 
  (h2 : q > 0) 
  (h3 : δ > γ) 
  (h4 : γ > 0) : 
  (p + δ) - (q - γ) > p - q := by
  sorry

end NUMINAMATH_CALUDE_jadens_estimate_l3152_315227


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l3152_315248

theorem algebraic_expression_value (a b : ℝ) : 
  (2 * a * (-1)^3 - 3 * b * (-1) + 8 = 18) → 
  (9 * b - 6 * a + 2 = 32) := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l3152_315248


namespace NUMINAMATH_CALUDE_shortest_distance_dasha_vasya_l3152_315250

-- Define the friends as vertices in a graph
inductive Friend : Type
| Asya : Friend
| Galia : Friend
| Borya : Friend
| Dasha : Friend
| Vasya : Friend

-- Define the distance function between friends
def distance : Friend → Friend → ℕ
| Friend.Asya, Friend.Galia => 12
| Friend.Galia, Friend.Asya => 12
| Friend.Galia, Friend.Borya => 10
| Friend.Borya, Friend.Galia => 10
| Friend.Asya, Friend.Borya => 8
| Friend.Borya, Friend.Asya => 8
| Friend.Dasha, Friend.Galia => 15
| Friend.Galia, Friend.Dasha => 15
| Friend.Vasya, Friend.Galia => 17
| Friend.Galia, Friend.Vasya => 17
| _, _ => 0  -- Default case for undefined distances

-- Define the shortest path function
def shortest_path (a b : Friend) : ℕ := sorry

-- Theorem statement
theorem shortest_distance_dasha_vasya :
  shortest_path Friend.Dasha Friend.Vasya = 18 := by sorry

end NUMINAMATH_CALUDE_shortest_distance_dasha_vasya_l3152_315250


namespace NUMINAMATH_CALUDE_museum_visitors_l3152_315266

theorem museum_visitors (V : ℕ) : 
  (∃ E : ℕ, 
    (E + 150 = V) ∧ 
    (E = (3 * V) / 4)) → 
  V = 600 := by
sorry

end NUMINAMATH_CALUDE_museum_visitors_l3152_315266


namespace NUMINAMATH_CALUDE_grandpa_mingming_age_ratio_l3152_315255

theorem grandpa_mingming_age_ratio :
  let grandpa_age : ℕ := 65
  let mingming_age : ℕ := 5
  let next_year_ratio : ℕ := (grandpa_age + 1) / (mingming_age + 1)
  next_year_ratio = 11 := by
  sorry

end NUMINAMATH_CALUDE_grandpa_mingming_age_ratio_l3152_315255


namespace NUMINAMATH_CALUDE_smallest_angle_tangent_equation_l3152_315283

theorem smallest_angle_tangent_equation (x : Real) : 
  (x > 0) →
  (Real.tan (6 * x * Real.pi / 180) = 
    (Real.cos (2 * x * Real.pi / 180) - Real.sin (2 * x * Real.pi / 180)) / 
    (Real.cos (2 * x * Real.pi / 180) + Real.sin (2 * x * Real.pi / 180))) →
  x = 5.625 := by
sorry

end NUMINAMATH_CALUDE_smallest_angle_tangent_equation_l3152_315283


namespace NUMINAMATH_CALUDE_sum_of_fourth_powers_l3152_315232

theorem sum_of_fourth_powers (x y : ℕ+) : x^4 + y^4 = 4721 → x + y = 13 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fourth_powers_l3152_315232


namespace NUMINAMATH_CALUDE_abe_age_sum_l3152_315202

theorem abe_age_sum : 
  let present_age : ℕ := 29
  let years_ago : ℕ := 7
  let past_age : ℕ := present_age - years_ago
  present_age + past_age = 51
  := by sorry

end NUMINAMATH_CALUDE_abe_age_sum_l3152_315202


namespace NUMINAMATH_CALUDE_highest_number_on_paper_l3152_315275

theorem highest_number_on_paper (n : ℕ) : 
  (1 : ℚ) / n = 0.010416666666666666 → n = 96 := by
  sorry

end NUMINAMATH_CALUDE_highest_number_on_paper_l3152_315275


namespace NUMINAMATH_CALUDE_select_cubes_eq_31_l3152_315237

/-- The number of ways to select 10 cubes from a set of 7 red cubes, 3 blue cubes, and 9 green cubes -/
def select_cubes : ℕ :=
  let red_cubes := 7
  let blue_cubes := 3
  let green_cubes := 9
  let total_selected := 10
  (Finset.range (red_cubes + 1)).sum (λ r => 
    (Finset.range (blue_cubes + 1)).sum (λ b => 
      let g := total_selected - r - b
      if g ≥ 0 ∧ g ≤ green_cubes then 1 else 0
    )
  )

theorem select_cubes_eq_31 : select_cubes = 31 := by sorry

end NUMINAMATH_CALUDE_select_cubes_eq_31_l3152_315237


namespace NUMINAMATH_CALUDE_fourth_term_in_geometric_sequence_l3152_315203

/-- Given a geometric sequence of 6 terms where the first term is 5 and the sixth term is 20,
    prove that the fourth term is approximately 6.6. -/
theorem fourth_term_in_geometric_sequence (a : ℕ → ℝ) (h1 : a 1 = 5) (h6 : a 6 = 20)
  (h_geometric : ∀ n ∈ Finset.range 5, a (n + 2) / a (n + 1) = a (n + 1) / a n) :
  ∃ ε > 0, |a 4 - 6.6| < ε :=
sorry

end NUMINAMATH_CALUDE_fourth_term_in_geometric_sequence_l3152_315203


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3152_315284

-- Define the sets M and N
def M : Set ℝ := {x | Real.sqrt x < 4}
def N : Set ℝ := {x | 3 * x ≥ 1}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {x | 1/3 ≤ x ∧ x < 16} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3152_315284


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3152_315218

/-- The hyperbola equation -/
def hyperbola (a b x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := x^2 = 4 * Real.sqrt 6 * y

/-- The point A lies on the hyperbola -/
def A_on_hyperbola (a b c m n : ℝ) : Prop := hyperbola a b m n

/-- The point B is on the imaginary axis of the hyperbola -/
def B_on_imaginary_axis (b : ℝ) : Prop := b = Real.sqrt 6

/-- The vector relation between BA and AF -/
def vector_relation (c m n : ℝ) : Prop :=
  m - 0 = 2 * (c - m) ∧ n - Real.sqrt 6 = 2 * (0 - n)

/-- The main theorem -/
theorem hyperbola_equation (a b c m n : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : hyperbola a b m n)
  (h2 : parabola m n)
  (h3 : A_on_hyperbola a b c m n)
  (h4 : B_on_imaginary_axis b)
  (h5 : vector_relation c m n) :
  a = 2 ∧ b = Real.sqrt 6 := by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3152_315218


namespace NUMINAMATH_CALUDE_birdhouse_distance_l3152_315234

/-- Proves that the birdhouse distance is 1200 feet given the problem conditions --/
theorem birdhouse_distance (car_distance : ℝ) (car_speed_mph : ℝ) 
  (lawn_chair_distance_multiplier : ℝ) (lawn_chair_time_multiplier : ℝ)
  (birdhouse_distance_multiplier : ℝ) (birdhouse_speed_percentage : ℝ) :
  car_distance = 200 →
  car_speed_mph = 80 →
  lawn_chair_distance_multiplier = 2 →
  lawn_chair_time_multiplier = 1.5 →
  birdhouse_distance_multiplier = 3 →
  birdhouse_speed_percentage = 0.6 →
  (birdhouse_distance_multiplier * lawn_chair_distance_multiplier * car_distance) = 1200 := by
  sorry

#check birdhouse_distance

end NUMINAMATH_CALUDE_birdhouse_distance_l3152_315234


namespace NUMINAMATH_CALUDE_percentage_of_liars_l3152_315228

theorem percentage_of_liars (truth_speakers : ℝ) (both_speakers : ℝ) (truth_or_lie_prob : ℝ) :
  truth_speakers = 0.3 →
  both_speakers = 0.1 →
  truth_or_lie_prob = 0.4 →
  ∃ (lie_speakers : ℝ), lie_speakers = 0.2 ∧ 
    truth_or_lie_prob = truth_speakers + lie_speakers - both_speakers :=
by sorry

end NUMINAMATH_CALUDE_percentage_of_liars_l3152_315228


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l3152_315245

theorem simplify_trig_expression (x : ℝ) (h : 1 + Real.sin x + Real.cos x ≠ 0) :
  (1 + Real.sin x - Real.cos x) / (1 + Real.sin x + Real.cos x) = Real.tan (x / 2) := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l3152_315245


namespace NUMINAMATH_CALUDE_cupcakes_frosted_l3152_315272

def cagney_rate : ℚ := 1 / 24
def lacey_rate : ℚ := 1 / 30
def casey_rate : ℚ := 1 / 40
def working_time : ℕ := 6 * 60  -- 6 minutes in seconds

theorem cupcakes_frosted :
  (cagney_rate + lacey_rate + casey_rate) * working_time = 36 := by
  sorry

end NUMINAMATH_CALUDE_cupcakes_frosted_l3152_315272


namespace NUMINAMATH_CALUDE_complex_equality_l3152_315229

theorem complex_equality (a b : ℝ) (h : Complex.I * (a + Complex.I) = b - Complex.I) : a - b = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_equality_l3152_315229


namespace NUMINAMATH_CALUDE_complex_exponential_sum_l3152_315269

theorem complex_exponential_sum (α β γ : ℝ) :
  Complex.exp (Complex.I * α) + Complex.exp (Complex.I * β) + Complex.exp (Complex.I * γ) = (2/5 : ℂ) + (1/3 : ℂ) * Complex.I →
  Complex.exp (-Complex.I * α) + Complex.exp (-Complex.I * β) + Complex.exp (-Complex.I * γ) = (2/5 : ℂ) - (1/3 : ℂ) * Complex.I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_exponential_sum_l3152_315269


namespace NUMINAMATH_CALUDE_intersection_line_slope_l3152_315215

/-- Given two lines y = 4 - 3x and y = 2x - 1, and a third line y = ax + 7 that passes through 
    their intersection point, prove that a = -6. -/
theorem intersection_line_slope (a : ℝ) : 
  (∃ x y : ℝ, y = 4 - 3*x ∧ y = 2*x - 1 ∧ y = a*x + 7) → a = -6 := by
  sorry

end NUMINAMATH_CALUDE_intersection_line_slope_l3152_315215


namespace NUMINAMATH_CALUDE_tangent_line_b_value_l3152_315273

/-- The curve function -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x + 1

/-- The derivative of the curve function -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + a

theorem tangent_line_b_value :
  ∀ a k b : ℝ,
  f a 2 = 3 →                        -- The curve passes through (2, 3)
  f' a 2 = k →                       -- The slope of the tangent line at x = 2
  3 = k * 2 + b →                    -- The tangent line passes through (2, 3)
  b = -15 := by sorry

end NUMINAMATH_CALUDE_tangent_line_b_value_l3152_315273


namespace NUMINAMATH_CALUDE_smallest_two_digit_prime_reverse_composite_l3152_315260

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def reverse_digits (n : ℕ) : ℕ :=
  if n < 10 then n else
  let tens := n / 10
  let ones := n % 10
  ones * 10 + tens

def is_composite (n : ℕ) : Prop := n > 1 ∧ ¬(is_prime n)

theorem smallest_two_digit_prime_reverse_composite :
  ∃ p : ℕ, 
    p ≥ 10 ∧ p < 100 ∧  -- two-digit number
    is_prime p ∧
    is_composite (reverse_digits p) ∧
    p / 10 ≤ 3 ∧  -- starts with a digit less than or equal to 3
    (∀ q : ℕ, q ≥ 10 ∧ q < p ∧ is_prime q ∧ q / 10 ≤ 3 → ¬(is_composite (reverse_digits q))) ∧
    p = 23 :=
by sorry

end NUMINAMATH_CALUDE_smallest_two_digit_prime_reverse_composite_l3152_315260


namespace NUMINAMATH_CALUDE_inequality_proof_l3152_315287

theorem inequality_proof (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  a + 1 / b > b + 1 / a := by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3152_315287


namespace NUMINAMATH_CALUDE_expression_factorization_l3152_315290

theorem expression_factorization (y : ℝ) : 
  5 * y * (y - 2) + 10 * (y - 2) - 15 * (y - 2) = 5 * (y - 2) * (y - 1) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l3152_315290


namespace NUMINAMATH_CALUDE_all_shaded_areas_different_l3152_315298

/-- Represents a square with its division and shaded area -/
structure Square where
  total_divisions : ℕ
  shaded_divisions : ℕ

/-- The three squares in the problem -/
def square_I : Square := { total_divisions := 8, shaded_divisions := 3 }
def square_II : Square := { total_divisions := 9, shaded_divisions := 3 }
def square_III : Square := { total_divisions := 8, shaded_divisions := 4 }

/-- Calculate the shaded fraction of a square -/
def shaded_fraction (s : Square) : ℚ :=
  (s.shaded_divisions : ℚ) / (s.total_divisions : ℚ)

/-- Theorem stating that the shaded areas of all three squares are different -/
theorem all_shaded_areas_different :
  shaded_fraction square_I ≠ shaded_fraction square_II ∧
  shaded_fraction square_I ≠ shaded_fraction square_III ∧
  shaded_fraction square_II ≠ shaded_fraction square_III :=
sorry

end NUMINAMATH_CALUDE_all_shaded_areas_different_l3152_315298


namespace NUMINAMATH_CALUDE_shift_left_one_unit_l3152_315265

/-- Represents a parabola of the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally -/
def shift_horizontal (p : Parabola) (h : ℝ) : Parabola :=
  { a := p.a
  , b := -2 * p.a * h + p.b
  , c := p.a * h^2 - p.b * h + p.c }

theorem shift_left_one_unit (p : Parabola) :
  p.a = 2 ∧ p.b = 0 ∧ p.c = -1 →
  let p_shifted := shift_horizontal p 1
  p_shifted.a = 2 ∧ p_shifted.b = 4 ∧ p_shifted.c = 1 :=
by sorry

end NUMINAMATH_CALUDE_shift_left_one_unit_l3152_315265


namespace NUMINAMATH_CALUDE_fred_age_difference_l3152_315222

theorem fred_age_difference (jim fred sam : ℕ) : 
  jim = 2 * fred →
  jim = 46 →
  jim - 6 = 5 * (sam - 6) →
  fred - sam = 9 := by sorry

end NUMINAMATH_CALUDE_fred_age_difference_l3152_315222


namespace NUMINAMATH_CALUDE_square_of_difference_l3152_315246

theorem square_of_difference (x : ℝ) : (8 - Real.sqrt (x^2 + 64))^2 = x^2 + 128 - 16 * Real.sqrt (x^2 + 64) := by
  sorry

end NUMINAMATH_CALUDE_square_of_difference_l3152_315246


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l3152_315286

theorem pure_imaginary_complex_number (b : ℝ) : 
  let z : ℂ := (1 + b * Complex.I) * (2 + Complex.I)
  (∃ (y : ℝ), z = y * Complex.I ∧ y ≠ 0) → b = 2 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l3152_315286


namespace NUMINAMATH_CALUDE_max_three_match_winners_200_l3152_315289

/-- Represents a single-elimination tournament --/
structure Tournament :=
  (participants : ℕ)

/-- Calculates the total number of matches in a single-elimination tournament --/
def total_matches (t : Tournament) : ℕ :=
  t.participants - 1

/-- Calculates the maximum number of participants who can win at least 3 matches --/
def max_participants_with_three_wins (t : Tournament) : ℕ :=
  (total_matches t) / 3

/-- Theorem stating the maximum number of participants who can win at least 3 matches
    in a tournament with 200 participants --/
theorem max_three_match_winners_200 :
  ∃ (t : Tournament), t.participants = 200 ∧ max_participants_with_three_wins t = 66 :=
by
  sorry


end NUMINAMATH_CALUDE_max_three_match_winners_200_l3152_315289


namespace NUMINAMATH_CALUDE_descending_order_l3152_315296

-- Define the numbers in their respective bases
def a : ℕ := 3 * 16 + 14
def b : ℕ := 2 * 6^2 + 1 * 6 + 0
def c : ℕ := 1 * 4^3 + 0 * 4^2 + 0 * 4 + 0
def d : ℕ := 1 * 2^5 + 1 * 2^4 + 1 * 2^3 + 0 * 2^2 + 1 * 2 + 1

-- Theorem statement
theorem descending_order : b > c ∧ c > a ∧ a > d := by
  sorry

end NUMINAMATH_CALUDE_descending_order_l3152_315296


namespace NUMINAMATH_CALUDE_route_down_is_24_miles_l3152_315233

/-- A hiking trip up and down a mountain -/
structure HikingTrip where
  rate_up : ℝ
  time_up : ℝ
  rate_down_factor : ℝ

/-- The length of the route down the mountain -/
def route_down_length (trip : HikingTrip) : ℝ :=
  trip.rate_up * trip.rate_down_factor * trip.time_up

/-- Theorem: The length of the route down the mountain is 24 miles -/
theorem route_down_is_24_miles (trip : HikingTrip)
  (h1 : trip.rate_up = 8)
  (h2 : trip.time_up = 2)
  (h3 : trip.rate_down_factor = 1.5) :
  route_down_length trip = 24 := by
  sorry

end NUMINAMATH_CALUDE_route_down_is_24_miles_l3152_315233


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3152_315201

theorem right_triangle_hypotenuse (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (1 / 3 * π * b * a^2 = 1280 * π) → (b / a = 3 / 4) → 
  Real.sqrt (a^2 + b^2) = 20 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3152_315201


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l3152_315235

-- First expression
theorem simplify_expression_1 (x y : ℝ) (h1 : x ≠ y) (h2 : x ≠ 0) :
  (4 * x^2) / (x^2 - y^2) / (x / (x + y)) = 4 * x / (x - y) := by sorry

-- Second expression
theorem simplify_expression_2 (m : ℝ) (h : m ≠ 1) :
  m / (m - 1) - 1 = 1 / (m - 1) := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l3152_315235


namespace NUMINAMATH_CALUDE_gordon_jamie_persian_ratio_l3152_315220

/-- Represents the number of cats of each type owned by each person -/
structure CatOwnership where
  jamie_persian : ℕ
  jamie_maine_coon : ℕ
  gordon_persian : ℕ
  gordon_maine_coon : ℕ
  hawkeye_persian : ℕ
  hawkeye_maine_coon : ℕ

/-- The theorem stating the ratio of Gordon's Persian cats to Jamie's Persian cats -/
theorem gordon_jamie_persian_ratio (cats : CatOwnership) : 
  cats.jamie_persian = 4 →
  cats.jamie_maine_coon = 2 →
  cats.gordon_maine_coon = cats.jamie_maine_coon + 1 →
  cats.hawkeye_persian = 0 →
  cats.hawkeye_maine_coon = cats.gordon_maine_coon - 1 →
  cats.jamie_persian + cats.jamie_maine_coon + 
  cats.gordon_persian + cats.gordon_maine_coon + 
  cats.hawkeye_persian + cats.hawkeye_maine_coon = 13 →
  2 * cats.gordon_persian = cats.jamie_persian := by
  sorry


end NUMINAMATH_CALUDE_gordon_jamie_persian_ratio_l3152_315220


namespace NUMINAMATH_CALUDE_total_earnings_l3152_315206

-- Define pizza types
inductive PizzaType
| Margherita
| Pepperoni
| VeggieSupreme
| MeatLovers
| Hawaiian

-- Define pizza prices
def slicePrice (p : PizzaType) : ℚ :=
  match p with
  | .Margherita => 3
  | .Pepperoni => 4
  | .VeggieSupreme => 5
  | .MeatLovers => 6
  | .Hawaiian => 4.5

def wholePizzaPrice (p : PizzaType) : ℚ :=
  match p with
  | .Margherita => 15
  | .Pepperoni => 18
  | .VeggieSupreme => 22
  | .MeatLovers => 25
  | .Hawaiian => 20

-- Define discount and promotion rules
def wholeDiscountRate : ℚ := 0.1
def wholeDiscountThreshold : ℕ := 3
def regularToppingPrice : ℚ := 2
def weekendToppingPrice : ℚ := 1
def happyHourPrice : ℚ := 3

-- Define sales data
structure SalesData where
  margheritaSlices : ℕ
  margheritaHappyHour : ℕ
  pepperoniSlices : ℕ
  pepperoniHappyHour : ℕ
  pepperoniToppings : ℕ
  veggieSupremeWhole : ℕ
  veggieSupremeToppings : ℕ
  margheritaWholePackage : ℕ
  meatLoversSlices : ℕ
  meatLoversHappyHour : ℕ
  hawaiianSlices : ℕ
  hawaiianToppings : ℕ
  pepperoniWholeWeekend : ℕ
  pepperoniWholeToppings : ℕ

def salesData : SalesData := {
  margheritaSlices := 24,
  margheritaHappyHour := 12,
  pepperoniSlices := 16,
  pepperoniHappyHour := 8,
  pepperoniToppings := 6,
  veggieSupremeWhole := 4,
  veggieSupremeToppings := 8,
  margheritaWholePackage := 3,
  meatLoversSlices := 20,
  meatLoversHappyHour := 10,
  hawaiianSlices := 12,
  hawaiianToppings := 4,
  pepperoniWholeWeekend := 1,
  pepperoniWholeToppings := 3
}

-- Theorem statement
theorem total_earnings (data : SalesData) :
  let earnings := 
    (data.margheritaSlices - data.margheritaHappyHour) * slicePrice PizzaType.Margherita +
    data.margheritaHappyHour * happyHourPrice +
    (data.pepperoniSlices - data.pepperoniHappyHour) * slicePrice PizzaType.Pepperoni +
    data.pepperoniHappyHour * happyHourPrice +
    data.pepperoniToppings * weekendToppingPrice +
    data.veggieSupremeWhole * wholePizzaPrice PizzaType.VeggieSupreme +
    data.veggieSupremeToppings * weekendToppingPrice +
    (data.margheritaWholePackage * wholePizzaPrice PizzaType.Margherita) * (1 - wholeDiscountRate) +
    (data.meatLoversSlices - data.meatLoversHappyHour) * slicePrice PizzaType.MeatLovers +
    data.meatLoversHappyHour * happyHourPrice +
    data.hawaiianSlices * slicePrice PizzaType.Hawaiian +
    data.hawaiianToppings * weekendToppingPrice +
    data.pepperoniWholeWeekend * wholePizzaPrice PizzaType.Pepperoni +
    data.pepperoniWholeToppings * weekendToppingPrice
  earnings = 439.5 := by sorry


end NUMINAMATH_CALUDE_total_earnings_l3152_315206


namespace NUMINAMATH_CALUDE_negation_of_implication_l3152_315231

theorem negation_of_implication :
  (¬(x = 3 → x^2 - 2*x - 3 = 0)) ↔ (x = 3 ∧ x^2 - 2*x - 3 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l3152_315231


namespace NUMINAMATH_CALUDE_lower_limit_of_g_l3152_315274

-- Define the function f(n)
def f (n : ℕ) : ℕ := Finset.prod (Finset.range (n^2 - 3)) (λ i => i + 4)

-- Define the function g(n) with a parameter m for the lower limit
def g (n m : ℕ) : ℕ := Finset.prod (Finset.range (n - m + 1)) (λ i => (i + m)^2)

-- State the theorem
theorem lower_limit_of_g : ∃ m : ℕ, 
  m = 2 ∧ 
  (∀ n : ℕ, n ≥ m → g n m ≠ 0) ∧
  (∃ k : ℕ, (f 3 / g 3 m).factorization 2 = 4) :=
sorry

end NUMINAMATH_CALUDE_lower_limit_of_g_l3152_315274


namespace NUMINAMATH_CALUDE_sum_four_consecutive_composite_sum_three_consecutive_composite_l3152_315221

-- Define the sum of four consecutive positive integers
def sum_four_consecutive (n : ℕ) : ℕ := n + (n + 1) + (n + 2) + (n + 3)

-- Define the sum of three consecutive positive integers
def sum_three_consecutive (n : ℕ) : ℕ := n + (n + 1) + (n + 2)

-- Theorem for four consecutive positive integers
theorem sum_four_consecutive_composite (n : ℕ) : 
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ sum_four_consecutive n = a * b :=
sorry

-- Theorem for three consecutive positive integers
theorem sum_three_consecutive_composite (n : ℕ) : 
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ sum_three_consecutive n = a * b :=
sorry

end NUMINAMATH_CALUDE_sum_four_consecutive_composite_sum_three_consecutive_composite_l3152_315221


namespace NUMINAMATH_CALUDE_cyclic_quadrilateral_perpendicular_diagonals_l3152_315294

/-- A convex quadrilateral with side lengths a, b, c, d in sequence, inscribed in a circle of radius R -/
structure CyclicQuadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  R : ℝ
  convex : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0
  cyclic : a^2 + b^2 + c^2 + d^2 = 8 * R^2

/-- The diagonals of a quadrilateral are perpendicular -/
def has_perpendicular_diagonals (q : CyclicQuadrilateral) : Prop :=
  ∃ (A B C D : ℝ × ℝ), 
    (A.1 - C.1) * (B.1 - D.1) + (A.2 - C.2) * (B.2 - D.2) = 0

/-- 
If a convex quadrilateral ABCD with side lengths a, b, c, d in sequence, 
inscribed in a circle with radius R, satisfies a^2 + b^2 + c^2 + d^2 = 8R^2, 
then the diagonals of the quadrilateral are perpendicular.
-/
theorem cyclic_quadrilateral_perpendicular_diagonals (q : CyclicQuadrilateral) :
  has_perpendicular_diagonals q :=
sorry

end NUMINAMATH_CALUDE_cyclic_quadrilateral_perpendicular_diagonals_l3152_315294


namespace NUMINAMATH_CALUDE_extremum_at_one_implies_a_eq_neg_two_l3152_315213

/-- Given a cubic function f(x) = x^3 + ax^2 + x + b with an extremum at x = 1, 
    prove that a = -2. -/
theorem extremum_at_one_implies_a_eq_neg_two (a b : ℝ) :
  let f : ℝ → ℝ := λ x => x^3 + a*x^2 + x + b
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f x ≤ f 1 ∨ ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f x ≥ f 1) →
  a = -2 := by
sorry

end NUMINAMATH_CALUDE_extremum_at_one_implies_a_eq_neg_two_l3152_315213


namespace NUMINAMATH_CALUDE_greatest_ABDBA_div_by_11_l3152_315212

/-- Represents a five-digit number in the form AB,DBA -/
structure ABDBA where
  a : Nat
  b : Nat
  d : Nat
  h1 : a < 10
  h2 : b < 10
  h3 : d < 10
  h4 : a ≠ b
  h5 : a ≠ d
  h6 : b ≠ d

/-- Converts ABDBA to its numerical value -/
def ABDBA.toNat (n : ABDBA) : Nat :=
  n.a * 10000 + n.b * 1000 + n.d * 100 + n.b * 10 + n.a

/-- Theorem stating the greatest ABDBA number divisible by 11 -/
theorem greatest_ABDBA_div_by_11 :
  ∀ n : ABDBA, n.toNat ≤ 96569 ∧ n.toNat % 11 = 0 →
  ∃ m : ABDBA, m.toNat = 96569 ∧ m.toNat % 11 = 0 :=
sorry

end NUMINAMATH_CALUDE_greatest_ABDBA_div_by_11_l3152_315212


namespace NUMINAMATH_CALUDE_yellow_bows_count_l3152_315288

theorem yellow_bows_count (total : ℕ) 
  (h_red : (total : ℚ) / 4 = total / 4)
  (h_blue : (total : ℚ) / 3 = total / 3)
  (h_green : (total : ℚ) / 6 = total / 6)
  (h_yellow : (total : ℚ) / 12 = total / 12)
  (h_white : (total : ℚ) - (total / 4 + total / 3 + total / 6 + total / 12) = 40) :
  (total : ℚ) / 12 = 20 := by
  sorry

end NUMINAMATH_CALUDE_yellow_bows_count_l3152_315288


namespace NUMINAMATH_CALUDE_product_of_binomials_l3152_315293

theorem product_of_binomials (m : ℝ) : (-m + 2) * (-m - 2) = m^2 - 4 := by
  sorry

end NUMINAMATH_CALUDE_product_of_binomials_l3152_315293


namespace NUMINAMATH_CALUDE_lucy_calculation_mistake_l3152_315278

theorem lucy_calculation_mistake (a b c : ℝ) 
  (h1 : a / (b * c) = 4)
  (h2 : (a / b) / c = 12)
  (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  a / b = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_lucy_calculation_mistake_l3152_315278


namespace NUMINAMATH_CALUDE_line_perp_plane_criterion_l3152_315252

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between lines and planes
variable (perp_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between lines
variable (perp_line_line : Line → Line → Prop)

-- State the theorem
theorem line_perp_plane_criterion 
  (α β γ : Plane) (m n l : Line) :
  perp_line_plane n α → 
  perp_line_plane n β → 
  perp_line_plane m α → 
  perp_line_plane m β :=
sorry

end NUMINAMATH_CALUDE_line_perp_plane_criterion_l3152_315252


namespace NUMINAMATH_CALUDE_baseball_gear_cost_l3152_315239

def initial_amount : ℕ := 67
def amount_left : ℕ := 33

theorem baseball_gear_cost :
  initial_amount - amount_left = 34 :=
by sorry

end NUMINAMATH_CALUDE_baseball_gear_cost_l3152_315239


namespace NUMINAMATH_CALUDE_card_partition_theorem_l3152_315258

/-- Represents a card with a number written on it -/
structure Card where
  number : Nat

/-- Represents a stack of cards -/
def Stack := List Card

/-- The sum of numbers on a stack of cards -/
def stackSum (s : Stack) : Nat :=
  s.map (λ c => c.number) |>.sum

theorem card_partition_theorem (n k : Nat) (cards : List Card) :
  (∀ c ∈ cards, c.number ≤ n) →
  (cards.map (λ c => c.number)).sum = k * n.factorial →
  ∃ (partition : List Stack),
    partition.length = k ∧
    partition.all (λ s => stackSum s = n.factorial) ∧
    partition.join = cards :=
  sorry

end NUMINAMATH_CALUDE_card_partition_theorem_l3152_315258


namespace NUMINAMATH_CALUDE_geometric_sequence_tangent_l3152_315253

/-- Given a geometric sequence {a_n} where a_2 * a_3 * a_4 = -a_7^2 = -64,
    prove that tan((a_4 * a_6 / 3) * π) = -√3 -/
theorem geometric_sequence_tangent (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 2) / a (n + 1) = a (n + 1) / a n) →  -- geometric sequence condition
  a 2 * a 3 * a 4 = -a 7^2 →                            -- given condition
  a 7^2 = 64 →                                          -- given condition
  Real.tan ((a 4 * a 6 / 3) * Real.pi) = -Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_tangent_l3152_315253


namespace NUMINAMATH_CALUDE_min_value_a_plus_2b_l3152_315226

theorem min_value_a_plus_2b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : (a + b) / (a * b) = 1) :
  ∀ x y, x > 0 → y > 0 → (x + y) / (x * y) = 1 → a + 2 * b ≤ x + 2 * y :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_plus_2b_l3152_315226


namespace NUMINAMATH_CALUDE_angles_with_same_terminal_side_l3152_315299

theorem angles_with_same_terminal_side (θ : Real) :
  θ = 150 * Real.pi / 180 →
  {β : Real | ∃ k : ℤ, β = 5 * Real.pi / 6 + 2 * k * Real.pi} =
  {β : Real | ∃ k : ℤ, β = θ + 2 * k * Real.pi} :=
by sorry

end NUMINAMATH_CALUDE_angles_with_same_terminal_side_l3152_315299


namespace NUMINAMATH_CALUDE_triangle_problem_l3152_315200

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) : 
  0 < A ∧ A < π / 2 →  -- Acute angle A
  0 < B ∧ B < π / 2 →  -- Acute angle B
  0 < C ∧ C < π / 2 →  -- Acute angle C
  A + B + C = π →      -- Sum of angles in a triangle
  b + c = 10 →         -- Given condition
  a = Real.sqrt 10 →   -- Given condition
  5 * b * Real.sin A * Real.cos C + 5 * c * Real.sin A * Real.cos B = 3 * Real.sqrt 10 → -- Given condition
  Real.cos A = 4 / 5 ∧ b = 5 ∧ c = 5 := by
sorry


end NUMINAMATH_CALUDE_triangle_problem_l3152_315200


namespace NUMINAMATH_CALUDE_train_distance_theorem_l3152_315249

/-- Calculates the distance a train can travel given its coal efficiency and remaining coal. -/
def train_distance (miles_per_unit : ℚ) (pounds_per_unit : ℚ) (coal_remaining : ℚ) : ℚ :=
  (coal_remaining / pounds_per_unit) * miles_per_unit

/-- Proves that a train with given efficiency and coal amount can travel the calculated distance. -/
theorem train_distance_theorem (miles_per_unit : ℚ) (pounds_per_unit : ℚ) (coal_remaining : ℚ) :
  miles_per_unit = 5 → pounds_per_unit = 2 → coal_remaining = 160 →
  train_distance miles_per_unit pounds_per_unit coal_remaining = 400 := by
  sorry

#check train_distance_theorem

end NUMINAMATH_CALUDE_train_distance_theorem_l3152_315249


namespace NUMINAMATH_CALUDE_triangle_is_right_angled_l3152_315236

/-- A triangle is right-angled if the square of its longest side equals the sum of squares of the other two sides. -/
def IsRightAngled (a b c : ℝ) : Prop :=
  (a ≥ b ∧ a ≥ c ∧ a^2 = b^2 + c^2) ∨
  (b ≥ a ∧ b ≥ c ∧ b^2 = a^2 + c^2) ∨
  (c ≥ a ∧ c ≥ b ∧ c^2 = a^2 + b^2)

/-- Given three real numbers a, b, and c that satisfy the equation
    a^2 + b^2 + c^2 - 12a - 16b - 20c + 200 = 0,
    prove that they form a right-angled triangle. -/
theorem triangle_is_right_angled (a b c : ℝ)
  (h : a^2 + b^2 + c^2 - 12*a - 16*b - 20*c + 200 = 0) :
  IsRightAngled a b c :=
sorry

end NUMINAMATH_CALUDE_triangle_is_right_angled_l3152_315236


namespace NUMINAMATH_CALUDE_trigonometric_identity_l3152_315241

theorem trigonometric_identity : 
  Real.sin (135 * π / 180) * Real.cos (-15 * π / 180) + 
  Real.cos (225 * π / 180) * Real.sin (15 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l3152_315241


namespace NUMINAMATH_CALUDE_x_range_l3152_315208

theorem x_range (x : ℝ) : (Real.sqrt ((5 - x)^2) = x - 5) → x ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_x_range_l3152_315208


namespace NUMINAMATH_CALUDE_intersection_of_sets_l3152_315264

theorem intersection_of_sets : 
  let M : Set ℕ := {1, 2, 3, 4}
  let N : Set ℕ := {0, 1, 2, 3}
  M ∩ N = {1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l3152_315264


namespace NUMINAMATH_CALUDE_p_range_q_range_p_or_q_false_range_l3152_315251

-- Define proposition p
def p (m : ℝ) : Prop := ∃ x₀ : ℝ, x₀^2 + 2*m*x₀ + 2 + m = 0

-- Define proposition q
def q (m : ℝ) : Prop := ∃ x y : ℝ, x^2/(1-2*m) + y^2/(m+2) = 1 ∧ (1-2*m)*(m+2) < 0

-- Theorem for the range of m when p is true
theorem p_range (m : ℝ) : p m ↔ m ≤ -2 ∨ m ≥ 1 :=
sorry

-- Theorem for the range of m when q is true
theorem q_range (m : ℝ) : q m ↔ m < -2 ∨ m > 1/2 :=
sorry

-- Theorem for the range of m when "p ∨ q" is false
theorem p_or_q_false_range (m : ℝ) : ¬(p m ∨ q m) ↔ -2 < m ∧ m ≤ 1/2 :=
sorry

end NUMINAMATH_CALUDE_p_range_q_range_p_or_q_false_range_l3152_315251


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3152_315216

/-- An arithmetic sequence with positive common difference -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  h_positive : d > 0
  h_arithmetic : ∀ n, a (n + 1) = a n + d

/-- The theorem statement -/
theorem arithmetic_sequence_problem (seq : ArithmeticSequence) 
  (h1 : seq.a 1 + seq.a 2 + seq.a 3 = 15)
  (h2 : seq.a 1 * seq.a 2 * seq.a 3 = 80) :
  seq.a 11 + seq.a 12 + seq.a 13 = 105 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3152_315216


namespace NUMINAMATH_CALUDE_floor_sqrt_50_squared_l3152_315242

theorem floor_sqrt_50_squared : ⌊Real.sqrt 50⌋^2 = 49 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_50_squared_l3152_315242


namespace NUMINAMATH_CALUDE_db_length_determined_l3152_315243

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the altitude CD to AB
def altitudeCD (t : Triangle) (D : ℝ × ℝ) : Prop :=
  let (xA, yA) := t.A
  let (xB, yB) := t.B
  let (xC, yC) := t.C
  let (xD, yD) := D
  (xD - xA) * (xB - xA) + (yD - yA) * (yB - yA) = 0 ∧
  (xC - xD) * (xB - xA) + (yC - yD) * (yB - yA) = 0

-- Define the altitude AE to BC
def altitudeAE (t : Triangle) (E : ℝ × ℝ) : Prop :=
  let (xA, yA) := t.A
  let (xB, yB) := t.B
  let (xC, yC) := t.C
  let (xE, yE) := E
  (xE - xB) * (xC - xB) + (yE - yB) * (yC - yB) = 0 ∧
  (xA - xE) * (xC - xB) + (yA - yE) * (yC - yB) = 0

-- Define the lengths of AB, CD, and AE
def lengthAB (t : Triangle) : ℝ := sorry
def lengthCD (t : Triangle) (D : ℝ × ℝ) : ℝ := sorry
def lengthAE (t : Triangle) (E : ℝ × ℝ) : ℝ := sorry

-- Define the length of DB
def lengthDB (t : Triangle) (D : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem db_length_determined (t : Triangle) (D E : ℝ × ℝ) :
  altitudeCD t D → altitudeAE t E →
  ∃! db : ℝ, db = lengthDB t D := by sorry

end NUMINAMATH_CALUDE_db_length_determined_l3152_315243


namespace NUMINAMATH_CALUDE_impossible_11_difference_l3152_315259

/-- Represents an L-shaped piece -/
structure LPiece where
  cells : ℕ
  odd_cells : Odd cells

/-- Represents a partition of a square into L-shaped pieces -/
structure Partition where
  pieces : List LPiece
  total_cells : (pieces.map LPiece.cells).sum = 120 * 120

theorem impossible_11_difference (p1 p2 : Partition) : 
  p2.pieces.length ≠ p1.pieces.length + 11 := by
  sorry

end NUMINAMATH_CALUDE_impossible_11_difference_l3152_315259


namespace NUMINAMATH_CALUDE_minimal_adjective_f_25_l3152_315240

/-- A function g: ℤ → ℤ is adjective if g(m) + g(n) > max(m², n²) for any integers m and n -/
def Adjective (g : ℤ → ℤ) : Prop :=
  ∀ m n : ℤ, g m + g n > max (m ^ 2) (n ^ 2)

/-- The sum of f(1) to f(30) -/
def SumF (f : ℤ → ℤ) : ℤ :=
  (Finset.range 30).sum (fun i => f (i + 1))

/-- f is an adjective function that minimizes SumF -/
def IsMinimalAdjective (f : ℤ → ℤ) : Prop :=
  Adjective f ∧ ∀ g : ℤ → ℤ, Adjective g → SumF f ≤ SumF g

theorem minimal_adjective_f_25 (f : ℤ → ℤ) (hf : IsMinimalAdjective f) : f 25 ≥ 498 := by
  sorry

end NUMINAMATH_CALUDE_minimal_adjective_f_25_l3152_315240


namespace NUMINAMATH_CALUDE_darius_age_l3152_315205

theorem darius_age (jenna_age darius_age : ℕ) : 
  jenna_age = 13 →
  jenna_age = darius_age + 5 →
  jenna_age + darius_age = 21 →
  darius_age = 8 := by
sorry

end NUMINAMATH_CALUDE_darius_age_l3152_315205


namespace NUMINAMATH_CALUDE_triangle_construction_theorem_l3152_315292

/-- A line in 2D space -/
structure Line where
  -- Define a line using two points
  p1 : ℝ × ℝ
  p2 : ℝ × ℝ

/-- A triangle in 2D space -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Check if a point lies on a line -/
def point_on_line (p : ℝ × ℝ) (l : Line) : Prop := sorry

/-- Check if a line is a perpendicular bisector of a triangle side -/
def is_perp_bisector (l : Line) (t : Triangle) : Prop := sorry

/-- The main theorem -/
theorem triangle_construction_theorem 
  (e f g : Line) -- Three given lines (perpendicular bisectors)
  (P : ℝ × ℝ)    -- Given point
  (h : point_on_line P e ∨ point_on_line P f ∨ point_on_line P g) -- P is on one of the lines
  : ∃ (t : Triangle), 
    (point_on_line P e ∧ is_perp_bisector e t) ∨ 
    (point_on_line P f ∧ is_perp_bisector f t) ∨ 
    (point_on_line P g ∧ is_perp_bisector g t) :=
sorry

end NUMINAMATH_CALUDE_triangle_construction_theorem_l3152_315292


namespace NUMINAMATH_CALUDE_quadratic_function_property_l3152_315247

def repeating_digits (k : ℕ) (p : ℕ) : ℚ := (k : ℚ) / 9 * (10^p - 1)

def f_k (k : ℕ) (x : ℚ) : ℚ := 9 / (k : ℚ) * x^2 + 2 * x

theorem quadratic_function_property (k : ℕ) (p : ℕ) 
  (h1 : 1 ≤ k) (h2 : k ≤ 9) (h3 : 0 < p) :
  f_k k (repeating_digits k p) = repeating_digits k (2 * p) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l3152_315247


namespace NUMINAMATH_CALUDE_log_c_27_is_0_75_implies_c_is_81_l3152_315261

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_c_27_is_0_75_implies_c_is_81 :
  ∀ c : ℝ, c > 0 → log c 27 = 0.75 → c = 81 := by
  sorry

end NUMINAMATH_CALUDE_log_c_27_is_0_75_implies_c_is_81_l3152_315261


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_cubic_equation_l3152_315254

theorem negation_of_existence (f : ℝ → ℝ) :
  (¬ ∃ x, f x = 0) ↔ ∀ x, f x ≠ 0 := by sorry

theorem negation_of_cubic_equation :
  (¬ ∃ x : ℝ, x^3 - 2*x + 1 = 0) ↔ ∀ x : ℝ, x^3 - 2*x + 1 ≠ 0 := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_cubic_equation_l3152_315254

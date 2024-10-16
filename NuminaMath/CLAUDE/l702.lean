import Mathlib

namespace NUMINAMATH_CALUDE_tea_milk_mixture_l702_70277

/-- Represents a cup with a certain amount of liquid -/
structure Cup where
  tea : ℚ
  milk : ℚ

/-- The problem setup and solution -/
theorem tea_milk_mixture : 
  let initial_cup1 : Cup := { tea := 6, milk := 0 }
  let initial_cup2 : Cup := { tea := 0, milk := 6 }
  let cup_size : ℚ := 12

  -- Transfer 1/3 of tea from Cup 1 to Cup 2
  let transfer1_amount : ℚ := initial_cup1.tea / 3
  let after_transfer1_cup1 : Cup := { tea := initial_cup1.tea - transfer1_amount, milk := initial_cup1.milk }
  let after_transfer1_cup2 : Cup := { tea := initial_cup2.tea + transfer1_amount, milk := initial_cup2.milk }

  -- Transfer 1/4 of mixture from Cup 2 back to Cup 1
  let total_liquid_cup2 : ℚ := after_transfer1_cup2.tea + after_transfer1_cup2.milk
  let transfer2_amount : ℚ := total_liquid_cup2 / 4
  let tea_ratio_cup2 : ℚ := after_transfer1_cup2.tea / total_liquid_cup2
  let milk_ratio_cup2 : ℚ := after_transfer1_cup2.milk / total_liquid_cup2
  let final_cup1 : Cup := {
    tea := after_transfer1_cup1.tea + transfer2_amount * tea_ratio_cup2,
    milk := after_transfer1_cup1.milk + transfer2_amount * milk_ratio_cup2
  }

  -- The fraction of milk in Cup 1 at the end
  let milk_fraction : ℚ := final_cup1.milk / (final_cup1.tea + final_cup1.milk)

  milk_fraction = 1/4 := by sorry

end NUMINAMATH_CALUDE_tea_milk_mixture_l702_70277


namespace NUMINAMATH_CALUDE_student_weight_is_90_l702_70259

/-- The student's weight in kilograms -/
def student_weight : ℝ := sorry

/-- The sister's weight in kilograms -/
def sister_weight : ℝ := sorry

/-- The combined weight of the student and his sister in kilograms -/
def combined_weight : ℝ := 132

/-- If the student loses 6 kilograms, he will weigh twice as much as his sister -/
axiom weight_relation : student_weight - 6 = 2 * sister_weight

/-- The combined weight of the student and his sister is 132 kilograms -/
axiom total_weight : student_weight + sister_weight = combined_weight

/-- Theorem: The student's present weight is 90 kilograms -/
theorem student_weight_is_90 : student_weight = 90 := by sorry

end NUMINAMATH_CALUDE_student_weight_is_90_l702_70259


namespace NUMINAMATH_CALUDE_even_function_decreasing_nonpositive_inequality_l702_70252

/-- A function f is even if f(x) = f(-x) for all x -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- A function f is decreasing on (-∞, 0] if f(x₂) < f(x₁) for x₁ < x₂ ≤ 0 -/
def DecreasingOnNonPositive (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ → x₂ ≤ 0 → f x₂ < f x₁

theorem even_function_decreasing_nonpositive_inequality
  (f : ℝ → ℝ)
  (heven : EvenFunction f)
  (hdecr : DecreasingOnNonPositive f) :
  f 1 < f (-2) ∧ f (-2) < f (-3) :=
sorry

end NUMINAMATH_CALUDE_even_function_decreasing_nonpositive_inequality_l702_70252


namespace NUMINAMATH_CALUDE_fraction_simplification_l702_70294

theorem fraction_simplification (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hsum : a + b + c ≠ 0) :
  (a^2 + a*b - b^2 + a*c) / (b^2 + b*c - c^2 + b*a) = (a - b) / (b - c) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l702_70294


namespace NUMINAMATH_CALUDE_f_2048_equals_121_l702_70282

/-- A function satisfying the given property for positive integers -/
def special_function (f : ℕ → ℝ) : Prop :=
  ∀ (a b n : ℕ), a > 0 → b > 0 → n > 0 → a * b = 2^n → f a + f b = n^2

/-- The main theorem to prove -/
theorem f_2048_equals_121 (f : ℕ → ℝ) (h : special_function f) : f 2048 = 121 := by
  sorry

end NUMINAMATH_CALUDE_f_2048_equals_121_l702_70282


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_equation_l702_70227

theorem purely_imaginary_complex_equation (a : ℝ) (z : ℂ) :
  z + 3 * Complex.I = a + a * Complex.I →
  z.re = 0 →
  a = 0 := by
sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_equation_l702_70227


namespace NUMINAMATH_CALUDE_expression_equivalence_l702_70274

theorem expression_equivalence (x y : ℝ) (h : x * y ≠ 0) :
  ((x^3 + 2) / x) * ((y^3 + 2) / y) + ((x^3 - 2) / y) * ((y^3 - 2) / x) = 2 * x^2 * y^2 + 8 / (x * y) := by
  sorry

end NUMINAMATH_CALUDE_expression_equivalence_l702_70274


namespace NUMINAMATH_CALUDE_not_all_monotonic_functions_have_extremum_l702_70208

-- Define a monotonic function
def MonotonicFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y

-- Define the existence of an extremum value
def HasExtremum (f : ℝ → ℝ) : Prop :=
  ∃ x, ∀ y, f y ≤ f x ∨ f x ≤ f y

-- Theorem statement
theorem not_all_monotonic_functions_have_extremum :
  ∃ f : ℝ → ℝ, MonotonicFunction f ∧ ¬HasExtremum f := by
  sorry

end NUMINAMATH_CALUDE_not_all_monotonic_functions_have_extremum_l702_70208


namespace NUMINAMATH_CALUDE_social_gathering_attendees_l702_70219

theorem social_gathering_attendees (men : ℕ) (women : ℕ) : 
  men = 12 →
  men * 4 = women * 3 →
  women = 16 := by
sorry

end NUMINAMATH_CALUDE_social_gathering_attendees_l702_70219


namespace NUMINAMATH_CALUDE_julias_mean_score_l702_70271

def scores : List ℝ := [88, 90, 92, 94, 95, 97, 98, 99]

def henry_mean : ℝ := 94

theorem julias_mean_score (h1 : scores.length = 8)
                          (h2 : ∃ henry_scores julia_scores : List ℝ,
                                henry_scores.length = 4 ∧
                                julia_scores.length = 4 ∧
                                henry_scores ++ julia_scores = scores)
                          (h3 : ∃ henry_scores : List ℝ,
                                henry_scores.length = 4 ∧
                                henry_scores.sum / 4 = henry_mean) :
  ∃ julia_scores : List ℝ,
    julia_scores.length = 4 ∧
    julia_scores.sum / 4 = 94.25 :=
sorry

end NUMINAMATH_CALUDE_julias_mean_score_l702_70271


namespace NUMINAMATH_CALUDE_tv_production_reduction_l702_70263

/-- Given a factory that produces televisions, calculate the percentage reduction in production from the first year to the second year. -/
theorem tv_production_reduction (daily_rate : ℕ) (second_year_total : ℕ) : 
  daily_rate = 10 →
  second_year_total = 3285 →
  (1 - (second_year_total : ℝ) / (daily_rate * 365 : ℝ)) * 100 = 10 := by
  sorry

#check tv_production_reduction

end NUMINAMATH_CALUDE_tv_production_reduction_l702_70263


namespace NUMINAMATH_CALUDE_wise_men_strategy_l702_70209

/-- Represents the color of a hat -/
inductive HatColor
| White
| Black

/-- Represents a wise man with a hat -/
structure WiseMan where
  hat : HatColor

/-- Represents the line of wise men -/
def WiseMenLine := List WiseMan

/-- A strategy is a function that takes the visible hats and returns a guess -/
def Strategy := (visible : WiseMenLine) → HatColor

/-- Counts the number of correct guesses given a line of wise men and a strategy -/
def countCorrectGuesses (line : WiseMenLine) (strategy : Strategy) : Nat :=
  sorry

/-- The main theorem: there exists a strategy where at least n-1 wise men guess correctly -/
theorem wise_men_strategy (n : Nat) :
  ∃ (strategy : Strategy), ∀ (line : WiseMenLine),
    line.length = n →
    countCorrectGuesses line strategy ≥ n - 1 :=
  sorry

end NUMINAMATH_CALUDE_wise_men_strategy_l702_70209


namespace NUMINAMATH_CALUDE_cube_sum_inequality_equality_iff_condition_l702_70288

/-- For any pairwise distinct natural numbers a, b, and c, 
    (a³ + b³ + c³) / 3 ≥ abc + a + b + c holds. -/
theorem cube_sum_inequality (a b c : ℕ) (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) :
  (a^3 + b^3 + c^3) / 3 ≥ a * b * c + a + b + c :=
sorry

/-- Characterization of when equality holds in the cube sum inequality. -/
def equality_condition (a b c : ℕ) : Prop :=
  (a = b + 1 ∧ b = c + 1) ∨ 
  (b = a + 1 ∧ a = c + 1) ∨ 
  (c = a + 1 ∧ a = b + 1)

/-- The equality condition is necessary and sufficient for the cube sum inequality to be an equality. -/
theorem equality_iff_condition (a b c : ℕ) (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) :
  (a^3 + b^3 + c^3) / 3 = a * b * c + a + b + c ↔ equality_condition a b c :=
sorry

end NUMINAMATH_CALUDE_cube_sum_inequality_equality_iff_condition_l702_70288


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l702_70213

theorem arithmetic_calculations :
  ((-15) + 4 + (-6) - (-11) = -6) ∧
  (-1^2024 + (-3)^2 * |(-1/18)| - 1 / (-2) = 0) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l702_70213


namespace NUMINAMATH_CALUDE_earrings_ratio_is_two_to_one_l702_70207

/-- The number of gumballs Kim gets for each pair of earrings -/
def gumballs_per_pair : ℕ := 9

/-- The number of pairs of earrings Kim brings on the first day -/
def first_day_pairs : ℕ := 3

/-- The number of gumballs Kim eats per day -/
def gumballs_eaten_per_day : ℕ := 3

/-- The number of days the gumballs should last -/
def total_days : ℕ := 42

/-- The number of pairs of earrings Kim brings on the second day -/
def second_day_pairs : ℕ := 6

theorem earrings_ratio_is_two_to_one :
  let total_gumballs := gumballs_per_pair * (first_day_pairs + second_day_pairs + (second_day_pairs - 1))
  total_gumballs = gumballs_eaten_per_day * total_days ∧
  second_day_pairs / first_day_pairs = 2 := by
  sorry

end NUMINAMATH_CALUDE_earrings_ratio_is_two_to_one_l702_70207


namespace NUMINAMATH_CALUDE_circle_tangent_ratio_l702_70256

-- Define the types for points and circles
variable (Point Circle : Type)

-- Define the basic geometric relations
variable (on_circle : Point → Circle → Prop)
variable (inside_circle : Circle → Circle → Prop)
variable (concentric : Circle → Circle → Prop)
variable (tangent_to : Point → Point → Circle → Prop)
variable (intersects : Point → Point → Circle → Point → Prop)
variable (midpoint : Point → Point → Point → Prop)
variable (line_through : Point → Point → Point → Prop)
variable (perp_bisector : Point → Point → Point → Point → Prop)
variable (ratio : Point → Point → Point → ℚ → Prop)

-- State the theorem
theorem circle_tangent_ratio 
  (Γ₁ Γ₂ : Circle) 
  (A B C D E F M : Point) :
  concentric Γ₁ Γ₂ →
  inside_circle Γ₂ Γ₁ →
  on_circle A Γ₁ →
  on_circle B Γ₂ →
  tangent_to A B Γ₂ →
  intersects A B Γ₁ C →
  midpoint D A B →
  line_through A E F →
  on_circle E Γ₂ →
  on_circle F Γ₂ →
  perp_bisector D E M B →
  perp_bisector C F M B →
  ratio A M C (3/2) :=
by sorry

end NUMINAMATH_CALUDE_circle_tangent_ratio_l702_70256


namespace NUMINAMATH_CALUDE_basketball_team_size_l702_70236

theorem basketball_team_size (total_points : ℕ) (min_score : ℕ) (max_score : ℕ) :
  total_points = 100 →
  min_score = 7 →
  max_score = 23 →
  ∃ (team_size : ℕ) (scores : List ℕ),
    team_size = 12 ∧
    scores.length = team_size ∧
    scores.sum = total_points ∧
    (∀ s ∈ scores, min_score ≤ s ∧ s ≤ max_score) :=
by sorry

end NUMINAMATH_CALUDE_basketball_team_size_l702_70236


namespace NUMINAMATH_CALUDE_age_ratio_proof_l702_70285

theorem age_ratio_proof (a b c : ℕ) : 
  a = b + 2 →
  a + b + c = 12 →
  b = 4 →
  b / c = 2 := by
sorry

end NUMINAMATH_CALUDE_age_ratio_proof_l702_70285


namespace NUMINAMATH_CALUDE_cubic_roots_sum_cubes_l702_70257

theorem cubic_roots_sum_cubes (a b c : ℝ) : 
  (2 * a^3 - 3 * a^2 + 165 * a - 4 = 0) →
  (2 * b^3 - 3 * b^2 + 165 * b - 4 = 0) →
  (2 * c^3 - 3 * c^2 + 165 * c - 4 = 0) →
  (a + b - 1)^3 + (b + c - 1)^3 + (c + a - 1)^3 = 117 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_cubes_l702_70257


namespace NUMINAMATH_CALUDE_line_inclination_l702_70228

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x - Real.sqrt 3 * y + 1 = 0

-- Define the angle of inclination
def angle_of_inclination (θ : ℝ) : Prop := Real.tan θ = 1 / Real.sqrt 3

-- Theorem statement
theorem line_inclination :
  ∃ θ, angle_of_inclination θ ∧ θ = 30 * π / 180 :=
sorry

end NUMINAMATH_CALUDE_line_inclination_l702_70228


namespace NUMINAMATH_CALUDE_permutations_of_six_objects_l702_70267

theorem permutations_of_six_objects : Nat.factorial 6 = 720 := by
  sorry

end NUMINAMATH_CALUDE_permutations_of_six_objects_l702_70267


namespace NUMINAMATH_CALUDE_f_composition_equals_three_l702_70224

noncomputable def f (x : ℂ) : ℂ :=
  if x.im = 0 then 1 + x else (1 - Complex.I) / Complex.abs Complex.I * x

theorem f_composition_equals_three :
  f (f (1 + Complex.I)) = 3 := by sorry

end NUMINAMATH_CALUDE_f_composition_equals_three_l702_70224


namespace NUMINAMATH_CALUDE_ice_cream_volume_l702_70242

/-- The volume of ice cream in a cone and hemisphere -/
theorem ice_cream_volume (h : ℝ) (r : ℝ) (h_pos : h > 0) (r_pos : r > 0) :
  let cone_volume := (1 / 3) * π * r^2 * h
  let hemisphere_volume := (2 / 3) * π * r^3
  h = 10 ∧ r = 3 →
  cone_volume + hemisphere_volume = 48 * π := by
sorry

end NUMINAMATH_CALUDE_ice_cream_volume_l702_70242


namespace NUMINAMATH_CALUDE_expression_evaluation_l702_70290

theorem expression_evaluation : 3^(1^(0^8)) + ((3^1)^0)^8 = 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l702_70290


namespace NUMINAMATH_CALUDE_system_two_solutions_l702_70223

/-- The system of inequalities has exactly two solutions if and only if a = 7 -/
theorem system_two_solutions (a : ℝ) : 
  (∃! x y z : ℝ, x ≠ y ∧ 
    (abs z + abs (z - x) ≤ a - abs (x - 1)) ∧
    ((z - 4) * (z + 3) ≥ (4 - x) * (3 + x)) ∧
    (abs z + abs (z - y) ≤ a - abs (y - 1)) ∧
    ((z - 4) * (z + 3) ≥ (4 - y) * (3 + y)))
  ↔ a = 7 := by sorry

end NUMINAMATH_CALUDE_system_two_solutions_l702_70223


namespace NUMINAMATH_CALUDE_right_triangle_sides_l702_70229

theorem right_triangle_sides : ∀ (a b c : ℝ), 
  (a > 0 ∧ b > 0 ∧ c > 0) →
  (a = 1 ∧ b = 2 ∧ c = Real.sqrt 3) ↔ a * a + b * b = c * c :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_sides_l702_70229


namespace NUMINAMATH_CALUDE_circle_intersection_theorem_l702_70233

-- Define the points A, B, and C
def A : ℝ × ℝ := (1, 3)
def B : ℝ × ℝ := (4, 2)
def C : ℝ × ℝ := (1, -7)

-- Define circle M passing through A, B, and C
def circle_M : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 2)^2 = 25}

-- Define the y-axis
def y_axis : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = 0}

-- Define the line on which the center of circle N moves
def center_line : Set (ℝ × ℝ) := {p : ℝ × ℝ | 2 * p.1 - p.2 + 6 = 0}

-- Define circle N with radius 10 and center (a, 2a + 6)
def circle_N (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - a)^2 + (p.2 - (2 * a + 6))^2 = 100}

-- Define the theorem
theorem circle_intersection_theorem :
  ∃ (P Q : ℝ × ℝ) (a : Set ℝ),
    P ∈ circle_M ∧ P ∈ y_axis ∧
    Q ∈ circle_M ∧ Q ∈ y_axis ∧
    (Q.2 - P.2)^2 = 96 ∧
    (∀ x ∈ a, ∃ y, (x, y) ∈ center_line ∧ (circle_N x ∩ circle_M).Nonempty) ∧
    a = {x : ℝ | -3 - Real.sqrt 41 ≤ x ∧ x ≤ -4 ∨ -2 ≤ x ∧ x ≤ -3 + Real.sqrt 41} :=
sorry

end NUMINAMATH_CALUDE_circle_intersection_theorem_l702_70233


namespace NUMINAMATH_CALUDE_cube_structure_extension_l702_70264

/-- Represents a cube structure with a central cube and attached cubes -/
structure CubeStructure :=
  (central : ℕ)
  (attached : ℕ)

/-- The number of cubes in the initial structure -/
def initial_cubes (s : CubeStructure) : ℕ := s.central + s.attached

/-- The number of exposed faces in the initial structure -/
def exposed_faces (s : CubeStructure) : ℕ := s.attached * 5

/-- The number of extra cubes needed for the extended structure -/
def extra_cubes_needed (s : CubeStructure) : ℕ := 12 + 6

theorem cube_structure_extension (s : CubeStructure) 
  (h1 : s.central = 1) 
  (h2 : s.attached = 6) : 
  extra_cubes_needed s = 18 := by sorry

end NUMINAMATH_CALUDE_cube_structure_extension_l702_70264


namespace NUMINAMATH_CALUDE_inscribed_sphere_pyramid_volume_l702_70254

/-- A pyramid with an inscribed sphere -/
structure InscribedSpherePyramid where
  /-- The volume of the pyramid -/
  volume : ℝ
  /-- The radius of the inscribed sphere -/
  radius : ℝ
  /-- The total surface area of the pyramid -/
  surface_area : ℝ
  /-- The radius is positive -/
  radius_pos : radius > 0
  /-- The surface area is positive -/
  surface_area_pos : surface_area > 0

/-- 
Theorem: The volume of a pyramid with an inscribed sphere is equal to 
one-third of the product of the radius of the sphere and the total surface area of the pyramid.
-/
theorem inscribed_sphere_pyramid_volume 
  (p : InscribedSpherePyramid) : p.volume = (1 / 3) * p.surface_area * p.radius := by
  sorry

end NUMINAMATH_CALUDE_inscribed_sphere_pyramid_volume_l702_70254


namespace NUMINAMATH_CALUDE_min_attacking_pairs_8x8_16rooks_l702_70278

/-- Represents a chessboard configuration -/
structure ChessBoard where
  size : Nat
  rooks : Nat

/-- Calculates the minimum number of attacking rook pairs -/
def minAttackingPairs (board : ChessBoard) : Nat :=
  sorry

/-- Theorem stating the minimum number of attacking rook pairs for a specific configuration -/
theorem min_attacking_pairs_8x8_16rooks :
  let board : ChessBoard := { size := 8, rooks := 16 }
  minAttackingPairs board = 16 := by
  sorry

end NUMINAMATH_CALUDE_min_attacking_pairs_8x8_16rooks_l702_70278


namespace NUMINAMATH_CALUDE_nearest_integer_to_three_plus_sqrt_three_fourth_l702_70215

theorem nearest_integer_to_three_plus_sqrt_three_fourth (x : ℝ) : 
  x = (3 + Real.sqrt 3)^4 → 
  ∃ n : ℤ, n = 504 ∧ ∀ m : ℤ, |x - n| ≤ |x - m| := by
  sorry

end NUMINAMATH_CALUDE_nearest_integer_to_three_plus_sqrt_three_fourth_l702_70215


namespace NUMINAMATH_CALUDE_third_to_second_package_ratio_is_half_l702_70226

/-- Represents the delivery driver's work for a day -/
structure DeliveryDay where
  miles_first_package : ℕ
  miles_second_package : ℕ
  total_pay : ℕ
  pay_per_mile : ℕ

/-- Calculates the ratio of the distance for the third package to the second package -/
def third_to_second_package_ratio (day : DeliveryDay) : ℚ :=
  let total_miles := day.total_pay / day.pay_per_mile
  let miles_third_package := total_miles - day.miles_first_package - day.miles_second_package
  miles_third_package / day.miles_second_package

/-- Theorem stating the ratio of the third package distance to the second package distance -/
theorem third_to_second_package_ratio_is_half (day : DeliveryDay) 
    (h1 : day.miles_first_package = 10)
    (h2 : day.miles_second_package = 28)
    (h3 : day.total_pay = 104)
    (h4 : day.pay_per_mile = 2) :
    third_to_second_package_ratio day = 1/2 := by
  sorry

#eval third_to_second_package_ratio { 
  miles_first_package := 10, 
  miles_second_package := 28, 
  total_pay := 104, 
  pay_per_mile := 2 
}

end NUMINAMATH_CALUDE_third_to_second_package_ratio_is_half_l702_70226


namespace NUMINAMATH_CALUDE_min_value_of_a_l702_70286

noncomputable def f (x : ℝ) : ℝ := x^3 - 3*x + 3 - x / (Real.exp x)

theorem min_value_of_a (a : ℝ) :
  (∃ x : ℝ, x ≥ -2 ∧ f x ≤ a) ↔ a ≥ 1 - 1 / Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_a_l702_70286


namespace NUMINAMATH_CALUDE_expression_equals_negative_one_l702_70283

theorem expression_equals_negative_one (a x : ℝ) (ha : a ≠ 0) (hx1 : x ≠ a) (hx2 : x ≠ -2*a) :
  (((a / (2*a + x)) - (x / (a - x))) / ((x / (2*a + x)) + (a / (a - x)))) = -1 ↔ x = a / 2 :=
by sorry

end NUMINAMATH_CALUDE_expression_equals_negative_one_l702_70283


namespace NUMINAMATH_CALUDE_basketball_activity_results_l702_70298

/-- Represents the outcome of a shot -/
inductive ShotResult
| Hit
| Miss

/-- Represents the game state -/
inductive GameState
| InProgress
| Cleared
| Failed

/-- Represents the possible coupon amounts -/
inductive CouponAmount
| Three
| Six
| Nine

/-- The shooting accuracy of Xiao Ming -/
def accuracy : ℚ := 2/3

/-- Updates the game state based on the current state and the new shot result -/
def updateGameState (state : GameState) (shot : ShotResult) : GameState :=
  sorry

/-- Simulates the game for a given number of shots -/
def simulateGame (n : ℕ) : GameState :=
  sorry

/-- Calculates the probability of ending the game after exactly 5 shots -/
def probEndAfterFiveShots : ℚ :=
  sorry

/-- Represents the distribution of the coupon amount -/
def couponDistribution : CouponAmount → ℚ :=
  sorry

/-- Calculates the expectation of the coupon amount -/
def expectedCouponAmount : ℚ :=
  sorry

theorem basketball_activity_results :
  probEndAfterFiveShots = 8/81 ∧
  couponDistribution CouponAmount.Three = 233/729 ∧
  couponDistribution CouponAmount.Six = 112/729 ∧
  couponDistribution CouponAmount.Nine = 128/243 ∧
  expectedCouponAmount = 1609/243 :=
sorry

end NUMINAMATH_CALUDE_basketball_activity_results_l702_70298


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l702_70218

-- Define the quadratic function f(x) = ax^2 + bx + c
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_function_properties 
  (a b c : ℝ) 
  (h1 : f a b c 1 = -a/2) 
  (h2 : 3*a > 2*c) 
  (h3 : 2*c > 2*b) :
  (a > 0 ∧ -3 < b/a ∧ b/a < -3/4) ∧ 
  (∃ x, 0 < x ∧ x < 2 ∧ f a b c x = 0) ∧
  (∀ x₁ x₂, f a b c x₁ = 0 → f a b c x₂ = 0 → 
    Real.sqrt 2 ≤ |x₁ - x₂| ∧ |x₁ - x₂| < Real.sqrt 57 / 4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l702_70218


namespace NUMINAMATH_CALUDE_boys_cannot_score_double_l702_70205

/-- Represents a player in the chess tournament -/
inductive Player
| Boy
| Girl

/-- Represents the outcome of a chess game -/
inductive GameResult
| Win
| Draw
| Loss

/-- The number of players in the tournament -/
def numPlayers : Nat := 6

/-- The number of boys in the tournament -/
def numBoys : Nat := 2

/-- The number of girls in the tournament -/
def numGirls : Nat := 4

/-- The number of games each player plays -/
def gamesPerPlayer : Nat := numPlayers - 1

/-- The total number of games in the tournament -/
def totalGames : Nat := (numPlayers * gamesPerPlayer) / 2

/-- The points awarded for each game result -/
def pointsForResult (result : GameResult) : Rat :=
  match result with
  | GameResult.Win => 1
  | GameResult.Draw => 1/2
  | GameResult.Loss => 0

/-- A function representing the total score of a group of players -/
def groupScore (players : List Player) (results : List (Player × Player × GameResult)) : Rat :=
  sorry

/-- The main theorem stating that boys cannot score twice as many points as girls -/
theorem boys_cannot_score_double :
  ¬∃ (results : List (Player × Player × GameResult)),
    (results.length = totalGames) ∧
    (groupScore [Player.Boy, Player.Boy] results = 2 * groupScore [Player.Girl, Player.Girl, Player.Girl, Player.Girl] results) :=
  sorry

end NUMINAMATH_CALUDE_boys_cannot_score_double_l702_70205


namespace NUMINAMATH_CALUDE_parabola_point_distance_l702_70276

theorem parabola_point_distance (a : ℝ) : 
  (∀ x y : ℝ, y^2 = 4*x → (x - a)^2 + y^2 ≥ a^2) → a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_distance_l702_70276


namespace NUMINAMATH_CALUDE_cricket_match_analysis_l702_70241

-- Define the cricket match parameters
def total_overs : ℕ := 50
def initial_overs : ℕ := 10
def remaining_overs : ℕ := total_overs - initial_overs
def initial_run_rate : ℚ := 32/10
def initial_wickets : ℕ := 2
def target_score : ℕ := 320
def min_additional_wickets : ℕ := 5

-- Define the theorem
theorem cricket_match_analysis :
  let initial_score := initial_run_rate * initial_overs
  let remaining_score := target_score - initial_score
  let required_run_rate := remaining_score / remaining_overs
  let total_wickets_needed := initial_wickets + min_additional_wickets
  (required_run_rate = 72/10) ∧ (total_wickets_needed = 7) := by
  sorry

end NUMINAMATH_CALUDE_cricket_match_analysis_l702_70241


namespace NUMINAMATH_CALUDE_equation_solution_l702_70216

theorem equation_solution (x : ℝ) (h : x > 4) :
  (Real.sqrt (x - 4 * Real.sqrt (x - 4)) + 2 = Real.sqrt (x + 4 * Real.sqrt (x - 4)) - 2) ↔ x ≥ 8 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l702_70216


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l702_70217

theorem quadratic_two_distinct_roots (a : ℝ) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
  x₁^2 + 2*a*x₁ + a^2 - 1 = 0 ∧ 
  x₂^2 + 2*a*x₂ + a^2 - 1 = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l702_70217


namespace NUMINAMATH_CALUDE_power_of_two_equation_solution_l702_70280

theorem power_of_two_equation_solution : ∃ k : ℕ, 
  2^2004 - 2^2003 - 2^2002 + 2^2001 = k * 2^2001 ∧ k = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_equation_solution_l702_70280


namespace NUMINAMATH_CALUDE_range_of_a_l702_70265

-- Define the sets A and B
def A : Set ℝ := {x | x ≤ 1 ∨ x ≥ 3}
def B (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 1}

-- State the theorem
theorem range_of_a (a : ℝ) : (A ∩ B a = B a) → (a ≤ 0 ∨ a ≥ 3) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l702_70265


namespace NUMINAMATH_CALUDE_initial_games_eq_sum_l702_70237

/-- Represents the number of video games Cody had initially -/
def initial_games : ℕ := 9

/-- Represents the number of video games Cody gave away -/
def games_given_away : ℕ := 4

/-- Represents the number of video games Cody still has -/
def games_remaining : ℕ := 5

/-- Theorem stating that the initial number of games equals the sum of games given away and games remaining -/
theorem initial_games_eq_sum : initial_games = games_given_away + games_remaining := by
  sorry

end NUMINAMATH_CALUDE_initial_games_eq_sum_l702_70237


namespace NUMINAMATH_CALUDE_absolute_value_not_always_zero_l702_70273

theorem absolute_value_not_always_zero : ¬ (∀ x : ℝ, |x| = 0) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_not_always_zero_l702_70273


namespace NUMINAMATH_CALUDE_triarc_area_sum_l702_70260

/-- A region bounded by three circular arcs -/
structure TriarcRegion where
  radius : ℝ
  central_angle : ℝ

/-- The area of a TriarcRegion in the form a√b + cπ -/
structure TriarcArea where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Compute the area of a TriarcRegion -/
noncomputable def compute_triarc_area (region : TriarcRegion) : TriarcArea :=
  sorry

theorem triarc_area_sum (region : TriarcRegion) 
  (h1 : region.radius = 5)
  (h2 : region.central_angle = 2 * π / 3) : 
  let area := compute_triarc_area region
  area.a + area.b + area.c = -28.25 := by
  sorry

end NUMINAMATH_CALUDE_triarc_area_sum_l702_70260


namespace NUMINAMATH_CALUDE_root_implies_k_value_l702_70258

theorem root_implies_k_value (k : ℝ) : 
  (2 * 7^2 + 3 * 7 - k = 0) → k = 119 := by
  sorry

end NUMINAMATH_CALUDE_root_implies_k_value_l702_70258


namespace NUMINAMATH_CALUDE_lunks_needed_for_apples_l702_70234

-- Define the exchange rates
def lunks_to_kunks (l : ℚ) : ℚ := l * (2/4)
def kunks_to_apples (k : ℚ) : ℚ := k * (5/3)

-- Theorem statement
theorem lunks_needed_for_apples (n : ℚ) : 
  kunks_to_apples (lunks_to_kunks 18) = 15 := by
  sorry

#check lunks_needed_for_apples

end NUMINAMATH_CALUDE_lunks_needed_for_apples_l702_70234


namespace NUMINAMATH_CALUDE_quadratic_root_proof_l702_70266

theorem quadratic_root_proof (v : ℝ) : 
  v = 7 → (5 * (((-21 - Real.sqrt 301) / 10) ^ 2) + 21 * ((-21 - Real.sqrt 301) / 10) + v = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_proof_l702_70266


namespace NUMINAMATH_CALUDE_cos15_cos45_minus_sin165_sin45_l702_70231

theorem cos15_cos45_minus_sin165_sin45 :
  Real.cos (15 * π / 180) * Real.cos (45 * π / 180) - 
  Real.sin (165 * π / 180) * Real.sin (45 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos15_cos45_minus_sin165_sin45_l702_70231


namespace NUMINAMATH_CALUDE_midpoints_locus_centers_locus_l702_70275

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a quadrilateral -/
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Represents a rectangle -/
structure Rectangle where
  center : Point
  width : ℝ
  height : ℝ

/-- Given a quadrilateral, returns a set of rectangles around it -/
def rectanglesAroundQuadrilateral (q : Quadrilateral) : Set Rectangle := sorry

/-- Returns the midpoints of a rectangle's sides -/
def midpointsOfRectangle (r : Rectangle) : Set Point := sorry

/-- Returns the center of a rectangle -/
def centerOfRectangle (r : Rectangle) : Point := sorry

/-- Checks if a point lies on a circle with given center and radius -/
def isOnCircle (p : Point) (center : Point) (radius : ℝ) : Prop := sorry

/-- Theorem: Locus of midpoints of rectangles' sides -/
theorem midpoints_locus (q : Quadrilateral) : 
  ∀ r ∈ rectanglesAroundQuadrilateral q, 
  ∀ m ∈ midpointsOfRectangle r,
  (isOnCircle m (Point.mk ((q.A.x + q.C.x) / 2) ((q.A.y + q.C.y) / 2)) ((q.C.x - q.A.x) / 2)) ∨
  (isOnCircle m (Point.mk ((q.B.x + q.D.x) / 2) ((q.B.y + q.D.y) / 2)) ((q.D.x - q.B.x) / 2)) := by
  sorry

/-- Theorem: Locus of centers of rectangles -/
theorem centers_locus (q : Quadrilateral) :
  let K1 : Point := Point.mk ((q.A.x + q.C.x) / 2) ((q.A.y + q.C.y) / 2)
  let K2 : Point := Point.mk ((q.B.x + q.D.x) / 2) ((q.B.y + q.D.y) / 2)
  ∀ r ∈ rectanglesAroundQuadrilateral q,
  isOnCircle (centerOfRectangle r) (Point.mk ((K1.x + K2.x) / 2) ((K1.y + K2.y) / 2)) ((K2.x - K1.x) / 2) := by
  sorry

end NUMINAMATH_CALUDE_midpoints_locus_centers_locus_l702_70275


namespace NUMINAMATH_CALUDE_powerSum7Seq_36th_l702_70206

/-- Sequence of sums of distinct powers of 7 -/
def powerSum7Seq : ℕ → ℕ
  | 0 => 1
  | n + 1 => powerSum7Seq n + 7^(n.log2)

/-- The 36th number in the sequence is 16856 -/
theorem powerSum7Seq_36th : powerSum7Seq 35 = 16856 := by
  sorry

end NUMINAMATH_CALUDE_powerSum7Seq_36th_l702_70206


namespace NUMINAMATH_CALUDE_tech_club_theorem_l702_70299

/-- The number of students in the tech club who take neither coding nor robotics -/
def students_taking_neither (total : ℕ) (coding : ℕ) (robotics : ℕ) (both : ℕ) : ℕ :=
  total - (coding + robotics - both)

/-- Theorem: Given the conditions from the problem, 20 students take neither coding nor robotics -/
theorem tech_club_theorem :
  students_taking_neither 150 80 70 20 = 20 := by
  sorry

end NUMINAMATH_CALUDE_tech_club_theorem_l702_70299


namespace NUMINAMATH_CALUDE_sum_seven_consecutive_integers_l702_70287

theorem sum_seven_consecutive_integers (m : ℤ) :
  m + (m + 1) + (m + 2) + (m + 3) + (m + 4) + (m + 5) + (m + 6) = 7 * m + 21 := by
  sorry

end NUMINAMATH_CALUDE_sum_seven_consecutive_integers_l702_70287


namespace NUMINAMATH_CALUDE_odd_function_value_l702_70281

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_value 
  (f : ℝ → ℝ) 
  (h_odd : is_odd_function f) 
  (h_neg : ∀ x < 0, f x = 1 / (x + 1)) : 
  f (1/2) = -2 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_value_l702_70281


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l702_70204

theorem necessary_but_not_sufficient (x : ℝ) :
  ((x - 5) / (2 - x) > 0 → abs (x - 1) < 4) ∧
  (∃ y : ℝ, abs (y - 1) < 4 ∧ ¬((y - 5) / (2 - y) > 0)) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l702_70204


namespace NUMINAMATH_CALUDE_average_age_combined_l702_70261

theorem average_age_combined (num_students : ℕ) (num_parents : ℕ) 
  (avg_age_students : ℚ) (avg_age_parents : ℚ) :
  num_students = 40 →
  num_parents = 60 →
  avg_age_students = 12 →
  avg_age_parents = 40 →
  ((num_students : ℚ) * avg_age_students + (num_parents : ℚ) * avg_age_parents) / 
    ((num_students : ℚ) + (num_parents : ℚ)) = 28.8 := by
  sorry

end NUMINAMATH_CALUDE_average_age_combined_l702_70261


namespace NUMINAMATH_CALUDE_police_force_ratio_l702_70289

/-- Given a police force with the following properties:
  * 20% of female officers were on duty
  * 100 officers were on duty that night
  * The police force has 250 female officers
  Prove that the ratio of female officers to total officers on duty is 1:2 -/
theorem police_force_ratio : 
  ∀ (total_female : ℕ) (on_duty : ℕ) (female_percent : ℚ),
  total_female = 250 →
  on_duty = 100 →
  female_percent = 1/5 →
  (female_percent * total_female) / on_duty = 1/2 := by
sorry

end NUMINAMATH_CALUDE_police_force_ratio_l702_70289


namespace NUMINAMATH_CALUDE_expected_value_unfair_coin_l702_70297

/-- The expected value of an unfair coin flip -/
theorem expected_value_unfair_coin : 
  let p_heads : ℚ := 2/3
  let p_tails : ℚ := 1/3
  let gain_heads : ℚ := 5
  let loss_tails : ℚ := -9
  p_heads * gain_heads + p_tails * loss_tails = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_expected_value_unfair_coin_l702_70297


namespace NUMINAMATH_CALUDE_probability_even_sum_two_wheels_l702_70246

theorem probability_even_sum_two_wheels : 
  let wheel1_total := 6
  let wheel1_even := 3
  let wheel2_total := 4
  let wheel2_even := 3
  let prob_wheel1_even := wheel1_even / wheel1_total
  let prob_wheel1_odd := 1 - prob_wheel1_even
  let prob_wheel2_even := wheel2_even / wheel2_total
  let prob_wheel2_odd := 1 - prob_wheel2_even
  let prob_both_even := prob_wheel1_even * prob_wheel2_even
  let prob_both_odd := prob_wheel1_odd * prob_wheel2_odd
  prob_both_even + prob_both_odd = 1/2
:= by sorry

end NUMINAMATH_CALUDE_probability_even_sum_two_wheels_l702_70246


namespace NUMINAMATH_CALUDE_people_per_car_l702_70230

theorem people_per_car (total_people : ℕ) (num_cars : ℕ) (h1 : total_people = 63) (h2 : num_cars = 9) :
  total_people / num_cars = 7 := by
sorry

end NUMINAMATH_CALUDE_people_per_car_l702_70230


namespace NUMINAMATH_CALUDE_triangle_inequality_expression_l702_70211

theorem triangle_inequality_expression (a b c : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) : 
  a^2 - 2*a*b + b^2 - c^2 < 0 :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_expression_l702_70211


namespace NUMINAMATH_CALUDE_ceiling_of_negative_two_point_four_l702_70238

theorem ceiling_of_negative_two_point_four :
  ⌈(-2.4 : ℝ)⌉ = -2 := by sorry

end NUMINAMATH_CALUDE_ceiling_of_negative_two_point_four_l702_70238


namespace NUMINAMATH_CALUDE_lana_morning_muffins_l702_70248

/-- Proves that Lana sold 12 muffins in the morning given the conditions of the bake sale -/
theorem lana_morning_muffins (total_goal : ℕ) (afternoon_sales : ℕ) (remaining : ℕ) 
  (h1 : total_goal = 20)
  (h2 : afternoon_sales = 4)
  (h3 : remaining = 4) :
  total_goal = afternoon_sales + remaining + 12 := by
  sorry

end NUMINAMATH_CALUDE_lana_morning_muffins_l702_70248


namespace NUMINAMATH_CALUDE_inverse_81_mod_101_l702_70222

theorem inverse_81_mod_101 (h : (9⁻¹ : ZMod 101) = 65) : (81⁻¹ : ZMod 101) = 84 := by
  sorry

end NUMINAMATH_CALUDE_inverse_81_mod_101_l702_70222


namespace NUMINAMATH_CALUDE_sum_equality_l702_70210

theorem sum_equality (a b : ℝ) (h : a/b + a/b^2 + a/b^3 + a/b^4 + a/b^5 = 3) :
  (∑' n, (2*a) / (a+b)^n) = (6*(1 - 1/b^5)) / (4 - 1/b^5) :=
by sorry

end NUMINAMATH_CALUDE_sum_equality_l702_70210


namespace NUMINAMATH_CALUDE_inscribed_quadrilateral_side_lengths_l702_70269

/-- A quadrilateral inscribed in a circle with given properties has specific side lengths -/
theorem inscribed_quadrilateral_side_lengths (R : ℝ) (d₁ d₂ : ℝ) :
  R = 25 →
  d₁ = 48 →
  d₂ = 40 →
  ∃ (a b c d : ℝ),
    a = 5 * Real.sqrt 10 ∧
    b = 9 * Real.sqrt 10 ∧
    c = 13 * Real.sqrt 10 ∧
    d = 15 * Real.sqrt 10 ∧
    a^2 + c^2 = d₁^2 ∧
    b^2 + d^2 = d₂^2 ∧
    a * c + b * d = d₁ * d₂ :=
by sorry

end NUMINAMATH_CALUDE_inscribed_quadrilateral_side_lengths_l702_70269


namespace NUMINAMATH_CALUDE_inequality_solution_l702_70293

theorem inequality_solution (x : ℝ) : 
  (x * (x - 1)) / ((x - 5)^2) ≥ 15 ↔ 
  (x ≤ 4.09 ∨ x ≥ 6.56) ∧ x ≠ 5 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l702_70293


namespace NUMINAMATH_CALUDE_equation_solutions_l702_70249

theorem equation_solutions :
  ∀ x : ℝ, (Real.sqrt ((3 + 2 * Real.sqrt 2) ^ x) + Real.sqrt ((3 - 2 * Real.sqrt 2) ^ x) = 6) ↔ (x = 2 ∨ x = -2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l702_70249


namespace NUMINAMATH_CALUDE_no_equal_volume_increase_l702_70291

theorem no_equal_volume_increase (x : ℝ) : ¬ (
  let R : ℝ := 10
  let H : ℝ := 5
  let V (r h : ℝ) := Real.pi * r^2 * h
  V (R + x) H - V R H = V R (H + x) - V R H
) := by sorry

end NUMINAMATH_CALUDE_no_equal_volume_increase_l702_70291


namespace NUMINAMATH_CALUDE_prob_snow_at_least_one_day_l702_70268

-- Define the probabilities
def prob_snow_friday : ℝ := 0.30
def prob_snow_monday : ℝ := 0.45

-- Theorem statement
theorem prob_snow_at_least_one_day : 
  let prob_no_snow_friday := 1 - prob_snow_friday
  let prob_no_snow_monday := 1 - prob_snow_monday
  let prob_no_snow_both := prob_no_snow_friday * prob_no_snow_monday
  1 - prob_no_snow_both = 0.615 := by
  sorry

end NUMINAMATH_CALUDE_prob_snow_at_least_one_day_l702_70268


namespace NUMINAMATH_CALUDE_marbles_exceed_200_l702_70292

theorem marbles_exceed_200 : ∃ k : ℕ, (∀ j : ℕ, j < k → 5 * 2^j ≤ 200) ∧ 5 * 2^k > 200 ∧ k = 6 := by
  sorry

end NUMINAMATH_CALUDE_marbles_exceed_200_l702_70292


namespace NUMINAMATH_CALUDE_inequality_proof_l702_70220

theorem inequality_proof (a b c d : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) 
  (h5 : a / b < c / d) : 
  a / b < (a + c) / (b + d) ∧ (a + c) / (b + d) < c / d := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l702_70220


namespace NUMINAMATH_CALUDE_subset_union_subset_l702_70255

theorem subset_union_subset (U M N : Set α) : M ⊆ U → N ⊆ U → (M ∪ N) ⊆ U := by sorry

end NUMINAMATH_CALUDE_subset_union_subset_l702_70255


namespace NUMINAMATH_CALUDE_factor_81_minus_27x_cubed_l702_70295

theorem factor_81_minus_27x_cubed (x : ℝ) : 81 - 27 * x^3 = 27 * (3 - x) * (9 + 3*x + x^2) := by
  sorry

end NUMINAMATH_CALUDE_factor_81_minus_27x_cubed_l702_70295


namespace NUMINAMATH_CALUDE_units_digit_of_power_product_l702_70284

theorem units_digit_of_power_product : 2^1201 * 4^1302 * 6^1403 ≡ 2 [ZMOD 10] := by sorry

end NUMINAMATH_CALUDE_units_digit_of_power_product_l702_70284


namespace NUMINAMATH_CALUDE_smallest_x_for_equation_l702_70262

theorem smallest_x_for_equation : 
  ∃ (x : ℝ), x ≠ 6 ∧ x ≠ -4 ∧
  (x^2 - 3*x - 18) / (x - 6) = 5 / (x + 4) ∧
  ∀ (y : ℝ), y ≠ 6 ∧ y ≠ -4 ∧ (y^2 - 3*y - 18) / (y - 6) = 5 / (y + 4) → x ≤ y ∧
  x = (-7 - Real.sqrt 21) / 2 :=
sorry

end NUMINAMATH_CALUDE_smallest_x_for_equation_l702_70262


namespace NUMINAMATH_CALUDE_cone_surface_area_l702_70201

/-- The surface area of a cone, given its lateral surface properties -/
theorem cone_surface_area (r : Real) (arc_length : Real) : 
  r = 4 → arc_length = 4 * Real.pi → 
  (π * (arc_length / (2 * π))^2) + (1/2 * r * arc_length) = 12 * π := by
sorry

end NUMINAMATH_CALUDE_cone_surface_area_l702_70201


namespace NUMINAMATH_CALUDE_ab_pos_necessary_not_sufficient_l702_70203

theorem ab_pos_necessary_not_sufficient (a b : ℝ) :
  (∃ a b : ℝ, (b / a + a / b > 2) ∧ (a * b > 0)) ∧
  (∃ a b : ℝ, (a * b > 0) ∧ ¬(b / a + a / b > 2)) ∧
  (∀ a b : ℝ, (b / a + a / b > 2) → (a * b > 0)) :=
by sorry

end NUMINAMATH_CALUDE_ab_pos_necessary_not_sufficient_l702_70203


namespace NUMINAMATH_CALUDE_specific_trapezoid_area_l702_70279

/-- Represents a trapezoid ABCD with given properties -/
structure IsoscelesTrapezoid where
  AB : ℝ
  CD : ℝ
  AD_eq_BC : AD = BC
  O_interior : O_in_interior
  OT : ℝ

/-- The area of the isosceles trapezoid with the given properties -/
def trapezoid_area (t : IsoscelesTrapezoid) : ℝ := sorry

/-- Theorem stating the area of the specific isosceles trapezoid -/
theorem specific_trapezoid_area :
  ∃ (t : IsoscelesTrapezoid),
    t.AB = 6 ∧ t.CD = 12 ∧ t.OT = 18 ∧
    trapezoid_area t = 54 + 27 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_specific_trapezoid_area_l702_70279


namespace NUMINAMATH_CALUDE_unique_common_roots_l702_70214

/-- Two cubic polynomials with two distinct common roots -/
def has_two_common_roots (p q : ℝ) : Prop :=
  ∃ (r₁ r₂ : ℝ), r₁ ≠ r₂ ∧
    r₁^3 + p*r₁^2 + 8*r₁ + 10 = 0 ∧
    r₁^3 + q*r₁^2 + 17*r₁ + 15 = 0 ∧
    r₂^3 + p*r₂^2 + 8*r₂ + 10 = 0 ∧
    r₂^3 + q*r₂^2 + 17*r₂ + 15 = 0

/-- The unique solution for p and q -/
theorem unique_common_roots :
  ∃! (p q : ℝ), has_two_common_roots p q ∧ p = 19 ∧ q = 28 := by
  sorry

end NUMINAMATH_CALUDE_unique_common_roots_l702_70214


namespace NUMINAMATH_CALUDE_square_root_three_expansion_l702_70272

theorem square_root_three_expansion {a b m n : ℕ+} :
  a + b * Real.sqrt 3 = (m + n * Real.sqrt 3) ^ 2 →
  a = m ^ 2 + 3 * n ^ 2 ∧ b = 2 * m * n :=
by sorry

end NUMINAMATH_CALUDE_square_root_three_expansion_l702_70272


namespace NUMINAMATH_CALUDE_problem_solution_l702_70212

def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0

def q (x : ℝ) : Prop := (x - 3) / (x - 2) < 0

theorem problem_solution :
  (∀ x : ℝ, (p x 1 ∧ q x) ↔ (2 < x ∧ x < 3)) ∧
  (∀ a : ℝ, (a > 0 ∧ (∀ x : ℝ, q x → p x a) ∧ (∃ x : ℝ, p x a ∧ ¬q x)) ↔ (1 ≤ a ∧ a ≤ 2)) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l702_70212


namespace NUMINAMATH_CALUDE_imoProof_l702_70251

theorem imoProof (d : ℕ) (h1 : d ≠ 2) (h2 : d ≠ 5) (h3 : d ≠ 13) (h4 : d > 0) : 
  ∃ (a b : ℕ), a ∈ ({2, 5, 13, d} : Set ℕ) ∧ 
               b ∈ ({2, 5, 13, d} : Set ℕ) ∧ 
               a ≠ b ∧ 
               ¬∃ (k : ℕ), a * b - 1 = k * k :=
by sorry

end NUMINAMATH_CALUDE_imoProof_l702_70251


namespace NUMINAMATH_CALUDE_book_pages_l702_70232

/-- The number of pages Hallie read on the first day -/
def pages_day1 : ℕ := 63

/-- The number of pages Hallie read on the second day -/
def pages_day2 : ℕ := 2 * pages_day1

/-- The number of pages Hallie read on the third day -/
def pages_day3 : ℕ := pages_day2 + 10

/-- The number of pages Hallie read on the fourth day -/
def pages_day4 : ℕ := 29

/-- The total number of pages in the book -/
def total_pages : ℕ := pages_day1 + pages_day2 + pages_day3 + pages_day4

theorem book_pages : total_pages = 354 := by sorry

end NUMINAMATH_CALUDE_book_pages_l702_70232


namespace NUMINAMATH_CALUDE_construct_angle_l702_70244

-- Define the given angle
def given_angle : ℝ := 70

-- Define the target angle
def target_angle : ℝ := 40

-- Theorem statement
theorem construct_angle (straight_angle : ℝ) (right_angle : ℝ) 
  (h1 : straight_angle = 180) 
  (h2 : right_angle = 90) : 
  ∃ (constructed_angle : ℝ), constructed_angle = target_angle :=
sorry

end NUMINAMATH_CALUDE_construct_angle_l702_70244


namespace NUMINAMATH_CALUDE_january_salary_l702_70221

-- Define variables for each month's salary
variable (jan feb mar apr may : ℕ)

-- Define the conditions
def condition1 : Prop := (jan + feb + mar + apr) / 4 = 8000
def condition2 : Prop := (feb + mar + apr + may) / 4 = 8700
def condition3 : Prop := may = 6500

-- Theorem statement
theorem january_salary 
  (h1 : condition1 jan feb mar apr)
  (h2 : condition2 feb mar apr may)
  (h3 : condition3 may) :
  jan = 3700 := by
  sorry

end NUMINAMATH_CALUDE_january_salary_l702_70221


namespace NUMINAMATH_CALUDE_point_on_line_value_l702_70202

theorem point_on_line_value (x y : ℝ) (h1 : y = x + 2) (h2 : 1 < y) (h3 : y < 3) :
  Real.sqrt (y^2 - 8*x) + Real.sqrt (y^2 + 2*x + 5) = 5 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_value_l702_70202


namespace NUMINAMATH_CALUDE_yard_length_l702_70270

theorem yard_length (n : ℕ) (d : ℝ) (h1 : n = 26) (h2 : d = 24) : 
  (n - 1) * d = 600 := by
  sorry

end NUMINAMATH_CALUDE_yard_length_l702_70270


namespace NUMINAMATH_CALUDE_pentagon_largest_angle_l702_70225

theorem pentagon_largest_angle (P Q R S T : ℝ) : 
  P = 75 →
  Q = 110 →
  R = S →
  T = 3 * R - 20 →
  P + Q + R + S + T = 540 →
  max P (max Q (max R (max S T))) = 217 :=
sorry

end NUMINAMATH_CALUDE_pentagon_largest_angle_l702_70225


namespace NUMINAMATH_CALUDE_classroom_attendance_l702_70240

theorem classroom_attendance (students_in_restroom : ℕ) 
  (total_students : ℕ) (rows : ℕ) (desks_per_row : ℕ) 
  (occupancy_rate : ℚ) :
  students_in_restroom = 2 →
  total_students = 23 →
  rows = 4 →
  desks_per_row = 6 →
  occupancy_rate = 2/3 →
  ∃ (m : ℕ), m * students_in_restroom - 1 = 
    total_students - (↑(rows * desks_per_row) * occupancy_rate).floor - students_in_restroom ∧
  m = 4 := by
sorry

end NUMINAMATH_CALUDE_classroom_attendance_l702_70240


namespace NUMINAMATH_CALUDE_train_passing_jogger_time_l702_70243

/-- The time taken for a train to pass a jogger under specific conditions -/
theorem train_passing_jogger_time : 
  let jogger_speed : ℝ := 9 -- km/hr
  let train_speed : ℝ := 45 -- km/hr
  let initial_distance : ℝ := 240 -- meters
  let train_length : ℝ := 120 -- meters

  let jogger_speed_ms : ℝ := jogger_speed * 1000 / 3600 -- Convert to m/s
  let train_speed_ms : ℝ := train_speed * 1000 / 3600 -- Convert to m/s
  let relative_speed : ℝ := train_speed_ms - jogger_speed_ms
  let total_distance : ℝ := initial_distance + train_length
  let time : ℝ := total_distance / relative_speed

  time = 36 := by sorry

end NUMINAMATH_CALUDE_train_passing_jogger_time_l702_70243


namespace NUMINAMATH_CALUDE_total_rainfall_l702_70296

def rainfall_problem (first_week : ℝ) (second_week : ℝ) : Prop :=
  (second_week = 1.5 * first_week) ∧
  (second_week = 15) ∧
  (first_week + second_week = 25)

theorem total_rainfall : ∃ (first_week second_week : ℝ), 
  rainfall_problem first_week second_week :=
by
  sorry

end NUMINAMATH_CALUDE_total_rainfall_l702_70296


namespace NUMINAMATH_CALUDE_exterior_angle_regular_hexagon_l702_70200

theorem exterior_angle_regular_hexagon :
  let n : ℕ := 6  -- Number of sides in a hexagon
  let sum_interior_angles : ℝ := 180 * (n - 2)  -- Sum of interior angles formula
  let interior_angle : ℝ := sum_interior_angles / n  -- Each interior angle in a regular polygon
  let exterior_angle : ℝ := 180 - interior_angle  -- Exterior angle is supplementary to interior angle
  exterior_angle = 60 := by sorry

end NUMINAMATH_CALUDE_exterior_angle_regular_hexagon_l702_70200


namespace NUMINAMATH_CALUDE_quadratic_minimum_at_positive_x_l702_70250

def f (x : ℝ) := 3 * x^2 - 9 * x + 2

theorem quadratic_minimum_at_positive_x :
  ∃ x : ℝ, x > 0 ∧ ∀ y : ℝ, f y ≥ f x :=
sorry

end NUMINAMATH_CALUDE_quadratic_minimum_at_positive_x_l702_70250


namespace NUMINAMATH_CALUDE_pattern_properties_l702_70245

/-- Represents a figure in the pattern -/
structure Figure where
  n : ℕ

/-- Number of squares in a figure -/
def num_squares (f : Figure) : ℕ :=
  3 + 2 * (f.n - 1)

/-- Perimeter of a figure in cm -/
def perimeter (f : Figure) : ℕ :=
  8 + 2 * (f.n - 1)

theorem pattern_properties :
  ∀ (f : Figure),
    (num_squares f = 3 + 2 * (f.n - 1)) ∧
    (perimeter f = 8 + 2 * (f.n - 1)) ∧
    (perimeter ⟨16⟩ = 38) ∧
    ((perimeter ⟨29⟩ : ℚ) / (perimeter ⟨85⟩ : ℚ) = 4 / 11) :=
by sorry

end NUMINAMATH_CALUDE_pattern_properties_l702_70245


namespace NUMINAMATH_CALUDE_trapezoid_median_length_l702_70247

/-- Given a triangle and a trapezoid with the same height, prove that the median of the trapezoid is 18 inches when the triangle's base is 36 inches and their areas are equal. -/
theorem trapezoid_median_length (h : ℝ) (h_pos : h > 0) : 
  let triangle_base : ℝ := 36
  let triangle_area : ℝ := (1 / 2) * triangle_base * h
  let trapezoid_median : ℝ := triangle_area / h
  trapezoid_median = 18 := by sorry

end NUMINAMATH_CALUDE_trapezoid_median_length_l702_70247


namespace NUMINAMATH_CALUDE_math_homework_pages_l702_70235

theorem math_homework_pages (total_pages reading_pages : ℕ) 
  (h1 : total_pages = 7)
  (h2 : reading_pages = 2) :
  total_pages - reading_pages = 5 := by
  sorry

end NUMINAMATH_CALUDE_math_homework_pages_l702_70235


namespace NUMINAMATH_CALUDE_complex_pure_imaginary_l702_70239

theorem complex_pure_imaginary (a : ℝ) : 
  (a^2 - 3*a + 2 : ℂ) + (a - 1 : ℂ) * Complex.I = Complex.I * (b : ℝ) → a = 2 :=
by sorry

end NUMINAMATH_CALUDE_complex_pure_imaginary_l702_70239


namespace NUMINAMATH_CALUDE_factorization_problem_l702_70253

theorem factorization_problem (A B : ℤ) : 
  (∀ y : ℝ, 20 * y^2 - 103 * y + 42 = (A * y - 21) * (B * y - 2)) →
  A * B + A = 30 := by
sorry

end NUMINAMATH_CALUDE_factorization_problem_l702_70253

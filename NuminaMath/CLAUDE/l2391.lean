import Mathlib

namespace unique_base_conversion_l2391_239167

/-- Converts a number from base 6 to base 10 -/
def base6ToBase10 (n : ℕ) : ℕ :=
  (n / 10) * 6 + (n % 10)

/-- Converts a number from base b to base 10 -/
def baseBToBase10 (n : ℕ) (b : ℕ) : ℕ :=
  (n / 100) * b^2 + ((n / 10) % 10) * b + (n % 10)

theorem unique_base_conversion : 
  ∃! (b : ℕ), b > 0 ∧ base6ToBase10 45 = baseBToBase10 113 b :=
by sorry

end unique_base_conversion_l2391_239167


namespace inequality_one_inequality_two_l2391_239151

-- Statement 1
theorem inequality_one (a : ℝ) (h : a > 3) : a + 4 / (a - 3) ≥ 7 := by
  sorry

-- Statement 2
theorem inequality_two (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  4 / x + 9 / y ≥ 25 := by
  sorry

end inequality_one_inequality_two_l2391_239151


namespace f_increasing_m_range_l2391_239192

/-- A function f(x) that depends on a parameter m -/
def f (m : ℝ) (x : ℝ) : ℝ := x * |x - m| + 2 * x - 3

/-- Theorem stating that if f is increasing on ℝ, then m is in the interval [-2, 2] -/
theorem f_increasing_m_range (m : ℝ) :
  (∀ x y : ℝ, x < y → f m x < f m y) → m ∈ Set.Icc (-2 : ℝ) 2 := by
  sorry

end f_increasing_m_range_l2391_239192


namespace quadratic_perfect_square_l2391_239106

theorem quadratic_perfect_square (x : ℝ) : ∃ (a b : ℝ), x^2 - 20*x + 100 = (a*x + b)^2 := by
  sorry

end quadratic_perfect_square_l2391_239106


namespace initial_bonus_is_500_l2391_239155

/-- Represents the bonus calculation for a teacher based on student test scores. -/
structure BonusCalculation where
  numStudents : Nat
  baseAverage : Nat
  bonusThreshold : Nat
  bonusPerPoint : Nat
  maxScore : Nat
  gradedTests : Nat
  gradedAverage : Nat
  lastTwoTestsScore : Nat
  totalBonus : Nat

/-- Calculates the initial bonus amount given the bonus calculation parameters. -/
def initialBonusAmount (bc : BonusCalculation) : Nat :=
  sorry

/-- Theorem stating that given the specific conditions, the initial bonus amount is $500. -/
theorem initial_bonus_is_500 (bc : BonusCalculation) 
  (h1 : bc.numStudents = 10)
  (h2 : bc.baseAverage = 75)
  (h3 : bc.bonusThreshold = 75)
  (h4 : bc.bonusPerPoint = 10)
  (h5 : bc.maxScore = 150)
  (h6 : bc.gradedTests = 8)
  (h7 : bc.gradedAverage = 70)
  (h8 : bc.lastTwoTestsScore = 290)
  (h9 : bc.totalBonus = 600) :
  initialBonusAmount bc = 500 :=
sorry

end initial_bonus_is_500_l2391_239155


namespace circle_center_sum_l2391_239118

theorem circle_center_sum (x y : ℝ) : 
  (∀ X Y : ℝ, X^2 + Y^2 = 6*X - 8*Y + 24 ↔ (X - x)^2 + (Y - y)^2 = (x^2 + y^2 - 6*x + 8*y - 24)) →
  x + y = -1 := by
  sorry

end circle_center_sum_l2391_239118


namespace largest_number_l2391_239105

theorem largest_number (a b c d e : ℝ) : 
  a = 13579 + 1 / 2468 →
  b = 13579 - 1 / 2468 →
  c = 13579 * (1 / 2468) →
  d = 13579 / (1 / 2468) →
  e = 13579.2468 →
  d > a ∧ d > b ∧ d > c ∧ d > e :=
by sorry

end largest_number_l2391_239105


namespace no_infinite_set_with_divisibility_property_l2391_239150

theorem no_infinite_set_with_divisibility_property :
  ¬ ∃ (S : Set ℤ), Set.Infinite S ∧ 
    ∀ (a b : ℤ), a ∈ S → b ∈ S → (a^2 + b^2 - a*b) ∣ (a*b)^2 :=
sorry

end no_infinite_set_with_divisibility_property_l2391_239150


namespace x_value_l2391_239138

theorem x_value (x : ℝ) : x = 88 * (1 + 0.3) → x = 114.4 := by
  sorry

end x_value_l2391_239138


namespace sum_of_forbidden_digits_units_digit_not_in_forbidden_sum_forbidden_digits_correct_l2391_239101

def S (n : ℕ+) : ℕ := n.val * (n.val + 1) / 2

def forbidden_digits : Finset ℕ := {2, 4, 7, 9}

theorem sum_of_forbidden_digits : (forbidden_digits.sum id) = 22 := by sorry

theorem units_digit_not_in_forbidden (n : ℕ+) :
  (S n) % 10 ∉ forbidden_digits := by sorry

theorem sum_forbidden_digits_correct :
  ∃ (digits : Finset ℕ), 
    (∀ (n : ℕ+), (S n) % 10 ∉ digits) ∧
    (digits.sum id = 22) ∧
    (∀ (d : ℕ), d ∉ digits → ∃ (n : ℕ+), (S n) % 10 = d) := by sorry

end sum_of_forbidden_digits_units_digit_not_in_forbidden_sum_forbidden_digits_correct_l2391_239101


namespace min_value_of_quadratic_expression_l2391_239177

theorem min_value_of_quadratic_expression :
  (∀ x y : ℝ, x^2 + 2*x*y + y^2 ≥ 0) ∧
  (∃ x y : ℝ, x^2 + 2*x*y + y^2 = 0) := by
sorry

end min_value_of_quadratic_expression_l2391_239177


namespace fruit_salad_weight_l2391_239178

/-- The amount of melon in pounds used in the fruit salad -/
def melon_weight : ℚ := 0.25

/-- The amount of berries in pounds used in the fruit salad -/
def berries_weight : ℚ := 0.38

/-- The total amount of fruit in pounds used in the fruit salad -/
def total_fruit_weight : ℚ := melon_weight + berries_weight

theorem fruit_salad_weight : total_fruit_weight = 0.63 := by
  sorry

end fruit_salad_weight_l2391_239178


namespace parabola_midpoint_locus_l2391_239134

/-- The locus of the midpoint of chord MN on a parabola -/
theorem parabola_midpoint_locus (p : ℝ) (x y : ℝ) :
  let parabola := fun (x y : ℝ) => y^2 - 2*p*x = 0
  let normal_intersection := fun (x y m : ℝ) => y - m*x + p*(m + m^3/2) = 0
  let conjugate_diameter := fun (y m : ℝ) => m*y - p = 0
  ∃ (x₁ y₁ x₂ y₂ m : ℝ),
    parabola x₁ y₁ ∧
    parabola x₂ y₂ ∧
    normal_intersection x₂ y₂ m ∧
    conjugate_diameter y₁ m ∧
    x = (x₁ + x₂) / 2 ∧
    y = (y₁ + y₂) / 2
  →
  y^4 - (p*x)*y^2 + p^4/2 = 0 :=
by sorry

end parabola_midpoint_locus_l2391_239134


namespace distance_between_points_l2391_239146

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (3, 3)
  let p2 : ℝ × ℝ := (-2, -3)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = Real.sqrt 61 := by
  sorry

end distance_between_points_l2391_239146


namespace matrix_det_times_two_l2391_239188

def matrix_det (a b c d : ℤ) : ℤ := a * d - b * c

theorem matrix_det_times_two :
  2 * (matrix_det 5 7 2 3) = 2 := by sorry

end matrix_det_times_two_l2391_239188


namespace sufficient_not_necessary_l2391_239144

theorem sufficient_not_necessary (x : ℝ) :
  (∀ x, x ≥ 0 → 2*x + 1/(2*x + 1) ≥ 1) ∧
  (∃ x, 2*x + 1/(2*x + 1) ≥ 1 ∧ x < 0) :=
by sorry

end sufficient_not_necessary_l2391_239144


namespace least_addition_for_divisibility_l2391_239175

theorem least_addition_for_divisibility : 
  ∃! x : ℕ, x > 0 ∧ x ≤ 23 ∧ (1055 + x) % 23 = 0 ∧ ∀ y : ℕ, y < x → (1055 + y) % 23 ≠ 0 :=
by sorry

end least_addition_for_divisibility_l2391_239175


namespace tan_difference_angle_sum_l2391_239160

-- Problem 1
theorem tan_difference (A B : Real) (h : 2 * Real.tan A = 3 * Real.tan B) :
  Real.tan (A - B) = Real.sin (2 * B) / (5 - Real.cos (2 * B)) := by sorry

-- Problem 2
theorem angle_sum (α β : Real) 
  (h1 : 0 < α ∧ α < π/2) 
  (h2 : 0 < β ∧ β < π/2) 
  (h3 : Real.tan α = 1/7) 
  (h4 : Real.sin β = Real.sqrt 10 / 10) :
  α + 2*β = π/4 := by sorry

end tan_difference_angle_sum_l2391_239160


namespace truncated_pyramid_volume_l2391_239135

/-- Given a truncated pyramid with base areas S₁ and S₂ (S₁ < S₂) and volume V,
    the volume of the complete pyramid is (V * S₂ * √S₂) / (S₂ * √S₂ - S₁ * √S₁) -/
theorem truncated_pyramid_volume 
  (S₁ S₂ V : ℝ) 
  (h₁ : 0 < S₁) 
  (h₂ : 0 < S₂) 
  (h₃ : S₁ < S₂) 
  (h₄ : 0 < V) : 
  ∃ (V_full : ℝ), V_full = (V * S₂ * Real.sqrt S₂) / (S₂ * Real.sqrt S₂ - S₁ * Real.sqrt S₁) := by
  sorry

end truncated_pyramid_volume_l2391_239135


namespace rectangle_side_problem_l2391_239182

theorem rectangle_side_problem (side1 : ℝ) (side2 : ℕ) (unknown_side : ℝ) : 
  side1 = 5 →
  side2 = 12 →
  (side1 * side2 = side1 * unknown_side + 25 ∨ side1 * unknown_side = side1 * side2 + 25) →
  unknown_side = 7 ∨ unknown_side = 17 := by
sorry

end rectangle_side_problem_l2391_239182


namespace scale_division_l2391_239193

/-- Converts feet and inches to total inches -/
def feetInchesToInches (feet : ℕ) (inches : ℕ) : ℕ :=
  feet * 12 + inches

/-- Converts inches to feet and inches -/
def inchesToFeetInches (inches : ℕ) : ℕ × ℕ :=
  (inches / 12, inches % 12)

theorem scale_division (totalFeet : ℕ) (totalInches : ℕ) (parts : ℕ) 
    (h1 : totalFeet = 7) 
    (h2 : totalInches = 6) 
    (h3 : parts = 5) : 
  inchesToFeetInches (feetInchesToInches totalFeet totalInches / parts) = (1, 6) := by
  sorry

end scale_division_l2391_239193


namespace exchange_divisibility_l2391_239141

theorem exchange_divisibility (p a d : ℤ) : 
  p = 4*a + d ∧ p = a + 5*d → 
  ∃ (t : ℤ), p = 19*t ∧ a = 4*t ∧ d = 3*t ∧ p + a + d = 26*t :=
by sorry

end exchange_divisibility_l2391_239141


namespace gervais_driving_days_l2391_239170

theorem gervais_driving_days :
  let gervais_avg_miles_per_day : ℝ := 315
  let henri_total_miles : ℝ := 1250
  let difference_in_miles : ℝ := 305
  let gervais_days : ℝ := (henri_total_miles - difference_in_miles) / gervais_avg_miles_per_day
  gervais_days = 3 := by
  sorry

end gervais_driving_days_l2391_239170


namespace trig_identity_l2391_239140

theorem trig_identity (x y : ℝ) :
  Real.sin x ^ 2 + Real.sin (x + y) ^ 2 - 2 * Real.sin x * Real.sin y * Real.sin (x + y) = Real.cos y ^ 2 := by
  sorry

end trig_identity_l2391_239140


namespace square_of_difference_l2391_239125

theorem square_of_difference (a b : ℝ) : (a - b)^2 = a^2 - 2*a*b + b^2 := by
  sorry

end square_of_difference_l2391_239125


namespace parabola_shift_left_l2391_239196

/-- The analytical expression of a parabola shifted to the left -/
theorem parabola_shift_left (x y : ℝ) :
  (∀ x, y = x^2) →  -- Original parabola
  (∀ x, y = (x + 1)^2) -- Parabola shifted 1 unit left
  := by sorry

end parabola_shift_left_l2391_239196


namespace intersection_sum_is_eight_l2391_239127

noncomputable def P : ℝ × ℝ := (0, 8)
noncomputable def Q : ℝ × ℝ := (0, 0)
noncomputable def R : ℝ × ℝ := (10, 0)

noncomputable def G : ℝ × ℝ := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)
noncomputable def H : ℝ × ℝ := ((Q.1 + R.1) / 2, (Q.2 + R.2) / 2)

noncomputable def line_PH (x : ℝ) : ℝ := 
  (H.2 - P.2) / (H.1 - P.1) * (x - P.1) + P.2

theorem intersection_sum_is_eight : 
  ∃ (I : ℝ × ℝ), I.1 = G.1 ∧ I.2 = line_PH I.1 ∧ I.1 + I.2 = 8 := by
  sorry

end intersection_sum_is_eight_l2391_239127


namespace matrix_equation_solution_l2391_239137

def B : Matrix (Fin 3) (Fin 3) ℚ := !![1, 2, 3; 2, 1, 2; 3, 2, 1]

theorem matrix_equation_solution :
  ∃ (a b c : ℚ), 
    B^3 + a • B^2 + b • B + c • (1 : Matrix (Fin 3) (Fin 3) ℚ) = 0 ∧ 
    a = 0 ∧ b = -283/13 ∧ c = 902/13 := by
  sorry

end matrix_equation_solution_l2391_239137


namespace maya_total_pages_l2391_239128

/-- The total number of pages Maya read in two weeks -/
def total_pages (books_last_week : ℕ) (pages_per_book : ℕ) (reading_increase : ℕ) : ℕ :=
  let pages_last_week := books_last_week * pages_per_book
  let pages_this_week := reading_increase * pages_last_week
  pages_last_week + pages_this_week

/-- Theorem stating that Maya read 4500 pages in total -/
theorem maya_total_pages :
  total_pages 5 300 2 = 4500 := by
  sorry

end maya_total_pages_l2391_239128


namespace sand_in_last_bag_l2391_239117

theorem sand_in_last_bag (total_sand : Nat) (bag_capacity : Nat) (h1 : total_sand = 757) (h2 : bag_capacity = 65) :
  total_sand % bag_capacity = 42 := by
sorry

end sand_in_last_bag_l2391_239117


namespace rectangle_width_problem_l2391_239184

theorem rectangle_width_problem (width length area : ℝ) : 
  length = 3 * width →
  area = length * width →
  area = 75 →
  width = 5 :=
by sorry

end rectangle_width_problem_l2391_239184


namespace train_length_l2391_239119

/-- The length of a train given its speed and the time it takes to cross a bridge of known length. -/
theorem train_length (bridge_length : ℝ) (crossing_time : ℝ) (train_speed : ℝ) :
  bridge_length = 300 →
  crossing_time = 24 →
  train_speed = 50 →
  train_speed * crossing_time - bridge_length = 900 :=
by sorry

end train_length_l2391_239119


namespace parabola_vertex_in_first_quadrant_l2391_239104

/-- Given a parabola y = -x^2 + (a+1)x + (a+2) where a > 1, 
    its vertex lies in the first quadrant -/
theorem parabola_vertex_in_first_quadrant (a : ℝ) (h : a > 1) :
  let f (x : ℝ) := -x^2 + (a+1)*x + (a+2)
  let vertex_x := (a+1)/2
  let vertex_y := f vertex_x
  vertex_x > 0 ∧ vertex_y > 0 := by
  sorry

end parabola_vertex_in_first_quadrant_l2391_239104


namespace field_area_minus_ponds_l2391_239199

/-- The area of a square field with sides of 10 meters, minus the area of three non-overlapping 
    circular ponds each with a radius of 3 meters, is equal to 100 - 27π square meters. -/
theorem field_area_minus_ponds (π : ℝ) : ℝ := by
  -- Define the side length of the square field
  let square_side : ℝ := 10
  -- Define the radius of each circular pond
  let pond_radius : ℝ := 3
  -- Define the number of ponds
  let num_ponds : ℕ := 3
  -- Calculate the area of the square field
  let square_area : ℝ := square_side ^ 2
  -- Calculate the area of one circular pond
  let pond_area : ℝ := π * pond_radius ^ 2
  -- Calculate the total area of all ponds
  let total_pond_area : ℝ := num_ponds * pond_area
  -- Calculate the remaining area (field area minus pond area)
  let remaining_area : ℝ := square_area - total_pond_area
  -- Prove that the remaining area is equal to 100 - 27π
  sorry

#check field_area_minus_ponds

end field_area_minus_ponds_l2391_239199


namespace sum_59_28_rounded_equals_90_l2391_239185

def round_to_nearest_ten (n : ℤ) : ℤ :=
  10 * ((n + 5) / 10)

theorem sum_59_28_rounded_equals_90 : 
  round_to_nearest_ten (59 + 28) = 90 := by
  sorry

end sum_59_28_rounded_equals_90_l2391_239185


namespace triangle_theorem_l2391_239156

/-- Triangle ABC with sides a, b, c opposite angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem to be proved -/
theorem triangle_theorem (t : Triangle) 
  (h1 : (t.a + t.b + t.c) * (t.a - t.b + t.c) = t.a * t.c)
  (h2 : Real.sin t.A * Real.sin t.C = (Real.sqrt 3 - 1) / 4) :
  t.B = 2 * Real.pi / 3 ∧ (t.C = Real.pi / 12 ∨ t.C = Real.pi / 4) := by
  sorry

end triangle_theorem_l2391_239156


namespace voronovich_inequality_l2391_239111

theorem voronovich_inequality {a b c : ℝ} (ha : 0 < a) (hab : a < b) (hbc : b < c) :
  a^20 * b^12 + b^20 * c^12 + c^20 * a^12 < b^20 * a^12 + a^20 * c^12 + c^20 * b^12 :=
by sorry

end voronovich_inequality_l2391_239111


namespace book_ratio_is_three_to_one_l2391_239161

-- Define the number of books for each person
def elmo_books : ℕ := 24
def stu_books : ℕ := 4
def laura_books : ℕ := 2 * stu_books

-- Define the ratio of Elmo's books to Laura's books
def book_ratio : ℚ := elmo_books / laura_books

-- Theorem to prove
theorem book_ratio_is_three_to_one : book_ratio = 3 / 1 := by
  sorry

end book_ratio_is_three_to_one_l2391_239161


namespace largest_prime_factor_of_2999_l2391_239115

theorem largest_prime_factor_of_2999 : ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 2999 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 2999 → q ≤ p := by
  sorry

end largest_prime_factor_of_2999_l2391_239115


namespace save_sign_white_area_l2391_239181

/-- Represents the area covered by a letter on the sign -/
structure LetterArea where
  s : ℕ
  a : ℕ
  v : ℕ
  e : ℕ

/-- The sign with the word "SAVE" painted on it -/
structure Sign where
  width : ℕ
  height : ℕ
  letterAreas : LetterArea

/-- Calculate the white area of the sign -/
def whiteArea (sign : Sign) : ℕ :=
  sign.width * sign.height - (sign.letterAreas.s + sign.letterAreas.a + sign.letterAreas.v + sign.letterAreas.e)

/-- Theorem stating the white area of the sign is 86 square units -/
theorem save_sign_white_area :
  ∀ (sign : Sign),
    sign.width = 20 ∧
    sign.height = 7 ∧
    sign.letterAreas.s = 14 ∧
    sign.letterAreas.a = 16 ∧
    sign.letterAreas.v = 12 ∧
    sign.letterAreas.e = 12 →
    whiteArea sign = 86 := by
  sorry

end save_sign_white_area_l2391_239181


namespace buttons_per_shirt_proof_l2391_239190

/-- The number of shirts Sally sews on Monday -/
def monday_shirts : ℕ := 4

/-- The number of shirts Sally sews on Tuesday -/
def tuesday_shirts : ℕ := 3

/-- The number of shirts Sally sews on Wednesday -/
def wednesday_shirts : ℕ := 2

/-- The total number of buttons Sally needs for all shirts -/
def total_buttons : ℕ := 45

/-- The number of buttons per shirt -/
def buttons_per_shirt : ℕ := 5

theorem buttons_per_shirt_proof :
  (monday_shirts + tuesday_shirts + wednesday_shirts) * buttons_per_shirt = total_buttons :=
by sorry

end buttons_per_shirt_proof_l2391_239190


namespace dividend_calculation_l2391_239162

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 16) 
  (h2 : quotient = 8) 
  (h3 : remainder = 4) : 
  divisor * quotient + remainder = 132 := by
  sorry

end dividend_calculation_l2391_239162


namespace scientific_notation_28400_l2391_239153

theorem scientific_notation_28400 : 28400 = 2.84 * (10 ^ 4) := by
  sorry

end scientific_notation_28400_l2391_239153


namespace modulus_of_complex_fraction_l2391_239197

theorem modulus_of_complex_fraction (i : ℂ) :
  i * i = -1 →
  Complex.abs (2 * i / (1 + i)) = Real.sqrt 2 := by
  sorry

end modulus_of_complex_fraction_l2391_239197


namespace nigels_initial_amount_l2391_239126

theorem nigels_initial_amount 
  (olivia_initial : ℕ) 
  (ticket_price : ℕ) 
  (num_tickets : ℕ) 
  (amount_left : ℕ) 
  (h1 : olivia_initial = 112)
  (h2 : ticket_price = 28)
  (h3 : num_tickets = 6)
  (h4 : amount_left = 83) :
  olivia_initial + (ticket_price * num_tickets - (olivia_initial - amount_left)) = 251 :=
by sorry

end nigels_initial_amount_l2391_239126


namespace sophie_goal_theorem_l2391_239157

def sophie_marks : List ℚ := [73/100, 82/100, 85/100]

def total_tests : ℕ := 5

def goal_average : ℚ := 80/100

def pair_D : List ℚ := [73/100, 83/100]
def pair_A : List ℚ := [79/100, 82/100]
def pair_B : List ℚ := [70/100, 91/100]
def pair_C : List ℚ := [76/100, 86/100]

theorem sophie_goal_theorem :
  (sophie_marks.sum + pair_D.sum) / total_tests < goal_average ∧
  (sophie_marks.sum + pair_A.sum) / total_tests ≥ goal_average ∧
  (sophie_marks.sum + pair_B.sum) / total_tests ≥ goal_average ∧
  (sophie_marks.sum + pair_C.sum) / total_tests ≥ goal_average :=
by sorry

end sophie_goal_theorem_l2391_239157


namespace village_population_l2391_239114

theorem village_population (P : ℝ) 
  (h1 : 0.9 * P * 0.8 = 4500) : P = 6250 := by
  sorry

end village_population_l2391_239114


namespace largest_gold_coins_distribution_l2391_239172

theorem largest_gold_coins_distribution (total : ℕ) : 
  (∃ (k : ℕ), total = 13 * k + 3) →
  total < 150 →
  (∀ n : ℕ, (∃ (k : ℕ), n = 13 * k + 3) → n < 150 → n ≤ total) →
  total = 146 :=
by sorry

end largest_gold_coins_distribution_l2391_239172


namespace polynomial_equivalence_l2391_239113

/-- Given a polynomial f(x,y,z) = x³ + 2y³ + 4z³ - 6xyz, prove that for all real numbers a, b, and c,
    f(a,b,c) = 0 if and only if a + b∛2 + c∛4 = 0 -/
theorem polynomial_equivalence (a b c : ℝ) :
  a^3 + 2*b^3 + 4*c^3 - 6*a*b*c = 0 ↔ a + b*(2^(1/3)) + c*(4^(1/3)) = 0 := by
  sorry

end polynomial_equivalence_l2391_239113


namespace unique_zero_location_l2391_239171

def has_unique_zero_in (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃! x, a < x ∧ x < b ∧ f x = 0

theorem unique_zero_location (f : ℝ → ℝ) :
  has_unique_zero_in f 0 16 ∧
  has_unique_zero_in f 0 8 ∧
  has_unique_zero_in f 0 6 ∧
  has_unique_zero_in f 2 4 →
  ¬ ∃ x, 0 < x ∧ x < 2 ∧ f x = 0 :=
by sorry

end unique_zero_location_l2391_239171


namespace first_day_cost_l2391_239176

/-- The cost of a hamburger -/
def hamburger_cost : ℚ := sorry

/-- The cost of a hot dog -/
def hot_dog_cost : ℚ := 1

/-- The cost of 2 hamburgers and 3 hot dogs -/
def second_day_cost : ℚ := 7

theorem first_day_cost : 3 * hamburger_cost + 4 * hot_dog_cost = 10 :=
  by sorry

end first_day_cost_l2391_239176


namespace decimal_order_l2391_239122

theorem decimal_order : 0.6 < 0.67 ∧ 0.67 < 0.676 ∧ 0.676 < 0.677 := by
  sorry

end decimal_order_l2391_239122


namespace central_angle_alice_bob_l2391_239186

/-- Represents a point on the Earth's surface with latitude and longitude -/
structure EarthPoint where
  latitude : Real
  longitude : Real

/-- Calculates the central angle between two points on a spherical Earth -/
noncomputable def centralAngle (a b : EarthPoint) : Real :=
  sorry

/-- The location of Alice near Quito, Ecuador -/
def alice : EarthPoint :=
  { latitude := 0, longitude := -78 }

/-- The location of Bob near Vladivostok, Russia -/
def bob : EarthPoint :=
  { latitude := 43, longitude := 132 }

/-- Theorem stating that the central angle between Alice and Bob is 150 degrees -/
theorem central_angle_alice_bob :
  centralAngle alice bob = 150 := by
  sorry

end central_angle_alice_bob_l2391_239186


namespace village_population_l2391_239158

theorem village_population (partial_population : ℕ) (percentage : ℚ) (total_population : ℕ) :
  percentage = 90 / 100 →
  partial_population = 23040 →
  (percentage * total_population : ℚ) = partial_population →
  total_population = 25600 := by
sorry

end village_population_l2391_239158


namespace remaining_money_after_bike_purchase_l2391_239173

/-- Calculates the remaining money after buying a bike with quarters from jars --/
theorem remaining_money_after_bike_purchase (num_jars : ℕ) (quarters_per_jar : ℕ) (bike_cost : ℕ) : 
  num_jars = 5 → 
  quarters_per_jar = 160 → 
  bike_cost = 180 → 
  (num_jars * quarters_per_jar * 25 - bike_cost * 100) / 100 = 20 := by
  sorry

#check remaining_money_after_bike_purchase

end remaining_money_after_bike_purchase_l2391_239173


namespace log_seven_forty_eight_l2391_239132

theorem log_seven_forty_eight (a b : ℝ) (h1 : Real.log 3 / Real.log 7 = a) (h2 : Real.log 4 / Real.log 7 = b) :
  Real.log 48 / Real.log 7 = a + 2 * b := by
  sorry

end log_seven_forty_eight_l2391_239132


namespace estimate_smaller_than_actual_l2391_239191

theorem estimate_smaller_than_actual (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxy : x > y) : 
  (x - z) - (y + z) < x - y := by
  sorry

end estimate_smaller_than_actual_l2391_239191


namespace largest_centrally_symmetric_polygon_area_l2391_239139

/-- A triangle in a 2D plane --/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- A polygon in a 2D plane --/
structure Polygon where
  vertices : List (ℝ × ℝ)

/-- Checks if a polygon is centrally symmetric --/
def isCentrallySymmetric (p : Polygon) : Prop := sorry

/-- Calculates the area of a polygon --/
def area (p : Polygon) : ℝ := sorry

/-- Checks if a polygon is inside a triangle --/
def isInside (p : Polygon) (t : Triangle) : Prop := sorry

/-- Theorem: The largest possible area of a centrally symmetric polygon inside a triangle is 2/3 of the triangle's area --/
theorem largest_centrally_symmetric_polygon_area (t : Triangle) :
  ∃ (p : Polygon), isCentrallySymmetric p ∧ isInside p t ∧
    ∀ (q : Polygon), isCentrallySymmetric q → isInside q t →
      area p ≥ area q ∧ area p = (2/3) * area (Polygon.mk [t.A, t.B, t.C]) :=
sorry

end largest_centrally_symmetric_polygon_area_l2391_239139


namespace marys_average_speed_l2391_239168

/-- Mary's round trip walking problem -/
theorem marys_average_speed (uphill_distance downhill_distance : ℝ)
                             (uphill_time downhill_time : ℝ)
                             (h1 : uphill_distance = 1.5)
                             (h2 : downhill_distance = 1.5)
                             (h3 : uphill_time = 45 / 60)
                             (h4 : downhill_time = 15 / 60) :
  (uphill_distance + downhill_distance) / (uphill_time + downhill_time) = 3 := by
  sorry

#check marys_average_speed

end marys_average_speed_l2391_239168


namespace company_employees_l2391_239108

/-- If a company has 15% more employees in December than in January,
    and it has 490 employees in December, then it had 426 employees in January. -/
theorem company_employees (december_employees : ℕ) (january_employees : ℕ) : 
  december_employees = 490 → 
  december_employees = january_employees + (january_employees * 15 / 100) →
  january_employees = 426 := by
  sorry

end company_employees_l2391_239108


namespace root_cubic_value_l2391_239165

theorem root_cubic_value (m : ℝ) (h : m^2 + m - 1 = 0) : m^3 + 2*m^2 - 7 = -6 := by
  sorry

end root_cubic_value_l2391_239165


namespace inequality_system_solution_set_l2391_239110

theorem inequality_system_solution_set :
  let S := {x : ℝ | 2 * x - 1 ≥ x + 1 ∧ x + 8 ≤ 4 * x - 1}
  S = {x : ℝ | x ≥ 3} := by
  sorry

end inequality_system_solution_set_l2391_239110


namespace smallest_upper_bound_l2391_239109

def S : Set (ℕ+ → ℝ) :=
  {f | f 1 = 2 ∧ ∀ n : ℕ+, f (n + 1) ≥ f n ∧ f n ≥ (n : ℝ) / (n + 1 : ℝ) * f (2 * n)}

theorem smallest_upper_bound :
  ∃ M : ℕ+, (∀ f ∈ S, ∀ n : ℕ+, f n < M) ∧
  (∀ M' : ℕ+, M' < M → ∃ f ∈ S, ∃ n : ℕ+, f n ≥ M') :=
by sorry

end smallest_upper_bound_l2391_239109


namespace f_minimum_l2391_239187

/-- The polynomial f(x) = x^2 + 6x + 10 -/
def f (x : ℝ) : ℝ := x^2 + 6*x + 10

/-- The point where f(x) reaches its minimum -/
def min_point : ℝ := -3

theorem f_minimum :
  ∀ x : ℝ, f x ≥ f min_point := by sorry

end f_minimum_l2391_239187


namespace shopping_money_l2391_239148

theorem shopping_money (initial_amount : ℝ) : 
  0.7 * initial_amount = 3500 → initial_amount = 5000 := by
  sorry

end shopping_money_l2391_239148


namespace monkey_reaches_top_monkey_reaches_top_in_19_minutes_l2391_239147

def pole_height : ℕ := 10
def ascend_distance : ℕ := 2
def slip_distance : ℕ := 1

def monkey_position (minutes : ℕ) : ℕ :=
  let full_cycles := minutes / 2
  let remainder := minutes % 2
  if remainder = 0 then
    full_cycles * (ascend_distance - slip_distance)
  else
    full_cycles * (ascend_distance - slip_distance) + ascend_distance

theorem monkey_reaches_top :
  ∃ (minutes : ℕ), monkey_position minutes ≥ pole_height ∧
                   ∀ (m : ℕ), m < minutes → monkey_position m < pole_height :=
by
  -- The proof would go here
  sorry

theorem monkey_reaches_top_in_19_minutes :
  monkey_position 19 ≥ pole_height ∧
  ∀ (m : ℕ), m < 19 → monkey_position m < pole_height :=
by
  -- The proof would go here
  sorry

end monkey_reaches_top_monkey_reaches_top_in_19_minutes_l2391_239147


namespace B_power_150_is_identity_l2391_239174

def B : Matrix (Fin 3) (Fin 3) ℕ :=
  !![0, 1, 0;
     0, 0, 1;
     1, 0, 0]

theorem B_power_150_is_identity :
  B^150 = (1 : Matrix (Fin 3) (Fin 3) ℕ) := by
  sorry

end B_power_150_is_identity_l2391_239174


namespace sum_of_products_l2391_239103

theorem sum_of_products (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x^2 + x*y + y^2 = 48)
  (eq2 : y^2 + y*z + z^2 = 9)
  (eq3 : z^2 + x*z + x^2 = 57) :
  x*y + y*z + x*z = 24 := by
sorry

end sum_of_products_l2391_239103


namespace odd_function_zeros_and_equation_root_l2391_239169

def isOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_zeros_and_equation_root (f : ℝ → ℝ) (zeros : Finset ℝ) :
  isOdd f →
  zeros.card = 2017 →
  (∀ x ∈ zeros, f x = 0) →
  ∃ r ∈ Set.Ioo 0 1, 2^r + r - 2 = 0 :=
sorry

end odd_function_zeros_and_equation_root_l2391_239169


namespace stating_arithmetic_sequence_length_is_twelve_l2391_239121

/-- 
The number of terms in an arithmetic sequence with 
first term 3, last term 69, and common difference 6
-/
def arithmetic_sequence_length : ℕ := 
  (69 - 3) / 6 + 1

/-- 
Theorem stating that the arithmetic sequence length is 12
-/
theorem arithmetic_sequence_length_is_twelve : 
  arithmetic_sequence_length = 12 := by sorry

end stating_arithmetic_sequence_length_is_twelve_l2391_239121


namespace one_minus_repeating_third_l2391_239112

/-- The definition of a repeating decimal 0.3333... -/
def repeating_third : ℚ := 1/3

/-- Proof that 1 - 0.3333... = 2/3 -/
theorem one_minus_repeating_third :
  1 - repeating_third = 2/3 := by
  sorry

end one_minus_repeating_third_l2391_239112


namespace arithmetic_calculation_l2391_239154

theorem arithmetic_calculation : 12 - 10 + 15 / 5 * 8 + 7 - 6 * 4 + 3 - 2 = 10 := by
  sorry

end arithmetic_calculation_l2391_239154


namespace least_integer_square_72_more_than_double_l2391_239189

theorem least_integer_square_72_more_than_double :
  ∃ x : ℤ, x^2 = 2*x + 72 ∧ ∀ y : ℤ, y^2 = 2*y + 72 → x ≤ y :=
by sorry

end least_integer_square_72_more_than_double_l2391_239189


namespace millie_bracelets_l2391_239163

theorem millie_bracelets (initial : ℕ) (lost : ℕ) (remaining : ℕ) 
  (h1 : lost = 2) 
  (h2 : remaining = 7) 
  (h3 : initial = remaining + lost) : initial = 9 := by
  sorry

end millie_bracelets_l2391_239163


namespace traveler_distance_l2391_239116

/-- The straight-line distance from start to end point given net northward and westward distances -/
theorem traveler_distance (north west : ℝ) (h_north : north = 12) (h_west : west = 12) :
  Real.sqrt (north ^ 2 + west ^ 2) = 12 * Real.sqrt 2 := by
  sorry

end traveler_distance_l2391_239116


namespace increasing_interval_of_f_l2391_239129

/-- Given two functions f and g with identical symmetry axes, 
    prove that [0, π/8] is an increasing interval of f on [0, π] -/
theorem increasing_interval_of_f (ω : ℝ) (h_ω : ω > 0) : 
  let f : ℝ → ℝ := λ x ↦ 2 * Real.sin (ω * x + π / 4)
  let g : ℝ → ℝ := λ x ↦ 2 * Real.cos (2 * x - π / 4)
  (∀ x y : ℝ, f x = f y ↔ g x = g y) →  -- Symmetry axes are identical
  (∀ x y : ℝ, x ∈ Set.Icc 0 (π / 8) → y ∈ Set.Icc 0 (π / 8) → x < y → f x < f y) ∧
  Set.Icc 0 (π / 8) ⊆ Set.Icc 0 π :=
by sorry

end increasing_interval_of_f_l2391_239129


namespace cody_chocolate_boxes_cody_bought_seven_boxes_l2391_239120

theorem cody_chocolate_boxes : ℕ → Prop :=
  fun x =>
    -- x is the number of boxes of chocolate candy
    -- 3 is the number of boxes of caramel candy
    -- 8 is the number of pieces in each box
    -- 80 is the total number of pieces
    x * 8 + 3 * 8 = 80 →
    x = 7

-- The proof
theorem cody_bought_seven_boxes : cody_chocolate_boxes 7 := by
  sorry

end cody_chocolate_boxes_cody_bought_seven_boxes_l2391_239120


namespace enhanced_mindmaster_codes_l2391_239198

/-- The number of colors available in the enhanced Mindmaster game -/
def num_colors : ℕ := 7

/-- The number of slots in a secret code -/
def num_slots : ℕ := 5

/-- The number of possible secret codes in the enhanced Mindmaster game -/
def num_codes : ℕ := num_colors ^ num_slots

/-- Theorem stating that the number of possible secret codes is 16807 -/
theorem enhanced_mindmaster_codes :
  num_codes = 16807 := by sorry

end enhanced_mindmaster_codes_l2391_239198


namespace pqu_theorem_l2391_239166

/-- A structure representing the relationship between P, Q, and U -/
structure PQU where
  P : ℝ
  Q : ℝ
  U : ℝ
  k : ℝ
  h : P = k * Q / U

/-- The theorem statement -/
theorem pqu_theorem (x y : PQU) (h1 : x.P = 6) (h2 : x.U = 4) (h3 : x.Q = 8)
                    (h4 : y.P = 18) (h5 : y.U = 9) : y.Q = 54 := by
  sorry

end pqu_theorem_l2391_239166


namespace perfect_square_condition_l2391_239183

theorem perfect_square_condition (n : ℕ) : 
  (∃ k : ℕ, n * 2^(n + 1) + 1 = k^2) ↔ n = 0 ∨ n = 3 :=
by sorry

end perfect_square_condition_l2391_239183


namespace circle_equation_l2391_239149

/-- The standard equation of a circle with center (0, -2) and radius 4 is x^2 + (y+2)^2 = 16 -/
theorem circle_equation (x y : ℝ) : 
  let center : ℝ × ℝ := (0, -2)
  let radius : ℝ := 4
  (x - center.1)^2 + (y - center.2)^2 = radius^2 ↔ x^2 + (y + 2)^2 = 16 :=
by sorry

end circle_equation_l2391_239149


namespace f_zero_equals_three_l2391_239145

-- Define the function f
noncomputable def f (t : ℝ) : ℝ :=
  let x := (t + 1) / 2
  (1 - x^2) / x^2

-- Theorem statement
theorem f_zero_equals_three :
  f 0 = 3 :=
by sorry

end f_zero_equals_three_l2391_239145


namespace max_product_sum_l2391_239136

theorem max_product_sum (A M C : ℕ) (h : A + M + C = 15) :
  (∀ a m c : ℕ, a + m + c = 15 → A * M * C + A * M + M * C + C * A ≥ a * m * c + a * m + m * c + c * a) ∧
  A * M * C + A * M + M * C + C * A = 200 := by
sorry

end max_product_sum_l2391_239136


namespace min_value_of_max_function_l2391_239142

theorem min_value_of_max_function (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x > 2*y) :
  ∃ (t : ℝ), t = max (x^2/2) (4/(y*(x-2*y))) ∧ t ≥ 4 ∧ 
  (∃ (x0 y0 : ℝ), x0 > 0 ∧ y0 > 0 ∧ x0 > 2*y0 ∧ 
    max (x0^2/2) (4/(y0*(x0-2*y0))) = 4) :=
by sorry

end min_value_of_max_function_l2391_239142


namespace trigonometric_identity_l2391_239164

theorem trigonometric_identity : 
  Real.tan (70 * π / 180) * Real.cos (10 * π / 180) * (Real.sqrt 3 * Real.tan (20 * π / 180) - 1) = -1 := by
  sorry

end trigonometric_identity_l2391_239164


namespace quadratic_inequality_solution_l2391_239124

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x : ℝ, -2 < x ∧ x < 3 → a * x^2 + x + b > 0) ∧
  (∀ x : ℝ, (x ≤ -2 ∨ x ≥ 3) → a * x^2 + x + b ≤ 0) →
  a + b = 5 := by sorry

end quadratic_inequality_solution_l2391_239124


namespace rectangular_box_diagonals_l2391_239179

/-- A rectangular box with given surface area and edge length sum has a specific sum of interior diagonal lengths. -/
theorem rectangular_box_diagonals 
  (a b c : ℝ) 
  (h_surface_area : 2 * (a * b + b * c + c * a) = 206)
  (h_edge_sum : 4 * (a + b + c) = 64) :
  4 * Real.sqrt (a^2 + b^2 + c^2) = 20 * Real.sqrt 2 := by
  sorry

end rectangular_box_diagonals_l2391_239179


namespace fraction_multiplication_l2391_239180

theorem fraction_multiplication (x : ℝ) : 
  (1 : ℝ) / 3 * 2 / 7 * 9 / 13 * x / 17 = 18 * x / 4911 := by
  sorry

end fraction_multiplication_l2391_239180


namespace fourth_month_sales_l2391_239159

def sales_month1 : ℕ := 6535
def sales_month2 : ℕ := 6927
def sales_month3 : ℕ := 6855
def sales_month5 : ℕ := 6562
def sales_month6 : ℕ := 4891
def required_average : ℕ := 6500
def num_months : ℕ := 6

theorem fourth_month_sales :
  ∃ (sales_month4 : ℕ),
    (sales_month1 + sales_month2 + sales_month3 + sales_month4 + sales_month5 + sales_month6) / num_months = required_average ∧
    sales_month4 = 7230 := by
  sorry

end fourth_month_sales_l2391_239159


namespace number_of_divisors_5005_l2391_239102

theorem number_of_divisors_5005 : Nat.card (Nat.divisors 5005) = 16 := by
  sorry

end number_of_divisors_5005_l2391_239102


namespace expression_simplification_l2391_239131

theorem expression_simplification :
  (5 + 7) * (5^2 + 7^2) * (5^4 + 7^4) * (5^8 + 7^8) * (5^16 + 7^16) = 5^32 + 7^32 := by
  sorry

end expression_simplification_l2391_239131


namespace beef_weight_loss_percentage_l2391_239123

/-- Calculates the percentage of weight loss during beef processing. -/
theorem beef_weight_loss_percentage 
  (weight_before : ℝ) 
  (weight_after : ℝ) 
  (h1 : weight_before = 876.9230769230769) 
  (h2 : weight_after = 570) : 
  (weight_before - weight_after) / weight_before * 100 = 35 := by
  sorry

end beef_weight_loss_percentage_l2391_239123


namespace marcus_earnings_l2391_239194

/-- Represents Marcus's work and earnings over two weeks -/
structure MarcusWork where
  hourly_wage : ℝ
  hours_week1 : ℝ
  hours_week2 : ℝ
  earnings_difference : ℝ

/-- Calculates the total earnings for two weeks given Marcus's work data -/
def total_earnings (w : MarcusWork) : ℝ :=
  w.hourly_wage * (w.hours_week1 + w.hours_week2)

theorem marcus_earnings :
  ∀ w : MarcusWork,
  w.hours_week1 = 12 ∧
  w.hours_week2 = 18 ∧
  w.earnings_difference = 36 ∧
  w.hourly_wage * (w.hours_week2 - w.hours_week1) = w.earnings_difference →
  total_earnings w = 180 :=
by
  sorry

end marcus_earnings_l2391_239194


namespace largest_multiple_of_seven_less_than_negative_fifty_l2391_239100

theorem largest_multiple_of_seven_less_than_negative_fifty :
  ∀ n : ℤ, 7 ∣ n ∧ n < -50 → n ≤ -56 :=
by
  sorry

end largest_multiple_of_seven_less_than_negative_fifty_l2391_239100


namespace parabola_tangent_property_l2391_239143

/-- Parabola type -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Circle in 2D space -/
structure Circle where
  center : Point
  radius : ℝ

/-- Given a parabola Γ: y² = 2px (p > 0) with focus F, and a point Q outside Γ (not on the x-axis),
    let tangents QA and QB intersect Γ at A and B respectively, and the y-axis at C and D.
    If M is the circumcenter of triangle QAB, then FM is tangent to the circumcircle of triangle FCD. -/
theorem parabola_tangent_property (Γ : Parabola) (F Q A B C D M : Point) :
  Q.x ≠ 0 →  -- Q is not on y-axis
  Q.y ≠ 0 →  -- Q is not on x-axis
  (∃ t : ℝ, A = Point.mk (t^2 / (2 * Γ.p)) t) →  -- A is on the parabola
  (∃ t : ℝ, B = Point.mk (t^2 / (2 * Γ.p)) t) →  -- B is on the parabola
  F = Point.mk (Γ.p / 2) 0 →  -- F is the focus
  C = Point.mk 0 C.y →  -- C is on y-axis
  D = Point.mk 0 D.y →  -- D is on y-axis
  (∃ l : Line, l.a * Q.x + l.b * Q.y + l.c = 0 ∧ l.a * A.x + l.b * A.y + l.c = 0) →  -- QA is a line
  (∃ l : Line, l.a * Q.x + l.b * Q.y + l.c = 0 ∧ l.a * B.x + l.b * B.y + l.c = 0) →  -- QB is a line
  (∀ P : Point, (P.x - M.x)^2 + (P.y - M.y)^2 = (A.x - M.x)^2 + (A.y - M.y)^2 →
               (P.x - M.x)^2 + (P.y - M.y)^2 = (B.x - M.x)^2 + (B.y - M.y)^2 →
               (P.x - M.x)^2 + (P.y - M.y)^2 = (Q.x - M.x)^2 + (Q.y - M.y)^2) →  -- M is circumcenter of QAB
  (∃ T : Point, ∃ r : ℝ,
    (T.x - F.x)^2 + (T.y - F.y)^2 = (C.x - F.x)^2 + (C.y - F.y)^2 ∧
    (T.x - F.x)^2 + (T.y - F.y)^2 = (D.x - F.x)^2 + (D.y - F.y)^2 ∧
    (M.x - F.x) * (T.x - F.x) + (M.y - F.y) * (T.y - F.y) = r^2) →  -- FM is tangent to circumcircle of FCD
  True :=
sorry

end parabola_tangent_property_l2391_239143


namespace special_function_inequality_l2391_239195

/-- A non-negative differentiable function satisfying certain conditions -/
structure SpecialFunction where
  f : ℝ → ℝ
  domain : Set ℝ := Set.Ioi 0
  non_negative : ∀ x ∈ domain, f x ≥ 0
  differentiable : DifferentiableOn ℝ f domain
  condition : ∀ x ∈ domain, x * (deriv f x) + f x ≤ 0

/-- Theorem statement -/
theorem special_function_inequality (φ : SpecialFunction) (a b : ℝ) 
    (ha : a > 0) (hb : b > 0) (hab : a < b) :
    b * φ.f b ≤ a * φ.f a := by
  sorry

end special_function_inequality_l2391_239195


namespace two_number_problem_l2391_239133

theorem two_number_problem (x y : ℝ) 
  (h1 : 0.35 * x = 0.50 * x - 24)
  (h2 : 0.30 * y = 0.55 * x - 36) : 
  x = 160 ∧ y = 520/3 := by
sorry

end two_number_problem_l2391_239133


namespace hiking_equipment_cost_l2391_239152

/-- Calculates the total cost of hiking equipment --/
theorem hiking_equipment_cost (hoodie_cost : ℚ) (boot_cost : ℚ) (flashlight_percentage : ℚ) (discount_percentage : ℚ) : 
  hoodie_cost = 80 →
  flashlight_percentage = 20 / 100 →
  boot_cost = 110 →
  discount_percentage = 10 / 100 →
  hoodie_cost + (flashlight_percentage * hoodie_cost) + (boot_cost - discount_percentage * boot_cost) = 195 := by
  sorry

end hiking_equipment_cost_l2391_239152


namespace power_equation_solution_l2391_239130

theorem power_equation_solution (N : ℕ) : (4^5)^2 * (2^5)^4 = 2^N → N = 30 := by
  sorry

end power_equation_solution_l2391_239130


namespace two_power_ten_minus_one_factors_l2391_239107

theorem two_power_ten_minus_one_factors : 
  ∃ (p q r : Nat), 
    Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧
    p ≠ q ∧ p ≠ r ∧ q ≠ r ∧
    (2^10 - 1 : Nat) = p * q * r ∧
    p + q + r = 45 :=
by
  sorry

end two_power_ten_minus_one_factors_l2391_239107

import Mathlib

namespace NUMINAMATH_CALUDE_first_year_after_2010_sum_15_is_correct_l3172_317271

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

/-- Check if a year is after 2010 -/
def is_after_2010 (year : ℕ) : Prop :=
  year > 2010

/-- First year after 2010 with sum of digits equal to 15 -/
def first_year_after_2010_sum_15 : ℕ :=
  2039

theorem first_year_after_2010_sum_15_is_correct :
  (is_after_2010 first_year_after_2010_sum_15) ∧ 
  (sum_of_digits first_year_after_2010_sum_15 = 15) ∧
  (∀ y : ℕ, is_after_2010 y ∧ y < first_year_after_2010_sum_15 → sum_of_digits y ≠ 15) :=
by sorry

end NUMINAMATH_CALUDE_first_year_after_2010_sum_15_is_correct_l3172_317271


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3172_317282

theorem quadratic_equation_solution :
  let x₁ : ℝ := (2 + Real.sqrt 3) / 2
  let x₂ : ℝ := (2 - Real.sqrt 3) / 2
  4 * x₁^2 - 8 * x₁ + 1 = 0 ∧ 4 * x₂^2 - 8 * x₂ + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3172_317282


namespace NUMINAMATH_CALUDE_k_is_negative_l3172_317255

/-- A linear function y = x + k passes through a quadrant if there exists a point (x, y) in that quadrant satisfying the equation. -/
def passes_through_quadrant (k : ℝ) (quadrant : ℕ) : Prop :=
  match quadrant with
  | 1 => ∃ x > 0, x + k > 0
  | 3 => ∃ x < 0, x + k < 0
  | 4 => ∃ x > 0, x + k < 0
  | _ => False

/-- If the graph of y = x + k passes through the first, third, and fourth quadrants, then k < 0. -/
theorem k_is_negative (k : ℝ) 
  (h1 : passes_through_quadrant k 1)
  (h3 : passes_through_quadrant k 3)
  (h4 : passes_through_quadrant k 4) : 
  k < 0 := by
  sorry


end NUMINAMATH_CALUDE_k_is_negative_l3172_317255


namespace NUMINAMATH_CALUDE_circle_center_from_equation_l3172_317241

/-- A circle in the 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The equation of a circle in standard form --/
def CircleEquation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

theorem circle_center_from_equation :
  ∃ (c : Circle), (∀ x y : ℝ, CircleEquation c x y ↔ (x - 1)^2 + (y - 2)^2 = 5) ∧ c.center = (1, 2) :=
sorry

end NUMINAMATH_CALUDE_circle_center_from_equation_l3172_317241


namespace NUMINAMATH_CALUDE_perpendicular_line_exists_l3172_317214

-- Define the concept of a line
def Line : Type := sorry

-- Define the concept of a plane
def Plane : Type := sorry

-- Define what it means for a line to be within a plane
def within_plane (l : Line) (p : Plane) : Prop := sorry

-- Define what it means for two lines to be perpendicular
def perpendicular (l1 l2 : Line) : Prop := sorry

-- State the theorem
theorem perpendicular_line_exists (l : Line) (α : Plane) :
  ∃ m : Line, within_plane m α ∧ perpendicular m l := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_exists_l3172_317214


namespace NUMINAMATH_CALUDE_math_club_exclusive_members_l3172_317291

theorem math_club_exclusive_members :
  ∀ (total_students : ℕ) (both_clubs : ℕ) (math_club : ℕ) (science_club : ℕ),
    total_students = 30 →
    both_clubs = 2 →
    math_club = 3 * science_club →
    total_students = math_club + science_club - both_clubs →
    math_club - both_clubs = 20 :=
by sorry

end NUMINAMATH_CALUDE_math_club_exclusive_members_l3172_317291


namespace NUMINAMATH_CALUDE_extra_crayons_l3172_317268

theorem extra_crayons (num_packs : ℕ) (crayons_per_pack : ℕ) (total_crayons : ℕ) : 
  num_packs = 4 →
  crayons_per_pack = 10 →
  total_crayons = 40 →
  total_crayons - (num_packs * crayons_per_pack) = 0 := by
  sorry

end NUMINAMATH_CALUDE_extra_crayons_l3172_317268


namespace NUMINAMATH_CALUDE_digit_sum_equation_l3172_317263

-- Define the digits as natural numbers
def X : ℕ := sorry
def Y : ℕ := sorry
def M : ℕ := sorry
def Z : ℕ := sorry
def F : ℕ := sorry

-- Define the two-digit numbers
def XY : ℕ := 10 * X + Y
def MZ : ℕ := 10 * M + Z

-- Define the three-digit number FFF
def FFF : ℕ := 100 * F + 10 * F + F

-- Theorem statement
theorem digit_sum_equation : 
  (X ≠ 0) ∧ (Y ≠ 0) ∧ (M ≠ 0) ∧ (Z ≠ 0) ∧ (F ≠ 0) ∧  -- non-zero digits
  (X ≠ Y) ∧ (X ≠ M) ∧ (X ≠ Z) ∧ (X ≠ F) ∧
  (Y ≠ M) ∧ (Y ≠ Z) ∧ (Y ≠ F) ∧
  (M ≠ Z) ∧ (M ≠ F) ∧
  (Z ≠ F) ∧  -- unique digits
  (X < 10) ∧ (Y < 10) ∧ (M < 10) ∧ (Z < 10) ∧ (F < 10) ∧  -- single digits
  (XY * MZ = FFF) →  -- equation condition
  X + Y + M + Z + F = 28 := by
sorry

end NUMINAMATH_CALUDE_digit_sum_equation_l3172_317263


namespace NUMINAMATH_CALUDE_variance_of_scores_l3172_317254

def scores : List ℝ := [9, 10, 9, 7, 10]

theorem variance_of_scores : 
  let n : ℕ := scores.length
  let mean : ℝ := (scores.sum) / n
  let variance : ℝ := (scores.map (λ x => (x - mean)^2)).sum / n
  variance = 6/5 := by sorry

end NUMINAMATH_CALUDE_variance_of_scores_l3172_317254


namespace NUMINAMATH_CALUDE_xy_plus_x_plus_y_odd_l3172_317212

def S : Set ℕ := {1, 3, 5, 7, 9, 11, 13, 15, 17, 19}

theorem xy_plus_x_plus_y_odd (x y : ℕ) (hx : x ∈ S) (hy : y ∈ S) (hxy : x ≠ y) :
  ¬Even (x * y + x + y) :=
by sorry

end NUMINAMATH_CALUDE_xy_plus_x_plus_y_odd_l3172_317212


namespace NUMINAMATH_CALUDE_square_difference_fourth_power_l3172_317273

theorem square_difference_fourth_power : (7^2 - 3^2)^4 = 2560000 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_fourth_power_l3172_317273


namespace NUMINAMATH_CALUDE_largest_y_satisfies_equation_forms_triangle_largest_y_forms_triangle_l3172_317253

def largest_y : ℝ := 23

theorem largest_y_satisfies_equation :
  |largest_y - 8| = 15 ∧
  ∀ y : ℝ, |y - 8| = 15 → y ≤ largest_y :=
sorry

theorem forms_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem largest_y_forms_triangle :
  forms_triangle largest_y 20 9 :=
sorry

end NUMINAMATH_CALUDE_largest_y_satisfies_equation_forms_triangle_largest_y_forms_triangle_l3172_317253


namespace NUMINAMATH_CALUDE_only_solutions_are_72_and_88_l3172_317222

/-- The product of digits of a positive integer -/
def product_of_digits (k : ℕ+) : ℕ :=
  sorry

/-- The main theorem stating that 72 and 88 are the only solutions -/
theorem only_solutions_are_72_and_88 :
  ∀ k : ℕ+, (product_of_digits k = (25 * k : ℚ) / 8 - 211) ↔ (k = 72 ∨ k = 88) :=
by sorry

end NUMINAMATH_CALUDE_only_solutions_are_72_and_88_l3172_317222


namespace NUMINAMATH_CALUDE_race_time_calculation_l3172_317206

theorem race_time_calculation (prejean_speed rickey_speed rickey_time prejean_time : ℝ) : 
  prejean_speed = (3 / 4) * rickey_speed →
  rickey_time + prejean_time = 70 →
  prejean_time = (4 / 3) * rickey_time →
  rickey_time = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_race_time_calculation_l3172_317206


namespace NUMINAMATH_CALUDE_different_color_chip_probability_l3172_317280

theorem different_color_chip_probability :
  let total_chips : ℕ := 12
  let red_chips : ℕ := 7
  let green_chips : ℕ := 5
  let prob_red : ℚ := red_chips / total_chips
  let prob_green : ℚ := green_chips / total_chips
  let prob_different_colors : ℚ := prob_red * prob_green + prob_green * prob_red
  prob_different_colors = 35 / 72 := by sorry

end NUMINAMATH_CALUDE_different_color_chip_probability_l3172_317280


namespace NUMINAMATH_CALUDE_max_list_length_l3172_317238

def is_valid_list (D : List Nat) : Prop :=
  ∀ x ∈ D, 1 ≤ x ∧ x ≤ 10

def count_occurrences (x : Nat) (L : List Nat) : Nat :=
  L.filter (· = x) |>.length

def generate_M (D : List Nat) : List Nat :=
  D.map (λ x => count_occurrences x D)

theorem max_list_length :
  ∃ (D : List Nat),
    is_valid_list D ∧
    D.length = 10 ∧
    generate_M D = D.reverse ∧
    ∀ (D' : List Nat),
      is_valid_list D' →
      generate_M D' = D'.reverse →
      D'.length ≤ 10 :=
by sorry

end NUMINAMATH_CALUDE_max_list_length_l3172_317238


namespace NUMINAMATH_CALUDE_housing_development_l3172_317294

theorem housing_development (total : ℕ) (garage : ℕ) (pool : ℕ) (neither : ℕ) 
  (h_total : total = 90)
  (h_garage : garage = 50)
  (h_pool : pool = 40)
  (h_neither : neither = 35) :
  garage + pool - (total - neither) = 35 := by
  sorry

end NUMINAMATH_CALUDE_housing_development_l3172_317294


namespace NUMINAMATH_CALUDE_teacher_grading_problem_l3172_317224

def remaining_problems (total_worksheets : ℕ) (problems_per_worksheet : ℕ) (graded_worksheets : ℕ) : ℕ :=
  (total_worksheets - graded_worksheets) * problems_per_worksheet

theorem teacher_grading_problem :
  remaining_problems 14 2 7 = 14 := by
  sorry

end NUMINAMATH_CALUDE_teacher_grading_problem_l3172_317224


namespace NUMINAMATH_CALUDE_carpet_width_in_cm_l3172_317287

/-- Proves that the width of the carpet is 1000 centimeters given the room dimensions and carpeting costs. -/
theorem carpet_width_in_cm (room_length room_breadth carpet_cost_per_meter total_cost : ℝ) 
  (h1 : room_length = 18)
  (h2 : room_breadth = 7.5)
  (h3 : carpet_cost_per_meter = 4.5)
  (h4 : total_cost = 810) : 
  (total_cost / carpet_cost_per_meter) / room_length * 100 = 1000 := by
  sorry

#check carpet_width_in_cm

end NUMINAMATH_CALUDE_carpet_width_in_cm_l3172_317287


namespace NUMINAMATH_CALUDE_circle_with_rational_center_multiple_lattice_points_l3172_317270

/-- A point in the 2D plane with rational coordinates -/
structure RationalPoint where
  x : ℚ
  y : ℚ

/-- A circle in the 2D plane -/
structure Circle where
  center : RationalPoint
  radius : ℝ

/-- A lattice point in the 2D plane -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- Check if a lattice point is on the circumference of a circle -/
def isOnCircumference (c : Circle) (p : LatticePoint) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

/-- The main theorem statement -/
theorem circle_with_rational_center_multiple_lattice_points
  (K : RationalPoint) (c : Circle) (p : LatticePoint) :
  c.center = K → isOnCircumference c p →
  ∃ q : LatticePoint, q ≠ p ∧ isOnCircumference c q :=
by sorry

end NUMINAMATH_CALUDE_circle_with_rational_center_multiple_lattice_points_l3172_317270


namespace NUMINAMATH_CALUDE_abs_sum_greater_than_one_necessary_not_sufficient_l3172_317205

theorem abs_sum_greater_than_one_necessary_not_sufficient (a b : ℝ) :
  (∀ b, b < -1 → ∀ a, |a| + |b| > 1) ∧
  (∃ a b, |a| + |b| > 1 ∧ b ≥ -1) := by
  sorry

end NUMINAMATH_CALUDE_abs_sum_greater_than_one_necessary_not_sufficient_l3172_317205


namespace NUMINAMATH_CALUDE_digit_count_theorem_l3172_317251

/-- The number of n-digit positive integers -/
def nDigitNumbers (n : ℕ) : ℕ := 9 * 10^(n-1)

/-- The total number of digits needed to write all natural numbers from 1 to 10^n (not including 10^n) -/
def totalDigits (n : ℕ) : ℚ := n * 10^n - (10^n - 1) / 9

theorem digit_count_theorem (n : ℕ) (h : n > 0) :
  (∀ k : ℕ, k > 0 → k ≤ n → nDigitNumbers k = 9 * 10^(k-1)) ∧
  totalDigits n = n * 10^n - (10^n - 1) / 9 :=
sorry

end NUMINAMATH_CALUDE_digit_count_theorem_l3172_317251


namespace NUMINAMATH_CALUDE_hyperbola_center_l3172_317292

/-- The center of a hyperbola with foci at (3, 6) and (11, 10) is at (7, 8) -/
theorem hyperbola_center (f1 f2 : ℝ × ℝ) (h1 : f1 = (3, 6)) (h2 : f2 = (11, 10)) :
  let center := ((f1.1 + f2.1) / 2, (f1.2 + f2.2) / 2)
  center = (7, 8) := by sorry

end NUMINAMATH_CALUDE_hyperbola_center_l3172_317292


namespace NUMINAMATH_CALUDE_equation_proof_l3172_317261

theorem equation_proof : ((12 : ℝ)^2 * (6 : ℝ)^4 / 432)^(1/2) = 4 * 3 * (3 : ℝ)^(1/2) := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l3172_317261


namespace NUMINAMATH_CALUDE_x_plus_y_equals_two_l3172_317299

theorem x_plus_y_equals_two (x y : ℝ) 
  (hx : (x - 1)^2017 + 2013 * (x - 1) = -1)
  (hy : (y - 1)^2017 + 2013 * (y - 1) = 1) : 
  x + y = 2 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_equals_two_l3172_317299


namespace NUMINAMATH_CALUDE_max_k_value_l3172_317249

theorem max_k_value (x y k : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_k : k > 0)
  (h_eq : 3 = k^2 * (x^2 / y^2 + y^2 / x^2) + k * (x / y + y / x)) :
  k ≤ (-1 + Real.sqrt 7) / 2 :=
sorry

end NUMINAMATH_CALUDE_max_k_value_l3172_317249


namespace NUMINAMATH_CALUDE_apple_boxes_weights_l3172_317275

theorem apple_boxes_weights (a b c d : ℝ) 
  (h1 : a + b + c = 70)
  (h2 : a + b + d = 80)
  (h3 : a + c + d = 73)
  (h4 : b + c + d = 77)
  (ha : a > 0)
  (hb : b > 0)
  (hc : c > 0)
  (hd : d > 0) :
  a = 23 ∧ b = 27 ∧ c = 20 ∧ d = 30 := by
sorry

end NUMINAMATH_CALUDE_apple_boxes_weights_l3172_317275


namespace NUMINAMATH_CALUDE_square_root_of_25_l3172_317202

theorem square_root_of_25 : 
  {x : ℝ | x^2 = 25} = {5, -5} := by sorry

end NUMINAMATH_CALUDE_square_root_of_25_l3172_317202


namespace NUMINAMATH_CALUDE_purely_imaginary_z_l3172_317267

theorem purely_imaginary_z (b : ℝ) :
  let z : ℂ := Complex.I * (1 + b * Complex.I) + 2 + 3 * b * Complex.I
  (z.re = 0) → z = 7 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_z_l3172_317267


namespace NUMINAMATH_CALUDE_balls_without_holes_count_l3172_317258

/-- The number of soccer balls Matthias has -/
def total_soccer_balls : ℕ := 40

/-- The number of basketballs Matthias has -/
def total_basketballs : ℕ := 15

/-- The number of soccer balls with holes -/
def soccer_balls_with_holes : ℕ := 30

/-- The number of basketballs with holes -/
def basketballs_with_holes : ℕ := 7

/-- The total number of balls without holes -/
def total_balls_without_holes : ℕ := 
  (total_soccer_balls - soccer_balls_with_holes) + (total_basketballs - basketballs_with_holes)

theorem balls_without_holes_count : total_balls_without_holes = 18 := by
  sorry

end NUMINAMATH_CALUDE_balls_without_holes_count_l3172_317258


namespace NUMINAMATH_CALUDE_second_offer_more_advantageous_l3172_317215

/-- Represents the total cost of four items -/
def S : ℕ := 1000

/-- Represents the minimum cost of any item -/
def X : ℕ := 99

/-- Represents the prices of four items -/
structure Prices where
  s₁ : ℕ
  s₂ : ℕ
  s₃ : ℕ
  s₄ : ℕ
  sum_eq_S : s₁ + s₂ + s₃ + s₄ = S
  ordered : s₁ ≥ s₂ ∧ s₂ ≥ s₃ ∧ s₃ ≥ s₄
  min_price : s₄ ≥ X

/-- The maximum N for which the second offer is more advantageous -/
def maxN : ℕ := 504

theorem second_offer_more_advantageous (prices : Prices) :
  ∀ N : ℕ, N ≤ maxN →
  (0.2 * prices.s₁ + 0.8 * S : ℚ) < (S - prices.s₄ : ℚ) ∧
  ¬∃ M : ℕ, M > maxN ∧ (0.2 * prices.s₁ + 0.8 * S : ℚ) < (S - prices.s₄ : ℚ) :=
sorry

end NUMINAMATH_CALUDE_second_offer_more_advantageous_l3172_317215


namespace NUMINAMATH_CALUDE_grandsons_age_l3172_317276

theorem grandsons_age (grandson_age grandfather_age : ℕ) : 
  grandfather_age = 6 * grandson_age →
  grandfather_age + 4 + grandson_age + 4 = 78 →
  grandson_age = 10 := by
sorry

end NUMINAMATH_CALUDE_grandsons_age_l3172_317276


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l3172_317296

/-- The function f(x) = -x³ + 2ax -/
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + 2*a*x

/-- f is monotonically decreasing on (-∞, 1] -/
def is_monotone_decreasing_on_interval (a : ℝ) : Prop :=
  ∀ x y, x ≤ y → y ≤ 1 → f a x ≥ f a y

theorem necessary_but_not_sufficient :
  (∀ a : ℝ, is_monotone_decreasing_on_interval a → a < 3/2) ∧
  (∃ a : ℝ, a < 3/2 ∧ ¬is_monotone_decreasing_on_interval a) :=
sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l3172_317296


namespace NUMINAMATH_CALUDE_divisibility_of_x_l3172_317279

theorem divisibility_of_x (x y : ℕ+) (h1 : 2 * x ^ 2 - 1 = y ^ 15) (h2 : x > 1) :
  5 ∣ x.val := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_x_l3172_317279


namespace NUMINAMATH_CALUDE_amp_fifteen_amp_l3172_317203

-- Define the ampersand operations
def amp_right (x : ℝ) : ℝ := 8 - x
def amp_left (x : ℝ) : ℝ := x - 9

-- State the theorem
theorem amp_fifteen_amp : amp_left (amp_right 15) = -16 := by
  sorry

end NUMINAMATH_CALUDE_amp_fifteen_amp_l3172_317203


namespace NUMINAMATH_CALUDE_hexagon_angle_Q_l3172_317293

/-- A hexagon with specified interior angles -/
structure Hexagon :=
  (angleS : ℝ)
  (angleT : ℝ)
  (angleU : ℝ)
  (angleV : ℝ)
  (angleW : ℝ)
  (h_angleS : angleS = 120)
  (h_angleT : angleT = 130)
  (h_angleU : angleU = 140)
  (h_angleV : angleV = 100)
  (h_angleW : angleW = 85)

/-- The measure of angle Q in the hexagon -/
def angleQ (h : Hexagon) : ℝ := 720 - (h.angleS + h.angleT + h.angleU + h.angleV + h.angleW)

theorem hexagon_angle_Q (h : Hexagon) : angleQ h = 145 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_angle_Q_l3172_317293


namespace NUMINAMATH_CALUDE_max_participants_answering_A_l3172_317285

theorem max_participants_answering_A (total : ℕ) (a b c : ℕ) : 
  total = 39 →
  a + b + c + (a + 3*b - 5) + 3*b + (total - (2*a + 6*b - 5)) = total →
  a = b + c →
  2*(total - (2*a + 6*b - 5)) = 3*b →
  (2*a + 9*b = 44 ∧ a ≥ 0 ∧ b ≥ 0) →
  (∃ max_A : ℕ, max_A = 2*a + 3*b - 5 ∧ max_A ≤ 23 ∧ 
   ∀ other_A : ℕ, other_A = 2*a' + 3*b' - 5 → 
   (2*a' + 9*b' = 44 ∧ a' ≥ 0 ∧ b' ≥ 0) → other_A ≤ max_A) :=
by sorry

end NUMINAMATH_CALUDE_max_participants_answering_A_l3172_317285


namespace NUMINAMATH_CALUDE_most_probable_hits_l3172_317220

-- Define the parameters
def n : ℕ := 5
def p : ℝ := 0.6

-- Define the binomial probability mass function
def binomialPMF (k : ℕ) : ℝ :=
  (Nat.choose n k) * p^k * (1 - p)^(n - k)

-- Theorem statement
theorem most_probable_hits :
  ∃ (k : ℕ), k ≤ n ∧ 
  (∀ (j : ℕ), j ≤ n → binomialPMF j ≤ binomialPMF k) ∧
  k = 3 := by
  sorry

end NUMINAMATH_CALUDE_most_probable_hits_l3172_317220


namespace NUMINAMATH_CALUDE_variance_of_letters_l3172_317209

def letters : List ℕ := [10, 6, 8, 5, 6]

def mean (xs : List ℕ) : ℚ :=
  (xs.sum : ℚ) / xs.length

def variance (xs : List ℕ) : ℚ :=
  let m := mean xs
  (xs.map (fun x => ((x : ℚ) - m) ^ 2)).sum / xs.length

theorem variance_of_letters : variance letters = 16/5 := by
  sorry

end NUMINAMATH_CALUDE_variance_of_letters_l3172_317209


namespace NUMINAMATH_CALUDE_three_intersection_points_l3172_317274

-- Define the three lines
def line1 (x y : ℝ) : Prop := 4 * y - 3 * x = 2
def line2 (x y : ℝ) : Prop := x + 3 * y = 3
def line3 (x y : ℝ) : Prop := 8 * x - 12 * y = 9

-- Define an intersection point
def is_intersection (x y : ℝ) : Prop :=
  (line1 x y ∧ line2 x y) ∨ (line1 x y ∧ line3 x y) ∨ (line2 x y ∧ line3 x y)

-- Theorem statement
theorem three_intersection_points :
  ∃ (p1 p2 p3 : ℝ × ℝ),
    is_intersection p1.1 p1.2 ∧
    is_intersection p2.1 p2.2 ∧
    is_intersection p3.1 p3.2 ∧
    p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3 ∧
    ∀ (x y : ℝ), is_intersection x y → (x, y) = p1 ∨ (x, y) = p2 ∨ (x, y) = p3 :=
by
  sorry

end NUMINAMATH_CALUDE_three_intersection_points_l3172_317274


namespace NUMINAMATH_CALUDE_boat_upstream_distance_l3172_317233

/-- Represents the distance traveled by a boat in one hour -/
def boat_distance (still_speed : ℝ) (stream_speed : ℝ) : ℝ :=
  still_speed - stream_speed

/-- The boat's speed in still water (km/hr) -/
def still_speed : ℝ := 7

/-- The distance the boat travels along the stream in one hour (km) -/
def downstream_distance : ℝ := 9

/-- The stream speed (km/hr) -/
def stream_speed : ℝ := downstream_distance - still_speed

theorem boat_upstream_distance :
  boat_distance still_speed stream_speed = 5 := by
  sorry


end NUMINAMATH_CALUDE_boat_upstream_distance_l3172_317233


namespace NUMINAMATH_CALUDE_log_equation_l3172_317278

theorem log_equation : 2 * Real.log 10 / Real.log 5 + Real.log 0.25 / Real.log 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_l3172_317278


namespace NUMINAMATH_CALUDE_conic_is_ellipse_l3172_317272

/-- The equation of the conic section -/
def conic_equation (x y : ℝ) : Prop :=
  x^2 - 72*y^2 - 12*x + 144 = 0

/-- Definition of an ellipse in standard form -/
def is_ellipse (a b h k : ℝ) (x y : ℝ) : Prop :=
  ((x - h)^2 / a^2) + ((y - k)^2 / b^2) = 1

/-- Theorem stating that the given equation represents an ellipse -/
theorem conic_is_ellipse :
  ∃ a b h k : ℝ, a > 0 ∧ b > 0 ∧
  ∀ x y : ℝ, conic_equation x y ↔ is_ellipse a b h k x y :=
sorry

end NUMINAMATH_CALUDE_conic_is_ellipse_l3172_317272


namespace NUMINAMATH_CALUDE_worker_selection_probability_l3172_317231

theorem worker_selection_probability 
  (total_workers : ℕ) 
  (eliminated_workers : ℕ) 
  (remaining_workers : ℕ) 
  (representatives : ℕ) 
  (h1 : total_workers = 2009)
  (h2 : eliminated_workers = 9)
  (h3 : remaining_workers = 2000)
  (h4 : representatives = 100)
  (h5 : remaining_workers = total_workers - eliminated_workers) :
  (representatives : ℚ) / (total_workers : ℚ) = 100 / 2009 :=
by sorry

end NUMINAMATH_CALUDE_worker_selection_probability_l3172_317231


namespace NUMINAMATH_CALUDE_alcohol_dilution_l3172_317250

theorem alcohol_dilution (initial_volume : ℝ) (initial_percentage : ℝ) (added_water : ℝ) :
  initial_volume = 15 →
  initial_percentage = 26 →
  added_water = 5 →
  let alcohol_volume := initial_volume * (initial_percentage / 100)
  let total_volume := initial_volume + added_water
  let final_percentage := (alcohol_volume / total_volume) * 100
  final_percentage = 19.5 := by
    sorry

end NUMINAMATH_CALUDE_alcohol_dilution_l3172_317250


namespace NUMINAMATH_CALUDE_unique_solution_l3172_317257

/-- A function satisfying the given conditions -/
def SatisfiesConditions (f : ℝ → ℝ) : Prop :=
  (∀ x, f x = -f (-x)) ∧
  (∀ x, f (x + 1) = f x + 1) ∧
  (∀ x ≠ 0, f (1 / x) = (1 / x^2) * f x)

/-- Theorem stating that the only function satisfying the conditions is f(x) = x -/
theorem unique_solution (f : ℝ → ℝ) (h : SatisfiesConditions f) :
  ∀ x, f x = x := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l3172_317257


namespace NUMINAMATH_CALUDE_factor_2x_squared_minus_8_l3172_317229

theorem factor_2x_squared_minus_8 (x : ℝ) : 2 * x^2 - 8 = 2 * (x + 2) * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_factor_2x_squared_minus_8_l3172_317229


namespace NUMINAMATH_CALUDE_pen_discount_problem_l3172_317234

/-- Proves that given a 12.5% discount on pens and the ability to buy 13 more pens
    after the discount, the original number of pens that could be bought before
    the discount is 91. -/
theorem pen_discount_problem (money : ℝ) (original_price : ℝ) 
  (original_price_positive : original_price > 0) :
  let discount_rate : ℝ := 0.125
  let discounted_price : ℝ := original_price * (1 - discount_rate)
  let original_quantity : ℝ := money / original_price
  let discounted_quantity : ℝ := money / discounted_price
  discounted_quantity - original_quantity = 13 →
  original_quantity = 91 := by
sorry


end NUMINAMATH_CALUDE_pen_discount_problem_l3172_317234


namespace NUMINAMATH_CALUDE_A_inter_B_l3172_317245

def A : Set ℝ := {-1, 0, 1, 2}
def B : Set ℝ := {x : ℝ | -2 ≤ x ∧ x < 2}

theorem A_inter_B : A ∩ B = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_A_inter_B_l3172_317245


namespace NUMINAMATH_CALUDE_product_division_equality_l3172_317252

theorem product_division_equality : ∃ x : ℝ, (400 * 7000 : ℝ) = 28000 * x^1 ∧ x = 100 := by
  sorry

end NUMINAMATH_CALUDE_product_division_equality_l3172_317252


namespace NUMINAMATH_CALUDE_juniper_bones_l3172_317298

theorem juniper_bones (initial_bones doubled_bones stolen_bones : ℕ) 
  (h1 : initial_bones = 4)
  (h2 : doubled_bones = initial_bones * 2)
  (h3 : stolen_bones = 2) :
  doubled_bones - stolen_bones = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_juniper_bones_l3172_317298


namespace NUMINAMATH_CALUDE_local_minimum_at_two_l3172_317228

def f (x : ℝ) := x^3 - 12*x

theorem local_minimum_at_two :
  ∃ δ > 0, ∀ x : ℝ, |x - 2| < δ → f x ≥ f 2 :=
sorry

end NUMINAMATH_CALUDE_local_minimum_at_two_l3172_317228


namespace NUMINAMATH_CALUDE_quadratic_increasing_condition_l3172_317236

/-- The function f(x) = ax^2 - 2x + 1 is increasing on [1, 2] iff a > 0 and 1/a < 1 -/
theorem quadratic_increasing_condition (a : ℝ) :
  (∀ x ∈ Set.Icc 1 2, Monotone (fun x => a * x^2 - 2 * x + 1)) ↔ (a > 0 ∧ 1 / a < 1) := by
  sorry


end NUMINAMATH_CALUDE_quadratic_increasing_condition_l3172_317236


namespace NUMINAMATH_CALUDE_total_oranges_count_l3172_317281

def initial_oranges : ℕ := 5
def received_oranges : ℕ := 3
def bought_skittles : ℕ := 9

theorem total_oranges_count : 
  initial_oranges + received_oranges = 8 :=
by sorry

end NUMINAMATH_CALUDE_total_oranges_count_l3172_317281


namespace NUMINAMATH_CALUDE_problem_statement_l3172_317256

theorem problem_statement (x y : ℝ) (hx : x = 7) (hy : y = 3) :
  (x - y)^2 * (x + y)^2 = 1600 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3172_317256


namespace NUMINAMATH_CALUDE_jaydee_typing_speed_l3172_317288

def typing_speed (hours : ℕ) (words : ℕ) : ℕ :=
  words / (hours * 60)

theorem jaydee_typing_speed :
  typing_speed 2 4560 = 38 :=
by sorry

end NUMINAMATH_CALUDE_jaydee_typing_speed_l3172_317288


namespace NUMINAMATH_CALUDE_sqrt_eight_plus_abs_sqrt_two_minus_two_plus_neg_half_inv_eq_sqrt_two_l3172_317213

theorem sqrt_eight_plus_abs_sqrt_two_minus_two_plus_neg_half_inv_eq_sqrt_two :
  Real.sqrt 8 + |Real.sqrt 2 - 2| + (-1/2)⁻¹ = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_eight_plus_abs_sqrt_two_minus_two_plus_neg_half_inv_eq_sqrt_two_l3172_317213


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_squared_l3172_317217

theorem sum_of_reciprocals_squared (a b c d : ℝ) : 
  a = Real.sqrt 5 + Real.sqrt 7 + Real.sqrt 35 + 2 →
  b = -Real.sqrt 5 + Real.sqrt 7 + Real.sqrt 35 + 2 →
  c = Real.sqrt 5 - Real.sqrt 7 + Real.sqrt 35 + 2 →
  d = -Real.sqrt 5 - Real.sqrt 7 + Real.sqrt 35 + 2 →
  (1/a + 1/b + 1/c + 1/d)^2 = 39/140 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_squared_l3172_317217


namespace NUMINAMATH_CALUDE_pucks_not_in_original_position_l3172_317237

/-- Represents the arrangement of three objects -/
inductive Arrangement
  | Clockwise
  | Counterclockwise

/-- Represents a single hit that changes the arrangement -/
def hit (a : Arrangement) : Arrangement :=
  match a with
  | Arrangement.Clockwise => Arrangement.Counterclockwise
  | Arrangement.Counterclockwise => Arrangement.Clockwise

/-- Applies n hits to the initial arrangement -/
def applyHits (initial : Arrangement) (n : Nat) : Arrangement :=
  match n with
  | 0 => initial
  | n + 1 => hit (applyHits initial n)

theorem pucks_not_in_original_position (initial : Arrangement) :
  applyHits initial 25 ≠ initial := by
  sorry


end NUMINAMATH_CALUDE_pucks_not_in_original_position_l3172_317237


namespace NUMINAMATH_CALUDE_evaporation_problem_l3172_317230

theorem evaporation_problem (x : ℚ) : 
  (1 - x) * (1 - 1/4) = 1/6 → x = 7/9 := by
sorry

end NUMINAMATH_CALUDE_evaporation_problem_l3172_317230


namespace NUMINAMATH_CALUDE_initial_amount_equals_sum_l3172_317227

/-- The amount of money Agatha initially had to spend on the bike. -/
def initial_amount : ℕ := 60

/-- The amount Agatha spent on the frame. -/
def frame_cost : ℕ := 15

/-- The amount Agatha spent on the front wheel. -/
def front_wheel_cost : ℕ := 25

/-- The amount Agatha has left for the seat and handlebar tape. -/
def remaining_amount : ℕ := 20

/-- Theorem stating that the initial amount equals the sum of all expenses and remaining amount. -/
theorem initial_amount_equals_sum :
  initial_amount = frame_cost + front_wheel_cost + remaining_amount :=
by sorry

end NUMINAMATH_CALUDE_initial_amount_equals_sum_l3172_317227


namespace NUMINAMATH_CALUDE_no_diametrical_opposition_possible_l3172_317290

/-- Represents a circular arrangement of numbers from 1 to 2014 -/
def CircularArrangement := Fin 2014 → Fin 2014

/-- Checks if a swap between two adjacent positions is valid -/
def validSwap (arr : CircularArrangement) (pos : Fin 2014) : Prop :=
  arr pos + arr ((pos + 1) % 2014) ≠ 2015

/-- Represents a sequence of swaps -/
def SwapSequence := List (Fin 2014)

/-- Applies a sequence of swaps to an arrangement -/
def applySwaps (initial : CircularArrangement) (swaps : SwapSequence) : CircularArrangement :=
  sorry

/-- Checks if a number is diametrically opposite its initial position -/
def isDiametricallyOpposite (initial final : CircularArrangement) (pos : Fin 2014) : Prop :=
  final pos = initial ((pos + 1007) % 2014)

/-- The main theorem stating that it's impossible to achieve diametrical opposition for all numbers -/
theorem no_diametrical_opposition_possible :
  ∀ (initial : CircularArrangement) (swaps : SwapSequence),
    (∀ (pos : Fin 2014), validSwap (applySwaps initial swaps) pos) →
    ¬(∀ (pos : Fin 2014), isDiametricallyOpposite initial (applySwaps initial swaps) pos) :=
  sorry

end NUMINAMATH_CALUDE_no_diametrical_opposition_possible_l3172_317290


namespace NUMINAMATH_CALUDE_log_10_7_exists_function_l3172_317243

-- Define the variables and conditions
variable (r s : ℝ)
variable (h1 : Real.log 3 / Real.log 4 = r)
variable (h2 : Real.log 5 / Real.log 7 = s)

-- State the theorem
theorem log_10_7_exists_function (r s : ℝ) (h1 : Real.log 3 / Real.log 4 = r) (h2 : Real.log 5 / Real.log 7 = s) :
  ∃ f : ℝ → ℝ → ℝ, Real.log 7 / Real.log 10 = f r s := by
  sorry

end NUMINAMATH_CALUDE_log_10_7_exists_function_l3172_317243


namespace NUMINAMATH_CALUDE_teacher_distribution_l3172_317210

/-- The number of ways to distribute n distinct objects among k groups, 
    with each group receiving at least one object -/
def distribute (n k : ℕ) : ℕ := sorry

theorem teacher_distribution : distribute 4 3 = 36 := by sorry

end NUMINAMATH_CALUDE_teacher_distribution_l3172_317210


namespace NUMINAMATH_CALUDE_f_at_two_l3172_317283

noncomputable section

open Real

-- Define the function f
variable (f : ℝ → ℝ)

-- State the conditions
axiom f_monotonic : Monotone f
axiom f_condition : ∀ x : ℝ, f (f x - exp x) = exp 1 + 1

-- Theorem to prove
theorem f_at_two : f 2 = exp 2 + 1 := by sorry

end NUMINAMATH_CALUDE_f_at_two_l3172_317283


namespace NUMINAMATH_CALUDE_shopping_tax_theorem_l3172_317286

/-- Calculates the total tax percentage given spending percentages and tax rates -/
def totalTaxPercentage (clothingPercent : ℝ) (foodPercent : ℝ) (otherPercent : ℝ)
                       (clothingTaxRate : ℝ) (foodTaxRate : ℝ) (otherTaxRate : ℝ) : ℝ :=
  clothingPercent * clothingTaxRate + foodPercent * foodTaxRate + otherPercent * otherTaxRate

theorem shopping_tax_theorem :
  totalTaxPercentage 0.4 0.3 0.3 0.04 0 0.08 = 0.04 := by
  sorry

end NUMINAMATH_CALUDE_shopping_tax_theorem_l3172_317286


namespace NUMINAMATH_CALUDE_sum_of_cubes_of_roots_l3172_317248

theorem sum_of_cubes_of_roots (p q r : ℝ) : 
  p^3 - 2*p^2 + 3*p - 4 = 0 →
  q^3 - 2*q^2 + 3*q - 4 = 0 →
  r^3 - 2*r^2 + 3*r - 4 = 0 →
  p^3 + q^3 + r^3 = 2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_cubes_of_roots_l3172_317248


namespace NUMINAMATH_CALUDE_abc_book_cost_l3172_317246

/-- The cost of the best-selling book "TOP" -/
def top_cost : ℝ := 8

/-- The number of "TOP" books sold -/
def top_sold : ℕ := 13

/-- The number of "ABC" books sold -/
def abc_sold : ℕ := 4

/-- The difference in earnings between "TOP" and "ABC" books -/
def earnings_difference : ℝ := 12

/-- The cost of the "ABC" book -/
def abc_cost : ℝ := 23

theorem abc_book_cost :
  top_cost * top_sold - abc_cost * abc_sold = earnings_difference :=
sorry

end NUMINAMATH_CALUDE_abc_book_cost_l3172_317246


namespace NUMINAMATH_CALUDE_quadratic_has_minimum_l3172_317284

/-- Given a quadratic function f(x) = ax^2 + bx + c where c = b^2 / (9a) and a > 0,
    prove that the graph of y = f(x) has a minimum. -/
theorem quadratic_has_minimum (a b : ℝ) (ha : a > 0) :
  let f : ℝ → ℝ := λ x => a * x^2 + b * x + b^2 / (9 * a)
  ∃ x_min : ℝ, ∀ x : ℝ, f x_min ≤ f x :=
by sorry

end NUMINAMATH_CALUDE_quadratic_has_minimum_l3172_317284


namespace NUMINAMATH_CALUDE_broker_investment_l3172_317242

theorem broker_investment (P : ℝ) (x : ℝ) (h : P > 0) :
  (P + x / 100 * P) * (1 - 30 / 100) = P * (1 + 26 / 100) →
  x = 80 := by
sorry

end NUMINAMATH_CALUDE_broker_investment_l3172_317242


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l3172_317277

/-- The asymptote equation of a hyperbola with specific properties -/
theorem hyperbola_asymptote (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_focal : 2 * Real.sqrt (a^2 + b^2) = Real.sqrt 3 * (2 * a)) :
  ∃ (k : ℝ), k = Real.sqrt 2 ∧ 
  (∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 → y = k * x ∨ y = -k * x) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l3172_317277


namespace NUMINAMATH_CALUDE_green_disks_count_l3172_317225

theorem green_disks_count (total : ℕ) (red green blue : ℕ) : 
  total = 14 →
  red = 2 * green →
  blue = green / 2 →
  total = red + green + blue →
  green = 4 := by
  sorry

end NUMINAMATH_CALUDE_green_disks_count_l3172_317225


namespace NUMINAMATH_CALUDE_expression_simplification_redundant_condition_l3172_317264

theorem expression_simplification (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (2 * x * (x^2 * y - x * y^2) + x * y * (2 * x * y - x^2)) / (x^2 * y) = x :=
by sorry

theorem redundant_condition (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  ∃ (z : ℝ), z ≠ y ∧
  (2 * x * (x^2 * y - x * y^2) + x * y * (2 * x * y - x^2)) / (x^2 * y) =
  (2 * x * (x^2 * z - x * z^2) + x * z * (2 * x * z - x^2)) / (x^2 * z) :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_redundant_condition_l3172_317264


namespace NUMINAMATH_CALUDE_marble_arrangement_theorem_l3172_317244

def num_marbles : ℕ := 4

def num_arrangements (n : ℕ) : ℕ := n.factorial

def num_adjacent_arrangements (n : ℕ) : ℕ := 2 * ((n - 1).factorial)

theorem marble_arrangement_theorem :
  num_arrangements num_marbles - num_adjacent_arrangements num_marbles = 12 :=
by sorry

end NUMINAMATH_CALUDE_marble_arrangement_theorem_l3172_317244


namespace NUMINAMATH_CALUDE_count_special_numbers_eq_56_l3172_317239

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def count_special_numbers : ℕ :=
  let thousands_digit := 8
  let units_digit := 2
  let hundreds_choices := 8
  let tens_choices := 7
  thousands_digit * units_digit * hundreds_choices * tens_choices

theorem count_special_numbers_eq_56 :
  count_special_numbers = 56 :=
sorry

end NUMINAMATH_CALUDE_count_special_numbers_eq_56_l3172_317239


namespace NUMINAMATH_CALUDE_proposition_evaluation_l3172_317207

-- Define propositions p and q
def p : Prop := ∀ x y : ℝ, x > y → -x < -y
def q : Prop := ∀ x y : ℝ, x > y → x^2 > y^2

-- State the theorem
theorem proposition_evaluation :
  (p ∧ q = False) ∧
  (p ∨ q = True) ∧
  (p ∧ (¬q) = True) ∧
  ((¬p) ∨ q = False) :=
by sorry

end NUMINAMATH_CALUDE_proposition_evaluation_l3172_317207


namespace NUMINAMATH_CALUDE_prism_surface_area_l3172_317262

/-- A rectangular prism with prime edge lengths and volume 627 has surface area 598 -/
theorem prism_surface_area : ∀ a b c : ℕ,
  Prime a → Prime b → Prime c →
  a * b * c = 627 →
  2 * (a * b + b * c + c * a) = 598 := by
sorry

end NUMINAMATH_CALUDE_prism_surface_area_l3172_317262


namespace NUMINAMATH_CALUDE_sum_range_for_cube_sum_two_l3172_317221

theorem sum_range_for_cube_sum_two (x y : ℝ) (h : x^3 + y^3 = 2) :
  0 < x + y ∧ x + y ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_range_for_cube_sum_two_l3172_317221


namespace NUMINAMATH_CALUDE_river_speed_is_two_l3172_317201

/-- The speed of the river that satisfies the given conditions -/
def river_speed (mans_speed : ℝ) (distance : ℝ) (total_time : ℝ) : ℝ :=
  2

/-- Theorem stating that the river speed is 2 kmph given the conditions -/
theorem river_speed_is_two :
  let mans_speed : ℝ := 4
  let distance : ℝ := 2.25
  let total_time : ℝ := 1.5
  river_speed mans_speed distance total_time = 2 := by
  sorry

#check river_speed_is_two

end NUMINAMATH_CALUDE_river_speed_is_two_l3172_317201


namespace NUMINAMATH_CALUDE_charity_ticket_sales_l3172_317265

theorem charity_ticket_sales (total_tickets : ℕ) (total_revenue : ℕ) 
  (donation : ℕ) (h_total_tickets : total_tickets = 200) 
  (h_total_revenue : total_revenue = 3200) (h_donation : donation = 200) :
  ∃ (full_price : ℕ) (half_price : ℕ) (price : ℕ),
    full_price + half_price = total_tickets ∧
    full_price * price + half_price * (price / 2) + donation = total_revenue ∧
    full_price * price = 1000 := by
  sorry

end NUMINAMATH_CALUDE_charity_ticket_sales_l3172_317265


namespace NUMINAMATH_CALUDE_x_over_y_value_l3172_317208

theorem x_over_y_value (x y : ℝ) 
  (h1 : 3 < (x - y) / (x + y)) 
  (h2 : (x - y) / (x + y) < 6)
  (h3 : ∃ (n : ℤ), x / y = n) : 
  x / y = -2 := by
sorry

end NUMINAMATH_CALUDE_x_over_y_value_l3172_317208


namespace NUMINAMATH_CALUDE_total_people_in_groups_l3172_317218

theorem total_people_in_groups (art_group : ℕ) (dance_group_ratio : ℚ) : 
  art_group = 25 → dance_group_ratio = 1.4 → 
  art_group + (↑art_group * dance_group_ratio) = 55 := by
  sorry

end NUMINAMATH_CALUDE_total_people_in_groups_l3172_317218


namespace NUMINAMATH_CALUDE_chord_length_l3172_317211

/-- The parabola M: y^2 = 2px where p > 0 -/
def parabola_M (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

/-- The circle C: x^2 + (y-4)^2 = a^2 -/
def circle_C (a : ℝ) (x y : ℝ) : Prop := x^2 + (y-4)^2 = a^2

/-- Point A is in the first quadrant -/
def point_A_first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

/-- Distance from A to focus of parabola M is a -/
def distance_A_to_focus (p a x y : ℝ) : Prop := (x - p/2)^2 + y^2 = a^2

/-- Sum of distances from a point on M to its directrix and to point C has a maximum of 2 -/
def max_distance_sum (p : ℝ) : Prop := ∃ (x y : ℝ), parabola_M p x y ∧ (x + p/2) + ((x - 0)^2 + (y - 4)^2).sqrt ≤ 2

/-- The theorem: Length of chord intercepted by line OA on circle C is 7√2/3 -/
theorem chord_length (p a x y : ℝ) : 
  parabola_M p x y → 
  circle_C a x y → 
  point_A_first_quadrant x y → 
  distance_A_to_focus p a x y → 
  max_distance_sum p → 
  ((2 * a)^2 - (8/3)^2).sqrt = 7 * Real.sqrt 2 / 3 := by sorry

end NUMINAMATH_CALUDE_chord_length_l3172_317211


namespace NUMINAMATH_CALUDE_surface_area_unchanged_after_cube_removal_l3172_317269

theorem surface_area_unchanged_after_cube_removal 
  (l w h : ℝ) (cube_side : ℝ) 
  (hl : l = 10) (hw : w = 5) (hh : h = 3) (hc : cube_side = 2) : 
  2 * (l * w + l * h + w * h) = 
  2 * (l * w + l * h + w * h) - 3 * cube_side^2 + 3 * cube_side^2 := by
  sorry

end NUMINAMATH_CALUDE_surface_area_unchanged_after_cube_removal_l3172_317269


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l3172_317226

theorem least_addition_for_divisibility (n : ℕ) : 
  let x := Nat.minFac (9 - n % 9)
  x > 0 ∧ (4499 + x) % 9 = 0 ∧ ∀ y : ℕ, y < x → (4499 + y) % 9 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l3172_317226


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3172_317223

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a where a₁₃ + a₅ = 32,
    prove that a₉ = 16 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_sum : a 13 + a 5 = 32) : 
  a 9 = 16 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3172_317223


namespace NUMINAMATH_CALUDE_min_f_gt_min_g_l3172_317216

open Set

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the condition given in the problem
variable (h : ∀ x : ℝ, ∃ x₀ : ℝ, f x > g x₀)

-- State the theorem to be proved
theorem min_f_gt_min_g : (⨅ x, f x) > (⨅ x, g x) := by sorry

end NUMINAMATH_CALUDE_min_f_gt_min_g_l3172_317216


namespace NUMINAMATH_CALUDE_quadratic_real_solutions_l3172_317200

theorem quadratic_real_solutions (m : ℝ) : 
  (∃ x : ℝ, m * x^2 + 2 * x + 1 = 0) ↔ (m ≤ 1 ∧ m ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_real_solutions_l3172_317200


namespace NUMINAMATH_CALUDE_solution_to_system_of_equations_l3172_317259

theorem solution_to_system_of_equations :
  let system (x y z : ℝ) : Prop :=
    x^2 - y^2 + z = 64 / (x * y) ∧
    y^2 - z^2 + x = 64 / (y * z) ∧
    z^2 - x^2 + y = 64 / (x * z)
  ∀ x y z : ℝ, system x y z →
    ((x = 4 ∧ y = 4 ∧ z = 4) ∨
     (x = -4 ∧ y = -4 ∧ z = 4) ∨
     (x = -4 ∧ y = 4 ∧ z = -4) ∨
     (x = 4 ∧ y = -4 ∧ z = -4)) :=
by
  sorry

#check solution_to_system_of_equations

end NUMINAMATH_CALUDE_solution_to_system_of_equations_l3172_317259


namespace NUMINAMATH_CALUDE_sams_new_books_l3172_317247

theorem sams_new_books (adventure_books : ℕ) (mystery_books : ℕ) (used_books : ℕ) 
  (h1 : adventure_books = 13)
  (h2 : mystery_books = 17)
  (h3 : used_books = 15) : 
  adventure_books + mystery_books - used_books = 15 := by
  sorry

end NUMINAMATH_CALUDE_sams_new_books_l3172_317247


namespace NUMINAMATH_CALUDE_diamond_two_seven_l3172_317232

-- Define the diamond operation
def diamond (x y : ℝ) : ℝ := 3 * x + 5 * y

-- State the theorem
theorem diamond_two_seven : diamond 2 7 = 41 := by
  sorry

end NUMINAMATH_CALUDE_diamond_two_seven_l3172_317232


namespace NUMINAMATH_CALUDE_library_book_combinations_l3172_317219

theorem library_book_combinations : Nat.choose 5 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_library_book_combinations_l3172_317219


namespace NUMINAMATH_CALUDE_min_value_of_f_zero_l3172_317235

/-- A quadratic function from reals to reals -/
def QuadraticFunction := ℝ → ℝ

/-- Predicate to check if a function is quadratic -/
def IsQuadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c

/-- Predicate to check if a function is ever more than another function -/
def EverMore (f g : ℝ → ℝ) : Prop :=
  ∀ x, f x ≥ g x

/-- The theorem statement -/
theorem min_value_of_f_zero
  (f : QuadraticFunction)
  (hquad : IsQuadratic f)
  (hf1 : f 1 = 16)
  (hg : EverMore f (fun x ↦ (x + 3)^2))
  (hh : EverMore f (fun x ↦ x^2 + 9)) :
  ∃ (min_f0 : ℝ), min_f0 = 21/2 ∧ f 0 ≥ min_f0 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_zero_l3172_317235


namespace NUMINAMATH_CALUDE_fashion_pricing_increase_l3172_317289

theorem fashion_pricing_increase (C : ℝ) : 
  let retailer_price := 1.40 * C
  let customer_price := 1.6100000000000001 * C
  ((customer_price - retailer_price) / retailer_price) * 100 = 15 := by
sorry

end NUMINAMATH_CALUDE_fashion_pricing_increase_l3172_317289


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_sqrt_192_l3172_317260

theorem sqrt_sum_equals_sqrt_192 (N : ℕ+) :
  Real.sqrt 12 + Real.sqrt 108 = Real.sqrt N.1 → N.1 = 192 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_sqrt_192_l3172_317260


namespace NUMINAMATH_CALUDE_chads_cat_food_packages_l3172_317204

/-- Chad's pet food purchase problem -/
theorem chads_cat_food_packages :
  ∀ (c : ℕ), -- c represents the number of packages of cat food
  (9 * c = 2 * 3 + 48) → -- Equation representing the difference in cans
  c = 6 := by
sorry

end NUMINAMATH_CALUDE_chads_cat_food_packages_l3172_317204


namespace NUMINAMATH_CALUDE_dice_arithmetic_progression_probability_l3172_317266

def num_dice : ℕ := 4
def faces_per_die : ℕ := 6

def total_outcomes : ℕ := faces_per_die ^ num_dice

def valid_progressions : List (List ℕ) := [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]

def favorable_outcomes : ℕ := valid_progressions.length * (num_dice.factorial)

theorem dice_arithmetic_progression_probability :
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 18 := by sorry

end NUMINAMATH_CALUDE_dice_arithmetic_progression_probability_l3172_317266


namespace NUMINAMATH_CALUDE_certain_number_problem_l3172_317295

theorem certain_number_problem (x : ℝ) : 
  (10 + x + 60) / 3 = (10 + 40 + 25) / 3 + 5 → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l3172_317295


namespace NUMINAMATH_CALUDE_sets_intersection_and_complement_l3172_317297

-- Define the sets A, B, and C
def A : Set ℝ := {x | x^2 - 4*x + 3 ≤ 0}
def B : Set ℝ := {x | (x - 2) / (x + 3) > 0}
def C (a : ℝ) : Set ℝ := {x | 1 < x ∧ x < a}

-- State the theorem
theorem sets_intersection_and_complement (a : ℝ) 
  (h : A ∩ C a = C a) : 
  (A ∩ B = Set.Ioc 2 3) ∧ 
  ((Set.univ \ A) ∪ (Set.univ \ B) = Set.Iic 2 ∪ Set.Ioi 3) ∧
  (a ≤ 3) := by sorry

end NUMINAMATH_CALUDE_sets_intersection_and_complement_l3172_317297


namespace NUMINAMATH_CALUDE_solution_of_equation_l3172_317240

theorem solution_of_equation (x : ℝ) : 
  (6 * x) / (x + 2) - 4 / (x + 2) = 2 / (x + 2) ↔ x = 1 := by
sorry

end NUMINAMATH_CALUDE_solution_of_equation_l3172_317240

import Mathlib

namespace solve_logarithmic_equation_l2321_232138

-- Define the base-10 logarithm function
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem solve_logarithmic_equation :
  ∃ x : ℝ, log10 (3 * x + 4) = 1 ∧ x = 2 := by
  sorry

end solve_logarithmic_equation_l2321_232138


namespace cookies_per_sitting_l2321_232173

/-- The number of times Theo eats cookies per day -/
def eats_per_day : ℕ := 3

/-- The number of days Theo eats cookies per month -/
def days_per_month : ℕ := 20

/-- The total number of cookies Theo eats in 3 months -/
def total_cookies : ℕ := 2340

/-- The number of months considered -/
def months : ℕ := 3

/-- Theorem stating the number of cookies Theo can eat in one sitting -/
theorem cookies_per_sitting :
  total_cookies / (eats_per_day * days_per_month * months) = 13 := by sorry

end cookies_per_sitting_l2321_232173


namespace polynomial_division_remainder_l2321_232113

theorem polynomial_division_remainder : ∃ q : Polynomial ℂ, 
  (X^4 - 1) * (X^3 - 1) = (X^2 + 1) * q + (2 + X) := by sorry

end polynomial_division_remainder_l2321_232113


namespace hyperbola_eccentricity_l2321_232157

noncomputable section

-- Define the hyperbola
def hyperbola (a b x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Define the right focus F
def right_focus (a b c : ℝ) : ℝ × ℝ := (c, 0)

-- Define the line l passing through the origin
def line_through_origin (m n : ℝ) : Set (ℝ × ℝ) :=
  {(x, y) | ∃ t : ℝ, x = m * t ∧ y = n * t}

-- Define the condition that M and N are on the hyperbola and the line
def points_on_hyperbola_and_line (a b m n : ℝ) (M N : ℝ × ℝ) : Prop :=
  hyperbola a b M.1 M.2 ∧ hyperbola a b N.1 N.2 ∧
  M ∈ line_through_origin m n ∧ N ∈ line_through_origin m n

-- Define the perpendicularity condition
def perpendicular_vectors (F M N : ℝ × ℝ) : Prop :=
  (M.1 - F.1) * (N.1 - F.1) + (M.2 - F.2) * (N.2 - F.2) = 0

-- Define the area condition
def triangle_area (F M N : ℝ × ℝ) (a b : ℝ) : Prop :=
  abs ((M.1 - F.1) * (N.2 - F.2) - (N.1 - F.1) * (M.2 - F.2)) / 2 = a * b

-- Main theorem
theorem hyperbola_eccentricity
  (a b c : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (hc : c > 0)
  (F : ℝ × ℝ)
  (hF : F = right_focus a b c)
  (M N : ℝ × ℝ)
  (h_points : ∃ m n : ℝ, points_on_hyperbola_and_line a b m n M N)
  (h_perp : perpendicular_vectors F M N)
  (h_area : triangle_area F M N a b) :
  c^2 / a^2 = 2 :=
sorry

end hyperbola_eccentricity_l2321_232157


namespace unfair_coin_flip_probability_l2321_232104

/-- The probability of getting heads in a single flip of the unfair coin -/
def p_heads : ℚ := 1/3

/-- The probability of getting tails in a single flip of the unfair coin -/
def p_tails : ℚ := 2/3

/-- The number of coin flips -/
def n : ℕ := 10

/-- The number of heads we want to get -/
def k : ℕ := 3

/-- The probability of getting exactly k heads in n flips of the unfair coin -/
def prob_k_heads (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k : ℚ) * p^k * (1 - p)^(n - k)

theorem unfair_coin_flip_probability :
  prob_k_heads n k p_heads = 512/1969 := by
  sorry

end unfair_coin_flip_probability_l2321_232104


namespace remainder_two_power_thirty_plus_three_mod_seven_l2321_232119

theorem remainder_two_power_thirty_plus_three_mod_seven :
  (2^30 + 3) % 7 = 4 := by
  sorry

end remainder_two_power_thirty_plus_three_mod_seven_l2321_232119


namespace inequality_implies_equality_l2321_232193

theorem inequality_implies_equality (a b : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0)
  (h_ineq : Real.log a + Real.log (b^2) ≥ 2*a + b^2/2 - 2) :
  a - 2*b = 1/2 - 2*Real.sqrt 2 := by
sorry

end inequality_implies_equality_l2321_232193


namespace four_square_figure_perimeter_l2321_232179

/-- A figure consisting of four identical squares -/
structure FourSquareFigure where
  /-- The side length of each square in the figure -/
  side_length : ℝ
  /-- The total area of the figure is 144 cm² -/
  area_eq : 4 * side_length ^ 2 = 144

/-- The perimeter of a four-square figure is 60 cm -/
theorem four_square_figure_perimeter (fig : FourSquareFigure) : 
  10 * fig.side_length = 60 := by
  sorry

#check four_square_figure_perimeter

end four_square_figure_perimeter_l2321_232179


namespace point_on_extension_line_l2321_232139

/-- Given two points P₁ and P₂ in ℝ², and a point P on the extension line of P₁P₂
    such that the distance between P₁ and P is twice the distance between P and P₂,
    prove that P has the coordinates (-2, 11). -/
theorem point_on_extension_line (P₁ P₂ P : ℝ × ℝ) : 
  P₁ = (2, -1) → 
  P₂ = (0, 5) → 
  (∃ t : ℝ, P = P₁ + t • (P₂ - P₁)) →
  dist P₁ P = 2 * dist P P₂ →
  P = (-2, 11) := by
  sorry

end point_on_extension_line_l2321_232139


namespace fraction_equality_l2321_232155

theorem fraction_equality : (5 * 7 - 3) / 9 = 32 / 9 := by
  sorry

end fraction_equality_l2321_232155


namespace exists_monochromatic_isosceles_right_triangle_l2321_232121

/-- A color type with three possible values -/
inductive Color
  | Red
  | Green
  | Blue

/-- A point in the infinite grid -/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- A coloring function that assigns a color to each point in the grid -/
def Coloring := GridPoint → Color

/-- An isosceles right triangle in the grid -/
structure IsoscelesRightTriangle where
  p1 : GridPoint
  p2 : GridPoint
  p3 : GridPoint
  is_isosceles : (p1.x - p2.x)^2 + (p1.y - p2.y)^2 = (p1.x - p3.x)^2 + (p1.y - p3.y)^2
  is_right : (p2.x - p1.x) * (p3.x - p1.x) + (p2.y - p1.y) * (p3.y - p1.y) = 0

/-- The main theorem: In any coloring of an infinite grid with three colors,
    there exists an isosceles right triangle with vertices of the same color -/
theorem exists_monochromatic_isosceles_right_triangle (c : Coloring) :
  ∃ (t : IsoscelesRightTriangle), c t.p1 = c t.p2 ∧ c t.p2 = c t.p3 := by
  sorry

end exists_monochromatic_isosceles_right_triangle_l2321_232121


namespace total_fruits_picked_l2321_232107

theorem total_fruits_picked (sara_pears tim_pears lily_apples max_oranges : ℕ)
  (h1 : sara_pears = 6)
  (h2 : tim_pears = 5)
  (h3 : lily_apples = 4)
  (h4 : max_oranges = 3) :
  sara_pears + tim_pears + lily_apples + max_oranges = 18 := by
  sorry

end total_fruits_picked_l2321_232107


namespace ant_on_red_after_six_moves_probability_on_red_after_six_moves_l2321_232149

/-- Represents the color of a dot on the lattice -/
inductive DotColor
| Red
| Blue

/-- Represents the state of the ant's position -/
structure AntState :=
  (color : DotColor)

/-- Defines a single move of the ant -/
def move (state : AntState) : AntState :=
  match state.color with
  | DotColor.Red => { color := DotColor.Blue }
  | DotColor.Blue => { color := DotColor.Red }

/-- Applies n moves to the initial state -/
def apply_moves (initial : AntState) (n : ℕ) : AntState :=
  match n with
  | 0 => initial
  | n + 1 => move (apply_moves initial n)

/-- The main theorem to prove -/
theorem ant_on_red_after_six_moves (initial : AntState) :
  initial.color = DotColor.Red →
  (apply_moves initial 6).color = DotColor.Red :=
sorry

/-- The probability of the ant being on a red dot after 6 moves -/
theorem probability_on_red_after_six_moves (initial : AntState) :
  initial.color = DotColor.Red →
  ∃ (p : ℝ), p = 1 ∧ 
  (∀ (final : AntState), (apply_moves initial 6).color = DotColor.Red → p = 1) :=
sorry

end ant_on_red_after_six_moves_probability_on_red_after_six_moves_l2321_232149


namespace rectangle_square_overlap_ratio_l2321_232153

/-- Given a rectangle ABCD and a square JKLM, if the rectangle shares 40% of its area with the square,
    and the square shares 25% of its area with the rectangle, then the ratio of the length (AB) to 
    the width (AD) of the rectangle is 15.625. -/
theorem rectangle_square_overlap_ratio : 
  ∀ (AB AD s : ℝ), 
  AB > 0 → AD > 0 → s > 0 →
  0.4 * AB * AD = 0.25 * s^2 →
  AB / AD = 15.625 := by
    sorry

end rectangle_square_overlap_ratio_l2321_232153


namespace fib_8_and_sum_2016_l2321_232100

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- Sum of first n terms of Fibonacci sequence -/
def fib_sum (n : ℕ) : ℕ :=
  (List.range n).map fib |>.sum

theorem fib_8_and_sum_2016 :
  fib 7 = 21 ∧
  ∀ m : ℕ, fib 2017 = m^2 + 1 → fib_sum 2016 = m^2 := by
  sorry

end fib_8_and_sum_2016_l2321_232100


namespace line_bisects_circle_l2321_232167

/-- The equation of a circle in the xy-plane -/
def Circle (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y + 1 = 0

/-- The equation of a line in the xy-plane -/
def Line (x y : ℝ) : Prop :=
  x - y + 1 = 0

/-- The center of the circle -/
def center : ℝ × ℝ := (1, 2)

/-- Theorem stating that the line bisects the circle -/
theorem line_bisects_circle :
  ∀ x y : ℝ, Circle x y → Line x y → (x, y) = center :=
sorry

end line_bisects_circle_l2321_232167


namespace max_sum_xy_l2321_232114

def associated_numbers (m : ℕ) : List ℕ :=
  sorry

def P (m : ℕ) : ℚ :=
  (associated_numbers m).sum / 22

def x (a b : ℕ) : ℕ := 100 * a + 10 * b + 3

def y (b : ℕ) : ℕ := 400 + 10 * b + 5

theorem max_sum_xy :
  ∀ a b : ℕ,
    1 ≤ a ∧ a ≤ 9 →
    1 ≤ b ∧ b ≤ 9 →
    (∀ d : ℕ, d ∈ associated_numbers (x a b) → d ≠ 0) →
    (∀ d : ℕ, d ∈ associated_numbers (y b) → d ≠ 0) →
    P (x a b) + P (y b) = 20 →
    x a b + y b ≤ 1028 :=
  sorry

end max_sum_xy_l2321_232114


namespace fraction_power_seven_l2321_232170

theorem fraction_power_seven : (5 / 7 : ℚ) ^ 7 = 78125 / 823543 := by sorry

end fraction_power_seven_l2321_232170


namespace quadratic_equation_root_zero_l2321_232185

theorem quadratic_equation_root_zero (k : ℝ) : 
  (∀ x : ℝ, (k - 1) * x^2 + 3 * x + k^2 - 1 = 0 → x = 0 ∨ x ≠ 0) →
  ((k - 1) * 0^2 + 3 * 0 + k^2 - 1 = 0) →
  k = -1 := by
sorry

end quadratic_equation_root_zero_l2321_232185


namespace shaded_area_sum_l2321_232137

/-- The sum of the areas of two pie-shaped regions in a circle with an inscribed square --/
theorem shaded_area_sum (d : ℝ) (h : d = 16) : 
  let r := d / 2
  let sector_area := 2 * (π * r^2 * (45 / 360))
  let triangle_area := 2 * (1 / 2 * r^2)
  sector_area - triangle_area = 32 * π - 64 := by
  sorry

end shaded_area_sum_l2321_232137


namespace polynomial_factorization_isosceles_triangle_l2321_232123

-- Part 1: Polynomial factorization
theorem polynomial_factorization (x y : ℝ) :
  x^2 - 2*x*y + y^2 - 16 = (x - y + 4) * (x - y - 4) := by sorry

-- Part 2: Triangle shape determination
def is_triangle (a b c : ℝ) : Prop := a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

theorem isosceles_triangle (a b c : ℝ) (h : is_triangle a b c) :
  a^2 - a*b + a*c - b*c = 0 → a = b := by sorry

end polynomial_factorization_isosceles_triangle_l2321_232123


namespace car_original_price_verify_car_price_l2321_232188

/-- Calculates the original price of a car given the final price after discounts, taxes, and fees. -/
theorem car_original_price (final_price : ℝ) (doc_fee : ℝ) 
  (discount1 discount2 discount3 tax_rate : ℝ) : ℝ :=
  let remaining_after_discounts := (1 - discount1) * (1 - discount2) * (1 - discount3)
  let price_with_tax := remaining_after_discounts * (1 + tax_rate)
  (final_price - doc_fee) / price_with_tax

/-- Proves that the calculated original price satisfies the given conditions. -/
theorem verify_car_price : 
  let original_price := car_original_price 7500 200 0.15 0.20 0.25 0.10
  0.561 * original_price + 200 = 7500 := by
  sorry

end car_original_price_verify_car_price_l2321_232188


namespace april_price_index_april_price_increase_l2321_232140

/-- Represents the price index for a given month -/
structure PriceIndex where
  month : Nat
  value : Real

/-- Calculates the price index for a given month based on the initial index and monthly decrease rate -/
def calculate_price_index (initial_index : Real) (monthly_decrease : Real) (month : Nat) : Real :=
  initial_index - (month - 1) * monthly_decrease

/-- Theorem stating that the price index in April is 1.12 given the conditions -/
theorem april_price_index 
  (january_index : PriceIndex)
  (monthly_decrease : Real)
  (h1 : january_index.month = 1)
  (h2 : january_index.value = 1.15)
  (h3 : monthly_decrease = 0.01)
  : ∃ (april_index : PriceIndex), 
    april_index.month = 4 ∧ 
    april_index.value = calculate_price_index january_index.value monthly_decrease 4 ∧
    april_index.value = 1.12 :=
sorry

/-- Theorem stating that the price in April has increased by 12% compared to the same month last year -/
theorem april_price_increase 
  (april_index : PriceIndex)
  (h : april_index.value = 1.12)
  : (april_index.value - 1) * 100 = 12 :=
sorry

end april_price_index_april_price_increase_l2321_232140


namespace solution_set_abs_inequality_l2321_232190

theorem solution_set_abs_inequality (x : ℝ) :
  (Set.Icc 1 3 : Set ℝ) = {x | |2 - x| ≤ 1} :=
sorry

end solution_set_abs_inequality_l2321_232190


namespace arrangementsWithRestrictionFor6_l2321_232101

/-- The number of ways to arrange n people in a line -/
def linearArrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n people in a line where one specific person
    cannot be placed on either end -/
def arrangementsWithRestriction (n : ℕ) : ℕ :=
  (n - 2) * linearArrangements (n - 1)

/-- Theorem stating that the number of ways to arrange 6 people in a line,
    where one specific person cannot be placed on either end, is 480 -/
theorem arrangementsWithRestrictionFor6 :
    arrangementsWithRestriction 6 = 480 := by
  sorry

end arrangementsWithRestrictionFor6_l2321_232101


namespace max_value_of_z_l2321_232111

/-- Given a system of inequalities, prove that the maximum value of z = 2x + 3y is 8 -/
theorem max_value_of_z (x y : ℝ) 
  (h1 : x + y - 1 ≥ 0) 
  (h2 : y - x - 1 ≤ 0) 
  (h3 : x ≤ 1) : 
  (∀ x' y' : ℝ, x' + y' - 1 ≥ 0 → y' - x' - 1 ≤ 0 → x' ≤ 1 → 2*x' + 3*y' ≤ 2*x + 3*y) →
  2*x + 3*y = 8 := by
  sorry

end max_value_of_z_l2321_232111


namespace three_letter_initials_count_l2321_232180

theorem three_letter_initials_count (n : ℕ) (h : n = 10) : n ^ 3 = 1000 := by
  sorry

end three_letter_initials_count_l2321_232180


namespace quadratic_equation_roots_l2321_232146

theorem quadratic_equation_roots : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  (x₁ + 1) * (x₁ - 1) = 2 * x₁ + 3 ∧ 
  (x₂ + 1) * (x₂ - 1) = 2 * x₂ + 3 :=
sorry

end quadratic_equation_roots_l2321_232146


namespace seventeen_stations_tickets_l2321_232145

/-- The number of unique, non-directional tickets needed for travel between any two stations -/
def num_tickets (n : ℕ) : ℕ :=
  (n * (n - 1)) / 2

/-- Theorem: For 17 stations, the number of unique, non-directional tickets is 68 -/
theorem seventeen_stations_tickets :
  num_tickets 17 = 68 := by
  sorry

end seventeen_stations_tickets_l2321_232145


namespace solution_set_of_inequality_l2321_232183

theorem solution_set_of_inequality (x : ℝ) : 
  (x + 3)^2 < 1 ↔ -4 < x ∧ x < -2 := by
  sorry

end solution_set_of_inequality_l2321_232183


namespace vacation_pictures_l2321_232120

theorem vacation_pictures (zoo museum beach amusement_park deleted : ℕ) :
  zoo = 802 →
  museum = 526 →
  beach = 391 →
  amusement_park = 868 →
  deleted = 1395 →
  zoo + museum + beach + amusement_park - deleted = 1192 := by
  sorry

end vacation_pictures_l2321_232120


namespace matrix_multiplication_example_l2321_232151

theorem matrix_multiplication_example :
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![3, 1; 2, -4]
  let B : Matrix (Fin 2) (Fin 2) ℤ := !![7, -3; 0, 2]
  A * B = !![21, -7; 14, -14] := by
  sorry

end matrix_multiplication_example_l2321_232151


namespace sector_arc_length_l2321_232172

theorem sector_arc_length (r : ℝ) (A : ℝ) (l : ℝ) : 
  r = 2 → A = π / 3 → A = 1 / 2 * r * l → l = π / 3 := by
  sorry

end sector_arc_length_l2321_232172


namespace divisor_not_zero_l2321_232109

theorem divisor_not_zero (a b : ℝ) : b ≠ 0 → ∃ (c : ℝ), a / b = c := by
  sorry

end divisor_not_zero_l2321_232109


namespace z_in_first_quadrant_l2321_232171

def z : ℂ := (4 + 3*Complex.I) * (2 + Complex.I)

theorem z_in_first_quadrant : 
  Real.sign (z.re) = 1 ∧ Real.sign (z.im) = 1 :=
by sorry

end z_in_first_quadrant_l2321_232171


namespace chef_almond_weight_l2321_232134

/-- The weight of pecans bought by the chef in kilograms. -/
def pecan_weight : ℝ := 0.38

/-- The total weight of nuts bought by the chef in kilograms. -/
def total_nut_weight : ℝ := 0.52

/-- The weight of almonds bought by the chef in kilograms. -/
def almond_weight : ℝ := total_nut_weight - pecan_weight

theorem chef_almond_weight :
  almond_weight = 0.14 := by sorry

end chef_almond_weight_l2321_232134


namespace probability_two_girls_l2321_232178

theorem probability_two_girls (total_members : ℕ) (girl_members : ℕ) : 
  total_members = 12 → 
  girl_members = 7 → 
  (Nat.choose girl_members 2 : ℚ) / (Nat.choose total_members 2 : ℚ) = 7 / 22 := by
  sorry

end probability_two_girls_l2321_232178


namespace geometric_series_sum_specific_l2321_232198

def geometric_series_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_series_sum_specific : 
  let a : ℚ := 1/4
  let r : ℚ := 1/4
  let n : ℕ := 8
  geometric_series_sum a r n = 65535/196608 := by
  sorry

end geometric_series_sum_specific_l2321_232198


namespace non_dividing_diagonals_count_l2321_232174

/-- The number of sides in the regular polygon -/
def n : ℕ := 150

/-- The total number of diagonals in a polygon with n sides -/
def total_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- The number of diagonals that divide the polygon into two equal parts -/
def equal_dividing_diagonals (n : ℕ) : ℕ := n / 2

/-- The number of diagonals that do not divide the polygon into two equal parts -/
def non_dividing_diagonals (n : ℕ) : ℕ := total_diagonals n - equal_dividing_diagonals n

theorem non_dividing_diagonals_count :
  non_dividing_diagonals n = 10950 :=
by sorry

end non_dividing_diagonals_count_l2321_232174


namespace min_sum_product_l2321_232131

theorem min_sum_product (a1 a2 a3 b1 b2 b3 c1 c2 c3 d : ℕ) : 
  (({a1, a2, a3, b1, b2, b3, c1, c2, c3, d} : Finset ℕ) = Finset.range 10) →
  a1 * a2 * a3 + b1 * b2 * b3 + c1 * c2 * c3 + d ≥ 609 ∧
  ∃ (p1 p2 p3 q1 q2 q3 r1 r2 r3 s : ℕ),
    ({p1, p2, p3, q1, q2, q3, r1, r2, r3, s} : Finset ℕ) = Finset.range 10 ∧
    p1 * p2 * p3 + q1 * q2 * q3 + r1 * r2 * r3 + s = 609 :=
by sorry

end min_sum_product_l2321_232131


namespace geometric_series_common_ratio_l2321_232159

theorem geometric_series_common_ratio : 
  let a₁ : ℚ := 4/7
  let a₂ : ℚ := 12/7
  let a₃ : ℚ := 36/7
  let r : ℚ := a₂ / a₁
  (∀ n : ℕ, n ≥ 1 → (a₁ * r^(n-1) : ℚ) = 4/7 * 3^(n-1)) →
  r = 3 :=
by sorry

end geometric_series_common_ratio_l2321_232159


namespace simplify_expression_l2321_232194

theorem simplify_expression (x y : ℝ) (hx : x ≠ 0) :
  (y - 1) * x⁻¹ - y = -((y * x - y + 1) / x) := by sorry

end simplify_expression_l2321_232194


namespace track_length_track_length_is_350_l2321_232158

/-- The length of a circular track given specific running conditions -/
theorem track_length : ℝ → ℝ → ℝ → Prop :=
  λ first_meet second_meet track_length =>
    -- Brenda and Sally start at diametrically opposite points
    -- They first meet after Brenda has run 'first_meet' meters
    -- They next meet after Sally has run 'second_meet' meters past their first meeting point
    -- 'track_length' is the length of the circular track
    first_meet = 150 ∧
    second_meet = 200 ∧
    track_length = 350 ∧
    -- The total distance run by both runners is twice the track length
    2 * track_length = 2 * first_meet + second_meet

theorem track_length_is_350 : ∃ (l : ℝ), track_length 150 200 l :=
  sorry

end track_length_track_length_is_350_l2321_232158


namespace odot_properties_l2321_232176

/-- The custom operation ⊙ -/
def odot (a : ℝ) (x y : ℝ) : ℝ := 18 + x - a * y

/-- Theorem stating the properties of the ⊙ operation -/
theorem odot_properties :
  ∃ a : ℝ, (odot a 2 3 = 8) ∧ (odot a 3 5 = 1) ∧ (odot a 5 3 = 11) := by
  sorry

end odot_properties_l2321_232176


namespace intersection_angle_relation_l2321_232122

-- Define the circles and points
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def Circle.intersect (c1 c2 : Circle) : Prop :=
  ∃ p : ℝ × ℝ, (p.1 - c1.center.1)^2 + (p.2 - c1.center.2)^2 = c1.radius^2 ∧
                (p.1 - c2.center.1)^2 + (p.2 - c2.center.2)^2 = c2.radius^2

-- Define the theorem
theorem intersection_angle_relation (c1 c2 : Circle) (α β : ℝ) :
  c1.radius = c2.radius →
  c1.radius > 0 →
  (c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2 > c1.radius^2 →
  Circle.intersect c1 c2 →
  -- Assume α and β are the angles formed at the intersection points
  -- (We don't formally define these angles as it would require more complex geometry)
  β = 3 * α :=
sorry

end intersection_angle_relation_l2321_232122


namespace chocolate_heart_bags_l2321_232102

theorem chocolate_heart_bags (total_candy : ℕ) (total_bags : ℕ) (kisses_bags : ℕ) (non_chocolate_pieces : ℕ)
  (h1 : total_candy = 63)
  (h2 : total_bags = 9)
  (h3 : kisses_bags = 3)
  (h4 : non_chocolate_pieces = 28)
  (h5 : total_candy % total_bags = 0) -- Ensure equal division
  : (total_bags - kisses_bags - (non_chocolate_pieces / (total_candy / total_bags))) = 2 := by
  sorry

end chocolate_heart_bags_l2321_232102


namespace mental_math_competition_l2321_232141

theorem mental_math_competition :
  ∃! (numbers : Finset ℕ),
    numbers.card = 4 ∧
    (∀ n ∈ numbers,
      ∃ (M m : ℕ),
        n = 15 * M + 11 * m ∧
        M > 1 ∧ m > 1 ∧
        Odd M ∧ Odd m ∧
        (∀ d : ℕ, d > 1 → Odd d → d ∣ n → m ≤ d ∧ d ≤ M) ∧
        numbers = {528, 880, 1232, 1936}) :=
by sorry

end mental_math_competition_l2321_232141


namespace arithmetic_sequence_first_term_l2321_232182

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- The median of a sequence with an odd number of terms is the middle term. -/
def median (a : ℕ → ℝ) (n : ℕ) : ℝ := a ((n + 1) / 2)

theorem arithmetic_sequence_first_term
  (a : ℕ → ℝ)  -- The sequence
  (n : ℕ)      -- The number of terms in the sequence
  (h1 : is_arithmetic_sequence a)
  (h2 : median a n = 1010)
  (h3 : a n = 2015) :
  a 1 = 5 := by
sorry

end arithmetic_sequence_first_term_l2321_232182


namespace cube_of_negative_two_times_t_l2321_232135

theorem cube_of_negative_two_times_t (t : ℝ) : (-2 * t)^3 = -8 * t^3 := by
  sorry

end cube_of_negative_two_times_t_l2321_232135


namespace solution_set_supremum_a_l2321_232124

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |x - 2|

-- Theorem 1: The solution set of f(x) > 3
theorem solution_set (x : ℝ) : f x > 3 ↔ x < 0 ∨ x > 3 := by sorry

-- Theorem 2: The supremum of a for which f(x) > a holds for all x
theorem supremum_a : ∀ a : ℝ, (∀ x : ℝ, f x > a) ↔ a < 1 := by sorry

end solution_set_supremum_a_l2321_232124


namespace geometry_propositions_l2321_232152

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (parallelPL : Line → Plane → Prop)
variable (perpendicularPL : Line → Plane → Prop)
variable (parallelPlanes : Plane → Plane → Prop)
variable (perpendicularPlanes : Plane → Plane → Prop)
variable (intersection : Plane → Plane → Line)

-- Theorem statement
theorem geometry_propositions 
  (α β : Plane) (m n : Line) : 
  -- Proposition 2
  (∀ m, perpendicularPL m α ∧ perpendicularPL m β → parallelPlanes α β) ∧
  -- Proposition 3
  (intersection α β = n ∧ parallelPL m α ∧ parallelPL m β → parallel m n) ∧
  -- Proposition 4
  (perpendicularPlanes α β ∧ perpendicularPL m α ∧ perpendicularPL n β → perpendicular m n) :=
by sorry

end geometry_propositions_l2321_232152


namespace kristin_laps_theorem_l2321_232132

/-- Kristin's running speed relative to Sarith's -/
def kristin_speed_ratio : ℚ := 3

/-- Ratio of adult field size to children's field size -/
def field_size_ratio : ℚ := 2

/-- Number of times Sarith went around the children's field -/
def sarith_laps : ℕ := 8

/-- Number of times Kristin went around the adult field -/
def kristin_laps : ℕ := 12

theorem kristin_laps_theorem (speed_ratio : ℚ) (field_ratio : ℚ) (sarith_runs : ℕ) :
  speed_ratio = kristin_speed_ratio →
  field_ratio = field_size_ratio →
  sarith_runs = sarith_laps →
  ↑kristin_laps = ↑sarith_runs * (speed_ratio / field_ratio) := by
  sorry

end kristin_laps_theorem_l2321_232132


namespace charlie_has_32_cards_l2321_232191

/-- The number of soccer cards Chris has -/
def chris_cards : ℕ := 18

/-- The difference in cards between Charlie and Chris -/
def card_difference : ℕ := 14

/-- Charlie's number of soccer cards -/
def charlie_cards : ℕ := chris_cards + card_difference

/-- Theorem stating that Charlie has 32 soccer cards -/
theorem charlie_has_32_cards : charlie_cards = 32 := by
  sorry

end charlie_has_32_cards_l2321_232191


namespace f_has_one_zero_a_equals_one_l2321_232143

noncomputable def f (x : ℝ) : ℝ := Real.exp x - 1 / (x^2)

theorem f_has_one_zero :
  ∃! x, f x = 0 :=
sorry

theorem a_equals_one (a : ℝ) :
  (∀ x > 0, f x ≥ (2 * a * Real.log x) / x^2 + a / x) ↔ a = 1 :=
sorry

end f_has_one_zero_a_equals_one_l2321_232143


namespace max_profit_thermos_l2321_232136

/-- Thermos cup prices and quantities -/
structure ThermosCups where
  price_a : ℝ
  price_b : ℝ
  quantity_a : ℕ
  quantity_b : ℕ

/-- Conditions for thermos cup problem -/
def thermos_conditions (t : ThermosCups) : Prop :=
  t.price_b = t.price_a + 10 ∧
  600 / t.price_b = 480 / t.price_a ∧
  t.quantity_a + t.quantity_b = 120 ∧
  t.quantity_a ≥ t.quantity_b / 2 ∧
  t.quantity_a ≤ t.quantity_b

/-- Profit calculation -/
def profit (t : ThermosCups) : ℝ :=
  (t.price_a - 30) * t.quantity_a + (t.price_b * 0.9 - 30) * t.quantity_b

/-- Theorem: Maximum profit for thermos cup sales -/
theorem max_profit_thermos :
  ∃ t : ThermosCups, thermos_conditions t ∧
    profit t = 1600 ∧
    (∀ t' : ThermosCups, thermos_conditions t' → profit t' ≤ profit t) :=
  sorry

end max_profit_thermos_l2321_232136


namespace odd_number_representation_l2321_232186

theorem odd_number_representation (k : ℤ) : 
  (k % 2 = 1) → 
  ((∃ n : ℤ, 2 * n + 3 = k) ∧ 
   ¬(∀ k : ℤ, k % 2 = 1 → ∃ n : ℤ, 4 * n - 1 = k)) := by
sorry

end odd_number_representation_l2321_232186


namespace ball_probabilities_l2321_232127

/-- Represents the number of red balls initially in the bag -/
def initial_red_balls : ℕ := 5

/-- Represents the number of yellow balls initially in the bag -/
def initial_yellow_balls : ℕ := 10

/-- Represents the total number of balls added to the bag -/
def added_balls : ℕ := 9

/-- Calculates the probability of drawing a red ball -/
def prob_red_ball : ℚ := initial_red_balls / (initial_red_balls + initial_yellow_balls)

/-- Represents the number of red balls added to the bag -/
def red_balls_added : ℕ := 7

/-- Represents the number of yellow balls added to the bag -/
def yellow_balls_added : ℕ := 2

theorem ball_probabilities :
  (prob_red_ball = 1/3) ∧
  ((initial_red_balls + red_balls_added) / (initial_red_balls + initial_yellow_balls + added_balls) =
   (initial_yellow_balls + yellow_balls_added) / (initial_red_balls + initial_yellow_balls + added_balls)) :=
by sorry

end ball_probabilities_l2321_232127


namespace complex_subtraction_magnitude_l2321_232105

theorem complex_subtraction_magnitude : 
  Complex.abs ((3 - 10 * Complex.I) - (2 + 5 * Complex.I)) = Real.sqrt 26 := by
  sorry

end complex_subtraction_magnitude_l2321_232105


namespace no_integer_solutions_for_equation_l2321_232164

theorem no_integer_solutions_for_equation :
  ¬ ∃ (x y z t : ℤ), x^2 + y^2 + z^2 = 8*t - 1 :=
by sorry

end no_integer_solutions_for_equation_l2321_232164


namespace total_spider_legs_l2321_232187

/-- The number of spiders in Christopher's room -/
def num_spiders : ℕ := 4

/-- The number of legs each spider has -/
def legs_per_spider : ℕ := 8

/-- Theorem: The total number of spider legs is 32 -/
theorem total_spider_legs : num_spiders * legs_per_spider = 32 := by
  sorry

end total_spider_legs_l2321_232187


namespace slower_bike_speed_l2321_232177

theorem slower_bike_speed 
  (distance : ℝ) 
  (fast_speed : ℝ) 
  (time_difference : ℝ) 
  (h1 : distance = 960) 
  (h2 : fast_speed = 64) 
  (h3 : time_difference = 1) :
  ∃ (slow_speed : ℝ), 
    slow_speed > 0 ∧ 
    distance / slow_speed = distance / fast_speed + time_difference ∧ 
    slow_speed = 60 := by
sorry

end slower_bike_speed_l2321_232177


namespace probability_of_black_ball_l2321_232106

theorem probability_of_black_ball (p_red p_white p_black : ℝ) : 
  p_red = 0.43 → p_white = 0.27 → p_red + p_white + p_black = 1 → p_black = 0.3 := by
  sorry

end probability_of_black_ball_l2321_232106


namespace angle_value_for_given_function_l2321_232163

/-- Given a function f(x) = sin x + √3 * cos x, prove that if there exists an acute angle θ
    such that f(θ) = 2, then θ = π/6 -/
theorem angle_value_for_given_function (θ : Real) :
  (∃ f : Real → Real, f = λ x => Real.sin x + Real.sqrt 3 * Real.cos x) →
  (0 < θ ∧ θ < π / 2) →
  (∃ f : Real → Real, f θ = 2) →
  θ = π / 6 := by
  sorry

end angle_value_for_given_function_l2321_232163


namespace candy_box_solution_l2321_232112

/-- Represents the number of candies of each type in a box -/
structure CandyBox where
  chocolate : ℕ
  hard : ℕ
  jelly : ℕ

/-- Conditions for the candy box problem -/
def CandyBoxConditions (box : CandyBox) : Prop :=
  (box.chocolate + box.hard + box.jelly = 110) ∧
  (box.chocolate + box.hard = 100) ∧
  (box.hard + box.jelly = box.chocolate + box.jelly + 20)

/-- Theorem stating the solution to the candy box problem -/
theorem candy_box_solution :
  ∃ (box : CandyBox), CandyBoxConditions box ∧ 
    box.chocolate = 40 ∧ box.hard = 60 ∧ box.jelly = 10 := by
  sorry

end candy_box_solution_l2321_232112


namespace sequence_properties_l2321_232197

/-- Given a sequence {a_n} with partial sum S_n satisfying 3a_n - 2S_n = 2 for all n,
    prove the general term formula and a property of partial sums. -/
theorem sequence_properties (a : ℕ → ℝ) (S : ℕ → ℝ) 
    (h : ∀ n, 3 * a n - 2 * S n = 2) : 
    (∀ n, a n = 2 * 3^(n-1)) ∧ 
    (∀ n, S (n+1)^2 - S n * S (n+2) = 4 * 3^n) := by
  sorry

end sequence_properties_l2321_232197


namespace f_extrema_l2321_232199

open Real

noncomputable def f (x : ℝ) : ℝ := cos x + (x + 1) * sin x + 1

theorem f_extrema :
  ∃ (min max : ℝ),
    (∀ x ∈ Set.Icc 0 (2 * π), f x ≥ min) ∧
    (∃ x ∈ Set.Icc 0 (2 * π), f x = min) ∧
    (∀ x ∈ Set.Icc 0 (2 * π), f x ≤ max) ∧
    (∃ x ∈ Set.Icc 0 (2 * π), f x = max) ∧
    min = -3 * π / 2 ∧
    max = π / 2 + 2 := by
  sorry

end f_extrema_l2321_232199


namespace books_per_shelf_l2321_232147

theorem books_per_shelf (total_books : ℕ) (books_taken : ℕ) (shelves : ℕ) : 
  total_books = 14 → books_taken = 2 → shelves = 4 → 
  (total_books - books_taken) / shelves = 3 := by
  sorry

end books_per_shelf_l2321_232147


namespace z_mod_nine_l2321_232130

theorem z_mod_nine (z : ℤ) (h : ∃ k : ℤ, (z + 3) / 9 = k) : z % 9 = 6 := by
  sorry

end z_mod_nine_l2321_232130


namespace heather_blocks_l2321_232110

/-- The number of blocks Heather ends up with after sharing -/
def blocks_remaining (initial : ℕ) (shared : ℕ) : ℕ :=
  initial - shared

/-- Theorem stating that Heather ends up with 45 blocks -/
theorem heather_blocks : blocks_remaining 86 41 = 45 := by
  sorry

end heather_blocks_l2321_232110


namespace can_capacity_is_30_liters_l2321_232162

/-- Represents the contents of a can with milk and water -/
structure CanContents where
  milk : ℝ
  water : ℝ

/-- The capacity of the can in liters -/
def canCapacity : ℝ := 30

/-- The amount of milk added in liters -/
def milkAdded : ℝ := 10

/-- Checks if the given contents match the initial ratio of 4:3 -/
def isInitialRatio (contents : CanContents) : Prop :=
  contents.milk / contents.water = 4 / 3

/-- Checks if the given contents match the final ratio of 5:2 -/
def isFinalRatio (contents : CanContents) : Prop :=
  contents.milk / contents.water = 5 / 2

/-- Theorem stating that given the conditions, the can capacity is 30 liters -/
theorem can_capacity_is_30_liters 
  (initialContents : CanContents) 
  (hInitialRatio : isInitialRatio initialContents)
  (hFinalRatio : isFinalRatio { milk := initialContents.milk + milkAdded, water := initialContents.water })
  (hFull : initialContents.milk + initialContents.water + milkAdded = canCapacity) : 
  canCapacity = 30 := by
  sorry


end can_capacity_is_30_liters_l2321_232162


namespace quadratic_function_bound_l2321_232196

/-- Theorem: Bound on quadratic function -/
theorem quadratic_function_bound (a b c : ℝ) (ha : a > 0) (hb : b ≠ 0)
  (hf0 : |a * 0^2 + b * 0 + c| ≤ 1)
  (hfn1 : |a * (-1)^2 + b * (-1) + c| ≤ 1)
  (hf1 : |a * 1^2 + b * 1 + c| ≤ 1)
  (hba : |b| ≤ a) :
  ∀ x : ℝ, |x| ≤ 1 → |a * x^2 + b * x + c| ≤ 5/4 := by
  sorry

end quadratic_function_bound_l2321_232196


namespace complement_of_M_in_U_l2321_232169

def U : Set Nat := {1,2,3,4,5,6}
def M : Set Nat := {1,2,4}

theorem complement_of_M_in_U :
  (U \ M) = {3,5,6} := by sorry

end complement_of_M_in_U_l2321_232169


namespace angle_B_measure_l2321_232150

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : ℝ)
  (sum_angles : A + B + C + D = 360)

-- Define the theorem
theorem angle_B_measure (q : Quadrilateral) (h : q.A + q.C = 150) : q.B = 105 := by
  sorry

end angle_B_measure_l2321_232150


namespace distance_AB_l2321_232115

/-- The equation of the line -/
def line_equation (x y : ℝ) : Prop := x - 2*y + 6 = 0

/-- The x-coordinate of point A (x-axis intersection) -/
def point_A : ℝ := -6

/-- The y-coordinate of point B (y-axis intersection) -/
def point_B : ℝ := 3

/-- Theorem stating that the distance between points A and B is 3√5 -/
theorem distance_AB :
  let A : ℝ × ℝ := (point_A, 0)
  let B : ℝ × ℝ := (0, point_B)
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 3 * Real.sqrt 5 :=
sorry

end distance_AB_l2321_232115


namespace cube_root_sixteen_to_sixth_l2321_232168

theorem cube_root_sixteen_to_sixth (x : ℝ) : x = (16 ^ (1/3 : ℝ)) → x^6 = 256 := by
  sorry

end cube_root_sixteen_to_sixth_l2321_232168


namespace class_artworks_l2321_232192

theorem class_artworks (num_students : ℕ) (artworks_group1 : ℕ) (artworks_group2 : ℕ) : 
  num_students = 10 →
  artworks_group1 = 3 →
  artworks_group2 = 4 →
  (num_students / 2 : ℕ) * artworks_group1 + (num_students / 2 : ℕ) * artworks_group2 = 35 := by
  sorry

end class_artworks_l2321_232192


namespace sqrt_15_minus_1_range_l2321_232189

theorem sqrt_15_minus_1_range : 2 < Real.sqrt 15 - 1 ∧ Real.sqrt 15 - 1 < 3 := by
  sorry

end sqrt_15_minus_1_range_l2321_232189


namespace apple_production_total_l2321_232133

/-- The number of apples produced by a tree over three years -/
def appleProduction : ℕ → ℕ
| 1 => 40
| 2 => 2 * appleProduction 1 + 8
| 3 => appleProduction 2 - (appleProduction 2 / 4)
| _ => 0

/-- The total number of apples produced over three years -/
def totalApples : ℕ := appleProduction 1 + appleProduction 2 + appleProduction 3

theorem apple_production_total : totalApples = 194 := by
  sorry

end apple_production_total_l2321_232133


namespace partial_fraction_decomposition_l2321_232184

theorem partial_fraction_decomposition (x : ℝ) (h1 : x ≠ 6) (h2 : x ≠ -5) :
  (7 * x + 11) / (x^2 - x - 30) = (53 / 11) / (x - 6) + (24 / 11) / (x + 5) := by
  sorry

end partial_fraction_decomposition_l2321_232184


namespace travel_time_equation_l2321_232181

theorem travel_time_equation (x : ℝ) : x > 3 → 
  (30 / (x - 3) - 30 / x = 40 / 60) ↔ 
  (30 = (x - 3) * (40 / 60) ∧ 30 = x * ((40 / 60) + (30 / (x - 3)))) := by
  sorry

#check travel_time_equation

end travel_time_equation_l2321_232181


namespace solve_system_l2321_232142

theorem solve_system (p q : ℚ) 
  (eq1 : 5 * p + 6 * q = 17) 
  (eq2 : 6 * p + 5 * q = 20) : 
  q = 2 / 11 := by
sorry

end solve_system_l2321_232142


namespace seed_germination_percentage_l2321_232161

theorem seed_germination_percentage 
  (seeds_plot1 : ℕ) 
  (seeds_plot2 : ℕ) 
  (germination_rate_plot1 : ℚ) 
  (germination_rate_plot2 : ℚ) 
  (h1 : seeds_plot1 = 300)
  (h2 : seeds_plot2 = 200)
  (h3 : germination_rate_plot1 = 25 / 100)
  (h4 : germination_rate_plot2 = 30 / 100) :
  (seeds_plot1 * germination_rate_plot1 + seeds_plot2 * germination_rate_plot2) / 
  (seeds_plot1 + seeds_plot2) = 27 / 100 := by
sorry

end seed_germination_percentage_l2321_232161


namespace cat_adoption_cost_l2321_232118

/-- The cost to get each cat ready for adoption -/
def cat_cost : ℝ := 50

/-- The cost to get each adult dog ready for adoption -/
def adult_dog_cost : ℝ := 100

/-- The cost to get each puppy ready for adoption -/
def puppy_cost : ℝ := 150

/-- The number of cats adopted -/
def num_cats : ℕ := 2

/-- The number of adult dogs adopted -/
def num_adult_dogs : ℕ := 3

/-- The number of puppies adopted -/
def num_puppies : ℕ := 2

/-- The total cost to get all adopted animals ready -/
def total_cost : ℝ := 700

/-- Theorem stating that the cost to get each cat ready for adoption is $50 -/
theorem cat_adoption_cost : 
  cat_cost * num_cats + adult_dog_cost * num_adult_dogs + puppy_cost * num_puppies = total_cost :=
by sorry

end cat_adoption_cost_l2321_232118


namespace sphere_volume_ratio_l2321_232148

theorem sphere_volume_ratio (r R : ℝ) (h : r > 0) (H : R > 0) : 
  (4 * π * r^2) / (4 * π * R^2) = 4 / 9 → 
  ((4 / 3) * π * r^3) / ((4 / 3) * π * R^3) = 8 / 27 := by
sorry

end sphere_volume_ratio_l2321_232148


namespace transportation_cost_comparison_l2321_232129

/-- The cost function for company A -/
def cost_A (x : ℝ) : ℝ := 0.6 * x

/-- The cost function for company B -/
def cost_B (x : ℝ) : ℝ := 0.3 * x + 750

theorem transportation_cost_comparison (x : ℝ) 
  (h_x_pos : 0 < x) (h_x_upper : x < 5000) :
  (x < 2500 → cost_A x < cost_B x) ∧
  (x > 2500 → cost_B x < cost_A x) ∧
  (x = 2500 → cost_A x = cost_B x) := by
  sorry


end transportation_cost_comparison_l2321_232129


namespace middle_number_of_seven_consecutive_l2321_232108

def is_middle_of_seven_consecutive (n : ℕ) : Prop :=
  ∃ (a : ℕ), a + (a + 1) + (a + 2) + n + (n + 1) + (n + 2) + (n + 3) = 63

theorem middle_number_of_seven_consecutive :
  ∃ (n : ℕ), is_middle_of_seven_consecutive n ∧ n = 9 := by
  sorry

end middle_number_of_seven_consecutive_l2321_232108


namespace set_operations_and_intersection_intersection_empty_iff_m_range_l2321_232144

def A : Set ℝ := {x | -4 < x ∧ x < 2}
def B : Set ℝ := {x | x < -5 ∨ x > 1}
def C (m : ℝ) : Set ℝ := {x | m - 1 ≤ x ∧ x ≤ m + 1}

theorem set_operations_and_intersection :
  (A ∪ B = {x | x < -5 ∨ x > -4}) ∧
  (A ∩ (Set.univ \ B) = {x | -4 < x ∧ x ≤ 1}) := by sorry

theorem intersection_empty_iff_m_range (m : ℝ) :
  (B ∩ C m = ∅) ↔ (-4 ≤ m ∧ m ≤ 0) := by sorry

end set_operations_and_intersection_intersection_empty_iff_m_range_l2321_232144


namespace all_left_probability_l2321_232125

/-- Represents the particle movement experiment -/
structure ParticleExperiment where
  total_particles : ℕ
  initial_left : ℕ
  initial_right : ℕ

/-- The probability of all particles ending on the left side -/
def probability_all_left (exp : ParticleExperiment) : ℚ :=
  1 / 2

/-- The main theorem stating the probability of all particles ending on the left side -/
theorem all_left_probability (exp : ParticleExperiment) 
  (h1 : exp.total_particles = 100)
  (h2 : exp.initial_left = 32)
  (h3 : exp.initial_right = 68)
  (h4 : exp.initial_left + exp.initial_right = exp.total_particles) :
  probability_all_left exp = 1 / 2 := by
  sorry

#eval (100 * 1 + 2 : ℕ)

end all_left_probability_l2321_232125


namespace non_shaded_perimeter_l2321_232175

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.width * r.height

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.width + r.height)

theorem non_shaded_perimeter (outer : Rectangle) (attached : Rectangle) (shaded : Rectangle) 
  (h_outer_width : outer.width = 12)
  (h_outer_height : outer.height = 10)
  (h_attached_width : attached.width = 3)
  (h_attached_height : attached.height = 4)
  (h_shaded_width : shaded.width = 3)
  (h_shaded_height : shaded.height = 5)
  (h_shaded_area : area shaded = 120)
  (h_shaded_center : shaded.width < outer.width ∧ shaded.height < outer.height) :
  ∃ (non_shaded : Rectangle), perimeter non_shaded = 19 := by
sorry

end non_shaded_perimeter_l2321_232175


namespace tennis_racket_price_l2321_232128

theorem tennis_racket_price
  (sneakers_cost sports_outfit_cost total_spent : ℝ)
  (racket_discount sales_tax : ℝ)
  (h1 : sneakers_cost = 200)
  (h2 : sports_outfit_cost = 250)
  (h3 : racket_discount = 0.2)
  (h4 : sales_tax = 0.1)
  (h5 : total_spent = 750)
  : ∃ (original_price : ℝ),
    (1 + sales_tax) * ((1 - racket_discount) * original_price + sneakers_cost + sports_outfit_cost) = total_spent ∧
    original_price = 255 / 0.88 :=
by sorry

end tennis_racket_price_l2321_232128


namespace negative_power_division_l2321_232126

theorem negative_power_division : -2^5 / (-2)^3 = 4 := by sorry

end negative_power_division_l2321_232126


namespace malfunction_time_proof_l2321_232103

/-- Represents a time in HH:MM format -/
structure Time where
  hours : Nat
  minutes : Nat
  hh_valid : hours < 24
  mm_valid : minutes < 60

/-- Represents a malfunctioning clock where each digit changed by ±1 -/
def is_malfunction (original : Time) (displayed : Time) : Prop :=
  (displayed.hours / 10 = original.hours / 10 + 1 ∨ displayed.hours / 10 = original.hours / 10 - 1) ∧
  (displayed.hours % 10 = (original.hours % 10 + 1) % 10 ∨ displayed.hours % 10 = (original.hours % 10 - 1 + 10) % 10) ∧
  (displayed.minutes / 10 = original.minutes / 10 + 1 ∨ displayed.minutes / 10 = original.minutes / 10 - 1) ∧
  (displayed.minutes % 10 = (original.minutes % 10 + 1) % 10 ∨ displayed.minutes % 10 = (original.minutes % 10 - 1 + 10) % 10)

theorem malfunction_time_proof (displayed : Time) 
  (h_displayed : displayed.hours = 20 ∧ displayed.minutes = 9) :
  ∃ (original : Time), is_malfunction original displayed ∧ original.hours = 11 ∧ original.minutes = 18 := by
  sorry

end malfunction_time_proof_l2321_232103


namespace power_relations_l2321_232165

theorem power_relations (a m n : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : a^m = 4) (h4 : a^n = 3) :
  a^(-m/2) = 1/2 ∧ a^(2*m-n) = 16/3 := by
  sorry

end power_relations_l2321_232165


namespace string_length_problem_l2321_232117

theorem string_length_problem (total_length remaining_length used_length : ℝ) : 
  total_length = 90 →
  remaining_length = total_length - 30 →
  used_length = (8 / 15) * remaining_length →
  used_length = 32 := by
sorry

end string_length_problem_l2321_232117


namespace max_attendance_l2321_232166

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
  | Diana

-- Define the availability function
def isAvailable (p : Person) (d : Day) : Bool :=
  match p, d with
  | Person.Anna, Day.Monday => false
  | Person.Anna, Day.Tuesday => true
  | Person.Anna, Day.Wednesday => false
  | Person.Anna, Day.Thursday => true
  | Person.Anna, Day.Friday => false
  | Person.Bill, Day.Monday => true
  | Person.Bill, Day.Tuesday => false
  | Person.Bill, Day.Wednesday => true
  | Person.Bill, Day.Thursday => false
  | Person.Bill, Day.Friday => false
  | Person.Carl, Day.Monday => false
  | Person.Carl, Day.Tuesday => false
  | Person.Carl, Day.Wednesday => true
  | Person.Carl, Day.Thursday => false
  | Person.Carl, Day.Friday => true
  | Person.Diana, Day.Monday => true
  | Person.Diana, Day.Tuesday => true
  | Person.Diana, Day.Wednesday => false
  | Person.Diana, Day.Thursday => true
  | Person.Diana, Day.Friday => false

-- Define the function to count available people on a given day
def countAvailable (d : Day) : Nat :=
  (List.filter (fun p => isAvailable p d) [Person.Anna, Person.Bill, Person.Carl, Person.Diana]).length

-- Theorem statement
theorem max_attendance :
  (∀ d : Day, countAvailable d ≤ 2) ∧
  (countAvailable Day.Monday = 2) ∧
  (countAvailable Day.Tuesday = 2) ∧
  (countAvailable Day.Wednesday = 2) ∧
  (countAvailable Day.Thursday < 2) ∧
  (countAvailable Day.Friday < 2) :=
sorry

end max_attendance_l2321_232166


namespace right_triangle_hypotenuse_segments_ratio_l2321_232116

theorem right_triangle_hypotenuse_segments_ratio 
  (a b c : ℝ) 
  (h_right : a^2 + b^2 = c^2) 
  (h_ratio : a / b = 3 / 4) :
  let d := (a * c) / (a + b)
  (c - d) / d = 16 / 9 := by sorry

end right_triangle_hypotenuse_segments_ratio_l2321_232116


namespace festival_selection_probability_l2321_232154

-- Define the number of festivals
def total_festivals : ℕ := 5

-- Define the number of festivals to be selected
def selected_festivals : ℕ := 2

-- Define the number of specific festivals we're interested in
def specific_festivals : ℕ := 2

-- Define the probability of selecting at least one of the specific festivals
def probability : ℚ := 0.7

-- Theorem statement
theorem festival_selection_probability :
  1 - (Nat.choose (total_festivals - specific_festivals) selected_festivals) / 
      (Nat.choose total_festivals selected_festivals) = probability := by
  sorry

end festival_selection_probability_l2321_232154


namespace most_likely_genotype_combination_l2321_232195

/-- Represents the alleles for rabbit fur type -/
inductive Allele
| H  -- Dominant hairy allele
| h  -- Recessive hairy allele
| S  -- Dominant smooth allele
| s  -- Recessive smooth allele

/-- Represents the genotype of a rabbit -/
structure Genotype where
  allele1 : Allele
  allele2 : Allele

/-- Determines if a rabbit has hairy fur based on its genotype -/
def hasHairyFur (g : Genotype) : Bool :=
  match g.allele1, g.allele2 with
  | Allele.H, _ => true
  | _, Allele.H => true
  | _, _ => false

/-- The probability of the hairy fur allele in the population -/
def p : ℝ := 0.1

/-- Represents the result of mating two rabbits -/
structure MatingResult where
  parent1 : Genotype
  parent2 : Genotype
  offspringCount : Nat
  allOffspringHairy : Bool

/-- The theorem to be proved -/
theorem most_likely_genotype_combination (result : MatingResult) 
  (h1 : result.parent1.allele1 = Allele.H ∨ result.parent1.allele2 = Allele.H)
  (h2 : result.parent2.allele1 = Allele.S ∨ result.parent2.allele2 = Allele.S)
  (h3 : result.offspringCount = 4)
  (h4 : result.allOffspringHairy = true) :
  (result.parent1 = Genotype.mk Allele.H Allele.H ∧ 
   result.parent2 = Genotype.mk Allele.S Allele.h) :=
sorry

end most_likely_genotype_combination_l2321_232195


namespace quotient_problem_l2321_232160

theorem quotient_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) 
  (h1 : dividend = 165)
  (h2 : divisor = 18)
  (h3 : remainder = 3)
  (h4 : dividend = quotient * divisor + remainder) :
  quotient = 9 := by
  sorry

end quotient_problem_l2321_232160


namespace problem_statement_l2321_232156

theorem problem_statement (x : ℝ) (h : x + 1/x = 3) :
  (x - 3)^2 + 16/((x - 3)^2) = 7 := by
  sorry

end problem_statement_l2321_232156

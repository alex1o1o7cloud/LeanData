import Mathlib

namespace lattice_points_count_l2680_268036

/-- The number of lattice points on a line segment with given integer endpoints -/
def countLatticePoints (x1 y1 x2 y2 : ℤ) : ℕ :=
  sorry

/-- Theorem: The number of lattice points on the line segment from (5,13) to (47,275) is 3 -/
theorem lattice_points_count : countLatticePoints 5 13 47 275 = 3 := by
  sorry

end lattice_points_count_l2680_268036


namespace weight_difference_l2680_268073

/-- Proves that Heather is 53.4 pounds lighter than Emily, Elizabeth, and George combined -/
theorem weight_difference (heather emily elizabeth george : ℝ) 
  (h1 : heather = 87.5)
  (h2 : emily = 45.3)
  (h3 : elizabeth = 38.7)
  (h4 : george = 56.9) :
  heather - (emily + elizabeth + george) = -53.4 := by
  sorry

end weight_difference_l2680_268073


namespace sphere_radius_from_surface_area_l2680_268015

theorem sphere_radius_from_surface_area (S : ℝ) (r : ℝ) (h : S = 4 * Real.pi) :
  S = 4 * Real.pi * r^2 → r = 1 := by
  sorry

end sphere_radius_from_surface_area_l2680_268015


namespace common_chord_length_l2680_268014

theorem common_chord_length (r₁ r₂ d : ℝ) (h₁ : r₁ = 8) (h₂ : r₂ = 12) (h₃ : d = 20) :
  let chord_length := 2 * Real.sqrt (r₂^2 - (d/2)^2)
  chord_length = 4 * Real.sqrt 11 := by
  sorry

end common_chord_length_l2680_268014


namespace fraction_inequality_l2680_268028

theorem fraction_inequality (x : ℝ) : (x - 1) / (x + 2) ≥ 0 ↔ x < -2 ∨ x ≥ 1 := by
  sorry

end fraction_inequality_l2680_268028


namespace solve_equation_l2680_268047

theorem solve_equation (x : ℚ) (h : (1/3 - 1/4) * 2 = 1/x) : x = 6 := by
  sorry

end solve_equation_l2680_268047


namespace a_100_value_l2680_268097

/-- Sequence S defined recursively -/
def S : ℕ → ℚ
| 0 => 0
| 1 => 3
| (n + 2) => 3 / (3 * n + 1)

/-- Sequence a defined in terms of S -/
def a : ℕ → ℚ
| 0 => 0
| 1 => 3
| (n + 2) => (3 * (S (n + 2))^2) / (3 * S (n + 2) - 2)

/-- Main theorem: a₁₀₀ = -9/84668 -/
theorem a_100_value : a 100 = -9/84668 := by sorry

end a_100_value_l2680_268097


namespace lattice_points_form_square_l2680_268005

-- Define a structure for a point in the plane
structure Point :=
  (x : ℤ)
  (y : ℤ)

-- Define a function to calculate the squared distance between two points
def squaredDistance (p q : Point) : ℤ :=
  (p.x - q.x)^2 + (p.y - q.y)^2

-- Define a function to calculate the area of a triangle given three points
def areaOfTriangle (p q r : Point) : ℚ :=
  let a := squaredDistance p q
  let b := squaredDistance q r
  let c := squaredDistance r p
  ((a + b + c)^2 - 2 * (a^2 + b^2 + c^2)) / 16

-- Theorem statement
theorem lattice_points_form_square (p q r : Point) 
  (h_distinct : p ≠ q ∧ q ≠ r ∧ r ≠ p)
  (h_inequality : (squaredDistance p q).sqrt + (squaredDistance q r).sqrt < (8 * areaOfTriangle p q r + 1).sqrt) :
  ∃ s : Point, s ≠ p ∧ s ≠ q ∧ s ≠ r ∧ 
    squaredDistance p q = squaredDistance q r ∧
    squaredDistance r s = squaredDistance s p ∧
    squaredDistance p q = squaredDistance r s :=
sorry

end lattice_points_form_square_l2680_268005


namespace andrews_numbers_l2680_268048

theorem andrews_numbers (x y : ℤ) : 
  3 * x + 4 * y = 161 → (x = 17 ∨ y = 17) → (x = 31 ∨ y = 31) := by
  sorry

end andrews_numbers_l2680_268048


namespace lindas_savings_l2680_268071

theorem lindas_savings (savings : ℝ) : 
  savings > 0 →
  (0.9 * (3/8) * savings) + (0.85 * (1/4) * savings) + 450 = savings →
  savings = 1000 := by
sorry

end lindas_savings_l2680_268071


namespace goats_bought_l2680_268065

theorem goats_bought (total_cost : ℕ) (cow_price goat_price : ℕ) (num_cows : ℕ) :
  total_cost = 1400 →
  cow_price = 460 →
  goat_price = 60 →
  num_cows = 2 →
  ∃ (num_goats : ℕ), num_goats = 8 ∧ total_cost = num_cows * cow_price + num_goats * goat_price :=
by sorry

end goats_bought_l2680_268065


namespace cosine_value_l2680_268056

theorem cosine_value (α : Real) 
  (h1 : π / 2 < α) 
  (h2 : α < π) 
  (h3 : Real.sin (α + π / 6) = 3 / 5) : 
  Real.cos (α - π / 6) = (3 * Real.sqrt 3 - 4) / 10 := by
  sorry

end cosine_value_l2680_268056


namespace train_crossing_time_l2680_268016

/-- Proves that a train 75 meters long, traveling at 54 km/hr, will take 5 seconds to cross an electric pole. -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 75 →
  train_speed_kmh = 54 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) →
  crossing_time = 5 := by
  sorry

#check train_crossing_time

end train_crossing_time_l2680_268016


namespace line_passes_through_point_l2680_268011

/-- The line equation mx - y + 1 - m = 0 passes through the point (1,1) for all real m -/
theorem line_passes_through_point (m : ℝ) : m * 1 - 1 + 1 - m = 0 := by
  sorry

end line_passes_through_point_l2680_268011


namespace girls_combined_avg_is_76_l2680_268013

-- Define the schools
inductive School
| Cedar
| Dale

-- Define the student types
inductive StudentType
| Boy
| Girl

-- Define the average score function
def avg_score (s : School) (st : StudentType) : ℝ :=
  match s, st with
  | School.Cedar, StudentType.Boy => 65
  | School.Cedar, StudentType.Girl => 70
  | School.Dale, StudentType.Boy => 75
  | School.Dale, StudentType.Girl => 82

-- Define the combined average score function for each school
def combined_avg_score (s : School) : ℝ :=
  match s with
  | School.Cedar => 68
  | School.Dale => 78

-- Define the combined average score for boys at both schools
def combined_boys_avg : ℝ := 73

-- Theorem to prove
theorem girls_combined_avg_is_76 :
  ∃ (c d : ℝ), c > 0 ∧ d > 0 ∧
  (c * avg_score School.Cedar StudentType.Boy + d * avg_score School.Dale StudentType.Boy) / (c + d) = combined_boys_avg ∧
  (c * combined_avg_score School.Cedar + d * combined_avg_score School.Dale) / (c + d) = (c * avg_score School.Cedar StudentType.Girl + d * avg_score School.Dale StudentType.Girl) / (c + d) ∧
  (avg_score School.Cedar StudentType.Girl + avg_score School.Dale StudentType.Girl) / 2 = 76 :=
sorry

end girls_combined_avg_is_76_l2680_268013


namespace cube_root_minus_square_root_plus_abs_l2680_268030

theorem cube_root_minus_square_root_plus_abs : 
  ((-8 : ℝ) ^ (1/3 : ℝ)) - Real.sqrt ((-3 : ℝ)^2) + |Real.sqrt 2 - 1| = Real.sqrt 2 - 6 := by
  sorry

end cube_root_minus_square_root_plus_abs_l2680_268030


namespace eight_S_three_l2680_268027

-- Define the operation §
def S (a b : ℤ) : ℤ := 4*a + 7*b

-- Theorem to prove
theorem eight_S_three : S 8 3 = 53 := by
  sorry

end eight_S_three_l2680_268027


namespace properties_of_negative_three_halves_l2680_268054

def x : ℚ := -3/2

theorem properties_of_negative_three_halves :
  (- x = 3/2) ∧ 
  (x⁻¹ = -2/3) ∧ 
  (|x| = 3/2) := by sorry

end properties_of_negative_three_halves_l2680_268054


namespace equidistant_points_l2680_268009

def equidistant (p q : ℝ × ℝ) : Prop :=
  max (|p.1|) (|p.2|) = max (|q.1|) (|q.2|)

theorem equidistant_points :
  (equidistant (-3, 7) (3, -7) ∧ equidistant (-3, 7) (7, 4)) ∧
  (equidistant (-4, 2) (-4, -3) ∧ equidistant (-4, 2) (3, 4)) :=
by sorry

end equidistant_points_l2680_268009


namespace light_travel_distance_l2680_268092

/-- The distance light travels in one year in kilometers -/
def light_year_distance : ℝ := 9460800000000

/-- The number of years we're calculating for -/
def years : ℕ := 50

/-- The expected distance light travels in 50 years -/
def expected_distance : ℝ := 473.04 * (10 ^ 12)

/-- Theorem stating that the distance light travels in 50 years is equal to the expected distance -/
theorem light_travel_distance : light_year_distance * (years : ℝ) = expected_distance := by
  sorry

end light_travel_distance_l2680_268092


namespace plane_speed_theorem_l2680_268051

/-- Given a plane's speed against a tailwind and the tailwind speed, 
    calculate the plane's speed with the tailwind. -/
def plane_speed_with_tailwind (speed_against_tailwind : ℝ) (tailwind_speed : ℝ) : ℝ :=
  speed_against_tailwind + 2 * tailwind_speed

/-- Theorem: The plane's speed with the tailwind is 460 mph given the conditions. -/
theorem plane_speed_theorem (speed_against_tailwind : ℝ) (tailwind_speed : ℝ) 
  (h1 : speed_against_tailwind = 310)
  (h2 : tailwind_speed = 75) :
  plane_speed_with_tailwind speed_against_tailwind tailwind_speed = 460 := by
  sorry

#eval plane_speed_with_tailwind 310 75

end plane_speed_theorem_l2680_268051


namespace bookstore_profit_rate_l2680_268061

/-- Calculates the overall rate of profit for three books given their cost and selling prices -/
theorem bookstore_profit_rate 
  (cost_A selling_A cost_B selling_B cost_C selling_C : ℚ) 
  (h1 : cost_A = 50) (h2 : selling_A = 70)
  (h3 : cost_B = 80) (h4 : selling_B = 100)
  (h5 : cost_C = 150) (h6 : selling_C = 180) :
  (selling_A - cost_A + selling_B - cost_B + selling_C - cost_C) / 
  (cost_A + cost_B + cost_C) * 100 = 25 := by
  sorry

end bookstore_profit_rate_l2680_268061


namespace chicken_buying_equation_l2680_268088

/-- Represents the scenario of a group buying chickens -/
structure ChickenBuying where
  people : ℕ
  cost : ℕ

/-- The excess when each person contributes 9 coins -/
def excess (cb : ChickenBuying) : ℤ :=
  9 * cb.people - cb.cost

/-- The shortage when each person contributes 6 coins -/
def shortage (cb : ChickenBuying) : ℤ :=
  cb.cost - 6 * cb.people

/-- The theorem representing the chicken buying scenario -/
theorem chicken_buying_equation (cb : ChickenBuying) 
  (h1 : excess cb = 11) 
  (h2 : shortage cb = 16) : 
  9 * cb.people - 11 = 6 * cb.people + 16 := by
  sorry

end chicken_buying_equation_l2680_268088


namespace room_length_is_five_l2680_268034

/-- Given a rectangular room with known width, total paving cost, and paving rate per square meter,
    prove that the length of the room is 5 meters. -/
theorem room_length_is_five (width : ℝ) (total_cost : ℝ) (rate_per_sqm : ℝ) :
  width = 4.75 →
  total_cost = 21375 →
  rate_per_sqm = 900 →
  (total_cost / rate_per_sqm) / width = 5 := by
  sorry

end room_length_is_five_l2680_268034


namespace intersection_M_N_l2680_268058

noncomputable def M : Set (ℝ × ℝ) := {p | p.2 = Real.sqrt (p.1 - 1)}

noncomputable def N : Set (ℝ × ℝ) := {p | p.2 = Real.log p.1}

theorem intersection_M_N :
  ∃! a : ℝ, a > 1 ∧ Real.sqrt (a - 1) = Real.log a ∧
  M ∩ N = {(a, Real.log a)} := by
  sorry

end intersection_M_N_l2680_268058


namespace negate_negative_equals_positive_l2680_268091

theorem negate_negative_equals_positive (n : ℤ) : -(-n) = n := by
  sorry

end negate_negative_equals_positive_l2680_268091


namespace smallest_a_value_l2680_268066

/-- Given two quadratic equations with integer roots less than -1, find the smallest possible 'a' -/
theorem smallest_a_value (a b c : ℤ) : 
  (∃ x y : ℤ, x < -1 ∧ y < -1 ∧ x^2 + b*x + a = 0 ∧ y^2 + b*y + a = 0) →
  (∃ z w : ℤ, z < -1 ∧ w < -1 ∧ z^2 + c*z + a = 1 ∧ w^2 + c*w + a = 1) →
  (∀ a' b' c' : ℤ, 
    (∃ x y : ℤ, x < -1 ∧ y < -1 ∧ x^2 + b'*x + a' = 0 ∧ y^2 + b'*y + a' = 0) →
    (∃ z w : ℤ, z < -1 ∧ w < -1 ∧ z^2 + c'*z + a' = 1 ∧ w^2 + c'*w + a' = 1) →
    a' ≥ a) →
  a = 15 :=
sorry

end smallest_a_value_l2680_268066


namespace eight_digit_divisibility_l2680_268020

-- Define a four-digit number
def four_digit_number (a b c d : ℕ) : ℕ := a * 1000 + b * 100 + c * 10 + d

-- Define the eight-digit number formed by repeating the four-digit number
def eight_digit_number (a b c d : ℕ) : ℕ := four_digit_number a b c d * 10000 + four_digit_number a b c d

-- Theorem statement
theorem eight_digit_divisibility (a b c d : ℕ) :
  (a < 10) → (b < 10) → (c < 10) → (d < 10) →
  (∃ k₁ k₂ : ℕ, eight_digit_number a b c d = 73 * k₁ ∧ eight_digit_number a b c d = 137 * k₂) := by
  sorry


end eight_digit_divisibility_l2680_268020


namespace two_digit_number_property_l2680_268076

theorem two_digit_number_property (a b j m : ℕ) (h1 : a < 10) (h2 : b < 10) 
  (h3 : 10 * a + b = j * (a^2 + b^2)) (h4 : 10 * b + a = m * (a^2 + b^2)) : m = j := by
  sorry

end two_digit_number_property_l2680_268076


namespace f_increasing_iff_a_in_range_l2680_268075

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 4 then a * x - 8 else x^2 - 2 * a * x

-- Define what it means for f to be increasing
def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- State the theorem
theorem f_increasing_iff_a_in_range (a : ℝ) :
  is_increasing (f a) ↔ (0 < a ∧ a ≤ 2) :=
sorry

end f_increasing_iff_a_in_range_l2680_268075


namespace area_of_region_R_approx_l2680_268095

/-- Represents a rhombus ABCD -/
structure Rhombus :=
  (side_length : ℝ)
  (angle_B : ℝ)

/-- Represents the region R inside the rhombus -/
def region_R (r : Rhombus) : Set (ℝ × ℝ) :=
  sorry

/-- The area of a set in ℝ² -/
def area (s : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- The theorem statement -/
theorem area_of_region_R_approx (r : Rhombus) :
  r.side_length = 4 ∧ r.angle_B = π/3 →
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ |area (region_R r) - 3| < ε :=
sorry

end area_of_region_R_approx_l2680_268095


namespace equation_solution_l2680_268062

theorem equation_solution : 
  ∃! x : ℝ, (1 / (x - 3) = 3 / (x + 1)) ∧ x = 5 := by
  sorry

end equation_solution_l2680_268062


namespace derivative_implies_limit_l2680_268083

theorem derivative_implies_limit (f : ℝ → ℝ) (x₀ a : ℝ) (h : HasDerivAt f a x₀) :
  ∀ ε > 0, ∃ δ > 0, ∀ Δx, 0 < |Δx| → |Δx| < δ →
    |(f (x₀ + Δx) - f (x₀ - Δx)) / Δx - 2*a| < ε :=
by sorry

end derivative_implies_limit_l2680_268083


namespace rogue_trader_goods_value_l2680_268086

def base7ToBase10 (n : ℕ) : ℕ := sorry

def spiceValue : ℕ := 5213
def metalValue : ℕ := 1653
def fruitValue : ℕ := 202

theorem rogue_trader_goods_value :
  base7ToBase10 spiceValue + base7ToBase10 metalValue + base7ToBase10 fruitValue = 2598 := by
  sorry

end rogue_trader_goods_value_l2680_268086


namespace determinant_of_specific_matrix_l2680_268079

theorem determinant_of_specific_matrix :
  let A : Matrix (Fin 4) (Fin 4) ℤ := !![3, 0, 2, 0;
                                       2, 3, -1, 4;
                                       0, 4, -2, 3;
                                       5, 2, 0, 1]
  Matrix.det A = -84 := by sorry

end determinant_of_specific_matrix_l2680_268079


namespace swimming_pool_receipts_l2680_268038

/-- Calculates the total receipts for a public swimming pool given the number of children and adults, and their respective admission prices. -/
def total_receipts (total_people : ℕ) (children : ℕ) (child_price : ℚ) (adult_price : ℚ) : ℚ :=
  let adults := total_people - children
  let children_total := child_price * children
  let adults_total := adult_price * adults
  children_total + adults_total

/-- Proves that the total receipts for the given scenario is $1405.50 -/
theorem swimming_pool_receipts :
  total_receipts 754 388 (3/2) (9/4) = 2811/2 :=
by
  sorry

end swimming_pool_receipts_l2680_268038


namespace remainder_b_96_mod_50_l2680_268045

theorem remainder_b_96_mod_50 : (7^96 + 9^96) % 50 = 2 := by
  sorry

end remainder_b_96_mod_50_l2680_268045


namespace soft_drink_cost_l2680_268063

/-- The cost of a soft drink given the total spent and the cost of candy bars. -/
theorem soft_drink_cost (total_spent : ℕ) (candy_bars : ℕ) (candy_bar_cost : ℕ) (soft_drink_cost : ℕ) : 
  total_spent = 27 ∧ candy_bars = 5 ∧ candy_bar_cost = 5 → soft_drink_cost = 2 := by
  sorry

end soft_drink_cost_l2680_268063


namespace f_equals_x_l2680_268031

-- Define the function f
def f (a b c : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2 - b*x + c

-- State the theorem
theorem f_equals_x (a b c : ℝ) :
  (∀ x, f a b c x + f a b c (-x) = 0) →  -- f is odd
  (∀ x ≥ 1, ∀ y ≥ 1, x < y → f a b c x < f a b c y) →  -- f is strictly increasing on [1, +∞)
  (a = 0 ∧ c = 0 ∧ b ≤ 3) →  -- conditions on a, b, c
  ∀ x ≥ 1, f a b c x ≥ 1 →  -- f(x) ≥ 1 for x ≥ 1
  (∀ x ≥ 1, f a b c (f a b c x) = x) →  -- f(f(x)) = x for x ≥ 1
  ∀ x ≥ 1, f a b c x = x :=  -- conclusion: f(x) = x for x ≥ 1
by sorry


end f_equals_x_l2680_268031


namespace jan_extra_miles_l2680_268000

/-- Represents the driving data for a person -/
structure DrivingData where
  time : ℝ
  speed : ℝ
  distance : ℝ

/-- The problem statement -/
theorem jan_extra_miles (ian : DrivingData) (han : DrivingData) (jan : DrivingData) : 
  han.time = ian.time + 2 →
  han.speed = ian.speed + 5 →
  jan.time = ian.time + 3 →
  jan.speed = ian.speed + 15 →
  han.distance = ian.distance + 110 →
  jan.distance = ian.distance + 195 := by
  sorry


end jan_extra_miles_l2680_268000


namespace OM_range_theorem_l2680_268098

-- Define the line equation
def line_eq (m n x y : ℝ) : Prop := 2 * m * x - (4 * m + n) * y + 2 * n = 0

-- Define point P
def point_P : ℝ × ℝ := (2, 6)

-- Define that m and n are not simultaneously zero
def not_zero (m n : ℝ) : Prop := m ≠ 0 ∨ n ≠ 0

-- Define the perpendicular line passing through P
def perp_line (m n : ℝ) (M : ℝ × ℝ) : Prop :=
  line_eq m n M.1 M.2 ∧ 
  (M.1 - point_P.1) * (2 * m) + (M.2 - point_P.2) * (-(4 * m + n)) = 0

-- Define the range of |OM|
def OM_range (x : ℝ) : Prop := 5 - Real.sqrt 5 ≤ x ∧ x ≤ 5 + Real.sqrt 5

-- Theorem statement
theorem OM_range_theorem (m n : ℝ) (M : ℝ × ℝ) :
  not_zero m n →
  perp_line m n M →
  OM_range (Real.sqrt (M.1^2 + M.2^2)) :=
sorry

end OM_range_theorem_l2680_268098


namespace dodecagon_triangles_l2680_268025

/-- A regular dodecagon is a 12-sided polygon. -/
def regular_dodecagon : ℕ := 12

/-- The number of triangles that can be formed using the vertices of a regular dodecagon. -/
def num_triangles (n : ℕ) : ℕ := Nat.choose n 3

theorem dodecagon_triangles :
  num_triangles regular_dodecagon = 220 := by
  sorry

end dodecagon_triangles_l2680_268025


namespace remainder_sum_factorials_25_l2680_268090

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def sum_factorials (n : ℕ) : ℕ :=
  match n with
  | 0 => factorial 0
  | n + 1 => factorial (n + 1) + sum_factorials n

theorem remainder_sum_factorials_25 :
  (sum_factorials 50) % 25 = (sum_factorials 4) % 25 :=
by sorry

end remainder_sum_factorials_25_l2680_268090


namespace limit_to_e_l2680_268022

theorem limit_to_e (x : ℕ → ℝ) (h : ∀ ε > 0, ∃ N, ∀ n ≥ N, |x n| > 1/ε) :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |(1 + 1 / x n) ^ (x n) - Real.exp 1| < ε :=
sorry

end limit_to_e_l2680_268022


namespace power_division_addition_l2680_268089

theorem power_division_addition (a : ℝ) : a^4 / a^2 + a^2 = 2 * a^2 := by
  sorry

end power_division_addition_l2680_268089


namespace function_inequality_condition_l2680_268008

/-- A function f(x) = ax^2 + b satisfies f(xy) + f(x + y) ≥ f(x)f(y) for all real x and y
    if and only if 0 < a < 1, 0 < b ≤ 1, and 2a + b - 2 ≤ 0 -/
theorem function_inequality_condition (a b : ℝ) :
  (∀ x y : ℝ, a * (x * y)^2 + b + a * (x + y)^2 + b ≥ (a * x^2 + b) * (a * y^2 + b)) ↔
  (0 < a ∧ a < 1 ∧ 0 < b ∧ b ≤ 1 ∧ 2 * a + b - 2 ≤ 0) :=
by sorry

end function_inequality_condition_l2680_268008


namespace bird_increase_l2680_268082

/-- The number of fish-eater birds Cohen saw over three days -/
def total_birds : ℕ := 1300

/-- The number of fish-eater birds Cohen saw on the first day -/
def first_day_birds : ℕ := 300

/-- The decrease in the number of birds from the first day to the third day -/
def third_day_decrease : ℕ := 200

/-- Theorem stating the increase in the number of birds from the first to the second day -/
theorem bird_increase : 
  ∃ (second_day_birds third_day_birds : ℕ), 
    first_day_birds + second_day_birds + third_day_birds = total_birds ∧
    third_day_birds = first_day_birds - third_day_decrease ∧
    second_day_birds = first_day_birds + 600 :=
by sorry

end bird_increase_l2680_268082


namespace largest_inscribed_equilateral_triangle_area_l2680_268057

/-- The area of the largest equilateral triangle inscribed in a circle with radius 10 cm,
    where one side of the triangle is a diameter of the circle. -/
theorem largest_inscribed_equilateral_triangle_area :
  let r : ℝ := 10  -- radius of the circle in cm
  let d : ℝ := 2 * r  -- diameter of the circle in cm
  let h : ℝ := r * Real.sqrt 3  -- height of the equilateral triangle
  let area : ℝ := (1 / 2) * d * h  -- area of the triangle
  area = 100 * Real.sqrt 3 := by sorry

end largest_inscribed_equilateral_triangle_area_l2680_268057


namespace goldfish_cost_graph_piecewise_linear_l2680_268001

/-- The cost function for goldfish purchases -/
def cost (n : ℕ) : ℚ :=
  if n ≤ 10 then 20 * n else 20 * n - 5

/-- The graph of the cost function is piecewise linear -/
theorem goldfish_cost_graph_piecewise_linear :
  ∃ (f g : ℚ → ℚ),
    (∀ x, f x = 20 * x) ∧
    (∀ x, g x = 20 * x - 5) ∧
    (∀ n : ℕ, 1 ≤ n ∧ n ≤ 15 →
      (n ≤ 10 ∧ cost n = f n) ∨
      (10 < n ∧ cost n = g n)) :=
by sorry

end goldfish_cost_graph_piecewise_linear_l2680_268001


namespace article_price_before_discount_l2680_268035

/-- 
Given an article whose price after a 24% decrease is 988 rupees, 
prove that its original price was 1300 rupees.
-/
theorem article_price_before_discount (price_after_discount : ℝ) 
  (h1 : price_after_discount = 988) 
  (h2 : price_after_discount = 0.76 * (original_price : ℝ)) : 
  original_price = 1300 := by
  sorry

end article_price_before_discount_l2680_268035


namespace smith_children_age_l2680_268064

theorem smith_children_age (age1 age2 age3 age4 : ℕ) 
  (h1 : age1 = 5)
  (h2 : age2 = 7)
  (h3 : age3 = 10)
  (h_avg : (age1 + age2 + age3 + age4) / 4 = 8) :
  age4 = 10 := by
sorry

end smith_children_age_l2680_268064


namespace crossout_theorem_l2680_268012

/-- The process of crossing out numbers and writing sums -/
def crossOutProcess (n : ℕ) : ℕ → ℕ
| 0 => n
| (m + 1) => let prev := crossOutProcess n m
             if prev > 4 then prev - 3 else prev

/-- The condition for n to be reduced to one number -/
def reducesToOne (n : ℕ) : Prop :=
  ∃ k, crossOutProcess n k = 1

/-- The sum of all numbers written during the process -/
def totalSum (n : ℕ) : ℕ :=
  sorry  -- Definition of totalSum would go here

/-- Main theorem combining both parts of the problem -/
theorem crossout_theorem :
  (∀ n : ℕ, reducesToOne n ↔ n % 3 = 1) ∧
  totalSum 2002 = 12881478 :=
sorry

end crossout_theorem_l2680_268012


namespace trapezoid_area_l2680_268093

/-- The area of a trapezoid with height x, one base 4x, and the other base 3x is 7x²/2 -/
theorem trapezoid_area (x : ℝ) : 
  x * ((4 * x + 3 * x) / 2) = 7 * x^2 / 2 := by
  sorry

end trapezoid_area_l2680_268093


namespace expression_value_l2680_268069

theorem expression_value (x y : ℝ) (h : (x - y) / (x + y) = 3) :
  2 * (x - y) / (x + y) - (x + y) / (3 * (x - y)) = 53 / 9 := by
  sorry

end expression_value_l2680_268069


namespace jeans_discount_impossibility_total_price_calculation_l2680_268042

/-- Represents the prices and discount rates for jeans --/
structure JeansSale where
  fox_price : ℝ
  pony_price : ℝ
  fox_quantity : ℕ
  pony_quantity : ℕ
  total_discount_rate : ℝ
  pony_discount_rate : ℝ

/-- Theorem stating the impossibility of the given discount rates --/
theorem jeans_discount_impossibility (sale : JeansSale)
  (h1 : sale.fox_price = 15)
  (h2 : sale.pony_price = 18)
  (h3 : sale.fox_quantity = 3)
  (h4 : sale.pony_quantity = 2)
  (h5 : sale.total_discount_rate = 0.18)
  (h6 : sale.pony_discount_rate = 0.5667) :
  False := by
  sorry

/-- Function to calculate the total regular price --/
def total_regular_price (sale : JeansSale) : ℝ :=
  sale.fox_price * sale.fox_quantity + sale.pony_price * sale.pony_quantity

/-- Theorem stating the total regular price for the given quantities --/
theorem total_price_calculation (sale : JeansSale)
  (h1 : sale.fox_price = 15)
  (h2 : sale.pony_price = 18)
  (h3 : sale.fox_quantity = 3)
  (h4 : sale.pony_quantity = 2) :
  total_regular_price sale = 81 := by
  sorry

end jeans_discount_impossibility_total_price_calculation_l2680_268042


namespace least_k_equals_2_pow_q_l2680_268018

/-- Represents a polynomial with integer coefficients -/
def IntPolynomial := ℕ → ℤ

/-- Given an even positive integer n, this function returns the least k₀ such that
    k₀ = f(x) · (x+1)^n + g(x) · (x^n + 1) for some polynomials f(x) and g(x) with integer coefficients -/
noncomputable def least_k (n : ℕ) : ℕ :=
  sorry

theorem least_k_equals_2_pow_q (n : ℕ) (q r : ℕ) (hn : Even n) (hq : Odd q) (hnqr : n = q * 2^r) :
  least_k n = 2^q :=
by sorry

end least_k_equals_2_pow_q_l2680_268018


namespace intersection_A_B_range_of_a_l2680_268067

-- Define the sets A, B, and C
def A : Set ℝ := {x | x^2 - 3*x < 0}
def B : Set ℝ := {x | (x + 2)*(4 - x) ≥ 0}
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x ≤ a + 1}

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = {x : ℝ | 0 < x ∧ x < 3} := by sorry

-- Theorem for the range of a when B ∪ C = B
theorem range_of_a (a : ℝ) (h : B ∪ C a = B) : -2 ≤ a ∧ a ≤ 3 := by sorry

end intersection_A_B_range_of_a_l2680_268067


namespace y_at_40_l2680_268029

/-- A line passing through three given points -/
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ
  point3 : ℝ × ℝ

/-- The line passing through the given points -/
def exampleLine : Line :=
  { point1 := (2, 5)
  , point2 := (6, 17)
  , point3 := (10, 29) }

/-- Function to calculate y-coordinate for a given x-coordinate on the line -/
def yCoordinate (l : Line) (x : ℝ) : ℝ :=
  sorry

theorem y_at_40 (l : Line) : l = exampleLine → yCoordinate l 40 = 119 := by
  sorry

end y_at_40_l2680_268029


namespace smallest_non_existent_count_l2680_268087

/-- The number of terms in the arithmetic progression -/
def progression_length : ℕ := 1999

/-- 
  Counts the number of integer terms in an arithmetic progression 
  with 'progression_length' terms and common difference 1/m
-/
def count_integer_terms (m : ℕ) : ℕ :=
  1 + (progression_length - 1) / m

/-- 
  Checks if there exists an arithmetic progression of 'progression_length' 
  real numbers containing exactly n integers
-/
def exists_progression_with_n_integers (n : ℕ) : Prop :=
  ∃ m : ℕ, m > 0 ∧ count_integer_terms m = n

theorem smallest_non_existent_count : 
  (∀ k < 70, exists_progression_with_n_integers k) ∧
  ¬exists_progression_with_n_integers 70 :=
sorry

end smallest_non_existent_count_l2680_268087


namespace visitation_problem_l2680_268081

/-- Represents the visitation schedule of a friend --/
structure VisitSchedule where
  period : ℕ+

/-- Calculates the number of days in a given period when exactly two friends visit --/
def exactlyTwoVisits (alice beatrix claire : VisitSchedule) (totalDays : ℕ) : ℕ :=
  sorry

/-- Theorem statement for the visitation problem --/
theorem visitation_problem :
  let alice : VisitSchedule := ⟨1⟩
  let beatrix : VisitSchedule := ⟨5⟩
  let claire : VisitSchedule := ⟨7⟩
  let totalDays : ℕ := 180
  exactlyTwoVisits alice beatrix claire totalDays = 51 := by sorry

end visitation_problem_l2680_268081


namespace monthly_salary_is_7600_l2680_268010

/-- Represents the monthly salary allocation problem --/
def SalaryAllocation (x : ℝ) : Prop :=
  let bank := x / 2
  let remaining := x / 2
  let mortgage := remaining / 2 - 300
  let meals := (remaining - mortgage) / 2 + 300
  let leftover := remaining - mortgage - meals
  (bank = x / 2) ∧
  (mortgage = remaining / 2 - 300) ∧
  (meals = (remaining - mortgage) / 2 + 300) ∧
  (leftover = 800)

/-- Theorem stating that the monthly salary satisfying the given conditions is 7600 --/
theorem monthly_salary_is_7600 :
  ∃ x : ℝ, SalaryAllocation x ∧ x = 7600 :=
sorry

end monthly_salary_is_7600_l2680_268010


namespace binomial_60_3_l2680_268053

theorem binomial_60_3 : Nat.choose 60 3 = 34220 := by sorry

end binomial_60_3_l2680_268053


namespace campaign_fliers_l2680_268017

theorem campaign_fliers (initial_fliers : ℕ) : 
  (initial_fliers : ℚ) * (4/5) * (3/4) = 600 → initial_fliers = 1000 := by
  sorry

end campaign_fliers_l2680_268017


namespace real_roots_quadratic_equation_l2680_268055

theorem real_roots_quadratic_equation (k : ℝ) :
  (∃ x : ℝ, k * x^2 - 2 * x + 1 = 0) → k ≤ 1 := by
  sorry

end real_roots_quadratic_equation_l2680_268055


namespace rainfall_solution_l2680_268049

def rainfall_problem (day1 day2 day3 : ℝ) : Prop :=
  day1 = 4 ∧
  day2 = 5 * day1 ∧
  day3 = day1 + day2 - 6

theorem rainfall_solution :
  ∀ day1 day2 day3 : ℝ,
  rainfall_problem day1 day2 day3 →
  day3 = 18 := by
sorry

end rainfall_solution_l2680_268049


namespace cottage_rent_division_l2680_268023

/-- The total rent for the cottage -/
def total_rent : ℤ := 300

/-- The amount paid by the first friend -/
def first_friend_payment (f2 f3 f4 : ℤ) : ℤ := (f2 + f3 + f4) / 2

/-- The amount paid by the second friend -/
def second_friend_payment (f1 f3 f4 : ℤ) : ℤ := (f1 + f3 + f4) / 3

/-- The amount paid by the third friend -/
def third_friend_payment (f1 f2 f4 : ℤ) : ℤ := (f1 + f2 + f4) / 4

/-- The amount paid by the fourth friend -/
def fourth_friend_payment (f1 f2 f3 : ℤ) : ℤ := total_rent - (f1 + f2 + f3)

theorem cottage_rent_division :
  ∃ (f1 f2 f3 f4 : ℤ),
    f1 = first_friend_payment f2 f3 f4 ∧
    f2 = second_friend_payment f1 f3 f4 ∧
    f3 = third_friend_payment f1 f2 f4 ∧
    f4 = fourth_friend_payment f1 f2 f3 ∧
    f1 + f2 + f3 + f4 = total_rent ∧
    f4 = 65 :=
by sorry

end cottage_rent_division_l2680_268023


namespace smallest_positive_integer_congruence_l2680_268032

theorem smallest_positive_integer_congruence :
  ∃ (x : ℕ), x > 0 ∧ (46 * x + 8) % 24 = 4 ∧ ∀ (y : ℕ), y > 0 ∧ (46 * y + 8) % 24 = 4 → x ≤ y :=
by sorry

end smallest_positive_integer_congruence_l2680_268032


namespace lg_sum_five_two_l2680_268099

-- Define lg as the base-10 logarithm
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem lg_sum_five_two : lg 5 + lg 2 = 1 := by sorry

end lg_sum_five_two_l2680_268099


namespace simple_interest_problem_l2680_268052

/-- Given a principal P at simple interest for 6 years, if increasing the
    interest rate by 4% results in $144 more interest, then P = $600. -/
theorem simple_interest_problem (P : ℝ) (R : ℝ) : 
  (P * (R + 4) * 6) / 100 - (P * R * 6) / 100 = 144 → P = 600 := by
sorry

end simple_interest_problem_l2680_268052


namespace simplify_expression_l2680_268070

theorem simplify_expression (a : ℝ) (ha : a ≠ 0) (ha' : a ≠ -1) :
  ((a^2 + 1) / a - 2) / ((a^2 - 1) / (a^2 + a)) = a - 1 := by
  sorry

end simplify_expression_l2680_268070


namespace one_third_minus_decimal_approx_l2680_268003

theorem one_third_minus_decimal_approx : 
  (1 : ℚ) / 3 - 33333333 / 100000000 = 1 / (3 * 100000000) := by sorry

end one_third_minus_decimal_approx_l2680_268003


namespace triangle_equality_l2680_268044

theorem triangle_equality (a b c : ℝ) 
  (h1 : |a| ≥ |b + c|) 
  (h2 : |b| ≥ |c + a|) 
  (h3 : |c| ≥ |a + b|) : 
  a + b + c = 0 := by
  sorry

end triangle_equality_l2680_268044


namespace jasons_grade_difference_l2680_268039

/-- Given the grades of Jenny and Bob, and the relationship between Bob's and Jason's grades,
    prove that Jason's grade is 25 points less than Jenny's. -/
theorem jasons_grade_difference (jenny_grade : ℕ) (bob_grade : ℕ) :
  jenny_grade = 95 →
  bob_grade = 35 →
  bob_grade * 2 = jenny_grade - 25 :=
by sorry

end jasons_grade_difference_l2680_268039


namespace cubic_roots_relation_l2680_268043

def f (x : ℝ) : ℝ := x^3 + x^2 + 2*x + 4

def g (x b c d : ℝ) : ℝ := x^3 + b*x^2 + c*x + d

theorem cubic_roots_relation (b c d : ℝ) :
  (∃ r₁ r₂ r₃ : ℝ, r₁ ≠ r₂ ∧ r₂ ≠ r₃ ∧ r₁ ≠ r₃ ∧
    f r₁ = 0 ∧ f r₂ = 0 ∧ f r₃ = 0 ∧
    g (r₁^3) b c d = 0 ∧ g (r₂^3) b c d = 0 ∧ g (r₃^3) b c d = 0) →
  b = 24 ∧ c = 32 ∧ d = 64 :=
by sorry

end cubic_roots_relation_l2680_268043


namespace janabel_widget_sales_l2680_268060

theorem janabel_widget_sales (n : ℕ) (a₁ : ℕ) (d : ℕ) : 
  n = 15 → a₁ = 2 → d = 2 → (n * (2 * a₁ + (n - 1) * d)) / 2 = 240 := by
  sorry

end janabel_widget_sales_l2680_268060


namespace simplify_expression_l2680_268006

theorem simplify_expression (x : ℝ) : 120 * x - 75 * x = 45 * x := by
  sorry

end simplify_expression_l2680_268006


namespace fourth_root_equation_solutions_l2680_268074

theorem fourth_root_equation_solutions :
  let f : ℝ → ℝ := λ x => Real.sqrt (Real.sqrt x)
  ∀ x : ℝ, (x > 0 ∧ f x = 16 / (9 - f x)) ↔ (x = 4096 ∨ x = 1) :=
by sorry

end fourth_root_equation_solutions_l2680_268074


namespace slope_of_l₃_l2680_268085

-- Define the lines and points
def l₁ : Set (ℝ × ℝ) := {(x, y) | 5 * x - 3 * y = 2}
def l₂ : Set (ℝ × ℝ) := {(x, y) | y = 2}
def A : ℝ × ℝ := (2, -2)

-- Define the existence of point B
def B_exists : Prop := ∃ B : ℝ × ℝ, B ∈ l₁ ∧ B ∈ l₂

-- Define the existence of point C and line l₃
def C_and_l₃_exist : Prop := ∃ C : ℝ × ℝ, ∃ l₃ : Set (ℝ × ℝ),
  C ∈ l₂ ∧ A ∈ l₃ ∧ C ∈ l₃ ∧
  (∀ x₁ y₁ x₂ y₂, (x₁, y₁) ∈ l₃ ∧ (x₂, y₂) ∈ l₃ ∧ x₁ ≠ x₂ → (y₂ - y₁) / (x₂ - x₁) > 0)

-- Define the area of triangle ABC
def area_ABC : ℝ := 5

-- Theorem statement
theorem slope_of_l₃ (h₁ : A ∈ l₁) (h₂ : B_exists) (h₃ : C_and_l₃_exist) (h₄ : area_ABC = 5) :
  ∃ C : ℝ × ℝ, ∃ l₃ : Set (ℝ × ℝ),
    C ∈ l₂ ∧ A ∈ l₃ ∧ C ∈ l₃ ∧
    (∀ x₁ y₁ x₂ y₂, (x₁, y₁) ∈ l₃ ∧ (x₂, y₂) ∈ l₃ ∧ x₁ ≠ x₂ → (y₂ - y₁) / (x₂ - x₁) = 20 / 9) :=
sorry

end slope_of_l₃_l2680_268085


namespace square_areas_and_perimeters_l2680_268002

theorem square_areas_and_perimeters (x : ℝ) : 
  (∃ s₁ s₂ : ℝ, 
    s₁^2 = x^2 + 4*x + 4 ∧ 
    s₂^2 = 4*x^2 - 12*x + 9 ∧ 
    4*s₁ + 4*s₂ = 32) → 
  x = 3 := by
sorry

end square_areas_and_perimeters_l2680_268002


namespace rectangular_prism_width_l2680_268078

theorem rectangular_prism_width (l h d : ℝ) (hl : l = 5) (hh : h = 8) (hd : d = 17) :
  ∃ w : ℝ, w > 0 ∧ w^2 + l^2 + h^2 = d^2 ∧ w = 10 * Real.sqrt 2 := by
  sorry

end rectangular_prism_width_l2680_268078


namespace distinct_towers_count_l2680_268096

/-- Represents the number of cubes of each color -/
structure CubeCount where
  red : Nat
  blue : Nat
  yellow : Nat

/-- Calculates the number of distinct towers -/
def countDistinctTowers (cubes : CubeCount) (towerHeight : Nat) : Nat :=
  sorry

/-- Theorem: The number of distinct towers of height 10 that can be built
    using 3 red cubes, 4 blue cubes, and 5 yellow cubes, with two cubes
    not being used, is equal to 6,812 -/
theorem distinct_towers_count :
  let cubes : CubeCount := { red := 3, blue := 4, yellow := 5 }
  let towerHeight : Nat := 10
  countDistinctTowers cubes towerHeight = 6812 := by
  sorry

end distinct_towers_count_l2680_268096


namespace total_wrapping_paper_l2680_268026

/-- The amount of wrapping paper needed for three presents -/
def wrapping_paper (first_present second_present third_present : ℝ) : ℝ :=
  first_present + second_present + third_present

/-- Theorem: The total amount of wrapping paper needed is 7 square feet -/
theorem total_wrapping_paper :
  let first_present := 2
  let second_present := 3/4 * first_present
  let third_present := first_present + second_present
  wrapping_paper first_present second_present third_present = 7 :=
by sorry

end total_wrapping_paper_l2680_268026


namespace sequence_conjecture_l2680_268021

theorem sequence_conjecture (a : ℕ → ℝ) :
  a 1 = 1 ∧
  (∀ n : ℕ, a (n + 1) - a n > 0) ∧
  (∀ n : ℕ, (a (n + 1) - a n)^2 - 2 * (a (n + 1) + a n) + 1 = 0) →
  ∀ n : ℕ, a n = n^2 := by
sorry

end sequence_conjecture_l2680_268021


namespace prism_with_18_edges_has_8_faces_l2680_268080

/-- A prism is a polyhedron with two congruent parallel faces (bases) and whose other faces (lateral faces) are parallelograms. -/
structure Prism where
  edges : ℕ

/-- The number of faces in a prism -/
def num_faces (p : Prism) : ℕ :=
  let L := p.edges / 3
  2 + L

theorem prism_with_18_edges_has_8_faces (p : Prism) (h : p.edges = 18) :
  num_faces p = 8 := by
  sorry

end prism_with_18_edges_has_8_faces_l2680_268080


namespace luke_final_sticker_count_l2680_268046

/-- Calculates the final number of stickers Luke has after various transactions -/
def final_sticker_count (initial : ℕ) (bought : ℕ) (from_friend : ℕ) (birthday : ℕ) 
                        (traded_out : ℕ) (traded_in : ℕ) (to_sister : ℕ) 
                        (for_card : ℕ) (to_charity : ℕ) : ℕ :=
  initial + bought + from_friend + birthday - traded_out + traded_in - to_sister - for_card - to_charity

/-- Theorem stating that Luke ends up with 67 stickers -/
theorem luke_final_sticker_count :
  final_sticker_count 20 12 25 30 10 15 5 8 12 = 67 := by
  sorry

end luke_final_sticker_count_l2680_268046


namespace rectangular_field_area_l2680_268037

/-- The area of a rectangular field with one side 16 m and a diagonal of 17 m is 16 * √33 square meters. -/
theorem rectangular_field_area (a b : ℝ) (h1 : a = 16) (h2 : a^2 + b^2 = 17^2) :
  a * b = 16 * Real.sqrt 33 := by
  sorry

end rectangular_field_area_l2680_268037


namespace wang_elevator_problem_l2680_268004

def floor_movements : List Int := [6, -3, 10, -8, 12, -7, -10]
def floor_height : ℝ := 3
def electricity_per_meter : ℝ := 0.2

theorem wang_elevator_problem :
  (List.sum floor_movements = 0) ∧
  (List.sum (List.map (λ x => floor_height * electricity_per_meter * |x|) floor_movements) = 33.6) := by
  sorry

end wang_elevator_problem_l2680_268004


namespace geometric_condition_implies_a_equals_two_l2680_268068

/-- The value of a for which the given geometric conditions are satisfied -/
def geometric_a : ℝ := 2

/-- The line equation y = 2x + 2 -/
def line (x : ℝ) : ℝ := 2 * x + 2

/-- The parabola equation y = ax^2 -/
def parabola (a x : ℝ) : ℝ := a * x^2

/-- Theorem stating that under the given geometric conditions, a = 2 -/
theorem geometric_condition_implies_a_equals_two (a : ℝ) 
  (h_pos : a > 0)
  (h_intersect : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ line x₁ = parabola a x₁ ∧ line x₂ = parabola a x₂)
  (h_midpoint : ∃ (x_mid : ℝ), x_mid = (x₁ + x₂) / 2 ∧ 
    parabola a x_mid = a * x_mid^2 ∧ 
    ∀ (y : ℝ), y ≠ a * x_mid^2 → |x_mid - x₁| + |y - line x₁| = |x_mid - x₂| + |y - line x₂|)
  (h_vector_condition : ∀ (A P Q : ℝ × ℝ), 
    P.1 ≠ Q.1 → 
    line P.1 = P.2 → line Q.1 = Q.2 → 
    parabola a A.1 = A.2 → 
    |(A.1 - P.1, A.2 - P.2)| + |(A.1 - Q.1, A.2 - Q.2)| = 
    |(A.1 - P.1, A.2 - P.2)| - |(A.1 - Q.1, A.2 - Q.2)|)
  : a = geometric_a := by sorry

end geometric_condition_implies_a_equals_two_l2680_268068


namespace last_integer_in_sequence_l2680_268094

def sequence_term (n : ℕ) : ℚ :=
  524288 / 2^n

def is_integer (q : ℚ) : Prop :=
  ∃ (z : ℤ), q = z

theorem last_integer_in_sequence :
  ∃ (k : ℕ), (∀ (n : ℕ), n ≤ k → is_integer (sequence_term n)) ∧
             (∀ (m : ℕ), m > k → ¬ is_integer (sequence_term m)) ∧
             sequence_term k = 1 :=
sorry

end last_integer_in_sequence_l2680_268094


namespace proposition_counterexample_l2680_268050

theorem proposition_counterexample : ∃ a b : ℝ, a > b ∧ a^2 ≤ b^2 := by
  sorry

end proposition_counterexample_l2680_268050


namespace new_tires_cost_calculation_l2680_268040

/-- The amount spent on speakers -/
def speakers_cost : ℝ := 118.54

/-- The total amount spent on car parts -/
def total_car_parts_cost : ℝ := 224.87

/-- The amount spent on new tires -/
def new_tires_cost : ℝ := total_car_parts_cost - speakers_cost

theorem new_tires_cost_calculation : 
  new_tires_cost = 106.33 := by sorry

end new_tires_cost_calculation_l2680_268040


namespace quadratic_inequality_implies_a_range_l2680_268019

theorem quadratic_inequality_implies_a_range :
  (∀ x : ℝ, x ∈ Set.Icc 0 2 → x^2 - 2*a*x + a + 2 ≥ 0) →
  a ∈ Set.Icc (-2) 2 := by
sorry

end quadratic_inequality_implies_a_range_l2680_268019


namespace complex_fraction_real_minus_imag_l2680_268059

theorem complex_fraction_real_minus_imag (z : ℂ) (a b : ℝ) : 
  z = 5 / (-3 - Complex.I) → 
  a = z.re → 
  b = z.im → 
  a - b = -2 := by
  sorry

end complex_fraction_real_minus_imag_l2680_268059


namespace fraction_reduction_l2680_268041

theorem fraction_reduction (b y : ℝ) (h : 4 * b^2 + y^4 ≠ 0) :
  ((Real.sqrt (4 * b^2 + y^4) - (y^4 - 4 * b^2) / Real.sqrt (4 * b^2 + y^4)) / (4 * b^2 + y^4)) ^ (2/3) = 
  (8 * b^2) / (4 * b^2 + y^4) :=
by sorry

end fraction_reduction_l2680_268041


namespace function_properties_l2680_268007

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem function_properties (f : ℝ → ℝ) 
  (h1 : is_odd (λ x => f (x + 1))) 
  (h2 : is_odd (λ x => f (x - 1))) : 
  (∀ x, f (x + 4) = f x) ∧ 
  (is_odd (λ x => f (x + 3))) := by
sorry

end function_properties_l2680_268007


namespace average_temperature_l2680_268084

def temperatures : List ℝ := [52, 62, 55, 59, 50]

theorem average_temperature : 
  (temperatures.sum / temperatures.length : ℝ) = 55.6 := by sorry

end average_temperature_l2680_268084


namespace adam_miles_l2680_268072

/-- Adam ran 25 miles more than Katie, and Katie ran 10 miles. -/
theorem adam_miles (katie_miles : ℕ) (adam_miles : ℕ) : 
  katie_miles = 10 → adam_miles = katie_miles + 25 → adam_miles = 35 := by
  sorry

end adam_miles_l2680_268072


namespace f_2019_eq_zero_l2680_268077

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period_3 (f : ℝ → ℝ) : Prop := ∀ x, f (3 - x) = f x

theorem f_2019_eq_zero 
  (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_period : has_period_3 f) : 
  f 2019 = 0 := by sorry

end f_2019_eq_zero_l2680_268077


namespace range_of_absolute_linear_function_l2680_268024

theorem range_of_absolute_linear_function 
  (a b : ℝ) (ha : a > 0) (hb : b < 0) :
  let f : ℝ → ℝ := fun x ↦ |a * x + b|
  (∀ x, 0 ≤ x ∧ x ≤ 1 → 0 ≤ f x ∧ f x ≤ max (|b|) (|a + b|)) ∧
  (∃ x, 0 ≤ x ∧ x ≤ 1 ∧ f x = 0) ∧
  (∃ x, 0 ≤ x ∧ x ≤ 1 ∧ f x = max (|b|) (|a + b|)) :=
by sorry

end range_of_absolute_linear_function_l2680_268024


namespace problem_statement_l2680_268033

theorem problem_statement :
  (¬(∀ x : ℝ, x > 0 → Real.log x ≥ 0)) ∧ (∃ x₀ : ℝ, Real.sin x₀ = Real.cos x₀) := by
  sorry

end problem_statement_l2680_268033

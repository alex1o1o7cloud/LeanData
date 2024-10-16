import Mathlib

namespace NUMINAMATH_CALUDE_chameleon_color_change_l982_98229

theorem chameleon_color_change (total : ℕ) (blue_factor red_factor : ℕ) : 
  total = 140 → blue_factor = 5 → red_factor = 3 →
  ∃ (initial_blue : ℕ),
    initial_blue > 0 ∧
    initial_blue * blue_factor ≤ total ∧
    (total - initial_blue * blue_factor) * red_factor + initial_blue = total →
    initial_blue * blue_factor - initial_blue = 80 := by
  sorry

#check chameleon_color_change

end NUMINAMATH_CALUDE_chameleon_color_change_l982_98229


namespace NUMINAMATH_CALUDE_factorization_of_x_power_difference_l982_98250

theorem factorization_of_x_power_difference (m : ℕ) (x : ℝ) (hm : m > 1) :
  x^m - x^(m-2) = x^(m-2) * (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_x_power_difference_l982_98250


namespace NUMINAMATH_CALUDE_point_inside_circle_m_range_l982_98294

/-- A point (x, y) is inside a circle with center (a, b) and radius r if the square of the distance
    from the point to the center is less than r^2 -/
def IsInsideCircle (x y a b : ℝ) (r : ℝ) : Prop :=
  (x - a)^2 + (y - b)^2 < r^2

theorem point_inside_circle_m_range :
  ∀ m : ℝ, IsInsideCircle 1 (-3) 2 (-1) (m^(1/2)) → m > 5 := by
  sorry

end NUMINAMATH_CALUDE_point_inside_circle_m_range_l982_98294


namespace NUMINAMATH_CALUDE_two_possible_values_for_k_l982_98270

theorem two_possible_values_for_k (a b c k : ℝ) : 
  (a / (b + c) = k ∧ b / (c + a) = k ∧ c / (a + b) = k) → 
  (k = 1/2 ∨ k = -1) ∧ ∀ x : ℝ, (x = 1/2 ∨ x = -1) → ∃ a b c : ℝ, a / (b + c) = x ∧ b / (c + a) = x ∧ c / (a + b) = x :=
by sorry

end NUMINAMATH_CALUDE_two_possible_values_for_k_l982_98270


namespace NUMINAMATH_CALUDE_remainder_of_1725_base14_div_9_l982_98216

/-- Converts a base-14 number to decimal --/
def base14ToDecimal (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => acc * 14 + d) 0

/-- The base-14 representation of 1725₁₄ --/
def number : List Nat := [1, 7, 2, 5]

theorem remainder_of_1725_base14_div_9 :
  (base14ToDecimal number) % 9 = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_1725_base14_div_9_l982_98216


namespace NUMINAMATH_CALUDE_combined_solid_sum_l982_98205

/-- A right rectangular prism -/
structure RectangularPrism :=
  (faces : ℕ)
  (edges : ℕ)
  (vertices : ℕ)

/-- A pyramid added to a rectangular prism -/
structure PrismWithPyramid :=
  (prism : RectangularPrism)
  (pyramid_base_face : ℕ)

/-- The combined solid (prism and pyramid) -/
def CombinedSolid (pw : PrismWithPyramid) : ℕ × ℕ × ℕ :=
  let new_faces := pw.prism.faces - pw.pyramid_base_face + 4
  let new_edges := pw.prism.edges + 4
  let new_vertices := pw.prism.vertices + 1
  (new_faces, new_edges, new_vertices)

theorem combined_solid_sum (pw : PrismWithPyramid) 
  (h1 : pw.prism.faces = 6)
  (h2 : pw.prism.edges = 12)
  (h3 : pw.prism.vertices = 8)
  (h4 : pw.pyramid_base_face = 1) :
  let (f, e, v) := CombinedSolid pw
  f + e + v = 34 := by sorry

end NUMINAMATH_CALUDE_combined_solid_sum_l982_98205


namespace NUMINAMATH_CALUDE_positive_solution_x_l982_98236

theorem positive_solution_x (x y z : ℝ) 
  (eq1 : x * y = 10 - 3 * x - 2 * y)
  (eq2 : y * z = 10 - 5 * y - 3 * z)
  (eq3 : x * z = 40 - 5 * x - 2 * z)
  (x_pos : x > 0) :
  x = 8 := by
  sorry

end NUMINAMATH_CALUDE_positive_solution_x_l982_98236


namespace NUMINAMATH_CALUDE_problem_statement_l982_98264

theorem problem_statement : (12 : ℕ)^5 * 6^5 / 432^3 = 24 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l982_98264


namespace NUMINAMATH_CALUDE_room_width_l982_98280

/-- Given a rectangular room with the specified length and paving cost, prove its width. -/
theorem room_width (length : ℝ) (total_cost : ℝ) (rate : ℝ) (width : ℝ) 
  (h1 : length = 5.5)
  (h2 : total_cost = 20625)
  (h3 : rate = 1000)
  (h4 : total_cost = rate * length * width) :
  width = 3.75 := by sorry

end NUMINAMATH_CALUDE_room_width_l982_98280


namespace NUMINAMATH_CALUDE_y1_greater_than_y2_l982_98208

/-- A linear function y = -3x + b -/
def linearFunction (x : ℝ) (b : ℝ) : ℝ := -3 * x + b

/-- Theorem: For a linear function y = -3x + b, if P₁(-3, y₁) and P₂(4, y₂) are points on the graph, then y₁ > y₂ -/
theorem y1_greater_than_y2 (b : ℝ) (y₁ y₂ : ℝ) 
  (h₁ : y₁ = linearFunction (-3) b) 
  (h₂ : y₂ = linearFunction 4 b) : 
  y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_y1_greater_than_y2_l982_98208


namespace NUMINAMATH_CALUDE_square_perimeter_equals_area_l982_98269

theorem square_perimeter_equals_area (x : ℝ) (h : x > 0) :
  4 * x = x^2 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_equals_area_l982_98269


namespace NUMINAMATH_CALUDE_complement_of_B_l982_98275

def U : Finset Nat := {1, 2, 3, 4, 5, 6, 7}
def B : Finset Nat := {1, 3, 5, 7}

theorem complement_of_B : 
  (U \ B) = {2, 4, 6} := by sorry

end NUMINAMATH_CALUDE_complement_of_B_l982_98275


namespace NUMINAMATH_CALUDE_negation_of_existential_quadratic_inequality_l982_98298

theorem negation_of_existential_quadratic_inequality :
  (¬ ∃ x : ℝ, x^2 + 2*x ≤ 1) ↔ (∀ x : ℝ, x^2 + 2*x > 1) := by sorry

end NUMINAMATH_CALUDE_negation_of_existential_quadratic_inequality_l982_98298


namespace NUMINAMATH_CALUDE_prob_at_least_one_white_correct_l982_98209

def total_balls : ℕ := 9
def red_balls : ℕ := 5
def white_balls : ℕ := 4

def prob_at_least_one_white : ℚ := 13 / 18

theorem prob_at_least_one_white_correct :
  let prob_two_red : ℚ := (red_balls / total_balls) * ((red_balls - 1) / (total_balls - 1))
  1 - prob_two_red = prob_at_least_one_white := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_white_correct_l982_98209


namespace NUMINAMATH_CALUDE_quadratic_root_range_l982_98214

theorem quadratic_root_range (a : ℝ) (x₁ x₂ : ℝ) : 
  (∃ x, a * x^2 + (a + 1) * x + 6 * a = 0) ∧ 
  (x₁ ≠ x₂) ∧ 
  (x₁ < 1 ∧ 1 < x₂) ∧
  (a * x₁^2 + (a + 1) * x₁ + 6 * a = 0) ∧
  (a * x₂^2 + (a + 1) * x₂ + 6 * a = 0) →
  -1/8 < a ∧ a < 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_range_l982_98214


namespace NUMINAMATH_CALUDE_carmichael_561_l982_98297

theorem carmichael_561 (a : ℤ) : a ^ 561 ≡ a [ZMOD 561] := by
  sorry

end NUMINAMATH_CALUDE_carmichael_561_l982_98297


namespace NUMINAMATH_CALUDE_problem_solution_l982_98203

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + 1

noncomputable def f_derivative (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

noncomputable def g (a b : ℝ) (x : ℝ) : ℝ := f_derivative a b x * Real.exp x

theorem problem_solution (a b : ℝ) :
  f_derivative a b 1 = 2*a ∧ f_derivative a b 2 = -b →
  (a = -3/2 ∧ b = -3) ∧
  (∀ x, g a b x ≥ g a b 1) ∧
  (∀ x, g a b x ≤ g a b (-2)) ∧
  g a b 1 = -3 * Real.exp 1 ∧
  g a b (-2) = 15 * Real.exp (-2) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l982_98203


namespace NUMINAMATH_CALUDE_chocolate_bars_distribution_l982_98238

theorem chocolate_bars_distribution (total_bars : ℕ) (num_people : ℕ) (bars_for_two : ℕ) : 
  total_bars = 12 → num_people = 3 → bars_for_two = (total_bars / num_people) * 2 → bars_for_two = 8 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bars_distribution_l982_98238


namespace NUMINAMATH_CALUDE_no_such_function_exists_l982_98218

theorem no_such_function_exists :
  ¬ ∃ f : ℤ → Fin 3,
    ∀ x y : ℤ, |x - y| ∈ ({2, 3, 5} : Set ℤ) → f x ≠ f y :=
by sorry

end NUMINAMATH_CALUDE_no_such_function_exists_l982_98218


namespace NUMINAMATH_CALUDE_mode_and_median_of_data_l982_98259

def data : List ℕ := [6, 8, 3, 6, 4, 6, 5]

def mode (l : List ℕ) : ℕ := sorry

def median (l : List ℕ) : ℚ := sorry

theorem mode_and_median_of_data :
  mode data = 6 ∧ median data = 6 := by sorry

end NUMINAMATH_CALUDE_mode_and_median_of_data_l982_98259


namespace NUMINAMATH_CALUDE_married_men_fraction_l982_98284

theorem married_men_fraction (total_women : ℕ) (h_total_women_pos : 0 < total_women) :
  let single_women := (3 * total_women) / 7
  let married_women := total_women - single_women
  let married_men := married_women
  let total_people := total_women + married_men
  (single_women : ℚ) / total_women = 3 / 7 →
  (married_men : ℚ) / total_people = 4 / 11 := by
  sorry

end NUMINAMATH_CALUDE_married_men_fraction_l982_98284


namespace NUMINAMATH_CALUDE_max_students_above_mean_l982_98222

/-- The maximum number of students who can score above the mean in a class of 107 students -/
theorem max_students_above_mean (n : ℕ) (h : n = 107) :
  ∃ (scores : Finset ℝ) (above_mean : Finset ℝ),
    Finset.card scores = n ∧
    Finset.card above_mean ≤ n - 1 ∧
    ∀ x ∈ above_mean, x > (Finset.sum scores id) / n ∧
    ∀ m : ℕ, m > Finset.card above_mean →
      ¬∃ (new_scores : Finset ℝ) (new_above_mean : Finset ℝ),
        Finset.card new_scores = n ∧
        Finset.card new_above_mean = m ∧
        ∀ x ∈ new_above_mean, x > (Finset.sum new_scores id) / n :=
sorry

#check max_students_above_mean

end NUMINAMATH_CALUDE_max_students_above_mean_l982_98222


namespace NUMINAMATH_CALUDE_total_watermelons_l982_98204

def jason_watermelons : ℕ := 37
def sandy_watermelons : ℕ := 11

theorem total_watermelons : jason_watermelons + sandy_watermelons = 48 := by
  sorry

end NUMINAMATH_CALUDE_total_watermelons_l982_98204


namespace NUMINAMATH_CALUDE_plan_a_fixed_charge_l982_98277

/-- The fixed charge for the first 5 minutes under plan A -/
def fixed_charge : ℝ := 0.60

/-- The per-minute rate after the first 5 minutes under plan A -/
def rate_a : ℝ := 0.06

/-- The per-minute rate for plan B -/
def rate_b : ℝ := 0.08

/-- The duration at which both plans charge the same amount -/
def equal_duration : ℝ := 14.999999999999996

theorem plan_a_fixed_charge :
  fixed_charge = rate_b * equal_duration - rate_a * (equal_duration - 5) :=
by sorry

end NUMINAMATH_CALUDE_plan_a_fixed_charge_l982_98277


namespace NUMINAMATH_CALUDE_triangle_inequality_holds_l982_98252

def is_valid_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def triangle_sides (x : ℕ) : ℕ × ℕ × ℕ :=
  (6, x + 3, 2 * x - 1)

theorem triangle_inequality_holds (x : ℕ) :
  (∃ (y : ℕ), y ∈ ({2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) ∧ x = y) ↔
  (let (a, b, c) := triangle_sides x
   is_valid_triangle a b c ∧ x > 0) :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_holds_l982_98252


namespace NUMINAMATH_CALUDE_white_car_rental_cost_l982_98234

/-- Represents the cost of renting a white car per minute -/
def white_car_cost : ℝ := 2

/-- Represents the number of red cars -/
def red_cars : ℕ := 3

/-- Represents the number of white cars -/
def white_cars : ℕ := 2

/-- Represents the cost of renting a red car per minute -/
def red_car_cost : ℝ := 3

/-- Represents the total rental time in minutes -/
def rental_time : ℕ := 3 * 60

/-- Represents the total earnings -/
def total_earnings : ℝ := 2340

theorem white_car_rental_cost :
  red_cars * red_car_cost * rental_time + white_cars * white_car_cost * rental_time = total_earnings :=
by sorry

end NUMINAMATH_CALUDE_white_car_rental_cost_l982_98234


namespace NUMINAMATH_CALUDE_log_inequality_l982_98244

theorem log_inequality (a b : ℝ) (ha : a = Real.log 2 / Real.log 3) (hb : b = Real.log 3 / Real.log 2) :
  Real.log a < (1/2) ^ b := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l982_98244


namespace NUMINAMATH_CALUDE_cloth_sale_meters_l982_98255

/-- Proves that the number of meters of cloth sold is 60, given the total selling price,
    profit per meter, and cost price per meter. -/
theorem cloth_sale_meters (total_selling_price : ℕ) (profit_per_meter : ℕ) (cost_price_per_meter : ℕ)
    (h1 : total_selling_price = 8400)
    (h2 : profit_per_meter = 12)
    (h3 : cost_price_per_meter = 128) :
    total_selling_price / (cost_price_per_meter + profit_per_meter) = 60 := by
  sorry

#check cloth_sale_meters

end NUMINAMATH_CALUDE_cloth_sale_meters_l982_98255


namespace NUMINAMATH_CALUDE_pencil_cost_l982_98261

theorem pencil_cost (x y : ℚ) 
  (eq1 : 5 * x + 2 * y = 286)
  (eq2 : 3 * x + 4 * y = 204) :
  y = 12 := by
  sorry

end NUMINAMATH_CALUDE_pencil_cost_l982_98261


namespace NUMINAMATH_CALUDE_find_multiple_l982_98296

theorem find_multiple (x : ℝ) (m : ℝ) (h1 : x = 13) (h2 : x + x + 2*x + m*x = 104) : m = 4 := by
  sorry

end NUMINAMATH_CALUDE_find_multiple_l982_98296


namespace NUMINAMATH_CALUDE_only_D_is_valid_assignment_l982_98225

-- Define what constitutes a valid assignment statement
def is_valid_assignment (s : String) : Prop :=
  ∃ (var : String) (expr : String), 
    s = var ++ "=" ++ expr ∧ 
    var ≠ expr ∧
    var.length = 1 ∧
    var.all Char.isLower

-- Define the given options
def option_A : String := "5=a"
def option_B : String := "a+2=a"
def option_C : String := "a=b=4"
def option_D : String := "a=2*a"

-- Theorem statement
theorem only_D_is_valid_assignment :
  ¬(is_valid_assignment option_A) ∧
  ¬(is_valid_assignment option_B) ∧
  ¬(is_valid_assignment option_C) ∧
  is_valid_assignment option_D :=
sorry

end NUMINAMATH_CALUDE_only_D_is_valid_assignment_l982_98225


namespace NUMINAMATH_CALUDE_polygon_diagonals_l982_98276

/-- Given a polygon with interior angle sum of 1800°, the number of diagonals from a vertex is 9 -/
theorem polygon_diagonals (n : ℕ) : 
  (n - 2) * 180 = 1800 → n - 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_polygon_diagonals_l982_98276


namespace NUMINAMATH_CALUDE_largest_prime_factor_l982_98260

theorem largest_prime_factor :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ (17^4 + 2*17^3 + 17^2 - 16^4) ∧
  ∀ (q : ℕ), Nat.Prime q → q ∣ (17^4 + 2*17^3 + 17^2 - 16^4) → q ≤ p :=
by
  use 17
  sorry

#check largest_prime_factor

end NUMINAMATH_CALUDE_largest_prime_factor_l982_98260


namespace NUMINAMATH_CALUDE_short_trees_after_planting_l982_98240

theorem short_trees_after_planting 
  (initial_short_trees : ℕ) 
  (short_trees_to_plant : ℕ) 
  (h1 : initial_short_trees = 112)
  (h2 : short_trees_to_plant = 105) :
  initial_short_trees + short_trees_to_plant = 217 := by
  sorry

end NUMINAMATH_CALUDE_short_trees_after_planting_l982_98240


namespace NUMINAMATH_CALUDE_park_fencing_cost_l982_98235

/-- Proves that for a rectangular park with sides in the ratio 3:2, area of 4704 sq m,
    and a total fencing cost of 140, the cost of fencing per meter is 50 paise. -/
theorem park_fencing_cost (length width : ℝ) (area perimeter total_cost : ℝ) : 
  length / width = 3 / 2 →
  area = 4704 →
  length * width = area →
  perimeter = 2 * (length + width) →
  total_cost = 140 →
  (total_cost / perimeter) * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_park_fencing_cost_l982_98235


namespace NUMINAMATH_CALUDE_reciprocal_sum_theorem_l982_98211

theorem reciprocal_sum_theorem (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  x + y = 7 * x * y → 1 / x + 1 / y = 7 := by
sorry

end NUMINAMATH_CALUDE_reciprocal_sum_theorem_l982_98211


namespace NUMINAMATH_CALUDE_multiples_of_2_or_5_not_6_l982_98217

def count_multiples (n : ℕ) (m : ℕ) : ℕ :=
  (n / m)

def count_multiples_of_2_or_5_not_6 (upper_bound : ℕ) : ℕ :=
  (count_multiples upper_bound 2) + (count_multiples upper_bound 5) - 
  (count_multiples upper_bound 10) - (count_multiples upper_bound 6)

theorem multiples_of_2_or_5_not_6 :
  count_multiples_of_2_or_5_not_6 200 = 87 := by
  sorry

end NUMINAMATH_CALUDE_multiples_of_2_or_5_not_6_l982_98217


namespace NUMINAMATH_CALUDE_michael_has_100_cards_l982_98292

/-- The number of Pokemon cards each person has -/
structure PokemonCards where
  lloyd : ℕ
  mark : ℕ
  michael : ℕ

/-- The conditions of the Pokemon card collection problem -/
def PokemonCardsProblem (cards : PokemonCards) : Prop :=
  (cards.mark = 3 * cards.lloyd) ∧
  (cards.michael = cards.mark + 10) ∧
  (cards.lloyd + cards.mark + cards.michael + 80 = 300)

/-- Theorem stating that under the given conditions, Michael has 100 cards -/
theorem michael_has_100_cards (cards : PokemonCards) :
  PokemonCardsProblem cards → cards.michael = 100 := by
  sorry


end NUMINAMATH_CALUDE_michael_has_100_cards_l982_98292


namespace NUMINAMATH_CALUDE_frog_arrangement_count_l982_98251

/-- Represents the number of valid arrangements of frogs -/
def validFrogArrangements (n g r b : ℕ) : ℕ :=
  if n = g + r + b ∧ g ≥ 1 ∧ r ≥ 1 ∧ b = 1 then
    2 * (Nat.factorial r * Nat.factorial g)
  else
    0

/-- Theorem stating the number of valid frog arrangements for the given problem -/
theorem frog_arrangement_count :
  validFrogArrangements 8 3 4 1 = 288 := by
  sorry

end NUMINAMATH_CALUDE_frog_arrangement_count_l982_98251


namespace NUMINAMATH_CALUDE_quadratic_equation_root_l982_98230

theorem quadratic_equation_root (m : ℝ) : 
  (∃ x : ℝ, x^2 - 4*x + m + 2 = 0 ∧ x = -1) → 
  (∃ y : ℝ, y^2 - 4*y + m + 2 = 0 ∧ y = 5) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_root_l982_98230


namespace NUMINAMATH_CALUDE_complex_power_magnitude_l982_98210

theorem complex_power_magnitude (z : ℂ) (h : z = 4/5 + 3/5 * I) :
  Complex.abs (z^8) = 1 := by sorry

end NUMINAMATH_CALUDE_complex_power_magnitude_l982_98210


namespace NUMINAMATH_CALUDE_geometric_series_sum_l982_98290

theorem geometric_series_sum : 
  let a : ℚ := 2/3
  let r : ℚ := -1/2
  let n : ℕ := 6
  let S : ℚ := a * (1 - r^n) / (1 - r)
  S = 7/16 := by sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l982_98290


namespace NUMINAMATH_CALUDE_expression_value_l982_98268

theorem expression_value (x y : ℤ) (hx : x = -6) (hy : y = -3) :
  4 * (x - y)^2 - x * y = 18 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l982_98268


namespace NUMINAMATH_CALUDE_power_of_two_not_sum_of_consecutive_integers_l982_98257

theorem power_of_two_not_sum_of_consecutive_integers :
  ∀ n : ℕ+, (∀ r : ℕ, r > 1 → ¬∃ k : ℕ, n = (k + r) * (k + r - 1) / 2 - k * (k - 1) / 2) ↔
  ∃ l : ℕ, n = 2^l := by sorry

end NUMINAMATH_CALUDE_power_of_two_not_sum_of_consecutive_integers_l982_98257


namespace NUMINAMATH_CALUDE_tangent_line_to_parabola_l982_98241

/-- Given that the line x - y - 1 = 0 is tangent to the parabola y = ax², prove that a = 1/4 -/
theorem tangent_line_to_parabola (a : ℝ) : 
  (∃ x y : ℝ, x - y - 1 = 0 ∧ y = a * x^2 ∧ 
   ∀ x' y' : ℝ, y' = a * x'^2 → (x - x') * (2 * a * x) = y - y') → 
  a = 1/4 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_to_parabola_l982_98241


namespace NUMINAMATH_CALUDE_complex_magnitude_sum_reciprocals_l982_98246

theorem complex_magnitude_sum_reciprocals (z w : ℂ) 
  (hz : Complex.abs z = 2)
  (hw : Complex.abs w = 4)
  (hzw : Complex.abs (z + w) = 3) :
  Complex.abs (1 / z + 1 / w) = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_sum_reciprocals_l982_98246


namespace NUMINAMATH_CALUDE_larger_number_proof_l982_98215

theorem larger_number_proof (a b : ℕ) : 
  a > 0 ∧ b > 0 ∧ 
  Nat.gcd a b = 23 ∧
  ∃ (x y : ℕ), x * y = Nat.lcm a b ∧ x = 13 ∧ y = 14 →
  max a b = 322 := by
sorry

end NUMINAMATH_CALUDE_larger_number_proof_l982_98215


namespace NUMINAMATH_CALUDE_group_size_proof_l982_98223

theorem group_size_proof (total_collection : ℚ) (h1 : total_collection = 92.16) : ∃ n : ℕ, 
  (n : ℚ) * (n : ℚ) / 100 = total_collection ∧ n = 96 := by
  sorry

end NUMINAMATH_CALUDE_group_size_proof_l982_98223


namespace NUMINAMATH_CALUDE_right_triangle_xy_length_l982_98295

theorem right_triangle_xy_length 
  (X Y Z : ℝ × ℝ) 
  (right_angle : (X.1 - Y.1) * (X.1 - Z.1) + (X.2 - Y.2) * (X.2 - Z.2) = 0) 
  (yz_length : Real.sqrt ((Y.1 - Z.1)^2 + (Y.2 - Z.2)^2) = 20)
  (tan_z_eq : (X.2 - Y.2) / (X.1 - Y.1) = 4 * (X.1 - Y.1) / 20) :
  Real.sqrt ((X.1 - Y.1)^2 + (X.2 - Y.2)^2) = 5 * Real.sqrt 15 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_xy_length_l982_98295


namespace NUMINAMATH_CALUDE_total_cost_of_items_l982_98206

theorem total_cost_of_items (wallet_cost : ℕ) 
  (h1 : wallet_cost = 22)
  (purse_cost : ℕ) 
  (h2 : purse_cost = 4 * wallet_cost - 3)
  (shoes_cost : ℕ) 
  (h3 : shoes_cost = wallet_cost + purse_cost + 7) :
  wallet_cost + purse_cost + shoes_cost = 221 := by
sorry

end NUMINAMATH_CALUDE_total_cost_of_items_l982_98206


namespace NUMINAMATH_CALUDE_min_brownies_is_36_l982_98228

/-- Represents the dimensions of a rectangular pan of brownies -/
structure BrowniePan where
  length : ℕ
  width : ℕ

/-- Calculates the total number of brownies in the pan -/
def total_brownies (pan : BrowniePan) : ℕ := pan.length * pan.width

/-- Calculates the number of brownies on the perimeter of the pan -/
def perimeter_brownies (pan : BrowniePan) : ℕ := 2 * (pan.length + pan.width) - 4

/-- Calculates the number of brownies in the interior of the pan -/
def interior_brownies (pan : BrowniePan) : ℕ := (pan.length - 2) * (pan.width - 2)

/-- Checks if the pan satisfies the perimeter-to-interior ratio condition -/
def satisfies_ratio (pan : BrowniePan) : Prop :=
  perimeter_brownies pan = 2 * interior_brownies pan

/-- The main theorem stating that 36 is the smallest number of brownies satisfying all conditions -/
theorem min_brownies_is_36 :
  ∃ (pan : BrowniePan), satisfies_ratio pan ∧
    total_brownies pan = 36 ∧
    (∀ (other_pan : BrowniePan), satisfies_ratio other_pan →
      total_brownies other_pan ≥ 36) :=
  sorry

end NUMINAMATH_CALUDE_min_brownies_is_36_l982_98228


namespace NUMINAMATH_CALUDE_pants_fabric_usage_l982_98299

/-- Proves that each pair of pants uses 5 yards of fabric given the conditions of Jenson and Kingsley's tailoring business. -/
theorem pants_fabric_usage
  (shirts_per_day : ℕ)
  (pants_per_day : ℕ)
  (fabric_per_shirt : ℕ)
  (total_fabric : ℕ)
  (days : ℕ)
  (h1 : shirts_per_day = 3)
  (h2 : pants_per_day = 5)
  (h3 : fabric_per_shirt = 2)
  (h4 : total_fabric = 93)
  (h5 : days = 3) :
  (total_fabric - shirts_per_day * days * fabric_per_shirt) / (pants_per_day * days) = 5 :=
sorry

end NUMINAMATH_CALUDE_pants_fabric_usage_l982_98299


namespace NUMINAMATH_CALUDE_range_of_m_part1_range_of_m_part2_l982_98220

-- Define sets A and B
def A : Set ℝ := {x | x^2 + 3*x - 28 < 0}
def B (m : ℝ) : Set ℝ := {x | m - 2 < x ∧ x < m + 1}

-- Part 1
theorem range_of_m_part1 (m : ℝ) (h : 3 ∈ B m) : 2 < m ∧ m < 5 := by
  sorry

-- Part 2
theorem range_of_m_part2 (m : ℝ) (h1 : B m ⊆ A) (h2 : B m ≠ A) : -5 ≤ m ∧ m ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_part1_range_of_m_part2_l982_98220


namespace NUMINAMATH_CALUDE_hyperbola_foci_distance_l982_98273

/-- The distance between the foci of a hyperbola defined by xy = 2 is 4. -/
theorem hyperbola_foci_distance :
  ∃ (f₁ f₂ : ℝ × ℝ), 
    (∀ (x y : ℝ), x * y = 2 → 
      (Real.sqrt ((x - f₁.1)^2 + (y - f₁.2)^2) + Real.sqrt ((x - f₂.1)^2 + (y - f₂.2)^2)) = 
      (Real.sqrt ((x + f₁.1)^2 + (y + f₁.2)^2) + Real.sqrt ((x + f₂.1)^2 + (y + f₂.2)^2))) ∧
    Real.sqrt ((f₁.1 - f₂.1)^2 + (f₁.2 - f₂.2)^2) = 4 :=
by
  sorry


end NUMINAMATH_CALUDE_hyperbola_foci_distance_l982_98273


namespace NUMINAMATH_CALUDE_highest_score_is_96_l982_98272

def standard_score : ℝ := 85

def deviations : List ℝ := [-9, -4, 11, -7, 0]

def actual_scores : List ℝ := deviations.map (λ x => x + standard_score)

theorem highest_score_is_96 : 
  ∀ (score : ℝ), score ∈ actual_scores → score ≤ 96 :=
by sorry

end NUMINAMATH_CALUDE_highest_score_is_96_l982_98272


namespace NUMINAMATH_CALUDE_greatest_power_of_two_factor_l982_98271

theorem greatest_power_of_two_factor (n : ℕ) : 
  (∃ k : ℕ, 2^k ∣ (10^1000 + 4^500) ∧ 
   ∀ m : ℕ, 2^m ∣ (10^1000 + 4^500) → m ≤ k) → 
  n = 1003 :=
sorry

end NUMINAMATH_CALUDE_greatest_power_of_two_factor_l982_98271


namespace NUMINAMATH_CALUDE_gcd_lcm_product_l982_98248

theorem gcd_lcm_product (a b : ℕ) (ha : a = 108) (hb : b = 250) :
  Nat.gcd a b * Nat.lcm a b = a * b := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_l982_98248


namespace NUMINAMATH_CALUDE_cosine_product_equality_l982_98289

theorem cosine_product_equality : 
  3.416 * Real.cos (π/33) * Real.cos (2*π/33) * Real.cos (4*π/33) * Real.cos (8*π/33) * Real.cos (16*π/33) = 1/32 := by
  sorry

end NUMINAMATH_CALUDE_cosine_product_equality_l982_98289


namespace NUMINAMATH_CALUDE_quadratic_curve_point_exclusion_l982_98212

theorem quadratic_curve_point_exclusion (a c : ℝ) (h : a * c > 0) :
  ¬∃ d : ℝ, 0 = a * 2018^2 + c * 2018 + d := by
  sorry

end NUMINAMATH_CALUDE_quadratic_curve_point_exclusion_l982_98212


namespace NUMINAMATH_CALUDE_correct_total_cost_l982_98224

/-- The cost of a single sandwich -/
def sandwich_cost : ℕ := 4

/-- The cost of a single soda -/
def soda_cost : ℕ := 3

/-- The number of sandwiches purchased -/
def num_sandwiches : ℕ := 4

/-- The number of sodas purchased -/
def num_sodas : ℕ := 5

/-- The total cost of the purchase -/
def total_cost : ℕ := sandwich_cost * num_sandwiches + soda_cost * num_sodas

theorem correct_total_cost : total_cost = 31 := by
  sorry

end NUMINAMATH_CALUDE_correct_total_cost_l982_98224


namespace NUMINAMATH_CALUDE_original_denominator_problem_l982_98207

theorem original_denominator_problem (d : ℚ) : 
  (3 : ℚ) / d ≠ 0 →
  (3 + 7 : ℚ) / (d + 7) = 1 / 3 →
  d = 23 := by
sorry

end NUMINAMATH_CALUDE_original_denominator_problem_l982_98207


namespace NUMINAMATH_CALUDE_amanda_remaining_money_l982_98283

/-- Calculates the remaining amount after purchases -/
def remaining_amount (initial_amount : ℕ) (item1_cost : ℕ) (item1_quantity : ℕ) (item2_cost : ℕ) : ℕ :=
  initial_amount - (item1_cost * item1_quantity + item2_cost)

/-- Proves that Amanda will have $7 left after her purchases -/
theorem amanda_remaining_money :
  remaining_amount 50 9 2 25 = 7 := by
  sorry

end NUMINAMATH_CALUDE_amanda_remaining_money_l982_98283


namespace NUMINAMATH_CALUDE_jiwon_walk_distance_l982_98254

theorem jiwon_walk_distance 
  (sets_of_steps : ℕ) 
  (steps_per_set : ℕ) 
  (distance_per_step : ℝ) : 
  sets_of_steps = 13 → 
  steps_per_set = 90 → 
  distance_per_step = 0.45 → 
  (sets_of_steps * steps_per_set : ℝ) * distance_per_step = 526.5 := by
sorry

end NUMINAMATH_CALUDE_jiwon_walk_distance_l982_98254


namespace NUMINAMATH_CALUDE_pencil_boxes_count_l982_98288

theorem pencil_boxes_count (pencils_per_box : ℝ) (total_pencils : ℕ) 
  (h1 : pencils_per_box = 648.0) 
  (h2 : total_pencils = 2592) : 
  ↑total_pencils / pencils_per_box = 4 := by
  sorry

end NUMINAMATH_CALUDE_pencil_boxes_count_l982_98288


namespace NUMINAMATH_CALUDE_min_value_theorem_l982_98258

theorem min_value_theorem (x y : ℝ) (hx : x > 2) (hy : y > 2) :
  (x + 2)^2 / (y - 2) + (y + 2)^2 / (x - 2) ≥ 50 ∧
  ((x + 2)^2 / (y - 2) + (y + 2)^2 / (x - 2) = 50 ↔ x = 3 ∧ y = 3) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l982_98258


namespace NUMINAMATH_CALUDE_bike_rental_cost_l982_98267

theorem bike_rental_cost 
  (daily_rate : ℝ) 
  (mileage_rate : ℝ) 
  (rental_days : ℕ) 
  (miles_biked : ℝ) 
  (h1 : daily_rate = 15)
  (h2 : mileage_rate = 0.1)
  (h3 : rental_days = 3)
  (h4 : miles_biked = 300) :
  daily_rate * ↑rental_days + mileage_rate * miles_biked = 75 :=
by sorry

end NUMINAMATH_CALUDE_bike_rental_cost_l982_98267


namespace NUMINAMATH_CALUDE_other_factor_proof_l982_98200

theorem other_factor_proof (w : ℕ) (h1 : w > 0) 
  (h2 : ∃ k : ℕ, 936 * w = 2^5 * 13^2 * k) 
  (h3 : w ≥ 156) 
  (h4 : ∀ v : ℕ, v > 0 → v < 156 → ¬(∃ k : ℕ, 936 * v = 2^5 * 13^2 * k)) : 
  ∃ m : ℕ, w = 3 * m ∧ ∃ k : ℕ, 936 * m = 2^5 * 13^2 * k := by
sorry

end NUMINAMATH_CALUDE_other_factor_proof_l982_98200


namespace NUMINAMATH_CALUDE_adjusted_smallest_part_is_correct_l982_98233

-- Define the total amount
def total : ℚ := 100

-- Define the proportions
def proportions : List ℚ := [1, 3, 4, 6]

-- Define the extra amount added to the smallest part
def extra : ℚ := 12

-- Define the function to calculate the adjusted smallest part
def adjusted_smallest_part (total : ℚ) (proportions : List ℚ) (extra : ℚ) : ℚ :=
  let sum_proportions := proportions.sum
  let smallest_part := total * (proportions.head! / sum_proportions)
  smallest_part + extra

-- Theorem statement
theorem adjusted_smallest_part_is_correct :
  adjusted_smallest_part total proportions extra = 19 + 1/7 := by
  sorry

end NUMINAMATH_CALUDE_adjusted_smallest_part_is_correct_l982_98233


namespace NUMINAMATH_CALUDE_environmental_group_allocation_l982_98281

theorem environmental_group_allocation :
  let total_members : ℕ := 8
  let num_locations : ℕ := 3
  let min_per_location : ℕ := 2

  let allocation_schemes : ℕ := 
    (Nat.choose total_members 2 * Nat.choose 6 2 * Nat.choose 4 4 / 2 +
     Nat.choose total_members 3 * Nat.choose 5 3 * Nat.choose 2 2 / 2) * 
    (Nat.factorial num_locations)

  allocation_schemes = 2940 :=
by sorry

end NUMINAMATH_CALUDE_environmental_group_allocation_l982_98281


namespace NUMINAMATH_CALUDE_fraction_zero_implies_a_neg_two_l982_98282

theorem fraction_zero_implies_a_neg_two (a : ℝ) :
  (a^2 - 4) / (a - 2) = 0 → a = -2 := by
sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_a_neg_two_l982_98282


namespace NUMINAMATH_CALUDE_union_of_sets_l982_98243

theorem union_of_sets : 
  let A : Set ℕ := {1, 2, 3}
  let B : Set ℕ := {2, 3, 4}
  A ∪ B = {1, 2, 3, 4} := by
  sorry

end NUMINAMATH_CALUDE_union_of_sets_l982_98243


namespace NUMINAMATH_CALUDE_new_ratio_after_addition_l982_98286

theorem new_ratio_after_addition : 
  ∀ (x y : ℤ), 
    x * 4 = y →  -- The two integers are in the ratio of 1 to 4
    y = 48 →     -- The larger integer is 48
    (x + 12) * 2 = y  -- The new ratio after adding 12 to the smaller integer is 1:2
    := by sorry

end NUMINAMATH_CALUDE_new_ratio_after_addition_l982_98286


namespace NUMINAMATH_CALUDE_salary_solution_l982_98219

def salary_problem (salary : ℝ) : Prop :=
  let food_expense := (1 / 5 : ℝ) * salary
  let rent_expense := (1 / 10 : ℝ) * salary
  let clothes_expense := (3 / 5 : ℝ) * salary
  let remaining := salary - (food_expense + rent_expense + clothes_expense)
  remaining = 19000

theorem salary_solution :
  ∃ (salary : ℝ), salary_problem salary ∧ salary = 190000 := by
  sorry

end NUMINAMATH_CALUDE_salary_solution_l982_98219


namespace NUMINAMATH_CALUDE_parabola_line_intersection_property_l982_98263

/-- Parabola type representing y² = 4x -/
structure Parabola where
  focus : ℝ × ℝ
  directrix : ℝ

/-- Line type representing y = k(x-1) -/
structure Line where
  k : ℝ

/-- Point type -/
structure Point where
  x : ℝ
  y : ℝ

/-- Vector dot product -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem parabola_line_intersection_property
  (C : Parabola)
  (l : Line)
  (A B M N O : Point)
  (hC : C.focus = (1, 0) ∧ C.directrix = -1)
  (hl : l.k ≠ 0)
  (hA : A.y^2 = 4 * A.x ∧ A.y = l.k * (A.x - 1))
  (hB : B.y^2 = 4 * B.x ∧ B.y = l.k * (B.x - 1))
  (hM : M.x = -1 ∧ M.y * A.x = -A.y)
  (hN : N.x = -1 ∧ N.y * B.x = -B.y)
  (hO : O.x = 0 ∧ O.y = 0) :
  dot_product (M.x - O.x, M.y - O.y) (N.x - O.x, N.y - O.y) =
  dot_product (A.x - O.x, A.y - O.y) (B.x - O.x, B.y - O.y) :=
sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_property_l982_98263


namespace NUMINAMATH_CALUDE_oak_trees_remaining_l982_98242

/-- The number of oak trees remaining after cutting down damaged trees -/
def remaining_oak_trees (initial : ℕ) (cut_down : ℕ) : ℕ :=
  initial - cut_down

/-- Theorem stating that the number of oak trees remaining is 7 -/
theorem oak_trees_remaining :
  remaining_oak_trees 9 2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_oak_trees_remaining_l982_98242


namespace NUMINAMATH_CALUDE_inequality_system_solution_range_l982_98201

theorem inequality_system_solution_range (a : ℝ) : 
  (∃ (x y z : ℤ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    (2 * ↑x - 1 > 3 ∧ ↑x ≤ 2 * a - 1) ∧
    (2 * ↑y - 1 > 3 ∧ ↑y ≤ 2 * a - 1) ∧
    (2 * ↑z - 1 > 3 ∧ ↑z ≤ 2 * a - 1) ∧
    (∀ (w : ℤ), w ≠ x ∧ w ≠ y ∧ w ≠ z → ¬(2 * ↑w - 1 > 3 ∧ ↑w ≤ 2 * a - 1))) →
  (3 ≤ a ∧ a < 3.5) :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_range_l982_98201


namespace NUMINAMATH_CALUDE_sequence_product_l982_98213

theorem sequence_product : 
  (1/4) * 16 * (1/64) * 256 * (1/1024) * 4096 * (1/16384) * 65536 = 256 := by
  sorry

end NUMINAMATH_CALUDE_sequence_product_l982_98213


namespace NUMINAMATH_CALUDE_only_zero_factorizable_l982_98239

/-- The polynomial we're considering -/
def poly (m : ℤ) (x y : ℤ) : ℤ := x^2 + 4*x*y + x + m*y - 2*m

/-- A linear factor with integer coefficients -/
def linear_factor (a b c : ℤ) (x y : ℤ) : ℤ := a*x + b*y + c

/-- Predicate to check if the polynomial can be factored into two linear factors with integer coefficients -/
def can_be_factored (m : ℤ) : Prop :=
  ∃ (a₁ b₁ c₁ a₂ b₂ c₂ : ℤ), ∀ (x y : ℤ),
    poly m x y = linear_factor a₁ b₁ c₁ x y * linear_factor a₂ b₂ c₂ x y

theorem only_zero_factorizable :
  ∀ m : ℤ, can_be_factored m ↔ m = 0 :=
sorry

end NUMINAMATH_CALUDE_only_zero_factorizable_l982_98239


namespace NUMINAMATH_CALUDE_max_teams_intramurals_l982_98247

/-- Represents the number of participants in each category -/
structure Participants where
  girls : Nat
  boys : Nat
  teenagers : Nat

/-- Represents the sports preferences for girls -/
structure GirlsPreferences where
  basketball : Nat
  volleyball : Nat
  soccer : Nat

/-- Represents the sports preferences for boys -/
structure BoysPreferences where
  basketball : Nat
  soccer : Nat

/-- Represents the sports preferences for teenagers -/
structure TeenagersPreferences where
  volleyball : Nat
  mixed_sports : Nat

/-- The main theorem statement -/
theorem max_teams_intramurals
  (total : Participants)
  (girls_pref : GirlsPreferences)
  (boys_pref : BoysPreferences)
  (teens_pref : TeenagersPreferences)
  (h1 : total.girls = 120)
  (h2 : total.boys = 96)
  (h3 : total.teenagers = 72)
  (h4 : girls_pref.basketball = 40)
  (h5 : girls_pref.volleyball = 50)
  (h6 : girls_pref.soccer = 30)
  (h7 : boys_pref.basketball = 48)
  (h8 : boys_pref.soccer = 48)
  (h9 : teens_pref.volleyball = 24)
  (h10 : teens_pref.mixed_sports = 48)
  (h11 : girls_pref.basketball + girls_pref.volleyball + girls_pref.soccer = total.girls)
  (h12 : boys_pref.basketball + boys_pref.soccer = total.boys)
  (h13 : teens_pref.volleyball + teens_pref.mixed_sports = total.teenagers) :
  ∃ (n : Nat), n = 24 ∧ 
    n ∣ total.girls ∧ 
    n ∣ total.boys ∧ 
    n ∣ total.teenagers ∧
    n ∣ girls_pref.basketball ∧
    n ∣ girls_pref.volleyball ∧
    n ∣ girls_pref.soccer ∧
    n ∣ boys_pref.basketball ∧
    n ∣ boys_pref.soccer ∧
    n ∣ teens_pref.volleyball ∧
    n ∣ teens_pref.mixed_sports ∧
    ∀ (m : Nat), (m > n) → 
      ¬(m ∣ total.girls ∧ 
        m ∣ total.boys ∧ 
        m ∣ total.teenagers ∧
        m ∣ girls_pref.basketball ∧
        m ∣ girls_pref.volleyball ∧
        m ∣ girls_pref.soccer ∧
        m ∣ boys_pref.basketball ∧
        m ∣ boys_pref.soccer ∧
        m ∣ teens_pref.volleyball ∧
        m ∣ teens_pref.mixed_sports) :=
by
  sorry

end NUMINAMATH_CALUDE_max_teams_intramurals_l982_98247


namespace NUMINAMATH_CALUDE_wizard_elixir_combinations_l982_98253

/-- The number of magical herbs available. -/
def num_herbs : ℕ := 4

/-- The number of mystical crystals available. -/
def num_crystals : ℕ := 6

/-- The number of incompatible crystals. -/
def num_incompatible_crystals : ℕ := 2

/-- The number of herbs incompatible with some crystals. -/
def num_incompatible_herbs : ℕ := 3

/-- The number of valid combinations for the wizard's elixir. -/
def valid_combinations : ℕ := num_herbs * num_crystals - num_incompatible_crystals * num_incompatible_herbs

theorem wizard_elixir_combinations :
  valid_combinations = 18 :=
sorry

end NUMINAMATH_CALUDE_wizard_elixir_combinations_l982_98253


namespace NUMINAMATH_CALUDE_min_points_for_isosceles_l982_98221

-- Define a type for the lattice points
def LatticePoint : Type := ℕ × ℕ

-- Define a regular triangle with 15 lattice points
def RegularTriangle : Type := List LatticePoint

-- Function to check if three points form an isosceles triangle
def isIsosceles (p1 p2 p3 : LatticePoint) : Prop :=
  sorry

-- Function to check if a list of points contains an isosceles triangle
def containsIsosceles (points : List LatticePoint) : Prop :=
  sorry

-- Theorem statement
theorem min_points_for_isosceles (triangle : RegularTriangle) 
  (h1 : triangle.length = 15) : 
  ∃ (n : ℕ), n = 6 ∧ 
  (∀ (chosen : List LatticePoint), 
    chosen.length ≥ n → chosen ⊆ triangle → containsIsosceles chosen) ∧
  (∀ (m : ℕ), m < n → 
    ∃ (chosen : List LatticePoint), 
      chosen.length = m ∧ chosen ⊆ triangle ∧ ¬containsIsosceles chosen) :=
sorry

end NUMINAMATH_CALUDE_min_points_for_isosceles_l982_98221


namespace NUMINAMATH_CALUDE_option_c_is_experimental_l982_98266

-- Define a type for survey methods
inductive SurveyMethod
| Direct
| Experimental
| SecondaryData

-- Define a type for survey options
inductive SurveyOption
| A
| B
| C
| D

-- Define a function that assigns a survey method to each option
def survey_method (option : SurveyOption) : SurveyMethod :=
  match option with
  | SurveyOption.A => SurveyMethod.Direct
  | SurveyOption.B => SurveyMethod.Direct
  | SurveyOption.C => SurveyMethod.Experimental
  | SurveyOption.D => SurveyMethod.SecondaryData

-- Define the experimental method suitability
def is_suitable_for_experimental (method : SurveyMethod) : Prop :=
  method = SurveyMethod.Experimental

-- Theorem: Option C is the only one suitable for the experimental method
theorem option_c_is_experimental :
  ∀ (option : SurveyOption),
    is_suitable_for_experimental (survey_method option) ↔ option = SurveyOption.C :=
by
  sorry

#check option_c_is_experimental

end NUMINAMATH_CALUDE_option_c_is_experimental_l982_98266


namespace NUMINAMATH_CALUDE_cookie_difference_l982_98274

/-- Given that Alyssa has 129 cookies and Aiyanna has 140 cookies,
    prove that Aiyanna has 11 more cookies than Alyssa. -/
theorem cookie_difference (alyssa_cookies : ℕ) (aiyanna_cookies : ℕ)
    (h1 : alyssa_cookies = 129)
    (h2 : aiyanna_cookies = 140) :
    aiyanna_cookies - alyssa_cookies = 11 := by
  sorry

end NUMINAMATH_CALUDE_cookie_difference_l982_98274


namespace NUMINAMATH_CALUDE_weight_loss_difference_l982_98279

-- Define weight loss patterns
def barbi_loss_year1 : ℝ := 1.5 * 12
def barbi_loss_year2_3 : ℝ := 2.2 * 12 * 2

def luca_loss_year1 : ℝ := 9
def luca_loss_year2 : ℝ := 12
def luca_loss_year3_7 : ℝ := (12 + 3 * 5)

def kim_loss_year1 : ℝ := 2 * 12
def kim_loss_year2_3 : ℝ := 3 * 12 * 2
def kim_loss_year4_6 : ℝ := 1 * 12 * 3

-- Calculate total weight loss for each person
def barbi_total_loss : ℝ := barbi_loss_year1 + barbi_loss_year2_3
def luca_total_loss : ℝ := luca_loss_year1 + luca_loss_year2 + 5 * luca_loss_year3_7
def kim_total_loss : ℝ := kim_loss_year1 + kim_loss_year2_3 + kim_loss_year4_6

-- Theorem to prove
theorem weight_loss_difference :
  luca_total_loss + kim_total_loss - barbi_total_loss = 217.2 := by
  sorry

end NUMINAMATH_CALUDE_weight_loss_difference_l982_98279


namespace NUMINAMATH_CALUDE_sqrt_6_plus_sqrt_6_equals_3_l982_98262

theorem sqrt_6_plus_sqrt_6_equals_3 :
  ∃ x : ℝ, x > 0 ∧ x = Real.sqrt (6 + x) → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_6_plus_sqrt_6_equals_3_l982_98262


namespace NUMINAMATH_CALUDE_band_size_correct_l982_98291

/-- The number of flutes that tried out -/
def flutes : ℕ := 20

/-- The number of clarinets that tried out -/
def clarinets : ℕ := 30

/-- The number of trumpets that tried out -/
def trumpets : ℕ := 60

/-- The number of pianists that tried out -/
def pianists : ℕ := 20

/-- The fraction of flutes that got in -/
def flute_acceptance : ℚ := 4/5

/-- The fraction of clarinets that got in -/
def clarinet_acceptance : ℚ := 1/2

/-- The fraction of trumpets that got in -/
def trumpet_acceptance : ℚ := 1/3

/-- The fraction of pianists that got in -/
def pianist_acceptance : ℚ := 1/10

/-- The total number of people in the band -/
def band_total : ℕ := 53

theorem band_size_correct :
  (flutes : ℚ) * flute_acceptance +
  (clarinets : ℚ) * clarinet_acceptance +
  (trumpets : ℚ) * trumpet_acceptance +
  (pianists : ℚ) * pianist_acceptance = band_total := by
  sorry

end NUMINAMATH_CALUDE_band_size_correct_l982_98291


namespace NUMINAMATH_CALUDE_g_equivalence_l982_98227

theorem g_equivalence (x : Real) : 
  Real.sqrt (Real.cos x ^ 4 + 4 * Real.sin x ^ 2) - 
  Real.sqrt (Real.sin x ^ 4 + 4 * Real.cos x ^ 2) = 
  -Real.cos (2 * x) := by sorry

end NUMINAMATH_CALUDE_g_equivalence_l982_98227


namespace NUMINAMATH_CALUDE_quotient_minus_fraction_number_plus_half_l982_98287

-- Question 1
theorem quotient_minus_fraction (a b c : ℚ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (c / a) / (c / b) - c = c * (b - a) / a / b :=
sorry

-- Question 2
theorem number_plus_half (x : ℚ) :
  x + (1/2) * x = 12/5 ↔ x = (12/5 - 1/2) / (3/2) :=
sorry

end NUMINAMATH_CALUDE_quotient_minus_fraction_number_plus_half_l982_98287


namespace NUMINAMATH_CALUDE_lawrence_county_kids_l982_98256

/-- The number of kids in Lawrence county -/
def total_kids : ℕ := 644997 + 893835

/-- The number of kids who stayed home -/
def kids_at_home : ℕ := 644997

/-- The number of kids who went to camp from the county -/
def kids_at_camp : ℕ := 893835

/-- Theorem: The total number of kids in Lawrence county is equal to
    the sum of kids who stayed home and kids who went to camp -/
theorem lawrence_county_kids : total_kids = kids_at_home + kids_at_camp := by
  sorry

end NUMINAMATH_CALUDE_lawrence_county_kids_l982_98256


namespace NUMINAMATH_CALUDE_calculation_proof_l982_98278

theorem calculation_proof : 2359 + 180 / 60 * 3 - 359 = 2009 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l982_98278


namespace NUMINAMATH_CALUDE_race_time_calculation_l982_98245

/-- Given a race where runner A beats runner B by both distance and time, 
    this theorem proves the time taken by runner A to complete the race. -/
theorem race_time_calculation (race_distance : ℝ) (distance_diff : ℝ) (time_diff : ℝ) :
  race_distance = 1000 ∧ 
  distance_diff = 48 ∧ 
  time_diff = 12 →
  ∃ (time_A : ℝ), time_A = 250 ∧ 
    race_distance / time_A = (race_distance - distance_diff) / (time_A + time_diff) :=
by sorry

end NUMINAMATH_CALUDE_race_time_calculation_l982_98245


namespace NUMINAMATH_CALUDE_comparison_of_powers_l982_98285

theorem comparison_of_powers (a b c : ℕ) : 
  a = 81^31 → b = 27^41 → c = 9^61 → a > b ∧ b > c :=
by sorry

end NUMINAMATH_CALUDE_comparison_of_powers_l982_98285


namespace NUMINAMATH_CALUDE_five_people_six_chairs_l982_98265

/-- The number of ways to arrange n people in m chairs -/
def arrange (n : ℕ) (m : ℕ) : ℕ := sorry

/-- There are 5 people and 6 chairs -/
def num_people : ℕ := 5
def num_chairs : ℕ := 6

theorem five_people_six_chairs :
  arrange num_people num_chairs = 720 := by sorry

end NUMINAMATH_CALUDE_five_people_six_chairs_l982_98265


namespace NUMINAMATH_CALUDE_f_has_unique_zero_and_g_max_a_l982_98232

noncomputable def f (x : ℝ) := (x - 2) * Real.log x + 2 * x - 3

noncomputable def g (a : ℝ) (x : ℝ) := (x - a) * Real.log x + a * (x - 1) / x

theorem f_has_unique_zero_and_g_max_a :
  (∃! x : ℝ, x ≥ 1 ∧ f x = 0) ∧
  (∀ a : ℝ, a > 6 → ∃ x₁ x₂ : ℝ, 1 ≤ x₁ ∧ x₁ < x₂ ∧ g a x₂ < g a x₁) ∧
  (∀ x₁ x₂ : ℝ, 1 ≤ x₁ → x₁ < x₂ → g 6 x₁ ≤ g 6 x₂) :=
by sorry

end NUMINAMATH_CALUDE_f_has_unique_zero_and_g_max_a_l982_98232


namespace NUMINAMATH_CALUDE_intersection_sum_l982_98226

-- Define the parabolas
def parabola1 (x y : ℝ) : Prop := y = (x - 2)^2
def parabola2 (x y : ℝ) : Prop := x + 6 = (y + 1)^2

-- Define the intersection points
def intersection_points (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ) : Prop :=
  parabola1 x₁ y₁ ∧ parabola2 x₁ y₁ ∧
  parabola1 x₂ y₂ ∧ parabola2 x₂ y₂ ∧
  parabola1 x₃ y₃ ∧ parabola2 x₃ y₃ ∧
  parabola1 x₄ y₄ ∧ parabola2 x₄ y₄

-- Theorem statement
theorem intersection_sum (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ) :
  intersection_points x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ →
  x₁ + x₂ + x₃ + x₄ + y₁ + y₂ + y₃ + y₄ = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_sum_l982_98226


namespace NUMINAMATH_CALUDE_two_cones_intersection_angle_l982_98202

/-- Represents a cone with height and base radius -/
structure Cone where
  height : ℝ
  baseRadius : ℝ

/-- Represents the configuration of two cones -/
structure TwoCones where
  cone1 : Cone
  cone2 : Cone
  commonVertex : Bool
  touchingEachOther : Bool
  touchingPlane : Bool

/-- The angle between the line of intersection of the base planes and the touching plane -/
def intersectionAngle (tc : TwoCones) : ℝ := sorry

theorem two_cones_intersection_angle 
  (tc : TwoCones) 
  (h1 : tc.cone1 = tc.cone2) 
  (h2 : tc.cone1.height = 2) 
  (h3 : tc.cone1.baseRadius = 1) 
  (h4 : tc.commonVertex = true) 
  (h5 : tc.touchingEachOther = true) 
  (h6 : tc.touchingPlane = true) : 
  intersectionAngle tc = Real.pi / 3 := by sorry

end NUMINAMATH_CALUDE_two_cones_intersection_angle_l982_98202


namespace NUMINAMATH_CALUDE_number_problem_l982_98249

theorem number_problem :
  ∃ x : ℝ, x = (1/4) * x + 93.33333333333333 ∧ x = 124.44444444444444 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l982_98249


namespace NUMINAMATH_CALUDE_smallest_norm_l982_98293

open Real

/-- Given a vector v in ℝ² such that ‖v + (4, -2)‖ = 10, 
    the smallest possible value of ‖v‖ is 10 - 2√5 -/
theorem smallest_norm (v : ℝ × ℝ) 
  (h : ‖v + (4, -2)‖ = 10) : 
  ∃ (w : ℝ × ℝ), ‖w‖ = 10 - 2 * sqrt 5 ∧ ∀ u : ℝ × ℝ, ‖u + (4, -2)‖ = 10 → ‖w‖ ≤ ‖u‖ :=
sorry

end NUMINAMATH_CALUDE_smallest_norm_l982_98293


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l982_98237

theorem tangent_line_to_circle (x y : ℝ) :
  (x^2 + y^2 = 4) →  -- Circle equation
  (1^2 + (Real.sqrt 3)^2 = 4) →  -- Point (1, √3) is on the circle
  (x + Real.sqrt 3 * y = 4) →  -- Proposed tangent line equation
  ∃ (k : ℝ), k * (x - 1) + Real.sqrt 3 * k * (y - Real.sqrt 3) = 0  -- Tangent line property
  :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l982_98237


namespace NUMINAMATH_CALUDE_flower_survival_rate_l982_98231

theorem flower_survival_rate 
  (total_flowers : ℕ) 
  (dead_flowers : ℕ) 
  (h1 : total_flowers = 150) 
  (h2 : dead_flowers = 3) :
  (total_flowers - dead_flowers : ℚ) / total_flowers * 100 = 98 := by
  sorry

end NUMINAMATH_CALUDE_flower_survival_rate_l982_98231

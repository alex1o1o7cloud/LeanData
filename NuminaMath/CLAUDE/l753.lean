import Mathlib

namespace arlene_hike_distance_l753_75328

/-- Calculates the total distance hiked given the hiking time and average pace. -/
def total_distance (time : ℝ) (pace : ℝ) : ℝ := time * pace

/-- Proves that Arlene hiked 24 miles on Saturday. -/
theorem arlene_hike_distance :
  let time : ℝ := 6 -- hours
  let pace : ℝ := 4 -- miles per hour
  total_distance time pace = 24 := by
  sorry

end arlene_hike_distance_l753_75328


namespace min_candies_count_l753_75399

theorem min_candies_count (c : ℕ) : 
  c % 6 = 5 → 
  c % 8 = 7 → 
  c % 9 = 6 → 
  c % 11 = 0 → 
  (∀ n : ℕ, n < c → 
    (n % 6 = 5 ∧ n % 8 = 7 ∧ n % 9 = 6 ∧ n % 11 = 0) → False) → 
  c = 359 := by
sorry

end min_candies_count_l753_75399


namespace smallest_factor_b_l753_75350

theorem smallest_factor_b : 
  ∀ b : ℕ+, 
    (∃ (p q : ℤ), (∀ x : ℝ, x^2 + b * x + 2016 = (x + p) * (x + q))) →
    b ≥ 92 :=
by sorry

end smallest_factor_b_l753_75350


namespace simplify_expression_l753_75349

theorem simplify_expression (x : ℝ) : 1 - (2 * (1 - (1 + (1 - (3 - x))))) = -3 + 2*x := by
  sorry

end simplify_expression_l753_75349


namespace simplify_expression_l753_75373

theorem simplify_expression (x : ℝ) : (3*x)^3 - (4*x^2)*(2*x^3) = 27*x^3 - 8*x^5 := by
  sorry

end simplify_expression_l753_75373


namespace imaginary_part_of_z_l753_75324

theorem imaginary_part_of_z (z : ℂ) (h : z * (1 + Complex.I) = 4) : 
  z.im = -2 := by sorry

end imaginary_part_of_z_l753_75324


namespace alice_stool_height_l753_75336

/-- The height of the ceiling above the floor in centimeters -/
def ceiling_height : ℝ := 300

/-- The distance of the light bulb below the ceiling in centimeters -/
def light_bulb_below_ceiling : ℝ := 15

/-- Alice's height in centimeters -/
def alice_height : ℝ := 150

/-- The distance Alice can reach above her head in centimeters -/
def alice_reach : ℝ := 40

/-- The minimum height of the stool Alice needs in centimeters -/
def stool_height : ℝ := 95

theorem alice_stool_height : 
  ceiling_height - light_bulb_below_ceiling = alice_height + alice_reach + stool_height := by
  sorry

end alice_stool_height_l753_75336


namespace yellow_ball_fraction_l753_75360

theorem yellow_ball_fraction (total : ℕ) (green blue white yellow : ℕ) : 
  (green : ℚ) / total = 1 / 4 →
  (blue : ℚ) / total = 1 / 8 →
  white = 26 →
  blue = 6 →
  total = green + blue + white + yellow →
  (yellow : ℚ) / total = 1 / 12 := by
  sorry

end yellow_ball_fraction_l753_75360


namespace sydney_initial_rocks_l753_75309

/-- Rock collecting contest between Sydney and Conner --/
def rock_contest (sydney_initial : ℕ) : Prop :=
  let conner_initial := 723
  let sydney_day1 := 4
  let conner_day1 := 8 * sydney_day1
  let sydney_day2 := 0
  let conner_day2 := 123
  let sydney_day3 := 2 * conner_day1
  let conner_day3 := 27

  let sydney_total := sydney_initial + sydney_day1 + sydney_day2 + sydney_day3
  let conner_total := conner_initial + conner_day1 + conner_day2 + conner_day3

  sydney_total ≤ conner_total ∧ sydney_initial = 837

theorem sydney_initial_rocks : rock_contest 837 := by
  sorry

end sydney_initial_rocks_l753_75309


namespace factor_x9_minus_512_l753_75356

theorem factor_x9_minus_512 (x : ℝ) : 
  x^9 - 512 = (x - 2) * (x^2 + 2*x + 4) * (x^6 + 8*x^3 + 64) := by
  sorry

end factor_x9_minus_512_l753_75356


namespace jennifer_cards_left_l753_75367

/-- Given that Jennifer has 72 cards initially and 61 cards are eaten,
    prove that she will have 11 cards left. -/
theorem jennifer_cards_left (initial_cards : ℕ) (eaten_cards : ℕ) 
  (h1 : initial_cards = 72) 
  (h2 : eaten_cards = 61) : 
  initial_cards - eaten_cards = 11 := by
  sorry

end jennifer_cards_left_l753_75367


namespace fraction_equality_l753_75387

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (5 * x + y) / (x - 5 * y) = -3) : 
  (x + 5 * y) / (5 * x - y) = 27 / 31 := by
  sorry

end fraction_equality_l753_75387


namespace jacob_coin_problem_l753_75365

theorem jacob_coin_problem :
  ∃ (p n d : ℕ),
    p + n + d = 50 ∧
    p + 5 * n + 10 * d = 220 ∧
    d = 18 := by
  sorry

end jacob_coin_problem_l753_75365


namespace peaches_eaten_l753_75311

/-- Represents the state of peaches in a bowl --/
structure PeachBowl where
  total : ℕ
  ripe : ℕ
  unripe : ℕ

/-- Calculates the state of peaches after a given number of days --/
def ripenPeaches (initial : PeachBowl) (days : ℕ) (ripeningRate : ℕ) : PeachBowl :=
  { total := initial.total,
    ripe := min initial.total (initial.ripe + days * ripeningRate),
    unripe := max 0 (initial.total - (initial.ripe + days * ripeningRate)) }

/-- Theorem stating the number of peaches eaten --/
theorem peaches_eaten 
  (initial : PeachBowl)
  (ripeningRate : ℕ)
  (days : ℕ)
  (finalState : PeachBowl)
  (h1 : initial.total = 18)
  (h2 : initial.ripe = 4)
  (h3 : ripeningRate = 2)
  (h4 : days = 5)
  (h5 : finalState.ripe = finalState.unripe + 7)
  (h6 : finalState.total + 3 = (ripenPeaches initial days ripeningRate).total) :
  3 = initial.total - finalState.total :=
by
  sorry


end peaches_eaten_l753_75311


namespace m_range_for_inequality_l753_75314

theorem m_range_for_inequality (m : ℝ) : 
  (∀ x : ℝ, x ≤ -1 → (m^2 - m) * 4^x - 2^x < 0) ↔ -1 < m ∧ m < 2 := by
  sorry

end m_range_for_inequality_l753_75314


namespace parabola_line_intersection_triangle_area_l753_75305

/-- Given a parabola y = x^2 + 2 and a line y = r, if the triangle formed by the vertex of the parabola
    and the two intersections of the line and parabola has an area A such that 10 ≤ A ≤ 50,
    then 10^(2/3) + 2 ≤ r ≤ 50^(2/3) + 2. -/
theorem parabola_line_intersection_triangle_area (r : ℝ) : 
  let parabola := fun x : ℝ => x^2 + 2
  let line := fun _ : ℝ => r
  let vertex := (0, parabola 0)
  let intersections := {x : ℝ | parabola x = line x}
  let triangle_area := (r - 2)^(3/2) / 2
  10 ≤ triangle_area ∧ triangle_area ≤ 50 → 10^(2/3) + 2 ≤ r ∧ r ≤ 50^(2/3) + 2 := by
sorry

end parabola_line_intersection_triangle_area_l753_75305


namespace power_of_two_equality_l753_75322

theorem power_of_two_equality (x : ℤ) : (1 / 8 : ℚ) * (2 ^ 50) = 2 ^ x → x = 47 := by
  sorry

end power_of_two_equality_l753_75322


namespace circle_radius_zero_l753_75370

/-- The radius of a circle defined by the equation 4x^2 - 8x + 4y^2 + 16y + 20 = 0 is 0 -/
theorem circle_radius_zero (x y : ℝ) : 
  (4 * x^2 - 8 * x + 4 * y^2 + 16 * y + 20 = 0) → 
  ∃ (h k : ℝ), ∀ (x y : ℝ), (x - h)^2 + (y - k)^2 = 0 := by
  sorry

end circle_radius_zero_l753_75370


namespace power_product_equals_128_l753_75312

theorem power_product_equals_128 (b : ℕ) (h : b = 2) : b^3 * b^4 = 128 := by
  sorry

end power_product_equals_128_l753_75312


namespace power_product_square_l753_75304

theorem power_product_square (x y : ℝ) : (x * y^2)^2 = x^2 * y^4 := by
  sorry

end power_product_square_l753_75304


namespace flower_problem_solution_l753_75377

/-- Given initial flowers and minimum flowers per bouquet, 
    calculate additional flowers needed and number of bouquets -/
def flower_arrangement (initial_flowers : ℕ) (min_per_bouquet : ℕ) : 
  {additional_flowers : ℕ // ∃ (num_bouquets : ℕ), 
    num_bouquets * min_per_bouquet = initial_flowers + additional_flowers ∧
    num_bouquets * min_per_bouquet > initial_flowers ∧
    ∀ (k : ℕ), k * min_per_bouquet > initial_flowers → 
      k * min_per_bouquet ≥ num_bouquets * min_per_bouquet} :=
sorry

theorem flower_problem_solution : 
  (flower_arrangement 1273 89).val = 62 ∧ 
  ∃ (num_bouquets : ℕ), num_bouquets = 15 ∧
    num_bouquets * 89 = 1273 + (flower_arrangement 1273 89).val :=
sorry

end flower_problem_solution_l753_75377


namespace meghans_money_is_550_l753_75361

/-- Represents the number of bills of a specific denomination --/
structure BillCount where
  count : Nat
  denomination : Nat

/-- Calculates the total value of bills given their count and denomination --/
def billValue (b : BillCount) : Nat := b.count * b.denomination

/-- Represents Meghan's money --/
structure MeghansMoney where
  hundreds : BillCount
  fifties : BillCount
  tens : BillCount

/-- Calculates the total value of Meghan's money --/
def totalValue (m : MeghansMoney) : Nat :=
  billValue m.hundreds + billValue m.fifties + billValue m.tens

/-- Theorem stating that Meghan's total money is $550 --/
theorem meghans_money_is_550 (m : MeghansMoney) 
  (h1 : m.hundreds = { count := 2, denomination := 100 })
  (h2 : m.fifties = { count := 5, denomination := 50 })
  (h3 : m.tens = { count := 10, denomination := 10 }) :
  totalValue m = 550 := by sorry

end meghans_money_is_550_l753_75361


namespace det_specific_matrix_l753_75386

theorem det_specific_matrix :
  let A : Matrix (Fin 3) (Fin 3) ℝ := !![2, 4, -1; 0, 3, 2; 5, -1, 3]
  Matrix.det A = 77 := by
sorry

end det_specific_matrix_l753_75386


namespace card_row_theorem_l753_75302

/-- Represents a row of nine cards --/
def CardRow := Fin 9 → ℕ

/-- Checks if three consecutive cards are in increasing order --/
def increasing_three (row : CardRow) (i : Fin 7) : Prop :=
  row i < row (i + 1) ∧ row (i + 1) < row (i + 2)

/-- Checks if three consecutive cards are in decreasing order --/
def decreasing_three (row : CardRow) (i : Fin 7) : Prop :=
  row i > row (i + 1) ∧ row (i + 1) > row (i + 2)

/-- The main theorem --/
theorem card_row_theorem (row : CardRow) : 
  (∀ i : Fin 9, row i ∈ Finset.range 10) →  -- Cards are numbered 1 to 9
  (∀ i j : Fin 9, i ≠ j → row i ≠ row j) →  -- All numbers are different
  (∀ i : Fin 7, ¬increasing_three row i) →  -- No three consecutive increasing
  (∀ i : Fin 7, ¬decreasing_three row i) →  -- No three consecutive decreasing
  row 0 = 1 →                               -- Given visible cards
  row 1 = 6 →
  row 2 = 3 →
  row 3 = 4 →
  row 6 = 8 →
  row 7 = 7 →
  row 4 = 5 ∧ row 5 = 2 ∧ row 8 = 9         -- Conclusion: A = 5, B = 2, C = 9
:= by sorry


end card_row_theorem_l753_75302


namespace line_intercepts_sum_l753_75364

/-- A line is described by the equation y + 3 = -3(x + 5).
    This theorem proves that the sum of its x-intercept and y-intercept is -24. -/
theorem line_intercepts_sum (x y : ℝ) : 
  (y + 3 = -3 * (x + 5)) → 
  (∃ x_int y_int : ℝ, (y_int + 3 = -3 * (x_int + 5)) ∧ 
                      (0 + 3 = -3 * (x_int + 5)) ∧ 
                      (y_int + 3 = -3 * (0 + 5)) ∧ 
                      (x_int + y_int = -24)) := by
  sorry

end line_intercepts_sum_l753_75364


namespace complex_sum_zero_l753_75347

noncomputable def w : ℂ := Complex.exp (Complex.I * (3 * Real.pi / 8))

theorem complex_sum_zero :
  w / (1 + w^3) + w^2 / (1 + w^5) + w^3 / (1 + w^7) = 0 := by sorry

end complex_sum_zero_l753_75347


namespace game_result_l753_75317

def g (n : Nat) : Nat :=
  if n % 6 = 0 then 8
  else if n % 3 = 0 then 4
  else if n % 2 = 0 then 3
  else 1

def cora_rolls : List Nat := [5, 4, 3, 6, 2, 1]
def dana_rolls : List Nat := [6, 3, 4, 3, 5, 3]

def total_points (rolls : List Nat) : Nat :=
  (rolls.map g).sum

theorem game_result : (total_points cora_rolls) * (total_points dana_rolls) = 480 := by
  sorry

end game_result_l753_75317


namespace bicycle_distance_l753_75351

theorem bicycle_distance (motorcycle_speed : ℝ) (bicycle_speed_ratio : ℝ) (time_minutes : ℝ) :
  motorcycle_speed = 90 →
  bicycle_speed_ratio = 2 / 3 →
  time_minutes = 15 →
  (bicycle_speed_ratio * motorcycle_speed) * (time_minutes / 60) = 15 := by
sorry

end bicycle_distance_l753_75351


namespace max_knights_between_knights_theorem_l753_75382

/-- Represents a seating arrangement of knights and samurais around a round table. -/
structure SeatingArrangement where
  total_knights : Nat
  total_samurais : Nat
  knights_with_samurai_right : Nat

/-- Calculates the maximum number of knights that could be seated next to two other knights. -/
def max_knights_between_knights (arrangement : SeatingArrangement) : Nat :=
  arrangement.total_knights - (arrangement.knights_with_samurai_right + 1)

/-- Theorem stating the maximum number of knights seated next to two other knights
    for the given arrangement. -/
theorem max_knights_between_knights_theorem (arrangement : SeatingArrangement) 
  (h1 : arrangement.total_knights = 40)
  (h2 : arrangement.total_samurais = 10)
  (h3 : arrangement.knights_with_samurai_right = 7) :
  max_knights_between_knights arrangement = 32 := by
  sorry

#eval max_knights_between_knights { total_knights := 40, total_samurais := 10, knights_with_samurai_right := 7 }

end max_knights_between_knights_theorem_l753_75382


namespace gum_distribution_l753_75321

/-- Given the number of gum pieces for each person and the total number of people,
    calculate the number of gum pieces each person will receive after equal distribution. -/
def distribute_gum (john_gum : ℕ) (cole_gum : ℕ) (aubrey_gum : ℕ) (num_people : ℕ) : ℕ :=
  (john_gum + cole_gum + aubrey_gum) / num_people

/-- Theorem stating that when 54 pieces of gum, 45 pieces of gum, and 0 pieces of gum
    are combined and divided equally among 3 people, each person will receive 33 pieces of gum. -/
theorem gum_distribution :
  distribute_gum 54 45 0 3 = 33 := by
  sorry

#eval distribute_gum 54 45 0 3

end gum_distribution_l753_75321


namespace tangent_line_equation_l753_75343

/-- The equation of the tangent line to the circle (x-1)^2 + y^2 = 5 at the point (2, 2) is x + 2y - 6 = 0 -/
theorem tangent_line_equation (x y : ℝ) : 
  (∀ x y, (x - 1)^2 + y^2 = 5) →  -- Circle equation
  (2 - 1)^2 + 2^2 = 5 →           -- Point (2, 2) lies on the circle
  x + 2*y - 6 = 0                 -- Equation of the tangent line
    ↔ 
  ((x - 1)^2 + y^2 = 5 → (x - 2) + 2*(y - 2) = 0) -- Tangent line property
  :=
by sorry

end tangent_line_equation_l753_75343


namespace square_area_with_four_circles_l753_75392

theorem square_area_with_four_circles (r : ℝ) (h : r = 7) : 
  let side_length := 2 * (2 * r)
  (side_length ^ 2 : ℝ) = 784 := by
  sorry

end square_area_with_four_circles_l753_75392


namespace chocolate_sales_l753_75345

theorem chocolate_sales (cost_price selling_price : ℝ) (n : ℕ) : 
  44 * cost_price = n * selling_price →
  selling_price = cost_price * (1 + 5/6) →
  n = 24 := by
sorry

end chocolate_sales_l753_75345


namespace min_value_theorem_min_value_achievable_l753_75355

theorem min_value_theorem (x : ℝ) : (x^2 + 9) / Real.sqrt (x^2 + 5) ≥ 5 := by
  sorry

theorem min_value_achievable : ∃ x : ℝ, (x^2 + 9) / Real.sqrt (x^2 + 5) = 5 := by
  sorry

end min_value_theorem_min_value_achievable_l753_75355


namespace root_existence_iff_a_ge_three_l753_75390

/-- The function f(x) = ln x + x + 2/x - a has a root for some x > 0 if and only if a ≥ 3 -/
theorem root_existence_iff_a_ge_three (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ Real.log x + x + 2 / x - a = 0) ↔ a ≥ 3 := by
  sorry

end root_existence_iff_a_ge_three_l753_75390


namespace hyperbola_eccentricity_l753_75308

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  ha : a > 0
  hb : b > 0

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ := sorry

/-- The sum of two line segments perpendicular to the asymptotes
    and passing through one of the foci -/
def sum_perp_segments (h : Hyperbola) : ℝ := sorry

theorem hyperbola_eccentricity (h : Hyperbola) 
  (h_sum : sum_perp_segments h = h.a) : 
  eccentricity h = Real.sqrt 5 / 2 := by sorry

end hyperbola_eccentricity_l753_75308


namespace triangle_perimeter_l753_75378

theorem triangle_perimeter : ∀ (a b c : ℝ),
  a = 4 ∧ b = 6 ∧ c^2 - 6*c + 8 = 0 ∧
  a + b > c ∧ a + c > b ∧ b + c > a →
  a + b + c = 14 := by
  sorry

end triangle_perimeter_l753_75378


namespace books_bought_at_yard_sale_l753_75318

theorem books_bought_at_yard_sale 
  (initial_books : ℕ) 
  (final_books : ℕ) 
  (h1 : initial_books = 35)
  (h2 : final_books = 56) :
  final_books - initial_books = 21 :=
by sorry

end books_bought_at_yard_sale_l753_75318


namespace inequality_proof_l753_75380

theorem inequality_proof (m n : ℕ) (h : m < n) :
  m^2 + Real.sqrt (m^2 + m) < n^2 - Real.sqrt (n^2 - n) := by
  sorry

end inequality_proof_l753_75380


namespace multiple_birth_statistics_l753_75330

theorem multiple_birth_statistics (total_babies : ℕ) 
  (twins triplets quadruplets quintuplets : ℕ) : 
  total_babies = 1200 →
  quintuplets = 2 * quadruplets →
  quadruplets = 3 * triplets →
  triplets = 2 * twins →
  2 * twins + 3 * triplets + 4 * quadruplets + 5 * quintuplets = total_babies →
  5 * quintuplets = 18000 / 23 := by
sorry

end multiple_birth_statistics_l753_75330


namespace function_max_min_difference_l753_75352

theorem function_max_min_difference (a : ℝ) (h1 : a > 1) :
  let f : ℝ → ℝ := fun x ↦ a^(x-1)
  (∀ x ∈ Set.Icc 2 3, f x ≤ f 3) ∧
  (∀ x ∈ Set.Icc 2 3, f 2 ≤ f x) ∧
  (f 3 - f 2 = a / 2) →
  a = 3/2 := by sorry

end function_max_min_difference_l753_75352


namespace inscribed_trapezoid_intersection_l753_75316

/-- A trapezoid inscribed in the parabola y = x^2 -/
structure InscribedTrapezoid where
  /-- Left x-coordinate of the upper base -/
  a : ℝ
  /-- Left x-coordinate of the lower base -/
  b : ℝ
  /-- The product of the lengths of the bases is k -/
  base_product : (2 * a) * (2 * b) = k
  /-- k is positive -/
  k_pos : k > 0

/-- The theorem stating that all inscribed trapezoids with base product k 
    have lateral sides intersecting at (0, -k/4) -/
theorem inscribed_trapezoid_intersection 
  (k : ℝ) (trap : InscribedTrapezoid) : 
  ∃ (x y : ℝ), x = 0 ∧ y = -k/4 ∧ 
  (∀ (t : ℝ), 
    ((t - trap.a) * (trap.b^2 - trap.a^2) = (trap.b - trap.a) * (t^2 - trap.a^2) ↔ 
     (t = x ∧ t^2 = y)) ∧
    ((t + trap.a) * (trap.b^2 - trap.a^2) = (trap.b + trap.a) * (t^2 - trap.a^2) ↔ 
     (t = x ∧ t^2 = y))) :=
by sorry


end inscribed_trapezoid_intersection_l753_75316


namespace square_difference_l753_75313

theorem square_difference (a b : ℝ) (h1 : a + b = 8) (h2 : a - b = 4) : a^2 - b^2 = 32 := by
  sorry

end square_difference_l753_75313


namespace quadratic_complete_square_l753_75391

theorem quadratic_complete_square (c : ℝ) (h1 : c > 0) :
  (∃ n : ℝ, ∀ x : ℝ, x^2 + c*x + 20 = (x + n)^2 + 12) →
  c = 4 * Real.sqrt 2 := by
sorry

end quadratic_complete_square_l753_75391


namespace simple_interest_problem_l753_75395

theorem simple_interest_problem (r : ℝ) (n : ℝ) :
  (400 * r * n) / 100 + 200 = (400 * (r + 5) * n) / 100 →
  n = 10 := by
  sorry

end simple_interest_problem_l753_75395


namespace weekend_pie_revenue_l753_75339

structure PieSlice where
  name : String
  slices_per_pie : ℕ
  price_per_slice : ℕ
  customers : ℕ

def apple_pie : PieSlice := {
  name := "Apple",
  slices_per_pie := 8,
  price_per_slice := 3,
  customers := 88
}

def peach_pie : PieSlice := {
  name := "Peach",
  slices_per_pie := 6,
  price_per_slice := 4,
  customers := 78
}

def cherry_pie : PieSlice := {
  name := "Cherry",
  slices_per_pie := 10,
  price_per_slice := 5,
  customers := 45
}

def revenue (pie : PieSlice) : ℕ :=
  pie.customers * pie.price_per_slice

def total_revenue (pies : List PieSlice) : ℕ :=
  pies.foldl (fun acc pie => acc + revenue pie) 0

theorem weekend_pie_revenue :
  total_revenue [apple_pie, peach_pie, cherry_pie] = 801 := by
  sorry

end weekend_pie_revenue_l753_75339


namespace f_properties_l753_75388

noncomputable def f (x : ℝ) : ℝ := (x - 1) / x^2

theorem f_properties :
  (∃ x : ℝ, x ≠ 0 ∧ f x = 0 ↔ x = 1) ∧
  (∃ x : ℝ, x ≠ 0 ∧ ∀ y : ℝ, y ≠ 0 → f y ≤ f x) ∧
  (∀ x : ℝ, x ≠ 0 → f x ≤ f 2) :=
sorry

end f_properties_l753_75388


namespace horner_v3_value_l753_75300

def horner_polynomial (x : ℝ) : ℝ := x^6 - 12*x^5 + 60*x^4 - 160*x^3 + 240*x^2 - 192*x + 64

def horner_step (v : ℝ) (x : ℝ) (a : ℝ) : ℝ := v * x + a

theorem horner_v3_value :
  let x := 2
  let v0 := 1
  let v1 := horner_step v0 x (-12)
  let v2 := horner_step v1 x 60
  let v3 := horner_step v2 x (-160)
  v3 = -80 := by sorry

end horner_v3_value_l753_75300


namespace negation_of_forall_geq_one_l753_75319

theorem negation_of_forall_geq_one :
  (¬ ∀ x : ℝ, x^2 + 1 ≥ 1) ↔ (∃ x : ℝ, x^2 + 1 < 1) := by sorry

end negation_of_forall_geq_one_l753_75319


namespace divisible_by_five_count_is_correct_l753_75335

/-- The number of different positive, seven-digit integers divisible by 5,
    formed using the digits 2 (three times), 5 (two times), and 9 (two times) -/
def divisible_by_five_count : ℕ :=
  let total_digits : ℕ := 7
  let two_count : ℕ := 3
  let five_count : ℕ := 2
  let nine_count : ℕ := 2
  60

theorem divisible_by_five_count_is_correct :
  divisible_by_five_count = 60 := by sorry

end divisible_by_five_count_is_correct_l753_75335


namespace partnership_profit_share_l753_75374

/-- Given a partnership where A invests 3 times as much as B and 2/3 of what C invests,
    and the total profit is 55000, prove that C's share of the profit is (9/17) * 55000. -/
theorem partnership_profit_share (a b c : ℝ) (total_profit : ℝ) : 
  a = 3 * b ∧ a = (2/3) * c ∧ total_profit = 55000 → 
  c * total_profit / (a + b + c) = (9/17) * 55000 := by
sorry

end partnership_profit_share_l753_75374


namespace unique_triple_solution_l753_75338

theorem unique_triple_solution (p q : Nat) (n : Nat) (h_p : Nat.Prime p) (h_q : Nat.Prime q)
    (h_n : n > 1) (h_p_odd : Odd p) (h_q_odd : Odd q)
    (h_cong1 : q^(n+2) ≡ 3^(n+2) [MOD p^n])
    (h_cong2 : p^(n+2) ≡ 3^(n+2) [MOD q^n]) :
    p = 3 ∧ q = 3 := by
  sorry

#check unique_triple_solution

end unique_triple_solution_l753_75338


namespace no_snuggly_numbers_l753_75306

/-- A two-digit positive integer is 'snuggly' if it equals the sum of its nonzero tens digit, 
    the cube of its units digit, and 5. -/
def is_snuggly (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧ ∃ a b : ℕ, 
    n = 10 * a + b ∧ 
    1 ≤ a ∧ a ≤ 9 ∧ 
    0 ≤ b ∧ b ≤ 9 ∧
    n = a + b^3 + 5

theorem no_snuggly_numbers : ¬∃ n : ℕ, is_snuggly n :=
sorry

end no_snuggly_numbers_l753_75306


namespace chromosome_stability_l753_75327

-- Define the number of chromosomes in somatic cells
def somaticChromosomes : ℕ := 46

-- Define the process of meiosis
def meiosis (n : ℕ) : ℕ := n / 2

-- Define the process of fertilization
def fertilization (n : ℕ) : ℕ := n * 2

-- Theorem: Meiosis and fertilization maintain chromosome stability across generations
theorem chromosome_stability :
  ∀ (generation : ℕ),
    fertilization (meiosis somaticChromosomes) = somaticChromosomes :=
by sorry

end chromosome_stability_l753_75327


namespace dress_sewing_time_l753_75334

/-- The time Allison and Al worked together on sewing dresses -/
def timeWorkedTogether (allisonRate alRate : ℚ) (allisonAloneTime : ℚ) : ℚ :=
  (1 - allisonRate * allisonAloneTime) / (allisonRate + alRate)

theorem dress_sewing_time : 
  let allisonRate : ℚ := 1/9
  let alRate : ℚ := 1/12
  let allisonAloneTime : ℚ := 15/4
  timeWorkedTogether allisonRate alRate allisonAloneTime = 3 := by
sorry

end dress_sewing_time_l753_75334


namespace coworker_lunch_pizzas_l753_75326

/-- Calculates the number of pizzas needed for a group lunch -/
def pizzas_ordered (coworkers : ℕ) (slices_per_pizza : ℕ) (slices_per_person : ℕ) : ℕ :=
  (coworkers * slices_per_person) / slices_per_pizza

/-- Proves that 12 coworkers each getting 2 slices from pizzas with 8 slices each requires 3 pizzas -/
theorem coworker_lunch_pizzas :
  pizzas_ordered 12 8 2 = 3 := by
  sorry

end coworker_lunch_pizzas_l753_75326


namespace range_of_m_l753_75303

-- Define proposition p
def p (m : ℝ) : Prop :=
  ∃ a b : ℝ, a > b ∧ a^2 = m/2 ∧ b^2 = m/2 - 1

-- Define proposition q
def q (m : ℝ) : Prop :=
  ∀ x : ℝ, 4*x^2 - 4*m*x + 4*m - 3 ≥ 0

-- Theorem statement
theorem range_of_m :
  ∃ m_min m_max : ℝ,
    m_min = 1 ∧ m_max = 2 ∧
    ∀ m : ℝ, (¬(p m) ∧ q m) ↔ m_min ≤ m ∧ m ≤ m_max :=
sorry

end range_of_m_l753_75303


namespace total_earnings_l753_75363

def wednesday_amount : ℚ := 1832
def sunday_amount : ℚ := 3162.5

theorem total_earnings : wednesday_amount + sunday_amount = 4994.5 := by
  sorry

end total_earnings_l753_75363


namespace simplify_expression_l753_75379

theorem simplify_expression (x : ℝ) (hx : x ≠ 0) :
  (25 * x^3) * (8 * x^2) * (1 / (4 * x)^3) = (25 / 8) * x^2 := by
  sorry

end simplify_expression_l753_75379


namespace nancy_carrots_l753_75325

/-- The number of carrots Nancy picked the next day -/
def carrots_picked_next_day (initial_carrots : ℕ) (thrown_out : ℕ) (total_carrots : ℕ) : ℕ :=
  total_carrots - (initial_carrots - thrown_out)

/-- Proof that Nancy picked 21 carrots the next day -/
theorem nancy_carrots :
  carrots_picked_next_day 12 2 31 = 21 := by
  sorry

end nancy_carrots_l753_75325


namespace smallest_integer_with_divisibility_pattern_l753_75315

def is_divisible (n m : ℕ) : Prop := m ≠ 0 ∧ n % m = 0

def consecutive_three (a b c : ℕ) : Prop := b = a + 1 ∧ c = b + 1

theorem smallest_integer_with_divisibility_pattern :
  ∃ (n : ℕ) (a : ℕ),
    n > 0 ∧
    a > 0 ∧
    a < 39 ∧
    consecutive_three a (a + 1) (a + 2) ∧
    (∀ (k : ℕ), k > 0 ∧ k ≤ 40 ∧ k ≠ a ∧ k ≠ (a + 1) ∧ k ≠ (a + 2) → is_divisible n k) ∧
    (¬ is_divisible n a ∧ ¬ is_divisible n (a + 1) ∧ ¬ is_divisible n (a + 2)) ∧
    n = 299576986419800 ∧
    (∀ (m : ℕ), m > 0 ∧ m < n →
      ¬(∃ (b : ℕ), b > 0 ∧ b < 39 ∧
        consecutive_three b (b + 1) (b + 2) ∧
        (∀ (k : ℕ), k > 0 ∧ k ≤ 40 ∧ k ≠ b ∧ k ≠ (b + 1) ∧ k ≠ (b + 2) → is_divisible m k) ∧
        (¬ is_divisible m b ∧ ¬ is_divisible m (b + 1) ∧ ¬ is_divisible m (b + 2))))
  := by sorry

end smallest_integer_with_divisibility_pattern_l753_75315


namespace q_transformation_l753_75371

theorem q_transformation (w d z z' : ℝ) (hw : w > 0) (hd : d > 0) (hz : z ≠ 0) :
  let q := 5 * w / (4 * d * z^2)
  let q' := 5 * (4 * w) / (4 * (2 * d) * z'^2)
  q' / q = 2/9 ↔ z' = 3 * Real.sqrt 2 * z := by
sorry

end q_transformation_l753_75371


namespace temperature_difference_l753_75353

def january_temp : ℝ := -3
def march_temp : ℝ := 2

theorem temperature_difference : march_temp - january_temp = 5 := by
  sorry

end temperature_difference_l753_75353


namespace laptop_price_l753_75384

theorem laptop_price (upfront_percentage : ℚ) (upfront_payment : ℚ) 
  (h1 : upfront_percentage = 20 / 100)
  (h2 : upfront_payment = 240) : 
  upfront_payment / upfront_percentage = 1200 := by
sorry

end laptop_price_l753_75384


namespace mean_study_hours_thompson_class_l753_75383

theorem mean_study_hours_thompson_class : 
  let study_hours := [0, 2, 4, 6, 8, 10, 12]
  let student_counts := [3, 6, 8, 5, 4, 2, 2]
  let total_students := 30
  let total_hours := (List.zip study_hours student_counts).map (fun (h, c) => h * c) |>.sum
  (total_hours : ℚ) / total_students = 5 := by
  sorry

end mean_study_hours_thompson_class_l753_75383


namespace quadratic_inequality_range_l753_75341

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, (x < 1 ∨ x > 5) → x^2 - 2*(a-2)*x + a > 0) ↔ 
  (1 < a ∧ a ≤ 5) := by sorry

end quadratic_inequality_range_l753_75341


namespace tiles_remaining_l753_75307

theorem tiles_remaining (initial_tiles : ℕ) : 
  initial_tiles = 2022 → 
  (initial_tiles - initial_tiles / 6 - (initial_tiles - initial_tiles / 6) / 5 - 
   (initial_tiles - initial_tiles / 6 - (initial_tiles - initial_tiles / 6) / 5) / 4) = 1011 := by
  sorry

end tiles_remaining_l753_75307


namespace typing_service_problem_l753_75344

/-- The typing service problem -/
theorem typing_service_problem 
  (total_pages : ℕ) 
  (twice_revised_pages : ℕ) 
  (initial_cost : ℕ) 
  (revision_cost : ℕ) 
  (total_cost : ℕ) 
  (h1 : total_pages = 100)
  (h2 : twice_revised_pages = 20)
  (h3 : initial_cost = 5)
  (h4 : revision_cost = 3)
  (h5 : total_cost = 710) :
  ∃ (once_revised_pages : ℕ),
    once_revised_pages = 30 ∧
    total_cost = 
      initial_cost * total_pages + 
      revision_cost * once_revised_pages + 
      2 * revision_cost * twice_revised_pages :=
by
  sorry


end typing_service_problem_l753_75344


namespace scientific_notation_of_given_number_l753_75396

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  property : 1 ≤ coefficient ∧ coefficient < 10

/-- Convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

/-- The given number -/
def givenNumber : ℝ := 0.0000046

/-- Theorem: The scientific notation of 0.0000046 is 4.6 × 10^(-6) -/
theorem scientific_notation_of_given_number :
  toScientificNotation givenNumber = ScientificNotation.mk 4.6 (-6) sorry := by
  sorry

end scientific_notation_of_given_number_l753_75396


namespace senate_committee_seating_l753_75348

/-- The number of ways to arrange n distinguishable objects in a circle -/
def circularPermutations (n : ℕ) : ℕ := (n - 1).factorial

/-- The number of politicians in the committee -/
def committeeSize : ℕ := 4 + 4 + 3

theorem senate_committee_seating :
  circularPermutations committeeSize = 3628800 := by
  sorry

end senate_committee_seating_l753_75348


namespace max_product_xyz_l753_75359

theorem max_product_xyz (x y z : ℝ) 
  (hx : x ≥ 20) (hy : y ≥ 40) (hz : z ≥ 1675) (hsum : x + y + z = 2015) : 
  x * y * z ≤ 721480000 / 27 := by
  sorry

end max_product_xyz_l753_75359


namespace units_digit_of_3_to_1987_l753_75337

theorem units_digit_of_3_to_1987 : 3^1987 % 10 = 7 := by
  sorry

end units_digit_of_3_to_1987_l753_75337


namespace max_packing_ge_min_covering_l753_75398

/-- Represents a polygon in 2D space -/
structure Polygon

/-- The largest number of non-overlapping circles with diameter 1 whose centers lie inside the polygon -/
def max_packing (M : Polygon) : ℕ :=
  sorry

/-- The smallest number of circles with radius 1 needed to cover the entire polygon -/
def min_covering (M : Polygon) : ℕ :=
  sorry

/-- Theorem stating that the maximum packing is greater than or equal to the minimum covering -/
theorem max_packing_ge_min_covering (M : Polygon) : max_packing M ≥ min_covering M :=
  sorry

end max_packing_ge_min_covering_l753_75398


namespace exist_unit_tetrahedron_with_interior_point_l753_75301

/-- A type representing a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculate the volume of a tetrahedron given four points -/
def tetrahedronVolume (p1 p2 p3 p4 : Point3D) : ℝ := sorry

/-- Check if four points are coplanar -/
def areCoplanar (p1 p2 p3 p4 : Point3D) : Prop := sorry

/-- Check if a point is inside a tetrahedron -/
def isInsideTetrahedron (p : Point3D) (t1 t2 t3 t4 : Point3D) : Prop := sorry

/-- Main theorem -/
theorem exist_unit_tetrahedron_with_interior_point 
  (n : ℕ) 
  (points : Fin n → Point3D) 
  (h_not_coplanar : ∀ (i j k l : Fin n), i ≠ j → j ≠ k → k ≠ l → i ≠ k → i ≠ l → j ≠ l → ¬areCoplanar (points i) (points j) (points k) (points l))
  (h_max_volume : ∀ (i j k l : Fin n), i ≠ j → j ≠ k → k ≠ l → i ≠ k → i ≠ l → j ≠ l → tetrahedronVolume (points i) (points j) (points k) (points l) ≤ 0.037)
  : ∃ (t1 t2 t3 t4 : Point3D), 
    tetrahedronVolume t1 t2 t3 t4 = 1 ∧ 
    ∃ (i : Fin n), isInsideTetrahedron (points i) t1 t2 t3 t4 := by
  sorry

end exist_unit_tetrahedron_with_interior_point_l753_75301


namespace rose_crystal_beads_l753_75331

/-- The number of beads in each bracelet -/
def beads_per_bracelet : ℕ := 8

/-- The number of metal beads Nancy has -/
def nancy_metal_beads : ℕ := 40

/-- The number of pearl beads Nancy has more than metal beads -/
def nancy_extra_pearl_beads : ℕ := 20

/-- The number of bracelets they can make -/
def total_bracelets : ℕ := 20

/-- The relation between Rose's crystal and stone beads -/
def rose_stone_to_crystal_ratio : ℕ := 2

/-- Theorem: Rose has 20 crystal beads -/
theorem rose_crystal_beads :
  ∃ (crystal_beads : ℕ),
    crystal_beads = 20 ∧
    crystal_beads * (rose_stone_to_crystal_ratio + 1) =
      total_bracelets * beads_per_bracelet -
      (nancy_metal_beads + nancy_metal_beads + nancy_extra_pearl_beads) :=
by sorry

end rose_crystal_beads_l753_75331


namespace cupcake_packages_l753_75375

theorem cupcake_packages (initial_cupcakes : ℕ) (eaten_cupcakes : ℕ) (cupcakes_per_package : ℕ) :
  initial_cupcakes = 18 →
  eaten_cupcakes = 8 →
  cupcakes_per_package = 2 →
  (initial_cupcakes - eaten_cupcakes) / cupcakes_per_package = 5 :=
by
  sorry

end cupcake_packages_l753_75375


namespace tap_b_fills_12_liters_l753_75342

/-- Represents the water flow problem with two taps filling a bucket. -/
structure WaterFlow where
  bucket_volume : ℝ
  tap_a_rate : ℝ
  fill_time_both : ℝ

/-- The amount of water tap B fills in 20 minutes. -/
def tap_b_fill_20min (w : WaterFlow) : ℝ :=
  2 * (w.bucket_volume - w.tap_a_rate * w.fill_time_both)

/-- Theorem stating that tap B fills 12 liters in 20 minutes under given conditions. -/
theorem tap_b_fills_12_liters (w : WaterFlow)
  (h1 : w.bucket_volume = 36)
  (h2 : w.tap_a_rate = 3)
  (h3 : w.fill_time_both = 10) :
  tap_b_fill_20min w = 12 := by
  sorry

end tap_b_fills_12_liters_l753_75342


namespace sum_of_nineteen_terms_l753_75362

/-- Given an arithmetic sequence {a_n}, S_n represents the sum of its first n terms -/
def S (n : ℕ) : ℝ := sorry

/-- {a_n} is an arithmetic sequence -/
def isArithmeticSequence (a : ℕ → ℝ) : Prop := sorry

/-- The second, ninth, and nineteenth terms of the sequence sum to 6 -/
axiom sum_condition (a : ℕ → ℝ) : a 2 + a 9 + a 19 = 6

theorem sum_of_nineteen_terms (a : ℕ → ℝ) (h : isArithmeticSequence a) : 
  S 19 = 38 := by sorry

end sum_of_nineteen_terms_l753_75362


namespace weekly_wage_problem_l753_75369

/-- The weekly wage problem -/
theorem weekly_wage_problem (Rm Hm Rn Hn : ℝ) 
  (h1 : Rm * Hm + Rn * Hn = 770)
  (h2 : Rm * Hm = 1.3 * (Rn * Hn)) :
  Rn * Hn = 335 := by
  sorry

end weekly_wage_problem_l753_75369


namespace no_rain_probability_l753_75310

theorem no_rain_probability (p : ℚ) (h : p = 2/3) :
  (1 - p)^4 = 1/81 := by
  sorry

end no_rain_probability_l753_75310


namespace no_two_digit_product_concatenation_l753_75320

theorem no_two_digit_product_concatenation : ¬∃ (a b c d : ℕ), 
  0 ≤ a ∧ a ≤ 9 ∧
  0 ≤ b ∧ b ≤ 9 ∧
  0 ≤ c ∧ c ≤ 9 ∧
  0 ≤ d ∧ d ≤ 9 ∧
  (10 * a + b) * (10 * c + d) = 1000 * a + 100 * b + 10 * c + d :=
by sorry

end no_two_digit_product_concatenation_l753_75320


namespace prop_a_prop_b_prop_d_l753_75372

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Proposition A
theorem prop_a (t : Triangle) (h : t.A > t.B) : Real.sin t.A > Real.sin t.B := by sorry

-- Proposition B
theorem prop_b (t : Triangle) (h : t.A < π/2 ∧ t.B < π/2 ∧ t.C < π/2) : Real.sin t.A > Real.cos t.B := by sorry

-- Proposition D
theorem prop_d (t : Triangle) (h1 : t.B = π/3) (h2 : t.b^2 = t.a * t.c) : t.A = π/3 ∧ t.B = π/3 ∧ t.C = π/3 := by sorry

end prop_a_prop_b_prop_d_l753_75372


namespace scientific_notation_proof_l753_75394

def number_to_express : ℝ := 460000000

theorem scientific_notation_proof :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ number_to_express = a * (10 : ℝ) ^ n ∧ a = 4.6 ∧ n = 8 := by
  sorry

end scientific_notation_proof_l753_75394


namespace trig_expression_equals_one_l753_75397

theorem trig_expression_equals_one :
  let x : Real := 40 * π / 180
  let y : Real := 50 * π / 180
  (Real.sqrt (1 - 2 * Real.sin x * Real.cos x)) / (Real.cos x - Real.sqrt (1 - Real.sin y ^ 2)) = 1 := by
  sorry

end trig_expression_equals_one_l753_75397


namespace bridge_building_time_l753_75357

/-- Represents the time taken to build a bridge given a number of workers -/
def build_time (workers : ℕ) : ℝ := sorry

/-- The constant representing the total work required -/
def total_work : ℝ := 18 * 6

theorem bridge_building_time :
  (build_time 18 = 6) →
  (∀ w₁ w₂ : ℕ, w₁ * build_time w₁ = w₂ * build_time w₂) →
  build_time 30 = 3.6 := by sorry

end bridge_building_time_l753_75357


namespace parabola_vertex_l753_75368

/-- The vertex of the parabola y = (x+2)^2 - 1 is at the point (-2, -1) -/
theorem parabola_vertex (x y : ℝ) : 
  y = (x + 2)^2 - 1 → (∀ x' y', y' = (x' + 2)^2 - 1 → y ≤ y') → x = -2 ∧ y = -1 := by
  sorry

end parabola_vertex_l753_75368


namespace lcm_gcf_ratio_252_675_l753_75329

theorem lcm_gcf_ratio_252_675 : 
  Nat.lcm 252 675 / Nat.gcd 252 675 = 2100 := by sorry

end lcm_gcf_ratio_252_675_l753_75329


namespace log_50_between_consecutive_integers_l753_75358

theorem log_50_between_consecutive_integers : 
  ∃ (m n : ℤ), m + 1 = n ∧ (m : ℝ) < Real.log 50 / Real.log 10 ∧ Real.log 50 / Real.log 10 < n ∧ m + n = 3 := by
sorry

end log_50_between_consecutive_integers_l753_75358


namespace f_max_value_f_solution_set_max_ab_plus_bc_l753_75366

-- Define the function f
def f (x : ℝ) : ℝ := |x - 3| - 2 * |x + 1|

-- Theorem for the maximum value of f
theorem f_max_value : ∃ m : ℝ, m = 4 ∧ ∀ x : ℝ, f x ≤ m :=
sorry

-- Theorem for the solution set of f(x) < 1
theorem f_solution_set : ∀ x : ℝ, f x < 1 ↔ x < -4 ∨ x > 0 :=
sorry

-- Theorem for the maximum value of ab + bc
theorem max_ab_plus_bc :
  ∀ a b c : ℝ, a > 0 → b > 0 → a^2 + 2*b^2 + c^2 = 4 →
  ∃ max : ℝ, max = 2 ∧ a*b + b*c ≤ max :=
sorry

end f_max_value_f_solution_set_max_ab_plus_bc_l753_75366


namespace rectangular_parallelepiped_surface_area_l753_75340

theorem rectangular_parallelepiped_surface_area
  (m n : ℤ)
  (h_m_lt_n : m < n)
  (x y z : ℤ)
  (h_x : x = n * (n - m))
  (h_y : y = m * n)
  (h_z : z = m * (n - m)) :
  2 * (x + y) * z = 2 * x * y := by
  sorry

end rectangular_parallelepiped_surface_area_l753_75340


namespace children_attending_show_l753_75393

/-- Proves that the number of children attending the show is 3 --/
theorem children_attending_show :
  let adult_ticket_price : ℕ := 12
  let child_ticket_price : ℕ := 10
  let num_adults : ℕ := 3
  let total_cost : ℕ := 66
  let num_children : ℕ := (total_cost - num_adults * adult_ticket_price) / child_ticket_price
  num_children = 3 := by
sorry


end children_attending_show_l753_75393


namespace division_problem_l753_75354

theorem division_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) 
  (h1 : dividend = 12)
  (h2 : divisor = 17)
  (h3 : remainder = 10)
  (h4 : dividend = divisor * quotient + remainder) :
  quotient = 0 := by
  sorry

end division_problem_l753_75354


namespace M_is_range_of_f_l753_75323

-- Define the set M
def M : Set ℝ := {y | ∃ x, y = x^2}

-- Define the function f(x) = x^2
def f : ℝ → ℝ := λ x ↦ x^2

-- Theorem statement
theorem M_is_range_of_f : M = Set.range f := by sorry

end M_is_range_of_f_l753_75323


namespace no_valid_A_l753_75389

theorem no_valid_A : ¬∃ (A : ℕ), A < 10 ∧ 75 % A = 0 ∧ (5361000 + 100 * A + 4) % 4 = 0 := by
  sorry

end no_valid_A_l753_75389


namespace train_crossing_time_l753_75333

/-- Calculates the time for a train to cross a signal pole given its length, 
    the platform length, and the time to cross the platform. -/
theorem train_crossing_time 
  (train_length : ℝ) 
  (platform_length : ℝ) 
  (platform_crossing_time : ℝ) 
  (h1 : train_length = 300)
  (h2 : platform_length = 550.0000000000001)
  (h3 : platform_crossing_time = 51) :
  (train_length / ((train_length + platform_length) / platform_crossing_time)) = 18 := by
sorry

end train_crossing_time_l753_75333


namespace gcd_45_75_l753_75376

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_45_75_l753_75376


namespace value_added_to_number_l753_75385

theorem value_added_to_number (sum number value : ℕ) : 
  sum = number + value → number = 81 → sum = 96 → value = 15 := by
  sorry

end value_added_to_number_l753_75385


namespace intersection_A_B_range_of_m_when_A_subset_C_l753_75381

-- Define sets A, B, and C
def A : Set ℝ := {x | x^2 - 5*x - 6 < 0}
def B : Set ℝ := {x | 6*x^2 - 5*x + 1 ≥ 0}
def C (m : ℝ) : Set ℝ := {x | (x - m)*(x - m - 9) < 0}

-- Theorem for the intersection of A and B
theorem intersection_A_B :
  A ∩ B = {x | -1 < x ∧ x ≤ 1/3 ∨ 1/2 ≤ x ∧ x < 6} :=
sorry

-- Theorem for the range of m when A is a subset of C
theorem range_of_m_when_A_subset_C :
  (∀ m : ℝ, A ⊆ C m → -3 ≤ m ∧ m ≤ -1) ∧
  (∀ m : ℝ, -3 ≤ m ∧ m ≤ -1 → A ⊆ C m) :=
sorry

end intersection_A_B_range_of_m_when_A_subset_C_l753_75381


namespace intersection_of_A_and_B_l753_75346

def A : Set ℝ := {x | x^2 - 2*x ≤ 0}
def B : Set ℝ := {-1, 0, 1, 2, 3}

theorem intersection_of_A_and_B : A ∩ B = {0, 1, 2} := by sorry

end intersection_of_A_and_B_l753_75346


namespace gcf_of_84_112_210_l753_75332

theorem gcf_of_84_112_210 : Nat.gcd 84 (Nat.gcd 112 210) = 14 := by
  sorry

end gcf_of_84_112_210_l753_75332

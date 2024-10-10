import Mathlib

namespace abs_even_and_increasing_l3373_337300

-- Define the absolute value function
def f (x : ℝ) : ℝ := |x|

-- State the theorem
theorem abs_even_and_increasing :
  (∀ x : ℝ, f x = f (-x)) ∧
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) :=
by sorry

end abs_even_and_increasing_l3373_337300


namespace triangle_arithmetic_geometric_sequence_l3373_337321

theorem triangle_arithmetic_geometric_sequence (A B C : ℝ) (a b c : ℝ) : 
  -- Angles form an arithmetic sequence
  2 * B = A + C →
  -- Sum of angles in a triangle
  A + B + C = π →
  -- Sides form a geometric sequence
  b^2 = a * c →
  -- Law of cosines
  b^2 = a^2 + c^2 - 2 * a * c * Real.cos B →
  -- Conclusions
  Real.cos B = 1 / 2 ∧ Real.sin A * Real.sin C = 3 / 4 := by
sorry

end triangle_arithmetic_geometric_sequence_l3373_337321


namespace r_profit_share_is_one_third_of_total_l3373_337351

/-- Represents the capital and investment duration of an investor -/
structure Investor where
  capital : ℝ
  duration : ℝ

/-- Calculates the profit share of an investor -/
def profitShare (i : Investor) : ℝ := i.capital * i.duration

/-- Theorem: Given the conditions, r's share of the total profit is one-third of the total profit -/
theorem r_profit_share_is_one_third_of_total
  (p q r : Investor)
  (h1 : 4 * p.capital = 6 * q.capital)
  (h2 : 6 * q.capital = 10 * r.capital)
  (h3 : p.duration = 2)
  (h4 : q.duration = 3)
  (h5 : r.duration = 5)
  (total_profit : ℝ)
  : profitShare r = total_profit / 3 := by
  sorry

#check r_profit_share_is_one_third_of_total

end r_profit_share_is_one_third_of_total_l3373_337351


namespace price_A_base_correct_min_A_bundles_correct_l3373_337373

-- Define the price of type A seedlings at the base
def price_A_base : ℝ := 20

-- Define the price of type B seedlings at the base
def price_B_base : ℝ := 30

-- Define the total number of bundles to purchase
def total_bundles : ℕ := 100

-- Define the maximum spending limit
def max_spending : ℝ := 2400

-- Theorem for the price of type A seedlings at the base
theorem price_A_base_correct :
  ∃ (x : ℝ), x > 0 ∧ 300 / x - 300 / (1.5 * x) = 5 ∧ x = price_A_base :=
sorry

-- Theorem for the minimum number of type A seedlings to purchase
theorem min_A_bundles_correct :
  ∃ (m : ℕ), m ≥ 60 ∧
    ∀ (n : ℕ), n < m →
      price_A_base * n + price_B_base * (total_bundles - n) > max_spending :=
sorry

end price_A_base_correct_min_A_bundles_correct_l3373_337373


namespace quadratic_roots_product_l3373_337319

theorem quadratic_roots_product (a b : ℝ) : 
  (3 * a^2 + 9 * a - 18 = 0) → 
  (3 * b^2 + 9 * b - 18 = 0) → 
  (3*a - 2) * (6*b - 9) = 27 := by
sorry

end quadratic_roots_product_l3373_337319


namespace max_students_equal_distribution_l3373_337359

/-- The maximum number of students for equal distribution of pens and pencils -/
theorem max_students_equal_distribution (pens pencils : ℕ) : 
  pens = 891 → pencils = 810 → 
  (∃ (max_students : ℕ), 
    max_students = Nat.gcd pens pencils ∧ 
    max_students > 0 ∧
    pens % max_students = 0 ∧ 
    pencils % max_students = 0 ∧
    ∀ (n : ℕ), n > max_students → (pens % n ≠ 0 ∨ pencils % n ≠ 0)) := by
  sorry

#eval Nat.gcd 891 810  -- Expected output: 81

end max_students_equal_distribution_l3373_337359


namespace hyperbola_eccentricity_l3373_337324

/-- The eccentricity of a hyperbola with specific properties -/
theorem hyperbola_eccentricity (a b c : ℝ) (m n : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : c > 0) (h4 : m * n = 2 / 9) :
  let f (x y : ℝ) := x^2 / a^2 - y^2 / b^2
  let asymptote (x : ℝ) := b / a * x
  let A : ℝ × ℝ := (c, asymptote c)
  let B : ℝ × ℝ := (c, -asymptote c)
  let P : ℝ × ℝ := ((m + n) * c, (m - n) * asymptote c)
  (f (P.1) (P.2) = 1) →
  (c / a = 3 * Real.sqrt 2 / 4) :=
by sorry

end hyperbola_eccentricity_l3373_337324


namespace power_inequality_l3373_337318

theorem power_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a ^ b > b ^ a) (hbc : b ^ c > c ^ b) : 
  a ^ c > c ^ a := by
  sorry

end power_inequality_l3373_337318


namespace sets_problem_l3373_337378

-- Define the sets A, B, and C
def A : Set ℝ := {x | 4 ≤ x ∧ x < 8}
def B : Set ℝ := {x | 5 < x ∧ x < 10}
def C (a : ℝ) : Set ℝ := {x | x > a}

-- Theorem statement
theorem sets_problem (a : ℝ) :
  (A ∪ B = {x : ℝ | 4 ≤ x ∧ x < 10}) ∧
  ((Set.univ \ A) ∩ B = {x : ℝ | 8 ≤ x ∧ x < 10}) ∧
  (Set.Nonempty (A ∩ C a) ↔ a < 8) := by
  sorry

end sets_problem_l3373_337378


namespace orange_harvest_l3373_337320

theorem orange_harvest (total_days : ℕ) (total_sacks : ℕ) (h1 : total_days = 14) (h2 : total_sacks = 56) :
  total_sacks / total_days = 4 := by
  sorry

end orange_harvest_l3373_337320


namespace problem_solution_l3373_337306

theorem problem_solution : (0.5 : ℝ)^3 - (0.1 : ℝ)^3 / (0.5 : ℝ)^2 + 0.05 + (0.1 : ℝ)^2 = 0.181 := by
  sorry

end problem_solution_l3373_337306


namespace stool_height_is_30_l3373_337399

/-- The height of the stool Alice needs to reach the light bulb -/
def stool_height : ℝ :=
  let ceiling_height : ℝ := 250  -- in cm
  let light_bulb_below_ceiling : ℝ := 15  -- in cm
  let alice_height : ℝ := 155  -- in cm
  let alice_reach : ℝ := 50  -- in cm
  let light_bulb_height : ℝ := ceiling_height - light_bulb_below_ceiling
  let alice_total_reach : ℝ := alice_height + alice_reach
  light_bulb_height - alice_total_reach

theorem stool_height_is_30 : stool_height = 30 := by
  sorry

end stool_height_is_30_l3373_337399


namespace sqrt500_approx_l3373_337362

/-- Approximate value of √5 -/
def sqrt5_approx : ℝ := 2.236

/-- Theorem stating that √500 is approximately 22.36 -/
theorem sqrt500_approx : ‖Real.sqrt 500 - 22.36‖ < 0.01 :=
  sorry

end sqrt500_approx_l3373_337362


namespace disjoint_circles_condition_l3373_337358

def circle1 (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 4
def circle2 (x y a : ℝ) : Prop := x^2 + (y - a)^2 = 1

def circles_disjoint (a : ℝ) : Prop :=
  ∀ x y, ¬(circle1 x y ∧ circle2 x y a)

theorem disjoint_circles_condition (a : ℝ) :
  circles_disjoint a ↔ (a > 1 + 2 * Real.sqrt 2 ∨ a < 1 - 2 * Real.sqrt 2) :=
sorry

end disjoint_circles_condition_l3373_337358


namespace altitude_intersection_location_depends_on_shape_l3373_337303

-- Define a triangle
structure Triangle where
  vertices : Fin 3 → ℝ × ℝ

-- Define the shape of a triangle
inductive TriangleShape
  | Acute
  | Right
  | Obtuse

-- Define the location of a point relative to a triangle
inductive PointLocation
  | Inside
  | OnVertex
  | Outside

-- Function to determine the shape of a triangle
def determineShape (t : Triangle) : TriangleShape :=
  sorry

-- Function to find the intersection point of altitudes
def altitudeIntersection (t : Triangle) : ℝ × ℝ :=
  sorry

-- Function to determine the location of a point relative to a triangle
def determinePointLocation (t : Triangle) (p : ℝ × ℝ) : PointLocation :=
  sorry

-- Theorem stating that the location of the altitude intersection depends on the triangle shape
theorem altitude_intersection_location_depends_on_shape (t : Triangle) :
  let shape := determineShape t
  let intersection := altitudeIntersection t
  let location := determinePointLocation t intersection
  (shape = TriangleShape.Acute → location = PointLocation.Inside) ∧
  (shape = TriangleShape.Right → location = PointLocation.OnVertex) ∧
  (shape = TriangleShape.Obtuse → location = PointLocation.Outside) :=
  sorry

end altitude_intersection_location_depends_on_shape_l3373_337303


namespace star_eight_ten_l3373_337352

/-- Custom operation * for rational numbers -/
def star (m n p : ℚ) (x y : ℚ) : ℚ := m * x + n * y + p

/-- Theorem stating that if 3 * 5 = 30 and 4 * 6 = 425, then 8 * 10 = 2005 -/
theorem star_eight_ten (m n p : ℚ) 
  (h1 : star m n p 3 5 = 30)
  (h2 : star m n p 4 6 = 425) : 
  star m n p 8 10 = 2005 := by
  sorry

end star_eight_ten_l3373_337352


namespace triangle_function_k_range_l3373_337361

-- Define the function f(x) = kx + 2
def f (k : ℝ) (x : ℝ) : ℝ := k * x + 2

-- Define the property of being a "triangle function" on a domain
def is_triangle_function (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ (x y z : ℝ), a ≤ x ∧ x ≤ b ∧ a ≤ y ∧ y ≤ b ∧ a ≤ z ∧ z ≤ b →
    f x + f y > f z ∧ f y + f z > f x ∧ f z + f x > f y

-- State the theorem
theorem triangle_function_k_range :
  ∀ k : ℝ, is_triangle_function (f k) 1 4 ↔ -2/7 < k ∧ k < 1 :=
sorry

end triangle_function_k_range_l3373_337361


namespace cubic_equation_roots_l3373_337316

theorem cubic_equation_roots : ∃ (x₁ x₂ x₃ : ℚ),
  x₁ = -3/4 ∧ x₂ = -4/3 ∧ x₃ = 5/2 ∧
  x₁ * x₂ = 1 ∧
  24 * x₁^3 - 10 * x₁^2 - 101 * x₁ - 60 = 0 ∧
  24 * x₂^3 - 10 * x₂^2 - 101 * x₂ - 60 = 0 ∧
  24 * x₃^3 - 10 * x₃^2 - 101 * x₃ - 60 = 0 :=
by
  sorry

#check cubic_equation_roots

end cubic_equation_roots_l3373_337316


namespace repeating_decimal_fraction_l3373_337381

-- Define the repeating decimals
def repeating_decimal_0_8 : ℚ := 8/9
def repeating_decimal_2_4 : ℚ := 22/9

-- State the theorem
theorem repeating_decimal_fraction :
  repeating_decimal_0_8 / repeating_decimal_2_4 = 4/11 := by
  sorry

end repeating_decimal_fraction_l3373_337381


namespace imaginary_part_of_complex_fraction_l3373_337311

theorem imaginary_part_of_complex_fraction (z : ℂ) : 
  z = (Complex.I : ℂ) / (1 + 2 * Complex.I) → Complex.im z = 1/5 := by
  sorry

end imaginary_part_of_complex_fraction_l3373_337311


namespace peter_stamps_l3373_337390

theorem peter_stamps (M : ℕ) : 
  M > 1 ∧ 
  M % 5 = 2 ∧ 
  M % 11 = 2 ∧ 
  M % 13 = 2 → 
  (∀ n : ℕ, n > 1 ∧ n % 5 = 2 ∧ n % 11 = 2 ∧ n % 13 = 2 → n ≥ M) → 
  M = 717 := by
sorry

end peter_stamps_l3373_337390


namespace mixture_weight_l3373_337391

/-- The weight of the mixture of two brands of vegetable ghee -/
theorem mixture_weight (brand_a_weight : ℝ) (brand_b_weight : ℝ) 
  (mix_ratio_a : ℝ) (mix_ratio_b : ℝ) (total_volume : ℝ) : 
  brand_a_weight = 900 →
  brand_b_weight = 750 →
  mix_ratio_a = 3 →
  mix_ratio_b = 2 →
  total_volume = 4 →
  (mix_ratio_a * total_volume * brand_a_weight + mix_ratio_b * total_volume * brand_b_weight) / 
  ((mix_ratio_a + mix_ratio_b) * 1000) = 3.36 := by
  sorry

#check mixture_weight

end mixture_weight_l3373_337391


namespace last_digit_for_multiple_of_five_l3373_337329

theorem last_digit_for_multiple_of_five (n : ℕ) : 
  (71360 ≤ n ∧ n ≤ 71369) ∧ (n % 5 = 0) → (n % 10 = 0 ∨ n % 10 = 5) :=
by sorry

end last_digit_for_multiple_of_five_l3373_337329


namespace max_value_on_ellipse_l3373_337325

theorem max_value_on_ellipse (x y : ℝ) (h : x^2 / 9 + y^2 / 4 = 1) :
  ∃ (max : ℝ), max = Real.sqrt 13 ∧ x + y ≤ max :=
by sorry

end max_value_on_ellipse_l3373_337325


namespace factor_expression_l3373_337335

theorem factor_expression (x : ℝ) : 54 * x^3 - 135 * x^5 = 27 * x^3 * (2 - 5 * x^2) := by
  sorry

end factor_expression_l3373_337335


namespace sum_of_squares_16_to_30_l3373_337357

theorem sum_of_squares_16_to_30 (sum_1_to_15 : ℕ) (sum_1_to_30 : ℕ) : 
  sum_1_to_15 = 1240 → 
  sum_1_to_30 = (30 * 31 * 61) / 6 →
  sum_1_to_30 - sum_1_to_15 = 8215 := by
  sorry

end sum_of_squares_16_to_30_l3373_337357


namespace lamps_per_room_l3373_337342

/-- Given a hotel with 147 lamps and 21 rooms, prove that each room gets 7 lamps. -/
theorem lamps_per_room :
  let total_lamps : ℕ := 147
  let total_rooms : ℕ := 21
  let lamps_per_room : ℕ := total_lamps / total_rooms
  lamps_per_room = 7 := by sorry

end lamps_per_room_l3373_337342


namespace nested_fraction_equality_l3373_337363

theorem nested_fraction_equality : 1 + 1 / (1 + 1 / (2 + 1)) = 7 / 4 := by
  sorry

end nested_fraction_equality_l3373_337363


namespace tenth_day_is_monday_l3373_337339

/-- Represents the days of the week -/
inductive DayOfWeek
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday
| Sunday

/-- Represents a month with its starting day and number of days -/
structure Month where
  startDay : DayOfWeek
  numDays : Nat

/-- Represents Teacher Zhang's running schedule -/
def runningDays : List DayOfWeek := [DayOfWeek.Monday, DayOfWeek.Saturday, DayOfWeek.Sunday]

/-- Calculate the day of the week for a given day in the month -/
def dayOfWeek (m : Month) (day : Nat) : DayOfWeek :=
  sorry

/-- The total running time in a month in minutes -/
def totalRunningTime : Nat := 5 * 60

/-- The theorem to be proved -/
theorem tenth_day_is_monday (m : Month) 
  (h1 : m.startDay = DayOfWeek.Saturday) 
  (h2 : m.numDays = 31) 
  (h3 : totalRunningTime = 5 * 60) : 
  dayOfWeek m 10 = DayOfWeek.Monday :=
sorry

end tenth_day_is_monday_l3373_337339


namespace area_ratio_circumference_ratio_l3373_337313

-- Define a circular park
structure CircularPark where
  diameter : ℝ
  diameter_pos : diameter > 0

-- Define the enlarged park
def enlargedPark (park : CircularPark) : CircularPark :=
  { diameter := 3 * park.diameter
    diameter_pos := by
      have h : park.diameter > 0 := park.diameter_pos
      linarith }

-- Theorem for area ratio
theorem area_ratio (park : CircularPark) :
  (enlargedPark park).diameter^2 / park.diameter^2 = 9 := by
sorry

-- Theorem for circumference ratio
theorem circumference_ratio (park : CircularPark) :
  (enlargedPark park).diameter / park.diameter = 3 := by
sorry

end area_ratio_circumference_ratio_l3373_337313


namespace friendly_integers_in_range_two_not_friendly_l3373_337304

def friendly (a : ℕ) : Prop :=
  ∃ m n : ℕ+, (m^2 + n) * (n^2 + m) = a * (m - n)^3

theorem friendly_integers_in_range :
  ∃ S : Finset ℕ, S.card ≥ 500 ∧ ∀ a ∈ S, a ∈ Finset.range 2013 ∧ friendly a :=
sorry

theorem two_not_friendly : ¬ friendly 2 :=
sorry

end friendly_integers_in_range_two_not_friendly_l3373_337304


namespace ratio_problem_l3373_337347

theorem ratio_problem (a b c x : ℝ) 
  (h1 : a / c = 3 / 7)
  (h2 : b / c = x / 7)
  (h3 : (a + b + c) / c = 2) :
  b / (a + c) = 2 / 5 := by
sorry

end ratio_problem_l3373_337347


namespace complex_magnitude_l3373_337336

theorem complex_magnitude (z : ℂ) : z = 1 + 2*I → Complex.abs z = Real.sqrt 5 := by
  sorry

end complex_magnitude_l3373_337336


namespace tan_alpha_eq_neg_one_third_l3373_337375

theorem tan_alpha_eq_neg_one_third (α : ℝ) 
  (h : (Real.cos (π/4 - α)) / (Real.cos (π/4 + α)) = 1/2) : 
  Real.tan α = -1/3 := by
  sorry

end tan_alpha_eq_neg_one_third_l3373_337375


namespace chord_length_l3373_337344

theorem chord_length (R : ℝ) (AB AC : ℝ) (h1 : R = 8) (h2 : AB = 10) 
  (h3 : AC = (2 * Real.pi * R) / 3) : 
  (AC : ℝ) = 8 * Real.sqrt 3 := by
  sorry

end chord_length_l3373_337344


namespace darias_remaining_balance_l3373_337326

/-- Calculates the remaining amount owed on a credit card after an initial payment --/
def remaining_balance (saved : ℕ) (couch_price : ℕ) (table_price : ℕ) (lamp_price : ℕ) : ℕ :=
  (couch_price + table_price + lamp_price) - saved

/-- Theorem stating that Daria's remaining balance is $400 --/
theorem darias_remaining_balance :
  remaining_balance 500 750 100 50 = 400 := by
  sorry

end darias_remaining_balance_l3373_337326


namespace prime_quadruple_theorem_l3373_337364

def is_valid_quadruple (p₁ p₂ p₃ p₄ : Nat) : Prop :=
  Nat.Prime p₁ ∧ Nat.Prime p₂ ∧ Nat.Prime p₃ ∧ Nat.Prime p₄ ∧
  p₁ < p₂ ∧ p₂ < p₃ ∧ p₃ < p₄ ∧
  p₁ * p₂ + p₂ * p₃ + p₃ * p₄ + p₄ * p₁ = 882

theorem prime_quadruple_theorem :
  ∀ p₁ p₂ p₃ p₄ : Nat,
  is_valid_quadruple p₁ p₂ p₃ p₄ ↔
  ((p₁, p₂, p₃, p₄) = (2, 5, 19, 37) ∨
   (p₁, p₂, p₃, p₄) = (2, 11, 19, 31) ∨
   (p₁, p₂, p₃, p₄) = (2, 13, 19, 29)) :=
by sorry

end prime_quadruple_theorem_l3373_337364


namespace remaining_wallpaper_time_l3373_337374

/-- Time to remove wallpaper from one wall -/
def time_per_wall : ℕ := 2

/-- Number of walls in dining room -/
def dining_walls : ℕ := 4

/-- Number of walls in living room -/
def living_walls : ℕ := 4

/-- Time already spent removing wallpaper -/
def time_spent : ℕ := 2

/-- Theorem: The remaining time to remove wallpaper is 14 hours -/
theorem remaining_wallpaper_time :
  (time_per_wall * dining_walls + time_per_wall * living_walls) - time_spent = 14 := by
  sorry

end remaining_wallpaper_time_l3373_337374


namespace all_equilateral_triangles_similar_l3373_337331

/-- An equilateral triangle -/
structure EquilateralTriangle :=
  (side : ℝ)
  (side_positive : side > 0)

/-- Definition of similarity for triangles -/
def similar_triangles (t1 t2 : EquilateralTriangle) : Prop :=
  ∃ k : ℝ, k > 0 ∧ t1.side = k * t2.side

/-- Theorem: All equilateral triangles are similar -/
theorem all_equilateral_triangles_similar (t1 t2 : EquilateralTriangle) :
  similar_triangles t1 t2 :=
sorry

end all_equilateral_triangles_similar_l3373_337331


namespace hillarys_reading_assignment_l3373_337372

theorem hillarys_reading_assignment 
  (total_assignment : ℕ) 
  (friday_reading : ℕ) 
  (saturday_reading : ℕ) :
  total_assignment = 60 →
  friday_reading = 16 →
  saturday_reading = 28 →
  total_assignment - (friday_reading + saturday_reading) = 16 :=
by sorry

end hillarys_reading_assignment_l3373_337372


namespace max_value_x_plus_reciprocal_l3373_337341

theorem max_value_x_plus_reciprocal (x : ℝ) (h : 11 = x^2 + 1/x^2) :
  ∃ (y : ℝ), y = x + 1/x ∧ y ≤ Real.sqrt 13 ∧ ∃ (z : ℝ), z = x + 1/x ∧ z = Real.sqrt 13 :=
sorry

end max_value_x_plus_reciprocal_l3373_337341


namespace smallest_m_no_real_roots_l3373_337345

theorem smallest_m_no_real_roots : 
  let equation (m x : ℝ) := 3 * x * ((m + 1) * x - 5) - x^2 + 8
  ∀ m : ℤ, (∀ x : ℝ, equation m x ≠ 0) → m ≥ 2 ∧ 
  ∃ m' : ℤ, m' < 2 ∧ ∃ x : ℝ, equation (m' : ℝ) x = 0 :=
by sorry

end smallest_m_no_real_roots_l3373_337345


namespace geometric_sequence_product_equality_l3373_337360

/-- Given four non-zero real numbers, prove that forming a geometric sequence
    is sufficient but not necessary for their product equality. -/
theorem geometric_sequence_product_equality (a b c d : ℝ) (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0) :
  (∃ r : ℝ, b = a * r ∧ c = b * r ∧ d = c * r) → a * d = b * c ∧
  ∃ a' b' c' d' : ℝ, a' * d' = b' * c' ∧ ¬(∃ r : ℝ, b' = a' * r ∧ c' = b' * r ∧ d' = c' * r) :=
by sorry

end geometric_sequence_product_equality_l3373_337360


namespace line_slope_angle_l3373_337334

theorem line_slope_angle (x y : ℝ) :
  x + Real.sqrt 3 * y - 2 = 0 →
  ∃ (m : ℝ), y = m * x + (2 * Real.sqrt 3) / 3 ∧
             m = -(Real.sqrt 3) / 3 ∧
             Real.tan (5 * Real.pi / 6) = m :=
by sorry

end line_slope_angle_l3373_337334


namespace wax_spilled_amount_l3373_337310

/-- The amount of wax spilled before use -/
def wax_spilled (car_wax SUV_wax initial_wax remaining_wax : ℕ) : ℕ :=
  initial_wax - (car_wax + SUV_wax) - remaining_wax

/-- Theorem stating that the amount of wax spilled is 2 ounces -/
theorem wax_spilled_amount :
  wax_spilled 3 4 11 2 = 2 := by sorry

end wax_spilled_amount_l3373_337310


namespace regular_polygon_sides_l3373_337349

theorem regular_polygon_sides (n : ℕ) (interior_angle : ℝ) : 
  n ≥ 3 → 
  interior_angle = 108 → 
  (n : ℝ) * (180 - interior_angle) = 360 → 
  n = 5 :=
by sorry

end regular_polygon_sides_l3373_337349


namespace z_in_first_quadrant_l3373_337332

theorem z_in_first_quadrant (z : ℂ) (h : (3 + 2*I)*z = 13*I) : 
  0 < z.re ∧ 0 < z.im := by
  sorry

end z_in_first_quadrant_l3373_337332


namespace greatest_number_with_odd_factors_l3373_337327

-- Define a function to count positive factors
def count_positive_factors (n : ℕ) : ℕ := sorry

-- Define a function to check if a number is a perfect square
def is_perfect_square (n : ℕ) : Prop := sorry

theorem greatest_number_with_odd_factors :
  ∀ n : ℕ, n < 150 → count_positive_factors n % 2 = 1 → n ≤ 144 :=
by sorry

end greatest_number_with_odd_factors_l3373_337327


namespace on_time_departure_rate_theorem_l3373_337343

/-- The number of flights that departed late -/
def late_flights : ℕ := 1

/-- The number of initial on-time flights -/
def initial_on_time : ℕ := 3

/-- The number of additional on-time flights needed -/
def additional_on_time : ℕ := 4

/-- The total number of flights -/
def total_flights : ℕ := late_flights + initial_on_time + additional_on_time

/-- The target on-time departure rate as a real number between 0 and 1 -/
def target_rate : ℝ := 0.875

theorem on_time_departure_rate_theorem :
  (initial_on_time + additional_on_time : ℝ) / total_flights > target_rate :=
sorry

end on_time_departure_rate_theorem_l3373_337343


namespace dart_score_proof_l3373_337371

def bullseye_points : ℕ := 50
def missed_points : ℕ := 0
def third_dart_points : ℕ := bullseye_points / 2

def total_score : ℕ := bullseye_points + missed_points + third_dart_points

theorem dart_score_proof : total_score = 75 := by
  sorry

end dart_score_proof_l3373_337371


namespace problem_statement_l3373_337388

theorem problem_statement : (5/12 : ℝ)^2022 * (-2.4)^2023 = -12/5 := by
  sorry

end problem_statement_l3373_337388


namespace julio_fish_count_l3373_337301

/-- Calculates the number of fish Julio has after fishing for a given number of hours and losing some fish. -/
def fish_count (catch_rate : ℕ) (hours : ℕ) (fish_lost : ℕ) : ℕ :=
  catch_rate * hours - fish_lost

/-- Theorem stating that Julio has 48 fish after 9 hours of fishing at 7 fish per hour and losing 15 fish. -/
theorem julio_fish_count :
  fish_count 7 9 15 = 48 := by
  sorry

end julio_fish_count_l3373_337301


namespace trajectory_is_line_segment_l3373_337366

/-- Fixed point F₁ -/
def F₁ : ℝ × ℝ := (-4, 0)

/-- Fixed point F₂ -/
def F₂ : ℝ × ℝ := (4, 0)

/-- The set of points M satisfying the condition |MF₁| + |MF₂| = 8 -/
def trajectory : Set (ℝ × ℝ) :=
  {M : ℝ × ℝ | dist M F₁ + dist M F₂ = 8}

/-- Theorem stating that the trajectory is a line segment -/
theorem trajectory_is_line_segment :
  ∃ (A B : ℝ × ℝ), trajectory = {M : ℝ × ℝ | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ M = (1 - t) • A + t • B} :=
sorry

end trajectory_is_line_segment_l3373_337366


namespace two_books_adjacent_probability_l3373_337323

theorem two_books_adjacent_probability (n : ℕ) (h : n = 10) :
  let total_arrangements := n.factorial
  let favorable_arrangements := ((n - 1).factorial * 2)
  (favorable_arrangements : ℚ) / total_arrangements = 1 / 5 := by
  sorry

end two_books_adjacent_probability_l3373_337323


namespace max_daily_profit_l3373_337307

/-- The daily profit function for a factory -/
def daily_profit (x : ℕ) : ℚ :=
  -4/3 * (x^3 : ℚ) + 3600 * (x : ℚ)

/-- The maximum daily production capacity -/
def max_production : ℕ := 40

/-- Theorem stating the maximum daily profit and the production quantity that achieves it -/
theorem max_daily_profit :
  ∃ (x : ℕ), x ≤ max_production ∧
    (∀ (y : ℕ), y ≤ max_production → daily_profit y ≤ daily_profit x) ∧
    x = 30 ∧ daily_profit x = 72000 := by
  sorry

end max_daily_profit_l3373_337307


namespace zero_in_interval_l3373_337370

-- Define the function f(x) = 2x + 3x
def f (x : ℝ) : ℝ := 2*x + 3*x

-- Theorem stating that the zero of f(x) is in the interval (-1, 0)
theorem zero_in_interval :
  ∃ x, x ∈ Set.Ioo (-1 : ℝ) 0 ∧ f x = 0 :=
by
  sorry

end zero_in_interval_l3373_337370


namespace power_four_mod_nine_l3373_337394

theorem power_four_mod_nine : 4^3023 % 9 = 7 := by sorry

end power_four_mod_nine_l3373_337394


namespace boyds_male_friends_percentage_l3373_337322

theorem boyds_male_friends_percentage 
  (julian_total : ℕ) 
  (julian_boys_percent : ℚ) 
  (boyd_total : ℕ) 
  (boyd_girls_multiplier : ℕ) : 
  julian_total = 80 → 
  julian_boys_percent = 60 / 100 → 
  boyd_total = 100 → 
  boyd_girls_multiplier = 2 → 
  (boyd_total - boyd_girls_multiplier * (julian_total * (1 - julian_boys_percent))) / boyd_total = 36 / 100 := by
  sorry

end boyds_male_friends_percentage_l3373_337322


namespace intersection_complement_equality_l3373_337398

def U : Set Nat := {0, 1, 2, 3}
def M : Set Nat := {0, 1, 2}
def N : Set Nat := {0, 2, 3}

theorem intersection_complement_equality :
  M ∩ (U \ N) = {1} := by sorry

end intersection_complement_equality_l3373_337398


namespace repeating_decimal_division_l3373_337397

-- Define the repeating decimals
def repeating_decimal_72 : ℚ := 8/11
def repeating_decimal_124 : ℚ := 41/33

-- State the theorem
theorem repeating_decimal_division :
  repeating_decimal_72 / repeating_decimal_124 = 264/451 := by
  sorry

end repeating_decimal_division_l3373_337397


namespace grocery_store_bottles_l3373_337314

/-- The total number of soda bottles in a grocery store. -/
def total_bottles (regular : ℕ) (diet : ℕ) (lite : ℕ) : ℕ :=
  regular + diet + lite

/-- Theorem stating that the total number of bottles is 110. -/
theorem grocery_store_bottles : total_bottles 57 26 27 = 110 := by
  sorry

end grocery_store_bottles_l3373_337314


namespace combined_work_rate_l3373_337315

/-- The combined work rate of three workers given their individual work rates -/
theorem combined_work_rate 
  (rate_A : ℚ) 
  (rate_B : ℚ) 
  (rate_C : ℚ) 
  (h_A : rate_A = 1 / 12)
  (h_B : rate_B = 1 / 6)
  (h_C : rate_C = 1 / 18) : 
  rate_A + rate_B + rate_C = 11 / 36 := by
  sorry

#check combined_work_rate

end combined_work_rate_l3373_337315


namespace intersection_height_l3373_337387

/-- Triangle ABC with vertices A(0, 7), B(3, 0), and C(9, 0) -/
structure Triangle :=
  (A : ℝ × ℝ)
  (B : ℝ × ℝ)
  (C : ℝ × ℝ)

/-- Horizontal line y = t intersecting AB at T and AC at U -/
structure Intersection (ABC : Triangle) (t : ℝ) :=
  (T : ℝ × ℝ)
  (U : ℝ × ℝ)

/-- The area of a triangle given three points -/
def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

/-- Main theorem -/
theorem intersection_height (ABC : Triangle) (t : ℝ) (intr : Intersection ABC t) :
  ABC.A = (0, 7) ∧ ABC.B = (3, 0) ∧ ABC.C = (9, 0) →
  triangleArea ABC.A intr.T intr.U = 18 →
  t = 7 - Real.sqrt 42 :=
by sorry

end intersection_height_l3373_337387


namespace circle_tangent_problem_l3373_337353

/-- Two lines l₁ and l₂ are perpendicular if their slopes multiply to -1 -/
def perpendicular (a : ℝ) : Prop := a * (1/a) = -1

/-- A line ax + by + c = 0 is tangent to the circle x² + y² = r² 
    if the distance from (0,0) to the line equals r -/
def tangent_to_circle (a b c r : ℝ) : Prop :=
  (c / (a^2 + b^2).sqrt)^2 = r^2

theorem circle_tangent_problem (a : ℝ) :
  perpendicular a →
  tangent_to_circle 1 0 2 (b^2).sqrt →
  b = 4 := by
  sorry

end circle_tangent_problem_l3373_337353


namespace quadratic_sum_l3373_337383

/-- Given a quadratic polynomial 12x^2 + 72x + 300, prove that when written
    in the form a(x+b)^2+c, where a, b, and c are constants, a + b + c = 207 -/
theorem quadratic_sum (x : ℝ) :
  ∃ (a b c : ℝ), (∀ x, 12*x^2 + 72*x + 300 = a*(x+b)^2 + c) ∧ (a + b + c = 207) := by
  sorry

end quadratic_sum_l3373_337383


namespace min_value_expression_l3373_337338

theorem min_value_expression (n : ℕ+) : 
  (n : ℝ) / 3 + 27 / (n : ℝ) ≥ 6 ∧ 
  ((n : ℝ) / 3 + 27 / (n : ℝ) = 6 ↔ n = 9) := by
sorry

end min_value_expression_l3373_337338


namespace speed_calculation_l3373_337309

/-- Given a distance of 900 meters covered in 180 seconds, prove that the speed is 18 km/h -/
theorem speed_calculation (distance : ℝ) (time : ℝ) (h1 : distance = 900) (h2 : time = 180) :
  (distance / 1000) / (time / 3600) = 18 := by
  sorry

end speed_calculation_l3373_337309


namespace complement_intersection_theorem_l3373_337333

def I : Finset Nat := {0, 1, 2, 3, 4}
def M : Finset Nat := {1, 2, 3}
def N : Finset Nat := {0, 3, 4}

theorem complement_intersection_theorem :
  (I \ M) ∩ N = {0, 4} := by sorry

end complement_intersection_theorem_l3373_337333


namespace opposite_sides_range_l3373_337356

/-- Given that the origin (0, 0) and the point (1, 1) are on opposite sides of the line x + y - a = 0,
    prove that the range of values for a is (0, 2) -/
theorem opposite_sides_range (a : ℝ) : 
  (∀ (x y : ℝ), x + y - a = 0 → (x = 0 ∧ y = 0) ∨ (x = 1 ∧ y = 1)) →
  (0 < a ∧ a < 2) :=
sorry

end opposite_sides_range_l3373_337356


namespace unique_two_digit_number_l3373_337365

/-- A two-digit number satisfying the given conditions -/
def TwoDigitNumber (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧
  ∃ (x : ℕ), 
    let tens := n / 10
    let ones := n % 10
    tens = ones^2 - 9 ∧
    10 * ones + tens = n - 27

theorem unique_two_digit_number : 
  ∃! (n : ℕ), TwoDigitNumber n ∧ n = 74 := by
  sorry

end unique_two_digit_number_l3373_337365


namespace opposite_of_negative_sqrt_two_l3373_337355

theorem opposite_of_negative_sqrt_two : -(-(Real.sqrt 2)) = Real.sqrt 2 := by
  sorry

end opposite_of_negative_sqrt_two_l3373_337355


namespace expected_rounds_range_l3373_337377

/-- Represents the game between players A and B -/
structure Game where
  p : ℝ
  h_p_pos : 0 < p
  h_p_lt_one : p < 1

/-- The expected number of rounds in the game -/
noncomputable def expected_rounds (g : Game) : ℝ :=
  2 * (1 - (2 * g.p * (1 - g.p))^10) / (1 - 2 * g.p * (1 - g.p))

/-- Theorem stating the range of the expected number of rounds -/
theorem expected_rounds_range (g : Game) :
  2 < expected_rounds g ∧ expected_rounds g ≤ 4 - (1/2)^8 :=
sorry

end expected_rounds_range_l3373_337377


namespace final_white_pieces_l3373_337393

/-- Recursively calculates the number of remaining white pieces after each round of removal -/
def remainingWhitePieces : ℕ → ℕ
| 0 => 1990  -- Initial number of white pieces
| (n + 1) =>
  let previous := remainingWhitePieces n
  if previous % 2 = 0 then
    previous / 2
  else
    (previous + 1) / 2

/-- Theorem stating that after the removal process, 124 white pieces remain -/
theorem final_white_pieces :
  ∃ n : ℕ, remainingWhitePieces n = 124 ∧ ∀ m > n, remainingWhitePieces m = 124 :=
sorry

end final_white_pieces_l3373_337393


namespace shadow_length_proportion_l3373_337382

/-- Given two objects side by side, if one object of height 20 units casts a shadow of 10 units,
    then another object of height 40 units will cast a shadow of 20 units. -/
theorem shadow_length_proportion
  (h1 : ℝ) (s1 : ℝ) (h2 : ℝ)
  (height_shadow_1 : h1 = 20)
  (shadow_1 : s1 = 10)
  (height_2 : h2 = 40)
  (proportion : h1 / s1 = h2 / (h2 / 2)) :
  h2 / 2 = 20 := by
sorry

end shadow_length_proportion_l3373_337382


namespace final_flow_rate_l3373_337385

/-- Represents the flow rate of cleaner through a pipe at different time intervals --/
structure FlowRate :=
  (initial : ℝ)
  (after15min : ℝ)
  (final : ℝ)

/-- Theorem stating that given the initial conditions and total cleaner used, 
    the final flow rate must be 4 ounces per minute --/
theorem final_flow_rate 
  (flow : FlowRate)
  (total_time : ℝ)
  (total_cleaner : ℝ)
  (h1 : flow.initial = 2)
  (h2 : flow.after15min = 3)
  (h3 : total_time = 30)
  (h4 : total_cleaner = 80)
  : flow.final = 4 := by
  sorry

#check final_flow_rate

end final_flow_rate_l3373_337385


namespace tart_base_flour_calculation_l3373_337302

theorem tart_base_flour_calculation (original_bases : ℕ) (original_flour : ℚ) 
  (new_bases : ℕ) (new_flour : ℚ) : 
  original_bases = 40 → 
  original_flour = 1/8 → 
  new_bases = 25 → 
  original_bases * original_flour = new_bases * new_flour → 
  new_flour = 1/5 := by
sorry

end tart_base_flour_calculation_l3373_337302


namespace georges_walk_to_school_l3373_337384

/-- Proves that given the conditions of George's walk to school, 
    the speed required for the second mile to arrive on time is 6 mph. -/
theorem georges_walk_to_school (total_distance : ℝ) (normal_speed : ℝ) 
  (normal_time : ℝ) (first_mile_speed : ℝ) :
  total_distance = 2 →
  normal_speed = 4 →
  normal_time = 0.5 →
  first_mile_speed = 3 →
  ∃ (second_mile_speed : ℝ),
    second_mile_speed = 6 ∧
    (1 / first_mile_speed + 1 / second_mile_speed = normal_time) :=
by sorry

end georges_walk_to_school_l3373_337384


namespace hospital_staff_count_l3373_337337

theorem hospital_staff_count (total : ℕ) (doctor_ratio nurse_ratio : ℕ) 
  (h1 : total = 250)
  (h2 : doctor_ratio = 2)
  (h3 : nurse_ratio = 3) :
  (nurse_ratio * total) / (doctor_ratio + nurse_ratio) = 150 := by
  sorry

end hospital_staff_count_l3373_337337


namespace min_hours_to_drive_l3373_337354

/-- The legal blood alcohol content (BAC) limit for driving -/
def legal_bac_limit : ℝ := 0.2

/-- The initial BAC after drinking -/
def initial_bac : ℝ := 0.8

/-- The rate at which BAC decreases per hour -/
def bac_decrease_rate : ℝ := 0.5

/-- The minimum number of hours to wait before driving -/
def min_wait_hours : ℕ := 2

/-- Theorem stating the minimum number of hours to wait before driving -/
theorem min_hours_to_drive :
  (initial_bac * (1 - bac_decrease_rate) ^ min_wait_hours ≤ legal_bac_limit) ∧
  (∀ h : ℕ, h < min_wait_hours → initial_bac * (1 - bac_decrease_rate) ^ h > legal_bac_limit) :=
sorry

end min_hours_to_drive_l3373_337354


namespace rug_selling_price_l3373_337380

/-- Proves that the selling price per rug is $60, given the cost price, number of rugs, and total profit --/
theorem rug_selling_price 
  (cost_price : ℝ) 
  (num_rugs : ℕ) 
  (total_profit : ℝ) 
  (h1 : cost_price = 40) 
  (h2 : num_rugs = 20) 
  (h3 : total_profit = 400) : 
  (cost_price * num_rugs + total_profit) / num_rugs = 60 := by
  sorry

end rug_selling_price_l3373_337380


namespace cosine_equation_solvability_l3373_337386

theorem cosine_equation_solvability (m : ℝ) :
  (∀ x : ℝ, ∃ y : ℝ, 4 * Real.cos y - Real.cos y ^ 2 + m - 3 = 0) ↔ 0 ≤ m ∧ m ≤ 8 := by
  sorry

end cosine_equation_solvability_l3373_337386


namespace reggies_book_cost_l3373_337389

theorem reggies_book_cost (initial_amount : ℕ) (books_bought : ℕ) (amount_left : ℕ) : 
  initial_amount = 48 →
  books_bought = 5 →
  amount_left = 38 →
  (initial_amount - amount_left) / books_bought = 2 :=
by
  sorry

end reggies_book_cost_l3373_337389


namespace liquid_X_percentage_l3373_337312

/-- The percentage of liquid X in solution A -/
def percent_X_in_A : ℝ := 1.464

/-- The percentage of liquid X in solution B -/
def percent_X_in_B : ℝ := 1.8

/-- The weight of solution A in grams -/
def weight_A : ℝ := 500

/-- The weight of solution B in grams -/
def weight_B : ℝ := 700

/-- The percentage of liquid X in the resulting mixture -/
def percent_X_in_mixture : ℝ := 1.66

theorem liquid_X_percentage :
  percent_X_in_A * weight_A / 100 + percent_X_in_B * weight_B / 100 =
  percent_X_in_mixture * (weight_A + weight_B) / 100 := by
  sorry

end liquid_X_percentage_l3373_337312


namespace ab_minimum_value_l3373_337379

theorem ab_minimum_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = a + b + 3) :
  a * b ≥ 9 := by
sorry

end ab_minimum_value_l3373_337379


namespace suit_price_increase_l3373_337330

/-- Proves that the percentage increase in the price of a suit was 20% --/
theorem suit_price_increase (original_price : ℝ) (coupon_discount : ℝ) (final_price : ℝ) :
  original_price = 150 →
  coupon_discount = 0.2 →
  final_price = 144 →
  ∃ (increase_percentage : ℝ),
    increase_percentage = 20 ∧
    final_price = (1 - coupon_discount) * (original_price * (1 + increase_percentage / 100)) :=
by sorry

end suit_price_increase_l3373_337330


namespace week_cycling_distance_l3373_337392

/-- Represents the cycling data for a single day -/
structure DailyRide where
  base_distance : Float
  speed_bonus : Float

/-- Calculates the effective distance for a single day -/
def effective_distance (ride : DailyRide) : Float :=
  ride.base_distance * (1 + ride.speed_bonus)

/-- Calculates the total effective distance for the week -/
def total_effective_distance (rides : List DailyRide) : Float :=
  rides.map effective_distance |> List.sum

/-- The main theorem: proves that the total effective distance is 367.05 km -/
theorem week_cycling_distance : 
  let monday : DailyRide := { base_distance := 40, speed_bonus := 0.05 }
  let tuesday : DailyRide := { base_distance := 50, speed_bonus := 0.03 }
  let wednesday : DailyRide := { base_distance := 25, speed_bonus := 0.07 }
  let thursday : DailyRide := { base_distance := 65, speed_bonus := 0.04 }
  let friday : DailyRide := { base_distance := 78, speed_bonus := 0.06 }
  let saturday : DailyRide := { base_distance := 58.5, speed_bonus := 0.02 }
  let sunday : DailyRide := { base_distance := 33.5, speed_bonus := 0.10 }
  let week_rides : List DailyRide := [monday, tuesday, wednesday, thursday, friday, saturday, sunday]
  total_effective_distance week_rides = 367.05 := by
  sorry


end week_cycling_distance_l3373_337392


namespace sixth_equation_pattern_l3373_337369

/-- The sum of n consecutive odd numbers starting from a given odd number -/
def sum_consecutive_odds (start : ℕ) (n : ℕ) : ℕ :=
  (start + n - 1) * n

/-- The nth cube -/
def cube (n : ℕ) : ℕ := n^3

theorem sixth_equation_pattern : sum_consecutive_odds 31 6 = cube 6 := by
  sorry

end sixth_equation_pattern_l3373_337369


namespace quadrant_I_solution_l3373_337367

theorem quadrant_I_solution (c : ℝ) :
  (∃ x y : ℝ, x - y = 3 ∧ c * x + y = 4 ∧ x > 0 ∧ y > 0) ↔ -1 < c ∧ c < 4/3 :=
by sorry

end quadrant_I_solution_l3373_337367


namespace security_to_bag_check_ratio_l3373_337376

def total_time : ℕ := 180
def uber_to_house : ℕ := 10
def check_bag_time : ℕ := 15
def wait_for_boarding : ℕ := 20

def uber_to_airport : ℕ := 5 * uber_to_house
def wait_for_takeoff : ℕ := 2 * wait_for_boarding

def known_time : ℕ := uber_to_house + uber_to_airport + check_bag_time + wait_for_boarding + wait_for_takeoff
def security_time : ℕ := total_time - known_time

theorem security_to_bag_check_ratio :
  security_time / check_bag_time = 3 ∧ security_time % check_bag_time = 0 :=
by sorry

end security_to_bag_check_ratio_l3373_337376


namespace gcd_1337_382_l3373_337328

theorem gcd_1337_382 : Nat.gcd 1337 382 = 191 := by
  sorry

end gcd_1337_382_l3373_337328


namespace max_queens_2017_l3373_337350

/-- Represents a chessboard of size n x n -/
def Chessboard (n : ℕ) := Fin n → Fin n → Bool

/-- Checks if a queen at position (x, y) attacks at most one other queen -/
def attacks_at_most_one (board : Chessboard 2017) (x y : Fin 2017) : Prop :=
  ∃! (x' y' : Fin 2017), x' ≠ x ∨ y' ≠ y ∧ board x' y' = true ∧
    (x' = x ∨ y' = y ∨ (x' : ℤ) - (x : ℤ) = (y' : ℤ) - (y : ℤ) ∨ 
     (x' : ℤ) - (x : ℤ) = (y : ℤ) - (y' : ℤ))

/-- The property that each queen on the board attacks at most one other queen -/
def valid_placement (board : Chessboard 2017) : Prop :=
  ∀ x y, board x y = true → attacks_at_most_one board x y

/-- Counts the number of queens on the board -/
def count_queens (board : Chessboard 2017) : ℕ :=
  (Finset.univ.filter (λ x : Fin 2017 × Fin 2017 => board x.1 x.2 = true)).card

/-- The main theorem: there exists a valid placement with 673359 queens -/
theorem max_queens_2017 : 
  ∃ (board : Chessboard 2017), valid_placement board ∧ count_queens board = 673359 :=
sorry

end max_queens_2017_l3373_337350


namespace total_deposit_l3373_337396

def mark_deposit : ℕ := 88
def bryan_deposit : ℕ := 5 * mark_deposit - 40

theorem total_deposit : mark_deposit + bryan_deposit = 488 := by
  sorry

end total_deposit_l3373_337396


namespace perimeter_of_square_C_l3373_337346

/-- Given squares A, B, and C with specific perimeter relationships, 
    prove that the perimeter of C is 100. -/
theorem perimeter_of_square_C (A B C : ℝ) : 
  (A > 0) →  -- A is positive (side length of a square)
  (B > 0) →  -- B is positive (side length of a square)
  (C > 0) →  -- C is positive (side length of a square)
  (4 * A = 20) →  -- Perimeter of A is 20
  (4 * B = 40) →  -- Perimeter of B is 40
  (C = A + 2 * B) →  -- Side length of C relationship
  (4 * C = 100) :=  -- Perimeter of C is 100
by sorry

end perimeter_of_square_C_l3373_337346


namespace one_pair_probability_l3373_337395

/-- The number of colors of socks -/
def num_colors : ℕ := 5

/-- The number of socks per color -/
def socks_per_color : ℕ := 2

/-- The total number of socks -/
def total_socks : ℕ := num_colors * socks_per_color

/-- The number of socks drawn -/
def socks_drawn : ℕ := 5

/-- The probability of drawing exactly one pair of socks with the same color -/
def prob_one_pair : ℚ := 20 / 31.5

theorem one_pair_probability : 
  (num_colors.choose 1 * socks_per_color.choose 2 * (num_colors - 1).choose 3 * (socks_per_color.choose 1)^3) / 
  (total_socks.choose socks_drawn) = prob_one_pair :=
sorry

end one_pair_probability_l3373_337395


namespace absolute_value_inequality_solution_set_l3373_337368

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |2*x - 5| > 1} = {x : ℝ | x < 2 ∨ x > 3} := by
sorry

end absolute_value_inequality_solution_set_l3373_337368


namespace equation_solution_l3373_337348

theorem equation_solution :
  ∀ (A B C : ℕ),
    3 * A - A = 10 →
    B + A = 12 →
    C - B = 6 →
    A ≠ B →
    B ≠ C →
    A ≠ C →
    C = 13 :=
by
  sorry

end equation_solution_l3373_337348


namespace program_sum_equals_expected_sum_l3373_337340

def program_sum (n : ℕ) : ℕ :=
  let rec inner_sum (k : ℕ) : ℕ :=
    match k with
    | 0 => 0
    | k+1 => k+1 + inner_sum k
  let rec outer_sum (i : ℕ) : ℕ :=
    match i with
    | 0 => 0
    | i+1 => inner_sum (i+1) + outer_sum i
  outer_sum n

def expected_sum (n : ℕ) : ℕ :=
  let rec sum_of_sums (k : ℕ) : ℕ :=
    match k with
    | 0 => 0
    | k+1 => (List.range (k+1)).sum + sum_of_sums k
  sum_of_sums n

theorem program_sum_equals_expected_sum (n : ℕ) :
  program_sum n = expected_sum n := by
  sorry

end program_sum_equals_expected_sum_l3373_337340


namespace vipers_count_l3373_337308

/-- The number of vipers in a swamp area -/
def num_vipers (num_crocodiles num_alligators total_animals : ℕ) : ℕ :=
  total_animals - (num_crocodiles + num_alligators)

/-- Theorem: The number of vipers in the swamp is 5 -/
theorem vipers_count : num_vipers 22 23 50 = 5 := by
  sorry

end vipers_count_l3373_337308


namespace vector_magnitude_range_l3373_337317

/-- Given unit vectors e₁ and e₂ with an angle of 120° between them, 
    and x, y ∈ ℝ such that |x*e₁ + y*e₂| = √3, 
    prove that 1 ≤ |x*e₁ - y*e₂| ≤ 3 -/
theorem vector_magnitude_range (e₁ e₂ : ℝ × ℝ) (x y : ℝ) :
  (e₁.1^2 + e₁.2^2 = 1) →  -- e₁ is a unit vector
  (e₂.1^2 + e₂.2^2 = 1) →  -- e₂ is a unit vector
  (e₁.1 * e₂.1 + e₁.2 * e₂.2 = -1/2) →  -- angle between e₁ and e₂ is 120°
  ((x*e₁.1 + y*e₂.1)^2 + (x*e₁.2 + y*e₂.2)^2 = 3) →  -- |x*e₁ + y*e₂| = √3
  1 ≤ ((x*e₁.1 - y*e₂.1)^2 + (x*e₁.2 - y*e₂.2)^2) ∧ 
  ((x*e₁.1 - y*e₂.1)^2 + (x*e₁.2 - y*e₂.2)^2) ≤ 9 := by
  sorry

end vector_magnitude_range_l3373_337317


namespace triangles_on_square_sides_l3373_337305

/-- The number of triangles formed by 12 points on the sides of a square -/
def num_triangles_on_square_sides : ℕ := 216

/-- The total number of points on the sides of the square -/
def total_points : ℕ := 12

/-- The number of sides of the square -/
def num_sides : ℕ := 4

/-- The number of points on each side of the square (excluding vertices) -/
def points_per_side : ℕ := 3

/-- Theorem stating the number of triangles formed by points on square sides -/
theorem triangles_on_square_sides :
  num_triangles_on_square_sides = 
    (total_points.choose 3) - (num_sides * points_per_side.choose 3) :=
by sorry

end triangles_on_square_sides_l3373_337305

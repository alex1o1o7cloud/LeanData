import Mathlib

namespace sqrt_50_between_consecutive_integers_product_l2463_246358

theorem sqrt_50_between_consecutive_integers_product : ∃ (n : ℕ), 
  (n : ℝ) < Real.sqrt 50 ∧ Real.sqrt 50 < (n + 1 : ℝ) ∧ n * (n + 1) = 56 := by
  sorry

end sqrt_50_between_consecutive_integers_product_l2463_246358


namespace intersection_sum_l2463_246318

-- Define the sets A, B, and C
def A : Set ℝ := {x : ℝ | -5 < x ∧ x < 5}
def B (a : ℝ) : Set ℝ := {x : ℝ | -7 < x ∧ x < a}
def C (b : ℝ) : Set ℝ := {x : ℝ | b < x ∧ x < 2}

-- State the theorem
theorem intersection_sum (a b : ℝ) (h : A ∩ B a = C b) : a + b = -3 := by
  sorry

end intersection_sum_l2463_246318


namespace ellipse_perpendicular_sum_l2463_246395

/-- Theorem about perpendicular distances in an ellipse -/
theorem ellipse_perpendicular_sum (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_a_ge_b : a ≥ b) :
  let e := Real.sqrt (a^2 - b^2)
  ∀ (x₀ y₀ : ℝ), x₀^2 / a^2 + y₀^2 / b^2 = 1 →
    let d₁ := |y₀ - b| / b / Real.sqrt ((x₀/a^2)^2 + (y₀/b^2)^2)
    let d₂ := |y₀ + b| / b / Real.sqrt ((x₀/a^2)^2 + (y₀/b^2)^2)
    d₁^2 + d₂^2 = 2 * a^2 :=
by sorry

end ellipse_perpendicular_sum_l2463_246395


namespace fib_mod_eq_closed_form_twelve_squared_eq_five_solutions_of_quadratic_inverse_of_twelve_l2463_246398

/-- The Fibonacci sequence modulo 139 -/
def fib_mod (n : ℕ) : Fin 139 :=
  if n = 0 then 0
  else if n = 1 then 1
  else (fib_mod (n - 1) + fib_mod (n - 2))

/-- The closed form expression for the Fibonacci sequence modulo 139 -/
def fib_closed_form (n : ℕ) : Fin 139 :=
  58 * (76^n - 64^n)

/-- Theorem stating that the Fibonacci sequence modulo 139 is equivalent to the closed form expression -/
theorem fib_mod_eq_closed_form (n : ℕ) : fib_mod n = fib_closed_form n := by
  sorry

/-- 12 is a solution of y² ≡ 5 (mod 139) -/
theorem twelve_squared_eq_five : (12 : Fin 139)^2 = 5 := by
  sorry

/-- 64 and 76 are solutions of x² - x - 1 ≡ 0 (mod 139) -/
theorem solutions_of_quadratic : 
  ((64 : Fin 139)^2 - 64 - 1 = 0) ∧ ((76 : Fin 139)^2 - 76 - 1 = 0) := by
  sorry

/-- 58 is the modular multiplicative inverse of 12 modulo 139 -/
theorem inverse_of_twelve : (12 : Fin 139) * 58 = 1 := by
  sorry

end fib_mod_eq_closed_form_twelve_squared_eq_five_solutions_of_quadratic_inverse_of_twelve_l2463_246398


namespace pencils_across_diameter_l2463_246301

-- Define the radius of the circle in feet
def radius : ℝ := 14

-- Define the length of a pencil in feet
def pencil_length : ℝ := 0.5

-- Theorem: The number of pencils that can be placed end-to-end across the diameter is 56
theorem pencils_across_diameter : 
  ⌊(2 * radius) / pencil_length⌋ = 56 := by sorry

end pencils_across_diameter_l2463_246301


namespace ink_covered_term_l2463_246386

variables {a b : ℝ}

theorem ink_covered_term (h : ∃ x, x * 3 * a * b = 6 * a * b - 3 * a * b ^ 3) :
  ∃ x, x = 2 - b ^ 2 ∧ x * 3 * a * b = 6 * a * b - 3 * a * b ^ 3 := by
sorry

end ink_covered_term_l2463_246386


namespace maria_earnings_l2463_246397

/-- The cost of brushes in dollars -/
def brush_cost : ℕ := 20

/-- The cost of canvas in dollars -/
def canvas_cost : ℕ := 3 * brush_cost

/-- The cost of paint per liter in dollars -/
def paint_cost_per_liter : ℕ := 8

/-- The minimum number of liters of paint needed -/
def paint_liters : ℕ := 5

/-- The selling price of the painting in dollars -/
def selling_price : ℕ := 200

/-- Maria's earnings from selling the painting -/
def earnings : ℕ := selling_price - (brush_cost + canvas_cost + paint_cost_per_liter * paint_liters)

theorem maria_earnings : earnings = 80 := by
  sorry

end maria_earnings_l2463_246397


namespace power_two_plus_one_div_three_l2463_246392

theorem power_two_plus_one_div_three (n : ℕ+) :
  3 ∣ (2^n.val + 1) ↔ n.val % 2 = 1 :=
sorry

end power_two_plus_one_div_three_l2463_246392


namespace trevor_eggs_end_wednesday_l2463_246312

/-- Represents the number of eggs laid by a chicken on a given day -/
structure ChickenEggs :=
  (monday : ℕ)
  (tuesday : ℕ)
  (wednesday : ℕ)

/-- Represents the egg-laying data for all chickens -/
def chicken_data : List ChickenEggs := [
  ⟨4, 6, 4⟩,  -- Gertrude
  ⟨3, 3, 3⟩,  -- Blanche
  ⟨2, 1, 2⟩,  -- Nancy
  ⟨3, 4, 3⟩,  -- Martha
  ⟨5, 3, 5⟩,  -- Ophelia
  ⟨1, 3, 1⟩,  -- Penelope
  ⟨3, 1, 3⟩,  -- Quinny
  ⟨4, 0, 4⟩   -- Rosie
]

def eggs_eaten_per_day : ℕ := 2
def eggs_dropped_monday : ℕ := 3
def eggs_dropped_wednesday : ℕ := 3

def total_eggs_collected (data : List ChickenEggs) : ℕ :=
  (data.map (·.monday)).sum + (data.map (·.tuesday)).sum + (data.map (·.wednesday)).sum

def eggs_eaten_total (days : ℕ) : ℕ :=
  eggs_eaten_per_day * days

def eggs_dropped_total : ℕ :=
  eggs_dropped_monday + eggs_dropped_wednesday

def eggs_sold (data : List ChickenEggs) : ℕ :=
  (data.map (·.tuesday)).sum / 2

theorem trevor_eggs_end_wednesday :
  total_eggs_collected chicken_data - 
  eggs_eaten_total 3 - 
  eggs_dropped_total - 
  eggs_sold chicken_data = 49 := by
  sorry

end trevor_eggs_end_wednesday_l2463_246312


namespace inequality_solution_set_l2463_246335

theorem inequality_solution_set (a : ℝ) : 
  (∃ x : ℝ, |x - 3| + |x - 4| < a) → a > 1 := by
  sorry

end inequality_solution_set_l2463_246335


namespace equation_solution_l2463_246377

theorem equation_solution : ∃ (q r : ℝ), 
  q ≠ r ∧ 
  q > r ∧
  ((5 * q - 15) / (q^2 + q - 20) = q + 3) ∧
  ((5 * r - 15) / (r^2 + r - 20) = r + 3) ∧
  q - r = 2 := by
  sorry

end equation_solution_l2463_246377


namespace complex_equation_solutions_l2463_246322

theorem complex_equation_solutions :
  ∃! (s : Finset ℂ), (∀ z ∈ s, Complex.abs z < 15 ∧ Complex.exp z = (z - 2) / (z + 2)) ∧ Finset.card s = 2 :=
by sorry

end complex_equation_solutions_l2463_246322


namespace proportion_solution_l2463_246348

theorem proportion_solution (x : ℝ) (h : (3/4) / x = 5/8) : x = 6/5 := by
  sorry

end proportion_solution_l2463_246348


namespace value_range_of_f_l2463_246343

/-- The function f(x) = x^2 - 2x -/
def f (x : ℝ) : ℝ := x^2 - 2*x

/-- The domain of f is [0, +∞) -/
def domain : Set ℝ := { x | x ≥ 0 }

theorem value_range_of_f :
  { y | ∃ x ∈ domain, f x = y } = { y | y ≥ -1 } := by sorry

end value_range_of_f_l2463_246343


namespace mika_initial_stickers_l2463_246307

def initial_stickers (total : ℝ) (store : ℝ) (birthday : ℝ) (sister : ℝ) (mother : ℝ) : ℝ :=
  total - (store + birthday + sister + mother)

theorem mika_initial_stickers :
  initial_stickers 130 26 20 6 58 = 20 := by
  sorry

end mika_initial_stickers_l2463_246307


namespace tech_group_selection_l2463_246315

theorem tech_group_selection (total : ℕ) (select : ℕ) (ways_with_girl : ℕ) :
  total = 6 →
  select = 3 →
  ways_with_girl = 16 →
  (Nat.choose total select - Nat.choose (total - (total - (Nat.choose total select - ways_with_girl))) select = ways_with_girl) →
  total - (Nat.choose total select - ways_with_girl) = 2 := by
  sorry

end tech_group_selection_l2463_246315


namespace f_properties_l2463_246337

def f (x : ℝ) := |2*x + 1| - |x - 2|

theorem f_properties :
  (∀ x : ℝ, f x > 2 ↔ (x > 1 ∨ x < -5)) ∧
  (∀ t : ℝ, (∀ x : ℝ, f x ≥ t^2 - (11/2)*t) ↔ (1/2 ≤ t ∧ t ≤ 5)) := by
  sorry

end f_properties_l2463_246337


namespace sin_240_degrees_l2463_246341

theorem sin_240_degrees : 
  Real.sin (240 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_240_degrees_l2463_246341


namespace product_one_sum_squares_and_products_inequality_l2463_246311

theorem product_one_sum_squares_and_products_inequality 
  (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) 
  (h_prod : a * b * c * d = 1) : 
  a^2 + b^2 + c^2 + d^2 + a*b + a*c + a*d + b*c + b*d + c*d ≥ 10 := by
  sorry

end product_one_sum_squares_and_products_inequality_l2463_246311


namespace total_spent_on_souvenirs_l2463_246319

/-- The amount spent on t-shirts -/
def t_shirts : ℝ := 201

/-- The amount spent on key chains and bracelets -/
def key_chains_and_bracelets : ℝ := 347

/-- The difference between key_chains_and_bracelets and t_shirts -/
def difference : ℝ := 146

theorem total_spent_on_souvenirs :
  key_chains_and_bracelets = t_shirts + difference →
  t_shirts + key_chains_and_bracelets = 548 :=
by
  sorry

end total_spent_on_souvenirs_l2463_246319


namespace cost_of_pens_l2463_246354

/-- Given that a box of 150 pens costs $45, prove that 4500 pens cost $1350 -/
theorem cost_of_pens (box_size : ℕ) (box_cost : ℚ) (num_pens : ℕ) :
  box_size = 150 →
  box_cost = 45 →
  num_pens = 4500 →
  (num_pens : ℚ) * (box_cost / box_size) = 1350 := by
  sorry

end cost_of_pens_l2463_246354


namespace smallest_value_for_x_between_1_and_2_l2463_246361

theorem smallest_value_for_x_between_1_and_2 (x : ℝ) (h : 1 < x ∧ x < 2) :
  (1 / x < x) ∧ (1 / x < x^2) ∧ (1 / x < 2*x) ∧ (1 / x < Real.sqrt x) := by
  sorry

end smallest_value_for_x_between_1_and_2_l2463_246361


namespace local_face_value_difference_l2463_246336

def number : ℕ := 96348621

def digit_position (n : ℕ) (d : ℕ) : ℕ :=
  (n.digits 10).reverse.indexOf d

def local_value (n : ℕ) (d : ℕ) : ℕ :=
  d * (10 ^ (digit_position n d))

def face_value (d : ℕ) : ℕ := d

theorem local_face_value_difference :
  local_value number 8 - face_value 8 = 7992 := by
  sorry

end local_face_value_difference_l2463_246336


namespace line_equation_from_intersections_and_midpoint_l2463_246394

/-- The equation of line l given its intersections with two other lines and its midpoint -/
theorem line_equation_from_intersections_and_midpoint 
  (l₁ : Set (ℝ × ℝ)) 
  (l₂ : Set (ℝ × ℝ)) 
  (P : ℝ × ℝ) :
  (∀ x y, (x, y) ∈ l₁ ↔ 4 * x + y + 3 = 0) →
  (∀ x y, (x, y) ∈ l₂ ↔ 3 * x - 5 * y - 5 = 0) →
  P = (-1, 2) →
  ∃ A B : ℝ × ℝ, A ∈ l₁ ∧ B ∈ l₂ ∧ P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  ∃ l : Set (ℝ × ℝ), (∀ x y, (x, y) ∈ l ↔ 3 * x + y + 1 = 0) :=
by sorry

end line_equation_from_intersections_and_midpoint_l2463_246394


namespace linear_function_through_point_l2463_246393

def f (x : ℝ) : ℝ := x + 1

theorem linear_function_through_point :
  (∀ x y : ℝ, f (x + y) = f x + f y - f 0) ∧ f 1 = 2 := by
  sorry

end linear_function_through_point_l2463_246393


namespace twenty_three_percent_of_200_is_46_l2463_246326

theorem twenty_three_percent_of_200_is_46 : 
  ∃ x : ℝ, (23 / 100) * x = 46 ∧ x = 200 := by
  sorry

end twenty_three_percent_of_200_is_46_l2463_246326


namespace repeating_decimal_equals_fraction_l2463_246338

/-- The repeating decimal 0.363636... -/
def repeating_decimal : ℚ := 0.363636

/-- The fraction 4/11 -/
def fraction : ℚ := 4 / 11

/-- Theorem stating that the repeating decimal 0.363636... equals 4/11 -/
theorem repeating_decimal_equals_fraction : repeating_decimal = fraction := by
  sorry

end repeating_decimal_equals_fraction_l2463_246338


namespace pentagon_to_squares_ratio_is_one_eighth_l2463_246366

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a square with given side length and bottom-left corner -/
structure Square :=
  (bottomLeft : Point)
  (sideLength : ℝ)

/-- Configuration of three squares as described in the problem -/
structure SquareConfiguration :=
  (square1 : Square)
  (square2 : Square)
  (square3 : Square)

/-- The ratio of the area of pentagon PAWSR to the total area of three squares -/
def pentagonToSquaresRatio (config : SquareConfiguration) : ℝ :=
  sorry

/-- Theorem stating that the ratio is 1/8 for the given configuration -/
theorem pentagon_to_squares_ratio_is_one_eighth
  (config : SquareConfiguration)
  (h1 : config.square1.sideLength = 1)
  (h2 : config.square2.sideLength = 1)
  (h3 : config.square3.sideLength = 1)
  (h4 : config.square1.bottomLeft.x = config.square2.bottomLeft.x)
  (h5 : config.square1.bottomLeft.y + 1 = config.square2.bottomLeft.y)
  (h6 : config.square2.bottomLeft.x + 1 = config.square3.bottomLeft.x)
  (h7 : config.square2.bottomLeft.y = config.square3.bottomLeft.y) :
  pentagonToSquaresRatio config = 1/8 :=
sorry

end pentagon_to_squares_ratio_is_one_eighth_l2463_246366


namespace canoe_kayak_ratio_l2463_246383

/-- Represents the rental business scenario -/
structure RentalBusiness where
  canoe_cost : ℕ
  kayak_cost : ℕ
  total_revenue : ℕ
  canoe_count : ℕ
  kayak_count : ℕ

/-- Theorem stating the ratio of canoes to kayaks is 3:1 given the conditions -/
theorem canoe_kayak_ratio (rb : RentalBusiness) :
  rb.canoe_cost = 14 →
  rb.kayak_cost = 15 →
  rb.total_revenue = 288 →
  rb.canoe_count = rb.kayak_count + 4 →
  rb.canoe_count = 3 * rb.kayak_count →
  rb.canoe_count / rb.kayak_count = 3 := by
  sorry


end canoe_kayak_ratio_l2463_246383


namespace M_intersect_N_equals_zero_set_l2463_246388

-- Define set M
def M : Set ℝ := {-1, 0, 1}

-- Define set N
def N : Set ℝ := {y | ∃ x ∈ M, y = Real.sin x}

-- Theorem statement
theorem M_intersect_N_equals_zero_set : M ∩ N = {0} := by sorry

end M_intersect_N_equals_zero_set_l2463_246388


namespace polynomial_remainder_l2463_246365

theorem polynomial_remainder (x : ℝ) : 
  (x^11 + 1) % (x + 1) = 0 := by sorry

end polynomial_remainder_l2463_246365


namespace series_sum_equals_five_l2463_246380

theorem series_sum_equals_five (k : ℝ) (h1 : k > 1) 
  (h2 : ∑' n, (6 * n + 1) / k^n = 5) : k = 1.2 + 0.2 * Real.sqrt 6 := by
  sorry

end series_sum_equals_five_l2463_246380


namespace right_triangle_hypotenuse_l2463_246368

theorem right_triangle_hypotenuse : 
  ∀ (longer_side shorter_side hypotenuse : ℝ),
  longer_side > 0 →
  shorter_side > 0 →
  hypotenuse > 0 →
  hypotenuse = longer_side + 2 →
  shorter_side = longer_side - 7 →
  shorter_side ^ 2 + longer_side ^ 2 = hypotenuse ^ 2 →
  hypotenuse = 17 := by
sorry

end right_triangle_hypotenuse_l2463_246368


namespace complex_ratio_theorem_l2463_246308

theorem complex_ratio_theorem (z₁ z₂ z₃ : ℂ) 
  (h₁ : Complex.abs z₁ = Real.sqrt 2)
  (h₂ : Complex.abs z₂ = Real.sqrt 2)
  (h₃ : Complex.abs z₃ = Real.sqrt 2) :
  Complex.abs (1 / z₁ + 1 / z₂ + 1 / z₃) / Complex.abs (z₁ + z₂ + z₃) = 1 / 2 := by
  sorry

end complex_ratio_theorem_l2463_246308


namespace sector_central_angle_l2463_246370

/-- Given a circular sector with arc length 2 and area 4, prove that its central angle is 1/2 radians. -/
theorem sector_central_angle (arc_length : ℝ) (area : ℝ) (h1 : arc_length = 2) (h2 : area = 4) :
  let r := 2 * area / arc_length
  (arc_length / r) = 1 / 2 := by sorry

end sector_central_angle_l2463_246370


namespace y_share_l2463_246352

theorem y_share (total : ℝ) (x_share y_share z_share : ℝ) : 
  total = 210 →
  y_share = 0.45 * x_share →
  z_share = 0.30 * x_share →
  total = x_share + y_share + z_share →
  y_share = 54 := by
  sorry

end y_share_l2463_246352


namespace fractional_equation_solution_l2463_246349

theorem fractional_equation_solution :
  ∃ x : ℝ, (x + 1) / (4 * x^2 - 1) = 3 / (2 * x + 1) - 4 / (4 * x - 2) ∧ x = 6 := by
  sorry

end fractional_equation_solution_l2463_246349


namespace scientific_notation_equivalence_l2463_246367

theorem scientific_notation_equivalence : 26900000 = 2.69 * (10 ^ 7) := by
  sorry

end scientific_notation_equivalence_l2463_246367


namespace cistern_wet_surface_area_l2463_246344

/-- Calculates the total wet surface area of a rectangular cistern --/
def total_wet_surface_area (length width depth : ℝ) : ℝ :=
  let bottom_area := length * width
  let side_area1 := 2 * (length * depth)
  let side_area2 := 2 * (width * depth)
  bottom_area + side_area1 + side_area2

/-- Theorem: The total wet surface area of a cistern with given dimensions is 62 m² --/
theorem cistern_wet_surface_area :
  total_wet_surface_area 4 8 1.25 = 62 := by
  sorry

#eval total_wet_surface_area 4 8 1.25

end cistern_wet_surface_area_l2463_246344


namespace cos_11pi_3_plus_tan_neg_3pi_4_l2463_246379

theorem cos_11pi_3_plus_tan_neg_3pi_4 :
  Real.cos (11 * π / 3) + Real.tan (-3 * π / 4) = -1/2 := by
  sorry

end cos_11pi_3_plus_tan_neg_3pi_4_l2463_246379


namespace jesses_room_width_l2463_246376

/-- Proves that the width of Jesse's room is 12 feet -/
theorem jesses_room_width (length : ℝ) (tile_area : ℝ) (num_tiles : ℕ) :
  length = 2 →
  tile_area = 4 →
  num_tiles = 6 →
  (length * (tile_area * num_tiles / length : ℝ) = length * 12) :=
by
  sorry

end jesses_room_width_l2463_246376


namespace quadratic_real_root_l2463_246350

theorem quadratic_real_root (b : ℝ) :
  (∃ x : ℝ, x^2 + b*x + 25 = 0) ↔ b ∈ Set.Ici 10 ∪ Set.Iic (-10) :=
sorry

end quadratic_real_root_l2463_246350


namespace right_triangle_sides_l2463_246310

theorem right_triangle_sides : ∃ (a b c : ℕ), a = 7 ∧ b = 24 ∧ c = 25 ∧ a^2 + b^2 = c^2 := by
  sorry

end right_triangle_sides_l2463_246310


namespace inequality_solution_set_l2463_246321

theorem inequality_solution_set (x : ℝ) :
  (((1 - x) / (x + 1) ≤ 0) ∧ (x ≠ -1)) ↔ (x < -1 ∨ x ≥ 1) :=
by sorry

end inequality_solution_set_l2463_246321


namespace integer_ratio_problem_l2463_246316

theorem integer_ratio_problem (A B C D : ℕ) : 
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D →
  A > 0 ∧ B > 0 ∧ C > 0 ∧ D > 0 →
  (A + B + C + D) / 4 = 16 →
  ∃ k : ℕ, A = k * B →
  B = C - 2 →
  D = 2 →
  A / B = 28 := by
sorry

end integer_ratio_problem_l2463_246316


namespace prism_volume_with_inscribed_sphere_l2463_246385

/-- The volume of a regular triangular prism with an inscribed sphere -/
theorem prism_volume_with_inscribed_sphere (r : ℝ) (h : r > 0) :
  let sphere_volume : ℝ := (4 / 3) * Real.pi * r^3
  let prism_side : ℝ := 2 * Real.sqrt 3 * r
  let prism_height : ℝ := 2 * r
  let prism_volume : ℝ := (Real.sqrt 3 / 4) * prism_side^2 * prism_height
  sphere_volume = 36 * Real.pi → prism_volume = 162 * Real.sqrt 3 := by
sorry

end prism_volume_with_inscribed_sphere_l2463_246385


namespace existence_of_special_set_l2463_246381

theorem existence_of_special_set : ∃ (A : Set ℕ), 
  ∀ (S : Set ℕ), (∀ p ∈ S, Nat.Prime p) → (Set.Infinite S) →
    ∃ (k : ℕ) (m n : ℕ),
      k ≥ 2 ∧
      m ∈ A ∧
      n ∉ A ∧
      (∃ (factors_m factors_n : Finset ℕ),
        factors_m.card = k ∧
        factors_n.card = k ∧
        (∀ p ∈ factors_m, p ∈ S) ∧
        (∀ p ∈ factors_n, p ∈ S) ∧
        (Finset.prod factors_m id = m) ∧
        (Finset.prod factors_n id = n)) :=
by sorry

end existence_of_special_set_l2463_246381


namespace factorial_division_l2463_246387

theorem factorial_division (h : Nat.factorial 9 = 362880) :
  Nat.factorial 9 / Nat.factorial 4 = 15120 := by
  sorry

end factorial_division_l2463_246387


namespace square_root_squared_sqrt_987654_squared_l2463_246362

theorem square_root_squared (n : ℝ) (hn : 0 ≤ n) : (Real.sqrt n) ^ 2 = n := by sorry

theorem sqrt_987654_squared : (Real.sqrt 987654) ^ 2 = 987654 := by sorry

end square_root_squared_sqrt_987654_squared_l2463_246362


namespace set_operations_l2463_246364

open Set

def U : Set ℝ := univ

def A : Set ℝ := {x | 1 ≤ x - 1 ∧ x - 1 < 3}

def B : Set ℝ := {x | 2*x - 9 ≥ 6 - 3*x}

theorem set_operations :
  (A ∪ B = {x | x ≥ 2}) ∧
  (Aᶜ ∩ Bᶜ = {x | x < 3 ∨ x ≥ 4}) :=
by sorry

end set_operations_l2463_246364


namespace charity_donation_percentage_l2463_246356

theorem charity_donation_percentage 
  (total_raised : ℝ)
  (num_organizations : ℕ)
  (amount_per_org : ℝ)
  (h1 : total_raised = 2500)
  (h2 : num_organizations = 8)
  (h3 : amount_per_org = 250) :
  (num_organizations * amount_per_org) / total_raised * 100 = 80 := by
sorry

end charity_donation_percentage_l2463_246356


namespace recipe_calculation_l2463_246333

/-- The amount of flour Julia uses in mL -/
def flour_amount : ℕ := 800

/-- The base amount of flour in mL for the recipe ratio -/
def base_flour : ℕ := 200

/-- The amount of milk in mL needed for the base amount of flour -/
def milk_per_base : ℕ := 60

/-- The number of eggs needed for the base amount of flour -/
def eggs_per_base : ℕ := 1

/-- The amount of milk needed for Julia's recipe -/
def milk_needed : ℕ := (flour_amount / base_flour) * milk_per_base

/-- The number of eggs needed for Julia's recipe -/
def eggs_needed : ℕ := (flour_amount / base_flour) * eggs_per_base

theorem recipe_calculation : 
  milk_needed = 240 ∧ eggs_needed = 4 := by sorry

end recipe_calculation_l2463_246333


namespace total_packs_is_108_l2463_246313

/-- The number of people buying baseball cards -/
def num_people : ℕ := 4

/-- The number of baseball cards each person bought -/
def cards_per_person : ℕ := 540

/-- The number of cards in each pack -/
def cards_per_pack : ℕ := 20

/-- Theorem: The total number of packs for all people is 108 -/
theorem total_packs_is_108 : 
  (num_people * cards_per_person) / cards_per_pack = 108 := by
  sorry

end total_packs_is_108_l2463_246313


namespace garden_perimeter_l2463_246351

/-- The perimeter of a rectangle given its length and breadth -/
def rectangle_perimeter (length : ℝ) (breadth : ℝ) : ℝ :=
  2 * (length + breadth)

/-- Theorem: The perimeter of a rectangular garden with length 140 m and breadth 100 m is 480 m -/
theorem garden_perimeter :
  rectangle_perimeter 140 100 = 480 := by
  sorry

end garden_perimeter_l2463_246351


namespace range_of_a_correct_l2463_246375

/-- Proposition p: The solution set of a^x > 1 (a > 0 and a ≠ 1) is {x | x < 0} -/
def p (a : ℝ) : Prop :=
  0 < a ∧ a ≠ 1 ∧ ∀ x, a^x > 1 ↔ x < 0

/-- Proposition q: The domain of y = log(x^2 - x + a) is ℝ -/
def q (a : ℝ) : Prop :=
  ∀ x, x^2 - x + a > 0

/-- The range of a satisfying the given conditions -/
def range_of_a : Set ℝ :=
  {a | (0 < a ∧ a ≤ 1/4) ∨ a ≥ 1}

theorem range_of_a_correct :
  ∀ a, (p a ∨ q a) ∧ ¬(p a ∧ q a) ↔ a ∈ range_of_a := by sorry

end range_of_a_correct_l2463_246375


namespace frog_jump_probability_l2463_246369

/-- Represents a point on a 2D grid -/
structure Point where
  x : ℕ
  y : ℕ

/-- Represents the square boundary -/
def square_boundary (p : Point) : Bool :=
  p.x = 0 ∨ p.x = 4 ∨ p.y = 0 ∨ p.y = 4

/-- Represents reaching a vertical side of the square -/
def vertical_side (p : Point) : Bool :=
  p.x = 0 ∨ p.x = 4

/-- Probability of ending on a vertical side when starting from a given point -/
noncomputable def prob_vertical_end (start : Point) : ℚ :=
  sorry

/-- The main theorem to prove -/
theorem frog_jump_probability :
  prob_vertical_end ⟨1, 2⟩ = 5/8 := by sorry

end frog_jump_probability_l2463_246369


namespace new_figure_has_five_sides_l2463_246396

/-- A regular polygon with n sides and side length 1 -/
structure RegularPolygon where
  sides : ℕ
  sideLength : ℝ
  sideLength_eq_one : sideLength = 1

/-- The new figure formed by connecting a hexagon and triangle -/
def NewFigure (hexagon triangle : RegularPolygon) : ℕ :=
  hexagon.sides + triangle.sides - 2

/-- Theorem stating that the new figure has 5 sides -/
theorem new_figure_has_five_sides
  (hexagon : RegularPolygon)
  (triangle : RegularPolygon)
  (hexagon_is_hexagon : hexagon.sides = 6)
  (triangle_is_triangle : triangle.sides = 3) :
  NewFigure hexagon triangle = 5 := by
  sorry

#eval NewFigure ⟨6, 1, rfl⟩ ⟨3, 1, rfl⟩

end new_figure_has_five_sides_l2463_246396


namespace total_amount_is_fifteen_l2463_246305

/-- Represents the share distribution among three people -/
structure ShareDistribution where
  first : ℝ
  second : ℝ
  third : ℝ

/-- Calculates the total amount given a share distribution -/
def totalAmount (s : ShareDistribution) : ℝ :=
  s.first + s.second + s.third

/-- Theorem: Given the specified share distribution and the first person's share, 
    the total amount is 15 rupees -/
theorem total_amount_is_fifteen 
  (s : ShareDistribution) 
  (h1 : s.first = 10)
  (h2 : s.second = 0.3 * s.first)
  (h3 : s.third = 0.2 * s.first) : 
  totalAmount s = 15 := by
  sorry

end total_amount_is_fifteen_l2463_246305


namespace repeating_decimal_as_fraction_l2463_246389

def repeating_decimal : ℚ := 2 + 35 / 99

theorem repeating_decimal_as_fraction :
  repeating_decimal = 233 / 99 := by sorry

end repeating_decimal_as_fraction_l2463_246389


namespace broken_marbles_percentage_l2463_246339

theorem broken_marbles_percentage (total_broken : ℕ) (set1_count : ℕ) (set2_count : ℕ) (set2_broken_percent : ℚ) :
  total_broken = 17 →
  set1_count = 50 →
  set2_count = 60 →
  set2_broken_percent = 20 / 100 →
  ∃ (set1_broken_percent : ℚ),
    set1_broken_percent = 10 / 100 ∧
    total_broken = set1_broken_percent * set1_count + set2_broken_percent * set2_count :=
by sorry

end broken_marbles_percentage_l2463_246339


namespace min_value_expression_l2463_246304

theorem min_value_expression (x y : ℝ) (hx : x > 1) (hy : y > 1) :
  (x^4 / (y - 1)) + (y^4 / (x - 1)) ≥ 12 ∧
  ((x^4 / (y - 1)) + (y^4 / (x - 1)) = 12 ↔ x = 2 ∧ y = 2) :=
by sorry

end min_value_expression_l2463_246304


namespace leapYears105_l2463_246330

/-- Calculates the maximum number of leap years in a given period under a system where
    leap years occur every 4 years and every 5th year. -/
def maxLeapYears (period : ℕ) : ℕ :=
  (period / 4) + (period / 5) - (period / 20)

/-- Theorem stating that in a 105-year period, the maximum number of leap years is 42
    under the given leap year system. -/
theorem leapYears105 : maxLeapYears 105 = 42 := by
  sorry

#eval maxLeapYears 105  -- Should output 42

end leapYears105_l2463_246330


namespace jake_ball_count_l2463_246309

/-- The number of balls each person has -/
structure BallCount where
  jake : ℕ
  audrey : ℕ
  charlie : ℕ

/-- The conditions of the problem -/
def problem_conditions (bc : BallCount) : Prop :=
  bc.audrey = bc.jake + 34 ∧
  bc.audrey = 2 * bc.charlie ∧
  bc.charlie + 7 = 41

/-- The theorem to be proved -/
theorem jake_ball_count (bc : BallCount) : 
  problem_conditions bc → bc.jake = 62 := by
  sorry

end jake_ball_count_l2463_246309


namespace daylight_duration_l2463_246353

/-- Given a day with 24 hours and a daylight to nighttime ratio of 9:7, 
    the duration of daylight is 13.5 hours. -/
theorem daylight_duration (total_hours : ℝ) (daylight_ratio nighttime_ratio : ℕ) 
    (h1 : total_hours = 24)
    (h2 : daylight_ratio = 9)
    (h3 : nighttime_ratio = 7) :
  (daylight_ratio : ℝ) / (daylight_ratio + nighttime_ratio : ℝ) * total_hours = 13.5 := by
sorry

end daylight_duration_l2463_246353


namespace sports_only_count_l2463_246359

theorem sports_only_count (total employees : ℕ) (sports_fans : ℕ) (art_fans : ℕ) (neither_fans : ℕ) :
  total = 60 →
  sports_fans = 28 →
  art_fans = 26 →
  neither_fans = 12 →
  sports_fans - (total - neither_fans - art_fans) = 22 :=
by
  sorry

end sports_only_count_l2463_246359


namespace xy_positive_sufficient_not_necessary_for_abs_sum_equality_l2463_246346

theorem xy_positive_sufficient_not_necessary_for_abs_sum_equality (x y : ℝ) :
  (∀ x y : ℝ, x * y > 0 → |x + y| = |x| + |y|) ∧
  (∃ x y : ℝ, |x + y| = |x| + |y| ∧ ¬(x * y > 0)) :=
sorry

end xy_positive_sufficient_not_necessary_for_abs_sum_equality_l2463_246346


namespace sin_pi_6_plus_cos_pi_3_simplification_l2463_246373

theorem sin_pi_6_plus_cos_pi_3_simplification (α : ℝ) : 
  Real.sin (π/6 + α) + Real.cos (π/3 + α) = Real.cos α := by
  sorry

end sin_pi_6_plus_cos_pi_3_simplification_l2463_246373


namespace prob_each_student_gets_book_l2463_246329

/-- The number of students --/
def num_students : ℕ := 4

/-- The number of books --/
def num_books : ℕ := 5

/-- The total number of possible distributions --/
def total_distributions : ℕ := num_students ^ num_books

/-- The number of valid distributions where each student gets at least one book --/
def valid_distributions : ℕ := 
  num_students ^ num_books - 
  num_students * (num_students - 1) ^ num_books + 
  (num_students.choose 2) * (num_students - 2) ^ num_books - 
  num_students

/-- The probability that each student receives at least one book --/
theorem prob_each_student_gets_book : 
  (valid_distributions : ℚ) / total_distributions = 15 / 64 := by
  sorry

end prob_each_student_gets_book_l2463_246329


namespace initial_amount_calculation_l2463_246360

/-- Represents the simple interest scenario --/
structure SimpleInterest where
  initialAmount : ℝ
  rate : ℝ
  time : ℝ
  finalAmount : ℝ

/-- The simple interest calculation is correct --/
def isValidSimpleInterest (si : SimpleInterest) : Prop :=
  si.finalAmount = si.initialAmount * (1 + si.rate * si.time / 100)

/-- Theorem stating the initial amount given the conditions --/
theorem initial_amount_calculation (si : SimpleInterest) 
  (h1 : si.finalAmount = 1050)
  (h2 : si.rate = 8)
  (h3 : si.time = 5)
  (h4 : isValidSimpleInterest si) : 
  si.initialAmount = 750 := by
  sorry

#check initial_amount_calculation

end initial_amount_calculation_l2463_246360


namespace max_im_part_is_sin_90_deg_l2463_246347

-- Define the polynomial
def p (z : ℂ) : ℂ := z^6 - z^4 + z^3 - z + 1

-- Define the set of roots
def roots : Set ℂ := {z : ℂ | p z = 0}

-- Define the imaginary part function
def imPart (z : ℂ) : ℝ := z.im

-- Define the theorem
theorem max_im_part_is_sin_90_deg :
  ∃ (z : ℂ), z ∈ roots ∧ 
  (∀ (w : ℂ), w ∈ roots → imPart w ≤ imPart z) ∧
  imPart z = Real.sin (π / 2) :=
sorry

end max_im_part_is_sin_90_deg_l2463_246347


namespace P_120_l2463_246371

/-- 
P(n) represents the number of ways to express a positive integer n 
as a product of integers greater than 1, where the order matters.
-/
def P (n : ℕ) : ℕ := sorry

/-- The prime factorization of 120 -/
def primeFactors120 : List ℕ := [2, 2, 2, 3, 5]

/-- 120 is the product of its prime factors -/
axiom is120 : (primeFactors120.prod = 120)

/-- All elements in primeFactors120 are prime numbers -/
axiom allPrime : ∀ p ∈ primeFactors120, Nat.Prime p

theorem P_120 : P 120 = 29 := by sorry

end P_120_l2463_246371


namespace sum_of_digits_oneOver99Squared_l2463_246332

/-- Represents a repeating decimal expansion -/
structure RepeatingDecimal where
  digits : List Nat
  period : Nat

/-- The repeating decimal expansion of 1/(99^2) -/
def oneOver99Squared : RepeatingDecimal :=
  { digits := sorry
    period := sorry }

/-- The sum of digits in one period of the repeating decimal expansion of 1/(99^2) -/
def sumOfDigits (rd : RepeatingDecimal) : Nat :=
  (rd.digits.take rd.period).sum

theorem sum_of_digits_oneOver99Squared :
  sumOfDigits oneOver99Squared = 883 := by
  sorry

end sum_of_digits_oneOver99Squared_l2463_246332


namespace max_side_length_24_perimeter_l2463_246302

/-- Represents a triangle with integer side lengths -/
structure Triangle where
  a : ℕ
  b : ℕ
  c : ℕ
  different_sides : a ≠ b ∧ b ≠ c ∧ a ≠ c
  triangle_inequality : a < b + c ∧ b < a + c ∧ c < a + b
  perimeter_24 : a + b + c = 24

/-- The maximum length of any side in a triangle with perimeter 24 and different integer side lengths is 12 -/
theorem max_side_length_24_perimeter (t : Triangle) : t.a ≤ 12 ∧ t.b ≤ 12 ∧ t.c ≤ 12 :=
sorry

end max_side_length_24_perimeter_l2463_246302


namespace tommy_nickels_l2463_246300

/-- Represents Tommy's coin collection --/
structure CoinCollection where
  quarters : ℕ
  dimes : ℕ
  nickels : ℕ
  pennies : ℕ

/-- Conditions for Tommy's coin collection --/
def tommy_collection (c : CoinCollection) : Prop :=
  c.quarters = 4 ∧
  c.dimes = c.pennies + 10 ∧
  c.nickels = 2 * c.dimes ∧
  c.pennies = 10 * c.quarters

theorem tommy_nickels (c : CoinCollection) : 
  tommy_collection c → c.nickels = 100 := by
  sorry

end tommy_nickels_l2463_246300


namespace line_perp_plane_from_conditions_l2463_246331

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between planes and lines
variable (perp_plane_line : Plane → Line → Prop)

-- Define the perpendicular relation between two planes
variable (perp_plane_plane : Plane → Plane → Prop)

-- State the theorem
theorem line_perp_plane_from_conditions 
  (α β : Plane) (m n : Line) 
  (h1 : perp_plane_line α n) 
  (h2 : perp_plane_line β n) 
  (h3 : perp_plane_line α m) : 
  perp_plane_line β m :=
sorry

end line_perp_plane_from_conditions_l2463_246331


namespace blocks_given_by_theresa_l2463_246390

theorem blocks_given_by_theresa (initial_blocks final_blocks : ℕ) 
  (h1 : initial_blocks = 4)
  (h2 : final_blocks = 83) :
  final_blocks - initial_blocks = 79 := by
  sorry

end blocks_given_by_theresa_l2463_246390


namespace expression_value_l2463_246345

theorem expression_value (a b c : ℝ) 
  (sum_eq : a + b + c = 3) 
  (sum_squares_eq : a^2 + b^2 + c^2 = 4) :
  (a^2 + b^2) / (2 - c) + (b^2 + c^2) / (2 - a) + (c^2 + a^2) / (2 - b) = 9 := by
  sorry

end expression_value_l2463_246345


namespace v_closed_under_mult_and_div_v_not_closed_under_addition_v_not_closed_under_negative_powers_l2463_246328

-- Define the set v as cubes of positive integers
def v : Set ℕ := {n : ℕ | ∃ k : ℕ+, n = k^3}

-- Theorem stating that v is closed under multiplication and division
theorem v_closed_under_mult_and_div :
  (∀ a b : ℕ, a ∈ v → b ∈ v → (a * b) ∈ v) ∧
  (∀ a b : ℕ, a ∈ v → b ∈ v → b ≠ 0 → (a / b) ∈ v) :=
sorry

-- Theorem stating that v is not closed under addition
theorem v_not_closed_under_addition :
  ∃ a b : ℕ, a ∈ v ∧ b ∈ v ∧ (a + b) ∉ v :=
sorry

-- Theorem stating that v is not closed under negative powers
theorem v_not_closed_under_negative_powers :
  ∃ a : ℕ, a ∈ v ∧ a ≠ 0 ∧ (1 / a) ∉ v :=
sorry

end v_closed_under_mult_and_div_v_not_closed_under_addition_v_not_closed_under_negative_powers_l2463_246328


namespace inequality_proof_l2463_246355

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^3 + 1/b^3 - 1) * (b^3 + 1/c^3 - 1) * (c^3 + 1/a^3 - 1) ≤ (a*b*c + 1/(a*b*c) - 1)^3 :=
by sorry

end inequality_proof_l2463_246355


namespace quadratic_real_roots_condition_l2463_246314

theorem quadratic_real_roots_condition (k : ℝ) :
  (∃ x : ℝ, k * x^2 + 2 * x + 1 = 0) → k ≤ 1 := by
  sorry

end quadratic_real_roots_condition_l2463_246314


namespace average_mile_time_l2463_246324

theorem average_mile_time (mile1 mile2 mile3 mile4 : ℕ) 
  (h1 : mile1 = 6)
  (h2 : mile2 = 5)
  (h3 : mile3 = 5)
  (h4 : mile4 = 4) :
  (mile1 + mile2 + mile3 + mile4) / 4 = 5 := by
  sorry

end average_mile_time_l2463_246324


namespace order_relationship_l2463_246325

theorem order_relationship (a b c d : ℝ) 
  (h1 : a < b) 
  (h2 : c < d) 
  (h3 : a + b < c + d) 
  (h4 : a * b = c * d) 
  (h5 : c * d < 0) : 
  a < c ∧ c < b ∧ b < d := by
sorry

end order_relationship_l2463_246325


namespace positive_solution_to_equation_l2463_246320

theorem positive_solution_to_equation (x : ℝ) :
  x > 0 ∧ x + 17 = 60 * (1 / x) → x = 3 := by
  sorry

end positive_solution_to_equation_l2463_246320


namespace stair_climbing_comparison_l2463_246382

/-- Given two people climbing stairs at different speeds, this theorem calculates
    how many steps the faster person climbs when the slower person reaches a certain height. -/
theorem stair_climbing_comparison
  (matt_speed : ℕ)  -- Matt's speed in steps per minute
  (tom_speed_diff : ℕ)  -- How many more steps per minute Tom climbs compared to Matt
  (matt_steps : ℕ)  -- Number of steps Matt has climbed
  (matt_speed_pos : 0 < matt_speed)  -- Matt's speed is positive
  (h_matt_speed : matt_speed = 20)  -- Matt's actual speed
  (h_tom_speed_diff : tom_speed_diff = 5)  -- Tom's speed difference
  (h_matt_steps : matt_steps = 220)  -- Steps Matt has climbed
  : (matt_steps + (matt_steps / matt_speed) * tom_speed_diff : ℕ) = 275 := by
  sorry

end stair_climbing_comparison_l2463_246382


namespace largest_power_of_seven_divisor_l2463_246340

theorem largest_power_of_seven_divisor : ∃ (n : ℕ), 
  (∀ (k : ℕ), 7^k ∣ (Nat.factorial 200 / (Nat.factorial 90 * Nat.factorial 30)) → k ≤ n) ∧
  (7^n ∣ (Nat.factorial 200 / (Nat.factorial 90 * Nat.factorial 30))) ∧
  n = 15 := by
  sorry

end largest_power_of_seven_divisor_l2463_246340


namespace count_numbers_with_three_l2463_246391

/-- Count of numbers from 1 to 800 without digit 3 -/
def count_without_three : ℕ := 729

/-- Count of numbers from 1 to 800 with at least one digit 3 -/
def count_with_three : ℕ := 800 - count_without_three

theorem count_numbers_with_three : count_with_three = 71 := by
  sorry

end count_numbers_with_three_l2463_246391


namespace gcd_of_three_numbers_l2463_246317

theorem gcd_of_three_numbers :
  Nat.gcd 105 (Nat.gcd 1001 2436) = 7 := by
  sorry

end gcd_of_three_numbers_l2463_246317


namespace distance_rode_bus_l2463_246334

/-- The distance Craig walked from the bus stop to home, in miles -/
def distance_walked : ℝ := 0.17

/-- The difference between the distance Craig rode the bus and the distance he walked, in miles -/
def distance_difference : ℝ := 3.67

/-- Theorem: The distance Craig rode the bus is 3.84 miles -/
theorem distance_rode_bus : distance_walked + distance_difference = 3.84 := by
  sorry

end distance_rode_bus_l2463_246334


namespace smallest_quotient_l2463_246357

def digit_sum_of_squares (n : ℕ) : ℕ :=
  if n < 10 then n * n else (n % 10) * (n % 10) + digit_sum_of_squares (n / 10)

theorem smallest_quotient (n : ℕ) (h : n > 0) :
  (n : ℚ) / (digit_sum_of_squares n) ≥ 1 / 9 ∧ ∃ m : ℕ, m > 0 ∧ (m : ℚ) / (digit_sum_of_squares m) = 1 / 9 :=
sorry

end smallest_quotient_l2463_246357


namespace union_A_B_complement_intersection_A_B_l2463_246372

-- Define the universal set U as ℝ
def U := Set ℝ

-- Define set A
def A : Set ℝ := {x | 1 ≤ x - 1 ∧ x - 1 < 3}

-- Define set B
def B : Set ℝ := {x | 2*x - 9 ≥ 6 - 3*x}

-- Theorem for A ∪ B
theorem union_A_B : A ∪ B = {x : ℝ | x ≥ 2} := by sorry

-- Theorem for ∁ᵤ(A ∩ B)
theorem complement_intersection_A_B : (A ∩ B)ᶜ = {x : ℝ | x < 3 ∨ x ≥ 4} := by sorry

end union_A_B_complement_intersection_A_B_l2463_246372


namespace exist_consecutive_sum_digits_div_13_l2463_246374

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

/-- Theorem: There exist two consecutive natural numbers such that
    the sum of the digits of each of them is divisible by 13 -/
theorem exist_consecutive_sum_digits_div_13 :
  ∃ n : ℕ, 13 ∣ sum_of_digits n ∧ 13 ∣ sum_of_digits (n + 1) :=
by
  sorry

end exist_consecutive_sum_digits_div_13_l2463_246374


namespace bucket_capacity_reduction_l2463_246303

theorem bucket_capacity_reduction (original_buckets reduced_buckets : ℚ) 
  (h1 : original_buckets = 25)
  (h2 : reduced_buckets = 62.5)
  (h3 : original_buckets * original_capacity = reduced_buckets * reduced_capacity) :
  reduced_capacity / original_capacity = 2 / 5 := by
  sorry

end bucket_capacity_reduction_l2463_246303


namespace min_values_ab_and_a_plus_2b_l2463_246363

theorem min_values_ab_and_a_plus_2b (a b : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : 1 / a + 2 / b = 1) : 
  (∀ x y : ℝ, x > 0 ∧ y > 0 ∧ 1 / x + 2 / y = 1 → x * y ≥ 8) ∧
  (∀ x y : ℝ, x > 0 ∧ y > 0 ∧ 1 / x + 2 / y = 1 → x + 2 * y ≥ 9) :=
by sorry

#check min_values_ab_and_a_plus_2b

end min_values_ab_and_a_plus_2b_l2463_246363


namespace frame_254_width_2_l2463_246306

/-- The number of cells in a square frame with given side length and width -/
def frame_cells (side_length : ℕ) (width : ℕ) : ℕ :=
  side_length ^ 2 - (side_length - 2 * width) ^ 2

/-- Theorem: A 254 × 254 frame with width 2 has 2016 cells -/
theorem frame_254_width_2 :
  frame_cells 254 2 = 2016 := by
  sorry

end frame_254_width_2_l2463_246306


namespace quadratic_integer_conjugate_theorem_l2463_246342

/-- A structure representing a quadratic integer of the form a + b√d -/
structure QuadraticInteger (d : ℕ) where
  a : ℤ
  b : ℤ

/-- The conjugate of a quadratic integer -/
def conjugate {d : ℕ} (z : QuadraticInteger d) : QuadraticInteger d :=
  ⟨z.a, -z.b⟩

theorem quadratic_integer_conjugate_theorem
  (d : ℕ) (x₀ y₀ x y X Y : ℤ) (r : ℕ) 
  (h_d : ¬ ∃ (n : ℕ), n ^ 2 = d)
  (h_pos : x₀ > 0 ∧ y₀ > 0 ∧ x > 0 ∧ y > 0)
  (h_eq : X + Y * d^(1/2) = (x + y * d^(1/2)) * (x₀ - y₀ * d^(1/2))^r) :
  X - Y * d^(1/2) = (x - y * d^(1/2)) * (x₀ + y₀ * d^(1/2))^r := by
  sorry

end quadratic_integer_conjugate_theorem_l2463_246342


namespace a_gt_b_necessary_not_sufficient_l2463_246327

theorem a_gt_b_necessary_not_sufficient (a b c : ℝ) :
  (∀ a b c : ℝ, a * c^2 > b * c^2 → a > b) ∧
  (∃ a b c : ℝ, a > b ∧ ¬(a * c^2 > b * c^2)) :=
by sorry

end a_gt_b_necessary_not_sufficient_l2463_246327


namespace least_number_divisible_by_five_primes_l2463_246384

theorem least_number_divisible_by_five_primes : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∃ (p₁ p₂ p₃ p₄ p₅ : ℕ), Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ ∧ Prime p₅ ∧ 
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₁ ≠ p₅ ∧ 
    p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₂ ≠ p₅ ∧ 
    p₃ ≠ p₄ ∧ p₃ ≠ p₅ ∧ 
    p₄ ≠ p₅ ∧ 
    n % p₁ = 0 ∧ n % p₂ = 0 ∧ n % p₃ = 0 ∧ n % p₄ = 0 ∧ n % p₅ = 0) ∧
  (∀ m : ℕ, m > 0 → 
    (∃ (q₁ q₂ q₃ q₄ q₅ : ℕ), Prime q₁ ∧ Prime q₂ ∧ Prime q₃ ∧ Prime q₄ ∧ Prime q₅ ∧ 
      q₁ ≠ q₂ ∧ q₁ ≠ q₃ ∧ q₁ ≠ q₄ ∧ q₁ ≠ q₅ ∧ 
      q₂ ≠ q₃ ∧ q₂ ≠ q₄ ∧ q₂ ≠ q₅ ∧ 
      q₃ ≠ q₄ ∧ q₃ ≠ q₅ ∧ 
      q₄ ≠ q₅ ∧ 
      m % q₁ = 0 ∧ m % q₂ = 0 ∧ m % q₃ = 0 ∧ m % q₄ = 0 ∧ m % q₅ = 0) → 
    m ≥ n) ∧
  n = 2310 := by
  sorry

#check least_number_divisible_by_five_primes

end least_number_divisible_by_five_primes_l2463_246384


namespace outfit_combinations_l2463_246323

theorem outfit_combinations (tshirts pants hats : ℕ) 
  (h1 : tshirts = 8) 
  (h2 : pants = 6) 
  (h3 : hats = 3) : 
  tshirts * pants * hats = 144 := by
sorry

end outfit_combinations_l2463_246323


namespace jackson_school_supplies_cost_l2463_246378

/-- Calculates the total cost of school supplies for a class, given the number of students,
    item quantities per student, item costs, and a teacher discount. -/
def totalCostOfSupplies (students : ℕ) 
                        (penPerStudent notebookPerStudent binderPerStudent highlighterPerStudent : ℕ)
                        (penCost notebookCost binderCost highlighterCost : ℚ)
                        (teacherDiscount : ℚ) : ℚ :=
  let totalPens := students * penPerStudent
  let totalNotebooks := students * notebookPerStudent
  let totalBinders := students * binderPerStudent
  let totalHighlighters := students * highlighterPerStudent
  let totalCost := totalPens * penCost + totalNotebooks * notebookCost + 
                   totalBinders * binderCost + totalHighlighters * highlighterCost
  totalCost - teacherDiscount

/-- Theorem stating that the total cost of school supplies for Jackson's class is $858.25 -/
theorem jackson_school_supplies_cost : 
  totalCostOfSupplies 45 6 4 2 3 (65/100) (145/100) (480/100) (85/100) 125 = 85825/100 := by
  sorry

end jackson_school_supplies_cost_l2463_246378


namespace complex_expression_simplification_l2463_246399

theorem complex_expression_simplification (p q : ℝ) (hp : p > 0) (hpq : p > q) :
  let numerator := Real.sqrt ((p^4 + q^4) / (p^4 - p^2 * q^2) + 2 * q^2 / (p^2 - q^2) * (p^3 - p * q^2)) - 2 * q * Real.sqrt p
  let denominator := Real.sqrt (p / (p - q) - q / (p + q) - 2 * p * q / (p^2 - q^2)) * (p - q)
  numerator / denominator = Real.sqrt (p^2 - q^2) / Real.sqrt p :=
by sorry

end complex_expression_simplification_l2463_246399

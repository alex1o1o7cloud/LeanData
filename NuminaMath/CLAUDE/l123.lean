import Mathlib

namespace hat_cost_l123_12397

/-- Given a sale of clothes where shirts cost $5 each, jeans cost $10 per pair,
    and the total cost for 3 shirts, 2 pairs of jeans, and 4 hats is $51,
    prove that each hat costs $4. -/
theorem hat_cost (shirt_cost jeans_cost total_cost : ℕ) (hat_cost : ℕ) :
  shirt_cost = 5 →
  jeans_cost = 10 →
  total_cost = 51 →
  3 * shirt_cost + 2 * jeans_cost + 4 * hat_cost = total_cost →
  hat_cost = 4 := by
  sorry

end hat_cost_l123_12397


namespace parabola_vertex_problem_parabola_vertex_l123_12318

/-- The coordinates of the vertex of a parabola in the form y = a(x-h)^2 + k are (h,k) -/
theorem parabola_vertex (a h k : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ a * (x - h)^2 + k
  (∀ x, f x ≥ f h) ∧ f h = k := by sorry

/-- The coordinates of the vertex of the parabola y = 3(x-7)^2 + 5 are (7,5) -/
theorem problem_parabola_vertex : 
  let f : ℝ → ℝ := λ x ↦ 3 * (x - 7)^2 + 5
  (∀ x, f x ≥ f 7) ∧ f 7 = 5 := by sorry

end parabola_vertex_problem_parabola_vertex_l123_12318


namespace multiple_of_nine_between_15_and_30_l123_12357

theorem multiple_of_nine_between_15_and_30 (x : ℕ) 
  (h1 : ∃ k : ℕ, x = 9 * k)
  (h2 : x^2 > 225)
  (h3 : x < 30) :
  x = 18 ∨ x = 27 := by
sorry

end multiple_of_nine_between_15_and_30_l123_12357


namespace point_coordinates_wrt_origin_l123_12338

/-- 
In a Cartesian coordinate system, the coordinates of a point (2, -3) 
with respect to the origin are (2, -3).
-/
theorem point_coordinates_wrt_origin : 
  let point : ℝ × ℝ := (2, -3)
  point = (2, -3) :=
by sorry

end point_coordinates_wrt_origin_l123_12338


namespace g_evaluation_l123_12340

def g (x : ℝ) : ℝ := 3 * x^2 - 5 * x + 7

theorem g_evaluation : 3 * g 2 + 4 * g (-4) = 327 := by
  sorry

end g_evaluation_l123_12340


namespace isosceles_triangle_circumradius_l123_12372

/-- The radius of a circle circumscribing an isosceles triangle -/
theorem isosceles_triangle_circumradius (a b c : ℝ) (h_isosceles : a = b) (h_sides : a = 13 ∧ c = 10) :
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  (a * b * c) / (4 * area) = 169 / 24 := by
  sorry

end isosceles_triangle_circumradius_l123_12372


namespace special_ellipse_major_axis_length_l123_12342

/-- An ellipse with specific properties -/
structure SpecialEllipse where
  /-- The ellipse is tangent to the line y = 1 -/
  tangent_to_y1 : Bool
  /-- The ellipse is tangent to the y-axis -/
  tangent_to_yaxis : Bool
  /-- The first focus of the ellipse -/
  focus1 : ℝ × ℝ
  /-- The second focus of the ellipse -/
  focus2 : ℝ × ℝ

/-- The length of the major axis of the special ellipse -/
def majorAxisLength (e : SpecialEllipse) : ℝ := sorry

/-- Theorem stating that the length of the major axis is 2 for the given ellipse -/
theorem special_ellipse_major_axis_length :
  ∀ (e : SpecialEllipse),
    e.tangent_to_y1 = true →
    e.tangent_to_yaxis = true →
    e.focus1 = (3, 2 + Real.sqrt 2) →
    e.focus2 = (3, 2 - Real.sqrt 2) →
    majorAxisLength e = 2 := by sorry

end special_ellipse_major_axis_length_l123_12342


namespace expression_evaluation_l123_12335

theorem expression_evaluation (m n : ℤ) (hm : m = 1) (hn : n = -2) :
  -2 * (m * n - 3 * m^2) - (2 * m * n - 5 * (m * n - m^2)) = -1 := by
  sorry

end expression_evaluation_l123_12335


namespace factorial_divisibility_l123_12387

theorem factorial_divisibility (m n : ℕ) : 
  ∃ k : ℕ, (Nat.factorial (2*m) * Nat.factorial (2*n)) = 
    k * (Nat.factorial m * Nat.factorial n * Nat.factorial (m+n)) :=
sorry

end factorial_divisibility_l123_12387


namespace base7_305_eq_base5_1102_l123_12349

/-- Converts a base-7 number to its decimal (base-10) representation -/
def base7ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- Converts a decimal (base-10) number to its base-5 representation -/
def decimalToBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec go (m : Nat) (acc : List Nat) :=
      if m = 0 then acc
      else go (m / 5) ((m % 5) :: acc)
    go n []

/-- States that the base-7 number 305 is equal to the base-5 number 1102 -/
theorem base7_305_eq_base5_1102 :
  decimalToBase5 (base7ToDecimal [5, 0, 3]) = [1, 1, 0, 2] := by
  sorry

#eval base7ToDecimal [5, 0, 3]
#eval decimalToBase5 152

end base7_305_eq_base5_1102_l123_12349


namespace absolute_value_of_h_l123_12358

theorem absolute_value_of_h (h : ℝ) : 
  (∃ x y : ℝ, x^2 + 2*h*x = 8 ∧ y^2 + 2*h*y = 8 ∧ x^2 + y^2 = 18) → 
  |h| = Real.sqrt 2 / 2 := by
sorry

end absolute_value_of_h_l123_12358


namespace arithmetic_progression_first_term_l123_12306

theorem arithmetic_progression_first_term
  (d : ℝ)
  (a₁₂ : ℝ)
  (h₁ : d = 8)
  (h₂ : a₁₂ = 90)
  : ∃ (a₁ : ℝ), a₁₂ = a₁ + (12 - 1) * d ∧ a₁ = 2 := by
  sorry

end arithmetic_progression_first_term_l123_12306


namespace perfect_square_quadratic_l123_12351

theorem perfect_square_quadratic (m : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, x^2 + 2*(m-3)*x + 16 = y^2) → (m = 7 ∨ m = -1) := by
  sorry

end perfect_square_quadratic_l123_12351


namespace volume_increase_rectangular_prism_l123_12322

theorem volume_increase_rectangular_prism 
  (l w h : ℝ) 
  (l_increase : ℝ) 
  (w_increase : ℝ) 
  (h_increase : ℝ) 
  (hl : l_increase = 0.15) 
  (hw : w_increase = 0.20) 
  (hh : h_increase = 0.10) :
  let new_volume := (l * (1 + l_increase)) * (w * (1 + w_increase)) * (h * (1 + h_increase))
  let original_volume := l * w * h
  let volume_increase_percentage := (new_volume - original_volume) / original_volume * 100
  volume_increase_percentage = 51.8 := by
sorry

end volume_increase_rectangular_prism_l123_12322


namespace ishaan_age_l123_12384

/-- Proves that Ishaan is 6 years old given the conditions of the problem -/
theorem ishaan_age (daniel_age : ℕ) (future_years : ℕ) (future_ratio : ℕ) : 
  daniel_age = 69 → 
  future_years = 15 → 
  future_ratio = 4 → 
  ∃ (ishaan_age : ℕ), 
    daniel_age + future_years = future_ratio * (ishaan_age + future_years) ∧ 
    ishaan_age = 6 := by
  sorry

end ishaan_age_l123_12384


namespace min_value_product_sum_l123_12390

theorem min_value_product_sum (p q r s t u v w : ℝ) 
  (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) (hs : 0 < s) 
  (ht : 0 < t) (hu : 0 < u) (hv : 0 < v) (hw : 0 < w)
  (h1 : p * q * r * s = 16) (h2 : t * u * v * w = 25) :
  (p * t)^2 + (q * u)^2 + (r * v)^2 + (s * w)^2 ≥ 40 ∧
  ∃ (p' q' r' s' t' u' v' w' : ℝ),
    0 < p' ∧ 0 < q' ∧ 0 < r' ∧ 0 < s' ∧
    0 < t' ∧ 0 < u' ∧ 0 < v' ∧ 0 < w' ∧
    p' * q' * r' * s' = 16 ∧
    t' * u' * v' * w' = 25 ∧
    (p' * t')^2 + (q' * u')^2 + (r' * v')^2 + (s' * w')^2 = 40 :=
by
  sorry

end min_value_product_sum_l123_12390


namespace modulus_of_z_l123_12328

theorem modulus_of_z (z : ℂ) (h : z / (1 - z) = Complex.I) : Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end modulus_of_z_l123_12328


namespace max_profit_at_16_l123_12380

/-- Represents the annual profit function for a factory -/
def annual_profit (x : ℕ+) : ℚ :=
  if x ≤ 20 then -x^2 + 32*x - 100 else 160 - x

/-- Theorem stating that the maximum annual profit occurs at 16 units -/
theorem max_profit_at_16 :
  ∀ x : ℕ+, annual_profit 16 ≥ annual_profit x :=
by sorry

end max_profit_at_16_l123_12380


namespace hyperbola_asymptote_l123_12375

/-- Given a hyperbola with equation x²/a² - y² = 1 where a > 0,
    if one of its asymptotes is √3x + y = 0, then a = √3/3 -/
theorem hyperbola_asymptote (a : ℝ) (h1 : a > 0) :
  (∃ x y : ℝ, x^2 / a^2 - y^2 = 1 ∧ Real.sqrt 3 * x + y = 0) →
  a = Real.sqrt 3 / 3 :=
by sorry

end hyperbola_asymptote_l123_12375


namespace least_sum_p_q_l123_12368

theorem least_sum_p_q : ∃ (p q : ℕ), 
  p > 1 ∧ q > 1 ∧ 
  17 * (p + 1) = 21 * (q + 1) ∧
  p + q = 38 ∧
  ∀ (p' q' : ℕ), p' > 1 → q' > 1 → 17 * (p' + 1) = 21 * (q' + 1) → p' + q' ≥ 38 :=
by sorry

end least_sum_p_q_l123_12368


namespace simplify_expression_l123_12398

theorem simplify_expression (x : ℝ) : (3*x - 6)*(x + 8) - (x + 6)*(3*x - 2) = 2*x - 36 := by
  sorry

end simplify_expression_l123_12398


namespace open_box_volume_l123_12376

/-- The volume of an open box formed by cutting squares from corners of a rectangular sheet -/
theorem open_box_volume 
  (sheet_length sheet_width cut_size : ℝ) 
  (h_length : sheet_length = 48)
  (h_width : sheet_width = 38)
  (h_cut : cut_size = 8) : 
  (sheet_length - 2 * cut_size) * (sheet_width - 2 * cut_size) * cut_size = 5632 :=
by sorry

end open_box_volume_l123_12376


namespace billy_bobbi_probability_zero_l123_12373

def billy_number (n : ℕ) : Prop := n > 0 ∧ n < 150 ∧ 15 ∣ n
def bobbi_number (n : ℕ) : Prop := n > 0 ∧ n < 150 ∧ 20 ∣ n
def is_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem billy_bobbi_probability_zero :
  ∀ (b₁ b₂ : ℕ), 
    billy_number b₁ → 
    bobbi_number b₂ → 
    (is_square b₁ ∨ is_square b₂) →
    b₁ = b₂ → 
    False :=
sorry

end billy_bobbi_probability_zero_l123_12373


namespace simplify_expression_l123_12394

theorem simplify_expression (x : ℝ) (h : x = Real.sqrt 2) :
  (x^2 - x) / (x^2 - 2*x + 1) = 2 + Real.sqrt 2 := by
  sorry

end simplify_expression_l123_12394


namespace largest_number_l123_12307

theorem largest_number : 
  (1 : ℝ) ≥ Real.sqrt 29 - Real.sqrt 21 ∧ 
  (1 : ℝ) ≥ Real.pi / 3.142 ∧ 
  (1 : ℝ) ≥ 5.1 * Real.sqrt 0.0361 ∧ 
  (1 : ℝ) ≥ 6 / (Real.sqrt 13 + Real.sqrt 7) := by
  sorry

end largest_number_l123_12307


namespace light_2004_is_red_l123_12343

def light_color (n : ℕ) : String :=
  match n % 6 with
  | 0 => "red"
  | 1 => "green"
  | 2 => "yellow"
  | 3 => "yellow"
  | 4 => "red"
  | 5 => "red"
  | _ => "error" -- This case should never occur

theorem light_2004_is_red : light_color 2004 = "red" := by
  sorry

end light_2004_is_red_l123_12343


namespace frac_greater_than_one_solution_set_l123_12334

theorem frac_greater_than_one_solution_set (x : ℝ) : 
  (1 / x > 1) ↔ (0 < x ∧ x < 1) :=
sorry

end frac_greater_than_one_solution_set_l123_12334


namespace platform_length_l123_12344

/-- The length of a platform given train speed, crossing time, and train length -/
theorem platform_length
  (train_speed : ℝ)
  (crossing_time : ℝ)
  (train_length : ℝ)
  (h1 : train_speed = 72)  -- km/hr
  (h2 : crossing_time = 26)  -- seconds
  (h3 : train_length = 250)  -- meters
  : (train_speed * (5/18) * crossing_time) - train_length = 270 := by
  sorry

end platform_length_l123_12344


namespace gcd_459_357_l123_12379

theorem gcd_459_357 : Nat.gcd 459 357 = 51 := by
  sorry

end gcd_459_357_l123_12379


namespace sum_of_compositions_l123_12315

def r (x : ℝ) : ℝ := |x + 1| - 3

def s (x : ℝ) : ℝ := -|x + 2|

def evaluation_points : List ℝ := [-5, -4, -3, -2, -1, 0, 1, 2, 3]

theorem sum_of_compositions :
  (evaluation_points.map (fun x => s (r x))).sum = -37 := by sorry

end sum_of_compositions_l123_12315


namespace eunji_pocket_money_l123_12385

theorem eunji_pocket_money (initial_money : ℕ) : 
  (initial_money / 4 : ℕ) + 
  ((3 * initial_money / 4) / 3 : ℕ) + 
  1600 = initial_money → 
  initial_money = 3200 := by
sorry

end eunji_pocket_money_l123_12385


namespace parabola_tangent_condition_l123_12333

/-- A parabola is tangent to a line if and only if their intersection has exactly one solution --/
def is_tangent (a b : ℝ) : Prop :=
  ∃! x, a * x^2 + b * x + 12 = 2 * x + 3

/-- The main theorem stating the conditions for the parabola to be tangent to the line --/
theorem parabola_tangent_condition (a b : ℝ) :
  is_tangent a b ↔ (b = 2 + 6 * Real.sqrt a ∨ b = 2 - 6 * Real.sqrt a) ∧ a ≥ 0 := by
  sorry

end parabola_tangent_condition_l123_12333


namespace hcf_problem_l123_12316

theorem hcf_problem (A B : ℕ) (H : ℕ) : 
  A = 900 → 
  A > B → 
  B > 0 →
  Nat.lcm A B = H * 11 * 15 →
  Nat.gcd A B = H →
  Nat.gcd A B = 165 := by
sorry

end hcf_problem_l123_12316


namespace parabola_equation_l123_12363

theorem parabola_equation (p : ℝ) (x₀ y₀ : ℝ) : 
  p > 0 → 
  y₀^2 = 2*p*x₀ → 
  (x₀ + p/2)^2 + y₀^2 = 100 → 
  y₀^2 = 36 → 
  (y^2 = 4*x ∨ y^2 = 36*x) := by
  sorry

end parabola_equation_l123_12363


namespace greeting_card_profit_l123_12369

/-- Represents the greeting card sale problem -/
theorem greeting_card_profit
  (purchase_price : ℚ)
  (total_sale : ℚ)
  (h_purchase : purchase_price = 21 / 100)
  (h_sale : total_sale = 1457 / 100)
  (h_price_limit : ∃ (selling_price : ℚ), 
    selling_price ≤ 2 * purchase_price ∧
    selling_price * (total_sale / selling_price) = total_sale)
  : ∃ (profit : ℚ), profit = 47 / 10 :=
sorry

end greeting_card_profit_l123_12369


namespace book_cost_calculation_l123_12381

theorem book_cost_calculation (initial_amount : ℕ) (books_bought : ℕ) (remaining_amount : ℕ) :
  initial_amount = 79 →
  books_bought = 9 →
  remaining_amount = 16 →
  (initial_amount - remaining_amount) / books_bought = 7 := by
  sorry

end book_cost_calculation_l123_12381


namespace evaluate_expression_l123_12341

theorem evaluate_expression : -(16 / 4 * 12 - 100 + 2^3 * 6) = 4 := by
  sorry

end evaluate_expression_l123_12341


namespace factorization_equality_l123_12386

theorem factorization_equality (x y : ℝ) : -4 * x^2 + y^2 = (y - 2*x) * (y + 2*x) := by
  sorry

end factorization_equality_l123_12386


namespace lemonade_sales_l123_12356

theorem lemonade_sales (katya ricky tina : ℕ) : 
  ricky = 9 →
  tina = 2 * (katya + ricky) →
  tina = katya + 26 →
  katya = 8 := by sorry

end lemonade_sales_l123_12356


namespace quadratic_inequality_solution_sets_l123_12378

theorem quadratic_inequality_solution_sets
  (a b : ℝ)
  (h : Set.Ioo (-1 : ℝ) (1/2) = {x | a * x^2 + b * x + 3 > 0}) :
  Set.Ioo (-1 : ℝ) 2 = {x | 3 * x^2 + b * x + a < 0} :=
sorry

end quadratic_inequality_solution_sets_l123_12378


namespace sqrt_equation_solutions_l123_12304

theorem sqrt_equation_solutions :
  ∀ x : ℝ, (Real.sqrt ((3 + Real.sqrt 5) ^ x) + Real.sqrt ((3 - Real.sqrt 5) ^ x) = 6) ↔ (x = 2 ∨ x = -2) := by
  sorry

end sqrt_equation_solutions_l123_12304


namespace abc_equality_l123_12370

theorem abc_equality (a b c : ℝ) (h : a + 1/b = b + 1/c ∧ b + 1/c = c + 1/a) :
  a^2 * b^2 * c^2 = 1 ∨ (a = b ∧ b = c) := by
  sorry

end abc_equality_l123_12370


namespace negation_of_p_l123_12345

-- Define the set M
def M : Set ℝ := {x | -1 < x ∧ x < 2}

-- Define the original proposition p
def p : Prop := ∃ x ∈ M, x^2 - x - 2 < 0

-- Statement: The negation of p is equivalent to ∀x ∈ M, x^2 - x - 2 ≥ 0
theorem negation_of_p : ¬p ↔ ∀ x ∈ M, x^2 - x - 2 ≥ 0 := by sorry

end negation_of_p_l123_12345


namespace average_of_eleven_numbers_l123_12323

theorem average_of_eleven_numbers (first_six_avg : ℝ) (last_six_avg : ℝ) (sixth_number : ℝ) :
  first_six_avg = 78 →
  last_six_avg = 75 →
  sixth_number = 258 →
  (6 * first_six_avg + 6 * last_six_avg - sixth_number) / 11 = 60 :=
by sorry

end average_of_eleven_numbers_l123_12323


namespace quadratic_rewrite_proof_l123_12360

theorem quadratic_rewrite_proof :
  ∃ (a b c : ℚ), 
    (∀ k, 12 * k^2 + 8 * k - 16 = a * (k + b)^2 + c) ∧
    (c + 3 * b = -49 / 3) := by
  sorry

end quadratic_rewrite_proof_l123_12360


namespace diagonals_in_nonagon_l123_12302

/-- The number of diagonals in a regular nine-sided polygon -/
theorem diagonals_in_nonagon : 
  let n : ℕ := 9  -- number of sides
  let total_connections := n.choose 2  -- total number of connections between vertices
  let num_sides := n  -- number of sides (which are not diagonals)
  total_connections - num_sides = 27 := by sorry

end diagonals_in_nonagon_l123_12302


namespace rectangle_area_l123_12371

-- Define the points
variable (A B C D E F : ℝ × ℝ)

-- Define the properties of the semicircle and rectangle
def is_semicircle (F E : ℝ × ℝ) : Prop := sorry

def is_inscribed_rectangle (A B C D : ℝ × ℝ) (F E : ℝ × ℝ) : Prop := sorry

def is_right_triangle (D F C : ℝ × ℝ) : Prop := sorry

-- Define the distances
def distance (P Q : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem rectangle_area (A B C D E F : ℝ × ℝ) :
  is_semicircle F E →
  is_inscribed_rectangle A B C D F E →
  is_right_triangle D F C →
  distance D A = 12 →
  distance F D = 7 →
  distance A E = 7 →
  distance D A * distance C D = 24 * Real.sqrt 30 := by sorry

end rectangle_area_l123_12371


namespace max_integer_k_for_distinct_roots_l123_12313

theorem max_integer_k_for_distinct_roots (k : ℤ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 - (4*k - 2)*x + 4*k^2 = 0 ∧ y^2 - (4*k - 2)*y + 4*k^2 = 0) →
  k ≤ 0 :=
by sorry

end max_integer_k_for_distinct_roots_l123_12313


namespace equation_solution_l123_12312

theorem equation_solution : 
  ∃! x : ℝ, x ≠ 2 ∧ (4 * x^2 + 3 * x + 2) / (x - 2) = 4 * x + 3 :=
by
  use -1
  sorry

end equation_solution_l123_12312


namespace special_numbers_count_l123_12331

/-- Sum of digits of a positive integer -/
def heartsuit (n : ℕ+) : ℕ :=
  sorry

/-- Counts the number of three-digit positive integers x such that heartsuit(heartsuit(x)) = 5 -/
def count_special_numbers : ℕ :=
  sorry

/-- Theorem stating that there are exactly 60 three-digit positive integers x 
    such that heartsuit(heartsuit(x)) = 5 -/
theorem special_numbers_count : count_special_numbers = 60 := by
  sorry

end special_numbers_count_l123_12331


namespace select_three_from_four_l123_12393

theorem select_three_from_four : Nat.choose 4 3 = 4 := by
  sorry

end select_three_from_four_l123_12393


namespace max_product_sum_2006_l123_12362

theorem max_product_sum_2006 : 
  (∃ (a b : ℤ), a + b = 2006 ∧ ∀ (x y : ℤ), x + y = 2006 → x * y ≤ a * b) ∧
  (∀ (a b : ℤ), a + b = 2006 → a * b ≤ 1006009) := by
  sorry

end max_product_sum_2006_l123_12362


namespace octal_5374_to_decimal_l123_12352

def octal_to_decimal (a b c d : Nat) : Nat :=
  d * 8^0 + c * 8^1 + b * 8^2 + a * 8^3

theorem octal_5374_to_decimal :
  octal_to_decimal 5 3 7 4 = 2812 := by
  sorry

end octal_5374_to_decimal_l123_12352


namespace max_value_quadratic_l123_12391

theorem max_value_quadratic (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^2 - 2*x*y + 3*y^2 = 10) : 
  x^2 + 2*x*y + 3*y^2 ≤ 20 + 10 * Real.sqrt 3 := by
  sorry

end max_value_quadratic_l123_12391


namespace total_apples_bought_l123_12326

theorem total_apples_bought (num_men num_women : ℕ) (apples_per_man : ℕ) (extra_apples_per_woman : ℕ) : 
  num_men = 2 → 
  num_women = 3 → 
  apples_per_man = 30 → 
  extra_apples_per_woman = 20 →
  num_men * apples_per_man + num_women * (apples_per_man + extra_apples_per_woman) = 210 := by
  sorry

#check total_apples_bought

end total_apples_bought_l123_12326


namespace max_value_of_f_on_interval_l123_12339

def f (x : ℝ) : ℝ := 2 * x^2 + 4 * x - 1

theorem max_value_of_f_on_interval :
  ∃ (c : ℝ), c ∈ Set.Icc (-2 : ℝ) 2 ∧
  (∀ x, x ∈ Set.Icc (-2 : ℝ) 2 → f x ≤ f c) ∧
  f c = 23 :=
sorry

end max_value_of_f_on_interval_l123_12339


namespace digit67_is_one_l123_12383

/-- The sequence of digits formed by concatenating integers from 50 down to 1 -/
def integerSequence : List Nat := sorry

/-- The 67th digit in the sequence -/
def digit67 : Nat := sorry

/-- Theorem stating that the 67th digit in the sequence is 1 -/
theorem digit67_is_one : digit67 = 1 := by sorry

end digit67_is_one_l123_12383


namespace reward_function_satisfies_requirements_l123_12320

theorem reward_function_satisfies_requirements :
  let f : ℝ → ℝ := λ x => 2 * Real.sqrt x - 6
  let domain : Set ℝ := { x | 25 ≤ x ∧ x ≤ 1600 }
  (∀ x ∈ domain, ∀ y ∈ domain, x < y → f x < f y) ∧
  (∀ x ∈ domain, f x ≤ 90) ∧
  (∀ x ∈ domain, f x ≤ x / 5) :=
by sorry

end reward_function_satisfies_requirements_l123_12320


namespace intersection_singleton_l123_12305

/-- The set A parameterized by a -/
def A (a : ℝ) : Set (ℝ × ℝ) := {p | p.2 = a * p.1 + 1}

/-- The set B -/
def B : Set (ℝ × ℝ) := {p | p.2 = |p.1|}

/-- The theorem stating the condition for A ∩ B to be a singleton -/
theorem intersection_singleton (a : ℝ) :
  (∃! p, p ∈ A a ∩ B) ↔ a ∈ Set.Iic (-1) ∪ Set.Ici 1 :=
sorry

end intersection_singleton_l123_12305


namespace perpendicular_lines_parallel_perpendicular_planes_parallel_l123_12365

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp_line_plane : Line → Plane → Prop)
variable (perp_plane_plane : Plane → Plane → Prop)
variable (parallel_line : Line → Line → Prop)
variable (parallel_plane : Plane → Plane → Prop)

-- Theorem 1: If two lines are perpendicular to the same plane, they are parallel
theorem perpendicular_lines_parallel (a b : Line) (α : Plane) :
  perp_line_plane a α → perp_line_plane b α → parallel_line a b :=
sorry

-- Theorem 2: If a line is perpendicular to a plane, and that plane is perpendicular to another plane, then the two planes are parallel
theorem perpendicular_planes_parallel (a : Line) (α β : Plane) :
  perp_line_plane a α → perp_plane_plane α β → parallel_plane α β :=
sorry

end perpendicular_lines_parallel_perpendicular_planes_parallel_l123_12365


namespace rebus_solution_exists_and_unique_l123_12350

def is_valid_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def are_distinct (a b c d e f g h i j : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧ a ≠ j ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧ b ≠ j ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧ c ≠ j ∧
  d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧ d ≠ j ∧
  e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧ e ≠ j ∧
  f ≠ g ∧ f ≠ h ∧ f ≠ i ∧ f ≠ j ∧
  g ≠ h ∧ g ≠ i ∧ g ≠ j ∧
  h ≠ i ∧ h ≠ j ∧
  i ≠ j

theorem rebus_solution_exists_and_unique :
  ∃! (a b c d e f g h i j : ℕ),
    is_valid_digit a ∧ is_valid_digit b ∧ is_valid_digit c ∧
    is_valid_digit d ∧ is_valid_digit e ∧ is_valid_digit f ∧
    is_valid_digit g ∧ is_valid_digit h ∧ is_valid_digit i ∧ is_valid_digit j ∧
    are_distinct a b c d e f g h i j ∧
    100 * a + 10 * b + c + 100 * d + 10 * e + f = 1000 * g + 100 * h + 10 * i + j :=
sorry

end rebus_solution_exists_and_unique_l123_12350


namespace sum_of_squares_of_roots_l123_12348

theorem sum_of_squares_of_roots (r s t : ℝ) : 
  (2 * r^3 + 3 * r^2 - 5 * r + 1 = 0) →
  (2 * s^3 + 3 * s^2 - 5 * s + 1 = 0) →
  (2 * t^3 + 3 * t^2 - 5 * t + 1 = 0) →
  (r ≠ s) → (r ≠ t) → (s ≠ t) →
  r^2 + s^2 + t^2 = -11/4 := by
  sorry

end sum_of_squares_of_roots_l123_12348


namespace abs_negative_seven_l123_12329

theorem abs_negative_seven : |(-7 : ℤ)| = 7 := by
  sorry

end abs_negative_seven_l123_12329


namespace lizzy_money_problem_l123_12311

/-- The amount of cents Lizzy's father gave her -/
def father_gave : ℕ := 40

/-- The amount of cents Lizzy spent on candy -/
def spent_on_candy : ℕ := 50

/-- The amount of cents Lizzy's uncle gave her -/
def uncle_gave : ℕ := 70

/-- The amount of cents Lizzy has now -/
def current_amount : ℕ := 140

/-- The amount of cents Lizzy's mother gave her -/
def mother_gave : ℕ := 80

theorem lizzy_money_problem :
  mother_gave = current_amount + spent_on_candy - (father_gave + uncle_gave) :=
by sorry

end lizzy_money_problem_l123_12311


namespace scale_length_l123_12337

/-- A scale is divided into equal parts -/
structure Scale :=
  (parts : ℕ)
  (part_length : ℕ)
  (total_length : ℕ)

/-- Convert inches to feet -/
def inches_to_feet (inches : ℕ) : ℚ :=
  inches / 12

/-- Theorem: A scale with 4 parts, each 24 inches long, is 8 feet long -/
theorem scale_length (s : Scale) (h1 : s.parts = 4) (h2 : s.part_length = 24) :
  inches_to_feet s.total_length = 8 := by
  sorry

#check scale_length

end scale_length_l123_12337


namespace chess_tournament_games_l123_12388

theorem chess_tournament_games (n : ℕ) 
  (total_players : ℕ) (total_games : ℕ) :
  total_players = 6 →
  total_games = 30 →
  total_games = n * (total_players.choose 2) →
  n = 2 := by
sorry

end chess_tournament_games_l123_12388


namespace quadratic_roots_problem_l123_12336

theorem quadratic_roots_problem (m : ℝ) (α β : ℝ) :
  (α > 0 ∧ β > 0) ∧ 
  (α^2 + (2*m - 1)*α + m^2 = 0) ∧ 
  (β^2 + (2*m - 1)*β + m^2 = 0) →
  ((m ≤ 1/4 ∧ m ≠ 0) ∧
   (α^2 + β^2 = 49 → m = -4)) :=
by sorry

end quadratic_roots_problem_l123_12336


namespace journey_solution_l123_12374

def journey_problem (total_time : ℝ) (speed1 speed2 speed3 speed4 : ℝ) : Prop :=
  let distance := total_time * (speed1 + speed2 + speed3 + speed4) / 4
  total_time = (distance / 4) / speed1 + (distance / 4) / speed2 + (distance / 4) / speed3 + (distance / 4) / speed4 ∧
  distance = 960

theorem journey_solution :
  journey_problem 60 20 10 15 30 := by
  sorry

end journey_solution_l123_12374


namespace symmetric_line_l123_12319

/-- Given a line L1 with equation x - 2y + 1 = 0 and a line of symmetry x = 1,
    the symmetric line L2 has the equation x + 2y - 3 = 0 -/
theorem symmetric_line (x y : ℝ) :
  (x - 2*y + 1 = 0) →  -- Original line L1
  (x = 1) →            -- Line of symmetry
  (x + 2*y - 3 = 0)    -- Symmetric line L2
:= by sorry

end symmetric_line_l123_12319


namespace gcd_of_B_is_two_l123_12399

def B : Set ℕ := {n : ℕ | ∃ x : ℕ, x > 0 ∧ n = x + (x + 1) + (x + 2) + (x + 3)}

theorem gcd_of_B_is_two : 
  ∃ d : ℕ, d > 0 ∧ (∀ n ∈ B, d ∣ n) ∧ (∀ m : ℕ, m > 0 → (∀ n ∈ B, m ∣ n) → m ≤ d) ∧ d = 2 := by
  sorry

end gcd_of_B_is_two_l123_12399


namespace arithmetic_sequence_problem_l123_12396

/-- Given an arithmetic sequence {a_n} with first term a₁ = -1 and common difference d = 2,
    prove that if a_{n-1} = 15, then n = 10. -/
theorem arithmetic_sequence_problem (a : ℕ → ℤ) (n : ℕ) :
  (∀ k, a (k + 1) = a k + 2) →  -- Common difference is 2
  a 1 = -1 →                    -- First term is -1
  a (n - 1) = 15 →              -- a_{n-1} = 15
  n = 10 :=
by
  sorry

end arithmetic_sequence_problem_l123_12396


namespace arrangement_problem_l123_12347

/-- The number of ways to arrange people in a row -/
def arrange (n : ℕ) (m : ℕ) : ℕ :=
  n.factorial * m.factorial * (n + 1).factorial

/-- The problem statement -/
theorem arrangement_problem : arrange 5 2 = 1440 := by
  sorry

end arrangement_problem_l123_12347


namespace fraction_equality_l123_12354

theorem fraction_equality (p q : ℚ) (h : p / q = 4 / 5) :
  18 / 7 + (2 * q - p) / (2 * q + p) = 3 := by
  sorry

end fraction_equality_l123_12354


namespace livestock_puzzle_l123_12353

theorem livestock_puzzle :
  ∃! (x y z : ℕ), 
    x + y + z = 100 ∧ 
    10 * x + 3 * y + (1/2) * z = 100 ∧
    x = 5 ∧ y = 1 ∧ z = 94 := by
  sorry

end livestock_puzzle_l123_12353


namespace quadratic_equal_roots_l123_12382

theorem quadratic_equal_roots (m : ℝ) : 
  (∃ x : ℝ, 3 * x^2 - m * x - 2 * x + 15 = 0 ∧ 
   ∀ y : ℝ, 3 * y^2 - m * y - 2 * y + 15 = 0 → y = x) ↔ 
  (m = 6 * Real.sqrt 5 - 2 ∨ m = -6 * Real.sqrt 5 - 2) :=
sorry

end quadratic_equal_roots_l123_12382


namespace inequality_proof_l123_12361

theorem inequality_proof (x y z : ℤ) :
  (x^2 + y^2*z^2) * (y^2 + x^2*z^2) * (z^2 + x^2*y^2) ≥ 8*x*y^2*z^3 := by
  sorry

end inequality_proof_l123_12361


namespace incorrect_assignment_l123_12364

-- Define valid assignment statements
def valid_assignment (stmt : String) : Prop :=
  stmt = "N = N + 1" ∨ stmt = "K = K * K" ∨ stmt = "C = A / B"

-- Define the statement in question
def questionable_statement : String := "C = A(B + D)"

-- Theorem to prove
theorem incorrect_assignment :
  (∀ stmt, valid_assignment stmt → stmt ≠ questionable_statement) →
  ¬(valid_assignment questionable_statement) :=
by
  sorry

end incorrect_assignment_l123_12364


namespace more_unrepresentable_ten_digit_numbers_l123_12332

theorem more_unrepresentable_ten_digit_numbers :
  let total_ten_digit_numbers := 9 * (10 ^ 9)
  let five_digit_numbers := 9 * (10 ^ 4)
  let max_representable := five_digit_numbers * (five_digit_numbers + 1)
  max_representable < total_ten_digit_numbers / 2 := by
sorry

end more_unrepresentable_ten_digit_numbers_l123_12332


namespace problem_solution_l123_12301

theorem problem_solution (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : x - y = 4 := by
  sorry

end problem_solution_l123_12301


namespace number_divided_by_seven_l123_12303

theorem number_divided_by_seven : ∃ x : ℚ, x / 7 = 5 / 14 ∧ x = 5 / 2 := by
  sorry

end number_divided_by_seven_l123_12303


namespace inequality_solution_l123_12395

theorem inequality_solution (x : ℝ) : 
  (2*x - 1) / (x + 3) > (3*x - 2) / (x - 4) ↔ x ∈ Set.Ioo (-9 : ℝ) (-3) ∪ Set.Ioo 2 4 := by
  sorry

end inequality_solution_l123_12395


namespace simplify_fraction_l123_12325

theorem simplify_fraction : 5 * (14 / 3) * (21 / -70) = -35 / 2 := by
  sorry

end simplify_fraction_l123_12325


namespace total_fruits_shared_l123_12367

def persimmons_to_yuna : ℕ := 2
def apples_to_minyoung : ℕ := 7

theorem total_fruits_shared : persimmons_to_yuna + apples_to_minyoung = 9 := by
  sorry

end total_fruits_shared_l123_12367


namespace add_25_to_number_l123_12317

theorem add_25_to_number (x : ℤ) : 43 + x = 81 → x + 25 = 63 := by
  sorry

end add_25_to_number_l123_12317


namespace square_formation_theorem_l123_12310

/-- Function to calculate the sum of integers from 1 to n -/
def sum_to_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Function to calculate the minimum number of sticks to break -/
def min_sticks_to_break (n : ℕ) : ℕ :=
  let total_length := sum_to_n n
  if total_length % 4 = 0 then 0
  else 
    let target_length := ((total_length + 3) / 4) * 4
    (target_length - total_length + 1) / 2

theorem square_formation_theorem :
  (min_sticks_to_break 12 = 2) ∧ (min_sticks_to_break 15 = 0) :=
sorry

end square_formation_theorem_l123_12310


namespace saree_ultimate_cost_l123_12300

/-- Calculates the ultimate cost of a saree after discounts and commission -/
def ultimate_cost (initial_price : ℝ) (discount1 discount2 discount3 commission : ℝ) : ℝ :=
  let price_after_discount1 := initial_price * (1 - discount1)
  let price_after_discount2 := price_after_discount1 * (1 - discount2)
  let price_after_discount3 := price_after_discount2 * (1 - discount3)
  let final_price := price_after_discount3 * (1 - commission)
  final_price

/-- Theorem stating the ultimate cost of the saree -/
theorem saree_ultimate_cost :
  ultimate_cost 340 0.2 0.15 0.1 0.05 = 197.676 := by
  sorry

end saree_ultimate_cost_l123_12300


namespace equal_area_rectangles_width_l123_12377

/-- Given two rectangles with equal areas, where one rectangle measures 8 inches by 15 inches
    and the other is 4 inches long, prove that the width of the second rectangle is 30 inches. -/
theorem equal_area_rectangles_width (carol_length carol_width jordan_length jordan_width : ℝ)
    (h1 : carol_length = 8)
    (h2 : carol_width = 15)
    (h3 : jordan_length = 4)
    (h4 : carol_length * carol_width = jordan_length * jordan_width) :
    jordan_width = 30 := by
  sorry

end equal_area_rectangles_width_l123_12377


namespace polynomial_factorization_sum_l123_12314

theorem polynomial_factorization_sum (b₁ b₂ b₃ c₁ c₂ c₃ : ℝ) 
  (h : ∀ x : ℝ, x^8 - 3*x^6 + 3*x^4 - x^2 + 2 = 
    (x^2 + b₁*x + c₁) * (x^2 + b₂*x + c₂) * (x^2 + 2*b₃*x + c₃)) : 
  b₁*c₁ + b₂*c₂ + 2*b₃*c₃ = 0 := by
  sorry

end polynomial_factorization_sum_l123_12314


namespace tank_emptied_in_two_minutes_l123_12359

/-- Represents the state and properties of a water tank system -/
structure WaterTank where
  initialFill : ℚ  -- Initial fill level of the tank (1/5)
  pipeARate : ℚ    -- Rate at which pipe A fills the tank (1/15 per minute)
  pipeBRate : ℚ    -- Rate at which pipe B empties the tank (1/6 per minute)

/-- Calculates the time to empty or fill the tank completely -/
def timeToEmptyOrFill (tank : WaterTank) : ℚ :=
  tank.initialFill / (tank.pipeBRate - tank.pipeARate)

/-- Theorem stating that the tank will be emptied in 2 minutes -/
theorem tank_emptied_in_two_minutes (tank : WaterTank) 
  (h1 : tank.initialFill = 1/5)
  (h2 : tank.pipeARate = 1/15)
  (h3 : tank.pipeBRate = 1/6) : 
  timeToEmptyOrFill tank = 2 := by
  sorry

#eval timeToEmptyOrFill { initialFill := 1/5, pipeARate := 1/15, pipeBRate := 1/6 }

end tank_emptied_in_two_minutes_l123_12359


namespace parabola_tangent_problem_l123_12308

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := x^2 = 2*p*y

-- Define the point M
def point_M (p : ℝ) : ℝ × ℝ := (2, -2*p)

-- Define a line touching the parabola at two points
def touching_line (p : ℝ) (A B : ℝ × ℝ) : Prop :=
  parabola p A.1 A.2 ∧ parabola p B.1 B.2 ∧
  ∃ (m c : ℝ), A.2 = m * A.1 + c ∧ B.2 = m * B.1 + c ∧
  point_M p = (2, m * 2 + c)

-- Define the midpoint condition
def midpoint_condition (A B : ℝ × ℝ) : Prop :=
  (A.2 + B.2) / 2 = 6

-- Theorem statement
theorem parabola_tangent_problem (p : ℝ) (A B : ℝ × ℝ) :
  p > 0 →
  touching_line p A B →
  midpoint_condition A B →
  p = 1 ∨ p = 2 := by
  sorry

end parabola_tangent_problem_l123_12308


namespace min_value_sum_product_l123_12389

theorem min_value_sum_product (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + y = x * y) :
  x + y ≥ 4 := by
sorry

end min_value_sum_product_l123_12389


namespace systematic_sampling_proof_l123_12392

/-- Systematic sampling function that returns the next sample number -/
def nextSample (total : ℕ) (sampleSize : ℕ) (current : ℕ) : ℕ :=
  (current + total / sampleSize) % total

/-- Proposition: In a systematic sampling of 4 items from 56 items, 
    if items 7 and 35 are selected, then the other two selected items 
    are numbered 21 and 49 -/
theorem systematic_sampling_proof :
  let total := 56
  let sampleSize := 4
  let first := 7
  let second := 35
  nextSample total sampleSize first = 21 ∧
  nextSample total sampleSize second = 49 := by
  sorry

#eval nextSample 56 4 7  -- Should output 21
#eval nextSample 56 4 35 -- Should output 49

end systematic_sampling_proof_l123_12392


namespace min_total_diff_three_students_l123_12324

/-- Represents a student's ability characteristics as a list of 12 binary values -/
def Student := List Bool

/-- Calculates the number of different ability characteristics between two students -/
def diffCount (a b : Student) : Nat :=
  List.sum (List.map (fun (x, y) => if x = y then 0 else 1) (List.zip a b))

/-- Checks if two students have a significant comprehensive ability difference -/
def significantDiff (a b : Student) : Prop :=
  diffCount a b ≥ 7

/-- Calculates the total number of different ability characteristics among three students -/
def totalDiff (a b c : Student) : Nat :=
  diffCount a b + diffCount b c + diffCount c a

/-- Theorem: The minimum total number of different ability characteristics among three students
    with significant differences between each pair is 22 -/
theorem min_total_diff_three_students (a b c : Student) :
  (List.length a = 12 ∧ List.length b = 12 ∧ List.length c = 12) →
  (significantDiff a b ∧ significantDiff b c ∧ significantDiff c a) →
  totalDiff a b c ≥ 22 ∧ ∃ (x y z : Student), totalDiff x y z = 22 :=
sorry

end min_total_diff_three_students_l123_12324


namespace problem_statement_l123_12366

/-- The number we're looking for -/
def x : ℝ := 640

/-- 50% of x is 190 more than 20% of 650 -/
theorem problem_statement : 0.5 * x = 0.2 * 650 + 190 := by
  sorry

end problem_statement_l123_12366


namespace three_digit_number_count_l123_12346

/-- A three-digit number where the hundreds digit equals the units digit -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  units : Nat
  is_valid : hundreds = units ∧ hundreds ≥ 1 ∧ hundreds ≤ 9 ∧ tens ≤ 9

/-- The value of a ThreeDigitNumber -/
def ThreeDigitNumber.value (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.units

/-- Predicate for divisibility by 4 -/
def divisible_by_four (n : Nat) : Prop :=
  n % 4 = 0

theorem three_digit_number_count :
  (∃ (s : Finset ThreeDigitNumber), s.card = 90) ∧
  (∃ (s : Finset ThreeDigitNumber), s.card = 20 ∧ ∀ n ∈ s, divisible_by_four n.value) :=
sorry

end three_digit_number_count_l123_12346


namespace suit_tie_discount_cost_l123_12355

/-- Represents the cost calculation for two discount options in a suit and tie sale. -/
theorem suit_tie_discount_cost 
  (suit_price : ℕ) 
  (tie_price : ℕ) 
  (num_suits : ℕ) 
  (num_ties : ℕ) 
  (h1 : suit_price = 500)
  (h2 : tie_price = 100)
  (h3 : num_suits = 20)
  (h4 : num_ties > 20) :
  (num_suits * suit_price + (num_ties - num_suits) * tie_price = 100 * num_ties + 8000) ∧ 
  (((num_suits * suit_price + num_ties * tie_price) * 90) / 100 = 90 * num_ties + 9000) := by
  sorry

end suit_tie_discount_cost_l123_12355


namespace meter_to_skips_l123_12309

theorem meter_to_skips 
  (a b c d e f g : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0) (hf : f > 0) (hg : g > 0)
  (hop_skip : a * 1 = b * 1)  -- a hops = b skips
  (jog_hop : c * 1 = d * 1)   -- c jogs = d hops
  (dash_jog : e * 1 = f * 1)  -- e dashes = f jogs
  (meter_dash : 1 = g * 1)    -- 1 meter = g dashes
  : 1 = (g * f * d * b) / (e * c * a) * 1 := by
  sorry

end meter_to_skips_l123_12309


namespace arithmetic_sequence_property_l123_12327

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ d : ℝ), ∀ n, a n = a₁ + (n - 1) * d

/-- The sum of specific terms in the sequence equals 120 -/
def sum_condition (a : ℕ → ℝ) : Prop :=
  a 4 + a 6 + a 8 + a 10 + a 12 = 120

theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) (h_sum : sum_condition a) :
  2 * a 10 - a 12 = 24 := by
  sorry

end arithmetic_sequence_property_l123_12327


namespace right_triangle_345_l123_12321

theorem right_triangle_345 : ∃ (a b c : ℝ), a = 3 ∧ b = 4 ∧ c = 5 ∧ a^2 + b^2 = c^2 :=
  sorry

end right_triangle_345_l123_12321


namespace new_person_weight_l123_12330

theorem new_person_weight (initial_count : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 8 →
  weight_increase = 2.5 →
  replaced_weight = 35 →
  (initial_count : ℝ) * weight_increase + replaced_weight = 55 :=
by sorry

end new_person_weight_l123_12330

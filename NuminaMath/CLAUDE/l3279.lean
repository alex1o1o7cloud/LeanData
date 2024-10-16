import Mathlib

namespace NUMINAMATH_CALUDE_problem_solution_l3279_327969

theorem problem_solution (x y : ℝ) 
  (h1 : x + y = 12) 
  (h2 : x * y = 9) 
  (h3 : x < y) : 
  (Real.sqrt x - Real.sqrt y) / (Real.sqrt x + Real.sqrt y) = -Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3279_327969


namespace NUMINAMATH_CALUDE_pea_patch_fraction_l3279_327908

theorem pea_patch_fraction (radish_patch : ℝ) (pea_patch : ℝ) (fraction : ℝ) : 
  radish_patch = 15 →
  pea_patch = 2 * radish_patch →
  fraction * pea_patch = 5 →
  fraction = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_pea_patch_fraction_l3279_327908


namespace NUMINAMATH_CALUDE_train_length_l3279_327964

/-- Proves the length of a train given its speed, time to pass a platform, and platform length -/
theorem train_length (train_speed : ℝ) (time_to_pass : ℝ) (platform_length : ℝ) :
  train_speed = 45 * (1000 / 3600) →
  time_to_pass = 39.2 →
  platform_length = 130 →
  train_speed * time_to_pass - platform_length = 360 := by
sorry

end NUMINAMATH_CALUDE_train_length_l3279_327964


namespace NUMINAMATH_CALUDE_binomial_expansion_sum_l3279_327997

theorem binomial_expansion_sum (a₁ a₂ : ℕ) : 
  (∀ k : ℕ, k ≤ 10 → a₁ = 20 ∧ a₂ = 180) → 
  a₁ + a₂ = 200 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_sum_l3279_327997


namespace NUMINAMATH_CALUDE_square_cut_parts_l3279_327956

/-- Represents a square grid paper -/
structure GridPaper :=
  (size : ℕ)

/-- Represents a folded square -/
structure FoldedSquare :=
  (original : GridPaper)
  (folded_size : ℕ)

/-- Represents a cut on the folded square -/
inductive Cut
  | Midpoint : Cut

/-- The number of parts resulting from unfolding after the cut -/
def num_parts_after_cut (fs : FoldedSquare) (c : Cut) : ℕ :=
  fs.original.size + 1

theorem square_cut_parts :
  ∀ (gp : GridPaper) (fs : FoldedSquare) (c : Cut),
    gp.size = 8 →
    fs.original = gp →
    fs.folded_size = 1 →
    c = Cut.Midpoint →
    num_parts_after_cut fs c = 9 :=
sorry

end NUMINAMATH_CALUDE_square_cut_parts_l3279_327956


namespace NUMINAMATH_CALUDE_cookies_given_to_friend_l3279_327924

theorem cookies_given_to_friend (initial_cookies : ℕ) (eaten_cookies : ℕ) (remaining_cookies : ℕ) : 
  initial_cookies = 36 →
  eaten_cookies = 10 →
  remaining_cookies = 12 →
  initial_cookies - eaten_cookies - remaining_cookies = 14 := by
sorry

end NUMINAMATH_CALUDE_cookies_given_to_friend_l3279_327924


namespace NUMINAMATH_CALUDE_product_prs_l3279_327910

theorem product_prs (p r s : ℕ) : 
  4^p + 4^3 = 272 → 
  3^r + 27 = 81 → 
  2^s + 7^2 = 1024 → 
  p * r * s = 160 := by
sorry

end NUMINAMATH_CALUDE_product_prs_l3279_327910


namespace NUMINAMATH_CALUDE_madeline_grocery_budget_l3279_327917

/-- Calculates the amount Madeline needs for groceries given her expenses and income. -/
theorem madeline_grocery_budget 
  (rent : ℕ) 
  (medical : ℕ) 
  (utilities : ℕ) 
  (emergency : ℕ) 
  (hourly_rate : ℕ) 
  (hours_worked : ℕ) 
  (h1 : rent = 1200)
  (h2 : medical = 200)
  (h3 : utilities = 60)
  (h4 : emergency = 200)
  (h5 : hourly_rate = 15)
  (h6 : hours_worked = 138) :
  hourly_rate * hours_worked - (rent + medical + utilities + emergency) = 410 := by
  sorry

end NUMINAMATH_CALUDE_madeline_grocery_budget_l3279_327917


namespace NUMINAMATH_CALUDE_largest_solution_floor_equation_l3279_327914

theorem largest_solution_floor_equation :
  let f (x : ℝ) := ⌊x⌋ = 10 + 50 * (x - ⌊x⌋)
  ∃ (max_sol : ℝ), f max_sol ∧ max_sol = 59.98 ∧ ∀ y, f y → y ≤ max_sol :=
by sorry

end NUMINAMATH_CALUDE_largest_solution_floor_equation_l3279_327914


namespace NUMINAMATH_CALUDE_division_simplification_l3279_327928

theorem division_simplification : 240 / (12 + 12 * 2 - 3) = 240 / 33 := by
  sorry

end NUMINAMATH_CALUDE_division_simplification_l3279_327928


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l3279_327937

theorem quadratic_expression_value (x : ℝ) (h : x^2 - 2*x - 3 = 0) :
  (2*x - 1)^2 - (x - 1)*(x + 3) = 13 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l3279_327937


namespace NUMINAMATH_CALUDE_product_three_consecutive_odds_divisible_by_three_l3279_327944

theorem product_three_consecutive_odds_divisible_by_three (n : ℤ) (h : n > 0) :
  ∃ k : ℤ, (2*n + 1) * (2*n + 3) * (2*n + 5) = 3 * k :=
by
  sorry

end NUMINAMATH_CALUDE_product_three_consecutive_odds_divisible_by_three_l3279_327944


namespace NUMINAMATH_CALUDE_largest_integer_in_interval_l3279_327950

theorem largest_integer_in_interval : 
  ∃ (y : ℤ), (1/4 : ℚ) < (y : ℚ)/7 ∧ (y : ℚ)/7 < 3/5 ∧ 
  ∀ (z : ℤ), ((1/4 : ℚ) < (z : ℚ)/7 ∧ (z : ℚ)/7 < 3/5) → z ≤ y :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_largest_integer_in_interval_l3279_327950


namespace NUMINAMATH_CALUDE_new_shipment_bears_l3279_327931

def initial_stock : ℕ := 4
def bears_per_shelf : ℕ := 7
def shelves_used : ℕ := 2

theorem new_shipment_bears :
  initial_stock + (bears_per_shelf * shelves_used) - initial_stock = 10 := by
  sorry

end NUMINAMATH_CALUDE_new_shipment_bears_l3279_327931


namespace NUMINAMATH_CALUDE_marilyn_shared_bottlecaps_l3279_327987

/-- 
Given that Marilyn starts with 51 bottle caps and ends up with 15 bottle caps,
prove that she shared 36 bottle caps with Nancy.
-/
theorem marilyn_shared_bottlecaps : 
  let initial_caps : ℕ := 51
  let remaining_caps : ℕ := 15
  let shared_caps : ℕ := initial_caps - remaining_caps
  shared_caps = 36 := by sorry

end NUMINAMATH_CALUDE_marilyn_shared_bottlecaps_l3279_327987


namespace NUMINAMATH_CALUDE_gcd_n_cube_minus_27_and_n_minus_3_l3279_327963

theorem gcd_n_cube_minus_27_and_n_minus_3 (n : ℕ) (h : n > 3^2) :
  Nat.gcd (n^3 - 27) (n - 3) = n - 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_n_cube_minus_27_and_n_minus_3_l3279_327963


namespace NUMINAMATH_CALUDE_inscribed_squares_ratio_l3279_327936

/-- A right triangle with sides 5, 12, and 13 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_triangle : a^2 + b^2 = c^2
  side_a : a = 5
  side_b : b = 12
  side_c : c = 13

/-- A square inscribed in the triangle with one side on leg a -/
def inscribed_square_a (t : RightTriangle) (x : ℝ) : Prop :=
  x = t.a

/-- A square inscribed in the triangle with one side on the hypotenuse -/
def inscribed_square_c (t : RightTriangle) (y : ℝ) : Prop :=
  y / t.c = (t.b - 2*y) / t.b ∧ y / t.c = (t.a - y) / t.a

theorem inscribed_squares_ratio (t : RightTriangle) (x y : ℝ) 
  (hx : inscribed_square_a t x) (hy : inscribed_square_c t y) : 
  x / y = 18 / 13 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_squares_ratio_l3279_327936


namespace NUMINAMATH_CALUDE_expand_product_l3279_327978

theorem expand_product (x : ℝ) : (3*x - 4) * (2*x + 7) = 6*x^2 + 13*x - 28 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l3279_327978


namespace NUMINAMATH_CALUDE_quadratic_function_value_bound_l3279_327949

theorem quadratic_function_value_bound (p q : ℝ) : 
  ¬(∀ x ∈ ({1, 2, 3} : Set ℝ), |x^2 + p*x + q| < (1/2 : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_value_bound_l3279_327949


namespace NUMINAMATH_CALUDE_perpendicular_tangents_ratio_l3279_327981

theorem perpendicular_tangents_ratio (a b : ℝ) : 
  (∃ (x y : ℝ), a*x + b*y - 5 = 0 ∧ y = x^3) →  -- Line and curve equations
  (∃ (x y : ℝ), x = 1 ∧ y = 1 ∧ a*x + b*y - 5 = 0 ∧ y = x^3) →  -- Point P(1, 1) satisfies both equations
  (∀ (m₁ m₂ : ℝ), (m₁ * m₂ = -1) → 
    (m₁ = -a/b ∧ m₂ = 3 * 1^2)) →  -- Perpendicular tangent lines condition
  a/b = 1/3 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_tangents_ratio_l3279_327981


namespace NUMINAMATH_CALUDE_fifty_third_number_is_53_l3279_327962

/-- Represents the sequence of numbers spoken in the modified counting game -/
def modifiedCountingSequence : ℕ → ℕ
| 0 => 1  -- Jo starts with 1
| n + 1 => 
  let prevNum := modifiedCountingSequence n
  if prevNum % 3 = 0 then prevNum + 2  -- Skip a number after multiples of 3
  else prevNum + 1  -- Otherwise, increment by 1

/-- The 53rd number in the modified counting sequence is 53 -/
theorem fifty_third_number_is_53 : modifiedCountingSequence 52 = 53 := by
  sorry

#eval modifiedCountingSequence 52  -- Evaluates to 53

end NUMINAMATH_CALUDE_fifty_third_number_is_53_l3279_327962


namespace NUMINAMATH_CALUDE_complex_equation_roots_l3279_327955

theorem complex_equation_roots : 
  let z₁ : ℂ := 3.5 - I
  let z₂ : ℂ := -2.5 + I
  (z₁^2 - z₁ = 6 - 6*I) ∧ (z₂^2 - z₂ = 6 - 6*I) := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_roots_l3279_327955


namespace NUMINAMATH_CALUDE_f_simplification_f_value_when_cos_eq_one_fifth_f_value_at_negative_1860_degrees_l3279_327921

noncomputable section

open Real

def f (α : ℝ) : ℝ := (sin (π - α) * cos (2 * π - α) * tan (-α - π)) / (tan (-α) * sin (-π - α))

theorem f_simplification (α : ℝ) (h : π < α ∧ α < 3 * π / 2) : f α = cos α := by
  sorry

theorem f_value_when_cos_eq_one_fifth (α : ℝ) (h1 : π < α ∧ α < 3 * π / 2) (h2 : cos (α - 3 * π / 2) = 1 / 5) :
  f α = -2 * Real.sqrt 6 / 5 := by
  sorry

theorem f_value_at_negative_1860_degrees :
  f (-1860 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_f_simplification_f_value_when_cos_eq_one_fifth_f_value_at_negative_1860_degrees_l3279_327921


namespace NUMINAMATH_CALUDE_extreme_values_of_f_l3279_327900

def f (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x + 5

theorem extreme_values_of_f :
  (∃ x : ℝ, f x = 10 ∧ ∀ y : ℝ, f y ≤ f x) ∧
  (∃ x : ℝ, f x = -22 ∧ ∀ y : ℝ, f y ≥ f x) :=
sorry

end NUMINAMATH_CALUDE_extreme_values_of_f_l3279_327900


namespace NUMINAMATH_CALUDE_coin_flip_expected_value_l3279_327983

def penny : ℚ := 1
def fifty_cent : ℚ := 50
def dime : ℚ := 10
def quarter : ℚ := 25

def coin_probability : ℚ := 1 / 2

def expected_value : ℚ := 
  coin_probability * penny + 
  coin_probability * fifty_cent + 
  coin_probability * dime + 
  coin_probability * quarter

theorem coin_flip_expected_value : expected_value = 43 := by
  sorry

end NUMINAMATH_CALUDE_coin_flip_expected_value_l3279_327983


namespace NUMINAMATH_CALUDE_triangle_inequality_max_l3279_327967

theorem triangle_inequality_max (a b c x y z : ℝ) 
  (triangle : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) 
  (positive : 0 < x ∧ 0 < y ∧ 0 < z) 
  (sum_one : x + y + z = 1) : 
  a * x * y + b * y * z + c * z * x ≤ 
    (a * b * c) / (2 * a * b + 2 * b * c + 2 * c * a - a^2 - b^2 - c^2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_max_l3279_327967


namespace NUMINAMATH_CALUDE_product_remainder_theorem_l3279_327973

theorem product_remainder_theorem : (2468 * 7391 * 90523) % 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_theorem_l3279_327973


namespace NUMINAMATH_CALUDE_polynomial_roots_k_values_l3279_327958

/-- The set of all distinct possible values of k for the polynomial x^2 - kx + 36 
    with only positive integer roots -/
def possible_k_values : Set ℤ := {12, 13, 15, 20, 37}

/-- A polynomial of the form x^2 - kx + 36 -/
def polynomial (k : ℤ) (x : ℝ) : ℝ := x^2 - k*x + 36

theorem polynomial_roots_k_values :
  ∀ k : ℤ, (∃ r₁ r₂ : ℤ, r₁ > 0 ∧ r₂ > 0 ∧ 
    ∀ x : ℝ, polynomial k x = 0 ↔ x = r₁ ∨ x = r₂) ↔ 
  k ∈ possible_k_values :=
sorry

end NUMINAMATH_CALUDE_polynomial_roots_k_values_l3279_327958


namespace NUMINAMATH_CALUDE_quadratic_radicals_combination_l3279_327939

theorem quadratic_radicals_combination (a : ℝ) : 
  (∃ k : ℝ, k * (1 + a) = 4 - 2*a ∧ k > 0) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_radicals_combination_l3279_327939


namespace NUMINAMATH_CALUDE_triangle_area_ratio_l3279_327968

-- Define the triangle ABC inscribed in a unit circle
def Triangle (A B C : ℝ) := True

-- Define the area of a triangle
def area (a b c : ℝ) : ℝ := sorry

-- Define the sine function
noncomputable def sin (θ : ℝ) : ℝ := sorry

-- Theorem statement
theorem triangle_area_ratio 
  (A B C : ℝ) 
  (h : Triangle A B C) :
  area (sin A) (sin B) (sin C) = (1/4 : ℝ) * area (2 * sin A) (2 * sin B) (2 * sin C) :=
sorry

end NUMINAMATH_CALUDE_triangle_area_ratio_l3279_327968


namespace NUMINAMATH_CALUDE_complex_fraction_sum_l3279_327942

theorem complex_fraction_sum : 
  let z₁ : ℂ := (1 + Complex.I)^2 / (1 + 2*Complex.I)
  let z₂ : ℂ := (1 - Complex.I)^2 / (2 - Complex.I)
  z₁ + z₂ = (6 - 2*Complex.I) / 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_sum_l3279_327942


namespace NUMINAMATH_CALUDE_rectangle_area_change_l3279_327922

theorem rectangle_area_change 
  (l w : ℝ) 
  (hl : l > 0) 
  (hw : w > 0) : 
  let new_length := l * 1.6
  let new_width := w * 0.4
  let initial_area := l * w
  let new_area := new_length * new_width
  (new_area - initial_area) / initial_area = -0.36 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_change_l3279_327922


namespace NUMINAMATH_CALUDE_x_13_plus_inv_x_13_l3279_327965

theorem x_13_plus_inv_x_13 (x : ℝ) (hx : x ≠ 0) :
  let y := x + 1/x
  x^13 + 1/x^13 = y^13 - 13*y^11 + 65*y^9 - 156*y^7 + 182*y^5 - 91*y^3 + 13*y :=
by
  sorry

end NUMINAMATH_CALUDE_x_13_plus_inv_x_13_l3279_327965


namespace NUMINAMATH_CALUDE_sum_of_cubes_l3279_327975

theorem sum_of_cubes (a b : ℝ) (h1 : a + b = 2) (h2 : a * b = -3) : a^3 + b^3 = 26 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l3279_327975


namespace NUMINAMATH_CALUDE_modulus_of_complex_number_l3279_327925

theorem modulus_of_complex_number : 
  let z : ℂ := (1 - I) / (2 * I + 1) * I
  (∃ (k : ℝ), z = k * I) → Complex.abs z = Real.sqrt 10 / 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_number_l3279_327925


namespace NUMINAMATH_CALUDE_optimization_scheme_sales_l3279_327996

/-- Given a sequence of three terms forming an arithmetic progression with a sum of 2.46 million,
    prove that the middle term (second term) is equal to 0.82 million. -/
theorem optimization_scheme_sales (a₁ a₂ a₃ : ℝ) : 
  a₁ + a₂ + a₃ = 2.46 ∧ 
  a₂ - a₁ = a₃ - a₂ → 
  a₂ = 0.82 := by
sorry

end NUMINAMATH_CALUDE_optimization_scheme_sales_l3279_327996


namespace NUMINAMATH_CALUDE_unique_natural_solution_l3279_327991

theorem unique_natural_solution : 
  ∃! (n : ℕ), n^5 - 2*n^4 - 7*n^2 - 7*n + 3 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_natural_solution_l3279_327991


namespace NUMINAMATH_CALUDE_num_men_first_group_l3279_327935

/-- The number of men in the first group -/
def num_men : ℕ := 36

/-- The time taken by the first group to complete the work (in hours) -/
def time_first_group : ℕ := 5

/-- The number of men in the second group -/
def num_men_second_group : ℕ := 15

/-- The time taken by the second group to complete the work (in hours) -/
def time_second_group : ℕ := 12

/-- The work rate is constant across both groups -/
axiom work_rate_constant : (1 : ℚ) / (num_men * time_first_group) = 1 / (num_men_second_group * time_second_group)

theorem num_men_first_group : num_men = 36 := by
  sorry

end NUMINAMATH_CALUDE_num_men_first_group_l3279_327935


namespace NUMINAMATH_CALUDE_volume_ratio_minimum_l3279_327980

noncomputable section

/-- The volume ratio of a cone to its circumscribed cylinder, given the sine of the cone's half-angle -/
def volume_ratio (s : ℝ) : ℝ := (1 + s)^3 / (6 * s * (1 - s^2))

/-- The theorem stating that the volume ratio is minimized when sin(θ) = 1/3 -/
theorem volume_ratio_minimum :
  ∀ s : ℝ, 0 < s → s < 1 →
  volume_ratio s ≥ 4/3 ∧
  (volume_ratio s = 4/3 ↔ s = 1/3) :=
sorry

end

end NUMINAMATH_CALUDE_volume_ratio_minimum_l3279_327980


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3279_327912

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) :
  (∀ x : ℝ, (x^2 - 3*x + 1)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + 
                                a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9 + a₁₀*x^10) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ = -2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3279_327912


namespace NUMINAMATH_CALUDE_decagon_triangle_probability_l3279_327913

/-- The number of vertices in a regular decagon -/
def decagon_vertices : ℕ := 10

/-- The number of vertices required to form a triangle -/
def triangle_vertices : ℕ := 3

/-- The total number of possible triangles formed from the decagon -/
def total_triangles : ℕ := Nat.choose decagon_vertices triangle_vertices

/-- The number of triangles with at least one side being a side of the decagon -/
def favorable_triangles : ℕ := 70

/-- The probability of forming a triangle with at least one side being a side of the decagon -/
def probability : ℚ := favorable_triangles / total_triangles

theorem decagon_triangle_probability :
  probability = 7 / 12 := by sorry

end NUMINAMATH_CALUDE_decagon_triangle_probability_l3279_327913


namespace NUMINAMATH_CALUDE_divisibility_property_l3279_327946

theorem divisibility_property (a b : ℕ+) : ∃ n : ℕ+, (a : ℕ) ∣ (b : ℕ)^(n : ℕ) - (n : ℕ) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_property_l3279_327946


namespace NUMINAMATH_CALUDE_thermostat_adjustment_l3279_327918

theorem thermostat_adjustment (x : ℝ) : 
  let initial_temp := 40
  let jerry_temp := 2 * initial_temp
  let dad_temp := jerry_temp - x
  let mom_temp := dad_temp * 0.7
  let sister_temp := mom_temp + 24
  sister_temp = 59 → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_thermostat_adjustment_l3279_327918


namespace NUMINAMATH_CALUDE_f_has_unique_root_l3279_327954

noncomputable def f (x : ℝ) := Real.exp x + x - 2

theorem f_has_unique_root :
  ∃! x, f x = 0 :=
sorry

end NUMINAMATH_CALUDE_f_has_unique_root_l3279_327954


namespace NUMINAMATH_CALUDE_total_dress_designs_l3279_327951

/-- The number of available fabric colors -/
def num_colors : ℕ := 5

/-- The number of available patterns -/
def num_patterns : ℕ := 6

/-- The number of available sizes -/
def num_sizes : ℕ := 3

/-- Theorem stating the total number of possible dress designs -/
theorem total_dress_designs : num_colors * num_patterns * num_sizes = 90 := by
  sorry

end NUMINAMATH_CALUDE_total_dress_designs_l3279_327951


namespace NUMINAMATH_CALUDE_smallest_perimeter_l3279_327976

/-- A triangle with side lengths that are three consecutive integers starting from 3 -/
structure ConsecutiveIntegerTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  consecutive : b = a + 1 ∧ c = b + 1
  start_from_three : a = 3

/-- The perimeter of a triangle -/
def perimeter (t : ConsecutiveIntegerTriangle) : ℕ :=
  t.a + t.b + t.c

/-- Theorem: The perimeter of a triangle with side lengths 3, 4, and 5 is 12 units -/
theorem smallest_perimeter (t : ConsecutiveIntegerTriangle) : perimeter t = 12 := by
  sorry

end NUMINAMATH_CALUDE_smallest_perimeter_l3279_327976


namespace NUMINAMATH_CALUDE_addition_equality_l3279_327992

theorem addition_equality : 12 + 36 = 48 := by
  sorry

end NUMINAMATH_CALUDE_addition_equality_l3279_327992


namespace NUMINAMATH_CALUDE_simplify_square_root_l3279_327943

theorem simplify_square_root (x y : ℝ) (h : x * y < 0) :
  x * Real.sqrt (-y / x^2) = Real.sqrt (-y) := by
  sorry

end NUMINAMATH_CALUDE_simplify_square_root_l3279_327943


namespace NUMINAMATH_CALUDE_problem_solution_l3279_327974

theorem problem_solution : 
  let M := 2024 / 4
  let N := M / 2
  let X := M + N
  X = 759 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l3279_327974


namespace NUMINAMATH_CALUDE_salesman_commission_l3279_327984

/-- Calculates the total commission for a salesman given the commission rates and bonus amount. -/
theorem salesman_commission
  (base_commission_rate : Real)
  (bonus_commission_rate : Real)
  (bonus_threshold : Real)
  (bonus_amount : Real) :
  base_commission_rate = 0.09 →
  bonus_commission_rate = 0.03 →
  bonus_threshold = 10000 →
  bonus_amount = 120 →
  ∃ (total_sales : Real),
    total_sales > bonus_threshold ∧
    bonus_commission_rate * (total_sales - bonus_threshold) = bonus_amount ∧
    base_commission_rate * total_sales + bonus_amount = 1380 :=
by sorry

end NUMINAMATH_CALUDE_salesman_commission_l3279_327984


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3279_327952

theorem quadratic_equation_solution (w : ℝ) :
  (w + 15)^2 = (4*w + 9) * (3*w + 6) →
  w^2 = (((-21 + Real.sqrt 7965) / 22)^2) ∨ w^2 = (((-21 - Real.sqrt 7965) / 22)^2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3279_327952


namespace NUMINAMATH_CALUDE_triangle_side_sum_max_l3279_327989

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    M is the midpoint of BC, AM = 1, and c*cos(B) + b*cos(C) = 2a*cos(A),
    prove that the maximum value of b + c is 4√3/3 -/
theorem triangle_side_sum_max (a b c : ℝ) (A B C : ℝ) (M : ℝ × ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  A + B + C = π →
  c * Real.cos B + b * Real.cos C = 2 * a * Real.cos A →
  M = ((b * Real.cos C) / (b + c), (c * Real.cos B) / (b + c)) →
  Real.sqrt ((M.1 - 1)^2 + M.2^2) = 1 →
  (∀ b' c', b' > 0 ∧ c' > 0 ∧ 
    c' * Real.cos B + b' * Real.cos C = 2 * a * Real.cos A ∧
    Real.sqrt (((b' * Real.cos C) / (b' + c') - 1)^2 + ((c' * Real.cos B) / (b' + c'))^2) = 1 →
    b' + c' ≤ b + c) →
  b + c = 4 * Real.sqrt 3 / 3 :=
sorry

end NUMINAMATH_CALUDE_triangle_side_sum_max_l3279_327989


namespace NUMINAMATH_CALUDE_function_inequality_implies_constant_bound_l3279_327901

open Real MeasureTheory

theorem function_inequality_implies_constant_bound 
  (f g : ℝ → ℝ) (a : ℝ) :
  (∀ x > 0, f x < g x) →
  (∀ x > 0, f x = a * exp (x / 2) - x) →
  (∀ x > 0, g x = x * log x - (1 / 2) * x^2) →
  a < -exp (-2) := by
sorry

end NUMINAMATH_CALUDE_function_inequality_implies_constant_bound_l3279_327901


namespace NUMINAMATH_CALUDE_min_students_solved_both_l3279_327970

theorem min_students_solved_both (total : ℕ) (first : ℕ) (second : ℕ) :
  total = 30 →
  first = 21 →
  second = 18 →
  ∃ (both : ℕ), both ≥ 9 ∧
    both ≤ first ∧
    both ≤ second ∧
    (∀ (x : ℕ), x < both → x + (first - x) + (second - x) > total) :=
by sorry

end NUMINAMATH_CALUDE_min_students_solved_both_l3279_327970


namespace NUMINAMATH_CALUDE_parabola_properties_l3279_327972

/-- Represents a parabola of the form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The y-coordinate for a given x on the parabola -/
def Parabola.y_at (p : Parabola) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

/-- Predicate to check if a point (x, y) is on the parabola -/
def Parabola.contains_point (p : Parabola) (x y : ℝ) : Prop :=
  p.y_at x = y

theorem parabola_properties (p : Parabola) 
  (h1 : p.contains_point (-1) 3)
  (h2 : p.contains_point 0 0)
  (h3 : p.contains_point 1 (-1))
  (h4 : p.contains_point 2 0)
  (h5 : p.contains_point 3 3) :
  (∃ x_sym : ℝ, x_sym = 1 ∧ ∀ x : ℝ, p.y_at (x_sym - x) = p.y_at (x_sym + x)) ∧ 
  (p.a > 0) ∧
  (∀ x y : ℝ, x < 0 ∧ y < 0 → ¬p.contains_point x y) :=
sorry

end NUMINAMATH_CALUDE_parabola_properties_l3279_327972


namespace NUMINAMATH_CALUDE_garden_fence_length_l3279_327909

theorem garden_fence_length (side_length : ℝ) (h : side_length = 28) : 
  4 * side_length = 112 := by
  sorry

end NUMINAMATH_CALUDE_garden_fence_length_l3279_327909


namespace NUMINAMATH_CALUDE_x_value_l3279_327940

theorem x_value (x y : ℤ) (h1 : x + y = 4) (h2 : x - y = 36) : x = 20 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l3279_327940


namespace NUMINAMATH_CALUDE_polynomial_factor_value_theorem_l3279_327915

theorem polynomial_factor_value_theorem (h k : ℝ) : 
  (∀ x : ℝ, (x + 2) * (x - 1) * (x + 3) ∣ (3 * x^4 - 2 * h * x^2 + h * x + k)) →
  |3 * h - 2 * k| = 11 := by
sorry

end NUMINAMATH_CALUDE_polynomial_factor_value_theorem_l3279_327915


namespace NUMINAMATH_CALUDE_evaluate_expression_l3279_327906

theorem evaluate_expression (x y z : ℚ) (hx : x = 1/2) (hy : y = 1/3) (hz : z = 2) :
  (x^3 * y^4 * z)^2 = 1/104976 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3279_327906


namespace NUMINAMATH_CALUDE_janet_walk_time_l3279_327993

-- Define Janet's walking pattern
def blocks_north : ℕ := 3
def blocks_west : ℕ := 7 * blocks_north
def blocks_south : ℕ := 8
def blocks_east : ℕ := 2 * blocks_south

-- Define Janet's walking speed
def blocks_per_minute : ℕ := 2

-- Calculate net distance from home
def net_south : ℤ := blocks_south - blocks_north
def net_west : ℤ := blocks_west - blocks_east

-- Total distance to walk home
def total_distance : ℕ := (net_south.natAbs + net_west.natAbs : ℕ)

-- Time to walk home
def time_to_home : ℚ := total_distance / blocks_per_minute

-- Theorem to prove
theorem janet_walk_time : time_to_home = 5 := by
  sorry

end NUMINAMATH_CALUDE_janet_walk_time_l3279_327993


namespace NUMINAMATH_CALUDE_ratio_of_divisor_sums_l3279_327927

def M : ℕ := 36 * 36 * 75 * 224

def sum_odd_divisors (n : ℕ) : ℕ := sorry
def sum_even_divisors (n : ℕ) : ℕ := sorry

theorem ratio_of_divisor_sums :
  (sum_odd_divisors M : ℚ) / (sum_even_divisors M : ℚ) = 1 / 510 := by sorry

end NUMINAMATH_CALUDE_ratio_of_divisor_sums_l3279_327927


namespace NUMINAMATH_CALUDE_inscribed_triangle_properties_l3279_327998

-- Define the triangle ABC
structure Triangle where
  A : ℝ  -- Angle A
  B : ℝ  -- Angle B
  C : ℝ  -- Angle C
  a : ℝ  -- Side opposite to A
  b : ℝ  -- Side opposite to B
  c : ℝ  -- Side opposite to C

-- Define the inscribed circle
structure InscribedCircle where
  O : ℝ × ℝ  -- Center of the inscribed circle
  r : ℝ      -- Radius of the inscribed circle

-- Define the intersection points
structure IntersectionPoints where
  A' : ℝ × ℝ
  B' : ℝ × ℝ
  C' : ℝ × ℝ

-- Define ζ as a function of the inscribed circle's radius
def ζ (r : ℝ) : ℝ := sorry

-- Theorem statement
theorem inscribed_triangle_properties
  (abc : Triangle)
  (insc : InscribedCircle)
  (int_points : IntersectionPoints) :
  let a' := 2 * ζ insc.r * Real.cos ((abc.B + abc.C) / 4)
  let b' := 2 * ζ insc.r * Real.cos ((abc.C + abc.A) / 4)
  let c' := 2 * ζ insc.r * Real.cos ((abc.A + abc.B) / 4)
  let T := 2 * (ζ insc.r)^2 * Real.cos ((abc.B + abc.C) / 4) *
           Real.cos ((abc.C + abc.A) / 4) * Real.cos ((abc.A + abc.B) / 4)
  (∃ (a'_actual b'_actual c'_actual : ℝ),
    a'_actual = a' ∧ b'_actual = b' ∧ c'_actual = c') ∧
  (∃ (T_actual : ℝ), T_actual = T) :=
by sorry

end NUMINAMATH_CALUDE_inscribed_triangle_properties_l3279_327998


namespace NUMINAMATH_CALUDE_min_sum_of_digits_3n2_plus_n_plus_1_l3279_327990

/-- Sum of digits of a natural number -/
def sum_of_digits (m : ℕ) : ℕ :=
  if m < 10 then m else m % 10 + sum_of_digits (m / 10)

/-- The main theorem -/
theorem min_sum_of_digits_3n2_plus_n_plus_1 :
  (∀ n : ℕ+, sum_of_digits (3 * n^2 + n + 1) ≥ 3) ∧
  (∃ n : ℕ+, sum_of_digits (3 * n^2 + n + 1) = 3) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_digits_3n2_plus_n_plus_1_l3279_327990


namespace NUMINAMATH_CALUDE_paper_fold_crease_length_l3279_327986

/-- Given a rectangular paper of width 8 inches, when folded so that the bottom right corner 
    touches the left edge dividing it in a 1:2 ratio, the length of the crease L is equal to 
    16/3 csc θ, where θ is the angle between the crease and the bottom edge. -/
theorem paper_fold_crease_length (width : ℝ) (θ : ℝ) (L : ℝ) :
  width = 8 →
  0 < θ → θ < π / 2 →
  L = (16 / 3) * (1 / Real.sin θ) := by
  sorry

end NUMINAMATH_CALUDE_paper_fold_crease_length_l3279_327986


namespace NUMINAMATH_CALUDE_range_of_a_l3279_327947

def A (a : ℝ) : Set ℝ := {x | (x - 1) * (x - a) ≥ 0}
def B (a : ℝ) : Set ℝ := {x | x ≥ a - 1}

theorem range_of_a (a : ℝ) (h : A a ∪ B a = Set.univ) : a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3279_327947


namespace NUMINAMATH_CALUDE_boarding_students_change_l3279_327977

theorem boarding_students_change (initial : ℝ) (increase_rate : ℝ) (decrease_rate : ℝ) 
  (h1 : increase_rate = 0.2) 
  (h2 : decrease_rate = 0.2) : 
  initial * (1 + increase_rate) * (1 - decrease_rate) = initial * 0.96 :=
by sorry

end NUMINAMATH_CALUDE_boarding_students_change_l3279_327977


namespace NUMINAMATH_CALUDE_candy_box_count_l3279_327902

/-- Given 2 boxes of chocolate candy and 5 boxes of caramel candy,
    with the same number of pieces in each box,
    and a total of 28 candies, prove that there are 4 pieces in each box. -/
theorem candy_box_count (chocolate_boxes : ℕ) (caramel_boxes : ℕ) (total_candies : ℕ) 
    (h1 : chocolate_boxes = 2)
    (h2 : caramel_boxes = 5)
    (h3 : total_candies = 28)
    (h4 : ∃ (pieces_per_box : ℕ), chocolate_boxes * pieces_per_box + caramel_boxes * pieces_per_box = total_candies) :
  ∃ (pieces_per_box : ℕ), pieces_per_box = 4 ∧ 
    chocolate_boxes * pieces_per_box + caramel_boxes * pieces_per_box = total_candies :=
by
  sorry


end NUMINAMATH_CALUDE_candy_box_count_l3279_327902


namespace NUMINAMATH_CALUDE_find_b_find_perimeter_l3279_327920

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Define the conditions
def triangle_condition (t : Triangle) : Prop :=
  t.a * Real.cos t.B = (3 * t.c - t.b) * Real.cos t.A

def side_condition (t : Triangle) : Prop :=
  t.a * Real.sin t.B = 2 * Real.sqrt 2

def area_condition (t : Triangle) : Prop :=
  1/2 * t.a * t.b * Real.sin t.C = Real.sqrt 2

-- Theorem 1
theorem find_b (t : Triangle) 
  (h1 : triangle_condition t) 
  (h2 : side_condition t) : 
  t.b = 3 :=
sorry

-- Theorem 2
theorem find_perimeter (t : Triangle) 
  (h1 : t.a = 2 * Real.sqrt 2) 
  (h2 : area_condition t) : 
  t.a + t.b + t.c = 4 + 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_find_b_find_perimeter_l3279_327920


namespace NUMINAMATH_CALUDE_initial_ratio_is_four_to_one_l3279_327979

-- Define the total initial volume of the mixture
def total_volume : ℚ := 60

-- Define the volume of water added
def added_water : ℚ := 60

-- Define the ratio of milk to water after adding water
def new_ratio : ℚ := 1 / 2

-- Theorem statement
theorem initial_ratio_is_four_to_one :
  ∀ (initial_milk initial_water : ℚ),
    initial_milk + initial_water = total_volume →
    initial_milk / (initial_water + added_water) = new_ratio →
    initial_milk / initial_water = 4 := by
  sorry

end NUMINAMATH_CALUDE_initial_ratio_is_four_to_one_l3279_327979


namespace NUMINAMATH_CALUDE_symmetric_matrix_square_sum_l3279_327905

theorem symmetric_matrix_square_sum (x y z : ℝ) : 
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![x, y; y, z]
  (∀ i j, B i j = B j i) →  -- B is symmetric
  B * B = (1 : Matrix (Fin 2) (Fin 2) ℝ) →  -- B^2 = I
  x^2 + 2*y^2 + z^2 = 2 := by
sorry

end NUMINAMATH_CALUDE_symmetric_matrix_square_sum_l3279_327905


namespace NUMINAMATH_CALUDE_area_of_specific_quadrilateral_l3279_327903

/-- Represents a quadrilateral EFGH with specific angle and side length properties -/
structure Quadrilateral :=
  (EF : ℝ)
  (FG : ℝ)
  (GH : ℝ)
  (angle_F : ℝ)
  (angle_G : ℝ)

/-- Calculates the area of the quadrilateral EFGH -/
def area (q : Quadrilateral) : ℝ :=
  sorry

/-- Theorem stating that for a quadrilateral EFGH with given properties, its area is (77√2)/4 -/
theorem area_of_specific_quadrilateral :
  ∀ (q : Quadrilateral),
    q.EF = 5 ∧
    q.FG = 7 ∧
    q.GH = 6 ∧
    q.angle_F = 135 ∧
    q.angle_G = 135 →
    area q = (77 * Real.sqrt 2) / 4 :=
by sorry

end NUMINAMATH_CALUDE_area_of_specific_quadrilateral_l3279_327903


namespace NUMINAMATH_CALUDE_circle_diameter_ratio_l3279_327945

theorem circle_diameter_ratio (R S : ℝ) (harea : π * R^2 = 0.04 * π * S^2) :
  2 * R = 0.4 * (2 * S) := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_ratio_l3279_327945


namespace NUMINAMATH_CALUDE_alices_favorite_number_l3279_327938

def is_multiple (a b : ℕ) : Prop := ∃ k, a = b * k

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

theorem alices_favorite_number :
  ∃! n : ℕ, 100 ≤ n ∧ n ≤ 200 ∧
  is_multiple n 13 ∧
  ¬is_multiple n 3 ∧
  is_multiple (digit_sum n) 5 ∧
  n = 104 := by
sorry

end NUMINAMATH_CALUDE_alices_favorite_number_l3279_327938


namespace NUMINAMATH_CALUDE_age_puzzle_l3279_327995

theorem age_puzzle (A : ℕ) (x : ℕ) (h1 : A = 24) (h2 : x = 3) :
  4 * (A + x) - 4 * (A - 3) = A := by
  sorry

end NUMINAMATH_CALUDE_age_puzzle_l3279_327995


namespace NUMINAMATH_CALUDE_bridge_length_l3279_327923

/-- The length of the bridge given train crossing times and train length -/
theorem bridge_length
  (train_length : ℝ)
  (bridge_crossing_time : ℝ)
  (lamppost_crossing_time : ℝ)
  (h1 : train_length = 600)
  (h2 : bridge_crossing_time = 70)
  (h3 : lamppost_crossing_time = 20) :
  ∃ (bridge_length : ℝ), bridge_length = 1500 := by
  sorry


end NUMINAMATH_CALUDE_bridge_length_l3279_327923


namespace NUMINAMATH_CALUDE_x_intercepts_count_l3279_327941

theorem x_intercepts_count : 
  (⌊(20000 : ℝ) / Real.pi⌋ : ℤ) - (⌊(2000 : ℝ) / Real.pi⌋ : ℤ) = 5729 := by sorry

end NUMINAMATH_CALUDE_x_intercepts_count_l3279_327941


namespace NUMINAMATH_CALUDE_consecutive_odd_numbers_problem_l3279_327961

theorem consecutive_odd_numbers_problem (x : ℤ) : 
  (∃ (y z : ℤ), y = x + 2 ∧ z = x + 4 ∧ 
   x % 2 = 1 ∧ y % 2 = 1 ∧ z % 2 = 1 ∧
   8 * x = 3 * z + 2 * y + 5) →
  x = 7 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_odd_numbers_problem_l3279_327961


namespace NUMINAMATH_CALUDE_clown_balloon_count_l3279_327948

/-- The number of balloons a clown has after a series of actions -/
def final_balloon_count (initial : ℕ) (additional : ℕ) (given_away : ℕ) : ℕ :=
  initial + additional - given_away

/-- Theorem stating that the clown has 149 balloons at the end -/
theorem clown_balloon_count :
  final_balloon_count 123 53 27 = 149 := by
  sorry

end NUMINAMATH_CALUDE_clown_balloon_count_l3279_327948


namespace NUMINAMATH_CALUDE_root_sum_squares_equality_l3279_327971

theorem root_sum_squares_equality (a b : ℝ) : 
  (∃ x y : ℝ, x^2 + a*x + b = 0 ∧ y^2 + b*y + a = 0) →  -- both equations have real roots
  (∃ p q r s : ℝ, p^2 + q^2 = r^2 + s^2 ∧               -- sum of squares of roots are equal
                  p^2 + a*p + b = 0 ∧ q^2 + a*q + b = 0 ∧ 
                  r^2 + b*r + a = 0 ∧ s^2 + b*s + a = 0) →
  a ≠ b →                                               -- a is not equal to b
  a + b = -2 :=                                         -- conclusion
by sorry

end NUMINAMATH_CALUDE_root_sum_squares_equality_l3279_327971


namespace NUMINAMATH_CALUDE_cube_edge_length_from_circumscribed_sphere_volume_l3279_327999

theorem cube_edge_length_from_circumscribed_sphere_volume :
  ∀ (edge_length : ℝ) (sphere_volume : ℝ),
    sphere_volume = 4 * Real.pi / 3 →
    (∃ (sphere_radius : ℝ),
      sphere_volume = 4 / 3 * Real.pi * sphere_radius ^ 3 ∧
      edge_length ^ 2 * 3 = (2 * sphere_radius) ^ 2) →
    edge_length = 2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_length_from_circumscribed_sphere_volume_l3279_327999


namespace NUMINAMATH_CALUDE_canoe_upstream_speed_l3279_327959

/-- The speed of a canoe rowing upstream, given its downstream speed and the stream speed -/
theorem canoe_upstream_speed (downstream_speed stream_speed : ℝ) : 
  downstream_speed = 10 → stream_speed = 2 → 
  downstream_speed - 2 * stream_speed = 6 := by sorry

end NUMINAMATH_CALUDE_canoe_upstream_speed_l3279_327959


namespace NUMINAMATH_CALUDE_prob_B_given_A₁_pairwise_mutually_exclusive_l3279_327953

-- Define the number of balls in each can
def can_A_red : ℕ := 5
def can_A_white : ℕ := 2
def can_A_black : ℕ := 3
def can_B_red : ℕ := 4
def can_B_white : ℕ := 3
def can_B_black : ℕ := 3

-- Define the total number of balls in each can
def total_A : ℕ := can_A_red + can_A_white + can_A_black
def total_B : ℕ := can_B_red + can_B_white + can_B_black

-- Define the events
def A₁ : Set ℕ := {x | x ≤ can_A_red}
def A₂ : Set ℕ := {x | can_A_red < x ∧ x ≤ can_A_red + can_A_white}
def A₃ : Set ℕ := {x | can_A_red + can_A_white < x ∧ x ≤ total_A}
def B : Set ℕ := {x | x ≤ can_B_red + 1}

-- Define the probability measure
noncomputable def P : Set ℕ → ℝ := sorry

-- Theorem 1: P(B|A₁) = 5/11
theorem prob_B_given_A₁ : P (B ∩ A₁) / P A₁ = 5 / 11 := by sorry

-- Theorem 2: A₁, A₂, A₃ are pairwise mutually exclusive
theorem pairwise_mutually_exclusive : 
  (A₁ ∩ A₂ = ∅) ∧ (A₁ ∩ A₃ = ∅) ∧ (A₂ ∩ A₃ = ∅) := by sorry

end NUMINAMATH_CALUDE_prob_B_given_A₁_pairwise_mutually_exclusive_l3279_327953


namespace NUMINAMATH_CALUDE_square_rectangle_area_relation_l3279_327929

theorem square_rectangle_area_relation :
  ∀ x : ℝ,
  let square_side := x - 3
  let rect_length := x - 2
  let rect_width := x + 5
  let square_area := square_side ^ 2
  let rect_area := rect_length * rect_width
  rect_area = 3 * square_area →
  ∃ x₁ x₂ : ℝ, 
    (x₁ + x₂ = 21/2) ∧ 
    (∀ y : ℝ, rect_area = 3 * square_area → y = x₁ ∨ y = x₂) :=
by
  sorry

#check square_rectangle_area_relation

end NUMINAMATH_CALUDE_square_rectangle_area_relation_l3279_327929


namespace NUMINAMATH_CALUDE_max_train_collection_l3279_327966

/-- The number of trains Max receives each year --/
def trains_per_year : ℕ := 3

/-- The number of years Max collects trains --/
def collection_years : ℕ := 5

/-- The total number of trains Max has after the collection period --/
def initial_trains : ℕ := trains_per_year * collection_years

/-- The factor by which Max's train collection is multiplied at the end --/
def doubling_factor : ℕ := 2

/-- The final number of trains Max has --/
def final_trains : ℕ := initial_trains * doubling_factor

theorem max_train_collection :
  final_trains = 30 :=
sorry

end NUMINAMATH_CALUDE_max_train_collection_l3279_327966


namespace NUMINAMATH_CALUDE_fly_ceiling_distance_l3279_327988

-- Define the room and fly position
def room_fly_distance (wall1_distance wall2_distance point_p_distance : ℝ) : Prop :=
  ∃ (ceiling_distance : ℝ),
    wall1_distance = 2 ∧
    wall2_distance = 7 ∧
    point_p_distance = 10 ∧
    ceiling_distance^2 + wall1_distance^2 + wall2_distance^2 = point_p_distance^2

-- Theorem statement
theorem fly_ceiling_distance :
  ∀ (wall1_distance wall2_distance point_p_distance ceiling_distance : ℝ),
    room_fly_distance wall1_distance wall2_distance point_p_distance →
    ceiling_distance = Real.sqrt 47 := by
  sorry

end NUMINAMATH_CALUDE_fly_ceiling_distance_l3279_327988


namespace NUMINAMATH_CALUDE_muffin_banana_price_ratio_l3279_327916

/-- The price of a muffin -/
def muffin_price : ℝ := sorry

/-- The price of a banana -/
def banana_price : ℝ := sorry

/-- Elaine's total expenditure -/
def elaine_total : ℝ := 5 * muffin_price + 4 * banana_price

/-- Derek's total expenditure -/
def derek_total : ℝ := 3 * muffin_price + 18 * banana_price

/-- Derek spends three times as much as Elaine -/
axiom derek_spends_triple : derek_total = 3 * elaine_total

theorem muffin_banana_price_ratio : muffin_price = 2 * banana_price := by
  sorry

end NUMINAMATH_CALUDE_muffin_banana_price_ratio_l3279_327916


namespace NUMINAMATH_CALUDE_average_of_remaining_numbers_l3279_327919

theorem average_of_remaining_numbers
  (total : ℕ)
  (avg_all : ℚ)
  (avg_first_two : ℚ)
  (avg_next_two : ℚ)
  (h_total : total = 6)
  (h_avg_all : avg_all = 3.95)
  (h_avg_first_two : avg_first_two = 3.6)
  (h_avg_next_two : avg_next_two = 3.85) :
  (total * avg_all - 2 * avg_first_two - 2 * avg_next_two) / 2 = 4.4 := by
  sorry

end NUMINAMATH_CALUDE_average_of_remaining_numbers_l3279_327919


namespace NUMINAMATH_CALUDE_hcf_of_156_324_672_l3279_327904

theorem hcf_of_156_324_672 : Nat.gcd 156 (Nat.gcd 324 672) = 12 := by
  sorry

end NUMINAMATH_CALUDE_hcf_of_156_324_672_l3279_327904


namespace NUMINAMATH_CALUDE_choose_two_from_four_eq_six_l3279_327932

def choose_two_from_four : ℕ := sorry

theorem choose_two_from_four_eq_six : choose_two_from_four = 6 := by sorry

end NUMINAMATH_CALUDE_choose_two_from_four_eq_six_l3279_327932


namespace NUMINAMATH_CALUDE_smallest_angle_sum_l3279_327994

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angle_A : ℝ
  angle_B : ℝ
  angle_C : ℝ
  sum_angles : angle_A + angle_B + angle_C = 180
  scalene : a ≠ b ∧ b ≠ c ∧ a ≠ c

-- Define the problem conditions
def problem_triangle (t : Triangle) : Prop :=
  (t.angle_A = 45 ∨ t.angle_B = 45 ∨ t.angle_C = 45) ∧
  (180 - t.angle_A = 135 ∨ 180 - t.angle_B = 135 ∨ 180 - t.angle_C = 135)

-- Theorem statement
theorem smallest_angle_sum (t : Triangle) (h : problem_triangle t) :
  ∃ x y, x ≤ t.angle_A ∧ x ≤ t.angle_B ∧ x ≤ t.angle_C ∧
         y ≤ t.angle_A ∧ y ≤ t.angle_B ∧ y ≤ t.angle_C ∧
         x + y = 90 :=
sorry

end NUMINAMATH_CALUDE_smallest_angle_sum_l3279_327994


namespace NUMINAMATH_CALUDE_fewer_cards_l3279_327907

/-- The number of soccer cards Chris has -/
def chris_cards : ℕ := 18

/-- The number of soccer cards Charlie has -/
def charlie_cards : ℕ := 32

/-- The difference in the number of cards between Charlie and Chris -/
def card_difference : ℕ := charlie_cards - chris_cards

theorem fewer_cards : card_difference = 14 := by
  sorry

end NUMINAMATH_CALUDE_fewer_cards_l3279_327907


namespace NUMINAMATH_CALUDE_units_digit_sum_of_powers_l3279_327982

theorem units_digit_sum_of_powers : (24^4 + 42^4 + 24^2 + 42^2) % 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_sum_of_powers_l3279_327982


namespace NUMINAMATH_CALUDE_student_assignment_count_l3279_327957

theorem student_assignment_count : ∀ (n m : ℕ),
  n = 4 ∧ m = 3 →
  (Nat.choose n 2 * (Nat.factorial m)) = (m * Nat.choose n 2 * 2) :=
by sorry

end NUMINAMATH_CALUDE_student_assignment_count_l3279_327957


namespace NUMINAMATH_CALUDE_extreme_point_property_l3279_327985

def f (a b x : ℝ) : ℝ := x^3 - a*x - b

theorem extreme_point_property (a b x₀ x₁ : ℝ) :
  (∃ (ε : ℝ), ε > 0 ∧ ∀ x, x ≠ x₀ ∧ |x - x₀| < ε → f a b x ≠ f a b x₀) →
  x₁ ≠ x₀ →
  f a b x₁ = f a b x₀ →
  x₁ + 2*x₀ = 0 := by
sorry

end NUMINAMATH_CALUDE_extreme_point_property_l3279_327985


namespace NUMINAMATH_CALUDE_angle_quadrant_from_point_l3279_327960

def point_in_third_quadrant (x y : ℝ) : Prop := x < 0 ∧ y < 0

def angle_in_fourth_quadrant (α : ℝ) : Prop := 
  3 * Real.pi / 2 < α ∧ α < 2 * Real.pi

theorem angle_quadrant_from_point (α : ℝ) :
  point_in_third_quadrant (Real.sin α) (Real.tan α) →
  angle_in_fourth_quadrant α := by
  sorry

end NUMINAMATH_CALUDE_angle_quadrant_from_point_l3279_327960


namespace NUMINAMATH_CALUDE_two_correct_statements_l3279_327911

theorem two_correct_statements (a b : ℝ) 
  (h : (a - Real.sqrt (a^2 - 1)) * (b - Real.sqrt (b^2 - 1)) = 1) :
  (a = b ∧ a * b = 1) ∧ 
  (a + b ≠ 0 ∧ a * b ≠ -1) :=
by sorry

end NUMINAMATH_CALUDE_two_correct_statements_l3279_327911


namespace NUMINAMATH_CALUDE_heartsuit_calculation_l3279_327930

-- Define the heartsuit operation
def heartsuit (x y : ℝ) : ℝ := x^2 - y^2

-- Theorem statement
theorem heartsuit_calculation :
  heartsuit 3 (heartsuit 4 5) = -72 := by
  sorry

end NUMINAMATH_CALUDE_heartsuit_calculation_l3279_327930


namespace NUMINAMATH_CALUDE_alex_win_probability_l3279_327933

-- Define the game conditions
def standardDie : ℕ := 6
def evenNumbers : Set ℕ := {2, 4, 6}

-- Define the probability of Kelvin winning on first roll
def kelvinFirstWinProb : ℚ := 1 / 2

-- Define the probability of Alex winning on first roll, given Kelvin didn't win
def alexFirstWinProb : ℚ := 2 / 3

-- Define the probability of Kelvin winning on second roll, given Alex didn't win on first
def kelvinSecondWinProb : ℚ := 5 / 6

-- Define the probability of Alex winning on second roll, given Kelvin didn't win on second
def alexSecondWinProb : ℚ := 2 / 3

-- State the theorem
theorem alex_win_probability :
  let totalAlexWinProb := 
    kelvinFirstWinProb * alexFirstWinProb + 
    (1 - kelvinFirstWinProb) * (1 - alexFirstWinProb) * (1 - kelvinSecondWinProb) * alexSecondWinProb
  totalAlexWinProb = 22 / 27 := by
  sorry

end NUMINAMATH_CALUDE_alex_win_probability_l3279_327933


namespace NUMINAMATH_CALUDE_skittles_pencils_difference_l3279_327934

def number_of_children : ℕ := 17
def pencils_per_child : ℕ := 3
def skittles_per_child : ℕ := 18

theorem skittles_pencils_difference :
  (number_of_children * skittles_per_child) - (number_of_children * pencils_per_child) = 255 := by
  sorry

end NUMINAMATH_CALUDE_skittles_pencils_difference_l3279_327934


namespace NUMINAMATH_CALUDE_janet_return_time_l3279_327926

/-- Represents the walking pattern of Janet in a grid system -/
structure WalkingPattern where
  north : ℕ
  west : ℕ
  south : ℕ
  east : ℕ

/-- Calculates the time taken to return home given a walking pattern and speed -/
def timeToReturnHome (pattern : WalkingPattern) (speed : ℕ) : ℕ :=
  let net_north := pattern.south - pattern.north
  let net_west := pattern.west - pattern.east
  let total_blocks := net_north + net_west
  total_blocks / speed

/-- Janet's specific walking pattern -/
def janetsPattern : WalkingPattern :=
  { north := 3
  , west := 7 * 3
  , south := 8
  , east := 2 * 8 }

/-- Janet's walking speed in blocks per minute -/
def janetsSpeed : ℕ := 2

/-- Theorem stating that it takes Janet 5 minutes to return home -/
theorem janet_return_time :
  timeToReturnHome janetsPattern janetsSpeed = 5 := by
  sorry

end NUMINAMATH_CALUDE_janet_return_time_l3279_327926

import Mathlib

namespace dorothy_age_proof_l933_93395

/-- Given Dorothy's age relationships with her sister, prove Dorothy's current age --/
theorem dorothy_age_proof (dorothy_age sister_age : ℕ) : 
  sister_age = 5 →
  dorothy_age = 3 * sister_age →
  dorothy_age + 5 = 2 * (sister_age + 5) →
  dorothy_age = 15 := by
  sorry

end dorothy_age_proof_l933_93395


namespace min_sum_reciprocals_l933_93379

theorem min_sum_reciprocals (x y : ℕ+) (h1 : x ≠ y) (h2 : (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 12) :
  ∃ (a b : ℕ+), a ≠ b ∧ ((1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 12) ∧ (a + b = 50) ∧ 
  (∀ (c d : ℕ+), c ≠ d → ((1 : ℚ) / c + (1 : ℚ) / d = (1 : ℚ) / 12) → (c + d ≥ 50)) :=
by sorry

end min_sum_reciprocals_l933_93379


namespace power_division_thirteen_l933_93338

theorem power_division_thirteen : 13^8 / 13^5 = 2197 := by sorry

end power_division_thirteen_l933_93338


namespace john_rejection_percentage_l933_93331

theorem john_rejection_percentage
  (jane_rejection_rate : ℝ)
  (total_rejection_rate : ℝ)
  (jane_inspection_fraction : ℝ)
  (h1 : jane_rejection_rate = 0.009)
  (h2 : total_rejection_rate = 0.0075)
  (h3 : jane_inspection_fraction = 0.625)
  : ∃ (john_rejection_rate : ℝ),
    john_rejection_rate = 0.005 ∧
    jane_rejection_rate * jane_inspection_fraction +
    john_rejection_rate * (1 - jane_inspection_fraction) =
    total_rejection_rate :=
by sorry

end john_rejection_percentage_l933_93331


namespace quadratic_inequality_solution_l933_93336

-- Define the quadratic function
def f (x : ℝ) : ℝ := -x^2 + 2*x + 3

-- Define the solution set
def solution_set : Set ℝ := {x | -1 < x ∧ x < 3}

-- Theorem statement
theorem quadratic_inequality_solution :
  {x : ℝ | f x > 0} = solution_set := by sorry

end quadratic_inequality_solution_l933_93336


namespace condition_relationship_l933_93375

theorem condition_relationship :
  (∀ x : ℝ, (2 ≤ x ∧ x ≤ 3) → (x < -3 ∨ x ≥ 1)) ∧
  (∃ x : ℝ, (x < -3 ∨ x ≥ 1) ∧ ¬(2 ≤ x ∧ x ≤ 3)) :=
by sorry

end condition_relationship_l933_93375


namespace total_amount_is_140_problem_solution_l933_93305

/-- Represents the division of money among three parties -/
structure MoneyDivision where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The conditions of the money division problem -/
def satisfiesConditions (d : MoneyDivision) : Prop :=
  d.y = 0.45 * d.x ∧ d.z = 0.3 * d.x ∧ d.y = 36

/-- The theorem stating that the total amount is 140 given the conditions -/
theorem total_amount_is_140 (d : MoneyDivision) (h : satisfiesConditions d) :
  d.x + d.y + d.z = 140 := by
  sorry

/-- The main result of the problem -/
theorem problem_solution :
  ∃ d : MoneyDivision, satisfiesConditions d ∧ d.x + d.y + d.z = 140 := by
  sorry

end total_amount_is_140_problem_solution_l933_93305


namespace right_angled_tetrahedron_volume_l933_93339

/-- A tetrahedron with all faces being right-angled triangles and three edges of length s -/
structure RightAngledTetrahedron (s : ℝ) where
  (s_pos : s > 0)
  (all_faces_right_angled : True)  -- This is a placeholder for the condition
  (three_edges_equal : True)  -- This is a placeholder for the condition

/-- The volume of a right-angled tetrahedron -/
noncomputable def volume (t : RightAngledTetrahedron s) : ℝ :=
  (s^3 * Real.sqrt 2) / 12

/-- Theorem stating the volume of a right-angled tetrahedron -/
theorem right_angled_tetrahedron_volume (s : ℝ) (t : RightAngledTetrahedron s) :
  volume t = (s^3 * Real.sqrt 2) / 12 := by
  sorry

end right_angled_tetrahedron_volume_l933_93339


namespace min_value_function_l933_93348

theorem min_value_function (x : ℝ) (h : x > 3) :
  1 / (x - 3) + x ≥ 5 ∧ (1 / (x - 3) + x = 5 ↔ x = 4) := by
  sorry

end min_value_function_l933_93348


namespace expression_value_l933_93391

theorem expression_value (a b c d x : ℝ) 
  (h1 : a + b = 0) 
  (h2 : c ≠ 0 ∧ d ≠ 0)
  (h3 : c * d = 1) 
  (h4 : |x| = Real.sqrt 7) : 
  (x^2 + (a + b + c * d) * x + Real.sqrt (a + b) + (c * d) ^ (1/3 : ℝ) = 8 + Real.sqrt 7) ∨
  (x^2 + (a + b + c * d) * x + Real.sqrt (a + b) + (c * d) ^ (1/3 : ℝ) = 8 - Real.sqrt 7) :=
by sorry

end expression_value_l933_93391


namespace sandy_comic_books_l933_93350

theorem sandy_comic_books (initial : ℕ) (final : ℕ) (bought : ℕ) : 
  initial = 14 →
  final = 13 →
  bought = final - (initial / 2) →
  bought = 6 := by
sorry

end sandy_comic_books_l933_93350


namespace correct_value_for_square_l933_93316

theorem correct_value_for_square (x : ℕ) : 60 + x * 5 = 500 ↔ x = 88 :=
by sorry

end correct_value_for_square_l933_93316


namespace flash_ace_chase_l933_93352

/-- The problem of Flash catching Ace -/
theorem flash_ace_chase (x y : ℝ) (hx : x > 1) : 
  let ace_speed := 1  -- We can set Ace's speed to 1 without loss of generality
  let flash_east_speed := x * ace_speed
  let flash_west_speed := (x + 1) * ace_speed
  let east_headstart := 2 * y
  let west_headstart := y
  let east_distance := (flash_east_speed * east_headstart) / (flash_east_speed - ace_speed)
  let west_distance := (flash_west_speed * west_headstart) / (flash_west_speed - ace_speed)
  east_distance + west_distance = (2 * x * y) / (x - 1) + ((x + 1) * y) / x :=
by sorry

end flash_ace_chase_l933_93352


namespace smallest_composite_no_small_factors_l933_93337

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def has_no_small_prime_factors (n : ℕ) : Prop := ∀ p, p < 15 → ¬(Nat.Prime p ∧ p ∣ n)

theorem smallest_composite_no_small_factors :
  (is_composite 289) ∧
  (has_no_small_prime_factors 289) ∧
  (∀ m : ℕ, m < 289 → ¬(is_composite m ∧ has_no_small_prime_factors m)) :=
sorry

end smallest_composite_no_small_factors_l933_93337


namespace cosine_is_even_l933_93386

-- Define the property of being an even function
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- State the theorem
theorem cosine_is_even : IsEven Real.cos := by
  sorry

end cosine_is_even_l933_93386


namespace price_reduction_effect_l933_93332

theorem price_reduction_effect (P Q : ℝ) (P_positive : P > 0) (Q_positive : Q > 0) :
  let new_price := P * (1 - 0.35)
  let new_quantity := Q * (1 + 0.8)
  let original_revenue := P * Q
  let new_revenue := new_price * new_quantity
  (new_revenue - original_revenue) / original_revenue = 0.17 := by
sorry

end price_reduction_effect_l933_93332


namespace min_Q_zero_at_two_thirds_l933_93300

/-- The quadratic form representing the expression to be minimized -/
def Q (k : ℝ) (x y : ℝ) : ℝ :=
  5 * x^2 - 8 * k * x * y + (4 * k^2 + 3) * y^2 - 5 * x - 6 * y + 7

/-- The theorem stating that 2/3 is the value of k that makes the minimum of Q zero -/
theorem min_Q_zero_at_two_thirds :
  (∃ (k : ℝ), ∀ (x y : ℝ), Q k x y ≥ 0 ∧ (∃ (x₀ y₀ : ℝ), Q k x₀ y₀ = 0)) ↔ k = 2/3 := by
  sorry

end min_Q_zero_at_two_thirds_l933_93300


namespace cos_alpha_value_l933_93327

theorem cos_alpha_value (α : ℝ) (h1 : α ∈ Set.Icc 0 (π / 2)) 
  (h2 : Real.cos (α + π / 6) = 1 / 3) : 
  Real.cos α = (2 * Real.sqrt 2 + Real.sqrt 3) / 6 := by
  sorry

end cos_alpha_value_l933_93327


namespace unique_solution_sqrt_equation_l933_93313

theorem unique_solution_sqrt_equation :
  ∃! z : ℝ, Real.sqrt (8 + 3 * z) = 10 :=
by
  -- The proof goes here
  sorry

end unique_solution_sqrt_equation_l933_93313


namespace polynomial_value_bound_l933_93343

/-- A polynomial with three distinct real roots -/
structure TripleRootPoly where
  a : ℝ
  b : ℝ
  c : ℝ
  has_three_distinct_roots : ∃ (r₁ r₂ r₃ : ℝ), r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃ ∧
    ∀ t, t^3 + a*t^2 + b*t + c = 0 ↔ t = r₁ ∨ t = r₂ ∨ t = r₃

/-- The polynomial P(t) = t^3 + at^2 + bt + c -/
def P (poly : TripleRootPoly) (t : ℝ) : ℝ :=
  t^3 + poly.a*t^2 + poly.b*t + poly.c

/-- The equation (x^2 + x + 2013)^3 + a(x^2 + x + 2013)^2 + b(x^2 + x + 2013) + c = 0 has no real roots -/
def no_real_roots (poly : TripleRootPoly) : Prop :=
  ∀ x : ℝ, (x^2 + x + 2013)^3 + poly.a*(x^2 + x + 2013)^2 + poly.b*(x^2 + x + 2013) + poly.c ≠ 0

/-- The main theorem -/
theorem polynomial_value_bound (poly : TripleRootPoly) (h : no_real_roots poly) : 
  P poly 2013 > 1/64 := by
  sorry

end polynomial_value_bound_l933_93343


namespace product_one_inequality_l933_93301

theorem product_one_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_prod : a * b * c = 1) :
  1 / (a * (a + 1)) + 1 / (b * (b + 1)) + 1 / (c * (c + 1)) ≥ 3 / 2 := by
  sorry

end product_one_inequality_l933_93301


namespace floor_square_minus_floor_product_l933_93317

theorem floor_square_minus_floor_product (x : ℝ) : x = 12.7 → ⌊x^2⌋ - ⌊x⌋ * ⌊x⌋ = 17 := by
  sorry

end floor_square_minus_floor_product_l933_93317


namespace min_value_quadratic_form_min_value_achievable_l933_93398

theorem min_value_quadratic_form (x y : ℝ) : 
  3 * x^2 + 2 * x * y + 3 * y^2 + 5 ≥ 5 :=
by sorry

theorem min_value_achievable : 
  ∃ (x y : ℝ), 3 * x^2 + 2 * x * y + 3 * y^2 + 5 = 5 :=
by sorry

end min_value_quadratic_form_min_value_achievable_l933_93398


namespace horner_evaluation_exclude_l933_93377

def horner_polynomial (x : ℤ) : ℤ :=
  ((7 * x + 3) * x - 5) * x + 11

def horner_step1 (x : ℤ) : ℤ :=
  7 * x + 3

def horner_step2 (x : ℤ) : ℤ :=
  (7 * x + 3) * x - 5

theorem horner_evaluation_exclude (x : ℤ) :
  x = 23 →
  horner_polynomial x ≠ 85169 ∧
  horner_step1 x ≠ 85169 ∧
  horner_step2 x ≠ 85169 :=
by sorry

end horner_evaluation_exclude_l933_93377


namespace two_roots_iff_twenty_l933_93341

/-- The quadratic equation in x parameterized by a -/
def f (a : ℝ) (x : ℝ) : ℝ := a^2 * (x - 2) + a * (39 - 20*x) + 20

/-- The proposition that the equation has at least two distinct roots -/
def has_two_distinct_roots (a : ℝ) : Prop :=
  ∃ x y, x ≠ y ∧ f a x = 0 ∧ f a y = 0

theorem two_roots_iff_twenty :
  ∀ a : ℝ, has_two_distinct_roots a ↔ a = 20 := by sorry

end two_roots_iff_twenty_l933_93341


namespace exists_sum_digits_div_11_l933_93374

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: Among any 39 consecutive natural numbers, there is always one whose sum of digits is divisible by 11 -/
theorem exists_sum_digits_div_11 (start : ℕ) : 
  ∃ k : ℕ, k ∈ Finset.range 39 ∧ (sum_of_digits (start + k)) % 11 = 0 := by sorry

end exists_sum_digits_div_11_l933_93374


namespace coat_value_problem_l933_93314

/-- Represents the problem of determining the value of a coat given to a worker --/
theorem coat_value_problem (total_pay : ℝ) (yearly_cash : ℝ) (months_worked : ℝ) 
  (partial_cash : ℝ) (h1 : total_pay = yearly_cash + coat_value) 
  (h2 : yearly_cash = 12) (h3 : months_worked = 7) (h4 : partial_cash = 5) :
  ∃ coat_value : ℝ, coat_value = 4.8 ∧ 
    (months_worked / 12) * total_pay = partial_cash + coat_value := by
  sorry


end coat_value_problem_l933_93314


namespace geometric_series_sum_l933_93347

/-- The sum of a geometric series with n terms, first term a, and common ratio r -/
def geometricSum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The sum of the specific geometric series in the problem -/
def specificSum : ℚ :=
  geometricSum (1/4) (1/4) 8

theorem geometric_series_sum :
  specificSum = 65535 / 196608 := by
  sorry

end geometric_series_sum_l933_93347


namespace unique_k_l933_93312

theorem unique_k : ∃! (k : ℕ), k > 0 ∧ (k + 2).factorial + (k + 3).factorial = k.factorial * 1344 := by
  sorry

end unique_k_l933_93312


namespace area_S_bounds_l933_93387

theorem area_S_bounds (t : ℝ) (k : ℤ) (h_t : t ≥ 0) (h_k : 2 ≤ k ∧ k ≤ 4) : 
  let T : ℝ := t - ⌊t⌋
  let S : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - T - 1)^2 + (p.2 - k)^2 ≤ (T + 1)^2}
  0 ≤ Real.pi * (T + 1)^2 ∧ Real.pi * (T + 1)^2 ≤ 4 * Real.pi :=
by sorry

end area_S_bounds_l933_93387


namespace susie_house_rooms_l933_93344

/-- The number of rooms in Susie's house -/
def number_of_rooms : ℕ := 6

/-- The time it takes Susie to vacuum the whole house (in hours) -/
def total_vacuum_time : ℚ := 2

/-- The time it takes Susie to vacuum one room (in minutes) -/
def time_per_room : ℕ := 20

/-- Theorem stating that the number of rooms in Susie's house is 6 -/
theorem susie_house_rooms :
  number_of_rooms = (total_vacuum_time * 60) / time_per_room := by
  sorry

end susie_house_rooms_l933_93344


namespace solution_set_reciprocal_inequality_l933_93370

theorem solution_set_reciprocal_inequality (x : ℝ) : 
  (1 / x > 2) ↔ (0 < x ∧ x < 1 / 2) :=
by sorry

end solution_set_reciprocal_inequality_l933_93370


namespace reciprocal_of_negative_three_arcseconds_to_degrees_conversion_negative_fraction_comparison_l933_93345

-- Define the conversion factor from arcseconds to degrees
def arcseconds_to_degrees (x : ℚ) : ℚ := x / 3600

-- Theorem statements
theorem reciprocal_of_negative_three : ((-3)⁻¹ : ℚ) = -1/3 := by sorry

theorem arcseconds_to_degrees_conversion : arcseconds_to_degrees 7200 = 2 := by sorry

theorem negative_fraction_comparison : (-3/4 : ℚ) > -4/5 := by sorry

end reciprocal_of_negative_three_arcseconds_to_degrees_conversion_negative_fraction_comparison_l933_93345


namespace probability_at_least_two_special_items_l933_93382

theorem probability_at_least_two_special_items (total : Nat) (special : Nat) (select : Nat) 
  (h1 : total = 8) (h2 : special = 3) (h3 : select = 4) : 
  (Nat.choose special 2 * Nat.choose (total - special) (select - 2) + 
   Nat.choose special 3 * Nat.choose (total - special) (select - 3)) / 
  Nat.choose total select = 1 / 2 := by
  sorry

end probability_at_least_two_special_items_l933_93382


namespace playstation_cost_proof_l933_93393

-- Define the given values
def birthday_money : ℝ := 200
def christmas_money : ℝ := 150
def game_price : ℝ := 7.5
def games_to_sell : ℕ := 20

-- Define the cost of the PlayStation
def playstation_cost : ℝ := 500

-- Theorem statement
theorem playstation_cost_proof :
  birthday_money + christmas_money + game_price * (games_to_sell : ℝ) = playstation_cost := by
  sorry

end playstation_cost_proof_l933_93393


namespace water_problem_solution_l933_93303

def water_problem (total_water : ℕ) (original_serving : ℕ) (serving_reduction : ℕ) : ℕ :=
  let original_servings := total_water / original_serving
  let new_servings := original_servings - serving_reduction
  total_water / new_servings

theorem water_problem_solution :
  water_problem 64 8 4 = 16 := by
  sorry

end water_problem_solution_l933_93303


namespace painting_time_for_six_stools_l933_93399

/-- Represents the painting process for stools -/
structure StoolPainting where
  num_stools : Nat
  first_coat_time : Nat
  wait_time : Nat

/-- Calculates the minimum time required to paint all stools -/
def minimum_painting_time (sp : StoolPainting) : Nat :=
  sp.num_stools * sp.first_coat_time + sp.wait_time + sp.first_coat_time

/-- Theorem stating that the minimum time to paint 6 stools is 24 minutes -/
theorem painting_time_for_six_stools :
  let sp : StoolPainting := {
    num_stools := 6,
    first_coat_time := 2,
    wait_time := 10
  }
  minimum_painting_time sp = 24 := by
  sorry


end painting_time_for_six_stools_l933_93399


namespace rope_folding_segments_l933_93359

/-- The number of segments produced by folding a rope n times and cutting in the middle of the last fold -/
def num_segments (n : ℕ) : ℕ := 2^n + 1

/-- Theorem stating that the number of segments follows the pattern for all natural numbers -/
theorem rope_folding_segments (n : ℕ) : num_segments n = 2^n + 1 := by
  sorry

/-- Verifying the given examples -/
example : num_segments 1 = 3 := by sorry
example : num_segments 2 = 5 := by sorry
example : num_segments 3 = 9 := by sorry

end rope_folding_segments_l933_93359


namespace lines_perpendicular_l933_93384

-- Define the lines l₁ and l
def l₁ (a : ℝ) (x y : ℝ) : Prop := 2 * x - a * y - 1 = 0
def l (x y : ℝ) : Prop := x + 2 * y = 0

-- Define the theorem
theorem lines_perpendicular :
  ∃ a : ℝ, 
    (l₁ a 1 1) ∧ 
    (∀ x y : ℝ, l₁ a x y → l x y → (2 : ℝ) * (-1/2 : ℝ) = -1) :=
by sorry

end lines_perpendicular_l933_93384


namespace jeans_discount_percentage_l933_93394

/-- Calculate the discount percentage on jeans --/
theorem jeans_discount_percentage
  (original_price : ℝ)
  (discounted_price_for_three : ℝ)
  (h1 : original_price = 40)
  (h2 : discounted_price_for_three = 112) :
  (original_price * 3 - discounted_price_for_three) / (original_price * 2) = 0.1 :=
by sorry

end jeans_discount_percentage_l933_93394


namespace tv_price_increase_l933_93373

theorem tv_price_increase (P : ℝ) (x : ℝ) (h1 : P > 0) :
  (0.80 * P + x / 100 * (0.80 * P) = 1.20 * P) → x = 50 := by
  sorry

end tv_price_increase_l933_93373


namespace plywood_length_l933_93324

/-- The length of a rectangular piece of plywood with given area and width -/
theorem plywood_length (area width : ℝ) (h1 : area = 24) (h2 : width = 6) :
  area / width = 4 := by
  sorry

end plywood_length_l933_93324


namespace triangle_angle_B_l933_93356

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_angle_B (t : Triangle) : 
  t.a = 2 * Real.sqrt 3 → 
  t.b = 2 → 
  t.A = π / 3 → 
  t.B = π / 6 := by
  sorry


end triangle_angle_B_l933_93356


namespace sin_cos_sixth_power_sum_l933_93378

theorem sin_cos_sixth_power_sum (θ : Real) (h : Real.sin (2 * θ) = 1/3) :
  Real.sin θ ^ 6 + Real.cos θ ^ 6 = 11/12 := by
  sorry

end sin_cos_sixth_power_sum_l933_93378


namespace min_value_expression_min_value_attained_l933_93351

theorem min_value_expression (x : ℝ) : 
  (13 - x) * (8 - x) * (13 + x) * (8 + x) ≥ -2746.25 :=
by
  sorry

theorem min_value_attained : 
  ∃ x : ℝ, (13 - x) * (8 - x) * (13 + x) * (8 + x) = -2746.25 :=
by
  sorry

end min_value_expression_min_value_attained_l933_93351


namespace inverse_variation_proof_l933_93368

/-- Given that x² varies inversely with y⁴, prove that when x = 5 for y = 2, 
    then x² = 25/16 when y = 4 -/
theorem inverse_variation_proof (x y : ℝ) (h : ∃ k : ℝ, x^2 * y^4 = k) 
  (h_initial : (5 : ℝ)^2 * 2^4 = x^2 * y^4) : 
  (∃ x' : ℝ, x'^2 * 4^4 = x^2 * y^4 ∧ x'^2 = 25/16) := by
  sorry

end inverse_variation_proof_l933_93368


namespace largest_perfect_square_factor_of_1512_l933_93371

/-- The largest perfect square factor of 1512 is 36 -/
theorem largest_perfect_square_factor_of_1512 :
  ∃ (n : ℕ), n * n = 36 ∧ n * n ∣ 1512 ∧ ∀ (m : ℕ), m * m ∣ 1512 → m * m ≤ n * n :=
by sorry

end largest_perfect_square_factor_of_1512_l933_93371


namespace express_vector_as_linear_combination_l933_93330

/-- Given two vectors a and b in ℝ², express vector c as a linear combination of a and b -/
theorem express_vector_as_linear_combination (a b c : ℝ × ℝ) 
  (ha : a = (1, 1)) (hb : b = (1, -1)) (hc : c = (2, 3)) :
  ∃ x y : ℝ, c = x • a + y • b ∧ x = (5 : ℝ) / 2 ∧ y = -(1 : ℝ) / 2 := by
  sorry

end express_vector_as_linear_combination_l933_93330


namespace age_sum_five_years_ago_l933_93361

theorem age_sum_five_years_ago (djibo_age : ℕ) (sister_age : ℕ) : 
  djibo_age = 17 → sister_age = 28 → djibo_age - 5 + (sister_age - 5) = 35 := by
  sorry

end age_sum_five_years_ago_l933_93361


namespace least_positive_integer_with_given_remainders_l933_93358

theorem least_positive_integer_with_given_remainders :
  ∃ (b : ℕ), b > 0 ∧
    b % 6 = 5 ∧
    b % 7 = 6 ∧
    b % 8 = 7 ∧
    b % 9 = 8 ∧
    (∀ (x : ℕ), x > 0 ∧
      x % 6 = 5 ∧
      x % 7 = 6 ∧
      x % 8 = 7 ∧
      x % 9 = 8 →
      x ≥ b) ∧
    b = 503 :=
by sorry

end least_positive_integer_with_given_remainders_l933_93358


namespace triangle_side_length_l933_93329

theorem triangle_side_length (a b c : ℝ) (B : ℝ) :
  b = 3 → c = 3 → B = π / 6 → a = 3 * Real.sqrt 3 := by
  sorry

end triangle_side_length_l933_93329


namespace f_of_f_one_equals_seven_l933_93342

def f (x : ℝ) : ℝ := 3 * x^2 - 5

theorem f_of_f_one_equals_seven : f (f 1) = 7 := by
  sorry

end f_of_f_one_equals_seven_l933_93342


namespace taxi_charge_calculation_l933_93357

/-- Calculates the total charge for a taxi trip -/
def taxiCharge (initialFee : ℚ) (additionalChargePerIncrement : ℚ) (incrementDistance : ℚ) (tripDistance : ℚ) : ℚ :=
  initialFee + (tripDistance / incrementDistance) * additionalChargePerIncrement

theorem taxi_charge_calculation :
  let initialFee : ℚ := 235/100
  let additionalChargePerIncrement : ℚ := 35/100
  let incrementDistance : ℚ := 2/5
  let tripDistance : ℚ := 36/10
  taxiCharge initialFee additionalChargePerIncrement incrementDistance tripDistance = 550/100 := by
  sorry

#eval taxiCharge (235/100) (35/100) (2/5) (36/10)

end taxi_charge_calculation_l933_93357


namespace constructible_heights_count_l933_93362

/-- A function that returns the number of constructible heights given a number of bricks and possible height increments. -/
def countConstructibleHeights (numBricks : ℕ) (heightIncrements : List ℕ) : ℕ :=
  sorry

/-- The theorem stating that with 25 bricks and height increments of 0, 3, and 4, there are 98 constructible heights. -/
theorem constructible_heights_count : 
  countConstructibleHeights 25 [0, 3, 4] = 98 :=
sorry

end constructible_heights_count_l933_93362


namespace preimage_of_four_one_l933_93321

/-- The mapping f from R² to R² defined by f(x,y) = (x+y, x-y) -/
def f (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + p.2, p.1 - p.2)

/-- Theorem stating that (2.5, 1.5) is the preimage of (4, 1) under f -/
theorem preimage_of_four_one :
  f (2.5, 1.5) = (4, 1) := by
  sorry

end preimage_of_four_one_l933_93321


namespace organize_60_toys_in_15_minutes_l933_93383

/-- Represents the toy organizing scenario with Mia and her dad -/
structure ToyOrganizing where
  totalToys : ℕ
  cycleTime : ℕ
  dadPlaces : ℕ
  miaTakesOut : ℕ

/-- Calculates the time in minutes to organize all toys -/
def timeToOrganize (scenario : ToyOrganizing) : ℚ :=
  sorry

/-- Theorem stating that the time to organize 60 toys is 15 minutes -/
theorem organize_60_toys_in_15_minutes :
  let scenario : ToyOrganizing := {
    totalToys := 60,
    cycleTime := 30,  -- in seconds
    dadPlaces := 6,
    miaTakesOut := 4
  }
  timeToOrganize scenario = 15 := by sorry

end organize_60_toys_in_15_minutes_l933_93383


namespace positive_real_as_infinite_sum_representations_l933_93320

theorem positive_real_as_infinite_sum_representations (k : ℝ) (hk : k > 0) :
  ∃ (f : ℕ → (ℕ → ℕ)), 
    (∀ n : ℕ, ∀ i j : ℕ, i < j → f n i < f n j) ∧ 
    (∀ n : ℕ, k = ∑' i, (1 : ℝ) / (10 ^ (f n i))) :=
sorry

end positive_real_as_infinite_sum_representations_l933_93320


namespace quadratic_function_properties_l933_93328

-- Define the quadratic function
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 3

-- Define the conditions
def min_at_2 (a b : ℝ) : Prop := ∀ x, f a b x ≥ f a b 2

def intercept_length_2 (a b : ℝ) : Prop :=
  ∃ x₁ x₂, x₁ < x₂ ∧ f a b x₁ = 0 ∧ f a b x₂ = 0 ∧ x₂ - x₁ = 2

-- Define g(x)
def g (a b m : ℝ) (x : ℝ) : ℝ := f a b x - m * x

-- Define the conditions for g(x)
def g_zeros_in_intervals (a b m : ℝ) : Prop :=
  ∃ x₁ x₂, 0 < x₁ ∧ x₁ < 2 ∧ 2 < x₂ ∧ x₂ < 3 ∧ g a b m x₁ = 0 ∧ g a b m x₂ = 0

-- Define the minimum value condition
def min_value_condition (a b : ℝ) (t : ℝ) : Prop :=
  ∀ x ∈ Set.Icc t (t + 1), f a b x ≥ -1/2 ∧ ∃ x₀ ∈ Set.Icc t (t + 1), f a b x₀ = -1/2

-- State the theorem
theorem quadratic_function_properties :
  ∀ a b : ℝ, min_at_2 a b → intercept_length_2 a b →
  (∃ m : ℝ, g_zeros_in_intervals a b m ∧ -1/2 < m ∧ m < 0) ∧
  (∃ t : ℝ, (min_value_condition a b t ∧ t = 1 - Real.sqrt 2 / 2) ∨
            (min_value_condition a b t ∧ t = 2 + Real.sqrt 2 / 2)) ∧
  a = 1 ∧ b = -4 := by sorry

end quadratic_function_properties_l933_93328


namespace range_of_a_when_a_minus_3_in_M_range_of_a_when_interval_subset_M_l933_93315

-- Define the functions f and g
def f (a x : ℝ) : ℝ := |x + a|
def g (x : ℝ) : ℝ := |x + 3| - x

-- Define the set M
def M (a : ℝ) : Set ℝ := {x | f a x < g x}

-- Theorem 1
theorem range_of_a_when_a_minus_3_in_M (a : ℝ) :
  (a - 3) ∈ M a → 0 < a ∧ a < 3 := by sorry

-- Theorem 2
theorem range_of_a_when_interval_subset_M (a : ℝ) :
  Set.Icc (-1) 1 ⊆ M a → -2 < a ∧ a < 2 := by sorry

end range_of_a_when_a_minus_3_in_M_range_of_a_when_interval_subset_M_l933_93315


namespace f_range_l933_93319

noncomputable def f (x : ℝ) : ℝ :=
  if x > 1 then (Real.log x) / x else Real.exp x + 1

theorem f_range :
  Set.range f = Set.union (Set.Ioc 0 (1 / Real.exp 1)) (Set.Ioo 1 (Real.exp 1 + 1)) := by
  sorry

end f_range_l933_93319


namespace sinC_value_sine_law_extension_l933_93363

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  area : ℝ

/-- The area of the triangle satisfies the given condition -/
def areaCondition (t : Triangle) : Prop :=
  t.area = (t.a + t.b)^2 - t.c^2

/-- The sum of two sides equals 4 -/
def sideSum (t : Triangle) : Prop :=
  t.a + t.b = 4

/-- Theorem 1: If the area condition and side sum condition hold, then sin C = 8/17 -/
theorem sinC_value (t : Triangle) (h1 : areaCondition t) (h2 : sideSum t) :
  Real.sin t.C = 8 / 17 := by sorry

/-- Theorem 2: The ratio of squared difference of sides to the square of the third side
    equals the ratio of sine of difference of angles to the sine of the third angle -/
theorem sine_law_extension (t : Triangle) :
  (t.a^2 - t.b^2) / t.c^2 = Real.sin (t.A - t.B) / Real.sin t.C := by sorry

end sinC_value_sine_law_extension_l933_93363


namespace planting_methods_result_l933_93372

/-- The number of rows in the field -/
def total_rows : ℕ := 10

/-- The minimum required interval between crops A and B -/
def min_interval : ℕ := 6

/-- The number of crops to be planted -/
def num_crops : ℕ := 2

/-- Calculates the number of ways to plant two crops with the given constraints -/
def planting_methods (n : ℕ) (k : ℕ) (m : ℕ) : ℕ :=
  -- n: total rows
  -- k: number of crops
  -- m: minimum interval
  sorry

theorem planting_methods_result : planting_methods total_rows num_crops min_interval = 12 := by
  sorry

end planting_methods_result_l933_93372


namespace smallest_n_satisfying_condition_l933_93388

theorem smallest_n_satisfying_condition : ∃ n : ℕ, 
  (n > 1) ∧ 
  (∀ p : ℕ, 2 ≤ p ∧ p ≤ 10 → p ∣ (n^(p-1) - 1)) ∧
  (∀ m : ℕ, 1 < m ∧ m < n → ∃ q : ℕ, 2 ≤ q ∧ q ≤ 10 ∧ ¬(q ∣ (m^(q-1) - 1))) ∧
  n = 2521 :=
sorry

end smallest_n_satisfying_condition_l933_93388


namespace largest_prime_factor_l933_93381

theorem largest_prime_factor : 
  let n := 20^3 + 15^4 - 10^5 + 2*25^3
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ n ∧ ∀ q, Nat.Prime q → q ∣ n → q ≤ p ∧ p = 11 := by
  sorry

end largest_prime_factor_l933_93381


namespace stratified_sampling_science_students_l933_93349

theorem stratified_sampling_science_students 
  (total_students : ℕ) 
  (science_students : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_students = 720) 
  (h2 : science_students = 480) 
  (h3 : sample_size = 90) :
  (science_students : ℚ) / (total_students : ℚ) * (sample_size : ℚ) = 60 := by
sorry

end stratified_sampling_science_students_l933_93349


namespace binary_ones_condition_theorem_l933_93333

/-- The number of 1's in the binary representation of a natural number -/
def binary_ones (n : ℕ) : ℕ := sorry

/-- A function satisfying the given condition -/
def satisfies_condition (f : ℕ → ℕ) : Prop :=
  ∀ x y : ℕ, binary_ones (f x + y) = binary_ones (f y + x)

/-- The main theorem -/
theorem binary_ones_condition_theorem (f : ℕ → ℕ) :
  satisfies_condition f → ∃ c : ℕ, ∀ x : ℕ, f x = x + c := by sorry

end binary_ones_condition_theorem_l933_93333


namespace amy_homework_time_l933_93397

theorem amy_homework_time (math_problems : ℕ) (spelling_problems : ℕ) (problems_per_hour : ℕ) : 
  math_problems = 18 → spelling_problems = 6 → problems_per_hour = 4 →
  (math_problems + spelling_problems) / problems_per_hour = 6 := by
  sorry

end amy_homework_time_l933_93397


namespace sam_initial_dimes_l933_93366

/-- Represents the number of cents in a coin -/
def cents_in_coin (coin : String) : ℕ :=
  match coin with
  | "dime" => 10
  | "quarter" => 25
  | _ => 0

/-- Represents Sam's initial coin counts and purchases -/
structure SamsPurchase where
  initial_quarters : ℕ
  candy_bars : ℕ
  candy_bar_price : ℕ
  lollipops : ℕ
  lollipop_price : ℕ
  cents_left : ℕ

/-- Theorem stating that Sam had 19 dimes initially -/
theorem sam_initial_dimes (purchase : SamsPurchase)
  (h1 : purchase.initial_quarters = 6)
  (h2 : purchase.candy_bars = 4)
  (h3 : purchase.candy_bar_price = 3)
  (h4 : purchase.lollipops = 1)
  (h5 : purchase.lollipop_price = 1)
  (h6 : purchase.cents_left = 195) :
  (purchase.cents_left +
   purchase.candy_bars * purchase.candy_bar_price * cents_in_coin "dime" +
   purchase.lollipops * cents_in_coin "quarter" -
   purchase.initial_quarters * cents_in_coin "quarter") / cents_in_coin "dime" = 19 := by
  sorry

#eval cents_in_coin "dime"  -- Should output 10
#eval cents_in_coin "quarter"  -- Should output 25

end sam_initial_dimes_l933_93366


namespace total_rainfall_l933_93322

def rainfall_problem (first_week : ℝ) (second_week : ℝ) : Prop :=
  (second_week = 1.5 * first_week) ∧
  (second_week = 18) ∧
  (first_week + second_week = 30)

theorem total_rainfall : ∃ (first_week second_week : ℝ),
  rainfall_problem first_week second_week :=
by sorry

end total_rainfall_l933_93322


namespace wood_carving_shelves_l933_93334

theorem wood_carving_shelves (total_carvings : ℕ) (carvings_per_shelf : ℕ) (shelves_filled : ℕ) : 
  total_carvings = 56 → 
  carvings_per_shelf = 8 → 
  shelves_filled = total_carvings / carvings_per_shelf → 
  shelves_filled = 7 := by
sorry

end wood_carving_shelves_l933_93334


namespace special_sequence_max_length_l933_93325

/-- A finite sequence of real numbers satisfying the given conditions -/
def SpecialSequence (a : ℕ → ℝ) (n : ℕ) : Prop :=
  (∀ i, i + 2 < n → a i + a (i + 1) + a (i + 2) < 0) ∧
  (∀ i, i + 3 < n → a i + a (i + 1) + a (i + 2) + a (i + 3) > 0)

/-- The maximum length of a SpecialSequence is 5 -/
theorem special_sequence_max_length :
  ∀ n : ℕ, ∀ a : ℕ → ℝ, SpecialSequence a n → n ≤ 5 :=
by sorry

end special_sequence_max_length_l933_93325


namespace right_angled_constructions_l933_93318

/-- Represents a triangle with angles in degrees -/
structure Triangle :=
  (angle1 : ℝ)
  (angle2 : ℝ)
  (angle3 : ℝ)

/-- Checks if a triangle is right-angled -/
def is_right_angled (t : Triangle) : Prop :=
  t.angle1 = 90 ∨ t.angle2 = 90 ∨ t.angle3 = 90

/-- The basic triangle obtained from dividing a regular hexagon into 12 parts -/
def basic_triangle : Triangle :=
  { angle1 := 30, angle2 := 60, angle3 := 90 }

/-- Represents the number of basic triangles used to form a larger triangle -/
inductive TriangleComposition
  | One
  | Three
  | Four
  | Nine

/-- Function to construct a triangle from a given number of basic triangles -/
def construct_triangle (n : TriangleComposition) : Triangle :=
  sorry

/-- Theorem stating that right-angled triangles can be formed using 1, 3, 4, or 9 basic triangles -/
theorem right_angled_constructions :
  ∀ n : TriangleComposition, is_right_angled (construct_triangle n) :=
sorry

end right_angled_constructions_l933_93318


namespace select_student_count_l933_93308

/-- The number of ways to select one student from a group of high school students -/
def select_student (first_year : ℕ) (second_year : ℕ) (third_year : ℕ) : ℕ :=
  first_year + second_year + third_year

/-- Theorem: Given the specified number of students in each year,
    the number of ways to select one student is 12 -/
theorem select_student_count :
  select_student 3 5 4 = 12 := by
  sorry

end select_student_count_l933_93308


namespace systematic_sampling_eighth_group_l933_93389

/-- Systematic sampling function -/
def systematicSample (totalSize : ℕ) (sampleSize : ℕ) (thirdGroupNumber : ℕ) (groupNumber : ℕ) : ℕ :=
  let groupCount := totalSize / sampleSize
  let commonDifference := groupCount
  thirdGroupNumber + (groupNumber - 3) * commonDifference

/-- Theorem: In a systematic sampling of 840 employees with a sample size of 42,
    if the number drawn in the third group is 44, then the number drawn in the eighth group is 144. -/
theorem systematic_sampling_eighth_group :
  systematicSample 840 42 44 8 = 144 := by
  sorry

end systematic_sampling_eighth_group_l933_93389


namespace abc_product_l933_93396

theorem abc_product (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a * b = 24 * Real.rpow 3 (1/3))
  (hac : a * c = 40 * Real.rpow 3 (1/3))
  (hbc : b * c = 15 * Real.rpow 3 (1/3)) :
  a * b * c = 120 * Real.sqrt 3 := by
sorry

end abc_product_l933_93396


namespace age_ratio_proof_l933_93385

/-- Represents a person's age -/
structure Age where
  years : ℕ

/-- Represents the ratio between two ages -/
structure AgeRatio where
  numerator : ℕ
  denominator : ℕ

def Arun : Age := ⟨20⟩
def Deepak : Age := ⟨30⟩

def currentRatio : AgeRatio := ⟨2, 3⟩

theorem age_ratio_proof :
  (Arun.years + 5 = 25) ∧
  (Deepak.years = 30) →
  (currentRatio.numerator * Deepak.years = currentRatio.denominator * Arun.years) :=
by sorry

end age_ratio_proof_l933_93385


namespace even_function_increasing_interval_l933_93346

def f (m : ℝ) (x : ℝ) : ℝ := (m - 2) * x^2 + (m - 1) * x + 2

theorem even_function_increasing_interval (m : ℝ) :
  (∀ x : ℝ, f m x = f m (-x)) →
  (∃ a : ℝ, ∀ x y : ℝ, x < y ∧ y ≤ 0 → f m x < f m y) ∧
  (∀ x y : ℝ, 0 < x ∧ x < y → f m x > f m y) :=
sorry

end even_function_increasing_interval_l933_93346


namespace floor_sum_equals_155_l933_93335

theorem floor_sum_equals_155 (p q r s : ℝ) : 
  p > 0 → q > 0 → r > 0 → s > 0 →
  p^2 + q^2 = 3024 →
  r^2 + s^2 = 3024 →
  p * r = 1500 →
  q * s = 1500 →
  ⌊p + q + r + s⌋ = 155 := by
sorry

end floor_sum_equals_155_l933_93335


namespace seven_minus_a_greater_than_b_l933_93354

theorem seven_minus_a_greater_than_b (a b : ℝ) (h : b < a ∧ a < 0) : 7 - a > b := by
  sorry

end seven_minus_a_greater_than_b_l933_93354


namespace union_of_subsets_l933_93369

def A : Set ℕ := {1, 3}
def B : Set ℕ := {1, 2, 3}

theorem union_of_subsets :
  A ⊆ B → A ∪ B = {1, 2, 3} := by
  sorry

end union_of_subsets_l933_93369


namespace set_equality_l933_93340

def S : Set ℤ := {x | -3 < x ∧ x < 3}

theorem set_equality : S = {-2, -1, 0, 1, 2} := by sorry

end set_equality_l933_93340


namespace counterexample_exists_l933_93355

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem counterexample_exists : ∃ n : ℕ, 
  (sum_of_digits n % 27 = 0) ∧ 
  (n % 27 ≠ 0) ∧ 
  (n = 9918) := by
  sorry

end counterexample_exists_l933_93355


namespace manuscript_cost_example_l933_93309

/-- Calculates the total cost of typing and revising a manuscript --/
def manuscript_cost (total_pages : ℕ) (pages_revised_once : ℕ) (pages_revised_twice : ℕ) 
  (initial_cost_per_page : ℕ) (revision_cost_per_page : ℕ) : ℕ :=
  let pages_not_revised := total_pages - pages_revised_once - pages_revised_twice
  let initial_typing_cost := total_pages * initial_cost_per_page
  let first_revision_cost := pages_revised_once * revision_cost_per_page
  let second_revision_cost := pages_revised_twice * revision_cost_per_page * 2
  initial_typing_cost + first_revision_cost + second_revision_cost

theorem manuscript_cost_example : 
  manuscript_cost 100 20 30 10 5 = 1400 := by
  sorry

end manuscript_cost_example_l933_93309


namespace polynomial_factorization_l933_93367

theorem polynomial_factorization (x : ℝ) : 
  x^4 - 4*x^3 + 6*x^2 - 4*x + 1 = (x - 1)^4 := by
  sorry

end polynomial_factorization_l933_93367


namespace simplify_like_terms_l933_93323

theorem simplify_like_terms (y : ℝ) : 5 * y + 7 * y + 8 * y = 20 * y := by
  sorry

end simplify_like_terms_l933_93323


namespace polygon_sides_l933_93304

theorem polygon_sides (n : ℕ) (angle_sum : ℝ) : 
  (n ≥ 3) →  -- Ensure it's a polygon
  (angle_sum = 2790) →  -- Given sum of angles except one
  (∃ x : ℝ, x > 0 ∧ x < 180 ∧ 180 * (n - 2) = angle_sum + x) →  -- Existence of the missing angle
  (n = 18) :=
by sorry

end polygon_sides_l933_93304


namespace graduation_photo_arrangements_l933_93380

/-- The number of students in the class -/
def num_students : ℕ := 6

/-- The total number of people (students + teacher) -/
def total_people : ℕ := num_students + 1

/-- The number of arrangements with the teacher in the middle -/
def total_arrangements : ℕ := (num_students.factorial)

/-- The number of arrangements with the teacher in the middle and students A and B adjacent -/
def adjacent_arrangements : ℕ := 4 * 2 * ((num_students - 2).factorial)

/-- The number of valid arrangements -/
def valid_arrangements : ℕ := total_arrangements - adjacent_arrangements

theorem graduation_photo_arrangements :
  valid_arrangements = 528 := by sorry

end graduation_photo_arrangements_l933_93380


namespace remainder_theorem_l933_93392

theorem remainder_theorem (w : ℤ) (h : (w + 3) % 11 = 0) : w % 13 = 8 := by
  sorry

end remainder_theorem_l933_93392


namespace divisibility_implies_B_zero_l933_93390

def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def number (B : ℕ) : ℕ := 3538084 * 10 + B

theorem divisibility_implies_B_zero (B : ℕ) (h_digit : is_digit B) :
  (∃ k₂ k₄ k₅ k₆ k₈ : ℕ, 
    number B = 2 * k₂ ∧
    number B = 4 * k₄ ∧
    number B = 5 * k₅ ∧
    number B = 6 * k₆ ∧
    number B = 8 * k₈) →
  B = 0 :=
by sorry

end divisibility_implies_B_zero_l933_93390


namespace prob_second_draw_black_l933_93307

/-- The probability of drawing a black ball on the second draw without replacement -/
def second_draw_black_prob (total : ℕ) (black : ℕ) (white : ℕ) : ℚ :=
  if total = black + white ∧ black > 0 ∧ white > 0 then
    black / (total - 1)
  else
    0

theorem prob_second_draw_black :
  second_draw_black_prob 10 3 7 = 3 / 10 := by
  sorry

end prob_second_draw_black_l933_93307


namespace expression_evaluation_l933_93365

theorem expression_evaluation :
  let x : ℚ := -1/2
  let y : ℚ := -3
  3 * (x^2 - 2*x*y) - (3*x^2 - 2*y + 2*(x*y + y)) = -12 :=
by
  sorry

end expression_evaluation_l933_93365


namespace a7_equals_one_l933_93326

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem a7_equals_one (a : ℕ → ℝ) :
  geometric_sequence a →
  a 1 * a 13 = 1 →
  a 1 + a 13 = 8 →
  a 7 = 1 :=
by sorry

end a7_equals_one_l933_93326


namespace charity_fundraising_l933_93353

theorem charity_fundraising 
  (total_amount : ℕ) 
  (num_friends : ℕ) 
  (min_amount : ℕ) 
  (h1 : total_amount = 3000)
  (h2 : num_friends = 10)
  (h3 : min_amount = 300) :
  (total_amount / num_friends = min_amount) ∧ 
  (∀ (amount : ℕ), amount ≥ min_amount → amount * num_friends = total_amount → amount = min_amount) :=
by sorry

end charity_fundraising_l933_93353


namespace runners_on_circular_track_l933_93376

/-- Represents a runner on a circular track -/
structure Runner where
  lap_time : ℝ
  speed : ℝ

/-- Theorem about two runners on a circular track -/
theorem runners_on_circular_track
  (track_length : ℝ)
  (troye daniella : Runner)
  (h1 : track_length > 0)
  (h2 : troye.lap_time = 56)
  (h3 : troye.speed = track_length / troye.lap_time)
  (h4 : daniella.speed = track_length / daniella.lap_time)
  (h5 : troye.speed + daniella.speed = track_length / 24) :
  daniella.lap_time = 42 := by
  sorry

end runners_on_circular_track_l933_93376


namespace fraction_zero_implies_x_equals_two_l933_93311

theorem fraction_zero_implies_x_equals_two (x : ℝ) : 
  (|x| - 2) / (x + 2) = 0 → x = 2 := by
  sorry

end fraction_zero_implies_x_equals_two_l933_93311


namespace sqrt_sum_squares_eq_sum_l933_93360

theorem sqrt_sum_squares_eq_sum (a b : ℝ) :
  Real.sqrt (a^2 + b^2) = a + b ↔ a * b = 0 ∧ a + b ≥ 0 := by
  sorry

end sqrt_sum_squares_eq_sum_l933_93360


namespace time_for_c_alone_l933_93364

/-- The time required for C to complete the job alone given the work rates of A, B, and C together -/
theorem time_for_c_alone (r_ab r_bc r_ca : ℚ) : 
  r_ab = 1/3 → r_bc = 1/6 → r_ca = 1/4 → (1 : ℚ) / (3/8 - 1/3) = 24/5 := by
  sorry

end time_for_c_alone_l933_93364


namespace grade_assignment_count_l933_93302

theorem grade_assignment_count :
  let num_students : ℕ := 12
  let num_grades : ℕ := 4
  num_grades ^ num_students = 16777216 :=
by sorry

end grade_assignment_count_l933_93302


namespace jimmy_calorie_consumption_l933_93306

def cracker_calories : ℕ := 15
def cookie_calories : ℕ := 50
def crackers_eaten : ℕ := 10
def cookies_eaten : ℕ := 7

theorem jimmy_calorie_consumption :
  cracker_calories * crackers_eaten + cookie_calories * cookies_eaten = 500 := by
  sorry

end jimmy_calorie_consumption_l933_93306


namespace sum_of_digits_l933_93310

def is_single_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def number_representation (A B : ℕ) : ℕ := A * 100000 + 44610 + B

theorem sum_of_digits (A B : ℕ) : 
  is_single_digit A → 
  is_single_digit B → 
  (number_representation A B) % 72 = 0 → 
  A + B = 12 := by
sorry

end sum_of_digits_l933_93310

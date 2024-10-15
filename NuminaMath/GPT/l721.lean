import Mathlib

namespace NUMINAMATH_GPT_fred_seashells_now_l721_72162

def seashells_initial := 47
def seashells_given := 25

theorem fred_seashells_now : seashells_initial - seashells_given = 22 := 
by 
  sorry

end NUMINAMATH_GPT_fred_seashells_now_l721_72162


namespace NUMINAMATH_GPT_cone_diameter_l721_72189

theorem cone_diameter (S : ℝ) (hS : S = 3 * Real.pi) (unfold_semicircle : ∃ (r l : ℝ), l = 2 * r ∧ S = π * r^2 + (1 / 2) * π * l^2) : 
∃ d : ℝ, d = Real.sqrt 6 := 
by
  sorry

end NUMINAMATH_GPT_cone_diameter_l721_72189


namespace NUMINAMATH_GPT_company_food_purchase_1_l721_72180

theorem company_food_purchase_1 (x y : ℕ) (h1: x + y = 170) (h2: 15 * x + 20 * y = 3000) : 
  x = 80 ∧ y = 90 := by
  sorry

end NUMINAMATH_GPT_company_food_purchase_1_l721_72180


namespace NUMINAMATH_GPT_log_inequality_l721_72111

theorem log_inequality (x y : ℝ) :
  let log2 := Real.log 2
  let log5 := Real.log 5
  let log3 := Real.log 3
  let log2_3 := log3 / log2
  let log5_3 := log3 / log5
  (log2_3 ^ x - log5_3 ^ x ≥ log2_3 ^ (-y) - log5_3 ^ (-y)) → (x + y ≥ 0) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_log_inequality_l721_72111


namespace NUMINAMATH_GPT_shared_total_l721_72120

theorem shared_total (total_amount : ℝ) (maggie_share : ℝ) (debby_percentage : ℝ)
  (h1 : debby_percentage = 0.25)
  (h2 : maggie_share = 4500)
  (h3 : maggie_share = (1 - debby_percentage) * total_amount) :
  total_amount = 6000 :=
by
  sorry

end NUMINAMATH_GPT_shared_total_l721_72120


namespace NUMINAMATH_GPT_total_photos_l721_72176

-- Define the number of photos Claire has taken
def photos_by_Claire : ℕ := 8

-- Define the number of photos Lisa has taken
def photos_by_Lisa : ℕ := 3 * photos_by_Claire

-- Define the number of photos Robert has taken
def photos_by_Robert : ℕ := photos_by_Claire + 16

-- State the theorem we want to prove
theorem total_photos : photos_by_Lisa + photos_by_Robert = 48 :=
by
  sorry

end NUMINAMATH_GPT_total_photos_l721_72176


namespace NUMINAMATH_GPT_total_fruits_is_43_apple_to_pear_ratio_is_24_to_19_l721_72150

def keith_pears : ℕ := 6
def keith_apples : ℕ := 4
def jason_pears : ℕ := 9
def jason_apples : ℕ := 8
def joan_pears : ℕ := 4
def joan_apples : ℕ := 12

def total_pears : ℕ := keith_pears + jason_pears + joan_pears
def total_apples : ℕ := keith_apples + jason_apples + joan_apples
def total_fruits : ℕ := total_pears + total_apples
def apple_to_pear_ratio : ℚ := total_apples / total_pears

theorem total_fruits_is_43 : total_fruits = 43 := by
  sorry

theorem apple_to_pear_ratio_is_24_to_19 : apple_to_pear_ratio = 24/19 := by
  sorry

end NUMINAMATH_GPT_total_fruits_is_43_apple_to_pear_ratio_is_24_to_19_l721_72150


namespace NUMINAMATH_GPT_speed_ratio_l721_72181

variable (vA vB : ℝ)
variable (H1 : 3 * vA = abs (-400 + 3 * vB))
variable (H2 : 10 * vA = abs (-400 + 10 * vB))

theorem speed_ratio (vA vB : ℝ) (H1 : 3 * vA = abs (-400 + 3 * vB)) (H2 : 10 * vA = abs (-400 + 10 * vB)) : 
  vA / vB = 5 / 6 :=
  sorry

end NUMINAMATH_GPT_speed_ratio_l721_72181


namespace NUMINAMATH_GPT_monroe_legs_total_l721_72158

def num_spiders : ℕ := 8
def num_ants : ℕ := 12
def legs_per_spider : ℕ := 8
def legs_per_ant : ℕ := 6

theorem monroe_legs_total :
  num_spiders * legs_per_spider + num_ants * legs_per_ant = 136 :=
by
  sorry

end NUMINAMATH_GPT_monroe_legs_total_l721_72158


namespace NUMINAMATH_GPT_find_positive_integer_x_l721_72175

def positive_integer (x : ℕ) : Prop :=
  x > 0

def n (x : ℕ) : ℕ :=
  x^2 + 3 * x + 20

def d (x : ℕ) : ℕ :=
  3 * x + 4

def division_property (x : ℕ) : Prop :=
  ∃ q r : ℕ, q = x ∧ r = 8 ∧ n x = q * d x + r

theorem find_positive_integer_x :
  ∃ x : ℕ, positive_integer x ∧ n x = x * d x + 8 :=
sorry

end NUMINAMATH_GPT_find_positive_integer_x_l721_72175


namespace NUMINAMATH_GPT_flat_rate_65_l721_72112

noncomputable def flat_rate_first_night (f n : ℝ) : Prop := 
  (f + 4 * n = 245) ∧ (f + 9 * n = 470)

theorem flat_rate_65 :
  ∃ (f n : ℝ), flat_rate_first_night f n ∧ f = 65 := 
by
  sorry

end NUMINAMATH_GPT_flat_rate_65_l721_72112


namespace NUMINAMATH_GPT_perpendicular_condition_l721_72178

-- Definitions based on the conditions
def line_l1 (m : ℝ) (x y : ℝ) : Prop := (m + 1) * x + (1 - m) * y - 1 = 0
def line_l2 (m : ℝ) (x y : ℝ) : Prop := (m - 1) * x + (2 * m + 1) * y + 4 = 0

-- Perpendicularity condition based on the definition in conditions
def perpendicular (m : ℝ) : Prop :=
  (m + 1) * (m - 1) + (1 - m) * (2 * m + 1) = 0

-- Sufficient but not necessary condition
def sufficient_but_not_necessary (m : ℝ) : Prop :=
  m = 0

-- Final statement to prove
theorem perpendicular_condition :
  sufficient_but_not_necessary 0 -> perpendicular 0 :=
by
  sorry

end NUMINAMATH_GPT_perpendicular_condition_l721_72178


namespace NUMINAMATH_GPT_scientific_notation_of_20000_l721_72199

def number : ℕ := 20000

theorem scientific_notation_of_20000 : number = 2 * 10 ^ 4 :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_of_20000_l721_72199


namespace NUMINAMATH_GPT_domain_of_function_l721_72127

theorem domain_of_function :
  {x : ℝ | x^3 + 5*x^2 + 6*x ≠ 0} =
  {x : ℝ | x < -3} ∪ {x : ℝ | -3 < x ∧ x < -2} ∪ {x : ℝ | -2 < x ∧ x < 0} ∪ {x : ℝ | 0 < x} :=
by
  sorry

end NUMINAMATH_GPT_domain_of_function_l721_72127


namespace NUMINAMATH_GPT_integer_satisfies_mod_l721_72126

theorem integer_satisfies_mod (n : ℕ) (h1 : 0 ≤ n) (h2 : n < 23) (h3 : 38635 % 23 = n % 23) :
  n = 18 := 
sorry

end NUMINAMATH_GPT_integer_satisfies_mod_l721_72126


namespace NUMINAMATH_GPT_function_satisfy_f1_function_satisfy_f2_l721_72134

noncomputable def f1 (x : ℝ) : ℝ := 2
noncomputable def f2 (x : ℝ) : ℝ := x

theorem function_satisfy_f1 : 
  ∀ x y : ℝ, x > 0 → y > 0 → f1 (x + y) + f1 x * f1 y = f1 (x * y) + f1 x + f1 y :=
by 
  intros x y hx hy
  unfold f1
  sorry

theorem function_satisfy_f2 :
  ∀ x y : ℝ, x > 0 → y > 0 → f2 (x + y) + f2 x * f2 y = f2 (x * y) + f2 x + f2 y :=
by 
  intros x y hx hy
  unfold f2
  sorry

end NUMINAMATH_GPT_function_satisfy_f1_function_satisfy_f2_l721_72134


namespace NUMINAMATH_GPT_Jack_has_18_dimes_l721_72194

theorem Jack_has_18_dimes :
  ∃ d q : ℕ, (d = q + 3 ∧ 10 * d + 25 * q = 555) ∧ d = 18 :=
by
  sorry

end NUMINAMATH_GPT_Jack_has_18_dimes_l721_72194


namespace NUMINAMATH_GPT_women_in_village_l721_72190

theorem women_in_village (W : ℕ) (men_present : ℕ := 150) (p : ℝ := 140.78099890167377) 
    (men_reduction_per_year: ℝ := 0.10) (year1_men : ℝ := men_present * (1 - men_reduction_per_year)) 
    (year2_men : ℝ := year1_men * (1 - men_reduction_per_year)) 
    (formula : ℝ := (year2_men^2 + W^2).sqrt) 
    (h : formula = p) : W = 71 := 
by
  sorry

end NUMINAMATH_GPT_women_in_village_l721_72190


namespace NUMINAMATH_GPT_right_triangle_min_perimeter_multiple_13_l721_72106

theorem right_triangle_min_perimeter_multiple_13 :
  ∃ (a b c : ℕ), 
    (a^2 + b^2 = c^2) ∧ 
    (a % 13 = 0 ∨ b % 13 = 0) ∧
    (a < b) ∧ 
    (a + b > c) ∧ 
    (a + b + c = 24) :=
sorry

end NUMINAMATH_GPT_right_triangle_min_perimeter_multiple_13_l721_72106


namespace NUMINAMATH_GPT_number_of_positive_integer_solutions_l721_72161

theorem number_of_positive_integer_solutions :
  ∃ n : ℕ, n = 84 ∧ (∀ x y z t : ℕ, 0 < x ∧ 0 < y ∧ 0 < z ∧ 0 < t ∧ x + y + z + t = 10 → true) :=
sorry

end NUMINAMATH_GPT_number_of_positive_integer_solutions_l721_72161


namespace NUMINAMATH_GPT_calculate_expression_l721_72177

theorem calculate_expression :
  ( (5^1010)^2 - (5^1008)^2) / ( (5^1009)^2 - (5^1007)^2) = 25 := 
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l721_72177


namespace NUMINAMATH_GPT_geometric_sequence_mean_l721_72196

theorem geometric_sequence_mean (a : ℕ → ℝ) (q : ℝ) (h_q : q = -2) 
  (h_condition : a 3 * a 7 = 4 * a 4) : 
  ((a 8 + a 11) / 2 = -56) 
:= sorry

end NUMINAMATH_GPT_geometric_sequence_mean_l721_72196


namespace NUMINAMATH_GPT_machine_does_not_require_repair_l721_72165

variable (nominal_mass max_deviation standard_deviation : ℝ)
variable (nominal_mass_ge : nominal_mass ≥ 370)
variable (max_deviation_le : max_deviation ≤ 0.1 * nominal_mass)
variable (all_deviations_le_max : ∀ d, d < max_deviation → d < 37)
variable (std_dev_le_max_dev : standard_deviation ≤ max_deviation)

theorem machine_does_not_require_repair :
  ¬ (standard_deviation > 37) :=
by 
  -- sorry annotation indicates the proof goes here
  sorry

end NUMINAMATH_GPT_machine_does_not_require_repair_l721_72165


namespace NUMINAMATH_GPT_quadratic_roots_equation_l721_72124

theorem quadratic_roots_equation (a b c r s : ℝ)
    (h1 : a ≠ 0)
    (h2 : a * r^2 + b * r + c = 0)
    (h3 : a * s^2 + b * s + c = 0) :
    ∃ p q : ℝ, (x^2 - b * x + a * c = 0) ∧ (ar + b, as + b) = (p, q) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_roots_equation_l721_72124


namespace NUMINAMATH_GPT_area_of_triangle_with_rational_vertices_on_unit_circle_is_rational_l721_72135

def rational_coords_on_unit_circle (x₁ y₁ x₂ y₂ x₃ y₃ : ℚ) : Prop :=
  x₁^2 + y₁^2 = 1 ∧ x₂^2 + y₂^2 = 1 ∧ x₃^2 + y₃^2 = 1

theorem area_of_triangle_with_rational_vertices_on_unit_circle_is_rational
  (x₁ y₁ x₂ y₂ x₃ y₃ : ℚ)
  (h : rational_coords_on_unit_circle x₁ y₁ x₂ y₂ x₃ y₃) :
  ∃ (A : ℚ), A = 1 / 2 * abs (x₁ * (y₂ - y₃) + x₂ * (y₃ - y₁) + x₃ * (y₁ - y₂)) :=
sorry

end NUMINAMATH_GPT_area_of_triangle_with_rational_vertices_on_unit_circle_is_rational_l721_72135


namespace NUMINAMATH_GPT_find_13th_result_l721_72149

theorem find_13th_result 
  (average_25 : ℕ) (average_12_first : ℕ) (average_12_last : ℕ) 
  (total_25 : average_25 * 25 = 600) 
  (total_12_first : average_12_first * 12 = 168) 
  (total_12_last : average_12_last * 12 = 204) 
: average_25 - average_12_first - average_12_last = 228 :=
by
  sorry

end NUMINAMATH_GPT_find_13th_result_l721_72149


namespace NUMINAMATH_GPT_sum_coords_A_eq_neg9_l721_72184

variable (A B C : ℝ × ℝ)
variable (h1 : (C.1 - A.1)^2 + (C.2 - A.2)^2 = (B.1 - A.1)^2 + (B.2 - A.2)^2 / 3)
variable (h2 : (C.1 - B.1)^2 + (C.2 - B.2)^2 = (A.1 - B.1)^2 + (A.2 - B.2)^2 / 3)
variable (hB : B = (2, 5))
variable (hC : C = (4, 11))

theorem sum_coords_A_eq_neg9 
  (A B C : ℝ × ℝ)
  (h1 : (C.1 - A.1)^2 + (C.2 - A.2)^2 = (B.1 - A.1)^2 + (B.2 - A.2)^2 / 3)
  (h2 : (C.1 - B.1)^2 + (C.2 - B.2)^2 = (A.1 - B.1)^2 + (A.2 - B.2)^2 / 3)
  (hB : B = (2, 5))
  (hC : C = (4, 11)) : 
  A.1 + A.2 = -9 :=
  sorry

end NUMINAMATH_GPT_sum_coords_A_eq_neg9_l721_72184


namespace NUMINAMATH_GPT_probability_of_negative_cosine_value_l721_72154

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∃ a1 d, ∀ n, a n = a1 + (n - 1) * d

def sum_arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
∀ n, S n = n * a 1 + (n * (n - 1) / 2) * (a 2 - a 1)

theorem probability_of_negative_cosine_value (a : ℕ → ℝ) (S : ℕ → ℝ) 
(h_arith_seq : arithmetic_sequence a)
(h_sum_seq : sum_arithmetic_sequence a S)
(h_S4 : S 4 = Real.pi)
(h_a4_eq_2a2 : a 4 = 2 * a 2) :
∃ p : ℝ, p = 7 / 15 ∧
  ∀ n, 1 ≤ n ∧ n ≤ 30 → 
  ((Real.cos (a n) < 0) → p = 7 / 15) :=
by sorry

end NUMINAMATH_GPT_probability_of_negative_cosine_value_l721_72154


namespace NUMINAMATH_GPT_find_x_l721_72132

theorem find_x
  (a b c d k : ℝ)
  (h1 : a ≠ b)
  (h2 : b ≠ 0)
  (h3 : d ≠ 0)
  (h4 : k ≠ 0)
  (h5 : k ≠ 1)
  (h_frac_change : (a + k * x) / (b + x) = c / d) :
  x = (b * c - a * d) / (k * d - c) := by
  sorry

end NUMINAMATH_GPT_find_x_l721_72132


namespace NUMINAMATH_GPT_min_value_x_plus_2y_l721_72148

variable (x y : ℝ) (hx : x > 0) (hy : y > 0)

theorem min_value_x_plus_2y (h : (2 / x) + (1 / y) = 1) : x + 2 * y ≥ 8 := 
  sorry

end NUMINAMATH_GPT_min_value_x_plus_2y_l721_72148


namespace NUMINAMATH_GPT_area_of_roof_l721_72147

def roof_area (w l : ℕ) : ℕ := l * w

theorem area_of_roof :
  ∃ (w l : ℕ), l = 4 * w ∧ l - w = 45 ∧ roof_area w l = 900 :=
by
  -- Defining witnesses for width and length
  use 15, 60
  -- Splitting the goals for clarity
  apply And.intro
  -- Proving the first condition: l = 4 * w
  · show 60 = 4 * 15
    rfl
  apply And.intro
  -- Proving the second condition: l - w = 45
  · show 60 - 15 = 45
    rfl
  -- Proving the area calculation: roof_area w l = 900
  · show roof_area 15 60 = 900
    rfl

end NUMINAMATH_GPT_area_of_roof_l721_72147


namespace NUMINAMATH_GPT_circle_symmetric_about_line_l721_72140

-- The main proof statement
theorem circle_symmetric_about_line (x y : ℝ) (k : ℝ) :
  (x - 1)^2 + (y - 1)^2 = 2 ∧ y = k * x + 3 → k = -2 :=
by
  sorry

end NUMINAMATH_GPT_circle_symmetric_about_line_l721_72140


namespace NUMINAMATH_GPT_fraction_exponentiation_l721_72183

theorem fraction_exponentiation :
  (1 / 3) ^ 5 = 1 / 243 :=
sorry

end NUMINAMATH_GPT_fraction_exponentiation_l721_72183


namespace NUMINAMATH_GPT_max_ratio_of_right_triangle_l721_72143

theorem max_ratio_of_right_triangle (a b c: ℝ) (h1: (1/2) * a * b = 30) (h2: a^2 + b^2 = c^2) : 
  (∀ x y z, (1/2 * x * y = 30) → (x^2 + y^2 = z^2) → 
  (x + y + z) / 30 ≤ (7.75 + 7.75 + 10.95) / 30) :=
by 
  sorry  -- The proof will show the maximum value is approximately 0.8817.

noncomputable def max_value := (7.75 + 7.75 + 10.95) / 30

end NUMINAMATH_GPT_max_ratio_of_right_triangle_l721_72143


namespace NUMINAMATH_GPT_distinct_values_l721_72153

-- Define the expressions as terms in Lean
def expr1 : ℕ := 3 ^ (3 ^ 3)
def expr2 : ℕ := (3 ^ 3) ^ 3

-- State the theorem that these terms yield exactly two distinct values
theorem distinct_values : (expr1 ≠ expr2) ∧ ((expr1 = 3^27) ∨ (expr1 = 19683)) ∧ ((expr2 = 3^27) ∨ (expr2 = 19683)) := 
  sorry

end NUMINAMATH_GPT_distinct_values_l721_72153


namespace NUMINAMATH_GPT_higher_amount_is_sixty_l721_72128

theorem higher_amount_is_sixty (R : ℕ) (n : ℕ) (H : ℝ) 
  (h1 : 2000 = 40 * n + H * R)
  (h2 : 1800 = 40 * (n + 10) + H * (R - 10)) :
  H = 60 :=
by
  sorry

end NUMINAMATH_GPT_higher_amount_is_sixty_l721_72128


namespace NUMINAMATH_GPT_axis_of_symmetry_parabola_l721_72131

theorem axis_of_symmetry_parabola : 
  (∃ a b c : ℝ, ∀ x : ℝ, (y = x^2 + 4 * x - 5) ∧ (a = 1) ∧ (b = 4) → ( x = -b / (2 * a) ) → ( x = -2 ) ) :=
by
  sorry

end NUMINAMATH_GPT_axis_of_symmetry_parabola_l721_72131


namespace NUMINAMATH_GPT_max_perimeter_isosceles_triangle_l721_72144

/-- Out of all triangles with the same base and the same angle at the vertex, 
    the triangle with the largest perimeter is isosceles -/
theorem max_perimeter_isosceles_triangle {α β γ : ℝ} (b : ℝ) (B : ℝ) (A C : ℝ) 
  (hB : 0 < B ∧ B < π) (hβ : α + C = B) (h1 : A = β) (h2 : γ = β) :
  α = γ := sorry

end NUMINAMATH_GPT_max_perimeter_isosceles_triangle_l721_72144


namespace NUMINAMATH_GPT_where_to_place_minus_sign_l721_72188

theorem where_to_place_minus_sign :
  (6 + 9 + 12 + 15 + 18 + 21 - 2 * 18) = 45 :=
by
  sorry

end NUMINAMATH_GPT_where_to_place_minus_sign_l721_72188


namespace NUMINAMATH_GPT_initial_population_l721_72100

theorem initial_population (P : ℝ) (h : (0.9 : ℝ)^2 * P = 4860) : P = 6000 :=
by
  sorry

end NUMINAMATH_GPT_initial_population_l721_72100


namespace NUMINAMATH_GPT_rod_total_length_l721_72167

theorem rod_total_length
  (n : ℕ) (l : ℝ)
  (h₁ : n = 50)
  (h₂ : l = 0.85) :
  n * l = 42.5 := by
  sorry

end NUMINAMATH_GPT_rod_total_length_l721_72167


namespace NUMINAMATH_GPT_quotient_remainder_difference_l721_72114

theorem quotient_remainder_difference :
  ∀ (N Q Q' R : ℕ), 
    N = 75 →
    N = 5 * Q →
    N = 34 * Q' + R →
    Q > R →
    Q - R = 8 :=
by
  intros N Q Q' R hN hDiv5 hDiv34 hGt
  sorry

end NUMINAMATH_GPT_quotient_remainder_difference_l721_72114


namespace NUMINAMATH_GPT_cubic_identity_l721_72193

theorem cubic_identity (x y z : ℝ) (h1 : x + y + z = 13) (h2 : xy + xz + yz = 32) :
  x^3 + y^3 + z^3 - 3 * x * y * z = 949 :=
by
  sorry

end NUMINAMATH_GPT_cubic_identity_l721_72193


namespace NUMINAMATH_GPT_triangle_side_height_inequality_l721_72182

theorem triangle_side_height_inequality (a b h_a h_b S : ℝ) (h1 : a > b) 
  (h2: h_a = 2 * S / a) (h3: h_b = 2 * S / b) :
  a + h_a ≥ b + h_b :=
by sorry

end NUMINAMATH_GPT_triangle_side_height_inequality_l721_72182


namespace NUMINAMATH_GPT_min_policemen_needed_l721_72164

-- Definitions of the problem parameters
def city_layout (n m : ℕ) := n > 0 ∧ m > 0

-- Function to calculate the minimum number of policemen
def min_policemen (n m : ℕ) : ℕ := (m - 1) * (n - 1)

-- The theorem to prove
theorem min_policemen_needed (n m : ℕ) (h : city_layout n m) : min_policemen n m = (m - 1) * (n - 1) :=
by
  unfold city_layout at h
  unfold min_policemen
  sorry

end NUMINAMATH_GPT_min_policemen_needed_l721_72164


namespace NUMINAMATH_GPT_iris_total_spending_l721_72136

theorem iris_total_spending :
  ∀ (price_jacket price_shorts price_pants : ℕ), 
  price_jacket = 10 → 
  price_shorts = 6 → 
  price_pants = 12 → 
  (3 * price_jacket + 2 * price_shorts + 4 * price_pants) = 90 :=
by
  intros price_jacket price_shorts price_pants
  sorry

end NUMINAMATH_GPT_iris_total_spending_l721_72136


namespace NUMINAMATH_GPT_sum_of_integers_l721_72152

theorem sum_of_integers (a b c : ℕ) :
  a > 1 → b > 1 → c > 1 →
  a * b * c = 1728 →
  gcd a b = 1 → gcd b c = 1 → gcd a c = 1 →
  a + b + c = 43 :=
by
  intro ha
  intro hb
  intro hc
  intro hproduct
  intro hgcd_ab
  intro hgcd_bc
  intro hgcd_ac
  sorry

end NUMINAMATH_GPT_sum_of_integers_l721_72152


namespace NUMINAMATH_GPT_innokentiy_games_l721_72137

def games_played_egor := 13
def games_played_nikita := 27
def games_played_innokentiy (N : ℕ) := N - games_played_egor

theorem innokentiy_games (N : ℕ) (h : N = games_played_nikita) : games_played_innokentiy N = 14 :=
by {
  sorry
}

end NUMINAMATH_GPT_innokentiy_games_l721_72137


namespace NUMINAMATH_GPT_abigail_fence_building_time_l721_72104

def abigail_time_per_fence (total_built: ℕ) (additional_hours: ℕ) (total_fences: ℕ): ℕ :=
  (additional_hours * 60) / (total_fences - total_built)

theorem abigail_fence_building_time :
  abigail_time_per_fence 10 8 26 = 30 :=
sorry

end NUMINAMATH_GPT_abigail_fence_building_time_l721_72104


namespace NUMINAMATH_GPT_find_13th_result_l721_72107

theorem find_13th_result
  (avg_25 : ℕ → ℕ)
  (avg_1_to_12 : ℕ → ℕ)
  (avg_14_to_25 : ℕ → ℕ)
  (h1 : avg_25 25 = 50)
  (h2 : avg_1_to_12 12 = 14)
  (h3 : avg_14_to_25 12 = 17) :
  ∃ (X : ℕ), X = 878 := sorry

end NUMINAMATH_GPT_find_13th_result_l721_72107


namespace NUMINAMATH_GPT_coordinate_plane_condition_l721_72169

theorem coordinate_plane_condition (a : ℝ) :
  a - 1 < 0 ∧ (3 * a + 1) / (a - 1) < 0 ↔ - (1 : ℝ)/3 < a ∧ a < 1 :=
by
  sorry

end NUMINAMATH_GPT_coordinate_plane_condition_l721_72169


namespace NUMINAMATH_GPT_working_mom_work_percentage_l721_72146

theorem working_mom_work_percentage :
  let total_hours_in_day := 24
  let work_hours := 8
  let gym_hours := 2
  let cooking_hours := 1.5
  let bath_hours := 0.5
  let homework_hours := 1
  let packing_hours := 0.5
  let cleaning_hours := 0.5
  let leisure_hours := 2
  let total_activity_hours := work_hours + gym_hours + cooking_hours + bath_hours + homework_hours + packing_hours + cleaning_hours + leisure_hours
  16 = total_activity_hours →
  (work_hours / total_hours_in_day) * 100 = 33.33 :=
by
  sorry

end NUMINAMATH_GPT_working_mom_work_percentage_l721_72146


namespace NUMINAMATH_GPT_contractor_absent_days_l721_72191

-- Definition of conditions
def total_days : ℕ := 30
def payment_per_work_day : ℝ := 25
def fine_per_absent_day : ℝ := 7.5
def total_payment : ℝ := 490

-- The proof statement
theorem contractor_absent_days : ∃ y : ℕ, (∃ x : ℕ, x + y = total_days ∧ payment_per_work_day * (x : ℝ) - fine_per_absent_day * (y : ℝ) = total_payment) ∧ y = 8 := 
by 
  sorry

end NUMINAMATH_GPT_contractor_absent_days_l721_72191


namespace NUMINAMATH_GPT_zero_sum_of_squares_l721_72102

theorem zero_sum_of_squares {a b : ℝ} (h : a^2 + b^2 = 0) : a = 0 ∧ b = 0 :=
sorry

end NUMINAMATH_GPT_zero_sum_of_squares_l721_72102


namespace NUMINAMATH_GPT_problem_inequality_l721_72160

theorem problem_inequality {a : ℝ} (h : ∀ x : ℝ, (x - a) * (1 - x - a) < 1) : 
  -1/2 < a ∧ a < 3/2 := by
  sorry

end NUMINAMATH_GPT_problem_inequality_l721_72160


namespace NUMINAMATH_GPT_combined_value_of_a_and_b_l721_72174

theorem combined_value_of_a_and_b :
  (∃ a b : ℝ,
    0.005 * a = 95 / 100 ∧
    b = 3 * a - 50 ∧
    a + b = 710) :=
sorry

end NUMINAMATH_GPT_combined_value_of_a_and_b_l721_72174


namespace NUMINAMATH_GPT_mean_value_of_quadrilateral_angles_l721_72166

-- Statement of the problem: mean value of interior angles in any quadrilateral is 90°
theorem mean_value_of_quadrilateral_angles : 
  (∀ (a b c d : ℝ), a + b + c + d = 360 → (a + b + c + d) / 4 = 90) :=
by
  sorry

end NUMINAMATH_GPT_mean_value_of_quadrilateral_angles_l721_72166


namespace NUMINAMATH_GPT_race_distance_l721_72123

theorem race_distance
  (A B : Type)
  (D : ℕ) -- D is the total distance of the race
  (Va Vb : ℕ) -- A's speed and B's speed
  (H1 : D / 28 = Va) -- A's speed calculated from D and time
  (H2 : (D - 56) / 28 = Vb) -- B's speed calculated from distance and time
  (H3 : 56 / 7 = Vb) -- B's speed can also be calculated directly
  (H4 : Va = D / 28)
  (H5 : Vb = (D - 56) / 28) :
  D = 280 := sorry

end NUMINAMATH_GPT_race_distance_l721_72123


namespace NUMINAMATH_GPT_find_AX_length_l721_72139

theorem find_AX_length (t BC AC BX : ℝ) (AX AB : ℝ)
  (h1 : t = 0.75)
  (h2 : AX = t * AB)
  (h3 : BC = 40)
  (h4 : AC = 35)
  (h5 : BX = 15) :
  AX = 105 / 8 := 
  sorry

end NUMINAMATH_GPT_find_AX_length_l721_72139


namespace NUMINAMATH_GPT_reading_enhusiasts_not_related_to_gender_l721_72155

noncomputable def contingency_table (boys_scores : List Nat) (girls_scores : List Nat) :
  (Nat × Nat × Nat × Nat × Nat × Nat) × (Nat × Nat × Nat × Nat × Nat × Nat) :=
  let boys_range := (2, 3, 5, 15, 18, 12)
  let girls_range := (0, 5, 10, 10, 7, 13)
  ((2, 3, 5, 15, 18, 12), (0, 5, 10, 10, 7, 13))

theorem reading_enhusiasts_not_related_to_gender (boys_scores : List Nat) (girls_scores : List Nat) :
  let table := contingency_table boys_scores girls_scores
  let (boys_range, girls_range) := table
  let a := 45 -- Boys who are reading enthusiasts
  let b := 10 -- Boys who are non-reading enthusiasts
  let c := 30 -- Girls who are reading enthusiasts
  let d := 15 -- Girls who are non-reading enthusiasts
  let n := a + b + c + d
  let k_squared := (n * (a * d - b * c)^2) / ((a + b) * (c + d) * (a + c) * (b + d))
  k_squared < 3.841 := 
sorry

end NUMINAMATH_GPT_reading_enhusiasts_not_related_to_gender_l721_72155


namespace NUMINAMATH_GPT_period_of_f_is_4_and_f_2pow_n_zero_l721_72108

noncomputable def f : ℝ → ℝ := sorry

variables (hf_diff : differentiable ℝ f)
          (hf_nonzero : ∃ x, f x ≠ 0)
          (hf_odd_2 : ∀ x, f (x + 2) = -f (-x - 2))
          (hf_even_2x1 : ∀ x, f (2 * x + 1) = f (-(2 * x + 1)))

theorem period_of_f_is_4_and_f_2pow_n_zero (n : ℕ) (hn : 0 < n) :
  (∀ x, f (x + 4) = f x) ∧ f (2^n) = 0 :=
sorry

end NUMINAMATH_GPT_period_of_f_is_4_and_f_2pow_n_zero_l721_72108


namespace NUMINAMATH_GPT_min_value_of_expression_l721_72187

theorem min_value_of_expression (a b c : ℝ) (ha : 0 < a ∧ a < 1) (hb : 0 < b ∧ b < 1) (hc : 0 < c ∧ c < 1)
  (habc : a + b + c = 1) (expected_value : 3 * a + 2 * b = 2) :
  ∃ a b, (a + b + (1 - a - b) = 1) ∧ (3 * a + 2 * b = 2) ∧ (∀ a b, ∃ m, m = (2/a + 1/(3*b)) ∧ m = 16/3) :=
sorry

end NUMINAMATH_GPT_min_value_of_expression_l721_72187


namespace NUMINAMATH_GPT_evaluate_expression_l721_72151

def S (a b c : ℤ) := a + b + c

theorem evaluate_expression (a b c : ℤ) (h1 : a = 12) (h2 : b = 14) (h3 : c = 18) :
  (144 * ((1 : ℚ) / b - (1 : ℚ) / c) + 196 * ((1 : ℚ) / c - (1 : ℚ) / a) + 324 * ((1 : ℚ) / a - (1 : ℚ) / b)) /
  (12 * ((1 : ℚ) / b - (1 : ℚ) / c) + 14 * ((1 : ℚ) / c - (1 : ℚ) / a) + 18 * ((1 : ℚ) / a - (1 : ℚ) / b)) = 44 := 
sorry

end NUMINAMATH_GPT_evaluate_expression_l721_72151


namespace NUMINAMATH_GPT_rationalize_denominator_sqrt_l721_72118

theorem rationalize_denominator_sqrt (x y : ℝ) (hx : x = 5) (hy : y = 12) :
  Real.sqrt (x / y) = Real.sqrt 15 / 6 :=
by
  rw [hx, hy]
  sorry

end NUMINAMATH_GPT_rationalize_denominator_sqrt_l721_72118


namespace NUMINAMATH_GPT_mean_of_combined_sets_l721_72119

theorem mean_of_combined_sets (A : Finset ℝ) (B : Finset ℝ)
  (hA_len : A.card = 7) (hB_len : B.card = 8)
  (hA_mean : (A.sum id) / 7 = 15) (hB_mean : (B.sum id) / 8 = 22) :
  (A.sum id + B.sum id) / 15 = 18.73 :=
by sorry

end NUMINAMATH_GPT_mean_of_combined_sets_l721_72119


namespace NUMINAMATH_GPT_find_x_l721_72171

theorem find_x (x y : ℚ) (h1 : 3 * x - 2 * y = 7) (h2 : x + 3 * y = 8) : x = 37 / 11 := by
  sorry

end NUMINAMATH_GPT_find_x_l721_72171


namespace NUMINAMATH_GPT_A_and_B_together_complete_work_in_24_days_l721_72138

-- Define the variables
variables {W_A W_B : ℝ} (completeTime : ℝ → ℝ → ℝ)

-- Define conditions
def A_better_than_B (W_A W_B : ℝ) := W_A = 2 * W_B
def A_takes_36_days (W_A : ℝ) := W_A = 1 / 36

-- The proposition to prove
theorem A_and_B_together_complete_work_in_24_days 
  (h1 : A_better_than_B W_A W_B)
  (h2 : A_takes_36_days W_A) :
  completeTime W_A W_B = 24 :=
sorry

end NUMINAMATH_GPT_A_and_B_together_complete_work_in_24_days_l721_72138


namespace NUMINAMATH_GPT_equation_of_circle_l721_72192

variable (x y : ℝ)

def center_line : ℝ → ℝ := fun x => -4 * x
def tangent_line : ℝ → ℝ := fun x => 1 - x

def P : ℝ × ℝ := (3, -2)
def center_O : ℝ × ℝ := (1, -4)

theorem equation_of_circle :
  (x - 1)^2 + (y + 4)^2 = 8 :=
sorry

end NUMINAMATH_GPT_equation_of_circle_l721_72192


namespace NUMINAMATH_GPT_temperature_drop_change_l721_72101

theorem temperature_drop_change (T : ℝ) (h1 : T + 2 = T + 2) :
  (T - 4) - T = -4 :=
by
  sorry

end NUMINAMATH_GPT_temperature_drop_change_l721_72101


namespace NUMINAMATH_GPT_max_value_of_g_l721_72173

def g (n : ℕ) : ℕ :=
  if n < 20 then n + 20 else g (n - 7)

theorem max_value_of_g : ∀ n : ℕ, g n ≤ 39 ∧ (∃ m : ℕ, g m = 39) := by
  sorry

end NUMINAMATH_GPT_max_value_of_g_l721_72173


namespace NUMINAMATH_GPT_problem_part_one_problem_part_two_l721_72195

theorem problem_part_one : 23 - 17 - (-6) + (-16) = -4 :=
by
  sorry

theorem problem_part_two : 0 - 32 / ((-2)^3 - (-4)) = 8 :=
by
  sorry

end NUMINAMATH_GPT_problem_part_one_problem_part_two_l721_72195


namespace NUMINAMATH_GPT_distance_rowed_upstream_l721_72179

noncomputable def speed_of_boat_in_still_water := 18 -- from solution step; b = 18 km/h
def speed_of_stream := 3 -- given
def time := 4 -- given
def distance_downstream := 84 -- given

theorem distance_rowed_upstream 
  (b : ℕ) (s : ℕ) (t : ℕ) (d_down : ℕ) (d_up : ℕ)
  (h_stream : s = 3) 
  (h_time : t = 4)
  (h_distance_downstream : d_down = 84) 
  (h_speed_boat : b = 18) 
  (h_effective_downstream_speed : b + s = d_down / t) :
  d_up = 60 := by
  sorry

end NUMINAMATH_GPT_distance_rowed_upstream_l721_72179


namespace NUMINAMATH_GPT_additional_distance_to_achieve_target_average_speed_l721_72159

-- Given conditions
def initial_distance : ℕ := 20
def initial_speed : ℕ := 40
def target_average_speed : ℕ := 55

-- Prove that the additional distance required to average target speed is 90 miles
theorem additional_distance_to_achieve_target_average_speed 
  (total_distance : ℕ) 
  (total_time : ℚ) 
  (additional_distance : ℕ) 
  (additional_speed : ℕ) :
  total_distance = initial_distance + additional_distance →
  total_time = (initial_distance / initial_speed) + (additional_distance / additional_speed) →
  additional_speed = 60 →
  total_distance / total_time = target_average_speed →
  additional_distance = 90 :=
by 
  sorry

end NUMINAMATH_GPT_additional_distance_to_achieve_target_average_speed_l721_72159


namespace NUMINAMATH_GPT_shrimp_price_l721_72197

theorem shrimp_price (y : ℝ) (h : 0.6 * (y / 4) = 2.25) : y = 15 :=
sorry

end NUMINAMATH_GPT_shrimp_price_l721_72197


namespace NUMINAMATH_GPT_tax_amount_self_employed_l721_72129

noncomputable def gross_income : ℝ := 350000.00
noncomputable def tax_rate : ℝ := 0.06

theorem tax_amount_self_employed :
  gross_income * tax_rate = 21000.00 :=
by
  sorry

end NUMINAMATH_GPT_tax_amount_self_employed_l721_72129


namespace NUMINAMATH_GPT_max_grandchildren_l721_72186

theorem max_grandchildren (children_count : ℕ) (common_gc : ℕ) (special_gc_count : ℕ) : 
  children_count = 8 ∧ common_gc = 8 ∧ special_gc_count = 5 →
  (6 * common_gc + 2 * special_gc_count) = 58 := by
  sorry

end NUMINAMATH_GPT_max_grandchildren_l721_72186


namespace NUMINAMATH_GPT_find_n_l721_72130

theorem find_n (n : ℤ) : 43^2 = 1849 ∧ 44^2 = 1936 ∧ 45^2 = 2025 ∧ 46^2 = 2116 ∧ n < Real.sqrt 2023 ∧ Real.sqrt 2023 < n + 1 → n = 44 :=
by
  sorry

end NUMINAMATH_GPT_find_n_l721_72130


namespace NUMINAMATH_GPT_inequality_has_exactly_one_solution_l721_72115

-- Definitions based on the conditions
def f (x a : ℝ) : ℝ := x^2 + 2 * a * x + 3 * a

-- The main theorem that encodes the proof problem
theorem inequality_has_exactly_one_solution (a : ℝ) : 
  (∃! x : ℝ, |f x a| ≤ 2) ↔ (a = 1 ∨ a = 2) :=
sorry

end NUMINAMATH_GPT_inequality_has_exactly_one_solution_l721_72115


namespace NUMINAMATH_GPT_greatest_possible_value_of_a_l721_72168

theorem greatest_possible_value_of_a :
  ∃ (a : ℕ), (∀ (x : ℤ), x * (x + a) = -21 → x^2 + a * x + 21 = 0) ∧
  (∀ (a' : ℕ), (∀ (x : ℤ), x * (x + a') = -21 → x^2 + a' * x + 21 = 0) → a' ≤ a) ∧
  a = 22 :=
sorry

end NUMINAMATH_GPT_greatest_possible_value_of_a_l721_72168


namespace NUMINAMATH_GPT_find_value_of_c_l721_72172

variable (a b c : ℚ)
variable (x : ℚ)

-- Conditions converted to Lean statements
def condition1 := a = 2 * x ∧ b = 3 * x ∧ c = 7 * x
def condition2 := a - b + 3 = c - 2 * b

theorem find_value_of_c : condition1 x a b c ∧ condition2 a b c → c = 21 / 2 :=
by 
  sorry

end NUMINAMATH_GPT_find_value_of_c_l721_72172


namespace NUMINAMATH_GPT_hypotenuse_length_l721_72156

theorem hypotenuse_length (a b c : ℝ) (h1 : a + b + c = 36) (h2 : 0.5 * a * b = 24) (h3 : a^2 + b^2 = c^2) :
  c = 50 / 3 :=
sorry

end NUMINAMATH_GPT_hypotenuse_length_l721_72156


namespace NUMINAMATH_GPT_inverse_proportion_range_l721_72198

theorem inverse_proportion_range (m : ℝ) :
  (∀ x : ℝ, x ≠ 0 → (y = (m + 5) / x) → ((x > 0 → y < 0) ∧ (x < 0 → y > 0))) →
  m < -5 :=
by
  intros h
  -- Skipping proof with sorry as specified
  sorry

end NUMINAMATH_GPT_inverse_proportion_range_l721_72198


namespace NUMINAMATH_GPT_evaluate_expression_eq_neg_one_evaluate_expression_only_value_l721_72117

variable (a y : ℝ)
variable (h1 : a ≠ 0)
variable (h2 : a ≠ 2 * y)
variable (h3 : a ≠ -2 * y)

theorem evaluate_expression_eq_neg_one
  (h : y = -a / 3) :
  ( (a / (a + 2 * y) + y / (a - 2 * y)) / (y / (a + 2 * y) - a / (a - 2 * y)) ) = -1 := 
sorry

theorem evaluate_expression_only_value :
  ( (a / (a + 2 * y) + y / (a - 2 * y)) / (y / (a + 2 * y) - a / (a - 2 * y)) = -1 ) ↔ 
  y = -a / 3 := 
sorry

end NUMINAMATH_GPT_evaluate_expression_eq_neg_one_evaluate_expression_only_value_l721_72117


namespace NUMINAMATH_GPT_fencing_required_l721_72121

theorem fencing_required (L W : ℕ) (hL : L = 30) (hArea : L * W = 720) : L + 2 * W = 78 :=
by
  sorry

end NUMINAMATH_GPT_fencing_required_l721_72121


namespace NUMINAMATH_GPT_shaded_region_area_eq_108_l721_72157

/-- There are two concentric circles, where the outer circle has twice the radius of the inner circle,
and the total boundary length of the shaded region is 36π. Prove that the area of the shaded region
is nπ, where n = 108. -/
theorem shaded_region_area_eq_108 (r : ℝ) (h_outer : ∀ (c₁ c₂ : ℝ), c₁ = 2 * c₂) 
  (h_boundary : 2 * Real.pi * r + 2 * Real.pi * (2 * r) = 36 * Real.pi) : 
  ∃ (n : ℕ), n = 108 ∧ (Real.pi * (2 * r)^2 - Real.pi * r^2) = n * Real.pi := 
sorry

end NUMINAMATH_GPT_shaded_region_area_eq_108_l721_72157


namespace NUMINAMATH_GPT_area_of_triangle_ABC_l721_72141

variable {α : Type} [LinearOrder α] [Field α]

-- Given: 
variables (A B C D E F : α) (area_ABC area_BDA area_DCA : α)

-- Conditions:
variable (midpoint_D : 2 * D = B + C)
variable (ratio_AE_EC : 3 * E = A + C)
variable (ratio_AF_FD : 2 * F = A + D)
variable (area_DEF : area_ABC / 6 = 12)

-- To Show:
theorem area_of_triangle_ABC :
  area_ABC = 96 :=
by
  sorry

end NUMINAMATH_GPT_area_of_triangle_ABC_l721_72141


namespace NUMINAMATH_GPT_solution_set_of_inequality_l721_72170

noncomputable def f : ℝ → ℝ := sorry

theorem solution_set_of_inequality 
  (hf_even : ∀ x : ℝ, f x = f (|x|))
  (hf_increasing : ∀ x y : ℝ, x < y → x < 0 → y < 0 → f x < f y)
  (hf_value : f 3 = 1) :
  {x : ℝ | f (x - 1) < 1} = {x : ℝ | x > 4 ∨ x < -2} := 
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l721_72170


namespace NUMINAMATH_GPT_prove_AB_and_circle_symmetry_l721_72163

-- Definition of point A
def pointA : ℝ × ℝ := (4, -3)

-- Lengths relation |AB| = 2|OA|
def lengths_relation(u v : ℝ) : Prop :=
  u^2 + v^2 = 100

-- Orthogonality condition for AB and OA
def orthogonality_condition(u v : ℝ) : Prop :=
  4 * u - 3 * v = 0

-- Condition that ordinate of B is greater than 0
def ordinate_condition(v : ℝ) : Prop :=
  v - 3 > 0

-- Equation of the circle given in the problem
def given_circle_eqn(x y : ℝ) : Prop :=
  (x - 3)^2 + (y + 1)^2 = 10

-- Symmetric circle equation to be proved
def symmetric_circle_eqn(x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 3)^2 = 10

theorem prove_AB_and_circle_symmetry :
  (∃ u v : ℝ, lengths_relation u v ∧ orthogonality_condition u v ∧ ordinate_condition v ∧ u = 6 ∧ v = 8) ∧
  (∃ x y : ℝ, given_circle_eqn x y → symmetric_circle_eqn x y) :=
by
  sorry

end NUMINAMATH_GPT_prove_AB_and_circle_symmetry_l721_72163


namespace NUMINAMATH_GPT_fraction_of_menu_l721_72133

def total_dishes (total : ℕ) : Prop := 
  6 = (1/4:ℚ) * total

def vegan_dishes (vegan : ℕ) (soy_free : ℕ) : Prop :=
  vegan = 6 ∧ soy_free = vegan - 5

theorem fraction_of_menu (total vegan soy_free : ℕ) (h1 : total_dishes total)
  (h2 : vegan_dishes vegan soy_free) : (soy_free:ℚ) / total = 1 / 24 := 
by sorry

end NUMINAMATH_GPT_fraction_of_menu_l721_72133


namespace NUMINAMATH_GPT_problem_solution_l721_72110

variables {f : ℝ → ℝ}

-- f is monotonically decreasing on [1, 3]
def monotone_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f y ≤ f x

-- f(x+3) is an even function
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 3) = f (3 - x)

-- Given conditions
axiom mono_dec : monotone_decreasing_on f 1 3
axiom even_f : even_function f

-- To prove: f(π) < f(2) < f(5)
theorem problem_solution : f π < f 2 ∧ f 2 < f 5 :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l721_72110


namespace NUMINAMATH_GPT_average_weight_of_whole_class_l721_72105

theorem average_weight_of_whole_class :
  let students_A := 26
  let students_B := 34
  let avg_weight_A := 50
  let avg_weight_B := 30
  let total_weight_A := avg_weight_A * students_A
  let total_weight_B := avg_weight_B * students_B
  let total_weight_class := total_weight_A + total_weight_B
  let total_students_class := students_A + students_B
  let avg_weight_class := total_weight_class / total_students_class
  avg_weight_class = 38.67 :=
by {
  sorry -- Proof is not required as per instructions
}

end NUMINAMATH_GPT_average_weight_of_whole_class_l721_72105


namespace NUMINAMATH_GPT_find_a_l721_72142

noncomputable def circle1 (x y : ℝ) := x^2 + y^2 + 4 * y = 0

noncomputable def circle2 (x y a : ℝ) := x^2 + y^2 + 2 * (a - 1) * x + 2 * y + a^2 = 0

theorem find_a (a : ℝ) :
  (∀ x y, circle1 x y → circle2 x y a → false) → a = -2 :=
by sorry

end NUMINAMATH_GPT_find_a_l721_72142


namespace NUMINAMATH_GPT_michael_class_choosing_l721_72145

open Nat

theorem michael_class_choosing :
  (choose 6 3) * (choose 4 2) + (choose 6 4) * (choose 4 1) + (choose 6 5) = 186 := 
by
  sorry

end NUMINAMATH_GPT_michael_class_choosing_l721_72145


namespace NUMINAMATH_GPT_total_price_of_order_l721_72125

theorem total_price_of_order :
  let num_ice_cream_bars := 225
  let price_per_ice_cream_bar := 0.60
  let num_sundaes := 125
  let price_per_sundae := 0.52
  (num_ice_cream_bars * price_per_ice_cream_bar + num_sundaes * price_per_sundae) = 200 := 
by
  -- The proof steps go here
  sorry

end NUMINAMATH_GPT_total_price_of_order_l721_72125


namespace NUMINAMATH_GPT_find_x_values_l721_72109

theorem find_x_values (x : ℝ) (h : x + 60 / (x - 3) = -12) : x = -3 ∨ x = -6 :=
sorry

end NUMINAMATH_GPT_find_x_values_l721_72109


namespace NUMINAMATH_GPT_sum_of_infinite_series_l721_72116

theorem sum_of_infinite_series :
  ∑' n, (1 : ℝ) / ((2 * n + 1)^2 - (2 * n - 1)^2) * ((1 : ℝ) / (2 * n - 1)^2 - (1 : ℝ) / (2 * n + 1)^2) = 1 :=
sorry

end NUMINAMATH_GPT_sum_of_infinite_series_l721_72116


namespace NUMINAMATH_GPT_count_three_digit_integers_with_product_thirty_l721_72113

theorem count_three_digit_integers_with_product_thirty :
  (∃ S : Finset (ℕ × ℕ × ℕ),
      (∀ (a b c : ℕ), (a, b, c) ∈ S → a * b * c = 30 ∧ 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ 1 ≤ c ∧ c ≤ 9) 
    ∧ S.card = 12) :=
by
  sorry

end NUMINAMATH_GPT_count_three_digit_integers_with_product_thirty_l721_72113


namespace NUMINAMATH_GPT_password_problem_l721_72103

theorem password_problem (n : ℕ) :
  (n^4 - n * (n - 1) * (n - 2) * (n - 3) = 936) → n = 6 :=
by
  sorry

end NUMINAMATH_GPT_password_problem_l721_72103


namespace NUMINAMATH_GPT_total_distance_100_l721_72185

-- Definitions for the problem conditions:
def initial_velocity : ℕ := 40
def common_difference : ℕ := 10
def total_time (v₀ : ℕ) (d : ℕ) : ℕ := (v₀ / d) + 1  -- The total time until the car stops
def distance_traveled (v₀ : ℕ) (d : ℕ) : ℕ :=
  (v₀ * total_time v₀ d) - (d * total_time v₀ d * (total_time v₀ d - 1)) / 2

-- Statement to prove:
theorem total_distance_100 : distance_traveled initial_velocity common_difference = 100 := by
  sorry

end NUMINAMATH_GPT_total_distance_100_l721_72185


namespace NUMINAMATH_GPT_gas_fee_calculation_l721_72122

theorem gas_fee_calculation (x : ℚ) (h_usage : x > 60) :
  60 * 0.8 + (x - 60) * 1.2 = 0.88 * x → x * 0.88 = 66 := by
  sorry

end NUMINAMATH_GPT_gas_fee_calculation_l721_72122

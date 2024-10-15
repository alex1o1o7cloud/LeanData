import Mathlib

namespace NUMINAMATH_CALUDE_photo_arrangement_l401_40122

/-- The number of ways to select and permute 3 people out of 8, keeping the rest in place -/
theorem photo_arrangement (n m : ℕ) (hn : n = 8) (hm : m = 3) : 
  (n.choose m) * (Nat.factorial m) = 336 := by
  sorry

end NUMINAMATH_CALUDE_photo_arrangement_l401_40122


namespace NUMINAMATH_CALUDE_expected_value_unfair_coin_expected_value_zero_l401_40163

/-- The expected monetary value of a single flip of an unfair coin -/
theorem expected_value_unfair_coin (p_heads : ℝ) (p_tails : ℝ) 
  (value_heads : ℝ) (value_tails : ℝ) : ℝ :=
  p_heads * value_heads + p_tails * value_tails

/-- Proof that the expected monetary value of the specific unfair coin is 0 -/
theorem expected_value_zero : 
  expected_value_unfair_coin (2/3) (1/3) 5 (-10) = 0 := by
sorry

end NUMINAMATH_CALUDE_expected_value_unfair_coin_expected_value_zero_l401_40163


namespace NUMINAMATH_CALUDE_parabola_line_intersection_property_l401_40103

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

end NUMINAMATH_CALUDE_parabola_line_intersection_property_l401_40103


namespace NUMINAMATH_CALUDE_ab_multiplier_l401_40123

theorem ab_multiplier (a b : ℚ) (h1 : 6 * a = 20) (h2 : 7 * b = 20) : ∃ n : ℚ, n * (a * b) = 800 ∧ n = 84 := by
  sorry

end NUMINAMATH_CALUDE_ab_multiplier_l401_40123


namespace NUMINAMATH_CALUDE_angle_equality_l401_40115

theorem angle_equality (α : Real) : 
  0 ≤ α ∧ α < 2 * Real.pi ∧ 
  (Real.sin α = Real.sin (215 * Real.pi / 180)) ∧ 
  (Real.cos α = Real.cos (215 * Real.pi / 180)) →
  α = 235 * Real.pi / 180 := by
sorry

end NUMINAMATH_CALUDE_angle_equality_l401_40115


namespace NUMINAMATH_CALUDE_set_C_elements_l401_40181

def A : Set ℕ := {1, 3, 5, 7}
def B : Set ℕ := {2, 4, 6}
def C : Set (ℕ × ℕ) := {p | p.1 ∈ A ∧ p.2 ∈ B}

theorem set_C_elements : C = {(1,2), (1,4), (1,6), (3,2), (3,4), (3,6), (5,2), (5,4), (5,6), (7,2), (7,4), (7,6)} := by
  sorry

end NUMINAMATH_CALUDE_set_C_elements_l401_40181


namespace NUMINAMATH_CALUDE_merchant_profit_l401_40102

theorem merchant_profit (cost : ℝ) (selling_price : ℝ) : 
  cost = 30 ∧ selling_price = 39 → 
  selling_price = cost + (cost * cost / 100) := by
sorry

end NUMINAMATH_CALUDE_merchant_profit_l401_40102


namespace NUMINAMATH_CALUDE_vector_operation_l401_40100

/-- Given vectors a and b in ℝ², prove that (1/2)a - (3/2)b equals (-1,2) -/
theorem vector_operation (a b : ℝ × ℝ) (h1 : a = (1, 1)) (h2 : b = (1, -1)) :
  (1/2 : ℝ) • a - (3/2 : ℝ) • b = (-1, 2) := by sorry

end NUMINAMATH_CALUDE_vector_operation_l401_40100


namespace NUMINAMATH_CALUDE_product_multiple_of_16_probability_l401_40146

def S : Finset ℕ := {3, 4, 8, 16}

theorem product_multiple_of_16_probability :
  let pairs := S.powerset.filter (λ p : Finset ℕ => p.card = 2)
  let valid_pairs := pairs.filter (λ p : Finset ℕ => (p.prod id) % 16 = 0)
  (valid_pairs.card : ℚ) / pairs.card = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_product_multiple_of_16_probability_l401_40146


namespace NUMINAMATH_CALUDE_sum_interior_angles_regular_pentagon_l401_40130

/-- The sum of interior angles of a regular polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- A regular pentagon has 5 sides -/
def regular_pentagon_sides : ℕ := 5

/-- Theorem: The sum of the interior angles of a regular pentagon is 540 degrees -/
theorem sum_interior_angles_regular_pentagon :
  sum_interior_angles regular_pentagon_sides = 540 := by
  sorry


end NUMINAMATH_CALUDE_sum_interior_angles_regular_pentagon_l401_40130


namespace NUMINAMATH_CALUDE_product_remainder_zero_l401_40138

theorem product_remainder_zero (a b c : ℕ) (ha : a = 1256) (hb : b = 7921) (hc : c = 70305) :
  (a * b * c) % 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_zero_l401_40138


namespace NUMINAMATH_CALUDE_solution_set_inequality_l401_40191

theorem solution_set_inequality (x : ℝ) :
  (x - 3) * (x - 1) > 0 ↔ x < 1 ∨ x > 3 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l401_40191


namespace NUMINAMATH_CALUDE_masha_sasha_numbers_l401_40125

theorem masha_sasha_numbers 
  (a b : ℕ) 
  (h_distinct : a ≠ b) 
  (h_greater_11 : a > 11 ∧ b > 11) 
  (h_sum_known : ∃ s, s = a + b) 
  (h_one_even : Even a ∨ Even b) 
  (h_unique : ∀ x y : ℕ, x ≠ y → x > 11 → y > 11 → x + y = a + b → (Even x ∨ Even y) → (x = a ∧ y = b) ∨ (x = b ∧ y = a)) :
  (a = 12 ∧ b = 16) ∨ (a = 16 ∧ b = 12) :=
sorry

end NUMINAMATH_CALUDE_masha_sasha_numbers_l401_40125


namespace NUMINAMATH_CALUDE_isosceles_triangle_l401_40164

theorem isosceles_triangle (A B C : Real) (h : (Real.sin A + Real.sin B) * (Real.cos A + Real.cos B) = 2 * Real.sin C) : A = B := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_l401_40164


namespace NUMINAMATH_CALUDE_library_books_taken_out_l401_40194

theorem library_books_taken_out (initial_books : ℕ) (books_returned : ℕ) (books_taken_out : ℕ) (final_books : ℕ) :
  initial_books = 235 →
  books_returned = 56 →
  books_taken_out = 35 →
  final_books = 29 →
  ∃ (tuesday_books : ℕ), tuesday_books = 227 ∧ 
    initial_books - tuesday_books + books_returned - books_taken_out = final_books :=
by
  sorry


end NUMINAMATH_CALUDE_library_books_taken_out_l401_40194


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fourth_term_l401_40137

/-- Given an arithmetic sequence where the sum of the third and fifth terms is 10,
    prove that the fourth term is 5 -/
theorem arithmetic_sequence_fourth_term
  (b : ℝ) -- third term
  (d : ℝ) -- common difference
  (h : b + (b + 2*d) = 10) -- sum of third and fifth terms is 10
  : b + d = 5 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fourth_term_l401_40137


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l401_40180

theorem complex_number_in_first_quadrant :
  let z : ℂ := Complex.I / (2 + Complex.I)
  (z.re > 0) ∧ (z.im > 0) :=
by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l401_40180


namespace NUMINAMATH_CALUDE_pencil_count_l401_40168

/-- The number of pencils Mitchell and Antonio have together -/
def total_pencils (mitchell_pencils : ℕ) (difference : ℕ) : ℕ :=
  mitchell_pencils + (mitchell_pencils - difference)

/-- Theorem stating the total number of pencils Mitchell and Antonio have -/
theorem pencil_count (mitchell_pencils : ℕ) (difference : ℕ) 
  (h1 : mitchell_pencils = 30)
  (h2 : difference = 6) : 
  total_pencils mitchell_pencils difference = 54 := by
  sorry

end NUMINAMATH_CALUDE_pencil_count_l401_40168


namespace NUMINAMATH_CALUDE_cube_volumes_sum_l401_40157

theorem cube_volumes_sum (a b c : ℕ) (h : 6 * (a^2 + b^2 + c^2) = 564) :
  a^3 + b^3 + c^3 = 764 ∨ a^3 + b^3 + c^3 = 586 :=
by sorry

end NUMINAMATH_CALUDE_cube_volumes_sum_l401_40157


namespace NUMINAMATH_CALUDE_difference_of_numbers_l401_40111

theorem difference_of_numbers (x y : ℝ) 
  (sum_condition : x + y = 36) 
  (product_condition : x * y = 320) : 
  |x - y| = 4 := by
sorry

end NUMINAMATH_CALUDE_difference_of_numbers_l401_40111


namespace NUMINAMATH_CALUDE_shadow_length_theorem_l401_40192

theorem shadow_length_theorem (α β : Real) (h : Real) 
  (shadow_length : Real → Real → Real)
  (h_shadow : ∀ θ, shadow_length h θ = h * Real.tan θ)
  (h_first_measurement : Real.tan α = 3)
  (h_angle_diff : Real.tan (α - β) = 1/3) :
  Real.tan β = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_shadow_length_theorem_l401_40192


namespace NUMINAMATH_CALUDE_complex_number_opposite_parts_l401_40109

theorem complex_number_opposite_parts (m : ℝ) : 
  let z : ℂ := (1 - m * I) / (1 - 2 * I)
  (∃ (a : ℝ), z = a - a * I) → m = -3 := by
sorry

end NUMINAMATH_CALUDE_complex_number_opposite_parts_l401_40109


namespace NUMINAMATH_CALUDE_function_properties_l401_40113

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (x^2 + 1) + x) + (2^x - 1) / (2^x + 1) + 3

def g (x : ℝ) : ℝ := sorry

theorem function_properties :
  (∀ x : ℝ, f x + f (-x) = 6) ∧
  (∀ x : ℝ, g x + g (-x) = 6) ∧
  (∀ a b : ℝ, f a + f b > 6 → a + b > 0) := by sorry

end NUMINAMATH_CALUDE_function_properties_l401_40113


namespace NUMINAMATH_CALUDE_four_digit_multiples_of_seven_l401_40152

theorem four_digit_multiples_of_seven : 
  (Finset.filter (fun n : ℕ => n % 7 = 0 ∧ 1000 ≤ n ∧ n ≤ 9999) (Finset.range 10000)).card = 1286 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_multiples_of_seven_l401_40152


namespace NUMINAMATH_CALUDE_sarah_today_cans_l401_40144

/-- The number of cans Sarah collected yesterday -/
def sarah_yesterday : ℕ := 50

/-- The number of additional cans Lara collected compared to Sarah yesterday -/
def lara_extra_yesterday : ℕ := 30

/-- The number of cans Lara collected today -/
def lara_today : ℕ := 70

/-- The difference in total cans collected between yesterday and today -/
def fewer_today : ℕ := 20

/-- Theorem: Sarah collected 40 cans today -/
theorem sarah_today_cans : 
  sarah_yesterday + (sarah_yesterday + lara_extra_yesterday) - fewer_today - lara_today = 40 := by
  sorry

end NUMINAMATH_CALUDE_sarah_today_cans_l401_40144


namespace NUMINAMATH_CALUDE_complex_arithmetic_expression_l401_40119

theorem complex_arithmetic_expression : 
  ∃ ε > 0, ε < 0.0001 ∧ 
  |(3.5 / 0.7) * (5/3 : ℝ) + (7.2 / 0.36) - ((5/3 : ℝ) * 0.75 / 0.25) - 23.3335| < ε := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_expression_l401_40119


namespace NUMINAMATH_CALUDE_unique_prime_square_equation_l401_40174

theorem unique_prime_square_equation : 
  ∃! p : ℕ, Prime p ∧ ∃ k : ℕ, 2 * p^4 - 7 * p^2 + 1 = k^2 := by sorry

end NUMINAMATH_CALUDE_unique_prime_square_equation_l401_40174


namespace NUMINAMATH_CALUDE_circle_and_tangents_l401_40159

-- Define the points A, B, and M
def A : ℝ × ℝ := (3, 2)
def B : ℝ × ℝ := (-1, 2)
def M : ℝ × ℝ := (3, 1)

-- Define circle C with AB as its diameter
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 2)^2 = 4}

-- Define the tangent lines
def tangentLine1 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = 3}
def tangentLine2 : Set (ℝ × ℝ) := {p : ℝ × ℝ | 3 * p.1 - 4 * p.2 - 5 = 0}

theorem circle_and_tangents :
  -- 1. Prove that C is the correct circle equation
  (∀ p : ℝ × ℝ, p ∈ C ↔ (p.1 - 1)^2 + (p.2 - 2)^2 = 4) ∧
  -- 2. Prove that tangentLine1 and tangentLine2 are tangent to C and pass through M
  (∀ p : ℝ × ℝ, p ∈ tangentLine1 → (p = M ∨ (∃! q : ℝ × ℝ, q ∈ C ∩ tangentLine1))) ∧
  (∀ p : ℝ × ℝ, p ∈ tangentLine2 → (p = M ∨ (∃! q : ℝ × ℝ, q ∈ C ∩ tangentLine2))) :=
sorry

end NUMINAMATH_CALUDE_circle_and_tangents_l401_40159


namespace NUMINAMATH_CALUDE_conic_is_ellipse_l401_40155

/-- Defines the equation of the conic section --/
def conic_equation (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + (y - 2)^2) + Real.sqrt ((x - 6)^2 + (y + 4)^2) = 12

/-- Theorem stating that the equation describes an ellipse --/
theorem conic_is_ellipse : ∃ (a b x₀ y₀ : ℝ), 
  (∀ x y : ℝ, conic_equation x y ↔ 
    ((x - x₀) / a)^2 + ((y - y₀) / b)^2 = 1) ∧ 
  a > 0 ∧ b > 0 ∧ a ≠ b :=
sorry

end NUMINAMATH_CALUDE_conic_is_ellipse_l401_40155


namespace NUMINAMATH_CALUDE_soccer_handshakes_l401_40186

theorem soccer_handshakes (team_size : Nat) (referee_count : Nat) (coach_count : Nat) :
  team_size = 7 ∧ referee_count = 3 ∧ coach_count = 2 →
  let player_count := 2 * team_size
  let player_player_handshakes := team_size * team_size
  let player_referee_handshakes := player_count * referee_count
  let coach_handshakes := coach_count * (player_count + referee_count)
  player_player_handshakes + player_referee_handshakes + coach_handshakes = 125 :=
by sorry


end NUMINAMATH_CALUDE_soccer_handshakes_l401_40186


namespace NUMINAMATH_CALUDE_ten_faucets_fifty_gallons_l401_40165

/-- The time (in seconds) it takes for a given number of faucets to fill a pool of a given volume. -/
def fill_time (num_faucets : ℕ) (volume : ℝ) : ℝ :=
  sorry

theorem ten_faucets_fifty_gallons
  (h1 : fill_time 5 200 = 15 * 60) -- Five faucets fill 200 gallons in 15 minutes
  (h2 : ∀ (n : ℕ) (v : ℝ), fill_time n v > 0) -- All fill times are positive
  (h3 : ∀ (n m : ℕ) (v : ℝ), n ≠ 0 → m ≠ 0 → fill_time n v * m = fill_time m v * n) -- Faucets dispense water at the same rate
  : fill_time 10 50 = 112.5 := by
  sorry

end NUMINAMATH_CALUDE_ten_faucets_fifty_gallons_l401_40165


namespace NUMINAMATH_CALUDE_sum_power_mod_five_l401_40167

theorem sum_power_mod_five (n : ℕ) :
  (1^n + 2^n + 3^n + 4^n) % 5 = 0 ↔ n % 4 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_power_mod_five_l401_40167


namespace NUMINAMATH_CALUDE_find_multiple_l401_40172

theorem find_multiple (x : ℝ) (m : ℝ) (h1 : x = 13) (h2 : x + x + 2*x + m*x = 104) : m = 4 := by
  sorry

end NUMINAMATH_CALUDE_find_multiple_l401_40172


namespace NUMINAMATH_CALUDE_floor_sum_equals_126_l401_40156

theorem floor_sum_equals_126 
  (x y z w : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) (pos_w : 0 < w)
  (eq1 : x^2 + y^2 = 2010)
  (eq2 : z^2 + w^2 = 2010)
  (eq3 : x * z = 1008)
  (eq4 : y * w = 1008) :
  ⌊x + y + z + w⌋ = 126 := by
sorry

end NUMINAMATH_CALUDE_floor_sum_equals_126_l401_40156


namespace NUMINAMATH_CALUDE_fourth_root_of_sqrt_fraction_l401_40154

theorem fourth_root_of_sqrt_fraction : 
  (32 / 10000 : ℝ)^(1/4 * 1/2) = (2 : ℝ)^(1/8) / (5 : ℝ)^(1/2) := by sorry

end NUMINAMATH_CALUDE_fourth_root_of_sqrt_fraction_l401_40154


namespace NUMINAMATH_CALUDE_james_fish_catch_l401_40148

/-- The total pounds of fish James caught -/
def total_fish (trout salmon tuna : ℕ) : ℕ := trout + salmon + tuna

/-- Proves that James caught 900 pounds of fish in total -/
theorem james_fish_catch : 
  let trout : ℕ := 200
  let salmon : ℕ := trout + trout / 2
  let tuna : ℕ := 2 * trout
  total_fish trout salmon tuna = 900 := by sorry

end NUMINAMATH_CALUDE_james_fish_catch_l401_40148


namespace NUMINAMATH_CALUDE_remainder_55_power_55_plus_10_mod_8_l401_40114

theorem remainder_55_power_55_plus_10_mod_8 : 55^55 + 10 ≡ 1 [ZMOD 8] := by
  sorry

end NUMINAMATH_CALUDE_remainder_55_power_55_plus_10_mod_8_l401_40114


namespace NUMINAMATH_CALUDE_dave_tshirts_l401_40178

/-- The number of white T-shirt packs Dave bought -/
def white_packs : ℕ := 3

/-- The number of T-shirts in each white pack -/
def white_per_pack : ℕ := 6

/-- The number of blue T-shirt packs Dave bought -/
def blue_packs : ℕ := 2

/-- The number of T-shirts in each blue pack -/
def blue_per_pack : ℕ := 4

/-- The total number of T-shirts Dave bought -/
def total_tshirts : ℕ := white_packs * white_per_pack + blue_packs * blue_per_pack

theorem dave_tshirts : total_tshirts = 26 := by
  sorry

end NUMINAMATH_CALUDE_dave_tshirts_l401_40178


namespace NUMINAMATH_CALUDE_paint_area_calculation_l401_40176

/-- Calculates the area to be painted on a wall with a door. -/
def areaToPaint (wallHeight wallLength doorHeight doorWidth : ℝ) : ℝ :=
  wallHeight * wallLength - doorHeight * doorWidth

/-- Proves that the area to be painted on a 10ft by 15ft wall with a 3ft by 5ft door is 135 sq ft. -/
theorem paint_area_calculation :
  areaToPaint 10 15 3 5 = 135 := by
  sorry

end NUMINAMATH_CALUDE_paint_area_calculation_l401_40176


namespace NUMINAMATH_CALUDE_container_count_l401_40124

theorem container_count (container_capacity : ℝ) (total_capacity : ℝ) : 
  (8 : ℝ) = 0.2 * container_capacity →
  total_capacity = 1600 →
  (total_capacity / container_capacity : ℝ) = 40 := by
  sorry

end NUMINAMATH_CALUDE_container_count_l401_40124


namespace NUMINAMATH_CALUDE_original_group_size_l401_40139

theorem original_group_size (initial_days work_days : ℕ) (absent_men : ℕ) : 
  initial_days = 15 →
  absent_men = 8 →
  work_days = 18 →
  ∃ (original_size : ℕ),
    original_size * initial_days = (original_size - absent_men) * work_days ∧
    original_size = 48 :=
by sorry

end NUMINAMATH_CALUDE_original_group_size_l401_40139


namespace NUMINAMATH_CALUDE_monotonicity_condition_max_k_value_l401_40190

noncomputable section

def f (x : ℝ) : ℝ := (x^2 - 3*x + 3) * Real.exp x

def is_monotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y ∨ (∀ z, a ≤ z ∧ z ≤ b → f z = f x)

theorem monotonicity_condition (t : ℝ) :
  (is_monotonic f (-2) t) ↔ -2 < t ∧ t ≤ 0 := by sorry

theorem max_k_value :
  ∃ k : ℕ, k = 6 ∧ 
  (∀ x : ℝ, x > 0 → (f x / Real.exp x) + 7*x - 2 > k * (x * Real.log x - 1)) ∧
  (∀ m : ℕ, m > k → ∃ x : ℝ, x > 0 ∧ (f x / Real.exp x) + 7*x - 2 ≤ m * (x * Real.log x - 1)) := by sorry

end

end NUMINAMATH_CALUDE_monotonicity_condition_max_k_value_l401_40190


namespace NUMINAMATH_CALUDE_quadratic_inequality_implies_a_geq_5_l401_40101

theorem quadratic_inequality_implies_a_geq_5 (a : ℝ) : 
  (∀ x : ℝ, 1 < x ∧ x < 5 → x^2 - 2*(a-2)*x + a < 0) → a ≥ 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_implies_a_geq_5_l401_40101


namespace NUMINAMATH_CALUDE_binomial_coefficient_x4_in_expansion_l401_40196

/-- The binomial coefficient of the term containing x^4 in the expansion of (x^2 + 1/x)^5 is 10 -/
theorem binomial_coefficient_x4_in_expansion : 
  ∃ k : ℕ, (Nat.choose 5 k) * (4 : ℤ) = (10 : ℤ) ∧ 
    10 - 3 * k = 4 := by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_x4_in_expansion_l401_40196


namespace NUMINAMATH_CALUDE_charity_show_girls_l401_40141

theorem charity_show_girls (initial_total : ℕ) (initial_girls : ℕ) : 
  initial_girls = initial_total / 2 →
  (initial_girls - 3 : ℚ) / (initial_total + 1 : ℚ) = 2/5 →
  initial_girls = 17 := by
sorry

end NUMINAMATH_CALUDE_charity_show_girls_l401_40141


namespace NUMINAMATH_CALUDE_promotion_payment_correct_l401_40170

/-- Represents the payment calculation for a clothing factory promotion -/
def promotion_payment (suit_price tie_price : ℕ) (num_suits num_ties : ℕ) : ℕ × ℕ :=
  let option1 := suit_price * num_suits + tie_price * (num_ties - num_suits)
  let option2 := ((suit_price * num_suits + tie_price * num_ties) * 9) / 10
  (option1, option2)

theorem promotion_payment_correct (x : ℕ) (h : x > 20) :
  promotion_payment 200 40 20 x = (40 * x + 3200, 3600 + 36 * x) := by
  sorry

#eval promotion_payment 200 40 20 30

end NUMINAMATH_CALUDE_promotion_payment_correct_l401_40170


namespace NUMINAMATH_CALUDE_fifteen_plus_neg_twentythree_l401_40162

-- Define the operation for adding a positive and negative rational number
def add_pos_neg (a b : ℚ) : ℚ := -(b - a)

-- Theorem statement
theorem fifteen_plus_neg_twentythree :
  15 + (-23) = add_pos_neg 15 23 :=
sorry

end NUMINAMATH_CALUDE_fifteen_plus_neg_twentythree_l401_40162


namespace NUMINAMATH_CALUDE_fraction_equality_l401_40151

theorem fraction_equality (p q : ℚ) (h : p / q = 4 / 5) :
  11 / 7 + (2 * q - p) / (2 * q + p) = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l401_40151


namespace NUMINAMATH_CALUDE_polynomial_value_l401_40182

theorem polynomial_value (x : ℝ) : 
  let a : ℝ := 2002 * x + 2003
  let b : ℝ := 2002 * x + 2004
  let c : ℝ := 2002 * x + 2005
  a^2 + b^2 + c^2 - a*b - b*c - c*a = 3 := by
sorry

end NUMINAMATH_CALUDE_polynomial_value_l401_40182


namespace NUMINAMATH_CALUDE_sin_plus_cos_range_l401_40147

theorem sin_plus_cos_range : ∀ x : ℝ, -Real.sqrt 2 ≤ Real.sin x + Real.cos x ∧ Real.sin x + Real.cos x ≤ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_plus_cos_range_l401_40147


namespace NUMINAMATH_CALUDE_exist_natural_solution_l401_40150

theorem exist_natural_solution :
  ∃ (x y z : ℕ), 28 * x + 30 * y + 31 * z = 365 := by
sorry

end NUMINAMATH_CALUDE_exist_natural_solution_l401_40150


namespace NUMINAMATH_CALUDE_count_valid_permutations_l401_40188

/-- The set of digits in the number 2033 -/
def digits : Finset ℕ := {2, 0, 3, 3}

/-- A function that checks if a number is a 4-digit number -/
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

/-- A function that calculates the sum of digits of a number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- The set of all 4-digit permutations of the digits in 2033 -/
def valid_permutations : Finset ℕ := sorry

theorem count_valid_permutations : Finset.card valid_permutations = 15 := by sorry

end NUMINAMATH_CALUDE_count_valid_permutations_l401_40188


namespace NUMINAMATH_CALUDE_sin_alpha_for_point_l401_40135

theorem sin_alpha_for_point (α : Real) :
  let P : ℝ × ℝ := (1, -Real.sqrt 3)
  (∃ (t : ℝ), t > 0 ∧ P = (t * Real.cos α, t * Real.sin α)) →
  Real.sin α = -Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_sin_alpha_for_point_l401_40135


namespace NUMINAMATH_CALUDE_michael_has_100_cards_l401_40116

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


end NUMINAMATH_CALUDE_michael_has_100_cards_l401_40116


namespace NUMINAMATH_CALUDE_number_equation_solution_l401_40193

theorem number_equation_solution : ∃ x : ℝ, 2 * x - 3 = 7 ∧ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_solution_l401_40193


namespace NUMINAMATH_CALUDE_square_sum_equals_69_l401_40142

/-- Given a system of equations, prove that x₀² + y₀² = 69 -/
theorem square_sum_equals_69 
  (x₀ y₀ c : ℝ) 
  (h1 : x₀ * y₀ = 6)
  (h2 : x₀^2 * y₀ + x₀ * y₀^2 + x₀ + y₀ + c = 2) :
  x₀^2 + y₀^2 = 69 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equals_69_l401_40142


namespace NUMINAMATH_CALUDE_longest_segment_proof_l401_40179

/-- The total length of all segments in the rectangular spiral -/
def total_length : ℕ := 3000

/-- Predicate to check if a given length satisfies the spiral condition -/
def satisfies_spiral_condition (n : ℕ) : Prop :=
  n * (n + 1) ≤ total_length

/-- The longest line segment in the rectangular spiral -/
def longest_segment : ℕ := 54

theorem longest_segment_proof :
  satisfies_spiral_condition longest_segment ∧
  ∀ m : ℕ, m > longest_segment → ¬satisfies_spiral_condition m :=
by sorry

end NUMINAMATH_CALUDE_longest_segment_proof_l401_40179


namespace NUMINAMATH_CALUDE_candy_mixture_problem_l401_40133

/-- Given two types of candy mixed to produce a mixture selling at a certain price,
    calculate the total amount of mixture produced. -/
theorem candy_mixture_problem (x : ℝ) : 
  x > 0 ∧ 
  3.50 * x + 4.30 * 6.25 = 4.00 * (x + 6.25) → 
  x + 6.25 = 10 := by
  sorry

#check candy_mixture_problem

end NUMINAMATH_CALUDE_candy_mixture_problem_l401_40133


namespace NUMINAMATH_CALUDE_square_side_length_l401_40131

theorem square_side_length (m n : ℝ) :
  let area := 9*m^2 + 24*m*n + 16*n^2
  ∃ (side : ℝ), side ≥ 0 ∧ side^2 = area ∧ side = |3*m + 4*n| :=
by sorry

end NUMINAMATH_CALUDE_square_side_length_l401_40131


namespace NUMINAMATH_CALUDE_hotline_probabilities_l401_40118

theorem hotline_probabilities (p1 p2 p3 p4 : ℝ)
  (h1 : p1 = 0.1)
  (h2 : p2 = 0.2)
  (h3 : p3 = 0.3)
  (h4 : p4 = 0.35) :
  (p1 + p2 + p3 + p4 = 0.95) ∧ (1 - (p1 + p2 + p3 + p4) = 0.05) := by
  sorry

end NUMINAMATH_CALUDE_hotline_probabilities_l401_40118


namespace NUMINAMATH_CALUDE_lake_half_covered_l401_40153

/-- Represents the number of lotuses in the lake on a given day -/
def lotuses (day : ℕ) : ℝ := 2^day

/-- The day when the lake is completely covered -/
def full_coverage_day : ℕ := 30

theorem lake_half_covered :
  lotuses (full_coverage_day - 1) = (1/2) * lotuses full_coverage_day :=
by sorry

end NUMINAMATH_CALUDE_lake_half_covered_l401_40153


namespace NUMINAMATH_CALUDE_base7_divisible_by_19_l401_40185

/-- Given a digit y, returns the decimal representation of 52y3 in base 7 -/
def base7ToDecimal (y : ℕ) : ℕ := 5 * 7^3 + 2 * 7^2 + y * 7 + 3

/-- Theorem stating that when 52y3 in base 7 is divisible by 19, y must be 8 -/
theorem base7_divisible_by_19 :
  ∃ y : ℕ, y < 7 ∧ (base7ToDecimal y) % 19 = 0 → y = 8 :=
by sorry

end NUMINAMATH_CALUDE_base7_divisible_by_19_l401_40185


namespace NUMINAMATH_CALUDE_part1_part2_l401_40121

-- Define the propositions p and q
def p (x : ℝ) : Prop := (x + 2) * (x - 4) ≤ 0
def q (m x : ℝ) : Prop := 2 - m ≤ x ∧ x ≤ 2 + m

-- Part 1
theorem part1 (m : ℝ) (h1 : m > 0) 
  (h2 : ∀ x, ¬(q m x) → ¬(p x)) 
  (h3 : ∃ x, ¬(p x) ∧ q m x) : 
  m ≥ 4 := by sorry

-- Part 2
theorem part2 (x : ℝ) 
  (h1 : p x ∨ q 5 x) 
  (h2 : ¬(p x ∧ q 5 x)) : 
  x ∈ Set.Icc (-3 : ℝ) (-2) ∪ Set.Ioc 4 7 := by sorry

end NUMINAMATH_CALUDE_part1_part2_l401_40121


namespace NUMINAMATH_CALUDE_calculate_expression_solve_equation_l401_40106

-- Problem 1
theorem calculate_expression : -3^2 + 5 * (-8/5) - (-4)^2 / (-8) = -13 := by sorry

-- Problem 2
theorem solve_equation : 
  ∃ x : ℚ, (x + 1) / 2 - 2 = x / 4 ∧ x = -4/3 := by sorry

end NUMINAMATH_CALUDE_calculate_expression_solve_equation_l401_40106


namespace NUMINAMATH_CALUDE_solve_for_N_l401_40198

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  first : ℤ
  diff : ℤ

/-- Represents the grid of numbers -/
structure NumberGrid where
  row : ArithmeticSequence
  col1 : ArithmeticSequence
  col2 : ArithmeticSequence

/-- The problem setup -/
def problem_setup : NumberGrid where
  row := { first := 21, diff := -5 }
  col1 := { first := 6, diff := 4 }
  col2 := { first := -7, diff := -2 }

/-- The theorem to prove -/
theorem solve_for_N (grid : NumberGrid) : 
  grid.row.first = 21 ∧ 
  (grid.col1.first + 3 * grid.col1.diff = 14) ∧
  (grid.col1.first + 4 * grid.col1.diff = 18) ∧
  (grid.col2.first + 4 * grid.col2.diff = -17) →
  grid.col2.first = -7 := by
  sorry

#eval problem_setup.col2.first

end NUMINAMATH_CALUDE_solve_for_N_l401_40198


namespace NUMINAMATH_CALUDE_money_in_pond_is_637_l401_40161

/-- The amount of money in cents left in the pond after all calculations -/
def moneyInPond : ℕ :=
  let dimeValue : ℕ := 10
  let quarterValue : ℕ := 25
  let halfDollarValue : ℕ := 50
  let dollarValue : ℕ := 100
  let nickelValue : ℕ := 5
  let pennyValue : ℕ := 1
  let foreignCoinValue : ℕ := 25

  let cindyMoney : ℕ := 5 * dimeValue + 3 * halfDollarValue
  let ericMoney : ℕ := 3 * quarterValue + 2 * dollarValue + halfDollarValue
  let garrickMoney : ℕ := 8 * nickelValue + 7 * pennyValue
  let ivyMoney : ℕ := 60 * pennyValue + 5 * foreignCoinValue

  let totalBefore : ℕ := cindyMoney + ericMoney + garrickMoney + ivyMoney

  let beaumontRemoval : ℕ := 2 * dimeValue + 3 * nickelValue + 10 * pennyValue
  let ericRemoval : ℕ := quarterValue + halfDollarValue

  totalBefore - beaumontRemoval - ericRemoval

theorem money_in_pond_is_637 : moneyInPond = 637 := by
  sorry

end NUMINAMATH_CALUDE_money_in_pond_is_637_l401_40161


namespace NUMINAMATH_CALUDE_floor_mod_equivalence_l401_40160

theorem floor_mod_equivalence (k : ℤ) (a b : ℝ) (h : k ≥ 2) :
  (∃ m : ℤ, a - b = m * k) ↔
  (∀ n : ℕ, n > 0 → ⌊a * n⌋ % k = ⌊b * n⌋ % k) :=
by sorry

end NUMINAMATH_CALUDE_floor_mod_equivalence_l401_40160


namespace NUMINAMATH_CALUDE_minutes_to_seconds_l401_40120

/-- The number of seconds in one minute -/
def seconds_per_minute : ℕ := 60

/-- The number of minutes we're converting to seconds -/
def minutes : ℚ := 12.5

/-- Theorem stating that 12.5 minutes is equal to 750 seconds -/
theorem minutes_to_seconds : (minutes * seconds_per_minute : ℚ) = 750 := by
  sorry

end NUMINAMATH_CALUDE_minutes_to_seconds_l401_40120


namespace NUMINAMATH_CALUDE_betty_herb_garden_l401_40187

theorem betty_herb_garden (basil thyme oregano : ℕ) : 
  basil = 5 →
  thyme = 4 →
  oregano = 2 * basil + 2 →
  basil = 3 * thyme - 3 →
  basil + oregano + thyme = 21 := by
  sorry

end NUMINAMATH_CALUDE_betty_herb_garden_l401_40187


namespace NUMINAMATH_CALUDE_race_speed_ratio_l401_40175

/-- Given a race with the following conditions:
  * The total race distance is 600 meters
  * Contestant A has a 100 meter head start
  * Contestant A wins by 200 meters
  This theorem proves that the ratio of the speeds of contestant A to contestant B is 5:4. -/
theorem race_speed_ratio (vA vB : ℝ) (vA_pos : vA > 0) (vB_pos : vB > 0) : 
  (600 - 100) / vA = 400 / vB → vA / vB = 5 / 4 := by
  sorry

end NUMINAMATH_CALUDE_race_speed_ratio_l401_40175


namespace NUMINAMATH_CALUDE_absolute_value_of_negative_2023_l401_40107

theorem absolute_value_of_negative_2023 : |(-2023 : ℝ)| = 2023 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_of_negative_2023_l401_40107


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l401_40136

theorem simplify_trig_expression :
  (Real.sin (30 * π / 180) + Real.sin (60 * π / 180)) /
  (Real.cos (30 * π / 180) + Real.cos (60 * π / 180)) =
  Real.tan (45 * π / 180) := by sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l401_40136


namespace NUMINAMATH_CALUDE_adjacent_vertices_probability_l401_40189

/-- A decagon is a polygon with 10 vertices -/
def Decagon := {n : ℕ // n = 10}

/-- The number of vertices in a decagon -/
def numVertices : Decagon → ℕ := fun _ ↦ 10

/-- The number of adjacent vertices for each vertex in a decagon -/
def numAdjacentVertices : Decagon → ℕ := fun _ ↦ 2

/-- The probability of selecting two adjacent vertices in a decagon -/
def probAdjacentVertices (d : Decagon) : ℚ :=
  (numAdjacentVertices d : ℚ) / ((numVertices d - 1) : ℚ)

theorem adjacent_vertices_probability (d : Decagon) :
  probAdjacentVertices d = 2/9 := by sorry

end NUMINAMATH_CALUDE_adjacent_vertices_probability_l401_40189


namespace NUMINAMATH_CALUDE_probability_not_triangle_l401_40112

theorem probability_not_triangle (total : ℕ) (triangles : ℕ) 
  (h1 : total = 10) (h2 : triangles = 4) : 
  (total - triangles : ℚ) / total = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_not_triangle_l401_40112


namespace NUMINAMATH_CALUDE_canadian_olympiad_2008_l401_40110

theorem canadian_olympiad_2008 (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : a + b + c = 1) :
  (a - b*c)/(a + b*c) + (b - c*a)/(b + c*a) + (c - a*b)/(c + a*b) ≤ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_canadian_olympiad_2008_l401_40110


namespace NUMINAMATH_CALUDE_ellipse_m_range_l401_40177

def is_ellipse_equation (m : ℝ) : Prop :=
  (m - 1 > 0) ∧ (3 - m > 0) ∧ (m - 1 ≠ 3 - m)

theorem ellipse_m_range :
  ∀ m : ℝ, is_ellipse_equation m → m ∈ Set.Ioo 1 2 ∪ Set.Ioo 2 3 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_m_range_l401_40177


namespace NUMINAMATH_CALUDE_meals_sold_equals_twelve_l401_40158

/-- Represents the number of meals sold during lunch in a restaurant -/
def meals_sold_during_lunch (lunch_meals : ℕ) (dinner_prep : ℕ) (dinner_available : ℕ) : ℕ :=
  lunch_meals + dinner_prep - dinner_available

/-- Theorem stating that the number of meals sold during lunch is 12 -/
theorem meals_sold_equals_twelve : 
  meals_sold_during_lunch 17 5 10 = 12 := by
  sorry

end NUMINAMATH_CALUDE_meals_sold_equals_twelve_l401_40158


namespace NUMINAMATH_CALUDE_equation_solutions_l401_40183

theorem equation_solutions : 
  ∀ x : ℝ, (x - 2)^2 = 9*x^2 ↔ x = -1 ∨ x = 1/2 := by sorry

end NUMINAMATH_CALUDE_equation_solutions_l401_40183


namespace NUMINAMATH_CALUDE_a_value_proof_l401_40145

theorem a_value_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (h1 : a ^ b = b ^ a) (h2 : b = 4 * a) : a = (4 : ℝ) ^ (1/3) := by
  sorry

end NUMINAMATH_CALUDE_a_value_proof_l401_40145


namespace NUMINAMATH_CALUDE_tangent_line_equation_l401_40129

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 4*x^2 + 4

-- Define the point of tangency
def point : ℝ × ℝ := (1, 1)

-- State the theorem
theorem tangent_line_equation :
  ∃ (m b : ℝ), 
    (∀ x, (deriv f) x = 3*x^2 - 8*x) ∧
    (deriv f) (point.1) = m ∧
    f point.1 = point.2 ∧
    (∀ x, m * (x - point.1) + point.2 = -5 * x + 6) := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l401_40129


namespace NUMINAMATH_CALUDE_solve_equation_l401_40169

theorem solve_equation (x : ℝ) (h : (0.12 / x) * 2 = 12) : x = 0.02 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l401_40169


namespace NUMINAMATH_CALUDE_sock_pair_combinations_l401_40140

def choose (n k : Nat) : Nat :=
  if k > n then 0
  else (List.range k).foldl (fun m i => m * (n - i) / (i + 1)) 1

theorem sock_pair_combinations : 
  let total_socks : Nat := 18
  let white_socks : Nat := 8
  let brown_socks : Nat := 6
  let blue_socks : Nat := 4
  choose white_socks 2 + choose brown_socks 2 + choose blue_socks 2 = 49 := by
  sorry

end NUMINAMATH_CALUDE_sock_pair_combinations_l401_40140


namespace NUMINAMATH_CALUDE_plan_a_fixed_charge_l401_40126

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

end NUMINAMATH_CALUDE_plan_a_fixed_charge_l401_40126


namespace NUMINAMATH_CALUDE_diamond_op_five_three_l401_40199

def diamond_op (x y : ℝ) : ℝ := 4 * x + 6 * y

theorem diamond_op_five_three : diamond_op 5 3 = 38 := by
  sorry

end NUMINAMATH_CALUDE_diamond_op_five_three_l401_40199


namespace NUMINAMATH_CALUDE_average_of_six_numbers_l401_40195

theorem average_of_six_numbers (a b c d e f : ℝ) 
  (h1 : (a + b) / 2 = 2.4)
  (h2 : (c + d) / 2 = 2.3)
  (h3 : (e + f) / 2 = 3.7) :
  (a + b + c + d + e + f) / 6 = 2.8 := by
  sorry

end NUMINAMATH_CALUDE_average_of_six_numbers_l401_40195


namespace NUMINAMATH_CALUDE_smallest_norm_l401_40117

open Real

/-- Given a vector v in ℝ² such that ‖v + (4, -2)‖ = 10, 
    the smallest possible value of ‖v‖ is 10 - 2√5 -/
theorem smallest_norm (v : ℝ × ℝ) 
  (h : ‖v + (4, -2)‖ = 10) : 
  ∃ (w : ℝ × ℝ), ‖w‖ = 10 - 2 * sqrt 5 ∧ ∀ u : ℝ × ℝ, ‖u + (4, -2)‖ = 10 → ‖w‖ ≤ ‖u‖ :=
sorry

end NUMINAMATH_CALUDE_smallest_norm_l401_40117


namespace NUMINAMATH_CALUDE_genevieve_coffee_consumption_l401_40108

-- Define the conversion factors
def ml_to_oz : ℝ := 0.0338
def l_to_oz : ℝ := 33.8

-- Define the thermos sizes
def small_thermos_ml : ℝ := 250
def medium_thermos_ml : ℝ := 400
def large_thermos_l : ℝ := 1

-- Calculate the amount of coffee in each thermos type in ounces
def small_thermos_oz : ℝ := small_thermos_ml * ml_to_oz
def medium_thermos_oz : ℝ := medium_thermos_ml * ml_to_oz
def large_thermos_oz : ℝ := large_thermos_l * l_to_oz

-- Define Genevieve's consumption
def genevieve_consumption : ℝ := small_thermos_oz + medium_thermos_oz + large_thermos_oz

-- Theorem statement
theorem genevieve_coffee_consumption :
  genevieve_consumption = 55.77 := by sorry

end NUMINAMATH_CALUDE_genevieve_coffee_consumption_l401_40108


namespace NUMINAMATH_CALUDE_unique_digit_divisibility_l401_40134

theorem unique_digit_divisibility : ∃! A : ℕ, A < 10 ∧ 41 % A = 0 ∧ (273100 + A * 10 + 8) % 8 = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_digit_divisibility_l401_40134


namespace NUMINAMATH_CALUDE_right_triangle_xy_length_l401_40171

theorem right_triangle_xy_length 
  (X Y Z : ℝ × ℝ) 
  (right_angle : (X.1 - Y.1) * (X.1 - Z.1) + (X.2 - Y.2) * (X.2 - Z.2) = 0) 
  (yz_length : Real.sqrt ((Y.1 - Z.1)^2 + (Y.2 - Z.2)^2) = 20)
  (tan_z_eq : (X.2 - Y.2) / (X.1 - Y.1) = 4 * (X.1 - Y.1) / 20) :
  Real.sqrt ((X.1 - Y.1)^2 + (X.2 - Y.2)^2) = 5 * Real.sqrt 15 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_xy_length_l401_40171


namespace NUMINAMATH_CALUDE_calculation_proof_l401_40127

theorem calculation_proof : 2359 + 180 / 60 * 3 - 359 = 2009 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l401_40127


namespace NUMINAMATH_CALUDE_problem_statement_l401_40104

theorem problem_statement : (12 : ℕ)^5 * 6^5 / 432^3 = 24 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l401_40104


namespace NUMINAMATH_CALUDE_equation_equality_l401_40173

theorem equation_equality (a b : ℝ) : -0.25 * a * b + (1/4) * a * b = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_equality_l401_40173


namespace NUMINAMATH_CALUDE_chicken_distribution_problem_l401_40132

/-- The multiple of Skylar's chickens that Quentin has 25 more than -/
def chicken_multiple (total colten skylar quentin : ℕ) : ℕ :=
  (quentin - 25) / skylar

/-- Proof of the chicken distribution problem -/
theorem chicken_distribution_problem (total colten skylar quentin : ℕ) 
  (h1 : total = 383)
  (h2 : colten = 37)
  (h3 : skylar = 3 * colten - 4)
  (h4 : quentin + skylar + colten = total)
  (h5 : ∃ m : ℕ, quentin = m * skylar + 25) :
  chicken_multiple total colten skylar quentin = 2 := by
  sorry

#eval chicken_multiple 383 37 107 239

end NUMINAMATH_CALUDE_chicken_distribution_problem_l401_40132


namespace NUMINAMATH_CALUDE_total_soccer_balls_l401_40184

/-- Represents a school with elementary and middle school classes --/
structure School where
  elementary_classes : ℕ
  middle_classes : ℕ
  elementary_students : List ℕ
  middle_students : List ℕ

/-- Calculates the number of soccer balls for a given number of students in an elementary class --/
def elementary_balls (students : ℕ) : ℕ :=
  if students ≤ 30 then 4 else 5

/-- Calculates the number of soccer balls for a given number of students in a middle school class --/
def middle_balls (students : ℕ) : ℕ :=
  if students ≤ 24 then 6 else 7

/-- Calculates the total number of soccer balls for a school --/
def school_balls (school : School) : ℕ :=
  (school.elementary_students.map elementary_balls).sum +
  (school.middle_students.map middle_balls).sum

/-- The three schools as described in the problem --/
def school_A : School :=
  { elementary_classes := 4
  , middle_classes := 5
  , elementary_students := List.replicate 4 28
  , middle_students := List.replicate 5 25 }

def school_B : School :=
  { elementary_classes := 5
  , middle_classes := 3
  , elementary_students := [32, 32, 32, 30, 30]
  , middle_students := [22, 22, 26] }

def school_C : School :=
  { elementary_classes := 6
  , middle_classes := 4
  , elementary_students := [30, 30, 30, 30, 31, 31]
  , middle_students := List.replicate 4 24 }

/-- The main theorem stating that the total number of soccer balls donated is 143 --/
theorem total_soccer_balls :
  school_balls school_A + school_balls school_B + school_balls school_C = 143 := by
  sorry


end NUMINAMATH_CALUDE_total_soccer_balls_l401_40184


namespace NUMINAMATH_CALUDE_greg_trip_distance_l401_40149

/-- Represents Greg's trip with given distances and speeds -/
structure GregTrip where
  workplace_to_market : ℝ
  market_to_friend : ℝ
  friend_to_aunt : ℝ
  aunt_to_grocery : ℝ
  grocery_to_home : ℝ

/-- Calculates the total distance of Greg's trip -/
def total_distance (trip : GregTrip) : ℝ :=
  trip.workplace_to_market + trip.market_to_friend + trip.friend_to_aunt + 
  trip.aunt_to_grocery + trip.grocery_to_home

/-- Theorem stating that Greg's total trip distance is 100 miles -/
theorem greg_trip_distance :
  ∃ (trip : GregTrip),
    trip.workplace_to_market = 30 ∧
    trip.market_to_friend = trip.workplace_to_market + 10 ∧
    trip.friend_to_aunt = 5 ∧
    trip.aunt_to_grocery = 7 ∧
    trip.grocery_to_home = 18 ∧
    total_distance trip = 100 :=
by
  sorry

end NUMINAMATH_CALUDE_greg_trip_distance_l401_40149


namespace NUMINAMATH_CALUDE_mean_of_cubic_solutions_l401_40128

theorem mean_of_cubic_solutions (x : ℝ) : 
  (x^3 + 3*x^2 - 44*x = 0) → 
  (∃ s : Finset ℝ, (∀ y ∈ s, y^3 + 3*y^2 - 44*y = 0) ∧ 
                   (s.card = 3) ∧ 
                   (s.sum id / s.card = -1)) :=
by sorry

end NUMINAMATH_CALUDE_mean_of_cubic_solutions_l401_40128


namespace NUMINAMATH_CALUDE_distance_to_third_side_l401_40105

/-- Represents a point inside an equilateral triangle -/
structure PointInTriangle where
  /-- Distance to the first side -/
  d1 : ℝ
  /-- Distance to the second side -/
  d2 : ℝ
  /-- Distance to the third side -/
  d3 : ℝ
  /-- The sum of distances equals the triangle's height -/
  sum_eq_height : d1 + d2 + d3 = 5 * Real.sqrt 3

/-- Theorem: In an equilateral triangle with side length 10, if a point inside
    has distances 1 and 3 to two sides, its distance to the third side is 5√3 - 4 -/
theorem distance_to_third_side
  (P : PointInTriangle)
  (h1 : P.d1 = 1)
  (h2 : P.d2 = 3) :
  P.d3 = 5 * Real.sqrt 3 - 4 := by
  sorry


end NUMINAMATH_CALUDE_distance_to_third_side_l401_40105


namespace NUMINAMATH_CALUDE_cone_height_ratio_l401_40197

theorem cone_height_ratio (base_circumference : ℝ) (original_height : ℝ) (shorter_volume : ℝ) :
  base_circumference = 20 * Real.pi →
  original_height = 24 →
  shorter_volume = 500 * Real.pi →
  ∃ (shorter_height : ℝ),
    shorter_volume = (1 / 3) * Real.pi * (base_circumference / (2 * Real.pi))^2 * shorter_height ∧
    shorter_height / original_height = 5 / 8 := by
  sorry

end NUMINAMATH_CALUDE_cone_height_ratio_l401_40197


namespace NUMINAMATH_CALUDE_function_characterization_l401_40166

/-- Euler's totient function -/
noncomputable def φ : ℕ+ → ℕ+ :=
  sorry

/-- The property that the function f satisfies -/
def satisfies_property (f : ℕ+ → ℕ+) : Prop :=
  ∀ (m n : ℕ+), m ≥ n → f (m * φ (n^3)) = f m * φ (n^3)

/-- The main theorem -/
theorem function_characterization :
  ∀ (f : ℕ+ → ℕ+), satisfies_property f →
  ∃ (b : ℕ+), ∀ (n : ℕ+), f n = b * n :=
sorry

end NUMINAMATH_CALUDE_function_characterization_l401_40166


namespace NUMINAMATH_CALUDE_profit_maximized_at_100_yuan_optimal_selling_price_l401_40143

/-- Profit function given price increase -/
def profit (x : ℝ) : ℝ := (10 + x) * (400 - 20 * x)

/-- The price increase that maximizes profit -/
def optimal_price_increase : ℝ := 10

theorem profit_maximized_at_100_yuan :
  ∀ x : ℝ, profit x ≤ profit optimal_price_increase :=
sorry

/-- The optimal selling price is 100 yuan -/
theorem optimal_selling_price :
  90 + optimal_price_increase = 100 :=
sorry

end NUMINAMATH_CALUDE_profit_maximized_at_100_yuan_optimal_selling_price_l401_40143

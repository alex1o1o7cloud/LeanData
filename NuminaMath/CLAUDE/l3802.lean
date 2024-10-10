import Mathlib

namespace x_power_six_plus_reciprocal_l3802_380246

theorem x_power_six_plus_reciprocal (x : ℝ) (h : x + 1/x = 3) : x^6 + 1/x^6 = 322 := by
  sorry

end x_power_six_plus_reciprocal_l3802_380246


namespace inequality_proof_l3802_380271

theorem inequality_proof (a b c d : ℝ) 
  (sum_condition : a + b + c + d = 6) 
  (sum_squares_condition : a^2 + b^2 + c^2 + d^2 = 12) : 
  36 ≤ 4 * (a^3 + b^3 + c^3 + d^3) - (a^4 + b^4 + c^4 + d^4) ∧ 
  4 * (a^3 + b^3 + c^3 + d^3) - (a^4 + b^4 + c^4 + d^4) ≤ 48 := by
  sorry

end inequality_proof_l3802_380271


namespace line_parallel_to_plane_theorem_l3802_380251

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the necessary relations
variable (subset : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (coplanar : Line → Line → Prop)
variable (parallel_lines : Line → Line → Prop)

-- State the theorem
theorem line_parallel_to_plane_theorem 
  (m n : Line) (α : Plane) 
  (h1 : subset m α) 
  (h2 : parallel n α) 
  (h3 : coplanar m n) : 
  parallel_lines m n :=
sorry

end line_parallel_to_plane_theorem_l3802_380251


namespace hyperbola_right_angle_area_l3802_380238

/-- The hyperbola with equation x²/9 - y²/16 = 1 -/
def hyperbola (x y : ℝ) : Prop := x^2 / 9 - y^2 / 16 = 1

/-- The left focus of the hyperbola -/
def F₁ : ℝ × ℝ := (-5, 0)

/-- The right focus of the hyperbola -/
def F₂ : ℝ × ℝ := (5, 0)

/-- A point on the hyperbola -/
def P : ℝ × ℝ := sorry

/-- The angle ∠F₁PF₂ is 90° -/
def right_angle : Prop :=
  let v₁ := (P.1 - F₁.1, P.2 - F₁.2)
  let v₂ := (P.1 - F₂.1, P.2 - F₂.2)
  v₁.1 * v₂.1 + v₁.2 * v₂.2 = 0

/-- The area of triangle ΔF₁PF₂ -/
def triangle_area : ℝ := sorry

theorem hyperbola_right_angle_area :
  hyperbola P.1 P.2 →
  right_angle →
  triangle_area = 16 := by sorry

end hyperbola_right_angle_area_l3802_380238


namespace num_beakers_is_three_l3802_380266

def volume_per_tube : ℚ := 7
def num_tubes : ℕ := 6
def volume_per_beaker : ℚ := 14

theorem num_beakers_is_three : 
  (volume_per_tube * num_tubes) / volume_per_beaker = 3 := by
  sorry

end num_beakers_is_three_l3802_380266


namespace no_finite_maximum_for_expression_l3802_380210

open Real

theorem no_finite_maximum_for_expression (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) 
  (sum_constraint : x + y + z = 9) : 
  ¬ ∃ (M : ℝ), ∀ (x y z : ℝ), 
    0 < x → 0 < y → 0 < z → x + y + z = 9 →
    (x^2 + 2*y^2)/(x + y) + (2*x^2 + z^2)/(x + z) + (y^2 + 2*z^2)/(y + z) ≤ M :=
sorry

end no_finite_maximum_for_expression_l3802_380210


namespace divisor_problem_l3802_380298

theorem divisor_problem (D : ℕ) (hD : D > 0) 
  (h1 : 242 % D = 6)
  (h2 : 698 % D = 13)
  (h3 : 940 % D = 5) : 
  D = 14 := by sorry

end divisor_problem_l3802_380298


namespace equation_solution_l3802_380286

theorem equation_solution : 
  ∃! x : ℚ, (x + 4) / (x - 3) = (x - 2) / (x + 2) ∧ x = -2/11 := by
  sorry

end equation_solution_l3802_380286


namespace smallest_angle_greater_than_36_degrees_l3802_380296

/-- Represents the angles of a convex pentagon in arithmetic progression -/
structure ConvexPentagonAngles where
  α : ℝ  -- smallest angle
  γ : ℝ  -- common difference
  convex : α > 0 ∧ γ ≥ 0 ∧ α + 4*γ < π  -- convexity condition
  sum : α + (α + γ) + (α + 2*γ) + (α + 3*γ) + (α + 4*γ) = 3*π  -- sum of angles

/-- 
Theorem: In a convex pentagon with angles in arithmetic progression, 
the smallest angle is greater than π/5 radians (36 degrees).
-/
theorem smallest_angle_greater_than_36_degrees (p : ConvexPentagonAngles) : 
  p.α > π/5 :=
sorry

end smallest_angle_greater_than_36_degrees_l3802_380296


namespace f_properties_l3802_380287

-- Define the function f(x) = -x|x| + 2x
def f (x : ℝ) : ℝ := -x * abs x + 2 * x

-- State the theorem
theorem f_properties :
  -- f is an odd function
  (∀ x, f (-x) = -f x) ∧
  -- f is monotonically decreasing on (-∞, -1)
  (∀ x y, x < y → y < -1 → f y < f x) ∧
  -- f is monotonically decreasing on (1, +∞)
  (∀ x y, 1 < x → x < y → f y < f x) :=
by sorry

end f_properties_l3802_380287


namespace sum_x_y_equals_one_third_l3802_380257

theorem sum_x_y_equals_one_third 
  (x y a : ℚ) 
  (eq1 : 17 * x + 19 * y = 6 - a) 
  (eq2 : 13 * x - 7 * y = 10 * a + 1) : 
  x + y = 1 / 3 := by
sorry

end sum_x_y_equals_one_third_l3802_380257


namespace remainder_71_73_mod_8_l3802_380216

theorem remainder_71_73_mod_8 : (71 * 73) % 8 = 7 := by
  sorry

end remainder_71_73_mod_8_l3802_380216


namespace line_slope_l3802_380236

/-- Given a line y = kx + 1 passing through points (4, b), (a, 5), and (a, b + 1),
    where a and b are real numbers, the value of k is 3/4. -/
theorem line_slope (k a b : ℝ) : 
  (b = 4 * k + 1) →
  (5 = a * k + 1) →
  (b + 1 = a * k + 1) →
  k = 3/4 := by sorry

end line_slope_l3802_380236


namespace halving_period_correct_l3802_380284

/-- The number of years it takes for the cost of a ticket to Mars to be halved. -/
def halving_period : ℕ := 10

/-- The initial cost of a ticket to Mars in dollars. -/
def initial_cost : ℕ := 1000000

/-- The cost of a ticket to Mars after 30 years in dollars. -/
def cost_after_30_years : ℕ := 125000

/-- The number of years passed. -/
def years_passed : ℕ := 30

/-- Theorem stating that the halving period is correct given the initial conditions. -/
theorem halving_period_correct : 
  initial_cost / (2 ^ (years_passed / halving_period)) = cost_after_30_years :=
sorry

end halving_period_correct_l3802_380284


namespace incorrect_expression_l3802_380221

theorem incorrect_expression (x y : ℚ) (h : x / y = 5 / 6) :
  ¬ ((x - y) / y = -1 / 6) := by
  sorry

end incorrect_expression_l3802_380221


namespace farm_heads_count_l3802_380244

theorem farm_heads_count (num_hens : ℕ) (total_feet : ℕ) : 
  num_hens = 30 → total_feet = 140 → 
  ∃ (num_cows : ℕ), 
    num_hens + num_cows = 50 ∧
    num_hens * 2 + num_cows * 4 = total_feet :=
by sorry

end farm_heads_count_l3802_380244


namespace volume_maximized_at_ten_l3802_380269

/-- The volume of the container as a function of the side length of the cut squares -/
def volume (x : ℝ) : ℝ := (90 - 2*x) * (48 - 2*x) * x

/-- The derivative of the volume function -/
def volume_derivative (x : ℝ) : ℝ := 12*x^2 - 552*x + 4320

theorem volume_maximized_at_ten :
  ∃ (x : ℝ), x > 0 ∧ x < 24 ∧
  (∀ (y : ℝ), y > 0 → y < 24 → volume y ≤ volume x) ∧
  x = 10 := by sorry

end volume_maximized_at_ten_l3802_380269


namespace odd_function_property_l3802_380214

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- State the theorem
theorem odd_function_property (f : ℝ → ℝ) (h1 : OddFunction f) (h2 : f (-3) = -2) :
  f 3 + f 0 = 2 := by
  sorry

end odd_function_property_l3802_380214


namespace square_sum_geq_linear_l3802_380288

theorem square_sum_geq_linear (a b : ℝ) : a^2 + b^2 ≥ 2*(a - b - 1) := by
  sorry

end square_sum_geq_linear_l3802_380288


namespace function_properties_l3802_380254

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period_one_negation (f : ℝ → ℝ) : Prop := ∀ x, f (x + 1) = -f x

theorem function_properties (f : ℝ → ℝ) 
  (h_odd : is_odd_function f) 
  (h_period_neg : has_period_one_negation f) : 
  (∀ x, f (x + 2) = f x) ∧ 
  (∀ (x : ℝ) (k : ℤ), f (2 * ↑k - x) = -f x) :=
sorry

end function_properties_l3802_380254


namespace complex_number_operation_l3802_380248

theorem complex_number_operation : 
  let z₁ : ℂ := -2 + 5*I
  let z₂ : ℂ := 3*I
  3 * z₁ + z₂ = -6 + 18*I :=
by sorry

end complex_number_operation_l3802_380248


namespace difference_max_min_both_languages_l3802_380247

/-- The number of students studying both Spanish and French -/
def students_both (S F : ℕ) : ℤ := S + F - 2001

theorem difference_max_min_both_languages :
  ∃ (S_min S_max F_min F_max : ℕ),
    1601 ≤ S_min ∧ S_max ≤ 1700 ∧
    601 ≤ F_min ∧ F_max ≤ 800 ∧
    (∀ S F, 1601 ≤ S ∧ S ≤ 1700 ∧ 601 ≤ F ∧ F ≤ 800 →
      students_both S_min F_min ≤ students_both S F ∧
      students_both S F ≤ students_both S_max F_max) ∧
    students_both S_max F_max - students_both S_min F_min = 298 := by
  sorry

end difference_max_min_both_languages_l3802_380247


namespace parallelogram_height_l3802_380225

theorem parallelogram_height (b h_t : ℝ) (h_t_pos : h_t > 0) :
  let a_t := b * h_t / 2
  let h_p := h_t / 2
  let a_p := b * h_p
  h_t = 10 → a_t = a_p → h_p = 5 := by sorry

end parallelogram_height_l3802_380225


namespace quadratic_root_property_l3802_380281

theorem quadratic_root_property (a : ℝ) : 
  2 * a^2 + 3 * a - 2022 = 0 → 2 - 6 * a - 4 * a^2 = -4042 := by
  sorry

end quadratic_root_property_l3802_380281


namespace gcd_product_is_square_l3802_380206

theorem gcd_product_is_square (x y z : ℕ+) 
  (h : (1 : ℚ) / x - (1 : ℚ) / y = (1 : ℚ) / z) :
  ∃ (k : ℕ), (Nat.gcd x.val (Nat.gcd y.val z.val)) * x.val * y.val * z.val = k ^ 2 := by
  sorry

end gcd_product_is_square_l3802_380206


namespace werewolf_identity_l3802_380241

structure Person where
  name : String
  is_knight : Bool
  is_werewolf : Bool
  is_liar : Bool

def A : Person := { name := "A", is_knight := true, is_werewolf := true, is_liar := false }
def B : Person := { name := "B", is_knight := false, is_werewolf := false, is_liar := true }
def C : Person := { name := "C", is_knight := false, is_werewolf := false, is_liar := true }

theorem werewolf_identity (A B C : Person) :
  (A.is_knight ↔ (A.is_liar ∨ B.is_liar ∨ C.is_liar)) →
  (B.is_knight ↔ C.is_knight) →
  ((A.is_werewolf ∧ A.is_knight) ∨ (B.is_werewolf ∧ B.is_knight)) →
  (A.is_werewolf ∨ B.is_werewolf) →
  ¬(A.is_werewolf ∧ B.is_werewolf) →
  A.is_werewolf := by
  sorry

#check werewolf_identity A B C

end werewolf_identity_l3802_380241


namespace area_of_r3_l3802_380276

/-- Given a square R1 with area 36, R2 formed by connecting midpoints of R1's sides,
    and R3 formed by moving R2's corners halfway to its center, prove R3's area is 4.5 -/
theorem area_of_r3 (r1 r2 r3 : Real) : 
  r1^2 = 36 → 
  r2 = r1 * Real.sqrt 2 / 2 → 
  r3 = r2 / 2 → 
  r3^2 = 4.5 := by sorry

end area_of_r3_l3802_380276


namespace geometric_sequence_problem_l3802_380256

/-- Given a geometric sequence with positive terms and common ratio 2, 
    if the product of the 3rd and 11th terms is 16, then the 10th term is 32. -/
theorem geometric_sequence_problem (a : ℕ → ℝ) : 
  (∀ n, a n > 0) →
  (∀ n, a (n + 1) = 2 * a n) →
  a 3 * a 11 = 16 →
  a 10 = 32 := by
  sorry

end geometric_sequence_problem_l3802_380256


namespace find_k_l3802_380291

theorem find_k (d m n : ℝ) : 
  (∃ k : ℝ, ∀ x : ℝ, (3*x^2 - 4*x + 2)*(d*x^3 + k*x^2 + m*x + n) = 6*x^5 - 11*x^4 + 14*x^3 - 4*x^2 + 8*x - 3) → 
  (∃ k : ℝ, k = -11/3 ∧ ∀ x : ℝ, (3*x^2 - 4*x + 2)*(d*x^3 + k*x^2 + m*x + n) = 6*x^5 - 11*x^4 + 14*x^3 - 4*x^2 + 8*x - 3) :=
by sorry

end find_k_l3802_380291


namespace tan_addition_formula_l3802_380227

theorem tan_addition_formula (x : Real) (h : Real.tan x = 3) :
  Real.tan (x + π / 6) = 5 + 2 * Real.sqrt 3 := by
  sorry

end tan_addition_formula_l3802_380227


namespace sunzi_remainder_problem_l3802_380212

theorem sunzi_remainder_problem :
  let S : Set ℕ := {n : ℕ | 1 ≤ n ∧ n ≤ 200 ∧ n % 3 = 2 ∧ n % 5 = 3}
  (∃ (min max : ℕ), min ∈ S ∧ max ∈ S ∧
    (∀ x ∈ S, min ≤ x) ∧
    (∀ x ∈ S, x ≤ max) ∧
    min + max = 196) :=
by sorry

end sunzi_remainder_problem_l3802_380212


namespace equation_solution_l3802_380213

theorem equation_solution : 
  ∃ x : ℚ, (7 * x + 3) / (x + 5) - 5 / (x + 5) = 2 / (x + 5) ∧ x = 4 / 7 := by
  sorry

end equation_solution_l3802_380213


namespace five_dice_not_same_probability_l3802_380261

theorem five_dice_not_same_probability :
  let n := 8  -- number of sides on each die
  let k := 5  -- number of dice rolled
  (1 - (n : ℚ) / n^k) = 4095 / 4096 :=
by sorry

end five_dice_not_same_probability_l3802_380261


namespace total_amount_is_2150_6_l3802_380218

/-- Calculates the total amount paid for fruits with discounts and taxes -/
def total_amount_paid (
  grapes_kg : ℝ)   (grapes_price : ℝ)
  (mangoes_kg : ℝ)  (mangoes_price : ℝ)
  (apples_kg : ℝ)   (apples_price : ℝ)
  (strawberries_kg : ℝ) (strawberries_price : ℝ)
  (oranges_kg : ℝ)  (oranges_price : ℝ)
  (kiwis_kg : ℝ)    (kiwis_price : ℝ)
  (grapes_apples_discount : ℝ)
  (oranges_kiwis_discount : ℝ)
  (mangoes_strawberries_tax : ℝ) : ℝ :=
  let grapes_total := grapes_kg * grapes_price
  let mangoes_total := mangoes_kg * mangoes_price
  let apples_total := apples_kg * apples_price
  let strawberries_total := strawberries_kg * strawberries_price
  let oranges_total := oranges_kg * oranges_price
  let kiwis_total := kiwis_kg * kiwis_price
  
  let total_before_discounts_taxes := grapes_total + mangoes_total + apples_total + 
                                      strawberries_total + oranges_total + kiwis_total
  
  let grapes_apples_discount_amount := (grapes_total + apples_total) * grapes_apples_discount
  let oranges_kiwis_discount_amount := (oranges_total + kiwis_total) * oranges_kiwis_discount
  let mangoes_strawberries_tax_amount := (mangoes_total + strawberries_total) * mangoes_strawberries_tax
  
  total_before_discounts_taxes - grapes_apples_discount_amount - oranges_kiwis_discount_amount + mangoes_strawberries_tax_amount

/-- Theorem stating that the total amount paid for fruits is 2150.6 -/
theorem total_amount_is_2150_6 :
  total_amount_paid 8 70 9 45 5 30 3 100 10 40 6 60 0.1 0.05 0.12 = 2150.6 := by
  sorry

end total_amount_is_2150_6_l3802_380218


namespace sandy_final_position_l3802_380226

-- Define the coordinate system
def Position := ℝ × ℝ

-- Define the starting position
def start : Position := (0, 0)

-- Define Sandy's movements
def move_south (p : Position) (distance : ℝ) : Position :=
  (p.1, p.2 - distance)

def move_east (p : Position) (distance : ℝ) : Position :=
  (p.1 + distance, p.2)

def move_north (p : Position) (distance : ℝ) : Position :=
  (p.1, p.2 + distance)

-- Define Sandy's final position after her movements
def final_position : Position :=
  move_east (move_north (move_east (move_south start 20) 20) 20) 20

-- Theorem to prove
theorem sandy_final_position :
  final_position = (40, 0) := by sorry

end sandy_final_position_l3802_380226


namespace solution_satisfies_congruences_solution_is_three_digit_solution_is_smallest_l3802_380274

/-- The smallest three-digit number satisfying the given congruences -/
def smallest_solution : ℕ := 230

/-- First congruence condition -/
def cong1 (x : ℕ) : Prop := 5 * x ≡ 15 [ZMOD 10]

/-- Second congruence condition -/
def cong2 (x : ℕ) : Prop := 3 * x + 4 ≡ 7 [ZMOD 8]

/-- Third congruence condition -/
def cong3 (x : ℕ) : Prop := -3 * x + 2 ≡ x [ZMOD 17]

/-- The solution satisfies all congruences -/
theorem solution_satisfies_congruences :
  cong1 smallest_solution ∧ 
  cong2 smallest_solution ∧ 
  cong3 smallest_solution :=
sorry

/-- The solution is a three-digit number -/
theorem solution_is_three_digit :
  100 ≤ smallest_solution ∧ smallest_solution < 1000 :=
sorry

/-- The solution is the smallest such number -/
theorem solution_is_smallest (n : ℕ) :
  (100 ≤ n ∧ n < smallest_solution) →
  ¬(cong1 n ∧ cong2 n ∧ cong3 n) :=
sorry

end solution_satisfies_congruences_solution_is_three_digit_solution_is_smallest_l3802_380274


namespace rectangle_area_l3802_380280

/-- Theorem: For a rectangle EFGH with vertices E(0, 0), F(0, y), G(y, 3y), and H(y, 0),
    where y > 0 and the area of the rectangle is 45 square units, the value of y is √15. -/
theorem rectangle_area (y : ℝ) (h1 : y > 0) (h2 : y * 3 * y = 45) : y = Real.sqrt 15 := by
  sorry

end rectangle_area_l3802_380280


namespace circular_table_theorem_l3802_380255

/-- Represents the setup of people sitting around a circular table -/
structure CircularTable where
  num_men : ℕ
  num_women : ℕ

/-- Defines when a man is considered satisfied -/
def is_satisfied (t : CircularTable) : Prop :=
  ∃ (i j : ℕ), i ≠ j ∧ i < t.num_men + t.num_women ∧ j < t.num_men + t.num_women

/-- The probability of a specific man being satisfied -/
def prob_man_satisfied (t : CircularTable) : ℚ :=
  25 / 33

/-- The expected number of satisfied men -/
def expected_satisfied_men (t : CircularTable) : ℚ :=
  1250 / 33

/-- Main theorem statement -/
theorem circular_table_theorem (t : CircularTable) 
  (h1 : t.num_men = 50) (h2 : t.num_women = 50) : 
  prob_man_satisfied t = 25 / 33 ∧ 
  expected_satisfied_men t = 1250 / 33 := by
  sorry

#check circular_table_theorem

end circular_table_theorem_l3802_380255


namespace rectangle_area_problem_l3802_380217

theorem rectangle_area_problem :
  ∃ (length width : ℕ+), 
    (length : ℝ) * (width : ℝ) = ((length : ℝ) + 3) * ((width : ℝ) - 1) ∧
    (length : ℝ) * (width : ℝ) = ((length : ℝ) - 3) * ((width : ℝ) + 2) ∧
    (length : ℝ) * (width : ℝ) = 15 := by
  sorry

end rectangle_area_problem_l3802_380217


namespace rectangle_diagonal_length_l3802_380264

-- Define the rectangle
structure Rectangle where
  length : ℝ
  width : ℝ
  area : ℝ
  diagonal : ℝ

-- State the theorem
theorem rectangle_diagonal_length 
  (rect : Rectangle) 
  (h1 : rect.length = 16) 
  (h2 : rect.area = 192) 
  (h3 : rect.area = rect.length * rect.width) 
  (h4 : rect.diagonal ^ 2 = rect.length ^ 2 + rect.width ^ 2) : 
  rect.diagonal = 20 := by
  sorry


end rectangle_diagonal_length_l3802_380264


namespace smallest_x_cosine_equality_l3802_380268

theorem smallest_x_cosine_equality : ∃ x : ℝ, 
  (x > 0 ∧ x < 30) ∧ 
  (∀ y : ℝ, y > 0 ∧ y < 30 ∧ Real.cos (3 * y * π / 180) = Real.cos ((2 * y^2 - y) * π / 180) → x ≤ y) ∧
  Real.cos (3 * x * π / 180) = Real.cos ((2 * x^2 - x) * π / 180) ∧
  x = 1 := by
sorry

end smallest_x_cosine_equality_l3802_380268


namespace min_value_complex_expression_l3802_380211

theorem min_value_complex_expression (z : ℂ) (h : Complex.abs (z - 3 + 3 * Complex.I) = 3) :
  ∃ (min : ℝ), min = 59 ∧ ∀ (w : ℂ), Complex.abs (w - 3 + 3 * Complex.I) = 3 →
    Complex.abs (w - 2 - Complex.I) ^ 2 + Complex.abs (w - 6 + 2 * Complex.I) ^ 2 ≥ min :=
by
  sorry

end min_value_complex_expression_l3802_380211


namespace min_value_theorem_l3802_380260

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1 / a + 4 / b) ≥ 9 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + b₀ = 1 ∧ 1 / a₀ + 4 / b₀ = 9 := by
  sorry

end min_value_theorem_l3802_380260


namespace optimal_parking_allocation_l3802_380243

/-- Represents the parking space allocation problem -/
structure ParkingAllocation where
  total_spaces : ℕ
  above_ground_cost : ℚ
  underground_cost : ℚ
  total_budget : ℚ
  above_ground_spaces : ℕ
  underground_spaces : ℕ

/-- The optimal allocation satisfies the given conditions -/
def is_optimal_allocation (p : ParkingAllocation) : Prop :=
  p.total_spaces = 5000 ∧
  3 * p.above_ground_cost + 2 * p.underground_cost = 0.8 ∧
  2 * p.above_ground_cost + 4 * p.underground_cost = 1.2 ∧
  p.above_ground_cost = 0.1 ∧
  p.underground_cost = 0.25 ∧
  p.total_budget = 950 ∧
  p.above_ground_spaces + p.underground_spaces = p.total_spaces ∧
  p.above_ground_spaces * p.above_ground_cost + p.underground_spaces * p.underground_cost ≤ p.total_budget

/-- The theorem stating the optimal allocation -/
theorem optimal_parking_allocation :
  ∃ (p : ParkingAllocation), is_optimal_allocation p ∧ 
    p.above_ground_spaces = 2000 ∧ p.underground_spaces = 3000 := by
  sorry


end optimal_parking_allocation_l3802_380243


namespace walking_problem_l3802_380275

/-- Proves that given the conditions of the walking problem, the speed of the second man is 3 km/h -/
theorem walking_problem (distance : ℝ) (speed_first : ℝ) (time_diff : ℝ) :
  distance = 6 →
  speed_first = 4 →
  time_diff = 0.5 →
  let time_first := distance / speed_first
  let time_second := time_first + time_diff
  let speed_second := distance / time_second
  speed_second = 3 := by
  sorry

end walking_problem_l3802_380275


namespace digit_difference_l3802_380222

theorem digit_difference (e : ℕ) (X Y : ℕ) : 
  e > 8 →
  X < e →
  Y < e →
  (e * X + Y) + (e * X + X) = 2 * e^2 + 4 * e + 3 →
  X - Y = (2 * e^2 + 4 * e - 726) / 3 := by
  sorry

end digit_difference_l3802_380222


namespace circle_radius_zero_l3802_380297

/-- The radius of the circle described by the equation 4x^2 - 8x + 4y^2 - 16y + 20 = 0 is 0 -/
theorem circle_radius_zero (x y : ℝ) : 
  (4*x^2 - 8*x + 4*y^2 - 16*y + 20 = 0) → 
  ∃ (h k : ℝ), ∀ (x y : ℝ), 4*x^2 - 8*x + 4*y^2 - 16*y + 20 = 0 ↔ (x - h)^2 + (y - k)^2 = 0 :=
by sorry

end circle_radius_zero_l3802_380297


namespace rational_division_and_linear_combination_l3802_380204

theorem rational_division_and_linear_combination (m a b c d k : ℤ) : 
  (∀ (x : ℤ), (x ∣ 5*m + 6 ∧ x ∣ 8*m + 7) ↔ (x = 1 ∨ x = -1 ∨ x = 13 ∨ x = -13)) ∧
  ((k ∣ a*m + b ∧ k ∣ c*m + d) → k ∣ a*d - b*c) := by
  sorry

end rational_division_and_linear_combination_l3802_380204


namespace percentage_problem_l3802_380223

theorem percentage_problem (N : ℝ) (P : ℝ) 
  (h1 : P / 100 * N = 160)
  (h2 : 60 / 100 * N = 240) : 
  P = 40 := by
sorry

end percentage_problem_l3802_380223


namespace resistance_of_single_rod_l3802_380200

/-- The resistance of the entire construction between points A and B -/
def R : ℝ := 8

/-- The number of identical metallic rods in the network -/
def num_rods : ℕ := 13

/-- The resistance of one rod -/
def R₀ : ℝ := 20

/-- The relation between the total resistance and the resistance of one rod -/
axiom resistance_relation : R = (4/10) * R₀

theorem resistance_of_single_rod : R₀ = 20 :=
  sorry

end resistance_of_single_rod_l3802_380200


namespace complement_A_intersect_B_l3802_380293

open Set

-- Define the universal set U as ℝ
def U : Set ℝ := univ

-- Define set A
def A : Set ℝ := {x | x > 3 ∨ x < -1}

-- Define set B
def B : Set ℝ := {x | x > 0}

-- State the theorem
theorem complement_A_intersect_B :
  (U \ A) ∩ B = {x : ℝ | 0 < x ∧ x ≤ 3} := by sorry

end complement_A_intersect_B_l3802_380293


namespace constant_term_in_system_l3802_380235

theorem constant_term_in_system (x y C : ℝ) : 
  (5 * x + y = 19) →
  (x + 3 * y = C) →
  (3 * x + 2 * y = 10) →
  C = 1 := by
sorry

end constant_term_in_system_l3802_380235


namespace sum_of_odd_divisors_180_l3802_380282

def sum_of_odd_divisors (n : ℕ) : ℕ := sorry

theorem sum_of_odd_divisors_180 : sum_of_odd_divisors 180 = 78 := by sorry

end sum_of_odd_divisors_180_l3802_380282


namespace equation_solution_l3802_380203

theorem equation_solution : ∃! x : ℝ, (81 : ℝ) ^ (x - 1) / (9 : ℝ) ^ (x - 1) = (729 : ℝ) ^ x ∧ x = -1/2 := by
  sorry

end equation_solution_l3802_380203


namespace winning_strategy_correct_l3802_380253

/-- A stone-picking game with two players. -/
structure StoneGame where
  total_stones : ℕ
  min_take : ℕ
  max_take : ℕ

/-- A winning strategy for the first player in the stone game. -/
def winning_first_move (game : StoneGame) : ℕ := 3

/-- Theorem stating that the winning strategy for the first player is correct. -/
theorem winning_strategy_correct (game : StoneGame) 
  (h1 : game.total_stones = 18)
  (h2 : game.min_take = 1)
  (h3 : game.max_take = 4) :
  ∃ (n : ℕ), n ≥ game.min_take ∧ n ≤ game.max_take ∧
  (winning_first_move game = n → 
   ∀ (m : ℕ), m ≥ game.min_take → m ≤ game.max_take → 
   ∃ (k : ℕ), k ≥ game.min_take ∧ k ≤ game.max_take ∧
   (game.total_stones - n - m - k) % 5 = 0) :=
sorry

end winning_strategy_correct_l3802_380253


namespace fifth_term_value_l3802_380294

def sequence_term (n : ℕ) : ℕ :=
  Finset.sum (Finset.range n) (fun i => 2^i + 5)

theorem fifth_term_value : sequence_term 5 = 56 := by
  sorry

end fifth_term_value_l3802_380294


namespace intersection_distance_l3802_380233

-- Define the lines and point A
def l₁ (t : ℝ) : ℝ × ℝ := (1 + 3*t, 2 - 4*t)
def l₂ (x y : ℝ) : Prop := 2*x - 4*y = 5
def A : ℝ × ℝ := (1, 2)

-- State the theorem
theorem intersection_distance :
  ∃ (t : ℝ), 
    let B := l₁ t
    l₂ B.1 B.2 ∧ 
    Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 5/2 := by
  sorry

end intersection_distance_l3802_380233


namespace smallest_special_number_l3802_380299

theorem smallest_special_number : ∃ (n : ℕ), 
  (100 ≤ n ∧ n ≤ 999) ∧ 
  (∃ (k : ℕ), n = 2 * k) ∧
  (∃ (k : ℕ), n + 1 = 3 * k) ∧
  (∃ (k : ℕ), n + 2 = 4 * k) ∧
  (∃ (k : ℕ), n + 3 = 5 * k) ∧
  (∃ (k : ℕ), n + 4 = 6 * k) ∧
  (∀ (m : ℕ), 
    (100 ≤ m ∧ m < n) →
    (¬(∃ (k : ℕ), m = 2 * k) ∨
     ¬(∃ (k : ℕ), m + 1 = 3 * k) ∨
     ¬(∃ (k : ℕ), m + 2 = 4 * k) ∨
     ¬(∃ (k : ℕ), m + 3 = 5 * k) ∨
     ¬(∃ (k : ℕ), m + 4 = 6 * k))) ∧
  n = 122 :=
by sorry

end smallest_special_number_l3802_380299


namespace circle_equation_l3802_380252

-- Define the circle C
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the line L: x - y - 1 = 0
def Line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 - p.2 - 1 = 0}

-- Define points A and B
def A : ℝ × ℝ := (4, 1)
def B : ℝ × ℝ := (2, 1)

-- State the theorem
theorem circle_equation :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    -- The circle passes through point A
    A ∈ Circle center radius ∧
    -- The circle is tangent to the line at point B
    B ∈ Circle center radius ∧
    B ∈ Line ∧
    -- The equation of the circle is (x-3)²+y²=2
    center = (3, 0) ∧ radius^2 = 2 := by
  sorry

end circle_equation_l3802_380252


namespace monomial_coefficient_and_degree_l3802_380290

/-- The monomial type -/
structure Monomial (α : Type*) [Ring α] where
  coeff : α
  vars : List (α × Nat)

/-- Definition of the coefficient of a monomial -/
def coefficient (m : Monomial ℚ) : ℚ := m.coeff

/-- Definition of the degree of a monomial -/
def degree (m : Monomial ℚ) : Nat := m.vars.foldr (λ (_, exp) acc => acc + exp) 0

/-- The given monomial -/
def m : Monomial ℚ := ⟨-3/5, [(1, 1), (1, 2)]⟩

theorem monomial_coefficient_and_degree :
  coefficient m = -3/5 ∧ degree m = 3 := by sorry

end monomial_coefficient_and_degree_l3802_380290


namespace x_is_28_percent_greater_than_150_l3802_380277

theorem x_is_28_percent_greater_than_150 :
  ∀ x : ℝ, x = 150 * (1 + 28/100) → x = 192 := by
  sorry

end x_is_28_percent_greater_than_150_l3802_380277


namespace college_choice_probability_l3802_380292

theorem college_choice_probability : 
  let num_examinees : ℕ := 2
  let num_colleges : ℕ := 3
  let prob_choose_college : ℚ := 1 / num_colleges
  
  -- Probability that both examinees choose the third college
  let prob_both_choose_third : ℚ := prob_choose_college ^ num_examinees
  
  -- Probability that at least one of the first two colleges is chosen
  let prob_at_least_one_first_two : ℚ := 1 - prob_both_choose_third
  
  prob_at_least_one_first_two = 8 / 9 :=
by sorry

end college_choice_probability_l3802_380292


namespace cos_36_cos_24_minus_sin_36_sin_24_l3802_380207

theorem cos_36_cos_24_minus_sin_36_sin_24 :
  Real.cos (36 * π / 180) * Real.cos (24 * π / 180) -
  Real.sin (36 * π / 180) * Real.sin (24 * π / 180) = 1 / 2 := by
  sorry

end cos_36_cos_24_minus_sin_36_sin_24_l3802_380207


namespace eraser_distribution_l3802_380289

theorem eraser_distribution (total_erasers : ℕ) (num_friends : ℕ) (erasers_per_friend : ℕ) :
  total_erasers = 3840 →
  num_friends = 48 →
  erasers_per_friend = total_erasers / num_friends →
  erasers_per_friend = 80 :=
by sorry

end eraser_distribution_l3802_380289


namespace min_value_x_plus_2y_min_value_is_18_min_value_exists_l3802_380202

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 8/x + 1/y = 1) :
  ∀ a b : ℝ, a > 0 → b > 0 → 8/a + 1/b = 1 → x + 2*y ≤ a + 2*b :=
by sorry

theorem min_value_is_18 (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 8/x + 1/y = 1) :
  x + 2*y ≥ 18 :=
by sorry

theorem min_value_exists :
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 8/x + 1/y = 1 ∧ x + 2*y = 18 :=
by sorry

end min_value_x_plus_2y_min_value_is_18_min_value_exists_l3802_380202


namespace negative_division_equals_positive_division_of_negative_integers_l3802_380232

theorem negative_division_equals_positive (a b : Int) (h : b ≠ 0) :
  (-a) / (-b) = a / b :=
sorry

theorem division_of_negative_integers :
  (-81) / (-9) = 9 :=
sorry

end negative_division_equals_positive_division_of_negative_integers_l3802_380232


namespace k_set_characterization_l3802_380228

/-- Given h = 2^r for some non-negative integer r, k(h) is the set of all natural numbers k
    such that there exist an odd natural number m > 1 and a natural number n where
    k divides m^k - 1 and m divides n^((n^k - 1)/k) + 1 -/
def k_set (h : ℕ) : Set ℕ :=
  {k : ℕ | ∃ (m n : ℕ), m > 1 ∧ m % 2 = 1 ∧ 
    (m^k - 1) % k = 0 ∧ (n^((n^k - 1)/k) + 1) % m = 0}

/-- For h = 2^r, the set k(h) is equal to {2^(r+s) * t | s, t ∈ ℕ, t is odd} -/
theorem k_set_characterization (r : ℕ) :
  k_set (2^r) = {k : ℕ | ∃ (s t : ℕ), k = 2^(r+s) * t ∧ t % 2 = 1} :=
by sorry

end k_set_characterization_l3802_380228


namespace root_implies_m_minus_n_l3802_380230

theorem root_implies_m_minus_n (m n : ℝ) : 
  ((-3)^2 + m*(-3) + 3*n = 0) → (m - n = 3) := by
  sorry

end root_implies_m_minus_n_l3802_380230


namespace equality_of_expressions_l3802_380229

theorem equality_of_expressions : 
  (-2^2 ≠ (-2)^2) ∧ 
  (2^3 ≠ 3^2) ∧ 
  (-3^3 = (-3)^3) ∧ 
  (-|-3| ≠ -(-3)) := by
  sorry

end equality_of_expressions_l3802_380229


namespace speech_contest_probabilities_l3802_380283

def num_boys : ℕ := 4
def num_girls : ℕ := 2
def num_selected : ℕ := 2

def total_combinations : ℕ := (num_boys + num_girls).choose num_selected

theorem speech_contest_probabilities :
  let p_two_boys := (num_boys.choose num_selected : ℚ) / total_combinations
  let p_one_girl := (num_boys.choose 1 * num_girls.choose 1 : ℚ) / total_combinations
  let p_at_least_one_girl := 1 - p_two_boys
  (p_two_boys = 2/5) ∧
  (p_one_girl = 8/15) ∧
  (p_at_least_one_girl = 3/5) := by sorry

end speech_contest_probabilities_l3802_380283


namespace sqrt_meaningful_l3802_380220

theorem sqrt_meaningful (x : ℝ) : (∃ y : ℝ, y ^ 2 = x + 2) ↔ x ≥ -2 := by sorry

end sqrt_meaningful_l3802_380220


namespace min_sum_squares_y_coords_l3802_380219

/-- 
Given a line passing through (4, 0) and intersecting the parabola y^2 = 4x at two points,
the minimum value of the sum of the squares of the y-coordinates of these two points is 32.
-/
theorem min_sum_squares_y_coords : 
  ∀ (m : ℝ) (y₁ y₂ : ℝ),
  y₁^2 = 4 * (m * y₁ + 4) →
  y₂^2 = 4 * (m * y₂ + 4) →
  y₁ ≠ y₂ →
  ∀ (z₁ z₂ : ℝ),
  z₁^2 = 4 * (m * z₁ + 4) →
  z₂^2 = 4 * (m * z₂ + 4) →
  z₁ ≠ z₂ →
  y₁^2 + y₂^2 ≤ z₁^2 + z₂^2 →
  y₁^2 + y₂^2 = 32 :=
by sorry


end min_sum_squares_y_coords_l3802_380219


namespace det_2x2_matrix_l3802_380265

/-- The determinant of a 2x2 matrix [[x, 4], [-3, y]] is xy + 12 -/
theorem det_2x2_matrix (x y : ℝ) : 
  Matrix.det !![x, 4; -3, y] = x * y + 12 := by
  sorry

end det_2x2_matrix_l3802_380265


namespace solution1_composition_l3802_380224

/-- Represents a solution with lemonade and carbonated water -/
structure Solution :=
  (lemonade : ℝ)
  (carbonated_water : ℝ)

/-- Represents a mixture of two solutions -/
structure Mixture :=
  (solution1 : Solution)
  (solution2 : Solution)
  (proportion1 : ℝ)

/-- Theorem stating the composition of Solution 1 given the mixture properties -/
theorem solution1_composition 
  (s1 : Solution)
  (s2 : Solution)
  (m : Mixture)
  (h1 : s1.lemonade = 20)
  (h2 : s2.lemonade = 45)
  (h3 : s2.carbonated_water = 55)
  (h4 : m.solution1 = s1)
  (h5 : m.solution2 = s2)
  (h6 : m.proportion1 = 20)
  (h7 : m.proportion1 * s1.carbonated_water + (100 - m.proportion1) * s2.carbonated_water = 60 * 100) :
  s1.carbonated_water = 80 := by
  sorry

end solution1_composition_l3802_380224


namespace lisa_caffeine_limit_l3802_380258

/-- The amount of caffeine (in mg) in one cup of coffee -/
def caffeine_per_cup : ℕ := 80

/-- The number of cups of coffee Lisa drinks -/
def cups_of_coffee : ℕ := 3

/-- The amount (in mg) by which Lisa exceeds her daily limit when drinking three cups -/
def excess_caffeine : ℕ := 40

/-- Lisa's daily caffeine limit (in mg) -/
def lisas_caffeine_limit : ℕ := 200

theorem lisa_caffeine_limit :
  lisas_caffeine_limit = cups_of_coffee * caffeine_per_cup - excess_caffeine :=
by sorry

end lisa_caffeine_limit_l3802_380258


namespace weight_after_deliveries_l3802_380245

/-- Calculates the remaining weight on a truck after two deliveries with given percentages -/
def remaining_weight (initial_load : ℝ) (first_unload_percent : ℝ) (second_unload_percent : ℝ) : ℝ :=
  let remaining_after_first := initial_load * (1 - first_unload_percent)
  remaining_after_first * (1 - second_unload_percent)

/-- Theorem stating the remaining weight after two deliveries -/
theorem weight_after_deliveries :
  remaining_weight 50000 0.1 0.2 = 36000 := by
  sorry

end weight_after_deliveries_l3802_380245


namespace circle_symmetry_line_l3802_380234

-- Define the circles C₁ and C₂
def C₁ (x y a : ℝ) : Prop := x^2 + y^2 - a = 0
def C₂ (x y a : ℝ) : Prop := x^2 + y^2 + 2*x - 2*a*y + 3 = 0

-- Define the line l
def line_l (x y : ℝ) : Prop := 2*x - 4*y + 5 = 0

-- Define symmetry between circles with respect to a line
def symmetric_circles (C₁ C₂ : (ℝ → ℝ → ℝ → Prop)) (l : (ℝ → ℝ → Prop)) : Prop :=
  ∃ a : ℝ, ∀ x y : ℝ, C₁ x y a ∧ C₂ x y a → l x y

-- Theorem statement
theorem circle_symmetry_line :
  symmetric_circles C₁ C₂ line_l :=
sorry

end circle_symmetry_line_l3802_380234


namespace hyperbolas_same_asymptotes_l3802_380231

/-- Given two hyperbolas with equations x²/9 - y²/16 = 1 and y²/25 - x²/N = 1,
    if they have the same asymptotes, then N = 225/16 -/
theorem hyperbolas_same_asymptotes (N : ℝ) :
  (∀ x y : ℝ, x^2 / 9 - y^2 / 16 = 1 ↔ y^2 / 25 - x^2 / N = 1) →
  N = 225 / 16 := by
  sorry

end hyperbolas_same_asymptotes_l3802_380231


namespace maya_total_pages_l3802_380285

def maya_reading_problem (books_last_week : ℕ) (pages_per_book : ℕ) (reading_increase : ℕ) : ℕ :=
  let pages_last_week := books_last_week * pages_per_book
  let pages_this_week := reading_increase * pages_last_week
  pages_last_week + pages_this_week

theorem maya_total_pages :
  maya_reading_problem 5 300 2 = 4500 :=
by
  sorry

end maya_total_pages_l3802_380285


namespace equal_sum_number_properties_l3802_380240

def is_equal_sum_number (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n < 10000 ∧
  (n / 1000 + (n / 10) % 10 = (n / 100) % 10 + n % 10)

def transform (n : ℕ) : ℕ :=
  let a := n / 1000
  let b := (n / 100) % 10
  let c := (n / 10) % 10
  let d := n % 10
  d * 1000 + c * 100 + b * 10 + a

def F (n : ℕ) : ℚ := (n + transform n : ℚ) / 101

def G (n : ℕ) : ℚ := (n - transform n : ℚ) / 99

theorem equal_sum_number_properties :
  (∀ n : ℕ, is_equal_sum_number n → 
    (F n - G n = 72 → n = 5236)) ∧
  (∃ n : ℕ, is_equal_sum_number n ∧ 
    (F n / 13).isInt ∧ (G n / 7).isInt ∧
    (∀ m : ℕ, is_equal_sum_number m ∧ 
      (F m / 13).isInt ∧ (G m / 7).isInt → m ≤ n) ∧
    n = 9647) := by
  sorry

end equal_sum_number_properties_l3802_380240


namespace coat_cost_proof_l3802_380242

/-- The cost of the more expensive coat -/
def expensive_coat_cost : ℝ := 300

/-- The lifespan of the more expensive coat in years -/
def expensive_coat_lifespan : ℕ := 15

/-- The cost of the cheaper coat -/
def cheaper_coat_cost : ℝ := 120

/-- The lifespan of the cheaper coat in years -/
def cheaper_coat_lifespan : ℕ := 5

/-- The number of years over which we compare the costs -/
def comparison_period : ℕ := 30

/-- The amount saved by buying the more expensive coat over the comparison period -/
def savings : ℝ := 120

theorem coat_cost_proof :
  expensive_coat_cost * (comparison_period / expensive_coat_lifespan) =
  cheaper_coat_cost * (comparison_period / cheaper_coat_lifespan) - savings :=
by sorry

end coat_cost_proof_l3802_380242


namespace three_zeros_condition_l3802_380278

/-- A cubic function with a parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x + 2

/-- The derivative of f with respect to x -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + a

/-- Theorem stating the condition for f to have exactly 3 zeros -/
theorem three_zeros_condition (a : ℝ) : 
  (∃ x₁ x₂ x₃, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ 
    f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0) ↔ 
  a < -3 ∧ a < 0 :=
sorry

end three_zeros_condition_l3802_380278


namespace partial_fraction_decomposition_constant_l3802_380209

theorem partial_fraction_decomposition_constant (A B C : ℝ) :
  (∀ x : ℝ, x ≠ 5 ∧ x ≠ -2 →
    1 / (x^3 + 2*x^2 - 17*x - 30) = A / (x - 5) + B / (x + 2) + C / (x + 2)^2) →
  A = 1 / 49 := by
sorry

end partial_fraction_decomposition_constant_l3802_380209


namespace bird_sanctuary_problem_l3802_380239

theorem bird_sanctuary_problem (total : ℝ) (total_positive : total > 0) :
  let geese := 0.2 * total
  let swans := 0.4 * total
  let herons := 0.1 * total
  let ducks := 0.2 * total
  let pigeons := total - (geese + swans + herons + ducks)
  let non_herons := total - herons
  (ducks / non_herons) * 100 = 22.22 := by sorry

end bird_sanctuary_problem_l3802_380239


namespace wendy_photo_albums_l3802_380272

theorem wendy_photo_albums 
  (phone_pics camera_pics pics_per_album : ℕ) 
  (h1 : phone_pics = 22)
  (h2 : camera_pics = 2)
  (h3 : pics_per_album = 6)
  : (phone_pics + camera_pics) / pics_per_album = 4 := by
  sorry

end wendy_photo_albums_l3802_380272


namespace similarity_transformation_result_l3802_380259

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the similarity transformation
def similarityTransform (p : Point2D) (ratio : ℝ) : Point2D :=
  { x := p.x * ratio, y := p.y * ratio }

-- Define the theorem
theorem similarity_transformation_result :
  let A : Point2D := { x := -4, y := 2 }
  let B : Point2D := { x := -6, y := -4 }
  let ratio : ℝ := 1/2
  let A' := similarityTransform A ratio
  (A'.x = -2 ∧ A'.y = 1) ∨ (A'.x = 2 ∧ A'.y = -1) :=
sorry

end similarity_transformation_result_l3802_380259


namespace cube_difference_l3802_380205

theorem cube_difference (c d : ℝ) (h1 : c - d = 7) (h2 : c^2 + d^2 = 85) : c^3 - d^3 = 721 := by
  sorry

end cube_difference_l3802_380205


namespace line_l_equation_line_l₃_equation_l3802_380279

-- Define the point M
def M : ℝ × ℝ := (3, 0)

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := 2 * x - y - 2 = 0
def l₂ (x y : ℝ) : Prop := x + y + 3 = 0

-- Define the line l
def l (x y : ℝ) : Prop := 8 * x - y - 24 = 0

-- Define the line l₃
def l₃ (x y : ℝ) : Prop := x - 2 * y - 5 = 0

-- Theorem for the equation of line l
theorem line_l_equation : 
  ∃ (P Q : ℝ × ℝ), 
    l₁ P.1 P.2 ∧ 
    l₂ Q.1 Q.2 ∧ 
    l M.1 M.2 ∧ 
    l P.1 P.2 ∧ 
    l Q.1 Q.2 ∧ 
    M = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2) :=
sorry

-- Theorem for the equation of line l₃
theorem line_l₃_equation :
  ∃ (I : ℝ × ℝ), 
    l₁ I.1 I.2 ∧ 
    l₂ I.1 I.2 ∧ 
    l₃ I.1 I.2 ∧
    ∀ (x y : ℝ), l₃ x y ↔ x - 2 * y - 5 = 0 :=
sorry

end line_l_equation_line_l₃_equation_l3802_380279


namespace groups_created_is_four_l3802_380263

/-- The number of groups created when dividing insects equally -/
def number_of_groups (boys_insects : ℕ) (girls_insects : ℕ) (insects_per_group : ℕ) : ℕ :=
  (boys_insects + girls_insects) / insects_per_group

/-- Theorem stating that the number of groups is 4 given the specific conditions -/
theorem groups_created_is_four :
  number_of_groups 200 300 125 = 4 := by
  sorry

end groups_created_is_four_l3802_380263


namespace tiles_per_row_l3802_380237

/-- Given a square room with an area of 256 square feet and tiles that are 8 inches wide,
    prove that 24 tiles can fit in one row. -/
theorem tiles_per_row (room_area : ℝ) (tile_width : ℝ) : 
  room_area = 256 → tile_width = 8 / 12 → (Real.sqrt room_area) * 12 / tile_width = 24 := by
  sorry

end tiles_per_row_l3802_380237


namespace sixPeopleRoundTable_l3802_380249

/-- Number of distinct seating arrangements for n people around a round table,
    considering rotational symmetry -/
def roundTableSeating (n : ℕ) : ℕ := Nat.factorial (n - 1)

/-- The number of distinct seating arrangements for 6 people around a round table,
    considering rotational symmetry, is 120 -/
theorem sixPeopleRoundTable : roundTableSeating 6 = 120 := by
  sorry

end sixPeopleRoundTable_l3802_380249


namespace triangle_area_l3802_380267

theorem triangle_area (A B C : ℝ) (b : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Acute triangle condition
  A + B + C = π ∧  -- Sum of angles in a triangle
  b = 2 ∧  -- Given condition
  B = π / 3 ∧  -- Given condition
  Real.sin (2 * A) + Real.sin (A - C) - Real.sin B = 0 →  -- Given equation
  (1 / 2) * b * b * Real.sin B = Real.sqrt 3  -- Area of the triangle
  := by sorry

end triangle_area_l3802_380267


namespace base_16_digits_for_5_digit_base_4_l3802_380215

theorem base_16_digits_for_5_digit_base_4 (n : ℕ) (h : 256 ≤ n ∧ n ≤ 1023) :
  (Nat.log 16 n).succ = 3 := by
  sorry

end base_16_digits_for_5_digit_base_4_l3802_380215


namespace polynomial_remainder_theorem_l3802_380201

def polynomial (x : ℝ) : ℝ := 10 * x^4 - 22 * x^3 + 5 * x^2 - 8 * x - 45

def divisor (x : ℝ) : ℝ := 5 * x - 10

theorem polynomial_remainder_theorem :
  ∃ (q : ℝ → ℝ), ∀ x, polynomial x = (divisor x) * q x + 3 := by
  sorry

end polynomial_remainder_theorem_l3802_380201


namespace triangle_angle_calculation_l3802_380270

theorem triangle_angle_calculation (a b c : Real) (A B C : Real) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively
  -- Law of sines: a / sin(A) = b / sin(B) = c / sin(C)
  (a / Real.sin A = b / Real.sin B) →
  -- Given conditions
  (a = 3) →
  (b = Real.sqrt 6) →
  (A = 2 * Real.pi / 3) →
  -- Conclusion
  B = Real.pi / 4 := by
sorry

end triangle_angle_calculation_l3802_380270


namespace hyperbola_eccentricity_l3802_380208

theorem hyperbola_eccentricity 
  (a b c : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (m n : ℝ) 
  (hmn : m * n = 2 / 9) :
  (((m + n) * c)^2 / a^2 - ((m - n) * b * c / a)^2 / b^2 = 1) →
  (c^2 / a^2 - 1 = (3 * Real.sqrt 2 / 4)^2) := by
sorry

end hyperbola_eccentricity_l3802_380208


namespace quadratic_function_minimum_ratio_quadratic_function_minimum_ratio_exact_l3802_380262

/-- A quadratic function f(x) = ax^2 + bx + c satisfying certain conditions -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  second_derivative_positive : 2 * a > 0
  non_negative : ∀ x : ℝ, a * x^2 + b * x + c ≥ 0

/-- The theorem stating the minimum value of f(1) / f''(0) for the quadratic function -/
theorem quadratic_function_minimum_ratio (f : QuadraticFunction) :
  (f.a + f.b + f.c) / (2 * f.a) ≥ 2 := by
  sorry

/-- The theorem stating that the minimum value of f(1) / f''(0) is exactly 2 -/
theorem quadratic_function_minimum_ratio_exact :
  ∃ f : QuadraticFunction, (f.a + f.b + f.c) / (2 * f.a) = 2 := by
  sorry

end quadratic_function_minimum_ratio_quadratic_function_minimum_ratio_exact_l3802_380262


namespace fraction_to_decimal_l3802_380250

-- Define the fraction 7/12
def fraction : ℚ := 7 / 12

-- Define the decimal approximation
def decimal_approx : ℝ := 0.5833

-- Define the maximum allowed error due to rounding
def max_error : ℝ := 0.00005

-- Theorem statement
theorem fraction_to_decimal :
  |((fraction : ℝ) - decimal_approx)| < max_error := by sorry

end fraction_to_decimal_l3802_380250


namespace tan_ratio_sum_l3802_380295

theorem tan_ratio_sum (x y : ℝ) 
  (h1 : (Real.sin x / Real.cos y) + (Real.sin y / Real.cos x) = 2)
  (h2 : (Real.cos x / Real.sin y) + (Real.cos y / Real.sin x) = 3) :
  (Real.tan x / Real.tan y) + (Real.tan y / Real.tan x) = 16/3 := by
  sorry

end tan_ratio_sum_l3802_380295


namespace line_segment_endpoint_l3802_380273

theorem line_segment_endpoint (y : ℝ) : 
  y > 0 → 
  Real.sqrt ((7 - 2)^2 + (y - 4)^2) = 6 → 
  y = 4 + Real.sqrt 11 := by
sorry

end line_segment_endpoint_l3802_380273

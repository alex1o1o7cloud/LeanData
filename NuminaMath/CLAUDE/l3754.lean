import Mathlib

namespace fraction_equality_l3754_375455

-- Define the @ operation
def at_op (a b : ℚ) : ℚ := a * b - b^3

-- Define the # operation
def hash_op (a b : ℚ) : ℚ := a^2 + b - a * b^2

-- Theorem statement
theorem fraction_equality : (at_op 4 3) / (hash_op 4 3) = 15 / 17 := by
  sorry

end fraction_equality_l3754_375455


namespace complex_expression_simplification_l3754_375448

theorem complex_expression_simplification (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (1.22 * (((Real.sqrt a + Real.sqrt b)^2 - 4*b) / ((a - b) / (Real.sqrt (1/b) + 3 * Real.sqrt (1/a)))) / 
  ((a + 9*b + 6 * Real.sqrt (a*b)) / (1 / Real.sqrt a + 1 / Real.sqrt b))) = 1 / (a * b) := by
  sorry

end complex_expression_simplification_l3754_375448


namespace tan_half_angle_l3754_375460

theorem tan_half_angle (α : Real) :
  (∃ (x y : Real), x = -1 ∧ y = 2 ∧ x = Real.cos α * Real.sqrt (x^2 + y^2) ∧ y = Real.sin α * Real.sqrt (x^2 + y^2)) →
  Real.tan (α / 2) = (1 + Real.sqrt 5) / 2 := by
  sorry

end tan_half_angle_l3754_375460


namespace unique_functional_equation_l3754_375479

/-- Given g: ℂ → ℂ, w ∈ ℂ, a ∈ ℂ, where w³ = 1 and w ≠ 1, 
    prove that the unique function f: ℂ → ℂ satisfying 
    f(z) + f(wz + a) = g(z) for all z ∈ ℂ 
    is given by f(z) = (g(z) + g(w²z + wa + a) - g(wz + a)) / 2 -/
theorem unique_functional_equation (g : ℂ → ℂ) (w a : ℂ) 
    (hw : w^3 = 1) (hw_neq : w ≠ 1) :
    ∃! f : ℂ → ℂ, ∀ z : ℂ, f z + f (w * z + a) = g z ∧
    f = fun z ↦ (g z + g (w^2 * z + w * a + a) - g (w * z + a)) / 2 := by
  sorry

end unique_functional_equation_l3754_375479


namespace sam_memorized_digits_l3754_375427

/-- Given information about the number of digits of pi memorized by Sam, Carlos, and Mina,
    prove that Sam memorized 10 digits. -/
theorem sam_memorized_digits (sam carlos mina : ℕ) 
  (h1 : sam = carlos + 6)
  (h2 : mina = 6 * carlos)
  (h3 : mina = 24) : 
  sam = 10 := by sorry

end sam_memorized_digits_l3754_375427


namespace fifteenth_term_is_101_l3754_375442

/-- Arithmetic sequence with first term a and common difference d -/
def arithmeticSequence (a d : ℤ) (n : ℕ) : ℤ := a + (n - 1) * d

/-- The 15th term of the arithmetic sequence with first term 3 and common difference 7 is 101 -/
theorem fifteenth_term_is_101 : arithmeticSequence 3 7 15 = 101 := by
  sorry

end fifteenth_term_is_101_l3754_375442


namespace trader_profit_l3754_375428

theorem trader_profit (original_price : ℝ) (original_price_positive : original_price > 0) : 
  let discount_rate : ℝ := 0.4
  let increase_rate : ℝ := 0.8
  let purchase_price : ℝ := original_price * (1 - discount_rate)
  let selling_price : ℝ := purchase_price * (1 + increase_rate)
  let profit : ℝ := selling_price - original_price
  let profit_percentage : ℝ := (profit / original_price) * 100
  profit_percentage = 8 := by sorry

end trader_profit_l3754_375428


namespace total_peaches_l3754_375400

theorem total_peaches (red_peaches green_peaches : ℕ) 
  (h1 : red_peaches = 13) 
  (h2 : green_peaches = 3) : 
  red_peaches + green_peaches = 16 := by
  sorry

end total_peaches_l3754_375400


namespace transformation_converts_curve_l3754_375416

-- Define the original curve
def original_curve (x y : ℝ) : Prop := y = 2 * Real.sin (3 * x)

-- Define the transformed curve
def transformed_curve (x' y' : ℝ) : Prop := y' = Real.sin x'

-- Define the transformation
def transformation (x y x' y' : ℝ) : Prop := x' = 3 * x ∧ y' = (1/2) * y

-- Theorem statement
theorem transformation_converts_curve :
  ∀ x y x' y' : ℝ,
  original_curve x y →
  transformation x y x' y' →
  transformed_curve x' y' :=
sorry

end transformation_converts_curve_l3754_375416


namespace right_triangle_shorter_leg_length_l3754_375420

theorem right_triangle_shorter_leg_length 
  (a : ℝ)  -- length of the shorter leg
  (h1 : a > 0)  -- ensure positive length
  (h2 : (a^2 + (2*a)^2)^(1/2) = a * 5^(1/2))  -- Pythagorean theorem
  (h3 : 12 = (1/2) * a * 5^(1/2))  -- median to hypotenuse formula
  : a = 24 * 5^(1/2) / 5 := by
  sorry

end right_triangle_shorter_leg_length_l3754_375420


namespace shaded_area_in_square_l3754_375487

/-- The area of a symmetric shaded region in a square -/
theorem shaded_area_in_square (square_side : ℝ) (point_A_x : ℝ) (point_B_x : ℝ) : 
  square_side = 10 →
  point_A_x = 7.5 →
  point_B_x = 7.5 →
  let shaded_area := 2 * (1/2 * (square_side/4) * (square_side/2))
  shaded_area = 28.125 := by sorry

end shaded_area_in_square_l3754_375487


namespace max_player_salary_max_salary_is_512000_l3754_375489

/-- The maximum possible salary for a single player in a minor league soccer team -/
theorem max_player_salary (n : ℕ) (min_salary : ℕ) (max_total : ℕ) : ℕ :=
  let max_single_salary := max_total - (n - 1) * min_salary
  max_single_salary

/-- The maximum possible salary for a single player in the given scenario is $512,000 -/
theorem max_salary_is_512000 :
  max_player_salary 25 12000 800000 = 512000 := by
  sorry

end max_player_salary_max_salary_is_512000_l3754_375489


namespace complement_A_intersect_B_l3754_375461

-- Define the sets
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x ≥ 2}
def B : Set ℝ := {x | 0 ≤ x ∧ x < 5}

-- State the theorem
theorem complement_A_intersect_B :
  (Aᶜ ∩ B) = {x : ℝ | 0 ≤ x ∧ x < 2} :=
by sorry

end complement_A_intersect_B_l3754_375461


namespace coin_piles_theorem_l3754_375494

/-- Represents the number of coins in each pile -/
structure CoinPiles :=
  (first : ℕ)
  (second : ℕ)
  (third : ℕ)

/-- Performs the coin transfers as described in the problem -/
def transfer (piles : CoinPiles) : CoinPiles :=
  let step1 := CoinPiles.mk (piles.first - piles.second) (piles.second + piles.second) piles.third
  let step2 := CoinPiles.mk step1.first (step1.second - step1.third) (step1.third + step1.third)
  CoinPiles.mk (step2.first + step2.third) step2.second (step2.third - step2.first)

/-- The main theorem stating the original number of coins in each pile -/
theorem coin_piles_theorem (piles : CoinPiles) :
  transfer piles = CoinPiles.mk 16 16 16 →
  piles = CoinPiles.mk 22 14 12 :=
by sorry

end coin_piles_theorem_l3754_375494


namespace min_value_of_f_l3754_375465

def f (x m : ℝ) : ℝ := 2 * x^3 - 6 * x^2 + m

theorem min_value_of_f (m : ℝ) :
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f y m ≤ f x m) ∧
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, f x m ≤ 3) →
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f x m ≤ f y m) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f x m = -37) :=
by sorry

end min_value_of_f_l3754_375465


namespace max_d_value_l3754_375424

def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def number_form (d e : ℕ) : ℕ := 707330 + d * 1000 + e

theorem max_d_value :
  ∃ (d e : ℕ),
    is_digit d ∧
    is_digit e ∧
    number_form d e % 33 = 0 ∧
    (∀ (d' e' : ℕ), is_digit d' ∧ is_digit e' ∧ number_form d' e' % 33 = 0 → d' ≤ d) ∧
    d = 6 :=
sorry

end max_d_value_l3754_375424


namespace max_area_between_lines_l3754_375464

-- Define the function
def f (x : ℝ) : ℝ := x^2 + 16

-- Define the area function
def area (x₀ : ℝ) : ℝ :=
  2 * (-2 * x₀^2 + 32 - 4 * x₀)

-- State the theorem
theorem max_area_between_lines :
  ∃ (x₀ : ℝ), x₀ ∈ Set.Icc (-3) 1 ∧ 
  (∀ (x : ℝ), x ∈ Set.Icc (-3) 1 → area x ≤ area x₀) ∧
  area x₀ = 68 := by
  sorry

end max_area_between_lines_l3754_375464


namespace minimum_beans_purchase_l3754_375454

theorem minimum_beans_purchase (r b : ℝ) : 
  (r ≥ 2 * b + 8 ∧ r ≤ 3 * b) → b ≥ 8 := by sorry

end minimum_beans_purchase_l3754_375454


namespace adams_change_l3754_375411

/-- Given that Adam has $5 and an airplane costs $4.28, prove that he will receive $0.72 in change. -/
theorem adams_change (adams_money : ℚ) (airplane_cost : ℚ) (h1 : adams_money = 5) (h2 : airplane_cost = 4.28) :
  adams_money - airplane_cost = 0.72 := by
  sorry

end adams_change_l3754_375411


namespace teacher_li_flags_l3754_375467

theorem teacher_li_flags : ∃ (x : ℕ), x > 0 ∧ 4 * x + 20 = 44 ∧ 4 * x + 20 > 8 * (x - 1) ∧ 4 * x + 20 < 8 * x :=
by sorry

end teacher_li_flags_l3754_375467


namespace cookout_attendance_l3754_375452

theorem cookout_attendance (kids_2004 kids_2005 kids_2006 : ℕ) : 
  kids_2005 = kids_2004 / 2 →
  kids_2006 = (2 * kids_2005) / 3 →
  kids_2006 = 20 →
  kids_2004 = 60 := by
  sorry

end cookout_attendance_l3754_375452


namespace sock_pair_count_l3754_375493

/-- The number of ways to choose a pair of socks of different colors -/
def different_color_pairs (white brown blue : ℕ) : ℕ :=
  white * brown + white * blue + brown * blue

/-- Theorem: The number of ways to choose a pair of socks of different colors
    from a drawer containing 5 white, 4 brown, and 3 blue distinguishable socks
    is equal to 47. -/
theorem sock_pair_count :
  different_color_pairs 5 4 3 = 47 := by
  sorry

end sock_pair_count_l3754_375493


namespace find_b_value_l3754_375495

theorem find_b_value (a b : ℝ) (h1 : 3 * a + 2 = 2) (h2 : b - 2 * a = 2) : b = 2 := by
  sorry

end find_b_value_l3754_375495


namespace area_equality_l3754_375486

-- Define the points
variable (A B C D M N K L : Plane)

-- Define the quadrilateral ABCD
def is_quadrilateral (A B C D : Plane) : Prop := sorry

-- Define that M is on AB and N is on CD
def on_segment (P Q R : Plane) : Prop := sorry

-- Define the ratio condition
def ratio_condition (A M B C N D : Plane) : Prop := sorry

-- Define the intersection points
def intersect_at (P Q R S T : Plane) : Prop := sorry

-- Define the area of a polygon
def area (points : List Plane) : ℝ := sorry

-- Theorem statement
theorem area_equality 
  (h1 : is_quadrilateral A B C D)
  (h2 : on_segment A M B)
  (h3 : on_segment C N D)
  (h4 : ratio_condition A M B C N D)
  (h5 : intersect_at A N D M K)
  (h6 : intersect_at B N C M L) :
  area [K, M, L, N] = area [A, D, K] + area [B, C, L] := by sorry

end area_equality_l3754_375486


namespace distance_product_l3754_375488

theorem distance_product (a₁ a₂ : ℝ) : 
  let p₁ := (3 * a₁, 2 * a₁ - 5)
  let p₂ := (6, -2)
  (p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2 = (3 * Real.sqrt 17)^2 →
  let p₁ := (3 * a₂, 2 * a₂ - 5)
  (p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2 = (3 * Real.sqrt 17)^2 →
  a₁ * a₂ = -2880 / 169 := by
sorry

end distance_product_l3754_375488


namespace inequality_proof_l3754_375476

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_given : (3 : ℝ) / (a * b * c) ≥ a + b + c) :
  1 / a + 1 / b + 1 / c ≥ a + b + c := by
  sorry

end inequality_proof_l3754_375476


namespace trees_left_l3754_375409

theorem trees_left (initial_trees dead_trees : ℕ) 
  (h1 : initial_trees = 150) 
  (h2 : dead_trees = 24) : 
  initial_trees - dead_trees = 126 := by
sorry

end trees_left_l3754_375409


namespace equal_areas_of_equal_ratios_l3754_375432

noncomputable def curvilinearTrapezoidArea (a b : ℝ) : ℝ := ∫ x in a..b, (1 / x)

theorem equal_areas_of_equal_ratios (a₁ b₁ a₂ b₂ : ℝ) 
  (ha₁ : 0 < a₁) (hb₁ : a₁ < b₁)
  (ha₂ : 0 < a₂) (hb₂ : a₂ < b₂)
  (h_ratio : b₁ / a₁ = b₂ / a₂) :
  curvilinearTrapezoidArea a₁ b₁ = curvilinearTrapezoidArea a₂ b₂ := by
  sorry

end equal_areas_of_equal_ratios_l3754_375432


namespace cylinder_height_equals_sphere_radius_l3754_375477

theorem cylinder_height_equals_sphere_radius 
  (r_sphere : ℝ) 
  (d_cylinder : ℝ) 
  (h_cylinder : ℝ) :
  r_sphere = 3 →
  d_cylinder = 6 →
  2 * π * (d_cylinder / 2) * h_cylinder = 4 * π * r_sphere^2 →
  h_cylinder = 6 :=
by sorry

end cylinder_height_equals_sphere_radius_l3754_375477


namespace sandwich_cost_l3754_375491

theorem sandwich_cost (sandwich_cost juice_cost milk_cost : ℝ) :
  juice_cost = 2 * sandwich_cost →
  milk_cost = 0.75 * (sandwich_cost + juice_cost) →
  sandwich_cost + juice_cost + milk_cost = 21 →
  sandwich_cost = 4 := by
sorry

end sandwich_cost_l3754_375491


namespace surface_integral_I_value_surface_integral_J_value_surface_integral_K_value_l3754_375414

-- Define the surface integrals
def surface_integral_I (a : ℝ) : ℝ := sorry

def surface_integral_J : ℝ := sorry

def surface_integral_K : ℝ := sorry

-- State the theorems to be proved
theorem surface_integral_I_value (a : ℝ) (h : a > 0) : 
  surface_integral_I a = (4 * Real.pi / 5) * a^(5/2) := by sorry

theorem surface_integral_J_value : 
  surface_integral_J = (8 * Real.pi / 3) - (4 / 15) := by sorry

theorem surface_integral_K_value : 
  surface_integral_K = -1 / 6 := by sorry

end surface_integral_I_value_surface_integral_J_value_surface_integral_K_value_l3754_375414


namespace fifth_term_is_five_l3754_375402

/-- An arithmetic sequence is represented by its first term and common difference. -/
structure ArithmeticSequence where
  first_term : ℝ
  common_difference : ℝ

/-- Get the nth term of an arithmetic sequence. -/
def ArithmeticSequence.nth_term (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  seq.first_term + (n - 1) * seq.common_difference

theorem fifth_term_is_five
  (seq : ArithmeticSequence)
  (h : seq.nth_term 2 + seq.nth_term 4 = 10) :
  seq.nth_term 5 = 5 := by
  sorry

end fifth_term_is_five_l3754_375402


namespace vector_operation_result_unique_linear_combination_l3754_375446

-- Define the vectors
def a : ℝ × ℝ := (3, 2)
def b : ℝ × ℝ := (-1, 2)
def c : ℝ × ℝ := (5, 6)

-- Theorem for part 1
theorem vector_operation_result : 
  (3 • a) + b - (2 • c) = (-2, -4) := by sorry

-- Theorem for part 2
theorem unique_linear_combination :
  ∃! (m n : ℝ), c = m • a + n • b ∧ m = 2 ∧ n = 1 := by sorry

-- Note: • is used for scalar multiplication in Lean

end vector_operation_result_unique_linear_combination_l3754_375446


namespace area_ABDE_l3754_375456

/-- A regular hexagon with side length 2 -/
structure RegularHexagon :=
  (side_length : ℝ)
  (is_regular : side_length = 2)

/-- The quadrilateral ABDE formed by four vertices of the regular hexagon -/
def ABDE (h : RegularHexagon) : Set (ℝ × ℝ) :=
  {(2, 0), (1, Real.sqrt 3), (-2, 0), (-1, -Real.sqrt 3)}

/-- The area of a set of points in ℝ² -/
noncomputable def area (s : Set (ℝ × ℝ)) : ℝ := sorry

/-- Theorem: The area of ABDE in a regular hexagon with side length 2 is 4√3 -/
theorem area_ABDE (h : RegularHexagon) : area (ABDE h) = 4 * Real.sqrt 3 := by
  sorry

end area_ABDE_l3754_375456


namespace relationship_2x_3sinx_l3754_375401

theorem relationship_2x_3sinx :
  ∃ θ : ℝ, 0 < θ ∧ θ < π / 2 ∧
  (∀ x : ℝ, 0 < x → x < θ → 2 * x < 3 * Real.sin x) ∧
  (2 * θ = 3 * Real.sin θ) ∧
  (∀ x : ℝ, θ < x → x < π / 2 → 2 * x > 3 * Real.sin x) := by
  sorry

end relationship_2x_3sinx_l3754_375401


namespace log_equation_equivalence_l3754_375425

theorem log_equation_equivalence (x : ℝ) :
  (∀ y : ℝ, y > 0 → ∃ z : ℝ, Real.exp z = y) →  -- This ensures logarithms are defined for positive reals
  (x > (3/2) ↔ (Real.log (x+5) + Real.log (2*x-3) = Real.log (2*x^2 + x - 15))) :=
by sorry

end log_equation_equivalence_l3754_375425


namespace simplify_and_rationalize_l3754_375437

theorem simplify_and_rationalize : 
  (Real.sqrt 3 / Real.sqrt 5) * (Real.sqrt 8 / Real.sqrt 10) * (Real.sqrt 15 / Real.sqrt 21) = 
  (2 * Real.sqrt 105) / 35 := by
  sorry

end simplify_and_rationalize_l3754_375437


namespace arccos_cos_gt_arcsin_sin_iff_l3754_375472

theorem arccos_cos_gt_arcsin_sin_iff (x : ℝ) : 
  (∃ k : ℤ, 2 * k * Real.pi + Real.pi / 2 < x ∧ x < 2 * (k + 1) * Real.pi) ↔ 
  Real.arccos (Real.cos x) > Real.arcsin (Real.sin x) :=
sorry

end arccos_cos_gt_arcsin_sin_iff_l3754_375472


namespace equation_solutions_l3754_375431

theorem equation_solutions (a b c d : ℤ) (hab : a ≠ b) :
  let f : ℤ × ℤ → ℤ := λ (x, y) ↦ (x + a * y + c) * (x + b * y + d)
  (∃ S : Finset (ℤ × ℤ), (∀ p ∈ S, f p = 2) ∧ S.card ≤ 4) ∧
  ((|a - b| = 1 ∨ |a - b| = 2) → (c - d) % 2 ≠ 0 →
    ∃ S : Finset (ℤ × ℤ), (∀ p ∈ S, f p = 2) ∧ S.card = 4) := by
  sorry

#check equation_solutions

end equation_solutions_l3754_375431


namespace mixed_repeating_decimal_denominator_divisibility_l3754_375499

/-- Represents a mixed repeating decimal -/
structure MixedRepeatingDecimal where
  non_repeating : ℕ
  repeating : ℕ

/-- Theorem: For any mixed repeating decimal that can be expressed as an irreducible fraction p/q,
    the denominator q is divisible by 2 or 5, or both. -/
theorem mixed_repeating_decimal_denominator_divisibility
  (x : MixedRepeatingDecimal)
  (p q : ℕ)
  (h_irreducible : Nat.Coprime p q)
  (h_fraction : (p : ℚ) / q = x.non_repeating + (x.repeating : ℚ) / (10^x.non_repeating.succ * (10^x.repeating.succ - 1))) :
  2 ∣ q ∨ 5 ∣ q :=
sorry

end mixed_repeating_decimal_denominator_divisibility_l3754_375499


namespace abs_p_minus_q_equals_five_l3754_375429

theorem abs_p_minus_q_equals_five (p q : ℝ) (h1 : p * q = 6) (h2 : p + q = 7) :
  |p - q| = 5 := by
  sorry

end abs_p_minus_q_equals_five_l3754_375429


namespace class_size_l3754_375475

theorem class_size (mini_cupcakes : ℕ) (donut_holes : ℕ) (desserts_per_student : ℕ) : 
  mini_cupcakes = 14 → 
  donut_holes = 12 → 
  desserts_per_student = 2 → 
  (mini_cupcakes + donut_holes) / desserts_per_student = 13 := by
sorry

end class_size_l3754_375475


namespace line_plane_perpendicularity_l3754_375405

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (subset : Line → Plane → Prop)
variable (perp_planes : Plane → Plane → Prop)

-- State the theorem
theorem line_plane_perpendicularity
  (m n : Line) (α β : Plane)
  (diff_lines : m ≠ n)
  (diff_planes : α ≠ β)
  (m_parallel_n : parallel m n)
  (n_perp_β : perpendicular n β)
  (m_subset_α : subset m α) :
  perp_planes α β :=
sorry

end line_plane_perpendicularity_l3754_375405


namespace least_number_with_remainder_two_five_six_satisfies_conditions_least_number_is_256_l3754_375436

theorem least_number_with_remainder (n : ℕ) : 
  (n % 7 = 4) ∧ (n % 9 = 4) ∧ (n % 12 = 4) ∧ (n % 18 = 4) →
  n ≥ 256 :=
by sorry

theorem two_five_six_satisfies_conditions : 
  (256 % 7 = 4) ∧ (256 % 9 = 4) ∧ (256 % 12 = 4) ∧ (256 % 18 = 4) :=
by sorry

theorem least_number_is_256 : 
  ∀ n : ℕ, n < 256 → 
  ¬((n % 7 = 4) ∧ (n % 9 = 4) ∧ (n % 12 = 4) ∧ (n % 18 = 4)) :=
by sorry

end least_number_with_remainder_two_five_six_satisfies_conditions_least_number_is_256_l3754_375436


namespace sue_candy_count_l3754_375415

/-- Represents the number of candies each person has -/
structure CandyCount where
  bob : Nat
  mary : Nat
  john : Nat
  sam : Nat
  sue : Nat

/-- The total number of candies for all friends -/
def totalCandies (cc : CandyCount) : Nat :=
  cc.bob + cc.mary + cc.john + cc.sam + cc.sue

theorem sue_candy_count (cc : CandyCount) 
  (h1 : cc.bob = 10)
  (h2 : cc.mary = 5)
  (h3 : cc.john = 5)
  (h4 : cc.sam = 10)
  (h5 : totalCandies cc = 50) :
  cc.sue = 20 := by
  sorry


end sue_candy_count_l3754_375415


namespace rectangular_prism_cut_out_l3754_375490

theorem rectangular_prism_cut_out (x y : ℤ) : 
  (15 * 5 * 4 - y * 5 * x = 120) → 
  (0 < x) → 
  (x < 4) → 
  (0 < y) → 
  (y < 15) → 
  (x = 3 ∧ y = 12) := by
sorry

end rectangular_prism_cut_out_l3754_375490


namespace max_abs_z_l3754_375421

/-- Given a complex number z satisfying |z - 8| + |z + 6i| = 10, 
    the maximum value of |z| is 8. -/
theorem max_abs_z (z : ℂ) (h : Complex.abs (z - 8) + Complex.abs (z + 6*I) = 10) : 
  ∃ (w : ℂ), Complex.abs (w - 8) + Complex.abs (w + 6*I) = 10 ∧ 
             ∀ (u : ℂ), Complex.abs (u - 8) + Complex.abs (u + 6*I) = 10 → 
             Complex.abs u ≤ Complex.abs w ∧
             Complex.abs w = 8 :=
sorry

end max_abs_z_l3754_375421


namespace expected_hit_targets_bound_l3754_375482

theorem expected_hit_targets_bound (n : ℕ) (hn : n > 0) :
  let p := 1 - (1 - 1 / n)^n
  n * p ≥ n / 2 := by
  sorry

end expected_hit_targets_bound_l3754_375482


namespace interest_rate_calculation_l3754_375469

theorem interest_rate_calculation (total : ℝ) (part2 : ℝ) (years1 : ℝ) (years2 : ℝ) (rate2 : ℝ) :
  total = 2665 →
  part2 = 1332.5 →
  years1 = 5 →
  years2 = 3 →
  rate2 = 0.05 →
  let part1 := total - part2
  let r := (part2 * rate2 * years2) / (part1 * years1)
  r = 0.03 :=
by sorry

end interest_rate_calculation_l3754_375469


namespace one_fourth_of_ten_times_twelve_divided_by_two_l3754_375459

theorem one_fourth_of_ten_times_twelve_divided_by_two : (1 / 4 : ℚ) * ((10 * 12) / 2) = 15 := by
  sorry

end one_fourth_of_ten_times_twelve_divided_by_two_l3754_375459


namespace edward_initial_money_l3754_375451

def toy_car_cost : ℚ := 0.95
def race_track_cost : ℚ := 6.00
def num_toy_cars : ℕ := 4
def remaining_money : ℚ := 8.00

theorem edward_initial_money :
  ∃ (initial_money : ℚ),
    initial_money = num_toy_cars * toy_car_cost + race_track_cost + remaining_money :=
by
  sorry

end edward_initial_money_l3754_375451


namespace min_value_of_function_l3754_375435

theorem min_value_of_function (x : ℝ) (h : x ≥ 2) :
  x + 5 / (x + 1) ≥ 11 / 3 ∧
  (x + 5 / (x + 1) = 11 / 3 ↔ x = 2) :=
by sorry

end min_value_of_function_l3754_375435


namespace sum_of_extrema_l3754_375462

theorem sum_of_extrema (a b c : ℝ) (h1 : a + b + c = 5) (h2 : a^2 + b^2 + c^2 = 11) :
  ∃ (m M : ℝ), (∀ x, (∃ y z, x + y + z = 5 ∧ x^2 + y^2 + z^2 = 11) → m ≤ x ∧ x ≤ M) ∧
                m + M = 3 :=
sorry

end sum_of_extrema_l3754_375462


namespace stratified_sampling_medium_stores_l3754_375412

theorem stratified_sampling_medium_stores
  (total_stores : ℕ)
  (medium_stores : ℕ)
  (sample_size : ℕ)
  (h1 : total_stores = 300)
  (h2 : medium_stores = 75)
  (h3 : sample_size = 20) :
  (medium_stores : ℚ) / total_stores * sample_size = 5 := by
  sorry

end stratified_sampling_medium_stores_l3754_375412


namespace ab_value_l3754_375480

theorem ab_value (a b : ℝ) 
  (h : ∀ x : ℝ, x ≥ 0 → 0 ≤ x^4 - x^3 + a*x + b ∧ x^4 - x^3 + a*x + b ≤ (x^2 - 1)^2) : 
  a * b = -1 := by
sorry

end ab_value_l3754_375480


namespace probability_A_and_B_selected_is_three_tenths_l3754_375458

def total_students : ℕ := 5
def students_to_select : ℕ := 3

def probability_A_and_B_selected : ℚ :=
  (total_students - students_to_select + 1 : ℚ) / total_students.choose students_to_select

theorem probability_A_and_B_selected_is_three_tenths :
  probability_A_and_B_selected = 3 / 10 := by
  sorry

end probability_A_and_B_selected_is_three_tenths_l3754_375458


namespace box_length_proof_l3754_375410

/-- Proves that the length of a box with given dimensions and cube requirements is 10 cm -/
theorem box_length_proof (width : ℝ) (height : ℝ) (cube_volume : ℝ) (min_cubes : ℕ) 
  (h_width : width = 13)
  (h_height : height = 5)
  (h_cube_volume : cube_volume = 5)
  (h_min_cubes : min_cubes = 130) :
  (min_cubes : ℝ) * cube_volume / (width * height) = 10 := by
  sorry

end box_length_proof_l3754_375410


namespace cherries_theorem_l3754_375492

def cherries_problem (initial : ℕ) (eaten : ℕ) : ℕ :=
  let remaining := initial - eaten
  let given_away := remaining / 2
  remaining - given_away

theorem cherries_theorem :
  cherries_problem 2450 1625 = 413 := by
  sorry

end cherries_theorem_l3754_375492


namespace star_running_back_yardage_l3754_375496

/-- Represents the yardage gained by a player in a football game -/
structure Yardage where
  running : ℕ
  catching : ℕ

/-- Calculates the total yardage for a player -/
def totalYardage (y : Yardage) : ℕ :=
  y.running + y.catching

/-- Theorem: The total yardage of a player who gained 90 yards running and 60 yards catching is 150 yards -/
theorem star_running_back_yardage :
  let y : Yardage := { running := 90, catching := 60 }
  totalYardage y = 150 := by
  sorry

end star_running_back_yardage_l3754_375496


namespace orange_boxes_pigeonhole_l3754_375417

theorem orange_boxes_pigeonhole (total_boxes : ℕ) (min_oranges max_oranges : ℕ) :
  total_boxes = 150 →
  min_oranges = 100 →
  max_oranges = 130 →
  ∃ n : ℕ, n ≥ 5 ∧ ∃ k : ℕ, k ≥ min_oranges ∧ k ≤ max_oranges ∧ 
    (∃ boxes : Finset (Fin total_boxes), boxes.card = n ∧ 
      ∀ i ∈ boxes, ∃ f : Fin total_boxes → ℕ, f i = k) :=
by sorry

end orange_boxes_pigeonhole_l3754_375417


namespace distance_proof_l3754_375406

def point : ℝ × ℝ × ℝ := (2, 1, -5)

def line_point : ℝ × ℝ × ℝ := (4, -3, 2)
def line_direction : ℝ × ℝ × ℝ := (-1, 4, 3)

def distance_to_line (p : ℝ × ℝ × ℝ) (l_point : ℝ × ℝ × ℝ) (l_dir : ℝ × ℝ × ℝ) : ℝ :=
  sorry

theorem distance_proof : 
  distance_to_line point line_point line_direction = Real.sqrt (34489 / 676) := by
  sorry

end distance_proof_l3754_375406


namespace symmetry_axis_implies_ratio_l3754_375407

/-- Given a function f(x) = 2m*sin(x) - n*cos(x), if x = π/3 is an axis of symmetry
    for the graph of f(x), then n/m = -2√3/3 -/
theorem symmetry_axis_implies_ratio (m n : ℝ) (h : m ≠ 0) :
  (∀ x : ℝ, 2 * m * Real.sin x - n * Real.cos x =
    2 * m * Real.sin (2 * π / 3 - x) - n * Real.cos (2 * π / 3 - x)) →
  n / m = -2 * Real.sqrt 3 / 3 := by
  sorry

end symmetry_axis_implies_ratio_l3754_375407


namespace four_people_permutations_l3754_375404

/-- The number of permutations of n distinct objects -/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- There are 4 distinct individuals -/
def num_people : ℕ := 4

/-- Theorem: The number of ways 4 people can stand in a line is 24 -/
theorem four_people_permutations : permutations num_people = 24 := by
  sorry

end four_people_permutations_l3754_375404


namespace increasing_sequence_count_l3754_375470

def sequence_count (n m : ℕ) : ℕ := Nat.choose (n + m - 1) m

theorem increasing_sequence_count : 
  let n := 675
  let m := 15
  sequence_count n m = Nat.choose 689 15 ∧ 689 % 1000 = 689 := by sorry

#eval sequence_count 675 15
#eval 689 % 1000

end increasing_sequence_count_l3754_375470


namespace jerry_age_l3754_375419

theorem jerry_age (mickey_age jerry_age : ℕ) : 
  mickey_age = 16 → 
  mickey_age = 2 * jerry_age - 6 → 
  jerry_age = 11 := by
sorry

end jerry_age_l3754_375419


namespace hyperbola_center_is_correct_l3754_375445

/-- The center of a hyperbola given by the equation 9x^2 - 54x - 36y^2 + 288y - 576 = 0 -/
def hyperbola_center : ℝ × ℝ := (3, 4)

/-- The equation of the hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop :=
  9 * x^2 - 54 * x - 36 * y^2 + 288 * y - 576 = 0

/-- Theorem stating that the center of the hyperbola is (3, 4) -/
theorem hyperbola_center_is_correct :
  let (h₁, h₂) := hyperbola_center
  ∀ (ε : ℝ), ε ≠ 0 →
    ∃ (δ : ℝ), δ > 0 ∧
      ∀ (x y : ℝ),
        hyperbola_equation x y →
        (x - h₁)^2 + (y - h₂)^2 < δ^2 →
        (x - h₁)^2 + (y - h₂)^2 < ε^2 :=
by sorry

end hyperbola_center_is_correct_l3754_375445


namespace unique_square_friendly_l3754_375418

/-- A function that checks if a number is a perfect square -/
def is_perfect_square (n : ℤ) : Prop :=
  ∃ k : ℤ, n = k^2

/-- Definition of a square-friendly integer -/
def is_square_friendly (c : ℤ) : Prop :=
  ∀ m : ℤ, is_perfect_square (m^2 + 18*m + c)

/-- Theorem: 81 is the only square-friendly integer -/
theorem unique_square_friendly :
  ∃! c : ℤ, is_square_friendly c ∧ c = 81 :=
sorry

end unique_square_friendly_l3754_375418


namespace suv_max_distance_l3754_375481

/-- Represents the fuel efficiency of an SUV in miles per gallon -/
structure FuelEfficiency where
  highway : ℝ
  city : ℝ

/-- Calculates the maximum distance an SUV can travel given its fuel efficiency and available fuel -/
def max_distance (efficiency : FuelEfficiency) (fuel : ℝ) : ℝ :=
  efficiency.highway * fuel

/-- Theorem: The maximum distance an SUV with 12.2 mpg highway efficiency can travel on 23 gallons of fuel is 280.6 miles -/
theorem suv_max_distance :
  let suv_efficiency : FuelEfficiency := { highway := 12.2, city := 7.6 }
  let available_fuel : ℝ := 23
  max_distance suv_efficiency available_fuel = 280.6 := by
  sorry


end suv_max_distance_l3754_375481


namespace coin_trick_theorem_l3754_375423

/-- Represents the state of a coin (heads or tails) -/
inductive CoinState
| Heads
| Tails

/-- Represents a row of 27 coins -/
def CoinRow := Vector CoinState 27

/-- Represents a selection of 5 coins from the row -/
def CoinSelection := Vector Nat 5

/-- Function to check if all coins in a selection are facing the same way -/
def allSameFacing (row : CoinRow) (selection : CoinSelection) : Prop :=
  ∀ i j, i < 5 → j < 5 → row.get (selection.get i) = row.get (selection.get j)

/-- The main theorem stating that it's always possible to select 10 coins facing the same way,
    such that 5 of them can determine the state of the other 5 -/
theorem coin_trick_theorem (row : CoinRow) :
  ∃ (selection1 selection2 : CoinSelection),
    allSameFacing row selection1 ∧
    allSameFacing row selection2 ∧
    (∀ i, i < 5 → selection1.get i ≠ selection2.get i) ∧
    (∃ f : CoinSelection → CoinSelection, f selection1 = selection2) :=
sorry

end coin_trick_theorem_l3754_375423


namespace munchausen_claim_correct_l3754_375430

-- Define a function to calculate the sum of digits
def sumOfDigits (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem munchausen_claim_correct : 
  ∃ (a b : ℕ), 
    (10^9 ≤ a ∧ a < 10^10) ∧ 
    (10^9 ≤ b ∧ b < 10^10) ∧ 
    a ≠ b ∧ 
    a % 10 ≠ 0 ∧ 
    b % 10 ≠ 0 ∧ 
    a + sumOfDigits (a^2) = b + sumOfDigits (b^2) := by
  sorry

end munchausen_claim_correct_l3754_375430


namespace square_area_from_diagonal_l3754_375463

theorem square_area_from_diagonal (d : ℝ) (h : d = 12 * Real.sqrt 2) :
  d^2 / 2 = 144 := by
  sorry

end square_area_from_diagonal_l3754_375463


namespace gcd_powers_of_two_l3754_375484

theorem gcd_powers_of_two : Nat.gcd (2^2025 - 1) (2^2007 - 1) = 2^18 - 1 := by
  sorry

end gcd_powers_of_two_l3754_375484


namespace stripe_area_on_cylindrical_silo_l3754_375413

/-- The area of a stripe wrapping around a cylindrical silo -/
theorem stripe_area_on_cylindrical_silo 
  (diameter : ℝ) 
  (stripe_width : ℝ) 
  (revolutions : ℕ) 
  (h1 : diameter = 20) 
  (h2 : stripe_width = 4) 
  (h3 : revolutions = 3) : 
  stripe_width * revolutions * (π * diameter) = 240 * π := by
sorry

end stripe_area_on_cylindrical_silo_l3754_375413


namespace bertha_family_childless_count_l3754_375485

/-- Represents a family tree with two generations -/
structure FamilyTree where
  daughters : ℕ
  granddaughters : ℕ

/-- Bertha's family tree -/
def berthas_family : FamilyTree := { daughters := 10, granddaughters := 32 }

/-- The number of Bertha's daughters who have children -/
def daughters_with_children : ℕ := 8

/-- The number of daughters each child-bearing daughter has -/
def granddaughters_per_daughter : ℕ := 4

theorem bertha_family_childless_count :
  berthas_family.daughters + berthas_family.granddaughters - daughters_with_children = 34 :=
by sorry

end bertha_family_childless_count_l3754_375485


namespace enclosed_area_theorem_l3754_375497

/-- The common area enclosed by 4 equilateral triangles with side length 1, 
    each sharing a side with one of the 4 sides of a unit square. -/
def commonAreaEnclosedByTriangles : ℝ := -1

/-- The side length of the square -/
def squareSideLength : ℝ := 1

/-- The side length of each equilateral triangle -/
def triangleSideLength : ℝ := 1

/-- The number of equilateral triangles -/
def numberOfTriangles : ℕ := 4

theorem enclosed_area_theorem :
  commonAreaEnclosedByTriangles = -1 := by sorry

end enclosed_area_theorem_l3754_375497


namespace smallest_number_square_and_cube_l3754_375440

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m ^ 2

def is_perfect_cube (n : ℕ) : Prop := ∃ m : ℕ, n = m ^ 3

theorem smallest_number_square_and_cube :
  ∃ n : ℕ, n = 72 ∧
    is_perfect_square (n * 2) ∧
    is_perfect_cube (n * 3) ∧
    ∀ m : ℕ, m < n →
      ¬(is_perfect_square (m * 2) ∧ is_perfect_cube (m * 3)) :=
by sorry

end smallest_number_square_and_cube_l3754_375440


namespace club_group_size_theorem_l3754_375471

theorem club_group_size_theorem (N : ℕ) (x : ℕ) 
  (h1 : 20 < N ∧ N < 50) 
  (h2 : (N - 5) % 6 = 0 ∧ (N - 5) % 7 = 0) 
  (h3 : N % x = 7) : 
  x = 8 := by
sorry

end club_group_size_theorem_l3754_375471


namespace right_triangle_third_side_product_l3754_375478

theorem right_triangle_third_side_product (a b : ℝ) (ha : a = 5) (hb : b = 7) :
  (Real.sqrt (a^2 + b^2)) * (Real.sqrt (max a b)^2 - (min a b)^2) = Real.sqrt 1776 := by
  sorry

end right_triangle_third_side_product_l3754_375478


namespace zeros_of_quadratic_function_l3754_375457

theorem zeros_of_quadratic_function (x : ℝ) :
  x^2 = 0 → x ∈ Set.Ioo (-1 : ℝ) 1 :=
by
  sorry

end zeros_of_quadratic_function_l3754_375457


namespace multiple_of_six_squared_gt_200_lt_30_l3754_375408

theorem multiple_of_six_squared_gt_200_lt_30 (x : ℕ) 
  (h1 : ∃ k : ℕ, x = 6 * k)
  (h2 : x^2 > 200)
  (h3 : x < 30) :
  x = 18 ∨ x = 24 :=
by sorry

end multiple_of_six_squared_gt_200_lt_30_l3754_375408


namespace seller_deception_l3754_375449

theorem seller_deception (a w : ℝ) (ha : a > 0) (hw : w > 0) (ha_neq_1 : a ≠ 1) :
  (a * w + w / a) / 2 ≥ w ∧
  ((a * w + w / a) / 2 = w ↔ a = 1) :=
by sorry

end seller_deception_l3754_375449


namespace ice_cream_bar_price_l3754_375453

theorem ice_cream_bar_price 
  (num_ice_cream_bars : ℕ)
  (num_sundaes : ℕ)
  (total_price : ℚ)
  (sundae_price : ℚ)
  (h1 : num_ice_cream_bars = 125)
  (h2 : num_sundaes = 125)
  (h3 : total_price = 225)
  (h4 : sundae_price = 6/5) :
  (total_price - num_sundaes * sundae_price) / num_ice_cream_bars = 3/5 := by
sorry

end ice_cream_bar_price_l3754_375453


namespace correct_transformation_l3754_375498

theorem correct_transformation (x : ℝ) : 2*x - 5 = 3*x + 3 → 2*x - 3*x = 3 + 5 := by
  sorry

end correct_transformation_l3754_375498


namespace perpendicular_implies_m_eq_neg_one_parallel_implies_m_eq_neg_one_l3754_375444

-- Define the lines l₁ and l₂
def l₁ (m : ℝ) (x y : ℝ) : Prop := (m - 2) * x + 3 * y + 2 * m = 0
def l₂ (m : ℝ) (x y : ℝ) : Prop := x + m * y + 6 = 0

-- Define perpendicularity of lines
def perpendicular (m : ℝ) : Prop :=
  ∀ x₁ y₁ x₂ y₂ : ℝ, l₁ m x₁ y₁ → l₂ m x₂ y₂ → (m - 2) / 3 * m = -1

-- Define parallelism of lines
def parallel (m : ℝ) : Prop :=
  ∀ x₁ y₁ x₂ y₂ : ℝ, l₁ m x₁ y₁ → l₂ m x₂ y₂ → (m - 2) / 3 = m

-- Theorem for perpendicular case
theorem perpendicular_implies_m_eq_neg_one :
  ∀ m : ℝ, perpendicular m → m = -1 := by sorry

-- Theorem for parallel case
theorem parallel_implies_m_eq_neg_one :
  ∀ m : ℝ, parallel m → m = -1 := by sorry

end perpendicular_implies_m_eq_neg_one_parallel_implies_m_eq_neg_one_l3754_375444


namespace identify_liars_in_two_questions_l3754_375474

/-- Represents a person who can be either a knight or a liar -/
inductive Person
| Knight
| Liar

/-- Represents a position on a regular decagon -/
structure Position :=
  (angle : ℝ)

/-- Represents the state of the problem -/
structure DecagonState :=
  (people : Fin 10 → Person)
  (positions : Fin 10 → Position)

/-- Represents a question asked by the traveler -/
structure Question :=
  (position : Position)

/-- Represents an answer given by a person -/
structure Answer :=
  (distance : ℝ)

/-- Function to determine the answer given by a person -/
def getAnswer (state : DecagonState) (person : Fin 10) (q : Question) : Answer :=
  sorry

/-- Function to determine if a person is a liar based on their answer -/
def isLiar (state : DecagonState) (person : Fin 10) (q : Question) (a : Answer) : Bool :=
  sorry

/-- Theorem stating that at most 2 questions are needed to identify all liars -/
theorem identify_liars_in_two_questions (state : DecagonState) :
  ∃ (q1 q2 : Question), ∀ (person : Fin 10),
    isLiar state person q1 (getAnswer state person q1) ∨
    isLiar state person q2 (getAnswer state person q2) =
    (state.people person = Person.Liar) :=
  sorry

end identify_liars_in_two_questions_l3754_375474


namespace smallest_number_l3754_375433

theorem smallest_number : ∀ (a b c d : ℚ), 
  a = -3 ∧ b = -2 ∧ c = 0 ∧ d = 1/3 → 
  a ≤ b ∧ a ≤ c ∧ a ≤ d := by sorry

end smallest_number_l3754_375433


namespace red_box_position_l3754_375439

/-- Given a collection of boxes with a red box among them, this function
    calculates the position of the red box from the right when arranged
    from largest to smallest, given its position from the right when
    arranged from smallest to largest. -/
def position_from_right_largest_to_smallest (total_boxes : ℕ) (position_smallest_to_largest : ℕ) : ℕ :=
  total_boxes - (position_smallest_to_largest - 1)

/-- Theorem stating that for 45 boxes with the red box 29th from the right
    when arranged smallest to largest, it will be 17th from the right
    when arranged largest to smallest. -/
theorem red_box_position (total_boxes : ℕ) (position_smallest_to_largest : ℕ) 
    (h1 : total_boxes = 45)
    (h2 : position_smallest_to_largest = 29) :
    position_from_right_largest_to_smallest total_boxes position_smallest_to_largest = 17 := by
  sorry

#eval position_from_right_largest_to_smallest 45 29

end red_box_position_l3754_375439


namespace midpoint_x_coordinate_sum_l3754_375422

theorem midpoint_x_coordinate_sum (a b c : ℝ) :
  let S := a + b + c
  let midpoint1 := (a + b) / 2
  let midpoint2 := (a + c) / 2
  let midpoint3 := (b + c) / 2
  midpoint1 + midpoint2 + midpoint3 = S := by sorry

end midpoint_x_coordinate_sum_l3754_375422


namespace ceiling_sqrt_225_l3754_375426

theorem ceiling_sqrt_225 : ⌈Real.sqrt 225⌉ = 15 := by
  sorry

end ceiling_sqrt_225_l3754_375426


namespace triangle_special_angle_relation_l3754_375483

/-- In a triangle ABC where α = 3β = 6γ, the equation bc² = (a+b)(a-b)² holds true. -/
theorem triangle_special_angle_relation (a b c : ℝ) (α β γ : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →  -- positive side lengths
  0 < α ∧ 0 < β ∧ 0 < γ →  -- positive angles
  α + β + γ = π →         -- sum of angles in a triangle
  α = 3*β →               -- given condition
  α = 6*γ →               -- given condition
  b*c^2 = (a+b)*(a-b)^2 := by
sorry

end triangle_special_angle_relation_l3754_375483


namespace sample_size_proof_l3754_375450

/-- Given a sample divided into 3 groups with specific frequencies, prove the sample size. -/
theorem sample_size_proof (f1 f2 f3 : ℝ) (n1 : ℝ) (h1 : f2 = 0.35) (h2 : f3 = 0.45) (h3 : n1 = 10) :
  ∃ M : ℝ, M = 50 ∧ f1 + f2 + f3 = 1 ∧ n1 / M = f1 := by
  sorry

end sample_size_proof_l3754_375450


namespace solution_in_fourth_quadrant_l3754_375434

-- Define the equation system
def equation_system (x y : ℝ) : Prop :=
  y = 2 * x - 5 ∧ y = -x + 1

-- Define the fourth quadrant
def fourth_quadrant (x y : ℝ) : Prop :=
  x > 0 ∧ y < 0

-- Theorem statement
theorem solution_in_fourth_quadrant :
  ∃ x y : ℝ, equation_system x y ∧ fourth_quadrant x y :=
sorry

end solution_in_fourth_quadrant_l3754_375434


namespace no_trapezoid_solution_l3754_375468

theorem no_trapezoid_solution : 
  ¬ ∃ (b₁ b₂ : ℕ), 
    (b₁ % 10 = 0) ∧ 
    (b₂ % 10 = 0) ∧ 
    ((b₁ + b₂) * 30 / 2 = 1080) :=
by sorry

end no_trapezoid_solution_l3754_375468


namespace pencil_sales_problem_l3754_375473

/-- The number of pencils initially sold for a rupee -/
def initial_pencils : ℝ := 11

/-- The number of pencils sold for a rupee to achieve a 20% gain -/
def gain_pencils : ℝ := 8.25

/-- The loss percentage when selling the initial number of pencils -/
def loss_percentage : ℝ := 10

/-- The gain percentage when selling 8.25 pencils -/
def gain_percentage : ℝ := 20

theorem pencil_sales_problem :
  (1 = (1 - loss_percentage / 100) * initial_pencils * (1 / gain_pencils)) ∧
  (1 = (1 + gain_percentage / 100) * 1) ∧
  initial_pencils = 11 := by sorry

end pencil_sales_problem_l3754_375473


namespace price_of_pants_l3754_375441

/-- Given Iris's shopping trip to the mall, this theorem proves the price of each pair of pants. -/
theorem price_of_pants (jacket_price : ℕ) (shorts_price : ℕ) (total_spent : ℕ) 
  (jacket_count : ℕ) (shorts_count : ℕ) (pants_count : ℕ) :
  jacket_price = 10 →
  shorts_price = 6 →
  jacket_count = 3 →
  shorts_count = 2 →
  pants_count = 4 →
  total_spent = 90 →
  ∃ (pants_price : ℕ), 
    pants_price * pants_count + jacket_price * jacket_count + shorts_price * shorts_count = total_spent ∧
    pants_price = 12 :=
by sorry

end price_of_pants_l3754_375441


namespace orthocenter_of_specific_triangle_l3754_375466

/-- Triangle ABC in 3D space -/
structure Triangle3D where
  A : ℝ × ℝ × ℝ
  B : ℝ × ℝ × ℝ
  C : ℝ × ℝ × ℝ

/-- The orthocenter of a triangle in 3D space -/
def orthocenter (t : Triangle3D) : ℝ × ℝ × ℝ := sorry

/-- Theorem: The orthocenter of the given triangle is (5/2, 3, 7/2) -/
theorem orthocenter_of_specific_triangle :
  let t : Triangle3D := {
    A := (1, 2, 3),
    B := (5, 3, 1),
    C := (3, 4, 5)
  }
  orthocenter t = (5/2, 3, 7/2) := by sorry

end orthocenter_of_specific_triangle_l3754_375466


namespace mike_changed_tires_on_12_motorcycles_l3754_375403

/-- The number of motorcycles Mike changed tires on -/
def num_motorcycles (total_tires num_cars tires_per_car tires_per_motorcycle : ℕ) : ℕ :=
  (total_tires - num_cars * tires_per_car) / tires_per_motorcycle

theorem mike_changed_tires_on_12_motorcycles :
  num_motorcycles 64 10 4 2 = 12 := by
  sorry

end mike_changed_tires_on_12_motorcycles_l3754_375403


namespace function_properties_l3754_375438

def f (x m : ℝ) : ℝ := |x - m| - |x + 3 * m|

theorem function_properties (m : ℝ) (h : m > 0) :
  (m = 1 → {x : ℝ | f x m ≥ 1} = {x : ℝ | x ≤ -3/2}) ∧
  ({m : ℝ | ∀ x t : ℝ, f x m < |2 + t| + |t - 1|} = {m : ℝ | 0 < m ∧ m < 3/4}) :=
sorry

end function_properties_l3754_375438


namespace ellipse_constant_dot_product_l3754_375443

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define the moving line
def moving_line (k x y : ℝ) : Prop := y = k * (x - 1)

-- Define the dot product of vectors MA and MB
def dot_product_MA_MB (m x1 y1 x2 y2 : ℝ) : ℝ :=
  (x1 - m) * (x2 - m) + y1 * y2

-- Statement of the theorem
theorem ellipse_constant_dot_product :
  ∃ (m : ℝ), 
    (∀ (k x1 y1 x2 y2 : ℝ), k ≠ 0 →
      ellipse_C x1 y1 → ellipse_C x2 y2 →
      moving_line k x1 y1 → moving_line k x2 y2 →
      dot_product_MA_MB m x1 y1 x2 y2 = -7/16) ∧
    m = 5/4 := by
  sorry

end ellipse_constant_dot_product_l3754_375443


namespace identify_tasty_candies_l3754_375447

/-- Represents a candy on the table. -/
structure Candy where
  tasty : Bool

/-- Represents the state of the game. -/
structure GameState where
  candies : Finset Candy
  moves_left : Nat

/-- Represents a query about a subset of candies. -/
def Query := Finset Candy → Nat

/-- The main theorem stating that all tasty candies can be identified within the given number of moves. -/
theorem identify_tasty_candies 
  (n : Nat) 
  (candies : Finset Candy) 
  (h1 : candies.card = 28) 
  (query : Query) : 
  (∃ (strategy : GameState → Finset Candy), 
    (∀ (gs : GameState), 
      gs.candies = candies → 
      gs.moves_left ≥ 21 → 
      strategy gs = {c ∈ candies | c.tasty})) ∧ 
    (∃ (strategy : GameState → Finset Candy), 
      (∀ (gs : GameState), 
        gs.candies = candies → 
        gs.moves_left ≥ 20 → 
        strategy gs = {c ∈ candies | c.tasty})) :=
by sorry

end identify_tasty_candies_l3754_375447

import Mathlib

namespace NUMINAMATH_CALUDE_jenny_lasagna_profit_l2708_270893

/-- Calculate Jenny's profit from selling lasagna pans -/
def jennys_profit (cost_per_pan : ℝ) (price_per_pan : ℝ) (num_pans : ℕ) : ℝ :=
  (price_per_pan * num_pans) - (cost_per_pan * num_pans)

/-- Theorem: Jenny's profit is $300 given the problem conditions -/
theorem jenny_lasagna_profit :
  jennys_profit 10 25 20 = 300 := by
  sorry

end NUMINAMATH_CALUDE_jenny_lasagna_profit_l2708_270893


namespace NUMINAMATH_CALUDE_four_hearts_probability_l2708_270897

-- Define a standard deck of cards
def standard_deck : ℕ := 52

-- Define the number of hearts in a standard deck
def hearts_in_deck : ℕ := 13

-- Define the number of cards we're drawing
def cards_drawn : ℕ := 4

-- Define the probability of drawing four hearts
def prob_four_hearts : ℚ := 2 / 95

-- Theorem statement
theorem four_hearts_probability :
  (hearts_in_deck.factorial / (hearts_in_deck - cards_drawn).factorial) /
  (standard_deck.factorial / (standard_deck - cards_drawn).factorial) = prob_four_hearts := by
  sorry

end NUMINAMATH_CALUDE_four_hearts_probability_l2708_270897


namespace NUMINAMATH_CALUDE_equation_solution_l2708_270887

theorem equation_solution : ∃ x : ℤ, x * (x + 2) + 1 = 36 ∧ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2708_270887


namespace NUMINAMATH_CALUDE_parabola_properties_l2708_270884

-- Define the parabola
def parabola (x : ℝ) : ℝ := -2 * (x + 1)^2 - 3

-- Define the shifted parabola
def shifted_parabola (x : ℝ) : ℝ := -2 * (x + 3)^2 + 1

-- Theorem statement
theorem parabola_properties :
  (∀ x : ℝ, parabola x ≤ parabola (-1)) ∧
  (parabola (-1) = -3) ∧
  (∀ x : ℝ, shifted_parabola x = parabola (x + 2) + 4) :=
sorry

end NUMINAMATH_CALUDE_parabola_properties_l2708_270884


namespace NUMINAMATH_CALUDE_problem_solution_l2708_270822

theorem problem_solution (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_sum : a + b + c = 1) : 
  (∀ x y z, x > 0 → y > 0 → z > 0 → x + y + z = 1 → 
    1/a + 1/b + 1/c ≤ 1/x + 1/y + 1/z) ∧
  (1/(1-a) + 1/(1-b) + 1/(1-c) ≥ 2/(1+a) + 2/(1+b) + 2/(1+c)) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2708_270822


namespace NUMINAMATH_CALUDE_income_comparison_l2708_270858

theorem income_comparison (a b : ℝ) (h : a = 0.75 * b) : 
  (b - a) / a = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_income_comparison_l2708_270858


namespace NUMINAMATH_CALUDE_actual_distance_traveled_l2708_270819

theorem actual_distance_traveled (original_speed : ℝ) (increased_speed : ℝ) (additional_distance : ℝ) :
  original_speed = 15 →
  increased_speed = 25 →
  additional_distance = 35 →
  (∃ (time : ℝ), time > 0 ∧ time * increased_speed = time * original_speed + additional_distance) →
  ∃ (actual_distance : ℝ), actual_distance = 52.5 ∧ actual_distance = original_speed * (actual_distance / original_speed) :=
by sorry

end NUMINAMATH_CALUDE_actual_distance_traveled_l2708_270819


namespace NUMINAMATH_CALUDE_complex_w_values_l2708_270857

theorem complex_w_values (z : ℂ) (w : ℂ) 
  (h1 : ∃ (r : ℝ), (1 + 3*I) * z = r)
  (h2 : w = z / (2 + I))
  (h3 : Complex.abs w = 5 * Real.sqrt 2) :
  w = 1 + 7*I ∨ w = -1 - 7*I := by
  sorry

end NUMINAMATH_CALUDE_complex_w_values_l2708_270857


namespace NUMINAMATH_CALUDE_product_zero_implies_factor_zero_l2708_270872

theorem product_zero_implies_factor_zero (a b : ℝ) : a * b = 0 → (a = 0 ∨ b = 0) := by
  contrapose
  intro h
  push_neg
  simp
  sorry

end NUMINAMATH_CALUDE_product_zero_implies_factor_zero_l2708_270872


namespace NUMINAMATH_CALUDE_set_operation_result_l2708_270889

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def A : Set Nat := {1, 3, 6}
def B : Set Nat := {1, 2}

theorem set_operation_result (C : Set Nat) (h : C ⊆ U) : 
  (C ∪ A) ∩ B = {2} := by sorry

end NUMINAMATH_CALUDE_set_operation_result_l2708_270889


namespace NUMINAMATH_CALUDE_equal_quantities_after_transfer_l2708_270888

def container_problem (initial_A initial_B initial_C transfer : ℝ) : Prop :=
  let final_B := initial_B + transfer
  let final_C := initial_C - transfer
  initial_A = 1184 ∧
  initial_B = 0.375 * initial_A ∧
  initial_C = initial_A - initial_B ∧
  transfer = 148 →
  final_B = final_C

theorem equal_quantities_after_transfer :
  ∃ (initial_A initial_B initial_C transfer : ℝ),
    container_problem initial_A initial_B initial_C transfer :=
  sorry

end NUMINAMATH_CALUDE_equal_quantities_after_transfer_l2708_270888


namespace NUMINAMATH_CALUDE_inequality_preservation_l2708_270879

theorem inequality_preservation (a b c : ℝ) (h : a > b) : a / (c^2 + 1) > b / (c^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l2708_270879


namespace NUMINAMATH_CALUDE_taehyungs_mother_age_l2708_270865

/-- Given the age differences and the age of Taehyung's younger brother, 
    prove that Taehyung's mother is 43 years old. -/
theorem taehyungs_mother_age :
  ∀ (taehyung_age brother_age mother_age : ℕ),
    taehyung_age = brother_age + 5 →
    brother_age = 7 →
    mother_age = taehyung_age + 31 →
    mother_age = 43 :=
by
  sorry

end NUMINAMATH_CALUDE_taehyungs_mother_age_l2708_270865


namespace NUMINAMATH_CALUDE_prime_sequence_l2708_270874

def f (p : ℕ) (x : ℕ) : ℕ := x^2 + x + p

theorem prime_sequence (p : ℕ) :
  (∀ k : ℕ, k ≤ Real.sqrt (p / 3) → Nat.Prime (f p k)) →
  (∀ n : ℕ, n ≤ p - 2 → Nat.Prime (f p n)) :=
sorry

end NUMINAMATH_CALUDE_prime_sequence_l2708_270874


namespace NUMINAMATH_CALUDE_cos_sum_eleventh_l2708_270810

theorem cos_sum_eleventh : 
  Real.cos (2 * Real.pi / 11) + Real.cos (6 * Real.pi / 11) + Real.cos (8 * Real.pi / 11) = 
    (-1 + Real.sqrt (-11)) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_sum_eleventh_l2708_270810


namespace NUMINAMATH_CALUDE_fraction_simplification_l2708_270846

theorem fraction_simplification (a : ℝ) (h : a ≠ 1) :
  a / (a - 1) + 1 / (1 - a) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2708_270846


namespace NUMINAMATH_CALUDE_quadratic_function_range_l2708_270870

/-- Given a quadratic function f(x) = ax^2 + bx, prove that if 1 ≤ f(-1) ≤ 2 and 2 ≤ f(1) ≤ 4, then 3 ≤ f(-2) ≤ 12. -/
theorem quadratic_function_range (a b : ℝ) : 
  let f : ℝ → ℝ := λ x => a * x^2 + b * x
  (1 ≤ f (-1) ∧ f (-1) ≤ 2) ∧ (2 ≤ f 1 ∧ f 1 ≤ 4) →
  3 ≤ f (-2) ∧ f (-2) ≤ 12 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_range_l2708_270870


namespace NUMINAMATH_CALUDE_hyperbola_perpendicular_points_sum_l2708_270886

/-- Given a hyperbola x²/a² - y²/b² = 1 with 0 < a < b, and points A and B on the hyperbola
    such that OA is perpendicular to OB, prove that 1/|OA|² + 1/|OB|² = 1/a² - 1/b² -/
theorem hyperbola_perpendicular_points_sum (a b : ℝ) (ha : 0 < a) (hb : a < b)
  (A B : ℝ × ℝ) (hA : A.1^2 / a^2 - A.2^2 / b^2 = 1) (hB : B.1^2 / a^2 - B.2^2 / b^2 = 1)
  (hperp : A.1 * B.1 + A.2 * B.2 = 0) :
  1 / (A.1^2 + A.2^2) + 1 / (B.1^2 + B.2^2) = 1 / a^2 - 1 / b^2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_perpendicular_points_sum_l2708_270886


namespace NUMINAMATH_CALUDE_shortest_major_axis_ellipse_l2708_270824

-- Define the line l
def line_l (x y : ℝ) : Prop := y = x + 2

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := 12 * x^2 - 4 * y^2 = 3

-- Define a general ellipse
def ellipse (a b x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the condition for an ellipse to share foci with the hyperbola
def shared_foci (a b : ℝ) : Prop := a^2 - b^2 = 1

-- Define the tangency condition
def is_tangent (a b : ℝ) : Prop := ∃ x y : ℝ, line_l x y ∧ ellipse a b x y

-- Theorem statement
theorem shortest_major_axis_ellipse :
  ∀ a b : ℝ, a > 0 → b > 0 →
  shared_foci a b →
  is_tangent a b →
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → shared_foci a' b' → is_tangent a' b' → a ≤ a') →
  a^2 = 5 ∧ b^2 = 4 :=
sorry

end NUMINAMATH_CALUDE_shortest_major_axis_ellipse_l2708_270824


namespace NUMINAMATH_CALUDE_chord_length_l2708_270835

theorem chord_length (r d : ℝ) (hr : r = 5) (hd : d = 4) :
  let chord_length := 2 * Real.sqrt (r^2 - d^2)
  chord_length = 6 := by
  sorry

end NUMINAMATH_CALUDE_chord_length_l2708_270835


namespace NUMINAMATH_CALUDE_f_at_negative_one_l2708_270847

def f (x : ℝ) : ℝ := 5 * (2 * x^3 - 3 * x^2 + 4 * x - 1)

theorem f_at_negative_one : f (-1) = -50 := by
  sorry

end NUMINAMATH_CALUDE_f_at_negative_one_l2708_270847


namespace NUMINAMATH_CALUDE_max_cross_section_area_l2708_270864

/-- A regular hexagonal prism with side length 8 and vertical edges parallel to the z-axis -/
structure HexagonalPrism where
  side_length : ℝ
  side_length_eq : side_length = 8

/-- The plane that cuts the prism -/
def cutting_plane (x y z : ℝ) : Prop :=
  5 * x - 8 * y + 3 * z = 40

/-- The cross-section formed by cutting the prism with the plane -/
def cross_section (p : HexagonalPrism) (x y z : ℝ) : Prop :=
  cutting_plane x y z

/-- The area of the cross-section -/
noncomputable def cross_section_area (p : HexagonalPrism) : ℝ :=
  sorry

/-- The theorem stating that the maximum area of the cross-section is 144√3 -/
theorem max_cross_section_area (p : HexagonalPrism) :
    ∃ (a : ℝ), cross_section_area p = a ∧ a ≤ 144 * Real.sqrt 3 ∧
    ∀ (b : ℝ), cross_section_area p ≤ b → b ≥ 144 * Real.sqrt 3 :=
  sorry

end NUMINAMATH_CALUDE_max_cross_section_area_l2708_270864


namespace NUMINAMATH_CALUDE_expression_simplification_l2708_270845

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 2 + 1) :
  (1 + 1 / (x^2 - 1)) / (x^2 / (x^2 + 2*x + 1)) = 1 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2708_270845


namespace NUMINAMATH_CALUDE_negation_of_proposition_l2708_270848

theorem negation_of_proposition (p : ∀ x : ℝ, x^2 - x + 1 > 0) :
  (∃ x_0 : ℝ, x_0^2 - x_0 + 1 ≤ 0) ↔ ¬(∀ x : ℝ, x^2 - x + 1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l2708_270848


namespace NUMINAMATH_CALUDE_f_has_one_zero_in_interval_l2708_270833

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^3 - 6 * x^2 + 7

-- State the theorem
theorem f_has_one_zero_in_interval :
  ∃! x : ℝ, 0 < x ∧ x < 2 ∧ f x = 0 :=
sorry

end NUMINAMATH_CALUDE_f_has_one_zero_in_interval_l2708_270833


namespace NUMINAMATH_CALUDE_article_gain_percentage_l2708_270825

/-- Calculates the cost price given the selling price and loss percentage -/
def costPrice (sellingPrice : ℚ) (lossPercentage : ℚ) : ℚ :=
  sellingPrice / (1 - lossPercentage / 100)

/-- Calculates the gain percentage given the cost price and selling price -/
def gainPercentage (costPrice : ℚ) (sellingPrice : ℚ) : ℚ :=
  (sellingPrice - costPrice) / costPrice * 100

theorem article_gain_percentage :
  let cp := costPrice 170 15
  gainPercentage cp 240 = 20 := by
  sorry

end NUMINAMATH_CALUDE_article_gain_percentage_l2708_270825


namespace NUMINAMATH_CALUDE_count_negative_numbers_l2708_270812

theorem count_negative_numbers : 
  let expressions := [-2^2, (-2)^2, -(-2), -|-2|]
  (expressions.filter (λ x => x < 0)).length = 2 := by
  sorry

end NUMINAMATH_CALUDE_count_negative_numbers_l2708_270812


namespace NUMINAMATH_CALUDE_lemons_for_new_recipe_l2708_270862

/-- Represents the number of lemons per gallon in the original recipe -/
def original_lemons_per_gallon : ℚ := 36 / 48

/-- Represents the additional lemons per gallon in the new recipe -/
def additional_lemons_per_gallon : ℚ := 2 / 6

/-- Represents the number of gallons we want to make -/
def gallons_to_make : ℚ := 18

/-- Theorem stating that 18 gallons of the new recipe requires 19.5 lemons -/
theorem lemons_for_new_recipe : 
  (original_lemons_per_gallon + additional_lemons_per_gallon) * gallons_to_make = 19.5 := by
  sorry

end NUMINAMATH_CALUDE_lemons_for_new_recipe_l2708_270862


namespace NUMINAMATH_CALUDE_acute_triangle_sine_sum_l2708_270806

theorem acute_triangle_sine_sum (α β γ : Real) 
  (acute_triangle : 0 < α ∧ α < π/2 ∧ 0 < β ∧ β < π/2 ∧ 0 < γ ∧ γ < π/2)
  (angle_sum : α + β + γ = π) : 
  Real.sin α + Real.sin β + Real.sin γ > 2 := by
sorry

end NUMINAMATH_CALUDE_acute_triangle_sine_sum_l2708_270806


namespace NUMINAMATH_CALUDE_student_contribution_l2708_270842

theorem student_contribution
  (total_contribution : ℕ)
  (class_funds : ℕ)
  (num_students : ℕ)
  (h1 : total_contribution = 90)
  (h2 : class_funds = 14)
  (h3 : num_students = 19) :
  (total_contribution - class_funds) / num_students = 4 :=
by sorry

end NUMINAMATH_CALUDE_student_contribution_l2708_270842


namespace NUMINAMATH_CALUDE_preimage_of_four_one_l2708_270869

/-- The mapping f from R² to R² defined by f(x,y) = (x+y, x-y) -/
def f (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + p.2, p.1 - p.2)

/-- Theorem stating that (2.5, 1.5) is the preimage of (4, 1) under f -/
theorem preimage_of_four_one :
  f (2.5, 1.5) = (4, 1) := by
  sorry

end NUMINAMATH_CALUDE_preimage_of_four_one_l2708_270869


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l2708_270820

theorem arithmetic_expression_equality : 2 + 8 * 3 - 4 + 7 * 2 / 2 = 29 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l2708_270820


namespace NUMINAMATH_CALUDE_gcd_twelve_digit_form_l2708_270832

def is_six_digit (n : ℕ) : Prop := 100000 ≤ n ∧ n ≤ 999999

def twelve_digit_form (m : ℕ) : ℕ := 1000001 * m

theorem gcd_twelve_digit_form :
  ∃ (g : ℕ), ∀ (m : ℕ), is_six_digit m → 
    (∃ (k : ℕ), twelve_digit_form m = g * k) ∧
    (∀ (d : ℕ), (∀ (n : ℕ), is_six_digit n → ∃ (k : ℕ), twelve_digit_form n = d * k) → d ≤ g) ∧
    g = 1000001 :=
by sorry

end NUMINAMATH_CALUDE_gcd_twelve_digit_form_l2708_270832


namespace NUMINAMATH_CALUDE_man_downstream_speed_l2708_270898

/-- Calculates the downstream speed given upstream and still water speeds -/
def downstream_speed (upstream : ℝ) (still_water : ℝ) : ℝ :=
  2 * still_water - upstream

theorem man_downstream_speed :
  let upstream_speed : ℝ := 12
  let still_water_speed : ℝ := 7
  downstream_speed upstream_speed still_water_speed = 14 := by
  sorry

end NUMINAMATH_CALUDE_man_downstream_speed_l2708_270898


namespace NUMINAMATH_CALUDE_f_range_l2708_270861

noncomputable def f (x : ℝ) : ℝ :=
  if x > 1 then (Real.log x) / x else Real.exp x + 1

theorem f_range :
  Set.range f = Set.union (Set.Ioc 0 (1 / Real.exp 1)) (Set.Ioo 1 (Real.exp 1 + 1)) := by
  sorry

end NUMINAMATH_CALUDE_f_range_l2708_270861


namespace NUMINAMATH_CALUDE_point_reflection_y_axis_l2708_270871

/-- Given a point Q with coordinates (-3,7) in the Cartesian coordinate system,
    its coordinates with respect to the y-axis are (3,7). -/
theorem point_reflection_y_axis :
  let Q : ℝ × ℝ := (-3, 7)
  let reflected_Q : ℝ × ℝ := (3, 7)
  reflected_Q = (- Q.1, Q.2) :=
by sorry

end NUMINAMATH_CALUDE_point_reflection_y_axis_l2708_270871


namespace NUMINAMATH_CALUDE_original_cube_volume_l2708_270866

theorem original_cube_volume (s : ℝ) : 
  (2 * s)^3 = 2744 → s^3 = 343 := by
  sorry

end NUMINAMATH_CALUDE_original_cube_volume_l2708_270866


namespace NUMINAMATH_CALUDE_exponent_sum_l2708_270815

theorem exponent_sum (x a b c : ℝ) (h1 : x ≠ 1) (h2 : x * x^a * x^b * x^c = x^2024) : 
  a + b + c = 2023 := by
sorry

end NUMINAMATH_CALUDE_exponent_sum_l2708_270815


namespace NUMINAMATH_CALUDE_f_not_prime_l2708_270891

def f (n : ℕ+) : ℤ := n.val^4 - 380 * n.val^2 + 841

theorem f_not_prime : ∀ n : ℕ+, ¬ Nat.Prime (Int.natAbs (f n)) := by
  sorry

end NUMINAMATH_CALUDE_f_not_prime_l2708_270891


namespace NUMINAMATH_CALUDE_keith_total_spent_l2708_270867

-- Define the amounts spent on each item
def speakers_cost : ℚ := 136.01
def cd_player_cost : ℚ := 139.38
def tires_cost : ℚ := 112.46

-- Define the total amount spent
def total_spent : ℚ := speakers_cost + cd_player_cost + tires_cost

-- Theorem to prove
theorem keith_total_spent :
  total_spent = 387.85 :=
by sorry

end NUMINAMATH_CALUDE_keith_total_spent_l2708_270867


namespace NUMINAMATH_CALUDE_only_whole_number_between_l2708_270826

theorem only_whole_number_between (N : ℤ) : 
  (9.25 < (N : ℚ) / 4 ∧ (N : ℚ) / 4 < 9.75) ↔ N = 38 := by
  sorry

end NUMINAMATH_CALUDE_only_whole_number_between_l2708_270826


namespace NUMINAMATH_CALUDE_simplify_like_terms_l2708_270852

theorem simplify_like_terms (y : ℝ) : 5 * y + 7 * y + 8 * y = 20 * y := by
  sorry

end NUMINAMATH_CALUDE_simplify_like_terms_l2708_270852


namespace NUMINAMATH_CALUDE_probability_of_red_ball_l2708_270853

theorem probability_of_red_ball (p_red_white p_red_black : ℝ) 
  (h1 : p_red_white = 0.58)
  (h2 : p_red_black = 0.62) :
  p_red_white + p_red_black - 1 = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_red_ball_l2708_270853


namespace NUMINAMATH_CALUDE_triangle_side_length_l2708_270859

theorem triangle_side_length (a b c : ℝ) (angle_C : ℝ) : 
  a = 3 → b = 5 → angle_C = 2 * π / 3 → c = 7 := by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2708_270859


namespace NUMINAMATH_CALUDE_no_triangle_with_cube_sum_equal_to_product_l2708_270836

theorem no_triangle_with_cube_sum_equal_to_product (x y z : ℝ) :
  (0 < x ∧ 0 < y ∧ 0 < z) →
  (x + y > z ∧ y + z > x ∧ z + x > y) →
  x^3 + y^3 + z^3 ≠ (x + y) * (y + z) * (z + x) :=
by sorry

end NUMINAMATH_CALUDE_no_triangle_with_cube_sum_equal_to_product_l2708_270836


namespace NUMINAMATH_CALUDE_framed_photo_area_l2708_270839

/-- The area of a framed rectangular photo -/
theorem framed_photo_area 
  (paper_width : ℝ) 
  (paper_length : ℝ) 
  (frame_width : ℝ) 
  (h1 : paper_width = 8) 
  (h2 : paper_length = 12) 
  (h3 : frame_width = 2) : 
  (paper_width + 2 * frame_width) * (paper_length + 2 * frame_width) = 192 :=
by sorry

end NUMINAMATH_CALUDE_framed_photo_area_l2708_270839


namespace NUMINAMATH_CALUDE_complement_intersection_problem_l2708_270844

def U : Set Nat := {2, 3, 4, 5, 6}
def A : Set Nat := {2, 5, 6}
def B : Set Nat := {3, 5}

theorem complement_intersection_problem : (U \ A) ∩ B = {3} := by
  sorry

end NUMINAMATH_CALUDE_complement_intersection_problem_l2708_270844


namespace NUMINAMATH_CALUDE_number_puzzle_l2708_270817

theorem number_puzzle (x : ℝ) : (x - 26) / 2 = 37 → 48 - x / 4 = 23 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l2708_270817


namespace NUMINAMATH_CALUDE_five_dice_probability_l2708_270850

/-- The probability of rolling a number greater than 1 on a single die -/
def prob_not_one : ℚ := 5/6

/-- The number of ways to choose 2 dice out of 5 -/
def choose_two_from_five : ℕ := 10

/-- The probability of two dice summing to 10 -/
def prob_sum_ten : ℚ := 1/12

/-- The probability of rolling five dice where none show 1 and two of them sum to 10 -/
def prob_five_dice : ℚ := (prob_not_one ^ 5) * choose_two_from_five * prob_sum_ten

theorem five_dice_probability :
  prob_five_dice = 2604.1667 / 7776 :=
sorry

end NUMINAMATH_CALUDE_five_dice_probability_l2708_270850


namespace NUMINAMATH_CALUDE_knicks_to_knocks_equivalence_l2708_270840

/-- Represents the number of units of a given type -/
structure UnitCount (α : Type) where
  count : ℚ

/-- Conversion rate between two types of units -/
def ConversionRate (α β : Type) : Type :=
  UnitCount α → UnitCount β

/-- Given conversion rates, prove that 40 knicks are equivalent to 36 knocks -/
theorem knicks_to_knocks_equivalence 
  (knick knack knock : Type)
  (knicks_to_knacks : ConversionRate knick knack)
  (knacks_to_knocks : ConversionRate knack knock)
  (h1 : knicks_to_knacks ⟨5⟩ = ⟨3⟩)
  (h2 : knacks_to_knocks ⟨4⟩ = ⟨6⟩)
  : ∃ (f : ConversionRate knick knock), f ⟨40⟩ = ⟨36⟩ := by
  sorry

end NUMINAMATH_CALUDE_knicks_to_knocks_equivalence_l2708_270840


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l2708_270837

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, |x| + x^2 ≥ 0) ↔ (∃ x₀ : ℝ, |x₀| + x₀^2 < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l2708_270837


namespace NUMINAMATH_CALUDE_lottery_win_probability_l2708_270881

def megaBallCount : ℕ := 30
def winnerBallCount : ℕ := 50
def winnerBallDrawCount : ℕ := 6

def lotteryWinProbability : ℚ :=
  2 / (megaBallCount * (winnerBallCount.choose winnerBallDrawCount))

theorem lottery_win_probability :
  lotteryWinProbability = 2 / 477621000 := by sorry

end NUMINAMATH_CALUDE_lottery_win_probability_l2708_270881


namespace NUMINAMATH_CALUDE_shaded_region_perimeter_l2708_270807

/-- Given a circle with center O and radius 8, where the shaded region is half of the circle plus two radii, 
    the perimeter of the shaded region is 16 + 8π. -/
theorem shaded_region_perimeter (O : Point) (r : ℝ) (h1 : r = 8) : 
  let perimeter := 2 * r + (π * r)
  perimeter = 16 + 8 * π := by sorry

end NUMINAMATH_CALUDE_shaded_region_perimeter_l2708_270807


namespace NUMINAMATH_CALUDE_michael_gave_two_crates_l2708_270896

/-- Calculates the number of crates Michael gave to Susan -/
def crates_given_to_susan (crates_tuesday : ℕ) (crates_thursday : ℕ) (eggs_per_crate : ℕ) (eggs_remaining : ℕ) : ℕ :=
  let total_crates := crates_tuesday + crates_thursday
  let total_eggs := total_crates * eggs_per_crate
  (total_eggs - eggs_remaining) / eggs_per_crate

theorem michael_gave_two_crates :
  crates_given_to_susan 6 5 30 270 = 2 := by
  sorry

end NUMINAMATH_CALUDE_michael_gave_two_crates_l2708_270896


namespace NUMINAMATH_CALUDE_consecutive_binomial_coefficient_ratio_l2708_270860

theorem consecutive_binomial_coefficient_ratio (n k : ℕ) : 
  (n.choose k : ℚ) / (n.choose (k + 1) : ℚ) = 1 / 3 ∧
  (n.choose (k + 1) : ℚ) / (n.choose (k + 2) : ℚ) = 1 / 2 →
  n + k = 12 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_binomial_coefficient_ratio_l2708_270860


namespace NUMINAMATH_CALUDE_wire_ratio_proof_l2708_270803

theorem wire_ratio_proof (total_length longer_length shorter_length : ℝ) 
  (h1 : total_length = 14)
  (h2 : shorter_length = 4)
  (h3 : longer_length = total_length - shorter_length) :
  shorter_length / longer_length = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_wire_ratio_proof_l2708_270803


namespace NUMINAMATH_CALUDE_triangle_side_equation_l2708_270816

/-- Given a triangle ABC with vertex A at (1,4), and angle bisectors of B and C
    represented by the equations x + y - 1 = 0 and x - 2y = 0 respectively,
    the side BC lies on the line with equation 4x + 17y + 12 = 0. -/
theorem triangle_side_equation (B C : ℝ × ℝ) : 
  let A : ℝ × ℝ := (1, 4)
  let angle_bisector_B : ℝ → ℝ → Prop := λ x y => x + y - 1 = 0
  let angle_bisector_C : ℝ → ℝ → Prop := λ x y => x - 2*y = 0
  let line_BC : ℝ → ℝ → Prop := λ x y => 4*x + 17*y + 12 = 0
  (∀ x y, x = B.1 ∧ y = B.2 → angle_bisector_B x y) →
  (∀ x y, x = C.1 ∧ y = C.2 → angle_bisector_C x y) →
  (∀ x y, (x = B.1 ∧ y = B.2) ∨ (x = C.1 ∧ y = C.2) → line_BC x y) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_equation_l2708_270816


namespace NUMINAMATH_CALUDE_x_intercept_ratio_l2708_270855

/-- Given two lines with the same non-zero y-intercept, prove that the ratio of their x-intercepts is 1/2 -/
theorem x_intercept_ratio (b s t : ℝ) (hb : b ≠ 0) : 
  (0 = 8 * s + b) → (0 = 4 * t + b) → s / t = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_x_intercept_ratio_l2708_270855


namespace NUMINAMATH_CALUDE_min_product_xyz_l2708_270863

theorem min_product_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hsum : x + y + z = 1) (hz_twice_y : z = 2 * y)
  (hx_le_2y : x ≤ 2 * y) (hy_le_2x : y ≤ 2 * x) (hz_le_2x : z ≤ 2 * x) :
  ∃ (min_val : ℝ), min_val = 8 / 243 ∧ x * y * z ≥ min_val :=
sorry

end NUMINAMATH_CALUDE_min_product_xyz_l2708_270863


namespace NUMINAMATH_CALUDE_total_rainfall_l2708_270851

def rainfall_problem (first_week : ℝ) (second_week : ℝ) : Prop :=
  (second_week = 1.5 * first_week) ∧
  (second_week = 18) ∧
  (first_week + second_week = 30)

theorem total_rainfall : ∃ (first_week second_week : ℝ),
  rainfall_problem first_week second_week :=
by sorry

end NUMINAMATH_CALUDE_total_rainfall_l2708_270851


namespace NUMINAMATH_CALUDE_team_sports_count_l2708_270873

theorem team_sports_count (total_score : ℕ) : ∃ (n : ℕ), 
  (n > 0) ∧ 
  ((97 + total_score) / n = 90) ∧ 
  ((73 + total_score) / n = 87) → 
  n = 8 := by
sorry

end NUMINAMATH_CALUDE_team_sports_count_l2708_270873


namespace NUMINAMATH_CALUDE_three_digit_seven_divisible_by_five_l2708_270827

theorem three_digit_seven_divisible_by_five (N : ℕ) : 
  (100 ≤ N ∧ N ≤ 999) →  -- N is a three-digit number
  (N % 10 = 7) →         -- N has a ones digit of 7
  (N % 5 = 0) →          -- N is divisible by 5
  False :=               -- This is impossible
by sorry

end NUMINAMATH_CALUDE_three_digit_seven_divisible_by_five_l2708_270827


namespace NUMINAMATH_CALUDE_product_of_fractions_l2708_270892

theorem product_of_fractions (a b : ℚ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_l2708_270892


namespace NUMINAMATH_CALUDE_floor_square_minus_floor_product_l2708_270885

theorem floor_square_minus_floor_product (x : ℝ) : x = 12.7 → ⌊x^2⌋ - ⌊x⌋ * ⌊x⌋ = 17 := by
  sorry

end NUMINAMATH_CALUDE_floor_square_minus_floor_product_l2708_270885


namespace NUMINAMATH_CALUDE_equation_solution_l2708_270831

theorem equation_solution : ∃ x : ℝ, 2 * ((x - 1) - (2 * x + 1)) = 6 ∧ x = -5 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l2708_270831


namespace NUMINAMATH_CALUDE_outfit_combinations_l2708_270878

theorem outfit_combinations : 
  let blue_shirts : ℕ := 6
  let green_shirts : ℕ := 4
  let pants : ℕ := 7
  let blue_hats : ℕ := 9
  let green_hats : ℕ := 7
  let blue_shirt_green_hat := blue_shirts * pants * green_hats
  let green_shirt_blue_hat := green_shirts * pants * blue_hats
  blue_shirt_green_hat + green_shirt_blue_hat = 546 := by
  sorry

end NUMINAMATH_CALUDE_outfit_combinations_l2708_270878


namespace NUMINAMATH_CALUDE_first_wheat_rate_calculation_l2708_270834

-- Define the variables and constants
def first_wheat_quantity : ℝ := 30
def second_wheat_quantity : ℝ := 20
def second_wheat_rate : ℝ := 14.25
def profit_percentage : ℝ := 0.10
def selling_rate : ℝ := 13.86

-- Define the theorem
theorem first_wheat_rate_calculation (x : ℝ) : 
  (1 + profit_percentage) * (first_wheat_quantity * x + second_wheat_quantity * second_wheat_rate) = 
  (first_wheat_quantity + second_wheat_quantity) * selling_rate → 
  x = 11.50 := by
sorry

end NUMINAMATH_CALUDE_first_wheat_rate_calculation_l2708_270834


namespace NUMINAMATH_CALUDE_train_delay_l2708_270841

/-- Calculates the time difference in minutes for a train traveling a given distance at two different speeds -/
theorem train_delay (distance : ℝ) (speed1 speed2 : ℝ) :
  distance > 0 ∧ speed1 > 0 ∧ speed2 > 0 ∧ speed1 > speed2 →
  (distance / speed2 - distance / speed1) * 60 = 15 ∧
  distance = 70 ∧ speed1 = 40 ∧ speed2 = 35 :=
by sorry

end NUMINAMATH_CALUDE_train_delay_l2708_270841


namespace NUMINAMATH_CALUDE_k_negative_sufficient_not_necessary_l2708_270894

-- Define the condition for the equation to represent a hyperbola
def is_hyperbola (k : ℝ) : Prop := k * (k - 1) > 0

-- State the theorem
theorem k_negative_sufficient_not_necessary :
  (∀ k : ℝ, k < 0 → is_hyperbola k) ∧
  (∃ k : ℝ, ¬(k < 0) ∧ is_hyperbola k) :=
sorry

end NUMINAMATH_CALUDE_k_negative_sufficient_not_necessary_l2708_270894


namespace NUMINAMATH_CALUDE_dima_wins_l2708_270818

-- Define the game board as a set of integers from 1 to 100
def GameBoard : Set ℕ := {n | 1 ≤ n ∧ n ≤ 100}

-- Define a type for player strategies
def Strategy := GameBoard → ℕ

-- Define the winning condition for Mitya
def MityaWins (a b : ℕ) : Prop := (a + b) % 7 = 0

-- Define the game result
inductive GameResult
| MityaVictory
| DimaVictory

-- Define the game play function
def playGame (mityaStrategy dimaStrategy : Strategy) : GameResult :=
  sorry -- Actual game logic would go here

-- Theorem statement
theorem dima_wins :
  ∃ (dimaStrategy : Strategy),
    ∀ (mityaStrategy : Strategy),
      playGame mityaStrategy dimaStrategy = GameResult.DimaVictory :=
sorry

end NUMINAMATH_CALUDE_dima_wins_l2708_270818


namespace NUMINAMATH_CALUDE_computer_time_theorem_l2708_270828

/-- Calculates the average time per person on a computer given the number of people, 
    number of computers, and working day duration. -/
def averageComputerTime (people : ℕ) (computers : ℕ) (workingHours : ℕ) (workingMinutes : ℕ) : ℕ :=
  let totalMinutes : ℕ := workingHours * 60 + workingMinutes
  let totalComputerTime : ℕ := totalMinutes * computers
  totalComputerTime / people

/-- Theorem stating that given 8 people, 5 computers, and a working day of 2 hours and 32 minutes, 
    the average time each person spends on a computer is 95 minutes. -/
theorem computer_time_theorem :
  averageComputerTime 8 5 2 32 = 95 := by
  sorry

end NUMINAMATH_CALUDE_computer_time_theorem_l2708_270828


namespace NUMINAMATH_CALUDE_carl_index_card_cost_l2708_270804

/-- The cost of index cards for Carl's classes -/
def total_cost (cards_per_student : ℕ) (periods : ℕ) (students_per_class : ℕ) (pack_size : ℕ) (pack_cost : ℚ) : ℚ :=
  let total_cards := cards_per_student * periods * students_per_class
  let packs_needed := (total_cards + pack_size - 1) / pack_size  -- Ceiling division
  packs_needed * pack_cost

/-- Proof that Carl spent $108 on index cards -/
theorem carl_index_card_cost :
  total_cost 10 6 30 50 3 = 108 := by
  sorry

end NUMINAMATH_CALUDE_carl_index_card_cost_l2708_270804


namespace NUMINAMATH_CALUDE_no_valid_formation_l2708_270823

/-- Represents a rectangular formation of musicians. -/
structure Formation where
  rows : ℕ
  musicians_per_row : ℕ

/-- Checks if a formation is valid according to the given conditions. -/
def is_valid_formation (f : Formation) : Prop :=
  f.rows * f.musicians_per_row = 400 ∧
  f.musicians_per_row % 4 = 0 ∧
  10 ≤ f.musicians_per_row ∧
  f.musicians_per_row ≤ 50

/-- Represents the constraint of having a triangle formation for brass section. -/
def has_triangle_brass_formation (f : Formation) : Prop :=
  f.rows ≥ 3 ∧
  ∃ (a b c : ℕ), a < b ∧ b < c ∧ a + b + c = 100 ∧
  a % (f.musicians_per_row / 4) = 0 ∧
  b % (f.musicians_per_row / 4) = 0 ∧
  c % (f.musicians_per_row / 4) = 0

/-- The main theorem stating that no valid formation exists. -/
theorem no_valid_formation :
  ¬∃ (f : Formation), is_valid_formation f ∧ has_triangle_brass_formation f :=
sorry

end NUMINAMATH_CALUDE_no_valid_formation_l2708_270823


namespace NUMINAMATH_CALUDE_product_one_inequality_l2708_270899

theorem product_one_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_prod : a * b * c = 1) :
  1 / (a * (a + 1)) + 1 / (b * (b + 1)) + 1 / (c * (c + 1)) ≥ 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_product_one_inequality_l2708_270899


namespace NUMINAMATH_CALUDE_smallest_screw_count_screw_packs_problem_l2708_270883

theorem smallest_screw_count : ℕ → Prop :=
  fun k => (∃ x y : ℕ, x ≠ y ∧ k = 10 * x ∧ k = 12 * y) ∧
           (∀ m : ℕ, m < k → ¬(∃ a b : ℕ, a ≠ b ∧ m = 10 * a ∧ m = 12 * b))

theorem screw_packs_problem : smallest_screw_count 60 := by
  sorry

end NUMINAMATH_CALUDE_smallest_screw_count_screw_packs_problem_l2708_270883


namespace NUMINAMATH_CALUDE_remainder_of_2_pow_33_mod_9_l2708_270890

theorem remainder_of_2_pow_33_mod_9 : 2^33 % 9 = 8 := by sorry

end NUMINAMATH_CALUDE_remainder_of_2_pow_33_mod_9_l2708_270890


namespace NUMINAMATH_CALUDE_min_sum_given_log_condition_l2708_270880

theorem min_sum_given_log_condition (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  Real.log m / Real.log 3 + Real.log n / Real.log 3 ≥ 4 → m + n ≥ 18 := by
  sorry


end NUMINAMATH_CALUDE_min_sum_given_log_condition_l2708_270880


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l2708_270808

theorem simplify_trig_expression :
  (Real.cos (5 * π / 180))^2 - (Real.sin (5 * π / 180))^2 =
  2 * Real.sin (40 * π / 180) * Real.cos (40 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l2708_270808


namespace NUMINAMATH_CALUDE_total_amount_is_140_problem_solution_l2708_270856

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

end NUMINAMATH_CALUDE_total_amount_is_140_problem_solution_l2708_270856


namespace NUMINAMATH_CALUDE_bottles_per_child_per_day_is_three_l2708_270801

/-- Represents a children's camp with water consumption information -/
structure ChildrenCamp where
  group1 : Nat
  group2 : Nat
  group3 : Nat
  initialCases : Nat
  bottlesPerCase : Nat
  campDuration : Nat
  additionalBottles : Nat

/-- Calculates the number of bottles each child consumes per day -/
def bottlesPerChildPerDay (camp : ChildrenCamp) : Rat :=
  let group4 := (camp.group1 + camp.group2 + camp.group3) / 2
  let totalChildren := camp.group1 + camp.group2 + camp.group3 + group4
  let initialBottles := camp.initialCases * camp.bottlesPerCase
  let totalBottles := initialBottles + camp.additionalBottles
  (totalBottles : Rat) / (totalChildren * camp.campDuration)

/-- Theorem stating that for the given camp configuration, each child consumes 3 bottles per day -/
theorem bottles_per_child_per_day_is_three :
  let camp := ChildrenCamp.mk 14 16 12 13 24 3 255
  bottlesPerChildPerDay camp = 3 := by sorry

end NUMINAMATH_CALUDE_bottles_per_child_per_day_is_three_l2708_270801


namespace NUMINAMATH_CALUDE_stratified_sampling_count_l2708_270802

-- Define the total number of students and their gender distribution
def total_students : ℕ := 60
def female_students : ℕ := 24
def male_students : ℕ := 36

-- Define the number of students to be selected
def selected_students : ℕ := 20

-- Define the number of female and male students to be selected
def selected_female : ℕ := 8
def selected_male : ℕ := 12

-- Theorem statement
theorem stratified_sampling_count :
  (Nat.choose female_students selected_female) * (Nat.choose male_students selected_male) =
  (Nat.choose female_students selected_female) * (Nat.choose male_students selected_male) := by
  sorry

-- Ensure the conditions are met
axiom total_students_sum : female_students + male_students = total_students
axiom selected_students_sum : selected_female + selected_male = selected_students

end NUMINAMATH_CALUDE_stratified_sampling_count_l2708_270802


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_two_l2708_270849

theorem fraction_zero_implies_x_equals_two (x : ℝ) : 
  (|x| - 2) / (x + 2) = 0 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_two_l2708_270849


namespace NUMINAMATH_CALUDE_largest_band_formation_l2708_270830

/-- Represents a band formation --/
structure BandFormation where
  totalMembers : ℕ
  rows : ℕ
  membersPerRow : ℕ

/-- Checks if a band formation is valid according to the problem conditions --/
def isValidFormation (bf : BandFormation) : Prop :=
  bf.totalMembers < 120 ∧
  bf.totalMembers = bf.rows * bf.membersPerRow + 3 ∧
  bf.totalMembers = (bf.rows - 1) * (bf.membersPerRow + 2)

/-- Theorem stating that 231 is the largest number of band members satisfying the conditions --/
theorem largest_band_formation :
  ∀ bf : BandFormation, isValidFormation bf → bf.totalMembers ≤ 231 :=
by
  sorry

#check largest_band_formation

end NUMINAMATH_CALUDE_largest_band_formation_l2708_270830


namespace NUMINAMATH_CALUDE_min_Q_zero_at_two_thirds_l2708_270882

/-- The quadratic form representing the expression to be minimized -/
def Q (k : ℝ) (x y : ℝ) : ℝ :=
  5 * x^2 - 8 * k * x * y + (4 * k^2 + 3) * y^2 - 5 * x - 6 * y + 7

/-- The theorem stating that 2/3 is the value of k that makes the minimum of Q zero -/
theorem min_Q_zero_at_two_thirds :
  (∃ (k : ℝ), ∀ (x y : ℝ), Q k x y ≥ 0 ∧ (∃ (x₀ y₀ : ℝ), Q k x₀ y₀ = 0)) ↔ k = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_min_Q_zero_at_two_thirds_l2708_270882


namespace NUMINAMATH_CALUDE_min_value_of_exponential_sum_l2708_270843

theorem min_value_of_exponential_sum (x y : ℝ) (h : x + 2 * y = 2) :
  ∃ (m : ℝ), m = 6 ∧ ∀ (z : ℝ), 3^x + 9^y ≥ z ∧ (∃ (a b : ℝ), a + 2 * b = 2 ∧ 3^a + 9^b = z) → m ≤ z :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_exponential_sum_l2708_270843


namespace NUMINAMATH_CALUDE_point_on_line_with_vector_relation_l2708_270895

/-- Given points A and B, if point P is on line AB and vector AB is twice vector AP, 
    then P has specific coordinates -/
theorem point_on_line_with_vector_relation (A B P : ℝ × ℝ) : 
  A = (2, 0) → 
  B = (4, 2) → 
  (∃ t : ℝ, P = (1 - t) • A + t • B) →  -- P is on line AB
  B - A = 2 • (P - A) → 
  P = (3, 1) := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_with_vector_relation_l2708_270895


namespace NUMINAMATH_CALUDE_square_plus_one_gt_x_l2708_270811

theorem square_plus_one_gt_x : ∀ x : ℝ, x^2 + 1 > x := by sorry

end NUMINAMATH_CALUDE_square_plus_one_gt_x_l2708_270811


namespace NUMINAMATH_CALUDE_freelancer_earnings_l2708_270800

theorem freelancer_earnings (x : ℝ) : 
  x + (50 + 2*x) + 4*(x + (50 + 2*x)) = 5500 → x = 5300/15 := by
  sorry

end NUMINAMATH_CALUDE_freelancer_earnings_l2708_270800


namespace NUMINAMATH_CALUDE_remaining_surface_area_after_removal_l2708_270875

/-- The remaining surface area of a cube after removing a smaller cube from its corner --/
theorem remaining_surface_area_after_removal (a b : ℝ) (ha : a > 0) (hb : b > 0) (hba : b < 3*a) :
  6 * (3*a)^2 - 3 * b^2 + 3 * b^2 = 54 * a^2 := by
  sorry

#check remaining_surface_area_after_removal

end NUMINAMATH_CALUDE_remaining_surface_area_after_removal_l2708_270875


namespace NUMINAMATH_CALUDE_distribute_five_into_four_l2708_270821

def distribute_objects (n : ℕ) (k : ℕ) : ℕ :=
  if n < k then 0
  else if k = 1 then 1
  else k

theorem distribute_five_into_four :
  distribute_objects 5 4 = 4 := by
  sorry

end NUMINAMATH_CALUDE_distribute_five_into_four_l2708_270821


namespace NUMINAMATH_CALUDE_piggy_bank_pennies_l2708_270809

theorem piggy_bank_pennies (initial_pennies : ℕ) : 
  (12 * (initial_pennies + 6) = 96) → initial_pennies = 2 := by
  sorry

end NUMINAMATH_CALUDE_piggy_bank_pennies_l2708_270809


namespace NUMINAMATH_CALUDE_manufacturing_cost_of_shoe_l2708_270854

/-- The manufacturing cost of a shoe given specific conditions -/
theorem manufacturing_cost_of_shoe (transportation_cost : ℚ) (selling_price : ℚ) (gain_percentage : ℚ) :
  transportation_cost = 500 / 100 →
  selling_price = 270 →
  gain_percentage = 20 / 100 →
  ∃ (manufacturing_cost : ℚ),
    manufacturing_cost = selling_price / (1 + gain_percentage) - transportation_cost ∧
    manufacturing_cost = 220 := by
  sorry

end NUMINAMATH_CALUDE_manufacturing_cost_of_shoe_l2708_270854


namespace NUMINAMATH_CALUDE_negation_of_proposition_negation_of_specific_proposition_l2708_270805

theorem negation_of_proposition (P : ℝ → Prop) :
  (¬ ∀ x : ℝ, P x) ↔ (∃ x : ℝ, ¬ P x) :=
by sorry

theorem negation_of_specific_proposition :
  (¬ ∀ x : ℝ, x^2 + x - 1 < 0) ↔ (∃ x : ℝ, x^2 + x - 1 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_negation_of_specific_proposition_l2708_270805


namespace NUMINAMATH_CALUDE_sum_of_digits_l2708_270877

def is_single_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def number_representation (A B : ℕ) : ℕ := A * 100000 + 44610 + B

theorem sum_of_digits (A B : ℕ) : 
  is_single_digit A → 
  is_single_digit B → 
  (number_representation A B) % 72 = 0 → 
  A + B = 12 := by
sorry

end NUMINAMATH_CALUDE_sum_of_digits_l2708_270877


namespace NUMINAMATH_CALUDE_problem_solution_l2708_270838

theorem problem_solution (p q r : ℝ) : 
  (∀ x : ℝ, (x - p) * (x - q) / (x - r) ≤ 0 ↔ (x < -6 ∨ |x - 30| ≤ 2)) →
  p < q →
  p + 2*q + 3*r = 74 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2708_270838


namespace NUMINAMATH_CALUDE_manuscript_cost_example_l2708_270876

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

end NUMINAMATH_CALUDE_manuscript_cost_example_l2708_270876


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2708_270813

theorem sufficient_not_necessary_condition :
  (∀ x : ℝ, x > 1 → x - 1/x > 0) ∧
  (∃ x : ℝ, x - 1/x > 0 ∧ x ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2708_270813


namespace NUMINAMATH_CALUDE_positive_real_as_infinite_sum_representations_l2708_270868

theorem positive_real_as_infinite_sum_representations (k : ℝ) (hk : k > 0) :
  ∃ (f : ℕ → (ℕ → ℕ)), 
    (∀ n : ℕ, ∀ i j : ℕ, i < j → f n i < f n j) ∧ 
    (∀ n : ℕ, k = ∑' i, (1 : ℝ) / (10 ^ (f n i))) :=
sorry

end NUMINAMATH_CALUDE_positive_real_as_infinite_sum_representations_l2708_270868


namespace NUMINAMATH_CALUDE_tangent_parallel_points_main_theorem_l2708_270829

/-- The function f(x) = x³ + x - 2 --/
def f (x : ℝ) : ℝ := x^3 + x - 2

/-- The derivative of f(x) --/
def f' (x : ℝ) : ℝ := 3*x^2 + 1

theorem tangent_parallel_points :
  {x : ℝ | f' x = 4} = {1, -1} :=
sorry

theorem main_theorem :
  {p : ℝ × ℝ | p.1 ∈ {x : ℝ | f' x = 4} ∧ p.2 = f p.1} = {(1, 0), (-1, -4)} :=
sorry

end NUMINAMATH_CALUDE_tangent_parallel_points_main_theorem_l2708_270829


namespace NUMINAMATH_CALUDE_nth_equation_proof_l2708_270814

theorem nth_equation_proof (n : ℕ) (hn : n > 0) : 
  (1 : ℚ) / n * ((n^2 + 2*n) / (n + 1)) - 1 / (n + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_nth_equation_proof_l2708_270814

import Mathlib

namespace NUMINAMATH_CALUDE_banknote_replacement_theorem_l1483_148356

/-- Represents the state of the banknote replacement process -/
structure BanknoteState where
  total_banknotes : ℕ
  remaining_banknotes : ℕ
  budget : ℕ
  days : ℕ

/-- Calculates the number of banknotes that can be replaced on a given day -/
def replace_banknotes (state : BanknoteState) (day : ℕ) : ℕ :=
  min state.remaining_banknotes (state.remaining_banknotes / (day + 1))

/-- Updates the state after a day of replacement -/
def update_state (state : BanknoteState) (day : ℕ) : BanknoteState :=
  let replaced := replace_banknotes state day
  { state with
    remaining_banknotes := state.remaining_banknotes - replaced
    budget := state.budget - 90000
    days := state.days + 1 }

/-- Checks if the budget is exceeded -/
def budget_exceeded (state : BanknoteState) : Prop :=
  state.budget < 0

/-- Checks if 80% of banknotes have been replaced -/
def eighty_percent_replaced (state : BanknoteState) : Prop :=
  state.remaining_banknotes ≤ state.total_banknotes / 5

/-- Main theorem statement -/
theorem banknote_replacement_theorem (initial_state : BanknoteState)
    (h_total : initial_state.total_banknotes = 3628800)
    (h_budget : initial_state.budget = 1000000) :
    ∃ (final_state : BanknoteState),
      final_state.days ≥ 4 ∧
      eighty_percent_replaced final_state ∧
      ¬∃ (complete_state : BanknoteState),
        complete_state.remaining_banknotes = 0 ∧
        ¬budget_exceeded complete_state :=
  sorry


end NUMINAMATH_CALUDE_banknote_replacement_theorem_l1483_148356


namespace NUMINAMATH_CALUDE_ellipse_line_intersection_l1483_148361

/-- An ellipse with given properties and a line intersecting it -/
structure EllipseWithLine where
  a : ℝ
  b : ℝ
  k : ℝ
  h_a_pos : 0 < a
  h_b_pos : 0 < b
  h_a_gt_b : b < a
  h_right_focus : (1 : ℝ) = a * (a^2 - b^2).sqrt / a
  h_eccentricity : (a^2 - b^2).sqrt / a = 1/2
  h_ellipse_eq : ∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 ↔ (x, y) ∈ {p : ℝ × ℝ | p.1^2/a^2 + p.2^2/b^2 = 1}
  h_line_eq : ∀ x : ℝ, (x, k*x + 1) ∈ {p : ℝ × ℝ | p.1^2/a^2 + p.2^2/b^2 = 1}
  h_intersect : ∃ A B : ℝ × ℝ, A ∈ {p : ℝ × ℝ | p.1^2/a^2 + p.2^2/b^2 = 1} ∧ 
                                B ∈ {p : ℝ × ℝ | p.1^2/a^2 + p.2^2/b^2 = 1} ∧
                                A.2 = k * A.1 + 1 ∧ B.2 = k * B.1 + 1 ∧ A ≠ B
  h_midpoints : ∀ A B : ℝ × ℝ, A ∈ {p : ℝ × ℝ | p.1^2/a^2 + p.2^2/b^2 = 1} ∧ 
                                B ∈ {p : ℝ × ℝ | p.1^2/a^2 + p.2^2/b^2 = 1} ∧
                                A.2 = k * A.1 + 1 ∧ B.2 = k * B.1 + 1 ∧ A ≠ B →
                ∃ M N : ℝ × ℝ, M = ((A.1 + 1)/2, A.2/2) ∧ N = ((B.1 + 1)/2, B.2/2)
  h_origin_on_circle : ∀ M N : ℝ × ℝ, M.1 * N.1 + M.2 * N.2 = 0

/-- The main theorem: given the ellipse and line with specified properties, k = -1/2 -/
theorem ellipse_line_intersection (e : EllipseWithLine) : e.k = -1/2 :=
sorry

end NUMINAMATH_CALUDE_ellipse_line_intersection_l1483_148361


namespace NUMINAMATH_CALUDE_inequality_proof_l1483_148382

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a^2018 + b^2018)^2019 > (a^2019 + b^2019)^2018 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1483_148382


namespace NUMINAMATH_CALUDE_division_remainder_problem_l1483_148366

theorem division_remainder_problem (a b r : ℕ) : 
  a - b = 2500 → 
  a = 2982 → 
  ∃ q, a = q * b + r ∧ q = 6 ∧ r < b → 
  r = 90 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l1483_148366


namespace NUMINAMATH_CALUDE_f_monotonicity_and_g_zeros_l1483_148390

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (x^3 - a*x)

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x - x^2

theorem f_monotonicity_and_g_zeros (a : ℝ) :
  (a ≤ 0 → ∀ x y : ℝ, x < y → f a x < f a y) ∧
  (a > 0 → ∀ x y : ℝ, 
    ((x < y ∧ y < -Real.sqrt (a/3)) ∨ (x > Real.sqrt (a/3) ∧ y > x)) → f a x < f a y) ∧
  (a > 0 → ∀ x y : ℝ, 
    (-Real.sqrt (a/3) < x ∧ x < y ∧ y < Real.sqrt (a/3)) → f a x > f a y) ∧
  (∃ x y : ℝ, x < y ∧ g a x = 0 ∧ g a y = 0 ∧ ∀ z : ℝ, z ≠ x ∧ z ≠ y → g a z ≠ 0) →
  a > 1 :=
sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_g_zeros_l1483_148390


namespace NUMINAMATH_CALUDE_round_trip_distance_l1483_148312

/-- Calculates the total distance of a round trip given the times for each direction and the average speed -/
theorem round_trip_distance 
  (time_to : ℝ) 
  (time_from : ℝ) 
  (avg_speed : ℝ) 
  (h1 : time_to > 0) 
  (h2 : time_from > 0) 
  (h3 : avg_speed > 0) : 
  ∃ (distance : ℝ), distance = avg_speed * (time_to + time_from) / 60 := by
  sorry

#check round_trip_distance

end NUMINAMATH_CALUDE_round_trip_distance_l1483_148312


namespace NUMINAMATH_CALUDE_linear_coefficient_of_quadratic_l1483_148307

/-- Given a quadratic equation equivalent to 5x - 2 = 3x^2, 
    prove that the coefficient of the linear term is -5 -/
theorem linear_coefficient_of_quadratic (a b c : ℝ) : 
  (5 : ℝ) * x - 2 = 3 * x^2 → 
  a * x^2 + b * x + c = 0 → 
  c = 2 →
  b = -5 := by
  sorry

#check linear_coefficient_of_quadratic

end NUMINAMATH_CALUDE_linear_coefficient_of_quadratic_l1483_148307


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l1483_148322

theorem smallest_n_congruence : 
  ∃ (n : ℕ), n > 0 ∧ 5 * n ≡ 105 [MOD 24] ∧ 
  ∀ (m : ℕ), m > 0 ∧ 5 * m ≡ 105 [MOD 24] → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l1483_148322


namespace NUMINAMATH_CALUDE_power_multiplication_l1483_148347

theorem power_multiplication (a : ℝ) : a ^ 2 * a ^ 5 = a ^ 7 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l1483_148347


namespace NUMINAMATH_CALUDE_complex_expression_eighth_root_of_unity_l1483_148348

theorem complex_expression_eighth_root_of_unity :
  let z := (Complex.tan (Real.pi / 4) + Complex.I) / (Complex.tan (Real.pi / 4) - Complex.I)
  z = Complex.I ∧
  z^8 = 1 ∧
  ∃ n : ℕ, n = 2 ∧ z = Complex.exp (Complex.I * (2 * ↑n * Real.pi / 8)) := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_eighth_root_of_unity_l1483_148348


namespace NUMINAMATH_CALUDE_geometric_sequence_a4_value_l1483_148321

/-- A geometric sequence of real numbers. -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) * a (n - 1) = a n ^ 2

/-- Given a geometric sequence satisfying certain conditions, a_4 equals 8. -/
theorem geometric_sequence_a4_value (a : ℕ → ℝ) 
    (h_geom : geometric_sequence a)
    (h_sum : a 2 + a 6 = 34)
    (h_prod : a 3 * a 5 = 64) : 
  a 4 = 8 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a4_value_l1483_148321


namespace NUMINAMATH_CALUDE_existence_of_point_l1483_148333

theorem existence_of_point :
  ∃ (x₀ y₀ z₀ : ℝ),
    (x₀ + y₀ + z₀ ≠ 0) ∧
    (0 < x₀^2 + y₀^2 + z₀^2) ∧
    (x₀^2 + y₀^2 + z₀^2 < 1 / 1999) ∧
    (1.999 < (x₀^2 + y₀^2 + z₀^2) / (x₀ + y₀ + z₀)) ∧
    ((x₀^2 + y₀^2 + z₀^2) / (x₀ + y₀ + z₀) < 2) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_point_l1483_148333


namespace NUMINAMATH_CALUDE_sara_lunch_cost_theorem_l1483_148364

/-- The cost of Sara's lunch given the prices of a hotdog and a salad -/
def lunch_cost (hotdog_price salad_price : ℚ) : ℚ :=
  hotdog_price + salad_price

/-- Theorem stating that Sara's lunch cost is the sum of hotdog and salad prices -/
theorem sara_lunch_cost_theorem (hotdog_price salad_price : ℚ) :
  lunch_cost hotdog_price salad_price = hotdog_price + salad_price :=
by sorry

end NUMINAMATH_CALUDE_sara_lunch_cost_theorem_l1483_148364


namespace NUMINAMATH_CALUDE_p_q_contradictory_l1483_148334

-- Define proposition p
def p : Prop := ∀ a : ℝ, a > 0 → a^2 ≠ 0

-- Define proposition q
def q : Prop := ∀ a : ℝ, a ≤ 0 → a^2 = 0

-- Theorem stating that p and q are contradictory
theorem p_q_contradictory : p ↔ ¬q := by
  sorry


end NUMINAMATH_CALUDE_p_q_contradictory_l1483_148334


namespace NUMINAMATH_CALUDE_counterexamples_exist_l1483_148349

def is_counterexample (n : ℕ) : Prop :=
  ¬(Nat.Prime n) ∧ ¬(Nat.Prime (n - 3))

theorem counterexamples_exist : 
  is_counterexample 18 ∧ is_counterexample 24 :=
by sorry

end NUMINAMATH_CALUDE_counterexamples_exist_l1483_148349


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1483_148330

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h : arithmetic_sequence a) (h1 : a 3 + a 8 = 10) :
  3 * a 5 + a 7 = 20 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1483_148330


namespace NUMINAMATH_CALUDE_perfect_square_quadratic_l1483_148303

theorem perfect_square_quadratic (m : ℝ) : 
  (∃ k : ℝ, ∀ x : ℝ, x^2 - (m+1)*x + 1 = k^2) → (m = 1 ∨ m = -3) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_quadratic_l1483_148303


namespace NUMINAMATH_CALUDE_new_speed_calculation_l1483_148379

/-- Theorem: Given a distance of 630 km and an original time of 6 hours,
    if the new time is 3/2 times the original time,
    then the new speed required to cover the same distance is 70 km/h. -/
theorem new_speed_calculation (distance : ℝ) (original_time : ℝ) (new_time_factor : ℝ) :
  distance = 630 →
  original_time = 6 →
  new_time_factor = 3 / 2 →
  let new_time := original_time * new_time_factor
  let new_speed := distance / new_time
  new_speed = 70 := by
  sorry

#check new_speed_calculation

end NUMINAMATH_CALUDE_new_speed_calculation_l1483_148379


namespace NUMINAMATH_CALUDE_gold_bar_value_proof_l1483_148326

/-- The value of one bar of gold -/
def gold_bar_value : ℝ := 2200

/-- The number of gold bars Legacy has -/
def legacy_bars : ℕ := 5

/-- The number of gold bars Aleena has -/
def aleena_bars : ℕ := legacy_bars - 2

/-- The total value of gold Legacy and Aleena have together -/
def total_value : ℝ := 17600

theorem gold_bar_value_proof : 
  gold_bar_value * (legacy_bars + aleena_bars : ℝ) = total_value := by
  sorry

end NUMINAMATH_CALUDE_gold_bar_value_proof_l1483_148326


namespace NUMINAMATH_CALUDE_intersection_M_N_l1483_148368

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x : ℝ | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1483_148368


namespace NUMINAMATH_CALUDE_parabola_transformation_l1483_148346

/-- Represents a parabola in the form y = (x + a)² + b -/
structure Parabola where
  a : ℝ
  b : ℝ

/-- Applies a horizontal shift to a parabola -/
def horizontal_shift (p : Parabola) (shift : ℝ) : Parabola :=
  { a := p.a - shift, b := p.b }

/-- Applies a vertical shift to a parabola -/
def vertical_shift (p : Parabola) (shift : ℝ) : Parabola :=
  { a := p.a, b := p.b + shift }

/-- The theorem to be proved -/
theorem parabola_transformation (p : Parabola) :
  p.a = 2 ∧ p.b = 3 →
  (vertical_shift (horizontal_shift p 3) (-2)) = { a := -1, b := 1 } := by
  sorry

end NUMINAMATH_CALUDE_parabola_transformation_l1483_148346


namespace NUMINAMATH_CALUDE_sin_cos_fourth_power_sum_l1483_148340

theorem sin_cos_fourth_power_sum (θ : Real) (h : Real.cos (2 * θ) = 1/3) :
  Real.sin θ ^ 4 + Real.cos θ ^ 4 = 5/9 := by
sorry

end NUMINAMATH_CALUDE_sin_cos_fourth_power_sum_l1483_148340


namespace NUMINAMATH_CALUDE_sin_two_theta_value_l1483_148367

/-- If e^(2iθ) = (2 + i√5) / 3, then sin 2θ = √3 / 3 -/
theorem sin_two_theta_value (θ : ℝ) (h : Complex.exp (2 * θ * Complex.I) = (2 + Complex.I * Real.sqrt 5) / 3) : 
  Real.sin (2 * θ) = Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_sin_two_theta_value_l1483_148367


namespace NUMINAMATH_CALUDE_nail_count_proof_l1483_148389

/-- The number of nails Violet has -/
def violet_nails : ℕ := 27

/-- The number of nails Tickletoe has -/
def tickletoe_nails : ℕ := (violet_nails - 3) / 2

/-- The number of nails SillySocks has -/
def sillysocks_nails : ℕ := 3 * tickletoe_nails - 2

/-- The total number of nails -/
def total_nails : ℕ := violet_nails + tickletoe_nails + sillysocks_nails

theorem nail_count_proof :
  total_nails = 73 :=
sorry

end NUMINAMATH_CALUDE_nail_count_proof_l1483_148389


namespace NUMINAMATH_CALUDE_cube_with_hole_volume_is_384_l1483_148342

/-- The volume of a cube with a square hole cut through its center. -/
def cube_with_hole_volume (cube_side : ℝ) (hole_side : ℝ) : ℝ :=
  cube_side ^ 3 - hole_side ^ 2 * cube_side

/-- Theorem stating that a cube with side length 8 cm and a square hole
    with side length 4 cm cut through its center has a volume of 384 cm³. -/
theorem cube_with_hole_volume_is_384 :
  cube_with_hole_volume 8 4 = 384 := by
  sorry

#eval cube_with_hole_volume 8 4

end NUMINAMATH_CALUDE_cube_with_hole_volume_is_384_l1483_148342


namespace NUMINAMATH_CALUDE_higher_probability_white_piece_l1483_148396

theorem higher_probability_white_piece (white_count black_count : ℕ) 
  (hw : white_count = 10) (hb : black_count = 2) : 
  (white_count : ℚ) / (white_count + black_count) > (black_count : ℚ) / (white_count + black_count) := by
  sorry

end NUMINAMATH_CALUDE_higher_probability_white_piece_l1483_148396


namespace NUMINAMATH_CALUDE_abc_product_range_l1483_148373

noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 9 then |Real.log x / Real.log 3 - 1|
  else if x > 9 then 4 - Real.sqrt x
  else 0

theorem abc_product_range (a b c : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  f a = f b ∧ f b = f c →
  81 < a * b * c ∧ a * b * c < 144 := by sorry

end NUMINAMATH_CALUDE_abc_product_range_l1483_148373


namespace NUMINAMATH_CALUDE_inequality_satisfied_l1483_148395

theorem inequality_satisfied (a b c : ℤ) : 
  a = 1 ∧ b = 2 ∧ c = 1 → a^2 + b^2 + c^2 + 3 < a*b + 3*b + 2*c := by sorry

end NUMINAMATH_CALUDE_inequality_satisfied_l1483_148395


namespace NUMINAMATH_CALUDE_roots_sum_of_reciprocal_squares_l1483_148359

theorem roots_sum_of_reciprocal_squares (r s : ℂ) : 
  (3 * r^2 - 2 * r + 4 = 0) → 
  (3 * s^2 - 2 * s + 4 = 0) → 
  (1 / r^2 + 1 / s^2 = -5 / 4) := by
sorry

end NUMINAMATH_CALUDE_roots_sum_of_reciprocal_squares_l1483_148359


namespace NUMINAMATH_CALUDE_polynomial_interpolation_l1483_148372

/-- A polynomial of degree 3 with real coefficients -/
def Polynomial3 := ℝ → ℝ

/-- The statement that a function is a polynomial of degree 3 -/
def IsPoly3 (f : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, ∀ x, f x = a * x^3 + b * x^2 + c * x + d

theorem polynomial_interpolation
  (P Q R : Polynomial3)
  (hP : IsPoly3 P)
  (hQ : IsPoly3 Q)
  (hR : IsPoly3 R)
  (h_le : ∀ x, P x ≤ Q x ∧ Q x ≤ R x)
  (h_eq : ∃ x₀, P x₀ = R x₀) :
  ∃ k : ℝ, 0 ≤ k ∧ k ≤ 1 ∧ ∀ x, Q x = k * P x + (1 - k) * R x :=
by sorry

end NUMINAMATH_CALUDE_polynomial_interpolation_l1483_148372


namespace NUMINAMATH_CALUDE_f_2013_value_l1483_148343

-- Define the properties of function f
def is_valid_f (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧  -- f is odd
  (∀ x, f (x + 3) = -f (1 - x)) ∧  -- given functional equation
  f 3 = 2  -- given value

-- State the theorem
theorem f_2013_value (f : ℝ → ℝ) (h : is_valid_f f) : f 2013 = -2 := by
  sorry

end NUMINAMATH_CALUDE_f_2013_value_l1483_148343


namespace NUMINAMATH_CALUDE_adam_book_spending_l1483_148363

theorem adam_book_spending :
  ∀ (initial_amount spent_amount : ℝ),
    initial_amount = 91 →
    (initial_amount - spent_amount) / spent_amount = 10 / 3 →
    spent_amount = 21 := by
  sorry

end NUMINAMATH_CALUDE_adam_book_spending_l1483_148363


namespace NUMINAMATH_CALUDE_ripe_oranges_per_day_l1483_148381

theorem ripe_oranges_per_day :
  ∀ (daily_ripe_oranges : ℕ),
    daily_ripe_oranges * 73 = 365 →
    daily_ripe_oranges = 5 := by
  sorry

end NUMINAMATH_CALUDE_ripe_oranges_per_day_l1483_148381


namespace NUMINAMATH_CALUDE_smallest_sixth_power_sum_equality_holds_l1483_148391

theorem smallest_sixth_power_sum (n : ℕ) : n > 150 ∧ 135^6 + 115^6 + 85^6 + 30^6 = n^6 → n ≥ 165 := by
  sorry

theorem equality_holds : 135^6 + 115^6 + 85^6 + 30^6 = 165^6 := by
  sorry

end NUMINAMATH_CALUDE_smallest_sixth_power_sum_equality_holds_l1483_148391


namespace NUMINAMATH_CALUDE_initial_milk_water_ratio_l1483_148370

theorem initial_milk_water_ratio 
  (total_initial_volume : ℝ)
  (additional_water : ℝ)
  (final_ratio : ℝ)
  (h1 : total_initial_volume = 45)
  (h2 : additional_water = 23)
  (h3 : final_ratio = 1.125)
  : ∃ (initial_milk initial_water : ℝ),
    initial_milk + initial_water = total_initial_volume ∧
    initial_milk / (initial_water + additional_water) = final_ratio ∧
    initial_milk / initial_water = 4 := by
  sorry

end NUMINAMATH_CALUDE_initial_milk_water_ratio_l1483_148370


namespace NUMINAMATH_CALUDE_chocolate_syrup_usage_l1483_148306

/-- The number of ounces of chocolate syrup used in each shake -/
def syrup_per_shake : ℝ := 4

/-- The number of ounces of chocolate syrup used on each cone -/
def syrup_per_cone : ℝ := 6

/-- The number of shakes sold -/
def num_shakes : ℕ := 2

/-- The number of cones sold -/
def num_cones : ℕ := 1

/-- The total number of ounces of chocolate syrup used -/
def total_syrup : ℝ := 14

theorem chocolate_syrup_usage :
  syrup_per_shake * num_shakes + syrup_per_cone * num_cones = total_syrup :=
by sorry

end NUMINAMATH_CALUDE_chocolate_syrup_usage_l1483_148306


namespace NUMINAMATH_CALUDE_square_side_length_l1483_148318

/-- Given a rectangle with sides 7 cm and 5 cm, and a square with the same perimeter as the rectangle,
    the length of one side of the square is 6 cm. -/
theorem square_side_length (rectangle_length : ℝ) (rectangle_width : ℝ) (square_side : ℝ) : 
  rectangle_length = 7 → 
  rectangle_width = 5 → 
  4 * square_side = 2 * (rectangle_length + rectangle_width) → 
  square_side = 6 := by
sorry

end NUMINAMATH_CALUDE_square_side_length_l1483_148318


namespace NUMINAMATH_CALUDE_trajectory_of_M_line_l_equation_when_OP_eq_OM_area_POM_when_OP_eq_OM_l1483_148329

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 8*y = 0

-- Define point P
def point_P : ℝ × ℝ := (2, 2)

-- Define the line l passing through P and intersecting C
def line_l (m : ℝ) (x y : ℝ) : Prop := y - 2 = m * (x - 2)

-- Define the midpoint M of AB
def point_M (x y : ℝ) : Prop := ∃ (x_a y_a x_b y_b : ℝ),
  circle_C x_a y_a ∧ circle_C x_b y_b ∧
  line_l ((y_b - y_a) / (x_b - x_a)) x_a y_a ∧
  line_l ((y_b - y_a) / (x_b - x_a)) x_b y_b ∧
  x = (x_a + x_b) / 2 ∧ y = (y_a + y_b) / 2

-- Theorem 1: Trajectory of M
theorem trajectory_of_M :
  ∀ (x y : ℝ), point_M x y → (x - 1)^2 + (y - 3)^2 = 2 :=
sorry

-- Theorem 2a: Equation of line l when |OP| = |OM|
theorem line_l_equation_when_OP_eq_OM :
  ∀ (x y : ℝ), point_M x y → (x^2 + y^2 = x^2 + (y - 3)^2 + 5) →
  (x + 3*y - 8 = 0) :=
sorry

-- Theorem 2b: Area of triangle POM when |OP| = |OM|
theorem area_POM_when_OP_eq_OM :
  ∀ (x y : ℝ), point_M x y → (x^2 + y^2 = x^2 + (y - 3)^2 + 5) →
  (1/2 * |x - 2| * |y - 2| = 16/5) :=
sorry

end NUMINAMATH_CALUDE_trajectory_of_M_line_l_equation_when_OP_eq_OM_area_POM_when_OP_eq_OM_l1483_148329


namespace NUMINAMATH_CALUDE_largest_number_with_equal_quotient_and_remainder_l1483_148308

theorem largest_number_with_equal_quotient_and_remainder : ∀ A B C : ℕ,
  A = 8 * B + C →
  B = C →
  C < 8 →
  A ≤ 63 ∧ ∃ A₀ : ℕ, A₀ = 63 ∧ ∃ B₀ C₀ : ℕ, A₀ = 8 * B₀ + C₀ ∧ B₀ = C₀ ∧ C₀ < 8 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_number_with_equal_quotient_and_remainder_l1483_148308


namespace NUMINAMATH_CALUDE_cat_teeth_count_l1483_148371

theorem cat_teeth_count (dog_teeth : ℕ) (pig_teeth : ℕ) (num_dogs : ℕ) (num_cats : ℕ) (num_pigs : ℕ) (total_teeth : ℕ) :
  dog_teeth = 42 →
  pig_teeth = 28 →
  num_dogs = 5 →
  num_cats = 10 →
  num_pigs = 7 →
  total_teeth = 706 →
  (total_teeth - num_dogs * dog_teeth - num_pigs * pig_teeth) / num_cats = 30 := by
sorry

end NUMINAMATH_CALUDE_cat_teeth_count_l1483_148371


namespace NUMINAMATH_CALUDE_num_ways_eq_1716_l1483_148315

/-- The number of distinct ways to choose 8 non-negative integers that sum to 6 -/
def num_ways : ℕ := Nat.choose 13 7

theorem num_ways_eq_1716 : num_ways = 1716 := by sorry

end NUMINAMATH_CALUDE_num_ways_eq_1716_l1483_148315


namespace NUMINAMATH_CALUDE_fourth_intersection_point_l1483_148374

def curve (x y : ℝ) : Prop := x * y = 2

theorem fourth_intersection_point (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ) 
  (h₁ : curve x₁ y₁) (h₂ : curve x₂ y₂) (h₃ : curve x₃ y₃) (h₄ : curve x₄ y₄)
  (p₁ : x₁ = 4 ∧ y₁ = 1/2) (p₂ : x₂ = -2 ∧ y₂ = -1) (p₃ : x₃ = 1/4 ∧ y₃ = 8)
  (distinct : x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄) :
  x₄ = 1 ∧ y₄ = 2 := by
  sorry

end NUMINAMATH_CALUDE_fourth_intersection_point_l1483_148374


namespace NUMINAMATH_CALUDE_reciprocal_sum_identity_l1483_148380

theorem reciprocal_sum_identity (x y z : ℝ) (h : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0) :
  1 / x + 1 / y = 1 / z → z = x * y / (y + x) := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_sum_identity_l1483_148380


namespace NUMINAMATH_CALUDE_minimum_male_students_l1483_148335

theorem minimum_male_students (num_benches : ℕ) (students_per_bench : ℕ) :
  num_benches = 29 →
  students_per_bench = 5 →
  ∃ (male_students : ℕ),
    male_students ≥ 29 ∧
    male_students * 5 ≥ num_benches * students_per_bench ∧
    ∀ m : ℕ, m < 29 → m * 5 < num_benches * students_per_bench :=
by sorry

end NUMINAMATH_CALUDE_minimum_male_students_l1483_148335


namespace NUMINAMATH_CALUDE_clock_twelve_strikes_l1483_148338

/-- Represents a grandfather clock with a given strike interval -/
structure GrandfatherClock where
  strike_interval : ℝ

/-- Calculates the time taken for a given number of strikes -/
def time_for_strikes (clock : GrandfatherClock) (num_strikes : ℕ) : ℝ :=
  clock.strike_interval * (num_strikes - 1)

theorem clock_twelve_strikes (clock : GrandfatherClock) 
  (h : time_for_strikes clock 6 = 30) :
  time_for_strikes clock 12 = 66 := by
  sorry


end NUMINAMATH_CALUDE_clock_twelve_strikes_l1483_148338


namespace NUMINAMATH_CALUDE_union_when_m_neg_two_intersection_equals_B_iff_l1483_148341

-- Define sets A and B
def A : Set ℝ := {x | x^2 - x - 12 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | 2*m - 1 < x ∧ x < 1 + m}

-- Part 1
theorem union_when_m_neg_two : 
  A ∪ B (-2) = {x : ℝ | -5 < x ∧ x ≤ 4} := by sorry

-- Part 2
theorem intersection_equals_B_iff : 
  ∀ m : ℝ, A ∩ B m = B m ↔ m ≥ -1 := by sorry

end NUMINAMATH_CALUDE_union_when_m_neg_two_intersection_equals_B_iff_l1483_148341


namespace NUMINAMATH_CALUDE_football_team_progress_l1483_148393

/-- Given a football team's yard changes, calculate their progress -/
def teamProgress (lost : ℤ) (gained : ℤ) : ℤ :=
  gained - lost

theorem football_team_progress :
  let lost : ℤ := 5
  let gained : ℤ := 7
  teamProgress lost gained = 2 := by
  sorry

end NUMINAMATH_CALUDE_football_team_progress_l1483_148393


namespace NUMINAMATH_CALUDE_goat_difference_l1483_148305

theorem goat_difference (adam_goats andrew_goats ahmed_goats : ℕ) : 
  adam_goats = 7 →
  ahmed_goats = 13 →
  andrew_goats = ahmed_goats + 6 →
  andrew_goats - 2 * adam_goats = 5 := by
sorry

end NUMINAMATH_CALUDE_goat_difference_l1483_148305


namespace NUMINAMATH_CALUDE_expression_simplification_l1483_148386

theorem expression_simplification : 120 * (120 - 12) - (120 * 120 - 12) = -1428 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1483_148386


namespace NUMINAMATH_CALUDE_symmedian_point_is_centroid_of_projections_l1483_148314

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a triangle in 2D space -/
structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)

/-- Projects a point onto a line segment -/
def projectOntoSegment (P : Point) (A : Point) (B : Point) : Point :=
  sorry

/-- Calculates the centroid of a triangle -/
def centroid (T : Triangle) : Point :=
  sorry

/-- Determines if a point is inside a triangle -/
def isInside (P : Point) (T : Triangle) : Prop :=
  sorry

/-- Calculates the Symmedian Point of a triangle -/
def symmedianPoint (T : Triangle) : Point :=
  sorry

/-- Main theorem: The Symmedian Point is the unique point inside the triangle
    that is the centroid of its projections -/
theorem symmedian_point_is_centroid_of_projections (T : Triangle) :
  let S := symmedianPoint T
  isInside S T ∧
  ∀ P, isInside P T →
    (S = P ↔
      let X := projectOntoSegment P T.B T.C
      let Y := projectOntoSegment P T.C T.A
      let Z := projectOntoSegment P T.A T.B
      P = centroid ⟨X, Y, Z⟩) :=
  sorry

end NUMINAMATH_CALUDE_symmedian_point_is_centroid_of_projections_l1483_148314


namespace NUMINAMATH_CALUDE_pins_purchased_proof_l1483_148398

/-- Calculates the number of pins purchased given the original price, discount percentage, and total amount spent. -/
def calculate_pins_purchased (original_price : ℚ) (discount_percent : ℚ) (total_spent : ℚ) : ℚ :=
  let discounted_price := original_price * (1 - discount_percent / 100)
  total_spent / discounted_price

/-- Proves that purchasing pins at a 15% discount from $20 each, spending $170 results in 10 pins. -/
theorem pins_purchased_proof :
  calculate_pins_purchased 20 15 170 = 10 := by
  sorry

end NUMINAMATH_CALUDE_pins_purchased_proof_l1483_148398


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1483_148369

theorem imaginary_part_of_complex_fraction : 
  let z : ℂ := 2 * Complex.I / (3 - 2 * Complex.I)
  Complex.im z = 6 / 13 := by
sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1483_148369


namespace NUMINAMATH_CALUDE_inequality_proof_l1483_148377

theorem inequality_proof (a b c : ℝ) 
  (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 1) :
  2 ≤ (1 - a^2)^2 + (1 - b^2)^2 + (1 - c^2)^2 ∧ 
  (1 - a^2)^2 + (1 - b^2)^2 + (1 - c^2)^2 ≤ (1 + a) * (1 + b) * (1 + c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1483_148377


namespace NUMINAMATH_CALUDE_abes_family_total_yen_l1483_148365

/-- Given Abe's family's checking and savings account balances in yen, 
    prove that their total amount of yen is 9844. -/
theorem abes_family_total_yen (checking : ℕ) (savings : ℕ) 
    (h1 : checking = 6359) (h2 : savings = 3485) : 
    checking + savings = 9844 := by
  sorry

end NUMINAMATH_CALUDE_abes_family_total_yen_l1483_148365


namespace NUMINAMATH_CALUDE_max_ratio_hyperbola_areas_max_ratio_hyperbola_areas_equality_l1483_148394

theorem max_ratio_hyperbola_areas (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (a * b) / (a^2 + b^2) ≤ (1 : ℝ) / 2 :=
sorry

theorem max_ratio_hyperbola_areas_equality (a : ℝ) (ha : a > 0) : 
  (a * a) / (a^2 + a^2) = (1 : ℝ) / 2 :=
sorry

end NUMINAMATH_CALUDE_max_ratio_hyperbola_areas_max_ratio_hyperbola_areas_equality_l1483_148394


namespace NUMINAMATH_CALUDE_equation_positive_roots_l1483_148339

theorem equation_positive_roots (b : ℝ) : 
  ∃ (r₁ r₂ : ℝ), r₁ > 0 ∧ r₂ > 0 ∧ r₁ ≠ r₂ ∧
  (∀ x : ℝ, x > 0 → ((x - b) * (x - 2) * (x + 1) = 3 * (x - b) * (x + 1)) ↔ (x = r₁ ∨ x = r₂)) :=
sorry

end NUMINAMATH_CALUDE_equation_positive_roots_l1483_148339


namespace NUMINAMATH_CALUDE_parking_lot_motorcycles_l1483_148324

/-- The number of cars in the parking lot -/
def num_cars : ℕ := 19

/-- The number of wheels per car -/
def wheels_per_car : ℕ := 5

/-- The number of wheels per motorcycle -/
def wheels_per_motorcycle : ℕ := 2

/-- The total number of wheels for all vehicles -/
def total_wheels : ℕ := 117

/-- The number of motorcycles in the parking lot -/
def num_motorcycles : ℕ := (total_wheels - num_cars * wheels_per_car) / wheels_per_motorcycle

theorem parking_lot_motorcycles : num_motorcycles = 11 := by
  sorry

end NUMINAMATH_CALUDE_parking_lot_motorcycles_l1483_148324


namespace NUMINAMATH_CALUDE_stream_speed_calculation_l1483_148313

/-- Represents the speed of a boat in still water (in kmph) -/
def boat_speed : ℝ := 48

/-- Represents the speed of the stream (in kmph) -/
def stream_speed : ℝ := 16

/-- Represents the time ratio of upstream to downstream travel -/
def time_ratio : ℝ := 2

theorem stream_speed_calculation :
  (boat_speed - stream_speed) / (boat_speed + stream_speed) = time_ratio :=
by sorry

end NUMINAMATH_CALUDE_stream_speed_calculation_l1483_148313


namespace NUMINAMATH_CALUDE_circle_area_ratio_l1483_148375

theorem circle_area_ratio (C D : Real) (hC : C > 0) (hD : D > 0)
  (h_arc : (60 / 360) * (2 * Real.pi * C) = (40 / 360) * (2 * Real.pi * D)) :
  (Real.pi * C^2) / (Real.pi * D^2) = 9/4 := by
sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l1483_148375


namespace NUMINAMATH_CALUDE_standard_poodle_height_l1483_148351

/-- The height of the toy poodle in inches -/
def toy_height : ℕ := 14

/-- The height difference between the miniature and toy poodle in inches -/
def mini_toy_diff : ℕ := 6

/-- The height difference between the standard and miniature poodle in inches -/
def standard_mini_diff : ℕ := 8

/-- The height of the standard poodle in inches -/
def standard_height : ℕ := toy_height + mini_toy_diff + standard_mini_diff

theorem standard_poodle_height : standard_height = 28 := by
  sorry

end NUMINAMATH_CALUDE_standard_poodle_height_l1483_148351


namespace NUMINAMATH_CALUDE_divisible_by_nine_l1483_148317

theorem divisible_by_nine (n : ℕ) : ∃ k : ℤ, 2^(2*n - 1) + 3*n + 4 = 9*k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_nine_l1483_148317


namespace NUMINAMATH_CALUDE_fraction_equality_l1483_148378

theorem fraction_equality (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a - 4*b ≠ 0) (h4 : 4*a - b ≠ 0)
  (h5 : (4*a + b) / (a - 4*b) = 3) : (a + 4*b) / (4*a - b) = 9/53 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1483_148378


namespace NUMINAMATH_CALUDE_bus_problem_l1483_148319

/-- The number of children initially on the bus -/
def initial_children : ℕ := sorry

/-- The number of children who got off the bus -/
def children_off : ℕ := 63

/-- The number of children who got on the bus -/
def children_on : ℕ := sorry

/-- The number of children on the bus after the exchange -/
def final_children : ℕ := 14

theorem bus_problem :
  (initial_children - children_off + children_on = final_children) ∧
  (children_on = children_off + 9) →
  initial_children = 5 := by
  sorry

end NUMINAMATH_CALUDE_bus_problem_l1483_148319


namespace NUMINAMATH_CALUDE_level_passing_game_l1483_148302

/-- A fair six-sided die -/
def Die := Finset.range 6

/-- The number of times the die is rolled at level n -/
def rolls (n : ℕ) : ℕ := n

/-- The condition for passing a level -/
def pass_level (n : ℕ) (sum : ℕ) : Prop := sum > 2^n

/-- The maximum number of levels that can be passed -/
def max_levels : ℕ := 4

/-- The probability of passing the first three levels consecutively -/
def prob_pass_three : ℚ := 100 / 243

theorem level_passing_game :
  (∀ n : ℕ, n > max_levels → ¬∃ sum : ℕ, sum ≤ 6 * rolls n ∧ pass_level n sum) ∧
  (∃ sum : ℕ, sum ≤ 6 * rolls max_levels ∧ pass_level max_levels sum) ∧
  prob_pass_three = (2/3) * (5/6) * (20/27) :=
sorry

end NUMINAMATH_CALUDE_level_passing_game_l1483_148302


namespace NUMINAMATH_CALUDE_complex_number_real_imag_equal_l1483_148325

theorem complex_number_real_imag_equal (a : ℝ) : 
  let z : ℂ := (6 + a * Complex.I) / (3 - Complex.I)
  (z.re = z.im) → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_real_imag_equal_l1483_148325


namespace NUMINAMATH_CALUDE_inscribed_circles_radii_sum_l1483_148353

/-- Given a triangle with an inscribed circle of radius r and three smaller triangles
    formed by tangent lines parallel to the sides of the original triangle, each with
    their own inscribed circles of radii r₁, r₂, and r₃, the sum of the radii of the
    smaller inscribed circles equals the radius of the original inscribed circle. -/
theorem inscribed_circles_radii_sum (r r₁ r₂ r₃ : ℝ) 
  (h : r > 0 ∧ r₁ > 0 ∧ r₂ > 0 ∧ r₃ > 0) : r₁ + r₂ + r₃ = r := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circles_radii_sum_l1483_148353


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l1483_148310

/-- The length of the major axis of an ellipse with given foci and tangent to x-axis -/
theorem ellipse_major_axis_length : 
  let f1 : ℝ × ℝ := (5, 15)
  let f2 : ℝ × ℝ := (40, 45)
  ∀ (E : Set (ℝ × ℝ)), 
    (∀ p ∈ E, dist p f1 + dist p f2 = dist p f1 + dist p f2) →  -- E is an ellipse with foci f1 and f2
    (∃ x, (x, 0) ∈ E) →  -- E is tangent to x-axis
    (∃ a : ℝ, ∀ p ∈ E, dist p f1 + dist p f2 = 2 * a) →  -- Definition of ellipse
    2 * (dist f1 f2) = 10 * Real.sqrt 193 :=
by
  sorry


end NUMINAMATH_CALUDE_ellipse_major_axis_length_l1483_148310


namespace NUMINAMATH_CALUDE_machine_selling_price_l1483_148300

/-- Calculates the selling price of a machine given its costs and desired profit percentage. -/
def selling_price (purchase_price repair_cost transport_cost profit_percentage : ℕ) : ℕ :=
  let total_cost := purchase_price + repair_cost + transport_cost
  let profit := total_cost * profit_percentage / 100
  total_cost + profit

/-- Proves that the selling price of the machine is 25500 Rs given the specified costs and profit percentage. -/
theorem machine_selling_price :
  selling_price 11000 5000 1000 50 = 25500 := by
  sorry

#eval selling_price 11000 5000 1000 50

end NUMINAMATH_CALUDE_machine_selling_price_l1483_148300


namespace NUMINAMATH_CALUDE_marble_box_count_l1483_148384

theorem marble_box_count : ∀ (p y u : ℕ),
  y + u = 10 →
  p + u = 12 →
  p + y = 6 →
  p + y + u = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_marble_box_count_l1483_148384


namespace NUMINAMATH_CALUDE_eldorado_license_plates_l1483_148309

theorem eldorado_license_plates : 
  let letter_choices : ℕ := 26
  let digit_choices : ℕ := 10
  let letter_spots : ℕ := 3
  let digit_spots : ℕ := 4
  letter_choices ^ letter_spots * digit_choices ^ digit_spots = 175760000 :=
by sorry

end NUMINAMATH_CALUDE_eldorado_license_plates_l1483_148309


namespace NUMINAMATH_CALUDE_closest_ratio_is_one_to_one_l1483_148304

def admission_fee_adult : ℕ := 30
def admission_fee_child : ℕ := 15
def total_collected : ℕ := 2250

def is_valid_combination (adults children : ℕ) : Prop :=
  adults ≥ 1 ∧ children ≥ 1 ∧
  adults * admission_fee_adult + children * admission_fee_child = total_collected

def ratio_difference_from_one (adults children : ℕ) : ℚ :=
  |((adults : ℚ) / (children : ℚ)) - 1|

theorem closest_ratio_is_one_to_one :
  ∃ (a c : ℕ), is_valid_combination a c ∧
    ∀ (x y : ℕ), is_valid_combination x y →
      ratio_difference_from_one a c ≤ ratio_difference_from_one x y :=
by sorry

end NUMINAMATH_CALUDE_closest_ratio_is_one_to_one_l1483_148304


namespace NUMINAMATH_CALUDE_sum_of_digits_square_count_l1483_148316

/-- Sum of digits function -/
def S (n : ℕ) : ℕ := sorry

/-- The count of valid numbers -/
def K : ℕ := sorry

theorem sum_of_digits_square_count :
  K % 1000 = 632 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_square_count_l1483_148316


namespace NUMINAMATH_CALUDE_opposite_of_2023_l1483_148350

/-- The opposite of a number is obtained by changing its sign -/
def opposite (x : ℤ) : ℤ := -x

/-- Prove that the opposite of 2023 is -2023 -/
theorem opposite_of_2023 : opposite 2023 = -2023 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_2023_l1483_148350


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_l1483_148345

theorem gcd_lcm_sum : Nat.gcd 45 75 + Nat.lcm 48 18 = 159 := by sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_l1483_148345


namespace NUMINAMATH_CALUDE_sound_reach_time_l1483_148320

/-- Time taken for sound to reach a moving receiver -/
theorem sound_reach_time (distance : ℝ) (sound_speed : ℝ) (receiver_speed : ℝ) 
  (h1 : distance = 1200)
  (h2 : sound_speed = 330)
  (h3 : receiver_speed = 30) :
  distance / (sound_speed - receiver_speed) = 4 := by
  sorry

end NUMINAMATH_CALUDE_sound_reach_time_l1483_148320


namespace NUMINAMATH_CALUDE_current_monthly_production_l1483_148358

/-- Represents the car manufacturing company's production data -/
structure CarProduction where
  targetAnnual : ℕ
  monthlyIncrease : ℕ
  currentMonthly : ℕ

/-- Theorem stating that the current monthly production is 100 cars -/
theorem current_monthly_production (cp : CarProduction) 
  (h1 : cp.targetAnnual = 1800)
  (h2 : cp.monthlyIncrease = 50)
  (h3 : cp.currentMonthly * 12 + cp.monthlyIncrease * 12 = cp.targetAnnual) :
  cp.currentMonthly = 100 := by
  sorry

#check current_monthly_production

end NUMINAMATH_CALUDE_current_monthly_production_l1483_148358


namespace NUMINAMATH_CALUDE_correct_statements_are_ACD_l1483_148383

-- Define the set of all statements
inductive Statement : Type
| A : Statement
| B : Statement
| C : Statement
| D : Statement

-- Define a function to check if a statement is correct
def is_correct : Statement → Prop
| Statement.A => ∀ (residual_width : ℝ) (fitting_quality : ℝ),
    residual_width < 0 → fitting_quality > 0
| Statement.B => ∀ (r_A r_B : ℝ),
    r_A = 0.97 ∧ r_B = -0.99 → abs r_A > abs r_B
| Statement.C => ∀ (R_squared fitting_quality : ℝ),
    R_squared < 0 → fitting_quality < 0
| Statement.D => ∀ (n k d : ℕ),
    n = 10 ∧ k = 2 ∧ d = 3 →
    (Nat.choose d 1 * Nat.choose (n - d) (k - 1)) / Nat.choose n k = 7 / 15

-- Define the set of correct statements
def correct_statements : Set Statement :=
  {s | is_correct s}

-- Theorem to prove
theorem correct_statements_are_ACD :
  correct_statements = {Statement.A, Statement.C, Statement.D} :=
sorry

end NUMINAMATH_CALUDE_correct_statements_are_ACD_l1483_148383


namespace NUMINAMATH_CALUDE_product_of_two_digit_numbers_l1483_148332

theorem product_of_two_digit_numbers (a b : ℕ) 
  (h1 : 10 ≤ a ∧ a ≤ 99) 
  (h2 : 10 ≤ b ∧ b ≤ 99) 
  (h3 : a * b = 4500) 
  (h4 : a ≤ b) : 
  a = 50 := by
sorry

end NUMINAMATH_CALUDE_product_of_two_digit_numbers_l1483_148332


namespace NUMINAMATH_CALUDE_wall_building_time_l1483_148354

/-- The number of days required for a group of workers to build a wall, given:
  * The number of workers in the reference group
  * The length of the wall built by the reference group
  * The number of days taken by the reference group
  * The number of workers in the new group
  * The length of the wall to be built by the new group
-/
def days_required (
  ref_workers : ℕ
  ) (ref_length : ℕ
  ) (ref_days : ℕ
  ) (new_workers : ℕ
  ) (new_length : ℕ
  ) : ℚ :=
  (ref_workers * ref_days * new_length : ℚ) / (new_workers * ref_length)

/-- Theorem stating that 30 workers will take 18 days to build a 100m wall,
    given that 18 workers can build a 140m wall in 42 days -/
theorem wall_building_time :
  days_required 18 140 42 30 100 = 18 := by
  sorry

end NUMINAMATH_CALUDE_wall_building_time_l1483_148354


namespace NUMINAMATH_CALUDE_will_uses_six_pages_l1483_148301

/-- The number of cards Will can put on each page -/
def cards_per_page : ℕ := 3

/-- The number of new cards Will has -/
def new_cards : ℕ := 8

/-- The number of old cards Will has -/
def old_cards : ℕ := 10

/-- The total number of cards Will has -/
def total_cards : ℕ := new_cards + old_cards

/-- The number of pages Will uses -/
def pages_used : ℕ := total_cards / cards_per_page

theorem will_uses_six_pages : pages_used = 6 := by
  sorry

end NUMINAMATH_CALUDE_will_uses_six_pages_l1483_148301


namespace NUMINAMATH_CALUDE_perpendicular_lines_from_perpendicular_planes_l1483_148387

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation for planes and lines
variable (perp_plane : Plane → Plane → Prop)
variable (perp_line_plane : Line → Plane → Prop)
variable (perp_line : Line → Line → Prop)

-- State the theorem
theorem perpendicular_lines_from_perpendicular_planes
  (α β : Plane) (a b : Line)
  (h1 : perp_plane α β)
  (h2 : perp_line_plane a α)
  (h3 : perp_line_plane b β) :
  perp_line a b :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_from_perpendicular_planes_l1483_148387


namespace NUMINAMATH_CALUDE_line_perp_plane_if_perp_two_intersecting_lines_planes_perp_if_line_in_one_perp_other_l1483_148323

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Line → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_plane_plane : Plane → Plane → Prop)
variable (contained_in : Line → Plane → Prop)
variable (intersect : Line → Line → Prop)

-- Theorem 1
theorem line_perp_plane_if_perp_two_intersecting_lines 
  (l : Line) (α : Plane) (m n : Line) :
  contained_in m α → contained_in n α → 
  intersect m n → 
  perpendicular l m → perpendicular l n → 
  perpendicular_line_plane l α :=
sorry

-- Theorem 2
theorem planes_perp_if_line_in_one_perp_other 
  (l : Line) (α β : Plane) :
  contained_in l β → perpendicular_line_plane l α → 
  perpendicular_plane_plane α β :=
sorry

end NUMINAMATH_CALUDE_line_perp_plane_if_perp_two_intersecting_lines_planes_perp_if_line_in_one_perp_other_l1483_148323


namespace NUMINAMATH_CALUDE_max_first_term_is_16_l1483_148355

/-- A sequence of positive real numbers satisfying the given conditions -/
def SpecialSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, n > 0 → a n > 0) ∧ 
  (∀ n, n > 0 → (2 * a (n + 1) - a n) * (a (n + 1) * a n - 1) = 0) ∧
  (a 1 = a 10)

/-- The maximum possible value of the first term in the special sequence is 16 -/
theorem max_first_term_is_16 (a : ℕ → ℝ) (h : SpecialSequence a) : 
  ∃ (M : ℝ), M = 16 ∧ a 1 ≤ M ∧ ∀ (b : ℕ → ℝ), SpecialSequence b → b 1 ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_first_term_is_16_l1483_148355


namespace NUMINAMATH_CALUDE_cafeteria_pies_l1483_148328

theorem cafeteria_pies (initial_apples handed_out apples_per_pie : ℕ) 
  (h1 : initial_apples = 62)
  (h2 : handed_out = 8)
  (h3 : apples_per_pie = 9) :
  (initial_apples - handed_out) / apples_per_pie = 6 := by
sorry

end NUMINAMATH_CALUDE_cafeteria_pies_l1483_148328


namespace NUMINAMATH_CALUDE_max_area_rectangle_l1483_148376

/-- Represents a rectangle with width and length -/
structure Rectangle where
  width : ℝ
  length : ℝ

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.width + r.length)

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.width * r.length

/-- Theorem: The maximum area of a rectangle with perimeter 60 and length 5 more than width -/
theorem max_area_rectangle :
  ∃ (r : Rectangle),
    perimeter r = 60 ∧
    r.length = r.width + 5 ∧
    area r = 218.75 ∧
    ∀ (r' : Rectangle),
      perimeter r' = 60 →
      r'.length = r'.width + 5 →
      area r' ≤ area r := by
  sorry

end NUMINAMATH_CALUDE_max_area_rectangle_l1483_148376


namespace NUMINAMATH_CALUDE_probability_of_purple_l1483_148336

def die_sides : ℕ := 6
def red_sides : ℕ := 3
def yellow_sides : ℕ := 2
def blue_sides : ℕ := 1

def prob_red : ℚ := red_sides / die_sides
def prob_blue : ℚ := blue_sides / die_sides

theorem probability_of_purple (h1 : die_sides = red_sides + yellow_sides + blue_sides)
  (h2 : prob_red = red_sides / die_sides)
  (h3 : prob_blue = blue_sides / die_sides) :
  prob_red * prob_blue + prob_blue * prob_red = 1 / 6 :=
sorry

end NUMINAMATH_CALUDE_probability_of_purple_l1483_148336


namespace NUMINAMATH_CALUDE_remainder_113_pow_113_plus_113_mod_137_l1483_148357

theorem remainder_113_pow_113_plus_113_mod_137 
  (h1 : Prime 113) 
  (h2 : Prime 137) 
  (h3 : 113 < 137) : 
  (113^113 + 113) % 137 = 89 := by
sorry

end NUMINAMATH_CALUDE_remainder_113_pow_113_plus_113_mod_137_l1483_148357


namespace NUMINAMATH_CALUDE_sum_of_first_ten_terms_l1483_148337

/-- Given a sequence {a_n} and its partial sum sequence {S_n}, prove that S_10 = 145 -/
theorem sum_of_first_ten_terms 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ) 
  (h1 : ∀ n, S (n + 1) = S n + a n + 3)
  (h2 : a 5 + a 6 = 29) :
  S 10 = 145 := by
sorry

end NUMINAMATH_CALUDE_sum_of_first_ten_terms_l1483_148337


namespace NUMINAMATH_CALUDE_root_difference_zero_l1483_148385

theorem root_difference_zero (x : ℝ) : 
  (x^2 + 40*x + 300 = -100) → 
  (∃ r₁ r₂ : ℝ, (r₁^2 + 40*r₁ + 300 = -100) ∧ 
                (r₂^2 + 40*r₂ + 300 = -100) ∧ 
                (|r₁ - r₂| = 0)) := by
  sorry

end NUMINAMATH_CALUDE_root_difference_zero_l1483_148385


namespace NUMINAMATH_CALUDE_investment_sum_l1483_148344

/-- Represents the investment scenario described in the problem -/
structure Investment where
  principal : ℝ  -- The initial sum invested
  rate : ℝ       -- The annual simple interest rate
  peter_years : ℕ := 3
  david_years : ℕ := 4
  peter_return : ℝ := 815
  david_return : ℝ := 854

/-- The amount returned after a given number of years with simple interest -/
def amount_after (i : Investment) (years : ℕ) : ℝ :=
  i.principal + (i.principal * i.rate * years)

/-- The theorem stating that the invested sum is 698 given the conditions -/
theorem investment_sum (i : Investment) : 
  (amount_after i i.peter_years = i.peter_return) → 
  (amount_after i i.david_years = i.david_return) → 
  i.principal = 698 := by
  sorry

end NUMINAMATH_CALUDE_investment_sum_l1483_148344


namespace NUMINAMATH_CALUDE_tshirt_cost_l1483_148397

def amusement_park_problem (initial_amount ticket_cost food_cost remaining_amount : ℕ) : Prop :=
  let total_spent := ticket_cost + food_cost + (initial_amount - ticket_cost - food_cost - remaining_amount)
  total_spent = initial_amount - remaining_amount

theorem tshirt_cost (initial_amount ticket_cost food_cost remaining_amount : ℕ) 
  (h1 : initial_amount = 75)
  (h2 : ticket_cost = 30)
  (h3 : food_cost = 13)
  (h4 : remaining_amount = 9)
  (h5 : amusement_park_problem initial_amount ticket_cost food_cost remaining_amount) :
  initial_amount - ticket_cost - food_cost - remaining_amount = 23 := by
  sorry

end NUMINAMATH_CALUDE_tshirt_cost_l1483_148397


namespace NUMINAMATH_CALUDE_inequality_proof_l1483_148360

theorem inequality_proof (a b c d : ℝ) (ha : a ≥ -1) (hb : b ≥ -1) (hc : c ≥ -1) (hd : d ≥ -1) :
  ∃! k : ℝ, ∀ (a b c d : ℝ), a ≥ -1 → b ≥ -1 → c ≥ -1 → d ≥ -1 →
    a^3 + b^3 + c^3 + d^3 + 1 ≥ k * (a + b + c + d) ∧ k = 3/4 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1483_148360


namespace NUMINAMATH_CALUDE_some_base_value_l1483_148311

theorem some_base_value (x y some_base : ℝ) 
  (h1 : x * y = 1) 
  (h2 : (some_base^((x + y)^2)) / (some_base^((x - y)^2)) = 2401) : 
  some_base = 7 := by sorry

end NUMINAMATH_CALUDE_some_base_value_l1483_148311


namespace NUMINAMATH_CALUDE_normal_line_at_x₀_l1483_148327

/-- The curve function -/
def f (x : ℝ) : ℝ := 2 * x^2 - 3 * x + 1

/-- The point of interest -/
def x₀ : ℝ := 1

/-- The normal line function -/
def normal_line (x : ℝ) : ℝ := -x + 1

theorem normal_line_at_x₀ :
  let y₀ := f x₀
  let m := (4 * x₀ - 3)⁻¹
  ∀ x, normal_line x = -m * (x - x₀) + y₀ :=
by sorry

end NUMINAMATH_CALUDE_normal_line_at_x₀_l1483_148327


namespace NUMINAMATH_CALUDE_hyperbola_foci_l1483_148331

/-- Given a hyperbola with equation x²/4 - y² = 1, prove that its foci are at (±√5, 0) -/
theorem hyperbola_foci (x y : ℝ) : 
  (x^2 / 4 - y^2 = 1) → (∃ (s : ℝ), s^2 = 5 ∧ ((x = s ∨ x = -s) ∧ y = 0)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_foci_l1483_148331


namespace NUMINAMATH_CALUDE_rightmost_bag_balls_l1483_148352

/-- The number of bags -/
def n : ℕ := 2003

/-- The number of balls in every 7 consecutive bags -/
def total_balls_in_seven : ℕ := 19

/-- The period of the ball distribution -/
def period : ℕ := 7

/-- The number of balls in the leftmost bag -/
def R : ℕ := total_balls_in_seven - (period - 1)

/-- A function representing the number of balls in the i-th bag -/
def balls (i : ℕ) : ℕ :=
  if i % period = 1 then R else (total_balls_in_seven - R) / (period - 1)

/-- The theorem to be proved -/
theorem rightmost_bag_balls : balls n = 8 := by
  sorry

end NUMINAMATH_CALUDE_rightmost_bag_balls_l1483_148352


namespace NUMINAMATH_CALUDE_no_prime_solution_l1483_148388

theorem no_prime_solution : ¬∃ (p q : ℕ), Prime p ∧ Prime q ∧ p > 5 ∧ q > 5 ∧ (p * q ∣ (5^p - 2^p) * (5^q - 2^q)) := by
  sorry

end NUMINAMATH_CALUDE_no_prime_solution_l1483_148388


namespace NUMINAMATH_CALUDE_integer_sum_proof_l1483_148392

theorem integer_sum_proof (x y : ℕ+) (h1 : x - y = 8) (h2 : x * y = 180) : x + y = 28 := by
  sorry

end NUMINAMATH_CALUDE_integer_sum_proof_l1483_148392


namespace NUMINAMATH_CALUDE_sqrt_twelve_times_sqrt_three_minus_five_equals_one_l1483_148399

theorem sqrt_twelve_times_sqrt_three_minus_five_equals_one :
  Real.sqrt 12 * Real.sqrt 3 - 5 = 1 := by sorry

end NUMINAMATH_CALUDE_sqrt_twelve_times_sqrt_three_minus_five_equals_one_l1483_148399


namespace NUMINAMATH_CALUDE_julia_watch_collection_l1483_148362

theorem julia_watch_collection (silver : ℕ) (bronze : ℕ) (gold : ℕ) : 
  silver = 20 →
  bronze = 3 * silver →
  gold = (silver + bronze) / 10 →
  silver + bronze + gold = 88 :=
by
  sorry

end NUMINAMATH_CALUDE_julia_watch_collection_l1483_148362

import Mathlib

namespace NUMINAMATH_CALUDE_simplify_sqrt_sum_l3857_385781

theorem simplify_sqrt_sum : 
  Real.sqrt 1 + Real.sqrt (1 + 2) + Real.sqrt (1 + 2 + 3) + Real.sqrt (1 + 2 + 3 + 4) = 
  1 + Real.sqrt 3 + Real.sqrt 6 + Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_sum_l3857_385781


namespace NUMINAMATH_CALUDE_system_solution_l3857_385745

theorem system_solution (x y m : ℝ) : 
  (2 * x + y = 7) → 
  (x + 2 * y = m - 3) → 
  (x - y = 2) → 
  (m = 8) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l3857_385745


namespace NUMINAMATH_CALUDE_age_sum_from_product_l3857_385731

theorem age_sum_from_product (a b c : ℕ+) : 
  a = b ∧ a > c ∧ a * b * c = 144 → a + b + c = 16 := by
  sorry

end NUMINAMATH_CALUDE_age_sum_from_product_l3857_385731


namespace NUMINAMATH_CALUDE_radius_of_circle_from_spherical_coords_l3857_385795

/-- The radius of the circle formed by points with spherical coordinates (1, θ, π/3) is √3/2 -/
theorem radius_of_circle_from_spherical_coords :
  let r : ℝ := Real.sqrt 3 / 2
  ∀ θ : ℝ,
  let x : ℝ := (1 : ℝ) * Real.sin (π / 3) * Real.cos θ
  let y : ℝ := (1 : ℝ) * Real.sin (π / 3) * Real.sin θ
  Real.sqrt (x^2 + y^2) = r :=
by sorry

end NUMINAMATH_CALUDE_radius_of_circle_from_spherical_coords_l3857_385795


namespace NUMINAMATH_CALUDE_bridge_length_bridge_length_problem_l3857_385740

/-- The length of a bridge given train parameters --/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (time_to_pass : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  let total_distance := train_speed_ms * time_to_pass
  total_distance - train_length

/-- Proof of the bridge length problem --/
theorem bridge_length_problem : 
  bridge_length 360 75 24 = 140 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_bridge_length_problem_l3857_385740


namespace NUMINAMATH_CALUDE_smallest_angle_EBC_l3857_385758

theorem smallest_angle_EBC (ABC ABD DBE : ℝ) (h1 : ABC = 40) (h2 : ABD = 30) (h3 : DBE = 10) : 
  ∃ (EBC : ℝ), EBC = 20 ∧ ∀ (x : ℝ), x ≥ 20 → x ≥ EBC := by
  sorry

end NUMINAMATH_CALUDE_smallest_angle_EBC_l3857_385758


namespace NUMINAMATH_CALUDE_min_value_of_a_is_two_l3857_385737

/-- Given an equation with parameter a and two real solutions, 
    prove that the minimum value of a is 2 -/
theorem min_value_of_a_is_two (a : ℝ) (x₁ x₂ : ℝ) : 
  (9 * x₁ - (4 + a) * 3 * x₁ + 4 = 0) ∧ 
  (9 * x₂ - (4 + a) * 3 * x₂ + 4 = 0) ∧ 
  (x₁ ≠ x₂) →
  ∀ b : ℝ, (∃ y₁ y₂ : ℝ, (9 * y₁ - (4 + b) * 3 * y₁ + 4 = 0) ∧ 
                         (9 * y₂ - (4 + b) * 3 * y₂ + 4 = 0) ∧ 
                         (y₁ ≠ y₂)) →
  b ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_a_is_two_l3857_385737


namespace NUMINAMATH_CALUDE_alissa_picked_16_flowers_l3857_385751

/-- The number of flowers Alissa picked -/
def alissa_flowers : ℕ := sorry

/-- The number of flowers Melissa picked -/
def melissa_flowers : ℕ := sorry

/-- The number of flowers given to their mother -/
def flowers_to_mother : ℕ := 18

/-- The number of flowers left after giving to their mother -/
def flowers_left : ℕ := 14

theorem alissa_picked_16_flowers :
  (alissa_flowers = melissa_flowers) ∧
  (alissa_flowers + melissa_flowers = flowers_to_mother + flowers_left) ∧
  (flowers_to_mother = 18) ∧
  (flowers_left = 14) →
  alissa_flowers = 16 := by sorry

end NUMINAMATH_CALUDE_alissa_picked_16_flowers_l3857_385751


namespace NUMINAMATH_CALUDE_right_triangle_legs_l3857_385777

/-- A right-angled triangle with a point inside it -/
structure RightTriangleWithPoint where
  /-- Length of one leg of the triangle -/
  x : ℝ
  /-- Length of the other leg of the triangle -/
  y : ℝ
  /-- Distance from the point to one side -/
  d1 : ℝ
  /-- Distance from the point to the other side -/
  d2 : ℝ
  /-- The triangle is right-angled -/
  right_angle : x > 0 ∧ y > 0
  /-- The point is inside the triangle -/
  point_inside : d1 > 0 ∧ d2 > 0 ∧ d1 < y ∧ d2 < x
  /-- The area of the triangle is 100 -/
  area : x * y / 2 = 100
  /-- The distances from the point to the sides are 4 and 8 -/
  distances : d1 = 4 ∧ d2 = 8

/-- The theorem stating the possible leg lengths of the triangle -/
theorem right_triangle_legs (t : RightTriangleWithPoint) :
  (t.x = 40 ∧ t.y = 5) ∨ (t.x = 10 ∧ t.y = 20) :=
sorry

end NUMINAMATH_CALUDE_right_triangle_legs_l3857_385777


namespace NUMINAMATH_CALUDE_min_value_squared_sum_l3857_385753

theorem min_value_squared_sum (x y z : ℝ) (h : 2*x + 3*y + z = 7) :
  x^2 + y^2 + z^2 ≥ 7/2 :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_squared_sum_l3857_385753


namespace NUMINAMATH_CALUDE_complex_number_problem_l3857_385787

theorem complex_number_problem (z : ℂ) (i : ℂ) : 
  i * i = -1 → z / (-i) = 1 + 2*i → z = 2 - i := by
  sorry

end NUMINAMATH_CALUDE_complex_number_problem_l3857_385787


namespace NUMINAMATH_CALUDE_f_simplification_f_value_at_specific_angle_l3857_385797

noncomputable def f (θ : ℝ) : ℝ :=
  (Real.sin (θ + 5 * Real.pi / 2) * Real.cos (3 * Real.pi / 2 - θ) * Real.cos (θ + 3 * Real.pi)) /
  (Real.cos (-Real.pi / 2 - θ) * Real.sin (-3 * Real.pi / 2 - θ))

theorem f_simplification (θ : ℝ) : f θ = -Real.cos θ := by sorry

theorem f_value_at_specific_angle (θ : ℝ) (h : Real.sin (θ - Real.pi / 6) = 3 / 5) :
  f (θ + Real.pi / 3) = 3 / 5 := by sorry

end NUMINAMATH_CALUDE_f_simplification_f_value_at_specific_angle_l3857_385797


namespace NUMINAMATH_CALUDE_abs_eq_sqrt_sq_l3857_385744

theorem abs_eq_sqrt_sq (x : ℝ) : |x| = Real.sqrt (x^2) := by
  sorry

end NUMINAMATH_CALUDE_abs_eq_sqrt_sq_l3857_385744


namespace NUMINAMATH_CALUDE_complement_S_union_T_l3857_385717

def S : Set ℝ := {x | x > -2}
def T : Set ℝ := {x | x^2 + 3*x - 4 ≤ 0}

theorem complement_S_union_T : (Set.univ \ S) ∪ T = {x : ℝ | x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_complement_S_union_T_l3857_385717


namespace NUMINAMATH_CALUDE_square_diff_equality_l3857_385722

theorem square_diff_equality : 1003^2 - 997^2 - 1001^2 + 999^2 = 8000 := by
  sorry

end NUMINAMATH_CALUDE_square_diff_equality_l3857_385722


namespace NUMINAMATH_CALUDE_q_satisfies_conditions_l3857_385755

/-- A quadratic polynomial satisfying specific conditions -/
def q (x : ℚ) : ℚ := (20/7) * x^2 + (40/7) * x - 300/7

/-- Proof that q satisfies the given conditions -/
theorem q_satisfies_conditions :
  q (-5) = 0 ∧ q 3 = 0 ∧ q 2 = -20 :=
by sorry


end NUMINAMATH_CALUDE_q_satisfies_conditions_l3857_385755


namespace NUMINAMATH_CALUDE_car_dealer_problem_l3857_385763

theorem car_dealer_problem (X Y : ℚ) (h1 : X > 0) (h2 : Y > 0) : 
  1.54 * (X + Y) = 1.4 * X + 1.6 * Y →
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ X * b = Y * a ∧ Nat.gcd a b = 1 ∧ 11 * a + 13 * b = 124 :=
by sorry

end NUMINAMATH_CALUDE_car_dealer_problem_l3857_385763


namespace NUMINAMATH_CALUDE_min_value_theorem_l3857_385754

theorem min_value_theorem (x : ℝ) (h : x > 0) : 3 * Real.sqrt x + 2 / x^2 ≥ 5 ∧
  (3 * Real.sqrt x + 2 / x^2 = 5 ↔ x = 1) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3857_385754


namespace NUMINAMATH_CALUDE_smallest_angle_triangle_range_l3857_385743

theorem smallest_angle_triangle_range (x : Real) : 
  (∀ y : Real, y = Real.sqrt 2 * Real.sin (x + π/4)) →
  (0 < x ∧ x ≤ π/3) →
  ∃ (a b : Real), a = 1 ∧ b = Real.sqrt 2 ∧ 
    (∀ y : Real, y = Real.sqrt 2 * Real.sin (x + π/4) → a < y ∧ y ≤ b) ∧
    (∀ z : Real, a < z ∧ z ≤ b → ∃ x₀ : Real, 0 < x₀ ∧ x₀ ≤ π/3 ∧ z = Real.sqrt 2 * Real.sin (x₀ + π/4)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_angle_triangle_range_l3857_385743


namespace NUMINAMATH_CALUDE_paper_pieces_difference_paper_pieces_problem_l3857_385764

theorem paper_pieces_difference : ℕ → ℕ → ℕ → ℕ → ℕ → Prop :=
  fun initial_squares initial_corners final_pieces final_corners corner_difference =>
    initial_squares * 4 = initial_corners →
    ∃ (triangles pentagonals : ℕ),
      triangles + pentagonals = final_pieces ∧
      3 * triangles + 5 * pentagonals = final_corners ∧
      triangles - pentagonals = corner_difference

theorem paper_pieces_problem :
  paper_pieces_difference 25 100 50 170 30 := by
  sorry

end NUMINAMATH_CALUDE_paper_pieces_difference_paper_pieces_problem_l3857_385764


namespace NUMINAMATH_CALUDE_car_travel_time_l3857_385742

theorem car_travel_time (actual_speed : ℝ) (actual_time : ℝ) 
  (h1 : actual_speed > 0) 
  (h2 : actual_time > 0) 
  (h3 : actual_speed * actual_time = (4/5 * actual_speed) * (actual_time + 15)) : 
  actual_time = 60 := by
sorry

end NUMINAMATH_CALUDE_car_travel_time_l3857_385742


namespace NUMINAMATH_CALUDE_four_digit_addition_l3857_385793

/-- Given four different natural numbers A, B, C, and D that satisfy the equation
    4A5B + C2D7 = 7070, prove that C = 2. -/
theorem four_digit_addition (A B C D : ℕ) 
  (diff : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) 
  (eq : 4000 * A + 50 * B + 1000 * C + 200 * D + 7 = 7070) : 
  C = 2 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_addition_l3857_385793


namespace NUMINAMATH_CALUDE_angle_sum_theorem_l3857_385773

theorem angle_sum_theorem (α β : Real) : 
  0 < α ∧ α < π/2 →  -- α is acute
  0 < β ∧ β < π/2 →  -- β is acute
  α + 2*β = 2*π/3 →
  Real.tan (α/2) * Real.tan β = 2 - Real.sqrt 3 →
  α + β = 5*π/12 := by
sorry

end NUMINAMATH_CALUDE_angle_sum_theorem_l3857_385773


namespace NUMINAMATH_CALUDE_triangle_height_l3857_385732

theorem triangle_height (a b : ℝ) (α : ℝ) (h_pos : 0 < a ∧ 0 < b) (h_angle : 0 < α ∧ α < π) :
  let c := Real.sqrt (a^2 + b^2 - 2*a*b*(Real.cos α))
  let h := (a * b * Real.sin α) / c
  0 < h ∧ h < a ∧ h < b ∧ h * c = a * b * Real.sin α :=
by sorry

end NUMINAMATH_CALUDE_triangle_height_l3857_385732


namespace NUMINAMATH_CALUDE_square_intersection_inverse_squares_sum_l3857_385706

/-- Given a unit square ABCD and a point E on side CD, prove that if F is the intersection
    of line AE and BC, then 1/|AE|^2 + 1/|AF|^2 = 1. -/
theorem square_intersection_inverse_squares_sum (A B C D E F : ℝ × ℝ) : 
  -- Square ABCD has side length 1
  A = (0, 1) ∧ B = (1, 1) ∧ C = (1, 0) ∧ D = (0, 0) →
  -- E lies on CD
  ∃ x : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ E = (x, 0) →
  -- F is the intersection of AE and BC
  F = (1, 0) →
  -- Then 1/|AE|^2 + 1/|AF|^2 = 1
  1 / (Real.sqrt ((E.1 - A.1)^2 + (E.2 - A.2)^2))^2 + 
  1 / (Real.sqrt ((F.1 - A.1)^2 + (F.2 - A.2)^2))^2 = 1 := by
  sorry


end NUMINAMATH_CALUDE_square_intersection_inverse_squares_sum_l3857_385706


namespace NUMINAMATH_CALUDE_carrot_sticks_total_l3857_385719

theorem carrot_sticks_total (before_dinner after_dinner total : ℕ) 
  (h1 : before_dinner = 22)
  (h2 : after_dinner = 15)
  (h3 : total = before_dinner + after_dinner) :
  total = 37 := by sorry

end NUMINAMATH_CALUDE_carrot_sticks_total_l3857_385719


namespace NUMINAMATH_CALUDE_problem_statement_l3857_385788

theorem problem_statement (a b c : ℤ) 
  (h1 : 0 < c) (h2 : c < 90) 
  (h3 : Real.sqrt (9 - 8 * Real.sin (50 * π / 180)) = a + b * Real.sin (c * π / 180)) :
  (a + b) / c = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3857_385788


namespace NUMINAMATH_CALUDE_sum_of_integers_l3857_385708

theorem sum_of_integers (x y z w : ℤ) 
  (eq1 : x - y + z = 7)
  (eq2 : y - z + w = 8)
  (eq3 : z - w + x = 4)
  (eq4 : w - x + y = 3) :
  x + y + z + w = 22 := by
sorry

end NUMINAMATH_CALUDE_sum_of_integers_l3857_385708


namespace NUMINAMATH_CALUDE_final_amount_correct_l3857_385771

/-- Represents the financial transactions in the boot-selling scenario -/
def boot_sale (original_price total_collected price_per_boot return_amount candy_cost actual_return : ℚ) : Prop :=
  -- Original intended price
  original_price = 25 ∧
  -- Total collected from selling two boots
  total_collected = 2 * price_per_boot ∧
  -- Price per boot
  price_per_boot = 12.5 ∧
  -- Amount to be returned per boot
  return_amount = 2.5 ∧
  -- Cost of candy Hans bought
  candy_cost = 3 ∧
  -- Actual amount returned to each customer
  actual_return = 1

/-- The theorem stating that the final amount Karl received is correct -/
theorem final_amount_correct 
  (original_price total_collected price_per_boot return_amount candy_cost actual_return : ℚ)
  (h : boot_sale original_price total_collected price_per_boot return_amount candy_cost actual_return) :
  total_collected - (2 * actual_return) = original_price - (2 * return_amount) :=
by sorry

end NUMINAMATH_CALUDE_final_amount_correct_l3857_385771


namespace NUMINAMATH_CALUDE_overlapping_squares_diagonal_l3857_385741

theorem overlapping_squares_diagonal (small_side large_side : ℝ) 
  (h1 : small_side = 1) 
  (h2 : large_side = 7) : 
  Real.sqrt ((small_side + large_side)^2 + (large_side - small_side)^2) = 10 := by
  sorry

end NUMINAMATH_CALUDE_overlapping_squares_diagonal_l3857_385741


namespace NUMINAMATH_CALUDE_ping_pong_tournament_l3857_385707

theorem ping_pong_tournament (n : ℕ) (k : ℕ) : 
  (∀ subset : Finset (Fin n), subset.card = n - 2 → Nat.choose subset.card 2 = 3^k) →
  n = 5 :=
sorry

end NUMINAMATH_CALUDE_ping_pong_tournament_l3857_385707


namespace NUMINAMATH_CALUDE_profit_calculation_l3857_385783

theorem profit_calculation (cost1 cost2 cost3 : ℝ) (profit_percentage : ℝ) :
  cost1 = 200 →
  cost2 = 300 →
  cost3 = 500 →
  profit_percentage = 0.1 →
  let total_cost := cost1 + cost2 + cost3
  let total_selling_price := total_cost + total_cost * profit_percentage
  total_selling_price = 1100 := by
  sorry

end NUMINAMATH_CALUDE_profit_calculation_l3857_385783


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l3857_385792

theorem fraction_equation_solution :
  ∃ x : ℚ, (x + 10) / (x - 4) = (x - 3) / (x + 6) ∧ x = -48 / 23 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l3857_385792


namespace NUMINAMATH_CALUDE_jen_age_proof_l3857_385711

/-- Jen's age when her son was born -/
def jen_age_at_birth : ℕ := 25

/-- Jen's son's current age -/
def son_current_age : ℕ := 16

/-- Jen's current age -/
def jen_current_age : ℕ := 3 * son_current_age - 7

theorem jen_age_proof : jen_current_age = 41 := by
  sorry

end NUMINAMATH_CALUDE_jen_age_proof_l3857_385711


namespace NUMINAMATH_CALUDE_product_sum_relation_l3857_385760

theorem product_sum_relation (a b : ℝ) : 
  a * b = 2 * (a + b) + 10 → b = 9 → b - a = 5 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_relation_l3857_385760


namespace NUMINAMATH_CALUDE_breath_holding_increase_l3857_385703

theorem breath_holding_increase (initial_time : ℝ) (final_time : ℝ) : 
  initial_time = 10 →
  final_time = 60 →
  let first_week := initial_time * 2
  let second_week := first_week * 2
  (final_time - second_week) / second_week * 100 = 50 := by
sorry

end NUMINAMATH_CALUDE_breath_holding_increase_l3857_385703


namespace NUMINAMATH_CALUDE_special_triangle_properties_l3857_385778

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given triangle satisfies the specified conditions. -/
def SpecialTriangle (t : Triangle) : Prop :=
  t.a + t.b = 5 ∧ t.c = Real.sqrt 7 ∧ Real.cos t.A + (1/2) * t.a = t.b

theorem special_triangle_properties (t : Triangle) (h : SpecialTriangle t) :
  t.C = π/3 ∧ (1/2) * t.a * t.b * Real.sin t.C = (3 * Real.sqrt 3) / 2 := by
  sorry

#check special_triangle_properties

end NUMINAMATH_CALUDE_special_triangle_properties_l3857_385778


namespace NUMINAMATH_CALUDE_min_lines_inequality_l3857_385794

/-- Represents the minimum number of lines required to compute a function using only disjunctions and conjunctions -/
def M (n : ℕ) : ℕ := sorry

/-- The theorem states that for n ≥ 4, the minimum number of lines to compute f_n is at least 3 more than the minimum number of lines to compute f_(n-2) -/
theorem min_lines_inequality (n : ℕ) (h : n ≥ 4) : M n ≥ M (n - 2) + 3 := by
  sorry

end NUMINAMATH_CALUDE_min_lines_inequality_l3857_385794


namespace NUMINAMATH_CALUDE_saras_money_theorem_l3857_385779

/-- Calculates Sara's remaining money after all expenses --/
def saras_remaining_money (hours_per_week : ℕ) (weeks : ℕ) (hourly_rate : ℚ) 
  (tax_rate : ℚ) (insurance_fee : ℚ) (misc_fee : ℚ) (tire_cost : ℚ) : ℚ :=
  let gross_pay := hours_per_week * weeks * hourly_rate
  let taxes := tax_rate * gross_pay
  let net_pay := gross_pay - taxes - insurance_fee - misc_fee - tire_cost
  net_pay

/-- Theorem stating that Sara's remaining money is $292 --/
theorem saras_money_theorem : 
  saras_remaining_money 40 2 (11.5) (0.15) 60 20 410 = 292 := by
  sorry

end NUMINAMATH_CALUDE_saras_money_theorem_l3857_385779


namespace NUMINAMATH_CALUDE_johns_money_left_l3857_385705

theorem johns_money_left (initial_amount : ℚ) (snack_fraction : ℚ) (necessity_fraction : ℚ) : 
  initial_amount = 20 →
  snack_fraction = 1/5 →
  necessity_fraction = 3/4 →
  let remaining_after_snacks := initial_amount - (initial_amount * snack_fraction)
  let final_amount := remaining_after_snacks - (remaining_after_snacks * necessity_fraction)
  final_amount = 4 := by
  sorry

end NUMINAMATH_CALUDE_johns_money_left_l3857_385705


namespace NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l3857_385746

/-- For an infinite geometric series with common ratio -1/3 and sum 12, the first term is 16 -/
theorem infinite_geometric_series_first_term :
  ∀ (a : ℝ), 
    (∃ (S : ℝ), S = a / (1 - (-1/3))) →  -- Infinite geometric series formula
    (a / (1 - (-1/3)) = 12) →             -- Sum of the series is 12
    a = 16 :=                             -- First term is 16
by
  sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l3857_385746


namespace NUMINAMATH_CALUDE_min_sum_of_quadratic_roots_l3857_385752

theorem min_sum_of_quadratic_roots (m n : ℝ) : 
  m > 0 → n > 0 → 
  (∃ x : ℝ, x^2 + m*x + 2*n = 0) → 
  (∃ x : ℝ, x^2 + 2*n*x + m = 0) → 
  m + n ≥ 6 := by sorry

end NUMINAMATH_CALUDE_min_sum_of_quadratic_roots_l3857_385752


namespace NUMINAMATH_CALUDE_liza_final_balance_l3857_385726

/-- Calculates the final balance in Liza's account after a series of transactions --/
def calculate_final_balance (initial_balance rent paycheck electricity internet phone additional_deposit : ℚ) : ℚ :=
  let balance_after_rent := initial_balance - rent
  let balance_after_paycheck := balance_after_rent + paycheck
  let balance_after_bills := balance_after_paycheck - electricity - internet
  let grocery_spending := balance_after_bills * (20 / 100)
  let balance_after_groceries := balance_after_bills - grocery_spending
  let interest := balance_after_groceries * (2 / 100)
  let balance_after_interest := balance_after_groceries + interest
  let final_balance := balance_after_interest - phone + additional_deposit
  final_balance

/-- Theorem stating that Liza's final account balance is $1562.528 --/
theorem liza_final_balance :
  calculate_final_balance 800 450 1500 117 100 70 300 = 1562.528 := by
  sorry

end NUMINAMATH_CALUDE_liza_final_balance_l3857_385726


namespace NUMINAMATH_CALUDE_internet_fee_calculation_l3857_385718

/-- The fixed monthly fee for Anna's internet service -/
def fixed_fee : ℝ := sorry

/-- The variable fee per hour of usage for Anna's internet service -/
def variable_fee : ℝ := sorry

/-- Anna's internet usage in November (in hours) -/
def november_usage : ℝ := sorry

/-- Anna's bill for November -/
def november_bill : ℝ := 20.60

/-- Anna's bill for December -/
def december_bill : ℝ := 33.20

theorem internet_fee_calculation :
  (fixed_fee + variable_fee * november_usage = november_bill) ∧
  (fixed_fee + variable_fee * (3 * november_usage) = december_bill) →
  fixed_fee = 14.30 := by
sorry

end NUMINAMATH_CALUDE_internet_fee_calculation_l3857_385718


namespace NUMINAMATH_CALUDE_function_identity_l3857_385772

theorem function_identity (f : ℕ → ℕ) (h : ∀ n : ℕ, f (n + 1) > f (f n)) :
  ∀ n : ℕ, f n = n := by
  sorry

end NUMINAMATH_CALUDE_function_identity_l3857_385772


namespace NUMINAMATH_CALUDE_marlas_errand_time_l3857_385759

/-- The total time Marla spends on her errand activities -/
def total_time (driving_time grocery_time gas_time parent_teacher_time coffee_time : ℕ) : ℕ :=
  2 * driving_time + grocery_time + gas_time + parent_teacher_time + coffee_time

/-- Theorem stating the total time Marla spends on her errand activities -/
theorem marlas_errand_time : 
  total_time 20 15 5 70 30 = 160 := by sorry

end NUMINAMATH_CALUDE_marlas_errand_time_l3857_385759


namespace NUMINAMATH_CALUDE_intersection_line_equation_l3857_385768

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 10
def circle2 (x y : ℝ) : Prop := (x - 1)^2 + (y - 3)^2 = 20

-- Define the line
def line (x y : ℝ) : Prop := x + 3*y = 0

-- Theorem statement
theorem intersection_line_equation :
  ∀ (x1 y1 x2 y2 : ℝ),
    circle1 x1 y1 ∧ circle2 x1 y1 ∧
    circle1 x2 y2 ∧ circle2 x2 y2 ∧
    (x1 ≠ x2 ∨ y1 ≠ y2) →
    line x1 y1 ∧ line x2 y2 :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_line_equation_l3857_385768


namespace NUMINAMATH_CALUDE_solve_equation_l3857_385780

/-- Given an equation 19(x + y) + z = 19(-x + y) - 21 where x = 1, prove that z = -59 -/
theorem solve_equation (y : ℝ) : 
  ∃ z : ℝ, 19 * (1 + y) + z = 19 * (-1 + y) - 21 ∧ z = -59 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3857_385780


namespace NUMINAMATH_CALUDE_proportion_not_true_l3857_385709

theorem proportion_not_true (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : 3 * a = 5 * b) :
  ¬(a / b = 3 / 5) := by
  sorry

end NUMINAMATH_CALUDE_proportion_not_true_l3857_385709


namespace NUMINAMATH_CALUDE_roots_product_plus_one_l3857_385789

theorem roots_product_plus_one (p q r : ℂ) : 
  p^3 - 15*p^2 + 25*p - 10 = 0 →
  q^3 - 15*q^2 + 25*q - 10 = 0 →
  r^3 - 15*r^2 + 25*r - 10 = 0 →
  (1+p)*(1+q)*(1+r) = 51 := by
sorry

end NUMINAMATH_CALUDE_roots_product_plus_one_l3857_385789


namespace NUMINAMATH_CALUDE_line_points_k_value_l3857_385738

/-- Given a line with equation x = 2y + 5, if (m, n) and (m + 1, n + k) are two points on this line, then k = 1/2 -/
theorem line_points_k_value (m n k : ℝ) : 
  (m = 2 * n + 5) →  -- (m, n) is on the line
  (m + 1 = 2 * (n + k) + 5) →  -- (m + 1, n + k) is on the line
  k = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_line_points_k_value_l3857_385738


namespace NUMINAMATH_CALUDE_hanging_spheres_mass_ratio_l3857_385730

/-- Given two hanging spheres with masses m₁ and m₂, where the tension in the upper thread (T_B)
    is three times the tension in the lower thread (T_H), prove that the ratio m₁/m₂ = 2. -/
theorem hanging_spheres_mass_ratio
  (m₁ m₂ : ℝ) -- masses of the spheres
  (T_B T_H : ℝ) -- tensions in the upper and lower threads
  (h1 : T_B = 3 * T_H) -- condition: upper tension is 3 times lower tension
  (h2 : T_H = m₂ * (9.8 : ℝ)) -- force balance for bottom sphere
  (h3 : T_B = m₁ * (9.8 : ℝ) + T_H) -- force balance for top sphere
  : m₁ / m₂ = 2 := by
  sorry

end NUMINAMATH_CALUDE_hanging_spheres_mass_ratio_l3857_385730


namespace NUMINAMATH_CALUDE_book_pages_theorem_l3857_385786

theorem book_pages_theorem (total_pages : ℕ) : 
  (total_pages / 5 : ℚ) + 24 + (3/2 : ℚ) * ((total_pages / 5 : ℚ) + 24) = (3/4 : ℚ) * total_pages →
  total_pages = 240 := by
sorry

end NUMINAMATH_CALUDE_book_pages_theorem_l3857_385786


namespace NUMINAMATH_CALUDE_fraction_simplification_l3857_385701

theorem fraction_simplification : (2468 * 2468) / (2468 + 2468) = 1234 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3857_385701


namespace NUMINAMATH_CALUDE_meixian_kiwi_profit_1200_meixian_kiwi_profit_1800_impossible_l3857_385775

/-- Represents the kiwi sale scenario -/
structure KiwiSale where
  purchase_price : ℝ
  initial_selling_price : ℝ
  initial_sales : ℝ
  sales_increase_rate : ℝ

/-- Calculates the daily profit for a given price reduction -/
def daily_profit (ks : KiwiSale) (price_reduction : ℝ) : ℝ :=
  (ks.initial_selling_price - price_reduction - ks.purchase_price) *
  (ks.initial_sales + ks.sales_increase_rate * price_reduction)

/-- The kiwi sale scenario from the problem -/
def meixian_kiwi_sale : KiwiSale :=
  { purchase_price := 80
    initial_selling_price := 120
    initial_sales := 20
    sales_increase_rate := 2 }

theorem meixian_kiwi_profit_1200 :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
  daily_profit meixian_kiwi_sale x₁ = 1200 ∧
  daily_profit meixian_kiwi_sale x₂ = 1200 ∧
  (x₁ = 10 ∨ x₁ = 20) ∧ (x₂ = 10 ∨ x₂ = 20) :=
sorry

theorem meixian_kiwi_profit_1800_impossible :
  ¬∃ y : ℝ, daily_profit meixian_kiwi_sale y = 1800 :=
sorry

end NUMINAMATH_CALUDE_meixian_kiwi_profit_1200_meixian_kiwi_profit_1800_impossible_l3857_385775


namespace NUMINAMATH_CALUDE_classify_books_count_l3857_385767

/-- The number of ways to classify 6 distinct books into two groups -/
def classify_books : ℕ :=
  let total_books : ℕ := 6
  let intersection_size : ℕ := 3
  let remaining_books : ℕ := total_books - intersection_size
  let ways_to_choose_intersection : ℕ := Nat.choose total_books intersection_size
  let ways_to_distribute_remaining : ℕ := 3^remaining_books
  (ways_to_choose_intersection * ways_to_distribute_remaining) / 2

/-- Theorem stating that the number of ways to classify the books is 270 -/
theorem classify_books_count : classify_books = 270 := by
  sorry

end NUMINAMATH_CALUDE_classify_books_count_l3857_385767


namespace NUMINAMATH_CALUDE_volume_alteration_percentage_l3857_385757

def original_volume : ℝ := 20 * 15 * 12

def removed_volume : ℝ := 4 * (4 * 4 * 4)

def added_volume : ℝ := 4 * (2 * 2 * 2)

def net_volume_change : ℝ := removed_volume - added_volume

theorem volume_alteration_percentage :
  (net_volume_change / original_volume) * 100 = 6.22 := by
  sorry

end NUMINAMATH_CALUDE_volume_alteration_percentage_l3857_385757


namespace NUMINAMATH_CALUDE_jacob_wage_is_6_l3857_385713

-- Define the given conditions
def jake_earnings_multiplier : ℚ := 3
def jake_total_earnings : ℚ := 720
def work_days : ℕ := 5
def hours_per_day : ℕ := 8

-- Define Jake's hourly wage
def jake_hourly_wage : ℚ := jake_total_earnings / (work_days * hours_per_day)

-- Define Jacob's hourly wage
def jacob_hourly_wage : ℚ := jake_hourly_wage / jake_earnings_multiplier

-- Theorem to prove
theorem jacob_wage_is_6 : jacob_hourly_wage = 6 := by
  sorry

end NUMINAMATH_CALUDE_jacob_wage_is_6_l3857_385713


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l3857_385733

theorem hyperbola_asymptote_slope (x y : ℝ) :
  (x^2 / 49 - y^2 / 36 = 4) →
  ∃ (m : ℝ), m > 0 ∧ (y = m * x ∨ y = -m * x) ∧ m = 6/7 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l3857_385733


namespace NUMINAMATH_CALUDE_probability_of_odd_product_l3857_385750

def range_start : ℕ := 4
def range_end : ℕ := 16

def count_integers : ℕ := range_end - range_start + 1
def count_odd_integers : ℕ := (range_end - range_start + 1) / 2

def total_combinations : ℕ := count_integers.choose 3
def odd_combinations : ℕ := count_odd_integers.choose 3

theorem probability_of_odd_product :
  (odd_combinations : ℚ) / total_combinations = 10 / 143 :=
sorry

end NUMINAMATH_CALUDE_probability_of_odd_product_l3857_385750


namespace NUMINAMATH_CALUDE_circle_max_min_sum_l3857_385729

theorem circle_max_min_sum (x y : ℝ) :
  (x - 1)^2 + (y + 2)^2 = 4 →
  ∃ (S_max S_min : ℝ),
    (∀ x' y', (x' - 1)^2 + (y' + 2)^2 = 4 → 2*x' + y' ≤ S_max) ∧
    (∀ x' y', (x' - 1)^2 + (y' + 2)^2 = 4 → 2*x' + y' ≥ S_min) ∧
    S_max = 4 + 2*Real.sqrt 5 ∧
    S_min = 4 - 2*Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_circle_max_min_sum_l3857_385729


namespace NUMINAMATH_CALUDE_valid_number_count_l3857_385784

/-- Represents a valid six-digit number configuration --/
structure ValidNumber :=
  (digits : Fin 6 → Fin 6)
  (no_repetition : Function.Injective digits)
  (one_not_at_ends : digits 0 ≠ 1 ∧ digits 5 ≠ 1)
  (one_adjacent_even_pair : ∃! (i : Fin 5), 
    (digits i).val % 2 = 0 ∧ (digits (i + 1)).val % 2 = 0 ∧
    (digits i).val ≠ (digits (i + 1)).val)

/-- The number of valid six-digit numbers --/
def count_valid_numbers : ℕ := sorry

/-- The main theorem stating the count of valid numbers --/
theorem valid_number_count : count_valid_numbers = 288 := by sorry

end NUMINAMATH_CALUDE_valid_number_count_l3857_385784


namespace NUMINAMATH_CALUDE_f_2012_is_zero_l3857_385714

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem f_2012_is_zero 
  (f : ℝ → ℝ) 
  (h_odd : is_odd_function f) 
  (h_f2 : f 2 = 0) 
  (h_period : ∀ x, f (x + 4) = f x + f 4) : 
  f 2012 = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_2012_is_zero_l3857_385714


namespace NUMINAMATH_CALUDE_polynomial_roots_l3857_385785

theorem polynomial_roots (p : ℝ) (hp : p > 5/4) : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁ > 0 ∧ x₂ > 0 ∧ 
    x₁^4 - 2*p*x₁^3 + x₁^2 - 2*p*x₁ + 1 = 0 ∧
    x₂^4 - 2*p*x₂^3 + x₂^2 - 2*p*x₂ + 1 = 0 :=
sorry

end NUMINAMATH_CALUDE_polynomial_roots_l3857_385785


namespace NUMINAMATH_CALUDE_A_equiv_B_l3857_385721

-- Define the sets A and B
def A : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 + p.2 = 1 ∧ 2 * p.1 - p.2 = 2}
def B : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = 1 ∧ p.2 = 0}

-- Theorem stating that A and B are equivalent
theorem A_equiv_B : A = B := by sorry

end NUMINAMATH_CALUDE_A_equiv_B_l3857_385721


namespace NUMINAMATH_CALUDE_cereal_box_bowls_l3857_385749

/-- Given a cereal box with the following properties:
  * Each spoonful contains 4 clusters of oats
  * Each bowl has 25 spoonfuls of cereal
  * Each box contains 500 clusters of oats
  Prove that there are 5 bowlfuls of cereal in each box. -/
theorem cereal_box_bowls (clusters_per_spoon : ℕ) (spoons_per_bowl : ℕ) (clusters_per_box : ℕ)
  (h1 : clusters_per_spoon = 4)
  (h2 : spoons_per_bowl = 25)
  (h3 : clusters_per_box = 500) :
  clusters_per_box / (clusters_per_spoon * spoons_per_bowl) = 5 := by
  sorry

end NUMINAMATH_CALUDE_cereal_box_bowls_l3857_385749


namespace NUMINAMATH_CALUDE_susan_money_left_l3857_385774

def susan_problem (swimming_income babysitting_income : ℝ) 
  (clothes_percentage books_percentage gifts_percentage : ℝ) : ℝ :=
  let total_income := swimming_income + babysitting_income
  let after_clothes := total_income * (1 - clothes_percentage)
  let after_books := after_clothes * (1 - books_percentage)
  let final_amount := after_books * (1 - gifts_percentage)
  final_amount

theorem susan_money_left : 
  susan_problem 1200 600 0.4 0.25 0.15 = 688.5 := by
  sorry

end NUMINAMATH_CALUDE_susan_money_left_l3857_385774


namespace NUMINAMATH_CALUDE_shopping_trip_percentage_l3857_385762

/-- Represents the percentage of the total amount spent on other items -/
def percentage_other : ℝ := sorry

theorem shopping_trip_percentage :
  let total_amount : ℝ := 100 -- Assume total amount is 100 for percentage calculations
  let clothing_percent : ℝ := 60
  let food_percent : ℝ := 10
  let clothing_tax_rate : ℝ := 4
  let other_tax_rate : ℝ := 8
  let total_tax_percent : ℝ := 4.8

  -- Condition 1, 2, and 3
  clothing_percent + food_percent + percentage_other = total_amount ∧
  -- Condition 4, 5, and 6 (tax calculations)
  clothing_percent * clothing_tax_rate / 100 + percentage_other * other_tax_rate / 100 =
    total_tax_percent ∧
  -- Conclusion
  percentage_other = 30 := by sorry

end NUMINAMATH_CALUDE_shopping_trip_percentage_l3857_385762


namespace NUMINAMATH_CALUDE_increase_by_percentage_l3857_385796

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) (final : ℝ) :
  initial = 700 ∧ percentage = 85 ∧ final = initial * (1 + percentage / 100) →
  final = 1295 := by
  sorry

end NUMINAMATH_CALUDE_increase_by_percentage_l3857_385796


namespace NUMINAMATH_CALUDE_second_white_given_first_white_probability_l3857_385736

/-- Represents the color of a ball -/
inductive Color
| White
| Black

/-- Represents the pocket containing balls -/
structure Pocket where
  white : Nat
  black : Nat

/-- Represents the result of two consecutive draws -/
structure TwoDraws where
  first : Color
  second : Color

/-- Calculates the probability of drawing a white ball on the second draw
    given that the first ball drawn is white -/
def probSecondWhiteGivenFirstWhite (p : Pocket) : Rat :=
  if p.white > 0 then
    (p.white - 1) / (p.white + p.black - 1)
  else
    0

theorem second_white_given_first_white_probability 
  (p : Pocket) (h1 : p.white = 3) (h2 : p.black = 2) :
  probSecondWhiteGivenFirstWhite p = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_second_white_given_first_white_probability_l3857_385736


namespace NUMINAMATH_CALUDE_expression_evaluation_l3857_385724

theorem expression_evaluation :
  36 + (150 / 15) + (12^2 * 5) - 300 - (270 / 9) = 436 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3857_385724


namespace NUMINAMATH_CALUDE_fast_food_cost_l3857_385734

/-- Represents the cost of items at a fast food restaurant -/
structure FastFoodCost where
  hamburger : ℝ
  milkshake : ℝ
  fries : ℝ

/-- Given the costs of different combinations, prove the cost of 2 hamburgers, 2 milkshakes, and 2 fries -/
theorem fast_food_cost (c : FastFoodCost) 
  (eq1 : 3 * c.hamburger + 5 * c.milkshake + c.fries = 23.5)
  (eq2 : 5 * c.hamburger + 9 * c.milkshake + c.fries = 39.5) :
  2 * c.hamburger + 2 * c.milkshake + 2 * c.fries = 15 := by
  sorry

end NUMINAMATH_CALUDE_fast_food_cost_l3857_385734


namespace NUMINAMATH_CALUDE_michaels_pets_l3857_385790

theorem michaels_pets (total_pets : ℕ) (dog_percentage : ℚ) (cat_percentage : ℚ) :
  total_pets = 36 →
  dog_percentage = 25 / 100 →
  cat_percentage = 50 / 100 →
  ↑(total_pets : ℕ) * (1 - dog_percentage - cat_percentage) = 9 := by
  sorry

end NUMINAMATH_CALUDE_michaels_pets_l3857_385790


namespace NUMINAMATH_CALUDE_extremal_point_and_range_l3857_385756

noncomputable def f (a x : ℝ) : ℝ := (x - a)^2 * Real.log x

theorem extremal_point_and_range (e : ℝ) (h_e : Real.exp 1 = e) :
  (∃ a : ℝ, (deriv (f a)) e = 0 ↔ (a = e ∨ a = 3*e)) ∧
  (∀ a : ℝ, (∀ x : ℝ, x ∈ Set.Ioc 0 (3*e) → f a x ≤ 4*e^2) ↔ 
    a ∈ Set.Icc (3*e - 2*e / Real.sqrt (Real.log (3*e))) (3*e)) :=
by sorry

end NUMINAMATH_CALUDE_extremal_point_and_range_l3857_385756


namespace NUMINAMATH_CALUDE_line_parallel_plane_l3857_385727

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation for lines and planes
variable (parallel_line : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (not_subset : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_plane 
  (a b : Line) (α : Plane)
  (h1 : parallel_line a b)
  (h2 : parallel_line_plane b α)
  (h3 : not_subset a α) :
  parallel_line_plane a α :=
sorry

end NUMINAMATH_CALUDE_line_parallel_plane_l3857_385727


namespace NUMINAMATH_CALUDE_min_throws_for_repeated_sum_min_throws_is_22_l3857_385770

/-- The number of sides on each die -/
def sides : ℕ := 6

/-- The number of dice being thrown -/
def num_dice : ℕ := 4

/-- The minimum possible sum when rolling the dice -/
def min_sum : ℕ := num_dice

/-- The maximum possible sum when rolling the dice -/
def max_sum : ℕ := num_dice * sides

/-- The number of distinct possible sums -/
def distinct_sums : ℕ := max_sum - min_sum + 1

/-- 
The minimum number of throws required to guarantee a repeated sum 
when rolling four fair six-sided dice
-/
theorem min_throws_for_repeated_sum : ℕ := distinct_sums + 1

/-- The main theorem to prove -/
theorem min_throws_is_22 : min_throws_for_repeated_sum = 22 := by sorry

end NUMINAMATH_CALUDE_min_throws_for_repeated_sum_min_throws_is_22_l3857_385770


namespace NUMINAMATH_CALUDE_sqrt_expression_equals_two_l3857_385702

theorem sqrt_expression_equals_two :
  Real.sqrt 72 / Real.sqrt 3 - Real.sqrt (1/2) * Real.sqrt 12 - abs (2 - Real.sqrt 6) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equals_two_l3857_385702


namespace NUMINAMATH_CALUDE_sour_count_theorem_l3857_385769

/-- Represents the number of sours of each type -/
structure SourCounts where
  cherry : ℕ
  lemon : ℕ
  orange : ℕ
  grape : ℕ

/-- Calculates the total number of sours -/
def total_sours (counts : SourCounts) : ℕ :=
  counts.cherry + counts.lemon + counts.orange + counts.grape

/-- Represents the ratio between two quantities -/
structure Ratio where
  num : ℕ
  denom : ℕ

theorem sour_count_theorem (counts : SourCounts) 
  (cherry_lemon_ratio : Ratio) (lemon_grape_ratio : Ratio) :
  counts.cherry = 32 →
  cherry_lemon_ratio = Ratio.mk 4 5 →
  counts.cherry * cherry_lemon_ratio.denom = counts.lemon * cherry_lemon_ratio.num →
  4 * (counts.cherry + counts.lemon + counts.orange) = 3 * (counts.cherry + counts.lemon) →
  lemon_grape_ratio = Ratio.mk 3 2 →
  counts.lemon * lemon_grape_ratio.denom = counts.grape * lemon_grape_ratio.num →
  total_sours counts = 123 := by
  sorry

#check sour_count_theorem

end NUMINAMATH_CALUDE_sour_count_theorem_l3857_385769


namespace NUMINAMATH_CALUDE_length_AB_with_equal_quarter_circles_l3857_385799

/-- The length of AB given two circles with equal quarter-circle areas --/
theorem length_AB_with_equal_quarter_circles 
  (r : ℝ) 
  (h_r : r = 4)
  (π_approx : ℝ) 
  (h_π : π_approx = 3) : 
  let quarter_circle_area := (1/4) * π_approx * r^2
  let total_shaded_area := 2 * quarter_circle_area
  let AB := total_shaded_area / (2 * r)
  AB = 6 := by sorry

end NUMINAMATH_CALUDE_length_AB_with_equal_quarter_circles_l3857_385799


namespace NUMINAMATH_CALUDE_salary_sum_l3857_385723

theorem salary_sum (average_salary : ℕ) (num_people : ℕ) (known_salary : ℕ) :
  average_salary = 9000 →
  num_people = 5 →
  known_salary = 9000 →
  (num_people * average_salary) - known_salary = 36000 := by
  sorry

end NUMINAMATH_CALUDE_salary_sum_l3857_385723


namespace NUMINAMATH_CALUDE_gray_area_calculation_l3857_385765

/-- Given two rectangles with dimensions 8x10 and 12x9, and an overlapping area of 37,
    the non-overlapping area in the second rectangle (gray part) is 65. -/
theorem gray_area_calculation (rect1_width rect1_height rect2_width rect2_height black_area : ℕ)
  (h1 : rect1_width = 8)
  (h2 : rect1_height = 10)
  (h3 : rect2_width = 12)
  (h4 : rect2_height = 9)
  (h5 : black_area = 37) :
  rect2_width * rect2_height - (rect1_width * rect1_height - black_area) = 65 := by
sorry

end NUMINAMATH_CALUDE_gray_area_calculation_l3857_385765


namespace NUMINAMATH_CALUDE_f_sin_cos_inequality_l3857_385791

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def has_period_two (f : ℝ → ℝ) : Prop := ∀ x, f x = f (x + 2)

def f_on_interval (f : ℝ → ℝ) : Prop := ∀ x ∈ Set.Icc 3 4, f x = x - 2

theorem f_sin_cos_inequality 
  (f : ℝ → ℝ) 
  (h_even : is_even f) 
  (h_period : has_period_two f) 
  (h_interval : f_on_interval f) : 
  f (Real.sin 1) < f (Real.cos 1) := by
  sorry

end NUMINAMATH_CALUDE_f_sin_cos_inequality_l3857_385791


namespace NUMINAMATH_CALUDE_comic_book_stacks_result_l3857_385712

/-- The number of ways to stack comic books with given constraints -/
def comic_book_stacks (spiderman_count archie_count garfield_count : ℕ) : ℕ :=
  spiderman_count.factorial * archie_count.factorial * garfield_count.factorial * 2

/-- Theorem stating the number of ways to stack the comic books -/
theorem comic_book_stacks_result : comic_book_stacks 7 6 5 = 91612800 := by
  sorry

end NUMINAMATH_CALUDE_comic_book_stacks_result_l3857_385712


namespace NUMINAMATH_CALUDE_equation_solutions_l3857_385782

-- Define the function f(x)
def f (x : ℝ) : ℝ := |3 * x - 2|

-- Define the domain of f
def domain_f (x : ℝ) : Prop := x ≠ 3 ∧ x ≠ 0

-- Define the equation to be solved
def equation (x a : ℝ) : Prop := |3 * x - 2| = |x + a|

-- Theorem statement
theorem equation_solutions :
  ∃ (a₁ a₂ : ℝ), a₁ ≠ a₂ ∧
  (∀ x, domain_f x → equation x a₁) ∧
  (∀ x, domain_f x → equation x a₂) ∧
  (∀ a, (∀ x, domain_f x → equation x a) → (a = a₁ ∨ a = a₂)) ∧
  a₁ = -2/3 ∧ a₂ = 2 :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l3857_385782


namespace NUMINAMATH_CALUDE_circumradius_side_ratio_not_unique_l3857_385747

/-- A triangle in a 2D plane -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The circumradius of a triangle -/
def circumradius (t : Triangle) : ℝ := sorry

/-- The length of a side of a triangle -/
def side_length (t : Triangle) (side : Fin 3) : ℝ := sorry

/-- The shape of a triangle, represented by its angles -/
def triangle_shape (t : Triangle) : ℝ × ℝ × ℝ := sorry

/-- Theorem: The ratio of circumradius to one side does not uniquely determine triangle shape -/
theorem circumradius_side_ratio_not_unique (r : ℝ) (side : Fin 3) :
  ∃ t1 t2 : Triangle, 
    circumradius t1 / side_length t1 side = r ∧
    circumradius t2 / side_length t2 side = r ∧
    triangle_shape t1 ≠ triangle_shape t2 := by
  sorry

end NUMINAMATH_CALUDE_circumradius_side_ratio_not_unique_l3857_385747


namespace NUMINAMATH_CALUDE_three_digit_divisibility_l3857_385798

theorem three_digit_divisibility (a b c : ℕ) (h : ∃ k : ℕ, 100 * a + 10 * b + c = 27 * k ∨ 100 * a + 10 * b + c = 37 * k) :
  ∃ m : ℕ, 100 * b + 10 * c + a = 27 * m ∨ 100 * b + 10 * c + a = 37 * m := by
sorry

end NUMINAMATH_CALUDE_three_digit_divisibility_l3857_385798


namespace NUMINAMATH_CALUDE_april_rose_price_l3857_385776

/-- Calculates the price per rose given the initial number of roses, remaining roses, and total earnings -/
def price_per_rose (initial_roses : ℕ) (remaining_roses : ℕ) (total_earnings : ℕ) : ℚ :=
  (total_earnings : ℚ) / ((initial_roses - remaining_roses) : ℚ)

theorem april_rose_price : price_per_rose 13 4 36 = 4 := by
  sorry

end NUMINAMATH_CALUDE_april_rose_price_l3857_385776


namespace NUMINAMATH_CALUDE_range_when_p_true_range_when_p_or_q_and_p_and_q_true_l3857_385716

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 15 > 0}
def B : Set ℝ := {x | x - 6 < 0}

-- Define propositions p and q
def p (m : ℝ) : Prop := m ∈ A
def q (m : ℝ) : Prop := m ∈ B

-- Theorem for the first part
theorem range_when_p_true :
  {m : ℝ | p m} = {x | x < -3 ∨ x > 5} :=
sorry

-- Theorem for the second part
theorem range_when_p_or_q_and_p_and_q_true :
  {m : ℝ | (p m ∨ q m) ∧ (p m ∧ q m)} = {x | x < -3} :=
sorry

end NUMINAMATH_CALUDE_range_when_p_true_range_when_p_or_q_and_p_and_q_true_l3857_385716


namespace NUMINAMATH_CALUDE_certain_number_proof_l3857_385739

theorem certain_number_proof (k : ℕ) (x : ℕ) 
  (h1 : 823435 % (15^k) = 0)
  (h2 : x^k - k^5 = 1) : x = 2 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l3857_385739


namespace NUMINAMATH_CALUDE_evaluate_64_to_5_6_l3857_385700

theorem evaluate_64_to_5_6 : (64 : ℝ) ^ (5/6) = 32 := by sorry

end NUMINAMATH_CALUDE_evaluate_64_to_5_6_l3857_385700


namespace NUMINAMATH_CALUDE_positive_expression_l3857_385735

theorem positive_expression (a b : ℝ) (h : a ≠ 0 ∨ b ≠ 0) :
  5 * a^2 - 6 * a * b + 5 * b^2 > 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_expression_l3857_385735


namespace NUMINAMATH_CALUDE_ceiling_plus_x_eq_two_x_l3857_385748

theorem ceiling_plus_x_eq_two_x (x : ℝ) (h : ⌈x⌉ + ⌊x⌋ = 2 * x) : ⌈x⌉ + x = 2 * x := by
  sorry

end NUMINAMATH_CALUDE_ceiling_plus_x_eq_two_x_l3857_385748


namespace NUMINAMATH_CALUDE_work_fraction_left_l3857_385715

theorem work_fraction_left (p q : ℕ) (h1 : p = 15) (h2 : q = 20) : 
  1 - 4 * (1 / p.cast + 1 / q.cast) = 8 / 15 := by
  sorry

end NUMINAMATH_CALUDE_work_fraction_left_l3857_385715


namespace NUMINAMATH_CALUDE_cupcake_distribution_l3857_385710

theorem cupcake_distribution (total_cupcakes : ℕ) (total_children : ℕ) 
  (ratio_1 ratio_2 ratio_3 : ℕ) :
  total_cupcakes = 144 →
  total_children = 12 →
  ratio_1 = 3 →
  ratio_2 = 2 →
  ratio_3 = 1 →
  total_children % 3 = 0 →
  let total_ratio := ratio_1 + ratio_2 + ratio_3
  let cupcakes_per_part := total_cupcakes / total_ratio
  let group_3_cupcakes := ratio_3 * cupcakes_per_part
  let children_per_group := total_children / 3
  group_3_cupcakes / children_per_group = 6 :=
by sorry

end NUMINAMATH_CALUDE_cupcake_distribution_l3857_385710


namespace NUMINAMATH_CALUDE_jerry_tickets_l3857_385720

def ticket_calculation (initial_tickets spent_tickets additional_tickets : ℕ) : ℕ :=
  initial_tickets - spent_tickets + additional_tickets

theorem jerry_tickets :
  ticket_calculation 4 2 47 = 49 := by
  sorry

end NUMINAMATH_CALUDE_jerry_tickets_l3857_385720


namespace NUMINAMATH_CALUDE_coin_count_proof_l3857_385761

/-- Represents the number of nickels -/
def n : ℕ := 7

/-- Represents the number of dimes -/
def d : ℕ := 3 * n

/-- Represents the number of quarters -/
def q : ℕ := 9 * n

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The total value of all coins in cents -/
def total_value : ℕ := 1820

theorem coin_count_proof :
  (n * nickel_value + d * dime_value + q * quarter_value = total_value) →
  (n + d + q = 91) := by
  sorry

end NUMINAMATH_CALUDE_coin_count_proof_l3857_385761


namespace NUMINAMATH_CALUDE_line_points_l3857_385728

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

theorem line_points : 
  let p1 : Point := ⟨8, 16⟩
  let p2 : Point := ⟨0, -8⟩
  let p3 : Point := ⟨4, 4⟩
  let p4 : Point := ⟨2, 0⟩
  let p5 : Point := ⟨9, 19⟩
  let p6 : Point := ⟨-1, -9⟩
  let p7 : Point := ⟨-2, -10⟩
  collinear p1 p2 p3 ∧ 
  collinear p1 p2 p5 ∧ 
  ¬collinear p1 p2 p4 ∧ 
  ¬collinear p1 p2 p6 ∧ 
  ¬collinear p1 p2 p7 :=
by sorry

end NUMINAMATH_CALUDE_line_points_l3857_385728


namespace NUMINAMATH_CALUDE_min_value_of_sum_l3857_385704

theorem min_value_of_sum (x y : ℝ) : 
  x > 0 → y > 0 → x * y + 2 * x + y = 4 → 
  x + y ≥ 2 * Real.sqrt 6 - 3 ∧ 
  ∃ x y, x > 0 ∧ y > 0 ∧ x * y + 2 * x + y = 4 ∧ x + y = 2 * Real.sqrt 6 - 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_sum_l3857_385704


namespace NUMINAMATH_CALUDE_max_primes_in_table_l3857_385725

/-- A number in the table is either prime or the product of two primes -/
inductive TableNumber
  | prime : Nat → TableNumber
  | product : Nat → Nat → TableNumber

/-- Definition of the table -/
def Table := Fin 80 → Fin 80 → TableNumber

/-- Predicate to check if two TableNumbers are not coprime -/
def not_coprime : TableNumber → TableNumber → Prop :=
  sorry

/-- Predicate to check if all numbers in the table are distinct -/
def all_distinct (t : Table) : Prop :=
  sorry

/-- Predicate to check if for any number, there's another number in the same row or column that's not coprime -/
def has_not_coprime_neighbor (t : Table) : Prop :=
  sorry

/-- Count the number of prime numbers in the table -/
def count_primes (t : Table) : Nat :=
  sorry

/-- The main theorem -/
theorem max_primes_in_table :
  ∀ t : Table,
    all_distinct t →
    has_not_coprime_neighbor t →
    count_primes t ≤ 4266 :=
  sorry

end NUMINAMATH_CALUDE_max_primes_in_table_l3857_385725


namespace NUMINAMATH_CALUDE_f_is_odd_g_is_even_l3857_385766

-- Define the functions
def f (x : ℝ) : ℝ := x + x^3 + x^5
def g (x : ℝ) : ℝ := x^2 + 1

-- Define odd and even functions
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Theorem statements
theorem f_is_odd : IsOdd f := by sorry

theorem g_is_even : IsEven g := by sorry

end NUMINAMATH_CALUDE_f_is_odd_g_is_even_l3857_385766

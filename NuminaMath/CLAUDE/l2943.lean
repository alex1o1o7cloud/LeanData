import Mathlib

namespace NUMINAMATH_CALUDE_boys_to_girls_ratio_l2943_294344

theorem boys_to_girls_ratio (T : ℚ) (G : ℚ) (h : (2/3) * G = (1/4) * T) : 
  (T - G) / G = 5/3 := by
sorry

end NUMINAMATH_CALUDE_boys_to_girls_ratio_l2943_294344


namespace NUMINAMATH_CALUDE_polynomial_factor_implies_t_value_l2943_294302

theorem polynomial_factor_implies_t_value :
  ∀ t : ℤ,
  (∃ a b : ℤ, ∀ x : ℤ, x^3 - x^2 - 7*x + t = (x + 1) * (x^2 + a*x + b)) →
  t = -5 :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_factor_implies_t_value_l2943_294302


namespace NUMINAMATH_CALUDE_angle_bisector_theorem_bisector_proportion_l2943_294381

/-- Represents a triangle with side lengths and an angle bisector -/
structure BisectedTriangle where
  -- Side lengths
  p : ℝ
  q : ℝ
  r : ℝ
  -- Length of angle bisector segments
  u : ℝ
  v : ℝ
  -- Conditions
  p_pos : 0 < p
  q_pos : 0 < q
  r_pos : 0 < r
  triangle_ineq : p < q + r ∧ q < p + r ∧ r < p + q
  bisector_sum : u + v = p

/-- The angle bisector theorem holds for this triangle -/
theorem angle_bisector_theorem (t : BisectedTriangle) : t.u / t.q = t.v / t.r := sorry

/-- The main theorem: proving the proportion involving v and r -/
theorem bisector_proportion (t : BisectedTriangle) : t.v / t.r = t.p / (t.q + t.r) := by
  sorry

end NUMINAMATH_CALUDE_angle_bisector_theorem_bisector_proportion_l2943_294381


namespace NUMINAMATH_CALUDE_square_root_of_factorial_fraction_l2943_294359

theorem square_root_of_factorial_fraction : 
  Real.sqrt (Nat.factorial 9 / 126) = 24 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_factorial_fraction_l2943_294359


namespace NUMINAMATH_CALUDE_set_equality_l2943_294340

def S : Set (ℕ × ℕ) := {(x, y) | 2 * x + 3 * y = 16}

theorem set_equality : S = {(2, 4), (5, 2), (8, 0)} := by
  sorry

end NUMINAMATH_CALUDE_set_equality_l2943_294340


namespace NUMINAMATH_CALUDE_cubic_factor_implies_c_zero_l2943_294384

/-- Represents a cubic polynomial of the form ax^3 + bx^2 + cx + d -/
structure CubicPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Represents a quadratic polynomial of the form ax^2 + bx + c -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a linear polynomial of the form ax + b -/
structure LinearPolynomial where
  a : ℝ
  b : ℝ

def has_factor (p : CubicPolynomial) (q : QuadraticPolynomial) : Prop :=
  ∃ l : LinearPolynomial, 
    p.a * (q.a * l.a) = p.a ∧
    p.b * (q.a * l.b + q.b * l.a) = p.b ∧
    p.c * (q.b * l.b + q.c * l.a) = p.c ∧
    p.d * (q.c * l.b) = p.d

theorem cubic_factor_implies_c_zero 
  (p : CubicPolynomial) 
  (h : p.a = 3 ∧ p.b = 0 ∧ p.d = 12) 
  (q : QuadraticPolynomial) 
  (hq : q.a = 1 ∧ q.c = 2) 
  (h_factor : has_factor p q) : 
  p.c = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_factor_implies_c_zero_l2943_294384


namespace NUMINAMATH_CALUDE_probability_no_three_consecutive_ones_l2943_294317

/-- Represents a sequence of 0s and 1s of length n that doesn't contain three consecutive 1s -/
def ValidSequence (n : ℕ) := Fin n → Fin 2

/-- The number of valid sequences of length n -/
def countValidSequences : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | 2 => 4
  | n + 3 => countValidSequences (n + 2) + countValidSequences (n + 1) + countValidSequences n

/-- The total number of possible binary sequences of length n -/
def totalSequences (n : ℕ) : ℕ := 2^n

theorem probability_no_three_consecutive_ones :
  (countValidSequences 15 : ℚ) / (totalSequences 15 : ℚ) = 10609 / 32768 := by
  sorry

#eval countValidSequences 15 + totalSequences 15

end NUMINAMATH_CALUDE_probability_no_three_consecutive_ones_l2943_294317


namespace NUMINAMATH_CALUDE_square_of_product_l2943_294301

theorem square_of_product (a b : ℝ) : (a * b)^2 = a^2 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_product_l2943_294301


namespace NUMINAMATH_CALUDE_root_sum_ratio_l2943_294368

theorem root_sum_ratio (m₁ m₂ : ℝ) : 
  (∃ a b : ℝ, m₁ * (a^2 - 2*a) + 3*a + 7 = 0 ∧ 
              m₂ * (b^2 - 2*b) + 3*b + 7 = 0 ∧ 
              a/b + b/a = 9/10) →
  m₁/m₂ + m₂/m₁ = ((323/40)^2 * 4 - 18) / 9 :=
by sorry

end NUMINAMATH_CALUDE_root_sum_ratio_l2943_294368


namespace NUMINAMATH_CALUDE_straight_insertion_sort_four_steps_l2943_294376

def initial_sequence : List Int := [7, 1, 3, 12, 8, 4, 9, 10]

def straight_insertion_sort (list : List Int) : List Int :=
  sorry

def first_four_steps (list : List Int) : List Int :=
  (straight_insertion_sort list).take 4

theorem straight_insertion_sort_four_steps :
  first_four_steps initial_sequence = [1, 3, 4, 7, 8, 12, 9, 10] :=
sorry

end NUMINAMATH_CALUDE_straight_insertion_sort_four_steps_l2943_294376


namespace NUMINAMATH_CALUDE_november_savings_l2943_294352

def september_savings : ℕ := 50
def october_savings : ℕ := 37
def mom_gift : ℕ := 25
def video_game_cost : ℕ := 87
def money_left : ℕ := 36

theorem november_savings :
  ∃ (november_savings : ℕ),
    september_savings + october_savings + november_savings + mom_gift - video_game_cost = money_left ∧
    november_savings = 11 :=
sorry

end NUMINAMATH_CALUDE_november_savings_l2943_294352


namespace NUMINAMATH_CALUDE_equality_comparison_l2943_294356

theorem equality_comparison : 
  (-2^2 ≠ (-2)^2) ∧ 
  (2^3 ≠ 3^2) ∧ 
  (-3^3 = (-3)^3) ∧ 
  ((-3 * 2)^2 ≠ -3^2 * 2^2) := by
  sorry

end NUMINAMATH_CALUDE_equality_comparison_l2943_294356


namespace NUMINAMATH_CALUDE_arcsin_of_one_equals_pi_div_two_l2943_294395

theorem arcsin_of_one_equals_pi_div_two : Real.arcsin 1 = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_of_one_equals_pi_div_two_l2943_294395


namespace NUMINAMATH_CALUDE_polygon_sides_from_angle_sum_l2943_294375

/-- The number of sides of a polygon given the sum of its interior angles -/
theorem polygon_sides_from_angle_sum (angle_sum : ℝ) : angle_sum = 1260 → ∃ n : ℕ, n = 9 ∧ (n - 2) * 180 = angle_sum := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_from_angle_sum_l2943_294375


namespace NUMINAMATH_CALUDE_constant_value_l2943_294308

theorem constant_value (t : ℝ) (c : ℝ) : 
  let x := 1 - 3 * t
  let y := 2 * t - c
  (x = y ∧ t = 0.8) → c = 3 := by
sorry

end NUMINAMATH_CALUDE_constant_value_l2943_294308


namespace NUMINAMATH_CALUDE_supplementary_to_complementary_ratio_l2943_294379

/-- 
Given an angle of 45 degrees, prove that the ratio of its supplementary angle 
to its complementary angle is 3:1.
-/
theorem supplementary_to_complementary_ratio 
  (angle : ℝ) 
  (h_angle : angle = 45) 
  (h_supplementary : ℝ → ℝ → Prop)
  (h_complementary : ℝ → ℝ → Prop)
  (h_supp_def : ∀ x y, h_supplementary x y ↔ x + y = 180)
  (h_comp_def : ∀ x y, h_complementary x y ↔ x + y = 90) :
  (180 - angle) / (90 - angle) = 3 := by
sorry

end NUMINAMATH_CALUDE_supplementary_to_complementary_ratio_l2943_294379


namespace NUMINAMATH_CALUDE_beverlys_bottle_caps_l2943_294351

def bottle_caps_per_box : ℝ := 35.0
def total_bottle_caps : ℕ := 245

theorem beverlys_bottle_caps :
  (total_bottle_caps : ℝ) / bottle_caps_per_box = 7 :=
sorry

end NUMINAMATH_CALUDE_beverlys_bottle_caps_l2943_294351


namespace NUMINAMATH_CALUDE_lcm_20_45_75_l2943_294358

theorem lcm_20_45_75 : Nat.lcm 20 (Nat.lcm 45 75) = 900 := by
  sorry

end NUMINAMATH_CALUDE_lcm_20_45_75_l2943_294358


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sums_l2943_294347

def P (x : ℝ) : ℝ := (2*x^2 - 2*x + 1)^17 * (3*x^2 - 3*x + 1)^17

theorem polynomial_coefficient_sums :
  (∀ x, P x = P 1) ∧
  (∀ x, (P x + P (-x)) / 2 = (1 + 35^17) / 2) ∧
  (∀ x, (P x - P (-x)) / 2 = (1 - 35^17) / 2) := by sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sums_l2943_294347


namespace NUMINAMATH_CALUDE_largest_n_two_solutions_exceed_two_l2943_294303

/-- The cubic polynomial in question -/
def f (n : ℤ) (x : ℝ) : ℝ :=
  x^3 - (n + 9 : ℝ) * x^2 + (2 * n^2 - 3 * n - 34 : ℝ) * x + 2 * (n - 4) * (n + 3 : ℝ)

/-- The statement that 8 is the largest integer for which the equation has two solutions > 2 -/
theorem largest_n_two_solutions_exceed_two :
  ∀ n : ℤ, (∃ x y : ℝ, x > 2 ∧ y > 2 ∧ x ≠ y ∧ f n x = 0 ∧ f n y = 0) → n ≤ 8 :=
sorry

end NUMINAMATH_CALUDE_largest_n_two_solutions_exceed_two_l2943_294303


namespace NUMINAMATH_CALUDE_triangle_area_proof_l2943_294394

/-- The length of a li in meters -/
def li_to_meters : ℝ := 500

/-- The sides of the triangle in li -/
def side1 : ℝ := 5
def side2 : ℝ := 12
def side3 : ℝ := 13

/-- The area of the triangle in square kilometers -/
def triangle_area : ℝ := 7.5

theorem triangle_area_proof :
  let side1_m := side1 * li_to_meters
  let side2_m := side2 * li_to_meters
  let side3_m := side3 * li_to_meters
  side1_m ^ 2 + side2_m ^ 2 = side3_m ^ 2 →
  (1 / 2) * side1_m * side2_m / 1000000 = triangle_area := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_proof_l2943_294394


namespace NUMINAMATH_CALUDE_second_order_implies_first_order_l2943_294342

/-- A function f: ℝ → ℝ is increasing on an interval D if for any x, y ∈ D, x < y implies f(x) < f(y) -/
def IncreasingOn (f : ℝ → ℝ) (D : Set ℝ) : Prop :=
  ∀ x y, x ∈ D → y ∈ D → x < y → f x < f y

/-- x₀ is a second-order fixed point of f if f(f(x₀)) = x₀ -/
def SecondOrderFixedPoint (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  f (f x₀) = x₀

/-- x₀ is a first-order fixed point of f if f(x₀) = x₀ -/
def FirstOrderFixedPoint (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  f x₀ = x₀

theorem second_order_implies_first_order
    (f : ℝ → ℝ) (D : Set ℝ) (x₀ : ℝ)
    (h_inc : IncreasingOn f D)
    (h_x₀ : x₀ ∈ D)
    (h_second : SecondOrderFixedPoint f x₀) :
    FirstOrderFixedPoint f x₀ := by
  sorry

end NUMINAMATH_CALUDE_second_order_implies_first_order_l2943_294342


namespace NUMINAMATH_CALUDE_smallest_factorial_divisible_by_7875_l2943_294353

def is_factor (a b : ℕ) : Prop := b % a = 0

theorem smallest_factorial_divisible_by_7875 :
  ∃ (n : ℕ), (n > 0) ∧ (is_factor 7875 (Nat.factorial n)) ∧
  (∀ (m : ℕ), m > 0 → m < n → ¬(is_factor 7875 (Nat.factorial m))) ∧
  n = 15 := by
sorry

end NUMINAMATH_CALUDE_smallest_factorial_divisible_by_7875_l2943_294353


namespace NUMINAMATH_CALUDE_unique_number_equality_l2943_294307

theorem unique_number_equality : ∃! x : ℝ, (x / 2) + 6 = 2 * x - 6 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_equality_l2943_294307


namespace NUMINAMATH_CALUDE_circle_radius_from_arc_and_angle_l2943_294382

/-- Given an arc length of 4 and a central angle of 2 radians, the radius of the circle is 2. -/
theorem circle_radius_from_arc_and_angle (arc_length : ℝ) (central_angle : ℝ) (radius : ℝ) 
    (h1 : arc_length = 4)
    (h2 : central_angle = 2)
    (h3 : arc_length = radius * central_angle) : 
  radius = 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_from_arc_and_angle_l2943_294382


namespace NUMINAMATH_CALUDE_sum_of_roots_equation_l2943_294309

theorem sum_of_roots_equation (x : ℝ) : 
  let f : ℝ → ℝ := λ x => x^3 - 3*x^2 - 10*x - 7*(x + 2)
  (∃ a b : ℝ, (∀ x, f x = (x - a) * (x - b) * (x + 2))) →
  a + b = 5 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_equation_l2943_294309


namespace NUMINAMATH_CALUDE_water_fraction_after_four_replacements_l2943_294331

/-- Represents the state of the water tank -/
structure TankState where
  water : ℚ
  antifreeze : ℚ

/-- Performs one replacement operation on the tank -/
def replace (state : TankState) : TankState :=
  let removed := state.water * (5 / 20) + state.antifreeze * (5 / 20)
  { water := state.water - removed + 2.5,
    antifreeze := state.antifreeze - removed + 2.5 }

/-- The initial state of the tank -/
def initialState : TankState :=
  { water := 20, antifreeze := 0 }

/-- Performs n replacements on the tank -/
def nReplacements (n : ℕ) : TankState :=
  match n with
  | 0 => initialState
  | n + 1 => replace (nReplacements n)

theorem water_fraction_after_four_replacements :
  (nReplacements 4).water / ((nReplacements 4).water + (nReplacements 4).antifreeze) = 21 / 32 :=
by sorry

end NUMINAMATH_CALUDE_water_fraction_after_four_replacements_l2943_294331


namespace NUMINAMATH_CALUDE_bananas_per_box_l2943_294387

/-- Given 40 bananas and 8 boxes, prove that the number of bananas per box is 5. -/
theorem bananas_per_box (total_bananas : ℕ) (num_boxes : ℕ) 
  (h1 : total_bananas = 40) (h2 : num_boxes = 8) : 
  total_bananas / num_boxes = 5 := by
  sorry

end NUMINAMATH_CALUDE_bananas_per_box_l2943_294387


namespace NUMINAMATH_CALUDE_notebook_cost_l2943_294322

/-- The cost of a purchase given the number of notebooks, number of pencils, and total paid -/
def purchase_cost (notebooks : ℕ) (pencils : ℕ) (total_paid : ℚ) : ℚ := total_paid

/-- The theorem stating the cost of each notebook -/
theorem notebook_cost :
  ∀ (notebook_price pencil_price : ℚ),
    purchase_cost 5 4 20 - 3.5 = 5 * notebook_price + 4 * pencil_price →
    purchase_cost 2 2 7 = 2 * notebook_price + 2 * pencil_price →
    notebook_price = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_notebook_cost_l2943_294322


namespace NUMINAMATH_CALUDE_exists_valid_five_by_five_division_l2943_294313

/-- Represents a square grid -/
structure SquareGrid :=
  (side : ℕ)

/-- Represents a division of a square grid -/
structure GridDivision :=
  (grid : SquareGrid)
  (num_parts : ℕ)
  (segment_length : ℕ)

/-- Checks if a division of a square grid is valid -/
def is_valid_division (d : GridDivision) : Prop :=
  d.grid.side * d.grid.side % d.num_parts = 0 ∧
  d.segment_length ≤ 16

/-- Theorem: There exists a valid division of a 5x5 square grid into 5 equal parts
    with total segment length not exceeding 16 units -/
theorem exists_valid_five_by_five_division :
  ∃ (d : GridDivision), d.grid.side = 5 ∧ d.num_parts = 5 ∧ is_valid_division d :=
sorry

end NUMINAMATH_CALUDE_exists_valid_five_by_five_division_l2943_294313


namespace NUMINAMATH_CALUDE_monotonicity_of_f_l2943_294366

def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + 1

theorem monotonicity_of_f (a : ℝ) :
  (a > 0 → (∀ x y, x < y → x < -2*a/3 → f a x < f a y) ∧
            (∀ x y, x < y → 0 < x → f a x < f a y) ∧
            (∀ x y, -2*a/3 < x → x < y → y < 0 → f a x > f a y)) ∧
  (a = 0 → (∀ x y, x < y → f a x < f a y)) ∧
  (a < 0 → (∀ x y, x < y → y < 0 → f a x < f a y) ∧
            (∀ x y, x < y → -2*a/3 < x → f a x < f a y) ∧
            (∀ x y, 0 < x → x < y → y < -2*a/3 → f a x > f a y)) :=
by sorry

end NUMINAMATH_CALUDE_monotonicity_of_f_l2943_294366


namespace NUMINAMATH_CALUDE_animal_farm_count_l2943_294357

theorem animal_farm_count (total_legs : ℕ) (chicken_count : ℕ) : 
  total_legs = 26 →
  chicken_count = 5 →
  ∃ (buffalo_count : ℕ),
    2 * chicken_count + 4 * buffalo_count = total_legs ∧
    chicken_count + buffalo_count = 9 :=
by sorry

end NUMINAMATH_CALUDE_animal_farm_count_l2943_294357


namespace NUMINAMATH_CALUDE_square_sum_of_coefficients_l2943_294360

theorem square_sum_of_coefficients (a b c : ℝ) : 
  36 - 4 * Real.sqrt 2 - 6 * Real.sqrt 3 + 12 * Real.sqrt 6 = (a * Real.sqrt 2 + b * Real.sqrt 3 + c)^2 →
  a^2 + b^2 + c^2 = 14 := by
sorry

end NUMINAMATH_CALUDE_square_sum_of_coefficients_l2943_294360


namespace NUMINAMATH_CALUDE_four_digit_permutations_2033_eq_18_l2943_294339

/-- The number of unique four-digit permutations of the digits in 2033 -/
def four_digit_permutations_2033 : ℕ := 18

/-- The set of digits in 2033 -/
def digits_2033 : Finset ℕ := {0, 2, 3}

/-- The function to count valid permutations -/
def count_valid_permutations (digits : Finset ℕ) : ℕ :=
  sorry

theorem four_digit_permutations_2033_eq_18 :
  count_valid_permutations digits_2033 = four_digit_permutations_2033 :=
by sorry

end NUMINAMATH_CALUDE_four_digit_permutations_2033_eq_18_l2943_294339


namespace NUMINAMATH_CALUDE_beatty_theorem_l2943_294392

theorem beatty_theorem (α β : ℝ) (hα : Irrational α) (hβ : Irrational β) 
  (hpos_α : α > 0) (hpos_β : β > 0) (h_sum : 1/α + 1/β = 1) :
  (∀ k : ℕ+, ∃! n : ℕ+, k = ⌊n * α⌋ ∨ k = ⌊n * β⌋) ∧ 
  (∀ k : ℕ+, ¬(∃ m n : ℕ+, k = ⌊m * α⌋ ∧ k = ⌊n * β⌋)) := by
  sorry

end NUMINAMATH_CALUDE_beatty_theorem_l2943_294392


namespace NUMINAMATH_CALUDE_trig_function_problem_l2943_294336

theorem trig_function_problem (f : ℝ → ℝ) (h : ∀ x, f (Real.cos x) = Real.cos (2 * x)) :
  f (Real.sin (π / 12)) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_function_problem_l2943_294336


namespace NUMINAMATH_CALUDE_complex_simplification_l2943_294328

theorem complex_simplification : 
  3 * (4 - 2 * Complex.I) + 2 * Complex.I * (3 + Complex.I) = (10 : ℂ) := by
  sorry

end NUMINAMATH_CALUDE_complex_simplification_l2943_294328


namespace NUMINAMATH_CALUDE_round_trip_average_speed_l2943_294306

/-- Calculates the average speed for a round trip boat journey on a river -/
theorem round_trip_average_speed
  (upstream_speed : ℝ)
  (downstream_speed : ℝ)
  (river_current : ℝ)
  (h1 : upstream_speed = 4)
  (h2 : downstream_speed = 7)
  (h3 : river_current = 2)
  : (2 / ((1 / (upstream_speed - river_current)) + (1 / (downstream_speed + river_current)))) = 36 / 11 := by
  sorry

end NUMINAMATH_CALUDE_round_trip_average_speed_l2943_294306


namespace NUMINAMATH_CALUDE_total_out_of_pocket_cost_l2943_294348

/-- Calculates the total out-of-pocket cost for medical treatment --/
theorem total_out_of_pocket_cost 
  (doctor_visit_cost : ℕ) 
  (cast_cost : ℕ) 
  (initial_insurance_coverage : ℚ) 
  (pt_sessions : ℕ) 
  (pt_cost_per_session : ℕ) 
  (pt_insurance_coverage : ℚ) : 
  doctor_visit_cost = 300 →
  cast_cost = 200 →
  initial_insurance_coverage = 60 / 100 →
  pt_sessions = 8 →
  pt_cost_per_session = 100 →
  pt_insurance_coverage = 40 / 100 →
  (1 - initial_insurance_coverage) * (doctor_visit_cost + cast_cost) +
  (1 - pt_insurance_coverage) * (pt_sessions * pt_cost_per_session) = 680 := by
sorry

end NUMINAMATH_CALUDE_total_out_of_pocket_cost_l2943_294348


namespace NUMINAMATH_CALUDE_fraction_equality_l2943_294374

theorem fraction_equality (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) 
  (h3 : (4*a + 2*b) / (2*a - 4*b) = 3) : 
  (2*a + 4*b) / (4*a - 2*b) = 9/13 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2943_294374


namespace NUMINAMATH_CALUDE_min_sum_squares_l2943_294326

theorem min_sum_squares (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3*x*y*z = 8) :
  ∃ (min : ℝ), min = 16/3 ∧ x^2 + y^2 + z^2 ≥ min ∧ ∃ (x₀ y₀ z₀ : ℝ), x₀^3 + y₀^3 + z₀^3 - 3*x₀*y₀*z₀ = 8 ∧ x₀^2 + y₀^2 + z₀^2 = min :=
sorry

end NUMINAMATH_CALUDE_min_sum_squares_l2943_294326


namespace NUMINAMATH_CALUDE_james_total_earnings_l2943_294310

def january_earnings : ℝ := 4000

def february_earnings (jan : ℝ) : ℝ := jan * 1.5

def march_earnings (feb : ℝ) : ℝ := feb * 0.8

def total_earnings (jan feb mar : ℝ) : ℝ := jan + feb + mar

theorem james_total_earnings :
  let feb := february_earnings january_earnings
  let mar := march_earnings feb
  total_earnings january_earnings feb mar = 14800 := by sorry

end NUMINAMATH_CALUDE_james_total_earnings_l2943_294310


namespace NUMINAMATH_CALUDE_true_discount_proof_l2943_294371

/-- Calculates the true discount given the banker's discount and sum due -/
def true_discount (bankers_discount : ℚ) (sum_due : ℚ) : ℚ :=
  let a : ℚ := 1
  let b : ℚ := sum_due
  let c : ℚ := -sum_due * bankers_discount
  (-b + (b^2 - 4*a*c).sqrt) / (2*a)

/-- Proves that the true discount is 246 given the banker's discount of 288 and sum due of 1440 -/
theorem true_discount_proof (bankers_discount sum_due : ℚ) 
  (h1 : bankers_discount = 288)
  (h2 : sum_due = 1440) : 
  true_discount bankers_discount sum_due = 246 := by
  sorry

#eval true_discount 288 1440

end NUMINAMATH_CALUDE_true_discount_proof_l2943_294371


namespace NUMINAMATH_CALUDE_trig_identity_quadratic_solution_l2943_294362

-- Part 1
theorem trig_identity : 
  Real.tan (π / 6) ^ 2 + 2 * Real.sin (π / 4) - 2 * Real.cos (π / 3) = (3 * Real.sqrt 2 - 2) / 3 := by
  sorry

-- Part 2
theorem quadratic_solution :
  let x₁ := (-2 + Real.sqrt 2) / 2
  let x₂ := (-2 - Real.sqrt 2) / 2
  2 * x₁ ^ 2 + 4 * x₁ + 1 = 0 ∧ 2 * x₂ ^ 2 + 4 * x₂ + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_quadratic_solution_l2943_294362


namespace NUMINAMATH_CALUDE_orange_harvest_per_day_l2943_294314

/-- Given a consistent daily harvest of oranges over 6 days resulting in 498 sacks,
    prove that the daily harvest is 83 sacks. -/
theorem orange_harvest_per_day :
  ∀ (daily_harvest : ℕ),
  daily_harvest * 6 = 498 →
  daily_harvest = 83 := by
  sorry

end NUMINAMATH_CALUDE_orange_harvest_per_day_l2943_294314


namespace NUMINAMATH_CALUDE_exactlyTwoVisitCount_l2943_294333

/-- Represents a visitor with a visiting frequency -/
structure Visitor where
  frequency : ℕ

/-- Calculates the number of days when exactly two out of three visitors visit -/
def exactlyTwoVisit (v1 v2 v3 : Visitor) (days : ℕ) : ℕ :=
  sorry

theorem exactlyTwoVisitCount :
  let alice : Visitor := ⟨2⟩
  let beatrix : Visitor := ⟨5⟩
  let claire : Visitor := ⟨7⟩
  exactlyTwoVisit alice beatrix claire 365 = 55 := by sorry

end NUMINAMATH_CALUDE_exactlyTwoVisitCount_l2943_294333


namespace NUMINAMATH_CALUDE_cody_initial_tickets_cody_initial_tickets_proof_l2943_294345

/-- Theorem: Cody's initial number of tickets
Given:
- Cody lost 6.0 tickets
- Cody spent 25.0 tickets
- Cody has 18 tickets left
Prove: Cody's initial number of tickets was 49.0
-/
theorem cody_initial_tickets : ℝ → Prop :=
  fun initial_tickets =>
    let lost_tickets : ℝ := 6.0
    let spent_tickets : ℝ := 25.0
    let remaining_tickets : ℝ := 18.0
    initial_tickets = lost_tickets + spent_tickets + remaining_tickets
    ∧ initial_tickets = 49.0

/-- Proof of the theorem -/
theorem cody_initial_tickets_proof : cody_initial_tickets 49.0 := by
  sorry

end NUMINAMATH_CALUDE_cody_initial_tickets_cody_initial_tickets_proof_l2943_294345


namespace NUMINAMATH_CALUDE_watch_correction_theorem_l2943_294389

/-- Represents the number of days between two dates -/
def daysBetween (startDate endDate : Nat) : Nat :=
  endDate - startDate

/-- Represents the number of hours from noon to 10 AM the next day -/
def hoursFromNoonTo10AM : Nat := 22

/-- Represents the rate at which the watch loses time in minutes per day -/
def watchLossRate : Rat := 13/4

/-- Calculates the total hours elapsed on the watch -/
def totalWatchHours (days : Nat) : Nat :=
  days * 24 + hoursFromNoonTo10AM

/-- Calculates the total time loss in minutes -/
def totalTimeLoss (hours : Nat) (lossRate : Rat) : Rat :=
  (hours : Rat) * (lossRate / 24)

/-- The initial time difference when the watch was set -/
def initialTimeDifference : Rat := 10

/-- Theorem: The positive correction to be added to the watch time is 35.7292 minutes -/
theorem watch_correction_theorem (startDate endDate : Nat) :
  let days := daysBetween startDate endDate
  let hours := totalWatchHours days
  let loss := totalTimeLoss hours watchLossRate
  loss + initialTimeDifference = 35.7292 := by sorry

end NUMINAMATH_CALUDE_watch_correction_theorem_l2943_294389


namespace NUMINAMATH_CALUDE_rock_paper_scissors_winning_probability_l2943_294311

/-- Represents the possible outcomes of a single round of Rock, Paper, Scissors -/
inductive RockPaperScissorsOutcome
  | Win
  | Lose
  | Draw

/-- Represents a two-player game of Rock, Paper, Scissors -/
structure RockPaperScissors where
  player1 : String
  player2 : String

/-- The probability of winning for each player in Rock, Paper, Scissors -/
def winningProbability (game : RockPaperScissors) : ℚ :=
  1 / 3

/-- Theorem: The probability of winning for each player in Rock, Paper, Scissors is 1/3 -/
theorem rock_paper_scissors_winning_probability (game : RockPaperScissors) :
  winningProbability game = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_rock_paper_scissors_winning_probability_l2943_294311


namespace NUMINAMATH_CALUDE_cloth_woven_approx_15_meters_l2943_294370

/-- The rate at which the loom weaves cloth in meters per second -/
def weaving_rate : ℝ := 0.127

/-- The time taken by the loom to weave the cloth in seconds -/
def weaving_time : ℝ := 118.11

/-- The amount of cloth woven in meters -/
def cloth_woven : ℝ := weaving_rate * weaving_time

/-- Theorem stating that the amount of cloth woven is approximately 15 meters -/
theorem cloth_woven_approx_15_meters : 
  ∃ ε > 0, |cloth_woven - 15| < ε := by sorry

end NUMINAMATH_CALUDE_cloth_woven_approx_15_meters_l2943_294370


namespace NUMINAMATH_CALUDE_power_of_power_l2943_294377

theorem power_of_power (a : ℝ) : (a^2)^3 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l2943_294377


namespace NUMINAMATH_CALUDE_inequality_solution_l2943_294397

theorem inequality_solution : ∃! x : ℝ, 
  (Real.sqrt (x^3 - 10*x + 7) + 1) * abs (x^3 - 18*x + 28) ≤ 0 ∧
  x^3 - 10*x + 7 ≥ 0 :=
by
  -- The unique solution is x = -1 + √15
  use -1 + Real.sqrt 15
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l2943_294397


namespace NUMINAMATH_CALUDE_palindrome_existence_l2943_294346

/-- A number is a palindrome if it reads the same backwards and forwards in its decimal representation -/
def IsPalindrome (m : ℕ) : Prop :=
  ∃ (digits : List ℕ), m = digits.foldl (fun acc d => 10 * acc + d) 0 ∧ digits = digits.reverse

/-- For any natural number n, there exists a natural number N such that 9 * 5^n * N is a palindrome -/
theorem palindrome_existence (n : ℕ) : ∃ (N : ℕ), IsPalindrome (9 * 5^n * N) := by
  sorry

end NUMINAMATH_CALUDE_palindrome_existence_l2943_294346


namespace NUMINAMATH_CALUDE_system_equations_proof_l2943_294312

theorem system_equations_proof (a x y : ℝ) : 
  x + y = -7 - a → 
  x - y = 1 + 3*a → 
  x ≤ 0 → 
  y < 0 → 
  (-2 < a ∧ a ≤ 3) ∧ 
  (abs (a - 3) + abs (a + 2) = 5) ∧ 
  (∀ (a : ℤ), -2 < a ∧ a ≤ 3 → (∀ x, 2*a*x + x > 2*a + 1 ↔ x < 1) ↔ a = -1) :=
by sorry

end NUMINAMATH_CALUDE_system_equations_proof_l2943_294312


namespace NUMINAMATH_CALUDE_intersection_implies_x_value_l2943_294383

def A (x : ℝ) : Set ℝ := {9, 2 - x, x^2 + 1}
def B (x : ℝ) : Set ℝ := {1, 2 * x^2}

theorem intersection_implies_x_value :
  ∀ x : ℝ, A x ∩ B x = {2} → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_x_value_l2943_294383


namespace NUMINAMATH_CALUDE_complex_number_problem_l2943_294399

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the property of being a purely imaginary number
def isPurelyImaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

-- Define the property of being a real number
def isRealNumber (z : ℂ) : Prop := z.im = 0

-- Theorem statement
theorem complex_number_problem (z : ℂ) 
  (h1 : isPurelyImaginary z) 
  (h2 : isRealNumber ((z + 2) / (1 - i))) : 
  z = -2 * i := by
  sorry

end NUMINAMATH_CALUDE_complex_number_problem_l2943_294399


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_same_foci_l2943_294369

/-- Given an ellipse and a hyperbola with the same foci, prove that the semi-major axis of the ellipse is 4 -/
theorem ellipse_hyperbola_same_foci (a : ℝ) : a > 0 →
  (∀ x y : ℝ, x^2 / a^2 + y^2 / 9 = 1) →
  (∀ x y : ℝ, x^2 / 4 - y^2 / 3 = 1) →
  (∀ c : ℝ, c^2 = 7 → 
    (∀ x y : ℝ, x^2 / a^2 + y^2 / 9 = 1 → (x + c)^2 + y^2 = a^2 ∧ (x - c)^2 + y^2 = a^2) ∧
    (∀ x y : ℝ, x^2 / 4 - y^2 / 3 = 1 → (x + c)^2 - y^2 = 4 ∧ (x - c)^2 - y^2 = 4)) →
  a = 4 := by
sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_same_foci_l2943_294369


namespace NUMINAMATH_CALUDE_painted_cube_probability_l2943_294373

/-- Represents a cube with painted faces --/
structure PaintedCube where
  size : ℕ
  painted_faces : ℕ

/-- Calculates the number of unit cubes with exactly three painted faces --/
def num_three_painted_faces (cube : PaintedCube) : ℕ :=
  if cube.painted_faces = 2 then 4 else 0

/-- Calculates the number of unit cubes with no painted faces --/
def num_no_painted_faces (cube : PaintedCube) : ℕ :=
  (cube.size - 2) ^ 3

/-- Calculates the total number of unit cubes --/
def total_unit_cubes (cube : PaintedCube) : ℕ :=
  cube.size ^ 3

/-- Calculates the number of ways to choose 2 cubes from the total --/
def choose_two (n : ℕ) : ℕ :=
  n * (n - 1) / 2

/-- Theorem: The probability of selecting one unit cube with three painted faces
    and one with no painted faces is 9/646 for a 5x5x5 cube with two adjacent
    painted faces --/
theorem painted_cube_probability (cube : PaintedCube)
    (h1 : cube.size = 5)
    (h2 : cube.painted_faces = 2) :
    (num_three_painted_faces cube * num_no_painted_faces cube : ℚ) /
    choose_two (total_unit_cubes cube) = 9 / 646 := by
  sorry

end NUMINAMATH_CALUDE_painted_cube_probability_l2943_294373


namespace NUMINAMATH_CALUDE_polynomial_remainder_theorem_l2943_294396

theorem polynomial_remainder_theorem (a b : ℝ) : 
  let f : ℝ → ℝ := λ x => a * x^3 - 3 * x^2 + b * x - 7
  (f 2 = -17) → (f (-1) = -11) → (a = 0 ∧ b = -1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_theorem_l2943_294396


namespace NUMINAMATH_CALUDE_land_conversion_equation_l2943_294330

/-- Represents the land conversion scenario in a village --/
theorem land_conversion_equation (x : ℝ) : 
  (54 - x = (20 / 100) * (108 + x)) ↔ 
  (54 - x = 0.2 * (108 + x) ∧ 
   0 ≤ x ∧ 
   x ≤ 54 ∧
   108 + x > 0) := by
  sorry

end NUMINAMATH_CALUDE_land_conversion_equation_l2943_294330


namespace NUMINAMATH_CALUDE_kitty_dusting_time_l2943_294391

/-- Represents the cleaning activities and their durations in Kitty's living room --/
structure CleaningActivities where
  pickingUpToys : ℕ
  vacuuming : ℕ
  cleaningWindows : ℕ
  totalWeeks : ℕ
  totalMinutes : ℕ

/-- Calculates the time spent dusting furniture each week --/
def dustingTime (c : CleaningActivities) : ℕ :=
  let otherTasksTime := c.pickingUpToys + c.vacuuming + c.cleaningWindows
  let totalOtherTasksTime := otherTasksTime * c.totalWeeks
  let totalDustingTime := c.totalMinutes - totalOtherTasksTime
  totalDustingTime / c.totalWeeks

/-- Theorem stating that Kitty spends 10 minutes each week dusting furniture --/
theorem kitty_dusting_time :
  ∀ (c : CleaningActivities),
    c.pickingUpToys = 5 →
    c.vacuuming = 20 →
    c.cleaningWindows = 15 →
    c.totalWeeks = 4 →
    c.totalMinutes = 200 →
    dustingTime c = 10 := by
  sorry

end NUMINAMATH_CALUDE_kitty_dusting_time_l2943_294391


namespace NUMINAMATH_CALUDE_cost_per_side_l2943_294349

-- Define the park as a square
structure SquarePark where
  side_cost : ℝ
  total_cost : ℝ

-- Define the properties of the square park
def is_valid_square_park (park : SquarePark) : Prop :=
  park.total_cost = 224 ∧ park.total_cost = 4 * park.side_cost

-- Theorem statement
theorem cost_per_side (park : SquarePark) (h : is_valid_square_park park) : 
  park.side_cost = 56 := by
  sorry

end NUMINAMATH_CALUDE_cost_per_side_l2943_294349


namespace NUMINAMATH_CALUDE_heartsuit_three_eight_l2943_294350

-- Define the operation ⊛
def heartsuit (x y : ℝ) : ℝ := 4 * x + 6 * y

-- Theorem statement
theorem heartsuit_three_eight : heartsuit 3 8 = 60 := by
  sorry

end NUMINAMATH_CALUDE_heartsuit_three_eight_l2943_294350


namespace NUMINAMATH_CALUDE_negation_existence_real_gt_one_l2943_294335

theorem negation_existence_real_gt_one :
  (¬ ∃ x : ℝ, x > 1) ↔ (∀ x : ℝ, x ≤ 1) := by sorry

end NUMINAMATH_CALUDE_negation_existence_real_gt_one_l2943_294335


namespace NUMINAMATH_CALUDE_six_digit_increase_characterization_l2943_294325

def is_six_digit (n : ℕ) : Prop := 100000 ≤ n ∧ n ≤ 999999

def last_digit (n : ℕ) : ℕ := n % 10

def move_last_to_first (n : ℕ) : ℕ :=
  (n / 10) + (last_digit n * 100000)

def increases_by_integer_factor (n : ℕ) : Prop :=
  ∃ k : ℕ, k > 0 ∧ move_last_to_first n = k * n

def S : Set ℕ := {111111, 222222, 333333, 444444, 555555, 666666, 777777, 888888, 999999, 
                  142857, 102564, 128205, 153846, 179487, 205128, 230769}

theorem six_digit_increase_characterization :
  ∀ n : ℕ, is_six_digit n ∧ increases_by_integer_factor n ↔ n ∈ S :=
sorry

end NUMINAMATH_CALUDE_six_digit_increase_characterization_l2943_294325


namespace NUMINAMATH_CALUDE_square_diff_fourth_power_l2943_294398

theorem square_diff_fourth_power : (7^2 - 3^2)^4 = 2560000 := by
  sorry

end NUMINAMATH_CALUDE_square_diff_fourth_power_l2943_294398


namespace NUMINAMATH_CALUDE_inequality_implication_l2943_294341

theorem inequality_implication (a b : ℝ) (h : a > b) : 2*a - 1 > 2*b - 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l2943_294341


namespace NUMINAMATH_CALUDE_five_saturdays_in_august_l2943_294320

/-- Represents days of the week --/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a month --/
structure Month where
  days : Nat
  first_day : DayOfWeek

/-- July of the given year --/
def july : Month := sorry

/-- August of the given year --/
def august : Month := sorry

/-- Counts the occurrences of a specific day in a month --/
def count_day_occurrences (m : Month) (d : DayOfWeek) : Nat := sorry

/-- The theorem to prove --/
theorem five_saturdays_in_august (h : count_day_occurrences july DayOfWeek.Wednesday = 5) :
  count_day_occurrences august DayOfWeek.Saturday = 5 := by sorry

end NUMINAMATH_CALUDE_five_saturdays_in_august_l2943_294320


namespace NUMINAMATH_CALUDE_unique_integer_solution_fourth_power_equation_l2943_294337

theorem unique_integer_solution_fourth_power_equation :
  ∀ x y : ℤ, x^4 + y^4 = 3*x^3*y → x = 0 ∧ y = 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_integer_solution_fourth_power_equation_l2943_294337


namespace NUMINAMATH_CALUDE_milk_added_to_full_can_l2943_294380

/-- Represents the contents of a can with milk and water -/
structure Can where
  milk : ℝ
  water : ℝ

/-- Represents the ratios of milk to water -/
structure Ratio where
  milk : ℝ
  water : ℝ

def Can.ratio (can : Can) : Ratio :=
  { milk := can.milk, water := can.water }

def Can.total (can : Can) : ℝ :=
  can.milk + can.water

theorem milk_added_to_full_can 
  (initial_ratio : Ratio) 
  (final_ratio : Ratio) 
  (capacity : ℝ) :
  initial_ratio.milk / initial_ratio.water = 4 / 3 →
  final_ratio.milk / final_ratio.water = 2 / 1 →
  capacity = 36 →
  ∃ (initial_can final_can : Can),
    initial_can.ratio = initial_ratio ∧
    final_can.ratio = final_ratio ∧
    final_can.total = capacity ∧
    final_can.water = initial_can.water ∧
    final_can.milk - initial_can.milk = 72 / 7 := by
  sorry

end NUMINAMATH_CALUDE_milk_added_to_full_can_l2943_294380


namespace NUMINAMATH_CALUDE_min_value_of_expression_l2943_294319

theorem min_value_of_expression (x : ℝ) (h : x > 0) :
  x^2 + 12*x + 81/x^3 ≥ 18 * Real.sqrt 3 ∧
  ∃ y > 0, y^2 + 12*y + 81/y^3 = 18 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l2943_294319


namespace NUMINAMATH_CALUDE_quadratic_function_j_value_l2943_294393

/-- A quadratic function with integer coefficients -/
def QuadraticFunction (a b c : ℤ) : ℝ → ℝ := fun x ↦ (a * x^2 : ℝ) + (b * x : ℝ) + (c : ℝ)

theorem quadratic_function_j_value
  (a b c : ℤ)
  (h1 : QuadraticFunction a b c 1 = 0)
  (h2 : QuadraticFunction a b c (-1) = 0)
  (h3 : 70 < QuadraticFunction a b c 7 ∧ QuadraticFunction a b c 7 < 90)
  (h4 : 110 < QuadraticFunction a b c 8 ∧ QuadraticFunction a b c 8 < 140)
  (h5 : ∃ j : ℤ, 1000 * j < QuadraticFunction a b c 50 ∧ QuadraticFunction a b c 50 < 1000 * (j + 1)) :
  ∃ j : ℤ, j = 4 ∧ 1000 * j < QuadraticFunction a b c 50 ∧ QuadraticFunction a b c 50 < 1000 * (j + 1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_j_value_l2943_294393


namespace NUMINAMATH_CALUDE_replaced_person_weight_l2943_294365

/-- Proves that the weight of the replaced person is 55 kg given the conditions -/
theorem replaced_person_weight (initial_count : ℕ) (weight_increase : ℝ) (new_person_weight : ℝ) :
  initial_count = 8 →
  weight_increase = 4 →
  new_person_weight = 87 →
  (initial_count : ℝ) * weight_increase + new_person_weight = 
    (initial_count : ℝ) * weight_increase + 55 := by
  sorry

end NUMINAMATH_CALUDE_replaced_person_weight_l2943_294365


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_property_l2943_294386

theorem quadratic_equation_solution_property (k : ℝ) : 
  (∃ a b : ℝ, 
    (3 * a^2 + 6 * a + k = 0) ∧ 
    (3 * b^2 + 6 * b + k = 0) ∧ 
    (abs (a - b) = 2 * (a^2 + b^2))) ↔ 
  (k = 3 ∨ k = 45/16) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_property_l2943_294386


namespace NUMINAMATH_CALUDE_common_difference_is_three_l2943_294385

/-- Arithmetic sequence with 10 terms -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, n < 9 → a (n + 1) = a n + d

/-- Sum of odd terms is 15 -/
def sum_odd_terms (a : ℕ → ℝ) : Prop :=
  a 1 + a 3 + a 5 + a 7 + a 9 = 15

/-- Sum of even terms is 30 -/
def sum_even_terms (a : ℕ → ℝ) : Prop :=
  a 2 + a 4 + a 6 + a 8 + a 10 = 30

theorem common_difference_is_three (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a) 
  (h2 : sum_odd_terms a) 
  (h3 : sum_even_terms a) : 
  ∃ d : ℝ, d = 3 ∧ ∀ n : ℕ, n < 9 → a (n + 1) = a n + d :=
sorry

end NUMINAMATH_CALUDE_common_difference_is_three_l2943_294385


namespace NUMINAMATH_CALUDE_wrong_height_calculation_l2943_294323

theorem wrong_height_calculation (n : ℕ) (initial_avg real_avg actual_height : ℝ) 
  (h1 : n = 35)
  (h2 : initial_avg = 180)
  (h3 : real_avg = 178)
  (h4 : actual_height = 106)
  : ∃ wrong_height : ℝ,
    (n * initial_avg - wrong_height + actual_height) / n = real_avg ∧ 
    wrong_height = 176 := by
  sorry

end NUMINAMATH_CALUDE_wrong_height_calculation_l2943_294323


namespace NUMINAMATH_CALUDE_range_of_m_l2943_294390

def proposition_p (m : ℝ) : Prop :=
  ∃ x y : ℝ, x^2 / (2*m) - y^2 / (m-1) = 1 ∧ 
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ x^2 / a^2 + y^2 / b^2 = 1 ∧ 
  ∃ c : ℝ, c^2 = a^2 - b^2 ∧ c < a

def proposition_q (m : ℝ) : Prop :=
  ∃ e : ℝ, 1 < e ∧ e < 2 ∧
  ∃ x y : ℝ, y^2 / 5 - x^2 / m = 1 ∧
  e^2 = (5 + m) / 5

theorem range_of_m :
  ∀ m : ℝ, (proposition_p m ∨ proposition_q m) → 0 < m ∧ m < 15 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l2943_294390


namespace NUMINAMATH_CALUDE_purple_gumdrops_after_replacement_l2943_294364

/-- Represents the number of gumdrops of each color in a jar -/
structure GumdropsJar where
  total : ℕ
  orange : ℕ
  purple : ℕ
  yellow : ℕ
  white : ℕ
  black : ℕ

/-- Checks if the distribution of gumdrops is valid -/
def is_valid_distribution (jar : GumdropsJar) : Prop :=
  jar.orange + jar.purple + jar.yellow + jar.white + jar.black = jar.total ∧
  jar.orange = (40 * jar.total) / 100 ∧
  jar.purple = (10 * jar.total) / 100 ∧
  jar.yellow = (25 * jar.total) / 100 ∧
  jar.white = (15 * jar.total) / 100

/-- Replaces one-third of orange gumdrops with purple gumdrops -/
def replace_orange_with_purple (jar : GumdropsJar) : GumdropsJar :=
  let replaced := jar.orange / 3
  { jar with
    orange := jar.orange - replaced
    purple := jar.purple + replaced
  }

/-- Theorem stating that after replacement, there will be 47 purple gumdrops -/
theorem purple_gumdrops_after_replacement (jar : GumdropsJar) :
  is_valid_distribution jar →
  (replace_orange_with_purple jar).purple = 47 := by
  sorry

end NUMINAMATH_CALUDE_purple_gumdrops_after_replacement_l2943_294364


namespace NUMINAMATH_CALUDE_students_in_different_clubs_l2943_294324

/-- The number of clubs in the school -/
def num_clubs : ℕ := 3

/-- The probability of a student joining any specific club -/
def prob_join_club : ℚ := 1 / num_clubs

/-- The probability of two students joining different clubs -/
def prob_different_clubs : ℚ := 2 / 3

theorem students_in_different_clubs :
  prob_different_clubs = 1 - (num_clubs : ℚ) * prob_join_club * prob_join_club := by
  sorry

end NUMINAMATH_CALUDE_students_in_different_clubs_l2943_294324


namespace NUMINAMATH_CALUDE_lcm_of_5_6_10_15_l2943_294316

theorem lcm_of_5_6_10_15 : 
  Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 10 15)) = 30 := by sorry

end NUMINAMATH_CALUDE_lcm_of_5_6_10_15_l2943_294316


namespace NUMINAMATH_CALUDE_trajectory_of_midpoint_l2943_294327

/-- Given a point P on the circle x^2 + y^2 = 16, and M being the midpoint of the perpendicular
    line segment from P to the x-axis, the trajectory of M satisfies the equation x^2/4 + y^2/16 = 1. -/
theorem trajectory_of_midpoint (x₀ y₀ x y : ℝ) : 
  x₀^2 + y₀^2 = 16 →  -- P is on the circle
  x₀ = 2*x →  -- M is the midpoint of PD (x-coordinate)
  y₀ = y →  -- M is the midpoint of PD (y-coordinate)
  x^2/4 + y^2/16 = 1 := by
sorry

end NUMINAMATH_CALUDE_trajectory_of_midpoint_l2943_294327


namespace NUMINAMATH_CALUDE_complex_magnitude_equation_l2943_294343

theorem complex_magnitude_equation (n : ℝ) : 
  (n > 0 ∧ Complex.abs (5 + n * Complex.I) = Real.sqrt 34) → n = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_equation_l2943_294343


namespace NUMINAMATH_CALUDE_jeff_fish_problem_l2943_294321

/-- The problem of finding the maximum mass of a single fish caught by Jeff. -/
theorem jeff_fish_problem (n : ℕ) (min_mass : ℝ) (first_three_mass : ℝ) :
  n = 21 ∧
  min_mass = 0.2 ∧
  first_three_mass = 1.5 ∧
  (∀ fish : ℕ, fish ≤ n → ∃ (mass : ℝ), mass ≥ min_mass) ∧
  (first_three_mass / 3 = (first_three_mass + (n - 3) * min_mass) / n) →
  ∃ (max_mass : ℝ), max_mass = 5.6 ∧ 
    ∀ (fish_mass : ℝ), (∃ (fish : ℕ), fish ≤ n ∧ fish_mass ≥ min_mass) → fish_mass ≤ max_mass :=
by sorry

end NUMINAMATH_CALUDE_jeff_fish_problem_l2943_294321


namespace NUMINAMATH_CALUDE_parabola_focus_for_x_squared_l2943_294361

/-- Represents a parabola of the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The focus of a parabola -/
def focus (p : Parabola) : ℝ × ℝ := sorry

/-- A parabola is symmetric about the y-axis if its equation has no x term -/
def isSymmetricAboutYAxis (p : Parabola) : Prop := p.b = 0

theorem parabola_focus_for_x_squared (p : Parabola) 
  (h1 : p.a = 1) 
  (h2 : p.b = 0) 
  (h3 : p.c = 0) 
  (h4 : isSymmetricAboutYAxis p) : 
  focus p = (0, 1/4) := by sorry

end NUMINAMATH_CALUDE_parabola_focus_for_x_squared_l2943_294361


namespace NUMINAMATH_CALUDE_john_videos_per_day_l2943_294354

/-- Represents the number of videos and their durations for a video creator --/
structure VideoCreator where
  short_videos_per_day : ℕ
  long_videos_per_day : ℕ
  short_video_duration : ℕ
  long_video_duration : ℕ
  days_per_week : ℕ
  total_weekly_minutes : ℕ

/-- Calculates the total number of videos released per day --/
def total_videos_per_day (vc : VideoCreator) : ℕ :=
  vc.short_videos_per_day + vc.long_videos_per_day

/-- Calculates the total minutes of video released per day --/
def total_minutes_per_day (vc : VideoCreator) : ℕ :=
  vc.short_videos_per_day * vc.short_video_duration +
  vc.long_videos_per_day * vc.long_video_duration

/-- Theorem stating that given the conditions, the total number of videos released per day is 3 --/
theorem john_videos_per_day :
  ∀ (vc : VideoCreator),
  vc.short_videos_per_day = 2 →
  vc.long_videos_per_day = 1 →
  vc.short_video_duration = 2 →
  vc.long_video_duration = 6 * vc.short_video_duration →
  vc.days_per_week = 7 →
  vc.total_weekly_minutes = 112 →
  vc.total_weekly_minutes = vc.days_per_week * (total_minutes_per_day vc) →
  total_videos_per_day vc = 3 := by
  sorry

end NUMINAMATH_CALUDE_john_videos_per_day_l2943_294354


namespace NUMINAMATH_CALUDE_inequality_solution_l2943_294338

theorem inequality_solution (x : ℝ) : (x - 2) / (x + 5) ≥ 0 ↔ x < -5 ∨ x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l2943_294338


namespace NUMINAMATH_CALUDE_a_equals_fibonacci_ratio_l2943_294300

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

def a : ℕ → ℕ
  | 0 => 3
  | (n + 1) => (a n)^2 - 2

theorem a_equals_fibonacci_ratio (n : ℕ) :
  a n = fibonacci (2^(n+1)) / fibonacci 4 := by
  sorry

end NUMINAMATH_CALUDE_a_equals_fibonacci_ratio_l2943_294300


namespace NUMINAMATH_CALUDE_max_sum_of_factors_l2943_294388

theorem max_sum_of_factors (P Q R : ℕ+) : 
  P ≠ Q → P ≠ R → Q ≠ R → P * Q * R = 5103 → 
  ∀ (X Y Z : ℕ+), X ≠ Y → X ≠ Z → Y ≠ Z → X * Y * Z = 5103 → 
  P + Q + R ≤ 136 ∧ (∃ (A B C : ℕ+), A ≠ B → A ≠ C → B ≠ C → A * B * C = 5103 ∧ A + B + C = 136) := by
sorry

end NUMINAMATH_CALUDE_max_sum_of_factors_l2943_294388


namespace NUMINAMATH_CALUDE_workshop_workers_l2943_294355

/-- Represents the total number of workers in the workshop -/
def total_workers : ℕ := 15

/-- Represents the number of technicians -/
def technicians : ℕ := 5

/-- Represents the average salary of all workers -/
def avg_salary_all : ℚ := 700

/-- Represents the average salary of technicians -/
def avg_salary_technicians : ℚ := 800

/-- Represents the average salary of the rest of the workers -/
def avg_salary_rest : ℚ := 650

theorem workshop_workers :
  (avg_salary_all * total_workers : ℚ) = 
  (avg_salary_technicians * technicians : ℚ) + 
  (avg_salary_rest * (total_workers - technicians) : ℚ) := by
  sorry

#check workshop_workers

end NUMINAMATH_CALUDE_workshop_workers_l2943_294355


namespace NUMINAMATH_CALUDE_line_segment_endpoint_l2943_294378

/-- Given a line segment with midpoint (-3, 4) and one endpoint (0, 2),
    prove that the other endpoint is (-6, 6) -/
theorem line_segment_endpoint (midpoint endpoint1 endpoint2 : ℝ × ℝ) :
  midpoint = (-3, 4) →
  endpoint1 = (0, 2) →
  (midpoint.1 = (endpoint1.1 + endpoint2.1) / 2 ∧
   midpoint.2 = (endpoint1.2 + endpoint2.2) / 2) →
  endpoint2 = (-6, 6) := by
  sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l2943_294378


namespace NUMINAMATH_CALUDE_siena_bookmarks_l2943_294305

/-- The number of bookmarked pages Siena will have at the end of March -/
def bookmarks_end_of_march (daily_rate : ℕ) (current_bookmarks : ℕ) : ℕ :=
  current_bookmarks + daily_rate * 31

/-- Theorem stating that Siena will have 1330 bookmarked pages at the end of March -/
theorem siena_bookmarks :
  bookmarks_end_of_march 30 400 = 1330 := by
  sorry

end NUMINAMATH_CALUDE_siena_bookmarks_l2943_294305


namespace NUMINAMATH_CALUDE_cell_phone_providers_l2943_294318

theorem cell_phone_providers (n : ℕ) (k : ℕ) : n = 20 ∧ k = 4 →
  (n.factorial / (n - k).factorial) = 116280 := by
  sorry

end NUMINAMATH_CALUDE_cell_phone_providers_l2943_294318


namespace NUMINAMATH_CALUDE_elberta_has_35_5_l2943_294304

/-- The amount of money Granny Smith has -/
def granny_smith_amount : ℚ := 81

/-- The amount of money Anjou has -/
def anjou_amount : ℚ := granny_smith_amount / 4

/-- The amount of money Elberta has -/
def elberta_amount : ℚ := 2 * anjou_amount - 5

/-- Theorem stating that Elberta has $35.5 -/
theorem elberta_has_35_5 : elberta_amount = 35.5 := by
  sorry

end NUMINAMATH_CALUDE_elberta_has_35_5_l2943_294304


namespace NUMINAMATH_CALUDE_welders_left_correct_l2943_294315

/-- The number of welders who started working on another project --/
def welders_left : ℕ := 12

/-- The initial number of welders --/
def initial_welders : ℕ := 36

/-- The number of days to complete the order with all welders --/
def initial_days : ℕ := 5

/-- The number of additional days needed after some welders left --/
def additional_days : ℕ := 6

/-- The rate at which each welder works --/
def welder_rate : ℝ := 1

/-- The total work to be done --/
def total_work : ℝ := initial_welders * initial_days * welder_rate

theorem welders_left_correct :
  (initial_welders - welders_left) * (additional_days * welder_rate) =
  total_work - (initial_welders * welder_rate) := by sorry

end NUMINAMATH_CALUDE_welders_left_correct_l2943_294315


namespace NUMINAMATH_CALUDE_intersection_point_x_coordinate_l2943_294332

theorem intersection_point_x_coordinate (x y : ℝ) : 
  y = 3 * x - 7 ∧ 5 * x + y = 48 → x = 55 / 8 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_x_coordinate_l2943_294332


namespace NUMINAMATH_CALUDE_circle_range_theta_l2943_294334

/-- The range of θ for a circle with center (2cos θ, 2sin θ) and radius 1,
    where all points (x,y) on the circle satisfy x ≤ y -/
theorem circle_range_theta :
  ∀ θ : ℝ,
  (∀ x y : ℝ, (x - 2 * Real.cos θ)^2 + (y - 2 * Real.sin θ)^2 = 1 → x ≤ y) →
  0 ≤ θ →
  θ ≤ 2 * Real.pi →
  5 * Real.pi / 12 ≤ θ ∧ θ ≤ 13 * Real.pi / 12 :=
by sorry

end NUMINAMATH_CALUDE_circle_range_theta_l2943_294334


namespace NUMINAMATH_CALUDE_theater_ticket_sales_l2943_294367

/-- Represents the number of tickets sold for a theater performance --/
structure TheaterTickets where
  orchestra : ℕ
  balcony : ℕ

/-- Calculates the total revenue from ticket sales --/
def totalRevenue (tickets : TheaterTickets) : ℕ :=
  12 * tickets.orchestra + 8 * tickets.balcony

/-- Represents the conditions of the theater ticket sales --/
structure TicketSalesConditions where
  tickets : TheaterTickets
  totalRevenue : ℕ
  balconyExcess : ℕ

/-- The theorem to be proved --/
theorem theater_ticket_sales 
  (conditions : TicketSalesConditions) 
  (h1 : totalRevenue conditions.tickets = conditions.totalRevenue)
  (h2 : conditions.tickets.balcony = conditions.tickets.orchestra + conditions.balconyExcess)
  (h3 : conditions.totalRevenue = 3320)
  (h4 : conditions.balconyExcess = 115) :
  conditions.tickets.orchestra + conditions.tickets.balcony = 355 := by
  sorry

#check theater_ticket_sales

end NUMINAMATH_CALUDE_theater_ticket_sales_l2943_294367


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_equals_one_l2943_294372

theorem sum_of_reciprocals_equals_one (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = x * y) : 
  1/x + 1/y = 1 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_equals_one_l2943_294372


namespace NUMINAMATH_CALUDE_michaels_fruit_cost_l2943_294363

/-- Calculates the total cost of fruit for pies -/
def total_fruit_cost (peach_pies apple_pies blueberry_pies : ℕ) 
                     (fruit_per_pie : ℕ) 
                     (apple_blueberry_price peach_price : ℚ) : ℚ :=
  let peach_pounds := peach_pies * fruit_per_pie
  let apple_pounds := apple_pies * fruit_per_pie
  let blueberry_pounds := blueberry_pies * fruit_per_pie
  let apple_blueberry_cost := (apple_pounds + blueberry_pounds) * apple_blueberry_price
  let peach_cost := peach_pounds * peach_price
  apple_blueberry_cost + peach_cost

/-- Theorem: The total cost of fruit for Michael's pie order is $51.00 -/
theorem michaels_fruit_cost :
  total_fruit_cost 5 4 3 3 1 2 = 51 := by
  sorry

end NUMINAMATH_CALUDE_michaels_fruit_cost_l2943_294363


namespace NUMINAMATH_CALUDE_condition_for_inequality_l2943_294329

theorem condition_for_inequality (a b c : ℝ) :
  (∀ a b c : ℝ, a * c^2 > b * c^2 → a > b) ∧
  (∃ a b c : ℝ, a > b ∧ a * c^2 ≤ b * c^2) :=
sorry

end NUMINAMATH_CALUDE_condition_for_inequality_l2943_294329

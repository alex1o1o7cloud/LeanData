import Mathlib

namespace arithmetic_mean_square_inequality_and_minimum_t_l1344_134456

theorem arithmetic_mean_square_inequality_and_minimum_t :
  (∀ a b c : ℝ, (((a + b + c) / 3) ^ 2 ≤ (a ^ 2 + b ^ 2 + c ^ 2) / 3) ∧
    (((a + b + c) / 3) ^ 2 = (a ^ 2 + b ^ 2 + c ^ 2) / 3 ↔ a = b ∧ b = c)) ∧
  (∀ x y z : ℝ, x > 0 → y > 0 → z > 0 →
    Real.sqrt x + Real.sqrt y + Real.sqrt z ≤ Real.sqrt 3 * Real.sqrt (x + y + z)) ∧
  (∀ t : ℝ, (∀ x y z : ℝ, x > 0 → y > 0 → z > 0 →
    Real.sqrt x + Real.sqrt y + Real.sqrt z ≤ t * Real.sqrt (x + y + z)) →
    t ≥ Real.sqrt 3) := by
  sorry

end arithmetic_mean_square_inequality_and_minimum_t_l1344_134456


namespace vector_calculation_l1344_134445

def v1 : Fin 2 → ℝ := ![3, -6]
def v2 : Fin 2 → ℝ := ![-1, 5]
def v3 : Fin 2 → ℝ := ![5, -20]

theorem vector_calculation :
  (2 • v1 + 4 • v2 - v3) = ![(-3 : ℝ), 28] := by sorry

end vector_calculation_l1344_134445


namespace train_length_l1344_134401

/-- Calculates the length of a train given its speed and time to cross an electric pole. -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 36 → 
  time_s = 9.99920006399488 → 
  (speed_kmh * 1000 / 3600) * time_s = 99.9920006399488 := by
  sorry

end train_length_l1344_134401


namespace earnings_increase_l1344_134435

theorem earnings_increase (last_year_earnings last_year_rent_percentage this_year_rent_percentage rent_increase_percentage : ℝ)
  (h1 : last_year_rent_percentage = 20)
  (h2 : this_year_rent_percentage = 30)
  (h3 : rent_increase_percentage = 187.5)
  (h4 : this_year_rent_percentage / 100 * (last_year_earnings * (1 + x / 100)) = 
        rent_increase_percentage / 100 * (last_year_rent_percentage / 100 * last_year_earnings)) :
  x = 25 := by sorry


end earnings_increase_l1344_134435


namespace product_divisible_by_ten_l1344_134412

theorem product_divisible_by_ten : ∃ k : ℤ, 1265 * 4233 * 254 * 1729 = 10 * k := by sorry

end product_divisible_by_ten_l1344_134412


namespace sum_a1_a5_l1344_134491

/-- Given a sequence {aₙ} where the sum of the first n terms Sₙ = n² + a₁/2,
    prove that a₁ + a₅ = 11 -/
theorem sum_a1_a5 (a : ℕ → ℝ) (S : ℕ → ℝ) 
    (h : ∀ n : ℕ, n > 0 → S n = n^2 + a 1 / 2) : 
    a 1 + a 5 = 11 := by
  sorry

end sum_a1_a5_l1344_134491


namespace father_age_at_second_son_birth_l1344_134433

/-- Represents the ages of a father and his three sons -/
structure FamilyAges where
  father : ℕ
  son1 : ℕ
  son2 : ℕ
  son3 : ℕ

/-- The problem conditions -/
def satisfiesConditions (ages : FamilyAges) : Prop :=
  ages.son1 = ages.son2 + ages.son3 ∧
  ages.father * ages.son1 * ages.son2 * ages.son3 = 27090

/-- The main theorem -/
theorem father_age_at_second_son_birth (ages : FamilyAges) 
  (h : satisfiesConditions ages) : 
  ages.father - ages.son2 = 34 := by
  sorry

end father_age_at_second_son_birth_l1344_134433


namespace lights_on_200_7_11_l1344_134481

/-- The number of lights that are on after the switching operation -/
def lights_on (total_lights : ℕ) (interval1 interval2 : ℕ) : ℕ :=
  (total_lights / interval1 + total_lights / interval2) -
  2 * (total_lights / (interval1 * interval2))

/-- Theorem stating the number of lights on after the switching operation -/
theorem lights_on_200_7_11 :
  lights_on 200 7 11 = 44 := by
sorry

end lights_on_200_7_11_l1344_134481


namespace pitcher_juice_distribution_l1344_134474

theorem pitcher_juice_distribution (C : ℝ) (h : C > 0) : 
  let juice_volume := (2 / 3) * C
  let cups := 6
  let juice_per_cup := juice_volume / cups
  (juice_per_cup / C) * 100 = 11.11 := by
  sorry

end pitcher_juice_distribution_l1344_134474


namespace cubic_root_sum_l1344_134410

theorem cubic_root_sum (a b : ℝ) : 
  (Complex.I : ℂ) ^ 2 = -1 →
  (2 + Complex.I : ℂ) ^ 3 + a * (2 + Complex.I : ℂ) + b = 0 →
  a + b = 9 := by sorry

end cubic_root_sum_l1344_134410


namespace derivative_sin_pi_third_l1344_134415

theorem derivative_sin_pi_third (f : ℝ → ℝ) (h : ∀ x, f x = Real.sin x) :
  deriv f (π / 3) = 1 / 2 := by
  sorry

end derivative_sin_pi_third_l1344_134415


namespace tangent_line_ln_curve_l1344_134418

/-- The equation of the tangent line to y = ln(x+1) at (1, ln 2) -/
theorem tangent_line_ln_curve (x y : ℝ) : 
  let f : ℝ → ℝ := λ t => Real.log (t + 1)
  let tangent_point : ℝ × ℝ := (1, Real.log 2)
  let tangent_line : ℝ → ℝ → Prop := λ a b => x - 2*y - 1 + 2*(Real.log 2) = 0
  (∀ t, (t, f t) ∈ Set.range (λ u => (u, f u))) →  -- curve condition
  tangent_point.1 = 1 ∧ tangent_point.2 = Real.log 2 → -- point condition
  (∃ k : ℝ, ∀ a b, tangent_line a b ↔ b - tangent_point.2 = k * (a - tangent_point.1)) -- tangent line property
  := by sorry

end tangent_line_ln_curve_l1344_134418


namespace project_completion_time_l1344_134465

-- Define the individual work rates
def renu_rate : ℚ := 1 / 5
def suma_rate : ℚ := 1 / 8
def arun_rate : ℚ := 1 / 10

-- Define the combined work rate
def combined_rate : ℚ := renu_rate + suma_rate + arun_rate

-- Theorem statement
theorem project_completion_time :
  (1 : ℚ) / combined_rate = 40 / 17 := by
  sorry

end project_completion_time_l1344_134465


namespace total_limes_is_57_l1344_134406

/-- The number of limes Alyssa picked -/
def alyssa_limes : ℕ := 25

/-- The number of limes Mike picked -/
def mike_limes : ℕ := 32

/-- The total number of limes picked -/
def total_limes : ℕ := alyssa_limes + mike_limes

/-- Theorem stating that the total number of limes picked is 57 -/
theorem total_limes_is_57 : total_limes = 57 := by
  sorry

end total_limes_is_57_l1344_134406


namespace triangle_angle_side_difference_l1344_134454

theorem triangle_angle_side_difference (y : ℝ) : 
  (y + 6 > 0) →  -- AB > 0
  (y + 3 > 0) →  -- AC > 0
  (2*y > 0) →    -- BC > 0
  (y + 6 + y + 3 > 2*y) →  -- AB + AC > BC
  (y + 6 + 2*y > y + 3) →  -- AB + BC > AC
  (y + 3 + 2*y > y + 6) →  -- AC + BC > AB
  (2*y > y + 6) →          -- BC > AB (for ∠A to be largest)
  (2*y > y + 3) →          -- BC > AC (for ∠A to be largest)
  (max (y + 6) (y + 3) - min (y + 6) (y + 3) ≥ 3) ∧ 
  (∃ (y : ℝ), max (y + 6) (y + 3) - min (y + 6) (y + 3) = 3) :=
by sorry

end triangle_angle_side_difference_l1344_134454


namespace imaginary_part_of_complex_fraction_l1344_134429

theorem imaginary_part_of_complex_fraction :
  let z : ℂ := (3 * I - 5) / (2 + I)
  Complex.im z = 11 / 5 := by
  sorry

end imaginary_part_of_complex_fraction_l1344_134429


namespace juice_remaining_l1344_134463

theorem juice_remaining (initial_amount : ℚ) (given_amount : ℚ) (result : ℚ) : 
  initial_amount = 5 → given_amount = 18 / 4 → result = initial_amount - given_amount → result = 1 / 2 := by
  sorry

end juice_remaining_l1344_134463


namespace chinese_remainder_theorem_l1344_134482

theorem chinese_remainder_theorem (x : ℤ) : 
  (x ≡ 2 [ZMOD 6] ∧ x ≡ 3 [ZMOD 5] ∧ x ≡ 4 [ZMOD 7]) ↔ 
  (∃ k : ℤ, x = 210 * k - 52) := by
  sorry

end chinese_remainder_theorem_l1344_134482


namespace aisha_shopping_money_l1344_134483

theorem aisha_shopping_money (initial_money : ℝ) : 
  let after_first := initial_money - (0.4 * initial_money + 4)
  let after_second := after_first - (0.5 * after_first + 5)
  let after_third := after_second - (0.6 * after_second + 6)
  after_third = 2 → initial_money = 90 := by
sorry

end aisha_shopping_money_l1344_134483


namespace min_value_of_sum_of_roots_l1344_134484

theorem min_value_of_sum_of_roots (x : ℝ) :
  ∃ (y : ℝ), ∀ (x : ℝ),
    Real.sqrt (5 * x^2 - 16 * x + 16) + Real.sqrt (5 * x^2 - 18 * x + 29) ≥ y ∧
    ∃ (z : ℝ), Real.sqrt (5 * z^2 - 16 * z + 16) + Real.sqrt (5 * z^2 - 18 * z + 29) = y :=
by
  use Real.sqrt 29
  sorry

end min_value_of_sum_of_roots_l1344_134484


namespace angle_A_measure_l1344_134405

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  sum_of_angles : A + B + C = 180
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C

-- Theorem statement
theorem angle_A_measure (t : Triangle) (h1 : t.C = 3 * t.B) (h2 : t.B = 15) : t.A = 120 := by
  sorry


end angle_A_measure_l1344_134405


namespace circle_equation_implies_sum_l1344_134443

theorem circle_equation_implies_sum (x y : ℝ) :
  x^2 + y^2 - 2*x + 4*y + 5 = 0 → 2*x + y = 0 := by
  sorry

end circle_equation_implies_sum_l1344_134443


namespace horner_v5_equals_761_l1344_134464

def f (x : ℝ) : ℝ := 3 * x^9 + 3 * x^6 + 5 * x^4 + x^3 + 7 * x^2 + 3 * x + 1

def horner_step (v : ℝ) (a : ℝ) (x : ℝ) : ℝ := v * x + a

def horner_method (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc coeff => horner_step acc coeff x) 0

def coefficients : List ℝ := [1, 3, 7, 1, 5, 0, 3, 0, 0, 3]

theorem horner_v5_equals_761 :
  let x : ℝ := 3
  let v₅ := (horner_method (coefficients.take 6) x)
  v₅ = 761 := by sorry

end horner_v5_equals_761_l1344_134464


namespace trapezium_area_l1344_134459

theorem trapezium_area (a b h : ℝ) (ha : a = 24) (hb : b = 14) (hh : h = 18) :
  (a + b) * h / 2 = 342 := by
  sorry

end trapezium_area_l1344_134459


namespace tomatoes_sold_to_wilson_l1344_134413

def total_harvest : Float := 245.5
def sold_to_maxwell : Float := 125.5
def not_sold : Float := 42.0

theorem tomatoes_sold_to_wilson :
  total_harvest - sold_to_maxwell - not_sold = 78 := by
  sorry

end tomatoes_sold_to_wilson_l1344_134413


namespace number_equation_solution_l1344_134402

theorem number_equation_solution : 
  ∃ x : ℝ, 5 * x + 4 = 19 ∧ x = 3 := by sorry

end number_equation_solution_l1344_134402


namespace cube_surface_area_l1344_134436

/-- The surface area of a cube with side length 8 cm is 384 cm². -/
theorem cube_surface_area : 
  let side_length : ℝ := 8
  let surface_area : ℝ := 6 * side_length^2
  surface_area = 384 := by sorry

end cube_surface_area_l1344_134436


namespace multiplier_is_three_l1344_134417

theorem multiplier_is_three (x y a : ℤ) : 
  a * x + y = 40 →
  2 * x - y = 20 →
  3 * y^2 = 48 →
  a = 3 :=
by sorry

end multiplier_is_three_l1344_134417


namespace jennifer_garden_max_area_l1344_134426

/-- Represents a rectangular garden with integer side lengths. -/
structure RectangularGarden where
  length : ℕ
  width : ℕ

/-- Calculates the perimeter of a rectangular garden. -/
def perimeter (g : RectangularGarden) : ℕ := 2 * (g.length + g.width)

/-- Calculates the area of a rectangular garden. -/
def area (g : RectangularGarden) : ℕ := g.length * g.width

/-- Theorem stating the maximum area of Jennifer's garden. -/
theorem jennifer_garden_max_area :
  ∃ (g : RectangularGarden),
    g.length = 30 ∧
    perimeter g = 160 ∧
    (∀ (h : RectangularGarden), h.length = 30 ∧ perimeter h = 160 → area h ≤ area g) ∧
    area g = 1500 := by
  sorry

end jennifer_garden_max_area_l1344_134426


namespace product_three_reciprocal_squares_sum_l1344_134494

theorem product_three_reciprocal_squares_sum :
  ∀ a b : ℕ+, 
  (a * b : ℕ+) = 3 →
  (1 : ℚ) / (a : ℚ)^2 + (1 : ℚ) / (b : ℚ)^2 = 10 / 9 := by
sorry

end product_three_reciprocal_squares_sum_l1344_134494


namespace no_triangle_from_divisibility_conditions_l1344_134473

theorem no_triangle_from_divisibility_conditions (a b c : ℕ+) :
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  Nat.gcd a.val (Nat.gcd b.val c.val) = 1 →
  a.val ∣ (b.val - c.val)^2 →
  b.val ∣ (a.val - c.val)^2 →
  c.val ∣ (a.val - b.val)^2 →
  ¬(a.val + b.val > c.val ∧ b.val + c.val > a.val ∧ c.val + a.val > b.val) := by
sorry

end no_triangle_from_divisibility_conditions_l1344_134473


namespace f_positive_before_zero_point_l1344_134411

noncomputable def f (x : ℝ) : ℝ := (1/3)^x + Real.log x / Real.log (1/3)

theorem f_positive_before_zero_point (a x₀ : ℝ) 
  (h_zero : f a = 0) 
  (h_decreasing : ∀ x y, 0 < x → x < y → f y < f x) 
  (h_range : 0 < x₀ ∧ x₀ < a) : 
  f x₀ > 0 := by
  sorry

end f_positive_before_zero_point_l1344_134411


namespace intersection_of_A_and_B_l1344_134442

-- Define the sets A and B
def A : Set ℝ := {x | -2 < x ∧ x < 2}
def B : Set ℝ := {x | x > Real.sqrt 3}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | Real.sqrt 3 < x ∧ x < 2} := by sorry

end intersection_of_A_and_B_l1344_134442


namespace equal_cost_at_60_messages_l1344_134439

/-- The cost per text message for Plan A -/
def plan_a_cost_per_text : ℚ := 25 / 100

/-- The monthly fee for Plan A -/
def plan_a_monthly_fee : ℚ := 9

/-- The cost per text message for Plan B -/
def plan_b_cost_per_text : ℚ := 40 / 100

/-- The number of text messages at which both plans cost the same -/
def equal_cost_messages : ℕ := 60

theorem equal_cost_at_60_messages :
  plan_a_cost_per_text * equal_cost_messages + plan_a_monthly_fee =
  plan_b_cost_per_text * equal_cost_messages :=
by sorry

end equal_cost_at_60_messages_l1344_134439


namespace ring_arrangement_count_l1344_134471

/-- The number of ways to arrange rings on fingers -/
def ring_arrangements (total_rings : ℕ) (rings_to_use : ℕ) (fingers : ℕ) : ℕ :=
  Nat.choose total_rings rings_to_use * 
  Nat.factorial rings_to_use * 
  Nat.choose (rings_to_use + fingers - 1) (fingers - 1)

/-- Theorem stating the number of ring arrangements for the given problem -/
theorem ring_arrangement_count : ring_arrangements 10 6 5 = 31752000 := by
  sorry

end ring_arrangement_count_l1344_134471


namespace problem_solution_l1344_134468

theorem problem_solution (a b c : ℝ) : 
  (∀ x : ℝ, (x - a) * (x - b) / (x - c) ≥ 0 ↔ x ≤ -2 ∨ |x - 30| < 2) →
  a < b →
  a + 2*b + 3*c = 86 :=
by sorry

end problem_solution_l1344_134468


namespace family_savings_correct_l1344_134489

def income_tax_rate : ℝ := 0.13

def ivan_salary : ℝ := 55000
def vasilisa_salary : ℝ := 45000
def vasilisa_mother_salary : ℝ := 18000
def vasilisa_father_salary : ℝ := 20000
def son_state_stipend : ℝ := 3000
def son_non_state_stipend : ℝ := 15000

def vasilisa_mother_pension : ℝ := 10000

def monthly_expenses : ℝ := 74000

def net_income (gross_income : ℝ) : ℝ :=
  gross_income * (1 - income_tax_rate)

def total_income_before_may2018 : ℝ :=
  net_income ivan_salary + net_income vasilisa_salary + 
  net_income vasilisa_mother_salary + net_income vasilisa_father_salary + 
  son_state_stipend

def total_income_may_to_aug2018 : ℝ :=
  net_income ivan_salary + net_income vasilisa_salary + 
  vasilisa_mother_pension + net_income vasilisa_father_salary + 
  son_state_stipend

def total_income_from_sep2018 : ℝ :=
  net_income ivan_salary + net_income vasilisa_salary + 
  vasilisa_mother_pension + net_income vasilisa_father_salary + 
  son_state_stipend + net_income son_non_state_stipend

theorem family_savings_correct :
  (total_income_before_may2018 - monthly_expenses = 49060) ∧
  (total_income_may_to_aug2018 - monthly_expenses = 43400) ∧
  (total_income_from_sep2018 - monthly_expenses = 56450) := by
  sorry

end family_savings_correct_l1344_134489


namespace golden_ratio_properties_l1344_134478

theorem golden_ratio_properties (x y : ℝ) 
  (hx : x^2 = x + 1) 
  (hy : y^2 = y + 1) 
  (hxy : x ≠ y) : 
  (x + y = 1) ∧ (x^5 + y^5 = 11) := by
  sorry

end golden_ratio_properties_l1344_134478


namespace find_r_l1344_134447

theorem find_r (k r : ℝ) (h1 : 5 = k * 3^r) (h2 : 45 = k * 9^r) : r = 2 := by
  sorry

end find_r_l1344_134447


namespace four_correct_propositions_l1344_134486

theorem four_correct_propositions 
  (a b c : ℝ) : 
  (((a < b → a + c < b + c) ∧                   -- Original proposition
    ((a + c < b + c) → (a < b)) ∧               -- Converse
    ((a ≥ b) → (a + c ≥ b + c)) ∧               -- Inverse
    ((a + c ≥ b + c) → (a ≥ b))) →              -- Contrapositive
   (4 = (Bool.toNat (a < b → a + c < b + c) +
         Bool.toNat ((a + c < b + c) → (a < b)) +
         Bool.toNat ((a ≥ b) → (a + c ≥ b + c)) +
         Bool.toNat ((a + c ≥ b + c) → (a ≥ b))))) :=
by sorry

end four_correct_propositions_l1344_134486


namespace power_of_two_equality_unique_exponent_l1344_134440

theorem power_of_two_equality : 32^3 * 4^3 = 2^21 := by sorry

theorem unique_exponent (h : 32^3 * 4^3 = 2^J) : J = 21 := by sorry

end power_of_two_equality_unique_exponent_l1344_134440


namespace triangle_angle_calculation_l1344_134496

/-- Configuration of triangles ABC and CDE --/
structure TriangleConfig where
  -- Angles in triangle ABC
  angle_A : ℝ
  angle_B : ℝ
  -- Angle y in triangle CDE
  angle_y : ℝ
  -- Assertions about the configuration
  angle_A_eq : angle_A = 50
  angle_B_eq : angle_B = 70
  right_angle_E : True  -- Represents the right angle at E
  angle_C_eq : True  -- Represents that angle at C is same in both triangles

/-- Theorem stating that in the given configuration, y = 30° --/
theorem triangle_angle_calculation (config : TriangleConfig) : config.angle_y = 30 := by
  sorry


end triangle_angle_calculation_l1344_134496


namespace volleyball_advancement_l1344_134488

def can_advance (k : ℕ) (t : ℕ) : Prop :=
  t ≤ k ∧ t * (t - 1) ≤ 2 * t * 1 ∧ (k - t) * (k - t - 1) ≥ 2 * (k - t) * (1 - t + 1)

theorem volleyball_advancement (k : ℕ) (h : k = 5 ∨ k = 6) :
  ∃ t : ℕ, t ≥ 0 ∧ t ≤ 3 ∧ can_advance k t :=
sorry

end volleyball_advancement_l1344_134488


namespace circle_equation_is_correct_l1344_134422

/-- A circle with center on the y-axis, radius 1, passing through (1, 2) -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  passes_through : ℝ × ℝ
  center_on_y_axis : center.1 = 0
  radius_is_one : radius = 1
  point_on_circle : passes_through = (1, 2)

/-- The equation of the circle -/
def circle_equation (c : Circle) (x y : ℝ) : Prop :=
  x^2 + (y - c.center.2)^2 = c.radius^2

theorem circle_equation_is_correct (c : Circle) :
  ∀ x y : ℝ, circle_equation c x y ↔ x^2 + (y - 2)^2 = 1 := by
  sorry

end circle_equation_is_correct_l1344_134422


namespace M_intersect_N_l1344_134499

def M : Set ℝ := {-1, 0, 1}
def N : Set ℝ := {x | x^2 ≤ x}

theorem M_intersect_N : M ∩ N = {0, 1} := by sorry

end M_intersect_N_l1344_134499


namespace unpainted_area_crossed_boards_l1344_134493

/-- The area of the unpainted region when two boards cross -/
theorem unpainted_area_crossed_boards (width1 width2 : ℝ) (angle : ℝ) :
  width1 = 5 →
  width2 = 8 →
  angle = π / 4 →
  (width1 * width2 * Real.sqrt 2) / 2 = 40 * Real.sqrt 2 :=
by sorry

end unpainted_area_crossed_boards_l1344_134493


namespace classroom_average_score_l1344_134400

theorem classroom_average_score (n : ℕ) (h1 : n > 15) :
  let total_average : ℚ := 10
  let subset_average : ℚ := 17
  let subset_size : ℕ := 15
  let remaining_average := (total_average * n - subset_average * subset_size) / (n - subset_size)
  remaining_average = (10 * n - 255 : ℚ) / (n - 15 : ℚ) := by
  sorry

end classroom_average_score_l1344_134400


namespace rectangle_existence_l1344_134476

/-- A point in a 2D plane -/
structure Point := (x : ℝ) (y : ℝ)

/-- A line in a 2D plane -/
structure Line := (a : ℝ) (b : ℝ) (c : ℝ)

/-- A triangle defined by three points -/
structure Triangle := (K : Point) (L : Point) (M : Point)

/-- A rectangle defined by four points -/
structure Rectangle := (A : Point) (B : Point) (C : Point) (D : Point)

/-- Check if a point lies on a line -/
def Point.on_line (P : Point) (l : Line) : Prop :=
  l.a * P.x + l.b * P.y + l.c = 0

/-- Check if a point lies on the extension of a line segment -/
def Point.on_extension (P : Point) (A : Point) (B : Point) : Prop :=
  ∃ (t : ℝ), t > 1 ∧ P.x = A.x + t * (B.x - A.x) ∧ P.y = A.y + t * (B.y - A.y)

/-- Theorem: Given a triangle and a point on the extension of one side, 
    there exists a rectangle with vertices on the triangle's sides -/
theorem rectangle_existence (T : Triangle) (A : Point) :
  A.on_extension T.L T.K →
  ∃ (R : Rectangle),
    R.A = A ∧
    R.B.on_line (Line.mk (T.M.y - T.K.y) (T.K.x - T.M.x) (T.M.x * T.K.y - T.K.x * T.M.y)) ∧
    R.C.on_line (Line.mk (T.L.y - T.K.y) (T.K.x - T.L.x) (T.L.x * T.K.y - T.K.x * T.L.y)) ∧
    R.D.on_line (Line.mk (T.M.y - T.L.y) (T.L.x - T.M.x) (T.M.x * T.L.y - T.L.x * T.M.y)) :=
by
  sorry

end rectangle_existence_l1344_134476


namespace not_in_range_quadratic_l1344_134492

theorem not_in_range_quadratic (b : ℝ) : 
  (∀ x : ℝ, x^2 + b*x + 3 ≠ -3) ↔ -Real.sqrt 24 < b ∧ b < Real.sqrt 24 := by
  sorry

end not_in_range_quadratic_l1344_134492


namespace am_gm_strict_inequality_l1344_134461

theorem am_gm_strict_inequality {a b : ℝ} (ha : a > b) (hb : b > 0) :
  Real.sqrt (a * b) < (a + b) / 2 := by
  sorry

end am_gm_strict_inequality_l1344_134461


namespace speed_reduction_proof_l1344_134425

/-- The speed reduction per passenger in MPH -/
def speed_reduction_per_passenger : ℝ := 2

/-- The speed of an empty plane in MPH -/
def empty_plane_speed : ℝ := 600

/-- The number of passengers on the first plane -/
def passengers_plane1 : ℕ := 50

/-- The number of passengers on the second plane -/
def passengers_plane2 : ℕ := 60

/-- The number of passengers on the third plane -/
def passengers_plane3 : ℕ := 40

/-- The average speed of the three planes in MPH -/
def average_speed : ℝ := 500

theorem speed_reduction_proof :
  (empty_plane_speed - speed_reduction_per_passenger * passengers_plane1 +
   empty_plane_speed - speed_reduction_per_passenger * passengers_plane2 +
   empty_plane_speed - speed_reduction_per_passenger * passengers_plane3) / 3 = average_speed :=
by sorry

end speed_reduction_proof_l1344_134425


namespace orange_weight_equivalence_l1344_134421

-- Define the weight relationship between oranges and apples
def orange_apple_ratio : ℚ := 6 / 9

-- Define the weight relationship between oranges and pears
def orange_pear_ratio : ℚ := 4 / 10

-- Define the number of oranges Jimmy has
def jimmy_oranges : ℕ := 36

-- Theorem statement
theorem orange_weight_equivalence :
  ∃ (apples pears : ℕ),
    (apples : ℚ) = jimmy_oranges * orange_apple_ratio ∧
    (pears : ℚ) = jimmy_oranges * orange_pear_ratio ∧
    apples = 24 ∧
    pears = 14 := by
  sorry

end orange_weight_equivalence_l1344_134421


namespace sum_of_prime_factors_of_nine_to_nine_minus_one_l1344_134449

theorem sum_of_prime_factors_of_nine_to_nine_minus_one : 
  ∃ (p₁ p₂ p₃ p₄ p₅ p₆ : ℕ), 
    Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ ∧ Prime p₅ ∧ Prime p₆ ∧
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₁ ≠ p₅ ∧ p₁ ≠ p₆ ∧
    p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₂ ≠ p₅ ∧ p₂ ≠ p₆ ∧
    p₃ ≠ p₄ ∧ p₃ ≠ p₅ ∧ p₃ ≠ p₆ ∧
    p₄ ≠ p₅ ∧ p₄ ≠ p₆ ∧
    p₅ ≠ p₆ ∧
    (9^9 - 1 : ℕ) = p₁ * p₂ * p₃ * p₄ * p₅ * p₆ ∧
    p₁ + p₂ + p₃ + p₄ + p₅ + p₆ = 835 := by
  sorry

#eval 9^9 - 1

end sum_of_prime_factors_of_nine_to_nine_minus_one_l1344_134449


namespace soccer_team_starters_l1344_134431

theorem soccer_team_starters (total_players : ℕ) (quadruplets : ℕ) (starters : ℕ) 
  (h1 : total_players = 16) 
  (h2 : quadruplets = 4) 
  (h3 : starters = 7) : 
  (Nat.choose quadruplets 3) * (Nat.choose (total_players - quadruplets) (starters - 3)) = 1980 := by
  sorry

end soccer_team_starters_l1344_134431


namespace min_packs_for_135_cans_l1344_134432

/-- Represents the available pack sizes for soda cans -/
inductive PackSize
  | small : PackSize
  | medium : PackSize
  | large : PackSize

/-- Returns the number of cans in a given pack size -/
def cansInPack (s : PackSize) : Nat :=
  match s with
  | .small => 8
  | .medium => 15
  | .large => 30

/-- Represents a combination of packs -/
structure PackCombination where
  small : Nat
  medium : Nat
  large : Nat

/-- Calculates the total number of cans in a pack combination -/
def totalCans (c : PackCombination) : Nat :=
  c.small * cansInPack PackSize.small +
  c.medium * cansInPack PackSize.medium +
  c.large * cansInPack PackSize.large

/-- Checks if a pack combination is valid for the target number of cans -/
def isValidCombination (c : PackCombination) (target : Nat) : Prop :=
  totalCans c = target

/-- Counts the total number of packs in a combination -/
def totalPacks (c : PackCombination) : Nat :=
  c.small + c.medium + c.large

/-- The main theorem: prove that the minimum number of packs to get 135 cans is 5 -/
theorem min_packs_for_135_cans :
  ∃ (c : PackCombination),
    isValidCombination c 135 ∧
    totalPacks c = 5 ∧
    (∀ (c' : PackCombination), isValidCombination c' 135 → totalPacks c ≤ totalPacks c') :=
by sorry

end min_packs_for_135_cans_l1344_134432


namespace polynomial_ratio_equals_infinite_sum_l1344_134475

theorem polynomial_ratio_equals_infinite_sum (x : ℝ) (h : x ∈ Set.Ioo 0 1) :
  x / (1 - x) = ∑' n, x^(2^n) / (1 - x^(2^n + 1)) :=
sorry

end polynomial_ratio_equals_infinite_sum_l1344_134475


namespace geometric_sequence_fourth_term_l1344_134480

/-- Given a sequence a_n where a_1 = 2 and {1 + a_n} forms a geometric sequence
    with common ratio 3, prove that a_4 = 80. -/
theorem geometric_sequence_fourth_term (a : ℕ → ℝ) :
  a 1 = 2 ∧
  (∀ n : ℕ, (1 + a (n + 1)) = 3 * (1 + a n)) →
  a 4 = 80 :=
by sorry

end geometric_sequence_fourth_term_l1344_134480


namespace distance_to_yz_plane_l1344_134407

/-- Given a point P(x, -6, z) where the distance from P to the x-axis is half
    the distance from P to the yz-plane, prove that the distance from P
    to the yz-plane is 12 units. -/
theorem distance_to_yz_plane (x z : ℝ) :
  let P : ℝ × ℝ × ℝ := (x, -6, z)
  abs (-6) = (1/2) * abs x →
  abs x = 12 :=
by sorry

end distance_to_yz_plane_l1344_134407


namespace some_number_value_l1344_134460

theorem some_number_value (some_number : ℝ) :
  (some_number * 14) / 100 = 0.045388 → some_number = 0.3242 := by
  sorry

end some_number_value_l1344_134460


namespace max_value_implies_a_l1344_134458

/-- The function f(x) with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x * (x - 2)^2

/-- Theorem stating that if f(x) has a maximum value of 16/9 and a ≠ 0, then a = 3/2 -/
theorem max_value_implies_a (a : ℝ) (h1 : a ≠ 0) 
  (h2 : ∃ (M : ℝ), M = 16/9 ∧ ∀ (x : ℝ), f a x ≤ M) : 
  a = 3/2 := by
  sorry

end max_value_implies_a_l1344_134458


namespace percentage_red_shirts_l1344_134423

theorem percentage_red_shirts (total_students : ℕ) 
  (blue_percentage green_percentage : ℚ) (other_count : ℕ) :
  total_students = 800 →
  blue_percentage = 45/100 →
  green_percentage = 15/100 →
  other_count = 136 →
  (blue_percentage + green_percentage + (other_count : ℚ)/total_students + 
    (total_students - (blue_percentage * total_students + green_percentage * total_students + other_count))/total_students) = 1 →
  (total_students - (blue_percentage * total_students + green_percentage * total_students + other_count))/total_students = 23/100 := by
  sorry

end percentage_red_shirts_l1344_134423


namespace count_four_digit_numbers_with_two_identical_l1344_134455

/-- The count of four-digit numbers starting with 9 and having exactly two identical digits -/
def four_digit_numbers_with_two_identical : ℕ :=
  let first_case := 9 * 8 * 3  -- when 9 is repeated
  let second_case := 9 * 8 * 3 -- when a digit other than 9 is repeated
  first_case + second_case

/-- Theorem stating that the count of such numbers is 432 -/
theorem count_four_digit_numbers_with_two_identical :
  four_digit_numbers_with_two_identical = 432 := by
  sorry

end count_four_digit_numbers_with_two_identical_l1344_134455


namespace fraction_power_zero_l1344_134409

theorem fraction_power_zero :
  let a : ℤ := 756321948
  let b : ℤ := -3958672103
  (a / b : ℚ) ^ (0 : ℤ) = 1 := by sorry

end fraction_power_zero_l1344_134409


namespace h_monotone_increasing_l1344_134450

/-- Given a real constant a and a function f(x) = x^2 - 2ax + a, 
    we define h(x) = f(x) / x and prove that h(x) is monotonically 
    increasing on [1, +∞) when a < 1. -/
theorem h_monotone_increasing (a : ℝ) (ha : a < 1) :
  ∀ x : ℝ, x ≥ 1 → (
    let f := fun x => x^2 - 2*a*x + a
    let h := fun x => f x / x
    (deriv h) x > 0
  ) := by
  sorry

end h_monotone_increasing_l1344_134450


namespace max_value_of_a_l1344_134498

theorem max_value_of_a (a b c : ℝ) 
  (sum_condition : a + b + c = 6)
  (product_sum_condition : a * b + a * c + b * c = 11) :
  a ≤ 2 + (2 * Real.sqrt 15) / 3 :=
sorry

end max_value_of_a_l1344_134498


namespace quadratic_root_implies_m_value_l1344_134419

theorem quadratic_root_implies_m_value (m : ℝ) : 
  (2 : ℝ)^2 + 5*(2 : ℝ) + m = 0 → m = -14 := by
  sorry

end quadratic_root_implies_m_value_l1344_134419


namespace line_obtuse_angle_range_l1344_134453

/-- Given a line passing through points P(1-a, 1+a) and Q(3, 2a) with an obtuse angle of inclination,
    prove that the range of the real number a is (-2, 1). -/
theorem line_obtuse_angle_range (a : ℝ) : 
  let P : ℝ × ℝ := (1 - a, 1 + a)
  let Q : ℝ × ℝ := (3, 2 * a)
  let slope : ℝ := (Q.2 - P.2) / (Q.1 - P.1)
  (slope < 0) → -2 < a ∧ a < 1 :=
by sorry

end line_obtuse_angle_range_l1344_134453


namespace rational_sqrt2_distance_l1344_134403

theorem rational_sqrt2_distance (a b : ℤ) (h₁ : b ≠ 0) (h₂ : 0 < a/b) (h₃ : a/b < 1) :
  |a/b - 1/Real.sqrt 2| > 1/(4*b^2) := by
  sorry

end rational_sqrt2_distance_l1344_134403


namespace least_seven_digit_binary_l1344_134452

theorem least_seven_digit_binary : ∀ n : ℕ, n > 0 →
  (64 ≤ n ↔ (Nat.log 2 n).succ ≥ 7) :=
by sorry

end least_seven_digit_binary_l1344_134452


namespace model_a_sample_size_l1344_134404

/-- Represents the number of cars to be sampled for a given model -/
def sample_size (total_cars : ℕ) (model_cars : ℕ) (total_sample : ℕ) : ℕ :=
  (model_cars * total_sample) / total_cars

/-- Proves that the sample size for Model A is 6 -/
theorem model_a_sample_size :
  let total_cars := 1200 + 6000 + 2000
  let model_a_cars := 1200
  let total_sample := 46
  sample_size total_cars model_a_cars total_sample = 6 := by
  sorry

#eval sample_size (1200 + 6000 + 2000) 1200 46

end model_a_sample_size_l1344_134404


namespace sum_of_digits_of_difference_gcd_l1344_134437

def difference_gcd (a b c : ℕ) : ℕ :=
  Nat.gcd (Nat.gcd (b - a) (c - b)) (c - a)

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem sum_of_digits_of_difference_gcd :
  sum_of_digits (difference_gcd 1305 4665 6905) = 4 := by
  sorry

end sum_of_digits_of_difference_gcd_l1344_134437


namespace cake_slices_left_l1344_134416

def cake_problem (first_cake_slices second_cake_slices : ℕ)
  (first_cake_friend_fraction second_cake_friend_fraction : ℚ)
  (family_fraction : ℚ)
  (first_cake_alex_eats second_cake_alex_eats : ℕ) : Prop :=
  let first_remaining_after_friends := first_cake_slices - (first_cake_slices * first_cake_friend_fraction).floor
  let second_remaining_after_friends := second_cake_slices - (second_cake_slices * second_cake_friend_fraction).floor
  let first_remaining_after_family := first_remaining_after_friends - (first_remaining_after_friends * family_fraction).floor
  let second_remaining_after_family := second_remaining_after_friends - (second_remaining_after_friends * family_fraction).floor
  let first_final := max 0 (first_remaining_after_family - first_cake_alex_eats)
  let second_final := max 0 (second_remaining_after_family - second_cake_alex_eats)
  first_final + second_final = 2

theorem cake_slices_left :
  cake_problem 8 12 (1/4) (1/3) (1/2) 3 2 :=
by sorry

end cake_slices_left_l1344_134416


namespace midpoint_expression_evaluation_l1344_134487

/-- Given two points P and Q in the plane, prove that the expression 3x - 5y 
    evaluates to -36 at their midpoint R(x, y). -/
theorem midpoint_expression_evaluation (P Q : ℝ × ℝ) (h1 : P = (12, 15)) (h2 : Q = (4, 9)) :
  let R : ℝ × ℝ := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)
  3 * R.1 - 5 * R.2 = -36 := by
sorry

end midpoint_expression_evaluation_l1344_134487


namespace min_value_M_l1344_134457

theorem min_value_M : ∃ (M : ℝ), (∀ (x : ℝ), -x^2 + 2*x ≤ M) ∧ (∀ (N : ℝ), (∀ (x : ℝ), -x^2 + 2*x ≤ N) → M ≤ N) := by
  sorry

end min_value_M_l1344_134457


namespace quadratic_inequality_solution_set_l1344_134467

theorem quadratic_inequality_solution_set :
  {x : ℝ | 2 * x^2 - x - 1 > 0} = {x : ℝ | x < -1/2 ∨ x > 1} := by sorry

end quadratic_inequality_solution_set_l1344_134467


namespace cups_in_smaller_purchase_is_40_l1344_134479

/-- The cost of a single paper plate -/
def plate_cost : ℝ := sorry

/-- The cost of a single paper cup -/
def cup_cost : ℝ := sorry

/-- The number of cups in the smaller purchase -/
def cups_in_smaller_purchase : ℕ := sorry

/-- The total cost of 100 plates and 200 cups is $6.00 -/
axiom total_cost_large : 100 * plate_cost + 200 * cup_cost = 6

/-- The total cost of 20 plates and the unknown number of cups is $1.20 -/
axiom total_cost_small : 20 * plate_cost + cups_in_smaller_purchase * cup_cost = 1.2

theorem cups_in_smaller_purchase_is_40 : cups_in_smaller_purchase = 40 := by sorry

end cups_in_smaller_purchase_is_40_l1344_134479


namespace like_terms_exponent_l1344_134434

theorem like_terms_exponent (x y : ℝ) (m n : ℕ) :
  (∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ a * x * y^(n + 1) = b * x^m * y^4) →
  m^n = 1 := by
  sorry

end like_terms_exponent_l1344_134434


namespace line_properties_l1344_134451

structure Line where
  slope : ℝ
  inclination : ℝ

def parallel (l1 l2 : Line) : Prop := l1.slope = l2.slope

theorem line_properties (l1 l2 : Line) 
  (h_non_overlapping : l1 ≠ l2) : 
  (l1.slope = l2.slope → parallel l1 l2) ∧ 
  (parallel l1 l2 → l1.inclination = l2.inclination) ∧
  (l1.inclination = l2.inclination → parallel l1 l2) := by
  sorry


end line_properties_l1344_134451


namespace percentage_difference_l1344_134448

theorem percentage_difference (x y : ℝ) (h : x = 18 * y) :
  (x - y) / x * 100 = 94.44 := by
  sorry

end percentage_difference_l1344_134448


namespace m_fourth_plus_n_fourth_l1344_134414

theorem m_fourth_plus_n_fourth (m n : ℝ) 
  (h1 : m - n = -5)
  (h2 : m^2 + n^2 = 13) : 
  m^4 + n^4 = 97 := by
  sorry

end m_fourth_plus_n_fourth_l1344_134414


namespace a_is_perfect_square_l1344_134470

def c : ℕ → ℤ
  | 0 => 1
  | 1 => 0
  | 2 => 2005
  | (n + 3) => -3 * c (n + 1) - 4 * c n + 2008

def a (n : ℕ) : ℤ :=
  5 * (c (n + 2) - c n) * (502 - c (n - 1) - c (n - 2)) + 4^n * 2004 * 501

theorem a_is_perfect_square (n : ℕ) (h : n > 2) : ∃ k : ℤ, a n = k^2 := by
  sorry

end a_is_perfect_square_l1344_134470


namespace cone_ratio_after_ten_rotations_l1344_134408

/-- Represents a right circular cone -/
structure RightCircularCone where
  r : ℝ  -- base radius
  h : ℝ  -- height

/-- Predicate for a cone that makes 10 complete rotations when rolling on its side -/
def makesTenRotations (cone : RightCircularCone) : Prop :=
  2 * Real.pi * Real.sqrt (cone.r^2 + cone.h^2) = 20 * Real.pi * cone.r

theorem cone_ratio_after_ten_rotations (cone : RightCircularCone) :
  makesTenRotations cone → cone.h / cone.r = 3 * Real.sqrt 11 := by
  sorry

end cone_ratio_after_ten_rotations_l1344_134408


namespace complex_modulus_problem_l1344_134466

theorem complex_modulus_problem (z : ℂ) : z = (-1 + 2*I) / I → Complex.abs z = Real.sqrt 5 := by
  sorry

end complex_modulus_problem_l1344_134466


namespace angle_sum_in_triangle_rectangle_l1344_134477

/-- A rectangle containing an equilateral triangle -/
structure TriangleInRectangle where
  /-- The measure of one angle between the rectangle and triangle sides -/
  x : ℝ
  /-- The measure of the other angle between the rectangle and triangle sides -/
  y : ℝ
  /-- The rectangle has right angles -/
  rectangle_right_angles : x + y + 60 + 90 + 90 = 540
  /-- The inner triangle is equilateral -/
  equilateral_triangle : True

/-- The sum of angles x and y in a rectangle containing an equilateral triangle is 60° -/
theorem angle_sum_in_triangle_rectangle (t : TriangleInRectangle) : t.x + t.y = 60 := by
  sorry

end angle_sum_in_triangle_rectangle_l1344_134477


namespace complement_of_A_in_U_l1344_134438

def U : Set ℤ := {-2, -1, 0, 1, 2}

def A : Set ℤ := {x : ℤ | 0 < |x| ∧ |x| < 2}

theorem complement_of_A_in_U :
  U \ A = {-2, 0, 2} := by sorry

end complement_of_A_in_U_l1344_134438


namespace alpha_value_l1344_134490

theorem alpha_value (α : Real) (h1 : 0 < α ∧ α < π / 2) 
  (h2 : Real.sin (α - π / 18) = Real.sqrt 3 / 2) : α = π / 180 * 70 := by
  sorry

end alpha_value_l1344_134490


namespace sam_earnings_l1344_134427

/-- The value of a penny in dollars -/
def penny_value : ℚ := 0.01

/-- The value of a nickel in dollars -/
def nickel_value : ℚ := 0.05

/-- The value of a dime in dollars -/
def dime_value : ℚ := 0.10

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 0.25

/-- The number of pennies Sam earned -/
def num_pennies : ℕ := 15

/-- The number of nickels Sam earned -/
def num_nickels : ℕ := 11

/-- The number of dimes Sam earned -/
def num_dimes : ℕ := 21

/-- The number of quarters Sam earned -/
def num_quarters : ℕ := 29

/-- The total value of Sam's earnings in dollars -/
def total_value : ℚ := 
  num_pennies * penny_value + 
  num_nickels * nickel_value + 
  num_dimes * dime_value + 
  num_quarters * quarter_value

theorem sam_earnings : total_value = 10.05 := by
  sorry

end sam_earnings_l1344_134427


namespace total_students_count_l1344_134420

/-- Represents the number of students who scored 60 marks -/
def x : ℕ := sorry

/-- The total number of students in the class -/
def total_students : ℕ := 10 + 15 + x

/-- The average marks for the whole class -/
def class_average : ℕ := 72

/-- The theorem stating the total number of students in the class -/
theorem total_students_count : total_students = 50 := by
  have h1 : (10 * 90 + 15 * 80 + x * 60) / total_students = class_average := by sorry
  sorry

end total_students_count_l1344_134420


namespace solution_set_inequality_l1344_134462

theorem solution_set_inequality (a : ℝ) (h : 0 < a ∧ a < 1) :
  {x : ℝ | (a - x) * (x - 1/a) > 0} = {x : ℝ | a < x ∧ x < 1/a} := by
  sorry

end solution_set_inequality_l1344_134462


namespace vector_parallelism_l1344_134469

theorem vector_parallelism (m : ℝ) : 
  let a : Fin 2 → ℝ := ![2, -1]
  let b : Fin 2 → ℝ := ![-1, m]
  let c : Fin 2 → ℝ := ![-1, 2]
  (∃ (k : ℝ), k ≠ 0 ∧ (a + b) = k • c) → m = -1 := by
sorry

end vector_parallelism_l1344_134469


namespace point_p_coordinates_l1344_134485

/-- Given a linear function and points A and P, proves that P satisfies the conditions of the problem -/
theorem point_p_coordinates (x y : ℝ) : 
  let f : ℝ → ℝ := λ t => -3/2 * t - 3
  let A : ℝ × ℝ := (-5, 0)
  let B : ℝ × ℝ := (f⁻¹ 0, 0)
  let P : ℝ × ℝ := (x, y)
  (P.2 = f P.1) →  -- P lies on the linear function
  (P ≠ B) →        -- P does not coincide with B
  (abs ((A.1 - B.1) * P.2) / 2 = 6) →  -- Area of triangle ABP is 6
  ((x = -14/3 ∧ y = 4) ∨ (x = 2/3 ∧ y = -4)) :=
by sorry

end point_p_coordinates_l1344_134485


namespace gcd_87654321_12345678_l1344_134472

theorem gcd_87654321_12345678 : Nat.gcd 87654321 12345678 = 75 := by
  sorry

end gcd_87654321_12345678_l1344_134472


namespace complex_trig_simplification_l1344_134446

open Complex

theorem complex_trig_simplification (θ : ℝ) :
  let z₁ := (cos θ - I * sin θ) ^ 8
  let z₂ := (1 + I * tan θ) ^ 5
  let z₃ := (cos θ + I * sin θ) ^ 2
  let z₄ := tan θ + I
  (z₁ * z₂) / (z₃ * z₄) = -((sin (4 * θ) + I * cos (4 * θ)) / (cos θ) ^ 4) :=
by sorry

end complex_trig_simplification_l1344_134446


namespace student_weight_replacement_l1344_134441

theorem student_weight_replacement (W : ℝ) :
  (W - 12) / 5 = 12 →
  W = 72 := by
sorry

end student_weight_replacement_l1344_134441


namespace unique_positive_solution_num_positive_solutions_correct_l1344_134495

/-- The polynomial function f(x) = x^11 + 8x^10 + 15x^9 + 1000x^8 - 1200x^7 -/
def f (x : ℝ) : ℝ := x^11 + 8*x^10 + 15*x^9 + 1000*x^8 - 1200*x^7

/-- The number of positive real solutions to the equation f(x) = 0 -/
def num_positive_solutions : ℕ := 1

theorem unique_positive_solution : 
  ∃! (x : ℝ), x > 0 ∧ f x = 0 :=
sorry

theorem num_positive_solutions_correct : 
  (∃! (x : ℝ), x > 0 ∧ f x = 0) ↔ num_positive_solutions = 1 :=
sorry

end unique_positive_solution_num_positive_solutions_correct_l1344_134495


namespace trapezoid_perimeter_l1344_134497

/-- Represents a trapezoid ABCD with given side lengths and a right angle at BCD -/
structure Trapezoid where
  AB : ℝ
  CD : ℝ
  BC : ℝ
  AD : ℝ
  angle_BCD_is_right : Bool

/-- Calculate the perimeter of the trapezoid -/
def perimeter (t : Trapezoid) : ℝ :=
  t.AB + t.CD + t.BC + t.AD

/-- Theorem: The perimeter of the given trapezoid is 118 units -/
theorem trapezoid_perimeter : 
  ∀ (t : Trapezoid), 
  t.AB = 33 ∧ t.CD = 15 ∧ t.BC = 45 ∧ t.AD = 25 ∧ t.angle_BCD_is_right = true → 
  perimeter t = 118 := by
  sorry

#check trapezoid_perimeter

end trapezoid_perimeter_l1344_134497


namespace sum_of_i_powers_l1344_134444

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem sum_of_i_powers : i^12 + i^17 + i^22 + i^27 + i^32 = 1 := by
  sorry

end sum_of_i_powers_l1344_134444


namespace sarah_weeds_proof_l1344_134424

def tuesday_weeds : ℕ := 25

def wednesday_weeds (t : ℕ) : ℕ := 3 * t

def thursday_weeds (w : ℕ) : ℕ := w / 5

def friday_weeds (th : ℕ) : ℕ := th - 10

def total_weeds (t w th f : ℕ) : ℕ := t + w + th + f

theorem sarah_weeds_proof :
  total_weeds tuesday_weeds 
               (wednesday_weeds tuesday_weeds) 
               (thursday_weeds (wednesday_weeds tuesday_weeds)) 
               (friday_weeds (thursday_weeds (wednesday_weeds tuesday_weeds))) = 120 := by
  sorry

end sarah_weeds_proof_l1344_134424


namespace lunules_area_equals_triangle_area_l1344_134428

theorem lunules_area_equals_triangle_area (a b c : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_pythagorean : c^2 = a^2 + b^2) : 
  π * a^2 + π * b^2 - π * c^2 = 2 * a * b := by
  sorry

end lunules_area_equals_triangle_area_l1344_134428


namespace cricket_average_l1344_134430

theorem cricket_average (innings : ℕ) (next_runs : ℕ) (increase : ℕ) (current_average : ℕ) : 
  innings = 20 → 
  next_runs = 120 → 
  increase = 4 → 
  (innings * current_average + next_runs) / (innings + 1) = current_average + increase →
  current_average = 36 := by
sorry

end cricket_average_l1344_134430

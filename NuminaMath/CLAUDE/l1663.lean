import Mathlib

namespace sofa_purchase_sum_l1663_166376

/-- The sum of Joan and Karl's sofa purchases -/
def total_purchase (joan_price karl_price : ℝ) : ℝ := joan_price + karl_price

/-- Theorem: Given the conditions, the sum of Joan and Karl's sofa purchases is $600 -/
theorem sofa_purchase_sum :
  ∀ (joan_price karl_price : ℝ),
  joan_price = 230 →
  2 * joan_price = karl_price + 90 →
  total_purchase joan_price karl_price = 600 := by
sorry

end sofa_purchase_sum_l1663_166376


namespace max_rectangle_area_l1663_166374

/-- The maximum area of a rectangle given constraints --/
theorem max_rectangle_area (perimeter : ℝ) (min_length min_width : ℝ) :
  perimeter = 400 ∧ min_length = 100 ∧ min_width = 50 →
  ∃ (length width : ℝ),
    length ≥ min_length ∧
    width ≥ min_width ∧
    2 * (length + width) = perimeter ∧
    ∀ (l w : ℝ),
      l ≥ min_length →
      w ≥ min_width →
      2 * (l + w) = perimeter →
      l * w ≤ length * width ∧
      length * width = 10000 :=
by sorry

end max_rectangle_area_l1663_166374


namespace greatest_divisor_with_remainders_l1663_166324

theorem greatest_divisor_with_remainders :
  ∃ (d : ℕ), d > 0 ∧
  (∃ (q1 : ℕ), 1428 = d * q1 + 9) ∧
  (∃ (q2 : ℕ), 2206 = d * q2 + 13) ∧
  (∀ (x : ℕ), x > 0 ∧
    (∃ (r1 : ℕ), 1428 = x * r1 + 9) ∧
    (∃ (r2 : ℕ), 2206 = x * r2 + 13) →
    x ≤ d) ∧
  d = 129 :=
by sorry

end greatest_divisor_with_remainders_l1663_166324


namespace correct_amount_returned_l1663_166330

/-- Calculates the amount to be returned in rubles given the initial deposit in USD and the exchange rate. -/
def amount_to_be_returned (initial_deposit : ℝ) (exchange_rate : ℝ) : ℝ :=
  initial_deposit * exchange_rate

/-- Proves that the amount to be returned is 581,500 rubles given the initial deposit and exchange rate. -/
theorem correct_amount_returned (initial_deposit : ℝ) (exchange_rate : ℝ) 
  (h1 : initial_deposit = 10000)
  (h2 : exchange_rate = 58.15) :
  amount_to_be_returned initial_deposit exchange_rate = 581500 := by
  sorry

#eval amount_to_be_returned 10000 58.15

end correct_amount_returned_l1663_166330


namespace triangle_abc_problem_l1663_166381

theorem triangle_abc_problem (A B C : Real) (a b c : Real) 
  (h1 : b * Real.sin A = 3 * c * Real.sin B)
  (h2 : a = 3)
  (h3 : Real.cos B = 2/3) : 
  b = Real.sqrt 6 ∧ Real.sin (2*B - π/3) = (4*Real.sqrt 5 + Real.sqrt 3) / 18 := by
  sorry

end triangle_abc_problem_l1663_166381


namespace interest_rate_calculation_l1663_166390

/-- Calculates the simple interest rate given the principal, time, and interest amount -/
def simple_interest_rate (principal time interest : ℚ) : ℚ :=
  (interest / (principal * time)) * 100

/-- Theorem stating that for the given conditions, the simple interest rate is 2.5% -/
theorem interest_rate_calculation :
  let principal : ℚ := 700
  let time : ℚ := 4
  let interest : ℚ := 70
  simple_interest_rate principal time interest = 2.5 := by
  sorry

end interest_rate_calculation_l1663_166390


namespace min_red_to_blue_l1663_166341

/-- Represents the colors of chameleons -/
inductive Color
  | Red
  | Blue
  | Green
  | Yellow
  | Purple

/-- Represents a chameleon -/
structure Chameleon where
  color : Color

/-- Represents the color change rule -/
def colorChangeRule (biter : Color) (bitten : Color) : Color :=
  sorry -- Specific implementation not provided in the problem

/-- Represents a sequence of bites -/
def BiteSequence := List (Nat × Nat)

/-- Function to apply a bite sequence to a list of chameleons -/
def applyBiteSequence (chameleons : List Chameleon) (sequence : BiteSequence) : List Chameleon :=
  sorry -- Implementation would depend on colorChangeRule

/-- Predicate to check if all chameleons in a list are blue -/
def allBlue (chameleons : List Chameleon) : Prop :=
  ∀ c ∈ chameleons, c.color = Color.Blue

/-- The main theorem to be proved -/
theorem min_red_to_blue :
  ∀ n : Nat,
    (n ≥ 5 →
      ∃ (sequence : BiteSequence),
        allBlue (applyBiteSequence (List.replicate n (Chameleon.mk Color.Red)) sequence)) ∧
    (n < 5 →
      ¬∃ (sequence : BiteSequence),
        allBlue (applyBiteSequence (List.replicate n (Chameleon.mk Color.Red)) sequence)) :=
  sorry


end min_red_to_blue_l1663_166341


namespace percentage_of_330_l1663_166350

theorem percentage_of_330 : (33 + 1/3 : ℚ) / 100 * 330 = 110 := by sorry

end percentage_of_330_l1663_166350


namespace basketball_tournament_games_l1663_166359

theorem basketball_tournament_games (x : ℕ) 
  (h1 : x > 0)
  (h2 : (3 * x) / 4 = (2 * (x + 4)) / 3 - 8) :
  x = 48 := by
sorry

end basketball_tournament_games_l1663_166359


namespace sector_radius_l1663_166344

theorem sector_radius (area : Real) (angle : Real) (π : Real) (h1 : area = 36.67) (h2 : angle = 42) (h3 : π = 3.14159) :
  ∃ r : Real, r = 10 ∧ area = (angle / 360) * π * r^2 := by
  sorry

end sector_radius_l1663_166344


namespace assistant_prof_charts_l1663_166315

theorem assistant_prof_charts (associate_profs assistant_profs : ℕ) 
  (charts_per_assistant : ℕ) :
  associate_profs + assistant_profs = 7 →
  2 * associate_profs + assistant_profs = 10 →
  associate_profs + assistant_profs * charts_per_assistant = 11 →
  charts_per_assistant = 2 :=
by sorry

end assistant_prof_charts_l1663_166315


namespace floor_product_eq_twenty_l1663_166361

theorem floor_product_eq_twenty (x : ℝ) : 
  ⌊x * ⌊x⌋⌋ = 20 ↔ 5 ≤ x ∧ x < (21 / 4) :=
sorry

end floor_product_eq_twenty_l1663_166361


namespace det_A_eq_90_l1663_166386

def A : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![3, 0, 2],
    ![8, 5, -2],
    ![3, 3, 6]]

theorem det_A_eq_90 : Matrix.det A = 90 := by
  sorry

end det_A_eq_90_l1663_166386


namespace total_apples_l1663_166316

/-- Proves that the total number of apples given out is 150, given that Harold gave 25 apples to each of 6 people. -/
theorem total_apples (apples_per_person : ℕ) (num_people : ℕ) (h1 : apples_per_person = 25) (h2 : num_people = 6) : apples_per_person * num_people = 150 := by
  sorry

end total_apples_l1663_166316


namespace unique_solution_condition_l1663_166391

theorem unique_solution_condition (a : ℝ) : 
  (∃! x : ℝ, x^4 - a*x^3 - 3*a*x^2 + 2*a^2*x + a^2 - 2 = 0) ↔ 
  a < (3/4)^2 + 3/4 - 2 :=
sorry

end unique_solution_condition_l1663_166391


namespace x_range_for_inequality_l1663_166366

theorem x_range_for_inequality (x : ℝ) : 
  (∀ p : ℝ, 0 ≤ p ∧ p ≤ 4 → x^2 + p*x > 4*x + p - 3) → 
  x > 3 ∨ x < -1 := by
sorry

end x_range_for_inequality_l1663_166366


namespace new_person_weight_l1663_166363

theorem new_person_weight
  (n : ℕ)
  (initial_average : ℝ)
  (weight_increase : ℝ)
  (replaced_weight : ℝ)
  (h1 : n = 8)
  (h2 : weight_increase = 4)
  (h3 : replaced_weight = 55)
  : ∃ (new_weight : ℝ),
    n * (initial_average + weight_increase) = (n - 1) * initial_average + new_weight ∧
    new_weight = 87
  := by sorry

end new_person_weight_l1663_166363


namespace max_inscribed_rectangle_area_l1663_166368

theorem max_inscribed_rectangle_area (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ (rect_area : ℝ),
    (∀ (inscribed_rect_area : ℝ),
      inscribed_rect_area ≤ rect_area) ∧
    rect_area = (a * b) / 4 :=
by sorry

end max_inscribed_rectangle_area_l1663_166368


namespace product_mod_seven_l1663_166332

theorem product_mod_seven : (2009 * 2010 * 2011 * 2012) % 7 = 0 := by
  sorry

end product_mod_seven_l1663_166332


namespace intersection_of_A_and_B_l1663_166356

def A : Set ℤ := {1, 2, 3}
def B : Set ℤ := {x | x^2 < 9}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by
  sorry

end intersection_of_A_and_B_l1663_166356


namespace additional_red_flowers_needed_l1663_166395

def white_flowers : ℕ := 555
def red_flowers : ℕ := 347

theorem additional_red_flowers_needed : white_flowers - red_flowers = 208 := by
  sorry

end additional_red_flowers_needed_l1663_166395


namespace min_weeks_to_sunday_rest_l1663_166385

/-- Represents the work schedule cycle in days -/
def work_cycle : ℕ := 10

/-- Represents the number of days in a week -/
def days_in_week : ℕ := 7

/-- Represents the initial offset in days (starting rest on Saturday) -/
def initial_offset : ℕ := 6

/-- 
Theorem: Given a work schedule of 8 days work followed by 2 days rest,
starting with rest on Saturday and Sunday, the minimum number of weeks
before resting on a Sunday again is 7.
-/
theorem min_weeks_to_sunday_rest : 
  ∃ (n : ℕ), n > 0 ∧ 
  (n * days_in_week + initial_offset) % work_cycle = work_cycle - 1 ∧
  ∀ (m : ℕ), m > 0 → m < n → 
  (m * days_in_week + initial_offset) % work_cycle ≠ work_cycle - 1 ∧
  n = 7 :=
sorry

end min_weeks_to_sunday_rest_l1663_166385


namespace unique_sum_product_solution_l1663_166364

theorem unique_sum_product_solution (S P : ℝ) (h : S^2 ≥ 4*P) :
  let x₁ := (S + Real.sqrt (S^2 - 4*P)) / 2
  let y₁ := S - x₁
  let x₂ := (S - Real.sqrt (S^2 - 4*P)) / 2
  let y₂ := S - x₂
  (∀ x y : ℝ, x + y = S ∧ x * y = P ↔ (x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂)) :=
by sorry

end unique_sum_product_solution_l1663_166364


namespace min_sum_fraction_l1663_166399

theorem min_sum_fraction (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / (3 * b) + b / (5 * c) + c / (7 * a) ≥ 3 / Real.rpow 105 (1/3) :=
sorry

end min_sum_fraction_l1663_166399


namespace equivalent_operations_l1663_166380

theorem equivalent_operations (x : ℝ) : 
  (x * (4/5)) / (2/7) = x * (14/5) :=
by sorry

end equivalent_operations_l1663_166380


namespace apple_picking_theorem_l1663_166343

/-- The number of apples Lexie picked -/
def lexie_apples : ℕ := 12

/-- Tom picked twice as many apples as Lexie -/
def tom_apples : ℕ := 2 * lexie_apples

/-- The total number of apples collected -/
def total_apples : ℕ := lexie_apples + tom_apples

theorem apple_picking_theorem : total_apples = 36 := by
  sorry

end apple_picking_theorem_l1663_166343


namespace special_triangle_properties_l1663_166394

/-- A right-angled triangle with special properties -/
structure SpecialTriangle where
  -- The hypotenuse of the triangle
  hypotenuse : ℝ
  -- The shorter leg of the triangle
  short_leg : ℝ
  -- The longer leg of the triangle
  long_leg : ℝ
  -- The hypotenuse is 1
  hyp_is_one : hypotenuse = 1
  -- The shorter leg is (√5 - 1) / 2
  short_leg_value : short_leg = (Real.sqrt 5 - 1) / 2
  -- The longer leg is the square root of the shorter leg
  long_leg_value : long_leg = Real.sqrt short_leg

/-- Theorem about the special triangle -/
theorem special_triangle_properties (t : SpecialTriangle) :
  -- The longer leg is the geometric mean of the hypotenuse and shorter leg
  t.long_leg ^ 2 = t.hypotenuse * t.short_leg ∧
  -- All segments formed by successive altitudes are powers of the longer leg
  ∀ n : ℕ, ∃ segment : ℝ, segment = t.long_leg ^ n ∧ 0 ≤ n ∧ n ≤ 9 :=
by
  sorry

end special_triangle_properties_l1663_166394


namespace no_universal_triangle_relation_l1663_166311

/-- A triangle with perimeter, circumradius, and inradius -/
structure Triangle where
  perimeter : ℝ
  circumradius : ℝ
  inradius : ℝ

/-- There is no universal relationship among perimeter, circumradius, and inradius for all triangles -/
theorem no_universal_triangle_relation :
  ¬(∀ t : Triangle,
    (t.perimeter > t.circumradius + t.inradius) ∨
    (t.perimeter ≤ t.circumradius + t.inradius) ∨
    (1/6 < t.circumradius + t.inradius ∧ t.circumradius + t.inradius < 6*t.perimeter)) :=
by sorry

end no_universal_triangle_relation_l1663_166311


namespace least_sum_p_q_l1663_166372

theorem least_sum_p_q (p q : ℕ) (hp : p > 1) (hq : q > 1) 
  (h_eq : 17 * (p + 1) = 25 * (q + 1)) : 
  (∀ p' q' : ℕ, p' > 1 → q' > 1 → 17 * (p' + 1) = 25 * (q' + 1) → p' + q' ≥ p + q) → 
  p + q = 168 := by
sorry

end least_sum_p_q_l1663_166372


namespace paige_pencils_l1663_166371

theorem paige_pencils (initial_pencils : ℕ) : 
  (initial_pencils - 3 = 91) → initial_pencils = 94 := by
  sorry

end paige_pencils_l1663_166371


namespace two_digit_number_sum_l1663_166333

theorem two_digit_number_sum (a b : ℕ) : 
  1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 →
  (10 * a + b) - (10 * b + a) = 7 * (a - b) →
  (10 * a + b) + (10 * b + a) = 33 := by
sorry

end two_digit_number_sum_l1663_166333


namespace counterexample_exists_l1663_166389

theorem counterexample_exists : ∃ (a b c : ℤ), a > b ∧ b > c ∧ a * b ≤ c ^ 2 := by
  sorry

end counterexample_exists_l1663_166389


namespace equation_solution_l1663_166377

theorem equation_solution : ∃ x : ℝ, 
  Real.sqrt (9 + Real.sqrt (18 + 6*x)) + Real.sqrt (3 + Real.sqrt (3 + x)) = 3 + 3*Real.sqrt 3 ∧ 
  x = 31 := by
  sorry

end equation_solution_l1663_166377


namespace parallel_to_plane_not_always_parallel_l1663_166336

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between lines and between a line and a plane
variable (parallel_lines : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)

-- State the theorem
theorem parallel_to_plane_not_always_parallel 
  (l m : Line) (α : Plane) : 
  ¬(∀ l m α, parallel_line_plane l α → parallel_line_plane m α → parallel_lines l m) :=
sorry

end parallel_to_plane_not_always_parallel_l1663_166336


namespace isosceles_triangle_area_l1663_166308

/-- An isosceles triangle with altitude 8 and perimeter 32 has area 48 -/
theorem isosceles_triangle_area (b s : ℝ) : 
  b > 0 → s > 0 → -- b and s are positive real numbers
  2 * s + 2 * b = 32 → -- perimeter condition
  b ^ 2 + 8 ^ 2 = s ^ 2 → -- Pythagorean theorem for half the triangle
  (2 * b) * 8 / 2 = 48 := by 
  sorry


end isosceles_triangle_area_l1663_166308


namespace square_root_sum_equality_l1663_166318

theorem square_root_sum_equality (x : ℝ) :
  Real.sqrt (5 + x) + Real.sqrt (20 - x) = 7 →
  (5 + x) * (20 - x) = 144 := by
  sorry

end square_root_sum_equality_l1663_166318


namespace turnover_equation_l1663_166384

/-- Represents the turnover equation for an online store over three months -/
theorem turnover_equation (x : ℝ) : 
  let july_turnover : ℝ := 16
  let august_turnover : ℝ := july_turnover * (1 + x)
  let september_turnover : ℝ := august_turnover * (1 + x)
  let total_turnover : ℝ := 120
  july_turnover + august_turnover + september_turnover = total_turnover :=
by
  sorry

#check turnover_equation

end turnover_equation_l1663_166384


namespace quadratic_roots_farthest_apart_l1663_166306

/-- The quadratic equation x^2 - 4ax + 5a^2 - 6a = 0 has roots that are farthest apart when a = 3 -/
theorem quadratic_roots_farthest_apart (a : ℝ) :
  let f : ℝ → ℝ := λ x => x^2 - 4*a*x + 5*a^2 - 6*a
  let discriminant := 4*a*(6 - a)
  (∀ b : ℝ, discriminant ≥ 4*b*(6 - b)) → a = 3 := by
  sorry

end quadratic_roots_farthest_apart_l1663_166306


namespace f_increasing_iff_a_in_open_interval_l1663_166300

/-- A piecewise function f defined on ℝ -/
noncomputable def f (a : ℝ) : ℝ → ℝ := fun x =>
  if x < 1 then (3 - a) * x - 4 * a else Real.log x / Real.log a

/-- Theorem stating the range of a for which f is increasing on ℝ -/
theorem f_increasing_iff_a_in_open_interval :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) ↔ a ∈ Set.Ioo 1 3 := by sorry

end f_increasing_iff_a_in_open_interval_l1663_166300


namespace isosceles_exterior_120_is_equilateral_equal_angles_is_equilateral_two_angles_70_40_is_isosceles_l1663_166378

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angle_a : ℝ
  angle_b : ℝ
  angle_c : ℝ
  sum_angles : angle_a + angle_b + angle_c = 180

-- Define an isosceles triangle
def IsoscelesTriangle (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.a = t.c

-- Define an equilateral triangle
def EquilateralTriangle (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c

-- Statement 1
theorem isosceles_exterior_120_is_equilateral (t : Triangle) (h : IsoscelesTriangle t) :
  ∃ (ext_angle : ℝ), ext_angle = 120 → EquilateralTriangle t :=
sorry

-- Statement 2
theorem equal_angles_is_equilateral (t : Triangle) :
  t.angle_a = t.angle_b ∧ t.angle_b = t.angle_c → EquilateralTriangle t :=
sorry

-- Statement 3
theorem two_angles_70_40_is_isosceles (t : Triangle) :
  t.angle_a = 70 ∧ t.angle_b = 40 → IsoscelesTriangle t :=
sorry

end isosceles_exterior_120_is_equilateral_equal_angles_is_equilateral_two_angles_70_40_is_isosceles_l1663_166378


namespace log_division_simplification_l1663_166388

theorem log_division_simplification :
  (Real.log 256 / Real.log 16) / (Real.log (1/256) / Real.log 16) = -1 := by
  sorry

end log_division_simplification_l1663_166388


namespace katy_june_books_l1663_166355

/-- The number of books Katy read in June -/
def june_books : ℕ := sorry

/-- The number of books Katy read in July -/
def july_books : ℕ := 2 * june_books

/-- The number of books Katy read in August -/
def august_books : ℕ := july_books - 3

/-- The total number of books Katy read during the summer -/
def total_books : ℕ := 37

theorem katy_june_books :
  june_books + july_books + august_books = total_books ∧ june_books = 8 := by sorry

end katy_june_books_l1663_166355


namespace double_price_profit_l1663_166387

theorem double_price_profit (cost_price : ℝ) (initial_selling_price : ℝ) :
  initial_selling_price = cost_price * 1.5 →
  let double_price := 2 * initial_selling_price
  (double_price - cost_price) / cost_price = 2 := by
  sorry

end double_price_profit_l1663_166387


namespace simplify_expression_l1663_166357

/-- Given a = 1 and b = -4, prove that 4(a²b+ab²)-3(a²b-1)+2ab²-6 = 89 -/
theorem simplify_expression (a b : ℝ) (ha : a = 1) (hb : b = -4) :
  4*(a^2*b + a*b^2) - 3*(a^2*b - 1) + 2*a*b^2 - 6 = 89 := by
  sorry

end simplify_expression_l1663_166357


namespace cube_edge_ratio_l1663_166354

theorem cube_edge_ratio (a b : ℝ) (h : a^3 / b^3 = 27 / 1) : a / b = 3 / 1 := by
  sorry

end cube_edge_ratio_l1663_166354


namespace negation_of_proposition_l1663_166373

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, x ∈ Set.Ici (0 : ℝ) → x^3 + x ≥ 0) ↔ 
  (∃ x : ℝ, x ∈ Set.Ici (0 : ℝ) ∧ x^3 + x < 0) :=
by sorry

end negation_of_proposition_l1663_166373


namespace equality_of_negative_powers_l1663_166309

theorem equality_of_negative_powers : -(-1)^99 = (-1)^100 := by
  sorry

end equality_of_negative_powers_l1663_166309


namespace streaming_service_subscriber_decrease_l1663_166383

/-- Proves the maximum percentage decrease in subscribers for a streaming service --/
theorem streaming_service_subscriber_decrease
  (initial_price : ℝ)
  (price_increase_percentage : ℝ)
  (h_initial_price : initial_price = 15)
  (h_price_increase : price_increase_percentage = 0.20) :
  let new_price := initial_price * (1 + price_increase_percentage)
  let max_decrease_percentage := 1 - (initial_price / new_price)
  ∃ (ε : ℝ), ε > 0 ∧ abs (max_decrease_percentage - (1/6)) < ε :=
by sorry

end streaming_service_subscriber_decrease_l1663_166383


namespace running_increase_per_week_l1663_166351

theorem running_increase_per_week 
  (initial_capacity : ℝ) 
  (increase_percentage : ℝ) 
  (days : ℕ) 
  (h1 : initial_capacity = 100)
  (h2 : increase_percentage = 0.2)
  (h3 : days = 280) :
  let new_capacity := initial_capacity * (1 + increase_percentage)
  let weeks := days / 7
  (new_capacity - initial_capacity) / weeks = 3 := by sorry

end running_increase_per_week_l1663_166351


namespace point_outside_circle_l1663_166334

theorem point_outside_circle (a b : ℝ) :
  (∃ x y : ℝ, a * x + b * y = 1 ∧ x^2 + y^2 = 1) →
  a^2 + b^2 > 1 := by
sorry

end point_outside_circle_l1663_166334


namespace pond_water_theorem_l1663_166320

/-- Calculates the amount of water remaining in a pond after a certain number of days,
    given initial water amount, evaporation rate, and rain addition rate. -/
def water_remaining (initial_water : ℝ) (evaporation_rate : ℝ) (rain_rate : ℝ) (days : ℕ) : ℝ :=
  initial_water - (evaporation_rate - rain_rate) * days

theorem pond_water_theorem (initial_water : ℝ) (evaporation_rate : ℝ) (rain_rate : ℝ) (days : ℕ) :
  initial_water = 500 ∧ evaporation_rate = 4 ∧ rain_rate = 2 ∧ days = 40 →
  water_remaining initial_water evaporation_rate rain_rate days = 420 := by
  sorry

#eval water_remaining 500 4 2 40

end pond_water_theorem_l1663_166320


namespace books_left_to_read_l1663_166337

def total_books : ℕ := 89
def mcgregor_finished : ℕ := 34
def floyd_finished : ℕ := 32

theorem books_left_to_read :
  total_books - (mcgregor_finished + floyd_finished) = 23 := by
sorry

end books_left_to_read_l1663_166337


namespace two_over_x_values_l1663_166375

theorem two_over_x_values (x : ℝ) (hx : 3 - 9/x + 6/x^2 = 0) :
  2/x = 1 ∨ 2/x = 2 :=
by sorry

end two_over_x_values_l1663_166375


namespace maintenance_check_time_l1663_166325

/-- The initial time between maintenance checks before using the additive -/
def initial_time : ℝ := 20

/-- The new time between maintenance checks after using the additive -/
def new_time : ℝ := 25

/-- The percentage increase in time between maintenance checks -/
def percentage_increase : ℝ := 0.25

theorem maintenance_check_time : 
  initial_time * (1 + percentage_increase) = new_time :=
by sorry

end maintenance_check_time_l1663_166325


namespace root_sum_theorem_l1663_166379

theorem root_sum_theorem (m n p : ℝ) : 
  (∀ x, x^2 + 4*x + p = 0 ↔ x = m ∨ x = n) → 
  m * n = 4 → 
  m + n = -4 := by
sorry

end root_sum_theorem_l1663_166379


namespace parabola_directrix_l1663_166345

/-- Given a parabola with equation y² = 2x, its directrix has the equation x = -1/2 -/
theorem parabola_directrix (x y : ℝ) : 
  (y^2 = 2*x) → (∃ (p : ℝ), p = 1/2 ∧ x = -p) :=
by sorry

end parabola_directrix_l1663_166345


namespace min_dimes_needed_l1663_166346

def jacket_cost : ℚ := 45.50
def ten_dollar_bills : ℕ := 4
def quarters : ℕ := 10
def nickels : ℕ := 15

def min_dimes : ℕ := 23

theorem min_dimes_needed (d : ℕ) : 
  (ten_dollar_bills * 10 + quarters * 0.25 + nickels * 0.05 + d * 0.10 : ℚ) ≥ jacket_cost → 
  d ≥ min_dimes := by
sorry

end min_dimes_needed_l1663_166346


namespace rectangular_solid_length_l1663_166398

/-- Represents the dimensions of a rectangular solid -/
structure RectangularSolid where
  length : ℝ
  width : ℝ
  depth : ℝ

/-- Calculates the surface area of a rectangular solid -/
def surfaceArea (rs : RectangularSolid) : ℝ :=
  2 * (rs.length * rs.width + rs.length * rs.depth + rs.width * rs.depth)

/-- Theorem: The length of a rectangular solid with width 4, depth 1, and surface area 58 is 5 -/
theorem rectangular_solid_length :
  ∃ (rs : RectangularSolid),
    rs.width = 4 ∧
    rs.depth = 1 ∧
    surfaceArea rs = 58 ∧
    rs.length = 5 := by
  sorry

end rectangular_solid_length_l1663_166398


namespace sum_natural_numbers_not_end_72_73_74_l1663_166326

theorem sum_natural_numbers_not_end_72_73_74 (N : ℕ) : 
  ¬ (∃ k : ℕ, (N * (N + 1)) / 2 = 100 * k + 72 ∨ 
               (N * (N + 1)) / 2 = 100 * k + 73 ∨ 
               (N * (N + 1)) / 2 = 100 * k + 74) := by
  sorry


end sum_natural_numbers_not_end_72_73_74_l1663_166326


namespace final_mixture_is_all_x_l1663_166349

/-- Represents a seed mixture -/
structure SeedMixture where
  ryegrass : ℝ
  bluegrass : ℝ
  fescue : ℝ

/-- The final mixture of X and Y -/
structure FinalMixture where
  x : ℝ
  y : ℝ

/-- Seed mixture X -/
def X : SeedMixture :=
  { ryegrass := 1 - 0.6
    bluegrass := 0.6
    fescue := 0 }

/-- Seed mixture Y -/
def Y : SeedMixture :=
  { ryegrass := 0.25
    bluegrass := 0
    fescue := 0.75 }

/-- Theorem stating that the percentage of seed mixture X in the final mixture is 100% -/
theorem final_mixture_is_all_x (m : FinalMixture) :
  X.ryegrass * m.x + Y.ryegrass * m.y = 0.4 * (m.x + m.y) →
  m.x + m.y = 1 →
  m.x = 1 := by
  sorry


end final_mixture_is_all_x_l1663_166349


namespace function_value_at_negative_a_l1663_166338

/-- Given a function f(x) = ax³ + bx + 1, prove that if f(a) = 8, then f(-a) = -6 -/
theorem function_value_at_negative_a (a b : ℝ) : 
  let f : ℝ → ℝ := λ x => a * x^3 + b * x + 1
  f a = 8 → f (-a) = -6 := by
  sorry

end function_value_at_negative_a_l1663_166338


namespace hyperbola_intersection_trajectory_l1663_166312

theorem hyperbola_intersection_trajectory
  (x1 y1 : ℝ)
  (h_on_hyperbola : x1^2 / 2 - y1^2 = 1)
  (h_distinct : x1 ≠ -Real.sqrt 2 ∧ x1 ≠ Real.sqrt 2)
  (x y : ℝ)
  (h_intersection : ∃ (t s : ℝ),
    x = -Real.sqrt 2 + t * (x1 + Real.sqrt 2) ∧
    y = t * y1 ∧
    x = Real.sqrt 2 + s * (x1 - Real.sqrt 2) ∧
    y = -s * y1) :
  x^2 / 2 + y^2 = 1 ∧ x ≠ 0 ∧ x ≠ -Real.sqrt 2 ∧ x ≠ Real.sqrt 2 :=
by sorry

end hyperbola_intersection_trajectory_l1663_166312


namespace vector_operation_l1663_166313

theorem vector_operation (a b : ℝ × ℝ) (h1 : a = (3, 2)) (h2 : b = (0, -1)) :
  2 • b - a = (-3, -4) := by sorry

end vector_operation_l1663_166313


namespace triangle_angle_sum_identity_l1663_166321

theorem triangle_angle_sum_identity (A B C : ℝ) (h : A + B + C = Real.pi) :
  Real.sin (3 * A) + Real.sin (3 * B) + Real.sin (3 * C) = 
  -4 * Real.cos (3/2 * A) * Real.cos (3/2 * B) * Real.cos (3/2 * C) := by
  sorry

end triangle_angle_sum_identity_l1663_166321


namespace series_sum_ln2_series_sum_1_minus_ln2_l1663_166328

/-- The sum of the series where the nth term is 1/((2n-1)(2n)) converges to ln 2 -/
theorem series_sum_ln2 : ∑' n, 1 / ((2 * n - 1) * (2 * n)) = Real.log 2 := by sorry

/-- The sum of the series where the nth term is 1/((2n)(2n+1)) converges to 1 - ln 2 -/
theorem series_sum_1_minus_ln2 : ∑' n, 1 / ((2 * n) * (2 * n + 1)) = 1 - Real.log 2 := by sorry

end series_sum_ln2_series_sum_1_minus_ln2_l1663_166328


namespace added_value_expression_max_value_m_gt_1_max_value_m_le_1_l1663_166323

noncomputable section

variables {a m : ℝ} (h_a : a > 0) (h_m : m > 0)

def x_range (a m : ℝ) : Set ℝ := Set.Ioo 0 ((2 * a * m) / (2 * m + 1))

def y (a x : ℝ) : ℝ := 8 * (a - x) * x^2

theorem added_value_expression (x : ℝ) (hx : x ∈ x_range a m) :
  y a x = 8 * (a - x) * x^2 := by sorry

theorem max_value_m_gt_1 (h_m_gt_1 : m > 1) :
  ∃ (x_max : ℝ), x_max ∈ x_range a m ∧
    y a x_max = (32 / 27) * a^3 ∧
    ∀ (x : ℝ), x ∈ x_range a m → y a x ≤ y a x_max := by sorry

theorem max_value_m_le_1 (h_m_le_1 : 0 < m ∧ m ≤ 1) :
  ∃ (x_max : ℝ), x_max ∈ x_range a m ∧
    y a x_max = (32 * m^2) / (2 * m + 1)^3 * a^3 ∧
    ∀ (x : ℝ), x ∈ x_range a m → y a x ≤ y a x_max := by sorry

end

end added_value_expression_max_value_m_gt_1_max_value_m_le_1_l1663_166323


namespace assignment_schemes_l1663_166304

def number_of_roles : ℕ := 5
def number_of_members : ℕ := 5

def roles_for_A : ℕ := number_of_roles - 2
def roles_for_B : ℕ := 1
def remaining_members : ℕ := number_of_members - 2
def remaining_roles : ℕ := number_of_roles - 2

theorem assignment_schemes :
  (roles_for_B) * (roles_for_A) * (remaining_members.factorial) = 18 := by
  sorry

end assignment_schemes_l1663_166304


namespace chord_diagonal_intersections_collinear_l1663_166396

namespace CircleChords

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point on the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a chord as a pair of points
structure Chord where
  p1 : Point
  p2 : Point

-- Define the problem setup
structure ChordConfiguration where
  circle : Circle
  chordAB : Chord
  chordCD : Chord
  chordEF : Chord
  -- Ensure chords are non-intersecting
  non_intersecting : 
    chordAB.p1 ≠ chordCD.p1 ∧ chordAB.p1 ≠ chordCD.p2 ∧
    chordAB.p2 ≠ chordCD.p1 ∧ chordAB.p2 ≠ chordCD.p2 ∧
    chordAB.p1 ≠ chordEF.p1 ∧ chordAB.p1 ≠ chordEF.p2 ∧
    chordAB.p2 ≠ chordEF.p1 ∧ chordAB.p2 ≠ chordEF.p2 ∧
    chordCD.p1 ≠ chordEF.p1 ∧ chordCD.p1 ≠ chordEF.p2 ∧
    chordCD.p2 ≠ chordEF.p1 ∧ chordCD.p2 ≠ chordEF.p2

-- Define the intersection of diagonals
def diagonalIntersection (q1 q2 q3 q4 : Point) : Point :=
  sorry -- Actual implementation would calculate the intersection

-- Define collinearity
def collinear (p1 p2 p3 : Point) : Prop :=
  sorry -- Actual implementation would define collinearity

-- Theorem statement
theorem chord_diagonal_intersections_collinear (config : ChordConfiguration) :
  let M := diagonalIntersection config.chordAB.p1 config.chordAB.p2 config.chordEF.p1 config.chordEF.p2
  let N := diagonalIntersection config.chordCD.p1 config.chordCD.p2 config.chordEF.p1 config.chordEF.p2
  let P := diagonalIntersection config.chordAB.p1 config.chordAB.p2 config.chordCD.p1 config.chordCD.p2
  collinear M N P :=
by
  sorry

end CircleChords

end chord_diagonal_intersections_collinear_l1663_166396


namespace ada_paul_scores_l1663_166370

/-- Ada and Paul's test scores problem -/
theorem ada_paul_scores (A1 A2 A3 P1 P2 P3 : ℤ) 
  (h1 : A1 > P1)
  (h2 : A2 = P2 + 4)
  (h3 : (P1 + P2 + P3) / 3 = (A1 + A2 + A3) / 3 + 4)
  (h4 : P3 = A3 + 26) :
  A1 - P1 = 10 := by
  sorry

end ada_paul_scores_l1663_166370


namespace extreme_values_and_bounds_l1663_166369

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x

-- State the theorem
theorem extreme_values_and_bounds (a b : ℝ) :
  (∃ (x : ℝ), x = 1 ∧ (∀ (h : ℝ), f a b x ≥ f a b h ∨ f a b x ≤ f a b h)) ∧
  (∃ (y : ℝ), y = -2/3 ∧ (∀ (h : ℝ), f a b y ≥ f a b h ∨ f a b y ≤ f a b h)) →
  (a = -1/2 ∧ b = -2) ∧
  (∀ x ∈ Set.Icc (-1) 2, f (-1/2) (-2) x ≤ 2) ∧
  (∀ x ∈ Set.Icc (-1) 2, f (-1/2) (-2) x ≥ -5/2) ∧
  (∃ x ∈ Set.Icc (-1) 2, f (-1/2) (-2) x = 2) ∧
  (∃ x ∈ Set.Icc (-1) 2, f (-1/2) (-2) x = -5/2) :=
by sorry

end extreme_values_and_bounds_l1663_166369


namespace election_votes_l1663_166360

theorem election_votes (total_votes : ℕ) (invalid_percent : ℚ) (winner_percent : ℚ) :
  total_votes = 9000 →
  invalid_percent = 30 / 100 →
  winner_percent = 60 / 100 →
  ∃ (other_votes : ℕ), other_votes = 2520 :=
by
  sorry

end election_votes_l1663_166360


namespace miranda_savings_duration_l1663_166347

def total_cost : ℕ := 260
def sister_contribution : ℕ := 50
def monthly_saving : ℕ := 70

theorem miranda_savings_duration :
  (total_cost - sister_contribution) / monthly_saving = 3 := by
  sorry

end miranda_savings_duration_l1663_166347


namespace diagonal_sum_equals_fibonacci_l1663_166303

/-- The sum of binomial coefficients in a diagonal of Pascal's Triangle -/
def diagonalSum (n : ℕ) : ℕ :=
  Finset.sum (Finset.range (n + 1)) (fun k => Nat.choose (n - k) k)

/-- The nth Fibonacci number -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- The main theorem: The diagonal sum equals the (n+1)th Fibonacci number -/
theorem diagonal_sum_equals_fibonacci (n : ℕ) : diagonalSum n = fib (n + 1) := by
  sorry

end diagonal_sum_equals_fibonacci_l1663_166303


namespace inscribed_sphere_volume_l1663_166365

/-- The volume of a sphere inscribed in a right circular cone -/
theorem inscribed_sphere_volume (d : ℝ) (h : d = 18) :
  let r := 9 - 9 * Real.sqrt 2 / 2
  (4 / 3 : ℝ) * Real.pi * r^3 = (4 / 3 : ℝ) * Real.pi * (9 - 9 * Real.sqrt 2 / 2)^3 := by
  sorry

end inscribed_sphere_volume_l1663_166365


namespace rectangle_area_increase_l1663_166317

theorem rectangle_area_increase (L W : ℝ) (hL : L > 0) (hW : W > 0) : 
  let original_area := L * W
  let new_length := L * 1.2
  let new_width := W * 1.2
  let new_area := new_length * new_width
  (new_area - original_area) / original_area * 100 = 44 := by
  sorry

end rectangle_area_increase_l1663_166317


namespace functional_equation_solution_l1663_166331

theorem functional_equation_solution 
  (f g h : ℝ → ℝ) 
  (hf : Continuous f) 
  (hg : Continuous g) 
  (hh : Continuous h) 
  (h_eq : ∀ x y, f (x + y) = g x + h y) :
  ∃ a b c : ℝ, 
    (∀ x, f x = c * x + a + b) ∧
    (∀ x, g x = c * x + a) ∧
    (∀ x, h x = c * x + b) :=
by sorry

end functional_equation_solution_l1663_166331


namespace house_rent_expenditure_l1663_166301

theorem house_rent_expenditure (total_income : ℝ) (petrol_spending : ℝ) 
  (h1 : petrol_spending = 0.3 * total_income)
  (h2 : petrol_spending = 300) : ℝ :=
by
  let remaining_income := total_income - petrol_spending
  let house_rent := 0.14 * remaining_income
  have : house_rent = 98 := by sorry
  exact house_rent

#check house_rent_expenditure

end house_rent_expenditure_l1663_166301


namespace max_square_sum_l1663_166397

def triangle_numbers : Finset ℕ := {5, 6, 7, 8, 9}

def circle_product (a b c : ℕ) : ℕ := a * b * c

def square_sum (f g h : ℕ) : ℕ := f + g + h

theorem max_square_sum :
  ∃ (a b c d e : ℕ),
    a ∈ triangle_numbers ∧
    b ∈ triangle_numbers ∧
    c ∈ triangle_numbers ∧
    d ∈ triangle_numbers ∧
    e ∈ triangle_numbers ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
    c ≠ d ∧ c ≠ e ∧
    d ≠ e ∧
    square_sum (circle_product a b c) (circle_product b c d) (circle_product c d e) = 1251 ∧
    ∀ (x y z w v : ℕ),
      x ∈ triangle_numbers →
      y ∈ triangle_numbers →
      z ∈ triangle_numbers →
      w ∈ triangle_numbers →
      v ∈ triangle_numbers →
      x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ x ≠ v ∧
      y ≠ z ∧ y ≠ w ∧ y ≠ v ∧
      z ≠ w ∧ z ≠ v ∧
      w ≠ v →
      square_sum (circle_product x y z) (circle_product y z w) (circle_product z w v) ≤ 1251 :=
sorry

end max_square_sum_l1663_166397


namespace part_one_part_two_l1663_166319

-- Define the function f(x) = |x-a| + 3x
def f (a : ℝ) (x : ℝ) : ℝ := abs (x - a) + 3 * x

theorem part_one :
  let f₁ := f 1
  (∀ x, f₁ x ≥ 3 * x + 2) ↔ (x ≥ 3 ∨ x ≤ -1) :=
sorry

theorem part_two (a : ℝ) (h : a > 0) :
  (∀ x, f a x ≤ 0 ↔ x ≤ -1) → a = 2 :=
sorry

end part_one_part_two_l1663_166319


namespace simplify_expression_l1663_166340

theorem simplify_expression (x : ℝ) : (x + 1)^2 + x*(x - 2) = 2*x^2 + 1 := by
  sorry

end simplify_expression_l1663_166340


namespace common_ratio_of_geometric_sequence_l1663_166305

/-- A geometric sequence with given third and sixth terms -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (q : ℝ), ∀ (n : ℕ), a (n + 1) = a n * q

theorem common_ratio_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_3 : a 3 = 8)
  (h_6 : a 6 = 64) :
  ∃ (q : ℝ), (∀ (n : ℕ), a (n + 1) = a n * q) ∧ q = 2 :=
sorry

end common_ratio_of_geometric_sequence_l1663_166305


namespace loss_fraction_l1663_166329

theorem loss_fraction (cost_price selling_price : ℚ) 
  (h1 : cost_price = 21)
  (h2 : selling_price = 20) :
  (cost_price - selling_price) / cost_price = 1 / 21 := by
  sorry

end loss_fraction_l1663_166329


namespace shooting_test_probability_l1663_166348

/-- Represents the probability of hitting a single shot -/
def hit_prob : ℝ := 0.6

/-- Represents the probability of missing a single shot -/
def miss_prob : ℝ := 1 - hit_prob

/-- Calculates the probability of passing the shooting test -/
def pass_prob : ℝ := 
  hit_prob^3 + hit_prob^2 * miss_prob + miss_prob * hit_prob^2

theorem shooting_test_probability : pass_prob = 0.504 := by
  sorry

end shooting_test_probability_l1663_166348


namespace birds_in_tree_l1663_166358

/-- The number of birds left in a tree after some fly away -/
def birds_left (initial : ℝ) (flew_away : ℝ) : ℝ :=
  initial - flew_away

/-- Theorem: Given 21.0 initial birds and 14.0 birds that flew away, 7.0 birds are left -/
theorem birds_in_tree : birds_left 21.0 14.0 = 7.0 := by
  sorry

end birds_in_tree_l1663_166358


namespace expression_undefined_at_twelve_l1663_166392

theorem expression_undefined_at_twelve :
  ∀ x : ℝ, x = 12 → (x^2 - 24*x + 144 = 0) := by sorry

end expression_undefined_at_twelve_l1663_166392


namespace dana_jayden_pencil_difference_l1663_166382

theorem dana_jayden_pencil_difference :
  ∀ (dana_pencils jayden_pencils marcus_pencils : ℕ),
    jayden_pencils = 20 →
    jayden_pencils = 2 * marcus_pencils →
    dana_pencils = marcus_pencils + 25 →
    dana_pencils - jayden_pencils = 15 :=
by
  sorry

end dana_jayden_pencil_difference_l1663_166382


namespace intersection_x_coordinate_l1663_166367

-- Define the two lines
def line1 (x y : ℝ) : Prop := y = 5 * x - 20
def line2 (x y : ℝ) : Prop := 3 * x + y = 110

-- Define the intersection point
def intersection (x y : ℝ) : Prop := line1 x y ∧ line2 x y

-- Theorem statement
theorem intersection_x_coordinate :
  ∃ x y : ℝ, intersection x y ∧ x = 16.25 := by
sorry

end intersection_x_coordinate_l1663_166367


namespace three_person_subcommittees_from_eight_l1663_166362

theorem three_person_subcommittees_from_eight (n : ℕ) (k : ℕ) : n = 8 → k = 3 → Nat.choose n k = 56 := by
  sorry

end three_person_subcommittees_from_eight_l1663_166362


namespace circle_symmetry_implies_a_value_l1663_166353

/-- A circle C with equation x^2 + y^2 + 2x + ay - 10 = 0, where a is a real number -/
def Circle (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 + 2*p.1 + a*p.2 - 10 = 0}

/-- The line l with equation x - y + 2 = 0 -/
def Line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 - p.2 + 2 = 0}

/-- A point is symmetric about a line if the line is the perpendicular bisector of the line segment
    joining the point and its reflection -/
def IsSymmetricAbout (p q : ℝ × ℝ) (l : Set (ℝ × ℝ)) : Prop :=
  q ∈ l ∧ (p.1 + q.1) / 2 = q.1 ∧ (p.2 + q.2) / 2 = q.2

theorem circle_symmetry_implies_a_value (a : ℝ) :
  (∀ p ∈ Circle a, ∃ q, q ∈ Circle a ∧ IsSymmetricAbout p q Line) →
  a = -2 := by
  sorry

end circle_symmetry_implies_a_value_l1663_166353


namespace average_equals_x_l1663_166335

theorem average_equals_x (x : ℝ) : 
  (2 + 5 + x + 14 + 15) / 5 = x → x = 9 := by
  sorry

end average_equals_x_l1663_166335


namespace sum_of_coefficients_l1663_166307

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (1 + x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + a₂ + a₃ + a₄ + a₅ = 31 := by
sorry

end sum_of_coefficients_l1663_166307


namespace root_implies_m_value_l1663_166352

theorem root_implies_m_value (m : ℝ) : 
  (1 : ℝ)^2 + m * (1 : ℝ) - 3 = 0 → m = 2 := by
  sorry

end root_implies_m_value_l1663_166352


namespace amount_to_find_l1663_166339

def water_bottles : ℕ := 5 * 12
def energy_bars : ℕ := 4 * 12
def original_water_price : ℚ := 2
def original_energy_price : ℚ := 3
def market_water_price : ℚ := 185/100
def market_energy_price : ℚ := 275/100
def discount_rate : ℚ := 1/10

def original_total : ℚ := water_bottles * original_water_price + energy_bars * original_energy_price

def discounted_water_price : ℚ := market_water_price * (1 - discount_rate)
def discounted_energy_price : ℚ := market_energy_price * (1 - discount_rate)

def discounted_total : ℚ := water_bottles * discounted_water_price + energy_bars * discounted_energy_price

theorem amount_to_find : original_total - discounted_total = 453/10 := by sorry

end amount_to_find_l1663_166339


namespace sqrt_49_times_sqrt_25_l1663_166322

theorem sqrt_49_times_sqrt_25 : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 := by
  sorry

end sqrt_49_times_sqrt_25_l1663_166322


namespace license_plate_theorem_l1663_166342

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The number of letter positions in the license plate -/
def letter_positions : ℕ := 4

/-- The number of digit positions in the license plate -/
def digit_positions : ℕ := 3

/-- The number of possible digits (0-9) -/
def digit_options : ℕ := 10

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

/-- Calculates the number of license plate combinations -/
def license_plate_combinations : ℕ :=
  alphabet_size *
  (choose (alphabet_size - 1) 2) *
  (choose letter_positions 2) *
  (digit_options * (digit_options - 1) * (digit_options - 2))

theorem license_plate_theorem :
  license_plate_combinations = 33696000 := by
  sorry

end license_plate_theorem_l1663_166342


namespace correct_average_after_error_correction_l1663_166327

/-- Calculates the correct average marks after correcting an error in one student's mark -/
theorem correct_average_after_error_correction 
  (num_students : ℕ) 
  (initial_average : ℚ) 
  (wrong_mark : ℚ) 
  (correct_mark : ℚ) : 
  num_students = 10 → 
  initial_average = 100 → 
  wrong_mark = 60 → 
  correct_mark = 10 → 
  (initial_average * num_students - wrong_mark + correct_mark) / num_students = 95 := by
sorry

end correct_average_after_error_correction_l1663_166327


namespace equation_solutions_l1663_166393

/-- The set of solutions to the equation (a³ + b³)ⁿ = 4(ab)¹⁹⁹⁵ where a, b, n are integers greater than 1 -/
def Solutions : Set (ℕ × ℕ × ℕ) :=
  {(1, 1, 2), (2, 2, 998), (32, 32, 1247), (2^55, 2^55, 1322), (2^221, 2^221, 1328)}

/-- The predicate that checks if a triple (a, b, n) satisfies the equation (a³ + b³)ⁿ = 4(ab)¹⁹⁹⁵ -/
def SatisfiesEquation (a b n : ℕ) : Prop :=
  a > 1 ∧ b > 1 ∧ n > 1 ∧ (a^3 + b^3)^n = 4 * (a * b)^1995

theorem equation_solutions :
  ∀ a b n : ℕ, SatisfiesEquation a b n ↔ (a, b, n) ∈ Solutions := by
  sorry

end equation_solutions_l1663_166393


namespace inequality_proof_l1663_166302

theorem inequality_proof (a b x : ℝ) (h1 : 0 < a) (h2 : a < b) :
  (b - a) / (b + a) ≤ (b + a * Real.sin x) / (b - a * Real.sin x) ∧
  (b + a * Real.sin x) / (b - a * Real.sin x) ≤ (b + a) / (b - a) :=
by sorry

end inequality_proof_l1663_166302


namespace square_difference_l1663_166314

theorem square_difference (x y : ℝ) (h1 : (x + y)^2 = 64) (h2 : x * y = 15) : (x - y)^2 = 4 := by
  sorry

end square_difference_l1663_166314


namespace function_passes_through_point_l1663_166310

theorem function_passes_through_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^(x - 1) + 1
  f 1 = 2 := by sorry

end function_passes_through_point_l1663_166310

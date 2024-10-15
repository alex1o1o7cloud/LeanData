import Mathlib

namespace NUMINAMATH_CALUDE_min_max_of_expression_l1195_119563

open Real

theorem min_max_of_expression (x : ℝ) (h : x > 2) :
  let f := fun x => (x + 9) / sqrt (x - 2)
  (∃ (m : ℝ), m = 2 * sqrt 11 ∧ 
    (∀ y, y > 2 → f y ≥ m) ∧ 
    f 13 = m) ∧
  (∀ M : ℝ, ∃ y, y > 2 ∧ f y > M) :=
by sorry

end NUMINAMATH_CALUDE_min_max_of_expression_l1195_119563


namespace NUMINAMATH_CALUDE_tv_show_payment_l1195_119502

theorem tv_show_payment (main_characters minor_characters : ℕ)
  (major_pay_ratio : ℕ) (total_payment : ℕ) :
  main_characters = 5 →
  minor_characters = 4 →
  major_pay_ratio = 3 →
  total_payment = 285000 →
  ∃ (minor_pay : ℕ),
    minor_pay = 15000 ∧
    minor_pay * (minor_characters + main_characters * major_pay_ratio) = total_payment :=
by sorry

end NUMINAMATH_CALUDE_tv_show_payment_l1195_119502


namespace NUMINAMATH_CALUDE_range_of_a_l1195_119554

/-- The range of real number a when "x=1" is a sufficient but not necessary condition for "(x-a)[x-(a+2)]≤0" -/
theorem range_of_a : ∃ (a_min a_max : ℝ), a_min = -1 ∧ a_max = 1 ∧
  ∀ (a : ℝ), (∀ (x : ℝ), x = 1 → (x - a) * (x - (a + 2)) ≤ 0) ∧
             (∃ (x : ℝ), x ≠ 1 ∧ (x - a) * (x - (a + 2)) ≤ 0) ↔
             a_min ≤ a ∧ a ≤ a_max := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1195_119554


namespace NUMINAMATH_CALUDE_luna_pink_crayons_percentage_l1195_119555

/-- Given information about Mara's and Luna's crayons, prove that 20% of Luna's crayons are pink -/
theorem luna_pink_crayons_percentage
  (mara_total : ℕ)
  (mara_pink_percent : ℚ)
  (luna_total : ℕ)
  (total_pink : ℕ)
  (h1 : mara_total = 40)
  (h2 : mara_pink_percent = 1/10)
  (h3 : luna_total = 50)
  (h4 : total_pink = 14)
  : (luna_total - (mara_pink_percent * mara_total).floor) / luna_total = 1/5 := by
  sorry

#eval (50 : ℚ) / 5  -- Expected output: 10

end NUMINAMATH_CALUDE_luna_pink_crayons_percentage_l1195_119555


namespace NUMINAMATH_CALUDE_contrapositive_prop2_true_l1195_119517

theorem contrapositive_prop2_true : 
  (∀ x : ℝ, (x + 2) * (x - 3) > 0 → (x < -2 ∨ x > 0)) := by sorry

end NUMINAMATH_CALUDE_contrapositive_prop2_true_l1195_119517


namespace NUMINAMATH_CALUDE_product_of_square_roots_l1195_119539

theorem product_of_square_roots (q : ℝ) (h : q > 0) :
  Real.sqrt (15 * q) * Real.sqrt (8 * q^3) * Real.sqrt (12 * q^5) = 12 * q^4 * Real.sqrt (10 * q) :=
by sorry

end NUMINAMATH_CALUDE_product_of_square_roots_l1195_119539


namespace NUMINAMATH_CALUDE_sufficient_condition_for_inequality_l1195_119509

theorem sufficient_condition_for_inequality (a : ℝ) :
  (a < 1) → (∀ x : ℝ, a ≤ |x| + |x - 1|) ∧
  ¬(∀ a : ℝ, (∀ x : ℝ, a ≤ |x| + |x - 1|) → (a < 1)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_inequality_l1195_119509


namespace NUMINAMATH_CALUDE_dance_school_relation_l1195_119587

/-- Represents the dance school scenario -/
structure DanceSchool where
  b : ℕ  -- number of boys
  g : ℕ  -- number of girls

/-- The number of girls the nth boy dances with -/
def girls_danced (n : ℕ) : ℕ := 2 * n + 4

/-- The dance school satisfies the given conditions -/
def valid_dance_school (ds : DanceSchool) : Prop :=
  ∀ n, n ≥ 1 → n ≤ ds.b → girls_danced n ≤ ds.g ∧
  girls_danced ds.b = ds.g

theorem dance_school_relation (ds : DanceSchool) 
  (h : valid_dance_school ds) : 
  ds.b = (ds.g - 4) / 2 :=
sorry

end NUMINAMATH_CALUDE_dance_school_relation_l1195_119587


namespace NUMINAMATH_CALUDE_company_employees_l1195_119562

theorem company_employees (wednesday_birthdays : ℕ) 
  (other_day_birthdays : ℕ) : 
  wednesday_birthdays = 13 →
  wednesday_birthdays > other_day_birthdays →
  (7 * other_day_birthdays + wednesday_birthdays - other_day_birthdays : ℕ) = 85 :=
by sorry

end NUMINAMATH_CALUDE_company_employees_l1195_119562


namespace NUMINAMATH_CALUDE_number_divided_by_three_l1195_119557

theorem number_divided_by_three : ∃ x : ℝ, (x / 3 = x - 48) ∧ (x = 72) := by
  sorry

end NUMINAMATH_CALUDE_number_divided_by_three_l1195_119557


namespace NUMINAMATH_CALUDE_unique_award_implies_all_defeated_l1195_119522

def Tournament (α : Type) := α → α → Prop

structure Award (α : Type) (t : Tournament α) (winner : α) : Prop :=
  (defeated_or_indirect : ∀ b : α, b ≠ winner → t winner b ∨ ∃ c, t winner c ∧ t c b)

theorem unique_award_implies_all_defeated 
  {α : Type} (t : Tournament α) (winner : α) :
  (∀ a b : α, a ≠ b → t a b ∨ t b a) →
  (∀ x : α, Award α t x ↔ x = winner) →
  (∀ b : α, b ≠ winner → t winner b) :=
sorry

end NUMINAMATH_CALUDE_unique_award_implies_all_defeated_l1195_119522


namespace NUMINAMATH_CALUDE_triangle_area_with_cosine_root_l1195_119575

theorem triangle_area_with_cosine_root (a b : ℝ) (cos_theta : ℝ) : 
  a = 3 → b = 5 → 
  5 * cos_theta^2 - 7 * cos_theta - 6 = 0 →
  cos_theta ≤ 1 →
  (1/2) * a * b * Real.sqrt (1 - cos_theta^2) = 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_with_cosine_root_l1195_119575


namespace NUMINAMATH_CALUDE_water_tank_problem_l1195_119579

theorem water_tank_problem (c : ℝ) (h1 : c > 0) : 
  let w := c / 3
  let w' := w + 5
  let w'' := w' + 4
  (w / c = 1 / 3) ∧ (w' / c = 2 / 5) → w'' / c = 34 / 75 := by
sorry

end NUMINAMATH_CALUDE_water_tank_problem_l1195_119579


namespace NUMINAMATH_CALUDE_intersection_with_complement_l1195_119545

open Set

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4}

-- Define set P
def P : Set Nat := {1, 2}

-- Define set Q
def Q : Set Nat := {2, 3}

-- Theorem statement
theorem intersection_with_complement : P ∩ (U \ Q) = {1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l1195_119545


namespace NUMINAMATH_CALUDE_no_interior_projection_on_all_sides_l1195_119570

/-- A triangle in a 2D plane -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The circumcircle of a triangle -/
def circumcircle (t : Triangle) : Set (ℝ × ℝ) := sorry

/-- Perpendicular projection of a point onto a line segment -/
def perp_projection (P : ℝ × ℝ) (A B : ℝ × ℝ) : ℝ × ℝ := sorry

/-- Check if a point is an interior point of a line segment -/
def is_interior (P A B : ℝ × ℝ) : Prop := sorry

/-- Main theorem -/
theorem no_interior_projection_on_all_sides (t : Triangle) :
  ¬ ∃ P ∈ circumcircle t,
    (is_interior (perp_projection P t.A t.B) t.A t.B) ∧
    (is_interior (perp_projection P t.B t.C) t.B t.C) ∧
    (is_interior (perp_projection P t.C t.A) t.C t.A) :=
sorry

end NUMINAMATH_CALUDE_no_interior_projection_on_all_sides_l1195_119570


namespace NUMINAMATH_CALUDE_john_burritos_per_day_l1195_119543

theorem john_burritos_per_day 
  (boxes : ℕ) 
  (burritos_per_box : ℕ) 
  (days : ℕ) 
  (remaining : ℕ) 
  (h1 : boxes = 3) 
  (h2 : burritos_per_box = 20) 
  (h3 : days = 10) 
  (h4 : remaining = 10) : 
  (boxes * burritos_per_box - boxes * burritos_per_box / 3 - remaining) / days = 3 := by
  sorry

end NUMINAMATH_CALUDE_john_burritos_per_day_l1195_119543


namespace NUMINAMATH_CALUDE_emilys_waist_size_conversion_l1195_119586

/-- Conversion of Emily's waist size from inches to centimeters -/
theorem emilys_waist_size_conversion (inches_per_foot : ℝ) (cm_per_foot : ℝ) (waist_inches : ℝ) :
  inches_per_foot = 12 →
  cm_per_foot = 30.48 →
  waist_inches = 28 →
  ∃ (waist_cm : ℝ), abs (waist_cm - (waist_inches / inches_per_foot * cm_per_foot)) < 0.1 ∧ waist_cm = 71.1 := by
  sorry

end NUMINAMATH_CALUDE_emilys_waist_size_conversion_l1195_119586


namespace NUMINAMATH_CALUDE_condition_relationship_l1195_119573

theorem condition_relationship (p q r s : Prop) 
  (h1 : (r → q) ∧ ¬(q → r))  -- q is necessary but not sufficient for r
  (h2 : (s ↔ r))             -- s is sufficient and necessary for r
  : (s → q) ∧ ¬(q → s) :=    -- s is sufficient but not necessary for q
by sorry

end NUMINAMATH_CALUDE_condition_relationship_l1195_119573


namespace NUMINAMATH_CALUDE_blue_given_not_red_probability_l1195_119581

-- Define the total number of balls
def total_balls : ℕ := 20

-- Define the number of red balls
def red_balls : ℕ := 5

-- Define the number of yellow balls
def yellow_balls : ℕ := 5

-- Define the number of blue balls
def blue_balls : ℕ := 10

-- Define the number of non-red balls
def non_red_balls : ℕ := yellow_balls + blue_balls

-- Theorem: The probability of drawing a blue ball given that it's not red is 2/3
theorem blue_given_not_red_probability : 
  (blue_balls : ℚ) / (non_red_balls : ℚ) = 2 / 3 :=
sorry

end NUMINAMATH_CALUDE_blue_given_not_red_probability_l1195_119581


namespace NUMINAMATH_CALUDE_cube_sum_reciprocal_l1195_119559

theorem cube_sum_reciprocal (a : ℝ) (h : (a + 1/a)^2 = 5) :
  a^3 + 1/a^3 = 2 * Real.sqrt 5 ∨ a^3 + 1/a^3 = -2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_reciprocal_l1195_119559


namespace NUMINAMATH_CALUDE_chess_players_never_lost_to_ai_l1195_119503

theorem chess_players_never_lost_to_ai (total_players : ℕ) (players_lost : ℕ) :
  total_players = 40 →
  players_lost = 30 →
  (total_players - players_lost : ℚ) / total_players = 1/4 := by
sorry

end NUMINAMATH_CALUDE_chess_players_never_lost_to_ai_l1195_119503


namespace NUMINAMATH_CALUDE_round_trip_average_speed_l1195_119577

/-- Proves that the average speed of a round trip is 34 mph, given that:
    1. The speed from A to B is 51 mph
    2. The return trip from B to A takes twice as long -/
theorem round_trip_average_speed : ∀ (distance : ℝ) (time : ℝ),
  distance > 0 → time > 0 →
  distance = 51 * time →
  (2 * distance) / (3 * time) = 34 :=
by sorry

end NUMINAMATH_CALUDE_round_trip_average_speed_l1195_119577


namespace NUMINAMATH_CALUDE_pipe_filling_time_l1195_119582

theorem pipe_filling_time (fill_rate_A fill_rate_B fill_rate_C : ℝ) 
  (h1 : fill_rate_A + fill_rate_B + fill_rate_C = 1 / 5)
  (h2 : fill_rate_B = 2 * fill_rate_A)
  (h3 : fill_rate_C = 2 * fill_rate_B) :
  1 / fill_rate_A = 35 := by
  sorry

end NUMINAMATH_CALUDE_pipe_filling_time_l1195_119582


namespace NUMINAMATH_CALUDE_problem_solution_l1195_119504

theorem problem_solution (a m n : ℚ) : 
  (∀ x, (a * x - 3) * (2 * x + 1) - 4 * x^2 + m = (a - 6) * x) → 
  a * n + m * n = 1 → 
  2 * n^3 - 9 * n^2 + 8 * n = 157 / 125 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1195_119504


namespace NUMINAMATH_CALUDE_light_reflection_l1195_119591

/-- A beam of light passing through a point and reflecting off a circle --/
structure LightBeam where
  M : ℝ × ℝ
  C : Set (ℝ × ℝ)

/-- Definition of the circle C --/
def is_circle (C : Set (ℝ × ℝ)) : Prop :=
  ∀ (x y : ℝ), (x, y) ∈ C ↔ x^2 + (y - 7)^2 = 25

/-- Definition of the reflected light ray equation --/
def reflected_ray_equation (x y : ℝ) : Prop :=
  x + y - 7 = 0

/-- Definition of the range of incident point A --/
def incident_point_range (A : ℝ) : Prop :=
  1 ≤ A ∧ A ≤ 23/2

/-- Main theorem --/
theorem light_reflection (beam : LightBeam) 
  (h_M : beam.M = (25, 18))
  (h_C : is_circle beam.C) :
  (∀ x y, reflected_ray_equation x y ↔ 
    (x, y) ∈ {p | ∃ t, p = ((1-t) * 25 + t * 0, (1-t) * (-18) + t * 7) ∧ 0 ≤ t ∧ t ≤ 1}) ∧
  (∀ A, incident_point_range A ↔ 
    ∃ (k : ℝ), (A, 0) ∈ {p | ∃ t, p = ((1-t) * 25 + t * A, (1-t) * (-18) + t * 0) ∧ 0 ≤ t ∧ t ≤ 1} ∧
               (0, 7) ∈ {p | ∃ t, p = ((1-t) * A + t * 0, (1-t) * 0 + t * 7) ∧ 0 ≤ t ∧ t ≤ 1}) :=
sorry

end NUMINAMATH_CALUDE_light_reflection_l1195_119591


namespace NUMINAMATH_CALUDE_isosceles_triangle_area_l1195_119592

/-- An isosceles triangle with two sides of length 5 and a median to the base of length 4 has an area of 12 -/
theorem isosceles_triangle_area (a b c : ℝ) (m : ℝ) (h_isosceles : a = b) (h_side : a = 5) (h_median : m = 4) : 
  (1/2 : ℝ) * c * m = 12 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_area_l1195_119592


namespace NUMINAMATH_CALUDE_rationalize_denominator_sqrt5_l1195_119507

theorem rationalize_denominator_sqrt5 :
  ∃ (A B C : ℤ),
    (A = -9 ∧ B = -4 ∧ C = 5) ∧
    (A * B * C = 180) ∧
    ∃ (x : ℝ),
      x = (2 + Real.sqrt 5) / (2 - Real.sqrt 5) ∧
      x = A + B * Real.sqrt C := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_sqrt5_l1195_119507


namespace NUMINAMATH_CALUDE_servant_payment_theorem_l1195_119569

/-- Calculates the money received by a servant who worked for 9 months, given a yearly salary and uniform value -/
def servant_payment (yearly_salary : ℚ) (uniform_value : ℚ) : ℚ :=
  (yearly_salary * 9 / 12) - uniform_value

/-- The servant payment theorem -/
theorem servant_payment_theorem (yearly_salary : ℚ) (uniform_value : ℚ) 
  (h1 : yearly_salary = 500)
  (h2 : uniform_value = 300) :
  servant_payment yearly_salary uniform_value = 75.03 := by
  sorry

#eval servant_payment 500 300

end NUMINAMATH_CALUDE_servant_payment_theorem_l1195_119569


namespace NUMINAMATH_CALUDE_n_value_proof_l1195_119526

theorem n_value_proof (n : ℕ) 
  (h1 : ∃ k : ℕ, 31 * 13 * n = k)
  (h2 : ∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ 2*x + 2*y + z = n)
  (h3 : (Finset.filter (λ (t : ℕ × ℕ × ℕ) => 
         t.1 > 0 ∧ t.2.1 > 0 ∧ t.2.2 > 0 ∧ 2*t.1 + 2*t.2.1 + t.2.2 = n) 
         (Finset.product (Finset.range n) (Finset.product (Finset.range n) (Finset.range n)))).card = 28) :
  n = 17 ∨ n = 18 := by
sorry

end NUMINAMATH_CALUDE_n_value_proof_l1195_119526


namespace NUMINAMATH_CALUDE_integer_sum_and_square_is_ten_l1195_119548

theorem integer_sum_and_square_is_ten (N : ℤ) : N^2 + N = 10 → N = 2 ∨ N = -5 := by
  sorry

end NUMINAMATH_CALUDE_integer_sum_and_square_is_ten_l1195_119548


namespace NUMINAMATH_CALUDE_problem_statement_l1195_119593

theorem problem_statement (a b c : ℝ) 
  (h_diff1 : a ≠ b) (h_diff2 : b ≠ c) (h_diff3 : a ≠ c)
  (h_eq : Real.sqrt (a^3 * (b-a)^3) - Real.sqrt (a^3 * (c-a)^3) = Real.sqrt (a-b) - Real.sqrt (c-a)) :
  a^2 + b^2 + c^2 - 2*a*b + 2*b*c - 2*a*c = 0 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l1195_119593


namespace NUMINAMATH_CALUDE_log_equality_difference_l1195_119564

theorem log_equality_difference (a b c d : ℕ+) 
  (h1 : (Real.log b) / (Real.log a) = (Real.log d) / (Real.log c))
  (h2 : a - c = 9) : 
  b - d = 93 := by sorry

end NUMINAMATH_CALUDE_log_equality_difference_l1195_119564


namespace NUMINAMATH_CALUDE_opposite_of_negative_three_halves_l1195_119500

theorem opposite_of_negative_three_halves :
  -((-3 : ℚ) / 2) = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_three_halves_l1195_119500


namespace NUMINAMATH_CALUDE_discount_equation_l1195_119540

/-- Represents the discount scenario for a clothing item -/
structure DiscountScenario where
  original_price : ℝ
  final_price : ℝ
  discount_percentage : ℝ

/-- Theorem stating the relationship between original price, discount, and final price -/
theorem discount_equation (scenario : DiscountScenario) 
  (h1 : scenario.original_price = 280)
  (h2 : scenario.final_price = 177)
  (h3 : scenario.discount_percentage ≥ 0)
  (h4 : scenario.discount_percentage < 1) :
  scenario.original_price * (1 - scenario.discount_percentage)^2 = scenario.final_price := by
  sorry

#check discount_equation

end NUMINAMATH_CALUDE_discount_equation_l1195_119540


namespace NUMINAMATH_CALUDE_work_completion_theorem_l1195_119529

/-- Calculates the number of additional workers needed to complete a job earlier -/
def additional_workers (initial_workers : ℕ) (initial_days : ℕ) (actual_days : ℕ) : ℕ :=
  (initial_workers * initial_days / actual_days) - initial_workers

theorem work_completion_theorem (initial_workers : ℕ) (initial_days : ℕ) (actual_days : ℕ) 
  (h1 : initial_workers = 30)
  (h2 : initial_days = 8)
  (h3 : actual_days = 5) :
  additional_workers initial_workers initial_days actual_days = 18 := by
  sorry

#eval additional_workers 30 8 5

end NUMINAMATH_CALUDE_work_completion_theorem_l1195_119529


namespace NUMINAMATH_CALUDE_vector_add_scale_l1195_119585

/-- Given two 3D vectors, prove that adding them and scaling the result by 2 yields the expected vector -/
theorem vector_add_scale (v1 v2 : Fin 3 → ℝ) (h1 : v1 = ![- 3, 2, 5]) (h2 : v2 = ![4, 7, - 3]) :
  (2 • (v1 + v2)) = ![2, 18, 4] := by
  sorry

end NUMINAMATH_CALUDE_vector_add_scale_l1195_119585


namespace NUMINAMATH_CALUDE_karls_trip_distance_l1195_119508

-- Define the problem parameters
def miles_per_gallon : ℚ := 30
def tank_capacity : ℚ := 16
def initial_distance : ℚ := 420
def gas_bought : ℚ := 10
def final_tank_fraction : ℚ := 3/4

-- Theorem statement
theorem karls_trip_distance :
  let initial_gas_used : ℚ := initial_distance / miles_per_gallon
  let remaining_gas : ℚ := tank_capacity - initial_gas_used
  let gas_after_refill : ℚ := remaining_gas + gas_bought
  let final_gas : ℚ := tank_capacity * final_tank_fraction
  gas_after_refill = final_gas →
  initial_distance = 420 := by
sorry

end NUMINAMATH_CALUDE_karls_trip_distance_l1195_119508


namespace NUMINAMATH_CALUDE_complex_square_root_of_negative_two_l1195_119518

theorem complex_square_root_of_negative_two (z : ℂ) : z^2 + 2 = 0 → z = Complex.I * Real.sqrt 2 ∨ z = -Complex.I * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_square_root_of_negative_two_l1195_119518


namespace NUMINAMATH_CALUDE_last_four_digits_of_m_l1195_119527

theorem last_four_digits_of_m (M : ℕ) (h1 : M > 0) 
  (h2 : ∃ (a b c d e : ℕ), a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 ∧
    M % 100000 = a * 10000 + b * 1000 + c * 100 + d * 10 + e ∧
    (M^2) % 100000 = a * 10000 + b * 1000 + c * 100 + d * 10 + e) :
  M % 10000 = 9687 := by
sorry

end NUMINAMATH_CALUDE_last_four_digits_of_m_l1195_119527


namespace NUMINAMATH_CALUDE_system_solution_l1195_119514

theorem system_solution (a b x y : ℝ) : 
  (2 * a - 3 * b = 13 ∧ 3 * a + 5 * b = 30.9 ∧ a = 8.3 ∧ b = 1.2) →
  (2 * (x + 2) - 3 * (y - 1) = 13 ∧ 3 * (x + 2) + 5 * (y - 1) = 30.9) →
  (x = 6.3 ∧ y = 2.2) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1195_119514


namespace NUMINAMATH_CALUDE_floor_tiling_l1195_119558

/-- 
Theorem: For an n × n floor to be completely covered by an equal number of 2 × 2 and 2 × 1 tiles, 
n must be divisible by 6.
-/
theorem floor_tiling (n : ℕ) : 
  (∃ x : ℕ, n * n = 6 * x) ↔ 6 ∣ n :=
by sorry

end NUMINAMATH_CALUDE_floor_tiling_l1195_119558


namespace NUMINAMATH_CALUDE_amy_hair_length_l1195_119531

/-- Amy's hair length before the haircut -/
def hair_length_before : ℕ := 11

/-- Amy's hair length after the haircut -/
def hair_length_after : ℕ := 7

/-- Length of hair cut off -/
def hair_cut_off : ℕ := 4

/-- Theorem: Amy's hair length before the haircut was 11 inches -/
theorem amy_hair_length : hair_length_before = hair_length_after + hair_cut_off := by
  sorry

end NUMINAMATH_CALUDE_amy_hair_length_l1195_119531


namespace NUMINAMATH_CALUDE_quadratic_origin_condition_l1195_119576

/-- A quadratic function passing through the origin -/
def passes_through_origin (m : ℝ) : Prop :=
  ∃ x y : ℝ, y = m * x^2 + x + m * (m - 2) ∧ x = 0 ∧ y = 0

/-- The theorem stating the conditions for the quadratic function to pass through the origin -/
theorem quadratic_origin_condition :
  ∀ m : ℝ, passes_through_origin m ↔ m = 2 ∨ m = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_origin_condition_l1195_119576


namespace NUMINAMATH_CALUDE_two_numbers_difference_l1195_119524

theorem two_numbers_difference (x y : ℝ) 
  (sum_eq : x + y = 10) 
  (square_diff_eq : x^2 - y^2 = 40) : 
  |x - y| = 4 := by
sorry

end NUMINAMATH_CALUDE_two_numbers_difference_l1195_119524


namespace NUMINAMATH_CALUDE_cone_surface_area_l1195_119574

/-- A cone with slant height 2 and lateral surface unfolding into a semicircle has surface area 3π -/
theorem cone_surface_area (h : ℝ) (r : ℝ) : 
  h = 2 → -- slant height is 2
  2 * π * r = 2 * π → -- lateral surface unfolds into a semicircle (circumference of base equals arc length of semicircle)
  π * r * (r + h) = 3 * π := by
  sorry


end NUMINAMATH_CALUDE_cone_surface_area_l1195_119574


namespace NUMINAMATH_CALUDE_seven_numbers_divisible_by_three_l1195_119556

theorem seven_numbers_divisible_by_three (S : Finset ℕ) (h : S.card = 7) :
  ∃ (a b c : ℕ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ (a + b + c) % 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_seven_numbers_divisible_by_three_l1195_119556


namespace NUMINAMATH_CALUDE_projection_length_l1195_119532

def vector_a : ℝ × ℝ := (3, 4)
def vector_b : ℝ × ℝ := (0, 1)

theorem projection_length :
  let a := vector_a
  let b := vector_b
  (a.1 * b.1 + a.2 * b.2) / Real.sqrt (b.1^2 + b.2^2) = 4 := by sorry

end NUMINAMATH_CALUDE_projection_length_l1195_119532


namespace NUMINAMATH_CALUDE_triangle_side_length_l1195_119525

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  -- Conditions
  (2 * b = a + c) →  -- arithmetic sequence
  (B = π / 6) →  -- 30° in radians
  (1 / 2 * a * c * Real.sin B = 3 / 2) →  -- area of triangle
  -- Conclusion
  b = Real.sqrt 3 + 1 := by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1195_119525


namespace NUMINAMATH_CALUDE_sqrt_sum_problem_l1195_119544

theorem sqrt_sum_problem (a b : ℝ) 
  (h1 : Real.sqrt 44 = 2 * Real.sqrt a) 
  (h2 : Real.sqrt 54 = 3 * Real.sqrt b) : 
  a + b = 17 := by
sorry

end NUMINAMATH_CALUDE_sqrt_sum_problem_l1195_119544


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1195_119516

/-- If the solution set of ax^2 + bx + c < 0 (a ≠ 0) is R, then a < 0 and b^2 - 4ac < 0 -/
theorem quadratic_inequality_solution_set (a b c : ℝ) : 
  a ≠ 0 → 
  (∀ x : ℝ, a * x^2 + b * x + c < 0) → 
  a < 0 ∧ b^2 - 4 * a * c < 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1195_119516


namespace NUMINAMATH_CALUDE_complex_root_theorem_l1195_119568

theorem complex_root_theorem (z : ℂ) (p : ℝ) : 
  (z^2 + 2*z + p = 0) → (Complex.abs z = 2) → (p = 4) := by
  sorry

end NUMINAMATH_CALUDE_complex_root_theorem_l1195_119568


namespace NUMINAMATH_CALUDE_gcd_37_power_plus_one_l1195_119589

theorem gcd_37_power_plus_one (h : Prime 37) : 
  Nat.gcd (37^11 + 1) (37^11 + 37^3 + 1) = 1 := by
sorry

end NUMINAMATH_CALUDE_gcd_37_power_plus_one_l1195_119589


namespace NUMINAMATH_CALUDE_correct_equation_only_E_is_true_l1195_119501

theorem correct_equation : 15618 = 1 + 5^6 - 1 * 8 := by
  sorry

-- The following definitions represent the conditions from the original problem
def equation_A : Prop := 15614 = 1 + 5^6 - 1 * 4
def equation_B : Prop := 15615 = 1 + 5^6 - 1 * 5
def equation_C : Prop := 15616 = 1 + 5^6 - 1 * 6
def equation_D : Prop := 15617 = 1 + 5^6 - 1 * 7
def equation_E : Prop := 15618 = 1 + 5^6 - 1 * 8

-- This theorem states that equation_E is the only true equation among the given options
theorem only_E_is_true : 
  ¬equation_A ∧ ¬equation_B ∧ ¬equation_C ∧ ¬equation_D ∧ equation_E := by
  sorry

end NUMINAMATH_CALUDE_correct_equation_only_E_is_true_l1195_119501


namespace NUMINAMATH_CALUDE_raccoon_lock_problem_l1195_119547

theorem raccoon_lock_problem (first_lock_duration second_lock_duration : ℕ) : 
  first_lock_duration = 5 →
  second_lock_duration < 3 * first_lock_duration →
  5 * second_lock_duration = 60 →
  3 * first_lock_duration - second_lock_duration = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_raccoon_lock_problem_l1195_119547


namespace NUMINAMATH_CALUDE_sqrt_x_fifth_power_eq_1024_l1195_119530

theorem sqrt_x_fifth_power_eq_1024 (x : ℝ) : (Real.sqrt x) ^ 5 = 1024 → x = 16 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_fifth_power_eq_1024_l1195_119530


namespace NUMINAMATH_CALUDE_even_function_inequality_range_l1195_119551

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def monotone_increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x ≤ y → f x ≤ f y

theorem even_function_inequality_range
  (f : ℝ → ℝ)
  (h_even : is_even_function f)
  (h_mono : monotone_increasing_on f (Set.Ici 0)) :
  {x : ℝ | f (2*x - 1) < f (-1)} = Set.Ioo 0 1 := by
sorry

end NUMINAMATH_CALUDE_even_function_inequality_range_l1195_119551


namespace NUMINAMATH_CALUDE_polygon_interior_angles_l1195_119546

theorem polygon_interior_angles (n : ℕ) : 
  (n - 2) * 180 = 720 → n = 6 := by
sorry

end NUMINAMATH_CALUDE_polygon_interior_angles_l1195_119546


namespace NUMINAMATH_CALUDE_average_fish_caught_l1195_119552

def aang_fish : ℕ := 7
def sokka_fish : ℕ := 5
def toph_fish : ℕ := 12
def total_people : ℕ := 3

theorem average_fish_caught :
  (aang_fish + sokka_fish + toph_fish) / total_people = 8 := by
  sorry

end NUMINAMATH_CALUDE_average_fish_caught_l1195_119552


namespace NUMINAMATH_CALUDE_no_intersection_quadratic_sets_l1195_119596

theorem no_intersection_quadratic_sets (A B : ℤ) :
  ∃ C : ℤ, ∀ x y : ℤ, x^2 + A*x + B ≠ 2*y^2 + 2*y + C := by
  sorry

end NUMINAMATH_CALUDE_no_intersection_quadratic_sets_l1195_119596


namespace NUMINAMATH_CALUDE_zachary_needs_additional_money_l1195_119521

def football_cost : ℚ := 3.75
def shorts_cost : ℚ := 2.40
def shoes_cost : ℚ := 11.85
def zachary_money : ℚ := 10.00

theorem zachary_needs_additional_money :
  football_cost + shorts_cost + shoes_cost - zachary_money = 7.00 := by
  sorry

end NUMINAMATH_CALUDE_zachary_needs_additional_money_l1195_119521


namespace NUMINAMATH_CALUDE_peaches_picked_proof_l1195_119542

/-- The number of peaches Mike had initially -/
def initial_peaches : ℕ := 34

/-- The total number of peaches Mike has now -/
def total_peaches : ℕ := 86

/-- The number of peaches Mike picked -/
def picked_peaches : ℕ := total_peaches - initial_peaches

theorem peaches_picked_proof :
  picked_peaches = total_peaches - initial_peaches :=
by sorry

end NUMINAMATH_CALUDE_peaches_picked_proof_l1195_119542


namespace NUMINAMATH_CALUDE_bales_in_barn_l1195_119533

/-- The number of bales in the barn after Tim added a couple more -/
def total_bales (initial_bales : ℕ) (added_bales : ℕ) : ℕ :=
  initial_bales + added_bales

/-- A couple is defined as 2 -/
def couple : ℕ := 2

theorem bales_in_barn (initial_bales : ℕ) (h : initial_bales = 540) :
  total_bales initial_bales couple = 542 := by
  sorry

end NUMINAMATH_CALUDE_bales_in_barn_l1195_119533


namespace NUMINAMATH_CALUDE_f_property_and_g_monotonicity_l1195_119599

-- Define the function f
def f : ℝ → ℝ := fun x ↦ x^2 + x + 1

-- Define the function g
def g (a : ℝ) : ℝ → ℝ := fun x ↦ |f x - a * x + 3|

theorem f_property_and_g_monotonicity :
  (∀ x : ℝ, f (1 - x) = x^2 - 3*x + 3) ∧
  (∀ x : ℝ, f x = x^2 + x + 1) ∧
  (∀ a : ℝ, (∀ x y : ℝ, 1 ≤ x ∧ x ≤ y ∧ y ≤ 3 → g a x ≤ g a y) ↔ 
    (a ≤ 3 ∨ 6 ≤ a)) :=
by sorry

end NUMINAMATH_CALUDE_f_property_and_g_monotonicity_l1195_119599


namespace NUMINAMATH_CALUDE_library_shelving_l1195_119523

theorem library_shelving (jason_books_per_time lexi_books_per_time total_books : ℕ) :
  jason_books_per_time = 6 →
  total_books = 102 →
  total_books % jason_books_per_time = 0 →
  total_books % lexi_books_per_time = 0 →
  total_books / jason_books_per_time = total_books / lexi_books_per_time →
  lexi_books_per_time = 6 := by
  sorry

end NUMINAMATH_CALUDE_library_shelving_l1195_119523


namespace NUMINAMATH_CALUDE_julien_contribution_julien_contribution_proof_l1195_119598

/-- The amount Julien needs to contribute to buy a pie -/
theorem julien_contribution (pie_cost : ℝ) (lucas_money : ℝ) (exchange_rate : ℝ) : ℝ :=
  pie_cost - lucas_money / exchange_rate

/-- Proof of Julien's required contribution -/
theorem julien_contribution_proof :
  julien_contribution 12 10 1.5 = 16 / 3 := by
  sorry

end NUMINAMATH_CALUDE_julien_contribution_julien_contribution_proof_l1195_119598


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_equation_l1195_119580

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a, b > 0,
    if its eccentricity e and the slope of its asymptotes k satisfy e = √2 |k|,
    then the equation of its asymptotes is y = ±x -/
theorem hyperbola_asymptotes_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let e := Real.sqrt ((a^2 + b^2) / a^2)
  let k := b / a
  e = Real.sqrt 2 * abs k →
  ∃ (f : ℝ → ℝ), (∀ x, f x = x ∨ f x = -x) ∧
    (∀ x y, y = f x ↔ (x^2 / a^2 - y^2 / b^2 = 1 → False)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_equation_l1195_119580


namespace NUMINAMATH_CALUDE_derivative_at_one_l1195_119590

open Real

theorem derivative_at_one (f : ℝ → ℝ) (f' : ℝ → ℝ) :
  (∀ x, deriv f x = f' x) →
  (∀ x, f x = 2 * x * f' 1 + log x) →
  f' 1 = -1 := by
sorry

end NUMINAMATH_CALUDE_derivative_at_one_l1195_119590


namespace NUMINAMATH_CALUDE_ellipse_equation_l1195_119537

-- Define the ellipse
structure Ellipse where
  foci : (ℝ × ℝ) × (ℝ × ℝ)
  distance_sum : ℝ

-- Define the standard form of an ellipse equation
def standard_ellipse_equation (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Theorem statement
theorem ellipse_equation (e : Ellipse) (x y : ℝ) : 
  e.foci = ((-4, 0), (4, 0)) →
  e.distance_sum = 10 →
  standard_ellipse_equation 25 9 x y :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l1195_119537


namespace NUMINAMATH_CALUDE_parallelepipeds_from_four_points_l1195_119510

/-- A type representing a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A predicate that checks if four points are coplanar -/
def coplanar (p1 p2 p3 p4 : Point3D) : Prop :=
  ∃ (a b c d : ℝ), a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0 ∨ d ≠ 0 ∧
    a * p1.x + b * p1.y + c * p1.z + d = 0 ∧
    a * p2.x + b * p2.y + c * p2.z + d = 0 ∧
    a * p3.x + b * p3.y + c * p3.z + d = 0 ∧
    a * p4.x + b * p4.y + c * p4.z + d = 0

/-- A function that counts the number of distinct parallelepipeds -/
def count_parallelepipeds (p1 p2 p3 p4 : Point3D) : ℕ :=
  sorry -- The actual implementation is not needed for the theorem statement

/-- Theorem stating that 4 non-coplanar points form 29 distinct parallelepipeds -/
theorem parallelepipeds_from_four_points (p1 p2 p3 p4 : Point3D) 
  (h : ¬coplanar p1 p2 p3 p4) : 
  count_parallelepipeds p1 p2 p3 p4 = 29 := by
  sorry

end NUMINAMATH_CALUDE_parallelepipeds_from_four_points_l1195_119510


namespace NUMINAMATH_CALUDE_no_simultaneous_integer_roots_l1195_119513

theorem no_simultaneous_integer_roots :
  ¬ ∃ (b c : ℝ),
    (∃ (k l m n : ℤ),
      (k ≠ l ∧ m ≠ n) ∧
      (∀ x : ℝ, x^2 + b*x + c = 0 ↔ x = k ∨ x = l) ∧
      (∀ x : ℝ, 2*x^2 + (b+1)*x + (c+1) = 0 ↔ x = m ∨ x = n)) :=
by sorry

end NUMINAMATH_CALUDE_no_simultaneous_integer_roots_l1195_119513


namespace NUMINAMATH_CALUDE_negative_integer_solution_exists_l1195_119515

theorem negative_integer_solution_exists : ∃ (x : ℤ), x < 0 ∧ 3 * x + 13 ≥ 0 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_negative_integer_solution_exists_l1195_119515


namespace NUMINAMATH_CALUDE_crate_stacking_probability_l1195_119534

def crate_height : Fin 3 → ℕ
| 0 => 2
| 1 => 3
| 2 => 5

def total_combinations : ℕ := 3^10

def valid_combinations : ℕ := 2940

theorem crate_stacking_probability :
  (valid_combinations : ℚ) / total_combinations = 980 / 19683 :=
sorry

end NUMINAMATH_CALUDE_crate_stacking_probability_l1195_119534


namespace NUMINAMATH_CALUDE_system_solution_l1195_119511

theorem system_solution :
  let solutions : List (ℝ × ℝ × ℝ) := [
    (1, 2, 3), (1, 5, -3), (3, -2, 5),
    (3, 3, -5), (6, -5, 2), (6, -3, -2)
  ]
  ∀ (x y z : ℝ),
    (3*x + 2*y + z = 10 ∧
     3*x^2 + 4*x*y + 2*x*z + y^2 + y*z = 27 ∧
     x^3 + 2*x^2*y + x^2*z + x*y^2 + x*y*z = 18) ↔
    (x, y, z) ∈ solutions := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1195_119511


namespace NUMINAMATH_CALUDE_equation_solution_system_solution_l1195_119571

-- Define the equation
def equation (x : ℚ) : Prop := 64 * (x - 1)^3 + 27 = 0

-- Define the system of equations
def system (x y : ℚ) : Prop := x + y = 3 ∧ 2*x - 3*y = 6

-- Theorem for the equation
theorem equation_solution : ∃ x : ℚ, equation x ∧ x = 1/4 := by sorry

-- Theorem for the system of equations
theorem system_solution : ∃ x y : ℚ, system x y ∧ x = 3 ∧ y = 0 := by sorry

end NUMINAMATH_CALUDE_equation_solution_system_solution_l1195_119571


namespace NUMINAMATH_CALUDE_complex_number_problem_l1195_119535

theorem complex_number_problem :
  let z : ℂ := ((1 - I)^2 + 3*(1 + I)) / (2 - I)
  ∃ (a b : ℝ), z^2 + a*z + b = 1 - I ∧ z = 1 + I ∧ a = -3 ∧ b = 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_problem_l1195_119535


namespace NUMINAMATH_CALUDE_complex_square_in_fourth_quadrant_l1195_119588

theorem complex_square_in_fourth_quadrant (z : ℂ) :
  (z.re > 0 ∧ z.im < 0) →  -- z is in the fourth quadrant
  z^2 - 2*z + 2 = 0 →      -- z satisfies the given equation
  z^2 = -2*Complex.I :=    -- conclusion: z^2 = -2i
by
  sorry

end NUMINAMATH_CALUDE_complex_square_in_fourth_quadrant_l1195_119588


namespace NUMINAMATH_CALUDE_largest_four_digit_divisible_by_24_l1195_119512

theorem largest_four_digit_divisible_by_24 : 
  ∀ n : ℕ, n ≤ 9999 ∧ n ≥ 1000 ∧ n % 24 = 0 → n ≤ 9984 :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_divisible_by_24_l1195_119512


namespace NUMINAMATH_CALUDE_hook_all_of_one_color_l1195_119595

/-- Represents a square sheet on the table -/
structure Sheet where
  color : Nat
  deriving Repr

/-- Represents the rectangular table with sheets -/
structure Table where
  sheets : List Sheet
  num_colors : Nat
  deriving Repr

/-- Two sheets can be hooked together -/
def can_hook (s1 s2 : Sheet) : Prop := sorry

/-- All sheets of the same color can be hooked together using the given number of hooks -/
def can_hook_color (t : Table) (c : Nat) (hooks : Nat) : Prop := sorry

/-- For any k different colored sheets, two can be hooked together -/
axiom hook_property (t : Table) :
  ∀ (diff_colored_sheets : List Sheet),
    diff_colored_sheets.length = t.num_colors →
    (∀ (s1 s2 : Sheet), s1 ∈ diff_colored_sheets → s2 ∈ diff_colored_sheets → s1 ≠ s2 → s1.color ≠ s2.color) →
    ∃ (s1 s2 : Sheet), s1 ∈ diff_colored_sheets ∧ s2 ∈ diff_colored_sheets ∧ s1 ≠ s2 ∧ can_hook s1 s2

/-- Main theorem: It's possible to hook all sheets of a certain color using 2k-2 hooks -/
theorem hook_all_of_one_color (t : Table) (h : t.num_colors ≥ 2) :
  ∃ (c : Nat), can_hook_color t c (2 * t.num_colors - 2) := by
  sorry

end NUMINAMATH_CALUDE_hook_all_of_one_color_l1195_119595


namespace NUMINAMATH_CALUDE_sin_960_degrees_l1195_119549

theorem sin_960_degrees : Real.sin (960 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_960_degrees_l1195_119549


namespace NUMINAMATH_CALUDE_intersection_A_B_when_m_zero_range_of_m_for_necessary_condition_l1195_119583

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def B (m : ℝ) : Set ℝ := {x | (x - m + 1) * (x - m - 1) ≥ 0}

-- Define propositions p and q
def p (x : ℝ) : Prop := x^2 - 2*x - 3 < 0
def q (x m : ℝ) : Prop := (x - m + 1) * (x - m - 1) ≥ 0

-- Theorem for part I
theorem intersection_A_B_when_m_zero :
  A ∩ B 0 = {x | 1 ≤ x ∧ x < 3} := by sorry

-- Theorem for part II
theorem range_of_m_for_necessary_condition :
  (∀ x, p x → q x m) ∧ (∃ x, q x m ∧ ¬p x) ↔ m ≥ 4 ∨ m ≤ -2 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_when_m_zero_range_of_m_for_necessary_condition_l1195_119583


namespace NUMINAMATH_CALUDE_seventh_term_is_seven_l1195_119567

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  -- First term
  a : ℝ
  -- Common difference
  d : ℝ
  -- Sum of first five terms is 15
  sum_first_five : a + (a + d) + (a + 2*d) + (a + 3*d) + (a + 4*d) = 15
  -- Fourth term is 4
  fourth_term : a + 3*d = 4

/-- The seventh term of the arithmetic sequence is 7 -/
theorem seventh_term_is_seven (seq : ArithmeticSequence) : seq.a + 6*seq.d = 7 := by
  sorry

end NUMINAMATH_CALUDE_seventh_term_is_seven_l1195_119567


namespace NUMINAMATH_CALUDE_exam_pass_percentage_l1195_119566

/-- Given an examination where 740 students appeared and 481 failed,
    prove that 35% of students passed. -/
theorem exam_pass_percentage 
  (total_students : ℕ) 
  (failed_students : ℕ) 
  (h1 : total_students = 740)
  (h2 : failed_students = 481) : 
  (total_students - failed_students : ℚ) / total_students * 100 = 35 := by
sorry

end NUMINAMATH_CALUDE_exam_pass_percentage_l1195_119566


namespace NUMINAMATH_CALUDE_symmetric_points_sum_power_l1195_119506

/-- 
Given two points A(m,3) and B(4,n) that are symmetric about the y-axis,
prove that (m+n)^2015 = -1
-/
theorem symmetric_points_sum_power (m n : ℝ) : 
  (∃ (A B : ℝ × ℝ), A = (m, 3) ∧ B = (4, n) ∧ 
   A.1 = -B.1 ∧ A.2 = B.2) → 
  (m + n)^2015 = -1 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_power_l1195_119506


namespace NUMINAMATH_CALUDE_black_area_from_white_area_l1195_119572

/-- Represents a square divided into 9 equal smaller squares -/
structure DividedSquare where
  total_area : ℝ
  white_squares : ℕ
  black_squares : ℕ
  white_area : ℝ

/-- Theorem stating the relation between white and black areas in the divided square -/
theorem black_area_from_white_area (s : DividedSquare) 
  (h1 : s.white_squares + s.black_squares = 9)
  (h2 : s.white_squares = 5)
  (h3 : s.black_squares = 4)
  (h4 : s.white_area = 180) :
  s.total_area * (s.black_squares / 9 : ℝ) = 144 := by
sorry

end NUMINAMATH_CALUDE_black_area_from_white_area_l1195_119572


namespace NUMINAMATH_CALUDE_emily_trivia_score_l1195_119541

/-- Emily's trivia game score calculation -/
theorem emily_trivia_score (first_round : ℤ) (second_round : ℤ) (last_round : ℤ) 
  (h1 : first_round = 16)
  (h2 : second_round = 33)
  (h3 : last_round = -48) :
  first_round + second_round + last_round = 1 := by
  sorry

end NUMINAMATH_CALUDE_emily_trivia_score_l1195_119541


namespace NUMINAMATH_CALUDE_trig_equation_solution_l1195_119528

theorem trig_equation_solution (x : ℝ) :
  8.469 * (Real.sin x)^4 + 2 * (Real.cos x)^3 + 2 * (Real.sin x)^2 - Real.cos x + 1 = 0 →
  ∃ k : ℤ, x = π * (2 * k + 1) := by
sorry

end NUMINAMATH_CALUDE_trig_equation_solution_l1195_119528


namespace NUMINAMATH_CALUDE_complement_A_inter_B_range_of_a_l1195_119553

-- Define the sets A, B, and C
def A : Set ℝ := {x | 2 ≤ x ∧ x ≤ 6}
def B : Set ℝ := {x | 3 * x - 7 ≥ 8 - 2 * x}
def C (a : ℝ) : Set ℝ := {x | x ≤ a}

-- Theorem for the complement of A ∩ B
theorem complement_A_inter_B :
  (A ∩ B)ᶜ = {x | x < 3 ∨ x > 6} := by sorry

-- Theorem for the range of a
theorem range_of_a (a : ℝ) (h : A ∪ C a = C a) :
  a ≥ 6 := by sorry

end NUMINAMATH_CALUDE_complement_A_inter_B_range_of_a_l1195_119553


namespace NUMINAMATH_CALUDE_lost_card_sum_l1195_119578

theorem lost_card_sum (a b c d : ℝ) : 
  let sums := [a + b, a + c, a + d, b + c, b + d, c + d]
  (∃ (s : Finset ℝ), s ⊆ sums.toFinset ∧ s.card = 5 ∧ 
    (270 ∈ s ∧ 360 ∈ s ∧ 390 ∈ s ∧ 500 ∈ s ∧ 620 ∈ s)) →
  530 ∈ sums.toFinset :=
by sorry

end NUMINAMATH_CALUDE_lost_card_sum_l1195_119578


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1195_119505

theorem imaginary_part_of_z (z : ℂ) (h : z / (1 + Complex.I) = -1 / (2 * Complex.I)) :
  z.im = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1195_119505


namespace NUMINAMATH_CALUDE_backpack_price_change_l1195_119584

theorem backpack_price_change (P : ℝ) (x : ℝ) (h : P > 0) :
  P * (1 + x / 100) * (1 - x / 100) = 0.64 * P →
  x = 60 := by
  sorry

end NUMINAMATH_CALUDE_backpack_price_change_l1195_119584


namespace NUMINAMATH_CALUDE_dragon_jewels_l1195_119536

theorem dragon_jewels (x : ℕ) (h1 : x / 3 = 6) : x + 6 = 24 := by
  sorry

#check dragon_jewels

end NUMINAMATH_CALUDE_dragon_jewels_l1195_119536


namespace NUMINAMATH_CALUDE_quadratic_roots_l1195_119560

theorem quadratic_roots (x : ℝ) : x * (x + 1) = 3 * (x + 1) ↔ x = 3 ∨ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_l1195_119560


namespace NUMINAMATH_CALUDE_range_of_a_l1195_119565

-- Define the circle C
def circle_C (a : ℝ) (x y : ℝ) : Prop :=
  (x - a)^2 + (y - a + 2)^2 = 1

-- Define point A
def point_A : ℝ × ℝ := (0, 2)

-- Define the condition for point M
def condition_M (a : ℝ) : Prop :=
  ∃ x y : ℝ, circle_C a x y ∧ 
    (x^2 + (y - 2)^2) + (x^2 + y^2) = 10

-- The main theorem
theorem range_of_a :
  ∀ a : ℝ, condition_M a ↔ 0 ≤ a ∧ a ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1195_119565


namespace NUMINAMATH_CALUDE_triangle_has_obtuse_angle_l1195_119594

/-- A triangle with vertices A(1, 2), B(-3, 4), and C(0, -2) has an obtuse angle. -/
theorem triangle_has_obtuse_angle :
  let A : ℝ × ℝ := (1, 2)
  let B : ℝ × ℝ := (-3, 4)
  let C : ℝ × ℝ := (0, -2)
  let AB : ℝ := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  let BC : ℝ := Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2)
  let AC : ℝ := Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)
  BC^2 > AB^2 + AC^2 := by sorry

end NUMINAMATH_CALUDE_triangle_has_obtuse_angle_l1195_119594


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_99_l1195_119561

theorem greatest_prime_factor_of_99 : ∃ p : ℕ, Nat.Prime p ∧ p ∣ 99 ∧ ∀ q : ℕ, Nat.Prime q → q ∣ 99 → q ≤ p :=
  sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_of_99_l1195_119561


namespace NUMINAMATH_CALUDE_triangle_inequality_sum_l1195_119597

theorem triangle_inequality_sum (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  (a^2 + 2*b*c) / (b^2 + c^2) + (b^2 + 2*a*c) / (c^2 + a^2) + (c^2 + 2*a*b) / (a^2 + b^2) > 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_sum_l1195_119597


namespace NUMINAMATH_CALUDE_range_of_a_l1195_119550

theorem range_of_a (a : ℝ) : 
  (∃ x₀ ∈ Set.Icc 1 3, |x₀^2 - a*x₀ + 4| ≤ 3*x₀) → 1 ≤ a ∧ a ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1195_119550


namespace NUMINAMATH_CALUDE_factorial_multiple_implies_inequality_l1195_119519

theorem factorial_multiple_implies_inequality (a b : ℕ+) 
  (h : (a.val.factorial * b.val.factorial) % (a.val.factorial + b.val.factorial) = 0) : 
  3 * a.val ≥ 2 * b.val + 2 := by
  sorry

end NUMINAMATH_CALUDE_factorial_multiple_implies_inequality_l1195_119519


namespace NUMINAMATH_CALUDE_extended_morse_code_symbols_l1195_119538

def morse_symbols (n : Nat) : Nat :=
  3^n

theorem extended_morse_code_symbols :
  (morse_symbols 1) + (morse_symbols 2) + (morse_symbols 3) + (morse_symbols 4) = 120 := by
  sorry

end NUMINAMATH_CALUDE_extended_morse_code_symbols_l1195_119538


namespace NUMINAMATH_CALUDE_largest_last_digit_l1195_119520

def is_valid_series (s : List Nat) : Prop :=
  s.length = 2023 ∧
  s.head? = some 1 ∧
  ∀ i, i < s.length - 1 →
    let two_digit := s[i]! * 10 + s[i+1]!
    two_digit % 17 = 0 ∨ two_digit % 29 = 0 ∨ two_digit % 23 = 0

theorem largest_last_digit (s : List Nat) (h : is_valid_series s) :
  s.getLast? = some 2 := by
  sorry

end NUMINAMATH_CALUDE_largest_last_digit_l1195_119520

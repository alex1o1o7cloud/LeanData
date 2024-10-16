import Mathlib

namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l1763_176330

theorem simplify_and_evaluate (x : ℝ) (h : x = 4) :
  ((2 * x - 2) / x - 1) / ((x^2 - 4*x + 4) / (x^2 - x)) = 3/2 :=
by sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l1763_176330


namespace NUMINAMATH_CALUDE_initial_mean_calculation_l1763_176322

theorem initial_mean_calculation (n : ℕ) (initial_wrong : ℝ) (corrected : ℝ) (new_mean : ℝ) :
  n = 50 →
  initial_wrong = 23 →
  corrected = 48 →
  new_mean = 30.5 →
  (n : ℝ) * new_mean = (n : ℝ) * (n * new_mean - (corrected - initial_wrong)) / n :=
by sorry

end NUMINAMATH_CALUDE_initial_mean_calculation_l1763_176322


namespace NUMINAMATH_CALUDE_worker_a_completion_time_l1763_176310

/-- The number of days it takes for two workers to complete a job together -/
def combined_days : ℝ := 18

/-- The ratio of worker a's speed to worker b's speed -/
def speed_ratio : ℝ := 1.5

/-- The number of days it takes for worker a to complete the job alone -/
def days_a : ℝ := 30

theorem worker_a_completion_time : 
  1 / combined_days = 1 / days_a + 1 / (speed_ratio * days_a) :=
by sorry

end NUMINAMATH_CALUDE_worker_a_completion_time_l1763_176310


namespace NUMINAMATH_CALUDE_pen_collection_l1763_176306

theorem pen_collection (initial_pens : ℕ) (received_pens : ℕ) (given_away : ℕ) : 
  initial_pens = 5 → received_pens = 20 → given_away = 10 → 
  ((initial_pens + received_pens) * 2 - given_away) = 40 := by
  sorry

end NUMINAMATH_CALUDE_pen_collection_l1763_176306


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1763_176374

theorem polynomial_factorization (x : ℤ) :
  4 * (x + 4) * (x + 7) * (x + 9) * (x + 11) - 5 * x^2 = (2 * x^2 + 72 * x + 126)^2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1763_176374


namespace NUMINAMATH_CALUDE_total_students_proof_l1763_176309

/-- The number of students who knew about the event -/
def students_who_knew : ℕ := 40

/-- The frequency of students who knew about the event -/
def frequency : ℚ := 8/10

/-- The total number of students participating in the competition -/
def total_students : ℕ := 50

/-- Theorem stating that the total number of students is 50 given the conditions -/
theorem total_students_proof : 
  (students_who_knew : ℚ) / frequency = total_students := by sorry

end NUMINAMATH_CALUDE_total_students_proof_l1763_176309


namespace NUMINAMATH_CALUDE_gcd_12345_6789_l1763_176331

theorem gcd_12345_6789 : Nat.gcd 12345 6789 = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_12345_6789_l1763_176331


namespace NUMINAMATH_CALUDE_discount_percentage_proof_l1763_176373

theorem discount_percentage_proof (jacket_price shirt_price : ℝ) 
  (jacket_discount shirt_discount : ℝ) : 
  jacket_price = 100 →
  shirt_price = 50 →
  jacket_discount = 0.3 →
  shirt_discount = 0.6 →
  (jacket_price * jacket_discount + shirt_price * shirt_discount) / (jacket_price + shirt_price) = 0.4 := by
sorry

end NUMINAMATH_CALUDE_discount_percentage_proof_l1763_176373


namespace NUMINAMATH_CALUDE_largest_four_digit_divisible_by_88_l1763_176308

theorem largest_four_digit_divisible_by_88 : 
  ∀ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 88 = 0 → n ≤ 9944 :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_divisible_by_88_l1763_176308


namespace NUMINAMATH_CALUDE_adjacent_supplementary_angles_l1763_176357

theorem adjacent_supplementary_angles (angle_AOB angle_BOC : ℝ) : 
  angle_AOB + angle_BOC = 180 →
  angle_AOB = angle_BOC + 18 →
  angle_AOB = 99 := by
sorry

end NUMINAMATH_CALUDE_adjacent_supplementary_angles_l1763_176357


namespace NUMINAMATH_CALUDE_balance_difference_theorem_l1763_176305

def initial_deposit : ℝ := 10000
def jasmine_rate : ℝ := 0.04
def lucas_rate : ℝ := 0.06
def years : ℕ := 20

def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

def simple_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate * time)

theorem balance_difference_theorem :
  ∃ ε > 0, ε < 1 ∧
  (simple_interest initial_deposit lucas_rate years -
   compound_interest initial_deposit jasmine_rate years) - 89 < ε :=
sorry

end NUMINAMATH_CALUDE_balance_difference_theorem_l1763_176305


namespace NUMINAMATH_CALUDE_triangle_theorem_l1763_176360

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_theorem (t : Triangle) 
  (h1 : 2 * t.c * Real.sin t.C = (2 * t.b + t.a) * Real.sin t.B + (2 * t.a - 3 * t.b) * Real.sin t.A) :
  t.C = π / 3 ∧ (t.c = 4 → 4 < t.a + t.b ∧ t.a + t.b ≤ 8) := by
  sorry


end NUMINAMATH_CALUDE_triangle_theorem_l1763_176360


namespace NUMINAMATH_CALUDE_rectangle_only_convex_four_right_angles_l1763_176300

/-- A polygon is a set of points in the plane -/
def Polygon : Type := Set (ℝ × ℝ)

/-- A polygon is convex if for any two points in the polygon, the line segment between them is entirely contained within the polygon -/
def is_convex (p : Polygon) : Prop := sorry

/-- The number of sides in a polygon -/
def num_sides (p : Polygon) : ℕ := sorry

/-- The number of right angles in a polygon -/
def num_right_angles (p : Polygon) : ℕ := sorry

/-- A rectangle is a polygon with exactly four sides and four right angles -/
def is_rectangle (p : Polygon) : Prop :=
  num_sides p = 4 ∧ num_right_angles p = 4

theorem rectangle_only_convex_four_right_angles (p : Polygon) :
  is_convex p ∧ num_right_angles p = 4 → is_rectangle p :=
sorry

end NUMINAMATH_CALUDE_rectangle_only_convex_four_right_angles_l1763_176300


namespace NUMINAMATH_CALUDE_min_value_f_range_of_m_l1763_176312

noncomputable section

def f (x : ℝ) : ℝ := x * Real.log x + 2

def g (m : ℝ) (x : ℝ) : ℝ := x^2 - m*x

theorem min_value_f (t : ℝ) (ht : t > 0) :
  (if t ≥ 1/Real.exp 1 then
    IsMinOn f (Set.Icc t (t + 2)) (f t)
   else
    IsMinOn f (Set.Icc t (t + 2)) (f (1/Real.exp 1))) ∧
  (if t ≥ 1/Real.exp 1 then
    ∀ x ∈ Set.Icc t (t + 2), f x ≥ t * Real.log t + 2
   else
    ∀ x ∈ Set.Icc t (t + 2), f x ≥ -1/Real.exp 1 + 2) :=
sorry

theorem range_of_m :
  {m : ℝ | ∃ x₀ ∈ Set.Icc (1/Real.exp 1) (Real.exp 1),
    m * (Real.log x₀ + 1) + g m x₀ ≥ 2*x₀ + m} = Set.Iic (-1) :=
sorry

end NUMINAMATH_CALUDE_min_value_f_range_of_m_l1763_176312


namespace NUMINAMATH_CALUDE_largest_divisor_of_n_l1763_176318

theorem largest_divisor_of_n (n : ℕ) (h1 : n > 0) (h2 : 1080 ∣ n^2) : ∃ q : ℕ, q > 0 ∧ q ∣ n ∧ ∀ m : ℕ, m > 0 ∧ m ∣ n → m ≤ q ∧ q = 6 := by
  sorry

end NUMINAMATH_CALUDE_largest_divisor_of_n_l1763_176318


namespace NUMINAMATH_CALUDE_no_rational_roots_l1763_176337

-- Define the polynomial
def f (x : ℚ) : ℚ := 5 * x^3 - 4 * x^2 - 8 * x + 3

-- Theorem statement
theorem no_rational_roots : ∀ x : ℚ, f x ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_rational_roots_l1763_176337


namespace NUMINAMATH_CALUDE_inclination_angle_range_l1763_176399

/-- Given a line with equation x cos θ + √3 y + 2 = 0, 
    the range of its inclination angle α (where 0 ≤ α < π) 
    is [0, π/6] ∪ [5π/6, π) -/
theorem inclination_angle_range (θ : ℝ) (α : ℝ) :
  (∃ y : ℝ, x * Real.cos θ + Real.sqrt 3 * y + 2 = 0) →
  0 ≤ α ∧ α < π →
  α ∈ Set.union (Set.Icc 0 (π/6)) (Set.Ioc (5*π/6) π) :=
by sorry

end NUMINAMATH_CALUDE_inclination_angle_range_l1763_176399


namespace NUMINAMATH_CALUDE_game_of_thrones_percentage_l1763_176386

/-- Represents the vote counts for each book --/
structure VoteCounts where
  gameOfThrones : ℕ
  twilight : ℕ
  artOfTheDeal : ℕ

/-- Calculates the altered vote counts after tampering --/
def alteredVotes (original : VoteCounts) : VoteCounts :=
  { gameOfThrones := original.gameOfThrones,
    twilight := original.twilight / 2,
    artOfTheDeal := original.artOfTheDeal / 5 }

/-- Calculates the total number of altered votes --/
def totalAlteredVotes (altered : VoteCounts) : ℕ :=
  altered.gameOfThrones + altered.twilight + altered.artOfTheDeal

/-- Theorem: The percentage of altered votes for Game of Thrones is 50% --/
theorem game_of_thrones_percentage (original : VoteCounts)
  (h1 : original.gameOfThrones = 10)
  (h2 : original.twilight = 12)
  (h3 : original.artOfTheDeal = 20) :
  (alteredVotes original).gameOfThrones * 100 / (totalAlteredVotes (alteredVotes original)) = 50 := by
  sorry


end NUMINAMATH_CALUDE_game_of_thrones_percentage_l1763_176386


namespace NUMINAMATH_CALUDE_semicircular_plot_radius_l1763_176379

/-- The radius of a semicircular plot given the total fence length and opening length. -/
theorem semicircular_plot_radius 
  (total_fence_length : ℝ) 
  (opening_length : ℝ) 
  (h1 : total_fence_length = 33) 
  (h2 : opening_length = 3) : 
  ∃ (radius : ℝ), radius = (total_fence_length - opening_length) / (Real.pi + 2) :=
sorry

end NUMINAMATH_CALUDE_semicircular_plot_radius_l1763_176379


namespace NUMINAMATH_CALUDE_number_transformation_l1763_176396

theorem number_transformation (initial_number : ℕ) : 
  initial_number = 6 → 3 * ((2 * initial_number) + 9) = 63 := by
  sorry

end NUMINAMATH_CALUDE_number_transformation_l1763_176396


namespace NUMINAMATH_CALUDE_marks_difference_l1763_176380

/-- Given the marks of five students a, b, c, d, and e, prove that e has 3 more marks than d -/
theorem marks_difference (a b c d e : ℕ) : 
  (a + b + c) / 3 = 48 →
  (a + b + c + d) / 4 = 47 →
  e > d →
  (b + c + d + e) / 4 = 48 →
  a = 43 →
  e - d = 3 := by
sorry

end NUMINAMATH_CALUDE_marks_difference_l1763_176380


namespace NUMINAMATH_CALUDE_johns_total_time_l1763_176317

theorem johns_total_time (exploring_time writing_book_time : ℝ) : 
  exploring_time = 3 →
  writing_book_time = 0.5 →
  exploring_time + (exploring_time / 2) + writing_book_time = 5 := by
  sorry

end NUMINAMATH_CALUDE_johns_total_time_l1763_176317


namespace NUMINAMATH_CALUDE_problem_1_l1763_176381

theorem problem_1 : Real.sqrt 32 - 3 * Real.sqrt (1/2) + Real.sqrt 2 = (7/2) * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l1763_176381


namespace NUMINAMATH_CALUDE_equation_solutions_l1763_176362

theorem equation_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = -2 ∧ x₂ = 1 ∧ 
    (∀ x : ℝ, x * (x + 2) = (x + 2) ↔ x = x₁ ∨ x = x₂)) ∧
  (∃ y₁ y₂ : ℝ, y₁ = (3 - Real.sqrt 7) / 2 ∧ y₂ = (3 + Real.sqrt 7) / 2 ∧ 
    (∀ x : ℝ, 2 * x^2 - 6 * x + 1 = 0 ↔ x = y₁ ∨ x = y₂)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1763_176362


namespace NUMINAMATH_CALUDE_nadine_chairs_purchase_l1763_176347

/-- Proves that Nadine bought 2 chairs given the conditions of her purchases -/
theorem nadine_chairs_purchase :
  ∀ (total_spent table_cost chair_cost : ℕ),
    total_spent = 56 →
    table_cost = 34 →
    chair_cost = 11 →
    ∃ (num_chairs : ℕ),
      num_chairs * chair_cost = total_spent - table_cost ∧
      num_chairs = 2 := by
  sorry

end NUMINAMATH_CALUDE_nadine_chairs_purchase_l1763_176347


namespace NUMINAMATH_CALUDE_complex_circle_equation_l1763_176319

open Complex

theorem complex_circle_equation (z : ℂ) (x y : ℝ) :
  z = x + y * I →
  abs (z - 2) = 1 →
  (x - 2)^2 + y^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_complex_circle_equation_l1763_176319


namespace NUMINAMATH_CALUDE_P_zero_for_floor_l1763_176344

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- The polynomial P(x,y) -/
def P (x y : ℤ) : ℤ :=
  (y - 2*x) * (y - 2*x - 1)

/-- Theorem stating that P(⌊a⌋, ⌊2a⌋) = 0 for all real numbers a -/
theorem P_zero_for_floor (a : ℝ) : P (floor a) (floor (2*a)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_P_zero_for_floor_l1763_176344


namespace NUMINAMATH_CALUDE_speeding_motorists_percentage_l1763_176342

theorem speeding_motorists_percentage
  (total_motorists : ℝ)
  (h1 : total_motorists > 0)
  (ticketed_speeders : ℝ)
  (h2 : ticketed_speeders = 0.2 * total_motorists)
  (h3 : ticketed_speeders = 0.8 * (ticketed_speeders + (0.2 * (ticketed_speeders / 0.8))))
  : (ticketed_speeders / 0.8) / total_motorists = 0.25 := by
sorry

end NUMINAMATH_CALUDE_speeding_motorists_percentage_l1763_176342


namespace NUMINAMATH_CALUDE_daniel_initial_noodles_l1763_176314

/-- The number of noodles Daniel gave to William -/
def noodles_given : ℕ := 12

/-- The number of noodles Daniel has now -/
def noodles_remaining : ℕ := 54

/-- The initial number of noodles Daniel had -/
def initial_noodles : ℕ := noodles_given + noodles_remaining

theorem daniel_initial_noodles :
  initial_noodles = 66 :=
by sorry

end NUMINAMATH_CALUDE_daniel_initial_noodles_l1763_176314


namespace NUMINAMATH_CALUDE_recreation_percentage_is_twenty_percent_l1763_176325

/-- Calculates the percentage of earnings allocated for recreation and relaxation -/
def recreation_percentage (earnings_per_customer : ℚ) (fixed_expenses : ℚ) 
  (num_customers : ℕ) (savings : ℚ) : ℚ :=
  let total_earnings := earnings_per_customer * num_customers
  let total_expenses := fixed_expenses + savings
  let recreation_money := total_earnings - total_expenses
  (recreation_money / total_earnings) * 100

/-- Proves that the percentage of earnings allocated for recreation and relaxation is 20% -/
theorem recreation_percentage_is_twenty_percent :
  recreation_percentage 18 280 80 872 = 20 := by
  sorry

end NUMINAMATH_CALUDE_recreation_percentage_is_twenty_percent_l1763_176325


namespace NUMINAMATH_CALUDE_number_fraction_problem_l1763_176368

theorem number_fraction_problem (N : ℝ) : 
  (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * N = 16 → (1/3 : ℝ) * (2/5 : ℝ) * N = 64 := by
  sorry

end NUMINAMATH_CALUDE_number_fraction_problem_l1763_176368


namespace NUMINAMATH_CALUDE_acute_angle_m_range_l1763_176375

def a : ℝ × ℝ := (1, 2)
def b (m : ℝ) : ℝ × ℝ := (4, m)

def angle_is_acute (v w : ℝ × ℝ) : Prop :=
  0 < v.1 * w.1 + v.2 * w.2 ∧ 
  (v.1 * w.1 + v.2 * w.2)^2 < (v.1^2 + v.2^2) * (w.1^2 + w.2^2)

theorem acute_angle_m_range :
  ∀ m : ℝ, angle_is_acute a (b m) → m ∈ Set.Ioo (-2) 8 ∪ Set.Ioi 8 :=
by sorry

end NUMINAMATH_CALUDE_acute_angle_m_range_l1763_176375


namespace NUMINAMATH_CALUDE_prism_no_circular_section_l1763_176336

/-- A solid object that can be cut by a plane -/
class Solid :=
  (can_produce_circular_section : Bool)

/-- A cone is a solid that can produce a circular cross-section -/
def Cone : Solid :=
  { can_produce_circular_section := true }

/-- A cylinder is a solid that can produce a circular cross-section -/
def Cylinder : Solid :=
  { can_produce_circular_section := true }

/-- A sphere is a solid that can produce a circular cross-section -/
def Sphere : Solid :=
  { can_produce_circular_section := true }

/-- A prism is a solid that cannot produce a circular cross-section -/
def Prism : Solid :=
  { can_produce_circular_section := false }

/-- Theorem: Among cones, cylinders, spheres, and prisms, only a prism cannot produce a circular cross-section -/
theorem prism_no_circular_section :
  ∀ s : Solid, s.can_produce_circular_section = false → s = Prism :=
by sorry

end NUMINAMATH_CALUDE_prism_no_circular_section_l1763_176336


namespace NUMINAMATH_CALUDE_find_d_value_l1763_176354

theorem find_d_value (x y d : ℝ) : 
  7^(3*x - 1) * 3^(4*y - 3) = 49^x * d^y ∧ x + y = 4 → d = 27 := by
  sorry

end NUMINAMATH_CALUDE_find_d_value_l1763_176354


namespace NUMINAMATH_CALUDE_enrollment_increase_l1763_176358

theorem enrollment_increase (E : ℝ) (E_1992 : ℝ) (E_1993 : ℝ)
  (h1 : E_1993 = 1.26 * E)
  (h2 : E_1993 = 1.05 * E_1992) :
  (E_1992 - E) / E * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_enrollment_increase_l1763_176358


namespace NUMINAMATH_CALUDE_arithmetic_computation_l1763_176378

theorem arithmetic_computation : -9 * 5 - (-7 * -2) + (-11 * -4) = -15 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_computation_l1763_176378


namespace NUMINAMATH_CALUDE_fundraising_solution_correct_l1763_176334

/-- Represents the prices and quantities of basketballs and soccer balls -/
structure BallPurchase where
  basketball_price : ℕ
  soccer_price : ℕ
  basketball_qty : ℕ
  soccer_qty : ℕ

/-- Represents the fundraising conditions -/
structure FundraisingConditions where
  original_budget : ℕ
  original_total_items : ℕ
  actual_raised : ℕ
  new_total_items : ℕ

/-- Checks if a purchase satisfies the original plan -/
def satisfies_original_plan (purchase : BallPurchase) (conditions : FundraisingConditions) : Prop :=
  purchase.basketball_qty + purchase.soccer_qty = conditions.original_total_items ∧
  purchase.basketball_price * purchase.basketball_qty + purchase.soccer_price * purchase.soccer_qty = conditions.original_budget

/-- Checks if a purchase is valid under the new conditions -/
def is_valid_new_purchase (purchase : BallPurchase) (conditions : FundraisingConditions) : Prop :=
  purchase.basketball_qty + purchase.soccer_qty = conditions.new_total_items ∧
  purchase.basketball_price * purchase.basketball_qty + purchase.soccer_price * purchase.soccer_qty ≤ conditions.actual_raised

/-- Theorem stating the correctness of the solution -/
theorem fundraising_solution_correct 
  (purchase : BallPurchase) 
  (conditions : FundraisingConditions) 
  (h_basketball_price : purchase.basketball_price = 100)
  (h_soccer_price : purchase.soccer_price = 80)
  (h_original_budget : conditions.original_budget = 5600)
  (h_original_total_items : conditions.original_total_items = 60)
  (h_actual_raised : conditions.actual_raised = 6890)
  (h_new_total_items : conditions.new_total_items = 80) :
  (satisfies_original_plan purchase conditions ∧ purchase.basketball_qty = 40 ∧ purchase.soccer_qty = 20) ∧
  (∀ new_purchase : BallPurchase, is_valid_new_purchase new_purchase conditions → new_purchase.basketball_qty ≤ 24) :=
by sorry

end NUMINAMATH_CALUDE_fundraising_solution_correct_l1763_176334


namespace NUMINAMATH_CALUDE_cafe_visits_l1763_176366

/-- The number of people in the club -/
def n : ℕ := 9

/-- The number of people who visit the cafe each day -/
def k : ℕ := 3

/-- The number of days -/
def days : ℕ := 360

/-- The number of times each pair visits the cafe -/
def x : ℕ := 30

theorem cafe_visits :
  (n.choose 2) * x = days * (k.choose 2) := by sorry

end NUMINAMATH_CALUDE_cafe_visits_l1763_176366


namespace NUMINAMATH_CALUDE_garden_area_l1763_176387

/-- Represents a rectangular garden with specific properties -/
structure Garden where
  width : ℝ
  length : ℝ
  perimeter_minus_one_side : ℝ

/-- The properties of the garden as described in the problem -/
def garden_properties (g : Garden) : Prop :=
  g.perimeter_minus_one_side = 60 ∧
  g.length = 2 * g.width

/-- The theorem stating that a garden with the given properties has an area of 450 square meters -/
theorem garden_area (g : Garden) (h : garden_properties g) : g.width * g.length = 450 := by
  sorry


end NUMINAMATH_CALUDE_garden_area_l1763_176387


namespace NUMINAMATH_CALUDE_vacation_fund_adjustment_l1763_176350

/-- Calculates the required weekly hours to meet a financial goal after losing one week of work --/
def required_hours (original_weeks : ℕ) (original_hours_per_week : ℕ) (total_earnings : ℕ) : ℚ :=
  let remaining_weeks := original_weeks - 1
  let hourly_rate := (total_earnings : ℚ) / (original_weeks * original_hours_per_week)
  let weekly_earnings_needed := (total_earnings : ℚ) / remaining_weeks
  weekly_earnings_needed / hourly_rate

theorem vacation_fund_adjustment (original_weeks : ℕ) (original_hours_per_week : ℕ) (total_earnings : ℕ) 
    (h1 : original_weeks = 10)
    (h2 : original_hours_per_week = 25)
    (h3 : total_earnings = 2500) :
  ∃ (n : ℕ), n ≤ required_hours original_weeks original_hours_per_week total_earnings ∧ 
             required_hours original_weeks original_hours_per_week total_earnings < n + 1 ∧
             n = 28 :=
  sorry

end NUMINAMATH_CALUDE_vacation_fund_adjustment_l1763_176350


namespace NUMINAMATH_CALUDE_negation_of_existence_l1763_176376

theorem negation_of_existence (p : Prop) :
  (¬ ∃ (x : ℤ), x^2 ≥ x) ↔ (∀ (x : ℤ), x^2 < x) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_l1763_176376


namespace NUMINAMATH_CALUDE_sum_integers_negative20_to_10_l1763_176369

def sum_integers (a b : Int) : Int :=
  (b - a + 1) * (a + b) / 2

theorem sum_integers_negative20_to_10 :
  sum_integers (-20) 10 = -155 := by
  sorry

end NUMINAMATH_CALUDE_sum_integers_negative20_to_10_l1763_176369


namespace NUMINAMATH_CALUDE_bowtie_equation_solution_l1763_176367

-- Define the operation ⊙
noncomputable def bowtie (x y : ℝ) : ℝ := x + Real.sqrt (y + Real.sqrt (y + Real.sqrt (y + Real.sqrt y)))

-- State the theorem
theorem bowtie_equation_solution (h : ℝ) : 
  bowtie 8 h = 12 → h = 12 := by sorry

end NUMINAMATH_CALUDE_bowtie_equation_solution_l1763_176367


namespace NUMINAMATH_CALUDE_complement_of_B_in_U_l1763_176391

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5}

-- Define set A
def A : Set Nat := {2, 3, 5}

-- Define set B
def B : Set Nat := {2, 5}

-- Theorem statement
theorem complement_of_B_in_U :
  U \ B = {1, 3, 4} := by sorry

end NUMINAMATH_CALUDE_complement_of_B_in_U_l1763_176391


namespace NUMINAMATH_CALUDE_hyperbola_parabola_intersection_l1763_176390

/-- The value of p for which the left focus of the hyperbola 
    x²/3 - 16y²/p² = 1 (p > 0) lies on the latus rectum of 
    the parabola y² = 2px -/
theorem hyperbola_parabola_intersection (p : ℝ) : 
  p > 0 → 
  (∃ x y : ℝ, x^2/3 - 16*y^2/p^2 = 1) → 
  (∃ x y : ℝ, y^2 = 2*p*x) → 
  (∃ x : ℝ, x^2/3 - 16*0^2/p^2 = 1 ∧ 0^2 = 2*p*x) → 
  p = 4 := by sorry

end NUMINAMATH_CALUDE_hyperbola_parabola_intersection_l1763_176390


namespace NUMINAMATH_CALUDE_city_g_highest_growth_l1763_176321

structure City where
  name : String
  pop1990 : ℕ
  pop2000 : ℕ

def cities : List City := [
  ⟨"F", 50, 60⟩,
  ⟨"G", 60, 90⟩,
  ⟨"H", 70, 80⟩,
  ⟨"I", 100, 110⟩,
  ⟨"J", 150, 180⟩
]

def growthRate (c : City) : ℚ :=
  (c.pop2000 : ℚ) / (c.pop1990 : ℚ)

def adjustedGrowthRate (c : City) : ℚ :=
  if c.name = "H" then
    growthRate c * (11 / 10)
  else
    growthRate c

theorem city_g_highest_growth :
  ∀ c ∈ cities, c.name ≠ "G" →
    adjustedGrowthRate (cities[1]) ≥ adjustedGrowthRate c := by
  sorry

end NUMINAMATH_CALUDE_city_g_highest_growth_l1763_176321


namespace NUMINAMATH_CALUDE_cube_root_y_fourth_root_y_five_eq_four_l1763_176377

theorem cube_root_y_fourth_root_y_five_eq_four (y : ℝ) :
  (y * (y^5)^(1/4))^(1/3) = 4 → y = 2^(8/3) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_y_fourth_root_y_five_eq_four_l1763_176377


namespace NUMINAMATH_CALUDE_square_sum_geq_root3_product_l1763_176311

theorem square_sum_geq_root3_product (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) 
  (h_product_leq_sum : a * b * c ≤ a + b + c) : 
  a^2 + b^2 + c^2 ≥ Real.sqrt 3 * (a * b * c) := by
  sorry

end NUMINAMATH_CALUDE_square_sum_geq_root3_product_l1763_176311


namespace NUMINAMATH_CALUDE_starting_lineup_count_l1763_176371

def team_size : Nat := 16
def lineup_size : Nat := 5
def twin_count : Nat := 2
def triplet_count : Nat := 3

theorem starting_lineup_count : 
  (triplet_count * Nat.choose (team_size - twin_count - triplet_count + 2) (lineup_size - twin_count - 1)) = 198 := by
  sorry

end NUMINAMATH_CALUDE_starting_lineup_count_l1763_176371


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l1763_176303

theorem solution_set_of_inequality (x : ℝ) : 
  x^2 ≥ 2*x ↔ x ∈ Set.Iic 0 ∪ Set.Ici 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l1763_176303


namespace NUMINAMATH_CALUDE_probability_two_aces_standard_deck_l1763_176338

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (ace_count : ℕ)

/-- The probability of drawing two Aces as the top two cards from a randomly arranged deck -/
def probability_two_aces (d : Deck) : ℚ :=
  (d.ace_count : ℚ) / d.total_cards * (d.ace_count - 1) / (d.total_cards - 1)

/-- Theorem: The probability of drawing two Aces as the top two cards from a standard deck is 1/221 -/
theorem probability_two_aces_standard_deck :
  probability_two_aces ⟨52, 4⟩ = 1 / 221 := by
  sorry

#eval probability_two_aces ⟨52, 4⟩

end NUMINAMATH_CALUDE_probability_two_aces_standard_deck_l1763_176338


namespace NUMINAMATH_CALUDE_snooker_ticket_difference_l1763_176382

/-- Represents the ticket sales for a snooker tournament --/
structure TicketSales where
  vipPrice : ℕ
  regularPrice : ℕ
  totalTickets : ℕ
  totalRevenue : ℕ

/-- Calculates the difference between regular and VIP tickets sold --/
def ticketDifference (sales : TicketSales) : ℕ :=
  let vipTickets := (sales.totalRevenue - sales.regularPrice * sales.totalTickets) / 
                    (sales.vipPrice - sales.regularPrice)
  let regularTickets := sales.totalTickets - vipTickets
  regularTickets - vipTickets

/-- Theorem stating the difference in ticket sales --/
theorem snooker_ticket_difference :
  let sales : TicketSales := {
    vipPrice := 45,
    regularPrice := 20,
    totalTickets := 320,
    totalRevenue := 7500
  }
  ticketDifference sales = 232 := by
  sorry


end NUMINAMATH_CALUDE_snooker_ticket_difference_l1763_176382


namespace NUMINAMATH_CALUDE_michelle_needs_three_more_racks_l1763_176394

/-- The number of additional drying racks Michelle needs -/
def additional_racks_needed : ℕ :=
  let total_flour : ℕ := 6 * 12 -- 6 bags * 12 cups per bag
  let flour_per_type : ℕ := total_flour / 2 -- equal amounts for both types
  let pasta_type1 : ℕ := flour_per_type / 3 -- 3 cups per pound for type 1
  let pasta_type2 : ℕ := flour_per_type / 4 -- 4 cups per pound for type 2
  let total_pasta : ℕ := pasta_type1 + pasta_type2
  let total_racks_needed : ℕ := (total_pasta + 4) / 5 -- Ceiling division by 5
  total_racks_needed - 2 -- Subtract the 2 racks she already owns

theorem michelle_needs_three_more_racks :
  additional_racks_needed = 3 := by
  sorry

end NUMINAMATH_CALUDE_michelle_needs_three_more_racks_l1763_176394


namespace NUMINAMATH_CALUDE_floor_of_negative_three_point_seven_l1763_176301

theorem floor_of_negative_three_point_seven :
  ⌊(-3.7 : ℝ)⌋ = -4 := by sorry

end NUMINAMATH_CALUDE_floor_of_negative_three_point_seven_l1763_176301


namespace NUMINAMATH_CALUDE_jimmy_stairs_l1763_176361

/-- The sum of an arithmetic sequence -/
def arithmetic_sum (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

/-- Jimmy's stair climbing problem -/
theorem jimmy_stairs : arithmetic_sum 30 10 8 = 520 := by
  sorry

end NUMINAMATH_CALUDE_jimmy_stairs_l1763_176361


namespace NUMINAMATH_CALUDE_min_value_complex_condition_l1763_176348

theorem min_value_complex_condition (x y : ℝ) :
  Complex.abs (Complex.mk x y - Complex.I * 4) = Complex.abs (Complex.mk x y + 2) →
  2^x + 4^y ≥ 4 * Real.sqrt 2 ∧ 
  ∃ (x₀ y₀ : ℝ), Complex.abs (Complex.mk x₀ y₀ - Complex.I * 4) = Complex.abs (Complex.mk x₀ y₀ + 2) ∧
                 2^x₀ + 4^y₀ = 4 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_complex_condition_l1763_176348


namespace NUMINAMATH_CALUDE_cricket_score_theorem_l1763_176324

/-- Represents the score of a cricket match -/
def CricketScore := ℕ

/-- Calculates the total runs from boundaries -/
def boundaryRuns (boundaries : ℕ) : ℕ := 4 * boundaries

/-- Calculates the total runs from sixes -/
def sixRuns (sixes : ℕ) : ℕ := 6 * sixes

/-- Theorem: Given the conditions of the cricket match, prove the total score is 142 runs -/
theorem cricket_score_theorem (boundaries sixes : ℕ) 
  (h1 : boundaries = 12)
  (h2 : sixes = 2)
  (h3 : (57.74647887323944 : ℚ) / 100 * 142 = 142 - boundaryRuns boundaries - sixRuns sixes) :
  142 = boundaryRuns boundaries + sixRuns sixes + 
    ((57.74647887323944 : ℚ) / 100 * 142).floor :=
by sorry

end NUMINAMATH_CALUDE_cricket_score_theorem_l1763_176324


namespace NUMINAMATH_CALUDE_reachable_cells_after_ten_moves_l1763_176356

-- Define the board size
def boardSize : ℕ := 21

-- Define the number of moves
def numMoves : ℕ := 10

-- Define a function to calculate the number of reachable cells
def reachableCells (moves : ℕ) : ℕ :=
  if moves % 2 = 0 then
    1 + 2 * moves * (moves + 1)
  else
    (moves + 1) ^ 2

-- Theorem statement
theorem reachable_cells_after_ten_moves :
  reachableCells numMoves = 121 := by
  sorry

end NUMINAMATH_CALUDE_reachable_cells_after_ten_moves_l1763_176356


namespace NUMINAMATH_CALUDE_double_acute_angle_l1763_176343

theorem double_acute_angle (θ : Real) (h : 0 < θ ∧ θ < Real.pi / 2) :
  0 < 2 * θ ∧ 2 * θ < Real.pi := by
  sorry

end NUMINAMATH_CALUDE_double_acute_angle_l1763_176343


namespace NUMINAMATH_CALUDE_scooter_cost_recovery_l1763_176346

/-- The minimum number of deliveries required to recover the initial cost of a scooter -/
def min_deliveries (initial_cost earnings fuel_cost parking_fee : ℕ) : ℕ :=
  (initial_cost + (earnings - fuel_cost - parking_fee) - 1) / (earnings - fuel_cost - parking_fee)

/-- Theorem stating the minimum number of deliveries required to recover the scooter cost -/
theorem scooter_cost_recovery :
  min_deliveries 3000 12 4 1 = 429 := by
  sorry

end NUMINAMATH_CALUDE_scooter_cost_recovery_l1763_176346


namespace NUMINAMATH_CALUDE_tree_branches_count_l1763_176345

/-- Proves that a tree with the given characteristics has 30 branches -/
theorem tree_branches_count : 
  ∀ (total_leaves : ℕ) (twigs_per_branch : ℕ) 
    (four_leaf_twig_percent : ℚ) (five_leaf_twig_percent : ℚ),
  total_leaves = 12690 →
  twigs_per_branch = 90 →
  four_leaf_twig_percent = 30 / 100 →
  five_leaf_twig_percent = 70 / 100 →
  four_leaf_twig_percent + five_leaf_twig_percent = 1 →
  ∃ (branches : ℕ),
    branches * (four_leaf_twig_percent * twigs_per_branch * 4 + 
                five_leaf_twig_percent * twigs_per_branch * 5) = total_leaves ∧
    branches = 30 := by
  sorry

end NUMINAMATH_CALUDE_tree_branches_count_l1763_176345


namespace NUMINAMATH_CALUDE_sqrt_difference_sum_l1763_176372

theorem sqrt_difference_sum (x : ℝ) : 
  Real.sqrt (64 - x^2) - Real.sqrt (36 - x^2) = 4 →
  Real.sqrt (64 - x^2) + Real.sqrt (36 - x^2) = 7 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_sum_l1763_176372


namespace NUMINAMATH_CALUDE_absolute_value_equation_solutions_l1763_176328

theorem absolute_value_equation_solutions :
  ∀ x : ℚ, (|2 * x - 3| = x + 1) ↔ (x = 4 ∨ x = 2/3) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solutions_l1763_176328


namespace NUMINAMATH_CALUDE_max_value_quadratic_l1763_176329

theorem max_value_quadratic (f : ℝ → ℝ) (h : ∀ x, f x = x^2 + 2*x + 1) :
  ∃ y ∈ Set.Icc (-2 : ℝ) 2, ∀ x ∈ Set.Icc (-2 : ℝ) 2, f x ≤ f y ∧ f y = 9 :=
sorry

end NUMINAMATH_CALUDE_max_value_quadratic_l1763_176329


namespace NUMINAMATH_CALUDE_books_checked_out_after_returns_l1763_176349

-- Define the initial state
def initial_books : ℕ := 15
def initial_movies : ℕ := 6

-- Define the number of books returned
def books_returned : ℕ := 8

-- Define the fraction of movies returned
def movie_return_fraction : ℚ := 1 / 3

-- Define the final total of items
def final_total : ℕ := 20

-- Theorem to prove
theorem books_checked_out_after_returns (checked_out : ℕ) : 
  checked_out = 9 :=
by
  sorry


end NUMINAMATH_CALUDE_books_checked_out_after_returns_l1763_176349


namespace NUMINAMATH_CALUDE_farm_ratio_l1763_176323

def cows : ℕ := 21
def horses : ℕ := 6

theorem farm_ratio : (cows / horses : ℚ) = 7 / 2 := by
  sorry

end NUMINAMATH_CALUDE_farm_ratio_l1763_176323


namespace NUMINAMATH_CALUDE_expression_value_at_four_l1763_176339

theorem expression_value_at_four :
  let x : ℚ := 4
  (x^2 - 3*x - 10) / (x - 5) = 6 := by sorry

end NUMINAMATH_CALUDE_expression_value_at_four_l1763_176339


namespace NUMINAMATH_CALUDE_basketball_evaluation_theorem_l1763_176398

/-- The number of rounds in the basketball evaluation. -/
def num_rounds : ℕ := 3

/-- The number of shots per round. -/
def shots_per_round : ℕ := 2

/-- The probability of player A making a shot. -/
def shot_probability : ℚ := 2/3

/-- The probability of passing a single round. -/
def pass_round_probability : ℚ := 1 - (1 - shot_probability) ^ shots_per_round

/-- The expected number of rounds player A will pass. -/
def expected_passed_rounds : ℚ := num_rounds * pass_round_probability

theorem basketball_evaluation_theorem :
  expected_passed_rounds = 8/3 :=
sorry

end NUMINAMATH_CALUDE_basketball_evaluation_theorem_l1763_176398


namespace NUMINAMATH_CALUDE_total_wheels_is_102_l1763_176363

/-- The number of wheels Dimitri saw at the park -/
def total_wheels : ℕ :=
  let bicycle_wheels := 2
  let tricycle_wheels := 3
  let unicycle_wheels := 1
  let scooter_wheels := 4
  let men_on_bicycles := 6
  let women_on_bicycles := 5
  let boys_on_tricycles := 8
  let girls_on_tricycles := 7
  let boys_on_unicycles := 2
  let girls_on_unicycles := 1
  let boys_on_scooters := 5
  let girls_on_scooters := 3
  (men_on_bicycles + women_on_bicycles) * bicycle_wheels +
  (boys_on_tricycles + girls_on_tricycles) * tricycle_wheels +
  (boys_on_unicycles + girls_on_unicycles) * unicycle_wheels +
  (boys_on_scooters + girls_on_scooters) * scooter_wheels

theorem total_wheels_is_102 : total_wheels = 102 := by
  sorry

end NUMINAMATH_CALUDE_total_wheels_is_102_l1763_176363


namespace NUMINAMATH_CALUDE_rain_probability_l1763_176397

/-- The probability of rain on Friday -/
def prob_friday : ℝ := 0.4

/-- The probability of rain on Saturday -/
def prob_saturday : ℝ := 0.5

/-- The probability of rain on Sunday, given it didn't rain on both Friday and Saturday -/
def prob_sunday_normal : ℝ := 0.3

/-- The probability of rain on Sunday, given it rained on both Friday and Saturday -/
def prob_sunday_conditional : ℝ := 0.6

/-- The probability of rain on all three days -/
def prob_all_days : ℝ := prob_friday * prob_saturday * prob_sunday_conditional

theorem rain_probability :
  prob_all_days = 0.12 :=
sorry

end NUMINAMATH_CALUDE_rain_probability_l1763_176397


namespace NUMINAMATH_CALUDE_x_minus_y_value_l1763_176370

theorem x_minus_y_value (x y : ℚ) 
  (eq1 : 3 * x - 4 * y = 17) 
  (eq2 : x + 3 * y = 1) : 
  x - y = 69 / 13 := by
sorry

end NUMINAMATH_CALUDE_x_minus_y_value_l1763_176370


namespace NUMINAMATH_CALUDE_polynomial_rewrite_l1763_176320

variable (x y : ℝ)

def original_polynomial := x^3 - 3*x^2*y + 3*x*y^2 - y^3

theorem polynomial_rewrite :
  ((x^3 - y^3) - (3*x^2*y - 3*x*y^2) = original_polynomial x y) ∧
  ((x^3 + 3*x*y^2) - (3*x^2*y + y^3) = original_polynomial x y) ∧
  ((3*x*y^2 - 3*x^2*y) - (y^3 - x^3) = original_polynomial x y) ∧
  ¬((x^3 - 3*x^2*y) - (3*x*y^2 + y^3) = original_polynomial x y) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_rewrite_l1763_176320


namespace NUMINAMATH_CALUDE_hyperbola_properties_l1763_176332

-- Define the hyperbolas
def C₁ (x y : ℝ) : Prop := x^2/4 - y^2/3 = 1
def C₂ (x y : ℝ) : Prop := x^2/4 - y^2/3 = -1

-- Define focal length
def focal_length (C : (ℝ → ℝ → Prop)) : ℝ := sorry

-- Define foci
def foci (C : (ℝ → ℝ → Prop)) : Set (ℝ × ℝ) := sorry

-- Define asymptotic lines
def asymptotic_lines (C : (ℝ → ℝ → Prop)) : Set (ℝ → ℝ → Prop) := sorry

-- Define eccentricity
def eccentricity (C : (ℝ → ℝ → Prop)) : ℝ := sorry

theorem hyperbola_properties :
  (focal_length C₁ = focal_length C₂) ∧
  (∃ (r : ℝ), ∀ (p : ℝ × ℝ), p ∈ foci C₁ ∪ foci C₂ → p.1^2 + p.2^2 = r^2) ∧
  (asymptotic_lines C₁ = asymptotic_lines C₂) ∧
  (eccentricity C₁ ≠ eccentricity C₂) := by sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l1763_176332


namespace NUMINAMATH_CALUDE_line_plane_intersection_l1763_176326

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the intersection operation for planes
variable (intersect : Plane → Plane → Set Line)

-- Define the subset relation for lines and planes
variable (subset : Line → Plane → Prop)

-- Define parallel relation for lines
variable (parallel : Line → Line → Prop)

-- Define intersection relation for lines
variable (intersects : Line → Line → Prop)

-- Theorem statement
theorem line_plane_intersection 
  (m n : Line) (α β : Plane) :
  (intersect α β = {m} ∧ subset n α) →
  (parallel m n ∨ intersects m n) :=
sorry

end NUMINAMATH_CALUDE_line_plane_intersection_l1763_176326


namespace NUMINAMATH_CALUDE_no_four_distinct_squares_sum_to_100_l1763_176335

theorem no_four_distinct_squares_sum_to_100 : 
  ¬ ∃ (a b c d : ℕ), 
    (0 < a) ∧ (a < b) ∧ (b < c) ∧ (c < d) ∧ 
    (a^2 + b^2 + c^2 + d^2 = 100) :=
sorry

end NUMINAMATH_CALUDE_no_four_distinct_squares_sum_to_100_l1763_176335


namespace NUMINAMATH_CALUDE_problem_pyramid_volume_l1763_176389

/-- Represents a truncated triangular pyramid -/
structure TruncatedPyramid where
  height : ℝ
  base1_sides : Fin 3 → ℝ
  base2_perimeter : ℝ

/-- Calculates the volume of a truncated triangular pyramid -/
def volume (p : TruncatedPyramid) : ℝ :=
  sorry

/-- The specific truncated pyramid from the problem -/
def problem_pyramid : TruncatedPyramid :=
  { height := 10
  , base1_sides := ![27, 29, 52]
  , base2_perimeter := 72 }

/-- Theorem stating that the volume of the problem pyramid is 1900 -/
theorem problem_pyramid_volume :
  volume problem_pyramid = 1900 := by sorry

end NUMINAMATH_CALUDE_problem_pyramid_volume_l1763_176389


namespace NUMINAMATH_CALUDE_cot_thirteen_pi_fourths_l1763_176353

theorem cot_thirteen_pi_fourths : Real.cos (13 * π / 4) / Real.sin (13 * π / 4) = -1 := by
  sorry

end NUMINAMATH_CALUDE_cot_thirteen_pi_fourths_l1763_176353


namespace NUMINAMATH_CALUDE_exists_m_in_range_l1763_176383

def sequence_x : ℕ → ℚ
  | 0 => 7
  | n + 1 => (sequence_x n ^ 2 + 7 * sequence_x n + 12) / (sequence_x n + 8)

theorem exists_m_in_range :
  ∃ m : ℕ, 81 ≤ m ∧ m ≤ 242 ∧
  sequence_x m ≤ 5 + 1 / 2^15 ∧
  ∀ k : ℕ, 0 < k ∧ k < m → sequence_x k > 5 + 1 / 2^15 := by
  sorry

end NUMINAMATH_CALUDE_exists_m_in_range_l1763_176383


namespace NUMINAMATH_CALUDE_sequence_property_l1763_176364

theorem sequence_property (a : ℕ → ℕ) (p : ℕ) : 
  (∀ (m n : ℕ), m ≥ n → a (m + n) + a (m - n) + 2 * m - 2 * n - 1 = (a (2 * m) + a (2 * n)) / 2) →
  a 1 = 0 →
  a p = 2019 * 2019 →
  p = 2020 := by
  sorry

end NUMINAMATH_CALUDE_sequence_property_l1763_176364


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_A_subset_C_iff_l1763_176341

-- Define the sets A, B, and C
def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}
def C (a : ℝ) : Set ℝ := {x | x < a}

-- Statement 1: Prove that (ℝ\A) ∩ B = {x | 2 < x < 3 or 7 ≤ x < 10}
theorem complement_A_intersect_B :
  (Set.univ \ A) ∩ B = {x | (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10)} := by sorry

-- Statement 2: Prove that A ⊆ C(a) if and only if a ≥ 7
theorem A_subset_C_iff (a : ℝ) :
  A ⊆ C a ↔ a ≥ 7 := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_A_subset_C_iff_l1763_176341


namespace NUMINAMATH_CALUDE_union_M_N_equals_interval_l1763_176302

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | x ≥ 0}
def N : Set ℝ := {x : ℝ | (x + 1) * (x - 3) < 0}

-- Define the interval (-1, +∞)
def openIntervalFromNegativeOneToInfinity : Set ℝ := {x : ℝ | x > -1}

-- State the theorem
theorem union_M_N_equals_interval : M ∪ N = openIntervalFromNegativeOneToInfinity := by
  sorry

end NUMINAMATH_CALUDE_union_M_N_equals_interval_l1763_176302


namespace NUMINAMATH_CALUDE_reciprocal_sum_l1763_176352

theorem reciprocal_sum (x y : ℝ) (h1 : x + y = 15) (h2 : x * y = 56) :
  1 / x + 1 / y = 15 / 56 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_sum_l1763_176352


namespace NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l1763_176388

theorem equilateral_triangle_perimeter (s : ℝ) (h : s > 0) : 
  (s^2 * Real.sqrt 3 / 4 = 2 * s) → (3 * s = 8 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l1763_176388


namespace NUMINAMATH_CALUDE_prism_36_edges_14_faces_l1763_176385

/-- A prism is a polyhedron with two congruent parallel faces (bases) and lateral faces that are parallelograms. -/
structure Prism where
  edges : ℕ

/-- The number of faces in a prism given its number of edges. -/
def num_faces (p : Prism) : ℕ :=
  2 + p.edges / 3

/-- Theorem: A prism with 36 edges has 14 faces. -/
theorem prism_36_edges_14_faces (p : Prism) (h : p.edges = 36) : num_faces p = 14 := by
  sorry


end NUMINAMATH_CALUDE_prism_36_edges_14_faces_l1763_176385


namespace NUMINAMATH_CALUDE_gas_volume_at_10_degrees_l1763_176327

-- Define the relationship between temperature change and volume change
def volume_change (temp_change : ℤ) : ℤ := (3 * temp_change) / 5

-- Define the initial conditions
def initial_temp : ℤ := 25
def initial_volume : ℤ := 40
def final_temp : ℤ := 10

-- Define the theorem
theorem gas_volume_at_10_degrees : 
  initial_volume + volume_change (final_temp - initial_temp) = 31 := by
  sorry

end NUMINAMATH_CALUDE_gas_volume_at_10_degrees_l1763_176327


namespace NUMINAMATH_CALUDE_positive_intervals_l1763_176392

-- Define the expression
def f (x : ℝ) : ℝ := (x - 2) * (x + 3)

-- Theorem statement
theorem positive_intervals (x : ℝ) : 
  f x > 0 ↔ x < -3 ∨ x > 2 := by
  sorry

end NUMINAMATH_CALUDE_positive_intervals_l1763_176392


namespace NUMINAMATH_CALUDE_final_sum_theorem_l1763_176355

/-- The number of participants in the game --/
def participants : ℕ := 53

/-- The initial value of the first calculator --/
def calc1_initial : ℤ := 2

/-- The initial value of the second calculator --/
def calc2_initial : ℤ := -2

/-- The initial value of the third calculator --/
def calc3_initial : ℕ := 5

/-- The operation applied to the first calculator --/
def op1 (n : ℤ) : ℤ := n ^ 2

/-- The operation applied to the second calculator --/
def op2 (n : ℤ) : ℤ := n ^ 3

/-- The operation applied to the third calculator --/
def op3 (n : ℕ) : ℕ := n + 2

/-- The final value of the first calculator after all participants --/
def calc1_final : ℤ := calc1_initial ^ (2 ^ participants)

/-- The final value of the second calculator after all participants --/
def calc2_final : ℤ := calc2_initial ^ (3 ^ participants)

/-- The final value of the third calculator after all participants --/
def calc3_final : ℕ := calc3_initial + 2 * participants

/-- The theorem stating the final sum of all calculators --/
theorem final_sum_theorem : 
  calc1_final + calc2_final + calc3_final = 
  calc1_initial ^ (2 ^ participants) + calc2_initial ^ (3 ^ participants) + (calc3_initial + 2 * participants) := by
  sorry

end NUMINAMATH_CALUDE_final_sum_theorem_l1763_176355


namespace NUMINAMATH_CALUDE_seths_ice_cream_purchase_l1763_176384

/-- Seth's ice cream purchase problem -/
theorem seths_ice_cream_purchase
  (ice_cream_cost : ℕ → ℕ)
  (yogurt_cost : ℕ)
  (yogurt_quantity : ℕ)
  (cost_difference : ℕ)
  (h1 : yogurt_quantity = 2)
  (h2 : ∀ n, ice_cream_cost n = 6 * n)
  (h3 : yogurt_cost = 1)
  (h4 : ∃ x : ℕ, ice_cream_cost x = yogurt_quantity * yogurt_cost + cost_difference)
  (h5 : cost_difference = 118) :
  ∃ x : ℕ, ice_cream_cost x = yogurt_quantity * yogurt_cost + cost_difference ∧ x = 20 := by
  sorry

end NUMINAMATH_CALUDE_seths_ice_cream_purchase_l1763_176384


namespace NUMINAMATH_CALUDE_expression_value_l1763_176304

theorem expression_value (a : ℝ) (h1 : a - 1 ≥ 0) (h2 : 1 - a ≥ 0) :
  a + 2 * Real.sqrt (a - 1) - Real.sqrt (1 - a) + 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1763_176304


namespace NUMINAMATH_CALUDE_stormi_bicycle_savings_l1763_176351

/-- The amount of additional money Stormi needs to afford a bicycle -/
def additional_money_needed (num_cars : ℕ) (price_per_car : ℕ) (num_lawns : ℕ) (price_per_lawn : ℕ) (bicycle_cost : ℕ) : ℕ :=
  bicycle_cost - (num_cars * price_per_car + num_lawns * price_per_lawn)

/-- Theorem stating that Stormi needs $24 more to afford the bicycle -/
theorem stormi_bicycle_savings : additional_money_needed 3 10 2 13 80 = 24 := by
  sorry

end NUMINAMATH_CALUDE_stormi_bicycle_savings_l1763_176351


namespace NUMINAMATH_CALUDE_permutations_congruence_l1763_176333

/-- The number of ways to arrange n elements, choosing k of them -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of valid permutations of "AAAABBBBCCCCDDDD" -/
def N : ℕ :=
  (choose 5 0 * choose 4 4 * choose 3 3 * choose 4 0) +
  (choose 5 1 * choose 4 3 * choose 3 2 * choose 4 1) +
  (choose 5 2 * choose 4 2 * choose 3 1 * choose 4 2) +
  (choose 5 3 * choose 4 1 * choose 3 0 * choose 4 3)

theorem permutations_congruence :
  N ≡ 581 [MOD 1000] := by sorry

end NUMINAMATH_CALUDE_permutations_congruence_l1763_176333


namespace NUMINAMATH_CALUDE_apps_difference_is_three_l1763_176315

/-- The difference between apps added and deleted -/
def appsDifference (initial final added : ℕ) : ℕ :=
  added - (initial + added - final)

/-- Proof that the difference between apps added and deleted is 3 -/
theorem apps_difference_is_three : appsDifference 21 24 89 = 3 := by
  sorry

end NUMINAMATH_CALUDE_apps_difference_is_three_l1763_176315


namespace NUMINAMATH_CALUDE_cube_number_placement_impossible_l1763_176393

/-- Represents a cube with 8 vertices -/
structure Cube :=
  (vertices : Fin 8 → ℕ)

/-- Predicate to check if two vertices are adjacent on a cube -/
def adjacent (i j : Fin 8) : Prop := sorry

/-- The theorem stating the impossibility of the number placement on a cube -/
theorem cube_number_placement_impossible :
  ¬ ∃ (c : Cube),
    (∀ i : Fin 8, 1 ≤ c.vertices i ∧ c.vertices i ≤ 220) ∧
    (∀ i j : Fin 8, i ≠ j → c.vertices i ≠ c.vertices j) ∧
    (∀ i j : Fin 8, adjacent i j → ∃ (d : ℕ), d > 1 ∧ d ∣ c.vertices i ∧ d ∣ c.vertices j) ∧
    (∀ i j : Fin 8, ¬adjacent i j → ∀ (d : ℕ), d > 1 → ¬(d ∣ c.vertices i ∧ d ∣ c.vertices j)) :=
sorry

end NUMINAMATH_CALUDE_cube_number_placement_impossible_l1763_176393


namespace NUMINAMATH_CALUDE_seven_awards_four_students_l1763_176340

/-- The number of ways to distribute n different awards among k students,
    where each student receives at least one award. -/
def distribute_awards (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem stating that distributing 7 awards among 4 students results in 920 ways -/
theorem seven_awards_four_students :
  distribute_awards 7 4 = 920 := by sorry

end NUMINAMATH_CALUDE_seven_awards_four_students_l1763_176340


namespace NUMINAMATH_CALUDE_largest_angle_in_triangle_l1763_176365

theorem largest_angle_in_triangle (x y z : ℝ) : 
  x = 30 ∧ y = 45 ∧ x + y + z = 180 → max x (max y z) = 105 :=
by sorry

end NUMINAMATH_CALUDE_largest_angle_in_triangle_l1763_176365


namespace NUMINAMATH_CALUDE_increasing_quadratic_function_condition_l1763_176359

/-- A function f(x) = x^2 - 2ax is increasing on [1, +∞) if and only if a ≤ 1 -/
theorem increasing_quadratic_function_condition (a : ℝ) : 
  (∀ x ≥ 1, Monotone (fun x => x^2 - 2*a*x)) ↔ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_increasing_quadratic_function_condition_l1763_176359


namespace NUMINAMATH_CALUDE_find_t_l1763_176395

def A (t : ℝ) : Set ℝ := {-4, t^2}
def B (t : ℝ) : Set ℝ := {t-5, 9, 1-t}

theorem find_t : ∀ t : ℝ, 9 ∈ A t ∩ B t → t = -3 := by sorry

end NUMINAMATH_CALUDE_find_t_l1763_176395


namespace NUMINAMATH_CALUDE_cryptarithm_no_solution_l1763_176307

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents a mapping from characters to digits -/
def DigitAssignment := Char → Digit

/-- Checks if all characters in a string are mapped to unique digits -/
def all_unique (s : String) (assignment : DigitAssignment) : Prop :=
  ∀ c₁ c₂, c₁ ∈ s.data → c₂ ∈ s.data → c₁ ≠ c₂ → assignment c₁ ≠ assignment c₂

/-- Converts a string to a number using the given digit assignment -/
def to_number (s : String) (assignment : DigitAssignment) : ℕ :=
  s.foldl (fun acc c => 10 * acc + (assignment c).val) 0

/-- The main theorem stating that the cryptarithm has no solution -/
theorem cryptarithm_no_solution :
  ¬ ∃ (assignment : DigitAssignment),
    all_unique "DONAKLENVG" assignment ∧
    to_number "DON" assignment + to_number "OKA" assignment +
    to_number "LENA" assignment + to_number "VOLGA" assignment =
    to_number "ANGARA" assignment :=
by sorry


end NUMINAMATH_CALUDE_cryptarithm_no_solution_l1763_176307


namespace NUMINAMATH_CALUDE_original_price_is_10000_l1763_176313

/-- Calculates the original price of a machine given repair cost, transportation cost, profit percentage, and selling price. -/
def calculate_original_price (repair_cost : ℕ) (transport_cost : ℕ) (profit_percent : ℕ) (selling_price : ℕ) : ℕ :=
  let total_additional_cost := repair_cost + transport_cost
  let total_cost_multiplier := 100 + profit_percent
  ((selling_price * 100) / total_cost_multiplier) - total_additional_cost

/-- Theorem stating that given the specific conditions, the original price of the machine was 10000. -/
theorem original_price_is_10000 :
  calculate_original_price 5000 1000 50 24000 = 10000 := by
  sorry

end NUMINAMATH_CALUDE_original_price_is_10000_l1763_176313


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l1763_176316

theorem cube_root_equation_solution :
  ∃ y : ℝ, y = 1/32 ∧ (5 - 1/y)^(1/3 : ℝ) = -3 :=
sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l1763_176316

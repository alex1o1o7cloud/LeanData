import Mathlib

namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l250_25051

theorem pure_imaginary_condition (a b : ℝ) : 
  (∀ x y : ℝ, Complex.mk x y = Complex.I * y → x = 0) ∧ 
  (∃ x y : ℝ, x = 0 ∧ Complex.mk x y ≠ Complex.I * y) :=
sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l250_25051


namespace NUMINAMATH_CALUDE_circle_line_distance_l250_25097

theorem circle_line_distance (x y : ℝ) (a : ℝ) :
  (x^2 + y^2 - 2*x - 4*y = 0) →
  ((1 - y + a) / Real.sqrt 2 = Real.sqrt 2 / 2 ∨
   (-1 + y - a) / Real.sqrt 2 = Real.sqrt 2 / 2) →
  (a = 0 ∨ a = 2) :=
sorry

end NUMINAMATH_CALUDE_circle_line_distance_l250_25097


namespace NUMINAMATH_CALUDE_volleyball_team_selection_l250_25092

def total_players : ℕ := 18
def quadruplets : ℕ := 4
def starters : ℕ := 7

theorem volleyball_team_selection :
  (Nat.choose total_players starters) - (Nat.choose (total_players - quadruplets) (starters - quadruplets)) = 31460 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_team_selection_l250_25092


namespace NUMINAMATH_CALUDE_cupboard_cost_price_l250_25077

theorem cupboard_cost_price (selling_price selling_price_increased : ℝ) 
  (h1 : selling_price = 0.84 * 3750)
  (h2 : selling_price_increased = 1.16 * 3750)
  (h3 : selling_price_increased = selling_price + 1200) : 
  ∃ (cost_price : ℝ), cost_price = 3750 := by
sorry

end NUMINAMATH_CALUDE_cupboard_cost_price_l250_25077


namespace NUMINAMATH_CALUDE_parallel_planes_properties_l250_25061

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (contains : Plane → Line → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the given condition
variable (a : Line) (α β : Plane)
variable (h : contains α a)

-- Theorem statement
theorem parallel_planes_properties :
  (∀ β, parallel_planes α β → parallel_line_plane a β) ∧
  (∀ β, ¬parallel_line_plane a β → ¬parallel_planes α β) ∧
  ¬(∀ β, parallel_line_plane a β → parallel_planes α β) :=
sorry

end NUMINAMATH_CALUDE_parallel_planes_properties_l250_25061


namespace NUMINAMATH_CALUDE_floor_length_approx_l250_25037

/-- Represents a rectangular floor with length and breadth -/
structure RectangularFloor where
  breadth : ℝ
  length : ℝ

/-- The properties of our specific rectangular floor -/
def floor_properties (floor : RectangularFloor) : Prop :=
  floor.length = 3 * floor.breadth ∧
  floor.length * floor.breadth = 60

/-- The theorem stating the length of the floor -/
theorem floor_length_approx (floor : RectangularFloor) 
  (h : floor_properties floor) : 
  ∃ ε > 0, abs (floor.length - 13.416) < ε :=
sorry

end NUMINAMATH_CALUDE_floor_length_approx_l250_25037


namespace NUMINAMATH_CALUDE_three_layer_runner_area_l250_25040

/-- Given three table runners with a combined area of 204 square inches covering 80% of a table
    with an area of 175 square inches, and an area of 24 square inches covered by exactly two
    layers of runner, prove that the area covered by three layers of runner is 20 square inches. -/
theorem three_layer_runner_area
  (total_runner_area : ℝ)
  (table_area : ℝ)
  (coverage_percent : ℝ)
  (two_layer_area : ℝ)
  (h1 : total_runner_area = 204)
  (h2 : table_area = 175)
  (h3 : coverage_percent = 0.8)
  (h4 : two_layer_area = 24)
  : ∃ (three_layer_area : ℝ),
    three_layer_area = 20 ∧
    coverage_percent * table_area = (total_runner_area - two_layer_area - three_layer_area) + 2 * two_layer_area + 3 * three_layer_area :=
by sorry

end NUMINAMATH_CALUDE_three_layer_runner_area_l250_25040


namespace NUMINAMATH_CALUDE_symmetric_function_minimum_value_l250_25041

-- Define the function f
def f (a b x : ℝ) : ℝ := (x - 1) * (x + 2) * (x^2 + a*x + b)

-- State the theorem
theorem symmetric_function_minimum_value (a b : ℝ) :
  (∀ x, f a b x = f a b (-x)) →  -- Symmetry condition
  (∃ x, ∀ y, f a b y ≥ f a b x ∧ f a b x = -9/4) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_function_minimum_value_l250_25041


namespace NUMINAMATH_CALUDE_hotel_profit_theorem_l250_25018

/-- Calculates the hotel's weekly profit given the operations expenses and service percentages --/
def hotel_profit (operations_expenses : ℚ) 
  (meetings_percent : ℚ) (events_percent : ℚ) (rooms_percent : ℚ)
  (meetings_tax : ℚ) (meetings_commission : ℚ)
  (events_tax : ℚ) (events_commission : ℚ)
  (rooms_tax : ℚ) (rooms_commission : ℚ) : ℚ :=
  let meetings_income := meetings_percent * operations_expenses
  let events_income := events_percent * operations_expenses
  let rooms_income := rooms_percent * operations_expenses
  let total_income := meetings_income + events_income + rooms_income
  let meetings_additional := meetings_income * (meetings_tax + meetings_commission)
  let events_additional := events_income * (events_tax + events_commission)
  let rooms_additional := rooms_income * (rooms_tax + rooms_commission)
  let total_additional := meetings_additional + events_additional + rooms_additional
  total_income - operations_expenses - total_additional

/-- The hotel's weekly profit is $1,283.75 given the specified conditions --/
theorem hotel_profit_theorem : 
  hotel_profit 5000 (5/8) (3/10) (11/20) (1/10) (1/20) (2/25) (3/50) (3/25) (3/100) = 1283.75 := by
  sorry


end NUMINAMATH_CALUDE_hotel_profit_theorem_l250_25018


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l250_25085

/-- A quadratic equation x^2 + mx + 9 has two distinct real roots if and only if m < -6 or m > 6 -/
theorem quadratic_two_distinct_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 9 = 0 ∧ y^2 + m*y + 9 = 0) ↔ m < -6 ∨ m > 6 :=
sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l250_25085


namespace NUMINAMATH_CALUDE_partition_contains_perfect_square_sum_l250_25014

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem partition_contains_perfect_square_sum (n : ℕ) : 
  (n ≥ 15) ↔ 
  (∀ (A B : Set ℕ), 
    (A ∪ B = Finset.range n.succ) → 
    (A ∩ B = ∅) → 
    ((∃ (x y : ℕ), x ∈ A ∧ y ∈ A ∧ x ≠ y ∧ is_perfect_square (x + y)) ∨
     (∃ (x y : ℕ), x ∈ B ∧ y ∈ B ∧ x ≠ y ∧ is_perfect_square (x + y)))) :=
by sorry

end NUMINAMATH_CALUDE_partition_contains_perfect_square_sum_l250_25014


namespace NUMINAMATH_CALUDE_kim_total_sweaters_l250_25067

/-- The number of sweaters Kim knit on each day of the week --/
structure SweaterCount where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ

/-- The conditions of Kim's sweater knitting for the week --/
def kim_sweater_conditions (sc : SweaterCount) : Prop :=
  sc.monday = 8 ∧
  sc.tuesday = sc.monday + 2 ∧
  sc.wednesday = sc.tuesday - 4 ∧
  sc.thursday = sc.wednesday ∧
  sc.friday = sc.monday / 2

/-- The theorem stating the total number of sweaters Kim knit that week --/
theorem kim_total_sweaters (sc : SweaterCount) 
  (h : kim_sweater_conditions sc) : 
  sc.monday + sc.tuesday + sc.wednesday + sc.thursday + sc.friday = 34 := by
  sorry


end NUMINAMATH_CALUDE_kim_total_sweaters_l250_25067


namespace NUMINAMATH_CALUDE_perpendicular_lines_a_value_l250_25011

-- Define the coefficients of the lines
def l1_coeff (a : ℝ) : ℝ × ℝ := (a + 2, 1 - a)
def l2_coeff (a : ℝ) : ℝ × ℝ := (a - 1, 2*a + 3)

-- Define the perpendicularity condition
def perpendicular (a : ℝ) : Prop :=
  (l1_coeff a).1 * (l2_coeff a).1 + (l1_coeff a).2 * (l2_coeff a).2 = 0

-- Theorem statement
theorem perpendicular_lines_a_value (a : ℝ) :
  perpendicular a → a = 1 ∨ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_a_value_l250_25011


namespace NUMINAMATH_CALUDE_unique_solution_implies_m_half_l250_25088

/-- Given m > 0, if the equation m ln x - (1/2)x^2 + mx = 0 has a unique real solution, then m = 1/2 -/
theorem unique_solution_implies_m_half (m : ℝ) (hm : m > 0) :
  (∃! x : ℝ, m * Real.log x - (1/2) * x^2 + m * x = 0) → m = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_unique_solution_implies_m_half_l250_25088


namespace NUMINAMATH_CALUDE_new_lamp_taller_by_exact_amount_l250_25089

/-- The height difference between two lamps -/
def lamp_height_difference (old_height new_height : ℝ) : ℝ :=
  new_height - old_height

/-- Theorem stating the height difference between the new and old lamps -/
theorem new_lamp_taller_by_exact_amount : 
  lamp_height_difference 1 2.3333333333333335 = 1.3333333333333335 := by
  sorry

end NUMINAMATH_CALUDE_new_lamp_taller_by_exact_amount_l250_25089


namespace NUMINAMATH_CALUDE_average_value_function_m_range_l250_25035

/-- A function is an average value function on [a, b] if there exists x₀ ∈ (a, b) such that f(x₀) = (f(b) - f(a)) / (b - a) -/
def IsAverageValueFunction (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x₀ : ℝ, a < x₀ ∧ x₀ < b ∧ f x₀ = (f b - f a) / (b - a)

/-- The quadratic function f(x) = x² - mx - 1 -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - m*x - 1

theorem average_value_function_m_range :
  ∀ m : ℝ, IsAverageValueFunction (f m) (-1) 1 ↔ 0 < m ∧ m < 2 := by
  sorry

end NUMINAMATH_CALUDE_average_value_function_m_range_l250_25035


namespace NUMINAMATH_CALUDE_projection_a_onto_b_l250_25058

def a : ℝ × ℝ := (1, 3)
def b : ℝ × ℝ := (-2, 4)

theorem projection_a_onto_b :
  let dot_product := a.1 * b.1 + a.2 * b.2
  let magnitude_b := Real.sqrt (b.1^2 + b.2^2)
  dot_product / magnitude_b = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_projection_a_onto_b_l250_25058


namespace NUMINAMATH_CALUDE_journal_writing_sessions_per_week_l250_25087

theorem journal_writing_sessions_per_week 
  (pages_per_session : ℕ) 
  (total_pages : ℕ) 
  (total_weeks : ℕ) 
  (h1 : pages_per_session = 4) 
  (h2 : total_pages = 72) 
  (h3 : total_weeks = 6) : 
  (total_pages / pages_per_session) / total_weeks = 3 := by
sorry

end NUMINAMATH_CALUDE_journal_writing_sessions_per_week_l250_25087


namespace NUMINAMATH_CALUDE_investment_problem_l250_25006

theorem investment_problem (total : ℝ) (rate_greater rate_smaller : ℝ) (income_diff : ℝ) :
  total = 10000 ∧ 
  rate_greater = 0.06 ∧ 
  rate_smaller = 0.05 ∧ 
  income_diff = 160 →
  ∃ (greater smaller : ℝ),
    greater + smaller = total ∧
    rate_greater * greater = rate_smaller * smaller + income_diff ∧
    smaller = 4000 :=
by sorry

end NUMINAMATH_CALUDE_investment_problem_l250_25006


namespace NUMINAMATH_CALUDE_line_equation_midpoint_line_equation_vector_ratio_l250_25024

-- Define the point P
def P : ℝ × ℝ := (-3, 1)

-- Define the line l passing through P and intersecting x-axis at A and y-axis at B
def line_l (A B : ℝ × ℝ) : Prop :=
  A.2 = 0 ∧ B.1 = 0 ∧ ∃ t : ℝ, P = t • A + (1 - t) • B

-- Define the midpoint condition
def is_midpoint (P A B : ℝ × ℝ) : Prop :=
  P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Define the vector ratio condition
def vector_ratio (P A B : ℝ × ℝ) : Prop :=
  (A.1 - P.1, A.2 - P.2) = (2 * (P.1 - B.1), 2 * (P.2 - B.2))

-- Theorem for case I
theorem line_equation_midpoint (A B : ℝ × ℝ) :
  line_l A B → is_midpoint P A B →
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x y : ℝ, (x - 3*y + 6 = 0) ↔ k * (x - A.1) = k * (y - A.2) :=
sorry

-- Theorem for case II
theorem line_equation_vector_ratio (A B : ℝ × ℝ) :
  line_l A B → vector_ratio P A B →
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x y : ℝ, (x - 6*y + 9 = 0) ↔ k * (x - A.1) = k * (y - A.2) :=
sorry

end NUMINAMATH_CALUDE_line_equation_midpoint_line_equation_vector_ratio_l250_25024


namespace NUMINAMATH_CALUDE_solution_check_l250_25020

def is_solution (x : ℝ) : Prop :=
  4 * x + 5 = 8 * x - 3

theorem solution_check :
  is_solution 2 ∧ ¬is_solution 3 := by
  sorry

end NUMINAMATH_CALUDE_solution_check_l250_25020


namespace NUMINAMATH_CALUDE_vector_problem_l250_25025

def a : ℝ × ℝ := (1, 2)
def b (x : ℝ) : ℝ × ℝ := (x, 6)
def c (x : ℝ) : ℝ × ℝ := 2 • a + b x

theorem vector_problem (x : ℝ) :
  (∃ y, b y ≠ r • a ∧ r ≠ 0) →  -- non-collinearity condition
  ‖a - b x‖ = 2 * Real.sqrt 5 →
  c x = (1, 10) := by
  sorry

end NUMINAMATH_CALUDE_vector_problem_l250_25025


namespace NUMINAMATH_CALUDE_rectangle_area_ratio_l250_25086

theorem rectangle_area_ratio (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) :
  (2 * (a + c) = 2 * (2 * (b + c))) → (a = 2 * b) →
  ((a * c) = 2 * (b * c)) := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_ratio_l250_25086


namespace NUMINAMATH_CALUDE_sara_spent_calculation_l250_25054

/-- Calculates the total amount Sara spent on movies and snacks -/
def sara_total_spent (ticket_price : ℝ) (num_tickets : ℕ) (student_discount : ℝ) 
  (rented_movie_price : ℝ) (purchased_movie_price : ℝ) (snacks_price : ℝ) (sales_tax_rate : ℝ) : ℝ :=
  let discounted_tickets := ticket_price * num_tickets * (1 - student_discount)
  let taxable_items := discounted_tickets + rented_movie_price + purchased_movie_price
  let sales_tax := taxable_items * sales_tax_rate
  discounted_tickets + rented_movie_price + purchased_movie_price + sales_tax + snacks_price

/-- Theorem stating that Sara's total spent is $43.89 -/
theorem sara_spent_calculation : 
  sara_total_spent 10.62 2 0.1 1.59 13.95 7.50 0.05 = 43.89 := by
  sorry

#eval sara_total_spent 10.62 2 0.1 1.59 13.95 7.50 0.05

end NUMINAMATH_CALUDE_sara_spent_calculation_l250_25054


namespace NUMINAMATH_CALUDE_customer_outreach_time_calculation_l250_25094

/-- Represents the daily work schedule of a social media account manager --/
structure WorkSchedule where
  total_time : ℝ
  marketing_time : ℝ
  customer_outreach_time : ℝ
  advertisement_time : ℝ

/-- Theorem stating the correct time spent on customer outreach posts --/
theorem customer_outreach_time_calculation (schedule : WorkSchedule) 
  (h1 : schedule.total_time = 8)
  (h2 : schedule.marketing_time = 2)
  (h3 : schedule.advertisement_time = schedule.customer_outreach_time / 2)
  (h4 : schedule.total_time = schedule.marketing_time + schedule.customer_outreach_time + schedule.advertisement_time) :
  schedule.customer_outreach_time = 4 := by
  sorry

end NUMINAMATH_CALUDE_customer_outreach_time_calculation_l250_25094


namespace NUMINAMATH_CALUDE_bug_total_distance_l250_25075

def bug_path : List ℤ := [-3, 0, -8, 10]

def total_distance (path : List ℤ) : ℕ :=
  (path.zip (path.tail!)).foldl (fun acc (a, b) => acc + (a - b).natAbs) 0

theorem bug_total_distance :
  total_distance bug_path = 29 := by
  sorry

end NUMINAMATH_CALUDE_bug_total_distance_l250_25075


namespace NUMINAMATH_CALUDE_matrix_equation_solution_l250_25032

theorem matrix_equation_solution :
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![3*x, 2; 3, 5*x]
  ∀ x : ℝ, (A.det = 16) ↔ (x = Real.sqrt (22/15) ∨ x = -Real.sqrt (22/15)) :=
by sorry

end NUMINAMATH_CALUDE_matrix_equation_solution_l250_25032


namespace NUMINAMATH_CALUDE_intersection_when_m_is_two_subset_condition_l250_25064

-- Define set A
def A : Set ℝ := {y | ∃ x, -13/2 ≤ x ∧ x ≤ 3/2 ∧ y = Real.sqrt (3 - 2*x)}

-- Define set B
def B (m : ℝ) : Set ℝ := {x | 1 - m ≤ x ∧ x ≤ m + 1}

-- Theorem 1: When m = 2, A ∩ B = [0, 3]
theorem intersection_when_m_is_two : 
  A ∩ B 2 = Set.Icc 0 3 := by sorry

-- Theorem 2: B ⊆ A if and only if m ≤ 1
theorem subset_condition : 
  ∀ m : ℝ, B m ⊆ A ↔ m ≤ 1 := by sorry

end NUMINAMATH_CALUDE_intersection_when_m_is_two_subset_condition_l250_25064


namespace NUMINAMATH_CALUDE_calculate_expression_l250_25000

theorem calculate_expression : 3 * 7.5 * (6 + 4) / 2.5 = 90 := by sorry

end NUMINAMATH_CALUDE_calculate_expression_l250_25000


namespace NUMINAMATH_CALUDE_major_axis_length_for_given_conditions_l250_25028

/-- The length of the major axis of an ellipse formed by intersecting a right circular cylinder --/
def majorAxisLength (cylinderRadius : ℝ) (majorToMinorRatio : ℝ) : ℝ :=
  2 * cylinderRadius * majorToMinorRatio

/-- Theorem: The major axis length is 6 for a cylinder of radius 2 and 50% longer major axis --/
theorem major_axis_length_for_given_conditions :
  majorAxisLength 2 1.5 = 6 := by
  sorry

end NUMINAMATH_CALUDE_major_axis_length_for_given_conditions_l250_25028


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l250_25074

def A : Set ℝ := {x | x > -2}
def B : Set ℝ := {x | 1 - x > 0}

theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | -2 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l250_25074


namespace NUMINAMATH_CALUDE_inequality_proof_l250_25005

theorem inequality_proof (x₁ x₂ y₁ y₂ z₁ z₂ : ℝ) 
  (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) 
  (hy₁ : x₁ * y₁ > z₁^2) (hy₂ : x₂ * y₂ > z₂^2) : 
  8 / ((x₁ + x₂) * (y₁ + y₂) - (z₁ + z₂)^2) ≤ 1 / (x₁ * y₁ - z₁^2) + 1 / (x₂ * y₂ - z₂^2) := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l250_25005


namespace NUMINAMATH_CALUDE_equation_solution_l250_25045

theorem equation_solution : ∃ k : ℤ, 2^4 - 6 = 3^3 + k ∧ k = -17 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l250_25045


namespace NUMINAMATH_CALUDE_original_sales_tax_percentage_l250_25010

/-- Proves that the original sales tax percentage was 5% given the conditions of the problem -/
theorem original_sales_tax_percentage
  (item_price : ℝ)
  (reduced_tax_rate : ℝ)
  (tax_difference : ℝ)
  (h1 : item_price = 1000)
  (h2 : reduced_tax_rate = 0.04)
  (h3 : tax_difference = 10)
  (h4 : item_price * reduced_tax_rate + tax_difference = item_price * (original_tax_rate / 100)) :
  original_tax_rate = 5 := by
  sorry

end NUMINAMATH_CALUDE_original_sales_tax_percentage_l250_25010


namespace NUMINAMATH_CALUDE_hex_count_and_sum_l250_25022

/-- Converts a positive integer to its hexadecimal representation --/
def toHex (n : ℕ+) : List (Fin 16) := sorry

/-- Checks if a hexadecimal representation uses only digits 0-9 --/
def usesOnlyDigits (hex : List (Fin 16)) : Prop := sorry

/-- Counts numbers in [1, n] whose hexadecimal representation uses only digits 0-9 --/
def countOnlyDigits (n : ℕ+) : ℕ := sorry

/-- Computes the sum of digits of a natural number --/
def sumOfDigits (n : ℕ) : ℕ := sorry

theorem hex_count_and_sum :
  let count := countOnlyDigits 500
  count = 199 ∧ sumOfDigits count = 19 := by sorry

end NUMINAMATH_CALUDE_hex_count_and_sum_l250_25022


namespace NUMINAMATH_CALUDE_inequality_holds_iff_a_less_than_negative_one_l250_25015

theorem inequality_holds_iff_a_less_than_negative_one (a : ℝ) : 
  (∀ x : ℝ, |x| ≤ 1 → x^2 - (a + 1) * x + a + 1 > 0) ↔ a < -1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_a_less_than_negative_one_l250_25015


namespace NUMINAMATH_CALUDE_emma_chocolates_l250_25093

theorem emma_chocolates (emma liam : ℕ) : 
  emma = liam + 10 →
  liam = emma / 3 →
  emma = 15 := by
sorry

end NUMINAMATH_CALUDE_emma_chocolates_l250_25093


namespace NUMINAMATH_CALUDE_num_technicians_is_eight_l250_25049

/-- Represents the number of technicians in the workshop -/
def num_technicians : ℕ := sorry

/-- Represents the total number of workers in the workshop -/
def total_workers : ℕ := 24

/-- Represents the average salary of all workers -/
def avg_salary_all : ℕ := 8000

/-- Represents the average salary of technicians -/
def avg_salary_technicians : ℕ := 12000

/-- Represents the average salary of non-technician workers -/
def avg_salary_others : ℕ := 6000

/-- Theorem stating that the number of technicians is 8 given the workshop conditions -/
theorem num_technicians_is_eight :
  num_technicians = 8 ∧
  num_technicians + (total_workers - num_technicians) = total_workers ∧
  num_technicians * avg_salary_technicians +
    (total_workers - num_technicians) * avg_salary_others =
    total_workers * avg_salary_all :=
by sorry

end NUMINAMATH_CALUDE_num_technicians_is_eight_l250_25049


namespace NUMINAMATH_CALUDE_carl_lemonade_sales_l250_25044

/-- 
Given:
- Stanley sells 4 cups of lemonade per hour
- Carl sells some cups of lemonade per hour
- Carl sold 9 more cups than Stanley in 3 hours

Prove that Carl sold 7 cups of lemonade per hour
-/
theorem carl_lemonade_sales (stanley_rate : ℕ) (carl_rate : ℕ) (hours : ℕ) (difference : ℕ) :
  stanley_rate = 4 →
  hours = 3 →
  difference = 9 →
  carl_rate * hours = stanley_rate * hours + difference →
  carl_rate = 7 :=
by
  sorry

#check carl_lemonade_sales

end NUMINAMATH_CALUDE_carl_lemonade_sales_l250_25044


namespace NUMINAMATH_CALUDE_fraction_division_problem_expression_evaluation_problem_l250_25009

-- Problem 1
theorem fraction_division_problem :
  (3/4 - 7/8) / (-7/8) = 1 + 1/7 := by sorry

-- Problem 2
theorem expression_evaluation_problem :
  2^1 - |0 - 4| + (1/3) * (-3^2) = -5 := by sorry

end NUMINAMATH_CALUDE_fraction_division_problem_expression_evaluation_problem_l250_25009


namespace NUMINAMATH_CALUDE_part_one_part_two_part_three_l250_25007

-- Define the system of equations
def system (a b x y : ℝ) : Prop :=
  x + y = 2 * a - b - 4 ∧ x - y = b - 4

-- Define point P
structure Point where
  x : ℝ
  y : ℝ

-- Part 1
theorem part_one (a b : ℝ) (P : Point) :
  a = 1 → b = 2 → system a b P.x P.y → P.x = -3 ∧ P.y = -1 := by sorry

-- Part 2
theorem part_two (a b : ℝ) (P : Point) :
  system a b P.x P.y →
  P.x < 0 ∧ P.y > 0 →
  (∃ (n : ℕ), n = 4 ∧ ∀ (m : ℤ), (∃ (a' : ℝ), a' = a ∧ system a' b P.x P.y) → m ≤ n) →
  -1 ≤ b ∧ b < 0 := by sorry

-- Part 3
theorem part_three (a b t : ℝ) (P : Point) :
  system a b P.x P.y →
  (∃! (z : ℝ), z = 2 ∧ P.y * z + P.x + 4 = 0) →
  (a * t > b ↔ t > 3/2 ∨ t < 3/2) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_part_three_l250_25007


namespace NUMINAMATH_CALUDE_initial_discount_percentage_l250_25029

-- Define the original price of the dress
variable (d : ℝ)
-- Define the initial discount percentage
variable (x : ℝ)

-- Theorem statement
theorem initial_discount_percentage
  (h1 : d > 0)  -- Assuming the original price is positive
  (h2 : 0 ≤ x ∧ x ≤ 100)  -- The discount percentage is between 0 and 100
  (h3 : d * (1 - x / 100) * (1 - 40 / 100) = d * 0.33)  -- The equation representing the final price
  : x = 45 := by
  sorry

end NUMINAMATH_CALUDE_initial_discount_percentage_l250_25029


namespace NUMINAMATH_CALUDE_fraction_increase_l250_25050

theorem fraction_increase (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (2*x * 2*y) / (2*x + 2*y) = 2 * (x*y / (x + y)) :=
sorry

end NUMINAMATH_CALUDE_fraction_increase_l250_25050


namespace NUMINAMATH_CALUDE_parabola_midpoint_locus_ratio_l250_25070

/-- A parabola with vertex and focus -/
structure Parabola where
  vertex : ℝ × ℝ
  focus : ℝ × ℝ

/-- The locus of midpoints of chords of a parabola -/
def midpointLocus (p : Parabola) (angle : ℝ) : Parabola :=
  sorry

/-- The ratio of distances between foci and vertices of two related parabolas -/
def focusVertexRatio (p1 p2 : Parabola) : ℝ :=
  sorry

theorem parabola_midpoint_locus_ratio (p : Parabola) :
  let q := midpointLocus p (π / 2)
  focusVertexRatio p q = 7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_parabola_midpoint_locus_ratio_l250_25070


namespace NUMINAMATH_CALUDE_valid_arrangement_exists_l250_25069

def is_valid_arrangement (n : ℕ) : Prop :=
  (∀ d : ℕ, d ∈ Finset.range 10 → (n.digits 10).count d = 1) ∧
  (∀ k : ℕ, 2 ≤ k → k ≤ 18 → n % k = 0)

theorem valid_arrangement_exists : ∃ n : ℕ, is_valid_arrangement n :=
sorry

end NUMINAMATH_CALUDE_valid_arrangement_exists_l250_25069


namespace NUMINAMATH_CALUDE_function_value_l250_25026

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then a^x else (1/4 - a)*x + 2*a

theorem function_value (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) < 0) →
  a = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_function_value_l250_25026


namespace NUMINAMATH_CALUDE_complex_number_real_condition_l250_25071

theorem complex_number_real_condition (a b : ℝ) :
  let z : ℂ := Complex.mk (a^2 + b^2) (a + |a|)
  (z.im = 0) ↔ (a ≤ 0) := by sorry

end NUMINAMATH_CALUDE_complex_number_real_condition_l250_25071


namespace NUMINAMATH_CALUDE_sin_negative_1740_degrees_l250_25019

theorem sin_negative_1740_degrees : Real.sin ((-1740 : ℝ) * π / 180) = Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_sin_negative_1740_degrees_l250_25019


namespace NUMINAMATH_CALUDE_min_value_theorem_l250_25023

theorem min_value_theorem (n : ℕ+) : 
  (n : ℝ) / 3 + 27 / (n : ℝ) ≥ 6 ∧ ∃ m : ℕ+, (m : ℝ) / 3 + 27 / (m : ℝ) = 6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l250_25023


namespace NUMINAMATH_CALUDE_wage_multiple_l250_25068

/-- Given Kem's hourly wage and Shem's daily wage for 8 hours, 
    calculate the multiple of Shem's hourly wage compared to Kem's. -/
theorem wage_multiple (kem_hourly_wage shem_daily_wage : ℚ) 
  (h1 : kem_hourly_wage = 4)
  (h2 : shem_daily_wage = 80)
  (h3 : shem_daily_wage = 8 * (shem_daily_wage / 8)) : 
  (shem_daily_wage / 8) / kem_hourly_wage = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_wage_multiple_l250_25068


namespace NUMINAMATH_CALUDE_relationship_abc_l250_25008

theorem relationship_abc : 
  let a : ℝ := (1/2)^2
  let b : ℝ := 2^(1/2)
  let c : ℝ := Real.log 2 / Real.log (1/2)
  c < a ∧ a < b := by sorry

end NUMINAMATH_CALUDE_relationship_abc_l250_25008


namespace NUMINAMATH_CALUDE_functions_properties_l250_25063

noncomputable def f (ω : ℝ) (φ : ℝ) (x : ℝ) : ℝ := Real.sqrt 2 * Real.sin (ω * x + φ)
noncomputable def g (ω : ℝ) (x : ℝ) : ℝ := Real.sqrt 2 * Real.cos (ω * x)

theorem functions_properties (ω φ : ℝ) :
  ω > 0 ∧
  0 ≤ φ ∧ φ < π ∧
  (∀ x : ℝ, f ω φ (x + π / ω) = f ω φ x) ∧
  (∀ x : ℝ, g ω (x + π / ω) = g ω x) ∧
  f ω φ (-π/6) + g ω (-π/6) = 0 →
  ω = 2 ∧
  φ = π/6 ∧
  ∀ x : ℝ, f ω φ x + g ω x = Real.sqrt 6 * Real.sin (2 * x + π/3) := by
sorry

end NUMINAMATH_CALUDE_functions_properties_l250_25063


namespace NUMINAMATH_CALUDE_line_l_theorem_l250_25003

/-- Definition of line l -/
def line_l (a : ℝ) (x y : ℝ) : Prop := (a + 1) * x + y + 2 - a = 0

/-- Intercepts are equal -/
def equal_intercepts (a : ℝ) : Prop :=
  ∃ k, k = a - 2 ∧ k = (a - 2) / (a + 1)

/-- Line does not pass through second quadrant -/
def not_in_second_quadrant (a : ℝ) : Prop :=
  -(a + 1) ≥ 0 ∧ a - 2 ≤ 0

theorem line_l_theorem :
  (∀ a : ℝ, equal_intercepts a → (a = 2 ∨ a = 0)) ∧
  (∀ a : ℝ, not_in_second_quadrant a ↔ a ≤ -1) :=
sorry

end NUMINAMATH_CALUDE_line_l_theorem_l250_25003


namespace NUMINAMATH_CALUDE_numbers_with_zero_from_1_to_700_l250_25079

def count_numbers_with_zero (lower_bound upper_bound : ℕ) : ℕ :=
  sorry

theorem numbers_with_zero_from_1_to_700 :
  count_numbers_with_zero 1 700 = 123 := by sorry

end NUMINAMATH_CALUDE_numbers_with_zero_from_1_to_700_l250_25079


namespace NUMINAMATH_CALUDE_extreme_values_and_zero_condition_l250_25056

/-- The cubic function f(x) = x^3 - x^2 - x - a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - x^2 - x - a

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3*x^2 - 2*x - 1

theorem extreme_values_and_zero_condition (a : ℝ) :
  (∃ x_max : ℝ, f a x_max = 5/27 - a ∧ ∀ x, f a x ≤ f a x_max) ∧
  (∃ x_min : ℝ, f a x_min = -1 - a ∧ ∀ x, f a x ≥ f a x_min) ∧
  (∃! x, f a x = 0) ↔ (a < -1 ∨ a > 5/27) :=
sorry

end NUMINAMATH_CALUDE_extreme_values_and_zero_condition_l250_25056


namespace NUMINAMATH_CALUDE_integer_triangle_properties_l250_25076

/-- A triangle with positive integer side lengths and circumradius -/
structure IntegerTriangle where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  R : ℕ+

/-- Properties of an integer triangle -/
theorem integer_triangle_properties (T : IntegerTriangle) :
  ∃ (r : ℕ+) (P : ℕ),
    (∃ (k : ℕ), P = 4 * k) ∧
    (∃ (m n l : ℕ), T.a = 2 * m ∧ T.b = 2 * n ∧ T.c = 2 * l) := by
  sorry


end NUMINAMATH_CALUDE_integer_triangle_properties_l250_25076


namespace NUMINAMATH_CALUDE_repeating_decimal_ratio_l250_25027

-- Define repeating decimal 0.overline{36}
def repeating_36 : ℚ := 36 / 99

-- Define repeating decimal 0.overline{09}
def repeating_09 : ℚ := 9 / 99

-- Theorem statement
theorem repeating_decimal_ratio : repeating_36 / repeating_09 = 4 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_ratio_l250_25027


namespace NUMINAMATH_CALUDE_smallest_sum_of_squared_ratios_l250_25066

theorem smallest_sum_of_squared_ratios (c d : ℕ) (hc : c > 0) (hd : d > 0) :
  ∃ (min : ℚ), min = 2 ∧
  (((c + d : ℚ) / (c - d : ℚ))^2 + ((c - d : ℚ) / (c + d : ℚ))^2 ≥ min) ∧
  ∃ (c' d' : ℕ), c' > 0 ∧ d' > 0 ∧
  ((c' + d' : ℚ) / (c' - d' : ℚ))^2 + ((c' - d' : ℚ) / (c' + d' : ℚ))^2 = min :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_squared_ratios_l250_25066


namespace NUMINAMATH_CALUDE_sum_of_x_values_is_zero_l250_25078

theorem sum_of_x_values_is_zero : 
  ∃ (x₁ x₂ : ℝ), 
    (x₁^2 + 6^2 = 144) ∧ 
    (x₂^2 + 6^2 = 144) ∧ 
    (x₁ + x₂ = 0) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_values_is_zero_l250_25078


namespace NUMINAMATH_CALUDE_prob_king_or_ace_eq_two_thirteenth_l250_25060

/-- A standard deck of cards -/
structure Deck :=
  (cards : Finset (Nat × Nat))
  (card_count : cards.card = 52)
  (rank_count : (cards.image (·.1)).card = 13)
  (suit_count : (cards.image (·.2)).card = 4)
  (unique_pairs : ∀ r s, (r, s) ∈ cards → r ∈ Finset.range 13 ∧ s ∈ Finset.range 4)

/-- The probability of drawing a King or an Ace from the top of a shuffled deck -/
def prob_king_or_ace (d : Deck) : ℚ :=
  (d.cards.filter (λ p => p.1 = 0 ∨ p.1 = 12)).card / d.cards.card

/-- Theorem: The probability of drawing a King or an Ace is 2/13 -/
theorem prob_king_or_ace_eq_two_thirteenth (d : Deck) : 
  prob_king_or_ace d = 2 / 13 := by
  sorry

end NUMINAMATH_CALUDE_prob_king_or_ace_eq_two_thirteenth_l250_25060


namespace NUMINAMATH_CALUDE_horner_method_for_f_l250_25012

def horner_polynomial (a : List ℝ) (x : ℝ) : ℝ :=
  a.foldl (fun acc coeff => acc * x + coeff) 0

def f (x : ℝ) : ℝ := 5 * x^6 + 3 * x^4 + 2 * x + 1

theorem horner_method_for_f :
  f 2 = horner_polynomial [1, 2, 0, 3, 0, 0, 5] 2 ∧ 
  horner_polynomial [1, 2, 0, 3, 0, 0, 5] 2 = 373 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_for_f_l250_25012


namespace NUMINAMATH_CALUDE_min_expression_l250_25047

theorem min_expression (k : ℝ) (x y z t : ℝ) 
  (h1 : k ≥ 0) 
  (h2 : x > 0) (h3 : y > 0) (h4 : z > 0) (h5 : t > 0) 
  (h6 : x + y + z + t = k) : 
  x / (1 + y^2) + y / (1 + x^2) + z / (1 + t^2) + t / (1 + z^2) ≥ 4 * k / (4 + k^2) := by
  sorry

end NUMINAMATH_CALUDE_min_expression_l250_25047


namespace NUMINAMATH_CALUDE_inequality_solution_set_l250_25017

theorem inequality_solution_set (x : ℝ) (h : x ≠ 3) :
  (2 * x - 1) / (x - 3) ≥ 1 ↔ x > 3 ∨ x ≤ -2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l250_25017


namespace NUMINAMATH_CALUDE_initial_marbles_equation_l250_25065

/-- The number of marbles Connie had initially -/
def initial_marbles : ℕ := sorry

/-- The number of marbles Connie gave to Juan -/
def marbles_given : ℕ := 73

/-- The number of marbles Connie has left -/
def marbles_left : ℕ := 70

/-- Theorem stating that the initial number of marbles is equal to
    the sum of marbles given away and marbles left -/
theorem initial_marbles_equation : initial_marbles = marbles_given + marbles_left := by
  sorry

end NUMINAMATH_CALUDE_initial_marbles_equation_l250_25065


namespace NUMINAMATH_CALUDE_divisor_of_sum_l250_25099

theorem divisor_of_sum (n : ℕ) (a : ℕ) (d : ℕ) : 
  n = 425897 → a = 7 → d = 7 → (n + a) % d = 0 := by
  sorry

end NUMINAMATH_CALUDE_divisor_of_sum_l250_25099


namespace NUMINAMATH_CALUDE_problem_statement_l250_25072

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = a * b) :
  (a + b ≥ 4) ∧ (a + 4 * b ≥ 9) ∧ (1 / a^2 + 2 / b^2 ≥ 2 / 3) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l250_25072


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_l250_25080

def binomial_coefficient (n k : ℕ) : ℕ := sorry

theorem binomial_expansion_coefficient (a : ℚ) : 
  (binomial_coefficient 6 3 : ℚ) * a^3 = 5/2 → a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_l250_25080


namespace NUMINAMATH_CALUDE_grade_assignment_count_l250_25059

/-- The number of students in the class -/
def num_students : ℕ := 12

/-- The number of possible grades -/
def num_grades : ℕ := 4

/-- The number of ways to assign grades to all students -/
def ways_to_assign_grades : ℕ := num_grades ^ num_students

theorem grade_assignment_count :
  ways_to_assign_grades = 16777216 :=
sorry

end NUMINAMATH_CALUDE_grade_assignment_count_l250_25059


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l250_25034

def A : Set ℝ := {x | -1 < x ∧ x ≤ 4}
def B : Set ℝ := {x | -3 ≤ x ∧ x < 1}

theorem union_of_A_and_B : A ∪ B = {x : ℝ | -3 ≤ x ∧ x ≤ 4} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l250_25034


namespace NUMINAMATH_CALUDE_children_clothing_production_l250_25042

-- Define the constants
def total_sets : ℕ := 50
def type_a_fabric : ℝ := 38
def type_b_fabric : ℝ := 26

-- Define the fabric requirements and profits for each size
def size_l_type_a : ℝ := 0.5
def size_l_type_b : ℝ := 1
def size_l_profit : ℝ := 45

def size_m_type_a : ℝ := 0.9
def size_m_type_b : ℝ := 0.2
def size_m_profit : ℝ := 30

-- Define the profit function
def profit_function (x : ℝ) : ℝ := 15 * x + 1500

-- Theorem statement
theorem children_clothing_production (x : ℝ) :
  (17.5 ≤ x ∧ x ≤ 20) →
  (∀ y : ℝ, y = profit_function x) ∧
  (x * size_l_type_a + (total_sets - x) * size_m_type_a ≤ type_a_fabric) ∧
  (x * size_l_type_b + (total_sets - x) * size_m_type_b ≤ type_b_fabric) :=
by sorry

end NUMINAMATH_CALUDE_children_clothing_production_l250_25042


namespace NUMINAMATH_CALUDE_exam_score_problem_l250_25062

theorem exam_score_problem (total_questions : ℕ) 
  (correct_score wrong_score total_score : ℤ) :
  total_questions = 60 →
  correct_score = 4 →
  wrong_score = -1 →
  total_score = 120 →
  ∃ (correct_answers : ℕ),
    correct_answers ≤ total_questions ∧
    correct_answers * correct_score + (total_questions - correct_answers) * wrong_score = total_score ∧
    correct_answers = 36 :=
by sorry

end NUMINAMATH_CALUDE_exam_score_problem_l250_25062


namespace NUMINAMATH_CALUDE_machine_ok_l250_25039

/-- The nominal portion weight -/
def nominal_weight : ℝ := 390

/-- The greatest deviation from the mean among preserved measurements -/
def max_deviation : ℝ := 39

/-- Condition: The greatest deviation doesn't exceed 10% of the nominal weight -/
axiom deviation_within_limit : max_deviation ≤ 0.1 * nominal_weight

/-- Condition: Deviations of unreadable measurements are less than the max deviation -/
axiom unreadable_deviations_less : ∀ x : ℝ, x < max_deviation → x < nominal_weight - 380

/-- Definition: A machine requires repair if the standard deviation exceeds max_deviation -/
def requires_repair (std_dev : ℝ) : Prop := std_dev > max_deviation

/-- Theorem: The machine does not require repair -/
theorem machine_ok : ∃ std_dev : ℝ, std_dev ≤ max_deviation ∧ ¬(requires_repair std_dev) := by
  sorry


end NUMINAMATH_CALUDE_machine_ok_l250_25039


namespace NUMINAMATH_CALUDE_is_point_of_tangency_l250_25043

/-- The point of tangency between two circles -/
def point_of_tangency : ℝ × ℝ := (2.5, 5)

/-- First circle equation -/
def circle1 (x y : ℝ) : Prop :=
  x^2 - 2*x + y^2 - 10*y + 17 = 0

/-- Second circle equation -/
def circle2 (x y : ℝ) : Prop :=
  x^2 - 8*x + y^2 - 10*y + 49 = 0

/-- Theorem stating that point_of_tangency is the point of tangency between the two circles -/
theorem is_point_of_tangency :
  let (x, y) := point_of_tangency
  circle1 x y ∧ circle2 x y ∧
  ∀ (x' y' : ℝ), (x' ≠ x ∨ y' ≠ y) → ¬(circle1 x' y' ∧ circle2 x' y') :=
by sorry

end NUMINAMATH_CALUDE_is_point_of_tangency_l250_25043


namespace NUMINAMATH_CALUDE_min_ratio_of_integers_with_mean_l250_25033

theorem min_ratio_of_integers_with_mean (x y : ℤ) : 
  10 ≤ x ∧ x ≤ 150 → 
  10 ≤ y ∧ y ≤ 150 → 
  (x + y) / 2 = 75 → 
  ∃ (x' y' : ℤ), 
    10 ≤ x' ∧ x' ≤ 150 ∧ 
    10 ≤ y' ∧ y' ≤ 150 ∧ 
    (x' + y') / 2 = 75 ∧ 
    x' / y' ≤ x / y ∧
    x' / y' = 1 / 14 :=
by sorry

end NUMINAMATH_CALUDE_min_ratio_of_integers_with_mean_l250_25033


namespace NUMINAMATH_CALUDE_marathon_remainder_yards_l250_25004

/-- Represents the distance of a marathon in miles and yards -/
structure MarathonDistance :=
  (miles : ℕ)
  (yards : ℕ)

/-- Represents a total distance in miles and yards -/
structure TotalDistance :=
  (miles : ℕ)
  (yards : ℕ)

def marathon_distance : MarathonDistance :=
  { miles := 26, yards := 385 }

def yards_per_mile : ℕ := 1760

def num_marathons : ℕ := 15

theorem marathon_remainder_yards :
  ∃ (m : ℕ) (y : ℕ), 
    y < yards_per_mile ∧
    TotalDistance.yards (
      {miles := m, 
       yards := y} : TotalDistance
    ) = 495 ∧
    m * yards_per_mile + y = 
      num_marathons * (marathon_distance.miles * yards_per_mile + marathon_distance.yards) :=
by sorry

end NUMINAMATH_CALUDE_marathon_remainder_yards_l250_25004


namespace NUMINAMATH_CALUDE_catch_up_distance_l250_25098

/-- Proves that B catches up with A 200 km from the start given the specified conditions -/
theorem catch_up_distance (a_speed b_speed : ℝ) (time_diff : ℝ) (catch_up_dist : ℝ) : 
  a_speed = 10 →
  b_speed = 20 →
  time_diff = 10 →
  catch_up_dist = 200 →
  catch_up_dist = b_speed * (time_diff + catch_up_dist / b_speed) :=
by sorry

end NUMINAMATH_CALUDE_catch_up_distance_l250_25098


namespace NUMINAMATH_CALUDE_expansion_property_l250_25095

/-- Given that for some natural number n, in the expansion of (x^4 + 1/x)^n,
    the binomial coefficient of the third term is 35 more than that of the second term,
    prove that n = 10 and the constant term in the expansion is 45. -/
theorem expansion_property (n : ℕ) 
  (h : Nat.choose n 2 - Nat.choose n 1 = 35) : 
  n = 10 ∧ Nat.choose 10 8 = 45 := by
  sorry

end NUMINAMATH_CALUDE_expansion_property_l250_25095


namespace NUMINAMATH_CALUDE_plot_length_is_80_l250_25083

/-- A rectangular plot with specific dimensions and fencing cost. -/
structure RectangularPlot where
  breadth : ℝ
  length : ℝ
  fencing_cost_per_meter : ℝ
  total_fencing_cost : ℝ
  length_breadth_difference : ℝ
  h_length : length = breadth + length_breadth_difference
  h_perimeter : 2 * (length + breadth) = total_fencing_cost / fencing_cost_per_meter

/-- The length of the rectangular plot is 80 meters given the specified conditions. -/
theorem plot_length_is_80 (plot : RectangularPlot)
  (h_length_diff : plot.length_breadth_difference = 60)
  (h_fencing_cost : plot.fencing_cost_per_meter = 26.5)
  (h_total_cost : plot.total_fencing_cost = 5300) :
  plot.length = 80 := by
  sorry

end NUMINAMATH_CALUDE_plot_length_is_80_l250_25083


namespace NUMINAMATH_CALUDE_vector_subtraction_l250_25002

/-- Given two vectors a and b in ℝ², prove that their difference is (1, 2). -/
theorem vector_subtraction (a b : ℝ × ℝ) (ha : a = (2, 3)) (hb : b = (1, 1)) :
  a - b = (1, 2) := by sorry

end NUMINAMATH_CALUDE_vector_subtraction_l250_25002


namespace NUMINAMATH_CALUDE_writers_birth_months_l250_25013

/-- The total number of famous writers -/
def total_writers : ℕ := 200

/-- The number of writers born in October -/
def october_births : ℕ := 15

/-- The number of writers born in July -/
def july_births : ℕ := 14

/-- The percentage of writers born in October -/
def october_percentage : ℚ := (october_births : ℚ) / (total_writers : ℚ) * 100

/-- The percentage of writers born in July -/
def july_percentage : ℚ := (july_births : ℚ) / (total_writers : ℚ) * 100

theorem writers_birth_months :
  october_percentage = 15/2 ∧
  july_percentage = 7 ∧
  october_percentage > july_percentage :=
by sorry

end NUMINAMATH_CALUDE_writers_birth_months_l250_25013


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l250_25096

theorem complex_magnitude_problem (z : ℂ) :
  Complex.abs z * (3 * z + 2 * Complex.I) = 2 * (Complex.I * z - 6) →
  Complex.abs z = 2 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l250_25096


namespace NUMINAMATH_CALUDE_intersection_implies_sum_l250_25030

-- Define the sets M, N, and K
def M : Set ℝ := {x : ℝ | x^2 - 4*x < 0}
def N (m : ℝ) : Set ℝ := {x : ℝ | m < x ∧ x < 5}
def K (n : ℝ) : Set ℝ := {x : ℝ | 3 < x ∧ x < n}

-- State the theorem
theorem intersection_implies_sum (m n : ℝ) : 
  M ∩ N m = K n → m + n = 7 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_sum_l250_25030


namespace NUMINAMATH_CALUDE_small_bottle_price_approx_l250_25091

/-- The price of a small bottle that results in the given average price -/
def price_small_bottle (large_quantity : ℕ) (small_quantity : ℕ) (large_price : ℚ) (average_price : ℚ) : ℚ :=
  ((average_price * (large_quantity + small_quantity : ℚ)) - (large_quantity : ℚ) * large_price) / (small_quantity : ℚ)

/-- Theorem stating that the price of small bottles is approximately $1.38 -/
theorem small_bottle_price_approx :
  let large_quantity : ℕ := 1300
  let small_quantity : ℕ := 750
  let large_price : ℚ := 189/100
  let average_price : ℚ := 17034/10000
  let calculated_price := price_small_bottle large_quantity small_quantity large_price average_price
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1/100 ∧ |calculated_price - 138/100| < ε :=
sorry

end NUMINAMATH_CALUDE_small_bottle_price_approx_l250_25091


namespace NUMINAMATH_CALUDE_sqrt_equation_equals_difference_l250_25021

theorem sqrt_equation_equals_difference (a b : ℤ) : 
  Real.sqrt (16 - 12 * Real.cos (40 * π / 180)) = a + b * (1 / Real.cos (40 * π / 180)) →
  a = 4 ∧ b = -1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_equals_difference_l250_25021


namespace NUMINAMATH_CALUDE_two_dice_same_side_probability_l250_25081

/-- Represents a 10-sided die with specific side distributions -/
structure TenSidedDie :=
  (gold : Nat)
  (silver : Nat)
  (diamond : Nat)
  (rainbow : Nat)
  (total : Nat)
  (sides_sum : gold + silver + diamond + rainbow = total)

/-- The probability of rolling two dice and getting the same color or pattern -/
def sameSideProbability (die : TenSidedDie) : ℚ :=
  (die.gold ^ 2 + die.silver ^ 2 + die.diamond ^ 2 + die.rainbow ^ 2) / die.total ^ 2

/-- Theorem: The probability of rolling two 10-sided dice with the given distribution
    and getting the same color or pattern is 3/10 -/
theorem two_dice_same_side_probability :
  ∃ (die : TenSidedDie),
    die.gold = 3 ∧
    die.silver = 4 ∧
    die.diamond = 2 ∧
    die.rainbow = 1 ∧
    die.total = 10 ∧
    sameSideProbability die = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_two_dice_same_side_probability_l250_25081


namespace NUMINAMATH_CALUDE_budget_allocation_l250_25046

def total_budget : ℝ := 40000000

def policing_percentage : ℝ := 0.35
def education_percentage : ℝ := 0.25
def healthcare_percentage : ℝ := 0.15

def remaining_budget : ℝ := total_budget * (1 - (policing_percentage + education_percentage + healthcare_percentage))

theorem budget_allocation :
  remaining_budget = 10000000 := by sorry

end NUMINAMATH_CALUDE_budget_allocation_l250_25046


namespace NUMINAMATH_CALUDE_nut_raisin_mixture_l250_25055

/-- The number of pounds of nuts mixed with raisins -/
def pounds_of_nuts : ℝ := 4

/-- The number of pounds of raisins -/
def pounds_of_raisins : ℝ := 3

/-- The ratio of the cost of nuts to the cost of raisins -/
def cost_ratio : ℝ := 4

/-- The ratio of the cost of raisins to the total cost of the mixture -/
def raisin_cost_ratio : ℝ := 0.15789473684210525

theorem nut_raisin_mixture :
  let total_cost := pounds_of_raisins + cost_ratio * pounds_of_nuts
  raisin_cost_ratio * total_cost = pounds_of_raisins :=
by sorry

end NUMINAMATH_CALUDE_nut_raisin_mixture_l250_25055


namespace NUMINAMATH_CALUDE_distance_in_one_hour_l250_25048

/-- The number of seconds in one hour -/
def seconds_per_hour : ℕ := 3600

/-- The speed of the object in feet per second -/
def speed : ℕ := 3

/-- The distance traveled by an object moving at a constant speed for a given time -/
def distance_traveled (speed : ℕ) (time : ℕ) : ℕ := speed * time

/-- Theorem: An object traveling at 3 feet per second will cover 10800 feet in one hour -/
theorem distance_in_one_hour :
  distance_traveled speed seconds_per_hour = 10800 := by
  sorry

end NUMINAMATH_CALUDE_distance_in_one_hour_l250_25048


namespace NUMINAMATH_CALUDE_stock_income_theorem_l250_25001

/-- Calculates the income from a stock investment given the rate, market value, and investment amount. -/
def calculate_income (rate : ℚ) (market_value : ℚ) (investment : ℚ) : ℚ :=
  (rate / 100) * (investment / market_value) * 100

/-- Theorem stating that given the specific conditions, the income is 650. -/
theorem stock_income_theorem (rate market_value investment : ℚ) 
  (h_rate : rate = 10)
  (h_market_value : market_value = 96)
  (h_investment : investment = 6240) :
  calculate_income rate market_value investment = 650 :=
by
  sorry

#eval calculate_income 10 96 6240

end NUMINAMATH_CALUDE_stock_income_theorem_l250_25001


namespace NUMINAMATH_CALUDE_prob_all_copresidents_theorem_l250_25057

def club_sizes : List Nat := [6, 8, 9, 10]
def num_clubs : Nat := 4
def num_copresidents : Nat := 3
def num_selected : Nat := 4

def prob_all_copresidents_selected : ℚ := 37/420

theorem prob_all_copresidents_theorem :
  let probs := club_sizes.map (λ n => (n - num_copresidents).choose 1 / n.choose num_selected)
  (1 / num_clubs) * (probs.sum) = prob_all_copresidents_selected := by
  sorry

end NUMINAMATH_CALUDE_prob_all_copresidents_theorem_l250_25057


namespace NUMINAMATH_CALUDE_no_positive_rational_root_l250_25016

theorem no_positive_rational_root : ¬∃ (q : ℚ), q > 0 ∧ q^3 - 10*q^2 + q - 2021 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_positive_rational_root_l250_25016


namespace NUMINAMATH_CALUDE_max_volume_pyramid_l250_25038

noncomputable def pyramid_volume (a b : ℝ) : ℝ :=
  (a * b * Real.sqrt (3 * a^2 - b^2)) / 6

theorem max_volume_pyramid (a : ℝ) (h : a > 0) :
  ∃ b : ℝ, b > 0 ∧ ∀ x : ℝ, x > 0 → pyramid_volume a b ≥ pyramid_volume a x :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_max_volume_pyramid_l250_25038


namespace NUMINAMATH_CALUDE_inequality_solution_l250_25090

def satisfies_inequality (x : ℤ) : Prop :=
  8.58 * (Real.log x / Real.log 4) + Real.log (Real.sqrt x - 1) / Real.log 2 < 
  Real.log (Real.log 5 / Real.log (Real.sqrt 5)) / Real.log 2

theorem inequality_solution :
  ∀ x : ℤ, satisfies_inequality x ↔ (x = 2 ∨ x = 3) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l250_25090


namespace NUMINAMATH_CALUDE_periodic_function_value_l250_25073

def is_periodic (f : ℝ → ℝ) (period : ℝ) : Prop :=
  ∀ x, f (x + period) = f x

theorem periodic_function_value (f : ℝ → ℝ) :
  is_periodic f 3 →
  (∀ x ∈ Set.Icc (-1) 2, f x = x + 1) →
  f 2017 = 2 := by
sorry

end NUMINAMATH_CALUDE_periodic_function_value_l250_25073


namespace NUMINAMATH_CALUDE_max_operations_l250_25052

def operation_count (a b : ℕ) : ℕ := sorry

theorem max_operations (a b : ℕ) (ha : a = 2000) (hb : b < 2000) :
  operation_count a b ≤ 10 := by sorry

end NUMINAMATH_CALUDE_max_operations_l250_25052


namespace NUMINAMATH_CALUDE_approximate_4_02_to_ten_thousandth_l250_25082

/-- Represents a decimal number with a specific precision -/
structure DecimalNumber where
  value : ℚ
  precision : ℕ

/-- Represents the place value in a decimal number -/
inductive PlaceValue
  | Ones
  | Tenths
  | Hundredths
  | Thousandths
  | TenThousandths

/-- Determines the place value of the last non-zero digit in a decimal number -/
def lastNonZeroDigitPlace (n : DecimalNumber) : PlaceValue :=
  sorry

/-- Approximates a decimal number to a given place value -/
def approximateTo (n : DecimalNumber) (place : PlaceValue) : DecimalNumber :=
  sorry

/-- Theorem stating that approximating 4.02 to the ten thousandth place
    results in a number accurate to the hundredth place -/
theorem approximate_4_02_to_ten_thousandth :
  let original := DecimalNumber.mk (402 / 100) 2
  let approximated := approximateTo original PlaceValue.TenThousandths
  lastNonZeroDigitPlace approximated = PlaceValue.Hundredths :=
sorry

end NUMINAMATH_CALUDE_approximate_4_02_to_ten_thousandth_l250_25082


namespace NUMINAMATH_CALUDE_quadratic_polynomial_conditions_l250_25084

-- Define the quadratic polynomial
def q (x : ℚ) : ℚ := (6/5) * x^2 - (18/5) * x - (108/5)

-- Theorem stating the conditions
theorem quadratic_polynomial_conditions :
  q (-3) = 0 ∧ q 6 = 0 ∧ q 2 = -24 := by sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_conditions_l250_25084


namespace NUMINAMATH_CALUDE_gopal_krishan_ratio_l250_25031

/-- The ratio of money between Gopal and Krishan given the conditions -/
theorem gopal_krishan_ratio :
  ∀ (ram gopal krishan : ℕ),
  ram = 735 →
  krishan = 4335 →
  7 * gopal = 17 * ram →
  (gopal : ℚ) / krishan = 1785 / 4335 :=
by sorry

end NUMINAMATH_CALUDE_gopal_krishan_ratio_l250_25031


namespace NUMINAMATH_CALUDE_intersection_with_complement_l250_25036

-- Define the universal set U
def U : Finset Nat := {1,2,3,4,5,6}

-- Define set P
def P : Finset Nat := {1,2,3,4}

-- Define set Q
def Q : Finset Nat := {3,4,5}

-- Theorem statement
theorem intersection_with_complement :
  P ∩ (U \ Q) = {1,2} := by sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l250_25036


namespace NUMINAMATH_CALUDE_jenny_bottle_payment_l250_25053

/-- Calculates the payment per bottle for Jenny's recycling --/
def payment_per_bottle (bottle_weight can_weight total_weight can_count can_payment total_payment : ℕ) : ℕ :=
  let remaining_weight := total_weight - can_count * can_weight
  let bottle_count := remaining_weight / bottle_weight
  let can_total_payment := can_count * can_payment
  let bottle_total_payment := total_payment - can_total_payment
  bottle_total_payment / bottle_count

/-- Theorem stating that Jenny's payment per bottle is 10 cents --/
theorem jenny_bottle_payment :
  payment_per_bottle 6 2 100 20 3 160 = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_jenny_bottle_payment_l250_25053

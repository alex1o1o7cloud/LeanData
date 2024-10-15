import Mathlib

namespace NUMINAMATH_CALUDE_functional_equation_solution_l3436_343680

-- Define the function type
def FunctionType := ℝ → ℝ

-- Define the functional equation
def SatisfiesEquation (f : FunctionType) : Prop :=
  ∀ x y : ℝ, x * f y + y * f x = (x + y) * f x * f y

-- Define the solution conditions
def IsSolution (f : FunctionType) : Prop :=
  (∀ x : ℝ, f x = 0) ∨
  ((∀ x : ℝ, x ≠ 0 → f x = 1) ∧ ∃ c : ℝ, f 0 = c)

-- Theorem statement
theorem functional_equation_solution (f : FunctionType) :
  SatisfiesEquation f → IsSolution f :=
sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l3436_343680


namespace NUMINAMATH_CALUDE_candy_mixture_problem_l3436_343682

theorem candy_mixture_problem (X Y : ℝ) : 
  X + Y = 10 →
  3.50 * X + 4.30 * Y = 40 →
  Y = 6.25 := by
sorry

end NUMINAMATH_CALUDE_candy_mixture_problem_l3436_343682


namespace NUMINAMATH_CALUDE_reusable_bags_estimate_conditional_probability_second_spender_l3436_343620

/-- Represents the survey data for each age group -/
structure AgeGroupData :=
  (spent_more : Nat)  -- Number of people who spent ≥ $188
  (spent_less : Nat)  -- Number of people who spent < $188

/-- Represents the survey results -/
def survey_data : List AgeGroupData := [
  ⟨8, 2⟩,   -- [20,30)
  ⟨15, 3⟩,  -- [30,40)
  ⟨23, 5⟩,  -- [40,50)
  ⟨15, 9⟩,  -- [50,60)
  ⟨9, 11⟩   -- [60,70]
]

/-- Total number of surveyed customers -/
def total_surveyed : Nat := 100

/-- Expected number of shoppers on the event day -/
def expected_shoppers : Nat := 5000

/-- Theorem for the number of reusable shopping bags to prepare -/
theorem reusable_bags_estimate :
  (expected_shoppers * (survey_data.map (·.spent_more)).sum / total_surveyed : Nat) = 3500 := by
  sorry

/-- Theorem for the conditional probability -/
theorem conditional_probability_second_spender :
  let total_spent_more := (survey_data.map (·.spent_more)).sum
  let total_spent_less := (survey_data.map (·.spent_less)).sum
  (total_spent_more : Rat) / (total_surveyed - 1) = 70 / 99 := by
  sorry

end NUMINAMATH_CALUDE_reusable_bags_estimate_conditional_probability_second_spender_l3436_343620


namespace NUMINAMATH_CALUDE_f_increasing_interval_f_greater_than_linear_l3436_343655

noncomputable section

def f (x : ℝ) := Real.log x - (1/2) * (x - 1)^2

theorem f_increasing_interval (x : ℝ) (hx : x > 1) :
  ∃ (a b : ℝ), a = 1 ∧ b = Real.sqrt 2 ∧
  ∀ y z, a < y ∧ y < z ∧ z < b → f y < f z :=
sorry

theorem f_greater_than_linear (k : ℝ) :
  (∃ x₀ > 1, ∀ x, 1 < x ∧ x < x₀ → f x > k * (x - 1)) ↔ k < 1 :=
sorry

end NUMINAMATH_CALUDE_f_increasing_interval_f_greater_than_linear_l3436_343655


namespace NUMINAMATH_CALUDE_notebooks_in_class2_l3436_343621

/-- The number of notebooks that do not belong to Class 1 -/
def not_class1 : ℕ := 162

/-- The number of notebooks that do not belong to Class 2 -/
def not_class2 : ℕ := 143

/-- The number of notebooks that belong to both Class 1 and Class 2 -/
def both_classes : ℕ := 87

/-- The total number of notebooks -/
def total_notebooks : ℕ := not_class1 + not_class2 - both_classes

theorem notebooks_in_class2 : 
  total_notebooks - (total_notebooks - not_class2) = 53 := by
  sorry

end NUMINAMATH_CALUDE_notebooks_in_class2_l3436_343621


namespace NUMINAMATH_CALUDE_triangle_side_length_l3436_343664

theorem triangle_side_length (a b c : ℝ) (A : ℝ) :
  a = Real.sqrt 2 →
  b = Real.sqrt 6 →
  A = π / 6 →
  c^2 - 2 * Real.sqrt 6 * c * Real.cos A + 2 = 6 →
  c = 2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3436_343664


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_area_l3436_343602

/-- Represents an isosceles trapezoid -/
structure IsoscelesTrapezoid where
  leg_length : ℝ
  diagonal_length : ℝ
  longer_base : ℝ

/-- Calculates the area of an isosceles trapezoid -/
def area (t : IsoscelesTrapezoid) : ℝ :=
  sorry

/-- Theorem stating the area of the specific isosceles trapezoid -/
theorem isosceles_trapezoid_area :
  let t : IsoscelesTrapezoid := {
    leg_length := 25,
    diagonal_length := 34,
    longer_base := 40
  }
  ∃ ε > 0, |area t - 569.275| < ε :=
sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_area_l3436_343602


namespace NUMINAMATH_CALUDE_xyz_value_l3436_343607

theorem xyz_value (a b c x y z : ℂ)
  (eq1 : a = (b + c) / (x - 2))
  (eq2 : b = (c + a) / (y - 2))
  (eq3 : c = (a + b) / (z - 2))
  (sum_prod : x * y + y * z + x * z = 67)
  (sum : x + y + z = 2010) :
  x * y * z = -5892 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l3436_343607


namespace NUMINAMATH_CALUDE_embankment_height_bounds_l3436_343681

/-- Represents the properties of a trapezoidal embankment -/
structure Embankment where
  length : ℝ
  lower_base : ℝ
  slope_angle : ℝ
  volume_min : ℝ
  volume_max : ℝ

/-- Theorem stating the height bounds for the embankment -/
theorem embankment_height_bounds (e : Embankment)
  (h_length : e.length = 100)
  (h_lower_base : e.lower_base = 5)
  (h_slope_angle : e.slope_angle = π/4)
  (h_volume : e.volume_min = 400 ∧ e.volume_max = 500)
  (h_upper_base_min : ∀ b, b ≥ 2 → 
    400 ≤ 25 * (5^2 - b^2) ∧ 25 * (5^2 - b^2) ≤ 500) :
  ∃ (h : ℝ), 1 ≤ h ∧ h ≤ (5 - Real.sqrt 5) / 2 :=
by sorry

end NUMINAMATH_CALUDE_embankment_height_bounds_l3436_343681


namespace NUMINAMATH_CALUDE_nested_fraction_evaluation_l3436_343696

theorem nested_fraction_evaluation :
  1 / (1 + 1 / (4 + 1 / 5)) = 21 / 26 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_evaluation_l3436_343696


namespace NUMINAMATH_CALUDE_line_through_parabola_vertex_l3436_343632

theorem line_through_parabola_vertex (a : ℝ) : 
  let line := fun x => x + a
  let parabola := fun x => x^2 + a^2
  let vertex_x := 0
  let vertex_y := parabola vertex_x
  (∃! (a1 a2 : ℝ), a1 ≠ a2 ∧ 
    line vertex_x = vertex_y ∧ 
    ∀ a', line vertex_x = vertex_y → (a' = a1 ∨ a' = a2)) := by
  sorry

end NUMINAMATH_CALUDE_line_through_parabola_vertex_l3436_343632


namespace NUMINAMATH_CALUDE_only_statement3_correct_l3436_343654

-- Define even and odd functions
def EvenFunction (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def OddFunction (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define the statements
def Statement1 : Prop := ∀ f : ℝ → ℝ, EvenFunction f → ∃ y, f 0 = y
def Statement2 : Prop := ∀ f : ℝ → ℝ, OddFunction f → f 0 = 0
def Statement3 : Prop := ∀ f : ℝ → ℝ, EvenFunction f → ∀ x, f x = f (-x)
def Statement4 : Prop := ∀ f : ℝ → ℝ, (EvenFunction f ∧ OddFunction f) → ∀ x, f x = 0

-- Theorem stating that only Statement3 is correct
theorem only_statement3_correct :
  ¬Statement1 ∧ ¬Statement2 ∧ Statement3 ∧ ¬Statement4 :=
sorry

end NUMINAMATH_CALUDE_only_statement3_correct_l3436_343654


namespace NUMINAMATH_CALUDE_a_value_l3436_343642

-- Define the system of inequalities
def system (x a : ℝ) : Prop :=
  3 * x + a < 0 ∧ 2 * x + 7 > 4 * x - 1

-- Define the solution set
def solution_set (x : ℝ) : Prop := x < 0

-- Theorem statement
theorem a_value (a : ℝ) :
  (∀ x, system x a ↔ solution_set x) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_a_value_l3436_343642


namespace NUMINAMATH_CALUDE_distribute_5_3_l3436_343634

/-- The number of ways to distribute n distinguishable objects into k distinguishable containers -/
def distribute (n k : ℕ) : ℕ := k^n

/-- Theorem: There are 243 ways to distribute 5 distinguishable balls into 3 distinguishable boxes -/
theorem distribute_5_3 : distribute 5 3 = 243 := by
  sorry

end NUMINAMATH_CALUDE_distribute_5_3_l3436_343634


namespace NUMINAMATH_CALUDE_order_of_cube_roots_l3436_343670

theorem order_of_cube_roots : ∀ (a b c : ℝ),
  a = 2^(4/3) →
  b = 3^(2/3) →
  c = 2.5^(1/3) →
  c < b ∧ b < a :=
by sorry

end NUMINAMATH_CALUDE_order_of_cube_roots_l3436_343670


namespace NUMINAMATH_CALUDE_division_theorem_l3436_343616

theorem division_theorem (dividend divisor remainder quotient : ℕ) : 
  dividend = 125 →
  divisor = 15 →
  remainder = 5 →
  quotient = (dividend - remainder) / divisor →
  quotient = 8 := by
sorry

end NUMINAMATH_CALUDE_division_theorem_l3436_343616


namespace NUMINAMATH_CALUDE_problem_statement_l3436_343613

theorem problem_statement (A B : ℝ) : 
  A^2 = 0.012345678987654321 * (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 8 + 7 + 6 + 5 + 4 + 3 + 2 + 1) →
  B^2 = 0.012345679 →
  9 * 10^9 * (1 - |A|) * B = 1 ∨ 9 * 10^9 * (1 - |A|) * B = -1 :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l3436_343613


namespace NUMINAMATH_CALUDE_binary_101101_equals_base5_140_l3436_343699

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

def decimal_to_base5 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) :=
      if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
    aux n []

theorem binary_101101_equals_base5_140 :
  decimal_to_base5 (binary_to_decimal [true, false, true, true, false, true]) = [1, 4, 0] :=
by sorry

end NUMINAMATH_CALUDE_binary_101101_equals_base5_140_l3436_343699


namespace NUMINAMATH_CALUDE_expression_undefined_l3436_343651

theorem expression_undefined (a : ℝ) : 
  ¬∃x, x = (a + 3) / (a^2 - 9*a + 20) ↔ a = 4 ∨ a = 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_undefined_l3436_343651


namespace NUMINAMATH_CALUDE_solution_f_greater_than_three_range_of_m_for_f_geq_g_l3436_343683

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 2|
def g (m : ℝ) (x : ℝ) : ℝ := m * |x| - 2

-- Theorem for the solution of f(x) > 3
theorem solution_f_greater_than_three :
  ∀ x : ℝ, f x > 3 ↔ x < -1 ∨ x > 5 := by sorry

-- Theorem for the range of m where f(x) ≥ g(x) for all x
theorem range_of_m_for_f_geq_g :
  ∀ m : ℝ, (∀ x : ℝ, f x ≥ g m x) ↔ m ≤ 1 := by sorry

end NUMINAMATH_CALUDE_solution_f_greater_than_three_range_of_m_for_f_geq_g_l3436_343683


namespace NUMINAMATH_CALUDE_complex_real_condition_l3436_343674

theorem complex_real_condition (a : ℝ) : 
  let z : ℂ := (a + Complex.I) / (2 - Complex.I)
  (∃ (x : ℝ), z = x) → a = -2 := by
sorry

end NUMINAMATH_CALUDE_complex_real_condition_l3436_343674


namespace NUMINAMATH_CALUDE_mans_age_double_sons_l3436_343660

/-- Represents the age difference between a man and his son -/
def age_difference : ℕ := 32

/-- Represents the current age of the son -/
def son_age : ℕ := 30

/-- Represents the number of years until the man's age is twice his son's age -/
def years_until_double : ℕ := 2

/-- Theorem stating that in 'years_until_double' years, the man's age will be twice his son's age -/
theorem mans_age_double_sons (y : ℕ) :
  y = years_until_double ↔ 
  (son_age + age_difference + y = 2 * (son_age + y)) :=
sorry

end NUMINAMATH_CALUDE_mans_age_double_sons_l3436_343660


namespace NUMINAMATH_CALUDE_plain_cookies_sold_l3436_343673

-- Define the types for our variables
def chocolate_chip_price : ℚ := 125 / 100
def plain_price : ℚ := 75 / 100
def total_boxes : ℕ := 1585
def total_value : ℚ := 158625 / 100

-- Define the theorem
theorem plain_cookies_sold :
  ∃ (c p : ℕ),
    c + p = total_boxes ∧
    c * chocolate_chip_price + p * plain_price = total_value ∧
    p = 790 := by
  sorry


end NUMINAMATH_CALUDE_plain_cookies_sold_l3436_343673


namespace NUMINAMATH_CALUDE_volleyball_teams_l3436_343652

theorem volleyball_teams (total_people : ℕ) (people_per_team : ℕ) (h1 : total_people = 6) (h2 : people_per_team = 2) :
  total_people / people_per_team = 3 := by
sorry

end NUMINAMATH_CALUDE_volleyball_teams_l3436_343652


namespace NUMINAMATH_CALUDE_completing_square_sum_l3436_343615

theorem completing_square_sum (d e f : ℤ) : 
  (100 : ℤ) * (x : ℚ)^2 + 60 * x - 90 = 0 ↔ (d * x + e)^2 = f →
  d > 0 →
  d + e + f = 112 :=
by sorry

end NUMINAMATH_CALUDE_completing_square_sum_l3436_343615


namespace NUMINAMATH_CALUDE_walnut_trees_planted_l3436_343639

/-- The number of walnut trees planted in a park --/
theorem walnut_trees_planted (initial final planted : ℕ) :
  initial = 22 →
  final = 55 →
  planted = final - initial →
  planted = 33 := by sorry

end NUMINAMATH_CALUDE_walnut_trees_planted_l3436_343639


namespace NUMINAMATH_CALUDE_initial_lives_calculation_l3436_343614

/-- Proves that the initial number of lives equals the current number of lives plus the number of lives lost -/
theorem initial_lives_calculation (current_lives lost_lives : ℕ) 
  (h1 : current_lives = 70) 
  (h2 : lost_lives = 13) : 
  current_lives + lost_lives = 83 := by
  sorry

end NUMINAMATH_CALUDE_initial_lives_calculation_l3436_343614


namespace NUMINAMATH_CALUDE_master_bathroom_towel_price_l3436_343633

/-- The price of towel sets for the master bathroom, given the following conditions:
  * 2 sets of towels for guest bathroom and 4 sets for master bathroom are bought
  * Guest bathroom towel sets cost $40.00 each
  * The store offers a 20% discount
  * The total spent on towel sets is $224
-/
theorem master_bathroom_towel_price :
  ∀ (x : ℝ),
    2 * 40 * (1 - 0.2) + 4 * x * (1 - 0.2) = 224 →
    x = 50 := by
  sorry

end NUMINAMATH_CALUDE_master_bathroom_towel_price_l3436_343633


namespace NUMINAMATH_CALUDE_find_m_l3436_343605

theorem find_m : ∃ m : ℝ, (∀ x : ℝ, x - m > 5 ↔ x > 2) → m = -3 := by sorry

end NUMINAMATH_CALUDE_find_m_l3436_343605


namespace NUMINAMATH_CALUDE_ba_atomic_weight_l3436_343669

/-- The atomic weight of Bromine (Br) -/
def atomic_weight_Br : ℝ := 79.9

/-- The molecular weight of the compound BaBr₂ -/
def molecular_weight_compound : ℝ := 297

/-- The atomic weight of Barium (Ba) -/
def atomic_weight_Ba : ℝ := molecular_weight_compound - 2 * atomic_weight_Br

theorem ba_atomic_weight :
  atomic_weight_Ba = 137.2 := by sorry

end NUMINAMATH_CALUDE_ba_atomic_weight_l3436_343669


namespace NUMINAMATH_CALUDE_x_power_five_minus_reciprocal_l3436_343667

theorem x_power_five_minus_reciprocal (x : ℝ) (h : x + 1/x = Real.sqrt 2) :
  x^5 - 1/x^5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_x_power_five_minus_reciprocal_l3436_343667


namespace NUMINAMATH_CALUDE_ptolemys_inequality_ptolemys_inequality_equality_l3436_343653

/-- Ptolemy's inequality in the complex plane -/
theorem ptolemys_inequality (a b c d : ℂ) :
  Complex.abs c * Complex.abs (d - b) ≤ 
  Complex.abs d * Complex.abs (c - b) + Complex.abs b * Complex.abs (c - d) :=
sorry

/-- Condition for equality in Ptolemy's inequality -/
def concyclic_or_collinear (a b c d : ℂ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ (b - a) * (d - c) = k * (c - b) * (d - a)

/-- Ptolemy's inequality with equality condition -/
theorem ptolemys_inequality_equality (a b c d : ℂ) :
  Complex.abs c * Complex.abs (d - b) = 
  Complex.abs d * Complex.abs (c - b) + Complex.abs b * Complex.abs (c - d) ↔
  concyclic_or_collinear a b c d :=
sorry

end NUMINAMATH_CALUDE_ptolemys_inequality_ptolemys_inequality_equality_l3436_343653


namespace NUMINAMATH_CALUDE_integer_divisibility_l3436_343617

theorem integer_divisibility (m : ℕ) : 
  Prime m → 
  ∃ k : ℕ+, m = 13 * k + 1 → 
  m ≠ 8191 → 
  ∃ n : ℤ, (2^(m-1) - 1) = 8191 * m * n :=
sorry

end NUMINAMATH_CALUDE_integer_divisibility_l3436_343617


namespace NUMINAMATH_CALUDE_solve_equation_l3436_343675

theorem solve_equation :
  ∃! y : ℚ, 2 * y + 3 * y = 200 - (4 * y + 10 * y / 2) ∧ y = 100 / 7 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3436_343675


namespace NUMINAMATH_CALUDE_riza_age_proof_l3436_343609

/-- Represents Riza's age when her son was born -/
def riza_age_at_birth : ℕ := 25

/-- Represents the current age of Riza's son -/
def son_current_age : ℕ := 40

/-- Represents the sum of Riza's and her son's current ages -/
def sum_of_ages : ℕ := 105

theorem riza_age_proof : 
  riza_age_at_birth + son_current_age + son_current_age = sum_of_ages := by
  sorry

end NUMINAMATH_CALUDE_riza_age_proof_l3436_343609


namespace NUMINAMATH_CALUDE_august_mail_total_l3436_343643

/-- The number of pieces of mail Vivian sent in a given month -/
def mail_sent (month : String) : ℕ :=
  match month with
  | "April" => 5
  | "May" => 10
  | "June" => 20
  | "July" => 40
  | _ => 0

/-- The number of business days in August -/
def august_business_days : ℕ := 23

/-- The number of holidays in August -/
def august_holidays : ℕ := 8

/-- The amount of mail sent on a business day in August -/
def august_business_day_mail : ℕ := 2 * mail_sent "July"

/-- The amount of mail sent on a holiday in August -/
def august_holiday_mail : ℕ := mail_sent "July" / 2

theorem august_mail_total :
  august_business_days * august_business_day_mail +
  august_holidays * august_holiday_mail = 2000 := by
  sorry

end NUMINAMATH_CALUDE_august_mail_total_l3436_343643


namespace NUMINAMATH_CALUDE_expression_value_l3436_343627

theorem expression_value : 
  let x : ℝ := 4
  let y : ℝ := -3
  let z : ℝ := 5
  x^2 + y^2 - z^2 + 2*y*z = -30 :=
by sorry

end NUMINAMATH_CALUDE_expression_value_l3436_343627


namespace NUMINAMATH_CALUDE_xy_value_l3436_343662

theorem xy_value (x y : ℝ) (h1 : x + y = 10) (h2 : x^3 + y^3 = 370) : x * y = 21 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l3436_343662


namespace NUMINAMATH_CALUDE_height_difference_l3436_343679

/-- Given three people A, B, and C, where A's height is 30% less than B's,
    and C's height is 20% more than A's, prove that the percentage difference
    between B's height and C's height is 16%. -/
theorem height_difference (h_b : ℝ) (h_b_pos : h_b > 0) : 
  let h_a := 0.7 * h_b
  let h_c := 1.2 * h_a
  ((h_b - h_c) / h_b) * 100 = 16 := by sorry

end NUMINAMATH_CALUDE_height_difference_l3436_343679


namespace NUMINAMATH_CALUDE_josie_remaining_money_l3436_343661

/-- Calculates the remaining amount after purchases -/
def remaining_amount (initial_amount : ℕ) (item1_cost : ℕ) (item1_quantity : ℕ) (item2_cost : ℕ) : ℕ :=
  initial_amount - (item1_cost * item1_quantity + item2_cost)

/-- Proves that given the specific initial amount and purchase costs, the remaining amount is correct -/
theorem josie_remaining_money :
  remaining_amount 50 9 2 25 = 7 := by
  sorry

end NUMINAMATH_CALUDE_josie_remaining_money_l3436_343661


namespace NUMINAMATH_CALUDE_return_trip_time_l3436_343618

/-- Represents the flight scenario between two cities -/
structure FlightScenario where
  d : ℝ  -- distance between cities
  p : ℝ  -- speed of plane in still air
  w : ℝ  -- speed of wind
  time_against_wind : ℝ  -- time taken against wind
  time_diff : ℝ  -- time difference between still air and with wind

/-- The main theorem about the return trip time -/
theorem return_trip_time (fs : FlightScenario) 
  (h1 : fs.time_against_wind = 90)
  (h2 : fs.d = fs.time_against_wind * (fs.p - fs.w))
  (h3 : fs.d / (fs.p + fs.w) = fs.d / fs.p - fs.time_diff)
  (h4 : fs.time_diff = 12) :
  fs.d / (fs.p + fs.w) = 18 ∨ fs.d / (fs.p + fs.w) = 60 := by
  sorry

end NUMINAMATH_CALUDE_return_trip_time_l3436_343618


namespace NUMINAMATH_CALUDE_symmetry_of_expressions_l3436_343636

-- Define a completely symmetric expression
def is_completely_symmetric (f : ℝ → ℝ → ℝ → ℝ) : Prop :=
  ∀ (a b c : ℝ), f a b c = f b a c ∧ f a b c = f a c b ∧ f a b c = f c b a

-- Define the three expressions
def expr1 (a b c : ℝ) : ℝ := (a - b)^2
def expr2 (a b c : ℝ) : ℝ := a * b + b * c + c * a
def expr3 (a b c : ℝ) : ℝ := a^2 * b + b^2 * c + c^2 * a

-- State the theorem
theorem symmetry_of_expressions :
  is_completely_symmetric expr1 ∧ 
  is_completely_symmetric expr2 ∧ 
  ¬is_completely_symmetric expr3 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_of_expressions_l3436_343636


namespace NUMINAMATH_CALUDE_no_divisibility_l3436_343626

theorem no_divisibility (d a n : ℕ) (h1 : 3 ≤ d) (h2 : d ≤ 2^(n+1)) :
  ¬(d ∣ a^(2^n) + 1) := by
  sorry

end NUMINAMATH_CALUDE_no_divisibility_l3436_343626


namespace NUMINAMATH_CALUDE_prob_rain_all_days_l3436_343604

def prob_rain_friday : ℚ := 40 / 100
def prob_rain_saturday : ℚ := 50 / 100
def prob_rain_sunday : ℚ := 30 / 100

def events_independent : Prop := True

theorem prob_rain_all_days : 
  prob_rain_friday * prob_rain_saturday * prob_rain_sunday = 6 / 100 :=
by sorry

end NUMINAMATH_CALUDE_prob_rain_all_days_l3436_343604


namespace NUMINAMATH_CALUDE_derivative_at_one_implies_a_value_l3436_343623

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x^2 - a) * log x

theorem derivative_at_one_implies_a_value (a : ℝ) :
  (deriv (f a)) 1 = -2 → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_derivative_at_one_implies_a_value_l3436_343623


namespace NUMINAMATH_CALUDE_max_sum_of_digits_l3436_343659

/-- A function that checks if a number is a digit (0-9) -/
def isDigit (n : ℕ) : Prop := n ≤ 9

/-- A function that checks if four numbers are distinct -/
def areDistinct (a b c d : ℕ) : Prop := a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

theorem max_sum_of_digits (A B C D : ℕ) :
  isDigit A → isDigit B → isDigit C → isDigit D →
  areDistinct A B C D →
  A + B + C + D = 17 →
  (A + B) % (C + D) = 0 →
  A + B ≤ 16 ∧ ∃ (A' B' C' D' : ℕ), 
    isDigit A' ∧ isDigit B' ∧ isDigit C' ∧ isDigit D' ∧
    areDistinct A' B' C' D' ∧
    A' + B' + C' + D' = 17 ∧
    (A' + B') % (C' + D') = 0 ∧
    A' + B' = 16 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_digits_l3436_343659


namespace NUMINAMATH_CALUDE_root_implies_q_value_l3436_343640

theorem root_implies_q_value (p q : ℝ) : 
  (Complex.I : ℂ).re = 0 ∧ (Complex.I : ℂ).im = 1 →
  (2 : ℂ) * (3 + 2 * Complex.I)^2 + p * (3 + 2 * Complex.I) + q = 0 →
  q = 26 := by
sorry

end NUMINAMATH_CALUDE_root_implies_q_value_l3436_343640


namespace NUMINAMATH_CALUDE_cube_space_division_theorem_l3436_343665

/-- The number of parts that space is divided into by the planes containing the faces of a cube -/
def cube_space_division : ℕ := 33

/-- The number of faces a cube has -/
def cube_faces : ℕ := 6

/-- Theorem stating that the planes containing the faces of a cube divide space into 33 parts -/
theorem cube_space_division_theorem :
  cube_space_division = 33 ∧ cube_faces = 6 :=
sorry

end NUMINAMATH_CALUDE_cube_space_division_theorem_l3436_343665


namespace NUMINAMATH_CALUDE_ae_bc_ratio_l3436_343656

-- Define the points
variable (A B C D E : ℝ × ℝ)

-- Define the triangles
def is_equilateral (X Y Z : ℝ × ℝ) : Prop :=
  dist X Y = dist Y Z ∧ dist Y Z = dist Z X

-- Define the midpoint
def is_midpoint (M X Y : ℝ × ℝ) : Prop :=
  M = ((X.1 + Y.1) / 2, (X.2 + Y.2) / 2)

-- State the theorem
theorem ae_bc_ratio (A B C D E : ℝ × ℝ) :
  is_equilateral A B C →
  is_equilateral B C D →
  is_equilateral C D E →
  is_midpoint E C D →
  dist A E / dist B C = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ae_bc_ratio_l3436_343656


namespace NUMINAMATH_CALUDE_common_point_in_intervals_l3436_343663

theorem common_point_in_intervals (n : ℕ) (a b : Fin n → ℝ) 
  (h_closed : ∀ i, a i ≤ b i) 
  (h_intersect : ∀ i j, ∃ x, a i ≤ x ∧ x ≤ b i ∧ a j ≤ x ∧ x ≤ b j) : 
  ∃ p, ∀ i, a i ≤ p ∧ p ≤ b i :=
sorry

end NUMINAMATH_CALUDE_common_point_in_intervals_l3436_343663


namespace NUMINAMATH_CALUDE_complex_set_characterization_l3436_343658

theorem complex_set_characterization (z : ℂ) :
  (z - 1)^2 = Complex.abs (z - 1)^2 ↔ z.im = 0 :=
sorry

end NUMINAMATH_CALUDE_complex_set_characterization_l3436_343658


namespace NUMINAMATH_CALUDE_farmer_land_usage_l3436_343612

/-- Represents the ratio of land used for beans, wheat, and corn -/
def land_ratio : Fin 3 → ℕ
  | 0 => 5  -- beans
  | 1 => 2  -- wheat
  | 2 => 4  -- corn
  | _ => 0  -- unreachable

/-- The total parts in the ratio -/
def total_parts : ℕ := (land_ratio 0) + (land_ratio 1) + (land_ratio 2)

/-- The number of acres used for corn -/
def corn_acres : ℕ := 376

theorem farmer_land_usage :
  let total_acres := (total_parts * corn_acres) / (land_ratio 2)
  total_acres = 1034 := by sorry

end NUMINAMATH_CALUDE_farmer_land_usage_l3436_343612


namespace NUMINAMATH_CALUDE_potato_planting_l3436_343638

theorem potato_planting (rows : ℕ) (additional_plants : ℕ) (total_plants : ℕ) 
  (h1 : rows = 7)
  (h2 : additional_plants = 15)
  (h3 : total_plants = 141)
  : (total_plants - additional_plants) / rows = 18 := by
  sorry

end NUMINAMATH_CALUDE_potato_planting_l3436_343638


namespace NUMINAMATH_CALUDE_worker_usual_time_l3436_343677

/-- The usual time for a worker to reach her office, given slower speed conditions -/
theorem worker_usual_time : ∃ (T : ℝ), T = 24 ∧ T > 0 := by
  -- Let T be the usual time in minutes
  -- Let S be the usual speed in distance per minute
  -- When walking at 3/4 speed, the new time is (T + 8) minutes
  -- The distance remains constant: S * T = (3/4 * S) * (T + 8)
  sorry


end NUMINAMATH_CALUDE_worker_usual_time_l3436_343677


namespace NUMINAMATH_CALUDE_consecutive_numbers_theorem_l3436_343631

theorem consecutive_numbers_theorem (n : ℕ) (avg : ℚ) (largest : ℕ) : 
  n > 0 ∧ 
  avg = 20 ∧ 
  largest = 23 ∧ 
  (↑largest - ↑(n - 1) + ↑largest) / 2 = avg → 
  n = 7 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_numbers_theorem_l3436_343631


namespace NUMINAMATH_CALUDE_wheel_radius_l3436_343668

theorem wheel_radius (total_distance : ℝ) (revolutions : ℕ) (h1 : total_distance = 798.2857142857142) (h2 : revolutions = 500) :
  ∃ (radius : ℝ), abs (radius - 0.254092376554174) < 0.000000000000001 :=
by
  sorry

end NUMINAMATH_CALUDE_wheel_radius_l3436_343668


namespace NUMINAMATH_CALUDE_income_ratio_l3436_343698

/-- Proof of the ratio of monthly incomes --/
theorem income_ratio (c_income b_income a_annual_income : ℝ) 
  (hb : b_income = c_income * 1.12)
  (hc : c_income = 12000)
  (ha : a_annual_income = 403200.0000000001) :
  (a_annual_income / 12) / b_income = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_income_ratio_l3436_343698


namespace NUMINAMATH_CALUDE_ivan_milkshake_cost_l3436_343687

/-- The cost of Ivan's milkshake -/
def milkshake_cost (initial_amount : ℚ) (cupcake_fraction : ℚ) (final_amount : ℚ) : ℚ :=
  initial_amount - initial_amount * cupcake_fraction - final_amount

/-- Theorem: The cost of Ivan's milkshake is $5 -/
theorem ivan_milkshake_cost :
  milkshake_cost 10 (1/5) 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ivan_milkshake_cost_l3436_343687


namespace NUMINAMATH_CALUDE_gretzky_street_length_proof_l3436_343606

/-- The length of Gretzky Street in kilometers -/
def gretzky_street_length : ℝ := 5.95

/-- The number of numbered intersecting streets -/
def num_intersections : ℕ := 15

/-- The distance between each intersecting street in meters -/
def intersection_distance : ℝ := 350

/-- The number of additional segments at the beginning and end -/
def additional_segments : ℕ := 2

/-- Theorem stating that the length of Gretzky Street is 5.95 kilometers -/
theorem gretzky_street_length_proof :
  gretzky_street_length = 
    (intersection_distance * (num_intersections + additional_segments)) / 1000 := by
  sorry

end NUMINAMATH_CALUDE_gretzky_street_length_proof_l3436_343606


namespace NUMINAMATH_CALUDE_exactly_one_positive_integer_solution_l3436_343649

theorem exactly_one_positive_integer_solution : 
  ∃! (n : ℕ), n > 0 ∧ 16 - 4 * n > 10 :=
by sorry

end NUMINAMATH_CALUDE_exactly_one_positive_integer_solution_l3436_343649


namespace NUMINAMATH_CALUDE_correct_stratified_sample_l3436_343628

/-- Represents the staff categories in the unit -/
inductive StaffCategory
  | Business
  | Management
  | Logistics

/-- Represents the staff distribution in the unit -/
structure StaffDistribution where
  total : ℕ
  business : ℕ
  management : ℕ
  logistics : ℕ
  sum_eq_total : business + management + logistics = total

/-- Represents the sample size and distribution -/
structure Sample where
  size : ℕ
  business : ℕ
  management : ℕ
  logistics : ℕ
  sum_eq_size : business + management + logistics = size

/-- Checks if a sample is proportionally correct for a given staff distribution -/
def is_proportional_sample (staff : StaffDistribution) (sample : Sample) : Prop :=
  staff.business * sample.size = sample.business * staff.total ∧
  staff.management * sample.size = sample.management * staff.total ∧
  staff.logistics * sample.size = sample.logistics * staff.total

/-- Theorem: The given sample is proportionally correct for the given staff distribution -/
theorem correct_stratified_sample :
  let staff : StaffDistribution := ⟨160, 112, 16, 32, rfl⟩
  let sample : Sample := ⟨20, 14, 2, 4, rfl⟩
  is_proportional_sample staff sample := by sorry


end NUMINAMATH_CALUDE_correct_stratified_sample_l3436_343628


namespace NUMINAMATH_CALUDE_largest_prime_divisor_of_sum_of_squares_l3436_343657

theorem largest_prime_divisor_of_sum_of_squares : 
  ∃ p : ℕ, 
    Nat.Prime p ∧ 
    p ∣ (36^2 + 45^2) ∧ 
    ∀ q : ℕ, Nat.Prime q → q ∣ (36^2 + 45^2) → q ≤ p :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_of_sum_of_squares_l3436_343657


namespace NUMINAMATH_CALUDE_solution_set_l3436_343622

theorem solution_set (x : ℝ) : 3 * x^2 + 9 * x + 6 ≤ 0 ∧ x + 2 > 0 → x ∈ Set.Ioo (-2) (-1) ∪ {-1} :=
by sorry

end NUMINAMATH_CALUDE_solution_set_l3436_343622


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3436_343697

/-- Given a parabola and a hyperbola with specific properties, prove that the eccentricity of the hyperbola is 1 + √2 -/
theorem hyperbola_eccentricity 
  (p a b : ℝ) 
  (hp : p > 0) 
  (ha : a > 0) 
  (hb : b > 0) 
  (parabola : ℝ → ℝ → Prop) 
  (hyperbola : ℝ → ℝ → Prop) 
  (parabola_eq : ∀ x y, parabola x y ↔ y^2 = 2*p*x) 
  (hyperbola_eq : ∀ x y, hyperbola x y ↔ x^2/a^2 - y^2/b^2 = 1) 
  (focus_shared : ∃ F : ℝ × ℝ, F.1 = p/2 ∧ F.2 = 0 ∧ 
    F.1^2/a^2 + F.2^2/b^2 = (a^2 + b^2)/a^2) 
  (intersection_line : ∃ I₁ I₂ : ℝ × ℝ, 
    parabola I₁.1 I₁.2 ∧ hyperbola I₁.1 I₁.2 ∧ 
    parabola I₂.1 I₂.2 ∧ hyperbola I₂.1 I₂.2 ∧ 
    (I₂.2 - I₁.2) * (p/2 - I₁.1) = (I₂.1 - I₁.1) * (0 - I₁.2)) :
  (a^2 + b^2)/a^2 = (1 + Real.sqrt 2)^2 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3436_343697


namespace NUMINAMATH_CALUDE_total_investment_equals_eight_thousand_l3436_343630

/-- Represents an investment account with a given balance and interest rate. -/
structure Account where
  balance : ℝ
  interestRate : ℝ

/-- Calculates the total investment given two accounts. -/
def totalInvestment (account1 account2 : Account) : ℝ :=
  account1.balance + account2.balance

/-- Theorem: The total investment in two accounts with $4,000 each is $8,000. -/
theorem total_investment_equals_eight_thousand 
  (account1 account2 : Account)
  (h1 : account1.balance = 4000)
  (h2 : account2.balance = 4000) :
  totalInvestment account1 account2 = 8000 := by
  sorry

#check total_investment_equals_eight_thousand

end NUMINAMATH_CALUDE_total_investment_equals_eight_thousand_l3436_343630


namespace NUMINAMATH_CALUDE_range_of_x_when_proposition_false_l3436_343629

theorem range_of_x_when_proposition_false (x : ℝ) :
  x^2 - 5*x + 4 ≤ 0 → 1 ≤ x ∧ x ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_when_proposition_false_l3436_343629


namespace NUMINAMATH_CALUDE_job_completion_time_l3436_343688

theorem job_completion_time (days : ℝ) (fraction_completed : ℝ) (h1 : fraction_completed = 5 / 8) (h2 : days = 10) :
  (days / fraction_completed) = 16 := by
  sorry

end NUMINAMATH_CALUDE_job_completion_time_l3436_343688


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_angle_l3436_343600

-- Define an isosceles triangle
structure IsoscelesTriangle where
  -- We'll use degrees for simplicity
  base_angle : ℝ
  vertex_angle : ℝ
  is_isosceles : base_angle * 2 + vertex_angle = 180

-- Define our specific isosceles triangle
def our_triangle (t : IsoscelesTriangle) : Prop :=
  t.base_angle = 50 ∧ t.vertex_angle = 80 ∨
  t.base_angle = 80 ∧ t.vertex_angle = 20

-- Theorem statement
theorem isosceles_triangle_base_angle :
  ∀ t : IsoscelesTriangle, (t.base_angle = 50 ∨ t.base_angle = 80) ↔ 
  (t.base_angle = 50 ∧ t.vertex_angle = 80 ∨ t.base_angle = 80 ∧ t.vertex_angle = 20) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_angle_l3436_343600


namespace NUMINAMATH_CALUDE_minimum_width_proof_l3436_343619

/-- Represents the width of the rectangular fence -/
def width : ℝ → ℝ := λ w => w

/-- Represents the length of the rectangular fence -/
def length : ℝ → ℝ := λ w => w + 20

/-- Represents the area of the rectangular fence -/
def area : ℝ → ℝ := λ w => width w * length w

/-- Represents the perimeter of the rectangular fence -/
def perimeter : ℝ → ℝ := λ w => 2 * (width w + length w)

/-- The minimum width of the rectangular fence that satisfies the given conditions -/
def min_width : ℝ := 10

theorem minimum_width_proof :
  (∀ w : ℝ, w > 0 → area w ≥ 200 → perimeter min_width ≤ perimeter w) ∧
  area min_width ≥ 200 := by sorry

end NUMINAMATH_CALUDE_minimum_width_proof_l3436_343619


namespace NUMINAMATH_CALUDE_largest_A_at_125_l3436_343676

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The sequence A_k -/
def A (k : ℕ) : ℝ := binomial 500 k * (0.3 ^ k)

theorem largest_A_at_125 : 
  ∀ k ∈ Finset.range 501, A 125 ≥ A k :=
sorry

end NUMINAMATH_CALUDE_largest_A_at_125_l3436_343676


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l3436_343641

theorem inequality_system_solution_set :
  ∀ x : ℝ, (2 - x > 0 ∧ 2*x + 3 > 1) ↔ (-1 < x ∧ x < 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l3436_343641


namespace NUMINAMATH_CALUDE_polygon_interior_angles_l3436_343608

theorem polygon_interior_angles (n : ℕ) : 
  (n ≥ 3) → 
  (2005 + 180 = (n - 2) * 180) → 
  n = 14 :=
by sorry

end NUMINAMATH_CALUDE_polygon_interior_angles_l3436_343608


namespace NUMINAMATH_CALUDE_triangle_angle_A_l3436_343611

theorem triangle_angle_A (a b : ℝ) (B : ℝ) (hA : 0 < a) (hB : 0 < b) (hab : a > b) 
  (ha : a = Real.sqrt 3) (hb : b = Real.sqrt 2) (hB : B = π / 4) :
  ∃ A : ℝ, (A = π / 3 ∨ A = 2 * π / 3) ∧ 
    Real.sin A = (a * Real.sin B) / b :=
sorry

end NUMINAMATH_CALUDE_triangle_angle_A_l3436_343611


namespace NUMINAMATH_CALUDE_shopkeeper_profit_percentage_l3436_343685

theorem shopkeeper_profit_percentage
  (cost_price : ℝ)
  (markup_percentage : ℝ)
  (discount : ℝ)
  (h_cp : cost_price = 180)
  (h_mp : markup_percentage = 45)
  (h_d : discount = 45) :
  let markup := cost_price * (markup_percentage / 100)
  let marked_price := cost_price + markup
  let selling_price := marked_price - discount
  let profit := selling_price - cost_price
  let profit_percentage := (profit / cost_price) * 100
  profit_percentage = 20 := by
sorry

end NUMINAMATH_CALUDE_shopkeeper_profit_percentage_l3436_343685


namespace NUMINAMATH_CALUDE_apples_found_l3436_343686

def initial_apples : ℕ := 7
def final_apples : ℕ := 81

theorem apples_found (found : ℕ) : found = final_apples - initial_apples := by
  sorry

end NUMINAMATH_CALUDE_apples_found_l3436_343686


namespace NUMINAMATH_CALUDE_expression_equality_l3436_343693

theorem expression_equality : 
  Real.sqrt 8 + |1 - Real.sqrt 2| - (1 / 2)⁻¹ + (π - Real.sqrt 3)^0 = 3 * Real.sqrt 2 - 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l3436_343693


namespace NUMINAMATH_CALUDE_ninth_grade_math_only_l3436_343624

theorem ninth_grade_math_only (total : ℕ) (math science foreign : ℕ) 
  (h_total : total = 120)
  (h_math : math = 85)
  (h_science : science = 70)
  (h_foreign : foreign = 54) :
  ∃ (math_science math_foreign science_foreign math_science_foreign : ℕ),
    math_science + math_foreign + science_foreign - math_science_foreign ≤ math ∧
    math_science + math_foreign + science_foreign - math_science_foreign ≤ science ∧
    math_science + math_foreign + science_foreign - math_science_foreign ≤ foreign ∧
    total = math + science + foreign - math_science - math_foreign - science_foreign + math_science_foreign ∧
    math - (math_science + math_foreign - math_science_foreign) = 45 := by
  sorry

end NUMINAMATH_CALUDE_ninth_grade_math_only_l3436_343624


namespace NUMINAMATH_CALUDE_solve_for_b_l3436_343671

theorem solve_for_b (b : ℚ) (h : 2 * b + b / 4 = 5 / 2) : b = 10 / 9 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_b_l3436_343671


namespace NUMINAMATH_CALUDE_max_remainder_when_divided_by_25_l3436_343647

theorem max_remainder_when_divided_by_25 (A B C : ℕ) : 
  A ≠ B ∧ B ≠ C ∧ A ≠ C →
  A = 25 * B + C →
  C ≤ 24 :=
by sorry

end NUMINAMATH_CALUDE_max_remainder_when_divided_by_25_l3436_343647


namespace NUMINAMATH_CALUDE_john_lap_time_improvement_l3436_343603

theorem john_lap_time_improvement :
  let initial_laps : ℚ := 15
  let initial_time : ℚ := 40
  let current_laps : ℚ := 18
  let current_time : ℚ := 36
  let initial_lap_time := initial_time / initial_laps
  let current_lap_time := current_time / current_laps
  let improvement := initial_lap_time - current_lap_time
  improvement = 2/3
:= by sorry

end NUMINAMATH_CALUDE_john_lap_time_improvement_l3436_343603


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3436_343690

theorem polynomial_factorization (x : ℝ) : 
  (x + 1) * (x + 2) * (x + 3) * (x + 4) + 1 = (x^2 + 5*x + 5)^2 := by sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3436_343690


namespace NUMINAMATH_CALUDE_carved_arc_angle_l3436_343625

/-- An equilateral triangle -/
structure EquilateralTriangle where
  height : ℝ
  height_pos : height > 0

/-- A circle rolling along the side of an equilateral triangle -/
structure RollingCircle (triangle : EquilateralTriangle) where
  radius : ℝ
  radius_eq_height : radius = triangle.height

/-- The arc carved out from the circle by the sides of the triangle -/
def carved_arc (triangle : EquilateralTriangle) (circle : RollingCircle triangle) : ℝ := sorry

/-- Theorem: The arc carved out from the circle subtends an angle of 60° at the center -/
theorem carved_arc_angle (triangle : EquilateralTriangle) (circle : RollingCircle triangle) :
  carved_arc triangle circle = 60 * π / 180 := by sorry

end NUMINAMATH_CALUDE_carved_arc_angle_l3436_343625


namespace NUMINAMATH_CALUDE_janet_extra_fica_tax_l3436_343678

/-- Represents Janet's employment situation -/
structure Employment where
  hours_per_week : ℕ
  current_hourly_rate : ℚ
  freelance_hourly_rate : ℚ
  healthcare_premium_per_month : ℚ
  additional_monthly_income_freelancing : ℚ

/-- Calculates the extra weekly FICA tax for freelancing -/
def extra_weekly_fica_tax (e : Employment) : ℚ :=
  let current_monthly_income := e.hours_per_week * e.current_hourly_rate * 4
  let freelance_monthly_income := e.hours_per_week * e.freelance_hourly_rate * 4
  let extra_monthly_income := freelance_monthly_income - current_monthly_income
  let extra_monthly_income_after_healthcare := extra_monthly_income - e.healthcare_premium_per_month
  (extra_monthly_income_after_healthcare - e.additional_monthly_income_freelancing) / 4

/-- Theorem stating that the extra weekly FICA tax for Janet's situation is $25 -/
theorem janet_extra_fica_tax :
  let janet : Employment := {
    hours_per_week := 40,
    current_hourly_rate := 30,
    freelance_hourly_rate := 40,
    healthcare_premium_per_month := 400,
    additional_monthly_income_freelancing := 1100
  }
  extra_weekly_fica_tax janet = 25 := by sorry

end NUMINAMATH_CALUDE_janet_extra_fica_tax_l3436_343678


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3436_343610

-- Define the sets A and B
def A : Set ℝ := {x | x > 0}
def B : Set ℝ := {x | -1 < x ∧ x < 3}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = Set.Ioo 0 3 := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3436_343610


namespace NUMINAMATH_CALUDE_notebooks_given_to_yujeong_l3436_343635

/-- The number of notebooks Minyoung initially had -/
def initial_notebooks : ℕ := 17

/-- The number of notebooks Minyoung had left after giving some to Yujeong -/
def remaining_notebooks : ℕ := 8

/-- The number of notebooks Minyoung gave to Yujeong -/
def notebooks_given : ℕ := initial_notebooks - remaining_notebooks

theorem notebooks_given_to_yujeong :
  notebooks_given = 9 :=
sorry

end NUMINAMATH_CALUDE_notebooks_given_to_yujeong_l3436_343635


namespace NUMINAMATH_CALUDE_hexagon_perimeter_l3436_343644

/-- The perimeter of a hexagon with side length 5 inches is 30 inches. -/
theorem hexagon_perimeter (side_length : ℝ) (h : side_length = 5) : 
  6 * side_length = 30 := by sorry

end NUMINAMATH_CALUDE_hexagon_perimeter_l3436_343644


namespace NUMINAMATH_CALUDE_game_cost_l3436_343684

/-- Given Frank's lawn mowing earnings, expenses, and game purchasing ability, prove the cost of each game. -/
theorem game_cost (total_earned : ℕ) (spent : ℕ) (num_games : ℕ) 
  (h1 : total_earned = 19)
  (h2 : spent = 11)
  (h3 : num_games = 4)
  (h4 : ∃ (cost : ℕ), (total_earned - spent) = num_games * cost) :
  ∃ (cost : ℕ), cost = 2 ∧ (total_earned - spent) = num_games * cost := by
  sorry

end NUMINAMATH_CALUDE_game_cost_l3436_343684


namespace NUMINAMATH_CALUDE_train_crossing_time_l3436_343637

/-- The time taken for a train to cross a man walking in the same direction -/
theorem train_crossing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) :
  train_length = 700 →
  train_speed = 63 * 1000 / 3600 →
  man_speed = 3 * 1000 / 3600 →
  (train_length / (train_speed - man_speed)) = 42 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l3436_343637


namespace NUMINAMATH_CALUDE_training_cost_calculation_l3436_343689

/-- Represents the financial details of a job applicant -/
structure Applicant where
  salary : ℝ
  revenue : ℝ
  trainingMonths : ℕ
  hiringBonus : ℝ

/-- Calculates the net gain for the company from an applicant -/
def netGain (a : Applicant) (trainingCostPerMonth : ℝ) : ℝ :=
  a.revenue - (a.salary + a.hiringBonus + a.trainingMonths * trainingCostPerMonth)

theorem training_cost_calculation (applicant1 applicant2 : Applicant) 
  (h1 : applicant1.salary = 42000)
  (h2 : applicant1.revenue = 93000)
  (h3 : applicant1.trainingMonths = 3)
  (h4 : applicant1.hiringBonus = 0)
  (h5 : applicant2.salary = 45000)
  (h6 : applicant2.revenue = 92000)
  (h7 : applicant2.trainingMonths = 0)
  (h8 : applicant2.hiringBonus = 0.01 * applicant2.salary)
  (h9 : ∃ (trainingCostPerMonth : ℝ), 
    netGain applicant1 trainingCostPerMonth - netGain applicant2 0 = 850 ∨
    netGain applicant2 0 - netGain applicant1 trainingCostPerMonth = 850) :
  ∃ (trainingCostPerMonth : ℝ), trainingCostPerMonth = 17866.67 := by
  sorry

end NUMINAMATH_CALUDE_training_cost_calculation_l3436_343689


namespace NUMINAMATH_CALUDE_initial_disappearance_percentage_l3436_343672

/-- Represents the population changes in a village --/
def village_population (initial_population : ℕ) (final_population : ℕ) (panic_exodus_percent : ℚ) : Prop :=
  ∃ (initial_disappearance_percent : ℚ),
    final_population = initial_population * (1 - initial_disappearance_percent / 100) * (1 - panic_exodus_percent / 100) ∧
    initial_disappearance_percent = 10

/-- Theorem stating the initial disappearance percentage in the village --/
theorem initial_disappearance_percentage :
  village_population 7800 5265 25 := by sorry

end NUMINAMATH_CALUDE_initial_disappearance_percentage_l3436_343672


namespace NUMINAMATH_CALUDE_largest_divisor_of_consecutive_odd_product_l3436_343691

theorem largest_divisor_of_consecutive_odd_product (n : ℕ) (h : Even n) (h' : n > 0) :
  ∃ (k : ℕ), k > 15 → ¬(∀ (m : ℕ), Even m → m > 0 →
    k ∣ (m + 1) * (m + 3) * (m + 5) * (m + 7) * (m + 9)) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_consecutive_odd_product_l3436_343691


namespace NUMINAMATH_CALUDE_det_A_plus_5_l3436_343645

def A : Matrix (Fin 2) (Fin 2) ℝ := !![6, -2; -3, 7]

theorem det_A_plus_5 : Matrix.det A + 5 = 41 := by
  sorry

end NUMINAMATH_CALUDE_det_A_plus_5_l3436_343645


namespace NUMINAMATH_CALUDE_percentage_subtraction_equivalence_l3436_343650

theorem percentage_subtraction_equivalence :
  ∀ (a : ℝ), a - (0.07 * a) = 0.93 * a :=
by
  sorry

end NUMINAMATH_CALUDE_percentage_subtraction_equivalence_l3436_343650


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3436_343692

-- Define the polynomial g(x)
def g (p q r s : ℝ) (x : ℂ) : ℂ := x^4 + p*x^3 + q*x^2 + r*x + s

-- State the theorem
theorem sum_of_coefficients (p q r s : ℝ) :
  (g p q r s (3*I) = 0) →
  (g p q r s (1 + 3*I) = 0) →
  p + q + r + s = 89 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3436_343692


namespace NUMINAMATH_CALUDE_solution_set_when_a_eq_2_range_of_a_l3436_343648

-- Define the functions f and g
def f (a x : ℝ) := |2 * x - a| + a
def g (x : ℝ) := |2 * x - 1|

-- Part 1
theorem solution_set_when_a_eq_2 :
  {x : ℝ | f 2 x ≤ 6} = {x : ℝ | -1 ≤ x ∧ x ≤ 3} :=
sorry

-- Part 2
theorem range_of_a :
  (∀ x : ℝ, f a x + g x ≥ 3) ↔ a ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_solution_set_when_a_eq_2_range_of_a_l3436_343648


namespace NUMINAMATH_CALUDE_circumscribed_trapezoid_radius_l3436_343646

/-- An isosceles trapezoid circumscribed around a circle -/
structure CircumscribedTrapezoid where
  /-- The area of the trapezoid -/
  area : ℝ
  /-- The height of the trapezoid -/
  height : ℝ
  /-- The lateral side of the trapezoid -/
  lateral_side : ℝ
  /-- The radius of the inscribed circle -/
  radius : ℝ
  /-- The height is half the lateral side -/
  height_half_lateral : height = lateral_side / 2
  /-- The area is positive -/
  area_pos : 0 < area

/-- 
  For an isosceles trapezoid circumscribed around a circle, 
  if the area of the trapezoid is S and its height is half of its lateral side, 
  then the radius of the circle is √(S/8).
-/
theorem circumscribed_trapezoid_radius 
  (t : CircumscribedTrapezoid) : t.radius = Real.sqrt (t.area / 8) := by
  sorry

end NUMINAMATH_CALUDE_circumscribed_trapezoid_radius_l3436_343646


namespace NUMINAMATH_CALUDE_minuend_not_integer_l3436_343601

theorem minuend_not_integer (M S : ℝ) : M + S + (M - S) = 555 → ¬(∃ n : ℤ, M = n) := by
  sorry

end NUMINAMATH_CALUDE_minuend_not_integer_l3436_343601


namespace NUMINAMATH_CALUDE_cab_speed_reduction_l3436_343666

theorem cab_speed_reduction (usual_time : ℝ) (delay : ℝ) :
  usual_time = 75 ∧ delay = 15 →
  (usual_time / (usual_time + delay)) = 5 / 6 := by
sorry

end NUMINAMATH_CALUDE_cab_speed_reduction_l3436_343666


namespace NUMINAMATH_CALUDE_fraction_problem_l3436_343695

theorem fraction_problem (x : ℚ) : x * 45 - 5 = 10 → x = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l3436_343695


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l3436_343694

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |x - 2| + |x + 3| ≥ 4} = {x : ℝ | x ≤ -5/2} := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l3436_343694

import Mathlib

namespace square_plot_area_l3945_394548

/-- Given a square plot with a fence, prove that the area is 289 square feet
    when the price per foot is 54 and the total cost is 3672. -/
theorem square_plot_area (side_length : ℝ) : 
  side_length > 0 →
  (4 * side_length * 54 = 3672) →
  side_length^2 = 289 := by
  sorry


end square_plot_area_l3945_394548


namespace japanese_students_fraction_l3945_394540

theorem japanese_students_fraction (J : ℚ) (h : J > 0) : 
  let S := 3 * J
  let seniors_studying := (1/3) * S
  let juniors_studying := (3/4) * J
  let total_students := S + J
  (seniors_studying + juniors_studying) / total_students = 7/16 := by
sorry

end japanese_students_fraction_l3945_394540


namespace stock_market_investment_l3945_394505

theorem stock_market_investment (initial_investment : ℝ) (h_positive : initial_investment > 0) :
  let first_year := initial_investment * 1.75
  let second_year := initial_investment * 1.225
  (first_year - second_year) / first_year = 0.3 := by
sorry

end stock_market_investment_l3945_394505


namespace cos_negative_245_deg_l3945_394569

theorem cos_negative_245_deg (a : ℝ) (h : Real.cos (25 * π / 180) = a) :
  Real.cos (-245 * π / 180) = -Real.sqrt (1 - a^2) := by
  sorry

end cos_negative_245_deg_l3945_394569


namespace smallest_integer_with_remainders_l3945_394596

theorem smallest_integer_with_remainders : ∃ (a : ℕ), 
  (a > 0) ∧ 
  (a % 3 = 2) ∧ 
  (a % 4 = 1) ∧ 
  (a % 5 = 3) ∧
  (∀ (b : ℕ), b > 0 ∧ b % 3 = 2 ∧ b % 4 = 1 ∧ b % 5 = 3 → a ≤ b) ∧
  (a = 53) := by
sorry

end smallest_integer_with_remainders_l3945_394596


namespace work_completion_time_l3945_394531

/-- 
Given:
- A person B can do a work in 20 days
- Persons A and B together can do the work in 15 days

Prove that A can do the work alone in 60 days
-/
theorem work_completion_time (b_time : ℝ) (ab_time : ℝ) (a_time : ℝ) : 
  b_time = 20 → ab_time = 15 → a_time = 60 → 
  1 / a_time + 1 / b_time = 1 / ab_time := by
  sorry

#check work_completion_time

end work_completion_time_l3945_394531


namespace power_of_seven_inverse_l3945_394575

theorem power_of_seven_inverse (x y : ℕ) : 
  (2^x : ℕ) = Nat.gcd 180 (2^Nat.succ x) →
  (3^y : ℕ) = Nat.gcd 180 (3^Nat.succ y) →
  (1/7 : ℚ)^(y - x) = 1 :=
by sorry

end power_of_seven_inverse_l3945_394575


namespace even_periodic_function_monotonicity_l3945_394581

-- Define the properties of the function
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

def increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x ≤ f y

def decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x ≥ f y

-- State the theorem
theorem even_periodic_function_monotonicity (f : ℝ → ℝ) 
  (h_even : is_even f) (h_period : has_period f 2) :
  increasing_on f 0 1 ↔ decreasing_on f 3 4 :=
sorry

end even_periodic_function_monotonicity_l3945_394581


namespace max_cables_sixty_cables_achievable_l3945_394566

/-- Represents the network of computers in the organization -/
structure ComputerNetwork where
  total_employees : ℕ
  brand_a_computers : ℕ
  brand_b_computers : ℕ
  cables : ℕ

/-- Predicate to check if the network satisfies the given conditions -/
def valid_network (n : ComputerNetwork) : Prop :=
  n.total_employees = 50 ∧
  n.brand_a_computers = 30 ∧
  n.brand_b_computers = 20 ∧
  n.cables ≤ n.brand_a_computers * n.brand_b_computers ∧
  n.cables ≥ 2 * n.brand_a_computers

/-- Predicate to check if all employees can communicate -/
def all_can_communicate (n : ComputerNetwork) : Prop :=
  n.cables ≥ n.total_employees - 1

/-- Theorem stating the maximum number of cables -/
theorem max_cables (n : ComputerNetwork) :
  valid_network n → all_can_communicate n → n.cables ≤ 60 :=
by
  sorry

/-- Theorem stating that 60 cables is achievable -/
theorem sixty_cables_achievable :
  ∃ n : ComputerNetwork, valid_network n ∧ all_can_communicate n ∧ n.cables = 60 :=
by
  sorry

end max_cables_sixty_cables_achievable_l3945_394566


namespace quadratic_equation_roots_l3945_394502

theorem quadratic_equation_roots (m : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, m * x^2 + 2*(m+1)*x + (m-1) = 0 ↔ x = x₁ ∨ x = x₂) →
  x₁ ≠ x₂ →
  x₁^2 + x₂^2 = 8 →
  m > -1/2 →
  m ≠ 0 →
  m = (6 + 2*Real.sqrt 33) / 8 :=
by sorry

end quadratic_equation_roots_l3945_394502


namespace coffee_mixture_price_l3945_394554

/-- The price of the second type of coffee bean -/
def second_coffee_price : ℝ := 36

/-- The total weight of the mixture in pounds -/
def total_mixture_weight : ℝ := 100

/-- The selling price of the mixture per pound -/
def mixture_price : ℝ := 11.25

/-- The price of the first type of coffee bean per pound -/
def first_coffee_price : ℝ := 9

/-- The weight of each type of coffee bean in the mixture -/
def each_coffee_weight : ℝ := 25

theorem coffee_mixture_price :
  second_coffee_price * each_coffee_weight +
  first_coffee_price * each_coffee_weight =
  mixture_price * total_mixture_weight :=
by sorry

end coffee_mixture_price_l3945_394554


namespace neg_f_squared_increasing_nonpos_neg_f_squared_decreasing_nonneg_a_range_l3945_394594

noncomputable section

variable (f : ℝ → ℝ)

-- Define the properties of f
axiom f_additive : ∀ x₁ x₂ : ℝ, f (x₁ + x₂) = f x₁ + f x₂
axiom f_increasing_nonneg : ∀ x₁ x₂ : ℝ, x₁ ≥ 0 ∧ x₂ ≥ 0 ∧ x₁ < x₂ → f x₁ < f x₂
axiom f_one_eq_two : f 1 = 2

-- State the theorems to be proved
theorem neg_f_squared_increasing_nonpos :
  ∀ x₁ x₂ : ℝ, x₁ ≤ 0 ∧ x₂ ≤ 0 ∧ x₁ < x₂ → -(f x₁)^2 < -(f x₂)^2 := by sorry

theorem neg_f_squared_decreasing_nonneg :
  ∀ x₁ x₂ : ℝ, x₁ ≥ 0 ∧ x₂ ≥ 0 ∧ x₁ < x₂ → -(f x₁)^2 > -(f x₂)^2 := by sorry

theorem a_range :
  ∀ a : ℝ, f (2*a^2 - 1) + 2*f a - 6 < 0 ↔ -2 < a ∧ a < 1 := by sorry

end

end neg_f_squared_increasing_nonpos_neg_f_squared_decreasing_nonneg_a_range_l3945_394594


namespace convex_ngon_regions_l3945_394551

/-- The number of regions in a convex n-gon divided by its diagonals -/
def num_regions (n : ℕ) : ℚ :=
  (n^4 - 6*n^3 + 23*n^2 - 36*n + 24) / 24

/-- Theorem: For a convex n-gon (n ≥ 4) with all its diagonals drawn and 
    no three diagonals intersecting at the same point, the number of regions 
    into which the n-gon is divided is (n^4 - 6n^3 + 23n^2 - 36n + 24) / 24 -/
theorem convex_ngon_regions (n : ℕ) (h : n ≥ 4) :
  num_regions n = (n^4 - 6*n^3 + 23*n^2 - 36*n + 24) / 24 :=
by sorry

end convex_ngon_regions_l3945_394551


namespace triangle_inequality_l3945_394544

/-- For any triangle with sides a, b, c and area S, 
    the inequality a^2 + b^2 + c^2 - 1/2(|a-b| + |b-c| + |c-a|)^2 ≥ 4√3 S holds. -/
theorem triangle_inequality (a b c S : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_area : S = Real.sqrt ((a + b + c) * (b + c - a) * (c + a - b) * (a + b - c)) / 4) :
  a^2 + b^2 + c^2 - 1/2 * (|a - b| + |b - c| + |c - a|)^2 ≥ 4 * Real.sqrt 3 * S := by
  sorry

end triangle_inequality_l3945_394544


namespace binomial_sum_identity_l3945_394534

theorem binomial_sum_identity (p q n : ℕ+) :
  (∑' k : ℕ, (Nat.choose (p.val + k) p.val) * (Nat.choose (q.val + n.val - k) q.val)) =
  Nat.choose (p.val + q.val + n.val + 1) (p.val + q.val + 1) :=
sorry

end binomial_sum_identity_l3945_394534


namespace cyclists_meet_time_l3945_394595

/-- Two cyclists on a circular track meet at the starting point -/
theorem cyclists_meet_time (v1 v2 C : ℝ) (h1 : v1 = 7) (h2 : v2 = 8) (h3 : C = 600) :
  C / (v1 + v2) = 40 := by
  sorry

end cyclists_meet_time_l3945_394595


namespace parabola_solutions_l3945_394528

/-- The parabola y = ax² + bx + c passes through points (-1, 3) and (2, 3).
    The solutions of a(x-2)² - 3 = 2b - bx - c are 1 and 4. -/
theorem parabola_solutions (a b c : ℝ) : 
  (∀ x : ℝ, a * x^2 + b * x + c = 3 ↔ x = -1 ∨ x = 2) →
  (∀ x : ℝ, a * (x - 2)^2 - 3 = 2 * b - b * x - c ↔ x = 1 ∨ x = 4) :=
by sorry

end parabola_solutions_l3945_394528


namespace min_value_f_l3945_394514

theorem min_value_f (x y : ℝ) (hx : 0 ≤ x ∧ x ≤ π) (hy : 0 ≤ y ∧ y ≤ 1) :
  (2 * y - 1) * Real.sin x + (1 - y) * Real.sin ((1 - y) * x) ≥ 0 := by
  sorry

end min_value_f_l3945_394514


namespace ellipse_focus_l3945_394586

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  center : Point
  majorAxisEnd1 : Point
  majorAxisEnd2 : Point
  minorAxisEnd1 : Point
  minorAxisEnd2 : Point

/-- Theorem: The focus with greater x-coordinate of the given ellipse -/
theorem ellipse_focus (e : Ellipse) 
  (h1 : e.center = ⟨4, -1⟩)
  (h2 : e.majorAxisEnd1 = ⟨0, -1⟩)
  (h3 : e.majorAxisEnd2 = ⟨8, -1⟩)
  (h4 : e.minorAxisEnd1 = ⟨4, 2⟩)
  (h5 : e.minorAxisEnd2 = ⟨4, -4⟩) :
  ∃ (focus : Point), focus.x = 4 + Real.sqrt 7 ∧ focus.y = -1 := by
  sorry

end ellipse_focus_l3945_394586


namespace largest_angle_in_quadrilateral_with_ratio_l3945_394599

/-- 
Given a quadrilateral divided into two triangles by a diagonal,
with the measures of the angles around this diagonal in the ratio 2:3:4:5,
prove that the largest of these angles is 900°/7.
-/
theorem largest_angle_in_quadrilateral_with_ratio (a b c d : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 →  -- Angles are positive
  a + b + c + d = 360 →  -- Sum of angles around a point is 360°
  ∃ (x : ℝ), a = 2*x ∧ b = 3*x ∧ c = 4*x ∧ d = 5*x →  -- Angles are in ratio 2:3:4:5
  (max a (max b (max c d))) = 900 / 7 :=
by sorry

end largest_angle_in_quadrilateral_with_ratio_l3945_394599


namespace symmetric_functions_properties_l3945_394539

/-- Given a > 1, f(x) is symmetric to g(x) = 4 - a^|x-2| - 2*a^(x-2) w.r.t (1, 2) -/
def SymmetricFunctions (a : ℝ) (f : ℝ → ℝ) : Prop :=
  a > 1 ∧ ∀ x y, f x = y ↔ 4 - a^|2-x| - 2*a^(2-x) = 4 - y

theorem symmetric_functions_properties {a : ℝ} {f : ℝ → ℝ} 
  (h : SymmetricFunctions a f) :
  (∀ x, f x = a^|x| + 2*a^(-x)) ∧ 
  (∀ m, (∃ x₁ x₂, 0 < x₁ ∧ 0 < x₂ ∧ x₁ ≠ x₂ ∧ f x₁ = m ∧ f x₂ = m) ↔ 
    2*(2:ℝ)^(1/2) < m ∧ m < 3) := by
  sorry

end symmetric_functions_properties_l3945_394539


namespace fraction_equality_l3945_394568

theorem fraction_equality (a b c x : ℝ) 
  (hx : x = a / b) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (hc : c ≠ 0) 
  (hab : a ≠ b) 
  (hbc : b ≠ c) 
  (hac : a ≠ c) : 
  (a + 2*b + 3*c) / (a - b - 3*c) = (b*(x + 2) + 3*c) / (b*(x - 1) - 3*c) := by
  sorry

end fraction_equality_l3945_394568


namespace perfect_square_binomial_l3945_394517

theorem perfect_square_binomial (x : ℝ) :
  ∃! a : ℝ, ∃ r s : ℝ, 
    a * x^2 + 18 * x + 16 = (r * x + s)^2 ∧ 
    a = (81 : ℝ) / 16 :=
sorry

end perfect_square_binomial_l3945_394517


namespace product_of_roots_l3945_394521

theorem product_of_roots (x : ℝ) : 
  (x^2 - 4*x - 42 = 0) → 
  ∃ y : ℝ, (y^2 - 4*y - 42 = 0) ∧ (x * y = -42) :=
by sorry

end product_of_roots_l3945_394521


namespace greatest_k_value_l3945_394584

theorem greatest_k_value (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ 
    x^2 + k*x + 7 = 0 ∧ 
    y^2 + k*y + 7 = 0 ∧ 
    |x - y| = Real.sqrt 85) →
  k ≤ Real.sqrt 113 :=
by sorry

end greatest_k_value_l3945_394584


namespace product_of_cosines_l3945_394577

theorem product_of_cosines : 
  (1 + Real.cos (π / 6)) * (1 + Real.cos (π / 3)) * 
  (1 + Real.cos ((2 * π) / 3)) * (1 + Real.cos ((5 * π) / 6)) = 3 / 16 := by
  sorry

end product_of_cosines_l3945_394577


namespace excess_students_equals_35_l3945_394585

/-- Represents a kindergarten classroom at Maple Ridge School -/
structure Classroom where
  students : Nat
  rabbits : Nat
  guinea_pigs : Nat

/-- The number of classrooms at Maple Ridge School -/
def num_classrooms : Nat := 5

/-- A standard classroom at Maple Ridge School -/
def standard_classroom : Classroom := {
  students := 15,
  rabbits := 3,
  guinea_pigs := 5
}

/-- The total number of students in all classrooms -/
def total_students : Nat := num_classrooms * standard_classroom.students

/-- The total number of rabbits in all classrooms -/
def total_rabbits : Nat := num_classrooms * standard_classroom.rabbits

/-- The total number of guinea pigs in all classrooms -/
def total_guinea_pigs : Nat := num_classrooms * standard_classroom.guinea_pigs

/-- 
Theorem: The sum of the number of students in excess of the number of pet rabbits 
and the number of guinea pigs in all 5 classrooms is equal to 35.
-/
theorem excess_students_equals_35 : 
  total_students - (total_rabbits + total_guinea_pigs) = 35 := by
  sorry

end excess_students_equals_35_l3945_394585


namespace inverse_g_sum_l3945_394580

-- Define the function g
def g (x : ℝ) : ℝ := x^3 * |x|

-- State the theorem
theorem inverse_g_sum : 
  ∃ (a b : ℝ), g a = 8 ∧ g b = -64 ∧ a + b = Real.sqrt 2 - 2 :=
sorry

end inverse_g_sum_l3945_394580


namespace sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l3945_394522

/-- Two lines are parallel if and only if their slopes are equal and not equal to 1/2 -/
def are_parallel (m : ℝ) : Prop :=
  m / 1 = 1 / m ∧ m / 1 ≠ 1 / 2

/-- The condition m = -1 is sufficient for the lines to be parallel -/
theorem sufficient_condition (m : ℝ) :
  m = -1 → are_parallel m :=
sorry

/-- The condition m = -1 is not necessary for the lines to be parallel -/
theorem not_necessary_condition :
  ∃ m : ℝ, m ≠ -1 ∧ are_parallel m :=
sorry

/-- m = -1 is a sufficient but not necessary condition for the lines to be parallel -/
theorem sufficient_but_not_necessary :
  (∀ m : ℝ, m = -1 → are_parallel m) ∧
  (∃ m : ℝ, m ≠ -1 ∧ are_parallel m) :=
sorry

end sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l3945_394522


namespace line_parallel_plane_perpendicular_line_l3945_394507

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (perpendicularLines : Line → Line → Prop)

-- State the theorem
theorem line_parallel_plane_perpendicular_line 
  (l m : Line) (α : Plane) :
  l ≠ m →
  parallel l α →
  perpendicular m α →
  perpendicularLines l m :=
sorry

end line_parallel_plane_perpendicular_line_l3945_394507


namespace lcm_ratio_sum_l3945_394529

theorem lcm_ratio_sum (a b : ℕ+) : 
  Nat.lcm a b = 30 → 
  a.val * 3 = b.val * 2 → 
  a + b = 50 := by
sorry

end lcm_ratio_sum_l3945_394529


namespace geometric_sequence_sum_l3945_394560

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  (∀ n : ℕ, a n > 0) →
  a 6 * a 10 + a 3 * a 5 = 41 →
  a 4 * a 8 = 5 →
  a 4 + a 8 = Real.sqrt 51 := by
  sorry

end geometric_sequence_sum_l3945_394560


namespace sum_and_fraction_difference_l3945_394597

theorem sum_and_fraction_difference (x y : ℝ) 
  (sum_eq : x + y = 450)
  (fraction_eq : x / y = 0.8) : 
  y - x = 50 := by
sorry

end sum_and_fraction_difference_l3945_394597


namespace gcd_1443_999_l3945_394506

theorem gcd_1443_999 : Nat.gcd 1443 999 = 111 := by
  sorry

end gcd_1443_999_l3945_394506


namespace function_not_monotonic_iff_a_in_range_l3945_394530

/-- The function f(x) is not monotonic on the interval (0, 4) if and only if a is in the open interval (-4, 9/4) -/
theorem function_not_monotonic_iff_a_in_range (a : ℝ) :
  (∃ x y, x ∈ (Set.Ioo 0 4) ∧ y ∈ (Set.Ioo 0 4) ∧ x < y ∧
    ((1/3 * x^3 - 3/2 * x^2 + a*x + 4) > (1/3 * y^3 - 3/2 * y^2 + a*y + 4) ∨
     (1/3 * x^3 - 3/2 * x^2 + a*x + 4) < (1/3 * y^3 - 3/2 * y^2 + a*y + 4)))
  ↔ a ∈ Set.Ioo (-4) (9/4) :=
by sorry

end function_not_monotonic_iff_a_in_range_l3945_394530


namespace employee_pay_percentage_l3945_394592

/-- Given two employees A and B who are paid a total of 580 per week, 
    with B being paid 232 per week, prove that the percentage of A's pay 
    compared to B's pay is 150%. -/
theorem employee_pay_percentage (total_pay b_pay a_pay : ℚ) : 
  total_pay = 580 → 
  b_pay = 232 → 
  a_pay = total_pay - b_pay →
  (a_pay / b_pay) * 100 = 150 := by
  sorry

end employee_pay_percentage_l3945_394592


namespace lunch_cost_with_tip_l3945_394588

theorem lunch_cost_with_tip (total_cost : ℝ) (tip_percentage : ℝ) (original_cost : ℝ) :
  total_cost = 58.075 ∧
  tip_percentage = 0.15 ∧
  total_cost = original_cost * (1 + tip_percentage) →
  original_cost = 50.5 := by
sorry

end lunch_cost_with_tip_l3945_394588


namespace porter_earnings_l3945_394511

/-- Porter's daily rate in dollars -/
def daily_rate : ℚ := 8

/-- Number of regular working days per week -/
def regular_days : ℕ := 5

/-- Number of weeks in a month -/
def weeks_per_month : ℕ := 4

/-- Overtime pay rate as a multiplier of regular rate -/
def overtime_rate : ℚ := 3/2

/-- Tax deduction rate -/
def tax_rate : ℚ := 1/10

/-- Insurance and benefits deduction rate -/
def insurance_rate : ℚ := 1/20

/-- Calculate Porter's monthly earnings after deductions and overtime -/
def monthly_earnings : ℚ :=
  let regular_weekly := daily_rate * regular_days
  let overtime_daily := daily_rate * overtime_rate
  let total_weekly := regular_weekly + overtime_daily
  let monthly_before_deductions := total_weekly * weeks_per_month
  let deductions := monthly_before_deductions * (tax_rate + insurance_rate)
  monthly_before_deductions - deductions

theorem porter_earnings :
  monthly_earnings = 1768/10 := by sorry

end porter_earnings_l3945_394511


namespace certain_number_proof_l3945_394572

theorem certain_number_proof (x : ℝ) : (15 * x) / 100 = 0.04863 → x = 0.3242 := by
  sorry

end certain_number_proof_l3945_394572


namespace volleyball_count_l3945_394567

theorem volleyball_count : ∃ (x y z : ℕ),
  x + y + z = 20 ∧
  60 * x + 30 * y + 10 * z = 330 ∧
  z = 15 := by
sorry

end volleyball_count_l3945_394567


namespace cone_surface_area_l3945_394545

/-- The surface area of a cone with base radius 1 and height √3 is 3π. -/
theorem cone_surface_area : 
  let r : ℝ := 1
  let h : ℝ := Real.sqrt 3
  let l : ℝ := Real.sqrt (r^2 + h^2)
  let surface_area : ℝ := π * r^2 + π * r * l
  surface_area = 3 * π := by sorry

end cone_surface_area_l3945_394545


namespace target_hit_probability_l3945_394583

theorem target_hit_probability (prob_A prob_B : ℝ) : 
  prob_A = 1/2 → 
  prob_B = 1/3 → 
  1 - (1 - prob_A) * (1 - prob_B) = 2/3 := by
sorry

end target_hit_probability_l3945_394583


namespace triangle_angle_A_l3945_394509

theorem triangle_angle_A (a b : ℝ) (B : ℝ) (h1 : a = Real.sqrt 3) (h2 : b = Real.sqrt 2) (h3 : B = π / 4) :
  let A := Real.arcsin ((a * Real.sin B) / b)
  A = π / 3 ∨ A = 2 * π / 3 :=
sorry

end triangle_angle_A_l3945_394509


namespace square_area_l3945_394591

-- Define the coordinates of the vertex and diagonal intersection
def vertex : ℝ × ℝ := (-6, -4)
def diagonal_intersection : ℝ × ℝ := (3, 2)

-- Define the theorem
theorem square_area (v : ℝ × ℝ) (d : ℝ × ℝ) (h1 : v = vertex) (h2 : d = diagonal_intersection) :
  let diagonal_length := Real.sqrt ((d.1 - v.1)^2 + (d.2 - v.2)^2)
  (diagonal_length^2) / 2 = 58.5 := by sorry

end square_area_l3945_394591


namespace tan_195_in_terms_of_cos_165_l3945_394579

theorem tan_195_in_terms_of_cos_165 (a : ℝ) (h : Real.cos (165 * π / 180) = a) :
  Real.tan (195 * π / 180) = -Real.sqrt (1 - a^2) / a := by
  sorry

end tan_195_in_terms_of_cos_165_l3945_394579


namespace pencil_length_l3945_394523

theorem pencil_length : ∀ (L : ℝ),
  (1/8 : ℝ) * L +  -- Black part
  (1/2 : ℝ) * ((7/8 : ℝ) * L) +  -- White part
  (7/2 : ℝ) = L  -- Blue part
  → L = 8 := by
sorry

end pencil_length_l3945_394523


namespace line_segment_param_sum_squares_l3945_394515

/-- 
Given a line segment from (-3,5) to (4,15) parameterized by x = at + b and y = ct + d,
where -1 ≤ t ≤ 2 and t = -1 corresponds to (-3,5), prove that a² + b² + c² + d² = 790/9
-/
theorem line_segment_param_sum_squares (a b c d : ℚ) : 
  (∀ t, -1 ≤ t → t ≤ 2 → ∃ x y, x = a * t + b ∧ y = c * t + d) →
  a * (-1) + b = -3 →
  c * (-1) + d = 5 →
  a * 2 + b = 4 →
  c * 2 + d = 15 →
  a^2 + b^2 + c^2 + d^2 = 790/9 := by
sorry

end line_segment_param_sum_squares_l3945_394515


namespace breadth_is_ten_l3945_394516

/-- A rectangular plot with specific properties -/
structure RectangularPlot where
  breadth : ℝ
  length : ℝ
  area : ℝ
  area_eq : area = 20 * breadth
  length_eq : length = breadth + 10

/-- The breadth of a rectangular plot with the given properties is 10 meters -/
theorem breadth_is_ten (plot : RectangularPlot) : plot.breadth = 10 := by
  sorry

end breadth_is_ten_l3945_394516


namespace subtract_multiply_real_l3945_394526

theorem subtract_multiply_real : 3.56 - 2.1 * 1.5 = 0.41 := by
  sorry

end subtract_multiply_real_l3945_394526


namespace andy_profit_per_cake_l3945_394557

/-- Calculates the profit per cake for Andy's cake business -/
def profit_per_cake (ingredient_cost_two_cakes : ℚ) (packaging_cost_per_cake : ℚ) (selling_price : ℚ) : ℚ :=
  selling_price - (ingredient_cost_two_cakes / 2 + packaging_cost_per_cake)

/-- Proves that Andy's profit per cake is $8 -/
theorem andy_profit_per_cake :
  profit_per_cake 12 1 15 = 8 := by
  sorry

end andy_profit_per_cake_l3945_394557


namespace intersection_when_a_neg_two_subset_condition_l3945_394582

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | 2*a - 1 ≤ x ∧ x ≤ a + 3}
def B : Set ℝ := {x | x < -1 ∨ x > 5}

-- Theorem 1: Intersection of A and B when a = -2
theorem intersection_when_a_neg_two :
  A (-2) ∩ B = {x | -5 ≤ x ∧ x < -1} := by sorry

-- Theorem 2: Condition for A to be a subset of B
theorem subset_condition (a : ℝ) :
  A a ⊆ B ↔ a ≤ -4 ∨ a ≥ 3 := by sorry

end intersection_when_a_neg_two_subset_condition_l3945_394582


namespace balloon_difference_l3945_394578

/-- The number of balloons Allan brought to the park -/
def allan_initial : ℕ := 2

/-- The number of balloons Allan bought at the park -/
def allan_bought : ℕ := 3

/-- The number of balloons Jake brought to the park -/
def jake_balloons : ℕ := 6

/-- The total number of balloons Allan had in the park -/
def allan_total : ℕ := allan_initial + allan_bought

theorem balloon_difference : jake_balloons - allan_total = 1 := by
  sorry

end balloon_difference_l3945_394578


namespace bridge_toll_fee_calculation_l3945_394552

/-- Represents the taxi fare structure -/
structure TaxiFare where
  start_fee : ℝ
  per_mile_rate : ℝ

/-- Calculates the total fare for a given distance -/
def calculate_fare (fare : TaxiFare) (distance : ℝ) : ℝ :=
  fare.start_fee + fare.per_mile_rate * distance

theorem bridge_toll_fee_calculation :
  let mike_fare : TaxiFare := { start_fee := 2.50, per_mile_rate := 0.25 }
  let annie_fare : TaxiFare := { start_fee := 2.50, per_mile_rate := 0.25 }
  let mike_distance : ℝ := 36
  let annie_distance : ℝ := 16
  let mike_total : ℝ := calculate_fare mike_fare mike_distance
  let annie_base : ℝ := calculate_fare annie_fare annie_distance
  let bridge_toll : ℝ := mike_total - annie_base
  bridge_toll = 5 := by sorry

end bridge_toll_fee_calculation_l3945_394552


namespace happy_water_consumption_l3945_394562

/-- Given Happy's current water consumption and recommended increase percentage,
    calculate the new recommended number of cups per week. -/
theorem happy_water_consumption (current : ℝ) (increase_percent : ℝ) :
  current = 25 → increase_percent = 75 →
  current + (increase_percent / 100) * current = 43.75 := by
  sorry

end happy_water_consumption_l3945_394562


namespace point_above_plane_l3945_394570

theorem point_above_plane (a : ℝ) : 
  (∀ x y : ℝ, x = 1 ∧ y = 2 → x + y + a > 0) ↔ a > -3 :=
sorry

end point_above_plane_l3945_394570


namespace complement_intersection_theorem_l3945_394547

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {5, 6, 7}

theorem complement_intersection_theorem :
  (U \ A) ∩ (U \ B) = {4, 8} := by sorry

end complement_intersection_theorem_l3945_394547


namespace natalie_shopping_money_left_l3945_394518

def initial_amount : ℕ := 26
def jumper_cost : ℕ := 9
def tshirt_cost : ℕ := 4
def heels_cost : ℕ := 5

theorem natalie_shopping_money_left :
  initial_amount - (jumper_cost + tshirt_cost + heels_cost) = 8 := by
  sorry

end natalie_shopping_money_left_l3945_394518


namespace connie_marble_count_l3945_394504

def marble_problem (connie_marbles juan_marbles : ℕ) : Prop :=
  juan_marbles = connie_marbles + 175 ∧ juan_marbles = 498

theorem connie_marble_count :
  ∀ connie_marbles juan_marbles,
    marble_problem connie_marbles juan_marbles →
    connie_marbles = 323 :=
by
  sorry

end connie_marble_count_l3945_394504


namespace sum_of_coordinates_after_reflection_l3945_394543

/-- Given a point C with coordinates (x, -5) and its reflection D over the y-axis,
    the sum of all coordinate values of C and D is -10. -/
theorem sum_of_coordinates_after_reflection (x : ℝ) :
  let C : ℝ × ℝ := (x, -5)
  let D : ℝ × ℝ := (-x, -5)  -- reflection of C over y-axis
  (C.1 + C.2 + D.1 + D.2) = -10 :=
by sorry

end sum_of_coordinates_after_reflection_l3945_394543


namespace camille_bird_counting_l3945_394555

theorem camille_bird_counting (cardinals : ℕ) 
  (h1 : cardinals > 0)
  (h2 : cardinals + 4 * cardinals + 2 * cardinals + (3 * cardinals + 1) = 31) :
  cardinals = 3 := by
sorry

end camille_bird_counting_l3945_394555


namespace complex_power_100_l3945_394546

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_power_100 : ((1 + i) / (1 - i)) ^ 100 = 1 := by
  sorry

end complex_power_100_l3945_394546


namespace soccer_league_female_fraction_l3945_394512

theorem soccer_league_female_fraction :
  ∀ (male_last_year female_last_year : ℕ)
    (total_this_year : ℚ)
    (male_this_year female_this_year : ℚ),
  male_last_year = 15 →
  total_this_year = 1.15 * (male_last_year + female_last_year) →
  male_this_year = 1.1 * male_last_year →
  female_this_year = 2 * female_last_year →
  female_this_year / total_this_year = 5 / 51 :=
by sorry

end soccer_league_female_fraction_l3945_394512


namespace tv_selection_theorem_l3945_394508

/-- The number of ways to select k items from n items -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The number of Type A TVs -/
def typeA : ℕ := 4

/-- The number of Type B TVs -/
def typeB : ℕ := 5

/-- The total number of TVs to be selected -/
def selectTotal : ℕ := 3

/-- The number of ways to select TVs satisfying the given conditions -/
def selectWays : ℕ :=
  typeA * binomial typeB (selectTotal - 1) +
  binomial typeA (selectTotal - 1) * typeB

theorem tv_selection_theorem : selectWays = 70 := by sorry

end tv_selection_theorem_l3945_394508


namespace decimal_to_fraction_l3945_394563

/-- The decimal representation of a repeating decimal ending in 6 -/
def S : ℚ := 0.666666

/-- Theorem stating that the decimal 0.666... is equal to 2/3 -/
theorem decimal_to_fraction : S = 2/3 := by sorry

end decimal_to_fraction_l3945_394563


namespace class_survey_is_comprehensive_l3945_394558

/-- Represents a survey population -/
structure SurveyPopulation where
  size : ℕ
  is_finite : Bool

/-- Defines what makes a survey comprehensive -/
def is_comprehensive_survey (pop : SurveyPopulation) : Prop :=
  pop.is_finite ∧ pop.size > 0

/-- Represents a class of students -/
def class_of_students : SurveyPopulation :=
  { size := 30,  -- Assuming an average class size
    is_finite := true }

/-- Theorem stating that a survey of a class is suitable for a comprehensive survey -/
theorem class_survey_is_comprehensive :
  is_comprehensive_survey class_of_students := by
  sorry


end class_survey_is_comprehensive_l3945_394558


namespace base3_to_base10_conversion_l3945_394527

/-- Converts a list of digits in base 3 to a base 10 integer -/
def base3ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (3 ^ i)) 0

/-- The base 3 representation of the number -/
def base3Digits : List Nat := [1, 2, 0, 2, 1]

theorem base3_to_base10_conversion :
  base3ToBase10 base3Digits = 142 := by
  sorry

end base3_to_base10_conversion_l3945_394527


namespace new_oranges_added_l3945_394593

theorem new_oranges_added (initial : ℕ) (thrown_away : ℕ) (final : ℕ) : 
  initial = 31 → thrown_away = 9 → final = 60 → final - (initial - thrown_away) = 38 := by
  sorry

end new_oranges_added_l3945_394593


namespace system_always_solvable_l3945_394556

/-- Given a system of linear equations:
    ax + by = c - 1
    (a+5)x + (b+3)y = c + 1
    This theorem states that for the system to always have a solution
    for any real a and b, c must equal (2a + 5) / 5. -/
theorem system_always_solvable (a b c : ℝ) :
  (∀ x y : ℝ, a * x + b * y = c - 1 ∧ (a + 5) * x + (b + 3) * y = c + 1) ↔
  c = (2 * a + 5) / 5 := by
  sorry


end system_always_solvable_l3945_394556


namespace object_speed_l3945_394500

/-- An object traveling 10800 feet in one hour has a speed of 3 feet per second. -/
theorem object_speed (distance : ℝ) (time_in_seconds : ℝ) (h1 : distance = 10800) (h2 : time_in_seconds = 3600) :
  distance / time_in_seconds = 3 := by
  sorry

end object_speed_l3945_394500


namespace distinct_odd_numbers_count_l3945_394549

-- Define the given number as a list of digits
def given_number : List Nat := [3, 4, 3, 9, 6]

-- Function to check if a number is odd
def is_odd (n : Nat) : Bool :=
  n % 2 = 1

-- Function to count distinct permutations
def count_distinct_permutations (digits : List Nat) : Nat :=
  sorry

-- Function to count distinct odd permutations
def count_distinct_odd_permutations (digits : List Nat) : Nat :=
  sorry

-- Theorem statement
theorem distinct_odd_numbers_count :
  count_distinct_odd_permutations given_number = 36 := by
  sorry

end distinct_odd_numbers_count_l3945_394549


namespace system_solution_l3945_394559

-- Define the system of linear equations
def system (k : ℝ) (x y : ℝ) : Prop :=
  x - y = 9 * k ∧ x + y = 5 * k

-- Define the additional equation
def additional_eq (x y : ℝ) : Prop :=
  2 * x + 3 * y = 8

-- Theorem statement
theorem system_solution :
  ∀ k x y, system k x y → additional_eq x y → k = 1 ∧ x = 7 ∧ y = -2 :=
by sorry

end system_solution_l3945_394559


namespace f_r_correct_l3945_394564

/-- The number of ways to select k elements from a permutation of n elements,
    such that any two selected elements are separated by at least r elements
    in the original permutation. -/
def f_r (n k r : ℕ) : ℕ :=
  Nat.choose (n - k * r + r) k

/-- Theorem stating that f_r(n, k, r) correctly counts the number of ways to select
    k elements from a permutation of n elements with the given separation condition. -/
theorem f_r_correct (n k r : ℕ) :
  f_r n k r = Nat.choose (n - k * r + r) k :=
by sorry

end f_r_correct_l3945_394564


namespace set_operation_result_arithmetic_expression_result_l3945_394519

-- Define the sets A, B, and C
def A : Set ℝ := {x | x > 1}
def B : Set ℝ := {x | -2 < x ∧ x < 2}
def C : Set ℝ := {x | -3 < x ∧ x < 5}

-- Theorem 1: Set operation result
theorem set_operation_result : (A ∪ B) ∩ C = {x : ℝ | -2 < x ∧ x < 5} := by sorry

-- Theorem 2: Arithmetic expression result
theorem arithmetic_expression_result :
  (2 + 1/4)^(1/2) - (-9.6)^0 - (3 + 3/8)^(-2/3) + (1.5)^(-2) = 1/2 := by sorry

end set_operation_result_arithmetic_expression_result_l3945_394519


namespace triangle_theorem_l3945_394598

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
  (h1 : t.a * Real.cos t.B + t.b * Real.cos t.A = 2 * t.c * Real.cos t.C) :
  -- Part 1: Angle C is 60 degrees (π/3 radians)
  t.C = π / 3 ∧
  -- Part 2: If c = 2, the maximum area is √3
  (t.c = 2 → ∃ (max_area : ℝ), max_area = Real.sqrt 3 ∧
    ∀ (area : ℝ), area = 1/2 * t.a * t.b * Real.sin t.C → area ≤ max_area) :=
by sorry

end triangle_theorem_l3945_394598


namespace sample_size_is_thirty_l3945_394510

/-- Represents the ratio of young, middle-aged, and elderly employees -/
structure EmployeeRatio :=
  (young : ℕ)
  (middle : ℕ)
  (elderly : ℕ)

/-- Calculates the total sample size given the ratio and number of young employees in the sample -/
def calculateSampleSize (ratio : EmployeeRatio) (youngInSample : ℕ) : ℕ :=
  let totalRatio := ratio.young + ratio.middle + ratio.elderly
  (youngInSample * totalRatio) / ratio.young

/-- Theorem stating that for the given ratio and number of young employees, the sample size is 30 -/
theorem sample_size_is_thirty :
  let ratio : EmployeeRatio := { young := 7, middle := 5, elderly := 3 }
  let youngInSample : ℕ := 14
  calculateSampleSize ratio youngInSample = 30 := by
  sorry

end sample_size_is_thirty_l3945_394510


namespace m_range_l3945_394532

theorem m_range : ∀ m : ℝ, m = 3 * Real.sqrt 2 - 1 → 3 < m ∧ m < 4 := by
  sorry

end m_range_l3945_394532


namespace juniper_bones_l3945_394533

theorem juniper_bones (b : ℕ) : 2 * b - 2 = (b + b) - 2 := by sorry

end juniper_bones_l3945_394533


namespace two_fifths_in_twice_one_tenth_l3945_394536

theorem two_fifths_in_twice_one_tenth : (2 * (1 / 10)) / (2 / 5) = 1 / 2 := by
  sorry

end two_fifths_in_twice_one_tenth_l3945_394536


namespace divisor_problem_l3945_394535

def divisor_count (n : ℕ) : ℕ := (Nat.divisors n).card

theorem divisor_problem :
  (∃! k : ℕ, 2 ∣ k ∧ 9 ∣ k ∧ divisor_count k = 14) ∧
  (∃ k₁ k₂ : ℕ, k₁ ≠ k₂ ∧ 2 ∣ k₁ ∧ 9 ∣ k₁ ∧ divisor_count k₁ = 15 ∧
                2 ∣ k₂ ∧ 9 ∣ k₂ ∧ divisor_count k₂ = 15) ∧
  (¬ ∃ k : ℕ, 2 ∣ k ∧ 9 ∣ k ∧ divisor_count k = 17) :=
by sorry

end divisor_problem_l3945_394535


namespace simplify_expression_l3945_394520

theorem simplify_expression (x : ℝ) : (2*x + 25) + (150*x^2 + 2*x + 25) = 150*x^2 + 4*x + 50 := by
  sorry

end simplify_expression_l3945_394520


namespace fraction_value_at_2017_l3945_394513

theorem fraction_value_at_2017 :
  let x : ℤ := 2017
  (x^2 + 6*x + 9) / (x + 3) = 2020 := by
  sorry

end fraction_value_at_2017_l3945_394513


namespace metallic_sheet_length_l3945_394590

/-- Proves that a rectangular sheet with one side 36 m, when cut to form a box of volume 3780 m³, has an original length of 48 m -/
theorem metallic_sheet_length (L : ℝ) : 
  L > 0 → 
  (L - 6) * (36 - 6) * 3 = 3780 → 
  L = 48 :=
by sorry

end metallic_sheet_length_l3945_394590


namespace derivative_at_one_l3945_394576

theorem derivative_at_one (f : ℝ → ℝ) (h : ∀ x, f x = x^3 - 2 * (deriv f 1) * x) :
  deriv f 1 = 1 :=
sorry

end derivative_at_one_l3945_394576


namespace range_of_f_range_of_g_l3945_394573

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 4*a*x + 2*a + 6

-- Define the function g
def g (a : ℝ) : ℝ := 2 - a * |a + 3|

-- Theorem 1: The range of f is [0,+∞) iff a = -1 or a = 3/2
theorem range_of_f (a : ℝ) : 
  (∀ y : ℝ, y ≥ 0 → ∃ x : ℝ, f a x = y) ∧ (∀ x : ℝ, f a x ≥ 0) ↔ 
  a = -1 ∨ a = 3/2 :=
sorry

-- Theorem 2: When f(x) ≥ 0 for all x, the range of g(a) is [-19/4, 4]
theorem range_of_g : 
  (∀ a : ℝ, (∀ x : ℝ, f a x ≥ 0) → 
    (∀ y : ℝ, -19/4 ≤ y ∧ y ≤ 4 → ∃ a : ℝ, g a = y) ∧ 
    (∀ a : ℝ, -19/4 ≤ g a ∧ g a ≤ 4)) :=
sorry

end range_of_f_range_of_g_l3945_394573


namespace a_55_mod_45_l3945_394537

/-- Definition of a_n as a function that concatenates integers from 1 to n -/
def a (n : ℕ) : ℕ := sorry

/-- The remainder when a_55 is divided by 45 is 10 -/
theorem a_55_mod_45 : a 55 % 45 = 10 := by sorry

end a_55_mod_45_l3945_394537


namespace function_value_at_two_l3945_394501

theorem function_value_at_two (f : ℝ → ℝ) (h : ∀ x, f (x + 1) = x^2 + 1) : f 2 = 2 := by
  sorry

end function_value_at_two_l3945_394501


namespace bank_account_withdrawal_l3945_394503

theorem bank_account_withdrawal (initial_balance : ℚ) : 
  initial_balance > 0 →
  let remaining_balance := initial_balance - 400
  let deposit := (1 / 4) * remaining_balance
  let final_balance := remaining_balance + deposit
  final_balance = 750 →
  400 / initial_balance = 2 / 5 := by
sorry

end bank_account_withdrawal_l3945_394503


namespace rectangle_diagonal_l3945_394571

theorem rectangle_diagonal (perimeter : ℝ) (length_ratio width_ratio : ℕ) 
  (h_perimeter : perimeter = 72) 
  (h_ratio : length_ratio = 5 ∧ width_ratio = 4) : 
  ∃ (diagonal : ℝ), diagonal = 4 * Real.sqrt 41 :=
by sorry

end rectangle_diagonal_l3945_394571


namespace problem_statement_l3945_394541

theorem problem_statement : (1 / ((-8^2)^4)) * (-8)^9 = -8 := by sorry

end problem_statement_l3945_394541


namespace race_theorem_l3945_394542

/-- A race with 40 kids where some finish under 6 minutes, some under 8 minutes, and the rest take longer. -/
structure Race where
  total_kids : ℕ
  under_6_min : ℕ
  under_8_min : ℕ
  over_certain_min : ℕ

/-- The race satisfies the given conditions. -/
def race_conditions (r : Race) : Prop :=
  r.total_kids = 40 ∧
  r.under_6_min = (10 : ℕ) * r.total_kids / 100 ∧
  r.under_8_min = 3 * r.under_6_min ∧
  r.over_certain_min = 4 ∧
  r.over_certain_min = (r.total_kids - (r.under_6_min + r.under_8_min)) / 6

/-- The theorem stating that the number of kids who take more than a certain number of minutes is 4. -/
theorem race_theorem (r : Race) (h : race_conditions r) : r.over_certain_min = 4 := by
  sorry


end race_theorem_l3945_394542


namespace min_value_of_expression_l3945_394574

theorem min_value_of_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + b^2) / c + (a^2 + c^2) / b + (b^2 + c^2) / a ≥ 6 ∧
  ∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧
    (a₀^2 + b₀^2) / c₀ + (a₀^2 + c₀^2) / b₀ + (b₀^2 + c₀^2) / a₀ = 6 :=
by sorry


end min_value_of_expression_l3945_394574


namespace equidistant_point_on_z_axis_l3945_394550

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculate the squared distance between two points -/
def squaredDistance (p1 p2 : Point3D) : ℝ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2

/-- The theorem stating that C(0, 0, 1) is equidistant from A(1, 0, 2) and B(1, 1, 1) -/
theorem equidistant_point_on_z_axis : 
  let A : Point3D := ⟨1, 0, 2⟩
  let B : Point3D := ⟨1, 1, 1⟩
  let C : Point3D := ⟨0, 0, 1⟩
  squaredDistance A C = squaredDistance B C := by
  sorry

end equidistant_point_on_z_axis_l3945_394550


namespace add_twice_equals_thrice_l3945_394553

theorem add_twice_equals_thrice (a : ℝ) : a + 2 * a = 3 * a := by
  sorry

end add_twice_equals_thrice_l3945_394553


namespace recurring_decimal_to_fraction_l3945_394524

theorem recurring_decimal_to_fraction :
  ∃ (n d : ℕ), n > 0 ∧ d > 0 ∧ (n.gcd d = 1) ∧
  (7 + 318 / 999 : ℚ) = n / d :=
sorry

end recurring_decimal_to_fraction_l3945_394524


namespace coupon_a_best_discount_correct_prices_l3945_394587

def coupon_a_discount (x : ℝ) : ℝ := 0.15 * x

def coupon_b_discount : ℝ := 30

def coupon_c_discount (x : ℝ) : ℝ := 0.22 * (x - 150)

theorem coupon_a_best_discount (x : ℝ) 
  (h1 : 200 < x) (h2 : x < 471.43) : 
  coupon_a_discount x > coupon_b_discount ∧ 
  coupon_a_discount x > coupon_c_discount x := by
  sorry

def price_list : List ℝ := [179.95, 199.95, 249.95, 299.95, 349.95]

theorem correct_prices (p : ℝ) (h : p ∈ price_list) :
  (200 < p ∧ p < 471.43) ↔ (p = 249.95 ∨ p = 299.95 ∨ p = 349.95) := by
  sorry

end coupon_a_best_discount_correct_prices_l3945_394587


namespace polynomial_functional_equation_l3945_394561

-- Define a polynomial with real coefficients
def RealPolynomial := ℝ → ℝ

-- Define the functional equation
def SatisfiesFunctionalEquation (P : RealPolynomial) : Prop :=
  ∀ x : ℝ, 1 + P x = (1 / 2) * (P (x - 1) + P (x + 1))

-- Define the quadratic form
def IsQuadraticForm (P : RealPolynomial) : Prop :=
  ∃ b c : ℝ, ∀ x : ℝ, P x = x^2 + b * x + c

-- Theorem statement
theorem polynomial_functional_equation :
  ∀ P : RealPolynomial, SatisfiesFunctionalEquation P → IsQuadraticForm P :=
by
  sorry


end polynomial_functional_equation_l3945_394561


namespace division_problem_l3945_394525

theorem division_problem (n : ℕ) (h1 : n % 11 = 1) (h2 : n / 11 = 13) : n = 144 := by
  sorry

end division_problem_l3945_394525


namespace intersection_M_N_l3945_394538

def M : Set ℝ := {x | x^2 < 4}
def N : Set ℝ := {-1, 1, 2}

theorem intersection_M_N : M ∩ N = {-1, 1} := by sorry

end intersection_M_N_l3945_394538


namespace sum_of_f_values_l3945_394589

noncomputable def f (x : ℝ) : ℝ := 2 / (2^x + 1) + Real.sin x

theorem sum_of_f_values : 
  f (-3) + f (-2) + f (-1) + f 0 + f 1 + f 2 + f 3 = 7 := by sorry

end sum_of_f_values_l3945_394589


namespace sugar_solution_replacement_l3945_394565

/-- Calculates the sugar percentage of a final solution after replacing part of an original solution with a new solution. -/
def final_sugar_percentage (original_percentage : ℚ) (replaced_fraction : ℚ) (new_percentage : ℚ) : ℚ :=
  (1 - replaced_fraction) * original_percentage + replaced_fraction * new_percentage

/-- Theorem stating that replacing 1/4 of a 10% sugar solution with a 38% sugar solution results in a 17% sugar solution. -/
theorem sugar_solution_replacement :
  final_sugar_percentage (10 / 100) (1 / 4) (38 / 100) = 17 / 100 := by
  sorry

#eval final_sugar_percentage (10 / 100) (1 / 4) (38 / 100)

end sugar_solution_replacement_l3945_394565

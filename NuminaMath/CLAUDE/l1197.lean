import Mathlib

namespace NUMINAMATH_CALUDE_max_exchanges_theorem_l1197_119761

/-- Represents a student with a height -/
structure Student where
  height : ℕ

/-- Represents a circle of students -/
def StudentCircle := List Student

/-- Condition for a student to switch places -/
def canSwitch (s₁ s₂ : Student) (prevHeight : ℕ) : Prop :=
  s₁.height > s₂.height ∧ s₂.height ≤ prevHeight

/-- The maximum number of exchanges possible -/
def maxExchanges (n : ℕ) : ℕ := n * (n - 1) * (n - 2) / 6

/-- Theorem stating the maximum number of exchanges -/
theorem max_exchanges_theorem (n : ℕ) (circle : StudentCircle) :
  (circle.length = n) →
  (∀ i j, i < j → (circle.get i).height < (circle.get j).height) →
  (∀ exchanges, exchanges ≤ maxExchanges n) := by
  sorry

end NUMINAMATH_CALUDE_max_exchanges_theorem_l1197_119761


namespace NUMINAMATH_CALUDE_rational_function_sum_l1197_119717

/-- Given rational functions p(x) and q(x) satisfying certain conditions,
    prove that their sum has a specific form. -/
theorem rational_function_sum (p q : ℝ → ℝ) : 
  (∀ x, ∃ y, q x = y * (x + 1) * (x - 2) * (x - 3)) →  -- q(x) is cubic with specific factors
  (∀ x, ∃ y, p x = y * (x + 1) * (x - 2)) →  -- p(x) is quadratic with specific factors
  p 2 = 2 →  -- p(2) = 2
  q (-1) = -1 →  -- q(-1) = -1
  ∀ x, p x + q x = x^3 - 3*x^2 + 4*x + 4 := by
sorry

end NUMINAMATH_CALUDE_rational_function_sum_l1197_119717


namespace NUMINAMATH_CALUDE_elon_has_13_teslas_l1197_119703

/-- The number of Teslas Chris has -/
def chris_teslas : ℕ := 6

/-- The number of Teslas Sam has -/
def sam_teslas : ℕ := chris_teslas / 2

/-- The number of Teslas Elon has -/
def elon_teslas : ℕ := sam_teslas + 10

theorem elon_has_13_teslas : elon_teslas = 13 := by
  sorry

end NUMINAMATH_CALUDE_elon_has_13_teslas_l1197_119703


namespace NUMINAMATH_CALUDE_boxes_per_case_l1197_119786

/-- Given that Shirley sold 10 boxes of trefoils and needs to deliver 5 cases of boxes,
    prove that there are 2 boxes in each case. -/
theorem boxes_per_case (total_boxes : ℕ) (num_cases : ℕ) 
    (h1 : total_boxes = 10) (h2 : num_cases = 5) :
  total_boxes / num_cases = 2 := by
  sorry

end NUMINAMATH_CALUDE_boxes_per_case_l1197_119786


namespace NUMINAMATH_CALUDE_journey_distance_l1197_119788

theorem journey_distance : ∀ (D : ℝ),
  (1/5 : ℝ) * D + (2/3 : ℝ) * D + 12 = D →
  D = 90 := by sorry

end NUMINAMATH_CALUDE_journey_distance_l1197_119788


namespace NUMINAMATH_CALUDE_no_solutions_factorial_equation_l1197_119715

theorem no_solutions_factorial_equation (n m : ℕ) (h : m ≥ 2) :
  n.factorial ≠ 2^m * m.factorial :=
sorry

end NUMINAMATH_CALUDE_no_solutions_factorial_equation_l1197_119715


namespace NUMINAMATH_CALUDE_number_in_set_l1197_119777

theorem number_in_set (initial_avg : ℝ) (wrong_num : ℝ) (correct_num : ℝ) (correct_avg : ℝ) :
  initial_avg = 23 →
  wrong_num = 26 →
  correct_num = 36 →
  correct_avg = 24 →
  ∃ n : ℕ, n > 0 ∧ 
    (n : ℝ) * initial_avg - wrong_num = (n : ℝ) * correct_avg - correct_num ∧
    n = 10 :=
by sorry

end NUMINAMATH_CALUDE_number_in_set_l1197_119777


namespace NUMINAMATH_CALUDE_triangle_perimeter_l1197_119792

theorem triangle_perimeter (a b c : ℝ) : 
  a = 2 → (b - 2)^2 + |c - 3| = 0 → a + b + c = 7 := by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l1197_119792


namespace NUMINAMATH_CALUDE_complex_magnitude_l1197_119787

theorem complex_magnitude (z : ℂ) (h : z + Complex.abs z = 2 + 8 * Complex.I) : Complex.abs z = 17 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l1197_119787


namespace NUMINAMATH_CALUDE_jose_profit_share_l1197_119794

/-- Calculates the share of profit for an investor given their investment amount, 
    investment duration, total investment-months, and total profit. -/
def shareOfProfit (investment : ℕ) (duration : ℕ) (totalInvestmentMonths : ℕ) (totalProfit : ℕ) : ℚ :=
  (investment * duration : ℚ) / totalInvestmentMonths * totalProfit

theorem jose_profit_share 
  (tom_investment : ℕ) (tom_duration : ℕ) 
  (jose_investment : ℕ) (jose_duration : ℕ) 
  (total_profit : ℕ) : 
  tom_investment = 30000 → 
  tom_duration = 12 → 
  jose_investment = 45000 → 
  jose_duration = 10 → 
  total_profit = 45000 → 
  shareOfProfit jose_investment jose_duration 
    (tom_investment * tom_duration + jose_investment * jose_duration) total_profit = 25000 := by
  sorry

#check jose_profit_share

end NUMINAMATH_CALUDE_jose_profit_share_l1197_119794


namespace NUMINAMATH_CALUDE_dave_tickets_proof_l1197_119738

/-- The number of tickets Dave won initially -/
def initial_tickets : ℕ := 11

/-- The number of tickets Dave spent on a beanie -/
def spent_tickets : ℕ := 5

/-- The number of additional tickets Dave won later -/
def additional_tickets : ℕ := 10

/-- The number of tickets Dave has now -/
def current_tickets : ℕ := 16

/-- Theorem stating that the initial number of tickets is correct given the conditions -/
theorem dave_tickets_proof :
  initial_tickets - spent_tickets + additional_tickets = current_tickets :=
by sorry

end NUMINAMATH_CALUDE_dave_tickets_proof_l1197_119738


namespace NUMINAMATH_CALUDE_largest_prime_divisor_l1197_119759

def base_8_number : ℕ := 201021022

theorem largest_prime_divisor :
  let decimal_number := 35661062
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ decimal_number ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ decimal_number → q ≤ p ∧ p = 17830531 := by
  sorry

#eval base_8_number

end NUMINAMATH_CALUDE_largest_prime_divisor_l1197_119759


namespace NUMINAMATH_CALUDE_remaining_salary_l1197_119789

theorem remaining_salary (salary : ℝ) (food_fraction house_rent_fraction clothes_fraction : ℝ) 
  (h1 : salary = 140000)
  (h2 : food_fraction = 1/5)
  (h3 : house_rent_fraction = 1/10)
  (h4 : clothes_fraction = 3/5)
  (h5 : food_fraction + house_rent_fraction + clothes_fraction < 1) :
  salary * (1 - (food_fraction + house_rent_fraction + clothes_fraction)) = 14000 :=
by sorry

end NUMINAMATH_CALUDE_remaining_salary_l1197_119789


namespace NUMINAMATH_CALUDE_average_age_is_35_l1197_119744

-- Define the ages of John, Mary, and Tonya
def john_age : ℕ := 30
def mary_age : ℕ := 15
def tonya_age : ℕ := 60

-- State the theorem
theorem average_age_is_35 :
  (john_age = 2 * mary_age) ∧  -- John is twice as old as Mary
  (2 * john_age = tonya_age) ∧  -- John is half as old as Tonya
  (tonya_age = 60) →  -- Tonya is 60 years old
  (john_age + mary_age + tonya_age) / 3 = 35 := by
  sorry

#check average_age_is_35

end NUMINAMATH_CALUDE_average_age_is_35_l1197_119744


namespace NUMINAMATH_CALUDE_triangle_division_ratio_l1197_119756

/-- Given a triangle ABC, this theorem proves that if point F divides side AC in the ratio 2:3,
    G is the midpoint of BF, and E is the point of intersection of side BC and AG,
    then E divides side BC in the ratio 2:5. -/
theorem triangle_division_ratio (A B C F G E : ℝ × ℝ) : 
  (∃ k : ℝ, F = A + (2/5 : ℝ) • (C - A)) →  -- F divides AC in ratio 2:3
  G = B + (1/2 : ℝ) • (F - B) →              -- G is midpoint of BF
  (∃ t : ℝ, E = B + t • (C - B) ∧ 
            E = A + t • (G - A)) →           -- E is intersection of BC and AG
  (∃ s : ℝ, E = B + (2/7 : ℝ) • (C - B)) :=   -- E divides BC in ratio 2:5
by sorry


end NUMINAMATH_CALUDE_triangle_division_ratio_l1197_119756


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l1197_119714

theorem inequality_and_equality_condition (x₁ x₂ : ℝ) 
  (h₁ : |x₁| ≤ 1) (h₂ : |x₂| ≤ 1) : 
  Real.sqrt (1 - x₁^2) + Real.sqrt (1 - x₂^2) ≤ 2 * Real.sqrt (1 - ((x₁ + x₂)/2)^2) ∧ 
  (Real.sqrt (1 - x₁^2) + Real.sqrt (1 - x₂^2) = 2 * Real.sqrt (1 - ((x₁ + x₂)/2)^2) ↔ x₁ = x₂) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l1197_119714


namespace NUMINAMATH_CALUDE_hyperbola_properties_l1197_119799

/-- The hyperbola defined by the equation x^2 - y^2 = 1 passes through (1, 0) and has asymptotes x ± y = 0 -/
theorem hyperbola_properties :
  ∃ (x y : ℝ), 
    (x^2 - y^2 = 1) ∧ 
    (x = 1 ∧ y = 0) ∧
    (∀ (t : ℝ), (x = t ∧ y = t) ∨ (x = t ∧ y = -t)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l1197_119799


namespace NUMINAMATH_CALUDE_matinee_ticket_cost_l1197_119742

/-- The cost of a matinee ticket in dollars -/
def matinee_cost : ℚ := 5

/-- The cost of an evening ticket in dollars -/
def evening_cost : ℚ := 7

/-- The cost of an opening night ticket in dollars -/
def opening_night_cost : ℚ := 10

/-- The cost of a bucket of popcorn in dollars -/
def popcorn_cost : ℚ := 10

/-- The number of matinee customers -/
def matinee_customers : ℕ := 32

/-- The number of evening customers -/
def evening_customers : ℕ := 40

/-- The number of opening night customers -/
def opening_night_customers : ℕ := 58

/-- The total revenue in dollars -/
def total_revenue : ℚ := 1670

theorem matinee_ticket_cost :
  matinee_cost * matinee_customers +
  evening_cost * evening_customers +
  opening_night_cost * opening_night_customers +
  popcorn_cost * ((matinee_customers + evening_customers + opening_night_customers) / 2) =
  total_revenue :=
sorry

end NUMINAMATH_CALUDE_matinee_ticket_cost_l1197_119742


namespace NUMINAMATH_CALUDE_pigeon_hole_problem_l1197_119749

theorem pigeon_hole_problem (pigeonholes : ℕ) (pigeons : ℕ) : 
  (pigeons = 6 * pigeonholes + 3) →
  (pigeons + 5 = 8 * pigeonholes) →
  (pigeons = 27 ∧ pigeonholes = 4) := by
  sorry

end NUMINAMATH_CALUDE_pigeon_hole_problem_l1197_119749


namespace NUMINAMATH_CALUDE_boy_running_speed_l1197_119795

/-- Calculates the speed of a boy running around a square field -/
theorem boy_running_speed (side : ℝ) (time : ℝ) (speed_kmh : ℝ) : 
  side = 40 → 
  time = 64 → 
  speed_kmh = (4 * side / time) * 3.6 →
  speed_kmh = 9 := by
  sorry

#check boy_running_speed

end NUMINAMATH_CALUDE_boy_running_speed_l1197_119795


namespace NUMINAMATH_CALUDE_postman_speeds_theorem_l1197_119720

/-- Represents the speeds of the postman on different terrains -/
structure PostmanSpeeds where
  uphill : ℝ
  flat : ℝ
  downhill : ℝ

/-- Checks if the given speeds satisfy the journey conditions -/
def satisfiesConditions (speeds : PostmanSpeeds) : Prop :=
  let uphill := speeds.uphill
  let flat := speeds.flat
  let downhill := speeds.downhill
  (2 / uphill + 4 / flat + 3 / downhill = 2.267) ∧
  (3 / uphill + 4 / flat + 2 / downhill = 2.4) ∧
  (1 / uphill + 2 / flat + 1.5 / downhill = 1.158)

/-- Theorem stating that the specific speeds satisfy the journey conditions -/
theorem postman_speeds_theorem :
  satisfiesConditions { uphill := 3, flat := 4, downhill := 5 } := by
  sorry

#check postman_speeds_theorem

end NUMINAMATH_CALUDE_postman_speeds_theorem_l1197_119720


namespace NUMINAMATH_CALUDE_tank_filling_time_l1197_119768

/-- The time required to fill a tank with different valve combinations -/
theorem tank_filling_time 
  (fill_time_xyz : Real) 
  (fill_time_xz : Real) 
  (fill_time_yz : Real) 
  (h1 : fill_time_xyz = 2)
  (h2 : fill_time_xz = 4)
  (h3 : fill_time_yz = 3) :
  let rate_x := 1 / fill_time_xz - 1 / fill_time_xyz
  let rate_y := 1 / fill_time_yz - 1 / fill_time_xyz
  1 / (rate_x + rate_y) = 2.4 := by
  sorry

end NUMINAMATH_CALUDE_tank_filling_time_l1197_119768


namespace NUMINAMATH_CALUDE_sqrt_x_minus_9_meaningful_l1197_119702

theorem sqrt_x_minus_9_meaningful (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 9) ↔ x ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_9_meaningful_l1197_119702


namespace NUMINAMATH_CALUDE_sin_double_angle_special_case_l1197_119752

theorem sin_double_angle_special_case (φ : ℝ) :
  (7 : ℝ) / 13 + Real.sin φ = Real.cos φ →
  Real.sin (2 * φ) = 120 / 169 := by
sorry

end NUMINAMATH_CALUDE_sin_double_angle_special_case_l1197_119752


namespace NUMINAMATH_CALUDE_meaningful_expression_range_l1197_119707

theorem meaningful_expression_range (x : ℝ) :
  (∃ y : ℝ, y = Real.sqrt (1 - x) ∧ x + 2 ≠ 0) ↔ x ≤ 1 ∧ x ≠ -2 := by sorry

end NUMINAMATH_CALUDE_meaningful_expression_range_l1197_119707


namespace NUMINAMATH_CALUDE_smallest_perimeter_circle_circle_center_on_L_l1197_119769

-- Define the points A and B
def A : ℝ × ℝ := (1, -2)
def B : ℝ × ℝ := (-1, 4)

-- Define the line L
def L (x y : ℝ) : Prop := 2 * x - y - 4 = 0

-- Define a general circle equation
def isCircle (h k r : ℝ) (x y : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

-- Define a circle passing through two points
def circlePassingThrough (h k r : ℝ) : Prop :=
  isCircle h k r A.1 A.2 ∧ isCircle h k r B.1 B.2

-- Theorem for the smallest perimeter circle
theorem smallest_perimeter_circle :
  ∃ (h k r : ℝ), circlePassingThrough h k r ∧
  isCircle h k r = fun x y => x^2 + (y - 1)^2 = 10 :=
sorry

-- Theorem for the circle with center on line L
theorem circle_center_on_L :
  ∃ (h k r : ℝ), circlePassingThrough h k r ∧
  L h k ∧
  isCircle h k r = fun x y => (x - 3)^2 + (y - 2)^2 = 20 :=
sorry

end NUMINAMATH_CALUDE_smallest_perimeter_circle_circle_center_on_L_l1197_119769


namespace NUMINAMATH_CALUDE_min_xy_min_x_plus_y_l1197_119773

-- Define the conditions
def condition (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ 2 * x + 8 * y - x * y = 0

-- Theorem for the minimum value of xy
theorem min_xy (x y : ℝ) (h : condition x y) :
  x * y ≥ 64 ∧ ∃ x y, condition x y ∧ x * y = 64 :=
sorry

-- Theorem for the minimum value of x + y
theorem min_x_plus_y (x y : ℝ) (h : condition x y) :
  x + y ≥ 18 ∧ ∃ x y, condition x y ∧ x + y = 18 :=
sorry

end NUMINAMATH_CALUDE_min_xy_min_x_plus_y_l1197_119773


namespace NUMINAMATH_CALUDE_saxbridge_parade_max_members_l1197_119785

theorem saxbridge_parade_max_members :
  ∀ n : ℕ,
  (15 * n < 1200) →
  (15 * n) % 24 = 3 →
  (∀ m : ℕ, (15 * m < 1200) ∧ (15 * m) % 24 = 3 → 15 * m ≤ 15 * n) →
  15 * n = 1155 :=
sorry

end NUMINAMATH_CALUDE_saxbridge_parade_max_members_l1197_119785


namespace NUMINAMATH_CALUDE_employee_pay_l1197_119748

theorem employee_pay (x y z : ℝ) : 
  x + y + z = 900 →
  x = 1.2 * y →
  z = 0.8 * y →
  y = 300 := by
sorry

end NUMINAMATH_CALUDE_employee_pay_l1197_119748


namespace NUMINAMATH_CALUDE_only_one_true_statement_l1197_119728

/-- Two lines are non-coincident -/
def NonCoincidentLines (m n : Line) : Prop :=
  m ≠ n

/-- Two planes are non-coincident -/
def NonCoincidentPlanes (α β : Plane) : Prop :=
  α ≠ β

/-- A line is parallel to a plane -/
def LineParallelToPlane (l : Line) (p : Plane) : Prop :=
  sorry

/-- A line is perpendicular to a plane -/
def LinePerpendicularToPlane (l : Line) (p : Plane) : Prop :=
  sorry

/-- Two lines are parallel -/
def ParallelLines (l1 l2 : Line) : Prop :=
  sorry

/-- Two lines intersect -/
def LinesIntersect (l1 l2 : Line) : Prop :=
  sorry

/-- Two planes are perpendicular -/
def PerpendicularPlanes (p1 p2 : Plane) : Prop :=
  sorry

/-- Two lines are perpendicular -/
def PerpendicularLines (l1 l2 : Line) : Prop :=
  sorry

/-- Projection of a line onto a plane -/
def ProjectionOntoPlane (l : Line) (p : Plane) : Line :=
  sorry

theorem only_one_true_statement
  (m n : Line) (α β : Plane)
  (h_lines : NonCoincidentLines m n)
  (h_planes : NonCoincidentPlanes α β) :
  (LineParallelToPlane m α ∧ LineParallelToPlane n α → ¬LinesIntersect m n) ∨
  (LinePerpendicularToPlane m α ∧ LinePerpendicularToPlane n α → ParallelLines m n) ∨
  (PerpendicularPlanes α β ∧ PerpendicularLines m n ∧ LinePerpendicularToPlane m α → LinePerpendicularToPlane n β) ∨
  (PerpendicularLines (ProjectionOntoPlane m α) (ProjectionOntoPlane n α) → PerpendicularLines m n) :=
by sorry

end NUMINAMATH_CALUDE_only_one_true_statement_l1197_119728


namespace NUMINAMATH_CALUDE_ham_bread_percentage_l1197_119718

theorem ham_bread_percentage (bread_cost ham_cost cake_cost : ℚ) 
  (h1 : bread_cost = 50)
  (h2 : ham_cost = 150)
  (h3 : cake_cost = 200) :
  (bread_cost + ham_cost) / (bread_cost + ham_cost + cake_cost) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ham_bread_percentage_l1197_119718


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1197_119772

theorem inequality_equivalence (x : ℝ) : 
  (12 * x^2 + 24 * x - 75) / ((3 * x - 5) * (x + 5)) < 4 ↔ -5 < x ∧ x < 5/3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1197_119772


namespace NUMINAMATH_CALUDE_second_number_proof_l1197_119746

theorem second_number_proof (x : ℝ) : 3 + x + 333 + 3.33 = 369.63 → x = 30.3 := by
  sorry

end NUMINAMATH_CALUDE_second_number_proof_l1197_119746


namespace NUMINAMATH_CALUDE_youngest_age_in_office_l1197_119753

/-- Proves that in a group of 4 people whose ages form an arithmetic sequence,
    if the oldest person is 50 years old and the sum of their ages is 158 years,
    then the youngest person is 29 years old. -/
theorem youngest_age_in_office (ages : Fin 4 → ℕ) 
  (arithmetic_sequence : ∀ i j k : Fin 4, i < j → j < k → 
    ages j - ages i = ages k - ages j)
  (oldest_age : ages 3 = 50)
  (sum_of_ages : (Finset.univ.sum ages) = 158) :
  ages 0 = 29 := by
sorry

end NUMINAMATH_CALUDE_youngest_age_in_office_l1197_119753


namespace NUMINAMATH_CALUDE_unique_solution_l1197_119775

/-- Sum of digits function -/
def S (n : ℕ) : ℕ := sorry

/-- Main theorem -/
theorem unique_solution :
  ∃! n : ℕ, n + S n = 1964 ∧ n = 1945 := by sorry

end NUMINAMATH_CALUDE_unique_solution_l1197_119775


namespace NUMINAMATH_CALUDE_sin_sq_plus_4cos_max_value_l1197_119716

theorem sin_sq_plus_4cos_max_value (x : ℝ) : 
  Real.sin x ^ 2 + 4 * Real.cos x ≤ 4 := by
sorry

end NUMINAMATH_CALUDE_sin_sq_plus_4cos_max_value_l1197_119716


namespace NUMINAMATH_CALUDE_sin_40_tan_10_minus_sqrt_3_l1197_119770

theorem sin_40_tan_10_minus_sqrt_3 : 
  Real.sin (40 * π / 180) * (Real.tan (10 * π / 180) - Real.sqrt 3) = -1 := by
  sorry

end NUMINAMATH_CALUDE_sin_40_tan_10_minus_sqrt_3_l1197_119770


namespace NUMINAMATH_CALUDE_profit_percentage_calculation_l1197_119709

/-- Calculate the profit percentage given the selling price and cost price -/
theorem profit_percentage_calculation (selling_price cost_price : ℚ) :
  selling_price = 800 ∧ cost_price = 640 →
  (selling_price - cost_price) / cost_price * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_calculation_l1197_119709


namespace NUMINAMATH_CALUDE_max_xy_value_l1197_119733

theorem max_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 1) :
  xy ≤ 1/8 ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 2 * x + y = 1 ∧ x * y = 1/8 :=
sorry

end NUMINAMATH_CALUDE_max_xy_value_l1197_119733


namespace NUMINAMATH_CALUDE_triangle_abc_is_acute_l1197_119706

/-- A triangle is acute if all its angles are less than 90 degrees --/
def IsAcuteTriangle (a b c : ℝ) : Prop :=
  let cosA := (b^2 + c^2 - a^2) / (2*b*c)
  let cosB := (a^2 + c^2 - b^2) / (2*a*c)
  let cosC := (a^2 + b^2 - c^2) / (2*a*b)
  0 < cosA ∧ cosA < 1 ∧
  0 < cosB ∧ cosB < 1 ∧
  0 < cosC ∧ cosC < 1

theorem triangle_abc_is_acute :
  let a : ℝ := 9
  let b : ℝ := 10
  let c : ℝ := 12
  IsAcuteTriangle a b c := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_is_acute_l1197_119706


namespace NUMINAMATH_CALUDE_janet_number_problem_l1197_119724

theorem janet_number_problem (x : ℤ) : 
  2 * (x + 7) - 4 = 28 → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_janet_number_problem_l1197_119724


namespace NUMINAMATH_CALUDE_bracket_6_times_3_l1197_119747

-- Define the custom bracket operation
def bracket (x : ℤ) : ℤ :=
  if x % 2 = 0 then x / 2 + 1 else 2 * x + 1

-- Theorem statement
theorem bracket_6_times_3 : bracket 6 * bracket 3 = 28 := by
  sorry

end NUMINAMATH_CALUDE_bracket_6_times_3_l1197_119747


namespace NUMINAMATH_CALUDE_sheila_picnic_probability_l1197_119736

/-- The probability of Sheila attending the picnic given the weather conditions and transport strike possibility. -/
theorem sheila_picnic_probability :
  let p_rain : ℝ := 0.5
  let p_sunny : ℝ := 1 - p_rain
  let p_attend_if_rain : ℝ := 0.25
  let p_attend_if_sunny : ℝ := 0.8
  let p_transport_strike : ℝ := 0.1
  let p_attend : ℝ := (p_rain * p_attend_if_rain + p_sunny * p_attend_if_sunny) * (1 - p_transport_strike)
  p_attend = 0.4725 := by
  sorry

end NUMINAMATH_CALUDE_sheila_picnic_probability_l1197_119736


namespace NUMINAMATH_CALUDE_cotton_collection_rate_l1197_119776

/-- The amount of cotton (in kg) that can be collected by a given number of workers in 2 days -/
def cotton_collected (w : ℕ) : ℝ := w * 8

theorem cotton_collection_rate 
  (h1 : 3 * (48 / 4) = 3 * 12)  -- 3 workers collect 48 kg in 4 days
  (h2 : 9 * 8 = 72) :  -- 9 workers collect 72 kg in 2 days
  ∀ w : ℕ, cotton_collected w = w * 8 := by
  sorry

#check cotton_collection_rate

end NUMINAMATH_CALUDE_cotton_collection_rate_l1197_119776


namespace NUMINAMATH_CALUDE_family_ages_solution_l1197_119735

/-- Represents the current ages of Jennifer, Jordana, and James -/
structure FamilyAges where
  jennifer : ℕ
  jordana : ℕ
  james : ℕ

/-- Checks if the given ages satisfy the problem conditions -/
def satisfiesConditions (ages : FamilyAges) : Prop :=
  ages.jennifer + 20 = 40 ∧
  ages.jordana + 20 = 2 * (ages.jennifer + 20) ∧
  ages.james + 20 = (ages.jennifer + 20) + (ages.jordana + 20) - 10

theorem family_ages_solution :
  ∃ (ages : FamilyAges), satisfiesConditions ages ∧ ages.jordana = 60 ∧ ages.james = 90 :=
sorry

end NUMINAMATH_CALUDE_family_ages_solution_l1197_119735


namespace NUMINAMATH_CALUDE_new_ratio_after_subtraction_l1197_119737

theorem new_ratio_after_subtraction :
  let a : ℚ := 72
  let b : ℚ := 192
  let subtrahend : ℚ := 24
  (a / b = 3 / 8) →
  ((a - subtrahend) / (b - subtrahend) = 1 / (7/2)) :=
by sorry

end NUMINAMATH_CALUDE_new_ratio_after_subtraction_l1197_119737


namespace NUMINAMATH_CALUDE_stationery_cost_theorem_l1197_119780

/-- Calculates the total cost of stationery given the number of pencil boxes, pencils per box,
    pencil cost, pen cost, and additional pens ordered. -/
def total_stationery_cost (pencil_boxes : ℕ) (pencils_per_box : ℕ) (pencil_cost : ℕ) 
                          (pen_cost : ℕ) (additional_pens : ℕ) : ℕ :=
  let total_pencils := pencil_boxes * pencils_per_box
  let total_pens := 2 * total_pencils + additional_pens
  let pencil_total_cost := total_pencils * pencil_cost
  let pen_total_cost := total_pens * pen_cost
  pencil_total_cost + pen_total_cost

/-- Theorem stating that the total cost of stationery for the given conditions is $18300. -/
theorem stationery_cost_theorem : 
  total_stationery_cost 15 80 4 5 300 = 18300 := by
  sorry

end NUMINAMATH_CALUDE_stationery_cost_theorem_l1197_119780


namespace NUMINAMATH_CALUDE_ball_probabilities_l1197_119763

/-- Represents the total number of balls in the bag -/
def total_balls : ℕ := 6

/-- Represents the number of white balls in the bag -/
def white_balls : ℕ := 4

/-- Represents the number of red balls in the bag -/
def red_balls : ℕ := 2

/-- Represents the number of balls drawn -/
def drawn_balls : ℕ := 2

/-- Calculates the probability of drawing two red balls -/
def prob_two_red : ℚ := (red_balls * (red_balls - 1)) / (total_balls * (total_balls - 1))

/-- Calculates the probability of drawing at least one red ball -/
def prob_at_least_one_red : ℚ := 1 - (white_balls * (white_balls - 1)) / (total_balls * (total_balls - 1))

theorem ball_probabilities :
  prob_two_red = 1/15 ∧ prob_at_least_one_red = 3/5 := by sorry

end NUMINAMATH_CALUDE_ball_probabilities_l1197_119763


namespace NUMINAMATH_CALUDE_odd_function_negative_domain_l1197_119723

/-- An odd function defined on ℝ -/
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_negative_domain
  (f : ℝ → ℝ)
  (h_odd : odd_function f)
  (h_positive : ∀ x ≥ 0, f x = x^2 - 2*x) :
  ∀ x < 0, f x = -x^2 - 2*x :=
by sorry

end NUMINAMATH_CALUDE_odd_function_negative_domain_l1197_119723


namespace NUMINAMATH_CALUDE_parabola_circle_intersection_l1197_119765

/-- Parabola M: y^2 = 4x -/
def parabola_M (x y : ℝ) : Prop := y^2 = 4*x

/-- Circle N: (x-1)^2 + y^2 = r^2 -/
def circle_N (x y r : ℝ) : Prop := (x - 1)^2 + y^2 = r^2

/-- Line l passing through (1, 0) -/
def line_l (m x y : ℝ) : Prop := x = m * y + 1

/-- Condition for |AC| = |BD| -/
def equal_distances (y₁ y₂ y₃ y₄ : ℝ) : Prop := |y₁ - y₃| = |y₂ - y₄|

/-- Main theorem -/
theorem parabola_circle_intersection (r : ℝ) :
  (r > 0) →
  (∃ (m₁ m₂ m₃ : ℝ),
    (∀ (m : ℝ), m ≠ m₁ ∧ m ≠ m₂ ∧ m ≠ m₃ →
      ¬(∃ (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ),
        parabola_M x₁ y₁ ∧ parabola_M x₂ y₂ ∧
        circle_N x₃ y₃ r ∧ circle_N x₄ y₄ r ∧
        line_l m x₁ y₁ ∧ line_l m x₂ y₂ ∧
        line_l m x₃ y₃ ∧ line_l m x₄ y₄ ∧
        equal_distances y₁ y₂ y₃ y₄))) →
  r ≥ 3/2 :=
sorry

end NUMINAMATH_CALUDE_parabola_circle_intersection_l1197_119765


namespace NUMINAMATH_CALUDE_tangent_line_at_one_zero_l1197_119790

/-- The equation of the tangent line to y = x^3 - 2x + 1 at (1, 0) is y = x - 1 -/
theorem tangent_line_at_one_zero (x y : ℝ) : 
  (y = x^3 - 2*x + 1) → -- curve equation
  (1^3 - 2*1 + 1 = 0) → -- point (1, 0) lies on the curve
  (∀ t, (t - 1) * (3*1^2 - 2) = y - 0) → -- point-slope form of tangent line
  (y = x - 1) -- equation of tangent line
  := by sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_zero_l1197_119790


namespace NUMINAMATH_CALUDE_inequality_not_always_true_l1197_119708

theorem inequality_not_always_true (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  ¬ (∀ a b, a > b ∧ b > 0 → a + 1/a < b + 1/b) :=
by sorry

end NUMINAMATH_CALUDE_inequality_not_always_true_l1197_119708


namespace NUMINAMATH_CALUDE_magic_sum_order_8_l1197_119766

def magic_sum (n : ℕ) : ℕ :=
  let total_sum := n^2 * (n^2 + 1) / 2
  total_sum / n

theorem magic_sum_order_8 :
  magic_sum 8 = 260 :=
by sorry

end NUMINAMATH_CALUDE_magic_sum_order_8_l1197_119766


namespace NUMINAMATH_CALUDE_x_squared_inequality_l1197_119779

theorem x_squared_inequality (x : ℝ) (h : x^2 + x < 0) : x < x^2 ∧ x^2 < -x := by
  sorry

end NUMINAMATH_CALUDE_x_squared_inequality_l1197_119779


namespace NUMINAMATH_CALUDE_route_time_proof_l1197_119726

/-- Proves that the time to run a 5-mile route one way is 1 hour, given the round trip average speed and return speed. -/
theorem route_time_proof (route_length : ℝ) (avg_speed : ℝ) (return_speed : ℝ) 
  (h1 : route_length = 5)
  (h2 : avg_speed = 8)
  (h3 : return_speed = 20) :
  let t := (2 * route_length / avg_speed - route_length / return_speed)
  t = 1 := by sorry

end NUMINAMATH_CALUDE_route_time_proof_l1197_119726


namespace NUMINAMATH_CALUDE_geometric_sum_mod_500_l1197_119750

theorem geometric_sum_mod_500 : (Finset.sum (Finset.range 1001) (fun i => 3^i)) % 500 = 1 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sum_mod_500_l1197_119750


namespace NUMINAMATH_CALUDE_bills_final_money_is_411_l1197_119740

/-- Calculates the final amount of money Bill has after all transactions and expenses. -/
def bills_final_money : ℝ :=
  let merchant_a_sale := 8 * 9
  let merchant_b_sale := 15 * 11
  let merchant_c_sale := 25 * 8
  let passerby_sale := 12 * 7
  let total_income := merchant_a_sale + merchant_b_sale + merchant_c_sale + passerby_sale
  let fine := 80
  let protection_cost := 30
  let total_expenses := fine + protection_cost
  total_income - total_expenses

/-- Theorem stating that Bill's final amount of money is $411. -/
theorem bills_final_money_is_411 : bills_final_money = 411 := by
  sorry

end NUMINAMATH_CALUDE_bills_final_money_is_411_l1197_119740


namespace NUMINAMATH_CALUDE_element_in_intersection_complement_l1197_119734

theorem element_in_intersection_complement (S : Type) (A B : Set S) (a : S) :
  Set.Nonempty A →
  Set.Nonempty B →
  A ⊂ Set.univ →
  B ⊂ Set.univ →
  a ∈ A →
  a ∉ B →
  a ∈ A ∩ (Set.univ \ B) :=
by sorry

end NUMINAMATH_CALUDE_element_in_intersection_complement_l1197_119734


namespace NUMINAMATH_CALUDE_f_odd_iff_l1197_119774

/-- A function f is odd if f(-x) = -f(x) for all x in the domain of f. -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The function f(x) = x|x + a| + b -/
def f (a b : ℝ) (x : ℝ) : ℝ :=
  x * |x + a| + b

/-- The necessary and sufficient condition for f to be an odd function -/
theorem f_odd_iff (a b : ℝ) :
  IsOdd (f a b) ↔ a^2 + b^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_odd_iff_l1197_119774


namespace NUMINAMATH_CALUDE_team7_cups_l1197_119701

-- Define the number of teams
def num_teams : Nat := 7

-- Define the total amount of soup required
def total_soup : Nat := 2500

-- Define the amount made by the first team
def first_team : Nat := 450

-- Define the amount made by the second team
def second_team : Nat := 300

-- Define the relationship between teams 3+4 and team 7
def teams_34_7_relation (team7 : Nat) : Nat := 2 * team7

-- Define the relationship between teams 1+2 and teams 5+6
def teams_12_56_relation : Nat := first_team + second_team

-- Define the function to calculate the total soup made by all teams
def total_soup_made (team7 : Nat) : Nat :=
  first_team + second_team + teams_34_7_relation team7 + teams_12_56_relation + team7

-- Theorem stating that team 7 should prepare 334 cups to meet the total required
theorem team7_cups : ∃ (team7 : Nat), team7 = 334 ∧ total_soup_made team7 = total_soup := by
  sorry

end NUMINAMATH_CALUDE_team7_cups_l1197_119701


namespace NUMINAMATH_CALUDE_average_price_per_book_l1197_119778

theorem average_price_per_book (books1 books2 : ℕ) (price1 price2 : ℚ) :
  books1 = 40 →
  books2 = 20 →
  price1 = 600 →
  price2 = 240 →
  (price1 + price2) / (books1 + books2 : ℚ) = 14 :=
by sorry

end NUMINAMATH_CALUDE_average_price_per_book_l1197_119778


namespace NUMINAMATH_CALUDE_larger_number_proof_l1197_119743

theorem larger_number_proof (L S : ℕ) (h1 : L > S) (h2 : L - S = 2342) (h3 : L = 9 * S + 23) : L = 2624 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l1197_119743


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l1197_119760

theorem absolute_value_inequality (x : ℝ) : 
  (1 ≤ |x - 2| ∧ |x - 2| ≤ 7) ↔ ((-5 ≤ x ∧ x ≤ 1) ∨ (3 ≤ x ∧ x ≤ 9)) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l1197_119760


namespace NUMINAMATH_CALUDE_max_value_problem_l1197_119764

theorem max_value_problem (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) 
  (h4 : a^2 + b^2 + c^2 = 1) :
  (2 * a * b * Real.sqrt 3 + 2 * a * c) ≤ Real.sqrt 3 ∧ 
  ∃ a₀ b₀ c₀ : ℝ, 0 ≤ a₀ ∧ 0 ≤ b₀ ∧ 0 ≤ c₀ ∧ a₀^2 + b₀^2 + c₀^2 = 1 ∧
    2 * a₀ * b₀ * Real.sqrt 3 + 2 * a₀ * c₀ = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_max_value_problem_l1197_119764


namespace NUMINAMATH_CALUDE_last_student_number_l1197_119729

def skip_pattern (n : ℕ) : ℕ := 3 * n - 1

def student_number (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => skip_pattern (student_number n)

theorem last_student_number :
  ∃ (k : ℕ), student_number k = 242 ∧ 
  ∀ (m : ℕ), m > k → student_number m > 500 := by
sorry

end NUMINAMATH_CALUDE_last_student_number_l1197_119729


namespace NUMINAMATH_CALUDE_ellipse_and_line_intersection_l1197_119725

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the line l
def line_l (x y : ℝ) : Prop := y = x - Real.sqrt 3

theorem ellipse_and_line_intersection :
  -- Conditions
  (∀ x y, ellipse_C x y → (x = 0 ∧ y = 0) → False) →  -- center at origin
  (∃ c > 0, ∀ x y, ellipse_C x y → x^2 / 4 + y^2 / c^2 = 1) →  -- standard form
  (ellipse_C 1 (Real.sqrt 3 / 2)) →  -- point on ellipse
  -- Conclusions
  (∀ x y, ellipse_C x y ↔ x^2 / 4 + y^2 = 1) ∧  -- equation of C
  (∃ x₁ x₂ y₁ y₂ : ℝ,
    ellipse_C x₁ y₁ ∧ ellipse_C x₂ y₂ ∧
    line_l x₁ y₁ ∧ line_l x₂ y₂ ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = (8/5)^2) :=  -- length of AB
by sorry

end NUMINAMATH_CALUDE_ellipse_and_line_intersection_l1197_119725


namespace NUMINAMATH_CALUDE_square_ratio_side_length_sum_l1197_119784

theorem square_ratio_side_length_sum (area_ratio : ℚ) :
  area_ratio = 135 / 45 →
  ∃ (a b c : ℕ), 
    (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) ∧
    (Real.sqrt (area_ratio) = (a * Real.sqrt b) / c) ∧
    (a + b + c = 5) := by
  sorry

end NUMINAMATH_CALUDE_square_ratio_side_length_sum_l1197_119784


namespace NUMINAMATH_CALUDE_f_properties_l1197_119754

noncomputable def f (x : ℝ) : ℝ := Real.sin (Real.pi / 2 + x) * Real.cos (Real.pi / 2 - x)

theorem f_properties :
  (∀ x₁ x₂ : ℝ, x₁ = -x₂ → f x₁ = -f x₂) ∧
  (∃ T : ℝ, T > 0 ∧ T < 2 * Real.pi ∧ ∀ x : ℝ, f (x + T) = f x) ∧
  (∀ x y : ℝ, -Real.pi/4 ≤ x ∧ x < y ∧ y ≤ Real.pi/4 → f x < f y) ∧
  (∀ x : ℝ, f (3 * Real.pi / 2 - x) = f (3 * Real.pi / 2 + x)) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1197_119754


namespace NUMINAMATH_CALUDE_gumball_sale_revenue_l1197_119755

theorem gumball_sale_revenue (num_gumballs : ℕ) (price_per_gumball : ℕ) : 
  num_gumballs = 4 → price_per_gumball = 8 → num_gumballs * price_per_gumball = 32 := by
  sorry

end NUMINAMATH_CALUDE_gumball_sale_revenue_l1197_119755


namespace NUMINAMATH_CALUDE_inequalities_comparison_l1197_119741

theorem inequalities_comparison (a b : ℝ) (h : a > b) : (a - 3 > b - 3) ∧ (-4*a < -4*b) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_comparison_l1197_119741


namespace NUMINAMATH_CALUDE_total_jumps_l1197_119783

/-- Given that Ronald jumped 157 times and Rupert jumped 86 more times than Ronald,
    prove that the total number of jumps by both is 400. -/
theorem total_jumps (ronald_jumps : ℕ) (rupert_extra_jumps : ℕ) 
  (h1 : ronald_jumps = 157)
  (h2 : rupert_extra_jumps = 86) : 
  ronald_jumps + (ronald_jumps + rupert_extra_jumps) = 400 := by
  sorry

end NUMINAMATH_CALUDE_total_jumps_l1197_119783


namespace NUMINAMATH_CALUDE_value_of_a_l1197_119705

/-- Given that 4 * ((a * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 3200.0000000000005,
    prove that a is approximately equal to 3.6 -/
theorem value_of_a (a : ℝ) : 
  4 * ((a * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 3200.0000000000005 → 
  ∃ ε > 0, |a - 3.6| < ε := by
sorry

end NUMINAMATH_CALUDE_value_of_a_l1197_119705


namespace NUMINAMATH_CALUDE_sequence_property_l1197_119793

theorem sequence_property (a : ℕ → ℤ) 
  (h : ∀ n : ℕ, a 1 + a (2 * n - 1) = 4 * n - 6) : 
  a 7 = 11 := by
sorry

end NUMINAMATH_CALUDE_sequence_property_l1197_119793


namespace NUMINAMATH_CALUDE_alien_eggs_conversion_l1197_119732

/-- Converts a number from base 7 to base 10 -/
def base7ToBase10 (n : ℕ) : ℕ :=
  (n / 100) * 7^2 + ((n / 10) % 10) * 7^1 + (n % 10) * 7^0

/-- The number of eggs laid by the alien creature in base 7 -/
def alienEggsBase7 : ℕ := 215

theorem alien_eggs_conversion :
  base7ToBase10 alienEggsBase7 = 110 := by
  sorry

end NUMINAMATH_CALUDE_alien_eggs_conversion_l1197_119732


namespace NUMINAMATH_CALUDE_largest_integer_with_remainder_l1197_119700

theorem largest_integer_with_remainder : ∃ n : ℕ, n = 95 ∧ 
  n < 100 ∧ 
  n % 7 = 4 ∧ 
  ∀ m : ℕ, m < 100 → m % 7 = 4 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_with_remainder_l1197_119700


namespace NUMINAMATH_CALUDE_function_c_injective_l1197_119798

theorem function_c_injective (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a + 2 / a = b + 2 / b → a = b := by sorry

end NUMINAMATH_CALUDE_function_c_injective_l1197_119798


namespace NUMINAMATH_CALUDE_students_in_both_groups_l1197_119713

theorem students_in_both_groups 
  (total : ℕ) 
  (math : ℕ) 
  (english : ℕ) 
  (h1 : total = 52) 
  (h2 : math = 32) 
  (h3 : english = 40) : 
  total = math + english - 20 :=
by sorry

end NUMINAMATH_CALUDE_students_in_both_groups_l1197_119713


namespace NUMINAMATH_CALUDE_correct_product_with_decimals_l1197_119771

theorem correct_product_with_decimals (x y : ℚ) (z : ℕ) : 
  x = 0.035 → y = 3.84 → z = 13440 → x * y = 0.1344 := by
  sorry

end NUMINAMATH_CALUDE_correct_product_with_decimals_l1197_119771


namespace NUMINAMATH_CALUDE_ruth_shared_apples_l1197_119757

/-- The number of apples Ruth shared with Peter -/
def apples_shared (initial : ℕ) (remaining : ℕ) : ℕ :=
  initial - remaining

/-- Theorem stating that Ruth shared 5 apples with Peter -/
theorem ruth_shared_apples : apples_shared 89 84 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ruth_shared_apples_l1197_119757


namespace NUMINAMATH_CALUDE_mnp_value_l1197_119710

theorem mnp_value (a b x y : ℝ) (m n p : ℤ) 
  (h1 : a^8*x*y - a^7*y - a^6*x = a^5*(b^5 - 1))
  (h2 : (a^m*x - a^n)*(a^p*y - a^3) = a^5*b^5) :
  m * n * p = 12 := by
  sorry

end NUMINAMATH_CALUDE_mnp_value_l1197_119710


namespace NUMINAMATH_CALUDE_log_inequality_l1197_119782

theorem log_inequality (a b : ℝ) (h1 : a > b) (h2 : b > 0) : Real.log a > Real.log b := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l1197_119782


namespace NUMINAMATH_CALUDE_expansion_equals_cube_l1197_119739

theorem expansion_equals_cube : 16^3 + 3*(16^2)*2 + 3*16*(2^2) + 2^3 = 5832 := by
  sorry

end NUMINAMATH_CALUDE_expansion_equals_cube_l1197_119739


namespace NUMINAMATH_CALUDE_sqrt_neg_two_squared_l1197_119722

theorem sqrt_neg_two_squared : Real.sqrt ((-2)^2) = 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_neg_two_squared_l1197_119722


namespace NUMINAMATH_CALUDE_problem_statement_l1197_119704

theorem problem_statement (a b : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : a + b - 1 / (2 * a) - 2 / b = 3 / 2) 
  (h4 : a < 1) : 
  b > 2 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l1197_119704


namespace NUMINAMATH_CALUDE_base5_divisibility_by_29_l1197_119711

def base5ToDecimal (a b c d : ℕ) : ℕ := a * 5^3 + b * 5^2 + c * 5^1 + d * 5^0

def isDivisibleBy29 (n : ℕ) : Prop := ∃ k : ℕ, n = 29 * k

theorem base5_divisibility_by_29 (y : ℕ) :
  isDivisibleBy29 (base5ToDecimal 4 2 y 3) ↔ y = 4 := by sorry

end NUMINAMATH_CALUDE_base5_divisibility_by_29_l1197_119711


namespace NUMINAMATH_CALUDE_closest_point_sum_l1197_119758

/-- The point (a, b) on the line y = -3x + 10 that is closest to (16, 8) satisfies a + b = 8.8 -/
theorem closest_point_sum (a b : ℝ) : 
  (b = -3 * a + 10) →  -- Mouse path equation
  (∀ x y : ℝ, y = -3 * x + 10 → (x - 16)^2 + (y - 8)^2 ≥ (a - 16)^2 + (b - 8)^2) →  -- (a, b) is closest to (16, 8)
  a + b = 8.8 := by
  sorry

end NUMINAMATH_CALUDE_closest_point_sum_l1197_119758


namespace NUMINAMATH_CALUDE_correct_division_result_l1197_119762

theorem correct_division_result (dividend : ℕ) 
  (h1 : dividend / 87 = 24) 
  (h2 : dividend % 87 = 0) : 
  dividend / 36 = 58 := by
sorry

end NUMINAMATH_CALUDE_correct_division_result_l1197_119762


namespace NUMINAMATH_CALUDE_card_value_decrease_l1197_119796

theorem card_value_decrease (v : ℝ) (h : v > 0) : 
  let value_after_first_year := v * (1 - 0.5)
  let value_after_second_year := value_after_first_year * (1 - 0.1)
  let total_decrease := (v - value_after_second_year) / v
  total_decrease = 0.55
:= by sorry

end NUMINAMATH_CALUDE_card_value_decrease_l1197_119796


namespace NUMINAMATH_CALUDE_least_multiple_of_next_three_primes_after_5_l1197_119797

def next_three_primes_after_5 : List Nat := [7, 11, 13]

theorem least_multiple_of_next_three_primes_after_5 :
  (∀ p ∈ next_three_primes_after_5, Nat.Prime p) →
  (∀ p ∈ next_three_primes_after_5, p > 5) →
  (∀ n < 1001, ∃ p ∈ next_three_primes_after_5, ¬(p ∣ n)) →
  (∀ p ∈ next_three_primes_after_5, p ∣ 1001) :=
by sorry

end NUMINAMATH_CALUDE_least_multiple_of_next_three_primes_after_5_l1197_119797


namespace NUMINAMATH_CALUDE_triangle_point_distance_inequality_triangle_point_distance_equality_condition_l1197_119751

-- Define a triangle ABC
variable (A B C : ℝ × ℝ)

-- Define a point P inside or on the boundary of triangle ABC
variable (P : ℝ × ℝ)

-- Define distances from P to sides of the triangle
def da : ℝ := sorry
def db : ℝ := sorry
def dc : ℝ := sorry

-- Define distances from P to vertices of the triangle
def AP : ℝ := sorry
def BP : ℝ := sorry
def CP : ℝ := sorry

-- Theorem statement
theorem triangle_point_distance_inequality :
  (max AP (max BP CP)) ≥ Real.sqrt (da^2 + db^2 + dc^2) :=
sorry

-- Equality condition
theorem triangle_point_distance_equality_condition :
  (max AP (max BP CP)) = Real.sqrt (da^2 + db^2 + dc^2) ↔
  (A = B ∧ B = C) ∧ P = ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3) :=
sorry

end NUMINAMATH_CALUDE_triangle_point_distance_inequality_triangle_point_distance_equality_condition_l1197_119751


namespace NUMINAMATH_CALUDE_total_hockey_games_l1197_119745

/-- The number of hockey games in a season -/
def hockey_games_in_season (games_per_month : ℕ) (months_in_season : ℕ) : ℕ :=
  games_per_month * months_in_season

/-- Theorem stating that there are 182 hockey games in the season -/
theorem total_hockey_games :
  hockey_games_in_season 13 14 = 182 := by
  sorry

end NUMINAMATH_CALUDE_total_hockey_games_l1197_119745


namespace NUMINAMATH_CALUDE_tangent_circles_parallelism_l1197_119721

-- Define the types for our points and circles
variable (Point : Type) (Circle : Type)

-- Define the basic geometric relations
variable (on_circle : Point → Circle → Prop)
variable (on_line : Point → Point → Point → Prop)
variable (between : Point → Point → Point → Prop)
variable (tangent : Point → Point → Circle → Prop)
variable (intersect : Circle → Circle → Point → Point → Prop)
variable (cuts : Point → Point → Circle → Point → Prop)
variable (parallel : Point → Point → Point → Point → Prop)

-- Define our specific points and circles
variable (A B C P Q R S X Y Z : Point)
variable (C1 C2 : Circle)

-- State the theorem
theorem tangent_circles_parallelism 
  (h1 : intersect C1 C2 A B)
  (h2 : on_line A B C ∧ between A B C)
  (h3 : on_circle P C1 ∧ on_circle Q C2)
  (h4 : tangent C P C1 ∧ tangent C Q C2)
  (h5 : ¬on_circle P C2 ∧ ¬on_circle Q C1)
  (h6 : cuts P Q C1 R ∧ cuts P Q C2 S)
  (h7 : R ≠ P ∧ R ≠ Q ∧ R ≠ B ∧ S ≠ P ∧ S ≠ Q ∧ S ≠ B)
  (h8 : cuts C R C1 X ∧ cuts C S C2 Y)
  (h9 : on_line X Y Z) :
  parallel S Z Q X ↔ parallel P Z R X :=
sorry

end NUMINAMATH_CALUDE_tangent_circles_parallelism_l1197_119721


namespace NUMINAMATH_CALUDE_binomial_coefficient_identity_l1197_119719

theorem binomial_coefficient_identity (n k : ℕ) (h : k ≤ n) :
  k * Nat.choose n k = n * Nat.choose (n - 1) (k - 1) := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_identity_l1197_119719


namespace NUMINAMATH_CALUDE_inverse_proportion_decrease_l1197_119767

theorem inverse_proportion_decrease (x y : ℝ) (k : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y = k) :
  let x_new := 1.1 * x
  let y_new := k / x_new
  (y - y_new) / y = 1 / 11 := by sorry

end NUMINAMATH_CALUDE_inverse_proportion_decrease_l1197_119767


namespace NUMINAMATH_CALUDE_fraction_simplification_l1197_119731

theorem fraction_simplification :
  (1/2 - 1/3) / ((3/7) * (2/8)) = 14/9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1197_119731


namespace NUMINAMATH_CALUDE_inequality_proof_l1197_119712

theorem inequality_proof (x y : ℝ) (hx : |x| < 1) (hy : |y| < 1) :
  (1 - x^2)⁻¹ + (1 - y^2)⁻¹ ≥ 2 * (1 - x*y)⁻¹ := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1197_119712


namespace NUMINAMATH_CALUDE_factorial_difference_l1197_119791

theorem factorial_difference : Nat.factorial 9 - Nat.factorial 8 = 322560 := by
  sorry

end NUMINAMATH_CALUDE_factorial_difference_l1197_119791


namespace NUMINAMATH_CALUDE_tables_needed_l1197_119781

theorem tables_needed (total_children : ℕ) (children_per_table : ℕ) (h1 : total_children = 152) (h2 : children_per_table = 7) :
  ∃ (tables : ℕ), tables = 22 ∧ tables * children_per_table ≥ total_children ∧ (tables - 1) * children_per_table < total_children :=
by sorry

end NUMINAMATH_CALUDE_tables_needed_l1197_119781


namespace NUMINAMATH_CALUDE_max_true_statements_l1197_119730

theorem max_true_statements : ∃ x : ℝ, 
  (-1 < x ∧ x < 1) ∧ 
  (-1 < x^3 ∧ x^3 < 1) ∧ 
  (0 < x ∧ x < 1) ∧ 
  (0 < x^2 ∧ x^2 < 1) ∧ 
  (0 < x^3 - x^2 ∧ x^3 - x^2 < 1) := by
  sorry

end NUMINAMATH_CALUDE_max_true_statements_l1197_119730


namespace NUMINAMATH_CALUDE_one_neither_prime_nor_composite_l1197_119727

theorem one_neither_prime_nor_composite : 
  ¬(Nat.Prime 1) ∧ ¬(∃ a b : Nat, a > 1 ∧ b > 1 ∧ a * b = 1) := by
  sorry

end NUMINAMATH_CALUDE_one_neither_prime_nor_composite_l1197_119727

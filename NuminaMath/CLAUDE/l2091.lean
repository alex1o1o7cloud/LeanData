import Mathlib

namespace sqrt_fifth_power_sixth_l2091_209163

theorem sqrt_fifth_power_sixth : (((5 : ℝ) ^ (1/2)) ^ 5) ^ (1/2) ^ 6 = 125 * (125 : ℝ) ^ (1/4) := by
  sorry

end sqrt_fifth_power_sixth_l2091_209163


namespace largest_prime_factors_difference_l2091_209104

theorem largest_prime_factors_difference (n : Nat) (h : n = 165033) :
  ∃ (p q : Nat), Nat.Prime p ∧ Nat.Prime q ∧ p > q ∧
  (∀ (r : Nat), Nat.Prime r ∧ r ∣ n → r ≤ p) ∧
  (∀ (r : Nat), Nat.Prime r ∧ r ∣ n ∧ r ≠ p → r ≤ q) ∧
  p - q = 140 := by
  sorry

#eval 165033

end largest_prime_factors_difference_l2091_209104


namespace math_homework_pages_l2091_209125

-- Define the variables
def reading_pages : ℕ := 4
def problems_per_page : ℕ := 3
def total_problems : ℕ := 30

-- Define the theorem
theorem math_homework_pages :
  ∃ (math_pages : ℕ), 
    math_pages * problems_per_page + reading_pages * problems_per_page = total_problems ∧
    math_pages = 6 := by
  sorry

end math_homework_pages_l2091_209125


namespace probability_no_growth_pies_l2091_209168

def total_pies : ℕ := 6
def growth_pies : ℕ := 2
def given_pies : ℕ := 3

theorem probability_no_growth_pies :
  let shrink_pies := total_pies - growth_pies
  let prob_mary_no_growth := (shrink_pies.choose given_pies : ℚ) / (total_pies.choose given_pies : ℚ)
  let prob_alice_no_growth := 1 - (1 - prob_mary_no_growth)
  prob_mary_no_growth + prob_alice_no_growth = 2/5 := by
  sorry

end probability_no_growth_pies_l2091_209168


namespace parabola_c_value_l2091_209165

/-- A parabola passing through three given points -/
structure Parabola where
  b : ℝ
  c : ℝ
  eq : ℝ → ℝ := λ x => x^2 + b*x + c
  point1 : eq 2 = 12
  point2 : eq (-2) = 0
  point3 : eq 4 = 40

/-- The value of c for the parabola passing through (2, 12), (-2, 0), and (4, 40) is 2 -/
theorem parabola_c_value (p : Parabola) : p.c = 2 := by
  sorry

#check parabola_c_value

end parabola_c_value_l2091_209165


namespace cookie_distribution_l2091_209119

theorem cookie_distribution (boxes : ℕ) (classes : ℕ) 
  (h1 : boxes = 3) (h2 : classes = 4) :
  (boxes : ℚ) / classes = 3 / 4 := by sorry

end cookie_distribution_l2091_209119


namespace company_a_profit_share_l2091_209123

/-- Prove that Company A's share of combined profits is 60% given the conditions -/
theorem company_a_profit_share :
  ∀ (total_profit : ℝ) (company_b_profit : ℝ) (company_a_profit : ℝ),
    company_b_profit = 0.4 * total_profit →
    company_b_profit = 60000 →
    company_a_profit = 90000 →
    company_a_profit / total_profit = 0.6 := by
  sorry

end company_a_profit_share_l2091_209123


namespace cut_cube_theorem_l2091_209110

/-- Represents a cube that has been cut into smaller cubes -/
structure CutCube where
  -- The number of smaller cubes painted on exactly 2 faces
  two_face_cubes : ℕ
  -- The total number of smaller cubes created
  total_cubes : ℕ

/-- Theorem stating that a cube cut into equal smaller cubes with 12 two-face cubes results in 27 total cubes -/
theorem cut_cube_theorem (c : CutCube) (h : c.two_face_cubes = 12) : c.total_cubes = 27 := by
  sorry


end cut_cube_theorem_l2091_209110


namespace zora_shorter_than_brixton_l2091_209158

/-- Proves that Zora is 8 inches shorter than Brixton given the conditions of the problem -/
theorem zora_shorter_than_brixton :
  ∀ (zora itzayana zara brixton : ℕ),
    itzayana = zora + 4 →
    zara = 64 →
    brixton = zara →
    (zora + itzayana + zara + brixton) / 4 = 61 →
    brixton - zora = 8 := by
  sorry

end zora_shorter_than_brixton_l2091_209158


namespace exists_counterexample_l2091_209147

-- Define a structure for the set S with the binary operation *
structure BinarySystem where
  S : Type u
  op : S → S → S
  at_least_two_elements : ∃ (a b : S), a ≠ b
  property : ∀ (a b : S), op a (op b a) = b

-- State the theorem
theorem exists_counterexample (B : BinarySystem) :
  ∃ (a b : B.S), B.op (B.op a b) a ≠ a := by sorry

end exists_counterexample_l2091_209147


namespace geometric_sequence_property_l2091_209159

/-- A geometric sequence with a_1 * a_3 = a_4 = 4 has a_6 = 8 -/
theorem geometric_sequence_property (a : ℕ → ℝ) (h_geometric : ∀ n, a (n + 1) / a n = a 2 / a 1) 
  (h_condition : a 1 * a 3 = a 4 ∧ a 4 = 4) : a 6 = 8 := by
  sorry

end geometric_sequence_property_l2091_209159


namespace solution_subset_nonpositive_l2091_209195

/-- The solution set of |x| > ax + 1 is a subset of {x | x ≤ 0} if and only if a ≥ 1 -/
theorem solution_subset_nonpositive (a : ℝ) :
  (∀ x : ℝ, |x| > a * x + 1 → x ≤ 0) ↔ a ≥ 1 := by
  sorry

end solution_subset_nonpositive_l2091_209195


namespace range_of_7a_minus_5b_l2091_209180

theorem range_of_7a_minus_5b (a b : ℝ) 
  (h1 : 5 ≤ a - b ∧ a - b ≤ 27) 
  (h2 : 6 ≤ a + b ∧ a + b ≤ 30) : 
  ∃ (x : ℝ), 36 ≤ 7*a - 5*b ∧ 7*a - 5*b ≤ 192 ∧
  (∀ (y : ℝ), 36 ≤ y ∧ y ≤ 192 → ∃ (a' b' : ℝ), 
    (5 ≤ a' - b' ∧ a' - b' ≤ 27) ∧ 
    (6 ≤ a' + b' ∧ a' + b' ≤ 30) ∧ 
    y = 7*a' - 5*b') :=
by sorry

end range_of_7a_minus_5b_l2091_209180


namespace least_reducible_fraction_l2091_209126

/-- A fraction a/b is reducible if gcd(a,b) > 1 -/
def IsReducible (a b : ℤ) : Prop := Int.gcd a b > 1

/-- The numerator of our fraction -/
def Numerator (m : ℕ) : ℤ := m - 17

/-- The denominator of our fraction -/
def Denominator (m : ℕ) : ℤ := 6 * m + 7

theorem least_reducible_fraction :
  (∀ m : ℕ, m < 126 → ¬(IsReducible (Numerator m) (Denominator m))) ∧
  IsReducible (Numerator 126) (Denominator 126) := by
  sorry

end least_reducible_fraction_l2091_209126


namespace min_value_sum_of_reciprocals_l2091_209156

theorem min_value_sum_of_reciprocals (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 3) :
  (1 / (1 + a) + 4 / (4 + b)) ≥ 9/8 ∧
  ∃ (a₀ b₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ a₀ + b₀ = 3 ∧ 1 / (1 + a₀) + 4 / (4 + b₀) = 9/8 :=
sorry

end min_value_sum_of_reciprocals_l2091_209156


namespace ratio_a_to_c_l2091_209115

theorem ratio_a_to_c (a b c d : ℝ) 
  (hab : a / b = 5 / 4)
  (hcd : c / d = 4 / 1)
  (hdb : d / b = 1 / 8) :
  a / c = 5 / 2 := by
sorry

end ratio_a_to_c_l2091_209115


namespace max_sum_of_functions_l2091_209148

-- Define the functions f and g
def f : ℝ → ℝ := sorry
def g : ℝ → ℝ := sorry

-- State the theorem
theorem max_sum_of_functions :
  (∀ x, -6 ≤ f x ∧ f x ≤ 4) →
  (∀ x, -3 ≤ g x ∧ g x ≤ 2) →
  (∃ d, ∀ x, f x + g x ≤ d ∧ ∃ y, f y + g y = d) →
  ∃ d, d = 6 ∧ ∀ x, f x + g x ≤ d ∧ ∃ y, f y + g y = d :=
by sorry

end max_sum_of_functions_l2091_209148


namespace total_notes_count_l2091_209130

/-- Given a total amount of 192 rupees in equal numbers of 1-rupee, 5-rupee, and 10-rupee notes,
    prove that the total number of notes is 36. -/
theorem total_notes_count (total_amount : ℕ) (note_count : ℕ) : 
  total_amount = 192 →
  note_count * 1 + note_count * 5 + note_count * 10 = total_amount →
  3 * note_count = 36 :=
by
  sorry

end total_notes_count_l2091_209130


namespace product_evaluation_l2091_209101

theorem product_evaluation : 
  (2 + 3) * (2^2 + 3^2) * (2^4 + 3^4) * (2^8 + 3^8) * (2^16 + 3^16) * 
  (2^32 + 3^32) * (2^64 + 3^64) * (2 + 1) = 3^129 - 3 * 2^128 := by
  sorry

end product_evaluation_l2091_209101


namespace stating_days_worked_when_net_zero_l2091_209139

/-- Represents the number of days in the work period -/
def total_days : ℕ := 30

/-- Represents the daily wage in su -/
def daily_wage : ℕ := 24

/-- Represents the daily penalty for skipping work in su -/
def daily_penalty : ℕ := 6

/-- 
Theorem stating that if a worker's net earnings are zero after the work period,
given the specified daily wage and penalty, then the number of days worked is 6.
-/
theorem days_worked_when_net_zero : 
  ∀ (days_worked : ℕ), 
    days_worked ≤ total_days →
    (daily_wage * days_worked - daily_penalty * (total_days - days_worked) = 0) →
    days_worked = 6 := by
  sorry

end stating_days_worked_when_net_zero_l2091_209139


namespace cost_is_ten_l2091_209118

/-- Represents the cost of piano lessons -/
structure LessonCost where
  lessons_per_week : ℕ
  lesson_duration_hours : ℕ
  weeks : ℕ
  total_earnings : ℕ

/-- Calculates the cost per half-hour of teaching -/
def cost_per_half_hour (lc : LessonCost) : ℚ :=
  lc.total_earnings / (2 * lc.lessons_per_week * lc.lesson_duration_hours * lc.weeks)

/-- Theorem: The cost per half-hour of teaching is $10 -/
theorem cost_is_ten (lc : LessonCost) 
  (h1 : lc.lessons_per_week = 1)
  (h2 : lc.lesson_duration_hours = 1)
  (h3 : lc.weeks = 5)
  (h4 : lc.total_earnings = 100) : 
  cost_per_half_hour lc = 10 := by
  sorry

end cost_is_ten_l2091_209118


namespace triangle_altitude_on_square_diagonal_l2091_209146

theorem triangle_altitude_on_square_diagonal (s : ℝ) (h : s > 0) :
  let square_area := s^2
  let diagonal := s * Real.sqrt 2
  let triangle_area := (1/2) * diagonal * altitude
  ∃ altitude : ℝ, 
    (square_area = triangle_area) ∧ 
    (altitude = s * Real.sqrt 2) :=
by sorry

end triangle_altitude_on_square_diagonal_l2091_209146


namespace rally_attendance_l2091_209186

/-- Represents the rally attendance problem --/
def RallyAttendance (total_receipts : ℚ) (before_rally_tickets : ℕ) 
  (before_rally_price : ℚ) (at_door_price : ℚ) : Prop :=
  ∃ (at_door_tickets : ℕ),
    total_receipts = before_rally_price * before_rally_tickets + at_door_price * at_door_tickets ∧
    before_rally_tickets + at_door_tickets = 750

/-- Theorem stating the total attendance at the rally --/
theorem rally_attendance :
  RallyAttendance (1706.25 : ℚ) 475 2 (2.75 : ℚ) :=
by
  sorry


end rally_attendance_l2091_209186


namespace toothpick_grid_theorem_l2091_209124

/-- Calculates the number of unique toothpicks in a rectangular grid frame. -/
def unique_toothpicks (height width : ℕ) : ℕ :=
  let horizontal_toothpicks := (height + 1) * width
  let vertical_toothpicks := height * (width + 1)
  let intersections := (height + 1) * (width + 1)
  horizontal_toothpicks + vertical_toothpicks - intersections

/-- Theorem stating that a 15x8 toothpick grid uses 119 unique toothpicks. -/
theorem toothpick_grid_theorem :
  unique_toothpicks 15 8 = 119 := by
  sorry

#eval unique_toothpicks 15 8

end toothpick_grid_theorem_l2091_209124


namespace quadratic_property_l2091_209182

def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_property (a b c : ℝ) (h1 : a ≠ 0) :
  (quadratic a b c 2 = 0.35) →
  (quadratic a b c 4 = 0.35) →
  (quadratic a b c 5 = 3) →
  (a + b + c) * (-b / a) = 18 := by
sorry

end quadratic_property_l2091_209182


namespace notebooks_distribution_l2091_209107

theorem notebooks_distribution (C : ℕ) (N : ℕ) : 
  (N / C = C / 8) →  -- Each child got one-eighth of the number of children in notebooks
  (N / (C / 2) = 16) →  -- If number of children halved, each would get 16 notebooks
  N = 512 := by  -- Total notebooks distributed is 512
sorry

end notebooks_distribution_l2091_209107


namespace quadratic_roots_sum_product_l2091_209184

theorem quadratic_roots_sum_product (m n : ℝ) : 
  (∃ x y : ℝ, 2 * x^2 - m * x + n = 0 ∧ 2 * y^2 - m * y + n = 0 ∧ x + y = 10 ∧ x * y = 24) →
  m + n = 68 := by
sorry

end quadratic_roots_sum_product_l2091_209184


namespace and_or_relationship_l2091_209169

theorem and_or_relationship (p q : Prop) : 
  ((p ∧ q) → (p ∨ q)) ∧ ¬((p ∨ q) → (p ∧ q)) := by
  sorry

end and_or_relationship_l2091_209169


namespace car_price_proof_l2091_209113

-- Define the original cost price
def original_price : ℝ := 52325.58

-- Define the first sale price (14% loss)
def first_sale_price : ℝ := original_price * 0.86

-- Define the second sale price (20% gain from first sale)
def second_sale_price : ℝ := 54000

-- Theorem statement
theorem car_price_proof :
  (first_sale_price * 1.2 = second_sale_price) ∧
  (original_price > 0) ∧
  (first_sale_price > 0) ∧
  (second_sale_price > 0) :=
sorry

end car_price_proof_l2091_209113


namespace perpendicular_line_through_point_l2091_209191

/-- Given a line L1 with equation 3x - 6y = 9 and a point P (2, -3),
    prove that the line L2 with equation y = -2x + 1 is perpendicular to L1 and passes through P. -/
theorem perpendicular_line_through_point (x y : ℝ) :
  let L1 : ℝ → ℝ → Prop := λ x y => 3 * x - 6 * y = 9
  let L2 : ℝ → ℝ → Prop := λ x y => y = -2 * x + 1
  let P : ℝ × ℝ := (2, -3)
  (∀ x y, L1 x y → (y = 1/2 * x - 3/2)) →  -- Slope of L1 is 1/2
  (L2 P.1 P.2) →  -- L2 passes through P
  (∀ x₁ y₁ x₂ y₂, L1 x₁ y₁ → L2 x₂ y₂ → (x₂ - x₁) * (y₂ - y₁) = -1) →  -- L1 and L2 are perpendicular
  ∃ m b, ∀ x y, L2 x y ↔ y = m * x + b ∧ m = -2 ∧ b = 1 :=
by sorry

end perpendicular_line_through_point_l2091_209191


namespace souvenir_october_price_l2091_209183

/-- Represents the selling price and sales data of a souvenir --/
structure SouvenirSales where
  september_price : ℝ
  september_revenue : ℝ
  october_discount : ℝ
  october_volume_increase : ℕ
  october_revenue_increase : ℝ

/-- Calculates the October price of a souvenir given its sales data --/
def october_price (s : SouvenirSales) : ℝ :=
  s.september_price * (1 - s.october_discount)

/-- Theorem stating the October price of the souvenir --/
theorem souvenir_october_price (s : SouvenirSales) 
  (h1 : s.september_revenue = 2000)
  (h2 : s.october_discount = 0.1)
  (h3 : s.october_volume_increase = 20)
  (h4 : s.october_revenue_increase = 700) :
  october_price s = 45 := by
  sorry

end souvenir_october_price_l2091_209183


namespace class_size_with_sports_participation_l2091_209167

/-- The number of students in a class with given sports participation. -/
theorem class_size_with_sports_participation
  (football : ℕ)
  (long_tennis : ℕ)
  (both : ℕ)
  (neither : ℕ)
  (h1 : football = 26)
  (h2 : long_tennis = 20)
  (h3 : both = 17)
  (h4 : neither = 6) :
  football + long_tennis - both + neither = 35 := by
  sorry

end class_size_with_sports_participation_l2091_209167


namespace intersection_value_l2091_209173

theorem intersection_value (m n : ℝ) (h1 : n = 3 / m) (h2 : n = m + 1) :
  (m - n)^2 * (1 / n - 1 / m) = -1 / 3 := by
  sorry

end intersection_value_l2091_209173


namespace f_monotonic_k_range_l2091_209143

def f (k : ℝ) (x : ℝ) : ℝ := 4 * x^2 - k * x - 8

def monotonic_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y ∨ (∀ z, a ≤ z ∧ z ≤ b → f z = f x)

theorem f_monotonic_k_range :
  ∀ k : ℝ, (monotonic_on (f k) 1 2) → k ≤ 8 ∨ k ≥ 16 := by
  sorry

end f_monotonic_k_range_l2091_209143


namespace project_budget_l2091_209102

theorem project_budget (total_spent : ℕ) (over_budget : ℕ) : 
  total_spent = 6580 →
  over_budget = 280 →
  ∃ (monthly_allocation : ℕ),
    monthly_allocation * 6 = total_spent - over_budget ∧
    monthly_allocation * 12 = 12600 := by
  sorry

end project_budget_l2091_209102


namespace expression_simplification_l2091_209141

theorem expression_simplification (x : ℝ) (h : x = 2) :
  (1 / (x + 1) - 1) / ((x^3 - x) / (x^2 + 2*x + 1)) = -1 := by
  sorry

end expression_simplification_l2091_209141


namespace range_of_m_for_decreasing_function_l2091_209157

/-- A function f is decreasing on ℝ -/
def DecreasingOnReals (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

/-- Given a decreasing function f on ℝ, if f(m-1) > f(2m-1), then m > 0 -/
theorem range_of_m_for_decreasing_function (f : ℝ → ℝ) (m : ℝ) 
  (h_decreasing : DecreasingOnReals f) (h_inequality : f (m - 1) > f (2 * m - 1)) : 
  m > 0 :=
by sorry

end range_of_m_for_decreasing_function_l2091_209157


namespace union_A_B_intersection_A_C_nonempty_implies_a_gt_one_l2091_209193

-- Define the sets A, B, and C
def A : Set ℝ := {x | 1 ≤ x ∧ x < 6}
def B : Set ℝ := {x | 2 < x ∧ x < 9}
def C (a : ℝ) : Set ℝ := {x | x < a}

-- Theorem for part (1)
theorem union_A_B : A ∪ B = {x : ℝ | 1 ≤ x ∧ x < 9} := by sorry

-- Theorem for part (2)
theorem intersection_A_C_nonempty_implies_a_gt_one (a : ℝ) :
  (A ∩ C a).Nonempty → a > 1 := by sorry

end union_A_B_intersection_A_C_nonempty_implies_a_gt_one_l2091_209193


namespace point_transformation_l2091_209108

def rotate90CounterClockwise (x y cx cy : ℝ) : ℝ × ℝ :=
  (cx - (y - cy), cy + (x - cx))

def reflectAboutNegativeDiagonal (x y : ℝ) : ℝ × ℝ :=
  (y, x)

theorem point_transformation (a b : ℝ) :
  let (x₁, y₁) := rotate90CounterClockwise a b 1 5
  let (x₂, y₂) := reflectAboutNegativeDiagonal x₁ y₁
  (x₂ = -6 ∧ y₂ = 3) → b - a = -5 := by sorry

end point_transformation_l2091_209108


namespace intersection_complement_equals_seven_l2091_209196

def U : Finset Nat := {4,5,6,7,8}
def M : Finset Nat := {5,8}
def N : Finset Nat := {1,3,5,7,9}

theorem intersection_complement_equals_seven :
  (N ∩ (U \ M)) = {7} := by sorry

end intersection_complement_equals_seven_l2091_209196


namespace gcf_60_72_l2091_209199

theorem gcf_60_72 : Nat.gcd 60 72 = 12 := by
  sorry

end gcf_60_72_l2091_209199


namespace divisors_of_360_l2091_209170

theorem divisors_of_360 : ∃ (d : Finset Nat), 
  (∀ x ∈ d, x ∣ 360) ∧ 
  (∀ x : Nat, x ∣ 360 → x ∈ d) ∧
  d.card = 24 ∧
  d.sum id = 1170 := by
  sorry

end divisors_of_360_l2091_209170


namespace four_must_be_in_A_l2091_209109

/-- A type representing the circles in the diagram -/
inductive Circle : Type
  | A | B | C | D | E | F | G

/-- The set of numbers to be placed in the circles -/
def NumberSet : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}

/-- A function that assigns a number to each circle -/
def Assignment := Circle → ℕ

/-- Predicate to check if an assignment is valid -/
def IsValidAssignment (f : Assignment) : Prop :=
  (∀ n ∈ NumberSet, ∃ c : Circle, f c = n) ∧
  (∀ c : Circle, f c ∈ NumberSet) ∧
  (f Circle.A + f Circle.D + f Circle.E = 
   f Circle.A + f Circle.C + f Circle.F) ∧
  (f Circle.A + f Circle.D + f Circle.E = 
   f Circle.A + f Circle.B + f Circle.G) ∧
  (f Circle.D + f Circle.C + f Circle.B = 
   f Circle.E + f Circle.F + f Circle.G)

theorem four_must_be_in_A (f : Assignment) 
  (h : IsValidAssignment f) : 
  f Circle.A = 4 ∧ f Circle.E ≠ 4 := by
  sorry

end four_must_be_in_A_l2091_209109


namespace shipment_average_weight_l2091_209103

/-- Represents the weight distribution of boxes in a shipment. -/
structure Shipment where
  total_boxes : ℕ
  light_boxes : ℕ
  heavy_boxes : ℕ
  light_weight : ℕ
  heavy_weight : ℕ

/-- Calculates the average weight of boxes after removing some heavy boxes. -/
def new_average (s : Shipment) (removed : ℕ) : ℚ :=
  (s.light_boxes * s.light_weight + (s.heavy_boxes - removed) * s.heavy_weight) /
  (s.light_boxes + s.heavy_boxes - removed)

/-- Theorem stating the average weight of boxes in the shipment. -/
theorem shipment_average_weight (s : Shipment) :
  s.total_boxes = 20 ∧
  s.light_weight = 10 ∧
  s.heavy_weight = 20 ∧
  s.light_boxes + s.heavy_boxes = s.total_boxes ∧
  new_average s 10 = 16 →
  (s.light_boxes * s.light_weight + s.heavy_boxes * s.heavy_weight) / s.total_boxes = 39/2 := by
  sorry

#check shipment_average_weight

end shipment_average_weight_l2091_209103


namespace library_visitors_on_sunday_l2091_209160

/-- Proves that the average number of visitors on Sundays is 140 given the specified conditions --/
theorem library_visitors_on_sunday (
  total_days : Nat) 
  (sunday_count : Nat)
  (avg_visitors_per_day : ℝ)
  (avg_visitors_non_sunday : ℝ)
  (h1 : total_days = 30)
  (h2 : sunday_count = 5)
  (h3 : avg_visitors_per_day = 90)
  (h4 : avg_visitors_non_sunday = 80)
  : ℝ :=
by
  -- Proof goes here
  sorry

#check library_visitors_on_sunday

end library_visitors_on_sunday_l2091_209160


namespace unique_fixed_point_of_odd_symmetries_l2091_209197

/-- A point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Central symmetry transformation about a point -/
def centralSymmetry (center : Point) (p : Point) : Point :=
  { x := 2 * center.x - p.x, y := 2 * center.y - p.y }

/-- Composition of central symmetries -/
def compositeSymmetry (centers : List Point) : Point → Point :=
  centers.foldl (λ f center p => f (centralSymmetry center p)) id

theorem unique_fixed_point_of_odd_symmetries (n : ℕ) :
  let m := 2 * n + 1
  ∀ (midpoints : List Point),
    midpoints.length = m →
    ∃! (fixedPoint : Point), compositeSymmetry midpoints fixedPoint = fixedPoint :=
by
  sorry

#check unique_fixed_point_of_odd_symmetries

end unique_fixed_point_of_odd_symmetries_l2091_209197


namespace rectangle_area_from_circular_wire_l2091_209112

/-- The area of a rectangle formed by bending a circular wire -/
theorem rectangle_area_from_circular_wire (r : ℝ) (ratio_l : ℝ) (ratio_b : ℝ) : 
  r = 3.5 → 
  ratio_l = 6 → 
  ratio_b = 5 → 
  let circumference := 2 * π * r
  let length := (circumference * ratio_l) / (2 * (ratio_l + ratio_b))
  let breadth := (circumference * ratio_b) / (2 * (ratio_l + ratio_b))
  length * breadth = (735 * π^2) / 242 := by
  sorry

#check rectangle_area_from_circular_wire

end rectangle_area_from_circular_wire_l2091_209112


namespace inequality_and_function_property_l2091_209129

def f (x : ℝ) : ℝ := |x - 1|

theorem inequality_and_function_property :
  (∀ x : ℝ, f (x - 1) + f (x + 3) ≥ 6 ↔ x ≤ -3 ∨ x ≥ 3) ∧
  (∀ a b : ℝ, |a| < 1 → |b| < 1 → a ≠ 0 → f (a * b) > |a| * f (b / a)) := by
  sorry

end inequality_and_function_property_l2091_209129


namespace exam_failure_percentage_l2091_209153

theorem exam_failure_percentage :
  let total_candidates : ℕ := 2000
  let girls : ℕ := 900
  let boys : ℕ := total_candidates - girls
  let boys_pass_rate : ℚ := 34 / 100
  let girls_pass_rate : ℚ := 32 / 100
  let passed_candidates : ℚ := boys_pass_rate * boys + girls_pass_rate * girls
  let failed_candidates : ℚ := total_candidates - passed_candidates
  let failure_percentage : ℚ := failed_candidates / total_candidates * 100
  failure_percentage = 669 / 10 := by sorry

end exam_failure_percentage_l2091_209153


namespace fraction_problem_l2091_209190

theorem fraction_problem : ∃ x : ℚ, x * 1206 = 3 * 134 ∧ x = 1 / 3 := by
  sorry

end fraction_problem_l2091_209190


namespace divisibility_by_twelve_l2091_209114

theorem divisibility_by_twelve (n : Nat) : n ≤ 9 → (512 * 10 + n) % 12 = 0 ↔ n = 4 := by
  sorry

end divisibility_by_twelve_l2091_209114


namespace subset_condition_empty_intersection_l2091_209166

def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}

theorem subset_condition (m : ℝ) : B m ⊆ A ↔ m ≤ 3 := by sorry

theorem empty_intersection (m : ℝ) : A ∩ B m = ∅ ↔ m < 2 ∨ m > 4 := by sorry

end subset_condition_empty_intersection_l2091_209166


namespace rectangular_plot_area_l2091_209135

/-- The area of a rectangular plot with length thrice its breadth and breadth of 14 meters is 588 square meters. -/
theorem rectangular_plot_area (breadth : ℝ) (length : ℝ) (area : ℝ) : 
  breadth = 14 →
  length = 3 * breadth →
  area = length * breadth →
  area = 588 := by
  sorry

end rectangular_plot_area_l2091_209135


namespace clothes_spending_fraction_l2091_209172

theorem clothes_spending_fraction (initial_amount : ℝ) (fraction_clothes : ℝ) : 
  initial_amount = 249.99999999999994 →
  (3/4 : ℝ) * (4/5 : ℝ) * (1 - fraction_clothes) * initial_amount = 100 →
  fraction_clothes = 11/15 := by
  sorry

end clothes_spending_fraction_l2091_209172


namespace coffee_purchase_problem_l2091_209105

/-- Given a gift card balance, coffee price per pound, and remaining balance,
    calculate the number of pounds of coffee purchased. -/
def coffee_pounds_purchased (gift_card_balance : ℚ) (coffee_price_per_pound : ℚ) (remaining_balance : ℚ) : ℚ :=
  (gift_card_balance - remaining_balance) / coffee_price_per_pound

theorem coffee_purchase_problem :
  let gift_card_balance : ℚ := 70
  let coffee_price_per_pound : ℚ := 8.58
  let remaining_balance : ℚ := 35.68
  coffee_pounds_purchased gift_card_balance coffee_price_per_pound remaining_balance = 4 := by
  sorry

end coffee_purchase_problem_l2091_209105


namespace brittany_second_test_score_l2091_209151

/-- Proves that given the conditions of Brittany's test scores, her second test score must be 83. -/
theorem brittany_second_test_score
  (first_test_score : ℝ)
  (first_test_weight : ℝ)
  (second_test_weight : ℝ)
  (final_weighted_average : ℝ)
  (h1 : first_test_score = 78)
  (h2 : first_test_weight = 0.4)
  (h3 : second_test_weight = 0.6)
  (h4 : final_weighted_average = 81)
  (h5 : first_test_weight + second_test_weight = 1) :
  ∃ (second_test_score : ℝ),
    first_test_weight * first_test_score + second_test_weight * second_test_score = final_weighted_average ∧
    second_test_score = 83 :=
by sorry

end brittany_second_test_score_l2091_209151


namespace binomial_150_150_l2091_209178

theorem binomial_150_150 : Nat.choose 150 150 = 1 := by
  sorry

end binomial_150_150_l2091_209178


namespace mitten_plug_difference_l2091_209106

theorem mitten_plug_difference (mittens : ℕ) (added_plugs : ℕ) (total_plugs : ℕ) : 
  mittens = 150 → added_plugs = 30 → total_plugs = 400 →
  (total_plugs / 2 - added_plugs) - mittens = 20 := by
  sorry

end mitten_plug_difference_l2091_209106


namespace pages_left_after_tuesday_l2091_209198

def pages_read_monday : ℕ := 15
def extra_pages_tuesday : ℕ := 16
def total_pages : ℕ := 64

def pages_left : ℕ := total_pages - (pages_read_monday + (pages_read_monday + extra_pages_tuesday))

theorem pages_left_after_tuesday : pages_left = 18 := by
  sorry

end pages_left_after_tuesday_l2091_209198


namespace trigonometric_equation_has_solution_l2091_209142

theorem trigonometric_equation_has_solution :
  ∃ x : ℝ, 2 * Real.sin x * Real.cos (3 * Real.pi / 2 + x) -
           3 * Real.sin (Real.pi - x) * Real.cos x +
           Real.sin (Real.pi / 2 + x) * Real.cos x = 0 := by
  sorry

end trigonometric_equation_has_solution_l2091_209142


namespace find_n_l2091_209175

theorem find_n : ∃ n : ℤ, n + (n + 1) + (n + 2) = 15 ∧ n = 4 := by
  sorry

end find_n_l2091_209175


namespace remainder_of_binary_div_four_l2091_209145

def binary_number : ℕ := 3005 -- 110110111101₂ in decimal

theorem remainder_of_binary_div_four :
  binary_number % 4 = 1 := by sorry

end remainder_of_binary_div_four_l2091_209145


namespace madeline_class_hours_l2091_209121

/-- Calculates the number of hours Madeline spends in class per week -/
def hours_in_class (hours_per_day : ℕ) (days_per_week : ℕ) 
  (homework_hours_per_day : ℕ) (sleep_hours_per_day : ℕ) 
  (work_hours_per_week : ℕ) (leftover_hours : ℕ) : ℕ :=
  hours_per_day * days_per_week - 
  (homework_hours_per_day * days_per_week + 
   sleep_hours_per_day * days_per_week + 
   work_hours_per_week + 
   leftover_hours)

theorem madeline_class_hours : 
  hours_in_class 24 7 4 8 20 46 = 18 := by
  sorry

end madeline_class_hours_l2091_209121


namespace banana_purchase_l2091_209185

theorem banana_purchase (banana_price apple_price total_weight total_cost : ℚ)
  (h1 : banana_price = 76 / 100)
  (h2 : apple_price = 59 / 100)
  (h3 : total_weight = 30)
  (h4 : total_cost = 1940 / 100) :
  ∃ (banana_weight : ℚ),
    banana_weight + (total_weight - banana_weight) = total_weight ∧
    banana_price * banana_weight + apple_price * (total_weight - banana_weight) = total_cost ∧
    banana_weight = 10 := by
sorry

end banana_purchase_l2091_209185


namespace square_odd_implies_odd_l2091_209128

theorem square_odd_implies_odd (n : ℤ) : Odd (n^2) → Odd n := by
  sorry

end square_odd_implies_odd_l2091_209128


namespace integral_f_minus_pi_to_zero_l2091_209187

noncomputable def f (x : ℝ) : ℝ := x + Real.sin x

theorem integral_f_minus_pi_to_zero :
  ∫ x in Set.Icc (-Real.pi) 0, f x = -2 - (1/2) * Real.pi^2 := by
  sorry

end integral_f_minus_pi_to_zero_l2091_209187


namespace distance_inequality_l2091_209174

theorem distance_inequality (a : ℝ) : 
  (|a - 1| < 3) → (-2 < a ∧ a < 4) := by
  sorry

end distance_inequality_l2091_209174


namespace student_arrangement_count_l2091_209122

/-- The number of students in the row -/
def total_students : ℕ := 4

/-- The number of students that must stand next to each other -/
def adjacent_students : ℕ := 2

/-- The number of different arrangements of students -/
def num_arrangements : ℕ := 12

/-- 
Theorem: Given 4 students standing in a row, where 2 specific students 
must stand next to each other, the number of different arrangements is 12.
-/
theorem student_arrangement_count :
  (total_students = 4) →
  (adjacent_students = 2) →
  (num_arrangements = 12) :=
by sorry

end student_arrangement_count_l2091_209122


namespace lace_makers_combined_time_l2091_209117

theorem lace_makers_combined_time (t1 t2 T : ℚ) : 
  t1 = 8 → t2 = 13 → (1 / t1 + 1 / t2) * T = 1 → T = 104 / 21 := by
  sorry

end lace_makers_combined_time_l2091_209117


namespace class_A_student_count_l2091_209138

theorem class_A_student_count :
  ∀ (girls boys : ℕ),
    girls = 25 →
    girls = boys + 3 →
    girls + boys = 47 :=
by sorry

end class_A_student_count_l2091_209138


namespace find_S_value_l2091_209154

/-- Represents the relationship between R, S, and T -/
def relationship (R S T : ℝ) : Prop :=
  ∃ (c : ℝ), R = c * S / T

theorem find_S_value (R₁ S₁ T₁ R₂ T₂ : ℝ) :
  relationship R₁ S₁ T₁ →
  R₁ = 4/3 →
  S₁ = 3/7 →
  T₁ = 9/14 →
  R₂ = Real.sqrt 48 →
  T₂ = Real.sqrt 75 →
  ∃ (S₂ : ℝ), relationship R₂ S₂ T₂ ∧ S₂ = 30 :=
by sorry

end find_S_value_l2091_209154


namespace candy_game_solution_l2091_209179

theorem candy_game_solution (total_questions : ℕ) (correct_reward : ℕ) (incorrect_penalty : ℕ) 
  (h1 : total_questions = 50)
  (h2 : correct_reward = 7)
  (h3 : incorrect_penalty = 3) :
  ∃ correct_answers : ℕ, 
    correct_answers * correct_reward = (total_questions - correct_answers) * incorrect_penalty ∧ 
    correct_answers = 15 := by
  sorry

end candy_game_solution_l2091_209179


namespace four_points_left_of_origin_l2091_209133

theorem four_points_left_of_origin : 
  let points : List ℝ := [-(-8), (-1)^2023, -(3^2), -1-11, -2/5]
  (points.filter (· < 0)).length = 4 := by
sorry

end four_points_left_of_origin_l2091_209133


namespace simplify_and_evaluate_expression_l2091_209171

theorem simplify_and_evaluate_expression (x : ℚ) (h : x = -4) :
  (x^2 / (x - 1) - x + 1) / ((4 * x^2 - 4 * x + 1) / (1 - x)) = 1 / 9 := by
  sorry

end simplify_and_evaluate_expression_l2091_209171


namespace sum_of_consecutive_iff_not_power_of_two_l2091_209132

def is_sum_of_consecutive (n : ℕ) : Prop :=
  ∃ (a k : ℕ), k > 0 ∧ n = (k * (2 * a + k - 1)) / 2

def is_power_of_two (n : ℕ) : Prop :=
  ∃ (s : ℕ), n = 2^s

theorem sum_of_consecutive_iff_not_power_of_two (n : ℕ) :
  ¬(is_sum_of_consecutive n) ↔ is_power_of_two n :=
sorry

end sum_of_consecutive_iff_not_power_of_two_l2091_209132


namespace garden_perimeter_l2091_209140

theorem garden_perimeter :
  ∀ (length breadth perimeter : ℝ),
    length = 258 →
    breadth = 82 →
    perimeter = 2 * (length + breadth) →
    perimeter = 680 := by
  sorry

end garden_perimeter_l2091_209140


namespace square_transformation_l2091_209162

-- Define the square in the xy-plane
def square_vertices : List (ℝ × ℝ) := [(0, 0), (1, 0), (1, 1), (0, 1)]

-- Define the transformation
def transform (p : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p
  (x^2 - y^2, 2*x*y)

-- Define the transformed square
def transformed_square : List (ℝ × ℝ) := square_vertices.map transform

-- Define the expected shape in the uv-plane
def expected_shape (u v : ℝ) : Prop :=
  (u = 0 ∧ 0 ≤ v ∧ v ≤ 1) ∨  -- Line segment from (0,0) to (1,0)
  (u = 1 - v^2/4 ∧ 0 ≤ v ∧ v ≤ 2) ∨  -- Parabola segment
  (u = v^2/4 - 1 ∧ 0 ≤ v ∧ v ≤ 2) ∨  -- Parabola segment
  (v = 0 ∧ -1 ≤ u ∧ u ≤ 0)  -- Line segment from (-1,0) to (0,0)

theorem square_transformation :
  ∀ (u v : ℝ), (∃ (x y : ℝ), (x, y) ∈ square_vertices ∧ transform (x, y) = (u, v)) ↔ expected_shape u v := by
  sorry

end square_transformation_l2091_209162


namespace power_multiplication_specific_power_multiplication_l2091_209137

theorem power_multiplication (a b c : ℕ) : (10 : ℕ) ^ a * (10 : ℕ) ^ b = (10 : ℕ) ^ (a + b) := by
  sorry

theorem specific_power_multiplication : (10 : ℕ) ^ 65 * (10 : ℕ) ^ 64 = (10 : ℕ) ^ 129 := by
  sorry

end power_multiplication_specific_power_multiplication_l2091_209137


namespace south_american_stamps_cost_l2091_209192

def brazil_stamp_price : ℚ := 7 / 100
def peru_stamp_price : ℚ := 5 / 100
def brazil_50s_stamps : ℕ := 5
def brazil_60s_stamps : ℕ := 9
def peru_50s_stamps : ℕ := 12
def peru_60s_stamps : ℕ := 8

def total_south_american_stamps_cost : ℚ :=
  (brazil_stamp_price * (brazil_50s_stamps + brazil_60s_stamps)) +
  (peru_stamp_price * (peru_50s_stamps + peru_60s_stamps))

theorem south_american_stamps_cost :
  total_south_american_stamps_cost = 198 / 100 := by
  sorry

end south_american_stamps_cost_l2091_209192


namespace kara_forgotten_doses_l2091_209177

/-- The number of times Kara takes medication per day -/
def doses_per_day : ℕ := 3

/-- The amount of water in ounces Kara drinks with each dose -/
def water_per_dose : ℕ := 4

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The total amount of water in ounces Kara drank with her medication over two weeks -/
def total_water_drunk : ℕ := 160

/-- The number of times Kara forgot to take her medication on one day in the second week -/
def forgotten_doses : ℕ := 2

theorem kara_forgotten_doses :
  (doses_per_day * water_per_dose * days_in_week * 2) - total_water_drunk = forgotten_doses * water_per_dose :=
by sorry

end kara_forgotten_doses_l2091_209177


namespace smallest_divisible_by_12_20_6_sixty_divisible_by_12_20_6_sixty_is_smallest_l2091_209136

theorem smallest_divisible_by_12_20_6 : ∀ n : ℕ, n > 0 → (12 ∣ n) → (20 ∣ n) → (6 ∣ n) → n ≥ 60 := by
  sorry

theorem sixty_divisible_by_12_20_6 : (12 ∣ 60) ∧ (20 ∣ 60) ∧ (6 ∣ 60) := by
  sorry

theorem sixty_is_smallest :
  ∀ n : ℕ, n > 0 → (12 ∣ n) → (20 ∣ n) → (6 ∣ n) → n = 60 := by
  sorry

end smallest_divisible_by_12_20_6_sixty_divisible_by_12_20_6_sixty_is_smallest_l2091_209136


namespace inverse_proportion_increasing_l2091_209149

theorem inverse_proportion_increasing (m : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ < 0 ∧ x₁ < x₂ →
    (1 - 2*m) / x₁ < (1 - 2*m) / x₂) ↔
  m > 1/2 := by
sorry

end inverse_proportion_increasing_l2091_209149


namespace distance_between_points_l2091_209164

def point1 : ℝ × ℝ := (-3, 1)
def point2 : ℝ × ℝ := (4, -9)

theorem distance_between_points : 
  Real.sqrt ((point2.1 - point1.1)^2 + (point2.2 - point1.2)^2) = Real.sqrt 149 := by
  sorry

end distance_between_points_l2091_209164


namespace exp_addition_property_l2091_209181

-- Define the exponential function
noncomputable def f (x : ℝ) : ℝ := Real.exp x

-- State the theorem
theorem exp_addition_property (x y : ℝ) : f (x + y) = f x * f y := by
  sorry

end exp_addition_property_l2091_209181


namespace subtracted_value_l2091_209152

theorem subtracted_value (x y : ℤ) (h1 : x = 122) (h2 : 2 * x - y = 106) : y = 138 := by
  sorry

end subtracted_value_l2091_209152


namespace set_operations_l2091_209188

def A : Set ℝ := {x | x^2 + 3*x - 4 > 0}
def B : Set ℝ := {x | x^2 - x - 6 < 0}

theorem set_operations :
  (A ∩ B = {x | 1 < x ∧ x < 3}) ∧
  (Set.compl (A ∩ B) = {x | x ≤ 1 ∨ x ≥ 3}) ∧
  (A ∪ Set.compl B = {x | x ≤ -2 ∨ x > 1}) := by
  sorry

end set_operations_l2091_209188


namespace rectangular_plot_breadth_l2091_209131

theorem rectangular_plot_breadth :
  ∀ (length breadth area : ℝ),
  length = 3 * breadth →
  area = length * breadth →
  area = 2028 →
  breadth = 26 :=
by
  sorry

end rectangular_plot_breadth_l2091_209131


namespace sum_of_reciprocals_B_l2091_209116

def B : Set ℕ := {n : ℕ | ∀ p : ℕ, Nat.Prime p → p ∣ n → p = 2 ∨ p = 3 ∨ p = 7}

theorem sum_of_reciprocals_B : ∑' (n : B), (1 : ℚ) / n = 7 / 2 := by sorry

end sum_of_reciprocals_B_l2091_209116


namespace logging_time_is_ten_months_l2091_209127

/-- Represents the forest and logging scenario --/
structure LoggingScenario where
  forestLength : ℕ
  forestWidth : ℕ
  treesPerSquareMile : ℕ
  loggersCount : ℕ
  treesPerLoggerPerDay : ℕ
  daysPerMonth : ℕ

/-- Calculates the number of months required to cut down all trees --/
def monthsToLogForest (scenario : LoggingScenario) : ℚ :=
  let totalArea := scenario.forestLength * scenario.forestWidth
  let totalTrees := totalArea * scenario.treesPerSquareMile
  let treesPerDay := scenario.loggersCount * scenario.treesPerLoggerPerDay
  (totalTrees : ℚ) / (treesPerDay * scenario.daysPerMonth)

/-- Theorem stating that it takes 10 months to log the forest under given conditions --/
theorem logging_time_is_ten_months :
  let scenario : LoggingScenario := {
    forestLength := 4,
    forestWidth := 6,
    treesPerSquareMile := 600,
    loggersCount := 8,
    treesPerLoggerPerDay := 6,
    daysPerMonth := 30
  }
  monthsToLogForest scenario = 10 := by sorry

end logging_time_is_ten_months_l2091_209127


namespace stickers_per_page_l2091_209134

theorem stickers_per_page (total_pages : ℕ) (remaining_stickers : ℕ) : 
  total_pages = 12 →
  remaining_stickers = 220 →
  (total_pages - 1) * (remaining_stickers / (total_pages - 1)) = remaining_stickers →
  remaining_stickers / (total_pages - 1) = 20 := by
sorry

end stickers_per_page_l2091_209134


namespace probability_of_white_ball_l2091_209189

-- Define the box contents
def white_balls : ℕ := 1
def red_balls : ℕ := 2

-- Define the total number of balls
def total_balls : ℕ := white_balls + red_balls

-- Define the probability of drawing a white ball
def prob_white : ℚ := white_balls / total_balls

-- Theorem statement
theorem probability_of_white_ball : prob_white = 1/3 := by
  sorry

end probability_of_white_ball_l2091_209189


namespace xyz_sum_mod_9_l2091_209111

theorem xyz_sum_mod_9 (x y z : ℕ) : 
  0 < x ∧ x < 9 ∧
  0 < y ∧ y < 9 ∧
  0 < z ∧ z < 9 ∧
  (x * y * z) % 9 = 1 ∧
  (7 * z) % 9 = 4 ∧
  (8 * y) % 9 = (5 + y) % 9 →
  (x + y + z) % 9 = 7 := by
sorry

end xyz_sum_mod_9_l2091_209111


namespace exam_time_allocation_l2091_209144

theorem exam_time_allocation (total_questions : ℕ) (exam_duration_hours : ℕ) 
  (type_a_problems : ℕ) (h1 : total_questions = 200) (h2 : exam_duration_hours = 3) 
  (h3 : type_a_problems = 25) :
  let exam_duration_minutes : ℕ := exam_duration_hours * 60
  let type_b_problems : ℕ := total_questions - type_a_problems
  let x : ℚ := (exam_duration_minutes : ℚ) / (type_a_problems * 2 + type_b_problems)
  (2 * x * type_a_problems : ℚ) = 40 := by
  sorry

#check exam_time_allocation

end exam_time_allocation_l2091_209144


namespace red_ball_probability_l2091_209150

/-- Given a bag of balls with the following properties:
  * There are n total balls
  * There are m white balls
  * The probability of drawing at least one red ball when two balls are drawn is 3/5
  * The expected number of white balls in 6 draws with replacement is 4
  Prove that the probability of drawing a red ball on the second draw,
  given that the first draw was red, is 1/5. -/
theorem red_ball_probability (n m : ℕ) 
  (h1 : 1 - (m.choose 2 : ℚ) / (n.choose 2 : ℚ) = 3/5)
  (h2 : 6 * (m : ℚ) / (n : ℚ) = 4) :
  (n - m : ℚ) / ((n - 1) : ℚ) = 1/5 := by
  sorry

end red_ball_probability_l2091_209150


namespace game_probability_l2091_209100

/-- Represents the probability of winning for each player -/
structure PlayerProbabilities where
  alex : ℝ
  mel : ℝ
  chelsea : ℝ
  sam : ℝ

/-- Calculates the probability of a specific outcome in the game -/
def probability_of_outcome (probs : PlayerProbabilities) : ℝ :=
  probs.alex^3 * probs.mel^2 * probs.chelsea^2 * probs.sam

/-- The number of ways to arrange the wins -/
def number_of_arrangements : ℕ := 420

theorem game_probability (probs : PlayerProbabilities) 
  (h1 : probs.alex = 1/3)
  (h2 : probs.mel = 3 * probs.sam)
  (h3 : probs.chelsea = probs.sam)
  (h4 : probs.alex + probs.mel + probs.chelsea + probs.sam = 1) :
  (probability_of_outcome probs) * (number_of_arrangements : ℝ) = 13440/455625 := by
  sorry


end game_probability_l2091_209100


namespace translation_preserves_segment_find_translated_point_l2091_209194

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A translation in 2D space -/
structure Translation where
  dx : ℝ
  dy : ℝ

/-- Apply a translation to a point -/
def apply_translation (t : Translation) (p : Point) : Point :=
  { x := p.x + t.dx, y := p.y + t.dy }

theorem translation_preserves_segment (A B A' : Point) (t : Translation) :
  apply_translation t A = A' →
  apply_translation t B = 
    { x := B.x + (A'.x - A.x), 
      y := B.y + (A'.y - A.y) } := by sorry

/-- The main theorem -/
theorem find_translated_point :
  let A : Point := { x := -1, y := 2 }
  let A' : Point := { x := 3, y := -4 }
  let B : Point := { x := 2, y := 4 }
  let t : Translation := { dx := A'.x - A.x, dy := A'.y - A.y }
  apply_translation t B = { x := 6, y := -2 } := by sorry

end translation_preserves_segment_find_translated_point_l2091_209194


namespace handshakes_at_event_l2091_209120

/-- Represents the number of married couples at the event -/
def num_couples : ℕ := 15

/-- Calculates the total number of handshakes at the event -/
def total_handshakes (n : ℕ) : ℕ :=
  let num_men := n
  let num_women := n
  let handshakes_among_men := n * (n - 1) / 2
  let handshakes_men_women := n * (n - 1)
  handshakes_among_men + handshakes_men_women

/-- Theorem stating that the total number of handshakes is 315 -/
theorem handshakes_at_event : 
  total_handshakes num_couples = 315 := by
  sorry

#eval total_handshakes num_couples

end handshakes_at_event_l2091_209120


namespace dart_second_session_score_l2091_209155

/-- Represents the points scored in each dart-throwing session -/
structure DartScores where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Checks if the given DartScores satisfy the problem conditions -/
def validScores (scores : DartScores) : Prop :=
  scores.second = 2 * scores.first ∧
  scores.third = 3 * scores.first ∧
  scores.first ≥ 8

theorem dart_second_session_score (scores : DartScores) :
  validScores scores → scores.second = 48 := by
  sorry

#check dart_second_session_score

end dart_second_session_score_l2091_209155


namespace fraction_simplification_l2091_209161

theorem fraction_simplification (a b : ℚ) (ha : a = 5) (hb : b = 4) :
  (1 / b) / (1 / a) = 5 / 4 := by
  sorry

end fraction_simplification_l2091_209161


namespace min_m_for_24m_equals_n4_l2091_209176

theorem min_m_for_24m_equals_n4 (m n : ℕ+) (h : 24 * m = n^4) :
  ∀ k : ℕ+, 24 * k = (some_nat : ℕ+)^4 → m ≤ k :=
by sorry

end min_m_for_24m_equals_n4_l2091_209176

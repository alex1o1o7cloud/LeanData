import Mathlib

namespace NUMINAMATH_CALUDE_abs_and_opposite_l2031_203104

theorem abs_and_opposite :
  (abs (-2) = 2) ∧ (-(1/2) = -1/2) := by sorry

end NUMINAMATH_CALUDE_abs_and_opposite_l2031_203104


namespace NUMINAMATH_CALUDE_division_problem_l2031_203119

theorem division_problem : (102 / 6) / 3 = 5 + 2/3 := by sorry

end NUMINAMATH_CALUDE_division_problem_l2031_203119


namespace NUMINAMATH_CALUDE_books_borrowed_by_lunch_correct_l2031_203184

/-- Represents the number of books borrowed by lunchtime -/
def books_borrowed_by_lunch : ℕ := 50

/-- Represents the initial number of books on the shelf -/
def initial_books : ℕ := 100

/-- Represents the number of books added after lunch -/
def books_added : ℕ := 40

/-- Represents the number of books borrowed by evening -/
def books_borrowed_by_evening : ℕ := 30

/-- Represents the number of books remaining by evening -/
def books_remaining : ℕ := 60

/-- Proves that the number of books borrowed by lunchtime is correct -/
theorem books_borrowed_by_lunch_correct :
  initial_books - books_borrowed_by_lunch + books_added - books_borrowed_by_evening = books_remaining :=
by sorry


end NUMINAMATH_CALUDE_books_borrowed_by_lunch_correct_l2031_203184


namespace NUMINAMATH_CALUDE_smallest_sum_with_same_probability_l2031_203157

/-- Represents a symmetrical die with 6 faces --/
structure SymmetricalDie :=
  (faces : Fin 6)

/-- Represents a set of symmetrical dice --/
def DiceSet := List SymmetricalDie

/-- The probability of getting a specific sum --/
def probability (dice : DiceSet) (sum : Nat) : ℝ := sorry

theorem smallest_sum_with_same_probability 
  (dice : DiceSet) 
  (p : ℝ) 
  (h1 : p > 0) 
  (h2 : probability dice 2022 = p) : 
  ∃ (smallest_sum : Nat), 
    smallest_sum = 337 ∧ 
    probability dice smallest_sum = p ∧ 
    ∀ (other_sum : Nat), 
      other_sum < smallest_sum → probability dice other_sum ≠ p :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_with_same_probability_l2031_203157


namespace NUMINAMATH_CALUDE_quadratic_polynomial_from_sum_and_product_l2031_203140

theorem quadratic_polynomial_from_sum_and_product (x y : ℝ) 
  (h_sum : x + y = 15) (h_product : x * y = 36) :
  (fun z : ℝ => z^2 - 15*z + 36) = (fun z : ℝ => (z - x) * (z - y)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_from_sum_and_product_l2031_203140


namespace NUMINAMATH_CALUDE_greatest_b_value_l2031_203120

theorem greatest_b_value (b : ℝ) : 
  (∀ x : ℝ, x^2 - 12*x + 35 ≤ 0 → x ≤ 7) ∧ 
  (7^2 - 12*7 + 35 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_greatest_b_value_l2031_203120


namespace NUMINAMATH_CALUDE_line_not_in_third_quadrant_l2031_203188

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in the 2D plane using the general form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point is in the first quadrant -/
def isInFirstQuadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y > 0

/-- Checks if a line passes through the third quadrant -/
def passesThroughThirdQuadrant (l : Line) : Prop :=
  ∃ (x y : ℝ), x < 0 ∧ y < 0 ∧ l.a * x + l.b * y + l.c = 0

/-- The main theorem -/
theorem line_not_in_third_quadrant 
  (a b : ℝ) 
  (h_first_quadrant : isInFirstQuadrant ⟨a*b, a+b⟩) :
  ¬passesThroughThirdQuadrant ⟨b, a, -a*b⟩ :=
by sorry

end NUMINAMATH_CALUDE_line_not_in_third_quadrant_l2031_203188


namespace NUMINAMATH_CALUDE_statement_C_is_incorrect_l2031_203142

theorem statement_C_is_incorrect : ∃ (p q : Prop), ¬(p ∧ q) ∧ (p ∨ q) := by
  sorry

end NUMINAMATH_CALUDE_statement_C_is_incorrect_l2031_203142


namespace NUMINAMATH_CALUDE_yulia_lemonade_expenses_l2031_203128

/-- Represents the financial data for Yulia's earnings --/
structure YuliaFinances where
  net_profit : ℝ
  lemonade_revenue : ℝ
  babysitting_earnings : ℝ

/-- Calculates the expenses for operating the lemonade stand --/
def lemonade_expenses (finances : YuliaFinances) : ℝ :=
  finances.lemonade_revenue + finances.babysitting_earnings - finances.net_profit

/-- Theorem stating that Yulia's lemonade stand expenses are $34 --/
theorem yulia_lemonade_expenses :
  let finances : YuliaFinances := {
    net_profit := 44,
    lemonade_revenue := 47,
    babysitting_earnings := 31
  }
  lemonade_expenses finances = 34 := by
  sorry

end NUMINAMATH_CALUDE_yulia_lemonade_expenses_l2031_203128


namespace NUMINAMATH_CALUDE_fraction_evaluation_l2031_203174

theorem fraction_evaluation (a b : ℝ) (h : a^2 + b^2 ≠ 0) :
  (a^4 + b^4) / (a^2 + b^2) = a^2 + b^2 - (2 * a^2 * b^2) / (a^2 + b^2) := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l2031_203174


namespace NUMINAMATH_CALUDE_gilda_marbles_l2031_203118

theorem gilda_marbles (M : ℝ) (h : M > 0) : 
  let remaining_after_pedro : ℝ := 0.70 * M
  let remaining_after_ebony : ℝ := 0.85 * remaining_after_pedro
  let remaining_after_jimmy : ℝ := 0.80 * remaining_after_ebony
  let final_remaining : ℝ := 0.90 * remaining_after_jimmy
  final_remaining / M = 0.4284 := by
sorry

end NUMINAMATH_CALUDE_gilda_marbles_l2031_203118


namespace NUMINAMATH_CALUDE_distinct_collections_biology_l2031_203100

def biology : Finset Char := {'B', 'I', 'O', 'L', 'O', 'G', 'Y'}

def vowels : Finset Char := {'I', 'O'}
def consonants : Finset Char := {'B', 'L', 'G', 'Y'}

def num_o : ℕ := (biology.filter (· = 'O')).card

theorem distinct_collections_biology :
  let total_selections := (Finset.powerset biology).filter (λ s => 
    (s.filter (λ c => c ∈ vowels)).card = 3 ∧ 
    (s.filter (λ c => c ∈ consonants)).card = 2)
  (Finset.powerset total_selections).card = 18 :=
sorry

end NUMINAMATH_CALUDE_distinct_collections_biology_l2031_203100


namespace NUMINAMATH_CALUDE_maximum_marks_calculation_l2031_203137

theorem maximum_marks_calculation (victor_percentage : ℝ) (victor_marks : ℝ) : 
  victor_percentage = 92 → 
  victor_marks = 460 → 
  (victor_marks / (victor_percentage / 100)) = 500 := by
sorry

end NUMINAMATH_CALUDE_maximum_marks_calculation_l2031_203137


namespace NUMINAMATH_CALUDE_geometric_sum_specific_l2031_203107

/-- Sum of a finite geometric series -/
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- Theorem: The sum of the first 6 terms of the geometric sequence with
    first term 1/5 and common ratio 1/5 is equal to 1953/7812 -/
theorem geometric_sum_specific : geometric_sum (1/5) (1/5) 6 = 1953/7812 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sum_specific_l2031_203107


namespace NUMINAMATH_CALUDE_intersection_nonempty_implies_m_value_l2031_203177

theorem intersection_nonempty_implies_m_value (m : ℤ) : 
  let P : Set ℤ := {0, m}
  let Q : Set ℤ := {x | 2 * x^2 - 5 * x < 0}
  (P ∩ Q).Nonempty → m = 1 ∨ m = 2 := by
sorry

end NUMINAMATH_CALUDE_intersection_nonempty_implies_m_value_l2031_203177


namespace NUMINAMATH_CALUDE_point_movement_l2031_203143

/-- Represents a point on a number line -/
structure Point where
  value : ℤ

/-- Moves a point on the number line -/
def move (p : Point) (units : ℤ) : Point :=
  ⟨p.value + units⟩

theorem point_movement (A B : Point) :
  A.value = -3 →
  B = move A 7 →
  B.value = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_point_movement_l2031_203143


namespace NUMINAMATH_CALUDE_minutes_in_year_scientific_notation_l2031_203151

/-- The number of days in a year -/
def days_in_year : ℕ := 360

/-- The number of hours in a day -/
def hours_in_day : ℕ := 24

/-- The number of minutes in an hour -/
def minutes_in_hour : ℕ := 60

/-- Converts a natural number to a real number -/
def to_real (n : ℕ) : ℝ := n

/-- Rounds a real number to three significant figures -/
noncomputable def round_to_three_sig_figs (x : ℝ) : ℝ := 
  sorry

/-- The main theorem stating that the number of minutes in a year,
    when expressed in scientific notation with three significant figures,
    is equal to 5.18 × 10^5 -/
theorem minutes_in_year_scientific_notation :
  round_to_three_sig_figs (to_real (days_in_year * hours_in_day * minutes_in_hour)) = 5.18 * 10^5 := by
  sorry

end NUMINAMATH_CALUDE_minutes_in_year_scientific_notation_l2031_203151


namespace NUMINAMATH_CALUDE_parallelepiped_to_cube_l2031_203148

/-- Represents a rectangular parallelepiped with side lengths (a, b, c) -/
structure Parallelepiped where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a cube with side length s -/
structure Cube where
  s : ℝ

/-- Predicate to check if a parallelepiped can be divided into four parts
    that can be reassembled to form a cube -/
def can_form_cube (p : Parallelepiped) : Prop :=
  ∃ (cube : Cube), 
    cube.s ^ 3 = p.a * p.b * p.c ∧ 
    (∃ (x : ℝ), p.a = 8*x ∧ p.b = 8*x ∧ p.c = 27*x ∧ cube.s = 12*x)

/-- Theorem stating that a rectangular parallelepiped with side ratio 8:8:27
    can be divided into four parts that can be reassembled to form a cube -/
theorem parallelepiped_to_cube : 
  ∀ (p : Parallelepiped), p.a / p.b = 1 ∧ p.b / p.c = 8 / 27 → can_form_cube p :=
by sorry

end NUMINAMATH_CALUDE_parallelepiped_to_cube_l2031_203148


namespace NUMINAMATH_CALUDE_simplify_expression_l2031_203166

theorem simplify_expression (w : ℝ) : 3*w + 6*w + 9*w + 12*w + 15*w + 18 = 45*w + 18 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2031_203166


namespace NUMINAMATH_CALUDE_unit_vector_parallel_to_a_l2031_203122

def vector_a : ℝ × ℝ := (12, 5)

theorem unit_vector_parallel_to_a :
  ∃ (u : ℝ × ℝ), (u.1 * u.1 + u.2 * u.2 = 1) ∧
  (∃ (k : ℝ), vector_a = (k * u.1, k * u.2)) ∧
  (u = (12/13, 5/13) ∨ u = (-12/13, -5/13)) :=
sorry

end NUMINAMATH_CALUDE_unit_vector_parallel_to_a_l2031_203122


namespace NUMINAMATH_CALUDE_monthly_salary_calculation_l2031_203158

/-- Represents the monthly salary in Rupees -/
def monthly_salary : ℝ := sorry

/-- Represents the savings rate as a decimal -/
def savings_rate : ℝ := 0.2

/-- Represents the expense increase rate as a decimal -/
def expense_increase_rate : ℝ := 0.2

/-- Represents the new monthly savings amount in Rupees -/
def new_savings : ℝ := 240

theorem monthly_salary_calculation :
  monthly_salary * (1 - (1 + expense_increase_rate) * (1 - savings_rate)) = new_savings ∧
  monthly_salary = 6000 := by sorry

end NUMINAMATH_CALUDE_monthly_salary_calculation_l2031_203158


namespace NUMINAMATH_CALUDE_xyz_value_l2031_203170

theorem xyz_value (x y z : ℝ) 
  (h1 : x / y = 1 / 2 ∧ y / z = 2 / 7)  -- Represents x : y : z = 1 : 2 : 7
  (h2 : 2 * x - y + 3 * z = 105) :  -- Represents 2x - y + 3z = 105
  x * y * z = 1750 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l2031_203170


namespace NUMINAMATH_CALUDE_cyclic_fraction_sum_l2031_203172

theorem cyclic_fraction_sum (a b c : ℝ) 
  (h1 : a + b + c = 0) 
  (h2 : a / b + b / c + c / a = 100) : 
  b / a + c / b + a / c = -103 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_fraction_sum_l2031_203172


namespace NUMINAMATH_CALUDE_polynomial_remainder_l2031_203192

theorem polynomial_remainder (x : ℝ) : (x^11 + 2) % (x - 1) = 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l2031_203192


namespace NUMINAMATH_CALUDE_range_of_m_l2031_203159

-- Define the propositions p and q as functions of m
def p (m : ℝ) : Prop := ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ < 0 ∧ x₂ < 0 ∧ x₁^2 + m*x₁ + 1 = 0 ∧ x₂^2 + m*x₂ + 1 = 0

def q (m : ℝ) : Prop := ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 > 0

-- Define the set of m values that satisfy the conditions
def S : Set ℝ := {m | (p m ∨ q m) ∧ ¬(p m ∧ q m)}

-- State the theorem
theorem range_of_m : S = Set.Ioo 1 2 ∪ Set.Ici 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l2031_203159


namespace NUMINAMATH_CALUDE_reciprocal_opposite_equation_l2031_203155

theorem reciprocal_opposite_equation (m : ℝ) : (1 / (-0.5) = -(m + 4)) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_opposite_equation_l2031_203155


namespace NUMINAMATH_CALUDE_total_lemons_l2031_203173

/-- Given the number of lemons for each person in terms of x, prove the total number of lemons. -/
theorem total_lemons (x : ℝ) :
  let L := x
  let J := x + 6
  let A := (4/3) * (x + 6)
  let E := (2/3) * (x + 6)
  let I := 2 * (2/3) * (x + 6)
  let N := (3/4) * x
  let O := (3/5) * (4/3) * (x + 6)
  L + J + A + E + I + N + O = (413/60) * x + 30.8 := by
sorry

end NUMINAMATH_CALUDE_total_lemons_l2031_203173


namespace NUMINAMATH_CALUDE_distribute_balls_eq_partitions_six_balls_four_boxes_l2031_203127

/-- Number of ways to distribute n indistinguishable balls into k indistinguishable boxes -/
def distribute_balls (n k : ℕ) : ℕ := sorry

/-- Number of partitions of n into at most k parts -/
def partitions (n k : ℕ) : ℕ := sorry

/-- Theorem stating that distributing n indistinguishable balls into k indistinguishable boxes
    is equivalent to finding partitions of n into at most k parts -/
theorem distribute_balls_eq_partitions (n k : ℕ) :
  distribute_balls n k = partitions n k := by sorry

/-- The specific case for 6 balls and 4 boxes -/
theorem six_balls_four_boxes :
  distribute_balls 6 4 = 9 := by sorry

end NUMINAMATH_CALUDE_distribute_balls_eq_partitions_six_balls_four_boxes_l2031_203127


namespace NUMINAMATH_CALUDE_simplify_expression_l2031_203129

theorem simplify_expression : (81 / 16) ^ (3 / 4) - (-1) ^ 0 = 19 / 8 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2031_203129


namespace NUMINAMATH_CALUDE_berry_picking_difference_l2031_203133

/-- Represents the berry picking scenario -/
structure BerryPicking where
  total_berries : ℕ
  dima_basket_ratio : ℚ
  sergei_basket_ratio : ℚ
  dima_speed_multiplier : ℕ

/-- Calculates the difference in berries placed in the basket between Dima and Sergei -/
def berry_difference (scenario : BerryPicking) : ℕ :=
  sorry

/-- The main theorem stating the difference in berries placed in the basket -/
theorem berry_picking_difference (scenario : BerryPicking) 
  (h1 : scenario.total_berries = 450)
  (h2 : scenario.dima_basket_ratio = 1/2)
  (h3 : scenario.sergei_basket_ratio = 2/3)
  (h4 : scenario.dima_speed_multiplier = 2) :
  berry_difference scenario = 50 :=
sorry

end NUMINAMATH_CALUDE_berry_picking_difference_l2031_203133


namespace NUMINAMATH_CALUDE_eight_power_x_equals_one_eighth_of_two_power_thirty_l2031_203167

theorem eight_power_x_equals_one_eighth_of_two_power_thirty (x : ℝ) : 
  (1/8 : ℝ) * (2^30) = 8^x → x = 9 := by
sorry

end NUMINAMATH_CALUDE_eight_power_x_equals_one_eighth_of_two_power_thirty_l2031_203167


namespace NUMINAMATH_CALUDE_taxi_overtakes_bus_l2031_203178

theorem taxi_overtakes_bus (taxi_speed : ℝ) (bus_delay : ℝ) (speed_difference : ℝ)
  (h1 : taxi_speed = 60)
  (h2 : bus_delay = 3)
  (h3 : speed_difference = 30) :
  let bus_speed := taxi_speed - speed_difference
  let overtake_time := (bus_speed * bus_delay) / (taxi_speed - bus_speed)
  overtake_time = 3 := by
sorry

end NUMINAMATH_CALUDE_taxi_overtakes_bus_l2031_203178


namespace NUMINAMATH_CALUDE_purely_imaginary_z_l2031_203154

theorem purely_imaginary_z (z : ℂ) : 
  (∃ b : ℝ, z = Complex.I * b) → -- z is purely imaginary
  (∃ c : ℝ, (z + 1)^2 - 2*Complex.I = Complex.I * c) → -- (z+1)^2 - 2i is purely imaginary
  z = -Complex.I := by
sorry

end NUMINAMATH_CALUDE_purely_imaginary_z_l2031_203154


namespace NUMINAMATH_CALUDE_propositions_correctness_l2031_203105

-- Proposition ①
def proposition_1 : Prop :=
  (¬∃ x : ℝ, x^2 + 1 > 3*x) ↔ (∀ x : ℝ, x^2 + 1 < 3*x)

-- Proposition ②
def proposition_2 : Prop :=
  ∀ p q : Prop, (¬(p ∨ q)) → (¬p ∧ ¬q)

-- Proposition ③
def proposition_3 : Prop :=
  ∀ a : ℝ, (a > 3 → a > Real.pi) ∧ ¬(a > Real.pi → a > 3)

-- Proposition ④
def proposition_4 : Prop :=
  ∀ a : ℝ, (∀ x : ℝ, (x + 2) * (x + a) = (-x + 2) * (-x + a)) → a = -2

theorem propositions_correctness :
  ¬proposition_1 ∧ proposition_2 ∧ ¬proposition_3 ∧ proposition_4 :=
sorry

end NUMINAMATH_CALUDE_propositions_correctness_l2031_203105


namespace NUMINAMATH_CALUDE_max_stamps_per_page_l2031_203134

theorem max_stamps_per_page (album1 album2 album3 : ℕ) 
  (h1 : album1 = 945)
  (h2 : album2 = 1260)
  (h3 : album3 = 1575) :
  Nat.gcd album1 (Nat.gcd album2 album3) = 315 :=
by sorry

end NUMINAMATH_CALUDE_max_stamps_per_page_l2031_203134


namespace NUMINAMATH_CALUDE_derivative_f_at_pi_l2031_203195

noncomputable def f (x : ℝ) : ℝ := x^2 / Real.cos x

theorem derivative_f_at_pi : 
  deriv f π = -2 * π := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_pi_l2031_203195


namespace NUMINAMATH_CALUDE_money_share_difference_l2031_203108

theorem money_share_difference (total : ℝ) (moses_percent : ℝ) (rachel_percent : ℝ) 
  (h1 : total = 80)
  (h2 : moses_percent = 0.35)
  (h3 : rachel_percent = 0.20) : 
  moses_percent * total - (total - (moses_percent * total + rachel_percent * total)) / 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_money_share_difference_l2031_203108


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2031_203163

def M : Set ℝ := {x | x - 1 = 0}
def N : Set ℝ := {x | x^2 - 3*x + 2 = 0}

theorem intersection_of_M_and_N : M ∩ N = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2031_203163


namespace NUMINAMATH_CALUDE_division_in_base5_l2031_203124

/-- Converts a number from base 5 to base 10 -/
def base5ToBase10 (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- Converts a number from base 10 to base 5 -/
def base10ToBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
    aux n []

theorem division_in_base5 :
  let dividend := base5ToBase10 [2, 3, 2, 3]  -- 3232 in base 5
  let divisor := base5ToBase10 [1, 2]         -- 21 in base 5
  let quotient := base5ToBase10 [0, 3, 1]     -- 130 in base 5
  let remainder := 2
  dividend = divisor * quotient + remainder ∧
  remainder < divisor ∧
  base10ToBase5 (dividend / divisor) = [0, 3, 1] ∧
  base10ToBase5 (dividend % divisor) = [2] :=
by sorry


end NUMINAMATH_CALUDE_division_in_base5_l2031_203124


namespace NUMINAMATH_CALUDE_pyramid_volume_l2031_203111

/-- Given a triangular pyramid SABC with base ABC being an equilateral triangle
    with side length a and edge SA = b, where the lateral faces are congruent,
    this theorem proves the possible volumes of the pyramid based on the
    relationship between a and b. -/
theorem pyramid_volume (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let V1 := a^2 / 12 * Real.sqrt (3 * b^2 - a^2)
  let V2 := a^2 * Real.sqrt 3 / 12 * Real.sqrt (b^2 - a^2)
  let V3 := a^2 * Real.sqrt 3 / 12 * Real.sqrt (b^2 - 3 * a^2)
  (a / Real.sqrt 3 < b ∧ b ≤ a → volume_pyramid = V1) ∧
  (a < b ∧ b ≤ a * Real.sqrt 3 → volume_pyramid = V1 ∨ volume_pyramid = V2) ∧
  (b > a * Real.sqrt 3 → volume_pyramid = V1 ∨ volume_pyramid = V2 ∨ volume_pyramid = V3) :=
by sorry

def volume_pyramid : ℝ := sorry

end NUMINAMATH_CALUDE_pyramid_volume_l2031_203111


namespace NUMINAMATH_CALUDE_estimate_keyboard_warriors_opposition_l2031_203136

/-- Estimates the number of people with a certain characteristic in a population based on a sample. -/
def estimatePopulation (totalPopulation : ℕ) (sampleSize : ℕ) (sampleOpposed : ℕ) : ℕ :=
  (totalPopulation * sampleOpposed) / sampleSize

/-- Theorem stating that the estimated number of people opposed to "keyboard warriors" is 6912. -/
theorem estimate_keyboard_warriors_opposition :
  let totalPopulation : ℕ := 9600
  let sampleSize : ℕ := 50
  let sampleOpposed : ℕ := 36
  estimatePopulation totalPopulation sampleSize sampleOpposed = 6912 := by
  sorry

#eval estimatePopulation 9600 50 36

end NUMINAMATH_CALUDE_estimate_keyboard_warriors_opposition_l2031_203136


namespace NUMINAMATH_CALUDE_survey_optimism_l2031_203103

theorem survey_optimism (a b c : ℕ) (m n : ℤ) : 
  a + b + c = 100 →
  m = a + b / 2 →
  n = a - c →
  m = 40 →
  n = -20 :=
by sorry

end NUMINAMATH_CALUDE_survey_optimism_l2031_203103


namespace NUMINAMATH_CALUDE_angle_B_is_30_degrees_l2031_203145

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the law of sines
def lawOfSines (t : Triangle) : Prop :=
  t.a / Real.sin t.A = t.b / Real.sin t.B

-- Define the theorem
theorem angle_B_is_30_degrees (t : Triangle) 
  (h1 : t.A = 45 * π / 180)
  (h2 : t.a = 6)
  (h3 : t.b = 3 * Real.sqrt 2) :
  t.B = 30 * π / 180 :=
sorry

end NUMINAMATH_CALUDE_angle_B_is_30_degrees_l2031_203145


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l2031_203190

theorem contrapositive_equivalence :
  (∀ x : ℝ, x^2 = 1 → (x = 1 ∨ x = -1)) ↔
  (∀ x : ℝ, (x ≠ 1 ∧ x ≠ -1) → x^2 ≠ 1) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l2031_203190


namespace NUMINAMATH_CALUDE_initial_workers_count_l2031_203193

theorem initial_workers_count (W : ℕ) : 
  (2 : ℚ) / 3 * W = W - (W / 3) →  -- Initially, 2/3 of workers are men
  (W / 3 + 10 : ℚ) / (W + 10) = 2 / 5 →  -- After hiring 10 women, 40% of workforce is female
  W = 90 := by
sorry

end NUMINAMATH_CALUDE_initial_workers_count_l2031_203193


namespace NUMINAMATH_CALUDE_coefficient_sum_equality_l2031_203187

theorem coefficient_sum_equality (n : ℕ) (h : n ≥ 5) :
  (Finset.range (n - 4)).sum (λ k => Nat.choose (k + 5) 5) = Nat.choose (n + 1) 6 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_sum_equality_l2031_203187


namespace NUMINAMATH_CALUDE_intersection_range_of_b_l2031_203125

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p | p.1^2 + 2*p.2^2 = 3}
def N (m b : ℝ) : Set (ℝ × ℝ) := {p | p.2 = m*p.1 + b}

-- State the theorem
theorem intersection_range_of_b :
  (∀ m : ℝ, (M ∩ N m b).Nonempty) ↔ b ∈ Set.Icc (-Real.sqrt 6 / 2) (Real.sqrt 6 / 2) :=
sorry

end NUMINAMATH_CALUDE_intersection_range_of_b_l2031_203125


namespace NUMINAMATH_CALUDE_same_distance_different_time_l2031_203135

/-- Proves that given Joann's average speed and time, Fran needs to ride at a specific speed to cover the same distance in less time -/
theorem same_distance_different_time (joann_speed : ℝ) (joann_time : ℝ) (fran_time : ℝ) 
  (h1 : joann_speed = 14)
  (h2 : joann_time = 4)
  (h3 : fran_time = 2) :
  joann_speed * joann_time = (joann_speed * joann_time / fran_time) * fran_time :=
by sorry

end NUMINAMATH_CALUDE_same_distance_different_time_l2031_203135


namespace NUMINAMATH_CALUDE_log_inequality_l2031_203182

theorem log_inequality (x : ℝ) (hx : x > 0) : Real.log (x + 1) ≥ x - (1/2) * x^2 := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l2031_203182


namespace NUMINAMATH_CALUDE_list_size_theorem_l2031_203168

theorem list_size_theorem (L : List ℝ) (n : ℝ) : 
  L.Nodup → 
  n ∈ L → 
  n = 5 * ((L.sum - n) / (L.length - 1)) → 
  n = 0.2 * L.sum → 
  L.length = 21 :=
sorry

end NUMINAMATH_CALUDE_list_size_theorem_l2031_203168


namespace NUMINAMATH_CALUDE_midpoint_locus_l2031_203164

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 8*x = 0

/-- The point M is the midpoint of OA -/
def is_midpoint (x y : ℝ) : Prop := ∃ (ax ay : ℝ), circle_equation ax ay ∧ x = ax/2 ∧ y = ay/2

/-- The locus equation -/
def locus_equation (x y : ℝ) : Prop := x^2 + y^2 - 4*x = 0

theorem midpoint_locus : ∀ (x y : ℝ), is_midpoint x y → locus_equation x y :=
by sorry

end NUMINAMATH_CALUDE_midpoint_locus_l2031_203164


namespace NUMINAMATH_CALUDE_x_power_6_minus_6x_equals_711_l2031_203186

theorem x_power_6_minus_6x_equals_711 (x : ℝ) (h : x = 3) : x^6 - 6*x = 711 := by
  sorry

end NUMINAMATH_CALUDE_x_power_6_minus_6x_equals_711_l2031_203186


namespace NUMINAMATH_CALUDE_vanessa_video_files_l2031_203150

theorem vanessa_video_files :
  ∀ (initial_music_files initial_video_files deleted_files remaining_files : ℕ),
    initial_music_files = 16 →
    deleted_files = 30 →
    remaining_files = 34 →
    initial_music_files + initial_video_files = deleted_files + remaining_files →
    initial_video_files = 48 :=
by
  sorry

end NUMINAMATH_CALUDE_vanessa_video_files_l2031_203150


namespace NUMINAMATH_CALUDE_license_plate_count_l2031_203109

/-- The number of letters in the alphabet. -/
def alphabet_size : ℕ := 26

/-- The number of possible odd digits. -/
def odd_digits : ℕ := 5

/-- The number of possible even digits. -/
def even_digits : ℕ := 5

/-- The number of possible digits that are multiples of 3. -/
def multiples_of_three : ℕ := 4

/-- The total number of license plates with the given constraints. -/
def total_license_plates : ℕ := alphabet_size ^ 3 * odd_digits * even_digits * multiples_of_three

theorem license_plate_count :
  total_license_plates = 17576000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l2031_203109


namespace NUMINAMATH_CALUDE_appetizer_cost_is_six_l2031_203189

/-- The cost of dinner for a group, including main meals, appetizers, tip, and rush order fee. --/
def dinner_cost (main_meal_cost : ℝ) (num_people : ℕ) (num_appetizers : ℕ) (appetizer_cost : ℝ) (tip_rate : ℝ) (rush_fee : ℝ) : ℝ :=
  let subtotal := main_meal_cost * num_people + appetizer_cost * num_appetizers
  subtotal + tip_rate * subtotal + rush_fee

/-- Theorem stating that the appetizer cost is $6.00 given the specified conditions. --/
theorem appetizer_cost_is_six :
  ∃ (appetizer_cost : ℝ),
    dinner_cost 12 4 2 appetizer_cost 0.2 5 = 77 ∧
    appetizer_cost = 6 := by
  sorry

end NUMINAMATH_CALUDE_appetizer_cost_is_six_l2031_203189


namespace NUMINAMATH_CALUDE_team_improvements_minimum_days_team_a_l2031_203153

-- Define the problem parameters
def team_a_rate : ℝ := 15
def team_b_rate : ℝ := 10
def total_days : ℝ := 25
def total_length : ℝ := 300
def team_a_cost : ℝ := 0.6
def team_b_cost : ℝ := 0.8
def max_total_cost : ℝ := 18

-- Theorem for part 1
theorem team_improvements :
  ∃ (x y : ℝ),
    x + y = total_length ∧
    x / team_a_rate + y / team_b_rate = total_days ∧
    x = 150 ∧ y = 150 := by sorry

-- Theorem for part 2
theorem minimum_days_team_a :
  ∃ (m : ℝ),
    m ≥ 10 ∧
    ∀ (n : ℝ),
      n < 10 →
      team_a_cost * n + team_b_cost * ((total_length - team_a_rate * n) / team_b_rate) > max_total_cost := by sorry

end NUMINAMATH_CALUDE_team_improvements_minimum_days_team_a_l2031_203153


namespace NUMINAMATH_CALUDE_project_monthly_allocations_l2031_203115

/-- Proves that the number of monthly allocations is 12 given the project budget conditions -/
theorem project_monthly_allocations
  (total_budget : ℕ)
  (months_passed : ℕ)
  (amount_spent : ℕ)
  (over_budget : ℕ)
  (h1 : total_budget = 12600)
  (h2 : months_passed = 6)
  (h3 : amount_spent = 6580)
  (h4 : over_budget = 280)
  (h5 : ∃ (monthly_allocation : ℕ), total_budget = monthly_allocation * (total_budget / monthly_allocation)) :
  total_budget / ((amount_spent - over_budget) / months_passed) = 12 := by
  sorry

end NUMINAMATH_CALUDE_project_monthly_allocations_l2031_203115


namespace NUMINAMATH_CALUDE_blue_segments_count_l2031_203194

/-- Represents the number of rows and columns in the square array -/
def n : ℕ := 10

/-- Represents the total number of red dots -/
def total_red_dots : ℕ := 52

/-- Represents the number of red dots at corners -/
def corner_red_dots : ℕ := 2

/-- Represents the number of red dots on edges (excluding corners) -/
def edge_red_dots : ℕ := 16

/-- Represents the number of green line segments -/
def green_segments : ℕ := 98

/-- Theorem stating that the number of blue line segments is 37 -/
theorem blue_segments_count :
  let total_segments := 2 * n * (n - 1)
  let interior_red_dots := total_red_dots - corner_red_dots - edge_red_dots
  let red_connections := 2 * corner_red_dots + 3 * edge_red_dots + 4 * interior_red_dots
  let red_segments := (red_connections - green_segments) / 2
  let blue_segments := total_segments - red_segments - green_segments
  blue_segments = 37 := by sorry

end NUMINAMATH_CALUDE_blue_segments_count_l2031_203194


namespace NUMINAMATH_CALUDE_consecutive_points_length_l2031_203131

/-- Given 5 consecutive points on a straight line, prove that ae = 21 -/
theorem consecutive_points_length (a b c d e : ℝ) : 
  (c - b = 3 * (d - c)) →  -- bc = 3 cd
  (e - d = 8) →            -- de = 8
  (b - a = 5) →            -- ab = 5
  (c - a = 11) →           -- ac = 11
  (e - a = 21) :=          -- ae = 21
by
  sorry


end NUMINAMATH_CALUDE_consecutive_points_length_l2031_203131


namespace NUMINAMATH_CALUDE_exp_sum_geq_sin_cos_square_l2031_203114

theorem exp_sum_geq_sin_cos_square (x : ℝ) : Real.exp x + Real.exp (-x) ≥ (Real.sin x + Real.cos x)^2 := by
  sorry

end NUMINAMATH_CALUDE_exp_sum_geq_sin_cos_square_l2031_203114


namespace NUMINAMATH_CALUDE_cos_pi_third_minus_alpha_l2031_203162

theorem cos_pi_third_minus_alpha (α : ℝ) (h : Real.sin (π / 6 + α) = 3 / 5) :
  Real.cos (π / 3 - α) = 3 / 5 := by sorry

end NUMINAMATH_CALUDE_cos_pi_third_minus_alpha_l2031_203162


namespace NUMINAMATH_CALUDE_solution_set_equivalence_l2031_203112

theorem solution_set_equivalence (m n : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) 
  (h : ∀ x : ℝ, mx + n > 0 ↔ x > 2/5) : 
  ∀ x : ℝ, nx - m < 0 ↔ x > -5/2 := by
sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_l2031_203112


namespace NUMINAMATH_CALUDE_triangle_problem_l2031_203156

open Real

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem to be proved -/
theorem triangle_problem (t : Triangle) 
  (h1 : cos (2 * t.A) - 3 * cos (t.B + t.C) = 1)
  (h2 : 1/2 * t.b * t.c * sin t.A = 5 * sqrt 3)
  (h3 : t.b = 5) :
  t.A = π/3 ∧ t.a = sqrt 21 := by
  sorry


end NUMINAMATH_CALUDE_triangle_problem_l2031_203156


namespace NUMINAMATH_CALUDE_simplify_expression_l2031_203126

theorem simplify_expression (w : ℝ) : 
  2 * w + 3 - 4 * w - 6 + 7 * w + 9 - 8 * w - 12 + 3 * (2 * w - 1) = 3 * w - 9 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2031_203126


namespace NUMINAMATH_CALUDE_number_divisibility_problem_l2031_203113

theorem number_divisibility_problem :
  ∃ (N : ℕ), N > 0 ∧ N % 44 = 0 ∧ N % 35 = 3 ∧ N / 44 = 12 :=
by sorry

end NUMINAMATH_CALUDE_number_divisibility_problem_l2031_203113


namespace NUMINAMATH_CALUDE_sundae_cost_l2031_203101

theorem sundae_cost (cherry_jubilee : ℝ) (peanut_butter : ℝ) (royal_banana : ℝ) 
  (tip_percentage : ℝ) (final_bill : ℝ) :
  cherry_jubilee = 9 →
  peanut_butter = 7.5 →
  royal_banana = 10 →
  tip_percentage = 0.2 →
  final_bill = 42 →
  ∃ (death_by_chocolate : ℝ),
    death_by_chocolate = 8.5 ∧
    (cherry_jubilee + peanut_butter + royal_banana + death_by_chocolate) * (1 + tip_percentage) = final_bill :=
by sorry

end NUMINAMATH_CALUDE_sundae_cost_l2031_203101


namespace NUMINAMATH_CALUDE_min_value_of_sum_of_roots_l2031_203199

theorem min_value_of_sum_of_roots (x : ℝ) :
  Real.sqrt (x^2 + 4*x + 5) + Real.sqrt (x^2 - 8*x + 25) ≥ 2 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_sum_of_roots_l2031_203199


namespace NUMINAMATH_CALUDE_bond_investment_problem_l2031_203175

theorem bond_investment_problem (interest_income : ℝ) (rate1 rate2 : ℝ) (amount1 : ℝ) :
  interest_income = 1900 →
  rate1 = 0.0575 →
  rate2 = 0.0625 →
  amount1 = 20000 →
  ∃ amount2 : ℝ,
    amount1 * rate1 + amount2 * rate2 = interest_income ∧
    amount1 + amount2 = 32000 := by
  sorry

#check bond_investment_problem

end NUMINAMATH_CALUDE_bond_investment_problem_l2031_203175


namespace NUMINAMATH_CALUDE_farmers_wheat_cleaning_l2031_203144

/-- The total number of acres to be cleaned -/
def total_acres : ℕ := 480

/-- The original cleaning rate in acres per day -/
def original_rate : ℕ := 80

/-- The new cleaning rate with machinery in acres per day -/
def new_rate : ℕ := 90

/-- The number of acres cleaned on the last day -/
def last_day_acres : ℕ := 30

/-- The number of days taken to clean all acres -/
def days : ℕ := 6

theorem farmers_wheat_cleaning :
  (days - 1) * new_rate + last_day_acres = total_acres ∧
  days * original_rate = total_acres := by sorry

end NUMINAMATH_CALUDE_farmers_wheat_cleaning_l2031_203144


namespace NUMINAMATH_CALUDE_matrix_power_2023_l2031_203165

def A : Matrix (Fin 2) (Fin 2) ℕ :=
  !![1, 0;
     2, 1]

theorem matrix_power_2023 :
  A ^ 2023 = !![1,    0;
                4046, 1] := by
  sorry

end NUMINAMATH_CALUDE_matrix_power_2023_l2031_203165


namespace NUMINAMATH_CALUDE_lamp_probability_l2031_203149

/-- Represents the total number of outlets available -/
def total_outlets : Nat := 7

/-- Represents the number of plugs to be connected -/
def num_plugs : Nat := 3

/-- Represents the number of ways to plug 3 plugs into 7 outlets -/
def total_ways : Nat := total_outlets * (total_outlets - 1) * (total_outlets - 2)

/-- Represents the number of favorable outcomes where the lamp lights up -/
def favorable_outcomes : Nat := 78

/-- Theorem stating that the probability of the lamp lighting up is 13/35 -/
theorem lamp_probability : 
  (favorable_outcomes : ℚ) / total_ways = 13 / 35 := by sorry

end NUMINAMATH_CALUDE_lamp_probability_l2031_203149


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2031_203123

-- Define the hyperbola and its properties
def hyperbola_foci (F₁ F₂ : ℝ × ℝ) : Prop :=
  F₁ = (-Real.sqrt 10, 0) ∧ F₂ = (Real.sqrt 10, 0)

def point_on_hyperbola (M : ℝ × ℝ) (F₁ F₂ : ℝ × ℝ) : Prop :=
  let MF₁ := (M.1 - F₁.1, M.2 - F₁.2)
  let MF₂ := (M.2 - F₂.1, M.2 - F₂.2)
  MF₁.1 * MF₂.1 + MF₁.2 * MF₂.2 = 0 ∧
  Real.sqrt (MF₁.1^2 + MF₁.2^2) * Real.sqrt (MF₂.1^2 + MF₂.2^2) = 2

-- Theorem statement
theorem hyperbola_equation (F₁ F₂ M : ℝ × ℝ) :
  hyperbola_foci F₁ F₂ →
  point_on_hyperbola M F₁ F₂ →
  ∃ (x y : ℝ), M = (x, y) ∧ x^2 / 9 - y^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2031_203123


namespace NUMINAMATH_CALUDE_sequence_inequality_l2031_203181

theorem sequence_inequality (n : ℕ) (a : ℕ → ℝ) 
  (h0 : a 0 = 1/2)
  (h1 : ∀ k, k < n → a (k + 1) = a k + (1/n) * (a k)^2) :
  1 - 1/n < a n ∧ a n < 1 := by
  sorry

end NUMINAMATH_CALUDE_sequence_inequality_l2031_203181


namespace NUMINAMATH_CALUDE_intersection_distance_l2031_203106

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = -4*y

-- Define the line
def line (x y : ℝ) : Prop := y = x - 1

-- Define the focus of the parabola
def focus : ℝ × ℝ := (0, -1)

-- Theorem statement
theorem intersection_distance : 
  ∃ (M N : ℝ × ℝ),
    (parabola M.1 M.2) ∧ 
    (parabola N.1 N.2) ∧
    (line M.1 M.2) ∧ 
    (line N.1 N.2) ∧
    (line focus.1 focus.2) ∧
    (M ≠ N) ∧
    (Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2) = 8) :=
sorry

end NUMINAMATH_CALUDE_intersection_distance_l2031_203106


namespace NUMINAMATH_CALUDE_drug_price_reduction_equation_l2031_203116

/-- Represents the price reduction scenario for a drug -/
def PriceReductionScenario (initial_price final_price : ℝ) (x : ℝ) : Prop :=
  initial_price * (1 - x)^2 = final_price

/-- Theorem stating the equation for the given drug price reduction scenario -/
theorem drug_price_reduction_equation :
  PriceReductionScenario 140 35 x ↔ 140 * (1 - x)^2 = 35 := by
  sorry

end NUMINAMATH_CALUDE_drug_price_reduction_equation_l2031_203116


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2031_203110

noncomputable def y : ℝ := Real.sqrt ((Real.sqrt 65 / 3) + (5 / 3))

theorem sum_of_coefficients (a b c : ℕ+) : 
  y^120 = 3*y^117 + 17*y^114 + 13*y^112 - y^60 + (a:ℝ)*y^55 + (b:ℝ)*y^53 + (c:ℝ)*y^50 →
  a + b + c = 131 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2031_203110


namespace NUMINAMATH_CALUDE_equations_consistency_l2031_203185

/-- Given a system of equations, prove its consistency -/
theorem equations_consistency 
  (r₁ r₂ r₃ s a b c : ℝ) 
  (eq1 : r₁ * r₂ + r₂ * r₃ + r₃ * r₁ = s^2)
  (eq2 : (s - b) * (s - c) * r₁ + (s - c) * (s - a) * r₂ + (s - a) * (s - b) * r₃ = r₁ * r₂ * r₃) :
  ∃ (r₁' r₂' r₃' s' a' b' c' : ℝ),
    r₁' * r₂' + r₂' * r₃' + r₃' * r₁' = s'^2 ∧
    (s' - b') * (s' - c') * r₁' + (s' - c') * (s' - a') * r₂' + (s' - a') * (s' - b') * r₃' = r₁' * r₂' * r₃' :=
by
  sorry


end NUMINAMATH_CALUDE_equations_consistency_l2031_203185


namespace NUMINAMATH_CALUDE_compute_expression_l2031_203146

theorem compute_expression : 3 * 3^4 - 9^60 / 9^57 = -486 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l2031_203146


namespace NUMINAMATH_CALUDE_platonic_self_coincidences_l2031_203196

/-- Represents a regular polyhedron -/
structure RegularPolyhedron where
  n : ℕ  -- number of sides of each face
  F : ℕ  -- number of faces
  is_regular : n ≥ 3 ∧ F ≥ 4  -- conditions for regularity

/-- Calculates the number of self-coincidences for a regular polyhedron -/
def self_coincidences (p : RegularPolyhedron) : ℕ :=
  2 * p.n * p.F

/-- Theorem stating the number of self-coincidences for each Platonic solid -/
theorem platonic_self_coincidences :
  ∃ (tetrahedron cube octahedron dodecahedron icosahedron : RegularPolyhedron),
    (self_coincidences tetrahedron = 24) ∧
    (self_coincidences cube = 48) ∧
    (self_coincidences octahedron = 48) ∧
    (self_coincidences dodecahedron = 120) ∧
    (self_coincidences icosahedron = 120) :=
by sorry

end NUMINAMATH_CALUDE_platonic_self_coincidences_l2031_203196


namespace NUMINAMATH_CALUDE_divisible_by_2_3_5_7_under_300_l2031_203197

theorem divisible_by_2_3_5_7_under_300 : 
  ∃! n : ℕ, n > 0 ∧ n < 300 ∧ 2 ∣ n ∧ 3 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n :=
by sorry

end NUMINAMATH_CALUDE_divisible_by_2_3_5_7_under_300_l2031_203197


namespace NUMINAMATH_CALUDE_largest_two_twos_l2031_203180

def two_twos_operation : ℕ → Prop :=
  λ n => ∃ (op : ℕ → ℕ → ℕ), n = op 2 2 ∨ n = 22

theorem largest_two_twos :
  ∀ n : ℕ, two_twos_operation n → n ≤ 22 :=
by
  sorry

#check largest_two_twos

end NUMINAMATH_CALUDE_largest_two_twos_l2031_203180


namespace NUMINAMATH_CALUDE_problem_statement_l2031_203179

theorem problem_statement (x y : ℚ) (hx : x = -2) (hy : y = 1/2) :
  y * (x + y) + (x + y) * (x - y) - x^2 = -1 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l2031_203179


namespace NUMINAMATH_CALUDE_equation_represents_hyperbola_and_parabola_l2031_203147

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the equation y^4 - 9x^6 = 3y^2 - 1 -/
def equation (p : Point) : Prop :=
  p.y^4 - 9*p.x^6 = 3*p.y^2 - 1

/-- Represents a hyperbola -/
def is_hyperbola (S : Set Point) : Prop :=
  ∃ a b c d e f : ℝ, ∀ p ∈ S, a*p.x^2 + b*p.y^2 + c*p.x*p.y + d*p.x + e*p.y + f = 0 ∧ a*b < 0

/-- Represents a parabola -/
def is_parabola (S : Set Point) : Prop :=
  ∃ a b c d : ℝ, a ≠ 0 ∧ ∀ p ∈ S, p.y = a*p.x^2 + b*p.x + c ∨ p.x = a*p.y^2 + b*p.y + d

/-- The theorem to be proved -/
theorem equation_represents_hyperbola_and_parabola :
  ∃ S₁ S₂ : Set Point,
    (∀ p, p ∈ S₁ ∪ S₂ ↔ equation p) ∧
    is_hyperbola S₁ ∧
    is_parabola S₂ :=
sorry

end NUMINAMATH_CALUDE_equation_represents_hyperbola_and_parabola_l2031_203147


namespace NUMINAMATH_CALUDE_nails_per_plank_l2031_203176

theorem nails_per_plank (large_planks : ℕ) (additional_nails : ℕ) (total_nails : ℕ) :
  large_planks = 13 →
  additional_nails = 8 →
  total_nails = 229 →
  ∃ (nails_per_plank : ℕ), nails_per_plank * large_planks + additional_nails = total_nails ∧ nails_per_plank = 17 :=
by
  sorry

end NUMINAMATH_CALUDE_nails_per_plank_l2031_203176


namespace NUMINAMATH_CALUDE_polygon_sides_count_l2031_203121

theorem polygon_sides_count (n : ℕ) : n ≥ 3 →
  (n - 2) * 180 = 2 * 360 ↔ n = 6 := by sorry

end NUMINAMATH_CALUDE_polygon_sides_count_l2031_203121


namespace NUMINAMATH_CALUDE_alien_tree_age_l2031_203141

/-- Converts a base-8 number to base-10 --/
def base8ToBase10 (hundreds : Nat) (tens : Nat) (ones : Nat) : Nat :=
  hundreds * 8^2 + tens * 8^1 + ones * 8^0

/-- The age of the tree in the alien's base-8 system --/
def alienAge : Nat := base8ToBase10 3 6 7

theorem alien_tree_age : alienAge = 247 := by
  sorry

end NUMINAMATH_CALUDE_alien_tree_age_l2031_203141


namespace NUMINAMATH_CALUDE_min_value_inequality_l2031_203139

theorem min_value_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_sum : b + c ≥ a) :
  b / c + c / (a + b) ≥ Real.sqrt 2 - 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_inequality_l2031_203139


namespace NUMINAMATH_CALUDE_union_equality_implies_m_values_l2031_203152

-- Define the sets A and B
def A (m : ℝ) : Set ℝ := {1, 3, Real.sqrt m}
def B (m : ℝ) : Set ℝ := {1, m}

-- State the theorem
theorem union_equality_implies_m_values (m : ℝ) :
  A m ∪ B m = A m → m = 0 ∨ m = 3 :=
by sorry

end NUMINAMATH_CALUDE_union_equality_implies_m_values_l2031_203152


namespace NUMINAMATH_CALUDE_warehouse_shoes_l2031_203171

/-- The number of pairs of shoes in a warehouse -/
def total_shoes (blue green purple : ℕ) : ℕ := blue + green + purple

/-- Theorem: The total number of shoes in the warehouse is 1250 -/
theorem warehouse_shoes : ∃ (green : ℕ), 
  let blue := 540
  let purple := 355
  (green = purple) ∧ (total_shoes blue green purple = 1250) := by
  sorry

end NUMINAMATH_CALUDE_warehouse_shoes_l2031_203171


namespace NUMINAMATH_CALUDE_power_of_1_01_gt_1000_l2031_203161

theorem power_of_1_01_gt_1000 : (1.01 : ℝ) ^ 1000 > 1000 := by
  sorry

end NUMINAMATH_CALUDE_power_of_1_01_gt_1000_l2031_203161


namespace NUMINAMATH_CALUDE_max_areas_is_9n_l2031_203198

/-- Represents a circular disk divided by radii and secant lines -/
structure DividedDisk where
  n : ℕ
  radii : Fin (3 * n)
  secant_lines : Fin 2

/-- The maximum number of non-overlapping areas in a divided disk -/
def max_areas (disk : DividedDisk) : ℕ :=
  9 * disk.n

/-- Theorem stating that the maximum number of non-overlapping areas is 9n -/
theorem max_areas_is_9n (disk : DividedDisk) :
  max_areas disk = 9 * disk.n :=
by sorry

end NUMINAMATH_CALUDE_max_areas_is_9n_l2031_203198


namespace NUMINAMATH_CALUDE_mark_apple_count_l2031_203130

/-- The number of apples Mark has chosen -/
def num_apples (total fruit_count banana_count orange_count : ℕ) : ℕ :=
  total - (banana_count + orange_count)

/-- Theorem stating that Mark has chosen 3 apples -/
theorem mark_apple_count :
  num_apples 12 4 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_mark_apple_count_l2031_203130


namespace NUMINAMATH_CALUDE_factorization_equality_l2031_203191

theorem factorization_equality (a : ℝ) : (a + 1) * (a + 2) + 1/4 = (a + 3/2)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2031_203191


namespace NUMINAMATH_CALUDE_digit_swap_difference_l2031_203102

theorem digit_swap_difference (a b c : ℕ) (ha : a < 10) (hb : b < 10) (hc : c < 10) :
  ∃ k : ℤ, (100 * a + 10 * b + c) - (10 * a + 100 * b + c) = 90 * k :=
sorry

end NUMINAMATH_CALUDE_digit_swap_difference_l2031_203102


namespace NUMINAMATH_CALUDE_circle_radius_tangent_to_lines_l2031_203160

/-- Theorem: For a circle with center (0, k) where k > 8, if the circle is tangent to the lines y = x, y = -x, and y = 8, then its radius is 8√2 + 8. -/
theorem circle_radius_tangent_to_lines (k : ℝ) (h : k > 8) : 
  let circle := {(x, y) : ℝ × ℝ | x^2 + (y - k)^2 = (k - 8)^2}
  (∀ (x y : ℝ), (x = y ∧ (x, y) ∈ circle) → (x, y) ∈ {(x, y) | x = y}) →
  (∀ (x y : ℝ), (x = -y ∧ (x, y) ∈ circle) → (x, y) ∈ {(x, y) | x = -y}) →
  (∀ (x : ℝ), (x, 8) ∈ circle → x = 0) →
  k - 8 = 8 * (Real.sqrt 2 + 1) := by
sorry

end NUMINAMATH_CALUDE_circle_radius_tangent_to_lines_l2031_203160


namespace NUMINAMATH_CALUDE_fraction_simplification_l2031_203138

variables {a b c x y z : ℝ}

theorem fraction_simplification :
  (cx * (a^2 * x^2 + 3 * a^2 * y^2 + c^2 * z^2) + bz * (a^2 * x^2 + 3 * c^2 * x^2 + c^2 * y^2)) / (cx + bz) =
  a^2 * x^2 + c^2 * y^2 + c^2 * z^2 :=
by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2031_203138


namespace NUMINAMATH_CALUDE_distinct_triangles_in_grid_l2031_203117

/-- The number of points in each row or column of the grid -/
def grid_size : ℕ := 3

/-- The total number of points in the grid -/
def total_points : ℕ := grid_size * grid_size

/-- The number of collinear cases (rows + columns + diagonals) -/
def collinear_cases : ℕ := 2 * grid_size + 2

/-- Calculates the number of combinations of k items from n items -/
def combinations (n k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/-- The number of distinct triangles in a 3x3 grid -/
theorem distinct_triangles_in_grid :
  combinations total_points 3 - collinear_cases = 76 := by
  sorry

end NUMINAMATH_CALUDE_distinct_triangles_in_grid_l2031_203117


namespace NUMINAMATH_CALUDE_rhombus_prism_lateral_area_l2031_203183

/-- Given a rectangular quadrilateral prism with a rhombus base, this theorem calculates its lateral surface area. -/
theorem rhombus_prism_lateral_area (side_length : ℝ) (diagonal_length : ℝ) (h1 : side_length = 2) (h2 : diagonal_length = 2 * Real.sqrt 3) :
  let lateral_edge := Real.sqrt (diagonal_length^2 - side_length^2)
  let perimeter := 4 * side_length
  let lateral_area := perimeter * lateral_edge
  lateral_area = 16 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_rhombus_prism_lateral_area_l2031_203183


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l2031_203169

/-- Given a hyperbola and an intersecting line, prove the eccentricity range -/
theorem hyperbola_eccentricity_range 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (m : ℝ) 
  (h_intersect : ∃ x y : ℝ, y = 2*x + m ∧ x^2/a^2 - y^2/b^2 = 1) :
  ∃ e : ℝ, e^2 = (a^2 + b^2) / a^2 ∧ e > Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l2031_203169


namespace NUMINAMATH_CALUDE_vector_MN_value_l2031_203132

def M : ℝ × ℝ := (-3, 3)
def N : ℝ × ℝ := (-5, -1)

def vector_MN : ℝ × ℝ := (N.1 - M.1, N.2 - M.2)

theorem vector_MN_value : vector_MN = (-2, -4) := by sorry

end NUMINAMATH_CALUDE_vector_MN_value_l2031_203132

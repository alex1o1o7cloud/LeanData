import Mathlib

namespace part1_part2_l4112_411292

-- Define the function f
def f (a b x : ℝ) : ℝ := x^2 - a*x + b

-- Define the solution set of f(x) < 0
def solution_set (a b : ℝ) : Set ℝ := {x | 2 < x ∧ x < 3}

-- Define the condition for part 2
def condition_part2 (a : ℝ) : Prop := ∀ x ∈ Set.Icc (-1) 0, f a (3 - a) x ≥ 0

-- Statement for part 1
theorem part1 (a b : ℝ) : 
  (∀ x, f a b x < 0 ↔ x ∈ solution_set a b) → 
  (∀ x, b*x^2 - a*x + 1 > 0 ↔ x < 1/3 ∨ x > 1/2) :=
sorry

-- Statement for part 2
theorem part2 (a : ℝ) :
  condition_part2 a → a ≤ 3 :=
sorry

end part1_part2_l4112_411292


namespace two_digit_number_proof_l4112_411262

theorem two_digit_number_proof : 
  ∀ (n : ℕ), 
  (n ≥ 10 ∧ n < 100) →  -- n is a two-digit number
  (n % 10 + n / 10 = 9) →  -- sum of digits is 9
  (10 * (n % 10) + n / 10 = n - 9) →  -- swapping digits results in n - 9
  n = 54 := by
sorry

end two_digit_number_proof_l4112_411262


namespace functional_equation_solution_l4112_411291

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x * y + 1) = x * f y + 2) →
  (∀ x : ℝ, f x = 2 * x) :=
by sorry

end functional_equation_solution_l4112_411291


namespace min_marking_for_range_l4112_411273

def covers (marked : Finset ℕ) (n : ℕ) : Prop :=
  n ∈ marked ∨ (∃ m ∈ marked, n ∣ m ∨ m ∣ n)

def covers_range (marked : Finset ℕ) (start finish : ℕ) : Prop :=
  ∀ n, start ≤ n → n ≤ finish → covers marked n

theorem min_marking_for_range :
  ∃ (marked : Finset ℕ), covers_range marked 2 30 ∧ marked.card = 5 ∧
    ∀ (other : Finset ℕ), covers_range other 2 30 → other.card ≥ 5 :=
by sorry

end min_marking_for_range_l4112_411273


namespace min_value_of_sum_l4112_411295

theorem min_value_of_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 1 / (x + 3) + 1 / (y + 3) = 1 / 4) : 
  x + 3 * y ≥ 4 + 8 * Real.sqrt 3 ∧ 
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 
    1 / (x₀ + 3) + 1 / (y₀ + 3) = 1 / 4 ∧ 
    x₀ + 3 * y₀ = 4 + 8 * Real.sqrt 3 :=
sorry

end min_value_of_sum_l4112_411295


namespace simplify_expression_l4112_411296

theorem simplify_expression (x : ℝ) : 2*x - 3*(2+x) + 4*(2-x) - 5*(2+3*x) = -20*x - 8 := by
  sorry

end simplify_expression_l4112_411296


namespace product_of_three_numbers_l4112_411227

theorem product_of_three_numbers (a b c : ℚ) : 
  a + b + c = 30 →
  a = 3 * (b + c) →
  b = 6 * c →
  a * b * c = 675 / 28 := by
sorry

end product_of_three_numbers_l4112_411227


namespace fraction_sum_difference_l4112_411212

theorem fraction_sum_difference : 1/2 + 3/4 - 5/8 = 5/8 := by
  sorry

end fraction_sum_difference_l4112_411212


namespace triangle_constructible_l4112_411255

/-- Given a side length, angle bisector length, and altitude length of a triangle,
    prove that the triangle can be constructed uniquely if and only if
    the angle bisector length is greater than the altitude length. -/
theorem triangle_constructible (a f_a m_a : ℝ) (h_pos : a > 0 ∧ f_a > 0 ∧ m_a > 0) :
  ∃! (b c : ℝ), (b > 0 ∧ c > 0) ∧
    (∃ (α β γ : ℝ), 
      α > 0 ∧ β > 0 ∧ γ > 0 ∧
      α + β + γ = π ∧
      a^2 = b^2 + c^2 - 2*b*c*Real.cos α ∧
      f_a^2 = (b*c / (b + c))^2 + (a/2)^2 ∧
      m_a = a * Real.sin β / 2) ↔
  f_a > m_a :=
sorry

end triangle_constructible_l4112_411255


namespace max_value_of_f_l4112_411222

-- Define the function
def f (x : ℝ) : ℝ := x * (3 - 2 * x)

-- Define the domain
def domain (x : ℝ) : Prop := 0 < x ∧ x ≤ 1

-- State the theorem
theorem max_value_of_f :
  ∃ (max : ℝ), max = 9/8 ∧ ∀ (x : ℝ), domain x → f x ≤ max :=
sorry

end max_value_of_f_l4112_411222


namespace age_equation_solution_l4112_411225

theorem age_equation_solution (A : ℝ) (N : ℝ) (h1 : A = 64) :
  (1 / 2) * ((A + 8) * N - N * (A - 8)) = A ↔ N = 8 := by
  sorry

end age_equation_solution_l4112_411225


namespace factory_problem_l4112_411275

/-- Represents the production rates and working days of two factories -/
structure FactoryProduction where
  initial_rate_B : ℝ
  initial_rate_A : ℝ
  total_days : ℕ
  adjustment_days : ℕ

/-- The solution to the factory production problem -/
def factory_problem_solution (fp : FactoryProduction) : ℝ :=
  3

/-- Theorem stating the solution to the factory production problem -/
theorem factory_problem (fp : FactoryProduction) :
  fp.initial_rate_A = (4/3) * fp.initial_rate_B →
  fp.total_days = 6 →
  fp.adjustment_days = 1 →
  let days_before := fp.total_days - fp.adjustment_days - (factory_problem_solution fp)
  let production_A := fp.initial_rate_A * fp.total_days
  let production_B := fp.initial_rate_B * days_before + 2 * fp.initial_rate_B * (factory_problem_solution fp)
  production_A = production_B :=
by sorry

end factory_problem_l4112_411275


namespace toothpick_grid_problem_l4112_411206

/-- Calculates the total number of toothpicks in a grid -/
def total_toothpicks (length width : ℕ) (has_divider : Bool) : ℕ :=
  let vertical_lines := length + 1 + (if has_divider then 1 else 0)
  let vertical_toothpicks := vertical_lines * width
  let horizontal_lines := width + 1
  let horizontal_toothpicks := horizontal_lines * length
  vertical_toothpicks + horizontal_toothpicks

/-- The problem statement -/
theorem toothpick_grid_problem :
  total_toothpicks 40 25 true = 2090 := by
  sorry


end toothpick_grid_problem_l4112_411206


namespace average_hours_worked_l4112_411281

/-- Represents the number of hours worked on a given day type in a month -/
structure MonthlyHours where
  weekday : ℕ
  saturday : ℕ
  sunday : ℕ

/-- Represents the work schedule for a month -/
structure MonthSchedule where
  days : ℕ
  weekdays : ℕ
  saturdays : ℕ
  sundays : ℕ
  hours : MonthlyHours
  vacation_days : ℕ

def april : MonthSchedule :=
  { days := 30
    weekdays := 22
    saturdays := 4
    sundays := 4
    hours := { weekday := 6, saturday := 4, sunday := 0 }
    vacation_days := 5 }

def june : MonthSchedule :=
  { days := 30
    weekdays := 30
    saturdays := 0
    sundays := 0
    hours := { weekday := 5, saturday := 5, sunday := 5 }
    vacation_days := 4 }

def september : MonthSchedule :=
  { days := 30
    weekdays := 22
    saturdays := 4
    sundays := 4
    hours := { weekday := 8, saturday := 0, sunday := 0 }
    vacation_days := 0 }

def calculate_hours (m : MonthSchedule) : ℕ :=
  (m.weekdays - m.vacation_days) * m.hours.weekday +
  m.saturdays * m.hours.saturday +
  m.sundays * m.hours.sunday

theorem average_hours_worked :
  (calculate_hours april + calculate_hours june + calculate_hours september) / 3 = 141 :=
sorry

end average_hours_worked_l4112_411281


namespace alcohol_dilution_l4112_411238

/-- Proves that adding 3 litres of water to 18 litres of a 20% alcohol mixture 
    results in a new mixture with 17.14285714285715% alcohol. -/
theorem alcohol_dilution (initial_volume : ℝ) (initial_percentage : ℝ) 
    (water_added : ℝ) (final_percentage : ℝ) : 
    initial_volume = 18 →
    initial_percentage = 0.20 →
    water_added = 3 →
    final_percentage = 0.1714285714285715 →
    (initial_volume * initial_percentage) / (initial_volume + water_added) = final_percentage := by
  sorry


end alcohol_dilution_l4112_411238


namespace tangent_slope_angle_l4112_411266

noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - x^2 + 5

theorem tangent_slope_angle (x : ℝ) : 
  x = 1 → 
  ∃ θ : ℝ, θ = 3 * Real.pi / 4 ∧ 
    θ = Real.pi + Real.arctan ((deriv f) x) :=
by sorry

end tangent_slope_angle_l4112_411266


namespace system_one_solutions_system_two_solutions_l4112_411293

-- System 1
theorem system_one_solutions (x y : ℝ) :
  (x^2 - 2*x = 0 ∧ x^3 + y = 6) ↔ ((x = 0 ∧ y = 6) ∨ (x = 2 ∧ y = -2)) :=
sorry

-- System 2
theorem system_two_solutions (x y : ℝ) :
  (y^2 - 4*y + 3 = 0 ∧ 2*x + y = 9) ↔ ((x = 4 ∧ y = 1) ∨ (x = 3 ∧ y = 3)) :=
sorry

end system_one_solutions_system_two_solutions_l4112_411293


namespace division_remainder_l4112_411240

theorem division_remainder (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) :
  dividend = 686 →
  divisor = 36 →
  quotient = 19 →
  dividend = divisor * quotient + remainder →
  remainder = 2 := by
sorry

end division_remainder_l4112_411240


namespace smallest_solution_of_quadratic_l4112_411279

theorem smallest_solution_of_quadratic (y : ℝ) : 
  (3 * y^2 + 15 * y - 90 = y * (y + 20)) → y ≥ -6 :=
by
  sorry

end smallest_solution_of_quadratic_l4112_411279


namespace percentage_relationship_l4112_411224

theorem percentage_relationship (A B n c : ℝ) : 
  A > 0 → B > 0 → B > A → 
  A * (1 + n / 100) = B → B * (1 - c / 100) = A →
  A * Real.sqrt (100 + n) = B * Real.sqrt (100 - c) := by
sorry

end percentage_relationship_l4112_411224


namespace smallest_area_three_interior_points_l4112_411247

/-- A square with diagonals aligned with coordinate axes -/
structure AlignedSquare where
  side : ℝ
  center : ℝ × ℝ

/-- Count of interior lattice points in a square -/
def interiorLatticePoints (s : AlignedSquare) : ℕ := sorry

/-- The area of an AlignedSquare -/
def area (s : AlignedSquare) : ℝ := s.side * s.side

/-- Theorem: Smallest area of an AlignedSquare with exactly three interior lattice points is 8 -/
theorem smallest_area_three_interior_points :
  ∃ (s : AlignedSquare), 
    interiorLatticePoints s = 3 ∧ 
    area s = 8 ∧
    ∀ (t : AlignedSquare), interiorLatticePoints t = 3 → area t ≥ 8 := by sorry

end smallest_area_three_interior_points_l4112_411247


namespace difference_of_squares_example_l4112_411282

theorem difference_of_squares_example : 81^2 - 49^2 = 4160 := by
  sorry

end difference_of_squares_example_l4112_411282


namespace xyz_inequality_l4112_411258

theorem xyz_inequality (x y z : ℝ) 
  (h_nonneg_x : 0 ≤ x) (h_nonneg_y : 0 ≤ y) (h_nonneg_z : 0 ≤ z)
  (h_sum : x + y + z = 1) : 
  0 ≤ x*y + y*z + z*x - 2*x*y*z ∧ x*y + y*z + z*x - 2*x*y*z ≤ 7/27 := by
  sorry

end xyz_inequality_l4112_411258


namespace balance_theorem_l4112_411231

-- Define the weights of balls as real numbers
variable (R G O B : ℝ)

-- Define the balance relationships
axiom red_green : 4 * R = 8 * G
axiom orange_green : 3 * O = 6 * G
axiom green_blue : 8 * G = 6 * B

-- Theorem to prove
theorem balance_theorem : 3 * R + 2 * O + 4 * B = (46/3) * G := by
  sorry

end balance_theorem_l4112_411231


namespace smallest_dual_base_representation_l4112_411248

/-- Converts a two-digit number in base b to base 10 -/
def to_base_10 (digit : Nat) (base : Nat) : Nat :=
  base * digit + digit

/-- Checks if a digit is valid in the given base -/
def is_valid_digit (digit : Nat) (base : Nat) : Prop :=
  digit < base

theorem smallest_dual_base_representation :
  ∃ (C D : Nat),
    is_valid_digit C 6 ∧
    is_valid_digit D 8 ∧
    to_base_10 C 6 = to_base_10 D 8 ∧
    to_base_10 C 6 = 63 ∧
    (∀ (C' D' : Nat),
      is_valid_digit C' 6 →
      is_valid_digit D' 8 →
      to_base_10 C' 6 = to_base_10 D' 8 →
      to_base_10 C' 6 ≥ 63) :=
by sorry

end smallest_dual_base_representation_l4112_411248


namespace find_N_l4112_411216

theorem find_N : ∃ N : ℕ, 
  (981 + 983 + 985 + 987 + 989 + 991 + 993 = 7000 - N) ∧ (N = 91) := by
  sorry

end find_N_l4112_411216


namespace matilda_earnings_l4112_411284

/-- Calculates the total earnings for a newspaper delivery job -/
def calculate_earnings (hourly_wage : ℚ) (per_newspaper : ℚ) (newspapers_per_hour : ℕ) (shift_duration : ℕ) : ℚ :=
  let wage_earnings := hourly_wage * shift_duration
  let newspaper_earnings := per_newspaper * newspapers_per_hour * shift_duration
  wage_earnings + newspaper_earnings

/-- Proves that Matilda's earnings for a 3-hour shift equal $40.50 -/
theorem matilda_earnings : 
  calculate_earnings 6 (1/4) 30 3 = 81/2 := by
  sorry

#eval calculate_earnings 6 (1/4) 30 3

end matilda_earnings_l4112_411284


namespace museum_revenue_calculation_l4112_411235

/-- Revenue calculation for The Metropolitan Museum of Art --/
theorem museum_revenue_calculation 
  (total_visitors : ℕ) 
  (nyc_resident_ratio : ℚ)
  (college_student_ratio : ℚ)
  (college_ticket_price : ℕ) :
  total_visitors = 200 →
  nyc_resident_ratio = 1/2 →
  college_student_ratio = 3/10 →
  college_ticket_price = 4 →
  (total_visitors : ℚ) * nyc_resident_ratio * college_student_ratio * college_ticket_price = 120 := by
  sorry

#check museum_revenue_calculation

end museum_revenue_calculation_l4112_411235


namespace condition_necessary_not_sufficient_l4112_411285

theorem condition_necessary_not_sufficient :
  (∀ x : ℝ, x > 0 → x ≠ 0) ∧
  (∃ x : ℝ, x ≠ 0 ∧ ¬(x > 0)) := by
  sorry

end condition_necessary_not_sufficient_l4112_411285


namespace win_bonus_area_l4112_411204

/-- The combined area of WIN and BONUS sectors in a circular spinner -/
theorem win_bonus_area (r : ℝ) (p_win : ℝ) (p_bonus : ℝ) : 
  r = 8 → p_win = 1/4 → p_bonus = 1/8 → 
  (p_win + p_bonus) * (π * r^2) = 24 * π := by
  sorry

end win_bonus_area_l4112_411204


namespace pentagon_perimeter_l4112_411288

theorem pentagon_perimeter (A B C D E : ℝ × ℝ) : 
  let AB := 2
  let BC := Real.sqrt 5
  let CD := Real.sqrt 3
  let DE := 1
  let AC := Real.sqrt ((AB ^ 2) + (BC ^ 2))
  let AD := Real.sqrt ((AC ^ 2) + (CD ^ 2))
  let AE := Real.sqrt ((AD ^ 2) + (DE ^ 2))
  AB + BC + CD + DE + AE = 3 + Real.sqrt 5 + Real.sqrt 3 + 1 + Real.sqrt 13 :=
by sorry

end pentagon_perimeter_l4112_411288


namespace square_division_l4112_411286

/-- A square can be divided into n smaller squares for any natural number n ≥ 6 -/
theorem square_division (n : ℕ) (h : n ≥ 6) : 
  ∃ (partition : List (ℕ × ℕ)), 
    (partition.length = n) ∧ 
    (∀ (x y : ℕ × ℕ), x ∈ partition → y ∈ partition → x ≠ y → 
      (x.1 < y.1 ∨ x.2 < y.2 ∨ y.1 < x.1 ∨ y.2 < x.2)) ∧
    (∃ (side : ℕ), ∀ (square : ℕ × ℕ), square ∈ partition → 
      square.1 ≤ side ∧ square.2 ≤ side) := by
  sorry

end square_division_l4112_411286


namespace dwarf_truth_count_l4112_411202

/-- Represents the number of dwarfs who always tell the truth -/
def truthful_dwarfs : ℕ := sorry

/-- Represents the number of dwarfs who always lie -/
def lying_dwarfs : ℕ := sorry

/-- The total number of dwarfs -/
def total_dwarfs : ℕ := 10

/-- The number of times hands were raised for vanilla ice cream -/
def vanilla_hands : ℕ := 10

/-- The number of times hands were raised for chocolate ice cream -/
def chocolate_hands : ℕ := 5

/-- The number of times hands were raised for fruit ice cream -/
def fruit_hands : ℕ := 1

/-- The total number of times hands were raised -/
def total_hands_raised : ℕ := vanilla_hands + chocolate_hands + fruit_hands

theorem dwarf_truth_count :
  truthful_dwarfs + lying_dwarfs = total_dwarfs ∧
  truthful_dwarfs + 2 * lying_dwarfs = total_hands_raised ∧
  truthful_dwarfs = 4 := by sorry

end dwarf_truth_count_l4112_411202


namespace circle_area_ratio_l4112_411264

-- Define the circle structure
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the point structure
structure Point where
  x : ℝ
  y : ℝ

-- Define the midpoint property
def isMidpoint (a b m : Point) : Prop :=
  m.x = (a.x + b.x) / 2 ∧ m.y = (a.y + b.y) / 2

-- Define the theorem
theorem circle_area_ratio
  (c1 c2 : Circle)
  (o p x : Point)
  (h1 : c1.center = c2.center)
  (h2 : c1.center = (o.x, o.y))
  (h3 : c1.radius = (p.x - o.x))
  (h4 : c2.radius = (x.x - o.x))
  (h5 : isMidpoint o p x) :
  (π * c2.radius^2) / (π * c1.radius^2) = 1/4 :=
by
  sorry


end circle_area_ratio_l4112_411264


namespace sum_with_radical_conjugate_l4112_411207

theorem sum_with_radical_conjugate :
  ∃ (x : ℝ), x^2 = 2023 ∧ (15 - x) + (15 + x) = 30 := by
  sorry

end sum_with_radical_conjugate_l4112_411207


namespace total_peaches_l4112_411234

/-- The number of red peaches in each basket -/
def red_peaches_per_basket : ℕ := 19

/-- The number of green peaches in each basket -/
def green_peaches_per_basket : ℕ := 4

/-- The total number of baskets -/
def number_of_baskets : ℕ := 15

/-- Theorem: The total number of peaches in all baskets is 345 -/
theorem total_peaches :
  (red_peaches_per_basket + green_peaches_per_basket) * number_of_baskets = 345 := by
  sorry

end total_peaches_l4112_411234


namespace paint_calculation_l4112_411242

theorem paint_calculation (total_paint : ℚ) : 
  (2 / 3 : ℚ) * total_paint + (1 / 5 : ℚ) * ((1 / 3 : ℚ) * total_paint) = 264 → 
  total_paint = 360 := by
sorry

end paint_calculation_l4112_411242


namespace coin_value_difference_l4112_411220

def total_coins : ℕ := 5050

def penny_value : ℕ := 1
def dime_value : ℕ := 10

def total_value (num_pennies : ℕ) : ℕ :=
  num_pennies * penny_value + (total_coins - num_pennies) * dime_value

theorem coin_value_difference :
  ∃ (max_value min_value : ℕ),
    (∀ (num_pennies : ℕ), 1 ≤ num_pennies ∧ num_pennies ≤ total_coins - 1 →
      min_value ≤ total_value num_pennies ∧ total_value num_pennies ≤ max_value) ∧
    max_value - min_value = 45432 :=
sorry

end coin_value_difference_l4112_411220


namespace function_increment_l4112_411237

theorem function_increment (f : ℝ → ℝ) 
  (h1 : ∀ x, f (x + 19) ≤ f x + 19) 
  (h2 : ∀ x, f (x + 94) ≥ f x + 94) : 
  ∀ x, f (x + 1) = f x + 1 := by
  sorry

end function_increment_l4112_411237


namespace subtraction_absolute_value_l4112_411260

theorem subtraction_absolute_value (x y : ℝ) : 
  |8 - 3| - |x - y| = 3 → |x - y| = 2 := by
  sorry

end subtraction_absolute_value_l4112_411260


namespace height_equals_median_implies_angle_leq_60_height_equals_median_and_bisector_implies_equilateral_l4112_411208

/-- Represents a triangle with vertices A, B, and C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Checks if a triangle is acute -/
def Triangle.isAcute (t : Triangle) : Prop := sorry

/-- Returns the length of the height from vertex A to side BC -/
def Triangle.heightAH (t : Triangle) : ℝ := sorry

/-- Returns the length of the median from vertex B to side AC -/
def Triangle.medianBM (t : Triangle) : ℝ := sorry

/-- Returns the length of the angle bisector from vertex C -/
def Triangle.angleBisectorCD (t : Triangle) : ℝ := sorry

/-- Returns the measure of angle ABC in degrees -/
def Triangle.angleABC (t : Triangle) : ℝ := sorry

/-- Checks if a triangle is equilateral -/
def Triangle.isEquilateral (t : Triangle) : Prop := sorry

/-- Theorem: If the largest height AH is equal to the median BM in an acute triangle,
    then angle ABC is not greater than 60 degrees -/
theorem height_equals_median_implies_angle_leq_60 (t : Triangle) 
  (h1 : t.isAcute) 
  (h2 : t.heightAH = t.medianBM) : 
  t.angleABC ≤ 60 := by sorry

/-- Theorem: If the height AH is equal to both the median BM and the angle bisector CD 
    in an acute triangle, then the triangle is equilateral -/
theorem height_equals_median_and_bisector_implies_equilateral (t : Triangle) 
  (h1 : t.isAcute) 
  (h2 : t.heightAH = t.medianBM) 
  (h3 : t.heightAH = t.angleBisectorCD) : 
  t.isEquilateral := by sorry

end height_equals_median_implies_angle_leq_60_height_equals_median_and_bisector_implies_equilateral_l4112_411208


namespace min_value_of_expression_l4112_411298

/-- Given a line ax - 2by = 2 (where a > 0 and b > 0) passing through the center of the circle 
    x² + y² - 4x + 2y + 1 = 0, the minimum value of 4/(a+2) + 1/(b+1) is 9/4. -/
theorem min_value_of_expression (a b : ℝ) : a > 0 → b > 0 → 
  (∃ x y : ℝ, x^2 + y^2 - 4*x + 2*y + 1 = 0 ∧ a*x - 2*b*y = 2) → 
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → 
    (∃ x' y' : ℝ, x'^2 + y'^2 - 4*x' + 2*y' + 1 = 0 ∧ a'*x' - 2*b'*y' = 2) → 
    4/(a+2) + 1/(b+1) ≤ 4/(a'+2) + 1/(b'+1)) → 
  4/(a+2) + 1/(b+1) = 9/4 :=
sorry

end min_value_of_expression_l4112_411298


namespace total_veranda_area_l4112_411211

/-- Calculates the total area of verandas in a multi-story building. -/
theorem total_veranda_area (floors : ℕ) (room_length room_width veranda_width : ℝ) :
  floors = 4 →
  room_length = 21 →
  room_width = 12 →
  veranda_width = 2 →
  (floors * ((room_length + 2 * veranda_width) * (room_width + 2 * veranda_width) - room_length * room_width)) = 592 :=
by sorry

end total_veranda_area_l4112_411211


namespace star_equality_implies_x_equals_four_l4112_411244

-- Define the binary operation ★
def star (a b c d : ℤ) : ℤ × ℤ := (a - c, b - d)

-- Theorem statement
theorem star_equality_implies_x_equals_four :
  ∀ x y : ℤ, star 5 5 2 1 = star x y 1 4 → x = 4 := by
  sorry

end star_equality_implies_x_equals_four_l4112_411244


namespace exists_parallelepiped_with_square_coverage_l4112_411251

/-- A rectangular parallelepiped with integer dimensions -/
structure Parallelepiped where
  length : ℕ+
  width : ℕ+
  height : ℕ+

/-- A square with integer side length -/
structure Square where
  side : ℕ+

/-- Represents the coverage of a parallelepiped by three squares -/
structure Coverage where
  parallelepiped : Parallelepiped
  squares : Fin 3 → Square
  covers_without_gaps : Bool
  each_pair_shares_edge : Bool

/-- Theorem stating the existence of a parallelepiped covered by three squares with shared edges -/
theorem exists_parallelepiped_with_square_coverage : 
  ∃ (c : Coverage), c.covers_without_gaps ∧ c.each_pair_shares_edge := by
  sorry

end exists_parallelepiped_with_square_coverage_l4112_411251


namespace binomial_150_150_equals_1_l4112_411219

theorem binomial_150_150_equals_1 : Nat.choose 150 150 = 1 := by
  sorry

end binomial_150_150_equals_1_l4112_411219


namespace plane_representations_l4112_411287

/-- Given a plane with equation 2x - 2y + z - 20 = 0, prove its representations in intercept and normal forms -/
theorem plane_representations (x y z : ℝ) :
  (2*x - 2*y + z - 20 = 0) →
  (x/10 + y/(-10) + z/20 = 1) ∧
  (-2/3*x + 2/3*y - 1/3*z + 20/3 = 0) :=
by sorry

end plane_representations_l4112_411287


namespace problem_statement_l4112_411210

theorem problem_statement (a b c A B C : ℝ) 
  (eq1 : a + b + c = 0)
  (eq2 : A + B + C = 0)
  (eq3 : a / A + b / B + c / C = 0)
  (hA : A ≠ 0)
  (hB : B ≠ 0)
  (hC : C ≠ 0) :
  a * A^2 + b * B^2 + c * C^2 = 0 := by
sorry

end problem_statement_l4112_411210


namespace ax5_plus_by5_exists_l4112_411213

theorem ax5_plus_by5_exists (a b x y : ℝ) 
  (h1 : a*x + b*y = 4)
  (h2 : a*x^2 + b*y^2 = 10)
  (h3 : a*x^3 + b*y^3 = 28)
  (h4 : a*x^4 + b*y^4 = 82) :
  ∃ s5 : ℝ, a*x^5 + b*y^5 = s5 :=
by
  sorry

end ax5_plus_by5_exists_l4112_411213


namespace positive_root_m_value_l4112_411205

theorem positive_root_m_value (m : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ (1 - x) / (x - 2) = m / (2 - x) - 2) → m = 1 := by
  sorry

end positive_root_m_value_l4112_411205


namespace snail_climb_theorem_l4112_411249

/-- The number of days it takes for a snail to climb out of a well -/
def snail_climb_days (well_depth : ℝ) (day_climb : ℝ) (night_slide : ℝ) : ℕ :=
  sorry

/-- Theorem: A snail starting 1 meter below the top of a well, 
    climbing 30 cm during the day and sliding down 20 cm each night, 
    will take 8 days to reach the top of the well -/
theorem snail_climb_theorem : 
  snail_climb_days 1 0.3 0.2 = 8 := by sorry

end snail_climb_theorem_l4112_411249


namespace part_one_part_two_l4112_411265

-- Define the function f
def f (a x : ℝ) : ℝ := a * x^2 - (2*a + 1) * x - 1

-- Part (1)
theorem part_one (a : ℝ) :
  (∀ x : ℝ, f a x ≤ -3/4) ↔ a ∈ Set.Icc (-1) (-1/4) :=
sorry

-- Part (2)
theorem part_two (a : ℝ) :
  a ≤ 0 → ((∀ x : ℝ, x > 0 → x * f a x ≤ 1) ↔ a ∈ Set.Icc (-3) 0) :=
sorry

end part_one_part_two_l4112_411265


namespace fraction_equality_l4112_411221

theorem fraction_equality : (5 * 6 + 3) / 9 = 11 / 3 := by
  sorry

end fraction_equality_l4112_411221


namespace first_term_of_arithmetic_sequence_l4112_411217

/-- An arithmetic sequence with common difference d -/
def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1 : ℤ) * d

theorem first_term_of_arithmetic_sequence :
  ∃ a₁ : ℤ, arithmetic_sequence a₁ 2 15 = -10 ∧ a₁ = -38 := by sorry

end first_term_of_arithmetic_sequence_l4112_411217


namespace quadratic_equations_integer_roots_l4112_411233

theorem quadratic_equations_integer_roots :
  ∃ (a b c : ℕ),
    (∃ (x y : ℤ), x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0) ∧
    (∃ (x y : ℤ), x ≠ y ∧ a * x^2 + b * x - c = 0 ∧ a * y^2 + b * y - c = 0) ∧
    (∃ (x y : ℤ), x ≠ y ∧ a * x^2 - b * x + c = 0 ∧ a * y^2 - b * y + c = 0) ∧
    (∃ (x y : ℤ), x ≠ y ∧ a * x^2 - b * x - c = 0 ∧ a * y^2 - b * y - c = 0) :=
by sorry

end quadratic_equations_integer_roots_l4112_411233


namespace cubic_fraction_equals_fifteen_l4112_411297

theorem cubic_fraction_equals_fifteen :
  let a : ℤ := 8
  let b : ℤ := a - 1
  (a^3 + b^3) / (a^2 - a*b + b^2) = 15 := by
  sorry

end cubic_fraction_equals_fifteen_l4112_411297


namespace fraction_reducibility_l4112_411229

theorem fraction_reducibility (n : ℕ) :
  (∃ k : ℕ, k > 1 ∧ (n^2 + 1).gcd (n + 1) = k) ↔ n % 2 = 1 := by
  sorry

end fraction_reducibility_l4112_411229


namespace roots_of_equation_l4112_411283

def f (x : ℝ) := x^2 - |x - 1| - 1

theorem roots_of_equation :
  ∃ (x₁ x₂ : ℝ), x₁ > x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ = 1 ∧ x₂ = -2 :=
by
  sorry

end roots_of_equation_l4112_411283


namespace museum_tour_time_l4112_411239

theorem museum_tour_time (total_students : ℕ) (num_groups : ℕ) (time_per_student : ℕ) 
  (h1 : total_students = 18)
  (h2 : num_groups = 3)
  (h3 : time_per_student = 4)
  (h4 : total_students % num_groups = 0) : -- Ensuring equal groups
  (total_students / num_groups) * time_per_student = 24 := by
  sorry

end museum_tour_time_l4112_411239


namespace line_passes_through_fixed_point_l4112_411209

theorem line_passes_through_fixed_point :
  ∀ (m : ℝ), (2*m - 1) * 2 - (m + 3) * 3 - (m - 11) = 0 :=
by
  sorry

end line_passes_through_fixed_point_l4112_411209


namespace approx_root_e_2019_l4112_411271

/-- Approximation of the 2019th root of e using tangent line method -/
theorem approx_root_e_2019 (e : ℝ) (h : e = Real.exp 1) :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.00001 ∧ |e^(1/2019) - (1 + 1/2019)| < ε :=
sorry

end approx_root_e_2019_l4112_411271


namespace max_sum_l4112_411289

/-- An arithmetic sequence {an} with sum Sn -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  sum : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) = a n + d
  sum_formula : ∀ n, sum n = n * a 1 + n * (n - 1) / 2 * d

/-- The conditions of the problem -/
def problem_conditions (seq : ArithmeticSequence) : Prop :=
  seq.a 1 + seq.a 2 + seq.a 3 = 156 ∧
  seq.a 2 + seq.a 3 + seq.a 4 = 147

/-- The theorem to prove -/
theorem max_sum (seq : ArithmeticSequence) 
  (h : problem_conditions seq) : 
  ∃ (n : ℕ), n = 19 ∧ 
  ∀ (m : ℕ), m > 0 → seq.sum n ≥ seq.sum m :=
sorry

end max_sum_l4112_411289


namespace polynomial_divisibility_l4112_411223

theorem polynomial_divisibility (a b c d m : ℤ) 
  (h1 : (5 : ℤ) ∣ (a * m^3 + b * m^2 + c * m + d))
  (h2 : ¬((5 : ℤ) ∣ d)) :
  ∃ n : ℤ, (5 : ℤ) ∣ (d * n^3 + c * n^2 + b * n + a) := by
  sorry

end polynomial_divisibility_l4112_411223


namespace prime_equation_solution_l4112_411215

theorem prime_equation_solution (p : ℕ) (hp : Prime p) :
  (∃ (n : ℤ) (k m : ℕ+), (m * k^2 + 2) * p - (m^2 + 2 * k^2) = n^2 * (m * p + 2)) →
  p = 3 ∨ p % 4 = 1 := by
  sorry

end prime_equation_solution_l4112_411215


namespace megan_folders_l4112_411218

/-- The number of folders Megan ended up with -/
def num_folders (initial_files : ℕ) (deleted_files : ℕ) (files_per_folder : ℕ) : ℕ :=
  (initial_files - deleted_files) / files_per_folder

/-- Proof that Megan ended up with 9 folders -/
theorem megan_folders : num_folders 93 21 8 = 9 := by
  sorry

end megan_folders_l4112_411218


namespace proposition_false_iff_a_in_range_l4112_411245

theorem proposition_false_iff_a_in_range (a : ℝ) : 
  (¬ ∃ x : ℝ, 2 * x^2 + (a - 1) * x + (1/2 : ℝ) ≤ 0) ↔ -1 < a ∧ a < 3 := by
  sorry

end proposition_false_iff_a_in_range_l4112_411245


namespace square_difference_fifty_fortynine_l4112_411267

theorem square_difference_fifty_fortynine : (50 : ℕ)^2 - (49 : ℕ)^2 = 99 := by
  sorry

end square_difference_fifty_fortynine_l4112_411267


namespace inequality_system_solution_set_l4112_411261

theorem inequality_system_solution_set :
  {x : ℝ | x + 2 ≤ 3 ∧ 1 + x > -2} = {x : ℝ | -3 < x ∧ x ≤ 1} := by
  sorry

end inequality_system_solution_set_l4112_411261


namespace root_equation_sum_l4112_411268

theorem root_equation_sum (a : ℝ) (h : a^2 + a - 1 = 0) : 
  (1 - a) / a + a / (1 + a) = 1 := by sorry

end root_equation_sum_l4112_411268


namespace min_square_sum_on_line_l4112_411254

/-- The minimum value of x^2 + y^2 for points on the line x + y - 4 = 0 is 8 -/
theorem min_square_sum_on_line :
  ∃ (min : ℝ), min = 8 ∧
  ∀ (x y : ℝ), x + y - 4 = 0 →
  x^2 + y^2 ≥ min ∧
  ∃ (x₀ y₀ : ℝ), x₀ + y₀ - 4 = 0 ∧ x₀^2 + y₀^2 = min :=
by sorry

end min_square_sum_on_line_l4112_411254


namespace order_of_6_wrt_f_l4112_411250

def f (x : ℕ) : ℕ := x^2 % 13

def iterateF (n : ℕ) (x : ℕ) : ℕ :=
  match n with
  | 0 => x
  | n+1 => f (iterateF n x)

theorem order_of_6_wrt_f :
  ∀ k : ℕ, k > 0 → k < 36 → iterateF k 6 ≠ 6 ∧ iterateF 36 6 = 6 := by sorry

end order_of_6_wrt_f_l4112_411250


namespace max_third_side_of_triangle_l4112_411236

theorem max_third_side_of_triangle (a b : ℝ) (ha : a = 7) (hb : b = 11) :
  ∃ (c : ℕ), c = 17 ∧ 
  (∀ (x : ℕ), (a + b > x ∧ x > b - a ∧ x > a - b) → x ≤ c) :=
sorry

end max_third_side_of_triangle_l4112_411236


namespace five_fourths_of_x_over_three_l4112_411226

theorem five_fourths_of_x_over_three (x : ℝ) : (5 / 4) * (x / 3) = 5 * x / 12 := by
  sorry

end five_fourths_of_x_over_three_l4112_411226


namespace intersection_M_N_l4112_411256

-- Define the sets M and N
def M : Set ℝ := {x | x ≤ 4}
def N : Set ℝ := {x | x > 0}

-- State the theorem
theorem intersection_M_N : M ∩ N = {x : ℝ | 0 < x ∧ x ≤ 4} := by
  sorry

end intersection_M_N_l4112_411256


namespace johns_piano_expenses_l4112_411294

/-- The total cost of John's piano learning expenses --/
def total_cost (piano_cost lesson_count lesson_price discount sheet_music maintenance : ℚ) : ℚ :=
  piano_cost + 
  (lesson_count * lesson_price * (1 - discount)) + 
  sheet_music + 
  maintenance

/-- Theorem stating that John's total piano learning expenses are $1275 --/
theorem johns_piano_expenses : 
  total_cost 500 20 40 (25/100) 75 100 = 1275 := by
  sorry

end johns_piano_expenses_l4112_411294


namespace prime_pair_divisibility_l4112_411280

theorem prime_pair_divisibility (p q : ℕ) (hp : Prime p) (hq : Prime q) :
  (∃ k₁ k₂ : ℤ, ((2 * p^2 - 1)^q + 1 : ℤ) = k₁ * (p + q) ∧ 
                ((2 * q^2 - 1)^p + 1 : ℤ) = k₂ * (p + q)) ↔ 
  p = q := by sorry

end prime_pair_divisibility_l4112_411280


namespace brass_composition_ratio_l4112_411230

theorem brass_composition_ratio (total_mass zinc_mass : ℝ) 
  (h_total : total_mass = 100)
  (h_zinc : zinc_mass = 35) :
  (total_mass - zinc_mass) / zinc_mass = 13 / 7 := by
  sorry

end brass_composition_ratio_l4112_411230


namespace max_y_coordinate_l4112_411272

theorem max_y_coordinate (x y : ℝ) : 
  (x^2 / 49) + ((y - 3)^2 / 25) = 0 → y ≤ 3 :=
by sorry

end max_y_coordinate_l4112_411272


namespace min_value_theorem_l4112_411277

/-- The line equation ax - by + 3 = 0 --/
def line_equation (a b x y : ℝ) : Prop := a * x - b * y + 3 = 0

/-- The circle equation x^2 + y^2 + 2x - 4y + 1 = 0 --/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y + 1 = 0

/-- The line divides the area of the circle in half --/
def line_bisects_circle (a b : ℝ) : Prop := 
  ∃ x y : ℝ, line_equation a b x y ∧ circle_equation x y

/-- The main theorem --/
theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 1) 
  (h_bisect : line_bisects_circle a b) : 
  (∀ a' b' : ℝ, a' > 0 → b' > 1 → line_bisects_circle a' b' → 
    2/a + 1/(b-1) ≤ 2/a' + 1/(b'-1)) → 
  2/a + 1/(b-1) = 8 :=
sorry

end min_value_theorem_l4112_411277


namespace squirrel_stockpiling_days_l4112_411232

/-- The number of busy squirrels -/
def busy_squirrels : ℕ := 2

/-- The number of nuts each busy squirrel stockpiles per day -/
def busy_squirrel_nuts_per_day : ℕ := 30

/-- The number of sleepy squirrels -/
def sleepy_squirrels : ℕ := 1

/-- The number of nuts the sleepy squirrel stockpiles per day -/
def sleepy_squirrel_nuts_per_day : ℕ := 20

/-- The total number of nuts found in Mason's car -/
def total_nuts : ℕ := 3200

/-- The number of days squirrels have been stockpiling nuts -/
def stockpiling_days : ℕ := 40

theorem squirrel_stockpiling_days :
  stockpiling_days * (busy_squirrels * busy_squirrel_nuts_per_day + sleepy_squirrels * sleepy_squirrel_nuts_per_day) = total_nuts :=
by sorry

end squirrel_stockpiling_days_l4112_411232


namespace hyperbola_eccentricity_l4112_411200

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1 is √(1 + b²/a²) -/
theorem hyperbola_eccentricity (a b : ℝ) (h : a ≠ 0 ∧ b ≠ 0) :
  let e := Real.sqrt (1 + b^2 / a^2)
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) → e = Real.sqrt 2 :=
by
  sorry

end hyperbola_eccentricity_l4112_411200


namespace sum_radii_greater_incircle_radius_l4112_411270

-- Define the triangle and circles
variable (A B C : EuclideanPlane) (S S₁ S₂ : Circle EuclideanPlane)

-- Define the radii
variable (r r₁ r₂ : ℝ)

-- Assumptions
variable (h_triangle : Triangle A B C)
variable (h_incircle : S.IsIncircle h_triangle)
variable (h_S₁_tangent : S₁.IsTangentTo (SegmentND A B) ∧ S₁.IsTangentTo (SegmentND A C))
variable (h_S₂_tangent : S₂.IsTangentTo (SegmentND A B) ∧ S₂.IsTangentTo (SegmentND B C))
variable (h_S₁S₂_tangent : S₁.IsExternallyTangentTo S₂)
variable (h_r : S.radius = r)
variable (h_r₁ : S₁.radius = r₁)
variable (h_r₂ : S₂.radius = r₂)

-- Theorem statement
theorem sum_radii_greater_incircle_radius : r₁ + r₂ > r := by
  sorry

end sum_radii_greater_incircle_radius_l4112_411270


namespace benzene_required_for_reaction_l4112_411241

-- Define the molecules and their molar ratios in the reaction
structure Reaction :=
  (benzene : ℚ)
  (methane : ℚ)
  (toluene : ℚ)
  (hydrogen : ℚ)

-- Define the balanced equation
def balanced_equation : Reaction := ⟨1, 1, 1, 1⟩

-- Theorem statement
theorem benzene_required_for_reaction 
  (methane_input : ℚ) 
  (hydrogen_output : ℚ) :
  methane_input = 2 →
  hydrogen_output = 2 →
  methane_input * balanced_equation.benzene / balanced_equation.methane = 2 :=
by sorry

end benzene_required_for_reaction_l4112_411241


namespace certain_number_for_prime_squared_l4112_411278

theorem certain_number_for_prime_squared (p : ℕ) (h_prime : Nat.Prime p) (h_gt_3 : p > 3) :
  ∃! n : ℕ, (p^2 + n) % 12 = 2 ∧ n = 1 := by
  sorry

end certain_number_for_prime_squared_l4112_411278


namespace find_a_l4112_411257

theorem find_a (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f x = a * x + 6) →
  f (-1) = 8 →
  a = -2 := by sorry

end find_a_l4112_411257


namespace inverse_function_property_l4112_411269

theorem inverse_function_property (f : ℝ → ℝ) (hf : Function.Bijective f) 
  (h : ∀ x : ℝ, f (x + 1) + f (-x - 4) = 2) :
  ∀ x : ℝ, (Function.invFun f) (2011 - x) + (Function.invFun f) (x - 2009) = -3 := by
  sorry

end inverse_function_property_l4112_411269


namespace jiwon_distance_to_school_l4112_411274

/-- The distance from Taehong's house to school in kilometers -/
def taehong_distance : ℝ := 1.05

/-- The difference between Taehong's and Jiwon's distances in kilometers -/
def distance_difference : ℝ := 0.46

/-- The distance from Jiwon's house to school in kilometers -/
def jiwon_distance : ℝ := taehong_distance - distance_difference

theorem jiwon_distance_to_school :
  jiwon_distance = 0.59 := by sorry

end jiwon_distance_to_school_l4112_411274


namespace gcd_problem_l4112_411276

theorem gcd_problem (a : ℤ) (h : ∃ k : ℤ, a = (2*k + 1) * 1183) :
  Int.gcd (2*a^2 + 29*a + 65) (a + 13) = 26 := by
  sorry

end gcd_problem_l4112_411276


namespace original_number_proof_l4112_411243

theorem original_number_proof : 
  ∃ x : ℝ, (x * 1.4 = 700) ∧ (x = 500) := by
  sorry

end original_number_proof_l4112_411243


namespace student_age_ratio_l4112_411214

/-- Represents the number of students in different age groups -/
structure SchoolPopulation where
  total : ℕ
  below_eight : ℕ
  eight_years : ℕ
  above_eight : ℕ

/-- Theorem stating the ratio of students above 8 years to 8 years old -/
theorem student_age_ratio (school : SchoolPopulation) 
  (h1 : school.total = 80)
  (h2 : school.below_eight = school.total / 4)
  (h3 : school.eight_years = 36)
  (h4 : school.above_eight = school.total - school.below_eight - school.eight_years) :
  (school.above_eight : ℚ) / school.eight_years = 2 / 3 := by
  sorry

end student_age_ratio_l4112_411214


namespace triangle_area_proof_l4112_411203

/-- The curve function f(x) = (x-5)^2 * (x+3) -/
def f (x : ℝ) : ℝ := (x - 5)^2 * (x + 3)

/-- The area of the triangle bounded by the axes and the curve y = f(x) -/
def triangle_area : ℝ := 300

theorem triangle_area_proof : 
  triangle_area = 300 := by sorry

end triangle_area_proof_l4112_411203


namespace winter_migration_l4112_411263

/-- The number of bird families living near the mountain -/
def mountain_families : ℕ := 18

/-- The number of bird families that flew to Africa -/
def africa_families : ℕ := 38

/-- The number of bird families that flew to Asia -/
def asia_families : ℕ := 80

/-- The total number of bird families that flew away for the winter -/
def total_migrated_families : ℕ := africa_families + asia_families

theorem winter_migration :
  total_migrated_families = 118 :=
by sorry

end winter_migration_l4112_411263


namespace abs_value_of_z_l4112_411252

/-- The absolute value of the complex number z = (2i)/(1+i) - 2i is √2 -/
theorem abs_value_of_z : Complex.abs ((2 * Complex.I) / (1 + Complex.I) - 2 * Complex.I) = Real.sqrt 2 := by
  sorry

end abs_value_of_z_l4112_411252


namespace max_edges_no_cycle4_l4112_411299

/-- A graph with no cycle of length 4 -/
structure NoCycle4Graph where
  vertexCount : ℕ
  edgeCount : ℕ
  noCycle4 : Bool

/-- The maximum number of edges in a graph with 8 vertices and no 4-cycle -/
def maxEdgesNoCycle4 (g : NoCycle4Graph) : Prop :=
  g.vertexCount = 8 ∧ g.noCycle4 = true → g.edgeCount ≤ 25

/-- Theorem stating the maximum number of edges in a graph with 8 vertices and no 4-cycle -/
theorem max_edges_no_cycle4 (g : NoCycle4Graph) : maxEdgesNoCycle4 g := by
  sorry

#check max_edges_no_cycle4

end max_edges_no_cycle4_l4112_411299


namespace distance_difference_l4112_411290

/-- The distance Aleena biked in 5 hours -/
def aleena_distance : ℕ := 75

/-- The distance Bob biked in 5 hours -/
def bob_distance : ℕ := 60

/-- Theorem stating the difference between Aleena's and Bob's distances after 5 hours -/
theorem distance_difference : aleena_distance - bob_distance = 15 := by
  sorry

end distance_difference_l4112_411290


namespace complement_of_M_in_U_l4112_411228

universe u

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 2, 4}

theorem complement_of_M_in_U : U \ M = {3, 5, 6} := by sorry

end complement_of_M_in_U_l4112_411228


namespace parabola_equation_after_coordinate_shift_l4112_411259

/-- Represents a parabola in a 2D coordinate system -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ
  eq : ℝ → ℝ := λ x => a * (x - h)^2 + k

/-- Represents a 2D coordinate system -/
structure CoordinateSystem where
  origin : ℝ × ℝ

/-- Translates a point from one coordinate system to another -/
def translate (p : ℝ × ℝ) (old_sys new_sys : CoordinateSystem) : ℝ × ℝ :=
  (p.1 - (new_sys.origin.1 - old_sys.origin.1), 
   p.2 - (new_sys.origin.2 - old_sys.origin.2))

theorem parabola_equation_after_coordinate_shift 
  (p : Parabola) 
  (old_sys new_sys : CoordinateSystem) :
  p.a = 3 ∧ 
  p.h = 0 ∧ 
  p.k = 0 ∧
  new_sys.origin = (-1, -1) →
  ∀ x y : ℝ, 
    (translate (x, y) new_sys old_sys).2 = p.eq (translate (x, y) new_sys old_sys).1 ↔
    y = 3 * (x + 1)^2 - 1 := by
  sorry

end parabola_equation_after_coordinate_shift_l4112_411259


namespace real_roots_imply_real_roots_l4112_411246

/-- Given a quadratic equation x^2 + px + q = 0 with real roots, 
    prove that related equations also have real roots. -/
theorem real_roots_imply_real_roots 
  (p q k x₁ x₂ : ℝ) 
  (hk : k ≠ 0) 
  (hx : x₁^2 + p*x₁ + q = 0 ∧ x₂^2 + p*x₂ + q = 0) :
  ∃ (y₁ y₂ z₁ z₂ z₁' z₂' : ℝ), 
    (y₁^2 + (k + 1/k)*p*y₁ + p^2 + q*(k - 1/k)^2 = 0 ∧ 
     y₂^2 + (k + 1/k)*p*y₂ + p^2 + q*(k - 1/k)^2 = 0) ∧
    (z₁^2 - y₁*z₁ + q = 0 ∧ z₂^2 - y₁*z₂ + q = 0) ∧
    (z₁'^2 - y₂*z₁' + q = 0 ∧ z₂'^2 - y₂*z₂' + q = 0) ∧
    y₁ = k*x₁ + (1/k)*x₂ ∧ 
    y₂ = k*x₂ + (1/k)*x₁ ∧
    z₁ = k*x₁ ∧ 
    z₂ = (1/k)*x₂ ∧ 
    z₁' = k*x₂ ∧ 
    z₂' = (1/k)*x₁ := by
  sorry

end real_roots_imply_real_roots_l4112_411246


namespace ferry_hat_count_l4112_411201

theorem ferry_hat_count :
  ∀ (total_adults : ℕ) (children : ℕ) 
    (women_hat_percent : ℚ) (men_hat_percent : ℚ) (children_hat_percent : ℚ),
  total_adults = 3000 →
  children = 500 →
  women_hat_percent = 25 / 100 →
  men_hat_percent = 15 / 100 →
  children_hat_percent = 30 / 100 →
  ∃ (women : ℕ) (men : ℕ),
    women = men ∧
    women + men = total_adults ∧
    (↑women * women_hat_percent + ↑men * men_hat_percent + ↑children * children_hat_percent : ℚ) = 750 :=
by sorry

end ferry_hat_count_l4112_411201


namespace star_polygon_n_value_l4112_411253

/-- Represents an n-pointed regular star polygon -/
structure RegularStarPolygon where
  n : ℕ
  angle_A : ℝ
  angle_B : ℝ

/-- Properties of the regular star polygon -/
def is_valid_star_polygon (star : RegularStarPolygon) : Prop :=
  star.n > 0 ∧
  star.angle_A > 0 ∧
  star.angle_B > 0 ∧
  star.angle_A = star.angle_B - 15 ∧
  star.n * (star.angle_A + star.angle_B) = 360

theorem star_polygon_n_value (star : RegularStarPolygon) 
  (h : is_valid_star_polygon star) : star.n = 24 :=
by sorry

end star_polygon_n_value_l4112_411253

import Mathlib

namespace largest_gcd_of_sum_1008_l1460_146038

theorem largest_gcd_of_sum_1008 (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1008) :
  ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ x + y = 1008 ∧ Nat.gcd x y = 504 ∧ 
  ∀ (c d : ℕ), c > 0 → d > 0 → c + d = 1008 → Nat.gcd c d ≤ 504 :=
sorry

end largest_gcd_of_sum_1008_l1460_146038


namespace solution_bounds_and_expression_l1460_146030

def system_of_equations (x y m : ℝ) : Prop :=
  3 * (x + 1) / 2 + y = 2 ∧ 3 * x - m = 2 * y

theorem solution_bounds_and_expression (x y m : ℝ) 
  (h_system : system_of_equations x y m) 
  (h_x_bound : x ≤ 1) 
  (h_y_bound : y ≤ 1) : 
  (-3 ≤ m ∧ m ≤ 5) ∧ 
  |x - 1| + |y - 1| + |m + 3| + |m - 5| - |x + y - 2| = 8 := by
  sorry

end solution_bounds_and_expression_l1460_146030


namespace janice_purchase_l1460_146071

theorem janice_purchase (a b c : ℕ) : 
  a + b + c = 50 →
  50 * a + 400 * b + 500 * c = 10000 →
  a = 23 :=
by sorry

end janice_purchase_l1460_146071


namespace distinguishable_triangles_count_l1460_146053

/-- Represents the number of available colors for triangles -/
def total_colors : ℕ := 8

/-- Represents the number of colors available for corner triangles -/
def corner_colors : ℕ := total_colors - 1

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- Calculates the number of distinguishable large triangles -/
def distinguishable_triangles : ℕ :=
  corner_colors +  -- All corners same color
  (corner_colors * (corner_colors - 1)) +  -- Two corners same color
  choose corner_colors 3  -- All corners different colors

theorem distinguishable_triangles_count :
  distinguishable_triangles = 84 :=
sorry

end distinguishable_triangles_count_l1460_146053


namespace consecutive_numbers_problem_l1460_146088

theorem consecutive_numbers_problem (x y z : ℤ) : 
  (y = z + 1) →  -- x, y, and z are consecutive
  (x = y + 1) →  -- x, y, and z are consecutive
  (x > y) →      -- x > y > z
  (y > z) →      -- x > y > z
  (2*x + 3*y + 3*z = 5*y + 11) →  -- given equation
  (z = 3) →      -- given value of z
  y = 4 :=
by sorry

end consecutive_numbers_problem_l1460_146088


namespace arctan_sum_tan_l1460_146046

theorem arctan_sum_tan (x y : Real) :
  x = 45 * π / 180 →
  y = 30 * π / 180 →
  Real.arctan (Real.tan x + 2 * Real.tan y) = 75 * π / 180 := by
  sorry

end arctan_sum_tan_l1460_146046


namespace machine_values_after_two_years_l1460_146027

def machineValue (initialValue : ℝ) (depreciationRate : ℝ) (years : ℕ) : ℝ :=
  initialValue - (initialValue * depreciationRate * years)

def combinedValue (valueA valueB valueC : ℝ) : ℝ :=
  valueA + valueB + valueC

theorem machine_values_after_two_years :
  let machineA := machineValue 8000 0.20 2
  let machineB := machineValue 10000 0.15 2
  let machineC := machineValue 12000 0.10 2
  combinedValue machineA machineB machineC = 21400 := by
  sorry

end machine_values_after_two_years_l1460_146027


namespace t_shape_perimeter_l1460_146092

/-- The perimeter of a T shape formed by two rectangles with given dimensions and overlap -/
theorem t_shape_perimeter (horizontal_width horizontal_height vertical_width vertical_height overlap : ℝ) :
  horizontal_width = 3 →
  horizontal_height = 5 →
  vertical_width = 2 →
  vertical_height = 4 →
  overlap = 1 →
  2 * (horizontal_width + horizontal_height) + 2 * (vertical_width + vertical_height) - 2 * overlap = 26 := by
  sorry

#check t_shape_perimeter

end t_shape_perimeter_l1460_146092


namespace blueberry_picking_l1460_146040

theorem blueberry_picking (annie kathryn ben : ℕ) 
  (h1 : kathryn = annie + 2)
  (h2 : ben = kathryn - 3)
  (h3 : annie + kathryn + ben = 25) :
  annie = 8 := by
sorry

end blueberry_picking_l1460_146040


namespace right_triangle_rational_sides_equiv_arithmetic_progression_l1460_146019

theorem right_triangle_rational_sides_equiv_arithmetic_progression (d : ℤ) :
  (∃ (a b c : ℚ), a^2 + b^2 = c^2 ∧ (1/2) * a * b = d) ↔
  (∃ (x y z : ℚ), 2 * y^2 = x^2 + z^2) :=
sorry

end right_triangle_rational_sides_equiv_arithmetic_progression_l1460_146019


namespace quadratic_function_minimum_l1460_146003

theorem quadratic_function_minimum (a b c : ℝ) (x₀ : ℝ) (ha : a > 0) (hx₀ : 2 * a * x₀ + b = 0) :
  ¬ (∀ x : ℝ, a * x^2 + b * x + c ≤ a * x₀^2 + b * x₀ + c) := by
  sorry

end quadratic_function_minimum_l1460_146003


namespace complex_equation_solution_l1460_146075

theorem complex_equation_solution (z : ℂ) : z + z * Complex.I = 1 + 5 * Complex.I → z = 3 + 2 * Complex.I := by
  sorry

end complex_equation_solution_l1460_146075


namespace spinner_probability_l1460_146015

theorem spinner_probability (p : ℝ) (n : ℕ) : 
  p = 3/4 → (p^n = 0.5625) → n = 2 := by
  sorry

end spinner_probability_l1460_146015


namespace retirement_total_is_70_l1460_146093

/-- Represents the retirement eligibility rule for a company -/
structure RetirementRule where
  hireYear : ℕ
  hireAge : ℕ
  retirementYear : ℕ

/-- Calculates the required total of age and years of employment for retirement -/
def requiredTotal (rule : RetirementRule) : ℕ :=
  let ageAtRetirement := rule.hireAge + (rule.retirementYear - rule.hireYear)
  let yearsOfEmployment := rule.retirementYear - rule.hireYear
  ageAtRetirement + yearsOfEmployment

/-- Theorem stating that the required total for retirement is 70 -/
theorem retirement_total_is_70 (rule : RetirementRule) 
    (h1 : rule.hireYear = 1989)
    (h2 : rule.hireAge = 32)
    (h3 : rule.retirementYear = 2008) :
  requiredTotal rule = 70 := by
  sorry

end retirement_total_is_70_l1460_146093


namespace geometric_sequence_first_term_l1460_146069

/-- A geometric sequence with third term 3 and fifth term 27 has first term 1/3 -/
theorem geometric_sequence_first_term (a : ℝ) (r : ℝ) :
  a * r^2 = 3 → a * r^4 = 27 → a = 1/3 := by
  sorry

end geometric_sequence_first_term_l1460_146069


namespace smallest_apocalyptic_number_l1460_146074

/-- A number is apocalyptic if it has 6 different positive divisors that sum to 3528 -/
def IsApocalyptic (n : ℕ) : Prop :=
  ∃ (d₁ d₂ d₃ d₄ d₅ d₆ : ℕ),
    d₁ ≠ d₂ ∧ d₁ ≠ d₃ ∧ d₁ ≠ d₄ ∧ d₁ ≠ d₅ ∧ d₁ ≠ d₆ ∧
    d₂ ≠ d₃ ∧ d₂ ≠ d₄ ∧ d₂ ≠ d₅ ∧ d₂ ≠ d₆ ∧
    d₃ ≠ d₄ ∧ d₃ ≠ d₅ ∧ d₃ ≠ d₆ ∧
    d₄ ≠ d₅ ∧ d₄ ≠ d₆ ∧
    d₅ ≠ d₆ ∧
    d₁ > 0 ∧ d₂ > 0 ∧ d₃ > 0 ∧ d₄ > 0 ∧ d₅ > 0 ∧ d₆ > 0 ∧
    d₁ ∣ n ∧ d₂ ∣ n ∧ d₃ ∣ n ∧ d₄ ∣ n ∧ d₅ ∣ n ∧ d₆ ∣ n ∧
    d₁ + d₂ + d₃ + d₄ + d₅ + d₆ = 3528

theorem smallest_apocalyptic_number :
  IsApocalyptic 1440 ∧ ∀ m : ℕ, m < 1440 → ¬IsApocalyptic m := by
  sorry

end smallest_apocalyptic_number_l1460_146074


namespace water_usage_fraction_l1460_146013

theorem water_usage_fraction (initial_water : ℚ) (car_water : ℚ) (num_cars : ℕ) 
  (plant_water_diff : ℚ) (plate_clothes_water : ℚ) : 
  initial_water = 65 → 
  car_water = 7 → 
  num_cars = 2 → 
  plant_water_diff = 11 → 
  plate_clothes_water = 24 → 
  let total_car_water := car_water * num_cars
  let plant_water := total_car_water - plant_water_diff
  let total_used_water := total_car_water + plant_water
  let remaining_water := initial_water - total_used_water
  plate_clothes_water / remaining_water = 1 / 2 := by
sorry

end water_usage_fraction_l1460_146013


namespace function_positive_iff_a_greater_half_l1460_146039

/-- The function f(x) = ax² - 2x + 2 is positive for all x in (1, 4) if and only if a > 1/2 -/
theorem function_positive_iff_a_greater_half (a : ℝ) :
  (∀ x : ℝ, 1 < x → x < 4 → a * x^2 - 2*x + 2 > 0) ↔ a > 1/2 :=
by sorry

end function_positive_iff_a_greater_half_l1460_146039


namespace marble_difference_l1460_146082

theorem marble_difference (connie_marbles juan_marbles : ℕ) 
  (h1 : connie_marbles = 39)
  (h2 : juan_marbles = 64)
  : juan_marbles - connie_marbles = 25 := by
  sorry

end marble_difference_l1460_146082


namespace smallest_divisible_by_1_to_12_and_15_l1460_146033

theorem smallest_divisible_by_1_to_12_and_15 : ∃ n : ℕ, 
  (∀ k : ℕ, k ≤ 12 → k ∣ n) ∧ 
  (15 ∣ n) ∧ 
  (∀ m : ℕ, m < n → ¬(∀ k : ℕ, k ≤ 12 → k ∣ m) ∨ ¬(15 ∣ m)) :=
by
  -- The proof goes here
  sorry

end smallest_divisible_by_1_to_12_and_15_l1460_146033


namespace monotonic_quadratic_condition_l1460_146076

/-- A function f is monotonic on an interval [a, b] if it is either
    non-decreasing or non-increasing on that interval. -/
def IsMonotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  (∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≤ f y) ∨
  (∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f y ≤ f x)

/-- The quadratic function f(x) = x^2 - 2ax + 3 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 3

theorem monotonic_quadratic_condition :
  ∀ a : ℝ, IsMonotonic (f a) 2 3 ↔ (a ≤ 2 ∨ a ≥ 3) :=
by sorry

end monotonic_quadratic_condition_l1460_146076


namespace infinitely_many_special_triangles_l1460_146023

/-- A triangle with integer area formed by square roots of distinct non-square integers -/
structure SpecialTriangle where
  a₁ : ℕ+
  a₂ : ℕ+
  a₃ : ℕ+
  distinct : a₁ ≠ a₂ ∧ a₁ ≠ a₃ ∧ a₂ ≠ a₃
  not_squares : ¬∃ m : ℕ, a₁ = m^2 ∧ ¬∃ n : ℕ, a₂ = n^2 ∧ ¬∃ k : ℕ, a₃ = k^2
  triangle_inequality : Real.sqrt a₁.val + Real.sqrt a₂.val > Real.sqrt a₃.val ∧
                        Real.sqrt a₁.val + Real.sqrt a₃.val > Real.sqrt a₂.val ∧
                        Real.sqrt a₂.val + Real.sqrt a₃.val > Real.sqrt a₁.val
  integer_area : ∃ S : ℕ, 16 * S^2 = (a₁ + a₂ + a₃)^2 - 2 * (a₁^2 + a₂^2 + a₃^2)

/-- There exist infinitely many SpecialTriangles -/
theorem infinitely_many_special_triangles : 
  ∀ n : ℕ, ∃ (triangles : Fin n → SpecialTriangle), 
    ∀ i j : Fin n, i ≠ j → 
      ¬∃ (k : ℚ), (k * (triangles i).a₁ : ℚ) = (triangles j).a₁ ∧ 
                   (k * (triangles i).a₂ : ℚ) = (triangles j).a₂ ∧ 
                   (k * (triangles i).a₃ : ℚ) = (triangles j).a₃ :=
sorry

end infinitely_many_special_triangles_l1460_146023


namespace greatest_lower_bound_l1460_146063

theorem greatest_lower_bound (x y : ℝ) (h1 : x ≠ y) (h2 : x * y = 2) :
  ((x + y)^2 - 6) * ((x - y)^2 + 8) / (x - y)^2 ≥ 18 ∧
  ∀ C > 18, ∃ x y : ℝ, x ≠ y ∧ x * y = 2 ∧
    ((x + y)^2 - 6) * ((x - y)^2 + 8) / (x - y)^2 < C :=
by sorry

end greatest_lower_bound_l1460_146063


namespace percent_relation_l1460_146097

theorem percent_relation (x y : ℝ) (h : 0.3 * (x - y) = 0.2 * (x + y)) : y = 0.2 * x := by
  sorry

end percent_relation_l1460_146097


namespace divisibility_condition_l1460_146006

theorem divisibility_condition (a b : ℕ+) : 
  (a * b^2 + b + 7) ∣ (a^2 * b + a + b) →
  ((a = 11 ∧ b = 1) ∨ (a = 49 ∧ b = 1) ∨ (∃ k : ℕ+, a = 7 * k^2 ∧ b = 7 * k)) :=
sorry

end divisibility_condition_l1460_146006


namespace derivative_x_exp_cos_l1460_146035

/-- The derivative of xe^(cos x) is -x sin x * e^(cos x) + e^(cos x) -/
theorem derivative_x_exp_cos (x : ℝ) :
  deriv (fun x => x * Real.exp (Real.cos x)) x =
  -x * Real.sin x * Real.exp (Real.cos x) + Real.exp (Real.cos x) := by
  sorry

end derivative_x_exp_cos_l1460_146035


namespace divisibility_implies_equality_l1460_146005

theorem divisibility_implies_equality (a b n : ℕ) :
  (∀ k : ℕ, k ≠ b → (b - k) ∣ (a - k^n)) →
  a = b^n :=
by sorry

end divisibility_implies_equality_l1460_146005


namespace zero_in_A_l1460_146004

def A : Set ℝ := {x | x * (x - 1) = 0}

theorem zero_in_A : 0 ∈ A := by
  sorry

end zero_in_A_l1460_146004


namespace johnson_family_seating_l1460_146024

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

-- Define the number of sons and daughters
def num_sons : ℕ := 5
def num_daughters : ℕ := 4

-- Define the total number of children
def total_children : ℕ := num_sons + num_daughters

-- Define the function to calculate the number of seating arrangements
def seating_arrangements : ℕ :=
  factorial total_children - (factorial num_sons * factorial num_daughters)

-- Theorem statement
theorem johnson_family_seating :
  seating_arrangements = 360000 :=
sorry

end johnson_family_seating_l1460_146024


namespace power_inequality_l1460_146086

theorem power_inequality (a : ℝ) (n : ℕ) :
  (a > 1 → a^n > 1) ∧ (a < 1 → a^n < 1) := by
  sorry

end power_inequality_l1460_146086


namespace consecutive_negative_integers_sum_l1460_146008

theorem consecutive_negative_integers_sum (n : ℤ) : 
  n < 0 ∧ n * (n + 1) = 812 → n + (n + 1) = -57 := by
  sorry

end consecutive_negative_integers_sum_l1460_146008


namespace simplify_expression_l1460_146056

theorem simplify_expression : (2^8 + 5^3) * (2^2 - (-1)^5)^7 = 29765625 := by
  sorry

end simplify_expression_l1460_146056


namespace hyperbola_eccentricity_l1460_146045

/-- The eccentricity of a hyperbola with the given condition -/
theorem hyperbola_eccentricity (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) : 
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (b = (a + c) / 2) →
  (c^2 = a^2 + b^2) →
  (c / a : ℝ) = 5/3 := by
sorry

end hyperbola_eccentricity_l1460_146045


namespace equation_value_l1460_146098

theorem equation_value (a b c : ℝ) 
  (eq1 : 3 * a - 2 * b - 2 * c = 30)
  (eq2 : a + b + c = 10) :
  Real.sqrt (3 * a) - Real.sqrt (2 * b + 2 * c) = Real.sqrt 30 := by
  sorry

end equation_value_l1460_146098


namespace alphametic_puzzle_solution_l1460_146043

def is_valid_assignment (K O A L V D : ℕ) : Prop :=
  K ≠ O ∧ K ≠ A ∧ K ≠ L ∧ K ≠ V ∧ K ≠ D ∧
  O ≠ A ∧ O ≠ L ∧ O ≠ V ∧ O ≠ D ∧
  A ≠ L ∧ A ≠ V ∧ A ≠ D ∧
  L ≠ V ∧ L ≠ D ∧
  V ≠ D ∧
  K < 10 ∧ O < 10 ∧ A < 10 ∧ L < 10 ∧ V < 10 ∧ D < 10

def satisfies_equation (K O A L V D : ℕ) : Prop :=
  1000 * K + 100 * O + 10 * K + A +
  1000 * K + 100 * O + 10 * L + A =
  1000 * V + 100 * O + 10 * D + A

theorem alphametic_puzzle_solution :
  ∃! (K O A L V D : ℕ), 
    is_valid_assignment K O A L V D ∧
    satisfies_equation K O A L V D ∧
    K = 3 ∧ O = 9 ∧ A = 0 ∧ L = 8 ∧ V = 7 ∧ D = 1 :=
by sorry

end alphametic_puzzle_solution_l1460_146043


namespace min_value_quadratic_l1460_146002

theorem min_value_quadratic (x : ℝ) :
  ∃ (min_y : ℝ), ∀ (y : ℝ), y = 2 * x^2 + 8 * x + 18 → y ≥ min_y ∧ min_y = 10 :=
by sorry

end min_value_quadratic_l1460_146002


namespace students_in_all_events_l1460_146022

theorem students_in_all_events 
  (total_students : ℕ) 
  (event_A_participants : ℕ) 
  (event_B_participants : ℕ) 
  (h1 : total_students = 45)
  (h2 : event_A_participants = 39)
  (h3 : event_B_participants = 28)
  (h4 : event_A_participants + event_B_participants - total_students ≤ event_A_participants)
  (h5 : event_A_participants + event_B_participants - total_students ≤ event_B_participants) :
  event_A_participants + event_B_participants - total_students = 22 := by
  sorry

end students_in_all_events_l1460_146022


namespace twenty_fourth_digit_is_8_l1460_146041

-- Define the decimal representations of 1/7 and 1/9
def decimal_1_7 : ℚ := 1 / 7
def decimal_1_9 : ℚ := 1 / 9

-- Define the sum of the decimal representations
def sum_decimals : ℚ := decimal_1_7 + decimal_1_9

-- Function to get the nth digit after the decimal point
def nth_digit_after_decimal (q : ℚ) (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem twenty_fourth_digit_is_8 :
  nth_digit_after_decimal sum_decimals 24 = 8 := by sorry

end twenty_fourth_digit_is_8_l1460_146041


namespace hyperbola_eccentricity_theorem_l1460_146052

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  pos_a : 0 < a
  pos_b : 0 < b

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola a b) : ℝ := sorry

/-- The right focus of a hyperbola -/
def right_focus (h : Hyperbola a b) : ℝ × ℝ := sorry

/-- Predicate to check if a point is on the hyperbola -/
def on_hyperbola (h : Hyperbola a b) (p : ℝ × ℝ) : Prop :=
  (p.1^2 / a^2) - (p.2^2 / b^2) = 1

/-- Predicate to check if four points form a parallelogram -/
def is_parallelogram (p q r s : ℝ × ℝ) : Prop := sorry

/-- The area of a quadrilateral given by four points -/
def quadrilateral_area (p q r s : ℝ × ℝ) : ℝ := sorry

/-- The main theorem -/
theorem hyperbola_eccentricity_theorem (a b c : ℝ) (h : Hyperbola a b) 
  (m n : ℝ × ℝ) (hm : on_hyperbola h m) (hn : on_hyperbola h n)
  (hpara : is_parallelogram (0, 0) (right_focus h) m n)
  (harea : quadrilateral_area (0, 0) (right_focus h) m n = Real.sqrt 3 * b * c) :
  eccentricity h = 4 := by sorry

end hyperbola_eccentricity_theorem_l1460_146052


namespace sqrt_750_minus_29_cube_l1460_146091

theorem sqrt_750_minus_29_cube (a b : ℕ+) :
  (Real.sqrt 750 - 29 : ℝ) = (Real.sqrt a.val - b.val : ℝ)^3 →
  a.val + b.val = 28 := by sorry

end sqrt_750_minus_29_cube_l1460_146091


namespace gcd_factorial_eight_ten_l1460_146051

theorem gcd_factorial_eight_ten : Nat.gcd (Nat.factorial 8) (Nat.factorial 10) = Nat.factorial 8 := by
  sorry

end gcd_factorial_eight_ten_l1460_146051


namespace complex_number_real_condition_l1460_146096

theorem complex_number_real_condition (a : ℝ) :
  (∃ (z : ℂ), z = (a + 1) + (a^2 - 1) * I ∧ z.im = 0) ↔ (a = 1 ∨ a = -1) :=
sorry

end complex_number_real_condition_l1460_146096


namespace arithmetic_sequence_a8_l1460_146000

def arithmetic_sequence (a : ℕ → ℝ) := 
  ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

theorem arithmetic_sequence_a8 (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_sum : a 5 + a 6 = 22)
  (h_a3 : a 3 = 7) :
  a 8 = 15 := by
sorry

end arithmetic_sequence_a8_l1460_146000


namespace rugby_team_lineup_count_l1460_146067

/-- The number of ways to form a team lineup -/
def team_lineup_ways (total_members : ℕ) (specialized_kickers : ℕ) (lineup_size : ℕ) : ℕ :=
  specialized_kickers * (Nat.choose (total_members - 1) (lineup_size - 1))

/-- Theorem: The number of ways to form the team lineup is 151164 -/
theorem rugby_team_lineup_count :
  team_lineup_ways 20 2 9 = 151164 := by
  sorry

end rugby_team_lineup_count_l1460_146067


namespace part_one_part_two_l1460_146065

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + a|

-- Part 1
theorem part_one (a : ℝ) :
  (∀ x, f a x ≥ |2*x + 3| ↔ x ∈ Set.Icc (-3) (-1)) →
  a = 0 :=
sorry

-- Part 2
theorem part_two (a : ℝ) :
  (∀ x, f a x + |x - a| ≥ a^2 - 2*a) →
  0 ≤ a ∧ a ≤ 4 :=
sorry

end part_one_part_two_l1460_146065


namespace total_savings_calculation_l1460_146042

-- Define the original prices and discount rates
def chlorine_price : ℝ := 10
def chlorine_discount : ℝ := 0.20
def soap_price : ℝ := 16
def soap_discount : ℝ := 0.25

-- Define the quantities
def chlorine_quantity : ℕ := 3
def soap_quantity : ℕ := 5

-- Theorem statement
theorem total_savings_calculation :
  let chlorine_savings := chlorine_price * chlorine_discount * chlorine_quantity
  let soap_savings := soap_price * soap_discount * soap_quantity
  chlorine_savings + soap_savings = 26 := by
  sorry

end total_savings_calculation_l1460_146042


namespace prob_even_sum_is_31_66_l1460_146032

/-- A set of twelve prime numbers including two even primes -/
def prime_set : Finset ℕ := sorry

/-- The number of prime numbers in the set -/
def n : ℕ := 12

/-- The number of even prime numbers in the set -/
def even_primes : ℕ := 2

/-- The number of primes to be selected -/
def k : ℕ := 5

/-- Predicate to check if a set of natural numbers has an even sum -/
def has_even_sum (s : Finset ℕ) : Prop := Even (s.sum id)

/-- The probability of selecting k primes from prime_set with two even primes such that their sum is even -/
def prob_even_sum : ℚ := sorry

theorem prob_even_sum_is_31_66 : prob_even_sum = 31 / 66 := by sorry

end prob_even_sum_is_31_66_l1460_146032


namespace no_prime_fraction_equality_l1460_146089

theorem no_prime_fraction_equality : ¬∃ (a b c d : ℕ), 
  Prime a ∧ Prime b ∧ Prime c ∧ Prime d ∧
  a < b ∧ b < c ∧ c < d ∧
  (1 : ℚ) / a + (1 : ℚ) / d = (1 : ℚ) / b + (1 : ℚ) / c := by
  sorry

end no_prime_fraction_equality_l1460_146089


namespace larry_lunch_cost_l1460_146090

/-- Calculates the amount spent on lunch given initial amount, final amount, and amount given to brother -/
def lunch_cost (initial : ℕ) (final : ℕ) (given_to_brother : ℕ) : ℕ :=
  initial - final - given_to_brother

/-- Proves that Larry's lunch cost is $5 given the problem conditions -/
theorem larry_lunch_cost :
  lunch_cost 22 15 2 = 5 := by sorry

end larry_lunch_cost_l1460_146090


namespace max_socks_pulled_correct_l1460_146001

/-- Represents the state of socks in the drawer and pulled out -/
structure SockState where
  white_in_drawer : ℕ
  black_in_drawer : ℕ
  white_pulled : ℕ
  black_pulled : ℕ

/-- The initial state of socks -/
def initial_state : SockState :=
  { white_in_drawer := 8
  , black_in_drawer := 15
  , white_pulled := 0
  , black_pulled := 0 }

/-- Predicate to check if more black socks than white socks have been pulled -/
def more_black_than_white (state : SockState) : Prop :=
  state.black_pulled > state.white_pulled

/-- The maximum number of socks that can be pulled -/
def max_socks_pulled : ℕ := 17

/-- Theorem stating the maximum number of socks that can be pulled -/
theorem max_socks_pulled_correct :
  ∀ (state : SockState),
    state.white_in_drawer + state.black_in_drawer + state.white_pulled + state.black_pulled = 23 →
    state.white_pulled + state.black_pulled ≤ max_socks_pulled →
    ¬(more_black_than_white state) :=
  sorry

#check max_socks_pulled_correct

end max_socks_pulled_correct_l1460_146001


namespace train_speed_problem_l1460_146009

theorem train_speed_problem (train_length : ℝ) (crossing_time : ℝ) (speed_ratio : ℝ) :
  train_length = 150 →
  crossing_time = 12 →
  speed_ratio = 3 →
  let slower_speed := (2 * train_length) / (crossing_time * (speed_ratio + 1))
  let faster_speed := speed_ratio * slower_speed
  faster_speed = 18.75 :=
by
  sorry

end train_speed_problem_l1460_146009


namespace problem_solution_l1460_146061

noncomputable section

-- Define the functions
def f (a : ℝ) (x : ℝ) : ℝ := a^x
def g (a m : ℝ) (x : ℝ) : ℝ := a^(2*x) + m
def h (m : ℝ) (x : ℝ) : ℝ := 2^(2*x) + m - 2*m*2^x

-- Define the minimum value function
def H (m : ℝ) : ℝ :=
  if m < 1 then 1 - m
  else if m ≤ 2 then m - m^2
  else 4 - 3*m

theorem problem_solution :
  ∀ (a m : ℝ),
  (a > 0 ∧ a ≠ 1 ∧ m > 0) →
  (∀ (x : ℝ), x ∈ Set.Icc (-1) 1 → f a x ≤ 5/2 ∧ f a x ≥ 0) →
  (f a 1 + f a (-1) = 5/2) →
  (a = 2) ∧
  (∀ (x : ℝ), x ∈ Set.Icc 0 1 → h m x ≥ H m) ∧
  (∀ (x : ℝ), x ∈ Set.Icc 0 1 → |1 - m*(2^x + m/2^x)| ≤ 1 → m ∈ Set.Icc 0 2) :=
by sorry

end problem_solution_l1460_146061


namespace num_arrangements_eq_192_l1460_146034

/-- The number of different arrangements for 7 students in a row with specific conditions -/
def num_arrangements : ℕ :=
  let total_students : ℕ := 7
  let middle_student : ℕ := 1
  let together_students : ℕ := 2
  let remaining_students : ℕ := total_students - middle_student - together_students
  let middle_positions : ℕ := 1
  let together_positions : ℕ := 2 * 4
  let remaining_positions : ℕ := remaining_students.factorial
  middle_positions * together_positions * remaining_positions

/-- Theorem stating that the number of arrangements is 192 -/
theorem num_arrangements_eq_192 : num_arrangements = 192 := by
  sorry

end num_arrangements_eq_192_l1460_146034


namespace train_travel_theorem_l1460_146057

/-- Represents the distance traveled by a train -/
def train_distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

theorem train_travel_theorem (initial_distance initial_time final_time : ℝ) 
  (h1 : initial_distance = 300)
  (h2 : initial_time = 20)
  (h3 : final_time = 600) : 
  train_distance (initial_distance / initial_time) final_time = 9000 := by
  sorry

#check train_travel_theorem

end train_travel_theorem_l1460_146057


namespace fib_150_mod_7_l1460_146059

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

theorem fib_150_mod_7 : fib 150 % 7 = 1 := by
  sorry

end fib_150_mod_7_l1460_146059


namespace min_phi_for_odd_function_l1460_146064

open Real

theorem min_phi_for_odd_function (φ : ℝ) : 
  (φ > 0 ∧ 
   (∀ x, cos (π * x - π * φ - π / 3) = -cos (π * (-x) - π * φ - π / 3))) 
  ↔ 
  φ = 1 / 6 := by
sorry

end min_phi_for_odd_function_l1460_146064


namespace light_flash_duration_l1460_146099

theorem light_flash_duration (flash_interval : ℕ) (num_flashes : ℕ) (seconds_per_hour : ℕ) : 
  flash_interval = 12 →
  num_flashes = 300 →
  seconds_per_hour = 3600 →
  (flash_interval * num_flashes) / seconds_per_hour = 1 := by
  sorry

end light_flash_duration_l1460_146099


namespace gift_combinations_count_l1460_146010

/-- The number of different gift packaging combinations -/
def gift_combinations (wrapping_paper : ℕ) (ribbon : ℕ) (gift_card : ℕ) (gift_box : ℕ) : ℕ :=
  wrapping_paper * ribbon * gift_card * gift_box

/-- Theorem stating the number of gift packaging combinations -/
theorem gift_combinations_count :
  gift_combinations 10 3 4 5 = 600 := by
  sorry

end gift_combinations_count_l1460_146010


namespace matchsticks_count_l1460_146044

/-- The number of matchsticks in a box -/
def initial_matchsticks : ℕ := sorry

/-- The number of matchsticks Elvis uses per square -/
def elvis_matchsticks_per_square : ℕ := 4

/-- The number of squares Elvis makes -/
def elvis_squares : ℕ := 5

/-- The number of matchsticks Ralph uses per square -/
def ralph_matchsticks_per_square : ℕ := 8

/-- The number of squares Ralph makes -/
def ralph_squares : ℕ := 3

/-- The number of matchsticks left in the box -/
def remaining_matchsticks : ℕ := 6

theorem matchsticks_count : initial_matchsticks = 50 := by sorry

end matchsticks_count_l1460_146044


namespace rectangle_perimeter_change_l1460_146016

theorem rectangle_perimeter_change (a b : ℝ) (h : 2 * (1.3 * a + 0.8 * b) = 2 * (a + b)) :
  2 * (0.8 * a + 1.3 * b) = 1.1 * (2 * (a + b)) :=
by sorry

end rectangle_perimeter_change_l1460_146016


namespace chord_length_parabola_l1460_146017

/-- Parabola structure -/
structure Parabola where
  equation : ℝ → ℝ → Prop

/-- Line structure -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Point structure -/
structure Point where
  x : ℝ
  y : ℝ

/-- Theorem statement -/
theorem chord_length_parabola (C : Parabola) (l : Line) (A B : Point) :
  C.equation = (fun x y => x^2 = 4*y) →
  l.intercept = 1 →
  C.equation A.x A.y →
  C.equation B.x B.y →
  (A.y + B.y) / 2 = 5 →
  ∃ k, l.slope = k ∧ k^2 = 2 →
  ∃ AB : ℝ, AB = 6 ∧ AB = Real.sqrt ((A.x - B.x)^2 + (A.y - B.y)^2) := by
  sorry

end chord_length_parabola_l1460_146017


namespace unique_two_digit_multiple_l1460_146084

theorem unique_two_digit_multiple : ∃! s : ℕ, 
  10 ≤ s ∧ s < 100 ∧ (13 * s) % 100 = 52 := by
  sorry

end unique_two_digit_multiple_l1460_146084


namespace quadratic_function_theorem_l1460_146095

/-- A quadratic function -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c

/-- A linear function -/
def LinearFunction (m n : ℝ) : ℝ → ℝ := λ x ↦ m * x + n

/-- The theorem statement -/
theorem quadratic_function_theorem (a b c m n : ℝ) :
  let f := QuadraticFunction a b c
  let g := LinearFunction m n
  (f (-1) = 2) ∧ (g (-1) = 2) ∧ (f 2 = 5) ∧ (g 2 = 5) ∧
  (∃ x₀, ∀ x, f x₀ ≤ f x) ∧ (f x₀ = 1) →
  (f = λ x ↦ x^2 + 1) ∨ (f = λ x ↦ (1/9) * x^2 + (8/9) * x + 25/9) :=
by sorry

end quadratic_function_theorem_l1460_146095


namespace product_inequality_l1460_146026

theorem product_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b) * (b + c) * (c + a) ≥ 8 * a * b * c := by
  sorry

end product_inequality_l1460_146026


namespace final_hair_length_l1460_146085

def hair_length (initial_length cut_length growth_length : ℕ) : ℕ :=
  initial_length - cut_length + growth_length

theorem final_hair_length :
  hair_length 16 11 12 = 17 := by
  sorry

end final_hair_length_l1460_146085


namespace cupcake_distribution_l1460_146011

/-- Given the initial number of cupcakes, the number of eaten cupcakes, and the number of packages,
    calculate the number of cupcakes in each package. -/
def cupcakes_per_package (initial : ℕ) (eaten : ℕ) (packages : ℕ) : ℕ :=
  (initial - eaten) / packages

/-- Theorem stating that given 71 initial cupcakes, 43 eaten cupcakes, and 4 packages,
    the number of cupcakes in each package is 7. -/
theorem cupcake_distribution :
  cupcakes_per_package 71 43 4 = 7 := by
  sorry


end cupcake_distribution_l1460_146011


namespace fence_poles_count_l1460_146020

def side_length : ℝ := 150
def pole_spacing : ℝ := 30

theorem fence_poles_count :
  let perimeter := 4 * side_length
  let poles_count := perimeter / pole_spacing
  poles_count = 20 := by sorry

end fence_poles_count_l1460_146020


namespace first_valid_year_is_2030_l1460_146081

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

def is_valid_year (year : ℕ) : Prop :=
  year > 2022 ∧ sum_of_digits year = 5

theorem first_valid_year_is_2030 :
  is_valid_year 2030 ∧ ∀ y, is_valid_year y → y ≥ 2030 :=
sorry

end first_valid_year_is_2030_l1460_146081


namespace not_necessarily_even_increasing_on_reals_max_at_turning_point_sqrt_convexity_l1460_146048

-- 1. A function f: ℝ → ℝ that satisfies f(-2) = f(2) is not necessarily an even function
theorem not_necessarily_even (f : ℝ → ℝ) (h : f (-2) = f 2) :
  ¬ ∀ x, f (-x) = f x :=
sorry

-- 2. If f: ℝ → ℝ is monotonically increasing on (-∞, 0] and [0, +∞), then f is increasing on ℝ
theorem increasing_on_reals (f : ℝ → ℝ)
  (h1 : ∀ x y, x ≤ y → x ≤ 0 → y ≤ 0 → f x ≤ f y)
  (h2 : ∀ x y, x ≤ y → 0 ≤ x → 0 ≤ y → f x ≤ f y) :
  ∀ x y, x ≤ y → f x ≤ f y :=
sorry

-- 3. If f: [a, b] → ℝ (where a < c < b) is increasing on [a, c) and decreasing on [c, b],
--    then f(c) is the maximum value of f on [a, b]
theorem max_at_turning_point {a b c : ℝ} (h : a < c ∧ c < b) (f : ℝ → ℝ)
  (h1 : ∀ x y, a ≤ x → x < y → y < c → f x ≤ f y)
  (h2 : ∀ x y, c < x → x < y → y ≤ b → f y ≤ f x) :
  ∀ x, a ≤ x → x ≤ b → f x ≤ f c :=
sorry

-- 4. For f(x) = √x and any x₁, x₂ ∈ (0, +∞), (f(x₁) + f(x₂))/2 ≤ f((x₁ + x₂)/2)
theorem sqrt_convexity (x₁ x₂ : ℝ) (h1 : 0 < x₁) (h2 : 0 < x₂) :
  (Real.sqrt x₁ + Real.sqrt x₂) / 2 ≤ Real.sqrt ((x₁ + x₂) / 2) :=
sorry

end not_necessarily_even_increasing_on_reals_max_at_turning_point_sqrt_convexity_l1460_146048


namespace A_contains_B_l1460_146028

def A (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 5

def B (x m : ℝ) : Prop := (x - m + 1) * (x - 2 * m - 1) < 0

theorem A_contains_B (m : ℝ) : 
  (∀ x, B x m → A x) ↔ (m = -2 ∨ (-1 ≤ m ∧ m ≤ 2)) := by sorry

end A_contains_B_l1460_146028


namespace solution_set_correct_l1460_146062

/-- The set of solutions to the equation 1/(x^2 + 13x - 12) + 1/(x^2 + 4x - 12) + 1/(x^2 - 11x - 12) = 0 -/
def solution_set : Set ℝ := {1, -12, 4, -3}

/-- The equation to be solved -/
def equation (x : ℝ) : Prop :=
  1 / (x^2 + 13*x - 12) + 1 / (x^2 + 4*x - 12) + 1 / (x^2 - 11*x - 12) = 0

theorem solution_set_correct :
  ∀ x : ℝ, equation x ↔ x ∈ solution_set := by sorry

end solution_set_correct_l1460_146062


namespace expected_value_of_winnings_l1460_146083

def fair_10_sided_die : Finset ℕ := Finset.range 10

def winnings (roll : ℕ) : ℚ :=
  if roll % 2 = 0 then roll else 0

theorem expected_value_of_winnings :
  (Finset.sum fair_10_sided_die (λ roll => (1 : ℚ) / 10 * winnings roll)) = 3 := by
  sorry

end expected_value_of_winnings_l1460_146083


namespace simple_interest_theorem_l1460_146087

def simple_interest_problem (principal rate time : ℝ) : Prop :=
  let simple_interest := principal * rate * time / 100
  principal - simple_interest = 2080

theorem simple_interest_theorem :
  simple_interest_problem 2600 4 5 := by
  sorry

end simple_interest_theorem_l1460_146087


namespace tangent_segments_area_l1460_146078

theorem tangent_segments_area (r : ℝ) (l : ℝ) (h1 : r = 3) (h2 : l = 4) :
  let inner_radius := r
  let outer_radius := Real.sqrt (r^2 + (l/2)^2)
  (π * outer_radius^2 - π * inner_radius^2) = 4 * π :=
by sorry

end tangent_segments_area_l1460_146078


namespace power_equation_solution_l1460_146060

theorem power_equation_solution (n : ℕ) : 2^n = 2 * 4^2 * 16^3 → n = 17 := by
  sorry

end power_equation_solution_l1460_146060


namespace number_puzzle_l1460_146036

theorem number_puzzle : ∃ x : ℝ, x^2 + 50 = (x - 10)^2 ∧ x = 2.5 := by
  sorry

end number_puzzle_l1460_146036


namespace eunji_lives_higher_l1460_146018

def yoojung_floor : ℕ := 17
def eunji_floor : ℕ := 25

theorem eunji_lives_higher : eunji_floor > yoojung_floor := by
  sorry

end eunji_lives_higher_l1460_146018


namespace cake_eaten_percentage_l1460_146066

theorem cake_eaten_percentage (total_pieces : ℕ) (sisters : ℕ) (pieces_per_sister : ℕ) 
  (h1 : total_pieces = 240)
  (h2 : sisters = 3)
  (h3 : pieces_per_sister = 32) :
  (total_pieces - sisters * pieces_per_sister) / total_pieces * 100 = 60 := by
  sorry

#check cake_eaten_percentage

end cake_eaten_percentage_l1460_146066


namespace complex_fraction_simplification_l1460_146058

theorem complex_fraction_simplification :
  let z₁ : ℂ := 4 + 6 * I
  let z₂ : ℂ := 4 - 6 * I
  (z₁ / z₂) + (z₂ / z₁) = (-10 : ℚ) / 13 := by
  sorry

end complex_fraction_simplification_l1460_146058


namespace fixed_point_of_exponential_function_l1460_146079

theorem fixed_point_of_exponential_function (a : ℝ) 
  (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x => a^(x-2) + 4
  f 2 = 5 := by sorry

end fixed_point_of_exponential_function_l1460_146079


namespace unique_prime_p_l1460_146049

theorem unique_prime_p (p : ℕ) : 
  Prime p ∧ 
  Prime (8 * p^4 - 3003) ∧ 
  (8 * p^4 - 3003 > 0) ↔ 
  p = 5 := by sorry

end unique_prime_p_l1460_146049


namespace arithmetic_sequence_neither_necessary_nor_sufficient_l1460_146094

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_neither_necessary_nor_sufficient :
  ∃ (a : ℕ → ℝ) (m n p q : ℕ),
    arithmetic_sequence a ∧
    m > 0 ∧ n > 0 ∧ p > 0 ∧ q > 0 ∧
    (a m + a n > a p + a q ∧ m + n ≤ p + q) ∧
    (m + n > p + q ∧ a m + a n ≤ a p + a q) :=
sorry

end arithmetic_sequence_neither_necessary_nor_sufficient_l1460_146094


namespace triangle_inequality_variant_l1460_146021

theorem triangle_inequality_variant (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  Real.sqrt (x^2 + 3*y^2) + Real.sqrt (x^2 + z^2 + x*z) > Real.sqrt (z^2 + 3*y^2 + 3*y*z) := by
  sorry

end triangle_inequality_variant_l1460_146021


namespace log_product_equals_one_l1460_146068

theorem log_product_equals_one :
  Real.log 5 / Real.log 2 * Real.log 2 / Real.log 3 * Real.log 3 / Real.log 5 = 1 := by
  sorry

end log_product_equals_one_l1460_146068


namespace knowledge_competition_theorem_l1460_146012

/-- Represents a player in the knowledge competition --/
structure Player where
  correct_prob : ℚ
  deriving Repr

/-- Represents the game setup --/
structure Game where
  player_a : Player
  player_b : Player
  num_questions : ℕ
  deriving Repr

/-- Calculates the probability of a specific score for a player --/
def prob_score (game : Game) (player : Player) (score : ℕ) : ℚ :=
  sorry

/-- Calculates the mathematical expectation of a player's score --/
def expected_score (game : Game) (player : Player) : ℚ :=
  sorry

/-- The main theorem to prove --/
theorem knowledge_competition_theorem (game : Game) :
  game.player_a = Player.mk (2/3)
  → game.player_b = Player.mk (4/5)
  → game.num_questions = 2
  → prob_score game game.player_b 10 = 337/900
  ∧ expected_score game game.player_a = 23/3 :=
  sorry

end knowledge_competition_theorem_l1460_146012


namespace fraction_doubling_l1460_146070

theorem fraction_doubling (x y : ℝ) (h : x + y ≠ 0) :
  (2*x)^2 / (2*x + 2*y) = 2 * (x^2 / (x + y)) :=
sorry

end fraction_doubling_l1460_146070


namespace infinite_points_in_region_l1460_146014

theorem infinite_points_in_region :
  ∃ (S : Set (ℚ × ℚ)), 
    (∀ (p : ℚ × ℚ), p ∈ S → p.1 > 0 ∧ p.2 > 0) ∧ 
    (∀ (p : ℚ × ℚ), p ∈ S → p.1 + p.2 ≤ 7) ∧
    (∀ (p : ℚ × ℚ), p ∈ S → p.1 ≥ 1) ∧
    Set.Infinite S :=
by
  sorry

end infinite_points_in_region_l1460_146014


namespace polynomial_divisibility_and_factor_l1460_146055

theorem polynomial_divisibility_and_factor :
  let p (x : ℝ) := 6 * x^3 - 18 * x^2 + 24 * x - 24
  let q (x : ℝ) := x - 2
  let r (x : ℝ) := 6 * x^2 + 4
  (∃ (s : ℝ → ℝ), p = q * s) ∧ (∃ (t : ℝ → ℝ), p = r * t) := by
  sorry

end polynomial_divisibility_and_factor_l1460_146055


namespace largest_four_digit_number_l1460_146077

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def first_two_digits (n : ℕ) : ℕ := n / 100

def last_two_digits (n : ℕ) : ℕ := n % 100

theorem largest_four_digit_number (n : ℕ) 
  (h1 : is_four_digit n)
  (h2 : n % 10 ≠ 0)
  (h3 : 2014 % (first_two_digits n) = 0)
  (h4 : 2014 % ((first_two_digits n) * (last_two_digits n)) = 0) :
  n ≤ 5376 ∧ ∃ m : ℕ, m = 5376 ∧ 
    is_four_digit m ∧ 
    m % 10 ≠ 0 ∧ 
    2014 % (first_two_digits m) = 0 ∧ 
    2014 % ((first_two_digits m) * (last_two_digits m)) = 0 :=
sorry

end largest_four_digit_number_l1460_146077


namespace workers_read_all_three_l1460_146080

/-- Represents the number of workers who have read books by different authors -/
structure BookReaders where
  total : ℕ
  saramago : ℕ
  kureishi : ℕ
  atwood : ℕ
  saramagoKureishi : ℕ
  allThree : ℕ

/-- The theorem to prove -/
theorem workers_read_all_three (r : BookReaders) : r.allThree = 6 :=
  by
  have h1 : r.total = 75 := by sorry
  have h2 : r.saramago = r.total / 2 := by sorry
  have h3 : r.kureishi = r.total / 4 := by sorry
  have h4 : r.atwood = r.total / 5 := by sorry
  have h5 : r.total - (r.saramago + r.kureishi + r.atwood - (r.saramagoKureishi + r.allThree)) = 
            r.saramago - (r.saramagoKureishi + r.allThree) - 1 := by sorry
  have h6 : r.saramagoKureishi = 2 * r.allThree := by sorry
  sorry

#check workers_read_all_three

end workers_read_all_three_l1460_146080


namespace sum_ratio_equals_55_49_l1460_146047

theorem sum_ratio_equals_55_49 : 
  let sum_n (n : ℕ) := n * (n + 1) / 2
  let sum_squares (n : ℕ) := n * (n + 1) * (2 * n + 1) / 6
  let sum_cubes (n : ℕ) := (sum_n n) ^ 2
  (sum_n 10 * sum_cubes 10) / (sum_squares 10) ^ 2 = 55 / 49 := by
  sorry

end sum_ratio_equals_55_49_l1460_146047


namespace coats_collected_from_high_schools_l1460_146072

theorem coats_collected_from_high_schools 
  (total_coats : ℕ) 
  (elementary_coats : ℕ) 
  (h1 : total_coats = 9437)
  (h2 : elementary_coats = 2515) :
  total_coats - elementary_coats = 6922 := by
sorry

end coats_collected_from_high_schools_l1460_146072


namespace sports_league_games_l1460_146050

/-- Represents a sports league with the given conditions -/
structure SportsLeague where
  total_teams : Nat
  teams_per_division : Nat
  intra_division_games : Nat
  inter_division_games : Nat

/-- Calculate the total number of games in the sports league -/
def total_games (league : SportsLeague) : Nat :=
  let games_per_team := (league.teams_per_division - 1) * league.intra_division_games +
                        league.teams_per_division * league.inter_division_games
  (games_per_team * league.total_teams) / 2

/-- Theorem stating the total number of games in the given sports league configuration -/
theorem sports_league_games :
  let league := SportsLeague.mk 16 8 3 2
  total_games league = 296 := by
  sorry

end sports_league_games_l1460_146050


namespace tree_break_height_l1460_146037

theorem tree_break_height (tree_height road_width break_height : ℝ) 
  (h_tree : tree_height = 36)
  (h_road : road_width = 12)
  (h_pythagoras : (tree_height - break_height)^2 = break_height^2 + road_width^2) :
  break_height = 16 := by
sorry

end tree_break_height_l1460_146037


namespace equalize_buses_l1460_146029

def students_first_bus : ℕ := 57
def students_second_bus : ℕ := 31

def students_to_move : ℕ := 13

theorem equalize_buses :
  (students_first_bus - students_to_move = students_second_bus + students_to_move) ∧
  (students_first_bus - students_to_move > 0) ∧
  (students_second_bus + students_to_move > 0) :=
by sorry

end equalize_buses_l1460_146029


namespace arrangements_with_adjacent_pair_l1460_146007

-- Define the number of students
def total_students : ℕ := 5

-- Define the function to calculate permutations
def permutations (n : ℕ) (k : ℕ) : ℕ :=
  Nat.factorial n / Nat.factorial (n - k)

-- Define the theorem
theorem arrangements_with_adjacent_pair :
  permutations 4 4 * permutations 2 2 = 48 := by
  sorry

end arrangements_with_adjacent_pair_l1460_146007


namespace parallelogram_area_l1460_146054

theorem parallelogram_area (base height : ℝ) (h1 : base = 26) (h2 : height = 16) : 
  base * height = 416 := by
  sorry

end parallelogram_area_l1460_146054


namespace like_terms_exponent_relation_l1460_146025

theorem like_terms_exponent_relation (m n : ℕ) : 
  (∀ (x y : ℝ), ∃ (k : ℝ), 3 * x^(3*m) * y^2 = k * x^6 * y^n) → m^n = 4 := by
  sorry

end like_terms_exponent_relation_l1460_146025


namespace percentage_of_muslim_boys_l1460_146073

theorem percentage_of_muslim_boys (total_boys : ℕ) (hindu_percentage : ℚ) (sikh_percentage : ℚ) (other_boys : ℕ) :
  total_boys = 300 →
  hindu_percentage = 28 / 100 →
  sikh_percentage = 10 / 100 →
  other_boys = 54 →
  (total_boys - (hindu_percentage * total_boys + sikh_percentage * total_boys + other_boys)) / total_boys = 44 / 100 := by
  sorry

end percentage_of_muslim_boys_l1460_146073


namespace chord_length_l1460_146031

/-- The length of the chord intercepted by the circle x^2 + y^2 = 4 on the line x - √3y + 2√3 = 0 is 2. -/
theorem chord_length (x y : ℝ) : 
  (x^2 + y^2 = 4) → (x - Real.sqrt 3 * y + 2 * Real.sqrt 3 = 0) → 
  ∃ (a b c d : ℝ), (a^2 + b^2 = 4) ∧ (c^2 + d^2 = 4) ∧ 
  (a - Real.sqrt 3 * b + 2 * Real.sqrt 3 = 0) ∧ 
  (c - Real.sqrt 3 * d + 2 * Real.sqrt 3 = 0) ∧ 
  ((a - c)^2 + (b - d)^2 = 4) :=
sorry

end chord_length_l1460_146031

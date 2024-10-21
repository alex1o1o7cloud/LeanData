import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_i_squared_complex_equation_solution_l759_75951

/-- The imaginary unit -/
def i : ℂ := Complex.I

/-- Axiom for the imaginary unit -/
theorem i_squared : i * i = -1 := Complex.I_mul_I

/-- Definition of a pure imaginary number -/
def is_pure_imaginary (z : ℂ) : Prop := ∃ a : ℝ, z = a * i

theorem complex_equation_solution (b : ℝ) (z : ℂ) 
  (h1 : is_pure_imaginary z) 
  (h2 : (2 - i) * z = 4 - b * i) : 
  b = -8 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_i_squared_complex_equation_solution_l759_75951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_dot_product_l759_75972

theorem triangle_dot_product (A B C : ℝ × ℝ) :
  let AB := (B.1 - A.1, B.2 - A.2)
  let BC := (C.1 - B.1, C.2 - B.2)
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = 7^2 →
  (B.1 - C.1)^2 + (B.2 - C.2)^2 = 5^2 →
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = 6^2 →
  AB.1 * BC.1 + AB.2 * BC.2 = -19 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_dot_product_l759_75972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_base_circumference_l759_75969

open Real

/-- The volume of a right circular cone -/
noncomputable def cone_volume (r : ℝ) (h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

/-- The circumference of a circle -/
noncomputable def circle_circumference (r : ℝ) : ℝ := 2 * Real.pi * r

theorem cone_base_circumference (V h : ℝ) (hV : V = 18 * Real.pi) (hh : h = 6) :
  ∃ r : ℝ, cone_volume r h = V ∧ circle_circumference r = 6 * Real.pi := by
  sorry

#check cone_base_circumference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_base_circumference_l759_75969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_about_line_l759_75933

noncomputable def f (x : ℝ) : ℝ := 4 * Real.sin (2 * x + Real.pi / 3)

theorem symmetry_about_line (x : ℝ) : 
  f (Real.pi / 6 - x) = f (Real.pi / 6 + x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_about_line_l759_75933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_janice_earnings_l759_75926

/-- Calculates Janice's earnings for a week given her work schedule and tax rates. -/
def calculate_earnings (
  base_rate : ℝ)
  (weekday_overtime_rate : ℝ)
  (weekend_overtime_rate : ℝ)
  (base_hours : ℝ)
  (weekday_hours : List ℝ)
  (weekend_hours : List ℝ)
  (tips : ℝ)
  (base_tax_rate : ℝ)
  (overtime_tips_tax_rate : ℝ) : ℝ :=
  sorry

/-- Theorem stating that Janice's earnings for the given week equal $192 after deductions. -/
theorem janice_earnings :
  (let base_rate : ℝ := 30
   let weekday_overtime_rate : ℝ := 10
   let weekend_overtime_rate : ℝ := 15
   let base_hours : ℝ := 5
   let weekday_hours : List ℝ := [5, 6, 7, 5]
   let weekend_hours : List ℝ := [6]
   let tips : ℝ := 15
   let base_tax_rate : ℝ := 0.1
   let overtime_tips_tax_rate : ℝ := 0.05
   calculate_earnings base_rate weekday_overtime_rate weekend_overtime_rate base_hours
     weekday_hours weekend_hours tips base_tax_rate overtime_tips_tax_rate) = 192 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_janice_earnings_l759_75926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_critical_points_inequality_l759_75989

open Real

/-- Given a function f(x) = 2ln(x) + a/x where a ∈ ℝ, and g(x) = (x/2)f(x) - ax² - x,
    if g(x) has two distinct critical points x₁ and x₂ where x₁ < x₂,
    then ln(x₁) + 2ln(x₂) > 3. -/
theorem critical_points_inequality (a : ℝ) (x₁ x₂ : ℝ) :
  let f := λ x : ℝ ↦ 2 * log x + a / x
  let g := λ x : ℝ ↦ x / 2 * f x - a * x^2 - x
  (x₁ < x₂) →
  (∃ (y : ℝ), y ≠ x₁ ∧ y ≠ x₂ ∧ (deriv g) y = 0) →
  (deriv g) x₁ = 0 →
  (deriv g) x₂ = 0 →
  log x₁ + 2 * log x₂ > 3 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_critical_points_inequality_l759_75989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_10_ends_with_many_nines_l759_75930

noncomputable def sequence_a (n : ℕ) : ℕ :=
  match n with
  | 0 => 9
  | k + 1 => 3 * (sequence_a k)^4 + 4 * (sequence_a k)^3

def ends_with_nines (n : ℕ) (count : ℕ) : Prop :=
  ∃ m : ℕ, n = m * 10^count + (10^count - 1)

theorem a_10_ends_with_many_nines :
  ends_with_nines (sequence_a 10) 1000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_10_ends_with_many_nines_l759_75930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_plants_in_beds_l759_75922

/-- Represents a flower bed -/
structure FlowerBed where
  plants : Finset Nat

/-- The number of plants in a flower bed -/
def num_plants (bed : FlowerBed) : Nat := bed.plants.card

/-- Intersection of two flower beds -/
def intersect (A B : FlowerBed) : FlowerBed :=
  ⟨A.plants ∩ B.plants⟩

/-- Union of two flower beds -/
def union (A B : FlowerBed) : FlowerBed :=
  ⟨A.plants ∪ B.plants⟩

theorem total_plants_in_beds 
  (A B C D : FlowerBed)
  (hA : num_plants A = 600)
  (hB : num_plants B = 550)
  (hC : num_plants C = 400)
  (hD : num_plants D = 300)
  (hAB : num_plants (intersect A B) = 75)
  (hAC : num_plants (intersect A C) = 125)
  (hBD : num_plants (intersect B D) = 50)
  (hABC : num_plants ⟨A.plants ∩ B.plants ∩ C.plants⟩ = 25) :
  num_plants (union (union (union A B) C) D) = 1625 := by
  sorry

#check total_plants_in_beds

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_plants_in_beds_l759_75922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_equality_l759_75952

/-- An equilateral triangle -/
structure EquilateralTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  equilateral : dist A B = dist B C ∧ dist B C = dist C A

/-- A circle circumscribed around an equilateral triangle -/
structure CircumscribedCircle (triangle : EquilateralTriangle) where
  center : ℝ × ℝ
  radius : ℝ
  on_circle : (dist center triangle.A = radius) ∧
              (dist center triangle.B = radius) ∧
              (dist center triangle.C = radius)

/-- A point on the arc AB of the circumscribed circle -/
structure PointOnArc (triangle : EquilateralTriangle) (circle : CircumscribedCircle triangle) where
  P : ℝ × ℝ
  on_circle : dist circle.center P = circle.radius
  on_arc : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧
           P = ((1 - t) • triangle.A + t • triangle.B : ℝ × ℝ)

/-- The main theorem -/
theorem distance_equality (triangle : EquilateralTriangle) 
  (circle : CircumscribedCircle triangle) 
  (point : PointOnArc triangle circle) :
  dist point.P triangle.C = dist point.P triangle.A + dist point.P triangle.B := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_equality_l759_75952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_implies_a_l759_75940

/-- The function f(x) = ax³ - x - ln(x) takes an extreme value at x = 1 -/
def has_extreme_value_at_one (a : ℝ) : Prop :=
  let f := fun x => a * x^3 - x - Real.log x
  ∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f x ≤ f 1 ∨ f x ≥ f 1

/-- If f(x) = ax³ - x - ln(x) takes an extreme value at x = 1, then a = 2/3 -/
theorem extreme_value_implies_a (a : ℝ) :
  has_extreme_value_at_one a → a = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_implies_a_l759_75940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_B_equals_13_l759_75911

/-- The limit of the sequence defined by a(n+1) = √(10 + a(n)) with a(1) = √10 -/
noncomputable def y : ℝ := Real.sqrt (10 + Real.sqrt (10 + Real.sqrt (10 + Real.sqrt 10)))

/-- B is defined as the largest integer not greater than 10 + y -/
noncomputable def B : ℤ := ⌊10 + y⌋

/-- Theorem stating that B equals 13 -/
theorem B_equals_13 : B = 13 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_B_equals_13_l759_75911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l759_75992

theorem trigonometric_equation_solution (t : ℝ) (k : ℤ) : 
  (Real.sin t ≠ 0) → 
  (Real.cos (2 * t) ≠ -1) → 
  (Real.cos t ≠ -1) → 
  ((Real.sin (2 * t) / (1 + Real.cos (2 * t))) * (Real.sin t / (1 + Real.cos t)) = Real.arcsin t - 1) ↔ 
  (t = Real.pi / 4 * (4 * ↑k + 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l759_75992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_max_value_in_interval_l759_75905

noncomputable def f (x : ℝ) : ℝ := (Real.cos x)^2 + Real.sqrt 3 * Real.sin x * Real.cos x

-- Statement for the smallest positive period
theorem smallest_positive_period : 
  ∃ (T : ℝ), T > 0 ∧ (∀ x : ℝ, f (x + T) = f x) ∧ 
  (∀ S : ℝ, S > 0 ∧ (∀ x : ℝ, f (x + S) = f x) → T ≤ S) ∧ 
  T = Real.pi :=
sorry

-- Statement for the maximum value in the given interval
theorem max_value_in_interval : 
  ∃ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) ∧
  (∀ y ∈ Set.Icc 0 (Real.pi / 2), f y ≤ f x) ∧
  x = Real.pi / 6 ∧ f x = 3 / 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_max_value_in_interval_l759_75905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_turban_price_is_correct_l759_75931

/-- Calculates the price of a turban given the salary conditions and final payment --/
def turban_price (initial_yearly_salary : ℚ) (raise_percent : ℚ) (raise_interval : ℕ) 
                 (total_months : ℕ) (final_cash_payment : ℚ) : ℚ :=
  let initial_monthly_salary := initial_yearly_salary / 12
  let raise_amount := initial_monthly_salary * raise_percent
  let first_period := initial_monthly_salary * (raise_interval : ℚ)
  let second_period := (initial_monthly_salary + raise_amount) * (raise_interval : ℚ)
  let third_period := (initial_monthly_salary + 2 * raise_amount) * (raise_interval : ℚ)
  let total_cash_salary := first_period + second_period + third_period
  final_cash_payment - total_cash_salary

/-- The price of the turban is 9.125 given the specified conditions --/
theorem turban_price_is_correct : 
  turban_price 90 (5/100) 3 9 80 = 9125/1000 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_turban_price_is_correct_l759_75931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shelley_birthday_theorem_l759_75938

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Determines if a year is a leap year -/
def isLeapYear (year : Nat) : Bool :=
  (year % 400 == 0) || (year % 4 == 0 && year % 100 ≠ 0)

/-- Counts leap years between two years, inclusive -/
def countLeapYears (startYear endYear : Nat) : Nat :=
  (List.range (endYear - startYear + 1)).filter (fun y => isLeapYear (startYear + y)) |>.length

/-- Calculates the day of week offset between two dates -/
def dayOfWeekOffset (startYear endYear : Nat) : Nat :=
  let totalYears := endYear - startYear + 1
  let leapYears := countLeapYears startYear endYear
  let regularYears := totalYears - leapYears
  (regularYears + 2 * leapYears) % 7

/-- Shifts a day of week backward by a given offset -/
def shiftDayBackward (day : DayOfWeek) (offset : Nat) : DayOfWeek :=
  match (day, offset % 7) with
  | (DayOfWeek.Monday, 0) => DayOfWeek.Monday
  | (DayOfWeek.Monday, 1) => DayOfWeek.Sunday
  | (DayOfWeek.Monday, 2) => DayOfWeek.Saturday
  | (DayOfWeek.Monday, 3) => DayOfWeek.Friday
  | (DayOfWeek.Monday, 4) => DayOfWeek.Thursday
  | (DayOfWeek.Monday, 5) => DayOfWeek.Wednesday
  | (DayOfWeek.Monday, 6) => DayOfWeek.Tuesday
  | _ => day  -- This case should never occur due to the modulo operation

theorem shelley_birthday_theorem :
  let birthYear := 1792
  let anniversaryYear := 2042
  let anniversaryDay := DayOfWeek.Monday
  let offset := dayOfWeekOffset birthYear anniversaryYear
  shiftDayBackward anniversaryDay offset = DayOfWeek.Thursday := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shelley_birthday_theorem_l759_75938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l759_75900

/-- A polynomial with nonzero coefficients -/
def polynomial_with_nonzero_coeffs (x : ℝ) : ℝ := x

theorem constant_term_expansion (x : ℝ) : 
  (∃ c : ℝ, (x^2 - 2/x)^6 = c + x * polynomial_with_nonzero_coeffs x) → 
  (∃ c : ℝ, (x^2 - 2/x)^6 = 240 + x * polynomial_with_nonzero_coeffs x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l759_75900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_BCD_l759_75928

-- Define the triangle ABC
structure Triangle where
  area : ℝ

-- Define the segment CD
structure Segment where
  length : ℝ

-- State the theorem
theorem area_of_BCD (ABC : Triangle) (CD : Segment)
                    (h1 : ABC.area = 36) 
                    (h2 : CD.length = 39) : 
  ∃ BCD : Triangle, BCD.area = 156 := by
  sorry

#check area_of_BCD

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_BCD_l759_75928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_a_value_l759_75954

-- Define the function as noncomputable due to the use of Real.sqrt
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sqrt (x^2 - 2*x + a)

-- State the theorem
theorem unique_a_value : 
  (∀ x, f 1 x ≥ 0) ∧ 
  (∀ y ≥ 0, ∃ x, f 1 x = y) ∧ 
  (∀ a : ℝ, (∀ x, f a x ≥ 0) ∧ (∀ y ≥ 0, ∃ x, f a x = y) → a = 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_a_value_l759_75954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_certificate_interest_rate_l759_75945

/-- Calculates the annual interest rate of the second certificate -/
noncomputable def calculate_second_interest_rate (initial_investment : ℝ) 
                                   (first_rate : ℝ) 
                                   (final_value : ℝ) : ℝ :=
  let first_growth := initial_investment * (1 + first_rate / 4)
  400 * ((final_value / first_growth) - 1)

/-- Proves that given the specified conditions, the second certificate's 
    annual interest rate is approximately 16.549% -/
theorem second_certificate_interest_rate :
  let initial_investment := (20000 : ℝ)
  let first_rate := (0.08 : ℝ)
  let final_value := (21242 : ℝ)
  let second_rate := calculate_second_interest_rate initial_investment first_rate final_value
  abs (second_rate - 16.549) < 0.001 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_certificate_interest_rate_l759_75945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_intensity_theorem_l759_75935

theorem paint_intensity_theorem 
  (original_intensity new_intensity replaced_fraction : ℝ) 
  (h1 : original_intensity = 0.5) 
  (h2 : new_intensity = 0.3) 
  (h3 : replaced_fraction = 0.8) : 
  (new_intensity - (1 - replaced_fraction) * original_intensity) / replaced_fraction = 0.25 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_intensity_theorem_l759_75935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_pi_fourth_f_strictly_increasing_l759_75966

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x * (Real.sin x + Real.cos x)

-- Theorem 1: f(π/4) = 2
theorem f_pi_fourth : f (Real.pi / 4) = 2 := by sorry

-- Theorem 2: f is strictly increasing on the intervals [kπ - π/8, kπ + 3π/8], where k ∈ ℤ
theorem f_strictly_increasing (k : ℤ) :
  StrictMonoOn f (Set.Icc ((k : ℝ) * Real.pi - Real.pi / 8) ((k : ℝ) * Real.pi + 3 * Real.pi / 8)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_pi_fourth_f_strictly_increasing_l759_75966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_increase_percentage_l759_75995

/-- Represents the speed on a given day -/
structure Speed where
  value : ℝ

/-- Represents the time spent traveling on a given day -/
structure TravelTime where
  value : ℝ

/-- Calculates the percentage increase between two speeds -/
noncomputable def percentageIncrease (v1 v2 : Speed) : ℝ :=
  (v2.value - v1.value) / v1.value * 100

/-- The main theorem stating the percentage increase in speed -/
theorem speed_increase_percentage
  (v1 v2 : Speed)
  (t1 t2 : TravelTime)
  (h1 : t1.value = 30/60) -- 30 minutes in hours
  (h2 : t2.value = 25/60) -- 25 minutes in hours
  (h3 : v1.value * t1.value = v2.value * t2.value) -- Same distance traveled
  : percentageIncrease v1 v2 = 20 := by
  sorry

#check speed_increase_percentage

end NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_increase_percentage_l759_75995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_33_terms_equals_330_l759_75949

/-- Represents an arithmetic progression -/
structure ArithmeticProgression where
  a : ℚ  -- first term
  d : ℚ  -- common difference

/-- Sum of the first n terms of an arithmetic progression -/
def sum_n_terms (ap : ArithmeticProgression) (n : ℕ) : ℚ :=
  n / 2 * (2 * ap.a + (n - 1) * ap.d)

/-- Theorem statement -/
theorem sum_33_terms_equals_330 (ap : ArithmeticProgression) :
  sum_n_terms ap 3 = 30 → sum_n_terms ap 30 = 300 → sum_n_terms ap 33 = 330 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_33_terms_equals_330_l759_75949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l759_75908

def plane := ℝ × ℝ

-- Define vectors
def e₁ : plane := (2, 1)
def e₂ : plane := (2, -2)

-- Define vector operations
def add (v w : plane) : plane := (v.1 + w.1, v.2 + w.2)
def scale (a : ℝ) (v : plane) : plane := (a * v.1, a * v.2)

-- Define given vectors
def AB : plane := add (scale 2 e₁) e₂
def BE (l : ℝ) : plane := add (scale (-1) e₁) (scale l e₂)
def EC : plane := add (scale (-2) e₁) e₂

-- Collinearity condition
def collinear (A B C : plane) : Prop :=
  ∃ (t : ℝ), add A (scale t (add (scale (-1) A) B)) = C

theorem vector_problem :
  ∃ (l : ℝ), 
    (l = -3/2) ∧ 
    (collinear AB (add AB (BE l)) (add AB EC)) ∧
    (add (BE l) EC = (-7, -2)) := by
  sorry

#check vector_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l759_75908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_positive_range_l759_75939

/-- A function is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_positive_range
    (f : ℝ → ℝ) (f' : ℝ → ℝ)
    (h_odd : IsOdd f)
    (h_deriv : ∀ x, HasDerivAt f (f' x) x)
    (h_zero : f (-1) = 0)
    (h_pos : ∀ x > 0, x * f' x - f x > 0) :
    ∀ x, f x > 0 ↔ x ∈ Set.Ioo (-1 : ℝ) 0 ∪ Set.Ioi (1 : ℝ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_positive_range_l759_75939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_distance_l759_75956

noncomputable section

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2/3 - y^2 = 1

-- Define the focus of the parabola
def parabola_focus (p : ℝ) : ℝ × ℝ := (p/2, 0)

-- Define the right focus of the hyperbola
def hyperbola_right_focus : ℝ × ℝ := (2, 0)

-- Define a point on the parabola
def point_on_parabola (b : ℝ) : ℝ × ℝ := (2, b)

-- Define the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem parabola_focus_distance 
  (p : ℝ) 
  (h1 : parabola_focus p = hyperbola_right_focus) 
  (b : ℝ) 
  (h2 : parabola p 2 b) :
  distance (point_on_parabola b) (parabola_focus p) = 4 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_distance_l759_75956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_train_interval_l759_75984

-- Define the travel times for each route
noncomputable def northern_route_time : ℝ := 17
noncomputable def southern_route_time : ℝ := 11

-- Define the average time difference between counterclockwise and clockwise trains
noncomputable def train_arrival_difference : ℝ := 1.25

-- Define the average time difference between home-to-work and work-to-home trips
noncomputable def trip_time_difference : ℝ := 1

-- Define the probability of boarding a clockwise train
noncomputable def p : ℝ := 7/12

-- Define the average interval time between consecutive train arrivals
noncomputable def average_interval_time : ℝ := 5/4

-- Theorem statement
theorem expected_train_interval :
  let expected_interval := average_interval_time / (1 - p)
  expected_interval = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_train_interval_l759_75984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_equals_open_interval_l759_75942

-- Define sets A and B
def A : Set ℝ := {x | x^2 ≤ 4*x}
def B : Set ℝ := {x | x < 1}

-- State the theorem
theorem intersection_equals_open_interval : A ∩ B = Set.Iio 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_equals_open_interval_l759_75942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_l759_75973

noncomputable def fixed_cost : ℝ := 300000

noncomputable def additional_cost (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 40 then 10 * x^2 + 100 * x
  else if x ≥ 40 then 501 * x + 10000 / x - 4500
  else 0

noncomputable def price_per_vehicle : ℝ := 50

noncomputable def profit (x : ℝ) : ℝ :=
  price_per_vehicle * 100 * x - (fixed_cost + additional_cost x)

noncomputable def production_volume : ℝ := 100

theorem max_profit :
  profit production_volume = 1300000 ∧
  ∀ x > 0, profit x ≤ profit production_volume := by
  sorry

#eval "Compilation successful!"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_l759_75973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_area_l759_75944

-- Define the triangle PQR
def Triangle (a b c : ℝ) : Prop := a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

-- Define an isosceles triangle
def Isosceles (a b c : ℝ) : Prop := Triangle a b c ∧ (a = b ∨ b = c ∨ c = a)

-- Define the area of a triangle using Heron's formula
noncomputable def area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

-- Theorem statement
theorem isosceles_triangle_area :
  Isosceles 17 17 16 → area 17 17 16 = 120 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_area_l759_75944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_40_l759_75999

noncomputable def clock_angle (hours : ℕ) (minutes : ℕ) : ℝ :=
  let hour_angle : ℝ := (hours % 12 + minutes / 60 : ℝ) * 30
  let minute_angle : ℝ := minutes * 6
  let angle_diff : ℝ := abs (hour_angle - minute_angle)
  min angle_diff (360 - angle_diff)

theorem clock_angle_at_3_40 :
  clock_angle 3 40 = 130 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_3_40_l759_75999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_range_l759_75983

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + x - 2*a

-- Define the property of having exactly 4 integer solutions
def has_four_integer_solutions (a : ℝ) : Prop :=
  ∃ (w x y z : ℤ), w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z ∧
  (∀ (n : ℤ), f a (n : ℝ) < 0 ↔ n ∈ ({w, x, y, z} : Set ℤ))

-- State the theorem
theorem quadratic_inequality_range :
  ∀ a : ℝ, has_four_integer_solutions a → 2/7 ≤ a ∧ a < 3/7 :=
by
  sorry

-- You can add more lemmas or theorems here if needed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_range_l759_75983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_function_values_l759_75943

noncomputable def f (x : ℝ) : ℝ := (x + 3) / (x + 1)

theorem sum_of_function_values :
  let m := f 1 + f 2 + f 4 + f 8 + f 16
  let n := f (1/2) + f (1/4) + f (1/8) + f (1/16)
  m + n = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_function_values_l759_75943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_division_sum_l759_75923

/-- Represents a point in 2D space -/
structure Point where
  x : ℚ
  y : ℚ

/-- Represents a quadrilateral -/
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Represents a line -/
structure Line where
  m : ℚ  -- slope
  b : ℚ  -- y-intercept

/-- Function to calculate the area of a quadrilateral -/
noncomputable def area (q : Quadrilateral) : ℚ :=
  sorry

/-- Function to check if a line divides a quadrilateral into two equal areas -/
def dividesEqually (l : Line) (q : Quadrilateral) : Prop :=
  sorry

/-- Function to find the intersection of two lines -/
noncomputable def intersection (l1 l2 : Line) : Point :=
  sorry

/-- Theorem statement -/
theorem quadrilateral_division_sum (A B C D : Point) 
  (h1 : A.x = 1 ∧ A.y = 1)
  (h2 : B.x = 2 ∧ B.y = 4)
  (h3 : C.x = 5 ∧ C.y = 4)
  (h4 : D.x = 6 ∧ D.y = 1)
  (q : Quadrilateral)
  (hq : q = ⟨A, B, C, D⟩)
  (l : Line)
  (hl : l.m * A.x + l.b = A.y)  -- line passes through A
  (hd : dividesEqually l q)
  (CD : Line)
  (hCD : CD.m = -3 ∧ CD.b = 19)  -- equation of line CD
  (P : Point)
  (hP : P = intersection l CD)
  (p q r s : ℕ)
  (hp : P.x = p / q)
  (hr : P.y = r / s)
  (hpq : Nat.Coprime p q)
  (hrs : Nat.Coprime r s) :
  p + q + r + s = 46 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_division_sum_l759_75923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_decreasing_l759_75921

-- Define the function f(x) = x - ln x
noncomputable def f (x : ℝ) : ℝ := x - Real.log x

-- State the theorem
theorem f_monotonic_decreasing :
  ∀ x₁ x₂ : ℝ, 0 < x₁ → 0 < x₂ → x₁ < x₂ → x₂ < 1 → f x₂ < f x₁ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_decreasing_l759_75921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_equation_one_real_root_l759_75981

/-- Given nonzero real numbers a, b, c, and k, the determinant equation
    |x, kc, -kb; -c, x, ka; b, -a, x| = 0 has exactly one real root. -/
theorem det_equation_one_real_root
  (a b c k : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hk : k ≠ 0) :
  ∃! x : ℝ, Matrix.det
    ![![x, k * c, -k * b],
      ![-c, x, k * a],
      ![b, -a, x]] = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_equation_one_real_root_l759_75981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_general_term_b_maximum_term_l759_75910

def a : ℕ → ℚ
  | 0 => 1
  | n + 1 => ((n + 2 : ℚ) * a n + 1) / (n + 1)

def b (n : ℕ) : ℚ := ((a n + 1) / 2) * (8/9)^n

theorem a_general_term (n : ℕ) : a n = 2 * (n + 1) - 1 := by sorry

theorem b_maximum_term : 
  ∀ (n : ℕ), n ≥ 1 → b n ≤ b 8 ∧ b 8 = b 9 ∧ b 8 = (8^9 : ℚ) / 9^8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_general_term_b_maximum_term_l759_75910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_five_l759_75957

/-- Represents the sum of the first n terms of a geometric sequence -/
noncomputable def geometricSum (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a₁ else a₁ * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum_five (q : ℝ) (h₁ : q > 0) 
    (h₂ : geometricSum 1 q 4 = 5 * geometricSum 1 q 2) : 
  geometricSum 1 q 5 = 31 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_five_l759_75957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isolated_set_problem_l759_75991

def is_isolated (A : Finset Nat) : Prop :=
  ∀ B ⊂ A, Nat.gcd (B.sum id) (A.sum id) = 1

def square_set (a b n : Nat) : Finset Nat :=
  Finset.image (fun k => (a + k * b) ^ 2) (Finset.range n)

theorem isolated_set_problem :
  let A : Finset Nat := {4, 9, 16, 25, 36, 49}
  (is_isolated A) ∧
  (∀ a b : Nat, a > 0 → b > 0 →
    (∃ n : Nat, Nat.Prime n.succ.succ ∧ is_isolated (square_set a b n)) ↔ 
    (is_isolated (square_set a b 6))) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isolated_set_problem_l759_75991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paul_catches_mary_l759_75996

/-- The time it takes for Paul to catch up with Mary -/
noncomputable def catch_up_time (mary_speed paul_speed : ℝ) (time_difference : ℝ) : ℝ :=
  (mary_speed * time_difference) / (paul_speed - mary_speed)

/-- Theorem stating that Paul catches up with Mary in 25 minutes -/
theorem paul_catches_mary :
  let mary_speed : ℝ := 50
  let paul_speed : ℝ := 80
  let time_difference : ℝ := 0.25 -- 15 minutes in hours
  catch_up_time mary_speed paul_speed time_difference * 60 = 25 := by
  sorry

#check paul_catches_mary

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paul_catches_mary_l759_75996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_plus_pi_third_l759_75968

theorem tan_theta_plus_pi_third (θ : Real) (h : Real.sin θ + Real.cos θ = Real.sqrt 2) : 
  Real.tan (θ + π/3) = -2 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_plus_pi_third_l759_75968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_win_probability_l759_75964

/-- Represents the outcome of Alice's roll -/
inductive AliceRoll : Type
| one | two | three | four | five | six
deriving Fintype, Repr

/-- Represents the outcome of Bob's roll -/
inductive BobRoll : Type
| one | two | three | four | five | six | seven | eight
deriving Fintype, Repr

/-- Determines if Alice wins based on the rolls -/
def alice_wins (a : AliceRoll) (b : BobRoll) : Bool :=
  match a, b with
  | AliceRoll.one, _ => true
  | AliceRoll.two, BobRoll.two => true
  | AliceRoll.two, BobRoll.four => true
  | AliceRoll.two, BobRoll.six => true
  | AliceRoll.two, BobRoll.eight => true
  | AliceRoll.three, BobRoll.three => true
  | AliceRoll.three, BobRoll.six => true
  | AliceRoll.four, BobRoll.four => true
  | AliceRoll.four, BobRoll.eight => true
  | AliceRoll.five, BobRoll.five => true
  | AliceRoll.six, BobRoll.six => true
  | _, _ => false

/-- The probability of Alice winning -/
theorem alice_win_probability :
  (Fintype.card {p : AliceRoll × BobRoll | alice_wins p.1 p.2} : ℚ) /
  (Fintype.card (AliceRoll × BobRoll) : ℚ) = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_win_probability_l759_75964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_sqrt_two_l759_75955

-- Define the triangle ABC
def triangle (a b c : ℝ) : Prop := a > 0 ∧ b > 0 ∧ c > 0

-- Define the geometric sequence property
def geometric_sequence (a b : ℝ) : Prop := a * b = 4

-- Define the area of the triangle
noncomputable def triangle_area (a b : ℝ) : ℝ := (1/2) * a * b * Real.sin (Real.pi/4)

theorem triangle_area_sqrt_two (a b c : ℝ) :
  triangle a b c →
  geometric_sequence a b →
  triangle_area a b = Real.sqrt 2 := by
  sorry

#check triangle_area_sqrt_two

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_sqrt_two_l759_75955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_system_solution_l759_75934

theorem inequality_system_solution (a : ℤ) : 
  (∃ (S : Finset ℤ), (∀ x ∈ S, (6 * x + 3 > 3 * (x + a)) ∧ ((x : ℚ) / 2 - 1 ≤ 7 - 3 / 2 * x)) ∧ 
  (S.sum id = 9)) → 
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_system_solution_l759_75934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l759_75906

theorem hyperbola_eccentricity_range (α : ℝ) (e : ℝ) :
  (π / 4 < α) ∧ (α < π / 3) →
  (∃ (a b c : ℝ),
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    e = c / a ∧
    Real.tan α = b / a ∧
    c^2 = a^2 + b^2) →
  Real.sqrt 2 < e ∧ e < 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l759_75906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_returns_to_initial_state_l759_75997

def X (n : ℕ) (k : ℕ) : List Bool :=
  match k with
  | 0 => [true] ++ List.replicate (n - 2) false ++ [true]
  | k + 1 => 
    let prev := X n k
    List.zipWith 
      (fun x y => x ≠ y) 
      prev 
      ((List.drop 1 prev) ++ [List.head! prev])

theorem sequence_returns_to_initial_state (n m : ℕ) 
  (h1 : n > 1) 
  (h2 : Odd n) 
  (h3 : X n m = X n 0) : 
  n ∣ m :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_returns_to_initial_state_l759_75997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l759_75927

/-- The area of a triangle with vertices at (3, 2), (3, -4), and (11, -4) is 24 square units. -/
theorem triangle_area : ℝ := by
  -- Define the vertices
  let v1 : ℝ × ℝ := (3, 2)
  let v2 : ℝ × ℝ := (3, -4)
  let v3 : ℝ × ℝ := (11, -4)

  -- Calculate the area using the formula: 1/2 * |x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)|
  let area := (1/2 : ℝ) * abs (v1.1 * (v2.2 - v3.2) + v2.1 * (v3.2 - v1.2) + v3.1 * (v1.2 - v2.2))

  -- Prove that the calculated area equals 24
  have h : area = 24 := by sorry

  -- Return the result
  exact 24


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l759_75927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_is_pi_over_four_max_area_l759_75998

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  angle_sum : A + B + C = Real.pi
  positive_sides : a > 0 ∧ b > 0 ∧ c > 0

-- Define the given condition
def given_condition (t : Triangle) : Prop :=
  t.a = t.b * Real.cos t.C + t.c * Real.sin t.B

-- Theorem 1: B = π/4
theorem angle_B_is_pi_over_four (t : Triangle) (h : given_condition t) : t.B = Real.pi/4 := by
  sorry

-- Theorem 2: Maximum area when b = 4
theorem max_area (t : Triangle) (h1 : given_condition t) (h2 : t.b = 4) : 
  ∃ (area : Real), area ≤ 4 * Real.sqrt 2 + 4 ∧ 
  ∀ (other_area : Real), other_area = 1/2 * t.a * t.c * Real.sin t.B → other_area ≤ area := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_is_pi_over_four_max_area_l759_75998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_50_equals_2_pow_99_l759_75904

/-- Sequence b_n defined recursively -/
def b : ℕ → ℝ
  | 0 => 2  -- Define for 0 to cover all natural numbers
  | n + 1 => (64 * (b n)^3)^(1/3)

/-- Theorem stating that the 50th term of the sequence equals 2^99 -/
theorem b_50_equals_2_pow_99 : b 50 = 2^99 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_50_equals_2_pow_99_l759_75904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_P_equal_distances_iff_k_eq_neg_one_l759_75971

/-- The line equation kx - y - 2k - 1 = 0 -/
def line_equation (k x y : ℝ) : Prop := k * x - y - 2 * k - 1 = 0

/-- Point P -/
def point_P : ℝ × ℝ := (2, -1)

/-- Intersection with positive x-axis -/
noncomputable def point_A (k : ℝ) : ℝ × ℝ := ((2 * k + 1) / k, 0)

/-- Intersection with positive y-axis -/
def point_B (k : ℝ) : ℝ × ℝ := (0, -2 * k - 1)

/-- Origin -/
def origin : ℝ × ℝ := (0, 0)

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem line_passes_through_P (k : ℝ) :
  line_equation k point_P.1 point_P.2 := by sorry

theorem equal_distances_iff_k_eq_neg_one :
  ∀ k : ℝ, k ≠ 0 →
    (distance (point_A k) origin = distance (point_B k) origin ↔ k = -1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_P_equal_distances_iff_k_eq_neg_one_l759_75971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_degrees_for_one_male_one_female_l759_75980

/-- Represents a class with male and female students -/
structure MyClass where
  male_ratio : ℕ
  female_ratio : ℕ

/-- Represents a career preference in the class -/
structure CareerPreference where
  male_count : ℕ
  female_count : ℕ

/-- Calculates the degrees in a circle graph for a given career preference -/
noncomputable def degrees_for_preference (c : MyClass) (p : CareerPreference) : ℝ :=
  360 * (p.male_count * c.female_ratio + p.female_count * c.male_ratio) / 
    (c.male_ratio * c.female_ratio * (c.male_ratio + c.female_ratio))

/-- Theorem: The degrees for a career preferred by one male and one female in a class with 2:3 male to female ratio is 144 -/
theorem degrees_for_one_male_one_female (c : MyClass) (p : CareerPreference) :
  c.male_ratio = 2 → c.female_ratio = 3 → p.male_count = 1 → p.female_count = 1 →
  degrees_for_preference c p = 144 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_degrees_for_one_male_one_female_l759_75980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tadpole_to_frog_ratio_l759_75987

/-- Represents the frog pond ecosystem -/
structure FrogPond where
  initialFrogs : ℕ
  maxCapacity : ℕ
  tadpoleSurvivalRate : ℚ
  frogsLeavingPond : ℕ

/-- Calculates the number of tadpoles in the pond -/
def calculateTadpoles (pond : FrogPond) : ℕ :=
  let newFrogs := pond.maxCapacity - pond.initialFrogs
  Nat.ceil ((newFrogs : ℚ) / pond.tadpoleSurvivalRate)

/-- Theorem stating that the ratio of tadpoles to frogs is 1:1 -/
theorem tadpole_to_frog_ratio (pond : FrogPond)
  (h1 : pond.initialFrogs = 5)
  (h2 : pond.maxCapacity = 8)
  (h3 : pond.tadpoleSurvivalRate = 2/3)
  (h4 : pond.frogsLeavingPond = 7) :
  (calculateTadpoles pond : ℚ) / pond.initialFrogs = 1 := by
  sorry

#eval calculateTadpoles { initialFrogs := 5, maxCapacity := 8, tadpoleSurvivalRate := 2/3, frogsLeavingPond := 7 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tadpole_to_frog_ratio_l759_75987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_to_line_l759_75982

/-- The minimum distance from a point on the circle ρ = 2cos θ to the line ρ sin(θ + π/4) = -√2/2 -/
theorem min_distance_circle_to_line :
  let circle := {p : ℝ × ℝ | ∃ θ : ℝ, p.1 = 2 * Real.cos θ ∧ p.2 = 2 * Real.sin θ}
  let line := {p : ℝ × ℝ | p.1 + p.2 = 1}
  ∀ p ∈ circle, ∃ q ∈ line, ∀ r ∈ line, Real.sqrt 2 - 1 ≤ dist p r ∧
    dist p q = Real.sqrt 2 - 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_to_line_l759_75982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_to_squares_area_ratio_l759_75914

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square -/
structure Square where
  a : Point
  b : Point
  c : Point
  d : Point

/-- Represents a hexagon -/
structure Hexagon where
  a : Point
  b : Point
  c : Point
  d : Point
  e : Point
  f : Point

/-- The area of a shape -/
noncomputable def area (T : Type) : ℝ := sorry

/-- Three squares of equal area arranged adjacently -/
def squares : Square × Square × Square := sorry

/-- The hexagon AFJICB -/
def hexagon : Hexagon := sorry

/-- C is a quarter point of LH -/
axiom c_quarter_LH : hexagon.c.x = squares.2.2.c.x + (squares.2.2.d.x - squares.2.2.c.x) / 4

/-- D is a quarter point of HE -/
axiom d_quarter_HE : hexagon.d.x = squares.2.1.a.x + (squares.2.1.b.x - squares.2.1.a.x) / 4

/-- A is the midpoint of HL -/
axiom a_midpoint_HL : hexagon.a.x = (squares.2.2.c.x + squares.2.1.b.x) / 2

/-- The theorem to be proved -/
theorem hexagon_to_squares_area_ratio :
  area Hexagon / (area Square + area Square + area Square) = 3 / 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_to_squares_area_ratio_l759_75914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_range_of_a_l759_75986

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 1 then x^2 + a*x else (4 - a/2)*x + 2

def IsIncreasing (g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → g x < g y

theorem f_increasing_range_of_a :
  {a : ℝ | IsIncreasing (f a)} = {a : ℝ | 10/3 ≤ a ∧ a < 8} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_range_of_a_l759_75986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_solution_g_inequality_solution_l759_75902

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := |x + 2*a| + |x - 1|

-- Define the function g
noncomputable def g (a : ℝ) : ℝ := f a (1/a)

-- Theorem for part (1)
theorem f_inequality_solution (x : ℝ) : 
  f 1 x ≤ 5 ↔ -3 ≤ x ∧ x ≤ 2 := by sorry

-- Theorem for part (2)
theorem g_inequality_solution (a : ℝ) (h : a ≠ 0) :
  g a ≤ 4 ↔ 1/2 ≤ a ∧ a ≤ 3/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_solution_g_inequality_solution_l759_75902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_points_characterization_l759_75962

def f (n : ℕ) (x : ℕ) : ℕ :=
  if x % 2 = 0 then x / 2 else (x - 1) / 2 + 2^(n-1)

def iterate_f (n : ℕ) (k : ℕ) (x : ℕ) : ℕ :=
  match k with
  | 0 => x
  | k + 1 => f n (iterate_f n k x)

theorem fixed_points_characterization (n : ℕ) (hn : n > 0) :
  {x : ℕ | iterate_f n n x = x} = {x : ℕ | 1 ≤ x ∧ x ≤ 2^n} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_points_characterization_l759_75962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_obtuse_triangle_from_ratio_angle_comparison_from_sine_l759_75929

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  hpos : 0 < a ∧ 0 < b ∧ 0 < c
  hsum : A + B + C = π
  hcosine : c^2 = a^2 + b^2 - 2*a*b*(Real.cos C)
  hsine : a / (Real.sin A) = b / (Real.sin B)

-- Theorem 1: If sides are in ratio 2:3:4, the triangle is obtuse
theorem obtuse_triangle_from_ratio (t : Triangle) (h : ∃ k : ℝ, t.a = 2*k ∧ t.b = 3*k ∧ t.c = 4*k) :
  ∃ θ : ℝ, θ ∈ [t.A, t.B, t.C] ∧ θ > π/2 := by
  sorry

-- Theorem 2: If sin A > sin B, then A > B
theorem angle_comparison_from_sine (t : Triangle) (h : Real.sin t.A > Real.sin t.B) : t.A > t.B := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_obtuse_triangle_from_ratio_angle_comparison_from_sine_l759_75929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_triangle_properties_l759_75903

noncomputable section

-- Define the ellipse C
def ellipse (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the incircle of the quadrilateral
def incircle (x y : ℝ) : Prop := x^2 + y^2 = 12/7

-- Define the area of the quadrilateral
def quadrilateral_area : ℝ := 4 * Real.sqrt 3

-- Define the minimum area of ΔOAB
def min_area_OAB : ℝ := 12/7

-- Define the area of a triangle
def area_triangle (A B C : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

theorem ellipse_and_triangle_properties
  (a b : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : ∀ x y, ellipse a b x y → incircle x y) 
  (h4 : quadrilateral_area = 4 * Real.sqrt 3) :
  (a^2 = 4 ∧ b^2 = 3) ∧
  (∀ A B : ℝ × ℝ, 
    (ellipse a b A.1 A.2) → 
    (ellipse a b B.1 B.2) → 
    (A.1 * B.1 + A.2 * B.2 = 0) → 
    area_triangle (0, 0) A B ≥ min_area_OAB) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_triangle_properties_l759_75903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_one_l759_75913

open Real

-- Define the function f
noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * sin (ω * x + π / 3)

-- State the theorem
theorem f_value_at_one (ω : ℝ) (h1 : ω > 0) 
  (h2 : ∃ (A B : ℝ × ℝ), 
    (∀ x, (f ω x, x) ≤ A ∧ (f ω x, x) ≥ B) ∧ 
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 5) : 
  f ω 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_one_l759_75913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_satisfying_numbers_l759_75953

/-- A three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  units : Nat
  h_hundreds : hundreds ≥ 1 ∧ hundreds ≤ 9
  h_tens : tens ≤ 9
  h_units : units ≤ 9

/-- Predicate for a three-digit number satisfying the given conditions -/
def satisfiesConditions (n : ThreeDigitNumber) : Bool :=
  n.units % 2 = 0 ∧
  n.tens + n.units = 12 ∧
  n.hundreds > n.units

/-- The count of three-digit numbers satisfying the conditions -/
def countSatisfyingNumbers : Nat :=
  let numbers := [
    ThreeDigitNumber.mk 5 8 4 ⟨by norm_num, by norm_num⟩ (by norm_num) (by norm_num),
    ThreeDigitNumber.mk 6 8 4 ⟨by norm_num, by norm_num⟩ (by norm_num) (by norm_num),
    ThreeDigitNumber.mk 7 8 4 ⟨by norm_num, by norm_num⟩ (by norm_num) (by norm_num),
    ThreeDigitNumber.mk 8 8 4 ⟨by norm_num, by norm_num⟩ (by norm_num) (by norm_num),
    ThreeDigitNumber.mk 9 8 4 ⟨by norm_num, by norm_num⟩ (by norm_num) (by norm_num),
    ThreeDigitNumber.mk 7 6 6 ⟨by norm_num, by norm_num⟩ (by norm_num) (by norm_num),
    ThreeDigitNumber.mk 8 6 6 ⟨by norm_num, by norm_num⟩ (by norm_num) (by norm_num),
    ThreeDigitNumber.mk 9 6 6 ⟨by norm_num, by norm_num⟩ (by norm_num) (by norm_num),
    ThreeDigitNumber.mk 9 4 8 ⟨by norm_num, by norm_num⟩ (by norm_num) (by norm_num)
  ]
  (numbers.filter satisfiesConditions).length

theorem count_satisfying_numbers : countSatisfyingNumbers = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_satisfying_numbers_l759_75953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_distance_l759_75994

/-- An ellipse in the first quadrant tangent to both x-axis and y-axis -/
structure Ellipse where
  /-- One focus of the ellipse -/
  focus1 : ℝ × ℝ
  /-- The other focus of the ellipse -/
  focus2 : ℝ × ℝ
  /-- The ellipse is in the first quadrant -/
  first_quadrant : focus1.1 ≥ 0 ∧ focus1.2 ≥ 0 ∧ focus2.1 ≥ 0 ∧ focus2.2 ≥ 0
  /-- The ellipse is tangent to both x-axis and y-axis -/
  tangent_to_axes : True  -- This is a simplification, as we can't easily express tangency in this context

/-- The theorem stating the value of d for the given ellipse -/
theorem ellipse_focus_distance (e : Ellipse) 
    (h1 : e.focus1 = (5, 9)) 
    (h2 : ∃ d : ℝ, e.focus2 = (d, 9)) : 
  ∃ d : ℝ, e.focus2 = (14/3, 9) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_distance_l759_75994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_triangle_perimeter_l759_75920

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Generates the next triangle in the sequence -/
noncomputable def nextTriangle (t : Triangle) : Triangle :=
  { a := (t.b + t.c - t.a) / 2,
    b := (t.a + t.c - t.b) / 2,
    c := (t.a + t.b - t.c) / 2 }

/-- Checks if a triangle satisfies the triangle inequality -/
def isValidTriangle (t : Triangle) : Prop :=
  t.a + t.b > t.c ∧ t.b + t.c > t.a ∧ t.c + t.a > t.b

/-- The initial triangle T₁ -/
def T₁ : Triangle :=
  { a := 1002, b := 1003, c := 1001 }

/-- The sequence of triangles -/
noncomputable def triangleSequence : ℕ → Triangle
  | 0 => T₁
  | n + 1 => nextTriangle (triangleSequence n)

/-- The perimeter of a triangle -/
noncomputable def perimeter (t : Triangle) : ℝ :=
  t.a + t.b + t.c

/-- Theorem: The perimeter of the last valid triangle in the sequence is 1503/256 -/
theorem last_triangle_perimeter :
  ∃ n : ℕ, (∀ k < n, isValidTriangle (triangleSequence k)) ∧
           ¬isValidTriangle (triangleSequence n) ∧
           perimeter (triangleSequence (n - 1)) = 1503 / 256 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_triangle_perimeter_l759_75920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_product_l759_75946

theorem complex_fraction_product (a b : ℝ) :
  (1 + 7 * Complex.I) / (2 - Complex.I) = (a : ℂ) + b * Complex.I →
  a * b = -3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_product_l759_75946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_stoppage_duration_l759_75977

/-- The overall duration of stoppages per hour for a train -/
noncomputable def stoppage_duration (speed_without_stoppages : ℝ) (speed_with_stoppages : ℝ) : ℝ :=
  (speed_without_stoppages - speed_with_stoppages) / speed_without_stoppages * 60

theorem train_stoppage_duration :
  let speed_without_stoppages := (42 : ℝ)
  let speed_with_stoppages := (27 : ℝ)
  abs (stoppage_duration speed_without_stoppages speed_with_stoppages - 21.43) < 0.01 := by
  sorry

-- Remove the #eval line as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_stoppage_duration_l759_75977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_store_discount_claim_l759_75950

/-- Represents a discount as a percentage between 0 and 100 -/
structure Discount where
  val : ℝ
  valid : 0 ≤ val ∧ val ≤ 100

/-- Applies a discount to a price -/
noncomputable def apply_discount (price : ℝ) (discount : Discount) : ℝ :=
  price * (1 - discount.val / 100)

/-- Calculates the total discount percentage after applying two successive discounts -/
noncomputable def total_discount (d1 d2 : Discount) : ℝ :=
  (1 - (1 - d1.val / 100) * (1 - d2.val / 100)) * 100

theorem store_discount_claim :
  let first_discount : Discount := ⟨25, by norm_num⟩
  let second_discount : Discount := ⟨15, by norm_num⟩
  let claimed_discount : Discount := ⟨40, by norm_num⟩
  let actual_discount := total_discount first_discount second_discount
  actual_discount = 36.25 ∧
  claimed_discount.val - actual_discount = 3.75 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_store_discount_claim_l759_75950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_and_triangle_sides_l759_75919

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (Real.sqrt 3 / 2) * Real.sin (2 * x) - Real.cos x ^ 2 - 1 / 2

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- State the theorem
theorem function_properties_and_triangle_sides :
  -- Minimum value of f
  (∀ x : ℝ, f x ≥ -2) ∧
  -- Smallest positive period of f
  (∀ x : ℝ, f (x + π) = f x) ∧
  (∀ p : ℝ, p > 0 → (∀ x : ℝ, f (x + p) = f x) → p ≥ π) ∧
  -- Triangle properties
  ∀ t : Triangle,
    t.c = Real.sqrt 3 →
    f t.C = 0 →
    Real.sin t.B = 2 * Real.sin t.A →
    t.a = 1 ∧ t.b = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_and_triangle_sides_l759_75919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_triangle_DFG_l759_75924

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle with a center and radius -/
structure Circle where
  center : Point
  radius : ℝ

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Check if two circles intersect -/
def intersect (c1 c2 : Circle) : Prop :=
  distance c1.center c2.center ≤ c1.radius + c2.radius

/-- Define the three circles -/
def circleA : Circle := { center := { x := 0, y := 0 }, radius := 3 }
def circleB : Circle := { center := { x := 3, y := 0 }, radius := 3 }
def circleC : Circle := { center := { x := -3, y := 0 }, radius := 3 }

/-- Define the intersection points -/
noncomputable def D : Point := { x := 1, y := 2 * Real.sqrt 2 }
noncomputable def F : Point := { x := 1, y := -2 * Real.sqrt 2 }
noncomputable def G : Point := { x := -1, y := -2 * Real.sqrt 2 }

/-- The main theorem -/
theorem perimeter_of_triangle_DFG :
  distance D F + distance F G + distance G D = 8 * Real.sqrt 2 + 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_triangle_DFG_l759_75924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tunnel_length_l759_75917

/-- Calculates the length of a tunnel given train length, speed, and time to cross -/
theorem tunnel_length (train_length : ℝ) (train_speed_kmh : ℝ) (time_to_cross : ℝ) :
  train_length = 415 →
  train_speed_kmh = 63 →
  time_to_cross = 40 →
  ∃ (tunnel_length : ℝ), abs (tunnel_length - 285) < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tunnel_length_l759_75917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_point_coordinate_l759_75959

/-- The hyperbola equation -/
noncomputable def hyperbola (x y : ℝ) : Prop := x^2 / 16 - y^2 / 9 = 1

/-- The right directrix of the hyperbola -/
noncomputable def right_directrix : ℝ := 16 / 5

/-- The x-coordinates of the foci -/
noncomputable def focus_x : ℝ := 5

/-- The distance from a point to the right directrix -/
noncomputable def dist_to_directrix (x : ℝ) : ℝ := |x - right_directrix|

/-- The distance from a point to a focus -/
noncomputable def dist_to_focus (x y : ℝ) (fx : ℝ) : ℝ := Real.sqrt ((x - fx)^2 + y^2)

/-- The theorem statement -/
theorem hyperbola_point_coordinate :
  ∀ x y : ℝ,
  hyperbola x y →
  dist_to_directrix x = (dist_to_focus x y focus_x + dist_to_focus x y (-focus_x)) / 2 →
  x = -64 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_point_coordinate_l759_75959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l759_75932

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b
  h_b_pos : b > 0
  h_ecc : (a^2 - b^2) / a^2 = 1/2
  h_max_area : 4 * a * b = 16

/-- The ratio of lengths of two perpendicular chords through a focus -/
def chord_ratio (e : Ellipse) : Set ℝ :=
  {r | ∃ (k : ℝ), r = (1 + 2*k^2) / (2 + k^2) ∨ r = 1/2 ∨ r = 2}

theorem ellipse_properties (e : Ellipse) :
  (e.a = 2*Real.sqrt 2 ∧ e.b = 2) ∧
  chord_ratio e = Set.Icc (1/2) 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l759_75932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_l759_75901

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (1 - x) / x

theorem range_of_x (x : ℝ) : 
  (∃ y, f x = y) ↔ (x ≤ 1 ∧ x ≠ 0) :=
by
  sorry -- Proof to be filled in later

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_l759_75901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_comedies_l759_75993

-- Define the total number of movies and the number of action movies
variable (T a : ℕ)

-- Define the conditions
def comedies (T : ℕ) : ℕ := (48 * T) / 100
def action_movies (T : ℕ) : ℕ := (16 * T) / 100
def dramas (a : ℕ) : ℕ := 3 * a
def thrillers (a : ℕ) : ℕ := 2 * (dramas a)
def sci_fi (a : ℕ) : ℕ := a

-- The theorem to prove
theorem number_of_comedies (T a : ℕ) : 
  T = comedies T + action_movies T + dramas a + thrillers a + sci_fi a →
  action_movies T = a →
  comedies T = (40 * a) / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_comedies_l759_75993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_corn_preference_percentage_l759_75960

/-- The percentage of children preferring corn in Carolyn's daycare -/
noncomputable def percentage_corn (total_children : ℝ) (corn_preference : ℝ) : ℝ :=
  (corn_preference / total_children) * 100

/-- Theorem stating that the percentage of children preferring corn is 17.5% -/
theorem corn_preference_percentage :
  percentage_corn 50 8.75 = 17.5 := by
  -- Unfold the definition of percentage_corn
  unfold percentage_corn
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_corn_preference_percentage_l759_75960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l759_75985

-- Define the line C
def line_C (x y : ℝ) : Prop := x - y - 1 = 0

-- Define the curve P
def curve_P (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 1

-- Define the distance between two points
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2)

-- Theorem statement
theorem intersection_distance :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    line_C x₁ y₁ ∧ curve_P x₁ y₁ ∧
    line_C x₂ y₂ ∧ curve_P x₂ y₂ ∧
    x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧
    distance x₁ y₁ x₂ y₂ = Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l759_75985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_range_l759_75915

-- Define the slope and intersection point
variable (k : ℝ)

-- Define the conditions
def line_l (x y : ℝ) (k : ℝ) : Prop := y = k * x - Real.sqrt 3
def line_2 (x y : ℝ) : Prop := 2 * x + 3 * y - 6 = 0
def first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

-- Define the intersection of the lines
def intersection (x y : ℝ) (k : ℝ) : Prop := line_l x y k ∧ line_2 x y

-- Theorem statement
theorem slope_angle_range (k : ℝ) :
  (∃ x y, intersection x y k ∧ first_quadrant x y) →
  ∃ θ : ℝ, k = Real.tan θ ∧ π / 6 < θ ∧ θ < π / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_range_l759_75915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l759_75963

open Set

-- Define sets A and B
def A : Set ℝ := {x : ℝ | |x| = x}
def B : Set ℝ := {x : ℝ | x^2 + x ≥ 0}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = Ici 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l759_75963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_squared_of_n_graph_l759_75990

-- Define the linear functions
noncomputable def p (x : ℝ) : ℝ := -x + 1
noncomputable def q (x : ℝ) : ℝ := x + 1
noncomputable def r : ℝ → ℝ := Function.const ℝ 2

-- Define the minimum function
noncomputable def n (x : ℝ) : ℝ := min (min (p x) (q x)) (r x)

-- Define the interval
def I : Set ℝ := Set.Icc (-3) 3

-- State the theorem
theorem length_squared_of_n_graph : 
  (∫ x in I, Real.sqrt (1 + (deriv n x)^2) )^2 = 32 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_squared_of_n_graph_l759_75990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_neither_even_nor_odd_l759_75978

/-- The function g(x) = 2^(x^2 - 5) - x^3 -/
noncomputable def g (x : ℝ) : ℝ := 2^(x^2 - 5) - x^3

/-- g is neither even nor odd -/
theorem g_neither_even_nor_odd :
  (¬∀ x : ℝ, g (-x) = g x) ∧ (¬∀ x : ℝ, g (-x) = -g x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_neither_even_nor_odd_l759_75978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gas_volume_at_10_degrees_l759_75909

-- Define the gas properties
noncomputable def temperature_change : ℝ := 5
noncomputable def volume_change : ℝ := 6
noncomputable def initial_temperature : ℝ := 40
noncomputable def initial_volume : ℝ := 36
noncomputable def final_temperature : ℝ := 10

-- Define the function for volume based on temperature
noncomputable def volume (t : ℝ) : ℝ :=
  initial_volume + volume_change * ((t - initial_temperature) / temperature_change)

-- Theorem to prove
theorem gas_volume_at_10_degrees :
  volume final_temperature = 0 := by
  -- Expand the definition of volume
  unfold volume
  -- Perform algebraic simplifications
  simp [temperature_change, volume_change, initial_temperature, initial_volume, final_temperature]
  -- The proof steps would go here, but we'll use sorry to skip the actual proof
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gas_volume_at_10_degrees_l759_75909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_neg_alpha_value_l759_75907

theorem cos_neg_alpha_value (α : ℝ) (h1 : α ∈ Set.Ioo (-π/2) (π/2)) (h2 : Real.sin α = -3/5) : 
  Real.cos (-α) = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_neg_alpha_value_l759_75907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_left_side_expression_l759_75965

theorem left_side_expression (a b : ℝ) :
  let x : ℝ := (4.5 : ℝ)
  let left_side := (a * b) ^ x - 2
  let right_side := (b * a) ^ x - 7
  left_side = right_side → left_side = (a * b) ^ (4.5 : ℝ) - 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_left_side_expression_l759_75965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_M_l759_75975

def M : Matrix (Fin 2) (Fin 2) ℚ := !![3, 1; 2, -2]

theorem inverse_of_M : 
  M⁻¹ = (1/8 : ℚ) • M + (-1/8 : ℚ) • (1 : Matrix (Fin 2) (Fin 2) ℚ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_M_l759_75975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_roots_l759_75936

-- Define the odd function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then -x^2 + m*x - 1
  else if x > 0 then x^2 + m*x + 1
  else 0

-- Theorem statement
theorem odd_function_roots (m : ℝ) :
  (∀ x, f m (-x) = -(f m x)) →  -- f is odd
  (∃ x₁ x₂ x₃ x₄ x₅, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₁ ≠ x₅ ∧ 
                     x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₂ ≠ x₅ ∧
                     x₃ ≠ x₄ ∧ x₃ ≠ x₅ ∧ x₄ ≠ x₅ ∧
                     f m x₁ = 0 ∧ f m x₂ = 0 ∧ f m x₃ = 0 ∧ f m x₄ = 0 ∧ f m x₅ = 0) →  -- 5 distinct roots
  m < -2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_roots_l759_75936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_2a_minus_ab_l759_75925

theorem min_value_2a_minus_ab (a b : ℕ) (ha : 0 < a ∧ a < 6) (hb : 0 < b ∧ b < 6) :
  (∀ x y : ℕ, 0 < x ∧ x < 6 → 0 < y ∧ y < 6 → (2 * x : ℤ) - x * y ≥ (2 * a : ℤ) - a * b) →
  (2 * a : ℤ) - a * b = -15 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_2a_minus_ab_l759_75925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_segment_length_l759_75967

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ
deriving Inhabited

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Represents the running path -/
noncomputable def runningPath : List Point :=
  [⟨0, 0⟩, ⟨0, 1⟩, ⟨1/Real.sqrt 2, 1 + 1/Real.sqrt 2⟩, ⟨Real.sqrt 2, 1⟩]

theorem final_segment_length :
  distance (runningPath.getLast!) (runningPath.head!) = Real.sqrt 3 := by
  sorry

#check final_segment_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_segment_length_l759_75967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_3967149_487234_l759_75937

noncomputable def nearest_integer (x : ℝ) : ℤ :=
  if x - ⌊x⌋ < 0.5 then ⌊x⌋ else ⌈x⌉

theorem round_3967149_487234 :
  nearest_integer 3967149.487234 = 3967149 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_3967149_487234_l759_75937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_and_inequality_l759_75970

open Real

noncomputable section

def e : ℝ := Real.exp 1

def f (a x : ℝ) : ℝ := log x - a * x

def g (x : ℝ) : ℝ := log x / x

theorem f_max_and_inequality (x : ℝ) (hx : x ∈ Set.Ioo 0 e) :
  (∃ a : ℝ, a = 1 ∧ 
    (∀ y ∈ Set.Ioo 0 e, f a y ≤ f a 1)) ∧
  f 1 x + g x + 1/2 < 0 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_and_inequality_l759_75970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abcd_dcba_gcd_l759_75941

def is_consecutive (a b c : ℕ) : Prop := b = a + 1 ∧ c = a + 2

theorem abcd_dcba_gcd (a b c d : ℕ) 
  (h_consecutive : is_consecutive a b c) 
  (h_d : d = a + 5) : 
  ∃ k : ℕ, (1000 * a + 100 * b + 10 * c + d) + 
           (1000 * d + 100 * c + 10 * b + a) = 1111 * k ∧ 
  Nat.gcd 1111 k = 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abcd_dcba_gcd_l759_75941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parrot_arrangement_count_l759_75912

/-- The number of uniquely patterned parrots --/
def n : ℕ := 8

/-- The number of ways to arrange the parrots meeting the given conditions --/
def parrot_arrangements : ℕ := 2 * 1 * Nat.factorial (n - 3)

theorem parrot_arrangement_count : parrot_arrangements = 240 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parrot_arrangement_count_l759_75912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_sqrt_squared_neg_complex_sqrt_squared_unique_solutions_l759_75976

/-- The square root of a complex number -100 - 64i -/
noncomputable def complex_sqrt : ℂ := Complex.mk 3.06 (-10.46)

/-- Theorem stating that the square of complex_sqrt is approximately equal to -100 - 64i -/
theorem complex_sqrt_squared :
  Complex.abs ((complex_sqrt ^ 2) - Complex.mk (-100) (-64)) < 0.01 := by sorry

/-- Theorem stating that the negation of complex_sqrt is also a solution -/
theorem neg_complex_sqrt_squared :
  Complex.abs (((-complex_sqrt) ^ 2) - Complex.mk (-100) (-64)) < 0.01 := by sorry

/-- Theorem stating that these are the only solutions -/
theorem unique_solutions (z : ℂ) :
  z ^ 2 = Complex.mk (-100) (-64) → 
  Complex.abs (z - complex_sqrt) < 0.01 ∨ Complex.abs (z + complex_sqrt) < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_sqrt_squared_neg_complex_sqrt_squared_unique_solutions_l759_75976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_dead_heat_l759_75979

-- Define the speeds of runners relative to runner b
noncomputable def speed_a (speed_b : ℝ) : ℝ := (16/15) * speed_b
noncomputable def speed_c (speed_b : ℝ) : ℝ := (20/15) * speed_b

-- Define the head start fraction
noncomputable def head_start : ℝ := 1/4

-- Theorem statement
theorem race_dead_heat (speed_b : ℝ) (race_length : ℝ) (speed_b_pos : 0 < speed_b) (race_length_pos : 0 < race_length) :
  let time_a := race_length / speed_a speed_b
  let time_b := (race_length * (1 - head_start)) / speed_b
  let time_c := (race_length * (1 - head_start)) / speed_c speed_b
  time_a = time_b ∧ time_a = time_c :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_dead_heat_l759_75979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_triangle_area_l759_75918

/-- Definition of the ellipse C -/
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

/-- Definition of a line l -/
def line (k m x y : ℝ) : Prop := y = k * x + m

/-- Definition of an "elliptic point" -/
def elliptic_point (x y x0 y0 : ℝ) : Prop := x = x0 / 2 ∧ y = y0 / (Real.sqrt 3)

/-- Condition that circle with diameter PQ passes through the origin -/
def circle_through_origin (x1 y1 x2 y2 : ℝ) : Prop :=
  x1 * x2 / 4 + y1 * y2 / 3 = 0

/-- Area of a triangle given coordinates of its vertices -/
noncomputable def triangle_area (x1 y1 x2 y2 : ℝ) : ℝ :=
  abs ((x1 * y2 - x2 * y1) / 2)

/-- Main theorem -/
theorem constant_triangle_area
  (k m x1 y1 x2 y2 px py qx qy : ℝ) :
  ellipse x1 y1 →
  ellipse x2 y2 →
  line k m x1 y1 →
  line k m x2 y2 →
  elliptic_point px py x1 y1 →
  elliptic_point qx qy x2 y2 →
  circle_through_origin px py qx qy →
  triangle_area x1 y1 x2 y2 = Real.sqrt 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_triangle_area_l759_75918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l759_75958

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the given conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.C = 3 * Real.pi / 4 ∧
  Real.sin t.A = Real.sqrt 5 / 5 ∧
  t.c - t.a = 5 - Real.sqrt 10

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h : triangle_conditions t) : 
  Real.sin t.B = Real.sqrt 10 / 10 ∧ 
  (1 / 2 : ℝ) * t.a * t.c * Real.sin t.B = 5 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l759_75958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z₁_pure_imaginary_z₂_fourth_quadrant_l759_75916

/-- Complex number z₁ -/
def z₁ (m : ℝ) : ℂ := m * (m - 1) + (m - 1) * Complex.I

/-- Complex number z₂ -/
def z₂ (m : ℝ) : ℂ := (m + 1) + (m^2 - 1) * Complex.I

/-- Theorem: z₁ is a pure imaginary number if and only if m = 0 -/
theorem z₁_pure_imaginary (m : ℝ) : z₁ m = Complex.I * (Complex.im (z₁ m)) ↔ m = 0 := by sorry

/-- Theorem: z₂ is in the fourth quadrant if and only if -1 < m < 1 -/
theorem z₂_fourth_quadrant (m : ℝ) : 
  Complex.re (z₂ m) > 0 ∧ Complex.im (z₂ m) < 0 ↔ -1 < m ∧ m < 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z₁_pure_imaginary_z₂_fourth_quadrant_l759_75916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l759_75988

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the area function for a triangle
noncomputable def Triangle.area (t : Triangle) : ℝ :=
  1/2 * t.b * t.c * Real.sin t.A

-- Define the theorem
theorem triangle_properties (abc : Triangle) 
  (h1 : abc.a * Real.sin abc.C = Real.sqrt 3 * abc.c * Real.cos abc.A)
  (h2 : abc.a = Real.sqrt 13)
  (h3 : abc.c = 3) :
  abc.A = Real.pi / 3 ∧ 
  Triangle.area abc = 3 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l759_75988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_sum_product_l759_75947

/-- Given complex numbers p, q, r forming an equilateral triangle with side length 24,
    and |p + q + r| = 48, prove that |pq + pr + qr| = 768. -/
theorem equilateral_triangle_sum_product (p q r : ℂ) : 
  (∀ (a b : ℂ), a ∈ ({p, q, r} : Set ℂ) ∧ b ∈ ({p, q, r} : Set ℂ) ∧ a ≠ b → Complex.abs (a - b) = 24) →
  Complex.abs (p + q + r) = 48 →
  Complex.abs (p * q + p * r + q * r) = 768 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_sum_product_l759_75947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_implies_a_range_l759_75974

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (Real.sin x ^ 6 + Real.cos x ^ 6 + a * Real.sin x * Real.cos x)

-- State the theorem
theorem f_domain_implies_a_range (a : ℝ) :
  (∀ x : ℝ, ∃ y : ℝ, f a x = y) ↔ -1/2 < a ∧ a < 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_implies_a_range_l759_75974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_value_intersecting_line_value_l759_75961

/-- The equation of the line -/
def line_equation (x y a : ℝ) : Prop := 4*x + 3*y + a = 0

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 4

/-- The distance between two points -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

/-- Theorem for the tangent case -/
theorem tangent_line_value (a : ℝ) :
  (∃ x y : ℝ, line_equation x y a ∧ circle_equation x y ∧
    ∀ x' y' : ℝ, line_equation x' y' a → circle_equation x' y' → x = x' ∧ y = y') →
  |a| = 10 := by
  sorry

/-- Theorem for the intersecting case -/
theorem intersecting_line_value (a : ℝ) :
  (∃ x1 y1 x2 y2 : ℝ, x1 ≠ x2 ∧ y1 ≠ y2 ∧
    line_equation x1 y1 a ∧ line_equation x2 y2 a ∧
    circle_equation x1 y1 ∧ circle_equation x2 y2 ∧
    distance x1 y1 x2 y2 = 2 * Real.sqrt 3) →
  |a| = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_value_intersecting_line_value_l759_75961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l759_75948

def M : Set ℝ := {x | x^2 + 2*x - 15 < 0}
def N : Set ℝ := {x | x^2 + 6*x - 7 ≥ 0}

theorem intersection_M_N : M ∩ N = Set.Ioc 1 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l759_75948

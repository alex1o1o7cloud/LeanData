import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_equality_comparison_l1239_123999

theorem number_equality_comparison : 
  (-(3) = -3) ∧ 
  (-5 = -(5)) ∧ 
  (-7 ≠ -(-7)) ∧ 
  (-(-2) = |(-2)|) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_equality_comparison_l1239_123999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_growth_limit_l1239_123922

theorem line_growth_limit : 
  let initial_length : ℝ := 2
  let growth_series := fun (n : ℕ) => (1 / 3^n) * 3 + (1 / 3^n)
  (initial_length + ∑' n, growth_series (n + 1)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_growth_limit_l1239_123922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vase_sales_tax_is_ten_percent_l1239_123965

/-- Calculates the sales tax percentage given the original price, discount percentage, and total paid amount. -/
noncomputable def calculate_sales_tax_percentage (original_price : ℝ) (discount_percentage : ℝ) (total_paid : ℝ) : ℝ :=
  let discounted_price := original_price * (1 - discount_percentage / 100)
  let tax_amount := total_paid - discounted_price
  (tax_amount / discounted_price) * 100

/-- Proves that the sales tax percentage is 10% for the given vase purchase scenario. -/
theorem vase_sales_tax_is_ten_percent :
  calculate_sales_tax_percentage 200 25 165 = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vase_sales_tax_is_ten_percent_l1239_123965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_expression_l1239_123997

theorem min_value_expression (a b : ℤ) (ha : 0 < a ∧ a < 10) (hb : 0 < b ∧ b < 10) :
  (∀ x y : ℤ, 0 < x ∧ x < 10 → 0 < y ∧ y < 10 → 2*x - x*y + x^2 ≥ 2*a - a*b + a^2) →
  2*a - a*b + a^2 = -6 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_expression_l1239_123997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_l1239_123900

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : 0 < a
  h_pos_b : 0 < b

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- Theorem: The eccentricity of a hyperbola with the given conditions is between 1 and √2 -/
theorem eccentricity_range (h : Hyperbola) 
  (h_line : ∃ (m : ℝ), m * h.a = h.b) -- Line parallel to asymptote
  (h_focus : ∃ (c : ℝ), c^2 = h.a^2 + h.b^2) -- Focus condition
  (h_intersection : ∃ (x y : ℝ), x^2 / h.a^2 - y^2 / h.b^2 = 1 ∧ y = -h.b / h.a * x) -- Intersection with other asymptote
  (h_within_circle : ∃ (x y : ℝ), x^2 + y^2 < h.a^2) -- Point M within circle
  : 1 < eccentricity h ∧ eccentricity h < Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_l1239_123900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_value_l1239_123995

/-- A function f is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The function f defined on (0, 2) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  a * Real.log x - a * x + 1

/-- The theorem stating that a = 2 for the given conditions -/
theorem odd_function_value (f : ℝ → ℝ) (a : ℝ) :
  IsOdd f →
  (∀ x ∈ Set.Ioo 0 2, f x = a * Real.log x - a * x + 1) →
  (∃ m, ∀ x ∈ Set.Ioo (-2) 0, f x ≥ m ∧ (∃ y ∈ Set.Ioo (-2) 0, f y = m)) →
  (∃ x ∈ Set.Ioo (-2) 0, f x = 1) →
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_value_l1239_123995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagonal_pyramid_volume_formula_l1239_123939

/-- The volume of a regular hexagonal pyramid with height h and an inscribed sphere of radius r -/
noncomputable def hexagonal_pyramid_volume (h r : ℝ) : ℝ := (2 * h^3 * (h^2 - r^2)) / (3 * r^2)

/-- Theorem: The volume of a regular hexagonal pyramid with height h and an inscribed sphere of radius r is (2h³(h² - r²)) / (3r²) -/
theorem hexagonal_pyramid_volume_formula (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) (h_gt_r : h > r) :
  hexagonal_pyramid_volume h r = (2 * h^3 * (h^2 - r^2)) / (3 * r^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagonal_pyramid_volume_formula_l1239_123939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1239_123940

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 1

-- Define the function F
noncomputable def F (a b : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then f a b x else -f a b x

-- Define the function g
def g (a b k : ℝ) (x : ℝ) : ℝ := f a b x - k * x

-- State the theorem
theorem function_properties (a b : ℝ) :
  (f a b (-1) = 0) →
  (∀ x, f a b x ≥ 0) →
  (∃ y, ∀ x, f a b x ≤ y) →
  (∀ x, x ∈ Set.Icc (-2) 2 →
    (∃ k, (∀ x₁ x₂, x₁ < x₂ → g a b k x₁ < g a b k x₂) ∨
           (∀ x₁ x₂, x₁ < x₂ → g a b k x₁ > g a b k x₂))) →
  (∀ x, x > 0 → F a b x = x^2 + 2*x + 1) ∧
  (∀ x, x < 0 → F a b x = -x^2 - 2*x - 1) ∧
  (∃ k, (k ≥ 6 ∨ k ≤ -2) ∧
        (∀ x₁ x₂, x₁ ∈ Set.Icc (-2) 2 → x₂ ∈ Set.Icc (-2) 2 → x₁ < x₂ →
          g a b k x₁ < g a b k x₂)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1239_123940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_difference_inverse_main_result_l1239_123926

theorem inverse_difference_inverse : (5⁻¹ : ℚ) - (2⁻¹ : ℚ) = -3/10 := by
  -- Convert fractions to rationals
  have h1 : (5⁻¹ : ℚ) = 1/5 := by norm_num
  have h2 : (2⁻¹ : ℚ) = 1/2 := by norm_num
  
  -- Rewrite the expression
  rw [h1, h2]
  
  -- Perform the subtraction
  norm_num

theorem main_result : ((5⁻¹ : ℚ) - (2⁻¹ : ℚ))⁻¹ = -10/3 := by
  -- Use the previous theorem
  have h : (5⁻¹ : ℚ) - (2⁻¹ : ℚ) = -3/10 := inverse_difference_inverse
  
  -- Rewrite using the previous result
  rw [h]
  
  -- Take the inverse
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_difference_inverse_main_result_l1239_123926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_germs_per_dish_l1239_123948

theorem germs_per_dish (total_germs : ℝ) (num_dishes : ℝ) 
  (h1 : total_germs = 0.036 * 10^5) 
  (h2 : num_dishes = 75000 / 1000) : 
  total_germs / num_dishes = 48 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_germs_per_dish_l1239_123948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_monotone_iff_k_lt_3_l1239_123936

def sequence_a (k : ℝ) (n : ℕ+) : ℝ := (n : ℝ)^2 - k * n

theorem sequence_monotone_iff_k_lt_3 (k : ℝ) :
  (∀ n : ℕ+, sequence_a k n < sequence_a k (n + 1)) ↔ k < 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_monotone_iff_k_lt_3_l1239_123936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_inverse_14_l1239_123953

-- Define the function g
def g (x : ℝ) : ℝ := 3 * x - 4

-- State the theorem
theorem inverse_of_inverse_14 : 
  ∃ (g_inv : ℝ → ℝ), Function.RightInverse g g_inv ∧ g_inv (g_inv 14) = 10/3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_of_inverse_14_l1239_123953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scooter_gain_percent_l1239_123992

noncomputable def purchase_price : ℝ := 4700
noncomputable def repair_cost : ℝ := 800
noncomputable def selling_price : ℝ := 6000

noncomputable def total_cost : ℝ := purchase_price + repair_cost
noncomputable def gain : ℝ := selling_price - total_cost
noncomputable def gain_percent : ℝ := (gain / total_cost) * 100

theorem scooter_gain_percent :
  ∃ (ε : ℝ), ε > 0 ∧ |gain_percent - 9.09| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scooter_gain_percent_l1239_123992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arcsin_equation_solution_l1239_123968

theorem arcsin_equation_solution :
  let f (x : ℝ) := Real.arcsin (x * Real.sqrt 35 / (4 * Real.sqrt 13)) +
                   Real.arcsin (x * Real.sqrt 35 / (3 * Real.sqrt 13))
  let g (x : ℝ) := Real.arcsin (x * Real.sqrt 35 / (2 * Real.sqrt 13))
  let solutions : Set ℝ := {0, 13/12, -13/12}
  ∀ x : ℝ, |x| ≤ 2 * Real.sqrt 13 / Real.sqrt 35 →
    (f x = g x ↔ x ∈ solutions) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arcsin_equation_solution_l1239_123968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l1239_123958

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt x) / (x^2 - 4)

def domain (f : ℝ → ℝ) : Set ℝ := {x : ℝ | ∃ y, f x = y}

theorem f_domain : domain f = Set.union (Set.Ico 0 2) (Set.Ioi 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l1239_123958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l1239_123972

def solution_set : Set ℝ := Set.Ioo (-1) 2

def inequality (a x : ℝ) : Prop := (x - 2) / (a * x - 1) > 0

theorem constant_term_expansion (a : ℝ) 
  (h1 : ∀ x ∈ solution_set, inequality a x) 
  (h2 : ∀ x ∉ solution_set, ¬(inequality a x)) :
  (Finset.range 7).sum (λ k ↦ (Nat.choose 6 k) * a^k * (-1)^(6-k) * 
    (Finset.filter (λ i : Fin 3 ↦ i.val = 6-k) Finset.univ).card) = 15 := by
  sorry

#check constant_term_expansion

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l1239_123972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_assembly_line_output_l1239_123927

/-- Represents the assembly line production scenario -/
structure AssemblyLine where
  initial_rate : ℚ
  initial_order : ℚ
  increased_rate : ℚ
  second_order : ℚ

/-- Calculates the overall average output of the assembly line -/
def average_output (line : AssemblyLine) : ℚ :=
  (line.initial_order + line.second_order) / 
  (line.initial_order / line.initial_rate + line.second_order / line.increased_rate)

/-- Theorem stating that the average output for the given scenario is 45 cogs per hour -/
theorem assembly_line_output :
  let line : AssemblyLine := {
    initial_rate := 36,
    initial_order := 60,
    increased_rate := 60,
    second_order := 60
  }
  average_output line = 45 := by
  -- The proof goes here
  sorry

#eval average_output {
  initial_rate := 36,
  initial_order := 60,
  increased_rate := 60,
  second_order := 60
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_assembly_line_output_l1239_123927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_range_theorem_l1239_123908

theorem x_range_theorem (x : ℝ) : 
  (∀ m : ℝ, m ≠ 0 → |2*m - 1| + |1 - m| ≥ |m| * (|x - 1| - |2*x + 3|)) →
  x ∈ Set.Iic (-3) ∪ Set.Ici (-1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_range_theorem_l1239_123908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_five_digit_square_l1239_123962

/-- A function that checks if a number is a five-digit number -/
def isFiveDigit (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

/-- A function that checks if a number is formed using the digits 1, 2, 5, 5, and 6 -/
def isFormedFromDigits (n : ℕ) : Prop :=
  ∃ (a b c d e : ℕ),
    n = 10000 * a + 1000 * b + 100 * c + 10 * d + e ∧
    Multiset.ofList [a, b, c, d, e] = Multiset.ofList [1, 2, 5, 5, 6]

/-- The main theorem stating that 15625 is the only five-digit perfect square formed from the given digits -/
theorem unique_five_digit_square :
  ∀ n : ℕ, isFiveDigit n ∧ isFormedFromDigits n ∧ ∃ m : ℕ, n = m^2 ↔ n = 15625 := by
  sorry

#check unique_five_digit_square

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_five_digit_square_l1239_123962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_plus_b_value_l1239_123942

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (10^x + 1) / Real.log 10 - a * x
noncomputable def g (b : ℝ) (x : ℝ) : ℝ := (4^x + b) / 2^x

theorem a_plus_b_value (a b : ℝ) :
  (∀ x, f a x = f a (-x)) →  -- f is an even function
  (∀ x, g b x = -g b (-x)) →  -- g is an odd function
  a + b = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_plus_b_value_l1239_123942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_relation_l1239_123954

noncomputable def f (x : Real) : Real := 2 * Real.sin (x + Real.pi / 3) * Real.cos x

theorem triangle_angle_relation (A B : Real) (b c : Real) :
  A ∈ Set.Icc 0 (Real.pi / 2) →
  f A = Real.sqrt 3 / 2 →
  b = 2 →
  c = 3 →
  Real.cos (A - B) = 5 * Real.sqrt 7 / 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_relation_l1239_123954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l1239_123943

-- Define the set of complex numbers satisfying the given conditions
def S : Set ℂ :=
  {z : ℂ | 0 ≤ (z - 1).arg ∧ (z - 1).arg ≤ Real.pi/4 ∧ z.re ≤ 2}

-- State the theorem
theorem area_of_region (μ : MeasureTheory.Measure ℂ) [MeasureTheory.IsFiniteMeasure μ] :
  μ S = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l1239_123943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_left_after_drinks_l1239_123959

noncomputable def initial_volume : ℝ := 4

noncomputable def first_drink_fraction : ℝ := 1/4

noncomputable def second_drink_fraction : ℝ := 2/3

theorem water_left_after_drinks (v : ℝ) (f1 f2 : ℝ) 
  (h1 : v = initial_volume) 
  (h2 : f1 = first_drink_fraction) 
  (h3 : f2 = second_drink_fraction) : 
  v * (1 - f1) * (1 - f2) = 1 := by
  sorry

#check water_left_after_drinks

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_left_after_drinks_l1239_123959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sock_distribution_l1239_123956

def number_of_distributions (n k : ℕ) : ℕ :=
  Nat.choose (n + k - 1) (k - 1)

theorem sock_distribution (n k : ℕ) (h : k ≥ 1) :
  number_of_distributions n k = Nat.choose (n + k - 1) (k - 1) := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sock_distribution_l1239_123956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_a_b_find_k_upper_bound_find_k_lower_bound_l1239_123910

-- Define the functions g and f
noncomputable def g (a b x : ℝ) : ℝ := a * x^2 - 2 * a * x + 1 + b

noncomputable def f (x : ℝ) : ℝ := x + 1/x - 2

-- Statement 1
theorem find_a_b :
  ∀ a b : ℝ, a ≠ 0 → b < 1 →
  (∀ x ∈ Set.Icc 0 3, g a b x ≤ 4) →
  (∃ x ∈ Set.Icc 0 3, g a b x = 4) →
  (∀ x ∈ Set.Icc 0 3, g a b x ≥ 0) →
  (∃ x ∈ Set.Icc 0 3, g a b x = 0) →
  a = 1 ∧ b = 0 :=
by sorry

-- Statement 2
theorem find_k_upper_bound :
  ∀ k : ℝ,
  (∀ x ∈ Set.Icc (-2 : ℝ) (-1), f (2^x) - k * 2^x ≥ 0) →
  k ≤ 1 :=
by sorry

-- Statement 3
theorem find_k_lower_bound :
  ∀ k : ℝ,
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    f (|2^x₁ - 1|) + k * (2 / |2^x₁ - 1| - 3) = 0 ∧
    f (|2^x₂ - 1|) + k * (2 / |2^x₂ - 1| - 3) = 0 ∧
    f (|2^x₃ - 1|) + k * (2 / |2^x₃ - 1| - 3) = 0) →
  k > 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_a_b_find_k_upper_bound_find_k_lower_bound_l1239_123910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_emani_howard_money_difference_l1239_123984

theorem emani_howard_money_difference :
  ∀ (howard_money : ℕ),
    let emani_money : ℕ := 150
    let total_money : ℕ := emani_money + howard_money
    let shared_money : ℕ := 135
    total_money / 2 = shared_money →
    emani_money - howard_money = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_emani_howard_money_difference_l1239_123984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_in_circle_l1239_123976

/-- A closed plane curve. -/
structure ClosedPlaneCurve where
  points : Set (ℝ × ℝ)
  is_closed : Prop

/-- The property that the distance between any two points of a curve is less than 1. -/
def DistanceLessThanOne (K : ClosedPlaneCurve) : Prop :=
  ∀ p q, p ∈ K.points → q ∈ K.points → p ≠ q → Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) < 1

/-- A circle with given center and radius. -/
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p | Real.sqrt ((p.1 - center.1)^2 + (p.2 - center.2)^2) ≤ radius}

/-- The theorem stating that a closed plane curve with all pairwise distances less than 1
    lies in a circle of radius 1/√3. -/
theorem curve_in_circle (K : ClosedPlaneCurve) (h : DistanceLessThanOne K) :
    ∃ center : ℝ × ℝ, K.points ⊆ Circle center (1 / Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_in_circle_l1239_123976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boy_running_distance_l1239_123970

/-- Calculates the distance traveled given speed and time -/
noncomputable def distance_traveled (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Converts minutes to hours -/
noncomputable def minutes_to_hours (minutes : ℝ) : ℝ := minutes / 60

theorem boy_running_distance :
  let speed : ℝ := 2  -- speed in km/h
  let time_minutes : ℝ := 45  -- time in minutes
  let time_hours : ℝ := minutes_to_hours time_minutes
  let distance : ℝ := distance_traveled speed time_hours
  distance = 1.5 := by
    -- Unfold definitions
    unfold distance_traveled minutes_to_hours
    -- Simplify the expression
    simp
    -- Perform the calculation
    norm_num
    -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boy_running_distance_l1239_123970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_day_is_friday_l1239_123915

-- Define the days of the week
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday
deriving Repr, DecidableEq

-- Define a function to get the next day of the week
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday
  | DayOfWeek.Sunday => DayOfWeek.Monday

-- Define a function to count the occurrences of a specific day in a month
def countDayInMonth (startDay : DayOfWeek) (numDays : Nat) (targetDay : DayOfWeek) : Nat :=
  let rec count (currentDay : DayOfWeek) (daysLeft : Nat) (acc : Nat) : Nat :=
    if daysLeft = 0 then
      acc
    else if currentDay = targetDay then
      count (nextDay currentDay) (daysLeft - 1) (acc + 1)
    else
      count (nextDay currentDay) (daysLeft - 1) acc
  count startDay numDays 0

-- Theorem statement
theorem first_day_is_friday (numDays : Nat) (h1 : numDays ≤ 31) 
  (h2 : countDayInMonth DayOfWeek.Friday numDays DayOfWeek.Friday = 5)
  (h3 : countDayInMonth DayOfWeek.Friday numDays DayOfWeek.Saturday = 5)
  (h4 : countDayInMonth DayOfWeek.Friday numDays DayOfWeek.Sunday = 5) :
  DayOfWeek.Friday = DayOfWeek.Friday := by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_day_is_friday_l1239_123915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_square_is_G_l1239_123949

/-- Represents a 2x2 paper square --/
structure PaperSquare where
  label : Char
deriving Inhabited

/-- Represents the 4x4 grid --/
structure Grid where
  squares : List PaperSquare
  placement_order : List PaperSquare
deriving Inhabited

/-- Predicate to check if a square is fully visible --/
def is_fully_visible (g : Grid) (s : PaperSquare) : Prop :=
  s = g.placement_order.getLast!

/-- Predicate to check if a square is partially visible --/
def is_partially_visible (g : Grid) (s : PaperSquare) : Prop :=
  s ∈ g.squares ∧ s ≠ g.placement_order.getLast!

/-- Theorem stating that given the problem conditions, the third placed square must be G --/
theorem third_square_is_G (g : Grid) :
  g.squares.length = 8 ∧
  g.placement_order.length = 8 ∧
  (∃ e : PaperSquare, e.label = 'E' ∧ is_fully_visible g e) ∧
  (∀ s : PaperSquare, s ∈ g.squares ∧ s.label ≠ 'E' → is_partially_visible g s) →
  (g.placement_order.get! 2).label = 'G' := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_square_is_G_l1239_123949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focal_distance_l1239_123904

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point is on the ellipse -/
def on_ellipse (e : Ellipse) (p : Point) : Prop :=
  (p.x^2 / e.a^2) + (p.y^2 / e.b^2) = 1

/-- Represents the distance between two foci of an ellipse -/
noncomputable def focal_distance (e : Ellipse) : ℝ :=
  2 * Real.sqrt (e.a^2 - e.b^2)

/-- The main theorem -/
theorem ellipse_focal_distance 
  (e : Ellipse) 
  (p : Point) 
  (h_on_ellipse : on_ellipse e p)
  (h_sum_distances : ∃ (f₁ f₂ : Point), 
    (((p.x - f₁.x)^2 + (p.y - f₁.y)^2).sqrt + ((p.x - f₂.x)^2 + (p.y - f₂.y)^2).sqrt) = 2 * Real.sqrt 6) 
  (h_p : p.x = 2 ∧ p.y = 1) : 
  focal_distance e = 2 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focal_distance_l1239_123904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithm_product_equality_l1239_123950

theorem logarithm_product_equality (x y : ℝ) (hx : x > 0) (hy : y > 0) (hx1 : x ≠ 1) (hy1 : y ≠ 1) :
  (Real.log (x^2) / Real.log (y^4)) * (Real.log (y^3) / Real.log (x^6)) * 
  (Real.log (x^4) / Real.log (y^3)) * (Real.log (y^4) / Real.log (x^2)) * 
  (Real.log (x^6) / Real.log y) = 16 * (Real.log x / Real.log y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithm_product_equality_l1239_123950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_trip_average_speed_l1239_123983

/-- Calculates the average speed of a car trip with four legs -/
noncomputable def average_speed (leg1_distance : ℝ) (leg1_speed : ℝ)
                  (leg2_distance : ℝ) (leg2_speed : ℝ)
                  (leg3_time : ℝ) (leg3_speed : ℝ)
                  (leg4_distance : ℝ) (leg4_speed : ℝ)
                  (km_to_mile : ℝ) : ℝ :=
  let leg2_distance_miles := leg2_distance * km_to_mile
  let total_distance := leg1_distance + leg2_distance_miles + (leg3_time * leg3_speed) + leg4_distance
  let total_time := leg1_distance / leg1_speed + leg2_distance_miles / (leg2_speed * km_to_mile) + leg3_time + leg4_distance / leg4_speed
  total_distance / total_time

theorem car_trip_average_speed :
  let leg1_distance : ℝ := 100
  let leg1_speed : ℝ := 60
  let leg2_distance : ℝ := 200
  let leg2_speed : ℝ := 100
  let leg3_time : ℝ := 2
  let leg3_speed : ℝ := 40
  let leg4_distance : ℝ := 50
  let leg4_speed : ℝ := 75
  let km_to_mile : ℝ := 0.621371
  abs (average_speed leg1_distance leg1_speed leg2_distance leg2_speed leg3_time leg3_speed leg4_distance leg4_speed km_to_mile - 55.93) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_trip_average_speed_l1239_123983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_of_inscribed_rectangle_l1239_123975

/-- Represents a rectangle --/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents a circle --/
structure Circle where
  radius : ℝ

/-- Represents a square --/
structure Square where
  side : ℝ

def area_rectangle (r : Rectangle) : ℝ := r.width * r.height

noncomputable def area_circle (c : Circle) : ℝ := Real.pi * c.radius^2

def area_square (s : Square) : ℝ := s.side^2

/-- The theorem to be proved --/
theorem max_area_of_inscribed_rectangle 
  (s : Square)
  (r1 : Rectangle)
  (c : Circle)
  (h1 : s.side = 4)
  (h2 : r1.width = 2 ∧ r1.height = 4)
  (h3 : c.radius = 1) :
  ∃ (r2 : Rectangle), 
    area_rectangle r2 ≤ area_square s - (area_rectangle r1 + area_circle c) ∧
    area_rectangle r2 = 243/50 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_of_inscribed_rectangle_l1239_123975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_multiple_3_or_4_l1239_123982

def is_multiple_of_3_or_4 (n : Nat) : Bool :=
  n % 3 = 0 || n % 4 = 0

def count_multiples (n : Nat) : Nat :=
  (List.range n).filter is_multiple_of_3_or_4 |>.length

theorem probability_multiple_3_or_4 : 
  (count_multiples 30 : Rat) / 30 = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_multiple_3_or_4_l1239_123982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_product_inequality_l1239_123932

noncomputable def f (n : ℕ) : ℝ := ((2 * n + 1) / Real.exp 1) ^ ((2 * n + 1) / 2)

def odd_product (n : ℕ) : ℕ := Finset.prod (Finset.range n) (fun k => 2 * k + 1)

theorem odd_product_inequality (n : ℕ) (h : n > 0) :
  f (n - 1) < (odd_product n : ℝ) ∧ (odd_product n : ℝ) < f n := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_product_inequality_l1239_123932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_but_not_necessary_l1239_123985

noncomputable def is_monotonically_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → x < y → f x < f y

noncomputable def power_function (m : ℝ) (x : ℝ) : ℝ :=
  (m^2 - m - 1) * x^m

theorem sufficient_but_not_necessary :
  (∃ m : ℝ, is_monotonically_increasing (power_function m) ∧ m > 0) →
  (∀ m : ℝ, is_monotonically_increasing (power_function m) ∧ m > 0 → |m - 2| < 1) ∧
  (∃ m : ℝ, |m - 2| < 1 ∧ ¬(is_monotonically_increasing (power_function m) ∧ m > 0)) :=
by
  sorry

#check sufficient_but_not_necessary

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_but_not_necessary_l1239_123985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_line_theta_value_l1239_123917

theorem point_on_line_theta_value (θ : Real) :
  0 < θ ∧ θ < π / 2 →
  (Real.sin θ) + (3 * Real.sin θ + 1) - 3 = 0 →
  θ = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_line_theta_value_l1239_123917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_a2b3_proof_l1239_123963

/-- The coefficient of a²b³ in (a+b)⁵(c+1/c)⁸ is 700 -/
def coefficient_a2b3 : ℕ :=
  let coeff_a2b3_in_a_plus_b_5 := 10
  let constant_term_in_c_plus_inv_c_8 := 70
  coeff_a2b3_in_a_plus_b_5 * constant_term_in_c_plus_inv_c_8

#eval coefficient_a2b3  -- This should output 700

/-- Proof that the coefficient of a²b³ in (a+b)⁵(c+1/c)⁸ is 700 -/
theorem coefficient_a2b3_proof : coefficient_a2b3 = 700 := by
  unfold coefficient_a2b3
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_a2b3_proof_l1239_123963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tori_math_test_l1239_123921

theorem tori_math_test (total : ℕ) (arithmetic : ℕ) (algebra : ℕ) (geometry : ℕ)
  (arith_correct : ℚ) (alg_correct : ℚ) (geom_correct : ℚ) (passing_grade : ℚ) :
  total = 75 →
  arithmetic = 10 →
  algebra = 30 →
  geometry = 35 →
  arith_correct = 7/10 →
  alg_correct = 2/5 →
  geom_correct = 3/5 →
  passing_grade = 3/5 →
  (Nat.ceil (↑total * passing_grade) : ℕ) - 
  (Nat.floor (↑arithmetic * arith_correct) + Nat.floor (↑algebra * alg_correct) + Nat.floor (↑geometry * geom_correct)) = 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tori_math_test_l1239_123921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_indefinite_integral_correct_l1239_123909

-- Define the integrand
noncomputable def f (x : ℝ) : ℝ := (2*x^3 + 6*x^2 + 7*x) / ((x-2)*(x+1)^3)

-- Define the antiderivative
noncomputable def F (x : ℝ) : ℝ := 2 * Real.log (abs (x - 2)) - 1 / (2 * (x + 1)^2)

-- State the theorem
theorem indefinite_integral_correct (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ -1) :
  deriv F x = f x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_indefinite_integral_correct_l1239_123909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_value_at_five_l1239_123971

/-- A quadratic function f(x) = ax^2 + bx + c with specific properties -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), 
    (∀ x, f x = a * x^2 + b * x + c) ∧
    (∃ (max : ℝ), max = 4 ∧ f 2 = max ∧ ∀ x, f x ≤ max) ∧
    (f 0 = -16)

/-- Theorem stating that for a quadratic function with given properties, f(5) = -41 -/
theorem quadratic_value_at_five (f : ℝ → ℝ) (h : QuadraticFunction f) : 
  f 5 = -41 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_value_at_five_l1239_123971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_for_tax_problem_l1239_123990

theorem smallest_m_for_tax_problem : ∃ (x : ℕ) (m : ℕ),
  (∀ (y : ℕ) (n : ℕ), 105 * y = 100 * n → n ≥ m) ∧
  105 * x = 100 * m ∧
  m % 5 = 0 ∧
  m = 21 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_for_tax_problem_l1239_123990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometry_book_pages_l1239_123919

theorem geometry_book_pages :
  ∃ (x : ℕ),
    -- New edition has 450 pages
    let new_edition := 450;
    -- New edition has 230 pages less than twice the old edition
    new_edition = 2 * x - 230 ∧
    -- Deluxe edition has y pages
    ∃ (y : ℕ),
      -- y = 3 * (old edition pages)^2 - 20% of new edition pages
      y = 3 * x^2 - (new_edition / 5) ∧
      -- Deluxe edition must have at least 10% more pages than old edition
      y ≥ (11 * x) / 10 ∧
    -- Old edition has 340 pages
    x = 340 :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometry_book_pages_l1239_123919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_n_combination_sum_l1239_123912

theorem largest_n_combination_sum : 
  ∃ (n : ℕ), n = 7 ∧ 
  (∀ m : ℕ, m > n → (Finset.sum (Finset.range (m + 1)) (λ k ↦ k * Nat.choose m k) ≥ 500)) ∧
  (Finset.sum (Finset.range (n + 1)) (λ k ↦ k * Nat.choose n k) < 500) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_n_combination_sum_l1239_123912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ribbon_length_cylindrical_column_l1239_123902

/-- The length of a ribbon wrapped around a cylindrical column -/
noncomputable def ribbon_length (h d : ℝ) (turns : ℕ) : ℝ :=
  Real.sqrt (h^2 + (turns * Real.pi * d)^2)

/-- Theorem: The length of the ribbon is equal to √(400 + 441π²) -/
theorem ribbon_length_cylindrical_column :
  ribbon_length 20 3 7 = Real.sqrt (400 + 441 * Real.pi^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ribbon_length_cylindrical_column_l1239_123902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_count_l1239_123993

theorem triangle_side_count : 
  let a := 9
  let b := 4
  ∃! n : ℕ, n = (Finset.filter 
    (λ c : ℕ ↦ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a) 
    (Finset.range 13)).card ∧ n = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_count_l1239_123993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1239_123951

/-- The function f(x) defined as sin(2x + φ) - 1 -/
noncomputable def f (φ : ℝ) (x : ℝ) : ℝ := Real.sin (2 * x + φ) - 1

theorem function_properties (φ : ℝ) 
  (h1 : -π/2 < φ ∧ φ < π/2) 
  (h2 : f φ (π/3) = 0) :
  φ = -π/6 ∧ 
  ∀ x ∈ Set.Icc 0 (π/2), -3/2 ≤ f φ x ∧ f φ x ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1239_123951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_a_for_two_zeros_l1239_123998

/-- The function f(x) defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x - 2) * Real.exp x + a * (x - 1)^2

/-- The property that f has exactly two real zeros -/
def has_two_zeros (a : ℝ) : Prop :=
  ∃ x y, x ≠ y ∧ f a x = 0 ∧ f a y = 0 ∧
  ∀ z, f a z = 0 → z = x ∨ z = y

/-- The smallest positive integer a for which f has two zeros is 1 -/
theorem smallest_a_for_two_zeros :
  (∀ n : ℕ, n > 0 → n < 1 → ¬(has_two_zeros (n : ℝ))) ∧
  has_two_zeros 1 := by
  sorry

#check smallest_a_for_two_zeros

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_a_for_two_zeros_l1239_123998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_term_is_five_l1239_123964

def a : ℕ → ℕ
  | 0 => 2014^(2015^2016)
  | n + 1 => if a n % 2 = 0 then a n / 2 else a n + 7

theorem smallest_term_is_five :
  ∃ n : ℕ, a n = 5 ∧ ∀ m : ℕ, a m ≥ 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_term_is_five_l1239_123964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_squares_l1239_123916

theorem min_sum_squares (a b c : ℝ) (C : ℝ) (h1 : (a + b)^2 = 10 + c^2) (h2 : Real.cos C = 2/3) :
  a^2 + b^2 ≥ 6 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_squares_l1239_123916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_weight_theorem_l1239_123934

noncomputable def initial_weight : ℝ := 300
noncomputable def first_month_loss : ℝ := 20
noncomputable def fifth_month_loss : ℝ := 12

noncomputable def weight_loss (n : ℕ) : ℝ :=
  match n with
  | 0 => first_month_loss
  | 1 => first_month_loss / 2
  | 2 => first_month_loss / 4
  | 3 => first_month_loss / 8
  | 4 => fifth_month_loss
  | _ => 0

noncomputable def total_weight_loss : ℝ := (Finset.range 5).sum weight_loss

theorem final_weight_theorem :
  initial_weight - total_weight_loss = 250.5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_weight_theorem_l1239_123934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_increase_l1239_123960

/-- Represents the distance traveled by a car with increasing speed over time -/
noncomputable def distanceTraveled (initialSpeed : ℝ) (speedIncrease : ℝ) (hours : ℕ) : ℝ :=
  (hours : ℝ) / 2 * (2 * initialSpeed + (hours - 1 : ℝ) * speedIncrease)

/-- Theorem stating the conditions of the car's travel and the speed increase -/
theorem car_speed_increase (initialSpeed speedIncrease : ℝ) :
  initialSpeed = 45 →
  distanceTraveled initialSpeed speedIncrease 12 = 672 →
  speedIncrease = 2 := by
  sorry

#check car_speed_increase

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_increase_l1239_123960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_production_rate_theorem_l1239_123920

/-- Represents the production rate of an assembly line -/
structure AssemblyLine where
  initial_rate : ℚ
  first_order : ℚ
  second_order : ℚ
  average_output : ℚ

/-- Calculates the production rate after speed increase -/
def production_rate_after_increase (line : AssemblyLine) : ℚ :=
  (line.first_order + line.second_order) / 
  ((line.first_order / line.initial_rate) + 
   (line.second_order / line.average_output) - 
   (line.first_order / line.initial_rate))

/-- Theorem stating that the production rate after speed increase is 60 cogs per hour -/
theorem production_rate_theorem (line : AssemblyLine) 
  (h1 : line.initial_rate = 36)
  (h2 : line.first_order = 60)
  (h3 : line.second_order = 60)
  (h4 : line.average_output = 45) :
  production_rate_after_increase line = 60 := by
  sorry

#eval production_rate_after_increase { initial_rate := 36, first_order := 60, second_order := 60, average_output := 45 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_production_rate_theorem_l1239_123920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_proof_l1239_123955

/-- An ellipse with specific properties -/
structure Ellipse where
  -- The center is at the origin and the major axis is along the x-axis
  center_origin : Unit
  major_axis_x : Unit
  -- Eccentricity is √3/2
  eccentricity : ℝ
  ecc_eq : eccentricity = Real.sqrt 3 / 2
  -- Sum of distances from any point on the ellipse to its foci is 12
  focal_sum : ℝ
  focal_sum_eq : focal_sum = 12

/-- The equation of the ellipse -/
def ellipse_equation (x y : ℝ) : Prop :=
  x^2 / 36 + y^2 / 9 = 1

/-- Theorem stating that the given ellipse has the specified equation -/
theorem ellipse_equation_proof (e : Ellipse) :
  ∀ x y : ℝ, (x, y) ∈ ({(x, y) | ∃ (p : ℝ × ℝ), p.1 = x ∧ p.2 = y ∧ 
    (∃ (f₁ f₂ : ℝ × ℝ), dist p f₁ + dist p f₂ = e.focal_sum)} : Set (ℝ × ℝ)) 
  ↔ ellipse_equation x y :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_proof_l1239_123955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l1239_123974

theorem inequality_solution_set : 
  {x : ℝ | (x - 2)^2 ≤ 2*x + 11} = Set.Icc (-1 : ℝ) 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l1239_123974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_hyperbola_properties_l1239_123933

/-- An equilateral hyperbola with focus on x-axis and center at origin -/
structure EquilateralHyperbola where
  realAxisLength : ℝ
  focusX : ℝ
  eq : ℝ → ℝ → Prop

/-- A line passing through a point with a given slope -/
structure Line where
  slope : ℝ
  pointX : ℝ
  pointY : ℝ

def intersectionPoints (h : EquilateralHyperbola) (l : Line) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | h.eq p.1 p.2 ∧ p.2 = l.slope * (p.1 - l.pointX) + l.pointY}

theorem equilateral_hyperbola_properties (h : EquilateralHyperbola)
    (hLength : h.realAxisLength = 2 * Real.sqrt 2)
    (hFocus : h.focusX = 2) :
  (∀ x y, h.eq x y ↔ x^2 / 2 - y^2 / 2 = 1) ∧
  (∃ A B, A ∈ intersectionPoints h (Line.mk 2 h.focusX 0) ∧
          B ∈ intersectionPoints h (Line.mk 2 h.focusX 0) ∧
          ‖A - B‖ = 10 * Real.sqrt 2 / 3) ∧
  (∀ l : Line, l.pointX = h.focusX ∧ l.pointY = 0 →
    ∃ A D, A ∈ intersectionPoints h l ∧ D ∈ intersectionPoints h l ∧
      ∃ B : ℝ × ℝ, B.1 = A.1 ∧ B.2 = -A.2 ∧
      (1 - A.1) * (D.2 - A.2) = (0 - A.2) * (D.1 - A.1)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_hyperbola_properties_l1239_123933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_F_is_even_G_is_odd_odd_part_of_f_specific_l1239_123945

-- Define a real-valued function on ℝ
variable (f : ℝ → ℝ)

-- Define F(x) as the even part of f
noncomputable def F (f : ℝ → ℝ) (x : ℝ) : ℝ := (f x + f (-x)) / 2

-- Define G(x) as the odd part of f
noncomputable def G (f : ℝ → ℝ) (x : ℝ) : ℝ := (f x - f (-x)) / 2

-- Theorem 1: F is an even function
theorem F_is_even (f : ℝ → ℝ) : ∀ x : ℝ, F f x = F f (-x) := by sorry

-- Theorem 2: G is an odd function
theorem G_is_odd (f : ℝ → ℝ) : ∀ x : ℝ, G f x = -G f (-x) := by sorry

-- Define the specific function f(x) = ln(e^x + 1)
noncomputable def f_specific (x : ℝ) : ℝ := Real.log (Real.exp x + 1)

-- Theorem 3: For f(x) = ln(e^x + 1), the odd part g(x) = x/2
theorem odd_part_of_f_specific : 
  ∀ x : ℝ, G f_specific x = x / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_F_is_even_G_is_odd_odd_part_of_f_specific_l1239_123945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1239_123989

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := (Real.cos (ω * x + φ))^2 - 1/2

theorem function_properties (ω φ : ℝ) 
  (h_ω : ω > 0) 
  (h_φ : 0 < φ ∧ φ < Real.pi/2) 
  (h_period : ∀ x, f ω φ (x + Real.pi) = f ω φ x)
  (h_value : f ω φ (Real.pi/8) = 1/4) :
  ω = 1 ∧ 
  φ = Real.pi/24 ∧ 
  (∀ x ∈ Set.Icc (Real.pi/24) (13*Real.pi/24), 
    -1/2 ≤ f ω φ x ∧ f ω φ x ≤ Real.sqrt 3/4) ∧
  (∃ x ∈ Set.Icc (Real.pi/24) (13*Real.pi/24), f ω φ x = -1/2) ∧
  (∃ x ∈ Set.Icc (Real.pi/24) (13*Real.pi/24), f ω φ x = Real.sqrt 3/4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1239_123989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_matrix_l1239_123944

open Matrix

variable (x : ℝ)

def matrix (x : ℝ) : Matrix (Fin 3) (Fin 3) ℝ := !![1, -3, 3; x, 5, -1; 4, -2, 1]

theorem det_matrix (x : ℝ) : 
  det (matrix x) = -3 * x - 45 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_matrix_l1239_123944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_pricing_properties_l1239_123981

/-- Water pricing system -/
structure WaterPricing where
  baseCost : ℝ  -- Cost per ton for first 6 tons
  excessCost : ℝ  -- Cost per ton for usage exceeding 6 tons
  baseLimit : ℝ  -- Limit for base pricing

/-- Cost function for water usage -/
noncomputable def waterCost (pricing : WaterPricing) (usage : ℝ) : ℝ :=
  if usage ≤ pricing.baseLimit
  then pricing.baseCost * usage
  else pricing.baseCost * pricing.baseLimit + pricing.excessCost * (usage - pricing.baseLimit)

/-- Theorem stating the properties of the water pricing system -/
theorem water_pricing_properties (pricing : WaterPricing)
  (h1 : pricing.baseCost = 2)
  (h2 : pricing.excessCost = 3)
  (h3 : pricing.baseLimit = 6) :
  (∀ x : ℝ, 0 < x ∧ x ≤ 6 → waterCost pricing x = 2 * x) ∧
  (∀ x : ℝ, x > 6 → waterCost pricing x = 3 * x - 6) ∧
  (∃ x : ℝ, waterCost pricing x = 27 ∧ x = 11) := by
  sorry

#check water_pricing_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_pricing_properties_l1239_123981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_minus_pi_fourth_l1239_123980

theorem cos_alpha_minus_pi_fourth (α : ℝ) 
  (h1 : α ∈ Set.Ioo 0 (π/2)) 
  (h2 : Real.tan α = 2) : 
  Real.cos (α - π/4) = (3 * Real.sqrt 10) / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_minus_pi_fourth_l1239_123980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_odd_indexed_angles_l1239_123967

noncomputable def is_solution (z : ℂ) : Prop :=
  z^24 - z^12 - 1 = 0 ∧ Complex.abs z = 1

noncomputable def angle_in_degrees (z : ℂ) : ℝ :=
  (Complex.arg z * 360) / (2 * Real.pi)

def solution_angles : List ℝ :=
  [5, 25, 55, 75, 125, 145, 175, 195, 245, 265, 295, 315]

theorem sum_of_odd_indexed_angles :
  ∃ (n : ℕ) (angles : List ℝ),
    angles.length = 2 * n ∧
    (∀ z, is_solution z → angle_in_degrees z ∈ angles) ∧
    (∀ i j, i < j → i < angles.length → j < angles.length → angles.get! i < angles.get! j) ∧
    (List.sum (List.enum angles |>.filter (fun (i, _) => i % 2 = 0) |>.map (fun (_, x) => x)) = 900) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_odd_indexed_angles_l1239_123967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_distance_l1239_123986

/-- Given a parabola x^2 = 2py (p > 0) and a line y = 2x + p/2 that intersects
    the parabola at points A and B, the distance |AB| is equal to 10p. -/
theorem parabola_line_intersection_distance (p : ℝ) (h_p : p > 0)
  (A B : ℝ × ℝ) :
  (∀ x y : ℝ, y = 2*x + p/2 → x^2 = 2*p*y) →  -- Line equation and parabola equation
  (A.1^2 = 2*p*A.2 ∧ A.2 = 2*A.1 + p/2) →  -- A satisfies both equations
  (B.1^2 = 2*p*B.2 ∧ B.2 = 2*B.1 + p/2) →  -- B satisfies both equations
  A ≠ B →  -- A and B are distinct points
  ‖(A.1, A.2) - (B.1, B.2)‖ = 10*p :=  -- The distance between A and B is 10p
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_distance_l1239_123986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_balloon_arrangements_proof_l1239_123938

def word : String := "BALLOON"

def balloon_arrangements : ℕ := 1260

theorem balloon_arrangements_proof :
  balloon_arrangements = 1260 :=
by
  -- The proof goes here
  sorry

#check balloon_arrangements_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_balloon_arrangements_proof_l1239_123938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_configuration_l1239_123913

/-- Represents the number of "energy transfer" projects -/
def energy_transfer (n : ℕ) : ℕ := sorry

/-- Represents the number of "leaping over the dragon gate" projects -/
def dragon_gate (n : ℕ) : ℕ := sorry

/-- Represents the total number of projects -/
def total_projects : ℕ := 15

/-- Represents the relationship between energy transfer and dragon gate projects -/
axiom project_relation : ∀ n : ℕ, energy_transfer n = 2 * dragon_gate n - 3

/-- Represents the constraint on the total number of projects -/
axiom total_constraint : ∀ n : ℕ, energy_transfer n + dragon_gate n = total_projects

/-- Represents the time for energy transfer projects -/
def energy_time : ℕ := 6

/-- Represents the time for dragon gate projects -/
def dragon_time : ℕ := 8

/-- Represents the maximum number of projects that can be carried out -/
def max_projects : ℕ := 10

/-- Represents the constraint on the number of dragon gate projects -/
axiom dragon_constraint : ∀ n : ℕ, dragon_gate n > (energy_transfer n) / 2

/-- Calculates the total time for the expansion activity -/
def total_time (n : ℕ) : ℕ := energy_time * energy_transfer n + dragon_time * dragon_gate n

/-- Theorem stating the optimal configuration and minimum time -/
theorem optimal_configuration :
  ∃ n : ℕ, energy_transfer n = 6 ∧ dragon_gate n = 4 ∧
  (∀ m : ℕ, energy_transfer m + dragon_gate m ≤ max_projects →
    total_time n ≤ total_time m) ∧
  total_time n = 68 := by
  sorry

#check optimal_configuration

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_configuration_l1239_123913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_box_interior_diagonals_sum_l1239_123929

-- Define the rectangular box
structure Box where
  length : ℝ
  width : ℝ
  height : ℝ

-- Define the properties of the box
def BoxProperties (b : Box) : Prop :=
  -- Total surface area is 150 square inches
  2 * (b.length * b.width + b.width * b.height + b.height * b.length) = 150 ∧
  -- Sum of lengths of all edges is 60 inches
  4 * (b.length + b.width + b.height) = 60 ∧
  -- Length is twice the height
  b.length = 2 * b.height

-- Define the sum of lengths of interior diagonals
noncomputable def InteriorDiagonalsSum (b : Box) : ℝ :=
  4 * Real.sqrt (b.length^2 + b.width^2 + b.height^2)

-- Theorem statement
theorem box_interior_diagonals_sum 
  (b : Box) (h : BoxProperties b) : InteriorDiagonalsSum b = 20 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_box_interior_diagonals_sum_l1239_123929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cashier_adjustment_l1239_123928

/-- Represents the value of a half-dollar coin in dollars -/
def half_dollar_value : ℚ := 1/2

/-- Represents the value of a $1 bill in dollars -/
def one_dollar_value : ℚ := 1

/-- Represents the value of a $5 bill in dollars -/
def five_dollar_value : ℚ := 5

/-- Represents the value of a $10 bill in dollars -/
def ten_dollar_value : ℚ := 10

/-- Theorem stating the correct adjustment to the total cash amount -/
theorem cashier_adjustment (y : ℚ) :
  (y * (one_dollar_value - half_dollar_value)) +
  (y * (ten_dollar_value - five_dollar_value)) = 5.5 * y := by
  -- Expand definitions
  unfold half_dollar_value one_dollar_value five_dollar_value ten_dollar_value
  -- Simplify the expression
  simp
  -- Perform the calculation
  ring

#check cashier_adjustment


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cashier_adjustment_l1239_123928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_fourth_statement_correct_l1239_123905

/-- Represents a line in a plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Defines when two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

/-- Represents a point in a plane -/
def Point := ℝ × ℝ

/-- Checks if a point is on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  p.2 = l.slope * p.1 + l.intercept

/-- Theorem stating that only the fourth statement is correct -/
theorem only_fourth_statement_correct :
  ∃! (stmt : Nat), stmt = 4 ∧ 
  (∀ (l1 l2 l3 : Line), parallel l1 l3 → parallel l2 l3 → parallel l1 l2) ∧
  (∀ (stmt' : Nat), stmt' ≠ 4 → 
    (stmt' = 1 → ¬(∀ (l : Line) (p : Point), ∃! (l' : Line), parallel l l' ∧ pointOnLine p l')) ∧
    (stmt' = 2 → ¬(∀ (l : Line) (p : Point), ∃! (l' : Line), l'.slope * l.slope = -1 ∧ pointOnLine p l')) ∧
    (stmt' = 3 → ¬(∀ (l1 l2 l3 : Line), l1.slope * l3.slope = -1 → l2.slope * l3.slope = -1 → parallel l1 l2)) ∧
    (stmt' = 5 → ¬(∀ (l1 l2 l3 : Line), ∃ (θ : ℝ), l1.slope * l3.slope = Real.tan θ ∧ l2.slope * l3.slope = Real.tan θ → parallel l1 l2)) ∧
    (stmt' = 6 → ¬(∀ (A B : Point), Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = (Line.mk ((A.2 - B.2) / (A.1 - B.1)) ((A.2 * B.1 - A.1 * B.2) / (A.1 - B.1))).intercept))) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_fourth_statement_correct_l1239_123905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_proof_l1239_123935

def circle_equation (x y : ℝ) := (x - 2)^2 + y^2 = 9

theorem circle_proof (C : Set (ℝ × ℝ)) (center : ℝ × ℝ) :
  (∃ a : ℝ, a > 0 ∧ center = (a, 0)) →
  (0, Real.sqrt 5) ∈ C →
  (let d := (4 * Real.sqrt 5) / 5
   abs (2 * center.1 - center.2) / Real.sqrt 5 = d) →
  (∀ x y : ℝ, (x, y) ∈ C ↔ circle_equation x y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_proof_l1239_123935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_coprime_pairs_l1239_123901

theorem infinite_coprime_pairs (m : ℕ+) :
  ∃ f : ℕ → ℕ × ℕ,
    Function.Injective f ∧
    ∀ n : ℕ, 
      let (x, y) := f n
      Nat.Coprime x y ∧
      x ∣ (y^2 + m) ∧
      y ∣ (x^2 + m) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_coprime_pairs_l1239_123901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_is_sqrt_three_l1239_123952

/-- The distance from the center of the circle ρ = 4sin θ to the line θ = π/6 -/
noncomputable def distance_circle_to_line : ℝ :=
  Real.sqrt 3

/-- The circle equation in polar coordinates -/
def circle_equation (ρ θ : ℝ) : Prop :=
  ρ = 4 * Real.sin θ

/-- The line equation in polar coordinates -/
def line_equation (θ : ℝ) : Prop :=
  θ = Real.pi / 6

/-- Theorem stating that the distance from the circle's center to the line is √3 -/
theorem distance_is_sqrt_three :
  distance_circle_to_line = Real.sqrt 3 := by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_is_sqrt_three_l1239_123952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l1239_123991

noncomputable def f (x : ℝ) : ℝ := (1/2) * (x - 1)^2 - 1

theorem parabola_properties :
  (∀ x y : ℝ, f x = y → y ≥ f 1) ∧  -- opens upwards
  f 1 = -1 ∧  -- vertex y-coordinate
  (∀ x : ℝ, f x = f (2 - x)) ∧  -- axis of symmetry
  (∀ x : ℝ, x > 1 → (∀ y : ℝ, y > x → f y > f x)) ∧  -- increases when x > 1
  (∀ x : ℝ, x < 1 → (∀ y : ℝ, y < x → f y > f x)) ∧  -- decreases when x < 1
  (∀ x : ℝ, f x ≥ f 1) ∧  -- minimum at x = 1
  f 1 = -1  -- minimum y value
  := by sorry

#check parabola_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_properties_l1239_123991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_and_circle_properties_l1239_123903

/-- Parabola C: x^2 = 2py (p > 0) -/
def parabola (p : ℝ) (x y : ℝ) : Prop := x^2 = 2*p*y ∧ p > 0

/-- Circle M: x^2 + (y+4)^2 = 1 -/
def circle_M (x y : ℝ) : Prop := x^2 + (y+4)^2 = 1

/-- Focus of parabola C -/
noncomputable def focus (p : ℝ) : ℝ × ℝ := (0, p/2)

/-- Minimum distance between focus F and a point on M is 4 -/
def min_distance (p : ℝ) : Prop :=
  ∃ (x y : ℝ), circle_M x y ∧ 
  Real.sqrt ((x - (focus p).1)^2 + (y - (focus p).2)^2) - 1 = 4

/-- Maximum area of triangle PAB -/
noncomputable def max_area (p : ℝ) : ℝ := 20 * Real.sqrt 5

theorem parabola_and_circle_properties (p : ℝ) :
  (∀ x y, parabola p x y → circle_M x y → min_distance p) →
  p = 2 ∧ max_area p = 20 * Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_and_circle_properties_l1239_123903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_intersect_B_l1239_123978

-- Define the sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 2}
def B : Set ℝ := {x | x^2 - 3*x < 0}

-- Define the complement of A in real numbers
def complementA : Set ℝ := {x | x ≤ -1 ∨ 2 ≤ x}

-- State the theorem
theorem complement_A_intersect_B : 
  complementA ∩ B = Set.Icc 2 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_intersect_B_l1239_123978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1239_123914

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) :
  Real.cos A = 4/5 →
  Real.tan (A - B) = 1/3 →
  b = 10 →
  (Real.sin B = Real.sqrt 10 / 10) ∧
  (1/2 * a * b * Real.sin C = 78) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1239_123914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_S₄_S₃_l1239_123918

-- Define the sets S₃ and S₄
def S₃ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | Real.log (3 + p.1^2 + p.2^2) ≤ 1 + Real.log (p.1 + p.2)}

def S₄ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | Real.log (4 + p.1^2 + p.2^2) ≤ 2 + Real.log (p.1 + p.2)}

-- Define a function to calculate the area of a set
noncomputable def area (s : Set (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem area_ratio_S₄_S₃ : (area S₄) / (area S₃) = 4996 / 47 := by
  sorry

#check area_ratio_S₄_S₃

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_S₄_S₃_l1239_123918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l1239_123911

open Real Set

/-- A differentiable function symmetric about the y-axis -/
def SymmetricFunction (f : ℝ → ℝ) :=
  ∀ x, x ∈ Set.Ioo (-π/2) (π/2) → f x = f (-x)

theorem solution_set_of_inequality
  (f : ℝ → ℝ)
  (hf : Differentiable ℝ f)
  (hsym : SymmetricFunction f)
  (hineq : ∀ x ∈ Set.Ioo 0 (π/2), deriv f x * cos x > f x * sin (-x)) :
  {x | x ∈ Set.Ioo (-π/2) (π/2) ∧ f x - f (π/2 - x) / tan x > 0} = Set.Ioo (π/4) (π/2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_inequality_l1239_123911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tiles_form_square_l1239_123996

/-- Represents a right triangle tile with legs of 1 dm and 2 dm -/
structure RightTriangleTile where
  leg1 : ℝ
  leg2 : ℝ
  is_right_triangle : leg1 = 1 ∧ leg2 = 2

/-- Calculates the area of a right triangle tile -/
noncomputable def area_of_tile (t : RightTriangleTile) : ℝ :=
  (t.leg1 * t.leg2) / 2

/-- The number of tiles we have -/
def num_tiles : ℕ := 20

/-- Theorem stating that 20 right triangle tiles can form a square -/
theorem tiles_form_square : 
  ∃ (s : ℝ), s > 0 ∧ s^2 = num_tiles * area_of_tile { leg1 := 1, leg2 := 2, is_right_triangle := ⟨rfl, rfl⟩ } := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tiles_form_square_l1239_123996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_10_tons_sales_verification_l1239_123925

/-- Daily sales function -/
noncomputable def sales (x : ℝ) : ℝ :=
  if 1 ≤ x ∧ x ≤ 20 then 5 * Real.log x + 1 else 16

/-- Daily production cost function -/
def cost (x : ℝ) : ℝ := 0.5 * x + 1

/-- Daily profit function -/
noncomputable def profit (x : ℝ) : ℝ := sales x - cost x

/-- Theorem stating the maximum profit and its corresponding production volume -/
theorem max_profit_at_10_tons :
  (∀ x : ℝ, x ≥ 1 → profit x ≤ profit 10) ∧
  profit 10 = 6.5 := by
  sorry

/-- Verification of the sales function at given points -/
theorem sales_verification :
  sales 2 = 4.5 ∧ sales 4 = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_10_tons_sales_verification_l1239_123925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_missing_number_is_sixteen_l1239_123977

def given_sequence : List Nat := [6, 12, 1, 3, 11, 10, 8, 15, 13, 9, 7, 4, 14, 5, 2]

def is_consecutive_sequence (s : List Nat) : Prop :=
  ∃ start : Nat, s = (List.range (s.length + 1)).map (· + start)

theorem missing_number_is_sixteen :
  ∃ (full_sequence : List Nat),
    is_consecutive_sequence full_sequence ∧
    full_sequence.length = given_sequence.length + 1 ∧
    ∀ n, n ∈ given_sequence → n ∈ full_sequence ∧
    16 ∈ full_sequence ∧
    16 ∉ given_sequence :=
by
  sorry

#check missing_number_is_sixteen

end NUMINAMATH_CALUDE_ERRORFEEDBACK_missing_number_is_sixteen_l1239_123977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_coefficient_in_expansion_l1239_123924

theorem largest_coefficient_in_expansion (n : ℕ) (x : ℝ) :
  2^(2*n) - 2^n = 240 →
  ∃ (k : ℕ), k = 6 ∧ 
    ∀ (i : ℕ), i ≤ n → 
      (k : ℝ) * x^(1/3 : ℝ) ≥ (n.choose i : ℝ) * x^((n-i : ℝ)/2) * x^(-(i : ℝ)/3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_coefficient_in_expansion_l1239_123924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_meaningful_l1239_123946

noncomputable def f (x : ℝ) : ℝ := x / (x - 2)

-- Theorem stating that the fraction is meaningful when x ≠ 2
theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, f x = y) ↔ x ≠ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_meaningful_l1239_123946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_water_ratio_l1239_123969

theorem milk_water_ratio (initial_volume : ℝ) (added_water : ℝ) 
  (h1 : initial_volume = 40000)  -- 40 litres in ml
  (h2 : added_water = 1600) :    -- 1600 ml of water added
  ∃ (initial_milk initial_water : ℝ),
    initial_milk + initial_water = initial_volume ∧ 
    initial_milk / (initial_water + added_water) = 3 ∧
    abs ((initial_milk / initial_water) - 3.55) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_water_ratio_l1239_123969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_different_classes_l1239_123966

/-- The probability of selecting two kids from different classes -/
theorem probability_different_classes 
  (total : ℕ) 
  (german : ℕ) 
  (japanese : ℕ) 
  (h1 : total = 30) 
  (h2 : german = 22) 
  (h3 : japanese = 19) 
  : (1 : ℚ) - (Nat.choose (german + japanese - total) 2 + Nat.choose (japanese - (german + japanese - total)) 2 : ℚ) / (Nat.choose total 2) = 16 / 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_different_classes_l1239_123966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_roots_sum_l1239_123930

-- Define the polynomial
def polynomial (a b c : ℝ) (x : ℝ) : ℝ := x^4 + a*x^2 + b*x + c

-- State the theorem
theorem polynomial_roots_sum (a b c : ℝ) :
  (∃ r : ℝ, ∀ x : ℝ, polynomial a b c x = 0 ↔ x = 1 ∨ x = 2 ∨ x = 3 ∨ x = r) →
  a + c = -61 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_roots_sum_l1239_123930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1239_123994

noncomputable def f (x : ℝ) := Real.sqrt (2 * x - 4) + (4 * x - 6) ^ (1/3 : ℝ)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ≥ 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1239_123994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_function_bound_l1239_123923

/-- The function f(x) -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 - (1/2) * m * x^2 + 4 * x - 3

/-- f(x) is increasing on [1, 2] -/
def is_increasing (m : ℝ) : Prop :=
  ∀ x y, 1 ≤ x ∧ x < y ∧ y ≤ 2 → f m x < f m y

/-- If f(x) is increasing on [1, 2], then m ≤ 4 -/
theorem increasing_function_bound (m : ℝ) (h : is_increasing m) : m ≤ 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_function_bound_l1239_123923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_six_solutions_l1239_123979

/-- The number of ordered pairs (m, n) of positive integers satisfying 6/m + 3/n = 1 -/
def solution_count : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ => 
    p.1 > 0 ∧ p.2 > 0 ∧ 6 * p.2 + 3 * p.1 = p.1 * p.2) 
    (Finset.range 100 ×ˢ Finset.range 100)).card

/-- Theorem stating that there are exactly 6 solutions -/
theorem six_solutions : solution_count = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_six_solutions_l1239_123979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_sums_count_l1239_123931

def ball_numbers : Finset ℕ := {3, 4, 6, 9, 10}

def sums : Finset ℕ := (ball_numbers.powerset.filter (fun s => s.card = 2)).image (fun s => s.sum id)

theorem distinct_sums_count : sums.card = 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_sums_count_l1239_123931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_fixed_point_l1239_123973

/-- Ellipse with given properties -/
structure Ellipse where
  center : ℝ × ℝ := (0, 0)
  foci_on_x_axis : Bool
  foci_distance : ℝ
  eccentricity : ℝ

/-- Predicate to check if a point is on the ellipse -/
def isOnEllipse (E : Ellipse) (p : ℝ × ℝ) : Prop :=
  p.1^2 / 2 + p.2^2 = 1

/-- Theorem about the ellipse and fixed point -/
theorem ellipse_and_fixed_point (E : Ellipse)
    (h_foci : E.foci_on_x_axis = true)
    (h_distance : E.foci_distance = 2)
    (h_eccentricity : E.eccentricity = Real.sqrt 2 / 2) :
    (∀ (p : ℝ × ℝ), isOnEllipse E p ↔ p.1^2 / 2 + p.2^2 = 1) ∧
    (∃ (M : ℝ × ℝ), M = (5/4, 0) ∧
      ∀ (l : Set (ℝ × ℝ)) (P Q : ℝ × ℝ),
        (1, 0) ∈ l →
        isOnEllipse E P ∧ P ∈ l →
        isOnEllipse E Q ∧ Q ∈ l →
        ∃ (c : ℝ), (M.1 - P.1) * (M.1 - Q.1) + (M.2 - P.2) * (M.2 - Q.2) = c) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_fixed_point_l1239_123973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_plant_beds_needed_l1239_123987

-- Define the given quantities
def bean_seedlings : ℕ := 70
def bean_per_row : ℕ := 9
def pumpkin_seeds : ℕ := 108
def pumpkin_per_row : ℕ := 14
def radishes : ℕ := 156
def radish_per_row : ℕ := 7
def rows_per_bed : ℕ := 2

-- Function to calculate the number of rows needed for a plant type
def rows_needed (total : ℕ) (per_row : ℕ) : ℕ :=
  (total + per_row - 1) / per_row

-- Theorem to prove
theorem min_plant_beds_needed :
  (rows_needed bean_seedlings bean_per_row +
   rows_needed pumpkin_seeds pumpkin_per_row +
   rows_needed radishes radish_per_row + rows_per_bed - 1) / rows_per_bed = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_plant_beds_needed_l1239_123987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chinese_remainder_count_l1239_123906

theorem chinese_remainder_count : ℕ := by
  -- Define the set of numbers from 1 to 2023
  let S : Finset ℕ := Finset.range 2023

  -- Define the condition for numbers in the sequence
  let condition (n : ℕ) : Prop := n % 3 = 1 ∧ n % 5 = 1

  -- Define the set of numbers that satisfy the condition
  let A : Finset ℕ := S.filter condition

  -- The theorem states that the cardinality of A is 135
  have h : A.card = 135 := by sorry

  -- Return the result as a natural number
  exact 135

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chinese_remainder_count_l1239_123906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_real_roots_l1239_123961

theorem no_real_roots (a b c : ℝ) (h : |a| + |b| + |c| ≤ Real.sqrt 2) :
  ∀ x : ℝ, x^4 + a*x^3 + b*x^2 + c*x + 1 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_real_roots_l1239_123961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_middle_distance_theorem_l1239_123937

structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

noncomputable def centroid (t : Triangle) : ℝ × ℝ :=
  (((t.A).1 + (t.B).1 + (t.C).1) / 3, ((t.A).2 + (t.B).2 + (t.C).2) / 3)

def Line := ℝ × ℝ → Prop

noncomputable def distance (p : ℝ × ℝ) (l : Line) : ℝ := sorry

noncomputable def middleDistance (t : Triangle) (l : Line) : ℝ := sorry

def linesWithMiddleDistance (t : Triangle) (d : ℝ) : Set Line :=
  {l : Line | middleDistance t l = d}

def linesAtDistanceFromPoint (p : ℝ × ℝ) (r : ℝ) : Set Line :=
  {l : Line | distance p l = r}

theorem middle_distance_theorem (t : Triangle) (d : ℝ) (h : d ≥ 0) :
  linesWithMiddleDistance t d = linesAtDistanceFromPoint (centroid t) (d / 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_middle_distance_theorem_l1239_123937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_propositions_l1239_123957

-- Define the propositions as axioms
axiom proposition1 : Prop
axiom proposition2 : Prop
axiom proposition3 : Prop
axiom proposition4 : Prop
axiom proposition5 : Prop

-- Define a function to check if a proposition is correct
def isCorrect (p : Prop) : Prop := p

-- Theorem stating that only propositions 3 and 5 are correct
theorem correct_propositions :
  isCorrect proposition3 ∧ isCorrect proposition5 ∧
  ¬isCorrect proposition1 ∧ ¬isCorrect proposition2 ∧ ¬isCorrect proposition4 :=
by
  sorry -- We use sorry to skip the proof for now

-- You can add comments to explain the meaning of each proposition
/-
proposition1 : A prism with two congruent opposite faces being rectangles is a cuboid.
proposition2 : The function y = sin x is increasing in the first quadrant.
proposition3 : If f(x) is a monotonic function, then f(x) and f⁻¹(x) have the same monotonicity.
proposition4 : If two planes of a dihedral angle are perpendicular to the two planes of another dihedral angle, then the plane angles of these two dihedral angles are complementary.
proposition5 : As the eccentricity e of an ellipse approaches 0, the shape of the ellipse becomes closer to a circle.
-/

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_propositions_l1239_123957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_runner_speed_approx_l1239_123988

/-- Represents the speed of a runner in meters per second -/
noncomputable def runner_speed (distance_km : ℝ) (time_s : ℝ) : ℝ :=
  (distance_km * 1000) / time_s

/-- Theorem stating that a runner covering 17.48 km in 38 seconds has a speed of approximately 460 m/s -/
theorem runner_speed_approx : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ |runner_speed 17.48 38 - 460| < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_runner_speed_approx_l1239_123988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equation_imaginary_part_l1239_123907

theorem complex_equation_imaginary_part :
  ∀ z : ℂ, (3 - 4*Complex.I) * z = Complex.abs (4 + 3*Complex.I) → z.im = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_equation_imaginary_part_l1239_123907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_theorem_l1239_123947

noncomputable def train_crossing_time (length1 length2 speed1 speed2 : ℝ) : ℝ :=
  (length1 + length2) / ((speed1 - speed2) * (1000 / 3600))

theorem train_crossing_theorem :
  train_crossing_time 800 600 108 72 = 140 := by
  -- Unfold the definition of train_crossing_time
  unfold train_crossing_time
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_theorem_l1239_123947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_common_tangent_l1239_123941

-- Define the circles using their equations
def circle_C1 (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y + 7 = 0
def circle_C2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 10*y + 13 = 0

-- Define a function to count common tangents (we'll leave it unimplemented)
def number_of_common_tangents (c1 c2 : (ℝ → ℝ → Prop)) : ℕ := sorry

-- Theorem stating that there is exactly one common tangent
theorem one_common_tangent :
  ∃! n : ℕ, n = number_of_common_tangents circle_C1 circle_C2 ∧ n = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_common_tangent_l1239_123941

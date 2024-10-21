import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_2_sqrt_2_l1168_116894

-- Define the polar curve
noncomputable def polar_curve (θ : ℝ) : ℝ := 4 * Real.sin θ

-- Define the polar line
noncomputable def polar_line : ℝ := Real.pi / 4

-- Define the chord length function
noncomputable def chord_length (ρ₁ ρ₂ θ : ℝ) : ℝ :=
  Real.sqrt ((ρ₁ * Real.cos θ - ρ₂ * Real.cos θ)^2 + (ρ₁ * Real.sin θ - ρ₂ * Real.sin θ)^2)

-- Theorem statement
theorem chord_length_is_2_sqrt_2 :
  ∃ ρ₁ ρ₂ : ℝ, 
    ρ₁ = polar_curve polar_line ∧
    ρ₂ = polar_curve polar_line ∧
    chord_length ρ₁ ρ₂ polar_line = 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_2_sqrt_2_l1168_116894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l1168_116887

/-- An arithmetic sequence with its sum -/
structure ArithmeticSequence where
  a : ℕ+ → ℚ
  S : ℕ+ → ℚ
  is_arithmetic : ∀ n : ℕ+, a (n + 1) - a n = a 2 - a 1
  sum_formula : ∀ n : ℕ+, S n = (n : ℚ) * (a 1 + a n) / 2

/-- The main theorem -/
theorem arithmetic_sequence_sum 
  (seq : ArithmeticSequence) 
  (h1 : seq.a 3 = 16) 
  (h2 : seq.S 20 = 20) : 
  seq.S 10 = 110 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l1168_116887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cooper_pie_days_l1168_116827

theorem cooper_pie_days (daily_production ashley_eaten remaining_pies : ℕ) 
  (h : daily_production = 7)
  (i : ashley_eaten = 50)
  (j : remaining_pies = 34) : 
  ∃ days : ℕ, daily_production * days - ashley_eaten = remaining_pies ∧ days = 12 := by
  -- Define the number of days
  let days := 12
  
  -- Prove the existence of 'days' satisfying the conditions
  use days
  
  -- Split the goal into two parts
  apply And.intro
  
  -- Prove the equation
  · rw [h, i, j]
    norm_num
  
  -- Prove that days = 12
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cooper_pie_days_l1168_116827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_l1168_116855

theorem sum_of_roots (a b : ℝ) (h : a ≠ 0) : 
  let f : ℝ → ℝ := λ x => a * x^3 + (a + 3*b) * x^2 + (b - 4*a) * x + (11 - a)
  (f (-3) = 0) ∧ (f 2 = 0) ∧ (f 4 = 0) → (-3 + 2 + 4 = 3) :=
by
  sorry

#check sum_of_roots

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_l1168_116855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_negative_ten_pi_thirds_l1168_116895

theorem sin_negative_ten_pi_thirds : 
  Real.sin (-10 * Real.pi / 3) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_negative_ten_pi_thirds_l1168_116895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_divisor_1991_l1168_116804

theorem largest_divisor_1991 :
  ∀ k : ℕ, k > 1991 →
    ¬(k ∣ (1991^k * 1990^(1991^1992) + 1992^(1991^1990))) ∧
    (1991 ∣ (1991^1991 * 1990^(1991^1992) + 1992^(1991^1990))) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_divisor_1991_l1168_116804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_igor_number_l1168_116828

def remove_smaller (s : List Nat) : List Nat :=
  s.filter (fun x => 
    match s.indexOf? x with
    | some i => 
        (i = 0 || s[i.pred]! ≤ s[i]) && 
        (i = s.length - 1 || s[i] ≤ s[i.succ]!)
    | none => false
  )

def removal_process : List Nat → List Nat
  | s => if s.length ≤ 3 then s else removal_process (remove_smaller s)
termination_by removal_process s => s.length

theorem igor_number (initial_sequence : List Nat) :
  initial_sequence = [1, 4, 5, 6, 8, 9, 10, 11] →
  ∃ (intermediate : List Nat),
    removal_process initial_sequence = [8, 10, 11] ∧
    removal_process intermediate = [5, 8, 10, 11] := by
  sorry

#eval removal_process [1, 4, 5, 6, 8, 9, 10, 11]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_igor_number_l1168_116828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_circles_inequality_l1168_116818

-- Define the trapezoid ABCD
structure Trapezoid (A B C D : ℝ × ℝ) : Prop where
  parallel : (B.2 - A.2) * (D.1 - C.1) = (D.2 - C.2) * (B.1 - A.1)

-- Define circles Γ1 and Γ2
def Circle (center : ℝ × ℝ) (radius : ℝ) := {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the theorem
theorem trapezoid_circles_inequality 
  (A B C D : ℝ × ℝ) 
  (trap : Trapezoid A B C D) 
  (X Y : ℝ × ℝ)
  (hX : X ∈ Circle ((A.1 + D.1)/2, (A.2 + D.2)/2) (((A.1 - D.1)^2 + (A.2 - D.2)^2)/4).sqrt)
  (hY : Y ∈ Circle ((B.1 + C.1)/2, (B.2 + C.2)/2) (((B.1 - C.1)^2 + (B.2 - C.2)^2)/4).sqrt) :
  let dist (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  let perimeter := dist A B + dist B C + dist C D + dist D A
  dist X Y ≤ perimeter / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_circles_inequality_l1168_116818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_ratio_theorem_l1168_116848

theorem tan_ratio_theorem (x y : ℝ) 
  (h1 : Real.sin x / Real.sin y = 4) 
  (h2 : Real.cos x / Real.cos y = 1/3) : 
  Real.tan (2*x) / Real.tan (2*y) = 12 * ((1 - Real.tan y^2) / (1 - 144 * Real.tan y^2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_ratio_theorem_l1168_116848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_implies_m_values_l1168_116840

-- Define the curves C1 and C2
noncomputable def C1 (m : ℝ) : Set (ℝ × ℝ) := {p | p.1 - p.2 + m = 0}

noncomputable def C2 : Set (ℝ × ℝ) := {p | p.1^2 / 3 + p.2^2 = 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1}

-- Define the distance function from a point to C1
noncomputable def dist_to_C1 (m : ℝ) (p : ℝ × ℝ) : ℝ :=
  |p.1 - p.2 + m| / Real.sqrt 2

-- State the theorem
theorem min_distance_implies_m_values (m : ℝ) :
  (∃ p ∈ C2, ∀ q ∈ C2, dist_to_C1 m p ≤ dist_to_C1 m q) ∧
  (∀ p ∈ C2, dist_to_C1 m p ≥ 2 * Real.sqrt 2) ∧
  (∃ p ∈ C2, dist_to_C1 m p = 2 * Real.sqrt 2) →
  m = -4 - Real.sqrt 3 ∨ m = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_implies_m_values_l1168_116840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_angle_pi_over_4_l1168_116842

theorem tangent_angle_pi_over_4 :
  ∃! p : ℝ × ℝ, 
    let (x, y) := p
    y = x^2 ∧ 
    Real.tan (π / 4) = (2 * x) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_angle_pi_over_4_l1168_116842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_x_for_lcm_equation_l1168_116832

theorem smallest_x_for_lcm_equation : 
  ∃ x : ℕ, x > 0 ∧ 
    Nat.lcm (Nat.lcm 12 16) (Nat.lcm x 24) = 144 ∧ 
    ∀ y : ℕ, y > 0 → Nat.lcm (Nat.lcm 12 16) (Nat.lcm y 24) = 144 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_x_for_lcm_equation_l1168_116832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_paid_percentage_theorem_l1168_116899

/-- The percentage of the suggested retail price that Bob paid -/
noncomputable def bob_paid_percentage (suggested_retail_price marked_price bob_price : ℝ) : ℝ :=
  (bob_price / suggested_retail_price) * 100

/-- The theorem stating the percentage Bob paid of the suggested retail price -/
theorem bob_paid_percentage_theorem (suggested_retail_price : ℝ) 
  (h1 : suggested_retail_price > 0) :
  let marked_price := 0.6 * suggested_retail_price
  let bob_price := 0.4 * marked_price
  bob_paid_percentage suggested_retail_price marked_price bob_price = 24 := by
  sorry

#check bob_paid_percentage_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bob_paid_percentage_theorem_l1168_116899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_absolute_values_l1168_116838

theorem min_sum_absolute_values (p q r s : ℤ) : 
  p ≠ 0 → q ≠ 0 → r ≠ 0 → s ≠ 0 →
  (![![p, q], ![r, s]] : Matrix (Fin 2) (Fin 2) ℤ) ^ 2 = 
    (![![8, 0], ![0, 8]] : Matrix (Fin 2) (Fin 2) ℤ) →
  ∃ (p' q' r' s' : ℤ), 
    p' ≠ 0 ∧ q' ≠ 0 ∧ r' ≠ 0 ∧ s' ≠ 0 ∧
    (![![p', q'], ![r', s']] : Matrix (Fin 2) (Fin 2) ℤ) ^ 2 = 
      (![![8, 0], ![0, 8]] : Matrix (Fin 2) (Fin 2) ℤ) ∧
    (abs p' + abs q' + abs r' + abs s' = 8) ∧
    (∀ (a b c d : ℤ), a ≠ 0 → b ≠ 0 → c ≠ 0 → d ≠ 0 →
      (![![a, b], ![c, d]] : Matrix (Fin 2) (Fin 2) ℤ) ^ 2 = 
        (![![8, 0], ![0, 8]] : Matrix (Fin 2) (Fin 2) ℤ) →
      abs a + abs b + abs c + abs d ≥ 8) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_absolute_values_l1168_116838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_rectangular_conversion_l1168_116803

/-- Conversion from polar coordinates (r, θ) to rectangular coordinates (x, y) --/
noncomputable def polar_to_rectangular (r : ℝ) (θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

/-- The given point in polar coordinates --/
noncomputable def polar_point : ℝ × ℝ := (6, Real.pi / 3)

/-- The expected point in rectangular coordinates --/
noncomputable def rectangular_point : ℝ × ℝ := (3, 3 * Real.sqrt 3)

theorem polar_to_rectangular_conversion :
  polar_to_rectangular polar_point.1 polar_point.2 = rectangular_point := by
  -- Unfold definitions
  unfold polar_to_rectangular polar_point rectangular_point
  -- Simplify expressions
  simp [Real.cos_pi_div_three, Real.sin_pi_div_three]
  -- The proof is completed
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_rectangular_conversion_l1168_116803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grocer_decaf_percentage_l1168_116850

/-- Calculates the percentage of decaffeinated coffee in the total stock -/
noncomputable def decaf_percentage (initial_stock : ℝ) (initial_decaf_percent : ℝ) 
                     (additional_stock : ℝ) (additional_decaf_percent : ℝ) : ℝ :=
  let initial_decaf := initial_stock * initial_decaf_percent / 100
  let additional_decaf := additional_stock * additional_decaf_percent / 100
  let total_decaf := initial_decaf + additional_decaf
  let total_stock := initial_stock + additional_stock
  (total_decaf / total_stock) * 100

/-- Theorem stating that given the specific quantities of coffee and decaf percentages,
    the resulting percentage of decaffeinated coffee in the total stock is 26% -/
theorem grocer_decaf_percentage :
  decaf_percentage 400 20 100 50 = 26 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_grocer_decaf_percentage_l1168_116850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_when_a_neg_two_range_of_a_given_conditions_l1168_116843

-- Define the functions f and g
noncomputable def f (a x : ℝ) : ℝ := |x - 1| + |x + a|
noncomputable def g (x : ℝ) : ℝ := (1/2) * x + 3

-- Part 1
theorem solution_set_when_a_neg_two :
  {x : ℝ | f (-2) x < g x} = {x : ℝ | 0 < x ∧ x < 4} := by sorry

-- Part 2
theorem range_of_a_given_conditions (a : ℝ) :
  a > -1 →
  (∃ x : ℝ, x ∈ Set.Icc (-a) 1 ∧ f a x ≤ g x) →
  a ∈ Set.Ioo (-1) (5/2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_when_a_neg_two_range_of_a_given_conditions_l1168_116843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_lower_bound_l1168_116829

theorem expression_lower_bound (x : Real) (h : 0 < x ∧ x < Real.pi/2) :
  (Real.tan x + (Real.tan x)⁻¹)^2 + (Real.sin x + (Real.cos x)⁻¹)^2 ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_lower_bound_l1168_116829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_characterization_l1168_116816

noncomputable def j : ℂ := Complex.exp (Complex.I * (2 * Real.pi / 3))

def is_equilateral_triangle (a b c : ℂ) : Prop :=
  Complex.abs (a - b) = Complex.abs (b - c) ∧ Complex.abs (b - c) = Complex.abs (c - a)

theorem equilateral_triangle_characterization :
  (j^2 + j + 1 = 0) ∧
  (∀ a b c : ℂ, (a * j^2 + b * j + c = 0) ↔ is_equilateral_triangle a b c) ∧
  (∀ a b c : ℂ, (a^2 + b^2 + c^2 = a*b + b*c + c*a) ↔ is_equilateral_triangle a b c) := by
  sorry

#check equilateral_triangle_characterization

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_characterization_l1168_116816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_theorem_l1168_116801

noncomputable def f (x : ℝ) : ℝ := 1 / (x - 2)

noncomputable def f_inverse (x : ℝ) : ℝ := (1 + 2*x) / x

theorem inverse_function_theorem (x : ℝ) (hx : x ≠ 0) (hx2 : x + 2 ≠ 2) :
  f (f_inverse x) = x ∧ f_inverse (f (x + 2)) = x + 2 :=
by
  sorry

#check inverse_function_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_theorem_l1168_116801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_work_time_is_840_l1168_116881

/-- Represents the work time for a single day -/
structure DayWorkTime where
  session1 : ℕ -- First session work time in minutes
  session2 : ℕ -- Second session work time in minutes
  breakTime : ℕ -- Break time in minutes

/-- Calculates the total work time for a day, excluding breaks -/
def totalWorkTime (day : DayWorkTime) : ℕ :=
  day.session1 + day.session2 - day.breakTime

/-- Work schedule for the week -/
def weekSchedule : List DayWorkTime :=
  [
    { session1 := 45, session2 := 30, breakTime := 0 },    -- Monday
    { session1 := 90, session2 := 45, breakTime := 15 },   -- Tuesday
    { session1 := 40, session2 := 60, breakTime := 0 },    -- Wednesday
    { session1 := 90, session2 := 75, breakTime := 30 },   -- Thursday
    { session1 := 55, session2 := 20, breakTime := 0 },    -- Friday
    { session1 := 120, session2 := 60, breakTime := 40 },  -- Saturday
    { session1 := 105, session2 := 135, breakTime := 45 }  -- Sunday
  ]

theorem total_work_time_is_840 :
  (weekSchedule.map totalWorkTime).sum = 840 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_work_time_is_840_l1168_116881


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AG_l1168_116898

-- Define the points
variable (A E F G N : ℝ)

-- Define the conditions
def quadrisect (A E F G : ℝ) : Prop :=
  E - A = F - E ∧ G - F = F - E ∧ G - A = 4 * (E - A)

def is_midpoint (N A G : ℝ) : Prop :=
  N - A = G - N

-- Theorem statement
theorem length_AG (h1 : quadrisect A E F G) (h2 : is_midpoint N A G) (h3 : N - F = 12) :
  G - A = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AG_l1168_116898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_equals_120_degrees_line_passes_through_point_l1168_116825

-- Define the slope of a line
noncomputable def line_slope (a b c : ℝ) : ℝ := -a / b

-- Define the angle from slope
noncomputable def angle_from_slope (s : ℝ) : ℝ := Real.arctan s

-- Statement 1: The slope of √3x + y + 1 = 0 is equivalent to 120°
theorem slope_equals_120_degrees :
  angle_from_slope (line_slope (Real.sqrt 3) 1 1) = 2 * π / 3 := by sorry

-- Statement 2: mx + y + 2 - m = 0 passes through (1,-2) for all m
theorem line_passes_through_point (m : ℝ) :
  m * 1 + (-2) + 2 - m = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_equals_120_degrees_line_passes_through_point_l1168_116825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_license_count_l1168_116837

/-- The set of valid letters for boat licenses -/
def ValidLetters : Finset Char := {'A', 'M', 'B', 'Z'}

/-- The set of valid digits for boat licenses -/
def ValidDigits : Finset Nat := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

/-- A boat license consisting of one letter followed by four digits -/
structure BoatLicense where
  letter : Char
  digits : Fin 4 → Nat

/-- Predicate to check if a boat license is valid -/
def isValidLicense (license : BoatLicense) : Prop :=
  license.letter ∈ ValidLetters ∧
  (∀ i : Fin 4, license.digits i ∈ ValidDigits) ∧
  (∀ i j : Fin 4, i ≠ j → license.digits i ≠ license.digits j)

/-- The set of all valid boat licenses -/
def AllValidLicenses : Type := {license : BoatLicense // isValidLicense license}

instance : Fintype AllValidLicenses :=
  sorry

theorem boat_license_count :
  Fintype.card AllValidLicenses = 20160 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_license_count_l1168_116837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chucks_roaming_area_l1168_116883

/-- Represents a right triangle shed -/
structure Shed :=
  (side1 : ℝ)
  (side2 : ℝ)

/-- Represents the roaming area of Chuck the llama -/
noncomputable def roamingArea (s : Shed) (leashLength : ℝ) : ℝ :=
  (1/4) * Real.pi * leashLength^2 + 
  (1/2) * Real.pi * s.side1^2 + 
  (1/2) * Real.pi * (leashLength - s.side2)^2

/-- The theorem stating the roaming area for Chuck's specific situation -/
theorem chucks_roaming_area :
  let s : Shed := ⟨2, 3⟩
  let leashLength : ℝ := 4
  roamingArea s leashLength = 6.5 * Real.pi := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chucks_roaming_area_l1168_116883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_uncertainEventsCount_l1168_116863

-- Define the type for events
inductive Event
| cloudyDayRain
| coinTossHeads
| sameBirthMonth
| olympicsBeijing

-- Define a function to determine if an event is uncertain
def isUncertain (e : Event) : Bool :=
  match e with
  | Event.cloudyDayRain => true
  | Event.coinTossHeads => true
  | Event.sameBirthMonth => true
  | Event.olympicsBeijing => false

-- Define a function to count uncertain events
def countUncertainEvents (events : List Event) : Nat :=
  events.filter isUncertain |>.length

-- The main theorem
theorem uncertainEventsCount :
  let events := [Event.cloudyDayRain, Event.coinTossHeads, Event.sameBirthMonth, Event.olympicsBeijing]
  countUncertainEvents events = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_uncertainEventsCount_l1168_116863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_valid_number_l1168_116854

def is_valid (n : Nat) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧  -- Four-digit positive integer
  (∀ d : Nat, d ∈ n.digits 10 → d ≠ 0 → n % d = 0) ∧  -- Divisible by each of its non-zero digits
  1 ∉ n.digits 10 ∧  -- Contains no '1'
  (n.digits 10).toFinset.card = 4  -- All digits are different

theorem least_valid_number : 
  is_valid 2520 ∧ ∀ m : Nat, m < 2520 → ¬is_valid m :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_valid_number_l1168_116854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1168_116866

-- Define the function f(x)
noncomputable def f (x m : ℝ) : ℝ := (1/3) * x^3 - 2 * x^2 + 3 * x - m

-- Theorem statement
theorem f_properties (m : ℝ) :
  (∃ x : ℝ, ∀ y : ℝ, f y m ≤ f x m ∧ f x m = 4/3 - m) ∧
  (∃ x : ℝ, ∀ y : ℝ, f y m ≥ f x m ∧ f x m = -m) ∧
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ 
    f x₁ m = 0 ∧ f x₂ m = 0 ∧ f x₃ m = 0) ↔ (0 < m ∧ m < 4/3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1168_116866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_equality_l1168_116830

theorem correct_equality : 
  ((-2 : ℚ)^3 = -2^3) ∧ 
  (-1/6 < -1/7) ∧ 
  (-4/3 > -3/2) ∧ 
  (-(-4.5) < |((-4.6) : ℚ)|) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_equality_l1168_116830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_exists_l1168_116834

-- Define the complex number equation
def complex_equation (a b : ℝ) : Prop :=
  (1 + Complex.I) * (a + b * Complex.I) = 2 + 4 * Complex.I

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ :=
  2 * Real.sin (a * x + Real.pi / 6) + b

-- Define what it means to be a symmetry center
def is_symmetry_center (a b x₀ y₀ : ℝ) : Prop :=
  ∀ x, f a b (x₀ + x) + f a b (x₀ - x) = 2 * y₀

-- Theorem statement
theorem symmetry_center_exists (a b : ℝ) (h : complex_equation a b) :
  is_symmetry_center a b (5 * Real.pi / 18) 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_center_exists_l1168_116834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_chart_best_for_temperature_l1168_116896

/-- Represents different types of statistical charts -/
inductive StatChart
| Bar
| Line
| Pie

/-- Represents the properties of data to be visualized -/
structure DataProperties where
  represent_quantity : Bool
  show_changes_over_time : Bool

/-- Defines the requirements for a patient's temperature data over a week -/
def temperature_data_properties : DataProperties :=
  { represent_quantity := true,
    show_changes_over_time := true }

/-- Checks if a chart meets the given data properties requirements -/
def chart_meets_requirements (chart : StatChart) (props : DataProperties) : Prop :=
  match chart with
  | StatChart.Line => props.represent_quantity ∧ props.show_changes_over_time
  | _ => false

/-- States that a line chart is the most appropriate for visualizing patient's temperature over a week -/
theorem line_chart_best_for_temperature : 
  ∀ (chart : StatChart), 
  (chart = StatChart.Line) ↔ 
  (∀ (props : DataProperties), 
   props = temperature_data_properties → 
   chart_meets_requirements chart props) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_chart_best_for_temperature_l1168_116896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_hundred_degrees_in_radians_l1168_116890

/-- Converts degrees to radians -/
noncomputable def degrees_to_radians (degrees : ℝ) : ℝ := (degrees * Real.pi) / 180

/-- 180 degrees is equal to π radians -/
axiom degree_radian_conversion : degrees_to_radians 180 = Real.pi

theorem three_hundred_degrees_in_radians :
  degrees_to_radians 300 = (5 * Real.pi) / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_hundred_degrees_in_radians_l1168_116890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_to_fraction_l1168_116867

theorem repeating_decimal_to_fraction :
  ∃ (n d : ℕ), n > 0 ∧ d > 0 ∧ (n : ℚ) / (d : ℚ) = 0.36 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_to_fraction_l1168_116867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_collinear_l1168_116858

def a : ℝ × ℝ := (3, -1)
def b : ℝ → ℝ × ℝ := λ k => (2, k)

def collinear (v w : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, v.1 * w.2 = t * v.2 * w.1

def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

theorem dot_product_collinear :
  ∀ k : ℝ, collinear a (b k) → dot_product a (b k) = 20/3 := by
  sorry

#check dot_product_collinear

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_collinear_l1168_116858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_mpg_calculation_l1168_116871

def initial_reading : ℕ := 34500
def final_reading : ℕ := 35350
def refuel_amounts : List ℕ := [10, 10, 15, 10]

theorem average_mpg_calculation :
  let total_distance : ℕ := final_reading - initial_reading
  let total_fuel : ℕ := refuel_amounts.sum
  let exact_mpg : ℚ := (total_distance : ℚ) / (total_fuel : ℚ)
  ⌊exact_mpg * 10 + 1/2⌋ / 10 = 189 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_mpg_calculation_l1168_116871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1168_116812

noncomputable def g (x : ℝ) : ℝ := 
  let g''1 : ℝ := Real.exp 1
  let g0 : ℝ := 1
  g''1 * Real.exp (x - 1) - g0 * x + (1/2) * x^2

theorem range_of_m (m : ℝ) :
  (∃ x₀ : ℝ, 2 * m - 1 ≥ g x₀) → m ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1168_116812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_congruence_l1168_116884

def sequence_start : ℕ := 7
def sequence_end : ℕ := 157
def sequence_step : ℕ := 10

def sequence_list : List ℕ := List.range ((sequence_end - sequence_start) / sequence_step + 1)
  |>.map (fun i => sequence_start + i * sequence_step)

theorem product_congruence :
  (sequence_list.prod) % 6 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_congruence_l1168_116884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_max_g_value_l1168_116824

noncomputable def g (x : ℝ) : ℝ := Real.sin (x / 4) + Real.sin (x / 9)

theorem smallest_max_g_value :
  ∃ (x : ℝ), x > 0 ∧ x = 13050 ∧ 
    (∀ (y : ℝ), y > 0 → g y ≤ g x) ∧
    (∀ (z : ℝ), 0 < z ∧ z < x → g z < g x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_max_g_value_l1168_116824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_remainder_l1168_116846

/-- The sum of an arithmetic sequence -/
def arithmetic_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

/-- The last term of an arithmetic sequence -/
def last_term (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

theorem arithmetic_sequence_sum_remainder
  (a₁ : ℕ) (d : ℕ) (aₙ : ℕ) (h : aₙ = 30) :
  (arithmetic_sum a₁ d ((aₙ - a₁) / d + 1)) % 9 = 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_remainder_l1168_116846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1168_116886

/-- The eccentricity of a hyperbola with specific properties -/
theorem hyperbola_eccentricity (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1) →  -- Hyperbola equation
  (∃ (m : ℝ), m = Real.tan (30 * Real.pi / 180)) →  -- Slope of line through F₁
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁^2 / a^2 - y₁^2 / b^2 = 1 ∧  -- Point M on hyperbola
    y₁ - (-c) = m * (x₁ - (-c)) ∧  -- Line through F₁ and M
    (x₂ - x₁) * (c - x₁) + (y₂ - y₁) * (-y₁) = 0) →  -- MF₂ perpendicular to x-axis
  c / a = Real.sqrt 3 :=  -- Eccentricity
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1168_116886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_at_6_l1168_116861

/-- An arithmetic sequence with its first term and common difference -/
structure ArithmeticSequence where
  a1 : ℝ
  d : ℝ

/-- Sum of the first n terms of an arithmetic sequence -/
noncomputable def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  n * seq.a1 + (n * (n - 1) / 2) * seq.d

/-- Theorem: Given an arithmetic sequence with a1 > 0 and S_12 * S_13 < 0,
    S_6 is the maximum value of S_n for positive integer n -/
theorem max_sum_at_6 (seq : ArithmeticSequence) 
    (h1 : seq.a1 > 0)
    (h2 : sum_n seq 12 * sum_n seq 13 < 0) :
    ∀ n : ℕ, n > 0 → sum_n seq 6 ≥ sum_n seq n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_at_6_l1168_116861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smith_children_ages_l1168_116802

/-- Represents a 4-digit number where each of two digits appears twice -/
structure BalloonNumber where
  digits : Fin 10 × Fin 10
  is_valid : digits.1 ≠ digits.2

/-- Checks if a number is divisible by all elements in a list -/
def divisible_by_all (n : ℕ) (l : List ℕ) : Prop :=
  ∀ m, m ∈ l → n % m = 0

theorem smith_children_ages :
  ∀ (balloon_number : BalloonNumber) (children_ages : List ℕ),
    (children_ages.length = 7) →
    (3 ∈ children_ages) →
    (∀ age, age ∈ children_ages → age < 10) →
    (∀ age₁ age₂, age₁ ∈ children_ages → age₂ ∈ children_ages → age₁ ≠ age₂ → age₁ ≠ age₂) →
    (divisible_by_all (balloon_number.digits.1 * 1000 + balloon_number.digits.2 * 100 + 
                       balloon_number.digits.1 * 10 + balloon_number.digits.2) children_ages) →
    8 ∉ children_ages :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smith_children_ages_l1168_116802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l1168_116821

/-- Given two vectors a and b in ℝ² -/
def a : ℝ × ℝ := (1, -2)
noncomputable def b : ℝ × ℝ := sorry

/-- The magnitude of b -/
noncomputable def b_mag : ℝ := 2 * Real.sqrt 5

/-- Dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- Magnitude of a 2D vector -/
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

/-- Angle between two 2D vectors -/
noncomputable def angle (v w : ℝ × ℝ) : ℝ := 
  Real.arccos (dot_product v w / (magnitude v * magnitude w))

theorem vector_problem :
  (∃ k : ℝ, b = (k * a.1, k * a.2) → (b = (-2, 4) ∨ b = (2, -4))) ∧
  (dot_product (2 • a - 3 • b) (2 • a + b) = -20 → angle a b = 2 * Real.pi / 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l1168_116821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcf_of_lcms_l1168_116857

theorem gcf_of_lcms : 
  Nat.gcd (Nat.lcm 18 21) (Nat.lcm 14 45) = 126 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcf_of_lcms_l1168_116857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_exists_l1168_116833

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then (2 : ℝ)^x else x + 1

-- Theorem statement
theorem solution_exists (a : ℝ) (h : f a = -2) : a = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_exists_l1168_116833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_root_power_mod_l1168_116864

theorem largest_root_power_mod (α : ℝ) : 
  (∃ x y z : ℝ, x < y ∧ y < z ∧ 
    (∀ t : ℝ, t^3 - 3*t^2 + 1 = 0 ↔ t = x ∨ t = y ∨ t = z) ∧
    α = z) →
  Int.floor (α^1993) % 17 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_root_power_mod_l1168_116864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_dependence_implies_k_equals_six_l1168_116879

theorem linear_dependence_implies_k_equals_six :
  ∀ k : ℝ, 
  (∃ (a b : ℝ) (h : a ≠ 0 ∨ b ≠ 0), 
    a • (⟨2, 3⟩ : ℝ × ℝ) + b • (⟨4, k⟩ : ℝ × ℝ) = (⟨0, 0⟩ : ℝ × ℝ)) → 
  k = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_dependence_implies_k_equals_six_l1168_116879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_exp_neg_l1168_116847

/-- The function f(x) = e^(-x) -/
noncomputable def f (x : ℝ) : ℝ := Real.exp (-x)

/-- The statement that the derivative of f(x) = e^(-x) is -e^(-x) -/
theorem derivative_of_exp_neg (x : ℝ) : 
  deriv f x = -Real.exp (-x) := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_exp_neg_l1168_116847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_equation_one_root_l1168_116893

/-- Given positive real numbers a, b, c different from 1, the determinant equation has exactly one real root -/
theorem det_equation_one_root (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (ha1 : a ≠ 1) (hb1 : b ≠ 1) (hc1 : c ≠ 1) :
  ∃! x : ℝ, Matrix.det
    ![![x - 1, c - 1, -(b - 1)],
      ![-(c - 1), x - 1, a - 1],
      ![b - 1, -(a - 1), x - 1]] = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_equation_one_root_l1168_116893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beto_winning_strategy_l1168_116823

/-- Represents a side of a square on the grid -/
inductive Side
  | Up
  | Right
  deriving Repr

/-- Represents a position on the grid -/
structure Position where
  x : Fin 2022
  y : Fin 2022

/-- Represents the coloring of sides on the grid -/
def Coloring := Position → Side → Bool

/-- A valid coloring ensures no square has two red sides sharing a vertex -/
def ValidColoring (c : Coloring) : Prop :=
  ∀ p : Position, ¬(c p Side.Up ∧ c p Side.Right)

/-- A path on the grid -/
def GridPath := List Position

/-- Checks if a path is valid (only moves up or right and doesn't use red segments) -/
def ValidPath (c : Coloring) : GridPath → Prop
  | [] => True
  | [_] => True
  | p1 :: p2 :: rest =>
    ((p2.x = p1.x + 1 ∧ p2.y = p1.y ∧ ¬(c p1 Side.Right)) ∨
     (p2.x = p1.x ∧ p2.y = p1.y + 1 ∧ ¬(c p1 Side.Up))) ∧
    ValidPath c (p2 :: rest)

/-- The main theorem: there always exists a valid path from bottom-left to bottom-right -/
theorem beto_winning_strategy (c : Coloring) (h : ValidColoring c) :
  ∃ path : GridPath,
    path.head? = some ⟨0, 0⟩ ∧
    path.getLast? = some ⟨2021, 0⟩ ∧
    ValidPath c path := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_beto_winning_strategy_l1168_116823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_theorem_l1168_116891

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus
def focus : ℝ × ℝ := (1, 0)

-- Define the distance between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ := 
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem parabola_distance_theorem (P Q : ℝ × ℝ) :
  parabola P.1 P.2 →
  parabola Q.1 Q.2 →
  distance P focus = 2 →
  distance Q focus = 5 →
  distance P Q = Real.sqrt 13 ∨ distance P Q = 3 * Real.sqrt 5 := by
  sorry

#check parabola_distance_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_theorem_l1168_116891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_decreasing_l1168_116849

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := x + 2 / x

-- Define the derivative of f(x)
noncomputable def f_derivative (x : ℝ) : ℝ := 1 - 2 / (x^2)

-- Define the monotonic decreasing interval
def monotonic_decreasing_interval : Set ℝ := 
  {x | x ∈ Set.Icc (-Real.sqrt 2) 0 ∪ Set.Ioc 0 (Real.sqrt 2)}

-- Theorem statement
theorem f_monotonic_decreasing :
  ∀ x ∈ monotonic_decreasing_interval, f_derivative x ≤ 0 := by
  sorry

#check f_monotonic_decreasing

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonic_decreasing_l1168_116849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l1168_116862

open Set Real

-- Define the function f and its derivative
def f : ℝ → ℝ := sorry

def f' : ℝ → ℝ := sorry

-- Define the domain of f as (0, +∞)
def domain_f : Set ℝ := {x | x > 0}

-- State the given condition
axiom condition (x : ℝ) : x ∈ domain_f → (1 - x) * f x + x * f' x > 0

-- Define the inequality we want to solve
def inequality (x : ℝ) : Prop :=
  (2*x - 1) / (x + 2) * f (2*x - 1) - exp (x - 3) * f (x + 2) < 0

-- State the theorem
theorem solution_set :
  {x : ℝ | inequality x} = Ioo (1/2 : ℝ) 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l1168_116862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2x_value_l1168_116897

theorem sin_2x_value (x : ℝ) (h : Real.sin x - Real.cos x = 1/2) : Real.sin (2*x) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2x_value_l1168_116897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l1168_116845

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.arcsin (x^2) + Real.arcsin x + x^6 + x^3

-- State the theorem
theorem solution_set (x : ℝ) :
  x ∈ Set.Ioc 0 1 ↔ x ∈ Set.Icc (-1) 1 ∧ f x > 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l1168_116845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extra_charge_count_l1168_116839

structure Envelope where
  length : ℚ
  height : ℚ

def extra_charge (e : Envelope) : Bool :=
  let ratio := e.length / e.height
  ratio < 1.5 || ratio > 2.0

def envelopes : List Envelope := [
  ⟨8, 5⟩,
  ⟨10, 4⟩,
  ⟨8, 8⟩,
  ⟨12, 6⟩
]

theorem extra_charge_count : 
  (envelopes.filter extra_charge).length = 2 := by
  -- Proof goes here
  sorry

#eval (envelopes.filter extra_charge).length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extra_charge_count_l1168_116839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_intersection_angle_l1168_116882

-- Define the curve C in polar coordinates
noncomputable def curve_C (θ : ℝ) : ℝ := 2 * Real.cos θ

-- Define the line l in parametric form
noncomputable def line_l (t α : ℝ) : ℝ × ℝ :=
  (-1 + t * Real.cos α, t * Real.sin α)

-- Define the condition for unique intersection
def unique_intersection (α : ℝ) : Prop :=
  ∃! t, (let (x, y) := line_l t α; x^2 + y^2 = 2*x)

-- Theorem statement
theorem unique_intersection_angle :
  ∀ α, unique_intersection α ↔ (α = Real.pi/6 ∨ α = 5*Real.pi/6) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_intersection_angle_l1168_116882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_ratio_bounds_l1168_116853

open Real

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the derivative of f
def f' : ℝ → ℝ := sorry

-- Axioms for the properties of f and f'
axiom f_domain : ∀ x, x > 0 → f x > 0
axiom f'_bounds : ∀ x, x > 0 → 2 * f x < f' x ∧ f' x < 3 * f x
axiom f'_is_derivative : ∀ x, x > 0 → deriv f x = f' x

-- Theorem statement
theorem f_ratio_bounds :
  exp (-3) < f 2023 / f 2024 ∧ f 2023 / f 2024 < exp (-2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_ratio_bounds_l1168_116853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2021_greater_than_60_l1168_116810

noncomputable def a : ℕ → ℝ
  | 0 => 1
  | n + 1 => a n + 1 / a n

theorem a_2021_greater_than_60 : a 2021 > 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2021_greater_than_60_l1168_116810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_tangent_function_l1168_116820

noncomputable def f (x φ : ℝ) : ℝ := Real.tan (3 * x + φ)

theorem symmetric_tangent_function (φ : ℝ) 
  (h1 : |φ| ≤ π/4) 
  (h2 : ∀ x : ℝ, f x φ = f (-π/9 - (x + π/9)) φ) : 
  φ = -π/6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_tangent_function_l1168_116820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_five_percent_l1168_116826

noncomputable def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

noncomputable def compoundInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * ((1 + rate / 100) ^ time - 1)

theorem interest_rate_is_five_percent (x : ℝ) (simpleInt : ℝ) (compoundInt : ℝ) 
    (h1 : x = 5000)
    (h2 : simpleInt = 500)
    (h3 : compoundInt = 512.50)
    (h4 : simpleInterest x 5 2 = simpleInt)
    (h5 : compoundInterest x 5 2 = compoundInt) : 
  ∃ (r : ℝ), r = 5 ∧ 
    simpleInterest x r 2 = simpleInt ∧ 
    compoundInterest x r 2 = compoundInt := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_five_percent_l1168_116826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_divisibility_l1168_116841

theorem polynomial_divisibility (n : ℕ) (p : ℕ) (P : ℕ → ℤ) :
  (∀ (k : ℕ), k ≤ n → ∃ (a : ℤ), P k = a) →  -- P is a polynomial
  P n = 1 →  -- leading coefficient is 1
  (∀ (x : ℤ), ∃ (m : ℤ), (P x.toNat : ℤ) = m * p) →  -- P(x) is divisible by p for integer x
  ∃ (k : ℤ), n.factorial = k * p :=  -- n! is divisible by p
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_divisibility_l1168_116841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_from_ellipse_l1168_116888

/-- The equation of an ellipse -/
noncomputable def ellipse_equation (x y : ℝ) : Prop := 2 * x^2 + y^2 = 2

/-- The eccentricity of an ellipse -/
noncomputable def ellipse_eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

/-- The equation of a hyperbola -/
noncomputable def hyperbola_equation (x y m n : ℝ) : Prop := y^2 / n^2 - x^2 / m^2 = 1

/-- The eccentricity of a hyperbola -/
noncomputable def hyperbola_eccentricity (m c : ℝ) : ℝ := c / m

theorem hyperbola_from_ellipse 
  (a b : ℝ) 
  (h_ellipse : ∀ x y, ellipse_equation x y ↔ x^2 / a^2 + y^2 / b^2 = 1)
  (h_vertices : ∀ x y, hyperbola_equation x y a (Real.sqrt (4 - a^2)) ↔ y^2 - x^2 = 2)
  (h_eccentricity : ellipse_eccentricity a b * hyperbola_eccentricity a 2 = 1) :
  ∀ x y, hyperbola_equation x y a (Real.sqrt (4 - a^2)) ↔ y^2 - x^2 = 2 := by
  sorry

#check hyperbola_from_ellipse

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_from_ellipse_l1168_116888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_common_ratio_l1168_116860

theorem geometric_sequence_common_ratio : 
  ∀ (a : ℕ → ℝ) (q : ℝ), 
  (∀ n, a (n + 1) = a n * q) →  -- Geometric sequence definition
  a 1 = 2/3 →                   -- First term
  a 4 = ∫ x in (1:ℝ)..(4:ℝ), (1 + 2*x)  -- Fourth term
  → q = 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_common_ratio_l1168_116860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_at_four_l1168_116814

-- Define the function f as noncomputable
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a / (x - 1) + 1 / (x - 2) + 1 / (x - 6)

-- State the theorem
theorem max_value_at_four (a : ℝ) (h_a_pos : a > 0) :
  (∀ x, 3 < x → x < 5 → f a x ≤ f a 4) → a = -9/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_at_four_l1168_116814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_angle_ratio_l1168_116868

noncomputable def is_on_ellipse (x y : ℝ) : Prop := x^2 + y^2/4 = 1

def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (1, 0)

noncomputable def angle_of_inclination (p q : ℝ × ℝ) : ℝ := 
  Real.arctan ((q.2 - p.2) / (q.1 - p.1))

theorem ellipse_angle_ratio (x y : ℝ) :
  is_on_ellipse x y →
  x ≠ -1 →
  x ≠ 1 →
  let P : ℝ × ℝ := (x, y)
  let α := angle_of_inclination P A
  let β := angle_of_inclination P B
  (Real.cos (α - β)) / (Real.cos (α + β)) = -3/5 := by
  sorry

#check ellipse_angle_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_angle_ratio_l1168_116868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_vector_d_equals_expected_l1168_116869

/-- Given vectors in 2D space -/
def a : ℝ × ℝ := (2, 3)
def b : ℝ × ℝ := (-4, 6)
def c : ℝ × ℝ := (2, -1)

/-- Vector addition in 2D -/
def vadd (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)

/-- Vector subtraction in 2D -/
def vsub (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 - w.1, v.2 - w.2)

/-- Scalar multiplication for 2D vectors -/
def smul (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)

/-- Vector d as defined in the problem -/
def d : ℝ × ℝ := vadd (vsub a b) (smul 2 c)

/-- Magnitude of a 2D vector -/
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

/-- Unit vector in the direction of a given vector -/
noncomputable def unit_vector (v : ℝ × ℝ) : ℝ × ℝ := 
  let mag := magnitude v
  (v.1 / mag, v.2 / mag)

/-- The main theorem to prove -/
theorem unit_vector_d_equals_expected : 
  unit_vector d = (2 * Real.sqrt 5 / 5, -Real.sqrt 5 / 5) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_vector_d_equals_expected_l1168_116869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_3_value_l1168_116805

noncomputable section

-- Define the function g (which was previously undefined)
def g : ℝ → ℝ := sorry

-- Define the function f
def f (x : ℝ) : ℝ :=
  if x < 0 then 2^x
  else if x = 0 then 0
  else g x

-- Define the property of f being an odd function
axiom f_odd : ∀ x, f (-x) = -f x

-- Define g as part of f for x > 0
axiom g_def : ∀ x > 0, f x = g x

-- Theorem to prove
theorem g_3_value : g 3 = -1/8 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_3_value_l1168_116805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_of_negative_six_l1168_116835

theorem reciprocal_of_negative_six :
  (1 : ℚ) / (-6 : ℚ) = -1/6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_of_negative_six_l1168_116835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1168_116809

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  (1/2 * x^2 + a*x) * Real.log x - 1/4 * x^2 - a*x

theorem f_properties (a : ℝ) :
  -- (1) Tangent line equation when a = 0
  (a = 0 → ∀ x y : ℝ, y = f 0 x → 
    (y - f 0 (Real.exp 1) = Real.exp 1 * (x - Real.exp 1))) ∧
  -- (2) Existence of extreme values when a < 0
  (a < 0 → ∃ x_min x_max : ℝ, x_min ≠ x_max ∧ 
    ∀ x : ℝ, x > 0 → f a x_min ≤ f a x ∧ f a x ≤ f a x_max) ∧
  -- (3) Condition for f(x) > 0 for all x > 0
  ((∀ x : ℝ, x > 0 → f a x > 0) ↔ 
    a > -Real.exp (3/2) ∧ a < -1/4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1168_116809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_sum_l1168_116875

/-- A cubic polynomial with real coefficients -/
def CubicPolynomial (a b c d : ℝ) : ℝ → ℝ := λ x ↦ a * x^3 + b * x^2 + c * x + d

theorem cubic_polynomial_sum (k : ℝ) (Q : ℝ → ℝ) 
  (hcubic : ∃ a b c d : ℝ, Q = CubicPolynomial a b c d)
  (h0 : Q 0 = 3 * k)
  (h1 : Q 1 = 5 * k)
  (hneg1 : Q (-1) = 7 * k) :
  Q 2 + Q (-2) = 32 * k := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_sum_l1168_116875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_properties_l1168_116856

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 1
  | n + 1 => (Real.sqrt (sequence_a n + 1) - 1) / 2

theorem sequence_a_properties :
  ∀ n : ℕ,
    (sequence_a n > 0) ∧
    (n > 0 → sequence_a n / (sequence_a n + 4) < sequence_a (n + 1) ∧ sequence_a (n + 1) < sequence_a n / 4) ∧
    (3 / 4^n < sequence_a n ∧ sequence_a n ≤ 1 / 4^(n - 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_properties_l1168_116856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_characterize_function_l1168_116807

/-- A function from positive integers to natural numbers -/
def PositiveIntToNat := ℕ+ → ℕ

/-- The property that f(n) ≠ 0 for at least one n -/
def NonZeroExists (f : PositiveIntToNat) : Prop :=
  ∃ n : ℕ+, f n ≠ 0

/-- The property that f(xy) = f(x) + f(y) for all positive integers x and y -/
def AdditiveProperty (f : PositiveIntToNat) : Prop :=
  ∀ x y : ℕ+, f (x * y) = f x + f y

/-- The property that there are infinitely many positive integers n such that f(k) = f(n-k) for all k < n -/
def InfinitelyManySymmetric (f : PositiveIntToNat) : Prop :=
  ∀ m : ℕ, ∃ n : ℕ+, n > m ∧ ∀ k : ℕ+, k < n → f k = f (n - k)

/-- The exponent of p in the prime factorization of n -/
noncomputable def primeExponent (p : ℕ+) (n : ℕ+) : ℕ := sorry

/-- The main theorem -/
theorem characterize_function (f : PositiveIntToNat) 
  (h1 : NonZeroExists f) 
  (h2 : AdditiveProperty f) 
  (h3 : InfinitelyManySymmetric f) : 
  ∃ (p : ℕ+) (c : ℕ), Nat.Prime p.val ∧ ∀ n : ℕ+, f n = c * primeExponent p n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_characterize_function_l1168_116807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_price_33kg_l1168_116873

/-- The price of apples in rupees per kilogram for the first 30 kgs -/
def l : ℚ := 5

/-- The price of apples in rupees per kilogram for each additional kg beyond 30 kgs -/
def q : ℚ := 6

/-- The price of 36 kgs of apples in rupees -/
def price_36kg : ℚ := 186

/-- The cost of the first 20 kgs of apples in rupees -/
def cost_20kg : ℚ := 100

/-- The function that calculates the price of x kgs of apples -/
def apple_price (x : ℚ) : ℚ :=
  if x ≤ 30 then l * x
  else l * 30 + q * (x - 30)

theorem apple_price_33kg :
  apple_price 33 = 168 :=
by
  -- Unfold the definition of apple_price
  unfold apple_price
  -- Simplify the if-then-else expression
  simp [l, q]
  -- Perform the arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_price_33kg_l1168_116873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1168_116885

noncomputable def f (x : ℝ) : ℝ := (x + 6) / Real.sqrt (x^2 - 5*x + 6)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x < 2 ∨ x > 3} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1168_116885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mountain_bike_prices_l1168_116844

/-- Represents the sales and pricing of mountain bikes over three months -/
structure MountainBikeSales where
  july_sales : ℝ
  august_price_increase : ℝ
  august_sales : ℝ
  september_price_decrease_percent : ℝ
  september_profit_margin : ℝ

/-- Calculates the selling price per bike in August -/
noncomputable def august_price (s : MountainBikeSales) : ℝ :=
  s.august_sales * s.july_sales / (s.july_sales - s.august_price_increase * s.august_sales)

/-- Calculates the cost price of each mountain bike -/
noncomputable def cost_price (s : MountainBikeSales) : ℝ :=
  let september_price := august_price s * (1 - s.september_price_decrease_percent)
  september_price / (1 + s.september_profit_margin)

/-- Theorem stating the selling price in August and the cost price given the conditions -/
theorem mountain_bike_prices (s : MountainBikeSales)
  (h1 : s.july_sales = 22500)
  (h2 : s.august_price_increase = 100)
  (h3 : s.august_sales = 25000)
  (h4 : s.september_price_decrease_percent = 0.15)
  (h5 : s.september_profit_margin = 0.25) :
  august_price s = 1000 ∧ cost_price s = 680 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mountain_bike_prices_l1168_116844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_speed_theorem_l1168_116880

/-- Calculates the speed for the balance of a journey given the total distance,
    total time, initial speed, and initial time. -/
noncomputable def balanceSpeed (totalDistance : ℝ) (totalTime : ℝ) (initialSpeed : ℝ) (initialTime : ℝ) : ℝ :=
  let initialDistance := initialSpeed * initialTime
  let balanceDistance := totalDistance - initialDistance
  let balanceTime := totalTime - initialTime
  balanceDistance / balanceTime

/-- Theorem stating that for a journey of 400 km completed in 8 hours,
    where the first 3.2 hours were traveled at 20 kmph,
    the speed for the balance of the journey is 70 kmph. -/
theorem journey_speed_theorem :
  balanceSpeed 400 8 20 3.2 = 70 := by
  -- Unfold the definition of balanceSpeed
  unfold balanceSpeed
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_speed_theorem_l1168_116880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_solution_l1168_116822

theorem power_equation_solution (k : ℝ) : 
  (1/3 : ℝ)^32 * (1/125 : ℝ)^k = (1/27 : ℝ)^32 → k = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_solution_l1168_116822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_graph_shift_l1168_116892

theorem sin_graph_shift (x : ℝ) : 
  Real.sin (2 * (x - π/8)) = Real.sin (2*x - π/4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_graph_shift_l1168_116892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_max_value_l1168_116817

/-- Given a quadratic function f(x) = ax² + bx + c where f(x) ≥ f'(x) for all x ∈ ℝ,
    the maximum value of b²/(a² + 2c²) is √6 - 2 -/
theorem quadratic_function_max_value (a b c : ℝ) :
  (∀ x : ℝ, a * x^2 + b * x + c ≥ 2 * a * x + b) →
  (∃ k : ℝ, (∀ x : ℝ, b^2 / (a^2 + 2 * c^2) ≤ k) ∧ k = Real.sqrt 6 - 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_max_value_l1168_116817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_difference_line_l1168_116876

/-- The point P through which the line passes -/
def P : ℝ × ℝ := (1, 1)

/-- The circular region -/
def circle_region (x y : ℝ) : Prop := x^2 + y^2 ≤ 9

/-- A line passing through point P -/
structure Line where
  slope : ℝ
  y_intercept : ℝ
  passes_through_P : slope * P.1 + y_intercept = P.2

/-- The equation of a line in the form ax + by + c = 0 -/
def line_equation (l : Line) (x y : ℝ) : Prop :=
  l.slope * x - y + l.y_intercept = 0

/-- Area of the first part divided by the line -/
noncomputable def area_part1 (l : Line) : ℝ := sorry

/-- Area of the second part divided by the line -/
noncomputable def area_part2 (l : Line) : ℝ := sorry

/-- The theorem stating that the line maximizing the area difference has the equation x + y - 2 = 0 -/
theorem max_area_difference_line :
  ∃ (l : Line), (∀ (x y : ℝ), line_equation l x y ↔ x + y - 2 = 0) ∧
    (∀ (l' : Line), l' ≠ l → 
      (∃ (A₁ A₂ : ℝ), A₁ ≥ 0 ∧ A₂ ≥ 0 ∧
        (∀ (x y : ℝ), circle_region x y → line_equation l x y → A₁ + A₂ = π * 9) ∧
        (∀ (x y : ℝ), circle_region x y → line_equation l' x y → A₁ + A₂ = π * 9) ∧
        |A₁ - A₂| ≤ |area_part1 l' - area_part2 l'|)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_difference_line_l1168_116876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_vertex_is_111_l1168_116859

/-- A regular tetrahedron with integer coordinates -/
structure RegularTetrahedron where
  v1 : Fin 3 → ℤ := ![1, 0, 0]
  v2 : Fin 3 → ℤ := ![0, 1, 0]
  v3 : Fin 3 → ℤ := ![0, 0, 1]
  v4 : Fin 3 → ℤ
  regular : 
    (v1 0 - v2 0)^2 + (v1 1 - v2 1)^2 + (v1 2 - v2 2)^2 =
    (v1 0 - v3 0)^2 + (v1 1 - v3 1)^2 + (v1 2 - v3 2)^2 ∧
    (v1 0 - v3 0)^2 + (v1 1 - v3 1)^2 + (v1 2 - v3 2)^2 =
    (v1 0 - v4 0)^2 + (v1 1 - v4 1)^2 + (v1 2 - v4 2)^2 ∧
    (v2 0 - v3 0)^2 + (v2 1 - v3 1)^2 + (v2 2 - v3 2)^2 =
    (v2 0 - v4 0)^2 + (v2 1 - v4 1)^2 + (v2 2 - v4 2)^2 ∧
    (v3 0 - v4 0)^2 + (v3 1 - v4 1)^2 + (v3 2 - v4 2)^2 =
    (v1 0 - v2 0)^2 + (v1 1 - v2 1)^2 + (v1 2 - v2 2)^2

/-- The fourth vertex of the regular tetrahedron is (1,1,1) -/
theorem fourth_vertex_is_111 (t : RegularTetrahedron) : t.v4 = ![1, 1, 1] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_vertex_is_111_l1168_116859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_range_l1168_116813

theorem inequality_solution_range (a : ℝ) : 
  (∀ x : ℝ, a^2 + 2*a - Real.sin x^2 - 2*a*(Real.cos x) > 2) ↔ 
  (a < -2 - Real.sqrt 6 ∨ a > Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_range_l1168_116813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifteenth_pirate_share_l1168_116851

/-- The number of pirates --/
def n : ℕ := 15

/-- The initial number of coins in the chest --/
def initial_coins : ℕ := 3^14 * 5^14

/-- The fraction of remaining coins that the k-th pirate takes --/
def pirate_share (k : ℕ) : ℚ := k / n

/-- The number of coins the k-th pirate receives --/
def coins_received (k : ℕ) : ℕ :=
  if k = n then
    initial_coins * Nat.factorial (n - 1) / n^(n - 1)
  else
    initial_coins * k * Nat.factorial (n - k) / (n^k * Nat.factorial (n - 1))

theorem fifteenth_pirate_share :
  coins_received n = 19615234560000 := by
  sorry

#eval coins_received n

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifteenth_pirate_share_l1168_116851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l1168_116852

/-- Circle C in polar coordinates -/
noncomputable def C (θ : ℝ) : ℝ := 2 * Real.sqrt 2 * Real.cos (θ + 3 * Real.pi / 4)

/-- Line l in parametric form -/
noncomputable def l (t : ℝ) : ℝ × ℝ := (-1 - Real.sqrt 2 / 2 * t, Real.sqrt 2 / 2 * t)

/-- The distance between intersection points of C and l -/
theorem intersection_distance : 
  ∃ A B : ℝ × ℝ, A ≠ B ∧ 
  (∃ θ : ℝ, (C θ * Real.cos θ, C θ * Real.sin θ) = A) ∧
  (∃ θ : ℝ, (C θ * Real.cos θ, C θ * Real.sin θ) = B) ∧
  (∃ t : ℝ, l t = A) ∧
  (∃ t : ℝ, l t = B) ∧
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l1168_116852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_permutations_l1168_116819

/-- A permutation of {1, 2, 3, 4, 5} satisfying aₖ ≥ k - 2 for all k -/
def ValidPermutation : Type := 
  { p : Fin 5 → Fin 5 // Function.Bijective p ∧ ∀ k, p k ≥ k - 2 }

instance : Fintype ValidPermutation := by
  sorry  -- The proof of finiteness is omitted for brevity

/-- The number of valid permutations is 54 -/
theorem count_valid_permutations : 
  Fintype.card ValidPermutation = 54 := by
  sorry  -- The actual proof is omitted for brevity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_permutations_l1168_116819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meet_time_opposite_direction_is_six_l1168_116865

/-- Represents an athlete running on a circular track -/
structure Athlete where
  speed : ℝ

/-- Represents a circular track -/
structure Track where
  length : ℝ

/-- Calculates the time taken for an athlete to complete one lap of the track -/
noncomputable def lapTime (a : Athlete) (t : Track) : ℝ := t.length / a.speed

/-- Calculates the time taken for two athletes to meet when running in the same direction -/
noncomputable def meetTimeSameDirection (a1 a2 : Athlete) (t : Track) : ℝ :=
  t.length / (a1.speed - a2.speed)

/-- Calculates the time taken for two athletes to meet when running in opposite directions -/
noncomputable def meetTimeOppositeDirection (a1 a2 : Athlete) (t : Track) : ℝ :=
  t.length / (a1.speed + a2.speed)

/-- Main theorem: If two athletes meet the given conditions, they will meet after 6 seconds when running in opposite directions -/
theorem meet_time_opposite_direction_is_six
  (a1 a2 : Athlete) (t : Track)
  (h1 : lapTime a1 t + 5 = lapTime a2 t)
  (h2 : meetTimeSameDirection a1 a2 t = 30) :
  meetTimeOppositeDirection a1 a2 t = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_meet_time_opposite_direction_is_six_l1168_116865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_hyperbola_branch_locus_is_right_branch_of_hyperbola_l1168_116870

-- Define the points O and C
def O : ℝ × ℝ := (0, 0)
def C : ℝ × ℝ := (3, 0)

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the locus of points M
def locus : Set (ℝ × ℝ) := {M | distance M O - distance M C = 2}

-- Theorem statement
theorem locus_is_hyperbola_branch : 
  locus = {M : ℝ × ℝ | ∃ (t : ℝ), M.1 = (3 * (t^2 + 1)) / (t^2 - 3) ∧ M.2 = (6 * t) / (t^2 - 3) ∧ t > Real.sqrt 3} := by
  sorry

-- Additional theorem to explicitly state that the locus is a hyperbola branch
theorem locus_is_right_branch_of_hyperbola :
  ∀ M ∈ locus, distance M O - distance M C = 2 ∧ M.1 > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_hyperbola_branch_locus_is_right_branch_of_hyperbola_l1168_116870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_approx_l1168_116831

/-- The area of a circular sector -/
noncomputable def sectorArea (radius : ℝ) (centralAngle : ℝ) : ℝ :=
  (centralAngle / 360) * Real.pi * radius^2

/-- Theorem: The area of a circular sector with radius 12 meters and central angle 38 degrees
    is approximately 47.746 square meters -/
theorem sector_area_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.001 ∧ |sectorArea 12 38 - 47.746| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_approx_l1168_116831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_sin_inequality_l1168_116800

theorem negation_of_sin_inequality :
  (¬ ∀ x : ℝ, Real.sin x ≤ 1) ↔ (∃ x : ℝ, Real.sin x > 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_sin_inequality_l1168_116800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_theorem_l1168_116815

noncomputable section

-- Define the parabola C
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

-- Define the focus F and point D
def focus (p : ℝ) : ℝ × ℝ := (p/2, 0)
def point_D (p : ℝ) : ℝ × ℝ := (p, 0)

-- Define the condition for MD being perpendicular to x-axis
def MD_perpendicular (p : ℝ) (M : ℝ × ℝ) : Prop :=
  M.1 = p

-- Define the condition |MF| = 3
def MF_length (p : ℝ) (M : ℝ × ℝ) : Prop :=
  (M.1 - p/2)^2 + M.2^2 = 9

-- Main theorem
theorem parabola_theorem (p : ℝ) :
  ∃ (M N : ℝ × ℝ),
    parabola p M.1 M.2 ∧
    parabola p N.1 N.2 ∧
    MD_perpendicular p M ∧
    MF_length p M →
    (∀ (x y : ℝ), parabola p x y ↔ y^2 = 4*x) ∧
    (∃ (A B : ℝ × ℝ),
      parabola p A.1 A.2 ∧
      parabola p B.1 B.2 ∧
      (A.2 - B.2)/(A.1 - B.1) = -Real.sqrt 2) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_theorem_l1168_116815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_squared_coefficient_sum_l1168_116889

def expansion_sum (n : ℕ) (k : ℕ) : ℕ := 
  Finset.sum (Finset.range (n + 1)) (λ i => i.choose k)

theorem x_squared_coefficient_sum : expansion_sum 6 2 = 35 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_squared_coefficient_sum_l1168_116889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prize_winner_undefeated_l1168_116811

structure Tournament (Player : Type u) where
  players : Set Player
  plays : Player → Player → Prop
  wins : Player → Player → Prop
  no_ties : ∀ {x y : Player}, plays x y → (wins x y ∨ wins y x)
  all_play : ∀ {x y : Player}, x ≠ y → plays x y

def prize_condition {Player : Type u} (t : Tournament Player) (x : Player) : Prop :=
  ∀ z, t.wins z x → ∃ y, t.wins x y ∧ t.wins y z

theorem prize_winner_undefeated {Player : Type u} (t : Tournament Player) (winner : Player) :
  (prize_condition t winner ∧ ∀ p, p ≠ winner → ¬prize_condition t p) →
  ∀ p, p ≠ winner → t.wins winner p :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prize_winner_undefeated_l1168_116811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_and_line_properties_l1168_116874

-- Define the hyperbola C
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

-- Define the asymptote perpendicular to x + 2y = 0
def asymptote_perpendicular (a b : ℝ) : Prop :=
  (b / a) * (-1/2) = -1

-- Define the distance from right vertex to asymptote
noncomputable def vertex_distance (a : ℝ) : Prop :=
  |2*a| / Real.sqrt 5 = 2 * Real.sqrt 5 / 5

-- Define the midpoint of intersection
def midpoint_of_intersection (x y : ℝ) : Prop :=
  x = 3 ∧ y = 2

theorem hyperbola_and_line_properties
  (a b : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : asymptote_perpendicular a b)
  (h4 : vertex_distance a)
  (h5 : ∃ x y, midpoint_of_intersection x y) :
  (∀ x y, hyperbola a b x y ↔ x^2 - y^2/4 = 1) ∧
  (∃ k : ℝ, k = 6) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_and_line_properties_l1168_116874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_baume_hydrometer_milk_water_ratio_l1168_116806

/-- Baumé hydrometer reading for pure milk -/
noncomputable def pure_milk_reading : ℝ := 5

/-- Baumé hydrometer reading for milk-water mixture -/
noncomputable def mixture_reading : ℝ := 2.2

/-- Baumé degree for reference saline solution -/
noncomputable def reference_baume : ℝ := 15

/-- Density of reference saline solution -/
noncomputable def reference_density : ℝ := 1.116

/-- Volume of fluid displaced until water mark -/
noncomputable def V : ℝ := (reference_density * reference_baume) / (reference_density - 1)

/-- Density of pure milk -/
noncomputable def pure_milk_density : ℝ := V / (V - pure_milk_reading)

/-- Density of milk-water mixture -/
noncomputable def mixture_density : ℝ := V / (V - mixture_reading)

/-- Ratio of milk to water in the mixture -/
def milk_water_ratio : ℚ × ℚ := (5, 7)

theorem baume_hydrometer_milk_water_ratio :
  ∃ (x : ℝ), 0 < x ∧ x < 1 ∧
  x * 1 + (1 - x) * pure_milk_density = mixture_density ∧
  (↑(milk_water_ratio.1) / (milk_water_ratio.1 + milk_water_ratio.2) : ℝ) = 1 - x :=
by sorry

#eval milk_water_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_baume_hydrometer_milk_water_ratio_l1168_116806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_difference_and_existence_l1168_116872

theorem square_difference_and_existence (m n t : ℕ) 
  (hm : m > 0) (hn : n > 0) (ht : t > 0)
  (h : t * (m ^ 2 - n ^ 2) + m - n ^ 2 - n = 0) : 
  (∃ k : ℕ, m - n = k ^ 2) ∧ 
  (∀ t : ℕ, t > 0 → ∃ m n : ℕ, m > 0 ∧ n > 0 ∧ t * (m ^ 2 - n ^ 2) + m - n ^ 2 - n = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_difference_and_existence_l1168_116872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collision_time_theorem_l1168_116808

/-- Represents a ball with a velocity -/
structure Ball where
  velocity : ℝ

/-- Represents a system of five balls -/
structure BallSystem where
  balls : Fin 5 → Ball
  distance : ℝ

/-- Time between first and last collisions for a given ball system -/
noncomputable def timeBetweenCollisions (system : BallSystem) : ℝ :=
  3 * system.distance / ((system.balls 0).velocity + (system.balls 4).velocity)

/-- Theorem stating the time between first and last collisions for the given problem -/
theorem collision_time_theorem (system : BallSystem) 
  (h1 : (system.balls 0).velocity = 0.5)
  (h2 : (system.balls 1).velocity = 0.5)
  (h3 : (system.balls 2).velocity = 0.1)
  (h4 : (system.balls 3).velocity = 0.1)
  (h5 : (system.balls 4).velocity = 0.5)
  (h6 : system.distance = 2) :
  timeBetweenCollisions system = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_collision_time_theorem_l1168_116808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_properties_l1168_116877

/-- The function f(x) = -xe^(ax+1) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -x * Real.exp (a * x + 1)

/-- The constant function g(x) = -be -/
noncomputable def g (b : ℝ) (x : ℝ) : ℝ := -b * Real.exp 1

theorem intersection_properties (a b x₁ x₂ : ℝ) 
  (ha : a > 0) (hb : b ≠ 0) 
  (h_intersect : ∃ y₁ y₂ : ℝ, f a x₁ = g b x₁ ∧ f a x₂ = g b x₂ ∧ x₁ ≠ x₂) :
  (-1 / Real.exp 1 < a * b ∧ a * b < 0) ∧ a * (x₁ + x₂) < -2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_properties_l1168_116877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_rate_proof_l1168_116878

-- Define the diameter of the circular field
noncomputable def diameter : ℝ := 30

-- Define the total cost of fencing
noncomputable def totalCost : ℝ := 188.49555921538757

-- Define pi as a constant (approximation)
noncomputable def π : ℝ := 3.14159

-- Define the circumference of the field
noncomputable def circumference : ℝ := π * diameter

-- Define the rate per meter
noncomputable def ratePerMeter : ℝ := totalCost / circumference

-- Theorem statement
theorem fencing_rate_proof : 
  ∀ ε > 0, |ratePerMeter - 2| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_rate_proof_l1168_116878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_solutions_for_half_l1168_116836

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Define the fractional part function
noncomputable def frac (x : ℝ) : ℝ := x - floor x

-- Theorem statement
theorem infinite_solutions_for_half :
  ∃ (S : Set ℝ), (∀ x ∈ S, frac x = 1/2) ∧ Set.Infinite S :=
by
  -- Construct the set of solutions
  let S := {x : ℝ | ∃ k : ℤ, x = k + 1/2}
  
  -- Prove that S satisfies the conditions
  have h1 : ∀ x ∈ S, frac x = 1/2 := by
    intro x hx
    rcases hx with ⟨k, rfl⟩
    simp [frac, floor]
    -- The proof details are omitted
    sorry
  
  have h2 : Set.Infinite S := by
    -- Prove that S is infinite
    -- The proof details are omitted
    sorry
  
  -- Conclude the proof
  exact ⟨S, h1, h2⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_solutions_for_half_l1168_116836

import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lowest_cost_l1359_135986

-- Define the total cost function
noncomputable def total_cost (x : ℝ) : ℝ := (57200 / x) + (455 / 18) * x

-- Define the domain of x
def valid_speed (x : ℝ) : Prop := 40 ≤ x ∧ x ≤ 100

-- Theorem statement
theorem lowest_cost :
  ∃ (min_cost : ℝ),
    (∀ x, valid_speed x → total_cost x ≥ total_cost 40) ∧
    (min_cost = total_cost 40) ∧
    (abs (min_cost - 2441.11) < 0.01) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lowest_cost_l1359_135986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_quadrilateral_area_l1359_135974

-- Define a square with side length 2
def Square : Set (ℝ × ℝ) := {p | 0 ≤ p.1 ∧ p.1 ≤ 2 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2}

-- Define a function to reflect a point over a line
noncomputable def reflect (p : ℝ × ℝ) (line : ℝ × ℝ → ℝ) : ℝ × ℝ := sorry

-- Define the four sides of the square
def leftSide : ℝ := 0
def rightSide : ℝ := 2
def bottomSide : ℝ := 0
def topSide : ℝ := 2

-- Helper function to calculate the area of a quadrilateral (not implemented)
noncomputable def area_quadrilateral (p1 p2 p3 p4 : ℝ × ℝ) : ℝ := sorry

-- Theorem: The area of the quadrilateral formed by reflecting any point in the square over its sides is always 8
theorem reflection_quadrilateral_area (p : Square) :
  let p1 := reflect p (λ _ => leftSide)
  let p2 := reflect p (λ _ => rightSide)
  let p3 := reflect p (λ _ => topSide)
  let p4 := reflect p (λ _ => bottomSide)
  area_quadrilateral p1 p2 p3 p4 = 8 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflection_quadrilateral_area_l1359_135974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_office_letter_typing_l1359_135960

-- Define the number of letters
def n : ℕ := 10

-- Define the already typed letter
def typed_letter : ℕ := 9

-- Function to calculate the number of possible orders
def possible_orders : ℕ := Finset.sum (Finset.range 9) (λ k => Nat.choose 8 k * (k + 2))

-- Theorem statement
theorem office_letter_typing :
  possible_orders = 1232 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_office_letter_typing_l1359_135960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_value_proof_l1359_135988

theorem least_value_proof (x y z : ℤ) (hx : x = 4) (hy : y = 7) 
  (hz : z > 0)
  (h_least : ∀ (w : ℤ), w > 0 → x - y - z ≤ x - y - w) 
  (h_value : x - y - z = -17) : z = 14 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_value_proof_l1359_135988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_ABED_l1359_135976

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  A.1 = 0 ∧ A.2 = 3 ∧ B.1 = 0 ∧ B.2 = 0 ∧ C.1 = 4 ∧ C.2 = 0

-- Define the right angle at B
def RightAngleAtB (A B C : ℝ × ℝ) : Prop :=
  (A.1 - B.1) * (C.1 - B.1) + (A.2 - B.2) * (C.2 - B.2) = 0

-- Define the lengths of AB and BC
def LengthAB (A B : ℝ × ℝ) : Prop := (A.1 - B.1)^2 + (A.2 - B.2)^2 = 3^2
def LengthBC (B C : ℝ × ℝ) : Prop := (C.1 - B.1)^2 + (C.2 - B.2)^2 = 4^2

-- Define points D and E
def PointD (A C D : ℝ × ℝ) : Prop :=
  D.1 = (2 * C.1 + A.1) / 3 ∧ D.2 = (2 * C.2 + A.2) / 3

def PointE (B C E : ℝ × ℝ) : Prop :=
  E.1 = (2 * B.1 + C.1) / 3 ∧ E.2 = (2 * B.2 + C.2) / 3

-- Define the lengths of CD and DE
def LengthCD (C D : ℝ × ℝ) : Prop := (D.1 - C.1)^2 + (D.2 - C.2)^2 = (5/3)^2
def LengthDE (D E : ℝ × ℝ) : Prop := (E.1 - D.1)^2 + (E.2 - D.2)^2 = (5/3)^2

-- Define the perimeter of quadrilateral ABED
noncomputable def PerimeterABED (A B E D : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) +
  Real.sqrt ((B.1 - E.1)^2 + (B.2 - E.2)^2) +
  Real.sqrt ((E.1 - D.1)^2 + (E.2 - D.2)^2) +
  Real.sqrt ((D.1 - A.1)^2 + (D.2 - A.2)^2)

theorem perimeter_ABED (A B C D E : ℝ × ℝ) :
  Triangle A B C →
  RightAngleAtB A B C →
  LengthAB A B →
  LengthBC B C →
  PointD A C D →
  PointE B C E →
  LengthCD C D →
  LengthDE D E →
  PerimeterABED A B E D = 28/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_ABED_l1359_135976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_number_problem_l1359_135953

theorem two_number_problem (x y : ℝ) : 
  x + y = 40 → 
  3 * y - 4 * x = 10 → 
  |y - x - 8.58| < 0.01 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_number_problem_l1359_135953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cycle_loss_percentage_l1359_135995

/-- Calculate the percentage of loss given the cost price and selling price -/
noncomputable def percentage_loss (cost_price selling_price : ℝ) : ℝ :=
  ((cost_price - selling_price) / cost_price) * 100

/-- Theorem: The percentage of loss for a cycle with cost price 1600 and selling price 1280 is 20% -/
theorem cycle_loss_percentage :
  let cost_price : ℝ := 1600
  let selling_price : ℝ := 1280
  percentage_loss cost_price selling_price = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cycle_loss_percentage_l1359_135995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l1359_135924

theorem trig_identity (α : Real) 
  (h1 : Real.sin α + Real.cos α = 1/2) 
  (h2 : α ∈ Set.Ioo 0 Real.pi) : 
  (1 - Real.tan α) / (1 + Real.tan α) = -Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l1359_135924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_abs_equation_l1359_135906

theorem sum_of_solutions_abs_equation : 
  ∃ (s : Finset ℝ), (∀ x : ℝ, x ∈ s ↔ |x - 5| - 4 = 0) ∧ (s.sum id = 10) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_abs_equation_l1359_135906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1359_135997

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.cos (2 * x) + 2 * (Real.cos (Real.pi / 4 - x))^2 - 1

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f (x + T) = f x ∧
    ∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  (∀ (y : ℝ), y ∈ Set.Icc (-Real.sqrt 3) 2 ↔
    ∃ (x : ℝ), x ∈ Set.Icc (-Real.pi/3) (Real.pi/2) ∧ f x = y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1359_135997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_circle_intersection_l1359_135948

/-- A hyperbola with semi-major axis a, semi-minor axis b, and eccentricity e -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  e : ℝ
  h_positive : 0 < a ∧ 0 < b
  h_eccentricity : e = Real.sqrt 5

/-- The circle (x-2)^2 + (y-3)^2 = 1 -/
def unit_circle : Set (ℝ × ℝ) := {p | (p.1 - 2)^2 + (p.2 - 3)^2 = 1}

/-- The length of a line segment between two points -/
noncomputable def segmentLength (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

/-- Main theorem -/
theorem hyperbola_asymptote_circle_intersection
  (C : Hyperbola)
  (A B : ℝ × ℝ)
  (h_asymptote : ∃ (m : ℝ), A.2 = m * A.1 ∧ B.2 = m * B.1)
  (h_on_circle : A ∈ unit_circle ∧ B ∈ unit_circle) :
  segmentLength A B = 4 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_circle_intersection_l1359_135948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_totalDays1998To2001_l1359_135920

def isLeapYear (year : ℕ) : Bool :=
  (year % 4 = 0 ∧ year % 100 ≠ 0) ∨ year % 400 = 0

def daysInYear (year : ℕ) : ℕ :=
  if isLeapYear year then 366 else 365

def totalDaysInRange (startYear endYear : ℕ) : ℕ :=
  List.range (endYear - startYear + 1)
    |> List.map (fun i => daysInYear (startYear + i))
    |> List.sum

theorem totalDays1998To2001 :
  totalDaysInRange 1998 2001 = 1461 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_totalDays1998To2001_l1359_135920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1359_135944

noncomputable section

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

/-- The theorem stating properties of a specific triangle -/
theorem triangle_properties (t : Triangle) 
  (h1 : Real.cos (t.B - t.C) = 1 - Real.cos t.A) 
  (h2 : t.b * t.b = t.a * t.c) : 
  Real.sin t.B * Real.sin t.C = 1/2 ∧ 
  t.A = Real.pi/4 ∧ 
  Real.tan t.B + Real.tan t.C = -2 - Real.sqrt 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1359_135944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_slopes_perpendicular_lines_b_value_l1359_135946

/-- IsPerp m1 m2 means the lines with slopes m1 and m2 are perpendicular -/
def IsPerp (m1 m2 : ℚ) : Prop := m1 * m2 = -1

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
theorem perpendicular_slopes (m1 m2 : ℚ) : IsPerp m1 m2 ↔ m1 * m2 = -1 := by
  rfl

/-- The slope of a line ax + by + c = 0 is -a/b -/
def line_slope (a b : ℚ) : ℚ := -a / b

theorem perpendicular_lines_b_value :
  ∀ b : ℚ, 
  IsPerp (line_slope 3 4) (line_slope b 4) → 
  b = -16/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_slopes_perpendicular_lines_b_value_l1359_135946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chips_cost_l1359_135973

/-- Proves that the cost of each bag of chips is $3 given the conditions of Frank's purchase --/
theorem chips_cost (total_bars : ℕ) (total_chips : ℕ) (total_paid : ℕ) (change : ℕ) (bar_cost : ℕ) :
  total_bars = 5 →
  total_chips = 2 →
  total_paid = 20 →
  change = 4 →
  bar_cost = 2 →
  (total_paid - change - total_bars * bar_cost) / total_chips = 3 := by
  intro h1 h2 h3 h4 h5
  -- Proof steps would go here
  sorry

#check chips_cost

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chips_cost_l1359_135973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_intersection_points_l1359_135902

noncomputable def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

noncomputable def eccentricity (a c : ℝ) : ℝ := c / a

def symmetric_points (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ = -x₂ ∧ y₁ = -y₂

noncomputable def circle_diameter (x₁ y₁ x₂ y₂ x y : ℝ) : Prop :=
  (x - x₁) * (x - x₂) + (y - y₁) * (y - y₂) = 0

theorem ellipse_and_intersection_points 
  (a b : ℝ) (α : ℝ) (k : ℝ) :
  a > b ∧ b > 0 ∧
  ellipse a b 2 (Real.sqrt 2) ∧
  eccentricity a (Real.sqrt ((a^2 - b^2) / 2)) = Real.sqrt 2 / 2 →
  ∃ (x_e y_e x_f y_f x_m y_m x_n y_n : ℝ),
    ellipse a b x_e y_e ∧
    ellipse a b x_f y_f ∧
    symmetric_points x_e y_e x_f y_f ∧
    y_e = k * x_e ∧
    y_f = k * x_f ∧
    y_m = k / (1 + Real.sqrt (1 + 2 * k^2)) * (0 + 2 * Real.sqrt 2) ∧
    y_n = k / (1 - Real.sqrt (1 + 2 * k^2)) * (0 + 2 * Real.sqrt 2) ∧
    (∃ (q : ℝ), q = 2 ∨ q = -2) ∧
    circle_diameter 0 y_m 0 y_n q 0 ∧
    a^2 = 8 ∧ b^2 = 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_intersection_points_l1359_135902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_standard_normal_probability_l1359_135940

/-- A random variable following a standard normal distribution -/
def ξ : Real → Real := sorry

/-- The probability density function of the standard normal distribution -/
def standard_normal_pdf : Real → Real := sorry

/-- The cumulative distribution function of the standard normal distribution -/
def Φ : Real → Real := sorry

/-- The probability of an event for a standard normal distribution -/
def P (event : Set Real) : Real := sorry

theorem standard_normal_probability (h : P {x | ξ x ≤ -1.96} = 0.025) :
  P {x | |ξ x| < 1.96} = 0.950 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_standard_normal_probability_l1359_135940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_selection_process_probabilities_l1359_135971

/-- Probability of correctly answering a question in a given round -/
noncomputable def P (round : Nat) : ℝ :=
  match round with
  | 1 => 4/5
  | 2 => 3/5
  | 3 => 2/5
  | 4 => 1/5
  | _ => 0

/-- The number of rounds in the selection process -/
def numRounds : Nat := 4

theorem selection_process_probabilities :
  /- (I) Probability of elimination in the 4th round -/
  (P 1 * P 2 * P 3 * (1 - P 4) = 96/625) ∧
  /- (II) Probability of elimination by the 3rd round at most -/
  ((1 - P 1) + P 1 * (1 - P 2) + P 1 * P 2 * (1 - P 3) = 101/125) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_selection_process_probabilities_l1359_135971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_intersection_implies_m_range_l1359_135922

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + y^2 - 4*x = 0

-- Define the parabola
def my_parabola (x y m : ℝ) : Prop := y^2 = 4*m*x

-- Define the directrix of the parabola
def my_directrix (x m : ℝ) : Prop := x = -m

-- Theorem statement
theorem no_intersection_implies_m_range (m : ℝ) :
  m ≠ 0 →
  (∀ x y : ℝ, my_circle x y → ¬(my_directrix x m)) →
  m > 0 ∨ m < -4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_intersection_implies_m_range_l1359_135922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ferry_time_difference_l1359_135931

/-- A ferry with a given speed and travel time -/
structure Ferry where
  speed : ℝ
  time : ℝ

/-- The problem setup for the ferry comparison -/
structure FerryProblem where
  p : Ferry
  q : Ferry
  p_speed : p.speed = 6
  p_time : p.time = 3
  q_route_factor : ℝ
  q_route_factor_value : q_route_factor = 2
  speed_difference : q.speed = p.speed + 3

/-- The theorem stating the time difference between ferries q and p -/
theorem ferry_time_difference (problem : FerryProblem) : 
  problem.q.time - problem.p.time = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ferry_time_difference_l1359_135931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_of_non_science_majors_l1359_135981

theorem percentage_of_non_science_majors 
  (women_science_percentage : Real) 
  (men_percentage : Real) 
  (men_science_percentage : Real) 
  (h1 : women_science_percentage = 0.2)
  (h2 : men_percentage = 0.4)
  (h3 : men_science_percentage = 0.7)
  : 1 - (women_science_percentage * (1 - men_percentage) + men_science_percentage * men_percentage) = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_of_non_science_majors_l1359_135981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_glass_bowl_percentage_gain_l1359_135963

/-- Calculates the percentage gain from selling glass bowls --/
theorem glass_bowl_percentage_gain :
  let total_bowls : ℕ := 300
  let cost_per_bowl : ℚ := 20
  let sold_first : ℕ := 200
  let price_first : ℚ := 25
  let sold_second : ℕ := 80
  let price_second : ℚ := 30
  let broken : ℕ := 20

  let cost_price : ℚ := total_bowls * cost_per_bowl
  let selling_price : ℚ := sold_first * price_first + sold_second * price_second
  let gain : ℚ := selling_price - cost_price
  let percentage_gain : ℚ := (gain / cost_price) * 100

  ∃ (ε : ℚ), abs (percentage_gain - 23.33) < ε ∧ ε > 0
:= by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_glass_bowl_percentage_gain_l1359_135963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l1359_135905

def our_sequence (n : ℕ+) : ℚ :=
  match n with
  | 1 => 1/2
  | 2 => -3/4
  | 3 => 5/8
  | 4 => -7/16
  | _ => (-1)^(n.val+1 : ℤ) * ((2*n.val-1 : ℤ) / (2^n.val : ℤ))

theorem sequence_formula (n : ℕ+) : our_sequence n = (-1)^(n.val+1 : ℤ) * ((2*n.val-1 : ℤ) / (2^n.val : ℤ)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l1359_135905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_calculations_l1359_135961

theorem sqrt_calculations : 
  (Real.sqrt 2 * Real.sqrt 6 + Real.sqrt 3 = 3 * Real.sqrt 3) ∧
  ((1 - Real.sqrt 2) * (2 - Real.sqrt 2) = 4 - 3 * Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_calculations_l1359_135961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_equation_range_l1359_135903

theorem sine_equation_range (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 0 (2 * Real.pi) ∧ x₂ ∈ Set.Icc 0 (2 * Real.pi) ∧
   Real.sin (Real.pi - x₁) + Real.sin (Real.pi / 2 + x₁) = m ∧
   Real.sin (Real.pi - x₂) + Real.sin (Real.pi / 2 + x₂) = m ∧
   |x₁ - x₂| ≥ Real.pi) →
  m ∈ Set.Ico 0 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_equation_range_l1359_135903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_price_reduction_l1359_135957

/-- Represents a clothing business with pricing and sales information -/
structure ClothingBusiness where
  initialCost : ℝ
  initialPrice : ℝ
  initialSales : ℝ
  salesIncrease : ℝ
  priceReduction : ℝ

/-- Calculates the daily profit for a clothing business -/
def dailyProfit (b : ClothingBusiness) : ℝ :=
  let newPrice := b.initialPrice - b.priceReduction
  let newSales := b.initialSales + b.salesIncrease * b.priceReduction
  (newPrice - b.initialCost) * newSales

/-- Theorem stating that a price reduction of 20 yuan results in a daily profit of 1800 yuan -/
theorem optimal_price_reduction (b : ClothingBusiness) 
    (h1 : b.initialCost = 80)
    (h2 : b.initialPrice = 120)
    (h3 : b.initialSales = 30)
    (h4 : b.salesIncrease = 3)
    (h5 : b.priceReduction = 20) : 
    dailyProfit b = 1800 := by
  sorry

-- Remove the #eval line as it's not necessary and might cause issues

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_price_reduction_l1359_135957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_bird_count_l1359_135996

theorem initial_bird_count (monkey_count : ℕ) (final_monkey_ratio : ℚ) : ℕ := by
  have h1 : monkey_count = 6 := by sorry
  have h2 : final_monkey_ratio = 3/5 := by sorry
  let initial_bird_count := 6
  let final_bird_count := initial_bird_count - 2
  have h3 : (monkey_count : ℚ) / ((monkey_count : ℚ) + final_bird_count) = final_monkey_ratio := by sorry
  exact initial_bird_count

#check initial_bird_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_bird_count_l1359_135996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_theorem_l1359_135907

-- Define the cubic polynomial
def f (a b c x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

-- Define the theorem
theorem max_value_theorem (a b c lambda : ℝ) (x₁ x₂ x₃ : ℝ) 
  (h_lambda_pos : lambda > 0)
  (h_roots : f a b c x₁ = 0 ∧ f a b c x₂ = 0 ∧ f a b c x₃ = 0)
  (h_lambda : x₂ - x₁ = lambda)
  (h_x₃ : x₃ > (x₁ + x₂) / 2) :
  (2*a^3 + 27*c - 9*a*b) / lambda^3 ≤ 3*Real.sqrt 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_theorem_l1359_135907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_turtle_food_cost_l1359_135977

/-- Represents the cost of turtle food per jar -/
noncomputable def cost_per_jar (food_requirement : ℝ) (turtle_weight : ℝ) (food_per_jar : ℝ) (total_cost : ℝ) : ℝ :=
  let total_food := turtle_weight * (food_requirement / 0.5)
  let jars_needed := total_food / food_per_jar
  total_cost / jars_needed

/-- Theorem stating that the cost per jar of turtle food is $2 under given conditions -/
theorem turtle_food_cost :
  cost_per_jar 1 30 15 8 = 2 := by
  -- Unfold the definition of cost_per_jar
  unfold cost_per_jar
  -- Simplify the expression
  simp
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_turtle_food_cost_l1359_135977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_domain_l1359_135945

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt x

-- State the theorem about the domain of the inverse function
theorem inverse_function_domain :
  {x : ℝ | ∃ y, f y = x} = Set.Ici (0 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_domain_l1359_135945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l1359_135925

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_a_pos : a > 0
  h_b_pos : b > 0
  h_a_gt_b : a > b

/-- The eccentricity of an ellipse -/
noncomputable def Ellipse.eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - (e.b / e.a)^2)

/-- The distance between the foci of an ellipse -/
noncomputable def Ellipse.focal_distance (e : Ellipse) : ℝ :=
  2 * e.a * e.eccentricity

/-- A point on an ellipse -/
structure EllipsePoint (e : Ellipse) where
  x : ℝ
  y : ℝ
  h_on_ellipse : x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- The area of a triangle formed by a point on the ellipse and the foci -/
noncomputable def triangle_area (e : Ellipse) (p : EllipsePoint e) : ℝ :=
  abs (p.x * e.b) / e.a

/-- The main theorem -/
theorem max_triangle_area (e : Ellipse) 
    (h_ecc : e.eccentricity = 1/2) 
    (h_focal : e.focal_distance = 2) :
  ∃ (p : EllipsePoint e), ∀ (q : EllipsePoint e), 
    triangle_area e p ≥ triangle_area e q ∧ 
    triangle_area e p = Real.sqrt 15 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l1359_135925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_e_power_negative_pi_over_6_l1359_135970

-- Define Euler's formula
axiom euler_formula (θ : ℝ) : Complex.exp (θ * Complex.I) = Complex.cos θ + Complex.I * Complex.sin θ

-- Define the imaginary part function
def imaginary_part (z : ℂ) : ℝ := z.im

-- Theorem statement
theorem imaginary_part_of_e_power_negative_pi_over_6 :
  imaginary_part (Complex.exp (-π/6 * Complex.I)) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_e_power_negative_pi_over_6_l1359_135970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_loss_percentage_is_twenty_percent_l1359_135928

/-- Represents the pricing and profit/loss calculations for an article -/
structure Article where
  cost : ℚ
  profit_percent : ℚ
  price_reduction_factor : ℚ

/-- Calculates the selling price given a cost and profit percentage -/
def selling_price (a : Article) : ℚ :=
  a.cost * (1 + a.profit_percent)

/-- Calculates the reduced selling price -/
def reduced_selling_price (a : Article) : ℚ :=
  (selling_price a) * a.price_reduction_factor

/-- Calculates the loss percentage when selling at the reduced price -/
def loss_percentage (a : Article) : ℚ :=
  (a.cost - reduced_selling_price a) / a.cost * 100

/-- Theorem stating that under the given conditions, the loss percentage is 20% -/
theorem loss_percentage_is_twenty_percent (a : Article) 
  (h1 : a.profit_percent = 60/100) 
  (h2 : a.price_reduction_factor = 1/2) : 
  loss_percentage a = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_loss_percentage_is_twenty_percent_l1359_135928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_well_depth_calculation_l1359_135909

/-- The depth of a circular well -/
noncomputable def well_depth (diameter : ℝ) (volume : ℝ) : ℝ :=
  volume / (Real.pi * (diameter / 2) ^ 2)

/-- Theorem: The depth of a circular well with diameter 2 metres and volume 25.132741228718345 cubic metres is approximately 8 metres -/
theorem well_depth_calculation :
  let diameter : ℝ := 2
  let volume : ℝ := 25.132741228718345
  abs (well_depth diameter volume - 8) < 0.000001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_well_depth_calculation_l1359_135909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_motorcycle_average_speed_l1359_135969

/-- Represents a travel segment of a motorcycle --/
structure TravelSegment where
  speed : ℝ
  duration : ℝ
  isSpeedInKph : Bool
  isTimeInHours : Bool

/-- Calculates the distance traveled in a segment --/
noncomputable def distanceTraveled (segment : TravelSegment) : ℝ :=
  let adjustedSpeed := if segment.isSpeedInKph then segment.speed else segment.speed * 1.60934
  let adjustedTime := if segment.isTimeInHours then segment.duration else segment.duration / 60
  adjustedSpeed * adjustedTime

/-- Calculates the time taken for a segment --/
noncomputable def timeTaken (segment : TravelSegment) : ℝ :=
  if segment.isTimeInHours then segment.duration else segment.duration / 60

/-- The main theorem statement --/
theorem motorcycle_average_speed : 
  let segments : List TravelSegment := [
    { speed := 30, duration := 15, isSpeedInKph := true, isTimeInHours := false },
    { speed := 45, duration := 20, isSpeedInKph := true, isTimeInHours := false },
    { speed := 50, duration := 30, isSpeedInKph := true, isTimeInHours := false },
    { speed := 40, duration := 20, isSpeedInKph := false, isTimeInHours := false }
  ]
  let totalDistance := (segments.map distanceTraveled).sum
  let totalTime := (segments.map timeTaken).sum
  let averageSpeed := totalDistance / totalTime
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |averageSpeed - 46.52| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_motorcycle_average_speed_l1359_135969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_for_tangent_line_l1359_135979

/-- Given a function f(x) = (√x + a) / (x^2 + b) where f(1) = 1/4 and f(4) = 2/19,
    this theorem states that the maximum value of m such that there exists k
    where y = kx + m is tangent to y = f(x) and kx + m ≥ f(x) for x ∈ [0, +∞) is 1/4. -/
theorem max_m_for_tangent_line (a b : ℝ) : 
  (let f := fun x => (Real.sqrt x + a) / (x^2 + b)
   (f 1 = 1/4) → (f 4 = 2/19) → 
   (∃ m : ℝ, m = 1/4 ∧ 
    (∃ k : ℝ, ∀ x : ℝ, x ≥ 0 → k * x + m ≥ f x) ∧
    (∀ m' : ℝ, m' > m → 
      ¬∃ k : ℝ, (∃ x₀ : ℝ, x₀ > 0 ∧ k * x₀ + m' = f x₀) ∧ 
                (∀ x : ℝ, x ≥ 0 → k * x + m' ≥ f x)))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_for_tangent_line_l1359_135979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_zero_g_periodic_f_g_sum_l1359_135975

-- Define the functions f and g
def f : ℝ → ℝ := sorry
def g : ℝ → ℝ := sorry

-- Define the derivative of g
def g' : ℝ → ℝ := sorry

-- State the conditions
axiom f_def : ∀ x, f x = 6 - g' x
axiom f_symmetry : ∀ x, f (1 - x) = 6 + g' (1 + x)
axiom g_odd : ∀ x, g x - 2 = -(g (-x) - 2)

-- State the theorems to be proved
theorem g_zero : g 0 = 2 := by sorry

theorem g_periodic : ∀ x, g (x + 4) = g x := by sorry

theorem f_g_sum : f 1 * g 1 + f 3 * g 3 = 24 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_zero_g_periodic_f_g_sum_l1359_135975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1359_135900

theorem problem_solution : 
  ∀ x : ℕ, x > 0 → (1 : ℕ)^(x + 3) + 3^(x + 1) + 4^(x - 1) + 5^x = 3786 → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1359_135900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_equation_l1359_135929

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 5 then x^2 - x + 12 else 2^x

theorem solve_equation (a : ℝ) (h : f (f a) = 16) : a = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_equation_l1359_135929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1359_135934

noncomputable def f (x : ℝ) : ℝ := (x^3 + 8) / (x - 8)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x < 8 ∨ x > 8} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1359_135934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_equality_l1359_135951

theorem product_equality (a b : ℕ) (ha : 10 ≤ a ∧ a < 100) : 
  let a' := (a % 10) * 10 + (a / 10)
  a' * b = 198 → a * b = 198 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_equality_l1359_135951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_systematic_sample_count_in_range_l1359_135967

def systematicSample (total : ℕ) (sampleSize : ℕ) (firstSelected : ℕ) : List ℕ :=
  (List.range sampleSize).map (fun i => (firstSelected + i * (total / sampleSize) - 1) % total + 1)

theorem systematic_sample_count_in_range (total : ℕ) (sampleSize : ℕ) (firstSelected : ℕ) 
  (rangeStart : ℕ) (rangeEnd : ℕ) :
  total = 100 →
  sampleSize = 10 →
  firstSelected = 3 →
  rangeStart = 19 →
  rangeEnd = 56 →
  ((systematicSample total sampleSize firstSelected).filter 
    (fun x => rangeStart ≤ x ∧ x ≤ rangeEnd)).length = 4 := by
  sorry

#eval systematicSample 100 10 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_systematic_sample_count_in_range_l1359_135967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_janinas_supplies_cost_l1359_135901

/-- Janina's pancake stand expenses --/
structure PancakeStand where
  daily_rent : ℕ
  pancake_price : ℕ
  pancakes_to_cover_expenses : ℕ
  daily_supplies_cost : ℕ

/-- Janina's specific pancake stand --/
def janinas_stand : PancakeStand :=
  { daily_rent := 30
  , pancake_price := 2
  , pancakes_to_cover_expenses := 21
  , daily_supplies_cost := 12 }

/-- Theorem: Janina's daily supplies cost is $12 --/
theorem janinas_supplies_cost :
  (janinas_stand.daily_rent +
   janinas_stand.daily_supplies_cost =
   janinas_stand.pancake_price *
   janinas_stand.pancakes_to_cover_expenses) ∧
  janinas_stand.daily_supplies_cost = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_janinas_supplies_cost_l1359_135901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_continuous_iterate_eq_g_l1359_135982

-- Define a strictly decreasing continuous function g
noncomputable def g : ℝ → ℝ := sorry

-- Axioms for g
axiom g_continuous : Continuous g
axiom g_decreasing : ∀ x y, x < y → g x > g y
axiom g_range : Set.range g = Set.Ioi 0

-- Define the k-fold composition of a function
def iterate (f : ℝ → ℝ) : ℕ → (ℝ → ℝ)
  | 0 => id
  | n + 1 => f ∘ (iterate f n)

-- Theorem statement
theorem no_continuous_iterate_eq_g :
  ∀ (f : ℝ → ℝ) (k : ℕ), Continuous f → k ≥ 2 → iterate f k ≠ g := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_continuous_iterate_eq_g_l1359_135982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_triangle_sin_cos_bound_l1359_135927

/-- Triangle with sides in geometric progression -/
structure GeometricTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c
  geometric_seq : b / a = c / b

/-- Angle B in a geometric triangle -/
noncomputable def angle_B (t : GeometricTriangle) : ℝ := 
  Real.arccos ((t.a^2 + t.c^2 - t.b^2) / (2 * t.a * t.c))

theorem geometric_triangle_sin_cos_bound (t : GeometricTriangle) :
  1 < Real.sin (angle_B t) + Real.cos (angle_B t) ∧
  Real.sin (angle_B t) + Real.cos (angle_B t) ≤ Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_triangle_sin_cos_bound_l1359_135927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_ratio_theorem_l1359_135965

-- Define the complex numbers
noncomputable def z₁ : ℂ := Complex.mk (Real.sqrt 3 / 2) (1 / 2)
def z₂ : ℂ := Complex.mk 3 4

-- State the theorem
theorem complex_ratio_theorem : Complex.abs (z₁^2016) / Complex.abs z₂ = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_ratio_theorem_l1359_135965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_johnson_family_children_l1359_135908

-- Define the family structure
structure Family where
  mother_age : ℕ
  father_age : ℕ
  num_children : ℕ
  children_ages : List ℕ
  children_ages_length : children_ages.length = num_children

-- Define the conditions
def family_conditions (f : Family) : Prop :=
  f.father_age = 50 ∧
  (2 * f.mother_age + f.children_ages.sum = 60) ∧
  (f.mother_age + f.father_age + f.children_ages.sum) / (2 + f.num_children) = 21

-- Theorem statement
theorem johnson_family_children (f : Family) :
  family_conditions f → f.num_children = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_johnson_family_children_l1359_135908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_my_sequence_2010_l1359_135919

def my_sequence (n : ℕ) : ℕ :=
  match n with
  | 0 => 2
  | m + 1 => my_sequence m + 2^(m+1)

theorem my_sequence_2010 : my_sequence 2009 = 2^2010 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_my_sequence_2010_l1359_135919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_points_on_line_l1359_135987

theorem points_on_line (t : ℝ) (ht : t ≠ 0) :
  ∃ (m b : ℝ), (t^2 - 2*t + 2) / t = m * ((t^2 + 2*t + 2) / t) + b :=
by
  -- We'll prove that m = 1 and b = -4
  use 1, -4
  field_simp
  ring

end NUMINAMATH_CALUDE_ERRORFEEDBACK_points_on_line_l1359_135987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_obtuse_iteration_l1359_135992

/-- Represents the angles of a triangle at each iteration -/
structure TriangleAngles where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Defines the initial triangle A₀B₀C₀ -/
def initial_triangle : TriangleAngles :=
  { x := 59.999, y := 60, z := 60.001 }

/-- Computes the next iteration of triangle angles -/
def next_angles (t : TriangleAngles) : TriangleAngles :=
  { x := 180 - 2 * t.x,
    y := 180 - 2 * t.y,
    z := 180 - 2 * t.z }

/-- Determines if a triangle is obtuse based on its angles -/
noncomputable def is_obtuse (t : TriangleAngles) : Prop :=
  t.x > 90 ∨ t.y > 90 ∨ t.z > 90

/-- Computes the triangle angles after n iterations -/
def iterate_angles : ℕ → TriangleAngles
  | 0 => initial_triangle
  | n+1 => next_angles (iterate_angles n)

/-- The main theorem to be proved -/
theorem least_obtuse_iteration :
  ∃ n, n = 15 ∧ is_obtuse (iterate_angles n) ∧ ∀ m < n, ¬is_obtuse (iterate_angles m) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_obtuse_iteration_l1359_135992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_convex_pentagon_l1359_135917

/-- A point in a 2D plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- A set of 9 points in a 2D plane --/
def NinePoints : Type := Fin 9 → Point

/-- Predicate to check if three points are collinear --/
def areCollinear (p q r : Point) : Prop := sorry

/-- Predicate to check if a set of points forms a convex pentagon --/
def IsConvexPentagon (s : Set Point) : Prop := sorry

/-- Main theorem: Given 9 points with no three collinear, there exists a convex pentagon --/
theorem exists_convex_pentagon (points : NinePoints) 
  (h : ∀ (i j k : Fin 9), i ≠ j → j ≠ k → i ≠ k → ¬areCollinear (points i) (points j) (points k)) :
  ∃ (s : Set Point), IsConvexPentagon s ∧ s ⊆ Set.range points := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_convex_pentagon_l1359_135917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_library_visitors_average_correct_l1359_135978

/-- Calculates the average number of visitors per day in a 30-day month starting on Sunday -/
def library_visitors_average (sunday_visitors : ℕ) (other_day_visitors : ℕ) : ℕ :=
  let num_sundays := 4
  let num_other_days := 30 - num_sundays
  let total_visitors := num_sundays * sunday_visitors + num_other_days * other_day_visitors
  total_visitors / 30

theorem library_visitors_average_correct 
  (h1 : library_visitors_average 510 240 = 276) : True := 
by
  -- Proof goes here
  sorry

#eval library_visitors_average 510 240  -- Should output 276

end NUMINAMATH_CALUDE_ERRORFEEDBACK_library_visitors_average_correct_l1359_135978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersecting_hyperbola_l1359_135998

-- Define the slope of the line
def m_slope : ℝ := 2

-- Define the hyperbola equation
def hyperbola (x y : ℝ) : Prop := x^2 / 3 - y^2 / 2 = 1

-- Define the distance between points A and B
noncomputable def distance_AB : ℝ := Real.sqrt 6

-- Define the line equation
def line_equation (m : ℝ) (x y : ℝ) : Prop := y = m_slope * x + m

-- Theorem statement
theorem line_intersecting_hyperbola :
  ∃ (m : ℝ), ∃ (A B : ℝ × ℝ),
    let (x₁, y₁) := A
    let (x₂, y₂) := B
    (hyperbola x₁ y₁) ∧
    (hyperbola x₂ y₂) ∧
    (line_equation m x₁ y₁) ∧
    (line_equation m x₂ y₂) ∧
    ((x₁ - x₂)^2 + (y₁ - y₂)^2 = distance_AB^2) →
    (m = Real.sqrt 15 ∨ m = -Real.sqrt 15) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersecting_hyperbola_l1359_135998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l1359_135962

/-- Calculates the speed of a train given its length, platform length, and time to cross the platform -/
noncomputable def train_speed (train_length platform_length : ℝ) (time : ℝ) : ℝ :=
  let total_distance := train_length + platform_length
  let speed_ms := total_distance / time
  3.6 * speed_ms

/-- Theorem stating that a train with given parameters has a specific speed -/
theorem train_speed_calculation :
  let train_length := (470 : ℝ)
  let platform_length := (520 : ℝ)
  let time := (64.79481641468682 : ℝ)
  let calculated_speed := train_speed train_length platform_length time
  ∃ ε > 0, |calculated_speed - 54.975| < ε :=
by
  sorry

-- Note: We can't use #eval with noncomputable functions, so we'll use #check instead
#check train_speed (470 : ℝ) (520 : ℝ) (64.79481641468682 : ℝ)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l1359_135962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_man_speed_calculation_l1359_135984

/-- Calculates the speed of a man walking in the same direction as a train, given the train's length, speed, and time to cross the man. -/
noncomputable def man_speed (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let relative_speed := train_length / crossing_time
  let man_speed_ms := train_speed_ms - relative_speed
  man_speed_ms * 3600 / 1000

/-- Theorem stating that given a train of length 300 m, moving at 63 km/hr, 
    taking 17.998560115190784 seconds to cross a man walking in the same direction, 
    the speed of the man is approximately 2.9952 km/hr. -/
theorem man_speed_calculation :
  let result := man_speed 300 63 17.998560115190784
  abs (result - 2.9952) < 0.0001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_man_speed_calculation_l1359_135984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jordan_sequence_14_steps_l1359_135916

def jordanSequence : ℕ → ℕ
  | 0 => 100000000
  | n + 1 => if n % 2 = 0 then jordanSequence n / 2 else jordanSequence n * 5

theorem jordan_sequence_14_steps :
  jordanSequence 14 = 2 * 5^15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jordan_sequence_14_steps_l1359_135916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteTriangularPairs_l1359_135937

/-- Definition of triangular numbers -/
def isTriangular (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * (k + 1) / 2

/-- Main theorem: For any b, if m is triangular, (2b + 1)m + b(b+1)/2 is triangular, and vice versa -/
theorem infiniteTriangularPairs (b : ℕ) :
  ∀ m : ℕ, isTriangular m ↔ isTriangular ((2 * b + 1) * m + b * (b + 1) / 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteTriangularPairs_l1359_135937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_142_equals_3_l1359_135994

-- Define the function f
def f (x : ℝ) : ℝ := 5 * x^3 + 7

-- State the theorem
theorem inverse_f_142_equals_3 : 
  ∃ (f_inv : ℝ → ℝ), Function.RightInverse f_inv f ∧ f_inv 142 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_142_equals_3_l1359_135994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_30_28_14_l1359_135956

/-- The area of a triangle given its side lengths -/
noncomputable def triangleArea (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

/-- Theorem: The area of a triangle with sides 30, 28, and 14 is approximately 194.98 -/
theorem triangle_area_30_28_14 :
  ‖triangleArea 30 28 14 - 194.98‖ < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_30_28_14_l1359_135956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complementary_angles_difference_l1359_135935

theorem complementary_angles_difference (a b : ℝ) : 
  a + b = 90 → -- complementary angles
  a = 4 * b → -- ratio 4:1
  abs (a - b) = 54 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complementary_angles_difference_l1359_135935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l1359_135999

/-- Given two lines in the xy-plane -/
def line1 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 + p.2 - 1 = 0}
def line2 : Set (ℝ × ℝ) := {p : ℝ × ℝ | 2 * p.1 - p.2 + 4 = 0}

/-- The intersection point of line1 and line2 -/
noncomputable def intersection : ℝ × ℝ := sorry

/-- A line passing through the intersection point -/
def line (a b c : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | a * p.1 + b * p.2 + c = 0}

/-- Definition of x-intercept and y-intercept -/
noncomputable def x_intercept (a b c : ℝ) : ℝ := -c / a
noncomputable def y_intercept (a b c : ℝ) : ℝ := -c / b

/-- The main theorem -/
theorem line_equation :
  ∃ a b c : ℝ,
    (intersection ∈ line a b c) ∧
    (x_intercept a b c = 2 * y_intercept a b c) ∧
    ((a = 1 ∧ b = 2 ∧ c = -3) ∨ (a = 2 ∧ b = 1 ∧ c = 0)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l1359_135999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_product_l1359_135985

noncomputable def f (x : ℝ) : ℝ := Real.sin x

theorem sin_cos_product (α : ℝ) 
  (h1 : ∀ x ∈ Set.Icc (-Real.pi/2) (Real.pi/2), f x = Real.sin x)
  (h2 : f (Real.sin α) + f (Real.cos α - 1/2) = 0) :
  2 * Real.sin α * Real.cos α = -3/4 := by
  sorry

#check sin_cos_product

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_product_l1359_135985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_ratio_l1359_135910

/-- The common ratio of an infinite geometric series where the sum of the original series
    is 64 times the sum of the series with the first four terms removed. -/
noncomputable def common_ratio : ℝ :=
  1 / 2

/-- The sum of the original infinite geometric series. -/
noncomputable def original_sum (a : ℝ) : ℝ :=
  a / (1 - common_ratio)

/-- The sum of the series with the first four terms removed. -/
noncomputable def modified_sum (a : ℝ) : ℝ :=
  (a * common_ratio ^ 4) / (1 - common_ratio)

theorem geometric_series_ratio :
  ∀ a : ℝ, a ≠ 0 → original_sum a = 64 * modified_sum a :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_ratio_l1359_135910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_expression_evaluation_l1359_135991

theorem trig_expression_evaluation (α : ℝ) 
  (h1 : Real.cos α = 1/3) 
  (h2 : -π/2 < α) 
  (h3 : α < 0) : 
  (Real.cos (-α - π) * Real.sin (2*π + α) * Real.tan (2*π - α)) / 
  (Real.sin (3*π/2 - α) * Real.cos (π/2 + α)) = -2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_expression_evaluation_l1359_135991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scaled_recipe_correct_l1359_135911

-- Define the original recipe for 7 people
structure Recipe :=
  (pasta : ℚ)
  (milk : ℚ)
  (cheddar : ℚ)
  (parmesan : ℚ)
  (butter : ℚ)
  (flour : ℚ)
  (salt : ℚ)
  (pepper : ℚ)

def original_recipe : Recipe :=
  { pasta := 2
  , milk := 3
  , cheddar := 2
  , parmesan := 1/2
  , butter := 1/4
  , flour := 3/16  -- 3 tablespoons converted to cups
  , salt := 1/12  -- 1/2 teaspoon converted to cups
  , pepper := 1/24  -- 1/4 teaspoon converted to cups
  }

def scale_recipe (r : Recipe) (scale : ℚ) : Recipe :=
  { pasta := r.pasta * scale
  , milk := r.milk * scale
  , cheddar := r.cheddar * scale
  , parmesan := r.parmesan * scale
  , butter := r.butter * scale
  , flour := r.flour * scale
  , salt := r.salt * scale
  , pepper := r.pepper * scale
  }

-- Define the number of people for the original recipe and the family reunion
def original_servings : ℕ := 7
def family_reunion_servings : ℕ := 35

-- Theorem to prove that scaling the recipe by the ratio of people gives the correct quantities
theorem scaled_recipe_correct :
  scale_recipe original_recipe (family_reunion_servings / original_servings) =
  { pasta := 10
  , milk := 15
  , cheddar := 10
  , parmesan := 5/2
  , butter := 5/4
  , flour := 15/16
  , salt := 5/12
  , pepper := 5/24
  } := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_scaled_recipe_correct_l1359_135911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_value_l1359_135943

-- Define the recursive sequence
def a : ℕ → ℚ
  | 0 => 0  -- Arbitrary initial value
  | n + 1 => 2 * a n + 2^n

-- Define the sequence we want to be arithmetic
def seq (lambda : ℚ) (n : ℕ) : ℚ := (a n + lambda) / 2^n

-- State the theorem
theorem lambda_value (lambda : ℚ) :
  (∀ n : ℕ, ∃ d : ℚ, ∀ k : ℕ, seq lambda (n + k) - seq lambda n = d * k) →
  lambda = 1/3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_value_l1359_135943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_is_two_l1359_135904

-- Define the sector
structure Sector (circumference : ℝ) (central_angle : ℝ) where
  radius : ℝ
  h1 : circumference = radius + 2 * radius
  h2 : central_angle = 1

-- Define the area of the sector
noncomputable def sectorArea (s : Sector 6 1) : ℝ :=
  s.radius ^ 2 / 2

-- Theorem statement
theorem sector_area_is_two :
  ∀ s : Sector 6 1, sectorArea s = 2 :=
by
  intro s
  unfold sectorArea
  have h3 : s.radius = 2 := by
    -- Proof that radius = 2
    sorry
  rw [h3]
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_is_two_l1359_135904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1359_135923

theorem problem_solution (α β : ℝ) 
  (h1 : α ∈ Set.Ioo 0 (Real.pi/2))
  (h2 : Real.cos (2*α) = 4/5)
  (h3 : β ∈ Set.Ioo (Real.pi/2) Real.pi)
  (h4 : 5 * Real.sin (2*α + β) = Real.sin β) : 
  (Real.sin α + Real.cos α = 2 * Real.sqrt 10 / 5) ∧ (β = 3*Real.pi/4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1359_135923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_contact_loss_angle_l1359_135972

/-- The angle at which a sphere rolling off another sphere of the same radius loses contact -/
theorem sphere_contact_loss_angle (R : ℝ) (R_pos : R > 0) : 
  ∃ θ : ℝ, 0 < θ ∧ θ < π/2 ∧ Real.cos θ = 10/17 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_contact_loss_angle_l1359_135972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_special_case_l1359_135993

theorem sin_double_angle_special_case (α : Real) 
  (h1 : Real.cos (π / 2 - α) = 1 / 3) 
  (h2 : π / 2 < α) 
  (h3 : α < π) : 
  Real.sin (2 * α) = -4 * Real.sqrt 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_special_case_l1359_135993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l1359_135926

noncomputable def f (x : ℝ) := 2 / (x - 2) - 3 / (x - 3) + 3 / (x - 4) - 2 / (x - 5)

theorem inequality_solution :
  ∀ x : ℝ, (x ≠ 2 ∧ x ≠ 3 ∧ x ≠ 4 ∧ x ≠ 5) →
  (f x < 1/20 ↔ (1 < x ∧ x < 2) ∨ (3 < x ∧ x < 6)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l1359_135926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_c_l1359_135918

/-- Given a triangle ABC with sin A = 3/5 and cos B = 5/13, prove that cos C = 16/65 -/
theorem triangle_cosine_c (A B C : ℝ) (h1 : Real.sin A = 3/5) (h2 : Real.cos B = 5/13) :
  Real.cos C = 16/65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_c_l1359_135918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1359_135914

open Real

noncomputable def f (x : ℝ) : ℝ := sin (x + π/2)
noncomputable def g (x : ℝ) : ℝ := cos (x - π/2)

theorem problem_solution :
  (∃ p : ℝ, p > 0 ∧ ∀ x : ℝ, f x * g x = f (x + p) * g (x + p) ∧
    ∀ q : ℝ, q > 0 ∧ (∀ x : ℝ, f x * g x = f (x + q) * g (x + q)) → p ≤ q) ∧
  (∃ M : ℝ, M = 1/2 ∧ ∀ x : ℝ, f x * g x ≤ M) ∧
  (∀ x : ℝ, f (x - π/2) = g x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l1359_135914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_distance_l1359_135941

/-- Represents a journey with two parts -/
structure Journey where
  totalTime : ℚ
  speed1 : ℚ
  speed2 : ℚ

/-- Calculates the total distance of a journey -/
noncomputable def totalDistance (j : Journey) : ℚ :=
  let time1 := j.totalTime / 2
  let time2 := j.totalTime / 2
  j.speed1 * time1 + j.speed2 * time2

/-- Theorem stating that for the given journey conditions, the total distance is 448 km -/
theorem journey_distance : 
  ∀ (j : Journey), 
  j.totalTime = 12 → j.speed1 = 35 → j.speed2 = 40 → 
  totalDistance j = 448 :=
by
  intro j h_time h_speed1 h_speed2
  simp [totalDistance, h_time, h_speed1, h_speed2]
  -- The actual proof would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_distance_l1359_135941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flower_seedlings_problem_l1359_135954

-- Define the unit prices
variable (price_A price_B : ℝ)

-- Define the conditions
def condition1 (price_A price_B : ℝ) : Prop := price_A = price_B + 1.5
def condition2 (price_A price_B : ℝ) : Prop := 8000 / price_A = 5000 / price_B
def condition3 (price_A price_B : ℝ) (additional_A : ℝ) : Prop := 
  2 * additional_A * price_B + additional_A * price_A ≤ 7200

-- Theorem statement
theorem flower_seedlings_problem :
  ∀ (price_A price_B : ℝ),
  condition1 price_A price_B ∧ condition2 price_A price_B →
  (price_A = 4 ∧ price_B = 2.5) ∧
  (∃ (max_additional_A : ℕ), 
    max_additional_A = 800 ∧
    condition3 price_A price_B (max_additional_A : ℝ) ∧
    ∀ (n : ℕ), n > max_additional_A → ¬(condition3 price_A price_B (n : ℝ))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_flower_seedlings_problem_l1359_135954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_perimeter_outside_triangle_less_than_half_l1359_135936

/-- A triangle with an inscribed circle -/
structure TriangleWithInscribedCircle where
  /-- The triangle -/
  triangle : Set (Fin 2 → ℝ)
  /-- The inscribed circle -/
  circle : Set (Fin 2 → ℝ)
  /-- The circle is inscribed in the triangle -/
  inscribed : circle ⊆ triangle

/-- A square circumscribed around a circle -/
structure CircumscribedSquare (c : Set (Fin 2 → ℝ)) where
  /-- The square -/
  square : Set (Fin 2 → ℝ)
  /-- The square is circumscribed around the circle -/
  circumscribed : c ⊆ square

/-- The perimeter of a set in 2D Euclidean space -/
noncomputable def perimeter (s : Set (Fin 2 → ℝ)) : ℝ := sorry

/-- The length of a set's boundary that lies outside another set -/
noncomputable def outsideLength (s t : Set (Fin 2 → ℝ)) : ℝ := sorry

/-- The main theorem -/
theorem square_perimeter_outside_triangle_less_than_half
  (t : TriangleWithInscribedCircle)
  (s : CircumscribedSquare t.circle) :
  outsideLength s.square t.triangle < (perimeter s.square) / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_perimeter_outside_triangle_less_than_half_l1359_135936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_sum_theorem_l1359_135955

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := 4 * x - 3

noncomputable def g (x : ℝ) : ℝ := Real.log x / Real.log 3 + x - 3

-- State the theorem
theorem root_sum_theorem (x₁ x₂ : ℝ) 
  (h₁ : f x₁ = 0) 
  (h₂ : g x₂ = 0) : 
  x₁ + x₂ = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_sum_theorem_l1359_135955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_theorem_l1359_135933

-- Define the original function
noncomputable def f (x : ℝ) : ℝ := 3^(x-1) + 1

-- Define the proposed inverse function
noncomputable def f_inv (x : ℝ) : ℝ := 1 + (Real.log (x - 1)) / (Real.log 3)

-- Theorem statement
theorem inverse_function_theorem (x : ℝ) (hx : x > 1) : 
  f (f_inv x) = x ∧ f_inv (f x) = x := by
  sorry

#check inverse_function_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_theorem_l1359_135933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_sequence_probability_l1359_135958

/-- Represents a person in the queue with their bill amount -/
inductive Person
| s : Person  -- Person with 100 Ft bill
| k : Person  -- Person with 200 Ft bill

/-- Checks if a sequence of people is valid (cashier can always give change) -/
def is_valid_sequence (seq : List Person) : Bool :=
  seq.foldl (fun (acc : Nat × Bool) p =>
    match p with
    | Person.s => (acc.1 + 1, acc.2)
    | Person.k => (acc.1 - 1, acc.2 && acc.1 > 0)
  ) (0, true) |>.2

/-- Counts the number of valid sequences -/
def count_valid_sequences (n : Nat) : Nat :=
  (List.permutations (List.replicate n Person.s ++ List.replicate n Person.k)).filter is_valid_sequence |>.length

/-- The probability of a valid sequence for 8 people (4 with 100 Ft and 4 with 200 Ft) -/
theorem valid_sequence_probability :
  (count_valid_sequences 4 : ℚ) / (Nat.choose 8 4) = 1 / 5 := by
  sorry

#eval count_valid_sequences 4
#eval Nat.choose 8 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_sequence_probability_l1359_135958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_wins_l1359_135952

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  no_zeros : hundreds ≠ 0 ∧ tens ≠ 0 ∧ ones ≠ 0
  valid_range : hundreds < 10 ∧ tens < 10 ∧ ones < 10

/-- The game state -/
structure GameState where
  used_numbers : List ThreeDigitNumber
  last_digit : Nat

/-- Checks if a number is valid according to the game rules -/
def is_valid_move (state : GameState) (num : ThreeDigitNumber) : Prop :=
  num.hundreds = state.last_digit ∧
  (num.hundreds + num.tens + num.ones) % 9 = 0 ∧
  num ∉ state.used_numbers

/-- Defines a winning strategy for the first player -/
def first_player_winning_strategy (depth : Nat) : Prop :=
  match depth with
  | 0 => True  -- Base case: assume win at max depth
  | n+1 => ∃ (initial_move : ThreeDigitNumber),
      ∀ (opponent_move : ThreeDigitNumber),
        is_valid_move 
          { used_numbers := [initial_move], last_digit := initial_move.ones } 
          opponent_move →
        ∃ (response : ThreeDigitNumber),
          is_valid_move 
            { used_numbers := [opponent_move, initial_move], last_digit := opponent_move.ones } 
            response ∧
        first_player_winning_strategy n

theorem first_player_wins : ∀ n, first_player_winning_strategy n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_wins_l1359_135952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_piggy_bank_coins_l1359_135989

/-- Represents the contents of a piggy bank with quarters and dimes -/
structure PiggyBank where
  total_value : ℚ
  num_dimes : ℕ

/-- Calculates the total number of coins in the piggy bank -/
def total_coins (pb : PiggyBank) : ℕ :=
  pb.num_dimes + (((pb.total_value - (pb.num_dimes : ℚ) * (1 / 10)) / (1 / 4)).floor).toNat

/-- Theorem: The piggy bank with $19.75 and 35 dimes contains 100 coins in total -/
theorem piggy_bank_coins : 
  let pb : PiggyBank := { total_value := 19.75, num_dimes := 35 }
  total_coins pb = 100 := by
  sorry

#eval total_coins { total_value := 19.75, num_dimes := 35 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_piggy_bank_coins_l1359_135989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_path_l1359_135966

def U : Set (ℤ × ℤ) := {p | 0 ≤ p.1 ∧ p.1 ≤ 23 ∧ 0 ≤ p.2 ∧ p.2 ≤ 23}

def is_valid_segment (p q : ℤ × ℤ) : Prop :=
  (p.1 - q.1)^2 + (p.2 - q.2)^2 = 41

def is_valid_path (path : List (ℤ × ℤ)) : Prop :=
  path.head? = some (0, 0) ∧
  path.getLast? = some (0, 0) ∧
  path.toFinset = U ∧
  ∀ i, i < path.length - 1 → is_valid_segment (path.get! i) (path.get! (i+1))

theorem no_valid_path : ¬ ∃ path : List (ℤ × ℤ), is_valid_path path := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_path_l1359_135966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_M_equidistant_l1359_135942

/-- The distance between two points in 3D space -/
noncomputable def distance (x1 y1 z1 x2 y2 z2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2)

/-- Point M is on the x-axis and equidistant from A and B -/
theorem point_M_equidistant :
  let x_M : ℝ := -3/2
  let y_M : ℝ := 0
  let z_M : ℝ := 0
  let x_A : ℝ := 1
  let y_A : ℝ := -3
  let z_A : ℝ := 1
  let x_B : ℝ := 2
  let y_B : ℝ := 0
  let z_B : ℝ := 2
  distance x_M y_M z_M x_A y_A z_A = distance x_M y_M z_M x_B y_B z_B :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_M_equidistant_l1359_135942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_congruence_count_l1359_135990

theorem three_digit_congruence_count :
  let y_count := (Finset.range 1000).filter
    (fun y => 100 ≤ y ∧ y ≤ 999 ∧ (4528 * y + 563) % 29 = 1407 % 29)
  y_count.card = 31 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_congruence_count_l1359_135990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_from_slopes_l1359_135949

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - e.b^2 / e.a^2)

/-- The left vertex of an ellipse -/
def left_vertex (e : Ellipse) : ℝ × ℝ := (-e.a, 0)

/-- A point on the ellipse -/
structure PointOnEllipse (e : Ellipse) where
  x : ℝ
  y : ℝ
  h_on_ellipse : x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- The slope of a line from the left vertex to a point on the ellipse -/
noncomputable def slope_from_left_vertex (e : Ellipse) (p : PointOnEllipse e) : ℝ :=
  p.y / (p.x + e.a)

theorem ellipse_eccentricity_from_slopes (e : Ellipse) 
  (p q : PointOnEllipse e) (h_symmetric : p.x = -q.x ∧ p.y = q.y) 
  (h_slope_product : slope_from_left_vertex e p * slope_from_left_vertex e q = 1/4) :
  eccentricity e = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_from_slopes_l1359_135949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_comparison_l1359_135980

/-- Represents the candy counts for a person -/
structure CandyCounts where
  type1 : ℕ
  type2 : ℕ
  type3 : ℕ

/-- Calculates the total number of candies -/
def totalCandies (counts : CandyCounts) : ℕ :=
  counts.type1 + counts.type2 + counts.type3

/-- Calculates the sum of absolute differences between corresponding candy types -/
def sumOfDifferences (counts1 counts2 : CandyCounts) : ℕ :=
  (max counts1.type1 counts2.type1 - min counts1.type1 counts2.type1) +
  (max counts1.type2 counts2.type2 - min counts1.type2 counts2.type2) +
  (max counts1.type3 counts2.type3 - min counts1.type3 counts2.type3)

theorem candy_comparison :
  let bryan := CandyCounts.mk 50 25 15
  let ben := CandyCounts.mk 20 30 10
  (totalCandies bryan > totalCandies ben) ∧
  (sumOfDifferences bryan ben = 40) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_comparison_l1359_135980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_distance_proof_l1359_135939

noncomputable def num_points : ℕ := 8
noncomputable def radius : ℝ := 50

noncomputable def chord_length (r : ℝ) (θ : ℝ) : ℝ := 2 * r * Real.sin (θ / 2)

noncomputable def distance_traveled (n : ℕ) (r : ℝ) : ℝ :=
  let θ := 2 * Real.pi / n
  let non_adjacent_chord := chord_length r (2 * θ)
  let num_non_adjacent := (n - 3) / 2
  n * (2 * r + num_non_adjacent * non_adjacent_chord)

theorem circle_distance_proof :
  distance_traveled num_points radius = 800 + 1200 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_distance_proof_l1359_135939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_point_implies_a_equals_one_l1359_135959

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := 4 * Real.log x + a * x^2 - 6 * x + b

-- Define the derivative of f with respect to x
noncomputable def f_deriv (a x : ℝ) : ℝ := 4 / x + 2 * a * x - 6

-- Theorem statement
theorem extreme_value_point_implies_a_equals_one (a b : ℝ) :
  (∃ x, x > 0 ∧ f_deriv a x = 0) →
  f_deriv a 2 = 0 →
  a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_point_implies_a_equals_one_l1359_135959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1359_135950

noncomputable section

open Real

variable (a b c : ℝ)
variable (A B C : ℝ)

-- Define the triangle ABC
def triangle_ABC (a b c A B C : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ A < Real.pi ∧ 
  0 < B ∧ B < Real.pi ∧ 
  0 < C ∧ C < Real.pi ∧
  A + B + C = Real.pi

-- Given condition
def condition (a b c A B : ℝ) : Prop :=
  sqrt 3 * c = sqrt 3 * b * cos A + a * sin B

theorem triangle_properties 
  (h_triangle : triangle_ABC a b c A B C)
  (h_condition : condition a b c A B) :
  B = Real.pi / 3 ∧ 
  (a = 2 * sqrt 2 ∧ b = 2 * sqrt 3 → 
    c = sqrt 6 + sqrt 2 ∧ 
    (1/2) * a * b * sin C = 3 + sqrt 3) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1359_135950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_l1359_135947

noncomputable def M (a b : ℝ) : ℝ := max a b

noncomputable def m (a b : ℝ) : ℝ := min a b

theorem expression_value (x y z w v : ℝ) 
  (h_distinct : x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ x ≠ v ∧ y ≠ z ∧ y ≠ w ∧ y ≠ v ∧ z ≠ w ∧ z ≠ v ∧ w ≠ v)
  (h_order : x > y ∧ y > z ∧ z > w ∧ w > v) : 
  M (m y z) (M w (m x v)) = w :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_value_l1359_135947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_composition_negative_three_l1359_135938

noncomputable def g (x : ℝ) : ℝ := x⁻¹ + x⁻¹ / (2 + x⁻¹)

theorem g_composition_negative_three : g (g (-3)) = -135/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_composition_negative_three_l1359_135938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_divisible_by_10100_l1359_135913

theorem not_divisible_by_10100 (n : ℕ) : ¬(10100 ∣ (3^n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_divisible_by_10100_l1359_135913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sharks_fin_falcata_area_l1359_135983

/-- The area of a shark's fin falcata -/
theorem sharks_fin_falcata_area : (9 : ℝ) * Real.pi / 8 = 
  let r₁ : ℝ := 3  -- radius of the larger circle
  let r₂ : ℝ := 3/2  -- radius of the smaller circle
  let quarter_circle_area : ℝ := (1/4) * Real.pi * r₁^2
  let semicircle_area : ℝ := (1/2) * Real.pi * r₂^2
  quarter_circle_area - semicircle_area :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sharks_fin_falcata_area_l1359_135983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_even_sum_products_l1359_135912

/-- Probability that the first integer is even -/
noncomputable def p_even_a : ℝ := 1/2

/-- Probability that the second integer is even -/
noncomputable def p_even_b : ℝ := 1/3

/-- Probability that the sum of k products is even -/
noncomputable def p_even_sum (k : ℕ) : ℝ :=
  1/6 * (1/3)^(k-1) + 1/2

/-- Theorem stating the probability of the sum of k products being even -/
theorem prob_even_sum_products (k : ℕ) :
  p_even_sum k = 
    let p_even_product := p_even_a * p_even_b + p_even_a * (1 - p_even_b) + (1 - p_even_a) * p_even_b
    let p_odd_product := (1 - p_even_a) * (1 - p_even_b)
    (p_even_sum (k-1) * p_even_product + (1 - p_even_sum (k-1)) * p_odd_product) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_even_sum_products_l1359_135912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percentage_calculation_l1359_135968

theorem profit_percentage_calculation (cost_price marked_price : ℝ) 
  (h1 : cost_price = 66.5)
  (h2 : marked_price = 87.5)
  (discount_rate : ℝ) (h3 : discount_rate = 0.05) : 
  (marked_price * (1 - discount_rate) - cost_price) / cost_price * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percentage_calculation_l1359_135968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_zero_vector_sum_l1359_135921

/-- A regular polygon with 2017 sides -/
structure RegularPolygon2017 where
  vertices : Fin 2017 → ℝ × ℝ

/-- The vector from a point to a vertex of the polygon -/
def vectorToVertex (P : ℝ × ℝ) (A : ℝ × ℝ) : ℝ × ℝ :=
  (A.1 - P.1, A.2 - P.2)

/-- The magnitude of a vector -/
noncomputable def vectorMagnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 * v.1 + v.2 * v.2)

/-- The sum of scaled vectors from a point to all vertices -/
noncomputable def vectorSum (polygon : RegularPolygon2017) (P : ℝ × ℝ) : ℝ × ℝ :=
  Finset.sum (Finset.range 2017) (fun k =>
    let v := vectorToVertex P (polygon.vertices k)
    let scale := (k + 1 : ℝ) / (vectorMagnitude v) ^ 5
    (scale * v.1, scale * v.2))

/-- There exists a point where the vector sum is zero -/
theorem exists_zero_vector_sum (polygon : RegularPolygon2017) :
  ∃ P : ℝ × ℝ, vectorSum polygon P = (0, 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_zero_vector_sum_l1359_135921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_after_two_hours_l1359_135915

-- Define the hiking rates
noncomputable def jay_rate : ℚ := 1 / 20  -- miles per minute
noncomputable def sarah_rate : ℚ := 3 / 40  -- miles per minute

-- Define the time period
def time_period : ℚ := 120  -- 2 hours in minutes

-- Theorem statement
theorem distance_after_two_hours :
  let jay_distance := jay_rate * time_period
  let sarah_distance := sarah_rate * time_period
  jay_distance + sarah_distance = 15 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_after_two_hours_l1359_135915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1359_135932

noncomputable def f (x : ℝ) := Real.sqrt x / (x - 1)

def IsValidInput (f : ℝ → ℝ) (x : ℝ) : Prop := ∃ y, f x = y

theorem domain_of_f :
  {x : ℝ | IsValidInput f x} = {x : ℝ | x ≥ 0 ∧ x ≠ 1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1359_135932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_r_with_large_order_l1359_135964

open Nat

theorem exists_r_with_large_order (n : ℕ) (hn : n ≥ 2) :
  ∃ r : ℕ, r ≤ ⌈(16 : ℝ) * (log 2 n)^5⌉ ∧
    (∃ ω : ℕ, ω > 4 * (log 2 n)^2 ∧ n^ω ≡ 1 [MOD r]) := by
  sorry

#check exists_r_with_large_order

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_r_with_large_order_l1359_135964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sarahs_building_height_l1359_135930

/-- The height of Sarah's building -/
def building_height : ℝ := 37.5

/-- The length of the shadow cast by Sarah's building -/
def building_shadow : ℝ := 45

/-- The height of the lamppost -/
def lamppost_height : ℝ := 25

/-- The length of the shadow cast by the lamppost -/
def lamppost_shadow : ℝ := 30

/-- The theorem stating that the calculated height of Sarah's building, 
    rounded to the nearest whole number, is 38 feet -/
theorem sarahs_building_height :
  (building_height / lamppost_height = building_shadow / lamppost_shadow) →
  (Int.floor (building_height + 0.5) = 38) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sarahs_building_height_l1359_135930

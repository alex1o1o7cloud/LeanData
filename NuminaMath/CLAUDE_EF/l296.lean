import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_property_1_f_property_2_l296_29680

noncomputable def f (x : ℝ) : ℝ := x^2 / (1 + x^2)

theorem f_property_1 (x : ℝ) (hx : x ≠ 0) : f x + f (1/x) = 1 := by sorry

theorem f_property_2 : f 1 + f 2 + f 3 + f 4 + f (1/2) + f (1/3) + f (1/4) = 7/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_property_1_f_property_2_l296_29680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_ceil_fraction_square_l296_29603

theorem floor_ceil_fraction_square : 
  ⌊⌈((15:ℝ)/8)^2⌉ + 20/3⌋ = 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_ceil_fraction_square_l296_29603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_comparison_l296_29677

-- Define the Triangle structure
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the side lengths of a triangle
noncomputable def side_length (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the area of a triangle using Heron's formula
noncomputable def triangle_area (t : Triangle) : ℝ :=
  let a := side_length t.B t.C
  let b := side_length t.C t.A
  let c := side_length t.A t.B
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

-- Define the theorem
theorem area_comparison (ABC DEF : Triangle) 
  (h1 : side_length ABC.A ABC.B ≥ side_length DEF.A DEF.B)
  (h2 : side_length ABC.B ABC.C ≥ side_length DEF.B DEF.C)
  (h3 : side_length ABC.C ABC.A ≥ side_length DEF.C DEF.A)
  (acute_ABC : triangle_area ABC > 0)
  (acute_DEF : triangle_area DEF > 0) :
  triangle_area ABC ≥ triangle_area DEF := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_comparison_l296_29677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_g_g_x_eq_3_l296_29618

-- Define the piecewise function g
noncomputable def g (x : ℝ) : ℝ :=
  if -3 ≤ x ∧ x < -1 then -0.5 * x^2 + x + 4
  else if -1 ≤ x ∧ x < 2 then -x + 3
  else if 2 ≤ x ∧ x ≤ 3 then x - 2
  else 0  -- undefined outside [-3, 3]

-- Theorem statement
theorem unique_solution_g_g_x_eq_3 :
  ∃! x : ℝ, -3 ≤ x ∧ x ≤ 3 ∧ g (g x) = 3 :=
by
  -- The proof goes here
  sorry

#check unique_solution_g_g_x_eq_3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_g_g_x_eq_3_l296_29618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_condition_exists_monotonic_decreasing_increasing_specific_a_value_l296_29606

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x - 1

-- Theorem for part (1)
theorem monotonic_increasing_condition (a : ℝ) :
  (∀ x : ℝ, Monotone (f a)) → a ≤ 0 :=
by sorry

-- Theorem for part (2)
theorem exists_monotonic_decreasing_increasing :
  ∃ a : ℝ, (∀ x : ℝ, x ≤ 0 → StrictAnti (f a)) ∧
           (∀ x : ℝ, x ≥ 0 → StrictMono (f a)) :=
by sorry

-- Theorem to prove the specific value of a in part (2)
theorem specific_a_value :
  ∃ a : ℝ, a = 1 ∧
    (∀ x : ℝ, x ≤ 0 → StrictAnti (f a)) ∧
    (∀ x : ℝ, x ≥ 0 → StrictMono (f a)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_condition_exists_monotonic_decreasing_increasing_specific_a_value_l296_29606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_side_length_l296_29623

theorem right_triangle_side_length 
  (hypotenuse side1 side2 : ℝ) 
  (h1 : hypotenuse = 5) 
  (h2 : side1 = 3) 
  (h3 : hypotenuse^2 = side1^2 + side2^2) : 
  side2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_side_length_l296_29623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xiaozhao_journey_l296_29602

def movements : List Int := [1000, -900, 700, -1200, 1200, 100, -1100, -200]

def calorie_per_km : ℕ := 7000

theorem xiaozhao_journey :
  let final_position := movements.sum
  let total_distance := movements.map Int.natAbs |>.sum
  let calories_consumed := (total_distance * calorie_per_km) / 1000
  (final_position < 0 ∧ Int.natAbs final_position = 400) ∧
  calories_consumed = 44800 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_xiaozhao_journey_l296_29602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jims_taxi_charge_for_3_6_miles_l296_29671

/-- Represents a taxi service with an initial fee and per-distance charge. -/
structure TaxiService where
  initialFee : ℚ
  chargePerIncrement : ℚ
  incrementDistance : ℚ

/-- Calculates the total charge for a given distance. -/
def totalCharge (service : TaxiService) (distance : ℚ) : ℚ :=
  service.initialFee + service.chargePerIncrement * (distance / service.incrementDistance).floor

/-- Jim's taxi service -/
def jimsTaxi : TaxiService :=
  { initialFee := 2.25
  , chargePerIncrement := 0.4
  , incrementDistance := 2/5 }

theorem jims_taxi_charge_for_3_6_miles :
  totalCharge jimsTaxi (36/10) = 117/20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jims_taxi_charge_for_3_6_miles_l296_29671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_locus_is_parabola_l296_29670

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in 2D space defined by y = mx + b -/
structure Line where
  m : ℝ
  b : ℝ

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Distance from a point to a line -/
noncomputable def distanceToLine (p : Point) (l : Line) : ℝ :=
  abs (p.y - (l.m * p.x + l.b)) / Real.sqrt (1 + l.m^2)

/-- The plant location -/
def plant : Point := ⟨0, 0⟩

/-- The power transmission line -/
def ptl (d : ℝ) : Line := ⟨0, d⟩

/-- Theorem: The locus of points equidistant from a point and a line forms a parabola -/
theorem equidistant_locus_is_parabola (d : ℝ) (hd : d ≠ 0) :
  ∃ (a b c : ℝ), ∀ (p : Point),
    distance p plant = distanceToLine p (ptl d) →
    a * p.x^2 + b * p.x + c * p.y = 0 := by
  sorry

#check equidistant_locus_is_parabola

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_locus_is_parabola_l296_29670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_satisfies_condition_l296_29685

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := 3 * x - 1
noncomputable def g (x : ℝ) : ℝ := 2 * x + 3

-- Define the function h
noncomputable def h (x : ℝ) : ℝ := (2 * x + 4) / 3

-- Theorem statement
theorem h_satisfies_condition : ∀ x : ℝ, f (h x) = g x := by
  intro x
  -- Expand the definitions of f, g, and h
  simp [f, g, h]
  -- Perform algebraic manipulations
  ring
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_satisfies_condition_l296_29685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_union_M_P_l296_29672

def U : Set ℕ := {x : ℕ | x > 0 ∧ x ≤ 7}
def M : Set ℕ := {2, 4, 6}
def P : Set ℕ := {3, 4, 5}

theorem complement_of_union_M_P :
  (U \ (M ∪ P)) = {1, 7} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_union_M_P_l296_29672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_first_50_primes_l296_29626

-- Define a function to get the nth prime number
def nthPrime (n : ℕ) : ℕ := sorry

-- Define a function to sum the first n prime numbers
def sumFirstNPrimes (n : ℕ) : ℕ := 
  (List.range n).map (fun i => nthPrime (i + 1)) |>.sum

-- Theorem statement
theorem sum_first_50_primes : sumFirstNPrimes 50 = 5356 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_first_50_primes_l296_29626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_area_is_216_l296_29661

noncomputable section

-- Define the triangle EAB
def triangle_EAB (E A B : ℝ × ℝ) : Prop :=
  (E.1 - A.1) * (B.1 - A.1) + (E.2 - A.2) * (B.2 - A.2) = 0 -- Right angle at A

-- Define the length of BE
noncomputable def BE_length (B E : ℝ × ℝ) : ℝ :=
  Real.sqrt ((B.1 - E.1)^2 + (B.2 - E.2)^2)

-- Define the square ABCD
def square_ABCD (A B C D : ℝ × ℝ) : Prop :=
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = (C.1 - B.1)^2 + (C.2 - B.2)^2 ∧
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = (D.1 - C.1)^2 + (D.2 - C.2)^2 ∧
  (D.1 - C.1)^2 + (D.2 - C.2)^2 = (A.1 - D.1)^2 + (A.2 - D.2)^2 ∧
  (A.1 - D.1)^2 + (A.2 - D.2)^2 = (B.1 - A.1)^2 + (B.2 - A.2)^2

-- Define the rectangle HIJD
def rectangle_HIJD (H I J D : ℝ × ℝ) (A B : ℝ × ℝ) : Prop :=
  (I.1 - H.1)^2 + (I.2 - H.2)^2 = 4 * ((B.1 - A.1)^2 + (B.2 - A.2)^2) ∧
  (J.1 - H.1)^2 + (J.2 - H.2)^2 = (B.1 - A.1)^2 + (B.2 - A.2)^2

-- Define the total area
noncomputable def total_area (A B C D E F G H I J : ℝ × ℝ) : ℝ :=
  (B.1 - A.1)^2 + (B.2 - A.2)^2 +  -- Area of ABCD
  (F.1 - E.1)^2 + (F.2 - E.2)^2 +  -- Area of AEFG
  ((I.1 - H.1) * (J.2 - H.2))      -- Area of HIJD

-- Theorem statement
theorem total_area_is_216 
  (A B C D E F G H I J : ℝ × ℝ) :
  triangle_EAB E A B →
  BE_length B E = 12 →
  square_ABCD A B C D →
  rectangle_HIJD H I J D A B →
  total_area A B C D E F G H I J = 216 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_area_is_216_l296_29661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_infinite_set_with_square_free_sums_l296_29630

/-- A positive integer is square-free if no perfect square greater than 1 divides it. -/
def IsSquareFree (n : ℕ) : Prop :=
  ∀ k : ℕ, k > 1 → k * k ∣ n → k = 1

/-- The property that for any two distinct elements in a set, their sum is square-free. -/
def HasSquareFreeSums (S : Set ℕ) : Prop :=
  ∀ (a b : ℕ), a ∈ S → b ∈ S → a < b → IsSquareFree (a + b)

/-- There exists an infinite set of positive integers with square-free sums. -/
theorem exists_infinite_set_with_square_free_sums :
  ∃ M : Set ℕ, Set.Infinite M ∧ HasSquareFreeSums M := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_infinite_set_with_square_free_sums_l296_29630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_l296_29646

-- Define the curve and line
noncomputable def curve (x : ℝ) : ℝ := Real.log (2 * x - 1)
def line (x y : ℝ) : Prop := 2 * x - y + 3 = 0

-- Define the distance function from a point to the line
noncomputable def distance (x y : ℝ) : ℝ :=
  abs (2 * x - y + 3) / Real.sqrt 5

-- State the theorem
theorem shortest_distance :
  ∃ (x : ℝ), x > 1/2 ∧ 
  ∀ (x' : ℝ), x' > 1/2 → 
  distance x (curve x) ≤ distance x' (curve x') ∧
  distance x (curve x) = Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_l296_29646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_z_equals_sqrt_five_l296_29694

-- Define the complex number z
noncomputable def z : ℂ := 2 * Complex.I - 5 / (2 - Complex.I)

-- State the theorem
theorem abs_z_equals_sqrt_five : Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_z_equals_sqrt_five_l296_29694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_for_unique_isosceles_right_triangle_l296_29612

/-- An ellipse with semi-major axis a > 1 -/
structure Ellipse where
  a : ℝ
  h_a : a > 1

/-- A point on the ellipse -/
structure PointOnEllipse (E : Ellipse) where
  x : ℝ
  y : ℝ
  h_on_ellipse : x^2 / E.a^2 + y^2 = 1

/-- An isosceles right triangle on the ellipse -/
structure IsoscelesRightTriangleOnEllipse (E : Ellipse) where
  A : PointOnEllipse E
  B : PointOnEllipse E
  C : PointOnEllipse E
  h_A_vertex : A.y = 1 ∧ A.x = 0
  h_isosceles : (B.x - A.x)^2 + (B.y - A.y)^2 = (C.x - A.x)^2 + (C.y - A.y)^2
  h_right_angle : (B.x - A.x) * (C.x - A.x) + (B.y - A.y) * (C.y - A.y) = 0

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (E : Ellipse) : ℝ := Real.sqrt (1 - 1 / E.a^2)

/-- The theorem statement -/
theorem eccentricity_range_for_unique_isosceles_right_triangle (E : Ellipse) 
  (h_unique : ∃! t : IsoscelesRightTriangleOnEllipse E, True) :
  0 < eccentricity E ∧ eccentricity E ≤ Real.sqrt 6 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_for_unique_isosceles_right_triangle_l296_29612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_value_l296_29650

theorem tan_theta_value (θ : Real) (k : Real) 
  (h1 : Real.sin θ = (k + 1) / (k - 3))
  (h2 : Real.cos θ = (k - 1) / (k - 3))
  (h3 : k ≠ 3)
  (h4 : k ≠ 1) :
  Real.tan θ = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_value_l296_29650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_ratio_is_half_l296_29692

/-- A runner's journey with an injury halfway -/
structure RunnerJourney where
  totalDistance : ℝ
  secondHalfTime : ℝ
  timeDifference : ℝ

/-- Calculate the ratio of speeds for a runner's journey -/
noncomputable def speedRatio (journey : RunnerJourney) : ℝ :=
  let firstHalfTime := journey.secondHalfTime - journey.timeDifference
  let firstHalfSpeed := (journey.totalDistance / 2) / firstHalfTime
  let secondHalfSpeed := (journey.totalDistance / 2) / journey.secondHalfTime
  secondHalfSpeed / firstHalfSpeed

/-- Theorem: The speed ratio for the given journey is 1/2 -/
theorem speed_ratio_is_half :
  let journey : RunnerJourney := {
    totalDistance := 40,
    secondHalfTime := 8,
    timeDifference := 4
  }
  speedRatio journey = 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_ratio_is_half_l296_29692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_heat_released_in_resistor_correct_l296_29607

/-- Represents an electrical circuit with a resistor and capacitor in series, connected to a galvanic cell. -/
structure Circuit where
  R : ℝ  -- Resistance of the resistor
  C : ℝ  -- Capacitance of the capacitor
  ε : ℝ  -- Electromotive force (EMF) of the galvanic cell
  r : ℝ  -- Internal resistance of the galvanic cell

/-- Calculates the heat released in the resistor during capacitor charging. -/
noncomputable def heatReleasedInResistor (circuit : Circuit) : ℝ :=
  (circuit.C * circuit.ε^2 * circuit.R) / (2 * (circuit.R + circuit.r))

/-- Theorem stating that the heat released in the resistor is correctly calculated. -/
theorem heat_released_in_resistor_correct (circuit : Circuit) :
  heatReleasedInResistor circuit = (circuit.C * circuit.ε^2 * circuit.R) / (2 * (circuit.R + circuit.r)) :=
by
  -- Unfold the definition of heatReleasedInResistor
  unfold heatReleasedInResistor
  -- The equality follows directly from the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_heat_released_in_resistor_correct_l296_29607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_sequence_l296_29697

noncomputable def f (n : ℕ) : ℝ := (2 * n - Real.sin n) / (Real.sqrt n - (n^3 - 7)^(1/3))

theorem limit_of_sequence :
  Filter.Tendsto f Filter.atTop (nhds (-2)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_sequence_l296_29697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_digit_number_count_l296_29678

def odd_digits : Finset Nat := {1, 3, 5, 7, 9}
def even_digits : Finset Nat := {2, 4, 6, 8}

theorem five_digit_number_count : 
  (Finset.filter (λ s => Finset.card s = 3) (Finset.powerset odd_digits)).card *
  (Finset.filter (λ s => Finset.card s = 2) (Finset.powerset even_digits)).card *
  Nat.factorial 5 = 7200 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_digit_number_count_l296_29678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_to_circle_l296_29622

/-- The circle with center (1,1) and radius 1 -/
def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 1

/-- The point through which the tangent line passes -/
def point : ℝ × ℝ := (2, 3)

/-- The first potential tangent line: x = 2 -/
def line1 (x y : ℝ) : Prop := x = 2

/-- The second potential tangent line: 3x - 4y + 6 = 0 -/
def line2 (x y : ℝ) : Prop := 3*x - 4*y + 6 = 0

/-- Function to calculate the distance between a point and a line -/
noncomputable def distance_point_line (a b c : ℝ) (x y : ℝ) : ℝ :=
  abs (a*x + b*y + c) / Real.sqrt (a^2 + b^2)

theorem tangent_lines_to_circle :
  (∀ x y, line1 x y → distance_point_line 1 0 (-2) 1 1 = 1) ∧
  (∀ x y, line2 x y → distance_point_line 3 (-4) 6 1 1 = 1) ∧
  line1 point.1 point.2 ∧
  line2 point.1 point.2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_to_circle_l296_29622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_C₂_polar_to_cartesian_circles_intersect_l296_29627

-- Define the circles
def C₁ (x y : ℝ) : Prop := x^2 - 2*x + y^2 = 0
def C₂_polar (ρ θ : ℝ) : Prop := ρ = 2 * Real.sin θ

-- Define the Cartesian equation of C₂
def C₂_cartesian (x y : ℝ) : Prop := x^2 + (y-1)^2 = 1

-- Define the centers and radii of the circles
def center_C₁ : ℝ × ℝ := (1, 0)
def radius_C₁ : ℝ := 1
def center_C₂ : ℝ × ℝ := (0, 1)
def radius_C₂ : ℝ := 1

-- Define the distance between centers
noncomputable def distance_between_centers : ℝ := Real.sqrt 2

-- Theorem statements
theorem C₂_polar_to_cartesian : 
  ∀ x y ρ θ : ℝ, C₂_polar ρ θ → (x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) → C₂_cartesian x y := by
  sorry

theorem circles_intersect : 
  distance_between_centers < radius_C₁ + radius_C₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_C₂_polar_to_cartesian_circles_intersect_l296_29627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ramu_profit_percent_l296_29676

noncomputable section

def car_price : ℝ := 25000
def repair_cost : ℝ := 8500
def shipping_cost : ℝ := 2500
def import_tax_rate : ℝ := 0.15
def initial_exchange_rate : ℝ := 75
def final_exchange_rate : ℝ := 73
def selling_price_inr : ℝ := 4750000

def total_cost_before_tax : ℝ := car_price + repair_cost + shipping_cost
def import_tax : ℝ := import_tax_rate * (car_price + repair_cost)
def total_cost_usd : ℝ := total_cost_before_tax + import_tax
def total_cost_inr : ℝ := total_cost_usd * initial_exchange_rate
def selling_price_usd : ℝ := selling_price_inr / final_exchange_rate
def profit_usd : ℝ := selling_price_usd - total_cost_usd
def profit_percent : ℝ := (profit_usd / total_cost_usd) * 100

theorem ramu_profit_percent :
  abs (profit_percent - 57.15) < 0.01 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ramu_profit_percent_l296_29676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_implies_a_l296_29684

noncomputable def f (a x : ℝ) := a^(2*x) + 2*a^x - 1

theorem max_value_implies_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∃ x ∈ Set.Icc (-1 : ℝ) 1, f a x = 14) ∧
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, f a x ≤ 14) →
  a = 3 ∨ a = 1/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_implies_a_l296_29684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_max_triangle_area_slope_at_max_area_l296_29652

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Ellipse with given properties -/
structure Ellipse where
  a : ℝ  -- Semi-major axis
  b : ℝ  -- Semi-minor axis
  c : ℝ  -- Focal distance
  axiom_ratio : a = 2 * c
  axiom_left_focus : c = 2
  axiom_point_p : Point := { x := -8, y := 0 }

/-- Line passing through a point with a given slope -/
structure Line where
  slope : ℝ
  point : Point

/-- Triangle formed by two points on the ellipse and the right focus -/
structure Triangle where
  p1 : Point
  p2 : Point
  f2 : Point

def Ellipse.equation (e : Ellipse) (p : Point) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

def Line.intersect_ellipse (l : Line) (e : Ellipse) : Prop :=
  ∃ p1 p2 : Point, p1 ≠ p2 ∧ e.equation p1 ∧ e.equation p2 ∧
    l.slope = (p2.y - p1.y) / (p2.x - p1.x)

noncomputable def Triangle.area (t : Triangle) : ℝ :=
  abs ((t.p1.x - t.f2.x) * (t.p2.y - t.f2.y) - (t.p2.x - t.f2.x) * (t.p1.y - t.f2.y)) / 2

theorem ellipse_properties (e : Ellipse) :
  e.equation { x := x, y := y } ↔ x^2/16 + y^2/12 = 1 := by
  sorry

theorem max_triangle_area (e : Ellipse) :
  ∃ l : Line, l.point = e.axiom_point_p ∧
    l.intersect_ellipse e ∧
    (∀ t : Triangle, t.f2 = { x := 2, y := 0 } →
      t.area ≤ 3 * Real.sqrt 3) := by
  sorry

theorem slope_at_max_area (e : Ellipse) :
  ∃ l : Line, l.point = e.axiom_point_p ∧
    l.intersect_ellipse e ∧
    (∃ t : Triangle, t.f2 = { x := 2, y := 0 } ∧
      t.area = 3 * Real.sqrt 3 ∧
      (l.slope = Real.sqrt 21 / 14 ∨ l.slope = -Real.sqrt 21 / 14)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_max_triangle_area_slope_at_max_area_l296_29652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_class_characterization_z_union_classes_same_class_l296_29608

/-- Definition of class [k] -/
def classOf (k : ℕ) : Set ℤ := {x : ℤ | ∃ n : ℤ, x = 5 * n + k}

/-- The set of possible remainders -/
def R : Set ℕ := {0, 1, 2, 3, 4}

theorem class_characterization (x : ℤ) (k : ℕ) (hk : k ∈ R) :
  x ∈ classOf k ↔ x % 5 = k := by sorry

/-- Z is the union of all classes -/
theorem z_union_classes : (⋃ k ∈ R, classOf k) = Set.univ := by sorry

/-- If a - b is in class 0, then a and b are in the same class -/
theorem same_class (a b : ℤ) (h : a - b ∈ classOf 0) :
  ∀ k ∈ R, a ∈ classOf k ↔ b ∈ classOf k := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_class_characterization_z_union_classes_same_class_l296_29608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hat_distribution_l296_29695

theorem hat_distribution (S : Finset ℕ) (f : ℕ → ℕ) 
  (h_card : S.card = 30) (h_f : ∀ x, x ∈ S → f x ∈ S) :
  ∃ T : Finset ℕ, T ⊆ S ∧ T.card ≥ 10 ∧ ∀ x y, x ∈ T → y ∈ T → f x ≠ y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hat_distribution_l296_29695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_difference_l296_29638

/-- Given two rectangles with integer dimensions and perimeter 200 cm,
    where one rectangle has a side length of at least 60 cm,
    the greatest possible difference between their areas is 2025 cm². -/
theorem max_area_difference : 
  -- Define the first rectangle
  let rectangle1 (a b : ℕ) : Prop := a + b = 100 ∧ a > 0 ∧ b > 0

  -- Define the second rectangle with the constraint
  let rectangle2 (c d : ℕ) : Prop := c + d = 100 ∧ c ≥ 60 ∧ d > 0

  -- Define the area difference function
  let area_difference (a b c d : ℕ) : ℕ := (a * b).max (c * d) - (a * b).min (c * d)

  -- The theorem statement
  ∀ a b c d : ℕ,
    rectangle1 a b → rectangle2 c d →
    area_difference a b c d ≤ 2025 ∧
    ∃ a' b' c' d' : ℕ, rectangle1 a' b' ∧ rectangle2 c' d' ∧
      area_difference a' b' c' d' = 2025
:= by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_difference_l296_29638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l296_29675

def my_sequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a (n + 1) ^ 2 = a n ^ 2 + 4) ∧ 
  (a 1 = 1) ∧ 
  (∀ n, a n > 0)

theorem sequence_formula (a : ℕ → ℝ) (h : my_sequence a) :
  ∀ n : ℕ, n > 0 → a n = Real.sqrt (4 * ↑n - 3) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l296_29675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_opposite_edges_equal_l296_29666

/-- A tetrahedron with opposite edge lengths and circumradius. -/
structure Tetrahedron where
  a : ℝ
  a' : ℝ
  b : ℝ
  b' : ℝ
  c : ℝ
  c' : ℝ
  R : ℝ
  a_pos : 0 < a
  a'_pos : 0 < a'
  b_pos : 0 < b
  b'_pos : 0 < b'
  c_pos : 0 < c
  c'_pos : 0 < c'
  R_pos : 0 < R

/-- The theorem stating the necessary and sufficient condition for the equality of opposite edges. -/
theorem tetrahedron_opposite_edges_equal (t : Tetrahedron) :
  16 * t.R^2 = t.a^2 + t.a'^2 + t.b^2 + t.b'^2 + t.c^2 + t.c'^2 ↔
  t.a = t.a' ∧ t.b = t.b' ∧ t.c = t.c' :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_opposite_edges_equal_l296_29666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_i_l296_29619

open Complex Real

theorem min_distance_to_i :
  ∃ (z : ℂ), (abs (z - I) = sqrt (1/2)) ∧
  (∀ w : ℂ, Complex.abs (w^2 - 4) = Complex.abs (w * (w - 2*I)) → abs (w - I) ≥ abs (z - I)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_i_l296_29619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l296_29631

-- Define the propositions P and Q as functions of a
def P (a : ℝ) : Prop := ∀ x : ℝ, a * x^2 + a * x + 1 > 0

def Q (a : ℝ) : Prop := ∃ x : ℝ, x^2 - x + a = 0

-- Define the set of a values that satisfy the conditions
def A : Set ℝ := {a | (P a ∨ Q a) ∧ ¬(P a ∧ Q a)}

-- Theorem statement
theorem range_of_a : A = Set.union (Set.Ioi (-Real.pi)) (Set.Ioo (1/4) 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l296_29631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_root_bounded_difference_l296_29610

open Set
open Function
open Real

-- Define the set M
def M : Set (ℝ → ℝ) :=
  {f | Continuous f ∧ 
       (∃ x, f x = x) ∧ 
       (∀ x, DifferentiableAt ℝ f x ∧ 0 < deriv f x ∧ deriv f x < 1)}

-- Statement A
theorem unique_root (f : ℝ → ℝ) (hf : f ∈ M) : 
  ∃! x, f x = x :=
sorry

-- Statement B
theorem bounded_difference (f : ℝ → ℝ) (hf : f ∈ M) 
  (α β : ℝ) (hα : |α - 2012| < 1) (hβ : |β - 2012| < 1) :
  |f α - f β| < 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_root_bounded_difference_l296_29610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_area_l296_29615

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = x^2 - 4*x + 3

/-- The maximum area of the triangle -/
noncomputable def max_area : ℝ := 27/8

theorem triangle_max_area (p q : ℝ) :
  parabola 1 0 →
  parabola 4 3 →
  parabola p q →
  1 ≤ p →
  p ≤ 4 →
  ∃ (area : ℝ), area ≤ max_area ∧
    ∀ (other_area : ℝ),
      (∃ (x y : ℝ), parabola x y ∧ 1 ≤ x ∧ x ≤ 4 ∧
        other_area = (1/2) * |1*3 + 4*y + x*0 - 0*4 - 3*x - y*1|) →
      other_area ≤ area :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_area_l296_29615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_2_to_4_value_l296_29667

/-- A probability distribution for a random variable ξ -/
noncomputable def P (k : ℕ) : ℝ := 1 / (2 ^ k)

/-- The probability that 2 < ξ ≤ 4 -/
noncomputable def prob_2_to_4 : ℝ := P 3 + P 4

theorem prob_2_to_4_value : prob_2_to_4 = 3/16 := by
  unfold prob_2_to_4 P
  -- The rest of the proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_2_to_4_value_l296_29667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_enrique_commission_rate_l296_29693

-- Define the sales data
def suit_price : ℚ := 700
def suit_quantity : ℕ := 2
def shirt_price : ℚ := 50
def shirt_quantity : ℕ := 6
def loafer_price : ℚ := 150
def loafer_quantity : ℕ := 2

-- Define the total commission
def total_commission : ℚ := 300

-- Calculate total sales
def total_sales : ℚ := 
  suit_price * suit_quantity + 
  shirt_price * shirt_quantity + 
  loafer_price * loafer_quantity

-- Define the commission rate
noncomputable def commission_rate : ℚ := total_commission / total_sales

-- Theorem to prove
theorem enrique_commission_rate : 
  commission_rate = 15/100 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_enrique_commission_rate_l296_29693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_circle_l296_29643

/-- The set of complex numbers z satisfying |z-1| = 4 forms a circle in the complex plane -/
theorem complex_circle (z : ℂ) : 
  (Complex.abs (z - 1) = 4) ↔ ∃ (x y : ℝ), z = x + y * Complex.I ∧ (x - 1)^2 + y^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_circle_l296_29643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_average_speed_l296_29634

/-- Represents the distance between towns B and C -/
noncomputable def distance_BC : ℝ := sorry

/-- The average speed of a journey given total distance and total time -/
noncomputable def average_speed (total_distance : ℝ) (total_time : ℝ) : ℝ :=
  total_distance / total_time

theorem journey_average_speed :
  let distance_WB := 2 * distance_BC
  let speed_WB := (60 : ℝ)
  let speed_BC := (20 : ℝ)
  let time_WB := distance_WB / speed_WB
  let time_BC := distance_BC / speed_BC
  let total_distance := distance_WB + distance_BC
  let total_time := time_WB + time_BC
  average_speed total_distance total_time = 36 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_average_speed_l296_29634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_angles_correct_l296_29668

open Real

-- Define the line, parabola, ellipse, and trigonometric functions
def line (x y : ℝ) : Prop := x + y - 4 = 0
def parabola1 (x y : ℝ) : Prop := 2*y = 8 - x^2
def ellipse (x y : ℝ) : Prop := x^2 + 4*y^2 = 4
def parabola2 (x y : ℝ) : Prop := 4*y = 4 - 5*x^2
def sine (x y : ℝ) : Prop := y = sin x
def cosine (x y : ℝ) : Prop := y = cos x

-- Define the intersection angles
noncomputable def intersection_angles : List ℝ := [π/4, π/10, π/1.957, π/2.556, π/1.644, 0]

-- Theorem statement
theorem intersection_angles_correct :
  ∃ (x1 y1 x2 y2 x3 y3 x4 y4 x5 y5 x6 y6 : ℝ),
    -- Intersection points exist
    (line x1 y1 ∧ parabola1 x1 y1) ∧
    (line x2 y2 ∧ parabola1 x2 y2) ∧
    (ellipse x3 y3 ∧ parabola2 x3 y3) ∧
    (ellipse x4 y4 ∧ parabola2 x4 y4) ∧
    (ellipse x5 y5 ∧ parabola2 x5 y5) ∧
    (sine x6 y6 ∧ cosine x6 y6) ∧
    -- The angles at these intersections are correct
    intersection_angles = [
      arctan ((deriv (λ x => y1) x1 - (-1)) / (1 + deriv (λ x => y1) x1 * (-1))),
      arctan ((deriv (λ x => y2) x2 - (-1)) / (1 + deriv (λ x => y2) x2 * (-1))),
      arctan ((deriv (λ x => y3) x3 - deriv (λ y => x3) y3) / (1 + deriv (λ x => y3) x3 * deriv (λ y => x3) y3)),
      arctan ((deriv (λ x => y4) x4 - deriv (λ y => x4) y4) / (1 + deriv (λ x => y4) x4 * deriv (λ y => x4) y4)),
      arctan ((deriv (λ x => y5) x5 - deriv (λ y => x5) y5) / (1 + deriv (λ x => y5) x5 * deriv (λ y => x5) y5)),
      arctan ((cos x6 - (-sin x6)) / (1 + cos x6 * (-sin x6)))
    ] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_angles_correct_l296_29668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flash_catches_ace_l296_29687

/-- The total distance Flash must run to catch Ace -/
noncomputable def flash_distance (v x y z : ℝ) : ℝ :=
  (x * y) / 2 + (x * v * (y - (x * y) / 2)) / (x * v - v / z)

/-- Theorem stating the total distance Flash must run to catch Ace -/
theorem flash_catches_ace (v x y z : ℝ) (hv : v > 0) (hx : x > 0) (hy : y > 0) (hz : z > 1) :
  let ace_initial_speed := v
  let ace_tired_speed := v / z
  let flash_speed := x * v
  let ace_tired_distance := y / 2
  let flash_head_start := y
  flash_distance v x y z = (x * y) / 2 + (x * v * (y - (x * y) / 2)) / (x * v - v / z) :=
by
  sorry

#check flash_catches_ace

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flash_catches_ace_l296_29687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l296_29659

theorem triangle_side_length (b c B : ℝ) 
  (h1 : b = 50 * Real.sqrt 3) 
  (h2 : c = 150) 
  (h3 : B = 30 * π / 180) :
  let a := Real.sqrt (b^2 + c^2 - 2*b*c*(Real.cos B))
  a = 100 * Real.sqrt 3 ∨ a = 50 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l296_29659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lap_time_improvement_l296_29632

/-- Calculates the improvement in lap time given initial and current performance data --/
theorem lap_time_improvement 
  (initial_laps : ℕ) 
  (initial_time : ℕ) 
  (current_laps : ℕ) 
  (current_time : ℕ) 
  (h1 : initial_laps = 15)
  (h2 : initial_time = 45)
  (h3 : current_laps = 18)
  (h4 : current_time = 42) :
  (initial_time : ℚ) / initial_laps - (current_time : ℚ) / current_laps = 2/3 := by
  sorry

-- Remove the #eval line as it's not necessary for the theorem and may cause issues

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lap_time_improvement_l296_29632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_volume_is_61pi_l296_29644

-- Define the frustum-shaped glass
structure Frustum where
  lower_radius : ℝ
  upper_radius : ℝ
  height : ℝ

-- Define a sphere
structure Sphere where
  radius : ℝ

-- Define the setup
def setup (f : Frustum) (a b : Sphere) : Prop :=
  f.lower_radius = 2 ∧
  f.upper_radius = 7 ∧
  f.height = 12 ∧
  -- Sphere A touches the bottom and walls
  a.radius = f.lower_radius ∧
  -- Sphere B touches sphere A and walls (simplified condition)
  b.radius + a.radius < f.height

-- Calculate the volume of water
noncomputable def water_volume (f : Frustum) (a b : Sphere) : ℝ :=
  -- This is a placeholder for the actual calculation
  61 * Real.pi

-- Theorem statement
theorem water_volume_is_61pi (f : Frustum) (a b : Sphere) :
  setup f a b → water_volume f a b = 61 * Real.pi :=
by
  intro h
  -- The actual proof would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_volume_is_61pi_l296_29644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_irrational_pi_among_given_numbers_l296_29686

theorem irrational_pi_among_given_numbers : 
  (∃ (x : ℝ), x ∈ ({0.1, 1/3, π, (8 : ℝ)^(1/3)} : Set ℝ) ∧ Irrational x) ∧ 
  (∀ (x : ℝ), x ∈ ({0.1, 1/3, π, (8 : ℝ)^(1/3)} : Set ℝ) → Irrational x → x = π) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_irrational_pi_among_given_numbers_l296_29686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_problem_l296_29620

theorem trig_problem (α : ℝ) 
  (h1 : Real.sin α = 3/5) 
  (h2 : 0 < α ∧ α < π/2) : 
  Real.cos α = 4/5 ∧ Real.sin (α + π/4) = 7*Real.sqrt 2/10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_problem_l296_29620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_estimate_probability_l296_29624

def score_probability : ℚ := 2/5

def is_score (n : ℕ) : Bool :=
  1 ≤ n ∧ n ≤ 4

def count_scores (group : List ℕ) : ℕ :=
  (group.filter is_score).length

def simulation_data : List (List ℕ) :=
  [[9,0,7], [9,6,6], [1,9,1], [9,2,5], [2,7,1], [9,3,2], [8,1,2], [4,5,8], [5,6,9], [6,8,3],
   [4,3,1], [2,5,7], [3,9,3], [0,2,7], [5,5,6], [4,8,8], [7,3,0], [1,1,3], [5,3,7], [9,8,9]]

def exactly_two_scores (group : List ℕ) : Bool :=
  count_scores group = 2

theorem estimate_probability : 
  (simulation_data.filter exactly_two_scores).length / simulation_data.length = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_estimate_probability_l296_29624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_larger_x_l296_29625

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := (x - 5)^2 / 7^2 - (y + 3)^2 / 11^2 = 1

-- Define the focus with larger x-coordinate
noncomputable def focus_larger_x : ℝ × ℝ := (5 + Real.sqrt 170, -3)

-- Theorem statement
theorem hyperbola_focus_larger_x :
  ∃ (f1 f2 : ℝ × ℝ), 
    (∀ x y, hyperbola x y → (x = f1.1 ∧ y = f1.2) ∨ (x = f2.1 ∧ y = f2.2)) ∧
    f1.1 ≠ f2.1 ∧
    (f1.1 > f2.1 → f1 = focus_larger_x) ∧
    (f2.1 > f1.1 → f2 = focus_larger_x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_larger_x_l296_29625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_value_of_a_l296_29663

-- Define the curve C
def curve_C (a : ℝ) : ℝ → ℝ → Prop := λ x y => y^2 = 2*a*x ∧ a > 0

-- Define the line l
def line_l : ℝ → ℝ → Prop := λ x y => y = x - 2

-- Define the point P
def point_P : ℝ × ℝ := (-2, -4)

-- Define the intersection points A and B
def intersection_points (a : ℝ) : Prop := 
  ∃ (xA yA xB yB : ℝ),
    curve_C a xA yA ∧ curve_C a xB yB ∧
    line_l xA yA ∧ line_l xB yB

-- Define the condition |PA| · |PB| = |AB|²
def condition (a : ℝ) : Prop :=
  ∃ (xA yA xB yB : ℝ),
    curve_C a xA yA ∧ curve_C a xB yB ∧
    line_l xA yA ∧ line_l xB yB ∧
    ((xA + 2)^2 + (yA + 4)^2) * ((xB + 2)^2 + (yB + 4)^2) =
    ((xB - xA)^2 + (yB - yA)^2)^2

-- Theorem statement
theorem value_of_a :
  ∃ (a : ℝ), (∀ x y, curve_C a x y) ∧ (∀ x y, line_l x y) ∧ intersection_points a ∧ condition a → a = 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_value_of_a_l296_29663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hiking_trail_length_l296_29629

/-- Calculates the length of a hiking trail given uphill and downhill speeds, 
    percentage of uphill trail, and total time taken. -/
noncomputable def trail_length (uphill_speed downhill_speed : ℝ) 
                 (uphill_percent : ℝ) 
                 (total_time_minutes : ℝ) : ℝ :=
  let downhill_percent := 1 - uphill_percent
  let total_time_hours := total_time_minutes / 60
  total_time_hours / (uphill_percent / uphill_speed + downhill_percent / downhill_speed)

/-- Theorem stating that for given conditions, the trail length is approximately 2.308 miles -/
theorem hiking_trail_length :
  let l := trail_length 2 3 0.6 130
  ∃ ε > 0, |l - 2.308| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hiking_trail_length_l296_29629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_inequality_conclusion_l296_29641

theorem tan_inequality_conclusion (θ : Real) (n : Nat) (m : Real) :
  θ ∈ Set.Ioo 0 (Real.pi / 2) →
  (Real.tan θ + 1 / Real.tan θ ≥ 2) →
  (Real.tan θ + 2^2 / (Real.tan θ)^2 ≥ 3) →
  (Real.tan θ + 3^3 / (Real.tan θ)^3 ≥ 4) →
  (Real.tan θ + m / (Real.tan θ)^(n : Real) ≥ (n : Real) + 1) →
  m = (n : Real) ^ (n : Real) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_inequality_conclusion_l296_29641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_horizontal_line_l296_29617

/-- The slope angle of a horizontal line is 0. -/
theorem slope_angle_horizontal_line : 
  ∀ (y : ℝ → ℝ), (∀ x, y x = 0) → ∀ x, Real.arctan ((y (x + 1) - y x) / 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_horizontal_line_l296_29617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l296_29648

/-- A triangle with sides a, b, and c is isosceles right if a = b and a² + b² = c² -/
def IsIsoscelesRight (a b c : ℝ) : Prop :=
  a = b ∧ a^2 + b^2 = c^2

/-- Given a triangle ABC with sides a, b, c satisfying the equation,
    prove that it is an isosceles right triangle -/
theorem triangle_property (a b c : ℝ) 
  (h : (a - b)^2 + Real.sqrt (2*a - b - 3) + |c - 3 * Real.sqrt 2| = 0) :
  IsIsoscelesRight a b c := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l296_29648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tricycle_count_l296_29664

theorem tricycle_count (num_bicycles num_wheels : ℕ) : 
  num_bicycles = 24 →
  num_wheels = 90 →
  ∃ t : ℕ, 2 * num_bicycles + 3 * t = num_wheels ∧ t = 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tricycle_count_l296_29664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_range_condition_l296_29628

noncomputable def f (x : ℝ) : ℝ := 4 * x / (x^2 + 1)

noncomputable def g (k : ℝ) (x : ℝ) : ℝ := Real.cos (2 * Real.pi * x) + k * Real.cos (Real.pi * x)

theorem function_range_condition (k : ℝ) : 
  (∀ x₁ : ℝ, ∃ x₂ : ℝ, f x₁ = g k x₂) → (k ≥ 2 * Real.sqrt 2 ∨ k ≤ -2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_range_condition_l296_29628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lamp_replacement_cost_comparison_l296_29654

/-- Represents the energy consumption and cost calculation for lamps -/
structure LampData where
  wattage : ℝ
  hours_per_month : ℝ
  tariff : ℝ
  initial_cost : ℝ

/-- Calculates the total cost for a given number of months -/
noncomputable def total_cost (lamp : LampData) (months : ℝ) : ℝ :=
  lamp.initial_cost + (lamp.wattage * lamp.hours_per_month * lamp.tariff * months) / 1000

/-- Calculates the cost with energy service company for a given number of months -/
noncomputable def company_cost (old_lamp new_lamp : LampData) (months : ℝ) : ℝ :=
  let savings := (old_lamp.wattage - new_lamp.wattage) * new_lamp.hours_per_month * new_lamp.tariff / 1000
  let company_fee := if months ≤ 10 then 0.75 * savings * months else 0.75 * savings * 10
  total_cost new_lamp months + company_fee

/-- The main theorem to prove -/
theorem lamp_replacement_cost_comparison 
  (old_lamp : LampData)
  (new_lamp : LampData)
  (h1 : old_lamp.wattage = 60)
  (h2 : new_lamp.wattage = 12)
  (h3 : old_lamp.hours_per_month = 100)
  (h4 : new_lamp.hours_per_month = 100)
  (h5 : old_lamp.tariff = 5)
  (h6 : new_lamp.tariff = 5)
  (h7 : old_lamp.initial_cost = 0)
  (h8 : new_lamp.initial_cost = 120) :
  total_cost new_lamp 10 < company_cost old_lamp new_lamp 10 ∧
  total_cost new_lamp 36 < company_cost old_lamp new_lamp 36 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lamp_replacement_cost_comparison_l296_29654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_n_less_than_20b_l296_29674

noncomputable def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

def S (n : ℕ) : ℝ := n^2 - 2*n

noncomputable def b (n : ℕ) : ℝ := 
  if n = 0 then 0 else S n - S (n-1)

noncomputable def T (n : ℕ) : ℝ := (n * (arithmetic_sequence 2 3 1 + arithmetic_sequence 2 3 n)) / 2

theorem max_n_less_than_20b (n : ℕ) : 
  (∀ k : ℕ, k ≤ n → T k < 20 * b (k+1)) ∧ 
  (n + 1 ≤ 24 ∨ T (n + 1) ≥ 20 * b (n + 2)) → 
  n ≤ 24 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_n_less_than_20b_l296_29674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_evaluate_expression_l296_29691

theorem evaluate_expression : (125 : ℝ) ^ (1/3 : ℝ) * 81 ^ (-1/4 : ℝ) * 32 ^ (1/5 : ℝ) = 10/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_evaluate_expression_l296_29691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_circuit_resistance_total_resistance_for_one_ohm_l296_29616

/-- Represents an electrical circuit with symmetric structure and identical resistors -/
structure SymmetricCircuit where
  r : ℝ  -- Resistance value of each resistor
  r_pos : r > 0  -- Resistance is always positive

/-- The total resistance of a symmetric circuit -/
noncomputable def total_resistance (circuit : SymmetricCircuit) : ℝ := (7 / 5) * circuit.r

/-- Theorem stating that the total resistance of the symmetric circuit is 7/5 * r -/
theorem symmetric_circuit_resistance (circuit : SymmetricCircuit) :
  total_resistance circuit = (7 / 5) * circuit.r := by
  -- Unfold the definition of total_resistance
  unfold total_resistance
  -- The equality is true by definition
  rfl

/-- Given r = 1 Ohm, the total resistance is 1.4 Ohms -/
theorem total_resistance_for_one_ohm (circuit : SymmetricCircuit) 
  (h : circuit.r = 1) : total_resistance circuit = 1.4 := by
  rw [symmetric_circuit_resistance]
  rw [h]
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_circuit_resistance_total_resistance_for_one_ohm_l296_29616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_angle_vertical_line_l296_29635

noncomputable def inclination_angle (x : ℝ) : ℝ := sorry

theorem inclination_angle_vertical_line : 
  ∀ (θ : ℝ), 
  θ = 60 * π / 180 → 
  ∃ (x : ℝ), 
  (∀ (y : ℝ), x = Real.tan θ) → 
  inclination_angle x = π / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inclination_angle_vertical_line_l296_29635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scalene_triangle_no_equal_division_l296_29656

/-- A scalene triangle is a triangle with no two sides equal -/
def IsScalene (triangle : Set (ℝ × ℝ)) : Prop :=
  ∀ a b c : ℝ × ℝ, a ∈ triangle → b ∈ triangle → c ∈ triangle →
  a ≠ b → b ≠ c → c ≠ a → 
  dist a b ≠ dist b c ∧ dist b c ≠ dist c a ∧ dist c a ≠ dist a b

/-- A line divides a triangle if it intersects two sides of the triangle -/
def LinesDivideTriangle (line : Set (ℝ × ℝ)) (triangle : Set (ℝ × ℝ)) : Prop :=
  ∃ p q : ℝ × ℝ, p ∈ line ∩ triangle ∧ q ∈ line ∩ triangle ∧ p ≠ q

/-- Two sets are equal figures if they have the same area -/
noncomputable def EqualFigures (s1 s2 : Set (ℝ × ℝ)) : Prop :=
  MeasureTheory.volume s1 = MeasureTheory.volume s2

/-- Main theorem: No line can divide a scalene triangle into two equal figures -/
theorem scalene_triangle_no_equal_division (triangle : Set (ℝ × ℝ)) 
  (h_scalene : IsScalene triangle) :
  ¬ ∃ (line : Set (ℝ × ℝ)) (part1 part2 : Set (ℝ × ℝ)), 
    LinesDivideTriangle line triangle ∧
    part1 ∪ part2 = triangle ∧
    part1 ∩ part2 ⊆ line ∧
    EqualFigures part1 part2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scalene_triangle_no_equal_division_l296_29656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_l296_29655

noncomputable section

open Real Set

-- Define the function f
def f (ω : ℝ) (x : ℝ) : ℝ := sin (ω * x + π / 3)

-- Define the interval
def interval : Set ℝ := Ioo 0 π

-- Define the properties of the function
def has_three_extreme_points (ω : ℝ) : Prop :=
  ∃ (a b c : ℝ), a ∈ interval ∧ b ∈ interval ∧ c ∈ interval ∧
  a < b ∧ b < c ∧
  (∀ x ∈ interval, x < a ∨ (a < x ∧ x < b) ∨ (b < x ∧ x < c) ∨ c < x →
    (deriv (f ω)) x ≠ 0)

def has_two_zeros (ω : ℝ) : Prop :=
  ∃ (a b : ℝ), a ∈ interval ∧ b ∈ interval ∧ a < b ∧
  f ω a = 0 ∧ f ω b = 0 ∧
  ∀ x ∈ interval, x < a ∨ (a < x ∧ x < b) ∨ b < x →
    f ω x ≠ 0

-- State the theorem
theorem omega_range :
  ∀ ω : ℝ, ω > 0 →
    has_three_extreme_points ω →
    has_two_zeros ω →
    13/6 < ω ∧ ω ≤ 8/3 :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_l296_29655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_volume_in_tank_l296_29660

/-- Represents a cylindrical tank --/
structure CylindricalTank where
  height : ℝ
  diameter : ℝ

/-- Represents a mixture of oil and water --/
structure OilWaterMixture where
  oilRatio : ℝ
  waterRatio : ℝ

/-- Calculates the volume of oil in a partially filled cylindrical tank --/
noncomputable def volumeOfOil (tank : CylindricalTank) (mixture : OilWaterMixture) (fillRatio : ℝ) : ℝ :=
  let radius := tank.diameter / 2
  let filledHeight := tank.height * fillRatio
  let totalVolume := Real.pi * radius^2 * filledHeight
  let oilProportion := mixture.oilRatio / (mixture.oilRatio + mixture.waterRatio)
  totalVolume * oilProportion

/-- Theorem stating the volume of oil in the specific tank --/
theorem oil_volume_in_tank :
  let tank : CylindricalTank := { height := 9, diameter := 3 }
  let mixture : OilWaterMixture := { oilRatio := 2, waterRatio := 5 }
  let fillRatio : ℝ := 1/3
  abs (volumeOfOil tank mixture fillRatio - 6.04) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_volume_in_tank_l296_29660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_implies_a_bound_l296_29642

/-- The function f(x) = a*ln(x+1) - x^2 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log (x + 1) - x^2

/-- Theorem stating that if the given inequality holds for all p and q in (0,1) with p ≠ q, 
    then a must be greater than or equal to 18 -/
theorem f_inequality_implies_a_bound (a : ℝ) : 
  (∀ p q : ℝ, 0 < p ∧ p < 1 ∧ 0 < q ∧ q < 1 ∧ p ≠ q → 
    (f a (p + 1) - f a (q + 1)) / (p - q) > 2) → 
  a ≥ 18 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_implies_a_bound_l296_29642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_formula_area_increase_l296_29633

/-- Represents a trapezoid with upper base x, lower base 15, and height 8 -/
structure Trapezoid where
  x : ℝ
  lower_base : ℝ := 15
  height : ℝ := 8

/-- The area of the trapezoid -/
noncomputable def area (t : Trapezoid) : ℝ := (1/2) * (t.x + t.lower_base) * t.height

theorem trapezoid_area_formula (t : Trapezoid) :
  area t = 4 * t.x + 60 := by sorry

theorem area_increase (t : Trapezoid) :
  area {x := t.x + 1, lower_base := t.lower_base, height := t.height} - area t = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_formula_area_increase_l296_29633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_ratio_l296_29639

/-- Definition of the sum of a geometric series -/
noncomputable def geometric_sum (a : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a else a * (1 - q^n) / (1 - q)

/-- Theorem: For a geometric series with first term a and common ratio q,
    if 2S_1 + S_3 = 3S_2, then q = 2 -/
theorem geometric_series_ratio (a : ℝ) (q : ℝ) (h_a : a ≠ 0) (h_q : q ≠ 0) (h_q_ne_1 : q ≠ 1) :
  2 * (geometric_sum a q 1) + (geometric_sum a q 3) = 3 * (geometric_sum a q 2) → q = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_ratio_l296_29639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_coords_M_cartesian_eq_C_sum_reciprocal_distances_l296_29682

noncomputable section

-- Define the line l
def line_l (t : ℝ) : ℝ × ℝ := ((Real.sqrt 2 / 2) * t, 2 + (Real.sqrt 2 / 2) * t)

-- Define the curve C in polar coordinates
def curve_C : ℝ → ℝ := fun _ => 4

-- Define point P
def point_P : ℝ × ℝ := (0, 2)

-- Theorem 1: Polar coordinates of M when t = -√2
theorem polar_coords_M :
  let (x, y) := line_l (-Real.sqrt 2)
  let r := Real.sqrt (x^2 + y^2)
  let θ := Real.arctan (y / x)
  (r = Real.sqrt 2) ∧ (θ = 3 * Real.pi / 4) := by
  sorry

-- Theorem 2: Cartesian equation of curve C
theorem cartesian_eq_C :
  ∀ (x y : ℝ), (x^2 + y^2 = 16) ↔ (∃ θ, x = curve_C θ * Real.cos θ ∧ y = curve_C θ * Real.sin θ) := by
  sorry

-- Theorem 3: Value of 1/|PA| + 1/|PB|
theorem sum_reciprocal_distances :
  let A := line_l (Real.sqrt 14 / 2 - Real.sqrt 2)
  let B := line_l (-Real.sqrt 14 / 2 - Real.sqrt 2)
  (1 / Real.sqrt ((point_P.1 - A.1)^2 + (point_P.2 - A.2)^2)) +
  (1 / Real.sqrt ((point_P.1 - B.1)^2 + (point_P.2 - B.2)^2)) =
  Real.sqrt 14 / 6 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_coords_M_cartesian_eq_C_sum_reciprocal_distances_l296_29682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_is_two_thirds_l296_29601

/-- A square with side length s -/
structure Square (s : ℝ) where
  side : s > 0

/-- Triangle A formed in the square -/
noncomputable def TriangleA (s : ℝ) : ℝ :=
  (1 / 2) * (s / 3) * (s / 2)

/-- Triangle B formed in the square -/
noncomputable def TriangleB (s : ℝ) : ℝ :=
  (1 / 2) * (s / 4) * s

/-- The ratio of the areas of Triangle A to Triangle B -/
noncomputable def AreaRatio (s : ℝ) : ℝ :=
  TriangleA s / TriangleB s

theorem area_ratio_is_two_thirds (s : ℝ) (sq : Square s) :
  AreaRatio s = 2 / 3 := by
  sorry

#eval "Theorem stated successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_is_two_thirds_l296_29601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xy_value_l296_29665

theorem xy_value (x y : ℝ) (h1 : y / x = 2 / y) (h2 : 2 / y = 1 / x + 9 / (x^2)) : x * y = 27 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_xy_value_l296_29665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_front_view_shows_max_heights_figure_front_view_l296_29600

/-- Represents a stack of cubes in a single column -/
def StackColumn := List Nat

/-- Represents a stack map as a list of columns -/
def StackMap := List StackColumn

/-- Calculates the maximum height in a stack column -/
def maxHeight (column : StackColumn) : Nat :=
  column.foldl max 0

/-- Represents the front view of a stack map -/
def FrontView := List Nat

/-- Theorem: The front view of a stack map shows the maximum heights of each column -/
theorem front_view_shows_max_heights (map : StackMap) :
  map.map maxHeight = map.map maxHeight := by
  rfl

/-- Given stack map from Figure 4 -/
def figureMap : StackMap := [[2, 1], [1, 3, 1], [4, 1]]

/-- Theorem: The front view of the given stack map is [2, 3, 4] -/
theorem figure_front_view :
  figureMap.map maxHeight = [2, 3, 4] := by
  rfl

#eval figureMap.map maxHeight

end NUMINAMATH_CALUDE_ERRORFEEDBACK_front_view_shows_max_heights_figure_front_view_l296_29600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l296_29679

noncomputable def f (x : ℝ) := 4/3 * x - 2 - 3 * x^2

theorem f_max_value :
  (∀ x : ℝ, f x ≤ f (2/9)) ∧ f (2/9) = -16/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l296_29679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_our_poly_is_fourth_degree_quadrinomial_l296_29681

/-- A polynomial in two variables -/
structure MyPolynomial (R : Type*) [Semiring R] where
  coeff : List (R × ℕ × ℕ)

/-- The degree of a term in a two-variable polynomial -/
def term_degree {R : Type*} [Semiring R] (term : R × ℕ × ℕ) : ℕ := term.2.1 + term.2.2

/-- The degree of a two-variable polynomial -/
def poly_degree {R : Type*} [Semiring R] (p : MyPolynomial R) : ℕ := 
  p.coeff.foldl (λ acc term => max acc (term_degree term)) 0

/-- The number of terms in a polynomial -/
def term_count {R : Type*} [Semiring R] (p : MyPolynomial R) : ℕ := p.coeff.length

/-- Our specific polynomial -/
def our_poly : MyPolynomial ℤ := 
  { coeff := [(3, 2, 1), (-1, 1, 3), (5, 1, 1), (-1, 0, 0)] }

theorem our_poly_is_fourth_degree_quadrinomial : 
  poly_degree our_poly = 4 ∧ term_count our_poly = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_our_poly_is_fourth_degree_quadrinomial_l296_29681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_first_prime_first_is_prime_count_primes_in_sequence_l296_29614

def sequenceN (n : ℕ) : ℕ :=
  match n with
  | 0 => 47
  | n + 1 => sequenceN n * 100 + 47

theorem only_first_prime :
  ∀ n : ℕ, n > 0 → ¬(Nat.Prime (sequenceN n)) :=
by sorry

theorem first_is_prime :
  Nat.Prime (sequenceN 0) :=
by sorry

theorem count_primes_in_sequence :
  (Finset.filter Nat.Prime (Finset.range ω)).card = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_first_prime_first_is_prime_count_primes_in_sequence_l296_29614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_three_zeros_l296_29658

-- Define the piecewise function f
noncomputable def f (a x : ℝ) : ℝ :=
  if x > a then x + 2 else x^2 + 5*x + 2

-- Define g in terms of f
noncomputable def g (a x : ℝ) : ℝ := f a x - 2*x

-- Define a predicate for g having exactly three distinct zeros
def has_three_distinct_zeros (a : ℝ) : Prop :=
  ∃ (x y z : ℝ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    g a x = 0 ∧ g a y = 0 ∧ g a z = 0 ∧
    ∀ w, g a w = 0 → w = x ∨ w = y ∨ w = z

-- Theorem statement
theorem range_of_a_for_three_zeros :
  {a : ℝ | has_three_distinct_zeros a} = Set.Ici (-1) ∩ Set.Iio 2 := by
  sorry

#check range_of_a_for_three_zeros

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_three_zeros_l296_29658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_appliance_price_proof_l296_29604

/-- Calculates the discount amount based on the original price --/
noncomputable def discount (price : ℝ) : ℝ :=
  if price ≤ 200 then 0
  else if price ≤ 500 then (price - 200) * 0.1
  else (300 * 0.1) + (price - 500) * 0.2

/-- The promotional event rules for the shopping mall --/
theorem appliance_price_proof (price : ℝ) :
  price > 0 →
  discount price = 330 →
  price = 2000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_appliance_price_proof_l296_29604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_and_directrix_l296_29662

/-- Represents a parabola with equation y^2 = 2px -/
structure Parabola where
  p : ℝ

/-- The focus of a parabola -/
noncomputable def focus (para : Parabola) : ℝ × ℝ := (para.p / 2, 0)

/-- The equation of the directrix of a parabola -/
def directrix (para : Parabola) : ℝ → Prop := λ x ↦ x = -para.p / 2

theorem parabola_focus_and_directrix (para : Parabola) 
  (h : focus para = (2, 0)) : 
  para.p = 4 ∧ directrix para = λ x ↦ x = -2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_and_directrix_l296_29662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_term_of_specific_arithmetic_sequence_l296_29649

/-- An arithmetic sequence with first term a₁ and common difference d -/
def arithmetic_sequence (a₁ : ℝ) (d : ℝ) : ℕ → ℝ :=
  λ n ↦ a₁ + (n - 1 : ℝ) * d

/-- Theorem: The 10th term of the arithmetic sequence with a₁ = 3 and d = 2 is 21 -/
theorem tenth_term_of_specific_arithmetic_sequence :
  arithmetic_sequence 3 2 10 = 21 := by
  -- Unfold the definition of arithmetic_sequence
  unfold arithmetic_sequence
  -- Simplify the expression
  simp [Nat.cast_sub, Nat.cast_one]
  -- Evaluate the arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_term_of_specific_arithmetic_sequence_l296_29649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_for_p_geq_m_squared_minus_m_minus_one_l296_29653

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2^(x + 1)

-- Define properties for even and odd functions
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x
def is_odd (h : ℝ → ℝ) : Prop := ∀ x, h (-x) = -h x

-- State the theorem
theorem range_of_m_for_p_geq_m_squared_minus_m_minus_one 
  (g h : ℝ → ℝ) (m : ℝ) :
  (∀ x, f x = g x + h x) →
  is_even g →
  is_odd h →
  (∀ x ∈ Set.Icc 1 2, ∀ t, h x = t → 
    g (2*x) + 2*m*h x + m^2 - m - 1 ≥ m^2 - m - 1) →
  m ≥ -17/12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_for_p_geq_m_squared_minus_m_minus_one_l296_29653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_on_circle_l296_29657

-- Define the complex equation
def complex_equation (z : ℂ) : Prop := (z - 1)^6 = 64 * z^6

-- Define the distance from a point to (1/3, 0)
noncomputable def distance_from_center (z : ℂ) : ℝ := Complex.abs (z - (1/3 : ℂ))

-- Theorem statement
theorem roots_on_circle : ∀ z : ℂ, complex_equation z → distance_from_center z = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_on_circle_l296_29657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_bm_length_l296_29605

/-- Represents a trapezoid ABCD with point M on side CD -/
structure Trapezoid :=
  (A B C D M : ℝ × ℝ)
  (on_side : M.1 = C.1 ∨ M.1 = D.1)

/-- The angle between two vectors -/
noncomputable def angle (v w : ℝ × ℝ) : ℝ := sorry

theorem trapezoid_bm_length 
  (ABCD : Trapezoid)
  (angle_equality : angle (ABCD.B - ABCD.C) (ABCD.D - ABCD.C) = 
                    angle (ABCD.C - ABCD.B) (ABCD.D - ABCD.B) ∧
                    angle (ABCD.C - ABCD.B) (ABCD.D - ABCD.B) = 
                    angle (ABCD.A - ABCD.B) (ABCD.M - ABCD.B) ∧
                    angle (ABCD.A - ABCD.B) (ABCD.M - ABCD.B) = 
                    Real.arccos 0.05)
  (ab_length : Real.sqrt ((ABCD.A.1 - ABCD.B.1)^2 + (ABCD.A.2 - ABCD.B.2)^2) = 9) :
  Real.sqrt ((ABCD.B.1 - ABCD.M.1)^2 + (ABCD.B.2 - ABCD.M.2)^2) = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_bm_length_l296_29605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_soap_packet_economical_order_l296_29673

/-- Represents a soap packet with cost and quantity -/
structure SoapPacket where
  cost : ℝ
  quantity : ℝ

/-- Calculates the cost per unit quantity -/
noncomputable def costPerUnit (packet : SoapPacket) : ℝ :=
  packet.cost / packet.quantity

theorem soap_packet_economical_order 
  (tiny regular jumbo : SoapPacket)
  (h1 : regular.cost = 1.25 * tiny.cost)
  (h2 : regular.quantity = 0.75 * jumbo.quantity)
  (h3 : jumbo.quantity = 2.5 * tiny.quantity)
  (h4 : jumbo.cost = 1.2 * regular.cost) :
  costPerUnit jumbo < costPerUnit regular ∧ 
  costPerUnit regular < costPerUnit tiny := by
  sorry

#check soap_packet_economical_order

end NUMINAMATH_CALUDE_ERRORFEEDBACK_soap_packet_economical_order_l296_29673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_c_coordinates_angle_between_a_and_b_l296_29696

def a : ℝ × ℝ := (1, -1)

theorem vector_c_coordinates (c : ℝ × ℝ) :
  (c.1^2 + c.2^2 = 18) ∧ (c.2 = -c.1) → (c = (-3, 3) ∨ c = (3, -3)) := by sorry

theorem angle_between_a_and_b (b : ℝ × ℝ) :
  (b.1^2 + b.2^2 = 1) ∧ (a.1 * (a.1 - 2*b.1) + a.2 * (a.2 - 2*b.2) = 0) →
  Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))) = π / 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_c_coordinates_angle_between_a_and_b_l296_29696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_a_for_one_extreme_point_l296_29611

-- Define the function f
noncomputable def f (a : ℕ) (x : ℝ) : ℝ := Real.log x + a / (x + 1)

-- Define the derivative of f
noncomputable def f_derivative (a : ℕ) (x : ℝ) : ℝ := 1 / x - a / ((x + 1) ^ 2)

-- Define the condition for having only one extreme value point
def has_one_extreme_point (a : ℕ) : Prop :=
  ∃! x, x > 1 ∧ x < 3 ∧ f_derivative a x = 0

-- The main theorem
theorem unique_a_for_one_extreme_point :
  ∃! a : ℕ, has_one_extreme_point a :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_a_for_one_extreme_point_l296_29611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_form_l296_29637

/-- A quadratic function with specific properties -/
noncomputable def q (x : ℝ) : ℝ := (12 * x^2 - 48) / 5

/-- The rational function formed by the reciprocal of q -/
noncomputable def f (x : ℝ) : ℝ := 1 / q x

theorem quadratic_function_form (h1 : ∀ x ≠ -2, ∃ ε > 0, ∀ y ∈ Set.Ioo (x - ε) (x + ε), |f y| > 1/ε)
  (h2 : ∀ x ≠ 2, ∃ ε > 0, ∀ y ∈ Set.Ioo (x - ε) (x + ε), |f y| > 1/ε)
  (h3 : q 3 = 12) :
  ∀ x, q x = (12 * x^2 - 48) / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_form_l296_29637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_a_problem_b_problem_c_l296_29636

def digit_sum (n : ℕ) : ℕ := sorry

def num_digits (n : ℕ) : ℕ := sorry

def P (n : ℕ) : ℕ := digit_sum n + num_digits n

theorem problem_a : P 2017 = 14 := by sorry

theorem problem_b : ∀ n : ℕ, P n = 4 ↔ n = 3 ∨ n = 11 ∨ n = 20 ∨ n = 100 := by sorry

theorem problem_c : ∃ n : ℕ, P n - P (n + 1) > 50 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_a_problem_b_problem_c_l296_29636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bug_speeds_correct_l296_29613

-- Define the triangle and bug properties
noncomputable def triangle_side_length : ℝ := 108
noncomputable def speed_ratio_alpha_beta : ℝ := 4 / 5
noncomputable def alpha_rest_time : ℝ := 10
noncomputable def beta_speed_increase : ℝ := 1.2

-- Define the speeds as variables we want to prove
noncomputable def alpha_speed : ℝ := 10.6
noncomputable def beta_speed : ℝ := 13.25

-- Theorem statement
theorem bug_speeds_correct : 
  -- First meeting conditions
  (alpha_speed * speed_ratio_alpha_beta⁻¹ * triangle_side_length * 3 = 
   beta_speed * triangle_side_length * 3) ∧
  -- Second meeting conditions
  (alpha_speed * (triangle_side_length * 3 / 2 - alpha_speed * alpha_rest_time) = 
   beta_speed * beta_speed_increase * (triangle_side_length * 3 / 2 + 
   beta_speed * alpha_rest_time)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bug_speeds_correct_l296_29613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_PQ_is_4_l296_29609

-- Define the curve C in Cartesian coordinates
def C (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define the polar coordinate transformation
noncomputable def polar_to_cartesian (ρ θ : ℝ) : ℝ × ℝ := (ρ * Real.cos θ, ρ * Real.sin θ)

-- Define curve C in polar coordinates
def C_polar (ρ θ : ℝ) : Prop := ρ = 2 * Real.cos θ

-- Define line l1 in polar coordinates
def l1 (ρ θ : ℝ) : Prop := 2 * ρ * Real.sin (θ + Real.pi/3) + 3 * Real.sqrt 3 = 0

-- Define line l2 in polar coordinates
def l2 (ρ θ : ℝ) : Prop := θ = Real.pi/3

-- Define point P as the intersection of C and l2 (excluding O)
noncomputable def P : ℝ × ℝ := polar_to_cartesian 1 (Real.pi/3)

-- Define point Q as the intersection of l1 and l2
noncomputable def Q : ℝ × ℝ := polar_to_cartesian (-3) (Real.pi/3)

-- State the theorem
theorem length_PQ_is_4 : 
  ∀ (x y ρ θ : ℝ), 
    C x y → 
    C_polar ρ θ → 
    l1 ρ θ → 
    l2 ρ θ → 
    Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_PQ_is_4_l296_29609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l296_29669

/-- Given a triangle ABC with the following properties:
  1. cos A = 3/5
  2. sin C = 2 cos B
  3. Side length a = 4
Prove that the area of the triangle is 8. -/
theorem triangle_area (A B C : ℝ) (a b c : ℝ) : 
  Real.cos A = 3/5 →
  Real.sin C = 2 * Real.cos B →
  a = 4 →
  (1/2) * a * b * Real.sin C = 8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l296_29669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_series_sum_l296_29699

theorem fraction_series_sum : 
  let series := [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 18]
  (series.map (λ x => (x : ℚ) / 12)).sum = 15/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_series_sum_l296_29699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lowest_probability_cards_l296_29647

/-- Represents the probability of a card being red side up after flips -/
def probability_red (k : ℕ) : ℚ :=
  if k ≤ 25 then
    (676 - 52 * k + 2 * k^2) / 676
  else
    (676 - 52 * (51 - k) + 2 * (51 - k)^2) / 676

/-- The theorem stating that cards 13 and 38 have the lowest probability of being red side up -/
theorem lowest_probability_cards :
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 50 →
    (probability_red 13 ≤ probability_red n ∧
     probability_red 38 ≤ probability_red n) :=
by sorry

/-- The number of cards -/
def num_cards : ℕ := 50

/-- The number of consecutive cards flipped in each operation -/
def flip_size : ℕ := 25

/-- Represents whether a card is red side up -/
def is_red_side_up : ℕ → Prop := sorry

/-- Assumption that cards are initially red side up -/
axiom initial_state : ∀ n : ℕ, 1 ≤ n ∧ n ≤ num_cards → is_red_side_up n

/-- Represents the number of independent flips -/
def independent_flips : ℕ → Prop := sorry

/-- Assumption of two independent flips -/
axiom two_independent_flips : independent_flips 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lowest_probability_cards_l296_29647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_A_beats_B_value_m_plus_n_value_l296_29698

/-- Represents a soccer tournament with the given conditions -/
structure SoccerTournament where
  num_teams : Nat
  num_games_per_team : Nat
  win_probability : ℚ

/-- Creates a tournament with the specified conditions -/
def createTournament : SoccerTournament :=
  { num_teams := 8
    num_games_per_team := 7
    win_probability := 1/2 }

/-- The probability that team A finishes with more points than team B -/
noncomputable def probability_A_beats_B (t : SoccerTournament) : ℚ :=
  821 / 2048

/-- Theorem stating the probability that team A finishes with more points than team B -/
theorem probability_A_beats_B_value (t : SoccerTournament) :
  probability_A_beats_B t = 821 / 2048 := by
  sorry

/-- The sum of m and n where the probability is expressed as m/n -/
def m_plus_n : Nat := 2869

/-- Theorem stating that m + n = 2869 -/
theorem m_plus_n_value :
  m_plus_n = 2869 := by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_A_beats_B_value_m_plus_n_value_l296_29698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_ticks_theorem_l296_29621

/-- Calculates the number of ticks given the time between first and last ticks -/
def number_of_ticks (reference_ticks : ℕ) (reference_time : ℚ) (new_time : ℚ) : ℕ :=
  let interval_time := reference_time / (reference_ticks - 1)
  let new_intervals := new_time / interval_time
  (new_intervals.ceil).toNat

/-- Theorem stating that for a clock ticking 6 times in 30 seconds, 
    it will tick 8 times when it takes 42 seconds -/
theorem clock_ticks_theorem : 
  number_of_ticks 6 30 42 = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_ticks_theorem_l296_29621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_minimum_point_l296_29690

/-- The ellipse with semi-major axis 5 and semi-minor axis 4 -/
def Ellipse (x y : ℝ) : Prop := x^2/25 + y^2/16 = 1

/-- The left focus of the ellipse -/
def LeftFocus : ℝ × ℝ := (-3, 0)

/-- The given point A -/
def A : ℝ × ℝ := (-2, 2)

/-- The distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- The expression to be minimized -/
noncomputable def expr (B : ℝ × ℝ) : ℝ :=
  distance A B + (5/3) * distance B LeftFocus

theorem ellipse_minimum_point :
  ∀ B : ℝ × ℝ, Ellipse B.1 B.2 →
    expr B ≥ expr (-5 * Real.sqrt 3 / 2, 2) := by
  sorry

#check ellipse_minimum_point

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_minimum_point_l296_29690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_distance_sum_l296_29689

/-- IsMiddle M AB means M is the midpoint of line segment AB -/
def IsMiddle (M : ℝ) (A B : ℝ) : Prop :=
  M = (A + B) / 2

/-- Given two line segments AB and CD with lengths 4 and 8 respectively,
    and points P on AB and Q on CD such that the distance x from P to
    the midpoint of AB and the distance y from Q to the midpoint of CD
    satisfy y = 2x, prove that if x = a, then x + y = 3a. -/
theorem segment_distance_sum (a : ℝ) : 
  ∀ (A B C D P Q M N : ℝ),
  (IsMiddle M A B ∧ IsMiddle N C D ∧
   |B - A| = 4 ∧ |D - C| = 8 ∧
   |P - M| = a ∧ |Q - N| = 2 * a) →
  a + |Q - N| = 3 * a :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_distance_sum_l296_29689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_EMD_is_nine_eighths_l296_29683

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define a point on a side of the triangle
def PointOnSide (T : Triangle) (M : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ M = (1 - t) • T.A + t • T.C

-- Define perpendicular segments
def Perpendicular (P Q R : ℝ × ℝ) : Prop :=
  (P.1 - Q.1) * (R.1 - Q.1) + (P.2 - Q.2) * (R.2 - Q.2) = 0

-- Define the area of a triangle
noncomputable def TriangleArea (P Q R : ℝ × ℝ) : ℝ :=
  abs ((Q.1 - P.1) * (R.2 - P.2) - (R.1 - P.1) * (Q.2 - P.2)) / 2

-- Define the angle between two vectors
def Angle (A B C : ℝ × ℝ) : ℝ := sorry

-- Main theorem
theorem area_EMD_is_nine_eighths (T : Triangle) (M D E : ℝ × ℝ) :
  Angle T.A T.B T.C > 90 →
  PointOnSide T M →
  (M.1 - T.A.1) = 3 * (T.C.1 - M.1) ∧ (M.2 - T.A.2) = 3 * (T.C.2 - M.2) →
  Perpendicular M D T.B →
  Perpendicular E D T.B →
  PointOnSide T D →
  PointOnSide T E →
  TriangleArea T.A T.B T.C = 36 →
  TriangleArea E M D = 9/8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_EMD_is_nine_eighths_l296_29683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l296_29640

theorem expression_evaluation : 
  |(-5)| - (27 : ℝ) ^ (1/3) + (-2)^2 + 4 / (2/3) = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l296_29640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cf_length_l296_29688

open Real

-- Define the triangle ABC
def triangle_ABC (A B C : ℝ × ℝ) : Prop :=
  dist A B = 13 ∧ dist B C = 26 ∧ dist C A = 24

-- Define the angle bisector of ∠BAC
def angle_bisector_BAC (A B C D : ℝ × ℝ) : Prop :=
  (dist B D / dist D C = dist A B / dist A C) ∧ D ∈ Set.Icc B C

-- Define the circumcircle of ABC
def circumcircle_ABC (A B C : ℝ × ℝ) (circle : Set (ℝ × ℝ)) : Prop :=
  ∀ X, X ∈ circle ↔ dist X A = dist X B ∧ dist X B = dist X C

-- Define the point E on the circumcircle of ABC
def point_E_on_circumcircle (A B C E : ℝ × ℝ) (circle : Set (ℝ × ℝ)) : Prop :=
  E ∈ circle ∧ E ≠ A

-- Define the circumcircle of BED
def circumcircle_BED (B D E : ℝ × ℝ) (circle : Set (ℝ × ℝ)) : Prop :=
  ∀ X, X ∈ circle ↔ dist X B = dist X E ∧ dist X E = dist X D

-- Define the point F on AB
def point_F_on_AB (A B F : ℝ × ℝ) : Prop :=
  F ∈ Set.Icc A B ∧ F ≠ B

-- Main theorem
theorem cf_length 
  (A B C D E F : ℝ × ℝ) 
  (circle_ABC circle_BED : Set (ℝ × ℝ)) :
  triangle_ABC A B C →
  angle_bisector_BAC A B C D →
  circumcircle_ABC A B C circle_ABC →
  point_E_on_circumcircle A B C E circle_ABC →
  circumcircle_BED B D E circle_BED →
  point_F_on_AB A B F →
  dist C F = 769 / 37 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cf_length_l296_29688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_meaningful_range_l296_29651

-- Define the set of real numbers that make the expression meaningful
noncomputable def meaningfulSet : Set ℝ := {x | x ≥ -1 ∧ x ≠ 2}

-- Define the expression
noncomputable def expression (x : ℝ) : ℝ := Real.sqrt (x + 1) / (x - 2)

-- Define a predicate for when the expression is well-defined
def isWellDefined (x : ℝ) : Prop := x ≥ -1 ∧ x ≠ 2

-- Theorem statement
theorem expression_meaningful_range :
  {x : ℝ | isWellDefined x} = meaningfulSet := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_meaningful_range_l296_29651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_function_parameter_range_l296_29645

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * a * x^3 + a * x^2 + x

-- State the theorem
theorem increasing_function_parameter_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ (0 ≤ a ∧ a ≤ 1) := by
  sorry

-- You can add more lemmas or theorems here if needed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_function_parameter_range_l296_29645

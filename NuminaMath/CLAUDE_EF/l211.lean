import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_purchase_change_l211_21131

/-- The change received when purchasing fruit -/
theorem fruit_purchase_change (a : ℝ) (h : a ≤ 30) : 
  100 - 3 * a = 100 - 3 * a := by
  let selling_price := a
  let quantity := 3
  let payment := 100
  let total_cost := selling_price * quantity
  let change := payment - total_cost
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fruit_purchase_change_l211_21131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_ellipse_focus_l211_21110

-- Define the parabola
noncomputable def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x

-- Define the ellipse
noncomputable def ellipse (x y : ℝ) : Prop := x^2/6 + y^2/2 = 1

-- Define the focus of the parabola
noncomputable def parabola_focus (p : ℝ) : ℝ × ℝ := (p/2, 0)

-- Define the right focus of the ellipse
noncomputable def ellipse_right_focus : ℝ × ℝ := (2, 0)

-- Theorem statement
theorem parabola_ellipse_focus (p : ℝ) : 
  (∀ x y, parabola p x y ∨ ellipse x y) → 
  parabola_focus p = ellipse_right_focus → 
  p = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_ellipse_focus_l211_21110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiple_formulas_exist_l211_21118

def sequenceA (n : ℕ) : ℤ := if n % 2 = 0 then -1 else 1

def formula1 (n : ℕ) : ℤ := if n % 2 = 0 then -1 else 1

def formula2 (n : ℕ) : ℤ := (-1)^(n+1)

noncomputable def formula3 (n : ℕ) : ℝ := Real.cos ((n + 1 : ℝ) * Real.pi)

theorem multiple_formulas_exist :
  (∀ n, sequenceA n = formula1 n) ∧
  (∀ n, sequenceA n = formula2 n) ∧
  (∀ n, (sequenceA n : ℝ) = formula3 n) ∧
  (∃ n, formula1 n ≠ formula2 n ∨ (formula1 n : ℝ) ≠ formula3 n ∨ (formula2 n : ℝ) ≠ formula3 n) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiple_formulas_exist_l211_21118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_sum_l211_21155

noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 + x^2 + x + 4/3

theorem odd_function_sum (a b : ℝ) : 
  (∀ x, f (x + a) + b = -(f (-x + a) + b)) → a + b = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_sum_l211_21155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_performances_proof_l211_21150

/-- The number of singers --/
def num_singers : ℕ := 8

/-- The number of singers per show --/
def singers_per_show : ℕ := 4

/-- The minimum number of performances --/
def min_performances : ℕ := 14

/-- The number of times each pair of singers perform together --/
def pair_performances : ℕ := 3

/-- Helper function to represent the number of joint performances for two singers --/
def number_of_joint_performances (i j m : ℕ) : ℕ := sorry

theorem minimum_performances_proof :
  ∀ (m : ℕ),
    (∀ (i j : ℕ), i < num_singers → j < num_singers → i ≠ j →
      (number_of_joint_performances i j m) = pair_performances) →
    m ≥ min_performances :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_performances_proof_l211_21150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_phi_l211_21196

theorem intersection_point_phi (φ : ℝ) : 
  (|φ| < π / 2) →
  (Real.cos (π / 3 - π / 6) = Real.sin (2 * π / 3 + φ)) →
  φ = -π / 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_phi_l211_21196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mars_moon_weight_ratio_l211_21137

/-- Represents the composition of a celestial body -/
structure CelestialComposition where
  iron : ℝ
  carbon : ℝ
  other : ℝ
  sum_to_one : iron + carbon + other = 1

/-- Represents a celestial body -/
structure CelestialBody where
  weight : ℝ
  composition : CelestialComposition

/-- The moon's properties -/
def moon : CelestialBody where
  weight := 250
  composition := {
    iron := 0.5
    carbon := 0.2
    other := 0.3
    sum_to_one := by ring
  }

/-- Mars' properties -/
def mars : CelestialBody where
  weight := moon.weight * 2
  composition := moon.composition

theorem mars_moon_weight_ratio :
  mars.weight / moon.weight = 2 := by
  simp [mars, moon]
  ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mars_moon_weight_ratio_l211_21137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marblesLeftPercentage_l211_21186

/-- The percentage of marbles Gilda has left after giving away to her friends and brother -/
def marblesLeft : ℝ :=
  let initialMarbles := 100
  let afterPedro := initialMarbles * (1 - 0.30)
  let afterEbony := afterPedro * (1 - 0.15)
  let afterJimmy := afterEbony * (1 - 0.20)
  let afterMarco := afterJimmy * (1 - 0.10)
  afterMarco

theorem marblesLeftPercentage :
  abs (marblesLeft - 42.84) < 0.01 := by
  sorry

#eval marblesLeft

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marblesLeftPercentage_l211_21186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l211_21154

-- Define the hyperbola
noncomputable def is_hyperbola (x y a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ x^2 / a^2 - y^2 / b^2 = 1

-- Define the eccentricity
noncomputable def eccentricity (a c : ℝ) : ℝ := c / a

-- Define the foci condition
noncomputable def foci_condition (P F₁ F₂ : ℝ × ℝ) : Prop :=
  4 * ‖(P.1 - F₁.1, P.2 - F₁.2) + (P.1 - F₂.1, P.2 - F₂.2)‖ ≥ 3 * ‖(F₁.1 - F₂.1, F₁.2 - F₂.2)‖

theorem hyperbola_eccentricity_range (a b : ℝ) (P F₁ F₂ : ℝ × ℝ) :
  is_hyperbola P.1 P.2 a b →
  foci_condition P F₁ F₂ →
  ∃ c, c > a ∧ 1 < eccentricity a c ∧ eccentricity a c ≤ 4/3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l211_21154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_factorial_with_1993_prefix_l211_21172

theorem exists_factorial_with_1993_prefix : ∃ n : ℕ, ∃ k : ℕ, n! = 1993 * 10^k + k ∧ k < 10^6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_factorial_with_1993_prefix_l211_21172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_vehicles_quotient_l211_21166

/-- Represents the maximum number of vehicles that can pass a sensor in one hour -/
def N : ℕ := 4000

/-- The length of each vehicle in meters -/
def vehicle_length : ℝ := 5

/-- The safety rule function: distance between vehicles in vehicle lengths -/
noncomputable def safety_distance (speed : ℝ) : ℝ := ⌈speed / 20⌉

/-- The total length of a vehicle unit (vehicle + safety distance) in meters -/
noncomputable def unit_length (speed : ℝ) : ℝ := vehicle_length * (1 + safety_distance speed)

/-- The number of vehicles passing the sensor per hour at a given speed -/
noncomputable def vehicles_per_hour (speed : ℝ) : ℝ := (speed * 1000) / (unit_length speed)

theorem max_vehicles_quotient :
  (N / 10 : ℕ) = 400 ∧ 
  ∀ speed : ℝ, speed > 0 → vehicles_per_hour speed ≤ N := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_vehicles_quotient_l211_21166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zoo_assignment_l211_21157

-- Define the enclosures and animals
inductive Enclosure : Type
| one | two | three | four | five

inductive Animal : Type
| giraffe | monkey | rhinoceros | lion | seal

-- Define the concept of neighboring enclosures
def neighbors : Enclosure → Enclosure → Prop :=
  sorry -- Implementation details omitted for brevity

-- Define the number of sides for each enclosure
def sides : Enclosure → ℕ :=
  sorry -- Implementation details omitted for brevity

-- Define which enclosure has a pool
def has_pool : Enclosure → Prop :=
  sorry -- Implementation details omitted for brevity

-- Define the assignment of animals to enclosures
def assignment : Animal → Enclosure :=
  sorry -- Implementation details omitted for brevity

-- State the conditions
axiom giraffe_enclosure : sides (assignment Animal.giraffe) = 5
axiom monkey_not_neighbor : ¬(neighbors (assignment Animal.monkey) (assignment Animal.rhinoceros)) ∧
                            ¬(neighbors (assignment Animal.monkey) (assignment Animal.giraffe))
axiom lion_monkey_sides : sides (assignment Animal.lion) = sides (assignment Animal.monkey)
axiom seal_has_pool : has_pool (assignment Animal.seal)

-- State the theorem
theorem zoo_assignment :
  assignment Animal.giraffe = Enclosure.three ∧
  assignment Animal.monkey = Enclosure.one ∧
  assignment Animal.rhinoceros = Enclosure.five ∧
  assignment Animal.lion = Enclosure.two ∧
  assignment Animal.seal = Enclosure.four :=
by
  sorry -- Proof omitted for brevity


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zoo_assignment_l211_21157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l211_21108

/-- The parabola y^2 = 4x -/
def parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- A point on the parabola -/
structure PointOnParabola where
  x : ℝ
  y : ℝ
  on_parabola : parabola x y

/-- The point P -/
def P : ℝ × ℝ := (11, 0)

/-- The slope of the line through P -/
def m : ℝ := 1

/-- The area of triangle PMN given M and N -/
noncomputable def triangle_area (M N : PointOnParabola) : ℝ :=
  let slope := (M.y - P.2) / (M.x - P.1)
  let d := |slope * P.1 - P.2| / Real.sqrt (1 + slope^2)
  let mn_length := Real.sqrt ((M.x - N.x)^2 + (M.y - N.y)^2)
  1/2 * d * mn_length

/-- The maximum area theorem -/
theorem max_triangle_area :
  ∃ (max_area : ℝ), max_area = 22 ∧
  ∀ (M N : PointOnParabola),
    (M.y - N.y) / (M.x - N.x) = m →
    triangle_area M N ≤ max_area :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l211_21108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_cube_condition_l211_21125

theorem perfect_cube_condition (n : ℕ) : 
  (n % 3 ≠ 0) ∧ (∃ (a : ℕ), 2^(n^2 - 10) + 2133 = a^3) → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_cube_condition_l211_21125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_and_distance_l211_21124

-- Define the circle
def my_circle (x y : ℝ) : Prop := (x - 1)^2 + (y + 2)^2 = 5

-- Define the line
def my_line (t : ℝ) : ℝ × ℝ := (1 + t, -1 + t)

-- Theorem statement
theorem line_intersects_circle_and_distance :
  (∃ t₁ t₂ : ℝ, t₁ ≠ t₂ ∧ 
    my_circle ((my_line t₁).1) ((my_line t₁).2) ∧ 
    my_circle ((my_line t₂).1) ((my_line t₂).2)) ∧
  (∃ A B : ℝ × ℝ, 
    my_circle A.1 A.2 ∧ 
    my_circle B.1 B.2 ∧ 
    (∃ t₁ t₂ : ℝ, my_line t₁ = A ∧ my_line t₂ = B) ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 18) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_and_distance_l211_21124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plot_length_is_approx_70_25_l211_21169

/-- Represents the dimensions and fencing costs of a rectangular plot -/
structure PlotData where
  breadth : ℝ
  northCost : ℝ
  eastCost : ℝ
  southCost : ℝ
  westCost : ℝ
  totalCost : ℝ
  terrainCharge : ℝ

/-- Calculates the length of a rectangular plot based on given conditions -/
def calculatePlotLength (data : PlotData) : ℝ :=
  data.breadth + 10

/-- Theorem stating that the length of the plot is approximately 70.25 meters -/
theorem plot_length_is_approx_70_25 (data : PlotData)
  (h1 : data.northCost = 26.5)
  (h2 : data.eastCost = 32)
  (h3 : data.southCost = 22)
  (h4 : data.westCost = 30)
  (h5 : data.totalCost = 7500)
  (h6 : data.terrainCharge = 0.05)
  (h7 : (1 + data.terrainCharge) * ((calculatePlotLength data + data.breadth) * (data.northCost + data.southCost) / 2 +
        (calculatePlotLength data + data.breadth) * (data.eastCost + data.westCost) / 2) = data.totalCost) :
  ∃ ε > 0, |calculatePlotLength data - 70.25| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plot_length_is_approx_70_25_l211_21169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_non_intersecting_division_l211_21141

/-- A rectangle in a 2D plane --/
structure Rectangle where
  x : ℝ
  y : ℝ
  width : ℝ
  height : ℝ
  width_pos : width > 0
  height_pos : height > 0

/-- A point in a 2D plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- The center of a rectangle --/
noncomputable def center (r : Rectangle) : Point :=
  { x := r.x + r.width / 2,
    y := r.y + r.height / 2 }

/-- Check if a point is inside a rectangle --/
def isInside (p : Point) (r : Rectangle) : Prop :=
  r.x ≤ p.x ∧ p.x ≤ r.x + r.width ∧ r.y ≤ p.y ∧ p.y ≤ r.y + r.height

/-- Check if a line segment intersects a rectangle --/
def intersects (p1 p2 : Point) (r : Rectangle) : Prop :=
  ∃ t : ℝ, 0 < t ∧ t < 1 ∧
    let x := p1.x + t * (p2.x - p1.x)
    let y := p1.y + t * (p2.y - p1.y)
    isInside { x := x, y := y } r

/-- The main theorem --/
theorem exists_non_intersecting_division :
  ∃ (n : ℕ) (original : Rectangle) (division : Fin n → Rectangle),
    (∀ i, isInside (center (division i)) original) ∧
    (∃ i j, i ≠ j ∧
      ∀ k, k ≠ i ∧ k ≠ j →
        ¬intersects (center (division i)) (center (division j)) (division k)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_non_intersecting_division_l211_21141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_and_odd_l211_21102

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (2^x - 1) / (2^x + 1)

-- Theorem statement
theorem f_increasing_and_odd :
  (∀ x y : ℝ, x < y → f x < f y) ∧
  (∀ x : ℝ, f (-x) = -f x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_and_odd_l211_21102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_detergent_ranking_l211_21183

/-- Represents the size of a detergent box -/
inductive Size
| S
| M
| L
| XL

/-- Cost of a detergent box -/
noncomputable def cost : Size → ℝ
| Size.S => 1
| Size.M => 1.6
| Size.L => 2.24
| Size.XL => 2.8

/-- Quantity of detergent in a box -/
noncomputable def quantity : Size → ℝ
| Size.S => 3.75
| Size.M => 10.5
| Size.L => 7.5
| Size.XL => 15

/-- Cost per unit of detergent -/
noncomputable def costPerUnit (s : Size) : ℝ := cost s / quantity s

/-- Checks if one size is a better buy than another -/
def isBetterBuy (s1 s2 : Size) : Prop := costPerUnit s1 < costPerUnit s2

theorem detergent_ranking : 
  isBetterBuy Size.M Size.XL ∧ 
  isBetterBuy Size.XL Size.S ∧ 
  isBetterBuy Size.S Size.L :=
by sorry

#check detergent_ranking

end NUMINAMATH_CALUDE_ERRORFEEDBACK_detergent_ranking_l211_21183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_line_l211_21129

-- Define the circle
def my_circle (x y : ℝ) : Prop := (x - 3)^2 + (y + 1)^2 = 1

-- Define the line
def my_line (x y : ℝ) : Prop := 3*x + 4*y = 0

-- Theorem statement
theorem circle_tangent_to_line :
  -- The circle has center (3, -1)
  (∀ x y : ℝ, my_circle x y ↔ ((x - 3)^2 + (y + 1)^2 = 1)) ∧
  -- The circle is tangent to the line 3x + 4y = 0
  (∃ x y : ℝ, my_circle x y ∧ my_line x y) ∧
  (∀ x y : ℝ, my_circle x y → my_line x y → 
    ∀ x' y' : ℝ, my_line x' y' → ((x - x')^2 + (y - y')^2 ≥ 1)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_to_line_l211_21129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_range_l211_21109

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then |Real.log x| else 2 * x + 6

-- State the theorem
theorem abc_range (a b c : ℝ) (h1 : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h2 : f a = f b ∧ f b = f c) : 
  -3 < a * b * c ∧ a * b * c ≤ 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_range_l211_21109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_archie_marbles_l211_21122

theorem archie_marbles (initial_marbles : ℕ) (street_loss_percent : ℚ) (final_marbles : ℕ) :
  initial_marbles = 100 →
  street_loss_percent = 60 / 100 →
  final_marbles = 20 →
  let remaining_after_street := initial_marbles - (street_loss_percent * ↑initial_marbles).floor
  let sewer_loss_fraction := (remaining_after_street - final_marbles) / remaining_after_street
  sewer_loss_fraction = 1 / 2 := by
  intro h1 h2 h3
  simp [h1, h2, h3]
  norm_num
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_archie_marbles_l211_21122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_2n_plus_1_l211_21193

/-- An arithmetic sequence with non-zero terms satisfying a specific condition -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- Changed to rational numbers for computability
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  nonzero : ∀ n, a n ≠ 0
  condition : ∀ n, (a (n + 1))^2 = a (n + 2) + a n

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (seq.a 1 + seq.a n) * n / 2

/-- Main theorem: Sum of first 2n+1 terms equals 4n+2 -/
theorem sum_2n_plus_1 (seq : ArithmeticSequence) (n : ℕ) :
  sum_n seq (2 * n + 1) = 4 * n + 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_2n_plus_1_l211_21193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_equal_tangents_line_l211_21128

/-- Circle C1 with equation x^2 + y^2 - 4x - 6y + 9 = 0 -/
def C1 (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 6*y + 9 = 0

/-- Circle C2 with equation x^2 + y^2 + 2x + 2y + 1 = 0 -/
def C2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 2*y + 1 = 0

/-- The line representing points P where |PA| = |PB| -/
def equal_tangents_line (x y : ℝ) : Prop := 3*x + 4*y - 4 = 0

/-- The distance from the origin (0, 0) to a point (x, y) -/
noncomputable def distance_from_origin (x y : ℝ) : ℝ := Real.sqrt (x^2 + y^2)

/-- Theorem stating that the minimum distance from the origin to the line of equal tangents is 4/5 -/
theorem min_distance_to_equal_tangents_line :
  ∃ (x y : ℝ), equal_tangents_line x y ∧
  (∀ (x' y' : ℝ), equal_tangents_line x' y' →
    distance_from_origin x y ≤ distance_from_origin x' y') ∧
  distance_from_origin x y = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_equal_tangents_line_l211_21128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l211_21198

theorem tan_alpha_value (α : ℝ) 
  (h1 : Real.sin (α + π/2) = 1/3) 
  (h2 : α > 0) 
  (h3 : α < π/2) : 
  Real.tan α = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l211_21198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2010_equals_neg_reciprocal_l211_21165

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (1 + x) / (1 - x)

-- Define the sequence f_k recursively
noncomputable def f_k : ℕ → (ℝ → ℝ)
  | 0 => λ x => x  -- Base case for n = 0
  | 1 => f
  | (n + 1) => λ x => f (f_k n x)

-- State the theorem
theorem f_2010_equals_neg_reciprocal (x : ℝ) (h : x ≠ 0) : f_k 2010 x = -1 / x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2010_equals_neg_reciprocal_l211_21165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_client_fraction_is_eleven_fifteenths_l211_21123

/-- Represents the payment scenario for Baylor's freelance work -/
structure PaymentScenario where
  initial_balance : ℚ
  first_client_fraction : ℚ
  third_client_multiplier : ℚ
  final_balance : ℚ

/-- Calculates the fraction of the second client's payment compared to the first client's payment -/
noncomputable def second_client_fraction (scenario : PaymentScenario) : ℚ :=
  let first_payment := scenario.initial_balance * scenario.first_client_fraction
  let x := (scenario.final_balance - scenario.initial_balance - first_payment * (1 + scenario.third_client_multiplier)) /
           (first_payment * (1 + scenario.third_client_multiplier))
  x

/-- Theorem stating that the fraction of the second client's payment is 11/15 -/
theorem second_client_fraction_is_eleven_fifteenths (scenario : PaymentScenario)
  (h1 : scenario.initial_balance = 4000)
  (h2 : scenario.first_client_fraction = 1/2)
  (h3 : scenario.third_client_multiplier = 2)
  (h4 : scenario.final_balance = 18400) :
  second_client_fraction scenario = 11/15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_client_fraction_is_eleven_fifteenths_l211_21123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_zero_l211_21140

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.exp x

-- State the theorem
theorem tangent_slope_at_zero :
  (deriv f) 0 = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_zero_l211_21140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_in_nickels_l211_21144

/-- Represents the number of quarters -/
def q : ℤ := sorry

/-- Alice's quarters -/
def alice_quarters (q : ℤ) : ℤ := 7 * q - 3

/-- Bob's quarters -/
def bob_quarters (q : ℤ) : ℤ := 3 * q + 7

/-- Conversion rate from quarters to nickels -/
def quarter_to_nickel : ℤ := 2

/-- Theorem stating the difference in Alice and Bob's money in nickels -/
theorem difference_in_nickels (q : ℤ) :
  (alice_quarters q - bob_quarters q) * quarter_to_nickel = 8 * q - 20 :=
by
  -- Expand the definitions
  simp [alice_quarters, bob_quarters, quarter_to_nickel]
  -- Perform algebraic simplification
  ring

end NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_in_nickels_l211_21144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_positive_difference_l211_21177

def geometric_sequence (a₁ r : ℝ) (n : ℕ) : ℝ := a₁ * r ^ (n - 1)

def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1 : ℝ)

def sequence_C (n : ℕ) : ℝ := geometric_sequence 3 3 n

def sequence_D (n : ℕ) : ℝ := arithmetic_sequence 15 15 n

def valid_C_terms : Set ℝ := {x | ∃ n : ℕ, sequence_C n = x ∧ x ≤ 450}

def valid_D_terms : Set ℝ := {x | ∃ n : ℕ, sequence_D n = x ∧ x ≤ 450}

theorem least_positive_difference :
  ∃ (c : ℝ) (d : ℝ), c ∈ valid_C_terms ∧ d ∈ valid_D_terms ∧
    3 = |c - d| ∧ ∀ (c' : ℝ) (d' : ℝ), c' ∈ valid_C_terms → d' ∈ valid_D_terms → |c' - d'| ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_positive_difference_l211_21177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l211_21139

-- Define the speed in km/hr
noncomputable def train_speed : ℝ := 54

-- Define the time to cross the pole in seconds
noncomputable def crossing_time : ℝ := 9

-- Define the conversion factor from km/hr to m/s
noncomputable def km_per_hr_to_m_per_s : ℝ := 1000 / 3600

-- Theorem statement
theorem train_length_calculation :
  let speed_m_per_s := train_speed * km_per_hr_to_m_per_s
  let length := speed_m_per_s * crossing_time
  length = 135 := by
  -- Unfold the definitions
  unfold train_speed crossing_time km_per_hr_to_m_per_s
  -- Perform the calculation
  norm_num
  -- Close the proof
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l211_21139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_inner_square_l211_21156

/-- Predicate to check if four points form a square -/
def is_square (A B C D : ℝ × ℝ) : Prop := sorry

/-- Function to calculate the side length between two points -/
def side_length (A B : ℝ × ℝ) : ℝ := sorry

/-- Predicate to check if one square is inside another -/
def is_inside (E F G H A B C D : ℝ × ℝ) : Prop := sorry

/-- Predicate to check if a point is on a line segment -/
def is_on_segment (E A B : ℝ × ℝ) : Prop := sorry

/-- Function to calculate the distance between two points -/
def distance (B E : ℝ × ℝ) : ℝ := sorry

/-- Function to calculate the area of a square given its four vertices -/
def area (E F G H : ℝ × ℝ) : ℝ := sorry

/-- Given a square ABCD with side length 10 and a smaller square EFGH inside it
    where E is on AB and BE = 2, prove that the area of EFGH is 100 - 16√6 -/
theorem area_of_inner_square (A B C D E F G H : ℝ × ℝ) : 
  is_square A B C D →
  side_length A B = 10 →
  is_square E F G H →
  is_inside E F G H A B C D →
  is_on_segment E A B →
  distance B E = 2 →
  area E F G H = 100 - 16 * Real.sqrt 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_inner_square_l211_21156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l211_21171

/-- Calculates the speed of a train in km/h given its length, the platform length, and the time taken to cross the platform. -/
noncomputable def trainSpeed (trainLength : ℝ) (platformLength : ℝ) (timeToCross : ℝ) : ℝ :=
  let totalDistance := trainLength + platformLength
  let speedInMetersPerSecond := totalDistance / timeToCross
  3.6 * speedInMetersPerSecond

/-- Theorem stating that a train of length 250 m crossing a platform of length 150.03 m in 20 seconds has a speed of approximately 72.0054 km/h. -/
theorem train_speed_calculation :
  let trainLength : ℝ := 250
  let platformLength : ℝ := 150.03
  let timeToCross : ℝ := 20
  abs (trainSpeed trainLength platformLength timeToCross - 72.0054) < 0.0001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l211_21171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_intervals_cos_2x0_value_l211_21151

open Real

noncomputable def f (x : ℝ) : ℝ := 
  sin x ^ 2 + 2 * sqrt 3 * sin x * cos x + 
  sin (x + π/4) * sin (x - π/4)

-- Theorem for the increasing intervals of f
theorem f_increasing_intervals (k : ℤ) :
  StrictMonoOn f (Set.Icc (k * π - π/6) (k * π + π/3)) := by sorry

-- Theorem for the value of cos(2x₀) when f(x₀) = 0
theorem cos_2x0_value (x₀ : ℝ) (h1 : f x₀ = 0) (h2 : 0 ≤ x₀) (h3 : x₀ ≤ π/2) :
  cos (2 * x₀) = (3 * sqrt 5 + 1) / 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_intervals_cos_2x0_value_l211_21151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bees_direction_at_15_feet_l211_21152

-- Define the bee's position type
structure BeePosition where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the bee's movement pattern
structure BeePattern where
  dx : ℝ
  dy : ℝ
  dz : ℝ

-- Define the bees' initial positions and patterns
def beeA_initial : BeePosition := ⟨0, 0, 0⟩
def beeB_initial : BeePosition := ⟨0, 0, 0⟩
def beeA_pattern : BeePattern := ⟨1, 1, 2⟩
def beeB_pattern : BeePattern := ⟨-1.5, -1, 0⟩

-- Function to calculate bee position after n cycles
def bee_position (initial : BeePosition) (pattern : BeePattern) (n : ℕ) : BeePosition :=
  ⟨initial.x + n * pattern.dx, initial.y + n * pattern.dy, initial.z + n * pattern.dz⟩

-- Function to calculate distance between two positions
noncomputable def distance (p1 p2 : BeePosition) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2)

-- Theorem statement
theorem bees_direction_at_15_feet :
  ∃ n : ℕ,
    let posA := bee_position beeA_initial beeA_pattern n
    let posB := bee_position beeB_initial beeB_pattern n
    distance posA posB = 15 ∧
    (n % 3 = 1 ∨ n % 3 = 2) -- Bee A moving east
    ∧ (n % 2 = 1) -- Bee B moving west
    := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bees_direction_at_15_feet_l211_21152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_students_in_college_l211_21190

/-- Given information about two classes in a college, prove the total number of students --/
theorem total_students_in_college 
  (class_a_boy_girl_ratio : ℚ) 
  (class_a_girls : ℕ) 
  (class_b_boy_girl_ratio : ℚ) 
  (class_b_total : ℕ) : ℕ :=
by
  -- Assumptions
  have h1 : class_a_boy_girl_ratio = 5 / 7 := by sorry
  have h2 : class_a_girls = 140 := by sorry
  have h3 : class_b_boy_girl_ratio = 3 / 5 := by sorry
  have h4 : class_b_total = 280 := by sorry

  -- Calculate total students in Class A
  let class_a_boys : ℕ := (5 * class_a_girls) / 7
  let class_a_total : ℕ := class_a_boys + class_a_girls

  -- Calculate total students in college
  let total_students : ℕ := class_a_total + class_b_total

  exact total_students

-- Example usage (commented out to avoid compilation issues)
-- #eval total_students_in_college (5/7) 140 (3/5) 280

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_students_in_college_l211_21190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circles_radii_l211_21146

/-- An isosceles triangle with base 24 and height 18 -/
structure IsoscelesTriangle where
  base : ℝ
  height : ℝ
  isBase24 : base = 24
  isHeight18 : height = 18

/-- A semicircle inscribed in the triangle with diameter along the base -/
noncomputable def semicircleRadius (t : IsoscelesTriangle) : ℝ := t.base / 2

/-- A full circle inscribed in the triangle touching the two equal sides and the base -/
noncomputable def fullCircleRadius (t : IsoscelesTriangle) : ℝ :=
  let sideLength := Real.sqrt (t.height ^ 2 + (t.base / 2) ^ 2)
  let semiperimeter := sideLength + sideLength + t.base
  (t.base * t.height / 2) / (semiperimeter / 2)

theorem inscribed_circles_radii (t : IsoscelesTriangle) :
  semicircleRadius t = 12 ∧ fullCircleRadius t = 216 / (6 * (Real.sqrt 13 + 2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circles_radii_l211_21146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_tiles_18gon_l211_21174

/-- A pentagon with given side length and angles -/
structure Pentagon where
  side_length : ℝ
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  angle4 : ℝ
  angle5 : ℝ

/-- A regular 18-sided polygon -/
structure Regular18Gon where
  side_length : ℝ

/-- Predicate to check if a tiling covers without gaps or overlaps -/
def covers_without_gaps_or_overlaps (tiling : Fin 18 → Pentagon) (g : Regular18Gon) : Prop :=
  sorry

/-- Theorem stating that 18 pentagonal tiles can cover a regular 18-gon -/
theorem pentagon_tiles_18gon (c : ℝ) (h : c > 0) :
  ∃ (p : Pentagon) (g : Regular18Gon),
    p.side_length = c ∧
    g.side_length = c ∧
    p.angle1 = 60 ∧
    p.angle2 = 160 ∧
    p.angle3 = 80 ∧
    p.angle4 = 100 ∧
    p.angle5 = 140 ∧
    (∃ (tiling : Fin 18 → Pentagon),
      (∀ i, tiling i = p) ∧
      (covers_without_gaps_or_overlaps tiling g)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_tiles_18gon_l211_21174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dried_grapes_weight_l211_21147

/-- Calculates the weight of dried grapes from fresh grapes -/
theorem dried_grapes_weight
  (fresh_weight : ℝ)
  (fresh_water_content : ℝ)
  (dried_water_content : ℝ)
  (h1 : fresh_weight = 30)
  (h2 : fresh_water_content = 0.9)
  (h3 : dried_water_content = 0.2) :
  (fresh_weight * (1 - fresh_water_content)) / (1 - dried_water_content) = 3.75 := by
  sorry

#check dried_grapes_weight

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dried_grapes_weight_l211_21147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_llama_play_area_theorem_l211_21113

/-- The area a llama can cover when tied to a rectangular shed --/
noncomputable def llamaPlayArea (shedLength : ℝ) (shedWidth : ℝ) (leashLength : ℝ) : ℝ :=
  (3/4) * Real.pi * leashLength^2 + 
  (1/4) * Real.pi * (leashLength - shedLength)^2 + 
  (1/4) * Real.pi * (leashLength - shedWidth)^2

/-- Theorem stating the area the llama can cover in the given scenario --/
theorem llama_play_area_theorem :
  llamaPlayArea 4 3 5 = 20 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_llama_play_area_theorem_l211_21113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_sin_sqrt3_cos_l211_21127

theorem min_value_sin_sqrt3_cos : 
  ∃ m : ℝ, m = -2 ∧ ∀ x : ℝ, Real.sin x + Real.sqrt 3 * Real.cos x ≥ m :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_sin_sqrt3_cos_l211_21127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_is_2pi_3_max_area_is_sqrt3_4_l211_21143

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the vectors m and n
def m (t : Triangle) : ℝ × ℝ := (t.b + t.c, t.a^2 + t.b * t.c)
def n (t : Triangle) : ℝ × ℝ := (t.b + t.c, -1)

-- Define the dot product
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Helper function for area
noncomputable def area (t : Triangle) : ℝ := 
  1/2 * t.b * t.c * Real.sin t.A

-- Theorem 1
theorem angle_A_is_2pi_3 (t : Triangle) 
  (h : dot_product (m t) (n t) = 0) : 
  t.A = 2 * Real.pi / 3 := by sorry

-- Theorem 2
theorem max_area_is_sqrt3_4 (t : Triangle) 
  (h : t.a = Real.sqrt 3) : 
  (∀ s : Triangle, s.a = Real.sqrt 3 → 
    area s ≤ Real.sqrt 3 / 4) ∧ 
  (∃ s : Triangle, s.a = Real.sqrt 3 ∧ 
    area s = Real.sqrt 3 / 4) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_is_2pi_3_max_area_is_sqrt3_4_l211_21143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_problem_l211_21170

/-- A circle in the plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A point in the plane -/
def Point := ℝ × ℝ

/-- The area of a circle -/
noncomputable def circleArea (c : Circle) : ℝ := Real.pi * c.radius^2

/-- The condition that a point lies on a circle -/
def onCircle (p : Point) (c : Circle) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

/-- The condition that a line is tangent to a circle at a point -/
def isTangent (p : Point) (c : Circle) : Prop :=
  onCircle p c ∧ ∀ q : Point, q ≠ p → ((q.1 - p.1)^2 + (q.2 - p.2)^2 < c.radius^2 ∨ (q.1 - p.1)^2 + (q.2 - p.2)^2 > c.radius^2)

theorem circle_area_problem (ω : Circle) (A B C : Point) :
  A = (4, 10) →
  B = (10, 8) →
  onCircle A ω →
  onCircle B ω →
  isTangent A ω →
  isTangent B ω →
  C.2 = 0 → -- C is on the x-axis
  (∃ t : ℝ, (1 - t) * A.1 + t * C.1 = (1 - t) * A.2 + t * C.2) → -- A and C are collinear
  (∃ t : ℝ, (1 - t) * B.1 + t * C.1 = (1 - t) * B.2 + t * C.2) → -- B and C are collinear
  circleArea ω = 100 * Real.pi / 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_problem_l211_21170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_phi_is_cone_l211_21142

/-- Spherical coordinates in 3D space -/
structure SphericalCoord where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

/-- A constant angle from the positive z-axis -/
def c : ℝ := 0  -- We define c as a constant, you can change this value

/-- The set of points satisfying φ = c in spherical coordinates -/
def constantPhiSet : Set SphericalCoord :=
  {p : SphericalCoord | p.φ = c}

/-- Definition of a cone with vertex at the origin -/
def isCone (S : Set SphericalCoord) : Prop :=
  ∃ (apex_angle : ℝ), ∀ p ∈ S, p.φ = apex_angle / 2

theorem constant_phi_is_cone :
  isCone constantPhiSet := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_phi_is_cone_l211_21142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercept_ratio_for_specific_lines_l211_21168

/-- Two lines with the same y-intercept -/
structure TwoLines where
  y_intercept : ℝ
  slope1 : ℝ
  slope2 : ℝ
  x_intercept1 : ℝ
  x_intercept2 : ℝ

/-- The ratio of x-intercepts for two lines with specific slopes and y-intercept -/
noncomputable def x_intercept_ratio (lines : TwoLines) : ℝ :=
  |lines.x_intercept1 / lines.x_intercept2|

/-- Theorem stating the ratio of x-intercepts for specific lines -/
theorem x_intercept_ratio_for_specific_lines :
  ∀ (lines : TwoLines),
  lines.y_intercept = 8 →
  lines.slope1 = 4 →
  lines.slope2 = 12 →
  x_intercept_ratio lines = 3 := by
  intro lines h1 h2 h3
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercept_ratio_for_specific_lines_l211_21168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l211_21194

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x + 1) / x

theorem domain_of_f :
  {x : ℝ | x ∈ Set.Ici (-1) ∧ x ≠ 0} = Set.Ioi (-1) ∪ Set.Ioi 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l211_21194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ant_meeting_q_value_l211_21192

-- Define the direction type
inductive Direction
| East | South | West | North

-- Define the ant's position type
structure Position where
  x : ℚ
  y : ℚ

-- Define the ant's path type
def AntPath := List Direction

-- Function to calculate new position after one move
def move (pos : Position) (dir : Direction) (q : ℚ) : Position :=
  match dir with
  | Direction.East  => ⟨pos.x + q, pos.y⟩
  | Direction.South => ⟨pos.x, pos.y - q⟩
  | Direction.West  => ⟨pos.x - q, pos.y⟩
  | Direction.North => ⟨pos.x, pos.y + q⟩

-- Function to calculate final position after following a path
def followPath (start : Position) (path : AntPath) (q : ℚ) : Position :=
  path.foldl (fun pos dir => move pos dir q) start

-- Main theorem
theorem ant_meeting_q_value
  (q : ℚ)
  (hq : q > 0)
  (path1 path2 : AntPath)
  (hpaths : path1 ≠ path2)
  (hmeeting : followPath ⟨0, 0⟩ path1 q = followPath ⟨0, 0⟩ path2 q) :
  q = 1 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ant_meeting_q_value_l211_21192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumradius_of_special_triangle_l211_21185

/-- A triangle with side lengths 8, 15, and 17 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a = 8
  hb : b = 15
  hc : c = 17
  right_angle : a^2 + b^2 = c^2

/-- The radius of the circumscribed circle of a right triangle -/
noncomputable def circumradius (t : RightTriangle) : ℝ := t.c / 2

/-- Theorem: The radius of the circumscribed circle of the given triangle is 17/2 -/
theorem circumradius_of_special_triangle (t : RightTriangle) : 
  circumradius t = 17 / 2 := by
  sorry

#check circumradius_of_special_triangle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumradius_of_special_triangle_l211_21185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_points_on_line_curve_is_line_l211_21136

theorem points_on_line :
  ∀ (t : ℝ), 
  (Real.cos t)^2 + (Real.sin t)^2 = 1 := by
  intro t
  exact Real.cos_sq_add_sin_sq t

theorem curve_is_line :
  ∀ (t : ℝ),
  let x := (Real.cos t)^2
  let y := (Real.sin t)^2
  x + y = 1 := by
  intro t
  exact points_on_line t

end NUMINAMATH_CALUDE_ERRORFEEDBACK_points_on_line_curve_is_line_l211_21136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_valid_number_l211_21167

def is_valid_number (n : ℕ) : Prop :=
  (n ≥ 100000000 ∧ n < 1000000000) ∧
  (∀ i j, i ≠ j → (Nat.digits 10 n).get? i ≠ (Nat.digits 10 n).get? j) ∧
  (∀ i, (Nat.digits 10 n).get? i ≠ some 0) ∧
  (∃ a b c : ℕ,
    a < 10 ∧ b < 10 ∧ c < 10 ∧
    Nat.sqrt n = a * 10000 + b * 1000 + a * 100 + b * 10 + c ∧
    a * 10 + b = c^3)

theorem unique_valid_number :
  ∃! n : ℕ, is_valid_number n ∧ n = 743816529 :=
by sorry

#check unique_valid_number

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_valid_number_l211_21167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_property_locus_of_circumcenters_locus_of_incenters_locus_of_orthocenters_l211_21134

-- Define the basic structures
structure Point where
  x : ℝ
  y : ℝ

structure Triangle where
  A : Point
  B : Point
  C : Point

structure Line where
  m : ℝ
  b : ℝ

-- Define the properties
def is_circumcenter (O : Point) (T : Triangle) : Prop := sorry
def is_incenter (O : Point) (T : Triangle) : Prop := sorry
def is_orthocenter (O : Point) (T : Triangle) : Prop := sorry

-- Define the construction
def construct_lines (O : Point) (T : Triangle) : (Line × Line × Line) := sorry
def intersection_points (l1 l2 l3 : Line) (T : Triangle) : Triangle := sorry

-- Main theorem
theorem center_property (ABC : Triangle) (O : Point) 
  (l1 l2 l3 : Line) (A_bar_B_bar_C_bar : Triangle) :
  (is_circumcenter O ABC ∨ is_incenter O ABC ∨ is_orthocenter O ABC) →
  (construct_lines O ABC = (l1, l2, l3)) →
  (intersection_points l1 l2 l3 ABC = A_bar_B_bar_C_bar) →
  (is_orthocenter O A_bar_B_bar_C_bar ∧ 
   is_circumcenter O A_bar_B_bar_C_bar ∧ 
   is_incenter O A_bar_B_bar_C_bar) := by
  sorry

-- Additional theorems for part (b)
theorem locus_of_circumcenters (O : Point) (ABC : Triangle) : Set Point := sorry

theorem locus_of_incenters (O : Point) (ABC : Triangle) : Set Point := sorry

theorem locus_of_orthocenters (O : Point) (ABC : Triangle) : Set Point := sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_property_locus_of_circumcenters_locus_of_incenters_locus_of_orthocenters_l211_21134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l211_21178

/-- The constant term in the expansion of (x + 1/x - 2)^5 -/
noncomputable def constantTerm (x : ℝ) : ℝ :=
  (x + 1/x - 2)^5

/-- Theorem stating that the constant term in the expansion of (x + 1/x - 2)^5 is -252 -/
theorem constant_term_expansion :
  ∃ (c : ℝ), ∀ (x : ℝ), x ≠ 0 →
    constantTerm x = c + x * (constantTerm x - c) / x ∧ c = -252 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_expansion_l211_21178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_indistinguishable_sets_l211_21145

theorem existence_of_indistinguishable_sets :
  ∃ (A B : Finset ℕ+) (p q : ℕ+),
    A.card = 10 ∧ B.card = 10 ∧
    A ≠ B ∧
    Nat.Prime p.val ∧ Nat.Prime q.val ∧ p ≠ q ∧
    A = {p, q} ∪ Finset.image (fun i => q ^ (i + 2)) (Finset.range 8) ∧
    B = {p * q, 1} ∪ Finset.image (fun i => q ^ (i + 2)) (Finset.range 8) ∧
    (∀ x y, x ∈ A → y ∈ A →
      Nat.gcd x.val y.val = Nat.gcd 
        (if x ∈ B then x.val else if x = p then (p * q).val else 1)
        (if y ∈ B then y.val else if y = p then (p * q).val else 1)) ∧
    (∀ x y, x ∈ A → y ∈ A →
      Nat.lcm x.val y.val = Nat.lcm
        (if x ∈ B then x.val else if x = p then (p * q).val else 1)
        (if y ∈ B then y.val else if y = p then (p * q).val else 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_indistinguishable_sets_l211_21145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lady_has_two_children_l211_21173

/-- Represents the number of children a lady has -/
def num_children : ℕ := sorry

/-- Represents the fact that at least one child is a boy -/
def has_boy : Prop := sorry

/-- Represents the probability of having all boys given that one is a boy -/
def prob_all_boys : ℝ := sorry

/-- Theorem stating that given the conditions, the lady must have two children -/
theorem lady_has_two_children 
  (h1 : num_children > 0)
  (h2 : has_boy)
  (h3 : prob_all_boys = 0.5) :
  num_children = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lady_has_two_children_l211_21173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_with_area_84_and_consecutive_sides_l211_21101

/-- Calculates the area of a triangle given its three sides using Heron's formula. -/
noncomputable def area_triangle (a b c : ℝ) : ℝ :=
  let s : ℝ := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

/-- A triangle with consecutive integer sides and area 84 has sides 13, 14, and 15. -/
theorem triangle_with_area_84_and_consecutive_sides :
  ∃ (x : ℕ), 
    (x > 1) ∧ 
    (area_triangle (x - 1 : ℝ) x (x + 1) = 84) ∧ 
    (x - 1 = 13 ∧ x = 14 ∧ x + 1 = 15) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_with_area_84_and_consecutive_sides_l211_21101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_attendees_l211_21121

theorem meeting_attendees (k : ℕ) : 
  (∃ (n : ℕ), 
    -- Total number of people is 12k
    (12 * k > 0) ∧ 
    -- Each person greets 3k+6 others
    (∀ person : Fin n, ∃ (greeted : Finset (Fin n)), greeted.card = 3 * k + 6) ∧ 
    -- For any two individuals, the number of people who greeted both of them is the same
    (∀ p q : Fin n, p ≠ q → ∃ (m : ℕ), ∀ r : Fin n, r ≠ p ∧ r ≠ q → 
      (∃ (greeted_p greeted_q : Finset (Fin n)), 
        greeted_p.card = 3 * k + 6 ∧ 
        greeted_q.card = 3 * k + 6 ∧ 
        (greeted_p ∩ greeted_q).card = m))) →
  k = 3 ∧ 12 * k = 36 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_attendees_l211_21121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_michael_jessica_meeting_time_l211_21133

/-- The time it takes for Michael and Jessica to meet given their initial conditions --/
noncomputable def meeting_time (initial_distance : ℝ) (speed_ratio : ℝ) (combined_speed : ℝ) (initial_bike_time : ℝ) (stop_time : ℝ) : ℝ :=
  let jessica_speed := combined_speed / (1 + speed_ratio)
  let michael_speed := jessica_speed * speed_ratio
  let initial_distance_covered := combined_speed * initial_bike_time
  let remaining_distance_after_initial := initial_distance - initial_distance_covered
  let distance_covered_during_stop := michael_speed * stop_time
  let final_remaining_distance := remaining_distance_after_initial - distance_covered_during_stop
  let final_bike_time := final_remaining_distance / combined_speed
  initial_bike_time + stop_time + final_bike_time

/-- Theorem stating that Michael and Jessica meet after approximately 21.33 minutes --/
theorem michael_jessica_meeting_time :
  ∃ ε > 0, |meeting_time 24 2 1.2 8 4 - 21.33| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_michael_jessica_meeting_time_l211_21133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_circle_distance_range_l211_21149

-- Define the line equation
def line_eq (x y : ℝ) : Prop := 3 * x + 4 * y + 3 = 0

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y + 1 = 0

-- Define a point on the line
def point_on_line (P : ℝ × ℝ) : Prop := line_eq P.1 P.2

-- Define a point on the circle
def point_on_circle (P : ℝ × ℝ) : Prop := circle_eq P.1 P.2

-- Define the distance between two points
noncomputable def distance (A B : ℝ × ℝ) : ℝ := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- Theorem statement
theorem tangent_line_circle_distance_range :
  ∀ P A B : ℝ × ℝ,
  point_on_line P →
  point_on_circle A →
  point_on_circle B →
  (∃ t : ℝ, distance P A = t ∧ distance P B = t) →
  Real.sqrt 3 < distance A B ∧ distance A B < 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_circle_distance_range_l211_21149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_unique_points_with_equal_perimeters_l211_21179

/-- Represents a half-line in a plane with origin O -/
structure HalfLine where
  angle : ℝ

/-- Represents a point on a half-line -/
structure PointOnHalfLine where
  hl : HalfLine
  distance : ℝ

/-- The perimeter of a triangle formed by the origin and two points -/
noncomputable def trianglePerimeter (a b : PointOnHalfLine) : ℝ :=
  a.distance + b.distance + Real.sqrt (a.distance^2 + b.distance^2 - 2*a.distance*b.distance*(Real.cos (a.hl.angle - b.hl.angle)))

/-- Theorem stating the existence and uniqueness of points A, B, C satisfying the perimeter condition -/
theorem exists_unique_points_with_equal_perimeters 
  (x y z : HalfLine) 
  (hxy : x.angle ≠ y.angle) 
  (hyz : y.angle ≠ z.angle) 
  (hzx : z.angle ≠ x.angle) 
  (p : ℝ) 
  (hp : p > 0) :
  ∃! (a b c : PointOnHalfLine), 
    a.hl = x ∧ b.hl = y ∧ c.hl = z ∧
    trianglePerimeter a b = 2*p ∧
    trianglePerimeter b c = 2*p ∧
    trianglePerimeter c a = 2*p := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_unique_points_with_equal_perimeters_l211_21179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_specific_sequence_l211_21182

def alternatingSequence (n : ℕ) : ℤ := 
  if n % 2 = 0 then -(n : ℤ) else n

def sequenceSum (n : ℕ) : ℤ := 
  (Finset.range n).sum (λ i => alternatingSequence (i + 1))

theorem sum_of_specific_sequence : 
  sequenceSum 101 = 51 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_specific_sequence_l211_21182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jello_mix_per_pound_l211_21176

/-- Proves that 1.5 tablespoons of jello mix are needed for every pound of water in James' bathtub jello experiment. -/
theorem jello_mix_per_pound
  (bathtub_capacity : ℝ)
  (cubic_feet_to_gallons : ℝ)
  (pounds_per_gallon : ℝ)
  (jello_mix_cost : ℝ)
  (total_spent : ℝ)
  (h1 : bathtub_capacity = 6)
  (h2 : cubic_feet_to_gallons = 7.5)
  (h3 : pounds_per_gallon = 8)
  (h4 : jello_mix_cost = 0.5)
  (h5 : total_spent = 270) :
  (total_spent / jello_mix_cost) / (bathtub_capacity * cubic_feet_to_gallons * pounds_per_gallon) = 1.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jello_mix_per_pound_l211_21176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_a2_plus_b2_l211_21106

theorem min_value_a2_plus_b2 (x₀ : ℝ) (a b : ℝ) 
  (h1 : x₀ ∈ Set.Icc (1/4 : ℝ) (Real.exp 1))
  (h2 : 2 * a * Real.sqrt x₀ + b - Real.exp (x₀ / 2) = 0) :
  (∀ a' b', ∃ x₀' ∈ Set.Icc (1/4 : ℝ) (Real.exp 1), 
    2 * a' * Real.sqrt x₀' + b' - Real.exp (x₀' / 2) = 0 
    → a' ^ 2 + b' ^ 2 ≥ Real.exp (3/4) / 4)
  ∧ (∃ a₀ b₀, ∃ x₀₀ ∈ Set.Icc (1/4 : ℝ) (Real.exp 1),
    2 * a₀ * Real.sqrt x₀₀ + b₀ - Real.exp (x₀₀ / 2) = 0 
    ∧ a₀ ^ 2 + b₀ ^ 2 = Real.exp (3/4) / 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_a2_plus_b2_l211_21106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_min_value_l211_21116

theorem sin_cos_min_value :
  (∀ x : ℝ, Real.sin x * Real.cos x ≥ -1/2) ∧ (∃ x : ℝ, Real.sin x * Real.cos x = -1/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_min_value_l211_21116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_subset_size_l211_21126

def is_valid_subset (S : Finset ℕ) : Prop :=
  ∀ x ∈ S, ∀ y ∈ S, x ≠ 4 * y ∧ y ≠ 4 * x

theorem max_subset_size :
  ∃ (S : Finset ℕ), 
    (∀ x ∈ S, 1 ≤ x ∧ x ≤ 150) ∧ 
    is_valid_subset S ∧
    S.card = 120 ∧
    (∀ T : Finset ℕ, (∀ x ∈ T, 1 ≤ x ∧ x ≤ 150) → is_valid_subset T → 
      T.card ≤ 120) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_subset_size_l211_21126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l211_21119

-- Define the hyperbola equation
def hyperbola_eq (x y : ℝ) : Prop := x^2 / 10 - y^2 / 2 = 1

-- Define the foci coordinates
noncomputable def foci : ℝ × ℝ := (2 * Real.sqrt 3, 0)

-- Define the imaginary semi-axis length
noncomputable def b : ℝ := Real.sqrt 2

-- Define the semi-focal distance
noncomputable def c : ℝ := 2 * Real.sqrt 3

-- Theorem statement
theorem hyperbola_properties :
  (∀ x y, hyperbola_eq x y → 
    (∃ t, foci = (t, 0) ∨ foci = (-t, 0))) ∧
  b = Real.sqrt 2 ∧
  c = 2 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l211_21119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_identity_l211_21197

theorem sin_identity (x : ℝ) 
  (h : Real.sin (x + π/6) = 1/3) : 
  Real.sin ((7*π)/6 - 2*x) = -7/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_identity_l211_21197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triathlon_average_speed_approx_5_l211_21114

/-- Represents a triathlon race with swimming, biking, and running segments. -/
structure Triathlon where
  swim_speed : ℝ
  bike_speed : ℝ
  run_speed : ℝ

/-- Calculates the average speed of a triathlon race. -/
noncomputable def average_speed (t : Triathlon) : ℝ :=
  let total_distance := 5  -- Assuming running segment is 1 unit, total is 5 units
  let total_time := 2 / t.swim_speed + 2 / t.bike_speed + 1 / t.run_speed
  total_distance / total_time

/-- Theorem stating that the average speed of the given triathlon is approximately 5 km/h. -/
theorem triathlon_average_speed_approx_5 :
  let t := Triathlon.mk 3 10 20
  ∃ ε > 0, |average_speed t - 5| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triathlon_average_speed_approx_5_l211_21114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lowest_possible_score_l211_21115

def test_count : ℕ := 5
def max_score : ℕ := 100
def target_average : ℕ := 85

def first_three_scores : List ℕ := [82, 88, 93]

theorem lowest_possible_score (scores : List ℕ) 
  (h1 : scores.length = 3)
  (h2 : ∀ s ∈ scores, s ≤ max_score)
  (h3 : scores = first_three_scores) :
  ∃ (x : ℕ), 
    x ≤ max_score ∧ 
    ((scores.sum + x + max_score : ℚ) / test_count) = target_average ∧
    ∀ y : ℕ, y < x → ((scores.sum + y + max_score : ℚ) / test_count) < target_average :=
by
  sorry

#eval first_three_scores.sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lowest_possible_score_l211_21115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_EFCD_is_187_5_l211_21107

/-- Represents a trapezoid ABCD with midpoints E and F -/
structure Trapezoid :=
  (AB : ℝ)
  (CD : ℝ)
  (altitude : ℝ)

/-- The area of quadrilateral EFCD in the given trapezoid -/
noncomputable def area_EFCD (t : Trapezoid) : ℝ :=
  ((t.CD + (t.AB + t.CD) / 2) * (t.altitude / 2)) / 2

/-- Theorem stating that the area of EFCD is 187.5 square units for the given trapezoid -/
theorem area_EFCD_is_187_5 (t : Trapezoid) 
    (h1 : t.AB = 10)
    (h2 : t.CD = 30)
    (h3 : t.altitude = 15) :
  area_EFCD t = 187.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_EFCD_is_187_5_l211_21107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trouser_sale_percent_decrease_l211_21160

/-- Calculates the percent decrease between two prices -/
noncomputable def percentDecrease (originalPrice salePrice : ℝ) : ℝ :=
  (originalPrice - salePrice) / originalPrice * 100

theorem trouser_sale_percent_decrease :
  percentDecrease 100 80 = 20 := by
  -- Unfold the definition of percentDecrease
  unfold percentDecrease
  -- Simplify the arithmetic
  simp [div_mul_eq_mul_div]
  -- Evaluate the expression
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trouser_sale_percent_decrease_l211_21160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_on_y_axis_fixed_points_roots_condition_l211_21175

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - (a + 1) * x - 4

-- Theorem 1: Vertex on y-axis iff a = -1
theorem vertex_on_y_axis (a : ℝ) : 
  (∃ y : ℝ, ∀ x : ℝ, f a x ≥ f a 0) ↔ a = -1 := by sorry

-- Theorem 2: Fixed points (0, -4) and (1, -5)
theorem fixed_points (a : ℝ) : 
  f a 0 = -4 ∧ f a 1 = -5 := by sorry

-- Theorem 3: Roots between (-1, 0) and (2, 3) iff a = 2
theorem roots_condition (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, -1 < x₁ ∧ x₁ < 0 ∧ 2 < x₂ ∧ x₂ < 3 ∧ 
   f a x₁ = 0 ∧ f a x₂ = 0) ↔ a = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_on_y_axis_fixed_points_roots_condition_l211_21175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l211_21112

/-- Represents the number of white balls in the urn -/
def n : ℕ := 19

/-- Represents the number of black balls in the urn -/
def k : ℕ := 6

/-- The total number of balls in the urn -/
def total_balls (n k : ℕ) : ℕ := n + k

/-- The probability of drawing a white ball first and a black ball second -/
noncomputable def probability (n k : ℕ) : ℚ :=
  (n : ℚ) / (total_balls n k) * (k : ℚ) / (total_balls n k - 1)

/-- The main theorem stating that n = 19 is the unique solution -/
theorem unique_solution (n k : ℕ) :
  (n ≥ 2) → (k ≥ 2) → (probability n k = n / 100) → (n = 19) :=
by sorry

/-- Verifies that the solution satisfies the conditions -/
example : probability n k = n / 100 ∧ n ≥ 2 ∧ k ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l211_21112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_function_l211_21161

-- Define the functions
noncomputable def f1 (x : ℝ) : ℝ := (1/2) * x
noncomputable def f2 (x : ℝ) : ℝ := -3 * x
noncomputable def f3 (x : ℝ) : ℝ := -x^2
noncomputable def f4 (x : ℝ) : ℝ := -1/x

-- Theorem statement
theorem increasing_function :
  (∀ x : ℝ, HasDerivAt f1 ((1/2) : ℝ) x) ∧
  (∃ x : ℝ, HasDerivAt f2 (-3 : ℝ) x) ∧
  (∃ x : ℝ, HasDerivAt f3 (-2*x) x) ∧
  (∃ x : ℝ, x ≠ 0 → HasDerivAt f4 (1/x^2) x) ∧
  (∀ x : ℝ, (1/2 : ℝ) > 0) ∧
  (∃ x : ℝ, (-3 : ℝ) ≤ 0) ∧
  (∃ x : ℝ, (-2*x) ≤ 0) ∧
  (∃ x : ℝ, x ≠ 0 → (1/x^2) ≤ 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_function_l211_21161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_empty_box_possible_l211_21138

/-- Represents the state of the three boxes -/
structure BoxState where
  box1 : ℕ
  box2 : ℕ
  box3 : ℕ

/-- Represents a single coin transfer operation -/
inductive CoinTransfer
  | transfer_1_2 (s : BoxState) : CoinTransfer
  | transfer_1_3 (s : BoxState) : CoinTransfer
  | transfer_2_1 (s : BoxState) : CoinTransfer
  | transfer_2_3 (s : BoxState) : CoinTransfer
  | transfer_3_1 (s : BoxState) : CoinTransfer
  | transfer_3_2 (s : BoxState) : CoinTransfer

/-- Defines a valid coin transfer operation -/
def valid_transfer : CoinTransfer → Prop
  | CoinTransfer.transfer_1_2 s => s.box1 > s.box2 ∧ s.box1 - s.box2 ≥ s.box2
  | CoinTransfer.transfer_1_3 s => s.box1 > s.box3 ∧ s.box1 - s.box3 ≥ s.box3
  | CoinTransfer.transfer_2_1 s => s.box2 > s.box1 ∧ s.box2 - s.box1 ≥ s.box1
  | CoinTransfer.transfer_2_3 s => s.box2 > s.box3 ∧ s.box2 - s.box3 ≥ s.box3
  | CoinTransfer.transfer_3_1 s => s.box3 > s.box1 ∧ s.box3 - s.box1 ≥ s.box1
  | CoinTransfer.transfer_3_2 s => s.box3 > s.box2 ∧ s.box3 - s.box2 ≥ s.box2

/-- Theorem: It's always possible to empty one box in a finite number of steps -/
theorem empty_box_possible (initial : BoxState) :
  ∃ (final : BoxState) (steps : List CoinTransfer),
    (final.box1 = 0 ∨ final.box2 = 0 ∨ final.box3 = 0) ∧
    (∀ t ∈ steps, valid_transfer t) ∧
    (steps.length < (↑(initial.box1 + initial.box2 + initial.box3) : ℕ∞)) :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_empty_box_possible_l211_21138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solutions_gg_equals_three_l211_21159

-- Define the function g
noncomputable def g (x : ℝ) : ℝ :=
  if -4 ≤ x ∧ x < 0 then x^3 + x^2 - 2*x
  else if 0 ≤ x ∧ x ≤ 4 then 2*x - 4
  else 0  -- Default value for x outside [-4, 4]

-- Theorem statement
theorem no_solutions_gg_equals_three :
  ∀ x : ℝ, -4 ≤ x ∧ x ≤ 4 → g (g x) ≠ 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solutions_gg_equals_three_l211_21159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l211_21162

noncomputable def z₁ (x y : ℝ) : ℂ := x + Real.sqrt 5 - y * Complex.I
noncomputable def z₂ (x y : ℝ) : ℂ := x - Real.sqrt 5 + y * Complex.I

def f (x y : ℝ) : ℝ := |2 * x - 3 * y - 12|

theorem min_value_of_f (x y : ℝ) :
  Complex.abs (z₁ x y) + Complex.abs (z₂ x y) = 6 →
  ∃ (min : ℝ), min = 12 - 6 * Real.sqrt 2 ∧ ∀ (a b : ℝ), f a b ≥ min :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l211_21162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_b_values_l211_21104

-- Define the function f
noncomputable def f (b : ℝ) (x : ℝ) : ℝ := b / (3 * x - 2)

-- Define the inverse function of f
noncomputable def f_inv (b : ℝ) (y : ℝ) : ℝ := 
  (b + 2 * y) / (3 * y)

-- State the theorem
theorem product_of_b_values : 
  ∃ (b₁ b₂ : ℝ), b₁ ≠ b₂ ∧ 
  (∀ (b : ℝ), f b 3 = f_inv b (b + 2) → b = b₁ ∨ b = b₂) ∧
  b₁ * b₂ = -28/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_b_values_l211_21104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_boarders_count_is_30_l211_21103

def new_boarders_count (initial_boarders : ℕ) (initial_ratio_boarders : ℕ) (initial_ratio_day : ℕ) 
  (final_ratio_boarders : ℕ) (final_ratio_day : ℕ) : ℕ :=
let initial_day_students := initial_boarders * initial_ratio_day / initial_ratio_boarders
let new_boarders := initial_day_students * final_ratio_boarders / final_ratio_day - initial_boarders
new_boarders

theorem new_boarders_count_is_30 :
  new_boarders_count 120 2 5 1 2 = 30 := by
  rfl

#eval new_boarders_count 120 2 5 1 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_boarders_count_is_30_l211_21103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_not_invertible_sum_values_l211_21158

theorem matrix_not_invertible_sum_values (x y z : ℝ) :
  let M : Matrix (Fin 3) (Fin 3) ℝ := !![x, y, z; y, z, x; z, x, y]
  ¬(IsUnit (Matrix.det M)) →
  (x / (y + z) + y / (x + z) + z / (x + y) = -3) ∨
  (x / (y + z) + y / (x + z) + z / (x + y) = 3/2) :=
by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_not_invertible_sum_values_l211_21158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_radii_ratio_is_two_thirds_l211_21163

/-- The radius of a sphere inscribed in a cube with edge length a -/
noncomputable def cube_inscribed_radius (a : ℝ) : ℝ := a / 2

/-- The radius of a sphere inscribed in a regular octahedron with edge length a -/
noncomputable def octahedron_inscribed_radius (a : ℝ) : ℝ := a * Real.sqrt 6 / 6

/-- The ratio of the inscribed sphere radii of a cube to a regular octahedron -/
noncomputable def inscribed_radii_ratio (a : ℝ) : ℝ :=
  cube_inscribed_radius a / octahedron_inscribed_radius a

theorem inscribed_radii_ratio_is_two_thirds (a : ℝ) (h : a > 0) :
  inscribed_radii_ratio a = 2 / 3 := by
  sorry

def product_m_n : ℕ := 2 * 3

#eval product_m_n

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_radii_ratio_is_two_thirds_l211_21163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pdf_is_derivative_of_cdf_l211_21184

noncomputable section

open Real

-- Define the cumulative distribution function F
noncomputable def F (x : ℝ) : ℝ :=
  if x ≤ 0 then 0
  else if x ≤ Real.pi/2 then Real.sin x
  else 1

-- Define the probability density function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then 0
  else if x ≤ Real.pi/2 then Real.cos x
  else 0

-- Theorem statement
theorem pdf_is_derivative_of_cdf (x : ℝ) (h : x ≠ 0 ∧ x ≠ Real.pi/2) : 
  deriv F x = f x := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pdf_is_derivative_of_cdf_l211_21184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_transformation_l211_21189

-- Define the original cosine function
noncomputable def original_function (x : ℝ) : ℝ := Real.cos x

-- Define the translation operation
def translate (f : ℝ → ℝ) (a : ℝ) : ℝ → ℝ := λ x ↦ f (x - a)

-- Define the horizontal scaling operation
def scale_x (f : ℝ → ℝ) (k : ℝ) : ℝ → ℝ := λ x ↦ f (k * x)

-- Define the resulting function after translation and scaling
noncomputable def resulting_function (x : ℝ) : ℝ := Real.cos (2 * x - Real.pi / 3)

-- Theorem statement
theorem cosine_transformation :
  ∀ x : ℝ, (scale_x (translate original_function (Real.pi / 3)) (1 / 2)) x = resulting_function x :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_transformation_l211_21189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_less_than_b_l211_21120

theorem a_less_than_b : ∀ a b : ℝ, a = (-3 : ℝ)^(0 : ℝ) ∧ b = (1/3 : ℝ)^(-1 : ℝ) → a < b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_less_than_b_l211_21120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_triangle_perimeter_l211_21180

/-- Represents a right triangle -/
def RightTriangle (triangle : Set ℝ × Set ℝ) : Prop := sorry

/-- Represents that two triangles are similar -/
def SimilarTriangles (triangle1 triangle2 : Set ℝ × Set ℝ) : Prop := sorry

/-- Calculates the area of a triangle -/
noncomputable def Area (triangle : Set ℝ × Set ℝ) : ℝ := sorry

/-- Returns the length of the hypotenuse of a right triangle -/
noncomputable def Hypotenuse (triangle : Set ℝ × Set ℝ) : ℝ := sorry

/-- Calculates the perimeter of a triangle -/
noncomputable def Perimeter (triangle : Set ℝ × Set ℝ) : ℝ := sorry

/-- Given two similar right triangles with areas 10 and 250 square inches respectively,
    and the smaller triangle having a hypotenuse of 10 inches,
    the perimeter of the larger triangle is 20√5 + 50√2 inches. -/
theorem larger_triangle_perimeter (small_triangle large_triangle : Set ℝ × Set ℝ)
    (h_similar : SimilarTriangles small_triangle large_triangle)
    (h_right : RightTriangle small_triangle ∧ RightTriangle large_triangle)
    (h_area_small : Area small_triangle = 10)
    (h_area_large : Area large_triangle = 250)
    (h_hypotenuse_small : Hypotenuse small_triangle = 10) :
    Perimeter large_triangle = 20 * Real.sqrt 5 + 50 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_triangle_perimeter_l211_21180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimals_count_l211_21105

theorem repeating_decimals_count : 
  (Finset.filter (fun n : ℕ => 1 ≤ n ∧ n ≤ 11 ∧ ¬(12 ∣ n * 5^100000)) (Finset.range 12)).card = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimals_count_l211_21105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_location_l211_21153

theorem max_value_location (f : ℝ → ℝ) (a b : ℝ) (h : a ≤ b) :
  Differentiable ℝ f → ∃ x ∈ Set.Icc a b, ∀ y ∈ Set.Icc a b, f y ≤ f x →
    x = a ∨ x = b ∨ (x ∈ Set.Ioo a b ∧ deriv f x = 0) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_location_l211_21153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_from_cubic_roots_l211_21117

theorem triangle_area_from_cubic_roots : ∃ (a b c : ℝ),
  (a^3 - 3*a^2 + 4*a - 8/5 = 0) ∧
  (b^3 - 3*b^2 + 4*b - 8/5 = 0) ∧
  (c^3 - 3*c^2 + 4*c - 8/5 = 0) ∧
  (a ≠ b) ∧ (b ≠ c) ∧ (a ≠ c) ∧
  (∃ (A : ℝ), A = Real.sqrt 6 / 4 ∧ 
    A^2 = (a + b + c) / 4 * ((a + b - c) / 2) * ((b + c - a) / 2) * ((c + a - b) / 2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_from_cubic_roots_l211_21117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_curve_with_inclination_l211_21187

/-- Defines a curve in parametric form -/
noncomputable def curve (θ : ℝ) : ℝ × ℝ :=
  (3 * Real.cos θ, 4 * Real.sin θ)

/-- Checks if a point is on the curve -/
def on_curve (p : ℝ × ℝ) : Prop :=
  ∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ Real.pi ∧ curve θ = p

/-- Checks if a point forms a line with the origin at an inclination of π/4 -/
def has_inclination_pi_4 (p : ℝ × ℝ) : Prop :=
  p.2 / p.1 = 1

/-- The main theorem -/
theorem point_on_curve_with_inclination (p : ℝ × ℝ) :
  on_curve p ∧ has_inclination_pi_4 p → p = (12/5, 12/5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_curve_with_inclination_l211_21187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frog_jump_probability_l211_21188

/-- Represents a point on the coordinate plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines the rectangle boundaries -/
def isInRectangle (p : Point) : Prop :=
  0 ≤ p.x ∧ p.x ≤ 6 ∧ 0 ≤ p.y ∧ p.y ≤ 6

/-- Defines when a point is on the vertical sides of the rectangle -/
def isOnVerticalSide (p : Point) : Prop :=
  (p.x = 0 ∨ p.x = 6) ∧ 0 ≤ p.y ∧ p.y ≤ 6

/-- Represents a single jump of the frog -/
inductive Jump
  | up
  | down
  | left
  | right

/-- Applies a jump to a point -/
def applyJump (p : Point) (j : Jump) : Point :=
  match j with
  | Jump.up => ⟨p.x, p.y + 1⟩
  | Jump.down => ⟨p.x, p.y - 1⟩
  | Jump.left => ⟨p.x - 1, p.y⟩
  | Jump.right => ⟨p.x + 1, p.y⟩

/-- Represents a sequence of jumps -/
def JumpSequence := List Jump

/-- Applies a sequence of jumps to a point -/
def applyJumpSequence (p : Point) : JumpSequence → Point
  | [] => p
  | (j::js) => applyJumpSequence (applyJump p j) js

/-- Defines when a jump sequence is valid (ends on the rectangle boundary) -/
def isValidJumpSequence (start : Point) (js : JumpSequence) : Prop :=
  let endPoint := applyJumpSequence start js
  isInRectangle endPoint ∧ (isOnVerticalSide endPoint ∨ endPoint.y = 0 ∨ endPoint.y = 6)

/-- The probability of ending on a vertical side -/
noncomputable def probabilityOnVerticalSide (start : Point) : ℝ :=
  sorry -- Definition of probability calculation

/-- The main theorem to prove -/
theorem frog_jump_probability :
  probabilityOnVerticalSide ⟨2, 3⟩ = 7/9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_frog_jump_probability_l211_21188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_queue_waiting_time_theorem_l211_21199

/-- Represents the waiting time statistics for a queue with Slowpokes and Quickies -/
structure QueueStats (m n : ℕ) (a b : ℝ) where
  min_time : ℝ
  max_time : ℝ
  expected_time : ℝ

/-- Calculates the waiting time statistics for a queue -/
noncomputable def calculate_queue_stats (m n : ℕ) (a b : ℝ) : QueueStats m n a b :=
  { min_time := a * (n.choose 2 : ℝ) + a * (m : ℝ) * (n : ℝ) + b * (m.choose 2 : ℝ),
    max_time := a * (n.choose 2 : ℝ) + b * (m : ℝ) * (n : ℝ) + b * (m.choose 2 : ℝ),
    expected_time := ((m + n).choose 2 : ℝ) * (b * (m : ℝ) + a * (n : ℝ)) / ((m + n) : ℝ) }

/-- Theorem stating the correctness of the queue waiting time calculations -/
theorem queue_waiting_time_theorem (m n : ℕ) (a b : ℝ) :
  let stats := calculate_queue_stats m n a b
  (stats.min_time = a * (n.choose 2 : ℝ) + a * (m : ℝ) * (n : ℝ) + b * (m.choose 2 : ℝ)) ∧
  (stats.max_time = a * (n.choose 2 : ℝ) + b * (m : ℝ) * (n : ℝ) + b * (m.choose 2 : ℝ)) ∧
  (stats.expected_time = ((m + n).choose 2 : ℝ) * (b * (m : ℝ) + a * (n : ℝ)) / ((m + n) : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_queue_waiting_time_theorem_l211_21199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_is_seven_l211_21132

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ -2 then (-4/2) * x - 3
  else if x ≤ -1 then -x - 3
  else if x ≤ 1 then 2 * x
  else if x ≤ 2 then -x + 3
  else 2 * x - 3

def g (x : ℝ) : ℝ := x + 2

theorem intersection_sum_is_seven :
  ∃ (S : Finset ℝ), (∀ x ∈ S, f x = g x) ∧ (S.sum id = 7) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_is_seven_l211_21132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_tangent_length_is_correct_l211_21195

/-- The line from which the tangent is drawn -/
def line (x : ℝ) : ℝ := x + 1

/-- The circle to which the tangent is drawn -/
def circle_eq (x y : ℝ) : Prop := (x - 3)^2 + (y + 2)^2 = 1

/-- The minimum length of the tangent line -/
noncomputable def min_tangent_length : ℝ := Real.sqrt 14 / 2

/-- Theorem stating the minimum length of the tangent line -/
theorem min_tangent_length_is_correct :
  ∃ (x₀ y₀ : ℝ), y₀ = line x₀ ∧
  ∀ (x y : ℝ), y = line x →
  (∃ (xt yt : ℝ), circle_eq xt yt ∧
    (x - xt)^2 + (y - yt)^2 ≥ min_tangent_length^2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_tangent_length_is_correct_l211_21195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_swindlers_in_cabinet_l211_21181

/-- Represents the property that among any n ministers, there is at least one swindler -/
def AtLeastOneSwindlerInEveryN (ministers : Finset ℕ) (swindlers : Finset ℕ) (n : ℕ) : Prop :=
  ∀ (subset : Finset ℕ), subset ⊆ ministers → subset.card = n → (subset ∩ swindlers).Nonempty

theorem min_swindlers_in_cabinet :
  ∀ (ministers swindlers : Finset ℕ),
    ministers.card = 100 →
    swindlers ⊆ ministers →
    AtLeastOneSwindlerInEveryN ministers swindlers 10 →
    swindlers.card ≥ 91 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_swindlers_in_cabinet_l211_21181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_is_fourteenth_term_l211_21191

-- Define the sequence
noncomputable def a (n : ℕ) : ℝ := Real.sqrt (6 * n - 3)

-- State the theorem
theorem nine_is_fourteenth_term :
  ∃ n : ℕ, n = 14 ∧ a n = 9 := by
  -- Proof goes here
  sorry

#check nine_is_fourteenth_term

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_is_fourteenth_term_l211_21191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l211_21135

theorem trigonometric_identity (x y : ℝ) 
  (h1 : Real.sin x / Real.cos y + Real.sin y / Real.cos x = 2)
  (h2 : Real.cos x / Real.sin y + Real.cos y / Real.sin x = 3) :
  Real.tan x / Real.tan y + Real.tan y / Real.tan x = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l211_21135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_correct_condition_equivalent_to_a_geq_one_l211_21164

-- Define the function f(x) = x ln x
noncomputable def f (x : ℝ) : ℝ := x * Real.log x

-- Define the tangent line at (1,0)
def tangent_line (x : ℝ) : ℝ := x - 1

-- Theorem 1: The tangent line to f(x) at (1,0) has the equation y = x - 1
theorem tangent_line_correct :
  ∀ x : ℝ, tangent_line x = (deriv f 1) * (x - 1) + f 1 := by
  sorry

-- Theorem 2: For any x > 0, x ln(ax) ≥ x - a if and only if a ≥ 1
theorem condition_equivalent_to_a_geq_one :
  ∀ a : ℝ, (∀ x : ℝ, x > 0 → x * Real.log (a * x) ≥ x - a) ↔ a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_correct_condition_equivalent_to_a_geq_one_l211_21164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l211_21111

variable (x₁ x₂ x₃ x₄ x₅ x₆ x₇ : ℝ)

def equation1 (x₁ x₂ x₃ x₄ x₅ x₆ x₇ : ℝ) : Prop := 10 * x₁ + 3 * x₂ + 4 * x₃ + x₄ + x₅ = 0
def equation2 (x₁ x₂ x₃ x₄ x₅ x₆ x₇ : ℝ) : Prop := 11 * x₂ + 2 * x₃ + 2 * x₄ + 3 * x₅ + x₆ = 0
def equation3 (x₁ x₂ x₃ x₄ x₅ x₆ x₇ : ℝ) : Prop := 15 * x₃ + 4 * x₄ + 5 * x₅ + 4 * x₆ + x₇ = 0
def equation4 (x₁ x₂ x₃ x₄ x₅ x₆ x₇ : ℝ) : Prop := 2 * x₁ + x₂ - 3 * x₃ + 12 * x₄ - 3 * x₅ + x₆ + x₇ = 0
def equation5 (x₁ x₂ x₃ x₄ x₅ x₆ x₇ : ℝ) : Prop := 6 * x₁ - 5 * x₂ + 3 * x₃ - x₄ + 17 * x₅ + x₆ = 0
def equation6 (x₁ x₂ x₃ x₄ x₅ x₆ x₇ : ℝ) : Prop := 3 * x₁ + 2 * x₂ - 3 * x₃ + 4 * x₄ + x₅ - 16 * x₆ + 2 * x₇ = 0
def equation7 (x₁ x₂ x₃ x₄ x₅ x₆ x₇ : ℝ) : Prop := 4 * x₁ - 8 * x₂ + x₃ + x₄ - 3 * x₅ + 19 * x₇ = 0

theorem unique_solution (h1 : equation1 x₁ x₂ x₃ x₄ x₅ x₆ x₇)
                        (h2 : equation2 x₁ x₂ x₃ x₄ x₅ x₆ x₇)
                        (h3 : equation3 x₁ x₂ x₃ x₄ x₅ x₆ x₇)
                        (h4 : equation4 x₁ x₂ x₃ x₄ x₅ x₆ x₇)
                        (h5 : equation5 x₁ x₂ x₃ x₄ x₅ x₆ x₇)
                        (h6 : equation6 x₁ x₂ x₃ x₄ x₅ x₆ x₇)
                        (h7 : equation7 x₁ x₂ x₃ x₄ x₅ x₆ x₇) :
  x₁ = 0 ∧ x₂ = 0 ∧ x₃ = 0 ∧ x₄ = 0 ∧ x₅ = 0 ∧ x₆ = 0 ∧ x₇ = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l211_21111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_odd_with_five_prime_factors_l211_21130

def has_exactly_five_prime_factors (n : ℕ) : Prop :=
  ∃ p₁ p₂ p₃ p₄ p₅ : ℕ,
    Nat.Prime p₁ ∧ Nat.Prime p₂ ∧ Nat.Prime p₃ ∧ Nat.Prime p₄ ∧ Nat.Prime p₅ ∧
    p₁ < p₂ ∧ p₂ < p₃ ∧ p₃ < p₄ ∧ p₄ < p₅ ∧
    n = p₁ * p₂ * p₃ * p₄ * p₅

theorem smallest_odd_with_five_prime_factors :
  ∀ n : ℕ,
    n % 2 = 1 ∧
    has_exactly_five_prime_factors n ∧
    (∀ p : ℕ, Nat.Prime p → p ∣ n → p ≠ 11) →
    n ≥ 23205 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_odd_with_five_prime_factors_l211_21130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_part_distance_is_15_l211_21100

/-- Represents a bicycle journey with specific conditions -/
structure BicycleJourney where
  speed : ℚ
  initial_ride_time : ℚ
  rest_time : ℚ
  final_distance : ℚ
  total_time : ℚ

/-- Calculates the distance covered in the second part of the journey -/
def second_part_distance (j : BicycleJourney) : ℚ :=
  let total_ride_time := j.total_time - j.rest_time
  let first_part_distance := j.speed * j.initial_ride_time
  let third_part_time := j.final_distance / j.speed
  let second_part_time := total_ride_time - j.initial_ride_time - third_part_time
  j.speed * second_part_time

/-- Theorem stating that for the given journey conditions, the second part distance is 15 miles -/
theorem second_part_distance_is_15 (j : BicycleJourney) 
  (h1 : j.speed = 10)
  (h2 : j.initial_ride_time = 1/2)
  (h3 : j.rest_time = 1/2)
  (h4 : j.final_distance = 20)
  (h5 : j.total_time = 9/2) :
  second_part_distance j = 15 := by
  sorry

#eval second_part_distance {
  speed := 10,
  initial_ride_time := 1/2,
  rest_time := 1/2,
  final_distance := 20,
  total_time := 9/2
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_part_distance_is_15_l211_21100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_palindrome_count_l211_21148

/-- Represents a time on a 12-hour digital clock using 24-hour format. -/
structure ClockTime where
  hours : Nat
  minutes : Nat
  hour_valid : hours > 0 ∧ hours ≤ 12
  minute_valid : minutes < 60

/-- Checks if a given clock time forms a palindrome. -/
def is_palindrome (t : ClockTime) : Bool :=
  let h := t.hours
  let m := t.minutes
  if h < 10 then
    h = m % 10 ∧ m / 10 = 0
  else
    h / 10 = m % 10 ∧ h % 10 = m / 10

/-- Counts the number of palindromes on the clock. -/
def count_palindromes : Nat :=
  (List.range 12).foldl (fun acc h =>
    acc + (List.range 60).foldl (fun inner_acc m =>
      if is_palindrome ⟨h + 1, m, by sorry, by sorry⟩ then
        inner_acc + 1
      else
        inner_acc
    ) 0
  ) 0

/-- The main theorem stating the number of palindromes on the clock. -/
theorem palindrome_count : count_palindromes = 57 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_palindrome_count_l211_21148

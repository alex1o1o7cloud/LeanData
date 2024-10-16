import Mathlib

namespace NUMINAMATH_CALUDE_first_day_earnings_10_l439_43928

/-- A sequence of 5 numbers where each subsequent number is 4 more than the previous one -/
def IceCreamEarnings (first_day : ℕ) : Fin 5 → ℕ
  | ⟨0, _⟩ => first_day
  | ⟨n + 1, h⟩ => IceCreamEarnings first_day ⟨n, Nat.lt_trans n.lt_succ_self h⟩ + 4

/-- The theorem stating that if the sum of the sequence is 90, the first day's earnings were 10 -/
theorem first_day_earnings_10 :
  (∃ (first_day : ℕ), (Finset.sum Finset.univ (IceCreamEarnings first_day)) = 90) →
  (∃ (first_day : ℕ), (Finset.sum Finset.univ (IceCreamEarnings first_day)) = 90 ∧ first_day = 10) :=
by sorry


end NUMINAMATH_CALUDE_first_day_earnings_10_l439_43928


namespace NUMINAMATH_CALUDE_expected_replacement_seeds_l439_43905

theorem expected_replacement_seeds :
  let germination_prob : ℝ := 0.9
  let initial_seeds : ℕ := 1000
  let replacement_per_failure : ℕ := 2
  let non_germination_prob : ℝ := 1 - germination_prob
  let expected_non_germinating : ℝ := initial_seeds * non_germination_prob
  let expected_replacements : ℝ := expected_non_germinating * replacement_per_failure
  expected_replacements = 200 := by sorry

end NUMINAMATH_CALUDE_expected_replacement_seeds_l439_43905


namespace NUMINAMATH_CALUDE_intersection_point_on_diagonal_l439_43912

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a line in 3D space -/
structure Line3D where
  point : Point3D
  direction : Point3D

/-- Represents a plane in 3D space -/
structure Plane3D where
  point : Point3D
  normal : Point3D

/-- Check if a point lies on a line -/
def pointOnLine (p : Point3D) (l : Line3D) : Prop :=
  sorry

/-- Check if a point lies on a plane -/
def pointOnPlane (p : Point3D) (plane : Plane3D) : Prop :=
  sorry

/-- Check if two lines intersect -/
def linesIntersect (l1 l2 : Line3D) : Prop :=
  sorry

theorem intersection_point_on_diagonal (A B C D E F G H P : Point3D)
  (AB : Line3D) (BC : Line3D) (CD : Line3D) (DA : Line3D)
  (EF : Line3D) (GH : Line3D) (AC : Line3D)
  (ABC : Plane3D) (ADC : Plane3D) :
  pointOnLine E AB →
  pointOnLine F BC →
  pointOnLine G CD →
  pointOnLine H DA →
  linesIntersect EF GH →
  pointOnLine P EF →
  pointOnLine P GH →
  pointOnPlane E ABC →
  pointOnPlane F ABC →
  pointOnPlane G ADC →
  pointOnPlane H ADC →
  pointOnLine P AC :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_on_diagonal_l439_43912


namespace NUMINAMATH_CALUDE_cos_minus_sin_seventeen_fourths_pi_l439_43906

theorem cos_minus_sin_seventeen_fourths_pi : 
  Real.cos (-17 * Real.pi / 4) - Real.sin (-17 * Real.pi / 4) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_minus_sin_seventeen_fourths_pi_l439_43906


namespace NUMINAMATH_CALUDE_factorization_identities_l439_43968

theorem factorization_identities (x y : ℝ) : 
  (x^3 + 6*x^2 + 9*x = x*(x + 3)^2) ∧ 
  (16*x^2 - 9*y^2 = (4*x - 3*y)*(4*x + 3*y)) ∧ 
  ((3*x+y)^2 - (x-3*y)*(3*x+y) = 2*(3*x+y)*(x+2*y)) := by
  sorry

end NUMINAMATH_CALUDE_factorization_identities_l439_43968


namespace NUMINAMATH_CALUDE_unique_division_problem_l439_43933

theorem unique_division_problem :
  ∀ (a b : ℕ),
  (a ≥ 1 ∧ a ≤ 9 ∧ b ≥ 1 ∧ b ≤ 9) →
  (∃ (p : ℕ), 111111 * a = 1111 * b * 233 + p) →
  (∃ (q : ℕ), 11111 * a = 111 * b * 233 + (q - 1000)) →
  (a = 7 ∧ b = 3) := by
sorry

end NUMINAMATH_CALUDE_unique_division_problem_l439_43933


namespace NUMINAMATH_CALUDE_last_locker_opened_l439_43907

/-- Represents the state of a locker (open or closed) -/
inductive LockerState
| Open
| Closed

/-- Represents the corridor with lockers -/
def Corridor := Fin 512 → LockerState

/-- The initial state of the corridor with all lockers closed -/
def initialCorridor : Corridor := fun _ => LockerState.Closed

/-- Represents a single pass of opening lockers -/
def openLockersPass (c : Corridor) (start : Nat) (step : Nat) : Corridor :=
  fun n => if (n.val - start) % step = 0 then LockerState.Open else c n

/-- Represents the process of opening lockers in multiple passes -/
def openLockers (c : Corridor) : Corridor :=
  -- Implementation details omitted
  sorry

/-- The theorem stating that the last locker to be opened is 342 -/
theorem last_locker_opened (c : Corridor) :
  openLockers initialCorridor (⟨341, sorry⟩ : Fin 512) = LockerState.Closed ∧
  openLockers initialCorridor (⟨342, sorry⟩ : Fin 512) = LockerState.Open :=
by sorry

end NUMINAMATH_CALUDE_last_locker_opened_l439_43907


namespace NUMINAMATH_CALUDE_least_five_digit_congruent_to_5_mod_15_l439_43966

theorem least_five_digit_congruent_to_5_mod_15 : ∃ (n : ℕ), 
  (n ≥ 10000 ∧ n < 100000) ∧ 
  n % 15 = 5 ∧
  (∀ m : ℕ, (m ≥ 10000 ∧ m < 100000) ∧ m % 15 = 5 → n ≤ m) ∧
  n = 10010 :=
sorry

end NUMINAMATH_CALUDE_least_five_digit_congruent_to_5_mod_15_l439_43966


namespace NUMINAMATH_CALUDE_quadratic_form_bounds_l439_43977

theorem quadratic_form_bounds (x y : ℝ) (h : x^2 + x*y + y^2 = 3) :
  1 ≤ x^2 - x*y + y^2 ∧ x^2 - x*y + y^2 ≤ 9 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_bounds_l439_43977


namespace NUMINAMATH_CALUDE_equation_solution_exists_l439_43981

theorem equation_solution_exists (a : ℝ) : 
  (∃ x : ℝ, 4^x - a * 2^x - a + 3 = 0) ↔ a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_exists_l439_43981


namespace NUMINAMATH_CALUDE_wall_height_is_ten_l439_43943

-- Define the dimensions of the rooms
def livingRoomSide : ℝ := 40
def bedroomLength : ℝ := 12
def bedroomWidth : ℝ := 10

-- Define the number of walls to be painted in each room
def livingRoomWalls : ℕ := 3
def bedroomWalls : ℕ := 4

-- Define the total area to be painted
def totalAreaToPaint : ℝ := 1640

-- Theorem statement
theorem wall_height_is_ten :
  let livingRoomPerimeter := livingRoomSide * 4
  let livingRoomPaintPerimeter := livingRoomPerimeter - livingRoomSide
  let bedroomPerimeter := 2 * (bedroomLength + bedroomWidth)
  let totalPerimeterToPaint := livingRoomPaintPerimeter + bedroomPerimeter
  totalAreaToPaint / totalPerimeterToPaint = 10 := by
  sorry

end NUMINAMATH_CALUDE_wall_height_is_ten_l439_43943


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_integers_from_neg3_to_6_l439_43958

def integers_range : List ℤ := List.range 10 |>.map (λ i => i - 3)

theorem arithmetic_mean_of_integers_from_neg3_to_6 :
  (integers_range.sum : ℚ) / integers_range.length = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_integers_from_neg3_to_6_l439_43958


namespace NUMINAMATH_CALUDE_payment_difference_l439_43920

/-- Represents the cost and distribution of a pizza -/
structure PizzaOrder where
  totalSlices : ℕ
  plainCost : ℚ
  mushroomCost : ℚ
  oliveCost : ℚ

/-- Calculates the total cost of the pizza -/
def totalCost (p : PizzaOrder) : ℚ :=
  p.plainCost + p.mushroomCost + p.oliveCost

/-- Calculates the cost per slice -/
def costPerSlice (p : PizzaOrder) : ℚ :=
  totalCost p / p.totalSlices

/-- Calculates the cost for Liam's portion -/
def liamCost (p : PizzaOrder) : ℚ :=
  costPerSlice p * (2 * p.totalSlices / 3 + 2)

/-- Calculates the cost for Emily's portion -/
def emilyCost (p : PizzaOrder) : ℚ :=
  costPerSlice p * 2

/-- The main theorem stating the difference in payment -/
theorem payment_difference (p : PizzaOrder) 
  (h1 : p.totalSlices = 12)
  (h2 : p.plainCost = 12)
  (h3 : p.mushroomCost = 3)
  (h4 : p.oliveCost = 4) :
  liamCost p - emilyCost p = 152 / 12 := by
  sorry

#eval (152 : ℚ) / 12  -- This should evaluate to 12.67

end NUMINAMATH_CALUDE_payment_difference_l439_43920


namespace NUMINAMATH_CALUDE_min_toothpicks_to_remove_for_given_figure_l439_43945

/-- Represents a figure made of toothpicks -/
structure ToothpickFigure where
  total_toothpicks : ℕ
  triangles : ℕ
  squares : ℕ

/-- The minimum number of toothpicks to remove to eliminate all shapes -/
def min_toothpicks_to_remove (figure : ToothpickFigure) : ℕ := sorry

/-- Theorem stating the minimum number of toothpicks to remove -/
theorem min_toothpicks_to_remove_for_given_figure :
  ∃ (figure : ToothpickFigure),
    figure.total_toothpicks = 40 ∧
    figure.triangles > 20 ∧
    figure.squares = 10 ∧
    min_toothpicks_to_remove figure = 20 := by
  sorry

end NUMINAMATH_CALUDE_min_toothpicks_to_remove_for_given_figure_l439_43945


namespace NUMINAMATH_CALUDE_parabola_hyperbola_focus_l439_43970

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2/6 - y^2/3 = 1

-- Define the right focus of the hyperbola
def right_focus_hyperbola (x y : ℝ) : Prop := x = 3 ∧ y = 0

-- Theorem statement
theorem parabola_hyperbola_focus (p : ℝ) : 
  (∃ x y : ℝ, parabola p x y ∧ right_focus_hyperbola x y) → p = 6 := by
  sorry

end NUMINAMATH_CALUDE_parabola_hyperbola_focus_l439_43970


namespace NUMINAMATH_CALUDE_integer_count_inequality_l439_43902

theorem integer_count_inequality (x : ℤ) : 
  (Finset.filter (fun i => (i - 2)^2 ≤ 4) (Finset.range 10)).card = 5 := by
  sorry

end NUMINAMATH_CALUDE_integer_count_inequality_l439_43902


namespace NUMINAMATH_CALUDE_savings_is_240_l439_43940

/-- Represents the window purchase scenario -/
structure WindowPurchase where
  regularPrice : ℕ
  discountThreshold : ℕ
  freeWindows : ℕ
  georgeNeeds : ℕ
  anneNeeds : ℕ

/-- Calculates the cost for a given number of windows -/
def calculateCost (wp : WindowPurchase) (windows : ℕ) : ℕ :=
  let freeWindowSets := windows / wp.discountThreshold
  let paidWindows := windows - freeWindowSets * wp.freeWindows
  paidWindows * wp.regularPrice

/-- Calculates the savings when purchasing together vs separately -/
def calculateSavings (wp : WindowPurchase) : ℕ :=
  let separateCost := calculateCost wp wp.georgeNeeds + calculateCost wp wp.anneNeeds
  let togetherCost := calculateCost wp (wp.georgeNeeds + wp.anneNeeds)
  separateCost - togetherCost

/-- Theorem stating that the savings is $240 -/
theorem savings_is_240 (wp : WindowPurchase) 
  (h1 : wp.regularPrice = 120)
  (h2 : wp.discountThreshold = 10)
  (h3 : wp.freeWindows = 2)
  (h4 : wp.georgeNeeds = 9)
  (h5 : wp.anneNeeds = 11) :
  calculateSavings wp = 240 := by
  sorry


end NUMINAMATH_CALUDE_savings_is_240_l439_43940


namespace NUMINAMATH_CALUDE_small_semicircle_radius_l439_43914

/-- Configuration of tangent semicircles and circle -/
structure TangentConfiguration where
  R : ℝ  -- Radius of the large semicircle
  r : ℝ  -- Radius of the circle
  x : ℝ  -- Radius of the small semicircle
  tangent : R > 0 ∧ r > 0 ∧ x > 0  -- All radii are positive
  large_semicircle : R = 12  -- Large semicircle has radius 12
  circle : r = 6  -- Circle has radius 6
  pythagorean : r^2 + (R - x)^2 = (r + x)^2  -- Pythagorean theorem for tangent configuration

/-- The radius of the small semicircle in the tangent configuration is 4 -/
theorem small_semicircle_radius (config : TangentConfiguration) : config.x = 4 :=
  sorry

end NUMINAMATH_CALUDE_small_semicircle_radius_l439_43914


namespace NUMINAMATH_CALUDE_log_equation_proof_l439_43962

-- Define the common logarithm (base 10)
noncomputable def log (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_equation_proof :
  (log 5) ^ 2 + log 2 * log 50 = 1 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_proof_l439_43962


namespace NUMINAMATH_CALUDE_probability_other_side_red_l439_43941

structure Card where
  side1 : String
  side2 : String

def Box : List Card := [
  {side1 := "black", side2 := "black"},
  {side1 := "black", side2 := "black"},
  {side1 := "black", side2 := "black"},
  {side1 := "black", side2 := "black"},
  {side1 := "black", side2 := "red"},
  {side1 := "black", side2 := "red"},
  {side1 := "black", side2 := "red"},
  {side1 := "red", side2 := "red"},
  {side1 := "red", side2 := "red"},
  {side1 := "blue", side2 := "blue"}
]

def isRed (s : String) : Bool := s == "red"

def countRedSides (cards : List Card) : Nat :=
  cards.foldl (fun acc card => acc + (if isRed card.side1 then 1 else 0) + (if isRed card.side2 then 1 else 0)) 0

def countBothRedCards (cards : List Card) : Nat :=
  cards.foldl (fun acc card => acc + (if isRed card.side1 && isRed card.side2 then 1 else 0)) 0

theorem probability_other_side_red (box : List Card := Box) :
  let totalRedSides := countRedSides box
  let bothRedCards := countBothRedCards box
  (2 * bothRedCards : Rat) / totalRedSides = 4 / 7 := by sorry

end NUMINAMATH_CALUDE_probability_other_side_red_l439_43941


namespace NUMINAMATH_CALUDE_maximal_k_inequality_l439_43985

theorem maximal_k_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∀ k : ℝ, (k + a/b) * (k + b/c) * (k + c/a) ≤ (a/b + b/c + c/a) * (b/a + c/b + a/c) ↔ k ≤ Real.rpow 9 (1/3) - 1 :=
by sorry

end NUMINAMATH_CALUDE_maximal_k_inequality_l439_43985


namespace NUMINAMATH_CALUDE_must_divide_p_l439_43978

theorem must_divide_p (p q r s : ℕ+) 
  (h1 : Nat.gcd p q = 30)
  (h2 : Nat.gcd q r = 40)
  (h3 : Nat.gcd r s = 60)
  (h4 : 120 < Nat.gcd s p)
  (h5 : Nat.gcd s p < 180) : 
  7 ∣ p.val := by
  sorry

end NUMINAMATH_CALUDE_must_divide_p_l439_43978


namespace NUMINAMATH_CALUDE_distance_between_three_points_l439_43976

-- Define a line
structure Line where
  -- Add any necessary properties for a line

-- Define a point on a line
structure Point (l : Line) where
  -- Add any necessary properties for a point on a line

-- Define the distance between two points on a line
def distance (l : Line) (p q : Point l) : ℝ :=
  sorry

-- Theorem statement
theorem distance_between_three_points (l : Line) (A B C : Point l) :
  distance l A B = 5 ∧ distance l B C = 3 →
  distance l A C = 8 ∨ distance l A C = 2 :=
sorry

end NUMINAMATH_CALUDE_distance_between_three_points_l439_43976


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l439_43923

/-- A geometric sequence with common ratio q satisfying 2a₄ = a₆ - a₅ -/
def GeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  (∀ n, a (n + 1) = q * a n) ∧ (2 * a 4 = a 6 - a 5)

/-- The common ratio of a geometric sequence satisfying 2a₄ = a₆ - a₅ is either -1 or 2 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  GeometricSequence a q → q = -1 ∨ q = 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l439_43923


namespace NUMINAMATH_CALUDE_greatest_divisible_by_seven_ninety_five_six_five_nine_is_valid_l439_43969

def is_valid_number (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000 ∧
  ∃ (a b c : ℕ),
    a < 10 ∧ b < 10 ∧ c < 10 ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    n = a * 10000 + b * 1000 + c * 100 + b * 10 + a

theorem greatest_divisible_by_seven :
  ∀ n : ℕ, is_valid_number n ∧ n % 7 = 0 → n ≤ 95659 :=
by sorry

theorem ninety_five_six_five_nine_is_valid :
  is_valid_number 95659 ∧ 95659 % 7 = 0 :=
by sorry

end NUMINAMATH_CALUDE_greatest_divisible_by_seven_ninety_five_six_five_nine_is_valid_l439_43969


namespace NUMINAMATH_CALUDE_xiaoming_mother_money_l439_43910

/-- The amount of money Xiaoming's mother brought to buy soap. -/
def money : ℕ := 36

/-- The price of one unit of brand A soap in yuan. -/
def price_A : ℕ := 6

/-- The price of one unit of brand B soap in yuan. -/
def price_B : ℕ := 9

/-- The number of units of brand A soap that can be bought with the money. -/
def units_A : ℕ := money / price_A

/-- The number of units of brand B soap that can be bought with the money. -/
def units_B : ℕ := money / price_B

theorem xiaoming_mother_money :
  (units_A = units_B + 2) ∧
  (money = units_A * price_A) ∧
  (money = units_B * price_B) := by
  sorry

end NUMINAMATH_CALUDE_xiaoming_mother_money_l439_43910


namespace NUMINAMATH_CALUDE_perpendicular_lines_imply_a_eq_neg_three_l439_43989

/-- Two lines are perpendicular if the sum of the products of their coefficients is zero -/
def are_perpendicular (a₁ b₁ a₂ b₂ : ℝ) : Prop :=
  a₁ * a₂ + b₁ * b₂ = 0

/-- The first line: ax + 3y + 1 = 0 -/
def line1 (a : ℝ) (x y : ℝ) : Prop :=
  a * x + 3 * y + 1 = 0

/-- The second line: 2x + (a+1)y + 1 = 0 -/
def line2 (a : ℝ) (x y : ℝ) : Prop :=
  2 * x + (a + 1) * y + 1 = 0

/-- Theorem: If the lines are perpendicular, then a = -3 -/
theorem perpendicular_lines_imply_a_eq_neg_three (a : ℝ) :
  are_perpendicular a 3 2 (a + 1) → a = -3 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_imply_a_eq_neg_three_l439_43989


namespace NUMINAMATH_CALUDE_empty_solution_set_range_l439_43935

theorem empty_solution_set_range (a : ℝ) : 
  (∀ x : ℝ, ¬(|x - 1| - |x - 2| ≥ a^2 + a + 1)) →
  (a < -1 ∨ a > 0) :=
by sorry

end NUMINAMATH_CALUDE_empty_solution_set_range_l439_43935


namespace NUMINAMATH_CALUDE_problem_1_l439_43987

theorem problem_1 : (1) - 3 + 8 - 15 - 6 = -16 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l439_43987


namespace NUMINAMATH_CALUDE_geometric_sum_value_l439_43965

/-- Sum of a geometric series with 15 terms, first term 4/5, and common ratio 4/5 -/
def geometricSum : ℚ :=
  let a : ℚ := 4/5
  let r : ℚ := 4/5
  let n : ℕ := 15
  a * (1 - r^n) / (1 - r)

/-- The sum of the geometric series is equal to 117775277204/30517578125 -/
theorem geometric_sum_value : geometricSum = 117775277204/30517578125 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sum_value_l439_43965


namespace NUMINAMATH_CALUDE_ratio_problem_l439_43980

theorem ratio_problem (a b c d : ℚ) 
  (hab : a / b = 5 / 4)
  (hcd : c / d = 4 / 3)
  (hdb : d / b = 1 / 5) :
  a / c = 75 / 16 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l439_43980


namespace NUMINAMATH_CALUDE_expand_expression_l439_43944

theorem expand_expression (x : ℝ) : (7*x + 5) * 3*x^2 = 21*x^3 + 15*x^2 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l439_43944


namespace NUMINAMATH_CALUDE_museum_ticket_price_l439_43951

theorem museum_ticket_price (group_size : ℕ) (total_with_tax : ℚ) (tax_rate : ℚ) :
  group_size = 25 →
  total_with_tax = 945 →
  tax_rate = 5 / 100 →
  ∃ (ticket_price : ℚ),
    ticket_price * group_size * (1 + tax_rate) = total_with_tax ∧
    ticket_price = 36 :=
by sorry

end NUMINAMATH_CALUDE_museum_ticket_price_l439_43951


namespace NUMINAMATH_CALUDE_train_speed_calculation_l439_43992

/-- Calculates the speed of a train crossing a platform -/
theorem train_speed_calculation (train_length platform_length : Real) 
  (crossing_time : Real) (h1 : train_length = 240) 
  (h2 : platform_length = 240) (h3 : crossing_time = 27) : 
  ∃ (speed : Real), abs (speed - 64) < 0.01 := by
  sorry

#check train_speed_calculation

end NUMINAMATH_CALUDE_train_speed_calculation_l439_43992


namespace NUMINAMATH_CALUDE_light_ray_equation_l439_43961

-- Define the circle M
def circle_M (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 4*y + 7 = 0

-- Define the point A
def point_A : ℝ × ℝ := (-3, 3)

-- Define the x-axis
def x_axis (x y : ℝ) : Prop := y = 0

-- Define the reflected ray equation
def reflected_ray (x y : ℝ) : Prop :=
  4*x - 3*y + 9 = 0

-- Theorem statement
theorem light_ray_equation :
  ∃ (x₀ y₀ : ℝ),
    -- The ray passes through point A
    reflected_ray x₀ y₀ ∧ (x₀, y₀) = point_A ∧
    -- The ray intersects the x-axis
    ∃ (x₁ : ℝ), reflected_ray x₁ 0 ∧
    -- The ray is tangent to circle M
    ∃ (x₂ y₂ : ℝ), circle_M x₂ y₂ ∧ reflected_ray x₂ y₂ ∧
      ∀ (x y : ℝ), circle_M x y → (x - x₂)^2 + (y - y₂)^2 ≥ 1 :=
by sorry

end NUMINAMATH_CALUDE_light_ray_equation_l439_43961


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_property_l439_43918

/-- An arithmetic sequence with non-zero common difference -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, d ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, b (n + 1) = b n * r

theorem arithmetic_geometric_sequence_property
  (a : ℕ → ℝ) (b : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_geom : geometric_sequence b)
  (h_eq : 2 * a 3 - (a 7)^2 + 2 * a 11 = 0)
  (h_b7 : b 7 = a 7) :
  b 6 * b 8 = 16 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_property_l439_43918


namespace NUMINAMATH_CALUDE_hcf_from_lcm_and_product_l439_43984

/-- Given two positive integers with LCM 560 and product 42000, their HCF is 75 -/
theorem hcf_from_lcm_and_product (A B : ℕ+) 
  (h_lcm : Nat.lcm A B = 560)
  (h_product : A * B = 42000) :
  Nat.gcd A B = 75 := by
  sorry

end NUMINAMATH_CALUDE_hcf_from_lcm_and_product_l439_43984


namespace NUMINAMATH_CALUDE_sin_cos_three_eighths_pi_l439_43901

theorem sin_cos_three_eighths_pi (π : Real) :
  Real.sin (3 * π / 8) * Real.cos (π / 8) = (2 + Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_three_eighths_pi_l439_43901


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l439_43952

theorem unique_solution_quadratic (p : ℝ) (hp : p ≠ 0) :
  (∃! x : ℝ, p * x^2 - 10 * x + 2 = 0) ↔ p = 25/2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l439_43952


namespace NUMINAMATH_CALUDE_field_trip_students_l439_43908

theorem field_trip_students (van_capacity : Nat) (num_vans : Nat) (num_adults : Nat) :
  van_capacity = 8 →
  num_vans = 3 →
  num_adults = 2 →
  (van_capacity * num_vans) - num_adults = 22 :=
by sorry

end NUMINAMATH_CALUDE_field_trip_students_l439_43908


namespace NUMINAMATH_CALUDE_five_million_times_eight_million_l439_43903

theorem five_million_times_eight_million :
  (5 * (10 : ℕ)^6) * (8 * (10 : ℕ)^6) = 40 * (10 : ℕ)^12 := by
  sorry

end NUMINAMATH_CALUDE_five_million_times_eight_million_l439_43903


namespace NUMINAMATH_CALUDE_pencil_groups_l439_43936

theorem pencil_groups (total_pencils : ℕ) (pencils_per_group : ℕ) (h1 : total_pencils = 25) (h2 : pencils_per_group = 5) :
  total_pencils / pencils_per_group = 5 :=
by sorry

end NUMINAMATH_CALUDE_pencil_groups_l439_43936


namespace NUMINAMATH_CALUDE_contingency_and_sampling_theorem_l439_43913

/-- Represents the contingency table --/
structure ContingencyTable :=
  (male_running : ℕ)
  (male_not_running : ℕ)
  (female_running : ℕ)
  (female_not_running : ℕ)

/-- Calculates the K^2 value for the contingency table --/
def calculate_k_squared (table : ContingencyTable) : ℚ :=
  let n := table.male_running + table.male_not_running + table.female_running + table.female_not_running
  let a := table.male_running
  let b := table.male_not_running
  let c := table.female_running
  let d := table.female_not_running
  (n * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

/-- Calculates the expected value of females selected in the sampling process --/
def expected_females_selected (male_count female_count : ℕ) : ℚ :=
  (0 * (male_count * (male_count - 1)) + 
   1 * (2 * male_count * female_count) + 
   2 * (female_count * (female_count - 1))) / 
  ((male_count + female_count) * (male_count + female_count - 1))

/-- Main theorem to prove --/
theorem contingency_and_sampling_theorem 
  (table : ContingencyTable) 
  (h_total : table.male_running + table.male_not_running + table.female_running + table.female_not_running = 80)
  (h_male_running : table.male_running = 20)
  (h_male_not_running : table.male_not_running = 20)
  (h_female_not_running : table.female_not_running = 30) :
  calculate_k_squared table < (6635 : ℚ) / 1000 ∧ 
  expected_females_selected 2 3 = 6 / 5 := by
  sorry

end NUMINAMATH_CALUDE_contingency_and_sampling_theorem_l439_43913


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l439_43990

theorem algebraic_expression_value (x : ℝ) (h : x^2 + x - 5 = 0) :
  (x - 1)^2 - x*(x - 3) + (x + 2)*(x - 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l439_43990


namespace NUMINAMATH_CALUDE_three_integers_sum_and_reciprocals_l439_43972

theorem three_integers_sum_and_reciprocals (a b c : ℕ+) : 
  (a + b + c : ℕ) = 15 ∧ 
  (1 / (a : ℚ) + 1 / (b : ℚ) + 1 / (c : ℚ) = 71 / 105) → 
  ({a, b, c} : Finset ℕ+) = {3, 5, 7} := by
sorry

end NUMINAMATH_CALUDE_three_integers_sum_and_reciprocals_l439_43972


namespace NUMINAMATH_CALUDE_line_segment_difference_l439_43939

theorem line_segment_difference (L₁ L₂ : ℝ) : 
  L₁ = 7 → 
  L₁^2 - L₂^2 = 32 → 
  L₁ - L₂ = 7 - Real.sqrt 17 := by
  sorry

end NUMINAMATH_CALUDE_line_segment_difference_l439_43939


namespace NUMINAMATH_CALUDE_divisor_problem_l439_43937

theorem divisor_problem (n : ℕ) (h1 : n = 1025) (h2 : ¬ (n - 4) % 41 = 0) :
  ∀ d : ℕ, d > 41 → d ∣ n → d ∣ (n - 4) :=
sorry

end NUMINAMATH_CALUDE_divisor_problem_l439_43937


namespace NUMINAMATH_CALUDE_addison_mountain_temp_decrease_l439_43925

/-- The temperature decrease of Addison mountain after one hour -/
def temperature_decrease (current_temp : ℝ) (decrease_factor : ℝ) : ℝ :=
  current_temp - (decrease_factor * current_temp)

/-- Theorem: The temperature decrease is 21 degrees -/
theorem addison_mountain_temp_decrease :
  temperature_decrease 84 (3/4) = 21 := by
  sorry

end NUMINAMATH_CALUDE_addison_mountain_temp_decrease_l439_43925


namespace NUMINAMATH_CALUDE_red_yellow_flowers_l439_43955

theorem red_yellow_flowers (total : ℕ) (yellow_white : ℕ) (red_white : ℕ) (red_excess : ℕ) :
  total = 44 →
  yellow_white = 13 →
  red_white = 14 →
  red_excess = 4 →
  ∃ (red_yellow : ℕ), red_yellow = 17 ∧
    total = yellow_white + red_white + red_yellow ∧
    red_white + red_yellow = yellow_white + red_white + red_excess :=
by sorry

end NUMINAMATH_CALUDE_red_yellow_flowers_l439_43955


namespace NUMINAMATH_CALUDE_min_turns_10x10_grid_l439_43960

/-- Represents a grid of intersecting streets -/
structure StreetGrid where
  parallel_streets : ℕ
  intersecting_streets : ℕ

/-- Calculates the minimum number of turns required for a closed route
    passing through all intersections in a grid of streets -/
def min_turns (grid : StreetGrid) : ℕ :=
  2 * grid.parallel_streets

/-- The theorem stating that in a 10x10 grid of intersecting streets,
    the minimum number of turns required for a closed route passing
    through all intersections is 20 -/
theorem min_turns_10x10_grid :
  let grid : StreetGrid := ⟨10, 10⟩
  min_turns grid = 20 := by
  sorry

end NUMINAMATH_CALUDE_min_turns_10x10_grid_l439_43960


namespace NUMINAMATH_CALUDE_tangent_line_at_point_one_three_l439_43947

def f (x : ℝ) : ℝ := x^3 - x + 3

theorem tangent_line_at_point_one_three :
  let p : ℝ × ℝ := (1, 3)
  let m : ℝ := (deriv f) p.1
  (λ (x y : ℝ) => 2*x - y + 1 = 0) = (λ (x y : ℝ) => y - p.2 = m * (x - p.1)) := by sorry

end NUMINAMATH_CALUDE_tangent_line_at_point_one_three_l439_43947


namespace NUMINAMATH_CALUDE_sum_of_constants_l439_43953

/-- Given constants a, b, and c satisfying the specified conditions, prove that a + 2b + 3c = 64 -/
theorem sum_of_constants (a b c : ℝ) 
  (h1 : ∀ x, (x - a) * (x - b) / (x - c) ≤ 0 ↔ x < -4 ∨ |x - 25| ≤ 1)
  (h2 : a < b) : 
  a + 2*b + 3*c = 64 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_constants_l439_43953


namespace NUMINAMATH_CALUDE_triangle_properties_l439_43950

/-- Properties of a triangle ABC with given circumradius, one side length, and ratio of other sides -/
theorem triangle_properties (R a t : ℝ) (h_R : R > 0) (h_a : a > 0) (h_t : t > 0) :
  ∃ (b c : ℝ) (A B C : ℝ),
    b = 2 * R * Real.sin B ∧
    c = b / t ∧
    A = Real.arcsin (a / (2 * R)) ∧
    B = Real.arctan ((t * Real.sin A) / (1 - t * Real.cos A)) ∧
    C = π - A - B :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l439_43950


namespace NUMINAMATH_CALUDE_total_shoes_l439_43927

/-- Given that Ellie has 8 pairs of shoes and Riley has 3 fewer pairs than Ellie,
    prove that they have 13 pairs of shoes in total. -/
theorem total_shoes (ellie_shoes : ℕ) (riley_difference : ℕ) :
  ellie_shoes = 8 →
  riley_difference = 3 →
  ellie_shoes + (ellie_shoes - riley_difference) = 13 := by
  sorry

end NUMINAMATH_CALUDE_total_shoes_l439_43927


namespace NUMINAMATH_CALUDE_line_slope_calculation_l439_43998

/-- Given a line in the xy-plane with y-intercept 20 and passing through the point (150, 600),
    its slope is equal to 580/150. -/
theorem line_slope_calculation (line : Set (ℝ × ℝ)) : 
  (∀ p ∈ line, ∃ m b : ℝ, p.2 = m * p.1 + b) →  -- Line equation
  (0, 20) ∈ line →                              -- y-intercept condition
  (150, 600) ∈ line →                           -- Point condition
  ∃ m : ℝ, m = 580 / 150 ∧                      -- Slope existence and value
    ∀ (x y : ℝ), (x, y) ∈ line → y = m * x + 20 -- Line equation with calculated slope
  := by sorry

end NUMINAMATH_CALUDE_line_slope_calculation_l439_43998


namespace NUMINAMATH_CALUDE_curve_represents_two_points_l439_43926

theorem curve_represents_two_points :
  ∃! (p1 p2 : ℝ × ℝ), p1 ≠ p2 ∧
  (∀ (x y : ℝ), ((x - y)^2 + (x*y - 1)^2 = 0) ↔ (x, y) = p1 ∨ (x, y) = p2) :=
sorry

end NUMINAMATH_CALUDE_curve_represents_two_points_l439_43926


namespace NUMINAMATH_CALUDE_rhombus_longest_diagonal_l439_43900

/-- Given a rhombus with area 150 square units and diagonals in the ratio 4:3,
    prove that the length of the longest diagonal is 20 units. -/
theorem rhombus_longest_diagonal (area : ℝ) (d₁ d₂ : ℝ) : 
  area = 150 →
  d₁ / d₂ = 4 / 3 →
  area = (1 / 2) * d₁ * d₂ →
  d₁ > d₂ →
  d₁ = 20 := by
sorry

end NUMINAMATH_CALUDE_rhombus_longest_diagonal_l439_43900


namespace NUMINAMATH_CALUDE_range_of_sum_l439_43916

theorem range_of_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 / a + 1 / b = 1) :
  a + b ≥ 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_sum_l439_43916


namespace NUMINAMATH_CALUDE_binary_sum_equals_1100000_l439_43956

/-- Converts a list of bits to its decimal representation -/
def binaryToDecimal (bits : List Bool) : Nat :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Represents a binary number as a list of booleans -/
def Binary := List Bool

theorem binary_sum_equals_1100000 :
  let a : Binary := [true, false, false, true, true]  -- 11001₂
  let b : Binary := [true, true, true]                -- 111₂
  let c : Binary := [false, false, true, false, true] -- 10100₂
  let d : Binary := [true, true, true, true]          -- 1111₂
  let e : Binary := [true, true, false, false, true, true] -- 110011₂
  let sum : Binary := [false, false, false, false, false, true, true] -- 1100000₂
  binaryToDecimal a + binaryToDecimal b + binaryToDecimal c +
  binaryToDecimal d + binaryToDecimal e = binaryToDecimal sum := by
  sorry


end NUMINAMATH_CALUDE_binary_sum_equals_1100000_l439_43956


namespace NUMINAMATH_CALUDE_shelter_dogs_l439_43919

theorem shelter_dogs (x : ℕ) (dogs cats : ℕ) 
  (h1 : dogs * 7 = x * cats) 
  (h2 : dogs * 11 = 15 * (cats + 8)) : 
  dogs = 77 := by
  sorry

end NUMINAMATH_CALUDE_shelter_dogs_l439_43919


namespace NUMINAMATH_CALUDE_box_ratio_l439_43964

/-- Represents the number of cardboards of each type -/
structure Cardboards where
  square : ℕ
  rectangular : ℕ

/-- Represents the number of boxes of each type -/
structure Boxes where
  vertical : ℕ
  horizontal : ℕ

/-- Represents the number of cardboards used for each type of box -/
structure BoxRequirements where
  vertical_square : ℕ
  vertical_rectangular : ℕ
  horizontal_square : ℕ
  horizontal_rectangular : ℕ

/-- The main theorem stating the ratio of vertical to horizontal boxes -/
theorem box_ratio 
  (c : Cardboards) 
  (b : Boxes) 
  (r : BoxRequirements) 
  (h1 : c.rectangular = 2 * c.square)  -- Ratio of cardboards is 1:2
  (h2 : r.vertical_square * b.vertical + r.horizontal_square * b.horizontal = c.square)  -- All square cardboards are used
  (h3 : r.vertical_rectangular * b.vertical + r.horizontal_rectangular * b.horizontal = c.rectangular)  -- All rectangular cardboards are used
  : b.vertical = b.horizontal / 2 := by
  sorry

end NUMINAMATH_CALUDE_box_ratio_l439_43964


namespace NUMINAMATH_CALUDE_euler_family_mean_age_l439_43921

def euler_family_ages : List ℝ := [8, 8, 10, 10, 15, 12]

theorem euler_family_mean_age :
  (euler_family_ages.sum / euler_family_ages.length : ℝ) = 10.5 := by
  sorry

end NUMINAMATH_CALUDE_euler_family_mean_age_l439_43921


namespace NUMINAMATH_CALUDE_five_solutions_l439_43911

/-- The system of equations has exactly 5 distinct real solutions -/
theorem five_solutions (x y z w : ℝ) : 
  (x = w + z + z*w*x) ∧
  (y = z + x + z*x*y) ∧
  (z = x + y + x*y*z) ∧
  (w = y + z + y*z*w) →
  ∃! (sol : Finset (ℝ × ℝ × ℝ × ℝ)), sol.card = 5 ∧ ∀ (a b c d : ℝ), (a, b, c, d) ∈ sol ↔
    (a = d + c + c*d*a) ∧
    (b = c + a + c*a*b) ∧
    (c = a + b + a*b*c) ∧
    (d = b + c + b*c*d) := by
  sorry

end NUMINAMATH_CALUDE_five_solutions_l439_43911


namespace NUMINAMATH_CALUDE_somu_age_proof_l439_43948

/-- Somu's present age -/
def somu_age : ℕ := 12

/-- Somu's father's present age -/
def father_age : ℕ := 3 * somu_age

theorem somu_age_proof :
  (somu_age = father_age / 3) ∧
  (somu_age - 6 = (father_age - 6) / 5) →
  somu_age = 12 := by
sorry

end NUMINAMATH_CALUDE_somu_age_proof_l439_43948


namespace NUMINAMATH_CALUDE_odd_implies_abs_symmetric_abs_symmetric_not_sufficient_for_odd_l439_43954

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x ∈ ℝ -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The graph of |f(x)| is symmetric about the y-axis if |f(-x)| = |f(x)| for all x ∈ ℝ -/
def IsAbsSymmetric (f : ℝ → ℝ) : Prop :=
  ∀ x, |f (-x)| = |f x|

theorem odd_implies_abs_symmetric (f : ℝ → ℝ) :
  IsOdd f → IsAbsSymmetric f :=
sorry

theorem abs_symmetric_not_sufficient_for_odd :
  ∃ f : ℝ → ℝ, IsAbsSymmetric f ∧ ¬IsOdd f :=
sorry

end NUMINAMATH_CALUDE_odd_implies_abs_symmetric_abs_symmetric_not_sufficient_for_odd_l439_43954


namespace NUMINAMATH_CALUDE_tree_height_problem_l439_43963

theorem tree_height_problem (h₁ h₂ : ℝ) : 
  h₁ = h₂ + 24 →  -- One tree is 24 feet taller than the other
  h₂ / h₁ = 2 / 3 →  -- The heights are in the ratio 2:3
  h₁ = 72 :=  -- The height of the taller tree is 72 feet
by
  sorry

end NUMINAMATH_CALUDE_tree_height_problem_l439_43963


namespace NUMINAMATH_CALUDE_min_value_of_sum_l439_43974

theorem min_value_of_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 2/y = 1) :
  ∃ (m : ℝ), m = 3 + 2 * Real.sqrt 2 ∧ x + y ≥ m ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 1/x₀ + 2/y₀ = 1 ∧ x₀ + y₀ = m :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_sum_l439_43974


namespace NUMINAMATH_CALUDE_afternoon_to_morning_ratio_l439_43999

def total_pears : ℕ := 420
def afternoon_pears : ℕ := 280

theorem afternoon_to_morning_ratio :
  let morning_pears := total_pears - afternoon_pears
  (afternoon_pears : ℚ) / morning_pears = 2 := by
  sorry

end NUMINAMATH_CALUDE_afternoon_to_morning_ratio_l439_43999


namespace NUMINAMATH_CALUDE_division_problem_l439_43979

theorem division_problem (n : ℕ) : 
  let first_part : ℕ := 19
  let second_part : ℕ := 36 - first_part
  n * first_part + 3 * second_part = 203 → n = 8 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l439_43979


namespace NUMINAMATH_CALUDE_project_completion_time_l439_43967

theorem project_completion_time 
  (days_A : ℝ) 
  (days_B : ℝ) 
  (work_days_A : ℝ) 
  (remaining_days_B : ℝ) 
  (h1 : days_A = 10) 
  (h2 : days_B = 15) 
  (h3 : work_days_A = 3) : 
  work_days_A / days_A + remaining_days_B / days_B = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_project_completion_time_l439_43967


namespace NUMINAMATH_CALUDE_city_d_sand_amount_l439_43959

/-- The amount of sand received by each city and the total amount --/
structure SandDistribution where
  cityA : Rat
  cityB : Rat
  cityC : Rat
  total : Rat

/-- The amount of sand received by City D --/
def sandCityD (sd : SandDistribution) : Rat :=
  sd.total - (sd.cityA + sd.cityB + sd.cityC)

/-- Theorem stating that City D received 28 tons of sand --/
theorem city_d_sand_amount :
  let sd : SandDistribution := {
    cityA := 33/2,
    cityB := 26,
    cityC := 49/2,
    total := 95
  }
  sandCityD sd = 28 := by sorry

end NUMINAMATH_CALUDE_city_d_sand_amount_l439_43959


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l439_43938

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

-- Define set B
def B : Set ℝ := {x | 0 < x ∧ x < 1}

-- State the theorem
theorem intersection_A_complement_B :
  A ∩ (U \ B) = {x | (-2 ≤ x ∧ x ≤ 0) ∨ (1 ≤ x ∧ x ≤ 2)} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l439_43938


namespace NUMINAMATH_CALUDE_square_8x_minus_5_l439_43949

theorem square_8x_minus_5 (x : ℝ) (h : 8 * x^2 + 7 = 12 * x + 17) : (8 * x - 5)^2 = 465 := by
  sorry

end NUMINAMATH_CALUDE_square_8x_minus_5_l439_43949


namespace NUMINAMATH_CALUDE_partnership_investment_ratio_l439_43934

/-- Represents the investment and profit structure of a partnership --/
structure Partnership where
  a_investment : ℝ
  b_investment_multiple : ℝ
  annual_gain : ℝ
  a_share : ℝ

/-- Calculates the ratio of B's investment to A's investment --/
def investment_ratio (p : Partnership) : ℝ := p.b_investment_multiple

theorem partnership_investment_ratio (p : Partnership) 
  (h1 : p.annual_gain = 18300)
  (h2 : p.a_share = 6100)
  (h3 : p.a_investment > 0) :
  investment_ratio p = 3 := by
  sorry

#check partnership_investment_ratio

end NUMINAMATH_CALUDE_partnership_investment_ratio_l439_43934


namespace NUMINAMATH_CALUDE_semester_weeks_calculation_l439_43975

/-- The number of weeks in a semester before midterms -/
def weeks_in_semester : ℕ := 6

/-- The number of hours Annie spends on extracurriculars per week -/
def hours_per_week : ℕ := 13

/-- The number of weeks Annie takes off sick -/
def sick_weeks : ℕ := 2

/-- The total number of hours Annie spends on extracurriculars before midterms -/
def total_hours : ℕ := 52

theorem semester_weeks_calculation :
  weeks_in_semester * hours_per_week - sick_weeks * hours_per_week = total_hours := by
  sorry

end NUMINAMATH_CALUDE_semester_weeks_calculation_l439_43975


namespace NUMINAMATH_CALUDE_farmer_euclid_field_l439_43986

theorem farmer_euclid_field (a b c x : ℝ) (h1 : a = 5) (h2 : b = 12) (h3 : c^2 = a^2 + b^2)
  (h4 : (b / c) * x + (a / c) * x = 3) :
  (a * b / 2 - x^2) / (a * b / 2) = 2393 / 2890 := by sorry

end NUMINAMATH_CALUDE_farmer_euclid_field_l439_43986


namespace NUMINAMATH_CALUDE_max_product_863_l439_43994

/-- A type representing the digits we can use -/
inductive Digit
  | three
  | five
  | six
  | eight
  | nine

/-- A function to convert our Digit type to a natural number -/
def digit_to_nat (d : Digit) : Nat :=
  match d with
  | Digit.three => 3
  | Digit.five => 5
  | Digit.six => 6
  | Digit.eight => 8
  | Digit.nine => 9

/-- A type representing a valid combination of digits -/
structure DigitCombination where
  a : Digit
  b : Digit
  c : Digit
  d : Digit
  e : Digit
  all_different : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e

/-- Calculate the product of the three-digit and two-digit numbers -/
def calculate_product (combo : DigitCombination) : Nat :=
  (100 * digit_to_nat combo.a + 10 * digit_to_nat combo.b + digit_to_nat combo.c) *
  (10 * digit_to_nat combo.d + digit_to_nat combo.e)

/-- The main theorem -/
theorem max_product_863 :
  ∀ combo : DigitCombination,
    calculate_product combo ≤ calculate_product
      { a := Digit.eight
      , b := Digit.six
      , c := Digit.three
      , d := Digit.nine
      , e := Digit.five
      , all_different := by simp } :=
by
  sorry


end NUMINAMATH_CALUDE_max_product_863_l439_43994


namespace NUMINAMATH_CALUDE_log_inequality_range_l439_43942

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem log_inequality_range (x : ℝ) :
  lg (x + 1) < lg (3 - x) ↔ -1 < x ∧ x < 1 :=
by sorry

end NUMINAMATH_CALUDE_log_inequality_range_l439_43942


namespace NUMINAMATH_CALUDE_min_sum_of_squares_l439_43996

theorem min_sum_of_squares (a b c t : ℝ) (h : a + b + c = t) :
  ∃ (m : ℝ), m = t^2 / 3 ∧ ∀ (x y z : ℝ), x + y + z = t → x^2 + y^2 + z^2 ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_squares_l439_43996


namespace NUMINAMATH_CALUDE_expression_simplification_l439_43932

theorem expression_simplification :
  (5^2010)^2 - (5^2008)^2 / (5^2009)^2 - (5^2007)^2 = 25 := by
sorry

end NUMINAMATH_CALUDE_expression_simplification_l439_43932


namespace NUMINAMATH_CALUDE_remaining_balloons_l439_43922

/-- The number of balloons in a dozen -/
def dozen : ℕ := 12

/-- The number of dozens of balloons the clown starts with -/
def initial_dozens : ℕ := 3

/-- The number of boys who buy a balloon -/
def boys : ℕ := 3

/-- The number of girls who buy a balloon -/
def girls : ℕ := 12

/-- Theorem: The clown is left with 21 balloons after selling to boys and girls -/
theorem remaining_balloons :
  initial_dozens * dozen - (boys + girls) = 21 := by
  sorry

end NUMINAMATH_CALUDE_remaining_balloons_l439_43922


namespace NUMINAMATH_CALUDE_monotone_increasing_condition_l439_43993

open Real

/-- A function f(x) = kx - ln(x) is monotonically increasing on (1, +∞) if and only if k ≥ 1 -/
theorem monotone_increasing_condition (k : ℝ) :
  (∀ x > 1, StrictMono (fun x => k * x - log x)) ↔ k ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_monotone_increasing_condition_l439_43993


namespace NUMINAMATH_CALUDE_cube_root_of_1331_l439_43924

theorem cube_root_of_1331 (y : ℝ) (h1 : y > 0) (h2 : y^3 = 1331) : y = 11 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_1331_l439_43924


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l439_43929

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem statement -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
    (h_arith : arithmetic_sequence a) 
    (h_sum : a 3 + a 8 = 10) : 
  3 * a 5 + a 7 = 20 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l439_43929


namespace NUMINAMATH_CALUDE_triangle_not_divisible_into_trapeziums_l439_43971

-- Define a shape as a type
inductive Shape
| Rectangle
| Square
| RegularHexagon
| Trapezium
| Triangle

-- Define a trapezium
def isTrapezium (s : Shape) : Prop :=
  ∃ (sides : ℕ), sides = 4 ∧ ∃ (parallel_sides : ℕ), parallel_sides ≥ 1

-- Define the property of being divisible into two trapeziums by a single straight line
def isDivisibleIntoTwoTrapeziums (s : Shape) : Prop :=
  ∃ (part1 part2 : Shape), isTrapezium part1 ∧ isTrapezium part2

-- State the theorem
theorem triangle_not_divisible_into_trapeziums :
  ¬(isDivisibleIntoTwoTrapeziums Shape.Triangle) :=
sorry

end NUMINAMATH_CALUDE_triangle_not_divisible_into_trapeziums_l439_43971


namespace NUMINAMATH_CALUDE_max_value_constraint_l439_43982

theorem max_value_constraint (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) 
  (h4 : a^2 + b^2 + c^2 = 1) :
  ∃ (max : ℝ), max = Real.sqrt 3 + Real.sqrt 2 / 3 ∧ 
    ∀ (x y z : ℝ), 0 ≤ x → 0 ≤ y → 0 ≤ z → x^2 + y^2 + z^2 = 1 → 
      x * y * Real.sqrt 3 + y * z * Real.sqrt 3 + z * x * Real.sqrt 2 ≤ max :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_constraint_l439_43982


namespace NUMINAMATH_CALUDE_shape_is_regular_tetrahedron_l439_43930

/-- A 3D shape with diagonals of adjacent sides meeting at a 60-degree angle -/
structure Shape3D where
  diagonalAngle : ℝ
  diagonalAngleIs60 : diagonalAngle = 60

/-- Definition of a regular tetrahedron -/
def RegularTetrahedron : Type := Unit

/-- Theorem: A 3D shape with diagonals of adjacent sides meeting at a 60-degree angle is a regular tetrahedron -/
theorem shape_is_regular_tetrahedron (s : Shape3D) : RegularTetrahedron := by
  sorry

end NUMINAMATH_CALUDE_shape_is_regular_tetrahedron_l439_43930


namespace NUMINAMATH_CALUDE_find_m_value_l439_43909

/-- Given two functions f and g, prove the value of m when f(5) - g(5) = 20 -/
theorem find_m_value (f g : ℝ → ℝ) (m : ℝ) : 
  (∀ x, f x = 4*x^2 + 3*x + 5) →
  (∀ x, g x = x^2 - m*x - 9) →
  f 5 - g 5 = 20 →
  m = -16.8 := by
sorry

end NUMINAMATH_CALUDE_find_m_value_l439_43909


namespace NUMINAMATH_CALUDE_smallest_n_is_correct_l439_43915

/-- The smallest positive integer n such that all roots of z^5 - z^3 + z = 0 are n^th roots of unity -/
def smallest_n : ℕ := 12

/-- The complex polynomial z^5 - z^3 + z -/
def f (z : ℂ) : ℂ := z^5 - z^3 + z

theorem smallest_n_is_correct :
  ∀ z : ℂ, f z = 0 → ∃ k : ℕ, z^smallest_n = 1 ∧
  ∀ m : ℕ, (∀ w : ℂ, f w = 0 → w^m = 1) → smallest_n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_is_correct_l439_43915


namespace NUMINAMATH_CALUDE_equidistant_point_x_coordinate_l439_43931

theorem equidistant_point_x_coordinate :
  ∃ (x y : ℝ),
    (abs y = abs x) ∧  -- Distance from y-axis equals distance from x-axis
    (abs y = abs ((x + y - 4) / Real.sqrt 2)) ∧  -- Distance from x-axis equals distance from line x + y = 4
    (x = 2) := by
  sorry

end NUMINAMATH_CALUDE_equidistant_point_x_coordinate_l439_43931


namespace NUMINAMATH_CALUDE_min_sum_4x4x4_dice_cube_l439_43991

/-- Represents a 4x4x4 cube made of dice -/
structure LargeCube where
  size : Nat
  total_dice : Nat
  opposite_face_sum : Nat

/-- Calculates the minimum visible sum on the large cube -/
def min_visible_sum (c : LargeCube) : Nat :=
  sorry

/-- Theorem stating the minimum visible sum for a 4x4x4 cube of dice -/
theorem min_sum_4x4x4_dice_cube :
  ∀ c : LargeCube, 
    c.size = 4 → 
    c.total_dice = 64 → 
    c.opposite_face_sum = 7 → 
    min_visible_sum c = 144 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_4x4x4_dice_cube_l439_43991


namespace NUMINAMATH_CALUDE_hyperbola_focus_to_asymptote_distance_l439_43904

/-- Given a hyperbola and a parabola with coinciding foci, prove the distance from the hyperbola's focus to its asymptote -/
theorem hyperbola_focus_to_asymptote_distance 
  (a : ℝ) 
  (h1 : ∀ x y : ℝ, x^2 / a^2 - y^2 / 5 = 1 → (∃ c : ℝ, x^2 / c^2 - y^2 / 5 = 1 ∧ c^2 = 4)) 
  (h2 : ∀ x y : ℝ, y^2 = 12*x → (∃ p : ℝ × ℝ, p = (3, 0))) : 
  ∃ d : ℝ, d = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_focus_to_asymptote_distance_l439_43904


namespace NUMINAMATH_CALUDE_vector_relation_l439_43995

variable {V : Type*} [AddCommGroup V] [Module ℝ V]
variable (P A B C : V)

theorem vector_relation (h : (A - P) + 2 • (B - P) + 3 • (C - P) = 0) :
  P - A = (1/3) • (B - A) + (1/2) • (C - A) := by sorry

end NUMINAMATH_CALUDE_vector_relation_l439_43995


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l439_43946

-- Define the universal set U
def U : Set ℕ := {x | x ≤ 5}

-- Define sets A and B
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {1, 4}

-- State the theorem
theorem complement_intersection_theorem :
  (Set.compl A ∩ Set.compl B) = {0, 5} := by
  sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l439_43946


namespace NUMINAMATH_CALUDE_log_inequality_l439_43997

theorem log_inequality : ∀ x : ℝ, x > 0 → x - 1 ≥ Real.log x ∧ (x - 1 = Real.log x ↔ x = 1) := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l439_43997


namespace NUMINAMATH_CALUDE_inequalities_not_always_hold_l439_43917

theorem inequalities_not_always_hold :
  ∃ (a b c x y z : ℝ),
    x < a ∧ y < b ∧ z < c ∧
    ¬(x * y + y * z + z * x < a * b + b * c + c * a) ∧
    ¬(x^2 + y^2 + z^2 < a^2 + b^2 + c^2) ∧
    ¬(x * y * z < a * b * c) :=
by sorry

end NUMINAMATH_CALUDE_inequalities_not_always_hold_l439_43917


namespace NUMINAMATH_CALUDE_max_a_value_l439_43988

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x

-- Define the theorem
theorem max_a_value (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, |f a (f a x)| ≤ 2) →
  a ≤ (3 + Real.sqrt 17) / 4 :=
by sorry

end NUMINAMATH_CALUDE_max_a_value_l439_43988


namespace NUMINAMATH_CALUDE_ralphs_initial_cards_l439_43957

theorem ralphs_initial_cards (cards_from_father cards_after : ℕ) :
  cards_from_father = 8 →
  cards_after = 12 →
  cards_after - cards_from_father = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_ralphs_initial_cards_l439_43957


namespace NUMINAMATH_CALUDE_range_of_a_l439_43983

def P (a : ℝ) := {x : ℝ | a - 4 < x ∧ x < a + 4}

def Q := {x : ℝ | x^2 - 4*x + 3 < 0}

theorem range_of_a :
  (∀ x, x ∈ Q → x ∈ P a) ↔ -1 ≤ a ∧ a ≤ 5 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l439_43983


namespace NUMINAMATH_CALUDE_impossible_partition_l439_43973

theorem impossible_partition : ¬ ∃ (A B C : Finset ℕ),
  (A ∪ B ∪ C = Finset.range 100) ∧
  (A ∩ B = ∅) ∧ (B ∩ C = ∅) ∧ (A ∩ C = ∅) ∧
  (∃ k₁ : ℕ, (A.sum id) = 102 * k₁) ∧
  (∃ k₂ : ℕ, (B.sum id) = 203 * k₂) ∧
  (∃ k₃ : ℕ, (C.sum id) = 304 * k₃) :=
by sorry

end NUMINAMATH_CALUDE_impossible_partition_l439_43973

import Mathlib

namespace NUMINAMATH_CALUDE_min_abs_z_l3140_314098

/-- Given a complex number z satisfying |z - 10| + |z + 3i| = 15, the minimum value of |z| is 2. -/
theorem min_abs_z (z : ℂ) (h : Complex.abs (z - 10) + Complex.abs (z + 3*I) = 15) : 
  ∃ (w : ℂ), Complex.abs (z - 10) + Complex.abs (z + 3*I) = 15 ∧ Complex.abs w = 2 ∧ 
  ∀ (v : ℂ), Complex.abs (v - 10) + Complex.abs (v + 3*I) = 15 → Complex.abs w ≤ Complex.abs v :=
sorry

end NUMINAMATH_CALUDE_min_abs_z_l3140_314098


namespace NUMINAMATH_CALUDE_curve_not_parabola_l3140_314039

-- Define the curve equation
def curve_equation (k : ℝ) (x y : ℝ) : Prop :=
  k * x^2 + y^2 = 1

-- Define what it means for a curve to be a parabola
-- (This is a simplified definition for the purpose of this statement)
def is_parabola (f : ℝ → ℝ → Prop) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x y, f x y ↔ y = a * x^2 + b * x + c

-- Theorem statement
theorem curve_not_parabola :
  ∀ k : ℝ, ¬(is_parabola (curve_equation k)) :=
sorry

end NUMINAMATH_CALUDE_curve_not_parabola_l3140_314039


namespace NUMINAMATH_CALUDE_inequality_solution_l3140_314089

theorem inequality_solution :
  ∀ x y : ℝ,
  (y^2)^2 < (x + 1)^2 ∧ (x + 1)^2 = y^4 + y^2 + 1 ∧ y^4 + y^2 + 1 ≤ (y^2 + 1)^2 →
  (x = 0 ∧ y = 0) ∨ (x = -2 ∧ y = 0) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3140_314089


namespace NUMINAMATH_CALUDE_max_value_of_f_l3140_314012

-- Define the function f(x) = x³ - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- State the theorem
theorem max_value_of_f :
  ∃ (M : ℝ), M = 18 ∧ ∀ x ∈ Set.Icc (-1 : ℝ) 3, f x ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_f_l3140_314012


namespace NUMINAMATH_CALUDE_infinite_triangular_pairs_l3140_314014

theorem infinite_triangular_pairs :
  ∃ (S : Set Nat), Set.Infinite S ∧ 
    (∀ p ∈ S, Prime p ∧ Odd p ∧
      (∀ t : Nat, t > 0 →
        (∃ n : Nat, t = n * (n + 1) / 2) ↔
        (∃ m : Nat, p^2 * t + (p^2 - 1) / 8 = m * (m + 1) / 2))) := by
  sorry

end NUMINAMATH_CALUDE_infinite_triangular_pairs_l3140_314014


namespace NUMINAMATH_CALUDE_max_sum_of_factors_l3140_314026

theorem max_sum_of_factors (x y : ℕ+) (h : x * y = 48) : 
  ∃ (a b : ℕ+), a * b = 48 ∧ a + b ≤ x + y ∧ a + b = 49 :=
sorry

end NUMINAMATH_CALUDE_max_sum_of_factors_l3140_314026


namespace NUMINAMATH_CALUDE_total_money_for_76_members_l3140_314057

/-- Calculates the total money collected in rupees given the number of members in a group -/
def totalMoneyCollected (members : ℕ) : ℚ :=
  (members * members : ℕ) / 100

/-- Proves that for a group of 76 members, the total money collected is ₹57.76 -/
theorem total_money_for_76_members :
  totalMoneyCollected 76 = 57.76 := by
  sorry

end NUMINAMATH_CALUDE_total_money_for_76_members_l3140_314057


namespace NUMINAMATH_CALUDE_base12_addition_l3140_314096

/-- Converts a base 12 number represented as a list of digits to its decimal equivalent -/
def toDecimal (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => acc * 12 + d) 0

/-- Converts a decimal number to its base 12 representation -/
def toBase12 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else aux (m / 12) ((m % 12) :: acc)
  aux n []

/-- The sum of 1704₁₂ and 259₁₂ in base 12 is equal to 1961₁₂ -/
theorem base12_addition :
  toBase12 (toDecimal [1, 7, 0, 4] + toDecimal [2, 5, 9]) = [1, 9, 6, 1] :=
by sorry

end NUMINAMATH_CALUDE_base12_addition_l3140_314096


namespace NUMINAMATH_CALUDE_division_problem_l3140_314059

theorem division_problem (dividend quotient remainder divisor : ℕ) : 
  dividend = 109 →
  quotient = 9 →
  remainder = 1 →
  dividend = divisor * quotient + remainder →
  divisor = 12 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l3140_314059


namespace NUMINAMATH_CALUDE_max_sum_reciprocal_cubes_l3140_314054

/-- The roots of a cubic polynomial satisfying a specific condition -/
structure CubicRoots where
  r₁ : ℝ
  r₂ : ℝ
  r₃ : ℝ
  sum_eq_sum_squares : r₁ + r₂ + r₃ = r₁^2 + r₂^2 + r₃^2

/-- The coefficients of a cubic polynomial -/
structure CubicCoeffs where
  s : ℝ
  p : ℝ
  q : ℝ

/-- The theorem stating the maximum value of the sum of reciprocal cubes of roots -/
theorem max_sum_reciprocal_cubes (roots : CubicRoots) (coeffs : CubicCoeffs) 
  (vieta₁ : roots.r₁ + roots.r₂ + roots.r₃ = coeffs.s)
  (vieta₂ : roots.r₁ * roots.r₂ + roots.r₂ * roots.r₃ + roots.r₃ * roots.r₁ = coeffs.p)
  (vieta₃ : roots.r₁ * roots.r₂ * roots.r₃ = coeffs.q) :
  ∃ (max : ℝ), max = 3 ∧ ∀ (roots' : CubicRoots),
    (1 / roots'.r₁^3 + 1 / roots'.r₂^3 + 1 / roots'.r₃^3) ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_sum_reciprocal_cubes_l3140_314054


namespace NUMINAMATH_CALUDE_sausage_pieces_l3140_314060

/-- Given a sausage with three sets of rings, this function calculates
    the number of pieces obtained by cutting along all rings. -/
def totalPieces (redPieces yellowPieces greenPieces : Nat) : Nat :=
  let redCuts := redPieces - 1
  let yellowCuts := yellowPieces - 1
  let greenCuts := greenPieces - 1
  redCuts + yellowCuts + greenCuts + 1

/-- Theorem stating that cutting a sausage along rings that individually
    result in 5, 7, and 11 pieces when cut separately will result in
    21 pieces when all rings are cut. -/
theorem sausage_pieces : totalPieces 5 7 11 = 21 := by
  sorry

end NUMINAMATH_CALUDE_sausage_pieces_l3140_314060


namespace NUMINAMATH_CALUDE_marbles_distribution_l3140_314061

/-- The number of marbles each boy received when 28 marbles were equally distributed among 14 boys -/
def marbles_per_boy : ℕ := sorry

/-- The total number of marbles Haley had -/
def total_marbles : ℕ := 28

/-- The number of boys who received marbles -/
def number_of_boys : ℕ := 14

theorem marbles_distribution :
  marbles_per_boy * number_of_boys = total_marbles ∧ marbles_per_boy = 2 := by sorry

end NUMINAMATH_CALUDE_marbles_distribution_l3140_314061


namespace NUMINAMATH_CALUDE_motorboat_problem_l3140_314075

/-- Represents the problem of calculating the time taken by a motorboat to reach an island in still water -/
theorem motorboat_problem (downstream_distance : ℝ) (downstream_time : ℝ) (upstream_time : ℝ) (island_distance : ℝ) :
  downstream_distance = 160 →
  downstream_time = 8 →
  upstream_time = 16 →
  island_distance = 100 →
  ∃ (boat_speed : ℝ) (current_speed : ℝ),
    boat_speed + current_speed = downstream_distance / downstream_time ∧
    boat_speed - current_speed = downstream_distance / upstream_time ∧
    island_distance / boat_speed = 20 / 3 :=
by sorry

end NUMINAMATH_CALUDE_motorboat_problem_l3140_314075


namespace NUMINAMATH_CALUDE_jeremy_payment_l3140_314044

/-- The total amount owed to Jeremy for cleaning rooms and washing windows -/
theorem jeremy_payment (room_rate : ℚ) (window_rate : ℚ) (rooms_cleaned : ℚ) (windows_washed : ℚ)
  (h1 : room_rate = 13 / 3)
  (h2 : window_rate = 5 / 2)
  (h3 : rooms_cleaned = 8 / 5)
  (h4 : windows_washed = 11 / 4) :
  room_rate * rooms_cleaned + window_rate * windows_washed = 553 / 40 :=
by sorry

end NUMINAMATH_CALUDE_jeremy_payment_l3140_314044


namespace NUMINAMATH_CALUDE_divisor_quotient_remainder_equality_l3140_314013

theorem divisor_quotient_remainder_equality (n : ℕ) (h : n > 1) :
  let divisors := {d : ℕ | d ∣ (n + 1)}
  let quotients := {q : ℕ | ∃ d ∈ divisors, q = n / d}
  let remainders := {r : ℕ | ∃ d ∈ divisors, r = n % d}
  quotients = remainders :=
by sorry

end NUMINAMATH_CALUDE_divisor_quotient_remainder_equality_l3140_314013


namespace NUMINAMATH_CALUDE_ed_limpet_shells_l3140_314081

/-- The number of limpet shells Ed found -/
def L : ℕ := sorry

/-- The initial number of shells in the collection -/
def initial_shells : ℕ := 2

/-- The number of oyster shells Ed found -/
def ed_oyster_shells : ℕ := 2

/-- The number of conch shells Ed found -/
def ed_conch_shells : ℕ := 4

/-- The total number of shells Ed found -/
def ed_total_shells : ℕ := L + ed_oyster_shells + ed_conch_shells

/-- The total number of shells Jacob found -/
def jacob_total_shells : ℕ := ed_total_shells + 2

/-- The total number of shells in the final collection -/
def total_shells : ℕ := 30

theorem ed_limpet_shells :
  initial_shells + ed_total_shells + jacob_total_shells = total_shells ∧ L = 7 := by
  sorry

end NUMINAMATH_CALUDE_ed_limpet_shells_l3140_314081


namespace NUMINAMATH_CALUDE_intersection_distance_l3140_314019

/-- The distance between the intersection points of y² = x and x + 2y = 10 is 2√55 -/
theorem intersection_distance : ∃ (p q : ℝ × ℝ),
  (p.2^2 = p.1 ∧ p.1 + 2*p.2 = 10) ∧
  (q.2^2 = q.1 ∧ q.1 + 2*q.2 = 10) ∧
  p ≠ q ∧
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = 2 * Real.sqrt 55 := by
  sorry

end NUMINAMATH_CALUDE_intersection_distance_l3140_314019


namespace NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l3140_314064

/-- An arithmetic sequence with 5 terms -/
structure ArithmeticSequence5 where
  a : ℝ  -- first term
  e : ℝ  -- last term
  y : ℝ  -- middle term

/-- Theorem: In an arithmetic sequence with 5 terms, where 12 is the first term,
    56 is the last term, and y is the middle term, y equals 34. -/
theorem arithmetic_sequence_middle_term 
  (seq : ArithmeticSequence5) 
  (h1 : seq.a = 12) 
  (h2 : seq.e = 56) : 
  seq.y = 34 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l3140_314064


namespace NUMINAMATH_CALUDE_simultaneous_equations_solution_l3140_314009

theorem simultaneous_equations_solution (m : ℝ) :
  (m ≠ 1) ↔ (∃ x y : ℝ, y = m * x + 2 ∧ y = (3 * m - 2) * x + 5) :=
by sorry

end NUMINAMATH_CALUDE_simultaneous_equations_solution_l3140_314009


namespace NUMINAMATH_CALUDE_intersection_theorem_l3140_314055

def A : Set ℝ := {x | x^2 + x - 6 ≤ 0}
def B : Set ℝ := {x | x + 1 < 0}

theorem intersection_theorem :
  A ∩ (Set.univ \ B) = Set.Icc (-1) 2 := by sorry

end NUMINAMATH_CALUDE_intersection_theorem_l3140_314055


namespace NUMINAMATH_CALUDE_square_of_one_plus_sqrt_two_l3140_314016

theorem square_of_one_plus_sqrt_two : (1 + Real.sqrt 2) ^ 2 = 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_one_plus_sqrt_two_l3140_314016


namespace NUMINAMATH_CALUDE_connie_watch_purchase_l3140_314090

/-- The additional amount Connie needs to buy a watch -/
def additional_amount (savings : ℕ) (watch_cost : ℕ) : ℕ :=
  watch_cost - savings

/-- Theorem: Given Connie's savings and the watch cost, prove the additional amount needed -/
theorem connie_watch_purchase (connie_savings : ℕ) (watch_price : ℕ) 
  (h1 : connie_savings = 39)
  (h2 : watch_price = 55) :
  additional_amount connie_savings watch_price = 16 := by
  sorry

end NUMINAMATH_CALUDE_connie_watch_purchase_l3140_314090


namespace NUMINAMATH_CALUDE_cheyenne_earnings_l3140_314079

def total_pots : ℕ := 80
def cracked_fraction : ℚ := 2/5
def price_per_pot : ℕ := 40

theorem cheyenne_earnings : 
  (total_pots - (cracked_fraction * total_pots).num) * price_per_pot = 1920 := by
  sorry

end NUMINAMATH_CALUDE_cheyenne_earnings_l3140_314079


namespace NUMINAMATH_CALUDE_seating_arrangements_l3140_314076

def total_people : ℕ := 10
def restricted_group : ℕ := 4

def arrangements_with_restriction (n : ℕ) (k : ℕ) : ℕ :=
  n.factorial - (n - k + 1).factorial * k.factorial

theorem seating_arrangements :
  arrangements_with_restriction total_people restricted_group = 3507840 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangements_l3140_314076


namespace NUMINAMATH_CALUDE_no_solution_for_all_a_b_l3140_314070

theorem no_solution_for_all_a_b : ∃ (a b : ℤ), a ≠ 0 ∧ b ≠ 0 ∧
  ¬∃ (x y : ℝ), (Real.tan (13 * x) * Real.tan (a * y) = 1) ∧
                (Real.tan (21 * x) * Real.tan (b * y) = 1) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_all_a_b_l3140_314070


namespace NUMINAMATH_CALUDE_intersection_and_perpendicular_line_l3140_314071

/-- Given three lines in the xy-plane:
    L₁: x + y - 2 = 0
    L₂: 3x + 2y - 5 = 0
    L₃: 3x + 4y - 12 = 0
    Prove that the line L: 4x - 3y - 1 = 0 passes through the intersection of L₁ and L₂,
    and is perpendicular to L₃. -/
theorem intersection_and_perpendicular_line 
  (L₁ : Set (ℝ × ℝ) := {p | p.1 + p.2 - 2 = 0})
  (L₂ : Set (ℝ × ℝ) := {p | 3 * p.1 + 2 * p.2 - 5 = 0})
  (L₃ : Set (ℝ × ℝ) := {p | 3 * p.1 + 4 * p.2 - 12 = 0})
  (L : Set (ℝ × ℝ) := {p | 4 * p.1 - 3 * p.2 - 1 = 0}) :
  (∃ p, p ∈ L₁ ∩ L₂ ∧ p ∈ L) ∧
  (∀ p q : ℝ × ℝ, p ≠ q → p ∈ L → q ∈ L → p ∈ L₃ → q ∈ L₃ → 
    (p.1 - q.1) * (p.1 - q.1) + (p.2 - q.2) * (p.2 - q.2) = 0) :=
by sorry

end NUMINAMATH_CALUDE_intersection_and_perpendicular_line_l3140_314071


namespace NUMINAMATH_CALUDE_parabola_coefficient_l3140_314041

/-- Given a parabola y = ax^2 + bx + c with vertex (p, kp) and y-intercept (0, -kp),
    where p ≠ 0 and k is a non-zero constant, prove that b = 4k/p -/
theorem parabola_coefficient (a b c p k : ℝ) (h1 : p ≠ 0) (h2 : k ≠ 0) : 
  (∀ x, a * x^2 + b * x + c = a * (x - p)^2 + k * p) →
  (a * 0^2 + b * 0 + c = -k * p) →
  b = 4 * k / p := by
  sorry

end NUMINAMATH_CALUDE_parabola_coefficient_l3140_314041


namespace NUMINAMATH_CALUDE_coordinates_sum_of_X_l3140_314073

def X : ℝ × ℝ := sorry
def Y : ℝ × ℝ := (3, 1)
def Z : ℝ × ℝ := (-1, 5)

theorem coordinates_sum_of_X :
  (X.1 + X.2 = 4) ∧
  (‖Z - X‖ / ‖Y - X‖ = 1/2) ∧
  (‖Y - Z‖ / ‖Y - X‖ = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_coordinates_sum_of_X_l3140_314073


namespace NUMINAMATH_CALUDE_lecture_orderings_l3140_314099

/-- Represents the number of lecturers --/
def n : ℕ := 7

/-- Represents the number of lecturers with specific ordering constraints --/
def k : ℕ := 3

/-- Calculates the number of valid orderings for n lecturers with k lecturers having specific ordering constraints --/
def validOrderings (n k : ℕ) : ℕ :=
  Nat.factorial (n - k + 1)

/-- Theorem stating that the number of valid orderings for 7 lecturers with 3 having specific constraints is 120 --/
theorem lecture_orderings : validOrderings n k = 120 := by
  sorry

end NUMINAMATH_CALUDE_lecture_orderings_l3140_314099


namespace NUMINAMATH_CALUDE_proposition_b_is_true_l3140_314088

theorem proposition_b_is_true : ∀ (a b : ℝ), a + b ≠ 6 → a ≠ 3 ∨ b ≠ 3 := by
  sorry

end NUMINAMATH_CALUDE_proposition_b_is_true_l3140_314088


namespace NUMINAMATH_CALUDE_distance_from_rate_and_time_l3140_314083

/-- Proves that given a constant walking rate and time, the distance covered is equal to the product of rate and time. -/
theorem distance_from_rate_and_time 
  (rate : ℝ) 
  (time : ℝ) 
  (h_rate : rate = 4) 
  (h_time : time = 2) : 
  rate * time = 8 := by
  sorry

#check distance_from_rate_and_time

end NUMINAMATH_CALUDE_distance_from_rate_and_time_l3140_314083


namespace NUMINAMATH_CALUDE_constant_theta_and_z_forms_line_l3140_314062

/-- A point in cylindrical coordinates -/
structure CylindricalPoint where
  r : ℝ
  θ : ℝ
  z : ℝ

/-- The set of points satisfying θ = c and z = d -/
def ConstantThetaAndZ (c d : ℝ) : Set CylindricalPoint :=
  {p : CylindricalPoint | p.θ = c ∧ p.z = d}

/-- Definition of a line in cylindrical coordinates -/
def IsLine (S : Set CylindricalPoint) : Prop :=
  ∃ (a b : ℝ), ∀ p ∈ S, p.r = a * p.θ + b

theorem constant_theta_and_z_forms_line (c d : ℝ) :
  IsLine (ConstantThetaAndZ c d) := by
  sorry


end NUMINAMATH_CALUDE_constant_theta_and_z_forms_line_l3140_314062


namespace NUMINAMATH_CALUDE_simplify_expression_l3140_314045

theorem simplify_expression (x : ℝ) : (3*x)^5 + (4*x)*(x^4) = 247*x^5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3140_314045


namespace NUMINAMATH_CALUDE_cake_division_theorem_l3140_314037

/-- Represents a piece of cake -/
structure CakePiece where
  cookies : ℕ
  roses : ℕ

/-- Represents the whole cake -/
structure Cake where
  totalCookies : ℕ
  totalRoses : ℕ
  pieces : ℕ

/-- Checks if a cake can be evenly divided -/
def isEvenlyDivisible (c : Cake) : Prop :=
  c.totalCookies % c.pieces = 0 ∧ c.totalRoses % c.pieces = 0

/-- Calculates the content of each piece when the cake is evenly divided -/
def pieceContent (c : Cake) (h : isEvenlyDivisible c) : CakePiece :=
  { cookies := c.totalCookies / c.pieces
  , roses := c.totalRoses / c.pieces }

/-- Theorem: If a cake with 48 cookies and 4 roses is cut into 4 equal pieces,
    each piece will have 12 cookies and 1 rose -/
theorem cake_division_theorem (c : Cake)
    (h1 : c.totalCookies = 48)
    (h2 : c.totalRoses = 4)
    (h3 : c.pieces = 4)
    (h4 : isEvenlyDivisible c) :
    pieceContent c h4 = { cookies := 12, roses := 1 } := by
  sorry


end NUMINAMATH_CALUDE_cake_division_theorem_l3140_314037


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3140_314034

theorem inequality_equivalence :
  ∀ y : ℝ, (3 ≤ |y - 4| ∧ |y - 4| ≤ 7) ↔ ((7 ≤ y ∧ y ≤ 11) ∨ (-3 ≤ y ∧ y ≤ 1)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3140_314034


namespace NUMINAMATH_CALUDE_same_function_D_l3140_314047

theorem same_function_D (x : ℝ) (h : x ≠ 1) : (x - 1) ^ 0 = (1 : ℝ) / ((x - 1) ^ 0) := by
  sorry

end NUMINAMATH_CALUDE_same_function_D_l3140_314047


namespace NUMINAMATH_CALUDE_unique_function_solution_l3140_314021

theorem unique_function_solution (f : ℝ → ℝ) :
  (∀ x : ℝ, 2 * f x + f (1 - x) = x^2) ↔ (∀ x : ℝ, f x = (1/3) * (x^2 + 2*x - 1)) :=
sorry

end NUMINAMATH_CALUDE_unique_function_solution_l3140_314021


namespace NUMINAMATH_CALUDE_blossom_room_area_l3140_314033

/-- Converts feet and inches to centimeters -/
def to_cm (feet : ℕ) (inches : ℕ) : ℝ :=
  (feet : ℝ) * 30.48 + (inches : ℝ) * 2.54

/-- Calculates the area of a room in square centimeters -/
def room_area (length_feet : ℕ) (length_inches : ℕ) (width_feet : ℕ) (width_inches : ℕ) : ℝ :=
  (to_cm length_feet length_inches) * (to_cm width_feet width_inches)

theorem blossom_room_area :
  room_area 14 8 10 5 = 141935.4 := by
  sorry

end NUMINAMATH_CALUDE_blossom_room_area_l3140_314033


namespace NUMINAMATH_CALUDE_sugar_calculation_l3140_314007

/-- The amount of sugar Pamela spilled in ounces -/
def spilled_sugar : ℝ := 5.2

/-- The amount of sugar left after spilling in ounces -/
def remaining_sugar : ℝ := 4.6

/-- The initial amount of sugar Pamela bought in ounces -/
def initial_sugar : ℝ := spilled_sugar + remaining_sugar

theorem sugar_calculation : initial_sugar = 9.8 := by
  sorry

end NUMINAMATH_CALUDE_sugar_calculation_l3140_314007


namespace NUMINAMATH_CALUDE_arithmetic_sequence_n_value_l3140_314024

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_n_value
  (a : ℕ → ℝ)
  (h_arithmetic : is_arithmetic_sequence a)
  (h_a4 : a 4 = 7)
  (h_a3_a6 : a 3 + a 6 = 16)
  (h_an : ∃ n : ℕ, a n = 31) :
  ∃ n : ℕ, a n = 31 ∧ n = 16 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_n_value_l3140_314024


namespace NUMINAMATH_CALUDE_nested_square_root_equality_l3140_314091

theorem nested_square_root_equality : Real.sqrt (25 * Real.sqrt (15 * Real.sqrt 9)) = 5 * Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_nested_square_root_equality_l3140_314091


namespace NUMINAMATH_CALUDE_circle_symmetry_l3140_314067

-- Define the original circle
def original_circle (x y : ℝ) : Prop :=
  x^2 + y^2 - 8*x + 4*y + 19 = 0

-- Define the line of symmetry
def symmetry_line (x y : ℝ) : Prop :=
  x + y + 1 = 0

-- Define the symmetric circle
def symmetric_circle (x y : ℝ) : Prop :=
  (x - 1)^2 + (y + 5)^2 = 1

-- Theorem statement
theorem circle_symmetry :
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
    original_circle x₁ y₁ →
    symmetric_circle x₂ y₂ →
    ∃ (x_mid y_mid : ℝ),
      symmetry_line x_mid y_mid ∧
      x_mid = (x₁ + x₂) / 2 ∧
      y_mid = (y₁ + y₂) / 2 :=
sorry

end NUMINAMATH_CALUDE_circle_symmetry_l3140_314067


namespace NUMINAMATH_CALUDE_max_ben_cookies_proof_l3140_314065

/-- The maximum number of cookies Ben can eat when sharing with Beth -/
def max_ben_cookies : ℕ := 12

/-- The total number of cookies shared between Ben and Beth -/
def total_cookies : ℕ := 36

/-- Predicate to check if a given number of cookies for Ben is valid -/
def valid_ben_cookies (ben : ℕ) : Prop :=
  (ben + 2 * ben = total_cookies) ∨ (ben + 3 * ben = total_cookies)

theorem max_ben_cookies_proof :
  (∀ ben : ℕ, valid_ben_cookies ben → ben ≤ max_ben_cookies) ∧
  valid_ben_cookies max_ben_cookies :=
sorry

end NUMINAMATH_CALUDE_max_ben_cookies_proof_l3140_314065


namespace NUMINAMATH_CALUDE_income_comparison_l3140_314030

theorem income_comparison (juan tim mart : ℝ) 
  (h1 : tim = 0.6 * juan) 
  (h2 : mart = 0.84 * juan) : 
  (mart - tim) / tim = 0.4 := by
sorry

end NUMINAMATH_CALUDE_income_comparison_l3140_314030


namespace NUMINAMATH_CALUDE_frequency_converges_to_probability_l3140_314002

/-- A random event in an experiment. -/
structure Event where
  -- Add necessary fields here
  mk :: -- Constructor

/-- The frequency of an event after a given number of trials. -/
def frequency (e : Event) (n : ℕ) : ℝ :=
  sorry

/-- The probability of an event. -/
def probability (e : Event) : ℝ :=
  sorry

/-- Statement: As the number of trials increases, the frequency of an event
    converges to its probability. -/
theorem frequency_converges_to_probability (e : Event) :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |frequency e n - probability e| < ε :=
sorry

end NUMINAMATH_CALUDE_frequency_converges_to_probability_l3140_314002


namespace NUMINAMATH_CALUDE_hall_mat_expenditure_l3140_314063

/-- The total expenditure to cover the floor of a rectangular hall with a mat -/
def total_expenditure (length width height cost_per_sqm : ℝ) : ℝ :=
  length * width * cost_per_sqm

/-- Theorem: The total expenditure to cover the floor of a rectangular hall
    with dimensions 20 m × 15 m × 5 m using a mat that costs Rs. 40 per square meter
    is equal to Rs. 12,000. -/
theorem hall_mat_expenditure :
  total_expenditure 20 15 5 40 = 12000 := by
  sorry

end NUMINAMATH_CALUDE_hall_mat_expenditure_l3140_314063


namespace NUMINAMATH_CALUDE_line_through_points_l3140_314080

-- Define the line equation
def line_equation (a b x : ℝ) : ℝ := a * x + b

-- Define the theorem
theorem line_through_points :
  ∀ (a b : ℝ),
  (line_equation a b 6 = 7) →
  (line_equation a b 10 = 23) →
  a + b = -13 := by
sorry

end NUMINAMATH_CALUDE_line_through_points_l3140_314080


namespace NUMINAMATH_CALUDE_maria_paper_count_l3140_314043

/-- Represents the number of sheets of paper -/
structure PaperCount where
  whole : ℕ
  half : ℕ

/-- Calculates the remaining papers after giving away and folding -/
def remaining_papers (desk : ℕ) (backpack : ℕ) (given_away : ℕ) (folded : ℕ) : PaperCount :=
  { whole := desk + backpack - given_away - folded,
    half := folded }

theorem maria_paper_count : 
  ∀ (x y : ℕ), x ≤ 91 → y ≤ 91 - x → 
  remaining_papers 50 41 x y = { whole := 91 - x - y, half := y } := by
sorry

end NUMINAMATH_CALUDE_maria_paper_count_l3140_314043


namespace NUMINAMATH_CALUDE_same_odd_dice_probability_l3140_314035

/-- The number of faces on each die -/
def num_faces : ℕ := 8

/-- The number of odd faces on each die -/
def num_odd_faces : ℕ := 4

/-- The number of dice rolled -/
def num_dice : ℕ := 4

/-- The probability of rolling the same odd number on all dice -/
def prob_same_odd : ℚ := 1 / 1024

theorem same_odd_dice_probability :
  (num_odd_faces : ℚ) / num_faces * (1 / num_faces) ^ (num_dice - 1) = prob_same_odd :=
by sorry

end NUMINAMATH_CALUDE_same_odd_dice_probability_l3140_314035


namespace NUMINAMATH_CALUDE_dividend_calculation_l3140_314053

theorem dividend_calculation (divisor quotient remainder : ℕ) : 
  divisor = 20 * quotient →
  divisor = 10 * remainder →
  remainder = 100 →
  divisor * quotient + remainder = 50100 :=
by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l3140_314053


namespace NUMINAMATH_CALUDE_cookie_radius_l3140_314092

theorem cookie_radius (x y : ℝ) :
  x^2 + y^2 + 26 = 6*x + 12*y →
  ∃ (center_x center_y : ℝ), (x - center_x)^2 + (y - center_y)^2 = 19 :=
by sorry

end NUMINAMATH_CALUDE_cookie_radius_l3140_314092


namespace NUMINAMATH_CALUDE_blue_eyed_blonde_proportion_l3140_314095

/-- Proves that if the proportion of blondes among blue-eyed people is greater
    than the proportion of blondes among all people, then the proportion of
    blue-eyed people among blondes is greater than the proportion of blue-eyed
    people among all people. -/
theorem blue_eyed_blonde_proportion
  (l : ℕ) -- total number of people
  (g : ℕ) -- number of blue-eyed people
  (b : ℕ) -- number of blond-haired people
  (a : ℕ) -- number of people who are both blue-eyed and blond-haired
  (hl : l > 0)
  (hg : g > 0)
  (hb : b > 0)
  (ha : a > 0)
  (h_subset : a ≤ g ∧ a ≤ b ∧ g ≤ l ∧ b ≤ l)
  (h_proportion : (a : ℚ) / g > (b : ℚ) / l) :
  (a : ℚ) / b > (g : ℚ) / l :=
sorry

end NUMINAMATH_CALUDE_blue_eyed_blonde_proportion_l3140_314095


namespace NUMINAMATH_CALUDE_mike_total_cards_l3140_314001

/-- The total number of baseball cards Mike has after his birthday -/
def total_cards (initial_cards birthday_cards : ℕ) : ℕ :=
  initial_cards + birthday_cards

/-- Theorem stating that Mike has 82 cards in total -/
theorem mike_total_cards : 
  total_cards 64 18 = 82 := by
  sorry

end NUMINAMATH_CALUDE_mike_total_cards_l3140_314001


namespace NUMINAMATH_CALUDE_hexagon_perimeter_is_24_l3140_314036

/-- Represents the side length of the larger equilateral triangle -/
def large_triangle_side : ℝ := 4

/-- Represents the side length of the regular hexagon -/
def hexagon_side : ℝ := large_triangle_side

/-- The number of sides in a regular hexagon -/
def hexagon_sides : ℕ := 6

/-- Calculates the perimeter of the regular hexagon -/
def hexagon_perimeter : ℝ := hexagon_side * hexagon_sides

theorem hexagon_perimeter_is_24 : hexagon_perimeter = 24 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_perimeter_is_24_l3140_314036


namespace NUMINAMATH_CALUDE_equal_slopes_imply_equal_angles_l3140_314056

/-- Theorem: For two lines with inclination angles in [0, π) and equal slopes, their inclination angles are equal. -/
theorem equal_slopes_imply_equal_angles (α₁ α₂ : Real) (k₁ k₂ : Real) :
  0 ≤ α₁ ∧ α₁ < π →
  0 ≤ α₂ ∧ α₂ < π →
  k₁ = Real.tan α₁ →
  k₂ = Real.tan α₂ →
  k₁ = k₂ →
  α₁ = α₂ := by
  sorry

end NUMINAMATH_CALUDE_equal_slopes_imply_equal_angles_l3140_314056


namespace NUMINAMATH_CALUDE_unpaired_numbers_mod_6_l3140_314038

theorem unpaired_numbers_mod_6 (n : ℕ) (hn : n = 800) : 
  ¬ (∃ (f : ℕ → ℕ), 
    (∀ x ∈ Finset.range n, f (f x) = x ∧ x ≠ f x) ∧ 
    (∀ x ∈ Finset.range n, (x + f x) % 6 = 0)) := by
  sorry

end NUMINAMATH_CALUDE_unpaired_numbers_mod_6_l3140_314038


namespace NUMINAMATH_CALUDE_even_function_positive_x_l3140_314027

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem even_function_positive_x 
  (f : ℝ → ℝ) 
  (h_even : is_even_function f) 
  (h_neg : ∀ x < 0, f x = x * (x - 1)) : 
  ∀ x > 0, f x = x * (x + 1) := by
sorry

end NUMINAMATH_CALUDE_even_function_positive_x_l3140_314027


namespace NUMINAMATH_CALUDE_john_account_balance_l3140_314046

/-- Calculates the final balance after a deposit and withdrawal -/
def finalBalance (initialBalance deposit withdrawal : ℝ) : ℝ :=
  initialBalance + deposit - withdrawal

/-- Theorem stating that given the specific amounts, the final balance is $43.8 -/
theorem john_account_balance :
  finalBalance 45.7 18.6 20.5 = 43.8 := by
  sorry

end NUMINAMATH_CALUDE_john_account_balance_l3140_314046


namespace NUMINAMATH_CALUDE_last_two_digits_product_l3140_314018

theorem last_two_digits_product (n : ℕ) : 
  (n % 100 ≥ 10) →  -- Ensure n has at least two digits
  (n % 4 = 0) →     -- Divisible by 4
  (n % 3 = 0) →     -- Divisible by 3
  ((n % 100) / 10 + n % 10 = 12) →  -- Sum of last two digits is 12
  ((n % 100) / 10 * (n % 10) = 32) :=
by sorry

end NUMINAMATH_CALUDE_last_two_digits_product_l3140_314018


namespace NUMINAMATH_CALUDE_games_per_box_l3140_314050

theorem games_per_box (initial_games : ℕ) (sold_games : ℕ) (num_boxes : ℕ) :
  initial_games = 76 →
  sold_games = 46 →
  num_boxes = 6 →
  (initial_games - sold_games) / num_boxes = 5 := by
  sorry

end NUMINAMATH_CALUDE_games_per_box_l3140_314050


namespace NUMINAMATH_CALUDE_can_calculate_average_if_complete_info_cannot_calculate_camerons_average_l3140_314004

/-- Represents a tour guide's daily work --/
structure TourGuideDay where
  numTours : Nat
  totalQuestions : Nat
  groupSizes : List Nat

/-- Calculates the average number of questions per tourist --/
def averageQuestionsPerTourist (day : TourGuideDay) : Option ℚ :=
  if day.groupSizes.length = day.numTours ∧ day.groupSizes.sum ≠ 0 then
    some ((day.totalQuestions : ℚ) / (day.groupSizes.sum : ℚ))
  else
    none

/-- Theorem: If we have complete information, we can calculate the average questions per tourist --/
theorem can_calculate_average_if_complete_info (day : TourGuideDay) :
    day.groupSizes.length = day.numTours ∧ day.groupSizes.sum ≠ 0 →
    ∃ avg : ℚ, averageQuestionsPerTourist day = some avg :=
  sorry

/-- Cameron's specific day --/
def cameronsDay : TourGuideDay :=
  { numTours := 4
  , totalQuestions := 68
  , groupSizes := [] }  -- Empty list because we don't know the group sizes

/-- Theorem: We cannot calculate the average for Cameron's day due to missing information --/
theorem cannot_calculate_camerons_average :
    averageQuestionsPerTourist cameronsDay = none :=
  sorry

end NUMINAMATH_CALUDE_can_calculate_average_if_complete_info_cannot_calculate_camerons_average_l3140_314004


namespace NUMINAMATH_CALUDE_floor_plus_x_eq_13_4_l3140_314074

theorem floor_plus_x_eq_13_4 :
  ∃! x : ℝ, ⌊x⌋ + x = 13.4 ∧ x = 6.4 := by sorry

end NUMINAMATH_CALUDE_floor_plus_x_eq_13_4_l3140_314074


namespace NUMINAMATH_CALUDE_cereal_expense_per_year_l3140_314066

def boxes_per_week : ℕ := 2
def cost_per_box : ℚ := 3
def weeks_per_year : ℕ := 52

theorem cereal_expense_per_year :
  (boxes_per_week * weeks_per_year * cost_per_box : ℚ) = 312 := by
  sorry

end NUMINAMATH_CALUDE_cereal_expense_per_year_l3140_314066


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3140_314087

theorem quadratic_inequality_range (k : ℝ) : 
  (∀ x : ℝ, 2 * k * x^2 + k * x - 3/8 < 0) ↔ k ∈ Set.Ioo (-3) 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3140_314087


namespace NUMINAMATH_CALUDE_experiment_sequences_l3140_314011

def num_procedures : ℕ → ℕ
  | n => 4 * Nat.factorial (n - 3)

theorem experiment_sequences (n : ℕ) (h : n ≥ 3) : num_procedures n = 96 := by
  sorry

end NUMINAMATH_CALUDE_experiment_sequences_l3140_314011


namespace NUMINAMATH_CALUDE_max_y_coordinate_polar_curve_l3140_314078

/-- The maximum y-coordinate of a point on the graph of r = cos 2θ is √2/2 -/
theorem max_y_coordinate_polar_curve : 
  let r : ℝ → ℝ := λ θ => Real.cos (2 * θ)
  let y : ℝ → ℝ := λ θ => (r θ) * Real.sin θ
  ∃ (θ_max : ℝ), ∀ (θ : ℝ), y θ ≤ y θ_max ∧ y θ_max = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_max_y_coordinate_polar_curve_l3140_314078


namespace NUMINAMATH_CALUDE_staffing_problem_l3140_314008

def number_of_staffing_ways (total_candidates : ℕ) (qualified_for_first : ℕ) (positions : ℕ) : ℕ :=
  qualified_for_first * (List.range (positions - 1)).foldl (fun acc i => acc * (total_candidates - i - 1)) 1

theorem staffing_problem (total_candidates : ℕ) (qualified_for_first : ℕ) (positions : ℕ) 
  (h1 : total_candidates = 15)
  (h2 : qualified_for_first = 8)
  (h3 : positions = 5)
  (h4 : qualified_for_first ≤ total_candidates) :
  number_of_staffing_ways total_candidates qualified_for_first positions = 17472 := by
  sorry

end NUMINAMATH_CALUDE_staffing_problem_l3140_314008


namespace NUMINAMATH_CALUDE_questionnaire_responses_l3140_314028

theorem questionnaire_responses (response_rate : ℝ) (min_questionnaires : ℕ) (responses_needed : ℕ) : 
  response_rate = 0.60 → 
  min_questionnaires = 370 → 
  responses_needed = ⌊response_rate * min_questionnaires⌋ →
  responses_needed = 222 := by
sorry

end NUMINAMATH_CALUDE_questionnaire_responses_l3140_314028


namespace NUMINAMATH_CALUDE_length_A_l3140_314077

-- Define the points
def A : ℝ × ℝ := (0, 6)
def B : ℝ × ℝ := (0, 10)
def C : ℝ × ℝ := (3, 7)

-- Define the line y = x
def line_y_eq_x (p : ℝ × ℝ) : Prop := p.2 = p.1

-- Define the condition that A' and B' are on the line y = x
axiom A'_on_line : ∃ A' : ℝ × ℝ, line_y_eq_x A'
axiom B'_on_line : ∃ B' : ℝ × ℝ, line_y_eq_x B'

-- Define the condition that AA' and BB' intersect at C
axiom AA'_BB'_intersect_at_C : 
  ∃ A' B' : ℝ × ℝ, line_y_eq_x A' ∧ line_y_eq_x B' ∧
  (∃ t₁ t₂ : ℝ, A + t₁ • (A' - A) = C ∧ B + t₂ • (B' - B) = C)

-- State the theorem
theorem length_A'B'_is_4_sqrt_2 : 
  ∃ A' B' : ℝ × ℝ, line_y_eq_x A' ∧ line_y_eq_x B' ∧
  (∃ t₁ t₂ : ℝ, A + t₁ • (A' - A) = C ∧ B + t₂ • (B' - B) = C) ∧
  Real.sqrt ((A'.1 - B'.1)^2 + (A'.2 - B'.2)^2) = 4 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_length_A_l3140_314077


namespace NUMINAMATH_CALUDE_consecutive_even_numbers_sum_l3140_314005

theorem consecutive_even_numbers_sum (n : ℤ) : 
  (∃ (a b c d : ℤ), 
    a = n ∧ b = n + 2 ∧ c = n + 4 ∧ d = n + 6 ∧  -- four consecutive even numbers
    a ^ 2 + b ^ 2 + c ^ 2 + d ^ 2 = 344) →        -- sum of squares is 344
  (n + (n + 2) + (n + 4) + (n + 6) = 36) :=       -- sum of the numbers is 36
by
  sorry

end NUMINAMATH_CALUDE_consecutive_even_numbers_sum_l3140_314005


namespace NUMINAMATH_CALUDE_cave_door_weight_calculation_l3140_314048

/-- The weight already on the switch (in pounds) -/
def weight_on_switch : ℕ := 234

/-- The total weight needed to open the cave doors (in pounds) -/
def total_weight_needed : ℕ := 712

/-- The additional weight needed to open the cave doors (in pounds) -/
def additional_weight_needed : ℕ := total_weight_needed - weight_on_switch

theorem cave_door_weight_calculation :
  additional_weight_needed = 478 := by
  sorry

end NUMINAMATH_CALUDE_cave_door_weight_calculation_l3140_314048


namespace NUMINAMATH_CALUDE_distance_after_translation_l3140_314093

/-- Given two points A and B in a 2D plane, and a translation vector,
    prove that the distance between A and the translated B is √153. -/
theorem distance_after_translation :
  let A : ℝ × ℝ := (2, -2)
  let B : ℝ × ℝ := (8, 6)
  let translation : ℝ × ℝ := (-3, 4)
  let C : ℝ × ℝ := (B.1 + translation.1, B.2 + translation.2)
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = 153 := by
  sorry

end NUMINAMATH_CALUDE_distance_after_translation_l3140_314093


namespace NUMINAMATH_CALUDE_negative_abs_comparison_l3140_314042

theorem negative_abs_comparison : -|(-8 : ℤ)| < -6 := by
  sorry

end NUMINAMATH_CALUDE_negative_abs_comparison_l3140_314042


namespace NUMINAMATH_CALUDE_absolute_value_of_seven_minus_sqrt_53_l3140_314082

theorem absolute_value_of_seven_minus_sqrt_53 :
  |7 - Real.sqrt 53| = Real.sqrt 53 - 7 := by sorry

end NUMINAMATH_CALUDE_absolute_value_of_seven_minus_sqrt_53_l3140_314082


namespace NUMINAMATH_CALUDE_circumcircle_equation_incircle_equation_l3140_314032

-- Define the Triangle type
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the Circle type
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def Triangle1 : Triangle := { A := (5, 1), B := (7, -3), C := (2, -8) }
def Triangle2 : Triangle := { A := (0, 0), B := (5, 0), C := (0, 12) }

def CircumcircleEquation (t : Triangle) (c : Circle) : Prop :=
  ∀ (x y : ℝ), (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2 ↔
    (x = t.A.1 ∧ y = t.A.2) ∨ (x = t.B.1 ∧ y = t.B.2) ∨ (x = t.C.1 ∧ y = t.C.2)

def IncircleEquation (t : Triangle) (c : Circle) : Prop :=
  ∀ (x y : ℝ), (x - c.center.1)^2 + (y - c.center.2)^2 ≤ c.radius^2 ↔
    (y ≥ 0 ∧ y ≤ 12 ∧ x ≥ 0 ∧ 5*y + 12*x ≤ 60)

theorem circumcircle_equation (t : Triangle) (h : t = Triangle1) :
  CircumcircleEquation t { center := (2, -3), radius := 5 } := by sorry

theorem incircle_equation (t : Triangle) (h : t = Triangle2) :
  IncircleEquation t { center := (2, 2), radius := 2 } := by sorry

end NUMINAMATH_CALUDE_circumcircle_equation_incircle_equation_l3140_314032


namespace NUMINAMATH_CALUDE_tangent_parallel_at_minus_one_l3140_314015

-- Define the curve
def f (x : ℝ) : ℝ := x^3 + x - 2

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

-- Theorem statement
theorem tangent_parallel_at_minus_one :
  ∃ (P : ℝ × ℝ), P.1 = -1 ∧ P.2 = f P.1 ∧ f' P.1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_tangent_parallel_at_minus_one_l3140_314015


namespace NUMINAMATH_CALUDE_only_rational_root_l3140_314068

def polynomial (x : ℚ) : ℚ := 6 * x^4 - 5 * x^3 - 17 * x^2 + 7 * x + 3

theorem only_rational_root :
  ∀ x : ℚ, polynomial x = 0 ↔ x = 1/2 := by sorry

end NUMINAMATH_CALUDE_only_rational_root_l3140_314068


namespace NUMINAMATH_CALUDE_seminar_invitations_count_l3140_314086

/-- The number of ways to select k items from n items -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The number of ways to select 6 teachers out of 10 for a seminar,
    where two specific teachers (A and B) cannot attend together -/
def seminar_invitations : ℕ :=
  2 * binomial 8 5 + binomial 8 6

theorem seminar_invitations_count : seminar_invitations = 140 := by
  sorry

end NUMINAMATH_CALUDE_seminar_invitations_count_l3140_314086


namespace NUMINAMATH_CALUDE_common_chord_equation_l3140_314058

/-- The equation of the common chord of two circles -/
theorem common_chord_equation (x y : ℝ) :
  (x^2 + y^2 + 2*x = 0) ∧ (x^2 + y^2 - 4*y = 0) → (x + 2*y = 0) := by
  sorry

end NUMINAMATH_CALUDE_common_chord_equation_l3140_314058


namespace NUMINAMATH_CALUDE_power_equation_solution_l3140_314097

theorem power_equation_solution (n : ℕ) : 5^29 * 4^15 = 2 * 10^n → n = 29 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l3140_314097


namespace NUMINAMATH_CALUDE_binomial_25_5_l3140_314020

theorem binomial_25_5 (h1 : (23 : ℕ).choose 3 = 1771)
                      (h2 : (23 : ℕ).choose 4 = 8855)
                      (h3 : (23 : ℕ).choose 5 = 33649) :
  (25 : ℕ).choose 5 = 53130 := by
  sorry

end NUMINAMATH_CALUDE_binomial_25_5_l3140_314020


namespace NUMINAMATH_CALUDE_smaller_solution_quadratic_l3140_314051

theorem smaller_solution_quadratic (x : ℝ) : 
  (x^2 - 13*x - 30 = 0) → 
  (∃ y : ℝ, y ≠ x ∧ y^2 - 13*y - 30 = 0) → 
  (x = -2 ∨ x = 15) ∧ 
  (∀ y : ℝ, y^2 - 13*y - 30 = 0 → y ≥ -2) :=
by sorry

end NUMINAMATH_CALUDE_smaller_solution_quadratic_l3140_314051


namespace NUMINAMATH_CALUDE_angle_inequality_l3140_314017

theorem angle_inequality (x : Real) :
  x ∈ Set.Ioo 0 (2 * Real.pi) →
  (2^x * (2 * Real.sin x - Real.sqrt 3) ≥ 0) ↔
  x ∈ Set.Icc (Real.pi / 3) (2 * Real.pi / 3) := by
  sorry

end NUMINAMATH_CALUDE_angle_inequality_l3140_314017


namespace NUMINAMATH_CALUDE_sum_of_variables_l3140_314006

theorem sum_of_variables (a b c d : ℚ) 
  (h1 : 2*a + 3 = 2*b + 5)
  (h2 : 2*b + 5 = 2*c + 7)
  (h3 : 2*c + 7 = 2*d + 9)
  (h4 : 2*d + 9 = 2*(a + b + c + d) + 13) :
  a + b + c + d = -14/3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_variables_l3140_314006


namespace NUMINAMATH_CALUDE_only_C_is_comprehensive_unique_comprehensive_survey_l3140_314029

/-- Represents a survey option -/
inductive SurveyOption
| A  -- Survey of the environmental awareness of the people nationwide
| B  -- Survey of the quality of mooncakes in the market during the Mid-Autumn Festival
| C  -- Survey of the weight of 40 students in a class
| D  -- Survey of the safety and quality of a certain type of fireworks and firecrackers

/-- Defines what makes a survey comprehensive -/
def isComprehensive (s : SurveyOption) : Prop :=
  match s with
  | SurveyOption.C => true
  | _ => false

/-- Theorem stating that only option C is suitable for a comprehensive survey -/
theorem only_C_is_comprehensive :
  ∀ s : SurveyOption, isComprehensive s ↔ s = SurveyOption.C :=
by sorry

/-- Corollary: There exists exactly one comprehensive survey option -/
theorem unique_comprehensive_survey :
  ∃! s : SurveyOption, isComprehensive s :=
by sorry

end NUMINAMATH_CALUDE_only_C_is_comprehensive_unique_comprehensive_survey_l3140_314029


namespace NUMINAMATH_CALUDE_paint_for_similar_statues_l3140_314023

/-- The amount of paint needed for similar statues -/
theorem paint_for_similar_statues
  (original_height : ℝ)
  (original_paint : ℝ)
  (new_height : ℝ)
  (num_statues : ℕ)
  (h1 : original_height = 8)
  (h2 : original_paint = 1)
  (h3 : new_height = 2)
  (h4 : num_statues = 320) :
  (num_statues : ℝ) * original_paint * (new_height / original_height) ^ 2 = 20 :=
by sorry

end NUMINAMATH_CALUDE_paint_for_similar_statues_l3140_314023


namespace NUMINAMATH_CALUDE_g_definition_l3140_314084

-- Define the function g
def g : ℝ → ℝ := fun x ↦ 2*x - 11

-- Theorem statement
theorem g_definition : ∀ x : ℝ, g (x + 2) = 2*x - 3 := by
  sorry

end NUMINAMATH_CALUDE_g_definition_l3140_314084


namespace NUMINAMATH_CALUDE_unintended_texts_per_week_l3140_314072

theorem unintended_texts_per_week 
  (old_daily_texts : ℕ) 
  (new_daily_texts : ℕ) 
  (days_in_week : ℕ) 
  (h1 : old_daily_texts = 20)
  (h2 : new_daily_texts = 55)
  (h3 : days_in_week = 7) :
  (new_daily_texts - old_daily_texts) * days_in_week = 245 :=
by sorry

end NUMINAMATH_CALUDE_unintended_texts_per_week_l3140_314072


namespace NUMINAMATH_CALUDE_product_pure_imaginary_solution_l3140_314052

theorem product_pure_imaginary_solution (x : ℝ) : 
  (∃ y : ℝ, (x + 2 * Complex.I) * ((x + 1) + 2 * Complex.I) * ((x + 2) + 2 * Complex.I) = y * Complex.I) ↔ 
  x = 1 :=
by sorry

end NUMINAMATH_CALUDE_product_pure_imaginary_solution_l3140_314052


namespace NUMINAMATH_CALUDE_kim_no_math_test_probability_kim_math_test_probability_probability_kim_no_math_test_l3140_314022

theorem kim_no_math_test_probability : ℚ → ℚ
  | p => 1 - p

theorem kim_math_test_probability : ℚ := 5/8

theorem probability_kim_no_math_test :
  kim_no_math_test_probability kim_math_test_probability = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_kim_no_math_test_probability_kim_math_test_probability_probability_kim_no_math_test_l3140_314022


namespace NUMINAMATH_CALUDE_solve_equation_l3140_314085

theorem solve_equation (x : ℚ) (h : 5 * x - 8 = 15 * x + 18) : 3 * (x + 9) = 96 / 5 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3140_314085


namespace NUMINAMATH_CALUDE_cat_litter_cost_210_days_l3140_314031

/-- Calculates the cost of cat litter for a given number of days -/
def catLitterCost (containerSize : ℕ) (containerPrice : ℕ) (litterBoxCapacity : ℕ) (changeDays : ℕ) (totalDays : ℕ) : ℕ :=
  let changes := totalDays / changeDays
  let totalLitter := changes * litterBoxCapacity
  let containers := (totalLitter + containerSize - 1) / containerSize  -- Ceiling division
  containers * containerPrice

/-- The cost of cat litter for 210 days is $210 -/
theorem cat_litter_cost_210_days :
  catLitterCost 45 21 15 7 210 = 210 := by sorry

end NUMINAMATH_CALUDE_cat_litter_cost_210_days_l3140_314031


namespace NUMINAMATH_CALUDE_parabola_y_range_l3140_314025

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 8*y

-- Define the focus-to-point distance
def focus_distance (y : ℝ) : ℝ := y + 2

-- Define the condition for intersection with directrix
def intersects_directrix (y : ℝ) : Prop := focus_distance y > 4

theorem parabola_y_range (x y : ℝ) :
  parabola x y → intersects_directrix y → y > 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_y_range_l3140_314025


namespace NUMINAMATH_CALUDE_barney_towel_count_l3140_314003

/-- The number of towels Barney owns -/
def num_towels : ℕ := 18

/-- The number of towels Barney uses per day -/
def towels_per_day : ℕ := 2

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of days Barney can use clean towels before running out -/
def days_before_running_out : ℕ := 9

/-- Theorem stating that Barney owns 18 towels -/
theorem barney_towel_count : 
  num_towels = towels_per_day * days_before_running_out :=
by sorry

end NUMINAMATH_CALUDE_barney_towel_count_l3140_314003


namespace NUMINAMATH_CALUDE_jays_change_l3140_314049

def book_cost : ℕ := 25
def pen_cost : ℕ := 4
def ruler_cost : ℕ := 1
def amount_paid : ℕ := 50

def total_cost : ℕ := book_cost + pen_cost + ruler_cost

theorem jays_change (change : ℕ) : change = amount_paid - total_cost → change = 20 := by
  sorry

end NUMINAMATH_CALUDE_jays_change_l3140_314049


namespace NUMINAMATH_CALUDE_ascending_order_proof_l3140_314040

theorem ascending_order_proof :
  222^2 < 2^(2^(2^2)) ∧
  2^(2^(2^2)) < 22^(2^2) ∧
  22^(2^2) < 22^22 ∧
  22^22 < 2^222 ∧
  2^222 < 2^(22^2) ∧
  2^(22^2) < 2^(2^22) := by
  sorry

end NUMINAMATH_CALUDE_ascending_order_proof_l3140_314040


namespace NUMINAMATH_CALUDE_expected_winnings_is_one_l3140_314010

/-- Represents the possible outcomes of the dice roll -/
inductive Outcome
| Star
| Moon
| Sun

/-- The probability of each outcome -/
def probability (o : Outcome) : ℚ :=
  match o with
  | Outcome.Star => 1/4
  | Outcome.Moon => 1/2
  | Outcome.Sun => 1/4

/-- The winnings (or losses) associated with each outcome -/
def winnings (o : Outcome) : ℤ :=
  match o with
  | Outcome.Star => 2
  | Outcome.Moon => 4
  | Outcome.Sun => -6

/-- The expected winnings from rolling the dice once -/
def expected_winnings : ℚ :=
  (probability Outcome.Star * winnings Outcome.Star) +
  (probability Outcome.Moon * winnings Outcome.Moon) +
  (probability Outcome.Sun * winnings Outcome.Sun)

theorem expected_winnings_is_one : expected_winnings = 1 := by
  sorry

end NUMINAMATH_CALUDE_expected_winnings_is_one_l3140_314010


namespace NUMINAMATH_CALUDE_min_value_of_quadratic_expression_l3140_314000

theorem min_value_of_quadratic_expression (x y : ℝ) :
  2 * x^2 + 3 * y^2 - 12 * x + 9 * y + 35 ≥ 41 / 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_quadratic_expression_l3140_314000


namespace NUMINAMATH_CALUDE_average_wage_calculation_l3140_314094

/-- Calculates the average wage per day for a contractor's workforce --/
theorem average_wage_calculation (male_workers female_workers child_workers : ℕ)
  (male_wage female_wage child_wage : ℚ)
  (h1 : male_workers = 20)
  (h2 : female_workers = 15)
  (h3 : child_workers = 5)
  (h4 : male_wage = 35)
  (h5 : female_wage = 20)
  (h6 : child_wage = 8) :
  let total_workers := male_workers + female_workers + child_workers
  let total_wage := male_workers * male_wage + female_workers * female_wage + child_workers * child_wage
  total_wage / total_workers = 26 := by
  sorry


end NUMINAMATH_CALUDE_average_wage_calculation_l3140_314094


namespace NUMINAMATH_CALUDE_ginas_account_fractions_l3140_314069

theorem ginas_account_fractions (betty_balance : ℝ) (gina_combined_balance : ℝ)
  (h1 : betty_balance = 3456)
  (h2 : gina_combined_balance = 1728) :
  ∃ (f1 f2 : ℝ), f1 + f2 = 1/2 ∧ f1 * betty_balance + f2 * betty_balance = gina_combined_balance :=
by sorry

end NUMINAMATH_CALUDE_ginas_account_fractions_l3140_314069

import Mathlib

namespace NUMINAMATH_CALUDE_smallest_positive_angle_theorem_l671_67169

theorem smallest_positive_angle_theorem (y : ℝ) : 
  (5 * Real.cos y * Real.sin y ^ 3 - 5 * Real.cos y ^ 3 * Real.sin y = 1 / 2) →
  y = (1 / 4) * Real.arcsin (2 / 5) ∧ y > 0 ∧ 
  ∀ z, z > 0 → (5 * Real.cos z * Real.sin z ^ 3 - 5 * Real.cos z ^ 3 * Real.sin z = 1 / 2) → y ≤ z :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_angle_theorem_l671_67169


namespace NUMINAMATH_CALUDE_mo_tea_consumption_l671_67129

/-- Represents the drinking habits of Mo --/
structure MoDrinkingHabits where
  n : ℕ  -- number of hot chocolate cups on rainy days
  t : ℕ  -- number of tea cups on non-rainy days
  total_cups : ℕ  -- total cups drunk in a week
  tea_chocolate_diff : ℕ  -- difference between tea and hot chocolate cups
  rainy_days : ℕ  -- number of rainy days in a week

/-- Theorem stating Mo's tea consumption on non-rainy days --/
theorem mo_tea_consumption (mo : MoDrinkingHabits) 
  (h1 : mo.total_cups = 36)
  (h2 : mo.tea_chocolate_diff = 14)
  (h3 : mo.rainy_days = 2)
  (h4 : mo.rainy_days * mo.n + (7 - mo.rainy_days) * mo.t = mo.total_cups)
  (h5 : (7 - mo.rainy_days) * mo.t = mo.rainy_days * mo.n + mo.tea_chocolate_diff) :
  mo.t = 5 := by
  sorry

#check mo_tea_consumption

end NUMINAMATH_CALUDE_mo_tea_consumption_l671_67129


namespace NUMINAMATH_CALUDE_expression_simplification_l671_67197

theorem expression_simplification (x : ℚ) (h : x = 3) : 
  (((x - 1) / (x + 2) + 1) / ((x - 1) / (x + 2) - 1)) = -7/3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l671_67197


namespace NUMINAMATH_CALUDE_complex_equation_solution_l671_67144

theorem complex_equation_solution (x : ℝ) (y : ℂ) 
  (h1 : y.re = 0)  -- y is purely imaginary
  (h2 : (3 * x + 1 : ℂ) - 2 * Complex.I = y) : 
  x = -1/3 ∧ y = -2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l671_67144


namespace NUMINAMATH_CALUDE_factor_expression_l671_67157

theorem factor_expression (x : ℝ) : 84 * x^7 - 306 * x^13 = 6 * x^7 * (14 - 51 * x^6) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l671_67157


namespace NUMINAMATH_CALUDE_communication_system_probabilities_l671_67105

/-- Represents a communication system with two signals A and B --/
structure CommunicationSystem where
  pTransmitA : ℝ  -- Probability of transmitting signal A
  pTransmitB : ℝ  -- Probability of transmitting signal B
  pDistortAtoB : ℝ  -- Probability of A being distorted to B
  pDistortBtoA : ℝ  -- Probability of B being distorted to A

/-- Theorem about probabilities in the communication system --/
theorem communication_system_probabilities (sys : CommunicationSystem) 
  (h1 : sys.pTransmitA = 0.72)
  (h2 : sys.pTransmitB = 0.28)
  (h3 : sys.pDistortAtoB = 1/6)
  (h4 : sys.pDistortBtoA = 1/7) :
  let pReceiveA := sys.pTransmitA * (1 - sys.pDistortAtoB) + sys.pTransmitB * sys.pDistortBtoA
  let pTransmittedAGivenReceivedA := (sys.pTransmitA * (1 - sys.pDistortAtoB)) / pReceiveA
  pReceiveA = 0.64 ∧ pTransmittedAGivenReceivedA = 0.9375 := by
  sorry


end NUMINAMATH_CALUDE_communication_system_probabilities_l671_67105


namespace NUMINAMATH_CALUDE_unique_dissection_solution_l671_67116

/-- Represents a square dissection into four-cell and five-cell figures -/
structure SquareDissection where
  size : ℕ
  four_cell_count : ℕ
  five_cell_count : ℕ

/-- Checks if a given dissection is valid for a square of size 6 -/
def is_valid_dissection (d : SquareDissection) : Prop :=
  d.size = 6 ∧ 
  d.four_cell_count > 0 ∧ 
  d.five_cell_count > 0 ∧
  d.size * d.size = 4 * d.four_cell_count + 5 * d.five_cell_count

/-- The unique solution to the square dissection problem -/
def unique_solution : SquareDissection :=
  { size := 6
    four_cell_count := 4
    five_cell_count := 4 }

/-- Theorem stating that the unique solution is the only valid dissection -/
theorem unique_dissection_solution :
  ∀ d : SquareDissection, is_valid_dissection d ↔ d = unique_solution :=
by sorry


end NUMINAMATH_CALUDE_unique_dissection_solution_l671_67116


namespace NUMINAMATH_CALUDE_range_of_a_l671_67175

theorem range_of_a (x : ℝ) (a : ℝ) : 
  (∀ x, (0 < x ∧ x < a) → (|x - 2| < 3)) ∧ 
  (∃ x, |x - 2| < 3 ∧ ¬(0 < x ∧ x < a)) ∧
  (a > 0) →
  (0 < a ∧ a ≤ 5) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l671_67175


namespace NUMINAMATH_CALUDE_length_FG_is_20_l671_67128

/-- Triangle PQR with points F and G -/
structure TrianglePQR where
  /-- Length of side PQ -/
  PQ : ℝ
  /-- Length of side PR -/
  PR : ℝ
  /-- Length of side QR -/
  QR : ℝ
  /-- Point F on PQ -/
  F : ℝ
  /-- Point G on PR -/
  G : ℝ
  /-- FG is parallel to QR -/
  FG_parallel_QR : Bool
  /-- G divides PR in ratio 2:1 -/
  G_divides_PR : G = (2/3) * PR

/-- The length of FG in the given triangle configuration -/
def length_FG (t : TrianglePQR) : ℝ := sorry

/-- Theorem stating that the length of FG is 20 under the given conditions -/
theorem length_FG_is_20 (t : TrianglePQR) 
  (h1 : t.PQ = 24) 
  (h2 : t.PR = 26) 
  (h3 : t.QR = 30) 
  (h4 : t.FG_parallel_QR = true) : 
  length_FG t = 20 := by sorry

end NUMINAMATH_CALUDE_length_FG_is_20_l671_67128


namespace NUMINAMATH_CALUDE_balcony_difference_l671_67146

/-- Represents the number of tickets sold for each section of the theater. -/
structure TheaterSales where
  orchestra : ℕ
  balcony : ℕ
  vip : ℕ

/-- Calculates the total revenue from ticket sales. -/
def totalRevenue (sales : TheaterSales) : ℕ :=
  15 * sales.orchestra + 10 * sales.balcony + 20 * sales.vip

/-- Calculates the total number of tickets sold. -/
def totalTickets (sales : TheaterSales) : ℕ :=
  sales.orchestra + sales.balcony + sales.vip

/-- Theorem stating the difference between balcony tickets and the sum of orchestra and VIP tickets. -/
theorem balcony_difference (sales : TheaterSales) 
    (h1 : totalTickets sales = 550)
    (h2 : totalRevenue sales = 8000) :
    sales.balcony - (sales.orchestra + sales.vip) = 370 := by
  sorry

end NUMINAMATH_CALUDE_balcony_difference_l671_67146


namespace NUMINAMATH_CALUDE_solution_sum_l671_67196

theorem solution_sum (x₁ y₁ x₂ y₂ : ℝ) : 
  (x₁ * y₁ - x₁ = 180 ∧ y₁ + x₁ * y₁ = 208) ∧
  (x₂ * y₂ - x₂ = 180 ∧ y₂ + x₂ * y₂ = 208) ∧
  (x₁ ≠ x₂) →
  x₁ + 10 * y₁ + x₂ + 10 * y₂ = 317 := by
sorry

end NUMINAMATH_CALUDE_solution_sum_l671_67196


namespace NUMINAMATH_CALUDE_jenny_investment_l671_67119

/-- Jenny's investment problem -/
theorem jenny_investment (total : ℝ) (real_estate : ℝ) (mutual_funds : ℝ) 
  (h1 : total = 200000)
  (h2 : real_estate = 3 * mutual_funds)
  (h3 : total = real_estate + mutual_funds) :
  real_estate = 150000 := by
  sorry

end NUMINAMATH_CALUDE_jenny_investment_l671_67119


namespace NUMINAMATH_CALUDE_sequence_length_is_602_l671_67142

/-- The number of terms in an arithmetic sequence -/
def arithmetic_sequence_length (a₁ aₙ d : ℕ) : ℕ :=
  (aₙ - a₁) / d + 1

/-- Theorem: The number of terms in the specified arithmetic sequence is 602 -/
theorem sequence_length_is_602 :
  arithmetic_sequence_length 3 3008 5 = 602 := by
  sorry

end NUMINAMATH_CALUDE_sequence_length_is_602_l671_67142


namespace NUMINAMATH_CALUDE_min_odd_in_A_P_l671_67132

/-- The set A_P for a polynomial P -/
def A_P (P : ℝ → ℝ) : Set ℝ := {x : ℝ | ∃ c : ℝ, P x = c}

/-- A polynomial is of degree 8 -/
def is_degree_8 (P : ℝ → ℝ) : Prop :=
  ∃ a₈ a₇ a₆ a₅ a₄ a₃ a₂ a₁ a₀ : ℝ, a₈ ≠ 0 ∧
    ∀ x, P x = a₈ * x^8 + a₇ * x^7 + a₆ * x^6 + a₅ * x^5 + 
           a₄ * x^4 + a₃ * x^3 + a₂ * x^2 + a₁ * x + a₀

theorem min_odd_in_A_P (P : ℝ → ℝ) (h : is_degree_8 P) (h8 : 8 ∈ A_P P) :
  ∃ x ∈ A_P P, Odd x :=
sorry

end NUMINAMATH_CALUDE_min_odd_in_A_P_l671_67132


namespace NUMINAMATH_CALUDE_percent_of_x_is_z_l671_67193

theorem percent_of_x_is_z (x y z : ℝ) 
  (h1 : 0.45 * z = 0.39 * y) 
  (h2 : y = 0.75 * x) : 
  z = 0.65 * x := by
  sorry

end NUMINAMATH_CALUDE_percent_of_x_is_z_l671_67193


namespace NUMINAMATH_CALUDE_fallen_pages_count_l671_67137

/-- Represents a page number as a triple of digits -/
structure PageNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  valid : hundreds ≥ 1 ∧ hundreds ≤ 9 ∧ tens ≤ 9 ∧ ones ≤ 9

/-- Converts a PageNumber to its numerical value -/
def PageNumber.toNat (p : PageNumber) : Nat :=
  p.hundreds * 100 + p.tens * 10 + p.ones

/-- Checks if a PageNumber is even -/
def PageNumber.isEven (p : PageNumber) : Prop :=
  p.toNat % 2 = 0

/-- Checks if a PageNumber is a permutation of another PageNumber -/
def PageNumber.isPermutationOf (p1 p2 : PageNumber) : Prop :=
  (p1.hundreds = p2.hundreds ∨ p1.hundreds = p2.tens ∨ p1.hundreds = p2.ones) ∧
  (p1.tens = p2.hundreds ∨ p1.tens = p2.tens ∨ p1.tens = p2.ones) ∧
  (p1.ones = p2.hundreds ∨ p1.ones = p2.tens ∨ p1.ones = p2.ones)

theorem fallen_pages_count 
  (first_page last_page : PageNumber)
  (h_first : first_page.toNat = 143)
  (h_perm : last_page.isPermutationOf first_page)
  (h_even : last_page.isEven)
  (h_greater : last_page.toNat > first_page.toNat) :
  last_page.toNat - first_page.toNat + 1 = 172 := by
  sorry

end NUMINAMATH_CALUDE_fallen_pages_count_l671_67137


namespace NUMINAMATH_CALUDE_haley_final_lives_l671_67111

/-- Calculate the final number of lives in a video game scenario -/
def final_lives (initial : ℕ) (lost : ℕ) (gained : ℕ) : ℕ :=
  initial - lost + gained

/-- Theorem stating that for the given scenario, the final number of lives is 46 -/
theorem haley_final_lives :
  final_lives 14 4 36 = 46 := by
  sorry

end NUMINAMATH_CALUDE_haley_final_lives_l671_67111


namespace NUMINAMATH_CALUDE_geometric_progression_ratio_l671_67171

/-- Given three terms of a geometric progression, prove that the common ratio is 52/25 -/
theorem geometric_progression_ratio (x : ℝ) (h_x : x ≠ 0) :
  let a₁ : ℝ := x / 2
  let a₂ : ℝ := 2 * x - 3
  let a₃ : ℝ := 18 / x + 1
  (a₁ * a₃ = a₂^2) → (a₂ / a₁ = 52 / 25) := by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_ratio_l671_67171


namespace NUMINAMATH_CALUDE_point_symmetry_l671_67101

/-- A point is symmetric to another point with respect to the origin if their coordinates sum to zero. -/
def symmetric_wrt_origin (p q : ℝ × ℝ) : Prop :=
  p.1 + q.1 = 0 ∧ p.2 + q.2 = 0

/-- The theorem states that the point (2, -3) is symmetric to the point (-2, 3) with respect to the origin. -/
theorem point_symmetry : symmetric_wrt_origin (-2, 3) (2, -3) := by
  sorry

end NUMINAMATH_CALUDE_point_symmetry_l671_67101


namespace NUMINAMATH_CALUDE_complex_number_coordinates_l671_67152

theorem complex_number_coordinates : 
  let z : ℂ := Complex.I * (2 - Complex.I)
  (z.re = 1 ∧ z.im = 2) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_coordinates_l671_67152


namespace NUMINAMATH_CALUDE_volume_of_rectangular_prism_l671_67170

/-- Represents a rectangular prism with dimensions a, d, and h -/
structure RectangularPrism where
  a : ℝ
  d : ℝ
  h : ℝ
  a_pos : 0 < a
  d_pos : 0 < d
  h_pos : 0 < h

/-- Calculates the volume of a rectangular prism -/
def volume (prism : RectangularPrism) : ℝ :=
  prism.a * prism.d * prism.h

/-- Theorem: The volume of a rectangular prism is equal to a * d * h -/
theorem volume_of_rectangular_prism (prism : RectangularPrism) :
  volume prism = prism.a * prism.d * prism.h :=
by sorry

end NUMINAMATH_CALUDE_volume_of_rectangular_prism_l671_67170


namespace NUMINAMATH_CALUDE_total_cost_mangoes_l671_67143

def prices : List Nat := [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
def boxes : Nat := 36

theorem total_cost_mangoes :
  (List.sum prices) * boxes = 3060 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_mangoes_l671_67143


namespace NUMINAMATH_CALUDE_rational_cube_sum_representation_l671_67150

theorem rational_cube_sum_representation (r : ℚ) (hr : 0 < r) :
  ∃ (a b c d : ℕ), 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧
    r = (a^3 + b^3 : ℚ) / (c^3 + d^3 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_rational_cube_sum_representation_l671_67150


namespace NUMINAMATH_CALUDE_sector_area_l671_67164

theorem sector_area (perimeter : ℝ) (central_angle : ℝ) (area : ℝ) : 
  perimeter = 8 → central_angle = 2 → area = 4 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l671_67164


namespace NUMINAMATH_CALUDE_grandmas_brownie_pan_l671_67100

/-- Represents a rectangular brownie pan with cuts -/
structure BrowniePan where
  m : ℕ+  -- length
  n : ℕ+  -- width
  length_cuts : ℕ
  width_cuts : ℕ

/-- Calculates the number of interior pieces -/
def interior_pieces (pan : BrowniePan) : ℕ :=
  (pan.m.val - pan.length_cuts - 1) * (pan.n.val - pan.width_cuts - 1)

/-- Calculates the number of perimeter pieces -/
def perimeter_pieces (pan : BrowniePan) : ℕ :=
  2 * (pan.m.val + pan.n.val) - 4

/-- The main theorem about Grandma's brownie pan -/
theorem grandmas_brownie_pan :
  ∃ (pan : BrowniePan),
    pan.length_cuts = 3 ∧
    pan.width_cuts = 5 ∧
    interior_pieces pan = 2 * perimeter_pieces pan ∧
    pan.m = 6 ∧
    pan.n = 12 := by
  sorry

end NUMINAMATH_CALUDE_grandmas_brownie_pan_l671_67100


namespace NUMINAMATH_CALUDE_min_value_of_expression_l671_67147

theorem min_value_of_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (x^2 / (x + 2) + y^2 / (y + 1)) ≥ 1/4 ∧ 
  ∃ x₀ y₀, x₀ > 0 ∧ y₀ > 0 ∧ x₀ + y₀ = 1 ∧ x₀^2 / (x₀ + 2) + y₀^2 / (y₀ + 1) = 1/4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l671_67147


namespace NUMINAMATH_CALUDE_log_equation_solution_l671_67123

theorem log_equation_solution (x : ℝ) (h : x > 0) :
  Real.log x / Real.log 8 + Real.log (x^3) / Real.log 2 = 9 → x = 2^(27/10) := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l671_67123


namespace NUMINAMATH_CALUDE_remainder_theorem_l671_67138

theorem remainder_theorem : 
  (2^300 + 405) % (2^150 + 2^75 + 1) = 404 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l671_67138


namespace NUMINAMATH_CALUDE_cubic_meter_to_cubic_cm_l671_67151

-- Define the conversion factor from meters to centimeters
def meters_to_cm : ℝ := 100

-- Theorem statement
theorem cubic_meter_to_cubic_cm :
  (1 : ℝ) * meters_to_cm^3 = 1000000 := by
  sorry

end NUMINAMATH_CALUDE_cubic_meter_to_cubic_cm_l671_67151


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l671_67108

universe u

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def A : Set Nat := {1, 2, 5}
def B : Set Nat := {1, 3, 4}

theorem complement_A_intersect_B :
  (Set.compl A ∩ B) = {3, 4} :=
sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l671_67108


namespace NUMINAMATH_CALUDE_hexagon_area_in_circle_l671_67136

/-- The area of a regular hexagon inscribed in a circle with area 196π square units is 294√3 square units. -/
theorem hexagon_area_in_circle (circle_area : ℝ) (hexagon_area : ℝ) : 
  circle_area = 196 * Real.pi → hexagon_area = 294 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_area_in_circle_l671_67136


namespace NUMINAMATH_CALUDE_camping_hike_distance_l671_67190

/-- The total distance hiked by Irwin's family during their camping trip -/
theorem camping_hike_distance 
  (car_to_stream : ℝ) 
  (stream_to_meadow : ℝ) 
  (meadow_to_campsite : ℝ)
  (h1 : car_to_stream = 0.2)
  (h2 : stream_to_meadow = 0.4)
  (h3 : meadow_to_campsite = 0.1) :
  car_to_stream + stream_to_meadow + meadow_to_campsite = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_camping_hike_distance_l671_67190


namespace NUMINAMATH_CALUDE_equation_solution_l671_67180

theorem equation_solution : ∃ x : ℝ, 5 * (x - 4) = 2 * (3 - 2 * x) + 10 ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l671_67180


namespace NUMINAMATH_CALUDE_scientific_notation_of_five_nm_l671_67187

theorem scientific_notation_of_five_nm :
  ∃ (a : ℝ) (n : ℤ), 0.000000005 = a * 10^n ∧ 1 ≤ a ∧ a < 10 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_five_nm_l671_67187


namespace NUMINAMATH_CALUDE_x_eq_2_sufficient_not_necessary_l671_67181

def a (x : ℝ) : Fin 2 → ℝ := ![1, x - 1]
def b (x : ℝ) : Fin 2 → ℝ := ![x + 1, 3]

def parallel (v w : Fin 2 → ℝ) : Prop :=
  ∃ k : ℝ, ∀ i : Fin 2, v i = k * w i

theorem x_eq_2_sufficient_not_necessary :
  (∃ x : ℝ, x ≠ 2 ∧ parallel (a x) (b x)) ∧
  (∀ x : ℝ, x = 2 → parallel (a x) (b x)) :=
sorry

end NUMINAMATH_CALUDE_x_eq_2_sufficient_not_necessary_l671_67181


namespace NUMINAMATH_CALUDE_function_properties_l671_67160

-- Define the function f(x) = x ln x
noncomputable def f (x : ℝ) : ℝ := x * Real.log x

-- Statement of the theorem
theorem function_properties :
  -- 1. The tangent line to y = f(x) at x = 1 is y = x - 1
  (∀ x, (f x - f 1) = (x - 1) * (Real.log 1 + 1)) ∧
  -- 2. There are exactly 2 lines tangent to y = f(x) passing through (1, -1)
  (∃! a b : ℝ, a ≠ b ∧ 
    (∀ x, f x = (Real.log a + 1) * (x - a) + f a) ∧
    (Real.log a + 1) * (1 - a) + f a = -1 ∧
    (∀ x, f x = (Real.log b + 1) * (x - b) + f b) ∧
    (Real.log b + 1) * (1 - b) + f b = -1) ∧
  -- 3. f(x) has a local minimum and no local maximum
  (∃ c : ℝ, ∀ x, x > 0 → x ≠ c → f x > f c) ∧
  (¬ ∃ d : ℝ, ∀ x, x > 0 → x ≠ d → f x < f d) ∧
  -- 4. The equation f(x) = 1 does not have two distinct solutions
  ¬ (∃ x y : ℝ, x ≠ y ∧ f x = 1 ∧ f y = 1) :=
by sorry


end NUMINAMATH_CALUDE_function_properties_l671_67160


namespace NUMINAMATH_CALUDE_marks_speeding_ticket_cost_l671_67113

/-- Calculates the total amount owed for a speeding ticket -/
def speeding_ticket_cost (base_fine speed_limit actual_speed additional_penalty_per_mph : ℕ)
  (school_zone : Bool) (court_costs lawyer_fee_per_hour lawyer_hours : ℕ) : ℕ :=
  let speed_difference := actual_speed - speed_limit
  let additional_penalty := speed_difference * additional_penalty_per_mph
  let total_fine := base_fine + additional_penalty
  let doubled_fine := if school_zone then 2 * total_fine else total_fine
  let fine_with_court_costs := doubled_fine + court_costs
  let lawyer_fees := lawyer_fee_per_hour * lawyer_hours
  fine_with_court_costs + lawyer_fees

/-- Theorem: Mark's speeding ticket cost is $820 -/
theorem marks_speeding_ticket_cost :
  speeding_ticket_cost 50 30 75 2 true 300 80 3 = 820 := by
  sorry

end NUMINAMATH_CALUDE_marks_speeding_ticket_cost_l671_67113


namespace NUMINAMATH_CALUDE_tara_book_sales_l671_67104

/-- Calculates the total number of books Tara needs to sell to buy a new clarinet and an accessory, given initial savings, clarinet cost, book price, and additional accessory cost. -/
def total_books_sold (initial_savings : ℕ) (clarinet_cost : ℕ) (book_price : ℕ) (accessory_cost : ℕ) : ℕ :=
  let initial_goal := clarinet_cost - initial_savings
  let halfway_books := (initial_goal / 2) / book_price
  let final_goal := initial_goal + accessory_cost
  let final_books := final_goal / book_price
  halfway_books + final_books

/-- Theorem stating that Tara needs to sell 28 books in total to reach her goal. -/
theorem tara_book_sales : total_books_sold 10 90 5 20 = 28 := by
  sorry

end NUMINAMATH_CALUDE_tara_book_sales_l671_67104


namespace NUMINAMATH_CALUDE_inequalities_with_squares_and_roots_l671_67109

theorem inequalities_with_squares_and_roots (a b : ℝ) : 
  (a > 0 ∧ b > 0 ∧ a^2 - b^2 = 1 → a - b ≤ 1) ∧
  (a > 0 ∧ b > 0 ∧ Real.sqrt a - Real.sqrt b = 1 → a - b ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_with_squares_and_roots_l671_67109


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l671_67194

theorem quadratic_inequality_range (a : ℝ) : 
  (a ≠ 0 ∧ ∀ x : ℝ, a * x^2 + 2 * a * x - 4 < 0) ↔ (-4 < a ∧ a < 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l671_67194


namespace NUMINAMATH_CALUDE_fraction_equality_l671_67172

theorem fraction_equality (a b y : ℝ) 
  (h1 : y = (a + 2*b) / a) 
  (h2 : a ≠ -2*b) 
  (h3 : a ≠ 0) : 
  (2*a + 2*b) / (a - 2*b) = (y + 1) / (3 - y) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l671_67172


namespace NUMINAMATH_CALUDE_sphere_impulse_theorem_l671_67195

/-- Represents a uniform sphere -/
structure UniformSphere where
  mass : ℝ
  radius : ℝ

/-- Represents the initial conditions and applied impulse -/
structure ImpulseConditions where
  sphere : UniformSphere
  impulse : ℝ
  beta : ℝ

/-- Theorem stating the final speed and condition for rolling without slipping -/
theorem sphere_impulse_theorem (conditions : ImpulseConditions) 
  (h1 : conditions.beta ≥ -1) 
  (h2 : conditions.beta ≤ 1) : 
  ∃ (v : ℝ), 
    v = (5 * conditions.impulse * conditions.beta) / (7 * conditions.sphere.mass) ∧
    (conditions.beta = 7/5 → 
      v * conditions.sphere.mass = conditions.impulse ∧ 
      v = conditions.sphere.radius * ((5 * conditions.impulse * conditions.beta) / 
        (7 * conditions.sphere.mass * conditions.sphere.radius))) := by
  sorry

end NUMINAMATH_CALUDE_sphere_impulse_theorem_l671_67195


namespace NUMINAMATH_CALUDE_sum_of_twenty_terms_l671_67161

/-- Given a sequence of non-zero terms {aₙ}, where Sₙ is the sum of the first n terms,
    prove that S₂₀ = 210 under the given conditions. -/
theorem sum_of_twenty_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n, a n ≠ 0) →
  (∀ n, S n = (a n * a (n + 1)) / 2) →
  a 1 = 1 →
  S 20 = 210 := by
sorry

end NUMINAMATH_CALUDE_sum_of_twenty_terms_l671_67161


namespace NUMINAMATH_CALUDE_exists_same_color_distance_exists_color_for_all_distances_l671_67145

-- Define a type for colors
inductive Color
| Red
| Blue

-- Define a point in a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a function that assigns a color to each point
def colorAssignment : Point → Color := sorry

-- Define a function to calculate distance between two points
def distance (p1 p2 : Point) : ℝ := sorry

-- Statement for part (i)
theorem exists_same_color_distance (x : ℝ) :
  ∃ (c : Color) (p1 p2 : Point),
    colorAssignment p1 = c ∧
    colorAssignment p2 = c ∧
    distance p1 p2 = x :=
  sorry

-- Statement for part (ii)
theorem exists_color_for_all_distances :
  ∃ (c : Color), ∀ (x : ℝ),
    ∃ (p1 p2 : Point),
      colorAssignment p1 = c ∧
      colorAssignment p2 = c ∧
      distance p1 p2 = x :=
  sorry

end NUMINAMATH_CALUDE_exists_same_color_distance_exists_color_for_all_distances_l671_67145


namespace NUMINAMATH_CALUDE_a_100_eq_344934_l671_67185

/-- Sequence defined by a(n) = a(n-1) + n^2 for n ≥ 1, with a(0) = 2009 -/
def a : ℕ → ℕ
  | 0 => 2009
  | n + 1 => a n + (n + 1)^2

/-- The 100th term of the sequence a is 344934 -/
theorem a_100_eq_344934 : a 100 = 344934 := by
  sorry

end NUMINAMATH_CALUDE_a_100_eq_344934_l671_67185


namespace NUMINAMATH_CALUDE_average_age_union_l671_67199

-- Define the student groups and their properties
def StudentGroup := Type
variables (A B C : StudentGroup)

-- Define the number of students in each group
variables (a b c : ℕ)

-- Define the sum of ages in each group
variables (sumA sumB sumC : ℕ)

-- Define the average age function
def avgAge (sum : ℕ) (count : ℕ) : ℚ := sum / count

-- State the theorem
theorem average_age_union (h_disjoint : A ≠ B ∧ B ≠ C ∧ A ≠ C)
  (h_avgA : avgAge sumA a = 34)
  (h_avgB : avgAge sumB b = 25)
  (h_avgC : avgAge sumC c = 45)
  (h_avgAB : avgAge (sumA + sumB) (a + b) = 30)
  (h_avgAC : avgAge (sumA + sumC) (a + c) = 42)
  (h_avgBC : avgAge (sumB + sumC) (b + c) = 36) :
  avgAge (sumA + sumB + sumC) (a + b + c) = 33 := by
  sorry


end NUMINAMATH_CALUDE_average_age_union_l671_67199


namespace NUMINAMATH_CALUDE_ellipse_properties_l671_67102

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the foci
def F1 : ℝ × ℝ := sorry
def F2 : ℝ × ℝ := sorry

-- Define a point on the ellipse
def P : ℝ × ℝ := sorry
axiom P_on_ellipse : ellipse P.1 P.2

-- Define the eccentricity
def eccentricity : ℝ := sorry

-- Define the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define the angle between three points
def angle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem ellipse_properties :
  eccentricity = 1/2 ∧
  ∃ (Q : ℝ × ℝ), ellipse Q.1 Q.2 ∧ distance Q F1 = 3 ∧ ∀ (R : ℝ × ℝ), ellipse R.1 R.2 → distance R F1 ≤ 3 ∧
  0 ≤ angle F1 P F2 ∧ angle F1 P F2 ≤ π/3 :=
sorry

end NUMINAMATH_CALUDE_ellipse_properties_l671_67102


namespace NUMINAMATH_CALUDE_x_value_l671_67103

theorem x_value : ∃ x : ℝ, (0.25 * x = 0.12 * 1500 - 15) ∧ (x = 660) := by
  sorry

end NUMINAMATH_CALUDE_x_value_l671_67103


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l671_67165

theorem geometric_sequence_ratio (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) = -1/2 * a n) →
  (a 1 + a 3 + a 5) / (a 2 + a 4 + a 6) = -2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l671_67165


namespace NUMINAMATH_CALUDE_problem_statements_l671_67176

theorem problem_statements :
  (({0} : Set ℕ) ⊆ Set.univ) ∧
  (∀ (α : Type) (A B : Set α) (x : α), x ∈ A ∩ B → x ∈ A ∪ B) ∧
  (∃ (a b : ℝ), b^2 < a^2 ∧ ¬(a < b ∧ b < 0)) ∧
  (¬(∀ (x : ℤ), x^2 > 0) ↔ ∃ (x : ℤ), x^2 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_problem_statements_l671_67176


namespace NUMINAMATH_CALUDE_seating_arrangements_l671_67134

/-- The number of ways to arrange n distinct objects. -/
def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

/-- The number of people to be seated. -/
def totalPeople : ℕ := 10

/-- The number of people who cannot sit in three consecutive seats. -/
def cannotSitTogether : ℕ := 3

/-- The number of people who must sit together. -/
def mustSitTogether : ℕ := 2

/-- The number of seating arrangements where Alice, Bob, and Cindy sit together. -/
def arrangementsTogether : ℕ := factorial (totalPeople - cannotSitTogether + 1) * factorial cannotSitTogether

/-- The number of seating arrangements where Dave and Emma sit together. -/
def arrangementsPairTogether : ℕ := factorial (totalPeople - mustSitTogether + 1) * factorial mustSitTogether

/-- The number of seating arrangements where both conditions are met simultaneously. -/
def arrangementsOverlap : ℕ := factorial (totalPeople - cannotSitTogether - mustSitTogether + 2) * factorial cannotSitTogether * factorial mustSitTogether

/-- The total number of valid seating arrangements. -/
def validArrangements : ℕ := factorial totalPeople - (arrangementsTogether + arrangementsPairTogether - arrangementsOverlap)

theorem seating_arrangements : validArrangements = 3144960 := by sorry

end NUMINAMATH_CALUDE_seating_arrangements_l671_67134


namespace NUMINAMATH_CALUDE_concentric_circles_radii_difference_l671_67127

theorem concentric_circles_radii_difference 
  (r R : ℝ) 
  (h_positive : r > 0) 
  (h_ratio : π * R^2 = 3 * π * r^2) : 
  ∃ ε > 0, |R - r - 0.73 * r| < ε := by
sorry

end NUMINAMATH_CALUDE_concentric_circles_radii_difference_l671_67127


namespace NUMINAMATH_CALUDE_grace_pool_volume_l671_67168

/-- The volume of water in Grace's pool -/
def pool_volume (first_hose_rate : ℝ) (first_hose_time : ℝ) (second_hose_rate : ℝ) (second_hose_time : ℝ) : ℝ :=
  first_hose_rate * first_hose_time + second_hose_rate * second_hose_time

/-- Theorem stating that Grace's pool contains 390 gallons of water -/
theorem grace_pool_volume :
  let first_hose_rate : ℝ := 50
  let first_hose_time : ℝ := 5
  let second_hose_rate : ℝ := 70
  let second_hose_time : ℝ := 2
  pool_volume first_hose_rate first_hose_time second_hose_rate second_hose_time = 390 :=
by
  sorry


end NUMINAMATH_CALUDE_grace_pool_volume_l671_67168


namespace NUMINAMATH_CALUDE_unique_x_with_three_prime_factors_l671_67130

theorem unique_x_with_three_prime_factors (x n : ℕ) : 
  x = 7^n + 1 →
  Odd n →
  (∃ p q : ℕ, Prime p ∧ Prime q ∧ p ≠ q ∧ p ≠ 11 ∧ q ≠ 11 ∧ x = 2 * 11 * p * q) →
  x = 16808 :=
by sorry

end NUMINAMATH_CALUDE_unique_x_with_three_prime_factors_l671_67130


namespace NUMINAMATH_CALUDE_total_pizza_pieces_l671_67162

/-- Given 10 children, each buying 20 pizzas, and each pizza containing 6 pieces,
    the total number of pizza pieces is 1200. -/
theorem total_pizza_pieces :
  let num_children : ℕ := 10
  let pizzas_per_child : ℕ := 20
  let pieces_per_pizza : ℕ := 6
  num_children * pizzas_per_child * pieces_per_pizza = 1200 :=
by
  sorry


end NUMINAMATH_CALUDE_total_pizza_pieces_l671_67162


namespace NUMINAMATH_CALUDE_no_three_digit_number_eight_times_smaller_l671_67184

theorem no_three_digit_number_eight_times_smaller : ¬ ∃ (a b c : ℕ), 
  (1 ≤ a ∧ a ≤ 9) ∧ 
  (b ≤ 9) ∧ 
  (c ≤ 9) ∧ 
  (100 * a + 10 * b + c = 8 * (10 * b + c)) := by
sorry

end NUMINAMATH_CALUDE_no_three_digit_number_eight_times_smaller_l671_67184


namespace NUMINAMATH_CALUDE_nails_per_paw_is_four_l671_67174

/-- The number of nails on one paw of a dog -/
def nails_per_paw : ℕ := sorry

/-- The total number of trimmed nails -/
def total_nails : ℕ := 164

/-- The number of dogs with three legs -/
def three_legged_dogs : ℕ := 3

/-- Theorem stating that the number of nails on one paw of a dog is 4 -/
theorem nails_per_paw_is_four : nails_per_paw = 4 := by sorry

end NUMINAMATH_CALUDE_nails_per_paw_is_four_l671_67174


namespace NUMINAMATH_CALUDE_least_years_to_double_l671_67107

theorem least_years_to_double (rate : ℝ) (h : rate = 0.5) : 
  (∃ t : ℕ, (1 + rate)^t > 2) ∧ 
  (∀ t : ℕ, (1 + rate)^t > 2 → t ≥ 2) :=
by
  sorry

#check least_years_to_double

end NUMINAMATH_CALUDE_least_years_to_double_l671_67107


namespace NUMINAMATH_CALUDE_perpendicular_and_minimum_points_l671_67124

-- Define the vectors
def OA : Fin 2 → ℝ := ![1, 7]
def OB : Fin 2 → ℝ := ![5, 1]
def OP : Fin 2 → ℝ := ![2, 1]

-- Define the dot product
def dot_product (v w : Fin 2 → ℝ) : ℝ := v 0 * w 0 + v 1 * w 1

-- Define the function for OQ based on parameter t
def OQ (t : ℝ) : Fin 2 → ℝ := ![2*t, t]

-- Define QA as a function of t
def QA (t : ℝ) : Fin 2 → ℝ := ![1 - 2*t, 7 - t]

-- Define QB as a function of t
def QB (t : ℝ) : Fin 2 → ℝ := ![5 - 2*t, 1 - t]

theorem perpendicular_and_minimum_points :
  (∃ t : ℝ, dot_product (QA t) OP = 0 ∧ OQ t = ![18/5, 9/5]) ∧
  (∃ t : ℝ, ∀ s : ℝ, dot_product OA (QB t) ≤ dot_product OA (QB s) ∧ OQ t = ![4, 2]) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_and_minimum_points_l671_67124


namespace NUMINAMATH_CALUDE_complex_number_validity_one_plus_i_is_valid_l671_67115

theorem complex_number_validity : Complex → Prop :=
  fun z => ∃ (a b : ℝ), z = Complex.mk a b

theorem one_plus_i_is_valid : complex_number_validity (1 + Complex.I) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_validity_one_plus_i_is_valid_l671_67115


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l671_67159

/-- Given a hyperbola with equation x²/4 - y²/m² = 1 where m > 0,
    and one of its asymptotes is 5x - 2y = 0, prove that m = 5. -/
theorem hyperbola_asymptote (m : ℝ) (h1 : m > 0) : 
  (∃ x y : ℝ, x^2/4 - y^2/m^2 = 1 ∧ 5*x - 2*y = 0) → m = 5 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l671_67159


namespace NUMINAMATH_CALUDE_joeys_route_length_l671_67173

/-- Given a round trip with total time 1 hour and average speed 3 miles/hour,
    prove that the one-way distance is 1.5 miles. -/
theorem joeys_route_length (total_time : ℝ) (avg_speed : ℝ) (one_way_distance : ℝ) :
  total_time = 1 →
  avg_speed = 3 →
  one_way_distance = avg_speed * total_time / 2 →
  one_way_distance = 1.5 := by
  sorry

#check joeys_route_length

end NUMINAMATH_CALUDE_joeys_route_length_l671_67173


namespace NUMINAMATH_CALUDE_fraction_sum_simplification_l671_67118

theorem fraction_sum_simplification : (1 : ℚ) / 210 + 17 / 30 = 4 / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_simplification_l671_67118


namespace NUMINAMATH_CALUDE_solve_equation_l671_67155

theorem solve_equation : (45 : ℚ) / (9 - 3/7) = 21/4 := by sorry

end NUMINAMATH_CALUDE_solve_equation_l671_67155


namespace NUMINAMATH_CALUDE_square_side_length_l671_67117

theorem square_side_length (area : ℝ) (side : ℝ) : 
  area = 144 → side ^ 2 = area → side = 12 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l671_67117


namespace NUMINAMATH_CALUDE_simplify_fraction_l671_67166

theorem simplify_fraction (a : ℝ) (h1 : a ≠ 0) (h2 : a ≠ 1) :
  (a - 1 / a) / ((a - 1) / a) = a + 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l671_67166


namespace NUMINAMATH_CALUDE_fraction_equals_zero_l671_67192

theorem fraction_equals_zero (x : ℝ) : (x^2 - 4) / (x + 2) = 0 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equals_zero_l671_67192


namespace NUMINAMATH_CALUDE_perpendicular_planes_not_necessarily_parallel_l671_67110

/-- Represents a plane in 3D space -/
structure Plane3D where
  -- Add necessary fields for a plane

/-- Represents a line in 3D space -/
structure Line3D where
  -- Add necessary fields for a line

/-- Two planes are perpendicular -/
def perpendicular (p1 p2 : Plane3D) : Prop := sorry

/-- Two planes are parallel -/
def parallel (p1 p2 : Plane3D) : Prop := sorry

/-- The statement that if two planes are perpendicular to a third plane, they are parallel to each other is false -/
theorem perpendicular_planes_not_necessarily_parallel (α β γ : Plane3D) :
  ¬(∀ α β γ : Plane3D, perpendicular α β → perpendicular β γ → parallel α γ) := by
  sorry

#check perpendicular_planes_not_necessarily_parallel

end NUMINAMATH_CALUDE_perpendicular_planes_not_necessarily_parallel_l671_67110


namespace NUMINAMATH_CALUDE_matrix_power_4_l671_67179

def A : Matrix (Fin 2) (Fin 2) ℝ := !![2, -1; 1, 1]

theorem matrix_power_4 : A^4 = !![0, -9; 9, -9] := by sorry

end NUMINAMATH_CALUDE_matrix_power_4_l671_67179


namespace NUMINAMATH_CALUDE_age_ratio_in_ten_years_l671_67167

/-- Represents the age difference between Pete and Claire -/
structure AgeDifference where
  pete : ℕ
  claire : ℕ

/-- The conditions of the problem -/
def age_conditions (ad : AgeDifference) : Prop :=
  ∃ (x : ℕ),
    -- Claire's age 2 years ago
    ad.claire = x + 2 ∧
    -- Pete's age 2 years ago
    ad.pete = 3 * x + 2 ∧
    -- Four years ago condition
    3 * x - 4 = 4 * (x - 4)

/-- The theorem to be proved -/
theorem age_ratio_in_ten_years (ad : AgeDifference) :
  age_conditions ad →
  (ad.pete + 10) / (ad.claire + 10) = 2 := by
sorry

end NUMINAMATH_CALUDE_age_ratio_in_ten_years_l671_67167


namespace NUMINAMATH_CALUDE_divisibility_property_l671_67189

theorem divisibility_property (n : ℕ) : 
  n > 0 ∧ n^2 ∣ 2^n + 1 ↔ n = 1 ∨ n = 3 := by sorry

end NUMINAMATH_CALUDE_divisibility_property_l671_67189


namespace NUMINAMATH_CALUDE_congruence_solution_l671_67141

theorem congruence_solution : ∃ n : ℤ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ -2187 [ZMOD 10] ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_congruence_solution_l671_67141


namespace NUMINAMATH_CALUDE_quadratic_intersection_properties_l671_67131

/-- A quadratic function f(x) = x^2 + 2x + b intersecting both coordinate axes at three points -/
structure QuadraticIntersection (b : ℝ) :=
  (intersects_axes : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁^2 + 2*x₁ + b = 0 ∧ x₂^2 + 2*x₂ + b = 0)
  (y_intercept : b ≠ 0)

/-- The circle passing through the three intersection points -/
def intersection_circle (b : ℝ) (h : QuadraticIntersection b) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 + 2*p.1 - (b + 1)*p.2 + b = 0}

/-- Main theorem: properties of the quadratic function and its intersection circle -/
theorem quadratic_intersection_properties (b : ℝ) (h : QuadraticIntersection b) :
  b < 1 ∧ 
  ∀ (p : ℝ × ℝ), p ∈ intersection_circle b h ↔ p.1^2 + p.2^2 + 2*p.1 - (b + 1)*p.2 + b = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_intersection_properties_l671_67131


namespace NUMINAMATH_CALUDE_exists_m_for_even_f_l671_67149

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

/-- The function f(x) = x^2 + mx for some m ∈ ℝ -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m*x

/-- There exists an m ∈ ℝ such that f(x) = x^2 + mx is an even function -/
theorem exists_m_for_even_f : ∃ m : ℝ, IsEven (f m) := by
  sorry

end NUMINAMATH_CALUDE_exists_m_for_even_f_l671_67149


namespace NUMINAMATH_CALUDE_greatest_integer_x_l671_67188

def is_integer (x : ℚ) : Prop := ∃ n : ℤ, x = n

def f (x : ℤ) : ℚ := (x^2 + 4*x + 13) / (x - 4)

theorem greatest_integer_x : 
  (∀ x : ℤ, x > 49 → ¬ is_integer (f x)) ∧ 
  is_integer (f 49) := by
sorry

end NUMINAMATH_CALUDE_greatest_integer_x_l671_67188


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l671_67139

theorem complex_fraction_simplification :
  let z₁ : ℂ := 2 + 4 * I
  let z₂ : ℂ := 2 - 4 * I
  z₁ / z₂ - z₂ / z₁ = -8/5 + 16/5 * I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l671_67139


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l671_67163

theorem completing_square_equivalence :
  ∀ x : ℝ, 4 * x^2 - 2 * x - 1 = 0 ↔ (x - 1/4)^2 = 5/16 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l671_67163


namespace NUMINAMATH_CALUDE_factorial_inequality_l671_67186

theorem factorial_inequality (n : ℕ) (h : n ≥ 1) : n.factorial ≤ ((n + 1) / 2 : ℝ) ^ n := by
  sorry

end NUMINAMATH_CALUDE_factorial_inequality_l671_67186


namespace NUMINAMATH_CALUDE_yellow_red_block_difference_l671_67156

/-- Given a toy bin with red, yellow, and blue blocks, prove the difference between yellow and red blocks -/
theorem yellow_red_block_difference 
  (red : ℕ) 
  (yellow : ℕ) 
  (blue : ℕ) 
  (h1 : red = 18) 
  (h2 : yellow > red) 
  (h3 : blue = red + 14) 
  (h4 : red + yellow + blue = 75) : 
  yellow - red = 7 := by
  sorry

end NUMINAMATH_CALUDE_yellow_red_block_difference_l671_67156


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l671_67120

theorem quadratic_no_real_roots (m : ℝ) : 
  (∀ x : ℝ, x^2 - 2*x + m ≠ 0) ↔ m > 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l671_67120


namespace NUMINAMATH_CALUDE_system_solution_l671_67198

theorem system_solution :
  ∃! (x y : ℚ), 3 * x - 2 * y = 5 ∧ 4 * x + 5 * y = 16 ∧ x = 57 / 23 ∧ y = 28 / 23 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l671_67198


namespace NUMINAMATH_CALUDE_solution_correctness_l671_67148

def solution_set : Set (ℝ × ℝ × ℝ) :=
  {(0, 1, 1), (0, -1, -1), (1, 1, 0), (-1, -1, 0), (1, 0, 1), (-1, 0, -1),
   (Real.sqrt 3 / 3, Real.sqrt 3 / 3, Real.sqrt 3 / 3),
   (-Real.sqrt 3 / 3, -Real.sqrt 3 / 3, -Real.sqrt 3 / 3)}

def satisfies_conditions (a b c : ℝ) : Prop :=
  a^2*b + c = b^2*c + a ∧ 
  b^2*c + a = c^2*a + b ∧
  a*b + b*c + c*a = 1

theorem solution_correctness :
  ∀ (a b c : ℝ), satisfies_conditions a b c ↔ (a, b, c) ∈ solution_set :=
by sorry

end NUMINAMATH_CALUDE_solution_correctness_l671_67148


namespace NUMINAMATH_CALUDE_missing_fraction_problem_l671_67140

theorem missing_fraction_problem (sum : ℚ) (f1 f2 f3 f4 f5 f6 f7 : ℚ) : 
  sum = 45/100 →
  f1 = 1/3 →
  f2 = 1/2 →
  f3 = -5/6 →
  f4 = 1/4 →
  f5 = -9/20 →
  f6 = -9/20 →
  f1 + f2 + f3 + f4 + f5 + f6 + f7 = sum →
  f7 = 11/10 := by
sorry

end NUMINAMATH_CALUDE_missing_fraction_problem_l671_67140


namespace NUMINAMATH_CALUDE_youtube_video_dislikes_l671_67114

theorem youtube_video_dislikes (initial_likes : ℕ) (initial_dislikes : ℕ) (additional_dislikes : ℕ) : 
  initial_likes = 3000 →
  initial_dislikes = initial_likes / 2 + 100 →
  additional_dislikes = 1000 →
  initial_dislikes + additional_dislikes = 2600 :=
by
  sorry

end NUMINAMATH_CALUDE_youtube_video_dislikes_l671_67114


namespace NUMINAMATH_CALUDE_equation_solution_l671_67133

theorem equation_solution : ∃ x : ℝ, 64 + x * 12 / (180 / 3) = 65 ∧ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l671_67133


namespace NUMINAMATH_CALUDE_largest_n_divisible_by_seven_largest_n_is_99999_l671_67153

theorem largest_n_divisible_by_seven (n : ℕ) : n < 100000 →
  (10 * (n - 3)^5 - n^2 + 20 * n - 30) % 7 = 0 →
  n ≤ 99999 :=
by sorry

theorem largest_n_is_99999 :
  (10 * (99999 - 3)^5 - 99999^2 + 20 * 99999 - 30) % 7 = 0 ∧
  ∀ m : ℕ, m > 99999 → m < 100000 →
    (10 * (m - 3)^5 - m^2 + 20 * m - 30) % 7 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_divisible_by_seven_largest_n_is_99999_l671_67153


namespace NUMINAMATH_CALUDE_polynomial_root_problem_l671_67182

/-- The polynomial h(x) -/
def h (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2 + 2*x + 15

/-- The polynomial f(x) -/
def f (b c : ℝ) (x : ℝ) : ℝ := x^4 - x^3 + b*x^2 + 120*x + c

/-- The theorem statement -/
theorem polynomial_root_problem (a b c : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    h a x = 0 ∧ h a y = 0 ∧ h a z = 0 ∧
    f b c x = 0 ∧ f b c y = 0 ∧ f b c z = 0) →
  f b c (-1) = -1995.25 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_root_problem_l671_67182


namespace NUMINAMATH_CALUDE_relationship_between_a_and_b_l671_67126

theorem relationship_between_a_and_b (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h : Real.exp a + 2 * a = Real.exp b + 3 * b) : a > b := by
  sorry

end NUMINAMATH_CALUDE_relationship_between_a_and_b_l671_67126


namespace NUMINAMATH_CALUDE_first_group_size_correct_l671_67125

/-- The number of persons in the first group that can repair a road -/
def first_group_size : ℕ := 33

/-- The number of days the first group works -/
def first_group_days : ℕ := 12

/-- The number of hours per day the first group works -/
def first_group_hours : ℕ := 5

/-- The number of persons in the second group -/
def second_group_size : ℕ := 30

/-- The number of days the second group works -/
def second_group_days : ℕ := 11

/-- The number of hours per day the second group works -/
def second_group_hours : ℕ := 6

/-- Theorem stating that the first group size is correct given the conditions -/
theorem first_group_size_correct :
  first_group_size * first_group_hours * first_group_days =
  second_group_size * second_group_hours * second_group_days :=
by sorry

end NUMINAMATH_CALUDE_first_group_size_correct_l671_67125


namespace NUMINAMATH_CALUDE_sum_not_exceeding_eight_probability_most_probable_sum_probability_of_most_probable_sum_l671_67177

def ball_count : ℕ := 8

def ball_labels : Finset ℕ := Finset.range ball_count

def sum_of_pair (i j : ℕ) : ℕ := i + j

def valid_pairs : Finset (ℕ × ℕ) :=
  (ball_labels.product ball_labels).filter (λ p => p.1 < p.2)

def pairs_with_sum (n : ℕ) : Finset (ℕ × ℕ) :=
  valid_pairs.filter (λ p => sum_of_pair p.1 p.2 = n)

def probability (favorable : Finset (ℕ × ℕ)) : ℚ :=
  favorable.card / valid_pairs.card

theorem sum_not_exceeding_eight_probability :
  probability (valid_pairs.filter (λ p => sum_of_pair p.1 p.2 ≤ 8)) = 3/7 := by sorry

theorem most_probable_sum :
  ∃ n : ℕ, n = 9 ∧ 
    ∀ m : ℕ, probability (pairs_with_sum n) ≥ probability (pairs_with_sum m) := by sorry

theorem probability_of_most_probable_sum :
  probability (pairs_with_sum 9) = 1/7 := by sorry

end NUMINAMATH_CALUDE_sum_not_exceeding_eight_probability_most_probable_sum_probability_of_most_probable_sum_l671_67177


namespace NUMINAMATH_CALUDE_negation_at_most_two_solutions_l671_67121

/-- Negation of "at most n" is "at least n+1" -/
axiom negation_at_most (n : ℕ) : ¬(∀ m : ℕ, m ≤ n) ↔ ∃ m : ℕ, m ≥ n + 1

/-- The negation of "there are at most two solutions" is equivalent to "there are at least three solutions" -/
theorem negation_at_most_two_solutions :
  ¬(∃ S : Set ℕ, (∀ n ∈ S, n ≤ 2)) ↔ ∃ S : Set ℕ, (∃ n ∈ S, n ≥ 3) :=
sorry

end NUMINAMATH_CALUDE_negation_at_most_two_solutions_l671_67121


namespace NUMINAMATH_CALUDE_domain_of_composed_function_l671_67112

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f
def domain_f : Set ℝ := Set.Icc (-2) 2

-- State the theorem
theorem domain_of_composed_function :
  (∀ x ∈ domain_f, f x ≠ 0) →
  {x : ℝ | f (2*x + 1) ≠ 0} = Set.Icc (-3/2) (1/2) := by
  sorry

end NUMINAMATH_CALUDE_domain_of_composed_function_l671_67112


namespace NUMINAMATH_CALUDE_divisible_by_ten_l671_67178

theorem divisible_by_ten : ∃ k : ℤ, 43^43 - 17^17 = 10 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_ten_l671_67178


namespace NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l671_67158

/-- An arithmetic progression with a non-zero difference -/
def arithmetic_progression (a : ℕ → ℝ) (d : ℝ) : Prop :=
  d ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n + d

/-- Consecutive terms of a geometric progression -/
def geometric_progression (x y z : ℝ) : Prop :=
  y * y = x * z

/-- The main theorem -/
theorem arithmetic_geometric_ratio
  (a : ℕ → ℝ) (d : ℝ)
  (h_arith : arithmetic_progression a d)
  (h_geom : geometric_progression (a 10) (a 13) (a 19)) :
  (a 12) / (a 18) = 5 / 11 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l671_67158


namespace NUMINAMATH_CALUDE_at_least_four_same_prob_l671_67183

-- Define the number of dice and sides
def num_dice : ℕ := 5
def num_sides : ℕ := 8

-- Define the probability of a specific outcome for a single die
def single_prob : ℚ := 1 / num_sides

-- Define the probability of all five dice showing the same number
def all_same_prob : ℚ := single_prob ^ (num_dice - 1)

-- Define the probability of exactly four dice showing the same number
def four_same_prob : ℚ := 
  (num_dice : ℚ) * single_prob ^ (num_dice - 2) * (1 - single_prob)

-- State the theorem
theorem at_least_four_same_prob : 
  all_same_prob + four_same_prob = 9 / 1024 := by sorry

end NUMINAMATH_CALUDE_at_least_four_same_prob_l671_67183


namespace NUMINAMATH_CALUDE_tank_insulation_cost_l671_67122

/-- Represents the dimensions of a rectangular tank -/
structure TankDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the surface area of a rectangular tank -/
def surfaceArea (d : TankDimensions) : ℝ :=
  2 * (d.length * d.width + d.length * d.height + d.width * d.height)

/-- Calculates the cost of insulation for a given surface area and cost per square foot -/
def insulationCost (area : ℝ) (costPerSqFt : ℝ) : ℝ :=
  area * costPerSqFt

/-- Theorem: The cost to insulate a tank with given dimensions is $1640 -/
theorem tank_insulation_cost :
  let tankDim : TankDimensions := { length := 7, width := 3, height := 2 }
  let costPerSqFt : ℝ := 20
  insulationCost (surfaceArea tankDim) costPerSqFt = 1640 := by
  sorry


end NUMINAMATH_CALUDE_tank_insulation_cost_l671_67122


namespace NUMINAMATH_CALUDE_one_in_set_zero_one_l671_67154

theorem one_in_set_zero_one : 1 ∈ ({0, 1} : Set ℕ) := by sorry

end NUMINAMATH_CALUDE_one_in_set_zero_one_l671_67154


namespace NUMINAMATH_CALUDE_initial_charge_correct_l671_67191

/-- The initial charge for renting a bike at Oceanside Bike Rental Shop -/
def initial_charge : ℝ := 17

/-- The hourly rate for renting a bike -/
def hourly_rate : ℝ := 7

/-- The number of hours Tom rented the bike -/
def rental_hours : ℝ := 9

/-- The total cost Tom paid for renting the bike -/
def total_cost : ℝ := 80

/-- Theorem stating that the initial charge is correct given the conditions -/
theorem initial_charge_correct : 
  initial_charge + hourly_rate * rental_hours = total_cost :=
by sorry

end NUMINAMATH_CALUDE_initial_charge_correct_l671_67191


namespace NUMINAMATH_CALUDE_sum_and_product_of_radical_conjugates_l671_67135

theorem sum_and_product_of_radical_conjugates (a b : ℝ) : 
  ((a + Real.sqrt b) + (a - Real.sqrt b) = -6) →
  ((a + Real.sqrt b) * (a - Real.sqrt b) = 9) →
  (a + b = -3) := by
  sorry

end NUMINAMATH_CALUDE_sum_and_product_of_radical_conjugates_l671_67135


namespace NUMINAMATH_CALUDE_total_nuts_equals_1_05_l671_67106

/-- The amount of walnuts Karen added to the trail mix in cups -/
def w : ℝ := 0.25

/-- The amount of almonds Karen added to the trail mix in cups -/
def a : ℝ := 0.25

/-- The amount of peanuts Karen added to the trail mix in cups -/
def p : ℝ := 0.15

/-- The amount of cashews Karen added to the trail mix in cups -/
def c : ℝ := 0.40

/-- The total amount of nuts Karen added to the trail mix -/
def total_nuts : ℝ := w + a + p + c

theorem total_nuts_equals_1_05 : total_nuts = 1.05 := by sorry

end NUMINAMATH_CALUDE_total_nuts_equals_1_05_l671_67106

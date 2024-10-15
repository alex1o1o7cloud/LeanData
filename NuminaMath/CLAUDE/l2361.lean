import Mathlib

namespace NUMINAMATH_CALUDE_part_one_part_two_l2361_236126

-- Define the sets A and B
def A : Set ℝ := {x | -1 < x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | -x^2 + 2*x + m > 0}

-- Part 1
theorem part_one : A ∩ (Set.univ \ B 3) = {x | 3 ≤ x ∧ x ≤ 5} := by sorry

-- Part 2
theorem part_two : ∃ m : ℝ, m = 8 ∧ A ∩ B m = {x | -1 < x ∧ x < 4} := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2361_236126


namespace NUMINAMATH_CALUDE_pythagorean_triple_identification_l2361_236134

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

theorem pythagorean_triple_identification :
  is_pythagorean_triple 3 4 5 ∧
  is_pythagorean_triple 5 12 13 ∧
  ¬(is_pythagorean_triple 4 5 6) ∧
  is_pythagorean_triple 8 15 17 :=
by sorry

end NUMINAMATH_CALUDE_pythagorean_triple_identification_l2361_236134


namespace NUMINAMATH_CALUDE_ellipse_properties_l2361_236107

-- Define the ellipse C
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the parabola
def parabola (x y : ℝ) : Prop :=
  y^2 = -8 * x

-- Define the line l
def line_l (x y : ℝ) : Prop :=
  y = x - 2

theorem ellipse_properties (a b c : ℝ) :
  a > b ∧ b > 0 ∧
  ellipse a b (-Real.sqrt 3) 1 ∧
  c = 2 ∧
  b^2 = a^2 - 4 ∧
  (∃ x y, ellipse a b x y ∧ parabola x y) →
  (a^2 = 6 ∧
   (∀ x y, ellipse a b x y ↔ x^2 / 6 + y^2 / 2 = 1) ∧
   (∃ x₁ y₁ x₂ y₂, 
      ellipse a b x₁ y₁ ∧ 
      ellipse a b x₂ y₂ ∧
      line_l x₁ y₁ ∧
      line_l x₂ y₂ ∧
      (x₂ - x₁)^2 + (y₂ - y₁)^2 = 6)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l2361_236107


namespace NUMINAMATH_CALUDE_function_not_in_second_quadrant_l2361_236128

/-- The function f(x) = a^x + b does not pass through the second quadrant when a > 1 and b < -1 -/
theorem function_not_in_second_quadrant (a b : ℝ) (ha : a > 1) (hb : b < -1) :
  ∀ x : ℝ, x < 0 → a^x + b ≤ 0 := by sorry

end NUMINAMATH_CALUDE_function_not_in_second_quadrant_l2361_236128


namespace NUMINAMATH_CALUDE_base_conversion_sum_rounded_to_28_l2361_236178

/-- Converts a number from base b to base 10 -/
def to_base_10 (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * b ^ i) 0

/-- Rounds a rational number to the nearest integer -/
def round_to_nearest (q : Rat) : Int :=
  (q + 1/2).floor

theorem base_conversion_sum_rounded_to_28 :
  let a := to_base_10 [4, 5, 2] 8  -- 254 in base 8
  let b := to_base_10 [3, 1] 4     -- 13 in base 4
  let c := to_base_10 [2, 3, 1] 5  -- 132 in base 5
  let d := to_base_10 [2, 3] 4     -- 32 in base 4
  round_to_nearest ((a / b : Rat) + (c / d : Rat)) = 28 := by
  sorry

#eval round_to_nearest ((172 / 7 : Rat) + (42 / 14 : Rat))

end NUMINAMATH_CALUDE_base_conversion_sum_rounded_to_28_l2361_236178


namespace NUMINAMATH_CALUDE_theater_ticket_price_l2361_236167

/-- Proves that the cost of an orchestra seat is $12 given the conditions of the theater problem -/
theorem theater_ticket_price
  (total_tickets : ℕ)
  (total_revenue : ℕ)
  (balcony_price : ℕ)
  (ticket_difference : ℕ)
  (h1 : total_tickets = 355)
  (h2 : total_revenue = 3320)
  (h3 : balcony_price = 8)
  (h4 : ticket_difference = 115) :
  ∃ (orchestra_price : ℕ),
    orchestra_price = 12 ∧
    ∃ (orchestra_tickets : ℕ),
      orchestra_tickets + (orchestra_tickets + ticket_difference) = total_tickets ∧
      orchestra_price * orchestra_tickets + balcony_price * (orchestra_tickets + ticket_difference) = total_revenue :=
by
  sorry

end NUMINAMATH_CALUDE_theater_ticket_price_l2361_236167


namespace NUMINAMATH_CALUDE_roberts_balls_l2361_236184

theorem roberts_balls (robert_initial : ℕ) (tim_initial : ℕ) : 
  robert_initial = 25 → 
  tim_initial = 40 → 
  robert_initial + tim_initial / 2 = 45 := by
  sorry

end NUMINAMATH_CALUDE_roberts_balls_l2361_236184


namespace NUMINAMATH_CALUDE_hotel_charge_difference_l2361_236146

/-- The charge for a single room at different hotels -/
structure HotelCharges where
  G : ℝ  -- Charge at hotel G
  R : ℝ  -- Charge at hotel R
  P : ℝ  -- Charge at hotel P

/-- The conditions given in the problem -/
def problem_conditions (h : HotelCharges) : Prop :=
  h.P = 0.9 * h.G ∧ 
  h.R = 1.125 * h.G

/-- The theorem stating the percentage difference between charges at hotel P and R -/
theorem hotel_charge_difference (h : HotelCharges) 
  (hcond : problem_conditions h) : 
  (h.R - h.P) / h.R = 0.2 := by
  sorry


end NUMINAMATH_CALUDE_hotel_charge_difference_l2361_236146


namespace NUMINAMATH_CALUDE_travel_time_proof_l2361_236114

/-- Given a person traveling at a constant speed, this theorem proves that
    the travel time is 5 hours when the distance is 500 km and the speed is 100 km/hr. -/
theorem travel_time_proof (distance : ℝ) (speed : ℝ) (time : ℝ) :
  distance = 500 ∧ speed = 100 ∧ time = distance / speed → time = 5 := by
  sorry

end NUMINAMATH_CALUDE_travel_time_proof_l2361_236114


namespace NUMINAMATH_CALUDE_greatest_three_digit_divisible_by_5_and_10_l2361_236179

theorem greatest_three_digit_divisible_by_5_and_10 : ∃ n : ℕ, 
  (100 ≤ n ∧ n ≤ 999) ∧ 
  n % 5 = 0 ∧ 
  n % 10 = 0 ∧ 
  ∀ m : ℕ, (100 ≤ m ∧ m ≤ 999 ∧ m % 5 = 0 ∧ m % 10 = 0) → m ≤ n :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_greatest_three_digit_divisible_by_5_and_10_l2361_236179


namespace NUMINAMATH_CALUDE_wood_length_l2361_236156

/-- The original length of a piece of wood, given the length sawed off and the remaining length -/
theorem wood_length (sawed_off : ℝ) (remaining : ℝ) (h1 : sawed_off = 2.3) (h2 : remaining = 6.6) :
  sawed_off + remaining = 8.9 := by
  sorry

end NUMINAMATH_CALUDE_wood_length_l2361_236156


namespace NUMINAMATH_CALUDE_caitlin_age_l2361_236170

/-- Proves that Caitlin's age is 13 years given the ages of Aunt Anna and Brianna -/
theorem caitlin_age (anna_age : ℕ) (brianna_age : ℕ) (caitlin_age : ℕ)
  (h1 : anna_age = 60)
  (h2 : brianna_age = anna_age / 3)
  (h3 : caitlin_age = brianna_age - 7) :
  caitlin_age = 13 := by
sorry

end NUMINAMATH_CALUDE_caitlin_age_l2361_236170


namespace NUMINAMATH_CALUDE_chocolate_gain_percent_l2361_236196

/-- Calculates the gain percent given the number of chocolates at cost price and selling price that are equal in value. -/
def gain_percent (cost_count : ℕ) (sell_count : ℕ) : ℚ :=
  ((cost_count - sell_count) / sell_count) * 100

/-- Theorem stating that when the cost price of 65 chocolates equals the selling price of 50 chocolates, the gain percent is 30%. -/
theorem chocolate_gain_percent :
  gain_percent 65 50 = 30 := by
  sorry

#eval gain_percent 65 50

end NUMINAMATH_CALUDE_chocolate_gain_percent_l2361_236196


namespace NUMINAMATH_CALUDE_range_of_a_l2361_236191

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -2 < x ∧ x < 4}
def B (a : ℝ) : Set ℝ := {x : ℝ | x < a}

-- Define the theorem
theorem range_of_a (a : ℝ) : 
  (A ∪ B a = {x : ℝ | x < 4}) ↔ (-2 < a ∧ a ≤ 4) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2361_236191


namespace NUMINAMATH_CALUDE_frustum_volume_l2361_236188

/-- The volume of a frustum formed by cutting a triangular pyramid parallel to its base --/
theorem frustum_volume (base_edge : ℝ) (altitude : ℝ) (small_base_edge : ℝ) (small_altitude : ℝ) :
  base_edge = 18 →
  altitude = 9 →
  small_base_edge = 9 →
  small_altitude = 3 →
  ∃ (v : ℝ), v = 212.625 * Real.sqrt 3 ∧ v = 
    ((1/3 * (Real.sqrt 3 / 4) * base_edge^2 * altitude) - 
     (1/3 * (Real.sqrt 3 / 4) * small_base_edge^2 * small_altitude)) :=
by sorry

end NUMINAMATH_CALUDE_frustum_volume_l2361_236188


namespace NUMINAMATH_CALUDE_two_number_problem_l2361_236111

theorem two_number_problem (x y : ℚ) 
  (sum_eq : x + y = 40)
  (double_subtract : 2 * y - 4 * x = 12) :
  |y - x| = 52 / 3 := by
sorry

end NUMINAMATH_CALUDE_two_number_problem_l2361_236111


namespace NUMINAMATH_CALUDE_pq_length_l2361_236186

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line
def line (x y : ℝ) : Prop := y = Real.sqrt 3 * (x - 1)

-- Define the intersection points
def intersection_points (P Q : ℝ × ℝ) : Prop :=
  parabola P.1 P.2 ∧ parabola Q.1 Q.2 ∧ line P.1 P.2 ∧ line Q.1 Q.2

-- Theorem statement
theorem pq_length (P Q : ℝ × ℝ) :
  intersection_points P Q →
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 16/3 := by sorry

end NUMINAMATH_CALUDE_pq_length_l2361_236186


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l2361_236106

theorem completing_square_equivalence :
  ∀ x : ℝ, x^2 + 4*x + 1 = 0 ↔ (x + 2)^2 = 3 := by
sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l2361_236106


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_l2361_236152

theorem pure_imaginary_complex (a : ℝ) : 
  (a - (10 : ℂ) / (3 - I)).im ≠ 0 ∧ (a - (10 : ℂ) / (3 - I)).re = 0 → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_l2361_236152


namespace NUMINAMATH_CALUDE_vector_magnitude_problem_l2361_236155

/-- Given vectors a and b in ℝ², where b = (-1, 2) and a + b = (1, 3),
    prove that the magnitude of a - 2b is equal to 5. -/
theorem vector_magnitude_problem (a b : ℝ × ℝ) 
  (h1 : b = (-1, 2))
  (h2 : a + b = (1, 3)) :
  ‖a - 2 • b‖ = 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_problem_l2361_236155


namespace NUMINAMATH_CALUDE_functional_equation_solutions_l2361_236198

open Real

-- Define the type of continuous functions from ℝ⁺ to ℝ⁺
def ContinuousPosFun := {f : ℝ → ℝ // Continuous f ∧ ∀ x, x > 0 → f x > 0}

-- Define the property that the function satisfies the given equation
def SatisfiesEquation (f : ContinuousPosFun) : Prop :=
  ∀ x, x > 0 → x + 1/x = f.val x + 1/(f.val x)

-- Define the set of possible solutions
def PossibleSolutions (x : ℝ) : Set ℝ :=
  {x, 1/x, max x (1/x), min x (1/x)}

-- State the theorem
theorem functional_equation_solutions (f : ContinuousPosFun) 
  (h : SatisfiesEquation f) :
  ∀ x, x > 0 → f.val x ∈ PossibleSolutions x := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solutions_l2361_236198


namespace NUMINAMATH_CALUDE_celine_initial_amount_l2361_236193

/-- The price of a laptop in dollars -/
def laptop_price : ℕ := 600

/-- The price of a smartphone in dollars -/
def smartphone_price : ℕ := 400

/-- The number of laptops Celine bought -/
def laptops_bought : ℕ := 2

/-- The number of smartphones Celine bought -/
def smartphones_bought : ℕ := 4

/-- The amount of change Celine received in dollars -/
def change_received : ℕ := 200

/-- Celine's initial amount of money in dollars -/
def initial_amount : ℕ := laptop_price * laptops_bought + smartphone_price * smartphones_bought + change_received

theorem celine_initial_amount : initial_amount = 3000 := by
  sorry

end NUMINAMATH_CALUDE_celine_initial_amount_l2361_236193


namespace NUMINAMATH_CALUDE_exists_same_color_unit_apart_l2361_236180

/-- A coloring of the plane using three colors -/
def Coloring := ℝ × ℝ → Fin 3

/-- Two points are one unit apart -/
def one_unit_apart (p q : ℝ × ℝ) : Prop :=
  (p.1 - q.1)^2 + (p.2 - q.2)^2 = 1

/-- Main theorem: In any three-coloring of the plane, there exist two points of the same color that are exactly one unit apart -/
theorem exists_same_color_unit_apart (c : Coloring) : 
  ∃ (p q : ℝ × ℝ), c p = c q ∧ one_unit_apart p q := by
  sorry

end NUMINAMATH_CALUDE_exists_same_color_unit_apart_l2361_236180


namespace NUMINAMATH_CALUDE_geometry_theorem_l2361_236166

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (perpendicular : Line → Line → Prop)
variable (perpendicular_plane_line : Plane → Line → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (contained_in : Line → Plane → Prop)
variable (intersection : Plane → Plane → Line → Prop)

-- Theorem statement
theorem geometry_theorem 
  (α β : Plane) (m n l : Line) : 
  (∀ m n α β, perpendicular m n → perpendicular_plane_line α m → perpendicular_plane_line β n → perpendicular_planes α β) ∧
  (∀ m α β, contained_in m α → parallel_planes α β → parallel_line_plane m β) ∧
  (∀ α β m l, intersection α β l → parallel_line_plane m α → parallel_line_plane m β → parallel_lines m l) ∧
  ¬(∀ m n α β, perpendicular m n → perpendicular_plane_line α m → parallel_line_plane n β → perpendicular_planes α β) :=
by sorry

end NUMINAMATH_CALUDE_geometry_theorem_l2361_236166


namespace NUMINAMATH_CALUDE_conditional_prob_B_given_A_l2361_236148

-- Define the sample space for a six-sided die
def Ω : Finset Nat := {1, 2, 3, 4, 5, 6}

-- Define event A: odd numbers
def A : Finset Nat := {1, 3, 5}

-- Define event B: getting 3 points
def B : Finset Nat := {3}

-- Define the probability measure
def P (S : Finset Nat) : ℚ := (S.card : ℚ) / (Ω.card : ℚ)

-- Theorem: P(B|A) = 1/3
theorem conditional_prob_B_given_A : 
  P (A ∩ B) / P A = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_conditional_prob_B_given_A_l2361_236148


namespace NUMINAMATH_CALUDE_inequality_solution_l2361_236157

theorem inequality_solution (x : ℝ) : (1 + x) / 3 < x / 2 ↔ x > 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l2361_236157


namespace NUMINAMATH_CALUDE_extra_time_with_speed_decrease_l2361_236185

/-- Given a 20% decrease in speed and an original travel time of 40 minutes,
    prove that the extra time taken to cover the same distance is 10 minutes. -/
theorem extra_time_with_speed_decrease (original_speed : ℝ) (original_time : ℝ) 
  (h1 : original_time = 40) 
  (h2 : original_speed > 0) : 
  let decreased_speed := 0.8 * original_speed
  let new_time := (original_speed * original_time) / decreased_speed
  new_time - original_time = 10 := by
  sorry

end NUMINAMATH_CALUDE_extra_time_with_speed_decrease_l2361_236185


namespace NUMINAMATH_CALUDE_sum_of_decimals_l2361_236101

theorem sum_of_decimals : 5.623 + 4.76 = 10.383 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_decimals_l2361_236101


namespace NUMINAMATH_CALUDE_octagon_area_in_square_l2361_236110

/-- Given a square with side length s, the area of the octagon formed by connecting each vertex
    to the midpoints of the opposite two sides is s^2/6. -/
theorem octagon_area_in_square (s : ℝ) (h : s > 0) :
  let square_area := s^2
  let octagon_area := square_area / 6
  octagon_area = square_area / 6 := by
sorry

end NUMINAMATH_CALUDE_octagon_area_in_square_l2361_236110


namespace NUMINAMATH_CALUDE_complex_equation_result_l2361_236174

theorem complex_equation_result (a b : ℝ) (h : (1 + Complex.I) * (1 - b * Complex.I) = a) : a / b = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_result_l2361_236174


namespace NUMINAMATH_CALUDE_rectangle_area_problem_l2361_236135

/-- Given a rectangle with initial dimensions 4 × 6 inches, if shortening one side by 2 inches
    results in an area of 12 square inches, then shortening the other side by 1 inch
    results in an area of 20 square inches. -/
theorem rectangle_area_problem :
  ∀ (length width : ℝ),
  length = 4 ∧ width = 6 →
  (∃ (shortened_side : ℝ),
    (shortened_side = length - 2 ∨ shortened_side = width - 2) ∧
    shortened_side * (if shortened_side = length - 2 then width else length) = 12) →
  (if length - 2 < width - 2 then (length * (width - 1)) else ((length - 1) * width)) = 20 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_problem_l2361_236135


namespace NUMINAMATH_CALUDE_paper_strip_to_squares_l2361_236177

/-- Represents a strip of paper with given width and length -/
structure PaperStrip where
  width : ℝ
  length : ℝ

/-- Represents a square -/
structure Square where
  side : ℝ

/-- Function to calculate the area of a square -/
def squareArea (s : Square) : ℝ := s.side * s.side

/-- Theorem stating that a paper strip of width 1 cm and length 4 cm
    can be transformed into squares of areas 1 sq cm and 2 sq cm -/
theorem paper_strip_to_squares 
  (strip : PaperStrip) 
  (h_width : strip.width = 1) 
  (h_length : strip.length = 4) :
  ∃ (s1 s2 : Square), 
    squareArea s1 = 1 ∧ 
    squareArea s2 = 2 := by
  sorry


end NUMINAMATH_CALUDE_paper_strip_to_squares_l2361_236177


namespace NUMINAMATH_CALUDE_cherries_purchase_l2361_236147

theorem cherries_purchase (cost_per_kg : ℝ) (short_amount : ℝ) (money_on_hand : ℝ) 
  (h1 : cost_per_kg = 8)
  (h2 : short_amount = 400)
  (h3 : money_on_hand = 1600) :
  (money_on_hand + short_amount) / cost_per_kg = 250 := by
  sorry

end NUMINAMATH_CALUDE_cherries_purchase_l2361_236147


namespace NUMINAMATH_CALUDE_phillips_cucumbers_l2361_236120

/-- Proves that Phillip has 8 cucumbers given the pickle-making conditions --/
theorem phillips_cucumbers :
  ∀ (jars : ℕ) (initial_vinegar : ℕ) (pickles_per_cucumber : ℕ) (pickles_per_jar : ℕ)
    (vinegar_per_jar : ℕ) (remaining_vinegar : ℕ),
  jars = 4 →
  initial_vinegar = 100 →
  pickles_per_cucumber = 6 →
  pickles_per_jar = 12 →
  vinegar_per_jar = 10 →
  remaining_vinegar = 60 →
  ∃ (cucumbers : ℕ),
    cucumbers = 8 ∧
    cucumbers * pickles_per_cucumber = jars * pickles_per_jar ∧
    initial_vinegar - remaining_vinegar = jars * vinegar_per_jar :=
by
  sorry


end NUMINAMATH_CALUDE_phillips_cucumbers_l2361_236120


namespace NUMINAMATH_CALUDE_writing_utensils_arrangement_l2361_236124

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def circular_permutations (n a b c d : ℕ) : ℕ :=
  factorial (n - 1) / (factorial a * factorial b * factorial c * factorial d)

def adjacent_arrangements (n a b c d : ℕ) : ℕ :=
  circular_permutations (n - 1) a 1 c d

theorem writing_utensils_arrangement :
  let total_items : ℕ := 5 + 3 + 1 + 1
  let black_pencils : ℕ := 5
  let blue_pens : ℕ := 3
  let red_pen : ℕ := 1
  let green_pen : ℕ := 1
  circular_permutations total_items black_pencils blue_pens red_pen green_pen -
  adjacent_arrangements total_items black_pencils blue_pens red_pen green_pen = 168 := by
sorry

end NUMINAMATH_CALUDE_writing_utensils_arrangement_l2361_236124


namespace NUMINAMATH_CALUDE_cos_pi_twelve_squared_identity_l2361_236142

theorem cos_pi_twelve_squared_identity : 2 * (Real.cos (π / 12))^2 - 1 = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_twelve_squared_identity_l2361_236142


namespace NUMINAMATH_CALUDE_min_value_6x_5y_l2361_236144

theorem min_value_6x_5y (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : 1 / (2 * x + y) + 3 / (x + y) = 2) :
  (∀ x' y' : ℝ, x' > 0 → y' > 0 → 1 / (2 * x' + y') + 3 / (x' + y') = 2 →
    6 * x + 5 * y ≤ 6 * x' + 5 * y') ∧
  6 * x + 5 * y = (13 + 4 * Real.sqrt 3) / 2 := by
sorry

end NUMINAMATH_CALUDE_min_value_6x_5y_l2361_236144


namespace NUMINAMATH_CALUDE_circle_area_ratio_l2361_236160

theorem circle_area_ratio (R_A R_B R_C : ℝ) 
  (h1 : (60 / 360) * (2 * π * R_A) = (40 / 360) * (2 * π * R_B))
  (h2 : (30 / 360) * (2 * π * R_B) = (90 / 360) * (2 * π * R_C)) :
  (π * R_A^2) / (π * R_C^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l2361_236160


namespace NUMINAMATH_CALUDE_simplify_expressions_l2361_236121

variable (a b x y : ℝ)

theorem simplify_expressions :
  (2 * a - (a + b) = a - b) ∧
  ((x^2 - 2*y^2) - 2*(3*y^2 - 2*x^2) = 5*x^2 - 8*y^2) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expressions_l2361_236121


namespace NUMINAMATH_CALUDE_stratified_by_stage_is_most_reasonable_l2361_236176

-- Define the possible sampling methods
inductive SamplingMethod
| SimpleRandom
| StratifiedByGender
| StratifiedByEducationalStage
| Systematic

-- Define the characteristics of the population
structure PopulationCharacteristics where
  significantDifferenceByStage : Bool
  significantDifferenceByGender : Bool

-- Define the function to determine the most reasonable sampling method
def mostReasonableSamplingMethod (pop : PopulationCharacteristics) : SamplingMethod :=
  sorry

-- Theorem statement
theorem stratified_by_stage_is_most_reasonable 
  (pop : PopulationCharacteristics) 
  (h1 : pop.significantDifferenceByStage = true) 
  (h2 : pop.significantDifferenceByGender = false) :
  mostReasonableSamplingMethod pop = SamplingMethod.StratifiedByEducationalStage :=
sorry

end NUMINAMATH_CALUDE_stratified_by_stage_is_most_reasonable_l2361_236176


namespace NUMINAMATH_CALUDE_pyramid_with_10_edges_has_6_vertices_l2361_236197

-- Define a pyramid structure
structure Pyramid where
  base_sides : ℕ
  edges : ℕ
  vertices : ℕ

-- Theorem statement
theorem pyramid_with_10_edges_has_6_vertices :
  ∀ p : Pyramid, p.edges = 10 → p.vertices = 6 := by
  sorry


end NUMINAMATH_CALUDE_pyramid_with_10_edges_has_6_vertices_l2361_236197


namespace NUMINAMATH_CALUDE_m_subset_p_subset_n_l2361_236199

/-- Set M definition -/
def M : Set (ℝ × ℝ) := {p | abs p.1 + abs p.2 < 1}

/-- Set N definition -/
def N : Set (ℝ × ℝ) := {p | Real.sqrt ((p.1 - 1/2)^2 + (p.2 + 1/2)^2) + Real.sqrt ((p.1 + 1/2)^2 + (p.2 - 1/2)^2) < 2 * Real.sqrt 2}

/-- Set P definition -/
def P : Set (ℝ × ℝ) := {p | abs (p.1 + p.2) < 1 ∧ abs p.1 < 1 ∧ abs p.2 < 1}

/-- Theorem stating M ⊂ P ⊂ N -/
theorem m_subset_p_subset_n : M ⊆ P ∧ P ⊆ N := by sorry

end NUMINAMATH_CALUDE_m_subset_p_subset_n_l2361_236199


namespace NUMINAMATH_CALUDE_employee_count_l2361_236158

/-- The number of employees in an organization (excluding the manager) -/
def num_employees : ℕ := 20

/-- The average monthly salary of employees (excluding the manager) -/
def avg_salary : ℕ := 1600

/-- The increase in average salary when the manager's salary is added -/
def salary_increase : ℕ := 100

/-- The manager's monthly salary -/
def manager_salary : ℕ := 3700

/-- Theorem stating the number of employees given the salary conditions -/
theorem employee_count :
  (num_employees * avg_salary + manager_salary) / (num_employees + 1) =
  avg_salary + salary_increase :=
by sorry

end NUMINAMATH_CALUDE_employee_count_l2361_236158


namespace NUMINAMATH_CALUDE_vector_difference_magnitude_l2361_236138

def a : Fin 2 → ℝ := ![1, 2]
def b (x : ℝ) : Fin 2 → ℝ := ![x, 6]

theorem vector_difference_magnitude 
  (h_parallel : ∃ (k : ℝ), ∀ i, a i = k * b x i) :
  ‖a - b x‖ = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_difference_magnitude_l2361_236138


namespace NUMINAMATH_CALUDE_expression_value_at_three_l2361_236116

theorem expression_value_at_three :
  let f (x : ℝ) := (x^2 - 3*x - 10) / (x - 5)
  f 3 = 5 := by
sorry

end NUMINAMATH_CALUDE_expression_value_at_three_l2361_236116


namespace NUMINAMATH_CALUDE_intersect_x_axis_and_derivative_negative_l2361_236140

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x + a

theorem intersect_x_axis_and_derivative_negative (a : ℝ) (x₁ x₂ : ℝ) :
  a > Real.exp 2 →
  x₁ < x₂ →
  f a x₁ = 0 →
  f a x₂ = 0 →
  let x₀ := Real.sqrt (x₁ * x₂)
  (deriv (f a)) x₀ < 0 :=
by sorry

end NUMINAMATH_CALUDE_intersect_x_axis_and_derivative_negative_l2361_236140


namespace NUMINAMATH_CALUDE_xy_value_l2361_236164

theorem xy_value (x y : ℝ) (h : (|x| - 1)^2 + (2*y + 1)^2 = 0) : x * y = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l2361_236164


namespace NUMINAMATH_CALUDE_nail_multiple_l2361_236131

theorem nail_multiple (violet_nails : ℕ) (total_nails : ℕ) (M : ℕ) : 
  violet_nails = 27 →
  total_nails = 39 →
  violet_nails = M * (total_nails - violet_nails) + 3 →
  M = 2 := by sorry

end NUMINAMATH_CALUDE_nail_multiple_l2361_236131


namespace NUMINAMATH_CALUDE_min_value_xyz_l2361_236153

/-- Given positive real numbers x, y, and z satisfying 1/x + 1/y + 1/z = 9,
    the minimum value of x^2 * y^3 * z is 729/6912 -/
theorem min_value_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : 1/x + 1/y + 1/z = 9) :
  ∃ (m : ℝ), m = 729/6912 ∧ ∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 →
    1/a + 1/b + 1/c = 9 → a^2 * b^3 * c ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_xyz_l2361_236153


namespace NUMINAMATH_CALUDE_arithmetic_equalities_l2361_236105

theorem arithmetic_equalities :
  (-16 - (-12) - 24 + 18 = -10) ∧
  (0.125 + 1/4 + (-2 - 1/8) + (-0.25) = -2) ∧
  ((-1/12 - 1/36 + 1/6) * (-36) = -2) ∧
  ((-2 + 3) * 3 - (-2)^3 / 4 = 5) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equalities_l2361_236105


namespace NUMINAMATH_CALUDE_minimum_time_for_given_problem_l2361_236195

/-- Represents the problem of replacing shades in chandeliers --/
structure ChandelierProblem where
  num_chandeliers : ℕ
  shades_per_chandelier : ℕ
  time_per_shade : ℕ
  num_electricians : ℕ

/-- Calculates the minimum time required to replace all shades --/
def minimum_replacement_time (p : ChandelierProblem) : ℕ :=
  let total_shades := p.num_chandeliers * p.shades_per_chandelier
  let total_work_time := total_shades * p.time_per_shade
  (total_work_time + p.num_electricians - 1) / p.num_electricians

/-- Theorem stating the minimum time for the given problem --/
theorem minimum_time_for_given_problem :
  let p : ChandelierProblem := {
    num_chandeliers := 60,
    shades_per_chandelier := 4,
    time_per_shade := 5,
    num_electricians := 48
  }
  minimum_replacement_time p = 25 := by sorry


end NUMINAMATH_CALUDE_minimum_time_for_given_problem_l2361_236195


namespace NUMINAMATH_CALUDE_condition_relationship_l2361_236150

theorem condition_relationship (θ : ℝ) (a : ℝ) : 
  ¬(∀ θ a, (Real.sqrt (1 + Real.sin θ) = a) ↔ (Real.sin (θ/2) + Real.cos (θ/2) = a)) :=
by sorry

end NUMINAMATH_CALUDE_condition_relationship_l2361_236150


namespace NUMINAMATH_CALUDE_range_of_m_l2361_236172

theorem range_of_m (x : ℝ) :
  (∀ x, (1/3 < x ∧ x < 1/2) → (m - 1 < x ∧ x < m + 1)) →
  (-1/2 ≤ m ∧ m ≤ 4/3) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l2361_236172


namespace NUMINAMATH_CALUDE_sector_arc_length_l2361_236132

theorem sector_arc_length (area : ℝ) (angle : ℝ) (arc_length : ℝ) : 
  area = 4 → angle = 2 → arc_length = 4 := by
  sorry

end NUMINAMATH_CALUDE_sector_arc_length_l2361_236132


namespace NUMINAMATH_CALUDE_quadratic_function_minimum_l2361_236108

/-- A quadratic function that takes values 6, 5, and 5 for three consecutive natural values. -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ n : ℕ, f n = 6 ∧ f (n + 1) = 5 ∧ f (n + 2) = 5

/-- The theorem stating that the minimum value of the quadratic function is 5. -/
theorem quadratic_function_minimum (f : ℝ → ℝ) (h : QuadraticFunction f) :
  ∃ x : ℝ, ∀ y : ℝ, f x ≤ f y ∧ f x = 5 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_minimum_l2361_236108


namespace NUMINAMATH_CALUDE_cube_rotation_theorem_l2361_236136

/-- Represents a cube with numbers on its faces -/
structure Cube where
  left : ℕ
  right : ℕ
  front : ℕ
  back : ℕ
  top : ℕ
  bottom : ℕ

/-- Represents the state of the cube after rotations -/
structure CubeState where
  bottom : ℕ
  front : ℕ
  right : ℕ

/-- Rotates the cube from left to right -/
def rotateLeftRight (c : Cube) : Cube := sorry

/-- Rotates the cube from front to back -/
def rotateFrontBack (c : Cube) : Cube := sorry

/-- Applies multiple rotations to the cube -/
def applyRotations (c : Cube) (leftRightRotations frontBackRotations : ℕ) : Cube := sorry

/-- Theorem stating the final state of the cube after rotations -/
theorem cube_rotation_theorem (c : Cube) 
  (h1 : c.left + c.right = 50)
  (h2 : c.front + c.back = 50)
  (h3 : c.top + c.bottom = 50) :
  let finalCube := applyRotations c 97 98
  CubeState.mk finalCube.bottom finalCube.front finalCube.right = CubeState.mk 13 35 11 := by sorry

end NUMINAMATH_CALUDE_cube_rotation_theorem_l2361_236136


namespace NUMINAMATH_CALUDE_program_output_is_44_l2361_236122

/-- The output value of the program -/
def program_output : ℕ := 44

/-- Theorem stating that the program output is 44 -/
theorem program_output_is_44 : program_output = 44 := by
  sorry

end NUMINAMATH_CALUDE_program_output_is_44_l2361_236122


namespace NUMINAMATH_CALUDE_robbery_trial_l2361_236162

theorem robbery_trial (A B C : Prop) 
  (h1 : (¬A ∨ B) → C)
  (h2 : ¬A → ¬C) : 
  A ∧ C ∧ (B ∨ ¬B) := by
sorry

end NUMINAMATH_CALUDE_robbery_trial_l2361_236162


namespace NUMINAMATH_CALUDE_doctor_nurse_ratio_l2361_236143

theorem doctor_nurse_ratio (total : ℕ) (nurses : ℕ) (h1 : total = 200) (h2 : nurses = 120) :
  (total - nurses) / nurses = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_doctor_nurse_ratio_l2361_236143


namespace NUMINAMATH_CALUDE_equation_solutions_l2361_236133

theorem equation_solutions : 
  {x : ℝ | 3 * x + 6 = |(-10 + 5 * x)|} = {8, (1/2 : ℝ)} := by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2361_236133


namespace NUMINAMATH_CALUDE_shirt_sale_price_l2361_236187

theorem shirt_sale_price (original_price : ℝ) (original_price_pos : original_price > 0) :
  let first_sale_price := original_price * (1 - 0.5)
  let final_price := first_sale_price * (1 - 0.1)
  final_price / original_price = 0.45 := by
sorry

end NUMINAMATH_CALUDE_shirt_sale_price_l2361_236187


namespace NUMINAMATH_CALUDE_factorization_a4_plus_4_l2361_236151

theorem factorization_a4_plus_4 (a : ℝ) : a^4 + 4 = (a^2 + 2*a + 2)*(a^2 - 2*a + 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_a4_plus_4_l2361_236151


namespace NUMINAMATH_CALUDE_sum_is_composite_l2361_236119

theorem sum_is_composite (a b c d : ℕ+) (h : a^2 + b^2 = c^2 + d^2) :
  ∃ (k m : ℕ), k > 1 ∧ m > 1 ∧ (a : ℕ) + b + c + d = k * m := by
  sorry

end NUMINAMATH_CALUDE_sum_is_composite_l2361_236119


namespace NUMINAMATH_CALUDE_fifth_power_sum_l2361_236145

theorem fifth_power_sum (a b x y : ℝ) 
  (h1 : a * x + b * y = 5)
  (h2 : a * x^2 + b * y^2 = 9)
  (h3 : a * x^3 + b * y^3 = 21)
  (h4 : a * x^4 + b * y^4 = 55) :
  a * x^5 + b * y^5 = -131 := by
  sorry

end NUMINAMATH_CALUDE_fifth_power_sum_l2361_236145


namespace NUMINAMATH_CALUDE_tournament_committee_count_l2361_236112

/-- The number of teams in the league -/
def num_teams : ℕ := 5

/-- The number of members in each team -/
def team_size : ℕ := 7

/-- The minimum number of female members in each team -/
def min_females : ℕ := 2

/-- The number of members selected for the committee by the host team -/
def host_committee_size : ℕ := 3

/-- The number of members selected for the committee by non-host teams -/
def non_host_committee_size : ℕ := 2

/-- The total number of members in the tournament committee -/
def total_committee_size : ℕ := 10

/-- The number of possible tournament committees -/
def num_committees : ℕ := 1296540

theorem tournament_committee_count :
  (num_teams > 0) →
  (team_size ≥ host_committee_size) →
  (team_size ≥ non_host_committee_size) →
  (min_females ≥ non_host_committee_size) →
  (min_females < host_committee_size) →
  (num_teams * non_host_committee_size + host_committee_size = total_committee_size) →
  (num_committees = (num_teams - 1) * (Nat.choose team_size host_committee_size) * 
    (Nat.choose team_size non_host_committee_size)^(num_teams - 2) * 
    (Nat.choose min_females non_host_committee_size)) :=
by sorry

#check tournament_committee_count

end NUMINAMATH_CALUDE_tournament_committee_count_l2361_236112


namespace NUMINAMATH_CALUDE_chinese_remainder_theorem_example_l2361_236104

theorem chinese_remainder_theorem_example : ∃ n : ℕ, 
  n > 0 ∧ 
  n % 3 = 2 ∧ 
  n % 5 = 3 ∧ 
  n % 7 = 2 ∧ 
  ∀ m : ℕ, m > 0 → m % 3 = 2 → m % 5 = 3 → m % 7 = 2 → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_chinese_remainder_theorem_example_l2361_236104


namespace NUMINAMATH_CALUDE_solutions_to_z_sixth_eq_neg_64_l2361_236123

theorem solutions_to_z_sixth_eq_neg_64 :
  {z : ℂ | z^6 = -64} =
    {2 * (Complex.cos (π / 6) + Complex.I * Complex.sin (π / 6)),
     2 * (Complex.cos (π / 2) + Complex.I * Complex.sin (π / 2)),
     2 * (Complex.cos (5 * π / 6) + Complex.I * Complex.sin (5 * π / 6)),
     2 * (Complex.cos (7 * π / 6) + Complex.I * Complex.sin (7 * π / 6)),
     2 * (Complex.cos (3 * π / 2) + Complex.I * Complex.sin (3 * π / 2)),
     2 * (Complex.cos (11 * π / 6) + Complex.I * Complex.sin (11 * π / 6))} :=
by sorry

end NUMINAMATH_CALUDE_solutions_to_z_sixth_eq_neg_64_l2361_236123


namespace NUMINAMATH_CALUDE_power_product_equality_l2361_236125

theorem power_product_equality (a : ℝ) : a^2 * (-a)^2 = a^4 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equality_l2361_236125


namespace NUMINAMATH_CALUDE_f_decreasing_range_f_less_than_g_range_l2361_236168

open Real

noncomputable def f (a x : ℝ) : ℝ := log x - a^2 * x^2 + a * x

noncomputable def g (a x : ℝ) : ℝ := (3*a + 1) * x - (a^2 + a) * x^2

theorem f_decreasing_range (a : ℝ) (h : a ≠ 0) :
  (∀ x ≥ 1, ∀ y ≥ x, f a x ≥ f a y) ↔ a ≥ 1 :=
sorry

theorem f_less_than_g_range (a : ℝ) (h : a ≠ 0) :
  (∀ x > 1, f a x < g a x) ↔ -1 < a ∧ a ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_f_decreasing_range_f_less_than_g_range_l2361_236168


namespace NUMINAMATH_CALUDE_line_through_point_parallel_to_line_line_equation_proof_l2361_236183

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Point2D.liesOn (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def Line2D.isParallel (l1 l2 : Line2D) : Prop :=
  l1.a * l2.b = l1.b * l2.a

theorem line_through_point_parallel_to_line 
  (given_point : Point2D) 
  (given_line : Line2D) 
  (result_line : Line2D) : Prop :=
  (given_point.liesOn result_line) ∧ 
  (result_line.isParallel given_line) →
  (result_line.a = 2 ∧ result_line.b = -1 ∧ result_line.c = 4)

#check line_through_point_parallel_to_line 
  (Point2D.mk 0 4) 
  (Line2D.mk 2 (-1) (-3)) 
  (Line2D.mk 2 (-1) 4)

theorem line_equation_proof : 
  line_through_point_parallel_to_line 
    (Point2D.mk 0 4) 
    (Line2D.mk 2 (-1) (-3)) 
    (Line2D.mk 2 (-1) 4) := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_parallel_to_line_line_equation_proof_l2361_236183


namespace NUMINAMATH_CALUDE_intersection_line_of_circles_l2361_236118

/-- Given two circles in the plane, this theorem states that the line passing through
    their intersection points has a specific equation. -/
theorem intersection_line_of_circles (x y : ℝ) : 
  (x^2 + y^2 + 2*x + 3*y = 0) →
  (x^2 + y^2 - 4*x + 2*y + 1 = 0) →
  (6*x + y - 1 = 0) := by
sorry

end NUMINAMATH_CALUDE_intersection_line_of_circles_l2361_236118


namespace NUMINAMATH_CALUDE_sum_interior_angles_polygon_l2361_236181

/-- The sum of interior angles of an n-sided polygon -/
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- The number of triangles formed by drawing diagonals from one vertex -/
def num_triangles (n : ℕ) : ℕ := n - 2

/-- The number of diagonals drawn from one vertex -/
def num_diagonals (n : ℕ) : ℕ := n - 3

theorem sum_interior_angles_polygon (n : ℕ) (h : n ≥ 3) :
  sum_interior_angles n = (num_triangles n) * 180 :=
by sorry


end NUMINAMATH_CALUDE_sum_interior_angles_polygon_l2361_236181


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2361_236117

theorem polynomial_factorization (a x : ℝ) : 
  a * x^3 + x + a + 1 = (x + 1) * (a * x^2 - a * x + a + 1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2361_236117


namespace NUMINAMATH_CALUDE_expression_evaluation_l2361_236141

theorem expression_evaluation : 1 + 3 + 5 + 7 - (2 + 4 + 6) + 3^2 + 5^2 = 38 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2361_236141


namespace NUMINAMATH_CALUDE_joe_lift_weight_l2361_236109

theorem joe_lift_weight (first_lift second_lift : ℕ) 
  (total_weight : first_lift + second_lift = 900)
  (lift_relation : 2 * first_lift = second_lift + 300) :
  first_lift = 400 := by
  sorry

end NUMINAMATH_CALUDE_joe_lift_weight_l2361_236109


namespace NUMINAMATH_CALUDE_complex_product_square_l2361_236192

/-- Given complex numbers Q, E, and D, prove that (Q * E * D)² equals 8400 + 8000i -/
theorem complex_product_square (Q E D : ℂ) 
  (hQ : Q = 7 + 3*I) 
  (hE : E = 1 + I) 
  (hD : D = 7 - 3*I) : 
  (Q * E * D)^2 = 8400 + 8000*I := by
  sorry

end NUMINAMATH_CALUDE_complex_product_square_l2361_236192


namespace NUMINAMATH_CALUDE_factorial_equation_l2361_236169

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem factorial_equation (n : ℕ) : factorial 6 / factorial (6 - n) = 120 → n = 3 := by
  sorry

end NUMINAMATH_CALUDE_factorial_equation_l2361_236169


namespace NUMINAMATH_CALUDE_max_containers_proof_l2361_236182

def oatmeal_cookies : ℕ := 50
def chocolate_chip_cookies : ℕ := 75
def sugar_cookies : ℕ := 36

theorem max_containers_proof :
  let gcd := Nat.gcd oatmeal_cookies (Nat.gcd chocolate_chip_cookies sugar_cookies)
  (sugar_cookies / gcd) = 7 ∧ 
  (oatmeal_cookies / gcd) ≥ 7 ∧ 
  (chocolate_chip_cookies / gcd) ≥ 7 :=
by sorry

end NUMINAMATH_CALUDE_max_containers_proof_l2361_236182


namespace NUMINAMATH_CALUDE_min_triangles_to_cover_l2361_236190

/-- The minimum number of small equilateral triangles needed to cover a large equilateral triangle -/
theorem min_triangles_to_cover (small_side : ℝ) (large_side : ℝ) : 
  small_side = 2 →
  large_side = 16 →
  (large_side / small_side) ^ 2 = 64 :=
by sorry

end NUMINAMATH_CALUDE_min_triangles_to_cover_l2361_236190


namespace NUMINAMATH_CALUDE_centroid_construction_condition_l2361_236163

/-- A function that checks if a number is divisible by all prime factors of another number -/
def isDivisibleByAllPrimeFactors (m n : ℕ) : Prop :=
  ∀ p : ℕ, p.Prime → p ∣ n → p ∣ m

/-- The main theorem stating the condition for constructing the centroid -/
theorem centroid_construction_condition (n m : ℕ) (h : n ≥ 3) :
  (∃ (construction : Unit), True) ↔ (2 ∣ m ∧ isDivisibleByAllPrimeFactors m n) :=
sorry

end NUMINAMATH_CALUDE_centroid_construction_condition_l2361_236163


namespace NUMINAMATH_CALUDE_pineapple_cost_theorem_l2361_236159

/-- The cost of each pineapple before shipping -/
def pineapple_cost_before_shipping (n : ℕ) (shipping_cost total_cost_per_pineapple : ℚ) : ℚ :=
  total_cost_per_pineapple - (shipping_cost / n)

/-- Theorem: The cost of each pineapple before shipping is $1.25 -/
theorem pineapple_cost_theorem (n : ℕ) (shipping_cost total_cost_per_pineapple : ℚ) 
  (h1 : n = 12)
  (h2 : shipping_cost = 21)
  (h3 : total_cost_per_pineapple = 3) :
  pineapple_cost_before_shipping n shipping_cost total_cost_per_pineapple = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_pineapple_cost_theorem_l2361_236159


namespace NUMINAMATH_CALUDE_complex_modulus_sqrt_two_l2361_236154

theorem complex_modulus_sqrt_two (z : ℂ) (h : z * (1 + Complex.I) = 2 * Complex.I) :
  Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_sqrt_two_l2361_236154


namespace NUMINAMATH_CALUDE_q_investment_time_l2361_236113

/-- Represents a partner in the investment problem -/
structure Partner where
  investment : ℝ
  time : ℝ
  profit : ℝ

/-- The investment problem setup -/
def InvestmentProblem (p q : Partner) : Prop :=
  p.investment / q.investment = 7 / 5 ∧
  p.profit / q.profit = 7 / 9 ∧
  p.time = 5 ∧
  p.investment * p.time / (q.investment * q.time) = p.profit / q.profit

/-- Theorem stating that Q's investment time is 9 months -/
theorem q_investment_time (p q : Partner) 
  (h : InvestmentProblem p q) : q.time = 9 := by
  sorry

end NUMINAMATH_CALUDE_q_investment_time_l2361_236113


namespace NUMINAMATH_CALUDE_max_leap_years_in_period_l2361_236189

/-- A calendrical system where leap years occur every five years -/
structure CalendarSystem where
  leap_year_interval : ℕ
  leap_year_interval_eq : leap_year_interval = 5

/-- The number of years in the period we're considering -/
def period : ℕ := 200

/-- The maximum number of leap years in the given period -/
def max_leap_years (c : CalendarSystem) : ℕ := period / c.leap_year_interval

/-- Theorem: The maximum number of leap years in a 200-year period is 40 -/
theorem max_leap_years_in_period (c : CalendarSystem) : max_leap_years c = 40 := by
  sorry

end NUMINAMATH_CALUDE_max_leap_years_in_period_l2361_236189


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2361_236173

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  (∀ n, a (n + 1) = 2 * a n) →  -- common ratio is 2
  a 2 + a 3 = 4 →               -- given condition
  a 1 + a 4 = 6 :=              -- conclusion to prove
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2361_236173


namespace NUMINAMATH_CALUDE_cos_two_theta_value_l2361_236115

theorem cos_two_theta_value (θ : Real) 
  (h : Real.exp (Real.log 2 * (-5/2 + 2 * Real.cos θ)) + 1 = Real.exp (Real.log 2 * (3/4 + Real.cos θ))) : 
  Real.cos (2 * θ) = 17/8 := by
sorry

end NUMINAMATH_CALUDE_cos_two_theta_value_l2361_236115


namespace NUMINAMATH_CALUDE_sqrt_50_between_consecutive_integers_l2361_236130

theorem sqrt_50_between_consecutive_integers : ∃ (n : ℕ), n > 0 ∧ n^2 < 50 ∧ (n+1)^2 > 50 ∧ n * (n+1) = 56 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_50_between_consecutive_integers_l2361_236130


namespace NUMINAMATH_CALUDE_xyz_sum_lower_bound_l2361_236171

theorem xyz_sum_lower_bound (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0)
  (h : x * y * z + x * y + y * z + z * x = 4) : 
  x + y + z ≥ 3 := by
sorry

end NUMINAMATH_CALUDE_xyz_sum_lower_bound_l2361_236171


namespace NUMINAMATH_CALUDE_smallest_percent_increase_l2361_236165

def question_value : Fin 15 → ℕ
  | 0 => 100
  | 1 => 300
  | 2 => 600
  | 3 => 800
  | 4 => 1500
  | 5 => 3000
  | 6 => 4500
  | 7 => 7000
  | 8 => 10000
  | 9 => 15000
  | 10 => 30000
  | 11 => 45000
  | 12 => 75000
  | 13 => 150000
  | 14 => 300000

def percent_increase (a b : ℕ) : ℚ :=
  (b - a : ℚ) / a * 100

def options : List (Fin 15 × Fin 15) :=
  [(1, 2), (3, 4), (6, 7), (11, 12), (13, 14)]

theorem smallest_percent_increase :
  ∀ (pair : Fin 15 × Fin 15),
    pair ∈ options →
    percent_increase (question_value pair.1) (question_value pair.2) ≥ 
    percent_increase (question_value 6) (question_value 7) :=
by sorry

end NUMINAMATH_CALUDE_smallest_percent_increase_l2361_236165


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l2361_236149

theorem fraction_sum_equality : 
  (1 : ℚ) / 15 + (2 : ℚ) / 25 + (3 : ℚ) / 35 + (4 : ℚ) / 45 = (506 : ℚ) / 1575 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l2361_236149


namespace NUMINAMATH_CALUDE_problem_statement_l2361_236103

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  -- Statement 1
  (a^2 - b^2 = 1 → a - b < 1) ∧
  -- Statement 2
  (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 1/b - 1/a = 1 ∧ a - b ≥ 1) ∧
  -- Statement 3
  (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ |Real.sqrt a - Real.sqrt b| = 1 ∧ |a - b| ≥ 1) ∧
  -- Statement 4
  (|a^3 - b^3| = 1 → |a - b| < 1) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l2361_236103


namespace NUMINAMATH_CALUDE_rectangle_geometric_mean_l2361_236100

/-- Given a rectangle with side lengths a and b, where b is the geometric mean
    of a and the perimeter, prove that b = a + a√3 -/
theorem rectangle_geometric_mean (a b : ℝ) (h_pos : 0 < a) :
  b^2 = a * (2*a + 2*b) → b = a + a * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_rectangle_geometric_mean_l2361_236100


namespace NUMINAMATH_CALUDE_smallest_average_of_valid_pair_l2361_236139

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Predicate for two numbers differing by 2 and having sum of digits divisible by 4 -/
def validPair (n m : ℕ) : Prop :=
  m = n + 2 ∧ (sumOfDigits n + sumOfDigits m) % 4 = 0

theorem smallest_average_of_valid_pair :
  ∃ (n m : ℕ), validPair n m ∧
  ∀ (k l : ℕ), validPair k l → (n + m : ℚ) / 2 ≤ (k + l : ℚ) / 2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_average_of_valid_pair_l2361_236139


namespace NUMINAMATH_CALUDE_set_7_24_25_is_pythagorean_triple_l2361_236194

/-- A Pythagorean triple is a set of three positive integers a, b, and c that satisfy a² + b² = c² -/
def is_pythagorean_triple (a b c : ℕ) : Prop := a * a + b * b = c * c

/-- The set (7, 24, 25) is a Pythagorean triple -/
theorem set_7_24_25_is_pythagorean_triple : is_pythagorean_triple 7 24 25 := by
  sorry

end NUMINAMATH_CALUDE_set_7_24_25_is_pythagorean_triple_l2361_236194


namespace NUMINAMATH_CALUDE_bob_daily_earnings_l2361_236102

/-- Proves that Bob makes $4 per day given the conditions of the problem -/
theorem bob_daily_earnings (sally_earnings : ℝ) (total_savings : ℝ) (days_in_year : ℕ) :
  sally_earnings = 6 →
  total_savings = 1825 →
  days_in_year = 365 →
  ∃ (bob_earnings : ℝ),
    bob_earnings = 4 ∧
    (sally_earnings / 2 + bob_earnings / 2) * days_in_year = total_savings :=
by sorry

end NUMINAMATH_CALUDE_bob_daily_earnings_l2361_236102


namespace NUMINAMATH_CALUDE_weight_sum_l2361_236129

/-- Given the weights of four people (a, b, c, d) in pairs,
    prove that the sum of the weights of the first and last person is 310 pounds. -/
theorem weight_sum (a b c d : ℝ) 
  (h1 : a + b = 280) 
  (h2 : b + c = 230) 
  (h3 : c + d = 260) : 
  a + d = 310 := by
  sorry

end NUMINAMATH_CALUDE_weight_sum_l2361_236129


namespace NUMINAMATH_CALUDE_jam_solution_l2361_236137

/-- Represents the amount and consumption rate of jam for a person -/
structure JamConsumption where
  amount : ℝ
  rate : ℝ

/-- The problem of determining jam consumption for Ponchik and Syropchik -/
def jam_problem (ponchik : JamConsumption) (syropchik : JamConsumption) : Prop :=
  -- Total amount of jam
  ponchik.amount + syropchik.amount = 100 ∧
  -- Same time to consume their own supplies
  ponchik.amount / ponchik.rate = syropchik.amount / syropchik.rate ∧
  -- Ponchik's consumption time if he had Syropchik's amount
  syropchik.amount / ponchik.rate = 45 ∧
  -- Syropchik's consumption time if he had Ponchik's amount
  ponchik.amount / syropchik.rate = 20

/-- The solution to the jam consumption problem -/
theorem jam_solution :
  ∃ (ponchik syropchik : JamConsumption),
    jam_problem ponchik syropchik ∧
    ponchik.amount = 40 ∧
    ponchik.rate = 4/3 ∧
    syropchik.amount = 60 ∧
    syropchik.rate = 2 :=
by sorry

end NUMINAMATH_CALUDE_jam_solution_l2361_236137


namespace NUMINAMATH_CALUDE_fixed_point_theorem_l2361_236175

/-- The ellipse C -/
def C (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

/-- The right focus F of the ellipse C -/
def F : ℝ × ℝ := (1, 0)

/-- The line L -/
def L (x : ℝ) : Prop := x = 4

/-- The left vertex A of the ellipse C -/
def A : ℝ × ℝ := (-2, 0)

/-- The ratio condition for any point P on C -/
def ratio_condition (P : ℝ × ℝ) : Prop :=
  C P.1 P.2 → 2 * Real.sqrt ((P.1 - F.1)^2 + (P.2 - F.2)^2) = |P.1 - 4|

/-- The theorem to be proved -/
theorem fixed_point_theorem :
  ∀ D E M N : ℝ × ℝ,
  (∃ t : ℝ, C (F.1 + t * (D.1 - F.1)) (F.2 + t * (D.2 - F.2))) →
  (∃ t : ℝ, C (F.1 + t * (E.1 - F.1)) (F.2 + t * (E.2 - F.2))) →
  (∃ t : ℝ, M = (4, A.2 + t * (D.2 - A.2))) →
  (∃ t : ℝ, N = (4, A.2 + t * (E.2 - A.2))) →
  (∀ P : ℝ × ℝ, ratio_condition P) →
  ∃ O : ℝ × ℝ, O = (1, 0) ∧ 
    (M.1 - O.1)^2 + (M.2 - O.2)^2 = (N.1 - O.1)^2 + (N.2 - O.2)^2 :=
by
  sorry

end NUMINAMATH_CALUDE_fixed_point_theorem_l2361_236175


namespace NUMINAMATH_CALUDE_hyperbola_ellipse_theorem_l2361_236161

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/2 = 1

-- Define the ellipse C
def ellipse_C (x y a b : ℝ) : Prop := x^2/a^2 + y^2/b^2 = 1

-- Define the line l
def line_l (x y k : ℝ) : Prop := y = k * (x - 1)

-- Define the complementary angles condition
def complementary_angles (x_A y_A x_B y_B m : ℝ) : Prop :=
  (y_A / (x_A - m)) + (y_B / (x_B - m)) = 0

theorem hyperbola_ellipse_theorem :
  ∀ (a b : ℝ), a > b ∧ b > 0 →
  (∃ (x_F y_F : ℝ), hyperbola x_F y_F ∧ 
    (∀ (x y : ℝ), ellipse_C x y a b ↔ 
      x^2/3 + y^2/2 = 1)) ∧
  (∃ (k x_A y_A x_B y_B : ℝ), k ≠ 0 ∧
    line_l x_A y_A k ∧ line_l x_B y_B k ∧
    ellipse_C x_A y_A 3 2 ∧ ellipse_C x_B y_B 3 2 ∧
    complementary_angles x_A y_A x_B y_B 3) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_ellipse_theorem_l2361_236161


namespace NUMINAMATH_CALUDE_smallest_spend_l2361_236127

/-- Represents a gift set with its composition and price -/
structure GiftSet where
  chocolates : ℕ
  caramels : ℕ
  price : ℕ

/-- The first type of gift set -/
def gift1 : GiftSet := { chocolates := 3, caramels := 15, price := 350 }

/-- The second type of gift set -/
def gift2 : GiftSet := { chocolates := 20, caramels := 5, price := 500 }

/-- Calculates the total cost of buying gift sets -/
def totalCost (m n : ℕ) : ℕ := m * gift1.price + n * gift2.price

/-- Calculates the total number of chocolate candies -/
def totalChocolates (m n : ℕ) : ℕ := m * gift1.chocolates + n * gift2.chocolates

/-- Calculates the total number of caramel candies -/
def totalCaramels (m n : ℕ) : ℕ := m * gift1.caramels + n * gift2.caramels

/-- Theorem stating the smallest non-zero amount Eugene needs to spend -/
theorem smallest_spend : 
  ∃ m n : ℕ, m + n > 0 ∧ 
    totalChocolates m n = totalCaramels m n ∧
    totalCost m n = 3750 ∧
    ∀ k l : ℕ, k + l > 0 → 
      totalChocolates k l = totalCaramels k l → 
      totalCost k l ≥ 3750 := by sorry

end NUMINAMATH_CALUDE_smallest_spend_l2361_236127

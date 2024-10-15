import Mathlib

namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_set_l1221_122172

theorem absolute_value_equation_solution_set :
  {x : ℝ | |x / (x - 1)| = x / (x - 1)} = {x : ℝ | x ≤ 0 ∨ x > 1} := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_set_l1221_122172


namespace NUMINAMATH_CALUDE_tetrahedron_special_points_l1221_122162

-- Define the tetrahedron P-ABC
structure Tetrahedron :=
  (P A B C : EuclideanSpace ℝ (Fin 3))

-- Define the projection O of P onto the base plane ABC
def projection (t : Tetrahedron) : EuclideanSpace ℝ (Fin 3) := sorry

-- Define the property of equal angles between lateral edges and base plane
def equal_lateral_base_angles (t : Tetrahedron) : Prop := sorry

-- Define the property of mutually perpendicular lateral edges
def perpendicular_lateral_edges (t : Tetrahedron) : Prop := sorry

-- Define the property of equal angles between side faces and base plane
def equal_face_base_angles (t : Tetrahedron) : Prop := sorry

-- Define the circumcenter of a triangle
def is_circumcenter (O A B C : EuclideanSpace ℝ (Fin 3)) : Prop := sorry

-- Define the orthocenter of a triangle
def is_orthocenter (O A B C : EuclideanSpace ℝ (Fin 3)) : Prop := sorry

-- Define the incenter of a triangle
def is_incenter (O A B C : EuclideanSpace ℝ (Fin 3)) : Prop := sorry

-- Theorem statements
theorem tetrahedron_special_points (t : Tetrahedron) :
  (equal_lateral_base_angles t → is_circumcenter (projection t) t.A t.B t.C) ∧
  (perpendicular_lateral_edges t → is_orthocenter (projection t) t.A t.B t.C) ∧
  (equal_face_base_angles t → is_incenter (projection t) t.A t.B t.C) := by sorry

end NUMINAMATH_CALUDE_tetrahedron_special_points_l1221_122162


namespace NUMINAMATH_CALUDE_root_in_interval_l1221_122164

def f (x : ℝ) := 3*x + x - 3

theorem root_in_interval : ∃ x ∈ Set.Ioo 0 1, f x = 0 := by sorry

end NUMINAMATH_CALUDE_root_in_interval_l1221_122164


namespace NUMINAMATH_CALUDE_rectangular_paper_to_hexagon_l1221_122171

/-- A rectangular sheet of paper with sides a and b can be folded into a regular hexagon
    if and only if the aspect ratio b/a is between 1/2 and 2. -/
theorem rectangular_paper_to_hexagon (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x : ℝ), x > 0 ∧ x < a ∧ x < b ∧ (a - x)^2 + (b - x)^2 = x^2) ↔
  (1/2 < b/a ∧ b/a < 2) :=
sorry

end NUMINAMATH_CALUDE_rectangular_paper_to_hexagon_l1221_122171


namespace NUMINAMATH_CALUDE_photo_border_area_l1221_122163

/-- The area of the border around a rectangular photograph -/
theorem photo_border_area (photo_height photo_width border_width : ℝ) 
  (h_height : photo_height = 9)
  (h_width : photo_width = 12)
  (h_border : border_width = 3) : 
  (photo_height + 2 * border_width) * (photo_width + 2 * border_width) - 
  photo_height * photo_width = 162 := by
  sorry

#check photo_border_area

end NUMINAMATH_CALUDE_photo_border_area_l1221_122163


namespace NUMINAMATH_CALUDE_meaningful_sqrt_range_l1221_122115

theorem meaningful_sqrt_range (x : ℝ) : 
  (∃ y : ℝ, y = Real.sqrt (2 / (x - 1))) → x > 1 := by
sorry

end NUMINAMATH_CALUDE_meaningful_sqrt_range_l1221_122115


namespace NUMINAMATH_CALUDE_gold_silver_board_theorem_l1221_122125

/-- A board configuration with gold and silver cells -/
structure Board :=
  (size : Nat)
  (is_gold : Fin size → Fin size → Bool)

/-- Count gold cells in a rectangle -/
def count_gold (b : Board) (x y w h : Nat) : Nat :=
  (Finset.range w).sum (λ i =>
    (Finset.range h).sum (λ j =>
      if b.is_gold ⟨x + i, sorry⟩ ⟨y + j, sorry⟩ then 1 else 0))

/-- Property that each 3x3 square has A gold cells -/
def three_by_three_property (b : Board) (A : Nat) : Prop :=
  ∀ x y, x + 3 ≤ b.size → y + 3 ≤ b.size →
    count_gold b x y 3 3 = A

/-- Property that each 2x4 or 4x2 rectangle has Z gold cells -/
def two_by_four_property (b : Board) (Z : Nat) : Prop :=
  (∀ x y, x + 2 ≤ b.size → y + 4 ≤ b.size →
    count_gold b x y 2 4 = Z) ∧
  (∀ x y, x + 4 ≤ b.size → y + 2 ≤ b.size →
    count_gold b x y 4 2 = Z)

/-- The main theorem -/
theorem gold_silver_board_theorem :
  ∀ (b : Board) (A Z : Nat),
    b.size = 2016 →
    three_by_three_property b A →
    two_by_four_property b Z →
    ((A = 0 ∧ Z = 0) ∨ (A = 9 ∧ Z = 8)) :=
sorry

end NUMINAMATH_CALUDE_gold_silver_board_theorem_l1221_122125


namespace NUMINAMATH_CALUDE_coefficient_properties_l1221_122168

-- Define the polynomial coefficients
variable (a : Fin 7 → ℝ)

-- Define the given equation
def equation (x : ℝ) : Prop :=
  (1 + x)^6 = a 0 + a 1 * (1 - x) + a 2 * (1 - x)^2 + a 3 * (1 - x)^3 + 
              a 4 * (1 - x)^4 + a 5 * (1 - x)^5 + a 6 * (1 - x)^6

-- State the theorem
theorem coefficient_properties (a : Fin 7 → ℝ) 
  (h : ∀ x, equation a x) : 
  a 6 = 1 ∧ a 1 + a 3 + a 5 = -364 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_properties_l1221_122168


namespace NUMINAMATH_CALUDE_king_requirement_requirement_met_for_6_requirement_not_met_for_1986_l1221_122197

/-- Represents a network of cities and roads -/
structure CityNetwork where
  n : ℕ                -- number of cities
  roads : ℕ             -- number of roads
  connected : Prop      -- any city can be reached from any other city
  distances : Finset ℕ  -- set of shortest distances between pairs of cities

/-- The condition for a valid city network -/
def validNetwork (net : CityNetwork) : Prop :=
  net.roads = net.n - 1 ∧
  net.connected ∧
  net.distances = Finset.range (net.n * (net.n - 1) / 2 + 1) \ {0}

/-- The condition for the network to meet the king's requirement -/
def meetsRequirement (n : ℕ) : Prop :=
  ∃ (net : CityNetwork), net.n = n ∧ validNetwork net

/-- The main theorem -/
theorem king_requirement (n : ℕ) :
  meetsRequirement n ↔ (∃ k : ℕ, n = k^2) ∨ (∃ k : ℕ, n = k^2 + 2) :=
sorry

/-- The requirement can be met for n = 6 -/
theorem requirement_met_for_6 : meetsRequirement 6 :=
sorry

/-- The requirement cannot be met for n = 1986 -/
theorem requirement_not_met_for_1986 : ¬meetsRequirement 1986 :=
sorry

end NUMINAMATH_CALUDE_king_requirement_requirement_met_for_6_requirement_not_met_for_1986_l1221_122197


namespace NUMINAMATH_CALUDE_f_of_g_5_l1221_122198

def g (x : ℝ) : ℝ := 4 * x + 9

def f (x : ℝ) : ℝ := 6 * x - 11

theorem f_of_g_5 : f (g 5) = 163 := by
  sorry

end NUMINAMATH_CALUDE_f_of_g_5_l1221_122198


namespace NUMINAMATH_CALUDE_cosine_identity_l1221_122109

theorem cosine_identity (z : ℂ) (α : ℝ) (h : z + 1/z = 2 * Real.cos α) :
  ∀ n : ℕ, z^n + 1/z^n = 2 * Real.cos (n * α) := by
  sorry

end NUMINAMATH_CALUDE_cosine_identity_l1221_122109


namespace NUMINAMATH_CALUDE_fraction_sum_equals_one_l1221_122110

theorem fraction_sum_equals_one (a : ℝ) (h : a ≠ -1) :
  (1 : ℝ) / (a + 1) + a / (a + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_one_l1221_122110


namespace NUMINAMATH_CALUDE_addSecondsCorrect_l1221_122165

-- Define a structure for time
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat

-- Define a function to add seconds to a time
def addSeconds (t : Time) (s : Nat) : Time :=
  sorry

-- Define the initial time
def initialTime : Time :=
  { hours := 7, minutes := 45, seconds := 0 }

-- Define the number of seconds to add
def secondsToAdd : Nat := 9999

-- Theorem to prove
theorem addSecondsCorrect : 
  addSeconds initialTime secondsToAdd = { hours := 10, minutes := 31, seconds := 39 } :=
sorry

end NUMINAMATH_CALUDE_addSecondsCorrect_l1221_122165


namespace NUMINAMATH_CALUDE_right_triangle_acute_angles_l1221_122113

theorem right_triangle_acute_angles (α β : Real) : 
  -- Conditions
  α + β = 90 →  -- Sum of acute angles in a right triangle is 90°
  α = 40 →      -- One acute angle is 40°
  -- Conclusion
  β = 50 :=     -- The other acute angle is 50°
by sorry

end NUMINAMATH_CALUDE_right_triangle_acute_angles_l1221_122113


namespace NUMINAMATH_CALUDE_line_relationships_l1221_122166

-- Define a type for lines in 3D space
structure Line3D where
  -- Add necessary fields to represent a line in 3D space
  -- This is a simplified representation
  dummy : Unit

-- Define perpendicularity for lines
def perpendicular (l1 l2 : Line3D) : Prop := sorry

-- Define parallelism for lines
def parallel (l1 l2 : Line3D) : Prop := sorry

theorem line_relationships (a b c : Line3D) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) : 
  (¬ ∀ (a b c : Line3D), perpendicular a b → perpendicular a c → parallel b c) ∧ 
  (¬ ∀ (a b c : Line3D), perpendicular a b → perpendicular a c → perpendicular b c) ∧
  (∀ (a b c : Line3D), parallel a b → perpendicular b c → perpendicular a c) := by
  sorry

end NUMINAMATH_CALUDE_line_relationships_l1221_122166


namespace NUMINAMATH_CALUDE_subsets_with_adjacent_chairs_12_l1221_122128

/-- The number of subsets with at least three adjacent chairs in a circular arrangement of 12 chairs -/
def subsets_with_adjacent_chairs (n : ℕ) : ℕ :=
  let adjacent_3_to_6 := 4 * n
  let adjacent_7_plus := (Finset.range 6).sum (fun k => Nat.choose n (n - k))
  adjacent_3_to_6 + adjacent_7_plus

/-- Theorem stating that the number of subsets with at least three adjacent chairs
    in a circular arrangement of 12 chairs is 1634 -/
theorem subsets_with_adjacent_chairs_12 :
  subsets_with_adjacent_chairs 12 = 1634 := by
  sorry

end NUMINAMATH_CALUDE_subsets_with_adjacent_chairs_12_l1221_122128


namespace NUMINAMATH_CALUDE_total_new_emails_formula_l1221_122188

/-- Represents the number of new emails received in one deletion cycle -/
def new_emails_per_cycle : ℕ := 15 + 5

/-- Represents the final batch of emails received -/
def final_batch : ℕ := 10

/-- Calculates the total number of new emails after n cycles and a final batch -/
def total_new_emails (n : ℕ) : ℕ := n * new_emails_per_cycle + final_batch

/-- Theorem stating the total number of new emails after n cycles and a final batch -/
theorem total_new_emails_formula (n : ℕ) : 
  total_new_emails n = 20 * n + 10 := by
  sorry

#eval total_new_emails 5  -- Example evaluation

end NUMINAMATH_CALUDE_total_new_emails_formula_l1221_122188


namespace NUMINAMATH_CALUDE_lemonade_recipe_correct_l1221_122135

/-- Represents the ratio of ingredients in the lemonade mixture -/
structure LemonadeRatio where
  water : ℕ
  lemon_juice : ℕ

/-- Converts gallons to quarts -/
def gallons_to_quarts (gallons : ℕ) : ℕ := 4 * gallons

/-- Calculates the amount of each ingredient needed for a given total volume -/
def ingredient_amount (ratio : LemonadeRatio) (total_volume : ℕ) (ingredient : ℕ) : ℕ :=
  (ingredient * total_volume) / (ratio.water + ratio.lemon_juice)

theorem lemonade_recipe_correct (ratio : LemonadeRatio) (total_gallons : ℕ) :
  ratio.water = 5 →
  ratio.lemon_juice = 3 →
  total_gallons = 2 →
  let total_quarts := gallons_to_quarts total_gallons
  ingredient_amount ratio total_quarts ratio.water = 5 ∧
  ingredient_amount ratio total_quarts ratio.lemon_juice = 3 := by
  sorry

end NUMINAMATH_CALUDE_lemonade_recipe_correct_l1221_122135


namespace NUMINAMATH_CALUDE_constant_seq_arithmetic_and_geometric_l1221_122153

/-- A sequence of real numbers -/
def Sequence := ℕ → ℝ

/-- A constant sequence with value a -/
def constantSeq (a : ℝ) : Sequence := λ _ => a

/-- An arithmetic sequence -/
def isArithmetic (s : Sequence) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, s (n + 1) - s n = d

/-- A geometric sequence (allowing zero terms) -/
def isGeometric (s : Sequence) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, s (n + 1) = r * s n

theorem constant_seq_arithmetic_and_geometric (a : ℝ) :
  isArithmetic (constantSeq a) ∧ isGeometric (constantSeq a) := by
  sorry

#check constant_seq_arithmetic_and_geometric

end NUMINAMATH_CALUDE_constant_seq_arithmetic_and_geometric_l1221_122153


namespace NUMINAMATH_CALUDE_initial_books_l1221_122152

theorem initial_books (initial sold bought final : ℕ) : 
  sold = 94 →
  bought = 150 →
  final = 58 →
  initial - sold + bought = final →
  initial = 2 :=
by sorry

end NUMINAMATH_CALUDE_initial_books_l1221_122152


namespace NUMINAMATH_CALUDE_square_area_reduction_l1221_122140

theorem square_area_reduction (S1_area : ℝ) (S1_area_eq : S1_area = 25) : 
  let S1_side := Real.sqrt S1_area
  let S2_side := S1_side / Real.sqrt 2
  let S3_side := S2_side / Real.sqrt 2
  S3_side ^ 2 = 6.25 := by
  sorry

end NUMINAMATH_CALUDE_square_area_reduction_l1221_122140


namespace NUMINAMATH_CALUDE_solve_percentage_problem_l1221_122132

theorem solve_percentage_problem (x : ℝ) : (0.7 * x = (1/3) * x + 110) → x = 300 := by
  sorry

end NUMINAMATH_CALUDE_solve_percentage_problem_l1221_122132


namespace NUMINAMATH_CALUDE_bankers_gain_l1221_122190

/-- Calculate the banker's gain given present worth, interest rate, and time period -/
theorem bankers_gain (present_worth : ℝ) (interest_rate : ℝ) (time_period : ℕ) : 
  present_worth = 600 → 
  interest_rate = 0.1 → 
  time_period = 2 → 
  present_worth * (1 + interest_rate) ^ time_period - present_worth = 126 := by
sorry

end NUMINAMATH_CALUDE_bankers_gain_l1221_122190


namespace NUMINAMATH_CALUDE_goldbach_conjecture_negation_l1221_122159

-- Define the Goldbach Conjecture
def goldbach_conjecture : Prop :=
  ∀ n : ℕ, n > 2 → Even n → ∃ p q : ℕ, Prime p ∧ Prime q ∧ n = p + q

-- State the theorem
theorem goldbach_conjecture_negation :
  ¬goldbach_conjecture ↔ ∃ n : ℕ, n > 2 ∧ Even n ∧ ¬∃ p q : ℕ, Prime p ∧ Prime q ∧ n = p + q :=
by
  sorry

end NUMINAMATH_CALUDE_goldbach_conjecture_negation_l1221_122159


namespace NUMINAMATH_CALUDE_domino_swap_incorrect_l1221_122154

/-- Represents a domino with a value from 0 to 9 -/
def Domino : Type := Fin 10

/-- Represents a multiplication problem with 5 dominoes -/
structure DominoMultiplication :=
  (d1 d2 d3 d4 d5 : Domino)

/-- Checks if the domino multiplication is correct -/
def isCorrectMultiplication (dm : DominoMultiplication) : Prop :=
  (dm.d1.val * 10 + dm.d2.val) * dm.d3.val = dm.d4.val * 10 + dm.d5.val

/-- Swaps two dominoes in the multiplication -/
def swapDominoes (dm : DominoMultiplication) (i j : Fin 5) : DominoMultiplication :=
  match i, j with
  | 0, 1 => { d1 := dm.d2, d2 := dm.d1, d3 := dm.d3, d4 := dm.d4, d5 := dm.d5 }
  | 0, 2 => { d1 := dm.d3, d2 := dm.d2, d3 := dm.d1, d4 := dm.d4, d5 := dm.d5 }
  | 0, 3 => { d1 := dm.d4, d2 := dm.d2, d3 := dm.d3, d4 := dm.d1, d5 := dm.d5 }
  | 0, 4 => { d1 := dm.d5, d2 := dm.d2, d3 := dm.d3, d4 := dm.d4, d5 := dm.d1 }
  | 1, 2 => { d1 := dm.d1, d2 := dm.d3, d3 := dm.d2, d4 := dm.d4, d5 := dm.d5 }
  | 1, 3 => { d1 := dm.d1, d2 := dm.d4, d3 := dm.d3, d4 := dm.d2, d5 := dm.d5 }
  | 1, 4 => { d1 := dm.d1, d2 := dm.d5, d3 := dm.d3, d4 := dm.d4, d5 := dm.d2 }
  | 2, 3 => { d1 := dm.d1, d2 := dm.d2, d3 := dm.d4, d4 := dm.d3, d5 := dm.d5 }
  | 2, 4 => { d1 := dm.d1, d2 := dm.d2, d3 := dm.d5, d4 := dm.d4, d5 := dm.d3 }
  | 3, 4 => { d1 := dm.d1, d2 := dm.d2, d3 := dm.d3, d4 := dm.d5, d5 := dm.d4 }
  | _, _ => dm  -- For any other combination, return the original multiplication

theorem domino_swap_incorrect
  (dm : DominoMultiplication)
  (h : isCorrectMultiplication dm)
  (i j : Fin 5)
  (hne : i ≠ j) :
  ¬(isCorrectMultiplication (swapDominoes dm i j)) :=
by sorry

end NUMINAMATH_CALUDE_domino_swap_incorrect_l1221_122154


namespace NUMINAMATH_CALUDE_age_ratio_proof_l1221_122170

/-- Given three people a, b, and c, prove that the ratio of b's age to c's age is 2:1 -/
theorem age_ratio_proof (a b c : ℕ) : 
  a = b + 2 →  -- a is two years older than b
  a + b + c = 22 →  -- sum of ages is 22
  b = 8 →  -- b is 8 years old
  b = 2 * c  -- ratio of b's age to c's age is 2:1
:= by sorry

end NUMINAMATH_CALUDE_age_ratio_proof_l1221_122170


namespace NUMINAMATH_CALUDE_peters_remaining_money_l1221_122167

/-- Peter's shopping trip to the market -/
theorem peters_remaining_money 
  (initial_amount : ℕ) 
  (potato_price potato_quantity : ℕ)
  (tomato_price tomato_quantity : ℕ)
  (cucumber_price cucumber_quantity : ℕ)
  (banana_price banana_quantity : ℕ)
  (h1 : initial_amount = 500)
  (h2 : potato_price = 2 ∧ potato_quantity = 6)
  (h3 : tomato_price = 3 ∧ tomato_quantity = 9)
  (h4 : cucumber_price = 4 ∧ cucumber_quantity = 5)
  (h5 : banana_price = 5 ∧ banana_quantity = 3) :
  initial_amount - 
  (potato_price * potato_quantity + 
   tomato_price * tomato_quantity + 
   cucumber_price * cucumber_quantity + 
   banana_price * banana_quantity) = 426 := by
  sorry

end NUMINAMATH_CALUDE_peters_remaining_money_l1221_122167


namespace NUMINAMATH_CALUDE_tangent_line_y_intercept_l1221_122187

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a line tangent to two circles -/
structure TangentLine where
  circle1 : Circle
  circle2 : Circle
  yIntercept : ℝ

/-- Checks if a point is in the first quadrant -/
def isFirstQuadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 > 0

/-- The main theorem -/
theorem tangent_line_y_intercept :
  ∀ (l : TangentLine),
    l.circle1 = { center := (3, 0), radius := 3 } →
    l.circle2 = { center := (7, 0), radius := 2 } →
    (∃ (p1 p2 : ℝ × ℝ), isFirstQuadrant p1 ∧ isFirstQuadrant p2) →
    l.yIntercept = 24 * Real.sqrt 55 / 55 := by
  sorry


end NUMINAMATH_CALUDE_tangent_line_y_intercept_l1221_122187


namespace NUMINAMATH_CALUDE_units_digit_of_six_to_seven_l1221_122120

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- Theorem: The units digit of 6^7 is 6 -/
theorem units_digit_of_six_to_seven :
  unitsDigit (6^7) = 6 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_six_to_seven_l1221_122120


namespace NUMINAMATH_CALUDE_smallest_square_from_smaller_squares_l1221_122189

theorem smallest_square_from_smaller_squares :
  ∀ n : ℕ,
  (∃ a : ℕ, a * a = n * (1 * 1 + 2 * 2 + 3 * 3)) →
  (∀ m : ℕ, m < n → ¬∃ b : ℕ, b * b = m * (1 * 1 + 2 * 2 + 3 * 3)) →
  n = 14 :=
by sorry

end NUMINAMATH_CALUDE_smallest_square_from_smaller_squares_l1221_122189


namespace NUMINAMATH_CALUDE_root_implies_b_value_l1221_122105

theorem root_implies_b_value (a b : ℚ) :
  (2 + Real.sqrt 5 : ℝ) ^ 3 + a * (2 + Real.sqrt 5 : ℝ) ^ 2 + b * (2 + Real.sqrt 5 : ℝ) - 20 = 0 →
  b = -24 := by
sorry

end NUMINAMATH_CALUDE_root_implies_b_value_l1221_122105


namespace NUMINAMATH_CALUDE_j_percentage_less_than_p_l1221_122145

/-- Given t = 6.25, t is t% less than p, and j is 20% less than t, prove j is 25% less than p -/
theorem j_percentage_less_than_p (t p j : ℝ) : 
  t = 6.25 →
  t = p * (100 - t) / 100 →
  j = t * 0.8 →
  j = p * 0.75 := by
  sorry

end NUMINAMATH_CALUDE_j_percentage_less_than_p_l1221_122145


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l1221_122160

theorem smallest_n_congruence (n : ℕ) : 
  (n > 0 ∧ 19 * n ≡ 567 [ZMOD 9]) → n ≥ 9 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l1221_122160


namespace NUMINAMATH_CALUDE_graduating_class_boys_count_l1221_122151

theorem graduating_class_boys_count (total : ℕ) (diff : ℕ) (boys : ℕ) : 
  total = 466 →
  diff = 212 →
  boys + (boys + diff) = total →
  boys = 127 := by
sorry

end NUMINAMATH_CALUDE_graduating_class_boys_count_l1221_122151


namespace NUMINAMATH_CALUDE_digit_configuration_impossible_l1221_122121

/-- Represents a configuration of digits on a shape with 6 segments -/
structure DigitConfiguration :=
  (digits : Finset ℕ)
  (segments : Finset (Finset ℕ))

/-- The property that all segments have the same sum -/
def has_equal_segment_sums (config : DigitConfiguration) : Prop :=
  ∃ (sum : ℕ), ∀ segment ∈ config.segments, (segment.sum id = sum)

/-- The main theorem stating the impossibility of the configuration -/
theorem digit_configuration_impossible : 
  ¬ ∃ (config : DigitConfiguration), 
    (config.digits = Finset.range 10) ∧ 
    (config.segments.card = 6) ∧
    (∀ segment ∈ config.segments, segment.card = 3) ∧
    (has_equal_segment_sums config) :=
sorry

end NUMINAMATH_CALUDE_digit_configuration_impossible_l1221_122121


namespace NUMINAMATH_CALUDE_perfect_square_difference_l1221_122192

theorem perfect_square_difference : ∃ (x a b : ℤ), 
  (x + 100 = a^2) ∧ 
  (x + 164 = b^2) ∧ 
  (x = 125 ∨ x = -64 ∨ x = -100) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_difference_l1221_122192


namespace NUMINAMATH_CALUDE_sqrt_sum_condition_l1221_122196

/-- For distinct positive numbers a, b, c that are not perfect squares,
    √a + √b = √c holds if and only if 2√(ab) = c - (a + b) and ab is a perfect square -/
theorem sqrt_sum_condition (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c)
  (hna : ¬ ∃ (n : ℕ), a = n^2)
  (hnb : ¬ ∃ (n : ℕ), b = n^2)
  (hnc : ¬ ∃ (n : ℕ), c = n^2) :
  (Real.sqrt a + Real.sqrt b = Real.sqrt c) ↔ 
  (2 * Real.sqrt (a * b) = c - (a + b) ∧ ∃ (n : ℕ), a * b = n^2) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_sum_condition_l1221_122196


namespace NUMINAMATH_CALUDE_polygon_sides_l1221_122103

/-- A polygon has n sides if its interior angles sum is 4 times its exterior angles sum -/
theorem polygon_sides (n : ℕ) : n = 10 :=
  by
  -- Define the sum of interior angles
  let interior_sum := (n - 2) * 180
  -- Define the sum of exterior angles
  let exterior_sum := 360
  -- State the condition that interior sum is 4 times exterior sum
  have h : interior_sum = 4 * exterior_sum := by sorry
  -- Prove that n = 10
  sorry


end NUMINAMATH_CALUDE_polygon_sides_l1221_122103


namespace NUMINAMATH_CALUDE_christophers_to_gabrielas_age_ratio_l1221_122102

/-- Proves that the ratio of Christopher's age to Gabriela's age is 2:1 given the conditions -/
theorem christophers_to_gabrielas_age_ratio :
  ∀ (c g : ℕ),
  c = 24 →  -- Christopher is now 24 years old
  c - 9 = 5 * (g - 9) →  -- Nine years ago, Christopher was 5 times as old as Gabriela
  c / g = 2 :=  -- The ratio of Christopher's age to Gabriela's age is 2:1
by
  sorry

#check christophers_to_gabrielas_age_ratio

end NUMINAMATH_CALUDE_christophers_to_gabrielas_age_ratio_l1221_122102


namespace NUMINAMATH_CALUDE_quadratic_intersection_l1221_122176

theorem quadratic_intersection
  (a b c d : ℝ)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hcd : c ≠ d) :
  let f (x : ℝ) := a * x^2 + b * x + c
  let g (x : ℝ) := a * x^2 - b * x + d
  let x_intersect := (d - c) / (2 * b)
  let y_intersect := (a * (d - c)^2) / (4 * b^2) + (d + c) / 2
  ∃ (x y : ℝ), f x = g x ∧ f x = y ∧ x = x_intersect ∧ y = y_intersect :=
by
  sorry


end NUMINAMATH_CALUDE_quadratic_intersection_l1221_122176


namespace NUMINAMATH_CALUDE_vector_operation_l1221_122122

def a : Fin 2 → ℝ := ![2, 1]
def b : Fin 2 → ℝ := ![2, -2]

theorem vector_operation : 2 • a - b = ![2, 4] := by sorry

end NUMINAMATH_CALUDE_vector_operation_l1221_122122


namespace NUMINAMATH_CALUDE_last_card_in_box_three_l1221_122131

/-- Represents the number of boxes --/
def num_boxes : ℕ := 7

/-- Represents the total number of cards --/
def total_cards : ℕ := 2015

/-- Represents the length of one complete cycle --/
def cycle_length : ℕ := 12

/-- Calculates the box number for a given card number --/
def box_for_card (card_num : ℕ) : ℕ :=
  let position_in_cycle := card_num % cycle_length
  if position_in_cycle ≤ num_boxes then
    position_in_cycle
  else
    num_boxes - (position_in_cycle - num_boxes)

/-- Theorem stating that the last card (2015th) will be placed in box 3 --/
theorem last_card_in_box_three :
  box_for_card total_cards = 3 := by
  sorry

end NUMINAMATH_CALUDE_last_card_in_box_three_l1221_122131


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1221_122139

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

/-- The main theorem -/
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25 →
  a 3 + a 5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1221_122139


namespace NUMINAMATH_CALUDE_z_in_first_quadrant_l1221_122156

def i : ℂ := Complex.I

theorem z_in_first_quadrant : ∃ z : ℂ, 
  (1 + i) * z = 1 - 2 * i^3 ∧ 
  z.re > 0 ∧ z.im > 0 := by
  sorry

end NUMINAMATH_CALUDE_z_in_first_quadrant_l1221_122156


namespace NUMINAMATH_CALUDE_distance_between_points_l1221_122136

/-- The distance between points (1, 2) and (5, 6) is 4√2 units. -/
theorem distance_between_points : Real.sqrt ((5 - 1)^2 + (6 - 2)^2) = 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l1221_122136


namespace NUMINAMATH_CALUDE_complex_simplification_l1221_122112

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_simplification :
  (5 * (1 + i^3)) / ((2 + i) * (2 - i)) = 1 - i :=
by sorry

end NUMINAMATH_CALUDE_complex_simplification_l1221_122112


namespace NUMINAMATH_CALUDE_limo_cost_per_hour_l1221_122149

/-- Calculates the cost of a limo per hour given prom expenses -/
theorem limo_cost_per_hour 
  (ticket_cost : ℝ) 
  (dinner_cost : ℝ) 
  (tip_percentage : ℝ) 
  (limo_hours : ℝ) 
  (total_cost : ℝ) 
  (h1 : ticket_cost = 100)
  (h2 : dinner_cost = 120)
  (h3 : tip_percentage = 0.3)
  (h4 : limo_hours = 6)
  (h5 : total_cost = 836) :
  (total_cost - (2 * ticket_cost + dinner_cost + tip_percentage * dinner_cost)) / limo_hours = 80 :=
by sorry

end NUMINAMATH_CALUDE_limo_cost_per_hour_l1221_122149


namespace NUMINAMATH_CALUDE_profit_40_percent_l1221_122175

/-- Calculates the profit percentage when selling a certain number of articles at a price equal to the cost of a different number of articles. -/
def profit_percentage (sold : ℕ) (cost_equivalent : ℕ) : ℚ :=
  ((cost_equivalent - sold) / sold) * 100

/-- Theorem stating that selling 50 articles at the cost price of 70 articles results in a 40% profit. -/
theorem profit_40_percent :
  profit_percentage 50 70 = 40 := by
  sorry

end NUMINAMATH_CALUDE_profit_40_percent_l1221_122175


namespace NUMINAMATH_CALUDE_usb_storage_capacity_l1221_122114

/-- Represents the capacity of a storage device in gigabytes -/
def StorageCapacityGB : ℕ := 2

/-- Represents the size of one gigabyte in megabytes -/
def GBtoMB : ℕ := 2^10

/-- Represents the file size of each photo in megabytes -/
def PhotoSizeMB : ℕ := 16

/-- Calculates the number of photos that can be stored -/
def NumberOfPhotos : ℕ := 2^7

theorem usb_storage_capacity :
  StorageCapacityGB * GBtoMB / PhotoSizeMB = NumberOfPhotos :=
sorry

end NUMINAMATH_CALUDE_usb_storage_capacity_l1221_122114


namespace NUMINAMATH_CALUDE_minimum_planting_cost_l1221_122129

/-- Represents the dimensions of a rectangular region -/
structure Region where
  width : ℝ
  height : ℝ

/-- Represents a type of flower with its cost -/
structure Flower where
  name : String
  cost : ℝ

/-- Calculates the area of a region -/
def area (r : Region) : ℝ := r.width * r.height

/-- Calculates the cost of planting a flower in a region -/
def plantingCost (f : Flower) (r : Region) : ℝ := f.cost * area r

/-- The flower bed configuration -/
def flowerBed : Region := { width := 11, height := 6 }

/-- The vertical strip -/
def verticalStrip : Region := { width := 3, height := 6 }

/-- The horizontal strip -/
def horizontalStrip : Region := { width := 11, height := 2 }

/-- The overlap region between vertical and horizontal strips -/
def overlapRegion : Region := { width := 3, height := 2 }

/-- The remaining region -/
def remainingRegion : Region :=
  { width := flowerBed.width - verticalStrip.width,
    height := flowerBed.height - horizontalStrip.height }

/-- The available flower types -/
def flowers : List Flower :=
  [{ name := "Easter Lily", cost := 3 },
   { name := "Dahlia", cost := 2.5 },
   { name := "Canna", cost := 2 }]

/-- Theorem: The minimum cost for planting the flowers is $157 -/
theorem minimum_planting_cost :
  plantingCost (flowers[2]) remainingRegion +
  plantingCost (flowers[1]) verticalStrip +
  plantingCost (flowers[0]) { width := horizontalStrip.width - verticalStrip.width,
                              height := horizontalStrip.height } = 157 := by
  sorry


end NUMINAMATH_CALUDE_minimum_planting_cost_l1221_122129


namespace NUMINAMATH_CALUDE_bobby_total_pieces_l1221_122134

/-- The total number of candy and chocolate pieces Bobby ate -/
def total_pieces (initial_candy : ℕ) (additional_candy : ℕ) (chocolate : ℕ) : ℕ :=
  initial_candy + additional_candy + chocolate

/-- Theorem stating that Bobby ate 51 pieces of candy and chocolate in total -/
theorem bobby_total_pieces :
  total_pieces 33 4 14 = 51 := by
  sorry

end NUMINAMATH_CALUDE_bobby_total_pieces_l1221_122134


namespace NUMINAMATH_CALUDE_intersection_of_S_and_T_l1221_122199

def S : Set ℝ := {x | x^2 + 2*x = 0}
def T : Set ℝ := {x | x^2 - 2*x = 0}

theorem intersection_of_S_and_T : S ∩ T = {0} := by sorry

end NUMINAMATH_CALUDE_intersection_of_S_and_T_l1221_122199


namespace NUMINAMATH_CALUDE_complex_power_problem_l1221_122126

theorem complex_power_problem (z : ℂ) (h : z = (1 + Complex.I)^2 / 2) : z^2023 = -Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_power_problem_l1221_122126


namespace NUMINAMATH_CALUDE_num_divisors_not_div_by_5_eq_4_l1221_122147

/-- The number of positive divisors of 150 that are not divisible by 5 -/
def num_divisors_not_div_by_5 : ℕ :=
  (Finset.filter (fun d => d ∣ 150 ∧ ¬(5 ∣ d)) (Finset.range 151)).card

/-- 150 has the prime factorization 2 * 3 * 5^2 -/
axiom prime_factorization : 150 = 2 * 3 * 5^2

theorem num_divisors_not_div_by_5_eq_4 : num_divisors_not_div_by_5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_num_divisors_not_div_by_5_eq_4_l1221_122147


namespace NUMINAMATH_CALUDE_smallest_x_squared_is_2135_l1221_122101

/-- An isosceles trapezoid with specific properties -/
structure IsoscelesTrapezoid where
  AB : ℝ
  CD : ℝ
  x : ℝ
  has_circle : Bool
  has_tangent_line : Bool

/-- The smallest possible value of x^2 for the given trapezoid -/
def smallest_x_squared (t : IsoscelesTrapezoid) : ℝ := 2135

/-- Theorem stating the smallest possible value of x^2 for the specific trapezoid -/
theorem smallest_x_squared_is_2135 (t : IsoscelesTrapezoid) 
  (h1 : t.AB = 122) 
  (h2 : t.CD = 26) 
  (h3 : t.has_circle = true) 
  (h4 : t.has_tangent_line = true) : 
  smallest_x_squared t = 2135 := by
  sorry

end NUMINAMATH_CALUDE_smallest_x_squared_is_2135_l1221_122101


namespace NUMINAMATH_CALUDE_rupert_weight_l1221_122186

/-- Proves that Rupert weighs 35 kilograms given the conditions -/
theorem rupert_weight (antoinette_weight rupert_weight : ℕ) : 
  antoinette_weight = 63 → 
  antoinette_weight = 2 * rupert_weight - 7 → 
  rupert_weight = 35 := by
  sorry

end NUMINAMATH_CALUDE_rupert_weight_l1221_122186


namespace NUMINAMATH_CALUDE_quadratic_intersection_theorem_l1221_122146

def line_l (x y : ℝ) : Prop := y = 4

def quadratic_function (x a : ℝ) : ℝ :=
  (x - a)^2 + (x - 2*a)^2 + (x - 3*a)^2 - 2*a^2 + a

def has_two_intersections (a : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    line_l x₁ (quadratic_function x₁ a) ∧
    line_l x₂ (quadratic_function x₂ a)

def axis_of_symmetry (a : ℝ) : ℝ := 2 * a

theorem quadratic_intersection_theorem (a : ℝ) :
  has_two_intersections a ∧ axis_of_symmetry a > 0 → 0 < a ∧ a < 4 :=
sorry

end NUMINAMATH_CALUDE_quadratic_intersection_theorem_l1221_122146


namespace NUMINAMATH_CALUDE_river_improvement_equation_l1221_122138

theorem river_improvement_equation (x : ℝ) (h : x > 0) : 
  (4800 / x) - (4800 / (x + 200)) = 4 ↔ 
  (∃ (planned_days actual_days : ℝ),
    planned_days = 4800 / x ∧
    actual_days = 4800 / (x + 200) ∧
    planned_days - actual_days = 4) :=
by sorry

end NUMINAMATH_CALUDE_river_improvement_equation_l1221_122138


namespace NUMINAMATH_CALUDE_garage_sale_necklace_cost_l1221_122142

/-- The cost of each necklace in Isabel's garage sale --/
def cost_per_necklace (total_necklaces : ℕ) (total_earnings : ℕ) : ℚ :=
  total_earnings / total_necklaces

/-- Theorem stating that the cost per necklace is $6 --/
theorem garage_sale_necklace_cost :
  cost_per_necklace 6 36 = 6 := by
  sorry

end NUMINAMATH_CALUDE_garage_sale_necklace_cost_l1221_122142


namespace NUMINAMATH_CALUDE_new_light_wattage_l1221_122137

theorem new_light_wattage (old_wattage : ℝ) (increase_percentage : ℝ) (new_wattage : ℝ) :
  old_wattage = 80 →
  increase_percentage = 0.25 →
  new_wattage = old_wattage * (1 + increase_percentage) →
  new_wattage = 100 :=
by
  sorry

end NUMINAMATH_CALUDE_new_light_wattage_l1221_122137


namespace NUMINAMATH_CALUDE_largest_term_is_115_div_3_l1221_122155

/-- An arithmetic sequence of 5 terms satisfying specific conditions -/
structure ArithmeticSequence where
  terms : Fin 5 → ℚ
  is_arithmetic : ∀ i j k : Fin 5, terms k - terms j = terms j - terms i
  sum_is_100 : (Finset.univ.sum terms) = 100
  ratio_condition : (terms 2 + terms 3 + terms 4) = (1/7) * (terms 0 + terms 1)

/-- The largest term in the arithmetic sequence is 115/3 -/
theorem largest_term_is_115_div_3 (seq : ArithmeticSequence) : seq.terms 4 = 115/3 := by
  sorry

end NUMINAMATH_CALUDE_largest_term_is_115_div_3_l1221_122155


namespace NUMINAMATH_CALUDE_david_airport_distance_l1221_122183

/-- The distance from David's home to the airport --/
def airport_distance : ℝ := by sorry

/-- David's initial speed --/
def initial_speed : ℝ := 35

/-- David's speed increase --/
def speed_increase : ℝ := 15

/-- Time saved by increasing speed --/
def time_saved : ℝ := 1.5

/-- Time early --/
def time_early : ℝ := 0.5

theorem david_airport_distance :
  airport_distance = initial_speed * (airport_distance / initial_speed) +
  (initial_speed + speed_increase) * (time_saved - time_early) ∧
  airport_distance = 210 := by sorry

end NUMINAMATH_CALUDE_david_airport_distance_l1221_122183


namespace NUMINAMATH_CALUDE_min_abs_a_plus_b_l1221_122117

theorem min_abs_a_plus_b (a b : ℤ) (h1 : |a| < |b|) (h2 : |b| ≤ 4) :
  ∃ (m : ℤ), (∀ (x y : ℤ), |x| < |y| → |y| ≤ 4 → m ≤ |x| + y) ∧ m = -4 :=
sorry

end NUMINAMATH_CALUDE_min_abs_a_plus_b_l1221_122117


namespace NUMINAMATH_CALUDE_second_term_is_three_l1221_122193

/-- An arithmetic sequence with common difference 2 -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + 2

/-- Three terms form a geometric sequence -/
def geometric_sequence (x y z : ℝ) : Prop :=
  y^2 = x * z

/-- The second term of the sequence is 3 -/
theorem second_term_is_three (a : ℕ → ℝ) :
  arithmetic_sequence a →
  geometric_sequence (a 1) (a 2) (a 5) →
  a 2 = 3 := by sorry

end NUMINAMATH_CALUDE_second_term_is_three_l1221_122193


namespace NUMINAMATH_CALUDE_mara_crayon_count_l1221_122124

theorem mara_crayon_count : ∀ (mara_crayons : ℕ),
  (mara_crayons : ℚ) * (1 / 10 : ℚ) + (50 : ℚ) * (1 / 5 : ℚ) = 14 →
  mara_crayons = 40 := by
  sorry

end NUMINAMATH_CALUDE_mara_crayon_count_l1221_122124


namespace NUMINAMATH_CALUDE_toothpick_pattern_l1221_122116

/-- Given an arithmetic sequence with first term 4 and common difference 4,
    the 150th term is equal to 600. -/
theorem toothpick_pattern (a : ℕ) (d : ℕ) (n : ℕ) :
  a = 4 → d = 4 → n = 150 → a + (n - 1) * d = 600 := by
  sorry

end NUMINAMATH_CALUDE_toothpick_pattern_l1221_122116


namespace NUMINAMATH_CALUDE_fraction_is_positive_integer_iff_p_18_l1221_122185

theorem fraction_is_positive_integer_iff_p_18 (p : ℕ+) :
  (∃ (n : ℕ+), (5 * p + 40 : ℚ) / (3 * p - 7 : ℚ) = n) ↔ p = 18 := by
  sorry

end NUMINAMATH_CALUDE_fraction_is_positive_integer_iff_p_18_l1221_122185


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1221_122184

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n > 0) →
  (∀ n, a (n + 1) = q * a n) →
  a 1 = 3 →
  a 1 + a 2 + a 3 = 21 →
  a 4 + a 5 + a 6 = 168 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1221_122184


namespace NUMINAMATH_CALUDE_cuboid_volume_doubled_l1221_122111

/-- Theorem: Doubling dimensions of a cuboid results in 8 times the original volume -/
theorem cuboid_volume_doubled (l w h : ℝ) (l_pos : 0 < l) (w_pos : 0 < w) (h_pos : 0 < h) :
  (2 * l) * (2 * w) * (2 * h) = 8 * (l * w * h) := by
  sorry

#check cuboid_volume_doubled

end NUMINAMATH_CALUDE_cuboid_volume_doubled_l1221_122111


namespace NUMINAMATH_CALUDE_max_values_l1221_122169

theorem max_values (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b^2 = 1) :
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y^2 = 1 ∧ b * Real.sqrt a ≤ x * Real.sqrt y) ∧
  (∀ (x y : ℝ), x > 0 → y > 0 → x + y^2 = 1 → b * Real.sqrt a ≤ x * Real.sqrt y) ∧
  b * Real.sqrt a ≤ 1/2 ∧
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y^2 = 1 ∧ Real.sqrt x + y ≤ Real.sqrt 2) ∧
  (∀ (x y : ℝ), x > 0 → y > 0 → x + y^2 = 1 → Real.sqrt x + y ≤ Real.sqrt 2) ∧
  Real.sqrt a + b ≤ Real.sqrt 2 :=
by sorry


end NUMINAMATH_CALUDE_max_values_l1221_122169


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_sum_of_divisors_200_l1221_122157

/-- The sum of divisors of a natural number n -/
def sumOfDivisors (n : ℕ) : ℕ := sorry

/-- The largest prime factor of a natural number n -/
def largestPrimeFactor (n : ℕ) : ℕ := sorry

theorem largest_prime_factor_of_sum_of_divisors_200 :
  largestPrimeFactor (sumOfDivisors 200) = 31 := by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_sum_of_divisors_200_l1221_122157


namespace NUMINAMATH_CALUDE_original_number_proof_l1221_122191

theorem original_number_proof (x : ℝ) (h1 : x > 0) (h2 : 1000 * x = 3 * (1 / x)) : 
  x = Real.sqrt 30 / 100 := by
sorry

end NUMINAMATH_CALUDE_original_number_proof_l1221_122191


namespace NUMINAMATH_CALUDE_b_received_15_pencils_l1221_122119

/-- The number of pencils each student received -/
structure PencilDistribution where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ

/-- The conditions of the pencil distribution problem -/
def ValidDistribution (p : PencilDistribution) : Prop :=
  p.a + p.b + p.c + p.d = 53 ∧
  (max p.a (max p.b (max p.c p.d))) - (min p.a (min p.b (min p.c p.d))) ≤ 5 ∧
  p.a + p.b = 2 * p.c ∧
  p.c + p.b = 2 * p.d

/-- The theorem stating that B received 15 pencils -/
theorem b_received_15_pencils (p : PencilDistribution) (h : ValidDistribution p) : p.b = 15 := by
  sorry

end NUMINAMATH_CALUDE_b_received_15_pencils_l1221_122119


namespace NUMINAMATH_CALUDE_escalator_ride_time_l1221_122180

/-- Represents the scenario of Clea walking on an escalator -/
structure EscalatorScenario where
  /-- Clea's walking speed on stationary escalator (units per second) -/
  walkingSpeed : ℝ
  /-- Total distance of the escalator (units) -/
  escalatorDistance : ℝ
  /-- Speed of the moving escalator (units per second) -/
  escalatorSpeed : ℝ

/-- Time taken for Clea to walk down the stationary escalator -/
def stationaryTime (scenario : EscalatorScenario) : ℝ := 70

/-- Time taken for Clea to walk down the moving escalator -/
def movingTime (scenario : EscalatorScenario) : ℝ := 30

/-- Clea's walking speed increase factor on moving escalator -/
def speedIncreaseFactor : ℝ := 1.5

/-- Theorem stating the time taken for Clea to ride the escalator without walking -/
theorem escalator_ride_time (scenario : EscalatorScenario) :
  scenario.escalatorDistance / scenario.escalatorSpeed = 84 :=
by sorry

end NUMINAMATH_CALUDE_escalator_ride_time_l1221_122180


namespace NUMINAMATH_CALUDE_set_equality_unordered_elements_l1221_122133

theorem set_equality_unordered_elements : 
  let M : Set ℕ := {4, 5}
  let N : Set ℕ := {5, 4}
  M = N :=
by sorry

end NUMINAMATH_CALUDE_set_equality_unordered_elements_l1221_122133


namespace NUMINAMATH_CALUDE_stating_assignment_methods_eq_36_l1221_122161

/-- Represents the number of workshops --/
def num_workshops : ℕ := 3

/-- Represents the total number of employees --/
def total_employees : ℕ := 5

/-- Represents the number of employees that must be assigned together --/
def paired_employees : ℕ := 2

/-- Represents the number of remaining employees after considering the paired employees --/
def remaining_employees : ℕ := total_employees - paired_employees

/-- 
  Calculates the number of ways to assign employees to workshops
  given the constraints mentioned in the problem
--/
def assignment_methods : ℕ := 
  num_workshops * (remaining_employees.factorial + remaining_employees.choose 2 * (num_workshops - 1))

/-- 
  Theorem stating that the number of assignment methods
  satisfying the given conditions is 36
--/
theorem assignment_methods_eq_36 : assignment_methods = 36 := by
  sorry

end NUMINAMATH_CALUDE_stating_assignment_methods_eq_36_l1221_122161


namespace NUMINAMATH_CALUDE_square_side_length_l1221_122108

/-- Given a rectangle with width 36 cm and length 64 cm, and a square whose perimeter
    equals the rectangle's perimeter, prove that the side length of the square is 50 cm. -/
theorem square_side_length (rectangle_width rectangle_length : ℝ)
                            (square_side : ℝ)
                            (h1 : rectangle_width = 36)
                            (h2 : rectangle_length = 64)
                            (h3 : 4 * square_side = 2 * (rectangle_width + rectangle_length)) :
  square_side = 50 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l1221_122108


namespace NUMINAMATH_CALUDE_high_quality_seed_probability_l1221_122179

/-- Represents the composition of seeds in a batch -/
structure SeedBatch where
  second_grade : ℝ
  third_grade : ℝ
  fourth_grade : ℝ

/-- Represents the probabilities of producing high-quality products for each seed grade -/
structure QualityProbabilities where
  first_grade : ℝ
  second_grade : ℝ
  third_grade : ℝ
  fourth_grade : ℝ

/-- Calculates the probability of selecting a high-quality seed from a given batch -/
def high_quality_probability (batch : SeedBatch) (probs : QualityProbabilities) : ℝ :=
  let first_grade_proportion := 1 - (batch.second_grade + batch.third_grade + batch.fourth_grade)
  first_grade_proportion * probs.first_grade +
  batch.second_grade * probs.second_grade +
  batch.third_grade * probs.third_grade +
  batch.fourth_grade * probs.fourth_grade

/-- Theorem stating the probability of selecting a high-quality seed from the given batch -/
theorem high_quality_seed_probability :
  let batch := SeedBatch.mk 0.02 0.015 0.01
  let probs := QualityProbabilities.mk 0.5 0.15 0.1 0.05
  high_quality_probability batch probs = 0.4825 := by
  sorry


end NUMINAMATH_CALUDE_high_quality_seed_probability_l1221_122179


namespace NUMINAMATH_CALUDE_smallest_k_for_sum_and_product_existence_of_solution_smallest_k_is_four_l1221_122143

theorem smallest_k_for_sum_and_product (k : ℝ) : 
  (k > 0 ∧ 
   ∃ a b : ℝ, a + b = k ∧ a * b = k) → 
  k ≥ 4 :=
by sorry

theorem existence_of_solution : 
  ∃ k a b : ℝ, k > 0 ∧ a + b = k ∧ a * b = k ∧ k = 4 :=
by sorry

theorem smallest_k_is_four : 
  ∃! k : ℝ, k > 0 ∧ 
  (∃ a b : ℝ, a + b = k ∧ a * b = k) ∧
  (∀ k' : ℝ, k' > 0 → (∃ a b : ℝ, a + b = k' ∧ a * b = k') → k' ≥ k) ∧
  k = 4 :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_for_sum_and_product_existence_of_solution_smallest_k_is_four_l1221_122143


namespace NUMINAMATH_CALUDE_correct_average_marks_l1221_122127

theorem correct_average_marks (n : ℕ) (incorrect_avg : ℝ) (wrong_mark correct_mark : ℝ) :
  n = 10 ∧ incorrect_avg = 100 ∧ wrong_mark = 60 ∧ correct_mark = 10 →
  (n * incorrect_avg - (wrong_mark - correct_mark)) / n = 95 := by
  sorry

end NUMINAMATH_CALUDE_correct_average_marks_l1221_122127


namespace NUMINAMATH_CALUDE_min_value_on_circle_l1221_122144

theorem min_value_on_circle (x y : ℝ) : 
  x^2 + y^2 - 8*x + 6*y + 16 = 0 → ∃ (m : ℝ), m = 4 ∧ ∀ (a b : ℝ), a^2 + b^2 - 8*a + 6*b + 16 = 0 → x^2 + y^2 ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_on_circle_l1221_122144


namespace NUMINAMATH_CALUDE_x_squared_minus_y_squared_l1221_122104

theorem x_squared_minus_y_squared (x y : ℚ) 
  (h1 : x + y = 3/8) 
  (h2 : x - y = 1/4) : 
  x^2 - y^2 = 3/32 := by sorry

end NUMINAMATH_CALUDE_x_squared_minus_y_squared_l1221_122104


namespace NUMINAMATH_CALUDE_light_intensity_reduction_l1221_122107

/-- Given light with original intensity a passing through n pieces of glass,
    each reducing intensity by 10%, calculate the final intensity -/
def final_intensity (a : ℝ) (n : ℕ) : ℝ :=
  a * (0.9 ^ n)

/-- Theorem: Light with original intensity a passing through 3 pieces of glass,
    each reducing intensity by 10%, results in a final intensity of 0.729a -/
theorem light_intensity_reduction (a : ℝ) :
  final_intensity a 3 = 0.729 * a := by
  sorry

end NUMINAMATH_CALUDE_light_intensity_reduction_l1221_122107


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l1221_122141

theorem complex_fraction_equality : 
  let z₁ : ℂ := 1 - I
  let z₂ : ℂ := 1 + I
  z₁ / (z₂ * I) = -2 * I := by
sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l1221_122141


namespace NUMINAMATH_CALUDE_decagon_triangle_probability_l1221_122148

/-- The number of vertices in a regular decagon -/
def decagon_vertices : ℕ := 10

/-- The number of vertices needed to form a triangle -/
def triangle_vertices : ℕ := 3

/-- The total number of possible triangles formed by choosing 3 vertices from a decagon -/
def total_triangles : ℕ := Nat.choose decagon_vertices triangle_vertices

/-- The number of triangles with at least one side being a side of the decagon -/
def favorable_triangles : ℕ := 60

/-- The probability of forming a triangle with at least one side being a side of the decagon -/
def probability : ℚ := favorable_triangles / total_triangles

theorem decagon_triangle_probability :
  probability = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_decagon_triangle_probability_l1221_122148


namespace NUMINAMATH_CALUDE_triangle_areas_product_l1221_122173

theorem triangle_areas_product (h₁ h₂ h₃ : ℝ) 
  (h1 : h₁ = 1)
  (h2 : h₂ = 1 + Real.sqrt 3 / 2)
  (h3 : h₃ = 1 - Real.sqrt 3 / 2) :
  (1/2 * 1 * h₁) * (1/2 * 1 * h₂) * (1/2 * 1 * h₃) = 1/32 := by
  sorry

#check triangle_areas_product

end NUMINAMATH_CALUDE_triangle_areas_product_l1221_122173


namespace NUMINAMATH_CALUDE_max_product_equals_sum_l1221_122100

theorem max_product_equals_sum (H M T : ℤ) : 
  H * M * M * T = H + M + M + T → H * M * M * T ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_max_product_equals_sum_l1221_122100


namespace NUMINAMATH_CALUDE_divisibility_by_42p_l1221_122178

theorem divisibility_by_42p (p : Nat) (h_prime : Nat.Prime p) (h_gt_3 : p > 3) :
  ∃ k : ℤ, (3^p - 2^p - 1 : ℤ) = 42 * p * k := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_42p_l1221_122178


namespace NUMINAMATH_CALUDE_frog_escapes_in_18_days_l1221_122181

/-- Represents the depth of the well in meters -/
def well_depth : ℕ := 20

/-- Represents the distance the frog climbs up each day in meters -/
def climb_distance : ℕ := 3

/-- Represents the distance the frog slips down each day in meters -/
def slip_distance : ℕ := 2

/-- Represents the net distance the frog climbs each day in meters -/
def net_daily_progress : ℕ := climb_distance - slip_distance

/-- Theorem stating that the frog can climb out of the well in 18 days -/
theorem frog_escapes_in_18_days :
  ∃ (n : ℕ), n = 18 ∧ n * net_daily_progress + climb_distance ≥ well_depth :=
sorry

end NUMINAMATH_CALUDE_frog_escapes_in_18_days_l1221_122181


namespace NUMINAMATH_CALUDE_round_trip_average_speed_river_boat_average_speed_l1221_122130

/-- The average speed of a round trip given upstream and downstream speeds -/
theorem round_trip_average_speed (upstream_speed downstream_speed : ℝ) 
  (upstream_speed_pos : 0 < upstream_speed)
  (downstream_speed_pos : 0 < downstream_speed) :
  (2 * upstream_speed * downstream_speed) / (upstream_speed + downstream_speed) =
  (2 * 6 * 8) / (6 + 8) :=
by sorry

/-- The specific case for the river boat problem -/
theorem river_boat_average_speed :
  (2 * 6 * 8) / (6 + 8) = 48 / 7 :=
by sorry

end NUMINAMATH_CALUDE_round_trip_average_speed_river_boat_average_speed_l1221_122130


namespace NUMINAMATH_CALUDE_remaining_episodes_l1221_122182

theorem remaining_episodes (total_seasons : ℕ) (episodes_per_season : ℕ) 
  (watched_fraction : ℚ) : 
  total_seasons = 12 → 
  episodes_per_season = 20 → 
  watched_fraction = 1/3 →
  total_seasons * episodes_per_season - (watched_fraction * (total_seasons * episodes_per_season)).num = 160 := by
  sorry

end NUMINAMATH_CALUDE_remaining_episodes_l1221_122182


namespace NUMINAMATH_CALUDE_sum_in_base7_l1221_122118

/-- Converts a base 7 number to base 10 --/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * 7^i) 0

/-- Converts a base 10 number to base 7 --/
def base10ToBase7 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) :=
      if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
    aux n []

/-- The statement to prove --/
theorem sum_in_base7 :
  let a := [2, 1]  -- 12 in base 7
  let b := [5, 4, 2]  -- 245 in base 7
  let sum := base7ToBase10 a + base7ToBase10 b
  base10ToBase7 sum = [0, 6, 2] := by
  sorry

end NUMINAMATH_CALUDE_sum_in_base7_l1221_122118


namespace NUMINAMATH_CALUDE_f_extremum_f_range_of_a_l1221_122158

noncomputable section

-- Define the function f
def f (a b x : ℝ) : ℝ := (a * (x - 1) + b * Real.exp x) / Real.exp x

-- Part 1
theorem f_extremum :
  let a : ℝ := -1
  let b : ℝ := 0
  (∃ x : ℝ, ∀ y : ℝ, f a b y ≥ f a b x) ∧
  (∀ x : ℝ, f a b x ≥ -1 / Real.exp 2) ∧
  (¬ ∃ M : ℝ, ∀ x : ℝ, f a b x ≤ M) := by sorry

-- Part 2
theorem f_range_of_a :
  let b : ℝ := 1
  (∀ a : ℝ, (∀ x : ℝ, f a b x ≠ 0) → a ∈ Set.Ioo (-Real.exp 2) 0) ∧
  (∀ a : ℝ, a ∈ Set.Ioo (-Real.exp 2) 0 → (∀ x : ℝ, f a b x ≠ 0)) := by sorry

end

end NUMINAMATH_CALUDE_f_extremum_f_range_of_a_l1221_122158


namespace NUMINAMATH_CALUDE_percentage_problem_l1221_122177

theorem percentage_problem (x : ℝ) (h : 0.4 * x = 160) : 0.1 * x = 40 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1221_122177


namespace NUMINAMATH_CALUDE_doctors_lawyers_ratio_l1221_122150

theorem doctors_lawyers_ratio (d l : ℕ) (h_total : d + l > 0) :
  (45 * d + 55 * l) / (d + l) = 47 →
  d = 4 * l :=
by
  sorry

end NUMINAMATH_CALUDE_doctors_lawyers_ratio_l1221_122150


namespace NUMINAMATH_CALUDE_greatest_divisor_of_product_l1221_122174

def S : Set (ℤ × ℤ × ℤ × ℤ × ℤ × ℤ) :=
  {t | let (a, b, c, d, e, f) := t
       a^2 + b^2 + c^2 + d^2 + e^2 = f^2}

theorem greatest_divisor_of_product (k : ℕ) : k = 24 ↔ 
  (∀ t ∈ S, let (a, b, c, d, e, f) := t
            (k : ℤ) ∣ a * b * c * d * e * f) ∧
  (∀ m > k, ∃ t ∈ S, let (a, b, c, d, e, f) := t
            ¬((m : ℤ) ∣ a * b * c * d * e * f)) := by
  sorry

end NUMINAMATH_CALUDE_greatest_divisor_of_product_l1221_122174


namespace NUMINAMATH_CALUDE_max_six_yuan_items_l1221_122195

/-- Represents the number of items bought at each price point -/
structure ItemCounts where
  twoYuan : ℕ
  fourYuan : ℕ
  sixYuan : ℕ

/-- The problem constraints -/
def isValidPurchase (items : ItemCounts) : Prop :=
  items.twoYuan + items.fourYuan + items.sixYuan = 16 ∧
  2 * items.twoYuan + 4 * items.fourYuan + 6 * items.sixYuan = 60

/-- The theorem stating the maximum number of 6-yuan items -/
theorem max_six_yuan_items :
  ∃ (max : ℕ), max = 7 ∧
  (∀ (items : ItemCounts), isValidPurchase items → items.sixYuan ≤ max) ∧
  (∃ (items : ItemCounts), isValidPurchase items ∧ items.sixYuan = max) := by
  sorry

end NUMINAMATH_CALUDE_max_six_yuan_items_l1221_122195


namespace NUMINAMATH_CALUDE_acute_triangle_condition_l1221_122123

/-- 
Given a unit circle with diameter AB, where A(-1, 0) and B(1, 0),
and a point D(x, 0) on AB, prove that AD, BD, and CD form an acute triangle
if and only if x is in the open interval (2 - √5, √5 - 2),
where C is the point where DC ⊥ AB intersects the circle.
-/
theorem acute_triangle_condition (x : ℝ) : 
  let A : ℝ × ℝ := (-1, 0)
  let B : ℝ × ℝ := (1, 0)
  let D : ℝ × ℝ := (x, 0)
  let C : ℝ × ℝ := (x, Real.sqrt (1 - x^2))
  let AD := Real.sqrt ((x + 1)^2)
  let BD := Real.sqrt ((1 - x)^2)
  let CD := Real.sqrt (1 - x^2)
  (AD^2 + BD^2 > CD^2 ∧ AD^2 + CD^2 > BD^2 ∧ BD^2 + CD^2 > AD^2) ↔ 
  (x > 2 - Real.sqrt 5 ∧ x < Real.sqrt 5 - 2) :=
by sorry


end NUMINAMATH_CALUDE_acute_triangle_condition_l1221_122123


namespace NUMINAMATH_CALUDE_jesse_has_21_bananas_l1221_122106

/-- The number of bananas Jesse has -/
def jesse_bananas : ℕ := 21

/-- The number of friends Jesse shares the bananas with -/
def num_friends : ℕ := 3

/-- The number of bananas each friend gets when Jesse shares his bananas -/
def bananas_per_friend : ℕ := 7

/-- Theorem stating that Jesse has 21 bananas -/
theorem jesse_has_21_bananas : 
  jesse_bananas = num_friends * bananas_per_friend := by
  sorry

end NUMINAMATH_CALUDE_jesse_has_21_bananas_l1221_122106


namespace NUMINAMATH_CALUDE_athletes_on_second_floor_l1221_122194

/-- Proves that given a hotel with three floors housing 38 athletes, where 26 athletes are on the first and second floors, and 27 athletes are on the second and third floors, the number of athletes on the second floor is 15. -/
theorem athletes_on_second_floor 
  (total_athletes : ℕ) 
  (first_second : ℕ) 
  (second_third : ℕ) 
  (h1 : total_athletes = 38) 
  (h2 : first_second = 26) 
  (h3 : second_third = 27) : 
  ∃ (first second third : ℕ), 
    first + second + third = total_athletes ∧ 
    first + second = first_second ∧ 
    second + third = second_third ∧ 
    second = 15 :=
by sorry

end NUMINAMATH_CALUDE_athletes_on_second_floor_l1221_122194

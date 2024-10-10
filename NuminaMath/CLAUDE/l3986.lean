import Mathlib

namespace paula_candies_l3986_398660

theorem paula_candies (x : ℕ) : 
  (x + 4 = 6 * 4) → x = 20 := by
sorry

end paula_candies_l3986_398660


namespace fraction_addition_l3986_398601

theorem fraction_addition : (1 : ℚ) / 3 + (-1 / 2) = -1 / 6 := by sorry

end fraction_addition_l3986_398601


namespace max_value_g_in_unit_interval_l3986_398616

-- Define the function g(x)
def g (x : ℝ) : ℝ := x * (x^2 - 1)

-- State the theorem
theorem max_value_g_in_unit_interval :
  ∃ (M : ℝ), M = 0 ∧ ∀ x, x ∈ Set.Icc 0 1 → g x ≤ M :=
by
  sorry

end max_value_g_in_unit_interval_l3986_398616


namespace sqrt_three_minus_sin_squared_fifteen_l3986_398638

theorem sqrt_three_minus_sin_squared_fifteen (π : Real) :
  (Real.sqrt 3) / 2 - Real.sqrt 3 * (Real.sin (π / 12))^2 = 3 / 4 := by
  sorry

end sqrt_three_minus_sin_squared_fifteen_l3986_398638


namespace laura_walk_distance_l3986_398658

/-- Calculates the total distance walked in miles given the number of blocks walked east and north, and the length of each block in miles. -/
def total_distance (blocks_east blocks_north : ℕ) (block_length : ℚ) : ℚ :=
  (blocks_east + blocks_north : ℚ) * block_length

/-- Proves that walking 8 blocks east and 14 blocks north, with each block being 1/4 mile, results in a total distance of 5.5 miles. -/
theorem laura_walk_distance : total_distance 8 14 (1/4) = 5.5 := by
  sorry

end laura_walk_distance_l3986_398658


namespace saree_price_after_discounts_l3986_398680

def original_price : ℝ := 1000

def discount1 : ℝ := 0.30
def discount2 : ℝ := 0.15
def discount3 : ℝ := 0.10
def discount4 : ℝ := 0.05

def apply_discount (price : ℝ) (discount : ℝ) : ℝ :=
  price * (1 - discount)

def final_price : ℝ :=
  apply_discount (apply_discount (apply_discount (apply_discount original_price discount1) discount2) discount3) discount4

theorem saree_price_after_discounts :
  ⌊final_price⌋ = 509 := by sorry

end saree_price_after_discounts_l3986_398680


namespace max_shapes_from_7x7_grid_l3986_398654

/-- Represents a grid with dimensions n x n -/
structure Grid (n : ℕ) where
  size : ℕ := n * n

/-- Represents a shape that can be cut from the grid -/
inductive Shape
  | Square : Shape  -- 2x2 square
  | Rectangle : Shape  -- 1x4 rectangle

/-- The size of a shape in terms of grid cells -/
def shapeSize : Shape → ℕ
  | Shape.Square => 4
  | Shape.Rectangle => 4

/-- The maximum number of shapes that can be cut from a grid -/
def maxShapes (g : Grid 7) : ℕ :=
  g.size / shapeSize Shape.Square

/-- Checks if a number of shapes can be equally divided between squares and rectangles -/
def isEquallyDivisible (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2 * k

theorem max_shapes_from_7x7_grid :
  ∃ (n : ℕ), maxShapes (Grid.mk 7) = n ∧ isEquallyDivisible n ∧ n = 12 := by
  sorry

end max_shapes_from_7x7_grid_l3986_398654


namespace family_total_weight_l3986_398646

/-- Represents the weights of a family consisting of a mother, daughter, and grandchild. -/
structure FamilyWeights where
  mother : ℝ
  daughter : ℝ
  grandchild : ℝ

/-- The total weight of the family members. -/
def FamilyWeights.total (fw : FamilyWeights) : ℝ :=
  fw.mother + fw.daughter + fw.grandchild

/-- The conditions given in the problem. -/
def satisfies_conditions (fw : FamilyWeights) : Prop :=
  fw.daughter + fw.grandchild = 60 ∧
  fw.grandchild = (1 / 5) * fw.mother ∧
  fw.daughter = 42

/-- Theorem stating that given the conditions, the total weight is 150 kg. -/
theorem family_total_weight (fw : FamilyWeights) 
  (h : satisfies_conditions fw) : fw.total = 150 := by
  sorry

end family_total_weight_l3986_398646


namespace perfect_squares_problem_l3986_398600

theorem perfect_squares_problem :
  ¬∃ (x : ℝ), x^2 = 5^2025 ∧
  ∃ (a : ℝ), a^2 = 3^2024 ∧
  ∃ (b : ℝ), b^2 = 7^2026 ∧
  ∃ (c : ℝ), c^2 = 8^2027 ∧
  ∃ (d : ℝ), d^2 = 9^2028 :=
by sorry

end perfect_squares_problem_l3986_398600


namespace smallest_value_theorem_l3986_398615

theorem smallest_value_theorem (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h : 21 * a * b + 2 * b * c + 8 * c * a ≤ 12) :
  ∀ x y z : ℝ, 0 < x → 0 < y → 0 < z → 
  21 * x * y + 2 * y * z + 8 * z * x ≤ 12 → 
  1 / a + 2 / b + 3 / c ≤ 1 / x + 2 / y + 3 / z :=
by sorry

end smallest_value_theorem_l3986_398615


namespace pastry_distribution_combinations_l3986_398626

/-- The number of ways to distribute additional items among a subset of groups,
    given that each group already has one item. -/
def distribute_additional_items (total_items : ℕ) (total_groups : ℕ) (subset_groups : ℕ) : ℕ :=
  Nat.choose (subset_groups + (total_items - total_groups) - 1) (subset_groups - 1)

/-- Theorem stating that distributing 3 additional items among 4 groups,
    given that 5 items have already been distributed among 5 groups, results in 20 combinations. -/
theorem pastry_distribution_combinations :
  distribute_additional_items 8 5 4 = 20 := by
  sorry

end pastry_distribution_combinations_l3986_398626


namespace mod_nineteen_problem_l3986_398672

theorem mod_nineteen_problem : ∃! n : ℤ, 0 ≤ n ∧ n < 19 ∧ 38574 ≡ n [ZMOD 19] := by sorry

end mod_nineteen_problem_l3986_398672


namespace quadrilateral_perimeter_l3986_398677

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the perimeter function
def perimeter (q : Quadrilateral) : ℝ := sorry

-- Define the length function
def length (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define perpendicularity
def perpendicular (p1 p2 p3 p4 : ℝ × ℝ) : Prop := sorry

theorem quadrilateral_perimeter (ABCD : Quadrilateral) :
  perpendicular ABCD.A ABCD.B ABCD.B ABCD.C →
  perpendicular ABCD.D ABCD.C ABCD.B ABCD.C →
  length ABCD.A ABCD.B = 7 →
  length ABCD.D ABCD.C = 3 →
  length ABCD.B ABCD.C = 10 →
  length ABCD.A ABCD.C = 15 →
  perimeter ABCD = 20 + 6 * Real.sqrt 6 := by
  sorry

end quadrilateral_perimeter_l3986_398677


namespace total_rectangles_3x3_grid_l3986_398667

/-- Represents a grid of points -/
structure Grid where
  rows : Nat
  cols : Nat

/-- Represents a rectangle on the grid -/
structure Rectangle where
  width : Nat
  height : Nat

/-- Counts the number of rectangles of a given size on the grid -/
def countRectangles (g : Grid) (r : Rectangle) : Nat :=
  sorry

/-- The total number of rectangles on a 3x3 grid -/
def totalRectangles : Nat :=
  let g : Grid := { rows := 3, cols := 3 }
  (countRectangles g { width := 1, height := 1 }) +
  (countRectangles g { width := 1, height := 2 }) +
  (countRectangles g { width := 1, height := 3 }) +
  (countRectangles g { width := 2, height := 1 }) +
  (countRectangles g { width := 2, height := 2 }) +
  (countRectangles g { width := 2, height := 3 }) +
  (countRectangles g { width := 3, height := 1 }) +
  (countRectangles g { width := 3, height := 2 })

/-- Theorem stating that the total number of rectangles on a 3x3 grid is 124 -/
theorem total_rectangles_3x3_grid :
  totalRectangles = 124 := by
  sorry

end total_rectangles_3x3_grid_l3986_398667


namespace origin_inside_ellipse_iff_k_range_l3986_398689

/-- The ellipse equation -/
def ellipse_equation (k x y : ℝ) : ℝ := k^2 * x^2 + y^2 - 4*k*x + 2*k*y + k^2 - 1

/-- The origin is inside the ellipse -/
def origin_inside_ellipse (k : ℝ) : Prop := ellipse_equation k 0 0 < 0

/-- Theorem: The origin is inside the ellipse if and only if 0 < |k| < 1 -/
theorem origin_inside_ellipse_iff_k_range (k : ℝ) : 
  origin_inside_ellipse k ↔ 0 < |k| ∧ |k| < 1 := by sorry

end origin_inside_ellipse_iff_k_range_l3986_398689


namespace solution_equality_l3986_398675

theorem solution_equality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (Real.sqrt 2 * x + 1 / (Real.sqrt 2 * x) +
   Real.sqrt 2 * y + 1 / (Real.sqrt 2 * y) +
   Real.sqrt 2 * z + 1 / (Real.sqrt 2 * z) =
   6 - 2 * Real.sqrt (2 * x) * |y - z| -
   Real.sqrt (2 * y) * (x - z)^2 -
   Real.sqrt (2 * z) * Real.sqrt |x - y|) ↔
  (x = 1 / Real.sqrt 2 ∧ y = 1 / Real.sqrt 2 ∧ z = 1 / Real.sqrt 2) :=
by sorry

end solution_equality_l3986_398675


namespace largest_digit_sum_l3986_398696

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def isDigit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

theorem largest_digit_sum (a b c : ℕ) (y : ℕ) :
  isDigit a → isDigit b → isDigit c →
  isPrime y →
  0 ≤ y ∧ y ≤ 7 →
  (a * 100 + b * 10 + c : ℚ) / 1000 = 1 / y →
  a + b + c ≤ 5 :=
sorry

end largest_digit_sum_l3986_398696


namespace add_and_convert_to_base7_37_45_l3986_398634

/-- Converts a number from base 10 to base 7 -/
def toBase7 (n : ℕ) : ℕ := sorry

/-- Adds two numbers in base 10 and returns the result in base 7 -/
def addAndConvertToBase7 (a b : ℕ) : ℕ :=
  toBase7 (a + b)

theorem add_and_convert_to_base7_37_45 :
  addAndConvertToBase7 37 45 = 145 := by sorry

end add_and_convert_to_base7_37_45_l3986_398634


namespace probability_AC_less_than_11_l3986_398657

-- Define the given lengths
def AB : ℝ := 10
def BC : ℝ := 6

-- Define the maximum length of AC
def AC_max : ℝ := 11

-- Define the angle α
def α : ℝ → Prop := λ x => 0 < x ∧ x < Real.pi / 2

-- Define the probability function
noncomputable def P : ℝ := (2 / Real.pi) * Real.arctan (4 / (3 * Real.sqrt 63))

-- State the theorem
theorem probability_AC_less_than_11 :
  ∀ x, α x → P = (2 / Real.pi) * Real.arctan (4 / (3 * Real.sqrt 63)) :=
by sorry

end probability_AC_less_than_11_l3986_398657


namespace angle_subtraction_l3986_398635

-- Define a structure for angles in degrees and minutes
structure Angle :=
  (degrees : ℕ)
  (minutes : ℕ)

-- Define the given angle
def angle1 : Angle := ⟨20, 18⟩

-- Define the operation to subtract an Angle from 90 degrees
def subtractFrom90 (a : Angle) : Angle :=
  let totalMinutes := 90 * 60 - (a.degrees * 60 + a.minutes)
  ⟨totalMinutes / 60, totalMinutes % 60⟩

-- State the theorem
theorem angle_subtraction :
  subtractFrom90 angle1 = ⟨69, 42⟩ := by
  sorry


end angle_subtraction_l3986_398635


namespace circle_intersection_and_common_chord_l3986_398699

-- Define the circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 6*y - 1 = 0
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 10*x - 12*y + 45 = 0

-- Define the line equation
def line_eq (x y : ℝ) : Prop := 4*x + 3*y - 23 = 0

-- Define the intersection of circles
def circles_intersect (C₁ C₂ : ℝ → ℝ → Prop) : Prop :=
  ∃ x y, C₁ x y ∧ C₂ x y

-- Define the common chord
def common_chord (C₁ C₂ line_eq : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, (C₁ x y ∧ C₂ x y) → line_eq x y

-- Theorem statement
theorem circle_intersection_and_common_chord :
  (circles_intersect C₁ C₂) ∧
  (common_chord C₁ C₂ line_eq) ∧
  (∃ a b, C₁ a b ∧ C₂ a b ∧ (a - 1)^2 + (b - 3)^2 = 7) :=
sorry

end circle_intersection_and_common_chord_l3986_398699


namespace moores_law_transistor_growth_l3986_398663

/-- Moore's Law Transistor Growth --/
theorem moores_law_transistor_growth
  (initial_year : Nat)
  (final_year : Nat)
  (initial_transistors : Nat)
  (doubling_period : Nat)
  (h1 : initial_year = 1985)
  (h2 : final_year = 2005)
  (h3 : initial_transistors = 500000)
  (h4 : doubling_period = 2) :
  initial_transistors * 2^((final_year - initial_year) / doubling_period) = 512000000 :=
by sorry

end moores_law_transistor_growth_l3986_398663


namespace add_fractions_three_fourths_five_ninths_l3986_398652

theorem add_fractions_three_fourths_five_ninths :
  (3 : ℚ) / 4 + (5 : ℚ) / 9 = (47 : ℚ) / 36 := by
  sorry

end add_fractions_three_fourths_five_ninths_l3986_398652


namespace no_ab_term_in_polynomial_l3986_398669

theorem no_ab_term_in_polynomial (m : ℝ) : 
  (∀ a b : ℝ, (a^2 + 2*a*b - b^2) - (a^2 + m*a*b + 2*b^2) = (-3:ℝ)*b^2) → m = 2 := by
  sorry

end no_ab_term_in_polynomial_l3986_398669


namespace rectangular_box_surface_area_l3986_398674

/-- 
Given a rectangular box with dimensions a, b, and c, 
if the sum of the lengths of its twelve edges is 172 
and the distance from one corner to the farthest corner is 21, 
then its total surface area is 1408.
-/
theorem rectangular_box_surface_area 
  (a b c : ℝ) 
  (h1 : 4 * a + 4 * b + 4 * c = 172) 
  (h2 : Real.sqrt (a^2 + b^2 + c^2) = 21) : 
  2 * (a * b + b * c + c * a) = 1408 := by
sorry

end rectangular_box_surface_area_l3986_398674


namespace milk_bottles_remaining_l3986_398614

/-- Calculates the number of milk bottles remaining after purchases. -/
def remaining_bottles (initial : ℕ) (jason : ℕ) (harry_more : ℕ) : ℕ :=
  initial - (jason + (jason + harry_more))

/-- Theorem stating the number of remaining bottles in the given scenario. -/
theorem milk_bottles_remaining : remaining_bottles 35 5 6 = 24 := by
  sorry

end milk_bottles_remaining_l3986_398614


namespace square_perimeter_l3986_398673

theorem square_perimeter (area : ℝ) (side : ℝ) (h1 : area = 625) (h2 : side * side = area) :
  4 * side = 100 := by
  sorry

end square_perimeter_l3986_398673


namespace sum_reciprocals_l3986_398603

theorem sum_reciprocals (a b c d : ℝ) (ω : ℂ) : 
  a ≠ -1 → b ≠ -1 → c ≠ -1 → d ≠ -1 →
  ω^4 = 1 → ω ≠ 1 →
  1 / (a + ω) + 1 / (b + ω) + 1 / (c + ω) + 1 / (d + ω) = 3 / ω^2 →
  1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) + 1 / (d + 1) = 2 := by
sorry

end sum_reciprocals_l3986_398603


namespace exists_n_for_all_k_l3986_398685

theorem exists_n_for_all_k (k : ℕ) : ∃ n : ℕ, 
  Real.sqrt (n + 1981^k) + Real.sqrt n = (Real.sqrt 1982 + 1)^k := by
  sorry

end exists_n_for_all_k_l3986_398685


namespace square_difference_sum_l3986_398682

theorem square_difference_sum : 
  21^2 - 19^2 + 17^2 - 15^2 + 13^2 - 11^2 + 9^2 - 7^2 + 5^2 - 3^2 = 220 := by
  sorry

end square_difference_sum_l3986_398682


namespace sam_washing_pennies_l3986_398688

/-- The number of pennies Sam earned from washing clothes -/
def pennies_from_washing (total_cents : ℕ) (num_quarters : ℕ) : ℕ :=
  total_cents - (num_quarters * 25)

/-- Theorem: Given 7 quarters and a total of $1.84, Sam earned 9 pennies from washing clothes -/
theorem sam_washing_pennies :
  pennies_from_washing 184 7 = 9 := by
  sorry

end sam_washing_pennies_l3986_398688


namespace fourth_student_number_l3986_398655

def class_size : ℕ := 54
def sample_size : ℕ := 4

def systematic_sample (start : ℕ) : Fin 4 → ℕ :=
  λ i => (start + i.val * 13) % class_size + 1

theorem fourth_student_number (h : ∃ start : ℕ, 
  (systematic_sample start 1 = 3 ∧ 
   systematic_sample start 2 = 29 ∧ 
   systematic_sample start 3 = 42)) : 
  ∃ start : ℕ, systematic_sample start 0 = 44 := by
  sorry

end fourth_student_number_l3986_398655


namespace determinant_equals_four_l3986_398608

/-- The determinant of a 2x2 matrix [[a, b], [c, d]] is ad - bc. -/
def det2x2 (a b c d : ℝ) : ℝ := a * d - b * c

/-- The matrix in question, parameterized by x. -/
def matrix (x : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![3*x, 2],
    ![x, 2*x]]

theorem determinant_equals_four (x : ℝ) : 
  det2x2 (3*x) 2 x (2*x) = 4 ↔ x = -2/3 ∨ x = 1 := by sorry

end determinant_equals_four_l3986_398608


namespace bike_shop_wheels_l3986_398607

/-- The number of wheels on all vehicles in a bike shop -/
def total_wheels (num_bicycles num_tricycles : ℕ) : ℕ :=
  2 * num_bicycles + 3 * num_tricycles

/-- Theorem stating that the total number of wheels from 50 bicycles and 20 tricycles is 160 -/
theorem bike_shop_wheels :
  total_wheels 50 20 = 160 := by
  sorry

end bike_shop_wheels_l3986_398607


namespace y_investment_l3986_398617

/-- Represents the investment and profit share of a person in a business. -/
structure Investor where
  investment : ℕ
  profitShare : ℕ

/-- Represents the business with three investors. -/
structure Business where
  x : Investor
  y : Investor
  z : Investor

/-- The theorem stating that given the conditions, y's investment is 15000 rupees. -/
theorem y_investment (b : Business) : 
  b.x.investment = 5000 ∧ 
  b.z.investment = 7000 ∧ 
  b.x.profitShare = 2 ∧ 
  b.y.profitShare = 6 ∧ 
  b.z.profitShare = 7 → 
  b.y.investment = 15000 :=
by sorry

end y_investment_l3986_398617


namespace cube_equation_solution_l3986_398692

theorem cube_equation_solution (x : ℝ) : (x + 1)^3 = -27 → x = -4 := by
  sorry

end cube_equation_solution_l3986_398692


namespace robin_gum_total_l3986_398639

theorem robin_gum_total (initial : ℕ) (additional : ℕ) (total : ℕ) : 
  initial = 18 → additional = 26 → total = initial + additional → total = 44 := by
  sorry

end robin_gum_total_l3986_398639


namespace remainder_3_pow_2017_mod_17_l3986_398640

theorem remainder_3_pow_2017_mod_17 : 3^2017 % 17 = 3 := by
  sorry

end remainder_3_pow_2017_mod_17_l3986_398640


namespace interest_rate_difference_l3986_398664

/-- Proves that given a principal of $600 invested for 6 years, if the difference in interest earned between two rates is $144, then the difference between these two rates is 4%. -/
theorem interest_rate_difference (principal : ℝ) (time : ℝ) (interest_diff : ℝ) 
  (h1 : principal = 600)
  (h2 : time = 6)
  (h3 : interest_diff = 144) :
  ∃ (original_rate higher_rate : ℝ),
    (principal * time * higher_rate / 100 - principal * time * original_rate / 100 = interest_diff) ∧
    (higher_rate - original_rate = 4) := by
  sorry

end interest_rate_difference_l3986_398664


namespace sum_of_digits_greater_than_4_l3986_398604

def digits_of_735 : List Nat := [7, 3, 5]

def is_valid_card (n : Nat) : Prop := 1 ≤ n ∧ n ≤ 9

theorem sum_of_digits_greater_than_4 :
  (∀ d ∈ digits_of_735, is_valid_card d) →
  (List.sum (digits_of_735.filter (λ x => x > 4))) = 12 := by
  sorry

end sum_of_digits_greater_than_4_l3986_398604


namespace grid_squares_count_l3986_398641

/-- Represents a square grid --/
structure Grid :=
  (size : Nat)

/-- Counts the number of squares of a given size in the grid --/
def countSquares (g : Grid) (squareSize : Nat) : Nat :=
  (g.size + 1 - squareSize) * (g.size + 1 - squareSize)

/-- Calculates the total number of squares in the grid --/
def totalSquares (g : Grid) : Nat :=
  (countSquares g 1) + (countSquares g 2) + (countSquares g 3) + (countSquares g 4)

/-- Theorem stating that the total number of squares in a 5x5 grid is 54 --/
theorem grid_squares_count :
  let g : Grid := ⟨5⟩
  totalSquares g = 54 := by
  sorry

end grid_squares_count_l3986_398641


namespace quick_customer_sale_l3986_398623

def chicken_problem (initial_chickens neighbor_sale remaining_chickens : ℕ) : ℕ :=
  initial_chickens - neighbor_sale - remaining_chickens

theorem quick_customer_sale :
  chicken_problem 80 12 43 = 25 := by
  sorry

end quick_customer_sale_l3986_398623


namespace mindmaster_codes_l3986_398695

/-- The number of available colors in the Mindmaster game -/
def num_colors : ℕ := 8

/-- The number of slots in each code -/
def num_slots : ℕ := 4

/-- The total number of possible secret codes in the Mindmaster game -/
def total_codes : ℕ := num_colors ^ num_slots

/-- Theorem stating that the total number of secret codes is 4096 -/
theorem mindmaster_codes : total_codes = 4096 := by
  sorry

end mindmaster_codes_l3986_398695


namespace lcm_of_ratio_and_hcf_l3986_398643

theorem lcm_of_ratio_and_hcf (a b : ℕ) (h_ratio : a * 3 = b * 2) (h_hcf : Nat.gcd a b = 6) : Nat.lcm a b = 36 := by
  sorry

end lcm_of_ratio_and_hcf_l3986_398643


namespace double_up_polynomial_properties_l3986_398618

/-- A double-up polynomial is a quadratic polynomial with two real roots, one of which is twice the other. -/
def DoubleUpPolynomial (p q : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ (k^2 + p*k + q = 0) ∧ ((2*k)^2 + p*(2*k) + q = 0)

theorem double_up_polynomial_properties :
  (∀ p q : ℝ, DoubleUpPolynomial p q →
    (p = -15 → q = 50) ∧
    (∃ k : ℝ, (k = 4 ∨ k = 2) → p + q = 20 ∨ p + q = 2) ∧
    (p + q = 9 → ∃ k : ℝ, k = 3 ∨ k = -3/2)) := by
  sorry

end double_up_polynomial_properties_l3986_398618


namespace at_least_one_false_l3986_398605

theorem at_least_one_false (p q : Prop) : 
  ¬(p ∧ q) → (¬p ∨ ¬q) := by
  sorry

end at_least_one_false_l3986_398605


namespace oranges_left_l3986_398671

def initial_oranges : ℕ := 55
def oranges_taken : ℕ := 35

theorem oranges_left : initial_oranges - oranges_taken = 20 := by
  sorry

end oranges_left_l3986_398671


namespace rectangle_area_is_72_l3986_398636

/-- Represents a circle with a center point and radius -/
structure Circle where
  center : Point
  radius : ℝ

/-- Represents a rectangle with four vertices -/
structure Rectangle where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Checks if three circles are congruent -/
def areCongruent (c1 c2 c3 : Circle) : Prop :=
  c1.radius = c2.radius ∧ c2.radius = c3.radius

/-- Checks if a circle is tangent to all sides of a rectangle -/
def isTangentToRectangle (c : Circle) (r : Rectangle) : Prop :=
  sorry

/-- Checks if a circle passes through two points -/
def passesThrough (c : Circle) (p1 p2 : Point) : Prop :=
  sorry

/-- Calculates the area of a rectangle -/
def rectangleArea (r : Rectangle) : ℝ :=
  sorry

theorem rectangle_area_is_72 
  (ABCD : Rectangle) (P Q R : Point) (circleP circleQ circleR : Circle) :
  circleP.center = P →
  circleQ.center = Q →
  circleR.center = R →
  areCongruent circleP circleQ circleR →
  isTangentToRectangle circleP ABCD →
  isTangentToRectangle circleQ ABCD →
  isTangentToRectangle circleR ABCD →
  circleQ.radius = 3 →
  passesThrough circleQ P R →
  rectangleArea ABCD = 72 :=
by sorry

end rectangle_area_is_72_l3986_398636


namespace square_sum_given_diff_and_product_l3986_398624

theorem square_sum_given_diff_and_product (a b : ℝ) 
  (h1 : a - b = 2) 
  (h2 : a * b = 3) : 
  (a + b)^2 = 16 := by
sorry

end square_sum_given_diff_and_product_l3986_398624


namespace both_selected_probability_l3986_398602

theorem both_selected_probability 
  (p_ram : ℝ) (p_ravi : ℝ) 
  (h_ram : p_ram = 1 / 7) 
  (h_ravi : p_ravi = 1 / 5) 
  (h_independent : True) -- Assuming independence
  : p_ram * p_ravi = 1 / 35 := by
  sorry

end both_selected_probability_l3986_398602


namespace negative_less_than_positive_l3986_398647

theorem negative_less_than_positive : 
  (∀ x y : ℝ, x < 0 ∧ y > 0 → x < y) →
  -897 < 0.01 := by sorry

end negative_less_than_positive_l3986_398647


namespace bacon_calorie_percentage_example_l3986_398662

/-- The percentage of calories from bacon in a sandwich -/
def bacon_calorie_percentage (total_calories : ℕ) (bacon_strips : ℕ) (calories_per_strip : ℕ) : ℚ :=
  (bacon_strips * calories_per_strip : ℚ) / total_calories * 100

/-- Theorem stating that the percentage of calories from bacon in a 1250-calorie sandwich with two 125-calorie bacon strips is 20% -/
theorem bacon_calorie_percentage_example :
  bacon_calorie_percentage 1250 2 125 = 20 := by
  sorry

end bacon_calorie_percentage_example_l3986_398662


namespace total_voters_l3986_398649

/-- Given information about voters in three districts, prove the total number of voters. -/
theorem total_voters (d1 d2 d3 : ℕ) : 
  d1 = 322 →
  d3 = 2 * d1 →
  d2 = d3 - 19 →
  d1 + d2 + d3 = 1591 := by
sorry

end total_voters_l3986_398649


namespace sequence_2023rd_term_l3986_398697

theorem sequence_2023rd_term (a : ℕ → ℝ) 
  (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, (a n / 2) - (1 / (2 * a (n + 1))) = a (n + 1) - (1 / a n)) :
  a 2023 = 1 ∨ a 2023 = (1 / 2) ^ 2022 := by
  sorry

end sequence_2023rd_term_l3986_398697


namespace percentage_problem_l3986_398642

theorem percentage_problem (x : ℝ) : 200 = 4 * x → x = 50 := by
  sorry

end percentage_problem_l3986_398642


namespace tangent_line_slope_l3986_398656

/-- Given a curve y = ax + e^x - 1 and its tangent line y = 3x at (0,0), prove a = 2 -/
theorem tangent_line_slope (a : ℝ) : 
  (∀ x y : ℝ, y = a * x + Real.exp x - 1) →  -- Curve equation
  (∃ m : ℝ, m = 3 ∧ ∀ x y : ℝ, y = m * x) →  -- Tangent line equation
  (∀ ε > 0, ∃ δ > 0, ∀ x, |x| < δ → |((a * x + Real.exp x - 1) - 0) / (x - 0) - 3| < ε) →  -- Tangent condition
  a = 2 := by
  sorry

end tangent_line_slope_l3986_398656


namespace window_area_ratio_l3986_398606

/-- Proves that for a rectangle with semicircles at either end, where the ratio of the length (AD)
    to the width (AB) is 3:2 and the width is 30 inches, the ratio of the area of the rectangle
    to the combined area of the semicircles is 6:π. -/
theorem window_area_ratio :
  let AB : ℝ := 30
  let AD : ℝ := (3/2) * AB
  let rectangle_area : ℝ := AD * AB
  let semicircle_radius : ℝ := AB / 2
  let semicircles_area : ℝ := π * semicircle_radius^2
  rectangle_area / semicircles_area = 6 / π :=
by sorry

end window_area_ratio_l3986_398606


namespace marcel_shopping_cost_l3986_398620

def pen_cost : ℝ := 4

def briefcase_cost : ℝ := 5 * pen_cost

def notebook_cost : ℝ := 2 * pen_cost

def calculator_cost : ℝ := 3 * notebook_cost

def total_cost_before_tax : ℝ := pen_cost + briefcase_cost + notebook_cost + calculator_cost

def tax_rate : ℝ := 0.1

def tax_amount : ℝ := tax_rate * total_cost_before_tax

def total_cost_with_tax : ℝ := total_cost_before_tax + tax_amount

theorem marcel_shopping_cost : total_cost_with_tax = 61.60 := by sorry

end marcel_shopping_cost_l3986_398620


namespace correct_factorization_l3986_398625

theorem correct_factorization (x : ℝ) : -x^2 + 2*x - 1 = -(x - 1)^2 := by
  sorry

end correct_factorization_l3986_398625


namespace probability_one_male_one_female_l3986_398610

/-- The probability of selecting exactly one male and one female student
    when randomly choosing two students from a group of four students
    (one male and three female) --/
theorem probability_one_male_one_female :
  let total_students : ℕ := 4
  let male_students : ℕ := 1
  let female_students : ℕ := 3
  let selected_students : ℕ := 2
  let ways_to_select_one_male_one_female : ℕ := male_students * female_students
  let total_ways_to_select : ℕ := Nat.choose total_students selected_students
  (ways_to_select_one_male_one_female : ℚ) / total_ways_to_select = 1 / 2 :=
by sorry

end probability_one_male_one_female_l3986_398610


namespace rahul_deepak_age_ratio_l3986_398651

theorem rahul_deepak_age_ratio :
  ∀ (rahul_age deepak_age : ℕ),
    deepak_age = 12 →
    rahul_age + 6 = 22 →
    rahul_age / deepak_age = 4 / 3 := by
  sorry

end rahul_deepak_age_ratio_l3986_398651


namespace coefficient_x2y4_in_expansion_l3986_398661

/-- The coefficient of x^2y^4 in the expansion of (1+x+y^2)^5 is 30 -/
theorem coefficient_x2y4_in_expansion : ℕ := by
  sorry

end coefficient_x2y4_in_expansion_l3986_398661


namespace other_root_of_quadratic_l3986_398693

theorem other_root_of_quadratic (m : ℝ) : 
  (2^2 - 5*2 - m = 0) → 
  ∃ (t : ℝ), t ≠ 2 ∧ t^2 - 5*t - m = 0 ∧ t = 3 :=
by sorry

end other_root_of_quadratic_l3986_398693


namespace inequality_proof_l3986_398621

theorem inequality_proof (a b c : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0) (h4 : a * b * c = 1) :
  (a^2 + 1) * (b^3 + 2) * (c^6 + 5) ≥ 36 := by
  sorry

end inequality_proof_l3986_398621


namespace geometric_series_equation_l3986_398644

theorem geometric_series_equation (x : ℝ) : x = 9 →
  (∑' n, (1/3)^n) * (∑' n, (-1/3)^n) = ∑' n, (1/x)^n := by
  sorry

end geometric_series_equation_l3986_398644


namespace min_p_value_l3986_398611

theorem min_p_value (p q : ℕ+) 
  (h1 : (2008 : ℚ) / 2009 < p / q)
  (h2 : p / q < (2009 : ℚ) / 2010) : 
  (∀ p' q' : ℕ+, (2008 : ℚ) / 2009 < p' / q' → p' / q' < (2009 : ℚ) / 2010 → p ≤ p') → 
  p = 4017 := by
sorry

end min_p_value_l3986_398611


namespace pirate_treasure_probability_l3986_398619

theorem pirate_treasure_probability :
  let n : ℕ := 8
  let p_treasure : ℚ := 1/3
  let p_trap : ℚ := 1/6
  let p_empty : ℚ := 1/2
  let k : ℕ := 4
  p_treasure + p_trap + p_empty = 1 →
  (n.choose k : ℚ) * p_treasure^k * p_empty^(n-k) = 35/648 :=
by sorry

end pirate_treasure_probability_l3986_398619


namespace minimum_duty_days_l3986_398622

theorem minimum_duty_days (total_members : ℕ) (duty_size_1 duty_size_2 : ℕ) :
  total_members = 33 →
  duty_size_1 = 9 →
  duty_size_2 = 10 →
  ∃ (k n m : ℕ), 
    k + n = 7 ∧ 
    duty_size_1 * k + duty_size_2 * n = total_members * m ∧
    ∀ (k' n' m' : ℕ), 
      k' + n' < 7 → 
      duty_size_1 * k' + duty_size_2 * n' ≠ total_members * m' :=
by sorry

end minimum_duty_days_l3986_398622


namespace rhombus_diagonal_l3986_398690

/-- Proves that in a rhombus with one diagonal of 160 m and an area of 5600 m², 
    the length of the other diagonal is 70 m. -/
theorem rhombus_diagonal (d1 : ℝ) (area : ℝ) (d2 : ℝ) : 
  d1 = 160 → area = 5600 → area = (d1 * d2) / 2 → d2 = 70 := by
  sorry

end rhombus_diagonal_l3986_398690


namespace quadratic_solution_difference_l3986_398633

theorem quadratic_solution_difference : ∃ (x₁ x₂ : ℝ), 
  (x₁^2 - 5*x₁ + 15 = x₁ + 55) ∧ 
  (x₂^2 - 5*x₂ + 15 = x₂ + 55) ∧ 
  (x₁ ≠ x₂) ∧
  (|x₁ - x₂| = 14) := by
sorry

end quadratic_solution_difference_l3986_398633


namespace infinite_log_3_64_equals_4_l3986_398687

noncomputable def log_3 (x : ℝ) : ℝ := Real.log x / Real.log 3

theorem infinite_log_3_64_equals_4 :
  ∃! x : ℝ, x > 0 ∧ x = log_3 (64 + x) ∧ x = 4 := by sorry

end infinite_log_3_64_equals_4_l3986_398687


namespace lukas_average_points_l3986_398665

/-- Given a basketball player's total points and number of games, 
    calculate their average points per game. -/
def average_points_per_game (total_points : ℕ) (num_games : ℕ) : ℚ :=
  (total_points : ℚ) / (num_games : ℚ)

/-- Theorem: A player who scores 60 points in 5 games averages 12 points per game. -/
theorem lukas_average_points : 
  average_points_per_game 60 5 = 12 := by
  sorry

end lukas_average_points_l3986_398665


namespace hyperbola_eccentricity_l3986_398686

theorem hyperbola_eccentricity (a b c : ℝ) (h1 : b^2 / a^2 = 1) (h2 : c^2 = a^2 + b^2) :
  c / a = Real.sqrt 2 :=
sorry

end hyperbola_eccentricity_l3986_398686


namespace batsman_matches_proof_l3986_398609

theorem batsman_matches_proof (first_matches : Nat) (second_matches : Nat) 
  (first_average : Nat) (second_average : Nat) (overall_average : Nat) :
  first_matches = 30 ∧ 
  second_matches = 15 ∧ 
  first_average = 50 ∧ 
  second_average = 26 ∧ 
  overall_average = 42 →
  first_matches + second_matches = 45 := by
  sorry

end batsman_matches_proof_l3986_398609


namespace opposite_sqrt_81_l3986_398668

theorem opposite_sqrt_81 : -(Real.sqrt (Real.sqrt 81)) = -9 := by
  sorry

end opposite_sqrt_81_l3986_398668


namespace common_tangent_sum_l3986_398679

/-- Given two curves f and g with a common tangent at their intersection point (0, m),
    prove that the sum of their coefficients a and b is 1. -/
theorem common_tangent_sum (a b m : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ a * Real.cos x
  let g : ℝ → ℝ := λ x ↦ x^2 + b*x + 1
  let f' : ℝ → ℝ := λ x ↦ -a * Real.sin x
  let g' : ℝ → ℝ := λ x ↦ 2*x + b
  (f 0 = g 0) ∧ (f' 0 = g' 0) → a + b = 1 := by
sorry

end common_tangent_sum_l3986_398679


namespace sum_with_radical_conjugate_l3986_398650

theorem sum_with_radical_conjugate : 
  let x : ℝ := 12 - Real.sqrt 5000
  let y : ℝ := 12 + Real.sqrt 5000  -- radical conjugate
  x + y = 24 := by
sorry

end sum_with_radical_conjugate_l3986_398650


namespace intersection_condition_l3986_398613

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 ≥ p.1^2}
def N (a : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + (p.2 - a)^2 ≤ 1}

-- State the theorem
theorem intersection_condition (a : ℝ) : M ∩ N a = N a ↔ a ≥ 5/4 := by
  sorry

end intersection_condition_l3986_398613


namespace necessary_condition_l3986_398670

theorem necessary_condition (p q : Prop) 
  (h : p → q) : ¬q → ¬p := by sorry

end necessary_condition_l3986_398670


namespace existsShapeWithCircularTopView_multipleShapesWithCircularTopView_l3986_398627

-- Define a type for 3D shapes
inductive Shape3D
  | Sphere
  | Cylinder
  | Cone
  | Frustum
  | Other

-- Define a property for having a circular top view
def hasCircularTopView (s : Shape3D) : Prop :=
  match s with
  | Shape3D.Sphere => True
  | Shape3D.Cylinder => True
  | Shape3D.Cone => True
  | Shape3D.Frustum => True
  | Shape3D.Other => False

-- Theorem stating that there exist shapes with circular top views
theorem existsShapeWithCircularTopView : ∃ (s : Shape3D), hasCircularTopView s :=
  sorry

-- Theorem stating that multiple shapes can have circular top views
theorem multipleShapesWithCircularTopView :
  ∃ (s1 s2 : Shape3D), s1 ≠ s2 ∧ hasCircularTopView s1 ∧ hasCircularTopView s2 :=
  sorry

end existsShapeWithCircularTopView_multipleShapesWithCircularTopView_l3986_398627


namespace draw_balls_theorem_l3986_398683

/-- The number of ways to draw balls from a bag under specific conditions -/
def draw_balls_count (total_white : ℕ) (total_red : ℕ) (total_black : ℕ) 
                     (draw_count : ℕ) (min_white : ℕ) (max_white : ℕ) 
                     (min_red : ℕ) (max_red : ℕ) (max_black : ℕ) : ℕ :=
  sorry

/-- Theorem stating the number of ways to draw balls under given conditions -/
theorem draw_balls_theorem : 
  draw_balls_count 9 5 6 10 3 7 2 5 3 = 14 := by
  sorry

end draw_balls_theorem_l3986_398683


namespace addition_is_unique_solution_l3986_398645

-- Define the possible operations
inductive Operation
  | Add
  | Sub
  | Mul
  | Div

-- Define a function to apply the operation
def applyOperation (op : Operation) (a b : Int) : Int :=
  match op with
  | Operation.Add => a + b
  | Operation.Sub => a - b
  | Operation.Mul => a * b
  | Operation.Div => a / b

-- Theorem statement
theorem addition_is_unique_solution :
  ∃! op : Operation, applyOperation op 7 (-7) = 0 ∧ 
  (op = Operation.Add ∨ op = Operation.Sub ∨ op = Operation.Mul ∨ op = Operation.Div) :=
by sorry

end addition_is_unique_solution_l3986_398645


namespace equation_solution_l3986_398628

theorem equation_solution :
  let f : ℝ → ℝ := λ x => (20 / (x^2 - 9)) + (5 / (x - 3))
  ∀ x : ℝ, f x = 2 ↔ x = (5 + Real.sqrt 449) / 4 ∨ x = (5 - Real.sqrt 449) / 4 :=
by sorry

end equation_solution_l3986_398628


namespace favorite_fruit_pears_l3986_398659

theorem favorite_fruit_pears (total students_oranges students_apples students_strawberries : ℕ) 
  (h1 : total = 450)
  (h2 : students_oranges = 70)
  (h3 : students_apples = 147)
  (h4 : students_strawberries = 113) :
  total - (students_oranges + students_apples + students_strawberries) = 120 := by
  sorry

end favorite_fruit_pears_l3986_398659


namespace jessica_shells_count_l3986_398684

def seashell_problem (sally_shells tom_shells total_shells : ℕ) : Prop :=
  ∃ jessica_shells : ℕ, 
    sally_shells + tom_shells + jessica_shells = total_shells

theorem jessica_shells_count (sally_shells tom_shells total_shells : ℕ) 
  (h : seashell_problem sally_shells tom_shells total_shells) :
  ∃ jessica_shells : ℕ, jessica_shells = total_shells - (sally_shells + tom_shells) :=
by
  sorry

#check jessica_shells_count 9 7 21

end jessica_shells_count_l3986_398684


namespace monthly_income_problem_l3986_398612

/-- Given the average monthly incomes of three people, prove the monthly income of one person. -/
theorem monthly_income_problem (income_AB income_BC income_AC : ℚ) 
  (h1 : income_AB = 4050)
  (h2 : income_BC = 5250)
  (h3 : income_AC = 4200) :
  ∃ (A B C : ℚ), 
    (A + B) / 2 = income_AB ∧
    (B + C) / 2 = income_BC ∧
    (A + C) / 2 = income_AC ∧
    A = 3000 := by
  sorry

end monthly_income_problem_l3986_398612


namespace units_digit_17_pow_2023_l3986_398666

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The result of raising a number to a power, considering only the units digit -/
def powerMod10 (base : ℕ) (exp : ℕ) : ℕ :=
  (base ^ exp) % 10

theorem units_digit_17_pow_2023 :
  unitsDigit (powerMod10 17 2023) = 3 :=
by sorry

end units_digit_17_pow_2023_l3986_398666


namespace geometric_series_common_ratio_l3986_398681

theorem geometric_series_common_ratio : 
  let a₁ : ℚ := 4/7
  let a₂ : ℚ := 16/49
  let a₃ : ℚ := 64/343
  (a₂ / a₁ = 4/7) ∧ (a₃ / a₂ = 4/7) → 
  ∃ (r : ℚ), ∀ (n : ℕ), n ≥ 1 → 
    (4/7) * (4/7)^(n-1) = (4/7) * r^(n-1) :=
by sorry

end geometric_series_common_ratio_l3986_398681


namespace mens_wages_are_fifty_l3986_398629

/-- Represents the wages of a group given the number of individuals and their equality relationships -/
def group_wages (men women boys : ℕ) (total_earnings : ℚ) : ℚ :=
  total_earnings / (men + women + boys : ℚ) * men

/-- Theorem stating that under given conditions, the men's wages are 50 -/
theorem mens_wages_are_fifty
  (men : ℕ) (women : ℕ) (boys : ℕ) (total_earnings : ℚ)
  (h1 : men = 5)
  (h2 : boys = 8)
  (h3 : men = women)  -- 5 men are equal to W women
  (h4 : women = boys) -- W women are equal to 8 boys
  (h5 : total_earnings = 150) :
  group_wages men women boys total_earnings = 50 := by
  sorry

#eval group_wages 5 5 8 150

end mens_wages_are_fifty_l3986_398629


namespace coin_problem_l3986_398631

theorem coin_problem (total_coins : ℕ) (total_value : ℚ) 
  (h_total_coins : total_coins = 336)
  (h_total_value : total_value = 71)
  : ∃ (coins_20p coins_25p : ℕ),
    coins_20p + coins_25p = total_coins ∧
    (20 : ℚ)/100 * coins_20p + (25 : ℚ)/100 * coins_25p = total_value ∧
    coins_20p = 260 := by
  sorry

end coin_problem_l3986_398631


namespace odot_inequality_range_l3986_398648

-- Define the operation ⊙
def odot (x y : ℝ) : ℝ := x * (1 - y)

-- State the theorem
theorem odot_inequality_range (a : ℝ) :
  (∀ x : ℝ, odot (x - a) (x + a) < 1) ↔ -1/2 < a ∧ a < 3/2 :=
by sorry

end odot_inequality_range_l3986_398648


namespace pirate_catch_caravel_l3986_398678

/-- Represents the velocity of a ship in nautical miles per hour -/
structure Velocity where
  speed : ℝ
  angle : ℝ

/-- Represents the position of a ship -/
structure Position where
  x : ℝ
  y : ℝ

/-- Calculate the minimum speed required for the pirate ship to catch the caravel -/
def min_pirate_speed (initial_distance : ℝ) (caravel_velocity : Velocity) : ℝ :=
  sorry

theorem pirate_catch_caravel (initial_distance : ℝ) (caravel_velocity : Velocity) :
  initial_distance = 10 ∧
  caravel_velocity.speed = 12 ∧
  caravel_velocity.angle = -5 * π / 6 →
  min_pirate_speed initial_distance caravel_velocity = 6 * Real.sqrt 6 :=
sorry

end pirate_catch_caravel_l3986_398678


namespace quadratic_form_sum_l3986_398698

theorem quadratic_form_sum (a h k : ℝ) : 
  (∀ x, 2 * x^2 + 8 * x + 6 = a * (x - h)^2 + k) → a + h + k = -2 := by
  sorry

end quadratic_form_sum_l3986_398698


namespace max_value_of_sin_cos_product_l3986_398637

theorem max_value_of_sin_cos_product (f : ℝ → ℝ) (α : ℝ) :
  (∀ x, f x = Real.sin (x + α) * Real.cos (x + α)) →
  (∀ x, f x ≤ f 1) →
  ∃ k : ℤ, α = Real.pi / 4 + k * Real.pi / 2 - 1 :=
by sorry

end max_value_of_sin_cos_product_l3986_398637


namespace eliza_siblings_l3986_398691

/-- The number of Eliza's siblings -/
def num_siblings : ℕ := 4

/-- The height of the tallest sibling -/
def tallest_sibling_height : ℕ := 70

/-- Eliza's height -/
def eliza_height : ℕ := tallest_sibling_height - 2

/-- The total height of all siblings including Eliza -/
def total_height : ℕ := 330

theorem eliza_siblings :
  (2 * 66 + 60 + tallest_sibling_height + eliza_height = total_height) →
  (num_siblings = 4) := by sorry

end eliza_siblings_l3986_398691


namespace weight_of_A_l3986_398632

/-- Prove that given the conditions, the weight of A is 79 kg -/
theorem weight_of_A (a b c d e : ℝ) : 
  (a + b + c) / 3 = 84 →
  (a + b + c + d) / 4 = 80 →
  e = d + 7 →
  (b + c + d + e) / 4 = 79 →
  a = 79 := by
sorry

end weight_of_A_l3986_398632


namespace comic_book_stacking_order_l3986_398694

theorem comic_book_stacking_order : 
  let spiderman := 6
  let archie := 5
  let garfield := 4
  let batman := 3
  let superman := 2
  let total_groups := 5
  (spiderman.factorial * archie.factorial * garfield.factorial * batman.factorial * superman.factorial * total_groups.factorial : ℕ) = 1492992000 := by
  sorry

end comic_book_stacking_order_l3986_398694


namespace pine_seedlings_in_sample_l3986_398653

/-- Represents a forest with seedlings -/
structure Forest where
  total_seedlings : ℕ
  pine_seedlings : ℕ
  sample_size : ℕ

/-- Calculates the expected number of pine seedlings in a sample -/
def expected_pine_seedlings (f : Forest) : ℚ :=
  (f.pine_seedlings : ℚ) * (f.sample_size : ℚ) / (f.total_seedlings : ℚ)

/-- Theorem stating the expected number of pine seedlings in the sample -/
theorem pine_seedlings_in_sample (f : Forest) 
  (h1 : f.total_seedlings = 30000)
  (h2 : f.pine_seedlings = 4000)
  (h3 : f.sample_size = 150) :
  expected_pine_seedlings f = 20 := by
  sorry

end pine_seedlings_in_sample_l3986_398653


namespace average_apples_picked_l3986_398630

theorem average_apples_picked (maggie kelsey layla : ℕ) (h1 : maggie = 40) (h2 : kelsey = 28) (h3 : layla = 22) :
  (maggie + kelsey + layla) / 3 = 30 := by
  sorry

end average_apples_picked_l3986_398630


namespace smallest_product_l3986_398676

def S : Finset Int := {-9, -7, -4, 2, 5, 7}

theorem smallest_product (a b : Int) :
  a ∈ S → b ∈ S → a * b ≥ -63 ∧ ∃ (x y : Int), x ∈ S ∧ y ∈ S ∧ x * y = -63 := by
  sorry

end smallest_product_l3986_398676

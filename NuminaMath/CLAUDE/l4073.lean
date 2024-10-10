import Mathlib

namespace product_equals_eighteen_l4073_407343

theorem product_equals_eighteen : 12 * 0.5 * 3 * 0.2 * 5 = 18 := by
  sorry

end product_equals_eighteen_l4073_407343


namespace remainder_7623_div_11_l4073_407379

theorem remainder_7623_div_11 : 7623 % 11 = 0 := by sorry

end remainder_7623_div_11_l4073_407379


namespace problem_solving_probability_l4073_407350

theorem problem_solving_probability (prob_a prob_b : ℝ) 
  (h_a : prob_a = 1/2)
  (h_b : prob_b = 1/3) :
  (1 - prob_a) * (1 - prob_b) = 1/3 := by
  sorry

end problem_solving_probability_l4073_407350


namespace triangle_angles_l4073_407337

theorem triangle_angles (a b c : ℝ) (h1 : a = 3) (h2 : b = Real.sqrt 8) (h3 : c = 2 + Real.sqrt 2) :
  ∃ (θ φ ψ : ℝ),
    Real.cos θ = (10 + Real.sqrt 2) / 18 ∧
    Real.cos φ = (11 - 4 * Real.sqrt 2) / (12 * Real.sqrt 2) ∧
    θ + φ + ψ = Real.pi := by
  sorry

end triangle_angles_l4073_407337


namespace min_positive_temperatures_l4073_407340

theorem min_positive_temperatures (n : ℕ) (pos_products neg_products : ℕ) : 
  n = 12 → 
  pos_products = 78 → 
  neg_products = 54 → 
  pos_products + neg_products = n * (n - 1) →
  ∃ y : ℕ, y ≥ 3 ∧ y * (y - 1) + (n - y) * (n - 1 - y) = pos_products ∧
  ∀ z : ℕ, z < 3 → z * (z - 1) + (n - z) * (n - 1 - z) ≠ pos_products :=
by sorry

end min_positive_temperatures_l4073_407340


namespace smallest_number_l4073_407306

theorem smallest_number (a b c d : ℝ) :
  a = 1 ∧ b = 0 ∧ c = -Real.sqrt 3 ∧ d = -Real.sqrt 2 →
  c ≤ a ∧ c ≤ b ∧ c ≤ d :=
by sorry

end smallest_number_l4073_407306


namespace total_students_l4073_407366

/-- Represents the number of students in each grade --/
def Students := Fin 8 → ℕ

/-- The total number of students in grades I-IV is 130 --/
def sum_I_to_IV (s : Students) : Prop :=
  s 0 + s 1 + s 2 + s 3 = 130

/-- Grade V has 7 more students than grade II --/
def grade_V_condition (s : Students) : Prop :=
  s 4 = s 1 + 7

/-- Grade VI has 5 fewer students than grade I --/
def grade_VI_condition (s : Students) : Prop :=
  s 5 = s 0 - 5

/-- Grade VII has 10 more students than grade IV --/
def grade_VII_condition (s : Students) : Prop :=
  s 6 = s 3 + 10

/-- Grade VIII has 4 fewer students than grade I --/
def grade_VIII_condition (s : Students) : Prop :=
  s 7 = s 0 - 4

/-- The theorem stating that the total number of students is 268 --/
theorem total_students (s : Students)
  (h1 : sum_I_to_IV s)
  (h2 : grade_V_condition s)
  (h3 : grade_VI_condition s)
  (h4 : grade_VII_condition s)
  (h5 : grade_VIII_condition s) :
  s 0 + s 1 + s 2 + s 3 + s 4 + s 5 + s 6 + s 7 = 268 := by
  sorry

end total_students_l4073_407366


namespace cross_number_puzzle_l4073_407347

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def is_power_of_3 (n : ℕ) : Prop := ∃ m : ℕ, n = 3^m

def is_power_of_7 (n : ℕ) : Prop := ∃ m : ℕ, n = 7^m

def digit_in_number (d : ℕ) (n : ℕ) : Prop :=
  ∃ a b : ℕ, n = a * 10 + d + b * 100 ∧ d < 10

theorem cross_number_puzzle :
  ∃! d : ℕ, 
    (∃ n : ℕ, is_three_digit n ∧ is_power_of_3 n ∧ digit_in_number d n) ∧
    (∃ m : ℕ, is_three_digit m ∧ is_power_of_7 m ∧ digit_in_number d m) ∧
    d = 4 :=
sorry

end cross_number_puzzle_l4073_407347


namespace magnitude_of_complex_fraction_l4073_407369

theorem magnitude_of_complex_fraction (i : ℂ) (h : i ^ 2 = -1) :
  Complex.abs (i / (2 - i)) = Real.sqrt 5 / 5 := by
  sorry

end magnitude_of_complex_fraction_l4073_407369


namespace tank_capacity_l4073_407329

/-- Represents a water tank with a given capacity --/
structure WaterTank where
  capacity : ℚ
  initial_fill : ℚ
  final_fill : ℚ
  added_water : ℚ

/-- Theorem stating the capacity of the tank given the conditions --/
theorem tank_capacity (tank : WaterTank)
  (h1 : tank.initial_fill = 3 / 4)
  (h2 : tank.final_fill = 7 / 8)
  (h3 : tank.added_water = 5)
  (h4 : tank.initial_fill * tank.capacity + tank.added_water = tank.final_fill * tank.capacity) :
  tank.capacity = 40 := by
  sorry

end tank_capacity_l4073_407329


namespace father_and_xiaolin_ages_l4073_407395

theorem father_and_xiaolin_ages :
  ∀ (f x : ℕ),
  f = 11 * x →
  f + 7 = 4 * (x + 7) →
  f = 33 ∧ x = 3 := by
sorry

end father_and_xiaolin_ages_l4073_407395


namespace julia_total_kids_l4073_407309

/-- The total number of kids Julia played or interacted with during the week -/
def total_kids : ℕ :=
  let monday_tag := 7
  let tuesday_tag := 13
  let thursday_tag := 18
  let wednesday_cards := 20
  let wednesday_hide_seek := 11
  let wednesday_puzzle := 9
  let friday_board_game := 15
  let friday_drawing := 12
  monday_tag + tuesday_tag + thursday_tag + wednesday_cards + wednesday_hide_seek + wednesday_puzzle + friday_board_game + friday_drawing

theorem julia_total_kids : total_kids = 105 := by
  sorry

end julia_total_kids_l4073_407309


namespace part_I_part_II_l4073_407373

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x * |x - 2*a| + a^2 - 4*a

-- Part I
theorem part_I :
  let f_neg_one (x : ℝ) := x * |x + 2| + 5
  ∃ (min max : ℝ), min = 2 ∧ max = 5 ∧
    (∀ x ∈ Set.Icc (-3) 0, f_neg_one x ≥ min ∧ f_neg_one x ≤ max) ∧
    (∃ x₁ ∈ Set.Icc (-3) 0, f_neg_one x₁ = min) ∧
    (∃ x₂ ∈ Set.Icc (-3) 0, f_neg_one x₂ = max) :=
sorry

-- Part II
theorem part_II :
  ∀ a : ℝ, (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0) →
  (∀ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0 →
    (1 + Real.sqrt 2) / 2 < 1 / x₁ + 1 / x₂ + 1 / x₃) :=
sorry

end part_I_part_II_l4073_407373


namespace rectangle_mn_value_l4073_407323

-- Define the rectangle ABCD
def Rectangle (AB BC : ℝ) : Prop :=
  AB > 0 ∧ BC > 0

-- Define the perimeter of the rectangle
def Perimeter (AB BC : ℝ) : ℝ :=
  2 * (AB + BC)

-- Define the area of the rectangle
def Area (AB BC : ℝ) : ℝ :=
  AB * BC

-- Define the quadratic equation
def QuadraticRoots (m n : ℝ) (x y : ℝ) : Prop :=
  x^2 + m*x + n = 0 ∧ y^2 + m*y + n = 0

-- State the theorem
theorem rectangle_mn_value (AB BC m n : ℝ) :
  Rectangle AB BC →
  Perimeter AB BC = 12 →
  Area AB BC = 5 →
  QuadraticRoots m n AB BC →
  m * n = -30 := by
  sorry

end rectangle_mn_value_l4073_407323


namespace arithmetic_mean_problem_l4073_407335

theorem arithmetic_mean_problem (p q r : ℝ) : 
  (p + q) / 2 = 10 →
  (q + r) / 2 = 27 →
  r - p = 34 →
  (q + r) / 2 = 27 := by
sorry

end arithmetic_mean_problem_l4073_407335


namespace correct_average_marks_l4073_407376

/-- Calculates the correct average marks for a class given the reported average, 
    number of students, and corrections for three students' marks. -/
def correctAverageMarks (reportedAverage : ℚ) (numStudents : ℕ) 
    (wrongMark1 wrongMark2 wrongMark3 : ℚ) 
    (correctMark1 correctMark2 correctMark3 : ℚ) : ℚ :=
  let incorrectTotal := reportedAverage * numStudents
  let wronglyNotedMarks := wrongMark1 + wrongMark2 + wrongMark3
  let correctMarks := correctMark1 + correctMark2 + correctMark3
  let correctTotal := incorrectTotal - wronglyNotedMarks + correctMarks
  correctTotal / numStudents

/-- The correct average marks for the class are 63.125 -/
theorem correct_average_marks :
  correctAverageMarks 65 40 100 85 15 20 50 55 = 63.125 := by
  sorry

end correct_average_marks_l4073_407376


namespace acute_triangle_cotangent_sum_range_l4073_407338

theorem acute_triangle_cotangent_sum_range (A B C : Real) (a b c : Real) :
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Acute triangle condition
  A + B + C = π ∧  -- Sum of angles in a triangle
  a > 0 ∧ b > 0 ∧ c > 0 ∧  -- Positive side lengths
  b^2 - a^2 = a * c →  -- Given condition
  1 < 1 / Real.tan A + 1 / Real.tan B ∧ 
  1 / Real.tan A + 1 / Real.tan B < 2 * Real.sqrt 3 / 3 := by
  sorry

end acute_triangle_cotangent_sum_range_l4073_407338


namespace pet_shop_hamsters_l4073_407393

theorem pet_shop_hamsters (total : ℕ) (kittens : ℕ) (birds : ℕ) 
  (h1 : total = 77)
  (h2 : kittens = 32)
  (h3 : birds = 30)
  : total - kittens - birds = 15 := by
  sorry

end pet_shop_hamsters_l4073_407393


namespace tobacco_acreage_increase_l4073_407320

/-- Calculates the increase in tobacco acreage when changing crop ratios -/
theorem tobacco_acreage_increase (total_land : ℝ) (initial_ratio_tobacco : ℝ) 
  (initial_ratio_total : ℝ) (new_ratio_tobacco : ℝ) (new_ratio_total : ℝ) :
  total_land = 1350 ∧ 
  initial_ratio_tobacco = 2 ∧ 
  initial_ratio_total = 9 ∧ 
  new_ratio_tobacco = 5 ∧ 
  new_ratio_total = 9 →
  (new_ratio_tobacco / new_ratio_total - initial_ratio_tobacco / initial_ratio_total) * total_land = 450 :=
by sorry

end tobacco_acreage_increase_l4073_407320


namespace symmetry_about_y_axis_l4073_407336

/-- Given two real numbers a and b such that log(a) + log(b) = 0, a ≠ 1, and b ≠ 1,
    prove that the functions f(x) = ax and g(x) = bx are symmetric about the y-axis. -/
theorem symmetry_about_y_axis 
  (a b : ℝ) 
  (h1 : Real.log a + Real.log b = 0) 
  (h2 : a ≠ 1) 
  (h3 : b ≠ 1) : 
  ∀ x : ℝ, ∃ y : ℝ, a * x = b * (-y) ∧ a * (-x) = b * y :=
sorry

end symmetry_about_y_axis_l4073_407336


namespace intersection_sum_modulo13_l4073_407377

theorem intersection_sum_modulo13 : 
  ∃ (x : ℤ), 0 ≤ x ∧ x < 13 ∧ 
  (∃ (y : ℤ), y ≡ 3*x + 4 [ZMOD 13] ∧ y ≡ 8*x + 9 [ZMOD 13]) ∧
  (∀ (x' : ℤ), 0 ≤ x' ∧ x' < 13 → 
    (∃ (y' : ℤ), y' ≡ 3*x' + 4 [ZMOD 13] ∧ y' ≡ 8*x' + 9 [ZMOD 13]) → 
    x' = x) ∧
  x = 6 := by
sorry

end intersection_sum_modulo13_l4073_407377


namespace same_solution_implies_a_b_values_l4073_407326

theorem same_solution_implies_a_b_values :
  ∀ (a b x y : ℚ),
  (3 * x - y = 7 ∧ a * x + y = b) ∧
  (x + b * y = a ∧ 2 * x + y = 8) →
  a = -7/5 ∧ b = -11/5 := by
sorry

end same_solution_implies_a_b_values_l4073_407326


namespace locus_equation_l4073_407382

/-- Parabola type representing y^2 = 4px -/
structure Parabola where
  p : ℝ
  hp : p > 0

/-- Point on a parabola -/
structure ParabolaPoint (par : Parabola) where
  y : ℝ
  eq : y^2 = 4 * par.p * (y^2 / (4 * par.p))

/-- The locus of point M given two points on a parabola -/
def locusM (par : Parabola) (A B : ParabolaPoint par) (M : ℝ × ℝ) : Prop :=
  let OA := (A.y^2 / (4 * par.p), A.y)
  let OB := (B.y^2 / (4 * par.p), B.y)
  let (x, y) := M
  (OA.1 * OB.1 + OA.2 * OB.2 = 0) ∧  -- OA ⊥ OB
  (x * (B.y^2 - A.y^2) / (4 * par.p) + y * (B.y - A.y) = 0) ∧  -- OM ⊥ AB
  (x - A.y^2 / (4 * par.p)) * (B.y - A.y) = 
    ((B.y^2 - A.y^2) / (4 * par.p)) * (y - A.y)  -- M is on line AB

theorem locus_equation (par : Parabola) (A B : ParabolaPoint par) (M : ℝ × ℝ) :
  locusM par A B M → M.1^2 + M.2^2 - 4 * par.p * M.1 = 0 :=
by sorry

end locus_equation_l4073_407382


namespace expand_and_complete_square_l4073_407324

theorem expand_and_complete_square (x : ℝ) : 
  -2 * (x - 3) * (x + 1/2) = -2 * (x - 5/4)^2 + 49/8 := by
  sorry

end expand_and_complete_square_l4073_407324


namespace q_div_p_eq_fifty_l4073_407368

/-- The number of cards in the box -/
def total_cards : ℕ := 30

/-- The number of different numbers on the cards -/
def num_types : ℕ := 6

/-- The number of cards for each number -/
def cards_per_num : ℕ := 5

/-- The number of cards drawn -/
def drawn_cards : ℕ := 4

/-- The probability of drawing four cards with the same number -/
def p : ℚ := (num_types * (cards_per_num.choose drawn_cards)) / (total_cards.choose drawn_cards)

/-- The probability of drawing two pairs of cards with different numbers -/
def q : ℚ := (num_types.choose 2 * (cards_per_num.choose 2)^2) / (total_cards.choose drawn_cards)

/-- The theorem stating that the ratio of q to p is 50 -/
theorem q_div_p_eq_fifty : q / p = 50 := by sorry

end q_div_p_eq_fifty_l4073_407368


namespace gift_price_gift_price_exact_l4073_407313

/-- The price of Lisa's gift given her savings and contributions from family and friends --/
theorem gift_price (lisa_savings : ℚ) (mother_fraction : ℚ) (brother_multiplier : ℚ) 
  (friend_fraction : ℚ) (short_amount : ℚ) : ℚ :=
  let mother_contribution := mother_fraction * lisa_savings
  let brother_contribution := brother_multiplier * mother_contribution
  let friend_contribution := friend_fraction * (mother_contribution + brother_contribution)
  let total_contributions := lisa_savings + mother_contribution + brother_contribution + friend_contribution
  total_contributions + short_amount

/-- The price of Lisa's gift is $3935.71 --/
theorem gift_price_exact : 
  gift_price 1600 (3/8) (5/4) (2/7) 600 = 3935.71 := by
  sorry

end gift_price_gift_price_exact_l4073_407313


namespace l_structure_surface_area_l4073_407312

/-- Represents the L-shaped structure composed of unit cubes -/
structure LStructure where
  bottom_row : Nat
  first_stack : Nat
  second_stack : Nat

/-- Calculates the surface area of the L-shaped structure -/
def surface_area (l : LStructure) : Nat :=
  let bottom_area := 2 * l.bottom_row + 2
  let first_stack_area := 1 + 1 + 3 + 3 + 2
  let second_stack_area := 1 + 5 + 5 + 2
  bottom_area + first_stack_area + second_stack_area

/-- Theorem stating that the surface area of the specific L-shaped structure is 39 square units -/
theorem l_structure_surface_area :
  surface_area { bottom_row := 7, first_stack := 3, second_stack := 5 } = 39 := by
  sorry

#eval surface_area { bottom_row := 7, first_stack := 3, second_stack := 5 }

end l_structure_surface_area_l4073_407312


namespace y_intercept_of_line_l4073_407387

/-- The y-intercept of the line 4x + 7y - 3xy = 28 is (0, 4) -/
theorem y_intercept_of_line (x y : ℝ) : 
  (4 * x + 7 * y - 3 * x * y = 28) → 
  (x = 0 → y = 4) :=
by sorry

end y_intercept_of_line_l4073_407387


namespace special_polygon_interior_sum_special_polygon_exists_l4073_407372

/-- A polygon where each interior angle is 7.5 times its corresponding exterior angle -/
structure SpecialPolygon where
  n : ℕ  -- number of sides
  interior_angle : ℝ  -- measure of each interior angle
  h_interior_exterior : interior_angle = 7.5 * (360 / n)  -- relation between interior and exterior angles

/-- The sum of interior angles of a SpecialPolygon is 2700° -/
theorem special_polygon_interior_sum (P : SpecialPolygon) : 
  P.n * P.interior_angle = 2700 := by
  sorry

/-- A SpecialPolygon with 17 sides exists -/
theorem special_polygon_exists : 
  ∃ P : SpecialPolygon, P.n = 17 := by
  sorry

end special_polygon_interior_sum_special_polygon_exists_l4073_407372


namespace inequality_proof_l4073_407398

theorem inequality_proof (x₁ x₂ x₃ x₄ : ℝ) 
  (h1 : x₁ ≥ x₂) (h2 : x₂ ≥ x₃) (h3 : x₃ ≥ x₄) (h4 : x₄ ≥ 2)
  (h5 : x₂ + x₃ + x₄ ≥ x₁) : 
  (x₁ + x₂ + x₃ + x₄)^2 ≤ 4 * x₁ * x₂ * x₃ * x₄ := by
sorry

end inequality_proof_l4073_407398


namespace problem_solution_l4073_407352

theorem problem_solution (x y : ℝ) (h1 : x + y = 5) (h2 : x * y = 3) :
  x + x^2 / y + y^2 / x + y = 95 / 3 := by
  sorry

end problem_solution_l4073_407352


namespace valid_selection_probability_l4073_407351

/-- Represents a glove with a color and handedness -/
structure Glove :=
  (color : Nat)
  (isLeft : Bool)

/-- Represents a pair of gloves -/
structure GlovePair :=
  (left : Glove)
  (right : Glove)

/-- The set of all glove pairs in the cabinet -/
def glovePairs : Finset GlovePair := sorry

/-- The total number of ways to select two gloves -/
def totalSelections : Nat := sorry

/-- The number of valid selections (one left, one right, different pairs) -/
def validSelections : Nat := sorry

/-- The probability of a valid selection -/
def probabilityValidSelection : Rat := sorry

theorem valid_selection_probability :
  glovePairs.card = 3 →
  (∀ p : GlovePair, p ∈ glovePairs → p.left.color = p.right.color) →
  (∀ p q : GlovePair, p ∈ glovePairs → q ∈ glovePairs → p ≠ q → p.left.color ≠ q.left.color) →
  probabilityValidSelection = 2 / 5 := by
  sorry

end valid_selection_probability_l4073_407351


namespace race_head_start_l4073_407363

theorem race_head_start (va vb : ℝ) (h : va = 20/15 * vb) :
  let x : ℝ := 1/4
  ∀ L : ℝ, L > 0 → L / va = (L - x * L) / vb :=
by sorry

end race_head_start_l4073_407363


namespace intersection_distance_l4073_407358

-- Define the ellipse E
def ellipse (x y : ℝ) : Prop :=
  x^2 / 16 + y^2 / 12 = 1

-- Define the parabola C
def parabola (x y : ℝ) : Prop :=
  y^2 = 8 * x

-- Define the directrix of C
def directrix (x : ℝ) : Prop :=
  x = -2

-- Theorem statement
theorem intersection_distance :
  ∃ (A B : ℝ × ℝ),
    ellipse A.1 A.2 ∧
    ellipse B.1 B.2 ∧
    directrix A.1 ∧
    directrix B.1 ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 36 :=
sorry

end intersection_distance_l4073_407358


namespace trapezoid_height_l4073_407314

theorem trapezoid_height (upper_side lower_side area height : ℝ) : 
  upper_side = 5 →
  lower_side = 9 →
  area = 56 →
  area = (1/2) * (upper_side + lower_side) * height →
  height = 8 := by
sorry

end trapezoid_height_l4073_407314


namespace manuscript_typing_cost_l4073_407383

/-- Represents the typing service rates and manuscript details --/
structure ManuscriptTyping where
  total_pages : Nat
  initial_cost : Nat
  first_revision_cost : Nat
  second_revision_cost : Nat
  subsequent_revision_cost : Nat
  pages_revised_once : Nat
  pages_revised_twice : Nat
  pages_revised_thrice : Nat
  pages_revised_four_times : Nat
  pages_revised_five_times : Nat

/-- Calculates the total cost of typing and revising a manuscript --/
def total_typing_cost (m : ManuscriptTyping) : Nat :=
  m.total_pages * m.initial_cost +
  m.pages_revised_once * m.first_revision_cost +
  m.pages_revised_twice * (m.first_revision_cost + m.second_revision_cost) +
  m.pages_revised_thrice * (m.first_revision_cost + m.second_revision_cost + m.subsequent_revision_cost) +
  m.pages_revised_four_times * (m.first_revision_cost + m.second_revision_cost + m.subsequent_revision_cost * 2) +
  m.pages_revised_five_times * (m.first_revision_cost + m.second_revision_cost + m.subsequent_revision_cost * 3)

/-- Theorem stating that the total cost for the given manuscript is $5750 --/
theorem manuscript_typing_cost :
  let m := ManuscriptTyping.mk 400 10 8 6 4 60 40 20 10 5
  total_typing_cost m = 5750 := by
  sorry

end manuscript_typing_cost_l4073_407383


namespace no_integer_solution_l4073_407302

theorem no_integer_solution : ¬∃ (n : ℕ+), (20 * n + 2) ∣ (2003 * n + 2002) := by
  sorry

end no_integer_solution_l4073_407302


namespace machine_a_production_time_l4073_407325

/-- The time (in minutes) it takes for Machine A to produce one item -/
def t : ℝ := sorry

/-- The time (in minutes) it takes for Machine B to produce one item -/
def machine_b_time : ℝ := 5

/-- The duration of the production period in minutes -/
def production_period : ℝ := 1440

/-- The ratio of items produced by Machine A compared to Machine B -/
def production_ratio : ℝ := 1.25

theorem machine_a_production_time : 
  (production_period / t = production_ratio * (production_period / machine_b_time)) → t = 4 := by
  sorry

end machine_a_production_time_l4073_407325


namespace fifth_boy_payment_is_35_l4073_407318

/-- The total cost of the video game system -/
def total_cost : ℚ := 120

/-- The amount paid by the fourth boy -/
def fourth_boy_payment : ℚ := 20

/-- The payment fractions for the first three boys -/
def first_boy_fraction : ℚ := 1/3
def second_boy_fraction : ℚ := 1/4
def third_boy_fraction : ℚ := 1/5

/-- The amounts paid by each boy -/
noncomputable def first_boy_payment (second third fourth fifth : ℚ) : ℚ :=
  first_boy_fraction * (second + third + fourth + fifth)

noncomputable def second_boy_payment (first third fourth fifth : ℚ) : ℚ :=
  second_boy_fraction * (first + third + fourth + fifth)

noncomputable def third_boy_payment (first second fourth fifth : ℚ) : ℚ :=
  third_boy_fraction * (first + second + fourth + fifth)

/-- The theorem stating that the fifth boy paid $35 -/
theorem fifth_boy_payment_is_35 :
  ∃ (first second third fifth : ℚ),
    first = first_boy_payment second third fourth_boy_payment fifth ∧
    second = second_boy_payment first third fourth_boy_payment fifth ∧
    third = third_boy_payment first second fourth_boy_payment fifth ∧
    first + second + third + fourth_boy_payment + fifth = total_cost ∧
    fifth = 35 := by
  sorry

end fifth_boy_payment_is_35_l4073_407318


namespace common_tangents_O₁_O₂_l4073_407307

/-- Circle represented by its equation -/
structure Circle where
  equation : ℝ → ℝ → Prop

/-- Number of common tangents between two circles -/
def num_common_tangents (c1 c2 : Circle) : ℕ := sorry

/-- Circle O₁: x² + y² - 2x = 0 -/
def O₁ : Circle :=
  { equation := λ x y => x^2 + y^2 - 2*x = 0 }

/-- Circle O₂: x² + y² - 4x = 0 -/
def O₂ : Circle :=
  { equation := λ x y => x^2 + y^2 - 4*x = 0 }

theorem common_tangents_O₁_O₂ :
  num_common_tangents O₁ O₂ = 1 := by sorry

end common_tangents_O₁_O₂_l4073_407307


namespace quiz_score_proof_l4073_407321

theorem quiz_score_proof (score1 score2 score3 : ℕ) : 
  score2 = 90 → score3 = 92 → (score1 + score2 + score3) / 3 = 91 → score1 = 91 := by
  sorry

end quiz_score_proof_l4073_407321


namespace max_profit_at_8_l4073_407374

noncomputable def C (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 8 then (1/2) * x^2 + 4*x
  else if x ≥ 8 then 11*x + 49/x - 35
  else 0

noncomputable def P (x : ℝ) : ℝ :=
  10*x - C x - 5

theorem max_profit_at_8 :
  ∀ x > 0, P x ≤ P 8 ∧ P 8 = 127/8 :=
by sorry

end max_profit_at_8_l4073_407374


namespace parallel_planes_properties_l4073_407328

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (contains : Plane → Line → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the given condition
variable (a : Line) (α β : Plane)
variable (h : contains α a)

-- Theorem statement
theorem parallel_planes_properties :
  (∀ β, parallel_planes α β → parallel_line_plane a β) ∧
  (∀ β, ¬parallel_line_plane a β → ¬parallel_planes α β) ∧
  ¬(∀ β, parallel_line_plane a β → parallel_planes α β) :=
sorry

end parallel_planes_properties_l4073_407328


namespace smallest_nonprime_no_small_factors_range_l4073_407391

-- Define the property of having no prime factors less than 20
def no_small_prime_factors (n : ℕ) : Prop :=
  ∀ p, p < 20 → p.Prime → ¬(p ∣ n)

-- Define the property of being the smallest nonprime with no small prime factors
def smallest_nonprime_no_small_factors (n : ℕ) : Prop :=
  n > 1 ∧ ¬n.Prime ∧ no_small_prime_factors n ∧
  ∀ m, m > 1 → ¬m.Prime → no_small_prime_factors m → n ≤ m

-- State the theorem
theorem smallest_nonprime_no_small_factors_range :
  ∃ n, smallest_nonprime_no_small_factors n ∧ 500 < n ∧ n ≤ 550 := by
  sorry

end smallest_nonprime_no_small_factors_range_l4073_407391


namespace similar_triangles_with_two_equal_sides_l4073_407355

theorem similar_triangles_with_two_equal_sides (a b c d e f : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0 →
  (a = 80 ∧ b = 100) →
  (d = 80 ∧ e = 100) →
  a / d = b / e →
  a / d = c / f →
  b / e = c / f →
  ((c = 64 ∧ f = 125) ∨ (c = 125 ∧ f = 64)) :=
by sorry

end similar_triangles_with_two_equal_sides_l4073_407355


namespace cone_base_radius_l4073_407397

/-- Given a cone with surface area 15π cm² and lateral surface that unfolds into a semicircle,
    prove that the radius of its base is √5 cm. -/
theorem cone_base_radius (surface_area : ℝ) (r : ℝ) :
  surface_area = 15 * Real.pi ∧
  (∃ l : ℝ, π * l = 2 * π * r ∧ surface_area = π * r^2 + π * r * l) →
  r = Real.sqrt 5 := by
  sorry


end cone_base_radius_l4073_407397


namespace smallest_integers_difference_smallest_integers_difference_exists_l4073_407392

theorem smallest_integers_difference : ℕ → Prop :=
  fun n =>
    (∃ a b : ℕ,
      (a > 1 ∧ b > 1 ∧ a < b) ∧
      (∀ k : ℕ, 3 ≤ k → k ≤ 12 → a % k = 1 ∧ b % k = 1) ∧
      (∀ x : ℕ, x > 1 ∧ x < a → ∃ k : ℕ, 3 ≤ k ∧ k ≤ 12 ∧ x % k ≠ 1) ∧
      (b - a = n)) →
    n = 13860

theorem smallest_integers_difference_exists : ∃ n : ℕ, smallest_integers_difference n :=
  sorry

end smallest_integers_difference_smallest_integers_difference_exists_l4073_407392


namespace sine_function_parameters_l4073_407364

theorem sine_function_parameters
  (y : ℝ → ℝ)
  (a b c : ℝ)
  (h1 : ∀ x, y x = a * Real.sin (b * x + c))
  (h2 : a > 0)
  (h3 : b > 0)
  (h4 : y (π / 6) = 3)
  (h5 : ∀ x, y (x + π) = y x) :
  a = 3 ∧ b = 2 ∧ c = π / 6 := by
sorry

end sine_function_parameters_l4073_407364


namespace fish_per_black_duck_is_ten_l4073_407380

/-- Represents the number of fish per duck for each duck color -/
structure FishPerDuck where
  white : ℕ
  multicolor : ℕ

/-- Represents the number of ducks for each color -/
structure DuckCounts where
  white : ℕ
  black : ℕ
  multicolor : ℕ

/-- Calculates the number of fish per black duck -/
def fishPerBlackDuck (fpd : FishPerDuck) (dc : DuckCounts) (totalFish : ℕ) : ℚ :=
  let fishForWhite := fpd.white * dc.white
  let fishForMulticolor := fpd.multicolor * dc.multicolor
  let fishForBlack := totalFish - fishForWhite - fishForMulticolor
  (fishForBlack : ℚ) / dc.black

theorem fish_per_black_duck_is_ten :
  let fpd : FishPerDuck := { white := 5, multicolor := 12 }
  let dc : DuckCounts := { white := 3, black := 7, multicolor := 6 }
  let totalFish : ℕ := 157
  fishPerBlackDuck fpd dc totalFish = 10 := by
  sorry

end fish_per_black_duck_is_ten_l4073_407380


namespace grape_purchase_amount_l4073_407344

/-- The price of grapes per kg -/
def grape_price : ℕ := 70

/-- The price of mangoes per kg -/
def mango_price : ℕ := 55

/-- The number of kg of mangoes purchased -/
def mango_kg : ℕ := 9

/-- The total amount paid -/
def total_paid : ℕ := 1195

/-- The number of kg of grapes purchased -/
def grape_kg : ℕ := (total_paid - mango_price * mango_kg) / grape_price

theorem grape_purchase_amount : grape_kg = 10 := by
  sorry

end grape_purchase_amount_l4073_407344


namespace parabola_focus_distance_l4073_407381

theorem parabola_focus_distance (p : ℝ) (h1 : p > 0) :
  ∃ (x y : ℝ),
    y^2 = 2*p*x ∧
    x + p/2 = 2 →
    Real.sqrt (x - p/2)^2 + y^2 = Real.sqrt (2*p*(2 - p/2)) :=
by sorry

end parabola_focus_distance_l4073_407381


namespace vector_magnitude_solution_l4073_407349

/-- Given a vector a = (5, x) with magnitude 9, prove that x = 2√14 or x = -2√14 -/
theorem vector_magnitude_solution (x : ℝ) : 
  let a : ℝ × ℝ := (5, x)
  (‖a‖ = 9) → (x = 2 * Real.sqrt 14 ∨ x = -2 * Real.sqrt 14) := by
  sorry

end vector_magnitude_solution_l4073_407349


namespace pages_per_night_l4073_407345

/-- Given a book with 1200 pages read over 10.0 days, prove that 120 pages are read each night. -/
theorem pages_per_night (total_pages : ℕ) (reading_days : ℝ) :
  total_pages = 1200 → reading_days = 10.0 → (total_pages : ℝ) / reading_days = 120 := by
  sorry

end pages_per_night_l4073_407345


namespace green_tile_probability_l4073_407390

theorem green_tile_probability :
  let total_tiles := 100
  let is_green (n : ℕ) := n % 5 = 3
  let green_tiles := Finset.filter is_green (Finset.range total_tiles)
  (green_tiles.card : ℚ) / total_tiles = 1 / 5 := by
sorry

end green_tile_probability_l4073_407390


namespace no_two_digit_number_exists_l4073_407384

theorem no_two_digit_number_exists : ¬∃ (n : ℕ), 
  (10 ≤ n ∧ n < 100) ∧ 
  (∃ (d₁ d₂ : ℕ), 
    d₁ < 10 ∧ d₂ < 10 ∧
    n = 10 * d₁ + d₂ ∧
    n = 2 * (d₁^2 + d₂^2) + 6 ∧
    n = 4 * (d₁ * d₂) + 6) :=
sorry

end no_two_digit_number_exists_l4073_407384


namespace integer_solution_equation_l4073_407396

theorem integer_solution_equation (x y : ℤ) : 
  9 * x + 2 = y * (y + 1) ↔ ∃ k : ℤ, x = k * (k + 1) ∧ y = 3 * k + 1 := by
  sorry

end integer_solution_equation_l4073_407396


namespace edward_final_earnings_l4073_407310

/-- Edward's lawn mowing business earnings and expenses --/
def edward_business (spring_earnings summer_earnings supplies_cost : ℕ) : ℕ :=
  spring_earnings + summer_earnings - supplies_cost

/-- Theorem: Edward's final earnings --/
theorem edward_final_earnings :
  edward_business 2 27 5 = 24 := by
  sorry

end edward_final_earnings_l4073_407310


namespace arithmetic_sequence_ninth_term_l4073_407308

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_ninth_term
  (a : ℕ → ℤ)
  (h_arith : ArithmeticSequence a)
  (h_diff : a 4 - a 2 = -2)
  (h_seventh : a 7 = -3) :
  a 9 = -5 := by
sorry

end arithmetic_sequence_ninth_term_l4073_407308


namespace factorization_x9_minus_512_l4073_407362

theorem factorization_x9_minus_512 (x : ℝ) : 
  x^9 - 512 = (x - 2) * (x^2 + 2*x + 4) * (x^6 + 2*x^3 + 4) := by
  sorry

end factorization_x9_minus_512_l4073_407362


namespace quadratic_function_range_l4073_407361

/-- Given a quadratic function f(x) = ax^2 - 2ax + c where a and c are real numbers,
    if f(2017) < f(-2016), then the set of real numbers m that satisfies f(m) ≤ f(0)
    is equal to the closed interval [0, 2]. -/
theorem quadratic_function_range (a c : ℝ) :
  let f := fun x : ℝ => a * x^2 - 2 * a * x + c
  (f 2017 < f (-2016)) →
  {m : ℝ | f m ≤ f 0} = Set.Icc 0 2 := by
sorry

end quadratic_function_range_l4073_407361


namespace gcd_difference_theorem_l4073_407367

theorem gcd_difference_theorem : Nat.gcd 5610 210 - 10 = 20 := by
  sorry

end gcd_difference_theorem_l4073_407367


namespace friends_pooling_money_l4073_407331

-- Define the friends
inductive Friend
| Emma
| Daya
| Jeff
| Brenda

-- Define a function to get the amount of money each friend has
def money (f : Friend) : ℚ :=
  match f with
  | Friend.Emma => 8
  | Friend.Daya => 8 * (1 + 1/4)
  | Friend.Jeff => (2/5) * (8 * (1 + 1/4))
  | Friend.Brenda => (2/5) * (8 * (1 + 1/4)) + 4

-- Theorem stating that there are 4 friends pooling money for pizza
theorem friends_pooling_money :
  (∃ (s : Finset Friend), s.card = 4 ∧ 
    (∀ f : Friend, f ∈ s) ∧
    (money Friend.Emma = 8) ∧
    (money Friend.Daya = money Friend.Emma * (1 + 1/4)) ∧
    (money Friend.Jeff = (2/5) * money Friend.Daya) ∧
    (money Friend.Brenda = money Friend.Jeff + 4) ∧
    (money Friend.Brenda = 8)) :=
by sorry

end friends_pooling_money_l4073_407331


namespace prob_even_sum_is_five_ninths_l4073_407360

/-- Represents a spinner with a list of numbers -/
def Spinner := List Nat

/-- The spinner X with numbers 1, 4, 5 -/
def X : Spinner := [1, 4, 5]

/-- The spinner Y with numbers 1, 2, 3 -/
def Y : Spinner := [1, 2, 3]

/-- The spinner Z with numbers 2, 4, 6 -/
def Z : Spinner := [2, 4, 6]

/-- Predicate to check if a number is even -/
def isEven (n : Nat) : Bool := n % 2 = 0

/-- Function to calculate the probability of an even sum when spinning X, Y, and Z -/
def probEvenSum (x y z : Spinner) : Rat :=
  let totalOutcomes := (x.length * y.length * z.length : Nat)
  let evenSumOutcomes := x.countP (fun a => 
    y.countP (fun b => 
      z.countP (fun c => 
        isEven (a + b + c)) = z.length) = y.length) * x.length
  evenSumOutcomes / totalOutcomes

/-- Theorem stating that the probability of an even sum when spinning X, Y, and Z is 5/9 -/
theorem prob_even_sum_is_five_ninths : probEvenSum X Y Z = 5/9 := by
  sorry

end prob_even_sum_is_five_ninths_l4073_407360


namespace square_root_fraction_equality_l4073_407399

theorem square_root_fraction_equality : 
  Real.sqrt (8^2 + 15^2) / Real.sqrt (25 + 16) = 17 / Real.sqrt 41 := by
  sorry

end square_root_fraction_equality_l4073_407399


namespace trig_equation_solution_l4073_407301

theorem trig_equation_solution (x : Real) : 
  (Real.sin (π/4 + 5*x) * Real.cos (π/4 + 2*x) = Real.sin (π/4 + x) * Real.sin (π/4 - 6*x)) ↔ 
  (∃ n : Int, x = n * π/4) :=
by sorry

end trig_equation_solution_l4073_407301


namespace geometric_series_proof_l4073_407370

def geometric_series_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_series_proof :
  let a : ℚ := 1/2
  let r : ℚ := -1/2
  let n : ℕ := 6
  geometric_series_sum a r n = 21/64 :=
by
  sorry

end geometric_series_proof_l4073_407370


namespace store_sales_problem_l4073_407346

theorem store_sales_problem (d : ℕ) : 
  (86 + 50 * d) / (d + 1) = 53 → d = 11 := by
  sorry

end store_sales_problem_l4073_407346


namespace quadrilateral_area_product_is_square_quadrilateral_area_product_not_end_1988_l4073_407317

/-- Represents a convex quadrilateral divided by its diagonals -/
structure ConvexQuadrilateral where
  /-- The area of the first triangle -/
  area1 : ℕ
  /-- The area of the second triangle -/
  area2 : ℕ
  /-- The area of the third triangle -/
  area3 : ℕ
  /-- The area of the fourth triangle -/
  area4 : ℕ

/-- The product of the areas of the four triangles in a convex quadrilateral
    divided by its diagonals is a perfect square -/
theorem quadrilateral_area_product_is_square (q : ConvexQuadrilateral) :
  ∃ (n : ℕ), q.area1 * q.area2 * q.area3 * q.area4 = n * n := by
  sorry

/-- The product of the areas of the four triangles in a convex quadrilateral
    divided by its diagonals cannot end in 1988 -/
theorem quadrilateral_area_product_not_end_1988 (q : ConvexQuadrilateral) :
  ¬(q.area1 * q.area2 * q.area3 * q.area4 % 10000 = 1988) := by
  sorry

end quadrilateral_area_product_is_square_quadrilateral_area_product_not_end_1988_l4073_407317


namespace sum_of_coordinates_is_50_l4073_407322

def is_valid_point (x y : ℝ) : Prop :=
  (y = 15 + 3 ∨ y = 15 - 3) ∧ 
  ((x - 5)^2 + (y - 15)^2 = 10^2)

theorem sum_of_coordinates_is_50 :
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ),
    is_valid_point x₁ y₁ ∧
    is_valid_point x₂ y₂ ∧
    is_valid_point x₃ y₃ ∧
    is_valid_point x₄ y₄ ∧
    x₁ + y₁ + x₂ + y₂ + x₃ + y₃ + x₄ + y₄ = 50 :=
  sorry

end sum_of_coordinates_is_50_l4073_407322


namespace sum_of_non_visible_faces_l4073_407353

/-- Represents a standard six-sided die -/
def StandardDie : Type := Fin 6

/-- The sum of all faces on a standard die -/
def sumOfDieFaces : ℕ := (List.range 6).map (· + 1) |>.sum

/-- The total number of dice -/
def numberOfDice : ℕ := 4

/-- The list of visible face values -/
def visibleFaces : List ℕ := [1, 1, 2, 2, 3, 3, 4, 5, 5, 6]

/-- The sum of visible face values -/
def sumOfVisibleFaces : ℕ := visibleFaces.sum

/-- Theorem: The sum of non-visible face values is 52 -/
theorem sum_of_non_visible_faces :
  numberOfDice * sumOfDieFaces - sumOfVisibleFaces = 52 := by
  sorry

end sum_of_non_visible_faces_l4073_407353


namespace min_value_of_3x_plus_4y_min_value_is_five_l4073_407300

theorem min_value_of_3x_plus_4y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 3 * y = 5 * x * y) :
  ∀ a b : ℝ, a > 0 → b > 0 → a + 3 * b = 5 * a * b → 3 * x + 4 * y ≤ 3 * a + 4 * b :=
by sorry

theorem min_value_is_five (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 3 * y = 5 * x * y) :
  3 * x + 4 * y ≥ 5 :=
by sorry

end min_value_of_3x_plus_4y_min_value_is_five_l4073_407300


namespace employee_not_working_first_day_l4073_407334

def total_employees : ℕ := 6
def days : ℕ := 3
def employees_per_day : ℕ := 2

def schedule_probability (n m : ℕ) : ℚ := (n.choose m : ℚ) / (total_employees.choose employees_per_day : ℚ)

theorem employee_not_working_first_day :
  schedule_probability (total_employees - 1) employees_per_day = 2/3 :=
sorry

end employee_not_working_first_day_l4073_407334


namespace correct_transformation_l4073_407304

theorem correct_transformation (x : ℝ) : (2/3 * x - 1 = x) ↔ (2*x - 3 = 3*x) := by
  sorry

end correct_transformation_l4073_407304


namespace horner_method_correct_l4073_407348

/-- Horner's method for polynomial evaluation -/
def horner (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 1 + x + 2x^2 + 3x^3 + 4x^4 + 5x^5 -/
def f (x : ℝ) : ℝ :=
  1 + x + 2*x^2 + 3*x^3 + 4*x^4 + 5*x^5

theorem horner_method_correct :
  horner [5, 4, 3, 2, 1, 1] (-1) = f (-1) ∧ f (-1) = -2 := by
  sorry

end horner_method_correct_l4073_407348


namespace quadratic_inequality_solution_set_l4073_407357

theorem quadratic_inequality_solution_set (a : ℝ) : 
  (∀ x : ℝ, x^2 - a*x + a > 0) ↔ (0 < a ∧ a < 4) :=
sorry

end quadratic_inequality_solution_set_l4073_407357


namespace geometric_series_sum_l4073_407316

/-- The sum of a geometric series with first term a, common ratio r, and n terms -/
def geometricSum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The first term of our geometric series -/
def a : ℚ := 1/2

/-- The common ratio of our geometric series -/
def r : ℚ := 1/2

/-- The number of terms in our geometric series -/
def n : ℕ := 8

theorem geometric_series_sum :
  geometricSum a r n = 255/256 := by
  sorry

end geometric_series_sum_l4073_407316


namespace paths_from_A_to_B_is_16_l4073_407330

/-- Represents the number of arrows of each color in the hexagonal lattice --/
structure ArrowCounts where
  red : Nat
  blue : Nat
  green : Nat
  purple : Nat
  orange : Nat

/-- Represents the connection rules between arrows of different colors --/
structure ConnectionRules where
  redToBlue : Nat
  blueToGreen : Nat
  greenToPurple : Nat
  purpleToOrange : Nat
  orangeToB : Nat

/-- Calculates the number of paths from A to B in the hexagonal lattice --/
def pathsFromAToB (counts : ArrowCounts) (rules : ConnectionRules) : Nat :=
  counts.red * rules.redToBlue * counts.blue * rules.blueToGreen * counts.green *
  rules.greenToPurple * counts.purple * rules.purpleToOrange * counts.orange * rules.orangeToB

/-- Theorem stating that the number of paths from A to B is 16 --/
theorem paths_from_A_to_B_is_16 (counts : ArrowCounts) (rules : ConnectionRules) :
  counts.red = 2 ∧ counts.blue = 2 ∧ counts.green = 4 ∧ counts.purple = 4 ∧ counts.orange = 4 ∧
  rules.redToBlue = 2 ∧ rules.blueToGreen = 3 ∧ rules.greenToPurple = 2 ∧
  rules.purpleToOrange = 1 ∧ rules.orangeToB = 1 →
  pathsFromAToB counts rules = 16 := by
  sorry

#check paths_from_A_to_B_is_16

end paths_from_A_to_B_is_16_l4073_407330


namespace complement_of_union_l4073_407356

open Set

def U : Set Nat := {1,2,3,4,5,6}
def S : Set Nat := {1,3,5}
def T : Set Nat := {3,6}

theorem complement_of_union : (U \ (S ∪ T)) = {2,4} := by sorry

end complement_of_union_l4073_407356


namespace division_problem_l4073_407332

theorem division_problem (dividend quotient divisor remainder multiple : ℕ) :
  remainder = 6 →
  dividend = 86 →
  divisor = 5 * quotient →
  divisor = multiple * remainder + 2 →
  dividend = divisor * quotient + remainder →
  multiple = 3 := by
  sorry

end division_problem_l4073_407332


namespace symmetric_points_difference_l4073_407339

/-- Given two points A and B symmetric about the y-axis, prove that m-n = -4 -/
theorem symmetric_points_difference (m n : ℝ) : 
  (∃ (A B : ℝ × ℝ), 
    A = (m - 2, 3) ∧ 
    B = (4, n + 1) ∧ 
    (A.1 = -B.1) ∧  -- x-coordinates are opposite
    (A.2 = B.2))    -- y-coordinates are equal
  → m - n = -4 := by
  sorry

end symmetric_points_difference_l4073_407339


namespace point_on_curve_l4073_407375

theorem point_on_curve : (3^2 : ℝ) - 3 * 10 + 2 * 10 + 1 = 0 := by sorry

end point_on_curve_l4073_407375


namespace rationalize_sqrt_five_twelfths_l4073_407394

theorem rationalize_sqrt_five_twelfths : 
  Real.sqrt (5 / 12) = Real.sqrt 15 / 6 := by
  sorry

end rationalize_sqrt_five_twelfths_l4073_407394


namespace photo_arrangement_count_l4073_407385

/-- The number of ways to arrange four people from two teachers and four students,
    where the teachers must be selected and adjacent. -/
def arrangement_count : ℕ := 72

/-- The number of teachers -/
def teacher_count : ℕ := 2

/-- The number of students -/
def student_count : ℕ := 4

/-- The total number of people to be selected -/
def selection_count : ℕ := 4

theorem photo_arrangement_count :
  arrangement_count = 
    (teacher_count.factorial) *              -- Ways to arrange teachers
    (student_count.choose (selection_count - teacher_count)) * -- Ways to choose students
    ((selection_count - 1).factorial) :=     -- Ways to arrange teachers bundle and students
  by sorry

end photo_arrangement_count_l4073_407385


namespace sticks_per_pot_l4073_407386

/-- Given:
  * There are 466 pots
  * Each pot has 53 flowers
  * There are 109044 flowers and sticks in total
  Prove that there are 181 sticks in each pot -/
theorem sticks_per_pot (num_pots : ℕ) (flowers_per_pot : ℕ) (total_items : ℕ) :
  num_pots = 466 →
  flowers_per_pot = 53 →
  total_items = 109044 →
  (total_items - num_pots * flowers_per_pot) / num_pots = 181 := by
  sorry

#eval (109044 - 466 * 53) / 466  -- Should output 181

end sticks_per_pot_l4073_407386


namespace lucy_mother_age_relation_l4073_407341

/-- Lucy's age in 2010 -/
def lucy_age_2010 : ℕ := 10

/-- Lucy's mother's age in 2010 -/
def mother_age_2010 : ℕ := 5 * lucy_age_2010

/-- The year when Lucy's mother's age will be twice Lucy's age -/
def target_year : ℕ := 2040

/-- The number of years from 2010 to the target year -/
def years_passed : ℕ := target_year - 2010

theorem lucy_mother_age_relation :
  mother_age_2010 + years_passed = 2 * (lucy_age_2010 + years_passed) :=
by sorry

end lucy_mother_age_relation_l4073_407341


namespace cannot_determine_books_left_l4073_407388

def initial_pens : ℕ := 42
def initial_books : ℕ := 143
def pens_sold : ℕ := 23
def pens_left : ℕ := 19

theorem cannot_determine_books_left : 
  ∀ (books_left : ℕ), 
  initial_pens = pens_sold + pens_left →
  ¬(∀ (books_sold : ℕ), initial_books = books_sold + books_left) :=
by
  sorry

end cannot_determine_books_left_l4073_407388


namespace total_distance_two_parts_l4073_407354

/-- Calculates the total distance traveled by a car with varying speeds -/
theorem total_distance_two_parts (v1 v2 t1 t2 D1 D2 : ℝ) :
  D1 = v1 * t1 →
  D2 = v2 * t2 →
  let D := D1 + D2
  D = v1 * t1 + v2 * t2 := by
  sorry

end total_distance_two_parts_l4073_407354


namespace triangle_problem_l4073_407305

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) (T : ℝ) :
  0 < A ∧ A < π ∧
  0 < B ∧ B < π ∧
  0 < C ∧ C < π ∧
  A + B + C = π ∧
  (2 * a - c) * Real.cos B = b * Real.cos C ∧
  b = Real.sqrt 3 ∧
  T = (1 / 2) * a * c * Real.sin B →
  B = π / 3 ∧ a + c = Real.sqrt 15 := by
  sorry

end triangle_problem_l4073_407305


namespace complex_fraction_problem_l4073_407311

theorem complex_fraction_problem (x y : ℂ) 
  (h1 : (x^2 + y^2) / (x + y) = 4)
  (h2 : (x^4 + y^4) / (x^3 + y^3) = 2)
  (h3 : x + y ≠ 0)
  (h4 : x^3 + y^3 ≠ 0)
  (h5 : x^5 + y^5 ≠ 0) :
  (x^6 + y^6) / (x^5 + y^5) = 4 := by
sorry

end complex_fraction_problem_l4073_407311


namespace opposite_and_reciprocal_expression_l4073_407315

theorem opposite_and_reciprocal_expression (a b c d : ℝ) 
  (h1 : a = -b) 
  (h2 : c * d = 1) : 
  (a + b) / 2 - c * d = -1 := by
  sorry

end opposite_and_reciprocal_expression_l4073_407315


namespace cube_sum_over_product_is_18_l4073_407389

theorem cube_sum_over_product_is_18 
  (a b c : ℂ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (sum_15 : a + b + c = 15)
  (squared_diff_sum : (a - b)^2 + (a - c)^2 + (b - c)^2 = 2*a*b*c) :
  (a^3 + b^3 + c^3) / (a*b*c) = 18 := by
  sorry

end cube_sum_over_product_is_18_l4073_407389


namespace max_pieces_theorem_l4073_407371

/-- Represents the size of the cake in inches -/
def cake_size : ℕ := 100

/-- Represents the size of each piece in inches -/
def piece_size : ℕ := 4

/-- Predicate to check if a number is even -/
def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

/-- Theorem stating the maximum number of pieces that can be cut from the cake -/
theorem max_pieces_theorem :
  (is_even cake_size) →
  (is_even piece_size) →
  (cake_size % piece_size = 0) →
  (cake_size / piece_size) * (cake_size / piece_size) = 625 := by
  sorry

#check max_pieces_theorem

end max_pieces_theorem_l4073_407371


namespace circumscribed_sphere_surface_area_l4073_407378

/-- The surface area of a sphere circumscribing a right circular cone -/
theorem circumscribed_sphere_surface_area 
  (base_radius : ℝ) 
  (slant_height : ℝ) 
  (h1 : base_radius = Real.sqrt 3)
  (h2 : slant_height = 2) :
  ∃ (sphere_radius : ℝ), 
    4 * Real.pi * sphere_radius^2 = 16 * Real.pi := by
  sorry

end circumscribed_sphere_surface_area_l4073_407378


namespace range_inequalities_l4073_407333

theorem range_inequalities 
  (a b x y : ℝ) 
  (ha : 12 < a ∧ a < 60) 
  (hb : 15 < b ∧ b < 36) 
  (hxy1 : -1/2 < x - y ∧ x - y < 1/2) 
  (hxy2 : 0 < x + y ∧ x + y < 1) : 
  (-12 < 2*a - b ∧ 2*a - b < 105) ∧ 
  (1/3 < a/b ∧ a/b < 4) ∧ 
  (-1 < 3*x - y ∧ 3*x - y < 2) := by
sorry

end range_inequalities_l4073_407333


namespace inscribed_quadrilateral_sides_l4073_407365

/-- A quadrilateral inscribed in a circle with perpendicular diagonals -/
structure InscribedQuadrilateral where
  R : ℝ  -- Radius of the circumscribed circle
  d1 : ℝ  -- Distance of first diagonal from circle center
  d2 : ℝ  -- Distance of second diagonal from circle center

/-- The sides of the quadrilateral -/
def quadrilateralSides (q : InscribedQuadrilateral) : Set ℝ :=
  {x | ∃ (n : ℤ), x = 4 * (2 * Real.sqrt 13 + n) ∨ x = 4 * (8 + 2 * n * Real.sqrt 13)}

/-- Theorem stating the sides of the quadrilateral given specific conditions -/
theorem inscribed_quadrilateral_sides 
  (q : InscribedQuadrilateral) 
  (h1 : q.R = 17) 
  (h2 : q.d1 = 8) 
  (h3 : q.d2 = 9) : 
  ∀ s, s ∈ quadrilateralSides q ↔ 
    (s = 4 * (2 * Real.sqrt 13 - 1) ∨ 
     s = 4 * (2 * Real.sqrt 13 + 1) ∨ 
     s = 4 * (8 - 2 * Real.sqrt 13) ∨ 
     s = 4 * (8 + 2 * Real.sqrt 13)) :=
by sorry

end inscribed_quadrilateral_sides_l4073_407365


namespace possible_tile_counts_l4073_407319

/-- Represents the dimensions of a rectangular floor in terms of tiles -/
structure FloorDimensions where
  width : ℕ
  length : ℕ

/-- Calculates the number of red tiles on the floor -/
def redTiles (d : FloorDimensions) : ℕ := 2 * d.width + 2 * d.length - 4

/-- Calculates the number of white tiles on the floor -/
def whiteTiles (d : FloorDimensions) : ℕ := d.width * d.length - redTiles d

/-- Checks if the number of red and white tiles are equal -/
def equalRedWhite (d : FloorDimensions) : Prop := redTiles d = whiteTiles d

/-- The theorem stating the possible total number of tiles -/
theorem possible_tile_counts : 
  ∀ d : FloorDimensions, 
    equalRedWhite d → 
    d.width * d.length = 48 ∨ d.width * d.length = 60 := by
  sorry

end possible_tile_counts_l4073_407319


namespace exists_max_a_l4073_407359

def is_valid_number (a d e : ℕ) : Prop :=
  0 ≤ a ∧ a ≤ 9 ∧
  0 ≤ d ∧ d ≤ 9 ∧
  0 ≤ e ∧ e ≤ 8 ∧
  (500000 + 100000 * a + 1000 * d + 500 + 20 + 4 + e) % 24 = 0

theorem exists_max_a : ∃ (d e : ℕ), is_valid_number 9 d e :=
sorry

end exists_max_a_l4073_407359


namespace ali_flower_sales_l4073_407342

def monday_sales : ℕ := 4
def tuesday_sales : ℕ := 8
def wednesday_sales : ℕ := monday_sales + 3
def thursday_sales : ℕ := 6
def friday_sales : ℕ := 2 * monday_sales
def saturday_bundles : ℕ := 5
def flowers_per_bundle : ℕ := 9
def saturday_sales : ℕ := saturday_bundles * flowers_per_bundle

def total_sales : ℕ := monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales + saturday_sales

theorem ali_flower_sales : total_sales = 78 := by
  sorry

end ali_flower_sales_l4073_407342


namespace digit_sum_problem_l4073_407303

theorem digit_sum_problem (p q r : ℕ) : 
  p ≠ 0 → q ≠ 0 → r ≠ 0 →
  p < 10 → q < 10 → r < 10 →
  100 * p + 10 * q + r + 10 * q + r + r = 912 →
  q = 5 := by
sorry

end digit_sum_problem_l4073_407303


namespace prob_king_or_ace_eq_two_thirteenth_l4073_407327

/-- A standard deck of cards -/
structure Deck :=
  (cards : Finset (Nat × Nat))
  (card_count : cards.card = 52)
  (rank_count : (cards.image (·.1)).card = 13)
  (suit_count : (cards.image (·.2)).card = 4)
  (unique_pairs : ∀ r s, (r, s) ∈ cards → r ∈ Finset.range 13 ∧ s ∈ Finset.range 4)

/-- The probability of drawing a King or an Ace from the top of a shuffled deck -/
def prob_king_or_ace (d : Deck) : ℚ :=
  (d.cards.filter (λ p => p.1 = 0 ∨ p.1 = 12)).card / d.cards.card

/-- Theorem: The probability of drawing a King or an Ace is 2/13 -/
theorem prob_king_or_ace_eq_two_thirteenth (d : Deck) : 
  prob_king_or_ace d = 2 / 13 := by
  sorry

end prob_king_or_ace_eq_two_thirteenth_l4073_407327

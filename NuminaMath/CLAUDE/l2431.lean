import Mathlib

namespace NUMINAMATH_CALUDE_males_chose_malt_l2431_243122

/-- Represents the number of cheerleaders who chose malt or coke -/
structure CheerleaderChoices where
  males : ℕ
  females : ℕ

/-- The properties of the cheerleader group and their choices -/
def CheerleaderProblem (choices : CheerleaderChoices) : Prop :=
  -- Total number of cheerleaders
  choices.males + choices.females = 26 ∧
  -- Number of males
  choices.males = 10 ∧
  -- Number of females
  choices.females = 16 ∧
  -- Number of malt choosers is double the number of coke choosers
  choices.males + choices.females = 3 * (26 - (choices.males + choices.females)) ∧
  -- 8 females chose malt
  choices.females = 8

/-- Theorem stating the number of males who chose malt -/
theorem males_chose_malt (choices : CheerleaderChoices) 
  (h : CheerleaderProblem choices) : choices.males = 9 := by
  sorry


end NUMINAMATH_CALUDE_males_chose_malt_l2431_243122


namespace NUMINAMATH_CALUDE_num_possible_lists_l2431_243129

def num_balls : ℕ := 15
def selections_per_list : ℕ := 2
def num_selections : ℕ := 2

def num_ways_to_select (n k : ℕ) : ℕ := Nat.choose n k

theorem num_possible_lists : 
  (num_ways_to_select num_balls selections_per_list) ^ num_selections = 11025 := by
  sorry

end NUMINAMATH_CALUDE_num_possible_lists_l2431_243129


namespace NUMINAMATH_CALUDE_simple_interest_problem_l2431_243138

theorem simple_interest_problem (interest : ℝ) (rate : ℝ) (time : ℝ) (principal : ℝ) : 
  interest = 4025.25 →
  rate = 9 →
  time = 5 →
  principal = interest / (rate * time / 100) →
  principal = 8950 :=
by sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l2431_243138


namespace NUMINAMATH_CALUDE_rook_placement_theorem_l2431_243181

theorem rook_placement_theorem (n : ℕ) (h1 : n > 2) (h2 : Even n) :
  ∃ (coloring : Fin n → Fin n → Fin (n^2/2))
    (rook_positions : Fin n → Fin n × Fin n),
    (∀ i j : Fin n, (∃! k : Fin n, coloring i k = coloring j k) ∨ i = j) ∧
    (∀ i j : Fin n, i ≠ j →
      (rook_positions i).1 ≠ (rook_positions j).1 ∧
      (rook_positions i).2 ≠ (rook_positions j).2) ∧
    (∀ i j : Fin n, i ≠ j →
      coloring (rook_positions i).1 (rook_positions i).2 ≠
      coloring (rook_positions j).1 (rook_positions j).2) :=
by sorry

end NUMINAMATH_CALUDE_rook_placement_theorem_l2431_243181


namespace NUMINAMATH_CALUDE_area_equals_perimeter_count_l2431_243115

-- Define a rectangle with integer sides
structure Rectangle where
  a : ℕ
  b : ℕ

-- Define a right triangle with integer sides
structure RightTriangle where
  a : ℕ
  b : ℕ

-- Function to check if a rectangle's area equals its perimeter
def Rectangle.areaEqualsPerimeter (r : Rectangle) : Prop :=
  r.a * r.b = 2 * (r.a + r.b)

-- Function to check if a right triangle's area equals its perimeter
def RightTriangle.areaEqualsPerimeter (t : RightTriangle) : Prop :=
  t.a * t.b / 2 = t.a + t.b + Int.sqrt (t.a^2 + t.b^2)

-- The main theorem
theorem area_equals_perimeter_count :
  (∃! (r₁ r₂ : Rectangle), r₁ ≠ r₂ ∧ r₁.areaEqualsPerimeter ∧ r₂.areaEqualsPerimeter) ∧
  (∃! (t₁ t₂ : RightTriangle), t₁ ≠ t₂ ∧ t₁.areaEqualsPerimeter ∧ t₂.areaEqualsPerimeter) :=
sorry

end NUMINAMATH_CALUDE_area_equals_perimeter_count_l2431_243115


namespace NUMINAMATH_CALUDE_triangle_side_equation_l2431_243154

theorem triangle_side_equation (a b x : ℝ) (θ : ℝ) : 
  a = 6 → b = 2 * Real.sqrt 7 → θ = π / 3 → 
  x ^ 2 = a ^ 2 + b ^ 2 - 2 * a * b * Real.cos θ → 
  x ^ 2 - 6 * x + 8 = 0 := by sorry

end NUMINAMATH_CALUDE_triangle_side_equation_l2431_243154


namespace NUMINAMATH_CALUDE_liam_monthly_savings_l2431_243145

/-- Calculates the monthly savings given the trip cost, bills cost, saving period in years, and amount left after paying bills. -/
def monthly_savings (trip_cost bills_cost : ℚ) (saving_period : ℕ) (amount_left : ℚ) : ℚ :=
  (amount_left + bills_cost + trip_cost) / (saving_period * 12)

/-- Theorem stating that Liam's monthly savings are $791.67 given the problem conditions. -/
theorem liam_monthly_savings :
  let trip_cost : ℚ := 7000
  let bills_cost : ℚ := 3500
  let saving_period : ℕ := 2
  let amount_left : ℚ := 8500
  monthly_savings trip_cost bills_cost saving_period amount_left = 791.67 := by
  sorry

#eval monthly_savings 7000 3500 2 8500

end NUMINAMATH_CALUDE_liam_monthly_savings_l2431_243145


namespace NUMINAMATH_CALUDE_area_third_face_l2431_243142

/-- Theorem: Area of the third adjacent face of a cuboidal box -/
theorem area_third_face (l w h : ℝ) : 
  l * w = 120 →
  w * h = 60 →
  l * w * h = 720 →
  l * h = 72 := by
sorry

end NUMINAMATH_CALUDE_area_third_face_l2431_243142


namespace NUMINAMATH_CALUDE_outfit_count_is_688_l2431_243120

/-- Represents the number of items of clothing -/
structure ClothingCounts where
  redShirts : Nat
  greenShirts : Nat
  bluePants : Nat
  greenPants : Nat
  greenHats : Nat
  redHats : Nat

/-- Calculates the number of valid outfits given clothing counts -/
def countOutfits (c : ClothingCounts) : Nat :=
  (c.redShirts * c.greenHats * c.greenPants) + (c.greenShirts * c.redHats * c.bluePants)

/-- Theorem stating that the number of outfits is 688 given the specific clothing counts -/
theorem outfit_count_is_688 :
  let counts : ClothingCounts := {
    redShirts := 5,
    greenShirts := 7,
    bluePants := 8,
    greenPants := 6,
    greenHats := 8,
    redHats := 8
  }
  countOutfits counts = 688 := by
  sorry

end NUMINAMATH_CALUDE_outfit_count_is_688_l2431_243120


namespace NUMINAMATH_CALUDE_expression_simplification_l2431_243137

theorem expression_simplification (b : ℝ) (h : b ≠ -1/2) :
  1 - (1 / (1 + b / (1 + b))) = b / (1 + 2*b) :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l2431_243137


namespace NUMINAMATH_CALUDE_magical_stack_with_157_has_470_cards_l2431_243162

/-- Represents a stack of cards -/
structure CardStack :=
  (total : ℕ)  -- Total number of cards
  (m : ℕ)      -- Number of cards in each pile
  (isMagical : Bool)  -- Whether the stack is magical

/-- Defines the conditions for a magical stack -/
def isMagicalStack (stack : CardStack) : Prop :=
  stack.total = 2 * stack.m ∧
  stack.isMagical = true ∧
  ∃ (card : ℕ), card ≤ stack.total ∧ card = 157 ∧
    (card % 2 = 1 → card ≤ stack.m) ∧
    (card % 2 = 0 → card > stack.m)

/-- The main theorem to prove -/
theorem magical_stack_with_157_has_470_cards :
  ∀ (stack : CardStack), isMagicalStack stack →
  (157 ≤ stack.m ∧ 157 + (156 / 2) = stack.m) →
  stack.total = 470 := by
  sorry

end NUMINAMATH_CALUDE_magical_stack_with_157_has_470_cards_l2431_243162


namespace NUMINAMATH_CALUDE_A_less_than_B_l2431_243151

theorem A_less_than_B : ∀ x : ℝ, (x + 3) * (x + 7) < (x + 4) * (x + 6) := by
  sorry

end NUMINAMATH_CALUDE_A_less_than_B_l2431_243151


namespace NUMINAMATH_CALUDE_sum_of_cubes_l2431_243164

theorem sum_of_cubes (a b c : ℝ) 
  (sum_eq : a + b + c = 3)
  (sum_prod_eq : a * b + a * c + b * c = 3)
  (prod_eq : a * b * c = 5) :
  a^3 + b^3 + c^3 = 15 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l2431_243164


namespace NUMINAMATH_CALUDE_mod_23_equivalence_l2431_243174

theorem mod_23_equivalence : ∃! n : ℤ, 0 ≤ n ∧ n < 23 ∧ 57846 ≡ n [ZMOD 23] ∧ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_mod_23_equivalence_l2431_243174


namespace NUMINAMATH_CALUDE_discount_equation_proof_l2431_243104

theorem discount_equation_proof (a : ℝ) : 
  (200 * (1 - a / 100)^2 = 148) ↔ 
  (∃ (original_price final_price : ℝ),
    original_price = 200 ∧
    final_price = 148 ∧
    final_price = original_price * (1 - a / 100)^2) :=
by sorry

end NUMINAMATH_CALUDE_discount_equation_proof_l2431_243104


namespace NUMINAMATH_CALUDE_f_is_even_l2431_243141

-- Define g as an even function
def g_even (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = g x

-- Define f in terms of g
def f (g : ℝ → ℝ) (x : ℝ) : ℝ :=
  |g (x^3)|

-- Theorem statement
theorem f_is_even (g : ℝ → ℝ) (h : g_even g) : 
  ∀ x : ℝ, f g (-x) = f g x :=
sorry

end NUMINAMATH_CALUDE_f_is_even_l2431_243141


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l2431_243146

/-- For all real x, the expression 
    sin(2x - π) * cos(x - 3π) + sin(2x - 9π/2) * cos(x + π/2) 
    is equal to sin(3x) -/
theorem trigonometric_simplification (x : ℝ) : 
  Real.sin (2 * x - Real.pi) * Real.cos (x - 3 * Real.pi) + 
  Real.sin (2 * x - 9 * Real.pi / 2) * Real.cos (x + Real.pi / 2) = 
  Real.sin (3 * x) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l2431_243146


namespace NUMINAMATH_CALUDE_sqrt_65_minus_1_bound_l2431_243172

theorem sqrt_65_minus_1_bound (n : ℕ) (hn : 0 < n) :
  (n : ℝ) < Real.sqrt 65 - 1 ∧ Real.sqrt 65 - 1 < (n : ℝ) + 1 → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_65_minus_1_bound_l2431_243172


namespace NUMINAMATH_CALUDE_desired_circle_properties_l2431_243108

-- Define the circles and line
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 4
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y + 4 = 0
def l (x y : ℝ) : Prop := x + 2*y = 0

-- Define the desired circle
def desiredCircle (x y : ℝ) : Prop := x^2 + y^2 - x - 2*y = 0

-- Theorem statement
theorem desired_circle_properties :
  ∀ (x y : ℝ),
    (C₁ x y ∧ C₂ x y → desiredCircle x y) ∧
    (∃ (x₀ y₀ : ℝ), desiredCircle x₀ y₀ ∧ l x₀ y₀ ∧
      ∀ (x y : ℝ), (x - x₀)^2 + (y - y₀)^2 < ε → ¬(desiredCircle x y ∧ l x y))
    := by sorry


end NUMINAMATH_CALUDE_desired_circle_properties_l2431_243108


namespace NUMINAMATH_CALUDE_calculate_expression_l2431_243109

theorem calculate_expression : 
  let a := (5 + 5/9) - 0.8 + (2 + 4/9)
  let b := 7.6 / (4/5) + (2 + 2/5) * 1.25
  a * b = 90 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l2431_243109


namespace NUMINAMATH_CALUDE_specific_grades_average_l2431_243165

/-- The overall average percentage of three subjects -/
def overall_average (math_grade : ℚ) (history_grade : ℚ) (third_subject_grade : ℚ) : ℚ :=
  (math_grade + history_grade + third_subject_grade) / 3

/-- Theorem stating that given specific grades, the overall average is 75% -/
theorem specific_grades_average :
  overall_average 74 84 67 = 75 := by
  sorry

end NUMINAMATH_CALUDE_specific_grades_average_l2431_243165


namespace NUMINAMATH_CALUDE_cubic_polynomial_value_l2431_243191

/-- A cubic polynomial function. -/
def CubicPolynomial (f : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, ∀ x, f x = a * x^3 + b * x^2 + c * x + d

/-- The main theorem stating that a cubic polynomial with given properties has f(1) = -23. -/
theorem cubic_polynomial_value (f : ℝ → ℝ) 
  (hcubic : CubicPolynomial f)
  (h1 : f (-2) = -4)
  (h2 : f 3 = -9)
  (h3 : f (-4) = -16) :
  f 1 = -23 := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_value_l2431_243191


namespace NUMINAMATH_CALUDE_triangle_determinant_l2431_243110

theorem triangle_determinant (A B C : Real) (h1 : A + B + C = π) (h2 : A ≠ π/2 ∧ B ≠ π/2 ∧ C ≠ π/2) : 
  let M : Matrix (Fin 3) (Fin 3) Real := !![2*Real.sin A, 1, 1; 1, 2*Real.sin B, 1; 1, 1, 2*Real.sin C]
  Matrix.det M = 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_determinant_l2431_243110


namespace NUMINAMATH_CALUDE_g_of_three_value_l2431_243171

/-- The function g satisfies 4g(x) - 3g(1/x) = x^2 for all x ≠ 0 -/
def g : ℝ → ℝ :=
  fun x => sorry

/-- The main theorem stating that g(3) = 36.333/7 -/
theorem g_of_three_value : g 3 = 36.333 / 7 := by
  sorry

end NUMINAMATH_CALUDE_g_of_three_value_l2431_243171


namespace NUMINAMATH_CALUDE_largest_number_l2431_243199

theorem largest_number (a b c d e : ℚ) 
  (ha : a = 0.989) 
  (hb : b = 0.998) 
  (hc : c = 0.899) 
  (hd : d = 0.9899) 
  (he : e = 0.8999) : 
  b = max a (max b (max c (max d e))) := by
sorry

end NUMINAMATH_CALUDE_largest_number_l2431_243199


namespace NUMINAMATH_CALUDE_f_three_zeros_implies_a_gt_sqrt_two_l2431_243131

/-- The function f(x) defined in the problem -/
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + 3*a^2*x - 4*a

/-- Theorem stating that if f has three zero points and a > 0, then a > √2 -/
theorem f_three_zeros_implies_a_gt_sqrt_two (a : ℝ) (h1 : a > 0) 
  (h2 : ∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f a x = 0 ∧ f a y = 0 ∧ f a z = 0) :
  a > Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_f_three_zeros_implies_a_gt_sqrt_two_l2431_243131


namespace NUMINAMATH_CALUDE_sum_of_roots_cubic_l2431_243103

theorem sum_of_roots_cubic (x : ℝ) : 
  let f : ℝ → ℝ := λ x => 3 * x^3 - 4 * x^2 - 9 * x
  (∃ a b c : ℝ, f x = (x - a) * (x - b) * (x - c)) →
  a + b + c = 4/3 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_cubic_l2431_243103


namespace NUMINAMATH_CALUDE_units_digit_G_5_l2431_243152

def G (n : ℕ) : ℕ := 2^(3^n) + 2

theorem units_digit_G_5 : G 5 % 10 = 0 := by sorry

end NUMINAMATH_CALUDE_units_digit_G_5_l2431_243152


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l2431_243130

theorem cyclic_sum_inequality (x1 x2 x3 x4 x5 : ℝ) 
  (h_pos : x1 > 0 ∧ x2 > 0 ∧ x3 > 0 ∧ x4 > 0 ∧ x5 > 0) 
  (h_prod : x1 * x2 * x3 * x4 * x5 = 1) : 
  (x1 + x1*x2*x3)/(1 + x1*x2 + x1*x2*x3*x4) +
  (x2 + x2*x3*x4)/(1 + x2*x3 + x2*x3*x4*x5) +
  (x3 + x3*x4*x5)/(1 + x3*x4 + x3*x4*x5*x1) +
  (x4 + x4*x5*x1)/(1 + x4*x5 + x4*x5*x1*x2) +
  (x5 + x5*x1*x2)/(1 + x5*x1 + x5*x1*x2*x3) ≥ 10/3 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l2431_243130


namespace NUMINAMATH_CALUDE_library_book_count_l2431_243187

theorem library_book_count : ∃ (initial_books : ℕ), 
  initial_books = 1750 ∧ 
  initial_books + 140 = (27 * initial_books) / 25 := by
sorry

end NUMINAMATH_CALUDE_library_book_count_l2431_243187


namespace NUMINAMATH_CALUDE_right_triangle_legs_l2431_243148

theorem right_triangle_legs (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a^2 = b^2 + c^2 →
  (1/2) * b * c = 150 →
  a = 25 →
  (b = 20 ∧ c = 15) ∨ (b = 15 ∧ c = 20) := by
sorry

end NUMINAMATH_CALUDE_right_triangle_legs_l2431_243148


namespace NUMINAMATH_CALUDE_total_items_l2431_243114

theorem total_items (bread : ℕ) (milk : ℕ) (cookies : ℕ) 
  (h1 : bread = 58)
  (h2 : bread = milk + 18)
  (h3 : bread = cookies - 27) : 
  bread + milk + cookies = 183 := by
  sorry

end NUMINAMATH_CALUDE_total_items_l2431_243114


namespace NUMINAMATH_CALUDE_number_of_students_selected_l2431_243192

/-- Given a class with boys and girls, prove that the number of students selected is 3 -/
theorem number_of_students_selected
  (num_boys : ℕ)
  (num_girls : ℕ)
  (num_ways : ℕ)
  (h_boys : num_boys = 13)
  (h_girls : num_girls = 10)
  (h_ways : num_ways = 780)
  (h_combination : num_ways = (num_girls.choose 1) * (num_boys.choose 2)) :
  3 = 1 + 2 := by
  sorry

#check number_of_students_selected

end NUMINAMATH_CALUDE_number_of_students_selected_l2431_243192


namespace NUMINAMATH_CALUDE_remainder_of_12345678901_mod_101_l2431_243167

theorem remainder_of_12345678901_mod_101 : 12345678901 % 101 = 24 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_12345678901_mod_101_l2431_243167


namespace NUMINAMATH_CALUDE_probability_six_diamonds_queen_hearts_l2431_243150

/-- Represents a standard deck of 52 cards -/
def StandardDeck : ℕ := 52

/-- Represents the number of each suit in a standard deck -/
def SuitCount : ℕ := 13

/-- Calculates the probability of drawing two specific cards in order from a standard deck -/
def probability_two_specific_cards (deck_size : ℕ) : ℚ :=
  1 / deck_size * (1 / (deck_size - 1))

/-- Theorem: The probability of drawing the 6 of diamonds first and the Queen of hearts second 
    from a standard deck of 52 cards is 1/2652 -/
theorem probability_six_diamonds_queen_hearts : 
  probability_two_specific_cards StandardDeck = 1 / 2652 := by
  sorry

end NUMINAMATH_CALUDE_probability_six_diamonds_queen_hearts_l2431_243150


namespace NUMINAMATH_CALUDE_nearest_integer_to_x_minus_y_is_zero_l2431_243180

theorem nearest_integer_to_x_minus_y_is_zero
  (x y : ℝ) 
  (hx : x ≠ 0)
  (hy : y ≠ 0)
  (h1 : 2 * abs x + y = 5)
  (h2 : abs x * y + x^2 = 0) :
  round (x - y) = 0 :=
sorry

end NUMINAMATH_CALUDE_nearest_integer_to_x_minus_y_is_zero_l2431_243180


namespace NUMINAMATH_CALUDE_intersection_nonempty_implies_b_range_l2431_243190

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + 2*p.2^2 = 3}
def N (m b : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = m*p.1 + b}

-- State the theorem
theorem intersection_nonempty_implies_b_range :
  (∀ m : ℝ, (M ∩ N m b).Nonempty) →
  b ∈ Set.Icc (-Real.sqrt 6 / 2) (Real.sqrt 6 / 2) :=
sorry

end NUMINAMATH_CALUDE_intersection_nonempty_implies_b_range_l2431_243190


namespace NUMINAMATH_CALUDE_age_difference_patrick_nathan_l2431_243121

theorem age_difference_patrick_nathan (patrick michael monica nathan : ℝ) 
  (ratio_patrick_michael : patrick / michael = 3 / 5)
  (ratio_michael_monica : michael / monica = 3 / 4)
  (ratio_monica_nathan : monica / nathan = 5 / 7)
  (sum_ages : patrick + michael + monica + nathan = 228) :
  nathan - patrick = 69.5 := by
sorry

end NUMINAMATH_CALUDE_age_difference_patrick_nathan_l2431_243121


namespace NUMINAMATH_CALUDE_cube_side_length_proof_l2431_243184

/-- The surface area of a cube in square centimeters -/
def surface_area : ℝ := 864

/-- The length of one side of the cube in centimeters -/
def side_length : ℝ := 12

/-- Theorem: For a cube with a surface area of 864 cm², the length of one side is 12 cm -/
theorem cube_side_length_proof :
  6 * side_length ^ 2 = surface_area := by sorry

end NUMINAMATH_CALUDE_cube_side_length_proof_l2431_243184


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2431_243186

-- Define the parabola and hyperbola
def parabola (b : ℝ) (x y : ℝ) : Prop := x^2 = -6*b*y
def hyperbola (a b : ℝ) (x y : ℝ) : Prop := x^2/a^2 - y^2/b^2 = 1

-- Define the points
def point_O : ℝ × ℝ := (0, 0)
def point_A (a b : ℝ) : ℝ × ℝ := (a, 0)

-- Define the angle equality
def angle_equality (O A B C : ℝ × ℝ) : Prop := 
  (C.2 - O.2) / (C.1 - O.1) = (C.2 - B.2) / (C.1 - B.1)

-- Main theorem
theorem hyperbola_eccentricity (a b : ℝ) 
  (ha : a > 0) (hb : b > 0)
  (hB : parabola b (-a*Real.sqrt 13/2) (3*b/2))
  (hC : parabola b (a*Real.sqrt 13/2) (3*b/2))
  (hBC : hyperbola a b (-a*Real.sqrt 13/2) (3*b/2) ∧ 
         hyperbola a b (a*Real.sqrt 13/2) (3*b/2))
  (hAOC : angle_equality point_O (point_A a b) 
    (-a*Real.sqrt 13/2, 3*b/2) (a*Real.sqrt 13/2, 3*b/2)) :
  Real.sqrt (1 + b^2/a^2) = 4*Real.sqrt 3/3 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2431_243186


namespace NUMINAMATH_CALUDE_constant_term_quadratic_equation_l2431_243133

theorem constant_term_quadratic_equation :
  ∀ (x : ℝ), x^2 - 5*x = 2 → ∃ (a b c : ℝ), a = 1 ∧ x^2 + b*x + c = 0 ∧ c = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_constant_term_quadratic_equation_l2431_243133


namespace NUMINAMATH_CALUDE_sqrt_simplification_l2431_243147

theorem sqrt_simplification : 
  Real.sqrt 80 - Real.sqrt 20 + Real.sqrt 5 = 3 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_simplification_l2431_243147


namespace NUMINAMATH_CALUDE_feifei_arrival_time_l2431_243195

/-- Represents the speed of an entity -/
structure Speed :=
  (value : ℝ)

/-- Represents a distance -/
structure Distance :=
  (value : ℝ)

/-- Represents a time duration in minutes -/
structure Duration :=
  (minutes : ℝ)

/-- Represents the scenario of Feifei walking to school -/
structure WalkToSchool :=
  (feifei_speed : Speed)
  (dog_speed : Speed)
  (first_catchup : Distance)
  (second_catchup : Distance)
  (total_distance : Distance)
  (dog_start_delay : Duration)

/-- The theorem stating that Feifei arrives at school 18 minutes after starting -/
theorem feifei_arrival_time (scenario : WalkToSchool) 
  (h1 : scenario.dog_speed.value = 3 * scenario.feifei_speed.value)
  (h2 : scenario.first_catchup.value = 200)
  (h3 : scenario.second_catchup.value = 400)
  (h4 : scenario.total_distance.value = 800)
  (h5 : scenario.dog_start_delay.minutes = 3) :
  ∃ (arrival_time : Duration), arrival_time.minutes = 18 :=
sorry

end NUMINAMATH_CALUDE_feifei_arrival_time_l2431_243195


namespace NUMINAMATH_CALUDE_task_completion_choices_l2431_243175

theorem task_completion_choices (method1 method2 : Finset Nat) : 
  method1.card = 3 → method2.card = 5 → method1 ∩ method2 = ∅ → 
  (method1 ∪ method2).card = 8 :=
by sorry

end NUMINAMATH_CALUDE_task_completion_choices_l2431_243175


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2431_243160

theorem inequality_system_solution (a b : ℝ) :
  (∀ x, x + a > 1 ∧ 2 * x - b < 2 ↔ -2 < x ∧ x < 3) →
  (a - b) ^ 2023 = -1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l2431_243160


namespace NUMINAMATH_CALUDE_four_dice_same_number_probability_l2431_243116

/-- The number of sides on a standard die -/
def standardDieSides : ℕ := 6

/-- The number of dice being tossed -/
def numberOfDice : ℕ := 4

/-- The probability of all dice showing the same number -/
def probabilitySameNumber : ℚ := 1 / (standardDieSides ^ (numberOfDice - 1))

/-- Theorem: The probability of four standard six-sided dice showing the same number when tossed simultaneously is 1/216 -/
theorem four_dice_same_number_probability : 
  probabilitySameNumber = 1 / 216 := by
  sorry

end NUMINAMATH_CALUDE_four_dice_same_number_probability_l2431_243116


namespace NUMINAMATH_CALUDE_max_value_of_a_l2431_243112

open Real

theorem max_value_of_a : 
  (∀ x > 0, Real.exp (x - 1) + 1 ≥ a + Real.log x) → 
  (∀ b, (∀ x > 0, Real.exp (x - 1) + 1 ≥ b + Real.log x) → b ≤ a) → 
  a = 2 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_a_l2431_243112


namespace NUMINAMATH_CALUDE_factorization_equality_l2431_243177

theorem factorization_equality (x y : ℝ) : 
  x^2 - 2*x - 2*y^2 + 4*y - x*y = (x - 2*y)*(x + y - 2) := by
sorry

end NUMINAMATH_CALUDE_factorization_equality_l2431_243177


namespace NUMINAMATH_CALUDE_millet_seed_amount_l2431_243166

/-- Given a mixture of millet and sunflower seeds, prove the amount of millet seed used -/
theorem millet_seed_amount 
  (millet_cost : ℝ) 
  (sunflower_cost : ℝ) 
  (mixture_cost : ℝ) 
  (sunflower_amount : ℝ) 
  (h1 : millet_cost = 0.60) 
  (h2 : sunflower_cost = 1.10) 
  (h3 : mixture_cost = 0.70) 
  (h4 : sunflower_amount = 25) :
  ∃ (millet_amount : ℝ), 
    millet_cost * millet_amount + sunflower_cost * sunflower_amount = 
    mixture_cost * (millet_amount + sunflower_amount) ∧ 
    millet_amount = 100 := by
  sorry

end NUMINAMATH_CALUDE_millet_seed_amount_l2431_243166


namespace NUMINAMATH_CALUDE_line_slope_l2431_243144

/-- The slope of the line x - √3y + 1 = 0 is 1/√3 -/
theorem line_slope (x y : ℝ) : x - Real.sqrt 3 * y + 1 = 0 → (y - 1) / x = 1 / Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_l2431_243144


namespace NUMINAMATH_CALUDE_circle_tangents_k_range_l2431_243185

-- Define the circle C
def C (k : ℝ) (x y : ℝ) : Prop := x^2 + y^2 + 2*k*x + 2*y + k^2 = 0

-- Define the point P
def P : ℝ × ℝ := (1, -1)

-- Define the condition for two tangents
def has_two_tangents (k : ℝ) : Prop := ∃ (t1 t2 : ℝ × ℝ), t1 ≠ t2 ∧ C k t1.1 t1.2 ∧ C k t2.1 t2.2

-- Theorem statement
theorem circle_tangents_k_range :
  ∀ k : ℝ, has_two_tangents k → (k > 0 ∨ k < -2) :=
by sorry

end NUMINAMATH_CALUDE_circle_tangents_k_range_l2431_243185


namespace NUMINAMATH_CALUDE_remaining_books_value_l2431_243132

def total_books : ℕ := 55
def hardback_books : ℕ := 10
def paperback_books : ℕ := total_books - hardback_books
def hardback_price : ℕ := 20
def paperback_price : ℕ := 10
def books_sold : ℕ := 14

def remaining_books : ℕ := total_books - books_sold
def remaining_hardback : ℕ := min hardback_books remaining_books
def remaining_paperback : ℕ := remaining_books - remaining_hardback

def total_value : ℕ := remaining_hardback * hardback_price + remaining_paperback * paperback_price

theorem remaining_books_value :
  total_value = 510 :=
sorry

end NUMINAMATH_CALUDE_remaining_books_value_l2431_243132


namespace NUMINAMATH_CALUDE_dan_marbles_l2431_243127

/-- Given an initial quantity of marbles and a number of marbles given away,
    calculate the remaining number of marbles. -/
def remaining_marbles (initial : ℕ) (given_away : ℕ) : ℕ :=
  initial - given_away

/-- Theorem stating that with 64 initial marbles and 14 given away,
    50 marbles remain. -/
theorem dan_marbles : remaining_marbles 64 14 = 50 := by
  sorry

end NUMINAMATH_CALUDE_dan_marbles_l2431_243127


namespace NUMINAMATH_CALUDE_eggs_per_group_l2431_243143

theorem eggs_per_group (total_eggs : ℕ) (num_groups : ℕ) (eggs_per_group : ℕ) 
  (h1 : total_eggs = 16) 
  (h2 : num_groups = 8) 
  (h3 : eggs_per_group * num_groups = total_eggs) : 
  eggs_per_group = 2 := by
  sorry

end NUMINAMATH_CALUDE_eggs_per_group_l2431_243143


namespace NUMINAMATH_CALUDE_x_power_24_equals_one_l2431_243153

theorem x_power_24_equals_one (x : ℝ) (h : x + 1/x = Real.sqrt 5) : x^24 = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_power_24_equals_one_l2431_243153


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2431_243118

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Theorem: In a geometric sequence where a_3 = 1 and a_5 = 4, a_7 = 16 -/
theorem geometric_sequence_property (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a) 
  (h_3 : a 3 = 1) 
  (h_5 : a 5 = 4) : 
  a 7 = 16 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l2431_243118


namespace NUMINAMATH_CALUDE_negation_of_for_all_leq_zero_l2431_243170

theorem negation_of_for_all_leq_zero :
  (¬ ∀ x : ℝ, Real.exp x - 2 * Real.sin x + 4 ≤ 0) ↔
  (∃ x : ℝ, Real.exp x - 2 * Real.sin x + 4 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_for_all_leq_zero_l2431_243170


namespace NUMINAMATH_CALUDE_one_dollar_bills_count_l2431_243198

/-- Represents the wallet contents -/
structure Wallet where
  ones : ℕ
  twos : ℕ
  fives : ℕ

/-- The wallet satisfies the given conditions -/
def satisfies_conditions (w : Wallet) : Prop :=
  w.ones + w.twos + w.fives = 55 ∧
  w.ones * 1 + w.twos * 2 + w.fives * 5 = 126

/-- The theorem stating the number of one-dollar bills -/
theorem one_dollar_bills_count :
  ∃ (w : Wallet), satisfies_conditions w ∧ w.ones = 18 := by
  sorry

end NUMINAMATH_CALUDE_one_dollar_bills_count_l2431_243198


namespace NUMINAMATH_CALUDE_square_side_length_l2431_243183

theorem square_side_length (x : ℝ) (h : x > 0) : x^2 = 2 * (4 * x) → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l2431_243183


namespace NUMINAMATH_CALUDE_hexagon_perimeter_l2431_243196

/-- The perimeter of a hexagon with side length 5 inches is 30 inches. -/
theorem hexagon_perimeter (side_length : ℝ) (h : side_length = 5) : 
  6 * side_length = 30 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_perimeter_l2431_243196


namespace NUMINAMATH_CALUDE_function_evaluation_l2431_243169

theorem function_evaluation (f : ℝ → ℝ) 
  (h : ∀ x, f (x - 1) = x^2 + 1) : 
  f (-1) = 1 := by
sorry

end NUMINAMATH_CALUDE_function_evaluation_l2431_243169


namespace NUMINAMATH_CALUDE_i_power_sum_l2431_243124

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- Define the property of i
axiom i_squared : i^2 = -1

-- Define the periodicity of i
axiom i_period (n : ℕ) : i^(n + 4) = i^n

-- State the theorem
theorem i_power_sum : i^45 + i^123 = 0 := by sorry

end NUMINAMATH_CALUDE_i_power_sum_l2431_243124


namespace NUMINAMATH_CALUDE_count_valid_a_l2431_243101

theorem count_valid_a : ∃! (S : Finset ℤ), 
  (∀ a ∈ S, (∃! (X : Finset ℤ), (∀ x ∈ X, 6*x - 5 ≥ a ∧ x/4 - (x-1)/6 < 1/2) ∧ X.card = 2) ∧
             (∃ y : ℚ, y > 0 ∧ 4*y - 3*a = 2*(y-3))) ∧
  S.card = 5 :=
sorry

end NUMINAMATH_CALUDE_count_valid_a_l2431_243101


namespace NUMINAMATH_CALUDE_ellipse_line_intersection_l2431_243106

/-- An ellipse with semi-major axis 2 and semi-minor axis √b -/
def Ellipse (b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / 4) + (p.2^2 / b) = 1}

/-- A line with slope m passing through (0, 1) -/
def Line (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = m * p.1 + 1}

/-- The theorem statement -/
theorem ellipse_line_intersection (b : ℝ) :
  (∀ m : ℝ, (Ellipse b ∩ Line m).Nonempty) →
  b ∈ Set.Icc 1 4 ∪ Set.Ioi 4 := by
  sorry


end NUMINAMATH_CALUDE_ellipse_line_intersection_l2431_243106


namespace NUMINAMATH_CALUDE_light_travel_distance_l2431_243158

/-- The distance light travels in one year in miles -/
def light_year_distance : ℝ := 5870000000000

/-- The number of years we want to calculate the light travel distance for -/
def years : ℝ := 150

/-- Theorem stating the distance light travels in 150 years -/
theorem light_travel_distance : light_year_distance * years = 8.805e14 := by
  sorry

end NUMINAMATH_CALUDE_light_travel_distance_l2431_243158


namespace NUMINAMATH_CALUDE_intersection_equals_Q_intersection_empty_l2431_243119

-- Define the sets P and Q
def P : Set ℝ := {x | 2 * x^2 - 5 * x - 3 < 0}
def Q (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 1}

-- Theorem for the first part
theorem intersection_equals_Q (a : ℝ) :
  (P ∩ Q a) = Q a ↔ a ∈ Set.Ioo (-1/2) 2 :=
sorry

-- Theorem for the second part
theorem intersection_empty (a : ℝ) :
  (P ∩ Q a) = ∅ ↔ a ∈ Set.Iic (-3/2) ∪ Set.Ici 3 :=
sorry

end NUMINAMATH_CALUDE_intersection_equals_Q_intersection_empty_l2431_243119


namespace NUMINAMATH_CALUDE_gcf_of_96_144_240_l2431_243111

theorem gcf_of_96_144_240 : Nat.gcd 96 (Nat.gcd 144 240) = 48 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_96_144_240_l2431_243111


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2431_243159

open Set

theorem inequality_solution_set (x : ℝ) :
  let S := {x : ℝ | (x^2 + 2*x + 2) / (x + 2) > 1 ∧ x ≠ -2}
  S = Ioo (-2 : ℝ) (-1) ∪ Ioi 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2431_243159


namespace NUMINAMATH_CALUDE_nike_cost_l2431_243139

theorem nike_cost (total_goal : ℝ) (adidas_cost reebok_cost : ℝ) 
  (nike_sold adidas_sold reebok_sold : ℕ) (excess : ℝ)
  (h1 : total_goal = 1000)
  (h2 : adidas_cost = 45)
  (h3 : reebok_cost = 35)
  (h4 : nike_sold = 8)
  (h5 : adidas_sold = 6)
  (h6 : reebok_sold = 9)
  (h7 : excess = 65) :
  ∃ (nike_cost : ℝ), 
    nike_cost * nike_sold + adidas_cost * adidas_sold + reebok_cost * reebok_sold 
    = total_goal + excess ∧ nike_cost = 60 :=
by sorry

end NUMINAMATH_CALUDE_nike_cost_l2431_243139


namespace NUMINAMATH_CALUDE_class_gpa_theorem_l2431_243113

/-- The grade point average (GPA) of a class -/
def classGPA (n : ℕ) (gpa1 : ℚ) (gpa2 : ℚ) : ℚ :=
  (1 / 3 : ℚ) * gpa1 + (2 / 3 : ℚ) * gpa2

/-- Theorem: The GPA of a class where one-third has a GPA of 45 and two-thirds has a GPA of 60 is 55 -/
theorem class_gpa_theorem :
  classGPA 3 45 60 = 55 := by
  sorry

end NUMINAMATH_CALUDE_class_gpa_theorem_l2431_243113


namespace NUMINAMATH_CALUDE_f_inequality_l2431_243140

-- Define the function f
def f (b c : ℝ) (x : ℝ) : ℝ := x^2 - b*x + c

-- State the theorem
theorem f_inequality (b c : ℝ) :
  (∀ x, f b c (1 + x) = f b c (1 - x)) →
  f b c 0 = 3 →
  ∀ x, f b c (c^x) ≥ f b c (b^x) :=
by sorry

end NUMINAMATH_CALUDE_f_inequality_l2431_243140


namespace NUMINAMATH_CALUDE_linear_equation_solution_l2431_243197

theorem linear_equation_solution (x m : ℝ) : 
  4 * x + 2 * m = 14 → x = 2 → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l2431_243197


namespace NUMINAMATH_CALUDE_transaction_loss_l2431_243128

def property_price : ℝ := 512456

def first_property_gain_percentage : ℝ := 0.25
def second_property_loss_percentage : ℝ := 0.30
def third_property_gain_percentage : ℝ := 0.35
def fourth_property_loss_percentage : ℝ := 0.40

def total_properties : ℕ := 4

def first_property_selling_price : ℝ := property_price * (1 + first_property_gain_percentage)
def second_property_selling_price : ℝ := property_price * (1 - second_property_loss_percentage)
def third_property_selling_price : ℝ := property_price * (1 + third_property_gain_percentage)
def fourth_property_selling_price : ℝ := property_price * (1 - fourth_property_loss_percentage)

def total_cost : ℝ := property_price * total_properties
def total_selling_price : ℝ := first_property_selling_price + second_property_selling_price + third_property_selling_price + fourth_property_selling_price

theorem transaction_loss : total_cost - total_selling_price = 61245.6 := by
  sorry

end NUMINAMATH_CALUDE_transaction_loss_l2431_243128


namespace NUMINAMATH_CALUDE_train_delay_l2431_243189

theorem train_delay (car_time train_time : ℝ) : 
  car_time = 4.5 → 
  car_time + train_time = 11 → 
  train_time - car_time = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_train_delay_l2431_243189


namespace NUMINAMATH_CALUDE_circle_center_is_one_one_l2431_243155

/-- A circle in the 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A point in the 2D plane --/
def Point := ℝ × ℝ

/-- The parabola y = x^2 --/
def parabola (x : ℝ) : ℝ := x^2

/-- Check if a point is on a circle --/
def isOnCircle (c : Circle) (p : Point) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

/-- Check if a circle is tangent to the parabola at a given point --/
def isTangentToParabola (c : Circle) (p : Point) : Prop :=
  isOnCircle c p ∧ p.2 = parabola p.1 ∧
  ∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), 0 < |x - p.1| ∧ |x - p.1| < ε →
    (x, parabola x) ∉ {q : Point | isOnCircle c q}

theorem circle_center_is_one_one :
  ∀ (c : Circle),
    isOnCircle c (0, 1) →
    isTangentToParabola c (1, 1) →
    c.center = (1, 1) := by
  sorry

end NUMINAMATH_CALUDE_circle_center_is_one_one_l2431_243155


namespace NUMINAMATH_CALUDE_rogers_initial_money_l2431_243117

theorem rogers_initial_money (initial_money : ℕ) : 
  (initial_money - 47 = 3 * 7) → initial_money = 68 := by
  sorry

end NUMINAMATH_CALUDE_rogers_initial_money_l2431_243117


namespace NUMINAMATH_CALUDE_job_choice_diploma_percentage_l2431_243188

theorem job_choice_diploma_percentage :
  let total_population : ℝ := 100
  let no_diploma_with_job : ℝ := 18
  let with_job_choice : ℝ := 40
  let with_diploma : ℝ := 37
  let without_job_choice : ℝ := total_population - with_job_choice
  let with_diploma_without_job : ℝ := with_diploma - (with_job_choice - no_diploma_with_job)
  (with_diploma_without_job / without_job_choice) * 100 = 25 := by
sorry

end NUMINAMATH_CALUDE_job_choice_diploma_percentage_l2431_243188


namespace NUMINAMATH_CALUDE_problem_solution_l2431_243176

-- Define proposition p
def p : Prop := ∀ x : ℝ, x^2 + x + 1 ≥ 0

-- Define proposition q
def q : Prop := ∀ x : ℝ, (x > 1 → x > 2) ∧ ¬(x > 2 → x > 1)

-- Theorem to prove
theorem problem_solution : p ∧ ¬q := by sorry

end NUMINAMATH_CALUDE_problem_solution_l2431_243176


namespace NUMINAMATH_CALUDE_simplify_expression_l2431_243194

theorem simplify_expression : 
  1 - 1 / (2 + Real.sqrt 5) + 1 / (2 - Real.sqrt 5) = 1 - 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2431_243194


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l2431_243163

theorem diophantine_equation_solutions :
  ∀ x y : ℤ, x^2 + y^2 + 3^3 = 456 * (x - y).sqrt →
    ((x = 30 ∧ y = 21) ∨ (x = -21 ∧ y = -30)) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l2431_243163


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l2431_243123

def U : Set Nat := {1, 2, 3, 4, 5, 6, 7}
def A : Set Nat := {2, 4, 6}
def B : Set Nat := {1, 3, 5, 7}

theorem intersection_A_complement_B : A ∩ (U \ B) = {2, 4, 6} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l2431_243123


namespace NUMINAMATH_CALUDE_liam_target_time_l2431_243173

-- Define Mia's run
def mia_distance : ℕ := 5
def mia_time : ℕ := 45

-- Define Liam's initial run
def liam_initial_distance : ℕ := 3

-- Define the relationship between Liam and Mia's times
def liam_initial_time : ℚ := mia_time / 3

-- Define Liam's target distance
def liam_target_distance : ℕ := 7

-- Theorem to prove
theorem liam_target_time : 
  (liam_target_distance : ℚ) * (liam_initial_time / liam_initial_distance) = 35 := by
  sorry

end NUMINAMATH_CALUDE_liam_target_time_l2431_243173


namespace NUMINAMATH_CALUDE_pizza_percentage_left_l2431_243126

theorem pizza_percentage_left (ravindra_ate hongshu_ate pizza_left : ℚ) : 
  ravindra_ate = 2/5 →
  hongshu_ate = ravindra_ate/2 →
  pizza_left = 1 - (ravindra_ate + hongshu_ate) →
  pizza_left * 100 = 40 := by
sorry

end NUMINAMATH_CALUDE_pizza_percentage_left_l2431_243126


namespace NUMINAMATH_CALUDE_first_term_of_geometric_sequence_l2431_243136

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem first_term_of_geometric_sequence 
  (a : ℕ → ℝ) 
  (h_geo : is_geometric_sequence a) 
  (h_prod : a 2 * a 3 * a 4 = 27) 
  (h_a7 : a 7 = 27) :
  a 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_first_term_of_geometric_sequence_l2431_243136


namespace NUMINAMATH_CALUDE_no_consecutive_digit_products_exist_l2431_243134

/-- Given a natural number, return the product of its digits -/
def digitProduct (n : ℕ) : ℕ := sorry

theorem no_consecutive_digit_products_exist (n : ℕ) : 
  let x := digitProduct n
  let y := digitProduct (n + 1)
  ¬∃ m : ℕ, digitProduct m = y - 1 ∧ digitProduct (m + 1) = x - 1 := by
  sorry

end NUMINAMATH_CALUDE_no_consecutive_digit_products_exist_l2431_243134


namespace NUMINAMATH_CALUDE_derivative_f_at_one_l2431_243157

noncomputable def f (x : ℝ) : ℝ := x^2 + Real.exp x

theorem derivative_f_at_one :
  deriv f 1 = 2 + Real.exp 1 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_one_l2431_243157


namespace NUMINAMATH_CALUDE_relationship_abc_l2431_243161

theorem relationship_abc (a b c : ℝ) : 
  a = (0.4 : ℝ) ^ (0.2 : ℝ) → 
  b = (0.4 : ℝ) ^ (0.6 : ℝ) → 
  c = (2.1 : ℝ) ^ (0.2 : ℝ) → 
  c > a ∧ a > b := by
sorry

end NUMINAMATH_CALUDE_relationship_abc_l2431_243161


namespace NUMINAMATH_CALUDE_intersection_dot_product_l2431_243193

/-- Given a line ax + by + c = 0 intersecting a circle x^2 + y^2 = 4 at points A and B,
    prove that the dot product of OA and OB is -2 when c^2 = a^2 + b^2 -/
theorem intersection_dot_product
  (a b c : ℝ) 
  (A B : ℝ × ℝ)
  (h1 : ∀ (x y : ℝ), a * x + b * y + c = 0 → x^2 + y^2 = 4 → (x, y) = A ∨ (x, y) = B)
  (h2 : c^2 = a^2 + b^2) :
  A.1 * B.1 + A.2 * B.2 = -2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_dot_product_l2431_243193


namespace NUMINAMATH_CALUDE_sugar_salt_difference_is_two_l2431_243179

/-- A baking recipe with specified amounts of ingredients -/
structure Recipe where
  sugar : ℕ
  flour : ℕ
  salt : ℕ

/-- The amount of ingredients Mary has already added -/
structure Added where
  flour : ℕ

/-- Calculate the difference between required sugar and salt -/
def sugarSaltDifference (recipe : Recipe) : ℤ :=
  recipe.sugar - recipe.salt

/-- Theorem: The difference between required sugar and salt is 2 cups -/
theorem sugar_salt_difference_is_two (recipe : Recipe) (added : Added) :
  recipe.sugar = 11 →
  recipe.flour = 6 →
  recipe.salt = 9 →
  added.flour = 12 →
  sugarSaltDifference recipe = 2 := by
  sorry

#eval sugarSaltDifference { sugar := 11, flour := 6, salt := 9 }

end NUMINAMATH_CALUDE_sugar_salt_difference_is_two_l2431_243179


namespace NUMINAMATH_CALUDE_marbles_given_to_dylan_l2431_243125

/-- Given that Cade had 87 marbles initially and was left with 79 marbles after giving some to Dylan,
    prove that Cade gave 8 marbles to Dylan. -/
theorem marbles_given_to_dylan :
  ∀ (initial_marbles remaining_marbles marbles_given : ℕ),
    initial_marbles = 87 →
    remaining_marbles = 79 →
    initial_marbles = remaining_marbles + marbles_given →
    marbles_given = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_marbles_given_to_dylan_l2431_243125


namespace NUMINAMATH_CALUDE_shortest_light_path_length_shortest_light_path_equals_12_l2431_243178

/-- The shortest path length of a light ray reflecting off the x-axis -/
theorem shortest_light_path_length : ℝ :=
  let A : ℝ × ℝ := (-3, 9)
  let C : ℝ × ℝ := (2, 3)  -- Center of the circle
  let r : ℝ := 1  -- Radius of the circle
  let C' : ℝ × ℝ := (2, -3)  -- Reflection of C across x-axis
  let AC' : ℝ := Real.sqrt ((-3 - 2)^2 + (9 - (-3))^2)
  AC' - r

theorem shortest_light_path_equals_12 :
  shortest_light_path_length = 12 := by sorry

end NUMINAMATH_CALUDE_shortest_light_path_length_shortest_light_path_equals_12_l2431_243178


namespace NUMINAMATH_CALUDE_simple_pairs_l2431_243168

theorem simple_pairs (n : ℕ) (h : n > 3) :
  ∃ (p₁ p₂ : ℕ), Nat.Prime p₁ ∧ Nat.Prime p₂ ∧ Odd p₁ ∧ Odd p₂ ∧ (p₂ ∣ (2 * n - p₁)) :=
sorry

end NUMINAMATH_CALUDE_simple_pairs_l2431_243168


namespace NUMINAMATH_CALUDE_first_catchup_theorem_second_catchup_theorem_l2431_243100

/-- Represents a circular track with two runners -/
structure Track :=
  (length : ℝ)
  (speed_fast : ℝ)
  (speed_slow : ℝ)

/-- Calculates the distance covered by each runner at first catch-up -/
def first_catchup (track : Track) : ℝ × ℝ := sorry

/-- Calculates the number of laps completed by each runner at second catch-up -/
def second_catchup (track : Track) : ℕ × ℕ := sorry

/-- Theorem for the first catch-up distances -/
theorem first_catchup_theorem (track : Track) 
  (h1 : track.length = 400)
  (h2 : track.speed_fast = 7)
  (h3 : track.speed_slow = 5) :
  first_catchup track = (1400, 1000) := by sorry

/-- Theorem for the second catch-up laps -/
theorem second_catchup_theorem (track : Track)
  (h1 : track.length = 400)
  (h2 : track.speed_fast = 7)
  (h3 : track.speed_slow = 5) :
  second_catchup track = (7, 5) := by sorry

end NUMINAMATH_CALUDE_first_catchup_theorem_second_catchup_theorem_l2431_243100


namespace NUMINAMATH_CALUDE_geometric_sequence_a3_l2431_243102

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_a3 (a : ℕ → ℝ) :
  geometric_sequence a → a 1 = 1 → a 5 = 4 → a 3 = 2 ∨ a 3 = -2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a3_l2431_243102


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2431_243135

def A : Set ℕ := {0, 1, 3, 5, 7}
def B : Set ℕ := {2, 4, 6, 8, 0}

theorem intersection_of_A_and_B : A ∩ B = {0} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2431_243135


namespace NUMINAMATH_CALUDE_range_of_a_l2431_243107

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x - 2*a| + |2*x - a| ≥ a^2) → -3/2 ≤ a ∧ a ≤ 3/2 :=
by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2431_243107


namespace NUMINAMATH_CALUDE_peach_stand_count_l2431_243149

/-- Calculates the number of peaches at Mike's fruit stand after various operations. -/
def final_peach_count (initial : ℝ) (picked : ℝ) (spoiled : ℝ) (sold : ℝ) : ℝ :=
  initial + picked - spoiled - sold

/-- Theorem stating that given the specific numbers from the problem, 
    the final peach count is 81.0. -/
theorem peach_stand_count : 
  final_peach_count 34.0 86.0 12.0 27.0 = 81.0 := by
  sorry

end NUMINAMATH_CALUDE_peach_stand_count_l2431_243149


namespace NUMINAMATH_CALUDE_robbery_participants_l2431_243156

/-- Represents a gangster --/
inductive Gangster : Type
  | Harry : Gangster
  | James : Gangster
  | Donald : Gangster
  | George : Gangster
  | Charlie : Gangster
  | Tom : Gangster

/-- Represents a statement made by a gangster about who participated in the robbery --/
def Statement : Gangster → Gangster × Gangster
  | Gangster.Harry => (Gangster.Charlie, Gangster.George)
  | Gangster.James => (Gangster.Donald, Gangster.Tom)
  | Gangster.Donald => (Gangster.Tom, Gangster.Charlie)
  | Gangster.George => (Gangster.Harry, Gangster.Charlie)
  | Gangster.Charlie => (Gangster.Donald, Gangster.James)
  | Gangster.Tom => (Gangster.Tom, Gangster.Tom)  -- Placeholder statement for Tom

/-- Checks if a gangster's statement is correct given the actual participants --/
def isCorrectStatement (g : Gangster) (participants : Gangster × Gangster) : Prop :=
  (Statement g).1 = participants.1 ∨ (Statement g).1 = participants.2 ∨
  (Statement g).2 = participants.1 ∨ (Statement g).2 = participants.2

/-- The main theorem stating that Charlie and James participated in the robbery --/
theorem robbery_participants :
  ∃ (participants : Gangster × Gangster),
    participants = (Gangster.Charlie, Gangster.James) ∧
    participants.1 ≠ Gangster.Tom ∧
    participants.2 ≠ Gangster.Tom ∧
    (∃ (incorrect : Gangster),
      incorrect ≠ participants.1 ∧
      incorrect ≠ participants.2 ∧
      ¬isCorrectStatement incorrect participants) ∧
    (∀ (g : Gangster),
      g ≠ participants.1 ∧
      g ≠ participants.2 ∧
      g ≠ Gangster.Tom →
      isCorrectStatement g participants) :=
  sorry


end NUMINAMATH_CALUDE_robbery_participants_l2431_243156


namespace NUMINAMATH_CALUDE_ice_cream_combinations_l2431_243105

theorem ice_cream_combinations : 
  (Nat.choose 7 2) = 21 := by sorry

end NUMINAMATH_CALUDE_ice_cream_combinations_l2431_243105


namespace NUMINAMATH_CALUDE_sum_of_specific_values_l2431_243182

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem sum_of_specific_values (f : ℝ → ℝ) 
  (h_odd : is_odd_function f)
  (h_period : has_period f 3)
  (h_f_1 : f 1 = 2014) :
  f 2013 + f 2014 + f 2015 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_specific_values_l2431_243182

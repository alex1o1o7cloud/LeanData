import Mathlib

namespace smallest_marble_count_l759_75970

/-- Represents the number of marbles of each color in the urn -/
structure MarbleCount where
  red : ℕ
  white : ℕ
  blue : ℕ
  green : ℕ

/-- Calculates the total number of marbles in the urn -/
def total_marbles (mc : MarbleCount) : ℕ :=
  mc.red + mc.white + mc.blue + mc.green

/-- Calculates the number of ways to select marbles according to the given events -/
def event_probability (mc : MarbleCount) (event : Fin 4) : ℕ :=
  match event with
  | 0 => mc.blue.choose 4
  | 1 => (mc.red.choose 2) * (mc.white.choose 2)
  | 2 => (mc.red.choose 2) * (mc.white.choose 1) * (mc.blue.choose 1)
  | 3 => mc.red * mc.white * mc.blue * mc.green

/-- Checks if all events have equal probability -/
def events_equally_likely (mc : MarbleCount) : Prop :=
  ∀ i j : Fin 4, event_probability mc i = event_probability mc j

/-- The main theorem stating the smallest number of marbles satisfying the conditions -/
theorem smallest_marble_count : 
  ∃ (mc : MarbleCount), 
    events_equally_likely mc ∧ 
    total_marbles mc = 13 ∧ 
    (∀ (mc' : MarbleCount), events_equally_likely mc' → total_marbles mc' ≥ 13) :=
sorry

end smallest_marble_count_l759_75970


namespace three_valid_configurations_l759_75989

/-- Represents a square in the figure -/
structure Square :=
  (id : Nat)

/-- Represents the cross-shaped figure -/
def CrossFigure := List Square

/-- Represents the additional squares -/
def AdditionalSquares := List Square

/-- Represents a configuration after adding a square to the cross figure -/
def Configuration := CrossFigure × Square

/-- Checks if a configuration can be folded into a topless cubical box -/
def canFoldIntoCube (config : Configuration) : Bool :=
  sorry

/-- The main theorem stating that exactly three configurations can be folded into a topless cubical box -/
theorem three_valid_configurations 
  (cross : CrossFigure) 
  (additional : AdditionalSquares) : 
  (cross.length = 5) → 
  (additional.length = 8) → 
  (∃! (n : Nat), n = (List.filter canFoldIntoCube (List.map (λ s => (cross, s)) additional)).length ∧ n = 3) :=
sorry

end three_valid_configurations_l759_75989


namespace number_of_red_balls_l759_75913

theorem number_of_red_balls (blue : ℕ) (green : ℕ) (red : ℕ) : 
  blue = 3 → green = 1 → (red : ℚ) / (blue + green + red : ℚ) = 1 / 2 → red = 2 := by
  sorry

end number_of_red_balls_l759_75913


namespace second_discount_percentage_l759_75961

theorem second_discount_percentage (original_price : ℝ) (first_discount : ℝ) (final_price : ℝ) : 
  original_price = 10000 →
  first_discount = 20 →
  final_price = 6840 →
  ∃ (second_discount : ℝ),
    final_price = original_price * (1 - first_discount / 100) * (1 - second_discount / 100) ∧
    second_discount = 14.5 := by
  sorry

end second_discount_percentage_l759_75961


namespace equation_solution_l759_75975

theorem equation_solution : ∃ (a b c d : ℕ+), 
  2014 = (a.val ^ 2 + b.val ^ 2) * (c.val ^ 3 - d.val ^ 3) ∧ 
  a.val = 5 ∧ b.val = 9 ∧ c.val = 3 ∧ d.val = 2 := by
  sorry

end equation_solution_l759_75975


namespace candy_distribution_proof_l759_75905

/-- Given a number of candy pieces and sisters, returns the minimum number of pieces to remove for equal distribution. -/
def minPiecesToRemove (totalPieces sisters : ℕ) : ℕ :=
  totalPieces % sisters

theorem candy_distribution_proof :
  minPiecesToRemove 20 3 = 2 := by
  sorry

#eval minPiecesToRemove 20 3

end candy_distribution_proof_l759_75905


namespace square_area_equal_perimeter_l759_75983

/-- The area of a square with perimeter equal to a triangle with sides 6.1, 8.2, and 9.7 -/
theorem square_area_equal_perimeter (s : Real) (h1 : s > 0) 
  (h2 : 4 * s = 6.1 + 8.2 + 9.7) : s^2 = 36 := by
  sorry

end square_area_equal_perimeter_l759_75983


namespace arithmetic_sequence_ratio_l759_75939

theorem arithmetic_sequence_ratio : 
  let n1 := (60 - 4) / 4 + 1
  let n2 := (75 - 5) / 5 + 1
  let sum1 := n1 * (4 + 60) / 2
  let sum2 := n2 * (5 + 75) / 2
  sum1 / sum2 = 4 / 5 := by sorry

end arithmetic_sequence_ratio_l759_75939


namespace horner_method_v3_l759_75914

def horner_polynomial (x : ℝ) : ℝ := 12 + 35*x - 8*x^2 + 79*x^3 + 6*x^4 + 5*x^5 + 3*x^6

def horner_v1 (x : ℝ) : ℝ := 3*x + 5

def horner_v2 (x : ℝ) : ℝ := horner_v1 x * x + 6

def horner_v3 (x : ℝ) : ℝ := horner_v2 x * x + 79

theorem horner_method_v3 :
  horner_v3 (-4) = -57 :=
by sorry

end horner_method_v3_l759_75914


namespace logs_per_tree_l759_75915

/-- The number of pieces of firewood produced from one log -/
def pieces_per_log : ℕ := 5

/-- The total number of pieces of firewood chopped -/
def total_pieces : ℕ := 500

/-- The number of trees chopped down -/
def trees_chopped : ℕ := 25

/-- Theorem: Given the conditions, each tree produces 4 logs -/
theorem logs_per_tree : 
  (total_pieces / pieces_per_log) / trees_chopped = 4 := by
  sorry

end logs_per_tree_l759_75915


namespace fraction_to_decimal_l759_75999

theorem fraction_to_decimal : (58 : ℚ) / 125 = (464 : ℚ) / 1000 := by sorry

end fraction_to_decimal_l759_75999


namespace x1_value_l759_75916

theorem x1_value (x1 x2 x3 x4 : ℝ) 
  (h_order : 0 ≤ x4 ∧ x4 ≤ x3 ∧ x3 ≤ x2 ∧ x2 ≤ x1 ∧ x1 ≤ 1) 
  (h_sum : (1-x1)^2 + (x1-x2)^2 + (x2-x3)^2 + (x3-x4)^2 + x4^2 = 9/16) : 
  x1 = 1 - 15 / Real.sqrt 80 := by
  sorry

end x1_value_l759_75916


namespace line_AB_equation_l759_75900

-- Define the ellipses
def C₁ (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1
def C₂ (x y : ℝ) : Prop := y^2 / 16 + x^2 / 4 = 1

-- Define points A and B
def A : ℝ × ℝ → Prop := λ p => C₁ p.1 p.2
def B : ℝ × ℝ → Prop := λ p => C₂ p.1 p.2

-- Define the relation between OA and OB
def OB_eq_2OA (a b : ℝ × ℝ) : Prop := b.1 = 2 * a.1 ∧ b.2 = 2 * a.2

-- Theorem statement
theorem line_AB_equation (a b : ℝ × ℝ) (ha : A a) (hb : B b) (hab : OB_eq_2OA a b) :
  (b.2 - a.2) / (b.1 - a.1) = 1 ∨ (b.2 - a.2) / (b.1 - a.1) = -1 :=
sorry

end line_AB_equation_l759_75900


namespace equilateral_hyperbola_equation_l759_75938

-- Define the hyperbola
def Hyperbola (x y : ℝ) := x^2 - y^2 = 8

-- Define the line containing one focus
def FocusLine (x y : ℝ) := 3*x - 4*y + 12 = 0

-- Theorem statement
theorem equilateral_hyperbola_equation :
  ∃ (a b : ℝ), 
    -- One focus is on the line
    FocusLine a b ∧
    -- The focus is on the x-axis (real axis)
    b = 0 ∧
    -- The hyperbola passes through this focus
    Hyperbola a b ∧
    -- The hyperbola is equilateral (a² = b²)
    a^2 = (8:ℝ) := by
  sorry

end equilateral_hyperbola_equation_l759_75938


namespace two_numbers_problem_l759_75982

theorem two_numbers_problem : ∃ (x y : ℕ), 
  (x + y = 1244) ∧ 
  (10 * x + 3 = (y - 2) / 10) ∧
  (x = 12) ∧ 
  (y = 1232) := by
  sorry

end two_numbers_problem_l759_75982


namespace equilateral_triangle_third_vertex_y_coord_l759_75937

/-- Given an equilateral triangle with two vertices at (3,7) and (13,7),
    prove that the y-coordinate of the third vertex in the first quadrant is 7 + 5√3 -/
theorem equilateral_triangle_third_vertex_y_coord :
  ∀ (x y : ℝ),
  let A : ℝ × ℝ := (3, 7)
  let B : ℝ × ℝ := (13, 7)
  let C : ℝ × ℝ := (x, y)
  (x > 0 ∧ y > 0) →  -- C is in the first quadrant
  (dist A B = dist B C ∧ dist B C = dist C A) →  -- Triangle is equilateral
  y = 7 + 5 * Real.sqrt 3 :=
by sorry

end equilateral_triangle_third_vertex_y_coord_l759_75937


namespace parallelogram_vertex_sum_l759_75925

/-- A parallelogram with vertices A, B, C, and D in 2D space -/
structure Parallelogram :=
  (A B C D : ℝ × ℝ)

/-- The property that the diagonals of a parallelogram bisect each other -/
def diagonals_bisect (p : Parallelogram) : Prop :=
  let midpoint_AD := ((p.A.1 + p.D.1) / 2, (p.A.2 + p.D.2) / 2)
  let midpoint_BC := ((p.B.1 + p.C.1) / 2, (p.B.2 + p.C.2) / 2)
  midpoint_AD = midpoint_BC

/-- The sum of coordinates of a point -/
def sum_coordinates (p : ℝ × ℝ) : ℝ :=
  p.1 + p.2

/-- The main theorem -/
theorem parallelogram_vertex_sum :
  ∀ (p : Parallelogram),
    p.A = (2, 3) →
    p.B = (5, 7) →
    p.D = (11, -1) →
    diagonals_bisect p →
    sum_coordinates p.C = 3 :=
by sorry

end parallelogram_vertex_sum_l759_75925


namespace equilateral_triangle_condition_l759_75947

/-- A function that checks if a natural number n satisfies the conditions for forming an equilateral triangle with sticks of lengths 1 to n. -/
def can_form_equilateral_triangle (n : ℕ) : Prop :=
  n ≥ 5 ∧ (n % 6 = 0 ∨ n % 6 = 2 ∨ n % 6 = 3 ∨ n % 6 = 5)

/-- The sum of the first n natural numbers. -/
def sum_of_first_n (n : ℕ) : ℕ :=
  n * (n + 1) / 2

/-- Theorem stating the necessary and sufficient conditions for forming an equilateral triangle with sticks of lengths 1 to n. -/
theorem equilateral_triangle_condition (n : ℕ) :
  (sum_of_first_n n % 3 = 0) ↔ can_form_equilateral_triangle n :=
sorry

end equilateral_triangle_condition_l759_75947


namespace book_choice_theorem_l759_75903

/-- The number of ways to choose 1 book from different sets of books -/
def choose_one_book (diff_lit : Nat) (diff_math : Nat) (id_lit : Nat) (id_math : Nat) : Nat :=
  if diff_lit > 0 ∧ diff_math > 0 then
    diff_lit + diff_math
  else if id_lit > 0 ∧ id_math > 0 then
    (if diff_math > 0 then diff_math + 1 else 2)
  else
    0

/-- Theorem stating the number of ways to choose 1 book in different scenarios -/
theorem book_choice_theorem :
  (choose_one_book 5 4 0 0 = 9) ∧
  (choose_one_book 0 0 5 4 = 5) ∧
  (choose_one_book 0 4 5 0 = 2) := by
  sorry

end book_choice_theorem_l759_75903


namespace only_integer_solution_l759_75967

theorem only_integer_solution (x y z : ℝ) (n : ℤ) : 
  x ≥ 0 → y ≥ 0 → z ≥ 0 → 
  2 * x^2 + 3 * y^2 + 6 * z^2 = n →
  3 * x + 4 * y + 5 * z = 23 →
  n = 127 :=
by sorry

end only_integer_solution_l759_75967


namespace paint_mixture_fraction_l759_75919

theorem paint_mixture_fraction (original_intensity replacement_intensity new_intensity : ℝ) 
  (h1 : original_intensity = 0.5)
  (h2 : replacement_intensity = 0.25)
  (h3 : new_intensity = 0.45) :
  ∃ (x : ℝ), 
    x ≥ 0 ∧ x ≤ 1 ∧
    original_intensity * (1 - x) + replacement_intensity * x = new_intensity ∧
    x = 0.2 := by
  sorry

end paint_mixture_fraction_l759_75919


namespace vector_sum_l759_75956

theorem vector_sum : 
  let v1 : Fin 3 → ℝ := ![4, -8, 10]
  let v2 : Fin 3 → ℝ := ![-7, 12, -15]
  v1 + v2 = ![-3, 4, -5] := by sorry

end vector_sum_l759_75956


namespace beau_age_proof_l759_75981

theorem beau_age_proof (sons_age_today : ℕ) (sons_are_triplets : Bool) : 
  sons_age_today = 16 ∧ sons_are_triplets = true → 42 = (sons_age_today - 3) * 3 + 3 :=
by
  sorry

end beau_age_proof_l759_75981


namespace sum_remainder_l759_75936

theorem sum_remainder (a b c : ℕ) : 
  a < 7 → b < 7 → c < 7 → a > 0 → b > 0 → c > 0 →
  (a * b * c) % 7 = 2 →
  (3 * c) % 7 = 1 →
  (4 * b) % 7 = (2 + b) % 7 →
  (a + b + c) % 7 = 3 := by
sorry

end sum_remainder_l759_75936


namespace chocolate_candy_cost_difference_l759_75901

/-- The difference in cost between chocolate and candy bar --/
def cost_difference (chocolate_cost candy_cost : ℕ) : ℕ :=
  chocolate_cost - candy_cost

/-- Theorem stating the difference in cost between chocolate and candy bar --/
theorem chocolate_candy_cost_difference :
  let chocolate_cost : ℕ := 7
  let candy_cost : ℕ := 2
  cost_difference chocolate_cost candy_cost = 5 := by
sorry

end chocolate_candy_cost_difference_l759_75901


namespace restaurant_location_l759_75980

theorem restaurant_location (A B C : ℝ × ℝ) : 
  let road_y : ℝ := 0
  let A_x : ℝ := 0
  let A_y : ℝ := 300
  let B_y : ℝ := road_y
  let dist_AB : ℝ := 500
  A = (A_x, A_y) →
  B.2 = road_y →
  Real.sqrt ((B.1 - A_x)^2 + (B.2 - A_y)^2) = dist_AB →
  C.2 = road_y →
  Real.sqrt ((C.1 - A_x)^2 + (C.2 - A_y)^2) = Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) →
  C.1 = 200 := by
sorry

end restaurant_location_l759_75980


namespace smallest_top_cube_sum_divisible_by_four_l759_75990

/-- Represents the configuration of the bottom layer of the pyramid -/
structure BottomLayer :=
  (a b c d e f g h i : ℕ)
  (all_different : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧
                   b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧
                   c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧
                   d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧
                   e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧
                   f ≠ g ∧ f ≠ h ∧ f ≠ i ∧
                   g ≠ h ∧ g ≠ i ∧
                   h ≠ i)

/-- Calculates the sum of the top cube given the bottom layer configuration -/
def topCubeSum (bl : BottomLayer) : ℕ :=
  bl.a + bl.c + bl.g + bl.i + 2 * (bl.b + bl.d + bl.f + bl.h) + 4 * bl.e

/-- Theorem stating that the smallest possible sum for the top cube divisible by 4 is 64 -/
theorem smallest_top_cube_sum_divisible_by_four :
  ∀ bl : BottomLayer, ∃ n : ℕ, n ≥ topCubeSum bl ∧ n % 4 = 0 ∧ n ≥ 64 :=
sorry

end smallest_top_cube_sum_divisible_by_four_l759_75990


namespace sum_of_products_l759_75995

theorem sum_of_products (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 281) 
  (h2 : a + b + c = 17) : 
  a*b + b*c + c*a = 4 := by
sorry

end sum_of_products_l759_75995


namespace binomial_square_constant_l759_75929

theorem binomial_square_constant (c : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, 9 * x^2 - 24 * x + c = (a * x + b)^2) → c = 16 := by
  sorry

end binomial_square_constant_l759_75929


namespace rabbit_clearing_theorem_l759_75935

/-- Represents the area one rabbit can clear in a day given the land dimensions, number of rabbits, and days to clear -/
def rabbit_clearing_rate (length width : ℕ) (num_rabbits days_to_clear : ℕ) : ℚ :=
  (length * width : ℚ) / 9 / (num_rabbits * days_to_clear)

/-- Theorem stating that given the specific conditions, one rabbit clears 10 square yards per day -/
theorem rabbit_clearing_theorem :
  rabbit_clearing_rate 200 900 100 20 = 10 := by
  sorry

#eval rabbit_clearing_rate 200 900 100 20

end rabbit_clearing_theorem_l759_75935


namespace equation_roots_l759_75969

theorem equation_roots (k : ℝ) : 
  (∃ x y : ℂ, x ≠ y ∧ 
    (x / (x + 1) + 2 * x / (x + 3) = k * x) ∧ 
    (y / (y + 1) + 2 * y / (y + 3) = k * y) ∧
    (∀ z : ℂ, z / (z + 1) + 2 * z / (z + 3) = k * z → z = x ∨ z = y)) ↔ 
  k = 5/3 :=
by sorry

end equation_roots_l759_75969


namespace plot_length_is_56_l759_75940

/-- Proves that the length of a rectangular plot is 56 meters given the specified conditions -/
theorem plot_length_is_56 (breadth : ℝ) (length : ℝ) (perimeter : ℝ) (cost_per_meter : ℝ) (total_cost : ℝ) :
  length = breadth + 12 →
  perimeter = 2 * (length + breadth) →
  cost_per_meter = 26.5 →
  total_cost = 5300 →
  total_cost = cost_per_meter * perimeter →
  length = 56 := by
  sorry

end plot_length_is_56_l759_75940


namespace correct_average_weight_l759_75907

/-- Proves that the correct average weight of a class is 61.2 kg given the initial miscalculation and corrections. -/
theorem correct_average_weight 
  (num_students : ℕ) 
  (initial_average : ℝ)
  (student_A_misread student_A_correct : ℝ)
  (student_B_misread student_B_correct : ℝ)
  (student_C_misread student_C_correct : ℝ)
  (h1 : num_students = 30)
  (h2 : initial_average = 60.2)
  (h3 : student_A_misread = 54)
  (h4 : student_A_correct = 64)
  (h5 : student_B_misread = 58)
  (h6 : student_B_correct = 68)
  (h7 : student_C_misread = 50)
  (h8 : student_C_correct = 60) :
  (num_students : ℝ) * initial_average + 
  (student_A_correct - student_A_misread) + 
  (student_B_correct - student_B_misread) + 
  (student_C_correct - student_C_misread) / num_students = 61.2 := by
  sorry

end correct_average_weight_l759_75907


namespace incorrect_simplification_l759_75997

theorem incorrect_simplification : 
  -(1 + 1/2) ≠ 1 + 1/2 := by
  sorry

end incorrect_simplification_l759_75997


namespace matchstick_rearrangement_l759_75917

theorem matchstick_rearrangement : |(22 : ℝ) / 7 - Real.pi| < (1 : ℝ) / 10 := by
  sorry

end matchstick_rearrangement_l759_75917


namespace probability_theorem_l759_75930

structure ProfessionalGroup where
  women_percentage : ℝ
  men_percentage : ℝ
  nonbinary_percentage : ℝ
  women_engineer_percentage : ℝ
  women_doctor_percentage : ℝ
  men_engineer_percentage : ℝ
  men_doctor_percentage : ℝ
  nonbinary_engineer_percentage : ℝ
  nonbinary_translator_percentage : ℝ

def probability_selection (group : ProfessionalGroup) : ℝ :=
  group.women_percentage * group.women_engineer_percentage +
  group.men_percentage * group.men_doctor_percentage +
  group.nonbinary_percentage * group.nonbinary_translator_percentage

theorem probability_theorem (group : ProfessionalGroup) 
  (h1 : group.women_percentage = 0.70)
  (h2 : group.men_percentage = 0.20)
  (h3 : group.nonbinary_percentage = 0.10)
  (h4 : group.women_engineer_percentage = 0.20)
  (h5 : group.men_doctor_percentage = 0.20)
  (h6 : group.nonbinary_translator_percentage = 0.20) :
  probability_selection group = 0.20 := by
  sorry

end probability_theorem_l759_75930


namespace negation_of_existential_proposition_l759_75963

theorem negation_of_existential_proposition :
  ¬(∃ x : ℝ, x < 0 ∧ x^2 - 2*x > 0) ↔ (∀ x : ℝ, x < 0 → x^2 - 2*x ≤ 0) := by
  sorry

end negation_of_existential_proposition_l759_75963


namespace expression_value_at_negative_two_l759_75979

theorem expression_value_at_negative_two :
  let x : ℤ := -2
  (3 * x + 4)^2 - 2 * x = 8 := by sorry

end expression_value_at_negative_two_l759_75979


namespace remainder_2519_div_7_l759_75911

theorem remainder_2519_div_7 : 2519 % 7 = 6 := by
  sorry

end remainder_2519_div_7_l759_75911


namespace triangle_area_from_altitudes_l759_75984

/-- Given a triangle ABC with altitudes h_a, h_b, and h_c, 
    its area S is equal to 
    1 / sqrt((1/h_a + 1/h_b + 1/h_c) * (1/h_a + 1/h_b - 1/h_c) * 
             (1/h_a + 1/h_c - 1/h_b) * (1/h_b + 1/h_c - 1/h_a)) -/
theorem triangle_area_from_altitudes (h_a h_b h_c : ℝ) (h_pos_a : h_a > 0) (h_pos_b : h_b > 0) (h_pos_c : h_c > 0) :
  let S := 1 / Real.sqrt ((1/h_a + 1/h_b + 1/h_c) * (1/h_a + 1/h_b - 1/h_c) * 
                          (1/h_a + 1/h_c - 1/h_b) * (1/h_b + 1/h_c - 1/h_a))
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    S = (a * h_a) / 2 ∧ S = (b * h_b) / 2 ∧ S = (c * h_c) / 2 := by
  sorry

end triangle_area_from_altitudes_l759_75984


namespace no_fraction_satisfies_conditions_l759_75976

theorem no_fraction_satisfies_conditions : ¬∃ (a b n : ℕ), 
  (a < b) ∧ 
  (n < a) ∧ 
  (n < b) ∧ 
  ((a + n : ℚ) / (b + n)) > (3 / 2) * (a / b) ∧
  ((a - n : ℚ) / (b - n)) > (1 / 2) * (a / b) := by
  sorry

end no_fraction_satisfies_conditions_l759_75976


namespace quadratic_equation_roots_condition_l759_75949

/-- 
Given a quadratic equation kx^2 - 2x - 1 = 0 with two distinct real roots,
prove that the range of values for k is k > -1 and k ≠ 0.
-/
theorem quadratic_equation_roots_condition (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ k * x^2 - 2 * x - 1 = 0 ∧ k * y^2 - 2 * y - 1 = 0) →
  (k > -1 ∧ k ≠ 0) :=
by sorry

end quadratic_equation_roots_condition_l759_75949


namespace animal_arrangement_count_l759_75998

def num_chickens : ℕ := 5
def num_dogs : ℕ := 3
def num_cats : ℕ := 6
def total_animals : ℕ := num_chickens + num_dogs + num_cats

def group_arrangements : ℕ := 3

theorem animal_arrangement_count :
  (group_arrangements * num_chickens.factorial * num_dogs.factorial * num_cats.factorial : ℕ) = 1555200 :=
by sorry

end animal_arrangement_count_l759_75998


namespace reflection_theorem_l759_75942

/-- A reflection in 2D space -/
structure Reflection2D where
  /-- The function that performs the reflection -/
  apply : ℝ × ℝ → ℝ × ℝ

/-- Theorem: Given a reflection that maps (3, -2) to (7, 6), it will map (0, 4) to (80/29, -84/29) -/
theorem reflection_theorem (r : Reflection2D) 
  (h1 : r.apply (3, -2) = (7, 6)) :
  r.apply (0, 4) = (80/29, -84/29) := by
sorry


end reflection_theorem_l759_75942


namespace field_fully_fenced_l759_75955

/-- Proves that a square field can be completely fenced given the specified conditions -/
theorem field_fully_fenced (field_area : ℝ) (wire_cost : ℝ) (budget : ℝ) : 
  field_area = 5000 → 
  wire_cost = 30 → 
  budget = 120000 → 
  ∃ (wire_length : ℝ), wire_length = budget / wire_cost ∧ 
    wire_length ≥ 4 * Real.sqrt field_area := by
  sorry

end field_fully_fenced_l759_75955


namespace sandy_first_shop_books_l759_75953

/-- Represents the problem of Sandy's book purchases -/
def SandyBookProblem (first_shop_books : ℕ) : Prop :=
  let total_spent : ℕ := 2160
  let second_shop_books : ℕ := 55
  let average_price : ℕ := 18
  (total_spent : ℚ) / (first_shop_books + second_shop_books : ℚ) = average_price

/-- Proves that Sandy bought 65 books from the first shop -/
theorem sandy_first_shop_books :
  SandyBookProblem 65 := by sorry

end sandy_first_shop_books_l759_75953


namespace response_rate_is_sixty_percent_l759_75986

/-- The response rate percentage for a mail questionnaire -/
def response_rate (responses_needed : ℕ) (questionnaires_mailed : ℕ) : ℚ :=
  (responses_needed : ℚ) / (questionnaires_mailed : ℚ) * 100

/-- Theorem: The response rate is 60% when 750 responses are needed and 1250 questionnaires are mailed -/
theorem response_rate_is_sixty_percent :
  response_rate 750 1250 = 60 := by
  sorry

#eval response_rate 750 1250

end response_rate_is_sixty_percent_l759_75986


namespace power_of_negative_cube_l759_75922

theorem power_of_negative_cube (a : ℝ) : (-a^3)^4 = a^12 := by
  sorry

end power_of_negative_cube_l759_75922


namespace car_round_trip_speed_l759_75964

theorem car_round_trip_speed 
  (distance : ℝ) 
  (speed_there : ℝ) 
  (avg_speed : ℝ) 
  (speed_back : ℝ) : 
  distance = 150 → 
  speed_there = 75 → 
  avg_speed = 50 → 
  (2 * distance) / (distance / speed_there + distance / speed_back) = avg_speed →
  speed_back = 37.5 := by
sorry

end car_round_trip_speed_l759_75964


namespace baron_munchausen_contradiction_l759_75966

theorem baron_munchausen_contradiction (d : ℝ) (T : ℝ) (h1 : d > 0) (h2 : T > 0) : 
  ¬(d / 2 = 5 * (d / (2 * 5)) ∧ d / 2 = 6 * (T / 2)) :=
by
  sorry

end baron_munchausen_contradiction_l759_75966


namespace average_listening_time_is_44_l759_75948

/-- Represents the distribution of audience members and their listening durations -/
structure AudienceDistribution where
  total_audience : ℕ
  lecture_duration : ℕ
  full_listeners_percent : ℚ
  non_listeners_percent : ℚ
  half_listeners_percent : ℚ

/-- Calculates the average listening time given an audience distribution -/
def average_listening_time (dist : AudienceDistribution) : ℚ :=
  sorry

/-- The theorem stating that the average listening time is 44 minutes -/
theorem average_listening_time_is_44 (dist : AudienceDistribution) : 
  dist.lecture_duration = 90 ∧ 
  dist.full_listeners_percent = 30/100 ∧ 
  dist.non_listeners_percent = 15/100 ∧
  dist.half_listeners_percent = 40/100 * (1 - dist.full_listeners_percent - dist.non_listeners_percent) →
  average_listening_time dist = 44 :=
sorry

end average_listening_time_is_44_l759_75948


namespace average_age_of_new_men_l759_75965

theorem average_age_of_new_men (n : ℕ) (initial_average : ℝ) 
  (replaced_ages : List ℝ) (age_increase : ℝ) :
  n = 20 ∧ 
  replaced_ages = [21, 23, 25, 27] ∧ 
  age_increase = 2 →
  (n * (initial_average + age_increase) - n * initial_average + replaced_ages.sum) / replaced_ages.length = 34 := by
  sorry

end average_age_of_new_men_l759_75965


namespace stating_count_valid_starters_l759_75968

/-- 
Represents the number of boys who can start the game to ensure it goes for at least a full turn 
in a circular arrangement of m boys and n girls.
-/
def valid_starters (m n : ℕ) : ℕ :=
  m - n

/-- 
Theorem stating that the number of valid starters is m - n, 
given that there are more boys than girls.
-/
theorem count_valid_starters (m n : ℕ) (h : m > n) : 
  valid_starters m n = m - n := by
  sorry

end stating_count_valid_starters_l759_75968


namespace solve_average_weight_l759_75923

def average_weight_problem (num_boys num_girls : ℕ) (avg_weight_boys avg_weight_girls : ℚ) : Prop :=
  let total_children := num_boys + num_girls
  let total_weight := (num_boys : ℚ) * avg_weight_boys + (num_girls : ℚ) * avg_weight_girls
  let avg_weight_all := total_weight / total_children
  (↑(round avg_weight_all) : ℚ) = 141

theorem solve_average_weight :
  average_weight_problem 8 5 160 110 := by
  sorry

end solve_average_weight_l759_75923


namespace exists_x_sin_minus_x_negative_l759_75904

open Real

theorem exists_x_sin_minus_x_negative :
  ∃ x : ℝ, 0 < x ∧ x < π / 2 ∧ sin x - x < 0 := by
  sorry

end exists_x_sin_minus_x_negative_l759_75904


namespace betty_age_l759_75972

theorem betty_age (albert mary betty : ℕ) 
  (h1 : albert = 2 * mary)
  (h2 : albert = 4 * betty)
  (h3 : mary = albert - 10) :
  betty = 5 := by
sorry

end betty_age_l759_75972


namespace vector_equation_solution_l759_75946

def vector_sum (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 + b.1, a.2 + b.2)

def scalar_mult (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)

theorem vector_equation_solution (x y : ℝ) 
  (h : vector_sum (x, 1) (scalar_mult 2 (2, y)) = (5, -3)) : 
  x + y = -1 := by
  sorry

end vector_equation_solution_l759_75946


namespace circle_hexagon_area_difference_l759_75950

theorem circle_hexagon_area_difference (r : ℝ) (s : ℝ) : 
  r = (Real.sqrt 2) / 2 →
  s = 1 →
  (π * r^2) - (3 * Real.sqrt 3 / 2 * s^2) = π / 2 - 3 * Real.sqrt 3 / 2 := by
  sorry

end circle_hexagon_area_difference_l759_75950


namespace intersection_with_complement_l759_75909

open Set

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def S : Set ℕ := {1, 4, 5}
def T : Set ℕ := {2, 3, 4}

theorem intersection_with_complement : S ∩ (U \ T) = {1, 5} := by sorry

end intersection_with_complement_l759_75909


namespace f_properties_l759_75921

noncomputable def f (x : ℝ) : ℝ := 2^(Real.sin x) + 2^(-Real.sin x)

theorem f_properties :
  -- f is an even function
  (∀ x, f (-x) = f x) ∧
  -- π is a period of f
  (∀ x, f (x + Real.pi) = f x) ∧
  -- π is a local minimum of f
  (∃ ε > 0, ∀ x, x ∈ Set.Ioo (Real.pi - ε) (Real.pi + ε) → f Real.pi ≤ f x) ∧
  -- f is strictly increasing on (0, π/2)
  (∀ x y, x ∈ Set.Ioo 0 (Real.pi / 2) → y ∈ Set.Ioo 0 (Real.pi / 2) → x < y → f x < f y) :=
by sorry

end f_properties_l759_75921


namespace circle_equation_l759_75944

/-- Theorem: Equation of a circle with specific properties -/
theorem circle_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∀ y : ℝ, (0 - a)^2 + (y - b)^2 = (a^2 + b^2) → |y| ≤ 1) →
  (∀ x : ℝ, (x - a)^2 + (0 - b)^2 = (a^2 + b^2) → |x| ≤ 2) →
  (∃ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ > 0 ∧ 
    (x₁ - a)^2 + (0 - b)^2 = (a^2 + b^2) ∧
    (x₂ - a)^2 + (0 - b)^2 = (a^2 + b^2) ∧
    (x₂ - x₁) / (4 - (x₂ - x₁)) = 3) →
  a = Real.sqrt 7 ∧ b = 2 ∧ a^2 + b^2 = 8 := by
sorry

end circle_equation_l759_75944


namespace travel_ways_theorem_l759_75991

/-- The number of different ways to travel between two places given the number of bus, train, and ferry routes -/
def total_travel_ways (buses trains ferries : ℕ) : ℕ :=
  buses + trains + ferries

/-- Theorem stating that with 5 buses, 6 trains, and 2 ferries, there are 13 ways to travel -/
theorem travel_ways_theorem :
  total_travel_ways 5 6 2 = 13 := by
  sorry

end travel_ways_theorem_l759_75991


namespace angle_A_measure_l759_75924

-- Define the triangle and its properties
structure Triangle :=
  (A B C : ℝ)
  (sum_of_angles : A + B + C = 180)

-- Define the configuration
def geometric_configuration (t : Triangle) (x y : ℝ) : Prop :=
  t.B = 120 ∧ 
  x = 50 ∧
  y = 130 ∧
  x + (180 - y) + t.C = 180

-- Theorem statement
theorem angle_A_measure (t : Triangle) (x y : ℝ) 
  (h : geometric_configuration t x y) : t.A = 120 :=
sorry

end angle_A_measure_l759_75924


namespace max_radius_circle_through_points_l759_75926

/-- A circle in a rectangular coordinate system -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if a point lies on a circle -/
def lieOnCircle (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

/-- The maximum possible radius of a circle passing through (16, 0) and (-16, 0) is 16 -/
theorem max_radius_circle_through_points :
  ∃ (c : Circle), lieOnCircle c (16, 0) ∧ lieOnCircle c (-16, 0) →
  ∀ (c' : Circle), lieOnCircle c' (16, 0) ∧ lieOnCircle c' (-16, 0) →
  c'.radius ≤ 16 :=
sorry

end max_radius_circle_through_points_l759_75926


namespace valid_distributions_count_l759_75959

/-- Represents a triangular array of squares with 11 rows -/
def TriangularArray := Fin 11 → Fin 11 → ℕ

/-- Represents the bottom row of the triangular array -/
def BottomRow := Fin 11 → Fin 2

/-- Calculates the value of a square in the array based on the two squares below it -/
def calculateSquare (array : TriangularArray) (row : Fin 11) (col : Fin 11) : ℕ :=
  if row = 10 then array row col
  else array (row + 1) col + array (row + 1) (col + 1)

/-- Fills the triangular array based on the bottom row -/
def fillArray (bottomRow : BottomRow) : TriangularArray :=
  sorry

/-- Checks if the top square of the array is a multiple of 3 -/
def isTopMultipleOfThree (array : TriangularArray) : Bool :=
  array 0 0 % 3 = 0

/-- Counts the number of valid bottom row distributions -/
def countValidDistributions : ℕ :=
  sorry

theorem valid_distributions_count :
  countValidDistributions = 640 := by sorry

end valid_distributions_count_l759_75959


namespace kids_from_unnamed_school_l759_75920

theorem kids_from_unnamed_school (riverside_total : ℕ) (mountaintop_total : ℕ) (total_admitted : ℕ)
  (riverside_denied_percent : ℚ) (mountaintop_denied_percent : ℚ) (unnamed_denied_percent : ℚ)
  (h1 : riverside_total = 120)
  (h2 : mountaintop_total = 50)
  (h3 : total_admitted = 148)
  (h4 : riverside_denied_percent = 1/5)
  (h5 : mountaintop_denied_percent = 1/2)
  (h6 : unnamed_denied_percent = 7/10) :
  ∃ (unnamed_total : ℕ),
    unnamed_total = 90 ∧
    total_admitted = 
      (riverside_total - riverside_total * riverside_denied_percent) +
      (mountaintop_total - mountaintop_total * mountaintop_denied_percent) +
      (unnamed_total - unnamed_total * unnamed_denied_percent) :=
by sorry

end kids_from_unnamed_school_l759_75920


namespace apple_juice_production_l759_75957

/-- Calculates the amount of apples used for apple juice production in million tons -/
def applesForJuice (totalApples : ℝ) (ciderPercent : ℝ) (freshPercent : ℝ) (juicePercent : ℝ) : ℝ :=
  let ciderApples := ciderPercent * totalApples
  let remainingApples := totalApples - ciderApples
  let freshApples := freshPercent * remainingApples
  let exportedApples := remainingApples - freshApples
  juicePercent * exportedApples

theorem apple_juice_production :
  applesForJuice 6 0.3 0.4 0.6 = 1.512 := by
  sorry

end apple_juice_production_l759_75957


namespace triangle_side_difference_l759_75958

-- Define the triangle sides
def side1 : ℝ := 7
def side2 : ℝ := 10

-- Define the valid range for x
def valid_x (x : ℤ) : Prop :=
  x > 0 ∧ x + side1 > side2 ∧ x + side2 > side1 ∧ side1 + side2 > x

-- Theorem statement
theorem triangle_side_difference :
  (∃ (max min : ℤ), 
    (∀ x : ℤ, valid_x x → x ≤ max) ∧
    (∀ x : ℤ, valid_x x → x ≥ min) ∧
    (∀ x : ℤ, valid_x x → min ≤ x ∧ x ≤ max) ∧
    (max - min = 12)) :=
sorry

end triangle_side_difference_l759_75958


namespace translation_theorem_l759_75985

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Translation of a point in 2D space -/
def translate (p : Point2D) (dx dy : ℝ) : Point2D :=
  ⟨p.x + dx, p.y + dy⟩

/-- Theorem: Given a translation of line segment AB to A'B', 
    if A(-2,3) corresponds to A'(3,2) and B corresponds to B'(4,0), 
    then the coordinates of B are (-1,1) -/
theorem translation_theorem 
  (A : Point2D) (A' : Point2D) (B' : Point2D)
  (h1 : A = ⟨-2, 3⟩)
  (h2 : A' = ⟨3, 2⟩)
  (h3 : B' = ⟨4, 0⟩)
  (h4 : ∃ (dx dy : ℝ), A' = translate A dx dy ∧ B' = translate ⟨-1, 1⟩ dx dy) :
  ∃ (B : Point2D), B = ⟨-1, 1⟩ ∧ B' = translate B (A'.x - A.x) (A'.y - A.y) :=
sorry

end translation_theorem_l759_75985


namespace nathaniel_tickets_l759_75993

/-- Given a person with initial tickets who gives a fixed number of tickets to each of their friends,
    calculate the number of remaining tickets. -/
def remaining_tickets (initial : ℕ) (given_per_friend : ℕ) (num_friends : ℕ) : ℕ :=
  initial - given_per_friend * num_friends

/-- Theorem stating that given 11 initial tickets, giving 2 tickets to each of 4 friends
    results in 3 remaining tickets. -/
theorem nathaniel_tickets : remaining_tickets 11 2 4 = 3 := by
  sorry

end nathaniel_tickets_l759_75993


namespace lines_dont_intersect_if_points_not_coplanar_l759_75918

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A line in 3D space defined by two points -/
structure Line3D where
  p1 : Point3D
  p2 : Point3D

/-- Check if four points are coplanar -/
def are_coplanar (a b c d : Point3D) : Prop := sorry

/-- Check if two lines intersect -/
def lines_intersect (l1 l2 : Line3D) : Prop := sorry

theorem lines_dont_intersect_if_points_not_coplanar 
  (a b c d : Point3D) 
  (h : ¬ are_coplanar a b c d) : 
  ¬ lines_intersect (Line3D.mk a b) (Line3D.mk c d) := by
  sorry

end lines_dont_intersect_if_points_not_coplanar_l759_75918


namespace circles_externally_tangent_l759_75971

/-- Two circles are externally tangent if the distance between their centers
    equals the sum of their radii -/
def externally_tangent (r1 r2 d : ℝ) : Prop := d = r1 + r2

/-- Given two circles with radii 2 and 3, whose centers are 5 units apart,
    prove that they are externally tangent -/
theorem circles_externally_tangent :
  let r1 : ℝ := 2
  let r2 : ℝ := 3
  let d : ℝ := 5
  externally_tangent r1 r2 d := by
sorry

end circles_externally_tangent_l759_75971


namespace matrix_cube_l759_75954

def A : Matrix (Fin 2) (Fin 2) ℝ := !![2, -1; 1, 1]

theorem matrix_cube : A^3 = !![3, -6; 6, -3] := by sorry

end matrix_cube_l759_75954


namespace three_lines_intersection_l759_75910

/-- The curve (x + 2y + a)(x^2 - y^2) = 0 represents three lines intersecting at a single point if and only if a = 0 -/
theorem three_lines_intersection (a : ℝ) : 
  (∃! p : ℝ × ℝ, ∀ x y : ℝ, (x + 2*y + a)*(x^2 - y^2) = 0 ↔ 
    (x = p.1 ∧ y = p.2) ∨ (x = -y ∧ x = p.1) ∨ (x = y ∧ x = p.1)) ↔ 
  a = 0 :=
sorry

end three_lines_intersection_l759_75910


namespace fermat_like_equation_l759_75987

theorem fermat_like_equation (a b c : ℕ) (h1 : Even c) (h2 : a^5 + 4*b^5 = c^5) : b = 0 := by
  sorry

end fermat_like_equation_l759_75987


namespace sin_sum_inverse_sin_tan_l759_75912

theorem sin_sum_inverse_sin_tan : 
  Real.sin (Real.arcsin (4/5) + Real.arctan 3) = 13 * Real.sqrt 10 / 50 := by
sorry

end sin_sum_inverse_sin_tan_l759_75912


namespace ceiling_2023_ceiling_quadratic_inequality_ceiling_equality_distance_l759_75933

-- Define the ceiling function
noncomputable def ceiling (x : ℝ) : ℤ :=
  Int.ceil x

-- Theorem 1
theorem ceiling_2023 (x : ℝ) :
  ceiling x = 2023 → x ∈ Set.Ioo 2022 2023 := by sorry

-- Theorem 2
theorem ceiling_quadratic_inequality (x : ℝ) :
  (ceiling x)^2 - 5*(ceiling x) + 6 ≤ 0 → x ∈ Set.Ioo 1 3 := by sorry

-- Theorem 3
theorem ceiling_equality_distance (x y : ℝ) :
  ceiling x = ceiling y → |x - y| < 1 := by sorry

end ceiling_2023_ceiling_quadratic_inequality_ceiling_equality_distance_l759_75933


namespace union_cardinality_lower_bound_equality_holds_l759_75994

theorem union_cardinality_lower_bound 
  (A B C : Finset ℕ) 
  (h : A ∩ B ∩ C = ∅) : 
  (A ∪ B ∪ C).card ≥ (A.card + B.card + C.card) / 2 := by
  sorry

def equality_example : Finset ℕ × Finset ℕ × Finset ℕ :=
  ({1, 2}, {2, 3}, {3, 1})

theorem equality_holds (A B C : Finset ℕ) 
  (h : (A, B, C) = equality_example) :
  (A ∪ B ∪ C).card = (A.card + B.card + C.card) / 2 := by
  sorry

end union_cardinality_lower_bound_equality_holds_l759_75994


namespace sum_of_fractions_l759_75927

theorem sum_of_fractions : (1 : ℚ) / 3 + (1 : ℚ) / 4 = 7 / 12 := by sorry

end sum_of_fractions_l759_75927


namespace inequality_for_negative_numbers_l759_75928

theorem inequality_for_negative_numbers (a b : ℝ) (h : a < b ∧ b < 0) :
  a^2 > a*b ∧ a*b > b^2 := by
  sorry

end inequality_for_negative_numbers_l759_75928


namespace greatest_integer_b_for_all_real_domain_l759_75978

theorem greatest_integer_b_for_all_real_domain : ∃ (b : ℤ), 
  (∀ (x : ℝ), x^2 + (b : ℝ) * x + 7 ≠ 0) ∧ 
  (∀ (b' : ℤ), (∀ (x : ℝ), x^2 + (b' : ℝ) * x + 7 ≠ 0) → b' ≤ b) ∧
  b = 5 := by sorry

end greatest_integer_b_for_all_real_domain_l759_75978


namespace algebraic_simplification_l759_75931

theorem algebraic_simplification (x y : ℝ) :
  (3 * x - 2 * y - 4) * (x + y + 5) - (x + 2 * y + 5) * (3 * x - y - 1) =
  -4 * x * y - 3 * x - 7 * y - 15 := by
  sorry

end algebraic_simplification_l759_75931


namespace incorrect_yeast_experiment_method_l759_75941

/-- Represents an experiment exploring dynamic changes of yeast cell numbers --/
structure YeastExperiment where
  /-- Whether the experiment requires repeated trials --/
  requires_repeated_trials : Bool
  /-- Whether the experiment needs a control group --/
  needs_control_group : Bool

/-- Theorem stating that the incorrect method for yeast cell number experiments 
    is the one claiming no need for repeated trials or control group --/
theorem incorrect_yeast_experiment_method :
  ∀ (e : YeastExperiment), 
    (e.requires_repeated_trials = true) → 
    ¬(e.requires_repeated_trials = false ∧ e.needs_control_group = false) :=
by sorry

end incorrect_yeast_experiment_method_l759_75941


namespace probability_of_prime_sum_two_dice_l759_75906

/-- A function that returns true if a number is prime, false otherwise -/
def isPrime (n : ℕ) : Bool := sorry

/-- The number of sides on each die -/
def numSides : ℕ := 8

/-- The set of possible sums when rolling two dice -/
def possibleSums : Finset ℕ := sorry

/-- The set of prime sums when rolling two dice -/
def primeSums : Finset ℕ := sorry

/-- The total number of possible outcomes when rolling two dice -/
def totalOutcomes : ℕ := numSides * numSides

theorem probability_of_prime_sum_two_dice :
  (Finset.card primeSums : ℚ) / totalOutcomes = 23 / 64 := by sorry

end probability_of_prime_sum_two_dice_l759_75906


namespace max_photo_area_l759_75943

/-- Given a rectangular frame with area 59.6 square centimeters,
    prove that the maximum area of each of four equal-sized,
    non-overlapping photos within the frame is 14.9 square centimeters. -/
theorem max_photo_area (frame_area : ℝ) (num_photos : ℕ) :
  frame_area = 59.6 ∧ num_photos = 4 →
  (frame_area / num_photos : ℝ) = 14.9 := by
  sorry

end max_photo_area_l759_75943


namespace sticker_distribution_l759_75902

theorem sticker_distribution (gold : ℕ) (students : ℕ) : 
  gold = 50 →
  students = 5 →
  (gold + 2 * gold + (2 * gold - 20)) / students = 46 := by
  sorry

end sticker_distribution_l759_75902


namespace prob_red_pen_is_two_fifths_l759_75945

/-- The number of colored pens -/
def total_pens : ℕ := 5

/-- The number of pens to be selected -/
def selected_pens : ℕ := 2

/-- The number of ways to select 2 pens out of 5 -/
def total_selections : ℕ := Nat.choose total_pens selected_pens

/-- The number of ways to select a red pen and another different color -/
def red_selections : ℕ := total_pens - 1

/-- The probability of selecting a red pen when choosing 2 different colored pens out of 5 -/
def prob_red_pen : ℚ := red_selections / total_selections

theorem prob_red_pen_is_two_fifths : prob_red_pen = 2 / 5 := by
  sorry

end prob_red_pen_is_two_fifths_l759_75945


namespace distinct_roots_iff_m_lt_half_m_value_when_inverse_sum_neg_two_l759_75932

/-- Given a quadratic equation x^2 - 2(m-1)x + m^2 = 0 -/
def quadratic_equation (m x : ℝ) : Prop :=
  x^2 - 2*(m-1)*x + m^2 = 0

/-- The discriminant of the quadratic equation -/
def discriminant (m : ℝ) : ℝ :=
  (-2*(m-1))^2 - 4*m^2

theorem distinct_roots_iff_m_lt_half (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic_equation m x₁ ∧ quadratic_equation m x₂) ↔ m < 1/2 :=
sorry

theorem m_value_when_inverse_sum_neg_two (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic_equation m x₁ ∧ quadratic_equation m x₂ ∧ 1/x₁ + 1/x₂ = -2) →
  m = (-1 - Real.sqrt 5) / 2 :=
sorry

end distinct_roots_iff_m_lt_half_m_value_when_inverse_sum_neg_two_l759_75932


namespace expression_evaluation_l759_75952

theorem expression_evaluation :
  11 + Real.sqrt (-4 + 6 * 4 / 3) = 13 := by
  sorry

end expression_evaluation_l759_75952


namespace decompose_375_l759_75973

theorem decompose_375 : 
  375 = 3 * 100 + 7 * 10 + 5 * 1 := by sorry

end decompose_375_l759_75973


namespace geometric_progression_min_sum_l759_75951

/-- A geometric progression with positive terms -/
def GeometricProgression (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n ∧ a n > 0

theorem geometric_progression_min_sum (a : ℕ → ℝ) (h : GeometricProgression a) 
    (h_prod : a 2 * a 10 = 9) : 
  a 5 + a 7 ≥ 6 := by
  sorry

end geometric_progression_min_sum_l759_75951


namespace three_pipes_used_l759_75992

def tank_filling_problem (rate_a rate_b rate_c : ℝ) : Prop :=
  let total_rate := rate_a + rate_b + rate_c
  rate_c = 2 * rate_b ∧
  rate_b = 2 * rate_a ∧
  rate_a = 1 / 70 ∧
  total_rate = 1 / 10

theorem three_pipes_used (rate_a rate_b rate_c : ℝ) 
  (h : tank_filling_problem rate_a rate_b rate_c) : 
  ∃ (n : ℕ), n = 3 ∧ n > 0 := by
  sorry

#check three_pipes_used

end three_pipes_used_l759_75992


namespace quadratic_one_solution_sum_l759_75934

theorem quadratic_one_solution_sum (a : ℝ) : 
  (∃! x, 3 * x^2 + a * x + 6 * x + 7 = 0) ↔ 
  (a = -6 + 2 * Real.sqrt 21 ∨ a = -6 - 2 * Real.sqrt 21) ∧
  (-6 + 2 * Real.sqrt 21) + (-6 - 2 * Real.sqrt 21) = -12 :=
by sorry

end quadratic_one_solution_sum_l759_75934


namespace correct_assignment_properties_l759_75962

-- Define the properties of assignment statements
inductive AssignmentProperty : Type
  | InitialValue : AssignmentProperty
  | AssignExpression : AssignmentProperty
  | MultipleAssignments : AssignmentProperty
  | NoMultipleAssignments : AssignmentProperty

-- Define a function to check if a property is correct
def isCorrectProperty (prop : AssignmentProperty) : Prop :=
  match prop with
  | AssignmentProperty.InitialValue => True
  | AssignmentProperty.AssignExpression => True
  | AssignmentProperty.MultipleAssignments => True
  | AssignmentProperty.NoMultipleAssignments => False

-- Theorem stating the correct properties of assignment statements
theorem correct_assignment_properties :
  ∀ (prop : AssignmentProperty),
    isCorrectProperty prop ↔
      (prop = AssignmentProperty.InitialValue ∨
       prop = AssignmentProperty.AssignExpression ∨
       prop = AssignmentProperty.MultipleAssignments) :=
by sorry

end correct_assignment_properties_l759_75962


namespace spencer_walk_distance_l759_75960

theorem spencer_walk_distance (house_to_library : ℝ) (library_to_post_office : ℝ) (post_office_to_home : ℝ)
  (h1 : house_to_library = 0.3)
  (h2 : library_to_post_office = 0.1)
  (h3 : post_office_to_home = 0.4) :
  house_to_library + library_to_post_office + post_office_to_home = 0.8 := by
  sorry

end spencer_walk_distance_l759_75960


namespace haleigh_cats_count_l759_75977

/-- The number of cats Haleigh has -/
def num_cats : ℕ := 10

/-- The number of dogs Haleigh has -/
def num_dogs : ℕ := 4

/-- The total number of leggings needed -/
def total_leggings : ℕ := 14

/-- Each animal needs one pair of leggings -/
def leggings_per_animal : ℕ := 1

theorem haleigh_cats_count :
  num_cats = total_leggings - num_dogs * leggings_per_animal :=
by sorry

end haleigh_cats_count_l759_75977


namespace data_transmission_time_l759_75974

/-- Represents the number of blocks to be sent -/
def num_blocks : ℕ := 30

/-- Represents the number of chunks in each block -/
def chunks_per_block : ℕ := 1024

/-- Represents the transmission rate in chunks per second -/
def transmission_rate : ℕ := 256

/-- Represents the number of seconds in a minute -/
def seconds_per_minute : ℕ := 60

/-- Proves that the time to send the data is 2 minutes -/
theorem data_transmission_time :
  (num_blocks * chunks_per_block) / transmission_rate / seconds_per_minute = 2 :=
sorry

end data_transmission_time_l759_75974


namespace sum_of_two_squares_equivalence_l759_75988

theorem sum_of_two_squares_equivalence (n : ℤ) : 
  (∃ (a b : ℤ), n = a^2 + b^2) ↔ (∃ (u v : ℤ), 2*n = u^2 + v^2) := by
  sorry

end sum_of_two_squares_equivalence_l759_75988


namespace original_list_size_l759_75908

theorem original_list_size (n : ℕ) (m : ℚ) : 
  (m + 3) * (n + 1) = m * n + 20 →
  (m + 1) * (n + 2) = m * n + 21 →
  n = 3 :=
by sorry

end original_list_size_l759_75908


namespace fifth_largest_divisor_of_1000800000_l759_75996

def n : ℕ := 1000800000

def is_fifth_largest_divisor (d : ℕ) : Prop :=
  d ∣ n ∧ (∃ (a b c e : ℕ), a > b ∧ b > c ∧ c > d ∧ d > e ∧
    a ∣ n ∧ b ∣ n ∧ c ∣ n ∧ e ∣ n ∧
    ∀ (x : ℕ), x ∣ n → x ≤ e ∨ x = d ∨ x = c ∨ x = b ∨ x = a ∨ x = n)

theorem fifth_largest_divisor_of_1000800000 :
  is_fifth_largest_divisor 62550000 :=
sorry

end fifth_largest_divisor_of_1000800000_l759_75996

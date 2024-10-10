import Mathlib

namespace cost_calculation_l228_22821

/-- Given the cost relationships between mangos, rice, and flour, 
    prove the total cost of a specific quantity of each. -/
theorem cost_calculation 
  (mango_rice_relation : ∀ (mango_cost rice_cost : ℝ), 10 * mango_cost = 24 * rice_cost)
  (flour_rice_relation : ∀ (flour_cost rice_cost : ℝ), 6 * flour_cost = 2 * rice_cost)
  (flour_cost : ℝ) (h_flour_cost : flour_cost = 23)
  : ∃ (mango_cost rice_cost : ℝ),
    4 * mango_cost + 3 * rice_cost + 5 * flour_cost = 984.4 :=
by sorry

end cost_calculation_l228_22821


namespace triangle_placement_theorem_l228_22829

-- Define the types for points and angles
def Point : Type := ℝ × ℝ
def Angle : Type := ℝ

-- Define a triangle as a triple of points
structure Triangle :=
  (E F G : Point)

-- Define the property that a point lies on an arm of an angle
def lies_on_arm (P : Point) (A : Point) (angle : Angle) : Prop := sorry

-- Define the property that an angle between three points equals a given angle
def angle_equals (A B C : Point) (angle : Angle) : Prop := sorry

theorem triangle_placement_theorem 
  (T : Triangle) (angle_ABC angle_CBD : Angle) : 
  ∃ (B : Point), 
    (lies_on_arm T.E B angle_ABC) ∧ 
    (lies_on_arm T.F B angle_ABC) ∧ 
    (lies_on_arm T.G B angle_CBD) ∧
    (angle_equals T.E B T.F angle_ABC) ∧
    (angle_equals T.F B T.G angle_CBD) := by
  sorry

end triangle_placement_theorem_l228_22829


namespace apartment_cost_increase_is_40_percent_l228_22833

/-- The percentage increase in cost of a new apartment compared to an old apartment. -/
def apartment_cost_increase (old_cost monthly_savings : ℚ) : ℚ := by
  -- Define John's share of the new apartment cost
  let johns_share := old_cost - monthly_savings
  -- Calculate the total new apartment cost (3 times John's share)
  let new_cost := 3 * johns_share
  -- Calculate the percentage increase
  exact ((new_cost - old_cost) / old_cost) * 100

/-- Theorem stating the percentage increase in apartment cost -/
theorem apartment_cost_increase_is_40_percent : 
  apartment_cost_increase 1200 (7680 / 12) = 40 := by
  sorry


end apartment_cost_increase_is_40_percent_l228_22833


namespace fruit_basket_combinations_l228_22894

/-- The number of possible fruit baskets given the constraints -/
def fruitBaskets (totalApples totalOranges : ℕ) (minApples minOranges : ℕ) : ℕ :=
  (totalApples - minApples + 1) * (totalOranges - minOranges + 1)

/-- Theorem stating the number of possible fruit baskets under given conditions -/
theorem fruit_basket_combinations :
  fruitBaskets 6 12 1 2 = 66 := by
  sorry

#eval fruitBaskets 6 12 1 2

end fruit_basket_combinations_l228_22894


namespace abs_equation_solution_difference_l228_22896

theorem abs_equation_solution_difference : ∃ x₁ x₂ : ℝ, 
  (|x₁ - 3| = 15 ∧ |x₂ - 3| = 15) ∧ 
  x₁ ≠ x₂ ∧
  |x₁ - x₂| = 30 :=
by sorry

end abs_equation_solution_difference_l228_22896


namespace locus_is_circle_l228_22864

/-- The locus of points satisfying the given equation is a circle -/
theorem locus_is_circle (x y : ℝ) : 
  (10 * Real.sqrt ((x - 1)^2 + (y - 2)^2) = |3*x - 4*y|) → 
  ∃ (center : ℝ × ℝ) (radius : ℝ), 
    (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
by sorry

end locus_is_circle_l228_22864


namespace max_y_value_l228_22888

theorem max_y_value (x y : ℤ) (h : x * y + 3 * x + 2 * y = -6) : 
  ∃ (max_y : ℤ), (∀ (z : ℤ), ∃ (w : ℤ), w * z + 3 * w + 2 * z = -6 → z ≤ max_y) ∧ max_y = 3 := by
  sorry

end max_y_value_l228_22888


namespace max_correct_answers_15_l228_22871

/-- Represents an exam with a fixed number of questions and scoring system. -/
structure Exam where
  total_questions : ℕ
  correct_score : ℤ
  incorrect_score : ℤ

/-- Represents a student's exam result. -/
structure ExamResult where
  exam : Exam
  correct_answers : ℕ
  incorrect_answers : ℕ
  blank_answers : ℕ
  total_score : ℤ

/-- Calculates the total score for an exam result. -/
def calculate_score (result : ExamResult) : ℤ :=
  result.correct_answers * result.exam.correct_score +
  result.incorrect_answers * result.exam.incorrect_score

/-- Verifies if an exam result is valid. -/
def is_valid_result (result : ExamResult) : Prop :=
  result.correct_answers + result.incorrect_answers + result.blank_answers = result.exam.total_questions ∧
  calculate_score result = result.total_score

/-- Theorem: The maximum number of correct answers for John's exam is 15. -/
theorem max_correct_answers_15 (john_exam : Exam) (john_result : ExamResult) :
  john_exam.total_questions = 25 ∧
  john_exam.correct_score = 6 ∧
  john_exam.incorrect_score = -3 ∧
  john_result.exam = john_exam ∧
  john_result.total_score = 60 ∧
  is_valid_result john_result →
  john_result.correct_answers ≤ 15 ∧
  ∃ (valid_result : ExamResult),
    valid_result.exam = john_exam ∧
    valid_result.total_score = 60 ∧
    is_valid_result valid_result ∧
    valid_result.correct_answers = 15 :=
by sorry

end max_correct_answers_15_l228_22871


namespace min_selections_for_multiple_of_five_l228_22873

theorem min_selections_for_multiple_of_five (n : ℕ) (h : n = 30) : 
  (∀ S : Finset ℕ, S ⊆ Finset.range n → S.card ≥ 25 → ∃ x ∈ S, x % 5 = 0) ∧
  (∃ S : Finset ℕ, S ⊆ Finset.range n ∧ S.card = 24 ∧ ∀ x ∈ S, x % 5 ≠ 0) :=
sorry

end min_selections_for_multiple_of_five_l228_22873


namespace william_marbles_left_l228_22803

/-- Given that William initially has 10 marbles and shares 3 marbles with Theresa,
    prove that William will have 7 marbles left. -/
theorem william_marbles_left (initial_marbles : ℕ) (shared_marbles : ℕ) 
  (h1 : initial_marbles = 10) (h2 : shared_marbles = 3) :
  initial_marbles - shared_marbles = 7 := by
  sorry

end william_marbles_left_l228_22803


namespace doll_difference_l228_22812

/-- The number of dolls Lindsay has with blonde hair -/
def blonde_dolls : ℕ := 4

/-- The number of dolls Lindsay has with brown hair -/
def brown_dolls : ℕ := 4 * blonde_dolls

/-- The number of dolls Lindsay has with black hair -/
def black_dolls : ℕ := brown_dolls - 2

/-- The theorem stating the difference between the combined number of black and brown-haired dolls
    and the number of blonde-haired dolls -/
theorem doll_difference : brown_dolls + black_dolls - blonde_dolls = 26 := by
  sorry

end doll_difference_l228_22812


namespace max_value_of_f_l228_22872

noncomputable def f (x : ℝ) : ℝ := x / Real.exp x

theorem max_value_of_f :
  ∃ (c : ℝ), c ∈ Set.Icc 0 2 ∧
  (∀ x, x ∈ Set.Icc 0 2 → f x ≤ f c) ∧
  f c = (1 : ℝ) / Real.exp 1 :=
sorry

end max_value_of_f_l228_22872


namespace even_product_probability_l228_22809

def set_A_odd : ℕ := 7
def set_A_even : ℕ := 9
def set_B_odd : ℕ := 5
def set_B_even : ℕ := 4

def total_A : ℕ := set_A_odd + set_A_even
def total_B : ℕ := set_B_odd + set_B_even

def prob_even_product : ℚ := 109 / 144

theorem even_product_probability :
  (set_A_even : ℚ) / total_A * (set_B_even : ℚ) / total_B +
  (set_A_odd : ℚ) / total_A * (set_B_even : ℚ) / total_B +
  (set_A_even : ℚ) / total_A * (set_B_odd : ℚ) / total_B = prob_even_product :=
by sorry

end even_product_probability_l228_22809


namespace box_width_l228_22898

/-- The width of a rectangular box given specific conditions -/
theorem box_width (num_cubes : ℕ) (cube_volume length height : ℝ) :
  num_cubes = 24 →
  cube_volume = 27 →
  length = 8 →
  height = 12 →
  (num_cubes : ℝ) * cube_volume / (length * height) = 6.75 := by
  sorry

end box_width_l228_22898


namespace remainder_2021_div_102_l228_22830

theorem remainder_2021_div_102 : 2021 % 102 = 83 := by
  sorry

end remainder_2021_div_102_l228_22830


namespace rectangular_prism_diagonals_l228_22838

/-- A rectangular prism with three distinct dimensions -/
structure RectangularPrism where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c
  distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a

/-- The number of face diagonals in a rectangular prism -/
def face_diagonals (prism : RectangularPrism) : ℕ := 12

/-- The number of space diagonals in a rectangular prism -/
def space_diagonals (prism : RectangularPrism) : ℕ := 4

/-- The total number of diagonals in a rectangular prism -/
def total_diagonals (prism : RectangularPrism) : ℕ :=
  face_diagonals prism + space_diagonals prism

/-- Theorem: A rectangular prism with three distinct dimensions has 16 total diagonals -/
theorem rectangular_prism_diagonals (prism : RectangularPrism) :
  total_diagonals prism = 16 := by
  sorry

end rectangular_prism_diagonals_l228_22838


namespace median_mode_difference_l228_22813

/-- Represents the monthly income data of employees --/
structure IncomeData where
  income : List Nat
  frequency : List Nat
  total_employees : Nat

/-- Calculates the mode of the income data --/
def mode (data : IncomeData) : Nat :=
  sorry

/-- Calculates the median of the income data --/
def median (data : IncomeData) : Nat :=
  sorry

/-- The income data for the company --/
def company_data : IncomeData := {
  income := [45000, 18000, 10000, 5500, 5000, 3400, 3000, 2500],
  frequency := [1, 1, 1, 3, 6, 1, 11, 1],
  total_employees := 25
}

/-- Theorem stating that the median is 400 yuan greater than the mode --/
theorem median_mode_difference (data : IncomeData) : 
  median data = mode data + 400 :=
sorry

end median_mode_difference_l228_22813


namespace circle_passes_through_points_l228_22827

/-- A circle is defined by the equation x^2 + y^2 + Dx + Ey + F = 0 -/
def Circle (D E F : ℝ) := fun (x y : ℝ) => x^2 + y^2 + D*x + E*y + F = 0

/-- The specific circle we're interested in -/
def SpecificCircle := Circle (-4) (-6) 0

theorem circle_passes_through_points :
  (SpecificCircle 0 0) ∧ 
  (SpecificCircle 4 0) ∧ 
  (SpecificCircle (-1) 1) := by
  sorry

end circle_passes_through_points_l228_22827


namespace hyperbola_parabola_intersection_l228_22865

/-- Given a hyperbola and a parabola, prove that if the left focus of the hyperbola
    lies on the directrix of the parabola, then p = 4 -/
theorem hyperbola_parabola_intersection (p : ℝ) (hp : p > 0) : 
  (∃ x y : ℝ, x^2 / 3 - 16 * y^2 / p^2 = 1) →  -- hyperbola equation
  (∃ x y : ℝ, y^2 = 2 * p * x) →              -- parabola equation
  (- Real.sqrt (3 + p^2 / 16) = - p / 2) →    -- left focus on directrix condition
  p = 4 := by
sorry

end hyperbola_parabola_intersection_l228_22865


namespace booklet_sheets_l228_22800

/-- Given a booklet created from folded A4 sheets, prove the number of original sheets. -/
theorem booklet_sheets (n : ℕ) (h : 2 * n + 2 = 74) : n / 4 = 9 := by
  sorry

#check booklet_sheets

end booklet_sheets_l228_22800


namespace largest_when_first_digit_changed_l228_22817

def original_number : ℚ := 0.123456

def change_digit (n : ℕ) (d : ℕ) : ℚ :=
  if n = 1 then 0.8 + (original_number - 0.1)
  else if n = 2 then 0.1 + 0.08 + (original_number - 0.12)
  else if n = 3 then 0.12 + 0.008 + (original_number - 0.123)
  else if n = 4 then 0.123 + 0.0008 + (original_number - 0.1234)
  else if n = 5 then 0.1234 + 0.00008 + (original_number - 0.12345)
  else 0.12345 + 0.000008 + (original_number - 0.123456)

theorem largest_when_first_digit_changed :
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 6 → change_digit 1 8 ≥ change_digit n 8 :=
by sorry

end largest_when_first_digit_changed_l228_22817


namespace square_equality_solution_l228_22825

theorem square_equality_solution (x : ℝ) : (2012 + x)^2 = x^2 ↔ x = -1006 := by
  sorry

end square_equality_solution_l228_22825


namespace max_value_product_sum_l228_22853

theorem max_value_product_sum (A M C : ℕ) (h : A + M + C = 15) :
  (∀ a m c : ℕ, a + m + c = 15 → A * M * C + A * M + M * C + C * A ≥ a * m * c + a * m + m * c + c * a) →
  A * M * C + A * M + M * C + C * A = 200 :=
by sorry

end max_value_product_sum_l228_22853


namespace right_triangle_third_side_product_l228_22883

theorem right_triangle_third_side_product (a b : ℝ) (ha : a = 6) (hb : b = 8) :
  let c := Real.sqrt (a^2 + b^2)
  let d := Real.sqrt (max a b ^ 2 - min a b ^ 2)
  c * d = 20 * Real.sqrt 7 := by sorry

end right_triangle_third_side_product_l228_22883


namespace minAreaLineEquation_l228_22828

/-- A line passing through a point (x₀, y₀) -/
structure Line where
  slope : ℝ
  x₀ : ℝ
  y₀ : ℝ

/-- The area of the triangle formed by a line and the coordinate axes -/
def triangleArea (l : Line) : ℝ :=
  sorry

/-- The line passing through (1, 2) that minimizes the triangle area -/
noncomputable def minAreaLine : Line :=
  sorry

theorem minAreaLineEquation :
  let l := minAreaLine
  l.x₀ = 1 ∧ l.y₀ = 2 ∧
  ∀ (m : Line), m.x₀ = 1 ∧ m.y₀ = 2 → triangleArea l ≤ triangleArea m ∧
  2 * l.x₀ + l.y₀ - 4 = 0 :=
sorry

end minAreaLineEquation_l228_22828


namespace cost_price_of_article_l228_22850

/-- 
Proves that the cost price of an article is 44, given that the profit obtained 
by selling it for 66 is the same as the loss obtained by selling it for 22.
-/
theorem cost_price_of_article : ∃ (x : ℝ), 
  (66 - x = x - 22) → x = 44 := by
  sorry

end cost_price_of_article_l228_22850


namespace f_is_quadratic_l228_22868

/-- Definition of a quadratic function -/
def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function f(x) = 5x^2 - 1 -/
def f (x : ℝ) : ℝ := 5 * x^2 - 1

/-- Theorem: f is a quadratic function -/
theorem f_is_quadratic : is_quadratic f := by
  sorry

end f_is_quadratic_l228_22868


namespace supermarket_profit_and_discount_l228_22860

-- Define the goods
structure Good where
  cost : ℝ
  price : ℝ

-- Define the problem parameters
def good_A : Good := { cost := 22, price := 29 }
def good_B : Good := { cost := 30, price := 40 }

-- Define the theorem
theorem supermarket_profit_and_discount 
  (total_cost : ℝ) 
  (num_A : ℕ) 
  (num_B : ℕ) 
  (second_profit_increase : ℝ) :
  total_cost = 6000 ∧ 
  num_B = (num_A / 2 + 15 : ℕ) ∧
  num_A * good_A.cost + num_B * good_B.cost = total_cost →
  (num_A * (good_A.price - good_A.cost) + num_B * (good_B.price - good_B.cost) = 1950) ∧
  ∃ discount_rate : ℝ,
    discount_rate ≥ 0 ∧ 
    discount_rate ≤ 1 ∧
    num_A * (good_A.price - good_A.cost) + 3 * num_B * ((1 - discount_rate) * good_B.price - good_B.cost) = 
    1950 + second_profit_increase ∧
    discount_rate = 0.085 := by
  sorry

-- Note: The proof is omitted as per the instructions

end supermarket_profit_and_discount_l228_22860


namespace average_marks_proof_l228_22806

/-- Given the marks in three subjects, prove that the average is 75 -/
theorem average_marks_proof (physics chemistry mathematics : ℝ) 
  (h1 : (physics + mathematics) / 2 = 90)
  (h2 : (physics + chemistry) / 2 = 70)
  (h3 : physics = 95) :
  (physics + chemistry + mathematics) / 3 = 75 := by
  sorry

end average_marks_proof_l228_22806


namespace vasyas_number_l228_22816

theorem vasyas_number : ∃! n : ℕ, 
  10 ≤ n ∧ n < 100 ∧ 
  1008 + 10 * n = 28 * n :=
by
  sorry

end vasyas_number_l228_22816


namespace parallel_vectors_x_value_l228_22842

/-- Two vectors are parallel if the ratio of their corresponding components is equal -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 / b.1 = a.2 / b.2

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (4, 2)
  let b : ℝ × ℝ := (x, 3)
  parallel a b → x = 6 := by
  sorry

end parallel_vectors_x_value_l228_22842


namespace square_sum_from_conditions_l228_22835

theorem square_sum_from_conditions (x y : ℝ) 
  (h1 : x + 2 * y = 6) 
  (h2 : x * y = -6) : 
  x^2 + 4 * y^2 = 60 := by
sorry

end square_sum_from_conditions_l228_22835


namespace probability_of_selecting_A_and_B_l228_22847

def total_plants : ℕ := 5
def selected_plants : ℕ := 3

theorem probability_of_selecting_A_and_B : 
  (Nat.choose total_plants selected_plants) > 0 → 
  (Nat.choose (total_plants - 2) (selected_plants - 2)) > 0 →
  (Nat.choose (total_plants - 2) (selected_plants - 2) : ℚ) / 
  (Nat.choose total_plants selected_plants : ℚ) = 3 / 10 := by
sorry

end probability_of_selecting_A_and_B_l228_22847


namespace line_y_coordinate_l228_22837

/-- A line passing through points (-12, y1) and (x2, 3) with x-intercept at (4, 0) has y1 = 0 -/
theorem line_y_coordinate (y1 x2 : ℝ) : 
  (∃ (m : ℝ), (3 - y1) = m * (x2 - (-12)) ∧ 
               0 - y1 = m * (4 - (-12)) ∧
               3 = m * (x2 - 4)) →
  y1 = 0 := by
sorry

end line_y_coordinate_l228_22837


namespace value_of_a_l228_22820

/-- A sequence where each term is the sum of the two terms to its left -/
def Sequence : Type := ℤ → ℤ

/-- Property that each term is the sum of the two terms to its left -/
def is_sum_of_previous_two (s : Sequence) : Prop :=
  ∀ n : ℤ, s (n + 2) = s (n + 1) + s n

/-- The specific sequence we're interested in -/
def our_sequence : Sequence := sorry

/-- The properties of our specific sequence -/
axiom our_sequence_property : is_sum_of_previous_two our_sequence
axiom our_sequence_known_values :
  ∃ k : ℤ,
    our_sequence (k + 3) = 0 ∧
    our_sequence (k + 4) = 1 ∧
    our_sequence (k + 5) = 1 ∧
    our_sequence (k + 6) = 2 ∧
    our_sequence (k + 7) = 3 ∧
    our_sequence (k + 8) = 5 ∧
    our_sequence (k + 9) = 8

/-- The theorem to prove -/
theorem value_of_a :
  ∃ k : ℤ, our_sequence k = -3 ∧
    our_sequence (k + 3) = 0 ∧
    our_sequence (k + 4) = 1 :=
sorry

end value_of_a_l228_22820


namespace polynomial_sum_equality_l228_22811

-- Define the polynomials
def p (x : ℝ) : ℝ := -4 * x^2 + 2 * x - 5
def q (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 7
def r (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2
def s (x : ℝ) : ℝ := 2 * x^2 - 3 * x - 1

-- State the theorem
theorem polynomial_sum_equality :
  ∀ x : ℝ, p x + q x + r x + s x = -2 * x^2 + 9 * x - 11 :=
by
  sorry

end polynomial_sum_equality_l228_22811


namespace propositions_proof_l228_22852

theorem propositions_proof :
  (∀ a b : ℝ, a > b ∧ (1 / a) > (1 / b) → a * b < 0) ∧
  (∀ a b : ℝ, a < b ∧ b < 0 → ¬(a^2 < a * b ∧ a * b < b^2)) ∧
  (∀ a b c : ℝ, c > a ∧ a > b ∧ b > 0 → ¬(a / (c - a) < b / (c - b))) ∧
  (∀ a b c : ℝ, a > b ∧ b > c ∧ c > 0 → a / b > (a + c) / (b + c)) :=
by sorry

end propositions_proof_l228_22852


namespace euler_totient_multiple_l228_22851

theorem euler_totient_multiple (m n : ℕ+) : ∃ a : ℕ+, ∀ i : ℕ, i ≤ n → (m : ℕ) ∣ Nat.totient (a + i) := by
  sorry

end euler_totient_multiple_l228_22851


namespace ed_has_27_pets_l228_22886

/-- Represents the number of pets Ed has -/
structure Pets where
  dogs : ℕ
  cats : ℕ
  fish : ℕ
  birds : ℕ
  turtles : ℕ

/-- The conditions given in the problem -/
def petConditions (p : Pets) : Prop :=
  p.dogs = 2 ∧
  p.cats = 3 ∧
  p.fish = 3 * p.birds ∧
  p.fish = 2 * (p.dogs + p.cats) ∧
  p.turtles = p.birds / 2

/-- The total number of pets -/
def totalPets (p : Pets) : ℕ :=
  p.dogs + p.cats + p.fish + p.birds + p.turtles

/-- Theorem stating that given the conditions, Ed has 27 pets in total -/
theorem ed_has_27_pets :
  ∃ p : Pets, petConditions p ∧ totalPets p = 27 := by
  sorry


end ed_has_27_pets_l228_22886


namespace square_difference_65_35_l228_22882

theorem square_difference_65_35 : 65^2 - 35^2 = 3000 := by
  sorry

end square_difference_65_35_l228_22882


namespace union_of_M_and_N_l228_22866

def M : Set ℝ := {x | x^2 = x}
def N : Set ℝ := {x | Real.log x > 0}

theorem union_of_M_and_N : M ∪ N = {x | x = 0 ∨ x ≥ 1} := by sorry

end union_of_M_and_N_l228_22866


namespace constant_dot_product_l228_22839

open Real

/-- Definition of the ellipse C -/
def ellipse (x y : ℝ) : Prop := x^2 / 27 + y^2 / 18 = 1

/-- Right focus of the ellipse -/
def F : ℝ × ℝ := (3, 0)

/-- The fixed point P -/
def P : ℝ × ℝ := (4, 0)

/-- A line passing through F -/
def line_through_F (k : ℝ) (x : ℝ) : ℝ := k * (x - F.1)

/-- Intersection points of the line with the ellipse -/
def intersection_points (k : ℝ) : Set (ℝ × ℝ) :=
  {p | ellipse p.1 p.2 ∧ p.2 = line_through_F k p.1}

/-- Dot product of vectors PA and PB -/
def dot_product (A B : ℝ × ℝ) : ℝ :=
  (A.1 - P.1) * (B.1 - P.1) + (A.2 - P.2) * (B.2 - P.2)

/-- Theorem: The dot product PA · PB is constant for any line through F -/
theorem constant_dot_product :
  ∃ (c : ℝ), ∀ (k : ℝ) (A B : ℝ × ℝ),
    A ∈ intersection_points k → B ∈ intersection_points k →
    A ≠ B → dot_product A B = c :=
sorry

end constant_dot_product_l228_22839


namespace twelfth_term_of_sequence_l228_22879

/-- The nth term of an arithmetic sequence -/
def arithmetic_sequence (a₁ d : ℚ) (n : ℕ) : ℚ :=
  a₁ + (n - 1 : ℚ) * d

/-- Theorem: The 12th term of the arithmetic sequence with a₁ = 1/4 and d = 1/2 is 23/4 -/
theorem twelfth_term_of_sequence :
  arithmetic_sequence (1/4) (1/2) 12 = 23/4 := by
  sorry

end twelfth_term_of_sequence_l228_22879


namespace infinitely_many_polynomials_l228_22885

/-- A polynomial with real coefficients -/
def RealPolynomial := ℝ → ℝ

/-- The condition that x, y, and z must satisfy -/
def SphereCondition (x y z : ℝ) : Prop :=
  x^2 + y^2 + z^2 + 2*x*y*z = 1

/-- The condition that the polynomial P must satisfy -/
def PolynomialCondition (P : RealPolynomial) : Prop :=
  ∀ x y z : ℝ, SphereCondition x y z →
    P x^2 + P y^2 + P z^2 + 2*(P x)*(P y)*(P z) = 1

/-- The main theorem stating that there are infinitely many polynomials satisfying the condition -/
theorem infinitely_many_polynomials :
  ∃ (S : Set RealPolynomial), (Set.Infinite S) ∧ (∀ P ∈ S, PolynomialCondition P) :=
sorry

end infinitely_many_polynomials_l228_22885


namespace intersection_implies_z_equals_i_l228_22897

theorem intersection_implies_z_equals_i : 
  let i : ℂ := Complex.I
  let P : Set ℂ := {1, -1}
  let Q : Set ℂ := {i, i^2}
  ∀ z : ℂ, (P ∩ Q = {z * i}) → z = i := by
sorry

end intersection_implies_z_equals_i_l228_22897


namespace paper_tray_height_l228_22815

theorem paper_tray_height (side_length : ℝ) (cut_distance : ℝ) (cut_angle : ℝ) :
  side_length = 120 →
  cut_distance = 6 →
  cut_angle = 45 →
  let tray_height := cut_distance
  tray_height = 6 := by sorry

end paper_tray_height_l228_22815


namespace system_solution_l228_22802

theorem system_solution (x y : Real) : 
  (Real.sin x)^2 + (Real.cos y)^2 = y^4 ∧ 
  (Real.sin y)^2 + (Real.cos x)^2 = x^2 → 
  (x = 1 ∨ x = -1) ∧ (y = 1 ∨ y = -1) := by
  sorry

end system_solution_l228_22802


namespace power_of_three_mod_nineteen_l228_22876

theorem power_of_three_mod_nineteen : 3^17 % 19 = 13 := by
  sorry

end power_of_three_mod_nineteen_l228_22876


namespace number_problem_l228_22877

theorem number_problem (n : ℚ) : (1/2 : ℚ) * (3/5 : ℚ) * n = 36 → n = 120 := by
  sorry

end number_problem_l228_22877


namespace max_rooks_per_color_exists_sixteen_rooks_config_max_rooks_is_sixteen_l228_22858

/-- Represents a chessboard configuration with white and black rooks -/
structure ChessboardConfig where
  board_size : Nat
  white_rooks : Nat
  black_rooks : Nat
  non_threatening : Bool

/-- Defines a valid chessboard configuration -/
def is_valid_config (c : ChessboardConfig) : Prop :=
  c.board_size = 8 ∧ 
  c.white_rooks = c.black_rooks ∧ 
  c.non_threatening = true

/-- Theorem stating the maximum number of rooks for each color -/
theorem max_rooks_per_color (c : ChessboardConfig) : 
  is_valid_config c → c.white_rooks ≤ 16 := by
  sorry

/-- Theorem proving the existence of a configuration with 16 rooks per color -/
theorem exists_sixteen_rooks_config : 
  ∃ c : ChessboardConfig, is_valid_config c ∧ c.white_rooks = 16 := by
  sorry

/-- Main theorem proving 16 is the maximum number of rooks per color -/
theorem max_rooks_is_sixteen : 
  ∀ c : ChessboardConfig, is_valid_config c → 
    c.white_rooks ≤ 16 ∧ (∃ c' : ChessboardConfig, is_valid_config c' ∧ c'.white_rooks = 16) := by
  sorry

end max_rooks_per_color_exists_sixteen_rooks_config_max_rooks_is_sixteen_l228_22858


namespace circle_passes_through_fixed_point_l228_22884

/-- A point on a parabola -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ
  parabola_eq : (y - 3)^2 = 8 * (x - 2)

/-- A circle tangent to the y-axis -/
structure TangentCircle where
  center : ParabolaPoint
  radius : ℝ
  tangent_to_y_axis : radius = center.x

theorem circle_passes_through_fixed_point (P : ParabolaPoint) (C : TangentCircle) 
  (h : C.center = P) : 
  (C.center.x - 4)^2 + (C.center.y - 3)^2 = C.radius^2 := by sorry

end circle_passes_through_fixed_point_l228_22884


namespace second_candidate_votes_l228_22819

theorem second_candidate_votes (total_votes : ℕ) (first_candidate_percentage : ℚ) : 
  total_votes = 800 → 
  first_candidate_percentage = 70 / 100 →
  (total_votes : ℚ) * (1 - first_candidate_percentage) = 240 := by
  sorry

#check second_candidate_votes

end second_candidate_votes_l228_22819


namespace no_integer_root_trinomials_l228_22861

theorem no_integer_root_trinomials : ¬∃ (a b c : ℤ),
  (∃ (x₁ x₂ : ℤ), a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0 ∧ x₁ ≠ x₂) ∧
  (∃ (y₁ y₂ : ℤ), (a + 1) * y₁^2 + (b + 1) * y₁ + (c + 1) = 0 ∧ (a + 1) * y₂^2 + (b + 1) * y₂ + (c + 1) = 0 ∧ y₁ ≠ y₂) :=
by sorry

end no_integer_root_trinomials_l228_22861


namespace break_time_calculation_l228_22822

-- Define the speeds of A and B
def speed_A : ℝ := 60
def speed_B : ℝ := 40

-- Define the distances from midpoint for the two meeting points
def distance_first_meeting : ℝ := 300
def distance_second_meeting : ℝ := 150

-- Define the total distance between A and B
def total_distance : ℝ := 2 * distance_first_meeting

-- Define the theorem
theorem break_time_calculation :
  ∃ (t : ℝ), (t = 6.25 ∨ t = 18.75) ∧
  ((speed_A * (total_distance / (speed_A + speed_B) - t) = distance_first_meeting + distance_second_meeting) ∨
   (speed_A * (total_distance / (speed_A + speed_B) - t) = total_distance - (distance_first_meeting + distance_second_meeting))) :=
by
  sorry


end break_time_calculation_l228_22822


namespace probability_one_science_one_humanities_l228_22810

def total_courses : ℕ := 5
def science_courses : ℕ := 3
def humanities_courses : ℕ := 2
def courses_chosen : ℕ := 2

theorem probability_one_science_one_humanities :
  (Nat.choose science_courses 1 * Nat.choose humanities_courses 1) / Nat.choose total_courses courses_chosen = 3 / 5 :=
by sorry

end probability_one_science_one_humanities_l228_22810


namespace red_peaches_count_l228_22801

theorem red_peaches_count (total : ℕ) (green : ℕ) (red : ℕ) : 
  total = 16 → green = 3 → total = red + green → red = 13 := by
  sorry

end red_peaches_count_l228_22801


namespace same_color_probability_l228_22867

/-- The probability of drawing two balls of the same color from a box with 2 red balls and 3 white balls,
    when drawing with replacement. -/
theorem same_color_probability (total_balls : ℕ) (red_balls : ℕ) (white_balls : ℕ) 
  (h1 : total_balls = red_balls + white_balls)
  (h2 : red_balls = 2)
  (h3 : white_balls = 3) :
  (red_balls : ℚ) / total_balls * (red_balls : ℚ) / total_balls + 
  (white_balls : ℚ) / total_balls * (white_balls : ℚ) / total_balls = 13 / 25 :=
sorry

end same_color_probability_l228_22867


namespace loan_duration_proof_l228_22831

/-- Simple interest calculation -/
def simple_interest (principal rate time : ℝ) : ℝ := principal * rate * time

theorem loan_duration_proof (principal rate total_returned : ℝ) 
  (h1 : principal = 5396.103896103896)
  (h2 : rate = 0.06)
  (h3 : total_returned = 8310) :
  ∃ t : ℝ, t = 9 ∧ total_returned = principal + simple_interest principal rate t := by
  sorry

#eval simple_interest 5396.103896103896 0.06 9

end loan_duration_proof_l228_22831


namespace complement_intersection_theorem_l228_22807

def U : Set Nat := {1,2,3,4,5,6,7}
def A : Set Nat := {1,3,5,7}
def B : Set Nat := {1,3,5,6,7}

theorem complement_intersection_theorem :
  (U \ (A ∩ B)) = {2,4,6} := by sorry

end complement_intersection_theorem_l228_22807


namespace irrational_condition_l228_22832

-- Define the set A(x)
def A (x : ℝ) : Set ℤ := {n : ℤ | ∃ m : ℕ, n = ⌊m * x⌋}

-- State the theorem
theorem irrational_condition (α : ℝ) (h_irr : Irrational α) (h_gt_two : α > 2) :
  ∀ β : ℝ, β > 0 → (A α ⊃ A β) → ∃ n : ℤ, β = n * α :=
by sorry

end irrational_condition_l228_22832


namespace line_slope_l228_22805

theorem line_slope (x y : ℝ) : 
  (2 * x + Real.sqrt 3 * y - 1 = 0) → 
  (∃ m : ℝ, m = -(2 * Real.sqrt 3) / 3 ∧ 
   ∀ x₁ x₂ y₁ y₂ : ℝ, 
   x₁ ≠ x₂ → 
   (2 * x₁ + Real.sqrt 3 * y₁ - 1 = 0) → 
   (2 * x₂ + Real.sqrt 3 * y₂ - 1 = 0) → 
   m = (y₂ - y₁) / (x₂ - x₁)) :=
by sorry

end line_slope_l228_22805


namespace complex_equation_solution_l228_22880

theorem complex_equation_solution (a b : ℝ) :
  (1 + 2 * Complex.I) * a + b = 2 * Complex.I → a = 1 ∧ b = -1 := by
  sorry

end complex_equation_solution_l228_22880


namespace fraction_difference_product_l228_22846

theorem fraction_difference_product : 
  let a : ℚ := 1/2
  let b : ℚ := 1/5
  a - b = 3 * (a * b) := by sorry

end fraction_difference_product_l228_22846


namespace hyperbola_eccentricity_l228_22843

/-- The eccentricity of the hyperbola x²/3 - y²/6 = 1 is √3 -/
theorem hyperbola_eccentricity : ∃ e : ℝ, e = Real.sqrt 3 ∧
  ∀ x y : ℝ, x^2 / 3 - y^2 / 6 = 1 → 
  e = Real.sqrt ((x^2 / 3 + y^2 / 6) / (x^2 / 3)) :=
by sorry

end hyperbola_eccentricity_l228_22843


namespace tank_empty_time_l228_22891

/- Define the tank capacity in liters -/
def tank_capacity : ℝ := 5760

/- Define the time it takes for the leak to empty the tank in hours -/
def leak_empty_time : ℝ := 6

/- Define the inlet pipe fill rate in liters per minute -/
def inlet_fill_rate : ℝ := 4

/- Define the time it takes to empty the tank with inlet open in hours -/
def empty_time_with_inlet : ℝ := 8

/- Theorem statement -/
theorem tank_empty_time :
  let leak_rate := tank_capacity / leak_empty_time
  let inlet_rate := inlet_fill_rate * 60
  let net_empty_rate := leak_rate - inlet_rate
  tank_capacity / net_empty_rate = empty_time_with_inlet :=
by sorry

end tank_empty_time_l228_22891


namespace min_value_theorem_l228_22870

theorem min_value_theorem (m n : ℝ) (h1 : m * n > 0) (h2 : 2 * m + n = 1) :
  (1 / m + 2 / n) ≥ 8 :=
by sorry

end min_value_theorem_l228_22870


namespace monomial_2023_matches_pattern_l228_22875

/-- Represents a monomial in the sequence -/
def monomial (n : ℕ) : ℚ × ℕ := ((2 * n + 1) / n, n)

/-- The 2023rd monomial in the sequence -/
def monomial_2023 : ℚ × ℕ := (4047 / 2023, 2023)

/-- Theorem stating that the 2023rd monomial matches the pattern -/
theorem monomial_2023_matches_pattern : monomial 2023 = monomial_2023 := by
  sorry

end monomial_2023_matches_pattern_l228_22875


namespace purple_cars_count_l228_22826

theorem purple_cars_count (purple red green : ℕ) : 
  green = 4 * red →
  red = purple + 6 →
  purple + red + green = 312 →
  purple = 47 := by
sorry

end purple_cars_count_l228_22826


namespace quadratic_inequality_l228_22878

/-- Given non-zero numbers a, b, c such that ax^2 + bx + c > cx for all real x,
    prove that cx^2 - bx + a > cx - b for all real x. -/
theorem quadratic_inequality (a b c : ℝ) (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
  (h_given : ∀ x : ℝ, a * x^2 + b * x + c > c * x) :
  ∀ x : ℝ, c * x^2 - b * x + a > c * x - b :=
by sorry

end quadratic_inequality_l228_22878


namespace f_continuous_at_5_l228_22844

def f (x : ℝ) : ℝ := 2 * x^2 + 8

theorem f_continuous_at_5 :
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - 5| < δ → |f x - f 5| < ε :=
by sorry

end f_continuous_at_5_l228_22844


namespace system_one_solution_l228_22824

theorem system_one_solution (x : ℝ) : 
  (2 * x > 1 - x ∧ x + 2 < 4 * x - 1) ↔ x > 1 := by
sorry

end system_one_solution_l228_22824


namespace odd_times_abs_even_is_odd_l228_22818

-- Define the properties of odd and even functions
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- State the theorem
theorem odd_times_abs_even_is_odd (f g : ℝ → ℝ) 
  (h_f_odd : IsOdd f) (h_g_even : IsEven g) : 
  IsOdd (fun x ↦ f x * |g x|) := by
  sorry

end odd_times_abs_even_is_odd_l228_22818


namespace sum_and_reciprocal_geq_two_l228_22814

theorem sum_and_reciprocal_geq_two (x : ℝ) (h : x > 0) : x + 1 / x ≥ 2 := by
  sorry

end sum_and_reciprocal_geq_two_l228_22814


namespace sum_of_squares_problem_l228_22895

theorem sum_of_squares_problem (x y z : ℝ) 
  (nonneg_x : x ≥ 0) (nonneg_y : y ≥ 0) (nonneg_z : z ≥ 0)
  (sum_of_squares : x^2 + y^2 + z^2 = 50)
  (sum_of_products : x*y + y*z + z*x = 28) :
  x + y + z = Real.sqrt 106 := by
sorry

end sum_of_squares_problem_l228_22895


namespace second_polygon_sides_l228_22881

/-- 
Given two regular polygons with the same perimeter, where the first polygon has 24 sides
and a side length that is three times as long as the second polygon,
prove that the second polygon has 72 sides.
-/
theorem second_polygon_sides (s : ℝ) (n : ℕ) : 
  s > 0 → 
  24 * (3 * s) = n * s → 
  n = 72 :=
by sorry

end second_polygon_sides_l228_22881


namespace max_value_of_S_l228_22889

theorem max_value_of_S (a b : ℝ) :
  3 * a^2 + 5 * abs b = 7 →
  let S := 2 * a^2 - 3 * abs b
  ∀ x y : ℝ, 3 * x^2 + 5 * abs y = 7 → 2 * x^2 - 3 * abs y ≤ |14| / 3 :=
by
  sorry


end max_value_of_S_l228_22889


namespace movie_date_candy_cost_l228_22808

theorem movie_date_candy_cost
  (ticket_cost : ℝ)
  (combo_cost : ℝ)
  (total_spend : ℝ)
  (num_candy : ℕ)
  (h1 : ticket_cost = 20)
  (h2 : combo_cost = 11)
  (h3 : total_spend = 36)
  (h4 : num_candy = 2) :
  (total_spend - ticket_cost - combo_cost) / num_candy = 2.5 := by
  sorry

end movie_date_candy_cost_l228_22808


namespace prob_not_blue_marble_l228_22887

/-- Given odds ratio for an event --/
structure OddsRatio :=
  (for_event : ℕ)
  (against_event : ℕ)

/-- Calculates the probability of an event not occurring given its odds ratio --/
def probability_of_not_occurring (odds : OddsRatio) : ℚ :=
  odds.against_event / (odds.for_event + odds.against_event)

/-- Theorem: The probability of not pulling a blue marble is 6/11 given odds of 5:6 --/
theorem prob_not_blue_marble (odds : OddsRatio) 
  (h : odds = OddsRatio.mk 5 6) : 
  probability_of_not_occurring odds = 6 / 11 := by
  sorry

end prob_not_blue_marble_l228_22887


namespace sin_max_value_l228_22823

theorem sin_max_value (ω : ℝ) (h1 : 0 < ω) (h2 : ω < 1) :
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ π/3 ∧ 
    (∀ y : ℝ, 0 ≤ y ∧ y ≤ π/3 → 2 * Real.sin (ω * y) ≤ 2 * Real.sin (ω * x)) ∧
    2 * Real.sin (ω * x) = Real.sqrt 2) →
  ω = 3/4 := by
sorry

end sin_max_value_l228_22823


namespace square_tile_count_l228_22857

theorem square_tile_count (n : ℕ) (h : n^2 = 81) : 
  n^2 * n^2 - (2*n - 1) = 6544 := by
  sorry

end square_tile_count_l228_22857


namespace red_marbles_count_l228_22845

theorem red_marbles_count (red green yellow different total : ℕ) : 
  green = 3 * red →
  yellow = green / 5 →
  total = 3 * green →
  different = 88 →
  total = red + green + yellow + different →
  red = 12 := by
  sorry

end red_marbles_count_l228_22845


namespace gcd_n4_plus_16_n_plus_3_l228_22874

theorem gcd_n4_plus_16_n_plus_3 (n : ℕ) (h : n > 16) :
  Nat.gcd (n^4 + 16) (n + 3) = 1 := by
  sorry

end gcd_n4_plus_16_n_plus_3_l228_22874


namespace bread_recipe_scaling_l228_22862

/-- Given a recipe that requires 60 mL of water and 80 mL of milk for every 400 mL of flour,
    this theorem proves the amount of water and milk needed for 1200 mL of flour. -/
theorem bread_recipe_scaling (flour : ℝ) (water : ℝ) (milk : ℝ) 
  (h1 : flour = 1200)
  (h2 : water = 60 * (flour / 400))
  (h3 : milk = 80 * (flour / 400)) :
  water = 180 ∧ milk = 240 := by
  sorry

end bread_recipe_scaling_l228_22862


namespace binary_sum_theorem_l228_22863

/-- Converts a binary number (represented as a list of bits) to a natural number. -/
def binary_to_nat (bits : List Bool) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

/-- Adds two binary numbers (represented as lists of bits) and returns the result as a list of bits. -/
def add_binary (a b : List Bool) : List Bool :=
  sorry -- Implementation details omitted

/-- Theorem: The sum of 1101₂, 100₂, 111₂, and 11010₂ is equal to 111001₂ -/
theorem binary_sum_theorem :
  let a := [true, false, true, true]  -- 1101₂
  let b := [false, false, true]       -- 100₂
  let c := [true, true, true]         -- 111₂
  let d := [false, true, false, true, true]  -- 11010₂
  let result := [true, false, false, true, true, true]  -- 111001₂
  add_binary (add_binary (add_binary a b) c) d = result := by
  sorry

#eval binary_to_nat [true, false, false, true, true, true]  -- Should output 57

end binary_sum_theorem_l228_22863


namespace mary_cards_left_l228_22859

/-- The number of baseball cards Mary has left after giving away promised cards -/
def cards_left (initial : ℕ) (promised_fred : ℕ) (promised_jane : ℕ) (promised_tom : ℕ) 
               (bought : ℕ) (received : ℕ) : ℕ :=
  initial + bought + received - (promised_fred + promised_jane + promised_tom)

/-- Theorem stating that Mary will have 6 cards left -/
theorem mary_cards_left : 
  cards_left 18 26 15 36 40 25 = 6 := by sorry

end mary_cards_left_l228_22859


namespace min_distance_to_i_l228_22848

theorem min_distance_to_i (z : ℂ) (h : Complex.abs (z + Complex.I * Real.sqrt 3) + Complex.abs (z - Complex.I * Real.sqrt 3) = 4) :
  ∃ (w : ℂ), Complex.abs (w + Complex.I * Real.sqrt 3) + Complex.abs (w - Complex.I * Real.sqrt 3) = 4 ∧
    Complex.abs (w - Complex.I) ≤ Complex.abs (z - Complex.I) ∧
    Complex.abs (w - Complex.I) = Real.sqrt 6 / 3 :=
by sorry

end min_distance_to_i_l228_22848


namespace town_population_problem_l228_22890

theorem town_population_problem (original_population : ℕ) : 
  (((original_population + 2000) * 85 / 100) : ℕ) = original_population - 50 →
  original_population = 11667 :=
by
  sorry

end town_population_problem_l228_22890


namespace algorithm_description_not_unique_l228_22836

/-- Definition of an algorithm -/
structure Algorithm where
  steps : List String
  solves_problem : Bool

/-- There can be different ways to describe an algorithm -/
theorem algorithm_description_not_unique : ∃ (a1 a2 : Algorithm), a1 ≠ a2 ∧ a1.solves_problem = a2.solves_problem := by
  sorry

end algorithm_description_not_unique_l228_22836


namespace birthday_probability_l228_22854

/-- The probability that 3 boys born in June 1990 have different birthdays -/
def probability_different_birthdays : ℚ :=
  1 * (29 / 30) * (28 / 30)

/-- The number of days in June -/
def days_in_june : ℕ := 30

/-- The number of boys -/
def number_of_boys : ℕ := 3

theorem birthday_probability :
  probability_different_birthdays = 203 / 225 :=
sorry

end birthday_probability_l228_22854


namespace total_shirts_made_l228_22841

-- Define the rate of shirt production
def shirts_per_minute : ℕ := 3

-- Define the working time
def working_time : ℕ := 2

-- Theorem to prove
theorem total_shirts_made : shirts_per_minute * working_time = 6 := by
  sorry

end total_shirts_made_l228_22841


namespace cabin_price_calculation_l228_22892

/-- The price of Alfonso's cabin that Gloria wants to buy -/
def cabin_price : ℕ := sorry

/-- Gloria's initial cash -/
def initial_cash : ℕ := 150

/-- Number of cypress trees Gloria has -/
def cypress_trees : ℕ := 20

/-- Number of pine trees Gloria has -/
def pine_trees : ℕ := 600

/-- Number of maple trees Gloria has -/
def maple_trees : ℕ := 24

/-- Price per cypress tree -/
def cypress_price : ℕ := 100

/-- Price per pine tree -/
def pine_price : ℕ := 200

/-- Price per maple tree -/
def maple_price : ℕ := 300

/-- Amount Gloria wants to have left after buying the cabin -/
def leftover_amount : ℕ := 350

/-- Total amount Gloria can get from selling her trees and her initial cash -/
def total_amount : ℕ :=
  initial_cash +
  cypress_trees * cypress_price +
  pine_trees * pine_price +
  maple_trees * maple_price

theorem cabin_price_calculation :
  cabin_price = total_amount - leftover_amount :=
by sorry

end cabin_price_calculation_l228_22892


namespace range_of_m_for_inequality_l228_22855

theorem range_of_m_for_inequality (m : ℝ) : 
  (∀ x : ℝ, Real.exp (abs (2 * x + 1)) + m ≥ 0) ↔ m ≥ -1 := by
  sorry

end range_of_m_for_inequality_l228_22855


namespace no_upper_bound_for_positive_second_order_ratio_increasing_l228_22893

open Set Real

-- Define the type for functions from (0, +∞) to ℝ
def PosRealFunc := { f : ℝ → ℝ // ∀ x, x > 0 → f x ≠ 0 }

-- Define second-order ratio increasing function
def SecondOrderRatioIncreasing (f : PosRealFunc) : Prop :=
  ∀ x y, 0 < x ∧ x < y → f.val x / x^2 < f.val y / y^2

-- Define the theorem
theorem no_upper_bound_for_positive_second_order_ratio_increasing
  (f : PosRealFunc)
  (h1 : SecondOrderRatioIncreasing f)
  (h2 : ∀ x, x > 0 → f.val x > 0) :
  ¬∃ k, ∀ x, x > 0 → f.val x < k :=
sorry

end no_upper_bound_for_positive_second_order_ratio_increasing_l228_22893


namespace square_root_of_four_l228_22869

theorem square_root_of_four :
  {x : ℝ | x^2 = 4} = {-2, 2} := by sorry

end square_root_of_four_l228_22869


namespace min_value_of_f_l228_22834

/-- The quadratic function f(x) = x^2 + 12x + 5 -/
def f (x : ℝ) : ℝ := x^2 + 12*x + 5

/-- The minimum value of f(x) is -31 -/
theorem min_value_of_f : ∀ x : ℝ, f x ≥ -31 ∧ ∃ y : ℝ, f y = -31 := by
  sorry

end min_value_of_f_l228_22834


namespace correct_change_l228_22899

/-- The change Sandy received after buying a football and a baseball -/
def sandys_change (football_cost baseball_cost payment : ℚ) : ℚ :=
  payment - (football_cost + baseball_cost)

theorem correct_change : sandys_change 9.14 6.81 20 = 4.05 := by
  sorry

end correct_change_l228_22899


namespace triangle_problem_l228_22804

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
def Triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0

theorem triangle_problem (a b c : ℝ) (h1 : Triangle a b c)
    (h2 : b * Real.cos C + c / 2 = a)
    (h3 : b = Real.sqrt 13)
    (h4 : a + c = 4) :
    Real.cos B = 1 / 2 ∧ 
    (1 / 2 : ℝ) * a * c * Real.sin B = Real.sqrt 3 / 4 := by
  sorry

end triangle_problem_l228_22804


namespace midpoint_of_number_line_l228_22856

theorem midpoint_of_number_line (a b : ℝ) (ha : a = -1) (hb : b = 3) :
  (a + b) / 2 = 1 := by sorry

end midpoint_of_number_line_l228_22856


namespace proportional_function_decreasing_l228_22840

/-- A proportional function passing through (2, -4) has a decreasing y as x increases -/
theorem proportional_function_decreasing (k : ℝ) (h1 : k ≠ 0) (h2 : k * 2 = -4) :
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → k * x₁ > k * x₂ := by
  sorry

end proportional_function_decreasing_l228_22840


namespace closest_fraction_l228_22849

def medals_won : ℚ := 25 / 160

def fractions : List ℚ := [1/5, 1/6, 1/7, 1/8, 1/9]

theorem closest_fraction :
  ∃ (f : ℚ), f ∈ fractions ∧ 
  ∀ (g : ℚ), g ∈ fractions → |medals_won - f| ≤ |medals_won - g| ∧
  f = 1/8 :=
sorry

end closest_fraction_l228_22849

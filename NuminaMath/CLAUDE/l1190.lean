import Mathlib

namespace NUMINAMATH_CALUDE_total_area_is_1800_l1190_119050

/-- Calculates the total area of rooms given initial dimensions and modifications -/
def total_area (length width increase_amount : ℕ) : ℕ :=
  let new_length := length + increase_amount
  let new_width := width + increase_amount
  let single_room_area := new_length * new_width
  let four_rooms_area := 4 * single_room_area
  let double_room_area := 2 * single_room_area
  four_rooms_area + double_room_area

/-- Theorem stating that the total area of rooms is 1800 square feet -/
theorem total_area_is_1800 :
  total_area 13 18 2 = 1800 := by sorry

end NUMINAMATH_CALUDE_total_area_is_1800_l1190_119050


namespace NUMINAMATH_CALUDE_initial_pens_l1190_119097

def double_weekly (initial : ℕ) : ℕ → ℕ
  | 0 => initial
  | n + 1 => 2 * double_weekly initial n

theorem initial_pens (initial : ℕ) :
  double_weekly initial 4 = 32 → initial = 2 := by
  sorry

end NUMINAMATH_CALUDE_initial_pens_l1190_119097


namespace NUMINAMATH_CALUDE_vector_problem_l1190_119073

def a : ℝ × ℝ := (1, 3)
def b (y : ℝ) : ℝ × ℝ := (2, y)

theorem vector_problem (y : ℝ) :
  (∀ y, (a.1 * (b y).1 + a.2 * (b y).2 = 5) → y = 1) ∧
  (∀ y, ((a.1 + (b y).1)^2 + (a.2 + (b y).2)^2 = (a.1 - (b y).1)^2 + (a.2 - (b y).2)^2) → y = -2/3) :=
by sorry

end NUMINAMATH_CALUDE_vector_problem_l1190_119073


namespace NUMINAMATH_CALUDE_exists_permutation_with_difference_l1190_119066

theorem exists_permutation_with_difference (x y z w : ℝ) 
  (sum_eq : x + y + z + w = 13)
  (sum_squares_eq : x^2 + y^2 + z^2 + w^2 = 43) :
  ∃ (a b c d : ℝ), ({a, b, c, d} : Finset ℝ) = {x, y, z, w} ∧ a * b - c * d ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_exists_permutation_with_difference_l1190_119066


namespace NUMINAMATH_CALUDE_oranges_taken_l1190_119049

theorem oranges_taken (initial : ℕ) (remaining : ℕ) (taken : ℕ) 
  (h1 : initial = 60)
  (h2 : remaining = 25)
  (h3 : initial = remaining + taken) : 
  taken = 35 := by
  sorry

end NUMINAMATH_CALUDE_oranges_taken_l1190_119049


namespace NUMINAMATH_CALUDE_smallest_n_for_quadratic_inequality_six_satisfies_inequality_smallest_n_is_six_l1190_119015

theorem smallest_n_for_quadratic_inequality :
  ∀ n : ℤ, n^2 - 9*n + 20 > 0 → n ≥ 6 :=
by
  sorry

theorem six_satisfies_inequality : (6 : ℤ)^2 - 9*(6 : ℤ) + 20 > 0 :=
by
  sorry

theorem smallest_n_is_six :
  ∃ n : ℤ, (∀ m : ℤ, m^2 - 9*m + 20 > 0 → m ≥ n) ∧ n^2 - 9*n + 20 > 0 ∧ n = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_quadratic_inequality_six_satisfies_inequality_smallest_n_is_six_l1190_119015


namespace NUMINAMATH_CALUDE_inverse_variation_y_sqrt_z_l1190_119088

theorem inverse_variation_y_sqrt_z (y z : ℝ) (k : ℝ) (h1 : y^2 * Real.sqrt z = k) 
  (h2 : 3^2 * Real.sqrt 4 = k) (h3 : y = 6) : z = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_y_sqrt_z_l1190_119088


namespace NUMINAMATH_CALUDE_sams_weight_l1190_119032

/-- Given the weights of Tyler, Sam, and Peter, prove Sam's weight -/
theorem sams_weight (tyler sam peter : ℝ) : 
  tyler = sam + 25 →
  peter = tyler / 2 →
  peter = 65 →
  sam = 105 := by
  sorry

end NUMINAMATH_CALUDE_sams_weight_l1190_119032


namespace NUMINAMATH_CALUDE_camel_cost_l1190_119086

/-- The cost of animals in a zoo -/
structure AnimalCosts where
  camel : ℕ
  horse : ℕ
  ox : ℕ
  elephant : ℕ
  giraffe : ℕ
  zebra : ℕ

/-- The conditions given in the problem -/
def zoo_conditions (c : AnimalCosts) : Prop :=
  10 * c.camel = 24 * c.horse ∧
  16 * c.horse = 4 * c.ox ∧
  6 * c.ox = 4 * c.elephant ∧
  3 * c.elephant = 15 * c.giraffe ∧
  8 * c.giraffe = 20 * c.zebra ∧
  12 * c.elephant = 180000

theorem camel_cost (c : AnimalCosts) :
  zoo_conditions c → c.camel = 6000 := by
  sorry

end NUMINAMATH_CALUDE_camel_cost_l1190_119086


namespace NUMINAMATH_CALUDE_expansion_coefficient_l1190_119013

/-- The coefficient of x^3 in the expansion of ((ax-1)^6) -/
def coefficient_x3 (a : ℝ) : ℝ := -20 * a^3

/-- The theorem states that if the coefficient of x^3 in the expansion of ((ax-1)^6) is 20, then a = -1 -/
theorem expansion_coefficient (a : ℝ) : coefficient_x3 a = 20 → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_expansion_coefficient_l1190_119013


namespace NUMINAMATH_CALUDE_penny_species_count_l1190_119012

theorem penny_species_count :
  let sharks : ℕ := 35
  let eels : ℕ := 15
  let whales : ℕ := 5
  sharks + eels + whales = 55 := by
  sorry

end NUMINAMATH_CALUDE_penny_species_count_l1190_119012


namespace NUMINAMATH_CALUDE_integer_roots_of_polynomial_l1190_119071

def polynomial (x : ℤ) : ℤ := x^3 + 3*x^2 - 4*x - 13

theorem integer_roots_of_polynomial :
  {x : ℤ | polynomial x = 0} = {-13, -1, 1, 13} := by sorry

end NUMINAMATH_CALUDE_integer_roots_of_polynomial_l1190_119071


namespace NUMINAMATH_CALUDE_collinear_probability_is_1_182_l1190_119065

/-- Represents a 4x4 square array of dots -/
def SquareArray : Type := Fin 4 × Fin 4

/-- The total number of dots in the array -/
def totalDots : Nat := 16

/-- The number of ways to choose 4 dots from the array -/
def totalChoices : Nat := Nat.choose totalDots 4

/-- The number of sets of 4 collinear dots in the array -/
def collinearSets : Nat := 10

/-- The probability of choosing 4 collinear dots -/
def collinearProbability : Rat := collinearSets / totalChoices

theorem collinear_probability_is_1_182 :
  collinearProbability = 1 / 182 := by sorry

end NUMINAMATH_CALUDE_collinear_probability_is_1_182_l1190_119065


namespace NUMINAMATH_CALUDE_min_value_3a_4b_l1190_119053

theorem min_value_3a_4b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h : (a + b) * (a + 2 * b) + a + b = 9) : 
  ∀ x y : ℝ, 0 < x ∧ 0 < y ∧ (x + y) * (x + 2 * y) + x + y = 9 → 
  3 * x + 4 * y ≥ 6 * Real.sqrt 2 - 1 :=
sorry

end NUMINAMATH_CALUDE_min_value_3a_4b_l1190_119053


namespace NUMINAMATH_CALUDE_joseph_cards_l1190_119064

theorem joseph_cards (initial_cards : ℕ) (cards_to_friend : ℕ) (remaining_fraction : ℚ) : 
  initial_cards = 16 →
  cards_to_friend = 2 →
  remaining_fraction = 1/2 →
  (initial_cards - cards_to_friend - (remaining_fraction * initial_cards)) / initial_cards = 3/8 := by
sorry

end NUMINAMATH_CALUDE_joseph_cards_l1190_119064


namespace NUMINAMATH_CALUDE_triangle_two_solutions_l1190_119007

-- Define the triangle
def Triangle (a b c : ℝ) (A B C : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ A > 0 ∧ B > 0 ∧ C > 0

-- State the theorem
theorem triangle_two_solutions 
  (a b c : ℝ) (A B C : ℝ) 
  (h_triangle : Triangle a b c A B C)
  (h_a : a = 2)
  (h_A : A = Real.pi / 3) -- 60° in radians
  (h_two_solutions : b * Real.sin A < a ∧ a < b) :
  2 < b ∧ b < 4 * Real.sqrt 3 / 3 :=
sorry

end NUMINAMATH_CALUDE_triangle_two_solutions_l1190_119007


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l1190_119016

theorem triangle_angle_calculation (a b : ℝ) (A B : ℝ) :
  a = Real.sqrt 3 * b →
  A = 2 * π / 3 →
  B = π / 6 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_calculation_l1190_119016


namespace NUMINAMATH_CALUDE_salad_cost_main_theorem_l1190_119009

/-- The cost of ingredients for Laura's dinner --/
structure DinnerCost where
  salad_price : ℝ
  beef_price : ℝ
  potato_price : ℝ
  juice_price : ℝ

/-- The quantities of ingredients Laura bought --/
structure DinnerQuantities where
  salad_qty : ℕ
  beef_qty : ℕ
  potato_qty : ℕ
  juice_qty : ℕ

/-- The theorem stating the cost of one salad --/
theorem salad_cost (d : DinnerCost) (q : DinnerQuantities) : d.salad_price = 3 :=
  by
    have h1 : d.beef_price = 2 * d.salad_price := sorry
    have h2 : d.potato_price = (1/3) * d.salad_price := sorry
    have h3 : d.juice_price = 1.5 := sorry
    have h4 : q.salad_qty = 2 ∧ q.beef_qty = 2 ∧ q.potato_qty = 1 ∧ q.juice_qty = 2 := sorry
    have h5 : q.salad_qty * d.salad_price + q.beef_qty * d.beef_price + 
              q.potato_qty * d.potato_price + q.juice_qty * d.juice_price = 22 := sorry
    sorry

/-- The main theorem proving the cost of one salad --/
theorem main_theorem : ∃ (d : DinnerCost) (q : DinnerQuantities), d.salad_price = 3 :=
  by
    sorry

end NUMINAMATH_CALUDE_salad_cost_main_theorem_l1190_119009


namespace NUMINAMATH_CALUDE_flower_color_difference_l1190_119024

/-- Given the following flower counts:
  - Total flowers: 60
  - Yellow and white flowers: 13
  - Red and yellow flowers: 17
  - Red and white flowers: 14
  - Blue and yellow flowers: 16

  Prove that there are 4 more flowers containing red than white. -/
theorem flower_color_difference
  (total : ℕ)
  (yellow_white : ℕ)
  (red_yellow : ℕ)
  (red_white : ℕ)
  (blue_yellow : ℕ)
  (h_total : total = 60)
  (h_yellow_white : yellow_white = 13)
  (h_red_yellow : red_yellow = 17)
  (h_red_white : red_white = 14)
  (h_blue_yellow : blue_yellow = 16) :
  (red_yellow + red_white) - (yellow_white + red_white) = 4 :=
by sorry

end NUMINAMATH_CALUDE_flower_color_difference_l1190_119024


namespace NUMINAMATH_CALUDE_queen_mary_heads_l1190_119004

/-- The number of heads on the luxury liner Queen Mary II -/
def total_heads : ℕ := by sorry

/-- The number of legs on the luxury liner Queen Mary II -/
def total_legs : ℕ := 41

/-- The number of cats on the ship -/
def num_cats : ℕ := 5

/-- The number of legs each cat has -/
def cat_legs : ℕ := 4

/-- The number of legs each sailor or cook has -/
def crew_legs : ℕ := 2

/-- The number of legs the captain has -/
def captain_legs : ℕ := 1

/-- The number of sailors and cooks combined -/
def num_crew : ℕ := by sorry

theorem queen_mary_heads :
  total_heads = num_cats + num_crew + 1 ∧
  total_legs = num_cats * cat_legs + num_crew * crew_legs + captain_legs ∧
  total_heads = 16 := by sorry

end NUMINAMATH_CALUDE_queen_mary_heads_l1190_119004


namespace NUMINAMATH_CALUDE_angle_magnification_l1190_119089

theorem angle_magnification (original_angle : ℝ) (magnification : ℝ) :
  original_angle = 20 ∧ magnification = 10 →
  original_angle = original_angle := by sorry

end NUMINAMATH_CALUDE_angle_magnification_l1190_119089


namespace NUMINAMATH_CALUDE_circumscribed_circle_area_l1190_119011

/-- The area of a circle circumscribed about an equilateral triangle with side length 12 units -/
theorem circumscribed_circle_area (s : ℝ) (h : s = 12) : 
  let r := 2 * s * Real.sqrt 3 / 3
  (π : ℝ) * r^2 = 48 * π := by sorry

end NUMINAMATH_CALUDE_circumscribed_circle_area_l1190_119011


namespace NUMINAMATH_CALUDE_expression_simplification_l1190_119023

theorem expression_simplification (m : ℝ) (h : m = Real.tan (60 * π / 180) - 1) :
  (1 - 2 / (m + 1)) / ((m^2 - 2*m + 1) / (m^2 - m)) = (3 - Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1190_119023


namespace NUMINAMATH_CALUDE_f_inequality_solution_set_l1190_119075

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := (m + 1) * x^2 - (m - 1) * x + m - 1

-- State the theorem
theorem f_inequality_solution_set (m : ℝ) :
  (∀ x : ℝ, f m x < 1) ↔ m < (1 - 2 * Real.sqrt 7) / 3 :=
sorry

end NUMINAMATH_CALUDE_f_inequality_solution_set_l1190_119075


namespace NUMINAMATH_CALUDE_batsman_average_is_59_l1190_119046

/-- Calculates the batting average given the total innings, highest score, 
    average excluding highest and lowest scores, and the difference between highest and lowest scores. -/
def battingAverage (totalInnings : ℕ) (highestScore : ℕ) (averageExcludingExtremes : ℕ) (scoreDifference : ℕ) : ℚ :=
  let lowestScore := highestScore - scoreDifference
  let totalScore := (totalInnings - 2) * averageExcludingExtremes + highestScore + lowestScore
  totalScore / totalInnings

/-- Theorem stating that under the given conditions, the batting average is 59 runs. -/
theorem batsman_average_is_59 :
  battingAverage 46 156 58 150 = 59 := by sorry

end NUMINAMATH_CALUDE_batsman_average_is_59_l1190_119046


namespace NUMINAMATH_CALUDE_original_number_proof_l1190_119096

theorem original_number_proof : 
  ∃ x : ℝ, 3 * (2 * (3 * x) - 9) = 90 ∧ x = 6.5 := by sorry

end NUMINAMATH_CALUDE_original_number_proof_l1190_119096


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l1190_119079

theorem trigonometric_simplification (α : ℝ) : 
  3.4113 * Real.sin α * Real.cos (3 * α) + 
  9 * Real.sin α * Real.cos α - 
  Real.sin (3 * α) * Real.cos (3 * α) - 
  3 * Real.sin (3 * α) * Real.cos α = 
  2 * (Real.sin (2 * α))^3 := by sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l1190_119079


namespace NUMINAMATH_CALUDE_area_of_polygon_l1190_119087

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a polygon -/
structure Polygon :=
  (vertices : List Point)

/-- Calculates the area of a polygon -/
def area (p : Polygon) : ℝ := sorry

/-- Checks if a quadrilateral is a rectangle -/
def is_rectangle (a b c d : Point) : Prop := sorry

/-- Checks if a quadrilateral is a square -/
def is_square (a b f e : Point) : Prop := sorry

/-- Checks if two line segments are perpendicular -/
def is_perpendicular (a f : Point) (f e : Point) : Prop := sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

theorem area_of_polygon (a b c d e f : Point) :
  is_rectangle a b c d →
  is_square a b f e →
  is_perpendicular a f f e →
  distance a f = 10 →
  distance f e = 15 →
  distance c d = 20 →
  area (Polygon.mk [a, f, e, d, c, b]) = 375 := by
  sorry

end NUMINAMATH_CALUDE_area_of_polygon_l1190_119087


namespace NUMINAMATH_CALUDE_angle_C_measure_triangle_area_l1190_119091

noncomputable section

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

/-- The given conditions for the triangle -/
def TriangleConditions (t : Triangle) : Prop :=
  t.c = 2 ∧ 2 * Real.sin t.A = Real.sqrt 3 * t.a * Real.cos t.C

theorem angle_C_measure (t : Triangle) (h : TriangleConditions t) : 
  t.C = π / 3 := by sorry

theorem triangle_area (t : Triangle) (h : TriangleConditions t) 
  (h2 : 2 * Real.sin (2 * t.A) + Real.sin (2 * t.B + t.C) = Real.sin t.C) : 
  (1/2) * t.a * t.b * Real.sin t.C = 2 * Real.sqrt 3 / 3 := by sorry

end NUMINAMATH_CALUDE_angle_C_measure_triangle_area_l1190_119091


namespace NUMINAMATH_CALUDE_pirate_treasure_probability_l1190_119072

def num_islands : ℕ := 8
def num_treasure_islands : ℕ := 4
def prob_treasure : ℚ := 1/5
def prob_trap : ℚ := 1/10
def prob_neither : ℚ := 7/10

theorem pirate_treasure_probability :
  (Nat.choose num_islands num_treasure_islands : ℚ) *
  prob_treasure ^ num_treasure_islands *
  prob_neither ^ (num_islands - num_treasure_islands) =
  33614/1250000 := by sorry

end NUMINAMATH_CALUDE_pirate_treasure_probability_l1190_119072


namespace NUMINAMATH_CALUDE_smallest_base_perfect_square_l1190_119025

theorem smallest_base_perfect_square : 
  ∃ (b : ℕ), b > 4 ∧ b = 6 ∧ ∃ (n : ℕ), 2 * b + 4 = n^2 ∧
  ∀ (c : ℕ), c > 4 ∧ c < b → ¬∃ (m : ℕ), 2 * c + 4 = m^2 :=
sorry

end NUMINAMATH_CALUDE_smallest_base_perfect_square_l1190_119025


namespace NUMINAMATH_CALUDE_smallest_positive_period_l1190_119044

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the symmetric points
variable (a b y₀ : ℝ)

-- Define the symmetry property
def isSymmetric (f : ℝ → ℝ) (x₁ x₂ y : ℝ) : Prop :=
  ∀ t, f (x₁ - t) = 2 * y - f (x₂ + t)

-- State the theorem
theorem smallest_positive_period
  (h₁ : isSymmetric f a a y₀)
  (h₂ : isSymmetric f b b y₀)
  (h₃ : ∀ x, a < x → x < b → ¬ isSymmetric f x x y₀)
  (h₄ : a < b) :
  ∃ T, T > 0 ∧ (∀ x, f (x + T) = f x) ∧
    (∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T') ∧
  T = 2 * (b - a) :=
sorry

end NUMINAMATH_CALUDE_smallest_positive_period_l1190_119044


namespace NUMINAMATH_CALUDE_profit_percentage_l1190_119078

theorem profit_percentage (selling_price cost_price : ℝ) :
  cost_price = 0.96 * selling_price →
  (selling_price - cost_price) / cost_price * 100 = (1 / 24) * 100 := by
sorry

end NUMINAMATH_CALUDE_profit_percentage_l1190_119078


namespace NUMINAMATH_CALUDE_max_correct_answers_l1190_119022

theorem max_correct_answers (total_questions : ℕ) (correct_points : ℤ) (blank_points : ℤ) (incorrect_points : ℤ) (total_score : ℤ) :
  total_questions = 50 →
  correct_points = 4 →
  blank_points = 0 →
  incorrect_points = -1 →
  total_score = 99 →
  ∃ (max_correct : ℕ), 
    max_correct ≤ total_questions ∧
    (∀ (correct blank incorrect : ℕ),
      correct + blank + incorrect = total_questions →
      correct_points * correct + blank_points * blank + incorrect_points * incorrect = total_score →
      correct ≤ max_correct) ∧
    max_correct = 29 :=
by sorry

end NUMINAMATH_CALUDE_max_correct_answers_l1190_119022


namespace NUMINAMATH_CALUDE_probability_one_second_class_l1190_119003

/-- The probability of drawing exactly one second-class product from a batch of products -/
theorem probability_one_second_class (total : ℕ) (first_class : ℕ) (second_class : ℕ) (drawn : ℕ) :
  total = first_class + second_class →
  total = 100 →
  first_class = 90 →
  second_class = 10 →
  drawn = 4 →
  (Nat.choose second_class 1 * Nat.choose first_class 3 : ℚ) / Nat.choose total drawn =
    Nat.choose second_class 1 * Nat.choose first_class 3 / Nat.choose total drawn :=
by sorry

end NUMINAMATH_CALUDE_probability_one_second_class_l1190_119003


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l1190_119038

theorem polynomial_coefficient_sum (A B C D : ℝ) : 
  (∀ x : ℝ, (x - 3) * (4 * x^2 + 2 * x - 6) = A * x^3 + B * x^2 + C * x + D) →
  A + B + C + D = 0 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l1190_119038


namespace NUMINAMATH_CALUDE_bingo_prize_distribution_l1190_119039

theorem bingo_prize_distribution (total_prize : ℝ) (remaining_winners : ℕ) (each_remaining_prize : ℝ) 
  (h1 : total_prize = 2400)
  (h2 : remaining_winners = 10)
  (h3 : each_remaining_prize = 160)
  (h4 : ∀ f : ℝ, (1 - f) * total_prize / remaining_winners = each_remaining_prize → f = 1/3) :
  ∃ f : ℝ, f * total_prize = total_prize / 3 ∧ 
    (1 - f) * total_prize / remaining_winners = each_remaining_prize := by
  sorry

#check bingo_prize_distribution

end NUMINAMATH_CALUDE_bingo_prize_distribution_l1190_119039


namespace NUMINAMATH_CALUDE_sum_of_four_sqrt_inequality_l1190_119008

theorem sum_of_four_sqrt_inequality (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_eq_one : a + b + c + d = 1) :
  Real.sqrt (4 * a + 1) + Real.sqrt (4 * b + 1) + Real.sqrt (4 * c + 1) + Real.sqrt (4 * d + 1) < 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_sqrt_inequality_l1190_119008


namespace NUMINAMATH_CALUDE_fraction_equals_sqrt_two_l1190_119093

theorem fraction_equals_sqrt_two (a b : ℝ) 
  (h1 : a > b) (h2 : b > 0) (h3 : a^2 + b^2 = 6*a*b) : 
  (a + b) / (a - b) = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_fraction_equals_sqrt_two_l1190_119093


namespace NUMINAMATH_CALUDE_tan_monotone_or_angle_sin_equivalence_l1190_119040

-- Define the tangent function
noncomputable def tan (x : ℝ) : ℝ := Real.tan x

-- Define a predicate for monotonically increasing functions
def MonotonicallyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- Define a triangle
structure Triangle :=
  (A B C : ℝ)
  (angle_sum : A + B + C = π)
  (positive_angles : 0 < A ∧ 0 < B ∧ 0 < C)

-- State the theorem
theorem tan_monotone_or_angle_sin_equivalence :
  (MonotonicallyIncreasing tan) ∨ 
  (∀ t : Triangle, t.A > t.B ↔ Real.sin t.A > Real.sin t.B) :=
sorry

end NUMINAMATH_CALUDE_tan_monotone_or_angle_sin_equivalence_l1190_119040


namespace NUMINAMATH_CALUDE_journey_speed_l1190_119081

/-- Given a journey with total distance D and total time T, prove that if a person
    travels 2/3 of D in 1/3 of T at 40 kmph, they must travel at 10 kmph for the
    remaining distance to arrive on time. -/
theorem journey_speed (D T : ℝ) (h_positive : D > 0 ∧ T > 0) : 
  (2/3 * D) / (1/3 * T) = 40 → (1/3 * D) / (2/3 * T) = 10 := by
  sorry

end NUMINAMATH_CALUDE_journey_speed_l1190_119081


namespace NUMINAMATH_CALUDE_area_of_overlapping_squares_l1190_119054

/-- Represents a square in a 2D plane -/
structure Square where
  sideLength : ℝ
  center : ℝ × ℝ

/-- Calculates the area of the region covered by two overlapping squares -/
def areaCoveredByOverlappingSquares (s1 s2 : Square) : ℝ :=
  sorry

/-- Theorem stating the area covered by two specific overlapping squares -/
theorem area_of_overlapping_squares :
  let s1 := Square.mk 12 (0, 0)
  let s2 := Square.mk 12 (6, 6)
  areaCoveredByOverlappingSquares s1 s2 = 144 := by
  sorry

end NUMINAMATH_CALUDE_area_of_overlapping_squares_l1190_119054


namespace NUMINAMATH_CALUDE_max_value_problem_l1190_119037

theorem max_value_problem (a b c : ℝ) (h : 9 * a^2 + 4 * b^2 + 25 * c^2 = 1) :
  (∃ (x y z : ℝ), 9 * x^2 + 4 * y^2 + 25 * z^2 = 1 ∧ 3 * x + 4 * y + 5 * z > 3 * a + 4 * b + 5 * c) →
  3 * a + 4 * b + 5 * c ≤ Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_max_value_problem_l1190_119037


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l1190_119082

theorem solve_exponential_equation :
  ∃ x : ℝ, (5 : ℝ) ^ (2 * x + 3) = 125 ^ (x + 1) ∧ x = 0 := by
sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l1190_119082


namespace NUMINAMATH_CALUDE_water_level_rise_l1190_119069

-- Define the cube's edge length
def cube_edge : ℝ := 16

-- Define the vessel's base dimensions
def vessel_length : ℝ := 20
def vessel_width : ℝ := 15

-- Define the volume of the cube
def cube_volume : ℝ := cube_edge ^ 3

-- Define the area of the vessel's base
def vessel_base_area : ℝ := vessel_length * vessel_width

-- Theorem statement
theorem water_level_rise :
  (cube_volume / vessel_base_area) = (cube_edge ^ 3) / (vessel_length * vessel_width) :=
by sorry

end NUMINAMATH_CALUDE_water_level_rise_l1190_119069


namespace NUMINAMATH_CALUDE_container_volume_scaling_l1190_119021

theorem container_volume_scaling (V k : ℝ) (h : k > 0) :
  let new_volume := V * k^3
  new_volume = V * k * k * k :=
by sorry

end NUMINAMATH_CALUDE_container_volume_scaling_l1190_119021


namespace NUMINAMATH_CALUDE_simplify_expression_l1190_119034

theorem simplify_expression (x : ℝ) (h1 : x ≠ -3) (h2 : x ≠ 1) :
  (x - 5 + 16 / (x + 3)) / ((x - 1) / (x^2 - 9)) = x^2 - 4*x + 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1190_119034


namespace NUMINAMATH_CALUDE_unique_solution_is_four_l1190_119099

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x - 5

-- State the theorem
theorem unique_solution_is_four :
  ∃! x : ℝ, 2 * (f x) - 19 = f (x - 4) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_solution_is_four_l1190_119099


namespace NUMINAMATH_CALUDE_rose_work_days_l1190_119074

/-- Given that Paul completes a work in 80 days and Paul and Rose together
    complete the same work in 48 days, prove that Rose completes the work
    alone in 120 days. -/
theorem rose_work_days (paul_days : ℕ) (together_days : ℕ) (rose_days : ℕ) : 
  paul_days = 80 → together_days = 48 → 
  1 / paul_days + 1 / rose_days = 1 / together_days →
  rose_days = 120 := by
  sorry

end NUMINAMATH_CALUDE_rose_work_days_l1190_119074


namespace NUMINAMATH_CALUDE_g_five_times_one_l1190_119058

def g (x : ℕ) : ℕ :=
  if x % 2 = 0 then x + 2 else 3 * x + 1

theorem g_five_times_one : g (g (g (g (g 1)))) = 12 := by
  sorry

end NUMINAMATH_CALUDE_g_five_times_one_l1190_119058


namespace NUMINAMATH_CALUDE_total_eggs_collected_l1190_119094

def benjamin_eggs : ℕ := 6

def carla_eggs : ℕ := 3 * benjamin_eggs

def trisha_eggs : ℕ := benjamin_eggs - 4

def total_eggs : ℕ := benjamin_eggs + carla_eggs + trisha_eggs

theorem total_eggs_collected :
  total_eggs = 26 := by sorry

end NUMINAMATH_CALUDE_total_eggs_collected_l1190_119094


namespace NUMINAMATH_CALUDE_full_price_revenue_is_2128_l1190_119052

/-- Represents the ticket sale scenario -/
structure TicketSale where
  total_tickets : ℕ
  total_revenue : ℕ
  full_price_tickets : ℕ
  discounted_tickets : ℕ
  full_price : ℕ

/-- The conditions of the ticket sale -/
def valid_ticket_sale (sale : TicketSale) : Prop :=
  sale.total_tickets = 200 ∧
  sale.total_revenue = 2688 ∧
  sale.full_price_tickets + sale.discounted_tickets = sale.total_tickets ∧
  sale.full_price_tickets * sale.full_price + sale.discounted_tickets * (sale.full_price / 3) = sale.total_revenue

/-- The theorem to be proved -/
theorem full_price_revenue_is_2128 (sale : TicketSale) :
  valid_ticket_sale sale →
  sale.full_price_tickets * sale.full_price = 2128 :=
by sorry

end NUMINAMATH_CALUDE_full_price_revenue_is_2128_l1190_119052


namespace NUMINAMATH_CALUDE_trig_identity_l1190_119062

theorem trig_identity : 
  Real.sin (-1071 * π / 180) * Real.sin (99 * π / 180) + 
  Real.sin (-171 * π / 180) * Real.sin (-261 * π / 180) + 
  Real.tan (-1089 * π / 180) * Real.tan (-540 * π / 180) = 0 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1190_119062


namespace NUMINAMATH_CALUDE_min_coach_handshakes_l1190_119017

/-- The total number of handshakes at the gymnastics meet -/
def total_handshakes : ℕ := 903

/-- The number of gymnasts at the meet -/
def n : ℕ := 43

/-- The number of coaches at the meet -/
def num_coaches : ℕ := 3

/-- Function to calculate the number of handshakes between gymnasts -/
def gymnast_handshakes (m : ℕ) : ℕ := m * (m - 1) / 2

/-- Theorem stating the minimum number of handshakes involving coaches -/
theorem min_coach_handshakes : 
  ∃ (k₁ k₂ k₃ : ℕ), 
    gymnast_handshakes n + k₁ + k₂ + k₃ = total_handshakes ∧ 
    k₁ + k₂ + k₃ = 0 := by
  sorry

end NUMINAMATH_CALUDE_min_coach_handshakes_l1190_119017


namespace NUMINAMATH_CALUDE_james_toys_problem_l1190_119042

theorem james_toys_problem (sell_percentage : Real) (buy_price : Real) (sell_price : Real) (total_profit : Real) :
  sell_percentage = 0.8 →
  buy_price = 20 →
  sell_price = 30 →
  total_profit = 800 →
  ∃ initial_toys : Real, initial_toys = 100 ∧ 
    sell_percentage * initial_toys * (sell_price - buy_price) = total_profit := by
  sorry

end NUMINAMATH_CALUDE_james_toys_problem_l1190_119042


namespace NUMINAMATH_CALUDE_exists_equitable_non_symmetric_polygon_l1190_119027

-- Define a polygon as a list of points in 2D space
def Polygon := List (ℝ × ℝ)

-- Function to check if a polygon has no self-intersections
def hasNoSelfIntersections (p : Polygon) : Prop :=
  sorry

-- Function to check if a line through the origin divides a polygon into two regions of equal area
def dividesEquallyThroughOrigin (p : Polygon) (line : ℝ → ℝ) : Prop :=
  sorry

-- Function to check if a polygon is equitable
def isEquitable (p : Polygon) : Prop :=
  ∀ line : ℝ → ℝ, dividesEquallyThroughOrigin p line

-- Function to check if a polygon is centrally symmetric about the origin
def isCentrallySymmetric (p : Polygon) : Prop :=
  sorry

-- Theorem statement
theorem exists_equitable_non_symmetric_polygon :
  ∃ p : Polygon, hasNoSelfIntersections p ∧ isEquitable p ∧ ¬(isCentrallySymmetric p) :=
sorry

end NUMINAMATH_CALUDE_exists_equitable_non_symmetric_polygon_l1190_119027


namespace NUMINAMATH_CALUDE_taxi_charge_proof_l1190_119080

/-- Calculates the total charge for a taxi trip -/
def total_charge (initial_fee : ℚ) (rate_per_increment : ℚ) (increment_distance : ℚ) (trip_distance : ℚ) : ℚ :=
  initial_fee + (trip_distance / increment_distance).floor * rate_per_increment

/-- Proves that the total charge for a 3.6-mile trip is $7.65 -/
theorem taxi_charge_proof :
  let initial_fee : ℚ := 9/4  -- $2.25
  let rate_per_increment : ℚ := 3/10  -- $0.3
  let increment_distance : ℚ := 2/5  -- 2/5 mile
  let trip_distance : ℚ := 18/5  -- 3.6 miles
  total_charge initial_fee rate_per_increment increment_distance trip_distance = 153/20  -- $7.65
  := by sorry


end NUMINAMATH_CALUDE_taxi_charge_proof_l1190_119080


namespace NUMINAMATH_CALUDE_xy_value_l1190_119041

theorem xy_value (x y : ℝ) (h : (x^2 + 6*x + 12) * (5*y^2 + 2*y + 1) = 12/5) : 
  x * y = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l1190_119041


namespace NUMINAMATH_CALUDE_yellow_balls_count_l1190_119018

theorem yellow_balls_count (total : ℕ) (white : ℕ) (green : ℕ) (red : ℕ) (purple : ℕ) (prob : ℚ) :
  total = 100 →
  white = 50 →
  green = 20 →
  red = 17 →
  purple = 3 →
  prob = 4/5 →
  (white + green + (total - white - green - red - purple) : ℚ) / total = prob →
  total - white - green - red - purple = 10 :=
by sorry

end NUMINAMATH_CALUDE_yellow_balls_count_l1190_119018


namespace NUMINAMATH_CALUDE_february_messages_l1190_119043

def text_messages (month : ℕ) : ℕ :=
  2^month

theorem february_messages :
  text_messages 3 = 8 ∧ text_messages 4 = 16 :=
by sorry

end NUMINAMATH_CALUDE_february_messages_l1190_119043


namespace NUMINAMATH_CALUDE_gasoline_price_increase_l1190_119076

/-- The percentage increase in gasoline price from 1972 to 1992 -/
theorem gasoline_price_increase (initial_price final_price : ℝ) : 
  initial_price = 29.90 →
  final_price = 149.70 →
  (final_price - initial_price) / initial_price * 100 = 400 := by
sorry

end NUMINAMATH_CALUDE_gasoline_price_increase_l1190_119076


namespace NUMINAMATH_CALUDE_problem_statement_l1190_119006

theorem problem_statement (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_sum : 1/a + 1/b + 1/c = 1) : 
  (∃ (min : ℝ), min = 36 ∧ ∀ (x y z : ℝ), x > 0 → y > 0 → z > 0 → 1/x + 1/y + 1/z = 1 → x + 4*y + 9*z ≥ min) ∧ 
  ((b+c)/Real.sqrt a + (a+c)/Real.sqrt b + (a+b)/Real.sqrt c ≥ 2 * Real.sqrt (a*b*c)) :=
sorry

end NUMINAMATH_CALUDE_problem_statement_l1190_119006


namespace NUMINAMATH_CALUDE_students_in_general_hall_l1190_119048

theorem students_in_general_hall (general : ℕ) (biology : ℕ) (math : ℕ) : 
  biology = 2 * general →
  math = (3 * (general + biology)) / 5 →
  general + biology + math = 144 →
  general = 30 := by
sorry

end NUMINAMATH_CALUDE_students_in_general_hall_l1190_119048


namespace NUMINAMATH_CALUDE_large_cheese_block_volume_l1190_119060

/-- Represents the dimensions and volume of a cheese block -/
structure CheeseBlock where
  width : ℝ
  depth : ℝ
  length : ℝ
  volume : ℝ

/-- Theorem: Volume of a large cheese block -/
theorem large_cheese_block_volume
  (normal : CheeseBlock)
  (large : CheeseBlock)
  (h1 : normal.volume = 3)
  (h2 : large.width = 2 * normal.width)
  (h3 : large.depth = 2 * normal.depth)
  (h4 : large.length = 3 * normal.length)
  (h5 : large.volume = large.width * large.depth * large.length) :
  large.volume = 36 := by
  sorry

#check large_cheese_block_volume

end NUMINAMATH_CALUDE_large_cheese_block_volume_l1190_119060


namespace NUMINAMATH_CALUDE_always_two_real_roots_root_less_than_one_implies_k_negative_l1190_119092

-- Define the quadratic equation
def quadratic (x k : ℝ) : ℝ := x^2 - (k+3)*x + 2*k + 2

-- Theorem 1: The equation always has two real roots
theorem always_two_real_roots (k : ℝ) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic x₁ k = 0 ∧ quadratic x₂ k = 0 :=
sorry

-- Theorem 2: When one root is less than 1, k < 0
theorem root_less_than_one_implies_k_negative (k : ℝ) :
  (∃ x : ℝ, x < 1 ∧ quadratic x k = 0) → k < 0 :=
sorry

end NUMINAMATH_CALUDE_always_two_real_roots_root_less_than_one_implies_k_negative_l1190_119092


namespace NUMINAMATH_CALUDE_inequality_proof_l1190_119020

theorem inequality_proof (a b c d e f : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (he : 0 < e) (hf : 0 < f) :
  Real.rpow (a * b * c / (a + b + d)) (1/3) + Real.rpow (d * e * f / (c + e + f)) (1/3) 
  < Real.rpow ((a + b + d) * (c + e + f)) (1/3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1190_119020


namespace NUMINAMATH_CALUDE_largest_cube_forming_integer_l1190_119057

theorem largest_cube_forming_integer : 
  ∀ n : ℕ, n > 19 → ¬∃ k : ℤ, n^3 + 4*n^2 - 15*n - 18 = k^3 :=
by sorry

end NUMINAMATH_CALUDE_largest_cube_forming_integer_l1190_119057


namespace NUMINAMATH_CALUDE_sheila_tue_thu_hours_l1190_119090

/-- Represents Sheila's work schedule and earnings -/
structure WorkSchedule where
  hoursPerDayMWF : ℕ  -- Hours worked on Monday, Wednesday, Friday
  daysWorkedMWF : ℕ   -- Number of days worked (Monday, Wednesday, Friday)
  weeklyEarnings : ℕ  -- Total earnings per week
  hourlyRate : ℕ      -- Hourly rate of pay

/-- Calculates the total hours worked on Tuesday and Thursday -/
def hoursTueThu (schedule : WorkSchedule) : ℕ :=
  let mwfHours := schedule.hoursPerDayMWF * schedule.daysWorkedMWF
  let mwfEarnings := mwfHours * schedule.hourlyRate
  let tueThuEarnings := schedule.weeklyEarnings - mwfEarnings
  tueThuEarnings / schedule.hourlyRate

/-- Theorem: Given Sheila's work schedule, she works 12 hours on Tuesday and Thursday combined -/
theorem sheila_tue_thu_hours (schedule : WorkSchedule) 
  (h1 : schedule.hoursPerDayMWF = 8)
  (h2 : schedule.daysWorkedMWF = 3)
  (h3 : schedule.weeklyEarnings = 360)
  (h4 : schedule.hourlyRate = 10) :
  hoursTueThu schedule = 12 := by
  sorry

#eval hoursTueThu { hoursPerDayMWF := 8, daysWorkedMWF := 3, weeklyEarnings := 360, hourlyRate := 10 }

end NUMINAMATH_CALUDE_sheila_tue_thu_hours_l1190_119090


namespace NUMINAMATH_CALUDE_smallest_multiple_with_last_four_digits_l1190_119000

theorem smallest_multiple_with_last_four_digits (n : ℕ) : 
  (n % 10000 = 2020) → (n % 77 = 0) → (∀ m : ℕ, m < n → (m % 10000 ≠ 2020 ∨ m % 77 ≠ 0)) → n = 722020 :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_with_last_four_digits_l1190_119000


namespace NUMINAMATH_CALUDE_largest_four_digit_base4_is_255_l1190_119077

/-- Converts a base-4 digit to its base-10 equivalent -/
def base4DigitToBase10 (digit : Nat) : Nat :=
  if digit < 4 then digit else 0

/-- Calculates the base-10 value of a four-digit base-4 number -/
def fourDigitBase4ToBase10 (d1 d2 d3 d4 : Nat) : Nat :=
  (base4DigitToBase10 d1) * (4^3) +
  (base4DigitToBase10 d2) * (4^2) +
  (base4DigitToBase10 d3) * (4^1) +
  (base4DigitToBase10 d4) * (4^0)

/-- The largest four-digit base-4 number, when converted to base-10, equals 255 -/
theorem largest_four_digit_base4_is_255 :
  fourDigitBase4ToBase10 3 3 3 3 = 255 := by
  sorry

#eval fourDigitBase4ToBase10 3 3 3 3

end NUMINAMATH_CALUDE_largest_four_digit_base4_is_255_l1190_119077


namespace NUMINAMATH_CALUDE_inequality_and_equality_conditions_l1190_119085

theorem inequality_and_equality_conditions (a b c : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 2) (hb : 0 ≤ b ∧ b ≤ 2) (hc : 0 ≤ c ∧ c ≤ 2) :
  (a - b) * (b - c) * (a - c) ≤ 2 ∧
  ((a - b) * (b - c) * (a - c) = 2 ↔ 
    ((a = 0 ∧ b = 2 ∧ c = 1) ∨ 
     (a = 2 ∧ b = 1 ∧ c = 0) ∨ 
     (a = 1 ∧ b = 0 ∧ c = 2))) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_conditions_l1190_119085


namespace NUMINAMATH_CALUDE_specific_conference_handshakes_l1190_119056

/-- The number of handshakes in a conference with gremlins and imps -/
def conference_handshakes (num_gremlins num_imps : ℕ) : ℕ :=
  let gremlin_gremlin := num_gremlins.choose 2
  let gremlin_imp := num_gremlins * num_imps
  gremlin_gremlin + gremlin_imp

/-- Theorem stating the number of handshakes in the specific conference -/
theorem specific_conference_handshakes :
  conference_handshakes 25 10 = 550 := by
  sorry

end NUMINAMATH_CALUDE_specific_conference_handshakes_l1190_119056


namespace NUMINAMATH_CALUDE_hyperbola_focal_distance_l1190_119028

/-- The hyperbola with equation x^2/16 - y^2/20 = 1 -/
def Hyperbola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / 16 - p.2^2 / 20 = 1}

/-- The left focus of the hyperbola -/
def F₁ : ℝ × ℝ := sorry

/-- The right focus of the hyperbola -/
def F₂ : ℝ × ℝ := sorry

/-- Distance between two points in ℝ² -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

theorem hyperbola_focal_distance 
  (P : ℝ × ℝ) 
  (h_P : P ∈ Hyperbola) 
  (h_dist : distance P F₁ = 9) : 
  distance P F₂ = 17 := by sorry

end NUMINAMATH_CALUDE_hyperbola_focal_distance_l1190_119028


namespace NUMINAMATH_CALUDE_polynomial_division_quotient_l1190_119026

theorem polynomial_division_quotient :
  let dividend : Polynomial ℚ := 10 * X^4 + 5 * X^3 - 8 * X^2 + 7 * X - 3
  let divisor : Polynomial ℚ := 3 * X + 2
  let quotient : Polynomial ℚ := (10/3) * X^3 - (5/9) * X^2 - (31/27) * X + 143/81
  (dividend.div divisor = quotient) ∧ (dividend.mod divisor).degree < divisor.degree :=
by sorry

end NUMINAMATH_CALUDE_polynomial_division_quotient_l1190_119026


namespace NUMINAMATH_CALUDE_median_and_mode_are_23_l1190_119059

/-- Represents the shoe size distribution of a class --/
structure ShoeSizeDistribution where
  sizes : List Nat
  frequencies : List Nat
  total_students : Nat
  h_sizes_freq : sizes.length = frequencies.length
  h_total : total_students = frequencies.sum

/-- Calculates the median of a shoe size distribution --/
def median (d : ShoeSizeDistribution) : Nat :=
  sorry

/-- Calculates the mode of a shoe size distribution --/
def mode (d : ShoeSizeDistribution) : Nat :=
  sorry

/-- The shoe size distribution for the class in the problem --/
def class_distribution : ShoeSizeDistribution :=
  { sizes := [20, 21, 22, 23, 24],
    frequencies := [2, 8, 9, 19, 2],
    total_students := 40,
    h_sizes_freq := by rfl,
    h_total := by rfl }

theorem median_and_mode_are_23 :
  median class_distribution = 23 ∧ mode class_distribution = 23 :=
sorry

end NUMINAMATH_CALUDE_median_and_mode_are_23_l1190_119059


namespace NUMINAMATH_CALUDE_largest_c_value_l1190_119067

theorem largest_c_value (c : ℝ) (h : (3 * c + 4) * (c - 2) = 9 * c) : 
  ∀ x : ℝ, (3 * x + 4) * (x - 2) = 9 * x → x ≤ 4 := by sorry

end NUMINAMATH_CALUDE_largest_c_value_l1190_119067


namespace NUMINAMATH_CALUDE_z_in_fourth_quadrant_l1190_119047

def z : ℂ := Complex.I * (-2 - Complex.I)

theorem z_in_fourth_quadrant : 
  Real.sign (z.re) = 1 ∧ Real.sign (z.im) = -1 :=
sorry

end NUMINAMATH_CALUDE_z_in_fourth_quadrant_l1190_119047


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l1190_119098

theorem regular_polygon_sides (n : ℕ) (h : n > 2) : 
  (180 * (n - 2) : ℝ) / n = 156 → n = 15 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l1190_119098


namespace NUMINAMATH_CALUDE_max_value_of_expression_l1190_119035

theorem max_value_of_expression (x y z : ℝ) 
  (h_nonneg_x : x ≥ 0) (h_nonneg_y : y ≥ 0) (h_nonneg_z : z ≥ 0)
  (h_sum : x + y + z = 1) :
  x + y^2 + z^3 ≤ 1 ∧ ∃ (x₀ y₀ z₀ : ℝ), 
    x₀ ≥ 0 ∧ y₀ ≥ 0 ∧ z₀ ≥ 0 ∧ x₀ + y₀ + z₀ = 1 ∧ x₀ + y₀^2 + z₀^3 = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l1190_119035


namespace NUMINAMATH_CALUDE_min_value_theorem_l1190_119051

theorem min_value_theorem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : 1 / x^2 + 1 / y + 1 / z = 6) :
  x^3 * y^2 * z^2 ≥ 1 / (8 * Real.sqrt 2) ∧
  ∃ x₀ y₀ z₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧
    1 / x₀^2 + 1 / y₀ + 1 / z₀ = 6 ∧
    x₀^3 * y₀^2 * z₀^2 = 1 / (8 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1190_119051


namespace NUMINAMATH_CALUDE_school_gender_difference_l1190_119019

theorem school_gender_difference 
  (initial_girls : ℕ) 
  (initial_boys : ℕ) 
  (additional_girls : ℕ) 
  (h1 : initial_girls = 632)
  (h2 : initial_boys = 410)
  (h3 : additional_girls = 465) :
  initial_girls + additional_girls - initial_boys = 687 := by
  sorry

end NUMINAMATH_CALUDE_school_gender_difference_l1190_119019


namespace NUMINAMATH_CALUDE_sector_central_angle_l1190_119005

/-- Given a circular sector with radius 10 cm and perimeter 45 cm, 
    its central angle is 2.5 radians. -/
theorem sector_central_angle : 
  ∀ (r p l α : ℝ), 
    r = 10 → 
    p = 45 → 
    l = p - 2 * r → 
    α = l / r → 
    α = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_sector_central_angle_l1190_119005


namespace NUMINAMATH_CALUDE_problem_statement_l1190_119033

theorem problem_statement (a : ℝ) (h : a^2 - 2*a = 1) : 3*a^2 - 6*a - 4 = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1190_119033


namespace NUMINAMATH_CALUDE_probability_different_colors_is_seven_ninths_l1190_119045

/-- The number of color options for socks -/
def sock_colors : ℕ := 3

/-- The number of color options for headband -/
def headband_colors : ℕ := 3

/-- The number of colors shared between socks and headband options -/
def shared_colors : ℕ := 1

/-- The total number of possible combinations -/
def total_combinations : ℕ := sock_colors * headband_colors

/-- The number of combinations where socks and headband have different colors -/
def different_color_combinations : ℕ := 
  sock_colors * headband_colors - sock_colors * shared_colors

/-- The probability of selecting different colors for socks and headband -/
def probability_different_colors : ℚ := 
  different_color_combinations / total_combinations

theorem probability_different_colors_is_seven_ninths : 
  probability_different_colors = 7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_probability_different_colors_is_seven_ninths_l1190_119045


namespace NUMINAMATH_CALUDE_square_completion_l1190_119030

theorem square_completion (x : ℝ) : x^2 + 5*x + 25/4 = (x + 5/2)^2 := by
  sorry

end NUMINAMATH_CALUDE_square_completion_l1190_119030


namespace NUMINAMATH_CALUDE_taxi_initial_fee_l1190_119031

/-- Represents the taxi service charging model -/
structure TaxiCharge where
  initialFee : ℝ
  additionalChargePerSegment : ℝ
  segmentLength : ℝ
  totalDistance : ℝ
  totalCharge : ℝ

/-- Theorem: Given the taxi service charging model, prove that the initial fee is $2.25 -/
theorem taxi_initial_fee (t : TaxiCharge) : 
  t.additionalChargePerSegment = 0.3 ∧ 
  t.segmentLength = 2/5 ∧ 
  t.totalDistance = 3.6 ∧ 
  t.totalCharge = 4.95 → 
  t.initialFee = 2.25 := by
  sorry

end NUMINAMATH_CALUDE_taxi_initial_fee_l1190_119031


namespace NUMINAMATH_CALUDE_is_center_of_symmetry_l1190_119084

/-- The function f(x) = (x+2)³ - x + 1 -/
def f (x : ℝ) : ℝ := (x + 2)^3 - x + 1

/-- The center of symmetry for the function f -/
def center_of_symmetry : ℝ × ℝ := (-2, 3)

/-- Theorem stating that the given point is the center of symmetry for f -/
theorem is_center_of_symmetry :
  ∀ x : ℝ, f (center_of_symmetry.1 + x) + f (center_of_symmetry.1 - x) = 2 * center_of_symmetry.2 :=
sorry

end NUMINAMATH_CALUDE_is_center_of_symmetry_l1190_119084


namespace NUMINAMATH_CALUDE_iggys_pace_l1190_119083

/-- Iggy's running schedule for the week -/
def daily_miles : List Nat := [3, 4, 6, 8, 3]

/-- Total time Iggy spent running in hours -/
def total_hours : Nat := 4

/-- Calculate Iggy's pace in minutes per mile -/
def calculate_pace (miles : List Nat) (hours : Nat) : Nat :=
  let total_miles := miles.sum
  let total_minutes := hours * 60
  total_minutes / total_miles

/-- Theorem: Iggy's pace is 10 minutes per mile -/
theorem iggys_pace :
  calculate_pace daily_miles total_hours = 10 := by
  sorry

end NUMINAMATH_CALUDE_iggys_pace_l1190_119083


namespace NUMINAMATH_CALUDE_meeting_percentage_is_35_percent_l1190_119001

/-- Represents the duration of a work day in minutes -/
def work_day_minutes : ℕ := 10 * 60

/-- Represents the duration of the first meeting in minutes -/
def first_meeting_minutes : ℕ := 35

/-- Represents the duration of the second meeting in minutes -/
def second_meeting_minutes : ℕ := 2 * first_meeting_minutes

/-- Represents the duration of the third meeting in minutes -/
def third_meeting_minutes : ℕ := first_meeting_minutes + second_meeting_minutes

/-- Represents the total time spent in meetings in minutes -/
def total_meeting_minutes : ℕ := first_meeting_minutes + second_meeting_minutes + third_meeting_minutes

/-- Represents the percentage of the work day spent in meetings -/
def meeting_percentage : ℚ := (total_meeting_minutes : ℚ) / (work_day_minutes : ℚ) * 100

theorem meeting_percentage_is_35_percent : meeting_percentage = 35 := by
  sorry

end NUMINAMATH_CALUDE_meeting_percentage_is_35_percent_l1190_119001


namespace NUMINAMATH_CALUDE_school_survey_sampling_params_l1190_119029

/-- Systematic sampling parameters for a given population and sample size -/
def systematic_sampling_params (population : ℕ) (sample_size : ℕ) : ℕ × ℕ :=
  let n := population % sample_size
  let m := sample_size
  (n, m)

/-- Theorem stating the correct systematic sampling parameters for the given problem -/
theorem school_survey_sampling_params :
  systematic_sampling_params 1553 50 = (3, 50) := by
sorry

end NUMINAMATH_CALUDE_school_survey_sampling_params_l1190_119029


namespace NUMINAMATH_CALUDE_john_bought_three_puzzles_l1190_119014

/-- Represents the number of puzzles John bought -/
def num_puzzles : ℕ := 3

/-- Represents the number of pieces in the first puzzle -/
def first_puzzle_pieces : ℕ := 1000

/-- Represents the number of pieces in the second and third puzzles -/
def other_puzzle_pieces : ℕ := (3 * first_puzzle_pieces) / 2

/-- The total number of pieces in all puzzles -/
def total_pieces : ℕ := 4000

/-- Theorem stating that the number of puzzles John bought is 3 -/
theorem john_bought_three_puzzles :
  num_puzzles = 3 ∧
  first_puzzle_pieces = 1000 ∧
  other_puzzle_pieces = (3 * first_puzzle_pieces) / 2 ∧
  total_pieces = first_puzzle_pieces + 2 * other_puzzle_pieces :=
by sorry

end NUMINAMATH_CALUDE_john_bought_three_puzzles_l1190_119014


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l1190_119063

theorem pure_imaginary_complex_number (m : ℝ) : 
  let z : ℂ := m * (m - 1) + (m - 1) * Complex.I
  (z.re = 0 ∧ z.im ≠ 0) → m = 0 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l1190_119063


namespace NUMINAMATH_CALUDE_burger_cost_l1190_119055

theorem burger_cost : ∃ (burger_cost : ℝ),
  burger_cost = 9 ∧
  ∃ (pizza_cost : ℝ),
  pizza_cost = 2 * burger_cost ∧
  pizza_cost + 3 * burger_cost = 45 := by
sorry

end NUMINAMATH_CALUDE_burger_cost_l1190_119055


namespace NUMINAMATH_CALUDE_infinitely_many_larger_divisor_sum_ratio_l1190_119010

-- Define the sum of divisors function
def sigma (n : ℕ) : ℕ := sorry

-- Define the theorem
theorem infinitely_many_larger_divisor_sum_ratio :
  ∀ t : ℕ, ∃ n : ℕ, n > t ∧ ∀ k : ℕ, k ∈ Finset.range n → (sigma n : ℚ) / n > (sigma k : ℚ) / k :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_larger_divisor_sum_ratio_l1190_119010


namespace NUMINAMATH_CALUDE_jane_max_tickets_l1190_119070

/-- Represents the maximum number of tickets Jane can buy given the conditions. -/
def max_tickets (regular_price : ℕ) (discount_price : ℕ) (budget : ℕ) (discount_threshold : ℕ) : ℕ :=
  let regular_tickets := min discount_threshold (budget / regular_price)
  let remaining_budget := budget - regular_tickets * regular_price
  let extra_tickets := remaining_budget / discount_price
  regular_tickets + extra_tickets

/-- Theorem stating that the maximum number of tickets Jane can buy is 19. -/
theorem jane_max_tickets :
  max_tickets 15 12 135 8 = 19 := by
  sorry

end NUMINAMATH_CALUDE_jane_max_tickets_l1190_119070


namespace NUMINAMATH_CALUDE_one_french_horn_player_l1190_119061

/-- Represents the number of players for each instrument in an orchestra -/
structure Orchestra :=
  (total : ℕ)
  (drummer : ℕ)
  (trombone : ℕ)
  (trumpet : ℕ)
  (violin : ℕ)
  (cello : ℕ)
  (contrabass : ℕ)
  (clarinet : ℕ)
  (flute : ℕ)
  (maestro : ℕ)

/-- Theorem stating that there is one French horn player in the orchestra -/
theorem one_french_horn_player (o : Orchestra) 
  (h_total : o.total = 21)
  (h_drummer : o.drummer = 1)
  (h_trombone : o.trombone = 4)
  (h_trumpet : o.trumpet = 2)
  (h_violin : o.violin = 3)
  (h_cello : o.cello = 1)
  (h_contrabass : o.contrabass = 1)
  (h_clarinet : o.clarinet = 3)
  (h_flute : o.flute = 4)
  (h_maestro : o.maestro = 1) :
  o.total = o.drummer + o.trombone + o.trumpet + o.violin + o.cello + o.contrabass + o.clarinet + o.flute + o.maestro + 1 :=
by sorry

end NUMINAMATH_CALUDE_one_french_horn_player_l1190_119061


namespace NUMINAMATH_CALUDE_omino_tilings_2_by_10_l1190_119095

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- Number of omino tilings for a 1-by-n rectangle -/
def ominoTilings1ByN (n : ℕ) : ℕ := fib (n + 1)

/-- Number of omino tilings for a 2-by-n rectangle -/
def ominoTilings2ByN (n : ℕ) : ℕ := (ominoTilings1ByN n) ^ 2

theorem omino_tilings_2_by_10 : ominoTilings2ByN 10 = 3025 := by
  sorry

end NUMINAMATH_CALUDE_omino_tilings_2_by_10_l1190_119095


namespace NUMINAMATH_CALUDE_cycle_original_price_l1190_119036

/-- Given a cycle sold at a 10% loss for Rs. 1080, prove its original price was Rs. 1200 -/
theorem cycle_original_price (selling_price : ℝ) (loss_percentage : ℝ) :
  selling_price = 1080 →
  loss_percentage = 10 →
  selling_price = (1 - loss_percentage / 100) * 1200 :=
by sorry

end NUMINAMATH_CALUDE_cycle_original_price_l1190_119036


namespace NUMINAMATH_CALUDE_sqrt_x_over_5_increase_l1190_119002

theorem sqrt_x_over_5_increase (x : ℝ) (hx : x > 0) :
  let x_new := x * 1.69
  let original := Real.sqrt (x / 5)
  let new_value := Real.sqrt (x_new / 5)
  (new_value - original) / original * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_over_5_increase_l1190_119002


namespace NUMINAMATH_CALUDE_triangle_angle_relation_l1190_119068

theorem triangle_angle_relation (X Y Z Z₁ Z₂ : ℝ) : 
  X = 40 → Y = 50 → X + Y + Z = 180 → Z = Z₁ + Z₂ → Z₁ - Z₂ = 10 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_relation_l1190_119068

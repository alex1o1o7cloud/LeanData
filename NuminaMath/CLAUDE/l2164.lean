import Mathlib

namespace sunlovers_always_happy_l2164_216455

theorem sunlovers_always_happy (D R : ℕ) : 
  (D^2 + 4) * (R^2 + 4) - 2 * D * (R^2 + 4) - 2 * R * (D^2 + 4) ≥ 0 := by
  sorry

end sunlovers_always_happy_l2164_216455


namespace tens_digit_of_36_pow_12_l2164_216486

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

theorem tens_digit_of_36_pow_12 : tens_digit (36^12) = 3 := by
  sorry

end tens_digit_of_36_pow_12_l2164_216486


namespace shopkeeper_loss_l2164_216481

theorem shopkeeper_loss (X : ℝ) (h : X > 0) : 
  let intended_sale_price := 1.1 * X
  let remaining_goods_value := 0.4 * X
  let actual_sale_price := 1.1 * remaining_goods_value
  let loss := X - actual_sale_price
  let percentage_loss := (loss / X) * 100
  percentage_loss = 56 := by sorry

end shopkeeper_loss_l2164_216481


namespace expired_milk_probability_l2164_216428

theorem expired_milk_probability (total_bags : ℕ) (expired_bags : ℕ) 
  (h1 : total_bags = 25) (h2 : expired_bags = 4) :
  (expired_bags : ℚ) / total_bags = 4 / 25 :=
by sorry

end expired_milk_probability_l2164_216428


namespace tangent_line_to_exponential_curve_l2164_216438

/-- Given that the line y = x - 1 is tangent to the curve y = e^(x+a), prove that a = -2 --/
theorem tangent_line_to_exponential_curve (a : ℝ) : 
  (∃ x₀ : ℝ, x₀ - 1 = Real.exp (x₀ + a) ∧ 1 = Real.exp (x₀ + a)) → a = -2 := by
  sorry

end tangent_line_to_exponential_curve_l2164_216438


namespace geometric_sum_n1_l2164_216464

theorem geometric_sum_n1 (a : ℝ) (h : a ≠ 1) :
  1 + a + a^2 = (1 - a^3) / (1 - a) := by sorry

end geometric_sum_n1_l2164_216464


namespace smallest_fourth_number_l2164_216477

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

theorem smallest_fourth_number (a b c d : ℕ) 
  (ha : is_two_digit a) (hb : is_two_digit b) (hc : is_two_digit c) (hd : is_two_digit d)
  (h1 : a = 45) (h2 : b = 26) (h3 : c = 63)
  (h4 : sum_of_digits a + sum_of_digits b + sum_of_digits c + sum_of_digits d = (a + b + c + d) / 3)
  (h5 : (a + b + c + d) % 7 = 0) :
  d ≥ 37 := by
sorry

end smallest_fourth_number_l2164_216477


namespace max_area_of_similar_triangle_l2164_216439

/-- Represents a point in a 2D grid -/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- Represents a triangle in the grid -/
structure GridTriangle where
  A : GridPoint
  B : GridPoint
  C : GridPoint

/-- The side length of the square grid -/
def gridSize : ℕ := 5

/-- Function to check if a point is within the grid -/
def isInGrid (p : GridPoint) : Prop :=
  0 ≤ p.x ∧ p.x < gridSize ∧ 0 ≤ p.y ∧ p.y < gridSize

/-- Function to check if a triangle is within the grid -/
def isTriangleInGrid (t : GridTriangle) : Prop :=
  isInGrid t.A ∧ isInGrid t.B ∧ isInGrid t.C

/-- Function to calculate the area of a triangle given its vertices -/
noncomputable def triangleArea (t : GridTriangle) : ℝ :=
  sorry -- Actual calculation would go here

/-- Function to check if two triangles are similar -/
def areSimilarTriangles (t1 t2 : GridTriangle) : Prop :=
  sorry -- Actual similarity check would go here

theorem max_area_of_similar_triangle :
  ∀ (ABC : GridTriangle),
    isTriangleInGrid ABC →
    ∃ (DEF : GridTriangle),
      isTriangleInGrid DEF ∧
      areSimilarTriangles ABC DEF ∧
      triangleArea DEF ≤ 2.5 ∧
      ∀ (XYZ : GridTriangle),
        isTriangleInGrid XYZ →
        areSimilarTriangles ABC XYZ →
        triangleArea XYZ ≤ triangleArea DEF :=
by sorry


end max_area_of_similar_triangle_l2164_216439


namespace populations_equal_after_16_years_l2164_216433

def village_x_initial_population : ℕ := 74000
def village_x_decrease_rate : ℕ := 1200
def village_y_initial_population : ℕ := 42000
def village_y_increase_rate : ℕ := 800

def population_equal_time : ℕ := 16

theorem populations_equal_after_16_years :
  village_x_initial_population - population_equal_time * village_x_decrease_rate =
  village_y_initial_population + population_equal_time * village_y_increase_rate :=
by sorry

end populations_equal_after_16_years_l2164_216433


namespace sally_plums_l2164_216410

theorem sally_plums (melanie_plums dan_plums total_plums : ℕ) 
  (h1 : melanie_plums = 4)
  (h2 : dan_plums = 9)
  (h3 : total_plums = 16)
  (h4 : ∃ sally_plums : ℕ, melanie_plums + dan_plums + sally_plums = total_plums) :
  ∃ sally_plums : ℕ, sally_plums = 3 ∧ melanie_plums + dan_plums + sally_plums = total_plums := by
  sorry

end sally_plums_l2164_216410


namespace three_digit_number_problem_l2164_216443

theorem three_digit_number_problem :
  ∃ (A : ℕ),
    (A ≥ 100 ∧ A < 1000) ∧  -- A is a three-digit number
    (A / 100 ≠ 0 ∧ (A / 10) % 10 ≠ 0 ∧ A % 10 ≠ 0) ∧  -- A does not contain zeroes
    (∃ (B : ℕ),
      (B ≥ 10 ∧ B < 100) ∧  -- B is a two-digit number
      (B = (A / 100 + (A / 10) % 10) * 10 + A % 10) ∧  -- B is formed by summing first two digits of A
      (A = 3 * B)) ∧  -- A = 3B
    A = 135  -- The specific value of A
  := by sorry

end three_digit_number_problem_l2164_216443


namespace arithmetic_sequence_formula_l2164_216412

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- The theorem states that for an arithmetic sequence satisfying
    the given conditions, the general term is 2n - 3. -/
theorem arithmetic_sequence_formula (a : ℕ → ℝ)
    (h_arith : is_arithmetic_sequence a)
    (h_mean1 : (a 2 + a 6) / 2 = 5)
    (h_mean2 : (a 3 + a 7) / 2 = 7) :
    ∀ n : ℕ, a n = 2 * n - 3 := by
  sorry

end arithmetic_sequence_formula_l2164_216412


namespace wheat_bags_theorem_l2164_216459

/-- Represents the deviation of each bag from the standard weight -/
def deviations : List Int := [-6, -3, -1, 7, 3, 4, -3, -2, -2, 1]

/-- The number of bags -/
def num_bags : Nat := 10

/-- The standard weight per bag in kg -/
def standard_weight : Int := 150

/-- The sum of all deviations -/
def total_deviation : Int := deviations.sum

/-- The average weight per bag -/
noncomputable def average_weight : ℚ := 
  (num_bags * standard_weight + total_deviation) / num_bags

theorem wheat_bags_theorem : 
  total_deviation = -2 ∧ average_weight = 149.8 := by sorry

end wheat_bags_theorem_l2164_216459


namespace expression_simplification_l2164_216414

theorem expression_simplification (x : ℝ) : 
  3 * x + 4 * (2 - x) - 2 * (3 - 2 * x) + 5 * (2 + 3 * x) = 18 * x + 12 := by
  sorry

end expression_simplification_l2164_216414


namespace count_valid_selections_32_card_deck_l2164_216451

/-- Represents a deck of cards -/
structure Deck :=
  (total_cards : Nat)
  (num_suits : Nat)
  (cards_per_suit : Nat)
  (h_total : total_cards = num_suits * cards_per_suit)

/-- Calculates the number of ways to choose 6 cards from a deck
    such that all suits are represented -/
def count_valid_selections (d : Deck) : Nat :=
  let s1 := Nat.choose 4 2 * (Nat.choose 8 2)^2 * 8^2
  let s2 := Nat.choose 4 1 * Nat.choose 8 3 * 8^3
  s1 + s2

/-- The main theorem to be proved -/
theorem count_valid_selections_32_card_deck :
  ∃ (d : Deck), d.total_cards = 32 ∧ d.num_suits = 4 ∧ d.cards_per_suit = 8 ∧
  count_valid_selections d = 415744 :=
by
  sorry

end count_valid_selections_32_card_deck_l2164_216451


namespace sufficient_but_not_necessary_l2164_216430

theorem sufficient_but_not_necessary (x : ℝ) : 
  (∀ x, x > 3 → x^2 - 2*x > 0) ∧ 
  (∃ x, x^2 - 2*x > 0 ∧ x ≤ 3) := by
  sorry

end sufficient_but_not_necessary_l2164_216430


namespace intersecting_line_passes_through_fixed_point_l2164_216425

/-- An ellipse with eccentricity 1/2 passing through (1, 3/2) -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b ∧ b > 0
  h_ecc : (a^2 - b^2) / a^2 = 1/4
  h_point : 1^2 / a^2 + (3/2)^2 / b^2 = 1

/-- A line intersecting the ellipse -/
structure IntersectingLine (E : Ellipse) where
  k : ℝ
  m : ℝ
  h_intersect : ∃ (x₁ y₁ x₂ y₂ : ℝ),
    x₁ ≠ x₂ ∧
    y₁ = k * x₁ + m ∧
    y₂ = k * x₂ + m ∧
    x₁^2 / E.a^2 + y₁^2 / E.b^2 = 1 ∧
    x₂^2 / E.a^2 + y₂^2 / E.b^2 = 1

/-- The theorem stating that the line passes through a fixed point -/
theorem intersecting_line_passes_through_fixed_point (E : Ellipse) (l : IntersectingLine E) :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    x₁ ≠ x₂ ∧
    y₁ = l.k * x₁ + l.m ∧
    y₂ = l.k * x₂ + l.m ∧
    x₁^2 / E.a^2 + y₁^2 / E.b^2 = 1 ∧
    x₂^2 / E.a^2 + y₂^2 / E.b^2 = 1 ∧
    (x₁ - E.a) * (x₂ - E.a) + y₁ * y₂ = 0 →
    l.k * (2/7) + l.m = 0 :=
by sorry

end intersecting_line_passes_through_fixed_point_l2164_216425


namespace school_classrooms_problem_l2164_216488

theorem school_classrooms_problem (original_desks new_desks new_classrooms : ℕ) 
  (h1 : original_desks = 539)
  (h2 : new_desks = 1080)
  (h3 : new_classrooms = 9)
  (h4 : ∃ (original_classrooms : ℕ), original_classrooms > 0 ∧ original_desks % original_classrooms = 0)
  (h5 : ∃ (current_classrooms : ℕ), current_classrooms = original_classrooms + new_classrooms)
  (h6 : ∃ (new_desks_per_classroom : ℕ), new_desks_per_classroom > 0 ∧ new_desks % current_classrooms = 0)
  (h7 : ∀ (original_desks_per_classroom : ℕ), 
    original_desks_per_classroom > 0 → 
    original_desks = original_classrooms * original_desks_per_classroom → 
    new_desks_per_classroom > original_desks_per_classroom) :
  current_classrooms = 20 :=
sorry

end school_classrooms_problem_l2164_216488


namespace triangle_side_length_l2164_216461

theorem triangle_side_length 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h1 : a = Real.sqrt 3) 
  (h2 : B = π / 4) 
  (h3 : A = π / 3) 
  (h4 : C = π - A - B) 
  (h5 : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h6 : 0 < A ∧ A < π) 
  (h7 : 0 < B ∧ B < π) 
  (h8 : 0 < C ∧ C < π) 
  (h9 : a / Real.sin A = b / Real.sin B) 
  (h10 : a / Real.sin A = c / Real.sin C) 
  (h11 : c^2 = a^2 + b^2 - 2*a*b*Real.cos C) : 
  c = (Real.sqrt 6 + Real.sqrt 2) / 2 := by
sorry

end triangle_side_length_l2164_216461


namespace alphabet_letter_count_l2164_216409

theorem alphabet_letter_count (total : ℕ) (both : ℕ) (line_only : ℕ) 
  (h1 : total = 40)
  (h2 : both = 11)
  (h3 : line_only = 24)
  (h4 : total = both + line_only + (total - (both + line_only))) :
  total - (both + line_only) = 5 := by
  sorry

end alphabet_letter_count_l2164_216409


namespace a_in_range_and_negative_one_in_A_l2164_216434

def A : Set ℝ := {x | x^2 - 2 < 0}

theorem a_in_range_and_negative_one_in_A (a : ℝ) (h : a ∈ A) :
  -Real.sqrt 2 < a ∧ a < Real.sqrt 2 ∧ -1 ∈ A := by
  sorry

end a_in_range_and_negative_one_in_A_l2164_216434


namespace problem_solution_l2164_216482

theorem problem_solution : (-15) / (1/3 - 3 - 3/2) * 6 = 108/5 := by
  sorry

end problem_solution_l2164_216482


namespace sum_equals_seventeen_l2164_216420

theorem sum_equals_seventeen 
  (a b c d : ℝ) 
  (h1 : a * (c + d) + b * (c + d) = 42) 
  (h2 : c + d = 3) : 
  a + b + c + d = 17 := by
sorry

end sum_equals_seventeen_l2164_216420


namespace complex_purely_imaginary_l2164_216475

theorem complex_purely_imaginary (a b : ℝ) :
  (∃ (z : ℂ), z = Complex.I * a + b ∧ z.re = 0 ∧ z.im ≠ 0) ↔ (a ≠ 0 ∧ b = 0) := by
  sorry

end complex_purely_imaginary_l2164_216475


namespace solution_set_part_i_solution_set_part_ii_l2164_216484

-- Define the function f
def f (x m : ℝ) : ℝ := |x + 1| + |x - 2| - m

-- Part I
theorem solution_set_part_i :
  ∀ x : ℝ, f x 5 > 0 ↔ x > 3 ∨ x < -2 :=
sorry

-- Part II
theorem solution_set_part_ii :
  ∀ m : ℝ, (∀ x : ℝ, f x m ≥ 2) ↔ m ≤ 1 :=
sorry

end solution_set_part_i_solution_set_part_ii_l2164_216484


namespace flour_for_cake_l2164_216404

theorem flour_for_cake (total_flour : ℚ) (scoop_size : ℚ) (num_scoops : ℕ) : 
  total_flour = 8 →
  scoop_size = 1/4 →
  num_scoops = 8 →
  total_flour - (↑num_scoops * scoop_size) = 6 :=
by sorry

end flour_for_cake_l2164_216404


namespace triangle_angle_C_l2164_216490

open Real

theorem triangle_angle_C (A B C : ℝ) (a b c : ℝ) : 
  A = π/3 → a = 3 → c = Real.sqrt 6 → 
  (sin C = sin (π/4) ∨ sin C = sin (3*π/4)) := by
  sorry

end triangle_angle_C_l2164_216490


namespace fraction_equality_l2164_216400

theorem fraction_equality (a b : ℝ) (h : a / b = 2 / 5) : a / (a + b) = 2 / 7 := by
  sorry

end fraction_equality_l2164_216400


namespace circle_center_and_radius_l2164_216416

/-- Given a circle described by the equation x^2 + y^2 - 8 = 2x - 4y,
    prove that its center is at (1, -2) and its radius is √13. -/
theorem circle_center_and_radius :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (1, -2) ∧
    radius = Real.sqrt 13 ∧
    ∀ (x y : ℝ), x^2 + y^2 - 8 = 2*x - 4*y ↔
      (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
by sorry

end circle_center_and_radius_l2164_216416


namespace correct_base_notation_l2164_216454

def is_valid_base_representation (digits : List Nat) (base : Nat) : Prop :=
  digits.all (· < base) ∧ digits.head! > 0

theorem correct_base_notation :
  is_valid_base_representation [7, 5, 1] 9 ∧
  ¬is_valid_base_representation [7, 5, 1] 7 ∧
  ¬is_valid_base_representation [0, 9, 5] 12 ∧
  ¬is_valid_base_representation [9, 0, 1] 2 :=
by sorry

end correct_base_notation_l2164_216454


namespace billys_restaurant_bill_l2164_216419

/-- Calculates the total bill for a group at Billy's Restaurant -/
def calculate_bill (num_adults : ℕ) (num_children : ℕ) (cost_per_meal : ℕ) : ℕ :=
  (num_adults + num_children) * cost_per_meal

/-- Proves that the bill for 2 adults and 5 children, with meals costing $3 each, is $21 -/
theorem billys_restaurant_bill :
  calculate_bill 2 5 3 = 21 := by
  sorry

end billys_restaurant_bill_l2164_216419


namespace triangle_ABC_properties_l2164_216408

variable (a b c : ℝ)
variable (A B C : ℝ)

-- Define triangle ABC
def triangle_ABC (a b c A B C : ℝ) : Prop :=
  -- Sides a, b, c are opposite to angles A, B, C respectively
  true

-- Define the given equation
def given_equation (a b c A C : ℝ) : Prop :=
  (2 * b - c) * Real.cos A = a * Real.cos C

-- Define the given conditions
def given_conditions (a b c : ℝ) : Prop :=
  a = 2 ∧ b + c = 4

-- Theorem statement
theorem triangle_ABC_properties 
  (h_triangle : triangle_ABC a b c A B C)
  (h_equation : given_equation a b c A C)
  (h_conditions : given_conditions a b c) :
  A = Real.pi / 3 ∧ 
  (1/2) * b * c * Real.sin A = Real.sqrt 3 :=
sorry

end triangle_ABC_properties_l2164_216408


namespace g_at_negative_two_l2164_216469

def g (x : ℝ) : ℝ := 3*x^4 - 20*x^3 + 35*x^2 - 28*x - 84

theorem g_at_negative_two : g (-2) = 320 := by
  sorry

end g_at_negative_two_l2164_216469


namespace min_value_of_function_min_value_attained_l2164_216493

theorem min_value_of_function (x : ℝ) (h : x > -1) :
  x - 4 + 9 / (x + 1) ≥ 1 :=
sorry

theorem min_value_attained (x : ℝ) (h : x > -1) :
  x - 4 + 9 / (x + 1) = 1 ↔ x = 2 :=
sorry

end min_value_of_function_min_value_attained_l2164_216493


namespace sum_of_largest_and_smallest_prime_factors_of_1140_l2164_216496

theorem sum_of_largest_and_smallest_prime_factors_of_1140 :
  ∃ (smallest largest : ℕ),
    smallest.Prime ∧ largest.Prime ∧
    smallest ∣ 1140 ∧ largest ∣ 1140 ∧
    (∀ p : ℕ, p.Prime → p ∣ 1140 → p ≤ largest) ∧
    (∀ p : ℕ, p.Prime → p ∣ 1140 → p ≥ smallest) ∧
    smallest + largest = 21 :=
by sorry

end sum_of_largest_and_smallest_prime_factors_of_1140_l2164_216496


namespace fourth_to_first_class_ratio_l2164_216497

def num_classes : ℕ := 6
def students_first_class : ℕ := 20
def students_second_third_class : ℕ := 25
def students_fifth_sixth_class : ℕ := 28
def total_students : ℕ := 136

theorem fourth_to_first_class_ratio :
  ∃ (students_fourth_class : ℕ),
    students_first_class +
    2 * students_second_third_class +
    students_fourth_class +
    2 * students_fifth_sixth_class = total_students ∧
    students_fourth_class * 2 = students_first_class :=
by
  sorry

end fourth_to_first_class_ratio_l2164_216497


namespace interesting_trapezoid_area_interesting_trapezoid_area_range_l2164_216450

/-- An interesting isosceles trapezoid inscribed in a unit square. -/
structure InterestingTrapezoid where
  /-- Parameter determining the position of the trapezoid's vertices. -/
  a : ℝ
  /-- The parameter a is between 0 and 1/2 inclusive. -/
  h_a_range : 0 ≤ a ∧ a ≤ 1/2

/-- The vertices of the trapezoid. -/
def vertices (t : InterestingTrapezoid) : Fin 4 → ℝ × ℝ
  | 0 => (t.a, 0)
  | 1 => (1, t.a)
  | 2 => (1 - t.a, 1)
  | 3 => (0, 1 - t.a)

/-- The area of an interesting isosceles trapezoid. -/
def area (t : InterestingTrapezoid) : ℝ := 1 - 2 * t.a

/-- Theorem: The area of an interesting isosceles trapezoid is 1 - 2a. -/
theorem interesting_trapezoid_area (t : InterestingTrapezoid) :
  area t = 1 - 2 * t.a :=
by sorry

/-- Theorem: The area of an interesting isosceles trapezoid is between 0 and 1 inclusive. -/
theorem interesting_trapezoid_area_range (t : InterestingTrapezoid) :
  0 ≤ area t ∧ area t ≤ 1 :=
by sorry

end interesting_trapezoid_area_interesting_trapezoid_area_range_l2164_216450


namespace fermat_500_units_digit_l2164_216476

def fermat (n : ℕ) : ℕ := 2^(2^n) + 1

theorem fermat_500_units_digit :
  fermat 500 % 10 = 7 := by sorry

end fermat_500_units_digit_l2164_216476


namespace non_matching_pairings_eq_twenty_l2164_216471

/-- The number of colors available for bowls and glasses -/
def num_colors : ℕ := 5

/-- The number of non-matching pairings between bowls and glasses -/
def non_matching_pairings : ℕ := num_colors * (num_colors - 1)

/-- Theorem stating that the number of non-matching pairings is 20 -/
theorem non_matching_pairings_eq_twenty : non_matching_pairings = 20 := by
  sorry

end non_matching_pairings_eq_twenty_l2164_216471


namespace y₁_less_than_y₂_l2164_216429

/-- Linear function f(x) = 8x - 1 -/
def f (x : ℝ) : ℝ := 8 * x - 1

/-- Point P₁ lies on the graph of f -/
def P₁_on_f (y₁ : ℝ) : Prop := f 3 = y₁

/-- Point P₂ lies on the graph of f -/
def P₂_on_f (y₂ : ℝ) : Prop := f 4 = y₂

theorem y₁_less_than_y₂ (y₁ y₂ : ℝ) (h₁ : P₁_on_f y₁) (h₂ : P₂_on_f y₂) : y₁ < y₂ := by
  sorry

end y₁_less_than_y₂_l2164_216429


namespace euler_line_concurrency_l2164_216405

/-- A point in the plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- The Euler line of a triangle -/
def EulerLine (A B C : Point) : Set Point := sorry

/-- The point of concurrency of three lines -/
def Concurrent (l1 l2 l3 : Set Point) : Point := sorry

/-- Predicate to check if a triangle is not obtuse -/
def NotObtuse (A B C : Point) : Prop := sorry

theorem euler_line_concurrency 
  (A B C D : Point) 
  (h1 : NotObtuse A B C) 
  (h2 : NotObtuse B C D) 
  (h3 : NotObtuse C A D) 
  (h4 : NotObtuse D A B) 
  (P : Point) 
  (hP : P = Concurrent (EulerLine A B C) (EulerLine B C D) (EulerLine C A D)) :
  P ∈ EulerLine D A B := by
  sorry

end euler_line_concurrency_l2164_216405


namespace current_intensity_bound_l2164_216422

/-- Given a voltage and a minimum resistance, the current intensity is bounded above. -/
theorem current_intensity_bound (U R : ℝ) (hU : U = 200) (hR : R ≥ 62.5) :
  let I := U / R
  I ≤ 3.2 := by
  sorry

end current_intensity_bound_l2164_216422


namespace simplify_expression_l2164_216436

theorem simplify_expression : 
  (3 * Real.sqrt 12 - 2 * Real.sqrt (1/3) + Real.sqrt 48) / (2 * Real.sqrt 3) = 14/3 := by
  sorry

end simplify_expression_l2164_216436


namespace quadratic_coefficient_of_equation_l2164_216494

theorem quadratic_coefficient_of_equation : ∃ (a b c d e f : ℝ),
  (∀ x, a * x^2 + b * x + c = d * x^2 + e * x + f) →
  (a = 5 ∧ b = -1 ∧ c = -3 ∧ d = 1 ∧ e = 1 ∧ f = -3) →
  (a - d = 4) := by sorry

end quadratic_coefficient_of_equation_l2164_216494


namespace probability_at_least_one_head_l2164_216447

theorem probability_at_least_one_head (p : ℝ) (h1 : p = 1 / 2) :
  1 - (1 - p)^4 = 15 / 16 := by
  sorry

end probability_at_least_one_head_l2164_216447


namespace snail_return_time_is_whole_hours_l2164_216499

/-- Represents the movement of a snail on a plane -/
structure SnailMovement where
  speed : ℝ
  turnInterval : ℝ
  turnAngle : ℝ

/-- Represents the time taken for the snail to return to its starting point -/
def returnTime (movement : SnailMovement) : ℝ := sorry

/-- Theorem stating that the return time is always an integer multiple of hours -/
theorem snail_return_time_is_whole_hours (movement : SnailMovement) 
  (h1 : movement.speed > 0)
  (h2 : movement.turnInterval = 0.25) -- 15 minutes = 0.25 hours
  (h3 : movement.turnAngle = π / 2) -- right angle
  : ∃ n : ℕ, returnTime movement = n := by sorry

end snail_return_time_is_whole_hours_l2164_216499


namespace only_C_is_lying_l2164_216456

-- Define the possible scores
def possible_scores : List ℕ := [1, 3, 5, 7, 9]

-- Define a structure for a person's statement
structure Statement where
  shots : ℕ
  hits : ℕ
  total_score : ℕ

-- Define the statements of A, B, C, and D
def statement_A : Statement := ⟨5, 5, 35⟩
def statement_B : Statement := ⟨6, 6, 36⟩
def statement_C : Statement := ⟨3, 3, 24⟩
def statement_D : Statement := ⟨4, 3, 21⟩

-- Define a function to check if a statement is valid
def is_valid_statement (s : Statement) (scores : List ℕ) : Prop :=
  ∃ (score_list : List ℕ),
    score_list.length = s.hits ∧
    score_list.sum = s.total_score ∧
    ∀ x ∈ score_list, x ∈ scores

-- Theorem stating that C's statement is false while others are true
theorem only_C_is_lying :
  is_valid_statement statement_A possible_scores ∧
  is_valid_statement statement_B possible_scores ∧
  ¬is_valid_statement statement_C possible_scores ∧
  is_valid_statement statement_D possible_scores :=
sorry

end only_C_is_lying_l2164_216456


namespace parabola_intersection_points_l2164_216462

/-- The parabola function -/
def f (x : ℝ) : ℝ := -x^2 + 4*x - 4

/-- Theorem: The number of intersection points between the parabola y = -x^2 + 4x - 4 
    and the coordinate axes is equal to 2 -/
theorem parabola_intersection_points : 
  (∃! x : ℝ, f x = 0) ∧ (∃! y : ℝ, f 0 = y) ∧ 
  (∀ x y : ℝ, (x = 0 ∨ y = 0) → (y = f x) → (x = 0 ∧ y = f 0) ∨ (y = 0 ∧ f x = 0)) :=
sorry

end parabola_intersection_points_l2164_216462


namespace rational_times_sqrt_two_rational_implies_zero_l2164_216480

theorem rational_times_sqrt_two_rational_implies_zero (x : ℚ) :
  (∃ (y : ℚ), y = x * Real.sqrt 2) → x = 0 := by
  sorry

end rational_times_sqrt_two_rational_implies_zero_l2164_216480


namespace soccer_substitutions_remainder_l2164_216421

/-- Represents the number of ways to make substitutions in a soccer game -/
def substitutions (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | m + 1 => 11 * (12 - m) * substitutions m

/-- The total number of ways to make 0 to 3 substitutions -/
def total_substitutions : ℕ :=
  substitutions 0 + substitutions 1 + substitutions 2 + substitutions 3

theorem soccer_substitutions_remainder :
  total_substitutions ≡ 122 [MOD 1000] := by
  sorry

end soccer_substitutions_remainder_l2164_216421


namespace friend_of_gcd_l2164_216413

/-- Two integers are friends if their product is a perfect square -/
def are_friends (a b : ℤ) : Prop := ∃ k : ℤ, a * b = k * k

/-- Main theorem: If a is a friend of b, then a is a friend of gcd(a, b) -/
theorem friend_of_gcd {a b : ℤ} (h : are_friends a b) : are_friends a (Int.gcd a b) := by
  sorry

end friend_of_gcd_l2164_216413


namespace normal_distribution_symmetry_l2164_216472

-- Define a random variable following a normal distribution
def normal_distribution (μ σ : ℝ) : Type := ℝ

-- Define the probability function
def P (ξ : normal_distribution 3 σ) (pred : ℝ → Prop) : ℝ := sorry

-- Theorem statement
theorem normal_distribution_symmetry (σ : ℝ) (c : ℝ) :
  (∀ (ξ : normal_distribution 3 σ), P ξ (λ x => x > c + 1) = P ξ (λ x => x < c - 1)) →
  c = 3 :=
by sorry

end normal_distribution_symmetry_l2164_216472


namespace james_total_earnings_l2164_216432

def january_earnings : ℝ := 4000

def february_earnings (jan : ℝ) : ℝ := jan * 1.5

def march_earnings (feb : ℝ) : ℝ := feb * 0.8

def total_earnings (jan feb mar : ℝ) : ℝ := jan + feb + mar

theorem james_total_earnings :
  let feb := february_earnings january_earnings
  let mar := march_earnings feb
  total_earnings january_earnings feb mar = 14800 := by sorry

end james_total_earnings_l2164_216432


namespace detergent_volume_in_new_solution_l2164_216479

/-- Represents the components of a cleaning solution -/
inductive Component
| Bleach
| Detergent
| Water

/-- Represents the ratio of components in a solution -/
def Ratio := Component → ℚ

def original_ratio : Ratio :=
  fun c => match c with
  | Component.Bleach => 4
  | Component.Detergent => 40
  | Component.Water => 100

def new_ratio : Ratio :=
  fun c => match c with
  | Component.Bleach => 3 * (original_ratio Component.Bleach)
  | Component.Detergent => (1/2) * (original_ratio Component.Detergent)
  | Component.Water => original_ratio Component.Water

def water_volume : ℚ := 300

theorem detergent_volume_in_new_solution :
  (new_ratio Component.Detergent / new_ratio Component.Water) * water_volume = 60 := by
  sorry

end detergent_volume_in_new_solution_l2164_216479


namespace probability_of_specific_draw_l2164_216402

-- Define the number of each type of clothing
def num_hats : ℕ := 3
def num_shirts : ℕ := 4
def num_shorts : ℕ := 5
def num_socks : ℕ := 6

-- Define the total number of articles
def total_articles : ℕ := num_hats + num_shirts + num_shorts + num_socks

-- Define the number of articles to be drawn
def draw_count : ℕ := 4

-- Theorem statement
theorem probability_of_specific_draw :
  (num_hats.choose 1 * num_shirts.choose 1 * num_shorts.choose 1 * num_socks.choose 1) /
  (total_articles.choose draw_count) = 2 / 17 := by
  sorry

end probability_of_specific_draw_l2164_216402


namespace fertilizer_mixture_problem_l2164_216468

/-- Given two fertilizer solutions, one with unknown percentage P and another with 53%,
    mixed to form 42 liters of 63% solution, where 20 liters of the first solution were used,
    prove that the percentage of fertilizer in the first solution is 74%. -/
theorem fertilizer_mixture_problem (P : ℝ) : 
  (20 * P / 100 + 22 * 53 / 100 = 42 * 63 / 100) → P = 74 := by
  sorry

end fertilizer_mixture_problem_l2164_216468


namespace dividend_calculation_l2164_216415

theorem dividend_calculation (remainder quotient divisor dividend : ℕ) : 
  remainder = 8 →
  divisor = 3 * quotient →
  divisor = 3 * remainder + 3 →
  dividend = divisor * quotient + remainder →
  dividend = 251 := by
sorry

end dividend_calculation_l2164_216415


namespace task_completion_time_l2164_216424

theorem task_completion_time 
  (time_A : ℝ) (time_B : ℝ) (time_C : ℝ) 
  (rest_A : ℝ) (rest_B : ℝ) :
  time_A = 10 →
  time_B = 15 →
  time_C = 15 →
  rest_A = 1 →
  rest_B = 2 →
  (1 - rest_B / time_B - (1 / time_A + 1 / time_B) * rest_A) / 
  (1 / time_A + 1 / time_B + 1 / time_C) + rest_A + rest_B = 29/7 :=
by sorry

end task_completion_time_l2164_216424


namespace susan_correct_percentage_l2164_216448

theorem susan_correct_percentage (y : ℝ) (h : y ≠ 0) :
  let total_questions : ℝ := 8 * y
  let unattempted_questions : ℝ := 2 * y + 3
  let correct_questions : ℝ := total_questions - unattempted_questions
  let percentage_correct : ℝ := (correct_questions / total_questions) * 100
  percentage_correct = 75 * (2 * y - 1) / y :=
by sorry

end susan_correct_percentage_l2164_216448


namespace gambler_final_amount_l2164_216445

def gamble (initial : ℚ) (rounds : ℕ) (wins : ℕ) (losses : ℕ) : ℚ :=
  let bet_fraction : ℚ := 1/3
  let win_multiplier : ℚ := 2
  let loss_multiplier : ℚ := 1
  sorry

theorem gambler_final_amount :
  let initial_amount : ℚ := 100
  let total_rounds : ℕ := 4
  let wins : ℕ := 2
  let losses : ℕ := 2
  gamble initial_amount total_rounds wins losses = 8000/81 := by sorry

end gambler_final_amount_l2164_216445


namespace rationalize_denominator_l2164_216485

theorem rationalize_denominator :
  ∃ (A B C D E F : ℚ),
    (1 : ℝ) / (Real.sqrt 2 + Real.sqrt 5 + Real.sqrt 7) =
    (A * Real.sqrt 2 + B * Real.sqrt 5 + C * Real.sqrt 7 + D * Real.sqrt E) / F ∧
    A = 5 ∧ B = 4 ∧ C = -1 ∧ D = 1 ∧ E = 70 ∧ F = 20 ∧ F > 0 :=
by sorry

end rationalize_denominator_l2164_216485


namespace negation_of_zero_product_l2164_216487

theorem negation_of_zero_product (a b : ℝ) :
  ¬(a * b = 0 → a = 0 ∨ b = 0) ↔ (a * b ≠ 0 → a ≠ 0 ∧ b ≠ 0) := by sorry

end negation_of_zero_product_l2164_216487


namespace triangle_area_from_circumradius_side_angle_l2164_216492

/-- The area of a triangle given its circumradius, one side, and one angle. -/
theorem triangle_area_from_circumradius_side_angle 
  (R a β : ℝ) (h_R : R > 0) (h_a : a > 0) (h_β : 0 < β ∧ β < π) : 
  ∃ (t : ℝ), t = (a^2 * Real.sin (2*β) / 4) + (a * Real.sin β^2 / 2) * Real.sqrt (4*R^2 - a^2) :=
by sorry

end triangle_area_from_circumradius_side_angle_l2164_216492


namespace arithmetic_geometric_comparison_l2164_216446

/-- An arithmetic sequence with positive terms and positive common difference -/
def ArithmeticSequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  d > 0 ∧ ∀ n, a n > 0 ∧ a (n + 1) = a n + d

/-- A geometric sequence with positive terms -/
def GeometricSequence (b : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 1 ∧ ∀ n, b n > 0 ∧ b (n + 1) = b n * q

theorem arithmetic_geometric_comparison
  (a b : ℕ → ℝ) (d : ℝ)
  (h_arith : ArithmeticSequence a d)
  (h_geom : GeometricSequence b)
  (h_equal_1 : a 1 = b 1)
  (h_equal_2 : a 2 = b 2) :
  ∀ n ≥ 3, a n < b n :=
sorry

end arithmetic_geometric_comparison_l2164_216446


namespace xiaogang_shooting_probability_l2164_216498

theorem xiaogang_shooting_probability (total_shots : ℕ) (successful_shots : ℕ) 
  (h1 : total_shots = 50) 
  (h2 : successful_shots = 38) : 
  (successful_shots : ℚ) / (total_shots : ℚ) = 0.76 := by
  sorry

end xiaogang_shooting_probability_l2164_216498


namespace power_equality_l2164_216401

theorem power_equality : (32 : ℕ)^4 * 4^5 = 2^30 := by sorry

end power_equality_l2164_216401


namespace count_scalene_triangles_l2164_216440

def is_valid_scalene_triangle (a b c : ℕ) : Prop :=
  a < b ∧ b < c ∧ 
  a + b + c < 15 ∧
  a + b > c ∧ a + c > b ∧ b + c > a

theorem count_scalene_triangles : 
  ∃ (S : Finset (ℕ × ℕ × ℕ)), 
    S.card = 6 ∧ 
    (∀ (t : ℕ × ℕ × ℕ), t ∈ S ↔ is_valid_scalene_triangle t.1 t.2.1 t.2.2) :=
sorry

end count_scalene_triangles_l2164_216440


namespace average_speed_last_segment_l2164_216465

theorem average_speed_last_segment (total_distance : ℝ) (total_time : ℝ) 
  (speed_first : ℝ) (speed_second : ℝ) :
  total_distance = 108 ∧ 
  total_time = 1.5 ∧ 
  speed_first = 70 ∧ 
  speed_second = 60 → 
  ∃ speed_last : ℝ, 
    speed_last = 86 ∧ 
    (speed_first + speed_second + speed_last) / 3 = total_distance / total_time :=
by sorry

end average_speed_last_segment_l2164_216465


namespace triangle_angle_sine_inequality_l2164_216478

theorem triangle_angle_sine_inequality (α β γ : Real) 
  (h_triangle : α + β + γ = π) 
  (h_positive : 0 < α ∧ 0 < β ∧ 0 < γ) : 
  Real.sin (α/2 + β) + Real.sin (β/2 + γ) + Real.sin (γ/2 + α) > 
  Real.sin α + Real.sin β + Real.sin γ := by
  sorry

end triangle_angle_sine_inequality_l2164_216478


namespace dealer_gain_percent_l2164_216426

theorem dealer_gain_percent (list_price : ℝ) (list_price_pos : list_price > 0) :
  let purchase_price := (3/4) * list_price
  let selling_price := (3/2) * list_price
  let gain := selling_price - purchase_price
  let gain_percent := (gain / purchase_price) * 100
  gain_percent = 100 := by sorry

end dealer_gain_percent_l2164_216426


namespace thirtythree_by_thirtythree_black_count_l2164_216449

/-- Represents a checkerboard with alternating colors -/
structure Checkerboard where
  size : ℕ
  blackInCorners : Bool

/-- Counts the number of black squares on a checkerboard -/
def countBlackSquares (board : Checkerboard) : ℕ :=
  sorry

/-- Theorem: A 33x33 checkerboard with black corners has 545 black squares -/
theorem thirtythree_by_thirtythree_black_count :
  ∀ (board : Checkerboard),
    board.size = 33 ∧ board.blackInCorners = true →
    countBlackSquares board = 545 :=
by sorry

end thirtythree_by_thirtythree_black_count_l2164_216449


namespace trigonometric_inequality_l2164_216457

theorem trigonometric_inequality (x y z : ℝ) 
  (hx : 0 < x ∧ x < π/2) 
  (hy : 0 < y ∧ y < π/2) 
  (hz : 0 < z ∧ z < π/2) : 
  π/2 + 2 * Real.sin x * Real.cos y + 2 * Real.sin y * Real.cos z > 
  Real.sin (2*x) + Real.sin (2*y) + Real.sin (2*z) := by
  sorry

end trigonometric_inequality_l2164_216457


namespace sphere_surface_area_l2164_216427

theorem sphere_surface_area (r : ℝ) (R : ℝ) :
  r > 0 → R > 0 →
  r^2 + 1^2 = R^2 →
  π * r^2 = π →
  4 * π * R^2 = 8 * π := by sorry

end sphere_surface_area_l2164_216427


namespace quadratic_perfect_square_l2164_216452

theorem quadratic_perfect_square (x : ℝ) : ∃ (a : ℝ), x^2 - 20*x + 100 = (x + a)^2 := by
  sorry

end quadratic_perfect_square_l2164_216452


namespace chocolate_bar_count_l2164_216407

/-- Represents the number of chocolate bars in a crate -/
def chocolate_bars_in_crate (large_boxes : ℕ) (small_boxes_per_large : ℕ) (bars_per_small : ℕ) : ℕ :=
  large_boxes * small_boxes_per_large * bars_per_small

/-- Proves that the total number of chocolate bars in the crate is 116,640 -/
theorem chocolate_bar_count :
  chocolate_bars_in_crate 45 36 72 = 116640 := by
  sorry

#eval chocolate_bars_in_crate 45 36 72

end chocolate_bar_count_l2164_216407


namespace system_solution_l2164_216403

theorem system_solution (x y k : ℝ) : 
  x + 2*y = 2*k ∧ 
  2*x + y = 4*k ∧ 
  x + y = 4 → 
  k = 2 := by sorry

end system_solution_l2164_216403


namespace max_covered_squares_l2164_216423

def checkerboard_width : ℕ := 15
def checkerboard_height : ℕ := 36
def tile_side_1 : ℕ := 7
def tile_side_2 : ℕ := 5

theorem max_covered_squares :
  ∃ (n m : ℕ),
    n * (tile_side_1 ^ 2) + m * (tile_side_2 ^ 2) = checkerboard_width * checkerboard_height ∧
    ∀ (k l : ℕ),
      k * (tile_side_1 ^ 2) + l * (tile_side_2 ^ 2) ≤ checkerboard_width * checkerboard_height →
      k * (tile_side_1 ^ 2) + l * (tile_side_2 ^ 2) ≤ n * (tile_side_1 ^ 2) + m * (tile_side_2 ^ 2) :=
by
  sorry

end max_covered_squares_l2164_216423


namespace dress_savings_theorem_l2164_216441

/-- Calculates the number of weeks needed to save for a dress -/
def weeks_to_save (dress_cost : ℕ) (initial_savings : ℕ) (weekly_allowance : ℕ) (weekly_spending : ℕ) : ℕ :=
  let additional_money_needed := dress_cost - initial_savings
  let weekly_savings := weekly_allowance - weekly_spending
  (additional_money_needed + weekly_savings - 1) / weekly_savings

theorem dress_savings_theorem (dress_cost : ℕ) (initial_savings : ℕ) (weekly_allowance : ℕ) (weekly_spending : ℕ)
  (h1 : dress_cost = 80)
  (h2 : initial_savings = 20)
  (h3 : weekly_allowance = 30)
  (h4 : weekly_spending = 10) :
  weeks_to_save dress_cost initial_savings weekly_allowance weekly_spending = 3 := by
  sorry

end dress_savings_theorem_l2164_216441


namespace snooker_ticket_difference_l2164_216442

theorem snooker_ticket_difference (vip_price gen_price : ℚ) 
  (total_tickets total_revenue : ℚ) (min_vip min_gen : ℕ) :
  vip_price = 40 →
  gen_price = 15 →
  total_tickets = 320 →
  total_revenue = 7500 →
  min_vip = 80 →
  min_gen = 100 →
  ∃ (vip_sold gen_sold : ℕ),
    vip_sold + gen_sold = total_tickets ∧
    vip_price * vip_sold + gen_price * gen_sold = total_revenue ∧
    vip_sold ≥ min_vip ∧
    gen_sold ≥ min_gen ∧
    gen_sold - vip_sold = 104 :=
by sorry

end snooker_ticket_difference_l2164_216442


namespace nested_fraction_equality_l2164_216466

theorem nested_fraction_equality : 2 + 1 / (2 + 1 / (2 + 2)) = 22 / 9 := by
  sorry

end nested_fraction_equality_l2164_216466


namespace smallest_c_inequality_l2164_216444

theorem smallest_c_inequality (c : ℝ) : 
  (∀ x y : ℝ, x ≥ 0 → y ≥ 0 → Real.sqrt (x * y) + c * |x^2 - y^2| ≥ (x + y) / 2) ↔ c ≥ (1/2 : ℝ) :=
by sorry

end smallest_c_inequality_l2164_216444


namespace circle_and_line_theorem_l2164_216453

-- Define the circle C
def circle_C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 3)^2 + (p.2 - 2)^2 = 5}

-- Define the parabola
def parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = -p.1^2 + 6*p.1 - 8}

-- Define the x-axis
def x_axis : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = 0}

-- Define the line y = x - 1
def center_line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = p.1 - 1}

-- Define the origin
def origin : ℝ × ℝ := (0, 0)

-- Define the lines l
def line_l1 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = 0}

def line_l2 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 12*p.1 - 5*p.2 = 0}

theorem circle_and_line_theorem :
  -- The center of circle_C lies on center_line
  (∃ c : ℝ × ℝ, c ∈ center_line ∧ ∀ p : ℝ × ℝ, p ∈ circle_C ↔ (p.1 - c.1)^2 + (p.2 - c.2)^2 = 5) ∧
  -- circle_C passes through the intersection of parabola and x_axis
  (∀ p : ℝ × ℝ, p ∈ parabola ∩ x_axis → p ∈ circle_C) ∧
  -- For any line through origin intersecting circle_C at M and N with ON = 2OM,
  -- the line is either line_l1 or line_l2
  (∀ l : Set (ℝ × ℝ), origin ∈ l →
    (∃ M N : ℝ × ℝ, M ∈ l ∩ circle_C ∧ N ∈ l ∩ circle_C ∧ 
      N.1 = 2*M.1 ∧ N.2 = 2*M.2) →
    l = line_l1 ∨ l = line_l2) :=
sorry

end circle_and_line_theorem_l2164_216453


namespace complex_product_real_l2164_216458

theorem complex_product_real (m : ℝ) :
  (Complex.I + 1) * (Complex.I * m + 1) ∈ Set.range Complex.ofReal → m = -1 := by
  sorry

end complex_product_real_l2164_216458


namespace quadratic_root_implies_v_value_l2164_216417

theorem quadratic_root_implies_v_value :
  ∀ v : ℝ,
  ((-15 - Real.sqrt 469) / 6 : ℝ) ∈ {x : ℝ | 3 * x^2 + 15 * x + v = 0} →
  v = -122/6 := by
sorry

end quadratic_root_implies_v_value_l2164_216417


namespace short_sleeve_shirts_count_l2164_216491

theorem short_sleeve_shirts_count :
  ∀ (total_shirts long_sleeve_shirts washed_shirts unwashed_shirts : ℕ),
    long_sleeve_shirts = 47 →
    washed_shirts = 20 →
    unwashed_shirts = 66 →
    total_shirts = washed_shirts + unwashed_shirts →
    total_shirts = (total_shirts - long_sleeve_shirts) + long_sleeve_shirts →
    (total_shirts - long_sleeve_shirts) = 39 :=
by sorry

end short_sleeve_shirts_count_l2164_216491


namespace sin_15_sin_75_equals_half_l2164_216418

theorem sin_15_sin_75_equals_half : 2 * Real.sin (15 * π / 180) * Real.sin (75 * π / 180) = 1 / 2 := by
  sorry

end sin_15_sin_75_equals_half_l2164_216418


namespace algae_coverage_on_day_17_algae_doubles_daily_full_coverage_on_day_20_l2164_216435

/-- Represents the coverage of algae on the lake on a given day -/
def algae_coverage (day : ℕ) : ℝ :=
  2^(day - 17)

/-- The day when the lake is completely covered with algae -/
def full_coverage_day : ℕ := 20

theorem algae_coverage_on_day_17 :
  algae_coverage 17 = 0.125 ∧ 1 - algae_coverage 17 = 0.875 := by sorry

theorem algae_doubles_daily (d : ℕ) (h : d < full_coverage_day) :
  algae_coverage (d + 1) = 2 * algae_coverage d := by sorry

theorem full_coverage_on_day_20 :
  algae_coverage full_coverage_day = 1 := by sorry

end algae_coverage_on_day_17_algae_doubles_daily_full_coverage_on_day_20_l2164_216435


namespace max_value_3a_plus_b_l2164_216406

theorem max_value_3a_plus_b (a b : ℝ) (h : 9 * a^2 + b^2 - 6 * a - 2 * b = 0) :
  ∀ x y : ℝ, 9 * x^2 + y^2 - 6 * x - 2 * y = 0 → 3 * x + y ≤ 3 * a + b → 3 * a + b ≤ 4 :=
by sorry

end max_value_3a_plus_b_l2164_216406


namespace triangle_side_length_l2164_216467

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2 →  -- acute triangle
  a = 4 →
  b = 5 →
  (1/2) * a * b * Real.sin C = 5 * Real.sqrt 3 →  -- area condition
  c = Real.sqrt 21 :=
by sorry

end triangle_side_length_l2164_216467


namespace parallel_line_length_l2164_216473

theorem parallel_line_length (base : ℝ) (h1 : base = 24) : 
  ∃ (parallel_line : ℝ), 
    parallel_line^2 / base^2 = 1/2 ∧ 
    parallel_line = 12 * Real.sqrt 2 := by
  sorry

end parallel_line_length_l2164_216473


namespace four_position_assignments_l2164_216474

def number_of_assignments (n : ℕ) : ℕ := n.factorial

theorem four_position_assignments :
  number_of_assignments 4 = 24 := by
  sorry

end four_position_assignments_l2164_216474


namespace minimum_jellybeans_l2164_216460

theorem minimum_jellybeans : ∃ n : ℕ,
  n ≥ 150 ∧
  n % 17 = 15 ∧
  (∀ m : ℕ, m ≥ 150 → m % 17 = 15 → n ≤ m) ∧
  n = 151 :=
by sorry

end minimum_jellybeans_l2164_216460


namespace positive_integer_solutions_inequality_l2164_216470

theorem positive_integer_solutions_inequality (x : ℕ+) : 
  (2 * x.val - 3 ≤ 5) ↔ x ∈ ({1, 2, 3, 4} : Set ℕ+) := by
  sorry

end positive_integer_solutions_inequality_l2164_216470


namespace jane_max_tickets_l2164_216489

/-- Calculates the maximum number of tickets that can be bought given a budget and pricing structure -/
def maxTickets (budget : ℕ) (normalPrice discountPrice : ℕ) (discountThreshold : ℕ) : ℕ :=
  let fullPriceTickets := min discountThreshold (budget / normalPrice)
  let remainingBudget := budget - fullPriceTickets * normalPrice
  fullPriceTickets + remainingBudget / discountPrice

/-- The maximum number of tickets Jane can buy is 11 -/
theorem jane_max_tickets :
  maxTickets 150 15 12 5 = 11 := by
  sorry

end jane_max_tickets_l2164_216489


namespace ball_bearing_sale_price_l2164_216463

/-- The sale price of ball bearings that satisfies the given conditions -/
def sale_price : ℝ := 0.75

theorem ball_bearing_sale_price :
  let num_machines : ℕ := 10
  let bearings_per_machine : ℕ := 30
  let normal_price : ℝ := 1
  let bulk_discount : ℝ := 0.2
  let total_savings : ℝ := 120
  
  let total_bearings : ℕ := num_machines * bearings_per_machine
  let normal_total_cost : ℝ := total_bearings * normal_price
  let sale_total_cost : ℝ := total_bearings * sale_price * (1 - bulk_discount)
  
  normal_total_cost - sale_total_cost = total_savings :=
by sorry

end ball_bearing_sale_price_l2164_216463


namespace sum_equality_implies_k_value_l2164_216431

/-- Given a real number k > 1 satisfying the infinite sum equation, prove k equals the specified value. -/
theorem sum_equality_implies_k_value (k : ℝ) 
  (h1 : k > 1) 
  (h2 : ∑' n, (7 * n - 3) / k^n = 2) : 
  k = 2 + 1.5 * Real.sqrt 2 := by
sorry

end sum_equality_implies_k_value_l2164_216431


namespace sqrt_50_plus_sqrt_32_l2164_216483

theorem sqrt_50_plus_sqrt_32 : Real.sqrt 50 + Real.sqrt 32 = 9 * Real.sqrt 2 := by
  sorry

end sqrt_50_plus_sqrt_32_l2164_216483


namespace vegetables_used_l2164_216495

def initial_beef : ℝ := 4
def unused_beef : ℝ := 1
def veg_to_beef_ratio : ℝ := 2

theorem vegetables_used : 
  let beef_used := initial_beef - unused_beef
  let vegetables_used := beef_used * veg_to_beef_ratio
  vegetables_used = 6 := by sorry

end vegetables_used_l2164_216495


namespace odd_divisor_of_3n_plus_1_l2164_216437

theorem odd_divisor_of_3n_plus_1 (n : ℕ) :
  n ≥ 1 ∧ Odd n ∧ n ∣ (3^n + 1) ↔ n = 1 := by
  sorry

end odd_divisor_of_3n_plus_1_l2164_216437


namespace equivalent_expression_proof_l2164_216411

theorem equivalent_expression_proof (n : ℕ) (hn : n > 1) :
  ∃ (p q : ℕ → ℕ),
    (∀ m : ℕ, m > 1 → 16^m + 4^m + 1 = (2^(p m) - 1) / (2^(q m) - 1)) ∧
    (∃ k : ℚ, ∀ m : ℕ, m > 1 → p m / q m = k) ∧
    p 2006 - q 2006 = 8024 :=
by
  sorry

end equivalent_expression_proof_l2164_216411

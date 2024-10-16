import Mathlib

namespace NUMINAMATH_CALUDE_zacks_marbles_l895_89586

/-- Zack's marble distribution problem -/
theorem zacks_marbles (initial_marbles : ℕ) (friends : ℕ) (marbles_per_friend : ℕ) 
  (h1 : initial_marbles = 65)
  (h2 : friends = 3)
  (h3 : marbles_per_friend = 20)
  (h4 : initial_marbles % friends ≠ 0) : 
  initial_marbles - friends * marbles_per_friend = 5 :=
by sorry

end NUMINAMATH_CALUDE_zacks_marbles_l895_89586


namespace NUMINAMATH_CALUDE_smallest_n_satisfying_condition_l895_89592

def trailing_zeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

def satisfies_condition (n : ℕ) : Prop :=
  n > 6 ∧ trailing_zeros (3 * n) = 4 * trailing_zeros n

theorem smallest_n_satisfying_condition :
  ∃ (n : ℕ), satisfies_condition n ∧ ∀ m, satisfies_condition m → n ≤ m :=
sorry

end NUMINAMATH_CALUDE_smallest_n_satisfying_condition_l895_89592


namespace NUMINAMATH_CALUDE_card_events_l895_89523

structure Card where
  suit : Fin 4
  number : Fin 10

def Deck : Finset Card := sorry

def isHeart (c : Card) : Prop := c.suit = 0
def isSpade (c : Card) : Prop := c.suit = 1
def isRed (c : Card) : Prop := c.suit = 0 ∨ c.suit = 2
def isBlack (c : Card) : Prop := c.suit = 1 ∨ c.suit = 3
def isMultipleOf5 (c : Card) : Prop := c.number % 5 = 0
def isGreaterThan9 (c : Card) : Prop := c.number = 9

def mutuallyExclusive (e1 e2 : Card → Prop) : Prop :=
  ∀ c : Card, ¬(e1 c ∧ e2 c)

def complementary (e1 e2 : Card → Prop) : Prop :=
  ∀ c : Card, e1 c ∨ e2 c

theorem card_events :
  (mutuallyExclusive isHeart isSpade ∧ ¬complementary isHeart isSpade) ∧
  (mutuallyExclusive isRed isBlack ∧ complementary isRed isBlack) ∧
  (¬mutuallyExclusive isMultipleOf5 isGreaterThan9 ∧ ¬complementary isMultipleOf5 isGreaterThan9) :=
by sorry

end NUMINAMATH_CALUDE_card_events_l895_89523


namespace NUMINAMATH_CALUDE_geometric_sequence_middle_term_l895_89562

theorem geometric_sequence_middle_term 
  (a b c : ℝ) 
  (pos_a : 0 < a) 
  (pos_b : 0 < b) 
  (pos_c : 0 < c) 
  (geom_seq : b^2 = a * c) 
  (def_a : a = 5 + 2 * Real.sqrt 3) 
  (def_c : c = 5 - 2 * Real.sqrt 3) : 
  b = Real.sqrt 13 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_middle_term_l895_89562


namespace NUMINAMATH_CALUDE_sum_largest_smallest_special_digits_l895_89576

def largest_three_digit (hundreds : Nat) (ones : Nat) : Nat :=
  hundreds * 100 + 90 + ones

def smallest_three_digit (hundreds : Nat) (ones : Nat) : Nat :=
  hundreds * 100 + ones

theorem sum_largest_smallest_special_digits :
  largest_three_digit 2 7 + smallest_three_digit 2 7 = 504 := by
  sorry

end NUMINAMATH_CALUDE_sum_largest_smallest_special_digits_l895_89576


namespace NUMINAMATH_CALUDE_vertex_of_quadratic_l895_89508

/-- The quadratic function f(x) = -2(x+1)^2-4 -/
def f (x : ℝ) : ℝ := -2 * (x + 1)^2 - 4

/-- The vertex of the quadratic function f -/
def vertex : ℝ × ℝ := (-1, -4)

/-- Theorem: The vertex of f(x) = -2(x+1)^2-4 is at (-1, -4) -/
theorem vertex_of_quadratic :
  (∀ x, f x ≤ f (vertex.1)) ∧ f (vertex.1) = vertex.2 := by
  sorry

end NUMINAMATH_CALUDE_vertex_of_quadratic_l895_89508


namespace NUMINAMATH_CALUDE_parcel_cost_correct_l895_89563

/-- The cost function for sending a parcel post package -/
def parcel_cost (P : ℕ) : ℕ :=
  10 + 3 * (P - 1)

/-- Theorem stating that the parcel_cost function correctly calculates the cost
    for a package of weight P pounds, where P is a positive integer -/
theorem parcel_cost_correct (P : ℕ) (h : P > 0) :
  parcel_cost P = 10 + 3 * (P - 1) ∧
  (P = 1 → parcel_cost P = 10) ∧
  (P > 1 → parcel_cost P = 10 + 3 * (P - 1)) :=
by sorry

end NUMINAMATH_CALUDE_parcel_cost_correct_l895_89563


namespace NUMINAMATH_CALUDE_lowest_price_per_component_l895_89505

/-- The lowest price per component that covers all costs for a computer manufacturer --/
theorem lowest_price_per_component 
  (cost_per_component : ℝ) 
  (shipping_cost_per_unit : ℝ) 
  (fixed_monthly_costs : ℝ) 
  (components_per_month : ℕ) 
  (h1 : cost_per_component = 80)
  (h2 : shipping_cost_per_unit = 7)
  (h3 : fixed_monthly_costs = 16500)
  (h4 : components_per_month = 150) : 
  ∃ (price : ℝ), price = 197 ∧ 
    price * (components_per_month : ℝ) = 
      (cost_per_component + shipping_cost_per_unit) * (components_per_month : ℝ) + fixed_monthly_costs :=
by sorry

end NUMINAMATH_CALUDE_lowest_price_per_component_l895_89505


namespace NUMINAMATH_CALUDE_october_price_reduction_november_profit_impossible_l895_89512

def initial_profit_per_box : ℝ := 50
def initial_monthly_sales : ℝ := 500
def sales_increase_per_dollar : ℝ := 20

def profit_function (x : ℝ) : ℝ :=
  (initial_profit_per_box - x) * (initial_monthly_sales + sales_increase_per_dollar * x)

theorem october_price_reduction :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
  profit_function x₁ = 28000 ∧
  profit_function x₂ = 28000 ∧
  (x₁ = 10 ∨ x₁ = 15) ∧
  (x₂ = 10 ∨ x₂ = 15) :=
sorry

theorem november_profit_impossible :
  ¬ ∃ x : ℝ, profit_function x = 30000 :=
sorry

end NUMINAMATH_CALUDE_october_price_reduction_november_profit_impossible_l895_89512


namespace NUMINAMATH_CALUDE_major_axis_length_eccentricity_l895_89533

/-- Definition of the ellipse E -/
def ellipse_E (x y : ℝ) : Prop := y^2 / 4 + x^2 / 3 = 1

/-- F₁ and F₂ are the foci of the ellipse E -/
axiom foci_on_ellipse : ∃ F₁ F₂ : ℝ × ℝ, ellipse_E F₁.1 F₁.2 ∧ ellipse_E F₂.1 F₂.2

/-- Point P lies on the ellipse E -/
axiom P_on_ellipse : ∃ P : ℝ × ℝ, ellipse_E P.1 P.2

/-- The length of the major axis of ellipse E is 4 -/
theorem major_axis_length : ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 
  (∀ x y : ℝ, ellipse_E x y ↔ (x/a)^2 + (y/b)^2 = 1) ∧ 
  max a b = 2 :=
sorry

/-- The eccentricity of ellipse E is 1/2 -/
theorem eccentricity : ∃ e : ℝ, e = 1/2 ∧
  ∃ c a : ℝ, c^2 = 4 - 3 ∧ a^2 = 3 ∧ e = c/a :=
sorry

end NUMINAMATH_CALUDE_major_axis_length_eccentricity_l895_89533


namespace NUMINAMATH_CALUDE_paper_clip_collection_l895_89578

theorem paper_clip_collection (num_boxes : ℕ) (clips_per_box : ℕ) 
  (h1 : num_boxes = 9) (h2 : clips_per_box = 9) : 
  num_boxes * clips_per_box = 81 := by
  sorry

end NUMINAMATH_CALUDE_paper_clip_collection_l895_89578


namespace NUMINAMATH_CALUDE_water_tank_capacity_l895_89502

theorem water_tank_capacity (w c : ℝ) (hw : w > 0) (hc : c > 0) : 
  w / c = 1 / 6 → (w + 4) / c = 1 / 3 → c = 12 := by
  sorry

end NUMINAMATH_CALUDE_water_tank_capacity_l895_89502


namespace NUMINAMATH_CALUDE_odd_square_minus_one_l895_89588

theorem odd_square_minus_one (n : ℕ) : (2*n + 1)^2 - 1 = 4*n*(n + 1) := by
  sorry

end NUMINAMATH_CALUDE_odd_square_minus_one_l895_89588


namespace NUMINAMATH_CALUDE_prob_two_bags_theorem_l895_89569

/-- Represents a bag of colored balls -/
structure Bag where
  red : Nat
  white : Nat

/-- Calculates the probability of drawing at least one white ball from two bags -/
def prob_at_least_one_white (bagA bagB : Bag) : Rat :=
  let total_outcomes := bagA.red * bagB.red + bagA.red * bagB.white + bagA.white * bagB.red + bagA.white * bagB.white
  let favorable_outcomes := bagA.white * bagB.red + bagA.red * bagB.white + bagA.white * bagB.white
  favorable_outcomes / total_outcomes

/-- The main theorem to prove -/
theorem prob_two_bags_theorem (bagA bagB : Bag) 
    (h1 : bagA.red = 3) (h2 : bagA.white = 2) 
    (h3 : bagB.red = 2) (h4 : bagB.white = 1) : 
    prob_at_least_one_white bagA bagB = 3/5 := by
  sorry

#eval prob_at_least_one_white ⟨3, 2⟩ ⟨2, 1⟩

end NUMINAMATH_CALUDE_prob_two_bags_theorem_l895_89569


namespace NUMINAMATH_CALUDE_min_additional_packs_needed_l895_89540

/-- The number of sticker packs in each basket -/
def packsPerBasket : ℕ := 7

/-- The current number of sticker packs Matilda has -/
def currentPacks : ℕ := 40

/-- The minimum number of additional packs needed -/
def minAdditionalPacks : ℕ := 2

/-- Theorem stating the minimum number of additional packs needed -/
theorem min_additional_packs_needed : 
  ∃ (totalPacks : ℕ), 
    totalPacks = currentPacks + minAdditionalPacks ∧ 
    totalPacks % packsPerBasket = 0 ∧
    ∀ (k : ℕ), k < minAdditionalPacks → 
      (currentPacks + k) % packsPerBasket ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_min_additional_packs_needed_l895_89540


namespace NUMINAMATH_CALUDE_circle_intersection_angle_equality_l895_89564

-- Define the types for points and circles
variable (Point Circle : Type)
[MetricSpace Point]

-- Define the intersection function
variable (intersect : Circle → Circle → Set Point)

-- Define the center of a circle
variable (center : Circle → Point)

-- Define the angle between three points
variable (angle : Point → Point → Point → ℝ)

-- State the theorem
theorem circle_intersection_angle_equality
  (c₁ c₂ c₃ : Circle)
  (P Q A B C D : Point)
  (h₁ : P ∈ intersect c₁ c₂)
  (h₂ : Q ∈ intersect c₁ c₂)
  (h₃ : center c₃ = P)
  (h₄ : A ∈ intersect c₁ c₃)
  (h₅ : B ∈ intersect c₁ c₃)
  (h₆ : C ∈ intersect c₂ c₃)
  (h₇ : D ∈ intersect c₂ c₃) :
  angle A Q D = angle B Q C :=
sorry

end NUMINAMATH_CALUDE_circle_intersection_angle_equality_l895_89564


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l895_89526

def A (a : ℝ) : Set ℝ := {4, 2, a^2}
def B (a : ℝ) : Set ℝ := {1, a}

theorem intersection_implies_a_value (a : ℝ) :
  A a ∩ B a = {1} → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l895_89526


namespace NUMINAMATH_CALUDE_system_solution_exists_l895_89531

theorem system_solution_exists (m : ℝ) :
  (∃ x y : ℝ, y = (m + 1) * x + 2 ∧ y = (3 * m - 2) * x + 5 ∧ y > 5) ↔ m ≠ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_exists_l895_89531


namespace NUMINAMATH_CALUDE_intersection_k_value_l895_89566

-- Define the two lines
def line1 (x y k : ℝ) : Prop := 3 * x + y = k
def line2 (x y : ℝ) : Prop := -1.2 * x + y = -20

-- Define the theorem
theorem intersection_k_value :
  ∃ (k : ℝ), ∃ (y : ℝ),
    line1 7 y k ∧ line2 7 y ∧ k = 9.4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_k_value_l895_89566


namespace NUMINAMATH_CALUDE_equation_solution_l895_89585

theorem equation_solution (x : ℝ) (h : x + 2 ≠ 0) :
  (x / (x + 2) + 1 = 1 / (x + 2)) ↔ (x = -1 / 2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l895_89585


namespace NUMINAMATH_CALUDE_percentage_difference_l895_89554

theorem percentage_difference (A B : ℝ) (h : A = B * (1 + 0.25)) :
  B = A * (1 - 0.2) := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l895_89554


namespace NUMINAMATH_CALUDE_sixth_term_of_geometric_sequence_l895_89506

/-- Given a geometric sequence with first term 3 and second term -1/2, 
    prove that its sixth term is -1/2592 -/
theorem sixth_term_of_geometric_sequence (a₁ a₂ : ℚ) (h₁ : a₁ = 3) (h₂ : a₂ = -1/2) :
  let r := a₂ / a₁
  let a_n (n : ℕ) := a₁ * r^(n - 1)
  a_n 6 = -1/2592 := by
sorry

end NUMINAMATH_CALUDE_sixth_term_of_geometric_sequence_l895_89506


namespace NUMINAMATH_CALUDE_only_three_solutions_l895_89561

/-- Represents a solution to the equation AB = B^V -/
structure Solution :=
  (a b v : Nat)
  (h1 : a ≠ b ∧ a ≠ v ∧ b ≠ v)
  (h2 : a > 0 ∧ a < 10 ∧ b > 0 ∧ b < 10 ∧ v > 0 ∧ v < 10)
  (h3 : 10 * a + b = b^v)

/-- The set of all valid solutions -/
def allSolutions : Set Solution := {s | s.a > 0 ∧ s.b > 0 ∧ s.v > 0}

/-- The theorem stating that there are only three solutions -/
theorem only_three_solutions :
  allSolutions = {
    ⟨3, 2, 5, sorry, sorry, sorry⟩,
    ⟨3, 6, 2, sorry, sorry, sorry⟩,
    ⟨6, 4, 3, sorry, sorry, sorry⟩
  } := by sorry

end NUMINAMATH_CALUDE_only_three_solutions_l895_89561


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l895_89596

theorem algebraic_expression_value (x : ℝ) : 
  3 * x^2 - 2 * x - 1 = 2 → -9 * x^2 + 6 * x - 1 = -10 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l895_89596


namespace NUMINAMATH_CALUDE_smallest_positive_integer_congruence_l895_89535

theorem smallest_positive_integer_congruence :
  ∃ x : ℕ+, (42 * x.val + 9) % 15 = 3 ∧
  ∀ y : ℕ+, (42 * y.val + 9) % 15 = 3 → x ≤ y ∧
  x = 2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_congruence_l895_89535


namespace NUMINAMATH_CALUDE_susan_average_speed_l895_89582

/-- Calculates the average speed of a trip with four segments -/
def average_speed (d1 d2 d3 d4 v1 v2 v3 v4 : ℚ) : ℚ :=
  let total_distance := d1 + d2 + d3 + d4
  let total_time := d1 / v1 + d2 / v2 + d3 / v3 + d4 / v4
  total_distance / total_time

/-- Theorem stating that the average speed for Susan's trip is 480/19 mph -/
theorem susan_average_speed :
  average_speed 40 40 60 20 30 15 45 20 = 480 / 19 := by
  sorry

end NUMINAMATH_CALUDE_susan_average_speed_l895_89582


namespace NUMINAMATH_CALUDE_no_x_satisfies_conditions_l895_89542

theorem no_x_satisfies_conditions : ¬ ∃ x : ℝ, 
  400 ≤ x ∧ x ≤ 600 ∧ 
  Int.floor (Real.sqrt x) = 23 ∧ 
  Int.floor (Real.sqrt (100 * x)) = 480 := by
sorry

end NUMINAMATH_CALUDE_no_x_satisfies_conditions_l895_89542


namespace NUMINAMATH_CALUDE_pythagorean_triple_divisibility_l895_89572

theorem pythagorean_triple_divisibility (a b c : ℕ) (h : a^2 + b^2 = c^2) :
  (3 ∣ a ∨ 3 ∣ b) ∧ (4 ∣ a ∨ 4 ∣ b) ∧ (5 ∣ a ∨ 5 ∣ b ∨ 5 ∣ c) := by
  sorry

end NUMINAMATH_CALUDE_pythagorean_triple_divisibility_l895_89572


namespace NUMINAMATH_CALUDE_intersection_of_A_and_complement_of_B_l895_89501

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | 2^x * (x - 2) < 1}

def B : Set ℝ := {x | ∃ y, y = Real.log (1 - x)}

theorem intersection_of_A_and_complement_of_B :
  A ∩ (Set.univ \ B) = {x : ℝ | 1 ≤ x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_complement_of_B_l895_89501


namespace NUMINAMATH_CALUDE_symmetric_points_coordinates_l895_89503

/-- Two points are symmetric about the x-axis if their x-coordinates are equal and their y-coordinates are opposite in sign -/
def symmetric_about_x_axis (p q : ℝ × ℝ) : Prop :=
  p.1 = q.1 ∧ p.2 = -q.2

/-- Given two points P(a,1) and Q(-4,b) that are symmetric about the x-axis, prove that a = -4 and b = -1 -/
theorem symmetric_points_coordinates (a b : ℝ) 
  (h : symmetric_about_x_axis (a, 1) (-4, b)) : 
  a = -4 ∧ b = -1 := by
  sorry


end NUMINAMATH_CALUDE_symmetric_points_coordinates_l895_89503


namespace NUMINAMATH_CALUDE_reciprocal_expression_l895_89558

theorem reciprocal_expression (m n : ℝ) (h : m * n = 1) :
  (2 * m - 2 / n) * (1 / m + n) = 0 := by sorry

end NUMINAMATH_CALUDE_reciprocal_expression_l895_89558


namespace NUMINAMATH_CALUDE_pokemon_card_ratio_l895_89551

theorem pokemon_card_ratio : 
  ∀ (jenny orlando richard : ℕ),
  jenny = 6 →
  orlando = jenny + 2 →
  ∃ k : ℕ, richard = k * orlando →
  jenny + orlando + richard = 38 →
  richard / orlando = 3 :=
by sorry

end NUMINAMATH_CALUDE_pokemon_card_ratio_l895_89551


namespace NUMINAMATH_CALUDE_acute_angle_range_l895_89543

/-- Given two vectors a and b in ℝ², prove that the angle between them is acute
    if and only if x is in the specified range. -/
theorem acute_angle_range (x : ℝ) :
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![x, 4]
  (∀ i, a i * b i > 0) ↔ x ∈ Set.Ioo (-8) 2 ∪ Set.Ioi 2 := by
  sorry

end NUMINAMATH_CALUDE_acute_angle_range_l895_89543


namespace NUMINAMATH_CALUDE_library_book_count_l895_89521

/-- Calculates the final number of books in a library given initial count and changes. -/
def finalBookCount (initial : ℕ) (takenTuesday : ℕ) (broughtThursday : ℕ) (takenFriday : ℕ) : ℕ :=
  initial - takenTuesday + broughtThursday - takenFriday

/-- Theorem stating that given the specific book counts and changes, the final count is 29. -/
theorem library_book_count :
  finalBookCount 235 227 56 35 = 29 := by
  sorry

end NUMINAMATH_CALUDE_library_book_count_l895_89521


namespace NUMINAMATH_CALUDE_lowest_number_three_probability_l895_89547

-- Define a six-sided die
def die := Fin 6

-- Define the probability of rolling at least 3 on a single die
def prob_at_least_3 : ℚ := 4 / 6

-- Define the probability of rolling at least 4 on a single die
def prob_at_least_4 : ℚ := 3 / 6

-- Define the number of dice rolled
def num_dice : ℕ := 4

-- Theorem statement
theorem lowest_number_three_probability :
  (prob_at_least_3 ^ num_dice - prob_at_least_4 ^ num_dice) = 175 / 1296 :=
by sorry

end NUMINAMATH_CALUDE_lowest_number_three_probability_l895_89547


namespace NUMINAMATH_CALUDE_expression_simplification_l895_89539

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 2) :
  (((2 * x - 1) / (x + 1) - x + 1) / ((x - 2) / (x^2 + 2*x + 1))) = -2 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l895_89539


namespace NUMINAMATH_CALUDE_hypotenuse_division_l895_89553

/-- A right triangle with one acute angle of 30° and hypotenuse of length 8 -/
structure RightTriangle30 where
  /-- The length of the hypotenuse -/
  hypotenuse : ℝ
  /-- The hypotenuse is 8 -/
  hyp_eq_8 : hypotenuse = 8
  /-- One acute angle is 30° -/
  acute_angle : ℝ
  acute_angle_eq_30 : acute_angle = 30

/-- The altitude from the right angle vertex to the hypotenuse -/
def altitude (t : RightTriangle30) : ℝ := sorry

/-- The shorter segment of the hypotenuse divided by the altitude -/
def short_segment (t : RightTriangle30) : ℝ := sorry

/-- The longer segment of the hypotenuse divided by the altitude -/
def long_segment (t : RightTriangle30) : ℝ := sorry

/-- Theorem stating that the altitude divides the hypotenuse into segments of length 4 and 6 -/
theorem hypotenuse_division (t : RightTriangle30) : 
  short_segment t = 4 ∧ long_segment t = 6 :=
sorry

end NUMINAMATH_CALUDE_hypotenuse_division_l895_89553


namespace NUMINAMATH_CALUDE_circle_center_polar_coordinates_l895_89536

-- Define the polar equation of the circle
def circle_equation (ρ θ : Real) : Prop := ρ = 2 * Real.sin θ ∧ 0 ≤ θ ∧ θ < 2 * Real.pi

-- Define the center in polar coordinates
def is_center (ρ θ : Real) : Prop := 
  (ρ = 1 ∧ θ = Real.pi / 2) ∨ (ρ = 1 ∧ θ = 3 * Real.pi / 2)

-- Theorem statement
theorem circle_center_polar_coordinates :
  ∀ ρ θ : Real, circle_equation ρ θ → 
  ∃ ρ_c θ_c : Real, is_center ρ_c θ_c ∧ 
  (ρ - ρ_c * Real.cos (θ - θ_c))^2 + (ρ * Real.sin θ - ρ_c * Real.sin θ_c)^2 = ρ_c^2 :=
sorry

end NUMINAMATH_CALUDE_circle_center_polar_coordinates_l895_89536


namespace NUMINAMATH_CALUDE_sum_of_generated_numbers_eq_5994_l895_89599

/-- The sum of all three-digit natural numbers created using digits 1, 2, and 3 -/
def sum_three_digit_numbers : ℕ := 5994

/-- The set of digits that can be used -/
def valid_digits : Finset ℕ := {1, 2, 3}

/-- A function to generate all possible three-digit numbers using the valid digits -/
def generate_numbers : Finset ℕ := sorry

/-- Theorem stating that the sum of all generated numbers equals sum_three_digit_numbers -/
theorem sum_of_generated_numbers_eq_5994 : 
  (generate_numbers.sum id) = sum_three_digit_numbers := by sorry

end NUMINAMATH_CALUDE_sum_of_generated_numbers_eq_5994_l895_89599


namespace NUMINAMATH_CALUDE_rectangle_width_l895_89516

theorem rectangle_width (width : ℝ) (h1 : width > 0) : 
  (2 * width) * width = 50 → width = 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_width_l895_89516


namespace NUMINAMATH_CALUDE_concert_ticket_cost_l895_89580

theorem concert_ticket_cost (current_amount : ℕ) (amount_needed : ℕ) (num_tickets : ℕ) : 
  current_amount = 189 →
  amount_needed = 171 →
  num_tickets = 4 →
  (current_amount + amount_needed) / num_tickets = 90 := by
sorry

end NUMINAMATH_CALUDE_concert_ticket_cost_l895_89580


namespace NUMINAMATH_CALUDE_all_conditions_possible_l895_89511

/-- Two circles with centers A and B, radii a and b respectively, where a > b -/
structure TwoCircles where
  a : ℝ
  b : ℝ
  AB : ℝ
  h_positive : 0 < b
  h_a_gt_b : a > b

/-- All four conditions can be satisfied for some configuration of two circles -/
theorem all_conditions_possible (c : TwoCircles) : 
  (∃ c1 : TwoCircles, c1.a - c1.b < c1.AB) ∧ 
  (∃ c2 : TwoCircles, c2.a + c2.b = c2.AB) ∧ 
  (∃ c3 : TwoCircles, c3.a + c3.b < c3.AB) ∧ 
  (∃ c4 : TwoCircles, c4.a - c4.b = c4.AB) :=
sorry

end NUMINAMATH_CALUDE_all_conditions_possible_l895_89511


namespace NUMINAMATH_CALUDE_circle_sequence_periodic_l895_89518

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a triangle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the sequence of circles
def circleSequence (ABC : Triangle) : Fin 7 → Circle
  | 1 => sorry  -- S₁ inscribed in angle A of ABC
  | 2 => sorry  -- S₂ inscribed in triangle formed by tangent from C to S₁
  | 3 => sorry  -- S₃ inscribed in triangle formed by tangent from A to S₂
  | 4 => sorry  -- S₄
  | 5 => sorry  -- S₅
  | 6 => sorry  -- S₆
  | 7 => sorry  -- S₇

-- Theorem statement
theorem circle_sequence_periodic (ABC : Triangle) :
  circleSequence ABC 7 = circleSequence ABC 1 := by
  sorry

end NUMINAMATH_CALUDE_circle_sequence_periodic_l895_89518


namespace NUMINAMATH_CALUDE_optimal_truck_loading_l895_89595

theorem optimal_truck_loading (total_load : ℕ) (large_capacity : ℕ) (small_capacity : ℕ)
  (h_total : total_load = 134)
  (h_large : large_capacity = 15)
  (h_small : small_capacity = 7) :
  ∃ (large_count small_count : ℕ),
    large_count * large_capacity + small_count * small_capacity = total_load ∧
    large_count = 8 ∧
    small_count = 2 ∧
    ∀ (l s : ℕ), l * large_capacity + s * small_capacity = total_load →
      l + s ≥ large_count + small_count :=
by sorry

end NUMINAMATH_CALUDE_optimal_truck_loading_l895_89595


namespace NUMINAMATH_CALUDE_marks_age_in_five_years_l895_89597

theorem marks_age_in_five_years :
  ∀ (amy_age mark_age : ℕ),
    amy_age = 15 →
    mark_age = amy_age + 7 →
    mark_age + 5 = 27 :=
by sorry

end NUMINAMATH_CALUDE_marks_age_in_five_years_l895_89597


namespace NUMINAMATH_CALUDE_digit_ratio_l895_89517

/-- Given a 3-digit integer x with hundreds digit a, tens digit b, and units digit c,
    where a > 0 and the difference between the two greatest possible values of x is 241,
    prove that the ratio of b to a is 5:7. -/
theorem digit_ratio (x a b c : ℕ) : 
  (100 ≤ x) ∧ (x < 1000) ∧  -- x is a 3-digit integer
  (x = 100 * a + 10 * b + c) ∧  -- x is composed of digits a, b, c
  (a > 0) ∧  -- a is positive
  (999 - x = 241) →  -- difference between greatest possible value and x is 241
  (b : ℚ) / a = 5 / 7 := by
sorry

end NUMINAMATH_CALUDE_digit_ratio_l895_89517


namespace NUMINAMATH_CALUDE_three_numbers_sum_l895_89537

theorem three_numbers_sum (a b c : ℝ) : 
  a ≤ b ∧ b ≤ c →
  (a + b + c) / 3 = a + 8 →
  (a + b + c) / 3 = c - 9 →
  c - a = 26 →
  a + b + c = 81 := by
sorry

end NUMINAMATH_CALUDE_three_numbers_sum_l895_89537


namespace NUMINAMATH_CALUDE_horner_rule_V₂_l895_89515

def f (x : ℝ) : ℝ := 2*x^6 + 3*x^5 + 5*x^3 + 6*x^2 + 7*x + 8

def V₂ (x : ℝ) : ℝ := 2

def V₁ (x : ℝ) : ℝ := V₂ x * x + 3

def V₂_final (x : ℝ) : ℝ := V₁ x * x + 0

theorem horner_rule_V₂ : V₂_final 2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_horner_rule_V₂_l895_89515


namespace NUMINAMATH_CALUDE_point_C_satisfies_condition_l895_89556

/-- Given points A(-2, 1) and B(1, 4) in the plane, prove that C(-1, 2) satisfies AC = (1/2)CB -/
theorem point_C_satisfies_condition :
  let A : ℝ × ℝ := (-2, 1)
  let B : ℝ × ℝ := (1, 4)
  let C : ℝ × ℝ := (-1, 2)
  (C.1 - A.1, C.2 - A.2) = (1/2 : ℝ) • (B.1 - C.1, B.2 - C.2) := by
  sorry

#check point_C_satisfies_condition

end NUMINAMATH_CALUDE_point_C_satisfies_condition_l895_89556


namespace NUMINAMATH_CALUDE_B_initial_investment_correct_l895_89552

/-- Represents the initial investment of B in rupees -/
def B_initial_investment : ℝ := 4866.67

/-- Represents A's initial investment in rupees -/
def A_initial_investment : ℝ := 2000

/-- Represents the amount A withdraws after 8 months in rupees -/
def A_withdrawal : ℝ := 1000

/-- Represents the amount B advances after 8 months in rupees -/
def B_advance : ℝ := 1000

/-- Represents the total profit at the end of the year in rupees -/
def total_profit : ℝ := 630

/-- Represents A's share of the profit in rupees -/
def A_profit_share : ℝ := 175

/-- Represents the number of months in a year -/
def months_in_year : ℕ := 12

/-- Represents the number of months before A withdraws and B advances -/
def months_before_change : ℕ := 8

theorem B_initial_investment_correct :
  B_initial_investment * months_before_change +
  (B_initial_investment + B_advance) * (months_in_year - months_before_change) =
  (total_profit - A_profit_share) / A_profit_share *
  (A_initial_investment * months_in_year) :=
by sorry

end NUMINAMATH_CALUDE_B_initial_investment_correct_l895_89552


namespace NUMINAMATH_CALUDE_paint_cans_theorem_l895_89593

/-- The number of rooms that can be painted with one can of paint -/
def rooms_per_can : ℚ :=
  (40 - 32) / 4

/-- The number of cans needed to paint 32 rooms -/
def cans_for_32_rooms : ℚ :=
  32 / rooms_per_can

theorem paint_cans_theorem :
  cans_for_32_rooms = 16 := by
  sorry

end NUMINAMATH_CALUDE_paint_cans_theorem_l895_89593


namespace NUMINAMATH_CALUDE_tripled_base_doubled_exponent_l895_89598

theorem tripled_base_doubled_exponent 
  (c y : ℝ) (d : ℝ) (h_d : d ≠ 0) :
  let s := (3 * c) ^ (2 * d)
  s = c^d / y^d →
  y = 1 / (9 * c) := by
sorry

end NUMINAMATH_CALUDE_tripled_base_doubled_exponent_l895_89598


namespace NUMINAMATH_CALUDE_dinner_seating_arrangements_l895_89568

/-- The number of ways to choose and seat people at a circular table. -/
def circular_seating_arrangements (total_people : ℕ) (seats : ℕ) : ℕ :=
  total_people * (seats - 1).factorial

/-- The problem statement -/
theorem dinner_seating_arrangements :
  circular_seating_arrangements 8 7 = 5760 := by
  sorry

end NUMINAMATH_CALUDE_dinner_seating_arrangements_l895_89568


namespace NUMINAMATH_CALUDE_base_prime_repr_225_l895_89524

/-- Base prime representation of a natural number -/
def base_prime_repr (n : ℕ) : List ℕ :=
  sorry

/-- Prime factorization of a natural number -/
def prime_factorization (n : ℕ) : List (ℕ × ℕ) :=
  sorry

/-- Theorem: The base prime representation of 225 is [2, 2, 0] -/
theorem base_prime_repr_225 : 
  base_prime_repr 225 = [2, 2, 0] :=
sorry

end NUMINAMATH_CALUDE_base_prime_repr_225_l895_89524


namespace NUMINAMATH_CALUDE_range_of_c_l895_89525

def p (c : ℝ) := ∀ x y : ℝ, x < y → c^x > c^y

def q (c : ℝ) := ∀ x : ℝ, x ∈ Set.Icc (1/2) 2 → x + 1/x > c

theorem range_of_c :
  ∀ c : ℝ, c > 0 →
  ((p c ∨ q c) ∧ ¬(p c ∧ q c)) →
  (c ∈ Set.Ioo 0 (1/2) ∪ Set.Ici 1) :=
sorry

end NUMINAMATH_CALUDE_range_of_c_l895_89525


namespace NUMINAMATH_CALUDE_cube_has_six_faces_l895_89584

/-- A cube is a three-dimensional solid object with six square faces -/
structure Cube where
  -- We don't need to define the specifics of a cube for this problem

/-- The number of faces of a cube -/
def num_faces (c : Cube) : ℕ := 6

/-- Theorem: A cube has 6 faces -/
theorem cube_has_six_faces (c : Cube) : num_faces c = 6 := by
  sorry

end NUMINAMATH_CALUDE_cube_has_six_faces_l895_89584


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l895_89557

/-- The equation of a line perpendicular to 2x+y-5=0 and passing through (2,3) is x-2y+4=0 -/
theorem perpendicular_line_equation : 
  ∀ (x y : ℝ), 
  (∃ (m : ℝ), (2 : ℝ) * x + y - 5 = 0 ↔ y = -2 * x + m) →
  (∃ (k : ℝ), k * (x - 2) + 3 = y ∧ k * 2 = -1) →
  x - 2 * y + 4 = 0 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l895_89557


namespace NUMINAMATH_CALUDE_recurring_decimal_division_l895_89579

theorem recurring_decimal_division :
  let a : ℚ := 36 / 99
  let b : ℚ := 12 / 99
  a / b = 3 := by sorry

end NUMINAMATH_CALUDE_recurring_decimal_division_l895_89579


namespace NUMINAMATH_CALUDE_total_distance_walked_l895_89549

def distance_to_water_fountain : ℕ := 30
def distance_to_staff_lounge : ℕ := 45
def trips_to_water_fountain : ℕ := 4
def trips_to_staff_lounge : ℕ := 3

theorem total_distance_walked :
  2 * (distance_to_water_fountain * trips_to_water_fountain +
       distance_to_staff_lounge * trips_to_staff_lounge) = 510 :=
by sorry

end NUMINAMATH_CALUDE_total_distance_walked_l895_89549


namespace NUMINAMATH_CALUDE_eggs_in_box_l895_89509

/-- The number of eggs initially in the box -/
def initial_eggs : ℝ := 47.0

/-- The number of eggs Harry adds to the box -/
def added_eggs : ℝ := 5.0

/-- The total number of eggs in the box after Harry adds eggs -/
def total_eggs : ℝ := initial_eggs + added_eggs

theorem eggs_in_box : total_eggs = 52.0 := by
  sorry

end NUMINAMATH_CALUDE_eggs_in_box_l895_89509


namespace NUMINAMATH_CALUDE_math_books_count_l895_89522

theorem math_books_count (total_books : ℕ) (math_price history_price total_price : ℕ) 
  (h1 : total_books = 80)
  (h2 : math_price = 4)
  (h3 : history_price = 5)
  (h4 : total_price = 390) :
  ∃ (math_books : ℕ), 
    math_books * math_price + (total_books - math_books) * history_price = total_price ∧ 
    math_books = 10 := by
  sorry

end NUMINAMATH_CALUDE_math_books_count_l895_89522


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l895_89591

def arithmeticSequence (a : ℕ → ℝ) := ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

theorem arithmetic_sequence_general_term
  (a : ℕ → ℝ)
  (h_arithmetic : arithmeticSequence a)
  (h_mean1 : (a 2 + a 6) / 2 = 5)
  (h_mean2 : (a 3 + a 7) / 2 = 7) :
  ∀ n : ℕ, a n = 2 * n - 3 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l895_89591


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l895_89560

theorem right_triangle_hypotenuse (a b : ℂ) (z₁ z₂ z₃ : ℂ) : 
  (z₁^3 + a*z₁ + b = 0) → 
  (z₂^3 + a*z₂ + b = 0) → 
  (z₃^3 + a*z₃ + b = 0) → 
  (Complex.abs z₁)^2 + (Complex.abs z₂)^2 + (Complex.abs z₃)^2 = 250 →
  ∃ (x y : ℝ), (x^2 + y^2 = (Complex.abs (z₁ - z₂))^2) ∧ 
                (x^2 = (Complex.abs (z₂ - z₃))^2 ∨ y^2 = (Complex.abs (z₂ - z₃))^2) →
  (Complex.abs (z₁ - z₂))^2 + (Complex.abs (z₂ - z₃))^2 + (Complex.abs (z₃ - z₁))^2 = 2 * ((5 * Real.sqrt 15)^2) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l895_89560


namespace NUMINAMATH_CALUDE_soccer_boys_percentage_l895_89550

theorem soccer_boys_percentage (total_students : ℕ) (boys : ℕ) (soccer_players : ℕ) (girls_not_playing : ℕ)
  (h1 : total_students = 420)
  (h2 : boys = 320)
  (h3 : soccer_players = 250)
  (h4 : girls_not_playing = 65) :
  (boys - (total_students - boys - girls_not_playing)) / soccer_players * 100 = 86 := by
  sorry

end NUMINAMATH_CALUDE_soccer_boys_percentage_l895_89550


namespace NUMINAMATH_CALUDE_binomial_coefficient_ratio_l895_89567

theorem binomial_coefficient_ratio (n k : ℕ) : 
  (Nat.choose n k : ℚ) / (Nat.choose n (k+1) : ℚ) = 1/2 ∧ 
  (Nat.choose n (k+1) : ℚ) / (Nat.choose n (k+2) : ℚ) = 2/3 → 
  n + k = 18 :=
by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_ratio_l895_89567


namespace NUMINAMATH_CALUDE_max_inscribed_sphere_volume_l895_89538

/-- The maximum volume of a sphere inscribed in a right triangular prism -/
theorem max_inscribed_sphere_volume (a b h : ℝ) (ha : 0 < a) (hb : 0 < b) (hh : 0 < h) :
  let r := min (a * b / (a + b + Real.sqrt (a^2 + b^2))) (h / 2)
  (4 / 3) * Real.pi * r^3 = (9 * Real.pi) / 2 :=
by sorry

end NUMINAMATH_CALUDE_max_inscribed_sphere_volume_l895_89538


namespace NUMINAMATH_CALUDE_factor_expression_l895_89581

theorem factor_expression (x : ℝ) : 5*x*(x-2) + 9*(x-2) - 4*(x-2) = 5*(x-2)*(x+1) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l895_89581


namespace NUMINAMATH_CALUDE_division_problem_l895_89528

theorem division_problem (dividend quotient divisor remainder n : ℕ) : 
  dividend = 86 →
  remainder = 6 →
  divisor = 5 * quotient →
  divisor = 3 * remainder + n →
  dividend = divisor * quotient + remainder →
  n = 2 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l895_89528


namespace NUMINAMATH_CALUDE_remainder_problem_l895_89500

theorem remainder_problem : (29 * 171997^2000) % 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l895_89500


namespace NUMINAMATH_CALUDE_line_intersection_y_axis_l895_89565

/-- A line passing through two points intersects the y-axis at a specific point -/
theorem line_intersection_y_axis (x₁ y₁ x₂ y₂ : ℝ) (hx : x₁ ≠ x₂) :
  let m := (y₂ - y₁) / (x₂ - x₁)
  let b := y₁ - m * x₁
  (x₁ = 2 ∧ y₁ = 9 ∧ x₂ = 4 ∧ y₂ = 17) →
  (0, m * 0 + b) = (0, 1) :=
sorry

end NUMINAMATH_CALUDE_line_intersection_y_axis_l895_89565


namespace NUMINAMATH_CALUDE_kelly_cheese_packages_l895_89589

-- Define the problem parameters
def days_per_week : ℕ := 5
def oldest_child_cheese_per_day : ℕ := 2
def youngest_child_cheese_per_day : ℕ := 1
def cheese_per_package : ℕ := 30
def weeks : ℕ := 4

-- Define the theorem
theorem kelly_cheese_packages :
  (days_per_week * (oldest_child_cheese_per_day + youngest_child_cheese_per_day) * weeks + cheese_per_package - 1) / cheese_per_package = 2 :=
by sorry

end NUMINAMATH_CALUDE_kelly_cheese_packages_l895_89589


namespace NUMINAMATH_CALUDE_plan1_more_cost_effective_l895_89507

/-- Represents the cost of a mobile phone plan based on talk time -/
def plan_cost (rental : ℝ) (rate : ℝ) (minutes : ℝ) : ℝ := rental + rate * minutes

/-- Theorem stating when Plan 1 is more cost-effective than Plan 2 -/
theorem plan1_more_cost_effective (minutes : ℝ) :
  minutes > 72 →
  plan_cost 36 0.1 minutes < plan_cost 0 0.6 minutes := by
  sorry

end NUMINAMATH_CALUDE_plan1_more_cost_effective_l895_89507


namespace NUMINAMATH_CALUDE_true_propositions_l895_89587

-- Define the propositions p and q
def p : Prop := ∀ x : ℝ, ∃ y : ℝ, y = Real.log (x^2)
def q : Prop := ∀ y : ℝ, y > 0 → ∃ x : ℝ, y = 3^x

-- Define the set of derived propositions
def derived_props : Set Prop := {p ∨ q, p ∧ q, ¬p, ¬q}

-- Define the set of true propositions
def true_props : Set Prop := {p ∨ q, ¬p}

-- Theorem statement
theorem true_propositions : 
  {prop ∈ derived_props | prop} = true_props := by sorry

end NUMINAMATH_CALUDE_true_propositions_l895_89587


namespace NUMINAMATH_CALUDE_wire_cutting_problem_l895_89577

theorem wire_cutting_problem (total_length : ℝ) (ratio : ℚ) (shorter_length : ℝ) : 
  total_length = 60 →
  ratio = 2 / 4 →
  shorter_length + ratio * shorter_length = total_length →
  shorter_length = 40 := by
sorry

end NUMINAMATH_CALUDE_wire_cutting_problem_l895_89577


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_m_value_l895_89544

/-- A trinomial ax^2 + bx + c is a perfect square if there exists a real number d such that ax^2 + bx + c = (dx + e)^2 for all x -/
def is_perfect_square_trinomial (a b c : ℝ) : Prop :=
  ∃ d e : ℝ, ∀ x : ℝ, a * x^2 + b * x + c = (d * x + e)^2

theorem perfect_square_trinomial_m_value :
  ∀ m : ℝ, is_perfect_square_trinomial 1 (-4) m → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_m_value_l895_89544


namespace NUMINAMATH_CALUDE_point_coordinates_wrt_origin_l895_89575

/-- The coordinates of a point (3, -2) with respect to the origin are (3, -2). -/
theorem point_coordinates_wrt_origin :
  let p : ℝ × ℝ := (3, -2)
  p = (3, -2) :=
by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_wrt_origin_l895_89575


namespace NUMINAMATH_CALUDE_parking_lot_search_time_l895_89559

/-- Calculates the time spent searching a parking lot given the layout and walking speed. -/
theorem parking_lot_search_time
  (section_g_rows : ℕ)
  (section_g_cars_per_row : ℕ)
  (section_h_rows : ℕ)
  (section_h_cars_per_row : ℕ)
  (cars_passed_per_minute : ℕ)
  (h_section_g_rows : section_g_rows = 15)
  (h_section_g_cars : section_g_cars_per_row = 10)
  (h_section_h_rows : section_h_rows = 20)
  (h_section_h_cars : section_h_cars_per_row = 9)
  (h_cars_passed : cars_passed_per_minute = 11)
  : (section_g_rows * section_g_cars_per_row + section_h_rows * section_h_cars_per_row) / cars_passed_per_minute = 30 := by
  sorry


end NUMINAMATH_CALUDE_parking_lot_search_time_l895_89559


namespace NUMINAMATH_CALUDE_square_sum_equals_90_l895_89574

theorem square_sum_equals_90 (x y : ℝ) (h1 : x + 3 * y = 6) (h2 : x * y = -9) :
  x^2 + 9 * y^2 = 90 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equals_90_l895_89574


namespace NUMINAMATH_CALUDE_small_box_dimension_l895_89573

/-- Given a large rectangular box and smaller boxes, proves the dimensions of the smaller boxes. -/
theorem small_box_dimension (large_length large_width large_height : ℕ)
                             (small_length small_height : ℕ)
                             (max_boxes : ℕ)
                             (h1 : large_length = 12)
                             (h2 : large_width = 14)
                             (h3 : large_height = 16)
                             (h4 : small_length = 3)
                             (h5 : small_height = 2)
                             (h6 : max_boxes = 64) :
  ∃ (small_width : ℕ), small_width = 7 ∧
    max_boxes * (small_length * small_width * small_height) = 
    large_length * large_width * large_height :=
by sorry

end NUMINAMATH_CALUDE_small_box_dimension_l895_89573


namespace NUMINAMATH_CALUDE_max_leftover_apples_l895_89529

theorem max_leftover_apples (n : ℕ) (h : n > 0) : 
  ∃ (m : ℕ), m > 0 ∧ m < n ∧ 
  ∀ (total : ℕ), total ≥ n * (total / n) + m → total / n = (total - m) / n :=
by
  sorry

end NUMINAMATH_CALUDE_max_leftover_apples_l895_89529


namespace NUMINAMATH_CALUDE_intersection_when_a_is_one_intersection_contains_one_integer_l895_89555

-- Define sets A and B
def A : Set ℝ := {x | x^2 + 2*x - 3 > 0}
def B (a : ℝ) : Set ℝ := {x | x^2 - 2*a*x - 1 ≤ 0 ∧ a > 0}

-- Theorem for part I
theorem intersection_when_a_is_one :
  A ∩ B 1 = {x | 1 < x ∧ x ≤ 1 + Real.sqrt 2} := by sorry

-- Theorem for part II
theorem intersection_contains_one_integer (a : ℝ) :
  (∃! (n : ℤ), (n : ℝ) ∈ A ∩ B a) ↔ 3/4 ≤ a ∧ a < 4/3 := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_is_one_intersection_contains_one_integer_l895_89555


namespace NUMINAMATH_CALUDE_x_squared_congruence_l895_89520

theorem x_squared_congruence (x : ℤ) 
  (h1 : 5 * x ≡ 10 [ZMOD 25]) 
  (h2 : 4 * x ≡ 20 [ZMOD 25]) : 
  x^2 ≡ 0 [ZMOD 25] := by
  sorry

end NUMINAMATH_CALUDE_x_squared_congruence_l895_89520


namespace NUMINAMATH_CALUDE_min_sum_xy_l895_89510

theorem min_sum_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x * y * (x - y)^2 = 1) : 
  x + y ≥ 2 := by
sorry

end NUMINAMATH_CALUDE_min_sum_xy_l895_89510


namespace NUMINAMATH_CALUDE_expression_change_l895_89532

theorem expression_change (x a : ℝ) (h : a > 0) :
  let f : ℝ → ℝ := λ y => y^2 - 5*y
  (f (x + a) - f x = 2*a*x + a^2 - 5*a) ∧ 
  (f (x - a) - f x = -2*a*x + a^2 + 5*a) :=
sorry

end NUMINAMATH_CALUDE_expression_change_l895_89532


namespace NUMINAMATH_CALUDE_square_pens_area_ratio_l895_89519

/-- Given four congruent square pens with side length s, prove that the ratio of their
    total area to the area of a single square pen formed by reusing the same amount
    of fencing is 1/4. -/
theorem square_pens_area_ratio (s : ℝ) (h : s > 0) : 
  (4 * s^2) / ((4 * s)^2) = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_square_pens_area_ratio_l895_89519


namespace NUMINAMATH_CALUDE_sqrt_21_minus_1_bounds_l895_89513

theorem sqrt_21_minus_1_bounds : 3 < Real.sqrt 21 - 1 ∧ Real.sqrt 21 - 1 < 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_21_minus_1_bounds_l895_89513


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l895_89545

theorem quadratic_two_distinct_roots (k : ℝ) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
  (x₁^2 + (k-5)*x₁ - 3*k = 0) ∧ 
  (x₂^2 + (k-5)*x₂ - 3*k = 0) := by
sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l895_89545


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_1729_l895_89570

theorem largest_prime_factor_of_1729 :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 1729 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 1729 → q ≤ p ∧ p = 19 :=
sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_1729_l895_89570


namespace NUMINAMATH_CALUDE_first_term_is_five_halves_l895_89541

/-- Sum of first n terms of an arithmetic sequence -/
def T (a : ℚ) (n : ℕ) : ℚ := n * (2 * a + (n - 1) * 5) / 2

/-- The ratio of T(3n) to T(n) is constant for all positive n -/
def ratio_is_constant (a : ℚ) : Prop :=
  ∃ k : ℚ, ∀ n : ℕ, n > 0 → T a (3*n) / T a n = k

theorem first_term_is_five_halves :
  ∀ a : ℚ, ratio_is_constant a → a = 5/2 := by sorry

end NUMINAMATH_CALUDE_first_term_is_five_halves_l895_89541


namespace NUMINAMATH_CALUDE_solve_for_k_l895_89527

theorem solve_for_k : ∃ k : ℤ, 2^5 - 10 = 5^2 + k ∧ k = -3 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_k_l895_89527


namespace NUMINAMATH_CALUDE_cos_300_degrees_l895_89546

theorem cos_300_degrees : Real.cos (300 * π / 180) = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_cos_300_degrees_l895_89546


namespace NUMINAMATH_CALUDE_polynomial_multiplication_l895_89594

theorem polynomial_multiplication (x y : ℝ) :
  (3 * x^2 - 4 * y^3) * (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = 27 * x^6 - 64 * y^9 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_multiplication_l895_89594


namespace NUMINAMATH_CALUDE_triangle_isosceles_condition_l895_89548

/-- A triangle with sides a, b, and c is isosceles if at least two of its sides are equal. -/
def IsIsosceles (a b c : ℝ) : Prop :=
  a = b ∨ b = c ∨ a = c

/-- If the three sides a, b, and c of a triangle ABC satisfy (a-b)(b²-2bc+c²) = 0,
    then the triangle ABC is isosceles. -/
theorem triangle_isosceles_condition (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) 
    (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
    (h_condition : (a - b) * (b^2 - 2*b*c + c^2) = 0) : 
    IsIsosceles a b c := by
  sorry


end NUMINAMATH_CALUDE_triangle_isosceles_condition_l895_89548


namespace NUMINAMATH_CALUDE_model_height_calculation_l895_89504

/-- The height of the Eiffel Tower in meters -/
def eiffel_height : ℝ := 320

/-- The capacity of the Eiffel Tower's observation deck in number of people -/
def eiffel_capacity : ℝ := 800

/-- The space required per person in square meters -/
def space_per_person : ℝ := 1

/-- The equivalent capacity of Mira's model in number of people -/
def model_capacity : ℝ := 0.8

/-- The height of Mira's model in meters -/
def model_height : ℝ := 10.12

theorem model_height_calculation :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  abs (model_height - eiffel_height * (model_capacity / eiffel_capacity).sqrt) < ε :=
sorry

end NUMINAMATH_CALUDE_model_height_calculation_l895_89504


namespace NUMINAMATH_CALUDE_binomial_expansion_simplification_l895_89514

theorem binomial_expansion_simplification (x : ℝ) : 
  (2*x+1)^5 - 5*(2*x+1)^4 + 10*(2*x+1)^3 - 10*(2*x+1)^2 + 5*(2*x+1) - 1 = 32*x^5 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_simplification_l895_89514


namespace NUMINAMATH_CALUDE_ellipse_max_value_l895_89571

theorem ellipse_max_value (x y : ℝ) : 
  ((x - 4)^2 / 4 + y^2 / 9 = 1) → (x^2 / 4 + y^2 / 9 ≤ 9) ∧ (∃ x y : ℝ, ((x - 4)^2 / 4 + y^2 / 9 = 1) ∧ (x^2 / 4 + y^2 / 9 = 9)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_max_value_l895_89571


namespace NUMINAMATH_CALUDE_roberts_expenses_l895_89534

theorem roberts_expenses (total : ℝ) (machinery : ℝ) (cash_percentage : ℝ) 
  (h1 : total = 250)
  (h2 : machinery = 125)
  (h3 : cash_percentage = 0.1)
  : total - machinery - (cash_percentage * total) = 100 := by
  sorry

end NUMINAMATH_CALUDE_roberts_expenses_l895_89534


namespace NUMINAMATH_CALUDE_date_calculation_l895_89530

/-- Given that December 1, 2015 was a Tuesday (day 2 of the week) and there are 31 days between
    December 1, 2015 and January 1, 2016, prove that January 1, 2016 was a Friday (day 5 of the week) -/
theorem date_calculation (start_day : Nat) (days_between : Nat) (end_day : Nat) : 
  start_day = 2 → days_between = 31 → end_day = (start_day + days_between) % 7 → end_day = 5 := by
  sorry

#check date_calculation

end NUMINAMATH_CALUDE_date_calculation_l895_89530


namespace NUMINAMATH_CALUDE_f_equal_implies_sum_negative_l895_89590

noncomputable def f (x : ℝ) : ℝ := ((1 - x) / (1 + x^2)) * Real.exp x

theorem f_equal_implies_sum_negative (x₁ x₂ : ℝ) (h₁ : f x₁ = f x₂) (h₂ : x₁ ≠ x₂) : x₁ + x₂ < 0 := by
  sorry

end NUMINAMATH_CALUDE_f_equal_implies_sum_negative_l895_89590


namespace NUMINAMATH_CALUDE_min_area_triangle_m_sum_l895_89583

/-- The sum of m values for minimum area triangle -/
theorem min_area_triangle_m_sum : 
  ∀ (m : ℤ), 
  let A : ℝ × ℝ := (2, 5)
  let B : ℝ × ℝ := (14, 13)
  let C : ℝ × ℝ := (6, m)
  let triangle_area (m : ℤ) : ℝ := sorry -- Function to calculate triangle area
  let min_area : ℝ := sorry -- Minimum area of the triangle
  (∃ (m₁ m₂ : ℤ), 
    m₁ ≠ m₂ ∧ 
    triangle_area m₁ = min_area ∧ 
    triangle_area m₂ = min_area ∧ 
    m₁ + m₂ = 16) := by sorry


end NUMINAMATH_CALUDE_min_area_triangle_m_sum_l895_89583

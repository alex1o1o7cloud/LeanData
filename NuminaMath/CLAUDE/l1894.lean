import Mathlib

namespace NUMINAMATH_CALUDE_pen_diary_cost_l1894_189465

/-- Given that 6 pens and 5 diaries cost $6.10, and 3 pens and 4 diaries cost $4.60,
    prove that 12 pens and 8 diaries cost $10.16 -/
theorem pen_diary_cost : ∃ (pen_cost diary_cost : ℝ),
  (6 * pen_cost + 5 * diary_cost = 6.10) ∧
  (3 * pen_cost + 4 * diary_cost = 4.60) ∧
  (12 * pen_cost + 8 * diary_cost = 10.16) := by
  sorry


end NUMINAMATH_CALUDE_pen_diary_cost_l1894_189465


namespace NUMINAMATH_CALUDE_chocolate_bars_distribution_l1894_189402

theorem chocolate_bars_distribution (total_bars : ℕ) (num_people : ℕ) 
  (h1 : total_bars = 12) (h2 : num_people = 3) :
  2 * (total_bars / num_people) = 8 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bars_distribution_l1894_189402


namespace NUMINAMATH_CALUDE_sum_of_roots_l1894_189411

theorem sum_of_roots (k c d : ℝ) (y₁ y₂ : ℝ) : 
  y₁ ≠ y₂ →
  5 * y₁^2 - k * y₁ - c = 0 →
  5 * y₂^2 - k * y₂ - c = 0 →
  5 * y₁^2 - k * y₁ = d →
  5 * y₂^2 - k * y₂ = d →
  d ≠ c →
  y₁ + y₂ = k / 5 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l1894_189411


namespace NUMINAMATH_CALUDE_equal_sum_groups_l1894_189451

/-- A function that checks if a list of natural numbers can be divided into three groups with equal sums -/
def canDivideIntoThreeEqualGroups (list : List Nat) : Prop :=
  ∃ (g1 g2 g3 : List Nat), 
    g1 ++ g2 ++ g3 = list ∧ 
    g1.sum = g2.sum ∧ 
    g2.sum = g3.sum

/-- The list of natural numbers from 1 to n -/
def naturalNumbersUpTo (n : Nat) : List Nat :=
  List.range n |>.map (· + 1)

/-- The main theorem stating the condition for when the natural numbers up to n can be divided into three groups with equal sums -/
theorem equal_sum_groups (n : Nat) : 
  canDivideIntoThreeEqualGroups (naturalNumbersUpTo n) ↔ 
  (∃ k : Nat, (k ≥ 2 ∧ (n = 3 * k ∨ n = 3 * k - 1))) :=
sorry

end NUMINAMATH_CALUDE_equal_sum_groups_l1894_189451


namespace NUMINAMATH_CALUDE_haley_halloween_candy_l1894_189443

/-- Represents the number of candy pieces Haley scored on Halloween -/
def initial_candy : ℕ := sorry

/-- Represents the number of candy pieces Haley ate -/
def eaten_candy : ℕ := 17

/-- Represents the number of candy pieces Haley received from her sister -/
def received_candy : ℕ := 19

/-- Represents the number of candy pieces Haley has now -/
def current_candy : ℕ := 35

/-- Proves that Haley scored 33 pieces of candy on Halloween -/
theorem haley_halloween_candy : initial_candy = 33 :=
  by
    have h : initial_candy - eaten_candy + received_candy = current_candy := sorry
    sorry

end NUMINAMATH_CALUDE_haley_halloween_candy_l1894_189443


namespace NUMINAMATH_CALUDE_range_of_a_l1894_189486

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*a*x + 4 > 0
def q (a : ℝ) : Prop := ∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - a ≥ 0

-- Define the set of valid values for a
def valid_a : Set ℝ := {a | (1 < a ∧ a < 2) ∨ a ≤ -2}

-- Theorem statement
theorem range_of_a (a : ℝ) :
  (p a ∨ q a) ∧ ¬(p a ∧ q a) → a ∈ valid_a :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1894_189486


namespace NUMINAMATH_CALUDE_symmetric_point_x_axis_example_l1894_189485

/-- The point symmetric to (x, y) with respect to the x-axis is (x, -y) -/
def symmetricPointXAxis (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

/-- The statement that the point symmetric to (-1, 2) with respect to the x-axis is (-1, -2) -/
theorem symmetric_point_x_axis_example : symmetricPointXAxis (-1, 2) = (-1, -2) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_x_axis_example_l1894_189485


namespace NUMINAMATH_CALUDE_expression_simplification_l1894_189461

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 2 + 1) :
  (((x^2 - 3) / (x + 2) - x + 2) / ((x^2 - 4) / (x^2 + 4*x + 4))) = Real.sqrt 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1894_189461


namespace NUMINAMATH_CALUDE_high_school_students_l1894_189454

theorem high_school_students (high_school middle_school lower_school : ℕ) : 
  high_school = 4 * lower_school →
  high_school + lower_school = 7 * middle_school →
  middle_school = 300 →
  high_school = 1680 := by
sorry

end NUMINAMATH_CALUDE_high_school_students_l1894_189454


namespace NUMINAMATH_CALUDE_ice_cost_l1894_189476

theorem ice_cost (cost_two_bags : ℝ) (num_bags : ℕ) : 
  cost_two_bags = 1.46 → num_bags = 4 → num_bags * (cost_two_bags / 2) = 2.92 := by
  sorry

end NUMINAMATH_CALUDE_ice_cost_l1894_189476


namespace NUMINAMATH_CALUDE_complex_product_example_l1894_189469

theorem complex_product_example : 
  let z₁ : ℂ := -1 + 2 * Complex.I
  let z₂ : ℂ := 2 + Complex.I
  z₁ * z₂ = -4 + 3 * Complex.I := by
sorry

end NUMINAMATH_CALUDE_complex_product_example_l1894_189469


namespace NUMINAMATH_CALUDE_perpendicular_vectors_imply_m_value_l1894_189491

/-- Given vectors a, b, and c in ℝ², prove that if a - b is perpendicular to c,
    then the value of m in b is -3. -/
theorem perpendicular_vectors_imply_m_value (a b c : ℝ × ℝ) (m : ℝ) :
  a = (-1, 2) →
  b = (m, -1) →
  c = (3, -2) →
  (a.1 - b.1) * c.1 + (a.2 - b.2) * c.2 = 0 →
  m = -3 := by
  sorry

#check perpendicular_vectors_imply_m_value

end NUMINAMATH_CALUDE_perpendicular_vectors_imply_m_value_l1894_189491


namespace NUMINAMATH_CALUDE_two_sqrt_two_minus_three_is_negative_l1894_189457

theorem two_sqrt_two_minus_three_is_negative : 2 * Real.sqrt 2 - 3 < 0 := by
  sorry

end NUMINAMATH_CALUDE_two_sqrt_two_minus_three_is_negative_l1894_189457


namespace NUMINAMATH_CALUDE_existence_of_prime_and_sequence_l1894_189496

theorem existence_of_prime_and_sequence (k : ℕ+) :
  ∃ (p : ℕ) (a : Fin (k+3) → ℕ), 
    Prime p ∧ 
    (∀ i : Fin (k+3), 1 ≤ a i ∧ a i < p) ∧
    (∀ i j : Fin (k+3), i ≠ j → a i ≠ a j) ∧
    (∀ i : Fin k, p ∣ (a i * a (i+1) * a (i+2) * a (i+3) - i)) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_prime_and_sequence_l1894_189496


namespace NUMINAMATH_CALUDE_max_volume_rectangular_prism_l1894_189483

/-- Represents the volume of a rectangular prism as a function of the shorter base edge length -/
def prism_volume (x : ℝ) : ℝ := x * (x + 0.5) * (3.2 - 2 * x)

/-- The theorem stating the maximum volume and corresponding height of the rectangular prism -/
theorem max_volume_rectangular_prism :
  ∃ (x : ℝ), 
    0 < x ∧ 
    x < 1.6 ∧
    prism_volume x = 1.8 ∧
    3.2 - 2 * x = 1.2 ∧
    ∀ (y : ℝ), 0 < y ∧ y < 1.6 → prism_volume y ≤ prism_volume x :=
sorry


end NUMINAMATH_CALUDE_max_volume_rectangular_prism_l1894_189483


namespace NUMINAMATH_CALUDE_sin_70_65_minus_sin_20_25_l1894_189444

theorem sin_70_65_minus_sin_20_25 : 
  Real.sin (70 * π / 180) * Real.sin (65 * π / 180) - 
  Real.sin (20 * π / 180) * Real.sin (25 * π / 180) = 
  Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_sin_70_65_minus_sin_20_25_l1894_189444


namespace NUMINAMATH_CALUDE_shopkeeper_profit_l1894_189466

theorem shopkeeper_profit (c s : ℝ) (p : ℝ) (h1 : c > 0) (h2 : s > c) :
  s = c * (1 + p / 100) ∧ 
  s = (0.9 * c) * (1 + (p + 12) / 100) →
  p = 8 := by
sorry

end NUMINAMATH_CALUDE_shopkeeper_profit_l1894_189466


namespace NUMINAMATH_CALUDE_milburg_population_l1894_189464

theorem milburg_population :
  let grown_ups : ℕ := 5256
  let children : ℕ := 2987
  grown_ups + children = 8243 := by
  sorry

end NUMINAMATH_CALUDE_milburg_population_l1894_189464


namespace NUMINAMATH_CALUDE_students_only_english_l1894_189416

theorem students_only_english (total : ℕ) (both : ℕ) (german : ℕ) (h1 : total = 52) (h2 : both = 12) (h3 : german = 22) (h4 : total ≥ german) : total - german + both = 30 := by
  sorry

end NUMINAMATH_CALUDE_students_only_english_l1894_189416


namespace NUMINAMATH_CALUDE_lamp_configurations_l1894_189434

/-- Represents the number of reachable configurations for n lamps -/
def reachableConfigurations (n : ℕ) : ℕ :=
  if n % 3 = 0 then 2^(n-2) else 2^n

/-- Theorem stating the number of reachable configurations for n cyclically connected lamps -/
theorem lamp_configurations (n : ℕ) (h : n > 2) :
  reachableConfigurations n = if n % 3 = 0 then 2^(n-2) else 2^n :=
by sorry

end NUMINAMATH_CALUDE_lamp_configurations_l1894_189434


namespace NUMINAMATH_CALUDE_factorial_difference_quotient_l1894_189471

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem factorial_difference_quotient : (factorial 13 - factorial 12) / factorial 10 = 1584 := by
  sorry

end NUMINAMATH_CALUDE_factorial_difference_quotient_l1894_189471


namespace NUMINAMATH_CALUDE_simplify_expression_l1894_189487

theorem simplify_expression (x : ℝ) (hx : x ≥ 0) : (3 * x^(1/2))^6 = 729 * x^3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1894_189487


namespace NUMINAMATH_CALUDE_circle_area_from_circumference_l1894_189495

theorem circle_area_from_circumference : 
  ∀ (r : ℝ), 
  (2 * π * r = 36 * π) → 
  (π * r^2 = 324 * π) := by
sorry

end NUMINAMATH_CALUDE_circle_area_from_circumference_l1894_189495


namespace NUMINAMATH_CALUDE_quadratic_term_elimination_l1894_189448

theorem quadratic_term_elimination (m : ℝ) : 
  (∀ x : ℝ, 36 * x^2 - 3 * x + 5 - (-3 * x^3 - 12 * m * x^2 + 5 * x - 7) = 3 * x^3 - 8 * x + 12) → 
  m^3 = -27 := by
sorry

end NUMINAMATH_CALUDE_quadratic_term_elimination_l1894_189448


namespace NUMINAMATH_CALUDE_circle_radius_proof_l1894_189477

theorem circle_radius_proof (num_pencils : ℕ) (pencil_length : ℚ) (inches_per_foot : ℕ) :
  num_pencils = 56 →
  pencil_length = 6 →
  inches_per_foot = 12 →
  (num_pencils * pencil_length / (2 * inches_per_foot) : ℚ) = 14 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_proof_l1894_189477


namespace NUMINAMATH_CALUDE_greatest_integer_inequality_l1894_189435

theorem greatest_integer_inequality : ∀ x : ℤ, (5 : ℚ) / 8 > (x : ℚ) / 17 ↔ x ≤ 10 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_inequality_l1894_189435


namespace NUMINAMATH_CALUDE_loads_required_l1894_189426

def washing_machine_capacity : ℕ := 9
def total_clothing : ℕ := 27

theorem loads_required : (total_clothing + washing_machine_capacity - 1) / washing_machine_capacity = 3 := by
  sorry

end NUMINAMATH_CALUDE_loads_required_l1894_189426


namespace NUMINAMATH_CALUDE_melanie_dimes_l1894_189412

theorem melanie_dimes (initial : ℕ) (from_dad : ℕ) (total : ℕ) : 
  initial = 7 → from_dad = 8 → total = 19 → total - (initial + from_dad) = 4 := by
sorry

end NUMINAMATH_CALUDE_melanie_dimes_l1894_189412


namespace NUMINAMATH_CALUDE_probability_no_more_than_five_girls_between_first_last_boys_l1894_189441

def total_children : ℕ := 20
def num_girls : ℕ := 11
def num_boys : ℕ := 9

def valid_arrangements (n k : ℕ) : ℕ := Nat.choose n k

theorem probability_no_more_than_five_girls_between_first_last_boys :
  (valid_arrangements 14 9 + 6 * valid_arrangements 13 8) / valid_arrangements total_children num_boys =
  (valid_arrangements 14 9 + 6 * valid_arrangements 13 8) / valid_arrangements 20 9 :=
by sorry

end NUMINAMATH_CALUDE_probability_no_more_than_five_girls_between_first_last_boys_l1894_189441


namespace NUMINAMATH_CALUDE_smallest_w_l1894_189467

theorem smallest_w (w : ℕ+) : 
  (∃ k : ℕ, 936 * w.val = k * 2^5) ∧ 
  (∃ k : ℕ, 936 * w.val = k * 3^3) ∧ 
  (∃ k : ℕ, 936 * w.val = k * 11^2) →
  w.val ≥ 4356 :=
by sorry

end NUMINAMATH_CALUDE_smallest_w_l1894_189467


namespace NUMINAMATH_CALUDE_cos_alpha_value_l1894_189462

def point_on_terminal_side (α : Real) (x y : Real) : Prop :=
  ∃ (r : Real), r > 0 ∧ x = r * Real.cos α ∧ y = r * Real.sin α

theorem cos_alpha_value (α : Real) :
  point_on_terminal_side α 1 3 → Real.cos α = 1 / Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l1894_189462


namespace NUMINAMATH_CALUDE_OTVSU_shape_l1894_189489

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- The figure OTVSU -/
structure OTVSU where
  O : Point2D
  T : Point2D
  V : Point2D
  S : Point2D
  U : Point2D

/-- Predicate to check if a figure is a parallelogram -/
def isParallelogram (f : OTVSU) : Prop := sorry

/-- Predicate to check if a figure is a straight line -/
def isStraightLine (f : OTVSU) : Prop := sorry

/-- Predicate to check if a figure is a trapezoid -/
def isTrapezoid (f : OTVSU) : Prop := sorry

theorem OTVSU_shape :
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
  let f : OTVSU := {
    O := ⟨0, 0⟩,
    T := ⟨x₁, y₁⟩,
    V := ⟨x₁ + x₂, y₁ + y₂⟩,
    S := ⟨x₁ - x₂, y₁ - y₂⟩,
    U := ⟨x₂, y₂⟩
  }
  (isParallelogram f ∨ isStraightLine f) ∧ ¬isTrapezoid f := by
  sorry

end NUMINAMATH_CALUDE_OTVSU_shape_l1894_189489


namespace NUMINAMATH_CALUDE_box_surface_area_l1894_189432

/-- Proves that the surface area of a rectangular box is 975 given specific conditions -/
theorem box_surface_area (a b c : ℝ) 
  (edge_sum : 4 * a + 4 * b + 4 * c = 160)
  (diagonal : Real.sqrt (a^2 + b^2 + c^2) = 25)
  (volume : a * b * c = 600) :
  2 * (a * b + b * c + c * a) = 975 := by
sorry

end NUMINAMATH_CALUDE_box_surface_area_l1894_189432


namespace NUMINAMATH_CALUDE_village_population_theorem_l1894_189418

theorem village_population_theorem (total_population : ℕ) 
  (h1 : total_population = 800) 
  (h2 : total_population % 4 = 0) 
  (h3 : 3 * (total_population / 4) = total_population - (total_population / 4)) :
  total_population / 4 = 200 :=
sorry

end NUMINAMATH_CALUDE_village_population_theorem_l1894_189418


namespace NUMINAMATH_CALUDE_expression_evaluation_l1894_189453

theorem expression_evaluation :
  let a : ℚ := 2
  let b : ℚ := -1
  3 * (2 * a^2 * b - a * b^2) - 2 * (5 * a^2 * b - 2 * a * b^2) = 18 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1894_189453


namespace NUMINAMATH_CALUDE_prob_sum_five_is_one_ninth_l1894_189403

/-- The number of faces on each die -/
def numFaces : ℕ := 6

/-- The set of possible outcomes when rolling two dice -/
def outcomes : Finset (ℕ × ℕ) :=
  Finset.product (Finset.range numFaces) (Finset.range numFaces)

/-- The set of outcomes where the sum of the dice is 5 -/
def sumFiveOutcomes : Finset (ℕ × ℕ) :=
  outcomes.filter (fun p => p.1 + p.2 + 2 = 5)

/-- The probability of the sum of two fair dice being 5 -/
theorem prob_sum_five_is_one_ninth :
  (sumFiveOutcomes.card : ℚ) / outcomes.card = 1 / 9 := by
  sorry


end NUMINAMATH_CALUDE_prob_sum_five_is_one_ninth_l1894_189403


namespace NUMINAMATH_CALUDE_root_in_interval_l1894_189473

def f (x : ℝ) := x^3 - 2*x - 1

theorem root_in_interval :
  f 1 < 0 →
  f 2 > 0 →
  f (3/2) < 0 →
  ∃ x : ℝ, 3/2 < x ∧ x < 2 ∧ f x = 0 :=
by sorry

end NUMINAMATH_CALUDE_root_in_interval_l1894_189473


namespace NUMINAMATH_CALUDE_remaining_pictures_l1894_189405

/-- The number of pictures Megan took at the zoo -/
def zoo_pictures : ℕ := 15

/-- The number of pictures Megan took at the museum -/
def museum_pictures : ℕ := 18

/-- The number of pictures Megan deleted -/
def deleted_pictures : ℕ := 31

/-- The theorem states that the number of pictures Megan still has from her vacation is 2 -/
theorem remaining_pictures :
  zoo_pictures + museum_pictures - deleted_pictures = 2 := by sorry

end NUMINAMATH_CALUDE_remaining_pictures_l1894_189405


namespace NUMINAMATH_CALUDE_camp_grouping_l1894_189468

theorem camp_grouping (total_children : ℕ) (max_group_size : ℕ) (h1 : total_children = 30) (h2 : max_group_size = 12) :
  ∃ (group_size : ℕ) (num_groups : ℕ),
    group_size ≤ max_group_size ∧
    group_size * num_groups = total_children ∧
    ∀ (k : ℕ), k ≤ max_group_size → k * (total_children / k) = total_children → num_groups ≤ (total_children / k) :=
by
  sorry

end NUMINAMATH_CALUDE_camp_grouping_l1894_189468


namespace NUMINAMATH_CALUDE_prob_at_least_one_from_three_suits_l1894_189417

/-- Represents a standard deck of 52 cards -/
def standardDeck : ℕ := 52

/-- Number of cards in each suit -/
def cardsPerSuit : ℕ := 13

/-- Number of cards drawn -/
def numDraws : ℕ := 5

/-- Number of specific suits considered -/
def numSpecificSuits : ℕ := 3

/-- Probability of drawing a card from the specific suits in one draw -/
def probSpecificSuits : ℚ := (cardsPerSuit * numSpecificSuits) / standardDeck

/-- Probability of drawing a card not from the specific suits in one draw -/
def probNotSpecificSuits : ℚ := 1 - probSpecificSuits

/-- 
Theorem: The probability of drawing at least one card from each of three specific suits 
when choosing five cards with replacement from a standard 52-card deck is 1023/1024.
-/
theorem prob_at_least_one_from_three_suits : 
  1 - probNotSpecificSuits ^ numDraws = 1023 / 1024 := by sorry

end NUMINAMATH_CALUDE_prob_at_least_one_from_three_suits_l1894_189417


namespace NUMINAMATH_CALUDE_tabletennis_arrangements_eq_252_l1894_189420

/-- The number of ways to arrange 5 players from a team of 10, 
    where 3 specific players must occupy positions 1, 3, and 5, 
    and 2 players from the remaining 7 must occupy positions 2 and 4 -/
def tabletennis_arrangements (total_players : ℕ) (main_players : ℕ) 
    (players_to_send : ℕ) (remaining_players : ℕ) : ℕ := 
  Nat.factorial main_players * (remaining_players * (remaining_players - 1))

theorem tabletennis_arrangements_eq_252 : 
  tabletennis_arrangements 10 3 5 7 = 252 := by
  sorry

#eval tabletennis_arrangements 10 3 5 7

end NUMINAMATH_CALUDE_tabletennis_arrangements_eq_252_l1894_189420


namespace NUMINAMATH_CALUDE_function_identity_l1894_189413

theorem function_identity (f : ℝ → ℝ) 
  (h₁ : f 0 = 1)
  (h₂ : ∀ x y : ℝ, f (x * y + 1) = f x * f y - f y - x + 2) :
  ∀ x : ℝ, f x = x + 1 := by
  sorry

end NUMINAMATH_CALUDE_function_identity_l1894_189413


namespace NUMINAMATH_CALUDE_natasha_quarters_l1894_189400

theorem natasha_quarters : ∃ n : ℕ,
  8 < n ∧ n < 80 ∧
  n % 4 = 3 ∧
  n % 5 = 1 ∧
  n % 7 = 3 ∧
  n = 31 := by
sorry

end NUMINAMATH_CALUDE_natasha_quarters_l1894_189400


namespace NUMINAMATH_CALUDE_rectangle_area_l1894_189447

/-- Proves that a rectangle with a perimeter of 176 inches and a length 8 inches more than its width has an area of 1920 square inches. -/
theorem rectangle_area (w : ℝ) (l : ℝ) : 
  (2 * l + 2 * w = 176) →  -- Perimeter condition
  (l = w + 8) →            -- Length-width relation
  (l * w = 1920)           -- Area to be proved
  := by sorry

end NUMINAMATH_CALUDE_rectangle_area_l1894_189447


namespace NUMINAMATH_CALUDE_x_values_l1894_189429

theorem x_values (x : ℝ) : (|2000 * x + 2000| = 20 * 2000) → (x = 19 ∨ x = -21) := by
  sorry

end NUMINAMATH_CALUDE_x_values_l1894_189429


namespace NUMINAMATH_CALUDE_lottery_probability_l1894_189427

def powerball_count : ℕ := 30
def luckyball_count : ℕ := 50
def luckyball_draw : ℕ := 6

theorem lottery_probability :
  (1 : ℚ) / (powerball_count * (Nat.choose luckyball_count luckyball_draw)) = 1 / 476721000 :=
by sorry

end NUMINAMATH_CALUDE_lottery_probability_l1894_189427


namespace NUMINAMATH_CALUDE_student_allowance_l1894_189497

theorem student_allowance (initial_allowance : ℚ) : 
  let remaining_after_clothes := initial_allowance * (4/7)
  let remaining_after_games := remaining_after_clothes * (3/5)
  let remaining_after_books := remaining_after_games * (5/9)
  let remaining_after_donation := remaining_after_books * (1/2)
  remaining_after_donation = 3.75 → initial_allowance = 39.375 := by
  sorry

end NUMINAMATH_CALUDE_student_allowance_l1894_189497


namespace NUMINAMATH_CALUDE_jackson_flight_distance_l1894_189415

theorem jackson_flight_distance (beka_distance : ℕ) (difference : ℕ) (jackson_distance : ℕ) : 
  beka_distance = 873 → 
  difference = 310 → 
  beka_distance = jackson_distance + difference → 
  jackson_distance = 563 :=
by sorry

end NUMINAMATH_CALUDE_jackson_flight_distance_l1894_189415


namespace NUMINAMATH_CALUDE_prob_three_even_out_of_five_l1894_189406

-- Define a fair 6-sided die
def FairDie := Fin 6

-- Define the probability of rolling an even number on a single die
def probEven : ℚ := 1 / 2

-- Define the number of dice
def numDice : ℕ := 5

-- Define the number of dice we want to show even
def numEven : ℕ := 3

-- Theorem statement
theorem prob_three_even_out_of_five :
  (Nat.choose numDice numEven : ℚ) * probEven ^ numDice = 5 / 16 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_even_out_of_five_l1894_189406


namespace NUMINAMATH_CALUDE_min_value_ab_l1894_189494

theorem min_value_ab (a b : ℝ) (h : 0 < a ∧ 0 < b) (eq : 1/a + 2/b = Real.sqrt (a*b)) : 
  2 * Real.sqrt 2 ≤ a * b := by
  sorry

end NUMINAMATH_CALUDE_min_value_ab_l1894_189494


namespace NUMINAMATH_CALUDE_chess_tournament_games_l1894_189408

theorem chess_tournament_games (n : ℕ) 
  (h1 : n > 0) 
  (h2 : (10 * 9 * n) / 2 = 90) : n = 2 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l1894_189408


namespace NUMINAMATH_CALUDE_total_yen_is_correct_l1894_189436

/-- Represents the total assets of a family in various currencies and investments -/
structure FamilyAssets where
  bahamian_dollars : ℝ
  us_dollars : ℝ
  euros : ℝ
  checking_account1 : ℝ
  checking_account2 : ℝ
  savings_account1 : ℝ
  savings_account2 : ℝ
  stocks : ℝ
  bonds : ℝ
  mutual_funds : ℝ

/-- Exchange rates for different currencies to Japanese yen -/
structure ExchangeRates where
  bahamian_to_yen : ℝ
  usd_to_yen : ℝ
  euro_to_yen : ℝ

/-- Calculates the total amount of yen from all assets -/
def total_yen (assets : FamilyAssets) (rates : ExchangeRates) : ℝ :=
  assets.bahamian_dollars * rates.bahamian_to_yen +
  assets.us_dollars * rates.usd_to_yen +
  assets.euros * rates.euro_to_yen +
  assets.checking_account1 +
  assets.checking_account2 +
  assets.savings_account1 +
  assets.savings_account2 +
  assets.stocks +
  assets.bonds +
  assets.mutual_funds

/-- Theorem stating that the total amount of yen is 1,716,611 -/
theorem total_yen_is_correct (assets : FamilyAssets) (rates : ExchangeRates) :
  assets.bahamian_dollars = 5000 →
  assets.us_dollars = 2000 →
  assets.euros = 3000 →
  assets.checking_account1 = 15000 →
  assets.checking_account2 = 6359 →
  assets.savings_account1 = 5500 →
  assets.savings_account2 = 3102 →
  assets.stocks = 200000 →
  assets.bonds = 150000 →
  assets.mutual_funds = 120000 →
  rates.bahamian_to_yen = 122.13 →
  rates.usd_to_yen = 110.25 →
  rates.euro_to_yen = 128.50 →
  total_yen assets rates = 1716611 := by
  sorry

end NUMINAMATH_CALUDE_total_yen_is_correct_l1894_189436


namespace NUMINAMATH_CALUDE_tiaorizhi_approximation_of_pi_l1894_189463

def tiaorizhi (a b c d : ℕ) : ℚ := (b + d) / (a + c)

theorem tiaorizhi_approximation_of_pi :
  let initial_lower : ℚ := 3 / 1
  let initial_upper : ℚ := 7 / 2
  let step1 : ℚ := tiaorizhi 1 3 2 7
  let step2 : ℚ := tiaorizhi 1 3 4 13
  let step3 : ℚ := tiaorizhi 1 3 5 16
  initial_lower < Real.pi ∧ Real.pi < initial_upper →
  step3 - Real.pi < 0.1 ∧ Real.pi < step3 := by
  sorry

end NUMINAMATH_CALUDE_tiaorizhi_approximation_of_pi_l1894_189463


namespace NUMINAMATH_CALUDE_possible_values_of_a_l1894_189409

def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | a*x - 2 = 0}

theorem possible_values_of_a :
  ∀ a : ℝ, (A ∪ B a = A) ↔ a ∈ ({0, 1, 2} : Set ℝ) :=
by sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l1894_189409


namespace NUMINAMATH_CALUDE_village_population_problem_l1894_189488

theorem village_population_problem (P : ℕ) : 
  (P : ℝ) * 0.9 * 0.85 = 2907 → P = 3801 := by
  sorry

end NUMINAMATH_CALUDE_village_population_problem_l1894_189488


namespace NUMINAMATH_CALUDE_gcd_1021_2729_l1894_189421

theorem gcd_1021_2729 : Nat.gcd 1021 2729 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1021_2729_l1894_189421


namespace NUMINAMATH_CALUDE_wheel_speed_is_seven_l1894_189459

noncomputable def wheel_speed (circumference : Real) (r : Real) : Prop :=
  let miles_per_rotation := circumference / 5280
  let t := miles_per_rotation / r
  let new_t := t - 1 / (3 * 3600)
  (r + 3) * new_t = miles_per_rotation

theorem wheel_speed_is_seven :
  ∀ (r : Real),
    wheel_speed 15 r →
    r = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_wheel_speed_is_seven_l1894_189459


namespace NUMINAMATH_CALUDE_f_contraction_implies_a_bound_l1894_189422

open Real

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * log x - x^2 + 1

-- State the theorem
theorem f_contraction_implies_a_bound
  (a : ℝ)
  (h_a_neg : a < 0)
  (h_contraction : ∀ (x₁ x₂ : ℝ), x₁ > 0 → x₂ > 0 →
    |f a x₁ - f a x₂| ≥ |x₁ - x₂|) :
  a ≤ -1/8 := by
  sorry

end NUMINAMATH_CALUDE_f_contraction_implies_a_bound_l1894_189422


namespace NUMINAMATH_CALUDE_total_soap_cost_two_years_l1894_189474

/-- Represents the types of soap -/
inductive SoapType
  | Lavender
  | Lemon
  | Sandalwood

/-- Represents the price of each soap type -/
def soapPrice (t : SoapType) : ℚ :=
  match t with
  | SoapType.Lavender => 4
  | SoapType.Lemon => 5
  | SoapType.Sandalwood => 6

/-- Represents the bulk discount for each soap type and quantity -/
def bulkDiscount (t : SoapType) (quantity : ℕ) : ℚ :=
  match t with
  | SoapType.Lavender =>
    if quantity ≥ 10 then 0.2
    else if quantity ≥ 5 then 0.1
    else 0
  | SoapType.Lemon =>
    if quantity ≥ 8 then 0.15
    else if quantity ≥ 4 then 0.05
    else 0
  | SoapType.Sandalwood =>
    if quantity ≥ 9 then 0.2
    else if quantity ≥ 6 then 0.1
    else if quantity ≥ 3 then 0.05
    else 0

/-- Calculates the cost of soap for a given type and quantity with bulk discount -/
def soapCost (t : SoapType) (quantity : ℕ) : ℚ :=
  let price := soapPrice t
  let discount := bulkDiscount t quantity
  quantity * price * (1 - discount)

/-- Theorem: The total amount Elias spends on soap in two years is $112.4 -/
theorem total_soap_cost_two_years :
  soapCost SoapType.Lavender 5 + soapCost SoapType.Lavender 3 +
  soapCost SoapType.Lemon 4 + soapCost SoapType.Lemon 4 +
  soapCost SoapType.Sandalwood 6 + soapCost SoapType.Sandalwood 2 = 112.4 := by
  sorry


end NUMINAMATH_CALUDE_total_soap_cost_two_years_l1894_189474


namespace NUMINAMATH_CALUDE_find_x_l1894_189498

theorem find_x : ∃ x : ℝ, 5.76 = 0.12 * (0.40 * x) ∧ x = 120 := by
  sorry

end NUMINAMATH_CALUDE_find_x_l1894_189498


namespace NUMINAMATH_CALUDE_value_of_b_l1894_189493

theorem value_of_b (a b : ℝ) (h1 : 3 * a + 2 = 2) (h2 : b - a = 3) : b = 3 := by
  sorry

end NUMINAMATH_CALUDE_value_of_b_l1894_189493


namespace NUMINAMATH_CALUDE_hexagon_segment_probability_l1894_189470

/-- The set of all sides and diagonals of a regular hexagon -/
def T : Finset ℝ := sorry

/-- The number of sides in a regular hexagon -/
def num_sides : ℕ := 6

/-- The number of short diagonals in a regular hexagon -/
def num_short_diagonals : ℕ := 3

/-- The number of long diagonals in a regular hexagon -/
def num_long_diagonals : ℕ := 6

/-- The probability of selecting two segments of the same length from T -/
def prob_same_length : ℚ := 11/35

theorem hexagon_segment_probability :
  let total := T.card
  let same_length_pairs := (num_sides.choose 2) + (num_short_diagonals.choose 2) + (num_long_diagonals.choose 2)
  prob_same_length = same_length_pairs / (total.choose 2) := by
  sorry

end NUMINAMATH_CALUDE_hexagon_segment_probability_l1894_189470


namespace NUMINAMATH_CALUDE_f_inequality_solution_set_l1894_189460

def f (x : ℝ) : ℝ := |2*x + 1| - |x - 2|

theorem f_inequality_solution_set :
  {x : ℝ | f x > 2} = {x : ℝ | x < -5 ∨ x > 1} := by sorry

end NUMINAMATH_CALUDE_f_inequality_solution_set_l1894_189460


namespace NUMINAMATH_CALUDE_car_distances_l1894_189456

/-- Represents the possible distances between two cars after one hour, given their initial distance and speeds. -/
def possible_distances (initial_distance : ℝ) (speed1 : ℝ) (speed2 : ℝ) : Set ℝ :=
  { d | ∃ (direction1 direction2 : Bool),
      d = |initial_distance + (if direction1 then speed1 else -speed1) - (if direction2 then speed2 else -speed2)| }

/-- Theorem stating the possible distances between two cars after one hour. -/
theorem car_distances (initial_distance : ℝ) (speed1 : ℝ) (speed2 : ℝ)
    (h_initial : initial_distance = 200)
    (h_speed1 : speed1 = 60)
    (h_speed2 : speed2 = 80) :
    possible_distances initial_distance speed1 speed2 = {60, 340, 180, 220} := by
  sorry

end NUMINAMATH_CALUDE_car_distances_l1894_189456


namespace NUMINAMATH_CALUDE_image_property_l1894_189439

class StarOperation (T : Type) where
  star : T → T → T

variable {T : Type} [StarOperation T]

def image (a : T) : Set T :=
  {c | ∃ b, c = StarOperation.star a b}

theorem image_property (a : T) (c : T) (h : c ∈ image a) :
  StarOperation.star a c = c := by
  sorry

end NUMINAMATH_CALUDE_image_property_l1894_189439


namespace NUMINAMATH_CALUDE_logical_propositions_l1894_189401

theorem logical_propositions (p q : Prop) : 
  (((p ∧ q) → (p ∨ q)) ∧ ¬((p ∨ q) → (p ∧ q))) ∧
  (((¬p) → ¬(p ∨ q)) ∧ ¬((p ∨ q) → ¬(¬p))) := by
  sorry

end NUMINAMATH_CALUDE_logical_propositions_l1894_189401


namespace NUMINAMATH_CALUDE_max_correct_percentage_l1894_189430

theorem max_correct_percentage
  (total : ℝ)
  (solo_portion : ℝ)
  (together_portion : ℝ)
  (chloe_solo_correct : ℝ)
  (chloe_overall_correct : ℝ)
  (max_solo_correct : ℝ)
  (h1 : solo_portion = 2/3)
  (h2 : together_portion = 1/3)
  (h3 : solo_portion + together_portion = 1)
  (h4 : chloe_solo_correct = 0.7)
  (h5 : chloe_overall_correct = 0.82)
  (h6 : max_solo_correct = 0.85)
  : max_solo_correct * solo_portion + (chloe_overall_correct - chloe_solo_correct * solo_portion) = 0.92 := by
  sorry

end NUMINAMATH_CALUDE_max_correct_percentage_l1894_189430


namespace NUMINAMATH_CALUDE_problem_one_problem_two_l1894_189481

-- Problem 1
theorem problem_one : Real.sqrt 9 - (-2023)^(0 : ℤ) + 2^(-1 : ℤ) = 5/2 := by sorry

-- Problem 2
theorem problem_two (a b : ℝ) (hb : b ≠ 0) :
  (a / b - 1) / ((a^2 - b^2) / (2*b)) = 2 / (a + b) := by sorry

end NUMINAMATH_CALUDE_problem_one_problem_two_l1894_189481


namespace NUMINAMATH_CALUDE_bamboo_nine_nodes_l1894_189438

theorem bamboo_nine_nodes (a : ℕ → ℚ) (d : ℚ) :
  (∀ n, a (n + 1) = a n + d) →  -- arithmetic sequence
  a 1 + a 2 + a 3 + a 4 = 3 →   -- sum of first 4 terms
  a 7 + a 8 + a 9 = 4 →         -- sum of last 3 terms
  a 1 + a 3 + a 9 = 17/6 :=     -- sum of 1st, 3rd, and 9th terms
by sorry

end NUMINAMATH_CALUDE_bamboo_nine_nodes_l1894_189438


namespace NUMINAMATH_CALUDE_card_selection_counts_l1894_189433

/-- Represents a standard deck of 52 playing cards -/
structure Deck :=
  (cards : Finset (Fin 4 × Fin 13))
  (card_count : cards.card = 52)

/-- Counts the number of ways to select 4 cards with different suits and ranks -/
def count_four_different (d : Deck) : ℕ := sorry

/-- Counts the number of ways to select 6 cards with all suits represented -/
def count_six_all_suits (d : Deck) : ℕ := sorry

/-- Theorem stating the correct counts for both selections -/
theorem card_selection_counts (d : Deck) : 
  count_four_different d = 17160 ∧ count_six_all_suits d = 8682544 := by sorry

end NUMINAMATH_CALUDE_card_selection_counts_l1894_189433


namespace NUMINAMATH_CALUDE_odd_function_negative_domain_l1894_189431

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_negative_domain
  (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_positive : ∀ x > 0, f x = -x + 2) :
  ∀ x < 0, f x = -x - 2 := by
sorry

end NUMINAMATH_CALUDE_odd_function_negative_domain_l1894_189431


namespace NUMINAMATH_CALUDE_tickets_purchased_l1894_189446

theorem tickets_purchased (olivia_money : ℕ) (nigel_money : ℕ) (ticket_cost : ℕ) (money_left : ℕ) :
  olivia_money = 112 →
  nigel_money = 139 →
  ticket_cost = 28 →
  money_left = 83 →
  (olivia_money + nigel_money - money_left) / ticket_cost = 6 :=
by sorry

end NUMINAMATH_CALUDE_tickets_purchased_l1894_189446


namespace NUMINAMATH_CALUDE_correct_mean_calculation_l1894_189499

theorem correct_mean_calculation (n : ℕ) (initial_mean : ℚ) (incorrect_value correct_value : ℚ) :
  n = 30 ∧ 
  initial_mean = 250 ∧ 
  incorrect_value = 135 ∧ 
  correct_value = 165 →
  (n * initial_mean + (correct_value - incorrect_value)) / n = 251 := by
  sorry

end NUMINAMATH_CALUDE_correct_mean_calculation_l1894_189499


namespace NUMINAMATH_CALUDE_square_sum_equals_eleven_halves_l1894_189437

theorem square_sum_equals_eleven_halves (a b : ℝ) 
  (h1 : (a + b)^2 = 7) 
  (h2 : (a - b)^2 = 4) : 
  a^2 + b^2 = 11/2 := by
sorry

end NUMINAMATH_CALUDE_square_sum_equals_eleven_halves_l1894_189437


namespace NUMINAMATH_CALUDE_tan_equation_solution_l1894_189455

theorem tan_equation_solution (θ : Real) :
  2 * Real.tan θ - Real.tan (θ + π/4) = 7 → Real.tan θ = 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_equation_solution_l1894_189455


namespace NUMINAMATH_CALUDE_circle_area_theorem_l1894_189428

-- Define the center and point on the circle
def center : ℝ × ℝ := (-2, 5)
def point : ℝ × ℝ := (8, -4)

-- Calculate the squared distance between two points
def distance_squared (p1 p2 : ℝ × ℝ) : ℝ :=
  (p2.1 - p1.1)^2 + (p2.2 - p1.2)^2

-- Define the theorem
theorem circle_area_theorem :
  let r := Real.sqrt (distance_squared center point)
  π * r^2 = 181 * π := by sorry

end NUMINAMATH_CALUDE_circle_area_theorem_l1894_189428


namespace NUMINAMATH_CALUDE_joan_quarters_l1894_189424

def total_cents : ℕ := 150
def cents_per_quarter : ℕ := 25

theorem joan_quarters : total_cents / cents_per_quarter = 6 := by
  sorry

end NUMINAMATH_CALUDE_joan_quarters_l1894_189424


namespace NUMINAMATH_CALUDE_bobby_total_blocks_l1894_189442

def bobby_blocks : ℕ := 2
def father_gift : ℕ := 6

theorem bobby_total_blocks :
  bobby_blocks + father_gift = 8 := by
  sorry

end NUMINAMATH_CALUDE_bobby_total_blocks_l1894_189442


namespace NUMINAMATH_CALUDE_school_population_after_new_students_l1894_189404

theorem school_population_after_new_students (initial_avg_age : ℝ) (new_students : ℕ) 
  (new_students_avg_age : ℝ) (avg_age_decrease : ℝ) :
  initial_avg_age = 48 →
  new_students = 120 →
  new_students_avg_age = 32 →
  avg_age_decrease = 4 →
  ∃ (initial_students : ℕ),
    (initial_students + new_students : ℝ) * (initial_avg_age - avg_age_decrease) = 
    initial_students * initial_avg_age + new_students * new_students_avg_age ∧
    initial_students + new_students = 480 :=
by sorry

end NUMINAMATH_CALUDE_school_population_after_new_students_l1894_189404


namespace NUMINAMATH_CALUDE_emmy_and_gerry_apples_l1894_189478

/-- The number of apples that can be bought with a given amount of money at a given price per apple -/
def apples_buyable (money : ℕ) (price : ℕ) : ℕ :=
  money / price

theorem emmy_and_gerry_apples : 
  let apple_price : ℕ := 2
  let emmy_money : ℕ := 200
  let gerry_money : ℕ := 100
  apples_buyable emmy_money apple_price + apples_buyable gerry_money apple_price = 150 :=
by sorry

end NUMINAMATH_CALUDE_emmy_and_gerry_apples_l1894_189478


namespace NUMINAMATH_CALUDE_ellipse_intersection_theorem_l1894_189492

/-- The ellipse C in standard form -/
def ellipse_C (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

/-- The line l that intersects the ellipse -/
def line_l (k m x y : ℝ) : Prop := y = k * x + m

/-- The condition for the intersection points A and B -/
def intersection_condition (xA yA xB yB : ℝ) : Prop :=
  (2 * xA + xB)^2 + (2 * yA + yB)^2 = (2 * xA - xB)^2 + (2 * yA - yB)^2

/-- The main theorem -/
theorem ellipse_intersection_theorem :
  ∀ (k m : ℝ),
    (∃ (xA yA xB yB : ℝ),
      ellipse_C xA yA ∧ ellipse_C xB yB ∧
      line_l k m xA yA ∧ line_l k m xB yB ∧
      intersection_condition xA yA xB yB) ↔
    (m < -Real.sqrt 3 / 2 ∨ m > Real.sqrt 3 / 2) :=
sorry

end NUMINAMATH_CALUDE_ellipse_intersection_theorem_l1894_189492


namespace NUMINAMATH_CALUDE_other_number_l1894_189475

theorem other_number (x : ℝ) : x + 0.525 = 0.650 → x = 0.125 := by
  sorry

end NUMINAMATH_CALUDE_other_number_l1894_189475


namespace NUMINAMATH_CALUDE_soda_cost_calculation_l1894_189490

theorem soda_cost_calculation (total_cost sandwich_cost : ℚ) 
  (num_sandwiches num_sodas : ℕ) :
  total_cost = 6.46 →
  sandwich_cost = 1.49 →
  num_sandwiches = 2 →
  num_sodas = 4 →
  (total_cost - (↑num_sandwiches * sandwich_cost)) / ↑num_sodas = 0.87 := by
  sorry

end NUMINAMATH_CALUDE_soda_cost_calculation_l1894_189490


namespace NUMINAMATH_CALUDE_real_estate_pricing_l1894_189423

theorem real_estate_pricing (retail_price : ℝ) (retail_price_pos : retail_price > 0) :
  let z_price := retail_price * (1 - 0.3)
  let x_price := z_price * (1 - 0.15)
  let y_price := ((z_price + x_price) / 2) * (1 - 0.4)
  y_price / x_price = 0.653 := by
sorry

end NUMINAMATH_CALUDE_real_estate_pricing_l1894_189423


namespace NUMINAMATH_CALUDE_repunit_primes_upper_bound_l1894_189410

def repunit (k : ℕ) : ℕ := (10^k - 1) / 9

def is_repunit_prime (n : ℕ) : Prop :=
  Nat.Prime n ∧ ∃ k, repunit k = n

theorem repunit_primes_upper_bound :
  (∃ (S : Finset ℕ), ∀ n ∈ S, is_repunit_prime n ∧ n < 10^29) →
  (∃ (S : Finset ℕ), ∀ n ∈ S, is_repunit_prime n ∧ n < 10^29 ∧ S.card ≤ 9) :=
sorry

end NUMINAMATH_CALUDE_repunit_primes_upper_bound_l1894_189410


namespace NUMINAMATH_CALUDE_system_solutions_l1894_189452

/-- The system of equations -/
def system (x y : ℝ) : Prop :=
  y^2 = x^3 - 3*x^2 + 2*x ∧ x^2 = y^3 - 3*y^2 + 2*y

/-- The set of solutions -/
def solutions : Set (ℝ × ℝ) :=
  {(0, 0), (2 + Real.sqrt 2, 2 + Real.sqrt 2), (2 - Real.sqrt 2, 2 - Real.sqrt 2)}

/-- Theorem stating that the solutions are correct and complete -/
theorem system_solutions :
  ∀ (x y : ℝ), system x y ↔ (x, y) ∈ solutions :=
sorry

end NUMINAMATH_CALUDE_system_solutions_l1894_189452


namespace NUMINAMATH_CALUDE_circle_properties_l1894_189479

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 6*x + 5 = 0

-- Define the center of the circle
def circle_center : ℝ × ℝ := (3, 0)

-- Define the radius of the circle
def circle_radius : ℝ := 2

-- Theorem statement
theorem circle_properties :
  ∀ x y : ℝ, circle_equation x y ↔ 
    ((x - circle_center.1)^2 + (y - circle_center.2)^2 = circle_radius^2) :=
by sorry

end NUMINAMATH_CALUDE_circle_properties_l1894_189479


namespace NUMINAMATH_CALUDE_number_divided_by_quarter_l1894_189445

theorem number_divided_by_quarter : ∀ x : ℝ, x / 0.25 = 400 → x = 100 := by
  sorry

end NUMINAMATH_CALUDE_number_divided_by_quarter_l1894_189445


namespace NUMINAMATH_CALUDE_building_area_theorem_l1894_189472

/-- Represents a rectangular building with three floors -/
structure Building where
  breadth : ℝ
  length : ℝ
  area_per_floor : ℝ

/-- Calculates the total painting cost for the building -/
def total_painting_cost (b : Building) : ℝ :=
  b.area_per_floor * (3 + 4 + 5)

/-- Theorem: If the length is 200% more than the breadth and the total painting cost is 3160,
    then the total area of the building is 790 sq m -/
theorem building_area_theorem (b : Building) :
  b.length = 3 * b.breadth →
  total_painting_cost b = 3160 →
  3 * b.area_per_floor = 790 :=
by
  sorry

#check building_area_theorem

end NUMINAMATH_CALUDE_building_area_theorem_l1894_189472


namespace NUMINAMATH_CALUDE_sequence_exists_l1894_189458

def is_valid_sequence (seq : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, seq n + seq (n + 1) + seq (n + 2) = 15

def is_repeating (seq : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, seq n = seq (n + 3)

theorem sequence_exists : ∃ seq : ℕ → ℕ, is_valid_sequence seq ∧ is_repeating seq :=
sorry

end NUMINAMATH_CALUDE_sequence_exists_l1894_189458


namespace NUMINAMATH_CALUDE_periodic_function_value_l1894_189407

/-- A function satisfying the given conditions -/
def periodic_function (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f (x + 2) * f x = 1) ∧ f 2 = 2

/-- Theorem stating the value of f(2016) given the conditions -/
theorem periodic_function_value (f : ℝ → ℝ) (h : periodic_function f) : f 2016 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_periodic_function_value_l1894_189407


namespace NUMINAMATH_CALUDE_distance_to_axes_l1894_189414

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the distance functions
def distToXAxis (p : Point2D) : ℝ := |p.y|
def distToYAxis (p : Point2D) : ℝ := |p.x|

-- State the theorem
theorem distance_to_axes (Q : Point2D) (hx : Q.x = -6) (hy : Q.y = 5) :
  distToXAxis Q = 5 ∧ distToYAxis Q = 6 := by
  sorry


end NUMINAMATH_CALUDE_distance_to_axes_l1894_189414


namespace NUMINAMATH_CALUDE_committee_with_chair_count_l1894_189449

theorem committee_with_chair_count : 
  let total_students : ℕ := 8
  let committee_size : ℕ := 5
  let committee_count : ℕ := Nat.choose total_students committee_size
  let chair_choices : ℕ := committee_size
  committee_count * chair_choices = 280 := by
sorry

end NUMINAMATH_CALUDE_committee_with_chair_count_l1894_189449


namespace NUMINAMATH_CALUDE_sum_of_cubes_up_to_8_l1894_189482

/-- Sum of cubes from 1³ to n³ equals the square of the sum of first n natural numbers -/
def sum_of_cubes (n : ℕ) : ℕ := (n * (n + 1) / 2) ^ 2

/-- The sum of cubes from 1³ to 8³ is 1296 -/
theorem sum_of_cubes_up_to_8 : sum_of_cubes 8 = 1296 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_up_to_8_l1894_189482


namespace NUMINAMATH_CALUDE_min_value_expression_l1894_189440

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (((x^2 + y^2) * (4*x^2 + y^2)).sqrt) / (x*y) ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l1894_189440


namespace NUMINAMATH_CALUDE_min_value_sum_product_l1894_189419

theorem min_value_sum_product (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a + b + c + d) * (1 / (a + b) + 1 / (a + c) + 1 / (b + d) + 1 / (c + d)) ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_product_l1894_189419


namespace NUMINAMATH_CALUDE_smallest_fourth_number_l1894_189484

/-- Sum of digits of a positive integer -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Theorem: The smallest two-digit positive integer that satisfies the given condition -/
theorem smallest_fourth_number : 
  ∃ x : ℕ, 
    x ≥ 10 ∧ x < 100 ∧ 
    (∀ y : ℕ, y ≥ 10 ∧ y < 100 → 
      sumOfDigits 28 + sumOfDigits 46 + sumOfDigits 59 + sumOfDigits y = (28 + 46 + 59 + y) / 4 →
      x ≤ y) ∧
    sumOfDigits 28 + sumOfDigits 46 + sumOfDigits 59 + sumOfDigits x = (28 + 46 + 59 + x) / 4 ∧
    x = 11 := by
  sorry

end NUMINAMATH_CALUDE_smallest_fourth_number_l1894_189484


namespace NUMINAMATH_CALUDE_positive_X_value_l1894_189480

-- Define the # relation
def hash (X Y : ℝ) : ℝ := X^2 + Y^2

-- Theorem statement
theorem positive_X_value (X : ℝ) :
  (hash X 7 = 170) → (X = 11 ∨ X = -11) :=
by sorry

end NUMINAMATH_CALUDE_positive_X_value_l1894_189480


namespace NUMINAMATH_CALUDE_function_always_positive_l1894_189450

theorem function_always_positive (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, 2 * f x + x * deriv f x > 0) : 
  ∀ x, f x > 0 := by sorry

end NUMINAMATH_CALUDE_function_always_positive_l1894_189450


namespace NUMINAMATH_CALUDE_perpendicular_diagonals_not_sufficient_for_rhombus_l1894_189425

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define diagonals of a quadrilateral
def diagonals (q : Quadrilateral) : (ℝ × ℝ) × (ℝ × ℝ) :=
  ((q.A.1 - q.C.1, q.A.2 - q.C.2), (q.B.1 - q.D.1, q.B.2 - q.D.2))

-- Define perpendicularity of two vectors
def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

-- Define a rhombus
def is_rhombus (q : Quadrilateral) : Prop :=
  let (AC, BD) := diagonals q
  perpendicular AC BD ∧ 
  AC.1^2 + AC.2^2 = BD.1^2 + BD.2^2 ∧
  (AC.1 / 2, AC.2 / 2) = (BD.1 / 2, BD.2 / 2)

-- Statement to prove
theorem perpendicular_diagonals_not_sufficient_for_rhombus :
  ∃ (q : Quadrilateral), 
    (let (AC, BD) := diagonals q; perpendicular AC BD) ∧ 
    ¬is_rhombus q :=
sorry

end NUMINAMATH_CALUDE_perpendicular_diagonals_not_sufficient_for_rhombus_l1894_189425

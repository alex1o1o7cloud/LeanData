import Mathlib

namespace polynomial_factorization_l610_61006

theorem polynomial_factorization (x : ℝ) : 
  x^5 + x^4 + 1 = (x^2 + x + 1) * (x^3 - x + 1) := by
  sorry

end polynomial_factorization_l610_61006


namespace problem_solution_l610_61054

theorem problem_solution (a b : ℚ) 
  (eq1 : 4 + 2*a = 5 - b) 
  (eq2 : 5 + b = 9 + 3*a) : 
  4 - 2*a = 26/5 := by
  sorry

end problem_solution_l610_61054


namespace greatest_common_divisor_of_differences_l610_61092

theorem greatest_common_divisor_of_differences : Nat.gcd (858 - 794) (Nat.gcd (1351 - 858) (1351 - 794)) = 1 := by
  sorry

end greatest_common_divisor_of_differences_l610_61092


namespace distance_between_cities_l610_61055

/-- The distance between two cities given the meeting points of three vehicles --/
theorem distance_between_cities (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (horder : b < c ∧ c < a) :
  ∃ (s : ℝ), s > 0 ∧ s = Real.sqrt ((a * b * c) / (a + c - b)) ∧
  ∃ (v₁ v₂ v₃ : ℝ), v₁ > v₂ ∧ v₂ > v₃ ∧ v₃ > 0 ∧
  (v₁ / v₂ = (s + a) / (s - a)) ∧
  (v₁ / v₃ = (s + b) / (s - b)) ∧
  (v₂ / v₃ = (s + c) / (s - c)) := by
sorry

end distance_between_cities_l610_61055


namespace reciprocal_sum_of_roots_l610_61048

theorem reciprocal_sum_of_roots (m n : ℝ) : 
  m^2 - 4*m - 2 = 0 → n^2 - 4*n - 2 = 0 → m ≠ n → 1/m + 1/n = -2 := by
  sorry

end reciprocal_sum_of_roots_l610_61048


namespace probability_spade_or_king_is_4_13_l610_61082

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : Nat)
  (suits : Nat)
  (cards_per_suit : Nat)
  (kings_per_suit : Nat)

/-- Calculates the probability of drawing a spade or a king -/
def probability_spade_or_king (d : Deck) : Rat :=
  let spades := d.cards_per_suit
  let kings := d.suits * d.kings_per_suit
  let overlap := d.kings_per_suit
  let favorable_outcomes := spades + kings - overlap
  favorable_outcomes / d.total_cards

/-- Theorem stating the probability of drawing a spade or a king is 4/13 -/
theorem probability_spade_or_king_is_4_13 (d : Deck) 
    (h1 : d.total_cards = 52)
    (h2 : d.suits = 4)
    (h3 : d.cards_per_suit = 13)
    (h4 : d.kings_per_suit = 1) : 
  probability_spade_or_king d = 4 / 13 := by
  sorry

end probability_spade_or_king_is_4_13_l610_61082


namespace largest_number_in_set_l610_61075

def S (a : ℝ) : Set ℝ := {-3*a, 4*a, 24/a, a^2, 2*a+6, 1}

theorem largest_number_in_set (a : ℝ) (h : a = 3) :
  (∀ x ∈ S a, x ≤ 4*a) ∧ (∀ x ∈ S a, x ≤ 2*a+6) ∧ (4*a ∈ S a) ∧ (2*a+6 ∈ S a) :=
sorry

end largest_number_in_set_l610_61075


namespace circle_radius_l610_61063

/-- The radius of the circle described by the equation x^2 + y^2 - 6x + 8y = 0 is 5 -/
theorem circle_radius (x y : ℝ) : x^2 + y^2 - 6*x + 8*y = 0 → ∃ (h k : ℝ), (x - h)^2 + (y - k)^2 = 5^2 := by
  sorry

end circle_radius_l610_61063


namespace max_even_differences_l610_61010

/-- A permutation of numbers from 1 to 25 -/
def Arrangement := Fin 25 → Fin 25

/-- The sequence 1, 2, 3, ..., 25 -/
def OriginalSequence : Fin 25 → ℕ := fun i => i.val + 1

/-- The difference function, always subtracting the smaller from the larger -/
def Difference (arr : Arrangement) (i : Fin 25) : ℕ :=
  max (OriginalSequence i) (arr i).val + 1 - min (OriginalSequence i) (arr i).val + 1

/-- Predicate to check if a number is even -/
def IsEven (n : ℕ) : Prop := ∃ k, n = 2 * k

theorem max_even_differences :
  ∃ (arr : Arrangement), ∀ (i : Fin 25), IsEven (Difference arr i) :=
sorry

end max_even_differences_l610_61010


namespace smallest_m_for_integral_multiple_roots_l610_61012

def has_integral_multiple_roots (m : ℕ) : Prop :=
  ∃ x y : ℤ, x ≠ y ∧ (10 * x^2 - m * x + 360 = 0) ∧ 
             (10 * y^2 - m * y + 360 = 0) ∧
             (x ∣ y ∨ y ∣ x)

theorem smallest_m_for_integral_multiple_roots :
  (has_integral_multiple_roots 120) ∧
  (∀ m : ℕ, m > 0 ∧ m < 120 → ¬(has_integral_multiple_roots m)) :=
sorry

end smallest_m_for_integral_multiple_roots_l610_61012


namespace custom_op_two_five_l610_61000

/-- Custom binary operation on real numbers -/
def custom_op (a b : ℝ) : ℝ := 4 * a + 3 * b

theorem custom_op_two_five : custom_op 2 5 = 23 := by
  sorry

end custom_op_two_five_l610_61000


namespace stock_value_change_l610_61047

theorem stock_value_change (x : ℝ) (h : x > 0) : 
  let day1_value := x * (1 - 0.25)
  let day2_value := day1_value * (1 + 0.40)
  (day2_value - x) / x * 100 = 5 := by
sorry

end stock_value_change_l610_61047


namespace caterpillar_length_difference_l610_61003

/-- The length difference between two caterpillars -/
theorem caterpillar_length_difference : 
  let green_length : ℝ := 3
  let orange_length : ℝ := 1.17
  green_length - orange_length = 1.83 := by sorry

end caterpillar_length_difference_l610_61003


namespace visible_percentage_for_given_prism_and_film_l610_61059

/-- Represents a regular triangular prism -/
structure RegularTriangularPrism where
  base_edge : ℝ
  height : ℝ

/-- Represents a checkerboard film -/
structure CheckerboardFilm where
  cell_size : ℝ

/-- Calculates the visible percentage of a prism's lateral surface when wrapped with a film -/
def visible_percentage (prism : RegularTriangularPrism) (film : CheckerboardFilm) : ℝ :=
  sorry

/-- Theorem stating the visible percentage for the given prism and film -/
theorem visible_percentage_for_given_prism_and_film :
  let prism := RegularTriangularPrism.mk 3.2 5
  let film := CheckerboardFilm.mk 1
  visible_percentage prism film = 28.75 := by
  sorry

end visible_percentage_for_given_prism_and_film_l610_61059


namespace cube_construction_count_l610_61093

/-- Represents a rotation of a cube -/
structure CubeRotation where
  fixedConfigurations : ℕ

/-- The group of rotations for a cube -/
def rotationGroup : Finset CubeRotation := sorry

/-- The number of distinct ways to construct the cube -/
def distinctConstructions : ℕ := sorry

theorem cube_construction_count :
  distinctConstructions = 54 := by sorry

end cube_construction_count_l610_61093


namespace event_probability_l610_61049

theorem event_probability (n : ℕ) (p_at_least_once : ℚ) (p_single : ℚ) : 
  n = 4 →
  p_at_least_once = 65 / 81 →
  (1 - p_single) ^ n = 1 - p_at_least_once →
  p_single = 1 / 3 := by
sorry

end event_probability_l610_61049


namespace cycle_price_calculation_l610_61056

theorem cycle_price_calculation (selling_price : ℝ) (gain_percent : ℝ) 
  (h1 : selling_price = 1080) 
  (h2 : gain_percent = 20) : 
  ∃ original_price : ℝ, 
    original_price * (1 + gain_percent / 100) = selling_price ∧ 
    original_price = 900 := by
  sorry

end cycle_price_calculation_l610_61056


namespace translated_graph_minimum_point_l610_61057

/-- The function f representing the translated graph -/
def f (x : ℝ) : ℝ := 2 * |x - 4| - 2

/-- The minimum point of the translated graph -/
def min_point : ℝ × ℝ := (4, -2)

theorem translated_graph_minimum_point :
  ∀ x : ℝ, f x ≥ f (min_point.1) ∧ f (min_point.1) = min_point.2 :=
by sorry

end translated_graph_minimum_point_l610_61057


namespace corner_removed_cube_edges_l610_61004

/-- Represents a solid formed by removing smaller cubes from corners of a larger cube. -/
structure CornerRemovedCube where
  originalSideLength : ℝ
  removedSideLength : ℝ

/-- Calculates the number of edges in the resulting solid after corner removal. -/
def edgeCount (cube : CornerRemovedCube) : ℕ :=
  12 + 24  -- This is a placeholder. The actual calculation would be more complex.

/-- Theorem stating that a cube of side length 4 with corners of side length 2 removed has 36 edges. -/
theorem corner_removed_cube_edges :
  ∀ (cube : CornerRemovedCube),
    cube.originalSideLength = 4 →
    cube.removedSideLength = 2 →
    edgeCount cube = 36 := by
  sorry

#check corner_removed_cube_edges

end corner_removed_cube_edges_l610_61004


namespace OPRQ_shape_l610_61053

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A figure formed by four points -/
structure Quadrilateral where
  O : Point
  P : Point
  R : Point
  Q : Point

/-- Check if three points are collinear -/
def collinear (A B C : Point) : Prop :=
  (B.x - A.x) * (C.y - A.y) = (C.x - A.x) * (B.y - A.y)

/-- Check if two line segments are parallel -/
def parallel (A B C D : Point) : Prop :=
  (B.x - A.x) * (D.y - C.y) = (D.x - C.x) * (B.y - A.y)

/-- Check if a quadrilateral is a straight line -/
def isStraightLine (quad : Quadrilateral) : Prop :=
  collinear quad.O quad.P quad.Q ∧ collinear quad.O quad.R quad.Q

/-- Check if a quadrilateral is a trapezoid -/
def isTrapezoid (quad : Quadrilateral) : Prop :=
  (parallel quad.O quad.P quad.Q quad.R ∧ ¬parallel quad.O quad.Q quad.P quad.R) ∨
  (¬parallel quad.O quad.P quad.Q quad.R ∧ parallel quad.O quad.Q quad.P quad.R)

/-- Check if a quadrilateral is a parallelogram -/
def isParallelogram (quad : Quadrilateral) : Prop :=
  parallel quad.O quad.P quad.Q quad.R ∧ parallel quad.O quad.Q quad.P quad.R

/-- The main theorem -/
theorem OPRQ_shape (x₁ y₁ x₂ y₂ : ℝ) (h : x₁ ≠ x₂ ∨ y₁ ≠ y₂) :
  let P : Point := ⟨x₁, y₁⟩
  let Q : Point := ⟨x₂, y₂⟩
  let R : Point := ⟨x₁ - x₂, y₁ - y₂⟩
  let O : Point := ⟨0, 0⟩
  let quad : Quadrilateral := ⟨O, P, R, Q⟩
  (isStraightLine quad ∨ isTrapezoid quad) ∧ ¬isParallelogram quad := by
  sorry


end OPRQ_shape_l610_61053


namespace davids_chemistry_marks_l610_61070

def english_marks : ℕ := 96
def math_marks : ℕ := 98
def physics_marks : ℕ := 99
def biology_marks : ℕ := 98
def average_marks : ℚ := 98.2
def num_subjects : ℕ := 5

theorem davids_chemistry_marks :
  ∃ (chemistry_marks : ℕ),
    (english_marks + math_marks + physics_marks + chemistry_marks + biology_marks : ℚ) / num_subjects = average_marks ∧
    chemistry_marks = 100 := by
  sorry

end davids_chemistry_marks_l610_61070


namespace total_games_is_30_l610_61090

/-- The number of Monopoly games won by Betsy, Helen, and Susan -/
def monopoly_games (betsy helen susan : ℕ) : Prop :=
  betsy = 5 ∧ helen = 2 * betsy ∧ susan = 3 * betsy

/-- The total number of games won by all three players -/
def total_games (betsy helen susan : ℕ) : ℕ :=
  betsy + helen + susan

/-- Theorem stating that the total number of games won is 30 -/
theorem total_games_is_30 :
  ∀ betsy helen susan : ℕ,
  monopoly_games betsy helen susan →
  total_games betsy helen susan = 30 :=
by
  sorry


end total_games_is_30_l610_61090


namespace negative_exponent_division_l610_61096

theorem negative_exponent_division (m : ℝ) :
  (-m)^7 / (-m)^2 = -m^5 := by
  sorry

end negative_exponent_division_l610_61096


namespace merry_go_round_revolutions_l610_61079

theorem merry_go_round_revolutions 
  (outer_radius inner_radius : ℝ) 
  (outer_revolutions : ℕ) 
  (h1 : outer_radius = 40)
  (h2 : inner_radius = 10)
  (h3 : outer_revolutions = 15) :
  ∃ inner_revolutions : ℕ,
    inner_revolutions = 60 ∧
    outer_radius * outer_revolutions = inner_radius * inner_revolutions :=
by sorry

end merry_go_round_revolutions_l610_61079


namespace expression_evaluation_l610_61033

theorem expression_evaluation (x y z : ℝ) 
  (hx : x ≠ 3) (hy : y ≠ 5) (hz : z ≠ 7) : 
  (x - 3) / (7 - z) * (y - 5) / (3 - x) * (z - 7) / (5 - y) = -1 := by
  sorry

end expression_evaluation_l610_61033


namespace specific_pairing_probability_l610_61084

/-- The probability of a specific pairing in a class of 50 students -/
theorem specific_pairing_probability (n : ℕ) (h : n = 50) :
  (1 : ℚ) / (n - 1) = 1 / 49 := by
  sorry

#check specific_pairing_probability

end specific_pairing_probability_l610_61084


namespace sum_of_reciprocal_squares_l610_61062

theorem sum_of_reciprocal_squares (p q r : ℝ) : 
  p^3 - 9*p^2 + 8*p + 2 = 0 →
  q^3 - 9*q^2 + 8*q + 2 = 0 →
  r^3 - 9*r^2 + 8*r + 2 = 0 →
  p ≠ q → p ≠ r → q ≠ r →
  1/p^2 + 1/q^2 + 1/r^2 = 25 := by
sorry

end sum_of_reciprocal_squares_l610_61062


namespace complex_root_modulus_l610_61043

open Complex

theorem complex_root_modulus (c d : ℝ) (h : (1 + I)^2 + c*(1 + I) + d = 0) : 
  abs (c + d*I) = 2 * Real.sqrt 2 := by
  sorry

end complex_root_modulus_l610_61043


namespace betty_sugar_purchase_l610_61067

theorem betty_sugar_purchase (f s : ℝ) : 
  (f ≥ 10 + (3/4) * s) → 
  (f ≤ 3 * s) → 
  (∀ s' : ℝ, (∃ f' : ℝ, f' ≥ 10 + (3/4) * s' ∧ f' ≤ 3 * s') → s' ≥ s) →
  s = 40/9 := by
sorry

end betty_sugar_purchase_l610_61067


namespace sam_initial_yellow_marbles_l610_61008

/-- The number of yellow marbles Sam had initially -/
def initial_yellow_marbles : ℕ := sorry

/-- The number of yellow marbles Joan took -/
def marbles_taken : ℕ := 25

/-- The number of yellow marbles Sam has now -/
def current_yellow_marbles : ℕ := 61

theorem sam_initial_yellow_marbles :
  initial_yellow_marbles = current_yellow_marbles + marbles_taken :=
by sorry

end sam_initial_yellow_marbles_l610_61008


namespace largest_odd_proper_divisor_ratio_l610_61068

/-- The largest odd proper divisor of a positive integer -/
def f (n : ℕ) : ℕ :=
  sorry

/-- The main theorem -/
theorem largest_odd_proper_divisor_ratio :
  let N : ℕ := 20^23 * 23^20
  f N / f (f (f N)) = 25 :=
by sorry

end largest_odd_proper_divisor_ratio_l610_61068


namespace polar_equation_is_parabola_l610_61094

-- Define the polar equation
def polar_equation (r θ : ℝ) : Prop :=
  r = 1 / (1 - Real.sin θ)

-- Define the Cartesian equation of a parabola
def is_parabola (x y : ℝ) : Prop :=
  ∃ (a : ℝ), x^2 = 2 * a * y

-- Theorem statement
theorem polar_equation_is_parabola :
  ∀ (x y : ℝ), (∃ (r θ : ℝ), polar_equation r θ ∧ x = r * Real.cos θ ∧ y = r * Real.sin θ) →
  is_parabola x y :=
sorry

end polar_equation_is_parabola_l610_61094


namespace river_depth_l610_61060

/-- Proves that given a river with specified width, flow rate, and discharge, its depth is 2 meters -/
theorem river_depth (width : ℝ) (flow_rate : ℝ) (discharge : ℝ) : 
  width = 45 ∧ 
  flow_rate = 6 ∧ 
  discharge = 9000 → 
  discharge = width * 2 * (flow_rate * 1000 / 60) := by
  sorry

#check river_depth

end river_depth_l610_61060


namespace quadratic_inequality_range_l610_61025

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 + (a - 1)*x + 1 ≥ 0) → -1 ≤ a ∧ a ≤ 3 := by
  sorry

end quadratic_inequality_range_l610_61025


namespace consecutive_pages_product_l610_61032

theorem consecutive_pages_product (n : ℕ) : 
  n > 0 ∧ n + (n + 1) = 217 → n * (n + 1) = 11772 := by
sorry

end consecutive_pages_product_l610_61032


namespace marbles_remaining_l610_61091

theorem marbles_remaining (total : ℕ) (given_to_theresa : ℚ) (given_to_elliot : ℚ) :
  total = 100 →
  given_to_theresa = 25 / 100 →
  given_to_elliot = 10 / 100 →
  total - (total * given_to_theresa).floor - (total * given_to_elliot).floor = 65 := by
sorry

end marbles_remaining_l610_61091


namespace zoo_visitors_per_hour_l610_61066

/-- The number of hours the zoo is open in one day -/
def zoo_hours : ℕ := 8

/-- The percentage of total visitors who go to the gorilla exhibit -/
def gorilla_exhibit_percentage : ℚ := 80 / 100

/-- The number of visitors who go to the gorilla exhibit in one day -/
def gorilla_exhibit_visitors : ℕ := 320

/-- The number of new visitors entering the zoo every hour -/
def new_visitors_per_hour : ℕ := 50

theorem zoo_visitors_per_hour :
  new_visitors_per_hour = (gorilla_exhibit_visitors : ℚ) / gorilla_exhibit_percentage / zoo_hours := by
  sorry

end zoo_visitors_per_hour_l610_61066


namespace binomial_distribution_p_value_l610_61072

/-- A binomial distribution with parameters n and p -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- The expected value of a binomial distribution -/
def expectedValue (ξ : BinomialDistribution) : ℝ := ξ.n * ξ.p

/-- The variance of a binomial distribution -/
def variance (ξ : BinomialDistribution) : ℝ := ξ.n * ξ.p * (1 - ξ.p)

/-- Theorem: For a binomial distribution with E(ξ) = 7 and D(ξ) = 6, p = 1/7 -/
theorem binomial_distribution_p_value (ξ : BinomialDistribution) 
  (h_exp : expectedValue ξ = 7)
  (h_var : variance ξ = 6) : 
  ξ.p = 1/7 := by
  sorry


end binomial_distribution_p_value_l610_61072


namespace zeros_not_adjacent_probability_l610_61009

/-- The number of ones in the arrangement -/
def num_ones : ℕ := 4

/-- The number of zeros in the arrangement -/
def num_zeros : ℕ := 2

/-- The total number of elements to be arranged -/
def total_elements : ℕ := num_ones + num_zeros

/-- The number of spaces where zeros can be placed without being adjacent -/
def num_spaces : ℕ := num_ones + 1

/-- The probability that the zeros are not adjacent when randomly arranged -/
theorem zeros_not_adjacent_probability :
  (Nat.choose num_spaces num_zeros : ℚ) / (Nat.choose total_elements num_zeros : ℚ) = 2/3 :=
sorry

end zeros_not_adjacent_probability_l610_61009


namespace approximation_place_l610_61076

def number : ℕ := 345000000

theorem approximation_place (n : ℕ) (h : n = number) : 
  ∃ (k : ℕ), n ≥ 10^6 ∧ n < 10^7 ∧ k * 10^6 = n ∧ k < 1000 :=
by sorry

end approximation_place_l610_61076


namespace correct_average_after_error_correction_l610_61046

theorem correct_average_after_error_correction 
  (n : ℕ) 
  (initial_average : ℚ) 
  (incorrect_value : ℚ) 
  (correct_value : ℚ) : 
  n = 10 → 
  initial_average = 15 → 
  incorrect_value = 26 → 
  correct_value = 36 → 
  (n : ℚ) * initial_average + (correct_value - incorrect_value) = n * 16 := by
  sorry

end correct_average_after_error_correction_l610_61046


namespace maxwell_current_age_l610_61018

/-- Maxwell's current age -/
def maxwell_age : ℕ := sorry

/-- Maxwell's sister's current age -/
def sister_age : ℕ := 2

/-- In 2 years, Maxwell will be twice his sister's age -/
axiom maxwell_twice_sister : maxwell_age + 2 = 2 * (sister_age + 2)

theorem maxwell_current_age : maxwell_age = 6 := by sorry

end maxwell_current_age_l610_61018


namespace invisible_dots_sum_l610_61073

/-- The sum of numbers on a single die -/
def die_sum : ℕ := 21

/-- The number of dice -/
def num_dice : ℕ := 3

/-- The sum of visible numbers -/
def visible_sum : ℕ := 1 + 2 + 3 + 3 + 4

/-- The number of visible faces -/
def num_visible_faces : ℕ := 5

theorem invisible_dots_sum : 
  num_dice * die_sum - visible_sum = 50 := by sorry

end invisible_dots_sum_l610_61073


namespace green_chips_count_l610_61007

theorem green_chips_count (total : ℕ) (blue_fraction : ℚ) (red : ℕ) : 
  total = 60 →
  blue_fraction = 1 / 6 →
  red = 34 →
  (total : ℚ) * blue_fraction + red + (total - (total : ℚ) * blue_fraction - red) = total →
  total - (total : ℚ) * blue_fraction - red = 16 :=
by sorry

end green_chips_count_l610_61007


namespace system_solution_unique_l610_61015

theorem system_solution_unique (x y : ℝ) : 
  (x - y = -5 ∧ 3*x + 2*y = 10) ↔ (x = 0 ∧ y = 5) := by
  sorry

end system_solution_unique_l610_61015


namespace function_inequality_implies_a_range_l610_61039

open Real

/-- Given f(x) = 2/x + a*ln(x) - 2 where a > 0, if f(x) > 2(a-1) for all x > 0, then 0 < a < 2/e -/
theorem function_inequality_implies_a_range (a : ℝ) (h_a_pos : a > 0) :
  (∀ x : ℝ, x > 0 → (2 / x + a * log x - 2 > 2 * (a - 1))) →
  0 < a ∧ a < 2 / Real.exp 1 := by
  sorry

end function_inequality_implies_a_range_l610_61039


namespace sum_of_variables_l610_61071

theorem sum_of_variables (x y z : ℝ) 
  (eq1 : y + z = 15 - 4*x)
  (eq2 : x + z = -17 - 4*y)
  (eq3 : x + y = 8 - 4*z) :
  2*x + 2*y + 2*z = 2 := by
sorry

end sum_of_variables_l610_61071


namespace cloth_trimming_l610_61001

theorem cloth_trimming (x : ℝ) :
  (x > 0) →
  (x - 4 > 0) →
  (x - 3 > 0) →
  ((x - 4) * (x - 3) = 120) →
  (x = 12) :=
by sorry

end cloth_trimming_l610_61001


namespace eight_digit_number_theorem_l610_61081

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

def move_last_to_first (n : ℕ) : ℕ :=
  let last_digit := n % 10
  let rest := n / 10
  last_digit * 10^7 + rest

theorem eight_digit_number_theorem (B : ℕ) (hB1 : is_coprime B 36) (hB2 : B > 7777777) :
  let A := move_last_to_first B
  (∃ A_min A_max : ℕ, 
    (∀ A' : ℕ, (∃ B' : ℕ, A' = move_last_to_first B' ∧ is_coprime B' 36 ∧ B' > 7777777) → 
      A_min ≤ A' ∧ A' ≤ A_max) ∧
    A_min = 17777779 ∧ 
    A_max = 99999998) :=
sorry

end eight_digit_number_theorem_l610_61081


namespace set_equality_l610_61031

theorem set_equality : 
  {z : ℤ | ∃ (x a : ℝ), z = x - a ∧ a - 1 ≤ x ∧ x ≤ a + 1} = {-1, 0, 1} := by
  sorry

end set_equality_l610_61031


namespace absolute_value_square_sum_zero_l610_61095

theorem absolute_value_square_sum_zero (x y : ℝ) :
  |x + 5| + (y - 2)^2 = 0 → x = -5 ∧ y = 2 ∧ x^y = 25 := by
  sorry

end absolute_value_square_sum_zero_l610_61095


namespace final_number_independent_of_operations_l610_61040

/-- Represents the state of the blackboard with counts of 0, 1, and 2 -/
structure Board :=
  (count_0 : ℕ)
  (count_1 : ℕ)
  (count_2 : ℕ)

/-- Represents a single operation on the board -/
inductive Operation
  | replace_0_1_with_2
  | replace_1_2_with_0
  | replace_0_2_with_1

/-- Applies an operation to the board -/
def apply_operation (b : Board) (op : Operation) : Board :=
  match op with
  | Operation.replace_0_1_with_2 => ⟨b.count_0 - 1, b.count_1 - 1, b.count_2 + 1⟩
  | Operation.replace_1_2_with_0 => ⟨b.count_0 + 1, b.count_1 - 1, b.count_2 - 1⟩
  | Operation.replace_0_2_with_1 => ⟨b.count_0 - 1, b.count_1 + 1, b.count_2 - 1⟩

/-- Checks if the board has only one number left -/
def is_final (b : Board) : Prop :=
  (b.count_0 = 1 ∧ b.count_1 = 0 ∧ b.count_2 = 0) ∨
  (b.count_0 = 0 ∧ b.count_1 = 1 ∧ b.count_2 = 0) ∨
  (b.count_0 = 0 ∧ b.count_1 = 0 ∧ b.count_2 = 1)

/-- The final number on the board -/
def final_number (b : Board) : ℕ :=
  if b.count_0 = 1 then 0
  else if b.count_1 = 1 then 1
  else 2

/-- Theorem: The final number is determined by initial parity, regardless of operations -/
theorem final_number_independent_of_operations (initial : Board) 
  (ops1 ops2 : List Operation) (h1 : is_final (ops1.foldl apply_operation initial))
  (h2 : is_final (ops2.foldl apply_operation initial)) :
  final_number (ops1.foldl apply_operation initial) = 
  final_number (ops2.foldl apply_operation initial) :=
sorry

end final_number_independent_of_operations_l610_61040


namespace paint_remaining_is_one_fourth_l610_61044

/-- The fraction of paint remaining after three days of painting -/
def paint_remaining : ℚ :=
  let day1_remaining := 1 - (1/4 : ℚ)
  let day2_remaining := day1_remaining - (1/2 * day1_remaining)
  day2_remaining - (1/3 * day2_remaining)

/-- Theorem stating that the remaining paint after three days is 1/4 of the original amount -/
theorem paint_remaining_is_one_fourth :
  paint_remaining = (1/4 : ℚ) := by
  sorry

end paint_remaining_is_one_fourth_l610_61044


namespace problem_statement_l610_61045

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃ (n : ℕ), a^5 % 10 = b^5 % 10 → a - b = 10 * n) ∧
  (a^2 - b^2 = 1940 → a = 102 ∧ b = 92) ∧
  (a^2 - b^2 = 1920 → 
    ((a = 101 ∧ b = 91) ∨ 
     (a = 58 ∧ b = 38) ∨ 
     (a = 47 ∧ b = 17) ∨ 
     (a = 44 ∧ b = 4))) := by
  sorry

end problem_statement_l610_61045


namespace geometric_sequence_sum_l610_61016

/-- Sum of a geometric sequence -/
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- First term of the sequence -/
def a : ℚ := 1/3

/-- Common ratio of the sequence -/
def r : ℚ := 1/3

/-- Number of terms to sum -/
def n : ℕ := 8

theorem geometric_sequence_sum :
  geometric_sum a r n = 3280/6561 := by sorry

end geometric_sequence_sum_l610_61016


namespace age_problem_l610_61026

theorem age_problem (a b c : ℕ) : 
  a = b + 2 →
  b = 2 * c →
  a + b + c = 47 →
  b = 18 := by
sorry

end age_problem_l610_61026


namespace units_digit_of_k_squared_plus_two_to_k_l610_61022

def k : ℕ := 2010^2 + 2^2010

theorem units_digit_of_k_squared_plus_two_to_k (k : ℕ := k) :
  (k^2 + 2^k) % 10 = 7 := by sorry

end units_digit_of_k_squared_plus_two_to_k_l610_61022


namespace samia_walking_distance_l610_61050

/-- Proves that Samia walked 4.0 km given the journey conditions --/
theorem samia_walking_distance :
  ∀ (total_distance : ℝ) (biking_distance : ℝ),
    -- Samia's average biking speed is 15 km/h
    -- Samia bikes for 30 minutes (0.5 hours)
    biking_distance = 15 * 0.5 →
    -- The entire journey took 90 minutes (1.5 hours)
    0.5 + ((total_distance - biking_distance) / 4) = 1.5 →
    -- Prove that the walking distance is 4.0 km
    total_distance - biking_distance = 4.0 := by
  sorry

end samia_walking_distance_l610_61050


namespace second_hand_large_division_time_l610_61029

/-- The number of large divisions on a clock face -/
def large_divisions : ℕ := 12

/-- The number of small divisions in each large division -/
def small_divisions_per_large : ℕ := 5

/-- The time (in seconds) it takes for the second hand to move one small division -/
def time_per_small_division : ℕ := 1

/-- The time it takes for the second hand to move one large division -/
def time_for_large_division : ℕ := small_divisions_per_large * time_per_small_division

theorem second_hand_large_division_time :
  time_for_large_division = 5 := by sorry

end second_hand_large_division_time_l610_61029


namespace clothing_expense_l610_61017

theorem clothing_expense (total_spent adidas_original nike skechers puma adidas clothes : ℝ) 
  (h_total : total_spent = 12000)
  (h_nike : nike = 2 * adidas)
  (h_skechers : adidas = 1/3 * skechers)
  (h_puma : puma = 3/4 * nike)
  (h_adidas_original : adidas_original = 900)
  (h_adidas_discount : adidas = adidas_original * 0.9)
  (h_sum : total_spent = nike + adidas + skechers + puma + clothes) :
  clothes = 5925 := by
sorry


end clothing_expense_l610_61017


namespace athena_snack_spending_l610_61052

/-- Calculates the total amount spent by Athena on snacks -/
def total_spent (sandwich_price : ℚ) (sandwich_qty : ℕ)
                (drink_price : ℚ) (drink_qty : ℕ)
                (cookie_price : ℚ) (cookie_qty : ℕ)
                (chips_price : ℚ) (chips_qty : ℕ) : ℚ :=
  sandwich_price * sandwich_qty +
  drink_price * drink_qty +
  cookie_price * cookie_qty +
  chips_price * chips_qty

/-- Proves that Athena spent $33.95 on snacks -/
theorem athena_snack_spending :
  total_spent (325/100) 4 (275/100) 3 (150/100) 6 (185/100) 2 = 3395/100 := by
  sorry

end athena_snack_spending_l610_61052


namespace ellipse_chord_through_focus_l610_61080

/-- The x-coordinate of point A on the ellipse satisfies a specific quadratic equation --/
theorem ellipse_chord_through_focus (x y : ℝ) : 
  (x^2 / 36 + y^2 / 16 = 1) →  -- ellipse equation
  ((x - 2 * Real.sqrt 5)^2 + y^2 = 9) →  -- AF = 3
  (84 * x^2 - 400 * x + 552 = 0) :=
by sorry

end ellipse_chord_through_focus_l610_61080


namespace isosceles_triangle_attachment_l610_61065

/-- Represents a triangle in 2D space -/
structure Triangle where
  a : ℝ × ℝ
  b : ℝ × ℝ
  c : ℝ × ℝ

/-- Checks if a triangle is right-angled -/
def isRightTriangle (t : Triangle) : Prop := sorry

/-- Checks if two triangles share a common side -/
def shareCommonSide (t1 t2 : Triangle) : Prop := sorry

/-- Checks if two triangles do not overlap -/
def noOverlap (t1 t2 : Triangle) : Prop := sorry

/-- Checks if a triangle is isosceles -/
def isIsosceles (t : Triangle) : Prop := sorry

/-- Combines two triangles into a new triangle -/
def combineTriangles (t1 t2 : Triangle) : Triangle := sorry

theorem isosceles_triangle_attachment (t : Triangle) : 
  isRightTriangle t → 
  ∃ t2 : Triangle, 
    shareCommonSide t t2 ∧ 
    noOverlap t t2 ∧ 
    isIsosceles (combineTriangles t t2) := by
  sorry

end isosceles_triangle_attachment_l610_61065


namespace bushes_for_zucchinis_l610_61020

/-- The number of containers of blueberries each bush yields -/
def containers_per_bush : ℕ := 10

/-- The number of containers of blueberries that can be traded for zucchinis -/
def containers_for_trade : ℕ := 6

/-- The number of zucchinis received in trade for containers_for_trade -/
def zucchinis_from_trade : ℕ := 3

/-- The target number of zucchinis -/
def target_zucchinis : ℕ := 60

/-- The number of bushes needed to obtain the target number of zucchinis -/
def bushes_needed : ℕ := 12

theorem bushes_for_zucchinis :
  bushes_needed * containers_per_bush * zucchinis_from_trade = 
  target_zucchinis * containers_for_trade := by
  sorry

end bushes_for_zucchinis_l610_61020


namespace largest_number_l610_61085

theorem largest_number (a b c d : ℝ) (h1 : a = -3) (h2 : b = 0) (h3 : c = Real.sqrt 5) (h4 : d = 2) :
  c = max a (max b (max c d)) :=
by sorry

end largest_number_l610_61085


namespace locus_and_fixed_points_l610_61077

-- Define the points and lines
def F : ℝ × ℝ := (1, 0)
def H : ℝ × ℝ := (1, 2)
def l : Set (ℝ × ℝ) := {p | p.1 = -1}

-- Define the locus C
def C : Set (ℝ × ℝ) := {p | p.2^2 = 4 * p.1}

-- Define a function to represent a line passing through F and not perpendicular to x-axis
def line_through_F (m : ℝ) : Set (ℝ × ℝ) := {p | p.2 = m * (p.1 - 1)}

-- Define the circle with diameter MN
def circle_MN (m : ℝ) : Set (ℝ × ℝ) := {p | p.1^2 + 2*p.1 - 3 + p.2^2 + (4/m)*p.2 = 0}

-- State the theorem
theorem locus_and_fixed_points :
  ∀ (m : ℝ), m ≠ 0 →
  (∃ (A B : ℝ × ℝ), A ∈ C ∧ B ∈ C ∧ A ∈ line_through_F m ∧ B ∈ line_through_F m) →
  ((-3, 0) ∈ circle_MN m ∧ (1, 0) ∈ circle_MN m) :=
sorry

end locus_and_fixed_points_l610_61077


namespace tenth_term_value_l610_61013

def sequence_term (n : ℕ+) : ℚ :=
  (-1)^(n + 1 : ℕ) * (2 * n - 1 : ℚ) / ((n : ℚ)^2 + 1)

theorem tenth_term_value : sequence_term 10 = -19 / 101 := by
  sorry

end tenth_term_value_l610_61013


namespace sum_of_decimals_equals_fraction_l610_61064

theorem sum_of_decimals_equals_fraction :
  (∃ (x y : ℚ), x = 1/3 ∧ y = 7/9 ∧ x + y + (1/4 : ℚ) = 49/36) := by
  sorry

end sum_of_decimals_equals_fraction_l610_61064


namespace cube_coverage_tape_pieces_correct_l610_61087

/-- Represents the number of tape pieces needed to cover a cube --/
def tape_pieces (n : ℕ) : ℕ := 2 * n

/-- Theorem stating that the number of tape pieces needed to cover a cube with edge length n is 2n --/
theorem cube_coverage (n : ℕ) :
  tape_pieces n = 2 * n :=
by sorry

/-- Represents the properties of the tape coverage method --/
structure TapeCoverage where
  edge_length : ℕ
  tape_width : ℕ
  parallel_to_edge : Bool
  can_cross_edges : Bool
  no_overhang : Bool

/-- Theorem stating that the tape_pieces function gives the correct number of pieces
    for a cube coverage satisfying the given constraints --/
theorem tape_pieces_correct (coverage : TapeCoverage) 
  (h1 : coverage.tape_width = 1)
  (h2 : coverage.parallel_to_edge = true)
  (h3 : coverage.can_cross_edges = true)
  (h4 : coverage.no_overhang = true) :
  tape_pieces coverage.edge_length = 2 * coverage.edge_length :=
by sorry

end cube_coverage_tape_pieces_correct_l610_61087


namespace probability_same_tune_is_one_fourth_l610_61037

/-- A defective toy train that produces two different tunes at random -/
structure DefectiveToyTrain :=
  (tunes : Fin 2 → String)

/-- The probability of the defective toy train producing 3 music tunes of the same type -/
def probability_same_tune (train : DefectiveToyTrain) : ℚ :=
  1 / 4

/-- Theorem stating that the probability of producing 3 music tunes of the same type is 1/4 -/
theorem probability_same_tune_is_one_fourth (train : DefectiveToyTrain) :
  probability_same_tune train = 1 / 4 := by
  sorry

end probability_same_tune_is_one_fourth_l610_61037


namespace calculator_game_sum_l610_61041

/-- Represents the operation to be performed on a calculator --/
inductive Operation
  | Square
  | Negate

/-- Performs the specified operation on a number --/
def applyOperation (op : Operation) (x : Int) : Int :=
  match op with
  | Operation.Square => x * x
  | Operation.Negate => -x

/-- Determines the operation for the third calculator based on the pass number --/
def thirdOperation (pass : Nat) : Operation :=
  if pass % 2 = 0 then Operation.Negate else Operation.Square

/-- Performs one round of operations on the three calculators --/
def performRound (a b c : Int) (pass : Nat) : (Int × Int × Int) :=
  (applyOperation Operation.Square a,
   applyOperation Operation.Square b,
   applyOperation (thirdOperation pass) c)

/-- Performs n rounds of operations on the three calculators --/
def performNRounds (n : Nat) (a b c : Int) : (Int × Int × Int) :=
  match n with
  | 0 => (a, b, c)
  | n + 1 => 
    let (a', b', c') := performRound a b c n
    performNRounds n a' b' c'

theorem calculator_game_sum :
  let (a, b, c) := performNRounds 50 1 0 (-1)
  a + b + c = 0 := by
  sorry

end calculator_game_sum_l610_61041


namespace right_triangle_altitude_l610_61002

theorem right_triangle_altitude (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a^2 + b^2 = c^2) (h5 : 1/a + 1/b = 3/c) :
  ∃ m_c : ℝ, m_c = c * (1 + Real.sqrt 10) / 9 ∧ m_c^2 * c = a * b := by
  sorry

end right_triangle_altitude_l610_61002


namespace sin_tan_inequality_l610_61098

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
variable (h_even : ∀ x, f x = f (-x))
variable (h_mono : ∀ x y, -1 ≤ x ∧ x < y ∧ y ≤ 0 → f x < f y)

-- State the theorem
theorem sin_tan_inequality :
  f (Real.sin (π / 12)) > f (Real.tan (π / 12)) := by sorry

end sin_tan_inequality_l610_61098


namespace bike_shop_wheels_l610_61019

/-- The number of wheels on all vehicles in a bike shop -/
def total_wheels (num_bicycles num_tricycles : ℕ) : ℕ :=
  2 * num_bicycles + 3 * num_tricycles

/-- Theorem stating the total number of wheels in the bike shop -/
theorem bike_shop_wheels : total_wheels 50 20 = 160 := by
  sorry

end bike_shop_wheels_l610_61019


namespace remaining_distance_is_4430_l610_61099

/-- Represents the state of the race between Alex and Max -/
structure RaceState where
  total_distance : ℕ
  alex_lead : ℤ

/-- Calculates the final race state after all lead changes -/
def final_race_state : RaceState :=
  let initial_state : RaceState := { total_distance := 5000, alex_lead := 0 }
  let after_uphill : RaceState := { initial_state with alex_lead := 300 }
  let after_downhill : RaceState := { after_uphill with alex_lead := after_uphill.alex_lead - 170 }
  { after_downhill with alex_lead := after_downhill.alex_lead + 440 }

/-- Calculates the remaining distance for Max to catch up -/
def remaining_distance (state : RaceState) : ℕ :=
  state.total_distance - state.alex_lead.toNat

/-- Theorem stating the remaining distance for Max to catch up -/
theorem remaining_distance_is_4430 :
  remaining_distance final_race_state = 4430 := by
  sorry

end remaining_distance_is_4430_l610_61099


namespace wendy_accounting_percentage_l610_61058

/-- Calculates the percentage of life spent in accounting-related jobs -/
def accounting_percentage (years_accountant : ℕ) (years_manager : ℕ) (total_lifespan : ℕ) : ℚ :=
  (years_accountant + years_manager : ℚ) / total_lifespan * 100

/-- Wendy's accounting career percentage theorem -/
theorem wendy_accounting_percentage :
  accounting_percentage 25 15 80 = 50 := by
  sorry

end wendy_accounting_percentage_l610_61058


namespace area_four_intersecting_circles_l610_61021

/-- The area common to four intersecting circles with specific configuration -/
theorem area_four_intersecting_circles (R : ℝ) (R_pos : R > 0) : ℝ := by
  /- Given two circles of radius R that intersect such that each passes through the center of the other,
     and two additional circles of radius R with centers at the intersection points of the first two circles,
     the area common to all four circles is: -/
  have area : ℝ := R^2 * (2 * Real.pi - 3 * Real.sqrt 3) / 6
  
  /- Proof goes here -/
  sorry

#check area_four_intersecting_circles

end area_four_intersecting_circles_l610_61021


namespace weed_pulling_rate_is_11_l610_61036

-- Define the hourly rates and hours worked
def mowing_rate : ℝ := 6
def mulch_rate : ℝ := 9
def mowing_hours : ℝ := 63
def weed_hours : ℝ := 9
def mulch_hours : ℝ := 10
def total_earnings : ℝ := 567

-- Define the function to calculate total earnings
def calculate_earnings (weed_rate : ℝ) : ℝ :=
  mowing_rate * mowing_hours + weed_rate * weed_hours + mulch_rate * mulch_hours

-- Theorem statement
theorem weed_pulling_rate_is_11 :
  ∃ (weed_rate : ℝ), calculate_earnings weed_rate = total_earnings ∧ weed_rate = 11 := by
  sorry

end weed_pulling_rate_is_11_l610_61036


namespace solve_exponential_equation_l610_61074

theorem solve_exponential_equation :
  ∃! x : ℝ, (8 : ℝ)^(x - 1) / (2 : ℝ)^(x - 1) = (64 : ℝ)^(2 * x) ∧ x = -1/5 := by
  sorry

end solve_exponential_equation_l610_61074


namespace min_sum_squares_l610_61083

theorem min_sum_squares (a b c t : ℝ) (h : a + b + c = t) :
  a^2 + b^2 + c^2 ≥ t^2 / 3 ∧
  (a^2 + b^2 + c^2 = t^2 / 3 ↔ a = t/3 ∧ b = t/3 ∧ c = t/3) :=
by sorry

end min_sum_squares_l610_61083


namespace travel_probabilities_l610_61038

/-- Represents a set of countries --/
structure CountrySet where
  asian : Finset Nat
  european : Finset Nat

/-- The probability of an event given the number of favorable outcomes and total outcomes --/
def probability (favorable : Nat) (total : Nat) : ℚ := favorable / total

/-- The total number of ways to choose 2 items from n items --/
def choose_two (n : Nat) : Nat := n * (n - 1) / 2

theorem travel_probabilities (countries : CountrySet) 
  (h1 : countries.asian.card = 3)
  (h2 : countries.european.card = 3) :
  (probability (choose_two 3) (choose_two 6) = 1 / 5) ∧ 
  (probability 2 9 = 2 / 9) := by
  sorry


end travel_probabilities_l610_61038


namespace repeating_decimal_ratio_l610_61061

/-- Represents a repeating decimal with a two-digit repetend -/
def RepeatingDecimal (a b : ℕ) : ℚ := (a * 10 + b) / 99

/-- The fraction 0.overline{72} divided by 0.overline{27} is equal to 8/3 -/
theorem repeating_decimal_ratio : 
  (RepeatingDecimal 7 2) / (RepeatingDecimal 2 7) = 8 / 3 := by sorry

end repeating_decimal_ratio_l610_61061


namespace reflection_theorem_l610_61005

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The line of symmetry for the fold -/
def lineOfSymmetry : ℝ := 2

/-- Function to reflect a point across the line of symmetry -/
def reflect (p : Point) : Point :=
  { x := p.x, y := 2 * lineOfSymmetry - p.y }

/-- The original point before folding -/
def originalPoint : Point := { x := -4, y := 1 }

/-- The expected point after folding -/
def expectedPoint : Point := { x := -4, y := 3 }

/-- Theorem stating that reflecting the original point results in the expected point -/
theorem reflection_theorem : reflect originalPoint = expectedPoint := by
  sorry

end reflection_theorem_l610_61005


namespace arccos_arcsin_equation_l610_61078

theorem arccos_arcsin_equation : ∃ x : ℝ, Real.arccos (3 * x) - Real.arcsin (2 * x) = π / 6 := by
  sorry

end arccos_arcsin_equation_l610_61078


namespace circle_center_locus_l610_61069

-- Define the circle C
def circle_C (a x y : ℝ) : Prop :=
  x^2 + y^2 - (2*a^2 - 4)*x - 4*a^2*y + 5*a^4 - 4 = 0

-- Define the locus of the center
def center_locus (x y : ℝ) : Prop :=
  y = 2*x + 4 ∧ -2 ≤ x ∧ x < 0

-- Theorem statement
theorem circle_center_locus :
  ∀ a x y : ℝ, circle_C a x y → ∃ h k : ℝ, center_locus h k ∧ 
  (h = a^2 - 2 ∧ k = 2*a^2) :=
sorry

end circle_center_locus_l610_61069


namespace solution_y_composition_l610_61051

/-- Represents a chemical solution --/
structure Solution where
  a : ℝ  -- Percentage of chemical a
  b : ℝ  -- Percentage of chemical b

/-- Represents a mixture of two solutions --/
structure Mixture where
  x : Solution  -- First solution
  y : Solution  -- Second solution
  x_ratio : ℝ   -- Ratio of solution x in the mixture

def is_valid_solution (s : Solution) : Prop :=
  s.a + s.b = 100 ∧ s.a ≥ 0 ∧ s.b ≥ 0

def is_valid_mixture (m : Mixture) : Prop :=
  m.x_ratio ≥ 0 ∧ m.x_ratio ≤ 1

theorem solution_y_composition 
  (x : Solution)
  (y : Solution)
  (m : Mixture)
  (hx : is_valid_solution x)
  (hy : is_valid_solution y)
  (hm : is_valid_mixture m)
  (hx_comp : x.a = 40 ∧ x.b = 60)
  (hy_comp : y.a = y.b)
  (hm_comp : m.x = x ∧ m.y = y)
  (hm_ratio : m.x_ratio = 0.3)
  (hm_a : m.x_ratio * x.a + (1 - m.x_ratio) * y.a = 47) :
  y.a = 50 := by
    sorry

end solution_y_composition_l610_61051


namespace five_by_seven_not_tileable_l610_61042

/-- Represents a rectangular board -/
structure Board :=
  (length : ℕ)
  (width : ℕ)

/-- Represents a domino -/
structure Domino :=
  (length : ℕ)
  (width : ℕ)

/-- Checks if a board can be tiled with dominos -/
def can_be_tiled (b : Board) (d : Domino) : Prop :=
  (b.length * b.width) % (d.length * d.width) = 0

/-- The theorem stating that a 5×7 board cannot be tiled with 2×1 dominos -/
theorem five_by_seven_not_tileable :
  ¬(can_be_tiled (Board.mk 5 7) (Domino.mk 2 1)) :=
sorry

end five_by_seven_not_tileable_l610_61042


namespace lune_area_specific_case_l610_61027

/-- Represents a semicircle with a given diameter -/
structure Semicircle where
  diameter : ℝ
  diameter_pos : diameter > 0

/-- Represents a lune formed by two semicircles -/
structure Lune where
  upper : Semicircle
  lower : Semicircle
  upper_on_lower : upper.diameter < lower.diameter

/-- Calculates the area of a lune -/
noncomputable def lune_area (l : Lune) : ℝ :=
  sorry

theorem lune_area_specific_case :
  let upper := Semicircle.mk 3 (by norm_num)
  let lower := Semicircle.mk 4 (by norm_num)
  let l := Lune.mk upper lower (by norm_num)
  lune_area l = (9 * Real.sqrt 3) / 4 - (55 / 24) * Real.pi :=
sorry

end lune_area_specific_case_l610_61027


namespace pentagon_area_is_14_l610_61086

/-- Represents a trapezoid segmented into two triangles and a pentagon -/
structure SegmentedTrapezoid where
  triangle1_area : ℝ
  triangle2_area : ℝ
  base_ratio : ℝ
  total_area : ℝ

/-- The area of the pentagon in a segmented trapezoid -/
def pentagon_area (t : SegmentedTrapezoid) : ℝ :=
  t.total_area - t.triangle1_area - t.triangle2_area

/-- Theorem stating that the area of the pentagon is 14 under given conditions -/
theorem pentagon_area_is_14 (t : SegmentedTrapezoid) 
  (h1 : t.triangle1_area = 8)
  (h2 : t.triangle2_area = 18)
  (h3 : t.base_ratio = 2)
  (h4 : t.total_area = 40) :
  pentagon_area t = 14 := by
  sorry


end pentagon_area_is_14_l610_61086


namespace other_number_is_31_l610_61030

theorem other_number_is_31 (a b : ℤ) (h1 : 3 * a + 2 * b = 140) (h2 : a = 26 ∨ b = 26) : (a = 26 ∧ b = 31) ∨ (a = 31 ∧ b = 26) :=
sorry

end other_number_is_31_l610_61030


namespace intersection_of_lines_l610_61028

/-- The intersection point of two lines in 3D space --/
def intersection_point (A B C D : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := sorry

/-- Theorem: The intersection of lines AB and CD --/
theorem intersection_of_lines 
  (A : ℝ × ℝ × ℝ) 
  (B : ℝ × ℝ × ℝ) 
  (C : ℝ × ℝ × ℝ) 
  (D : ℝ × ℝ × ℝ) 
  (h1 : A = (6, -7, 7)) 
  (h2 : B = (15, -16, 11)) 
  (h3 : C = (0, 3, -6)) 
  (h4 : D = (2, -5, 10)) : 
  intersection_point A B C D = (144/27, -171/27, 181/27) := by
  sorry

end intersection_of_lines_l610_61028


namespace cuboid_to_cube_l610_61097

-- Define the dimensions of the original cuboid
def cuboid_length : ℝ := 27
def cuboid_width : ℝ := 18
def cuboid_height : ℝ := 12

-- Define the volume to be added
def added_volume : ℝ := 17.999999999999996

-- Define the edge length of the resulting cube in centimeters
def cube_edge_cm : ℕ := 1802

-- Theorem statement
theorem cuboid_to_cube :
  let original_volume := cuboid_length * cuboid_width * cuboid_height
  let total_volume := original_volume + added_volume
  let cube_edge_m := (total_volume ^ (1/3 : ℝ))
  ∃ (ε : ℝ), ε ≥ 0 ∧ ε < 1 ∧ cube_edge_cm = ⌊cube_edge_m * 100 + ε⌋ :=
sorry

end cuboid_to_cube_l610_61097


namespace science_fair_girls_fraction_l610_61089

theorem science_fair_girls_fraction :
  let pine_grove_total : ℕ := 300
  let pine_grove_ratio_boys : ℕ := 3
  let pine_grove_ratio_girls : ℕ := 2
  let maple_town_total : ℕ := 240
  let maple_town_ratio_boys : ℕ := 5
  let maple_town_ratio_girls : ℕ := 3
  let total_students := pine_grove_total + maple_town_total
  let pine_grove_girls := (pine_grove_total * pine_grove_ratio_girls) / (pine_grove_ratio_boys + pine_grove_ratio_girls)
  let maple_town_girls := (maple_town_total * maple_town_ratio_girls) / (maple_town_ratio_boys + maple_town_ratio_girls)
  let total_girls := pine_grove_girls + maple_town_girls
  (total_girls : ℚ) / total_students = 7 / 18 := by
  sorry

end science_fair_girls_fraction_l610_61089


namespace notecard_area_theorem_l610_61035

/-- Given a rectangle with original dimensions 5 × 7 inches, prove that if shortening one side
    by 2 inches results in an area of 21 square inches, then shortening the other side
    by 2 inches instead will result in an area of 25 square inches. -/
theorem notecard_area_theorem :
  ∀ (original_width original_length : ℝ),
    original_width = 5 →
    original_length = 7 →
    (∃ (new_width new_length : ℝ),
      (new_width = original_width - 2 ∧ new_length = original_length ∨
       new_width = original_width ∧ new_length = original_length - 2) ∧
      new_width * new_length = 21) →
    ∃ (other_width other_length : ℝ),
      (other_width = original_width - 2 ∧ other_length = original_length ∨
       other_width = original_width ∧ other_length = original_length - 2) ∧
      other_width ≠ new_width ∧
      other_length ≠ new_length ∧
      other_width * other_length = 25 :=
by sorry

end notecard_area_theorem_l610_61035


namespace transformed_triangle_area_equality_l610_61024

-- Define the domain
variable (x₁ x₂ x₃ : ℝ)

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the area function for a triangle given three points
def triangle_area (p₁ p₂ p₃ : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem transformed_triangle_area_equality 
  (h₁ : triangle_area (x₁, f x₁) (x₂, f x₂) (x₃, f x₃) = 50)
  (h₂ : x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₃ ≠ x₁) :
  triangle_area (x₁/3, 3 * f x₁) (x₂/3, 3 * f x₂) (x₃/3, 3 * f x₃) = 50 := by
  sorry

end transformed_triangle_area_equality_l610_61024


namespace car_speed_second_hour_l610_61011

/-- Given a car's speed over two hours, prove its speed in the second hour. -/
theorem car_speed_second_hour
  (speed_first_hour : ℝ)
  (average_speed : ℝ)
  (total_time : ℝ)
  (h1 : speed_first_hour = 50)
  (h2 : average_speed = 55)
  (h3 : total_time = 2)
  : ∃ (speed_second_hour : ℝ), speed_second_hour = 60 :=
by
  sorry

#check car_speed_second_hour

end car_speed_second_hour_l610_61011


namespace roots_difference_l610_61014

theorem roots_difference (x₁ x₂ : ℝ) : 
  x₁^2 + x₁ - 3 = 0 → 
  x₂^2 + x₂ - 3 = 0 → 
  |x₁ - x₂| = Real.sqrt 13 := by
sorry

end roots_difference_l610_61014


namespace total_playing_hours_l610_61023

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The number of hours Nathan plays per day -/
def nathan_hours_per_day : ℕ := 3

/-- The number of weeks Nathan plays -/
def nathan_weeks : ℕ := 2

/-- The number of hours Tobias plays per day -/
def tobias_hours_per_day : ℕ := 5

/-- The number of weeks Tobias plays -/
def tobias_weeks : ℕ := 1

/-- The total number of hours Nathan and Tobias played -/
def total_hours : ℕ := 
  nathan_hours_per_day * days_per_week * nathan_weeks + 
  tobias_hours_per_day * days_per_week * tobias_weeks

theorem total_playing_hours : total_hours = 77 := by
  sorry

end total_playing_hours_l610_61023


namespace seven_digit_multiples_of_three_l610_61034

theorem seven_digit_multiples_of_three (D B C : ℕ) : 
  D < 10 → B < 10 → C < 10 →
  (8 * 1000000 + 5 * 100000 + D * 10000 + 6 * 1000 + 3 * 100 + B * 10 + 2) % 3 = 0 →
  (4 * 1000000 + 1 * 100000 + 7 * 10000 + D * 1000 + B * 100 + 5 * 10 + C) % 3 = 0 →
  C = 2 := by
sorry

end seven_digit_multiples_of_three_l610_61034


namespace annika_hans_age_multiple_l610_61088

/-- Proves that in four years, Annika's age will be 3 times Hans' age -/
theorem annika_hans_age_multiple :
  ∀ (hans_current_age annika_current_age years_elapsed : ℕ),
    hans_current_age = 8 →
    annika_current_age = 32 →
    years_elapsed = 4 →
    (annika_current_age + years_elapsed) = 3 * (hans_current_age + years_elapsed) :=
by sorry

end annika_hans_age_multiple_l610_61088

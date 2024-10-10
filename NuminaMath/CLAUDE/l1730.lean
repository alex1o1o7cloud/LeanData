import Mathlib

namespace last_digit_sum_powers_l1730_173073

theorem last_digit_sum_powers : (1023^3923 + 3081^3921) % 10 = 8 := by
  sorry

end last_digit_sum_powers_l1730_173073


namespace complete_square_h_l1730_173088

theorem complete_square_h (x : ℝ) :
  ∃ (a k : ℝ), 3 * x^2 + 9 * x + 20 = a * (x - (-3/2))^2 + k :=
by sorry

end complete_square_h_l1730_173088


namespace candy_fundraiser_profit_l1730_173077

def candy_fundraiser (boxes_total : ℕ) (boxes_discounted : ℕ) (bars_per_box : ℕ) 
  (selling_price : ℚ) (regular_price : ℚ) (discounted_price : ℚ) : ℚ :=
  let boxes_regular := boxes_total - boxes_discounted
  let total_revenue := boxes_total * bars_per_box * selling_price
  let cost_regular := boxes_regular * bars_per_box * regular_price
  let cost_discounted := boxes_discounted * bars_per_box * discounted_price
  let total_cost := cost_regular + cost_discounted
  total_revenue - total_cost

theorem candy_fundraiser_profit :
  candy_fundraiser 5 3 10 (3/2) 1 (4/5) = 31 := by
  sorry

end candy_fundraiser_profit_l1730_173077


namespace unique_solution_l1730_173060

theorem unique_solution :
  ∃! (A B C D : ℕ),
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
    A ≤ 9 ∧ B ≤ 9 ∧ C ≤ 9 ∧ D ≤ 9 ∧
    1000 * A + 100 * A + 10 * B + C - (1000 * B + 100 * A + 10 * C + B) = 1000 * A + 100 * B + 10 * C + D ∧
    A = 9 ∧ B = 6 ∧ C = 8 ∧ D = 2 := by
  sorry

end unique_solution_l1730_173060


namespace cricket_matches_l1730_173017

theorem cricket_matches (score1 score2 overall_avg : ℚ) (matches1 matches2 : ℕ) 
  (h1 : score1 = 60)
  (h2 : score2 = 50)
  (h3 : overall_avg = 54)
  (h4 : matches1 = 2)
  (h5 : matches2 = 3) :
  matches1 + matches2 = 5 := by
  sorry

end cricket_matches_l1730_173017


namespace josh_new_marbles_l1730_173065

/-- The number of marbles Josh lost -/
def marbles_lost : ℕ := 8

/-- The additional marbles Josh found compared to those he lost -/
def additional_marbles : ℕ := 2

/-- The number of new marbles Josh found -/
def new_marbles : ℕ := marbles_lost + additional_marbles

theorem josh_new_marbles : new_marbles = 10 := by sorry

end josh_new_marbles_l1730_173065


namespace triangle_problem_l1730_173080

theorem triangle_problem (A B C : Real) (a b c : Real) :
  -- Triangle ABC is acute
  0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2 →
  -- a, b, c are sides opposite to angles A, B, C
  a > 0 ∧ b > 0 ∧ c > 0 →
  -- Law of Sines holds
  a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C →
  -- Given condition
  a / Real.sin A = 2 * c / Real.sqrt 3 →
  -- c = √7
  c = Real.sqrt 7 →
  -- Area of triangle ABC is 3√3/2
  1/2 * a * b * Real.sin C = 3 * Real.sqrt 3 / 2 →
  -- Prove:
  C = π/3 ∧ a^2 + b^2 = 13 := by
sorry

end triangle_problem_l1730_173080


namespace a_investment_l1730_173008

/-- Calculates the investment of partner A in a business partnership --/
def calculate_investment_A (investment_B investment_C total_profit profit_share_A : ℚ) : ℚ :=
  let total_investment := investment_B + investment_C + profit_share_A * (investment_B + investment_C) / (total_profit - profit_share_A)
  profit_share_A * total_investment / total_profit

/-- Theorem stating that A's investment is 6300 given the problem conditions --/
theorem a_investment (investment_B investment_C total_profit profit_share_A : ℚ)
  (hB : investment_B = 4200)
  (hC : investment_C = 10500)
  (hProfit : total_profit = 12100)
  (hShareA : profit_share_A = 3630) :
  calculate_investment_A investment_B investment_C total_profit profit_share_A = 6300 := by
  sorry

#eval calculate_investment_A 4200 10500 12100 3630

end a_investment_l1730_173008


namespace convex_pentagon_probability_l1730_173030

/-- The number of points on the circle -/
def n : ℕ := 8

/-- The number of chords to be selected -/
def k : ℕ := 5

/-- The total number of possible chords -/
def total_chords : ℕ := n.choose 2

/-- The number of ways to select k chords from total_chords -/
def total_selections : ℕ := total_chords.choose k

/-- The number of ways to select k points from n points -/
def convex_pentagons : ℕ := n.choose k

/-- The probability of forming a convex pentagon -/
def probability : ℚ := convex_pentagons / total_selections

theorem convex_pentagon_probability :
  probability = 1 / 1755 :=
sorry

end convex_pentagon_probability_l1730_173030


namespace square_sum_pattern_l1730_173020

theorem square_sum_pattern : 
  (1^2 + 3^2 = 10) → (2^2 + 4^2 = 20) → (3^2 + 5^2 = 34) → (4^2 + 6^2 = 52) := by
  sorry

end square_sum_pattern_l1730_173020


namespace product_in_S_and_counterexample_l1730_173024

def S : Set ℤ := {x | ∃ n : ℤ, x = n^2 + n + 1}

theorem product_in_S_and_counterexample :
  (∀ n : ℤ, (n^2 + n + 1) * ((n+1)^2 + (n+1) + 1) ∈ S) ∧
  (∃ a b : ℤ, a ∈ S ∧ b ∈ S ∧ a * b ∉ S) := by
  sorry

end product_in_S_and_counterexample_l1730_173024


namespace product_of_numbers_with_given_sum_and_difference_l1730_173055

theorem product_of_numbers_with_given_sum_and_difference :
  ∀ x y : ℝ, x + y = 70 ∧ x - y = 10 → x * y = 1200 := by
  sorry

end product_of_numbers_with_given_sum_and_difference_l1730_173055


namespace trees_planted_l1730_173089

/-- Given the initial number of short trees and the final number after planting,
    prove that the number of trees planted is the difference between these two values. -/
theorem trees_planted (initial_short_trees final_short_trees : ℕ) :
  final_short_trees ≥ initial_short_trees →
  final_short_trees - initial_short_trees = final_short_trees - initial_short_trees :=
by
  sorry

/-- Solve the specific problem instance -/
def solve_tree_planting_problem : ℕ :=
  98 - 41

#eval solve_tree_planting_problem

end trees_planted_l1730_173089


namespace quadratic_function_properties_l1730_173058

/-- Given a quadratic function f(x) = ax^2 - (1/2)x + c where a and c are real numbers,
    f(1) = 0, and f(x) ≥ 0 for all real x, prove that a = 1/4, c = 1/4, and
    there exists m = 3 such that g(x) = 4f(x) - mx has a minimum value of -5 
    in the interval [m, m+2] -/
theorem quadratic_function_properties (a c : ℝ) 
    (f : ℝ → ℝ)
    (h1 : ∀ x, f x = a * x^2 - (1/2) * x + c)
    (h2 : f 1 = 0)
    (h3 : ∀ x, f x ≥ 0) :
    a = (1/4) ∧ c = (1/4) ∧
    ∃ m : ℝ, m = 3 ∧
    (∀ x ∈ Set.Icc m (m + 2), 4 * (f x) - m * x ≥ -5) ∧
    (∃ x₀ ∈ Set.Icc m (m + 2), 4 * (f x₀) - m * x₀ = -5) := by
  sorry


end quadratic_function_properties_l1730_173058


namespace binomial_9_choose_5_l1730_173000

theorem binomial_9_choose_5 : Nat.choose 9 5 = 126 := by sorry

end binomial_9_choose_5_l1730_173000


namespace rice_mixture_cost_l1730_173074

/-- The cost of a mixture of two rice varieties -/
def mixture_cost (c1 c2 r : ℚ) : ℚ :=
  (c1 * r + c2 * 1) / (r + 1)

theorem rice_mixture_cost :
  let c1 : ℚ := 5.5
  let c2 : ℚ := 8.75
  let r : ℚ := 5/8
  mixture_cost c1 c2 r = 7.5 := by
sorry

end rice_mixture_cost_l1730_173074


namespace jason_pokemon_cards_jason_initial_cards_l1730_173059

theorem jason_pokemon_cards : ℕ → Prop :=
  fun initial_cards =>
    let given_away := 9
    let remaining := 4
    initial_cards = given_away + remaining

theorem jason_initial_cards : ∃ x : ℕ, jason_pokemon_cards x ∧ x = 13 :=
sorry

end jason_pokemon_cards_jason_initial_cards_l1730_173059


namespace mod_nine_power_difference_l1730_173090

theorem mod_nine_power_difference : 54^2023 - 27^2023 ≡ 0 [ZMOD 9] := by sorry

end mod_nine_power_difference_l1730_173090


namespace pencils_in_drawer_l1730_173009

/-- Given an initial number of pencils and a number of pencils added, 
    calculate the total number of pencils -/
def total_pencils (initial : ℕ) (added : ℕ) : ℕ :=
  initial + added

/-- Theorem stating that 27 initial pencils plus 45 added pencils equal 72 total pencils -/
theorem pencils_in_drawer : total_pencils 27 45 = 72 := by
  sorry

end pencils_in_drawer_l1730_173009


namespace power_of_ten_thousand_zeros_after_one_l1730_173096

theorem power_of_ten_thousand (n : ℕ) : (10000 : ℕ) ^ n = (10 : ℕ) ^ (4 * n) := by sorry

theorem zeros_after_one : (10000 : ℕ) ^ 50 = (10 : ℕ) ^ 200 := by sorry

end power_of_ten_thousand_zeros_after_one_l1730_173096


namespace product_pure_imaginary_l1730_173049

theorem product_pure_imaginary (a : ℝ) : 
  let z₁ : ℂ := a + 2*Complex.I
  let z₂ : ℂ := 2 + Complex.I
  (∃ b : ℝ, z₁ * z₂ = b * Complex.I) → a = 1 := by
sorry

end product_pure_imaginary_l1730_173049


namespace minimum_value_and_inequality_l1730_173048

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 1| + |x - 5|

-- State the theorem
theorem minimum_value_and_inequality :
  (∃ m : ℝ, (∀ x : ℝ, f x ≥ m) ∧ (∃ x₀ : ℝ, f x₀ = m) ∧ m = 6) ∧
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a + b + c = 6 → a^2 + b^2 + c^2 ≥ 12) :=
by sorry

end minimum_value_and_inequality_l1730_173048


namespace matrix_N_satisfies_conditions_l1730_173015

theorem matrix_N_satisfies_conditions : ∃ (N : Matrix (Fin 2) (Fin 2) ℝ),
  N = !![2, 1; 7, -2] ∧
  N.mulVec ![2, 0] = ![4, 14] ∧
  N.mulVec ![-2, 10] = ![6, -34] := by
  sorry

end matrix_N_satisfies_conditions_l1730_173015


namespace jerry_shelf_capacity_l1730_173083

/-- Given the total number of books, the number of books taken by the librarian,
    and the number of shelves needed, calculate the number of books that can fit on each shelf. -/
def books_per_shelf (total_books : ℕ) (books_taken : ℕ) (shelves_needed : ℕ) : ℕ :=
  (total_books - books_taken) / shelves_needed

/-- Prove that Jerry can fit 3 books on each shelf. -/
theorem jerry_shelf_capacity : books_per_shelf 34 7 9 = 3 := by
  sorry

end jerry_shelf_capacity_l1730_173083


namespace polynomial_remainder_l1730_173087

def polynomial (x : ℝ) : ℝ := 6*x^8 - 2*x^7 - 10*x^6 + 3*x^4 + 5*x^3 - 15

def divisor (x : ℝ) : ℝ := 3*x - 6

theorem polynomial_remainder :
  ∃ (q : ℝ → ℝ), ∀ x, polynomial x = q x * divisor x + 713 := by
  sorry

end polynomial_remainder_l1730_173087


namespace colored_grid_rectangle_exists_l1730_173057

-- Define the color type
inductive Color
  | Red
  | White
  | Blue

-- Define the grid type
def Grid := Fin 12 → Fin 12 → Color

-- Define a rectangle in the grid
structure Rectangle where
  x1 : Fin 12
  y1 : Fin 12
  x2 : Fin 12
  y2 : Fin 12
  h_x : x1 < x2
  h_y : y1 < y2

-- Define a function to check if a rectangle has all vertices of the same color
def sameColorVertices (g : Grid) (r : Rectangle) : Prop :=
  g r.x1 r.y1 = g r.x1 r.y2 ∧
  g r.x1 r.y1 = g r.x2 r.y1 ∧
  g r.x1 r.y1 = g r.x2 r.y2

-- State the theorem
theorem colored_grid_rectangle_exists (g : Grid) :
  ∃ r : Rectangle, sameColorVertices g r := by
  sorry

end colored_grid_rectangle_exists_l1730_173057


namespace logarithm_inequality_and_root_comparison_l1730_173001

theorem logarithm_inequality_and_root_comparison : 
  (∀ a b : ℝ, a > 0 → b > 0 → Real.log (((a + b) / 2) : ℝ) ≥ (Real.log a + Real.log b) / 2) ∧
  (Real.sqrt 6 + Real.sqrt 7 > 2 * Real.sqrt 2 + Real.sqrt 5) :=
by sorry

end logarithm_inequality_and_root_comparison_l1730_173001


namespace no_simultaneous_squares_l1730_173052

theorem no_simultaneous_squares : ¬∃ (x y : ℕ), ∃ (a b : ℕ), x^2 + y = a^2 ∧ x + y^2 = b^2 := by
  sorry

end no_simultaneous_squares_l1730_173052


namespace perfect_square_polynomial_l1730_173022

theorem perfect_square_polynomial (x : ℤ) : 
  (∃ y : ℤ, x^4 + x^3 + x^2 + x + 1 = y^2) ↔ x = 0 := by
  sorry

end perfect_square_polynomial_l1730_173022


namespace average_speed_distance_expression_time_range_l1730_173085

-- Define the boat's movement
structure BoatMovement where
  distance : ℕ → ℝ
  time : ℕ → ℝ

-- Define the given data
def givenData : BoatMovement := {
  distance := λ n => match n with
    | 0 => 200
    | 1 => 150
    | 2 => 100
    | 3 => 50
    | _ => 0
  time := λ n => 2 * n
}

-- Theorem for the average speed
theorem average_speed (b : BoatMovement) : 
  (b.distance 0 - b.distance 3) / (b.time 3 - b.time 0) = 25 := by
  sorry

-- Theorem for the analytical expression
theorem distance_expression (b : BoatMovement) (x : ℝ) : 
  ∃ y : ℝ, y = 200 - 25 * x := by
  sorry

-- Theorem for the range of x
theorem time_range (b : BoatMovement) (x : ℝ) : 
  0 ≤ x ∧ x ≤ 8 := by
  sorry

end average_speed_distance_expression_time_range_l1730_173085


namespace expression_value_for_x_3_l1730_173067

theorem expression_value_for_x_3 :
  let x : ℕ := 3
  x + x * (x ^ (x + 1)) = 246 :=
by sorry

end expression_value_for_x_3_l1730_173067


namespace only_setC_is_right_triangle_l1730_173039

-- Define a function to check if three numbers satisfy the Pythagorean theorem
def isPythagoreanTriple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

-- Define the sets of line segments
def setA : List ℕ := [1, 2, 3]
def setB : List ℕ := [5, 11, 12]
def setC : List ℕ := [5, 12, 13]
def setD : List ℕ := [6, 8, 9]

-- Theorem stating that only setC forms a right triangle
theorem only_setC_is_right_triangle :
  (¬ isPythagoreanTriple setA[0]! setA[1]! setA[2]!) ∧
  (¬ isPythagoreanTriple setB[0]! setB[1]! setB[2]!) ∧
  (isPythagoreanTriple setC[0]! setC[1]! setC[2]!) ∧
  (¬ isPythagoreanTriple setD[0]! setD[1]! setD[2]!) :=
by sorry

end only_setC_is_right_triangle_l1730_173039


namespace beth_crayons_left_l1730_173040

/-- The number of crayons Beth has left after giving some away -/
def crayons_left (initial : ℕ) (given_away : ℕ) : ℕ :=
  initial - given_away

/-- Proof that Beth has 52 crayons left -/
theorem beth_crayons_left :
  let initial_crayons : ℕ := 106
  let crayons_given_away : ℕ := 54
  crayons_left initial_crayons crayons_given_away = 52 := by
sorry

end beth_crayons_left_l1730_173040


namespace polynomial_evaluation_l1730_173045

theorem polynomial_evaluation : 
  let x : ℝ := 2
  2 * x^2 - 3 * x + 4 = 6 := by sorry

end polynomial_evaluation_l1730_173045


namespace cos_2phi_nonpositive_l1730_173025

theorem cos_2phi_nonpositive (α β φ : Real) 
  (h : Real.tan φ = 1 / (Real.cos α * Real.cos β + Real.tan α * Real.tan β)) : 
  Real.cos (2 * φ) ≤ 0 := by
  sorry

end cos_2phi_nonpositive_l1730_173025


namespace angle_with_special_supplement_complement_relation_l1730_173071

theorem angle_with_special_supplement_complement_relation :
  ∀ x : ℝ,
  (0 < x) ∧ (x < 180) →
  (180 - x = 3 * (90 - x)) →
  x = 45 := by
sorry

end angle_with_special_supplement_complement_relation_l1730_173071


namespace analysis_time_per_bone_l1730_173092

/-- The number of bones in the human body -/
def num_bones : ℕ := 206

/-- The total time Aria needs to analyze all bones (in hours) -/
def total_analysis_time : ℕ := 1030

/-- The time spent analyzing each bone (in hours) -/
def time_per_bone : ℚ := total_analysis_time / num_bones

theorem analysis_time_per_bone : time_per_bone = 5 := by
  sorry

end analysis_time_per_bone_l1730_173092


namespace inequality_proof_l1730_173099

theorem inequality_proof (a b : ℝ) (h : a > b) : a^2 - a*b > b*a - b^2 := by
  sorry

end inequality_proof_l1730_173099


namespace ratio_equality_l1730_173011

theorem ratio_equality (a b : ℝ) (h : 4 * a = 5 * b) : a / b = 5 / 4 := by
  sorry

end ratio_equality_l1730_173011


namespace negative_one_squared_plus_cubed_equals_zero_l1730_173066

theorem negative_one_squared_plus_cubed_equals_zero :
  (-1 : ℤ)^2 + (-1 : ℤ)^3 = 0 := by sorry

end negative_one_squared_plus_cubed_equals_zero_l1730_173066


namespace keegan_class_time_l1730_173093

theorem keegan_class_time (total_hours : Real) (num_classes : Nat) (other_class_time : Real) :
  total_hours = 7.5 →
  num_classes = 7 →
  other_class_time = 72 / 60 →
  let other_classes_time := other_class_time * (num_classes - 2 : Real)
  let history_chem_time := total_hours - other_classes_time
  history_chem_time = 1.5 := by
sorry

end keegan_class_time_l1730_173093


namespace total_vowels_written_l1730_173043

/-- The number of vowels in the English alphabet -/
def num_vowels : ℕ := 5

/-- The number of times each vowel is written -/
def times_written : ℕ := 4

/-- Theorem: The total number of vowels written on the board is 20 -/
theorem total_vowels_written : num_vowels * times_written = 20 := by
  sorry

end total_vowels_written_l1730_173043


namespace inscribed_rectangle_area_max_l1730_173072

theorem inscribed_rectangle_area_max (R : ℝ) (x y : ℝ) 
  (h1 : x > 0) (h2 : y > 0) (h3 : x^2 + y^2 = 4*R^2) : 
  x * y ≤ 2 * R^2 ∧ (x * y = 2 * R^2 ↔ x = y ∧ x = R * Real.sqrt 2) :=
sorry

end inscribed_rectangle_area_max_l1730_173072


namespace distance_range_m_l1730_173079

-- Define the distance function
def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := |x₁ - x₂| + 2 * |y₁ - y₂|

-- Define the theorem
theorem distance_range_m :
  ∀ m : ℝ,
  (distance 2 1 (-1) m ≤ 5) ↔ (0 ≤ m ∧ m ≤ 2) :=
by sorry

end distance_range_m_l1730_173079


namespace negative_463_terminal_side_l1730_173003

-- Define the concept of terminal side equality for angles
def same_terminal_side (α β : ℝ) : Prop :=
  ∃ k : ℤ, α = β + k * 360

-- State the theorem
theorem negative_463_terminal_side :
  ∀ K : ℤ, same_terminal_side (-463) (K * 360 + 257) :=
sorry

end negative_463_terminal_side_l1730_173003


namespace sample_size_is_sampled_l1730_173091

/-- A survey about middle school students riding electric bikes to school -/
structure Survey where
  population : ℕ
  sampled : ℕ
  negative_attitude : ℕ

/-- The sample size of a survey is equal to the number of people sampled -/
theorem sample_size_is_sampled (s : Survey) (h : s.population = 823 ∧ s.sampled = 150 ∧ s.negative_attitude = 136) : 
  s.sampled = 150 := by
  sorry

end sample_size_is_sampled_l1730_173091


namespace max_both_writers_and_editors_l1730_173037

/-- Conference attendees -/
structure Conference where
  total : ℕ
  writers : ℕ
  editors : ℕ
  both : ℕ
  neither : ℕ

/-- Conference constraints -/
def valid_conference (c : Conference) : Prop :=
  c.total = 100 ∧
  c.writers = 35 ∧
  c.editors > 38 ∧
  c.neither = 2 * c.both ∧
  c.total = c.writers + c.editors - c.both + c.neither

/-- Theorem: The maximum number of people who can be both writers and editors is 26 -/
theorem max_both_writers_and_editors (c : Conference) (h : valid_conference c) :
  c.both ≤ 26 := by
  sorry

end max_both_writers_and_editors_l1730_173037


namespace circle_area_difference_l1730_173004

theorem circle_area_difference : ∀ (π : ℝ), 
  let r1 : ℝ := 30
  let d2 : ℝ := 30
  let area1 : ℝ := π * r1^2
  let area2 : ℝ := π * (d2/2)^2
  area1 - area2 = 675 * π :=
by sorry

end circle_area_difference_l1730_173004


namespace isosceles_triangle_obtuse_iff_quadratic_roots_l1730_173078

theorem isosceles_triangle_obtuse_iff_quadratic_roots 
  (A B C : Real) 
  (triangle_sum : A + B + C = π) 
  (isosceles : A = C) : 
  (B > π / 2) ↔ 
  ∃ (x₁ x₂ : Real), x₁ ≠ x₂ ∧ A * x₁^2 + B * x₁ + C = 0 ∧ A * x₂^2 + B * x₂ + C = 0 :=
by sorry

end isosceles_triangle_obtuse_iff_quadratic_roots_l1730_173078


namespace stating_ancient_chinese_problem_correct_l1730_173006

/-- Represents the system of equations for the ancient Chinese mathematical problem. -/
def ancient_chinese_problem (x y : ℝ) : Prop :=
  (y = 8 * x - 3) ∧ (y = 7 * x + 4)

/-- 
Theorem stating that the system of equations correctly represents the given problem,
where x is the number of people and y is the price of the items in coins.
-/
theorem ancient_chinese_problem_correct (x y : ℝ) :
  ancient_chinese_problem x y ↔
  (∃ (total_price : ℝ),
    (8 * x = total_price + 3) ∧
    (7 * x = total_price - 4) ∧
    (y = total_price)) :=
by sorry

end stating_ancient_chinese_problem_correct_l1730_173006


namespace jumping_game_l1730_173082

theorem jumping_game (n : ℕ) 
  (h_odd : Odd n)
  (h_mod3 : n % 3 = 2)
  (h_mod5 : n % 5 = 2) : 
  n = 47 := by
  sorry

end jumping_game_l1730_173082


namespace sum_of_binary_digits_300_l1730_173002

/-- Converts a natural number to its binary representation as a list of digits (0 or 1) --/
def toBinary (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec toBinaryAux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else toBinaryAux (m / 2) ((m % 2) :: acc)
    toBinaryAux n []

/-- Sums a list of natural numbers --/
def sumList (l : List ℕ) : ℕ :=
  l.foldl (· + ·) 0

/-- The sum of digits in the binary representation of 300 is 3 --/
theorem sum_of_binary_digits_300 :
  sumList (toBinary 300) = 3 := by
  sorry

end sum_of_binary_digits_300_l1730_173002


namespace shopping_expense_l1730_173012

theorem shopping_expense (initial_amount : ℝ) (amount_left : ℝ) : 
  initial_amount = 158 →
  amount_left = 78 →
  ∃ (shoes_price bag_price lunch_price : ℝ),
    bag_price = shoes_price - 17 ∧
    lunch_price = bag_price / 4 ∧
    initial_amount = shoes_price + bag_price + lunch_price + amount_left ∧
    shoes_price = 45 := by
  sorry

end shopping_expense_l1730_173012


namespace sum_of_xyz_l1730_173098

theorem sum_of_xyz (x y z : ℝ) 
  (eq1 : x = y + z + 2)
  (eq2 : y = z + x + 1)
  (eq3 : z = x + y + 4) :
  x + y + z = -7 := by sorry

end sum_of_xyz_l1730_173098


namespace root_product_fourth_power_l1730_173097

theorem root_product_fourth_power (r s t : ℂ) : 
  (r^3 + 5*r + 4 = 0) → 
  (s^3 + 5*s + 4 = 0) → 
  (t^3 + 5*t + 4 = 0) → 
  (r+s)^4 * (s+t)^4 * (t+r)^4 = 256 := by
sorry

end root_product_fourth_power_l1730_173097


namespace units_digit_of_sum_of_products_l1730_173026

def consecutive_product (n : ℕ) (count : ℕ) : ℕ :=
  (List.range count).foldl (λ acc _ => acc * n) 1

def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_of_sum_of_products : 
  units_digit (consecutive_product 2017 2016 + consecutive_product 2016 2017) = 7 := by
  sorry

end units_digit_of_sum_of_products_l1730_173026


namespace gcd_problem_l1730_173051

theorem gcd_problem (a b : ℕ+) (h : Nat.gcd a b = 12) :
  (∃ (x y : ℕ+), Nat.gcd x y = 12 ∧ Nat.gcd (12 * x) (18 * y) = 72) ∧
  (∀ (c d : ℕ+), Nat.gcd c d = 12 → Nat.gcd (12 * c) (18 * d) ≥ 72) :=
sorry

end gcd_problem_l1730_173051


namespace employed_males_percentage_l1730_173005

theorem employed_males_percentage (total_population : ℝ) (employed_population : ℝ) (employed_females : ℝ) :
  employed_population = 0.64 * total_population →
  employed_females = 0.21875 * employed_population →
  0.4996 * total_population = employed_population - employed_females :=
by sorry

end employed_males_percentage_l1730_173005


namespace perpendicular_vectors_magnitude_l1730_173027

def a : ℝ × ℝ := (1, 2)
def b (y : ℝ) : ℝ × ℝ := (2, y)

theorem perpendicular_vectors_magnitude (y : ℝ) 
  (h : a.1 * (b y).1 + a.2 * (b y).2 = 0) : 
  ‖(2 : ℝ) • a + b y‖ = 5 := by
  sorry

end perpendicular_vectors_magnitude_l1730_173027


namespace system_solutions_l1730_173023

/-- The system of equations has exactly eight solutions -/
theorem system_solutions :
  ∃! (solutions : Finset (ℝ × ℝ × ℝ)),
    solutions.card = 8 ∧
    (∀ (x y z : ℝ), (x, y, z) ∈ solutions ↔
      ((x - 2)^2 + (y + 1)^2 = 5 ∧
       (x - 2)^2 + (z - 3)^2 = 13 ∧
       (y + 1)^2 + (z - 3)^2 = 10)) ∧
    solutions = {(0, 0, 0), (0, -2, 0), (0, 0, 6), (0, -2, 6),
                 (4, 0, 0), (4, -2, 0), (4, 0, 6), (4, -2, 6)} := by
  sorry


end system_solutions_l1730_173023


namespace total_beneficial_insects_l1730_173038

theorem total_beneficial_insects (ladybugs_with_spots : Nat) (ladybugs_without_spots : Nat) (green_lacewings : Nat) (trichogramma_wasps : Nat)
  (h1 : ladybugs_with_spots = 12170)
  (h2 : ladybugs_without_spots = 54912)
  (h3 : green_lacewings = 67923)
  (h4 : trichogramma_wasps = 45872) :
  ladybugs_with_spots + ladybugs_without_spots + green_lacewings + trichogramma_wasps = 180877 := by
  sorry

end total_beneficial_insects_l1730_173038


namespace monitor_pixel_count_l1730_173032

/-- Calculates the total number of pixels on a monitor given its dimensions and pixel density. -/
def total_pixels (width : ℕ) (height : ℕ) (pixel_density : ℕ) : ℕ :=
  (width * pixel_density) * (height * pixel_density)

/-- Theorem: A monitor that is 21 inches wide and 12 inches tall with a pixel density of 100 dots per inch has 2,520,000 pixels. -/
theorem monitor_pixel_count :
  total_pixels 21 12 100 = 2520000 := by
  sorry

end monitor_pixel_count_l1730_173032


namespace cube_root_of_three_cubes_of_three_to_fifth_l1730_173064

theorem cube_root_of_three_cubes_of_three_to_fifth (x : ℝ) : 
  x = (3^5 + 3^5 + 3^5)^(1/3) → x = 9 := by
  sorry

end cube_root_of_three_cubes_of_three_to_fifth_l1730_173064


namespace tan_alpha_value_l1730_173068

theorem tan_alpha_value (α β : ℝ) 
  (h1 : Real.tan (3 * α - 2 * β) = 1 / 2)
  (h2 : Real.tan (5 * α - 4 * β) = 1 / 4) : 
  Real.tan α = 13 / 16 := by
  sorry

end tan_alpha_value_l1730_173068


namespace opposite_to_A_is_F_l1730_173046

/-- Represents the labels of the squares --/
inductive Label
  | A | B | C | D | E | F

/-- Represents a cube formed by folding six connected squares --/
structure Cube where
  faces : Fin 6 → Label
  is_valid : ∀ (l : Label), ∃ (i : Fin 6), faces i = l

/-- Defines the opposite face relation on a cube --/
def opposite (c : Cube) (l1 l2 : Label) : Prop :=
  ∃ (i j : Fin 6), c.faces i = l1 ∧ c.faces j = l2 ∧ i ≠ j ∧
    ∀ (k : Fin 6), k ≠ i → k ≠ j → 
      ∃ (m : Fin 6), m ≠ i ∧ m ≠ j ∧ (c.faces k = c.faces m)

/-- Theorem stating that F is opposite to A in the cube --/
theorem opposite_to_A_is_F (c : Cube) : opposite c Label.A Label.F := by
  sorry

end opposite_to_A_is_F_l1730_173046


namespace cafeteria_stacking_l1730_173095

theorem cafeteria_stacking (initial_cartons : ℕ) (cartons_per_stack : ℕ) (teacher_cartons : ℕ) :
  initial_cartons = 799 →
  cartons_per_stack = 6 →
  teacher_cartons = 23 →
  let remaining_cartons := initial_cartons - teacher_cartons
  let full_stacks := remaining_cartons / cartons_per_stack
  let double_stacks := full_stacks / 2
  let leftover_cartons := remaining_cartons % cartons_per_stack + (full_stacks % 2) * cartons_per_stack
  double_stacks = 64 ∧ leftover_cartons = 8 :=
by sorry

end cafeteria_stacking_l1730_173095


namespace cubic_root_sum_product_l1730_173054

theorem cubic_root_sum_product (p q r : ℝ) : 
  (6 * p^3 - 4 * p^2 + 15 * p - 10 = 0) ∧ 
  (6 * q^3 - 4 * q^2 + 15 * q - 10 = 0) ∧ 
  (6 * r^3 - 4 * r^2 + 15 * r - 10 = 0) →
  p * q + q * r + r * p = 5/2 := by
sorry

end cubic_root_sum_product_l1730_173054


namespace quarters_per_machine_l1730_173086

/-- Represents the number of machines in the launderette -/
def num_machines : ℕ := 3

/-- Represents the number of dimes in each machine -/
def dimes_per_machine : ℕ := 100

/-- Represents the total amount of money from all machines in cents -/
def total_money : ℕ := 9000  -- $90 in cents

/-- Represents the value of a dime in cents -/
def dime_value : ℕ := 10

/-- Represents the value of a quarter in cents -/
def quarter_value : ℕ := 25

theorem quarters_per_machine :
  ∃ (q : ℕ), 
    q * quarter_value * num_machines + 
    dimes_per_machine * dime_value * num_machines = 
    total_money ∧ 
    q = 80 := by
  sorry

end quarters_per_machine_l1730_173086


namespace student_age_fraction_l1730_173044

theorem student_age_fraction (total_students : ℕ) (below_8_percent : ℚ) (age_8_students : ℕ) : 
  total_students = 50 →
  below_8_percent = 1/5 →
  age_8_students = 24 →
  (total_students - (total_students * below_8_percent).num - age_8_students : ℚ) / age_8_students = 2/3 := by
  sorry

end student_age_fraction_l1730_173044


namespace system_solution_l1730_173063

theorem system_solution (x y k : ℝ) : 
  (4 * x + 2 * y = 5 * k - 4) → 
  (2 * x + 4 * y = -1) → 
  (x - y = 1) → 
  (k = 1) := by
sorry

end system_solution_l1730_173063


namespace spring_math_camp_attendance_l1730_173033

theorem spring_math_camp_attendance : ∃ (total boys girls : ℕ),
  total = boys + girls ∧
  50 ≤ total ∧ total ≤ 70 ∧
  3 * boys + 9 * girls = 8 * boys + 2 * girls ∧
  total = 60 := by
  sorry

end spring_math_camp_attendance_l1730_173033


namespace remainder_of_3_99_plus_5_mod_9_l1730_173094

theorem remainder_of_3_99_plus_5_mod_9 : (3^99 + 5) % 9 = 5 := by
  sorry

end remainder_of_3_99_plus_5_mod_9_l1730_173094


namespace line_parallel_perpendicular_characterization_l1730_173061

/-- Two lines in 3D space -/
structure Line3D where
  m : ℝ
  n : ℝ
  p : ℝ

/-- Condition for two lines to be parallel -/
def parallel (l₁ l₂ : Line3D) : Prop :=
  l₁.m / l₂.m = l₁.n / l₂.n ∧ l₁.n / l₂.n = l₁.p / l₂.p

/-- Condition for two lines to be perpendicular -/
def perpendicular (l₁ l₂ : Line3D) : Prop :=
  l₁.m * l₂.m + l₁.n * l₂.n + l₁.p * l₂.p = 0

/-- Theorem: Characterization of parallel and perpendicular lines in 3D space -/
theorem line_parallel_perpendicular_characterization (l₁ l₂ : Line3D) :
  (parallel l₁ l₂ ↔ l₁.m / l₂.m = l₁.n / l₂.n ∧ l₁.n / l₂.n = l₁.p / l₂.p) ∧
  (perpendicular l₁ l₂ ↔ l₁.m * l₂.m + l₁.n * l₂.n + l₁.p * l₂.p = 0) := by
  sorry

end line_parallel_perpendicular_characterization_l1730_173061


namespace expected_red_lights_l1730_173041

-- Define the number of intersections
def num_intersections : ℕ := 3

-- Define the probability of encountering a red light at each intersection
def red_light_prob : ℝ := 0.3

-- State the theorem
theorem expected_red_lights :
  let num_intersections : ℕ := 3
  let red_light_prob : ℝ := 0.3
  (num_intersections : ℝ) * red_light_prob = 0.9 := by
  sorry

end expected_red_lights_l1730_173041


namespace aaron_can_lids_l1730_173047

/-- The number of can lids Aaron is taking to the recycling center -/
def total_can_lids (num_boxes : ℕ) (existing_lids : ℕ) (lids_per_box : ℕ) : ℕ :=
  num_boxes * lids_per_box + existing_lids

/-- Proof that Aaron is taking 53 can lids to the recycling center -/
theorem aaron_can_lids : total_can_lids 3 14 13 = 53 := by
  sorry

end aaron_can_lids_l1730_173047


namespace problem_solution_l1730_173081

theorem problem_solution (a b c d m : ℝ) 
  (h1 : a = -b)  -- a and b are opposite numbers
  (h2 : c * d = 1)  -- c and d are reciprocals
  (h3 : |m| = 3)  -- absolute value of m is 3
  : (a + b) / 2023 - 4 * c * d + m^2 = 5 := by
  sorry

end problem_solution_l1730_173081


namespace cube_fourth_root_inverse_prop_l1730_173084

-- Define the inverse proportionality between a^3 and b^(1/4)
def inverse_prop (a b : ℝ) : Prop := ∃ k : ℝ, a^3 * b^(1/4) = k

-- Define the initial condition
def initial_condition (a b : ℝ) : Prop := a = 3 ∧ b = 16

-- Define the final condition
def final_condition (a b : ℝ) : Prop := a^2 * b = 54

theorem cube_fourth_root_inverse_prop 
  (a b : ℝ) 
  (h_inv_prop : inverse_prop a b) 
  (h_init : initial_condition a b) 
  (h_final : final_condition a b) : 
  b = 54^(2/5) := by
  sorry

end cube_fourth_root_inverse_prop_l1730_173084


namespace symmetry_center_l1730_173062

open Real

/-- Given a function f and its symmetric function g, prove that (π/4, 0) is a center of symmetry of g -/
theorem symmetry_center (f g : ℝ → ℝ) : 
  (∀ x, f x = sin (2*x + π/6)) →
  (∀ x, f (π/6 - x) = g x) →
  (π/4, 0) ∈ {p : ℝ × ℝ | ∀ x, g (p.1 + x) = g (p.1 - x)} :=
by sorry

end symmetry_center_l1730_173062


namespace solve_for_y_l1730_173019

theorem solve_for_y : ∃ y : ℚ, ((2^5 : ℚ) * y) / ((8^2 : ℚ) * (3^5 : ℚ)) = 1/6 ∧ y = 81 := by
  sorry

end solve_for_y_l1730_173019


namespace initial_principal_is_8000_l1730_173069

/-- The compound interest formula for annual compounding -/
def compound_interest (P : ℝ) (r : ℝ) (t : ℕ) : ℝ :=
  P * (1 + r) ^ t

/-- Theorem: Given the conditions of the problem, the initial principal is 8000 -/
theorem initial_principal_is_8000 :
  ∃ P : ℝ,
    compound_interest P 0.05 2 = 8820 ∧
    P = 8000 := by
  sorry

end initial_principal_is_8000_l1730_173069


namespace max_sum_with_constraints_l1730_173076

theorem max_sum_with_constraints (x y : ℝ) 
  (h1 : 3 * x + 2 * y ≤ 7) 
  (h2 : 2 * x + 4 * y ≤ 8) : 
  x + y ≤ 11 / 4 := by
  sorry

end max_sum_with_constraints_l1730_173076


namespace factorization_proof_l1730_173035

theorem factorization_proof (a b : ℝ) : a * b^2 - 5 * a * b = a * b * (b - 5) := by
  sorry

end factorization_proof_l1730_173035


namespace power_of_power_three_l1730_173036

theorem power_of_power_three : (3^2)^4 = 6561 := by
  sorry

end power_of_power_three_l1730_173036


namespace product_calculation_l1730_173053

theorem product_calculation : 1500 * 2023 * 0.5023 * 50 = 306903675 := by
  sorry

end product_calculation_l1730_173053


namespace red_balloons_count_l1730_173034

def total_balloons : ℕ := 17
def green_balloons : ℕ := 9

theorem red_balloons_count :
  total_balloons - green_balloons = 8 := by
  sorry

end red_balloons_count_l1730_173034


namespace tory_cookie_sales_l1730_173028

/-- Proves that Tory sold 7 packs of cookies to his uncle given the problem conditions -/
theorem tory_cookie_sales : 
  ∀ (total_goal : ℕ) (sold_to_grandmother : ℕ) (sold_to_neighbor : ℕ) (remaining_to_sell : ℕ),
    total_goal = 50 →
    sold_to_grandmother = 12 →
    sold_to_neighbor = 5 →
    remaining_to_sell = 26 →
    ∃ (sold_to_uncle : ℕ),
      sold_to_uncle = total_goal - remaining_to_sell - sold_to_grandmother - sold_to_neighbor ∧
      sold_to_uncle = 7 :=
by sorry

end tory_cookie_sales_l1730_173028


namespace percentage_calculation_l1730_173018

theorem percentage_calculation (total : ℝ) (part : ℝ) (h1 : total = 500) (h2 : part = 125) :
  (part / total) * 100 = 25 := by
  sorry

end percentage_calculation_l1730_173018


namespace solution_pairs_l1730_173016

-- Define the predicate for the conditions
def satisfies_conditions (x y : ℕ+) : Prop :=
  (y ∣ x^2 + 1) ∧ (x^2 ∣ y^3 + 1)

-- State the theorem
theorem solution_pairs :
  ∀ x y : ℕ+, satisfies_conditions x y →
    ((x = 1 ∧ y = 1) ∨ (x = 1 ∧ y = 2) ∨ (x = 3 ∧ y = 2)) :=
by sorry

end solution_pairs_l1730_173016


namespace linear_function_quadrants_l1730_173056

/-- A linear function passing through three given points. -/
def LinearFunction (k b : ℝ) : ℝ → ℝ := fun x ↦ k * x + b

/-- The function passes through the given points. -/
def PassesThroughPoints (k b : ℝ) : Prop :=
  LinearFunction k b (-2) = 7 ∧
  LinearFunction k b 1 = 4 ∧
  LinearFunction k b 3 = 2

/-- The function passes through the first, second, and fourth quadrants. -/
def PassesThroughQuadrants (k b : ℝ) : Prop :=
  (∃ x y, x > 0 ∧ y > 0 ∧ LinearFunction k b x = y) ∧
  (∃ x y, x < 0 ∧ y > 0 ∧ LinearFunction k b x = y) ∧
  (∃ x y, x > 0 ∧ y < 0 ∧ LinearFunction k b x = y)

/-- Main theorem: If the linear function passes through the given points,
    then it passes through the first, second, and fourth quadrants. -/
theorem linear_function_quadrants (k b : ℝ) (h : k ≠ 0) :
  PassesThroughPoints k b → PassesThroughQuadrants k b :=
by
  sorry


end linear_function_quadrants_l1730_173056


namespace petyas_class_l1730_173070

theorem petyas_class (x y : ℕ) : 
  (2 * x : ℚ) / 3 + y / 7 = (x + y : ℚ) / 3 →  -- Condition 1, 2, 3
  x + y ≤ 40 →                                -- Condition 4
  x = 12                                      -- Conclusion
  := by sorry

end petyas_class_l1730_173070


namespace largest_power_dividing_factorial_l1730_173050

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem largest_power_dividing_factorial :
  ∃ (n : ℕ), n = 7 ∧ 
  (∀ m : ℕ, m > n → ¬(factorial 30 % (18^m) = 0)) ∧
  (factorial 30 % (18^n) = 0) :=
by sorry

end largest_power_dividing_factorial_l1730_173050


namespace min_value_a_plus_2b_min_value_equals_3_plus_2sqrt2_l1730_173031

theorem min_value_a_plus_2b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = 1) :
  ∀ x y : ℝ, x > 0 → y > 0 → 1/x + 1/y = 1 → a + 2*b ≤ x + 2*y :=
by sorry

theorem min_value_equals_3_plus_2sqrt2 (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = 1) :
  a + 2*b = 3 + 2*Real.sqrt 2 :=
by sorry

end min_value_a_plus_2b_min_value_equals_3_plus_2sqrt2_l1730_173031


namespace common_chord_equation_l1730_173075

/-- Two circles in a 2D plane -/
structure TwoCircles where
  a : ℝ
  b : ℝ

/-- The equation of a line in 2D -/
structure LineEquation where
  A : ℝ
  B : ℝ
  C : ℝ

/-- Given two circles with a common chord of length 1, 
    prove that the equation of the common chord is 2ax + 2by - 3 = 0 -/
theorem common_chord_equation (circles : TwoCircles) : 
  ∃ (line : LineEquation), 
    line.A = 2 * circles.a ∧ 
    line.B = 2 * circles.b ∧ 
    line.C = -3 := by
  sorry

end common_chord_equation_l1730_173075


namespace original_number_is_four_fifths_l1730_173029

theorem original_number_is_four_fifths (x : ℚ) :
  1 + 1 / x = 9 / 4 → x = 4 / 5 := by
  sorry

end original_number_is_four_fifths_l1730_173029


namespace number_problem_l1730_173021

theorem number_problem (x : ℝ) : x - (3/5) * x = 64 → x = 160 := by
  sorry

end number_problem_l1730_173021


namespace cube_sum_theorem_l1730_173014

def cube_numbers (start : ℕ) : List ℕ := List.range 6 |>.map (· + start)

def opposite_faces_sum_equal (numbers : List ℕ) : Prop :=
  numbers.length = 6 ∧ 
  ∃ (sum : ℕ), 
    numbers[0]! + numbers[5]! = sum ∧
    numbers[1]! + numbers[4]! = sum ∧
    numbers[2]! + numbers[3]! = sum

theorem cube_sum_theorem :
  let numbers := cube_numbers 15
  opposite_faces_sum_equal numbers →
  numbers.sum = 105 := by
sorry

end cube_sum_theorem_l1730_173014


namespace find_divisor_l1730_173013

theorem find_divisor (dividend : Nat) (quotient : Nat) (h1 : dividend = 62976) (h2 : quotient = 123) :
  dividend / quotient = 512 := by
  sorry

end find_divisor_l1730_173013


namespace family_ages_l1730_173042

/-- Given the ages and relationships of family members, prove the ages of the younger siblings after 30 years -/
theorem family_ages (elder_son_age : ℕ) (declan_age_diff : ℕ) (younger_son_age_diff : ℕ) (third_sibling_age_diff : ℕ) (years_later : ℕ)
  (h1 : elder_son_age = 40)
  (h2 : declan_age_diff = 25)
  (h3 : younger_son_age_diff = 10)
  (h4 : third_sibling_age_diff = 5)
  (h5 : years_later = 30) :
  let younger_son_age := elder_son_age - younger_son_age_diff
  let third_sibling_age := younger_son_age - third_sibling_age_diff
  (younger_son_age + years_later = 60) ∧ (third_sibling_age + years_later = 55) :=
by sorry

end family_ages_l1730_173042


namespace division_remainder_problem_l1730_173007

theorem division_remainder_problem : ∃ (x : ℕ+), 
  19250 % x.val = 11 ∧ 
  20302 % x.val = 3 ∧ 
  x.val = 53 := by
  sorry

end division_remainder_problem_l1730_173007


namespace isosceles_right_triangle_locus_l1730_173010

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the square of the distance between two points -/
def distanceSquared (p1 p2 : Point) : ℝ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

/-- Theorem: Locus of points for isosceles right triangle -/
theorem isosceles_right_triangle_locus (s : ℝ) (h : s > 0) :
  let A : Point := ⟨0, 0⟩
  let B : Point := ⟨s, 0⟩
  let C : Point := ⟨0, s⟩
  let center : Point := ⟨s/3, s/3⟩
  let radius : ℝ := Real.sqrt (s^2/3)
  ∀ P : Point, 
    (distanceSquared P A + distanceSquared P B + distanceSquared P C = 4 * s^2) ↔ 
    (distanceSquared P center = radius^2) := by
  sorry

end isosceles_right_triangle_locus_l1730_173010

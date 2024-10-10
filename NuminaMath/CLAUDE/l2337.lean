import Mathlib

namespace a_geq_one_l2337_233789

-- Define the conditions
def p (x : ℝ) : Prop := |x + 1| > 2
def q (x a : ℝ) : Prop := x > a

-- Define the relationship between p and q
axiom not_p_sufficient_for_not_q : ∀ x a : ℝ, (¬p x → ¬q x a) ∧ ∃ x a : ℝ, ¬p x ∧ q x a

-- Theorem to prove
theorem a_geq_one : ∀ a : ℝ, (∀ x : ℝ, q x a → p x) → a ≥ 1 := by
  sorry

end a_geq_one_l2337_233789


namespace m_range_theorem_l2337_233769

-- Define the quadratic function p
def p (x : ℝ) : Prop := x^2 - 8*x - 20 > 0

-- Define the function q
def q (x m : ℝ) : Prop := (x - (1 - m)) * (x - (1 + m)) > 0

-- State the theorem
theorem m_range_theorem (h_sufficient : ∀ x m : ℝ, p x → q x m) 
                        (h_not_necessary : ∃ x m : ℝ, q x m ∧ ¬(p x))
                        (h_m_positive : m > 0) :
  0 < m ∧ m ≤ 3 :=
sorry

end m_range_theorem_l2337_233769


namespace quadratic_sum_l2337_233788

/-- A quadratic function with specific properties -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_sum (a b c : ℝ) :
  (∃ (x_max : ℝ), ∀ x, QuadraticFunction a b c x ≤ QuadraticFunction a b c x_max ∧
    QuadraticFunction a b c x_max = 72) →
  QuadraticFunction a b c 0 = -1 →
  QuadraticFunction a b c 6 = -1 →
  a + b + c = 356 / 9 := by
  sorry

end quadratic_sum_l2337_233788


namespace evaluate_expression_l2337_233721

theorem evaluate_expression : ((3^5 / 3^2) * 2^10) + (1/2) = 27648.5 := by
  sorry

end evaluate_expression_l2337_233721


namespace quadratic_inequality_all_reals_l2337_233700

/-- The quadratic inequality ax^2 + bx + c > 0 has all real numbers as its solution set
    if and only if a > 0 and the discriminant is negative. -/
theorem quadratic_inequality_all_reals 
  (a b c : ℝ) (Δ : ℝ) (hΔ : Δ = b^2 - 4*a*c) :
  (∀ x : ℝ, a * x^2 + b * x + c > 0) ↔ (a > 0 ∧ Δ < 0) :=
sorry

end quadratic_inequality_all_reals_l2337_233700


namespace intersection_of_M_and_N_l2337_233761

-- Define the set M
def M : Set ℝ := {α | ∃ k : ℤ, α = k * 90 - 36}

-- Define the set N
def N : Set ℝ := {α | -180 < α ∧ α < 180}

-- Define the intersection set
def intersection : Set ℝ := {-36, 54, -126, 144}

-- Theorem statement
theorem intersection_of_M_and_N : M ∩ N = intersection := by sorry

end intersection_of_M_and_N_l2337_233761


namespace cube_surface_area_from_prisms_l2337_233746

-- Define the dimensions of a single prism
def prism_length : ℝ := 10
def prism_width : ℝ := 3
def prism_height : ℝ := 30

-- Define the number of prisms
def num_prisms : ℕ := 2

-- Theorem statement
theorem cube_surface_area_from_prisms :
  let prism_volume := prism_length * prism_width * prism_height
  let total_volume := num_prisms * prism_volume
  let cube_edge := total_volume ^ (1/3 : ℝ)
  let cube_surface_area := 6 * cube_edge^2
  cube_surface_area = 600 := by sorry

end cube_surface_area_from_prisms_l2337_233746


namespace rectangular_hall_area_l2337_233720

theorem rectangular_hall_area (length width : ℝ) : 
  width = (1/2) * length →
  length - width = 10 →
  length * width = 200 := by
sorry

end rectangular_hall_area_l2337_233720


namespace intersection_with_complement_of_B_l2337_233718

open Set

theorem intersection_with_complement_of_B (U A B : Set ℕ) : 
  U = {1, 2, 3, 4, 5, 6} →
  A = {2, 4, 6} →
  B = {1, 3} →
  A ∩ (U \ B) = {2, 4, 6} := by
sorry

end intersection_with_complement_of_B_l2337_233718


namespace plot_perimeter_l2337_233792

/-- A rectangular plot with specific dimensions and fencing costs -/
structure RectangularPlot where
  width : ℝ
  length : ℝ
  fencingCostPerMeter : ℝ
  totalFencingCost : ℝ
  length_eq : length = width + 10
  cost_eq : totalFencingCost = fencingCostPerMeter * (2 * (length + width))

/-- The perimeter of the rectangular plot is 340 meters -/
theorem plot_perimeter (plot : RectangularPlot) (h : plot.fencingCostPerMeter = 6.5 ∧ plot.totalFencingCost = 2210) :
  2 * (plot.length + plot.width) = 340 := by
  sorry


end plot_perimeter_l2337_233792


namespace steps_calculation_l2337_233726

/-- The number of steps Benjamin took from the hotel to Times Square. -/
def total_steps : ℕ := 582

/-- The number of steps Benjamin took from the hotel to Rockefeller Center. -/
def steps_to_rockefeller : ℕ := 354

/-- The number of steps Benjamin took from Rockefeller Center to Times Square. -/
def steps_rockefeller_to_times_square : ℕ := total_steps - steps_to_rockefeller

theorem steps_calculation :
  steps_rockefeller_to_times_square = 228 :=
by sorry

end steps_calculation_l2337_233726


namespace unique_perfect_square_and_cube_factor_of_1800_l2337_233795

/-- A number is a perfect square if it's equal to some integer squared. -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k^2

/-- A number is a perfect cube if it's equal to some integer cubed. -/
def is_perfect_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k^3

/-- A number is both a perfect square and a perfect cube. -/
def is_perfect_square_and_cube (n : ℕ) : Prop :=
  is_perfect_square n ∧ is_perfect_cube n

/-- The set of positive factors of a natural number. -/
def positive_factors (n : ℕ) : Set ℕ :=
  {k : ℕ | k > 0 ∧ n % k = 0}

/-- There is exactly one positive factor of 1800 that is both a perfect square and a perfect cube. -/
theorem unique_perfect_square_and_cube_factor_of_1800 :
  ∃! x : ℕ, x ∈ positive_factors 1800 ∧ is_perfect_square_and_cube x :=
sorry

end unique_perfect_square_and_cube_factor_of_1800_l2337_233795


namespace poultry_farm_loss_l2337_233730

theorem poultry_farm_loss (initial_chickens initial_turkeys initial_guinea_fowls : ℕ)
  (daily_turkey_loss daily_guinea_fowl_loss : ℕ)
  (total_birds_after_week : ℕ)
  (h1 : initial_chickens = 300)
  (h2 : initial_turkeys = 200)
  (h3 : initial_guinea_fowls = 80)
  (h4 : daily_turkey_loss = 8)
  (h5 : daily_guinea_fowl_loss = 5)
  (h6 : total_birds_after_week = 349)
  (h7 : initial_chickens + initial_turkeys + initial_guinea_fowls
      - (7 * daily_turkey_loss + 7 * daily_guinea_fowl_loss)
      - total_birds_after_week = 7 * daily_chicken_loss) :
  daily_chicken_loss = 20 := by sorry

end poultry_farm_loss_l2337_233730


namespace prob_at_least_two_different_fruits_l2337_233714

def num_fruit_types : ℕ := 4
def num_meals : ℕ := 4

def prob_same_fruit : ℚ := (1 / num_fruit_types) ^ num_meals * num_fruit_types

theorem prob_at_least_two_different_fruits :
  1 - prob_same_fruit = 63 / 64 := by sorry

end prob_at_least_two_different_fruits_l2337_233714


namespace mary_final_cards_l2337_233775

def initial_cards : ℕ := 18
def torn_cards : ℕ := 8
def cards_from_fred : ℕ := 26
def cards_bought : ℕ := 40
def cards_exchanged : ℕ := 10
def cards_lost : ℕ := 5

theorem mary_final_cards : 
  initial_cards - torn_cards + cards_from_fred + cards_bought - cards_lost = 71 := by
  sorry

end mary_final_cards_l2337_233775


namespace arithmetic_progression_product_power_l2337_233705

theorem arithmetic_progression_product_power : ∃ (a b : ℕ), 
  a > 0 ∧ 
  (a * (2*a) * (3*a) * (4*a) * (5*a) = b^2008) := by
  sorry

end arithmetic_progression_product_power_l2337_233705


namespace convex_n_gon_division_possible_values_l2337_233741

/-- A convex n-gon divided into three convex polygons -/
structure ConvexNGonDivision (n : ℕ) where
  (polygon1 : ℕ)  -- Number of sides of the first polygon
  (polygon2 : ℕ)  -- Number of sides of the second polygon
  (polygon3 : ℕ)  -- Number of sides of the third polygon
  (h1 : polygon1 = n)  -- First polygon has n sides
  (h2 : polygon2 > n)  -- Second polygon has more than n sides
  (h3 : polygon3 < n)  -- Third polygon has fewer than n sides

/-- The theorem stating the possible values of n -/
theorem convex_n_gon_division_possible_values :
  ∀ n : ℕ, (∃ d : ConvexNGonDivision n, True) → n = 4 ∨ n = 5 :=
sorry

end convex_n_gon_division_possible_values_l2337_233741


namespace inscribed_circles_chord_length_l2337_233780

/-- Given two circles, one inscribed in an angle α with radius r and another of radius R 
    touching one side of the angle at the same point as the first circle and intersecting 
    the other side at points A and B, the length of AB can be calculated. -/
theorem inscribed_circles_chord_length (α r R : ℝ) (h_pos_r : r > 0) (h_pos_R : R > 0) :
  ∃ (AB : ℝ), AB = 4 * Real.cos (α / 2) * Real.sqrt ((R - r) * (R * Real.sin (α / 2)^2 + r * Real.cos (α / 2)^2)) := by
  sorry

end inscribed_circles_chord_length_l2337_233780


namespace smallest_three_digit_palindrome_not_six_digit_palindrome_product_l2337_233790

/-- Checks if a number is a three-digit palindrome -/
def isThreeDigitPalindrome (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ (n / 100 = n % 10)

/-- Checks if a number is a six-digit palindrome -/
def isSixDigitPalindrome (n : ℕ) : Prop :=
  100000 ≤ n ∧ n ≤ 999999 ∧ (n / 100000 = n % 10) ∧ ((n / 10000) % 10 = (n / 10) % 10) ∧ ((n / 1000) % 10 = (n / 100) % 10)

/-- The main theorem -/
theorem smallest_three_digit_palindrome_not_six_digit_palindrome_product :
  isThreeDigitPalindrome 404 ∧
  ¬(isSixDigitPalindrome (404 * 102)) ∧
  ∀ n : ℕ, isThreeDigitPalindrome n → n < 404 → isSixDigitPalindrome (n * 102) :=
by sorry

end smallest_three_digit_palindrome_not_six_digit_palindrome_product_l2337_233790


namespace range_of_a_l2337_233760

theorem range_of_a (p q : Prop) (h_p : ∀ x ∈ Set.Icc 1 2, 2 * x^2 - a ≥ 0)
  (h_q : ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0) (h_pq : p ∧ q) :
  a ≤ -2 ∨ (1 ≤ a ∧ a ≤ 2) :=
sorry

end range_of_a_l2337_233760


namespace angle_in_full_rotation_l2337_233764

theorem angle_in_full_rotation (y : ℝ) : y + 90 = 360 → y = 270 := by
  sorry

end angle_in_full_rotation_l2337_233764


namespace class_size_l2337_233743

theorem class_size (s : ℕ) (r : ℕ) : 
  (0 * 2 + 1 * 12 + 2 * 10 + 3 * r) / s = 2 →
  s = 2 + 12 + 10 + r →
  s = 40 :=
by sorry

end class_size_l2337_233743


namespace power_multiplication_equals_512_l2337_233777

theorem power_multiplication_equals_512 : 2^3 * 2^6 = 512 := by
  sorry

end power_multiplication_equals_512_l2337_233777


namespace base_6_representation_of_231_base_6_to_decimal_l2337_233763

/-- Converts a natural number to its base 6 representation --/
def toBase6 (n : ℕ) : List ℕ :=
  if n < 6 then [n]
  else (n % 6) :: toBase6 (n / 6)

/-- Converts a list of digits in base 6 to a natural number --/
def fromBase6 (digits : List ℕ) : ℕ :=
  digits.foldr (fun d acc => d + 6 * acc) 0

theorem base_6_representation_of_231 :
  toBase6 231 = [3, 2, 0, 1] :=
sorry

theorem base_6_to_decimal :
  fromBase6 [3, 2, 0, 1] = 231 :=
sorry

end base_6_representation_of_231_base_6_to_decimal_l2337_233763


namespace problem_statement_l2337_233787

theorem problem_statement (x y : ℝ) (hx : x = 12) (hy : y = 7) :
  (x - y) * (2 * x + y) = 155 := by
  sorry

end problem_statement_l2337_233787


namespace manufacturer_measures_l2337_233715

def samples_A : List ℝ := [3, 4, 5, 6, 8, 8, 8, 10]
def samples_B : List ℝ := [4, 6, 6, 6, 8, 9, 12, 13]
def samples_C : List ℝ := [3, 3, 4, 7, 9, 10, 11, 12]

def claimed_lifespan : ℝ := 8

def mode (l : List ℝ) : ℝ := sorry
def mean (l : List ℝ) : ℝ := sorry
def median (l : List ℝ) : ℝ := sorry

theorem manufacturer_measures :
  mode samples_A = claimed_lifespan ∧
  mean samples_B = claimed_lifespan ∧
  median samples_C = claimed_lifespan :=
sorry

end manufacturer_measures_l2337_233715


namespace no_function_satisfies_condition_l2337_233708

theorem no_function_satisfies_condition :
  ¬ ∃ f : ℝ → ℝ, ∀ x y z : ℝ, f (x * y) + f (x * z) - f x * f (y * z) > 1 := by
  sorry

end no_function_satisfies_condition_l2337_233708


namespace zero_neither_positive_nor_negative_l2337_233793

theorem zero_neither_positive_nor_negative :
  ¬(0 > 0) ∧ ¬(0 < 0) :=
by sorry

end zero_neither_positive_nor_negative_l2337_233793


namespace green_pill_cost_l2337_233745

theorem green_pill_cost (daily_green : ℕ) (daily_pink : ℕ) (days : ℕ) 
  (green_pink_diff : ℚ) (total_cost : ℚ) :
  daily_green = 2 →
  daily_pink = 1 →
  days = 21 →
  green_pink_diff = 1 →
  total_cost = 819 →
  ∃ (green_cost : ℚ), 
    green_cost = 40 / 3 ∧ 
    (daily_green * green_cost + daily_pink * (green_cost - green_pink_diff)) * days = total_cost :=
by sorry

end green_pill_cost_l2337_233745


namespace montoya_family_budget_l2337_233756

theorem montoya_family_budget (budget : ℝ) 
  (grocery_fraction : ℝ) (total_food_fraction : ℝ) :
  grocery_fraction = 0.6 →
  total_food_fraction = 0.8 →
  total_food_fraction = grocery_fraction + (budget - grocery_fraction * budget) / budget →
  (budget - grocery_fraction * budget) / budget = 0.2 := by
  sorry

end montoya_family_budget_l2337_233756


namespace rhombus_perimeter_rhombus_perimeter_is_20_l2337_233757

/-- The perimeter of a rhombus whose diagonals are the roots of x^2 - 14x + 48 = 0 -/
theorem rhombus_perimeter : ℝ → Prop :=
  fun p =>
    ∀ (x₁ x₂ : ℝ),
      x₁^2 - 14*x₁ + 48 = 0 →
      x₂^2 - 14*x₂ + 48 = 0 →
      x₁ ≠ x₂ →
      let s := Real.sqrt ((x₁^2 + x₂^2) / 4)
      p = 4 * s

/-- The perimeter of the rhombus is 20 -/
theorem rhombus_perimeter_is_20 : rhombus_perimeter 20 := by
  sorry

end rhombus_perimeter_rhombus_perimeter_is_20_l2337_233757


namespace jungkook_paper_count_l2337_233771

/-- Calculates the total number of pieces of colored paper given the number of bundles,
    pieces per bundle, and individual pieces. -/
def total_pieces (bundles : ℕ) (pieces_per_bundle : ℕ) (individual_pieces : ℕ) : ℕ :=
  bundles * pieces_per_bundle + individual_pieces

/-- Proves that given 3 bundles of 10 pieces each and 8 individual pieces,
    the total number of pieces is 38. -/
theorem jungkook_paper_count :
  total_pieces 3 10 8 = 38 := by
  sorry

end jungkook_paper_count_l2337_233771


namespace solution_set_characterization_l2337_233798

def quadratic_inequality (a : ℝ) (x : ℝ) : Prop :=
  x^2 - 2*a*x + a + 2 ≤ 0

def solution_set (a : ℝ) : Set ℝ :=
  {x : ℝ | quadratic_inequality a x}

theorem solution_set_characterization (a : ℝ) :
  (solution_set a ⊆ Set.Icc 1 3) ↔ a ∈ Set.Ioo (-1) (11/5) :=
sorry

end solution_set_characterization_l2337_233798


namespace triangle_circumcircle_radius_l2337_233707

theorem triangle_circumcircle_radius (a b c : ℝ) (h1 : a = 3) (h2 : b = 5) (h3 : c = 7) :
  let R := c / (2 * Real.sqrt (1 - ((a^2 + b^2 - c^2) / (2 * a * b))^2))
  R = 7 * Real.sqrt 3 / 3 := by sorry

end triangle_circumcircle_radius_l2337_233707


namespace chess_club_election_proof_l2337_233766

def total_candidates : ℕ := 20
def previous_board_members : ℕ := 10
def board_positions : ℕ := 6

theorem chess_club_election_proof :
  (Nat.choose total_candidates board_positions) - 
  (Nat.choose (total_candidates - previous_board_members) board_positions) = 38550 := by
  sorry

end chess_club_election_proof_l2337_233766


namespace determinant_in_terms_of_coefficients_l2337_233716

theorem determinant_in_terms_of_coefficients 
  (s p q : ℝ) (a b c : ℝ) 
  (h1 : a^3 + s*a^2 + p*a + q = 0)
  (h2 : b^3 + s*b^2 + p*b + q = 0)
  (h3 : c^3 + s*c^2 + p*c + q = 0) :
  let M : Matrix (Fin 3) (Fin 3) ℝ := !![1+a, 1, 1; 1, 1+b, 1; 1, 1, 1+c]
  Matrix.det M = -q + p - s := by
  sorry

end determinant_in_terms_of_coefficients_l2337_233716


namespace intersection_complement_theorem_l2337_233737

-- Define the sets A and B
def A : Set ℝ := {x | |x + 3| - |x - 3| > 3}
def B : Set ℝ := {x | ∃ t > 0, x = (t^2 - 4*t + 1) / t}

-- State the theorem
theorem intersection_complement_theorem : B ∩ (Set.univ \ A) = Set.Icc (-2) (3/2) := by sorry

end intersection_complement_theorem_l2337_233737


namespace peters_height_l2337_233797

theorem peters_height (tree_height : ℝ) (tree_shadow : ℝ) (peter_shadow : ℝ) :
  tree_height = 100 → 
  tree_shadow = 25 → 
  peter_shadow = 1.5 → 
  (tree_height / tree_shadow) * peter_shadow * 12 = 72 := by
  sorry

end peters_height_l2337_233797


namespace factorization_of_difference_of_squares_l2337_233729

theorem factorization_of_difference_of_squares (x y : ℝ) :
  4 * x^2 - y^4 = (2*x + y^2) * (2*x - y^2) := by sorry

end factorization_of_difference_of_squares_l2337_233729


namespace min_value_f_l2337_233754

def f (a x : ℝ) : ℝ := x^2 - 2*a*x - 1

theorem min_value_f (a : ℝ) :
  (∀ x ∈ Set.Icc (-1) 1, f a x ≥ 
    (if a < -1 then 2*a
     else if a ≤ 1 then -1 - a^2
     else -2*a)) ∧
  (∃ x ∈ Set.Icc (-1) 1, f a x = 
    (if a < -1 then 2*a
     else if a ≤ 1 then -1 - a^2
     else -2*a)) := by
  sorry

end min_value_f_l2337_233754


namespace sufficient_not_necessary_l2337_233702

theorem sufficient_not_necessary (a b : ℝ) : 
  ((a > b ∧ b > 1) → (a - b < a^2 - b^2)) ∧ 
  ¬((a - b < a^2 - b^2) → (a > b ∧ b > 1)) := by
sorry

end sufficient_not_necessary_l2337_233702


namespace circle_center_perpendicular_line_l2337_233759

-- Define the circle C
def circle_C : Set (ℝ × ℝ) := sorry

-- Define the center of the circle
def center : ℝ × ℝ := sorry

-- Define the line l
def line_l (x y : ℝ) : Prop := y = x - 1

-- Define the perpendicular line passing through the center
def perpendicular_line (x y : ℝ) : Prop := x + y - 3 = 0

theorem circle_center_perpendicular_line :
  (1, 0) ∈ circle_C ∧
  center.1 > 0 ∧
  center.2 = 0 ∧
  (∃ (a b : ℝ), (a, b) ∈ circle_C ∧ line_l a b ∧
    (a - center.1)^2 + (b - center.2)^2 = 8) →
  ∀ x y, perpendicular_line x y ↔ 
    (x - center.1) * 1 + (y - center.2) * 1 = 0 ∧
    center ∈ ({p : ℝ × ℝ | perpendicular_line p.1 p.2} : Set (ℝ × ℝ)) :=
by sorry

end circle_center_perpendicular_line_l2337_233759


namespace fruit_bowl_problem_l2337_233704

theorem fruit_bowl_problem (initial_oranges : ℕ) : 
  (14 : ℝ) / ((14 : ℝ) + (initial_oranges - 19)) = 0.7 → 
  initial_oranges = 25 := by
  sorry

end fruit_bowl_problem_l2337_233704


namespace cubic_root_ratio_l2337_233753

theorem cubic_root_ratio (r : ℝ) (h : r > 1) : 
  (∃ a b c : ℝ, 
    (81 * a^3 - 243 * a^2 + 216 * a - 64 = 0) ∧ 
    (81 * b^3 - 243 * b^2 + 216 * b - 64 = 0) ∧ 
    (81 * c^3 - 243 * c^2 + 216 * c - 64 = 0) ∧ 
    (b = a * r) ∧ 
    (c = b * r)) → 
  (c / a = r^2) :=
by sorry

end cubic_root_ratio_l2337_233753


namespace angles_with_same_terminal_side_as_45_l2337_233782

def same_terminal_side (α β : ℝ) : Prop :=
  ∃ k : ℤ, β = α + k * 360

theorem angles_with_same_terminal_side_as_45 :
  ∀ θ : ℝ, -720 ≤ θ ∧ θ < 0 ∧ same_terminal_side 45 θ →
    θ = -675 ∨ θ = -315 := by sorry

end angles_with_same_terminal_side_as_45_l2337_233782


namespace variance_scaling_l2337_233713

/-- Given a set of data points, this function returns the variance of the data set. -/
noncomputable def variance (data : Finset ℝ) : ℝ := sorry

/-- Given a set of data points, this function multiplies each point by a scalar. -/
def scaleData (data : Finset ℝ) (scalar : ℝ) : Finset ℝ := sorry

theorem variance_scaling (data : Finset ℝ) (s : ℝ) :
  variance data = s^2 → variance (scaleData data 2) = 4 * s^2 := by sorry

end variance_scaling_l2337_233713


namespace principal_is_15000_l2337_233762

/-- Calculates the principal amount given simple interest, rate, and time -/
def calculate_principal (interest : ℚ) (rate : ℚ) (time : ℚ) : ℚ :=
  (interest * 100) / (rate * time)

/-- Theorem: Given the specified conditions, the principal sum is 15000 -/
theorem principal_is_15000 :
  let interest : ℚ := 2700
  let rate : ℚ := 6
  let time : ℚ := 3
  calculate_principal interest rate time = 15000 := by
  sorry

end principal_is_15000_l2337_233762


namespace number_difference_l2337_233719

theorem number_difference (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) :
  |x - y| = 4 := by
sorry

end number_difference_l2337_233719


namespace f_2x_l2337_233711

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 1

-- State the theorem
theorem f_2x (x : ℝ) : f (2 * x) = 4 * x^2 - 1 := by
  sorry

end f_2x_l2337_233711


namespace afternoon_eggs_l2337_233778

theorem afternoon_eggs (total_eggs day_eggs morning_eggs : ℕ) 
  (h1 : total_eggs = 1339)
  (h2 : morning_eggs = 816) :
  total_eggs - morning_eggs = 523 := by
  sorry

end afternoon_eggs_l2337_233778


namespace hannah_grapes_count_l2337_233725

def sophie_oranges_daily : ℕ := 20
def observation_days : ℕ := 30
def total_fruits : ℕ := 1800

def hannah_grapes_daily : ℕ := (total_fruits - sophie_oranges_daily * observation_days) / observation_days

theorem hannah_grapes_count : hannah_grapes_daily = 40 := by
  sorry

end hannah_grapes_count_l2337_233725


namespace train_length_l2337_233742

/-- Given a train traveling at a certain speed that crosses a bridge of known length in a specific time,
    this theorem proves the length of the train. -/
theorem train_length
  (train_speed : ℝ)
  (bridge_length : ℝ)
  (crossing_time : ℝ)
  (h1 : train_speed = 45)  -- km/hr
  (h2 : bridge_length = 235)  -- meters
  (h3 : crossing_time = 30)  -- seconds
  : ∃ (train_length : ℝ), train_length = 140 := by
  sorry

end train_length_l2337_233742


namespace average_glasses_per_box_l2337_233712

/-- Proves that the average number of glasses per box is 15 given the specified conditions -/
theorem average_glasses_per_box (small_box_count : ℕ) (large_box_count : ℕ) : 
  large_box_count = small_box_count + 16 →
  12 * small_box_count + 16 * large_box_count = 480 →
  (480 : ℚ) / (small_box_count + large_box_count) = 15 := by
sorry

end average_glasses_per_box_l2337_233712


namespace number_operations_l2337_233731

theorem number_operations (x : ℝ) : (3 * ((x - 50) / 4) + 28 = 73) ↔ (x = 110) := by
  sorry

end number_operations_l2337_233731


namespace carrot_theorem_l2337_233767

def carrot_problem (initial_carrots additional_carrots final_total : ℕ) : Prop :=
  initial_carrots + additional_carrots - final_total = 4

theorem carrot_theorem : carrot_problem 19 46 61 := by
  sorry

end carrot_theorem_l2337_233767


namespace intersection_of_sets_l2337_233723

theorem intersection_of_sets : 
  let A : Set ℕ := {1, 2, 3}
  let B : Set ℕ := {3, 4}
  A ∩ B = {3} := by
sorry

end intersection_of_sets_l2337_233723


namespace exist_tetrahedra_volume_area_paradox_l2337_233751

/-- A tetrahedron in 3D space -/
structure Tetrahedron where
  vertices : Fin 4 → ℝ × ℝ × ℝ

/-- Calculate the volume of a tetrahedron -/
def volume (t : Tetrahedron) : ℝ := sorry

/-- Calculate the area of a face of a tetrahedron -/
def face_area (t : Tetrahedron) (face : Fin 4) : ℝ := sorry

/-- Theorem: There exist two tetrahedra such that one has greater volume
    but smaller or equal face areas compared to the other -/
theorem exist_tetrahedra_volume_area_paradox :
  ∃ (t₁ t₂ : Tetrahedron),
    volume t₁ > volume t₂ ∧
    ∀ (face₁ : Fin 4), ∃ (face₂ : Fin 4),
      face_area t₁ face₁ ≤ face_area t₂ face₂ :=
sorry

end exist_tetrahedra_volume_area_paradox_l2337_233751


namespace simplify_expressions_l2337_233772

variable (a b t : ℝ)

theorem simplify_expressions :
  (6 * a^2 - 2 * a * b - 2 * (3 * a^2 - (1/2) * a * b) = -a * b) ∧
  (-(t^2 - t - 1) + (2 * t^2 - 3 * t + 1) = t^2 - 2 * t + 2) := by
  sorry

end simplify_expressions_l2337_233772


namespace inverse_proportion_relation_l2337_233774

theorem inverse_proportion_relation :
  ∀ (y₁ y₂ y₃ : ℝ),
    y₁ = 1 / (-1) →
    y₂ = 1 / (-2) →
    y₃ = 1 / 3 →
    y₃ > y₂ ∧ y₂ > y₁ :=
by
  sorry

end inverse_proportion_relation_l2337_233774


namespace polynomial_simplification_l2337_233784

theorem polynomial_simplification (x : ℝ) : 
  (3 * x^2 + 4 * x - 5) - (2 * x^2 - 3 * x + 8) = x^2 + 7 * x - 13 := by
sorry

end polynomial_simplification_l2337_233784


namespace relationship_abc_l2337_233779

theorem relationship_abc : 3^(1/5) > 0.3^2 ∧ 0.3^2 > Real.log 0.3 / Real.log 2 := by sorry

end relationship_abc_l2337_233779


namespace circles_externally_tangent_l2337_233755

def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 2*y + 1 = 0

def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 4*y - 1 = 0

def center1 : ℝ × ℝ := (2, -1)

def center2 : ℝ × ℝ := (-2, 2)

def radius1 : ℝ := 2

def radius2 : ℝ := 3

theorem circles_externally_tangent :
  let d := Real.sqrt ((center1.1 - center2.1)^2 + (center1.2 - center2.2)^2)
  d = radius1 + radius2 := by sorry

end circles_externally_tangent_l2337_233755


namespace lillian_candy_count_l2337_233722

/-- The total number of candies Lillian has after receiving candies from her father and best friend. -/
def lillian_total_candies (initial : ℕ) (father_gave : ℕ) (friend_multiplier : ℕ) : ℕ :=
  initial + father_gave + friend_multiplier * father_gave

/-- Theorem stating that Lillian will have 113 candies given the initial conditions. -/
theorem lillian_candy_count :
  lillian_total_candies 88 5 4 = 113 := by
  sorry

#eval lillian_total_candies 88 5 4

end lillian_candy_count_l2337_233722


namespace buses_needed_l2337_233733

theorem buses_needed (num_students : ℕ) (seats_per_bus : ℕ) (h1 : num_students = 14) (h2 : seats_per_bus = 2) :
  (num_students + seats_per_bus - 1) / seats_per_bus = 7 :=
by
  sorry

end buses_needed_l2337_233733


namespace largest_five_digit_sum_20_l2337_233770

def is_five_digit (n : ℕ) : Prop := n ≥ 10000 ∧ n < 100000

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  (List.sum digits)

theorem largest_five_digit_sum_20 :
  ∀ n : ℕ, is_five_digit n → digit_sum n = 20 → n ≤ 99200 :=
by sorry

end largest_five_digit_sum_20_l2337_233770


namespace max_area_rectangular_fence_l2337_233703

/-- Represents a rectangular fence with given constraints -/
structure RectangularFence where
  length : ℝ
  width : ℝ
  perimeter_constraint : length + width = 200
  length_constraint : length ≥ 100
  width_constraint : width ≥ 50

/-- Calculates the area of a rectangular fence -/
def area (fence : RectangularFence) : ℝ :=
  fence.length * fence.width

/-- Theorem stating the maximum area of the rectangular fence -/
theorem max_area_rectangular_fence :
  ∃ (fence : RectangularFence), ∀ (other : RectangularFence), area fence ≥ area other ∧ area fence = 10000 := by
  sorry

end max_area_rectangular_fence_l2337_233703


namespace difference_one_third_and_decimal_l2337_233799

theorem difference_one_third_and_decimal : (1 : ℚ) / 3 - 333 / 1000 = 1 / 3000 := by sorry

end difference_one_third_and_decimal_l2337_233799


namespace corresponding_angles_equal_l2337_233786

/-- Given two triangles ABC and A₁B₁C₁, where for each pair of corresponding angles,
    either the angles are equal or their sum is 180°, all corresponding angles are equal. -/
theorem corresponding_angles_equal 
  (α β γ α₁ β₁ γ₁ : ℝ) 
  (triangle_ABC : α + β + γ = 180)
  (triangle_A₁B₁C₁ : α₁ + β₁ + γ₁ = 180)
  (h1 : α = α₁ ∨ α + α₁ = 180)
  (h2 : β = β₁ ∨ β + β₁ = 180)
  (h3 : γ = γ₁ ∨ γ + γ₁ = 180) :
  α = α₁ ∧ β = β₁ ∧ γ = γ₁ := by
  sorry

end corresponding_angles_equal_l2337_233786


namespace triangle_angle_and_side_length_l2337_233735

theorem triangle_angle_and_side_length 
  (A B C : ℝ) 
  (a b c : ℝ) 
  (m n : ℝ × ℝ) :
  A > 0 ∧ A < π ∧
  B > 0 ∧ B < π ∧
  C > 0 ∧ C < π ∧
  A + B + C = π ∧
  m = (Real.sqrt 3, Real.cos A + 1) ∧
  n = (Real.sin A, -1) ∧
  m.1 * n.1 + m.2 * n.2 = 0 ∧
  a = 2 ∧
  Real.cos B = Real.sqrt 3 / 3 →
  A = π / 3 ∧ b = 4 * Real.sqrt 2 / 3 := by
sorry

end triangle_angle_and_side_length_l2337_233735


namespace inequality_and_range_l2337_233749

-- Define the function representing the left side of the inequality
def f (x : ℝ) : ℝ := |x - 3| + |x + 1|

-- State the theorem
theorem inequality_and_range : 
  (∀ x : ℝ, f x ≥ 4) ∧ 
  (∀ x : ℝ, f x = 4 ↔ -1 ≤ x ∧ x ≤ 3) :=
by sorry

end inequality_and_range_l2337_233749


namespace cannot_transform_to_target_l2337_233706

/-- Represents a natural number with its digits. -/
structure DigitNumber where
  digits : List Nat
  first_nonzero : digits.head? ≠ some 0

/-- Represents the allowed operations on the number. -/
inductive Operation
  | multiply_by_five
  | rearrange_digits

/-- Defines the target 150-digit number 5222...2223. -/
def target_number : DigitNumber := {
  digits := 5 :: List.replicate 148 2 ++ [2, 3]
  first_nonzero := by simp
}

/-- Applies an operation to a DigitNumber. -/
def apply_operation (n : DigitNumber) (op : Operation) : DigitNumber :=
  sorry

/-- Checks if a DigitNumber can be transformed into the target number using the allowed operations. -/
def can_transform (n : DigitNumber) : Prop :=
  ∃ (ops : List Operation), (ops.foldl apply_operation n) = target_number

/-- The initial number 1. -/
def initial_number : DigitNumber := {
  digits := [1]
  first_nonzero := by simp
}

/-- The main theorem stating that it's impossible to transform 1 into the target number. -/
theorem cannot_transform_to_target : ¬(can_transform initial_number) :=
  sorry

end cannot_transform_to_target_l2337_233706


namespace bank_interest_calculation_l2337_233710

def initial_deposit : ℝ := 5600
def interest_rate : ℝ := 0.07
def time_period : ℕ := 2

theorem bank_interest_calculation :
  let interest_per_year := initial_deposit * interest_rate
  let total_interest := interest_per_year * time_period
  initial_deposit + total_interest = 6384 := by
  sorry

end bank_interest_calculation_l2337_233710


namespace shopping_mall_probabilities_l2337_233773

/-- Probability of a customer buying product A -/
def prob_A : ℝ := 0.5

/-- Probability of a customer buying product B -/
def prob_B : ℝ := 0.6

/-- Probability of a customer buying neither product A nor B -/
def prob_neither : ℝ := (1 - prob_A) * (1 - prob_B)

/-- Probability of a customer buying at least one product -/
def prob_at_least_one : ℝ := 1 - prob_neither

theorem shopping_mall_probabilities :
  (1 - (prob_A * prob_B) - prob_neither = 0.5) ∧
  (1 - (prob_at_least_one^3 + 3 * prob_at_least_one^2 * prob_neither) = 0.104) :=
sorry

end shopping_mall_probabilities_l2337_233773


namespace sum_of_fractions_equals_one_l2337_233717

theorem sum_of_fractions_equals_one (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h : a + b + c = -a*b*c) :
  (a^2*b^2)/((a^2+b*c)*(b^2+a*c)) + (a^2*c^2)/((a^2+b*c)*(c^2+a*b)) +
  (b^2*c^2)/((b^2+a*c)*(c^2+a*b)) = 1 := by
sorry

end sum_of_fractions_equals_one_l2337_233717


namespace square_diagonal_shorter_path_l2337_233734

theorem square_diagonal_shorter_path (ε : Real) (h : ε > 0) : 
  ∃ (diff : Real), 
    abs (diff - 0.3) < ε ∧ 
    (2 - Real.sqrt 2) / 2 = diff :=
by sorry

end square_diagonal_shorter_path_l2337_233734


namespace taya_jenna_meet_l2337_233738

/-- The floor where Taya and Jenna meet -/
def meeting_floor : ℕ := 32

/-- The starting floor -/
def start_floor : ℕ := 22

/-- Time Jenna waits for the elevator (in seconds) -/
def wait_time : ℕ := 120

/-- Time Taya takes to go up one floor (in seconds) -/
def taya_time_per_floor : ℕ := 15

/-- Time the elevator takes to go up one floor (in seconds) -/
def elevator_time_per_floor : ℕ := 3

/-- Theorem stating that Taya and Jenna arrive at the meeting floor at the same time -/
theorem taya_jenna_meet :
  taya_time_per_floor * (meeting_floor - start_floor) =
  wait_time + elevator_time_per_floor * (meeting_floor - start_floor) :=
by sorry

end taya_jenna_meet_l2337_233738


namespace min_value_problem_l2337_233727

theorem min_value_problem (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : 1 / (x + 3) + 1 / (y + 4) = 1 / 2) :
  ∀ a b : ℝ, a > 0 ∧ b > 0 ∧ 1 / (a + 3) + 1 / (b + 4) = 1 / 2 →
  2 * x + y ≤ 2 * a + b ∧ 2 * x + y = 1 + 4 * Real.sqrt 2 :=
sorry

end min_value_problem_l2337_233727


namespace amy_final_money_l2337_233747

-- Define the initial conditions
def initial_money : ℕ := 2
def num_neighbors : ℕ := 5
def chore_pay : ℕ := 13
def birthday_money : ℕ := 3
def toy_cost : ℕ := 12

-- Define the calculation steps
def money_after_chores : ℕ := initial_money + num_neighbors * chore_pay
def money_after_birthday : ℕ := money_after_chores + birthday_money
def money_after_toy : ℕ := money_after_birthday - toy_cost
def grandparents_gift : ℕ := 2 * money_after_toy

-- Theorem to prove
theorem amy_final_money :
  money_after_toy + grandparents_gift = 174 := by
  sorry


end amy_final_money_l2337_233747


namespace sum_of_selection_l2337_233752

/-- Represents a selection of numbers from an 8x8 grid -/
def Selection := Fin 8 → Fin 8

/-- The sum of numbers in a selection -/
def sum_selection (s : Selection) : ℕ :=
  Finset.sum Finset.univ (λ i => s i + 1 + 8 * i)

/-- Theorem: The sum of any valid selection is 260 -/
theorem sum_of_selection (s : Selection) (h : Function.Injective s) : sum_selection s = 260 := by
  sorry

#eval sum_selection (λ i => i)  -- Should output 260

end sum_of_selection_l2337_233752


namespace product_with_floor_l2337_233732

theorem product_with_floor (x : ℝ) : 
  x > 0 → x * ⌊x⌋ = 48 → x = 8 := by sorry

end product_with_floor_l2337_233732


namespace rare_integer_existence_and_uniqueness_l2337_233709

/-- A function f: ℤ → ℤ satisfying the given functional equation -/
def FunctionalEquation (f : ℤ → ℤ) : Prop :=
  ∀ x y : ℤ, f (f (x + y) + y) = f (f x + y)

/-- An integer v is f-rare if the set {x ∈ ℤ : f(x) = v} is finite and nonempty -/
def IsRare (f : ℤ → ℤ) (v : ℤ) : Prop :=
  let X_v := {x : ℤ | f x = v}
  Set.Finite X_v ∧ Set.Nonempty X_v

theorem rare_integer_existence_and_uniqueness :
  (∃ f : ℤ → ℤ, FunctionalEquation f ∧ ∃ v : ℤ, IsRare f v) ∧
  (∀ f : ℤ → ℤ, FunctionalEquation f → ∀ v w : ℤ, IsRare f v → IsRare f w → v = w) :=
by sorry

end rare_integer_existence_and_uniqueness_l2337_233709


namespace problem_solution_l2337_233748

theorem problem_solution (a b : ℤ) (h1 : a > b) (h2 : b > 0) (h3 : a + b + a * b = 152) : a = 50 := by
  sorry

end problem_solution_l2337_233748


namespace flower_prices_l2337_233776

theorem flower_prices (x y z : ℚ) 
  (eq1 : 3 * x + 7 * y + z = 14)
  (eq2 : 4 * x + 10 * y + z = 16) :
  3 * (x + y + z) = 30 := by
sorry

end flower_prices_l2337_233776


namespace one_third_percent_of_200_plus_50_l2337_233724

/-- Calculates the result of taking a percentage of a number and adding a constant to it. -/
def percentageOfPlusConstant (percentage : ℚ) (number : ℚ) (constant : ℚ) : ℚ :=
  percentage / 100 * number + constant

/-- The main theorem stating that 1/3% of 200 plus 50 is approximately 50.6667 -/
theorem one_third_percent_of_200_plus_50 :
  ∃ (result : ℚ), abs (percentageOfPlusConstant (1/3) 200 50 - result) < 0.00005 ∧ result = 50.6667 := by
  sorry

#eval percentageOfPlusConstant (1/3) 200 50

end one_third_percent_of_200_plus_50_l2337_233724


namespace common_solutions_iff_y_values_l2337_233783

theorem common_solutions_iff_y_values (x y : ℝ) : 
  (∃ x : ℝ, x^2 + y^2 - 16 = 0 ∧ x^2 - 3*y - 12 = 0) ↔ (y = -4 ∨ y = 1) :=
by sorry

end common_solutions_iff_y_values_l2337_233783


namespace volume_ratio_theorem_l2337_233728

/-- A right rectangular prism with edge lengths 2, 3, and 5 -/
structure RectPrism where
  length : ℝ := 2
  width : ℝ := 3
  height : ℝ := 5

/-- The set of points within distance r of the prism -/
def S (B : RectPrism) (r : ℝ) : Set (ℝ × ℝ × ℝ) := sorry

/-- The volume of S(r) -/
noncomputable def volume_S (B : RectPrism) (r : ℝ) : ℝ := sorry

/-- Coefficients of the volume function -/
structure VolumeCoeffs where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  all_positive : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d

theorem volume_ratio_theorem (B : RectPrism) (coeffs : VolumeCoeffs) 
  (h : ∀ r : ℝ, volume_S B r = coeffs.a * r^3 + coeffs.b * r^2 + coeffs.c * r + coeffs.d) :
  coeffs.b * coeffs.c / (coeffs.a * coeffs.d) = 15.5 := by sorry

end volume_ratio_theorem_l2337_233728


namespace gcd_lcm_sum_50_5005_l2337_233796

theorem gcd_lcm_sum_50_5005 : 
  Nat.gcd 50 5005 + Nat.lcm 50 5005 = 50055 := by
  sorry

end gcd_lcm_sum_50_5005_l2337_233796


namespace quadratic_properties_l2337_233739

/-- A quadratic function f(x) = ax² + bx + c with specific properties -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_properties (a b c d : ℝ) :
  (∀ x, QuadraticFunction a b c x = a * x^2 + b * x + c) →
  QuadraticFunction a b c 0 = 3 →
  QuadraticFunction a b c (-1/2) = 0 →
  QuadraticFunction a b c 3 = 0 →
  (∃ x, QuadraticFunction a b c x = x + d ∧ 
        ∀ y, y ≠ x → QuadraticFunction a b c y > y + d) →
  a = -2 ∧ b = 5 ∧ c = 3 ∧ d = 5 := by
sorry

end quadratic_properties_l2337_233739


namespace tower_divisibility_l2337_233744

/-- Represents the number of towers that can be built with cubes up to edge-length n -/
def S : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | 2 => 2
  | n + 1 => S n * (min 5 (n + 1))

/-- The problem statement -/
theorem tower_divisibility : S 9 % 1000 = 0 := by
  sorry

end tower_divisibility_l2337_233744


namespace least_froods_to_drop_l2337_233781

def score_dropping (n : ℕ) : ℕ := n * (n + 1) / 2

def score_eating (n : ℕ) : ℕ := 15 * n

theorem least_froods_to_drop : 
  ∀ k < 30, score_dropping k ≤ score_eating k ∧ 
  score_dropping 30 > score_eating 30 := by
  sorry

end least_froods_to_drop_l2337_233781


namespace perpendicular_polygon_perimeter_l2337_233791

/-- A polygon with adjacent sides perpendicular to each other -/
structure PerpendicularPolygon where
  a : ℝ  -- Sum of all vertical sides
  b : ℝ  -- Sum of all horizontal sides

/-- The perimeter of a perpendicular polygon -/
def perimeter (p : PerpendicularPolygon) : ℝ := 2 * (p.a + p.b)

/-- Theorem: The perimeter of a perpendicular polygon is 2(a+b) -/
theorem perpendicular_polygon_perimeter (p : PerpendicularPolygon) :
  perimeter p = 2 * (p.a + p.b) := by sorry

end perpendicular_polygon_perimeter_l2337_233791


namespace stairs_climbed_together_l2337_233785

theorem stairs_climbed_together (jonny_stairs : ℕ) (julia_stairs : ℕ) : 
  jonny_stairs = 1269 →
  julia_stairs = jonny_stairs / 3 - 7 →
  jonny_stairs + julia_stairs = 1685 := by
sorry

end stairs_climbed_together_l2337_233785


namespace cost_of_500_pencils_l2337_233740

/-- The cost of 500 pencils in dollars, given that 1 pencil costs 3 cents -/
theorem cost_of_500_pencils :
  let cost_of_one_pencil_cents : ℕ := 3
  let number_of_pencils : ℕ := 500
  let cents_per_dollar : ℕ := 100
  (cost_of_one_pencil_cents * number_of_pencils) / cents_per_dollar = 15 := by
  sorry

end cost_of_500_pencils_l2337_233740


namespace geometric_sequence_a6_l2337_233750

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_a6 (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n, a n > 0) →
  a 2 = 1 →
  a 7 = a 5 + 2 * a 3 →
  a 6 = 4 := by
sorry

end geometric_sequence_a6_l2337_233750


namespace arrange_digits_eq_96_l2337_233765

/-- The number of ways to arrange the digits of 60,402 to form a 5-digit number, 
    ensuring numbers do not begin with 0 -/
def arrange_digits : ℕ :=
  let digits : List ℕ := [6, 0, 4, 0, 2]
  let non_zero_digits : List ℕ := digits.filter (· ≠ 0)
  let zero_count : ℕ := digits.count 0
  let digit_count : ℕ := digits.length
  (digit_count - 1) * (non_zero_digits.length).factorial

theorem arrange_digits_eq_96 : arrange_digits = 96 := by
  sorry

end arrange_digits_eq_96_l2337_233765


namespace not_divisible_by_seven_divisible_by_others_l2337_233758

theorem not_divisible_by_seven (n : ℤ) : ¬(7 ∣ (n^2225 - n^2005)) :=
sorry

theorem divisible_by_others (n : ℤ) : 
  (3 ∣ (n^2225 - n^2005)) ∧ 
  (5 ∣ (n^2225 - n^2005)) ∧ 
  (11 ∣ (n^2225 - n^2005)) ∧ 
  (23 ∣ (n^2225 - n^2005)) :=
sorry

end not_divisible_by_seven_divisible_by_others_l2337_233758


namespace gwen_games_remaining_l2337_233736

/-- The number of games remaining after giving some away -/
def remaining_games (initial : ℕ) (given_away : ℕ) : ℕ :=
  initial - given_away

/-- Theorem: Given 98 initial games and 7 games given away, 91 games remain -/
theorem gwen_games_remaining :
  remaining_games 98 7 = 91 := by
  sorry

end gwen_games_remaining_l2337_233736


namespace second_player_wins_l2337_233794

/-- Represents the state of the game -/
structure GameState where
  pile1 : Nat
  pile2 : Nat

/-- Represents a move in the game -/
structure Move where
  pile : Nat -- 1 or 2
  balls : Nat

/-- Defines a valid move in the game -/
def validMove (state : GameState) (move : Move) : Prop :=
  (move.pile = 1 ∧ move.balls ≤ state.pile1) ∨
  (move.pile = 2 ∧ move.balls ≤ state.pile2)

/-- Applies a move to the game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  if move.pile = 1 then
    { pile1 := state.pile1 - move.balls, pile2 := state.pile2 }
  else
    { pile1 := state.pile1, pile2 := state.pile2 - move.balls }

/-- Defines the winning condition -/
def isWinningState (state : GameState) : Prop :=
  state.pile1 = 0 ∧ state.pile2 = 0

/-- Defines a winning strategy for the second player -/
def secondPlayerWinningStrategy (initialState : GameState) : Prop :=
  ∀ (move : Move), 
    validMove initialState move → 
    ∃ (response : Move), 
      validMove (applyMove initialState move) response ∧
      (applyMove (applyMove initialState move) response).pile1 = 
      (applyMove (applyMove initialState move) response).pile2

/-- Theorem: The second player has a winning strategy in the two-pile game -/
theorem second_player_wins (initialState : GameState) 
  (h1 : initialState.pile1 = 30) (h2 : initialState.pile2 = 30) : 
  secondPlayerWinningStrategy initialState :=
sorry


end second_player_wins_l2337_233794


namespace english_only_students_l2337_233701

theorem english_only_students (total : ℕ) (all_three : ℕ) (english_only : ℕ) (french_only : ℕ) :
  total = 35 →
  all_three = 2 →
  english_only = 3 * french_only →
  english_only + french_only + all_three = total →
  english_only - all_three = 23 := by
  sorry

end english_only_students_l2337_233701


namespace acid_mixture_problem_l2337_233768

/-- Represents the contents of a jar --/
structure Jar where
  volume : ℚ
  acid_concentration : ℚ

/-- Represents the problem setup --/
structure ProblemSetup where
  jar_a : Jar
  jar_b : Jar
  jar_c : Jar
  m : ℕ
  n : ℕ

/-- The initial setup of the problem --/
def initial_setup (k : ℚ) : ProblemSetup where
  jar_a := { volume := 4, acid_concentration := 45/100 }
  jar_b := { volume := 5, acid_concentration := 48/100 }
  jar_c := { volume := 1, acid_concentration := k/100 }
  m := 2
  n := 3

/-- The final state after mixing --/
def final_state (setup : ProblemSetup) : Prop :=
  let new_jar_a_volume := setup.jar_a.volume + setup.m / setup.n
  let new_jar_b_volume := setup.jar_b.volume + (1 - setup.m / setup.n)
  let new_jar_a_acid := setup.jar_a.volume * setup.jar_a.acid_concentration + 
                        setup.jar_c.volume * setup.jar_c.acid_concentration * (setup.m / setup.n)
  let new_jar_b_acid := setup.jar_b.volume * setup.jar_b.acid_concentration + 
                        setup.jar_c.volume * setup.jar_c.acid_concentration * (1 - setup.m / setup.n)
  (new_jar_a_acid / new_jar_a_volume = 1/2) ∧ (new_jar_b_acid / new_jar_b_volume = 1/2)

/-- The main theorem --/
theorem acid_mixture_problem (k : ℚ) :
  final_state (initial_setup k) → k + 2 + 3 = 85 := by
  sorry


end acid_mixture_problem_l2337_233768

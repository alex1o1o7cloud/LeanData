import Mathlib

namespace sum_of_ages_l978_97885

theorem sum_of_ages (henry_age jill_age : ℕ) : 
  henry_age = 23 →
  jill_age = 17 →
  henry_age - 11 = 2 * (jill_age - 11) →
  henry_age + jill_age = 40 :=
by
  sorry

end sum_of_ages_l978_97885


namespace cosine_product_identity_l978_97888

theorem cosine_product_identity (n : ℕ) (hn : n = 7 ∨ n = 9) : 
  Real.cos (2 * Real.pi / n) * Real.cos (4 * Real.pi / n) * Real.cos (8 * Real.pi / n) = 
  (-1 : ℝ) ^ ((n - 1) / 2) * (1 / 8) :=
by sorry

end cosine_product_identity_l978_97888


namespace total_cupcakes_is_52_l978_97831

/-- Represents the number of cupcakes ordered by the mum -/
def total_cupcakes : ℕ := 52

/-- Represents the number of vegan cupcakes -/
def vegan_cupcakes : ℕ := 24

/-- Represents the number of non-vegan cupcakes containing gluten -/
def non_vegan_gluten_cupcakes : ℕ := 28

/-- States that half of all cupcakes are gluten-free -/
axiom half_gluten_free : total_cupcakes / 2 = total_cupcakes - (vegan_cupcakes / 2 + non_vegan_gluten_cupcakes)

/-- States that half of vegan cupcakes are gluten-free -/
axiom half_vegan_gluten_free : vegan_cupcakes / 2 = total_cupcakes / 2 - non_vegan_gluten_cupcakes

/-- Theorem: The total number of cupcakes is 52 -/
theorem total_cupcakes_is_52 : total_cupcakes = 52 := by
  sorry

end total_cupcakes_is_52_l978_97831


namespace square_root_representation_l978_97869

theorem square_root_representation (x : ℝ) (h : x = 0.25) :
  ∃ y : ℝ, y > 0 ∧ y^2 = x ∧ (∀ z : ℝ, z^2 = x → z = y ∨ z = -y) :=
by sorry

end square_root_representation_l978_97869


namespace total_games_in_season_l978_97836

theorem total_games_in_season (total_teams : ℕ) (num_divisions : ℕ) (teams_per_division : ℕ)
  (intra_division_games : ℕ) (inter_division_games : ℕ)
  (h1 : total_teams = 24)
  (h2 : num_divisions = 3)
  (h3 : teams_per_division = 8)
  (h4 : total_teams = num_divisions * teams_per_division)
  (h5 : intra_division_games = 3)
  (h6 : inter_division_games = 2) :
  (total_teams * (((teams_per_division - 1) * intra_division_games) +
   ((total_teams - teams_per_division) * inter_division_games))) / 2 = 636 := by
  sorry

end total_games_in_season_l978_97836


namespace expected_value_decahedral_die_l978_97827

/-- A fair decahedral die with faces numbered 1 to 10 -/
def DecahedralDie : Finset ℕ := Finset.range 10

/-- The probability of each outcome for a fair die -/
def prob (n : ℕ) : ℚ := 1 / 10

/-- The expected value of rolling the decahedral die -/
def expected_value : ℚ := (DecahedralDie.sum (fun i => prob i * (i + 1)))

/-- Theorem: The expected value of rolling a fair decahedral die with faces numbered 1 to 10 is 5.5 -/
theorem expected_value_decahedral_die : expected_value = 11 / 2 := by
  sorry

end expected_value_decahedral_die_l978_97827


namespace largest_common_divisor_342_285_l978_97856

theorem largest_common_divisor_342_285 : ∃ (n : ℕ), n > 0 ∧ n ∣ 342 ∧ n ∣ 285 ∧ ∀ (m : ℕ), m > n → (m ∣ 342 ∧ m ∣ 285 → False) :=
  sorry

end largest_common_divisor_342_285_l978_97856


namespace sequence_inequality_l978_97802

theorem sequence_inequality (A B : ℝ) (a : ℕ → ℝ) 
  (hA : A > 1) (hB : B > 1) (ha : ∀ n, 1 ≤ a n ∧ a n ≤ A * B) :
  ∃ b : ℕ → ℝ, (∀ n, 1 ≤ b n ∧ b n ≤ A) ∧
    (∀ m n : ℕ, a m / a n ≤ B * (b m / b n)) :=
by sorry

end sequence_inequality_l978_97802


namespace equation_holds_iff_k_equals_neg_15_l978_97899

theorem equation_holds_iff_k_equals_neg_15 :
  (∀ x : ℝ, -x^2 - (k + 9)*x - 8 = -(x - 2)*(x - 4)) ↔ k = -15 :=
by sorry

end equation_holds_iff_k_equals_neg_15_l978_97899


namespace lines_properties_l978_97832

-- Define the lines l₁ and l₂
def l₁ (a x y : ℝ) : Prop := a * x - y + 1 = 0
def l₂ (a x y : ℝ) : Prop := x + a * y + 1 = 0

-- Theorem statement
theorem lines_properties (a : ℝ) :
  -- 1. Perpendicularity condition
  (a ≠ 0 → (∀ x₁ y₁ x₂ y₂ : ℝ, l₁ a x₁ y₁ ∧ l₂ a x₂ y₂ → (x₂ - x₁) * (y₂ - y₁) = -1)) ∧
  -- 2. Fixed points condition
  (l₁ a 0 1 ∧ l₂ a (-1) 0) ∧
  -- 3. Maximum distance condition
  (∀ x y : ℝ, l₁ a x y ∧ l₂ a x y → x^2 + y^2 ≤ 2) :=
by sorry

end lines_properties_l978_97832


namespace dave_initial_money_l978_97805

-- Define the given amounts
def derek_initial : ℕ := 40
def derek_spend1 : ℕ := 14
def derek_spend2 : ℕ := 11
def derek_spend3 : ℕ := 5
def dave_spend : ℕ := 7
def dave_extra : ℕ := 33

-- Define Derek's total spending
def derek_total_spend : ℕ := derek_spend1 + derek_spend2 + derek_spend3

-- Define Derek's remaining money
def derek_remaining : ℕ := derek_initial - derek_total_spend

-- Define Dave's remaining money
def dave_remaining : ℕ := derek_remaining + dave_extra

-- Theorem to prove
theorem dave_initial_money : dave_remaining + dave_spend = 50 := by
  sorry

end dave_initial_money_l978_97805


namespace cardinals_second_inning_l978_97898

def cubs_third_inning : ℕ := 2
def cubs_fifth_inning : ℕ := 1
def cubs_eighth_inning : ℕ := 2
def cardinals_fifth_inning : ℕ := 1
def cubs_advantage : ℕ := 3

def cubs_total : ℕ := cubs_third_inning + cubs_fifth_inning + cubs_eighth_inning
def cardinals_total : ℕ := cubs_total - cubs_advantage

theorem cardinals_second_inning : cardinals_total - cardinals_fifth_inning = 1 := by
  sorry

end cardinals_second_inning_l978_97898


namespace favorite_fruit_strawberries_l978_97815

theorem favorite_fruit_strawberries (total : ℕ) (oranges pears apples : ℕ)
  (h1 : total = 450)
  (h2 : oranges = 70)
  (h3 : pears = 120)
  (h4 : apples = 147) :
  total - (oranges + pears + apples) = 113 :=
by sorry

end favorite_fruit_strawberries_l978_97815


namespace largest_difference_l978_97897

def A : ℕ := 3 * 2010^2011
def B : ℕ := 2010^2011
def C : ℕ := 2009 * 2010^2010
def D : ℕ := 3 * 2010^2010
def E : ℕ := 2010^2010
def F : ℕ := 2010^2009

theorem largest_difference : 
  (A - B > B - C) ∧ 
  (A - B > C - D) ∧ 
  (A - B > D - E) ∧ 
  (A - B > E - F) := by sorry

end largest_difference_l978_97897


namespace sine_cosine_zero_points_l978_97896

theorem sine_cosine_zero_points (ω : ℝ) (h_ω_pos : ω > 0) :
  let f : ℝ → ℝ := λ x => Real.sin (ω * x) + Real.sqrt 3 * Real.cos (ω * x)
  (∃! (s : Finset ℝ), s.card = 5 ∧ ∀ x ∈ s, 0 < x ∧ x < 4 * Real.pi ∧ f x = 0) →
  7 / 6 < ω ∧ ω ≤ 17 / 12 := by
sorry

end sine_cosine_zero_points_l978_97896


namespace cookie_bags_count_l978_97841

/-- Given a total number of cookies and the fact that each bag contains an equal number of cookies,
    prove that the number of bags is 14. -/
theorem cookie_bags_count (total_cookies : ℕ) (cookies_per_bag : ℕ) (total_candies : ℕ) :
  total_cookies = 28 →
  cookies_per_bag > 0 →
  total_cookies = 14 * cookies_per_bag →
  (∃ (num_bags : ℕ), num_bags = 14 ∧ num_bags * cookies_per_bag = total_cookies) :=
by sorry

end cookie_bags_count_l978_97841


namespace no_real_roots_of_quadratic_l978_97850

theorem no_real_roots_of_quadratic (x : ℝ) : ¬∃x, x^2 - 4*x + 8 = 0 := by
  sorry

end no_real_roots_of_quadratic_l978_97850


namespace y_plus_2z_positive_l978_97804

theorem y_plus_2z_positive (x y z : ℝ) 
  (hx : 0 < x ∧ x < 2) 
  (hy : -2 < y ∧ y < 0) 
  (hz : 0 < z ∧ z < 3) : 
  y + 2*z > 0 := by
  sorry

end y_plus_2z_positive_l978_97804


namespace seventh_term_of_geometric_sequence_l978_97861

def geometric_sequence (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * r ^ (n - 1)

theorem seventh_term_of_geometric_sequence 
  (a r : ℝ) 
  (h_positive : ∀ n, geometric_sequence a r n > 0)
  (h_fifth : geometric_sequence a r 5 = 16)
  (h_ninth : geometric_sequence a r 9 = 2) :
  geometric_sequence a r 7 = 8 * Real.sqrt 2 :=
sorry

end seventh_term_of_geometric_sequence_l978_97861


namespace least_divisible_by_240_cubed_l978_97890

theorem least_divisible_by_240_cubed (a : ℕ) : 
  (∀ n : ℕ, n < 60 → ¬(240 ∣ n^3)) ∧ (240 ∣ 60^3) := by
sorry

end least_divisible_by_240_cubed_l978_97890


namespace savings_difference_l978_97880

def initial_order : ℝ := 15000

def option1_discounts : List ℝ := [0.10, 0.25, 0.15]
def option2_discounts : List ℝ := [0.30, 0.10, 0.05]

def apply_discount (price : ℝ) (discount : ℝ) : ℝ :=
  price * (1 - discount)

def apply_discounts (price : ℝ) (discounts : List ℝ) : ℝ :=
  discounts.foldl apply_discount price

theorem savings_difference :
  apply_discounts initial_order option2_discounts - 
  apply_discounts initial_order option1_discounts = 371.25 := by
  sorry

end savings_difference_l978_97880


namespace solution_set_eq_l978_97879

-- Define a decreasing function f
def f : ℝ → ℝ := sorry

-- Define the properties of f
axiom f_decreasing : ∀ x y, x < y → f x > f y
axiom f_0 : f 0 = -2
axiom f_neg_3 : f (-3) = 2

-- Define the solution set
def solution_set : Set ℝ := {x | |f (x - 2)| > 2}

-- State the theorem
theorem solution_set_eq : solution_set = Set.Iic (-1) ∪ Set.Ioi 2 := by sorry

end solution_set_eq_l978_97879


namespace trees_on_promenade_l978_97855

/-- The number of trees planted along a circular promenade -/
def number_of_trees (promenade_length : ℕ) (tree_interval : ℕ) : ℕ :=
  promenade_length / tree_interval

/-- Theorem: The number of trees planted along a circular promenade of length 1200 meters, 
    with trees planted at intervals of 30 meters, is equal to 40. -/
theorem trees_on_promenade : number_of_trees 1200 30 = 40 := by
  sorry

end trees_on_promenade_l978_97855


namespace simultaneous_congruences_l978_97891

theorem simultaneous_congruences (x : ℤ) :
  x % 2 = 1 ∧ x % 3 = 2 ∧ x % 5 = 3 ∧ x % 7 = 4 → x % 210 = 53 := by
  sorry

end simultaneous_congruences_l978_97891


namespace line_ellipse_intersection_range_l978_97823

/-- The range of m for which the line 2kx-y+1=0 always intersects the ellipse x²/9 + y²/m = 1 -/
theorem line_ellipse_intersection_range :
  ∀ (k : ℝ), (∀ (x y : ℝ), 2 * k * x - y + 1 = 0 → x^2 / 9 + y^2 / m = 1) →
  m ∈ Set.Icc 1 9 ∪ Set.Ioi 9 :=
sorry

end line_ellipse_intersection_range_l978_97823


namespace superhero_movie_count_l978_97835

theorem superhero_movie_count (total_movies : ℕ) (dalton_movies : ℕ) (alex_movies : ℕ) (shared_movies : ℕ) :
  total_movies = 30 →
  dalton_movies = 7 →
  alex_movies = 15 →
  shared_movies = 2 →
  ∃ (hunter_movies : ℕ), hunter_movies = total_movies - dalton_movies - alex_movies + shared_movies :=
by
  sorry

end superhero_movie_count_l978_97835


namespace limit_sin_difference_l978_97819

theorem limit_sin_difference (ε : ℝ) (hε : ε > 0) :
  ∃ δ > 0, ∀ x : ℝ, 0 < |x| ∧ |x| < δ →
    |(1 / (4 * Real.sin x ^ 2) - 1 / Real.sin (2 * x) ^ 2) - (-1/4)| < ε :=
by sorry

end limit_sin_difference_l978_97819


namespace water_difference_l978_97881

theorem water_difference (s h : ℝ) 
  (h1 : s > h) 
  (h2 : (s - 0.43) - (h + 0.43) = 0.88) : 
  s - h = 1.74 := by
  sorry

end water_difference_l978_97881


namespace tamara_kim_height_ratio_l978_97801

/-- Given Tamara's height and the combined height of Tamara and Kim, 
    prove that Tamara is 17/6 times taller than Kim. -/
theorem tamara_kim_height_ratio :
  ∀ (tamara_height kim_height : ℝ),
    tamara_height = 68 →
    tamara_height + kim_height = 92 →
    tamara_height / kim_height = 17 / 6 :=
by
  sorry

end tamara_kim_height_ratio_l978_97801


namespace three_hundredth_term_omit_squares_l978_97847

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def omit_squares_sequence (n : ℕ) : ℕ :=
  n + (Nat.sqrt n)

theorem three_hundredth_term_omit_squares : omit_squares_sequence 300 = 317 := by
  sorry

end three_hundredth_term_omit_squares_l978_97847


namespace second_longest_piece_length_l978_97893

/-- The length of the second longest piece of rope when a 142.75-inch rope is cut into five pieces
    in the ratio (√2):6:(4/3):(3^2):(1/2) is approximately 46.938 inches. -/
theorem second_longest_piece_length (total_length : ℝ) (piece1 piece2 piece3 piece4 piece5 : ℝ)
  (h1 : total_length = 142.75)
  (h2 : piece1 / (Real.sqrt 2) = piece2 / 6)
  (h3 : piece2 / 6 = piece3 / (4/3))
  (h4 : piece3 / (4/3) = piece4 / 9)
  (h5 : piece4 / 9 = piece5 / (1/2))
  (h6 : piece1 + piece2 + piece3 + piece4 + piece5 = total_length) :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.001 ∧ |piece2 - 46.938| < ε ∧
  (piece2 > piece1 ∨ piece2 > piece3 ∨ piece2 > piece5) ∧
  (piece4 > piece2 ∨ piece4 = piece2) :=
by sorry

end second_longest_piece_length_l978_97893


namespace geometry_book_pages_multiple_l978_97817

/-- Given that:
    - The old edition of a Geometry book has 340 pages
    - The new edition has 450 pages
    - The new edition has 230 pages less than m times the old edition's pages
    Prove that m = 2 -/
theorem geometry_book_pages_multiple (old_pages new_pages less_pages : ℕ) 
    (h1 : old_pages = 340)
    (h2 : new_pages = 450)
    (h3 : less_pages = 230) :
    ∃ m : ℚ, old_pages * m - less_pages = new_pages ∧ m = 2 := by
  sorry

end geometry_book_pages_multiple_l978_97817


namespace fourth_largest_divisor_l978_97870

def n : ℕ := 1234560000

-- Define a function to get the list of divisors
def divisors (m : ℕ) : List ℕ := sorry

-- Define a function to get the nth largest element from a list
def nthLargest (l : List ℕ) (k : ℕ) : ℕ := sorry

theorem fourth_largest_divisor :
  nthLargest (divisors n) 4 = 154320000 := by sorry

end fourth_largest_divisor_l978_97870


namespace sum_m_n_equals_three_l978_97887

theorem sum_m_n_equals_three (m n : ℕ) (h1 : 0 < m) (h2 : 0 < n) (h3 : m + 5 < n)
  (h4 : (m + (m + 3) + (m + 5) + n + (n + 2) + (2 * n - 1)) / 6 = n + 1)
  (h5 : ((m + 5) + n) / 2 = n + 1) : m + n = 3 := by
  sorry

end sum_m_n_equals_three_l978_97887


namespace unique_g_3_l978_97854

-- Define the function g
def g : ℝ → ℝ := sorry

-- State the conditions
axiom g_1 : g 1 = -1
axiom g_property : ∀ x y : ℝ, g (x^2 - y^2) = (x - y) * (g x - g y)

-- Define m as the number of possible values of g(3)
def m : ℕ := sorry

-- Define t as the sum of all possible values of g(3)
def t : ℝ := sorry

-- Theorem statement
theorem unique_g_3 : m = 1 ∧ t = -3 := by sorry

end unique_g_3_l978_97854


namespace antons_number_l978_97806

def matches_one_digit (a b : ℕ) : Prop :=
  (a / 100 = b / 100 ∧ a % 100 ≠ b % 100) ∨
  (a % 100 / 10 = b % 100 / 10 ∧ a / 100 ≠ b / 100 ∧ a % 10 ≠ b % 10) ∨
  (a % 10 = b % 10 ∧ a / 10 ≠ b / 10)

theorem antons_number (x : ℕ) :
  100 ≤ x ∧ x < 1000 ∧
  matches_one_digit x 109 ∧
  matches_one_digit x 704 ∧
  matches_one_digit x 124 →
  x = 729 := by
  sorry

end antons_number_l978_97806


namespace max_product_sum_2000_l978_97825

theorem max_product_sum_2000 : 
  ∀ x y : ℤ, x + y = 2000 → x * y ≤ 1000000 := by
  sorry

end max_product_sum_2000_l978_97825


namespace f_10_equals_756_l978_97877

def f (x : ℝ) : ℝ := x^3 - 2*x^2 - 5*x + 6

theorem f_10_equals_756 : f 10 = 756 := by
  sorry

end f_10_equals_756_l978_97877


namespace interest_rate_multiple_l978_97800

theorem interest_rate_multiple (P r m : ℝ) 
  (h1 : P * r^2 = 40)
  (h2 : P * (m * r)^2 = 360)
  : m = 3 := by
  sorry

end interest_rate_multiple_l978_97800


namespace six_balls_two_boxes_at_least_two_l978_97846

/-- The number of ways to distribute n distinguishable balls into 2 distinguishable boxes -/
def totalArrangements (n : ℕ) : ℕ := 2^n

/-- The number of ways to choose k balls from n distinguishable balls -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of ways to distribute n distinguishable balls into 2 distinguishable boxes
    where one box must contain at least m balls -/
def validArrangements (n m : ℕ) : ℕ :=
  totalArrangements n - (choose n 0 + choose n 1)

theorem six_balls_two_boxes_at_least_two :
  validArrangements 6 2 = 57 := by sorry

end six_balls_two_boxes_at_least_two_l978_97846


namespace inequality_and_equality_condition_l978_97807

theorem inequality_and_equality_condition (a b : ℝ) 
  (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : a + b < 2) :
  (1 / (1 + a^2) + 1 / (1 + b^2) ≤ 2 / (1 + a * b)) ∧
  (1 / (1 + a^2) + 1 / (1 + b^2) = 2 / (1 + a * b) ↔ 0 < a ∧ a = b ∧ b < 1) :=
by sorry

end inequality_and_equality_condition_l978_97807


namespace lateral_edge_length_l978_97866

/-- A regular pyramid with a square base -/
structure RegularPyramid where
  -- The side length of the square base
  base_side : ℝ
  -- The volume of the pyramid
  volume : ℝ
  -- The length of a lateral edge
  lateral_edge : ℝ

/-- Theorem: In a regular pyramid with square base, if the volume is 4/3 and the base side length is 2, 
    then the lateral edge length is √3 -/
theorem lateral_edge_length (p : RegularPyramid) 
  (h1 : p.volume = 4/3) 
  (h2 : p.base_side = 2) : 
  p.lateral_edge = Real.sqrt 3 := by
  sorry


end lateral_edge_length_l978_97866


namespace polynomial_uniqueness_l978_97838

def Q (x : ℝ) (Q0 Q1 Q2 : ℝ) : ℝ := Q0 + Q1 * x + Q2 * x^2

theorem polynomial_uniqueness (Q0 Q1 Q2 : ℝ) :
  Q (-1) Q0 Q1 Q2 = -3 →
  Q 3 Q0 Q1 Q2 = 5 →
  ∀ x, Q x Q0 Q1 Q2 = 3 * x^2 + 7 * x - 5 :=
by sorry

end polynomial_uniqueness_l978_97838


namespace instantaneous_velocity_at_5_l978_97822

/-- A particle moving in a straight line with distance-time relationship s(t) = 4t^2 - 3 -/
def s (t : ℝ) : ℝ := 4 * t^2 - 3

/-- The instantaneous velocity function v(t) -/
def v (t : ℝ) : ℝ := 8 * t

theorem instantaneous_velocity_at_5 : v 5 = 40 := by
  sorry

end instantaneous_velocity_at_5_l978_97822


namespace tyler_cake_servings_l978_97842

/-- The number of people the original recipe serves -/
def original_recipe_servings : ℕ := 4

/-- The number of eggs required for the original recipe -/
def original_recipe_eggs : ℕ := 2

/-- The total number of eggs Tyler needs for his cake -/
def tylers_eggs : ℕ := 4

/-- The number of people Tyler wants to make the cake for -/
def tylers_servings : ℕ := 8

theorem tyler_cake_servings :
  tylers_servings = original_recipe_servings * (tylers_eggs / original_recipe_eggs) :=
by sorry

end tyler_cake_servings_l978_97842


namespace alexander_pencil_difference_alexander_pencil_difference_proof_l978_97820

/-- Proves that Alexander has 60 more pencils than Asaf given the problem conditions -/
theorem alexander_pencil_difference : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun asaf_age alexander_age asaf_pencils alexander_pencils =>
    asaf_age = 50 ∧
    asaf_age + alexander_age = 140 ∧
    alexander_age - asaf_age = asaf_pencils / 2 ∧
    asaf_pencils + alexander_pencils = 220 →
    alexander_pencils - asaf_pencils = 60

/-- Proof of the theorem -/
theorem alexander_pencil_difference_proof :
  ∃ (asaf_age alexander_age asaf_pencils alexander_pencils : ℕ),
    alexander_pencil_difference asaf_age alexander_age asaf_pencils alexander_pencils :=
by
  sorry

end alexander_pencil_difference_alexander_pencil_difference_proof_l978_97820


namespace function_value_at_two_l978_97878

/-- Given a function f: ℝ → ℝ such that f(x) = ax^5 + bx^3 + cx + 8 for some real constants a, b, and c, 
    and f(-2) = 10, prove that f(2) = 6. -/
theorem function_value_at_two (a b c : ℝ) (f : ℝ → ℝ) 
    (h1 : ∀ x, f x = a * x^5 + b * x^3 + c * x + 8)
    (h2 : f (-2) = 10) : 
  f 2 = 6 := by
  sorry

end function_value_at_two_l978_97878


namespace missing_mark_calculation_l978_97852

def calculate_missing_mark (english math chemistry biology average : ℕ) : ℕ :=
  5 * average - (english + math + chemistry + biology)

theorem missing_mark_calculation (english math chemistry biology average : ℕ) :
  calculate_missing_mark english math chemistry biology average =
  5 * average - (english + math + chemistry + biology) :=
by sorry

end missing_mark_calculation_l978_97852


namespace type_B_first_is_better_l978_97857

/-- Represents the score distribution for a two-question quiz -/
structure ScoreDistribution where
  p0 : ℝ  -- Probability of scoring 0
  p1 : ℝ  -- Probability of scoring the first question's points
  p2 : ℝ  -- Probability of scoring both questions' points
  sum_to_one : p0 + p1 + p2 = 1

/-- Calculates the expected score given a score distribution and point values -/
def expectedScore (d : ScoreDistribution) (points1 points2 : ℝ) : ℝ :=
  d.p1 * points1 + d.p2 * (points1 + points2)

/-- Represents the quiz setup -/
structure QuizSetup where
  probA : ℝ  -- Probability of correctly answering type A
  probB : ℝ  -- Probability of correctly answering type B
  pointsA : ℝ  -- Points for correct answer in type A
  pointsB : ℝ  -- Points for correct answer in type B
  probA_bounds : 0 ≤ probA ∧ probA ≤ 1
  probB_bounds : 0 ≤ probB ∧ probB ≤ 1
  positive_points : pointsA > 0 ∧ pointsB > 0

/-- Theorem: Starting with type B questions yields a higher expected score -/
theorem type_B_first_is_better (q : QuizSetup) : 
  let distA : ScoreDistribution := {
    p0 := 1 - q.probA,
    p1 := q.probA * (1 - q.probB),
    p2 := q.probA * q.probB,
    sum_to_one := by sorry
  }
  let distB : ScoreDistribution := {
    p0 := 1 - q.probB,
    p1 := q.probB * (1 - q.probA),
    p2 := q.probB * q.probA,
    sum_to_one := by sorry
  }
  expectedScore distB q.pointsB q.pointsA > expectedScore distA q.pointsA q.pointsB :=
by sorry

end type_B_first_is_better_l978_97857


namespace equal_money_after_11_weeks_l978_97833

/-- Carol's initial amount in dollars -/
def carol_initial : ℕ := 40

/-- Carol's weekly savings in dollars -/
def carol_savings : ℕ := 12

/-- Mike's initial amount in dollars -/
def mike_initial : ℕ := 150

/-- Mike's weekly savings in dollars -/
def mike_savings : ℕ := 2

/-- The number of weeks it takes for Carol and Mike to have the same amount of money -/
def weeks_to_equal_money : ℕ := 11

theorem equal_money_after_11_weeks :
  carol_initial + carol_savings * weeks_to_equal_money =
  mike_initial + mike_savings * weeks_to_equal_money :=
by sorry

end equal_money_after_11_weeks_l978_97833


namespace regular_polygon_interior_angle_l978_97812

theorem regular_polygon_interior_angle (C : ℕ) : 
  C > 2 → (288 : ℝ) = (C - 2 : ℝ) * 180 / C → C = 10 := by
  sorry

end regular_polygon_interior_angle_l978_97812


namespace election_results_l978_97818

theorem election_results (total_votes : ℕ) (invalid_percentage : ℚ) 
  (candidate_A_percentage : ℚ) (candidate_B_percentage : ℚ) :
  total_votes = 1250000 →
  invalid_percentage = 1/5 →
  candidate_A_percentage = 9/20 →
  candidate_B_percentage = 7/20 →
  ∃ (valid_votes : ℕ) (votes_A votes_B votes_C : ℕ),
    valid_votes = total_votes * (1 - invalid_percentage) ∧
    votes_A = valid_votes * candidate_A_percentage ∧
    votes_B = valid_votes * candidate_B_percentage ∧
    votes_C = valid_votes * (1 - candidate_A_percentage - candidate_B_percentage) ∧
    votes_A = 450000 ∧
    votes_B = 350000 ∧
    votes_C = 200000 :=
by sorry

end election_results_l978_97818


namespace original_price_proof_l978_97875

/-- The original price of a part before discount -/
def original_price : ℝ := 62.71

/-- The number of parts Clark bought -/
def num_parts : ℕ := 7

/-- The total amount Clark paid after discount -/
def total_paid : ℝ := 439

/-- Theorem stating that the original price multiplied by the number of parts equals the total amount paid -/
theorem original_price_proof : original_price * num_parts = total_paid := by sorry

end original_price_proof_l978_97875


namespace shaded_to_unshaded_ratio_l978_97816

/-- Represents a rectangular grid with specific properties -/
structure Grid :=
  (width : ℕ)
  (height : ℕ)
  (qr_length : ℕ)
  (st_length : ℕ)
  (rstu_height : ℕ)

/-- Calculates the area of a right triangle given its base and height -/
def triangle_area (base height : ℕ) : ℚ :=
  (base * height : ℚ) / 2

/-- Calculates the area of a rectangle given its width and height -/
def rectangle_area (width height : ℕ) : ℕ :=
  width * height

/-- Calculates the shaded area of the grid -/
def shaded_area (g : Grid) : ℚ :=
  triangle_area g.qr_length g.height +
  triangle_area g.st_length (g.height - g.rstu_height) +
  rectangle_area (g.st_length) g.rstu_height

/-- Calculates the total area of the grid -/
def total_area (g : Grid) : ℕ :=
  rectangle_area g.width g.height

/-- Calculates the unshaded area of the grid -/
def unshaded_area (g : Grid) : ℚ :=
  (total_area g : ℚ) - shaded_area g

/-- Theorem stating the ratio of shaded to unshaded area -/
theorem shaded_to_unshaded_ratio (g : Grid) (h1 : g.width = 9) (h2 : g.height = 4)
    (h3 : g.qr_length = 3) (h4 : g.st_length = 4) (h5 : g.rstu_height = 2) :
    (shaded_area g) / (unshaded_area g) = 5 / 4 := by
  sorry

end shaded_to_unshaded_ratio_l978_97816


namespace polar_coordinates_of_point_l978_97863

theorem polar_coordinates_of_point (x y : ℝ) (ρ θ : ℝ) 
  (h1 : x = -1) 
  (h2 : y = 1) 
  (h3 : ρ > 0) 
  (h4 : 0 < θ ∧ θ < π) 
  (h5 : ρ = Real.sqrt (x^2 + y^2)) 
  (h6 : θ = Real.arctan (y / x) + π) : 
  (ρ, θ) = (Real.sqrt 2, 3 * π / 4) := by
sorry

end polar_coordinates_of_point_l978_97863


namespace select_one_each_select_at_least_two_surgical_l978_97843

/-- The number of nursing experts -/
def num_nursing : ℕ := 3

/-- The number of surgical experts -/
def num_surgical : ℕ := 5

/-- The number of psychological therapy experts -/
def num_psych : ℕ := 2

/-- The total number of experts to be selected -/
def num_selected : ℕ := 4

/-- Function to calculate the number of ways to select k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- Theorem for part 1 -/
theorem select_one_each : 
  choose num_surgical 1 * choose num_psych 1 * choose num_nursing 2 = 30 := by sorry

/-- Theorem for part 2 -/
theorem select_at_least_two_surgical :
  (choose 4 1 * choose 4 2 + choose 4 2 * choose 4 1 + choose 4 3) +
  (choose 4 2 * choose 5 2 + choose 4 3 * choose 5 1 + choose 4 4) = 133 := by sorry

end select_one_each_select_at_least_two_surgical_l978_97843


namespace reflection_sum_l978_97876

/-- Given a point C with coordinates (3, y) that is reflected over the line y = x to point D,
    the sum of all coordinate values of C and D is equal to 2y + 6. -/
theorem reflection_sum (y : ℝ) : 
  let C := (3, y)
  let D := (y, 3)
  (C.1 + C.2 + D.1 + D.2) = 2 * y + 6 := by
  sorry

end reflection_sum_l978_97876


namespace sqrt_2m_minus_n_equals_sqrt_2_l978_97830

theorem sqrt_2m_minus_n_equals_sqrt_2 (m n : ℝ) : 
  (2 * m + n = 8 ∧ 2 * n - m = 1) → Real.sqrt (2 * m - n) = Real.sqrt 2 := by
  sorry

end sqrt_2m_minus_n_equals_sqrt_2_l978_97830


namespace mans_speed_with_current_is_25_l978_97895

/-- Given a man's speed against a current and the current's speed, 
    calculate the man's speed with the current. -/
def mans_speed_with_current (speed_against_current : ℝ) (current_speed : ℝ) : ℝ :=
  speed_against_current + 2 * current_speed

/-- Theorem stating that given the specific conditions, 
    the man's speed with the current is 25 km/hr. -/
theorem mans_speed_with_current_is_25 :
  mans_speed_with_current 20 2.5 = 25 := by
  sorry

#eval mans_speed_with_current 20 2.5

end mans_speed_with_current_is_25_l978_97895


namespace similar_triangles_segment_length_l978_97872

/-- Triangle similarity is a relation between two triangles -/
def TriangleSimilar (t1 t2 : Type) : Prop := sorry

/-- Length of a segment -/
def SegmentLength (s : Type) : ℝ := sorry

theorem similar_triangles_segment_length 
  (PQR XYZ GHI : Type) 
  (h1 : TriangleSimilar PQR XYZ) 
  (h2 : TriangleSimilar XYZ GHI) 
  (h3 : SegmentLength PQ = 5) 
  (h4 : SegmentLength QR = 15) 
  (h5 : SegmentLength HI = 30) : 
  SegmentLength XY = 2.5 := by sorry

end similar_triangles_segment_length_l978_97872


namespace min_sum_squares_l978_97845

theorem min_sum_squares (a b c d e f g h : ℤ) : 
  a ∈ ({-7, -5, -3, -2, 2, 4, 6, 13} : Set ℤ) →
  b ∈ ({-7, -5, -3, -2, 2, 4, 6, 13} : Set ℤ) →
  c ∈ ({-7, -5, -3, -2, 2, 4, 6, 13} : Set ℤ) →
  d ∈ ({-7, -5, -3, -2, 2, 4, 6, 13} : Set ℤ) →
  e ∈ ({-7, -5, -3, -2, 2, 4, 6, 13} : Set ℤ) →
  f ∈ ({-7, -5, -3, -2, 2, 4, 6, 13} : Set ℤ) →
  g ∈ ({-7, -5, -3, -2, 2, 4, 6, 13} : Set ℤ) →
  h ∈ ({-7, -5, -3, -2, 2, 4, 6, 13} : Set ℤ) →
  a ≠ b → a ≠ c → a ≠ d → a ≠ e → a ≠ f → a ≠ g → a ≠ h →
  b ≠ c → b ≠ d → b ≠ e → b ≠ f → b ≠ g → b ≠ h →
  c ≠ d → c ≠ e → c ≠ f → c ≠ g → c ≠ h →
  d ≠ e → d ≠ f → d ≠ g → d ≠ h →
  e ≠ f → e ≠ g → e ≠ h →
  f ≠ g → f ≠ h →
  g ≠ h →
  34 ≤ (a + b + c + d)^2 + (e + f + g + h)^2 :=
by
  sorry

end min_sum_squares_l978_97845


namespace quadratic_inequality_solutions_l978_97828

/-- Solution set A for x^2 - 3x + 2 > 0 -/
def A : Set ℝ := {x | x^2 - 3*x + 2 > 0}

/-- Solution set B for mx^2 - (m+2)x + 2 < 0, where m ∈ ℝ -/
def B (m : ℝ) : Set ℝ := {x | m*x^2 - (m+2)*x + 2 < 0}

/-- Complement of A in ℝ -/
def complement_A : Set ℝ := {x | ¬(x ∈ A)}

theorem quadratic_inequality_solutions (m : ℝ) :
  (B m ⊆ complement_A ↔ 1 ≤ m ∧ m ≤ 2) ∧
  ((A ∩ B m).Nonempty ↔ m < 1 ∨ m > 2) ∧
  (A ∪ B m = A ↔ m ≥ 2) := by
  sorry

end quadratic_inequality_solutions_l978_97828


namespace equation_solution_l978_97886

theorem equation_solution : 
  ∃! x : ℚ, (x - 15) / 3 = (3 * x + 10) / 8 :=
by
  use (-150)
  constructor
  · -- Prove that x = -150 satisfies the equation
    sorry
  · -- Prove uniqueness
    sorry

end equation_solution_l978_97886


namespace storage_unit_blocks_l978_97837

/-- Represents the dimensions of a rectangular storage unit -/
structure StorageUnit where
  length : ℝ
  width : ℝ
  height : ℝ
  wallThickness : ℝ

/-- Calculates the number of blocks needed for a storage unit -/
def blocksNeeded (unit : StorageUnit) : ℝ :=
  let totalVolume := unit.length * unit.width * unit.height
  let interiorLength := unit.length - 2 * unit.wallThickness
  let interiorWidth := unit.width - 2 * unit.wallThickness
  let interiorHeight := unit.height - unit.wallThickness
  let interiorVolume := interiorLength * interiorWidth * interiorHeight
  totalVolume - interiorVolume

/-- Theorem stating that the storage unit with given dimensions requires 738 blocks -/
theorem storage_unit_blocks :
  let unit : StorageUnit := {
    length := 15,
    width := 12,
    height := 8,
    wallThickness := 1.5
  }
  blocksNeeded unit = 738 := by sorry

end storage_unit_blocks_l978_97837


namespace disjoint_sets_range_l978_97884

def set_A : Set (ℝ × ℝ) := {p | p.2 = -|p.1| - 2}

def set_B (a : ℝ) : Set (ℝ × ℝ) := {p | (p.1 - a)^2 + p.2^2 = a^2}

theorem disjoint_sets_range (a : ℝ) :
  set_A ∩ set_B a = ∅ ↔ -2 * Real.sqrt 2 - 2 < a ∧ a < 2 * Real.sqrt 2 + 2 :=
sorry

end disjoint_sets_range_l978_97884


namespace problem_solution_l978_97865

theorem problem_solution : 
  (Real.sqrt 27 + Real.sqrt 2 * Real.sqrt 6 + Real.sqrt 20 - 5 * Real.sqrt (1/5) = 5 * Real.sqrt 3 + Real.sqrt 5) ∧
  ((Real.sqrt 2 - 1) * (Real.sqrt 2 + 1) + (Real.sqrt 3 - 2) = Real.sqrt 3 - 1) := by
  sorry

end problem_solution_l978_97865


namespace distribute_six_balls_three_boxes_l978_97834

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 132 ways to distribute 6 distinguishable balls into 3 indistinguishable boxes -/
theorem distribute_six_balls_three_boxes : distribute_balls 6 3 = 132 := by
  sorry

end distribute_six_balls_three_boxes_l978_97834


namespace cone_base_circumference_l978_97862

/-- The circumference of the base of a right circular cone formed by removing a 180° sector from a circle with radius 6 inches is equal to 6π inches. -/
theorem cone_base_circumference (r : ℝ) (θ : ℝ) : 
  r = 6 → θ = 180 → 2 * π * r * (θ / 360) = 6 * π := by sorry

end cone_base_circumference_l978_97862


namespace sheet_to_box_volume_l978_97867

/-- Represents the dimensions of a rectangular sheet. -/
structure SheetDimensions where
  length : ℝ
  width : ℝ

/-- Represents the sizes of squares cut from corners. -/
structure CornerCuts where
  cut1 : ℝ
  cut2 : ℝ
  cut3 : ℝ
  cut4 : ℝ

/-- Represents the dimensions of the resulting box. -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a box given its dimensions. -/
def boxVolume (box : BoxDimensions) : ℝ :=
  box.length * box.width * box.height

/-- Theorem stating the relationship between the original sheet, corner cuts, and resulting box. -/
theorem sheet_to_box_volume 
  (sheet : SheetDimensions) 
  (cuts : CornerCuts) 
  (box : BoxDimensions) : 
  sheet.length = 48 ∧ 
  sheet.width = 36 ∧
  cuts.cut1 = 7 ∧ 
  cuts.cut2 = 5 ∧ 
  cuts.cut3 = 6 ∧ 
  cuts.cut4 = 4 ∧
  box.length = sheet.length - (cuts.cut1 + cuts.cut4) ∧
  box.width = sheet.width - (cuts.cut2 + cuts.cut3) ∧
  box.height = min cuts.cut1 (min cuts.cut2 (min cuts.cut3 cuts.cut4)) →
  boxVolume box = 3700 ∧ 
  box.length = 37 ∧ 
  box.width = 25 := by
  sorry

end sheet_to_box_volume_l978_97867


namespace inequality_solution_set_l978_97889

theorem inequality_solution_set :
  {x : ℝ | |x - 1| + 2*x > 4} = {x : ℝ | x ≥ 1} := by sorry

end inequality_solution_set_l978_97889


namespace log_sum_equals_zero_l978_97844

theorem log_sum_equals_zero (a b : ℝ) 
  (ha : a > 1) 
  (hb : b > 1) 
  (h_log : Real.log (a + b) = Real.log a + Real.log b) : 
  Real.log (a - 1) + Real.log (b - 1) = 0 := by
sorry

end log_sum_equals_zero_l978_97844


namespace P_sufficient_not_necessary_for_Q_l978_97809

-- Define the propositions P and Q
def P (x : ℝ) : Prop := |2*x - 3| < 1
def Q (x : ℝ) : Prop := x*(x - 3) < 0

-- Theorem stating that P is a sufficient but not necessary condition for Q
theorem P_sufficient_not_necessary_for_Q :
  (∀ x : ℝ, P x → Q x) ∧ 
  (∃ x : ℝ, Q x ∧ ¬(P x)) := by
  sorry

end P_sufficient_not_necessary_for_Q_l978_97809


namespace tank_water_problem_l978_97849

theorem tank_water_problem (added_saline : ℝ) (salt_concentration_added : ℝ) 
  (salt_concentration_final : ℝ) (initial_water : ℝ) : 
  added_saline = 66.67 →
  salt_concentration_added = 0.25 →
  salt_concentration_final = 0.10 →
  initial_water = 100 →
  salt_concentration_added * added_saline = 
    salt_concentration_final * (initial_water + added_saline) :=
by sorry

end tank_water_problem_l978_97849


namespace stratified_sampling_most_appropriate_l978_97840

/-- Represents a sampling method -/
inductive SamplingMethod
  | Simple
  | Stratified
  | Systematic

/-- Represents a population with subgroups -/
structure Population where
  subgroups : List (Set α)
  significant_differences : Bool

/-- Determines the most appropriate sampling method for a given population -/
def most_appropriate_sampling_method (pop : Population) : SamplingMethod :=
  if pop.significant_differences then
    SamplingMethod.Stratified
  else
    SamplingMethod.Simple

/-- Theorem stating that stratified sampling is most appropriate for populations with significant differences between subgroups -/
theorem stratified_sampling_most_appropriate
  (pop : Population)
  (h : pop.significant_differences = true) :
  most_appropriate_sampling_method pop = SamplingMethod.Stratified :=
by sorry

end stratified_sampling_most_appropriate_l978_97840


namespace solution_set_equivalence_l978_97859

/-- The set of points (x, y) satisfying y(x+1) = x^2 - 1 -/
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 * (p.1 + 1) = p.1^2 - 1}

/-- The vertical line x = -1 -/
def L1 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = -1}

/-- The line y = x - 1 -/
def L2 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = p.1 - 1}

/-- Theorem stating that S is equivalent to the union of L1 and L2 -/
theorem solution_set_equivalence : S = L1 ∪ L2 := by
  sorry

end solution_set_equivalence_l978_97859


namespace last_two_digits_of_product_squared_l978_97810

theorem last_two_digits_of_product_squared : 
  (301 * 402 * 503 * 604 * 646 * 547 * 448 * 349)^2 % 100 = 76 := by
  sorry

end last_two_digits_of_product_squared_l978_97810


namespace sara_joe_height_difference_l978_97813

/-- Given the heights of Sara, Joe, and Roy, prove that Sara is 6 inches taller than Joe. -/
theorem sara_joe_height_difference :
  ∀ (sara_height joe_height roy_height : ℕ),
    sara_height = 45 →
    joe_height = roy_height + 3 →
    roy_height = 36 →
    sara_height - joe_height = 6 := by
sorry

end sara_joe_height_difference_l978_97813


namespace line_equations_l978_97824

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 4*y - 4 = 0

-- Define point P
def point_P : ℝ × ℝ := (2, 0)

-- Define the property of line l1
def line_l1_property (l : ℝ → ℝ → Prop) : Prop :=
  l point_P.1 point_P.2 ∧
  ∃ a b : ℝ, ∀ x y : ℝ, l x y ↔ (x = a ∨ b*x - y = 0) ∧
  ∃ x1 y1 x2 y2 : ℝ,
    circle_C x1 y1 ∧ circle_C x2 y2 ∧ l x1 y1 ∧ l x2 y2 ∧
    (x1 - x2)^2 + (y1 - y2)^2 = 32

-- Define the property of line l2
def line_l2_property (l : ℝ → ℝ → Prop) : Prop :=
  (∀ x y z w : ℝ, l x y ∧ l z w → y - x = w - z) ∧
  ∃ x1 y1 x2 y2 : ℝ,
    circle_C x1 y1 ∧ circle_C x2 y2 ∧ l x1 y1 ∧ l x2 y2 ∧
    x1*x2 + y1*y2 = 0

-- Theorem statement
theorem line_equations :
  ∃ l1 l2 : ℝ → ℝ → Prop,
    line_l1_property l1 ∧ line_l2_property l2 ∧
    (∀ x y : ℝ, l1 x y ↔ (x = 2 ∨ 3*x - 4*y - 6 = 0)) ∧
    (∀ x y : ℝ, l2 x y ↔ (x - y - 4 = 0 ∨ x - y + 1 = 0)) :=
by sorry

end line_equations_l978_97824


namespace motorcycles_parked_count_l978_97821

/-- The number of motorcycles parked between cars on a road -/
def motorcycles_parked (foreign_cars : ℕ) (domestic_cars_between : ℕ) : ℕ :=
  let total_cars := foreign_cars + (foreign_cars - 1) * domestic_cars_between
  total_cars - 1

/-- Theorem stating that given 5 foreign cars and 2 domestic cars between each pair,
    the number of motorcycles parked between all adjacent cars is 12 -/
theorem motorcycles_parked_count :
  motorcycles_parked 5 2 = 12 := by
  sorry

end motorcycles_parked_count_l978_97821


namespace parabola_tangent_to_line_l978_97829

/-- A parabola y = ax^2 + 8 is tangent to the line y = 2x + 3 if and only if a = 1/5 -/
theorem parabola_tangent_to_line (a : ℝ) : 
  (∃! x : ℝ, a * x^2 + 8 = 2 * x + 3) ↔ a = 1/5 := by
  sorry

end parabola_tangent_to_line_l978_97829


namespace polynomial_roots_l978_97814

theorem polynomial_roots : ∃ (p : ℝ → ℝ), 
  (∀ x, p x = 8*x^4 + 14*x^3 - 66*x^2 + 40*x) ∧ 
  (p 0 = 0) ∧ (p (1/2) = 0) ∧ (p 2 = 0) ∧ (p (-5) = 0) := by
  sorry

end polynomial_roots_l978_97814


namespace subtraction_of_negative_two_minus_negative_four_equals_six_l978_97871

theorem subtraction_of_negative (a b : ℤ) : a - (-b) = a + b := by sorry

theorem two_minus_negative_four_equals_six : 2 - (-4) = 6 := by sorry

end subtraction_of_negative_two_minus_negative_four_equals_six_l978_97871


namespace product_of_sum_and_sum_of_cubes_l978_97894

theorem product_of_sum_and_sum_of_cubes (a b : ℝ) 
  (h1 : a + b = 5) 
  (h2 : a^3 + b^3 = 35) : 
  a * b = 6 := by
sorry

end product_of_sum_and_sum_of_cubes_l978_97894


namespace fraction_simplification_l978_97858

theorem fraction_simplification (a b : ℕ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : a / b = (100 * a + a) / (100 * b + b)) 
  (h4 : a / b = (10000 * a + 100 * a + a) / (10000 * b + 100 * b + b)) :
  ∀ (d : ℕ), d > 1 → d ∣ a → d ∣ b → False :=
by sorry

end fraction_simplification_l978_97858


namespace inequality_proof_l978_97864

theorem inequality_proof (a b : ℝ) (h1 : a > b) (h2 : b > 1) : a * Real.exp b < b * Real.exp a := by
  sorry

end inequality_proof_l978_97864


namespace abs_sum_minimum_l978_97892

theorem abs_sum_minimum (x : ℝ) : 
  |x + 3| + |x + 5| + |x + 6| ≥ 1 ∧ ∃ y : ℝ, |y + 3| + |y + 5| + |y + 6| = 1 := by
  sorry

end abs_sum_minimum_l978_97892


namespace first_discount_percentage_l978_97826

theorem first_discount_percentage (original_price final_price : ℝ) 
  (h1 : original_price = 10000)
  (h2 : final_price = 6840) : ∃ x : ℝ,
  final_price = original_price * (100 - x) / 100 * 90 / 100 * 95 / 100 ∧ x = 20 := by
  sorry

end first_discount_percentage_l978_97826


namespace childs_running_speed_l978_97853

/-- Proves that the child's running speed on a still sidewalk is 74 m/min given the problem conditions -/
theorem childs_running_speed 
  (speed_still : ℝ) 
  (sidewalk_speed : ℝ) 
  (distance_against : ℝ) 
  (time_against : ℝ) 
  (h1 : speed_still = 74) 
  (h2 : distance_against = 165) 
  (h3 : time_against = 3) 
  (h4 : (speed_still - sidewalk_speed) * time_against = distance_against) : 
  speed_still = 74 := by
sorry

end childs_running_speed_l978_97853


namespace unique_multiplication_problem_l978_97873

theorem unique_multiplication_problem :
  ∃! (a b : ℕ), 
    10 ≤ a ∧ a < 100 ∧
    10 ≤ b ∧ b < 100 ∧
    100 ≤ a * b ∧ a * b < 1000 ∧
    (a * b) % 100 / 10 = 1 ∧
    (a * b) % 10 = 2 ∧
    (b * (a % 10)) % 100 = 0 ∧
    (a % 10 + b % 10) = 6 ∧
    a * b = 612 :=
by sorry

end unique_multiplication_problem_l978_97873


namespace largest_number_l978_97860

theorem largest_number : 
  let numbers : List ℝ := [0.978, 0.9719, 0.9781, 0.917, 0.9189]
  ∀ x ∈ numbers, x ≤ 0.9781 := by
  sorry

end largest_number_l978_97860


namespace max_sum_of_square_roots_l978_97811

theorem max_sum_of_square_roots (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 7) :
  (Real.sqrt (3 * x + 2) + Real.sqrt (3 * y + 2) + Real.sqrt (3 * z + 2)) ≤ 9 ∧
  ∃ x y z, x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ x + y + z = 7 ∧
    Real.sqrt (3 * x + 2) + Real.sqrt (3 * y + 2) + Real.sqrt (3 * z + 2) = 9 :=
by sorry

end max_sum_of_square_roots_l978_97811


namespace max_gcd_sum_1729_l978_97851

theorem max_gcd_sum_1729 (a b : ℕ+) (h : a + b = 1729) : 
  ∃ (x y : ℕ+), x + y = 1729 ∧ Nat.gcd x y = 247 ∧ 
  ∀ (c d : ℕ+), c + d = 1729 → Nat.gcd c d ≤ 247 := by
  sorry

end max_gcd_sum_1729_l978_97851


namespace square_root_of_64_l978_97803

theorem square_root_of_64 : {x : ℝ | x^2 = 64} = {-8, 8} := by sorry

end square_root_of_64_l978_97803


namespace a_range_l978_97868

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x + a^2 / (4 * x)

noncomputable def g (x : ℝ) : ℝ := x - log x

theorem a_range (a : ℝ) (h1 : a > 1) 
  (h2 : ∀ (x₁ x₂ : ℝ), 1 ≤ x₁ ∧ x₁ ≤ Real.exp 1 ∧ 1 ≤ x₂ ∧ x₂ ≤ Real.exp 1 → f a x₁ ≥ g x₂) : 
  a ≥ 2 * sqrt (Real.exp 1 - 2) :=
sorry

end a_range_l978_97868


namespace perpendicular_lines_a_values_l978_97848

theorem perpendicular_lines_a_values 
  (a : ℝ) 
  (h_perp : (a * (2*a - 1)) + (-1 * a) = 0) : 
  a = 1 ∨ a = 0 := by
sorry

end perpendicular_lines_a_values_l978_97848


namespace simplify_sqrt_fraction_simplify_sqrt_expression_l978_97883

-- Part 1
theorem simplify_sqrt_fraction :
  (Real.sqrt 5 + 1) / (Real.sqrt 5 - 1) = (3 + Real.sqrt 5) / 2 := by sorry

-- Part 2
theorem simplify_sqrt_expression :
  Real.sqrt 12 * Real.sqrt 2 / Real.sqrt ((-3)^2) = 2 * Real.sqrt 6 / 3 := by sorry

end simplify_sqrt_fraction_simplify_sqrt_expression_l978_97883


namespace two_consecutive_late_charges_l978_97882

theorem two_consecutive_late_charges (original_bill : ℝ) (late_charge_rate : ℝ) : 
  original_bill = 500 →
  late_charge_rate = 0.02 →
  (original_bill * (1 + late_charge_rate) * (1 + late_charge_rate)) = 520.20 := by
  sorry

end two_consecutive_late_charges_l978_97882


namespace probability_of_speaking_hindi_l978_97874

/-- The probability of speaking Hindi in a village -/
theorem probability_of_speaking_hindi 
  (total_population : ℕ) 
  (tamil_speakers : ℕ) 
  (english_speakers : ℕ) 
  (both_speakers : ℕ) 
  (h_total : total_population = 1024)
  (h_tamil : tamil_speakers = 720)
  (h_english : english_speakers = 562)
  (h_both : both_speakers = 346)
  (h_non_negative : total_population ≥ tamil_speakers + english_speakers - both_speakers) :
  (total_population - (tamil_speakers + english_speakers - both_speakers)) / total_population = 
  (1024 - (720 + 562 - 346)) / 1024 := by
  sorry

end probability_of_speaking_hindi_l978_97874


namespace f_properties_l978_97808

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x)^2 + 2 * Real.sin x * Real.cos x

theorem f_properties :
  (f (π / 8) = Real.sqrt 2 + 1) ∧
  (∀ T > 0, (∀ x, f (x + T) = f x) → T ≥ π) ∧
  (∀ x, f x ≥ 1 - Real.sqrt 2) ∧
  (∃ x, f x = 1 - Real.sqrt 2) := by
sorry

end f_properties_l978_97808


namespace united_additional_charge_value_l978_97839

/-- Represents the additional charge per minute for United Telephone -/
def united_additional_charge : ℝ := sorry

/-- The base rate for United Telephone -/
def united_base_rate : ℝ := 11

/-- The base rate for Atlantic Call -/
def atlantic_base_rate : ℝ := 12

/-- The additional charge per minute for Atlantic Call -/
def atlantic_additional_charge : ℝ := 0.2

/-- The number of minutes for which the bills are equal -/
def equal_bill_minutes : ℝ := 20

theorem united_additional_charge_value : 
  (united_base_rate + equal_bill_minutes * united_additional_charge = 
   atlantic_base_rate + equal_bill_minutes * atlantic_additional_charge) → 
  united_additional_charge = 0.25 := by sorry

end united_additional_charge_value_l978_97839

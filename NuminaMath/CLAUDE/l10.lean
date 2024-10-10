import Mathlib

namespace two_fifths_of_n_is_80_l10_1062

theorem two_fifths_of_n_is_80 (n : ℚ) : n = 5 / 6 * 240 → 2 / 5 * n = 80 := by
  sorry

end two_fifths_of_n_is_80_l10_1062


namespace field_trip_vans_l10_1011

/-- The number of vans needed for a field trip --/
def vans_needed (students : ℕ) (adults : ℕ) (van_capacity : ℕ) : ℕ :=
  ((students + adults + van_capacity - 1) / van_capacity : ℕ)

/-- Theorem: For 33 students, 9 adults, and vans with capacity 7, 6 vans are needed --/
theorem field_trip_vans : vans_needed 33 9 7 = 6 := by
  sorry

end field_trip_vans_l10_1011


namespace five_colored_flags_count_l10_1095

/-- The number of different colors available -/
def num_colors : ℕ := 11

/-- The number of stripes in the flag -/
def num_stripes : ℕ := 5

/-- The number of ways to choose and arrange colors for the flag -/
def num_flags : ℕ := (num_colors.choose num_stripes) * num_stripes.factorial

/-- Theorem stating the number of different five-colored flags -/
theorem five_colored_flags_count : num_flags = 55440 := by
  sorry

end five_colored_flags_count_l10_1095


namespace charles_vowel_learning_time_l10_1067

/-- The number of days Charles takes to learn one alphabet. -/
def days_per_alphabet : ℕ := 7

/-- The number of vowels in the English alphabet. -/
def number_of_vowels : ℕ := 5

/-- The total number of days Charles needs to finish learning all vowels. -/
def total_days : ℕ := days_per_alphabet * number_of_vowels

theorem charles_vowel_learning_time : total_days = 35 := by
  sorry

end charles_vowel_learning_time_l10_1067


namespace b_95_mod_49_l10_1068

def b (n : ℕ) : ℕ := 7^n + 9^n

theorem b_95_mod_49 : b 95 ≡ 28 [ZMOD 49] := by sorry

end b_95_mod_49_l10_1068


namespace team_ate_63_slices_l10_1096

/-- Represents the number of slices in different pizza sizes -/
structure PizzaSlices where
  extraLarge : Nat
  large : Nat
  medium : Nat

/-- Represents the number of pizzas of each size -/
structure PizzaCounts where
  extraLarge : Nat
  large : Nat
  medium : Nat

/-- Calculates the total number of slices from all pizzas -/
def totalSlices (slices : PizzaSlices) (counts : PizzaCounts) : Nat :=
  slices.extraLarge * counts.extraLarge +
  slices.large * counts.large +
  slices.medium * counts.medium

/-- Theorem stating that the team ate 63 slices of pizza -/
theorem team_ate_63_slices 
  (slices : PizzaSlices)
  (counts : PizzaCounts)
  (h1 : slices.extraLarge = 16)
  (h2 : slices.large = 12)
  (h3 : slices.medium = 8)
  (h4 : counts.extraLarge = 3)
  (h5 : counts.large = 2)
  (h6 : counts.medium = 1)
  (h7 : totalSlices slices counts - 17 = 63) :
  63 = totalSlices slices counts - 17 := by
  sorry

#eval totalSlices ⟨16, 12, 8⟩ ⟨3, 2, 1⟩ - 17

end team_ate_63_slices_l10_1096


namespace min_value_implies_b_range_l10_1083

-- Define the function f
def f (b : ℝ) (x : ℝ) : ℝ := x^3 - 6*b*x + 3*b

-- Define the derivative of f
def f' (b : ℝ) (x : ℝ) : ℝ := 3*x^2 - 6*b

-- State the theorem
theorem min_value_implies_b_range (b : ℝ) :
  (∃ x ∈ (Set.Ioo 0 1), ∀ y ∈ (Set.Ioo 0 1), f b x ≤ f b y) →
  b ∈ (Set.Ioo 0 (1/2)) :=
by sorry

end min_value_implies_b_range_l10_1083


namespace chess_tournament_matches_chess_tournament_problem_l10_1033

/-- Represents a single elimination chess tournament --/
structure ChessTournament where
  total_players : ℕ
  bye_players : ℕ
  matches_played : ℕ

/-- Theorem stating the number of matches in the given tournament --/
theorem chess_tournament_matches 
  (tournament : ChessTournament) 
  (h1 : tournament.total_players = 128) 
  (h2 : tournament.bye_players = 32) : 
  tournament.matches_played = 127 := by
  sorry

/-- Main theorem to be proved --/
theorem chess_tournament_problem : 
  ∃ (t : ChessTournament), t.total_players = 128 ∧ t.bye_players = 32 ∧ t.matches_played = 127 := by
  sorry

end chess_tournament_matches_chess_tournament_problem_l10_1033


namespace jacket_price_restoration_l10_1034

theorem jacket_price_restoration (initial_price : ℝ) (h_pos : initial_price > 0) :
  let price_after_first_reduction := initial_price * (1 - 0.15)
  let price_after_second_reduction := price_after_first_reduction * (1 - 0.30)
  let required_increase := (initial_price / price_after_second_reduction) - 1
  abs (required_increase - 0.6807) < 0.0001 := by
  sorry

end jacket_price_restoration_l10_1034


namespace base_n_not_prime_l10_1028

/-- For a positive integer n ≥ 2, 2002_n represents the number in base n notation -/
def base_n (n : ℕ) : ℕ := 2 * n^3 + 2

/-- A number is prime if it has exactly two distinct positive divisors -/
def is_prime (p : ℕ) : Prop := p > 1 ∧ (∀ m : ℕ, m > 0 → m < p → p % m ≠ 0)

theorem base_n_not_prime (n : ℕ) (h : n ≥ 2) : ¬ (is_prime (base_n n)) := by
  sorry

end base_n_not_prime_l10_1028


namespace total_textbook_cost_l10_1025

/-- The total cost of textbooks given specific pricing conditions -/
theorem total_textbook_cost : 
  ∀ (sale_price : ℕ) (online_total : ℕ) (sale_count online_count bookstore_count : ℕ),
    sale_price = 10 →
    online_total = 40 →
    sale_count = 5 →
    online_count = 2 →
    bookstore_count = 3 →
    sale_count * sale_price + online_total + bookstore_count * online_total = 210 :=
by sorry

end total_textbook_cost_l10_1025


namespace two_common_tangents_l10_1055

/-- The number of common tangents between two circles -/
def num_common_tangents (C₁ C₂ : ℝ → ℝ → Prop) : ℕ :=
  sorry

/-- Circle C₁ equation -/
def C₁ (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x + 8*y - 8 = 0

/-- Circle C₂ equation -/
def C₂ (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 4*y - 1 = 0

/-- Theorem stating that there are 2 common tangents between C₁ and C₂ -/
theorem two_common_tangents : num_common_tangents C₁ C₂ = 2 :=
  sorry

end two_common_tangents_l10_1055


namespace x_plus_y_equals_one_l10_1087

theorem x_plus_y_equals_one (x y : ℝ) 
  (h1 : 2021 * x + 2025 * y = 2029)
  (h2 : 2023 * x + 2027 * y = 2031) : 
  x + y = 1 := by
sorry

end x_plus_y_equals_one_l10_1087


namespace pencil_price_l10_1098

theorem pencil_price (num_pens : ℕ) (num_pencils : ℕ) (total_cost : ℚ) (pen_price : ℚ) :
  num_pens = 30 →
  num_pencils = 75 →
  total_cost = 510 →
  pen_price = 12 →
  (total_cost - num_pens * pen_price) / num_pencils = 2 :=
by sorry

end pencil_price_l10_1098


namespace least_possible_b_in_right_triangle_l10_1043

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | n + 2 => fib n + fib (n + 1)

-- Define a predicate to check if a number is in the Fibonacci sequence
def is_fibonacci (n : ℕ) : Prop :=
  ∃ k, fib k = n

theorem least_possible_b_in_right_triangle :
  ∀ a b : ℕ,
  a + b = 90 →  -- Sum of acute angles in a right triangle is 90°
  a > b →  -- a is greater than b
  is_fibonacci a →  -- a is in the Fibonacci sequence
  is_fibonacci b →  -- b is in the Fibonacci sequence
  b ≥ 1 →  -- b is at least 1 (as it's an angle)
  ∀ c : ℕ, (c < b ∧ is_fibonacci c) → c < 1 :=
by sorry

#check least_possible_b_in_right_triangle

end least_possible_b_in_right_triangle_l10_1043


namespace ball_drawing_probabilities_l10_1038

/-- Represents the color of a ball -/
inductive BallColor
| Red
| White

/-- Represents a ball with its color and number -/
structure Ball :=
  (color : BallColor)
  (number : Nat)

/-- The box of balls -/
def box : List Ball := [
  ⟨BallColor.Red, 1⟩, ⟨BallColor.Red, 2⟩, ⟨BallColor.Red, 3⟩, ⟨BallColor.Red, 4⟩,
  ⟨BallColor.White, 3⟩, ⟨BallColor.White, 4⟩
]

/-- The number of balls to draw -/
def drawCount : Nat := 3

/-- Calculates the probability of drawing 3 balls with maximum number 3 -/
def probMaxThree : ℚ := 1 / 5

/-- Calculates the mathematical expectation of the maximum number among red balls drawn -/
def expectationMaxRed : ℚ := 13 / 4

theorem ball_drawing_probabilities :
  (probMaxThree = 1 / 5) ∧
  (expectationMaxRed = 13 / 4) := by
  sorry


end ball_drawing_probabilities_l10_1038


namespace partner_calculation_l10_1021

theorem partner_calculation (x : ℝ) : 3 * (3 * (x + 2) - 2) = 3 * (3 * x + 4) := by
  sorry

#check partner_calculation

end partner_calculation_l10_1021


namespace standard_deviation_measures_stability_l10_1065

/-- A measure of stability for a set of numbers -/
def stability_measure (data : List ℝ) : ℝ := sorry

/-- Standard deviation of a list of real numbers -/
def standard_deviation (data : List ℝ) : ℝ := sorry

/-- Theorem stating that the standard deviation is a valid measure of stability for crop yields -/
theorem standard_deviation_measures_stability 
  (n : ℕ) 
  (yields : List ℝ) 
  (h1 : yields.length = n) 
  (h2 : n > 0) :
  stability_measure yields = standard_deviation yields := by sorry

end standard_deviation_measures_stability_l10_1065


namespace max_distance_to_line_l10_1074

/-- The intersection point of two lines -/
def intersection_point (l1 l2 : ℝ × ℝ → Prop) : ℝ × ℝ :=
  sorry

/-- The distance between two points in ℝ² -/
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  sorry

/-- A line in ℝ² represented by its equation -/
def line (a b c : ℝ) : ℝ × ℝ → Prop :=
  fun p => a * p.1 + b * p.2 + c = 0

theorem max_distance_to_line :
  let l1 := line 1 1 (-1)
  let l2 := line 1 (-2) (-4)
  let p := intersection_point l1 l2
  ∀ k : ℝ,
    let l3 := line k (-1) (1 + 2*k)
    ∀ q : ℝ × ℝ,
      l3 q →
      distance p q ≤ 2 * Real.sqrt 5 :=
by sorry

end max_distance_to_line_l10_1074


namespace ellipse_focus_k_value_l10_1075

/-- An ellipse with equation 5x^2 + ky^2 = 5 and one focus at (0, 2) has k = 1 -/
theorem ellipse_focus_k_value (k : ℝ) :
  (∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    ∀ (x y : ℝ), 5 * x^2 + k * y^2 = 5 ↔
      (x^2 / a^2 + y^2 / b^2 = 1 ∧
       c^2 = a^2 - b^2 ∧
       2^2 = c^2)) →
  k = 1 :=
by sorry

end ellipse_focus_k_value_l10_1075


namespace four_numbers_sum_product_l10_1026

def satisfies_condition (x₁ x₂ x₃ x₄ : ℝ) : Prop :=
  x₁ + x₂ * x₃ * x₄ = 2 ∧
  x₂ + x₁ * x₃ * x₄ = 2 ∧
  x₃ + x₁ * x₂ * x₄ = 2 ∧
  x₄ + x₁ * x₂ * x₃ = 2

def is_permutation (a b c d : ℝ) (x₁ x₂ x₃ x₄ : ℝ) : Prop :=
  (x₁ = a ∧ x₂ = b ∧ x₃ = c ∧ x₄ = d) ∨
  (x₁ = a ∧ x₂ = b ∧ x₃ = d ∧ x₄ = c) ∨
  (x₁ = a ∧ x₂ = c ∧ x₃ = b ∧ x₄ = d) ∨
  (x₁ = a ∧ x₂ = c ∧ x₃ = d ∧ x₄ = b) ∨
  (x₁ = a ∧ x₂ = d ∧ x₃ = b ∧ x₄ = c) ∨
  (x₁ = a ∧ x₂ = d ∧ x₃ = c ∧ x₄ = b) ∨
  (x₁ = b ∧ x₂ = a ∧ x₃ = c ∧ x₄ = d) ∨
  (x₁ = b ∧ x₂ = a ∧ x₃ = d ∧ x₄ = c) ∨
  (x₁ = b ∧ x₂ = c ∧ x₃ = a ∧ x₄ = d) ∨
  (x₁ = b ∧ x₂ = c ∧ x₃ = d ∧ x₄ = a) ∨
  (x₁ = b ∧ x₂ = d ∧ x₃ = a ∧ x₄ = c) ∨
  (x₁ = b ∧ x₂ = d ∧ x₃ = c ∧ x₄ = a) ∨
  (x₁ = c ∧ x₂ = a ∧ x₃ = b ∧ x₄ = d) ∨
  (x₁ = c ∧ x₂ = a ∧ x₃ = d ∧ x₄ = b) ∨
  (x₁ = c ∧ x₂ = b ∧ x₃ = a ∧ x₄ = d) ∨
  (x₁ = c ∧ x₂ = b ∧ x₃ = d ∧ x₄ = a) ∨
  (x₁ = c ∧ x₂ = d ∧ x₃ = a ∧ x₄ = b) ∨
  (x₁ = c ∧ x₂ = d ∧ x₃ = b ∧ x₄ = a) ∨
  (x₁ = d ∧ x₂ = a ∧ x₃ = b ∧ x₄ = c) ∨
  (x₁ = d ∧ x₂ = a ∧ x₃ = c ∧ x₄ = b) ∨
  (x₁ = d ∧ x₂ = b ∧ x₃ = a ∧ x₄ = c) ∨
  (x₁ = d ∧ x₂ = b ∧ x₃ = c ∧ x₄ = a) ∨
  (x₁ = d ∧ x₂ = c ∧ x₃ = a ∧ x₄ = b) ∨
  (x₁ = d ∧ x₂ = c ∧ x₃ = b ∧ x₄ = a)

theorem four_numbers_sum_product (x₁ x₂ x₃ x₄ : ℝ) :
  satisfies_condition x₁ x₂ x₃ x₄ ↔
  (x₁ = 1 ∧ x₂ = 1 ∧ x₃ = 1 ∧ x₄ = 1) ∨
  is_permutation 3 (-1) (-1) (-1) x₁ x₂ x₃ x₄ :=
sorry

end four_numbers_sum_product_l10_1026


namespace right_handed_players_count_l10_1057

theorem right_handed_players_count (total_players : ℕ) (throwers : ℕ) : 
  total_players = 70 →
  throwers = 46 →
  (total_players - throwers) / 3 * 2 + throwers = 62 :=
by
  sorry

end right_handed_players_count_l10_1057


namespace volume_to_surface_area_ratio_l10_1088

-- Define the shape
structure CubeShape :=
  (base_length : ℕ)
  (base_width : ℕ)
  (column_height : ℕ)

-- Define the properties of our specific shape
def our_shape : CubeShape :=
  { base_length := 3
  , base_width := 3
  , column_height := 3 }

-- Calculate the volume of the shape
def volume (shape : CubeShape) : ℕ :=
  shape.base_length * shape.base_width + shape.column_height - 1

-- Calculate the surface area of the shape
def surface_area (shape : CubeShape) : ℕ :=
  let base_area := 2 * shape.base_length * shape.base_width
  let side_area := 2 * shape.base_length * shape.column_height + 2 * shape.base_width * shape.column_height
  let column_area := 4 * (shape.column_height - 1)
  base_area + side_area + column_area - (shape.base_length * shape.base_width - 1)

-- Theorem: The ratio of volume to surface area is 9:40
theorem volume_to_surface_area_ratio (shape : CubeShape) :
  shape = our_shape → volume shape * 40 = surface_area shape * 9 := by
  sorry

end volume_to_surface_area_ratio_l10_1088


namespace polyhedron_edges_l10_1044

theorem polyhedron_edges (F V E : ℕ) : F + V - E = 2 → F = 6 → V = 8 → E = 12 := by
  sorry

end polyhedron_edges_l10_1044


namespace percentage_of_returns_l10_1013

/-- Calculate the percentage of customers returning books -/
theorem percentage_of_returns (total_customers : ℕ) (price_per_book : ℚ) (sales_after_returns : ℚ) :
  total_customers = 1000 →
  price_per_book = 15 →
  sales_after_returns = 9450 →
  (((total_customers : ℚ) * price_per_book - sales_after_returns) / price_per_book) / (total_customers : ℚ) * 100 = 37 :=
by sorry

end percentage_of_returns_l10_1013


namespace trigonometric_identities_l10_1051

theorem trigonometric_identities (α : Real) (h : Real.tan α = 2) :
  (Real.tan (α + π/4) = -3) ∧
  (Real.cos α)^2 = 1/5 ∧
  (Real.sin α / (Real.sin α + Real.cos α) = 2/3) := by
  sorry

end trigonometric_identities_l10_1051


namespace smiths_children_ages_l10_1089

def is_divisible (n m : ℕ) : Prop := m ∣ n

theorem smiths_children_ages (children_ages : Finset ℕ) : 
  children_ages.card = 7 ∧ 
  (∀ a ∈ children_ages, 2 ≤ a ∧ a ≤ 11) ∧
  (∃ x, 2 ≤ x ∧ x ≤ 11 ∧ x ∉ children_ages) ∧
  5 ∈ children_ages ∧
  (∀ a ∈ children_ages, is_divisible 3339 a) ∧
  39 ∉ children_ages →
  6 ∉ children_ages :=
by sorry

end smiths_children_ages_l10_1089


namespace consecutive_card_picks_standard_deck_l10_1003

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (num_suits : ℕ)
  (cards_per_suit : ℕ)
  (face_cards_per_suit : ℕ)
  (number_cards_per_suit : ℕ)

/-- Calculates the number of ways to pick two consecutive cards from the same suit,
    where one is a face card and the other is a number card -/
def consecutive_card_picks (d : Deck) : ℕ :=
  d.num_suits * (d.face_cards_per_suit * d.number_cards_per_suit * 2)

/-- Theorem stating that for a standard deck, there are 240 ways to pick two consecutive
    cards from the same suit, where one is a face card and the other is a number card -/
theorem consecutive_card_picks_standard_deck :
  let d : Deck := {
    total_cards := 48,
    num_suits := 4,
    cards_per_suit := 12,
    face_cards_per_suit := 3,
    number_cards_per_suit := 10
  }
  consecutive_card_picks d = 240 := by
  sorry

end consecutive_card_picks_standard_deck_l10_1003


namespace cubic_not_prime_l10_1081

theorem cubic_not_prime (n : ℕ+) : ¬ Nat.Prime (n.val^3 - 7*n.val^2 + 16*n.val - 12) := by
  sorry

end cubic_not_prime_l10_1081


namespace point_trajectory_l10_1015

-- Define the condition for point M(x,y)
def point_condition (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + (y+2)^2) + Real.sqrt (x^2 + (y-2)^2) = 8

-- Define the trajectory equation
def trajectory_equation (x y : ℝ) : Prop :=
  x^2 / 12 + y^2 / 16 = 1

-- Theorem statement
theorem point_trajectory : ∀ x y : ℝ, point_condition x y → trajectory_equation x y :=
by sorry

end point_trajectory_l10_1015


namespace spring_mows_count_l10_1082

def total_mows : ℕ := 11
def summer_mows : ℕ := 5

theorem spring_mows_count : total_mows - summer_mows = 6 := by
  sorry

end spring_mows_count_l10_1082


namespace sector_central_angle_l10_1064

/-- Given a sector with radius 8 and area 32, prove that its central angle in radians is 1 -/
theorem sector_central_angle (r : ℝ) (area : ℝ) (h1 : r = 8) (h2 : area = 32) :
  let α := 2 * area / (r * r)
  α = 1 := by sorry

end sector_central_angle_l10_1064


namespace cube_sum_of_cyclic_matrix_cube_is_identity_l10_1007

/-- N is a 3x3 matrix with real entries x, y, z -/
def N (x y z : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  !![x, y, z; y, z, x; z, x, y]

/-- The theorem statement -/
theorem cube_sum_of_cyclic_matrix_cube_is_identity
  (x y z : ℝ) (h1 : N x y z ^ 3 = 1) (h2 : x * y * z = -1) :
  x^3 + y^3 + z^3 = -2 := by
  sorry

end cube_sum_of_cyclic_matrix_cube_is_identity_l10_1007


namespace connor_date_expense_l10_1014

/-- The total amount Connor spends on his movie date -/
def connor_total_spent (ticket_price : ℚ) (combo_price : ℚ) (candy_price : ℚ) (cup_price : ℚ) : ℚ :=
  2 * ticket_price + combo_price + 2 * candy_price + cup_price

/-- Theorem: Connor spends $49.00 on his movie date -/
theorem connor_date_expense :
  connor_total_spent 14 11 2.5 5 = 49 := by
  sorry

end connor_date_expense_l10_1014


namespace min_value_expression_l10_1056

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let expr := (|2*a - b + 2*a*(b - a)| + |b + 2*a - a*(b + 4*a)|) / Real.sqrt (4*a^2 + b^2)
  ∃ (min_val : ℝ), (∀ (a' b' : ℝ), a' > 0 → b' > 0 → expr ≥ min_val) ∧ min_val = Real.sqrt 5 / 5 :=
sorry

end min_value_expression_l10_1056


namespace sequence_properties_l10_1016

def sequence_a (n : ℕ) : ℚ := sorry

def sequence_S (n : ℕ) : ℚ := sorry

axiom a_1 : sequence_a 1 = 3

axiom S_def : ∀ n : ℕ, n ≥ 2 → 2 * sequence_a n = sequence_S n * sequence_S (n - 1)

theorem sequence_properties :
  (∃ d : ℚ, ∀ n : ℕ, n ≥ 1 → (1 / sequence_S (n + 1) - 1 / sequence_S n = d)) ∧
  (∀ n : ℕ, n ≥ 2 → sequence_a n = 18 / ((5 - 3 * n) * (8 - 3 * n))) :=
sorry

end sequence_properties_l10_1016


namespace always_odd_l10_1099

theorem always_odd (k : ℤ) : Odd (2007 + 2 * k^2) := by sorry

end always_odd_l10_1099


namespace todd_ingredient_cost_l10_1061

/-- Represents the financial details of Todd's snow-cone business --/
structure SnowConeBusiness where
  borrowed : ℝ
  repay : ℝ
  snowConesSold : ℕ
  pricePerSnowCone : ℝ
  remainingAfterRepay : ℝ

/-- Calculates the amount spent on ingredients for the snow-cone business --/
def ingredientCost (business : SnowConeBusiness) : ℝ :=
  business.borrowed + business.snowConesSold * business.pricePerSnowCone - business.repay - business.remainingAfterRepay

/-- Theorem stating that Todd spent $25 on ingredients --/
theorem todd_ingredient_cost :
  let business : SnowConeBusiness := {
    borrowed := 100,
    repay := 110,
    snowConesSold := 200,
    pricePerSnowCone := 0.75,
    remainingAfterRepay := 65
  }
  ingredientCost business = 25 := by
  sorry


end todd_ingredient_cost_l10_1061


namespace factor_twoOnesWithZeros_l10_1036

/-- Creates a number with two ones and n zeros between them -/
def twoOnesWithZeros (n : ℕ) : ℕ :=
  10^(n + 1) + 1

/-- The other factor in the decomposition -/
def otherFactor (k : ℕ) : ℕ :=
  (10^(3*k + 3) - 1) / 9999

theorem factor_twoOnesWithZeros (k : ℕ) :
  ∃ (m : ℕ), twoOnesWithZeros (3*k + 2) = 73 * 137 * m :=
sorry

end factor_twoOnesWithZeros_l10_1036


namespace reflection_about_x_axis_l10_1048

/-- The reflection of a line about the x-axis -/
def reflect_about_x_axis (line : ℝ → ℝ → Prop) : ℝ → ℝ → Prop :=
  λ x y => line x (-y)

/-- The original line -/
def original_line : ℝ → ℝ → Prop :=
  λ x y => x - y + 1 = 0

/-- The reflected line -/
def reflected_line : ℝ → ℝ → Prop :=
  λ x y => x + y + 1 = 0

theorem reflection_about_x_axis :
  reflect_about_x_axis original_line = reflected_line := by
  sorry

end reflection_about_x_axis_l10_1048


namespace difference_of_numbers_l10_1020

theorem difference_of_numbers (x y : ℝ) 
  (sum_eq : x + y = 580)
  (ratio_eq : x / y = 0.75) : 
  y - x = 83 := by
sorry

end difference_of_numbers_l10_1020


namespace cube_difference_l10_1005

theorem cube_difference (a b : ℝ) (h1 : a - b = 7) (h2 : a^2 + b^2 = 50) : 
  a^3 - b^3 = 353.5 := by sorry

end cube_difference_l10_1005


namespace diophantine_equation_solutions_l10_1041

theorem diophantine_equation_solutions :
  ∀ x y : ℕ, 2 * y^2 - x * y - x^2 + 2 * y + 7 * x - 84 = 0 ↔ (x = 1 ∧ y = 6) ∨ (x = 14 ∧ y = 13) := by
  sorry

end diophantine_equation_solutions_l10_1041


namespace cassie_parrot_count_l10_1069

/-- Represents the number of nails Cassie needs to cut for her pets -/
def total_nails : ℕ := 113

/-- Represents the number of dogs Cassie has -/
def num_dogs : ℕ := 4

/-- Represents the number of nails each dog has -/
def nails_per_dog : ℕ := 16

/-- Represents the number of claws each regular parrot has -/
def claws_per_parrot : ℕ := 6

/-- Represents the number of claws the special parrot with an extra toe has -/
def claws_special_parrot : ℕ := 7

/-- Theorem stating that the number of parrots Cassie has is 8 -/
theorem cassie_parrot_count : 
  ∃ (num_parrots : ℕ), 
    num_parrots * claws_per_parrot + 
    (claws_special_parrot - claws_per_parrot) + 
    (num_dogs * nails_per_dog) = total_nails ∧ 
    num_parrots = 8 := by
  sorry

end cassie_parrot_count_l10_1069


namespace circle_points_count_l10_1012

/-- A circle with n equally spaced points, labeled from 1 to n. -/
structure LabeledCircle where
  n : ℕ
  points : Fin n → ℕ
  labeled_from_1_to_n : ∀ i, points i = i.val + 1

/-- Two points are diametrically opposite if their distance is half the total number of points. -/
def diametrically_opposite (c : LabeledCircle) (i j : Fin c.n) : Prop :=
  (j.val - i.val) % c.n = c.n / 2

/-- The main theorem: if points 7 and 35 are diametrically opposite in a labeled circle, then n = 56. -/
theorem circle_points_count (c : LabeledCircle) 
  (h : ∃ (i j : Fin c.n), c.points i = 7 ∧ c.points j = 35 ∧ diametrically_opposite c i j) : 
  c.n = 56 := by
  sorry

end circle_points_count_l10_1012


namespace planes_parallel_if_skew_lines_parallel_l10_1000

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallelism relation for planes and lines
variable (parallel : Plane → Plane → Prop)
variable (lineParallelPlane : Line → Plane → Prop)

-- Define the containment relation for lines in planes
variable (contains : Plane → Line → Prop)

-- Define the skew relation for lines
variable (skew : Line → Line → Prop)

-- State the theorem
theorem planes_parallel_if_skew_lines_parallel
  (α β : Plane) (a b : Line)
  (h1 : contains α a)
  (h2 : contains β b)
  (h3 : lineParallelPlane a β)
  (h4 : lineParallelPlane b α)
  (h5 : skew a b) :
  parallel α β :=
sorry

end planes_parallel_if_skew_lines_parallel_l10_1000


namespace number_of_students_in_class_l10_1058

/-- Represents the problem of calculating the number of students in a class based on their payments for a science project. -/
theorem number_of_students_in_class
  (full_payment : ℕ)
  (half_payment : ℕ)
  (num_half_payers : ℕ)
  (total_collected : ℕ)
  (h1 : full_payment = 50)
  (h2 : half_payment = 25)
  (h3 : num_half_payers = 4)
  (h4 : total_collected = 1150) :
  ∃ (num_students : ℕ),
    num_students * full_payment - num_half_payers * (full_payment - half_payment) = total_collected ∧
    num_students = 25 :=
by sorry

end number_of_students_in_class_l10_1058


namespace susie_pizza_sales_l10_1031

/-- Represents the pizza sales scenario --/
structure PizzaSales where
  slice_price : ℕ
  whole_price : ℕ
  slices_sold : ℕ
  total_earnings : ℕ

/-- Calculates the number of whole pizzas sold --/
def whole_pizzas_sold (s : PizzaSales) : ℕ :=
  (s.total_earnings - s.slice_price * s.slices_sold) / s.whole_price

/-- Theorem stating that under the given conditions, 3 whole pizzas were sold --/
theorem susie_pizza_sales :
  let s : PizzaSales := {
    slice_price := 3,
    whole_price := 15,
    slices_sold := 24,
    total_earnings := 117
  }
  whole_pizzas_sold s = 3 := by sorry

end susie_pizza_sales_l10_1031


namespace watch_cost_price_l10_1071

/-- The cost price of a watch, given certain selling conditions. -/
def cost_price : ℝ := 1166.67

/-- The selling price at a loss. -/
def selling_price_loss : ℝ := 0.90 * cost_price

/-- The selling price at a gain. -/
def selling_price_gain : ℝ := 1.02 * cost_price

/-- Theorem stating the cost price of the watch given the selling conditions. -/
theorem watch_cost_price :
  (selling_price_loss = 0.90 * cost_price) ∧
  (selling_price_gain = 1.02 * cost_price) ∧
  (selling_price_gain - selling_price_loss = 140) →
  cost_price = 1166.67 :=
by sorry

end watch_cost_price_l10_1071


namespace stating_roper_lawn_cutting_l10_1053

/-- Represents the number of times Mr. Roper cuts his lawn in different periods --/
structure LawnCutting where
  summer_months : ℕ  -- Number of months from April to September
  winter_months : ℕ  -- Number of months from October to March
  summer_cuts : ℕ    -- Number of cuts per month in summer
  average_cuts : ℕ   -- Average number of cuts per month over a year
  total_months : ℕ   -- Total number of months in a year

/-- 
Theorem stating that given the conditions, 
Mr. Roper cuts his lawn 3 times a month from October to March 
-/
theorem roper_lawn_cutting (l : LawnCutting) 
  (h1 : l.summer_months = 6)
  (h2 : l.winter_months = 6)
  (h3 : l.summer_cuts = 15)
  (h4 : l.average_cuts = 9)
  (h5 : l.total_months = 12) :
  (l.total_months * l.average_cuts - l.summer_months * l.summer_cuts) / l.winter_months = 3 := by
  sorry

#check roper_lawn_cutting

end stating_roper_lawn_cutting_l10_1053


namespace area_inequality_l10_1097

/-- Triangle type -/
structure Triangle :=
  (A B C : ℝ × ℝ)

/-- Point on a line segment -/
def PointOnSegment (P : ℝ × ℝ) (A B : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (t * A.1 + (1 - t) * B.1, t * A.2 + (1 - t) * B.2)

/-- Area of a triangle -/
noncomputable def TriangleArea (T : Triangle) : ℝ :=
  abs ((T.B.1 - T.A.1) * (T.C.2 - T.A.2) - (T.C.1 - T.A.1) * (T.B.2 - T.A.2)) / 2

/-- Theorem statement -/
theorem area_inequality (ABC : Triangle) (X Y Z : ℝ × ℝ) 
  (hX : PointOnSegment X ABC.B ABC.C)
  (hY : PointOnSegment Y ABC.C ABC.A)
  (hZ : PointOnSegment Z ABC.A ABC.B)
  (hBX : dist ABC.B X ≤ dist X ABC.C)
  (hCY : dist ABC.C Y ≤ dist Y ABC.A)
  (hAZ : dist ABC.A Z ≤ dist Z ABC.B) :
  4 * TriangleArea ⟨X, Y, Z⟩ ≥ TriangleArea ABC :=
sorry

end area_inequality_l10_1097


namespace max_enclosed_area_l10_1024

/-- Represents the side length of the square garden -/
def s : ℕ := 49

/-- Represents the non-shared side length of the rectangular garden -/
def x : ℕ := 2

/-- The total perimeter of both gardens combined -/
def total_perimeter : ℕ := 200

/-- The maximum area that can be enclosed -/
def max_area : ℕ := 2499

/-- Theorem stating the maximum area that can be enclosed given the constraints -/
theorem max_enclosed_area :
  (4 * s + 2 * x = total_perimeter) → 
  (∀ s' x' : ℕ, (4 * s' + 2 * x' = total_perimeter) → (s' * s' + s' * x' ≤ max_area)) →
  (s * s + s * x = max_area) := by
  sorry

end max_enclosed_area_l10_1024


namespace smallest_four_digit_divisible_by_5_and_9_with_even_digits_l10_1017

def is_even_digit (d : Nat) : Prop := d % 2 = 0 ∧ d < 10

def has_only_even_digits (n : Nat) : Prop :=
  ∀ d, d ∈ n.digits 10 → is_even_digit d

def is_four_digit (n : Nat) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

theorem smallest_four_digit_divisible_by_5_and_9_with_even_digits :
  ∀ n : Nat,
    is_four_digit n ∧
    has_only_even_digits n ∧
    n % 5 = 0 ∧
    n % 9 = 0 →
    2880 ≤ n :=
by sorry

end smallest_four_digit_divisible_by_5_and_9_with_even_digits_l10_1017


namespace aloks_order_l10_1091

/-- Given Alok's order and payment information, prove the number of mixed vegetable plates ordered -/
theorem aloks_order (chapati_count : ℕ) (rice_count : ℕ) (icecream_count : ℕ) 
  (chapati_cost : ℕ) (rice_cost : ℕ) (vegetable_cost : ℕ) (total_paid : ℕ) :
  chapati_count = 16 →
  rice_count = 5 →
  icecream_count = 6 →
  chapati_cost = 6 →
  rice_cost = 45 →
  vegetable_cost = 70 →
  total_paid = 1015 →
  ∃ (vegetable_count : ℕ), 
    total_paid = chapati_count * chapati_cost + rice_count * rice_cost + vegetable_count * vegetable_cost + 
      (total_paid - (chapati_count * chapati_cost + rice_count * rice_cost + vegetable_count * vegetable_cost)) ∧
    vegetable_count = 9 :=
by sorry

end aloks_order_l10_1091


namespace ellipse_eccentricity_l10_1084

/-- The complex equation whose roots define the points on the ellipse -/
def complex_equation (z : ℂ) : Prop :=
  (z + 1) * (z^2 + 6*z + 10) * (z^2 + 8*z + 18) = 0

/-- The set of solutions to the complex equation -/
def solution_set : Set ℂ :=
  {z : ℂ | complex_equation z}

/-- The condition that the solutions are in the form x_k + y_k*i with x_k and y_k real -/
axiom solutions_form : ∀ z ∈ solution_set, ∃ (x y : ℝ), z = x + y * Complex.I

/-- The unique ellipse passing through the points defined by the solutions -/
axiom exists_unique_ellipse : ∃! E : Set (ℝ × ℝ), 
  ∀ z ∈ solution_set, (z.re, z.im) ∈ E

/-- The eccentricity of the ellipse -/
def eccentricity (E : Set (ℝ × ℝ)) : ℝ := sorry

/-- The theorem stating that the eccentricity of the ellipse is √(3/4) -/
theorem ellipse_eccentricity : 
  ∀ E : Set (ℝ × ℝ), (∀ z ∈ solution_set, (z.re, z.im) ∈ E) → 
    eccentricity E = Real.sqrt (3/4) := 
by sorry

end ellipse_eccentricity_l10_1084


namespace polynomial_remainder_l10_1094

def P (x : ℝ) : ℝ := 5*x^3 - 12*x^2 + 6*x - 15

theorem polynomial_remainder : 
  ∃ (q : ℝ → ℝ), P = λ x => q x * (x - 3) + 30 :=
sorry

end polynomial_remainder_l10_1094


namespace factorization_of_2x_squared_minus_8_l10_1030

theorem factorization_of_2x_squared_minus_8 (x : ℝ) : 2 * x^2 - 8 = 2 * (x + 2) * (x - 2) := by
  sorry

end factorization_of_2x_squared_minus_8_l10_1030


namespace sin_double_angle_given_sin_pi_fourth_minus_x_l10_1073

theorem sin_double_angle_given_sin_pi_fourth_minus_x
  (x : ℝ) (h : Real.sin (π/4 - x) = 3/5) :
  Real.sin (2*x) = 7/25 := by
  sorry

end sin_double_angle_given_sin_pi_fourth_minus_x_l10_1073


namespace course_ratio_l10_1050

theorem course_ratio (max_courses sid_courses : ℕ) (m : ℚ) : 
  max_courses = 40 →
  max_courses + sid_courses = 200 →
  sid_courses = m * max_courses →
  m = 4 ∧ sid_courses / max_courses = 4 := by
  sorry

end course_ratio_l10_1050


namespace line_tangent_to_circle_l10_1045

/-- The equation of a circle in polar coordinates -/
def circle_equation (ρ θ : ℝ) : Prop := ρ = 4 * Real.sin θ

/-- The equation of a line in polar coordinates -/
def line_equation (ρ θ : ℝ) : Prop := ρ * Real.cos θ = 2

/-- Theorem stating that the line is tangent to the circle -/
theorem line_tangent_to_circle :
  ∃ (ρ₀ θ₀ : ℝ), circle_equation ρ₀ θ₀ ∧ line_equation ρ₀ θ₀ ∧
  ∀ (ρ θ : ℝ), circle_equation ρ θ ∧ line_equation ρ θ → (ρ, θ) = (ρ₀, θ₀) :=
sorry

end line_tangent_to_circle_l10_1045


namespace roses_unchanged_l10_1004

/-- Represents the number of flowers in a vase -/
structure FlowerVase where
  roses : ℕ
  orchids : ℕ

/-- The initial state of the flower vase -/
def initial_vase : FlowerVase := { roses := 13, orchids := 84 }

/-- The final state of the flower vase -/
def final_vase : FlowerVase := { roses := 13, orchids := 91 }

/-- Theorem stating that the number of roses remains unchanged -/
theorem roses_unchanged (initial : FlowerVase) (final : FlowerVase) 
  (h_initial : initial = initial_vase) 
  (h_final_orchids : final.orchids = 91) :
  final.roses = initial.roses := by sorry

end roses_unchanged_l10_1004


namespace cube_sum_over_product_l10_1035

theorem cube_sum_over_product (x y z : ℂ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h_sum : x + y + z = 10)
  (h_eq : (x - y)^2 + (x - z)^2 + (y - z)^2 + 6 = x*y*z) :
  (x^3 + y^3 + z^3 - 3*x*y*z) / (x*y*z) = 5 - 30 / (x*y*z) := by
sorry

end cube_sum_over_product_l10_1035


namespace unique_solution_l10_1047

-- Define the system of equations
def equation1 (x y : ℝ) : Prop :=
  x^2*y + x*y^2 + 3*x + 3*y + 24 = 0

def equation2 (x y : ℝ) : Prop :=
  x^3*y - x*y^3 + 3*x^2 - 3*y^2 - 48 = 0

-- Theorem stating that (-3, -1) is the unique solution
theorem unique_solution :
  (∃! p : ℝ × ℝ, equation1 p.1 p.2 ∧ equation2 p.1 p.2) ∧
  (equation1 (-3) (-1) ∧ equation2 (-3) (-1)) := by
  sorry

end unique_solution_l10_1047


namespace tires_in_parking_lot_parking_lot_tire_count_l10_1023

/-- The number of tires in a parking lot with four-wheel drive cars and spare tires -/
theorem tires_in_parking_lot (num_cars : ℕ) (wheels_per_car : ℕ) (has_spare : Bool) : ℕ :=
  let regular_tires := num_cars * wheels_per_car
  let spare_tires := if has_spare then num_cars else 0
  regular_tires + spare_tires

/-- Proof that there are 150 tires in the parking lot with 30 four-wheel drive cars and spare tires -/
theorem parking_lot_tire_count :
  tires_in_parking_lot 30 4 true = 150 := by
  sorry

end tires_in_parking_lot_parking_lot_tire_count_l10_1023


namespace inequality_contradiction_l10_1009

theorem inequality_contradiction (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  ¬(a + b < c + d ∧ (a + b) * c * d < a * b * (c + d) ∧ (a + b) * (c + d) < a * b + c * d) :=
by sorry

end inequality_contradiction_l10_1009


namespace max_value_inequality_l10_1042

theorem max_value_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (4 * x * z + y * z) / (x^2 + y^2 + z^2) ≤ Real.sqrt 17 / 2 := by
  sorry

end max_value_inequality_l10_1042


namespace min_reciprocal_sum_l10_1092

theorem min_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2*x + y = 1) :
  (1/x + 1/y) ≥ 3 + 2*Real.sqrt 2 := by
  sorry

end min_reciprocal_sum_l10_1092


namespace sqrt_x_div_sqrt_y_equals_five_halves_l10_1060

theorem sqrt_x_div_sqrt_y_equals_five_halves (x y : ℝ) 
  (h : (1/3)^2 + (1/4)^2 / ((1/5)^2 + (1/6)^2) = 25*x / (73*y)) : 
  Real.sqrt x / Real.sqrt y = 5/2 := by
  sorry

end sqrt_x_div_sqrt_y_equals_five_halves_l10_1060


namespace samsung_tv_cost_l10_1070

/-- The cost of a Samsung TV based on Latia's work hours and wages -/
theorem samsung_tv_cost (hourly_wage : ℕ) (weekly_hours : ℕ) (weeks : ℕ) (additional_hours : ℕ) : 
  hourly_wage = 10 →
  weekly_hours = 30 →
  weeks = 4 →
  additional_hours = 50 →
  hourly_wage * (weekly_hours * weeks + additional_hours) = 1700 := by
sorry

end samsung_tv_cost_l10_1070


namespace corresponding_sides_proportional_in_similar_triangles_l10_1049

-- Define what it means for two triangles to be similar
def similar_triangles (t1 t2 : Triangle) : Prop := sorry

-- Define what it means for sides to be corresponding
def corresponding_sides (s1 : Segment) (t1 : Triangle) (s2 : Segment) (t2 : Triangle) : Prop := sorry

-- Define what it means for two segments to be proportional
def proportional (s1 s2 : Segment) : Prop := sorry

-- Theorem statement
theorem corresponding_sides_proportional_in_similar_triangles 
  (t1 t2 : Triangle) (s1 s3 : Segment) (s2 s4 : Segment) :
  similar_triangles t1 t2 →
  corresponding_sides s1 t1 s2 t2 →
  corresponding_sides s3 t1 s4 t2 →
  proportional s1 s2 ∧ proportional s3 s4 := by sorry

end corresponding_sides_proportional_in_similar_triangles_l10_1049


namespace angle_range_from_cosine_bounds_l10_1040

theorem angle_range_from_cosine_bounds (A : Real) (h_acute : 0 < A ∧ A < Real.pi / 2) 
  (h_cos_bounds : 1 / 2 < Real.cos A ∧ Real.cos A < Real.sqrt 3 / 2) : 
  Real.pi / 6 < A ∧ A < Real.pi / 3 :=
sorry

end angle_range_from_cosine_bounds_l10_1040


namespace investment_problem_l10_1032

/-- The investment problem -/
theorem investment_problem (b_investment c_investment c_profit total_profit : ℕ) 
  (hb : b_investment = 16000)
  (hc : c_investment = 20000)
  (hcp : c_profit = 36000)
  (htp : total_profit = 86400) :
  ∃ a_investment : ℕ, 
    a_investment * total_profit = 
      (total_profit - b_investment * total_profit / (a_investment + b_investment + c_investment) - 
       c_investment * total_profit / (a_investment + b_investment + c_investment)) :=
by sorry

end investment_problem_l10_1032


namespace polynomial_with_specific_roots_l10_1019

theorem polynomial_with_specific_roots :
  ∃ (P : ℂ → ℂ) (r s : ℤ),
    (∀ x, P x = x^4 + (a : ℤ) * x^3 + (b : ℤ) * x^2 + (c : ℤ) * x + (d : ℤ)) ∧
    (P r = 0) ∧ (P s = 0) ∧ (P ((1 + Complex.I * Real.sqrt 15) / 2) = 0) :=
by sorry

end polynomial_with_specific_roots_l10_1019


namespace jimmy_has_five_figures_l10_1037

/-- Represents the collection of action figures Jimmy has --/
structure ActionFigures where
  regular : ℕ  -- number of regular figures worth $15
  special : ℕ  -- number of special figures worth $20
  h_special : special = 1  -- there is exactly one special figure

/-- The total value of the collection before the price reduction --/
def total_value (af : ActionFigures) : ℕ :=
  15 * af.regular + 20 * af.special

/-- The total earnings after selling all figures with $5 discount --/
def total_earnings (af : ActionFigures) : ℕ :=
  10 * af.regular + 15 * af.special

/-- Theorem stating that Jimmy has 5 action figures in total --/
theorem jimmy_has_five_figures :
  ∃ (af : ActionFigures), total_earnings af = 55 ∧ af.regular + af.special = 5 :=
sorry

end jimmy_has_five_figures_l10_1037


namespace journey_distance_total_distance_value_l10_1010

/-- Represents the total distance travelled by a family in a journey -/
def total_distance : ℝ := sorry

/-- The total travel time in hours -/
def total_time : ℝ := 18

/-- The speed for the first third of the journey in km/h -/
def speed1 : ℝ := 35

/-- The speed for the second third of the journey in km/h -/
def speed2 : ℝ := 40

/-- The speed for the last third of the journey in km/h -/
def speed3 : ℝ := 45

/-- Theorem stating the relationship between distance, time, and speeds -/
theorem journey_distance : 
  (total_distance / 3) / speed1 + 
  (total_distance / 3) / speed2 + 
  (total_distance / 3) / speed3 = total_time :=
by sorry

/-- Theorem stating that the total distance is approximately 712.46 km -/
theorem total_distance_value : 
  ∃ ε > 0, abs (total_distance - 712.46) < ε :=
by sorry

end journey_distance_total_distance_value_l10_1010


namespace intersection_with_complement_l10_1029

def U : Finset Nat := {0, 1, 2, 3, 4}
def A : Finset Nat := {0, 1, 2, 3}
def B : Finset Nat := {0, 2, 4}

theorem intersection_with_complement :
  A ∩ (U \ B) = {1, 3} := by sorry

end intersection_with_complement_l10_1029


namespace ball_probabilities_l10_1080

/-- Represents the number of white balls initially in the bag -/
def initial_white_balls : ℕ := 8

/-- Represents the number of red balls initially in the bag -/
def initial_red_balls : ℕ := 12

/-- Represents the total number of balls in the bag -/
def total_balls : ℕ := initial_white_balls + initial_red_balls

/-- Represents the probability of drawing a yellow ball -/
def prob_yellow : ℚ := 0

/-- Represents the probability of drawing at least one red ball out of 9 balls drawn at once -/
def prob_at_least_one_red : ℚ := 1

/-- Represents the probability of drawing a red ball at random -/
def prob_red : ℚ := 3 / 5

/-- Represents the number of red balls removed and white balls added -/
def x : ℕ := 8

theorem ball_probabilities :
  (prob_yellow = 0) ∧
  (prob_at_least_one_red = 1) ∧
  (prob_red = 3 / 5) ∧
  (((initial_white_balls + x : ℚ) / total_balls) = 4 / 5 → x = 8) :=
sorry

end ball_probabilities_l10_1080


namespace forum_posts_l10_1018

/-- Calculates the total number of questions and answers posted on a forum in a day -/
def total_posts_per_day (members : ℕ) (questions_per_hour : ℕ) (answer_ratio : ℕ) : ℕ :=
  let questions_per_day := questions_per_hour * 24
  let total_questions := members * questions_per_day
  let total_answers := members * questions_per_day * answer_ratio
  total_questions + total_answers

/-- Theorem stating the total number of posts on the forum in a day -/
theorem forum_posts :
  total_posts_per_day 500 5 4 = 300000 := by
  sorry

end forum_posts_l10_1018


namespace function_properties_l10_1046

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.log x + (1/2) * m * x^2 - 2

theorem function_properties (m : ℝ) :
  (((Real.exp 1)⁻¹ + m = -(1/2)) → m = -(3/2)) ∧
  (∀ x > 0, f m x + 2 ≤ m * x^2 + (m - 1) * x - 1) →
  m ≥ 2 :=
sorry

end function_properties_l10_1046


namespace f_inequality_l10_1006

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
variable (h1 : ∀ x ≠ 1, (x - 1) * (deriv f x) < 0)
variable (h2 : ∀ x, f (x + 1) = f (-x + 1))

-- Define the theorem
theorem f_inequality (x₁ x₂ : ℝ) (h3 : |x₁ - 1| < |x₂ - 1|) : f x₁ > f x₂ := by
  sorry

end f_inequality_l10_1006


namespace quadratic_function_and_tangent_line_l10_1039

/-- A function f is even if f(x) = f(-x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- A point x is a zero of function f if f(x) = 0 -/
def IsZero (f : ℝ → ℝ) (x : ℝ) : Prop :=
  f x = 0

/-- A line y = kx + m is tangent to the graph of f if there exists exactly one point
    where the line touches the graph of f -/
def IsTangent (f : ℝ → ℝ) (k m : ℝ) : Prop :=
  ∃! x, f x = k * x + m

theorem quadratic_function_and_tangent_line 
  (f : ℝ → ℝ) (b c k m : ℝ) 
  (h1 : ∀ x, f x = x^2 + b*x + c)
  (h2 : IsEven f)
  (h3 : IsZero f 1)
  (h4 : k > 0)
  (h5 : IsTangent f k m) :
  (∀ x, f x = x^2 - 1) ∧ 
  (∀ k m, k > 0 → IsTangent f k m → m * k ≤ -4) ∧
  (∃ k m, k > 0 ∧ IsTangent f k m ∧ m * k = -4) :=
by sorry

end quadratic_function_and_tangent_line_l10_1039


namespace negation_equivalence_l10_1054

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (Teacher : U → Prop)
variable (ExcellentAtMath : U → Prop)
variable (PoorAtMath : U → Prop)

-- Define the statements
def AllTeachersExcellent : Prop := ∀ x, Teacher x → ExcellentAtMath x
def AtLeastOneTeacherPoor : Prop := ∃ x, Teacher x ∧ PoorAtMath x

-- Theorem statement
theorem negation_equivalence : 
  AtLeastOneTeacherPoor U Teacher PoorAtMath ↔ ¬(AllTeachersExcellent U Teacher ExcellentAtMath) :=
sorry

end negation_equivalence_l10_1054


namespace plot_length_is_57_l10_1090

/-- A rectangular plot with specific fencing cost and length-breadth relationship -/
structure RectangularPlot where
  breadth : ℝ
  length : ℝ
  fencing_cost_per_meter : ℝ
  total_fencing_cost : ℝ
  length_breadth_relation : length = breadth + 14
  fencing_cost_equation : total_fencing_cost = fencing_cost_per_meter * (2 * length + 2 * breadth)

/-- The length of the rectangular plot is 57 meters -/
theorem plot_length_is_57 (plot : RectangularPlot) 
    (h1 : plot.fencing_cost_per_meter = 26.5)
    (h2 : plot.total_fencing_cost = 5300) : 
  plot.length = 57 := by
  sorry


end plot_length_is_57_l10_1090


namespace regular_polygon_sides_l10_1078

theorem regular_polygon_sides (b : ℕ) (h : b ≥ 3) : (180 * (b - 2) = 1080) → b = 8 := by
  sorry

end regular_polygon_sides_l10_1078


namespace spinster_cat_ratio_l10_1093

theorem spinster_cat_ratio : 
  ∀ (x : ℚ), 
    (22 : ℚ) / x = 7 →  -- ratio of spinsters to cats is x:7
    x = 22 + 55 →      -- there are 55 more cats than spinsters
    (2 : ℚ) / 7 = 22 / x -- the ratio of spinsters to cats is 2:7
  := by sorry

end spinster_cat_ratio_l10_1093


namespace stones_sent_away_l10_1072

theorem stones_sent_away (original_stones kept_stones : ℕ) 
  (h1 : original_stones = 78) 
  (h2 : kept_stones = 15) : 
  original_stones - kept_stones = 63 := by
  sorry

end stones_sent_away_l10_1072


namespace chris_winning_configurations_l10_1059

/-- Modified nim-value for a single wall in the brick removal game -/
def modified_nim_value (n : ℕ) : ℕ := sorry

/-- Nim-sum of a list of natural numbers -/
def nim_sum (l : List ℕ) : ℕ := sorry

/-- Represents a game configuration as a list of wall sizes -/
def GameConfig := List ℕ

/-- Determines if Chris (second player) can guarantee a win for a given game configuration -/
def chris_wins (config : GameConfig) : Prop :=
  nim_sum (config.map modified_nim_value) = 0

theorem chris_winning_configurations :
  ∀ config : GameConfig,
    (chris_wins config ↔ 
      (config = [7, 5, 2] ∨ config = [7, 5, 3])) :=
by sorry

end chris_winning_configurations_l10_1059


namespace coin_toss_and_match_probability_l10_1022

/-- Represents the outcome of a coin toss -/
inductive CoinToss
| Head
| Tail

/-- Represents the weather condition during a match -/
inductive Weather
| Rainy
| NotRainy

/-- Represents the result of a match -/
inductive MatchResult
| Draw
| NotDraw

/-- Represents a football match with its associated coin toss, weather, and result -/
structure Match where
  toss : CoinToss
  weather : Weather
  result : MatchResult

def coin_tosses : ℕ := 25
def heads_count : ℕ := 11
def draw_on_heads : ℕ := 7
def rainy_on_tails : ℕ := 4

/-- The main theorem to prove -/
theorem coin_toss_and_match_probability :
  (coin_tosses - heads_count = 14) ∧
  (∀ m : Match, m.toss = CoinToss.Head → m.result = MatchResult.Draw → 
               m.toss = CoinToss.Tail → m.weather = Weather.Rainy → False) :=
by sorry

end coin_toss_and_match_probability_l10_1022


namespace system_of_equations_solutions_l10_1027

theorem system_of_equations_solutions :
  -- First system
  (∃ x y : ℝ, 2*x + 3*y = 7 ∧ x = -2*y + 3 ∧ x = 5 ∧ y = -1) ∧
  -- Second system
  (∃ x y : ℝ, 5*x + y = 4 ∧ 2*x - 3*y - 5 = 0 ∧ x = 1 ∧ y = -1) :=
by
  sorry

end system_of_equations_solutions_l10_1027


namespace program_output_equals_b_l10_1076

def program (a b : ℕ) : ℕ :=
  if a > b then a else b

theorem program_output_equals_b :
  let a : ℕ := 2
  let b : ℕ := 3
  program a b = b := by sorry

end program_output_equals_b_l10_1076


namespace discriminant_divisibility_l10_1008

theorem discriminant_divisibility (a b : ℝ) (n : ℤ) : 
  (∃ x₁ x₂ : ℝ, (2018 * x₁^2 + a * x₁ + b = 0) ∧ 
                (2018 * x₂^2 + a * x₂ + b = 0) ∧ 
                (x₁ - x₂ = n)) → 
  ∃ k : ℤ, a^2 - 4 * 2018 * b = 2018^2 * k := by
sorry

end discriminant_divisibility_l10_1008


namespace decimal_digits_of_fraction_l10_1077

theorem decimal_digits_of_fraction : ∃ (n : ℚ), 
  n = (5^7 : ℚ) / ((10^5 : ℚ) * 125) ∧ 
  ∃ (d : ℕ), d = 4 ∧ 
  (∃ (m : ℕ), n = (m : ℚ) / (10^d : ℚ) ∧ 
   m % 10 ≠ 0 ∧ 
   (∀ (k : ℕ), k > d → (m * 10^(k-d)) % 10 = 0)) :=
by sorry

end decimal_digits_of_fraction_l10_1077


namespace hyperbola_slope_product_l10_1001

/-- The product of slopes for a hyperbola -/
theorem hyperbola_slope_product (a b c : ℝ) (P Q : ℝ × ℝ) :
  a > 0 →
  b > 0 →
  b^2 = a * c →
  ((P.1^2 / a^2) - (P.2^2 / b^2) = 1) →
  ((Q.1^2 / a^2) - (Q.2^2 / b^2) = 1) →
  let M := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)
  let k_PQ := (Q.2 - P.2) / (Q.1 - P.1)
  let k_OM := M.2 / M.1
  k_PQ ≠ 0 →
  M.1 ≠ 0 →
  k_PQ * k_OM = (1 + Real.sqrt 5) / 2 := by
  sorry

end hyperbola_slope_product_l10_1001


namespace mallory_journey_expenses_l10_1052

theorem mallory_journey_expenses :
  let initial_fuel_cost : ℚ := 45
  let miles_per_tank : ℚ := 500
  let total_miles : ℚ := 2000
  let food_cost_ratio : ℚ := 3/5
  let hotel_nights : ℕ := 3
  let hotel_cost_per_night : ℚ := 80
  let fuel_cost_increase : ℚ := 5

  let num_refills : ℕ := (total_miles / miles_per_tank).ceil.toNat
  let fuel_costs : List ℚ := List.range num_refills |>.map (λ i => initial_fuel_cost + i * fuel_cost_increase)
  let total_fuel_cost : ℚ := fuel_costs.sum
  let food_cost : ℚ := food_cost_ratio * total_fuel_cost
  let hotel_cost : ℚ := hotel_nights * hotel_cost_per_night
  let total_expenses : ℚ := total_fuel_cost + food_cost + hotel_cost

  total_expenses = 576
  := by sorry

end mallory_journey_expenses_l10_1052


namespace balloons_left_l10_1086

theorem balloons_left (round_bags : ℕ) (round_per_bag : ℕ) (long_bags : ℕ) (long_per_bag : ℕ) (burst : ℕ) : 
  round_bags = 5 → 
  round_per_bag = 20 → 
  long_bags = 4 → 
  long_per_bag = 30 → 
  burst = 5 → 
  round_bags * round_per_bag + long_bags * long_per_bag - burst = 215 := by
  sorry

end balloons_left_l10_1086


namespace streetlight_purchase_l10_1085

theorem streetlight_purchase (squares : Nat) (lights_per_square : Nat) (repair_lights : Nat) (bought_lights : Nat) : 
  squares = 15 → 
  lights_per_square = 12 → 
  repair_lights = 35 → 
  bought_lights = 200 → 
  squares * lights_per_square + repair_lights - bought_lights = 15 := by
  sorry

end streetlight_purchase_l10_1085


namespace eleven_only_divisor_l10_1066

theorem eleven_only_divisor : ∃! n : ℕ, 
  (∃ k : ℕ, n = (10^k - 1) / 9) ∧ 
  (∃ m : ℕ, (10^m + 1) % n = 0) ∧
  n = 11 := by
sorry

end eleven_only_divisor_l10_1066


namespace largest_divisible_n_l10_1002

theorem largest_divisible_n : ∃ (n : ℕ), 
  (∀ (m : ℕ), m > n → ¬((m + 20) ∣ (m^3 - 100))) ∧ 
  ((n + 20) ∣ (n^3 - 100)) ∧ 
  n = 2080 := by
  sorry

end largest_divisible_n_l10_1002


namespace sample_size_equals_sampled_students_l10_1079

/-- Represents a survey conducted on eighth-grade students -/
structure Survey where
  sampled_students : ℕ

/-- The sample size of a survey is equal to the number of sampled students -/
theorem sample_size_equals_sampled_students (s : Survey) : s.sampled_students = 1500 → s.sampled_students = 1500 := by
  sorry

end sample_size_equals_sampled_students_l10_1079


namespace parabolas_intersection_l10_1063

/-- Parabola 1 equation -/
def parabola1 (x : ℝ) : ℝ := 4 * x^2 + 3 * x - 7

/-- Parabola 2 equation -/
def parabola2 (x : ℝ) : ℝ := 2 * x^2 + 5

/-- The intersection points of the two parabolas -/
def intersection_points : Set (ℝ × ℝ) := {(-4, 37), (3/2, 9.5)}

theorem parabolas_intersection :
  ∀ p : ℝ × ℝ, (parabola1 p.1 = parabola2 p.1) ↔ p ∈ intersection_points :=
by sorry

end parabolas_intersection_l10_1063

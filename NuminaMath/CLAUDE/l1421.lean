import Mathlib

namespace unique_positive_solution_l1421_142194

theorem unique_positive_solution :
  ∃! (x : ℝ), x > 0 ∧ (x - 6) / 12 = 6 / (x - 12) ∧ x = 18 := by
  sorry

end unique_positive_solution_l1421_142194


namespace sqrt_1_0201_l1421_142129

theorem sqrt_1_0201 (h : Real.sqrt 102.01 = 10.1) : Real.sqrt 1.0201 = 1.01 := by
  sorry

end sqrt_1_0201_l1421_142129


namespace diagonal_passes_through_600_cubes_l1421_142175

/-- The number of cubes an internal diagonal passes through in a rectangular solid -/
def cubes_passed_by_diagonal (a b c : ℕ) : ℕ :=
  a + b + c - (Nat.gcd a b + Nat.gcd b c + Nat.gcd c a) + Nat.gcd a (Nat.gcd b c)

/-- Theorem: For a 120 × 270 × 300 rectangular solid, the internal diagonal passes through 600 cubes -/
theorem diagonal_passes_through_600_cubes :
  cubes_passed_by_diagonal 120 270 300 = 600 := by
  sorry

end diagonal_passes_through_600_cubes_l1421_142175


namespace inequality_proof_l1421_142146

theorem inequality_proof (x₁ x₂ x₃ x₄ : ℝ) 
  (h1 : x₁ ≥ x₂) (h2 : x₂ ≥ x₃) (h3 : x₃ ≥ x₄)
  (h4 : x₂ + x₃ + x₄ ≥ x₁) : 
  (x₁ + x₂ + x₃ + x₄)^2 ≤ 4 * x₁ * x₂ * x₃ * x₄ := by
  sorry

end inequality_proof_l1421_142146


namespace pizza_toppings_combinations_l1421_142133

/-- The number of combinations of k items chosen from a set of n items -/
def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- The number of available pizza toppings -/
def n : ℕ := 7

/-- The number of toppings to be chosen -/
def k : ℕ := 3

/-- Theorem: The number of combinations of 3 toppings chosen from 7 available toppings is 35 -/
theorem pizza_toppings_combinations : binomial n k = 35 := by
  sorry

end pizza_toppings_combinations_l1421_142133


namespace inscribed_square_area_l1421_142100

/-- The parabola function y = x^2 - 10x + 21 --/
def parabola (x : ℝ) : ℝ := x^2 - 10*x + 21

/-- A square inscribed in the region bound by the parabola and the x-axis --/
structure InscribedSquare where
  center : ℝ  -- x-coordinate of the square's center
  side : ℝ    -- length of the square's side
  h1 : center - side/2 ≥ 0  -- Left side of square is non-negative
  h2 : center + side/2 ≤ 10 -- Right side of square is at most the x-intercept
  h3 : parabola (center - side/2) = 0  -- Left bottom corner on x-axis
  h4 : parabola (center + side/2) = 0  -- Right bottom corner on x-axis
  h5 : parabola center = side          -- Top of square touches parabola

/-- The theorem stating the area of the inscribed square --/
theorem inscribed_square_area (s : InscribedSquare) :
  s.side^2 = 24 - 8*Real.sqrt 5 :=
sorry

end inscribed_square_area_l1421_142100


namespace smallest_norm_v_l1421_142118

theorem smallest_norm_v (v : ℝ × ℝ) (h : ‖v + (4, 2)‖ = 10) :
  ∃ (w : ℝ × ℝ), ‖w + (4, 2)‖ = 10 ∧ ∀ (u : ℝ × ℝ), ‖u + (4, 2)‖ = 10 → ‖w‖ ≤ ‖u‖ ∧ ‖w‖ = 10 - 2 * Real.sqrt 5 := by
  sorry

end smallest_norm_v_l1421_142118


namespace rectangle_area_18_l1421_142126

def rectangle_pairs : Set (Nat × Nat) :=
  {(1, 18), (2, 9), (3, 6), (6, 3), (9, 2), (18, 1)}

theorem rectangle_area_18 :
  ∀ (w l : Nat), w > 0 ∧ l > 0 →
  (w * l = 18 ↔ (w, l) ∈ rectangle_pairs) :=
by sorry

end rectangle_area_18_l1421_142126


namespace square_remainder_sum_quotient_l1421_142110

theorem square_remainder_sum_quotient : 
  let squares := List.map (fun n => n^2) (List.range 6)
  let remainders := List.map (fun x => x % 13) squares
  let distinct_remainders := List.eraseDups remainders
  let m := distinct_remainders.sum
  m / 13 = 3 := by
sorry

end square_remainder_sum_quotient_l1421_142110


namespace line_ellipse_intersection_slopes_l1421_142174

/-- Given a line y = mx + 3 intersecting the ellipse 4x^2 + 25y^2 = 100,
    prove that the possible slopes m satisfy m^2 ≥ 1/55. -/
theorem line_ellipse_intersection_slopes (m : ℝ) : 
  (∃ x y : ℝ, 4 * x^2 + 25 * y^2 = 100 ∧ y = m * x + 3) → m^2 ≥ 1/55 := by
  sorry

end line_ellipse_intersection_slopes_l1421_142174


namespace right_triangle_area_l1421_142108

theorem right_triangle_area (hypotenuse : ℝ) (angle : ℝ) :
  hypotenuse = 10 →
  angle = 30 * π / 180 →
  let shorter_leg := hypotenuse / 2
  let longer_leg := shorter_leg * Real.sqrt 3
  let area := (shorter_leg * longer_leg) / 2
  area = (25 * Real.sqrt 3) / 2 := by
  sorry

end right_triangle_area_l1421_142108


namespace f_nonnegative_range_l1421_142107

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x * (Real.exp x - a) - a^2 * x

theorem f_nonnegative_range (a : ℝ) :
  (∀ x, f a x ≥ 0) ↔ a ∈ Set.Icc (-2 * Real.exp (3/4)) 1 :=
by sorry

end f_nonnegative_range_l1421_142107


namespace wednesday_kids_count_l1421_142158

def monday_kids : ℕ := 17
def tuesday_kids : ℕ := 15
def total_kids : ℕ := 34

theorem wednesday_kids_count : total_kids - monday_kids - tuesday_kids = 2 := by
  sorry

end wednesday_kids_count_l1421_142158


namespace arithmetic_sqrt_of_nine_l1421_142115

theorem arithmetic_sqrt_of_nine (x : ℝ) : x ≥ 0 ∧ x ^ 2 = 9 → x = 3 := by sorry

end arithmetic_sqrt_of_nine_l1421_142115


namespace magic_deck_problem_l1421_142185

/-- Given a magician selling magic card decks, this theorem proves
    the number of decks left unsold at the end of the day. -/
theorem magic_deck_problem (initial_decks : ℕ) (price_per_deck : ℕ) (total_earnings : ℕ) :
  initial_decks = 16 →
  price_per_deck = 7 →
  total_earnings = 56 →
  initial_decks - (total_earnings / price_per_deck) = 8 := by
  sorry

end magic_deck_problem_l1421_142185


namespace consecutive_integers_product_sum_l1421_142123

theorem consecutive_integers_product_sum (n : ℕ) : 
  n > 0 ∧ n * (n + 1) = 2720 → n + (n + 1) = 103 := by
  sorry

end consecutive_integers_product_sum_l1421_142123


namespace max_duck_moves_l1421_142178

/-- 
Given positive integers a, b, and c representing the number of ducks 
picking rock, paper, and scissors respectively in a circular arrangement, 
the maximum number of possible moves according to the rock-paper-scissors 
switching rules is max(a × b, b × c, c × a).
-/
theorem max_duck_moves (a b c : ℕ+) : 
  ∃ (max_moves : ℕ), max_moves = max (a * b) (max (b * c) (c * a)) ∧
  ∀ (moves : ℕ), moves ≤ max_moves := by
sorry


end max_duck_moves_l1421_142178


namespace tangent_lines_to_circle_l1421_142140

/-- A line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A circle in 2D space represented by (x - h)^2 + (y - k)^2 = r^2 -/
structure Circle where
  h : ℝ
  k : ℝ
  r : ℝ

/-- Check if a point (x, y) is on a line -/
def Line.contains (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y + l.c = 0

/-- Check if a line is tangent to a circle -/
def Line.isTangentTo (l : Line) (c : Circle) : Prop :=
  (c.h - l.a * (c.h * l.a + c.k * l.b - l.c) / (l.a^2 + l.b^2))^2 +
  (c.k - l.b * (c.h * l.a + c.k * l.b - l.c) / (l.a^2 + l.b^2))^2 = c.r^2

theorem tangent_lines_to_circle (c : Circle) :
  c.h = 1 ∧ c.k = 0 ∧ c.r = 2 →
  ∃ (l₁ l₂ : Line),
    (l₁.a = 3 ∧ l₁.b = 4 ∧ l₁.c = -13) ∧
    (l₂.a = 1 ∧ l₂.b = 0 ∧ l₂.c = -3) ∧
    l₁.contains 3 1 ∧
    l₂.contains 3 1 ∧
    l₁.isTangentTo c ∧
    l₂.isTangentTo c ∧
    ∀ (l : Line), l.contains 3 1 ∧ l.isTangentTo c → l = l₁ ∨ l = l₂ :=
by sorry

end tangent_lines_to_circle_l1421_142140


namespace circle_center_sum_l1421_142198

/-- Given a circle with equation x^2 + y^2 = 4x - 6y + 9, 
    the sum of the x and y coordinates of its center is -1. -/
theorem circle_center_sum (x y : ℝ) : 
  x^2 + y^2 = 4*x - 6*y + 9 → 
  ∃ (h k : ℝ), (∀ (a b : ℝ), (a - h)^2 + (b - k)^2 = (x - h)^2 + (y - k)^2) ∧ h + k = -1 := by
  sorry

end circle_center_sum_l1421_142198


namespace star_product_six_equals_twentyfour_l1421_142109

/-- Custom operation definition -/
def star (x y : ℝ) : ℝ := (x + y)^2 - (x - y)^2

/-- Theorem stating that if a * b = 6, then a ¤ b = 24 -/
theorem star_product_six_equals_twentyfour (a b : ℝ) (h : a * b = 6) : star a b = 24 := by
  sorry

end star_product_six_equals_twentyfour_l1421_142109


namespace gift_price_proof_l1421_142128

def gift_price_calculation (lisa_savings : ℝ) (mother_fraction : ℝ) (brother_multiplier : ℝ) (price_difference : ℝ) : Prop :=
  let mother_contribution := mother_fraction * lisa_savings
  let brother_contribution := brother_multiplier * mother_contribution
  let total_amount := lisa_savings + mother_contribution + brother_contribution
  let gift_price := total_amount + price_difference
  gift_price = 3760

theorem gift_price_proof :
  gift_price_calculation 1200 (3/5) 2 400 := by
  sorry

end gift_price_proof_l1421_142128


namespace median_to_longest_side_l1421_142142

/-- Given a triangle with side lengths 10, 24, and 26, the length of the median to the longest side is 13. -/
theorem median_to_longest_side (a b c : ℝ) (h1 : a = 10) (h2 : b = 24) (h3 : c = 26) :
  let m := (1/2) * Real.sqrt (2 * a^2 + 2 * b^2 - c^2)
  m = 13 := by sorry

end median_to_longest_side_l1421_142142


namespace researcher_can_reach_oasis_l1421_142169

/-- Represents a traveler in the desert -/
structure Traveler where
  food : ℕ
  position : ℕ

/-- Represents the state of the journey -/
structure JourneyState where
  researcher : Traveler
  porters : List Traveler
  day : ℕ

def oasisDistance : ℕ := 380
def dailyTravel : ℕ := 60
def maxFood : ℕ := 4

def canReachOasis (initialState : JourneyState) : Prop :=
  ∃ (finalState : JourneyState),
    finalState.researcher.position = oasisDistance ∧
    finalState.day ≤ initialState.researcher.food * maxFood ∧
    ∀ porter ∈ finalState.porters, porter.position = 0

theorem researcher_can_reach_oasis :
  ∃ (initialState : JourneyState),
    initialState.researcher.food = maxFood ∧
    initialState.researcher.position = 0 ∧
    initialState.porters.length = 2 ∧
    (∀ porter ∈ initialState.porters, porter.food = maxFood ∧ porter.position = 0) ∧
    initialState.day = 0 ∧
    canReachOasis initialState :=
  sorry

end researcher_can_reach_oasis_l1421_142169


namespace f_properties_l1421_142197

/-- A function with a local minimum at x = 1 -/
def f (a b x : ℝ) : ℝ := x^3 - 3*a*x^2 + 2*b*x

/-- The function has a local minimum of -1 at x = 1 -/
def has_local_min (a b : ℝ) : Prop :=
  ∃ ε > 0, ∀ x, |x - 1| < ε → f a b x ≥ f a b 1 ∧ f a b 1 = -1

/-- The range of f on [0,2] -/
def range_f (a b : ℝ) : Set ℝ :=
  {y | ∃ x ∈ Set.Icc 0 2, f a b x = y}

theorem f_properties :
  ∃ a b : ℝ, has_local_min a b ∧ a = 1/3 ∧ b = -1/2 ∧ range_f a b = Set.Icc (-1) 2 :=
sorry

end f_properties_l1421_142197


namespace not_divisible_by_5_and_9_l1421_142193

def count_not_divisible (n : ℕ) (a b : ℕ) : ℕ :=
  n - (n / a + n / b - n / (a * b))

theorem not_divisible_by_5_and_9 :
  count_not_divisible 1199 5 9 = 853 := by
  sorry

end not_divisible_by_5_and_9_l1421_142193


namespace geometric_sequence_decreasing_l1421_142151

def geometric_sequence (n : ℕ) : ℝ := 4 * (3 ^ (1 - n))

theorem geometric_sequence_decreasing :
  ∀ n : ℕ, geometric_sequence (n + 1) < geometric_sequence n :=
by
  sorry

end geometric_sequence_decreasing_l1421_142151


namespace perpendicular_vectors_k_value_l1421_142173

/-- Given two vectors in ℝ², prove that if k * a + b is perpendicular to a - 3 * b, then k = 19 -/
theorem perpendicular_vectors_k_value (a b : ℝ × ℝ) (k : ℝ) 
    (h1 : a = (1, 2))
    (h2 : b = (-3, 2))
    (h3 : (k * a.1 + b.1, k * a.2 + b.2) • (a.1 - 3 * b.1, a.2 - 3 * b.2) = 0) :
  k = 19 := by
  sorry

end perpendicular_vectors_k_value_l1421_142173


namespace horner_method_f_2_l1421_142124

def f (x : ℝ) : ℝ := 2*x^5 - 3*x^3 + 2*x^2 + x - 3

def horner_v3 (f : ℝ → ℝ) (x : ℝ) : ℝ :=
  let v0 := 2
  let v1 := v0 * x
  let v2 := (v1 - 3) * x + 2
  (v2 * x + 1) * x - 3

theorem horner_method_f_2 :
  horner_v3 f 2 = 12 := by sorry

end horner_method_f_2_l1421_142124


namespace negative_two_less_than_negative_three_halves_l1421_142132

theorem negative_two_less_than_negative_three_halves : -2 < -3/2 := by
  sorry

end negative_two_less_than_negative_three_halves_l1421_142132


namespace agricultural_equipment_problem_l1421_142161

theorem agricultural_equipment_problem 
  (cost_2A_1B : ℝ) 
  (cost_1A_3B : ℝ) 
  (total_budget : ℝ) :
  cost_2A_1B = 4.2 →
  cost_1A_3B = 5.1 →
  total_budget = 10 →
  ∃ (cost_A cost_B : ℝ) (max_units_A : ℕ),
    cost_A = 1.5 ∧
    cost_B = 1.2 ∧
    max_units_A = 3 ∧
    2 * cost_A + cost_B = cost_2A_1B ∧
    cost_A + 3 * cost_B = cost_1A_3B ∧
    (∀ m : ℕ, m * cost_A + (2 * m - 3) * cost_B ≤ total_budget → m ≤ max_units_A) :=
by sorry

end agricultural_equipment_problem_l1421_142161


namespace quartic_roots_product_l1421_142163

theorem quartic_roots_product (x y z w : ℝ) 
  (sum_zero : x + y + z + w = 0)
  (sum_seventh_power_zero : x^7 + y^7 + z^7 + w^7 = 0) :
  w * (w + x) * (w + y) * (w + z) = 0 := by
  sorry

end quartic_roots_product_l1421_142163


namespace purple_book_pages_purple_book_pages_proof_l1421_142121

theorem purple_book_pages : ℕ → Prop :=
  fun p =>
    let orange_pages : ℕ := 510
    let purple_books_read : ℕ := 5
    let orange_books_read : ℕ := 4
    let page_difference : ℕ := 890
    orange_books_read * orange_pages - purple_books_read * p = page_difference →
    p = 230

-- The proof goes here
theorem purple_book_pages_proof : purple_book_pages 230 := by
  sorry

end purple_book_pages_purple_book_pages_proof_l1421_142121


namespace sequence_not_periodic_l1421_142184

/-- The sequence (a_n) defined by a_n = ⌊x^(n+1)⌋ - x⌊x^n⌋ is not periodic for any real x > 1 that is not an integer. -/
theorem sequence_not_periodic (x : ℝ) (hx : x > 1) (hx_not_int : ¬ ∃ n : ℤ, x = n) :
  ¬ ∃ p : ℕ, p > 0 ∧ ∀ n : ℕ, 
    (⌊x^(n+1)⌋ - x * ⌊x^n⌋ : ℝ) = (⌊x^(n+p+1)⌋ - x * ⌊x^(n+p)⌋ : ℝ) :=
by sorry

end sequence_not_periodic_l1421_142184


namespace ellipse_hyperbola_foci_l1421_142180

/-- Given an ellipse and a hyperbola with coinciding foci, prove that b^2 = 75/4 for the ellipse -/
theorem ellipse_hyperbola_foci (b : ℝ) : 
  (∀ x y : ℝ, x^2/25 + y^2/b^2 = 1 → x^2/64 - y^2/36 = 1/16) →
  (∃ c : ℝ, c^2 = 25 - b^2 ∧ c^2 = 64 - 36) →
  b^2 = 75/4 := by
sorry

end ellipse_hyperbola_foci_l1421_142180


namespace five_ruble_coins_l1421_142147

theorem five_ruble_coins (total_coins : ℕ) 
  (not_two_ruble : ℕ) (not_ten_ruble : ℕ) (not_one_ruble : ℕ) :
  total_coins = 25 →
  not_two_ruble = 19 →
  not_ten_ruble = 20 →
  not_one_ruble = 16 →
  total_coins - (total_coins - not_two_ruble + total_coins - not_ten_ruble + total_coins - not_one_ruble) = 5 :=
by sorry

end five_ruble_coins_l1421_142147


namespace election_win_percentage_l1421_142159

/-- The required percentage to win an election --/
def required_percentage_to_win (total_votes : ℕ) (candidate_votes : ℕ) (additional_votes_needed : ℕ) : ℚ :=
  (candidate_votes + additional_votes_needed : ℚ) / total_votes * 100

/-- Theorem stating the required percentage to win the election --/
theorem election_win_percentage 
  (total_votes : ℕ) 
  (candidate_votes : ℕ) 
  (additional_votes_needed : ℕ) 
  (h1 : total_votes = 6000)
  (h2 : candidate_votes = total_votes / 100)
  (h3 : additional_votes_needed = 3000) :
  required_percentage_to_win total_votes candidate_votes additional_votes_needed = 51 := by
sorry

#eval required_percentage_to_win 6000 60 3000

end election_win_percentage_l1421_142159


namespace inequality_equivalence_l1421_142165

/-- The inequality holds for all positive q if and only if p is in the interval [0, 2) -/
theorem inequality_equivalence (p : ℝ) : 
  (∀ q : ℝ, q > 0 → (4 * (p * q^2 + 2 * p^2 * q + 4 * q^2 + 5 * p * q)) / (p + q) > 3 * p^2 * q) ↔ 
  (0 ≤ p ∧ p < 2) := by sorry

end inequality_equivalence_l1421_142165


namespace cosine_function_triangle_constraint_l1421_142179

open Real

theorem cosine_function_triangle_constraint (ω : ℝ) : 
  ω > 0 →
  let f : ℝ → ℝ := λ x => cos (ω * x)
  let A : ℝ × ℝ := (2 * π / ω, 1)
  let B : ℝ × ℝ := (π / ω, -1)
  let O : ℝ × ℝ := (0, 0)
  (∀ x > 0, x < 2 * π / ω → f x ≤ 1) →
  (∀ x > 0, x < π / ω → f x ≥ -1) →
  (A.1 - O.1) * (B.1 - O.1) + (A.2 - O.2) * (B.2 - O.2) > 0 →
  (O.1 - A.1) * (B.1 - A.1) + (O.2 - A.2) * (B.2 - A.2) > 0 →
  (O.1 - B.1) * (A.1 - B.1) + (O.2 - B.2) * (A.2 - B.2) > 0 →
  sqrt 2 * π / 2 < ω ∧ ω < sqrt 2 * π :=
by sorry

end cosine_function_triangle_constraint_l1421_142179


namespace total_laundry_loads_l1421_142127

/-- The number of families sharing the vacation rental -/
def num_families : ℕ := 7

/-- The number of days of the vacation -/
def num_days : ℕ := 12

/-- The number of adults in each family -/
def adults_per_family : ℕ := 2

/-- The number of children in each family -/
def children_per_family : ℕ := 4

/-- The number of towels used by each adult per day -/
def towels_per_adult : ℕ := 2

/-- The number of towels used by each child per day -/
def towels_per_child : ℕ := 1

/-- The washing machine capacity for the first half of the vacation -/
def machine_capacity_first_half : ℕ := 8

/-- The washing machine capacity for the second half of the vacation -/
def machine_capacity_second_half : ℕ := 6

/-- The number of days in each half of the vacation -/
def days_per_half : ℕ := 6

/-- Theorem stating that the total number of loads of laundry is 98 -/
theorem total_laundry_loads : 
  let towels_per_family := adults_per_family * towels_per_adult + children_per_family * towels_per_child
  let total_towels_per_day := num_families * towels_per_family
  let total_towels := total_towels_per_day * num_days
  let loads_first_half := (total_towels_per_day * days_per_half) / machine_capacity_first_half
  let loads_second_half := (total_towels_per_day * days_per_half) / machine_capacity_second_half
  loads_first_half + loads_second_half = 98 := by
  sorry

end total_laundry_loads_l1421_142127


namespace a3_value_geometric_sequence_max_sum_value_l1421_142181

-- Define the sequence type
def Sequence := ℕ → ℝ

-- Define the conditions for the sequence
def SequenceConditions (a : Sequence) : Prop :=
  (∀ n ≥ 2, a n ≥ 0) ∧
  (∀ n ≥ 2, (2 * a n = a (n+1) + a (n-1)) ∨ (2 * a (n+1) = a n + a (n-1)))

-- Theorem 1
theorem a3_value (a : Sequence) (h : SequenceConditions a) :
  a 1 = 5 ∧ a 2 = 3 ∧ a 4 = 2 → a 3 = 1 :=
sorry

-- Theorem 2
theorem geometric_sequence (a : Sequence) (h : SequenceConditions a) :
  a 1 = 0 ∧ a 4 = 0 ∧ a 7 = 0 ∧ a 2 > 0 ∧ a 5 > 0 ∧ a 8 > 0 →
  ∃ q : ℝ, q = 1/4 ∧ a 5 = a 2 * q ∧ a 8 = a 5 * q :=
sorry

-- Theorem 3
theorem max_sum_value (a : Sequence) (h : SequenceConditions a) :
  a 1 = 1 ∧ a 2 = 2 ∧
  (∃ r s t : ℕ, 2 < r ∧ r < s ∧ s < t ∧ a r = 0 ∧ a s = 0 ∧ a t = 0 ∧
    (∀ n : ℕ, n ≠ r ∧ n ≠ s ∧ n ≠ t → a n ≠ 0)) →
  (∀ r s t : ℕ, 2 < r ∧ r < s ∧ s < t ∧ a r = 0 ∧ a s = 0 ∧ a t = 0 →
    a (r+1) + a (s+1) + a (t+1) ≤ 21/64) :=
sorry

end a3_value_geometric_sequence_max_sum_value_l1421_142181


namespace cost_price_calculation_l1421_142122

/-- Given a sale price including tax, sales tax rate, and profit rate,
    calculate the approximate cost price of an article. -/
theorem cost_price_calculation (sale_price_with_tax : ℝ)
                                (sales_tax_rate : ℝ)
                                (profit_rate : ℝ)
                                (h1 : sale_price_with_tax = 616)
                                (h2 : sales_tax_rate = 0.1)
                                (h3 : profit_rate = 0.17) :
  ∃ (cost_price : ℝ), 
    (cost_price * (1 + profit_rate) * (1 + sales_tax_rate) = sale_price_with_tax) ∧
    (abs (cost_price - 478.77) < 0.01) := by
  sorry

end cost_price_calculation_l1421_142122


namespace no_positive_a_satisfies_inequality_l1421_142138

theorem no_positive_a_satisfies_inequality :
  ∀ a : ℝ, a > 0 → ∃ x : ℝ, |Real.cos x| + |Real.cos (a * x)| ≤ Real.sin x + Real.sin (a * x) := by
  sorry

end no_positive_a_satisfies_inequality_l1421_142138


namespace library_book_count_l1421_142135

/-- The number of shelves in the library -/
def num_shelves : ℕ := 14240

/-- The number of books on each shelf -/
def books_per_shelf : ℕ := 8

/-- The total number of books in the library -/
def total_books : ℕ := num_shelves * books_per_shelf

theorem library_book_count : total_books = 113920 := by
  sorry

end library_book_count_l1421_142135


namespace rotated_rectangle_height_l1421_142144

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- The configuration of three rectangles with the middle one rotated -/
structure RectangleConfiguration where
  left : Rectangle
  middle : Rectangle
  right : Rectangle
  rotated : Bool

/-- Calculate the height of the top vertex of the middle rectangle when rotated -/
def heightOfRotatedMiddle (config : RectangleConfiguration) : ℝ :=
  if config.rotated then config.middle.width else config.middle.height

/-- The main theorem stating that the height of the rotated middle rectangle is 2 inches -/
theorem rotated_rectangle_height
  (config : RectangleConfiguration)
  (h1 : config.left.width = 2 ∧ config.left.height = 1)
  (h2 : config.middle.width = 2 ∧ config.middle.height = 1)
  (h3 : config.right.width = 2 ∧ config.right.height = 1)
  (h4 : config.rotated = true) :
  heightOfRotatedMiddle config = 2 := by
  sorry

end rotated_rectangle_height_l1421_142144


namespace circle_equation_from_parabola_intersection_l1421_142111

/-- Parabola type -/
structure Parabola where
  equation : ℝ → ℝ → Prop

/-- Circle type -/
structure Circle where
  center : ℝ × ℝ
  equation : ℝ → ℝ → Prop

/-- Theorem: Given a parabola and a circle with specific properties, 
    prove the equation of the circle -/
theorem circle_equation_from_parabola_intersection 
  (C₁ : Parabola) 
  (C₂ : Circle) 
  (F : ℝ × ℝ) 
  (A B C D : ℝ × ℝ) :
  C₁.equation = fun x y ↦ x^2 = 2*y →  -- Parabola equation
  C₂.center = F →                      -- Circle center at focus
  C₂.equation A.1 A.2 →                -- Circle intersects parabola at A
  C₂.equation B.1 B.2 →                -- Circle intersects parabola at B
  C₂.equation C.1 C.2 →                -- Circle intersects directrix at C
  C₂.equation D.1 D.2 →                -- Circle intersects directrix at D
  (A.1 - C.1)^2 + (A.2 - C.2)^2 = (B.1 - D.1)^2 + (B.2 - D.2)^2 →  -- ABCD is rectangle
  C₂.equation = fun x y ↦ x^2 + (y - 1/2)^2 = 4 :=  -- Conclusion: Circle equation
by sorry

end circle_equation_from_parabola_intersection_l1421_142111


namespace intersection_x_coordinate_l1421_142157

-- Define the two lines
def line1 (x : ℝ) : ℝ := 3 * x + 4
def line2 (x y : ℝ) : Prop := 3 * x + y = 25

-- Theorem statement
theorem intersection_x_coordinate :
  ∃ (x y : ℝ), line2 x y ∧ y = line1 x ∧ x = 3.5 := by
  sorry

end intersection_x_coordinate_l1421_142157


namespace valid_street_distances_l1421_142187

/-- Represents the position of a house on a street. -/
structure House where
  position : ℝ

/-- The street with four houses. -/
structure Street where
  andrei : House
  borya : House
  vova : House
  gleb : House

/-- The distance between two houses. -/
def distance (h1 h2 : House) : ℝ :=
  |h1.position - h2.position|

/-- A street satisfying the given conditions. -/
def validStreet (s : Street) : Prop :=
  distance s.andrei s.borya = 600 ∧
  distance s.vova s.gleb = 600 ∧
  distance s.andrei s.gleb = 3 * distance s.borya s.vova

theorem valid_street_distances (s : Street) (h : validStreet s) :
  distance s.andrei s.gleb = 900 ∨ distance s.andrei s.gleb = 1800 :=
sorry

end valid_street_distances_l1421_142187


namespace charity_arrangements_l1421_142125

/-- The number of people selected from the class -/
def total_people : ℕ := 6

/-- The maximum number of people that can participate in each activity -/
def max_per_activity : ℕ := 4

/-- The number of charity activities -/
def num_activities : ℕ := 2

/-- The function to calculate the number of different arrangements -/
def num_arrangements (n : ℕ) (k : ℕ) (m : ℕ) : ℕ := sorry

theorem charity_arrangements :
  num_arrangements total_people max_per_activity num_activities = 50 := by sorry

end charity_arrangements_l1421_142125


namespace smallest_area_of_2020th_square_l1421_142137

theorem smallest_area_of_2020th_square (n : ℕ) : 
  n > 0 →
  n^2 = 2019 + (n^2 - 2019) →
  (∀ i : Fin 2019, 1 = 1) →
  n^2 - 2019 ≠ 1 →
  n^2 - 2019 ≥ 6 ∧ 
  ∀ m : ℕ, m > 0 → m^2 = 2019 + (m^2 - 2019) → (∀ i : Fin 2019, 1 = 1) → m^2 - 2019 ≠ 1 → m^2 - 2019 ≥ n^2 - 2019 :=
by sorry

#check smallest_area_of_2020th_square

end smallest_area_of_2020th_square_l1421_142137


namespace intersection_of_M_and_N_l1421_142139

def M : Set ℝ := {x | x^2 + x - 6 < 0}
def N : Set ℝ := {x | |x - 1| ≤ 2}

theorem intersection_of_M_and_N : M ∩ N = {x : ℝ | -1 ≤ x ∧ x < 2} := by sorry

end intersection_of_M_and_N_l1421_142139


namespace can_capacity_l1421_142195

/-- Represents the contents of a can with milk and water -/
structure CanContents where
  milk : ℝ
  water : ℝ

/-- Represents a can with its contents and capacity -/
structure Can where
  contents : CanContents
  capacity : ℝ

/-- The theorem stating the capacity of the can given the conditions -/
theorem can_capacity (initial : CanContents) (final : CanContents) : 
  (initial.milk / initial.water = 5 / 3) →
  (final.milk / final.water = 2 / 1) →
  (final.milk = initial.milk + 8) →
  (final.water = initial.water) →
  (∃ (can : Can), can.contents = final ∧ can.capacity = 72) :=
by sorry

end can_capacity_l1421_142195


namespace quadrilateral_perpendicular_diagonals_l1421_142101

/-- Given a quadrilateral ABCD in the complex plane, construct points O₁, O₂, O₃, O₄
    and prove that O₁O₃ is perpendicular and equal to O₂O₄ -/
theorem quadrilateral_perpendicular_diagonals
  (a b c d : ℂ) : 
  let g₁ : ℂ := (a + d) / 2
  let g₂ : ℂ := (b + a) / 2
  let g₃ : ℂ := (c + b) / 2
  let g₄ : ℂ := (d + c) / 2
  let o₁ : ℂ := g₁ + (d - a) / 2 * Complex.I
  let o₂ : ℂ := g₂ + (a - b) / 2 * Complex.I
  let o₃ : ℂ := g₃ + (c - b) / 2 * Complex.I
  let o₄ : ℂ := g₄ + (d - c) / 2 * Complex.I
  (o₃ - o₁) = (o₄ - o₂) * Complex.I ∧ Complex.abs (o₃ - o₁) = Complex.abs (o₄ - o₂) :=
by
  sorry

end quadrilateral_perpendicular_diagonals_l1421_142101


namespace sweets_distribution_l1421_142192

theorem sweets_distribution (total_sweets : ℕ) (remaining_sweets : ℕ) (alt_children : ℕ) (alt_remaining : ℕ) :
  total_sweets = 358 →
  remaining_sweets = 8 →
  alt_children = 28 →
  alt_remaining = 22 →
  ∃ (children : ℕ), 
    children * ((total_sweets - remaining_sweets) / children) + remaining_sweets = total_sweets ∧
    alt_children * ((total_sweets - alt_remaining) / alt_children) + alt_remaining = total_sweets ∧
    children = 29 :=
by sorry

end sweets_distribution_l1421_142192


namespace quadratic_two_positive_roots_l1421_142189

theorem quadratic_two_positive_roots (a : ℝ) :
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x ≠ y ∧
    (1 - a) * x^2 + (a + 2) * x - 4 = 0 ∧
    (1 - a) * y^2 + (a + 2) * y - 4 = 0) ↔
  (1 < a ∧ a ≤ 2) ∨ a ≥ 10 :=
sorry

end quadratic_two_positive_roots_l1421_142189


namespace inequality_solutions_l1421_142153

def inequality (a x : ℝ) := a * x^2 - (a + 2) * x + 2 < 0

theorem inequality_solutions :
  ∀ a : ℝ,
    (a = -1 → {x : ℝ | inequality a x} = {x : ℝ | x < -2 ∨ x > 1}) ∧
    (a = 0 → {x : ℝ | inequality a x} = {x : ℝ | x > 1}) ∧
    (a < 0 → {x : ℝ | inequality a x} = {x : ℝ | x < 2/a ∨ x > 1}) ∧
    (0 < a ∧ a < 2 → {x : ℝ | inequality a x} = {x : ℝ | 1 < x ∧ x < 2/a}) ∧
    (a = 2 → {x : ℝ | inequality a x} = ∅) ∧
    (a > 2 → {x : ℝ | inequality a x} = {x : ℝ | 2/a < x ∧ x < 1}) :=
by sorry

end inequality_solutions_l1421_142153


namespace work_completion_time_l1421_142152

/-- Given Johnson's and Vincent's individual work rates, calculates the time required for them to complete the work together -/
theorem work_completion_time (johnson_rate vincent_rate : ℚ) 
  (h1 : johnson_rate = 1 / 10)
  (h2 : vincent_rate = 1 / 40) :
  1 / (johnson_rate + vincent_rate) = 8 := by
  sorry

end work_completion_time_l1421_142152


namespace savings_percentage_l1421_142136

/-- Represents a person's financial situation over two years --/
structure FinancialSituation where
  income_year1 : ℝ
  savings_year1 : ℝ
  income_year2 : ℝ
  savings_year2 : ℝ

/-- Calculates the expenditure for a given year --/
def expenditure (income : ℝ) (savings : ℝ) : ℝ :=
  income - savings

/-- Theorem stating the conditions and the result to be proved --/
theorem savings_percentage (f : FinancialSituation) 
  (h1 : f.income_year2 = 1.2 * f.income_year1)
  (h2 : f.savings_year2 = 2 * f.savings_year1)
  (h3 : expenditure f.income_year1 f.savings_year1 + 
        expenditure f.income_year2 f.savings_year2 = 
        2 * expenditure f.income_year1 f.savings_year1) :
  f.savings_year1 / f.income_year1 = 0.2 := by
  sorry

end savings_percentage_l1421_142136


namespace probability_at_least_one_multiple_of_four_l1421_142103

def range_start : ℕ := 1
def range_end : ℕ := 60
def multiples_of_four : ℕ := 15

theorem probability_at_least_one_multiple_of_four :
  let total_numbers := range_end - range_start + 1
  let non_multiples := total_numbers - multiples_of_four
  let prob_neither_multiple := (non_multiples / total_numbers) ^ 2
  1 - prob_neither_multiple = 7 / 16 := by
sorry

end probability_at_least_one_multiple_of_four_l1421_142103


namespace relative_speed_calculation_l1421_142102

/-- Convert meters per second to kilometers per hour -/
def ms_to_kmh (speed_ms : ℝ) : ℝ :=
  speed_ms * 3.6

/-- Convert centimeters per minute to kilometers per hour -/
def cmpm_to_kmh (speed_cmpm : ℝ) : ℝ :=
  speed_cmpm * 0.0006

/-- Calculate the relative speed of two objects moving in opposite directions -/
def relative_speed (speed1 : ℝ) (speed2 : ℝ) : ℝ :=
  speed1 + speed2

theorem relative_speed_calculation (speed1_ms : ℝ) (speed2_cmpm : ℝ) 
  (h1 : speed1_ms = 12.5)
  (h2 : speed2_cmpm = 1800) :
  relative_speed (ms_to_kmh speed1_ms) (cmpm_to_kmh speed2_cmpm) = 46.08 := by
  sorry

#check relative_speed_calculation

end relative_speed_calculation_l1421_142102


namespace stating_count_numbers_with_five_or_six_in_base_eight_l1421_142116

/-- 
Given a positive integer n and a base b, returns the number of integers 
from 1 to n (inclusive) in base b that contain at least one digit d or e.
-/
def count_numbers_with_digits (n : ℕ) (b : ℕ) (d e : ℕ) : ℕ :=
  sorry

/-- 
Theorem stating that the number of integers from 1 to 512 (inclusive) 
in base 8 that contain at least one digit 5 or 6 is equal to 296.
-/
theorem count_numbers_with_five_or_six_in_base_eight : 
  count_numbers_with_digits 512 8 5 6 = 296 := by
  sorry

end stating_count_numbers_with_five_or_six_in_base_eight_l1421_142116


namespace monthly_interest_payment_l1421_142150

/-- Calculate the monthly interest payment given the annual interest rate and investment amount -/
theorem monthly_interest_payment 
  (annual_rate : ℝ) 
  (investment : ℝ) 
  (h1 : annual_rate = 0.09) 
  (h2 : investment = 28800) : 
  (investment * annual_rate) / 12 = 216 := by
  sorry

end monthly_interest_payment_l1421_142150


namespace oil_price_reduction_l1421_142113

/-- Calculates the percentage reduction in oil price given the conditions --/
theorem oil_price_reduction (total_cost : ℝ) (additional_kg : ℝ) (reduced_price : ℝ) : 
  total_cost = 1100 ∧ 
  additional_kg = 5 ∧ 
  reduced_price = 55 →
  (((total_cost / (total_cost / reduced_price - additional_kg)) - reduced_price) / 
   (total_cost / (total_cost / reduced_price - additional_kg))) * 100 = 25 := by
  sorry


end oil_price_reduction_l1421_142113


namespace average_people_per_hour_rounded_l1421_142155

def people_moving : ℕ := 3000
def days : ℕ := 4
def hours_per_day : ℕ := 24

def average_per_hour : ℚ :=
  people_moving / (days * hours_per_day)

theorem average_people_per_hour_rounded :
  round average_per_hour = 31 := by
  sorry

end average_people_per_hour_rounded_l1421_142155


namespace fraction_sum_l1421_142162

theorem fraction_sum (a b c : ℝ) 
  (h : a / (30 - a) + b / (70 - b) + c / (75 - c) = 9) :
  6 / (30 - a) + 14 / (70 - b) + 15 / (75 - c) = 35 := by
  sorry

end fraction_sum_l1421_142162


namespace frog_jumps_l1421_142172

/-- A jump sequence represents the frog's movements, where
    true represents a jump to the right and false represents a jump to the left. -/
def JumpSequence := List Bool

/-- The position after following a jump sequence -/
def position (p q : ℕ) (jumps : JumpSequence) : ℤ :=
  jumps.foldl (λ acc jump => if jump then acc + p else acc - q) 0

/-- A jump sequence is valid if it starts and ends at 0 -/
def is_valid_sequence (p q : ℕ) (jumps : JumpSequence) : Prop :=
  position p q jumps = 0

theorem frog_jumps (p q : ℕ) (jumps : JumpSequence) (d : ℕ) :
  Nat.Coprime p q →
  is_valid_sequence p q jumps →
  d < p + q →
  ∃ (i j : ℕ), i < jumps.length ∧ j < jumps.length ∧
    abs (position p q (jumps.take i) - position p q (jumps.take j)) = d :=
sorry

end frog_jumps_l1421_142172


namespace prank_combinations_l1421_142186

theorem prank_combinations (choices : List Nat) : 
  choices = [2, 3, 0, 6, 1] → List.prod choices = 0 := by
  sorry

end prank_combinations_l1421_142186


namespace parabola_equation_and_vertex_l1421_142134

/-- A parabola passing through points (1, 0) and (3, 0) -/
def Parabola (x y : ℝ) : Prop :=
  ∃ b c : ℝ, y = -x^2 + b*x + c ∧ 0 = -1 + b + c ∧ 0 = -9 + 3*b + c

theorem parabola_equation_and_vertex :
  (∀ x y : ℝ, Parabola x y ↔ y = -x^2 + 4*x - 3) ∧
  (∃ x y : ℝ, Parabola x y ∧ x = 2 ∧ y = 1 ∧
    ∀ x' y' : ℝ, Parabola x' y' → y' ≤ y) :=
by sorry

end parabola_equation_and_vertex_l1421_142134


namespace problem_solution_l1421_142168

theorem problem_solution (x y : ℝ) 
  (h1 : x + Real.sin y = 2021)
  (h2 : x + 2021 * Real.cos y = 2020)
  (h3 : π / 2 ≤ y ∧ y ≤ π) :
  x + y = 2020 + π / 2 := by
  sorry

end problem_solution_l1421_142168


namespace hall_length_proof_l1421_142156

theorem hall_length_proof (breadth : ℝ) (length : ℝ) (area : ℝ) : 
  length = breadth + 5 →
  area = length * breadth →
  area = 750 →
  length = 30 := by
sorry

end hall_length_proof_l1421_142156


namespace triangle_angle_b_sixty_degrees_l1421_142170

theorem triangle_angle_b_sixty_degrees 
  (a b c : ℝ) 
  (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
  (condition : c / (a + b) + a / (b + c) = 1) : 
  angle_b = π / 3 :=
sorry

end triangle_angle_b_sixty_degrees_l1421_142170


namespace imaginary_part_of_complex_number_l1421_142145

theorem imaginary_part_of_complex_number (z : ℂ) : z = 1 + 1 / Complex.I → z.im = -1 := by
  sorry

end imaginary_part_of_complex_number_l1421_142145


namespace min_z_value_l1421_142167

theorem min_z_value (x y z : ℤ) : 
  x < y → y < z → 
  y - x > 5 → 
  Even x → Odd y → Odd z → 
  z - x ≥ 9 → 
  (∀ w, w < z → ¬(x < w ∧ w < z ∧ w - x > 5 ∧ Odd w)) →
  z = 9 := by
sorry

end min_z_value_l1421_142167


namespace spider_web_production_l1421_142105

def spider_webs (num_spiders : ℕ) (num_webs : ℕ) (days : ℕ) : Prop :=
  num_spiders = num_webs ∧ days > 0

theorem spider_web_production 
  (h1 : spider_webs 7 7 (7 : ℕ)) 
  (h2 : spider_webs 1 1 7) : 
  ∀ s, s ≤ 7 → spider_webs 1 1 7 :=
sorry

end spider_web_production_l1421_142105


namespace sum_of_greatest_b_values_l1421_142182

theorem sum_of_greatest_b_values (c : ℝ) (h : c ≠ 0) :
  ∃ (b₁ b₂ : ℝ), b₁ > b₂ ∧ b₂ > 0 ∧
  (4 * b₁^4 - 41 * b₁^2 + 100) * c = 0 ∧
  (4 * b₂^4 - 41 * b₂^2 + 100) * c = 0 ∧
  ∀ (b : ℝ), (4 * b^4 - 41 * b^2 + 100) * c = 0 → b ≤ b₁ ∧
  b₁ + b₂ = 4.5 :=
sorry

end sum_of_greatest_b_values_l1421_142182


namespace imaginary_part_of_complex_fraction_l1421_142148

theorem imaginary_part_of_complex_fraction : 
  Complex.im ((Complex.I / (1 + 2 * Complex.I)) * Complex.I) = 1 / 5 := by
  sorry

end imaginary_part_of_complex_fraction_l1421_142148


namespace marbles_ratio_l1421_142130

def marbles_problem (initial_marbles : ℕ) (current_marbles : ℕ) (brother_marbles : ℕ) : Prop :=
  let savanna_marbles := 3 * current_marbles
  let sister_marbles := initial_marbles - current_marbles - brother_marbles - savanna_marbles
  (sister_marbles : ℚ) / brother_marbles = 2

theorem marbles_ratio :
  marbles_problem 300 30 60 := by
  sorry

end marbles_ratio_l1421_142130


namespace h_max_value_f_leq_g_condition_l1421_142188

noncomputable section

-- Define the functions
def f (x : ℝ) : ℝ := 1 - Real.exp (-x)
def g (a : ℝ) (x : ℝ) : ℝ := x / (a * x + 1)
def h (x : ℝ) : ℝ := x * Real.exp (-x)

-- Theorem for the maximum value of h(x)
theorem h_max_value :
  ∃ (x : ℝ), ∀ (y : ℝ), h y ≤ h x ∧ h x = 1 / Real.exp 1 :=
sorry

-- Theorem for the range of a
theorem f_leq_g_condition (a : ℝ) :
  (∀ x : ℝ, x ≥ 0 → f x ≤ g a x) ↔ (0 ≤ a ∧ a ≤ 1/2) :=
sorry

end h_max_value_f_leq_g_condition_l1421_142188


namespace taxi_fare_for_8_2km_l1421_142166

/-- Calculates the taxi fare for a given distance -/
def taxiFare (distance : Float) : Float :=
  let baseFare := 6
  let midRateDistance := 4
  let midRate := 1
  let highRate := 0.8
  let baseDistance := 3
  let midDistanceEnd := 7
  if distance ≤ baseDistance then
    baseFare
  else if distance ≤ midDistanceEnd then
    baseFare + midRate * (Float.ceil (distance - baseDistance))
  else
    baseFare + midRate * midRateDistance + highRate * (Float.ceil (distance - midDistanceEnd))

theorem taxi_fare_for_8_2km :
  taxiFare 8.2 = 11.6 := by
  sorry

end taxi_fare_for_8_2km_l1421_142166


namespace road_length_proof_l1421_142117

/-- The length of a road given round trip conditions -/
theorem road_length_proof (total_time : ℝ) (walking_speed : ℝ) (bus_speed : ℝ)
  (h1 : total_time = 2)
  (h2 : walking_speed = 5)
  (h3 : bus_speed = 20) :
  ∃ (road_length : ℝ), road_length / walking_speed + road_length / bus_speed = total_time ∧ road_length = 8 := by
  sorry

end road_length_proof_l1421_142117


namespace mishas_current_money_l1421_142141

/-- Misha's current amount of money in dollars -/
def current_money : ℕ := sorry

/-- The amount Misha needs to earn in dollars -/
def money_to_earn : ℕ := 13

/-- The total amount Misha will have after earning more money, in dollars -/
def total_money : ℕ := 47

/-- Theorem stating Misha's current amount of money -/
theorem mishas_current_money : current_money = 34 := by sorry

end mishas_current_money_l1421_142141


namespace trigonometric_expression_equals_neg_five_thirds_l1421_142106

theorem trigonometric_expression_equals_neg_five_thirds :
  (Real.tan (30 * π / 180))^2 - (Real.cos (30 * π / 180))^2
  / ((Real.tan (30 * π / 180))^2 * (Real.cos (30 * π / 180))^2) = -5/3 := by
  sorry

end trigonometric_expression_equals_neg_five_thirds_l1421_142106


namespace special_integers_count_l1421_142149

/-- Sum of all positive divisors of n -/
def g (n : ℕ) : ℕ := sorry

/-- The number of integers j such that 1 ≤ j ≤ 5041 and g(j) = 1 + √j + j -/
def count_special_integers : ℕ := sorry

theorem special_integers_count :
  count_special_integers = 20 := by sorry

end special_integers_count_l1421_142149


namespace quadratic_expression_value_l1421_142190

theorem quadratic_expression_value (x y : ℝ) 
  (eq1 : 3 * x + 2 * y = 8) 
  (eq2 : 2 * x + 3 * y = 11) : 
  13 * x^2 + 22 * x * y + 13 * y^2 = 184 := by
  sorry

end quadratic_expression_value_l1421_142190


namespace sum_of_roots_eq_40_l1421_142114

/-- The parabola P defined by y = x^2 + 4 -/
def P : ℝ → ℝ := λ x ↦ x^2 + 4

/-- The point Q -/
def Q : ℝ × ℝ := (10, 6)

/-- The slope-intercept form of a line through Q with slope m -/
def line (m : ℝ) : ℝ → ℝ := λ x ↦ m * (x - Q.1) + Q.2

/-- The quadratic equation representing the intersection of the line and parabola -/
def intersection_eq (m : ℝ) : ℝ → ℝ := λ x ↦ x^2 - m * x + (10 * m - 2)

/-- The discriminant of the intersection equation -/
def discriminant (m : ℝ) : ℝ := m^2 - 4 * (10 * m - 2)

theorem sum_of_roots_eq_40 :
  ∃ r s : ℝ, r + s = 40 ∧
    ∀ m : ℝ, discriminant m < 0 ↔ r < m ∧ m < s :=
sorry

end sum_of_roots_eq_40_l1421_142114


namespace pages_difference_l1421_142164

/-- Represents the number of pages in a purple book -/
def purple_pages : ℕ := 230

/-- Represents the number of pages in an orange book -/
def orange_pages : ℕ := 510

/-- Represents the number of purple books Mirella read -/
def purple_books_read : ℕ := 5

/-- Represents the number of orange books Mirella read -/
def orange_books_read : ℕ := 4

/-- Theorem stating the difference in pages read between orange and purple books -/
theorem pages_difference : 
  orange_pages * orange_books_read - purple_pages * purple_books_read = 890 := by
  sorry

end pages_difference_l1421_142164


namespace range_of_a_l1421_142143

def prop_A (a : ℝ) : Prop :=
  ∀ x, x^2 + (a - 1) * x + a^2 > 0

def prop_B (a : ℝ) : Prop :=
  ∀ x y, x < y → (2 * a^2 - a)^x < (2 * a^2 - a)^y

def exclusive_or (P Q : Prop) : Prop :=
  (P ∧ ¬Q) ∨ (¬P ∧ Q)

theorem range_of_a : 
  {a : ℝ | exclusive_or (prop_A a) (prop_B a)} = 
  {a : ℝ | -1 ≤ a ∧ a < -1/2 ∨ 1/3 < a ∧ a ≤ 1} :=
sorry

end range_of_a_l1421_142143


namespace batsman_boundaries_l1421_142112

theorem batsman_boundaries (total_runs : ℕ) (sixes : ℕ) (run_percentage : ℚ) : 
  total_runs = 120 →
  sixes = 8 →
  run_percentage = 1/2 →
  (∃ (boundaries : ℕ), 
    total_runs = run_percentage * total_runs + sixes * 6 + boundaries * 4 ∧
    boundaries = 3) :=
by
  sorry

end batsman_boundaries_l1421_142112


namespace sphere_surface_area_with_cone_l1421_142176

/-- The surface area of a sphere containing a cone with base radius 1 and height √3 -/
theorem sphere_surface_area_with_cone (R : ℝ) : 
  (R : ℝ) > 0 → -- Radius is positive
  R^2 = (R - Real.sqrt 3)^2 + 1 → -- Cone geometry condition
  4 * π * R^2 = 16 * π / 3 := by
sorry

end sphere_surface_area_with_cone_l1421_142176


namespace real_condition_implies_a_equals_one_l1421_142120

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the property that a complex number is real
def is_real (z : ℂ) : Prop := z.im = 0

-- Theorem statement
theorem real_condition_implies_a_equals_one (a : ℝ) :
  is_real ((1 + i) * (1 - a * i)) → a = 1 := by
  sorry

end real_condition_implies_a_equals_one_l1421_142120


namespace wall_length_proof_l1421_142183

/-- Proves that the length of a wall is 900 cm given specific brick and wall dimensions --/
theorem wall_length_proof (brick_length : ℝ) (brick_width : ℝ) (brick_height : ℝ)
                          (wall_height : ℝ) (wall_width : ℝ) (num_bricks : ℕ) :
  brick_length = 25 →
  brick_width = 11.25 →
  brick_height = 6 →
  wall_height = 600 →
  wall_width = 22.5 →
  num_bricks = 7200 →
  (brick_length * brick_width * brick_height * num_bricks) / (wall_height * wall_width) = 900 :=
by
  sorry

#check wall_length_proof

end wall_length_proof_l1421_142183


namespace root_sum_reciprocal_l1421_142131

theorem root_sum_reciprocal (α β γ : ℂ) : 
  (α^3 - 2*α^2 - α + 2 = 0) → 
  (β^3 - 2*β^2 - β + 2 = 0) → 
  (γ^3 - 2*γ^2 - γ + 2 = 0) → 
  (1 / (α + 2) + 1 / (β + 2) + 1 / (γ + 2) = -19 / 14) := by
  sorry

end root_sum_reciprocal_l1421_142131


namespace arithmetic_sequence_sin_sum_l1421_142154

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sin_sum 
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : a 1 + a 5 + a 9 = 2 * Real.pi) : 
  Real.sin (a 2 + a 8) = -Real.sqrt 3 / 2 := by
  sorry

end arithmetic_sequence_sin_sum_l1421_142154


namespace golf_cart_capacity_l1421_142196

/-- The number of patrons that can fit in a golf cart -/
def patrons_per_cart (patrons_from_cars : ℕ) (patrons_from_bus : ℕ) (total_carts : ℕ) : ℕ :=
  (patrons_from_cars + patrons_from_bus) / total_carts

/-- Theorem: Given the conditions from the problem, prove that 3 patrons can fit in a golf cart -/
theorem golf_cart_capacity :
  patrons_per_cart 12 27 13 = 3 := by
  sorry

end golf_cart_capacity_l1421_142196


namespace total_books_count_l1421_142171

/-- Given that Sandy has 10 books, Benny has 24 books, and Tim has 33 books,
    prove that they have 67 books in total. -/
theorem total_books_count (sandy_books benny_books tim_books : ℕ) 
  (h1 : sandy_books = 10)
  (h2 : benny_books = 24)
  (h3 : tim_books = 33) : 
  sandy_books + benny_books + tim_books = 67 := by
  sorry

end total_books_count_l1421_142171


namespace total_weight_of_baskets_l1421_142104

def basket_weight : ℕ := 30
def num_baskets : ℕ := 8

theorem total_weight_of_baskets : basket_weight * num_baskets = 240 := by
  sorry

end total_weight_of_baskets_l1421_142104


namespace value_of_2a_minus_b_l1421_142191

-- Define the functions
def f (a b x : ℝ) : ℝ := a * x + b
def g (x : ℝ) : ℝ := -4 * x + 6
def h (a b x : ℝ) : ℝ := f a b (g x)

-- State the theorem
theorem value_of_2a_minus_b (a b : ℝ) :
  (∀ x, h a b x = x - 9) →  -- h is the inverse of x + 9
  2 * a - b = 7 := by
  sorry

end value_of_2a_minus_b_l1421_142191


namespace binomial_properties_l1421_142119

-- Define the binomial distribution
def binomial_distribution (n : ℕ) (p : ℝ) : ℕ → ℝ := sorry

-- Define the probability of X being odd
def prob_odd (n : ℕ) (p : ℝ) : ℝ := sorry

-- Define the probability of X being even
def prob_even (n : ℕ) (p : ℝ) : ℝ := sorry

theorem binomial_properties (n : ℕ) (p : ℝ) 
  (h1 : n > 0) (h2 : 0 < p) (h3 : p < 1) :
  -- 1. The sum of probabilities of X being odd and even equals 1
  (prob_odd n p + prob_even n p = 1) ∧ 
  -- 2. When p = 1/2, the probability of X being odd equals the probability of X being even
  (p = 1/2 → prob_odd n p = prob_even n p) ∧ 
  -- 3. When 0 < p < 1/2, the probability of X being odd increases as n increases
  (p < 1/2 → ∀ m, n < m → prob_odd n p < prob_odd m p) :=
by sorry

end binomial_properties_l1421_142119


namespace bugs_eating_flowers_l1421_142177

/-- Given 3 bugs, each eating 2 flowers, the total number of flowers eaten is 6. -/
theorem bugs_eating_flowers :
  let num_bugs : ℕ := 3
  let flowers_per_bug : ℕ := 2
  num_bugs * flowers_per_bug = 6 := by
  sorry

end bugs_eating_flowers_l1421_142177


namespace mean_height_is_70_625_l1421_142160

def heights : List ℝ := [58, 59, 60, 61, 64, 65, 68, 70, 73, 73, 75, 76, 77, 78, 78, 79]

theorem mean_height_is_70_625 :
  (heights.sum / heights.length : ℝ) = 70.625 := by
  sorry

end mean_height_is_70_625_l1421_142160


namespace power_ratio_simplification_l1421_142199

theorem power_ratio_simplification : (10^2003 + 10^2001) / (10^2002 + 10^2002) = 101/20 := by
  sorry

end power_ratio_simplification_l1421_142199

import Mathlib

namespace NUMINAMATH_CALUDE_m_range_l1837_183772

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the theorem
theorem m_range (h1 : ∀ x, x ∈ Set.Icc (-2) 2 → f x ∈ Set.Icc (-2) 2)
                (h2 : StrictMono f)
                (h3 : ∀ m, f (1 - m) < f m) :
  ∀ m, m ∈ Set.Ioo (1/2) 2 :=
sorry

end NUMINAMATH_CALUDE_m_range_l1837_183772


namespace NUMINAMATH_CALUDE_z_in_first_quadrant_l1837_183753

theorem z_in_first_quadrant (z : ℂ) (h : z * (1 - 2*I) = 3 - I) : 
  (z.re > 0) ∧ (z.im > 0) :=
by sorry

end NUMINAMATH_CALUDE_z_in_first_quadrant_l1837_183753


namespace NUMINAMATH_CALUDE_rectangle_ratio_l1837_183731

theorem rectangle_ratio (area : ℝ) (length : ℝ) (breadth : ℝ) :
  area = 6075 →
  length = 135 →
  area = length * breadth →
  length / breadth = 3 := by
sorry

end NUMINAMATH_CALUDE_rectangle_ratio_l1837_183731


namespace NUMINAMATH_CALUDE_probability_12_draws_10_red_l1837_183798

/-- The number of yellow balls in the bag -/
def yellow_balls : ℕ := 5

/-- The number of red balls in the bag -/
def red_balls : ℕ := 3

/-- The total number of balls in the bag -/
def total_balls : ℕ := yellow_balls + red_balls

/-- The probability of drawing a red ball -/
def p_red : ℚ := red_balls / total_balls

/-- The probability of drawing a yellow ball -/
def p_yellow : ℚ := yellow_balls / total_balls

/-- The number of red balls needed to stop the process -/
def red_balls_needed : ℕ := 10

/-- The number of draws when the process stops -/
def total_draws : ℕ := 12

/-- The probability of drawing exactly 12 balls to get 10 red balls -/
theorem probability_12_draws_10_red (ξ : ℕ → ℚ) : 
  ξ total_draws = (Nat.choose (total_draws - 1) (red_balls_needed - 1)) * 
                  (p_red ^ red_balls_needed) * 
                  (p_yellow ^ (total_draws - red_balls_needed)) := by
  sorry

end NUMINAMATH_CALUDE_probability_12_draws_10_red_l1837_183798


namespace NUMINAMATH_CALUDE_fraction_equality_l1837_183790

theorem fraction_equality (p q s u : ℚ) 
  (h1 : p / q = 5 / 4) 
  (h2 : s / u = 8 / 15) : 
  (2 * p * s - 3 * q * u) / (5 * q * u - 4 * p * s) = -5 / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1837_183790


namespace NUMINAMATH_CALUDE_probability_three_heads_out_of_five_probability_three_heads_proof_l1837_183766

/-- The probability of three specific coins out of five coming up heads when all five are flipped simultaneously -/
theorem probability_three_heads_out_of_five : ℚ :=
  1 / 8

/-- The total number of possible outcomes when flipping five coins -/
def total_outcomes : ℕ := 2^5

/-- The number of successful outcomes where three specific coins are heads -/
def successful_outcomes : ℕ := 2^2

theorem probability_three_heads_proof :
  (successful_outcomes : ℚ) / total_outcomes = probability_three_heads_out_of_five :=
sorry

end NUMINAMATH_CALUDE_probability_three_heads_out_of_five_probability_three_heads_proof_l1837_183766


namespace NUMINAMATH_CALUDE_nathan_tokens_used_l1837_183715

/-- The total number of tokens used by Nathan at the arcade --/
def total_tokens (air_hockey_plays : ℕ) (basketball_plays : ℕ) (skee_ball_plays : ℕ)
                 (air_hockey_cost : ℕ) (basketball_cost : ℕ) (skee_ball_cost : ℕ) : ℕ :=
  air_hockey_plays * air_hockey_cost +
  basketball_plays * basketball_cost +
  skee_ball_plays * skee_ball_cost

/-- Theorem stating that Nathan used 64 tokens in total --/
theorem nathan_tokens_used :
  total_tokens 5 7 3 4 5 3 = 64 := by
  sorry

end NUMINAMATH_CALUDE_nathan_tokens_used_l1837_183715


namespace NUMINAMATH_CALUDE_gcd_12569_36975_l1837_183733

theorem gcd_12569_36975 : Nat.gcd 12569 36975 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_12569_36975_l1837_183733


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1837_183771

/-- Given a geometric sequence starting with 25, -50, 100, -200, prove that its common ratio is -2 -/
theorem geometric_sequence_common_ratio :
  let a₁ : ℝ := 25
  let a₂ : ℝ := -50
  let a₃ : ℝ := 100
  let a₄ : ℝ := -200
  ∀ r : ℝ, (a₂ = r * a₁ ∧ a₃ = r * a₂ ∧ a₄ = r * a₃) → r = -2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1837_183771


namespace NUMINAMATH_CALUDE_sum_of_six_numbers_l1837_183756

theorem sum_of_six_numbers : (36 : ℕ) + 17 + 32 + 54 + 28 + 3 = 170 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_six_numbers_l1837_183756


namespace NUMINAMATH_CALUDE_derivative_f_at_pi_fourth_l1837_183719

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x + 1

theorem derivative_f_at_pi_fourth : 
  (deriv f) (π/4) = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_pi_fourth_l1837_183719


namespace NUMINAMATH_CALUDE_complex_fraction_equals_neg_two_i_l1837_183703

theorem complex_fraction_equals_neg_two_i :
  let z : ℂ := 1 + I
  (z^2 - 2*z) / (1 - z) = -2*I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_equals_neg_two_i_l1837_183703


namespace NUMINAMATH_CALUDE_green_ratio_l1837_183743

theorem green_ratio (total : ℕ) (girls : ℕ) (yellow : ℕ) 
  (h_total : total = 30)
  (h_girls : girls = 18)
  (h_yellow : yellow = 9)
  (h_pink : girls / 3 = 6) :
  (total - (girls / 3 + yellow)) / total = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_green_ratio_l1837_183743


namespace NUMINAMATH_CALUDE_coloring_book_shelves_l1837_183761

theorem coloring_book_shelves (initial_stock : ℝ) (acquired : ℝ) (books_per_shelf : ℝ) 
  (h1 : initial_stock = 40.0)
  (h2 : acquired = 20.0)
  (h3 : books_per_shelf = 4.0) :
  (initial_stock + acquired) / books_per_shelf = 15 := by
  sorry

end NUMINAMATH_CALUDE_coloring_book_shelves_l1837_183761


namespace NUMINAMATH_CALUDE_rectangular_solid_surface_area_l1837_183768

theorem rectangular_solid_surface_area (a b c : ℕ) : 
  Prime a → Prime b → Prime c → 
  a * b * c = 273 → 
  2 * (a * b + b * c + c * a) = 302 := by sorry

end NUMINAMATH_CALUDE_rectangular_solid_surface_area_l1837_183768


namespace NUMINAMATH_CALUDE_square_sum_theorem_l1837_183707

theorem square_sum_theorem (x y : ℝ) (h1 : x - y = 5) (h2 : -x*y = 4) : x^2 + y^2 = 17 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_theorem_l1837_183707


namespace NUMINAMATH_CALUDE_sufficient_condition_for_inequality_l1837_183757

theorem sufficient_condition_for_inequality (a x : ℝ) : 
  (-2 < x ∧ x < -1) → (a > 2 → (a + x) * (1 + x) < 0) := by sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_inequality_l1837_183757


namespace NUMINAMATH_CALUDE_sum_of_radii_is_14_l1837_183799

/-- The sum of all possible radii of a circle that is tangent to both positive x and y-axes
    and externally tangent to another circle centered at (5,0) with radius 2 is equal to 14. -/
theorem sum_of_radii_is_14 : ∃ r₁ r₂ : ℝ,
  (∀ x y : ℝ, x^2 + y^2 = r₁^2 ∧ x ≥ 0 ∧ y ≥ 0 → (x = r₁ ∧ y = 0) ∨ (x = 0 ∧ y = r₁)) ∧
  (∀ x y : ℝ, x^2 + y^2 = r₂^2 ∧ x ≥ 0 ∧ y ≥ 0 → (x = r₂ ∧ y = 0) ∨ (x = 0 ∧ y = r₂)) ∧
  ((r₁ - 5)^2 + r₁^2 = (r₁ + 2)^2) ∧
  ((r₂ - 5)^2 + r₂^2 = (r₂ + 2)^2) ∧
  r₁ + r₂ = 14 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_radii_is_14_l1837_183799


namespace NUMINAMATH_CALUDE_base_of_exponent_l1837_183776

theorem base_of_exponent (x : ℕ) (h : x = 14) :
  (∀ y : ℕ, y > x → ¬(3^y ∣ 9^7)) ∧ (3^x ∣ 9^7) →
  ∃ b : ℕ, b^7 = 9^7 ∧ b = 9 :=
by sorry

end NUMINAMATH_CALUDE_base_of_exponent_l1837_183776


namespace NUMINAMATH_CALUDE_total_balls_purchased_l1837_183723

/-- The number of golf balls in a dozen -/
def balls_per_dozen : ℕ := 12

/-- The number of dozens Dan buys -/
def dan_dozens : ℕ := 5

/-- The number of dozens Gus buys -/
def gus_dozens : ℕ := 2

/-- The number of golf balls Chris buys -/
def chris_balls : ℕ := 48

/-- The total number of golf balls purchased -/
def total_balls : ℕ := dan_dozens * balls_per_dozen + gus_dozens * balls_per_dozen + chris_balls

theorem total_balls_purchased :
  total_balls = 132 := by sorry

end NUMINAMATH_CALUDE_total_balls_purchased_l1837_183723


namespace NUMINAMATH_CALUDE_min_sum_squares_l1837_183784

def S : Finset Int := {-8, -6, -4, -1, 3, 5, 7, 14}

theorem min_sum_squares (p q r s t u v w : Int) 
  (hp : p ∈ S) (hq : q ∈ S) (hr : r ∈ S) (hs : s ∈ S)
  (ht : t ∈ S) (hu : u ∈ S) (hv : v ∈ S) (hw : w ∈ S)
  (hdistinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ p ≠ u ∧ p ≠ v ∧ p ≠ w ∧
               q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ q ≠ u ∧ q ≠ v ∧ q ≠ w ∧
               r ≠ s ∧ r ≠ t ∧ r ≠ u ∧ r ≠ v ∧ r ≠ w ∧
               s ≠ t ∧ s ≠ u ∧ s ≠ v ∧ s ≠ w ∧
               t ≠ u ∧ t ≠ v ∧ t ≠ w ∧
               u ≠ v ∧ u ≠ w ∧
               v ≠ w) :
  (p + q + r + s)^2 + (t + u + v + w)^2 ≥ 98 ∧ 
  ∃ (p' q' r' s' t' u' v' w' : Int),
    p' ∈ S ∧ q' ∈ S ∧ r' ∈ S ∧ s' ∈ S ∧ t' ∈ S ∧ u' ∈ S ∧ v' ∈ S ∧ w' ∈ S ∧
    p' ≠ q' ∧ p' ≠ r' ∧ p' ≠ s' ∧ p' ≠ t' ∧ p' ≠ u' ∧ p' ≠ v' ∧ p' ≠ w' ∧
    q' ≠ r' ∧ q' ≠ s' ∧ q' ≠ t' ∧ q' ≠ u' ∧ q' ≠ v' ∧ q' ≠ w' ∧
    r' ≠ s' ∧ r' ≠ t' ∧ r' ≠ u' ∧ r' ≠ v' ∧ r' ≠ w' ∧
    s' ≠ t' ∧ s' ≠ u' ∧ s' ≠ v' ∧ s' ≠ w' ∧
    t' ≠ u' ∧ t' ≠ v' ∧ t' ≠ w' ∧
    u' ≠ v' ∧ u' ≠ w' ∧
    v' ≠ w' ∧
    (p' + q' + r' + s')^2 + (t' + u' + v' + w')^2 = 98 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squares_l1837_183784


namespace NUMINAMATH_CALUDE_paint_cans_used_l1837_183751

theorem paint_cans_used (initial_capacity : ℕ) (lost_cans : ℕ) (remaining_capacity : ℕ) : 
  initial_capacity = 40 → 
  lost_cans = 4 → 
  remaining_capacity = 30 → 
  ∃ (cans_per_room : ℚ), 
    cans_per_room > 0 ∧
    initial_capacity = (initial_capacity - remaining_capacity) / lost_cans * lost_cans + remaining_capacity ∧
    (initial_capacity : ℚ) / cans_per_room - lost_cans = remaining_capacity / cans_per_room ∧
    remaining_capacity / cans_per_room = 12 := by
  sorry

end NUMINAMATH_CALUDE_paint_cans_used_l1837_183751


namespace NUMINAMATH_CALUDE_katerina_weight_l1837_183791

theorem katerina_weight (total_weight : ℕ) (alexa_weight : ℕ) 
  (h1 : total_weight = 95)
  (h2 : alexa_weight = 46) :
  total_weight - alexa_weight = 49 := by
  sorry

end NUMINAMATH_CALUDE_katerina_weight_l1837_183791


namespace NUMINAMATH_CALUDE_exists_maximal_element_l1837_183758

/-- A family of subsets of ℕ satisfying the chain condition -/
structure ChainFamily where
  C : Set (Set ℕ)
  chain_condition : ∀ (chain : ℕ → Set ℕ), (∀ n m, n ≤ m → chain n ⊆ chain m) →
    (∀ n, chain n ∈ C) → ∃ S ∈ C, ∀ n, chain n ⊆ S

/-- The existence of a maximal element in a chain family -/
theorem exists_maximal_element (F : ChainFamily) :
  ∃ S ∈ F.C, ∀ T ∈ F.C, S ⊆ T → S = T := by sorry

end NUMINAMATH_CALUDE_exists_maximal_element_l1837_183758


namespace NUMINAMATH_CALUDE_sum_of_roots_f_2y_eq_10_l1837_183789

/-- The function f as defined in the problem -/
def f (x : ℝ) : ℝ := (3*x)^2 + 3*x + 1

/-- The theorem stating the sum of roots of f(2y) = 10 -/
theorem sum_of_roots_f_2y_eq_10 :
  ∃ y₁ y₂ : ℝ, f (2*y₁) = 10 ∧ f (2*y₂) = 10 ∧ y₁ + y₂ = -0.17 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_f_2y_eq_10_l1837_183789


namespace NUMINAMATH_CALUDE_expand_product_l1837_183704

theorem expand_product (x : ℝ) : (x + 4) * (x - 5) = x^2 - x - 20 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l1837_183704


namespace NUMINAMATH_CALUDE_rotation_equivalence_l1837_183732

def clockwise_rotation : ℝ := 480
def counterclockwise_rotation : ℝ := 240

theorem rotation_equivalence :
  ∀ y : ℝ,
  y < 360 →
  (clockwise_rotation % 360 = (360 - y) % 360) →
  y = counterclockwise_rotation :=
by
  sorry

end NUMINAMATH_CALUDE_rotation_equivalence_l1837_183732


namespace NUMINAMATH_CALUDE_soccer_team_selection_l1837_183745

theorem soccer_team_selection (n m k : ℕ) (h1 : n = 18) (h2 : m = 7) (h3 : k = 2) :
  (Nat.choose (n - k) (m - k)) = (Nat.choose 16 5) :=
by sorry

end NUMINAMATH_CALUDE_soccer_team_selection_l1837_183745


namespace NUMINAMATH_CALUDE_palindrome_count_l1837_183760

/-- A multiset representing the available digits -/
def availableDigits : Multiset ℕ := {1, 1, 2, 2, 2, 4, 4, 5, 5}

/-- The length of the palindrome -/
def palindromeLength : ℕ := 9

/-- Function to count 9-digit palindromes formed from the given digits -/
def countPalindromes (digits : Multiset ℕ) (length : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the number of 9-digit palindromes is 36 -/
theorem palindrome_count : countPalindromes availableDigits palindromeLength = 36 :=
  sorry

end NUMINAMATH_CALUDE_palindrome_count_l1837_183760


namespace NUMINAMATH_CALUDE_smallest_x_equals_f_2003_l1837_183747

/-- A function satisfying the given conditions -/
def f (x : ℝ) : ℝ := sorry

/-- The problem statement -/
theorem smallest_x_equals_f_2003 :
  (∀ x > 0, f (3 * x) = 3 * f x) →
  (∀ x ∈ Set.Icc 2 4, f x = 1 - |x - 2|) →
  (∃ x₀ > 0, f x₀ = f 2003 ∧ ∀ x > 0, f x = f 2003 → x ≥ x₀) →
  (∃ x₀ > 0, f x₀ = f 2003 ∧ ∀ x > 0, f x = f 2003 → x ≥ x₀ ∧ x₀ = 1422817) :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_equals_f_2003_l1837_183747


namespace NUMINAMATH_CALUDE_equation_solutions_l1837_183797

theorem equation_solutions : ∃ (x₁ x₂ : ℝ) (z₁ z₂ : ℂ),
  (∀ x : ℝ, (15*x - x^2)/(x + 2) * (x + (15 - x)/(x + 2)) = 48 ↔ x = x₁ ∨ x = x₂) ∧
  (∀ z : ℂ, (15*z - z^2)/(z + 2) * (z + (15 - z)/(z + 2)) = 48 ↔ z = z₁ ∨ z = z₂) ∧
  x₁ = 4 ∧ x₂ = -3 ∧ z₁ = 3 + Complex.I * Real.sqrt 2 ∧ z₂ = 3 - Complex.I * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1837_183797


namespace NUMINAMATH_CALUDE_banana_count_l1837_183755

def fruit_bowl (apples pears bananas : ℕ) : Prop :=
  (pears = apples + 2) ∧
  (bananas = pears + 3) ∧
  (apples + pears + bananas = 19)

theorem banana_count :
  ∀ (a p b : ℕ), fruit_bowl a p b → b = 9 := by
  sorry

end NUMINAMATH_CALUDE_banana_count_l1837_183755


namespace NUMINAMATH_CALUDE_division_result_and_thousandths_digit_l1837_183724

theorem division_result_and_thousandths_digit : 
  let result : ℚ := 57 / 5000
  (result = 0.0114) ∧ 
  (⌊result * 1000⌋ % 10 = 4) := by
  sorry

end NUMINAMATH_CALUDE_division_result_and_thousandths_digit_l1837_183724


namespace NUMINAMATH_CALUDE_mapping_A_to_B_l1837_183785

def f (x : ℝ) : ℝ := 2 * x - 1

def B : Set ℝ := {-3, -1, 3}

theorem mapping_A_to_B :
  ∃ A : Set ℝ, (∀ x ∈ A, f x ∈ B) ∧ (∀ y ∈ B, ∃ x ∈ A, f x = y) ∧ A = {-1, 0, 2} := by
  sorry

end NUMINAMATH_CALUDE_mapping_A_to_B_l1837_183785


namespace NUMINAMATH_CALUDE_monty_hall_probabilities_l1837_183709

/-- Represents the three doors in the Monty Hall problem -/
inductive Door : Type
  | door1 : Door
  | door2 : Door
  | door3 : Door

/-- Represents the possible contents behind a door -/
inductive Content : Type
  | car : Content
  | goat : Content

/-- The Monty Hall game setup -/
structure MontyHallGame where
  prize_door : Door
  initial_choice : Door
  opened_door : Door
  h_prize_not_opened : opened_door ≠ prize_door
  h_opened_is_goat : opened_door ≠ initial_choice

/-- The probability of winning by sticking with the initial choice -/
def prob_stick_wins (game : MontyHallGame) : ℚ :=
  1 / 3

/-- The probability of winning by switching doors -/
def prob_switch_wins (game : MontyHallGame) : ℚ :=
  2 / 3

theorem monty_hall_probabilities (game : MontyHallGame) :
  prob_stick_wins game = 1 / 3 ∧ prob_switch_wins game = 2 / 3 := by
  sorry

#check monty_hall_probabilities

end NUMINAMATH_CALUDE_monty_hall_probabilities_l1837_183709


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l1837_183727

theorem inscribed_circle_radius (DE DF EF : ℝ) (h1 : DE = 26) (h2 : DF = 15) (h3 : EF = 17) :
  let s := (DE + DF + EF) / 2
  let area := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF))
  area / s = 81 * Real.sqrt 2 / 29 := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l1837_183727


namespace NUMINAMATH_CALUDE_smallest_sum_in_S_l1837_183774

def S : Set ℚ := {2, 0, -1, -3}

theorem smallest_sum_in_S : 
  ∃ (x y : ℚ), x ∈ S ∧ y ∈ S ∧ x ≠ y ∧ 
  (∀ (a b : ℚ), a ∈ S → b ∈ S → a ≠ b → x + y ≤ a + b) ∧
  x + y = -4 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_in_S_l1837_183774


namespace NUMINAMATH_CALUDE_shaded_area_is_nine_eighths_pi_l1837_183711

/-- Represents a right triangle with circles at its vertices and an additional circle --/
structure TriangleWithCircles where
  -- Side lengths of the right triangle
  ac : ℝ
  ab : ℝ
  bc : ℝ
  -- Radius of circles at triangle vertices
  y : ℝ
  -- Radius of circle P
  x : ℝ
  -- Conditions
  right_triangle : ac^2 + ab^2 = bc^2
  side_ac : y + 2*x + y = ac
  side_ab : y + 4*x + y = ab
  area_ratio : (2*x)^2 = 4*x^2

/-- The shaded area in the triangle configuration is 9π/8 square units --/
theorem shaded_area_is_nine_eighths_pi (t : TriangleWithCircles)
  (h1 : t.ac = 3)
  (h2 : t.ab = 4)
  (h3 : t.bc = 5) :
  3 * (π * t.y^2 / 2) + π * t.x^2 / 2 = 9 * π / 8 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_is_nine_eighths_pi_l1837_183711


namespace NUMINAMATH_CALUDE_fraction_decimal_digits_l1837_183714

theorem fraction_decimal_digits : 
  let f : ℚ := 90 / (3^2 * 2^5)
  ∃ (d : ℕ) (n : ℕ), f = d.cast / 10^n ∧ (d % 10 ≠ 0) ∧ n = 4 :=
by sorry

end NUMINAMATH_CALUDE_fraction_decimal_digits_l1837_183714


namespace NUMINAMATH_CALUDE_inscribed_square_area_l1837_183726

/-- The area of a square inscribed in an ellipse -/
theorem inscribed_square_area (x y : ℝ) :
  (x^2 / 4 + y^2 / 8 = 1) →  -- Ellipse equation
  (∃ t : ℝ, t > 0 ∧ x = t ∧ y = t) →  -- Square vertex condition
  (4 * t^2 = 32 / 3) :=  -- Area of the square
by
  sorry


end NUMINAMATH_CALUDE_inscribed_square_area_l1837_183726


namespace NUMINAMATH_CALUDE_bond_energy_OF_bond_energy_OF_proof_l1837_183780

-- Define the molecules and atoms
inductive Molecule | OF₂ | O₂ | F₂
inductive Atom | O | F

-- Define the enthalpy of formation for OF₂
def enthalpy_formation_OF₂ : ℝ := 22

-- Define the bond energies for O₂ and F₂
def bond_energy_O₂ : ℝ := 498
def bond_energy_F₂ : ℝ := 159

-- Define the thermochemical equations
def thermochem_OF₂ (x : ℝ) : Prop :=
  x = 1 * bond_energy_F₂ + 0.5 * bond_energy_O₂ - enthalpy_formation_OF₂

-- Theorem: The bond energy of O-F in OF₂ is 215 kJ/mol
theorem bond_energy_OF : ℝ :=
  215

-- Proof of the theorem
theorem bond_energy_OF_proof : 
  thermochem_OF₂ bond_energy_OF := by
  sorry


end NUMINAMATH_CALUDE_bond_energy_OF_bond_energy_OF_proof_l1837_183780


namespace NUMINAMATH_CALUDE_largest_number_l1837_183775

def toDecimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * base ^ i) 0

def A : Nat := toDecimal [5, 8] 9
def B : Nat := toDecimal [0, 1, 2] 6
def C : Nat := toDecimal [0, 0, 0, 1] 4
def D : Nat := toDecimal [1, 1, 1, 1, 1] 2

theorem largest_number : 
  B > A ∧ B > C ∧ B > D := by sorry

end NUMINAMATH_CALUDE_largest_number_l1837_183775


namespace NUMINAMATH_CALUDE_intersection_point_correct_l1837_183735

/-- The point of intersection of two lines -/
def intersection_point : ℚ × ℚ := (-9/13, 32/13)

/-- First line equation: 3y = -2x + 6 -/
def line1 (x y : ℚ) : Prop := 3 * y = -2 * x + 6

/-- Second line equation: 2y = -10x - 2 -/
def line2 (x y : ℚ) : Prop := 2 * y = -10 * x - 2

theorem intersection_point_correct :
  let (x, y) := intersection_point
  line1 x y ∧ line2 x y :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_correct_l1837_183735


namespace NUMINAMATH_CALUDE_spade_nested_calculation_l1837_183786

def spade (a b : ℝ) : ℝ := |a - b|

theorem spade_nested_calculation : spade 5 (spade 3 (spade 8 12)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_spade_nested_calculation_l1837_183786


namespace NUMINAMATH_CALUDE_tyler_meal_choices_l1837_183742

-- Define the number of options for each food category
def num_meat_options : ℕ := 3
def num_vegetable_options : ℕ := 5
def num_dessert_options : ℕ := 5

-- Define the number of vegetables to be chosen
def num_vegetables_to_choose : ℕ := 2

-- Theorem statement
theorem tyler_meal_choices :
  (num_meat_options) *
  (num_vegetable_options.choose num_vegetables_to_choose) *
  (num_dessert_options) = 150 := by
  sorry


end NUMINAMATH_CALUDE_tyler_meal_choices_l1837_183742


namespace NUMINAMATH_CALUDE_box_width_proof_l1837_183787

/-- Proves that the width of a box with given dimensions and constraints is 18 cm -/
theorem box_width_proof (length height : ℝ) (cube_volume min_cubes : ℕ) :
  length = 7 →
  height = 3 →
  cube_volume = 9 →
  min_cubes = 42 →
  ∃ width : ℝ,
    width * length * height = min_cubes * cube_volume ∧
    width = 18 := by
  sorry

end NUMINAMATH_CALUDE_box_width_proof_l1837_183787


namespace NUMINAMATH_CALUDE_triangle_side_length_l1837_183788

/-- Given a triangle XYZ with side lengths and median, prove the length of XZ -/
theorem triangle_side_length (XY YZ XM : ℝ) (h1 : XY = 7) (h2 : YZ = 10) (h3 : XM = 5) :
  ∃ (XZ : ℝ), XZ = Real.sqrt 51 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1837_183788


namespace NUMINAMATH_CALUDE_opposite_numbers_subtraction_not_always_smaller_l1837_183734

-- Statement 1
theorem opposite_numbers (a b : ℝ) : a + b = 0 → a = -b := by sorry

-- Statement 2
theorem subtraction_not_always_smaller : ∃ x y : ℚ, x - y > y := by sorry

end NUMINAMATH_CALUDE_opposite_numbers_subtraction_not_always_smaller_l1837_183734


namespace NUMINAMATH_CALUDE_stack_b_tallest_l1837_183765

/-- Represents the height of a stack of wood blocks -/
def stack_height (num_pieces : ℕ) (block_height : ℝ) : ℝ :=
  (num_pieces : ℝ) * block_height

/-- Proves that stack B is the tallest among the three stacks of wood blocks -/
theorem stack_b_tallest (height_a height_b height_c : ℝ) 
  (h_height_a : height_a = 2)
  (h_height_b : height_b = 1.5)
  (h_height_c : height_c = 2.5) :
  stack_height 11 height_b > stack_height 8 height_a ∧ 
  stack_height 11 height_b > stack_height 6 height_c :=
by
  sorry

#check stack_b_tallest

end NUMINAMATH_CALUDE_stack_b_tallest_l1837_183765


namespace NUMINAMATH_CALUDE_pyramid_sum_is_25_l1837_183722

/-- Calculates the sum of blocks in a pyramid with given parameters -/
def pyramidSum (levels : Nat) (firstRowBlocks : Nat) (decrease : Nat) : Nat :=
  let blockSequence := List.range levels |>.map (fun i => firstRowBlocks - i * decrease)
  blockSequence.sum

/-- The sum of blocks in a 5-level pyramid with specific parameters is 25 -/
theorem pyramid_sum_is_25 : pyramidSum 5 9 2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_sum_is_25_l1837_183722


namespace NUMINAMATH_CALUDE_minimize_f_l1837_183782

/-- The function f(x) = x^2 + 8x + 3 -/
def f (x : ℝ) : ℝ := x^2 + 8*x + 3

/-- Theorem: The value of x that minimizes f(x) = x^2 + 8x + 3 is -4 -/
theorem minimize_f :
  ∃ (x_min : ℝ), x_min = -4 ∧ ∀ (x : ℝ), f x ≥ f x_min :=
by sorry

end NUMINAMATH_CALUDE_minimize_f_l1837_183782


namespace NUMINAMATH_CALUDE_manuscript_pages_count_l1837_183793

/-- Represents the cost structure and revision information for a manuscript --/
structure ManuscriptInfo where
  firstTypingCost : ℕ
  revisionCost : ℕ
  pagesRevisedOnce : ℕ
  pagesRevisedTwice : ℕ
  totalCost : ℕ

/-- Calculates the total number of pages in a manuscript given its cost information --/
def calculateTotalPages (info : ManuscriptInfo) : ℕ :=
  sorry

/-- Theorem stating that for the given manuscript information, the total number of pages is 100 --/
theorem manuscript_pages_count (info : ManuscriptInfo) 
  (h1 : info.firstTypingCost = 10)
  (h2 : info.revisionCost = 5)
  (h3 : info.pagesRevisedOnce = 30)
  (h4 : info.pagesRevisedTwice = 20)
  (h5 : info.totalCost = 1350) :
  calculateTotalPages info = 100 := by
  sorry

end NUMINAMATH_CALUDE_manuscript_pages_count_l1837_183793


namespace NUMINAMATH_CALUDE_focus_of_parabola_is_correct_l1837_183721

/-- A circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A parabola in the 2D plane -/
structure Parabola where
  p : ℝ
  focus : ℝ × ℝ

/-- The given circle from the problem -/
def given_circle : Circle :=
  { center := (3, 0), radius := 4 }

/-- Checks if a point is on the given circle -/
def on_circle (x y : ℝ) : Prop :=
  (x - 3)^2 + y^2 = 16

/-- The parabola from the problem -/
def given_parabola : Parabola :=
  { p := 1, focus := (1, 0) }

/-- Checks if a point is on the given parabola -/
def on_parabola (x y : ℝ) : Prop :=
  y^2 = 2 * given_parabola.p * x

/-- The theorem to be proved -/
theorem focus_of_parabola_is_correct :
  given_parabola.focus = (1, 0) ∧
  given_parabola.p > 0 ∧
  ∃ (x y : ℝ), on_circle x y ∧ on_parabola x y :=
sorry

end NUMINAMATH_CALUDE_focus_of_parabola_is_correct_l1837_183721


namespace NUMINAMATH_CALUDE_triangle_max_area_l1837_183792

theorem triangle_max_area (A B C : Real) (a b c : Real) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  a * c = 6 →
  Real.sin B + 2 * Real.sin C * Real.cos A = 0 →
  (∀ S : Real, S = (1/2) * a * c * Real.sin B → S ≤ 3/2) ∧
  (∃ S : Real, S = (1/2) * a * c * Real.sin B ∧ S = 3/2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_area_l1837_183792


namespace NUMINAMATH_CALUDE_inequality_solution_l1837_183720

theorem inequality_solution (x : ℝ) : 
  1 / (x * (x + 1)) - 1 / ((x + 2) * (x + 3)) < 1 / 5 ↔ 
  x < -3 ∨ (-2 < x ∧ x < -1) ∨ x > 2 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1837_183720


namespace NUMINAMATH_CALUDE_rhombus_other_diagonal_length_l1837_183739

-- Define the rhombus properties
def diagonal1 : ℝ := 7.4
def area : ℝ := 21.46

-- Theorem to prove
theorem rhombus_other_diagonal_length :
  let diagonal2 := (2 * area) / diagonal1
  ∃ ε > 0, abs (diagonal2 - 5.8) < ε :=
by sorry

end NUMINAMATH_CALUDE_rhombus_other_diagonal_length_l1837_183739


namespace NUMINAMATH_CALUDE_absolute_value_equality_l1837_183737

theorem absolute_value_equality (x : ℝ) :
  |x - 3| = |x - 5| → x = 4 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_equality_l1837_183737


namespace NUMINAMATH_CALUDE_shirt_cost_l1837_183738

theorem shirt_cost (total_money : ℕ) (num_shirts : ℕ) (pants_cost : ℕ) (money_left : ℕ) :
  total_money = 109 →
  num_shirts = 2 →
  pants_cost = 13 →
  money_left = 74 →
  ∃ shirt_cost : ℕ, shirt_cost * num_shirts + pants_cost = total_money - money_left ∧ shirt_cost = 11 :=
by sorry

end NUMINAMATH_CALUDE_shirt_cost_l1837_183738


namespace NUMINAMATH_CALUDE_smallest_prime_sum_of_five_primes_l1837_183716

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def isSumOfFiveDifferentPrimes (n : ℕ) : Prop :=
  ∃ p₁ p₂ p₃ p₄ p₅ : ℕ,
    isPrime p₁ ∧ isPrime p₂ ∧ isPrime p₃ ∧ isPrime p₄ ∧ isPrime p₅ ∧
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₁ ≠ p₅ ∧
    p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₂ ≠ p₅ ∧
    p₃ ≠ p₄ ∧ p₃ ≠ p₅ ∧
    p₄ ≠ p₅ ∧
    n = p₁ + p₂ + p₃ + p₄ + p₅

theorem smallest_prime_sum_of_five_primes :
  isPrime 43 ∧
  isSumOfFiveDifferentPrimes 43 ∧
  ∀ n : ℕ, n < 43 → ¬(isPrime n ∧ isSumOfFiveDifferentPrimes n) :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_sum_of_five_primes_l1837_183716


namespace NUMINAMATH_CALUDE_sum_of_digits_next_l1837_183710

def S (n : ℕ+) : ℕ := sorry

theorem sum_of_digits_next (n : ℕ+) (h : S n = 876) : S (n + 1) = 877 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_next_l1837_183710


namespace NUMINAMATH_CALUDE_square_root_of_four_l1837_183700

theorem square_root_of_four : 
  {x : ℝ | x^2 = 4} = {2, -2} := by sorry

end NUMINAMATH_CALUDE_square_root_of_four_l1837_183700


namespace NUMINAMATH_CALUDE_number_of_black_balls_l1837_183750

/-- Given a bag with red, white, and black balls, prove the number of black balls -/
theorem number_of_black_balls
  (total_balls : ℕ)
  (red_balls : ℕ)
  (white_balls : ℕ)
  (black_balls : ℕ)
  (prob_red : ℚ)
  (prob_white : ℚ)
  (h1 : red_balls = 21)
  (h2 : prob_red = 21 / total_balls)
  (h3 : prob_white = white_balls / total_balls)
  (h4 : prob_red = 42 / 100)
  (h5 : prob_white = 28 / 100)
  (h6 : total_balls = red_balls + white_balls + black_balls) :
  black_balls = 15 := by
  sorry

end NUMINAMATH_CALUDE_number_of_black_balls_l1837_183750


namespace NUMINAMATH_CALUDE_postman_speeds_theorem_l1837_183725

/-- Represents the speeds of the postman on different terrains -/
structure PostmanSpeeds where
  uphill : ℝ
  flat : ℝ
  downhill : ℝ

/-- Checks if the given speeds satisfy the journey conditions -/
def satisfiesConditions (speeds : PostmanSpeeds) : Prop :=
  let uphill := speeds.uphill
  let flat := speeds.flat
  let downhill := speeds.downhill
  (2 / uphill + 4 / flat + 3 / downhill = 2.267) ∧
  (3 / uphill + 4 / flat + 2 / downhill = 2.4) ∧
  (1 / uphill + 2 / flat + 1.5 / downhill = 1.158)

/-- Theorem stating that the specific speeds satisfy the journey conditions -/
theorem postman_speeds_theorem :
  satisfiesConditions { uphill := 3, flat := 4, downhill := 5 } := by
  sorry

#check postman_speeds_theorem

end NUMINAMATH_CALUDE_postman_speeds_theorem_l1837_183725


namespace NUMINAMATH_CALUDE_right_triangle_bisector_inscribed_circle_l1837_183796

/-- 
Theorem: In a right triangle with an inscribed circle of radius ρ 
and an angle bisector of length f for one of its acute angles, 
the condition f > √(8ρ) must hold.
-/
theorem right_triangle_bisector_inscribed_circle 
  (ρ f : ℝ) 
  (h_positive_ρ : ρ > 0) 
  (h_positive_f : f > 0) 
  (h_right_triangle : ∃ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    a^2 + b^2 = c^2 ∧ 
    (a * b) / (a + b + c) = ρ ∧
    f = (2 * a * b) / (a + b)) :
  f > Real.sqrt (8 * ρ) := by
sorry

end NUMINAMATH_CALUDE_right_triangle_bisector_inscribed_circle_l1837_183796


namespace NUMINAMATH_CALUDE_tan_angle_equality_l1837_183717

theorem tan_angle_equality (n : Int) :
  -90 < n ∧ n < 90 →
  Real.tan (n * Real.pi / 180) = Real.tan (75 * Real.pi / 180) →
  n = 75 := by
sorry

end NUMINAMATH_CALUDE_tan_angle_equality_l1837_183717


namespace NUMINAMATH_CALUDE_smallest_solution_congruence_l1837_183783

theorem smallest_solution_congruence :
  ∃! x : ℕ+, (5 * x.val ≡ 14 [ZMOD 26]) ∧
    ∀ y : ℕ+, (5 * y.val ≡ 14 [ZMOD 26]) → x ≤ y :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_solution_congruence_l1837_183783


namespace NUMINAMATH_CALUDE_inequalities_solution_l1837_183708

theorem inequalities_solution (x : ℝ) : 
  (2 * (-x + 2) > -3 * x + 5 → x > 1) ∧
  ((7 - x) / 3 ≤ (x + 2) / 2 + 1 → x ≥ 2 / 5) := by
sorry

end NUMINAMATH_CALUDE_inequalities_solution_l1837_183708


namespace NUMINAMATH_CALUDE_max_value_problem_l1837_183763

theorem max_value_problem (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) 
  (h4 : a^2 + b^2 + c^2 = 1) :
  (2 * a * b * Real.sqrt 3 + 2 * a * c) ≤ Real.sqrt 3 ∧ 
  ∃ a₀ b₀ c₀ : ℝ, 0 ≤ a₀ ∧ 0 ≤ b₀ ∧ 0 ≤ c₀ ∧ a₀^2 + b₀^2 + c₀^2 = 1 ∧
    2 * a₀ * b₀ * Real.sqrt 3 + 2 * a₀ * c₀ = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_max_value_problem_l1837_183763


namespace NUMINAMATH_CALUDE_triangle_longest_side_l1837_183706

/-- Given a triangle with sides 8, y+5, and 3y+2, and perimeter 45, the longest side is 24.5 -/
theorem triangle_longest_side (y : ℝ) :
  8 + (y + 5) + (3 * y + 2) = 45 →
  max 8 (max (y + 5) (3 * y + 2)) = 24.5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_longest_side_l1837_183706


namespace NUMINAMATH_CALUDE_fishing_competition_l1837_183740

theorem fishing_competition (n : ℕ) : 
  (∃ (m : ℕ), n * m + 11 * (m + 10) = n^2 + 5*n + 22) → n = 11 :=
by sorry

end NUMINAMATH_CALUDE_fishing_competition_l1837_183740


namespace NUMINAMATH_CALUDE_dice_sum_not_sixteen_l1837_183744

theorem dice_sum_not_sixteen (a b c d e : ℕ) : 
  1 ≤ a ∧ a ≤ 6 →
  1 ≤ b ∧ b ≤ 6 →
  1 ≤ c ∧ c ≤ 6 →
  1 ≤ d ∧ d ≤ 6 →
  1 ≤ e ∧ e ≤ 6 →
  a * b * c * d * e = 72 →
  a + b + c + d + e ≠ 16 := by
sorry

end NUMINAMATH_CALUDE_dice_sum_not_sixteen_l1837_183744


namespace NUMINAMATH_CALUDE_probability_at_least_three_aces_l1837_183762

/-- The number of cards in a standard deck without jokers -/
def deck_size : ℕ := 52

/-- The number of Aces in a standard deck -/
def num_aces : ℕ := 4

/-- The number of cards drawn -/
def draw_size : ℕ := 5

/-- The probability of drawing at least 3 Aces when randomly selecting 5 cards from a standard 52-card deck (without jokers) -/
theorem probability_at_least_three_aces :
  (Nat.choose num_aces 3 * Nat.choose (deck_size - num_aces) 2 +
   Nat.choose num_aces 4 * Nat.choose (deck_size - num_aces) 1) /
  Nat.choose deck_size draw_size =
  (Nat.choose 4 3 * Nat.choose 48 2 + Nat.choose 4 4 * Nat.choose 48 1) /
  Nat.choose 52 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_three_aces_l1837_183762


namespace NUMINAMATH_CALUDE_residue_theorem_l1837_183795

theorem residue_theorem (m k : ℕ) (hm : m > 0) (hk : k > 0) :
  (Nat.gcd m k = 1 →
    ∃ (a b : ℕ → ℕ),
      ∀ i j s t, 1 ≤ i ∧ i ≤ m ∧ 1 ≤ j ∧ j ≤ k ∧
                 1 ≤ s ∧ s ≤ m ∧ 1 ≤ t ∧ t ≤ k ∧
                 (i ≠ s ∨ j ≠ t) →
                 (a i * b j) % (m * k) ≠ (a s * b t) % (m * k)) ∧
  (Nat.gcd m k > 1 →
    ∀ (a b : ℕ → ℕ),
      ∃ i j s t, 1 ≤ i ∧ i ≤ m ∧ 1 ≤ j ∧ j ≤ k ∧
                 1 ≤ s ∧ s ≤ m ∧ 1 ≤ t ∧ t ≤ k ∧
                 (i ≠ s ∨ j ≠ t) ∧
                 (a i * b j) % (m * k) = (a s * b t) % (m * k)) :=
by sorry

end NUMINAMATH_CALUDE_residue_theorem_l1837_183795


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1837_183741

/-- The eccentricity of a hyperbola with specific conditions -/
theorem hyperbola_eccentricity (p a b m n : ℝ) (hp : p > 0) (ha : a > 0) (hb : b > 0) :
  let parabola := fun x y => y^2 = 2*p*x
  let hyperbola := fun x y => x^2/a^2 - y^2/b^2 = 1
  let focus : ℝ × ℝ := (p/2, 0)
  let A : ℝ × ℝ := (p/2, p)
  let B : ℝ × ℝ := (p/2, -p)
  let M : ℝ × ℝ := (p/2, b^2/a)
  (∀ x y, parabola x y → hyperbola x y → (x = p/2 ∧ y = 0)) →
  (m + n = 1) →
  (m - n = b^2/(a*p)) →
  (m * n = 1/8) →
  let e := c/a
  let c := Real.sqrt (a^2 + b^2)
  e = (Real.sqrt 6 + Real.sqrt 2) / 2 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1837_183741


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_expression_l1837_183777

theorem imaginary_part_of_complex_expression : 
  Complex.im ((1 - Complex.I) / (1 + Complex.I) * Complex.I) = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_expression_l1837_183777


namespace NUMINAMATH_CALUDE_sequence_formula_l1837_183718

def S (n : ℕ+) (a : ℕ+ → ℚ) : ℚ := 2 * n.val - a n

theorem sequence_formula (a : ℕ+ → ℚ) 
  (h : ∀ n : ℕ+, S n a = 2 * n.val - a n) : 
  ∀ n : ℕ+, a n = (2^n.val - 1) / 2^(n.val - 1) := by
sorry

end NUMINAMATH_CALUDE_sequence_formula_l1837_183718


namespace NUMINAMATH_CALUDE_square_sum_equals_one_l1837_183749

theorem square_sum_equals_one (a b : ℝ) 
  (h : a * Real.sqrt (1 - b^2) + b * Real.sqrt (1 - a^2) = 1) : 
  a^2 + b^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equals_one_l1837_183749


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_reciprocal_l1837_183746

theorem imaginary_part_of_complex_reciprocal (z : ℂ) (h : z = -2 + I) :
  (1 / z).im = -1 / 5 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_reciprocal_l1837_183746


namespace NUMINAMATH_CALUDE_shelter_final_count_l1837_183779

/-- Represents the number of cats in the shelter at different points in time. -/
structure CatCount where
  initial : Nat
  afterDoubling : Nat
  afterMonday : Nat
  afterTuesday : Nat
  afterWednesday : Nat
  afterThursday : Nat
  afterFriday : Nat
  afterReclaiming : Nat
  final : Nat

/-- Represents the events that occurred during the week at the animal shelter. -/
def shelterWeek (c : CatCount) : Prop :=
  c.afterDoubling = c.initial * 2 ∧
  c.afterDoubling = 48 ∧
  c.afterMonday = c.afterDoubling - 3 ∧
  c.afterTuesday = c.afterMonday + 5 ∧
  c.afterWednesday = c.afterTuesday - 3 ∧
  c.afterThursday = c.afterWednesday + 5 ∧
  c.afterFriday = c.afterThursday - 3 ∧
  c.afterReclaiming = c.afterFriday - 3 ∧
  c.final = c.afterReclaiming - 5

/-- Theorem stating that after the events of the week, the shelter has 41 cats. -/
theorem shelter_final_count (c : CatCount) :
  shelterWeek c → c.final = 41 := by
  sorry


end NUMINAMATH_CALUDE_shelter_final_count_l1837_183779


namespace NUMINAMATH_CALUDE_beths_sister_age_l1837_183736

theorem beths_sister_age (beth_age : ℕ) (future_years : ℕ) (sister_age : ℕ) : 
  beth_age = 18 → 
  future_years = 8 → 
  beth_age + future_years = 2 * (sister_age + future_years) → 
  sister_age = 5 := by
sorry

end NUMINAMATH_CALUDE_beths_sister_age_l1837_183736


namespace NUMINAMATH_CALUDE_career_preference_proof_l1837_183729

/-- Represents the ratio of boys to girls in a class -/
def boyGirlRatio : ℚ := 2/3

/-- Represents the fraction of the circle graph allocated to a specific career -/
def careerFraction : ℚ := 192/360

/-- Represents the fraction of girls who prefer the specific career -/
def girlPreferenceFraction : ℚ := 2/3

/-- Represents the fraction of boys who prefer the specific career -/
def boyPreferenceFraction : ℚ := 1/3

theorem career_preference_proof :
  let totalStudents := boyGirlRatio + 1
  let boyFraction := boyGirlRatio / totalStudents
  let girlFraction := 1 / totalStudents
  careerFraction = boyFraction * boyPreferenceFraction + girlFraction * girlPreferenceFraction :=
by sorry

end NUMINAMATH_CALUDE_career_preference_proof_l1837_183729


namespace NUMINAMATH_CALUDE_log_sum_equals_zero_l1837_183713

theorem log_sum_equals_zero :
  Real.log 2 + Real.log 5 + Real.log 0.5 / Real.log 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equals_zero_l1837_183713


namespace NUMINAMATH_CALUDE_pink_highlighters_l1837_183769

theorem pink_highlighters (total : ℕ) (yellow : ℕ) (blue : ℕ) (h1 : yellow = 2) (h2 : blue = 4) (h3 : total = 12) :
  total - yellow - blue = 6 := by
  sorry

end NUMINAMATH_CALUDE_pink_highlighters_l1837_183769


namespace NUMINAMATH_CALUDE_vector_properties_l1837_183759

/-- Given vectors in R^2 -/
def a : ℝ × ℝ := (2, -1)
def b (m : ℝ) : ℝ × ℝ := (m, 2)

/-- Parallel vectors condition -/
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 * w.2 = k * v.2 * w.1

/-- Perpendicular vectors condition -/
def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

/-- Vector addition -/
def add (v w : ℝ × ℝ) : ℝ × ℝ :=
  (v.1 + w.1, v.2 + w.2)

/-- Scalar multiplication -/
def smul (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  (k * v.1, k * v.2)

/-- Vector subtraction -/
def sub (v w : ℝ × ℝ) : ℝ × ℝ :=
  add v (smul (-1) w)

/-- Squared norm of a vector -/
def norm_sq (v : ℝ × ℝ) : ℝ :=
  v.1^2 + v.2^2

theorem vector_properties (m : ℝ) :
  (parallel a (b m) ↔ m = -4) ∧
  (perpendicular a (b m) ↔ m = 1) ∧
  ¬(norm_sq (sub (smul 2 a) (b m)) = norm_sq (add a (b m)) → m = 1) ∧
  ¬(norm_sq (add a (b m)) = norm_sq a → m = -4) := by
  sorry

end NUMINAMATH_CALUDE_vector_properties_l1837_183759


namespace NUMINAMATH_CALUDE_x_range_l1837_183730

theorem x_range (x : ℝ) : (1 / x < 4 ∧ 1 / x > -2) → (x < -1/2 ∨ x > 1/4) := by
  sorry

end NUMINAMATH_CALUDE_x_range_l1837_183730


namespace NUMINAMATH_CALUDE_inequality_proof_l1837_183702

theorem inequality_proof (a b c d : ℝ) (h : a * d - b * c = 1) :
  a^2 + b^2 + c^2 + d^2 + a * b + c * d ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1837_183702


namespace NUMINAMATH_CALUDE_diane_honey_harvest_l1837_183770

/-- Diane's honey harvest problem -/
theorem diane_honey_harvest 
  (last_year_harvest : ℕ) 
  (harvest_increase : ℕ) 
  (h1 : last_year_harvest = 2479)
  (h2 : harvest_increase = 6085) :
  last_year_harvest + harvest_increase = 8564 :=
by sorry

end NUMINAMATH_CALUDE_diane_honey_harvest_l1837_183770


namespace NUMINAMATH_CALUDE_number_problem_l1837_183752

theorem number_problem (x : ℚ) : (x = (3/8) * x + 40) → x = 64 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1837_183752


namespace NUMINAMATH_CALUDE_curve_properties_l1837_183748

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the parametric curve C -/
def C (a : ℝ) (t : ℝ) : Point where
  x := 1 + 2 * t
  y := a * t^2

/-- The point M lies on the curve C -/
def M : Point := ⟨5, 4⟩

theorem curve_properties (a : ℝ) :
  (∃ t, C a t = M) →
  (a = 1 ∧ ∀ x y, (C 1 ((x - 1) / 2)).y = y ↔ 4 * y = (x - 1)^2) := by
  sorry


end NUMINAMATH_CALUDE_curve_properties_l1837_183748


namespace NUMINAMATH_CALUDE_total_outlets_is_42_l1837_183767

/-- The number of rooms in the house -/
def num_rooms : ℕ := 7

/-- The number of outlets required per room -/
def outlets_per_room : ℕ := 6

/-- The total number of outlets needed for the house -/
def total_outlets : ℕ := num_rooms * outlets_per_room

/-- Theorem stating that the total number of outlets needed is 42 -/
theorem total_outlets_is_42 : total_outlets = 42 := by
  sorry

end NUMINAMATH_CALUDE_total_outlets_is_42_l1837_183767


namespace NUMINAMATH_CALUDE_start_time_is_6am_l1837_183781

/-- Represents the hiking scenario with two hikers --/
structure HikingScenario where
  meetTime : ℝ       -- Time when hikers meet (in hours after midnight)
  rychlyEndTime : ℝ  -- Time when Mr. Rychlý finishes (in hours after midnight)
  loudaEndTime : ℝ   -- Time when Mr. Louda finishes (in hours after midnight)

/-- Calculates the start time of the hike given a HikingScenario --/
def calculateStartTime (scenario : HikingScenario) : ℝ :=
  scenario.meetTime - (scenario.rychlyEndTime - scenario.meetTime)

/-- Theorem stating that the start time is 6 AM (6 hours after midnight) --/
theorem start_time_is_6am (scenario : HikingScenario) 
  (h1 : scenario.meetTime = 10)
  (h2 : scenario.rychlyEndTime = 12)
  (h3 : scenario.loudaEndTime = 18) :
  calculateStartTime scenario = 6 := by
  sorry

#eval calculateStartTime { meetTime := 10, rychlyEndTime := 12, loudaEndTime := 18 }

end NUMINAMATH_CALUDE_start_time_is_6am_l1837_183781


namespace NUMINAMATH_CALUDE_fraction_subtraction_simplification_l1837_183778

theorem fraction_subtraction_simplification :
  (8 : ℚ) / 29 - (5 : ℚ) / 87 = (19 : ℚ) / 87 ∧ 
  (∀ n d : ℤ, n ≠ 0 → (19 : ℚ) / 87 = (n : ℚ) / d → (abs n = 19 ∧ abs d = 87)) :=
by sorry

end NUMINAMATH_CALUDE_fraction_subtraction_simplification_l1837_183778


namespace NUMINAMATH_CALUDE_octagon_arc_length_l1837_183728

/-- The length of an arc intercepted by one side of a regular octagon inscribed in a circle -/
theorem octagon_arc_length (side_length : ℝ) (h : side_length = 5) :
  let circumference := 2 * Real.pi * side_length
  let arc_length := circumference / 8
  arc_length = 1.25 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_octagon_arc_length_l1837_183728


namespace NUMINAMATH_CALUDE_positive_integer_solution_exists_l1837_183712

theorem positive_integer_solution_exists (a : ℤ) (h : a > 2) :
  ∃ x y : ℤ, x > 0 ∧ y > 0 ∧ x^2 - y^2 = a^2 := by
  sorry

end NUMINAMATH_CALUDE_positive_integer_solution_exists_l1837_183712


namespace NUMINAMATH_CALUDE_mass_of_man_sinking_boat_l1837_183705

/-- The mass of a man who causes a boat to sink in water -/
theorem mass_of_man_sinking_boat (length width sinkage : ℝ) (water_density : ℝ) : 
  length = 4 →
  width = 2 →
  sinkage = 0.01 →
  water_density = 1000 →
  length * width * sinkage * water_density = 80 := by
sorry

end NUMINAMATH_CALUDE_mass_of_man_sinking_boat_l1837_183705


namespace NUMINAMATH_CALUDE_interest_difference_implies_principal_l1837_183764

/-- Prove that for a given principal amount, if the difference between compound
    interest (compounded annually) and simple interest over 2 years at 4% per annum
    is 1, then the principal amount is 625. -/
theorem interest_difference_implies_principal (P : ℝ) : 
  P * (1 + 0.04)^2 - P - (P * 0.04 * 2) = 1 → P = 625 := by
  sorry

end NUMINAMATH_CALUDE_interest_difference_implies_principal_l1837_183764


namespace NUMINAMATH_CALUDE_constant_function_from_square_plus_k_l1837_183794

/-- A continuous function satisfying f(x) = f(x² + k) for non-negative k is constant. -/
theorem constant_function_from_square_plus_k 
  (f : ℝ → ℝ) (hf : Continuous f) (k : ℝ) (hk : k ≥ 0) 
  (h : ∀ x, f x = f (x^2 + k)) : 
  ∃ C, ∀ x, f x = C :=
sorry

end NUMINAMATH_CALUDE_constant_function_from_square_plus_k_l1837_183794


namespace NUMINAMATH_CALUDE_ln_exp_relationship_l1837_183773

theorem ln_exp_relationship :
  (∀ x : ℝ, (Real.log x > 0) → (Real.exp x > 1)) ∧
  (∃ x : ℝ, Real.exp x > 1 ∧ Real.log x ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_ln_exp_relationship_l1837_183773


namespace NUMINAMATH_CALUDE_division_problem_l1837_183754

theorem division_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) 
  (h1 : dividend = 109)
  (h2 : divisor = 12)
  (h3 : remainder = 1)
  (h4 : dividend = divisor * quotient + remainder) :
  quotient = 9 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l1837_183754


namespace NUMINAMATH_CALUDE_min_tiles_required_l1837_183701

-- Define the dimensions
def tile_length : ℕ := 5
def tile_width : ℕ := 3
def floor_length_feet : ℕ := 5
def floor_width_feet : ℕ := 3

-- Convert feet to inches
def inches_per_foot : ℕ := 12

-- Calculate floor dimensions in inches
def floor_length_inches : ℕ := floor_length_feet * inches_per_foot
def floor_width_inches : ℕ := floor_width_feet * inches_per_foot

-- Calculate areas
def tile_area : ℕ := tile_length * tile_width
def floor_area : ℕ := floor_length_inches * floor_width_inches

-- Theorem to prove
theorem min_tiles_required : 
  (floor_area / tile_area : ℕ) = 144 := by sorry

end NUMINAMATH_CALUDE_min_tiles_required_l1837_183701

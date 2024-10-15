import Mathlib

namespace NUMINAMATH_CALUDE_tissue_usage_l2289_228969

theorem tissue_usage (initial_tissues : ℕ) (remaining_tissues : ℕ) 
  (alice_usage : ℕ) (bob_multiplier : ℕ) (eve_reduction : ℕ) : 
  initial_tissues = 97 →
  remaining_tissues = 47 →
  alice_usage = 12 →
  bob_multiplier = 2 →
  eve_reduction = 3 →
  initial_tissues - remaining_tissues + 
  alice_usage + bob_multiplier * alice_usage + (alice_usage - eve_reduction) = 95 :=
by sorry

end NUMINAMATH_CALUDE_tissue_usage_l2289_228969


namespace NUMINAMATH_CALUDE_parabola_intersection_l2289_228981

/-- 
Given a parabola y = 2x² translated right by p units and down by q units,
prove that it intersects y = x - 4 at exactly one point when p = q = 31/8.
-/
theorem parabola_intersection (p q : ℝ) : 
  (∃! x, 2*(x - p)^2 - q = x - 4) ↔ (p = 31/8 ∧ q = 31/8) := by
  sorry

#check parabola_intersection

end NUMINAMATH_CALUDE_parabola_intersection_l2289_228981


namespace NUMINAMATH_CALUDE_percentage_difference_l2289_228920

theorem percentage_difference : 
  (0.12 * 24.2) - (0.10 * 14.2) = 1.484 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l2289_228920


namespace NUMINAMATH_CALUDE_total_pencils_eq_twelve_l2289_228947

/-- The number of rows of pencils -/
def num_rows : ℕ := 3

/-- The number of pencils in each row -/
def pencils_per_row : ℕ := 4

/-- The total number of pencils -/
def total_pencils : ℕ := num_rows * pencils_per_row

theorem total_pencils_eq_twelve : total_pencils = 12 := by
  sorry

end NUMINAMATH_CALUDE_total_pencils_eq_twelve_l2289_228947


namespace NUMINAMATH_CALUDE_rain_forest_animals_l2289_228946

theorem rain_forest_animals (reptile_house : ℕ) (rain_forest : ℕ) : 
  reptile_house = 16 → 
  reptile_house = 3 * rain_forest - 5 → 
  rain_forest = 7 := by
sorry

end NUMINAMATH_CALUDE_rain_forest_animals_l2289_228946


namespace NUMINAMATH_CALUDE_intersection_complement_equals_singleton_l2289_228988

open Set

universe u

def U : Finset ℕ := {1, 2, 3, 4}

theorem intersection_complement_equals_singleton
  (A B : Finset ℕ)
  (h1 : A ⊆ U)
  (h2 : B ⊆ U)
  (h3 : (U \ (A ∪ B)) = {4})
  (h4 : B = {1, 2}) :
  A ∩ (U \ B) = {3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equals_singleton_l2289_228988


namespace NUMINAMATH_CALUDE_hyperbola_intersection_l2289_228998

/-- Hyperbola with given properties -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive : 0 < a ∧ 0 < b
  h_real_axis : a = 1
  h_focus : a^2 + b^2 = 5

/-- Line intersecting the hyperbola -/
def intersecting_line (x : ℝ) : ℝ := x + 2

/-- Theorem about the hyperbola and its intersecting line -/
theorem hyperbola_intersection (C : Hyperbola) :
  (∀ x y, x^2 / C.a^2 - y^2 / C.b^2 = 1 ↔ x^2 - y^2 / 4 = 1) ∧
  (∃ A B : ℝ × ℝ,
    A ≠ B ∧
    (A.1^2 - A.2^2 / 4 = 1) ∧
    (B.1^2 - B.2^2 / 4 = 1) ∧
    (A.2 = intersecting_line A.1) ∧
    (B.2 = intersecting_line B.1) ∧
    ((A.1 - B.1)^2 + (A.2 - B.2)^2 = 32 / 3)) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_intersection_l2289_228998


namespace NUMINAMATH_CALUDE_sock_pairs_count_l2289_228928

def white_socks : ℕ := 5
def brown_socks : ℕ := 4
def blue_socks : ℕ := 2
def red_socks : ℕ := 1

def total_socks : ℕ := white_socks + brown_socks + blue_socks + red_socks

theorem sock_pairs_count :
  (blue_socks * white_socks) + (blue_socks * brown_socks) + (blue_socks * red_socks) = 20 := by
  sorry

end NUMINAMATH_CALUDE_sock_pairs_count_l2289_228928


namespace NUMINAMATH_CALUDE_first_player_winning_strategy_l2289_228903

/-- Represents the state of the game with two piles of candies -/
structure GameState where
  p : Nat
  q : Nat

/-- Determines if a given number is a winning number (congruent to 0, 1, or 4 mod 5) -/
def isWinningNumber (n : Nat) : Prop :=
  n % 5 = 0 ∨ n % 5 = 1 ∨ n % 5 = 4

/-- Determines if a given game state is a winning state for the first player -/
def isWinningState (state : GameState) : Prop :=
  isWinningNumber state.p ∨ isWinningNumber state.q

/-- Theorem stating the winning condition for the first player -/
theorem first_player_winning_strategy (state : GameState) :
  (∃ (strategy : GameState → GameState), 
    (∀ (opponent_move : GameState → GameState), 
      strategy (opponent_move (strategy state)) = state)) ↔ 
  isWinningState state :=
sorry

end NUMINAMATH_CALUDE_first_player_winning_strategy_l2289_228903


namespace NUMINAMATH_CALUDE_quadratic_properties_l2289_228985

-- Define the quadratic function
def f (x : ℝ) : ℝ := 3 * x^2 - 9 * x + 2

-- Define the interval
def interval : Set ℝ := Set.Icc 1 4

theorem quadratic_properties :
  ∃ (x_min : ℝ), x_min ∈ interval ∧ 
  (∀ (x : ℝ), x ∈ interval → f x_min ≤ f x) ∧
  (∀ (x y : ℝ), x ∈ Set.Icc 1 1.5 → y ∈ Set.Icc 1 1.5 → x ≤ y → f x ≥ f y) ∧
  (∀ (x y : ℝ), x ∈ Set.Icc 1.5 4 → y ∈ Set.Icc 1.5 4 → x ≤ y → f x ≤ f y) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_properties_l2289_228985


namespace NUMINAMATH_CALUDE_min_sum_squares_l2289_228934

theorem min_sum_squares (y₁ y₂ y₃ : ℝ) (h_pos : y₁ > 0 ∧ y₂ > 0 ∧ y₃ > 0) 
    (h_sum : 3 * y₁ + 2 * y₂ + y₃ = 30) : 
  y₁^2 + y₂^2 + y₃^2 ≥ 450/7 ∧ ∃ y₁' y₂' y₃', y₁'^2 + y₂'^2 + y₃'^2 = 450/7 ∧ 
    y₁' > 0 ∧ y₂' > 0 ∧ y₃' > 0 ∧ 3 * y₁' + 2 * y₂' + y₃' = 30 :=
by sorry


end NUMINAMATH_CALUDE_min_sum_squares_l2289_228934


namespace NUMINAMATH_CALUDE_team_games_count_l2289_228949

/-- Proves that a team playing under specific win conditions played 120 games in total -/
theorem team_games_count (first_games : ℕ) (first_win_rate : ℚ) (remaining_win_rate : ℚ) (total_win_rate : ℚ) : 
  first_games = 30 →
  first_win_rate = 2/5 →
  remaining_win_rate = 4/5 →
  total_win_rate = 7/10 →
  ∃ (total_games : ℕ), 
    total_games = 120 ∧
    (first_win_rate * first_games + remaining_win_rate * (total_games - first_games) : ℚ) = total_win_rate * total_games :=
by sorry

end NUMINAMATH_CALUDE_team_games_count_l2289_228949


namespace NUMINAMATH_CALUDE_lcm_of_462_and_150_l2289_228961

theorem lcm_of_462_and_150 :
  let a : ℕ := 462
  let b : ℕ := 150
  let hcf : ℕ := 30
  Nat.lcm a b = 2310 :=
by
  sorry

end NUMINAMATH_CALUDE_lcm_of_462_and_150_l2289_228961


namespace NUMINAMATH_CALUDE_fraction_ceiling_evaluation_l2289_228990

theorem fraction_ceiling_evaluation : 
  (⌈(23 : ℚ) / 11 - ⌈(31 : ℚ) / 19⌉⌉) / (⌈(35 : ℚ) / 9 + ⌈(9 * 19 : ℚ) / 35⌉⌉) = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_ceiling_evaluation_l2289_228990


namespace NUMINAMATH_CALUDE_negation_of_forall_positive_negation_of_greater_than_zero_l2289_228953

theorem negation_of_forall_positive (p : ℝ → Prop) : 
  (¬ ∀ x : ℝ, p x) ↔ (∃ x : ℝ, ¬ p x) :=
by sorry

theorem negation_of_greater_than_zero :
  (¬ ∀ x : ℝ, x^2 + x + 2 > 0) ↔ (∃ x : ℝ, x^2 + x + 2 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_forall_positive_negation_of_greater_than_zero_l2289_228953


namespace NUMINAMATH_CALUDE_equation_satisfies_condition_l2289_228999

theorem equation_satisfies_condition (x y z : ℤ) : 
  x = z ∧ y = x - 1 → x * (x - y) + y * (y - z) + z * (z - x) = 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_satisfies_condition_l2289_228999


namespace NUMINAMATH_CALUDE_sum_of_coefficients_sum_of_coefficients_is_negative_nineteen_l2289_228970

theorem sum_of_coefficients (x : ℝ) : 
  3 * (x^8 - 2*x^5 + x^3 - 7) - 5 * (x^6 + 3*x^2 - 6) + 2 * (x^4 - 5) = 
  3*x^8 - 5*x^6 - 6*x^5 + 2*x^4 + 3*x^3 - 15*x^2 - 1 :=
by sorry

theorem sum_of_coefficients_is_negative_nineteen : 
  (3 * (1^8 - 2*1^5 + 1^3 - 7) - 5 * (1^6 + 3*1^2 - 6) + 2 * (1^4 - 5)) = -19 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_sum_of_coefficients_is_negative_nineteen_l2289_228970


namespace NUMINAMATH_CALUDE_four_links_sufficient_l2289_228902

/-- Represents a chain of links -/
structure Chain :=
  (length : ℕ)
  (link_weight : ℕ)

/-- Represents the ability to create all weights up to the chain's total weight -/
def can_create_all_weights (c : Chain) (separated_links : ℕ) : Prop :=
  ∀ w : ℕ, w ≤ c.length → ∃ (subset : Finset ℕ), 
    subset.card ≤ separated_links ∧ 
    (subset.sum (λ _ => c.link_weight) = w ∨ 
     ∃ (remaining : Finset ℕ), remaining.card + subset.card = c.length ∧ 
       remaining.sum (λ _ => c.link_weight) = w)

/-- The main theorem stating that separating 4 links is sufficient for a chain of 150 links -/
theorem four_links_sufficient (c : Chain) (h1 : c.length = 150) (h2 : c.link_weight = 1) : 
  can_create_all_weights c 4 := by
sorry

end NUMINAMATH_CALUDE_four_links_sufficient_l2289_228902


namespace NUMINAMATH_CALUDE_total_nail_polishes_l2289_228987

/-- The number of nail polishes each person has -/
structure NailPolishes where
  kim : ℕ
  heidi : ℕ
  karen : ℕ
  laura : ℕ
  simon : ℕ

/-- The conditions of the nail polish problem -/
def nail_polish_conditions (np : NailPolishes) : Prop :=
  np.kim = 25 ∧
  np.heidi = np.kim + 8 ∧
  np.karen = np.kim - 6 ∧
  np.laura = 2 * np.kim ∧
  np.simon = (np.kim / 2 + 10)

/-- The theorem stating the total number of nail polishes -/
theorem total_nail_polishes (np : NailPolishes) :
  nail_polish_conditions np →
  np.heidi + np.karen + np.laura + np.simon = 125 :=
by
  sorry


end NUMINAMATH_CALUDE_total_nail_polishes_l2289_228987


namespace NUMINAMATH_CALUDE_painting_scheme_combinations_l2289_228908

def number_of_color_choices : ℕ := 10
def colors_to_choose : ℕ := 2
def number_of_texture_choices : ℕ := 3
def textures_to_choose : ℕ := 1

theorem painting_scheme_combinations :
  (number_of_color_choices.choose colors_to_choose) * (number_of_texture_choices.choose textures_to_choose) = 135 := by
  sorry

end NUMINAMATH_CALUDE_painting_scheme_combinations_l2289_228908


namespace NUMINAMATH_CALUDE_markup_rate_calculation_l2289_228945

/-- Represents the markup rate calculation for a product with given profit and expense percentages. -/
theorem markup_rate_calculation (profit_percent : ℝ) (expense_percent : ℝ) :
  profit_percent = 0.12 →
  expense_percent = 0.18 →
  let cost_percent := 1 - profit_percent - expense_percent
  let markup_rate := (1 / cost_percent - 1) * 100
  ∃ ε > 0, abs (markup_rate - 42.857) < ε :=
by
  sorry

end NUMINAMATH_CALUDE_markup_rate_calculation_l2289_228945


namespace NUMINAMATH_CALUDE_two_digit_sum_square_property_l2289_228960

def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

def satisfiesCondition (A : ℕ) : Prop :=
  (sumOfDigits A)^2 = sumOfDigits (A^2)

def isTwoDigit (A : ℕ) : Prop :=
  10 ≤ A ∧ A ≤ 99

theorem two_digit_sum_square_property :
  ∀ A : ℕ, isTwoDigit A →
    (satisfiesCondition A ↔ 
      A = 10 ∨ A = 11 ∨ A = 12 ∨ A = 13 ∨ A = 20 ∨ A = 21 ∨ A = 22 ∨ A = 30 ∨ A = 31) :=
by sorry

end NUMINAMATH_CALUDE_two_digit_sum_square_property_l2289_228960


namespace NUMINAMATH_CALUDE_quadratic_root_triple_l2289_228991

/-- 
For a quadratic equation ax^2 + bx + c = 0, if one root is triple the other, 
then 3b^2 = 16ac.
-/
theorem quadratic_root_triple (a b c : ℝ) (x₁ x₂ : ℝ) : 
  a ≠ 0 →
  a * x₁^2 + b * x₁ + c = 0 →
  a * x₂^2 + b * x₂ + c = 0 →
  x₂ = 3 * x₁ →
  3 * b^2 = 16 * a * c :=
by sorry


end NUMINAMATH_CALUDE_quadratic_root_triple_l2289_228991


namespace NUMINAMATH_CALUDE_friend_c_spent_26_l2289_228952

/-- Friend C's lunch cost given the conditions of the problem -/
def friend_c_cost (your_cost friend_a_extra friend_b_less : ℕ) : ℕ :=
  2 * (your_cost + friend_a_extra - friend_b_less)

/-- Theorem stating that Friend C's lunch cost is $26 -/
theorem friend_c_spent_26 : friend_c_cost 12 4 3 = 26 := by
  sorry

end NUMINAMATH_CALUDE_friend_c_spent_26_l2289_228952


namespace NUMINAMATH_CALUDE_tan_alpha_minus_pi_sixth_l2289_228992

theorem tan_alpha_minus_pi_sixth (α : Real) : 
  (∃ (x y : Real), x = -Real.sqrt 3 ∧ y = 2 ∧ 
   Real.tan α = y / x) →
  Real.tan (α - π/6) = -3 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_tan_alpha_minus_pi_sixth_l2289_228992


namespace NUMINAMATH_CALUDE_gold_coin_problem_l2289_228936

theorem gold_coin_problem (n : ℕ) (c : ℕ) : 
  (n = 10 * (c - 4)) →
  (n = 7 * c + 5) →
  n = 110 := by
sorry

end NUMINAMATH_CALUDE_gold_coin_problem_l2289_228936


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_247_l2289_228905

theorem greatest_prime_factor_of_247 : ∃ p : ℕ, 
  Nat.Prime p ∧ p ∣ 247 ∧ ∀ q : ℕ, Nat.Prime q → q ∣ 247 → q ≤ p :=
sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_of_247_l2289_228905


namespace NUMINAMATH_CALUDE_room_width_l2289_228918

/-- Given a rectangular room with the specified length and paving cost, prove its width. -/
theorem room_width (length : ℝ) (total_cost : ℝ) (rate : ℝ) (width : ℝ) 
  (h1 : length = 5.5)
  (h2 : total_cost = 20625)
  (h3 : rate = 1000)
  (h4 : total_cost = rate * length * width) :
  width = 3.75 := by sorry

end NUMINAMATH_CALUDE_room_width_l2289_228918


namespace NUMINAMATH_CALUDE_pizza_slices_per_friend_l2289_228924

theorem pizza_slices_per_friend (num_friends : ℕ) (total_slices : ℕ) (h1 : num_friends = 4) (h2 : total_slices = 16) :
  ∃ (slices_per_friend : ℕ),
    slices_per_friend * num_friends = total_slices ∧
    slices_per_friend = 4 := by
  sorry

end NUMINAMATH_CALUDE_pizza_slices_per_friend_l2289_228924


namespace NUMINAMATH_CALUDE_simplify_expression_l2289_228909

/-- Proves that the simplified expression (√3 - 1)^(1 - √2) / (√3 + 1)^(1 + √2) equals 2^(1-√2)(4 - 2√3) -/
theorem simplify_expression :
  let x := (Real.sqrt 3 - 1)^(1 - Real.sqrt 2) / (Real.sqrt 3 + 1)^(1 + Real.sqrt 2)
  let y := 2^(1 - Real.sqrt 2) * (4 - 2 * Real.sqrt 3)
  x = y := by sorry

end NUMINAMATH_CALUDE_simplify_expression_l2289_228909


namespace NUMINAMATH_CALUDE_vector_equation_and_parallelism_l2289_228940

def a : ℝ × ℝ := (3, 2)
def b : ℝ × ℝ := (-1, 2)
def c : ℝ × ℝ := (4, 1)

theorem vector_equation_and_parallelism :
  (a = (5/9 : ℝ) • b + (8/9 : ℝ) • c) ∧
  (∃ (t : ℝ), t • (a + (-16/13 : ℝ) • c) = 2 • b - a) :=
by sorry

end NUMINAMATH_CALUDE_vector_equation_and_parallelism_l2289_228940


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l2289_228989

theorem absolute_value_inequality_solution_set :
  {x : ℝ | x ≠ 0 ∧ |((x - 2) / x)| > (x - 2) / x} = Set.Ioo 0 2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l2289_228989


namespace NUMINAMATH_CALUDE_dessert_eating_contest_l2289_228967

theorem dessert_eating_contest (student1_pie : ℚ) (student2_pie : ℚ) (student3_cake : ℚ) :
  student1_pie = 5/6 ∧ student2_pie = 7/8 ∧ student3_cake = 1/2 →
  max student1_pie student2_pie - student3_cake = 1/3 :=
by sorry

end NUMINAMATH_CALUDE_dessert_eating_contest_l2289_228967


namespace NUMINAMATH_CALUDE_triangle_side_ratio_l2289_228986

theorem triangle_side_ratio (a b c q : ℝ) : 
  c = b * q ∧ c = a * q^2 → ((Real.sqrt 5 - 1) / 2 < q ∧ q < (Real.sqrt 5 + 1) / 2) :=
sorry

end NUMINAMATH_CALUDE_triangle_side_ratio_l2289_228986


namespace NUMINAMATH_CALUDE_remainder_invariance_l2289_228913

theorem remainder_invariance (n : ℤ) : (n + 22) % 9 = 2 → (n + 31) % 9 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_invariance_l2289_228913


namespace NUMINAMATH_CALUDE_edwards_initial_money_l2289_228958

theorem edwards_initial_money :
  ∀ (initial_book_cost : ℝ) (discount_rate : ℝ) (num_books : ℕ) (pen_cost : ℝ) (num_pens : ℕ) (money_left : ℝ),
    initial_book_cost = 40 →
    discount_rate = 0.25 →
    num_books = 100 →
    pen_cost = 2 →
    num_pens = 3 →
    money_left = 6 →
    ∃ (initial_money : ℝ),
      initial_money = initial_book_cost * (1 - discount_rate) + (pen_cost * num_pens) + money_left ∧
      initial_money = 42 :=
by sorry

end NUMINAMATH_CALUDE_edwards_initial_money_l2289_228958


namespace NUMINAMATH_CALUDE_gcd_153_119_l2289_228907

theorem gcd_153_119 : Nat.gcd 153 119 = 17 := by
  sorry

end NUMINAMATH_CALUDE_gcd_153_119_l2289_228907


namespace NUMINAMATH_CALUDE_study_group_composition_l2289_228910

def number_of_selections (n m : ℕ) : ℕ :=
  (Nat.choose n 2) * (Nat.choose m 1) * 6

theorem study_group_composition :
  ∃ (n m : ℕ),
    n + m = 8 ∧
    number_of_selections n m = 90 ∧
    n = 3 ∧
    m = 5 := by
  sorry

end NUMINAMATH_CALUDE_study_group_composition_l2289_228910


namespace NUMINAMATH_CALUDE_band_total_earnings_l2289_228901

/-- Calculates the total earnings of a band given the number of members, 
    earnings per member per gig, and number of gigs played. -/
def bandEarnings (members : ℕ) (earningsPerMember : ℕ) (gigs : ℕ) : ℕ :=
  members * earningsPerMember * gigs

/-- Theorem stating that a band with 4 members, each earning $20 per gig, 
    and having played 5 gigs, earns a total of $400. -/
theorem band_total_earnings : 
  bandEarnings 4 20 5 = 400 := by
  sorry

end NUMINAMATH_CALUDE_band_total_earnings_l2289_228901


namespace NUMINAMATH_CALUDE_max_value_of_f_l2289_228916

noncomputable section

variable (a : ℝ)
variable (x : ℝ)

def f (x : ℝ) : ℝ := a * x^2 * Real.exp x

theorem max_value_of_f (h : a ≠ 0) :
  (a > 0 → ∃ (M : ℝ), M = 4 * a * Real.exp (-2) ∧ ∀ x, f a x ≤ M) ∧
  (a < 0 → ∃ (M : ℝ), M = 0 ∧ ∀ x, f a x ≤ M) :=
sorry

end

end NUMINAMATH_CALUDE_max_value_of_f_l2289_228916


namespace NUMINAMATH_CALUDE_shekar_average_marks_l2289_228955

def shekar_scores : List ℕ := [76, 65, 82, 62, 85]

theorem shekar_average_marks :
  (shekar_scores.sum / shekar_scores.length : ℚ) = 74 := by
  sorry

end NUMINAMATH_CALUDE_shekar_average_marks_l2289_228955


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2289_228957

theorem quadratic_equation_roots (x : ℝ) : 
  (∃ r₁ r₂ : ℝ, r₁ ≠ r₂ ∧ x * (x - 2) = x - 2 → x = r₁ ∨ x = r₂) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2289_228957


namespace NUMINAMATH_CALUDE_discriminant_always_positive_roots_as_triangle_legs_m_values_l2289_228927

-- Define the quadratic equation
def quadratic_eq (m x : ℝ) : ℝ := x^2 - (2+3*m)*x + 2*m^2 + 5*m - 4

-- Define the discriminant of the quadratic equation
def discriminant (m : ℝ) : ℝ := (2+3*m)^2 - 4*(2*m^2 + 5*m - 4)

-- Theorem 1: The discriminant is always positive for any real m
theorem discriminant_always_positive (m : ℝ) : discriminant m > 0 := by
  sorry

-- Define the condition for the roots being legs of a right-angled triangle
def roots_are_triangle_legs (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, 
    quadratic_eq m x₁ = 0 ∧ 
    quadratic_eq m x₂ = 0 ∧ 
    x₁^2 + x₂^2 = (2 * Real.sqrt 7)^2

-- Theorem 2: When the roots are legs of a right-angled triangle with hypotenuse 2√7, m is either -2 or 8/5
theorem roots_as_triangle_legs_m_values : 
  ∀ m : ℝ, roots_are_triangle_legs m → (m = -2 ∨ m = 8/5) := by
  sorry

end NUMINAMATH_CALUDE_discriminant_always_positive_roots_as_triangle_legs_m_values_l2289_228927


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l2289_228997

theorem trigonometric_equation_solution (n : ℤ) :
  let f (x : ℝ) := (Real.sin x) ^ (Real.arctan (Real.sin x + Real.cos x))
  let g (x : ℝ) := (1 / Real.sin x) ^ (Real.arctan (Real.sin (2 * x)) + π / 4)
  let x₁ := 2 * n * π + π / 2
  let x₂ := 2 * n * π + 3 * π / 4
  ∀ x ∈ Set.Ioo (2 * n * π) ((2 * n + 1) * π),
    f x = g x ↔ (x = x₁ ∨ x = x₂) := by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l2289_228997


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_arithmetic_sequence_l2289_228942

/-- 
Given an ellipse with major axis length 2a, minor axis length 2b, and focal length 2c,
where these lengths form an arithmetic sequence, prove that the eccentricity is 3/5.
-/
theorem ellipse_eccentricity_arithmetic_sequence 
  (a b c : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_arithmetic : 2 * b = a + c)
  (h_ellipse : b^2 = a^2 - c^2)
  (e : ℝ) 
  (h_eccentricity : e = c / a) :
  e = 3/5 := by
sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_arithmetic_sequence_l2289_228942


namespace NUMINAMATH_CALUDE_sum_of_digits_2003n_l2289_228931

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem statement -/
theorem sum_of_digits_2003n (n : ℕ) 
  (h_pos : n > 0)
  (h_sum_n : sum_of_digits n = 111)
  (h_sum_7002n : sum_of_digits (7002 * n) = 990) : 
  sum_of_digits (2003 * n) = 555 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_2003n_l2289_228931


namespace NUMINAMATH_CALUDE_tutor_schedule_lcm_l2289_228926

theorem tutor_schedule_lcm : Nat.lcm (Nat.lcm (Nat.lcm 5 6) 9) 10 = 90 := by
  sorry

end NUMINAMATH_CALUDE_tutor_schedule_lcm_l2289_228926


namespace NUMINAMATH_CALUDE_smallest_four_digit_solution_l2289_228943

def is_valid (x : ℕ) : Prop :=
  (3 * x) % 12 = 6 ∧
  (5 * x + 20) % 15 = 25 ∧
  (3 * x - 2) % 35 = (2 * x) % 35

def is_four_digit (x : ℕ) : Prop :=
  1000 ≤ x ∧ x ≤ 9999

theorem smallest_four_digit_solution :
  is_valid 1274 ∧ is_four_digit 1274 ∧
  ∀ y : ℕ, (is_valid y ∧ is_four_digit y) → 1274 ≤ y :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_solution_l2289_228943


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fourth_term_l2289_228914

/-- Given an arithmetic sequence where the sum of the third and fifth terms is 10,
    prove that the fourth term is 5. -/
theorem arithmetic_sequence_fourth_term
  (a : ℕ → ℝ)  -- The arithmetic sequence
  (h_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))  -- Arithmetic sequence condition
  (h_sum : a 3 + a 5 = 10)  -- Sum of third and fifth terms is 10
  : a 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fourth_term_l2289_228914


namespace NUMINAMATH_CALUDE_rectangle_area_l2289_228994

theorem rectangle_area (L B : ℝ) 
  (h1 : L - B = 23) 
  (h2 : 2 * L + 2 * B = 226) : 
  L * B = 3060 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_l2289_228994


namespace NUMINAMATH_CALUDE_inseparable_triangles_exist_l2289_228982

-- Define a Point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a Triangle in 3D space
structure Triangle3D where
  a : Point3D
  b : Point3D
  c : Point3D

-- Define a function to check if two triangles can be separated by a plane
def canBeSeparated (t1 t2 : Triangle3D) : Prop :=
  ∃ (a b c d : ℝ), ∀ (p : Point3D),
    (p = t1.a ∨ p = t1.b ∨ p = t1.c) →
      a * p.x + b * p.y + c * p.z + d > 0 ∧
    (p = t2.a ∨ p = t2.b ∨ p = t2.c) →
      a * p.x + b * p.y + c * p.z + d < 0

-- Theorem statement
theorem inseparable_triangles_exist (points : Fin 6 → Point3D) :
  ∃ (t1 t2 : Triangle3D),
    (∀ i j k : Fin 6, i ≠ j ∧ j ≠ k ∧ i ≠ k →
      (t1 = Triangle3D.mk (points i) (points j) (points k) ∨
       t2 = Triangle3D.mk (points i) (points j) (points k))) ∧
    ¬(canBeSeparated t1 t2) :=
sorry

end NUMINAMATH_CALUDE_inseparable_triangles_exist_l2289_228982


namespace NUMINAMATH_CALUDE_factory_sampling_is_systematic_l2289_228938

/-- Represents a sampling method -/
inductive SamplingMethod
| Simple
| Stratified
| Systematic

/-- Represents a sampling scenario -/
structure SamplingScenario where
  totalItems : Nat
  sampleSize : Nat
  method : SamplingMethod

/-- Determines if a sampling scenario is suitable for systematic sampling -/
def isSuitableForSystematic (scenario : SamplingScenario) : Prop :=
  scenario.method = SamplingMethod.Systematic ∧
  scenario.totalItems ≥ scenario.sampleSize ∧
  scenario.totalItems % scenario.sampleSize = 0

/-- The given sampling scenario -/
def factorySampling : SamplingScenario :=
  { totalItems := 2000
    sampleSize := 200
    method := SamplingMethod.Systematic }

/-- Theorem stating that the factory sampling scenario is suitable for systematic sampling -/
theorem factory_sampling_is_systematic :
  isSuitableForSystematic factorySampling :=
by
  sorry


end NUMINAMATH_CALUDE_factory_sampling_is_systematic_l2289_228938


namespace NUMINAMATH_CALUDE_exponent_multiplication_l2289_228972

theorem exponent_multiplication (a : ℝ) : a^3 * a^2 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l2289_228972


namespace NUMINAMATH_CALUDE_geometric_sequences_theorem_l2289_228911

/-- Two geometric sequences satisfying given conditions -/
structure GeometricSequences where
  a : ℕ → ℝ
  b : ℕ → ℝ
  a_pos : a 1 > 0
  a_geom : ∀ n : ℕ, a (n + 1) = a n * (a 2 / a 1)
  b_geom : ∀ n : ℕ, b (n + 1) = b n * (b 2 / b 1)
  diff_1 : b 1 - a 1 = 1
  diff_2 : b 2 - a 2 = 2
  diff_3 : b 3 - a 3 = 3

/-- The unique value of a_1 in the geometric sequences -/
def unique_a (gs : GeometricSequences) : ℝ := gs.a 1

/-- The statement to be proved -/
theorem geometric_sequences_theorem (gs : GeometricSequences) :
  unique_a gs = 1/3 ∧
  ¬∃ (a b : ℕ → ℝ) (q₁ q₂ : ℝ),
    (∀ n, a (n + 1) = a n * q₁) ∧
    (∀ n, b (n + 1) = b n * q₂) ∧
    ∃ (d : ℝ), d ≠ 0 ∧
    ∀ n : ℕ, n ≤ 3 →
      (b (n + 1) - a (n + 1)) - (b n - a n) = d :=
sorry

end NUMINAMATH_CALUDE_geometric_sequences_theorem_l2289_228911


namespace NUMINAMATH_CALUDE_largest_prime_divisor_check_l2289_228939

theorem largest_prime_divisor_check (n : ℕ) : 
  1200 ≤ n ∧ n ≤ 1250 → 
  (∀ p : ℕ, p.Prime → p ≤ 31 → n % p ≠ 0) → 
  n.Prime := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_check_l2289_228939


namespace NUMINAMATH_CALUDE_shopping_total_l2289_228975

def tuesday_discount : ℝ := 0.1
def jimmy_shorts_count : ℕ := 3
def jimmy_shorts_price : ℝ := 15
def irene_shirts_count : ℕ := 5
def irene_shirts_price : ℝ := 17

theorem shopping_total : 
  let total_before_discount := jimmy_shorts_count * jimmy_shorts_price + 
                               irene_shirts_count * irene_shirts_price
  let discount := total_before_discount * tuesday_discount
  let final_amount := total_before_discount - discount
  final_amount = 117 := by sorry

end NUMINAMATH_CALUDE_shopping_total_l2289_228975


namespace NUMINAMATH_CALUDE_scarf_problem_l2289_228948

theorem scarf_problem (initial_scarves : ℕ) (num_girls : ℕ) (final_scarves : ℕ) : 
  initial_scarves = 20 →
  num_girls = 17 →
  final_scarves ≠ 10 :=
by
  intro h_initial h_girls
  sorry


end NUMINAMATH_CALUDE_scarf_problem_l2289_228948


namespace NUMINAMATH_CALUDE_chicago_bulls_wins_conditions_satisfied_l2289_228932

/-- The number of games won by the Chicago Bulls -/
def bulls_wins : ℕ := 70

/-- The number of games won by the Miami Heat -/
def heat_wins : ℕ := bulls_wins + 5

/-- The total number of games won by both teams -/
def total_wins : ℕ := 145

/-- Theorem stating that the Chicago Bulls won 70 games -/
theorem chicago_bulls_wins : bulls_wins = 70 := by sorry

/-- Theorem proving the conditions are satisfied -/
theorem conditions_satisfied :
  (heat_wins = bulls_wins + 5) ∧ (bulls_wins + heat_wins = total_wins) := by sorry

end NUMINAMATH_CALUDE_chicago_bulls_wins_conditions_satisfied_l2289_228932


namespace NUMINAMATH_CALUDE_environmental_group_allocation_l2289_228919

theorem environmental_group_allocation :
  let total_members : ℕ := 8
  let num_locations : ℕ := 3
  let min_per_location : ℕ := 2

  let allocation_schemes : ℕ := 
    (Nat.choose total_members 2 * Nat.choose 6 2 * Nat.choose 4 4 / 2 +
     Nat.choose total_members 3 * Nat.choose 5 3 * Nat.choose 2 2 / 2) * 
    (Nat.factorial num_locations)

  allocation_schemes = 2940 :=
by sorry

end NUMINAMATH_CALUDE_environmental_group_allocation_l2289_228919


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2289_228995

/-- An arithmetic sequence of 5 terms starting with 3 and ending with 15 -/
def ArithmeticSequence (a b c : ℝ) : Prop :=
  ∃ d : ℝ, (a = 3 + d) ∧ (b = 3 + 2*d) ∧ (c = 3 + 3*d) ∧ (15 = 3 + 4*d)

/-- The sum of the middle three terms of the arithmetic sequence is 27 -/
theorem arithmetic_sequence_sum (a b c : ℝ) 
  (h : ArithmeticSequence a b c) : a + b + c = 27 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2289_228995


namespace NUMINAMATH_CALUDE_inscribed_circle_exists_l2289_228935

-- Define a convex polygon
def ConvexPolygon : Type := sorry

-- Define the area of a polygon
def area (p : ConvexPolygon) : ℝ := sorry

-- Define the perimeter of a polygon
def perimeter (p : ConvexPolygon) : ℝ := sorry

-- Define a point inside a polygon
def PointInside (p : ConvexPolygon) : Type := sorry

-- Define the distance from a point to a side of the polygon
def distanceToSide (point : PointInside p) (side : sorry) : ℝ := sorry

-- Theorem statement
theorem inscribed_circle_exists (p : ConvexPolygon) (h : area p > 0) :
  ∃ (center : PointInside p), ∀ (side : sorry),
    distanceToSide center side ≥ (area p) / (perimeter p) := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_exists_l2289_228935


namespace NUMINAMATH_CALUDE_monochromatic_right_triangle_exists_l2289_228971

-- Define the color type
inductive Color
| Black
| White

-- Define a point type
structure Point where
  x : ℝ
  y : ℝ

-- Define an equilateral triangle
structure EquilateralTriangle where
  A : Point
  B : Point
  C : Point
  is_equilateral : sorry -- Condition that ABC is equilateral

-- Define the set G of points on the sides of the triangle
def G (t : EquilateralTriangle) : Set Point :=
  sorry -- Definition of G as described in the problem

-- Define a coloring function
def coloring (p : Point) : Color :=
  sorry -- Some function that assigns Black or White to each point

-- Define what it means for a triangle to be inscribed and right-angled
def is_inscribed_right_triangle (t : EquilateralTriangle) (p q r : Point) : Prop :=
  sorry -- Condition for p, q, r to form an inscribed right triangle in t

-- The main theorem
theorem monochromatic_right_triangle_exists (t : EquilateralTriangle) :
  ∃ p q r : Point, p ∈ G t ∧ q ∈ G t ∧ r ∈ G t ∧
  is_inscribed_right_triangle t p q r ∧
  coloring p = coloring q ∧ coloring q = coloring r :=
sorry

end NUMINAMATH_CALUDE_monochromatic_right_triangle_exists_l2289_228971


namespace NUMINAMATH_CALUDE_ann_speed_l2289_228921

/-- Given cyclists' speeds, prove Ann's speed -/
theorem ann_speed (tom_speed : ℚ) (jerry_speed : ℚ) (ann_speed : ℚ) : 
  tom_speed = 6 →
  jerry_speed = 3/4 * tom_speed →
  ann_speed = 4/3 * jerry_speed →
  ann_speed = 6 := by
sorry

end NUMINAMATH_CALUDE_ann_speed_l2289_228921


namespace NUMINAMATH_CALUDE_smallest_m_is_26_l2289_228925

def S : Finset Nat := Finset.range 100

-- Define the property we want to prove
def has_divisor (A : Finset Nat) : Prop :=
  ∃ x ∈ A, x ∣ (A.prod (fun y => if y ≠ x then y else 1))

theorem smallest_m_is_26 : 
  (∀ A : Finset Nat, A ⊆ S → A.card = 26 → has_divisor A) ∧ 
  (∀ m < 26, ∃ A : Finset Nat, A ⊆ S ∧ A.card = m ∧ ¬has_divisor A) :=
sorry

end NUMINAMATH_CALUDE_smallest_m_is_26_l2289_228925


namespace NUMINAMATH_CALUDE_unique_extremum_implies_a_range_l2289_228929

noncomputable def f (a x : ℝ) : ℝ := a * (x - 2) * Real.exp x + Real.log x + 1 / x

theorem unique_extremum_implies_a_range (a : ℝ) :
  (∃! x, ∀ y, f a y ≤ f a x) →
  (∃ x, ∀ y, f a y ≤ f a x ∧ f a x > 0) →
  0 ≤ a ∧ a < 1 / Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_unique_extremum_implies_a_range_l2289_228929


namespace NUMINAMATH_CALUDE_intersection_M_P_l2289_228900

-- Define the sets M and P
def M : Set ℝ := {y : ℝ | ∃ x : ℝ, y = 2^x}
def P : Set ℝ := {x : ℝ | ∃ y : ℝ, y = Real.sqrt (x - 1)}

-- State the theorem
theorem intersection_M_P : M ∩ P = {x : ℝ | x ≥ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_P_l2289_228900


namespace NUMINAMATH_CALUDE_remainder_problem_l2289_228930

theorem remainder_problem (n : ℕ) (h1 : n = 349) (h2 : n % 17 = 9) : n % 13 = 11 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2289_228930


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2289_228906

/-- The system of inequalities:
    11x² + 8xy + 8y² ≤ 3
    x - 4y ≤ -3
    has the solution (-1/3, 2/3) -/
theorem inequality_system_solution :
  let x : ℚ := -1/3
  let y : ℚ := 2/3
  11 * x^2 + 8 * x * y + 8 * y^2 ≤ 3 ∧
  x - 4 * y ≤ -3 := by
  sorry

#check inequality_system_solution

end NUMINAMATH_CALUDE_inequality_system_solution_l2289_228906


namespace NUMINAMATH_CALUDE_square_of_real_not_always_positive_l2289_228983

theorem square_of_real_not_always_positive : 
  ¬(∀ (a : ℝ), a^2 > 0) :=
by
  sorry

end NUMINAMATH_CALUDE_square_of_real_not_always_positive_l2289_228983


namespace NUMINAMATH_CALUDE_total_ribbons_used_l2289_228996

def dresses_per_day_first_week : ℕ := 2
def days_first_week : ℕ := 7
def dresses_per_day_second_week : ℕ := 3
def days_second_week : ℕ := 2
def ribbons_per_dress : ℕ := 2

theorem total_ribbons_used :
  (dresses_per_day_first_week * days_first_week + 
   dresses_per_day_second_week * days_second_week) * 
  ribbons_per_dress = 40 := by
  sorry

end NUMINAMATH_CALUDE_total_ribbons_used_l2289_228996


namespace NUMINAMATH_CALUDE_min_socks_for_five_correct_min_socks_for_five_optimal_l2289_228966

/-- Represents the colors of socks --/
inductive Color
  | Red
  | White
  | Blue

/-- Represents a drawer of socks --/
structure SockDrawer where
  red : ℕ
  white : ℕ
  blue : ℕ
  red_min : red ≥ 5
  white_min : white ≥ 5
  blue_min : blue ≥ 5

/-- The minimum number of socks to guarantee 5 of the same color --/
def minSocksForFive (drawer : SockDrawer) : ℕ := 13

theorem min_socks_for_five_correct (drawer : SockDrawer) :
  ∀ n : ℕ, n < minSocksForFive drawer →
    ∃ (r w b : ℕ), r < 5 ∧ w < 5 ∧ b < 5 ∧ r + w + b = n :=
  sorry

theorem min_socks_for_five_optimal (drawer : SockDrawer) :
  ∃ (r w b : ℕ), (r = 5 ∨ w = 5 ∨ b = 5) ∧ r + w + b = minSocksForFive drawer :=
  sorry

end NUMINAMATH_CALUDE_min_socks_for_five_correct_min_socks_for_five_optimal_l2289_228966


namespace NUMINAMATH_CALUDE_sequence_divisibility_and_conditions_l2289_228950

def a (n : ℕ) : ℤ := 15 * n + 2 + (15 * n - 32) * 16^(n - 1)

theorem sequence_divisibility_and_conditions :
  (∀ n : ℕ, (15^3 : ℤ) ∣ a n) ∧
  (∀ n : ℕ, (1991 : ℤ) ∣ a n ∧ (1991 : ℤ) ∣ a (n + 1) ∧ (1991 : ℤ) ∣ a (n + 2) ↔ ∃ k : ℕ, n = 89595 * k) :=
sorry

end NUMINAMATH_CALUDE_sequence_divisibility_and_conditions_l2289_228950


namespace NUMINAMATH_CALUDE_brick_height_specific_brick_height_l2289_228993

/-- The height of a brick given its dimensions and the wall it's used to build -/
theorem brick_height (brick_length brick_width : ℝ)
                     (wall_length wall_width wall_height : ℝ)
                     (num_bricks : ℕ) : ℝ :=
  let wall_volume := wall_length * wall_width * wall_height
  let brick_volume := wall_volume / num_bricks
  brick_volume / (brick_length * brick_width)

/-- The height of the brick is 7.5 cm given the specified conditions -/
theorem specific_brick_height :
  brick_height 20 10 2500 200 75 25000 = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_brick_height_specific_brick_height_l2289_228993


namespace NUMINAMATH_CALUDE_tan_theta_value_l2289_228979

theorem tan_theta_value (θ : ℝ) (z₁ z₂ : ℂ) 
  (h1 : z₁ = Complex.mk (Real.sin θ) (-4/5))
  (h2 : z₂ = Complex.mk (3/5) (-Real.cos θ))
  (h3 : (z₁ - z₂).re = 0) : 
  Real.tan θ = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_value_l2289_228979


namespace NUMINAMATH_CALUDE_max_product_of_three_l2289_228974

def S : Set Int := {-9, -5, -3, 0, 2, 6, 8}

theorem max_product_of_three (a b c : Int) (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) 
  (hdistinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) : 
  a * b * c ≤ 360 :=
sorry

end NUMINAMATH_CALUDE_max_product_of_three_l2289_228974


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_expression_l2289_228976

theorem largest_prime_factor_of_expression : 
  ∃ (p : ℕ), p.Prime ∧ p ∣ (25^3 + 15^4 - 5^6 + 20^3) ∧ 
  ∀ (q : ℕ), q.Prime → q ∣ (25^3 + 15^4 - 5^6 + 20^3) → q ≤ p ∧ p = 97 := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_expression_l2289_228976


namespace NUMINAMATH_CALUDE_quadratic_roots_result_l2289_228965

theorem quadratic_roots_result (k p : ℕ) (hk : k > 0) 
  (h_roots : ∃ (x₁ x₂ : ℕ), x₁ > 0 ∧ x₂ > 0 ∧ 
    (k - 1) * x₁^2 - p * x₁ + k = 0 ∧
    (k - 1) * x₂^2 - p * x₂ + k = 0 ∧
    x₁ ≠ x₂) :
  k^(k*p) * (p^p + k^k) + (p + k) = 1989 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_result_l2289_228965


namespace NUMINAMATH_CALUDE_t_shape_area_is_20_l2289_228954

/-- Represents the structure inside the square WXYZ -/
structure InternalStructure where
  top_left_side : ℕ
  top_right_side : ℕ
  bottom_right_side : ℕ
  bottom_left_side : ℕ
  rectangle_width : ℕ
  rectangle_height : ℕ

/-- Calculates the area of the T-shaped region -/
def t_shape_area (s : InternalStructure) : ℕ :=
  s.top_left_side * s.top_left_side +
  s.bottom_right_side * s.bottom_right_side +
  s.bottom_left_side * s.bottom_left_side +
  s.rectangle_width * s.rectangle_height

/-- The theorem stating that the area of the T-shaped region is 20 -/
theorem t_shape_area_is_20 (s : InternalStructure)
  (h1 : s.top_left_side = 2)
  (h2 : s.top_right_side = 2)
  (h3 : s.bottom_right_side = 2)
  (h4 : s.bottom_left_side = 2)
  (h5 : s.rectangle_width = 4)
  (h6 : s.rectangle_height = 2) :
  t_shape_area s = 20 := by
  sorry

end NUMINAMATH_CALUDE_t_shape_area_is_20_l2289_228954


namespace NUMINAMATH_CALUDE_natashas_average_speed_l2289_228959

/-- Natasha's hill climbing problem -/
theorem natashas_average_speed 
  (time_up : ℝ) 
  (time_down : ℝ) 
  (speed_up : ℝ) 
  (h1 : time_up = 4)
  (h2 : time_down = 2)
  (h3 : speed_up = 1.5)
  : (2 * speed_up * time_up) / (time_up + time_down) = 2 := by
  sorry

#check natashas_average_speed

end NUMINAMATH_CALUDE_natashas_average_speed_l2289_228959


namespace NUMINAMATH_CALUDE_intersection_of_E_l2289_228944

def E (k : ℕ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 ≤ |p.1|^k ∧ |p.1| ≥ 1}

theorem intersection_of_E :
  (⋂ k ∈ Finset.range 1991, E (k + 1)) = {p : ℝ × ℝ | p.2 ≤ |p.1| ∧ |p.1| ≥ 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_E_l2289_228944


namespace NUMINAMATH_CALUDE_distance_AB_is_correct_l2289_228933

/-- The distance between two points A and B, where two people start simultaneously
    and move towards each other under specific conditions. -/
def distance_AB : ℝ :=
  let speed_A : ℝ := 12.5 -- km/h
  let speed_B : ℝ := 10   -- km/h
  let time_to_bank : ℝ := 0.5 -- hours
  let time_to_return : ℝ := 0.5 -- hours
  let time_to_find_card : ℝ := 0.5 -- hours
  let remaining_time : ℝ := 0.25 -- hours
  62.5 -- km

theorem distance_AB_is_correct :
  let speed_A : ℝ := 12.5 -- km/h
  let speed_B : ℝ := 10   -- km/h
  let time_to_bank : ℝ := 0.5 -- hours
  let time_to_return : ℝ := 0.5 -- hours
  let time_to_find_card : ℝ := 0.5 -- hours
  let remaining_time : ℝ := 0.25 -- hours
  distance_AB = 62.5 := by
  sorry

#check distance_AB_is_correct

end NUMINAMATH_CALUDE_distance_AB_is_correct_l2289_228933


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l2289_228922

theorem inequality_and_equality_condition (a b c : ℝ) 
  (h_nonneg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0) 
  (h_condition : a * b + b * c + c * a + 2 * a * b * c = 1) : 
  Real.sqrt a + Real.sqrt b + Real.sqrt c ≥ 2 ∧ 
  (Real.sqrt a + Real.sqrt b + Real.sqrt c = 2 ↔ 
    a = (-3 + Real.sqrt 17) / 4 ∧ 
    b = (-3 + Real.sqrt 17) / 4 ∧ 
    c = (-3 + Real.sqrt 17) / 4) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l2289_228922


namespace NUMINAMATH_CALUDE_complex_on_ellipse_real_fraction_l2289_228968

theorem complex_on_ellipse_real_fraction (z : ℂ) :
  let x : ℝ := z.re
  let y : ℝ := z.im
  (x^2 / 9 + y^2 / 16 = 1) →
  ((z - (1 + I)) / (z - I)).im = 0 →
  z = Complex.mk ((3 * Real.sqrt 15) / 4) 1 ∨
  z = Complex.mk (-(3 * Real.sqrt 15) / 4) 1 :=
by sorry

end NUMINAMATH_CALUDE_complex_on_ellipse_real_fraction_l2289_228968


namespace NUMINAMATH_CALUDE_segment_sum_is_132_div_7_l2289_228980

/-- Represents an acute triangle with two altitudes dividing its sides. -/
structure AcuteTriangleWithAltitudes where
  /-- Length of the first known segment -/
  a : ℝ
  /-- Length of the second known segment -/
  b : ℝ
  /-- Length of the third known segment -/
  c : ℝ
  /-- Length of the unknown segment -/
  y : ℝ
  /-- Condition that all segment lengths are positive -/
  ha : a > 0
  hb : b > 0
  hc : c > 0
  hy : y > 0
  /-- Condition that the triangle is acute -/
  acute : True  -- We don't have enough information to express this condition precisely

/-- The sum of all segments on the sides of the triangle cut by the altitudes -/
def segmentSum (t : AcuteTriangleWithAltitudes) : ℝ :=
  t.a + t.b + t.c + t.y

/-- Theorem stating that for a triangle with segments 7, 4, 5, and y, the sum is 132/7 -/
theorem segment_sum_is_132_div_7 (t : AcuteTriangleWithAltitudes)
  (h1 : t.a = 7) (h2 : t.b = 4) (h3 : t.c = 5) :
  segmentSum t = 132 / 7 := by
  sorry

end NUMINAMATH_CALUDE_segment_sum_is_132_div_7_l2289_228980


namespace NUMINAMATH_CALUDE_remainder_problem_l2289_228923

theorem remainder_problem (x : ℤ) : 
  x % 62 = 7 → (x + 11) % 31 = 18 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l2289_228923


namespace NUMINAMATH_CALUDE_string_average_length_l2289_228978

theorem string_average_length : 
  let string1 : ℝ := 2
  let string2 : ℝ := 6
  let string3 : ℝ := 9
  let num_strings : ℕ := 3
  (string1 + string2 + string3) / num_strings = 17 / 3 := by
  sorry

end NUMINAMATH_CALUDE_string_average_length_l2289_228978


namespace NUMINAMATH_CALUDE_stack_height_probability_l2289_228937

/-- Represents the possible heights of a crate -/
inductive CrateHeight : Type
  | Two : CrateHeight
  | Three : CrateHeight
  | Five : CrateHeight

/-- The number of crates in the stack -/
def numCrates : ℕ := 5

/-- The target height of the stack -/
def targetHeight : ℕ := 16

/-- Calculates the total number of possible arrangements -/
def totalArrangements : ℕ := 3^numCrates

/-- Calculates the number of valid arrangements that sum to the target height -/
def validArrangements : ℕ := 20

/-- The probability of achieving the target height -/
def probabilityTargetHeight : ℚ := validArrangements / totalArrangements

theorem stack_height_probability :
  probabilityTargetHeight = 20 / 243 := by sorry

end NUMINAMATH_CALUDE_stack_height_probability_l2289_228937


namespace NUMINAMATH_CALUDE_boys_without_pencils_l2289_228951

theorem boys_without_pencils (total_students : ℕ) (total_boys : ℕ) (students_with_pencils : ℕ) (girls_with_pencils : ℕ)
  (h1 : total_students = 30)
  (h2 : total_boys = 18)
  (h3 : students_with_pencils = 25)
  (h4 : girls_with_pencils = 15) :
  total_boys - (students_with_pencils - girls_with_pencils) = 8 :=
by sorry

end NUMINAMATH_CALUDE_boys_without_pencils_l2289_228951


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_in_one_third_sector_l2289_228964

/-- The radius of a circle inscribed in a sector that is one-third of a circle with radius 5 cm -/
theorem inscribed_circle_radius_in_one_third_sector :
  ∃ (r : ℝ), 
    r > 0 ∧ 
    r * (Real.sqrt 3 + 1) = 5 ∧
    r = (5 * Real.sqrt 3 - 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_in_one_third_sector_l2289_228964


namespace NUMINAMATH_CALUDE_shaded_area_theorem_l2289_228941

/-- The area of the upper triangle formed by a diagonal line in a 20cm x 15cm rectangle, 
    where the diagonal starts from the corner of a 5cm x 5cm square within the rectangle. -/
theorem shaded_area_theorem (total_width total_height small_square_side : ℝ) 
  (hw : total_width = 20)
  (hh : total_height = 15)
  (hs : small_square_side = 5) : 
  let large_width := total_width - small_square_side
  let large_height := total_height
  let diagonal_slope := large_height / total_width
  let intersection_y := diagonal_slope * small_square_side
  let triangle_base := large_width
  let triangle_height := large_height - intersection_y
  triangle_base * triangle_height / 2 = 84.375 := by
  sorry

#eval (20 - 5) * (15 - 15 / 20 * 5) / 2

end NUMINAMATH_CALUDE_shaded_area_theorem_l2289_228941


namespace NUMINAMATH_CALUDE_lawrence_marbles_l2289_228984

theorem lawrence_marbles (total_marbles : ℕ) (marbles_per_friend : ℕ) 
  (h1 : total_marbles = 5504) 
  (h2 : marbles_per_friend = 86) : 
  total_marbles / marbles_per_friend = 64 := by
  sorry

end NUMINAMATH_CALUDE_lawrence_marbles_l2289_228984


namespace NUMINAMATH_CALUDE_power_equality_l2289_228977

theorem power_equality (m : ℝ) : (16 : ℝ) ^ (3/4) = 2^m → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l2289_228977


namespace NUMINAMATH_CALUDE_dogs_can_prevent_escape_l2289_228912

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square field -/
structure SquareField where
  sideLength : ℝ
  center : Point
  corners : Fin 4 → Point

/-- Represents the game setup -/
structure WolfDogGame where
  field : SquareField
  wolfSpeed : ℝ
  dogSpeed : ℝ

/-- Checks if a point is on the perimeter of the square field -/
def isOnPerimeter (field : SquareField) (p : Point) : Prop :=
  let a := field.sideLength / 2
  (p.x = field.center.x - a ∨ p.x = field.center.x + a) ∧
  (p.y ≥ field.center.y - a ∧ p.y ≤ field.center.y + a) ∨
  (p.y = field.center.y - a ∨ p.y = field.center.y + a) ∧
  (p.x ≥ field.center.x - a ∧ p.x ≤ field.center.x + a)

/-- Theorem: Dogs can prevent the wolf from escaping -/
theorem dogs_can_prevent_escape (game : WolfDogGame) 
  (h1 : game.field.sideLength > 0)
  (h2 : game.dogSpeed = 1.5 * game.wolfSpeed)
  (h3 : game.wolfSpeed > 0) :
  ∀ (p : Point), isOnPerimeter game.field p →
    ∃ (t : ℝ), t ≥ 0 ∧ 
      (∃ (i : Fin 4), (t * game.dogSpeed)^2 ≥ 
        ((game.field.corners i).x - p.x)^2 + ((game.field.corners i).y - p.y)^2) ∧
      (t * game.wolfSpeed)^2 < 
        (game.field.center.x - p.x)^2 + (game.field.center.y - p.y)^2 :=
sorry

end NUMINAMATH_CALUDE_dogs_can_prevent_escape_l2289_228912


namespace NUMINAMATH_CALUDE_right_triangle_acute_angles_l2289_228956

theorem right_triangle_acute_angles (α β : ℝ) : 
  α = 30 → -- One acute angle is 30 degrees
  α + β + 90 = 180 → -- Sum of angles in a triangle is 180 degrees, and one angle is right (90 degrees)
  β = 60 := by -- The other acute angle is 60 degrees
sorry

end NUMINAMATH_CALUDE_right_triangle_acute_angles_l2289_228956


namespace NUMINAMATH_CALUDE_impossibleConsecutive_l2289_228963

/-- A move that replaces one number with the sum of both numbers -/
def move (a b : ℕ) : ℕ × ℕ := (a + b, b)

/-- The sequence of numbers obtained after applying moves -/
def boardSequence : ℕ → ℕ × ℕ
  | 0 => (2, 5)
  | n + 1 => let (a, b) := boardSequence n; move a b

/-- The difference between the two numbers on the board after n moves -/
def difference (n : ℕ) : ℕ :=
  let (a, b) := boardSequence n
  max a b - min a b

theorem impossibleConsecutive : ∀ n : ℕ, difference n ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_impossibleConsecutive_l2289_228963


namespace NUMINAMATH_CALUDE_bob_initial_nickels_l2289_228904

theorem bob_initial_nickels (a b : ℕ) 
  (h1 : b + 1 = 4 * (a - 1)) 
  (h2 : b - 1 = 3 * (a + 1)) : 
  b = 31 := by sorry

end NUMINAMATH_CALUDE_bob_initial_nickels_l2289_228904


namespace NUMINAMATH_CALUDE_special_number_fraction_l2289_228917

theorem special_number_fraction (list : List ℝ) (n : ℝ) :
  list.length = 21 ∧
  n ∉ list ∧
  n = 5 * (list.sum / list.length) →
  n / (list.sum + n) = 1 / 5 := by
sorry

end NUMINAMATH_CALUDE_special_number_fraction_l2289_228917


namespace NUMINAMATH_CALUDE_constant_distance_between_bikers_l2289_228973

def distance_between_bikers (t : ℝ) (initial_distance : ℝ) (speed_a : ℝ) (speed_b : ℝ) : ℝ :=
  initial_distance + speed_b * t - speed_a * t

theorem constant_distance_between_bikers
  (speed_a : ℝ)
  (speed_b : ℝ)
  (initial_distance : ℝ)
  (h1 : speed_a = 350 / 7)
  (h2 : speed_b = 500 / 10)
  (h3 : initial_distance = 75)
  (t : ℝ) :
  distance_between_bikers t initial_distance speed_a speed_b = initial_distance :=
by sorry

end NUMINAMATH_CALUDE_constant_distance_between_bikers_l2289_228973


namespace NUMINAMATH_CALUDE_sin_cos_equation_solutions_l2289_228962

/-- The number of solutions to sin(π/4 * sin x) = cos(π/4 * cos x) in [0, 2π] -/
theorem sin_cos_equation_solutions :
  ∃! (s : Finset ℝ), s.card = 4 ∧ 
  (∀ x ∈ s, 0 ≤ x ∧ x ≤ 2 * π ∧ 
    Real.sin (π/4 * Real.sin x) = Real.cos (π/4 * Real.cos x)) ∧
  (∀ y, 0 ≤ y ∧ y ≤ 2 * π ∧ 
    Real.sin (π/4 * Real.sin y) = Real.cos (π/4 * Real.cos y) → y ∈ s) :=
by sorry

end NUMINAMATH_CALUDE_sin_cos_equation_solutions_l2289_228962


namespace NUMINAMATH_CALUDE_right_triangle_medians_l2289_228915

theorem right_triangle_medians (D E F : ℝ × ℝ) : 
  -- D is the right angle
  (E.1 - D.1) * (F.1 - D.1) + (E.2 - D.2) * (F.2 - D.2) = 0 →
  -- Length of median from D to midpoint of EF is 3√5
  ((E.1 + F.1) / 2 - D.1)^2 + ((E.2 + F.2) / 2 - D.2)^2 = 45 →
  -- Length of median from E to midpoint of DF is 5
  ((D.1 + F.1) / 2 - E.1)^2 + ((D.2 + F.2) / 2 - E.2)^2 = 25 →
  -- Then the length of DE is 2√14
  (E.1 - D.1)^2 + (E.2 - D.2)^2 = 56 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_medians_l2289_228915

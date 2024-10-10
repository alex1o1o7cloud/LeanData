import Mathlib

namespace max_first_term_is_16_l2107_210778

/-- A sequence of positive real numbers satisfying the given conditions -/
def SpecialSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, n > 0 → a n > 0) ∧ 
  (∀ n, n > 0 → (2 * a (n + 1) - a n) * (a (n + 1) * a n - 1) = 0) ∧
  (a 1 = a 10)

/-- The maximum possible value of the first term in the special sequence is 16 -/
theorem max_first_term_is_16 (a : ℕ → ℝ) (h : SpecialSequence a) : 
  ∃ (M : ℝ), M = 16 ∧ a 1 ≤ M ∧ ∀ (b : ℕ → ℝ), SpecialSequence b → b 1 ≤ M :=
sorry

end max_first_term_is_16_l2107_210778


namespace beatrice_tv_ratio_l2107_210738

/-- Proves that the ratio of TVs Beatrice looked at in the online store to the first store is 3:1 -/
theorem beatrice_tv_ratio : 
  ∀ (first_store online_store auction_site total : ℕ),
  first_store = 8 →
  auction_site = 10 →
  total = 42 →
  first_store + online_store + auction_site = total →
  online_store / first_store = 3 := by
sorry

end beatrice_tv_ratio_l2107_210738


namespace correct_ball_arrangements_l2107_210771

/-- The number of ways to arrange 9 balls with 2 red, 3 yellow, and 4 white balls -/
def ballArrangements : ℕ := 2520

/-- The total number of balls -/
def totalBalls : ℕ := 9

/-- The number of red balls -/
def redBalls : ℕ := 2

/-- The number of yellow balls -/
def yellowBalls : ℕ := 3

/-- The number of white balls -/
def whiteBalls : ℕ := 4

theorem correct_ball_arrangements :
  ballArrangements = Nat.factorial totalBalls / (Nat.factorial redBalls * Nat.factorial yellowBalls * Nat.factorial whiteBalls) :=
by sorry

end correct_ball_arrangements_l2107_210771


namespace same_terminal_side_l2107_210701

theorem same_terminal_side (k : ℤ) : ∃ k, (11 * π) / 6 = 2 * k * π - π / 6 := by
  sorry

end same_terminal_side_l2107_210701


namespace complex_magnitude_bounds_l2107_210739

/-- Given a complex number z satisfying 2|z-3-3i| = |z|, prove that the maximum value of |z| is 6√2 and the minimum value of |z| is 2√2. -/
theorem complex_magnitude_bounds (z : ℂ) (h : 2 * Complex.abs (z - (3 + 3*I)) = Complex.abs z) :
  (∃ (w : ℂ), 2 * Complex.abs (w - (3 + 3*I)) = Complex.abs w ∧ Complex.abs w = 6 * Real.sqrt 2) ∧
  (∃ (v : ℂ), 2 * Complex.abs (v - (3 + 3*I)) = Complex.abs v ∧ Complex.abs v = 2 * Real.sqrt 2) ∧
  (∀ (u : ℂ), 2 * Complex.abs (u - (3 + 3*I)) = Complex.abs u → 
    2 * Real.sqrt 2 ≤ Complex.abs u ∧ Complex.abs u ≤ 6 * Real.sqrt 2) :=
by sorry

end complex_magnitude_bounds_l2107_210739


namespace system_equation_solution_l2107_210746

theorem system_equation_solution (x y a b : ℝ) (h1 : 2 * x - y = a) (h2 : 4 * y - 8 * x = b) (h3 : b ≠ 0) :
  a / b = -1 / 4 := by
  sorry

end system_equation_solution_l2107_210746


namespace complex_sum_magnitude_possible_values_complete_l2107_210716

-- Define the set of possible values for |a + b + c|
def PossibleValues : Set ℝ := {1, 2, 3}

-- Main theorem
theorem complex_sum_magnitude (a b c : ℂ) :
  Complex.abs a = 1 →
  Complex.abs b = 1 →
  Complex.abs c = 1 →
  a^2 / (b * c) + b^2 / (a * c) + c^2 / (a * b) = 1 →
  Complex.abs (a + b + c) ∈ PossibleValues := by
  sorry

-- Completeness of the set of possible values
theorem possible_values_complete (x : ℝ) :
  x ∈ PossibleValues →
  ∃ (a b c : ℂ), Complex.abs a = 1 ∧
                  Complex.abs b = 1 ∧
                  Complex.abs c = 1 ∧
                  a^2 / (b * c) + b^2 / (a * c) + c^2 / (a * b) = 1 ∧
                  Complex.abs (a + b + c) = x := by
  sorry

end complex_sum_magnitude_possible_values_complete_l2107_210716


namespace expand_expression_l2107_210758

theorem expand_expression (x : ℝ) : 2 * (5 * x^2 - 3 * x + 4 - x^3) = -2 * x^3 + 10 * x^2 - 6 * x + 8 := by
  sorry

end expand_expression_l2107_210758


namespace jerry_lawn_mowing_money_l2107_210708

/-- The amount of money Jerry made mowing lawns -/
def M : ℝ := sorry

/-- The amount of money Jerry made from weed eating -/
def weed_eating_money : ℝ := 31

/-- The number of weeks Jerry's money would last -/
def weeks : ℝ := 9

/-- The amount Jerry would spend per week -/
def weekly_spending : ℝ := 5

theorem jerry_lawn_mowing_money :
  M = 14 :=
by
  have total_money : M + weed_eating_money = weeks * weekly_spending := by sorry
  sorry

end jerry_lawn_mowing_money_l2107_210708


namespace prob_same_color_is_one_twentieth_l2107_210756

/-- Represents the number of marbles of each color -/
def marbles_per_color : ℕ := 3

/-- Represents the number of girls selecting marbles -/
def number_of_girls : ℕ := 3

/-- Calculates the probability of all girls selecting the same colored marble -/
def prob_same_color : ℚ :=
  2 * (marbles_per_color.factorial / (marbles_per_color + number_of_girls).factorial)

/-- Theorem stating that the probability of all girls selecting the same colored marble is 1/20 -/
theorem prob_same_color_is_one_twentieth : prob_same_color = 1 / 20 := by
  sorry

end prob_same_color_is_one_twentieth_l2107_210756


namespace inequality_solution_set_l2107_210725

theorem inequality_solution_set (x : ℝ) : -x^2 + 4*x + 5 < 0 ↔ x > 5 ∨ x < -1 := by
  sorry

end inequality_solution_set_l2107_210725


namespace triangle_property_l2107_210796

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Sides opposite to A, B, C respectively
  (h1 : A + B + C = π)  -- Sum of angles in a triangle
  (h2 : a > 0 ∧ b > 0 ∧ c > 0)  -- Positive side lengths
  (h3 : b < c)  -- Given condition

-- Define the existence of points E and F
def points_exist (t : Triangle) : Prop :=
  ∃ E F : ℝ, 
    E > 0 ∧ F > 0 ∧
    E ≤ t.c ∧ F ≤ t.b ∧
    E = F ∧
    ∃ D : ℝ, D > 0 ∧ D < t.a ∧
    (t.A / 2 = Real.arctan (D / E) + Real.arctan (D / F))

-- Theorem statement
theorem triangle_property (t : Triangle) (h : points_exist t) :
  t.A / 2 ≤ t.B ∧ (t.a * t.c) / (t.b + t.c) = t.c * (t.a / (t.b + t.c)) :=
sorry

end triangle_property_l2107_210796


namespace only_set_C_forms_triangle_l2107_210766

-- Define a function to check if three lengths can form a triangle
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

-- Define the given sets of lengths
def set_A : (ℝ × ℝ × ℝ) := (3, 4, 8)
def set_B : (ℝ × ℝ × ℝ) := (2, 5, 2)
def set_C : (ℝ × ℝ × ℝ) := (3, 5, 6)
def set_D : (ℝ × ℝ × ℝ) := (5, 6, 11)

-- Theorem stating that only set_C can form a triangle
theorem only_set_C_forms_triangle :
  ¬(can_form_triangle set_A.1 set_A.2.1 set_A.2.2) ∧
  ¬(can_form_triangle set_B.1 set_B.2.1 set_B.2.2) ∧
  (can_form_triangle set_C.1 set_C.2.1 set_C.2.2) ∧
  ¬(can_form_triangle set_D.1 set_D.2.1 set_D.2.2) :=
by sorry

end only_set_C_forms_triangle_l2107_210766


namespace isosceles_triangle_side_length_l2107_210783

/-- An isosceles triangle with perimeter 7 and one side length 3 has equal sides of length 3 or 2 -/
theorem isosceles_triangle_side_length (a b c : ℝ) : 
  a + b + c = 7 →  -- perimeter is 7
  (a = 3 ∨ b = 3 ∨ c = 3) →  -- one side length is 3
  ((a = b ∧ a ≠ c) ∨ (b = c ∧ b ≠ a) ∨ (a = c ∧ a ≠ b)) →  -- isosceles triangle condition
  (a = 3 ∧ b = 3) ∨ (a = 2 ∧ b = 2) ∨ (b = 3 ∧ c = 3) ∨ (b = 2 ∧ c = 2) ∨ (a = 3 ∧ c = 3) ∨ (a = 2 ∧ c = 2) :=
by sorry

end isosceles_triangle_side_length_l2107_210783


namespace like_terms_exponent_l2107_210757

theorem like_terms_exponent (x y : ℝ) (m : ℤ) : 
  (∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ a * x^3 * y^(m+3) = b * x^3 * y^5) → m = 2 := by
  sorry

end like_terms_exponent_l2107_210757


namespace min_value_squared_sum_l2107_210723

theorem min_value_squared_sum (a b : ℝ) (h : a^2 + 2*a*b - 3*b^2 = 1) :
  ∃ (m : ℝ), m = (Real.sqrt 5 + 1) / 4 ∧ ∀ (x y : ℝ), x^2 + 2*x*y - 3*y^2 = 1 → x^2 + y^2 ≥ m :=
sorry

end min_value_squared_sum_l2107_210723


namespace negation_of_implication_negation_of_greater_than_one_l2107_210767

theorem negation_of_implication (p q : Prop) :
  ¬(p → q) ↔ (p ∧ ¬q) := by sorry

theorem negation_of_greater_than_one :
  ¬(∀ x : ℝ, x > 1 → x^2 > 1) ↔ (∃ x : ℝ, x ≤ 1 ∧ x^2 ≤ 1) := by sorry

end negation_of_implication_negation_of_greater_than_one_l2107_210767


namespace unique_digit_sum_l2107_210721

theorem unique_digit_sum (A₁₂ B C D : ℕ) : 
  (∃! (B C D : ℕ), 
    (10 > A₁₂ ∧ A₁₂ > B ∧ B > C ∧ C > D ∧ D > 0) ∧
    (1000 * A₁₂ + 100 * B + 10 * C + D) - (1000 * D + 100 * C + 10 * B + A₁₂) = 
    (1000 * B + 100 * D + 10 * A₁₂ + C)) →
  B + C + D = 11 := by
sorry

end unique_digit_sum_l2107_210721


namespace cos_arcsin_eight_seventeenths_l2107_210728

theorem cos_arcsin_eight_seventeenths : 
  Real.cos (Real.arcsin (8 / 17)) = 15 / 17 := by
sorry

end cos_arcsin_eight_seventeenths_l2107_210728


namespace video_game_sales_earnings_l2107_210755

/-- The amount of money Zachary received from selling his video games -/
def zachary_earnings : ℕ := 40 * 5

/-- The amount of money Jason received from selling his video games -/
def jason_earnings : ℕ := zachary_earnings + (zachary_earnings * 30 / 100)

/-- The amount of money Ryan received from selling his video games -/
def ryan_earnings : ℕ := jason_earnings + 50

/-- The amount of money Emily received from selling her video games -/
def emily_earnings : ℕ := ryan_earnings - (ryan_earnings * 20 / 100)

/-- The amount of money Lily received from selling her video games -/
def lily_earnings : ℕ := emily_earnings + 70

/-- The total amount of money received by all five friends -/
def total_earnings : ℕ := zachary_earnings + jason_earnings + ryan_earnings + emily_earnings + lily_earnings

theorem video_game_sales_earnings : total_earnings = 1336 := by
  sorry

end video_game_sales_earnings_l2107_210755


namespace triangle_similarity_fc_value_l2107_210780

/-- Given a triangle ADE with point C on AD and point B on AC, prove that FC = 13.875 -/
theorem triangle_similarity_fc_value (DC CB AD AB ED : ℝ) : 
  DC = 10 →
  CB = 9 →
  AB = (1/3) * AD →
  ED = (3/4) * AD →
  ∃ (FC : ℝ), FC = 13.875 := by
  sorry

end triangle_similarity_fc_value_l2107_210780


namespace remainder_17_63_mod_7_l2107_210714

theorem remainder_17_63_mod_7 : 17^63 ≡ 6 [ZMOD 7] := by sorry

end remainder_17_63_mod_7_l2107_210714


namespace sheep_price_is_30_l2107_210793

/-- Represents the farm animals and their sale --/
structure FarmSale where
  goats : ℕ
  sheep : ℕ
  goat_price : ℕ
  sheep_price : ℕ
  goats_sold_ratio : ℚ
  sheep_sold_ratio : ℚ
  total_sale : ℕ

/-- The conditions of the farm sale problem --/
def farm_conditions (s : FarmSale) : Prop :=
  s.goats * 7 = s.sheep * 5 ∧
  s.goats + s.sheep = 360 ∧
  s.goats_sold_ratio = 1/2 ∧
  s.sheep_sold_ratio = 2/3 ∧
  s.goat_price = 40 ∧
  s.total_sale = 7200

/-- The theorem stating that the sheep price is $30 --/
theorem sheep_price_is_30 (s : FarmSale) (h : farm_conditions s) : s.sheep_price = 30 := by
  sorry


end sheep_price_is_30_l2107_210793


namespace ratio_sum_problem_l2107_210779

theorem ratio_sum_problem (a b c d : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) 
  (h_ratio : b = 2*a ∧ c = 4*a ∧ d = 5*a) 
  (h_sum_squares : a^2 + b^2 + c^2 + d^2 = 2540) : 
  a + b + c + d = 12 * Real.sqrt 55 := by
  sorry

end ratio_sum_problem_l2107_210779


namespace student_selection_probability_l2107_210710

theorem student_selection_probability (n : ℕ) : 
  (4 : ℝ) ≥ 0 ∧ (n : ℝ) ≥ 0 →
  (((n + 4) * (n + 3) / 2 - 6) / ((n + 4) * (n + 3) / 2) = 5 / 6) →
  n = 5 :=
by sorry

end student_selection_probability_l2107_210710


namespace find_number_l2107_210777

theorem find_number : ∃ n : ℕ, 72519 * n = 724827405 ∧ n = 10005 := by
  sorry

end find_number_l2107_210777


namespace inequality_proof_l2107_210748

theorem inequality_proof (a b c d : ℝ) 
  (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c ≥ d)
  (h4 : a + b + c + d = 13)
  (h5 : a^2 + b^2 + c^2 + d^2 = 43) :
  a * b ≥ 3 + c * d := by
sorry

end inequality_proof_l2107_210748


namespace arun_weight_upper_bound_l2107_210745

-- Define the weight range according to Arun's opinion
def arun_lower_bound : ℝ := 66
def arun_upper_bound : ℝ := 72

-- Define the weight range according to Arun's brother's opinion
def brother_lower_bound : ℝ := 60
def brother_upper_bound : ℝ := 70

-- Define the average weight
def average_weight : ℝ := 68

-- Define mother's upper bound (to be proven)
def mother_upper_bound : ℝ := 70

-- Theorem statement
theorem arun_weight_upper_bound :
  ∀ w : ℝ,
  (w > arun_lower_bound ∧ w < arun_upper_bound) →
  (w > brother_lower_bound ∧ w < brother_upper_bound) →
  (w ≤ mother_upper_bound) →
  (∃ w_min w_max : ℝ, 
    w_min > max arun_lower_bound brother_lower_bound ∧
    w_max < min arun_upper_bound brother_upper_bound ∧
    (w_min + w_max) / 2 = average_weight) →
  mother_upper_bound = 70 := by
sorry

end arun_weight_upper_bound_l2107_210745


namespace zach_needs_six_dollars_l2107_210784

/-- The amount of money Zach needs to earn to buy the bike -/
def money_needed (bike_cost allowance lawn_money babysit_rate babysit_hours savings : ℕ) : ℕ :=
  let total_earned := allowance + lawn_money + babysit_rate * babysit_hours
  let total_available := savings + total_earned
  if total_available ≥ bike_cost then 0
  else bike_cost - total_available

/-- Theorem stating how much more money Zach needs to earn -/
theorem zach_needs_six_dollars :
  money_needed 100 5 10 7 2 65 = 6 := by
  sorry

end zach_needs_six_dollars_l2107_210784


namespace negation_of_universal_proposition_l2107_210724

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x ∈ (Set.Ioo 0 1) → x^2 - x < 0) ↔
  (∃ x : ℝ, x ∈ (Set.Ioo 0 1) ∧ x^2 - x ≥ 0) :=
by sorry

end negation_of_universal_proposition_l2107_210724


namespace carol_stereo_savings_l2107_210798

theorem carol_stereo_savings : 
  ∀ (stereo_fraction : ℚ),
  (stereo_fraction + (1/3) * stereo_fraction = 1/4) →
  stereo_fraction = 3/16 := by
sorry

end carol_stereo_savings_l2107_210798


namespace square_sum_equals_nineteen_l2107_210786

theorem square_sum_equals_nineteen (x y : ℝ) (h1 : x + y = 5) (h2 : x * y = 3) :
  x^2 + y^2 = 19 := by
sorry

end square_sum_equals_nineteen_l2107_210786


namespace max_pies_without_ingredients_l2107_210770

theorem max_pies_without_ingredients (total_pies : ℕ) 
  (chocolate_pies marshmallow_pies cayenne_pies peanut_pies : ℕ) : 
  total_pies = 48 →
  chocolate_pies ≥ total_pies / 2 →
  marshmallow_pies ≥ 2 * total_pies / 3 →
  cayenne_pies ≥ 3 * total_pies / 5 →
  peanut_pies ≥ total_pies / 8 →
  ∃ (pies_without_ingredients : ℕ), 
    pies_without_ingredients ≤ 16 ∧
    pies_without_ingredients + chocolate_pies + marshmallow_pies + cayenne_pies + peanut_pies ≥ total_pies :=
by sorry

end max_pies_without_ingredients_l2107_210770


namespace telephone_bill_proof_l2107_210751

theorem telephone_bill_proof (F C : ℝ) : 
  F + C = 40 →
  F + 2*C = 76 →
  F + C = 40 := by
sorry

end telephone_bill_proof_l2107_210751


namespace pencils_given_l2107_210761

theorem pencils_given (initial_pencils total_pencils : ℕ) 
  (h1 : initial_pencils = 9)
  (h2 : total_pencils = 65) :
  total_pencils - initial_pencils = 56 := by
  sorry

end pencils_given_l2107_210761


namespace y_derivative_l2107_210711

noncomputable def y (x : ℝ) : ℝ :=
  (1 / Real.sqrt 2) * Real.arctan ((3 * x - 1) / Real.sqrt 2) + 
  (1 / 3) * (3 * x - 1) / (3 * x^2 - 2 * x + 1)

theorem y_derivative (x : ℝ) :
  deriv y x = 4 / (3 * (3 * x^2 - 2 * x + 1)^2) :=
sorry

end y_derivative_l2107_210711


namespace problem_solution_l2107_210733

theorem problem_solution (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : x - y = 4 := by
  sorry

end problem_solution_l2107_210733


namespace inscribed_quadrilateral_with_given_lengths_l2107_210787

/-- A quadrilateral that can be inscribed in a circle -/
structure InscribedQuadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  diagonal : ℝ

/-- Theorem: A quadrilateral with side lengths 15, 36, 48, 27, and diagonal 54
    can be inscribed in a circle with diameter 54 -/
theorem inscribed_quadrilateral_with_given_lengths :
  ∃ (q : InscribedQuadrilateral),
    q.a = 15 ∧
    q.b = 36 ∧
    q.c = 48 ∧
    q.d = 27 ∧
    q.diagonal = 54 ∧
    (∃ (r : ℝ), r = 54 ∧ r = q.diagonal) :=
by sorry

end inscribed_quadrilateral_with_given_lengths_l2107_210787


namespace max_cubes_in_box_l2107_210788

/-- The maximum number of cubes that can fit in a rectangular box -/
def max_cubes (box_length box_width box_height cube_volume : ℕ) : ℕ :=
  (box_length * box_width * box_height) / cube_volume

/-- Theorem stating the maximum number of 43 cm³ cubes in a 13x17x22 cm box -/
theorem max_cubes_in_box : max_cubes 13 17 22 43 = 114 := by
  sorry

end max_cubes_in_box_l2107_210788


namespace equation_positive_root_l2107_210703

theorem equation_positive_root (n : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ n / (x - 1) + 2 / (1 - x) = 1) → n = 2 := by
  sorry

end equation_positive_root_l2107_210703


namespace linear_equation_with_integer_roots_l2107_210706

theorem linear_equation_with_integer_roots 
  (m : ℤ) (n : ℕ) 
  (h1 : m ≠ 1) 
  (h2 : n = 1) 
  (h3 : ∃ x : ℤ, (m - 1) * x - 3 = 0) :
  m = -2 ∨ m = 0 ∨ m = 2 ∨ m = 4 := by
sorry

end linear_equation_with_integer_roots_l2107_210706


namespace right_triangle_exists_l2107_210743

/-- Checks if three line segments can form a right triangle --/
def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

theorem right_triangle_exists :
  ∃ (a b c : ℕ), is_right_triangle a b c ∧
  (a = 3 ∧ b = 4 ∧ c = 5) ∧
  ¬(is_right_triangle 2 3 4) ∧
  ¬(is_right_triangle 4 5 6) ∧
  ¬(is_right_triangle 5 6 7) :=
sorry

end right_triangle_exists_l2107_210743


namespace games_comparison_l2107_210754

/-- Given Henry's and Neil's initial game counts and the number of games Henry gave to Neil,
    calculate how many times more games Henry has than Neil after the transfer. -/
theorem games_comparison (henry_initial : ℕ) (neil_initial : ℕ) (games_given : ℕ) : 
  henry_initial = 33 →
  neil_initial = 2 →
  games_given = 5 →
  (henry_initial - games_given) / (neil_initial + games_given) = 4 := by
sorry

end games_comparison_l2107_210754


namespace sum_reciprocals_l2107_210782

theorem sum_reciprocals (a b c : ℝ) 
  (h1 : a + b + c = 0) 
  (h2 : a/b + b/c + c/a = 100) : 
  b/a + c/b + a/c = -101 := by
sorry

end sum_reciprocals_l2107_210782


namespace solution_positivity_l2107_210769

theorem solution_positivity (m : ℝ) :
  (∃ x : ℝ, x > 0 ∧ m * x - 1 = 2 * x) ↔ m > 2 := by
  sorry

end solution_positivity_l2107_210769


namespace problem_solution_l2107_210799

def problem (g : ℝ → ℝ) (g_inv : ℝ → ℝ) : Prop :=
  (∀ x, g_inv (g x) = x) ∧
  (∀ y, g (g_inv y) = y) ∧
  g 4 = 6 ∧
  g 6 = 2 ∧
  g 3 = 7 ∧
  g_inv (g_inv 6 + g_inv 7) = 3

theorem problem_solution :
  ∃ (g : ℝ → ℝ) (g_inv : ℝ → ℝ), problem g g_inv :=
by
  sorry

end problem_solution_l2107_210799


namespace total_sum_calculation_l2107_210774

theorem total_sum_calculation (share_a share_b share_c : ℝ) : 
  3 * share_a = 4 * share_b ∧ 
  3 * share_a = 7 * share_c ∧ 
  share_c = 83.99999999999999 → 
  share_a + share_b + share_c = 426.9999999999999 := by
sorry

end total_sum_calculation_l2107_210774


namespace even_decreasing_inequality_l2107_210731

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def is_decreasing_on_positive (f : ℝ → ℝ) : Prop := ∀ x y, 0 < x → x < y → f y < f x

-- State the theorem
theorem even_decreasing_inequality (h1 : is_even f) (h2 : is_decreasing_on_positive f) :
  f (1/2) > f (-2/3) ∧ f (-2/3) > f (3/4) := by sorry

end even_decreasing_inequality_l2107_210731


namespace product_rule_l2107_210797

theorem product_rule (b a : ℤ) (h : 0 ≤ a ∧ a < 10) : 
  (10 * b + a) * (10 * b + 10 - a) = 100 * b * (b + 1) + a * (10 - a) := by
  sorry

end product_rule_l2107_210797


namespace second_number_is_seventeen_l2107_210727

theorem second_number_is_seventeen (first_number second_number third_number : ℕ) :
  first_number = 16 →
  third_number = 20 →
  3 * first_number + 3 * second_number + 3 * third_number + 11 = 170 →
  second_number = 17 := by
sorry

end second_number_is_seventeen_l2107_210727


namespace remainder_of_b_86_mod_50_l2107_210741

theorem remainder_of_b_86_mod_50 : (7^86 + 9^86) % 50 = 40 := by
  sorry

end remainder_of_b_86_mod_50_l2107_210741


namespace unique_solution_base_6_l2107_210773

def base_6_to_decimal (n : ℕ) : ℕ := 
  (n / 100) * 36 + ((n / 10) % 10) * 6 + (n % 10)

def decimal_to_base_6 (n : ℕ) : ℕ := 
  (n / 36) * 100 + ((n / 6) % 6) * 10 + (n % 6)

theorem unique_solution_base_6 :
  ∃! (A B C : ℕ), 
    A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧
    A < 6 ∧ B < 6 ∧ C < 6 ∧
    A ≠ B ∧ B ≠ C ∧ A ≠ C ∧
    base_6_to_decimal (100 * A + 10 * B + C) + base_6_to_decimal (10 * B + C) = 
      base_6_to_decimal (100 * A + 10 * C + A) ∧
    A = 3 ∧ B = 1 ∧ C = 2 ∧
    decimal_to_base_6 (A + B + C) = 10 :=
by sorry

end unique_solution_base_6_l2107_210773


namespace original_advertisers_from_university_a_l2107_210772

/-- Represents the fraction of advertisers from University A -/
def fractionFromUniversityA : ℚ := 3/4

/-- Represents the total number of original network advertisers -/
def totalOriginalAdvertisers : ℕ := 20

/-- Represents the percentage of computer advertisers from University A -/
def percentageFromUniversityA : ℚ := 75/100

theorem original_advertisers_from_university_a :
  (↑⌊(percentageFromUniversityA * totalOriginalAdvertisers)⌋ : ℚ) / totalOriginalAdvertisers = fractionFromUniversityA :=
sorry

end original_advertisers_from_university_a_l2107_210772


namespace f_value_theorem_l2107_210785

-- Define the polynomial equation
def polynomial_equation (a1 a2 a3 a4 b1 b2 b3 b4 : ℝ) : Prop :=
  ∀ x : ℝ, x^4 + a1*x^3 + a2*x^2 + a3*x + a4 = (x+1)^4 + b1*(x+1)^3 + b2*(x+1)^2 + b3*(x+1) + b4

-- Define the mapping f
def f (a1 a2 a3 a4 b1 b2 b3 b4 : ℝ) : ℝ := b1 - b2 + b3 - b4

-- Theorem statement
theorem f_value_theorem :
  ∀ b1 b2 b3 b4 : ℝ, polynomial_equation 2 0 1 6 b1 b2 b3 b4 → f 2 0 1 6 b1 b2 b3 b4 = -3 :=
by sorry

end f_value_theorem_l2107_210785


namespace scott_total_earnings_l2107_210744

/-- 
Proves that the total money Scott made from selling smoothies and cakes is $156, 
given the prices and quantities of items sold.
-/
theorem scott_total_earnings : 
  let smoothie_price : ℕ := 3
  let cake_price : ℕ := 2
  let smoothies_sold : ℕ := 40
  let cakes_sold : ℕ := 18
  
  smoothie_price * smoothies_sold + cake_price * cakes_sold = 156 := by
  sorry

end scott_total_earnings_l2107_210744


namespace grade_assignment_count_l2107_210760

/-- The number of ways to assign grades to students. -/
def assignGrades (numStudents : ℕ) (numGrades : ℕ) : ℕ :=
  numGrades ^ numStudents

/-- Theorem: The number of ways to assign 4 different grades to 15 students is 4^15. -/
theorem grade_assignment_count :
  assignGrades 15 4 = 1073741824 := by
  sorry

end grade_assignment_count_l2107_210760


namespace dormitory_problem_l2107_210717

theorem dormitory_problem (rooms : ℕ) (students : ℕ) : 
  (students % 4 = 19) ∧ 
  (0 < students - 6 * (rooms - 1)) ∧ 
  (students - 6 * (rooms - 1) < 6) →
  ((rooms = 10 ∧ students = 59) ∨ 
   (rooms = 11 ∧ students = 63) ∨ 
   (rooms = 12 ∧ students = 67)) :=
by sorry

end dormitory_problem_l2107_210717


namespace building_shadow_length_l2107_210713

/-- Given a flagstaff and a building with their respective heights and the flagstaff's shadow length,
    calculate the length of the building's shadow under similar conditions. -/
theorem building_shadow_length 
  (flagstaff_height : ℝ) 
  (flagstaff_shadow : ℝ) 
  (building_height : ℝ) 
  (h1 : flagstaff_height = 17.5)
  (h2 : flagstaff_shadow = 40.25)
  (h3 : building_height = 12.5) :
  (building_height * flagstaff_shadow) / flagstaff_height = 28.75 :=
by sorry

end building_shadow_length_l2107_210713


namespace inequality_proof_l2107_210732

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : -1 < b ∧ b < 0) :
  a * b < a * b^2 ∧ a * b^2 < a := by
  sorry

end inequality_proof_l2107_210732


namespace prob_white_balls_same_color_l2107_210735

/-- The number of white balls in the box -/
def white_balls : ℕ := 6

/-- The number of black balls in the box -/
def black_balls : ℕ := 5

/-- The number of balls drawn -/
def balls_drawn : ℕ := 3

/-- The probability that the drawn balls are white, given they are the same color -/
def prob_white_given_same_color : ℚ := 2/3

theorem prob_white_balls_same_color :
  let total_same_color := Nat.choose white_balls balls_drawn + Nat.choose black_balls balls_drawn
  let prob := (Nat.choose white_balls balls_drawn : ℚ) / total_same_color
  prob = prob_white_given_same_color := by sorry

end prob_white_balls_same_color_l2107_210735


namespace candy_purchase_calculation_l2107_210775

/-- Calculates the change and discounted price per pack for a candy purchase. -/
theorem candy_purchase_calculation (packs : ℕ) (regular_price discount payment : ℚ) 
  (h_packs : packs = 3)
  (h_regular_price : regular_price = 12)
  (h_discount : discount = 15 / 100)
  (h_payment : payment = 20) :
  let discounted_total := regular_price * (1 - discount)
  let change := payment - discounted_total
  let price_per_pack := discounted_total / packs
  change = 980 / 100 ∧ price_per_pack = 340 / 100 := by
  sorry

end candy_purchase_calculation_l2107_210775


namespace not_necessarily_right_triangle_l2107_210718

/-- A triangle with angles A, B, and C. -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  angle_sum : A + B + C = 180
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C

/-- A right triangle is a triangle with one 90-degree angle. -/
def RightTriangle (t : Triangle) : Prop :=
  t.A = 90 ∨ t.B = 90 ∨ t.C = 90

/-- The condition that angles A and B are equal and twice angle C. -/
def AngleCondition (t : Triangle) : Prop :=
  t.A = t.B ∧ t.A = 2 * t.C

theorem not_necessarily_right_triangle :
  ∃ t : Triangle, AngleCondition t ∧ ¬RightTriangle t := by
  sorry

end not_necessarily_right_triangle_l2107_210718


namespace system_solution_l2107_210719

theorem system_solution (x y : ℝ) 
  (h1 : Real.log (x + y) - Real.log 5 = Real.log x + Real.log y - Real.log 6)
  (h2 : Real.log x / (Real.log (y + 6) - (Real.log y + Real.log 6)) = -1)
  (hx : x > 0)
  (hy : y > 0)
  (hny : y ≠ 6/5)
  (hyb : y > -6) :
  x = 2 ∧ y = 3 := by
sorry

end system_solution_l2107_210719


namespace inequality_theorem_l2107_210762

theorem inequality_theorem (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : a - d > b - c := by
  sorry

end inequality_theorem_l2107_210762


namespace tamtam_yellow_shells_l2107_210752

/-- The number of shells Tamtam collected of each color --/
structure ShellCollection where
  total : ℕ
  purple : ℕ
  pink : ℕ
  blue : ℕ
  orange : ℕ

/-- Calculates the number of yellow shells in a collection --/
def yellowShells (s : ShellCollection) : ℕ :=
  s.total - (s.purple + s.pink + s.blue + s.orange)

/-- Tamtam's shell collection --/
def tamtamShells : ShellCollection :=
  { total := 65
    purple := 13
    pink := 8
    blue := 12
    orange := 14 }

/-- Theorem stating that Tamtam collected 18 yellow shells --/
theorem tamtam_yellow_shells : yellowShells tamtamShells = 18 := by
  sorry

end tamtam_yellow_shells_l2107_210752


namespace volleyball_tournament_teams_l2107_210749

/-- Represents a volleyball tournament -/
structure VolleyballTournament where
  teams : ℕ
  no_win_fraction : ℚ
  single_round : Bool

/-- Theorem: In a single round volleyball tournament where 20% of teams did not win a single game,
    the total number of teams must be 5. -/
theorem volleyball_tournament_teams
  (t : VolleyballTournament)
  (h1 : t.no_win_fraction = 1/5)
  (h2 : t.single_round = true)
  : t.teams = 5 := by
  sorry

end volleyball_tournament_teams_l2107_210749


namespace parallel_vectors_l2107_210795

theorem parallel_vectors (a b : ℝ × ℝ) :
  a = (-1, 3) →
  b.1 = 2 →
  (a.1 * b.2 = a.2 * b.1) →
  b.2 = -6 := by sorry

end parallel_vectors_l2107_210795


namespace otherSideHeadsProbabilityIsCorrect_l2107_210759

/-- Represents the three types of coins -/
inductive Coin
  | Normal
  | DoubleHeads
  | DoubleTails

/-- Represents the possible outcomes of a coin flip -/
inductive FlipResult
  | Heads
  | Tails

/-- The probability of selecting each coin -/
def coinProbability : Coin → ℚ
  | Coin.Normal => 1/3
  | Coin.DoubleHeads => 1/3
  | Coin.DoubleTails => 1/3

/-- The probability of getting heads given a specific coin -/
def headsGivenCoin : Coin → ℚ
  | Coin.Normal => 1/2
  | Coin.DoubleHeads => 1
  | Coin.DoubleTails => 0

/-- The probability that the other side is heads given that heads was observed -/
def otherSideHeadsProbability : ℚ := by sorry

theorem otherSideHeadsProbabilityIsCorrect :
  otherSideHeadsProbability = 2/3 := by sorry

end otherSideHeadsProbabilityIsCorrect_l2107_210759


namespace ninth_term_is_negative_256_l2107_210729

/-- A geometric sequence with specific properties -/
structure GeometricSequence where
  a : ℕ → ℤ
  is_geometric : ∀ n : ℕ, ∃ q : ℤ, a (n + 1) = a n * q
  a2a5 : a 2 * a 5 = -32
  a3a4_sum : a 3 + a 4 = 4

/-- The 9th term of the geometric sequence is -256 -/
theorem ninth_term_is_negative_256 (seq : GeometricSequence) : seq.a 9 = -256 := by
  sorry

end ninth_term_is_negative_256_l2107_210729


namespace consecutive_pair_sum_divisible_by_five_l2107_210763

theorem consecutive_pair_sum_divisible_by_five (n : ℕ) : 
  n < 1500 → 
  (n + (n + 1)) % 5 = 0 → 
  (57 + 58) % 5 = 0 → 
  57 = n := by
sorry

end consecutive_pair_sum_divisible_by_five_l2107_210763


namespace hiking_rate_theorem_l2107_210789

/-- Represents the hiking scenario with given conditions -/
structure HikingScenario where
  rate_up : ℝ
  time : ℝ
  route_down_length : ℝ
  rate_down_multiplier : ℝ

/-- The hiking scenario satisfies the given conditions -/
def satisfies_conditions (h : HikingScenario) : Prop :=
  h.time = 2 ∧
  h.route_down_length = 12 ∧
  h.rate_down_multiplier = 1.5

/-- The theorem stating that under the given conditions, the rate going up is 4 miles per day -/
theorem hiking_rate_theorem (h : HikingScenario) 
  (hc : satisfies_conditions h) : h.rate_up = 4 := by
  sorry

end hiking_rate_theorem_l2107_210789


namespace inequality_proof_l2107_210715

theorem inequality_proof (x : ℝ) : 
  (|(7 - x) / 4| < 3) ∧ (x ≥ 0) → (0 ≤ x ∧ x < 19) := by
  sorry

end inequality_proof_l2107_210715


namespace cross_number_puzzle_l2107_210705

def is_multiple_of_7 (n : ℕ) : Prop := ∃ k : ℕ, n = 7 * k

def is_multiple_of_odd_prime_square (n : ℕ) : Prop :=
  ∃ p : ℕ, Prime p ∧ Odd p ∧ ∃ k : ℕ, n = k * (p^2)

def is_internal_angle (n : ℕ) : Prop :=
  ∃ m : ℕ, m > 2 ∧ n = (m - 2) * 180 / m

def is_proper_factor (a b : ℕ) : Prop :=
  a ≠ 1 ∧ a ≠ b ∧ b % a = 0

theorem cross_number_puzzle :
  ∀ (across_1 across_3 across_5 down_1 down_2 down_4 : ℕ),
    across_1 > 0 ∧ across_3 > 0 ∧ across_5 > 0 ∧
    down_1 > 0 ∧ down_2 > 0 ∧ down_4 > 0 →
    is_multiple_of_7 across_1 →
    across_5 > 10 →
    is_multiple_of_odd_prime_square down_1 ∧ ¬(∃ k : ℕ, down_1 = k^2) ∧ ¬(∃ k : ℕ, down_1 = k^3) →
    is_internal_angle down_2 ∧ 170 < down_2 ∧ down_2 < 180 →
    is_proper_factor down_4 across_5 ∧ ¬is_proper_factor down_4 down_1 →
    across_3 = 961 := by
  sorry

end cross_number_puzzle_l2107_210705


namespace sum_of_roots_l2107_210790

/-- The function f(x) = x³ + 3x² + 6x + 14 -/
def f (x : ℝ) : ℝ := x^3 + 3*x^2 + 6*x + 14

/-- Theorem: If f(a) + f(b) = 20, then a + b = -2 -/
theorem sum_of_roots (a b : ℝ) (h : f a + f b = 20) : a + b = -2 := by
  sorry

end sum_of_roots_l2107_210790


namespace tv_price_reduction_l2107_210740

/-- Proves that the price reduction percentage is 10% given the conditions of the problem -/
theorem tv_price_reduction (x : ℝ) : 
  (1 - x / 100) * 1.85 = 1.665 → x = 10 := by
  sorry

end tv_price_reduction_l2107_210740


namespace bottle_caps_problem_l2107_210750

/-- The number of bottle caps left in a jar after removing some. -/
def bottle_caps_left (original : ℕ) (removed : ℕ) : ℕ :=
  original - removed

/-- Theorem stating that 40 bottle caps are left when 47 are removed from 87. -/
theorem bottle_caps_problem :
  bottle_caps_left 87 47 = 40 := by
  sorry

end bottle_caps_problem_l2107_210750


namespace vector_problem_l2107_210722

/-- Given vectors a, b, c in ℝ², prove the coordinates of c and the cosine of the angle between a and b -/
theorem vector_problem (a b c : ℝ × ℝ) : 
  a = (-Real.sqrt 2, 1) →
  (c.1 * c.1 + c.2 * c.2 = 4) →
  (∃ (k : ℝ), c = k • a) →
  (b.1 * b.1 + b.2 * b.2 = 2) →
  ((a.1 + 3 * b.1) * (a.1 - b.1) + (a.2 + 3 * b.2) * (a.2 - b.2) = 0) →
  ((c = (-2 * Real.sqrt 6 / 3, 2 * Real.sqrt 3 / 3)) ∨ 
   (c = (2 * Real.sqrt 6 / 3, -2 * Real.sqrt 3 / 3))) ∧
  ((a.1 * b.1 + a.2 * b.2) / 
   (Real.sqrt (a.1 * a.1 + a.2 * a.2) * Real.sqrt (b.1 * b.1 + b.2 * b.2)) = Real.sqrt 6 / 4) :=
by sorry

end vector_problem_l2107_210722


namespace set_union_problem_l2107_210726

theorem set_union_problem (a b : ℕ) :
  let A : Set ℕ := {0, a}
  let B : Set ℕ := {2^a, b}
  A ∪ B = {0, 1, 2} → b = 0 ∨ b = 1 := by
sorry

end set_union_problem_l2107_210726


namespace homeless_donation_distribution_l2107_210792

theorem homeless_donation_distribution (total spent second_set third_set first_set : ℚ) : 
  total = 900 ∧ second_set = 260 ∧ third_set = 315 ∧ 
  total = first_set + second_set + third_set →
  first_set = 325 := by sorry

end homeless_donation_distribution_l2107_210792


namespace sequence_problem_l2107_210776

/-- Given S_n = n^2 - 1 for all natural numbers n, prove that a_2016 = 4031 where a_n = S_n - S_(n-1) for n ≥ 2 -/
theorem sequence_problem (n : ℕ) (S : ℕ → ℤ) (a : ℕ → ℤ) 
    (h1 : ∀ n, S n = n^2 - 1)
    (h2 : ∀ n ≥ 2, a n = S n - S (n-1)) :
  a 2016 = 4031 := by
  sorry

end sequence_problem_l2107_210776


namespace sum_of_roots_l2107_210737

theorem sum_of_roots (a b : ℝ) : 
  a ≠ b → 
  let M : Set ℝ := {a^2 - 4*a, -1}
  let N : Set ℝ := {b^2 - 4*b + 1, -2}
  ∃ f : ℝ → ℝ, (∀ x ∈ M, f x = x ∧ f x ∈ N) →
  a + b = 4 := by
sorry

end sum_of_roots_l2107_210737


namespace union_equals_B_implies_B_is_real_l2107_210747

-- Define set A
def A : Set ℝ := {x : ℝ | x ≥ 0}

-- Define the theorem
theorem union_equals_B_implies_B_is_real (B : Set ℝ) (h : A ∪ B = B) : B = Set.univ :=
sorry

end union_equals_B_implies_B_is_real_l2107_210747


namespace initial_rabbits_forest_rabbits_l2107_210742

theorem initial_rabbits (initial_weasels : ℕ) (foxes : ℕ) (weeks : ℕ) 
  (weasels_caught_per_fox_per_week : ℕ) (rabbits_caught_per_fox_per_week : ℕ)
  (remaining_animals : ℕ) : ℕ :=
  let total_caught := foxes * weeks * (weasels_caught_per_fox_per_week + rabbits_caught_per_fox_per_week)
  let total_initial := remaining_animals + total_caught
  total_initial - initial_weasels

theorem forest_rabbits : 
  initial_rabbits 100 3 3 4 2 96 = 50 := by
  sorry

end initial_rabbits_forest_rabbits_l2107_210742


namespace area_of_triangle_APO_l2107_210753

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a parallelogram -/
structure Parallelogram :=
  (A B C D : Point)

/-- Area of a triangle given three points -/
def triangleArea (P Q R : Point) : ℝ := sorry

/-- Area of a parallelogram -/
def parallelogramArea (p : Parallelogram) : ℝ := sorry

/-- Check if a point is on a line segment between two other points -/
def onSegment (P Q R : Point) : Prop := sorry

/-- Check if a line bisects another line segment -/
def bisectsSegment (P Q R S : Point) : Prop := sorry

/-- Main theorem -/
theorem area_of_triangle_APO (ABCD : Parallelogram) (P Q O : Point) (k : ℝ) :
  parallelogramArea ABCD = k →
  bisectsSegment ABCD.D P ABCD.C O →
  bisectsSegment ABCD.A Q ABCD.B O →
  onSegment ABCD.A P ABCD.B →
  onSegment ABCD.C Q ABCD.D →
  triangleArea ABCD.A P O = k / 2 := by
  sorry

end area_of_triangle_APO_l2107_210753


namespace average_age_problem_l2107_210702

theorem average_age_problem (age_a age_b age_c : ℝ) :
  age_b = 20 →
  (age_a + age_c) / 2 = 29 →
  (age_a + age_b + age_c) / 3 = 26 := by
sorry

end average_age_problem_l2107_210702


namespace largest_A_k_l2107_210730

def A (k : ℕ) : ℝ := (Nat.choose 1000 k) * (0.2 ^ k)

theorem largest_A_k : 
  ∃ (k : ℕ), k = 166 ∧ 
  (∀ (j : ℕ), j ≤ 1000 → A k ≥ A j) := by
sorry

end largest_A_k_l2107_210730


namespace triangle_max_area_l2107_210700

/-- Given a triangle ABC with the following properties:
    1. (cos A / sin B) + (cos B / sin A) = 2
    2. The perimeter of the triangle is 12
    The maximum possible area of the triangle is 36(3 - 2√2) -/
theorem triangle_max_area (A B C : ℝ) (h1 : (Real.cos A / Real.sin B) + (Real.cos B / Real.sin A) = 2)
  (h2 : A + B + C = π) (h3 : Real.sin A > 0) (h4 : Real.sin B > 0) (h5 : Real.sin C > 0)
  (a b c : ℝ) (h6 : a + b + c = 12) (h7 : a > 0) (h8 : b > 0) (h9 : c > 0)
  (h10 : a / Real.sin A = b / Real.sin B) (h11 : b / Real.sin B = c / Real.sin C) :
  (1/2) * a * b * Real.sin C ≤ 36 * (3 - 2 * Real.sqrt 2) := by
  sorry

end triangle_max_area_l2107_210700


namespace coin_counting_fee_percentage_l2107_210734

def coinValue (coin : String) : ℚ :=
  match coin with
  | "quarter" => 25 / 100
  | "dime" => 10 / 100
  | "nickel" => 5 / 100
  | "penny" => 1 / 100
  | _ => 0

def totalValue (quarters dimes nickels pennies : ℕ) : ℚ :=
  quarters * coinValue "quarter" + 
  dimes * coinValue "dime" + 
  nickels * coinValue "nickel" + 
  pennies * coinValue "penny"

theorem coin_counting_fee_percentage 
  (quarters dimes nickels pennies : ℕ) 
  (amountAfterFee : ℚ) : 
  quarters = 76 → 
  dimes = 85 → 
  nickels = 20 → 
  pennies = 150 → 
  amountAfterFee = 27 → 
  (totalValue quarters dimes nickels pennies - amountAfterFee) / 
  (totalValue quarters dimes nickels pennies) = 1 / 10 := by
  sorry

end coin_counting_fee_percentage_l2107_210734


namespace reciprocal_of_negative_three_and_half_l2107_210791

theorem reciprocal_of_negative_three_and_half (x : ℚ) :
  x = -3.5 → (1 / x) = -2/7 := by sorry

end reciprocal_of_negative_three_and_half_l2107_210791


namespace amount_of_b_l2107_210781

theorem amount_of_b (a b : ℚ) : 
  a + b = 1210 → 
  (4 / 15 : ℚ) * a = (2 / 5 : ℚ) * b → 
  b = 484 := by
sorry

end amount_of_b_l2107_210781


namespace not_neighboring_root_eq1_is_neighboring_root_eq2_neighboring_root_eq3_l2107_210709

/-- Definition of a neighboring root equation -/
def is_neighboring_root_equation (a b c : ℝ) : Prop :=
  a ≠ 0 ∧ ∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 ∧ (x - y = 1 ∨ y - x = 1)

/-- Theorem for the first equation -/
theorem not_neighboring_root_eq1 : ¬is_neighboring_root_equation 1 1 (-6) :=
sorry

/-- Theorem for the second equation -/
theorem is_neighboring_root_eq2 : is_neighboring_root_equation 2 (-2 * Real.sqrt 5) 2 :=
sorry

/-- Theorem for the third equation -/
theorem neighboring_root_eq3 (m : ℝ) : 
  is_neighboring_root_equation 1 (-(m-2)) (-2*m) ↔ m = -1 ∨ m = -3 :=
sorry

end not_neighboring_root_eq1_is_neighboring_root_eq2_neighboring_root_eq3_l2107_210709


namespace arithmetic_sequence_x_value_l2107_210736

/-- An arithmetic sequence with first four terms 1, x, a, and 2x has x = 2 -/
theorem arithmetic_sequence_x_value (x a : ℝ) : 
  (∃ d : ℝ, x = 1 + d ∧ a = x + d ∧ 2*x = a + d) → x = 2 := by
sorry

end arithmetic_sequence_x_value_l2107_210736


namespace employee_pay_percentage_l2107_210704

/-- Proves that an employee's pay after a raise and subsequent cut is 75% of their pay after the raise -/
theorem employee_pay_percentage (initial_pay : ℝ) (raise_percentage : ℝ) (final_pay : ℝ) : 
  initial_pay = 10 →
  raise_percentage = 20 →
  final_pay = 9 →
  final_pay / (initial_pay * (1 + raise_percentage / 100)) = 0.75 := by
  sorry

end employee_pay_percentage_l2107_210704


namespace geometric_sequence_and_curve_max_l2107_210768

/-- Given real numbers a, b, c, and d forming a geometric sequence, 
    if the curve y = 3x - x^3 has a local maximum at x = b with the value c, 
    then ad = 2 -/
theorem geometric_sequence_and_curve_max (a b c d : ℝ) : 
  (∃ r : ℝ, r ≠ 0 ∧ b = a * r ∧ c = b * r ∧ d = c * r) →  -- geometric sequence condition
  (∀ x : ℝ, 3 * b - b^3 ≥ 3 * x - x^3) →                 -- local maximum condition
  (3 * b - b^3 = c) →                                    -- value at local maximum
  a * d = 2 := by
  sorry

end geometric_sequence_and_curve_max_l2107_210768


namespace f_zero_at_three_l2107_210720

def f (x r : ℝ) : ℝ := 3 * x^4 + 2 * x^3 - x^2 - 4 * x + r

theorem f_zero_at_three (r : ℝ) : f 3 r = 0 ↔ r = -276 := by sorry

end f_zero_at_three_l2107_210720


namespace cone_base_radius_l2107_210712

/-- A cone with surface area 3π whose lateral surface unfolds into a semicircle has base radius 1. -/
theorem cone_base_radius (r : ℝ) : 
  r > 0 → -- r is positive (implicit in the problem)
  3 * π * r^2 = 3 * π → -- surface area condition
  π * (2 * r) = 2 * π * r → -- lateral surface unfolds into semicircle condition
  r = 1 := by
sorry

end cone_base_radius_l2107_210712


namespace hcf_of_210_and_517_l2107_210707

theorem hcf_of_210_and_517 (lcm_value : ℕ) (a b : ℕ) (h_lcm : Nat.lcm a b = lcm_value) 
  (h_a : a = 210) (h_b : b = 517) (h_lcm_value : lcm_value = 2310) : Nat.gcd a b = 47 := by
  sorry

end hcf_of_210_and_517_l2107_210707


namespace isosceles_obtuse_triangle_smallest_angle_l2107_210765

/-- An isosceles, obtuse triangle with one angle 75% larger than a right angle has smallest angles of 11.25°. -/
theorem isosceles_obtuse_triangle_smallest_angle :
  ∀ (a b c : ℝ),
  0 < a ∧ 0 < b ∧ 0 < c →  -- angles are positive
  a + b + c = 180 →  -- sum of angles in a triangle
  a = b →  -- isosceles condition
  c = 90 + 0.75 * 90 →  -- largest angle is 75% larger than right angle
  a = 11.25 :=
by
  sorry

#check isosceles_obtuse_triangle_smallest_angle

end isosceles_obtuse_triangle_smallest_angle_l2107_210765


namespace domain_range_sum_l2107_210764

-- Define the function f
def f (x : ℝ) : ℝ := -x^2 + x

-- Define the theorem
theorem domain_range_sum (m n : ℝ) : 
  (∀ x, m ≤ x ∧ x ≤ n → 2*m ≤ f x ∧ f x ≤ 2*n) →
  (∀ y, 2*m ≤ y ∧ y ≤ 2*n → ∃ x, m ≤ x ∧ x ≤ n ∧ f x = y) →
  m + n = -2 := by
sorry

end domain_range_sum_l2107_210764


namespace basketball_probability_l2107_210794

-- Define the success rate
def success_rate : ℚ := 1/2

-- Define the total number of shots
def total_shots : ℕ := 10

-- Define the number of successful shots we're interested in
def successful_shots : ℕ := 3

-- Define the probability function
def probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k) * p^k * (1-p)^(n-k)

-- Theorem statement
theorem basketball_probability :
  probability total_shots successful_shots success_rate = 15/128 := by
  sorry

end basketball_probability_l2107_210794

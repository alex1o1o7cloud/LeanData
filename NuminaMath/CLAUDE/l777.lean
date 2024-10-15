import Mathlib

namespace NUMINAMATH_CALUDE_james_pizza_fraction_l777_77700

theorem james_pizza_fraction (num_pizzas : ℕ) (slices_per_pizza : ℕ) (james_slices : ℕ) :
  num_pizzas = 2 →
  slices_per_pizza = 6 →
  james_slices = 8 →
  (james_slices : ℚ) / (num_pizzas * slices_per_pizza : ℚ) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_james_pizza_fraction_l777_77700


namespace NUMINAMATH_CALUDE_tournament_matches_l777_77761

/-- A tournament with the given rules --/
structure Tournament :=
  (num_players : ℕ)
  (num_players_per_match : ℕ)
  (points_per_match : Fin 3 → ℕ)
  (eliminated_per_match : ℕ)

/-- The number of matches played in a tournament --/
def num_matches (t : Tournament) : ℕ :=
  t.num_players - 1

/-- The theorem stating the number of matches in the specific tournament --/
theorem tournament_matches :
  ∀ t : Tournament,
    t.num_players = 999 ∧
    t.num_players_per_match = 3 ∧
    t.points_per_match 0 = 2 ∧
    t.points_per_match 1 = 1 ∧
    t.points_per_match 2 = 0 ∧
    t.eliminated_per_match = 1 →
    num_matches t = 998 :=
by sorry

end NUMINAMATH_CALUDE_tournament_matches_l777_77761


namespace NUMINAMATH_CALUDE_marshmallow_challenge_l777_77752

/-- The number of marshmallows Haley can hold -/
def haley_marshmallows : ℕ := sorry

/-- The number of marshmallows Michael can hold -/
def michael_marshmallows : ℕ := 3 * haley_marshmallows

/-- The number of marshmallows Brandon can hold -/
def brandon_marshmallows : ℕ := michael_marshmallows / 2

/-- The total number of marshmallows all three kids can hold -/
def total_marshmallows : ℕ := 44

theorem marshmallow_challenge : 
  haley_marshmallows + michael_marshmallows + brandon_marshmallows = total_marshmallows ∧ 
  haley_marshmallows = 8 := by sorry

end NUMINAMATH_CALUDE_marshmallow_challenge_l777_77752


namespace NUMINAMATH_CALUDE_marys_nickels_l777_77736

/-- Given Mary's initial nickels and the additional nickels from her dad,
    calculate the total number of nickels Mary has now. -/
theorem marys_nickels (initial : ℕ) (additional : ℕ) : 
  initial = 7 → additional = 5 → initial + additional = 12 := by
  sorry

end NUMINAMATH_CALUDE_marys_nickels_l777_77736


namespace NUMINAMATH_CALUDE_inequality_necessary_not_sufficient_l777_77783

-- Define what it means for the equation to represent an ellipse
def represents_ellipse (m : ℝ) : Prop :=
  5 - m > 0 ∧ m + 3 > 0 ∧ 5 - m ≠ m + 3

-- Define the inequality condition
def inequality_condition (m : ℝ) : Prop :=
  -3 < m ∧ m < 5

-- Theorem statement
theorem inequality_necessary_not_sufficient :
  (∀ m : ℝ, represents_ellipse m → inequality_condition m) ∧
  (∃ m : ℝ, inequality_condition m ∧ ¬represents_ellipse m) :=
sorry

end NUMINAMATH_CALUDE_inequality_necessary_not_sufficient_l777_77783


namespace NUMINAMATH_CALUDE_mirella_purple_books_l777_77738

/-- The number of pages in each purple book -/
def purple_book_pages : ℕ := 230

/-- The number of pages in each orange book -/
def orange_book_pages : ℕ := 510

/-- The number of orange books Mirella read -/
def orange_books_read : ℕ := 4

/-- The difference between orange and purple pages read -/
def page_difference : ℕ := 890

/-- The number of purple books Mirella read -/
def purple_books_read : ℕ := 5

theorem mirella_purple_books :
  orange_books_read * orange_book_pages - purple_books_read * purple_book_pages = page_difference :=
by sorry

end NUMINAMATH_CALUDE_mirella_purple_books_l777_77738


namespace NUMINAMATH_CALUDE_negation_of_proposition_l777_77767

theorem negation_of_proposition (a b x : ℝ) :
  (¬(x ≥ a^2 + b^2 → x ≥ 2*a*b)) ↔ (x < a^2 + b^2 → x < 2*a*b) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l777_77767


namespace NUMINAMATH_CALUDE_monotonic_quadratic_l777_77763

def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x - 3

def monotonic_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y ∨ (∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x > f y)

theorem monotonic_quadratic (a : ℝ) :
  monotonic_on (f a) 1 2 ↔ a ≤ 1 ∨ a ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_monotonic_quadratic_l777_77763


namespace NUMINAMATH_CALUDE_geometric_sequence_a12_l777_77785

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_a12 (a : ℕ → ℝ) :
  geometric_sequence a →
  a 6 * a 10 = 16 →
  a 4 = 1 →
  a 12 = 16 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_a12_l777_77785


namespace NUMINAMATH_CALUDE_rockham_soccer_league_members_l777_77789

theorem rockham_soccer_league_members :
  let sock_cost : ℕ := 6
  let tshirt_cost : ℕ := sock_cost + 10
  let cap_cost : ℕ := 3
  let items_per_member : ℕ := 2  -- for both home and away games
  let cost_per_member : ℕ := items_per_member * (sock_cost + tshirt_cost + cap_cost)
  let total_expenditure : ℕ := 4620
  total_expenditure / cost_per_member = 92 :=
by sorry

end NUMINAMATH_CALUDE_rockham_soccer_league_members_l777_77789


namespace NUMINAMATH_CALUDE_union_when_a_neg_two_subset_condition_l777_77779

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | (x - 2) * (x - (3 * a + 1)) < 0}
def B (a : ℝ) : Set ℝ := {x | 2 * a < x ∧ x < a^2 + 1}

-- Statement for part (i)
theorem union_when_a_neg_two :
  A (-2) ∪ B (-2) = {x : ℝ | -5 < x ∧ x < 5} := by sorry

-- Statement for part (ii)
theorem subset_condition :
  ∀ a : ℝ, B a ⊆ A a ↔ a ∈ ({x : ℝ | 1 ≤ x ∧ x ≤ 3} ∪ {-1}) := by sorry

end NUMINAMATH_CALUDE_union_when_a_neg_two_subset_condition_l777_77779


namespace NUMINAMATH_CALUDE_smallest_a_value_l777_77724

theorem smallest_a_value (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) 
  (h : ∀ x : ℤ, Real.cos (a * ↑x + b) = Real.cos (31 * ↑x)) :
  ∀ a' ≥ 0, (∀ x : ℤ, Real.cos (a' * ↑x + b) = Real.cos (31 * ↑x)) → a' ≥ 31 :=
sorry

end NUMINAMATH_CALUDE_smallest_a_value_l777_77724


namespace NUMINAMATH_CALUDE_theresa_final_count_l777_77781

/-- Represents the number of crayons each person has -/
structure CrayonCount where
  theresa : ℕ
  janice : ℕ
  nancy : ℕ
  mark : ℕ

/-- Represents the initial state and actions taken -/
def initial_state : CrayonCount := {
  theresa := 32,
  janice := 12,
  nancy := 0,
  mark := 0
}

/-- Janice shares half of her crayons with Nancy and gives 3 to Mark -/
def share_crayons (state : CrayonCount) : CrayonCount := {
  theresa := state.theresa,
  janice := state.janice - (state.janice / 2) - 3,
  nancy := state.nancy + (state.janice / 2),
  mark := state.mark + 3
}

/-- Nancy gives 8 crayons to Theresa -/
def give_to_theresa (state : CrayonCount) : CrayonCount := {
  theresa := state.theresa + 8,
  janice := state.janice,
  nancy := state.nancy - 8,
  mark := state.mark
}

/-- The final state after all actions -/
def final_state : CrayonCount := give_to_theresa (share_crayons initial_state)

theorem theresa_final_count : final_state.theresa = 40 := by
  sorry

end NUMINAMATH_CALUDE_theresa_final_count_l777_77781


namespace NUMINAMATH_CALUDE_josh_spending_l777_77798

def initial_amount : ℚ := 9
def drink_cost : ℚ := 1.75
def final_amount : ℚ := 6

theorem josh_spending (amount_spent_after_drink : ℚ) : 
  initial_amount - drink_cost - amount_spent_after_drink = final_amount → 
  amount_spent_after_drink = 1.25 := by
sorry

end NUMINAMATH_CALUDE_josh_spending_l777_77798


namespace NUMINAMATH_CALUDE_no_real_roots_quadratic_l777_77791

theorem no_real_roots_quadratic (m : ℝ) :
  (∀ x : ℝ, x^2 + m*x + 1 ≠ 0) ↔ -2 < m ∧ m < 2 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_quadratic_l777_77791


namespace NUMINAMATH_CALUDE_divisibility_condition_l777_77764

theorem divisibility_condition (a b : ℕ+) :
  (a * b^2 + b + 7) ∣ (a^2 * b + a + b) ↔
  (∃ k : ℕ+, a = 7 * k^2 ∧ b = 7 * k) ∨ (a = 11 ∧ b = 1) ∨ (a = 49 ∧ b = 1) :=
sorry

end NUMINAMATH_CALUDE_divisibility_condition_l777_77764


namespace NUMINAMATH_CALUDE_train_speed_train_speed_approx_66_l777_77726

/-- The speed of a train given its length, the time it takes to pass a man running in the opposite direction, and the man's speed. -/
theorem train_speed (train_length : ℝ) (passing_time : ℝ) (man_speed_kmh : ℝ) : ℝ :=
  let man_speed_ms := man_speed_kmh * (1000 / 3600)
  let relative_speed := train_length / passing_time
  let train_speed_ms := relative_speed - man_speed_ms
  let train_speed_kmh := train_speed_ms * (3600 / 1000)
  train_speed_kmh

/-- The speed of the train is approximately 66 km/h given the specified conditions. -/
theorem train_speed_approx_66 :
  ∃ ε > 0, abs (train_speed 120 6 6 - 66) < ε :=
sorry

end NUMINAMATH_CALUDE_train_speed_train_speed_approx_66_l777_77726


namespace NUMINAMATH_CALUDE_apollo_total_cost_l777_77711

/-- Represents the cost structure for a blacksmith --/
structure BlacksmithCost where
  monthly_rates : List ℕ
  installation_fee : ℕ
  installation_frequency : ℕ

/-- Calculates the total cost for a blacksmith for a year --/
def calculate_blacksmith_cost (cost : BlacksmithCost) : ℕ :=
  (cost.monthly_rates.sum) + 
  (12 / cost.installation_frequency * cost.installation_fee)

/-- Hephaestus's cost structure --/
def hephaestus_cost : BlacksmithCost := {
  monthly_rates := [3, 3, 3, 3, 6, 6, 6, 6, 9, 9, 9, 9],
  installation_fee := 2,
  installation_frequency := 1
}

/-- Athena's cost structure --/
def athena_cost : BlacksmithCost := {
  monthly_rates := [5, 5, 5, 5, 5, 5, 7, 7, 7, 7, 7, 7],
  installation_fee := 10,
  installation_frequency := 12
}

/-- Ares's cost structure --/
def ares_cost : BlacksmithCost := {
  monthly_rates := [4, 4, 4, 6, 6, 6, 6, 6, 6, 8, 8, 8],
  installation_fee := 3,
  installation_frequency := 3
}

/-- The total cost for Apollo's chariot wheels for a year --/
theorem apollo_total_cost : 
  calculate_blacksmith_cost hephaestus_cost + 
  calculate_blacksmith_cost athena_cost + 
  calculate_blacksmith_cost ares_cost = 265 := by
  sorry

end NUMINAMATH_CALUDE_apollo_total_cost_l777_77711


namespace NUMINAMATH_CALUDE_f_properties_l777_77743

-- Define the function f
def f (a b x : ℝ) : ℝ := 3 * a * x^2 - 2 * (a + b) * x + b

-- State the theorem
theorem f_properties (a b : ℝ) (ha : a > 0) :
  -- Part I
  (b = 1/2 ∧ 
   ∃ x₁ x₂ : ℝ, 0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 1 ∧ 
   f a (1/2) x₁ = |x₁ - 1/2| ∧ 
   f a (1/2) x₂ = |x₂ - 1/2|) →
  a ≥ 1 ∧
  -- Part II
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → |f a b 0| ≤ 2 ∧ |f a b 1| ≤ 2) →
  ∃ M : ℝ, M = 2 ∧ ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → |f a b x| ≤ M :=
by
  sorry

end NUMINAMATH_CALUDE_f_properties_l777_77743


namespace NUMINAMATH_CALUDE_sticker_distribution_ways_l777_77751

/-- The number of ways to distribute stickers across sheets of paper -/
def distribute_stickers (total_stickers : ℕ) (total_sheets : ℕ) : ℕ :=
  Nat.choose (total_stickers - total_sheets + total_sheets - 1) (total_sheets - 1)

/-- Theorem: There are 126 ways to distribute 10 stickers across 5 sheets -/
theorem sticker_distribution_ways :
  distribute_stickers 10 5 = 126 := by
  sorry

#eval distribute_stickers 10 5

end NUMINAMATH_CALUDE_sticker_distribution_ways_l777_77751


namespace NUMINAMATH_CALUDE_correct_average_marks_l777_77757

/-- Calculates the correct average marks for a class given the following conditions:
  * There are 40 students in the class
  * The reported average marks are 65
  * Three students' marks were wrongly noted:
    - First student: 100 instead of 20
    - Second student: 85 instead of 50
    - Third student: 15 instead of 55
-/
theorem correct_average_marks (num_students : ℕ) (reported_average : ℚ)
  (incorrect_mark1 incorrect_mark2 incorrect_mark3 : ℕ)
  (correct_mark1 correct_mark2 correct_mark3 : ℕ) :
  num_students = 40 →
  reported_average = 65 →
  incorrect_mark1 = 100 →
  incorrect_mark2 = 85 →
  incorrect_mark3 = 15 →
  correct_mark1 = 20 →
  correct_mark2 = 50 →
  correct_mark3 = 55 →
  (num_students * reported_average - (incorrect_mark1 + incorrect_mark2 + incorrect_mark3) +
    (correct_mark1 + correct_mark2 + correct_mark3)) / num_students = 63125 / 1000 := by
  sorry

end NUMINAMATH_CALUDE_correct_average_marks_l777_77757


namespace NUMINAMATH_CALUDE_dividing_line_b_range_l777_77717

/-- Triangle ABC with vertices A(-1,0), B(1,0), and C(0,1) -/
structure Triangle where
  A : ℝ × ℝ := (-1, 0)
  B : ℝ × ℝ := (1, 0)
  C : ℝ × ℝ := (0, 1)

/-- Line y = ax + b that divides the triangle -/
structure DividingLine where
  a : ℝ
  b : ℝ
  a_pos : a > 0

/-- The line divides the triangle into two parts of equal area -/
def dividesEqualArea (t : Triangle) (l : DividingLine) : Prop := sorry

/-- The range of b values that satisfy the condition -/
def validRange : Set ℝ := Set.Ioo (1 - Real.sqrt 2 / 2) (1 / 2)

/-- Theorem stating the range of b values -/
theorem dividing_line_b_range (t : Triangle) (l : DividingLine) 
  (h : dividesEqualArea t l) : l.b ∈ validRange := by
  sorry

end NUMINAMATH_CALUDE_dividing_line_b_range_l777_77717


namespace NUMINAMATH_CALUDE_log_product_equals_four_l777_77762

theorem log_product_equals_four : (Real.log 9 / Real.log 2) * (Real.log 4 / Real.log 3) = 4 := by
  sorry

end NUMINAMATH_CALUDE_log_product_equals_four_l777_77762


namespace NUMINAMATH_CALUDE_mod_seven_equality_l777_77737

theorem mod_seven_equality : (45^1234 - 25^1234) % 7 = 5 := by
  sorry

end NUMINAMATH_CALUDE_mod_seven_equality_l777_77737


namespace NUMINAMATH_CALUDE_multiple_of_72_digits_l777_77775

theorem multiple_of_72_digits (n : ℕ) (x y : Fin 10) :
  (n = 320000000 + x * 10000000 + 35717 * 10 + y) →
  (n % 72 = 0) →
  (x * y = 12) :=
by sorry

end NUMINAMATH_CALUDE_multiple_of_72_digits_l777_77775


namespace NUMINAMATH_CALUDE_energy_saving_product_analysis_l777_77721

/-- Represents the sales volume in ten thousand items -/
def y (x : ℝ) : ℝ := -x + 120

/-- Represents the profit in ten thousand dollars -/
def W (x : ℝ) : ℝ := -(x - 100)^2 - 80

/-- Represents the profit in the second year considering donations -/
def W2 (x : ℝ) : ℝ := (x - 82) * (-x + 120)

theorem energy_saving_product_analysis :
  (∀ x, 90 ≤ x → x ≤ 110 → y x = -x + 120) ∧
  (∀ x, 90 ≤ x ∧ x ≤ 110 → W x ≤ 0) ∧
  (∃ x, 90 ≤ x ∧ x ≤ 110 ∧ W x = -80) ∧
  (∃ x, 92 ≤ x ∧ x ≤ 110 ∧ W2 x ≥ 280 ∧
    ∀ x', 92 ≤ x' ∧ x' ≤ 110 → W2 x' ≤ W2 x) :=
by sorry

end NUMINAMATH_CALUDE_energy_saving_product_analysis_l777_77721


namespace NUMINAMATH_CALUDE_fourth_root_63504000_l777_77748

theorem fourth_root_63504000 : 
  (63504000 : ℝ)^(1/4) = 2 * (2 : ℝ)^(1/2) * (3 : ℝ)^(1/2) * (11 : ℝ)^(1/4) * 10^(3/4) := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_63504000_l777_77748


namespace NUMINAMATH_CALUDE_potion_price_l777_77793

theorem potion_price (discounted_price original_price : ℝ) : 
  discounted_price = 8 → 
  discounted_price = (1 / 5) * original_price → 
  original_price = 40 := by
  sorry

end NUMINAMATH_CALUDE_potion_price_l777_77793


namespace NUMINAMATH_CALUDE_triangle_inequality_l777_77730

theorem triangle_inequality (a b c : ℝ) 
  (h_nonneg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0) 
  (h_triangle : a^2 + b^2 ≥ c^2 ∧ b^2 + c^2 ≥ a^2 ∧ c^2 + a^2 ≥ b^2) :
  (a + b + c) * (a^2 + b^2 + c^2) * (a^3 + b^3 + c^3) ≥ 4 * (a^6 + b^6 + c^6) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l777_77730


namespace NUMINAMATH_CALUDE_special_fraction_equality_l777_77728

theorem special_fraction_equality (a b : ℝ) 
  (h : a / (1 + a) + b / (1 + b) = 1) : 
  a / (1 + b^2) - b / (1 + a^2) = a - b := by
  sorry

end NUMINAMATH_CALUDE_special_fraction_equality_l777_77728


namespace NUMINAMATH_CALUDE_birdhouse_to_lawn_chair_ratio_l777_77795

def car_distance : ℝ := 200
def lawn_chair_distance : ℝ := 2 * car_distance
def birdhouse_distance : ℝ := 1200

theorem birdhouse_to_lawn_chair_ratio :
  birdhouse_distance / lawn_chair_distance = 3 := by sorry

end NUMINAMATH_CALUDE_birdhouse_to_lawn_chair_ratio_l777_77795


namespace NUMINAMATH_CALUDE_sarah_snack_purchase_l777_77782

/-- The number of dimes used by Sarah to buy a $2 snack -/
def num_dimes : ℕ := 10

theorem sarah_snack_purchase :
  ∃ (n : ℕ),
    num_dimes + n = 50 ∧
    10 * num_dimes + 5 * n = 200 :=
by sorry

end NUMINAMATH_CALUDE_sarah_snack_purchase_l777_77782


namespace NUMINAMATH_CALUDE_sufficient_condition_increasing_f_increasing_on_interval_l777_77787

/-- A sufficient condition for f(x) = x^2 + 2ax + 1 to be increasing on (1, +∞) -/
theorem sufficient_condition_increasing (a : ℝ) (h : a = -1) :
  ∀ x y, 1 < x → x < y → x^2 + 2*a*x + 1 < y^2 + 2*a*y + 1 := by
  sorry

/-- Definition of the function f(x) = x^2 + 2ax + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x + 1

/-- The function f is increasing on (1, +∞) when a = -1 -/
theorem f_increasing_on_interval (a : ℝ) (h : a = -1) :
  StrictMonoOn (f a) (Set.Ioi 1) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_increasing_f_increasing_on_interval_l777_77787


namespace NUMINAMATH_CALUDE_cake_brownie_calorie_difference_l777_77778

/-- Represents the number of slices in the cake -/
def cake_slices : ℕ := 8

/-- Represents the number of calories in each cake slice -/
def calories_per_cake_slice : ℕ := 347

/-- Represents the number of brownies in a pan -/
def brownies_count : ℕ := 6

/-- Represents the number of calories in each brownie -/
def calories_per_brownie : ℕ := 375

/-- Theorem stating the difference in total calories between the cake and the brownies -/
theorem cake_brownie_calorie_difference :
  cake_slices * calories_per_cake_slice - brownies_count * calories_per_brownie = 526 := by
  sorry


end NUMINAMATH_CALUDE_cake_brownie_calorie_difference_l777_77778


namespace NUMINAMATH_CALUDE_anns_shopping_trip_l777_77716

/-- Calculates the cost of each top in Ann's shopping trip -/
def cost_per_top (total_spent : ℚ) (num_shorts : ℕ) (price_shorts : ℚ) 
  (num_shoes : ℕ) (price_shoes : ℚ) (num_tops : ℕ) : ℚ :=
  let total_shorts := num_shorts * price_shorts
  let total_shoes := num_shoes * price_shoes
  let total_tops := total_spent - total_shorts - total_shoes
  total_tops / num_tops

/-- Proves that the cost per top is $5 given the conditions of Ann's shopping trip -/
theorem anns_shopping_trip : 
  cost_per_top 75 5 7 2 10 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_anns_shopping_trip_l777_77716


namespace NUMINAMATH_CALUDE_starfish_arms_l777_77741

theorem starfish_arms (num_starfish : ℕ) (seastar_arms : ℕ) (total_arms : ℕ) :
  num_starfish = 7 →
  seastar_arms = 14 →
  total_arms = 49 →
  ∃ (starfish_arms : ℕ), num_starfish * starfish_arms + seastar_arms = total_arms ∧ starfish_arms = 5 :=
by sorry

end NUMINAMATH_CALUDE_starfish_arms_l777_77741


namespace NUMINAMATH_CALUDE_distribute_students_count_l777_77749

/-- The number of ways to distribute 4 students among 3 universities --/
def distribute_students : ℕ :=
  -- We define the function without implementation
  sorry

/-- Theorem stating that the number of ways to distribute 4 students
    among 3 universities, with each university receiving at least 1 student,
    is equal to 36 --/
theorem distribute_students_count :
  distribute_students = 36 := by
  sorry

end NUMINAMATH_CALUDE_distribute_students_count_l777_77749


namespace NUMINAMATH_CALUDE_floor_sqrt_inequality_l777_77792

theorem floor_sqrt_inequality (x : ℝ) : 
  150 ≤ x ∧ x ≤ 300 ∧ ⌊Real.sqrt x⌋ = 16 → ⌊Real.sqrt (10 * x)⌋ ≠ 160 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_inequality_l777_77792


namespace NUMINAMATH_CALUDE_smallest_integers_difference_smallest_integers_difference_exists_l777_77702

theorem smallest_integers_difference : ℕ → Prop := fun n =>
  (∃ m : ℕ, m > 1 ∧ 
    (∀ k : ℕ, 2 ≤ k → k ≤ 13 → m % k = 1) ∧
    (∀ j : ℕ, j > 1 → 
      (∀ k : ℕ, 2 ≤ k → k ≤ 13 → j % k = 1) → 
      j ≥ m) ∧
    (∃ p : ℕ, p > m ∧ 
      (∀ k : ℕ, 2 ≤ k → k ≤ 13 → p % k = 1) ∧
      (∀ q : ℕ, q > m → 
        (∀ k : ℕ, 2 ≤ k → k ≤ 13 → q % k = 1) → 
        q ≥ p) ∧
      p - m = n)) →
  n = 360360

theorem smallest_integers_difference_exists : 
  ∃ n : ℕ, smallest_integers_difference n := by sorry

end NUMINAMATH_CALUDE_smallest_integers_difference_smallest_integers_difference_exists_l777_77702


namespace NUMINAMATH_CALUDE_domain_of_f_2x_minus_1_l777_77733

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f(x+1)
def domain_f_x_plus_1 : Set ℝ := Set.Icc (-2) 3

-- State the theorem
theorem domain_of_f_2x_minus_1 :
  (∀ x ∈ domain_f_x_plus_1, f (x + 1) = f (x + 1)) →
  {x : ℝ | f (2 * x - 1) = f (2 * x - 1)} = Set.Icc 0 (5/2) := by sorry

end NUMINAMATH_CALUDE_domain_of_f_2x_minus_1_l777_77733


namespace NUMINAMATH_CALUDE_regular_polygon_with_12_degree_exterior_angles_has_30_sides_l777_77786

/-- A regular polygon with exterior angles measuring 12 degrees has 30 sides. -/
theorem regular_polygon_with_12_degree_exterior_angles_has_30_sides :
  ∀ n : ℕ, 
  n > 0 →
  (360 : ℝ) / n = 12 →
  n = 30 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_with_12_degree_exterior_angles_has_30_sides_l777_77786


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l777_77774

theorem complex_magnitude_problem (z w : ℂ) 
  (h1 : Complex.abs z = 1)
  (h2 : Complex.abs w = 2)
  (h3 : Complex.arg z = Real.pi / 2)
  (h4 : Complex.abs (z + w) = 3) :
  Complex.abs (1 / z + 1 / w) = (Real.sqrt (5 + 2 * Real.sqrt 2)) / (2 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l777_77774


namespace NUMINAMATH_CALUDE_joshua_and_justin_shared_money_l777_77708

/-- Given that Joshua's share is $30 and it is thrice as much as Justin's share,
    prove that the total amount of money shared by Joshua and Justin is $40. -/
theorem joshua_and_justin_shared_money (joshua_share : ℕ) (justin_share : ℕ) : 
  joshua_share = 30 → joshua_share = 3 * justin_share → joshua_share + justin_share = 40 := by
  sorry

end NUMINAMATH_CALUDE_joshua_and_justin_shared_money_l777_77708


namespace NUMINAMATH_CALUDE_cube_paint_theorem_l777_77713

/-- Represents a cube with edge length n -/
structure Cube (n : ℕ) where
  edge_length : n > 0

/-- Count of unit cubes with one red face -/
def one_face_count (n : ℕ) : ℕ := 6 * (n - 2)^2

/-- Count of unit cubes with two red faces -/
def two_face_count (n : ℕ) : ℕ := 12 * (n - 2)

/-- The main theorem stating the condition for n = 26 -/
theorem cube_paint_theorem (n : ℕ) (c : Cube n) :
  one_face_count n = 12 * two_face_count n ↔ n = 26 := by
  sorry

#check cube_paint_theorem

end NUMINAMATH_CALUDE_cube_paint_theorem_l777_77713


namespace NUMINAMATH_CALUDE_dice_game_probability_l777_77772

theorem dice_game_probability (n : ℕ) (max_score : ℕ) (num_dice : ℕ) (num_faces : ℕ) :
  let p_max_score := (1 / num_faces : ℚ) ^ num_dice
  let p_not_max_score := 1 - p_max_score
  n = 23 ∧ max_score = 18 ∧ num_dice = 3 ∧ num_faces = 6 →
  p_max_score * p_not_max_score ^ (n - 1) = (1 / 216 : ℚ) * (1 - 1 / 216 : ℚ) ^ 22 :=
by sorry

end NUMINAMATH_CALUDE_dice_game_probability_l777_77772


namespace NUMINAMATH_CALUDE_fraction_simplification_l777_77709

theorem fraction_simplification :
  (3 : ℝ) / (Real.sqrt 75 + Real.sqrt 48 + Real.sqrt 18) = Real.sqrt 3 / 12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l777_77709


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l777_77796

/-- Given two points P and Q symmetric with respect to the y-axis, prove that the sum of their x-coordinates is -8 --/
theorem symmetric_points_sum (a b : ℝ) : 
  (∃ (P Q : ℝ × ℝ), P = (a, -3) ∧ Q = (4, b) ∧ 
   (P.1 = -Q.1) ∧ (P.2 = Q.2)) → 
  a + b = -7 :=
by sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l777_77796


namespace NUMINAMATH_CALUDE_seating_arrangements_count_l777_77707

/-- Represents the number of people in the group -/
def num_people : Nat := 5

/-- Represents the number of seats in the car -/
def num_seats : Nat := 5

/-- Represents the number of people who can drive (Mr. and Mrs. Lopez) -/
def num_drivers : Nat := 2

/-- Calculates the number of seating arrangements -/
def seating_arrangements : Nat :=
  num_drivers * (num_people - 1) * Nat.factorial (num_seats - 2)

/-- Theorem stating that the number of seating arrangements is 48 -/
theorem seating_arrangements_count :
  seating_arrangements = 48 := by sorry

end NUMINAMATH_CALUDE_seating_arrangements_count_l777_77707


namespace NUMINAMATH_CALUDE_quadratic_factorization_l777_77797

/-- A quadratic expression can be factored completely if and only if its discriminant is a perfect square. -/
def is_factorable (a b c : ℝ) : Prop :=
  ∃ k : ℤ, (b^2 - 4*a*c : ℝ) = (k : ℝ)^2

theorem quadratic_factorization (m : ℝ) :
  (is_factorable 1 (3 - m) 25) → (m = -7 ∨ m = 13) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l777_77797


namespace NUMINAMATH_CALUDE_two_less_than_negative_one_l777_77780

theorem two_less_than_negative_one : -1 - 2 = -3 := by
  sorry

end NUMINAMATH_CALUDE_two_less_than_negative_one_l777_77780


namespace NUMINAMATH_CALUDE_f_root_condition_and_inequality_l777_77750

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a / x

theorem f_root_condition_and_inequality (a : ℝ) (b : ℝ) :
  (a > 0 ∧ (∃ x > 0, f a x = 0) ↔ 0 < a ∧ a ≤ 1 / Real.exp 1) ∧
  (a ≥ 2 / Real.exp 1 ∧ b > 1 → f a (Real.log b) > 1 / b) := by
  sorry

end NUMINAMATH_CALUDE_f_root_condition_and_inequality_l777_77750


namespace NUMINAMATH_CALUDE_regular_polygon_144_degree_angles_has_10_sides_l777_77747

/-- A regular polygon with interior angles of 144 degrees has 10 sides. -/
theorem regular_polygon_144_degree_angles_has_10_sides :
  ∀ n : ℕ,
  n > 2 →
  (180 * (n - 2) : ℝ) = 144 * n →
  n = 10 :=
by sorry

end NUMINAMATH_CALUDE_regular_polygon_144_degree_angles_has_10_sides_l777_77747


namespace NUMINAMATH_CALUDE_fraction_product_result_l777_77768

def fraction_product (n : ℕ) : ℚ :=
  if n < 6 then 1
  else (n : ℚ) / (n + 5) * fraction_product (n - 1)

theorem fraction_product_result : fraction_product 95 = 1 / 75287520 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_result_l777_77768


namespace NUMINAMATH_CALUDE_shortest_distance_line_to_circle_l777_77701

/-- The shortest distance from a point on the line y=x-1 to the circle x^2+y^2+4x-2y+4=0 is 2√2 - 1 -/
theorem shortest_distance_line_to_circle : ∃ d : ℝ, d = 2 * Real.sqrt 2 - 1 ∧
  ∀ (x y : ℝ),
    (y = x - 1) →
    (x^2 + y^2 + 4*x - 2*y + 4 = 0) →
    d ≤ Real.sqrt ((x - 0)^2 + (y - 0)^2) :=
by sorry

end NUMINAMATH_CALUDE_shortest_distance_line_to_circle_l777_77701


namespace NUMINAMATH_CALUDE_identity_function_proof_l777_77784

theorem identity_function_proof (f : ℝ → ℝ) 
  (h_increasing : ∀ x y, x < y → f x < f y) 
  (h_inverse : ∀ x, f (f x) = x) : 
  ∀ x, f x = x := by
sorry

end NUMINAMATH_CALUDE_identity_function_proof_l777_77784


namespace NUMINAMATH_CALUDE_division_multiplication_identity_l777_77753

theorem division_multiplication_identity (a : ℝ) (h : a ≠ 0) : 1 / a * a = 1 := by
  sorry

end NUMINAMATH_CALUDE_division_multiplication_identity_l777_77753


namespace NUMINAMATH_CALUDE_contest_end_time_l777_77776

-- Define a custom time type
structure Time where
  hours : Nat
  minutes : Nat

-- Define a function to add minutes to a time
def addMinutes (t : Time) (m : Nat) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + m
  { hours := totalMinutes / 60 % 24, minutes := totalMinutes % 60 }

-- Define the start time (3:00 p.m.)
def startTime : Time := { hours := 15, minutes := 0 }

-- Define the duration in minutes
def duration : Nat := 720

-- Theorem to prove
theorem contest_end_time :
  addMinutes startTime duration = { hours := 3, minutes := 0 } := by
  sorry

end NUMINAMATH_CALUDE_contest_end_time_l777_77776


namespace NUMINAMATH_CALUDE_binary_to_octal_conversion_l777_77745

/-- Converts a binary number represented as a list of bits to a natural number. -/
def binary_to_natural (bits : List Bool) : ℕ :=
  bits.foldr (fun b n => 2 * n + if b then 1 else 0) 0

/-- Converts a natural number to its octal representation as a list of digits. -/
def natural_to_octal (n : ℕ) : List ℕ :=
  if n < 8 then [n]
  else (n % 8) :: natural_to_octal (n / 8)

/-- The binary representation of the number to be converted. -/
def binary_number : List Bool := [true, true, true, false, false]

/-- The expected octal representation. -/
def expected_octal : List ℕ := [4, 3]

theorem binary_to_octal_conversion :
  natural_to_octal (binary_to_natural binary_number) = expected_octal := by
  sorry

#eval binary_to_natural binary_number
#eval natural_to_octal (binary_to_natural binary_number)

end NUMINAMATH_CALUDE_binary_to_octal_conversion_l777_77745


namespace NUMINAMATH_CALUDE_time_after_850_hours_l777_77754

/-- Represents a time on a 12-hour clock -/
structure Time12Hour where
  hour : Nat
  minute : Nat
  period : Bool  -- false for AM, true for PM
  h_valid : hour ≥ 1 ∧ hour ≤ 12
  m_valid : minute ≥ 0 ∧ minute < 60

/-- Adds hours to a given time on a 12-hour clock -/
def addHours (t : Time12Hour) (h : Nat) : Time12Hour :=
  sorry

theorem time_after_850_hours : 
  let start_time := Time12Hour.mk 3 15 true (by norm_num) (by norm_num)
  let end_time := Time12Hour.mk 1 15 false (by norm_num) (by norm_num)
  addHours start_time 850 = end_time := by sorry

end NUMINAMATH_CALUDE_time_after_850_hours_l777_77754


namespace NUMINAMATH_CALUDE_square_perimeter_problem_l777_77756

/-- Given squares A and B with perimeters 16 and 32 respectively, 
    when placed side by side to form square C, the perimeter of C is 48. -/
theorem square_perimeter_problem (A B C : ℝ → ℝ → Prop) :
  (∀ x, A x x → 4 * x = 16) →  -- Square A has perimeter 16
  (∀ y, B y y → 4 * y = 32) →  -- Square B has perimeter 32
  (∀ z, C z z → ∃ x y, A x x ∧ B y y ∧ z = x + y) →  -- C is formed by A and B side by side
  (∀ z, C z z → 4 * z = 48) :=  -- The perimeter of C is 48
by sorry

end NUMINAMATH_CALUDE_square_perimeter_problem_l777_77756


namespace NUMINAMATH_CALUDE_cube_preserves_order_l777_77704

theorem cube_preserves_order (a b : ℝ) : a > b → a^3 > b^3 := by sorry

end NUMINAMATH_CALUDE_cube_preserves_order_l777_77704


namespace NUMINAMATH_CALUDE_tangent_slope_at_point_l777_77722

noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - 2

theorem tangent_slope_at_point :
  let x₀ : ℝ := 1
  let y₀ : ℝ := -5/3
  let slope : ℝ := deriv f x₀
  (f x₀ = y₀) ∧ (slope = 1) := by sorry

end NUMINAMATH_CALUDE_tangent_slope_at_point_l777_77722


namespace NUMINAMATH_CALUDE_pharmacist_weights_impossibility_l777_77760

theorem pharmacist_weights_impossibility :
  ¬∃ (w₁ w₂ w₃ : ℝ),
    w₁ < 90 ∧ w₂ < 90 ∧ w₃ < 90 ∧
    w₁ + w₂ + w₃ = 100 ∧
    w₁ + 2*w₂ + w₃ = 101 ∧
    w₁ + w₂ + 2*w₃ = 102 :=
sorry

end NUMINAMATH_CALUDE_pharmacist_weights_impossibility_l777_77760


namespace NUMINAMATH_CALUDE_a_bounds_l777_77725

def a : ℕ → ℚ
  | 0 => 1/2
  | n+1 => a n + (1 / (n+1)^2) * (a n)^2

theorem a_bounds (n : ℕ) : (n+1)/(n+2) < a n ∧ a n < n+1 := by
  sorry

end NUMINAMATH_CALUDE_a_bounds_l777_77725


namespace NUMINAMATH_CALUDE_larry_stickers_l777_77758

theorem larry_stickers (initial_stickers lost_stickers : ℕ) 
  (h1 : initial_stickers = 93)
  (h2 : lost_stickers = 6) :
  initial_stickers - lost_stickers = 87 := by
  sorry

end NUMINAMATH_CALUDE_larry_stickers_l777_77758


namespace NUMINAMATH_CALUDE_pentagram_star_angle_pentagram_star_angle_proof_l777_77706

/-- The angle at each point of a regular pentagram formed by extending the sides of a regular pentagon inscribed in a circle is 216°. -/
theorem pentagram_star_angle : ℝ :=
  let regular_pentagon_external_angle : ℝ := 360 / 5
  let star_point_angle : ℝ := 360 - 2 * regular_pentagon_external_angle
  216

/-- Proof of the pentagram star angle theorem. -/
theorem pentagram_star_angle_proof : pentagram_star_angle = 216 := by
  sorry

end NUMINAMATH_CALUDE_pentagram_star_angle_pentagram_star_angle_proof_l777_77706


namespace NUMINAMATH_CALUDE_locus_is_finite_l777_77770

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance squared between two points -/
def distanceSquared (p1 p2 : Point) : ℝ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

/-- Definition of the right triangle -/
def rightTriangle (c : ℝ) : Set Point :=
  {p : Point | 0 ≤ p.x ∧ p.x ≤ c ∧ 0 ≤ p.y ∧ p.y ≤ c ∧ p.x + p.y ≤ c}

/-- The set of points satisfying the given conditions -/
def locusSet (c : ℝ) : Set Point :=
  {p ∈ rightTriangle c |
    distanceSquared p ⟨0, 0⟩ + distanceSquared p ⟨c, 0⟩ = 2 * c^2 ∧
    distanceSquared p ⟨0, c⟩ = c^2}

theorem locus_is_finite (c : ℝ) (h : c > 0) : Set.Finite (locusSet c) :=
  sorry

end NUMINAMATH_CALUDE_locus_is_finite_l777_77770


namespace NUMINAMATH_CALUDE_pet_store_cages_used_l777_77788

def pet_store_problem (initial_puppies sold_puppies puppies_per_cage : ℕ) : ℕ :=
  (initial_puppies - sold_puppies) / puppies_per_cage

theorem pet_store_cages_used :
  pet_store_problem 78 30 8 = 6 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_cages_used_l777_77788


namespace NUMINAMATH_CALUDE_brick_length_proof_l777_77734

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular object given its dimensions -/
def volume (d : Dimensions) : ℝ :=
  d.length * d.width * d.height

theorem brick_length_proof (wall : Dimensions) (brick : Dimensions) (num_bricks : ℝ) :
  wall.length = 8 →
  wall.width = 6 →
  wall.height = 0.02 →
  brick.length = 0.11 →
  brick.width = 0.05 →
  brick.height = 0.06 →
  num_bricks = 2909.090909090909 →
  volume wall / volume brick = num_bricks →
  brick.length = 0.11 := by
  sorry

end NUMINAMATH_CALUDE_brick_length_proof_l777_77734


namespace NUMINAMATH_CALUDE_smallest_n_for_probability_threshold_l777_77723

theorem smallest_n_for_probability_threshold (n : ℕ) : 
  (∀ k, k < n → 1 / (k * (k + 1)) ≥ 1 / 2010) ∧
  1 / (n * (n + 1)) < 1 / 2010 →
  n = 45 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_probability_threshold_l777_77723


namespace NUMINAMATH_CALUDE_opposite_sides_line_range_l777_77773

theorem opposite_sides_line_range (a : ℝ) : 
  (0 + 0 < a ∧ a < 1 + 1) ∨ (0 + 0 > a ∧ a > 1 + 1) ↔ a < 0 ∨ a > 2 := by
sorry

end NUMINAMATH_CALUDE_opposite_sides_line_range_l777_77773


namespace NUMINAMATH_CALUDE_no_positive_sheep_solution_l777_77739

theorem no_positive_sheep_solution : ¬∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ y = 3 * x + 15 ∧ x = y - y / 3 := by
  sorry

end NUMINAMATH_CALUDE_no_positive_sheep_solution_l777_77739


namespace NUMINAMATH_CALUDE_proposition_b_l777_77705

theorem proposition_b (a : ℝ) : 0 < a → a < 1 → a^3 < a := by
  sorry

end NUMINAMATH_CALUDE_proposition_b_l777_77705


namespace NUMINAMATH_CALUDE_f_properties_l777_77799

/-- The function f(x) = mx^2 + (1-3m)x - 4 -/
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 + (1 - 3*m) * x - 4

theorem f_properties :
  -- Part I
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, f 1 x ≤ 4 ∧ f 1 x ≥ -5) ∧
  (∃ x₁ ∈ Set.Icc (-2 : ℝ) 2, f 1 x₁ = 4) ∧
  (∃ x₂ ∈ Set.Icc (-2 : ℝ) 2, f 1 x₂ = -5) ∧

  -- Part II (simplified representation of the solution sets)
  (∀ m : ℝ, ∃ S : Set ℝ, ∀ x : ℝ, f m x > -1 ↔ x ∈ S) ∧

  -- Part III
  (∀ m < 0, (∃ x₀ > 1, f m x₀ > 0) → m < -1 ∨ (-1/9 < m ∧ m < 0)) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l777_77799


namespace NUMINAMATH_CALUDE_problem_solution_l777_77755

theorem problem_solution (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_prod : x * y * z = 1)
  (h_eq1 : x + 1 / z = 6)
  (h_eq2 : y + 1 / x = 15) :
  z + 1 / y = 23 / 89 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l777_77755


namespace NUMINAMATH_CALUDE_number_of_mappings_l777_77735

/-- Given two finite sets A and B, where |A| = n and |B| = k, this function
    represents the number of order-preserving surjective mappings from A to B. -/
def orderPreservingSurjections (n k : ℕ) : ℕ :=
  Nat.choose (n - 1) (k - 1)

/-- The sets A and B -/
def A : Set ℝ := {a | ∃ i : Fin 60, a = i}
def B : Set ℝ := {b | ∃ i : Fin 25, b = i}

/-- The mapping f from A to B -/
def f : A → B := sorry

/-- f is surjective -/
axiom f_surjective : Function.Surjective f

/-- f preserves order -/
axiom f_order_preserving :
  ∀ (a₁ a₂ : A), (a₁ : ℝ) ≤ (a₂ : ℝ) → (f a₁ : ℝ) ≥ (f a₂ : ℝ)

/-- The main theorem: The number of such mappings is C₅₉²⁴ -/
theorem number_of_mappings :
  orderPreservingSurjections 60 25 = Nat.choose 59 24 := by sorry

end NUMINAMATH_CALUDE_number_of_mappings_l777_77735


namespace NUMINAMATH_CALUDE_otimes_neg_two_three_otimes_commutative_four_neg_two_l777_77710

-- Define the ⊗ operation for rational numbers
def otimes (a b : ℚ) : ℚ := a * b - a - b - 2

-- Theorem 1: (-2) ⊗ 3 = -9
theorem otimes_neg_two_three : otimes (-2) 3 = -9 := by sorry

-- Theorem 2: 4 ⊗ (-2) = (-2) ⊗ 4
theorem otimes_commutative_four_neg_two : otimes 4 (-2) = otimes (-2) 4 := by sorry

end NUMINAMATH_CALUDE_otimes_neg_two_three_otimes_commutative_four_neg_two_l777_77710


namespace NUMINAMATH_CALUDE_path_count_theorem_l777_77712

/-- The number of paths on a grid from point C to point D, where D is 6 units right and 2 units up from C, and the path consists of exactly 8 steps. -/
def number_of_paths : ℕ := 28

/-- The horizontal distance between points C and D on the grid. -/
def horizontal_distance : ℕ := 6

/-- The vertical distance between points C and D on the grid. -/
def vertical_distance : ℕ := 2

/-- The total number of steps in the path. -/
def total_steps : ℕ := 8

theorem path_count_theorem :
  number_of_paths = Nat.choose total_steps vertical_distance :=
by sorry

end NUMINAMATH_CALUDE_path_count_theorem_l777_77712


namespace NUMINAMATH_CALUDE_complex_multiplication_l777_77729

def A : ℂ := 6 - 2 * Complex.I
def M : ℂ := -3 + 4 * Complex.I
def S : ℂ := 2 * Complex.I
def P : ℂ := 3
def C : ℂ := 1 + Complex.I

theorem complex_multiplication :
  (A - M + S - P) * C = 10 + 2 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_multiplication_l777_77729


namespace NUMINAMATH_CALUDE_sandy_marks_calculation_l777_77766

theorem sandy_marks_calculation :
  ∀ (total_attempts : ℕ) (correct_attempts : ℕ) (marks_per_correct : ℕ) (marks_per_incorrect : ℕ),
    total_attempts = 30 →
    correct_attempts = 24 →
    marks_per_correct = 3 →
    marks_per_incorrect = 2 →
    (correct_attempts * marks_per_correct) - ((total_attempts - correct_attempts) * marks_per_incorrect) = 60 :=
by sorry

end NUMINAMATH_CALUDE_sandy_marks_calculation_l777_77766


namespace NUMINAMATH_CALUDE_min_illuminated_points_l777_77794

/-- Represents the number of illuminated points for a laser at angle θ --/
def illuminatedPoints (θ : ℕ) : ℕ := 180 / Nat.gcd 180 θ

/-- The problem statement --/
theorem min_illuminated_points :
  ∃ (n : ℕ), n < 90 ∧ 
  (∀ (m : ℕ), m < 90 → 
    illuminatedPoints n + illuminatedPoints (n + 1) - 1 ≤ 
    illuminatedPoints m + illuminatedPoints (m + 1) - 1) ∧
  illuminatedPoints n + illuminatedPoints (n + 1) - 1 = 28 :=
sorry

end NUMINAMATH_CALUDE_min_illuminated_points_l777_77794


namespace NUMINAMATH_CALUDE_specific_right_triangle_l777_77715

/-- A right triangle with inscribed and circumscribed circles -/
structure RightTriangle where
  /-- The radius of the inscribed circle -/
  inradius : ℝ
  /-- The radius of the circumscribed circle -/
  circumradius : ℝ
  /-- The shortest side of the triangle -/
  a : ℝ
  /-- The middle-length side of the triangle -/
  b : ℝ
  /-- The hypotenuse of the triangle -/
  c : ℝ
  /-- The triangle is right-angled -/
  right_angle : a^2 + b^2 = c^2
  /-- The inradius is correct for this triangle -/
  inradius_correct : inradius = (a + b - c) / 2
  /-- The circumradius is correct for this triangle -/
  circumradius_correct : circumradius = c / 2

/-- The main theorem about the specific right triangle -/
theorem specific_right_triangle :
  ∃ (t : RightTriangle), t.inradius = 8 ∧ t.circumradius = 41 ∧ t.a = 18 ∧ t.b = 80 ∧ t.c = 82 := by
  sorry

end NUMINAMATH_CALUDE_specific_right_triangle_l777_77715


namespace NUMINAMATH_CALUDE_quarter_point_quadrilateral_area_is_3_plus_2root2_l777_77718

/-- Regular octagon with apothem 2 -/
structure RegularOctagon :=
  (apothem : ℝ)
  (is_regular : apothem = 2)

/-- Quarter point on a side of the octagon -/
def quarter_point (O : RegularOctagon) (i : Fin 8) : ℝ × ℝ := sorry

/-- The area of the quadrilateral formed by connecting quarter points -/
def quarter_point_quadrilateral_area (O : RegularOctagon) : ℝ :=
  let Q1 := quarter_point O 0
  let Q3 := quarter_point O 2
  let Q5 := quarter_point O 4
  let Q7 := quarter_point O 6
  sorry -- Area calculation

/-- Theorem: The area of the quadrilateral formed by connecting
    the quarter points of every other side of a regular octagon
    with apothem 2 is 3 + 2√2 -/
theorem quarter_point_quadrilateral_area_is_3_plus_2root2 (O : RegularOctagon) :
  quarter_point_quadrilateral_area O = 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_quarter_point_quadrilateral_area_is_3_plus_2root2_l777_77718


namespace NUMINAMATH_CALUDE_teresa_social_studies_score_l777_77720

/-- Teresa's exam scores -/
structure ExamScores where
  science : ℕ
  music : ℕ
  physics : ℕ
  social_studies : ℕ
  total : ℕ

/-- Theorem: Given Teresa's exam scores satisfying certain conditions, her social studies score is 85 -/
theorem teresa_social_studies_score (scores : ExamScores) 
  (h1 : scores.science = 70)
  (h2 : scores.music = 80)
  (h3 : scores.physics = scores.music / 2)
  (h4 : scores.total = 275)
  (h5 : scores.total = scores.science + scores.music + scores.physics + scores.social_studies) :
  scores.social_studies = 85 := by
    sorry

#check teresa_social_studies_score

end NUMINAMATH_CALUDE_teresa_social_studies_score_l777_77720


namespace NUMINAMATH_CALUDE_scout_troop_girls_l777_77742

theorem scout_troop_girls (initial_total : ℕ) : 
  let initial_girls : ℕ := (6 * initial_total) / 10
  let final_total : ℕ := initial_total
  let final_girls : ℕ := initial_girls - 4
  (initial_girls : ℚ) / initial_total = 6 / 10 →
  (final_girls : ℚ) / final_total = 1 / 2 →
  initial_girls = 24 := by
sorry

end NUMINAMATH_CALUDE_scout_troop_girls_l777_77742


namespace NUMINAMATH_CALUDE_february_roses_l777_77740

def rose_sequence (october november december january : ℕ) : Prop :=
  november - october = december - november ∧
  december - november = january - december ∧
  november > october ∧ december > november ∧ january > december

theorem february_roses 
  (october november december january : ℕ) 
  (h : rose_sequence october november december january) 
  (oct_val : october = 108) 
  (nov_val : november = 120) 
  (dec_val : december = 132) 
  (jan_val : january = 144) : 
  january + (january - december) = 156 := by
sorry

end NUMINAMATH_CALUDE_february_roses_l777_77740


namespace NUMINAMATH_CALUDE_middle_group_frequency_l777_77719

theorem middle_group_frequency 
  (n : ℕ) 
  (total_area : ℝ) 
  (middle_area : ℝ) 
  (sample_size : ℕ) 
  (h1 : n > 0) 
  (h2 : middle_area = (1 / 5) * (total_area - middle_area)) 
  (h3 : sample_size = 300) : 
  (middle_area / total_area) * sample_size = 50 := by
sorry

end NUMINAMATH_CALUDE_middle_group_frequency_l777_77719


namespace NUMINAMATH_CALUDE_exponent_relationship_l777_77746

theorem exponent_relationship (x y z a b : ℝ) 
  (h1 : 4^x = a) 
  (h2 : 2^y = b) 
  (h3 : 8^z = a * b) : 
  3 * z = 2 * x + y := by
sorry

end NUMINAMATH_CALUDE_exponent_relationship_l777_77746


namespace NUMINAMATH_CALUDE_total_shoes_l777_77771

/-- The number of shoes owned by each person -/
structure ShoeCount where
  daniel : ℕ
  christopher : ℕ
  brian : ℕ
  edward : ℕ
  jacob : ℕ

/-- The conditions of the shoe ownership problem -/
def shoe_conditions (s : ShoeCount) : Prop :=
  s.daniel = 15 ∧
  s.christopher = 37 ∧
  s.brian = s.christopher + 5 ∧
  s.edward = (7 * s.brian) / 2 ∧
  s.jacob = (2 * s.edward) / 3

/-- The theorem stating the total number of shoes -/
theorem total_shoes (s : ShoeCount) (h : shoe_conditions s) :
  s.daniel + s.christopher + s.brian + s.edward + s.jacob = 339 := by
  sorry

end NUMINAMATH_CALUDE_total_shoes_l777_77771


namespace NUMINAMATH_CALUDE_largest_multiple_of_nine_less_than_hundred_l777_77703

theorem largest_multiple_of_nine_less_than_hundred : 
  ∀ n : ℕ, n % 9 = 0 ∧ n < 100 → n ≤ 99 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_nine_less_than_hundred_l777_77703


namespace NUMINAMATH_CALUDE_pascal_triangle_61_row_third_number_l777_77744

theorem pascal_triangle_61_row_third_number : 
  let n : ℕ := 60  -- The row number (61 numbers means it's the 60th row, 0-indexed)
  let k : ℕ := 2   -- The position of the number we're interested in (3rd number, 0-indexed)
  Nat.choose n k = 1770 := by
sorry

end NUMINAMATH_CALUDE_pascal_triangle_61_row_third_number_l777_77744


namespace NUMINAMATH_CALUDE_min_value_zero_iff_c_eq_four_l777_77777

/-- The quadratic expression in x and y with parameter c -/
def f (c x y : ℝ) : ℝ :=
  5 * x^2 - 8 * c * x * y + (4 * c^2 + 3) * y^2 - 5 * x - 5 * y + 7

/-- The theorem stating that c = 4 is the unique value for which the minimum of f is 0 -/
theorem min_value_zero_iff_c_eq_four :
  (∃ (c : ℝ), ∀ (x y : ℝ), f c x y ≥ 0 ∧ (∃ (x₀ y₀ : ℝ), f c x₀ y₀ = 0)) ↔ c = 4 :=
sorry

end NUMINAMATH_CALUDE_min_value_zero_iff_c_eq_four_l777_77777


namespace NUMINAMATH_CALUDE_probability_at_least_half_even_dice_l777_77759

theorem probability_at_least_half_even_dice (dice : Nat) (p_even : ℝ) :
  dice = 4 →
  p_even = 1/2 →
  let p_two_even := Nat.choose dice 2 * p_even^2 * (1 - p_even)^2
  let p_three_even := Nat.choose dice 3 * p_even^3 * (1 - p_even)
  let p_four_even := p_even^4
  p_two_even + p_three_even + p_four_even = 11/16 := by
sorry

end NUMINAMATH_CALUDE_probability_at_least_half_even_dice_l777_77759


namespace NUMINAMATH_CALUDE_fractional_exponent_simplification_l777_77731

theorem fractional_exponent_simplification (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a ^ (2 * b ^ (1/4))) / (a ^ (1/2) * b ^ (1/4)) = a ^ (3/2) := by
  sorry

end NUMINAMATH_CALUDE_fractional_exponent_simplification_l777_77731


namespace NUMINAMATH_CALUDE_derivative_f_l777_77727

noncomputable def f (x : ℝ) : ℝ := (1 / (4 * Real.sqrt 5)) * Real.log ((2 + Real.sqrt 5 * Real.tanh x) / (2 - Real.sqrt 5 * Real.tanh x))

theorem derivative_f (x : ℝ) : 
  deriv f x = 1 / (4 - Real.sinh x ^ 2) :=
sorry

end NUMINAMATH_CALUDE_derivative_f_l777_77727


namespace NUMINAMATH_CALUDE_x_range_for_decreasing_sequence_l777_77769

def decreasing_sequence (x : ℝ) : ℕ → ℝ
  | 0 => 1 - x
  | n + 1 => (1 - x) ^ (n + 2)

theorem x_range_for_decreasing_sequence (x : ℝ) :
  (∀ n : ℕ, decreasing_sequence x n > decreasing_sequence x (n + 1)) ↔ 0 < x ∧ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_x_range_for_decreasing_sequence_l777_77769


namespace NUMINAMATH_CALUDE_second_polygon_sides_l777_77714

/-- A regular polygon -/
structure RegularPolygon where
  sides : ℕ
  sideLength : ℝ

/-- The perimeter of a regular polygon -/
def perimeter (p : RegularPolygon) : ℝ := p.sides * p.sideLength

theorem second_polygon_sides 
  (p1 p2 : RegularPolygon) 
  (h1 : p1.sides = 42)
  (h2 : p1.sideLength = 3 * p2.sideLength)
  (h3 : perimeter p1 = perimeter p2) :
  p2.sides = 126 := by
sorry

end NUMINAMATH_CALUDE_second_polygon_sides_l777_77714


namespace NUMINAMATH_CALUDE_equation_solution_l777_77790

theorem equation_solution : ∃ (x : ℚ), (3/4 : ℚ) + 1/x = 7/8 ∧ x = 8 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l777_77790


namespace NUMINAMATH_CALUDE_seventh_term_is_four_l777_77732

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 2) / a (n + 1) = a (n + 1) / a n
  a₁_eq_one : a 1 = 1
  a₃_a₅_eq_4a₄_minus_4 : a 3 * a 5 = 4 * (a 4 - 1)

/-- The 7th term of the geometric sequence is 4 -/
theorem seventh_term_is_four (seq : GeometricSequence) : seq.a 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_seventh_term_is_four_l777_77732


namespace NUMINAMATH_CALUDE_total_savings_theorem_l777_77765

def weekday_savings : ℝ := 24
def weekend_savings : ℝ := 30
def monthly_subscription : ℝ := 45
def annual_interest_rate : ℝ := 0.03
def weeks_in_year : ℕ := 52
def days_in_year : ℕ := 365

def total_savings : ℝ :=
  let weekday_count : ℕ := days_in_year - 2 * weeks_in_year
  let weekend_count : ℕ := 2 * weeks_in_year
  let total_savings_before_interest : ℝ :=
    weekday_count * weekday_savings + weekend_count * weekend_savings - 12 * monthly_subscription
  total_savings_before_interest * (1 + annual_interest_rate)

theorem total_savings_theorem :
  total_savings = 9109.32 := by
  sorry

end NUMINAMATH_CALUDE_total_savings_theorem_l777_77765

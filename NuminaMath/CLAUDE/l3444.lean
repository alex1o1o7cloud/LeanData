import Mathlib

namespace second_number_value_l3444_344442

theorem second_number_value (A B C D : ℝ) : 
  C = 4.5 * B →
  B = 2.5 * A →
  D = 0.5 * (A + B) →
  (A + B + C + D) / 4 = 165 →
  B = 100 := by
sorry

end second_number_value_l3444_344442


namespace stair_cleaning_problem_l3444_344496

theorem stair_cleaning_problem (a b c : ℕ) (h1 : a > c) (h2 : 101 * (a + c) + 20 * b = 746) :
  let n := 100 * a + 10 * b + c
  (2 * n = 944) ∨ (2 * n = 1142) := by
  sorry

end stair_cleaning_problem_l3444_344496


namespace john_ben_difference_l3444_344476

/-- Represents the marble transfer problem --/
structure MarbleTransfer where
  ben_initial : ℝ
  john_initial : ℝ
  lisa_initial : ℝ
  max_initial : ℝ
  ben_to_john_percent : ℝ
  ben_to_lisa_percent : ℝ
  john_to_max_percent : ℝ
  lisa_to_john_percent : ℝ

/-- Calculates the final marble counts after all transfers --/
def finalCounts (mt : MarbleTransfer) : ℝ × ℝ × ℝ × ℝ :=
  let ben_to_john := mt.ben_initial * mt.ben_to_john_percent
  let ben_to_lisa := mt.ben_initial * mt.ben_to_lisa_percent
  let ben_final := mt.ben_initial - ben_to_john - ben_to_lisa
  let john_from_ben := ben_to_john
  let john_to_max := john_from_ben * mt.john_to_max_percent
  let lisa_with_ben := mt.lisa_initial + ben_to_lisa
  let lisa_to_john := mt.lisa_initial * mt.lisa_to_john_percent + ben_to_lisa
  let john_final := mt.john_initial + john_from_ben - john_to_max + lisa_to_john
  let max_final := mt.max_initial + john_to_max
  let lisa_final := lisa_with_ben - lisa_to_john
  (ben_final, john_final, lisa_final, max_final)

/-- Theorem stating the difference in marbles between John and Ben after transfers --/
theorem john_ben_difference (mt : MarbleTransfer) 
  (h1 : mt.ben_initial = 18)
  (h2 : mt.john_initial = 17)
  (h3 : mt.lisa_initial = 12)
  (h4 : mt.max_initial = 9)
  (h5 : mt.ben_to_john_percent = 0.5)
  (h6 : mt.ben_to_lisa_percent = 0.25)
  (h7 : mt.john_to_max_percent = 0.65)
  (h8 : mt.lisa_to_john_percent = 0.2) :
  (finalCounts mt).2.1 - (finalCounts mt).1 = 22.5 := by
  sorry

end john_ben_difference_l3444_344476


namespace a_33_mod_33_l3444_344487

/-- The integer obtained by writing all integers from 1 to n sequentially -/
def a (n : ℕ) : ℕ :=
  sorry

/-- The theorem stating that a₃₃ mod 33 = 22 -/
theorem a_33_mod_33 : a 33 % 33 = 22 := by
  sorry

end a_33_mod_33_l3444_344487


namespace vector_sum_in_triangle_l3444_344470

-- Define the triangle ABC and points E and F
variable (A B C E F : ℝ × ℝ)

-- Define vectors
def AB : ℝ × ℝ := B - A
def AC : ℝ × ℝ := C - A
def AE : ℝ × ℝ := E - A
def CF : ℝ × ℝ := F - C
def FA : ℝ × ℝ := A - F
def EF : ℝ × ℝ := F - E

-- Define conditions
variable (h1 : AE = (1/2 : ℝ) • AB)
variable (h2 : CF = (2 : ℝ) • FA)
variable (x y : ℝ)
variable (h3 : EF = x • AB + y • AC)

-- Theorem statement
theorem vector_sum_in_triangle : x + y = -1/6 := by
  sorry

end vector_sum_in_triangle_l3444_344470


namespace abs_diff_sqrt_two_l3444_344474

theorem abs_diff_sqrt_two : ∀ x : ℝ, 1 < x ∧ x < 2 ∧ x^2 = 2 → |3 - x| - |x - 2| = 1 := by
  sorry

end abs_diff_sqrt_two_l3444_344474


namespace cubic_root_product_l3444_344414

theorem cubic_root_product (a b c : ℝ) : 
  (a^3 - 15*a^2 + 22*a - 8 = 0) ∧ 
  (b^3 - 15*b^2 + 22*b - 8 = 0) ∧ 
  (c^3 - 15*c^2 + 22*c - 8 = 0) → 
  (2+a)*(2+b)*(2+c) = 120 := by
sorry

end cubic_root_product_l3444_344414


namespace arithmetic_sequence_sum_l3444_344413

/-- Given an arithmetic sequence 3, 7, 11, ..., x, y, 31, prove that x + y = 50 -/
theorem arithmetic_sequence_sum (x y : ℝ) : 
  (∃ (a : ℕ → ℝ), a 0 = 3 ∧ a 1 = 7 ∧ a 2 = 11 ∧ (∃ i j : ℕ, a i = x ∧ a (i + 1) = y ∧ a (j + 2) = 31) ∧ 
  (∀ n : ℕ, a (n + 1) - a n = a 1 - a 0)) → 
  x + y = 50 := by
sorry

end arithmetic_sequence_sum_l3444_344413


namespace largest_four_digit_divisible_by_88_l3444_344437

theorem largest_four_digit_divisible_by_88 : 
  ∀ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 88 = 0 → n ≤ 9944 :=
by sorry

end largest_four_digit_divisible_by_88_l3444_344437


namespace incorrect_negation_even_multiple_of_seven_l3444_344497

theorem incorrect_negation_even_multiple_of_seven :
  ¬(∀ n : ℕ, ¬(2 * n % 7 = 0)) ↔ ∃ n : ℕ, 2 * n % 7 = 0 :=
by sorry

end incorrect_negation_even_multiple_of_seven_l3444_344497


namespace largest_four_digit_divisible_by_9_and_3_digit_by_4_l3444_344463

/-- Represents a four-digit number -/
structure FourDigitNumber where
  value : Nat
  is_four_digit : 1000 ≤ value ∧ value ≤ 9999

/-- Returns the three-digit number obtained by removing the last digit -/
def remove_last_digit (n : FourDigitNumber) : Nat :=
  n.value / 10

/-- Returns the last digit of a number -/
def last_digit (n : FourDigitNumber) : Nat :=
  n.value % 10

theorem largest_four_digit_divisible_by_9_and_3_digit_by_4 (n : FourDigitNumber) 
  (h1 : n.value % 9 = 0)
  (h2 : remove_last_digit n % 4 = 0)
  (h3 : ∀ m : FourDigitNumber, m.value % 9 = 0 → remove_last_digit m % 4 = 0 → m.value ≤ n.value) :
  last_digit n = 3 := by
  sorry

end largest_four_digit_divisible_by_9_and_3_digit_by_4_l3444_344463


namespace least_x_value_l3444_344431

theorem least_x_value (x p : ℕ) (h1 : x > 0) (h2 : Nat.Prime p) 
  (h3 : ∃ q : ℕ, Nat.Prime q ∧ q % 2 = 1 ∧ x = 12 * p * q) : 
  x ≥ 72 ∧ (∃ x₀ : ℕ, x₀ ≥ 72 → 
    (∃ p₀ q₀ : ℕ, Nat.Prime p₀ ∧ Nat.Prime q₀ ∧ q₀ % 2 = 1 ∧ x₀ = 12 * p₀ * q₀) → x₀ ≥ x) :=
by sorry

end least_x_value_l3444_344431


namespace billy_crayons_l3444_344446

theorem billy_crayons (initial_crayons eaten_crayons remaining_crayons : ℕ) :
  eaten_crayons = 52 →
  remaining_crayons = 10 →
  initial_crayons = eaten_crayons + remaining_crayons →
  initial_crayons = 62 := by
  sorry

end billy_crayons_l3444_344446


namespace investment_interest_rate_l3444_344448

/-- Proves that given the specified investment conditions, the unknown interest rate is 8% --/
theorem investment_interest_rate : 
  ∀ (x y r : ℚ),
  x + y = 2000 →
  y = 650 →
  x * (1/10) - y * r = 83 →
  r = 8/100 := by
  sorry

end investment_interest_rate_l3444_344448


namespace olivias_groceries_cost_l3444_344439

/-- The total cost of Olivia's groceries is $42 -/
theorem olivias_groceries_cost (banana_cost bread_cost milk_cost apple_cost : ℕ)
  (h1 : banana_cost = 12)
  (h2 : bread_cost = 9)
  (h3 : milk_cost = 7)
  (h4 : apple_cost = 14) :
  banana_cost + bread_cost + milk_cost + apple_cost = 42 := by
  sorry

end olivias_groceries_cost_l3444_344439


namespace geometric_progression_problem_l3444_344471

theorem geometric_progression_problem (b₂ b₆ : ℚ) 
  (h₂ : b₂ = 37 + 1/3) 
  (h₆ : b₆ = 2 + 1/3) : 
  ∃ (a q : ℚ), a * q = b₂ ∧ a * q^5 = b₆ ∧ a = 224/3 ∧ q = 1/2 := by
  sorry

end geometric_progression_problem_l3444_344471


namespace sally_weekend_pages_l3444_344444

/-- The number of pages Sally reads on weekdays -/
def weekday_pages : ℕ := 10

/-- The number of weeks it takes Sally to finish the book -/
def weeks_to_finish : ℕ := 2

/-- The total number of pages in the book -/
def total_pages : ℕ := 180

/-- The number of weekdays in a week -/
def weekdays_per_week : ℕ := 5

/-- The number of weekend days in a week -/
def weekend_days_per_week : ℕ := 2

/-- Theorem: Sally reads 20 pages on each weekend day -/
theorem sally_weekend_pages : 
  (total_pages - weekday_pages * weekdays_per_week * weeks_to_finish) / (weekend_days_per_week * weeks_to_finish) = 20 := by
  sorry

end sally_weekend_pages_l3444_344444


namespace gcd_of_225_and_135_l3444_344485

theorem gcd_of_225_and_135 : Nat.gcd 225 135 = 45 := by
  sorry

end gcd_of_225_and_135_l3444_344485


namespace coke_to_sprite_ratio_l3444_344418

/-- Represents the ratio of ingredients in a drink -/
structure DrinkRatio where
  coke : ℚ
  sprite : ℚ
  mountainDew : ℚ

/-- Represents the composition of a drink -/
structure Drink where
  ratio : DrinkRatio
  cokeAmount : ℚ
  totalAmount : ℚ

/-- Theorem: Given a drink with the specified ratio and amounts, prove that the ratio of Coke to Sprite is 2:1 -/
theorem coke_to_sprite_ratio 
  (drink : Drink) 
  (h1 : drink.ratio.sprite = 1)
  (h2 : drink.ratio.mountainDew = 3)
  (h3 : drink.cokeAmount = 6)
  (h4 : drink.totalAmount = 18) :
  drink.ratio.coke / drink.ratio.sprite = 2 := by
  sorry


end coke_to_sprite_ratio_l3444_344418


namespace inequality_proof_l3444_344438

theorem inequality_proof (A B C ε : Real) 
  (hA : 0 ≤ A ∧ A ≤ π) 
  (hB : 0 ≤ B ∧ B ≤ π) 
  (hC : 0 ≤ C ∧ C ≤ π) 
  (hε : ε ≥ 1) : 
  ε * (Real.sin A + Real.sin B + Real.sin C) ≤ Real.sin A * Real.sin B * Real.sin C + 1 + ε^3 ∧ 
  (1 + ε + Real.sin A) * (1 + ε + Real.sin B) * (1 + ε + Real.sin C) ≥ 9 * ε * (Real.sin A + Real.sin B + Real.sin C) := by
  sorry

end inequality_proof_l3444_344438


namespace total_votes_is_330_l3444_344449

/-- Proves that the total number of votes is 330 given the specified conditions -/
theorem total_votes_is_330 :
  ∀ (total_votes votes_for votes_against : ℕ),
    votes_for = votes_against + 66 →
    votes_against = (40 * total_votes) / 100 →
    total_votes = votes_for + votes_against →
    total_votes = 330 := by
  sorry

end total_votes_is_330_l3444_344449


namespace smaller_prime_factor_l3444_344426

theorem smaller_prime_factor : ∃ p : ℕ, 
  Prime p ∧ 
  p > 4002001 ∧ 
  316990099009901 = 4002001 * p ∧
  316990099009901 = 32016000000000001 / 101 := by
sorry

end smaller_prime_factor_l3444_344426


namespace equal_number_of_boys_and_girls_l3444_344411

theorem equal_number_of_boys_and_girls 
  (m : ℕ) (d : ℕ) (M : ℝ) (D : ℝ) 
  (h1 : M / m ≠ D / d) 
  (h2 : (M / m + D / d) / 2 = (M + D) / (m + d)) : 
  m = d := by sorry

end equal_number_of_boys_and_girls_l3444_344411


namespace clothing_percentage_is_fifty_percent_l3444_344429

/-- Represents the shopping breakdown and tax rates for Jill's purchases --/
structure ShoppingBreakdown where
  clothing_percentage : ℝ
  food_percentage : ℝ
  other_percentage : ℝ
  clothing_tax_rate : ℝ
  food_tax_rate : ℝ
  other_tax_rate : ℝ
  total_tax_rate : ℝ

/-- Calculates the percentage spent on clothing given the shopping breakdown --/
def calculate_clothing_percentage (sb : ShoppingBreakdown) : ℝ :=
  sb.clothing_percentage

/-- Theorem stating that the percentage spent on clothing is 50% --/
theorem clothing_percentage_is_fifty_percent (sb : ShoppingBreakdown) 
  (h1 : sb.food_percentage = 0.25)
  (h2 : sb.other_percentage = 0.25)
  (h3 : sb.clothing_tax_rate = 0.10)
  (h4 : sb.food_tax_rate = 0)
  (h5 : sb.other_tax_rate = 0.20)
  (h6 : sb.total_tax_rate = 0.10)
  (h7 : sb.clothing_percentage + sb.food_percentage + sb.other_percentage = 1) :
  calculate_clothing_percentage sb = 0.5 := by
  sorry

#eval calculate_clothing_percentage { 
  clothing_percentage := 0.5,
  food_percentage := 0.25,
  other_percentage := 0.25,
  clothing_tax_rate := 0.10,
  food_tax_rate := 0,
  other_tax_rate := 0.20,
  total_tax_rate := 0.10
}

end clothing_percentage_is_fifty_percent_l3444_344429


namespace factors_of_1320_eq_24_l3444_344492

/-- The number of distinct positive factors of 1320 -/
def factors_of_1320 : ℕ :=
  (3 : ℕ) * 2 * 2 * 2

/-- Theorem stating that the number of distinct positive factors of 1320 is 24 -/
theorem factors_of_1320_eq_24 : factors_of_1320 = 24 := by
  sorry

end factors_of_1320_eq_24_l3444_344492


namespace painted_cube_equality_l3444_344425

theorem painted_cube_equality (n : ℝ) (h : n > 2) :
  12 * (n - 2) = (n - 2)^3 ↔ n = 2 + 2 * Real.sqrt 3 := by sorry

end painted_cube_equality_l3444_344425


namespace floor_sqrt_72_l3444_344409

theorem floor_sqrt_72 : ⌊Real.sqrt 72⌋ = 8 := by
  sorry

end floor_sqrt_72_l3444_344409


namespace walk_legs_and_wheels_l3444_344479

/-- Calculates the total number of legs and wheels for a group of organisms and a wheelchair -/
def total_legs_and_wheels (humans : ℕ) (dogs : ℕ) (cats : ℕ) (horses : ℕ) (monkeys : ℕ) (wheelchair_wheels : ℕ) : ℕ :=
  humans * 2 + dogs * 4 + cats * 4 + horses * 4 + monkeys * 4 + wheelchair_wheels

/-- Proves that the total number of legs and wheels for the given group is 46 -/
theorem walk_legs_and_wheels :
  total_legs_and_wheels 9 3 1 1 1 4 = 46 := by
  sorry

end walk_legs_and_wheels_l3444_344479


namespace distribute_5_3_l3444_344486

/-- The number of ways to distribute n distinct objects into k distinct containers,
    with each container receiving at least one object. -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 5 distinct objects into 3 distinct containers,
    with each container receiving at least one object, is 150. -/
theorem distribute_5_3 : distribute 5 3 = 150 := by sorry

end distribute_5_3_l3444_344486


namespace dice_sum_product_l3444_344412

theorem dice_sum_product (a b c d : ℕ) : 
  1 ≤ a ∧ a ≤ 6 ∧
  1 ≤ b ∧ b ≤ 6 ∧
  1 ≤ c ∧ c ≤ 6 ∧
  1 ≤ d ∧ d ≤ 6 ∧
  a * b * c * d = 180 →
  a + b + c + d ≠ 19 := by
sorry

end dice_sum_product_l3444_344412


namespace min_value_a_l3444_344489

theorem min_value_a (a b : ℕ) (h : 1998 * a = b^4) : 1215672 ≤ a := by
  sorry

end min_value_a_l3444_344489


namespace handshakes_in_specific_tournament_l3444_344405

/-- Represents a tennis tournament with teams of women -/
structure WomensTennisTournament where
  total_teams : Nat
  women_per_team : Nat
  participating_teams : Nat

/-- Calculates the number of handshakes in the tournament -/
def calculate_handshakes (tournament : WomensTennisTournament) : Nat :=
  let total_women := tournament.participating_teams * tournament.women_per_team
  let handshakes_per_woman := (tournament.participating_teams - 1) * tournament.women_per_team
  (total_women * handshakes_per_woman) / 2

/-- Theorem stating the number of handshakes in the specific tournament scenario -/
theorem handshakes_in_specific_tournament :
  let tournament := WomensTennisTournament.mk 4 2 3
  calculate_handshakes tournament = 12 := by
  sorry

end handshakes_in_specific_tournament_l3444_344405


namespace tree_height_difference_l3444_344450

theorem tree_height_difference : 
  let maple_height : ℚ := 13 + 1/4
  let pine_height : ℚ := 19 + 3/8
  pine_height - maple_height = 6 + 1/8 := by
sorry

end tree_height_difference_l3444_344450


namespace joan_seashells_l3444_344408

theorem joan_seashells (sam_shells : ℕ) (total_shells : ℕ) (joan_shells : ℕ) 
  (h1 : sam_shells = 35)
  (h2 : total_shells = 53)
  (h3 : total_shells = sam_shells + joan_shells) :
  joan_shells = 18 := by
  sorry

end joan_seashells_l3444_344408


namespace inscribed_hemisphere_volume_l3444_344422

/-- Given a cone with height 4 cm and slant height 5 cm, the volume of an inscribed hemisphere
    whose base lies on the base of the cone is (1152/125)π cm³. -/
theorem inscribed_hemisphere_volume (h : ℝ) (l : ℝ) (r : ℝ) :
  h = 4 →
  l = 5 →
  l^2 = h^2 + r^2 →
  (∃ x, x > 0 ∧ x < h ∧ r^2 + (l - x)^2 = h^2 ∧ x^2 + r^2 = r^2) →
  (2/3) * π * ((12/5)^3) = (1152/125) * π :=
by sorry

end inscribed_hemisphere_volume_l3444_344422


namespace alternative_increase_is_nineteen_cents_l3444_344416

/-- Represents the fine structure for overdue books in a library --/
structure OverdueFine where
  initial_fine : ℚ
  standard_increase : ℚ
  alternative_increase : ℚ
  fifth_day_fine : ℚ

/-- Calculates the fine for a given number of days overdue --/
def calculate_fine (f : OverdueFine) (days : ℕ) : ℚ :=
  f.initial_fine + (days - 1) * min f.standard_increase f.alternative_increase

/-- Theorem stating that the alternative increase is $0.19 --/
theorem alternative_increase_is_nineteen_cents 
  (f : OverdueFine) 
  (h1 : f.initial_fine = 7/100)
  (h2 : f.standard_increase = 30/100)
  (h3 : f.fifth_day_fine = 86/100) : 
  f.alternative_increase = 19/100 := by
  sorry

#eval let f : OverdueFine := {
  initial_fine := 7/100,
  standard_increase := 30/100,
  alternative_increase := 19/100,
  fifth_day_fine := 86/100
}
calculate_fine f 5

end alternative_increase_is_nineteen_cents_l3444_344416


namespace min_expression_upper_bound_l3444_344451

theorem min_expression_upper_bound (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  min (min (min (1 / a) (2 / b)) (4 / c)) (Real.rpow (a * b * c) (1 / 3)) ≤ Real.sqrt 2 ∧
  ∃ (a b c : ℝ), 0 < a ∧ 0 < b ∧ 0 < c ∧
    min (min (min (1 / a) (2 / b)) (4 / c)) (Real.rpow (a * b * c) (1 / 3)) = Real.sqrt 2 :=
by sorry

end min_expression_upper_bound_l3444_344451


namespace distance_between_points_l3444_344462

theorem distance_between_points : 
  let A : ℝ × ℝ := (1, 3)
  let B : ℝ × ℝ := (-4, 7)
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = Real.sqrt 41 := by
  sorry

end distance_between_points_l3444_344462


namespace subset_implies_m_equals_three_l3444_344452

theorem subset_implies_m_equals_three (A B : Set ℕ) (m : ℕ) :
  A = {1, 3} →
  B = {1, 2, m} →
  A ⊆ B →
  m = 3 := by sorry

end subset_implies_m_equals_three_l3444_344452


namespace parabola_vertex_l3444_344433

/-- The vertex of the parabola y = 2x^2 + 16x + 50 is (-4, 18) -/
theorem parabola_vertex (x y : ℝ) : 
  y = 2 * x^2 + 16 * x + 50 → (∃ m n : ℝ, m = -4 ∧ n = 18 ∧ 
    ∀ x, y = 2 * (x - m)^2 + n) :=
by sorry

end parabola_vertex_l3444_344433


namespace trigonometric_equation_consequences_l3444_344456

open Real

theorem trigonometric_equation_consequences (α : ℝ) 
  (h : sin (π - α) * cos (2*π - α) / (tan (π - α) * sin (π/2 + α) * cos (π/2 - α)) = 1/2) : 
  (cos α - 2*sin α) / (3*cos α + sin α) = 5 ∧ 
  1 - 2*sin α*cos α + cos α^2 = 2/5 := by
sorry

end trigonometric_equation_consequences_l3444_344456


namespace vishal_investment_percentage_l3444_344490

/-- Represents the investment amounts in rupees -/
structure Investment where
  vishal : ℝ
  trishul : ℝ
  raghu : ℝ

/-- The given conditions of the investment problem -/
def investment_conditions (i : Investment) : Prop :=
  i.raghu = 2300 ∧
  i.trishul = 0.9 * i.raghu ∧
  i.vishal + i.trishul + i.raghu = 6647

/-- The theorem stating that Vishal invested 10% more than Trishul -/
theorem vishal_investment_percentage (i : Investment) 
  (h : investment_conditions i) : 
  (i.vishal - i.trishul) / i.trishul = 0.1 := by
  sorry

end vishal_investment_percentage_l3444_344490


namespace aruns_weight_estimation_l3444_344481

/-- Arun's weight estimation problem -/
theorem aruns_weight_estimation (W : ℝ) (L : ℝ) : 
  (L < W ∧ W < 72) →  -- Arun's estimation
  (60 < W ∧ W < 70) →  -- Brother's estimation
  (W ≤ 68) →  -- Mother's estimation
  (∃ (a b : ℝ), 60 < a ∧ a < b ∧ b ≤ 68 ∧ (a + b) / 2 = 67) →  -- Average condition
  L > 60 := by
sorry

end aruns_weight_estimation_l3444_344481


namespace sarahs_age_l3444_344420

theorem sarahs_age :
  ∀ (s : ℚ), -- Sarah's age
  (∃ (g : ℚ), -- Grandmother's age
    g = 10 * s ∧ -- Grandmother is ten times Sarah's age
    g - s = 60) -- Grandmother was 60 when Sarah was born
  → s = 20 / 3 := by
sorry

end sarahs_age_l3444_344420


namespace function_equality_condition_l3444_344410

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2 - 2*x
def g (a x : ℝ) : ℝ := a*x - 1

-- Define the domain interval
def I : Set ℝ := Set.Icc (-1) 2

-- Define the theorem
theorem function_equality_condition (a : ℝ) : 
  (∀ x₁ ∈ I, ∃ x₂ ∈ I, f x₁ = g a x₂) ↔ 
  (a ≤ -4 ∨ a ≥ 2) :=
sorry

end function_equality_condition_l3444_344410


namespace correct_number_of_arrangements_l3444_344453

/-- The number of arrangements for 3 boys and 3 girls in a line, where students of the same gender are adjacent -/
def number_of_arrangements : ℕ := 72

/-- The number of boys -/
def num_boys : ℕ := 3

/-- The number of girls -/
def num_girls : ℕ := 3

/-- The total number of students -/
def total_students : ℕ := num_boys + num_girls

/-- Theorem stating that the number of arrangements is correct -/
theorem correct_number_of_arrangements :
  number_of_arrangements = (Nat.factorial num_boys) * (Nat.factorial num_girls) * 2 :=
by sorry

end correct_number_of_arrangements_l3444_344453


namespace quadratic_minimum_value_l3444_344457

/-- A quadratic function that takes values 6, 5, and 5 for three consecutive natural numbers. -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ n : ℕ, f n = 6 ∧ f (n + 1) = 5 ∧ f (n + 2) = 5

/-- The theorem stating that the minimum value of the quadratic function is 39/8. -/
theorem quadratic_minimum_value (f : ℝ → ℝ) (h : QuadraticFunction f) :
  ∃ x : ℝ, f x = 39/8 ∧ ∀ y : ℝ, f y ≥ 39/8 :=
by
  sorry

end quadratic_minimum_value_l3444_344457


namespace sum_bounds_l3444_344406

theorem sum_bounds (a b c : ℝ) (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c) 
  (h_eq : a^2 + b^2 + c^2 + a*b + 2/3*a*c + 4/3*b*c = 1) : 
  1 ≤ a + b + c ∧ a + b + c ≤ Real.sqrt 345 / 15 := by
  sorry

end sum_bounds_l3444_344406


namespace gcd_of_256_180_600_l3444_344472

theorem gcd_of_256_180_600 : Nat.gcd 256 (Nat.gcd 180 600) = 12 := by
  sorry

end gcd_of_256_180_600_l3444_344472


namespace escape_theorem_l3444_344477

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circular pond -/
structure Pond where
  center : Point
  radius : ℝ

/-- Represents a person with swimming and running speeds -/
structure Person where
  position : Point
  swimSpeed : ℝ
  runSpeed : ℝ

/-- Checks if a person can escape from another in a circular pond -/
def canEscape (pond : Pond) (escaper : Person) (chaser : Person) : Prop :=
  ∃ (t : ℝ), t > 0 ∧
  ∃ (escapePoint : Point),
    (escapePoint.x - pond.center.x)^2 + (escapePoint.y - pond.center.y)^2 > pond.radius^2 ∧
    (escapePoint.x - escaper.position.x)^2 + (escapePoint.y - escaper.position.y)^2 ≤ (escaper.swimSpeed * t)^2 ∧
    (escapePoint.x - chaser.position.x)^2 + (escapePoint.y - chaser.position.y)^2 > (chaser.runSpeed * t)^2

theorem escape_theorem (pond : Pond) (x y : Person) :
  x.position = pond.center →
  (y.position.x - pond.center.x)^2 + (y.position.y - pond.center.y)^2 = pond.radius^2 →
  y.runSpeed = 4 * x.swimSpeed →
  x.runSpeed > 4 * x.swimSpeed →
  canEscape pond x y :=
sorry

end escape_theorem_l3444_344477


namespace wilsons_theorem_l3444_344441

theorem wilsons_theorem (p : ℕ) (h : p > 1) :
  Nat.Prime p ↔ (Nat.factorial (p - 1) % p = p - 1) := by
  sorry

end wilsons_theorem_l3444_344441


namespace weight_of_new_person_l3444_344427

theorem weight_of_new_person (initial_count : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 12 →
  weight_increase = 3.7 →
  replaced_weight = 57.3 →
  (initial_count : ℝ) * weight_increase + replaced_weight = 101.7 := by
  sorry

end weight_of_new_person_l3444_344427


namespace triangle_area_after_transformation_l3444_344445

def a : Fin 2 → ℝ := ![3, 2]
def b : Fin 2 → ℝ := ![-1, 5]
def T : Matrix (Fin 2) (Fin 2) ℝ := !![1, 1; 0, 1]

theorem triangle_area_after_transformation :
  let Ta := T.mulVec a
  let Tb := T.mulVec b
  (1/2) * abs (Ta 0 * Tb 1 - Ta 1 * Tb 0) = 8.5 := by sorry

end triangle_area_after_transformation_l3444_344445


namespace max_value_on_interval_l3444_344495

def f_properties (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ 
  (f 1 = 2) ∧
  (∀ x y, x > 0 → y > 0 → f x < f y) ∧
  (∀ x y, f (x + y) = f x + f y)

theorem max_value_on_interval 
  (f : ℝ → ℝ) 
  (h : f_properties f) :
  ∃ x ∈ Set.Icc (-3) (-2), ∀ y ∈ Set.Icc (-3) (-2), f y ≤ f x ∧ f x = -4 :=
sorry

end max_value_on_interval_l3444_344495


namespace absolute_difference_of_numbers_l3444_344460

theorem absolute_difference_of_numbers (x y : ℝ) 
  (sum_eq : x + y = 36) 
  (product_eq : x * y = 320) : 
  |x - y| = 4 := by sorry

end absolute_difference_of_numbers_l3444_344460


namespace bathtub_fill_time_l3444_344498

/-- Proves that a bathtub with given capacity filled by a tap with given flow rate takes the calculated time to fill -/
theorem bathtub_fill_time (bathtub_capacity : ℝ) (tap_volume : ℝ) (tap_time : ℝ) (fill_time : ℝ) 
    (h1 : bathtub_capacity = 140)
    (h2 : tap_volume = 15)
    (h3 : tap_time = 3)
    (h4 : fill_time = bathtub_capacity / (tap_volume / tap_time)) :
  fill_time = 28 := by
  sorry

end bathtub_fill_time_l3444_344498


namespace trig_fraction_value_l3444_344464

theorem trig_fraction_value (θ : Real) (h : Real.sin θ + 2 * Real.cos θ = 1) :
  (Real.sin θ - Real.cos θ) / (Real.sin θ + Real.cos θ) = -7 ∨
  (Real.sin θ - Real.cos θ) / (Real.sin θ + Real.cos θ) = 1 := by
  sorry

end trig_fraction_value_l3444_344464


namespace third_team_wins_l3444_344430

/-- Represents the amount of wood processed by a team of lumberjacks -/
structure WoodProcessed where
  amount : ℝ
  amount_pos : amount > 0

/-- The competition between three teams of lumberjacks -/
structure LumberjackCompetition where
  team1 : WoodProcessed
  team2 : WoodProcessed
  team3 : WoodProcessed
  first_third_twice_second : team1.amount + team3.amount = 2 * team2.amount
  second_third_thrice_first : team2.amount + team3.amount = 3 * team1.amount

/-- The third team processes the most wood in the competition -/
theorem third_team_wins (comp : LumberjackCompetition) : 
  comp.team3.amount > comp.team1.amount ∧ comp.team3.amount > comp.team2.amount := by
  sorry

#check third_team_wins

end third_team_wins_l3444_344430


namespace find_x_value_l3444_344466

theorem find_x_value (X : ℝ) (h : (X + 200 / 90) * 90 = 18200) : X = 18000 := by
  sorry

end find_x_value_l3444_344466


namespace pond_length_l3444_344434

theorem pond_length (field_length : ℝ) (field_width : ℝ) (pond_area : ℝ) : 
  field_length = 48 →
  field_width = field_length / 2 →
  pond_area = (field_length * field_width) / 18 →
  Real.sqrt pond_area = 8 := by
sorry

end pond_length_l3444_344434


namespace quadratic_equation_roots_l3444_344403

theorem quadratic_equation_roots (p q : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁^2 + p*x₁ + q = 0 ∧ x₂^2 + p*x₂ + q = 0 ∧ 
   x₁ - x₂ = 5 ∧ x₁^3 - x₂^3 = 35) →
  ((p = 1 ∧ q = -6) ∨ (p = -1 ∧ q = -6)) :=
by sorry

end quadratic_equation_roots_l3444_344403


namespace tangency_and_tangent_line_l3444_344469

-- Define the curves C₁ and C₂
def C₁ (x y : ℝ) : Prop := x = Real.sqrt (2 * y^2 + 25/2)
def C₂ (a x y : ℝ) : Prop := y = a * x^2

-- Define the tangency condition
def is_tangent (a : ℝ) : Prop :=
  ∃ x y : ℝ, C₁ x y ∧ C₂ a x y ∧
  (∀ x' y' : ℝ, C₁ x' y' → C₂ a x' y' → (x = x' ∧ y = y'))

-- State the theorem
theorem tangency_and_tangent_line :
  ∃ a : ℝ, a > 0 ∧ is_tangent a ∧
  (∀ x y : ℝ, C₁ x y ∧ C₂ a x y → x = 5 ∧ y = 5/2) ∧
  (∀ x y : ℝ, 2*x - 2*y - 5 = 0 ↔ (C₁ x y ∧ C₂ a x y ∨ (x = 5 ∧ y = 5/2))) :=
sorry

end tangency_and_tangent_line_l3444_344469


namespace zero_in_P_l3444_344455

def P : Set ℝ := {x | x > -1}

theorem zero_in_P : (0 : ℝ) ∈ P := by sorry

end zero_in_P_l3444_344455


namespace fence_cost_l3444_344447

/-- The cost of building a fence around a square plot -/
theorem fence_cost (area : ℝ) (price_per_foot : ℝ) (h1 : area = 289) (h2 : price_per_foot = 58) :
  let side_length := Real.sqrt area
  let perimeter := 4 * side_length
  let total_cost := perimeter * price_per_foot
  total_cost = 3944 := by sorry

end fence_cost_l3444_344447


namespace rectangle_rotation_path_length_l3444_344484

/-- The length of the path traveled by point P of a rectangle PQRS after two 90° rotations -/
theorem rectangle_rotation_path_length (P Q R S : ℝ × ℝ) : 
  let pq : ℝ := 2
  let rs : ℝ := 2
  let qr : ℝ := 6
  let sp : ℝ := 6
  let first_rotation_radius : ℝ := Real.sqrt (pq^2 + qr^2)
  let first_rotation_arc_length : ℝ := (π / 2) * first_rotation_radius
  let second_rotation_radius : ℝ := sp
  let second_rotation_arc_length : ℝ := (π / 2) * second_rotation_radius
  let total_path_length : ℝ := first_rotation_arc_length + second_rotation_arc_length
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = pq^2 →
  (R.1 - S.1)^2 + (R.2 - S.2)^2 = rs^2 →
  (Q.1 - R.1)^2 + (Q.2 - R.2)^2 = qr^2 →
  (S.1 - P.1)^2 + (S.2 - P.2)^2 = sp^2 →
  total_path_length = (3 + Real.sqrt 10) * π := by
sorry

end rectangle_rotation_path_length_l3444_344484


namespace lemon_pie_degrees_l3444_344494

theorem lemon_pie_degrees (total_students : ℕ) (chocolate : ℕ) (apple : ℕ) (blueberry : ℕ)
  (h1 : total_students = 40)
  (h2 : chocolate = 15)
  (h3 : apple = 9)
  (h4 : blueberry = 7)
  (h5 : (total_students - (chocolate + apple + blueberry)) % 2 = 0) :
  let remaining := total_students - (chocolate + apple + blueberry)
  let lemon := remaining / 2
  (lemon : ℚ) / total_students * 360 = 40.5 := by
  sorry

end lemon_pie_degrees_l3444_344494


namespace point_not_in_region_l3444_344419

def planar_region (x y : ℝ) : Prop := 2 * x + 3 * y < 6

theorem point_not_in_region :
  ¬ (planar_region 0 2) ∧
  (planar_region 0 0) ∧
  (planar_region 1 1) ∧
  (planar_region 2 0) :=
by sorry

end point_not_in_region_l3444_344419


namespace required_speed_for_average_l3444_344415

/-- Proves the required speed for the last part of a journey to achieve a desired average speed --/
theorem required_speed_for_average 
  (total_time : ℝ) 
  (initial_time : ℝ) 
  (initial_speed : ℝ) 
  (desired_avg_speed : ℝ) 
  (h1 : total_time = 5) 
  (h2 : initial_time = 3) 
  (h3 : initial_speed = 60) 
  (h4 : desired_avg_speed = 70) : 
  (desired_avg_speed * total_time - initial_speed * initial_time) / (total_time - initial_time) = 85 := by
  sorry

#check required_speed_for_average

end required_speed_for_average_l3444_344415


namespace levels_for_110_blocks_l3444_344480

/-- The number of blocks in the nth level of the pattern -/
def blocks_in_level (n : ℕ) : ℕ := 2 + 2 * (n - 1)

/-- The total number of blocks used up to the nth level -/
def total_blocks (n : ℕ) : ℕ := n * (n + 1)

/-- The theorem stating that 10 levels are needed to use exactly 110 blocks -/
theorem levels_for_110_blocks :
  ∃ (n : ℕ), n > 0 ∧ total_blocks n = 110 ∧ 
  (∀ (m : ℕ), m > 0 ∧ m ≠ n → total_blocks m ≠ 110) :=
sorry

end levels_for_110_blocks_l3444_344480


namespace greatest_prime_factor_of_factorial_sum_l3444_344401

theorem greatest_prime_factor_of_factorial_sum : 
  ∃ (p : ℕ), Nat.Prime p ∧ 
  p ∣ (Nat.factorial 15 + Nat.factorial 17) ∧ 
  ∀ (q : ℕ), Nat.Prime q → q ∣ (Nat.factorial 15 + Nat.factorial 17) → q ≤ p :=
by sorry

end greatest_prime_factor_of_factorial_sum_l3444_344401


namespace fuel_used_calculation_l3444_344432

/-- Calculates the total fuel used given the initial capacity, intermediate reading, and final reading after refill -/
def total_fuel_used (initial_capacity : ℝ) (intermediate_reading : ℝ) (final_reading : ℝ) : ℝ :=
  (initial_capacity - intermediate_reading) + (initial_capacity - final_reading)

/-- Theorem stating that the total fuel used is 4582 L given the specific readings -/
theorem fuel_used_calculation :
  let initial_capacity : ℝ := 3000
  let intermediate_reading : ℝ := 180
  let final_reading : ℝ := 1238
  total_fuel_used initial_capacity intermediate_reading final_reading = 4582 := by
  sorry

#eval total_fuel_used 3000 180 1238

end fuel_used_calculation_l3444_344432


namespace sum_of_a_and_c_l3444_344483

theorem sum_of_a_and_c (a b c d : ℝ) 
  (h1 : a * b + b * c + c * d + d * a = 20) 
  (h2 : b + d = 4) : 
  a + c = 5 := by
sorry

end sum_of_a_and_c_l3444_344483


namespace quadratic_equation_from_roots_l3444_344493

theorem quadratic_equation_from_roots (r s : ℝ) 
  (sum_roots : r + s = 12)
  (product_roots : r * s = 27)
  (root_relation : s = 3 * r) : 
  ∀ x : ℝ, x^2 - 12*x + 27 = (x - r) * (x - s) := by
sorry

end quadratic_equation_from_roots_l3444_344493


namespace complex_power_2006_l3444_344421

def i : ℂ := Complex.I

theorem complex_power_2006 : ((1 + i) / (1 - i)) ^ 2006 = -1 := by
  sorry

end complex_power_2006_l3444_344421


namespace divisibility_problem_l3444_344499

theorem divisibility_problem (n : ℕ) (h1 : n = 6268440) (h2 : n % 5 = 0) : n % 30 = 0 := by
  sorry

end divisibility_problem_l3444_344499


namespace odd_divisors_implies_perfect_square_l3444_344424

/-- The number of divisors of a natural number -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- A natural number is a perfect square if there exists an integer k such that n = k^2 -/
def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

/-- If a natural number has an odd number of divisors, then it is a perfect square -/
theorem odd_divisors_implies_perfect_square (n : ℕ) : 
  Odd (num_divisors n) → is_perfect_square n := by sorry

end odd_divisors_implies_perfect_square_l3444_344424


namespace imaginary_part_of_complex_fraction_l3444_344467

theorem imaginary_part_of_complex_fraction :
  Complex.im ((3 * Complex.I + 4) / (1 + 2 * Complex.I)) = -1 := by
  sorry

end imaginary_part_of_complex_fraction_l3444_344467


namespace no_simultaneous_squares_l3444_344491

theorem no_simultaneous_squares : ∀ n : ℤ, ¬(∃ a b c : ℤ, (10 * n - 1 = a^2) ∧ (13 * n - 1 = b^2) ∧ (85 * n - 1 = c^2)) := by
  sorry

end no_simultaneous_squares_l3444_344491


namespace old_manufacturing_cost_calculation_l3444_344473

def selling_price : ℝ := 100
def new_profit_percentage : ℝ := 0.50
def old_profit_percentage : ℝ := 0.20
def new_manufacturing_cost : ℝ := 50

theorem old_manufacturing_cost_calculation :
  let old_manufacturing_cost := selling_price * (1 - old_profit_percentage)
  old_manufacturing_cost = 80 :=
by sorry

end old_manufacturing_cost_calculation_l3444_344473


namespace work_left_fraction_l3444_344423

theorem work_left_fraction (a_days b_days work_days : ℚ) 
  (ha : a_days = 20)
  (hb : b_days = 30)
  (hw : work_days = 4) : 
  1 - work_days * (1 / a_days + 1 / b_days) = 2 / 3 := by
  sorry

end work_left_fraction_l3444_344423


namespace problem_solution_l3444_344459

theorem problem_solution : (10^3 - (270 * (1/3))) + Real.sqrt 144 = 922 := by
  sorry

end problem_solution_l3444_344459


namespace linear_system_det_proof_l3444_344488

/-- Given a linear equation system represented by an augmented matrix,
    prove that the determinant of a specific matrix using the solution is -1 -/
theorem linear_system_det_proof (a₁ a₂ b₁ b₂ c₁ c₂ : ℝ) :
  a₁ = 2 ∧ b₁ = 0 ∧ c₁ = 2 ∧ a₂ = 3 ∧ b₂ = 1 ∧ c₂ = 2 →
  ∃ x y : ℝ, a₁ * x + b₁ * y = c₁ ∧ a₂ * x + b₂ * y = c₂ →
  x * 2 - y * (-3) = -1 := by
sorry


end linear_system_det_proof_l3444_344488


namespace geometric_sequence_general_term_l3444_344402

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def GeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

/-- Theorem: For a geometric sequence with first term a₁ and common ratio q, 
    the general term a_n is equal to a₁qⁿ⁻¹. -/
theorem geometric_sequence_general_term 
  (a : ℕ → ℝ) (q : ℝ) (a₁ : ℝ) (h : GeometricSequence a q) (h₁ : a 1 = a₁) :
  ∀ n : ℕ, a n = a₁ * q ^ (n - 1) := by
  sorry

end geometric_sequence_general_term_l3444_344402


namespace sum_of_fractions_equals_two_ninths_l3444_344443

theorem sum_of_fractions_equals_two_ninths :
  let sum := (1 : ℚ) / (3 * 4) + 1 / (4 * 5) + 1 / (5 * 6) + 1 / (6 * 7) + 1 / (7 * 8) + 1 / (8 * 9)
  sum = 2 / 9 := by
  sorry

end sum_of_fractions_equals_two_ninths_l3444_344443


namespace symmetric_partitions_generating_function_main_theorem_l3444_344465

/-- A partition is a non-increasing sequence of positive integers. -/
def Partition := List Nat

/-- A partition is symmetric if its Ferrers diagram is symmetric with respect to the diagonal. -/
def IsSymmetric (p : Partition) : Prop := sorry

/-- A partition consists of distinct odd parts if all its parts are odd and unique. -/
def HasDistinctOddParts (p : Partition) : Prop := sorry

/-- The generating function for partitions with a given property. -/
noncomputable def GeneratingFunction (P : Partition → Prop) : ℕ → ℚ := sorry

/-- The infinite product ∏_{k=1}^{∞} (1 + x^(2k+1)) -/
noncomputable def InfiniteProduct : ℕ → ℚ := sorry

theorem symmetric_partitions_generating_function :
  GeneratingFunction IsSymmetric = GeneratingFunction HasDistinctOddParts :=
by sorry

theorem main_theorem :
  GeneratingFunction IsSymmetric = InfiniteProduct :=
by sorry

end symmetric_partitions_generating_function_main_theorem_l3444_344465


namespace imaginary_part_of_complex_fraction_l3444_344435

theorem imaginary_part_of_complex_fraction (z : ℂ) : z = (2 + I) / (3 - I) → z.im = 1/2 := by
  sorry

end imaginary_part_of_complex_fraction_l3444_344435


namespace real_roots_sum_product_l3444_344478

theorem real_roots_sum_product (a b : ℝ) : 
  (a^4 - 6*a^2 - a + 2 = 0) → 
  (b^4 - 6*b^2 - b + 2 = 0) → 
  (∀ x : ℝ, x^4 - 6*x^2 - x + 2 = 0 → x = a ∨ x = b) →
  a * b + a + b = 1 := by
  sorry

end real_roots_sum_product_l3444_344478


namespace director_dividends_director_dividends_calculation_l3444_344468

/-- Calculates the dividends for the General Director given the financial data of the company. -/
theorem director_dividends (revenue : ℝ) (expenses : ℝ) (tax_rate : ℝ)
                           (monthly_loan_payment : ℝ) (annual_interest : ℝ)
                           (total_shares : ℕ) (director_shares : ℕ) : ℝ :=
  let net_profit := (revenue - expenses) - (revenue - expenses) * tax_rate
  let total_loan_payments := monthly_loan_payment * 12 - annual_interest
  let profits_for_dividends := net_profit - total_loan_payments
  let dividend_per_share := profits_for_dividends / total_shares
  dividend_per_share * director_shares

/-- The General Director's dividends are 246,400.0 rubles given the specified financial conditions. -/
theorem director_dividends_calculation :
  director_dividends 1500000 674992 0.2 23914 74992 1000 550 = 246400 := by
  sorry

end director_dividends_director_dividends_calculation_l3444_344468


namespace total_miles_walked_l3444_344440

-- Define the number of islands
def num_islands : ℕ := 4

-- Define the number of days to explore each island
def days_per_island : ℚ := 3/2

-- Define the daily walking distances for each type of island
def miles_per_day_type1 : ℕ := 20
def miles_per_day_type2 : ℕ := 25

-- Define the number of islands for each type
def num_islands_type1 : ℕ := 2
def num_islands_type2 : ℕ := 2

-- Theorem to prove
theorem total_miles_walked :
  (num_islands_type1 * miles_per_day_type1 + num_islands_type2 * miles_per_day_type2) * days_per_island = 135 := by
  sorry


end total_miles_walked_l3444_344440


namespace corrected_mean_problem_l3444_344458

/-- Calculates the corrected mean of a set of observations after fixing an error -/
def corrected_mean (n : ℕ) (original_mean : ℚ) (wrong_value : ℚ) (correct_value : ℚ) : ℚ :=
  (n * original_mean - wrong_value + correct_value) / n

/-- Theorem stating that the corrected mean for the given problem is 36.42 -/
theorem corrected_mean_problem : 
  corrected_mean 50 36 23 44 = 36.42 := by
  sorry

end corrected_mean_problem_l3444_344458


namespace complex_number_problem_l3444_344417

theorem complex_number_problem (a : ℝ) (z : ℂ) (h1 : z = a + I) 
  (h2 : (Complex.I * 2 + 1) * z ∈ {w : ℂ | w.re = 0 ∧ w.im ≠ 0}) :
  z = 2 + I ∧ Complex.abs (z / (2 - I)) = 1 := by sorry

end complex_number_problem_l3444_344417


namespace ordering_abc_l3444_344475

theorem ordering_abc (a b c : ℝ) : 
  a = 31/32 → 
  b = Real.cos (1/4) → 
  c = 4 * Real.sin (1/4) → 
  c > b ∧ b > a := by
  sorry

end ordering_abc_l3444_344475


namespace parallel_vectors_x_value_l3444_344454

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 * b.2 = k * a.2 * b.1

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (2, 5)
  let b : ℝ × ℝ := (x, -2)
  parallel a b → x = -4/5 :=
by
  sorry

end parallel_vectors_x_value_l3444_344454


namespace S_inter_T_eq_T_l3444_344436

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem S_inter_T_eq_T : S ∩ T = T := by sorry

end S_inter_T_eq_T_l3444_344436


namespace middle_part_of_proportional_division_l3444_344404

theorem middle_part_of_proportional_division (total : ℚ) (p1 p2 p3 : ℚ) :
  total = 96 →
  p1 + p2 + p3 = total →
  p2 = (1/4) * p1 →
  p3 = (1/8) * p1 →
  p2 = 17 + 21/44 :=
by sorry

end middle_part_of_proportional_division_l3444_344404


namespace divisibility_by_eight_l3444_344407

theorem divisibility_by_eight (a b c d : ℤ) :
  8 ∣ (1000 * a + 100 * b + 10 * c + d) ↔ 8 ∣ (4 * b + 2 * c + d) := by
  sorry

end divisibility_by_eight_l3444_344407


namespace wicket_keeper_age_difference_l3444_344428

theorem wicket_keeper_age_difference (team_size : ℕ) (team_avg_age : ℝ) (remaining_players : ℕ) (age_difference : ℝ) :
  team_size = 11 →
  team_avg_age = 21 →
  remaining_players = 9 →
  age_difference = 1 →
  let total_age := team_size * team_avg_age
  let remaining_avg_age := team_avg_age - age_difference
  let remaining_total_age := remaining_players * remaining_avg_age
  let wicket_keeper_age := total_age - (remaining_total_age + team_avg_age)
  wicket_keeper_age - team_avg_age = 9 := by
  sorry

end wicket_keeper_age_difference_l3444_344428


namespace quadratic_completion_l3444_344400

/-- The quadratic function we're working with -/
def f (x : ℝ) : ℝ := x^2 - 24*x + 50

/-- The completed square form of our quadratic -/
def g (x b c : ℝ) : ℝ := (x + b)^2 + c

/-- Theorem stating that f can be written in the form of g, and b + c = -106 -/
theorem quadratic_completion (b c : ℝ) : 
  (∀ x, f x = g x b c) → b + c = -106 := by
  sorry

end quadratic_completion_l3444_344400


namespace hexagon_planes_count_l3444_344482

/-- A regular dodecahedron in three-dimensional space. -/
structure RegularDodecahedron

/-- A plane in three-dimensional space. -/
structure Plane

/-- The number of large diagonals in a regular dodecahedron. -/
def num_large_diagonals : ℕ := 10

/-- The number of planes perpendicular to each large diagonal that produce a regular hexagon slice. -/
def planes_per_diagonal : ℕ := 3

/-- A function that counts the number of planes intersecting a regular dodecahedron to produce a regular hexagon. -/
def count_hexagon_planes (d : RegularDodecahedron) : ℕ :=
  num_large_diagonals * planes_per_diagonal

/-- Theorem stating that the number of planes intersecting a regular dodecahedron to produce a regular hexagon is 30. -/
theorem hexagon_planes_count (d : RegularDodecahedron) :
  count_hexagon_planes d = 30 := by sorry

end hexagon_planes_count_l3444_344482


namespace f_neg_five_halves_l3444_344461

-- Define the function f
def f : ℝ → ℝ := sorry

-- f is an odd function
axiom f_odd : ∀ x, f (-x) = -f x

-- f has a period of 2
axiom f_periodic : ∀ x, f (x + 2) = f x

-- f(x) = 2x(1-x) when 0 ≤ x ≤ 1
axiom f_def : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = 2 * x * (1 - x)

-- Theorem to prove
theorem f_neg_five_halves : f (-5/2) = -1/2 := sorry

end f_neg_five_halves_l3444_344461

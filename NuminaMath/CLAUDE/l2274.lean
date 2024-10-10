import Mathlib

namespace fence_bricks_l2274_227459

/-- Calculates the number of bricks needed for a rectangular fence -/
def bricks_needed (length width height depth : ℕ) : ℕ :=
  4 * length * width * depth

theorem fence_bricks :
  bricks_needed 20 5 2 1 = 800 := by
  sorry

end fence_bricks_l2274_227459


namespace dealer_profit_approx_89_99_l2274_227475

/-- Calculates the dealer's profit percentage given the number of articles purchased,
    total purchase price, number of articles sold, and total selling price. -/
def dealer_profit_percentage (articles_purchased : ℕ) (purchase_price : ℚ) 
                             (articles_sold : ℕ) (selling_price : ℚ) : ℚ :=
  let cp_per_article := purchase_price / articles_purchased
  let sp_per_article := selling_price / articles_sold
  let profit_per_article := sp_per_article - cp_per_article
  (profit_per_article / cp_per_article) * 100

/-- Theorem stating that the dealer's profit percentage is approximately 89.99%
    given the specific conditions of the problem. -/
theorem dealer_profit_approx_89_99 :
  ∃ ε > 0, abs (dealer_profit_percentage 15 25 12 38 - 89.99) < ε :=
sorry

end dealer_profit_approx_89_99_l2274_227475


namespace a_representation_theorem_l2274_227410

theorem a_representation_theorem (m n : ℕ) (hm : m ≥ 2) (hn : n ≥ 2) :
  ∃ k : ℕ, ((n + Real.sqrt (n^2 - 4)) / 2) ^ m = (k + Real.sqrt (k^2 - 4)) / 2 := by
  sorry

end a_representation_theorem_l2274_227410


namespace complex_number_problem_l2274_227404

theorem complex_number_problem (α β : ℂ) 
  (h1 : (α + β).re > 0)
  (h2 : (Complex.I * (α - 3 * β)).re > 0)
  (h3 : β = 4 + 3 * Complex.I) :
  α = 12 - 3 * Complex.I := by
  sorry

end complex_number_problem_l2274_227404


namespace fraction_equality_l2274_227463

theorem fraction_equality : (1 / 4 - 1 / 6) / (1 / 3 + 1 / 2) = 1 / 10 := by
  sorry

end fraction_equality_l2274_227463


namespace donut_combinations_l2274_227457

/-- The number of donut types available -/
def num_types : ℕ := 5

/-- The total number of donuts to be purchased -/
def total_donuts : ℕ := 8

/-- The minimum number of type A donuts required -/
def min_type_a : ℕ := 2

/-- The minimum number of donuts required for each of the other types -/
def min_other_types : ℕ := 1

/-- The number of remaining donuts to be distributed -/
def remaining_donuts : ℕ := total_donuts - (min_type_a + (num_types - 1) * min_other_types)

/-- The number of ways to distribute the remaining donuts -/
def num_combinations : ℕ := num_types + (num_types.choose 2)

theorem donut_combinations :
  num_combinations = 15 :=
sorry

end donut_combinations_l2274_227457


namespace infinite_points_in_circle_l2274_227485

/-- A point in the xy-plane with rational coordinates -/
structure RationalPoint where
  x : ℚ
  y : ℚ

/-- Predicate to check if a point is strictly inside the circle -/
def is_inside_circle (p : RationalPoint) : Prop :=
  p.x^2 + p.y^2 < 9

/-- Predicate to check if a point has positive coordinates -/
def has_positive_coordinates (p : RationalPoint) : Prop :=
  p.x > 0 ∧ p.y > 0

theorem infinite_points_in_circle :
  ∃ (S : Set RationalPoint), (∀ p ∈ S, is_inside_circle p ∧ has_positive_coordinates p) ∧ Set.Infinite S :=
sorry

end infinite_points_in_circle_l2274_227485


namespace geometric_sequence_306th_term_l2274_227482

/-- Given a geometric sequence with first term 9 and second term -18,
    the 306th term is -9 * 2^305 -/
theorem geometric_sequence_306th_term :
  ∀ (a : ℕ → ℤ), a 1 = 9 ∧ a 2 = -18 →
  (∀ n : ℕ, n ≥ 1 → a (n + 1) = a n * (-2)) →
  a 306 = -9 * 2^305 := by
sorry

end geometric_sequence_306th_term_l2274_227482


namespace min_value_of_sum_perpendicular_vectors_l2274_227467

/-- Given vectors a = (x, -1) and b = (y, 2) where a ⊥ b, the minimum value of |a + b| is 3 -/
theorem min_value_of_sum_perpendicular_vectors 
  (x y : ℝ) 
  (a b : ℝ × ℝ) 
  (ha : a = (x, -1)) 
  (hb : b = (y, 2)) 
  (h_perp : a.1 * b.1 + a.2 * b.2 = 0) : 
  (∀ (x' y' : ℝ) (a' b' : ℝ × ℝ), 
    a' = (x', -1) → b' = (y', 2) → a'.1 * b'.1 + a'.2 * b'.2 = 0 → 
    Real.sqrt ((a'.1 + b'.1)^2 + (a'.2 + b'.2)^2) ≥ 3) ∧ 
  (∃ (x' y' : ℝ) (a' b' : ℝ × ℝ), 
    a' = (x', -1) ∧ b' = (y', 2) ∧ a'.1 * b'.1 + a'.2 * b'.2 = 0 ∧
    Real.sqrt ((a'.1 + b'.1)^2 + (a'.2 + b'.2)^2) = 3) :=
by sorry

end min_value_of_sum_perpendicular_vectors_l2274_227467


namespace binomial_12_11_l2274_227483

theorem binomial_12_11 : Nat.choose 12 11 = 12 := by sorry

end binomial_12_11_l2274_227483


namespace count_numbers_with_5_or_7_up_to_700_l2274_227455

def count_numbers_with_5_or_7 (n : ℕ) : ℕ :=
  n - (
    -- Three-digit numbers without 5 or 7
    6 * 8 * 8 +
    -- Two-digit numbers without 5 or 7
    8 * 8 +
    -- One-digit numbers without 5 or 7
    7 +
    -- Special case: 700
    1
  )

theorem count_numbers_with_5_or_7_up_to_700 :
  count_numbers_with_5_or_7 700 = 244 := by
  sorry

end count_numbers_with_5_or_7_up_to_700_l2274_227455


namespace weekly_calories_burned_l2274_227473

/-- Represents the duration of activities in minutes for a spinning class -/
structure ClassDuration :=
  (cycling : Nat)
  (strength : Nat)
  (stretching : Nat)

/-- Represents the calorie burn rates per minute for each activity -/
structure CalorieBurnRates :=
  (cycling : Nat)
  (strength : Nat)
  (stretching : Nat)

def monday_class : ClassDuration := ⟨40, 20, 10⟩
def wednesday_class : ClassDuration := ⟨50, 25, 5⟩
def friday_class : ClassDuration := ⟨30, 30, 15⟩

def burn_rates : CalorieBurnRates := ⟨12, 8, 3⟩

def total_calories_burned (classes : List ClassDuration) (rates : CalorieBurnRates) : Nat :=
  let total_cycling := classes.foldl (fun acc c => acc + c.cycling) 0
  let total_strength := classes.foldl (fun acc c => acc + c.strength) 0
  let total_stretching := classes.foldl (fun acc c => acc + c.stretching) 0
  total_cycling * rates.cycling + total_strength * rates.strength + total_stretching * rates.stretching

theorem weekly_calories_burned :
  total_calories_burned [monday_class, wednesday_class, friday_class] burn_rates = 2130 := by
  sorry

end weekly_calories_burned_l2274_227473


namespace max_product_sum_2006_l2274_227402

theorem max_product_sum_2006 :
  ∃ (a b : ℤ), a + b = 2006 ∧
    ∀ (x y : ℤ), x + y = 2006 → x * y ≤ a * b ∧
    a * b = 1006009 := by
  sorry

end max_product_sum_2006_l2274_227402


namespace second_number_difference_l2274_227449

theorem second_number_difference (first second : ℤ) : 
  first + second = 56 → second = 30 → second - first = 4 := by
  sorry

end second_number_difference_l2274_227449


namespace total_toy_cost_l2274_227412

def football_cost : ℝ := 5.71
def marbles_cost : ℝ := 6.59

theorem total_toy_cost : football_cost + marbles_cost = 12.30 := by
  sorry

end total_toy_cost_l2274_227412


namespace unique_sequence_existence_l2274_227461

theorem unique_sequence_existence :
  ∃! (a : ℕ → ℤ), 
    a 1 = 1 ∧ 
    a 2 = 2 ∧ 
    ∀ n : ℕ, n ≥ 1 → (a (n + 1))^3 + 1 = (a n) * (a (n + 2)) := by
  sorry

end unique_sequence_existence_l2274_227461


namespace fraction_simplification_l2274_227440

theorem fraction_simplification : (252 : ℚ) / 21 * 7 / 168 * 12 / 4 = 3 / 2 := by
  sorry

end fraction_simplification_l2274_227440


namespace max_quotient_bound_l2274_227469

theorem max_quotient_bound (a b : ℝ) 
  (ha : 400 ≤ a ∧ a ≤ 800)
  (hb : 400 ≤ b ∧ b ≤ 1600)
  (hab : a + b ≤ 2000) :
  b / a ≤ 4 :=
by sorry

end max_quotient_bound_l2274_227469


namespace test_failure_probability_l2274_227447

theorem test_failure_probability
  (total : ℕ)
  (passed_first : ℕ)
  (passed_second : ℕ)
  (passed_third : ℕ)
  (passed_first_and_second : ℕ)
  (passed_second_and_third : ℕ)
  (passed_first_and_third : ℕ)
  (passed_all : ℕ)
  (h_total : total = 200)
  (h_first : passed_first = 110)
  (h_second : passed_second = 80)
  (h_third : passed_third = 70)
  (h_first_second : passed_first_and_second = 35)
  (h_second_third : passed_second_and_third = 30)
  (h_first_third : passed_first_and_third = 40)
  (h_all : passed_all = 20) :
  (total - (passed_first + passed_second + passed_third
          - passed_first_and_second - passed_second_and_third - passed_first_and_third
          + passed_all)) / total = 1 / 8 := by
  sorry

end test_failure_probability_l2274_227447


namespace quadratic_inequality_solution_set_l2274_227464

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 - 3*x - 10 > 0} = {x : ℝ | x < -2 ∨ x > 5} :=
by sorry

end quadratic_inequality_solution_set_l2274_227464


namespace paperback_ratio_total_books_nonfiction_difference_l2274_227439

/-- Thabo's book collection -/
structure BookCollection where
  total : ℕ
  paperback_fiction : ℕ
  paperback_nonfiction : ℕ
  hardcover_nonfiction : ℕ

/-- The properties of Thabo's book collection -/
def thabos_books : BookCollection :=
  { total := 280,
    paperback_fiction := 150,
    paperback_nonfiction := 75,
    hardcover_nonfiction := 55 }

/-- Theorem stating the ratio of paperback fiction to paperback nonfiction books -/
theorem paperback_ratio (b : BookCollection) (h1 : b = thabos_books) :
  b.paperback_fiction / b.paperback_nonfiction = 2 := by
  sorry

/-- All books are accounted for -/
theorem total_books (b : BookCollection) (h1 : b = thabos_books) :
  b.total = b.paperback_fiction + b.paperback_nonfiction + b.hardcover_nonfiction := by
  sorry

/-- Relationship between paperback nonfiction and hardcover nonfiction books -/
theorem nonfiction_difference (b : BookCollection) (h1 : b = thabos_books) :
  b.paperback_nonfiction = b.hardcover_nonfiction + 20 := by
  sorry

end paperback_ratio_total_books_nonfiction_difference_l2274_227439


namespace sum_simplification_l2274_227420

theorem sum_simplification : -1^2004 + (-1)^2005 + 1^2006 - 1^2007 = -2 := by
  sorry

end sum_simplification_l2274_227420


namespace triangle_construction_l2274_227422

/-- Given a triangle ABC with vertices A(-1, 0), B(1, 0), and C(3a, 3b),
    this theorem proves that it satisfies the specified conditions. -/
theorem triangle_construction (a b : ℝ) (OH_length AB_length : ℝ) :
  let A : ℝ × ℝ := (-1, 0)
  let B : ℝ × ℝ := (1, 0)
  let C : ℝ × ℝ := (3*a, 3*b)
  let O : ℝ × ℝ := (0, b)  -- Circumcenter
  let H : ℝ × ℝ := (3*a, b)  -- Orthocenter
  -- Distance between O and H
  (O.1 - H.1)^2 + (O.2 - H.2)^2 = OH_length^2 ∧
  -- OH parallel to AB
  (O.2 - H.2) * (A.1 - B.1) = (O.1 - H.1) * (A.2 - B.2) ∧
  -- Length of AB
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = AB_length^2 := by
  sorry

end triangle_construction_l2274_227422


namespace thirty_percent_of_hundred_l2274_227497

theorem thirty_percent_of_hundred : (30 : ℝ) = 100 * (30 / 100) := by
  sorry

end thirty_percent_of_hundred_l2274_227497


namespace absolute_value_theorem_l2274_227431

theorem absolute_value_theorem (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a^2 + b^2 = 6*a*b) :
  |((a + 2*b) / (a - b))| = Real.sqrt 14 / 2 := by
sorry

end absolute_value_theorem_l2274_227431


namespace roots_of_equation_l2274_227436

def equation (x : ℝ) : ℝ := (x^2 - 5*x + 6) * (x - 1) * (x + 3)

theorem roots_of_equation :
  {x : ℝ | equation x = 0} = {-3, 1, 2, 3} := by sorry

end roots_of_equation_l2274_227436


namespace baseball_card_packs_l2274_227417

/-- The number of packs of baseball cards for a group of people -/
def total_packs (num_people : ℕ) (cards_per_person : ℕ) (cards_per_pack : ℕ) : ℕ :=
  num_people * (cards_per_person / cards_per_pack)

/-- Theorem: Four people buying 540 cards each, with 20 cards per pack, have 108 packs in total -/
theorem baseball_card_packs :
  total_packs 4 540 20 = 108 := by
  sorry

end baseball_card_packs_l2274_227417


namespace heart_ratio_l2274_227403

-- Define the ♡ operation
def heart (n m : ℕ) : ℚ := 3 * (n^3 : ℚ) * (m^2 : ℚ)

-- State the theorem
theorem heart_ratio : (heart 3 5) / (heart 5 3) = 3 / 5 := by
  sorry

end heart_ratio_l2274_227403


namespace arithmetic_mean_of_given_numbers_l2274_227421

theorem arithmetic_mean_of_given_numbers : 
  let numbers : List ℕ := [16, 24, 40, 32]
  (numbers.sum / numbers.length : ℚ) = 28 := by
sorry

end arithmetic_mean_of_given_numbers_l2274_227421


namespace circle_line_intersection_point_P_on_line_distance_range_l2274_227451

-- Define the circle C
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 - 3)^2 = 4}

-- Define the line l
def l (m : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | (m + 2) * p.1 + (2 * m + 1) * p.2 = 7 * m + 8}

-- Define the point P when m = 1
def P : ℝ × ℝ := (0, 5)

theorem circle_line_intersection (m : ℝ) : (C ∩ l m).Nonempty := by sorry

theorem point_P_on_line : P ∈ l 1 := by sorry

theorem distance_range : 
  ∀ Q ∈ C, (2 * Real.sqrt 2 - 2) ≤ Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) ∧ 
             Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) ≤ (2 * Real.sqrt 2 + 2) := by sorry

end circle_line_intersection_point_P_on_line_distance_range_l2274_227451


namespace soup_feeding_theorem_l2274_227472

/-- Represents the number of people a can of soup can feed -/
structure SoupCan where
  adults : ℕ
  children : ℕ

/-- Represents the soup feeding scenario -/
structure SoupScenario where
  can_capacity : SoupCan
  total_cans : ℕ
  children_fed : ℕ

/-- Calculates the number of adults that can be fed with the remaining soup -/
def remaining_adults_fed (scenario : SoupScenario) : ℕ :=
  let cans_for_children := scenario.children_fed / scenario.can_capacity.children
  let remaining_cans := scenario.total_cans - cans_for_children
  remaining_cans * scenario.can_capacity.adults

/-- Theorem: Given 8 cans of soup, where each can feeds 4 adults or 6 children,
    after feeding 24 children, the remaining soup can feed 16 adults -/
theorem soup_feeding_theorem (scenario : SoupScenario)
  (h1 : scenario.can_capacity = ⟨4, 6⟩)
  (h2 : scenario.total_cans = 8)
  (h3 : scenario.children_fed = 24) :
  remaining_adults_fed scenario = 16 := by
  sorry

end soup_feeding_theorem_l2274_227472


namespace equivalent_expression_l2274_227471

theorem equivalent_expression (a b c d : ℝ) (h1 : a = 0.37) (h2 : b = 15) (h3 : c = 3.7) (h4 : d = 1.5) (h5 : c = a * 10) (h6 : d = b / 10) : c * d = a * b := by
  sorry

end equivalent_expression_l2274_227471


namespace least_number_for_divisibility_l2274_227429

theorem least_number_for_divisibility (n : ℕ) : 
  (∀ m : ℕ, m < n → ¬((5432 + m) % 5 = 0 ∧ (5432 + m) % 6 = 0 ∧ (5432 + m) % 4 = 0 ∧ (5432 + m) % 3 = 0)) ∧ 
  ((5432 + n) % 5 = 0 ∧ (5432 + n) % 6 = 0 ∧ (5432 + n) % 4 = 0 ∧ (5432 + n) % 3 = 0) →
  n = 28 := by
sorry

end least_number_for_divisibility_l2274_227429


namespace largest_prime_factor_of_expression_l2274_227441

theorem largest_prime_factor_of_expression (p : Nat) :
  (p.Prime ∧ p ∣ (12^3 + 15^4 - 6^5) ∧ ∀ q, q.Prime → q ∣ (12^3 + 15^4 - 6^5) → q ≤ p) ↔ p = 97 :=
by sorry

end largest_prime_factor_of_expression_l2274_227441


namespace shaded_areas_comparison_l2274_227445

/-- Represents a square with its division and shading pattern -/
structure Square where
  total_divisions : ℕ
  shaded_divisions : ℕ

/-- The three squares in the problem -/
def square_I : Square := { total_divisions := 4, shaded_divisions := 2 }
def square_II : Square := { total_divisions := 9, shaded_divisions := 3 }
def square_III : Square := { total_divisions := 12, shaded_divisions := 4 }

/-- Calculates the shaded area fraction of a square -/
def shaded_area_fraction (s : Square) : ℚ :=
  s.shaded_divisions / s.total_divisions

/-- Theorem stating the relationship between the shaded areas -/
theorem shaded_areas_comparison :
  shaded_area_fraction square_II = shaded_area_fraction square_III ∧
  shaded_area_fraction square_I ≠ shaded_area_fraction square_II :=
by sorry

end shaded_areas_comparison_l2274_227445


namespace area_code_digits_l2274_227490

/-- The number of valid area codes for n digits -/
def validCodes (n : ℕ) : ℕ := 3^n - 1

theorem area_code_digits : 
  ∃ n : ℕ, n > 0 ∧ validCodes n = 26 := by sorry

end area_code_digits_l2274_227490


namespace jewelry_store_problem_l2274_227454

theorem jewelry_store_problem (S P : ℝ) 
  (h1 : S = P + 0.25 * S)
  (h2 : 16 = 0.8 * S - P) :
  P = 240 := by
  sorry

end jewelry_store_problem_l2274_227454


namespace income_comparison_l2274_227478

theorem income_comparison (juan tim mary : ℝ) 
  (h1 : tim = 0.9 * juan) 
  (h2 : mary = 1.44 * juan) : 
  (mary - tim) / tim = 0.6 := by
sorry

end income_comparison_l2274_227478


namespace exactly_one_true_l2274_227453

def p : Prop := ∀ x y : ℝ, x^2 + y^2 = 0 → x = 0 ∧ y = 0

def q : Prop := ∀ (f : ℝ → ℝ) (a b : ℝ), a < b →
  (∃ c ∈ Set.Icc a b, ∀ x ∈ Set.Icc a b, f x ≤ f c) →
  (∃ c ∈ Set.Ioo a b, ∃ ε > 0, ∀ x ∈ Set.Icc a b ∩ Set.Ioo (c - ε) (c + ε), f x ≤ f c)

theorem exactly_one_true : (p ∧ q) ∨ (p ∨ q) ∨ (¬p) ∧ ¬((p ∧ q) ∧ (p ∨ q)) ∧ ¬((p ∧ q) ∧ (¬p)) ∧ ¬((p ∨ q) ∧ (¬p)) := by
  sorry

end exactly_one_true_l2274_227453


namespace aaron_brothers_count_l2274_227423

theorem aaron_brothers_count : ∃ (a : ℕ), a = 4 ∧ 6 = 2 * a - 2 := by
  sorry

end aaron_brothers_count_l2274_227423


namespace license_plate_count_l2274_227443

-- Define the number of possible digits (0-9)
def num_digits : ℕ := 10

-- Define the number of possible letters (A-Z)
def num_letters : ℕ := 26

-- Define the number of digits in the license plate
def digits_in_plate : ℕ := 4

-- Define the number of letters in the license plate
def letters_in_plate : ℕ := 2

-- Define the number of positions where the letter pair can be placed
def letter_pair_positions : ℕ := digits_in_plate + 1

-- Theorem statement
theorem license_plate_count :
  letter_pair_positions * num_digits ^ digits_in_plate * num_letters ^ letters_in_plate = 33800000 :=
by sorry

end license_plate_count_l2274_227443


namespace focus_coordinates_l2274_227468

/-- Represents an ellipse with given properties -/
structure Ellipse where
  center : ℝ × ℝ
  major_axis_endpoints : (ℝ × ℝ) × (ℝ × ℝ)
  minor_axis_endpoints : (ℝ × ℝ) × (ℝ × ℝ)

/-- Calculates the coordinates of the focus with greater y-coordinate for a given ellipse -/
def focus_with_greater_y (e : Ellipse) : ℝ × ℝ :=
  sorry

/-- Theorem stating that for the given ellipse, the focus with greater y-coordinate is at (0, √5/2) -/
theorem focus_coordinates (e : Ellipse) 
  (h1 : e.center = (0, 0))
  (h2 : e.major_axis_endpoints = ((0, 3), (0, -3)))
  (h3 : e.minor_axis_endpoints = ((2, 0), (-2, 0))) :
  focus_with_greater_y e = (0, Real.sqrt 5 / 2) :=
sorry

end focus_coordinates_l2274_227468


namespace johns_earnings_l2274_227491

def tax_calculation (earnings : ℕ) : Prop :=
  let deductions : ℕ := 30000
  let taxable_income : ℕ := earnings - deductions
  let first_bracket : ℕ := 20000
  let first_rate : ℚ := 1/10
  let second_rate : ℚ := 1/5
  let total_tax : ℕ := 12000
  (min taxable_income first_bracket) * first_rate +
  (max (taxable_income - first_bracket) 0) * second_rate = total_tax

theorem johns_earnings : ∃ (earnings : ℕ), tax_calculation earnings ∧ earnings = 100000 :=
sorry

end johns_earnings_l2274_227491


namespace clock_angle_at_2_20_clock_angle_at_2_20_is_50_l2274_227484

/-- The angle between clock hands at 2:20 -/
theorem clock_angle_at_2_20 : ℝ → Prop :=
  λ angle =>
    let total_degrees : ℝ := 360
    let hours_on_clock : ℕ := 12
    let minutes_in_hour : ℕ := 60
    let degrees_per_hour : ℝ := total_degrees / hours_on_clock
    let degrees_per_minute : ℝ := degrees_per_hour / minutes_in_hour
    let hour : ℕ := 2
    let minute : ℕ := 20
    let hour_hand_angle : ℝ := hour * degrees_per_hour + minute * degrees_per_minute
    let minute_hand_angle : ℝ := minute * (total_degrees / minutes_in_hour)
    let angle_diff : ℝ := |minute_hand_angle - hour_hand_angle|
    angle = min angle_diff (total_degrees - angle_diff)

/-- The smaller angle between the hour-hand and minute-hand of a clock at 2:20 is 50° -/
theorem clock_angle_at_2_20_is_50 : clock_angle_at_2_20 50 := by
  sorry

end clock_angle_at_2_20_clock_angle_at_2_20_is_50_l2274_227484


namespace frog_egg_hatching_fraction_l2274_227450

theorem frog_egg_hatching_fraction (total_eggs : ℕ) (dry_up_percent : ℚ) (eaten_percent : ℚ) (hatched_frogs : ℕ) :
  total_eggs = 800 →
  dry_up_percent = 1/10 →
  eaten_percent = 7/10 →
  hatched_frogs = 40 →
  (hatched_frogs : ℚ) / (total_eggs * (1 - dry_up_percent - eaten_percent)) = 1/4 := by
sorry

end frog_egg_hatching_fraction_l2274_227450


namespace simplify_sqrt_expression_l2274_227407

theorem simplify_sqrt_expression (x : ℝ) (h : x ≠ 0) :
  Real.sqrt (1 + ((x^6 - 1) / (2*x^3))^2) = (x^3 / 2) + (1 / (2*x^3)) := by
  sorry

end simplify_sqrt_expression_l2274_227407


namespace unique_x_floor_product_l2274_227434

theorem unique_x_floor_product : ∃! x : ℝ, x > 0 ∧ x * ↑(⌊x⌋) = 80 ∧ x = 80 / 9 := by sorry

end unique_x_floor_product_l2274_227434


namespace parabola_equation_l2274_227496

/-- Prove that for a parabola y² = 2px (p > 0) with focus F(p/2, 0), if there exists a point A 
on the parabola such that AF = 4 and a point B(0, 2) on the y-axis satisfying BA · BF = 0, 
then p = 4. -/
theorem parabola_equation (p : ℝ) (h_p : p > 0) : 
  ∃ (A : ℝ × ℝ), 
    (A.2)^2 = 2 * p * A.1 ∧  -- A is on the parabola
    (A.1 - p/2)^2 + (A.2)^2 = 16 ∧  -- AF = 4
    (A.1 * p/2 + A.2 * (-2)) = 0  -- BA · BF = 0
  → p = 4 := by
  sorry

end parabola_equation_l2274_227496


namespace correct_student_activities_and_championships_l2274_227418

/-- The number of ways for students to sign up for activities and the number of possible championship outcomes -/
def student_activities_and_championships 
  (num_students : ℕ) 
  (num_activities : ℕ) 
  (num_championships : ℕ) : ℕ × ℕ :=
  (num_activities ^ num_students, num_students ^ num_championships)

/-- Theorem stating the correct number of ways for 4 students to sign up for 3 activities and compete in 3 championships -/
theorem correct_student_activities_and_championships :
  student_activities_and_championships 4 3 3 = (3^4, 4^3) := by
  sorry

end correct_student_activities_and_championships_l2274_227418


namespace min_value_of_f_b_minimizes_f_l2274_227488

def f (b : ℝ) : ℝ := 2 * b^2 + 8 * b - 4

theorem min_value_of_f (b : ℝ) (h : b ∈ Set.Icc (-10) 0) :
  f b ≥ f (-2) := by sorry

theorem b_minimizes_f :
  ∃ b ∈ Set.Icc (-10 : ℝ) 0, ∀ x ∈ Set.Icc (-10 : ℝ) 0, f b ≤ f x :=
by sorry

end min_value_of_f_b_minimizes_f_l2274_227488


namespace arithmetic_sequence_common_difference_l2274_227446

def arithmetic_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a) 
  (h_a4 : a 4 = 7) 
  (h_sum : a 3 + a 6 = 16) : 
  ∃ d : ℝ, (∀ n, a (n + 1) - a n = d) ∧ d = 2 :=
sorry

end arithmetic_sequence_common_difference_l2274_227446


namespace larger_integer_value_l2274_227448

theorem larger_integer_value (a b : ℕ+) 
  (h_quotient : (a : ℚ) / (b : ℚ) = 7 / 3)
  (h_product : (a : ℕ) * (b : ℕ) = 189) :
  a = 21 := by
sorry

end larger_integer_value_l2274_227448


namespace circle_parabola_tangency_height_difference_l2274_227408

/-- Given a parabola y = 4x^2 and a circle tangent to it at two points,
    the height difference between the circle's center and the points of tangency is 1/8 -/
theorem circle_parabola_tangency_height_difference :
  ∀ (a b r : ℝ),
  (∀ x y : ℝ, y = 4 * x^2 → x^2 + (y - b)^2 = r^2) →  -- Circle equation
  (a^2 + (4 * a^2 - b)^2 = r^2) →                     -- Tangency condition at (a, 4a^2)
  ((-a)^2 + (4 * (-a)^2 - b)^2 = r^2) →               -- Tangency condition at (-a, 4a^2)
  b - 4 * a^2 = 1/8 :=
by sorry

end circle_parabola_tangency_height_difference_l2274_227408


namespace total_price_after_tax_l2274_227411

def original_price : ℝ := 200
def tax_rate : ℝ := 0.15

theorem total_price_after_tax :
  original_price * (1 + tax_rate) = 230 := by sorry

end total_price_after_tax_l2274_227411


namespace min_value_theorem_l2274_227474

theorem min_value_theorem (a b c d e f g h : ℝ) 
  (h1 : a * b * c * d = 8) 
  (h2 : e * f * g * h = 16) : 
  (a * e)^2 + (b * f)^2 + (c * g)^2 + (d * h)^2 + 
  (a * b)^2 + (c * d)^2 + (e * f)^2 + (g * h)^2 ≥ 64 :=
sorry

end min_value_theorem_l2274_227474


namespace no_positive_integer_solutions_l2274_227487

theorem no_positive_integer_solutions (A : ℕ) : 
  A > 0 → A < 10 → ¬∃ x : ℕ, x > 0 ∧ x^2 - (2*A + 1)*x + (A + 1)*(10 + A) = 0 := by
  sorry

end no_positive_integer_solutions_l2274_227487


namespace semicircle_path_equality_l2274_227416

theorem semicircle_path_equality :
  let large_diameter : ℝ := 20
  let small_diameter : ℝ := 10
  let large_arc_length := π * large_diameter / 2
  let small_arc_length := π * small_diameter / 2
  large_arc_length = 2 * small_arc_length :=
by sorry

end semicircle_path_equality_l2274_227416


namespace product_twice_prime_p_squared_minus_2q_prime_p_plus_2q_prime_l2274_227465

/-- Given that p and q are primes and x^2 - px + 2q = 0 has integral roots which are consecutive primes -/
def consecutive_prime_roots (p q : ℕ) : Prop :=
  ∃ (r s : ℕ), Prime r ∧ Prime s ∧ s = r + 1 ∧ 
  r * s = 2 * q ∧ r + s = p

/-- The product of the roots is twice a prime -/
theorem product_twice_prime {p q : ℕ} (h : consecutive_prime_roots p q) : 
  ∃ (k : ℕ), Prime k ∧ 2 * q = 2 * k :=
sorry

/-- p^2 - 2q is prime -/
theorem p_squared_minus_2q_prime {p q : ℕ} (h : consecutive_prime_roots p q) :
  Prime (p^2 - 2*q) :=
sorry

/-- p + 2q is prime -/
theorem p_plus_2q_prime {p q : ℕ} (h : consecutive_prime_roots p q) :
  Prime (p + 2*q) :=
sorry

end product_twice_prime_p_squared_minus_2q_prime_p_plus_2q_prime_l2274_227465


namespace large_pizza_price_l2274_227479

/-- The price of a large pizza given the sales information -/
theorem large_pizza_price (small_price : ℕ) (total_sales : ℕ) (small_sold : ℕ) (large_sold : ℕ)
  (h1 : small_price = 2)
  (h2 : total_sales = 40)
  (h3 : small_sold = 8)
  (h4 : large_sold = 3) :
  (total_sales - small_price * small_sold) / large_sold = 8 := by
  sorry

#check large_pizza_price

end large_pizza_price_l2274_227479


namespace total_turnips_count_l2274_227425

/-- The number of turnips grown by Melanie -/
def melanie_turnips : ℕ := 1395

/-- The number of turnips grown by Benny -/
def benny_turnips : ℕ := 11380

/-- The number of turnips grown by Jack -/
def jack_turnips : ℕ := 15825

/-- The number of turnips grown by Lynn -/
def lynn_turnips : ℕ := 23500

/-- The total number of turnips grown by all four people -/
def total_turnips : ℕ := melanie_turnips + benny_turnips + jack_turnips + lynn_turnips

theorem total_turnips_count : total_turnips = 52100 := by
  sorry

end total_turnips_count_l2274_227425


namespace midsize_rental_cost_l2274_227433

/-- Represents the types of rental cars --/
inductive CarType
| Economy
| MidSize
| Luxury

/-- Represents the rental rates for a car type --/
structure RentalRates where
  dailyRate : Nat
  weeklyRate : Nat
  discountedRateUpTo10Days : Nat
  discountedRateAfter10Days : Nat

/-- Calculate the rental cost for a given number of days --/
def calculateRentalCost (rates : RentalRates) (days : Nat) : Nat :=
  if days ≤ 7 then
    min (days * rates.dailyRate) rates.weeklyRate
  else
    rates.weeklyRate + 
    (min (days - 7) 3 * rates.discountedRateUpTo10Days) +
    (max (days - 10) 0 * rates.discountedRateAfter10Days)

/-- Apply a percentage discount to a given amount --/
def applyDiscount (amount : Nat) (discountPercent : Nat) : Nat :=
  amount - (amount * discountPercent / 100)

/-- Theorem: The cost of renting a mid-size car for 13 days with a 10% discount is $306 --/
theorem midsize_rental_cost : 
  let midSizeRates : RentalRates := {
    dailyRate := 30,
    weeklyRate := 190,
    discountedRateUpTo10Days := 25,
    discountedRateAfter10Days := 20
  }
  let rentalDays := 13
  let discountPercent := 10
  applyDiscount (calculateRentalCost midSizeRates rentalDays) discountPercent = 306 := by
  sorry

end midsize_rental_cost_l2274_227433


namespace tyler_clay_age_sum_l2274_227427

theorem tyler_clay_age_sum :
  ∀ (tyler_age clay_age : ℕ),
    tyler_age = 5 →
    tyler_age = 3 * clay_age + 1 →
    tyler_age + clay_age = 6 :=
by
  sorry

end tyler_clay_age_sum_l2274_227427


namespace marathon_training_duration_l2274_227489

theorem marathon_training_duration (d : ℕ) : 
  (5 * d + 10 * d + 20 * d = 1050) → d = 30 := by
  sorry

end marathon_training_duration_l2274_227489


namespace fran_travel_time_l2274_227414

/-- Proves that given Joann's speed and time, and Fran's speed, Fran will take 3 hours to travel the same distance as Joann. -/
theorem fran_travel_time (joann_speed fran_speed : ℝ) (joann_time : ℝ) 
  (h1 : joann_speed = 15)
  (h2 : joann_time = 4)
  (h3 : fran_speed = 20) :
  (joann_speed * joann_time) / fran_speed = 3 := by
  sorry

#check fran_travel_time

end fran_travel_time_l2274_227414


namespace candidate_count_l2274_227430

theorem candidate_count (selection_ways : ℕ) : 
  (selection_ways = 90) → 
  (∃ n : ℕ, n * (n - 1) = selection_ways ∧ n > 1) → 
  (∃ n : ℕ, n * (n - 1) = selection_ways ∧ n = 10) :=
by
  sorry

end candidate_count_l2274_227430


namespace inverse_157_mod_263_l2274_227405

/-- The multiplicative inverse of 157 modulo 263 is 197 -/
theorem inverse_157_mod_263 : ∃ x : ℕ, x < 263 ∧ (157 * x) % 263 = 1 :=
by
  use 197
  sorry

end inverse_157_mod_263_l2274_227405


namespace coefficient_expansion_l2274_227437

/-- The coefficient of x³ in the expansion of (x²-1)(x-2)⁷ -/
def coefficient_x_cubed : ℤ := -112

/-- The binomial coefficient (n choose k) -/
def binomial (n k : ℕ) : ℕ := sorry

theorem coefficient_expansion :
  coefficient_x_cubed =
    binomial 7 6 * (-2)^6 - binomial 7 4 * (-2)^4 :=
by sorry

end coefficient_expansion_l2274_227437


namespace candy_mixture_cost_l2274_227476

/-- Given the conditions of mixing two types of candy, prove the cost of the first type --/
theorem candy_mixture_cost 
  (weight_first : ℝ) 
  (weight_second : ℝ) 
  (cost_second : ℝ) 
  (cost_mixture : ℝ) 
  (h1 : weight_first = 15) 
  (h2 : weight_second = 30) 
  (h3 : cost_second = 5) 
  (h4 : cost_mixture = 6) : 
  ∃ (cost_first : ℝ), cost_first = 8 ∧ 
    weight_first * cost_first + weight_second * cost_second = 
    (weight_first + weight_second) * cost_mixture :=
sorry

end candy_mixture_cost_l2274_227476


namespace quadratic_roots_to_coefficients_l2274_227409

theorem quadratic_roots_to_coefficients :
  ∀ (b c : ℝ),
  (∀ x : ℝ, x^2 - b*x + c = 0 ↔ x = 1 ∨ x = -2) →
  b = -1 ∧ c = -2 := by
  sorry

end quadratic_roots_to_coefficients_l2274_227409


namespace average_equation_solution_l2274_227419

theorem average_equation_solution (x : ℝ) : 
  ((2*x + 12) + (3*x + 3) + (5*x - 8)) / 3 = 3*x + 2 → x = -1 := by
  sorry

end average_equation_solution_l2274_227419


namespace pears_for_strawberries_l2274_227481

-- Define variables for each type of fruit
variable (s r c b p : ℚ)

-- Define exchange rates
def exchange1 : Prop := 11 * s = 14 * r
def exchange2 : Prop := 22 * c = 21 * r
def exchange3 : Prop := 10 * c = 3 * b
def exchange4 : Prop := 5 * p = 2 * b

-- Theorem to prove
theorem pears_for_strawberries 
  (h1 : exchange1 s r)
  (h2 : exchange2 c r)
  (h3 : exchange3 c b)
  (h4 : exchange4 p b) :
  7 * s = 7 * p :=
sorry

end pears_for_strawberries_l2274_227481


namespace noon_temperature_l2274_227499

theorem noon_temperature (T : ℝ) : 
  let temp_4pm := T + 8
  let temp_8pm := temp_4pm - 11
  temp_8pm = T + 1 → T = 4 := by sorry

end noon_temperature_l2274_227499


namespace proportional_function_and_point_l2274_227456

/-- A function representing the relationship between x and y -/
def f (x : ℝ) : ℝ := -2 * x + 2

theorem proportional_function_and_point (k : ℝ) :
  (∀ x y, y + 4 = k * (x - 3)) →  -- Condition 1
  (f 1 = 0) →                    -- Condition 2
  (∃ m, f (m + 1) = 2 * m) →     -- Condition 3
  (∀ x, f x = -2 * x + 2) ∧      -- Conclusion 1
  (f 1 = 0 ∧ f 2 = 0)            -- Conclusion 2 (coordinates of M)
  := by sorry

end proportional_function_and_point_l2274_227456


namespace chocolate_division_l2274_227477

theorem chocolate_division (total_chocolate : ℚ) (num_piles : ℕ) (piles_to_edward : ℕ) :
  total_chocolate = 75 / 7 →
  num_piles = 5 →
  piles_to_edward = 2 →
  piles_to_edward * (total_chocolate / num_piles) = 30 / 7 := by
  sorry

end chocolate_division_l2274_227477


namespace sum_lent_is_1000_l2274_227458

/-- Proves that the sum lent is $1000 given the specified conditions -/
theorem sum_lent_is_1000 
  (interest_rate : ℝ) 
  (loan_duration : ℝ) 
  (interest_difference : ℝ) 
  (h1 : interest_rate = 0.06)
  (h2 : loan_duration = 8)
  (h3 : interest_difference = 520)
  (simple_interest : ℝ → ℝ → ℝ → ℝ)
  (h4 : ∀ P r t, simple_interest P r t = P * r * t) :
  ∃ P : ℝ, P = 1000 ∧ simple_interest P interest_rate loan_duration = P - interest_difference :=
by sorry

end sum_lent_is_1000_l2274_227458


namespace line_m_equation_l2274_227495

/-- Two distinct lines in the xy-plane that intersect at the origin -/
structure IntersectingLines where
  ℓ : Set (ℝ × ℝ)
  m : Set (ℝ × ℝ)
  distinct : ℓ ≠ m
  intersect_origin : (0, 0) ∈ ℓ ∩ m

/-- Reflection of a point about a line -/
def reflect (p : ℝ × ℝ) (line : Set (ℝ × ℝ)) : ℝ × ℝ := sorry

/-- The equation of line ℓ is 2x - y = 0 -/
def line_ℓ_eq (p : ℝ × ℝ) : Prop := 2 * p.1 - p.2 = 0

theorem line_m_equation (lines : IntersectingLines) 
  (h_ℓ_eq : ∀ p ∈ lines.ℓ, line_ℓ_eq p)
  (h_Q : reflect (reflect (-2, 3) lines.ℓ) lines.m = (3, -1)) :
  ∀ p ∈ lines.m, 3 * p.1 + p.2 = 0 := by sorry

end line_m_equation_l2274_227495


namespace circle_c_equation_l2274_227498

/-- A circle C with center on the line 2x-y-7=0 and intersecting the y-axis at points (0, -4) and (0, -2) -/
structure CircleC where
  /-- The center of the circle lies on the line 2x-y-7=0 -/
  center_on_line : ∀ (x y : ℝ), y = 2*x - 7 → (∃ (t : ℝ), x = t ∧ y = 2*t - 7)
  /-- The circle intersects the y-axis at points (0, -4) and (0, -2) -/
  intersects_y_axis : ∃ (r : ℝ), r > 0 ∧ 
    (∃ (cx cy : ℝ), cx^2 + (cy + 4)^2 = r^2 ∧ cx^2 + (cy + 2)^2 = r^2)

/-- The equation of circle C is (x-2)^2+(y+3)^2=5 -/
theorem circle_c_equation (c : CircleC) : 
  ∃ (cx cy : ℝ), (∀ (x y : ℝ), (x - cx)^2 + (y - cy)^2 = 5 ↔ 
    (x - 2)^2 + (y + 3)^2 = 5) :=
sorry

end circle_c_equation_l2274_227498


namespace set_intersection_empty_implies_a_range_l2274_227466

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x : ℝ | |x - a| < 1}
def B : Set ℝ := {x : ℝ | 1 < x ∧ x < 5}

-- State the theorem
theorem set_intersection_empty_implies_a_range (a : ℝ) : 
  A a ∩ B = ∅ → a ≤ 0 ∨ a ≥ 6 := by
  sorry

end set_intersection_empty_implies_a_range_l2274_227466


namespace no_real_roots_l2274_227444

def P : ℕ → (ℝ → ℝ)
  | 0 => λ _ => 1
  | n + 1 => λ x => x^(17*(n+1)) - P n x

theorem no_real_roots : ∀ (n : ℕ) (x : ℝ), P n x ≠ 0 := by
  sorry

end no_real_roots_l2274_227444


namespace shaded_area_square_with_circles_l2274_227401

/-- The area of the shaded region inside a square with circles at its vertices -/
theorem shaded_area_square_with_circles (side_length : ℝ) (circle_radius : ℝ) 
  (h_side : side_length = 8)
  (h_radius : circle_radius = 3 * Real.sqrt 2) : 
  let square_area := side_length ^ 2
  let triangle_area := (side_length / 2) ^ 2 / 2
  let circle_sector_area := π * circle_radius ^ 2 / 4
  let total_excluded_area := 4 * (triangle_area + circle_sector_area)
  square_area - total_excluded_area = 46 - 18 * π :=
by sorry

end shaded_area_square_with_circles_l2274_227401


namespace perfect_square_in_base_k_l2274_227438

theorem perfect_square_in_base_k (k : ℤ) (h : k ≥ 6) :
  (1 * k^8 + 2 * k^7 + 3 * k^6 + 4 * k^5 + 5 * k^4 + 4 * k^3 + 3 * k^2 + 2 * k + 1) =
  (k^4 + k^3 + k^2 + k + 1)^2 := by
  sorry

end perfect_square_in_base_k_l2274_227438


namespace largest_difference_even_digits_l2274_227426

/-- A function that checks if a natural number has all even digits -/
def allEvenDigits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → Even d

/-- A function that checks if a natural number has at least one odd digit -/
def hasOddDigit (n : ℕ) : Prop :=
  ∃ d, d ∈ n.digits 10 ∧ Odd d

/-- The theorem stating the largest possible difference between two 6-digit numbers
    with all even digits, where any number between them has at least one odd digit -/
theorem largest_difference_even_digits :
  ∃ (a b : ℕ),
    (100000 ≤ a ∧ a < 1000000) ∧
    (100000 ≤ b ∧ b < 1000000) ∧
    allEvenDigits a ∧
    allEvenDigits b ∧
    (∀ n, a < n ∧ n < b → hasOddDigit n) ∧
    b - a = 111112 ∧
    (∀ a' b', (100000 ≤ a' ∧ a' < 1000000) →
              (100000 ≤ b' ∧ b' < 1000000) →
              allEvenDigits a' →
              allEvenDigits b' →
              (∀ n, a' < n ∧ n < b' → hasOddDigit n) →
              b' - a' ≤ 111112) :=
by sorry

end largest_difference_even_digits_l2274_227426


namespace men_seated_on_bus_l2274_227480

theorem men_seated_on_bus (total_passengers : ℕ) 
  (h1 : total_passengers = 48) 
  (h2 : (2 : ℕ) * (total_passengers - total_passengers / 3) = total_passengers / 3) 
  (h3 : (8 : ℕ) * ((total_passengers - total_passengers / 3) / 8) = total_passengers - total_passengers / 3) :
  total_passengers - total_passengers / 3 - ((total_passengers - total_passengers / 3) / 8) = 14 := by
  sorry

#check men_seated_on_bus

end men_seated_on_bus_l2274_227480


namespace polynomial_factorization_l2274_227424

theorem polynomial_factorization (m : ℝ) : 
  (∀ x : ℝ, x^2 - m*x - 35 = (x - 5) * (x + 7)) → m = -2 := by
  sorry

end polynomial_factorization_l2274_227424


namespace sqrt_sum_inequality_l2274_227492

theorem sqrt_sum_inequality (a b c x y z : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (hxa : x ≤ a) (hyb : y ≤ b) (hzc : z ≤ c) : 
  Real.sqrt x + Real.sqrt y + Real.sqrt z < Real.sqrt a + Real.sqrt b + Real.sqrt c := by
  sorry

end sqrt_sum_inequality_l2274_227492


namespace solution_set_inequality_l2274_227460

theorem solution_set_inequality (x : ℝ) :
  (Set.Ioo (-2 : ℝ) 0) = {x | |1 + x + x^2/2| < 1} :=
by sorry

end solution_set_inequality_l2274_227460


namespace impossible_transformation_l2274_227432

def sum_of_range (n : ℕ) : ℕ := n * (n + 1) / 2

def is_odd (n : ℕ) : Prop := ∃ k, n = 2 * k + 1

theorem impossible_transformation :
  is_odd (sum_of_range 2021) →
  ¬ ∃ (operations : ℕ), 
    ∃ (final_state : List ℕ),
      (final_state.length = 1 ∧ 
       final_state.head? = some 2048 ∧
       ∀ (intermediate_state : List ℕ),
         intermediate_state.sum = sum_of_range 2021) :=
by sorry

end impossible_transformation_l2274_227432


namespace fred_cards_left_l2274_227452

/-- Represents the number of baseball cards Fred has left after Melanie's purchase. -/
def cards_left (initial : ℕ) (bought : ℕ) : ℕ := initial - bought

/-- Theorem stating that Fred has 2 baseball cards left after Melanie's purchase. -/
theorem fred_cards_left : cards_left 5 3 = 2 := by sorry

end fred_cards_left_l2274_227452


namespace projectile_meeting_time_l2274_227406

/-- Time for two projectiles to meet --/
theorem projectile_meeting_time (initial_distance : ℝ) (speed1 speed2 : ℝ) :
  initial_distance = 1998 →
  speed1 = 444 →
  speed2 = 555 →
  (initial_distance / (speed1 + speed2)) * 60 = 120 := by
  sorry

end projectile_meeting_time_l2274_227406


namespace smaller_pond_green_percentage_is_twenty_percent_l2274_227428

/-- Represents the duck population in two ponds -/
structure DuckPonds where
  total_ducks : ℕ
  smaller_pond_ducks : ℕ
  larger_pond_ducks : ℕ
  larger_pond_green_percentage : ℚ
  total_green_percentage : ℚ

/-- Calculates the percentage of green ducks in the smaller pond -/
def smaller_pond_green_percentage (ponds : DuckPonds) : ℚ :=
  let total_green_ducks := ponds.total_ducks * ponds.total_green_percentage
  let larger_pond_green_ducks := ponds.larger_pond_ducks * ponds.larger_pond_green_percentage
  let smaller_pond_green_ducks := total_green_ducks - larger_pond_green_ducks
  smaller_pond_green_ducks / ponds.smaller_pond_ducks

/-- Theorem: The percentage of green ducks in the smaller pond is 20% -/
theorem smaller_pond_green_percentage_is_twenty_percent (ponds : DuckPonds) 
  (h1 : ponds.total_ducks = 100)
  (h2 : ponds.smaller_pond_ducks = 45)
  (h3 : ponds.larger_pond_ducks = 55)
  (h4 : ponds.larger_pond_green_percentage = 2/5)
  (h5 : ponds.total_green_percentage = 31/100) :
  smaller_pond_green_percentage ponds = 1/5 := by
  sorry

end smaller_pond_green_percentage_is_twenty_percent_l2274_227428


namespace train_distance_theorem_l2274_227400

/-- Calculates the total distance traveled by a train given its speed, initial distance, and time -/
def total_distance (speed : ℚ) (initial_distance : ℚ) (time : ℚ) : ℚ :=
  speed * time + initial_distance

/-- Proves that a train traveling at 1 mile every 2 minutes, starting with an initial distance of 5 miles, 
    will cover a total distance of 50 miles in 1 hour and 30 minutes -/
theorem train_distance_theorem : 
  let speed : ℚ := 1 / 2  -- 1 mile per 2 minutes
  let initial_distance : ℚ := 5
  let time : ℚ := 90  -- 1 hour and 30 minutes in minutes
  total_distance speed initial_distance time = 50 := by
sorry

#eval total_distance (1/2) 5 90

end train_distance_theorem_l2274_227400


namespace tom_seashells_l2274_227494

/-- The number of seashells Tom has after giving some away -/
def remaining_seashells (initial : ℕ) (given_away : ℕ) : ℕ :=
  initial - given_away

/-- Theorem: Tom has 3 seashells after initially finding 5 and giving away 2 -/
theorem tom_seashells : remaining_seashells 5 2 = 3 := by
  sorry

end tom_seashells_l2274_227494


namespace negation_of_existence_l2274_227415

theorem negation_of_existence (x : ℝ) : 
  (¬ ∃ x : ℝ, x^2 - x + 1 = 0) ↔ (∀ x : ℝ, x^2 - x + 1 ≠ 0) := by
  sorry

end negation_of_existence_l2274_227415


namespace f_symmetry_l2274_227435

/-- A function f(x) defined as ax^5 - bx^3 + cx + 1 -/
def f (a b c x : ℝ) : ℝ := a * x^5 - b * x^3 + c * x + 1

/-- Theorem: If f(-2) = -1, then f(2) = 3 -/
theorem f_symmetry (a b c : ℝ) : f a b c (-2) = -1 → f a b c 2 = 3 := by
  sorry

end f_symmetry_l2274_227435


namespace factorization_ax2_minus_a_l2274_227470

theorem factorization_ax2_minus_a (a x : ℝ) : a * x^2 - a = a * (x + 1) * (x - 1) := by sorry

end factorization_ax2_minus_a_l2274_227470


namespace opposite_of_one_l2274_227413

/-- Two real numbers are opposites if their sum is zero -/
def IsOpposite (x y : ℝ) : Prop := x + y = 0

/-- If a is the opposite of 1, then a = -1 -/
theorem opposite_of_one (a : ℝ) (h : IsOpposite a 1) : a = -1 := by
  sorry

end opposite_of_one_l2274_227413


namespace line_equation_l2274_227462

/-- A line passing through (1,2) and intersecting x^2 + y^2 = 9 with chord length 4√2 -/
def line_through_circle (l : Set (ℝ × ℝ)) : Prop :=
  ∃ (A B : ℝ × ℝ),
    (1, 2) ∈ l ∧
    A ∈ l ∧ B ∈ l ∧
    A.1^2 + A.2^2 = 9 ∧
    B.1^2 + B.2^2 = 9 ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 32

/-- The equation of the line is either x = 1 or 3x - 4y + 5 = 0 -/
theorem line_equation (l : Set (ℝ × ℝ)) (h : line_through_circle l) :
  (∀ (x y : ℝ), (x, y) ∈ l ↔ x = 1) ∨
  (∀ (x y : ℝ), (x, y) ∈ l ↔ 3*x - 4*y + 5 = 0) :=
sorry

end line_equation_l2274_227462


namespace slope_problem_l2274_227486

theorem slope_problem (m : ℝ) (h1 : m > 0) 
  (h2 : (2*m - 1) / ((m + 1) - 2*m) = m) : m = Real.sqrt 2 / 2 := by
  sorry

end slope_problem_l2274_227486


namespace compare_cube_roots_l2274_227442

open Real

theorem compare_cube_roots : 
  (25 / 3) ^ (1/3) < (25 ^ (1/3)) / 3 + (6 / 5) ^ (1/3) ∧ 
  (25 ^ (1/3)) / 3 + (6 / 5) ^ (1/3) < (1148 / 135) ^ (1/3) := by
  sorry

end compare_cube_roots_l2274_227442


namespace sum_of_squares_l2274_227493

theorem sum_of_squares (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (sum_zero : a + b + c = 0) (cube_eq_seventh : a^3 + b^3 + c^3 = a^7 + b^7 + c^7) :
  a^2 + b^2 + c^2 = -6/7 := by sorry

end sum_of_squares_l2274_227493

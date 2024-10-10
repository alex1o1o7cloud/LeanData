import Mathlib

namespace min_value_of_expression_l1978_197873

theorem min_value_of_expression (a b c : ℤ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  4 * (a^2 + b^2 + c^2) - (a + b + c)^2 ≥ 8 :=
sorry

end min_value_of_expression_l1978_197873


namespace evaluate_expression_l1978_197899

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end evaluate_expression_l1978_197899


namespace nancy_sweaters_count_l1978_197856

/-- Represents the washing machine capacity -/
def machine_capacity : ℕ := 9

/-- Represents the number of shirts Nancy had to wash -/
def number_of_shirts : ℕ := 19

/-- Represents the total number of loads Nancy did -/
def total_loads : ℕ := 3

/-- Calculates the number of sweaters Nancy had to wash -/
def number_of_sweaters : ℕ := machine_capacity

theorem nancy_sweaters_count :
  number_of_sweaters = machine_capacity := by sorry

end nancy_sweaters_count_l1978_197856


namespace percentage_problem_l1978_197898

theorem percentage_problem (P : ℝ) : 
  (P / 100) * 24 + 0.1 * 40 = 5.92 ↔ P = 8 := by
sorry

end percentage_problem_l1978_197898


namespace intersection_point_satisfies_equations_intersection_point_unique_l1978_197829

/-- The intersection point of two lines -/
def intersection_point : ℝ × ℝ := (2, 3)

/-- First line equation -/
def line1 (x y : ℝ) : Prop := 9 * x - 4 * y = 6

/-- Second line equation -/
def line2 (x y : ℝ) : Prop := 7 * x + y = 17

/-- Theorem stating that the intersection_point satisfies both line equations -/
theorem intersection_point_satisfies_equations :
  let (x, y) := intersection_point
  line1 x y ∧ line2 x y :=
by sorry

/-- Theorem stating that the intersection_point is the unique solution -/
theorem intersection_point_unique :
  ∀ (x y : ℝ), line1 x y ∧ line2 x y → (x, y) = intersection_point :=
by sorry

end intersection_point_satisfies_equations_intersection_point_unique_l1978_197829


namespace min_value_of_expression_l1978_197849

theorem min_value_of_expression (x : ℝ) (h : x > 2) : 
  4 / (x - 2) + 4 * x ≥ 16 ∧ ∃ y > 2, 4 / (y - 2) + 4 * y = 16 :=
by sorry

end min_value_of_expression_l1978_197849


namespace value_of_x_l1978_197811

theorem value_of_x (x y z : ℝ) 
  (h1 : x = (1/2) * y) 
  (h2 : y = (1/4) * z) 
  (h3 : z = 100) : 
  x = 12.5 := by
sorry

end value_of_x_l1978_197811


namespace problem_1_l1978_197812

theorem problem_1 (a b c : ℝ) (h : a * c + b * c + c^2 < 0) : b^2 > 4 * a * c := by
  sorry

end problem_1_l1978_197812


namespace largest_initial_number_l1978_197861

theorem largest_initial_number :
  ∃ (a b c d e : ℕ), 
    189 + a + b + c + d + e = 200 ∧
    a ≥ 2 ∧ b ≥ 2 ∧ c ≥ 2 ∧ d ≥ 2 ∧ e ≥ 2 ∧
    ¬(189 ∣ a) ∧ ¬(189 ∣ b) ∧ ¬(189 ∣ c) ∧ ¬(189 ∣ d) ∧ ¬(189 ∣ e) ∧
    ∀ (n : ℕ), n > 189 → 
      ¬∃ (x y z w v : ℕ),
        n + x + y + z + w + v = 200 ∧
        x ≥ 2 ∧ y ≥ 2 ∧ z ≥ 2 ∧ w ≥ 2 ∧ v ≥ 2 ∧
        ¬(n ∣ x) ∧ ¬(n ∣ y) ∧ ¬(n ∣ z) ∧ ¬(n ∣ w) ∧ ¬(n ∣ v) :=
by sorry

end largest_initial_number_l1978_197861


namespace cosine_range_in_triangle_l1978_197825

theorem cosine_range_in_triangle (A B C : Real) (h : 1 / Real.tan B + 1 / Real.tan C = 1 / Real.tan A) :
  2/3 ≤ Real.cos A ∧ Real.cos A < 1 := by
  sorry

end cosine_range_in_triangle_l1978_197825


namespace salary_increase_percentage_l1978_197821

def current_salary : ℝ := 300

theorem salary_increase_percentage : 
  (∃ (increase_percent : ℝ), 
    current_salary * (1 + 0.16) = 348 ∧ 
    current_salary * (1 + increase_percent / 100) = 330 ∧ 
    increase_percent = 10) :=
by sorry

end salary_increase_percentage_l1978_197821


namespace f_2018_is_zero_l1978_197835

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the properties of f
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def has_period_property (f : ℝ → ℝ) : Prop := ∀ x, f (x + 4) - f x = 2 * f 2

-- State the theorem
theorem f_2018_is_zero 
  (h_even : is_even f) 
  (h_period : has_period_property f) : 
  f 2018 = 0 := by sorry

end f_2018_is_zero_l1978_197835


namespace find_unknown_areas_l1978_197862

/-- Represents the areas of rectangles in a divided larger rectangle -/
structure RectangleAreas where
  area1 : ℝ
  area2 : ℝ
  area3 : ℝ
  area4 : ℝ
  area5 : ℝ
  a : ℝ
  b : ℝ

/-- Theorem stating the values of unknown areas a and b given other known areas -/
theorem find_unknown_areas (areas : RectangleAreas) 
  (h1 : areas.area1 = 25)
  (h2 : areas.area2 = 104)
  (h3 : areas.area3 = 40)
  (h4 : areas.area4 = 143)
  (h5 : areas.area5 = 66)
  (h6 : areas.area2 / areas.area3 = areas.area4 / areas.b)
  (h7 : areas.area1 / areas.a = areas.b / areas.area5) :
  areas.a = 30 ∧ areas.b = 55 := by
  sorry

end find_unknown_areas_l1978_197862


namespace number_puzzle_l1978_197842

theorem number_puzzle (x : ℝ) : 9 * (((x + 1.4) / 3) - 0.7) = 5.4 → x = 2.5 := by
  sorry

end number_puzzle_l1978_197842


namespace paper_clip_cost_l1978_197867

/-- The cost of one box of paper clips and one package of index cards satisfying given conditions -/
def paper_clip_and_index_card_cost (p i : ℝ) : Prop :=
  15 * p + 7 * i = 55.40 ∧ 12 * p + 10 * i = 61.70

/-- The theorem stating that the cost of one box of paper clips is 1.835 -/
theorem paper_clip_cost : ∃ (p i : ℝ), paper_clip_and_index_card_cost p i ∧ p = 1.835 := by
  sorry

end paper_clip_cost_l1978_197867


namespace rug_overlap_problem_l1978_197877

theorem rug_overlap_problem (total_rug_area : ℝ) (covered_floor_area : ℝ) (two_layer_area : ℝ)
  (h1 : total_rug_area = 200)
  (h2 : covered_floor_area = 140)
  (h3 : two_layer_area = 24) :
  ∃ (three_layer_area : ℝ),
    three_layer_area = 18 ∧
    total_rug_area - covered_floor_area = two_layer_area + 2 * three_layer_area :=
by sorry

end rug_overlap_problem_l1978_197877


namespace pathway_area_is_196_l1978_197886

/-- Represents the farm layout --/
structure FarmLayout where
  rows : Nat
  columns : Nat
  bed_width : Nat
  bed_height : Nat
  pathway_width : Nat

/-- Calculates the total area of pathways in the farm --/
def pathway_area (farm : FarmLayout) : Nat :=
  let total_width := farm.columns * farm.bed_width + (farm.columns + 1) * farm.pathway_width
  let total_height := farm.rows * farm.bed_height + (farm.rows + 1) * farm.pathway_width
  let total_area := total_width * total_height
  let beds_area := farm.rows * farm.columns * farm.bed_width * farm.bed_height
  total_area - beds_area

/-- Theorem stating that the pathway area for the given farm layout is 196 square feet --/
theorem pathway_area_is_196 (farm : FarmLayout) 
    (h1 : farm.rows = 4)
    (h2 : farm.columns = 3)
    (h3 : farm.bed_width = 4)
    (h4 : farm.bed_height = 3)
    (h5 : farm.pathway_width = 2) : 
  pathway_area farm = 196 := by
  sorry

end pathway_area_is_196_l1978_197886


namespace sum_of_three_numbers_l1978_197819

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 560) 
  (h2 : a*b + b*c + c*a = 8) : 
  a + b + c = 24 := by
sorry

end sum_of_three_numbers_l1978_197819


namespace insect_growth_theorem_l1978_197888

/-- An insect that doubles in size daily -/
structure GrowingInsect where
  initialSize : ℝ
  daysToReach10cm : ℕ

/-- The number of days it takes for the insect to reach 2.5 cm -/
def daysToReach2_5cm (insect : GrowingInsect) : ℕ :=
  sorry

theorem insect_growth_theorem (insect : GrowingInsect) 
  (h1 : insect.daysToReach10cm = 10) 
  (h2 : 2 ^ insect.daysToReach10cm * insect.initialSize = 10) :
  daysToReach2_5cm insect = 8 :=
sorry

end insect_growth_theorem_l1978_197888


namespace thor_fraction_is_two_ninths_l1978_197860

-- Define the friends
inductive Friend
| Moe
| Loki
| Nick
| Thor
| Ott

-- Define the function that returns the fraction of money given by each friend
def fractionGiven (f : Friend) : ℚ :=
  match f with
  | Friend.Moe => 1/6
  | Friend.Loki => 1/5
  | Friend.Nick => 1/4
  | Friend.Ott => 1/3
  | Friend.Thor => 0

-- Define the amount of money given by each friend
def amountGiven : ℚ := 2

-- Define the total money of the group
def totalMoney : ℚ := (amountGiven / fractionGiven Friend.Moe) +
                      (amountGiven / fractionGiven Friend.Loki) +
                      (amountGiven / fractionGiven Friend.Nick) +
                      (amountGiven / fractionGiven Friend.Ott)

-- Define Thor's share
def thorShare : ℚ := 4 * amountGiven

-- Theorem to prove
theorem thor_fraction_is_two_ninths :
  thorShare / totalMoney = 2/9 := by
  sorry

end thor_fraction_is_two_ninths_l1978_197860


namespace cost_calculation_l1978_197800

/-- The cost of mangos per kg -/
def mango_cost : ℝ := sorry

/-- The cost of rice per kg -/
def rice_cost : ℝ := sorry

/-- The cost of flour per kg -/
def flour_cost : ℝ := 21

/-- The total cost of 4 kg of mangos, 3 kg of rice, and 5 kg of flour -/
def total_cost : ℝ := 4 * mango_cost + 3 * rice_cost + 5 * flour_cost

theorem cost_calculation :
  (10 * mango_cost = 24 * rice_cost) →
  (6 * flour_cost = 2 * rice_cost) →
  total_cost = 898.8 := by sorry

end cost_calculation_l1978_197800


namespace four_square_base_boxes_l1978_197832

/-- A box with a square base that can contain exactly 64 unit cubes. -/
structure SquareBaseBox where
  base : ℕ
  height : ℕ
  volume_eq_64 : base * base * height = 64

/-- The set of all possible SquareBaseBox configurations. -/
def all_square_base_boxes : Set SquareBaseBox :=
  { box | box.base * box.base * box.height = 64 }

/-- The theorem stating that there are exactly four possible SquareBaseBox configurations. -/
theorem four_square_base_boxes :
  all_square_base_boxes = {
    ⟨1, 64, rfl⟩,
    ⟨2, 16, rfl⟩,
    ⟨4, 4, rfl⟩,
    ⟨8, 1, rfl⟩
  } := by sorry

end four_square_base_boxes_l1978_197832


namespace all_terms_are_squares_l1978_197839

/-- Definition of the n-th term of the sequence -/
def sequence_term (n : ℕ) : ℕ :=
  10^(2*n + 1) + 5 * (10^n - 1) * 10^n + 6

/-- Theorem stating that all terms in the sequence are perfect squares -/
theorem all_terms_are_squares :
  ∀ n : ℕ, ∃ k : ℕ, sequence_term n = k^2 :=
by sorry

end all_terms_are_squares_l1978_197839


namespace quadratic_vertex_form_equivalence_l1978_197816

/-- The quadratic function in standard form -/
def f (x : ℝ) : ℝ := x^2 - 2*x + 3

/-- The quadratic function in vertex form -/
def g (x : ℝ) : ℝ := (x - 1)^2 + 2

/-- Theorem stating that f and g are equivalent -/
theorem quadratic_vertex_form_equivalence :
  ∀ x : ℝ, f x = g x := by sorry

end quadratic_vertex_form_equivalence_l1978_197816


namespace intersection_range_l1978_197875

/-- The function f(x) = x^3 - 3x - 1 -/
def f (x : ℝ) : ℝ := x^3 - 3*x - 1

/-- Predicate to check if a line y = m intersects f at three distinct points -/
def has_three_distinct_intersections (m : ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ 
    f x₁ = m ∧ f x₂ = m ∧ f x₃ = m

/-- Theorem stating the range of m for which y = m intersects f at three distinct points -/
theorem intersection_range :
  ∀ m : ℝ, has_three_distinct_intersections m ↔ m > -3 ∧ m < 1 := by
  sorry

end intersection_range_l1978_197875


namespace storks_on_fence_storks_count_l1978_197892

theorem storks_on_fence (initial_birds : ℕ) (additional_birds : ℕ) (bird_stork_difference : ℕ) : ℕ :=
  let total_birds := initial_birds + additional_birds
  let storks := total_birds - bird_stork_difference
  storks

theorem storks_count : storks_on_fence 3 4 2 = 5 := by
  sorry

end storks_on_fence_storks_count_l1978_197892


namespace inverse_variation_problem_l1978_197874

/-- Given that y varies inversely as the square of x, and y = 15 when x = 5,
    prove that y = 375/9 when x = 3. -/
theorem inverse_variation_problem (y : ℝ → ℝ) (k : ℝ) :
  (∀ x : ℝ, x ≠ 0 → y x = k / (x^2)) →  -- y varies inversely as the square of x
  y 5 = 15 →                           -- y = 15 when x = 5
  y 3 = 375 / 9 :=                     -- y = 375/9 when x = 3
by
  sorry


end inverse_variation_problem_l1978_197874


namespace skittles_and_erasers_grouping_l1978_197878

theorem skittles_and_erasers_grouping :
  let skittles : ℕ := 4502
  let erasers : ℕ := 4276
  let total_items : ℕ := skittles + erasers
  let num_groups : ℕ := 154
  total_items / num_groups = 57 := by
  sorry

end skittles_and_erasers_grouping_l1978_197878


namespace negation_distribution_l1978_197897

theorem negation_distribution (x : ℝ) : -(3*x - 2) = -3*x + 2 := by sorry

end negation_distribution_l1978_197897


namespace no_simultaneous_integer_fractions_l1978_197805

theorem no_simultaneous_integer_fractions :
  ¬ ∃ (n : ℤ), (∃ (A B : ℤ), (n - 6) / 15 = A ∧ (n - 5) / 24 = B) := by
sorry

end no_simultaneous_integer_fractions_l1978_197805


namespace sum_of_odd_prime_divisors_of_90_l1978_197818

-- Define a function to check if a number is prime
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

-- Define a function to check if a number is a divisor of 90
def isDivisorOf90 (n : ℕ) : Prop :=
  90 % n = 0

-- Define a function to check if a number is odd
def isOdd (n : ℕ) : Prop :=
  n % 2 ≠ 0

-- Theorem statement
theorem sum_of_odd_prime_divisors_of_90 :
  ∃ (S : Finset ℕ), (∀ n ∈ S, isPrime n ∧ isOdd n ∧ isDivisorOf90 n) ∧ 
    (∀ n : ℕ, isPrime n → isOdd n → isDivisorOf90 n → n ∈ S) ∧
    (S.sum id = 8) :=
sorry

end sum_of_odd_prime_divisors_of_90_l1978_197818


namespace maggie_grandfather_subscriptions_l1978_197847

/-- Represents the number of magazine subscriptions Maggie sold to her grandfather. -/
def grandfather_subscriptions : ℕ := sorry

/-- The amount Maggie earns per subscription in dollars. -/
def earnings_per_subscription : ℕ := 5

/-- The number of subscriptions Maggie sold to her parents. -/
def parent_subscriptions : ℕ := 4

/-- The number of subscriptions Maggie sold to the next-door neighbor. -/
def neighbor_subscriptions : ℕ := 2

/-- The number of subscriptions Maggie sold to another neighbor. -/
def other_neighbor_subscriptions : ℕ := 2 * neighbor_subscriptions

/-- The total amount Maggie earned in dollars. -/
def total_earnings : ℕ := 55

/-- Theorem stating that Maggie sold 1 subscription to her grandfather. -/
theorem maggie_grandfather_subscriptions : grandfather_subscriptions = 1 := by
  sorry

end maggie_grandfather_subscriptions_l1978_197847


namespace infinite_geometric_series_first_term_l1978_197817

theorem infinite_geometric_series_first_term 
  (r : ℚ) (S : ℚ) (a : ℚ) 
  (h1 : r = -1/3) 
  (h2 : S = 9) 
  (h3 : S = a / (1 - r)) : 
  a = 12 := by sorry

end infinite_geometric_series_first_term_l1978_197817


namespace product_of_roots_l1978_197828

theorem product_of_roots (b : ℝ) : 
  (∃ x₁ x₂ : ℝ, 2020 * x₁^2 + b * x₁ + 2021 = 0 ∧ 2020 * x₂^2 + b * x₂ + 2021 = 0 ∧ x₁ ≠ x₂) →
  (∃ y₁ y₂ : ℝ, 2019 * y₁^2 + b * y₁ + 2020 = 0 ∧ 2019 * y₂^2 + b * y₂ + 2020 = 0 ∧ y₁ ≠ y₂) →
  (∃ z₁ z₂ : ℝ, z₁^2 + b * z₁ + 2019 = 0 ∧ z₂^2 + b * z₂ + 2019 = 0 ∧ z₁ ≠ z₂) →
  (2021 / 2020) * (2020 / 2019) * 2019 = 2021 :=
by sorry

end product_of_roots_l1978_197828


namespace empty_subset_of_disjoint_nonempty_l1978_197853

theorem empty_subset_of_disjoint_nonempty (A B : Set α) :
  A ≠ ∅ → A ∩ B = ∅ → ∅ ⊆ B := by sorry

end empty_subset_of_disjoint_nonempty_l1978_197853


namespace simplify_expression_1_simplify_expression_2_l1978_197841

-- Problem 1
theorem simplify_expression_1 (a : ℝ) : 2*a*(a-3) - a^2 = a^2 - 6*a := by
  sorry

-- Problem 2
theorem simplify_expression_2 (x : ℝ) : (x-1)*(x+2) - x*(x+1) = -2 := by
  sorry

end simplify_expression_1_simplify_expression_2_l1978_197841


namespace square_root_of_square_l1978_197803

theorem square_root_of_square (x : ℝ) (h : x = 36) : Real.sqrt (x^2) = |x| := by
  sorry

end square_root_of_square_l1978_197803


namespace daily_medicine_dose_l1978_197836

theorem daily_medicine_dose (total_medicine : ℝ) (daily_fraction : ℝ) :
  total_medicine = 426 →
  daily_fraction = 0.06 →
  total_medicine * daily_fraction = 25.56 := by
  sorry

end daily_medicine_dose_l1978_197836


namespace recipe_liquid_sum_l1978_197831

/-- Given the amounts of oil and water used in a recipe, 
    prove that the total amount of liquid is their sum. -/
theorem recipe_liquid_sum (oil water : ℝ) 
  (h_oil : oil = 0.17) 
  (h_water : water = 1.17) : 
  oil + water = 1.34 := by
  sorry

end recipe_liquid_sum_l1978_197831


namespace quadratic_roots_difference_l1978_197806

theorem quadratic_roots_difference (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁^2 - 4*a*x₁ + 5*a^2 - 6*a = 0 ∧ 
    x₂^2 - 4*a*x₂ + 5*a^2 - 6*a = 0 ∧
    |x₁ - x₂| = 6) → 
  a = 3 := by
sorry

end quadratic_roots_difference_l1978_197806


namespace eulers_formula_l1978_197807

/-- A convex polyhedron is a structure with faces, vertices, and edges. -/
structure ConvexPolyhedron where
  faces : ℕ
  vertices : ℕ
  edges : ℕ

/-- Euler's formula for convex polyhedra states that V + F - E = 2 -/
theorem eulers_formula (P : ConvexPolyhedron) : 
  P.vertices + P.faces - P.edges = 2 := by
  sorry

end eulers_formula_l1978_197807


namespace no_seven_digit_number_divisible_by_another_l1978_197869

def is_valid_number (n : ℕ) : Prop :=
  ∃ (d1 d2 d3 d4 d5 d6 d7 : ℕ),
    d1 ∈ ({1, 2, 3, 4, 5, 6, 7} : Set ℕ) ∧
    d2 ∈ ({1, 2, 3, 4, 5, 6, 7} : Set ℕ) ∧
    d3 ∈ ({1, 2, 3, 4, 5, 6, 7} : Set ℕ) ∧
    d4 ∈ ({1, 2, 3, 4, 5, 6, 7} : Set ℕ) ∧
    d5 ∈ ({1, 2, 3, 4, 5, 6, 7} : Set ℕ) ∧
    d6 ∈ ({1, 2, 3, 4, 5, 6, 7} : Set ℕ) ∧
    d7 ∈ ({1, 2, 3, 4, 5, 6, 7} : Set ℕ) ∧
    d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d1 ≠ d5 ∧ d1 ≠ d6 ∧ d1 ≠ d7 ∧
    d2 ≠ d3 ∧ d2 ≠ d4 ∧ d2 ≠ d5 ∧ d2 ≠ d6 ∧ d2 ≠ d7 ∧
    d3 ≠ d4 ∧ d3 ≠ d5 ∧ d3 ≠ d6 ∧ d3 ≠ d7 ∧
    d4 ≠ d5 ∧ d4 ≠ d6 ∧ d4 ≠ d7 ∧
    d5 ≠ d6 ∧ d5 ≠ d7 ∧
    d6 ≠ d7 ∧
    n = d1 * 1000000 + d2 * 100000 + d3 * 10000 + d4 * 1000 + d5 * 100 + d6 * 10 + d7

theorem no_seven_digit_number_divisible_by_another :
  ∀ a b : ℕ, is_valid_number a → is_valid_number b → a ≠ b → ¬(a % b = 0) := by
  sorry

end no_seven_digit_number_divisible_by_another_l1978_197869


namespace tank_filling_ratio_l1978_197813

theorem tank_filling_ratio (tank_capacity : ℝ) (inflow_rate : ℝ) (outflow_rate1 : ℝ) (outflow_rate2 : ℝ) (filling_time : ℝ) :
  tank_capacity = 1 →
  inflow_rate = 0.5 →
  outflow_rate1 = 0.25 →
  outflow_rate2 = 1/6 →
  filling_time = 6 →
  (tank_capacity - (inflow_rate - outflow_rate1 - outflow_rate2) * filling_time) / tank_capacity = 0.5 := by
  sorry

#check tank_filling_ratio

end tank_filling_ratio_l1978_197813


namespace liam_keeps_three_balloons_l1978_197815

/-- The number of balloons Liam keeps for himself when distributing
    balloons evenly among his friends. -/
def balloons_kept (total : ℕ) (friends : ℕ) : ℕ :=
  total % friends

theorem liam_keeps_three_balloons :
  balloons_kept 243 10 = 3 := by
  sorry

end liam_keeps_three_balloons_l1978_197815


namespace smallest_possible_value_l1978_197830

theorem smallest_possible_value (x : ℕ+) (m n : ℕ+) : 
  m = 60 →
  Nat.gcd m n = x + 5 →
  Nat.lcm m n = x * (x + 5)^2 →
  n ≥ 2000 :=
sorry

end smallest_possible_value_l1978_197830


namespace square_ratio_side_length_sum_l1978_197885

theorem square_ratio_side_length_sum (area_ratio : ℚ) : 
  area_ratio = 50 / 98 →
  ∃ (a b c : ℕ), 
    (a * Real.sqrt b / c : ℝ) = Real.sqrt (area_ratio) ∧ 
    a = 5 ∧ b = 1 ∧ c = 7 ∧
    a + b + c = 13 := by
  sorry

end square_ratio_side_length_sum_l1978_197885


namespace greatest_common_divisor_620_180_under_100_l1978_197876

theorem greatest_common_divisor_620_180_under_100 :
  ∃ (d : ℕ), d = Nat.gcd 620 180 ∧ d < 100 ∧ d ∣ 620 ∧ d ∣ 180 ∧
  ∀ (x : ℕ), x < 100 → x ∣ 620 → x ∣ 180 → x ≤ d :=
by
  -- The proof goes here
  sorry

end greatest_common_divisor_620_180_under_100_l1978_197876


namespace complex_magnitude_problem_l1978_197887

theorem complex_magnitude_problem (z : ℂ) : z = 1 + 2 * I + I ^ 3 → Complex.abs z = Real.sqrt 2 := by
  sorry

end complex_magnitude_problem_l1978_197887


namespace tangent_line_and_m_range_l1978_197823

noncomputable def f (x : ℝ) : ℝ := x * (Real.log x - 1) + Real.log x + 1

theorem tangent_line_and_m_range :
  (∀ x : ℝ, x > 0 → (x - f x - 1 = 0 → x = 1)) ∧
  (∀ m : ℝ, (∀ x : ℝ, x > 0 → x^2 + x * (m - (Real.log x + 1/x)) + 1 ≥ 0) ↔ m ≥ -1) := by
  sorry

end tangent_line_and_m_range_l1978_197823


namespace beaver_change_l1978_197834

theorem beaver_change (initial_beavers initial_chipmunks chipmunk_decrease total_animals : ℕ) :
  initial_beavers = 20 →
  initial_chipmunks = 40 →
  chipmunk_decrease = 10 →
  total_animals = 130 →
  (total_animals - (initial_beavers + initial_chipmunks)) - (initial_chipmunks - chipmunk_decrease) - initial_beavers = 20 := by
  sorry

end beaver_change_l1978_197834


namespace fraction_power_equality_l1978_197857

theorem fraction_power_equality : (72000 ^ 4) / (24000 ^ 4) = 81 := by
  sorry

end fraction_power_equality_l1978_197857


namespace square_decomposition_l1978_197883

theorem square_decomposition (a b c k : ℕ) (n : ℕ) (h1 : c^2 = n * a^2 + n * b^2) 
  (h2 : (5*k)^2 = (4*k)^2 + (3*k)^2) (h3 : n = k^2) (h4 : n = 9) : c = 15 :=
sorry

end square_decomposition_l1978_197883


namespace distribute_7_balls_3_boxes_l1978_197890

/-- The number of ways to distribute n indistinguishable objects into k distinguishable boxes -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- Theorem: There are 36 ways to distribute 7 indistinguishable balls into 3 distinguishable boxes -/
theorem distribute_7_balls_3_boxes : distribute 7 3 = 36 := by
  sorry

end distribute_7_balls_3_boxes_l1978_197890


namespace mothers_day_discount_l1978_197851

theorem mothers_day_discount (original_price : ℝ) : 
  (original_price * 0.9 * 0.96 = 108) → original_price = 125 := by
  sorry

end mothers_day_discount_l1978_197851


namespace complementary_angle_measure_l1978_197863

-- Define the angle
def angle : ℝ := 45

-- Define the relationship between supplementary and complementary angles
def supplementary_complementary_relation (supplementary complementary : ℝ) : Prop :=
  supplementary = 3 * complementary

-- Define the supplementary angle
def supplementary (a : ℝ) : ℝ := 180 - a

-- Define the complementary angle
def complementary (a : ℝ) : ℝ := 90 - a

-- Theorem statement
theorem complementary_angle_measure :
  supplementary_complementary_relation (supplementary angle) (complementary angle) →
  complementary angle = 45 := by
  sorry

end complementary_angle_measure_l1978_197863


namespace company_fund_problem_l1978_197843

theorem company_fund_problem (n : ℕ) (initial_fund : ℕ) : 
  initial_fund = 60 * n - 10 →
  initial_fund = 50 * n + 120 →
  initial_fund = 770 :=
by
  sorry

end company_fund_problem_l1978_197843


namespace janet_investment_l1978_197865

/-- Calculates the total investment amount given the conditions of Janet's investment -/
theorem janet_investment
  (rate1 rate2 : ℚ)
  (interest_total : ℚ)
  (investment_at_rate1 : ℚ)
  (h1 : rate1 = 1/10)
  (h2 : rate2 = 1/100)
  (h3 : interest_total = 1390)
  (h4 : investment_at_rate1 = 12000)
  (h5 : investment_at_rate1 * rate1 + (total - investment_at_rate1) * rate2 = interest_total) :
  ∃ (total : ℚ), total = 31000 := by
  sorry

end janet_investment_l1978_197865


namespace largest_three_digit_square_base_7_l1978_197882

/-- The largest integer whose square has exactly 3 digits in base 7 -/
def M : ℕ := 18

/-- Conversion of a natural number to its base 7 representation -/
def to_base_7 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
  aux n []

theorem largest_three_digit_square_base_7 :
  (M * M ≥ 7^2) ∧ 
  (M * M < 7^3) ∧ 
  (∀ n : ℕ, n > M → n * n ≥ 7^3) ∧
  (to_base_7 M = [2, 4]) :=
sorry

end largest_three_digit_square_base_7_l1978_197882


namespace sqrt_25_times_sqrt_25_l1978_197833

theorem sqrt_25_times_sqrt_25 : Real.sqrt (25 * Real.sqrt 25) = 5 * Real.sqrt 5 := by
  sorry

end sqrt_25_times_sqrt_25_l1978_197833


namespace assembly_line_theorem_l1978_197895

/-- Represents the number of tasks in the assembly line -/
def num_tasks : ℕ := 6

/-- Represents the number of freely arrangeable tasks -/
def num_free_tasks : ℕ := 5

/-- The number of ways to arrange the assembly line -/
def assembly_line_arrangements : ℕ := Nat.factorial num_free_tasks

/-- Theorem stating the number of ways to arrange the assembly line -/
theorem assembly_line_theorem : 
  assembly_line_arrangements = 120 := by sorry

end assembly_line_theorem_l1978_197895


namespace parallel_resistance_l1978_197808

theorem parallel_resistance (x y r : ℝ) : 
  x = 4 → y = 5 → (1 / r = 1 / x + 1 / y) → r = 20 / 9 := by
  sorry

end parallel_resistance_l1978_197808


namespace condition_iff_prime_l1978_197894

def satisfies_condition (n : ℕ) : Prop :=
  (n = 2) ∨ (n > 2 ∧ ∀ k : ℕ, 2 ≤ k → k < n → ¬(k ∣ n))

theorem condition_iff_prime (n : ℕ) : satisfies_condition n ↔ Nat.Prime n :=
  sorry

end condition_iff_prime_l1978_197894


namespace base_6_representation_of_1729_base_6_to_decimal_1729_l1978_197866

/-- Converts a natural number to its base-6 representation as a list of digits -/
def toBase6 (n : ℕ) : List ℕ :=
  if n < 6 then [n]
  else (n % 6) :: toBase6 (n / 6)

/-- Converts a list of base-6 digits to a natural number -/
def fromBase6 (digits : List ℕ) : ℕ :=
  digits.foldr (fun d acc => d + 6 * acc) 0

theorem base_6_representation_of_1729 :
  toBase6 1729 = [1, 0, 0, 0, 2, 1] :=
sorry

theorem base_6_to_decimal_1729 :
  fromBase6 [1, 0, 0, 0, 2, 1] = 1729 :=
sorry

end base_6_representation_of_1729_base_6_to_decimal_1729_l1978_197866


namespace range_of_m_l1978_197852

-- Define propositions p and q as functions of m
def p (m : ℝ) : Prop := ∃ x y : ℝ, x ≠ y ∧ x^2 - 2*x + m = 0 ∧ y^2 - 2*y + m = 0

def q (m : ℝ) : Prop := ∀ x y : ℝ, x < y → (m + 2)*x - 1 < (m + 2)*y - 1

-- Define the theorem
theorem range_of_m :
  ∀ m : ℝ, ((p m ∨ q m) ∧ ¬(p m ∧ q m)) → (m ≤ -2 ∨ m ≥ 1) :=
by sorry

end range_of_m_l1978_197852


namespace max_consecutive_good_proof_l1978_197846

/-- Sum of all positive divisors of n -/
def α (n : ℕ) : ℕ := sorry

/-- A number n is "good" if gcd(n, α(n)) = 1 -/
def is_good (n : ℕ) : Prop := Nat.gcd n (α n) = 1

/-- The maximum number of consecutive good numbers -/
def max_consecutive_good : ℕ := 5

theorem max_consecutive_good_proof :
  ∀ k : ℕ, k > max_consecutive_good →
    ∃ n : ℕ, n ≥ 2 ∧ ∃ i : Fin k, ¬is_good (n + i) :=
by sorry

end max_consecutive_good_proof_l1978_197846


namespace river_distance_l1978_197804

/-- The distance between two points on a river, given boat speeds and time difference -/
theorem river_distance (v_down v_up : ℝ) (time_diff : ℝ) (h1 : v_down = 20)
  (h2 : v_up = 15) (h3 : time_diff = 5) :
  ∃ d : ℝ, d = 300 ∧ d / v_up - d / v_down = time_diff :=
by sorry

end river_distance_l1978_197804


namespace function_inequality_l1978_197848

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
variable (h1 : Differentiable ℝ f)
variable (h2 : ∀ x, (2 - x) / (deriv^[2] f x) ≤ 0)

-- State the theorem
theorem function_inequality : f 1 + f 3 > 2 * f 2 := by sorry

end function_inequality_l1978_197848


namespace min_value_sum_l1978_197827

theorem min_value_sum (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 2 = Real.sqrt (a * b)) :
  ∀ x y, x > 0 → y > 0 → 2 = Real.sqrt (x * y) → a + 4 * b ≤ x + 4 * y :=
sorry

end min_value_sum_l1978_197827


namespace sine_sum_greater_cosine_sum_increasing_geometric_sequence_l1978_197884

-- Define an acute-angled triangle
structure AcuteTriangle where
  A : Real
  B : Real
  C : Real
  acute_A : 0 < A ∧ A < π / 2
  acute_B : 0 < B ∧ B < π / 2
  acute_C : 0 < C ∧ C < π / 2
  sum_to_pi : A + B + C = π

-- Define a geometric sequence
def GeometricSequence (a : ℕ → ℝ) := ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

-- Statement for proposition ③
theorem sine_sum_greater_cosine_sum (t : AcuteTriangle) :
  Real.sin t.A + Real.sin t.B + Real.sin t.C > Real.cos t.A + Real.cos t.B + Real.cos t.C :=
sorry

-- Statement for proposition ④
theorem increasing_geometric_sequence (a : ℕ → ℝ) :
  (GeometricSequence a ∧ (∃ q > 1, ∀ n : ℕ, a (n + 1) = q * a n)) →
  (∀ n : ℕ, a (n + 1) > a n) ∧
  ¬((∀ n : ℕ, a (n + 1) > a n) → (∃ q > 1, ∀ n : ℕ, a (n + 1) = q * a n)) :=
sorry

end sine_sum_greater_cosine_sum_increasing_geometric_sequence_l1978_197884


namespace cubic_integer_root_l1978_197896

theorem cubic_integer_root
  (a b c : ℚ)
  (h1 : ∀ x : ℝ, x^3 + a*x^2 + b*x + c = 0 ↔ (x = 3 - Real.sqrt 5 ∨ x = 3 + Real.sqrt 5 ∨ (∃ n : ℤ, x = n)))
  (h2 : ∃ x : ℝ, x^3 + a*x^2 + b*x + c = 0 ∧ x = 3 - Real.sqrt 5)
  (h3 : ∃ n : ℤ, (n : ℝ)^3 + a*(n : ℝ)^2 + b*(n : ℝ) + c = 0) :
  ∃ n : ℤ, n^3 + a*n^2 + b*n + c = 0 ∧ n = -6 :=
sorry

end cubic_integer_root_l1978_197896


namespace sum_of_parts_l1978_197859

theorem sum_of_parts (x y : ℝ) : 
  x + y = 52 → 
  y = 30.333333333333332 → 
  10 * x + 22 * y = 884 := by
sorry

end sum_of_parts_l1978_197859


namespace sons_age_l1978_197870

theorem sons_age (son_age man_age : ℕ) : 
  man_age = son_age + 24 →
  man_age + 2 = 2 * (son_age + 2) →
  son_age = 22 := by
sorry

end sons_age_l1978_197870


namespace star_associativity_l1978_197879

-- Define the universal set
variable {U : Type}

-- Define the * operation
def star (X Y : Set U) : Set U := (X ∩ Y)ᶜ

-- State the theorem
theorem star_associativity (X Y Z : Set U) : 
  star (star X Y) Z = (Xᶜ ∩ Yᶜ) ∪ Z := by sorry

end star_associativity_l1978_197879


namespace square_plus_inverse_square_l1978_197840

theorem square_plus_inverse_square (x : ℝ) (h : x^4 + 1/x^4 = 2) : x^2 + 1/x^2 = 2 := by
  sorry

end square_plus_inverse_square_l1978_197840


namespace marie_messages_l1978_197814

/-- The number of new messages Marie gets per day -/
def new_messages_per_day : ℕ := 6

/-- The initial number of unread messages -/
def initial_messages : ℕ := 98

/-- The number of messages Marie reads per day -/
def messages_read_per_day : ℕ := 20

/-- The number of days it takes Marie to read all messages -/
def days_to_read_all : ℕ := 7

theorem marie_messages :
  initial_messages + days_to_read_all * new_messages_per_day = 
  days_to_read_all * messages_read_per_day :=
by sorry

end marie_messages_l1978_197814


namespace task_completion_time_l1978_197854

/-- The number of days A takes to complete the task -/
def days_A : ℚ := 12

/-- The efficiency ratio of B compared to A -/
def efficiency_B : ℚ := 1.75

/-- The number of days B takes to complete the task -/
def days_B : ℚ := 48 / 7

theorem task_completion_time :
  days_B = days_A / efficiency_B := by sorry

end task_completion_time_l1978_197854


namespace marble_selection_combinations_l1978_197850

def total_marbles : ℕ := 15
def special_marbles : ℕ := 6
def marbles_to_choose : ℕ := 5
def special_marbles_to_choose : ℕ := 2

theorem marble_selection_combinations :
  (Nat.choose special_marbles special_marbles_to_choose) *
  (Nat.choose (total_marbles - special_marbles) (marbles_to_choose - special_marbles_to_choose)) = 1260 := by
  sorry

end marble_selection_combinations_l1978_197850


namespace nancy_wednesday_pots_l1978_197891

/-- The number of clay pots Nancy created on each day of the week --/
structure ClayPots where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ

/-- The conditions of Nancy's clay pot creation --/
def nancy_pots : ClayPots where
  monday := 12
  tuesday := 2 * 12
  wednesday := 50 - (12 + 2 * 12)

/-- Theorem stating that Nancy created 14 clay pots on Wednesday --/
theorem nancy_wednesday_pots : nancy_pots.wednesday = 14 := by
  sorry

#eval nancy_pots.wednesday

end nancy_wednesday_pots_l1978_197891


namespace power_mod_thirteen_l1978_197864

theorem power_mod_thirteen : 7^2000 % 13 = 1 := by
  sorry

end power_mod_thirteen_l1978_197864


namespace first_term_of_sequence_l1978_197872

/-- Given a sequence of points scored in a game where the second to sixth terms
    are 3, 5, 8, 12, and 17, and the differences between consecutive terms
    form an arithmetic sequence, prove that the first term of the sequence is 2. -/
theorem first_term_of_sequence (a : ℕ → ℕ) : 
  a 2 = 3 ∧ a 3 = 5 ∧ a 4 = 8 ∧ a 5 = 12 ∧ a 6 = 17 ∧ 
  (∃ d : ℕ, ∀ n : ℕ, n ≥ 2 → a (n+1) - a n = d + n - 2) →
  a 1 = 2 :=
by sorry

end first_term_of_sequence_l1978_197872


namespace largest_number_in_sampling_l1978_197844

/-- Represents a systematic sampling of students. -/
structure SystematicSampling where
  total_students : ℕ
  smallest_number : ℕ
  second_smallest : ℕ
  selected_count : ℕ
  common_difference : ℕ

/-- The largest number in a systematic sampling. -/
def largest_number (s : SystematicSampling) : ℕ :=
  s.smallest_number + (s.selected_count - 1) * s.common_difference

/-- Theorem stating the largest number in the given systematic sampling. -/
theorem largest_number_in_sampling :
  let s : SystematicSampling := {
    total_students := 80,
    smallest_number := 6,
    second_smallest := 14,
    selected_count := 10,
    common_difference := 8
  }
  largest_number s = 78 := by sorry

end largest_number_in_sampling_l1978_197844


namespace parabola_two_axis_intersections_l1978_197871

/-- A parabola has only two common points with the coordinate axes if and only if m is 0 or 8 --/
theorem parabola_two_axis_intersections (m : ℝ) : 
  (∃! x y : ℝ, (y = 2*x^2 + 8*x + m ∧ (x = 0 ∨ y = 0)) ∧ 
   (∃ x' y' : ℝ, (y' = 2*x'^2 + 8*x' + m ∧ (x' = 0 ∨ y' = 0)) ∧ (x ≠ x' ∨ y ≠ y'))) ↔ 
  (m = 0 ∨ m = 8) :=
sorry

end parabola_two_axis_intersections_l1978_197871


namespace sum_of_squares_ratio_l1978_197801

theorem sum_of_squares_ratio (a b c : ℚ) : 
  a + b + c = 14 → 
  b = 2 * a → 
  c = 3 * a → 
  a^2 + b^2 + c^2 = 686/9 := by
sorry

end sum_of_squares_ratio_l1978_197801


namespace alpha_beta_sum_l1978_197889

theorem alpha_beta_sum (α β : ℝ) : 
  (∀ x : ℝ, (x - α) / (x + β) = (x^2 - 80*x + 1551) / (x^2 + 57*x - 2970)) →
  α + β = 137 := by
sorry

end alpha_beta_sum_l1978_197889


namespace equation_solutions_l1978_197845

theorem equation_solutions : 
  let f : ℝ → ℝ := λ x => 1 / ((x - 2) * (x - 3)) + 1 / ((x - 3) * (x - 4)) + 1 / ((x - 4) * (x - 5))
  ∀ x : ℝ, f x = 1 / 12 ↔ x = 5 + Real.sqrt 19 ∨ x = 5 - Real.sqrt 19 :=
by sorry

end equation_solutions_l1978_197845


namespace next_term_is_2500x4_l1978_197838

def geometric_sequence (x : ℝ) : ℕ → ℝ
  | 0 => 4
  | 1 => 20 * x
  | 2 => 100 * x^2
  | 3 => 500 * x^3
  | (n + 4) => geometric_sequence x n * 5 * x

theorem next_term_is_2500x4 (x : ℝ) : geometric_sequence x 4 = 2500 * x^4 := by
  sorry

end next_term_is_2500x4_l1978_197838


namespace tan_x_2_implies_expression_half_l1978_197858

theorem tan_x_2_implies_expression_half (x : ℝ) (h : Real.tan x = 2) :
  (2 * Real.sin (Real.pi + x) * Real.cos (Real.pi - x) - Real.cos (Real.pi + x)) /
  (1 + Real.sin x ^ 2 + Real.sin (Real.pi - x) - Real.cos (Real.pi - x) ^ 2) = 1 / 2 := by
  sorry

end tan_x_2_implies_expression_half_l1978_197858


namespace parabola_focus_distance_l1978_197810

/-- Given a parabola y² = 2px (p > 0) with a point A(4, m) on it,
    if the distance from A to the focus is 17/4, then p = 1/2. -/
theorem parabola_focus_distance (p : ℝ) (m : ℝ) : 
  p > 0 → 
  m^2 = 2*p*4 → 
  (4 - p/2)^2 + m^2 = (17/4)^2 → 
  p = 1/2 := by
sorry

end parabola_focus_distance_l1978_197810


namespace conic_is_hyperbola_l1978_197809

/-- A conic section type -/
inductive ConicType
  | Circle
  | Parabola
  | Ellipse
  | Hyperbola
  | None

/-- The equation of the conic section -/
def conic_equation (x y : ℝ) : Prop :=
  (x - 3)^2 = (3*y + 4)^2 - 90

/-- Function to determine the type of conic section -/
def determine_conic_type (eq : (ℝ → ℝ → Prop)) : ConicType :=
  sorry

/-- Theorem stating that the given equation describes a hyperbola -/
theorem conic_is_hyperbola :
  determine_conic_type conic_equation = ConicType.Hyperbola :=
sorry

end conic_is_hyperbola_l1978_197809


namespace smallest_constant_degenerate_triangle_l1978_197893

/-- A degenerate triangle is represented by three non-negative real numbers a, b, and c,
    where a + b = c --/
structure DegenerateTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  non_neg_a : 0 ≤ a
  non_neg_b : 0 ≤ b
  non_neg_c : 0 ≤ c
  sum_eq_c : a + b = c

/-- The smallest constant N such that (a^2 + b^2) / c^2 < N for all degenerate triangles
    is 1/2 --/
theorem smallest_constant_degenerate_triangle :
  ∃ N : ℝ, (∀ t : DegenerateTriangle, (t.a^2 + t.b^2) / t.c^2 < N) ∧
  (∀ ε > 0, ∃ t : DegenerateTriangle, (t.a^2 + t.b^2) / t.c^2 > N - ε) ∧
  N = 1/2 := by
  sorry

end smallest_constant_degenerate_triangle_l1978_197893


namespace sqrt_real_implies_x_leq_two_l1978_197820

theorem sqrt_real_implies_x_leq_two (x : ℝ) : (∃ y : ℝ, y * y = 2 - x) → x ≤ 2 := by
  sorry

end sqrt_real_implies_x_leq_two_l1978_197820


namespace tangent_line_to_circle_l1978_197824

/-- The equation of the tangent line to the circle x^2 + y^2 = 5 at the point (2, 1) is 2x + y - 5 = 0 -/
theorem tangent_line_to_circle (x y : ℝ) : 
  (2 : ℝ)^2 + 1^2 = 5 →  -- Point (2, 1) lies on the circle
  (∀ (a b : ℝ), a^2 + b^2 = 5 → (2*a + b = 5 → a = 2 ∧ b = 1)) →  -- (2, 1) is the only point of intersection
  2*x + y - 5 = 0 ↔ (x - 2)*(2) + (y - 1)*(1) = 0  -- Equation of tangent line
  := by sorry

end tangent_line_to_circle_l1978_197824


namespace power_of_five_mod_ten_thousand_l1978_197881

theorem power_of_five_mod_ten_thousand :
  5^2023 ≡ 8125 [ZMOD 10000] := by sorry

end power_of_five_mod_ten_thousand_l1978_197881


namespace team_organization_theorem_l1978_197880

/-- The number of ways to organize a team of 13 members into a specific hierarchy -/
def team_organization_count : ℕ := 4804800

/-- The total number of team members -/
def total_members : ℕ := 13

/-- The number of project managers -/
def project_managers : ℕ := 3

/-- The number of subordinates per project manager -/
def subordinates_per_manager : ℕ := 3

/-- Theorem stating the correct number of ways to organize the team -/
theorem team_organization_theorem :
  team_organization_count = 
    total_members * 
    (Nat.choose (total_members - 1) project_managers) * 
    (Nat.choose (total_members - 1 - project_managers) subordinates_per_manager) * 
    (Nat.choose (total_members - 1 - project_managers - subordinates_per_manager) subordinates_per_manager) * 
    (Nat.choose (total_members - 1 - project_managers - 2 * subordinates_per_manager) subordinates_per_manager) :=
by
  sorry

#eval team_organization_count

end team_organization_theorem_l1978_197880


namespace two_places_distribution_three_places_distribution_ambulance_distribution_l1978_197868

/-- The number of volunteers --/
def num_volunteers : ℕ := 4

/-- The number of places --/
def num_places : ℕ := 3

/-- The number of ambulances --/
def num_ambulances : ℕ := 20

/-- The number of ways to distribute 4 volunteers to 2 places with 2 volunteers in each place --/
theorem two_places_distribution (n : ℕ) (h : n = num_volunteers) : 
  Nat.choose n 2 = 6 := by sorry

/-- The number of ways to distribute 4 volunteers to 3 places with at least one volunteer in each place --/
theorem three_places_distribution (n m : ℕ) (h1 : n = num_volunteers) (h2 : m = num_places) : 
  6 * Nat.factorial (m - 1) = 36 := by sorry

/-- The number of ways to distribute 20 identical ambulances to 3 places with at least one ambulance in each place --/
theorem ambulance_distribution (a m : ℕ) (h1 : a = num_ambulances) (h2 : m = num_places) : 
  Nat.choose (a - 1) (m - 1) = 171 := by sorry

end two_places_distribution_three_places_distribution_ambulance_distribution_l1978_197868


namespace nells_baseball_cards_l1978_197826

/-- Nell's baseball card collection problem -/
theorem nells_baseball_cards 
  (cards_given_to_jeff : ℕ) 
  (cards_left : ℕ) 
  (h1 : cards_given_to_jeff = 28) 
  (h2 : cards_left = 276) : 
  cards_given_to_jeff + cards_left = 304 := by
sorry

end nells_baseball_cards_l1978_197826


namespace trigonometric_identity_l1978_197837

theorem trigonometric_identity (α : ℝ) : 
  (Real.cos (2 * α))^4 - 6 * (Real.cos (2 * α))^2 * (Real.sin (2 * α))^2 + (Real.sin (2 * α))^4 = Real.cos (8 * α) := by
  sorry

end trigonometric_identity_l1978_197837


namespace lunch_combinations_l1978_197822

/-- The number of different types of meat dishes -/
def num_meat_dishes : ℕ := 4

/-- The number of different types of vegetable dishes -/
def num_veg_dishes : ℕ := 7

/-- The number of meat dishes chosen in the first combination method -/
def meat_choice_1 : ℕ := 2

/-- The number of vegetable dishes chosen in both combination methods -/
def veg_choice : ℕ := 2

/-- The number of meat dishes chosen in the second combination method -/
def meat_choice_2 : ℕ := 1

/-- The total number of lunch combinations -/
def total_combinations : ℕ := Nat.choose num_meat_dishes meat_choice_1 * Nat.choose num_veg_dishes veg_choice +
                               Nat.choose num_meat_dishes meat_choice_2 * Nat.choose num_veg_dishes veg_choice

theorem lunch_combinations : total_combinations = 210 := by
  sorry

end lunch_combinations_l1978_197822


namespace margin_relation_l1978_197802

theorem margin_relation (n : ℝ) (C S M : ℝ) 
  (h1 : M = (1/n) * C) 
  (h2 : S = C + M) : 
  M = (1/(n+1)) * S := by
sorry

end margin_relation_l1978_197802


namespace characterize_square_property_functions_l1978_197855

/-- A function f: ℕ → ℕ satisfies the square property if (f(m) + n)(m + f(n)) is a square for all m, n ∈ ℕ -/
def satisfies_square_property (f : ℕ → ℕ) : Prop :=
  ∀ m n : ℕ, ∃ k : ℕ, (f m + n) * (m + f n) = k * k

/-- The main theorem characterizing functions satisfying the square property -/
theorem characterize_square_property_functions :
  ∀ f : ℕ → ℕ, satisfies_square_property f ↔ ∃ c : ℕ, ∀ n : ℕ, f n = n + c :=
sorry

end characterize_square_property_functions_l1978_197855

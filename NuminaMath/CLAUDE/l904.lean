import Mathlib

namespace no_perfect_power_triple_l904_90404

theorem no_perfect_power_triple (n r : ℕ) (hn : n ≥ 1) (hr : r ≥ 2) :
  ¬∃ m : ℤ, (n : ℤ) * (n + 1) * (n + 2) = m ^ r :=
sorry

end no_perfect_power_triple_l904_90404


namespace determine_constant_b_l904_90432

theorem determine_constant_b (b c : ℝ) : 
  (∀ x, (3*x^2 - 4*x + 8/3)*(2*x^2 + b*x + c) = 6*x^4 - 17*x^3 + 21*x^2 - 16/3*x + 9/3) → 
  b = -3 := by
sorry

end determine_constant_b_l904_90432


namespace floor_sqrt_sum_eq_floor_sqrt_product_l904_90412

theorem floor_sqrt_sum_eq_floor_sqrt_product (n : ℕ+) :
  ⌊Real.sqrt n + Real.sqrt (n + 1)⌋ = ⌊Real.sqrt (4 * n + 2)⌋ := by
  sorry

end floor_sqrt_sum_eq_floor_sqrt_product_l904_90412


namespace immediate_boarding_probability_l904_90446

def train_departure_interval : ℝ := 15
def train_stop_duration : ℝ := 2

theorem immediate_boarding_probability :
  (train_stop_duration / train_departure_interval : ℝ) = 2 / 15 := by sorry

end immediate_boarding_probability_l904_90446


namespace picture_area_l904_90497

theorem picture_area (x y : ℤ) 
  (h1 : x > 1) 
  (h2 : y > 1) 
  (h3 : (2*x + 4)*(y + 2) - x*y = 56) : 
  x * y = 24 := by
  sorry

end picture_area_l904_90497


namespace security_code_combinations_l904_90431

theorem security_code_combinations : Nat.factorial 4 = 24 := by
  sorry

end security_code_combinations_l904_90431


namespace rectangle_area_diagonal_l904_90499

theorem rectangle_area_diagonal (length width diagonal : ℝ) (k : ℝ) : 
  length > 0 → width > 0 → diagonal > 0 →
  length / width = 3 / 2 → 
  diagonal^2 = length^2 + width^2 →
  k = 6 / 13 →
  length * width = k * diagonal^2 := by
sorry

end rectangle_area_diagonal_l904_90499


namespace non_adjacent_arrangement_count_l904_90483

/-- Represents the number of ways to arrange balls in a row -/
def arrangement_count : ℕ := 12

/-- Represents the number of white balls -/
def white_ball_count : ℕ := 1

/-- Represents the number of red balls -/
def red_ball_count : ℕ := 1

/-- Represents the number of yellow balls -/
def yellow_ball_count : ℕ := 3

/-- Theorem stating that the number of arrangements where white and red balls are not adjacent is 12 -/
theorem non_adjacent_arrangement_count :
  (white_ball_count = 1) →
  (red_ball_count = 1) →
  (yellow_ball_count = 3) →
  (arrangement_count = 12) := by
  sorry

#check non_adjacent_arrangement_count

end non_adjacent_arrangement_count_l904_90483


namespace extra_workers_for_road_project_l904_90403

/-- Represents the road construction project -/
structure RoadProject where
  totalLength : ℝ
  totalDays : ℕ
  initialWorkers : ℕ
  completedLength : ℝ
  completedDays : ℕ

/-- Calculates the number of extra workers needed to complete the project on time -/
def extraWorkersNeeded (project : RoadProject) : ℕ :=
  sorry

/-- Theorem stating that for the given project parameters, approximately 53 extra workers are needed -/
theorem extra_workers_for_road_project :
  let project : RoadProject := {
    totalLength := 15,
    totalDays := 300,
    initialWorkers := 35,
    completedLength := 2.5,
    completedDays := 100
  }
  ∃ n : ℕ, n ≥ 53 ∧ n ≤ 54 ∧ extraWorkersNeeded project = n :=
sorry

end extra_workers_for_road_project_l904_90403


namespace validSelectionsCount_l904_90471

/-- Represents the set of available colors --/
inductive Color
| Red
| Blue
| Yellow
| Green

/-- Represents a ball with a color and number --/
structure Ball where
  color : Color
  number : Fin 6

/-- The set of all balls --/
def allBalls : Finset Ball :=
  sorry

/-- Checks if three numbers are non-consecutive --/
def areNonConsecutive (n1 n2 n3 : Fin 6) : Prop :=
  sorry

/-- Checks if three balls have different colors --/
def haveDifferentColors (b1 b2 b3 : Ball) : Prop :=
  sorry

/-- The set of valid selections of 3 balls --/
def validSelections : Finset (Fin 24 × Fin 24 × Fin 24) :=
  sorry

theorem validSelectionsCount :
  Finset.card validSelections = 96 := by
  sorry

end validSelectionsCount_l904_90471


namespace right_triangle_sides_l904_90466

theorem right_triangle_sides (t k : ℝ) (ht : t = 84) (hk : k = 56) :
  ∃ (a b c : ℝ),
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a + b + c = k ∧
    (1 / 2) * a * b = t ∧
    c * c = a * a + b * b ∧
    (a = 7 ∧ b = 24 ∧ c = 25) ∨ (a = 24 ∧ b = 7 ∧ c = 25) :=
by sorry

end right_triangle_sides_l904_90466


namespace consecutive_integers_equality_l904_90460

theorem consecutive_integers_equality (n : ℕ) (h : n > 0) : 
  (n + (n+1) + (n+2) + (n+3) = (n+4) + (n+5) + (n+6)) ↔ n = 9 :=
sorry

end consecutive_integers_equality_l904_90460


namespace stratified_sampling_probability_l904_90469

/-- Represents the number of students in each year of high school. -/
structure SchoolPopulation where
  first_year : ℕ
  second_year : ℕ
  third_year : ℕ

/-- Represents the number of students selected from each year in the sample. -/
structure SampleSize where
  first_year : ℕ
  second_year : ℕ
  third_year : ℕ

/-- The probability of a student being selected in a stratified sampling survey. -/
def selectionProbability (population : SchoolPopulation) (sample : SampleSize) : ℚ :=
  sample.third_year / population.third_year

theorem stratified_sampling_probability
  (population : SchoolPopulation)
  (sample : SampleSize)
  (h1 : population.first_year = 800)
  (h2 : population.second_year = 600)
  (h3 : population.third_year = 500)
  (h4 : sample.third_year = 25) :
  selectionProbability population sample = 1 / 20 := by
  sorry

#check stratified_sampling_probability

end stratified_sampling_probability_l904_90469


namespace marys_age_l904_90485

theorem marys_age (mary_age rahul_age : ℕ) : 
  rahul_age = mary_age + 30 →
  rahul_age + 20 = 2 * (mary_age + 20) →
  mary_age = 10 := by
sorry

end marys_age_l904_90485


namespace expand_product_l904_90441

theorem expand_product (x : ℝ) : (x^2 + 3*x + 3) * (x^2 - 3*x + 3) = x^4 - 3*x^2 + 9 := by
  sorry

end expand_product_l904_90441


namespace unique_nonnegative_integer_solution_l904_90448

theorem unique_nonnegative_integer_solution :
  ∃! (x y z : ℕ), 5 * x + 7 * y + 5 * z = 37 ∧ 6 * x - y - 10 * z = 3 ∧ x = 4 ∧ y = 1 ∧ z = 2 := by
  sorry

end unique_nonnegative_integer_solution_l904_90448


namespace function_with_finitely_many_discontinuities_doesnt_satisfy_condition1_l904_90452

-- Define the function type
def RealFunction (a b : ℝ) := ℝ → ℝ

-- Define the property of having finitely many discontinuities
def HasFinitelyManyDiscontinuities (f : RealFunction a b) : Prop := sorry

-- Define condition (1) (we don't know what it is exactly, so we'll leave it abstract)
def SatisfiesCondition1 (f : RealFunction a b) : Prop := sorry

-- The main theorem
theorem function_with_finitely_many_discontinuities_doesnt_satisfy_condition1 
  {a b : ℝ} (f : RealFunction a b) 
  (h_finite : HasFinitelyManyDiscontinuities f) : 
  ¬(SatisfiesCondition1 f) := by
  sorry


end function_with_finitely_many_discontinuities_doesnt_satisfy_condition1_l904_90452


namespace x_value_l904_90410

/-- Given that 20% of x is 15 less than 15% of 1500, prove that x = 1050 -/
theorem x_value : ∃ x : ℝ, (0.2 * x = 0.15 * 1500 - 15) ∧ x = 1050 := by
  sorry

end x_value_l904_90410


namespace intersection_implies_a_value_subset_implies_a_range_l904_90424

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x : ℝ | -a < x ∧ x < a + 1}
def B (a : ℝ) : Set ℝ := {x : ℝ | 4 - x < a}

-- Part 1
theorem intersection_implies_a_value (a : ℝ) :
  A a ∩ B a = Set.Ioo 2 3 → a = 2 := by sorry

-- Part 2
theorem subset_implies_a_range (a : ℝ) :
  A a ⊆ (Set.univ \ B a) → a ≤ 3/2 := by sorry

end intersection_implies_a_value_subset_implies_a_range_l904_90424


namespace product_quotient_calculation_l904_90465

theorem product_quotient_calculation : 16 * 0.0625 / 4 * 0.5 * 2 = 1/4 := by
  sorry

end product_quotient_calculation_l904_90465


namespace measure_one_kg_l904_90430

theorem measure_one_kg (n : ℕ) (h : ¬ 3 ∣ n) : 
  ∃ (k : ℕ), n - 3 * k = 1 ∨ n - 3 * k = 2 :=
sorry

end measure_one_kg_l904_90430


namespace garys_to_harrys_book_ratio_l904_90438

/-- Proves that the ratio of Gary's books to Harry's books is 1:2 given the specified conditions -/
theorem garys_to_harrys_book_ratio :
  ∀ (harry_books flora_books gary_books : ℕ),
    harry_books = 50 →
    flora_books = 2 * harry_books →
    harry_books + flora_books + gary_books = 175 →
    gary_books = (1 : ℚ) / 2 * harry_books := by
  sorry

end garys_to_harrys_book_ratio_l904_90438


namespace room_length_proof_l904_90474

/-- Proves that the length of a rectangular room is 5.5 meters given specific conditions -/
theorem room_length_proof (width : ℝ) (total_cost : ℝ) (paving_rate : ℝ) :
  width = 3.75 →
  total_cost = 24750 →
  paving_rate = 1200 →
  (total_cost / paving_rate) / width = 5.5 := by
  sorry

end room_length_proof_l904_90474


namespace june_score_june_score_correct_l904_90470

theorem june_score (april_may_avg : ℕ) (april_may_june_avg : ℕ) : ℕ :=
  let april_may_total := april_may_avg * 2
  let april_may_june_total := april_may_june_avg * 3
  april_may_june_total - april_may_total

theorem june_score_correct :
  june_score 89 88 = 86 := by sorry

end june_score_june_score_correct_l904_90470


namespace cycle_price_proof_l904_90464

/-- Represents the original price of a cycle -/
def original_price : ℝ := 800

/-- Represents the selling price of the cycle -/
def selling_price : ℝ := 680

/-- Represents the loss percentage -/
def loss_percentage : ℝ := 15

theorem cycle_price_proof :
  selling_price = original_price * (1 - loss_percentage / 100) :=
by sorry

end cycle_price_proof_l904_90464


namespace sarah_candy_problem_l904_90425

/-- The number of candy pieces Sarah received from neighbors -/
def candy_from_neighbors : ℕ := 66

/-- The number of candy pieces Sarah ate per day -/
def candy_per_day : ℕ := 9

/-- The number of days Sarah's candy lasted -/
def days_candy_lasted : ℕ := 9

/-- The number of candy pieces Sarah received from her older sister -/
def candy_from_sister : ℕ := 15

theorem sarah_candy_problem :
  candy_from_sister = days_candy_lasted * candy_per_day - candy_from_neighbors :=
by sorry

end sarah_candy_problem_l904_90425


namespace hexagonal_pyramid_base_edge_l904_90491

/-- Represents a hexagonal pyramid -/
structure HexagonalPyramid where
  base_edge : ℝ
  side_edge : ℝ

/-- Calculates the sum of all edge lengths in a hexagonal pyramid -/
def total_edge_length (p : HexagonalPyramid) : ℝ :=
  6 * p.base_edge + 6 * p.side_edge

/-- Theorem stating the length of the base edge in a specific hexagonal pyramid -/
theorem hexagonal_pyramid_base_edge :
  ∃ (p : HexagonalPyramid),
    p.side_edge = 8 ∧
    total_edge_length p = 120 ∧
    p.base_edge = 12 := by
  sorry

end hexagonal_pyramid_base_edge_l904_90491


namespace inequality_proof_l904_90479

theorem inequality_proof (a : ℝ) (x : ℝ) (h1 : 0 ≤ a) (h2 : a ≤ 1/2) (h3 : x ≥ 0) :
  let f : ℝ → ℝ := λ y => Real.exp y
  let g : ℝ → ℝ := λ y => a * y + 1
  1 / f x + x / g x ≥ 1 := by
  sorry

end inequality_proof_l904_90479


namespace trig_identity_l904_90416

theorem trig_identity (α : Real) :
  (∃ P : Real × Real, P.1 = Real.sin 2 ∧ P.2 = Real.cos 2 ∧ 
    P.1^2 + P.2^2 = 1 ∧ Real.sin α = P.2) →
  Real.sqrt (2 * (1 - Real.sin α)) = 2 * Real.sin 1 := by
  sorry

end trig_identity_l904_90416


namespace sam_initial_money_l904_90411

/-- The amount of money Sam had initially -/
def initial_money (num_books : ℕ) (cost_per_book : ℕ) (money_left : ℕ) : ℕ :=
  num_books * cost_per_book + money_left

/-- Theorem stating that Sam's initial money was 79 dollars -/
theorem sam_initial_money :
  initial_money 9 7 16 = 79 := by
  sorry

end sam_initial_money_l904_90411


namespace c_profit_is_3600_l904_90406

def initial_home_value : ℝ := 20000
def profit_percentage : ℝ := 0.20
def loss_percentage : ℝ := 0.15

def sale_price : ℝ := initial_home_value * (1 + profit_percentage)
def repurchase_price : ℝ := sale_price * (1 - loss_percentage)

theorem c_profit_is_3600 : sale_price - repurchase_price = 3600 := by sorry

end c_profit_is_3600_l904_90406


namespace negate_sum_diff_l904_90443

theorem negate_sum_diff (a b c : ℝ) : -(a - b + c) = -a + b - c := by sorry

end negate_sum_diff_l904_90443


namespace emily_bought_seven_songs_l904_90414

/-- The number of songs Emily bought later -/
def songs_bought_later (initial_songs total_songs : ℕ) : ℕ :=
  total_songs - initial_songs

/-- Proof that Emily bought 7 songs later -/
theorem emily_bought_seven_songs :
  let initial_songs := 6
  let total_songs := 13
  songs_bought_later initial_songs total_songs = 7 := by
  sorry

end emily_bought_seven_songs_l904_90414


namespace eight_stairs_climbs_l904_90467

-- Define the function for the number of ways to climb n stairs
def climbStairs (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 1
  | 2 => 2
  | 3 => 4
  | m + 4 => climbStairs m + climbStairs (m + 1) + climbStairs (m + 2) + climbStairs (m + 3)

-- Theorem statement
theorem eight_stairs_climbs : climbStairs 8 = 108 := by
  sorry


end eight_stairs_climbs_l904_90467


namespace segment_and_polygon_inequalities_l904_90421

/-- Segment with projections a and b on perpendicular lines has length c -/
structure Segment where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Polygon with projections a and b on coordinate axes has perimeter P -/
structure Polygon where
  a : ℝ
  b : ℝ
  P : ℝ

/-- Theorem about segment length and polygon perimeter -/
theorem segment_and_polygon_inequalities 
  (s : Segment) (p : Polygon) : 
  s.c ≥ (s.a + s.b) / Real.sqrt 2 ∧ 
  p.P ≥ Real.sqrt 2 * (p.a + p.b) := by
  sorry


end segment_and_polygon_inequalities_l904_90421


namespace polynomial_divisibility_l904_90473

/-- A polynomial of degree 4 with coefficients a, b, and c -/
def P (a b c : ℝ) (x : ℝ) : ℝ := x^4 + a*x^2 + b*x + c

/-- The condition for P to be divisible by (x-1)^3 -/
def isDivisibleBy (a b c : ℝ) : Prop :=
  ∃ q : ℝ → ℝ, ∀ x, P a b c x = (x - 1)^3 * q x

/-- The theorem stating the necessary and sufficient conditions for P to be divisible by (x-1)^3 -/
theorem polynomial_divisibility (a b c : ℝ) :
  isDivisibleBy a b c ↔ a = -6 ∧ b = 8 ∧ c = -3 := by
  sorry


end polynomial_divisibility_l904_90473


namespace soda_cost_l904_90459

/-- The cost of items in a fast food restaurant. -/
structure FastFoodCosts where
  burger : ℕ  -- Cost of a burger in cents
  soda : ℕ    -- Cost of a soda in cents

/-- Alice's purchase -/
def alicePurchase (c : FastFoodCosts) : ℕ := 3 * c.burger + 2 * c.soda

/-- Bob's purchase -/
def bobPurchase (c : FastFoodCosts) : ℕ := 2 * c.burger + 4 * c.soda

/-- The theorem stating the cost of a soda given the purchase information -/
theorem soda_cost :
  ∃ (c : FastFoodCosts),
    alicePurchase c = 360 ∧
    bobPurchase c = 480 ∧
    c.soda = 90 := by
  sorry

end soda_cost_l904_90459


namespace hyperbola_axis_relation_l904_90494

-- Define the hyperbola equation
def hyperbola_equation (x y b : ℝ) : Prop := x^2 - y^2 / b^2 = 1

-- Define the length of the conjugate axis
def conjugate_axis_length (b : ℝ) : ℝ := 2 * b

-- Define the length of the transverse axis
def transverse_axis_length : ℝ := 2

-- State the theorem
theorem hyperbola_axis_relation (b : ℝ) :
  b > 0 →
  hyperbola_equation x y b →
  conjugate_axis_length b = 2 * transverse_axis_length →
  b = 2 := by
  sorry

end hyperbola_axis_relation_l904_90494


namespace base7_521_equals_260_l904_90402

/-- Converts a base-7 number to base 10 --/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- The problem statement --/
theorem base7_521_equals_260 :
  base7ToBase10 [1, 2, 5] = 260 := by
  sorry

end base7_521_equals_260_l904_90402


namespace mariela_cards_l904_90440

/-- The total number of get well cards Mariela received -/
def total_cards (hospital_cards : ℕ) (home_cards : ℕ) : ℕ :=
  hospital_cards + home_cards

/-- Theorem stating the total number of cards Mariela received -/
theorem mariela_cards : 
  total_cards 403 287 = 690 := by
  sorry

end mariela_cards_l904_90440


namespace cash_realized_proof_l904_90445

/-- Given an amount before brokerage and a brokerage rate, calculates the cash realized after brokerage. -/
def cash_realized (amount_before_brokerage : ℚ) (brokerage_rate : ℚ) : ℚ :=
  amount_before_brokerage - (amount_before_brokerage * brokerage_rate)

/-- Theorem stating that for the given conditions, the cash realized is 104.7375 -/
theorem cash_realized_proof :
  let amount_before_brokerage : ℚ := 105
  let brokerage_rate : ℚ := 1 / 400
  cash_realized amount_before_brokerage brokerage_rate = 104.7375 := by
  sorry

#eval cash_realized 105 (1/400)

end cash_realized_proof_l904_90445


namespace tangent_line_equations_l904_90450

/-- Given a cubic curve and a point, prove the equations of tangent lines passing through the point. -/
theorem tangent_line_equations (x y : ℝ → ℝ) (P : ℝ × ℝ) : 
  (∀ t, y t = (1/3) * (x t)^3 + 4/3) →  -- Curve equation
  P = (2, 4) →  -- Point P
  ∃ (A B : ℝ), 
    ((4 * x A - y A - 4 = 0) ∨ (x B - y B + 2 = 0)) ∧ 
    (∀ t, (4 * t - y A - 4 = 0) → (x A, y A) = (2, 4)) ∧
    (∀ t, (t - y B + 2 = 0) → (x B, y B) = (2, 4)) :=
by sorry

end tangent_line_equations_l904_90450


namespace arc_length_sector_l904_90472

/-- The length of an arc in a sector with radius 3 and central angle 120° is 2π -/
theorem arc_length_sector (r : ℝ) (θ : ℝ) : 
  r = 3 → θ = 120 → 2 * π * r * (θ / 360) = 2 * π := by sorry

end arc_length_sector_l904_90472


namespace present_age_of_B_l904_90429

/-- Given two people A and B, proves that B's current age is 70 years -/
theorem present_age_of_B (A B : ℕ) : 
  (A + 20 = 2 * (B - 20)) →  -- In 20 years, A will be twice as old as B was 20 years ago
  (A = B + 10) →             -- A is now 10 years older than B
  B = 70 :=                  -- B's current age is 70 years
by
  sorry

end present_age_of_B_l904_90429


namespace substitution_result_l904_90487

theorem substitution_result (x y : ℝ) :
  y = 2 * x + 1 ∧ 5 * x - 2 * y = 7 →
  5 * x - 4 * x - 2 = 7 :=
by
  sorry

end substitution_result_l904_90487


namespace expression_evaluation_l904_90480

theorem expression_evaluation (x y : ℝ) 
  (h : (x - 1)^2 + |y + 2| = 0) : 
  (3/2) * x^2 * y - (x^2 * y - 3 * (2 * x * y - x^2 * y) - x * y) = -9 := by
  sorry

end expression_evaluation_l904_90480


namespace rods_to_furlongs_l904_90420

/-- Conversion factor from furlongs to rods -/
def furlong_to_rods : ℕ := 50

/-- The number of rods we want to convert -/
def total_rods : ℕ := 1000

/-- The theorem states that 1000 rods is equal to 20 furlongs -/
theorem rods_to_furlongs : 
  (total_rods : ℚ) / furlong_to_rods = 20 := by sorry

end rods_to_furlongs_l904_90420


namespace p_cubed_plus_mp_l904_90454

theorem p_cubed_plus_mp (p m : ℤ) (h_p_odd : Odd p) : 
  Odd (p^3 + m*p) ↔ Even m := by
  sorry

end p_cubed_plus_mp_l904_90454


namespace inscribed_circles_diameter_l904_90419

/-- A sequence of circles inscribed in a parabola -/
def InscribedCircles (ω : ℕ → Set (ℝ × ℝ)) : Prop :=
  ∀ n : ℕ, 
    -- Each circle is inscribed in the parabola y = x²
    (∀ (x y : ℝ), (x, y) ∈ ω n → y = x^2) ∧
    -- Each circle is tangent to the next one
    (∃ (x y : ℝ), (x, y) ∈ ω n ∧ (x, y) ∈ ω (n + 1)) ∧
    -- The first circle has diameter 1 and touches the parabola at (0,0)
    (n = 1 → (0, 0) ∈ ω 1 ∧ ∃ (x y : ℝ), (x, y) ∈ ω 1 ∧ x^2 + y^2 = 1/4)

/-- The diameter of a circle -/
def Diameter (ω : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- Theorem: The diameter of the nth circle is 2n - 1 -/
theorem inscribed_circles_diameter 
  (ω : ℕ → Set (ℝ × ℝ)) 
  (h : InscribedCircles ω) :
  ∀ n : ℕ, n > 0 → Diameter (ω n) = 2 * n - 1 :=
sorry

end inscribed_circles_diameter_l904_90419


namespace horse_distribution_exists_l904_90422

/-- Represents the distribution of horses to sons -/
structure Distribution (b₁ b₂ b₃ : ℕ) :=
  (x₁₁ x₁₂ x₁₃ : ℕ)
  (x₂₁ x₂₂ x₂₃ : ℕ)
  (x₃₁ x₃₂ x₃₃ : ℕ)
  (sum_eq_b₁ : x₁₁ + x₂₁ + x₃₁ = b₁)
  (sum_eq_b₂ : x₁₂ + x₂₂ + x₃₂ = b₂)
  (sum_eq_b₃ : x₁₃ + x₂₃ + x₃₃ = b₃)

/-- Represents the value matrix for horses -/
def ValueMatrix := Matrix (Fin 3) (Fin 3) ℚ

/-- The theorem statement -/
theorem horse_distribution_exists :
  ∃ n : ℕ, ∀ b₁ b₂ b₃ : ℕ, ∀ A : ValueMatrix,
    (∀ i j : Fin 3, i ≠ j → A i i > A i j) →
    min b₁ (min b₂ b₃) > n →
    ∃ d : Distribution b₁ b₂ b₃,
      (A 0 0 * d.x₁₁ + A 0 1 * d.x₁₂ + A 0 2 * d.x₁₃ > A 0 0 * d.x₂₁ + A 0 1 * d.x₂₂ + A 0 2 * d.x₂₃) ∧
      (A 0 0 * d.x₁₁ + A 0 1 * d.x₁₂ + A 0 2 * d.x₁₃ > A 0 0 * d.x₃₁ + A 0 1 * d.x₃₂ + A 0 2 * d.x₃₃) ∧
      (A 1 0 * d.x₂₁ + A 1 1 * d.x₂₂ + A 1 2 * d.x₂₃ > A 1 0 * d.x₁₁ + A 1 1 * d.x₁₂ + A 1 2 * d.x₁₃) ∧
      (A 1 0 * d.x₂₁ + A 1 1 * d.x₂₂ + A 1 2 * d.x₂₃ > A 1 0 * d.x₃₁ + A 1 1 * d.x₃₂ + A 1 2 * d.x₃₃) ∧
      (A 2 0 * d.x₃₁ + A 2 1 * d.x₃₂ + A 2 2 * d.x₃₃ > A 2 0 * d.x₁₁ + A 2 1 * d.x₁₂ + A 2 2 * d.x₁₃) ∧
      (A 2 0 * d.x₃₁ + A 2 1 * d.x₃₂ + A 2 2 * d.x₃₃ > A 2 0 * d.x₂₁ + A 2 1 * d.x₂₂ + A 2 2 * d.x₂₃) :=
sorry

end horse_distribution_exists_l904_90422


namespace ellipse_k_range_l904_90427

theorem ellipse_k_range (k : ℝ) : 
  (∃ x y : ℝ, x^2 / (k - 4) + y^2 / (10 - k) = 1) ∧ 
  (∀ x y : ℝ, x^2 / (k - 4) + y^2 / (10 - k) = 1 → 
    ∃ a b c : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ∧ c^2 = a^2 - b^2 ∧ c ≠ 0) →
  k > 7 ∧ k < 10 :=
sorry

end ellipse_k_range_l904_90427


namespace units_digit_G_2000_l904_90489

/-- Definition of G_n -/
def G (n : ℕ) : ℕ := 2^(2^n) + 5^(5^n)

/-- Property of units digit for powers of 2 -/
axiom units_digit_power_2 (n : ℕ) : n % 4 = 0 → (2^(2^n)) % 10 = 6

/-- Property of units digit for powers of 5 -/
axiom units_digit_power_5 (n : ℕ) : (5^(5^n)) % 10 = 5

/-- Theorem: The units digit of G_2000 is 1 -/
theorem units_digit_G_2000 : G 2000 % 10 = 1 := by
  sorry

end units_digit_G_2000_l904_90489


namespace route2_faster_l904_90408

-- Define the probabilities and delay times for each route
def prob_green_A : ℚ := 1/2
def prob_green_B : ℚ := 2/3
def delay_A : ℕ := 2
def delay_B : ℕ := 3
def time_green_AB : ℕ := 20

def prob_green_a : ℚ := 3/4
def prob_green_b : ℚ := 2/5
def delay_a : ℕ := 8
def delay_b : ℕ := 5
def time_green_ab : ℕ := 15

-- Define the expected delay for each route
def expected_delay_route1 : ℚ := 
  (1 - prob_green_A) * delay_A + (1 - prob_green_B) * delay_B

def expected_delay_route2 : ℚ := 
  (1 - prob_green_a) * delay_a + (1 - prob_green_b) * delay_b

-- Define the expected travel time for each route
def expected_time_route1 : ℚ := time_green_AB + expected_delay_route1
def expected_time_route2 : ℚ := time_green_ab + expected_delay_route2

-- Theorem statement
theorem route2_faster : expected_time_route2 < expected_time_route1 :=
  sorry

end route2_faster_l904_90408


namespace max_subsets_with_intersection_condition_l904_90482

/-- Given a positive integer n ≥ 2, prove that the maximum number of mutually distinct subsets
    that can be selected from an n-element set, satisfying (Aᵢ ∩ Aₖ) ⊆ Aⱼ for all 1 ≤ i < j < k ≤ m,
    is 2n. -/
theorem max_subsets_with_intersection_condition (n : ℕ) (hn : n ≥ 2) :
  (∃ (m : ℕ) (S : Finset (Finset (Fin n))),
    (∀ A ∈ S, A ⊆ Finset.univ) ∧
    (Finset.card S = m) ∧
    (∀ (A B C : Finset (Fin n)), A ∈ S → B ∈ S → C ∈ S →
      (Finset.toList S).indexOf A < (Finset.toList S).indexOf B →
      (Finset.toList S).indexOf B < (Finset.toList S).indexOf C →
      (A ∩ C) ⊆ B) ∧
    (∀ (m' : ℕ) (S' : Finset (Finset (Fin n))),
      (∀ A ∈ S', A ⊆ Finset.univ) →
      (Finset.card S' = m') →
      (∀ (A B C : Finset (Fin n)), A ∈ S' → B ∈ S' → C ∈ S' →
        (Finset.toList S').indexOf A < (Finset.toList S').indexOf B →
        (Finset.toList S').indexOf B < (Finset.toList S').indexOf C →
        (A ∩ C) ⊆ B) →
      m' ≤ m)) ∧
  (m = 2 * n) :=
by sorry

end max_subsets_with_intersection_condition_l904_90482


namespace beef_weight_before_processing_l904_90484

/-- If a side of beef loses 50 percent of its weight in processing and weighs 750 pounds after processing, then it weighed 1500 pounds before processing. -/
theorem beef_weight_before_processing (weight_after : ℝ) (h1 : weight_after = 750) :
  ∃ weight_before : ℝ, weight_before * 0.5 = weight_after ∧ weight_before = 1500 := by
  sorry

end beef_weight_before_processing_l904_90484


namespace house_sale_profit_l904_90400

-- Define the initial home value
def initial_value : ℝ := 12000

-- Define the profit percentage for the first sale
def profit_percentage : ℝ := 0.20

-- Define the loss percentage for the second sale
def loss_percentage : ℝ := 0.15

-- Define the net profit
def net_profit : ℝ := 2160

-- Theorem statement
theorem house_sale_profit :
  let first_sale_price := initial_value * (1 + profit_percentage)
  let second_sale_price := first_sale_price * (1 - loss_percentage)
  first_sale_price - second_sale_price = net_profit := by
sorry

end house_sale_profit_l904_90400


namespace store_pricing_theorem_l904_90458

/-- Represents the cost of pencils and notebooks in a store -/
structure StorePricing where
  pencil_price : ℝ
  notebook_price : ℝ
  h1 : 9 * pencil_price + 5 * notebook_price = 3.45
  h2 : 6 * pencil_price + 4 * notebook_price = 2.40

/-- The cost of 18 pencils and 9 notebooks is $6.75 -/
theorem store_pricing_theorem (sp : StorePricing) :
  18 * sp.pencil_price + 9 * sp.notebook_price = 6.75 := by
  sorry


end store_pricing_theorem_l904_90458


namespace min_distance_complex_l904_90463

theorem min_distance_complex (z : ℂ) (h : Complex.abs (z - (1 + 2*I)) = 2) :
  ∃ (min_val : ℝ), min_val = 2*Real.sqrt 2 - 2 ∧
    ∀ (w : ℂ), Complex.abs (w - (1 + 2*I)) = 2 → Complex.abs (w - 3) ≥ min_val :=
by sorry

end min_distance_complex_l904_90463


namespace store_breaks_even_l904_90426

/-- Represents the financial outcome of selling two items -/
def break_even (selling_price : ℝ) (profit_percent : ℝ) (loss_percent : ℝ) : Prop :=
  let cost_price_1 := selling_price / (1 + profit_percent / 100)
  let cost_price_2 := selling_price / (1 - loss_percent / 100)
  cost_price_1 + cost_price_2 = selling_price * 2

/-- Theorem: A store breaks even when selling two items at $150 each, 
    with one making 50% profit and the other incurring 25% loss -/
theorem store_breaks_even : break_even 150 50 25 := by
  sorry

end store_breaks_even_l904_90426


namespace S_intersect_T_eq_T_l904_90453

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem S_intersect_T_eq_T : S ∩ T = T := by sorry

end S_intersect_T_eq_T_l904_90453


namespace equation_solution_l904_90433

theorem equation_solution : ∃ x : ℚ, (x - 30) / 3 = (4 - 3*x) / 7 ∧ x = 111/8 := by sorry

end equation_solution_l904_90433


namespace otimes_nested_equality_l904_90428

/-- The custom operation ⊗ -/
def otimes (x y : ℝ) : ℝ := x^3 + 3 - y

/-- Theorem stating that k ⊗ (k ⊗ (k ⊗ k)) = k^3 + 3 - k -/
theorem otimes_nested_equality (k : ℝ) : otimes k (otimes k (otimes k k)) = k^3 + 3 - k := by
  sorry

end otimes_nested_equality_l904_90428


namespace pages_needed_is_twelve_l904_90451

/-- Calculates the number of pages needed to organize sports cards -/
def pages_needed (new_baseball old_baseball new_basketball old_basketball new_football old_football cards_per_page : ℕ) : ℕ :=
  let total_baseball := new_baseball + old_baseball
  let total_basketball := new_basketball + old_basketball
  let total_football := new_football + old_football
  let baseball_pages := (total_baseball + cards_per_page - 1) / cards_per_page
  let basketball_pages := (total_basketball + cards_per_page - 1) / cards_per_page
  let football_pages := (total_football + cards_per_page - 1) / cards_per_page
  baseball_pages + basketball_pages + football_pages

/-- Theorem stating that the number of pages needed is 12 -/
theorem pages_needed_is_twelve :
  pages_needed 3 9 4 6 7 5 3 = 12 := by
  sorry

end pages_needed_is_twelve_l904_90451


namespace area_of_quadrilateral_l904_90435

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the properties of the quadrilateral
def quadrilateral_properties (ABCD : Quadrilateral) : Prop :=
  let (xa, ya) := ABCD.A
  let (xb, yb) := ABCD.B
  let (xc, yc) := ABCD.C
  let (xd, yd) := ABCD.D
  ∃ (angle_BCD : ℝ),
    angle_BCD = 120 ∧
    (xb - xa)^2 + (yb - ya)^2 = 13^2 ∧
    (xc - xb)^2 + (yc - yb)^2 = 6^2 ∧
    (xd - xc)^2 + (yd - yc)^2 = 5^2 ∧
    (xa - xd)^2 + (ya - yd)^2 = 12^2

-- Define the area of a triangle
def triangle_area (A B C : ℝ × ℝ) : ℝ := sorry

-- Define the area of quadrilateral ABCD
def quadrilateral_area (ABCD : Quadrilateral) : ℝ := sorry

-- Theorem statement
theorem area_of_quadrilateral (ABCD : Quadrilateral) :
  quadrilateral_properties ABCD →
  quadrilateral_area ABCD = (15 * Real.sqrt 3) / 2 + triangle_area ABCD.B ABCD.D ABCD.A :=
sorry

end area_of_quadrilateral_l904_90435


namespace reciprocal_of_2023_l904_90439

theorem reciprocal_of_2023 : 
  (∀ x : ℝ, x ≠ 0 → (1 / x) = x⁻¹) → 2023⁻¹ = (1 : ℝ) / 2023 := by
  sorry

end reciprocal_of_2023_l904_90439


namespace expression_order_l904_90418

theorem expression_order (a b : ℝ) (ha : a > 0) (hb : b > 0) (hne : a ≠ b) :
  (2 * a * b) / (a + b) < Real.sqrt (a * b) ∧
  Real.sqrt (a * b) < (a + b) / 2 ∧
  (a + b) / 2 < Real.sqrt ((a^2 + b^2) / 2) := by
  sorry

end expression_order_l904_90418


namespace ice_cream_picnic_tickets_l904_90481

theorem ice_cream_picnic_tickets (total_tickets : ℕ) (student_price non_student_price total_collected : ℚ) 
  (h1 : total_tickets = 193)
  (h2 : student_price = 1/2)
  (h3 : non_student_price = 3/2)
  (h4 : total_collected = 825/4) :
  ∃ (student_tickets : ℕ), 
    student_tickets ≤ total_tickets ∧ 
    (student_tickets : ℚ) * student_price + (total_tickets - student_tickets : ℚ) * non_student_price = total_collected ∧
    student_tickets = 83 := by
  sorry

end ice_cream_picnic_tickets_l904_90481


namespace opposite_of_ten_l904_90496

theorem opposite_of_ten : ∃ x : ℝ, (x + 10 = 0) ∧ (x = -10) := by
  sorry

end opposite_of_ten_l904_90496


namespace material_left_proof_l904_90447

theorem material_left_proof (material1 material2 used_material : ℚ) : 
  material1 = 5/11 →
  material2 = 2/3 →
  used_material = 2/3 →
  material1 + material2 - used_material = 5/11 := by
sorry

end material_left_proof_l904_90447


namespace function_composition_property_l904_90498

theorem function_composition_property (f : ℤ → ℤ) (m : ℕ+) :
  (∀ n : ℤ, (f^[m] n = n + 2017)) → (m = 1 ∨ m = 2017) := by
  sorry

#check function_composition_property

end function_composition_property_l904_90498


namespace expected_value_is_91_div_6_l904_90488

/-- The expected value of rolling a fair 6-sided die where the win is n^2 dollars for rolling n -/
def expected_value : ℚ :=
  (1 / 6 : ℚ) * (1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2)

/-- Theorem stating that the expected value is equal to 91/6 -/
theorem expected_value_is_91_div_6 : expected_value = 91 / 6 := by
  sorry

end expected_value_is_91_div_6_l904_90488


namespace sequence_median_l904_90434

def sequence_sum (n : ℕ) : ℕ := n * (n + 1) / 2

def median_position (n : ℕ) : ℕ := (sequence_sum n + 1) / 2

theorem sequence_median :
  ∃ (m : ℕ), m = 106 ∧
  sequence_sum (m - 1) < median_position 150 ∧
  median_position 150 ≤ sequence_sum m :=
sorry

end sequence_median_l904_90434


namespace angle_greater_if_sine_greater_l904_90476

theorem angle_greater_if_sine_greater (A B C : Real) (a b c : Real) :
  -- Define triangle ABC
  (A + B + C = Real.pi) →
  (a > 0) → (b > 0) → (c > 0) →
  -- Law of sines
  (a / Real.sin A = b / Real.sin B) →
  (b / Real.sin B = c / Real.sin C) →
  -- Given condition
  (Real.sin B > Real.sin C) →
  -- Conclusion
  B > C := by
  sorry


end angle_greater_if_sine_greater_l904_90476


namespace encyclopedia_interest_percentage_l904_90455

/-- Calculates the interest percentage given the conditions of an encyclopedia purchase --/
theorem encyclopedia_interest_percentage 
  (down_payment : ℚ)
  (total_cost : ℚ)
  (monthly_payment : ℚ)
  (num_monthly_payments : ℕ)
  (final_payment : ℚ)
  (h1 : down_payment = 300)
  (h2 : total_cost = 750)
  (h3 : monthly_payment = 57)
  (h4 : num_monthly_payments = 9)
  (h5 : final_payment = 21) :
  let total_paid := down_payment + (monthly_payment * num_monthly_payments) + final_payment
  let amount_borrowed := total_cost - down_payment
  let interest_paid := total_paid - total_cost
  interest_paid / amount_borrowed = 8533 / 10000 := by
sorry


end encyclopedia_interest_percentage_l904_90455


namespace coloring_books_remaining_l904_90436

theorem coloring_books_remaining (initial : Real) (first_giveaway : Real) (second_giveaway : Real) :
  initial = 48.0 →
  first_giveaway = 34.0 →
  second_giveaway = 3.0 →
  initial - first_giveaway - second_giveaway = 11.0 := by
  sorry

end coloring_books_remaining_l904_90436


namespace product_of_cube_and_square_l904_90442

theorem product_of_cube_and_square (x : ℝ) : 2 * x^3 * (-3 * x)^2 = 18 * x^5 := by
  sorry

end product_of_cube_and_square_l904_90442


namespace dans_to_barrys_dimes_ratio_l904_90462

/-- The ratio of Dan's initial dimes to Barry's dimes -/
theorem dans_to_barrys_dimes_ratio :
  let barry_dimes : ℕ := 1000 / 10
  let dan_final_dimes : ℕ := 52
  let dan_initial_dimes : ℕ := dan_final_dimes - 2
  (dan_initial_dimes : ℚ) / barry_dimes = 1 / 2 := by sorry

end dans_to_barrys_dimes_ratio_l904_90462


namespace matrix_product_AB_l904_90437

def A : Matrix (Fin 4) (Fin 3) ℝ := !![0, -1, 2; 2, 1, 1; 3, 0, 1; 3, 7, 1]
def B : Matrix (Fin 3) (Fin 2) ℝ := !![3, 1; 2, 1; 1, 0]

theorem matrix_product_AB :
  A * B = !![0, -1; 9, 3; 10, 3; 24, 10] := by sorry

end matrix_product_AB_l904_90437


namespace only_y_eq_0_is_equation_l904_90478

-- Define a type for the expressions
inductive Expression
  | Addition : Expression
  | Equation : Expression
  | Inequality : Expression
  | NotEqual : Expression

-- Define a function to check if an expression is an equation
def isEquation (e : Expression) : Prop :=
  match e with
  | Expression.Equation => True
  | _ => False

-- State the theorem
theorem only_y_eq_0_is_equation :
  let x_plus_1_5 := Expression.Addition
  let y_eq_0 := Expression.Equation
  let six_plus_x_lt_5 := Expression.Inequality
  let ab_neq_60 := Expression.NotEqual
  (¬ isEquation x_plus_1_5) ∧
  (isEquation y_eq_0) ∧
  (¬ isEquation six_plus_x_lt_5) ∧
  (¬ isEquation ab_neq_60) :=
by sorry


end only_y_eq_0_is_equation_l904_90478


namespace expression_value_approximation_l904_90477

def x : ℝ := 102
def y : ℝ := 98

theorem expression_value_approximation :
  let expr := (x^2 - y^2) / (x + y)^3 - (x^3 + y^3) * Real.log (x*y)
  ∃ ε > 0, |expr + 18446424.7199| < ε := by
  sorry

end expression_value_approximation_l904_90477


namespace wedge_volume_l904_90415

/-- The volume of a wedge cut from a cylindrical log -/
theorem wedge_volume (d : ℝ) (α : ℝ) : 
  d = 10 → α = 60 → (π * (d / 2)^2 * (d / 2 * Real.cos (α * π / 180))) = 125 * π := by
  sorry

end wedge_volume_l904_90415


namespace roses_per_day_l904_90461

theorem roses_per_day (total_roses : ℕ) (days : ℕ) (dozens_per_day : ℕ) 
  (h1 : total_roses = 168) 
  (h2 : days = 7) 
  (h3 : dozens_per_day * 12 * days = total_roses) : 
  dozens_per_day = 2 := by
  sorry

end roses_per_day_l904_90461


namespace dividend_calculation_l904_90493

theorem dividend_calculation (quotient : ℕ) (k : ℕ) (h1 : quotient = 4) (h2 : k = 14) :
  quotient * k = 56 := by
  sorry

end dividend_calculation_l904_90493


namespace february_to_january_ratio_l904_90492

-- Define the oil bills for January and February
def january_bill : ℚ := 120
def february_bill : ℚ := 180

-- Define the condition that February's bill is more than January's
axiom february_more_than_january : february_bill > january_bill

-- Define the condition about the 5:3 ratio if February's bill was $20 more
axiom ratio_condition : (february_bill + 20) / january_bill = 5 / 3

-- Theorem to prove
theorem february_to_january_ratio :
  february_bill / january_bill = 3 / 2 := by sorry

end february_to_january_ratio_l904_90492


namespace fraction_operations_l904_90401

theorem fraction_operations :
  let a := 2
  let b := 9
  let c := 5
  let d := 11
  (a / b) * (c / d) = 10 / 99 ∧ (a / b) + (c / d) = 67 / 99 := by
  sorry

end fraction_operations_l904_90401


namespace line_hyperbola_intersection_l904_90486

/-- A line y = kx + 2 intersects the left branch of x^2 - y^2 = 4 at two distinct points iff k ∈ (1, √2) -/
theorem line_hyperbola_intersection (k : ℝ) : 
  (∃ x₁ x₂ y₁ y₂ : ℝ, x₁ < x₂ ∧ x₁ < 0 ∧ x₂ < 0 ∧
    y₁ = k * x₁ + 2 ∧ y₂ = k * x₂ + 2 ∧
    x₁^2 - y₁^2 = 4 ∧ x₂^2 - y₂^2 = 4) ↔ 
  (1 < k ∧ k < Real.sqrt 2) :=
sorry

end line_hyperbola_intersection_l904_90486


namespace floor_ceiling_sum_l904_90468

theorem floor_ceiling_sum (x y : ℝ) (hx : 1 < x ∧ x < 2) (hy : 3 < y ∧ y < 4) :
  ⌊x⌋ + ⌈y⌉ = 5 := by
  sorry

end floor_ceiling_sum_l904_90468


namespace male_female_ratio_l904_90407

/-- Represents an association with male and female members selling raffle tickets -/
structure Association where
  male_members : ℕ
  female_members : ℕ
  total_tickets : ℕ
  male_tickets : ℕ
  female_tickets : ℕ

/-- The conditions given in the problem -/
def association_conditions (a : Association) : Prop :=
  (a.total_tickets : ℚ) / (a.male_members + a.female_members : ℚ) = 66 ∧
  (a.female_tickets : ℚ) / (a.female_members : ℚ) = 70 ∧
  (a.male_tickets : ℚ) / (a.male_members : ℚ) = 58 ∧
  a.total_tickets = a.male_tickets + a.female_tickets

/-- The theorem stating that under the given conditions, the male to female ratio is 1:2 -/
theorem male_female_ratio (a : Association) (h : association_conditions a) :
  (a.male_members : ℚ) / (a.female_members : ℚ) = 1 / 2 := by
  sorry

end male_female_ratio_l904_90407


namespace sum_inequality_l904_90417

theorem sum_inequality {a b c d : ℝ} (h1 : a > b) (h2 : c > d) : c + a > d + b := by
  sorry

end sum_inequality_l904_90417


namespace min_fraction_sum_l904_90475

def digits : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

def is_valid_selection (P Q R S : ℕ) : Prop :=
  P ∈ digits ∧ Q ∈ digits ∧ R ∈ digits ∧ S ∈ digits ∧ P < Q ∧ Q < R ∧ R < S

def fraction_sum (P Q R S : ℕ) : ℚ :=
  (P : ℚ) / (R : ℚ) + (Q : ℚ) / (S : ℚ)

theorem min_fraction_sum :
  ∃ (P Q R S : ℕ), is_valid_selection P Q R S ∧
    (∀ (P' Q' R' S' : ℕ), is_valid_selection P' Q' R' S' →
      fraction_sum P Q R S ≤ fraction_sum P' Q' R' S') ∧
    fraction_sum P Q R S = 25 / 72 := by
  sorry

end min_fraction_sum_l904_90475


namespace isosceles_triangle_side_length_l904_90449

/-- An isosceles triangle with specific properties -/
structure IsoscelesTriangle where
  -- The length of the base
  base : ℝ
  -- The length of the median drawn to one of the congruent sides
  median : ℝ
  -- The length of each congruent side
  side : ℝ
  -- Condition that the base is 4√2
  base_eq : base = 4 * Real.sqrt 2
  -- Condition that the median is 5
  median_eq : median = 5

/-- Theorem stating the length of the congruent sides in the specific isosceles triangle -/
theorem isosceles_triangle_side_length (t : IsoscelesTriangle) : t.side = Real.sqrt 34 := by
  sorry


end isosceles_triangle_side_length_l904_90449


namespace embroidery_time_l904_90409

-- Define the stitches per minute
def stitches_per_minute : ℕ := 4

-- Define the number of stitches for each design
def flower_stitches : ℕ := 60
def unicorn_stitches : ℕ := 180
def godzilla_stitches : ℕ := 800

-- Define the number of each design
def num_flowers : ℕ := 50
def num_unicorns : ℕ := 3
def num_godzilla : ℕ := 1

-- Theorem to prove
theorem embroidery_time :
  (num_flowers * flower_stitches + num_unicorns * unicorn_stitches + num_godzilla * godzilla_stitches) / stitches_per_minute = 1085 := by
  sorry

end embroidery_time_l904_90409


namespace jenny_run_distance_l904_90456

theorem jenny_run_distance (walked : Real) (ran_extra : Real) : 
  walked = 0.4 → ran_extra = 0.2 → walked + ran_extra = 0.6 := by
sorry

end jenny_run_distance_l904_90456


namespace root_implies_sum_l904_90490

/-- Given that 2 + i is a root of the polynomial x^4 + px^2 + qx + 1 = 0,
    where p and q are real numbers, prove that p + q = 4 -/
theorem root_implies_sum (p q : ℝ) 
  (h : (2 + Complex.I) ^ 4 + p * (2 + Complex.I) ^ 2 + q * (2 + Complex.I) + 1 = 0) : 
  p + q = 4 := by
  sorry

end root_implies_sum_l904_90490


namespace temperature_difference_is_eight_l904_90413

-- Define the temperatures
def highest_temp : ℤ := 5
def lowest_temp : ℤ := -3

-- Define the temperature difference
def temp_difference : ℤ := highest_temp - lowest_temp

-- Theorem to prove
theorem temperature_difference_is_eight :
  temp_difference = 8 :=
by sorry

end temperature_difference_is_eight_l904_90413


namespace negation_of_universal_statement_l904_90405

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ (∃ x : ℝ, x^3 - x^2 + 1 > 0) :=
by sorry

end negation_of_universal_statement_l904_90405


namespace root_exists_in_interval_l904_90444

-- Define the function f
def f (x : ℝ) : ℝ := -x^3 - 3*x + 5

-- State the theorem
theorem root_exists_in_interval :
  ∃ x₀ ∈ Set.Ioo 1 2, f x₀ = 0 :=
by
  -- Proof goes here
  sorry

end root_exists_in_interval_l904_90444


namespace semicircular_cubicle_perimeter_approx_l904_90457

/-- The perimeter of a semicircular cubicle with radius 14 units is approximately 71.96 units. -/
theorem semicircular_cubicle_perimeter_approx : ∃ (p : ℝ), 
  (abs (p - (28 + π * 14)) < 0.01) ∧ (abs (p - 71.96) < 0.01) := by
  sorry

end semicircular_cubicle_perimeter_approx_l904_90457


namespace part_one_part_two_l904_90423

-- Define the function f
def f (x m : ℝ) : ℝ := |x + m| + |2*x - 1|

-- Part I
theorem part_one :
  let m : ℝ := -1
  {x : ℝ | f x m ≤ 2} = {x : ℝ | 0 ≤ x ∧ x ≤ 4/3} := by sorry

-- Part II
theorem part_two :
  ∀ m : ℝ, (∀ x ∈ Set.Icc (3/4 : ℝ) 2, f x m ≤ |2*x + 1|) →
  -11/4 ≤ m ∧ m ≤ 0 := by sorry

end part_one_part_two_l904_90423


namespace festival_allowance_petty_cash_l904_90495

theorem festival_allowance_petty_cash (staff_count : ℕ) (days : ℕ) (daily_rate : ℕ) (total_given : ℕ) :
  staff_count = 20 →
  days = 30 →
  daily_rate = 100 →
  total_given = 65000 →
  total_given - (staff_count * days * daily_rate) = 5000 := by
sorry

end festival_allowance_petty_cash_l904_90495

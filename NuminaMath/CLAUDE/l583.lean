import Mathlib

namespace jalapeno_slices_per_pepper_l583_58373

/-- The number of jalapeno strips required per sandwich -/
def strips_per_sandwich : ℕ := 4

/-- The time in minutes between serving each sandwich -/
def minutes_per_sandwich : ℕ := 5

/-- The number of hours the shop operates per day -/
def operating_hours : ℕ := 8

/-- The number of jalapeno peppers required for a full day of operation -/
def peppers_per_day : ℕ := 48

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

theorem jalapeno_slices_per_pepper : 
  (operating_hours * minutes_per_hour / minutes_per_sandwich) * strips_per_sandwich / peppers_per_day = 8 := by
  sorry

end jalapeno_slices_per_pepper_l583_58373


namespace line_intersection_y_axis_l583_58389

/-- A line passing through two points (2, 9) and (4, 13) intersects the y-axis at (0, 5) -/
theorem line_intersection_y_axis :
  ∀ (f : ℝ → ℝ),
  (f 2 = 9) →
  (f 4 = 13) →
  (∀ x y, f x = y ↔ y = 2*x + 5) →
  f 0 = 5 := by
sorry

end line_intersection_y_axis_l583_58389


namespace kitchen_tiles_l583_58333

/-- The number of tiles needed to cover a rectangular floor -/
def tiles_needed (floor_length floor_width tile_area : ℕ) : ℕ :=
  (floor_length * floor_width) / tile_area

/-- Proof that 576 tiles are needed for the given floor and tile specifications -/
theorem kitchen_tiles :
  tiles_needed 48 72 6 = 576 := by
  sorry

end kitchen_tiles_l583_58333


namespace income_left_percentage_man_income_left_l583_58359

/-- Given a man's spending pattern, calculate the percentage of income left --/
theorem income_left_percentage (total_income : ℝ) (food_percent : ℝ) (education_percent : ℝ) 
  (transport_percent : ℝ) (rent_percent : ℝ) : ℝ :=
  let initial_expenses := food_percent + education_percent + transport_percent
  let remaining_after_initial := 100 - initial_expenses
  let rent_amount := rent_percent * remaining_after_initial / 100
  let total_expenses := initial_expenses + rent_amount
  100 - total_expenses

/-- Prove that the man is left with 12.6% of his income --/
theorem man_income_left :
  income_left_percentage 100 42 18 12 55 = 12.6 := by
  sorry

end income_left_percentage_man_income_left_l583_58359


namespace roberts_reading_l583_58300

/-- Given Robert's reading speed, book size, and available time, 
    prove the maximum number of complete books he can read. -/
theorem roberts_reading (
  reading_speed : ℕ) 
  (book_size : ℕ) 
  (available_time : ℕ) 
  (h1 : reading_speed = 120) 
  (h2 : book_size = 360) 
  (h3 : available_time = 8) : 
  (available_time * reading_speed) / book_size = 2 := by
  sorry

end roberts_reading_l583_58300


namespace dans_pokemon_cards_l583_58344

/-- The number of Pokemon cards Dan has -/
def dans_cards : ℕ := 41

/-- Sally's initial number of Pokemon cards -/
def sallys_initial_cards : ℕ := 27

/-- The number of Pokemon cards Sally bought -/
def cards_sally_bought : ℕ := 20

/-- The difference between Sally's and Dan's cards -/
def card_difference : ℕ := 6

theorem dans_pokemon_cards :
  sallys_initial_cards + cards_sally_bought = dans_cards + card_difference :=
sorry

end dans_pokemon_cards_l583_58344


namespace three_circles_collinearity_l583_58357

-- Define the circle structure
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the point structure
structure Point where
  x : ℝ
  y : ℝ

-- Define the line structure
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the theorem
theorem three_circles_collinearity 
  (circleA circleB circleC : Circle)
  (A B C : Point)
  (B₁ C₁ C₂ A₂ A₃ B₃ : Point)
  (X Y Z : Point) :
  -- Conditions
  circleA.center = (A.x, A.y) →
  circleB.center = (B.x, B.y) →
  circleC.center = (C.x, C.y) →
  circleA.radius = circleB.radius →
  circleB.radius = circleC.radius →
  -- B₁ and C₁ are on circleA
  (B₁.x - A.x)^2 + (B₁.y - A.y)^2 = circleA.radius^2 →
  (C₁.x - A.x)^2 + (C₁.y - A.y)^2 = circleA.radius^2 →
  -- C₂ and A₂ are on circleB
  (C₂.x - B.x)^2 + (C₂.y - B.y)^2 = circleB.radius^2 →
  (A₂.x - B.x)^2 + (A₂.y - B.y)^2 = circleB.radius^2 →
  -- A₃ and B₃ are on circleC
  (A₃.x - C.x)^2 + (A₃.y - C.y)^2 = circleC.radius^2 →
  (B₃.x - C.x)^2 + (B₃.y - C.y)^2 = circleC.radius^2 →
  -- X is the intersection of B₁C₁ and BC
  (∃ (l₁ : Line), l₁.a * B₁.x + l₁.b * B₁.y + l₁.c = 0 ∧
                  l₁.a * C₁.x + l₁.b * C₁.y + l₁.c = 0 ∧
                  l₁.a * X.x + l₁.b * X.y + l₁.c = 0) →
  (∃ (l₂ : Line), l₂.a * B.x + l₂.b * B.y + l₂.c = 0 ∧
                  l₂.a * C.x + l₂.b * C.y + l₂.c = 0 ∧
                  l₂.a * X.x + l₂.b * X.y + l₂.c = 0) →
  -- Y is the intersection of C₂A₂ and CA
  (∃ (l₃ : Line), l₃.a * C₂.x + l₃.b * C₂.y + l₃.c = 0 ∧
                  l₃.a * A₂.x + l₃.b * A₂.y + l₃.c = 0 ∧
                  l₃.a * Y.x + l₃.b * Y.y + l₃.c = 0) →
  (∃ (l₄ : Line), l₄.a * C.x + l₄.b * C.y + l₄.c = 0 ∧
                  l₄.a * A.x + l₄.b * A.y + l₄.c = 0 ∧
                  l₄.a * Y.x + l₄.b * Y.y + l₄.c = 0) →
  -- Z is the intersection of A₃B₃ and AB
  (∃ (l₅ : Line), l₅.a * A₃.x + l₅.b * A₃.y + l₅.c = 0 ∧
                  l₅.a * B₃.x + l₅.b * B₃.y + l₅.c = 0 ∧
                  l₅.a * Z.x + l₅.b * Z.y + l₅.c = 0) →
  (∃ (l₆ : Line), l₆.a * A.x + l₆.b * A.y + l₆.c = 0 ∧
                  l₆.a * B.x + l₆.b * B.y + l₆.c = 0 ∧
                  l₆.a * Z.x + l₆.b * Z.y + l₆.c = 0) →
  -- Conclusion: X, Y, and Z are collinear
  ∃ (l : Line), l.a * X.x + l.b * X.y + l.c = 0 ∧
                l.a * Y.x + l.b * Y.y + l.c = 0 ∧
                l.a * Z.x + l.b * Z.y + l.c = 0 :=
by
  sorry


end three_circles_collinearity_l583_58357


namespace heather_final_blocks_l583_58364

-- Define the initial number of blocks Heather has
def heather_initial : ℝ := 86.0

-- Define the number of blocks Jose shares
def jose_shares : ℝ := 41.0

-- Theorem statement
theorem heather_final_blocks : 
  heather_initial + jose_shares = 127.0 := by
  sorry

end heather_final_blocks_l583_58364


namespace product_without_x_cube_term_l583_58353

theorem product_without_x_cube_term (m : ℚ) : 
  (∀ a b c d : ℚ, (m * X^4 + a * X^3 + b * X^2 + c * X + d) = 
    (m * X^2 - 3 * X) * (X^2 - 2 * X - 1) → a = 0) → 
  m = -3/2 := by sorry

end product_without_x_cube_term_l583_58353


namespace allan_plum_count_l583_58360

/-- The number of plums Sharon has -/
def sharon_plums : ℕ := 7

/-- The difference between Sharon's plums and Allan's plums -/
def plum_difference : ℕ := 3

/-- The number of plums Allan has -/
def allan_plums : ℕ := sharon_plums - plum_difference

theorem allan_plum_count : allan_plums = 4 := by
  sorry

end allan_plum_count_l583_58360


namespace complex_equation_solution_l583_58371

theorem complex_equation_solution (z : ℂ) : z * (1 - Complex.I) = 2 → z = 1 + Complex.I := by
  sorry

end complex_equation_solution_l583_58371


namespace exists_perpendicular_line_l583_58365

-- Define a plane
variable (α : Set (ℝ × ℝ × ℝ))

-- Define a line
variable (l : Set (ℝ × ℝ × ℝ))

-- Define a predicate for a line being in a plane
def LineInPlane (line : Set (ℝ × ℝ × ℝ)) (plane : Set (ℝ × ℝ × ℝ)) : Prop :=
  line ⊆ plane

-- Define a predicate for two lines being perpendicular
def Perpendicular (line1 line2 : Set (ℝ × ℝ × ℝ)) : Prop :=
  sorry -- Definition of perpendicularity

-- Theorem statement
theorem exists_perpendicular_line (α : Set (ℝ × ℝ × ℝ)) (l : Set (ℝ × ℝ × ℝ)) :
  ∃ m : Set (ℝ × ℝ × ℝ), LineInPlane m α ∧ Perpendicular m l :=
sorry

end exists_perpendicular_line_l583_58365


namespace expected_digits_is_1_55_l583_58317

/-- A fair 20-sided die with numbers from 1 to 20 -/
def icosahedral_die : Finset ℕ := Finset.range 20

/-- The probability of rolling any specific number on the die -/
def prob_roll (n : ℕ) : ℚ := if n ∈ icosahedral_die then 1 / 20 else 0

/-- The number of digits in a natural number -/
def num_digits (n : ℕ) : ℕ := 
  if n < 10 then 1 else 2

/-- The expected number of digits when rolling the icosahedral die -/
def expected_digits : ℚ := 
  (icosahedral_die.sum (λ n => prob_roll n * num_digits n))

/-- Theorem: The expected number of digits when rolling a fair 20-sided die 
    with numbers from 1 to 20 is 1.55 -/
theorem expected_digits_is_1_55 : expected_digits = 31 / 20 := by
  sorry

end expected_digits_is_1_55_l583_58317


namespace product_remainder_l583_58314

theorem product_remainder (a b c : ℕ) (ha : a = 1234) (hb : b = 1567) (hc : c = 1912) :
  (a * b * c) % 5 = 1 := by
  sorry

end product_remainder_l583_58314


namespace basketball_success_rate_l583_58370

theorem basketball_success_rate (p : ℝ) 
  (h : 1 - p^2 = 16/25) : p = 3/5 := by sorry

end basketball_success_rate_l583_58370


namespace book_cost_l583_58339

/-- Given that three identical books cost $36, prove that seven of these books cost $84. -/
theorem book_cost (cost_of_three : ℝ) (h : cost_of_three = 36) : 
  (7 / 3) * cost_of_three = 84 := by
  sorry

end book_cost_l583_58339


namespace correct_stratified_sample_l583_58394

/-- Represents the number of employees in each age group -/
structure AgeGroup where
  middleAged : ℕ
  young : ℕ
  elderly : ℕ

/-- Represents the ratio of employees in each age group -/
structure AgeRatio where
  middleAged : ℕ
  young : ℕ
  elderly : ℕ

/-- Calculates the stratified sample size for each age group -/
def stratifiedSample (totalPopulation : ℕ) (sampleSize : ℕ) (ratio : AgeRatio) : AgeGroup :=
  let totalRatio := ratio.middleAged + ratio.young + ratio.elderly
  { middleAged := sampleSize * ratio.middleAged / totalRatio,
    young := sampleSize * ratio.young / totalRatio,
    elderly := sampleSize * ratio.elderly / totalRatio }

theorem correct_stratified_sample :
  let totalPopulation : ℕ := 3200
  let sampleSize : ℕ := 400
  let ratio : AgeRatio := { middleAged := 5, young := 3, elderly := 2 }
  let sample : AgeGroup := stratifiedSample totalPopulation sampleSize ratio
  sample.middleAged = 200 ∧ sample.young = 120 ∧ sample.elderly = 80 := by
  sorry

end correct_stratified_sample_l583_58394


namespace circles_tangent_internally_l583_58311

theorem circles_tangent_internally (r₁ r₂ d : ℝ) :
  r₁ = 4 → r₂ = 7 → d = 3 → d = r₂ - r₁ := by
  sorry

end circles_tangent_internally_l583_58311


namespace midpoint_coordinate_product_l583_58338

/-- Given a line segment CD where C(6,-1) is one endpoint and M(4,3) is the midpoint,
    the product of the coordinates of point D is 14. -/
theorem midpoint_coordinate_product : 
  let C : ℝ × ℝ := (6, -1)
  let M : ℝ × ℝ := (4, 3)
  let D : ℝ × ℝ := (2 * M.1 - C.1, 2 * M.2 - C.2)  -- Midpoint formula solved for D
  (D.1 * D.2 = 14) := by sorry

end midpoint_coordinate_product_l583_58338


namespace intersection_distance_l583_58316

def C₁ (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 7

def C₂ (ρ θ : ℝ) : Prop := ρ = 2 * Real.cos θ

theorem intersection_distance :
  ∃ (ρ₁ ρ₂ : ℝ),
    C₁ (ρ₁ * Real.cos (π/6)) (ρ₁ * Real.sin (π/6)) ∧
    C₂ ρ₂ (π/6) ∧
    ρ₁ > 0 ∧ ρ₂ > 0 ∧
    ρ₁ - ρ₂ = 3 - Real.sqrt 3 :=
  sorry

end intersection_distance_l583_58316


namespace min_value_of_function_l583_58356

theorem min_value_of_function (x : ℝ) (h : x > 0) :
  x + 2 / (2 * x + 1) - 3 / 2 ≥ 0 ∧
  ∃ y > 0, y + 2 / (2 * y + 1) - 3 / 2 = 0 :=
by sorry

end min_value_of_function_l583_58356


namespace birds_on_fence_l583_58327

/-- Given an initial number of birds on a fence and an additional number of birds that land on the fence,
    calculate the total number of birds on the fence. -/
def total_birds (initial : Nat) (additional : Nat) : Nat :=
  initial + additional

/-- Theorem stating that with 12 initial birds and 8 additional birds, the total is 20 -/
theorem birds_on_fence : total_birds 12 8 = 20 := by
  sorry

end birds_on_fence_l583_58327


namespace adjacent_triangle_number_l583_58325

/-- Given a triangular arrangement of natural numbers where the k-th row 
    contains numbers from (k-1)^2 + 1 to k^2, if 267 is in one triangle, 
    then 301 is in the adjacent triangle that shares a horizontal side. -/
theorem adjacent_triangle_number : ∀ (k : ℕ),
  (k - 1)^2 + 1 ≤ 267 ∧ 267 ≤ k^2 →
  ∃ (n : ℕ), n ≤ k^2 - ((k - 1)^2 + 1) + 1 ∧
  301 = (k + 1)^2 - (n + k - 1) :=
by sorry

end adjacent_triangle_number_l583_58325


namespace prize_distribution_l583_58332

theorem prize_distribution (total_prize : ℕ) (num_prizes : ℕ) (first_prize : ℕ) (second_prize : ℕ) (third_prize : ℕ)
  (h_total : total_prize = 4200)
  (h_num : num_prizes = 7)
  (h_first : first_prize = 800)
  (h_second : second_prize = 700)
  (h_third : third_prize = 300) :
  ∃ (x y z : ℕ),
    x + y + z = num_prizes ∧
    x * first_prize + y * second_prize + z * third_prize = total_prize ∧
    x = 1 ∧ y = 4 ∧ z = 2 := by
  sorry

end prize_distribution_l583_58332


namespace jelly_beans_problem_l583_58308

theorem jelly_beans_problem (b c : ℕ) : 
  b = 3 * c →                   -- Initially, blueberry count is 3 times cherry count
  b - 20 = 4 * (c - 20) →       -- After eating 20 of each, blueberry count is 4 times cherry count
  b = 180                       -- Prove that initial blueberry count was 180
  := by sorry

end jelly_beans_problem_l583_58308


namespace wire_cutting_l583_58375

theorem wire_cutting (total_length : ℝ) (ratio : ℝ) (shorter_length : ℝ) : 
  total_length = 70 →
  ratio = 2 / 5 →
  shorter_length + (shorter_length / ratio) = total_length →
  shorter_length = 20 := by
sorry

end wire_cutting_l583_58375


namespace hobby_store_sales_l583_58334

/-- The combined sales of trading cards in June and July -/
def combined_sales (normal_sales : ℕ) (june_extra : ℕ) : ℕ :=
  (normal_sales + june_extra) + normal_sales

/-- Theorem stating the combined sales of trading cards in June and July -/
theorem hobby_store_sales : combined_sales 21122 3922 = 46166 := by
  sorry

end hobby_store_sales_l583_58334


namespace positive_correlation_implies_positive_slope_negative_correlation_implies_negative_slope_positive_slope_implies_positive_correlation_negative_slope_implies_negative_correlation_l583_58324

/-- Represents a linear regression equation -/
structure LinearRegression where
  slope : ℝ
  intercept : ℝ

/-- Determines if two variables are positively correlated based on the slope of their linear regression equation -/
def positively_correlated (eq : LinearRegression) : Prop :=
  eq.slope > 0

/-- Determines if two variables are negatively correlated based on the slope of their linear regression equation -/
def negatively_correlated (eq : LinearRegression) : Prop :=
  eq.slope < 0

/-- States that a linear regression equation with positive slope implies positive correlation -/
theorem positive_correlation_implies_positive_slope (eq : LinearRegression) :
  positively_correlated eq → eq.slope > 0 := by sorry

/-- States that a linear regression equation with negative slope implies negative correlation -/
theorem negative_correlation_implies_negative_slope (eq : LinearRegression) :
  negatively_correlated eq → eq.slope < 0 := by sorry

/-- States that positive slope implies positive correlation -/
theorem positive_slope_implies_positive_correlation (eq : LinearRegression) :
  eq.slope > 0 → positively_correlated eq := by sorry

/-- States that negative slope implies negative correlation -/
theorem negative_slope_implies_negative_correlation (eq : LinearRegression) :
  eq.slope < 0 → negatively_correlated eq := by sorry

end positive_correlation_implies_positive_slope_negative_correlation_implies_negative_slope_positive_slope_implies_positive_correlation_negative_slope_implies_negative_correlation_l583_58324


namespace men_in_second_group_l583_58366

/-- Given the conditions of the problem, prove that the number of men in the second group is 9 -/
theorem men_in_second_group : 
  let first_group_men : ℕ := 4
  let first_group_hours_per_day : ℕ := 10
  let first_group_earnings : ℕ := 1200
  let second_group_hours_per_day : ℕ := 6
  let second_group_earnings : ℕ := 1620
  let days_per_week : ℕ := 7
  
  ∃ (second_group_men : ℕ),
    second_group_men * second_group_hours_per_day * days_per_week * first_group_earnings = 
    first_group_men * first_group_hours_per_day * days_per_week * second_group_earnings ∧
    second_group_men = 9 :=
by
  sorry

end men_in_second_group_l583_58366


namespace obtuse_triangle_x_range_l583_58330

/-- Represents the side lengths of a triangle --/
structure TriangleSides where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a triangle is obtuse --/
def isObtuse (t : TriangleSides) : Prop :=
  (t.a ^ 2 + t.b ^ 2 < t.c ^ 2) ∨ (t.a ^ 2 + t.c ^ 2 < t.b ^ 2) ∨ (t.b ^ 2 + t.c ^ 2 < t.a ^ 2)

/-- The theorem stating the range of x for the given obtuse triangle --/
theorem obtuse_triangle_x_range :
  ∀ x : ℝ,
  let t := TriangleSides.mk 3 4 x
  isObtuse t →
  (1 < x ∧ x < Real.sqrt 7) ∨ (5 < x ∧ x < 7) :=
by sorry

end obtuse_triangle_x_range_l583_58330


namespace max_omega_for_monotonic_sin_l583_58392

theorem max_omega_for_monotonic_sin (f : ℝ → ℝ) (ω : ℝ) :
  (∀ x, f x = Real.sin (ω * x)) →
  ω > 0 →
  (∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ π / 3 → f x < f y) →
  ω ≤ 3 / 2 ∧ ∀ ω' > 3 / 2, ∃ x y, 0 ≤ x ∧ x < y ∧ y ≤ π / 3 ∧ f x ≥ f y :=
by sorry

end max_omega_for_monotonic_sin_l583_58392


namespace fraction_equality_l583_58323

theorem fraction_equality (a b : ℝ) (x : ℝ) (h1 : x = a / b) (h2 : a ≠ b) (h3 : b ≠ 0) :
  (a + b) / (a - b) = (x + 1) / (x - 1) := by
  sorry

end fraction_equality_l583_58323


namespace chongqing_population_scientific_notation_l583_58380

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- The population of Chongqing at the end of 2022 -/
def chongqing_population : ℕ := 32000000

/-- Converts a natural number to its scientific notation representation -/
def to_scientific_notation (n : ℕ) : ScientificNotation :=
  sorry

theorem chongqing_population_scientific_notation :
  to_scientific_notation chongqing_population =
    ScientificNotation.mk 3.2 7 (by norm_num) :=
  sorry

end chongqing_population_scientific_notation_l583_58380


namespace max_min_values_of_f_l583_58354

-- Define the function f(x) = x^3 - 3x + 1
def f (x : ℝ) : ℝ := x^3 - 3*x + 1

-- Define the closed interval [-3, 0]
def interval : Set ℝ := { x | -3 ≤ x ∧ x ≤ 0 }

-- Theorem statement
theorem max_min_values_of_f :
  (∃ x ∈ interval, ∀ y ∈ interval, f y ≤ f x) ∧
  (∃ x ∈ interval, ∀ y ∈ interval, f x ≤ f y) ∧
  (∃ x ∈ interval, f x = 3) ∧
  (∃ x ∈ interval, f x = -17) :=
sorry

end max_min_values_of_f_l583_58354


namespace polynomial_sum_squares_l583_58329

theorem polynomial_sum_squares (a : ℝ) (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : ℝ) :
  (∀ x : ℝ, (x - 2)^8 = a + a₁*(x-1) + a₂*(x-1)^2 + a₃*(x-1)^3 + a₄*(x-1)^4 + 
                        a₅*(x-1)^5 + a₆*(x-1)^6 + a₇*(x-1)^7 + a₈*(x-1)^8) →
  (a₂ + a₄ + a₆ + a₈)^2 - (a₁ + a₃ + a₅ + a₇)^2 = -255 := by
sorry

end polynomial_sum_squares_l583_58329


namespace remove_one_gives_average_eight_point_five_l583_58387

def original_list : List ℕ := [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

def remove_number (list : List ℕ) (n : ℕ) : List ℕ :=
  list.filter (λ x => x ≠ n)

def average (list : List ℕ) : ℚ :=
  (list.sum : ℚ) / list.length

theorem remove_one_gives_average_eight_point_five :
  average (remove_number original_list 1) = 8.5 := by
  sorry

end remove_one_gives_average_eight_point_five_l583_58387


namespace vector_subtraction_l583_58381

/-- Given two vectors OM and ON in ℝ², prove that the vector MN has coordinates (-8, 1) -/
theorem vector_subtraction (OM ON : ℝ × ℝ) (h1 : OM = (3, -2)) (h2 : ON = (-5, -1)) :
  ON - OM = (-8, 1) := by
  sorry

end vector_subtraction_l583_58381


namespace g_sum_zero_l583_58383

def g (x : ℝ) : ℝ := x^2 - 2013*x

theorem g_sum_zero (a b : ℝ) (h1 : g a = g b) (h2 : a ≠ b) : g (a + b) = 0 := by
  sorry

end g_sum_zero_l583_58383


namespace vector_collinearity_l583_58379

/-- Given vectors in ℝ², prove that if 3a + b is collinear with c, then x = -4 -/
theorem vector_collinearity (a b c : ℝ × ℝ) (x : ℝ) :
  a = (-2, 0) →
  b = (2, 1) →
  c = (x, 1) →
  ∃ (k : ℝ), k • (3 • a + b) = c →
  x = -4 := by
  sorry

end vector_collinearity_l583_58379


namespace brian_stones_l583_58326

theorem brian_stones (total : ℕ) (white black : ℕ) (h1 : total = 100) 
  (h2 : white + black = total) 
  (h3 : white * 60 = black * 40) 
  (h4 : white > black) : white = 40 := by
  sorry

end brian_stones_l583_58326


namespace congruence_problem_l583_58321

theorem congruence_problem (x : ℤ) : 
  (5 * x + 9) % 19 = 3 → (3 * x + 14) % 19 = 18 := by
  sorry

end congruence_problem_l583_58321


namespace exists_polynomial_composition_l583_58391

-- Define the polynomials P and Q
variable (K : Type*) [Field K]
variable (P Q : K → K)

-- Define the condition for the existence of R
variable (R : K → K → K)
variable (h : ∀ x y, P x - P y = R x y * (Q x - Q y))

-- Theorem statement
theorem exists_polynomial_composition :
  ∃ S : K → K, ∀ x, P x = S (Q x) := by
  sorry

end exists_polynomial_composition_l583_58391


namespace condition_necessary_not_sufficient_l583_58304

theorem condition_necessary_not_sufficient : 
  (∀ x : ℝ, x = 0 → (2*x - 1)*x = 0) ∧ 
  ¬(∀ x : ℝ, (2*x - 1)*x = 0 → x = 0) := by
  sorry

end condition_necessary_not_sufficient_l583_58304


namespace range_of_a_l583_58377

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 + 2*a*x + 4 > 0) ∧
  (∀ x y : ℝ, x < y → (3 - 2*a)^x < (3 - 2*a)^y) →
  a > -2 ∧ a < 1 :=
by sorry

end range_of_a_l583_58377


namespace pencils_per_child_l583_58315

theorem pencils_per_child (num_children : ℕ) (total_pencils : ℕ) (h1 : num_children = 2) (h2 : total_pencils = 12) :
  total_pencils / num_children = 6 := by
sorry

end pencils_per_child_l583_58315


namespace three_number_sum_l583_58350

theorem three_number_sum (a b c : ℝ) (h1 : a < b) (h2 : b < c) 
  (h3 : ((a + b)/2 + (b + c)/2) / 2 = (a + b + c) / 3)
  (h4 : (a + c) / 2 = 2022) : 
  a + b + c = 6066 := by
  sorry

end three_number_sum_l583_58350


namespace right_triangle_m_values_l583_58322

-- Define points A, B, and P
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (4, 0)
def P (m : ℝ) : ℝ × ℝ := (m, 0.5 * m + 2)

-- Define the condition for a right-angled triangle
def isRightAngled (a b c : ℝ × ℝ) : Prop :=
  (b.1 - a.1) * (c.1 - a.1) + (b.2 - a.2) * (c.2 - a.2) = 0 ∨
  (a.1 - b.1) * (c.1 - b.1) + (a.2 - b.2) * (c.2 - b.2) = 0 ∨
  (a.1 - c.1) * (b.1 - c.1) + (a.2 - c.2) * (b.2 - c.2) = 0

-- State the theorem
theorem right_triangle_m_values :
  ∀ m : ℝ, isRightAngled A B (P m) →
    m = -2 ∨ m = 4 ∨ m = (4 * Real.sqrt 5) / 5 ∨ m = -(4 * Real.sqrt 5) / 5 := by
  sorry

end right_triangle_m_values_l583_58322


namespace greg_situps_l583_58358

/-- 
Given:
- For every sit-up Peter does, Greg does 4.
- Peter did 24 sit-ups.

Prove that Greg did 96 sit-ups.
-/
theorem greg_situps (peter_situps : ℕ) (greg_ratio : ℕ) : 
  peter_situps = 24 → greg_ratio = 4 → peter_situps * greg_ratio = 96 := by
  sorry

end greg_situps_l583_58358


namespace five_digit_numbers_count_l583_58348

theorem five_digit_numbers_count : 
  (Finset.filter (fun n : Nat => 
    n ≥ 10000 ∧ n < 100000 ∧ 
    (n / 10000) ≠ 5 ∧
    (n % 10) ≠ 2 ∧
    (Finset.card (Finset.image (fun i => (n / (10 ^ i)) % 10) (Finset.range 5))) = 5
  ) (Finset.range 100000)).card = 8 * 9 * 8 * 7 * 6 := by
  sorry

end five_digit_numbers_count_l583_58348


namespace multiply_sum_power_l583_58361

theorem multiply_sum_power (n : ℕ) (h : n > 0) :
  n * (n^n + 1) = n^(n + 1) + n :=
by sorry

end multiply_sum_power_l583_58361


namespace single_digit_square_5929_l583_58367

theorem single_digit_square_5929 :
  ∃! (A : ℕ), A < 10 ∧ (10 * A + A)^2 = 5929 :=
by
  -- The proof goes here
  sorry

end single_digit_square_5929_l583_58367


namespace optimal_price_maximizes_profit_l583_58384

/-- Represents the profit function for a product with given pricing conditions -/
def profit_function (x : ℝ) : ℝ := -x^2 + 190*x - 7800

/-- The optimal selling price that maximizes profit -/
def optimal_price : ℝ := 95

/-- Theorem stating that the optimal price maximizes the profit function -/
theorem optimal_price_maximizes_profit :
  ∀ x : ℝ, 60 ≤ x ∧ x ≤ 130 → profit_function x ≤ profit_function optimal_price :=
by sorry

end optimal_price_maximizes_profit_l583_58384


namespace empty_truck_weight_l583_58362

-- Define the constants
def bridge_limit : ℕ := 20000
def soda_crates : ℕ := 20
def soda_crate_weight : ℕ := 50
def dryers : ℕ := 3
def dryer_weight : ℕ := 3000
def loaded_truck_weight : ℕ := 24000

-- Define the theorem
theorem empty_truck_weight :
  let soda_weight := soda_crates * soda_crate_weight
  let produce_weight := 2 * soda_weight
  let dryers_weight := dryers * dryer_weight
  let cargo_weight := soda_weight + produce_weight + dryers_weight
  loaded_truck_weight - cargo_weight = 12000 := by
  sorry


end empty_truck_weight_l583_58362


namespace base_ten_to_base_seven_l583_58312

theorem base_ten_to_base_seven : 
  ∃ (a b c d : ℕ), 
    1357 = a * 7^3 + b * 7^2 + c * 7^1 + d * 7^0 ∧ 
    a < 7 ∧ b < 7 ∧ c < 7 ∧ d < 7 ∧
    a = 3 ∧ b = 6 ∧ c = 4 ∧ d = 6 := by
  sorry

end base_ten_to_base_seven_l583_58312


namespace lineup_combinations_l583_58307

-- Define the total number of players
def total_players : ℕ := 18

-- Define the number of players in a triplet
def triplet_size : ℕ := 3

-- Define the number of triplet sets
def triplet_sets : ℕ := 2

-- Define the number of players to choose for the lineup
def lineup_size : ℕ := 7

-- Define the maximum number of players that can be chosen from a triplet set
def max_from_triplet : ℕ := 2

-- Define the function to calculate the number of ways to choose the lineup
def choose_lineup : ℕ := sorry

-- Theorem stating that the number of ways to choose the lineup is 21582
theorem lineup_combinations : choose_lineup = 21582 := by sorry

end lineup_combinations_l583_58307


namespace frank_reading_rate_l583_58328

/-- Given a book with a certain number of chapters read over a certain number of days,
    calculate the number of chapters read per day. -/
def chapters_per_day (total_chapters : ℕ) (days : ℕ) : ℚ :=
  (total_chapters : ℚ) / (days : ℚ)

/-- Theorem: For a book with 2 chapters read over 664 days,
    the number of chapters read per day is 2/664. -/
theorem frank_reading_rate : chapters_per_day 2 664 = 2 / 664 := by
  sorry

end frank_reading_rate_l583_58328


namespace complex_equation_solution_l583_58398

theorem complex_equation_solution (z : ℂ) : (1 + Complex.I) / (z - Complex.I) = Complex.I → z = 1 := by
  sorry

end complex_equation_solution_l583_58398


namespace arithmetic_sequence_ratio_l583_58378

theorem arithmetic_sequence_ratio (x y d₁ d₂ : ℝ) (h₁ : d₁ ≠ 0) (h₂ : d₂ ≠ 0) 
  (h₃ : x + 4 * d₁ = y) (h₄ : x + 5 * d₂ = y) : d₁ / d₂ = 5 / 4 := by
  sorry

end arithmetic_sequence_ratio_l583_58378


namespace hyperbola_equation_l583_58376

/-- The trajectory of a point P satisfying |PF₂| - |PF₁| = 4, where F₁(-4, 0) and F₂(4, 0) are fixed points -/
def hyperbola_trajectory (P : ℝ × ℝ) : Prop :=
  let F₁ : ℝ × ℝ := (-4, 0)
  let F₂ : ℝ × ℝ := (4, 0)
  let d (A B : ℝ × ℝ) := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  d P F₂ - d P F₁ = 4 →
  P.1^2 / 4 - P.2^2 / 12 = 1 ∧ P.1 ≤ -2

theorem hyperbola_equation : 
  ∀ P : ℝ × ℝ, hyperbola_trajectory P :=
sorry

end hyperbola_equation_l583_58376


namespace students_playing_neither_l583_58319

theorem students_playing_neither (total : ℕ) (football : ℕ) (tennis : ℕ) (both : ℕ) 
  (h1 : total = 36)
  (h2 : football = 26)
  (h3 : tennis = 20)
  (h4 : both = 17) :
  total - (football + tennis - both) = 7 := by
  sorry

end students_playing_neither_l583_58319


namespace last_triangle_perimeter_l583_58352

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  valid : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

/-- Generates the next triangle in the sequence based on the current triangle -/
def nextTriangle (T : Triangle) : Option Triangle := sorry

/-- The sequence of triangles starting from T₁ -/
def triangleSequence : ℕ → Option Triangle
  | 0 => some ⟨1003, 1004, 1005, sorry⟩
  | n + 1 => (triangleSequence n).bind nextTriangle

/-- The perimeter of a triangle -/
def perimeter (T : Triangle) : ℝ := T.a + T.b + T.c

/-- Finds the last existing triangle in the sequence -/
def lastTriangle : Option Triangle := sorry

theorem last_triangle_perimeter :
  ∀ T, lastTriangle = some T → perimeter T = 753 / 128 := by sorry

end last_triangle_perimeter_l583_58352


namespace inequality_proof_l583_58343

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_xyz : x * y * z = 1) :
  (x^2 / (y + z) + y^2 / (z + x) + z^2 / (x + y)) ≥ 
  (1/2) * ((x^2 + y^2) / (x + y) + (y^2 + z^2) / (y + z) + (z^2 + x^2) / (z + x)) ∧
  (1/2) * ((x^2 + y^2) / (x + y) + (y^2 + z^2) / (y + z) + (z^2 + x^2) / (z + x)) ≥ (x + y + z) / 2 ∧
  (x + y + z) / 2 ≥ 3/2 := by
sorry

end inequality_proof_l583_58343


namespace bolded_area_percentage_l583_58368

theorem bolded_area_percentage (s : ℝ) (h : s > 0) : 
  let square_area := s^2
  let total_area := 4 * square_area
  let bolded_area_1 := (1/2) * square_area
  let bolded_area_2 := (1/2) * square_area
  let bolded_area_3 := (1/8) * square_area
  let bolded_area_4 := (1/4) * square_area
  let total_bolded_area := bolded_area_1 + bolded_area_2 + bolded_area_3 + bolded_area_4
  (total_bolded_area / total_area) * 100 = 100/3
:= by sorry

end bolded_area_percentage_l583_58368


namespace index_cards_per_student_l583_58397

theorem index_cards_per_student 
  (num_classes : ℕ) 
  (students_per_class : ℕ) 
  (total_packs : ℕ) 
  (h1 : num_classes = 6) 
  (h2 : students_per_class = 30) 
  (h3 : total_packs = 360) : 
  total_packs / (num_classes * students_per_class) = 2 := by
  sorry

end index_cards_per_student_l583_58397


namespace sqrt_meaningful_range_l583_58363

theorem sqrt_meaningful_range (a : ℝ) : (∃ x : ℝ, x^2 = a - 1) → a ≥ 1 := by
  sorry

end sqrt_meaningful_range_l583_58363


namespace largest_four_digit_divisible_by_six_l583_58306

theorem largest_four_digit_divisible_by_six :
  ∀ n : ℕ, n ≤ 9999 ∧ n ≥ 1000 ∧ n % 6 = 0 → n ≤ 9996 :=
by sorry

end largest_four_digit_divisible_by_six_l583_58306


namespace opposite_values_imply_a_half_l583_58318

theorem opposite_values_imply_a_half (a : ℚ) : (2 * a) + (1 - 4 * a) = 0 → a = 1 / 2 := by
  sorry

end opposite_values_imply_a_half_l583_58318


namespace complement_A_in_U_l583_58302

open Set

-- Define the universal set U
def U : Set ℝ := {x | x^2 > 1}

-- Define set A
def A : Set ℝ := {x | x^2 - 4*x + 3 < 0}

-- Theorem statement
theorem complement_A_in_U :
  (U \ A) = {x : ℝ | x ≥ 3 ∨ x < -1} := by sorry

end complement_A_in_U_l583_58302


namespace coprime_elements_bound_l583_58393

/-- The number of elements in [1, n] coprime to M -/
def h (M n : ℕ+) : ℕ := sorry

/-- The proportion of numbers in [1, M] coprime to M -/
def β (M : ℕ+) : ℚ := (h M M : ℚ) / M

/-- ω(M) is the number of distinct prime factors of M -/
def ω (M : ℕ+) : ℕ := sorry

theorem coprime_elements_bound (M : ℕ+) :
  ∃ S : Finset ℕ+,
    S.card ≥ M / 3 ∧
    ∀ n ∈ S, n ≤ M ∧
    |h M n - β M * n| ≤ Real.sqrt (β M * 2^(ω M - 3)) + 1 :=
  sorry

end coprime_elements_bound_l583_58393


namespace area_between_parallel_chords_l583_58305

theorem area_between_parallel_chords (r : ℝ) (d : ℝ) (h1 : r = 8) (h2 : d = 8) :
  let chord_length := 2 * Real.sqrt (r ^ 2 - (d / 2) ^ 2)
  let segment_area := (1 / 3) * π * r ^ 2 - (1 / 2) * chord_length * (d / 2)
  2 * segment_area = 32 * Real.sqrt 3 + 64 * π / 3 :=
by sorry

end area_between_parallel_chords_l583_58305


namespace hexagon_not_possible_after_cut_l583_58396

-- Define a polygon
structure Polygon :=
  (sides : ℕ)
  (sides_ge_3 : sides ≥ 3)

-- Define the operation of cutting off a corner
def cut_corner (p : Polygon) : Polygon :=
  ⟨p.sides - 1, by sorry⟩

-- Theorem statement
theorem hexagon_not_possible_after_cut (p : Polygon) :
  (cut_corner p).sides = 4 → p.sides ≠ 6 :=
by sorry

end hexagon_not_possible_after_cut_l583_58396


namespace batch_not_qualified_l583_58399

-- Define the parameters of the normal distribution
def mean : ℝ := 4
def std_dev : ℝ := 0.5  -- sqrt(0.25)

-- Define the measured diameter
def measured_diameter : ℝ := 5.7

-- Define a function to determine if a batch is qualified
def is_qualified (x : ℝ) : Prop :=
  (x - mean) / std_dev ≤ 3 ∧ (x - mean) / std_dev ≥ -3

-- Theorem statement
theorem batch_not_qualified : ¬(is_qualified measured_diameter) :=
sorry

end batch_not_qualified_l583_58399


namespace rectangleAreaStage4_l583_58310

/-- The area of a rectangle formed by four squares with side lengths
    starting from 2 inches and increasing by 1 inch per stage. -/
def rectangleArea : ℕ → ℕ
| 1 => 2^2
| 2 => 2^2 + 3^2
| 3 => 2^2 + 3^2 + 4^2
| 4 => 2^2 + 3^2 + 4^2 + 5^2
| _ => 0

/-- The area of the rectangle at Stage 4 is 54 square inches. -/
theorem rectangleAreaStage4 : rectangleArea 4 = 54 := by
  sorry

end rectangleAreaStage4_l583_58310


namespace flower_planting_area_l583_58351

/-- Represents a square lawn with flowers -/
structure FlowerLawn where
  side_length : ℝ
  flower_area : ℝ

/-- Theorem: A square lawn with side length 16 meters can have a flower planting area of 144 square meters -/
theorem flower_planting_area (lawn : FlowerLawn) (h1 : lawn.side_length = 16) 
  (h2 : lawn.flower_area = 144) : 
  lawn.flower_area ≤ lawn.side_length ^ 2 ∧ lawn.flower_area > 0 := by
  sorry

#check flower_planting_area

end flower_planting_area_l583_58351


namespace festival_average_surfers_l583_58345

/-- The average number of surfers at the Rip Curl Myrtle Beach Surf Festival -/
def average_surfers (day1 : ℕ) (day2 : ℕ) (day3 : ℕ) : ℚ :=
  (day1 + day2 + day3) / 3

/-- Theorem: The average number of surfers at the Festival for three days is 1400 -/
theorem festival_average_surfers :
  let day1 : ℕ := 1500
  let day2 : ℕ := day1 + 600
  let day3 : ℕ := day1 * 2 / 5
  average_surfers day1 day2 day3 = 1400 := by
  sorry

end festival_average_surfers_l583_58345


namespace angle_of_inclination_sqrt3_l583_58349

theorem angle_of_inclination_sqrt3 :
  let line : ℝ → ℝ := λ x ↦ Real.sqrt 3 * x - 2
  let slope : ℝ := Real.sqrt 3
  let angle_of_inclination : ℝ := Real.arctan slope
  angle_of_inclination = π / 3 := by sorry

end angle_of_inclination_sqrt3_l583_58349


namespace exponential_function_sum_of_extrema_l583_58355

theorem exponential_function_sum_of_extrema (a : ℝ) : 
  (a > 0) → 
  (∀ x ∈ Set.Icc 0 1, ∃ y, y = a^x) →
  (a^1 + a^0 = 3) →
  (a = 2) := by
sorry

end exponential_function_sum_of_extrema_l583_58355


namespace closest_point_l583_58309

/-- The vector that depends on the scalar parameter s -/
def u (s : ℝ) : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => 3 + 4*s
  | 1 => -2 - 6*s
  | 2 => 1 + 2*s

/-- The constant vector b -/
def b : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => 1
  | 1 => 5
  | 2 => -3

/-- The direction vector of the line -/
def v : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => 4
  | 1 => -6
  | 2 => 2

/-- Theorem stating that s = 13/8 minimizes the distance between u and b -/
theorem closest_point (s : ℝ) :
  (∀ i, (u s i - b i) * v i = 0) ↔ s = 13/8 := by
  sorry

end closest_point_l583_58309


namespace point_transformation_l583_58336

/-- Rotation of 90° counterclockwise around a point -/
def rotate90 (x y cx cy : ℝ) : ℝ × ℝ :=
  (cx - (y - cy), cy + (x - cx))

/-- Reflection about y = x line -/
def reflectYeqX (x y : ℝ) : ℝ × ℝ := (y, x)

/-- The main theorem -/
theorem point_transformation (a b : ℝ) :
  let p := (a, b)
  let rotated := rotate90 a b 2 3
  let final := reflectYeqX rotated.1 rotated.2
  final = (5, 1) → b - a = 2 := by
  sorry

end point_transformation_l583_58336


namespace unique_circle_through_three_points_l583_58346

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define a function to check if three points are collinear
def collinear (p1 p2 p3 : Point) : Prop := sorry

-- Define a function to determine the number of circles through three points
def circles_through_points (p1 p2 p3 : Point) : ℕ := sorry

-- Theorem statement
theorem unique_circle_through_three_points (p1 p2 p3 : Point) :
  ¬collinear p1 p2 p3 → circles_through_points p1 p2 p3 = 1 := by sorry

end unique_circle_through_three_points_l583_58346


namespace a_formula_a_2_2_l583_58390

/-- The number of ordered subset groups with empty intersection -/
def a (i j : ℕ+) : ℕ :=
  (2^j.val - 1)^i.val

/-- The theorem stating the formula for a(i,j) -/
theorem a_formula (i j : ℕ+) :
  a i j = (Finset.univ.filter (fun s : Finset (Fin i.val) => s.card > 0)).card ^ j.val :=
by sorry

/-- Specific case for a(2,2) -/
theorem a_2_2 : a 2 2 = 9 :=
by sorry

end a_formula_a_2_2_l583_58390


namespace parabola_kite_sum_l583_58337

/-- Given two parabolas that intersect the coordinate axes in four points forming a kite -/
structure ParabolaKite where
  a' : ℝ
  b' : ℝ
  intersection_points : Fin 4 → ℝ × ℝ
  is_kite : Bool
  kite_area : ℝ

/-- The theorem stating the sum of a' and b' -/
theorem parabola_kite_sum (pk : ParabolaKite)
  (h1 : pk.is_kite = true)
  (h2 : pk.kite_area = 18)
  (h3 : ∀ (i : Fin 4), (pk.intersection_points i).1 = 0 ∨ (pk.intersection_points i).2 = 0)
  (h4 : ∀ (x y : ℝ), y = pk.a' * x^2 + 3 ∨ y = 6 - pk.b' * x^2) :
  pk.a' + pk.b' = 2/9 := by
  sorry

end parabola_kite_sum_l583_58337


namespace constant_term_expansion_l583_58335

theorem constant_term_expansion (a : ℝ) : 
  a > 0 → (∃ c : ℝ, c = 80 ∧ c = (5 : ℕ).choose 4 * a^4) → a = 2 := by
  sorry

end constant_term_expansion_l583_58335


namespace largest_repeated_product_365_l583_58313

def is_eight_digit_repeated (n : ℕ) : Prop :=
  100000000 > n ∧ n ≥ 10000000 ∧ 
  ∃ (a b c d : ℕ), n = a * 10000000 + b * 1000000 + c * 100000 + d * 10000 + 
                    a * 1000 + b * 100 + c * 10 + d

theorem largest_repeated_product_365 : 
  (∀ m : ℕ, m > 273863 → ¬(is_eight_digit_repeated (m * 365))) ∧ 
  is_eight_digit_repeated (273863 * 365) := by
sorry

#eval 273863 * 365  -- Should output 99959995

end largest_repeated_product_365_l583_58313


namespace complex_fraction_simplification_l583_58395

variables (a b x y : ℝ)

theorem complex_fraction_simplification :
  (a * x * (3 * a^2 * x^2 + 5 * b^2 * y^2) + b * y * (2 * a^2 * x^2 + 4 * b^2 * y^2)) / (a * x + b * y) = 3 * a^2 * x^2 + 4 * b^2 * y^2 :=
by sorry

end complex_fraction_simplification_l583_58395


namespace lost_ship_depth_l583_58301

/-- The depth of a lost ship given the descent rate and time taken to reach it. -/
def ship_depth (descent_rate : ℝ) (time_taken : ℝ) : ℝ := descent_rate * time_taken

/-- Theorem stating the depth of the lost ship -/
theorem lost_ship_depth :
  let descent_rate : ℝ := 80
  let time_taken : ℝ := 50
  ship_depth descent_rate time_taken = 4000 := by
  sorry

end lost_ship_depth_l583_58301


namespace license_plate_palindrome_probability_l583_58303

def letter_count : ℕ := 26
def digit_count : ℕ := 10
def plate_length : ℕ := 4

def is_palindrome (s : List α) : Prop :=
  s = s.reverse

def prob_palindrome_letters : ℚ :=
  (letter_count ^ 2 : ℚ) / (letter_count ^ plate_length)

def prob_palindrome_digits : ℚ :=
  (digit_count ^ 2 : ℚ) / (digit_count ^ plate_length)

theorem license_plate_palindrome_probability :
  let prob := prob_palindrome_letters + prob_palindrome_digits - 
              prob_palindrome_letters * prob_palindrome_digits
  prob = 775 / 67600 := by
  sorry

end license_plate_palindrome_probability_l583_58303


namespace sum_of_coefficients_plus_a_l583_58331

theorem sum_of_coefficients_plus_a (a : ℝ) (as : Fin 2007 → ℝ) :
  (∀ x : ℝ, (1 - 2 * x)^2006 = a + (Finset.sum (Finset.range 2007) (λ i => as i * x^i))) →
  Finset.sum (Finset.range 2007) (λ i => a + as i) = 2006 := by
sorry

end sum_of_coefficients_plus_a_l583_58331


namespace optimal_quadruple_l583_58374

def is_valid_quadruple (k l m n : ℕ) : Prop :=
  k > l ∧ l > m ∧ m > n

def sum_inverse (k l m n : ℕ) : ℚ :=
  1 / k + 1 / l + 1 / m + 1 / n

theorem optimal_quadruple :
  ∀ k l m n : ℕ,
    is_valid_quadruple k l m n →
    sum_inverse k l m n < 1 →
    sum_inverse k l m n ≤ sum_inverse 43 7 3 2 :=
by sorry

end optimal_quadruple_l583_58374


namespace same_gender_leaders_count_l583_58385

/-- Represents the number of ways to select a captain and co-captain of the same gender
    from a team with an equal number of men and women. -/
def select_same_gender_leaders (team_size : ℕ) : ℕ :=
  2 * (team_size * (team_size - 1))

/-- Theorem: In a team of 12 men and 12 women, there are 264 ways to select
    a captain and co-captain of the same gender. -/
theorem same_gender_leaders_count :
  select_same_gender_leaders 12 = 264 := by
  sorry

#eval select_same_gender_leaders 12

end same_gender_leaders_count_l583_58385


namespace quadratic_equations_solutions_l583_58342

theorem quadratic_equations_solutions :
  (∃ x1 x2 : ℝ, x1 = -1 + Real.sqrt 5 ∧ x2 = -1 - Real.sqrt 5 ∧
    x1^2 + 2*x1 - 4 = 0 ∧ x2^2 + 2*x2 - 4 = 0) ∧
  (∃ x1 x2 : ℝ, x1 = 3 ∧ x2 = -2 ∧
    2*x1 - 6 = x1*(3-x1) ∧ 2*x2 - 6 = x2*(3-x2)) :=
by
  sorry

#check quadratic_equations_solutions

end quadratic_equations_solutions_l583_58342


namespace prime_sum_square_cube_l583_58341

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def validSolution (p q r : ℕ) : Prop :=
  isPrime p ∧ isPrime q ∧ isPrime r ∧ p + q^2 + r^3 = 200

theorem prime_sum_square_cube :
  {(p, q, r) : ℕ × ℕ × ℕ | validSolution p q r} =
  {(167, 5, 2), (71, 11, 2), (23, 13, 2), (71, 2, 5)} :=
by sorry

end prime_sum_square_cube_l583_58341


namespace max_value_f_when_a_2_range_of_a_for_F_unique_solution_implies_m_1_l583_58347

-- Define the functions
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - (1/2) * a * x^2 + x

noncomputable def F (a : ℝ) (x : ℝ) : ℝ := Real.log x + a / x

-- Theorem 1
theorem max_value_f_when_a_2 :
  ∃ (x : ℝ), x > 0 ∧ f 2 x = 0 ∧ ∀ (y : ℝ), y > 0 → f 2 y ≤ f 2 x :=
sorry

-- Theorem 2
theorem range_of_a_for_F :
  ∀ (a : ℝ), (∀ (x : ℝ), 0 < x ∧ x ≤ 3 → (deriv (F a)) x ≤ 1/2) → a ≥ 1/2 :=
sorry

-- Theorem 3
theorem unique_solution_implies_m_1 :
  ∃! (m : ℝ), m > 0 ∧ ∃! (x : ℝ), x > 0 ∧ m * (f 0 x) = x^2 :=
sorry

end max_value_f_when_a_2_range_of_a_for_F_unique_solution_implies_m_1_l583_58347


namespace complex_number_properties_l583_58340

theorem complex_number_properties (z₁ z₂ : ℂ) (h : Complex.abs z₁ * Complex.abs z₂ ≠ 0) :
  (Complex.abs (z₁ + z₂) ≤ Complex.abs z₁ + Complex.abs z₂) ∧
  (Complex.abs (z₁ * z₂) = Complex.abs z₁ * Complex.abs z₂) :=
by sorry

end complex_number_properties_l583_58340


namespace fourth_number_is_ten_l583_58386

def sequence_property (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n ≥ 3 → n ≤ 10 → a n = a (n - 1) + a (n - 2)

theorem fourth_number_is_ten (a : ℕ → ℕ) 
  (h_seq : sequence_property a) 
  (h_7 : a 7 = 42) 
  (h_9 : a 9 = 110) : 
  a 4 = 10 := by
  sorry

end fourth_number_is_ten_l583_58386


namespace min_distinct_values_l583_58369

theorem min_distinct_values (n : ℕ) (mode_freq : ℕ) (second_freq : ℕ) 
  (h1 : n = 3000)
  (h2 : mode_freq = 15)
  (h3 : second_freq = 14)
  (h4 : ∀ k : ℕ, k ≠ mode_freq → k ≤ second_freq) :
  (∃ x : ℕ, x * mode_freq + x * second_freq + (n - x * mode_freq - x * second_freq) ≤ n ∧ 
   ∀ y : ℕ, y < x → y * mode_freq + y * second_freq + (n - y * mode_freq - y * second_freq) > n) →
  x = 232 := by
sorry

end min_distinct_values_l583_58369


namespace face_mask_profit_l583_58372

/-- Calculates the total profit from selling face masks given specific conditions --/
theorem face_mask_profit : 
  let original_price : ℝ := 10
  let discount1 : ℝ := 0.2
  let discount2 : ℝ := 0.3
  let discount3 : ℝ := 0.4
  let packs1 : ℕ := 20
  let packs2 : ℕ := 30
  let packs3 : ℕ := 40
  let masks_per_pack : ℕ := 5
  let sell_price1 : ℝ := 0.75
  let sell_price2 : ℝ := 0.85
  let sell_price3 : ℝ := 0.95

  let cost1 : ℝ := original_price * (1 - discount1)
  let cost2 : ℝ := original_price * (1 - discount2)
  let cost3 : ℝ := original_price * (1 - discount3)

  let total_cost : ℝ := cost1 + cost2 + cost3

  let revenue1 : ℝ := (packs1 * masks_per_pack : ℝ) * sell_price1
  let revenue2 : ℝ := (packs2 * masks_per_pack : ℝ) * sell_price2
  let revenue3 : ℝ := (packs3 * masks_per_pack : ℝ) * sell_price3

  let total_revenue : ℝ := revenue1 + revenue2 + revenue3

  let total_profit : ℝ := total_revenue - total_cost

  total_profit = 371.5 := by sorry

end face_mask_profit_l583_58372


namespace abs_h_eq_half_l583_58320

/-- Given a quadratic equation x^2 - 4hx = 8, if the sum of squares of its roots is 20,
    then the absolute value of h is 1/2 -/
theorem abs_h_eq_half (h : ℝ) : 
  (∃ x y : ℝ, x^2 - 4*h*x = 8 ∧ y^2 - 4*h*y = 8 ∧ x^2 + y^2 = 20) → |h| = 1/2 := by
  sorry

end abs_h_eq_half_l583_58320


namespace remainder_three_pow_twenty_mod_seven_l583_58388

theorem remainder_three_pow_twenty_mod_seven : 3^20 % 7 = 2 := by
  sorry

end remainder_three_pow_twenty_mod_seven_l583_58388


namespace root_power_equality_l583_58382

theorem root_power_equality (x₀ : ℝ) (h : x₀^11 + x₀^7 + x₀^3 = 1) :
  x₀^4 + x₀^3 - 1 = x₀^15 := by
  sorry

end root_power_equality_l583_58382

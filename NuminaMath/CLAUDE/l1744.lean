import Mathlib

namespace NUMINAMATH_CALUDE_sum_intersections_four_lines_l1744_174443

/-- The number of intersections for a given number of lines -/
def intersections (k : ℕ) : ℕ := 
  if k ≤ 1 then 0
  else Nat.choose k 2

/-- The sum of all possible numbers of intersections for up to 4 lines -/
def sum_intersections : ℕ :=
  (List.range 5).map intersections |>.sum

/-- Theorem: The sum of all possible numbers of intersections for four distinct lines in a plane is 19 -/
theorem sum_intersections_four_lines :
  sum_intersections = 19 := by sorry

end NUMINAMATH_CALUDE_sum_intersections_four_lines_l1744_174443


namespace NUMINAMATH_CALUDE_regions_on_sphere_l1744_174464

/-- 
Given n great circles on a sphere where no three circles intersect at the same point,
a_n represents the number of regions formed by these circles.
-/
def a_n (n : ℕ) : ℕ := n^2 - n + 2

/-- 
Theorem: The number of regions formed by n great circles on a sphere,
where no three circles intersect at the same point, is equal to n^2 - n + 2.
-/
theorem regions_on_sphere (n : ℕ) : 
  a_n n = n^2 - n + 2 := by sorry

end NUMINAMATH_CALUDE_regions_on_sphere_l1744_174464


namespace NUMINAMATH_CALUDE_partner_A_profit_share_l1744_174437

/-- Calculates the share of profit for partner A in a business venture --/
theorem partner_A_profit_share 
  (initial_investment : ℕ) 
  (a_withdrawal b_withdrawal c_investment : ℕ)
  (total_profit : ℕ) :
  let a_investment_months := initial_investment * 5 + (initial_investment - a_withdrawal) * 7
  let b_investment_months := initial_investment * 5 + (initial_investment - b_withdrawal) * 7
  let c_investment_months := initial_investment * 5 + (initial_investment + c_investment) * 7
  let total_investment_months := a_investment_months + b_investment_months + c_investment_months
  (a_investment_months : ℚ) / total_investment_months * total_profit = 20500 :=
by
  sorry

#check partner_A_profit_share 20000 5000 4000 6000 69900

end NUMINAMATH_CALUDE_partner_A_profit_share_l1744_174437


namespace NUMINAMATH_CALUDE_farm_animals_relation_l1744_174468

/-- Given a farm with pigs, cows, and goats, prove the relationship between the number of goats and cows -/
theorem farm_animals_relation (pigs cows goats : ℕ) : 
  pigs = 10 →
  cows = 2 * pigs - 3 →
  pigs + cows + goats = 50 →
  goats = cows + 6 := by
sorry

end NUMINAMATH_CALUDE_farm_animals_relation_l1744_174468


namespace NUMINAMATH_CALUDE_parabola_y_axis_intersection_l1744_174485

def parabola (x y : ℝ) : Prop := y = -(x + 2)^2 + 6

theorem parabola_y_axis_intersection :
  ∃ (y : ℝ), parabola 0 y ∧ y = 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_y_axis_intersection_l1744_174485


namespace NUMINAMATH_CALUDE_product_mod_seven_l1744_174428

theorem product_mod_seven : (2007 * 2008 * 2009 * 2010) % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_seven_l1744_174428


namespace NUMINAMATH_CALUDE_daisy_seeds_count_l1744_174430

/-- The number of daisy seeds planted by Hortense -/
def daisy_seeds : ℕ := sorry

/-- The number of sunflower seeds planted by Hortense -/
def sunflower_seeds : ℕ := 25

/-- The percentage of daisy seeds that germinate -/
def daisy_germination_rate : ℚ := 60 / 100

/-- The percentage of sunflower seeds that germinate -/
def sunflower_germination_rate : ℚ := 80 / 100

/-- The percentage of germinated plants that produce flowers -/
def flower_production_rate : ℚ := 80 / 100

/-- The total number of plants that produce flowers -/
def total_flowering_plants : ℕ := 28

theorem daisy_seeds_count :
  (↑daisy_seeds * daisy_germination_rate * flower_production_rate +
   ↑sunflower_seeds * sunflower_germination_rate * flower_production_rate : ℚ) = total_flowering_plants ∧
  daisy_seeds = 25 :=
sorry

end NUMINAMATH_CALUDE_daisy_seeds_count_l1744_174430


namespace NUMINAMATH_CALUDE_x4_plus_81_factorization_l1744_174417

theorem x4_plus_81_factorization (x : ℝ) :
  x^4 + 81 = (x^2 - 3*x + 4.5) * (x^2 + 3*x + 4.5) := by
  sorry

end NUMINAMATH_CALUDE_x4_plus_81_factorization_l1744_174417


namespace NUMINAMATH_CALUDE_n_squared_not_divides_factorial_l1744_174426

theorem n_squared_not_divides_factorial (n : ℕ) :
  ¬(n^2 ∣ n!) ↔ n = 4 ∨ Nat.Prime n :=
sorry

end NUMINAMATH_CALUDE_n_squared_not_divides_factorial_l1744_174426


namespace NUMINAMATH_CALUDE_find_x_l1744_174431

theorem find_x (x y : ℝ) (h1 : x - y = 10) (h2 : x + y = 8) : x = 9 := by
  sorry

end NUMINAMATH_CALUDE_find_x_l1744_174431


namespace NUMINAMATH_CALUDE_log_216_equals_3_log_6_l1744_174438

theorem log_216_equals_3_log_6 : Real.log 216 = 3 * Real.log 6 := by
  sorry

end NUMINAMATH_CALUDE_log_216_equals_3_log_6_l1744_174438


namespace NUMINAMATH_CALUDE_hexagonal_tiles_count_l1744_174492

/-- Represents the number of edges for each type of tile -/
def edges_per_tile : Fin 3 → ℕ
| 0 => 3  -- triangular
| 1 => 4  -- square
| 2 => 6  -- hexagonal

/-- The total number of tiles in the box -/
def total_tiles : ℕ := 35

/-- The total number of edges from all tiles -/
def total_edges : ℕ := 128

theorem hexagonal_tiles_count :
  ∃ (a b c : ℕ),
    a + b + c = total_tiles ∧
    3 * a + 4 * b + 6 * c = total_edges ∧
    c = 6 :=
sorry

end NUMINAMATH_CALUDE_hexagonal_tiles_count_l1744_174492


namespace NUMINAMATH_CALUDE_power_expression_simplification_l1744_174478

theorem power_expression_simplification :
  (1 : ℚ) / ((-8^2)^4) * (-8)^9 = -8 := by sorry

end NUMINAMATH_CALUDE_power_expression_simplification_l1744_174478


namespace NUMINAMATH_CALUDE_cost_difference_l1744_174471

def vacation_cost (tom dorothy sammy : ℝ) : Prop :=
  tom + dorothy + sammy = 400 ∧ tom = 95 ∧ dorothy = 140 ∧ sammy = 165

theorem cost_difference (tom dorothy sammy t d : ℝ) 
  (h : vacation_cost tom dorothy sammy) :
  t - d = 45 :=
sorry

end NUMINAMATH_CALUDE_cost_difference_l1744_174471


namespace NUMINAMATH_CALUDE_geometric_sequence_iff_t_eq_neg_one_l1744_174465

/-- Given a sequence {a_n} with sum of first n terms S_n = 2^n + t,
    prove it's a geometric sequence iff t = -1 -/
theorem geometric_sequence_iff_t_eq_neg_one
  (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (t : ℝ)
  (h_S : ∀ n, S n = 2^n + t)
  (h_a : ∀ n, a n = S n - S (n-1)) :
  (∃ r : ℝ, ∀ n > 1, a (n+1) = r * a n) ↔ t = -1 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_iff_t_eq_neg_one_l1744_174465


namespace NUMINAMATH_CALUDE_monotonic_increase_interval_l1744_174404

-- Define the function
def f (x : ℝ) := -x^2

-- State the theorem
theorem monotonic_increase_interval (a b : ℝ) :
  (∀ x y, x < y → x ∈ Set.Iio 0 → y ∈ Set.Iio 0 → f x < f y) ∧
  (∀ x, x ∈ Set.Iic 0 → f x ≤ f 0) ∧
  (∀ x, x > 0 → f x < f 0) :=
sorry

end NUMINAMATH_CALUDE_monotonic_increase_interval_l1744_174404


namespace NUMINAMATH_CALUDE_brick_length_is_8_l1744_174415

-- Define the surface area function for a rectangular prism
def surface_area (l w h : ℝ) : ℝ := 2 * l * w + 2 * l * h + 2 * w * h

-- Theorem statement
theorem brick_length_is_8 :
  ∃ (l : ℝ), l > 0 ∧ surface_area l 6 2 = 152 ∧ l = 8 :=
by sorry

end NUMINAMATH_CALUDE_brick_length_is_8_l1744_174415


namespace NUMINAMATH_CALUDE_exact_one_second_class_probability_l1744_174484

/-- The probability of selecting exactly one second-class product when randomly
    selecting three products from a batch of 100 products containing 90 first-class
    and 10 second-class products. -/
theorem exact_one_second_class_probability
  (total : ℕ) (first_class : ℕ) (second_class : ℕ) (selected : ℕ)
  (h_total : total = 100)
  (h_first : first_class = 90)
  (h_second : second_class = 10)
  (h_selected : selected = 3)
  (h_sum : first_class + second_class = total) :
  (Nat.choose first_class 2 * Nat.choose second_class 1) / Nat.choose total selected = 267 / 1078 :=
sorry

end NUMINAMATH_CALUDE_exact_one_second_class_probability_l1744_174484


namespace NUMINAMATH_CALUDE_smallest_perimeter_consecutive_odd_triangle_l1744_174487

/-- Three consecutive odd integers -/
def ConsecutiveOddIntegers (a b c : ℕ) : Prop :=
  (∃ k : ℕ, a = 2 * k + 1 ∧ b = 2 * k + 3 ∧ c = 2 * k + 5)

/-- Triangle inequality -/
def IsValidTriangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ a + c > b

/-- Perimeter of a triangle -/
def Perimeter (a b c : ℕ) : ℕ := a + b + c

/-- The smallest possible perimeter of a triangle with consecutive odd integer side lengths is 15 -/
theorem smallest_perimeter_consecutive_odd_triangle :
  ∀ a b c : ℕ,
  ConsecutiveOddIntegers a b c →
  IsValidTriangle a b c →
  Perimeter a b c ≥ 15 :=
sorry

end NUMINAMATH_CALUDE_smallest_perimeter_consecutive_odd_triangle_l1744_174487


namespace NUMINAMATH_CALUDE_gcd_lcm_calculation_l1744_174409

theorem gcd_lcm_calculation (a b : ℕ) (ha : a = 84) (hb : b = 3780) :
  (Nat.gcd a b + Nat.lcm a b) * (Nat.lcm a b * Nat.gcd a b) - 
  (Nat.lcm a b * Nat.gcd a b) = 1227194880 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_calculation_l1744_174409


namespace NUMINAMATH_CALUDE_quadratic_cubic_relation_l1744_174418

theorem quadratic_cubic_relation (x₀ : ℝ) (h : x₀^2 + x₀ - 1 = 0) :
  x₀^3 + 2*x₀^2 + 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_cubic_relation_l1744_174418


namespace NUMINAMATH_CALUDE_cricket_average_l1744_174445

theorem cricket_average (A : ℝ) : 
  (11 * (A + 4) = 10 * A + 86) → A = 42 := by
  sorry

end NUMINAMATH_CALUDE_cricket_average_l1744_174445


namespace NUMINAMATH_CALUDE_price_increase_problem_l1744_174439

theorem price_increase_problem (candy_initial : ℝ) (soda_initial : ℝ) 
  (candy_increase : ℝ) (soda_increase : ℝ) 
  (h1 : candy_initial = 20) 
  (h2 : soda_initial = 6) 
  (h3 : candy_increase = 0.25) 
  (h4 : soda_increase = 0.50) : 
  candy_initial + soda_initial = 26 := by
  sorry

#check price_increase_problem

end NUMINAMATH_CALUDE_price_increase_problem_l1744_174439


namespace NUMINAMATH_CALUDE_equality_abs_condition_l1744_174434

theorem equality_abs_condition (x y : ℝ) : 
  (x = y → abs x = abs y) ∧ 
  ∃ a b : ℝ, abs a = abs b ∧ a ≠ b := by
sorry

end NUMINAMATH_CALUDE_equality_abs_condition_l1744_174434


namespace NUMINAMATH_CALUDE_grid_rectangles_l1744_174402

/-- The number of points in each row or column of the grid -/
def gridSize : ℕ := 4

/-- The number of ways to choose 2 items from n items -/
def choose2 (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The number of rectangles in a gridSize × gridSize grid -/
def numRectangles : ℕ := (choose2 gridSize) * (choose2 gridSize)

theorem grid_rectangles :
  numRectangles = 36 := by sorry

end NUMINAMATH_CALUDE_grid_rectangles_l1744_174402


namespace NUMINAMATH_CALUDE_log_eight_three_equals_512_l1744_174410

theorem log_eight_three_equals_512 (y : ℝ) :
  Real.log y / Real.log 8 = 3 → y = 512 := by
  sorry

end NUMINAMATH_CALUDE_log_eight_three_equals_512_l1744_174410


namespace NUMINAMATH_CALUDE_coffee_syrup_combinations_l1744_174469

theorem coffee_syrup_combinations :
  let coffee_types : ℕ := 5
  let syrup_types : ℕ := 7
  let syrup_choices : ℕ := 3
  coffee_types * (syrup_types.choose syrup_choices) = 175 :=
by sorry

end NUMINAMATH_CALUDE_coffee_syrup_combinations_l1744_174469


namespace NUMINAMATH_CALUDE_roberto_outfits_l1744_174499

/-- The number of different outfits Roberto can create -/
def number_of_outfits (trousers shirts jackets : ℕ) : ℕ :=
  trousers * shirts * jackets

/-- Theorem stating the number of outfits Roberto can create -/
theorem roberto_outfits :
  let trousers : ℕ := 5
  let shirts : ℕ := 8
  let jackets : ℕ := 4
  number_of_outfits trousers shirts jackets = 160 := by
  sorry

end NUMINAMATH_CALUDE_roberto_outfits_l1744_174499


namespace NUMINAMATH_CALUDE_distributeBallsWithRedBox_eq_1808_l1744_174451

/-- Number of ways to distribute n distinguishable balls into k distinguishable boxes -/
def distribute (n k : ℕ) : ℕ := k^n

/-- Number of ways to choose r items from n items -/
def choose (n r : ℕ) : ℕ := Nat.choose n r

/-- Number of ways to distribute 7 distinguishable balls into 3 distinguishable boxes,
    where one box (red) can contain at most 3 balls -/
def distributeBallsWithRedBox : ℕ :=
  choose 7 3 * distribute 4 2 +
  choose 7 2 * distribute 5 2 +
  choose 7 1 * distribute 6 2 +
  distribute 7 2

theorem distributeBallsWithRedBox_eq_1808 :
  distributeBallsWithRedBox = 1808 := by sorry

end NUMINAMATH_CALUDE_distributeBallsWithRedBox_eq_1808_l1744_174451


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1744_174456

theorem inequality_solution_set :
  ∀ x : ℝ, (1/2: ℝ)^(x - x^2) < Real.log 81 / Real.log 3 ↔ -1 < x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1744_174456


namespace NUMINAMATH_CALUDE_total_people_on_large_seats_l1744_174459

/-- The number of large seats on the Ferris wheel -/
def large_seats : ℕ := 7

/-- The number of people each large seat can hold -/
def people_per_large_seat : ℕ := 12

/-- Theorem: The total number of people who can ride on large seats is 84 -/
theorem total_people_on_large_seats : large_seats * people_per_large_seat = 84 := by
  sorry

end NUMINAMATH_CALUDE_total_people_on_large_seats_l1744_174459


namespace NUMINAMATH_CALUDE_median_in_70_74_l1744_174474

/-- Represents a score interval with its lower bound and student count -/
structure ScoreInterval :=
  (lower_bound : ℕ)
  (student_count : ℕ)

/-- The list of score intervals -/
def score_intervals : List ScoreInterval :=
  [⟨85, 10⟩, ⟨80, 15⟩, ⟨75, 20⟩, ⟨70, 25⟩, ⟨65, 15⟩, ⟨60, 10⟩, ⟨55, 5⟩]

/-- The total number of students -/
def total_students : ℕ := 100

/-- Find the interval containing the median score -/
def median_interval (intervals : List ScoreInterval) (total : ℕ) : Option ScoreInterval :=
  sorry

/-- Theorem: The interval containing the median score is 70-74 -/
theorem median_in_70_74 :
  median_interval score_intervals total_students = some ⟨70, 25⟩ :=
sorry

end NUMINAMATH_CALUDE_median_in_70_74_l1744_174474


namespace NUMINAMATH_CALUDE_least_sum_of_bases_l1744_174421

theorem least_sum_of_bases (c d : ℕ+) : 
  (6 * c.val + 5 = 5 * d.val + 6) →
  (∀ c' d' : ℕ+, (6 * c'.val + 5 = 5 * d'.val + 6) → c'.val + d'.val ≥ c.val + d.val) →
  c.val + d.val = 13 := by
sorry

end NUMINAMATH_CALUDE_least_sum_of_bases_l1744_174421


namespace NUMINAMATH_CALUDE_equation_solution_l1744_174476

theorem equation_solution (x y : ℝ) : 
  x / (x - 2) = (y^3 + 3*y - 2) / (y^3 + 3*y - 5) → 
  x = (2*y^3 + 6*y - 4) / 3 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l1744_174476


namespace NUMINAMATH_CALUDE_sin_2alpha_minus_cos_pi_minus_2alpha_l1744_174408

theorem sin_2alpha_minus_cos_pi_minus_2alpha (α : Real) (h : Real.tan α = 2/3) :
  Real.sin (2 * α) - Real.cos (π - 2 * α) = 17/13 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_minus_cos_pi_minus_2alpha_l1744_174408


namespace NUMINAMATH_CALUDE_polynomial_value_at_two_l1744_174496

/-- A polynomial with non-negative integer coefficients -/
def NonNegIntPoly (f : ℕ → ℕ) : Prop :=
  ∃ n : ℕ, ∀ k : ℕ, k > n → f k = 0

theorem polynomial_value_at_two
  (f : ℕ → ℕ)
  (h_poly : NonNegIntPoly f)
  (h_one : f 1 = 6)
  (h_seven : f 7 = 3438) :
  f 2 = 43 := by
sorry

end NUMINAMATH_CALUDE_polynomial_value_at_two_l1744_174496


namespace NUMINAMATH_CALUDE_infinitely_many_palindromes_l1744_174470

/-- A function that checks if a natural number is a palindrome in decimal representation -/
def isPalindrome (n : ℕ) : Prop := sorry

/-- The sequence defined in the problem -/
def x (n : ℕ) : ℕ := 2013 + 317 * n

/-- The main theorem to prove -/
theorem infinitely_many_palindromes :
  ∀ k : ℕ, ∃ n : ℕ, n ≥ k ∧ isPalindrome (x n) := by sorry

end NUMINAMATH_CALUDE_infinitely_many_palindromes_l1744_174470


namespace NUMINAMATH_CALUDE_common_difference_is_negative_three_l1744_174488

/-- An arithmetic sequence with given terms -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  third_term : a 3 = 7
  seventh_term : a 7 = -5

/-- The common difference of an arithmetic sequence -/
def common_difference (seq : ArithmeticSequence) : ℝ :=
  seq.a 2 - seq.a 1

theorem common_difference_is_negative_three (seq : ArithmeticSequence) :
  common_difference seq = -3 := by
  sorry

end NUMINAMATH_CALUDE_common_difference_is_negative_three_l1744_174488


namespace NUMINAMATH_CALUDE_tailor_cut_difference_l1744_174424

theorem tailor_cut_difference : 
  let skirt_cut : ℚ := 7/8
  let pants_cut : ℚ := 5/6
  skirt_cut - pants_cut = 1/24 := by sorry

end NUMINAMATH_CALUDE_tailor_cut_difference_l1744_174424


namespace NUMINAMATH_CALUDE_flower_shop_profit_l1744_174441

-- Define the profit function
def profit (n : ℕ) : ℤ :=
  if n < 16 then 10 * n - 80 else 80

-- Define the probability distribution
def prob (x : ℤ) : ℝ :=
  if x = 60 then 0.1
  else if x = 70 then 0.2
  else if x = 80 then 0.7
  else 0

-- Define the expected value
def expected_profit : ℝ :=
  60 * prob 60 + 70 * prob 70 + 80 * prob 80

-- Define the variance
def variance_profit : ℝ :=
  (60 - expected_profit)^2 * prob 60 +
  (70 - expected_profit)^2 * prob 70 +
  (80 - expected_profit)^2 * prob 80

-- Theorem statement
theorem flower_shop_profit :
  expected_profit = 76 ∧ variance_profit = 44 :=
sorry

end NUMINAMATH_CALUDE_flower_shop_profit_l1744_174441


namespace NUMINAMATH_CALUDE_nuts_division_proof_l1744_174444

/-- The number of boys dividing nuts -/
def num_boys : ℕ := 4

/-- The number of nuts each boy receives at the end -/
def nuts_per_boy : ℕ := 3 * num_boys

/-- The number of nuts taken by the nth boy -/
def nuts_taken (n : ℕ) : ℕ := 3 * n

/-- The remaining nuts after the nth boy's turn -/
def remaining_nuts (n : ℕ) : ℕ :=
  if n = num_boys then 0
  else 5 * (nuts_per_boy - nuts_taken n)

theorem nuts_division_proof :
  (∀ n : ℕ, n ≤ num_boys → nuts_per_boy = nuts_taken n + remaining_nuts n / 5) ∧
  remaining_nuts num_boys = 0 :=
sorry

end NUMINAMATH_CALUDE_nuts_division_proof_l1744_174444


namespace NUMINAMATH_CALUDE_polynomial_conclusions_l1744_174450

theorem polynomial_conclusions (x a : ℝ) : 
  let M : ℝ → ℝ := λ x => 2 * x^2 - 3 * x - 2
  let N : ℝ → ℝ := λ x => x^2 - a * x + 3
  (∃! i : Fin 3, 
    (i = 0 → (M x = 0 → (13 * x) / (x^2 - 3 * x - 1) = 26 / 3)) ∧
    (i = 1 → (a = -3 → (∀ y ≥ 4, M y - N y ≥ -14) → (∃ z ≥ 4, M z - N z = -14))) ∧
    (i = 2 → (a = 0 → (M x * N x = 0 → ∃ r s : ℝ, r ≠ s ∧ M r = 0 ∧ M s = 0))))
  := by sorry

end NUMINAMATH_CALUDE_polynomial_conclusions_l1744_174450


namespace NUMINAMATH_CALUDE_transform_is_right_shift_graph_transform_is_right_shift_l1744_174463

-- Define a continuous function f from reals to reals
variable (f : ℝ → ℝ) (hf : Continuous f)

-- Define the transformation function
def transform (f : ℝ → ℝ) : ℝ → ℝ := λ x ↦ f (x - 1)

-- Theorem stating that the transformation is equivalent to a right shift
theorem transform_is_right_shift :
  ∀ x y : ℝ, transform f x = y ↔ f (x - 1) = y :=
by sorry

-- Theorem stating that the graph of the transformed function
-- is equivalent to the original graph shifted 1 unit right
theorem graph_transform_is_right_shift :
  ∀ x y : ℝ, (x, y) ∈ (Set.range (λ x ↦ (x, transform f x))) ↔
             (x - 1, y) ∈ (Set.range (λ x ↦ (x, f x))) :=
by sorry

end NUMINAMATH_CALUDE_transform_is_right_shift_graph_transform_is_right_shift_l1744_174463


namespace NUMINAMATH_CALUDE_simplify_trig_expression_130_degrees_simplify_trig_expression_second_quadrant_l1744_174489

-- Part 1
theorem simplify_trig_expression_130_degrees :
  (Real.sqrt (1 - 2 * Real.sin (130 * π / 180) * Real.cos (130 * π / 180))) /
  (Real.sin (130 * π / 180) + Real.sqrt (1 - Real.sin (130 * π / 180) ^ 2)) = 1 := by
sorry

-- Part 2
theorem simplify_trig_expression_second_quadrant (α : Real) 
  (h : π / 2 < α ∧ α < π) :
  Real.cos α * Real.sqrt ((1 - Real.sin α) / (1 + Real.sin α)) +
  Real.sin α * Real.sqrt ((1 - Real.cos α) / (1 + Real.cos α)) =
  Real.sin α - Real.cos α := by
sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_130_degrees_simplify_trig_expression_second_quadrant_l1744_174489


namespace NUMINAMATH_CALUDE_sunlight_is_ray_telephone_line_is_segment_l1744_174420

-- Define the different types of lines
inductive LineType
  | Ray
  | LineSegment
  | StraightLine

-- Define the number of endpoints for each line type
def numberOfEndpoints (lt : LineType) : Nat :=
  match lt with
  | .Ray => 1
  | .LineSegment => 2
  | .StraightLine => 0

-- Define the light emitted by the sun
def sunlight : LineType := LineType.Ray

-- Define the line between telephone poles
def telephoneLine : LineType := LineType.LineSegment

-- Theorem stating that the light emitted by the sun is a ray
theorem sunlight_is_ray : sunlight = LineType.Ray := by sorry

-- Theorem stating that the line between telephone poles is a line segment
theorem telephone_line_is_segment : telephoneLine = LineType.LineSegment := by sorry

end NUMINAMATH_CALUDE_sunlight_is_ray_telephone_line_is_segment_l1744_174420


namespace NUMINAMATH_CALUDE_sales_minimum_value_l1744_174435

/-- A quadratic function f(x) representing monthly sales -/
def f (p q : ℝ) (x : ℝ) : ℝ := x^2 + p*x + q

/-- The theorem stating the minimum value of the sales function -/
theorem sales_minimum_value (p q : ℝ) 
  (h1 : f p q 1 = 10) 
  (h2 : f p q 3 = 2) : 
  ∃ x, ∀ y, f p q x ≤ f p q y ∧ f p q x = -1/4 :=
sorry

end NUMINAMATH_CALUDE_sales_minimum_value_l1744_174435


namespace NUMINAMATH_CALUDE_company_workforce_l1744_174481

theorem company_workforce (initial_employees : ℕ) 
  (h1 : (60 : ℚ) / 100 * initial_employees = (55 : ℚ) / 100 * (initial_employees + 30)) :
  initial_employees + 30 = 360 := by
sorry

end NUMINAMATH_CALUDE_company_workforce_l1744_174481


namespace NUMINAMATH_CALUDE_class_average_problem_l1744_174419

theorem class_average_problem (x : ℝ) : 
  let total_students : ℕ := 20
  let group1_students : ℕ := 10
  let group2_students : ℕ := 10
  let group2_average : ℝ := 60
  let class_average : ℝ := 70
  (group1_students : ℝ) * x + (group2_students : ℝ) * group2_average = 
    (total_students : ℝ) * class_average → x = 80 := by
  sorry

end NUMINAMATH_CALUDE_class_average_problem_l1744_174419


namespace NUMINAMATH_CALUDE_theorem_1_theorem_2_l1744_174458

-- Define proposition p
def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0

-- Define proposition q
def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

-- Theorem 1
theorem theorem_1 (a : ℝ) : p a → a ≤ 1 := by sorry

-- Theorem 2
theorem theorem_2 (a : ℝ) : ¬(p a ∧ q a) → a ∈ Set.Ioo (-2) 1 ∪ Set.Ioi 1 := by sorry

end NUMINAMATH_CALUDE_theorem_1_theorem_2_l1744_174458


namespace NUMINAMATH_CALUDE_trapezoid_perimeter_l1744_174403

/-- A trapezoid with sides A, B, C, and D -/
structure Trapezoid where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ

/-- The perimeter of a trapezoid -/
def perimeter (t : Trapezoid) : ℝ := t.A + t.B + t.C + t.D

/-- Theorem: The perimeter of the given trapezoid ABCD is 180 units -/
theorem trapezoid_perimeter : 
  ∀ (ABCD : Trapezoid), 
  ABCD.B = 50 → 
  ABCD.A = 30 → 
  ABCD.C = 25 → 
  ABCD.D = 75 → 
  perimeter ABCD = 180 := by
  sorry


end NUMINAMATH_CALUDE_trapezoid_perimeter_l1744_174403


namespace NUMINAMATH_CALUDE_exam_candidates_count_l1744_174475

theorem exam_candidates_count :
  ∀ (T : ℕ),
    (T : ℚ) * (49 / 100) = T * (percent_failed_english : ℚ) →
    (T : ℚ) * (36 / 100) = T * (percent_failed_hindi : ℚ) →
    (T : ℚ) * (15 / 100) = T * (percent_failed_both : ℚ) →
    (T : ℚ) * ((51 / 100) - (15 / 100)) = 630 →
    T = 1750 :=
by
  sorry

end NUMINAMATH_CALUDE_exam_candidates_count_l1744_174475


namespace NUMINAMATH_CALUDE_parallel_line_slope_l1744_174453

/-- A point P with coordinates depending on a parameter p -/
def P (p : ℝ) : ℝ × ℝ := (2*p, -4*p + 1)

/-- The line y = kx + 2 -/
def line (k : ℝ) (x : ℝ) : ℝ := k*x + 2

/-- The theorem stating that k must be -2 for the line to be parallel to the locus of P -/
theorem parallel_line_slope (k : ℝ) : 
  (∀ p : ℝ, P p ∉ {xy : ℝ × ℝ | xy.2 = line k xy.1}) → k = -2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_slope_l1744_174453


namespace NUMINAMATH_CALUDE_rectangle_diagonal_l1744_174423

/-- Given a rectangle with perimeter 72 meters and length-to-width ratio 5:2,
    prove that its diagonal length is 194/7 meters. -/
theorem rectangle_diagonal (length width : ℝ) : 
  (2 * (length + width) = 72) →
  (length / width = 5 / 2) →
  Real.sqrt (length^2 + width^2) = 194 / 7 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_l1744_174423


namespace NUMINAMATH_CALUDE_remainder_problem_l1744_174446

theorem remainder_problem (N : ℕ) (h1 : N = 184) (h2 : N % 15 = 4) : N % 13 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l1744_174446


namespace NUMINAMATH_CALUDE_spider_legs_count_l1744_174412

/-- The number of spiders in the room -/
def num_spiders : ℕ := 4

/-- The number of legs each spider has -/
def legs_per_spider : ℕ := 8

/-- The total number of spider legs in the room -/
def total_legs : ℕ := num_spiders * legs_per_spider

theorem spider_legs_count : total_legs = 32 := by
  sorry

end NUMINAMATH_CALUDE_spider_legs_count_l1744_174412


namespace NUMINAMATH_CALUDE_smallest_difference_ef_de_l1744_174432

/-- Represents a triangle with integer side lengths --/
structure Triangle where
  de : ℕ
  ef : ℕ
  fd : ℕ

/-- Checks if the given lengths satisfy the triangle inequality --/
def is_valid_triangle (t : Triangle) : Prop :=
  t.de + t.ef > t.fd ∧ t.ef + t.fd > t.de ∧ t.fd + t.de > t.ef

/-- Theorem stating the smallest possible difference between EF and DE --/
theorem smallest_difference_ef_de (t : Triangle) : 
  t.de < t.ef ∧ t.ef ≤ t.fd ∧ 
  t.de + t.ef + t.fd = 1024 ∧
  is_valid_triangle t →
  ∀ (t' : Triangle), 
    t'.de < t'.ef ∧ t'.ef ≤ t'.fd ∧
    t'.de + t'.ef + t'.fd = 1024 ∧
    is_valid_triangle t' →
    t.ef - t.de ≤ t'.ef - t'.de ∧
    t.ef - t.de = 1 :=
by sorry

end NUMINAMATH_CALUDE_smallest_difference_ef_de_l1744_174432


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_one_l1744_174491

theorem fraction_zero_implies_x_equals_one (x : ℝ) :
  (x^2 - 1) / (x + 1) = 0 ∧ x + 1 ≠ 0 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_one_l1744_174491


namespace NUMINAMATH_CALUDE_number_of_large_boats_proof_number_of_large_boats_l1744_174448

theorem number_of_large_boats (total_students : ℕ) (total_boats : ℕ) 
  (large_boat_capacity : ℕ) (small_boat_capacity : ℕ) : ℕ :=
  let number_of_large_boats := 
    total_boats - (total_students - large_boat_capacity * total_boats) / 
      (large_boat_capacity - small_boat_capacity)
  number_of_large_boats

#check number_of_large_boats 50 10 6 4 = 5

theorem proof_number_of_large_boats :
  number_of_large_boats 50 10 6 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_number_of_large_boats_proof_number_of_large_boats_l1744_174448


namespace NUMINAMATH_CALUDE_all_options_incorrect_l1744_174411

-- Define the types for functions
def Function := ℝ → ℝ

-- Define properties of functions
def Periodic (f : Function) : Prop := 
  ∃ T > 0, ∀ x, f (x + T) = f x

def Monotonic (f : Function) : Prop := 
  ∀ x y, x < y → f x < f y

-- Original proposition
def OriginalProposition : Prop :=
  ∀ f : Function, Periodic f → ¬(Monotonic f)

-- Theorem to prove
theorem all_options_incorrect (original : OriginalProposition) : 
  (¬(∀ f : Function, Monotonic f → ¬(Periodic f))) ∧ 
  (¬(∀ f : Function, Periodic f → Monotonic f)) ∧ 
  (¬(∀ f : Function, Monotonic f → Periodic f)) :=
sorry

end NUMINAMATH_CALUDE_all_options_incorrect_l1744_174411


namespace NUMINAMATH_CALUDE_set_relationship_l1744_174480

theorem set_relationship (A B C : Set α) 
  (h1 : A ∩ B = C) 
  (h2 : B ∩ C = A) : 
  A = C ∧ A ⊆ B := by
sorry

end NUMINAMATH_CALUDE_set_relationship_l1744_174480


namespace NUMINAMATH_CALUDE_inscribed_circle_area_ratio_l1744_174436

/-- A circle inscribed in a convex polygon -/
structure InscribedCircle where
  /-- The circumference of the inscribed circle -/
  circle_circumference : ℝ
  /-- The perimeter of the convex polygon -/
  polygon_perimeter : ℝ
  /-- The area of the inscribed circle -/
  circle_area : ℝ
  /-- The area of the convex polygon -/
  polygon_area : ℝ

/-- Theorem stating that for a circle inscribed in a convex polygon with given circumference and perimeter,
    the ratio of the circle's area to the polygon's area is 2/3 -/
theorem inscribed_circle_area_ratio
  (ic : InscribedCircle)
  (h_circle_circumference : ic.circle_circumference = 10)
  (h_polygon_perimeter : ic.polygon_perimeter = 15) :
  ic.circle_area / ic.polygon_area = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_area_ratio_l1744_174436


namespace NUMINAMATH_CALUDE_tony_remaining_money_l1744_174457

/-- Calculates the remaining money after expenses -/
def remaining_money (initial : ℕ) (ticket : ℕ) (hot_dog : ℕ) (soda : ℕ) : ℕ :=
  initial - ticket - hot_dog - soda

/-- Proves that Tony has $26 left after his expenses -/
theorem tony_remaining_money :
  remaining_money 50 15 5 4 = 26 := by
  sorry

#eval remaining_money 50 15 5 4

end NUMINAMATH_CALUDE_tony_remaining_money_l1744_174457


namespace NUMINAMATH_CALUDE_expression_evaluation_l1744_174498

theorem expression_evaluation (a b : ℤ) (ha : a = -1) (hb : b = 4) :
  (a + b)^2 - 2*a*(a - b) + (a + 2*b)*(a - 2*b) = -64 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1744_174498


namespace NUMINAMATH_CALUDE_usual_bus_time_l1744_174422

/-- Proves that the usual time to catch the bus is 12 minutes, given that walking
    at 4/5 of the usual speed results in missing the bus by 3 minutes. -/
theorem usual_bus_time (T : ℝ) (h : (5 / 4) * T = T + 3) : T = 12 := by
  sorry

end NUMINAMATH_CALUDE_usual_bus_time_l1744_174422


namespace NUMINAMATH_CALUDE_function_value_sum_l1744_174467

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem function_value_sum (f : ℝ → ℝ) 
    (h_periodic : is_periodic f 2)
    (h_odd : is_odd f)
    (h_interval : ∀ x, 0 < x → x < 1 → f x = 4^x) :
  f (-5/2) + f 2 = -2 := by
  sorry


end NUMINAMATH_CALUDE_function_value_sum_l1744_174467


namespace NUMINAMATH_CALUDE_expression_evaluation_l1744_174477

theorem expression_evaluation (x y z : ℝ) :
  let P := x + y + z
  let Q := x - y - z
  (P + Q) / (P - Q) - (P - Q) / (P + Q) = (x^2 - y^2 - 2*y*z - z^2) / (x*(y + z)) :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1744_174477


namespace NUMINAMATH_CALUDE_binary_11111_equals_31_l1744_174494

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_11111_equals_31 :
  binary_to_decimal [true, true, true, true, true] = 31 := by
  sorry

end NUMINAMATH_CALUDE_binary_11111_equals_31_l1744_174494


namespace NUMINAMATH_CALUDE_traffic_light_probability_l1744_174482

theorem traffic_light_probability (m : ℕ) : 
  (35 : ℝ) / (38 + m) > (m : ℝ) / (38 + m) → m = 30 :=
by sorry

end NUMINAMATH_CALUDE_traffic_light_probability_l1744_174482


namespace NUMINAMATH_CALUDE_price_reduction_sales_increase_l1744_174455

theorem price_reduction_sales_increase (price_reduction : Real) 
  (sales_increase : Real) (net_sale_increase : Real) :
  price_reduction = 20 → 
  net_sale_increase = 44 → 
  (1 - price_reduction / 100) * (1 + sales_increase / 100) = 1 + net_sale_increase / 100 →
  sales_increase = 80 := by
sorry

end NUMINAMATH_CALUDE_price_reduction_sales_increase_l1744_174455


namespace NUMINAMATH_CALUDE_projection_property_l1744_174497

def projection (v : ℝ × ℝ) (w : ℝ × ℝ) : ℝ × ℝ := sorry

theorem projection_property :
  ∀ (p : (ℝ × ℝ) → (ℝ × ℝ)),
  (p (2, -4) = (3, -3)) →
  (p = projection (1, -1)) →
  (p (-8, 2) = (-5, 5)) := by sorry

end NUMINAMATH_CALUDE_projection_property_l1744_174497


namespace NUMINAMATH_CALUDE_quadratic_roots_l1744_174440

theorem quadratic_roots : ∃ x₁ x₂ : ℝ, 
  (x₁^2 + x₁ = 0 ∧ x₂^2 + x₂ = 0) ∧ 
  x₁ = 0 ∧ x₂ = -1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_l1744_174440


namespace NUMINAMATH_CALUDE_negative_125_to_four_thirds_l1744_174407

theorem negative_125_to_four_thirds : (-125 : ℝ) ^ (4/3) = 625 := by sorry

end NUMINAMATH_CALUDE_negative_125_to_four_thirds_l1744_174407


namespace NUMINAMATH_CALUDE_total_ways_eq_64_l1744_174483

/-- The number of sports available to choose from -/
def num_sports : ℕ := 4

/-- The number of people choosing sports -/
def num_people : ℕ := 3

/-- The total number of different ways to choose sports -/
def total_ways : ℕ := num_sports ^ num_people

/-- Theorem stating that the total number of ways to choose sports is 64 -/
theorem total_ways_eq_64 : total_ways = 64 := by
  sorry

end NUMINAMATH_CALUDE_total_ways_eq_64_l1744_174483


namespace NUMINAMATH_CALUDE_correct_average_weight_l1744_174472

theorem correct_average_weight 
  (n : ℕ) 
  (initial_average : ℝ) 
  (misread_weight : ℝ) 
  (correct_weight : ℝ) :
  n = 20 →
  initial_average = 58.4 →
  misread_weight = 56 →
  correct_weight = 61 →
  (n * initial_average + (correct_weight - misread_weight)) / n = 58.65 :=
by sorry

end NUMINAMATH_CALUDE_correct_average_weight_l1744_174472


namespace NUMINAMATH_CALUDE_cloud_counting_proof_l1744_174449

def carson_clouds : ℕ := 6

def brother_clouds : ℕ := 3 * carson_clouds

def total_clouds : ℕ := carson_clouds + brother_clouds

theorem cloud_counting_proof : total_clouds = 24 := by
  sorry

end NUMINAMATH_CALUDE_cloud_counting_proof_l1744_174449


namespace NUMINAMATH_CALUDE_gcd_of_specific_numbers_l1744_174427

def m : ℕ := 55555555
def n : ℕ := 5555555555

theorem gcd_of_specific_numbers : Nat.gcd m n = 5 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_specific_numbers_l1744_174427


namespace NUMINAMATH_CALUDE_remainder_of_permutation_number_l1744_174405

-- Define a type for permutations of numbers from 1 to 2018
def Permutation := Fin 2018 → Fin 2018

-- Define a function that creates a number from a permutation
def numberFromPermutation (p : Permutation) : ℕ := sorry

-- Theorem statement
theorem remainder_of_permutation_number (p : Permutation) :
  numberFromPermutation p % 3 = 0 := by sorry

end NUMINAMATH_CALUDE_remainder_of_permutation_number_l1744_174405


namespace NUMINAMATH_CALUDE_average_of_a_and_b_l1744_174479

theorem average_of_a_and_b (a b : ℝ) : 
  (4 + 5 + 9 + a + b) / 5 = 18 → (a + b) / 2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_average_of_a_and_b_l1744_174479


namespace NUMINAMATH_CALUDE_ones_digit_of_6_to_34_l1744_174425

theorem ones_digit_of_6_to_34 : ∃ k : ℕ, 6^34 = 10 * k + 6 := by
  sorry

end NUMINAMATH_CALUDE_ones_digit_of_6_to_34_l1744_174425


namespace NUMINAMATH_CALUDE_angle_sum_theorem_l1744_174447

-- Define the angles
variable (A B C D E F : ℝ)

-- Define the theorem
theorem angle_sum_theorem (h : A + B + C + D + E + F = 90 * n) : n = 4 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_theorem_l1744_174447


namespace NUMINAMATH_CALUDE_smallest_x_for_prime_abs_f_l1744_174400

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def f (x : ℤ) : ℤ := 4 * x^2 - 34 * x + 21

theorem smallest_x_for_prime_abs_f :
  ∃ (x : ℤ), (∀ (y : ℤ), y < x → ¬(is_prime (Int.natAbs (f y)))) ∧
             (is_prime (Int.natAbs (f x))) ∧
             x = 1 := by
  sorry

end NUMINAMATH_CALUDE_smallest_x_for_prime_abs_f_l1744_174400


namespace NUMINAMATH_CALUDE_finite_solutions_l1744_174461

/-- The function F_{n,k}(x,y) as defined in the problem -/
def F (n k x y : ℕ) : ℤ := (Nat.factorial x : ℤ) + n^k + n + 1 - y^k

/-- Theorem stating that the set of solutions is finite -/
theorem finite_solutions (n k : ℕ) (hn : n > 0) (hk : k > 1) :
  Set.Finite {p : ℕ × ℕ | F n k p.1 p.2 = 0 ∧ p.1 > 0 ∧ p.2 > 0} :=
sorry

end NUMINAMATH_CALUDE_finite_solutions_l1744_174461


namespace NUMINAMATH_CALUDE_consecutive_integers_around_sqrt3_l1744_174429

theorem consecutive_integers_around_sqrt3 (a b : ℤ) : 
  (b = a + 1) → (a < Real.sqrt 3) → (Real.sqrt 3 < b) → (a + b = 3) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_around_sqrt3_l1744_174429


namespace NUMINAMATH_CALUDE_consecutive_integers_problem_l1744_174466

theorem consecutive_integers_problem (x y z : ℤ) :
  (x = y + 1) →  -- x, y are consecutive
  (y = z + 1) →  -- y, z are consecutive
  (x > y) →
  (y > z) →
  (2 * x + 3 * y + 3 * z = 5 * y + 11) →
  (z = 3) →
  (5 * y = 20) := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_problem_l1744_174466


namespace NUMINAMATH_CALUDE_parts_production_proportion_l1744_174433

/-- The relationship between parts produced per minute and total parts is direct proportion -/
theorem parts_production_proportion (parts_per_minute parts_total : ℝ → ℝ) (t : ℝ) :
  (∀ t, parts_total t = (parts_per_minute t) * t) →
  ∃ k : ℝ, ∀ t, parts_total t = k * (parts_per_minute t) := by
  sorry

end NUMINAMATH_CALUDE_parts_production_proportion_l1744_174433


namespace NUMINAMATH_CALUDE_negation_of_existence_of_real_roots_l1744_174413

theorem negation_of_existence_of_real_roots :
  (¬ ∃ m : ℝ, ∃ x : ℝ, x^2 + m*x + 1 = 0) ↔
  (∀ m : ℝ, ∀ x : ℝ, x^2 + m*x + 1 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_of_real_roots_l1744_174413


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l1744_174401

/-- The equation of a hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop := y^2 / 3 - x^2 = 1

/-- The equation of the asymptotes -/
def asymptote_equation (x y : ℝ) : Prop := y = x * Real.sqrt 3 ∨ y = -x * Real.sqrt 3

/-- Theorem: The asymptotes of the given hyperbola are y = ±√3x -/
theorem hyperbola_asymptotes :
  ∀ x y : ℝ, hyperbola_equation x y → asymptote_equation x y := by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l1744_174401


namespace NUMINAMATH_CALUDE_air_quality_consecutive_good_days_l1744_174495

/-- Represents the air quality index for a given day -/
def AirQualityIndex := ℕ → ℝ

/-- Determines if the air quality is good for a given index -/
def is_good (index : ℝ) : Prop := index < 100

/-- Determines if two consecutive days have good air quality -/
def consecutive_good_days (aqi : AirQualityIndex) (day : ℕ) : Prop :=
  is_good (aqi day) ∧ is_good (aqi (day + 1))

/-- The air quality index for the 10 days -/
axiom aqi : AirQualityIndex

/-- The theorem to prove -/
theorem air_quality_consecutive_good_days :
  (consecutive_good_days aqi 1 ∧ consecutive_good_days aqi 5) ∧
  (∀ d : ℕ, d ≠ 1 ∧ d ≠ 5 → ¬consecutive_good_days aqi d) :=
sorry

end NUMINAMATH_CALUDE_air_quality_consecutive_good_days_l1744_174495


namespace NUMINAMATH_CALUDE_mans_age_fraction_l1744_174406

theorem mans_age_fraction (mans_age father_age : ℕ) : 
  father_age = 25 →
  mans_age + 5 = (father_age + 5) / 2 →
  mans_age / father_age = 2 / 5 := by
sorry

end NUMINAMATH_CALUDE_mans_age_fraction_l1744_174406


namespace NUMINAMATH_CALUDE_geometric_sequence_solution_l1744_174452

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_solution
  (a : ℕ → ℝ)
  (h_geometric : is_geometric_sequence a)
  (h_sum : a 1 + a 2 + a 3 = 21)
  (h_product : a 1 * a 2 * a 3 = 216)
  (h_product_35 : a 3 * a 5 = 18)
  (h_product_48 : a 4 * a 8 = 72) :
  (∃ n : ℕ, a n = 3 * 2^(n-1) ∨ a n = 12 * (1/2)^(n-1)) ∧
  (∃ q : ℝ, q = Real.sqrt 2 ∨ q = -Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_solution_l1744_174452


namespace NUMINAMATH_CALUDE_balls_in_urns_l1744_174460

/-- The number of ways to place k identical balls into n urns with at most one ball per urn -/
def place_balls_limited (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of ways to place k identical balls into n urns with unlimited balls per urn -/
def place_balls_unlimited (n k : ℕ) : ℕ := Nat.choose (k+n-1) (n-1)

theorem balls_in_urns (n k : ℕ) :
  (place_balls_limited n k = Nat.choose n k) ∧
  (place_balls_unlimited n k = Nat.choose (k+n-1) (n-1)) := by
  sorry

end NUMINAMATH_CALUDE_balls_in_urns_l1744_174460


namespace NUMINAMATH_CALUDE_distance_between_vertices_l1744_174454

-- Define the two quadratic functions
def f (x : ℝ) : ℝ := x^2 - 4*x + 5
def g (x : ℝ) : ℝ := x^2 + 6*x + 20

-- Define the vertices of the two graphs
def C : ℝ × ℝ := (2, f 2)
def D : ℝ × ℝ := (-3, g (-3))

-- Theorem statement
theorem distance_between_vertices : 
  Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) = 5 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_vertices_l1744_174454


namespace NUMINAMATH_CALUDE_parabola_intersection_dot_product_l1744_174462

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola of the form y^2 = 2px -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Represents a line of the form y = mx + b -/
structure Line where
  m : ℝ
  b : ℝ

def Parabola.contains (c : Parabola) (pt : Point) : Prop :=
  pt.y^2 = 2 * c.p * pt.x

def Line.contains (l : Line) (pt : Point) : Prop :=
  pt.y = l.m * pt.x + l.b

def dotProduct (a b : Point) : ℝ :=
  a.x * b.x + a.y * b.y

theorem parabola_intersection_dot_product 
  (c : Parabola)
  (l : Line)
  (h1 : c.contains ⟨2, -2⟩)
  (h2 : l.m = 1 ∧ l.b = -1)
  (A B : Point)
  (h3 : c.contains A ∧ l.contains A)
  (h4 : c.contains B ∧ l.contains B)
  (h5 : A ≠ B) :
  dotProduct A B = -1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_dot_product_l1744_174462


namespace NUMINAMATH_CALUDE_smallest_a_value_l1744_174486

theorem smallest_a_value (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) 
  (h3 : ∀ x : ℤ, Real.sin (a * (x : ℝ) + b) = Real.sin (17 * (x : ℝ))) : 
  a ≥ 17 ∧ ∃ (a₀ : ℝ), a₀ ≥ 17 ∧ (∀ a' ≥ 17, (∀ x : ℤ, Real.sin (a' * (x : ℝ) + b) = Real.sin (17 * (x : ℝ))) → a' ≥ a₀) :=
sorry

end NUMINAMATH_CALUDE_smallest_a_value_l1744_174486


namespace NUMINAMATH_CALUDE_abs_of_complex_fraction_l1744_174473

open Complex

theorem abs_of_complex_fraction : 
  let z : ℂ := (4 - 2*I) / (1 + I)
  abs z = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_abs_of_complex_fraction_l1744_174473


namespace NUMINAMATH_CALUDE_equation_solution_l1744_174416

theorem equation_solution : 
  ∃! x : ℝ, x > 0 ∧ 7.61 * Real.log 3 / Real.log 2 + 2 * Real.log x / Real.log 4 = x^(Real.log 16 / Real.log 9 / (Real.log x / Real.log 3)) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1744_174416


namespace NUMINAMATH_CALUDE_jumps_per_meter_l1744_174490

/-- Given the relationships between different units of length, 
    this theorem proves how many jumps are in one meter. -/
theorem jumps_per_meter 
  (x y a b p q s t : ℚ) 
  (hops_to_skips : x * 1 = y)
  (jumps_to_hops : a = b * 1)
  (skips_to_leaps : p * 1 = q)
  (leaps_to_meters : s = t * 1)
  (x_pos : 0 < x) (y_pos : 0 < y) (a_pos : 0 < a) (b_pos : 0 < b)
  (p_pos : 0 < p) (q_pos : 0 < q) (s_pos : 0 < s) (t_pos : 0 < t) :
  1 = (s * p * x * a) / (t * q * y * b) :=
sorry

end NUMINAMATH_CALUDE_jumps_per_meter_l1744_174490


namespace NUMINAMATH_CALUDE_cube_root_identity_l1744_174442

theorem cube_root_identity : (2^3 * 5^6 * 7^3 : ℝ)^(1/3 : ℝ) = 350 := by sorry

end NUMINAMATH_CALUDE_cube_root_identity_l1744_174442


namespace NUMINAMATH_CALUDE_jimmy_garden_servings_l1744_174493

/-- Represents the number of servings produced by a single plant of each vegetable type -/
structure ServingsPerPlant where
  carrot : ℕ
  corn : ℕ
  greenBean : ℕ
  tomato : ℕ
  zucchini : ℕ
  bellPepper : ℕ

/-- Represents the number of plants for each vegetable type in Jimmy's garden -/
structure PlantsPerPlot where
  carrot : ℕ
  corn : ℕ
  greenBean : ℕ
  tomato : ℕ
  zucchini : ℕ
  bellPepper : ℕ

/-- Calculates the total number of servings in Jimmy's garden -/
def totalServings (s : ServingsPerPlant) (p : PlantsPerPlot) : ℕ :=
  s.carrot * p.carrot +
  s.corn * p.corn +
  s.greenBean * p.greenBean +
  s.tomato * p.tomato +
  s.zucchini * p.zucchini +
  s.bellPepper * p.bellPepper

/-- Theorem stating that Jimmy's garden produces 963 servings of vegetables -/
theorem jimmy_garden_servings 
  (s : ServingsPerPlant)
  (p : PlantsPerPlot)
  (h1 : s.carrot = 4)
  (h2 : s.corn = 5 * s.carrot)
  (h3 : s.greenBean = s.corn / 2)
  (h4 : s.tomato = s.carrot + 3)
  (h5 : s.zucchini = 4 * s.greenBean)
  (h6 : s.bellPepper = s.corn - 2)
  (h7 : p.greenBean = 10)
  (h8 : p.carrot = 8)
  (h9 : p.corn = 12)
  (h10 : p.tomato = 15)
  (h11 : p.zucchini = 9)
  (h12 : p.bellPepper = 7) :
  totalServings s p = 963 := by
  sorry


end NUMINAMATH_CALUDE_jimmy_garden_servings_l1744_174493


namespace NUMINAMATH_CALUDE_min_value_theorem_l1744_174414

theorem min_value_theorem (m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : 4*m + n = 1) :
  (4/m + 1/n) ≥ 25 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1744_174414

import Mathlib

namespace distinct_numbers_ratio_l2598_259825

theorem distinct_numbers_ratio (a b c : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : c ≠ a) 
  (h4 : (b - a)^2 - 4*(b - c)*(c - a) = 0) : 
  (b - c) / (c - a) = -1 := by
  sorry

end distinct_numbers_ratio_l2598_259825


namespace jovana_shells_added_l2598_259820

/-- The amount of shells added to a bucket -/
def shells_added (initial_amount final_amount : ℕ) : ℕ :=
  final_amount - initial_amount

/-- Theorem: The amount of shells Jovana added is 23 pounds -/
theorem jovana_shells_added :
  let initial_amount : ℕ := 5
  let final_amount : ℕ := 28
  shells_added initial_amount final_amount = 23 := by
  sorry

end jovana_shells_added_l2598_259820


namespace tan_triple_angle_l2598_259871

theorem tan_triple_angle (θ : ℝ) (h : Real.tan θ = 3) : Real.tan (3 * θ) = 9 / 13 := by
  sorry

end tan_triple_angle_l2598_259871


namespace flavoring_corn_syrup_ratio_comparison_l2598_259893

/-- Represents the ratio of flavoring to corn syrup to water in a drink formulation -/
structure DrinkRatio :=
  (flavoring : ℚ)
  (corn_syrup : ℚ)
  (water : ℚ)

/-- The standard formulation of the drink -/
def standard_formulation : DrinkRatio :=
  { flavoring := 1, corn_syrup := 12, water := 30 }

/-- The sport formulation of the drink -/
def sport_formulation : DrinkRatio :=
  { flavoring := 1.25, corn_syrup := 5, water := 75 }

/-- The ratio of flavoring to water in the sport formulation is half that of the standard formulation -/
axiom sport_water_ratio : 
  sport_formulation.flavoring / sport_formulation.water = 
  (standard_formulation.flavoring / standard_formulation.water) / 2

/-- The theorem to be proved -/
theorem flavoring_corn_syrup_ratio_comparison : 
  (sport_formulation.flavoring / sport_formulation.corn_syrup) / 
  (standard_formulation.flavoring / standard_formulation.corn_syrup) = 3 := by
  sorry

end flavoring_corn_syrup_ratio_comparison_l2598_259893


namespace min_value_expression_l2598_259877

theorem min_value_expression (x y z : ℝ) 
  (hx : -1 < x ∧ x < 1) 
  (hy : -1 < y ∧ y < 1) 
  (hz : -1 < z ∧ z < 1) : 
  1 / ((1 - x) * (1 - y) * (1 - z)) + 1 / ((1 + x) * (1 + y) * (1 + z)) ≥ 2 ∧
  (x = 0 ∧ y = 0 ∧ z = 0 → 1 / ((1 - x) * (1 - y) * (1 - z)) + 1 / ((1 + x) * (1 + y) * (1 + z)) = 2) :=
by sorry

end min_value_expression_l2598_259877


namespace quadratic_solution_property_l2598_259834

theorem quadratic_solution_property (a : ℝ) : 
  a^2 - 2*a - 1 = 0 → 2*a^2 - 4*a + 2022 = 2024 := by
  sorry

end quadratic_solution_property_l2598_259834


namespace derivative_sin_squared_minus_cos_squared_l2598_259847

theorem derivative_sin_squared_minus_cos_squared (x : ℝ) : 
  deriv (λ x => Real.sin x ^ 2 - Real.cos x ^ 2) x = 2 * Real.sin (2 * x) := by
  sorry

end derivative_sin_squared_minus_cos_squared_l2598_259847


namespace min_value_sum_reciprocals_l2598_259887

theorem min_value_sum_reciprocals (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_one : a + b + c + d = 1) :
  1 / (a + b) + 1 / (a + c) + 1 / (a + d) + 1 / (b + c) + 1 / (b + d) + 1 / (c + d) ≥ 18 ∧
  (1 / (a + b) + 1 / (a + c) + 1 / (a + d) + 1 / (b + c) + 1 / (b + d) + 1 / (c + d) = 18 ↔ 
   a = 1/4 ∧ b = 1/4 ∧ c = 1/4 ∧ d = 1/4) :=
by sorry

end min_value_sum_reciprocals_l2598_259887


namespace conditional_inequality_l2598_259815

theorem conditional_inequality (a b c : ℝ) (h1 : c > 0) (h2 : a * c^2 > b * c^2) : a > b := by
  sorry

end conditional_inequality_l2598_259815


namespace staircase_covering_l2598_259873

/-- A staircase tile with dimensions 6 × 1 -/
structure StaircaseTile where
  length : Nat := 6
  width : Nat := 1

/-- Predicate to check if a field can be covered with staircase tiles -/
def canCoverField (m n : Nat) : Prop :=
  ∃ (a b c d : Nat), 
    ((m = 12 * a ∧ n ≥ b ∧ b ≥ 6) ∨ 
     (n = 12 * a ∧ m ≥ b ∧ b ≥ 6) ∨
     (m = 3 * c ∧ n = 4 * d ∧ c ≥ 2 ∧ d ≥ 3) ∨
     (n = 3 * c ∧ m = 4 * d ∧ c ≥ 2 ∧ d ≥ 3))

theorem staircase_covering (m n : Nat) (hm : m ≥ 6) (hn : n ≥ 6) :
  canCoverField m n ↔ 
    ∃ (tiles : List StaircaseTile), 
      (tiles.length * 6 = m * n) ∧ 
      (∀ t ∈ tiles, t.length = 6 ∧ t.width = 1) :=
by sorry

end staircase_covering_l2598_259873


namespace slope_angle_of_parametric_line_l2598_259880

/-- Slope angle of a line with given parametric equations -/
theorem slope_angle_of_parametric_line :
  ∀ (t : ℝ),
  let x := -3 + t
  let y := 1 + Real.sqrt 3 * t
  let k := (y - 1) / (x + 3)  -- Slope calculation
  let α := Real.arctan k      -- Angle calculation
  α = π / 3 := by sorry

end slope_angle_of_parametric_line_l2598_259880


namespace pure_imaginary_product_l2598_259807

theorem pure_imaginary_product (a : ℝ) : 
  (Complex.I * Complex.I = -1) →
  (∃ b : ℝ, (2 - Complex.I) * (a + 2 * Complex.I) = Complex.I * b) →
  a = -1 := by
  sorry

end pure_imaginary_product_l2598_259807


namespace equilateral_triangle_probability_l2598_259870

/-- Given a circle divided into 30 equal parts, the probability of randomly selecting
    3 different points that form an equilateral triangle is 1/406. -/
theorem equilateral_triangle_probability (n : ℕ) (h : n = 30) :
  let total_combinations := n.choose 3
  let equilateral_triangles := n / 3
  (equilateral_triangles : ℚ) / total_combinations = 1 / 406 :=
by sorry

end equilateral_triangle_probability_l2598_259870


namespace pizza_delivery_time_per_stop_l2598_259890

theorem pizza_delivery_time_per_stop 
  (total_pizzas : ℕ) 
  (double_order_stops : ℕ) 
  (total_delivery_time : ℕ) 
  (h1 : total_pizzas = 12) 
  (h2 : double_order_stops = 2) 
  (h3 : total_delivery_time = 40) : 
  (total_delivery_time : ℚ) / (total_pizzas - double_order_stops : ℚ) = 4 := by
  sorry

end pizza_delivery_time_per_stop_l2598_259890


namespace equation_represents_hyperbola_l2598_259832

/-- The equation 9x^2 - 36y^2 = 36 represents a hyperbola -/
theorem equation_represents_hyperbola :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  (∀ (x y : ℝ), 9 * x^2 - 36 * y^2 = 36 ↔ x^2 / a^2 - y^2 / b^2 = 1) :=
by sorry

end equation_represents_hyperbola_l2598_259832


namespace quadratic_root_l2598_259806

theorem quadratic_root (a b c : ℚ) (r : ℝ) : 
  a ≠ 0 → 
  r = 2 * Real.sqrt 2 - 3 → 
  a * r^2 + b * r + c = 0 → 
  a * (1 : ℝ) = 1 ∧ b = 6 ∧ c = 1 :=
by sorry

end quadratic_root_l2598_259806


namespace regular_24gon_symmetry_sum_l2598_259829

/-- Represents a regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  -- Additional properties of regular polygons can be added here if needed

/-- The number of lines of symmetry for a regular polygon -/
def linesOfSymmetry (p : RegularPolygon n) : ℕ := sorry

/-- The smallest positive angle of rotational symmetry in degrees for a regular polygon -/
def smallestRotationAngle (p : RegularPolygon n) : ℝ := sorry

/-- Theorem: For a regular 24-gon, the sum of its number of lines of symmetry
    and its smallest positive angle of rotational symmetry (in degrees) is 39 -/
theorem regular_24gon_symmetry_sum :
  ∀ (p : RegularPolygon 24),
    (linesOfSymmetry p : ℝ) + smallestRotationAngle p = 39 := by sorry

end regular_24gon_symmetry_sum_l2598_259829


namespace conference_seating_arrangements_l2598_259888

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def bench_arrangements (g1 g2 g3 g4 : ℕ) : ℕ :=
  factorial g1 * factorial g2 * factorial g3 * factorial g4 * factorial 4

def table_arrangements (g1 g2 g3 g4 : ℕ) : ℕ :=
  factorial g1 * factorial g2 * factorial g3 * factorial g4 * factorial 3

theorem conference_seating_arrangements :
  bench_arrangements 4 2 3 4 = 165888 ∧
  table_arrangements 4 2 3 4 = 41472 := by
  sorry

end conference_seating_arrangements_l2598_259888


namespace combination_sum_equals_466_l2598_259861

theorem combination_sum_equals_466 (n : ℕ) 
  (h1 : 38 ≥ n) 
  (h2 : 3 * n ≥ 38 - n) 
  (h3 : n + 21 ≥ 3 * n) : 
  Nat.choose (3 * n) (38 - n) + Nat.choose (n + 21) (3 * n) = 466 := by
  sorry

end combination_sum_equals_466_l2598_259861


namespace mean_sales_is_five_point_five_l2598_259810

def monday_sales : ℕ := 8
def tuesday_sales : ℕ := 3
def wednesday_sales : ℕ := 10
def thursday_sales : ℕ := 4
def friday_sales : ℕ := 4
def saturday_sales : ℕ := 4

def total_sales : ℕ := monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales + saturday_sales
def number_of_days : ℕ := 6

theorem mean_sales_is_five_point_five :
  (total_sales : ℚ) / (number_of_days : ℚ) = 5.5 := by
  sorry

end mean_sales_is_five_point_five_l2598_259810


namespace coin_toss_probability_l2598_259803

/-- Represents the possible outcomes of a coin toss -/
inductive CoinToss
| Heads
| Tails

/-- The probability of getting heads in a single toss -/
def heads_prob : ℚ := 2/3

/-- The probability of getting tails in a single toss -/
def tails_prob : ℚ := 1/3

/-- The number of coin tosses -/
def num_tosses : ℕ := 10

/-- The target position to reach -/
def target_pos : ℤ := 6

/-- The position to avoid -/
def avoid_pos : ℤ := -3

/-- A function that calculates the probability of reaching the target position
    without hitting the avoid position in the given number of tosses -/
def prob_reach_target (heads_prob : ℚ) (tails_prob : ℚ) (num_tosses : ℕ) 
                      (target_pos : ℤ) (avoid_pos : ℤ) : ℚ :=
  sorry

theorem coin_toss_probability : 
  prob_reach_target heads_prob tails_prob num_tosses target_pos avoid_pos = 5120/59049 :=
sorry

end coin_toss_probability_l2598_259803


namespace car_speed_time_relationship_l2598_259864

theorem car_speed_time_relationship 
  (distance : ℝ) 
  (speed_A time_A : ℝ) 
  (speed_B time_B : ℝ) 
  (h1 : distance > 0) 
  (h2 : speed_A > 0) 
  (h3 : speed_B = 3 * speed_A) 
  (h4 : distance = speed_A * time_A) 
  (h5 : distance = speed_B * time_B) : 
  time_B = time_A / 3 := by
sorry


end car_speed_time_relationship_l2598_259864


namespace prime_sum_probability_l2598_259878

def first_twelve_primes : List Nat := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

def is_valid_pair (p q : Nat) : Bool :=
  p ∈ first_twelve_primes ∧ q ∈ first_twelve_primes ∧ p ≠ q ∧
  Nat.Prime (p + q) ∧ p + q > 20

def count_valid_pairs : Nat :=
  (List.filter (fun (pair : Nat × Nat) => is_valid_pair pair.1 pair.2)
    (List.product first_twelve_primes first_twelve_primes)).length

def total_pairs : Nat := (first_twelve_primes.length * (first_twelve_primes.length - 1)) / 2

theorem prime_sum_probability :
  count_valid_pairs / total_pairs = 1 / 66 := by sorry

end prime_sum_probability_l2598_259878


namespace shooter_scores_equal_l2598_259833

/-- The expected value of a binomial distribution -/
def binomialExpectation (n : ℕ) (p : ℝ) : ℝ := n * p

/-- The score of shooter A -/
def X₁ : ℝ := binomialExpectation 10 0.9

/-- The score of shooter Y (intermediate for shooter B) -/
def Y : ℝ := binomialExpectation 5 0.8

/-- The score of shooter B -/
def X₂ : ℝ := 2 * Y + 1

theorem shooter_scores_equal : X₁ = X₂ := by sorry

end shooter_scores_equal_l2598_259833


namespace c_increases_as_n_increases_l2598_259859

/-- Given a formula for C, prove that C increases as n increases. -/
theorem c_increases_as_n_increases
  (e R r : ℝ)
  (he : e > 0)
  (hR : R > 0)
  (hr : r > 0)
  (C : ℝ → ℝ)
  (hC : ∀ n, n > 0 → C n = (e^2 * n) / (R + n*r)) :
  ∀ n₁ n₂, 0 < n₁ → n₁ < n₂ → C n₁ < C n₂ :=
by sorry

end c_increases_as_n_increases_l2598_259859


namespace greatest_integer_of_2e_minus_5_l2598_259892

theorem greatest_integer_of_2e_minus_5 :
  ⌊2 * Real.exp 1 - 5⌋ = 0 := by sorry

end greatest_integer_of_2e_minus_5_l2598_259892


namespace mariams_neighborhood_houses_l2598_259824

/-- The number of houses in Mariam's neighborhood -/
def total_houses (houses_one_side : ℕ) (multiplier : ℕ) : ℕ :=
  houses_one_side + houses_one_side * multiplier

/-- Theorem stating the total number of houses in Mariam's neighborhood -/
theorem mariams_neighborhood_houses : 
  total_houses 40 3 = 160 := by sorry

end mariams_neighborhood_houses_l2598_259824


namespace parallel_line_plane_not_implies_parallel_all_lines_l2598_259849

-- Define the basic geometric objects
variable (Point Line Plane : Type)

-- Define the geometric relations
variable (contains : Plane → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)

-- State the theorem
theorem parallel_line_plane_not_implies_parallel_all_lines 
  (α : Plane) (a b : Line) : 
  ¬(∀ (p : Plane) (l m : Line), 
    parallel_line_plane l p → 
    contains p m → 
    parallel_lines l m) := by
  sorry

end parallel_line_plane_not_implies_parallel_all_lines_l2598_259849


namespace tetrahedral_pyramid_marbles_hypertetrahedron_marbles_formula_l2598_259869

/-- The number of marbles in a d-dimensional hypertetrahedron with N layers -/
def hypertetrahedron_marbles (d : ℕ) (N : ℕ) : ℕ := Nat.choose (N + d - 1) d

/-- Theorem: The number of marbles in a tetrahedral pyramid with N layers is (N + 2) choose 3 -/
theorem tetrahedral_pyramid_marbles (N : ℕ) : 
  hypertetrahedron_marbles 3 N = Nat.choose (N + 2) 3 := by sorry

/-- Theorem: The number of marbles in a d-dimensional hypertetrahedron with N layers is (N + d - 1) choose d -/
theorem hypertetrahedron_marbles_formula (d : ℕ) (N : ℕ) : 
  hypertetrahedron_marbles d N = Nat.choose (N + d - 1) d := by sorry

end tetrahedral_pyramid_marbles_hypertetrahedron_marbles_formula_l2598_259869


namespace crayon_cost_theorem_l2598_259858

/-- The number of crayons in half a dozen -/
def half_dozen : ℕ := 6

/-- The number of half dozens bought -/
def num_half_dozens : ℕ := 4

/-- The cost of each crayon in dollars -/
def cost_per_crayon : ℕ := 2

/-- The total number of crayons bought -/
def total_crayons : ℕ := num_half_dozens * half_dozen

/-- The total cost of the crayons in dollars -/
def total_cost : ℕ := total_crayons * cost_per_crayon

theorem crayon_cost_theorem : total_cost = 48 := by
  sorry

end crayon_cost_theorem_l2598_259858


namespace total_value_of_treats_l2598_259856

-- Define the given values
def hotel_price_per_night : ℝ := 4000
def hotel_nights : ℕ := 2
def hotel_discount : ℝ := 0.05
def car_price : ℝ := 30000
def car_tax : ℝ := 0.10
def house_multiplier : ℝ := 4
def house_tax : ℝ := 0.02
def yacht_multiplier : ℝ := 2
def yacht_discount : ℝ := 0.07
def gold_multiplier : ℝ := 1.5
def gold_tax : ℝ := 0.03

-- Define the calculated values
def hotel_total : ℝ := hotel_price_per_night * hotel_nights * (1 - hotel_discount)
def car_total : ℝ := car_price * (1 + car_tax)
def house_total : ℝ := car_price * house_multiplier * (1 + house_tax)
def yacht_total : ℝ := (hotel_price_per_night * hotel_nights + car_price) * yacht_multiplier * (1 - yacht_discount)
def gold_total : ℝ := (hotel_price_per_night * hotel_nights + car_price) * yacht_multiplier * gold_multiplier * (1 + gold_tax)

-- Theorem statement
theorem total_value_of_treats : 
  hotel_total + car_total + house_total + yacht_total + gold_total = 339100 := by
  sorry

end total_value_of_treats_l2598_259856


namespace juice_ratio_is_three_to_one_l2598_259855

/-- Represents the ratio of water cans to concentrate cans -/
structure JuiceRatio where
  water : ℕ
  concentrate : ℕ

/-- Calculates the juice ratio given the problem parameters -/
def calculateJuiceRatio (servings : ℕ) (servingSize : ℕ) (concentrateCans : ℕ) (canSize : ℕ) : JuiceRatio :=
  let totalOunces := servings * servingSize
  let totalCans := totalOunces / canSize
  let waterCans := totalCans - concentrateCans
  { water := waterCans, concentrate := concentrateCans }

theorem juice_ratio_is_three_to_one :
  let ratio := calculateJuiceRatio 320 6 40 12
  ratio.water = 3 * ratio.concentrate := by sorry

end juice_ratio_is_three_to_one_l2598_259855


namespace faster_train_length_l2598_259838

/-- Given two trains moving in the same direction, this theorem calculates the length of the faster train. -/
theorem faster_train_length
  (faster_speed slower_speed : ℝ)
  (crossing_time : ℝ)
  (h1 : faster_speed = 180)
  (h2 : slower_speed = 90)
  (h3 : crossing_time = 15)
  (h4 : faster_speed > slower_speed) :
  let relative_speed := (faster_speed - slower_speed) * (5/18)
  (relative_speed * crossing_time) = 375 := by
sorry

end faster_train_length_l2598_259838


namespace journey_distance_journey_distance_proof_l2598_259848

theorem journey_distance : ℝ → Prop :=
  fun d : ℝ =>
    let t := d / 40
    t + 1/4 = d / 35 →
    d = 70

-- The proof is omitted
theorem journey_distance_proof : journey_distance 70 := by
  sorry

end journey_distance_journey_distance_proof_l2598_259848


namespace area_ratio_of_squares_l2598_259845

/-- Given four square regions with specified perimeters and a relation between sides,
    prove that the ratio of areas of region III to region IV is 9/4. -/
theorem area_ratio_of_squares (perimeter_I perimeter_II perimeter_IV : ℝ) 
    (h1 : perimeter_I = 16)
    (h2 : perimeter_II = 20)
    (h3 : perimeter_IV = 32)
    (h4 : ∀ s : ℝ, s > 0 → perimeter_I = 4 * s → 3 * s = side_length_III) :
    (side_length_III ^ 2) / ((perimeter_IV / 4) ^ 2) = 9 / 4 := by
  sorry

end area_ratio_of_squares_l2598_259845


namespace tv_purchase_months_l2598_259844

/-- Calculates the number of months required to purchase a TV given income and expenses -/
def monthsToTV (monthlyIncome : ℕ) (foodExpense : ℕ) (utilitiesExpense : ℕ) (otherExpenses : ℕ)
                (currentSavings : ℕ) (tvCost : ℕ) : ℕ :=
  let totalExpenses := foodExpense + utilitiesExpense + otherExpenses
  let disposableIncome := monthlyIncome - totalExpenses
  let amountNeeded := tvCost - currentSavings
  (amountNeeded + disposableIncome - 1) / disposableIncome

theorem tv_purchase_months :
  monthsToTV 30000 15000 5000 2500 10000 25000 = 2 :=
sorry

end tv_purchase_months_l2598_259844


namespace swimming_ratio_proof_l2598_259863

/-- Given information about the swimming abilities of Yvonne, Joel, and their younger sister,
    prove that the ratio of laps swum by the younger sister to Yvonne is 1:2. -/
theorem swimming_ratio_proof (yvonne_laps joel_laps : ℕ) (joel_ratio : ℕ) :
  yvonne_laps = 10 →
  joel_laps = 15 →
  joel_ratio = 3 →
  (joel_laps / joel_ratio : ℚ) / yvonne_laps = 1 / 2 := by
  sorry

end swimming_ratio_proof_l2598_259863


namespace mans_speed_with_current_l2598_259867

/-- 
Given a man's speed against a current and the speed of the current,
this theorem proves the man's speed with the current.
-/
theorem mans_speed_with_current 
  (speed_against_current : ℝ) 
  (current_speed : ℝ) 
  (h1 : speed_against_current = 12) 
  (h2 : current_speed = 5) : 
  speed_against_current + 2 * current_speed = 22 :=
by sorry

end mans_speed_with_current_l2598_259867


namespace simplify_expression_l2598_259889

theorem simplify_expression (p : ℝ) : 
  ((7 * p + 3) - 3 * p * 2) * 4 + (5 - 2 / 4) * (8 * p - 12) = 40 * p - 42 := by
  sorry

end simplify_expression_l2598_259889


namespace sum_of_coefficients_l2598_259896

/-- Given (1-2x)^7 = a + a₁x + a₂x² + ... + a₇x⁷, prove that a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = 12 -/
theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x, (1 - 2*x)^7 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = 12 := by
  sorry

end sum_of_coefficients_l2598_259896


namespace inequality_solution_set_l2598_259884

def solution_set (x : ℝ) : Prop := x ≥ 3 ∨ x ≤ 1

theorem inequality_solution_set
  (f : ℝ → ℝ)
  (f_even : ∀ x, f x = f (-x))
  (f_increasing : ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y)
  (f_one_eq_zero : f 1 = 0) :
  ∀ x, f (x - 2) ≥ 0 ↔ solution_set x :=
sorry

end inequality_solution_set_l2598_259884


namespace quadratic_equation_solution_l2598_259894

theorem quadratic_equation_solution (x : ℝ) : 16 * x^2 = 81 ↔ x = 2.25 ∨ x = -2.25 := by
  sorry

end quadratic_equation_solution_l2598_259894


namespace cylinder_volume_increase_l2598_259857

/-- Theorem: Doubling the radius of a right circular cylinder with volume 6 liters increases its volume by 18 liters -/
theorem cylinder_volume_increase (r h : ℝ) (h1 : r > 0) (h2 : h > 0) : 
  π * r^2 * h = 6 → π * (2*r)^2 * h - π * r^2 * h = 18 := by
  sorry

#check cylinder_volume_increase

end cylinder_volume_increase_l2598_259857


namespace complement_intersection_M_N_l2598_259821

def U : Set Nat := {1, 2, 3, 4}
def M : Set Nat := {1, 2, 3}
def N : Set Nat := {2, 3, 4}

theorem complement_intersection_M_N :
  (U \ (M ∩ N)) = {1, 4} := by sorry

end complement_intersection_M_N_l2598_259821


namespace root_in_interval_l2598_259801

def f (x : ℝ) := x^3 - 2*x - 5

theorem root_in_interval :
  (f 2 < 0) →
  (f 3 > 0) →
  (f 2.5 > 0) →
  ∃ x, x ∈ Set.Ioo 2 2.5 ∧ f x = 0 :=
by sorry

end root_in_interval_l2598_259801


namespace katies_new_friends_games_l2598_259809

/-- The number of games Katie's new friends have -/
def new_friends_games (total_friends_games old_friends_games : ℕ) : ℕ :=
  total_friends_games - old_friends_games

/-- Theorem: Katie's new friends have 88 games -/
theorem katies_new_friends_games :
  new_friends_games 141 53 = 88 := by
  sorry

end katies_new_friends_games_l2598_259809


namespace inscribed_box_dimension_l2598_259872

/-- A rectangular box inscribed in a sphere -/
structure InscribedBox where
  x : ℝ
  y : ℝ
  z : ℝ
  sphere_radius : ℝ
  surface_area : ℝ
  edge_sum : ℝ
  sphere_constraint : x^2 + y^2 + z^2 = 4 * sphere_radius^2
  surface_area_constraint : 2*x*y + 2*y*z + 2*x*z = surface_area
  edge_sum_constraint : 4*(x + y + z) = edge_sum

/-- Theorem: For a rectangular box inscribed in a sphere of radius 10,
    with surface area 416 and sum of edge lengths 120,
    one of its dimensions is 10 -/
theorem inscribed_box_dimension (Q : InscribedBox)
    (h_radius : Q.sphere_radius = 10)
    (h_surface : Q.surface_area = 416)
    (h_edges : Q.edge_sum = 120) :
    Q.x = 10 ∨ Q.y = 10 ∨ Q.z = 10 := by
  sorry

end inscribed_box_dimension_l2598_259872


namespace min_vertical_distance_l2598_259837

/-- The absolute value function -/
def f (x : ℝ) : ℝ := |x|

/-- The quadratic function -/
def g (x : ℝ) : ℝ := -x^2 - 4*x - 3

/-- The vertical distance between f and g -/
def vertical_distance (x : ℝ) : ℝ := |f x - g x|

/-- Theorem stating the minimum vertical distance between f and g -/
theorem min_vertical_distance :
  ∃ (min_dist : ℝ), min_dist = 3/4 ∧ ∀ (x : ℝ), vertical_distance x ≥ min_dist :=
sorry

end min_vertical_distance_l2598_259837


namespace distance_between_foci_l2598_259826

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop :=
  Real.sqrt ((x - 2)^2 + (y + 3)^2) + Real.sqrt ((x + 6)^2 + (y - 9)^2) = 24

-- Define the foci
def focus1 : ℝ × ℝ := (2, -3)
def focus2 : ℝ × ℝ := (-6, 9)

-- Theorem stating the distance between foci
theorem distance_between_foci :
  let (x1, y1) := focus1
  let (x2, y2) := focus2
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2) = 4 * Real.sqrt 13 := by sorry

end distance_between_foci_l2598_259826


namespace train_length_proof_l2598_259883

/-- Proves that a train with given speed and crossing time has a specific length -/
theorem train_length_proof (speed : ℝ) (crossing_time : ℝ) (train_length : ℝ) : 
  speed = 90 → -- speed in km/hr
  crossing_time = 1 / 60 → -- crossing time in hours (1 minute = 1/60 hour)
  train_length = speed * crossing_time / 2 → -- length calculation
  train_length = 750 / 1000 -- length in km (750 m = 0.75 km)
  := by sorry

end train_length_proof_l2598_259883


namespace charging_pile_equation_l2598_259802

/-- Represents the growth of smart charging piles over two months -/
def charging_pile_growth (initial : ℕ) (growth_rate : ℝ) : ℝ :=
  initial * (1 + growth_rate)^2

/-- Theorem stating the relationship between the number of charging piles
    in the first and third months, given the monthly average growth rate -/
theorem charging_pile_equation (x : ℝ) : charging_pile_growth 301 x = 500 := by
  sorry

end charging_pile_equation_l2598_259802


namespace M_intersect_N_l2598_259840

def M : Set ℤ := {-1, 0, 1, 2}

def N : Set ℤ := {y | ∃ x ∈ M, y = 2*x + 1}

theorem M_intersect_N : M ∩ N = {-1, 1} := by sorry

end M_intersect_N_l2598_259840


namespace lauryn_company_men_count_l2598_259800

theorem lauryn_company_men_count :
  ∀ (men women : ℕ),
    men + women = 180 →
    women = men + 20 →
    men = 80 := by
sorry

end lauryn_company_men_count_l2598_259800


namespace simplify_expression_l2598_259808

theorem simplify_expression : (27 * (10 ^ 9)) / (9 * (10 ^ 5)) = 30000 := by
  sorry

end simplify_expression_l2598_259808


namespace circle_area_with_diameter_two_l2598_259818

theorem circle_area_with_diameter_two (π : Real) : Real :=
  let diameter : Real := 2
  let radius : Real := diameter / 2
  let area : Real := π * radius^2
  area

#check circle_area_with_diameter_two

end circle_area_with_diameter_two_l2598_259818


namespace problem_one_problem_two_l2598_259811

-- Problem 1
theorem problem_one : (Real.sqrt 48 + Real.sqrt 20) - (Real.sqrt 12 - Real.sqrt 5) = 2 * Real.sqrt 3 + 3 * Real.sqrt 5 := by
  sorry

-- Problem 2
theorem problem_two : |2 - Real.sqrt 2| - Real.sqrt (1/12) * Real.sqrt 27 + Real.sqrt 12 / Real.sqrt 6 = 1/2 := by
  sorry

end problem_one_problem_two_l2598_259811


namespace min_value_a_plus_2b_min_value_is_5_plus_2sqrt6_equality_condition_l2598_259828

theorem min_value_a_plus_2b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2*a + b = a*b - 1) :
  ∀ x y : ℝ, x > 0 → y > 0 → 2*x + y = x*y - 1 → a + 2*b ≤ x + 2*y :=
by sorry

theorem min_value_is_5_plus_2sqrt6 (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2*a + b = a*b - 1) :
  a + 2*b ≥ 5 + 2*Real.sqrt 6 :=
by sorry

theorem equality_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2*a + b = a*b - 1) :
  a + 2*b = 5 + 2*Real.sqrt 6 ↔ b = 2 + Real.sqrt 6 / 2 :=
by sorry

end min_value_a_plus_2b_min_value_is_5_plus_2sqrt6_equality_condition_l2598_259828


namespace loan_duration_to_c_l2598_259898

/-- Proves that the number of years A lent money to C is 4, given the specified conditions. -/
theorem loan_duration_to_c (principal_b principal_c total_interest : ℚ) 
  (duration_b : ℚ) (rate : ℚ) : 
  principal_b = 5000 →
  principal_c = 3000 →
  duration_b = 2 →
  rate = 7.000000000000001 / 100 →
  total_interest = 1540 →
  total_interest = principal_b * rate * duration_b + principal_c * rate * (4 : ℚ) :=
by sorry

end loan_duration_to_c_l2598_259898


namespace cottage_configuration_exists_l2598_259899

/-- A configuration of points on a circle -/
def Configuration := List ℕ

/-- The sum of elements in a list -/
def list_sum (l : List ℕ) : ℕ := l.foldl (·+·) 0

/-- Check if all elements in a list are unique -/
def all_unique (l : List ℕ) : Prop := l.Nodup

/-- Generate all distances between points in a circular configuration -/
def generate_distances (config : Configuration) : List ℕ :=
  let n := config.length
  let total := list_sum config
  List.range n >>= fun i =>
    List.range n >>= fun j =>
      if i < j then
        let dist := (list_sum (config.take j) - list_sum (config.take i) + total) % total
        [min dist (total - dist)]
      else
        []

/-- The main theorem statement -/
theorem cottage_configuration_exists : ∃ (config : Configuration),
  (config.length = 6) ∧
  (list_sum config = 27) ∧
  (all_unique (generate_distances config)) ∧
  (∀ d, d ∈ generate_distances config → d ≥ 1 ∧ d ≤ 26) :=
sorry

end cottage_configuration_exists_l2598_259899


namespace intersection_of_sets_l2598_259891

theorem intersection_of_sets (A B : Set ℝ) : 
  A = {x : ℝ | x^2 + 2*x - 3 > 0} →
  B = {-1, 0, 1, 2} →
  A ∩ B = {2} := by sorry

end intersection_of_sets_l2598_259891


namespace events_mutually_exclusive_not_contradictory_l2598_259850

/-- A bag containing red and white balls -/
structure Bag where
  red : ℕ
  white : ℕ

/-- An event when drawing balls from the bag -/
structure Event (bag : Bag) where
  pred : (ℕ × ℕ) → Prop

/-- The bag in our problem -/
def problem_bag : Bag := { red := 3, white := 3 }

/-- The event "At least 2 white balls" -/
def at_least_2_white (bag : Bag) : Event bag :=
  { pred := λ (r, w) => w ≥ 2 }

/-- The event "All red balls" -/
def all_red (bag : Bag) : Event bag :=
  { pred := λ (r, w) => r = 3 ∧ w = 0 }

/-- Two events are mutually exclusive -/
def mutually_exclusive (bag : Bag) (e1 e2 : Event bag) : Prop :=
  ∀ r w, (r + w = 3) → ¬(e1.pred (r, w) ∧ e2.pred (r, w))

/-- Two events are contradictory -/
def contradictory (bag : Bag) (e1 e2 : Event bag) : Prop :=
  ∀ r w, (r + w = 3) → (e1.pred (r, w) ↔ ¬e2.pred (r, w))

/-- The main theorem to prove -/
theorem events_mutually_exclusive_not_contradictory :
  mutually_exclusive problem_bag (at_least_2_white problem_bag) (all_red problem_bag) ∧
  ¬contradictory problem_bag (at_least_2_white problem_bag) (all_red problem_bag) :=
sorry

end events_mutually_exclusive_not_contradictory_l2598_259850


namespace hallie_monday_tips_l2598_259868

/-- Represents Hallie's work and earnings over three days --/
structure WaitressEarnings where
  hourly_rate : ℝ
  monday_hours : ℝ
  tuesday_hours : ℝ
  wednesday_hours : ℝ
  tuesday_tips : ℝ
  wednesday_tips : ℝ
  total_earnings : ℝ

/-- Calculates Hallie's tips on Monday given her work schedule and earnings --/
def monday_tips (e : WaitressEarnings) : ℝ :=
  e.total_earnings -
  (e.hourly_rate * (e.monday_hours + e.tuesday_hours + e.wednesday_hours)) -
  e.tuesday_tips - e.wednesday_tips

/-- Theorem stating that Hallie's tips on Monday were $18 --/
theorem hallie_monday_tips (e : WaitressEarnings)
  (h1 : e.hourly_rate = 10)
  (h2 : e.monday_hours = 7)
  (h3 : e.tuesday_hours = 5)
  (h4 : e.wednesday_hours = 7)
  (h5 : e.tuesday_tips = 12)
  (h6 : e.wednesday_tips = 20)
  (h7 : e.total_earnings = 240) :
  monday_tips e = 18 := by
  sorry

end hallie_monday_tips_l2598_259868


namespace marble_count_l2598_259862

/-- Given a bag of marbles with red, blue, and yellow marbles in the ratio 2:3:4,
    and 36 yellow marbles, prove that there are 81 marbles in total. -/
theorem marble_count (red blue yellow total : ℕ) : 
  red + blue + yellow = total →
  red = 2 * n ∧ blue = 3 * n ∧ yellow = 4 * n →
  yellow = 36 →
  total = 81 :=
by
  sorry

#check marble_count

end marble_count_l2598_259862


namespace mask_selection_probability_l2598_259854

theorem mask_selection_probability :
  let total_colors : ℕ := 5
  let selected_masks : ℕ := 3
  let favorable_outcomes : ℕ := (total_colors - 2).choose 1
  let total_outcomes : ℕ := total_colors.choose selected_masks
  (favorable_outcomes : ℚ) / total_outcomes = 3 / 10 := by
sorry

end mask_selection_probability_l2598_259854


namespace smo_board_sum_l2598_259886

/-- Represents the state of the board at any given step -/
structure BoardState where
  numbers : List Nat

/-- Represents a single step in the process -/
def step (state : BoardState) : BoardState :=
  sorry

/-- The sum of all numbers on the board -/
def board_sum (state : BoardState) : Nat :=
  state.numbers.sum

theorem smo_board_sum (m : Nat) : 
  ∀ (final_state : BoardState),
    (∃ (initial_state : BoardState),
      initial_state.numbers = List.replicate (2^m) 1 ∧
      final_state = (step^[m * 2^(m-1)]) initial_state) →
    board_sum final_state ≥ 4^m :=
  sorry

end smo_board_sum_l2598_259886


namespace sqrt_fraction_sum_equals_sqrt_433_over_18_l2598_259853

theorem sqrt_fraction_sum_equals_sqrt_433_over_18 :
  Real.sqrt (25 / 36 + 16 / 81 + 4 / 9) = Real.sqrt 433 / 18 := by
  sorry

end sqrt_fraction_sum_equals_sqrt_433_over_18_l2598_259853


namespace sqrt_expression_equals_sqrt_845_l2598_259814

theorem sqrt_expression_equals_sqrt_845 :
  Real.sqrt 80 - 3 * Real.sqrt 5 + Real.sqrt 720 / Real.sqrt 3 = Real.sqrt 845 := by
  sorry

end sqrt_expression_equals_sqrt_845_l2598_259814


namespace davids_english_marks_l2598_259813

/-- Represents the marks of a student in various subjects -/
structure Marks where
  mathematics : ℕ
  physics : ℕ
  chemistry : ℕ
  biology : ℕ
  english : ℕ

/-- Calculates the average of a list of natural numbers -/
def average (list : List ℕ) : ℚ :=
  (list.sum : ℚ) / list.length

theorem davids_english_marks (m : Marks) (h1 : m.mathematics = 60) 
    (h2 : m.physics = 78) (h3 : m.chemistry = 60) (h4 : m.biology = 65) 
    (h5 : average [m.mathematics, m.physics, m.chemistry, m.biology, m.english] = 66.6) :
    m.english = 70 := by
  sorry

#check davids_english_marks

end davids_english_marks_l2598_259813


namespace movie_profit_calculation_l2598_259874

/-- Calculate profit for a movie given its earnings and costs -/
def movie_profit (
  opening_weekend : ℝ
  ) (
  domestic_multiplier : ℝ
  ) (
  international_multiplier : ℝ
  ) (
  domestic_tax_rate : ℝ
  ) (
  international_tax_rate : ℝ
  ) (
  royalty_rate : ℝ
  ) (
  production_cost : ℝ
  ) (
  marketing_cost : ℝ
  ) : ℝ :=
  let domestic_earnings := opening_weekend * domestic_multiplier
  let international_earnings := domestic_earnings * international_multiplier
  let domestic_after_tax := domestic_earnings * domestic_tax_rate
  let international_after_tax := international_earnings * international_tax_rate
  let total_after_tax := domestic_after_tax + international_after_tax
  let total_earnings := domestic_earnings + international_earnings
  let royalties := total_earnings * royalty_rate
  total_after_tax - royalties - production_cost - marketing_cost

/-- The profit calculation for the given movie is correct -/
theorem movie_profit_calculation :
  movie_profit 120 3.5 1.8 0.6 0.45 0.05 60 40 = 433.4 :=
by sorry

end movie_profit_calculation_l2598_259874


namespace min_value_of_f_l2598_259827

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem min_value_of_f :
  ∃ (x_min : ℝ), f x_min = Real.exp (-1) ∧ ∀ (x : ℝ), f x ≥ Real.exp (-1) :=
sorry

end min_value_of_f_l2598_259827


namespace line_properties_l2598_259831

/-- A line in the xy-plane represented by the equation x = ky + b -/
structure Line where
  k : ℝ
  b : ℝ

/-- Predicate for a line being perpendicular to the y-axis -/
def perpendicular_to_y_axis (l : Line) : Prop :=
  ∃ (x : ℝ), ∀ (y : ℝ), x = l.k * y + l.b

/-- Predicate for a line being perpendicular to the x-axis -/
def perpendicular_to_x_axis (l : Line) : Prop :=
  ∀ (y : ℝ), l.k * y + l.b = l.b

theorem line_properties :
  (¬ ∃ (l : Line), perpendicular_to_y_axis l) ∧
  (∃ (l : Line), perpendicular_to_x_axis l) :=
sorry

end line_properties_l2598_259831


namespace unique_number_l2598_259804

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def middle_digits_39 (n : ℕ) : Prop :=
  ∃ a b : ℕ, n = a * 1000 + 390 + b ∧ 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9

theorem unique_number :
  ∃! n : ℕ, is_four_digit n ∧ middle_digits_39 n ∧ n % 45 = 0 ∧ n ≤ 5000 ∧ n = 1395 :=
by sorry

end unique_number_l2598_259804


namespace red_highest_probability_l2598_259876

/-- Represents the colors of the balls in the box -/
inductive Color
  | Red
  | Yellow
  | Black

/-- Represents the box of balls -/
structure Box where
  total : Nat
  red : Nat
  yellow : Nat
  black : Nat

/-- Calculates the probability of drawing a ball of a given color -/
def probability (box : Box) (color : Color) : Rat :=
  match color with
  | Color.Red => box.red / box.total
  | Color.Yellow => box.yellow / box.total
  | Color.Black => box.black / box.total

/-- The box with the given conditions -/
def givenBox : Box :=
  { total := 10
    red := 7
    yellow := 2
    black := 1 }

theorem red_highest_probability :
  probability givenBox Color.Red > probability givenBox Color.Yellow ∧
  probability givenBox Color.Red > probability givenBox Color.Black :=
by sorry

end red_highest_probability_l2598_259876


namespace max_distance_line_circle_l2598_259875

/-- Given a line ax + 2by = 1 intersecting a circle x^2 + y^2 = 1 at points A and B,
    where triangle AOB is right-angled (O is the origin), prove that the maximum
    distance between P(a,b) and Q(0,0) is √2. -/
theorem max_distance_line_circle (a b : ℝ) : 
  (∃ A B : ℝ × ℝ, (a * A.1 + 2 * b * A.2 = 1 ∧ A.1^2 + A.2^2 = 1) ∧
                   (a * B.1 + 2 * b * B.2 = 1 ∧ B.1^2 + B.2^2 = 1) ∧
                   ((A.1 - B.1) * (A.1 + B.1) + (A.2 - B.2) * (A.2 + B.2) = 0)) →
  (∃ P : ℝ × ℝ, P.1 = a ∧ P.2 = b) →
  (∃ d : ℝ, d = Real.sqrt (a^2 + b^2) ∧ d ≤ Real.sqrt 2 ∧
            (∀ a' b' : ℝ, Real.sqrt (a'^2 + b'^2) ≤ d)) :=
by sorry


end max_distance_line_circle_l2598_259875


namespace box_C_in_A_l2598_259852

/-- The number of Box B that can fill one Box A -/
def box_B_in_A : ℕ := 4

/-- The number of Box C that can fill one Box B -/
def box_C_in_B : ℕ := 6

/-- The theorem stating that 24 Box C are needed to fill Box A -/
theorem box_C_in_A : box_B_in_A * box_C_in_B = 24 := by
  sorry

end box_C_in_A_l2598_259852


namespace software_hours_calculation_l2598_259885

def total_hours : ℝ := 68.33333333333333
def help_user_hours : ℝ := 17
def other_services_percentage : ℝ := 0.4

theorem software_hours_calculation :
  let other_services_hours := total_hours * other_services_percentage
  let software_hours := total_hours - help_user_hours - other_services_hours
  software_hours = 24 := by sorry

end software_hours_calculation_l2598_259885


namespace min_value_expression_l2598_259846

theorem min_value_expression (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : a + c = 2 * b) (h4 : a ≠ 0) :
  ((a + b)^2 + (b - c)^2 + (c - a)^2) / a^2 ≥ 7/2 ∧
  ∃ a b c : ℝ, a > b ∧ b > c ∧ a + c = 2 * b ∧ a ≠ 0 ∧
    ((a + b)^2 + (b - c)^2 + (c - a)^2) / a^2 = 7/2 :=
by sorry

end min_value_expression_l2598_259846


namespace minimum_b_value_l2598_259805

theorem minimum_b_value (a b : ℕ) : 
  a = 23 →
  (a + b) % 10 = 5 →
  (a + b) % 7 = 4 →
  b ≥ 2 ∧ ∃ (b' : ℕ), b' ≥ 2 → b ≤ b' :=
by sorry

end minimum_b_value_l2598_259805


namespace lowest_unique_score_above_100_unique_solution_for_105_l2598_259823

/-- Represents the score calculation function for the math examination. -/
def score (c w : ℕ) : ℕ := 50 + 5 * c - 2 * w

/-- Theorem stating that 105 is the lowest score above 100 with a unique solution. -/
theorem lowest_unique_score_above_100 : 
  ∀ s : ℕ, s > 100 → s < 105 → 
  (∃ c w : ℕ, c + w ≤ 50 ∧ score c w = s) → 
  (∃ c₁ w₁ c₂ w₂ : ℕ, 
    c₁ + w₁ ≤ 50 ∧ c₂ + w₂ ≤ 50 ∧ 
    score c₁ w₁ = s ∧ score c₂ w₂ = s ∧ 
    (c₁ ≠ c₂ ∨ w₁ ≠ w₂)) :=
by sorry

/-- Theorem stating that 105 has a unique solution for c and w. -/
theorem unique_solution_for_105 : 
  ∃! c w : ℕ, c + w ≤ 50 ∧ score c w = 105 :=
by sorry

end lowest_unique_score_above_100_unique_solution_for_105_l2598_259823


namespace base_b_square_theorem_l2598_259881

/-- Converts a number from base b representation to base 10 -/
def base_b_to_10 (digits : List Nat) (b : Nat) : Nat :=
  digits.foldr (fun digit acc => b * acc + digit) 0

/-- Theorem: If 1325 in base b is the square of 35 in base b, then b = 7 in base 10 -/
theorem base_b_square_theorem :
  ∀ b : Nat,
  (base_b_to_10 [1, 3, 2, 5] b = (base_b_to_10 [3, 5] b) ^ 2) →
  b = 7 :=
by sorry

end base_b_square_theorem_l2598_259881


namespace complex_fraction_simplification_l2598_259865

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (6 - i) / (1 + i) = Complex.mk (5/2) (-7/2) := by
  sorry

end complex_fraction_simplification_l2598_259865


namespace longer_strap_length_l2598_259882

theorem longer_strap_length (short long : ℕ) : 
  long = short + 72 →
  short + long = 348 →
  long = 210 :=
by sorry

end longer_strap_length_l2598_259882


namespace percentage_calculation_l2598_259879

theorem percentage_calculation (P : ℝ) : 
  (0.05 * (P / 100 * 1600) = 20) → P = 25 := by
  sorry

end percentage_calculation_l2598_259879


namespace sine_amplitude_negative_a_l2598_259812

theorem sine_amplitude_negative_a (a b : ℝ) (h1 : a < 0) (h2 : b > 0) :
  (∀ x, ∃ y, y = a * Real.sin (b * x)) →
  (∀ x, a * Real.sin (b * x) ≥ -2 ∧ a * Real.sin (b * x) ≤ 0) →
  (∃ x, a * Real.sin (b * x) = -2) →
  a = -2 := by
sorry

end sine_amplitude_negative_a_l2598_259812


namespace packs_needed_for_360_days_l2598_259897

/-- The number of dog walks per day -/
def walks_per_day : ℕ := 2

/-- The number of wipes used per walk -/
def wipes_per_walk : ℕ := 1

/-- The number of wipes in a pack -/
def wipes_per_pack : ℕ := 120

/-- The number of days we need to cover -/
def days_to_cover : ℕ := 360

/-- The number of packs needed for the given number of days -/
def packs_needed : ℕ := 
  (days_to_cover * walks_per_day * wipes_per_walk + wipes_per_pack - 1) / wipes_per_pack

theorem packs_needed_for_360_days : packs_needed = 6 := by
  sorry

end packs_needed_for_360_days_l2598_259897


namespace robot_rascals_shipment_l2598_259835

theorem robot_rascals_shipment (total : ℝ) : 
  (0.7 * total = 168) → total = 240 := by
  sorry

end robot_rascals_shipment_l2598_259835


namespace perpendicular_to_same_line_implies_parallel_l2598_259839

-- Define a structure for a line in a plane
structure Line where
  -- You can add more properties if needed
  mk :: (id : Nat)

-- Define perpendicularity between two lines
def perpendicular (l1 l2 : Line) : Prop :=
  sorry -- Definition of perpendicularity

-- Define parallelism between two lines
def parallel (l1 l2 : Line) : Prop :=
  sorry -- Definition of parallelism

-- Theorem statement
theorem perpendicular_to_same_line_implies_parallel 
  (l1 l2 l3 : Line) : 
  perpendicular l1 l3 → perpendicular l2 l3 → parallel l1 l2 :=
by
  sorry -- Proof goes here

end perpendicular_to_same_line_implies_parallel_l2598_259839


namespace parabola_vertex_l2598_259841

/-- The vertex of the parabola y = 3x^2 + 2 has coordinates (0, 2) -/
theorem parabola_vertex (x y : ℝ) : y = 3 * x^2 + 2 → (0, 2) = (x, y) := by
  sorry

end parabola_vertex_l2598_259841


namespace olivia_payment_l2598_259816

/-- Represents the number of quarters in a dollar -/
def quarters_per_dollar : ℕ := 4

/-- Represents the number of quarters Olivia pays for chips -/
def quarters_for_chips : ℕ := 4

/-- Represents the number of quarters Olivia pays for soda -/
def quarters_for_soda : ℕ := 12

/-- Calculates the total amount Olivia pays in dollars -/
def total_payment : ℚ :=
  (quarters_for_chips + quarters_for_soda) / quarters_per_dollar

theorem olivia_payment :
  total_payment = 4 := by sorry

end olivia_payment_l2598_259816


namespace not_sixth_power_l2598_259830

theorem not_sixth_power (n : ℕ) : ¬ ∃ (k : ℤ), 6 * (n : ℤ)^3 + 3 = k^6 := by
  sorry

end not_sixth_power_l2598_259830


namespace characterize_M_and_m_l2598_259819

-- Define the set S
def S : Set ℝ := {1, 2, 3, 6}

-- Define the set M
def M (m : ℝ) : Set ℝ := {x | x^2 - m*x + 6 = 0}

-- State the theorem
theorem characterize_M_and_m :
  ∀ m : ℝ, (M m ∩ S = M m) →
  ((M m = {2, 3} ∧ m = 7) ∨
   (M m = {1, 6} ∧ m = 5) ∨
   (M m = ∅ ∧ m > -2*Real.sqrt 6 ∧ m < 2*Real.sqrt 6)) :=
by sorry

end characterize_M_and_m_l2598_259819


namespace cubic_integer_roots_l2598_259817

/-- Represents a cubic polynomial with integer coefficients -/
structure CubicPolynomial where
  b : ℤ
  c : ℤ
  d : ℤ

/-- Counts the number of integer roots of a cubic polynomial, including multiplicity -/
def count_integer_roots (p : CubicPolynomial) : ℕ := sorry

/-- Theorem stating that the number of integer roots of a cubic polynomial with integer coefficients is 0, 1, 2, or 3 -/
theorem cubic_integer_roots (p : CubicPolynomial) :
  count_integer_roots p = 0 ∨ count_integer_roots p = 1 ∨ count_integer_roots p = 2 ∨ count_integer_roots p = 3 := by
  sorry

end cubic_integer_roots_l2598_259817


namespace fraction_comparison_l2598_259895

def numerator (x : ℝ) : ℝ := 5 * x + 3

def denominator (x : ℝ) : ℝ := 8 - 3 * x

theorem fraction_comparison (x : ℝ) (h : -3 ≤ x ∧ x ≤ 3) :
  numerator x > denominator x ↔ 5/8 < x ∧ x ≤ 3 := by
  sorry

end fraction_comparison_l2598_259895


namespace complex_angle_in_second_quadrant_l2598_259842

theorem complex_angle_in_second_quadrant 
  (z : ℂ) (θ : ℝ) 
  (h1 : z = Complex.exp (θ * Complex.I))
  (h2 : Real.cos θ < 0)
  (h3 : Real.sin θ > 0) : 
  π / 2 < θ ∧ θ < π :=
by sorry

end complex_angle_in_second_quadrant_l2598_259842


namespace students_playing_both_sports_l2598_259851

/-- Given a school with students playing football and cricket, calculate the number of students playing both sports. -/
theorem students_playing_both_sports 
  (total : ℕ) 
  (football : ℕ) 
  (cricket : ℕ) 
  (neither : ℕ) 
  (h1 : total = 470) 
  (h2 : football = 325) 
  (h3 : cricket = 175) 
  (h4 : neither = 50) : 
  football + cricket - (total - neither) = 80 := by
  sorry

end students_playing_both_sports_l2598_259851


namespace interchanged_digits_theorem_l2598_259822

/-- 
Given a two-digit number n = 10a + b, where n = 3(a + b),
prove that the number formed by interchanging its digits (10b + a) 
is equal to 8 times the sum of its digits (8(a + b)).
-/
theorem interchanged_digits_theorem (a b : ℕ) (h1 : a ≤ 9) (h2 : b ≤ 9) (h3 : a ≠ 0)
  (h4 : 10 * a + b = 3 * (a + b)) :
  10 * b + a = 8 * (a + b) := by
  sorry

end interchanged_digits_theorem_l2598_259822


namespace equation_solution_l2598_259866

theorem equation_solution (x : ℝ) (h1 : x ≠ 6) (h2 : x ≠ 3/4) :
  (x^2 - 10*x + 24)/(x - 6) + (4*x^2 + 20*x - 24)/(4*x - 3) + 2*x = 5 ↔ x = 1/4 := by
  sorry

#check equation_solution

end equation_solution_l2598_259866


namespace fraction_sum_equality_l2598_259843

theorem fraction_sum_equality : (2 : ℚ) / 5 - 1 / 10 + 3 / 5 = 9 / 10 := by
  sorry

end fraction_sum_equality_l2598_259843


namespace quadratic_coefficient_l2598_259836

/-- A quadratic function with vertex form (x + h)^2 + k -/
def QuadraticFunction (a h k : ℝ) (x : ℝ) : ℝ := a * (x + h)^2 + k

theorem quadratic_coefficient (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f x = QuadraticFunction a 3 0 x) →  -- vertex at (-3, 0)
  f 2 = -50 →                              -- passes through (2, -50)
  a = -2 := by
sorry

end quadratic_coefficient_l2598_259836


namespace josh_marbles_l2598_259860

theorem josh_marbles (initial : ℕ) (lost : ℕ) (remaining : ℕ) : 
  initial = 16 → lost = 7 → remaining = initial - lost → remaining = 9 := by
  sorry

end josh_marbles_l2598_259860

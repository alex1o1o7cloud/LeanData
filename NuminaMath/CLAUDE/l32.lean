import Mathlib

namespace group_size_calculation_l32_3200

theorem group_size_calculation (initial_avg : ℝ) (final_avg : ℝ) (new_member1 : ℝ) (new_member2 : ℝ) :
  initial_avg = 48 →
  final_avg = 51 →
  new_member1 = 78 →
  new_member2 = 93 →
  ∃ n : ℕ, n * initial_avg + new_member1 + new_member2 = (n + 2) * final_avg ∧ n = 23 :=
by
  sorry

end group_size_calculation_l32_3200


namespace fisherman_catch_l32_3247

/-- The number of fish caught by a fisherman -/
def total_fish (num_boxes : ℕ) (fish_per_box : ℕ) (fish_outside : ℕ) : ℕ :=
  num_boxes * fish_per_box + fish_outside

/-- Theorem stating the total number of fish caught by the fisherman -/
theorem fisherman_catch :
  total_fish 15 20 6 = 306 := by
  sorry

end fisherman_catch_l32_3247


namespace smallest_number_divisible_l32_3299

def divisors : List ℕ := [12, 16, 18, 21, 28, 35, 40, 45, 55]

theorem smallest_number_divisible (n : ℕ) : 
  (∀ d ∈ divisors, (n - 10) % d = 0) →
  (∀ m < n, ∃ d ∈ divisors, (m - 10) % d ≠ 0) →
  n = 55450 := by
sorry

end smallest_number_divisible_l32_3299


namespace sqrt_3_irrational_l32_3204

theorem sqrt_3_irrational : Irrational (Real.sqrt 3) := by
  sorry

end sqrt_3_irrational_l32_3204


namespace negative_integer_solutions_l32_3202

def inequality_system (x : ℤ) : Prop :=
  2 * x + 9 ≥ 3 ∧ (1 + 2 * x) / 3 + 1 > x

def is_negative_integer (x : ℤ) : Prop :=
  x < 0

theorem negative_integer_solutions :
  {x : ℤ | inequality_system x ∧ is_negative_integer x} = {-3, -2, -1} :=
sorry

end negative_integer_solutions_l32_3202


namespace hyperbola_line_slope_l32_3233

/-- Given a hyperbola and a line intersecting it, prove that the slope of the line is 6 -/
theorem hyperbola_line_slope :
  ∀ (A B : ℝ × ℝ),
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  let P := (2, 1)
  -- Hyperbola equation
  (x₁^2 - y₁^2/3 = 1) →
  (x₂^2 - y₂^2/3 = 1) →
  -- P is the midpoint of AB
  (2 = (x₁ + x₂)/2) →
  (1 = (y₁ + y₂)/2) →
  -- Slope of AB
  ((y₁ - y₂)/(x₁ - x₂) = 6) :=
by
  sorry


end hyperbola_line_slope_l32_3233


namespace four_digit_number_relation_l32_3205

theorem four_digit_number_relation : 
  let n : ℕ := 1197
  let thousands : ℕ := n / 1000
  let hundreds : ℕ := (n / 100) % 10
  let tens : ℕ := (n / 10) % 10
  let units : ℕ := n % 10
  units = hundreds - 2 →
  thousands + hundreds + tens + units = 18 →
  thousands = hundreds - 2 :=
by sorry

end four_digit_number_relation_l32_3205


namespace rose_flowers_l32_3275

/-- The number of flowers Rose bought -/
def total_flowers : ℕ := 12

/-- The number of daisies -/
def daisies : ℕ := 2

/-- The number of sunflowers -/
def sunflowers : ℕ := 4

/-- The number of tulips -/
def tulips : ℕ := (3 * (total_flowers - daisies)) / 5

theorem rose_flowers :
  total_flowers = daisies + tulips + sunflowers ∧
  tulips = (3 * (total_flowers - daisies)) / 5 ∧
  sunflowers = (2 * (total_flowers - daisies)) / 5 :=
by sorry

end rose_flowers_l32_3275


namespace distance_gable_to_citadel_l32_3216

/-- The distance from the point (1600, 1200) to the origin (0, 0) on a complex plane is 2000. -/
theorem distance_gable_to_citadel : 
  Complex.abs (Complex.mk 1600 1200) = 2000 := by
  sorry

end distance_gable_to_citadel_l32_3216


namespace power_calculation_l32_3215

theorem power_calculation : 16^16 * 8^8 / 4^32 = 2^24 := by
  sorry

end power_calculation_l32_3215


namespace sqrt_less_than_2x_iff_x_greater_than_quarter_l32_3250

theorem sqrt_less_than_2x_iff_x_greater_than_quarter (x : ℝ) (hx : x > 0) :
  Real.sqrt x < 2 * x ↔ x > (1 / 4 : ℝ) := by sorry

end sqrt_less_than_2x_iff_x_greater_than_quarter_l32_3250


namespace xyz_equation_solutions_l32_3298

theorem xyz_equation_solutions :
  ∀ (x y z : ℕ), x * y * z = x + y → ((x = 2 ∧ y = 2 ∧ z = 1) ∨ (x = 1 ∧ y = 1 ∧ z = 2)) :=
by sorry

end xyz_equation_solutions_l32_3298


namespace root_equation_implies_sum_l32_3251

theorem root_equation_implies_sum (a b : ℝ) : 
  (1 : ℝ)^2 + a * (1 : ℝ) - b = 0 → a - b + 2023 = 2022 := by
  sorry

end root_equation_implies_sum_l32_3251


namespace map_scale_conversion_l32_3289

/-- Given a map scale where 15 cm represents 90 km, prove that 20 cm represents 120 km -/
theorem map_scale_conversion (scale : ℝ → ℝ) (h1 : scale 15 = 90) : scale 20 = 120 := by
  sorry

end map_scale_conversion_l32_3289


namespace days_at_sisters_house_l32_3287

/-- Calculates the number of days spent at the sister's house during a vacation --/
theorem days_at_sisters_house (total_vacation_days : ℕ) 
  (days_to_grandparents days_at_grandparents days_to_brother days_at_brother 
   days_to_sister days_from_sister : ℕ) : 
  total_vacation_days = 21 →
  days_to_grandparents = 1 →
  days_at_grandparents = 5 →
  days_to_brother = 1 →
  days_at_brother = 5 →
  days_to_sister = 2 →
  days_from_sister = 2 →
  total_vacation_days - (days_to_grandparents + days_at_grandparents + 
    days_to_brother + days_at_brother + days_to_sister + days_from_sister) = 5 := by
  sorry

end days_at_sisters_house_l32_3287


namespace pencil_gain_percentage_l32_3273

/-- Represents the cost price of a single pencil in rupees -/
def cost_price_per_pencil : ℚ := 1 / 12

/-- Represents the selling price of 15 pencils in rupees -/
def selling_price_15 : ℚ := 1

/-- Represents the selling price of 10 pencils in rupees -/
def selling_price_10 : ℚ := 1

/-- The loss percentage when selling 15 pencils for a rupee -/
def loss_percentage : ℚ := 20 / 100

theorem pencil_gain_percentage :
  let cost_15 := 15 * cost_price_per_pencil
  let cost_10 := 10 * cost_price_per_pencil
  selling_price_15 = (1 - loss_percentage) * cost_15 →
  (selling_price_10 - cost_10) / cost_10 = 1 / 5 := by
  sorry

end pencil_gain_percentage_l32_3273


namespace negation_equivalence_l32_3208

theorem negation_equivalence (a b : ℝ) : 
  ¬(a * b = 0 → a = 0 ∨ b = 0) ↔ (a * b ≠ 0 → a ≠ 0 ∧ b ≠ 0) :=
by sorry

end negation_equivalence_l32_3208


namespace zero_not_in_empty_set_l32_3267

theorem zero_not_in_empty_set : 0 ∉ (∅ : Set ℕ) := by
  sorry

end zero_not_in_empty_set_l32_3267


namespace no_half_parallel_diagonals_l32_3296

/-- A regular polygon with n sides -/
structure RegularPolygon where
  n : ℕ
  h : n > 2

/-- The number of diagonals in a polygon -/
def numDiagonals (p : RegularPolygon) : ℕ :=
  p.n * (p.n - 3) / 2

/-- The number of diagonals parallel to sides in a polygon -/
def numParallelDiagonals (p : RegularPolygon) : ℕ :=
  if p.n % 2 = 1 then numDiagonals p else (p.n / 2) - 1

/-- Theorem: No regular polygon has exactly half of its diagonals parallel to its sides -/
theorem no_half_parallel_diagonals (p : RegularPolygon) :
  2 * numParallelDiagonals p ≠ numDiagonals p :=
sorry

end no_half_parallel_diagonals_l32_3296


namespace beach_waders_l32_3277

/-- Proves that 3 people from the first row got up to wade in the water, given the conditions of the beach scenario. -/
theorem beach_waders (first_row : ℕ) (second_row : ℕ) (third_row : ℕ) 
  (h1 : first_row = 24)
  (h2 : second_row = 20)
  (h3 : third_row = 18)
  (h4 : ∃ x : ℕ, first_row - x + (second_row - 5) + third_row = 54) :
  ∃ x : ℕ, x = 3 ∧ first_row - x + (second_row - 5) + third_row = 54 :=
by sorry

end beach_waders_l32_3277


namespace drone_image_trees_l32_3288

theorem drone_image_trees (T : ℕ) (h1 : T ≥ 100) (h2 : T ≥ 90) (h3 : T ≥ 82) : 
  (T - 82) + (T - 82) = 26 := by
sorry

end drone_image_trees_l32_3288


namespace restaurant_cooks_count_l32_3238

theorem restaurant_cooks_count (initial_cooks : ℕ) (initial_waiters : ℕ) 
  (h1 : initial_cooks * 8 = initial_waiters * 3) 
  (h2 : initial_cooks * 4 = (initial_waiters + 12) * 1) : 
  initial_cooks = 9 := by
sorry

end restaurant_cooks_count_l32_3238


namespace factorial_difference_l32_3232

theorem factorial_difference : Nat.factorial 10 - Nat.factorial 9 = 3265920 := by
  sorry

end factorial_difference_l32_3232


namespace correct_operation_l32_3297

theorem correct_operation (a b : ℝ) : 3 * a^2 * b^3 - 2 * a^2 * b^3 = a^2 * b^3 := by
  sorry

end correct_operation_l32_3297


namespace mini_van_tank_capacity_l32_3206

/-- Represents the problem of determining the capacity of a mini-van's tank. -/
theorem mini_van_tank_capacity 
  (service_cost : ℝ) 
  (fuel_cost : ℝ) 
  (num_mini_vans : ℕ) 
  (num_trucks : ℕ) 
  (total_cost : ℝ) 
  (truck_tank_ratio : ℝ) 
  (h1 : service_cost = 2.30)
  (h2 : fuel_cost = 0.70)
  (h3 : num_mini_vans = 4)
  (h4 : num_trucks = 2)
  (h5 : total_cost = 396)
  (h6 : truck_tank_ratio = 2.20) : 
  ∃ (mini_van_capacity : ℝ),
    mini_van_capacity = 65 ∧
    total_cost = 
      (num_mini_vans + num_trucks) * service_cost + 
      (num_mini_vans * mini_van_capacity + num_trucks * (truck_tank_ratio * mini_van_capacity)) * fuel_cost :=
by sorry

end mini_van_tank_capacity_l32_3206


namespace fruit_basket_combinations_l32_3253

def num_apple_options : ℕ := 7
def num_orange_options : ℕ := 13

def total_combinations : ℕ := num_apple_options * num_orange_options

theorem fruit_basket_combinations :
  total_combinations - 1 = 90 := by sorry

end fruit_basket_combinations_l32_3253


namespace problem_one_problem_two_l32_3262

-- Problem 1
theorem problem_one : Real.sqrt 12 - Real.sqrt 3 + 3 * Real.sqrt (1/3) = Real.sqrt 3 + 3 := by
  sorry

-- Problem 2
theorem problem_two : Real.sqrt 18 / Real.sqrt 6 * Real.sqrt 3 = 3 := by
  sorry

end problem_one_problem_two_l32_3262


namespace degree_to_radian_conversion_l32_3245

theorem degree_to_radian_conversion (x : Real) : 
  x * (π / 180) = -5 * π / 3 → x = -300 :=
by sorry

end degree_to_radian_conversion_l32_3245


namespace candle_length_correct_l32_3281

/-- Represents the remaining length of a burning candle after t hours. -/
def candle_length (t : ℝ) : ℝ := 20 - 5 * t

theorem candle_length_correct (t : ℝ) (h : 0 ≤ t ∧ t ≤ 4) : 
  candle_length t = 20 - 5 * t ∧ candle_length t ≥ 0 := by
  sorry

#check candle_length_correct

end candle_length_correct_l32_3281


namespace F_of_4_f_of_5_equals_68_l32_3285

-- Define the function f
def f (a : ℝ) : ℝ := a + 3

-- Define the function F
def F (a b : ℝ) : ℝ := b^2 + a

-- Theorem to prove
theorem F_of_4_f_of_5_equals_68 : F 4 (f 5) = 68 := by
  sorry

end F_of_4_f_of_5_equals_68_l32_3285


namespace ellipse_equation_l32_3227

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ  -- Semi-major axis
  b : ℝ  -- Semi-minor axis

/-- The foci of the ellipse -/
def f1 : Point := ⟨-4, 0⟩
def f2 : Point := ⟨4, 0⟩

/-- Distance between foci -/
def focalDistance : ℝ := 8

/-- Maximum area of triangle PF₁F₂ -/
def maxTriangleArea : ℝ := 12

/-- Theorem: Given an ellipse with foci at (-4,0) and (4,0), and maximum area of triangle PF₁F₂ is 12,
    the equation of the ellipse is x²/25 + y²/9 = 1 -/
theorem ellipse_equation (e : Ellipse) : 
  (focalDistance = 8) → 
  (maxTriangleArea = 12) → 
  (e.a^2 = 25 ∧ e.b^2 = 9) := by
  sorry

#check ellipse_equation

end ellipse_equation_l32_3227


namespace arithmetic_sequence_max_third_term_l32_3244

/-- An arithmetic sequence with a positive first term and a_1 * a_2 = -2 -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  a 1 > 0 ∧ a 1 * a 2 = -2 ∧ ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The common difference of an arithmetic sequence -/
def CommonDifference (a : ℕ → ℝ) : ℝ :=
  (a 2) - (a 1)

/-- The third term of an arithmetic sequence -/
def ThirdTerm (a : ℕ → ℝ) : ℝ :=
  a 3

theorem arithmetic_sequence_max_third_term
  (a : ℕ → ℝ) (h : ArithmeticSequence a) :
  (∀ d : ℝ, CommonDifference a ≤ d → ThirdTerm a ≤ ThirdTerm (fun n ↦ a 1 + (n - 1) * d)) →
  CommonDifference a = -3 := by
  sorry

end arithmetic_sequence_max_third_term_l32_3244


namespace linear_function_proof_l32_3228

/-- A linear function passing through (-2, -1) and parallel to y = 2x - 3 -/
def f (x : ℝ) : ℝ := 2 * x + 3

theorem linear_function_proof :
  (∀ x y : ℝ, f y - f x = 2 * (y - x)) ∧  -- linearity and slope
  f (-2) = -1 ∧                           -- passes through (-2, -1)
  (∀ x : ℝ, f x - (2 * x - 3) = 3) :=     -- parallel to y = 2x - 3
by sorry

end linear_function_proof_l32_3228


namespace system_inequalities_solution_range_l32_3246

theorem system_inequalities_solution_range (a : ℚ) : 
  (∃! (s : Finset ℤ), s.card = 5 ∧ 
    (∀ x : ℤ, x ∈ s ↔ (((2 * x + 5 : ℚ) / 3 > x - 5) ∧ ((x + 3 : ℚ) / 2 < x + a)))) →
  (-6 < a ∧ a ≤ -11/2) :=
sorry

end system_inequalities_solution_range_l32_3246


namespace one_in_set_zero_one_l32_3217

theorem one_in_set_zero_one : 1 ∈ ({0, 1} : Set ℕ) := by
  sorry

end one_in_set_zero_one_l32_3217


namespace can_find_genuine_coin_l32_3243

/-- Represents the result of a weighing -/
inductive WeighResult
  | Equal : WeighResult
  | LeftHeavier : WeighResult
  | RightHeavier : WeighResult

/-- Represents a group of coins -/
structure CoinGroup where
  size : Nat
  counterfeit : Nat

/-- Represents the state of coins -/
structure CoinState where
  total : Nat
  counterfeit : Nat

/-- Represents a weighing operation -/
def weigh (left right : CoinGroup) : WeighResult :=
  sorry

/-- Represents the process of finding a genuine coin -/
def findGenuineCoin (state : CoinState) : Prop :=
  ∃ (g1 g2 g3 : CoinGroup),
    g1.size + g2.size + g3.size = state.total ∧
    g1.counterfeit + g2.counterfeit + g3.counterfeit = state.counterfeit ∧
    (∃ (result : WeighResult),
      result = weigh g1 g2 ∧
      (result = WeighResult.Equal →
        ∃ (c1 c2 : CoinGroup),
          c1.size = 1 ∧ c2.size = 1 ∧
          c1.size + c2.size ≤ g3.size ∧
          (weigh c1 c2 = WeighResult.Equal ∨
           weigh c1 c2 = WeighResult.LeftHeavier ∨
           weigh c1 c2 = WeighResult.RightHeavier)) ∧
      ((result = WeighResult.LeftHeavier ∨ result = WeighResult.RightHeavier) →
        ∃ (c1 c2 : CoinGroup),
          c1.size = 1 ∧ c2.size = 1 ∧
          (c1.size ≤ g1.size ∧ c2.size ≤ g2.size) ∧
          (weigh c1 c2 = WeighResult.Equal ∨
           weigh c1 c2 = WeighResult.LeftHeavier ∨
           weigh c1 c2 = WeighResult.RightHeavier)))

theorem can_find_genuine_coin (state : CoinState)
  (h1 : state.total = 100)
  (h2 : state.counterfeit = 4)
  (h3 : state.counterfeit < state.total) :
  findGenuineCoin state :=
  sorry

end can_find_genuine_coin_l32_3243


namespace quadratic_equation_solution_l32_3213

theorem quadratic_equation_solution : ∃ (x c d : ℕ), 
  (x^2 + 14*x = 84) ∧ 
  (x = Real.sqrt c - d) ∧ 
  (c > 0) ∧ 
  (d > 0) ∧ 
  (c + d = 140) := by
sorry

end quadratic_equation_solution_l32_3213


namespace max_m_value_max_m_is_75_l32_3280

theorem max_m_value (m n : ℕ+) (h : 8 * m + 9 * n = m * n + 6) : 
  ∀ k : ℕ+, 8 * k + 9 * n = k * n + 6 → k ≤ m :=
sorry

theorem max_m_is_75 : ∃ m n : ℕ+, 8 * m + 9 * n = m * n + 6 ∧ m = 75 :=
sorry

end max_m_value_max_m_is_75_l32_3280


namespace polynomial_inequality_l32_3294

theorem polynomial_inequality (x : ℝ) : x * (x + 1) * (x + 2) * (x + 3) ≥ -1 := by
  sorry

end polynomial_inequality_l32_3294


namespace ab_product_theorem_l32_3261

theorem ab_product_theorem (a b : ℝ) 
  (h1 : (27 : ℝ) ^ a = 3 ^ (10 * (b + 2)))
  (h2 : (125 : ℝ) ^ b = 5 ^ (a - 3)) : 
  a * b = 330 := by sorry

end ab_product_theorem_l32_3261


namespace conditional_probability_coin_flips_l32_3283

-- Define the sample space for two coin flips
def CoinFlip := Bool × Bool

-- Define the probability measure
noncomputable def P : Set CoinFlip → ℝ := sorry

-- Define event A: heads on the first flip
def A : Set CoinFlip := {x | x.1 = true}

-- Define event B: heads on the second flip
def B : Set CoinFlip := {x | x.2 = true}

-- Define the intersection of events A and B
def AB : Set CoinFlip := A ∩ B

-- State the theorem
theorem conditional_probability_coin_flips :
  P B / P A = 1 / 2 := by sorry

end conditional_probability_coin_flips_l32_3283


namespace no_real_solutions_l32_3272

theorem no_real_solutions :
  ¬∃ (x : ℝ), x ≠ -9 ∧ (8*x^2 + 90*x + 2) / (3*x + 27) = 4*x + 2 := by
  sorry

end no_real_solutions_l32_3272


namespace committee_selection_ways_l32_3278

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem committee_selection_ways : choose 12 5 = 792 := by
  sorry

end committee_selection_ways_l32_3278


namespace distance_swam_against_current_l32_3242

/-- Proves that the distance swam against the current is 10 km -/
theorem distance_swam_against_current
  (still_water_speed : ℝ)
  (water_speed : ℝ)
  (time_taken : ℝ)
  (h1 : still_water_speed = 12)
  (h2 : water_speed = 2)
  (h3 : time_taken = 1) :
  still_water_speed - water_speed * time_taken = 10 := by
  sorry

end distance_swam_against_current_l32_3242


namespace exponential_inverse_existence_uniqueness_l32_3274

theorem exponential_inverse_existence_uniqueness (a x : ℝ) (ha : 0 < a) (ha_neq : a ≠ 1) (hx : 0 < x) :
  ∃! y : ℝ, a^y = x :=
by sorry

end exponential_inverse_existence_uniqueness_l32_3274


namespace least_product_of_distinct_primes_above_50_l32_3266

theorem least_product_of_distinct_primes_above_50 : 
  ∃ (p q : ℕ), 
    p.Prime ∧ 
    q.Prime ∧ 
    p ≠ q ∧ 
    p > 50 ∧ 
    q > 50 ∧ 
    p * q = 3127 ∧ 
    ∀ (r s : ℕ), r.Prime → s.Prime → r ≠ s → r > 50 → s > 50 → r * s ≥ 3127 :=
by sorry

end least_product_of_distinct_primes_above_50_l32_3266


namespace line_ellipse_intersection_slopes_l32_3231

/-- Given a line y = mx - 3 intersecting the ellipse 4x^2 + 25y^2 = 100,
    the possible slopes m satisfy m^2 ≥ 4/41 -/
theorem line_ellipse_intersection_slopes (m : ℝ) : 
  (∃ x y : ℝ, 4 * x^2 + 25 * y^2 = 100 ∧ y = m * x - 3) → m^2 ≥ 4/41 := by
  sorry

end line_ellipse_intersection_slopes_l32_3231


namespace total_nails_needed_l32_3292

/-- The total number of nails needed is equal to the sum of initial nails, 
    found nails, and nails to buy. -/
theorem total_nails_needed 
  (initial_nails : ℕ) 
  (found_nails : ℕ) 
  (nails_to_buy : ℕ) : 
  initial_nails + found_nails + nails_to_buy = 
  initial_nails + found_nails + nails_to_buy := by
  sorry

#eval 247 + 144 + 109

end total_nails_needed_l32_3292


namespace jellybean_distribution_l32_3252

theorem jellybean_distribution (total_jellybeans : ℕ) (nephews : ℕ) (nieces : ℕ) 
  (h1 : total_jellybeans = 70)
  (h2 : nephews = 3)
  (h3 : nieces = 2) :
  total_jellybeans / (nephews + nieces) = 14 := by
  sorry

end jellybean_distribution_l32_3252


namespace dilation_determinant_l32_3271

/-- A dilation matrix with scale factor k -/
def dilationMatrix (k : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![k, 0; 0, k]

/-- The determinant of a 2x2 matrix -/
def det2x2 (M : Matrix (Fin 2) (Fin 2) ℝ) : ℝ :=
  M 0 0 * M 1 1 - M 0 1 * M 1 0

theorem dilation_determinant :
  let E := dilationMatrix 12
  det2x2 E = 144 := by sorry

end dilation_determinant_l32_3271


namespace distinct_integers_count_l32_3270

def odd_squares_list : List ℤ :=
  (List.range 500).map (fun k => ⌊((2*k + 1)^2 : ℚ) / 500⌋)

theorem distinct_integers_count : (odd_squares_list.eraseDups).length = 469 := by
  sorry

end distinct_integers_count_l32_3270


namespace problem_solution_l32_3230

theorem problem_solution (a b : ℤ) (h : 6 * a * b = 9 * a - 10 * b + 303) : a + b = 15 := by
  sorry

end problem_solution_l32_3230


namespace rachel_math_problems_l32_3255

def problems_per_minute : ℕ := 5
def minutes_solved : ℕ := 12
def problems_second_day : ℕ := 16

theorem rachel_math_problems :
  problems_per_minute * minutes_solved + problems_second_day = 76 := by
  sorry

end rachel_math_problems_l32_3255


namespace relay_race_time_l32_3279

/-- Represents the time taken by each runner in the relay race -/
structure RelayTimes where
  mary : ℕ
  susan : ℕ
  jen : ℕ
  tiffany : ℕ

/-- Calculates the total time of the relay race -/
def total_time (times : RelayTimes) : ℕ :=
  times.mary + times.susan + times.jen + times.tiffany

/-- Theorem stating that the total time of the relay race is 223 seconds -/
theorem relay_race_time : ∃ (times : RelayTimes), 
  times.mary = 2 * times.susan ∧
  times.susan = times.jen + 10 ∧
  times.jen = 30 ∧
  times.tiffany = times.mary - 7 ∧
  total_time times = 223 := by
  sorry


end relay_race_time_l32_3279


namespace improper_fraction_decomposition_l32_3218

theorem improper_fraction_decomposition (x : ℝ) :
  (6 * x^3 + 5 * x^2 + 3 * x - 4) / (x^2 + 4) = 6 * x + 5 - (21 * x + 24) / (x^2 + 4) := by
  sorry

end improper_fraction_decomposition_l32_3218


namespace die_roll_probability_l32_3254

def standard_die : Finset ℕ := {1, 2, 3, 4, 5, 6}

def roll_twice : Finset (ℕ × ℕ) :=
  standard_die.product standard_die

def favorable_outcomes : Finset (ℕ × ℕ) :=
  {(1, 3), (2, 6)}

theorem die_roll_probability :
  (favorable_outcomes.card : ℚ) / roll_twice.card = 1 / 18 := by
  sorry

end die_roll_probability_l32_3254


namespace ship_passengers_theorem_l32_3220

theorem ship_passengers_theorem (total_passengers : ℝ) (round_trip_with_car : ℝ) 
  (round_trip_without_car : ℝ) (h1 : round_trip_with_car > 0) 
  (h2 : round_trip_with_car + round_trip_without_car ≤ total_passengers) 
  (h3 : round_trip_with_car / total_passengers = 0.3) :
  (round_trip_with_car + round_trip_without_car) / total_passengers = 
  round_trip_with_car / total_passengers := by
sorry

end ship_passengers_theorem_l32_3220


namespace unique_lock_code_satisfies_conditions_unique_lock_code_is_unique_l32_3259

/-- Represents a seven-digit lock code -/
structure LockCode where
  digits : Fin 7 → Nat
  first_three_same : ∀ i j, i < 3 → j < 3 → digits i = digits j
  last_four_same : ∀ i j, 3 ≤ i → i < 7 → 3 ≤ j → j < 7 → digits i = digits j
  all_digits : ∀ i, digits i < 10

/-- The sum of digits in a lock code -/
def digit_sum (code : LockCode) : Nat :=
  (Finset.range 7).sum (λ i => code.digits i)

/-- The unique lock code satisfying all conditions -/
def unique_lock_code : LockCode where
  digits := λ i => if i < 3 then 3 else 7
  first_three_same := by sorry
  last_four_same := by sorry
  all_digits := by sorry

theorem unique_lock_code_satisfies_conditions :
  let s := digit_sum unique_lock_code
  (10 ≤ s ∧ s < 100) ∧
  (s / 10 = unique_lock_code.digits 0) ∧
  (s % 10 = unique_lock_code.digits 6) :=
by sorry

theorem unique_lock_code_is_unique (code : LockCode) :
  let s := digit_sum code
  (10 ≤ s ∧ s < 100) →
  (s / 10 = code.digits 0) →
  (s % 10 = code.digits 6) →
  code = unique_lock_code :=
by sorry

end unique_lock_code_satisfies_conditions_unique_lock_code_is_unique_l32_3259


namespace chocolate_division_l32_3207

theorem chocolate_division (total_chocolate : ℚ) (num_piles : ℕ) (piles_for_shaina : ℕ) : 
  total_chocolate = 70 / 7 →
  num_piles = 5 →
  piles_for_shaina = 2 →
  (total_chocolate / num_piles) * piles_for_shaina = 4 := by
  sorry

end chocolate_division_l32_3207


namespace modulus_of_complex_fraction_l32_3260

theorem modulus_of_complex_fraction :
  let z : ℂ := (-3 + Complex.I) / (2 + Complex.I)
  Complex.abs z = Real.sqrt 2 := by
  sorry

end modulus_of_complex_fraction_l32_3260


namespace dani_pants_reward_l32_3249

/-- The number of pairs of pants Dani gets each year -/
def pants_per_year (initial_pants : ℕ) (pants_after_5_years : ℕ) : ℕ :=
  ((pants_after_5_years - initial_pants) / 5) / 2

/-- Theorem stating that Dani gets 4 pairs of pants each year -/
theorem dani_pants_reward (initial_pants : ℕ) (pants_after_5_years : ℕ) 
  (h1 : initial_pants = 50) 
  (h2 : pants_after_5_years = 90) : 
  pants_per_year initial_pants pants_after_5_years = 4 := by
  sorry

end dani_pants_reward_l32_3249


namespace total_income_calculation_l32_3258

def original_cupcake_price : ℚ := 3
def original_cookie_price : ℚ := 2
def cupcake_discount : ℚ := 0.3
def cookie_discount : ℚ := 0.45
def cupcakes_sold : ℕ := 25
def cookies_sold : ℕ := 18

theorem total_income_calculation :
  let new_cupcake_price := original_cupcake_price * (1 - cupcake_discount)
  let new_cookie_price := original_cookie_price * (1 - cookie_discount)
  let total_income := (new_cupcake_price * cupcakes_sold) + (new_cookie_price * cookies_sold)
  total_income = 72.3 := by sorry

end total_income_calculation_l32_3258


namespace characterization_of_M_l32_3237

/-- S(n) represents the sum of digits of n -/
def S (n : ℕ) : ℕ := sorry

/-- M satisfies the given property -/
def satisfies_property (M : ℕ) : Prop :=
  M > 0 ∧ ∀ k : ℕ, 0 < k ∧ k ≤ M → S (M * k) = S M

/-- Main theorem -/
theorem characterization_of_M :
  ∀ M : ℕ, satisfies_property M ↔ ∃ n : ℕ, n > 0 ∧ M = 10^n - 1 :=
sorry

end characterization_of_M_l32_3237


namespace total_running_time_l32_3236

/-- Represents the running data for a single day -/
structure DailyRun where
  distance : ℕ
  basePace : ℕ
  additionalTime : ℕ

/-- Calculates the total time for a single run -/
def runTime (run : DailyRun) : ℕ :=
  run.distance * (run.basePace + run.additionalTime)

/-- The running data for each day of the week -/
def weeklyRuns : List DailyRun :=
  [
    { distance := 3, basePace := 10, additionalTime := 1 },  -- Monday
    { distance := 4, basePace := 9,  additionalTime := 1 },  -- Tuesday
    { distance := 6, basePace := 12, additionalTime := 0 },  -- Wednesday
    { distance := 8, basePace := 8,  additionalTime := 2 },  -- Thursday
    { distance := 3, basePace := 10, additionalTime := 0 }   -- Friday
  ]

/-- The theorem stating that the total running time for the week is 255 minutes -/
theorem total_running_time :
  (weeklyRuns.map runTime).sum = 255 := by
  sorry


end total_running_time_l32_3236


namespace measure_45_minutes_l32_3235

/-- Represents a cord that can be burned --/
structure Cord :=
  (burn_time : ℝ)
  (burn_rate_uniform : Bool)

/-- Represents the state of burning a cord --/
inductive BurnState
  | Unlit
  | LitOneEnd (time : ℝ)
  | LitBothEnds (time : ℝ)
  | Burned

/-- Represents the measurement setup --/
structure MeasurementSetup :=
  (cord1 : Cord)
  (cord2 : Cord)
  (state1 : BurnState)
  (state2 : BurnState)

/-- The main theorem stating that 45 minutes can be measured --/
theorem measure_45_minutes 
  (c1 c2 : Cord) 
  (h1 : c1.burn_time = 60) 
  (h2 : c2.burn_time = 60) : 
  ∃ (process : List MeasurementSetup), 
    (∃ (t : ℝ), t = 45 ∧ 
      (∃ (final : MeasurementSetup), final ∈ process ∧ 
        final.state1 = BurnState.Burned ∧ 
        final.state2 = BurnState.Burned)) :=
sorry

end measure_45_minutes_l32_3235


namespace train_length_proof_l32_3226

/-- Proves that the length of a train is 260 meters, given its speed and the time it takes to cross a platform of known length. -/
theorem train_length_proof (train_speed : ℝ) (platform_length : ℝ) (crossing_time : ℝ) :
  train_speed = 72 * (1000 / 3600) →
  platform_length = 260 →
  crossing_time = 26 →
  (train_speed * crossing_time) - platform_length = 260 :=
by sorry

end train_length_proof_l32_3226


namespace factor_t_squared_minus_64_l32_3222

theorem factor_t_squared_minus_64 (t : ℝ) : t^2 - 64 = (t - 8) * (t + 8) := by
  sorry

end factor_t_squared_minus_64_l32_3222


namespace square_of_difference_negative_first_l32_3286

theorem square_of_difference_negative_first (x y : ℝ) : (-x + y)^2 = x^2 - 2*x*y + y^2 := by
  sorry

end square_of_difference_negative_first_l32_3286


namespace infinitely_many_squares_l32_3209

theorem infinitely_many_squares (k : ℕ+) :
  ∀ (B : ℕ), ∃ (n m : ℕ), n > B ∧ m > B ∧ (2 * k.val * n - 7 = m^2) :=
sorry

end infinitely_many_squares_l32_3209


namespace jerry_vote_difference_l32_3295

def total_votes : ℕ := 196554
def jerry_votes : ℕ := 108375

theorem jerry_vote_difference : 
  jerry_votes - (total_votes - jerry_votes) = 20196 := by
  sorry

end jerry_vote_difference_l32_3295


namespace office_age_problem_l32_3214

theorem office_age_problem (total_persons : ℕ) (avg_age_all : ℕ) 
  (group1_persons : ℕ) (avg_age_group1 : ℕ) (group2_persons : ℕ) 
  (age_15th_person : ℕ) : 
  total_persons = 16 → 
  avg_age_all = 15 → 
  group1_persons = 5 → 
  avg_age_group1 = 14 → 
  group2_persons = 9 → 
  age_15th_person = 26 → 
  (avg_age_all * total_persons - avg_age_group1 * group1_persons - age_15th_person) / group2_persons = 16 := by
sorry

end office_age_problem_l32_3214


namespace max_a_value_l32_3229

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2 - 4*x + 3 else -x^2 - 2*x + 3

theorem max_a_value (a : ℝ) :
  (∀ x ∈ Set.Icc a (a + 1), f (x + a) ≥ f (2*a - x)) →
  a ≤ -2 :=
sorry

end max_a_value_l32_3229


namespace pig_price_calculation_l32_3276

/-- Given the total cost of 3 pigs and 10 hens, and the average price of a hen,
    calculate the average price of a pig. -/
theorem pig_price_calculation (total_cost hen_price : ℚ) 
    (h1 : total_cost = 1200)
    (h2 : hen_price = 30) : 
    (total_cost - 10 * hen_price) / 3 = 300 :=
by sorry

end pig_price_calculation_l32_3276


namespace double_volume_double_capacity_l32_3282

/-- Represents the capacity of a container in number of marbles -/
def ContainerCapacity (volume : ℝ) : ℝ := sorry

theorem double_volume_double_capacity :
  let v₁ : ℝ := 36
  let v₂ : ℝ := 72
  let c₁ : ℝ := 120
  ContainerCapacity v₁ = c₁ →
  ContainerCapacity v₂ = 2 * c₁ :=
by sorry

end double_volume_double_capacity_l32_3282


namespace lcm_problem_l32_3212

theorem lcm_problem (a b c : ℕ) : 
  lcm a b = 24 → lcm b c = 28 → 
  ∃ (m : ℕ), m = lcm a c ∧ ∀ (n : ℕ), n = lcm a c → m ≤ n := by
sorry

end lcm_problem_l32_3212


namespace village_population_after_events_l32_3225

theorem village_population_after_events (initial_population : ℕ) : 
  initial_population = 7600 → 
  (initial_population - initial_population / 10 - 
   (initial_population - initial_population / 10) / 4) = 5130 := by
sorry

end village_population_after_events_l32_3225


namespace smallest_result_l32_3240

def S : Finset ℕ := {2, 4, 6, 8, 10, 12}

def process (a b c : ℕ) : ℕ := (a + b) * c

def valid_choice (a b c : ℕ) : Prop :=
  a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c

theorem smallest_result :
  ∃ (a b c : ℕ), valid_choice a b c ∧
  process a b c = 20 ∧
  ∀ (x y z : ℕ), valid_choice x y z → process x y z ≥ 20 :=
sorry

end smallest_result_l32_3240


namespace eight_faucets_fill_time_l32_3201

/-- The time (in seconds) it takes for a given number of faucets to fill a tub of a given volume -/
def fill_time (num_faucets : ℕ) (volume : ℝ) : ℝ :=
  -- Definition to be filled based on the problem conditions
  sorry

theorem eight_faucets_fill_time :
  -- Given conditions
  (fill_time 4 200 = 8 * 60) →  -- 4 faucets fill 200 gallons in 8 minutes (converted to seconds)
  (∀ n v, fill_time n v = fill_time 1 v / n) →  -- All faucets dispense water at the same rate
  -- Conclusion
  (fill_time 8 50 = 60) :=
by
  sorry

end eight_faucets_fill_time_l32_3201


namespace school_play_chairs_l32_3211

theorem school_play_chairs (chairs_per_row : ℕ) (unoccupied_seats : ℕ) (occupied_seats : ℕ) :
  chairs_per_row = 20 →
  unoccupied_seats = 10 →
  occupied_seats = 790 →
  (occupied_seats + unoccupied_seats) / chairs_per_row = 40 :=
by
  sorry

end school_play_chairs_l32_3211


namespace auction_tv_initial_price_l32_3256

/-- Given an auction event where:
    - The price of a TV increased by 2/5 times its initial price
    - The price of a phone, initially $400, increased by 40%
    - The total amount received after sale is $1260
    Prove that the initial price of the TV was $500 -/
theorem auction_tv_initial_price (tv_initial : ℝ) (phone_initial : ℝ) (total : ℝ) :
  phone_initial = 400 →
  total = 1260 →
  total = (tv_initial + 2/5 * tv_initial) + (phone_initial + 0.4 * phone_initial) →
  tv_initial = 500 := by
  sorry


end auction_tv_initial_price_l32_3256


namespace average_of_combined_sets_l32_3219

theorem average_of_combined_sets (m n : ℕ) (a b : ℝ) :
  let sum_m := m * a
  let sum_n := n * b
  (sum_m + sum_n) / (m + n) = (a * m + b * n) / (m + n) := by
  sorry

end average_of_combined_sets_l32_3219


namespace arithmetic_expression_equality_l32_3269

theorem arithmetic_expression_equality : 3 + 15 / 3 - 2^2 + 1 = 5 := by
  sorry

end arithmetic_expression_equality_l32_3269


namespace jenny_recycling_money_is_160_l32_3264

/-- Calculates the money Jenny makes from recycling cans and bottles -/
def jenny_recycling_money : ℕ :=
let bottle_weight : ℕ := 6
let can_weight : ℕ := 2
let total_capacity : ℕ := 100
let cans_collected : ℕ := 20
let bottle_price : ℕ := 10
let can_price : ℕ := 3
let remaining_capacity : ℕ := total_capacity - (can_weight * cans_collected)
let bottles_collected : ℕ := remaining_capacity / bottle_weight
bottles_collected * bottle_price + cans_collected * can_price

theorem jenny_recycling_money_is_160 :
  jenny_recycling_money = 160 := by
sorry

end jenny_recycling_money_is_160_l32_3264


namespace expression_evaluation_l32_3223

theorem expression_evaluation : (60 + 5 * 12) / (Real.sqrt 180 / 3) ^ 2 = 6 := by
  sorry

end expression_evaluation_l32_3223


namespace xyz_maximum_l32_3203

theorem xyz_maximum (x y z : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z)
  (h_eq : x * y - z = (x - z) * (y - z)) (h_sum : x + y + z = 1) :
  x * y * z ≤ 1 / 27 :=
sorry

end xyz_maximum_l32_3203


namespace complement_of_A_in_U_l32_3224

def U : Set Int := {x | x^2 ≤ 2*x + 3}
def A : Set Int := {0, 1, 2}

theorem complement_of_A_in_U :
  (U \ A) = {-1, 3} := by sorry

end complement_of_A_in_U_l32_3224


namespace sum_inverse_max_min_S_l32_3263

/-- Given real numbers x and y satisfying 4x^2 - 5xy + 4y^2 = 5,
    and S defined as x^2 + y^2, prove that the maximum and minimum
    values of S exist, and 1/S_max + 1/S_min = 8/5. -/
theorem sum_inverse_max_min_S :
  ∃ (S_max S_min : ℝ),
    (∀ x y : ℝ, 4 * x^2 - 5 * x * y + 4 * y^2 = 5 →
      let S := x^2 + y^2
      S ≤ S_max ∧ S_min ≤ S) ∧
    1 / S_max + 1 / S_min = 8 / 5 := by
  sorry

#check sum_inverse_max_min_S

end sum_inverse_max_min_S_l32_3263


namespace distinct_power_tower_values_l32_3290

def power_tower (base : ℕ) (height : ℕ) : ℕ :=
  match height with
  | 0 => 1
  | n + 1 => base ^ (power_tower base n)

def parenthesized_expressions (base : ℕ) (height : ℕ) : Finset ℕ :=
  sorry

theorem distinct_power_tower_values :
  (parenthesized_expressions 3 4).card = 5 :=
sorry

end distinct_power_tower_values_l32_3290


namespace sundae_price_l32_3284

/-- Proves that the price of each sundae is $0.60 given the conditions of the catering order --/
theorem sundae_price (ice_cream_bars sundaes : ℕ) (total_price ice_cream_price : ℚ) :
  ice_cream_bars = 200 →
  sundaes = 200 →
  total_price = 200 →
  ice_cream_price = 0.4 →
  (total_price - ice_cream_bars * ice_cream_price) / sundaes = 0.6 :=
by
  sorry

#eval (200 : ℚ) - 200 * 0.4  -- Expected output: 120
#eval 120 / 200              -- Expected output: 0.6

end sundae_price_l32_3284


namespace union_covers_reals_l32_3221

def set_A : Set ℝ := {x | x ≤ 2}
def set_B (a : ℝ) : Set ℝ := {x | x > a}

theorem union_covers_reals (a : ℝ) :
  set_A ∪ set_B a = Set.univ → a ≤ 2 := by
  sorry

end union_covers_reals_l32_3221


namespace hannah_easter_eggs_l32_3293

theorem hannah_easter_eggs 
  (total : ℕ) 
  (helen : ℕ) 
  (hannah : ℕ) 
  (h1 : total = 63)
  (h2 : hannah = 2 * helen)
  (h3 : total = helen + hannah) : 
  hannah = 42 := by
sorry

end hannah_easter_eggs_l32_3293


namespace correct_transformation_l32_3239

theorem correct_transformation : (-2 : ℚ) * (1/2 : ℚ) * (-5 : ℚ) = 5 := by
  sorry

end correct_transformation_l32_3239


namespace rectangle_circle_area_ratio_l32_3265

/-- Given a rectangle with perimeter equal to the circumference of a circle,
    and the length of the rectangle is twice its width,
    prove that the ratio of the area of the rectangle to the area of the circle is 2π/9. -/
theorem rectangle_circle_area_ratio (w : ℝ) (r : ℝ) (h1 : w > 0) (h2 : r > 0) :
  let l := 2 * w
  let rectangle_perimeter := 2 * l + 2 * w
  let circle_circumference := 2 * Real.pi * r
  let rectangle_area := l * w
  let circle_area := Real.pi * r^2
  rectangle_perimeter = circle_circumference →
  rectangle_area / circle_area = 2 * Real.pi / 9 := by
sorry

end rectangle_circle_area_ratio_l32_3265


namespace freds_marbles_l32_3248

theorem freds_marbles (total : ℕ) (dark_blue red green : ℕ) : 
  dark_blue ≥ total / 3 →
  red = 38 →
  green = 4 →
  total = dark_blue + red + green →
  total ≥ 63 :=
by sorry

end freds_marbles_l32_3248


namespace max_profit_is_270000_l32_3257

/-- Represents the production and profit details for a company's two products. -/
structure ProductionProblem where
  materialA_for_A : ℝ  -- tons of Material A needed for 1 ton of Product A
  materialB_for_A : ℝ  -- tons of Material B needed for 1 ton of Product A
  materialA_for_B : ℝ  -- tons of Material A needed for 1 ton of Product B
  materialB_for_B : ℝ  -- tons of Material B needed for 1 ton of Product B
  profit_A : ℝ         -- profit (in RMB) for 1 ton of Product A
  profit_B : ℝ         -- profit (in RMB) for 1 ton of Product B
  max_materialA : ℝ    -- maximum available tons of Material A
  max_materialB : ℝ    -- maximum available tons of Material B

/-- Calculates the maximum profit given the production constraints. -/
def maxProfit (p : ProductionProblem) : ℝ :=
  sorry

/-- States that the maximum profit for the given problem is 270,000 RMB. -/
theorem max_profit_is_270000 (p : ProductionProblem) 
  (h1 : p.materialA_for_A = 3)
  (h2 : p.materialB_for_A = 2)
  (h3 : p.materialA_for_B = 1)
  (h4 : p.materialB_for_B = 3)
  (h5 : p.profit_A = 50000)
  (h6 : p.profit_B = 30000)
  (h7 : p.max_materialA = 13)
  (h8 : p.max_materialB = 18) :
  maxProfit p = 270000 :=
by sorry

end max_profit_is_270000_l32_3257


namespace expression_evaluation_l32_3291

theorem expression_evaluation : 121 + 2 * 11 * 4 + 16 + 7 = 232 := by
  sorry

end expression_evaluation_l32_3291


namespace circle_equation_l32_3234

/-- The equation of a circle with center (-1, 2) and radius √5 is x² + y² + 2x - 4y = 0 -/
theorem circle_equation (x y : ℝ) :
  let center : ℝ × ℝ := (-1, 2)
  let radius : ℝ := Real.sqrt 5
  (x - center.1)^2 + (y - center.2)^2 = radius^2 ↔ x^2 + y^2 + 2*x - 4*y = 0 :=
by sorry

end circle_equation_l32_3234


namespace negation_equivalence_l32_3268

theorem negation_equivalence :
  (¬ ∃ x ∈ Set.Ioo (-1 : ℝ) 1, x^2 + 2*x ≤ 1) ↔ 
  (∀ x ∈ Set.Ioo (-1 : ℝ) 1, x^2 + 2*x > 1) :=
by sorry

end negation_equivalence_l32_3268


namespace average_salary_proof_l32_3210

def salary_a : ℕ := 8000
def salary_b : ℕ := 5000
def salary_c : ℕ := 15000
def salary_d : ℕ := 7000
def salary_e : ℕ := 9000

def total_salary : ℕ := salary_a + salary_b + salary_c + salary_d + salary_e
def num_individuals : ℕ := 5

theorem average_salary_proof :
  total_salary / num_individuals = 9000 := by
  sorry

end average_salary_proof_l32_3210


namespace negation_of_universal_real_proposition_l32_3241

theorem negation_of_universal_real_proposition :
  (¬ ∀ x : ℝ, x^2 ≠ x) ↔ (∃ x : ℝ, x^2 = x) :=
by sorry

end negation_of_universal_real_proposition_l32_3241

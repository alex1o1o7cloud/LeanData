import Mathlib

namespace video_release_week_l1284_128419

/-- Proves that the number of days in a week is 7, given John's video release schedule --/
theorem video_release_week (short_video_length : ℕ) (long_video_multiplier : ℕ) 
  (videos_per_day : ℕ) (short_videos_per_day : ℕ) (total_weekly_minutes : ℕ) :
  short_video_length = 2 →
  long_video_multiplier = 6 →
  videos_per_day = 3 →
  short_videos_per_day = 2 →
  total_weekly_minutes = 112 →
  (total_weekly_minutes / (short_videos_per_day * short_video_length + 
    (videos_per_day - short_videos_per_day) * (long_video_multiplier * short_video_length))) = 7 := by
  sorry

#check video_release_week

end video_release_week_l1284_128419


namespace solve_equation_l1284_128428

theorem solve_equation (x : ℝ) : (x ^ 3).sqrt = 9 * (81 ^ (1 / 9 : ℝ)) → x = 9 := by
  sorry

end solve_equation_l1284_128428


namespace middle_circle_radius_l1284_128466

/-- Configuration of five circles tangent to each other and two parallel lines -/
structure CircleConfiguration where
  /-- Radius of the smallest circle -/
  r_min : ℝ
  /-- Radius of the largest circle -/
  r_max : ℝ
  /-- Radius of the middle circle -/
  r_mid : ℝ

/-- The theorem stating the relationship between the radii of the circles -/
theorem middle_circle_radius (c : CircleConfiguration) 
  (h_min : c.r_min = 12)
  (h_max : c.r_max = 24) :
  c.r_mid = 12 * Real.sqrt 2 := by
  sorry


end middle_circle_radius_l1284_128466


namespace max_profit_toy_sales_exists_max_profit_price_l1284_128424

/-- Represents the profit function for toy sales -/
def profit_function (x : ℝ) : ℝ := -10 * x^2 + 1300 * x - 30000

/-- Represents the sales volume function for toy sales -/
def sales_volume (x : ℝ) : ℝ := -10 * x + 1000

/-- The maximum profit theorem for toy sales -/
theorem max_profit_toy_sales :
  ∀ x : ℝ,
  (x ≥ 44) →
  (x ≤ 46) →
  (sales_volume x ≥ 540) →
  profit_function x ≤ 8640 :=
by
  sorry

/-- The existence of a selling price that achieves the maximum profit -/
theorem exists_max_profit_price :
  ∃ x : ℝ,
  (x ≥ 44) ∧
  (x ≤ 46) ∧
  (sales_volume x ≥ 540) ∧
  profit_function x = 8640 :=
by
  sorry

end max_profit_toy_sales_exists_max_profit_price_l1284_128424


namespace consecutive_even_sum_l1284_128441

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

def is_consecutive_even (a b c : ℕ) : Prop :=
  is_even a ∧ is_even b ∧ is_even c ∧ b = a + 2 ∧ c = b + 2

def has_valid_digits (n : ℕ) : Prop :=
  n ≥ 20000 ∧ n < 30000 ∧
  n % 10 = 0 ∧
  (n / 10000 : ℕ) = 2 ∧
  ((n / 10) % 10 ≠ (n / 100) % 10) ∧
  ((n / 10) % 10 ≠ (n / 1000) % 10) ∧
  ((n / 100) % 10 ≠ (n / 1000) % 10)

theorem consecutive_even_sum (a b c : ℕ) :
  is_consecutive_even a b c →
  has_valid_digits (a * b * c) →
  a + b + c = 84 :=
by sorry

end consecutive_even_sum_l1284_128441


namespace parabola_c_is_one_l1284_128480

/-- A parabola with equation y = 2x^2 + c and vertex at (0, 1) -/
structure Parabola where
  c : ℝ
  vertex_x : ℝ
  vertex_y : ℝ
  eq_vertex : vertex_y = 2 * vertex_x^2 + c
  is_vertex_zero_one : vertex_x = 0 ∧ vertex_y = 1

/-- The value of c for a parabola with equation y = 2x^2 + c and vertex at (0, 1) is 1 -/
theorem parabola_c_is_one (p : Parabola) : p.c = 1 := by
  sorry

end parabola_c_is_one_l1284_128480


namespace x_squared_minus_one_is_quadratic_l1284_128447

/-- Definition of a quadratic equation in one variable -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing x^2 - 1 -/
def f (x : ℝ) : ℝ := x^2 - 1

/-- Theorem: x^2 - 1 = 0 is a quadratic equation in one variable -/
theorem x_squared_minus_one_is_quadratic : is_quadratic_equation f := by
  sorry

end x_squared_minus_one_is_quadratic_l1284_128447


namespace master_percentage_is_76_l1284_128444

/-- Represents a team of junior and master players -/
structure Team where
  juniors : ℕ
  masters : ℕ

/-- The average score of the entire team -/
def teamAverage (t : Team) (juniorAvg masterAvg : ℚ) : ℚ :=
  (juniorAvg * t.juniors + masterAvg * t.masters) / (t.juniors + t.masters)

/-- The percentage of masters in the team -/
def masterPercentage (t : Team) : ℚ :=
  t.masters * 100 / (t.juniors + t.masters)

theorem master_percentage_is_76 (t : Team) :
  teamAverage t 22 47 = 41 →
  masterPercentage t = 76 := by
  sorry

#check master_percentage_is_76

end master_percentage_is_76_l1284_128444


namespace more_boys_than_girls_l1284_128408

/-- Given a school with 34 girls and 841 boys, prove that there are 807 more boys than girls. -/
theorem more_boys_than_girls (girls : ℕ) (boys : ℕ) 
  (h1 : girls = 34) (h2 : boys = 841) : boys - girls = 807 := by
  sorry

end more_boys_than_girls_l1284_128408


namespace point_light_source_theorem_l1284_128492

/-- Represents a person with their height and shadow length -/
structure Person where
  height : ℝ
  shadowLength : ℝ

/-- Represents different types of light sources -/
inductive LightSource
  | Point
  | Other

/-- Given two people under the same light source, 
    if the shorter person has a longer shadow, 
    then the light source must be a point light -/
theorem point_light_source_theorem 
  (personA personB : Person) 
  (light : LightSource) 
  (h1 : personA.height < personB.height) 
  (h2 : personA.shadowLength > personB.shadowLength) : 
  light = LightSource.Point := by
  sorry

end point_light_source_theorem_l1284_128492


namespace hexagon_regular_iff_equiangular_l1284_128470

/-- A hexagon is a polygon with 6 sides -/
structure Hexagon where
  sides : Fin 6 → ℝ
  angles : Fin 6 → ℝ

/-- A hexagon is equiangular if all its angles are equal -/
def is_equiangular (h : Hexagon) : Prop :=
  ∀ i j : Fin 6, h.angles i = h.angles j

/-- A hexagon is equilateral if all its sides are equal -/
def is_equilateral (h : Hexagon) : Prop :=
  ∀ i j : Fin 6, h.sides i = h.sides j

/-- A hexagon is regular if it is both equiangular and equilateral -/
def is_regular (h : Hexagon) : Prop :=
  is_equiangular h ∧ is_equilateral h

/-- Theorem: A hexagon is regular if and only if it is equiangular -/
theorem hexagon_regular_iff_equiangular (h : Hexagon) :
  is_regular h ↔ is_equiangular h :=
sorry

end hexagon_regular_iff_equiangular_l1284_128470


namespace right_triangle_ratio_l1284_128405

theorem right_triangle_ratio (a b c r s : ℝ) : 
  a > 0 → b > 0 → c > 0 → r > 0 → s > 0 →
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  c = r + s →        -- c is divided into r and s
  r * c = a^2 →      -- Geometric mean theorem for r
  s * c = b^2 →      -- Geometric mean theorem for s
  a / b = 2 / 5 →    -- Given ratio of a to b
  r / s = 4 / 25     -- Conclusion: ratio of r to s
  := by sorry

end right_triangle_ratio_l1284_128405


namespace systematic_sample_theorem_l1284_128478

/-- Represents a systematic sample from a population -/
structure SystematicSample where
  population_size : ℕ
  sample_size : ℕ
  start : ℕ
  interval : ℕ

/-- Checks if a number is in the systematic sample -/
def SystematicSample.contains (s : SystematicSample) (n : ℕ) : Prop :=
  ∃ k : ℕ, n = s.start + k * s.interval ∧ n ≤ s.population_size

theorem systematic_sample_theorem (s : SystematicSample)
    (h_pop : s.population_size = 48)
    (h_sample : s.sample_size = 4)
    (h_interval : s.interval = s.population_size / s.sample_size)
    (h5 : s.contains 5)
    (h29 : s.contains 29)
    (h41 : s.contains 41) :
    s.contains 17 := by
  sorry

#check systematic_sample_theorem

end systematic_sample_theorem_l1284_128478


namespace sum_of_squares_lower_bound_l1284_128499

theorem sum_of_squares_lower_bound (a b c : ℝ) (h : a + b + c = 1) :
  a^2 + b^2 + c^2 ≥ 1/3 := by
  sorry

end sum_of_squares_lower_bound_l1284_128499


namespace bisection_method_accuracy_l1284_128483

theorem bisection_method_accuracy (initial_interval_width : ℝ) (desired_accuracy : ℝ) : 
  initial_interval_width = 2 →
  desired_accuracy = 0.1 →
  ∃ n : ℕ, (n ≥ 5 ∧ initial_interval_width / (2^n : ℝ) < desired_accuracy) ∧
           ∀ m : ℕ, m < 5 → initial_interval_width / (2^m : ℝ) ≥ desired_accuracy :=
by sorry

end bisection_method_accuracy_l1284_128483


namespace product_of_sums_and_differences_l1284_128493

theorem product_of_sums_and_differences (P Q R S : ℝ) : 
  P = Real.sqrt 2012 + Real.sqrt 2013 →
  Q = -Real.sqrt 2012 - Real.sqrt 2013 →
  R = Real.sqrt 2012 - Real.sqrt 2013 →
  S = Real.sqrt 2013 - Real.sqrt 2012 →
  P * Q * R * S = 1 := by
  sorry

end product_of_sums_and_differences_l1284_128493


namespace negation_divisible_by_five_l1284_128412

theorem negation_divisible_by_five (n : ℕ) : 
  ¬(∀ n : ℕ, n % 5 = 0 → n % 10 = 0) ↔ 
  ∃ n : ℕ, n % 5 = 0 ∧ n % 10 ≠ 0 :=
by sorry

end negation_divisible_by_five_l1284_128412


namespace derivative_at_one_l1284_128420

-- Define the function f(x) = (x+1)^2(x-1)
def f (x : ℝ) : ℝ := (x + 1)^2 * (x - 1)

-- State the theorem
theorem derivative_at_one :
  deriv f 1 = 4 := by sorry

end derivative_at_one_l1284_128420


namespace fruit_cost_l1284_128484

/-- The cost of buying apples and oranges at given prices and quantities -/
theorem fruit_cost (apple_price : ℚ) (apple_weight : ℚ) (orange_price : ℚ) (orange_weight : ℚ)
  (apple_buy : ℚ) (orange_buy : ℚ) :
  apple_price = 3 →
  apple_weight = 4 →
  orange_price = 5 →
  orange_weight = 6 →
  apple_buy = 12 →
  orange_buy = 18 →
  (apple_price / apple_weight * apple_buy + orange_price / orange_weight * orange_buy : ℚ) = 24 :=
by sorry

end fruit_cost_l1284_128484


namespace optimal_rectangle_area_l1284_128431

/-- Given a rectangle with perimeter 400 feet, length at least 100 feet, and width at least 50 feet,
    the maximum possible area is 10,000 square feet. -/
theorem optimal_rectangle_area (l w : ℝ) (h1 : l + w = 200) (h2 : l ≥ 100) (h3 : w ≥ 50) :
  l * w ≤ 10000 :=
by sorry

end optimal_rectangle_area_l1284_128431


namespace largest_prime_factor_is_17_l1284_128430

def numbers : List Nat := [210, 255, 143, 187, 169]

def is_prime (n : Nat) : Prop := 
  n > 1 ∧ ∀ m : Nat, m > 1 → m < n → ¬(n % m = 0)

def prime_factors (n : Nat) : Set Nat :=
  {p : Nat | is_prime p ∧ n % p = 0}

theorem largest_prime_factor_is_17 : 
  ∃ (n : Nat), n ∈ numbers ∧ 
    (∃ (p : Nat), p ∈ prime_factors n ∧ p = 17 ∧ 
      ∀ (m : Nat) (q : Nat), m ∈ numbers → q ∈ prime_factors m → q ≤ 17) :=
by sorry

end largest_prime_factor_is_17_l1284_128430


namespace function_value_equality_l1284_128481

theorem function_value_equality (f : ℝ → ℝ) (m : ℝ) :
  (∀ x, f x = 2^x - 5) → f m = 3 → m = 3 := by
  sorry

end function_value_equality_l1284_128481


namespace pyramid_frustum_theorem_l1284_128402

-- Define the pyramid
structure Pyramid :=
  (base_side : ℝ)
  (height : ℝ)

-- Define the frustum
structure Frustum :=
  (base_side : ℝ)
  (top_side : ℝ)
  (height : ℝ)

-- Define the theorem
theorem pyramid_frustum_theorem (P : Pyramid) (F : Frustum) (P' : Pyramid) :
  P.base_side = 10 →
  P.height = 15 →
  F.base_side = P.base_side →
  F.top_side = P'.base_side →
  F.height + P'.height = P.height →
  (P.base_side^2 * P.height) = 9 * (P'.base_side^2 * P'.height) →
  ∃ (S : ℝ × ℝ × ℝ) (V : ℝ × ℝ × ℝ),
    S.2.2 = F.height / 2 + P'.height ∧
    V.2.2 = P.height ∧
    Real.sqrt ((S.1 - V.1)^2 + (S.2.1 - V.2.1)^2 + (S.2.2 - V.2.2)^2) = 2.5 :=
by sorry

end pyramid_frustum_theorem_l1284_128402


namespace i_13_times_1_plus_i_l1284_128488

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem i_13_times_1_plus_i : i^13 * (1 + i) = -1 + i := by sorry

end i_13_times_1_plus_i_l1284_128488


namespace remainder_theorem_l1284_128443

theorem remainder_theorem (x : ℤ) (h : x % 11 = 7) : (x^3 - (2*x)^2) % 11 = 4 := by
  sorry

end remainder_theorem_l1284_128443


namespace raccoon_nut_distribution_l1284_128462

/-- Represents the number of nuts taken by each raccoon -/
structure NutDistribution where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Checks if the given distribution satisfies all conditions -/
def isValidDistribution (d : NutDistribution) : Prop :=
  -- First raccoon's final nuts
  let first_final := d.first * 5 / 6 + d.second / 18 + d.third * 7 / 48
  -- Second raccoon's final nuts
  let second_final := d.first / 9 + d.second / 3 + d.third * 7 / 48
  -- Third raccoon's final nuts
  let third_final := d.first / 9 + d.second / 9 + d.third / 8
  -- All distributions result in whole numbers
  (d.first * 5 % 6 = 0) ∧ (d.second % 18 = 0) ∧ (d.third * 7 % 48 = 0) ∧
  (d.first % 9 = 0) ∧ (d.second % 3 = 0) ∧
  (d.first % 9 = 0) ∧ (d.second % 9 = 0) ∧ (d.third % 8 = 0) ∧
  -- Final ratio is 4:3:2
  (3 * first_final = 4 * second_final) ∧ (3 * first_final = 6 * third_final)

/-- The minimum total number of nuts -/
def minTotalNuts : ℕ := 864

theorem raccoon_nut_distribution :
  ∃ (d : NutDistribution), isValidDistribution d ∧
    d.first + d.second + d.third = minTotalNuts ∧
    (∀ (d' : NutDistribution), isValidDistribution d' →
      d'.first + d'.second + d'.third ≥ minTotalNuts) :=
  sorry


end raccoon_nut_distribution_l1284_128462


namespace equilateral_triangle_height_l1284_128422

/-- Given an equilateral triangle with two vertices at (0,0) and (10,0), 
    and the third vertex (x,y) in the first quadrant, 
    prove that the y-coordinate of the third vertex is 5√3. -/
theorem equilateral_triangle_height : 
  ∀ (x y : ℝ), 
  x ≥ 0 → y > 0 →  -- First quadrant condition
  (x^2 + y^2 = 100) →  -- Distance from (0,0) to (x,y) is 10
  ((x-10)^2 + y^2 = 100) →  -- Distance from (10,0) to (x,y) is 10
  y = 5 * Real.sqrt 3 := by
  sorry

end equilateral_triangle_height_l1284_128422


namespace system_solution_l1284_128474

theorem system_solution : 
  ∃! (x y : ℚ), (4 * x - 3 * y = -17) ∧ (5 * x + 6 * y = -4) ∧ 
  (x = -74/13) ∧ (y = -25/13) := by
sorry

end system_solution_l1284_128474


namespace max_value_of_f_l1284_128473

-- Define the function
def f (x : ℝ) : ℝ := -3 * x^2 + 6

-- State the theorem
theorem max_value_of_f :
  ∃ (M : ℝ), (∀ x, f x ≤ M) ∧ (∃ x₀, f x₀ = M) ∧ M = 6 := by
  sorry

end max_value_of_f_l1284_128473


namespace remainder_product_theorem_l1284_128436

theorem remainder_product_theorem (P Q R k : ℤ) (hk : k > 0) (hprod : P * Q = R) :
  (P % k * Q % k) % k = R % k :=
by sorry

end remainder_product_theorem_l1284_128436


namespace katy_brownies_l1284_128406

/-- The number of brownies Katy made and ate over three days. -/
def brownies_problem (monday : ℕ) : Prop :=
  ∃ (tuesday wednesday : ℕ),
    tuesday = 2 * monday ∧
    wednesday = 3 * tuesday ∧
    monday + tuesday + wednesday = 45

/-- Theorem stating that Katy made 45 brownies in total. -/
theorem katy_brownies : brownies_problem 5 := by
  sorry

end katy_brownies_l1284_128406


namespace flowers_in_vase_l1284_128400

/-- Given that Lara bought 52 stems of flowers, gave 15 to her mom, and gave 6 more to her grandma
    than to her mom, prove that she put 16 stems in the vase. -/
theorem flowers_in_vase (total : ℕ) (to_mom : ℕ) (extra_to_grandma : ℕ)
    (h1 : total = 52)
    (h2 : to_mom = 15)
    (h3 : extra_to_grandma = 6)
    : total - (to_mom + (to_mom + extra_to_grandma)) = 16 := by
  sorry

end flowers_in_vase_l1284_128400


namespace ferry_tourists_sum_l1284_128446

/-- The number of trips the ferry makes in a day -/
def num_trips : ℕ := 6

/-- The number of tourists on the first trip -/
def initial_tourists : ℕ := 100

/-- The decrease in number of tourists per trip -/
def tourist_decrease : ℕ := 1

/-- The sum of an arithmetic sequence -/
def arithmetic_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

/-- The total number of tourists transported by the ferry in a day -/
def total_tourists : ℕ := arithmetic_sum initial_tourists tourist_decrease num_trips

theorem ferry_tourists_sum :
  total_tourists = 585 := by sorry

end ferry_tourists_sum_l1284_128446


namespace tangent_circle_height_difference_l1284_128404

/-- A circle tangent to the parabola y = x^2 + 1 at two points and lying inside the parabola -/
structure TangentCircle where
  /-- x-coordinate of the point of tangency -/
  a : ℝ
  /-- y-coordinate of the center of the circle -/
  b : ℝ
  /-- radius of the circle -/
  r : ℝ
  /-- The circle is tangent to the parabola at (a, a^2 + 1) and (-a, a^2 + 1) -/
  tangent_point : b = a^2 + 1/2
  /-- The circle equation satisfies the tangency condition -/
  circle_eq : b^2 - r^2 = a^4 + 1

/-- The difference in height between the center of the circle and the points of tangency is -1/2 -/
theorem tangent_circle_height_difference (c : TangentCircle) :
  c.b - (c.a^2 + 1) = -1/2 := by
  sorry

end tangent_circle_height_difference_l1284_128404


namespace square_of_105_l1284_128461

theorem square_of_105 : (105 : ℕ)^2 = 11025 := by
  sorry

end square_of_105_l1284_128461


namespace greatest_perimeter_of_special_triangle_l1284_128459

theorem greatest_perimeter_of_special_triangle :
  ∀ (a b c : ℕ+),
  (a = 4 * b ∨ b = 4 * a ∨ a = 4 * c ∨ c = 4 * a ∨ b = 4 * c ∨ c = 4 * b) →
  (a = 18 ∨ b = 18 ∨ c = 18) →
  (a + b > c) →
  (a + c > b) →
  (b + c > a) →
  (a + b + c ≤ 43) :=
by sorry

end greatest_perimeter_of_special_triangle_l1284_128459


namespace largest_common_term_up_to_150_l1284_128477

theorem largest_common_term_up_to_150 :
  ∀ k ∈ Finset.range 151,
    (∃ n : ℕ, k = 2 + 8 * n) ∧
    (∃ m : ℕ, k = 3 + 9 * m) →
    k ≤ 138 :=
by sorry

end largest_common_term_up_to_150_l1284_128477


namespace dress_shirt_cost_l1284_128453

theorem dress_shirt_cost (num_shirts : ℕ) (tax_rate : ℝ) (total_paid : ℝ) :
  num_shirts = 3 ∧ tax_rate = 0.1 ∧ total_paid = 66 →
  ∃ (shirt_cost : ℝ), 
    shirt_cost * num_shirts * (1 + tax_rate) = total_paid ∧
    shirt_cost = 20 :=
by
  sorry

end dress_shirt_cost_l1284_128453


namespace correct_conclusions_l1284_128416

theorem correct_conclusions :
  (∀ x : ℝ, |x| = |-3| → x = 3 ∨ x = -3) ∧
  (∀ a b c : ℚ, a ≠ 0 → b ≠ 0 → c ≠ 0 →
    a < 0 → a + b < 0 → a + b + c < 0 →
    (|a| / a + |b| / b + |c| / c - |a * b * c| / (a * b * c) = 2 ∨
     |a| / a + |b| / b + |c| / c - |a * b * c| / (a * b * c) = -2)) :=
by sorry

end correct_conclusions_l1284_128416


namespace jakes_weight_l1284_128411

theorem jakes_weight (jake_weight sister_weight : ℝ) 
  (h1 : jake_weight - 8 = 2 * sister_weight)
  (h2 : jake_weight + sister_weight = 278) : 
  jake_weight = 188 := by
sorry

end jakes_weight_l1284_128411


namespace necessary_condition_when_m_is_one_necessary_condition_range_l1284_128409

/-- Proposition P -/
def P : Set ℝ := {x | -2 ≤ x ∧ x ≤ 10}

/-- Proposition q -/
def q (m : ℝ) : Set ℝ := {x | 1 - m ≤ x ∧ x ≤ 1 + m}

/-- P is a necessary but not sufficient condition for q -/
def necessary_not_sufficient (m : ℝ) : Prop :=
  (q m ⊆ P) ∧ (q m ≠ P) ∧ (m > 0)

theorem necessary_condition_when_m_is_one :
  necessary_not_sufficient 1 := by sorry

theorem necessary_condition_range :
  ∀ m : ℝ, necessary_not_sufficient m ↔ m ≥ 9 := by sorry

end necessary_condition_when_m_is_one_necessary_condition_range_l1284_128409


namespace quadratic_equation_solution_l1284_128432

theorem quadratic_equation_solution : 
  ∃ (x₁ x₂ : ℝ), x₁ = 5 ∧ x₂ = 1 ∧ 
  (x₁^2 - 6*x₁ + 5 = 0) ∧ (x₂^2 - 6*x₂ + 5 = 0) :=
by
  sorry

end quadratic_equation_solution_l1284_128432


namespace geometry_theorem_l1284_128438

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (intersects : Plane → Plane → Line → Prop)
variable (parallel : Line → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (perp_planes : Plane → Plane → Prop)
variable (perp_lines : Line → Line → Prop)

-- State the theorem
theorem geometry_theorem 
  (α β γ : Plane) (l m : Line)
  (h1 : intersects β γ l)
  (h2 : parallel l α)
  (h3 : contains α m)
  (h4 : perpendicular m γ) :
  perp_planes α γ ∧ perp_lines l m :=
sorry

end geometry_theorem_l1284_128438


namespace marsh_bird_difference_l1284_128427

theorem marsh_bird_difference (canadian_geese mallard_ducks great_egrets red_winged_blackbirds : ℕ) 
  (h1 : canadian_geese = 58)
  (h2 : mallard_ducks = 37)
  (h3 : great_egrets = 21)
  (h4 : red_winged_blackbirds = 15) :
  canadian_geese - mallard_ducks = 21 := by
  sorry

end marsh_bird_difference_l1284_128427


namespace man_speed_man_speed_result_l1284_128491

/-- Calculates the speed of a man given the parameters of a train passing him --/
theorem man_speed (train_length : ℝ) (train_speed_kmph : ℝ) (passing_time : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * 1000 / 3600
  let relative_speed := train_length / passing_time
  let man_speed_mps := relative_speed - train_speed_mps
  let man_speed_kmph := man_speed_mps * 3600 / 1000
  man_speed_kmph

/-- The speed of the man is approximately 5.976 kmph --/
theorem man_speed_result : 
  ∃ ε > 0, |man_speed 605 60 33 - 5.976| < ε :=
sorry

end man_speed_man_speed_result_l1284_128491


namespace right_triangle_acute_angles_l1284_128448

theorem right_triangle_acute_angles (a b : ℝ) : 
  a > 0 ∧ b > 0 ∧  -- Angles are positive
  a + b = 90 ∧     -- Sum of acute angles in a right triangle is 90°
  a / b = 3 / 2 →  -- Ratio of angles is 3:2
  a = 54 ∧ b = 36 := by sorry

end right_triangle_acute_angles_l1284_128448


namespace sum_of_three_different_digits_is_18_l1284_128465

/-- Represents a non-zero digit (1-9) -/
def NonZeroDigit := {n : ℕ // 1 ≤ n ∧ n ≤ 9}

/-- The sum of three different non-zero digits is 18 -/
theorem sum_of_three_different_digits_is_18 :
  ∃ (a b c : NonZeroDigit), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a.val + b.val + c.val = 18 := by
  sorry

end sum_of_three_different_digits_is_18_l1284_128465


namespace reciprocal_problem_l1284_128413

theorem reciprocal_problem (x : ℚ) (h : 8 * x = 3) : 50 * (1 / x) = 400 / 3 := by
  sorry

end reciprocal_problem_l1284_128413


namespace crab_meat_cost_per_pound_l1284_128467

/-- The cost of crab meat per pound given Johnny's crab dish production and expenses -/
theorem crab_meat_cost_per_pound 
  (dishes_per_day : ℕ) 
  (meat_per_dish : ℚ) 
  (weekly_expense : ℕ) 
  (closed_days : ℕ) : 
  dishes_per_day = 40 → 
  meat_per_dish = 3/2 → 
  weekly_expense = 1920 → 
  closed_days = 3 → 
  (weekly_expense : ℚ) / ((7 - closed_days) * dishes_per_day * meat_per_dish) = 8 := by
  sorry

end crab_meat_cost_per_pound_l1284_128467


namespace rectangle_diagonal_l1284_128456

/-- Given a rectangle with perimeter 60 meters and length-to-width ratio of 5:2,
    prove that its diagonal length is 162/7 meters. -/
theorem rectangle_diagonal (length width : ℝ) : 
  (2 * (length + width) = 60) →  -- Perimeter condition
  (length = (5/2) * width) →     -- Ratio condition
  Real.sqrt (length^2 + width^2) = 162/7 := by
sorry

end rectangle_diagonal_l1284_128456


namespace trigonometric_simplification_l1284_128495

theorem trigonometric_simplification (θ : ℝ) : 
  (1 + Real.sin θ + Real.cos θ) / (1 + Real.sin θ - Real.cos θ) + 
  (1 - Real.cos θ + Real.sin θ) / (1 + Real.cos θ + Real.sin θ) = 
  2 * (Real.sin θ)⁻¹ := by sorry

end trigonometric_simplification_l1284_128495


namespace animal_weight_comparison_l1284_128434

theorem animal_weight_comparison (chicken_weight duck_weight cow_weight : ℕ) 
  (h1 : chicken_weight = 3)
  (h2 : duck_weight = 6)
  (h3 : cow_weight = 624) :
  (cow_weight / chicken_weight = 208) ∧ (cow_weight / duck_weight = 104) := by
  sorry

end animal_weight_comparison_l1284_128434


namespace abc_divides_sum_pow13_l1284_128423

theorem abc_divides_sum_pow13 (a b c : ℕ+) 
  (h1 : a ∣ b^3) 
  (h2 : b ∣ c^3) 
  (h3 : c ∣ a^3) : 
  (a * b * c) ∣ (a + b + c)^13 := by
  sorry

end abc_divides_sum_pow13_l1284_128423


namespace wall_penetrating_skill_l1284_128449

theorem wall_penetrating_skill (n : ℕ) : 
  (8 * Real.sqrt (8 / n) = Real.sqrt (8 * (8 / n))) ↔ n = 63 := by
  sorry

end wall_penetrating_skill_l1284_128449


namespace poverty_alleviation_volunteers_l1284_128429

/-- Represents the age group frequencies in the histogram -/
structure AgeDistribution :=
  (f1 f2 f3 f4 f5 : ℝ)
  (sum_to_one : f1 + f2 + f3 + f4 + f5 = 1)

/-- Represents the stratified sample -/
structure StratifiedSample :=
  (total : ℕ)
  (under_35 : ℕ)
  (over_35 : ℕ)
  (sum_equal : under_35 + over_35 = total)

/-- The main theorem -/
theorem poverty_alleviation_volunteers 
  (dist : AgeDistribution) 
  (sample : StratifiedSample) 
  (h1 : dist.f1 = 0.01)
  (h2 : dist.f2 = 0.02)
  (h3 : dist.f3 = 0.04)
  (h5 : dist.f5 = 0.07)
  (h_sample : sample.total = 10 ∧ sample.under_35 = 6 ∧ sample.over_35 = 4) :
  dist.f4 = 0.06 ∧ 
  ∃ (X : Fin 4 → ℝ), 
    X 0 = 1/30 ∧ 
    X 1 = 3/10 ∧ 
    X 2 = 1/2 ∧ 
    X 3 = 1/6 ∧
    (X 0 * 0 + X 1 * 1 + X 2 * 2 + X 3 * 3 = 1.8) := by
  sorry

end poverty_alleviation_volunteers_l1284_128429


namespace line_ellipse_intersection_slopes_l1284_128496

/-- Given a line y = mx + 5 intersecting the ellipse 9x^2 + 16y^2 = 144,
    prove that the possible slopes m satisfy m ∈ (-∞,-1] ∪ [1,∞). -/
theorem line_ellipse_intersection_slopes (m : ℝ) : 
  (∃ x y : ℝ, 9 * x^2 + 16 * y^2 = 144 ∧ y = m * x + 5) ↔ m ≤ -1 ∨ m ≥ 1 := by
  sorry

#check line_ellipse_intersection_slopes

end line_ellipse_intersection_slopes_l1284_128496


namespace max_color_transitions_l1284_128463

/-- Represents a strategy for painting fence sections -/
def PaintingStrategy := Nat → Bool

/-- The number of fence sections -/
def numSections : Nat := 100

/-- Counts the number of color transitions in a given painting strategy -/
def countTransitions (strategy : PaintingStrategy) : Nat :=
  (List.range (numSections - 1)).filter (fun i => strategy i ≠ strategy (i + 1)) |>.length

/-- Theorem stating that the maximum number of guaranteed color transitions is 49 -/
theorem max_color_transitions :
  ∃ (strategy : PaintingStrategy),
    ∀ (otherStrategy : PaintingStrategy),
      countTransitions (fun i => if i % 2 = 0 then strategy (i / 2) else otherStrategy (i / 2)) ≥ 49 := by
  sorry

end max_color_transitions_l1284_128463


namespace power_relation_l1284_128421

theorem power_relation (a m n : ℝ) (hm : a^m = 2) (hn : a^n = 5) : 
  a^(3*m - 2*n) = 8/25 := by
sorry

end power_relation_l1284_128421


namespace trig_expression_equality_l1284_128433

theorem trig_expression_equality : 
  let sin30 : ℝ := 1/2
  let cos45 : ℝ := Real.sqrt 2 / 2
  let tan60 : ℝ := Real.sqrt 3
  sin30 - Real.sqrt 3 * cos45 + Real.sqrt 2 * tan60 = (1 + Real.sqrt 6) / 2 := by
sorry

end trig_expression_equality_l1284_128433


namespace vacation_cost_is_120_l1284_128426

/-- Calculates the total cost of a vacation for two people. -/
def vacationCost (planeTicketCost hotelCostPerDay : ℕ) (durationInDays : ℕ) : ℕ :=
  2 * planeTicketCost + 2 * hotelCostPerDay * durationInDays

/-- Proves that the total cost of the vacation is $120. -/
theorem vacation_cost_is_120 :
  vacationCost 24 12 3 = 120 := by
  sorry

end vacation_cost_is_120_l1284_128426


namespace smallest_even_integer_abs_inequality_l1284_128479

theorem smallest_even_integer_abs_inequality :
  ∃ (x : ℤ), 
    (∀ (y : ℤ), (y % 2 = 0 ∧ |3*y - 4| ≤ 20) → x ≤ y) ∧
    (x % 2 = 0) ∧
    (|3*x - 4| ≤ 20) ∧
    x = -4 :=
by sorry

end smallest_even_integer_abs_inequality_l1284_128479


namespace bobs_age_multiple_l1284_128445

theorem bobs_age_multiple (bob_age carol_age : ℕ) (m : ℚ) : 
  bob_age = 16 →
  carol_age = 50 →
  carol_age = m * bob_age + 2 →
  m = 3 := by
sorry

end bobs_age_multiple_l1284_128445


namespace worm_length_difference_l1284_128475

def worm_lengths : List ℝ := [0.8, 0.1, 1.2, 0.4, 0.7]

theorem worm_length_difference : 
  let max_length := worm_lengths.maximum?
  let min_length := worm_lengths.minimum?
  ∀ max min, max_length = some max → min_length = some min →
    max - min = 1.1 := by sorry

end worm_length_difference_l1284_128475


namespace abc_inequality_l1284_128414

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^a * b^b * c^c ≥ (a*b*c)^((a+b+c)/3) := by
  sorry

end abc_inequality_l1284_128414


namespace coin_denomination_problem_l1284_128410

/-- Given a total of 334 coins, with 250 coins of 20 paise each, and a total sum of 7100 paise,
    the denomination of the remaining coins is 25 paise. -/
theorem coin_denomination_problem (total_coins : ℕ) (twenty_paise_coins : ℕ) (total_sum : ℕ) :
  total_coins = 334 →
  twenty_paise_coins = 250 →
  total_sum = 7100 →
  (total_coins - twenty_paise_coins) * (total_sum - twenty_paise_coins * 20) / (total_coins - twenty_paise_coins) = 25 := by
  sorry

#eval (334 - 250) * (7100 - 250 * 20) / (334 - 250)  -- Should output 25

end coin_denomination_problem_l1284_128410


namespace monotonicity_and_range_l1284_128482

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x + a * Real.log x

theorem monotonicity_and_range :
  ∀ (a : ℝ), a ≤ 0 →
  (∀ (x : ℝ), x > 0 → f (-2) x < f (-2) (1 + Real.sqrt 2) → x < 1 + Real.sqrt 2) ∧
  (∀ (x : ℝ), x > 0 → f (-2) x > f (-2) (1 + Real.sqrt 2) → x > 1 + Real.sqrt 2) ∧
  (∀ (x : ℝ), x > 0 → f a x > (1/2)*(2*Real.exp 1 + 1)*a ↔ a ∈ Set.Ioo (-2*(Real.exp 1)^2/(2*Real.exp 1 + 1)) 0) :=
by sorry

end monotonicity_and_range_l1284_128482


namespace sqrt_twelve_minus_sqrt_three_l1284_128498

theorem sqrt_twelve_minus_sqrt_three : Real.sqrt 12 - Real.sqrt 3 = Real.sqrt 3 := by
  sorry

end sqrt_twelve_minus_sqrt_three_l1284_128498


namespace interval_for_720_recordings_l1284_128460

/-- Calculates the time interval between recordings given the number of recordings in an hour -/
def timeInterval (recordings : ℕ) : ℚ :=
  3600 / recordings

/-- Theorem stating that 720 recordings in an hour results in a 5-second interval -/
theorem interval_for_720_recordings :
  timeInterval 720 = 5 := by
  sorry

end interval_for_720_recordings_l1284_128460


namespace range_of_f_greater_than_x_l1284_128454

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then (1/2) * x - 1 else 1/x

-- State the theorem
theorem range_of_f_greater_than_x :
  ∀ a : ℝ, f a > a ↔ a ∈ Set.Iio (-1) :=
by sorry

end range_of_f_greater_than_x_l1284_128454


namespace smallest_norwegian_l1284_128458

def is_norwegian (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), a < b ∧ b < c ∧ a ∣ n ∧ b ∣ n ∧ c ∣ n ∧ a + b + c = 2022

theorem smallest_norwegian : ∀ n : ℕ, is_norwegian n → n ≥ 1344 :=
sorry

end smallest_norwegian_l1284_128458


namespace chubby_checkerboard_black_squares_l1284_128437

/-- Represents a square on the checkerboard -/
inductive Square
| Black
| Red

/-- Represents a checkerboard -/
def Checkerboard := Array (Array Square)

/-- Creates a checkerboard with the given dimensions and pattern -/
def createCheckerboard (n : Nat) : Checkerboard :=
  sorry

/-- Counts the number of black squares on the checkerboard -/
def countBlackSquares (board : Checkerboard) : Nat :=
  sorry

theorem chubby_checkerboard_black_squares :
  let board := createCheckerboard 29
  countBlackSquares board = 421 := by
  sorry

end chubby_checkerboard_black_squares_l1284_128437


namespace ones_digit_of_power_l1284_128485

-- Define a function to get the ones digit of a natural number
def onesDigit (n : ℕ) : ℕ := n % 10

-- Define the exponent
def exponent : ℕ := 22 * (11^11)

-- Theorem statement
theorem ones_digit_of_power : onesDigit (22^exponent) = 4 := by
  sorry

end ones_digit_of_power_l1284_128485


namespace capital_growth_l1284_128497

def capital_sequence : ℕ → ℝ
  | 0 => 60
  | n + 1 => 1.5 * capital_sequence n - 15

theorem capital_growth (n : ℕ) :
  -- a₁ = 60
  capital_sequence 0 = 60 ∧
  -- {aₙ - 3} forms a geometric sequence
  (∀ k : ℕ, capital_sequence (k + 1) - 3 = 1.5 * (capital_sequence k - 3)) ∧
  -- By the end of 2026 (6 years from 2021), the remaining capital will exceed 210 million yuan
  ∃ m : ℕ, m ≤ 6 ∧ capital_sequence m > 210 :=
by sorry

end capital_growth_l1284_128497


namespace min_sum_reciprocals_equality_condition_l1284_128417

theorem min_sum_reciprocals (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / (3 * b) + b / (5 * c) + c / (7 * a) ≥ 3 / Real.rpow 105 (1/3) :=
sorry

theorem equality_condition (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / (3 * b) = b / (5 * c) ∧ b / (5 * c) = c / (7 * a)) ↔
  (a / (3 * b) = 1 / Real.rpow 105 (1/3) ∧
   b / (5 * c) = 1 / Real.rpow 105 (1/3) ∧
   c / (7 * a) = 1 / Real.rpow 105 (1/3)) :=
sorry

end min_sum_reciprocals_equality_condition_l1284_128417


namespace last_digit_89_base5_l1284_128457

def decimal_to_base5 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 5) ((m % 5) :: acc)
    aux n []

theorem last_digit_89_base5 : 
  (decimal_to_base5 89).getLast? = some 4 := by
  sorry

end last_digit_89_base5_l1284_128457


namespace egg_difference_l1284_128476

/-- The number of eggs needed for one chocolate cake -/
def chocolate_cake_eggs : ℕ := 3

/-- The number of eggs needed for one cheesecake -/
def cheesecake_eggs : ℕ := 8

/-- The number of chocolate cakes -/
def num_chocolate_cakes : ℕ := 5

/-- The number of cheesecakes -/
def num_cheesecakes : ℕ := 9

/-- Theorem: The difference in eggs needed for 9 cheesecakes and 5 chocolate cakes is 57 -/
theorem egg_difference : 
  num_cheesecakes * cheesecake_eggs - num_chocolate_cakes * chocolate_cake_eggs = 57 := by
  sorry

end egg_difference_l1284_128476


namespace angle_4_value_l1284_128494

theorem angle_4_value (angle_1 angle_2 angle_3 angle_4 angle_A angle_B : ℝ) :
  angle_1 + angle_2 = 180 →
  angle_3 = angle_4 →
  angle_3 = (1 / 2) * angle_4 →
  angle_A = 80 →
  angle_B = 50 →
  angle_4 = 100 / 3 := by
  sorry

end angle_4_value_l1284_128494


namespace walking_time_proportional_l1284_128455

/-- Given a constant walking rate, prove that if it takes 6 minutes to walk 2 miles, 
    then it will take 12 minutes to walk 4 miles. -/
theorem walking_time_proportional (rate : ℝ) : 
  (rate * 2 = 6) → (rate * 4 = 12) := by
  sorry

end walking_time_proportional_l1284_128455


namespace only_negative_sqrt_two_less_than_zero_l1284_128471

theorem only_negative_sqrt_two_less_than_zero :
  let numbers : List ℝ := [5, 2, 0, -Real.sqrt 2]
  (∀ x ∈ numbers, x < 0) ↔ (x = -Real.sqrt 2) :=
by sorry

end only_negative_sqrt_two_less_than_zero_l1284_128471


namespace f_geq_kx_implies_k_range_l1284_128442

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 2 * x^2 - 3 * x else Real.exp x + Real.exp 2

-- State the theorem
theorem f_geq_kx_implies_k_range :
  (∀ x : ℝ, f x ≥ k * x) → -3 ≤ k ∧ k ≤ Real.exp 2 :=
by sorry

end f_geq_kx_implies_k_range_l1284_128442


namespace intersection_points_form_rectangle_l1284_128487

/-- The set of points satisfying xy = 18 and x^2 + y^2 = 45 -/
def IntersectionPoints : Set (ℝ × ℝ) :=
  {p | p.1 * p.2 = 18 ∧ p.1^2 + p.2^2 = 45}

/-- A function to check if four points form a rectangle -/
def IsRectangle (p1 p2 p3 p4 : ℝ × ℝ) : Prop :=
  let d12 := (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2
  let d23 := (p2.1 - p3.1)^2 + (p2.2 - p3.2)^2
  let d34 := (p3.1 - p4.1)^2 + (p3.2 - p4.2)^2
  let d41 := (p4.1 - p1.1)^2 + (p4.2 - p1.2)^2
  let d13 := (p1.1 - p3.1)^2 + (p1.2 - p3.2)^2
  let d24 := (p2.1 - p4.1)^2 + (p2.2 - p4.2)^2
  (d12 = d34 ∧ d23 = d41) ∧ (d13 = d24)

theorem intersection_points_form_rectangle :
  ∃ p1 p2 p3 p4 : ℝ × ℝ, p1 ∈ IntersectionPoints ∧ p2 ∈ IntersectionPoints ∧
    p3 ∈ IntersectionPoints ∧ p4 ∈ IntersectionPoints ∧
    IsRectangle p1 p2 p3 p4 :=
  sorry

end intersection_points_form_rectangle_l1284_128487


namespace milk_replacement_problem_l1284_128401

/-- Given a container initially full of milk, prove that if x liters are drawn out and 
    replaced with water twice, resulting in a milk to water ratio of 9:16 in a 
    total mixture of 15 liters, then x must equal 12 liters. -/
theorem milk_replacement_problem (x : ℝ) : 
  x > 0 →
  (15 - x) - x * ((15 - x) / 15) = (9 / 25) * 15 →
  x = 12 := by
  sorry

end milk_replacement_problem_l1284_128401


namespace unique_triple_l1284_128450

theorem unique_triple : ∃! (x y z : ℕ), 
  x > 1 ∧ y > 1 ∧ z > 1 ∧
  (yz - 1) % x = 0 ∧ 
  (zx - 1) % y = 0 ∧ 
  (xy - 1) % z = 0 ∧
  x = 5 ∧ y = 3 ∧ z = 2 := by
  sorry

end unique_triple_l1284_128450


namespace work_time_ratio_l1284_128415

theorem work_time_ratio (time_A : ℝ) (combined_rate : ℝ) : 
  time_A = 10 → combined_rate = 0.3 → 
  ∃ time_B : ℝ, time_B / time_A = 1 / 2 :=
by
  sorry

end work_time_ratio_l1284_128415


namespace calculate_expression_l1284_128435

theorem calculate_expression : (2023 - Real.pi) ^ 0 - (1 / 4)⁻¹ + |(-2)| + Real.sqrt 9 = 2 := by
  sorry

end calculate_expression_l1284_128435


namespace sin_cos_shift_l1284_128452

theorem sin_cos_shift (x : ℝ) : 
  Real.sin (2 * x + π / 6) = Real.cos (2 * x - π / 6 + π / 2 - π / 12) := by
  sorry

end sin_cos_shift_l1284_128452


namespace rectangular_box_surface_area_l1284_128418

theorem rectangular_box_surface_area 
  (a b c : ℝ) 
  (h1 : 4 * a + 4 * b + 4 * c = 160) 
  (h2 : Real.sqrt (a^2 + b^2 + c^2) = 25) : 
  2 * (a * b + b * c + c * a) = 975 := by
  sorry

end rectangular_box_surface_area_l1284_128418


namespace xyz_divides_product_l1284_128469

/-- A proposition stating that if x, y, and z are distinct positive integers
    such that xyz divides (xy-1)(yz-1)(zx-1), then (x, y, z) is a permutation of (2, 3, 5) -/
theorem xyz_divides_product (x y z : ℕ) : 
  x > 0 → y > 0 → z > 0 → 
  x ≠ y → y ≠ z → x ≠ z →
  (x * y * z) ∣ ((x * y - 1) * (y * z - 1) * (z * x - 1)) →
  (x = 2 ∧ y = 3 ∧ z = 5) ∨ 
  (x = 2 ∧ y = 5 ∧ z = 3) ∨ 
  (x = 3 ∧ y = 2 ∧ z = 5) ∨ 
  (x = 3 ∧ y = 5 ∧ z = 2) ∨ 
  (x = 5 ∧ y = 2 ∧ z = 3) ∨ 
  (x = 5 ∧ y = 3 ∧ z = 2) := by
  sorry

#check xyz_divides_product

end xyz_divides_product_l1284_128469


namespace parity_of_expression_l1284_128407

theorem parity_of_expression (p m : ℤ) (h_p_odd : Odd p) :
  Odd (p^2 + 3*m*p) ↔ Even m := by
sorry

end parity_of_expression_l1284_128407


namespace zeros_of_f_l1284_128451

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2 - 2*x - 3 else -2 + Real.log x

-- State the theorem about the zeros of f
theorem zeros_of_f :
  ∃ (x₁ x₂ : ℝ), x₁ = -1 ∧ x₂ = Real.exp 2 ∧
  (∀ x : ℝ, f x = 0 ↔ x = x₁ ∨ x = x₂) :=
sorry

end zeros_of_f_l1284_128451


namespace project_profit_analysis_l1284_128489

/-- Represents the net profit of a project in millions of yuan -/
def net_profit (n : ℕ+) : ℚ :=
  100 * n - (4 * n^2 + 40 * n) - 144

/-- Represents the average annual profit of a project in millions of yuan -/
def avg_annual_profit (n : ℕ+) : ℚ :=
  net_profit n / n

theorem project_profit_analysis :
  ∀ n : ℕ+,
  (net_profit n = -4 * (n - 3) * (n - 12)) ∧
  (net_profit n > 0 ↔ 3 < n ∧ n < 12) ∧
  (∀ m : ℕ+, avg_annual_profit m ≤ avg_annual_profit 6) := by
  sorry

#check project_profit_analysis

end project_profit_analysis_l1284_128489


namespace parallelogram_height_l1284_128490

theorem parallelogram_height (base area : ℝ) (h_base : base = 28) (h_area : area = 896) :
  area / base = 32 := by
sorry

end parallelogram_height_l1284_128490


namespace absolute_value_inequality_l1284_128472

theorem absolute_value_inequality (x : ℝ) : 
  |((7 - x) / 5)| < 3 ↔ -8 < x ∧ x < 22 := by sorry

end absolute_value_inequality_l1284_128472


namespace N_is_composite_l1284_128486

theorem N_is_composite : ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ (2011 * 2012 * 2013 * 2014 + 1 = a * b) := by
  sorry

end N_is_composite_l1284_128486


namespace diophantine_equation_solutions_l1284_128403

theorem diophantine_equation_solutions :
  ∀ (x y : ℕ) (p : ℕ), 
    Prime p → 
    p^x - y^p = 1 → 
    ((x = 1 ∧ y = 1 ∧ p = 2) ∨ (x = 2 ∧ y = 2 ∧ p = 3)) :=
by sorry

end diophantine_equation_solutions_l1284_128403


namespace unique_5digit_number_l1284_128468

/-- A function that generates all 3-digit numbers from a list of 5 digits -/
def generate_3digit_numbers (digits : List Nat) : List Nat :=
  sorry

/-- The sum of all 3-digit numbers generated from the digits of a 5-digit number -/
def sum_3digit_numbers (n : Nat) : Nat :=
  sorry

/-- Checks if a number has 5 different non-zero digits -/
def has_5_different_nonzero_digits (n : Nat) : Prop :=
  sorry

theorem unique_5digit_number : 
  ∃! n : Nat, 
    10000 ≤ n ∧ n < 100000 ∧
    has_5_different_nonzero_digits n ∧
    n = sum_3digit_numbers n ∧
    n = 35964 :=
  sorry

end unique_5digit_number_l1284_128468


namespace student_number_problem_l1284_128425

theorem student_number_problem (x : ℝ) : 3 * x - 220 = 110 → x = 110 := by
  sorry

end student_number_problem_l1284_128425


namespace min_value_reciprocal_sum_l1284_128439

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 3 * y = 1) :
  (∀ x' y' : ℝ, x' > 0 → y' > 0 → x' + 3 * y' = 1 → 1 / x' + 1 / y' ≥ 1 / x + 1 / y) →
  1 / x + 1 / y = 3 + Real.sqrt 3 :=
by sorry

end min_value_reciprocal_sum_l1284_128439


namespace probability_at_least_one_three_l1284_128464

/-- The number of sides on each die -/
def num_sides : ℕ := 8

/-- The probability of at least one die showing a 3 when two fair dice are rolled -/
def prob_at_least_one_three : ℚ := 15 / 64

/-- Theorem stating that the probability of at least one die showing a 3
    when two fair 8-sided dice are rolled is 15/64 -/
theorem probability_at_least_one_three :
  prob_at_least_one_three = 15 / 64 := by
  sorry

end probability_at_least_one_three_l1284_128464


namespace geometric_series_common_ratio_l1284_128440

theorem geometric_series_common_ratio 
  (a : ℕ → ℝ) 
  (h_geometric : ∀ n : ℕ, a (n + 1) = a n * (a 2 / a 1)) 
  (h_sum1 : a 1 + a 3 = 10) 
  (h_sum2 : a 4 + a 6 = 5/4) : 
  a 2 / a 1 = 1/2 := by
sorry

end geometric_series_common_ratio_l1284_128440

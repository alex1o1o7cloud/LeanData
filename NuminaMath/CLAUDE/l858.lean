import Mathlib

namespace NUMINAMATH_CALUDE_max_students_for_equal_distribution_l858_85881

theorem max_students_for_equal_distribution (pens pencils : ℕ) 
  (h1 : pens = 1001) (h2 : pencils = 910) : 
  Nat.gcd pens pencils = 91 := by
sorry

end NUMINAMATH_CALUDE_max_students_for_equal_distribution_l858_85881


namespace NUMINAMATH_CALUDE_smallest_positive_b_existence_l858_85897

theorem smallest_positive_b_existence :
  ∃ (b y : ℝ), b > 0 ∧ y > 0 ∧
  ((9 * Real.sqrt ((3*b)^2 + 2^2) + 5*b^2 - 2) / (Real.sqrt (2 + 5*b^2) - 5) = -4) ∧
  (y^4 + 105*y^2 + 562 = 0) ∧
  (y^2 > 2) ∧
  (b = Real.sqrt (y^2 - 2) / Real.sqrt 5) ∧
  (∀ (b' : ℝ), b' > 0 → 
    ((9 * Real.sqrt ((3*b')^2 + 2^2) + 5*b'^2 - 2) / (Real.sqrt (2 + 5*b'^2) - 5) = -4) →
    b ≤ b') :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_b_existence_l858_85897


namespace NUMINAMATH_CALUDE_M_equals_N_l858_85899

def M : Set ℤ := {-1, 0, 1}

def N : Set ℤ := {x | ∃ a b, a ∈ M ∧ b ∈ M ∧ x = a * b}

theorem M_equals_N : M = N := by
  sorry

end NUMINAMATH_CALUDE_M_equals_N_l858_85899


namespace NUMINAMATH_CALUDE_ladonnas_cans_correct_l858_85811

/-- The number of cans collected by LaDonna, given that:
    - The total number of cans collected is 85
    - Prikya collected twice as many cans as LaDonna
    - Yoki collected 10 cans
-/
def ladonnas_cans : ℕ := 25

/-- The total number of cans collected -/
def total_cans : ℕ := 85

/-- The number of cans collected by Yoki -/
def yokis_cans : ℕ := 10

theorem ladonnas_cans_correct :
  ladonnas_cans + 2 * ladonnas_cans + yokis_cans = total_cans :=
by sorry

end NUMINAMATH_CALUDE_ladonnas_cans_correct_l858_85811


namespace NUMINAMATH_CALUDE_min_max_sum_reciprocals_l858_85882

open Real

theorem min_max_sum_reciprocals (x y z : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0) (h_sum : x + y + z = 3) :
  let f := (1 / (x + y) + 1 / (x + z) + 1 / (y + z))
  ∃ (min_val : ℝ), (∀ a b c, a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 3 →
    (1 / (a + b) + 1 / (a + c) + 1 / (b + c)) ≥ min_val) ∧
  min_val = (3 / 2) ∧
  ¬∃ (max_val : ℝ), ∀ a b c, a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 3 →
    (1 / (a + b) + 1 / (a + c) + 1 / (b + c)) ≤ max_val :=
by sorry

end NUMINAMATH_CALUDE_min_max_sum_reciprocals_l858_85882


namespace NUMINAMATH_CALUDE_same_color_shoe_probability_l858_85850

/-- The number of pairs of shoes -/
def num_pairs : ℕ := 7

/-- The total number of shoes -/
def total_shoes : ℕ := 2 * num_pairs

/-- The number of shoes to be selected -/
def selected_shoes : ℕ := 2

/-- The probability of selecting two shoes of the same color -/
def same_color_prob : ℚ := 1 / 13

theorem same_color_shoe_probability :
  (num_pairs : ℚ) / (total_shoes.choose selected_shoes) = same_color_prob :=
sorry

end NUMINAMATH_CALUDE_same_color_shoe_probability_l858_85850


namespace NUMINAMATH_CALUDE_smallest_triangle_side_l858_85829

def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def smallest_t : ℕ → Prop
| t => is_triangle 7.5 11 (t : ℝ) ∧ ∀ s : ℕ, s < t → ¬is_triangle 7.5 11 (s : ℝ)

theorem smallest_triangle_side : smallest_t 4 := by
  sorry

end NUMINAMATH_CALUDE_smallest_triangle_side_l858_85829


namespace NUMINAMATH_CALUDE_largest_number_in_sample_l858_85885

/-- Represents a systematic sampling process -/
structure SystematicSample where
  population_size : ℕ
  start : ℕ
  interval : ℕ

/-- Calculates the largest number in a systematic sample -/
def largest_sample_number (s : SystematicSample) : ℕ :=
  s.start + s.interval * ((s.population_size - s.start) / s.interval)

/-- Theorem: The largest number in the given systematic sample is 1468 -/
theorem largest_number_in_sample :
  let s : SystematicSample := ⟨1500, 18, 50⟩
  largest_sample_number s = 1468 := by
  sorry

end NUMINAMATH_CALUDE_largest_number_in_sample_l858_85885


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l858_85824

theorem quadratic_inequality_solution (k : ℝ) : 
  (∀ x : ℝ, x^2 - (k - 4)*x - k + 8 > 0) ↔ k ∈ Set.Ioo (-8/3) 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l858_85824


namespace NUMINAMATH_CALUDE_unique_sequence_exists_l858_85842

def sequence_property (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ a 2 = 2 ∧ ∀ n : ℕ, n ≥ 1 → a n * a (n + 2) = (a (n + 1))^3 + 1

theorem unique_sequence_exists : ∃! a : ℕ → ℕ, sequence_property a := by
  sorry

end NUMINAMATH_CALUDE_unique_sequence_exists_l858_85842


namespace NUMINAMATH_CALUDE_corner_removed_cube_vertex_count_l858_85833

/-- Represents a cube with a given side length. -/
structure Cube where
  sideLength : ℝ
  sideLength_pos : sideLength > 0

/-- Represents the resulting solid after removing smaller cubes from the corners of a larger cube. -/
structure CornerRemovedCube where
  originalCube : Cube
  removedCubeSideLength : ℝ
  removedCubeSideLength_pos : removedCubeSideLength > 0
  removedCubeSideLength_lt : removedCubeSideLength < originalCube.sideLength

/-- Calculates the number of vertices in the resulting solid after removing smaller cubes from the corners of a larger cube. -/
def vertexCount (c : CornerRemovedCube) : ℕ :=
  8 * 5  -- Each corner of the original cube contributes 5 vertices

/-- Theorem stating that removing cubes of side length 2 from each corner of a cube with side length 5 results in a solid with 40 vertices. -/
theorem corner_removed_cube_vertex_count :
  ∀ (c : CornerRemovedCube),
  c.originalCube.sideLength = 5 →
  c.removedCubeSideLength = 2 →
  vertexCount c = 40 :=
by
  sorry


end NUMINAMATH_CALUDE_corner_removed_cube_vertex_count_l858_85833


namespace NUMINAMATH_CALUDE_company_merger_profit_distribution_l858_85819

theorem company_merger_profit_distribution (company_a_profit company_b_profit : ℝ) 
  (company_a_percentage : ℝ) :
  company_a_profit = 90000 ∧ 
  company_b_profit = 60000 ∧ 
  company_a_percentage = 60 →
  (company_b_profit / (company_a_profit + company_b_profit)) * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_company_merger_profit_distribution_l858_85819


namespace NUMINAMATH_CALUDE_arithmetic_sequence_30th_term_l858_85895

/-- A sequence {a_n} with sum S_n satisfying the given conditions -/
def ArithmeticSequence (a : ℕ → ℚ) (S : ℕ → ℚ) : Prop :=
  (∀ n : ℕ, 3 * S n / n + n = 3 * a n + 1) ∧ (a 1 = -1/3)

/-- Theorem stating that the 30th term of the sequence is 19 -/
theorem arithmetic_sequence_30th_term
  (a : ℕ → ℚ) (S : ℕ → ℚ) (h : ArithmeticSequence a S) :
  a 30 = 19 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_30th_term_l858_85895


namespace NUMINAMATH_CALUDE_cuboid_breadth_calculation_l858_85815

/-- The surface area of a cuboid given its length, breadth, and height. -/
def cuboidSurfaceArea (l b h : ℝ) : ℝ := 2 * (l * b + b * h + h * l)

/-- Theorem stating that a cuboid with given dimensions has the specified breadth. -/
theorem cuboid_breadth_calculation (l h : ℝ) (surface_area : ℝ) 
  (hl : l = 10) (hh : h = 6) (hsa : surface_area = 480) :
  ∃ b : ℝ, cuboidSurfaceArea l b h = surface_area ∧ b = 11.25 := by
  sorry

#check cuboid_breadth_calculation

end NUMINAMATH_CALUDE_cuboid_breadth_calculation_l858_85815


namespace NUMINAMATH_CALUDE_kyle_stars_theorem_l858_85893

/-- The number of stars needed to fill all bottles Kyle bought -/
def total_stars (initial_bottles : ℕ) (additional_bottles : ℕ) (stars_per_bottle : ℕ) : ℕ :=
  (initial_bottles + additional_bottles) * stars_per_bottle

/-- Theorem stating the total number of stars Kyle needs to make -/
theorem kyle_stars_theorem :
  total_stars 2 3 15 = 75 := by
  sorry

end NUMINAMATH_CALUDE_kyle_stars_theorem_l858_85893


namespace NUMINAMATH_CALUDE_jerrys_action_figures_l858_85888

theorem jerrys_action_figures (initial_figures : ℕ) : 
  (10 : ℕ) = initial_figures + 4 + 4 → initial_figures = 2 := by
sorry

end NUMINAMATH_CALUDE_jerrys_action_figures_l858_85888


namespace NUMINAMATH_CALUDE_fixed_points_are_corresponding_l858_85845

/-- A type representing a geometric figure -/
structure Figure where
  -- Add necessary fields here
  mk :: -- Constructor

/-- A type representing a point in a geometric figure -/
structure Point where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Predicate to check if three figures are similar -/
def are_similar (f1 f2 f3 : Figure) : Prop :=
  sorry

/-- Predicate to check if a point is fixed (invariant) in a figure -/
def is_fixed_point (p : Point) (f : Figure) : Prop :=
  sorry

/-- Predicate to check if two points are corresponding in two figures -/
def are_corresponding_points (p1 p2 : Point) (f1 f2 : Figure) : Prop :=
  sorry

/-- Theorem stating that fixed points of three similar figures are corresponding points -/
theorem fixed_points_are_corresponding
  (f1 f2 f3 : Figure)
  (h_similar : are_similar f1 f2 f3)
  (p1 : Point)
  (h_fixed1 : is_fixed_point p1 f1)
  (p2 : Point)
  (h_fixed2 : is_fixed_point p2 f2)
  (p3 : Point)
  (h_fixed3 : is_fixed_point p3 f3) :
  are_corresponding_points p1 p2 f1 f2 ∧
  are_corresponding_points p2 p3 f2 f3 ∧
  are_corresponding_points p1 p3 f1 f3 :=
by
  sorry

end NUMINAMATH_CALUDE_fixed_points_are_corresponding_l858_85845


namespace NUMINAMATH_CALUDE_diagonals_not_parallel_in_32gon_l858_85812

/-- The number of sides in the regular polygon -/
def n : ℕ := 32

/-- The total number of diagonals in an n-sided polygon -/
def total_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- The number of pairs of parallel sides in an n-sided polygon -/
def parallel_side_pairs (n : ℕ) : ℕ := n / 2

/-- The number of diagonals parallel to one pair of sides -/
def diagonals_per_parallel_pair (n : ℕ) : ℕ := (n - 4) / 2

/-- The total number of parallel diagonals -/
def total_parallel_diagonals (n : ℕ) : ℕ :=
  parallel_side_pairs n * diagonals_per_parallel_pair n

/-- The number of diagonals not parallel to any side in a regular 32-gon -/
theorem diagonals_not_parallel_in_32gon :
  total_diagonals n - total_parallel_diagonals n = 240 := by
  sorry

end NUMINAMATH_CALUDE_diagonals_not_parallel_in_32gon_l858_85812


namespace NUMINAMATH_CALUDE_perfect_square_sum_l858_85866

theorem perfect_square_sum (n : ℕ+) : 
  (∃ m : ℕ, 4^7 + 4^n.val + 4^1998 = m^2) → (n.val = 1003 ∨ n.val = 3988) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_sum_l858_85866


namespace NUMINAMATH_CALUDE_bucket_capacity_reduction_l858_85856

theorem bucket_capacity_reduction (current_buckets : ℕ) (reduction_factor : ℚ) : 
  current_buckets = 25 → 
  reduction_factor = 2 / 5 →
  ↑(Nat.ceil ((current_buckets : ℚ) / reduction_factor)) = 63 := by
  sorry

end NUMINAMATH_CALUDE_bucket_capacity_reduction_l858_85856


namespace NUMINAMATH_CALUDE_range_of_a_is_closed_interval_two_three_l858_85869

noncomputable def f (x : ℝ) : ℝ := Real.exp (x - 1) + x - 2

def g (a x : ℝ) : ℝ := x^2 - a*x - a + 3

theorem range_of_a_is_closed_interval_two_three :
  ∃ (a : ℝ), ∀ x₁ x₂ : ℝ,
    f x₁ = 0 ∧ g a x₂ = 0 ∧ |x₁ - x₂| ≤ 1 →
    a ∈ Set.Icc 2 3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_is_closed_interval_two_three_l858_85869


namespace NUMINAMATH_CALUDE_polynomial_roots_unit_circle_l858_85840

theorem polynomial_roots_unit_circle (a b c : ℂ) :
  (∀ w : ℂ, w^3 + Complex.abs a * w^2 + Complex.abs b * w + Complex.abs c = 0 → Complex.abs w = 1) →
  (Complex.abs c = 1 ∧ 
   ∀ x : ℂ, x^3 + Complex.abs a * x^2 + Complex.abs b * x + Complex.abs c = 0 ↔ 
            x^3 + Complex.abs a * x^2 + Complex.abs a * x + 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_roots_unit_circle_l858_85840


namespace NUMINAMATH_CALUDE_yanna_purchase_l858_85821

def shirts_cost : ℕ := 10 * 5
def sandals_cost : ℕ := 3 * 3
def hats_cost : ℕ := 5 * 8
def bags_cost : ℕ := 7 * 14
def sunglasses_cost : ℕ := 2 * 12

def total_cost : ℕ := shirts_cost + sandals_cost + hats_cost + bags_cost + sunglasses_cost
def payment : ℕ := 200

theorem yanna_purchase :
  total_cost = payment + 21 :=
by sorry

end NUMINAMATH_CALUDE_yanna_purchase_l858_85821


namespace NUMINAMATH_CALUDE_cement_mixture_weight_l858_85890

theorem cement_mixture_weight (sand water gravel cement limestone total : ℚ) : 
  sand = 2/9 →
  water = 5/18 →
  gravel = 1/6 →
  cement = 7/36 →
  limestone = 1 - (sand + water + gravel + cement) →
  limestone * total = 12 →
  total = 86.4 := by
sorry

end NUMINAMATH_CALUDE_cement_mixture_weight_l858_85890


namespace NUMINAMATH_CALUDE_odd_composite_sum_representation_l858_85889

/-- A number is composite if it has a factor other than 1 and itself -/
def IsComposite (n : ℕ) : Prop :=
  ∃ k m : ℕ, k > 1 ∧ m > 1 ∧ n = k * m

/-- An odd number can be represented as the sum of two composite numbers -/
def CanBeRepresentedAsCompositeSum (n : ℕ) : Prop :=
  ∃ a b : ℕ, IsComposite a ∧ IsComposite b ∧ n = a + b

theorem odd_composite_sum_representation :
  ∀ n : ℕ, n ≥ 13 → Odd n → CanBeRepresentedAsCompositeSum n := by
  sorry

#check odd_composite_sum_representation

end NUMINAMATH_CALUDE_odd_composite_sum_representation_l858_85889


namespace NUMINAMATH_CALUDE_sqrt_comparison_l858_85830

theorem sqrt_comparison : 2 * Real.sqrt 7 < 3 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_comparison_l858_85830


namespace NUMINAMATH_CALUDE_cos_105_degrees_l858_85868

theorem cos_105_degrees : Real.cos (105 * π / 180) = (Real.sqrt 2 - Real.sqrt 6) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_105_degrees_l858_85868


namespace NUMINAMATH_CALUDE_train_lengths_l858_85854

/-- Theorem: Train Lengths
Given:
- A bridge of length 800 meters
- Train A takes 45 seconds to cross the bridge
- Train B takes 40 seconds to cross the bridge
- Train A takes 15 seconds to pass a lamp post
- Train B takes 10 seconds to pass a lamp post

Prove that the length of Train A is 400 meters and the length of Train B is 800/3 meters.
-/
theorem train_lengths (bridge_length : ℝ) (time_A_bridge time_B_bridge time_A_post time_B_post : ℝ)
  (h1 : bridge_length = 800)
  (h2 : time_A_bridge = 45)
  (h3 : time_B_bridge = 40)
  (h4 : time_A_post = 15)
  (h5 : time_B_post = 10) :
  ∃ (length_A length_B : ℝ),
    length_A = 400 ∧ length_B = 800 / 3 ∧
    length_A + bridge_length = (length_A / time_A_post) * time_A_bridge ∧
    length_B + bridge_length = (length_B / time_B_post) * time_B_bridge :=
by
  sorry

end NUMINAMATH_CALUDE_train_lengths_l858_85854


namespace NUMINAMATH_CALUDE_f_negative_2016_l858_85809

def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 1

theorem f_negative_2016 (a : ℝ) :
  f a 2016 = 5 → f a (-2016) = -7 := by
  sorry

end NUMINAMATH_CALUDE_f_negative_2016_l858_85809


namespace NUMINAMATH_CALUDE_product_of_repeating_decimals_l858_85810

-- Define the repeating decimal 0.080808...
def repeating_08 : ℚ := 8 / 99

-- Define the repeating decimal 0.333333...
def repeating_3 : ℚ := 1 / 3

-- Theorem statement
theorem product_of_repeating_decimals : 
  repeating_08 * repeating_3 = 8 / 297 := by
  sorry

end NUMINAMATH_CALUDE_product_of_repeating_decimals_l858_85810


namespace NUMINAMATH_CALUDE_prime_power_of_two_l858_85852

theorem prime_power_of_two (n : ℕ) : 
  Prime (2^n + 1) → ∃ k : ℕ, n = 2^k :=
by sorry

end NUMINAMATH_CALUDE_prime_power_of_two_l858_85852


namespace NUMINAMATH_CALUDE_employee_pay_l858_85853

/-- Given two employees with a total pay of 528 and one paid 120% of the other, prove the lower-paid employee's wage --/
theorem employee_pay (x y : ℝ) (h1 : x + y = 528) (h2 : x = 1.2 * y) : y = 240 := by
  sorry

end NUMINAMATH_CALUDE_employee_pay_l858_85853


namespace NUMINAMATH_CALUDE_eventual_stability_l858_85875

/-- Represents the state of the circular arrangement at a given time step -/
def CircularState := Vector Bool 101

/-- Defines the update rule for a single element based on its neighbors -/
def updateElement (left right current : Bool) : Bool :=
  if left ≠ current ∧ right ≠ current then !current else current

/-- Applies the update rule to the entire circular arrangement -/
def updateState (state : CircularState) : CircularState :=
  Vector.ofFn (fun i =>
    updateElement
      (state.get ((i - 1 + 101) % 101))
      (state.get ((i + 1) % 101))
      (state.get i))

/-- Predicate to check if a state is stable (doesn't change under update) -/
def isStable (state : CircularState) : Prop :=
  updateState state = state

/-- The main theorem: there exists a stable state reachable from any initial state -/
theorem eventual_stability :
  ∀ (initialState : CircularState),
  ∃ (n : ℕ) (stableState : CircularState),
  (∀ k, k ≥ n → (updateState^[k] initialState) = stableState) ∧
  isStable stableState :=
sorry


end NUMINAMATH_CALUDE_eventual_stability_l858_85875


namespace NUMINAMATH_CALUDE_root_in_interval_l858_85827

def f (x : ℝ) : ℝ := x^3 + x + 3

theorem root_in_interval :
  ∃ x ∈ Set.Ioo (-2 : ℝ) (-1), f x = 0 :=
sorry

end NUMINAMATH_CALUDE_root_in_interval_l858_85827


namespace NUMINAMATH_CALUDE_concentric_circles_ratio_l858_85837

theorem concentric_circles_ratio (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) 
  (h3 : π * b^2 - π * a^2 = 4 * (π * a^2)) : a / b = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_concentric_circles_ratio_l858_85837


namespace NUMINAMATH_CALUDE_theo_cookie_eating_frequency_l858_85822

/-- The number of cookies Theo eats each time -/
def cookies_per_time : ℕ := 13

/-- The number of days Theo eats cookies each month -/
def days_per_month : ℕ := 20

/-- The number of cookies Theo eats in 3 months -/
def cookies_in_three_months : ℕ := 2340

/-- The number of times Theo eats cookies per day -/
def times_per_day : ℕ := 3

theorem theo_cookie_eating_frequency :
  times_per_day * cookies_per_time * days_per_month * 3 = cookies_in_three_months :=
by sorry

end NUMINAMATH_CALUDE_theo_cookie_eating_frequency_l858_85822


namespace NUMINAMATH_CALUDE_cow_milk_production_l858_85898

/-- Given a number of cows and total weekly milk production, 
    calculate the daily milk production per cow. -/
def daily_milk_per_cow (num_cows : ℕ) (weekly_milk : ℕ) : ℚ :=
  (weekly_milk : ℚ) / 7 / num_cows

theorem cow_milk_production : daily_milk_per_cow 52 1820 = 5 := by
  sorry

end NUMINAMATH_CALUDE_cow_milk_production_l858_85898


namespace NUMINAMATH_CALUDE_train_length_l858_85806

/-- The length of a train given its speed and time to cross a pole --/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 60 → time_s = 18 → ∃ (length_m : ℝ), abs (length_m - 300) < 1 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l858_85806


namespace NUMINAMATH_CALUDE_cat_food_consumption_l858_85814

/-- Represents the amount of food eaten by the cat each day -/
def daily_consumption : ℚ := 1/3 + 1/4

/-- Represents the total number of cans available -/
def total_cans : ℚ := 6

/-- Represents the day on which the cat finishes all the food -/
def finish_day : ℕ := 4

theorem cat_food_consumption :
  ∃ (n : ℕ), n * daily_consumption > total_cans ∧ (n - 1) * daily_consumption ≤ total_cans ∧ n = finish_day :=
by sorry

end NUMINAMATH_CALUDE_cat_food_consumption_l858_85814


namespace NUMINAMATH_CALUDE_tiger_deer_chase_theorem_l858_85828

/-- Represents the chase between a tiger and a deer --/
structure TigerDeerChase where
  tiger_leaps_per_minute : ℕ
  deer_leaps_per_minute : ℕ
  tiger_meters_per_leap : ℕ
  deer_meters_per_leap : ℕ
  catch_distance : ℕ

/-- The number of leaps the tiger is initially behind the deer --/
def initial_leap_difference (chase : TigerDeerChase) : ℕ :=
  sorry

/-- Theorem stating the initial leap difference for the given chase scenario --/
theorem tiger_deer_chase_theorem (chase : TigerDeerChase) 
  (h1 : chase.tiger_leaps_per_minute = 5)
  (h2 : chase.deer_leaps_per_minute = 4)
  (h3 : chase.tiger_meters_per_leap = 8)
  (h4 : chase.deer_meters_per_leap = 5)
  (h5 : chase.catch_distance = 800) :
  initial_leap_difference chase = 40 := by
  sorry

end NUMINAMATH_CALUDE_tiger_deer_chase_theorem_l858_85828


namespace NUMINAMATH_CALUDE_ewan_sequence_contains_113_l858_85800

def ewanSequence (n : ℕ) : ℤ := 3 + 11 * (n - 1)

theorem ewan_sequence_contains_113 :
  ∃ n : ℕ, ewanSequence n = 113 ∧
  (∀ m : ℕ, ewanSequence m ≠ 111) ∧
  (∀ m : ℕ, ewanSequence m ≠ 112) ∧
  (∀ m : ℕ, ewanSequence m ≠ 110) ∧
  (∀ m : ℕ, ewanSequence m ≠ 114) :=
by sorry


end NUMINAMATH_CALUDE_ewan_sequence_contains_113_l858_85800


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l858_85801

theorem geometric_sequence_sum (a : ℕ → ℝ) (l : ℝ) :
  (∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r) →  -- geometric sequence condition
  a 4 = 8 * a 1 →  -- given condition
  (a 1 + (a 2 + l) + a 3) * 2 = a 1 + (a 2 + l) * 2 + a 3 →  -- arithmetic sequence condition
  a 1 + a 2 + a 3 + a 4 + a 5 = 62 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l858_85801


namespace NUMINAMATH_CALUDE_rest_of_body_length_l858_85834

theorem rest_of_body_length (total_height legs_ratio head_ratio : ℚ) : 
  total_height = 60 →
  legs_ratio = 1/3 →
  head_ratio = 1/4 →
  total_height - (legs_ratio * total_height + head_ratio * total_height) = 25 := by
sorry

end NUMINAMATH_CALUDE_rest_of_body_length_l858_85834


namespace NUMINAMATH_CALUDE_cyclic_quadrilateral_intersection_product_l858_85831

-- Define the types for points and circles
variable (Point Circle : Type)

-- Define the necessary geometric relations
variable (is_convex_cyclic_quadrilateral : Point → Point → Point → Point → Prop)
variable (is_center_of_circumcircle : Point → Point → Point → Point → Point → Prop)
variable (is_on_circle : Point → Circle → Prop)
variable (circumcircle : Point → Point → Point → Circle)
variable (intersection_point : Circle → Circle → Point)

-- Define the distance function
variable (distance : Point → Point → ℝ)

theorem cyclic_quadrilateral_intersection_product
  (A B C D O Q : Point)
  (h1 : is_convex_cyclic_quadrilateral A B C D)
  (h2 : is_center_of_circumcircle O A B C D)
  (h3 : Q = intersection_point (circumcircle O A B) (circumcircle O C D))
  : distance Q A * distance Q B = distance Q C * distance Q D := by
  sorry

end NUMINAMATH_CALUDE_cyclic_quadrilateral_intersection_product_l858_85831


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l858_85859

theorem partial_fraction_decomposition :
  ∀ x : ℝ, x ≠ 6 → x ≠ -3 →
  (4 * x - 3) / (x^2 - 3 * x - 18) = (7 / 3) / (x - 6) + (5 / 3) / (x + 3) := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l858_85859


namespace NUMINAMATH_CALUDE_functional_equation_implies_constant_l858_85880

/-- A function from ℤ² to [0,1] satisfying the given functional equation -/
def FunctionalEquation (f : ℤ × ℤ → ℝ) : Prop :=
  ∀ x y : ℤ, 0 ≤ f (x, y) ∧ f (x, y) ≤ 1 ∧ 
  f (x, y) = (f (x - 1, y) + f (x, y - 1)) / 2

/-- Theorem stating that any function satisfying the functional equation must be constant -/
theorem functional_equation_implies_constant 
  (f : ℤ × ℤ → ℝ) 
  (h : FunctionalEquation f) : 
  ∃ c : ℝ, c ∈ Set.Icc 0 1 ∧ ∀ x y : ℤ, f (x, y) = c :=
sorry

end NUMINAMATH_CALUDE_functional_equation_implies_constant_l858_85880


namespace NUMINAMATH_CALUDE_quadrilateral_tile_exists_l858_85823

/-- A quadrilateral tile with angles measured in degrees -/
structure QuadTile where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  angle4 : ℝ

/-- The property that six tiles meet at a vertex -/
def sixTilesMeet (t : QuadTile) : Prop :=
  ∃ (i : Fin 4), t.angle1 * (i.val : ℝ) + t.angle2 * ((4 - i).val : ℝ) = 360

/-- The sum of angles in a quadrilateral is 360° -/
def validQuadrilateral (t : QuadTile) : Prop :=
  t.angle1 + t.angle2 + t.angle3 + t.angle4 = 360

/-- The main theorem: there exists a quadrilateral tile with the specified angles -/
theorem quadrilateral_tile_exists : ∃ (t : QuadTile), 
  t.angle1 = 45 ∧ t.angle2 = 60 ∧ t.angle3 = 105 ∧ t.angle4 = 150 ∧
  sixTilesMeet t ∧ validQuadrilateral t :=
by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_tile_exists_l858_85823


namespace NUMINAMATH_CALUDE_correct_operation_l858_85876

theorem correct_operation (m : ℝ) : (2 * m^3)^2 / (2 * m)^2 = m^4 := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l858_85876


namespace NUMINAMATH_CALUDE_stream_speed_l858_85855

/-- Proves that given a boat with a speed of 22 km/hr in still water,
    traveling 189 km downstream in 7 hours, the speed of the stream is 5 km/hr. -/
theorem stream_speed (boat_speed : ℝ) (downstream_distance : ℝ) (downstream_time : ℝ) :
  boat_speed = 22 →
  downstream_distance = 189 →
  downstream_time = 7 →
  ∃ stream_speed : ℝ,
    stream_speed = 5 ∧
    downstream_distance = (boat_speed + stream_speed) * downstream_time :=
by
  sorry


end NUMINAMATH_CALUDE_stream_speed_l858_85855


namespace NUMINAMATH_CALUDE_circle_C_equation_max_y_over_x_l858_85835

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 2

-- Define the line that intersects with x-axis to form the center of circle C
def center_line (x y : ℝ) : Prop := x - y + 1 = 0

-- Define the line tangent to circle C
def tangent_line (x y : ℝ) : Prop := x + y + 3 = 0

-- Define the circle for the second part of the problem
def circle_P (x y : ℝ) : Prop := x^2 + y^2 - 4*y + 3 = 0

-- Theorem for the first part of the problem
theorem circle_C_equation :
  ∀ x y : ℝ, 
  (∃ x₀, center_line x₀ 0 ∧ (∀ x y, circle_C x y → (x - x₀)^2 + y^2 = 2)) →
  (∃ d : ℝ, d > 0 ∧ ∀ x y, circle_C x y → d = |x + y + 3| / Real.sqrt 2) →
  circle_C x y ↔ (x + 1)^2 + y^2 = 2 :=
sorry

-- Theorem for the second part of the problem
theorem max_y_over_x :
  (∃ k : ℝ, k = Real.sqrt 3 / 3 ∧ 
   ∀ x y : ℝ, circle_P x y → |y / x| ≤ k ∧ 
   ∃ x₀ y₀ : ℝ, circle_P x₀ y₀ ∧ |y₀ / x₀| = k) :=
sorry

end NUMINAMATH_CALUDE_circle_C_equation_max_y_over_x_l858_85835


namespace NUMINAMATH_CALUDE_quadratic_integral_mean_value_l858_85807

/-- Given a quadratic function f(x) = ax^2 + c (a ≠ 0), 
    if the integral of f from 0 to 1 equals f(x_0) and 0 ≤ x_0 ≤ 1,
    then x_0 = √3/3 -/
theorem quadratic_integral_mean_value (a c x₀ : ℝ) (ha : a ≠ 0) :
  (∫ x in (0:ℝ)..1, a * x^2 + c) = a * x₀^2 + c →
  0 ≤ x₀ ∧ x₀ ≤ 1 →
  x₀ = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_integral_mean_value_l858_85807


namespace NUMINAMATH_CALUDE_base_k_conversion_l858_85883

theorem base_k_conversion (k : ℕ) : 
  (1 * k^2 + 3 * k + 2 = 30) → k = 4 := by
  sorry

end NUMINAMATH_CALUDE_base_k_conversion_l858_85883


namespace NUMINAMATH_CALUDE_smallest_n_mod_congruence_l858_85858

theorem smallest_n_mod_congruence :
  ∃ (n : ℕ), n > 0 ∧ (17 * n) % 7 = 1234 % 7 ∧
  ∀ (m : ℕ), m > 0 ∧ (17 * m) % 7 = 1234 % 7 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_mod_congruence_l858_85858


namespace NUMINAMATH_CALUDE_expression_factorization_l858_85808

theorem expression_factorization (a b c : ℝ) :
  ((a^2 - b^2)^3 + (b^2 - c^2)^3 + (c^2 - a^2)^3) / ((a - b)^3 + (b - c)^3 + (c - a)^3) = (a + b) * (a + c) * (b + c) :=
by sorry

end NUMINAMATH_CALUDE_expression_factorization_l858_85808


namespace NUMINAMATH_CALUDE_inclination_angle_range_l858_85818

/-- Given a line with slope k in [-1, √3] and inclination angle α in [0, π),
    prove that the range of α is [0, π/3] ∪ [3π/4, π) -/
theorem inclination_angle_range (k α : ℝ) :
  k ∈ Set.Icc (-1) (Real.sqrt 3) →
  α ∈ Set.Ico 0 π →
  k = Real.tan α →
  α ∈ Set.Icc 0 (π / 3) ∪ Set.Ico (3 * π / 4) π :=
sorry

end NUMINAMATH_CALUDE_inclination_angle_range_l858_85818


namespace NUMINAMATH_CALUDE_angle_sum_equals_pi_over_four_l858_85879

theorem angle_sum_equals_pi_over_four (α β : Real) 
  (h1 : 0 < α) (h2 : α < π / 2) 
  (h3 : 0 < β) (h4 : β < π / 2)
  (h5 : Real.tan α = 1 / 7)
  (h6 : Real.tan β = 3 / 4) : 
  α + β = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_equals_pi_over_four_l858_85879


namespace NUMINAMATH_CALUDE_units_digit_17_times_29_l858_85817

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_17_times_29 : units_digit (17 * 29) = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_17_times_29_l858_85817


namespace NUMINAMATH_CALUDE_smallest_group_size_l858_85803

theorem smallest_group_size : ∃ n : ℕ, n > 0 ∧ n % 2 = 0 ∧ (∃ m : ℕ, m > 2 ∧ n % m = 0) ∧ (∀ k : ℕ, k > 0 ∧ k % 2 = 0 ∧ (∃ l : ℕ, l > 2 ∧ k % l = 0) → k ≥ 6) :=
by sorry

end NUMINAMATH_CALUDE_smallest_group_size_l858_85803


namespace NUMINAMATH_CALUDE_smaller_solution_of_quadratic_l858_85863

theorem smaller_solution_of_quadratic (x : ℝ) :
  x^2 - 14*x + 45 = 0 → ∃ y : ℝ, y^2 - 14*y + 45 = 0 ∧ y ≠ x ∧ (∀ z : ℝ, z^2 - 14*z + 45 = 0 → z = x ∨ z = y) ∧ min x y = 5 :=
by sorry

end NUMINAMATH_CALUDE_smaller_solution_of_quadratic_l858_85863


namespace NUMINAMATH_CALUDE_sum_interior_angles_regular_polygon_l858_85864

/-- 
Given a regular polygon where each exterior angle measures 40°, 
the sum of its interior angles is 1260°.
-/
theorem sum_interior_angles_regular_polygon (n : ℕ) 
  (h_exterior : (360 : ℝ) / n = 40) : 
  (n - 2 : ℝ) * 180 = 1260 := by
  sorry

end NUMINAMATH_CALUDE_sum_interior_angles_regular_polygon_l858_85864


namespace NUMINAMATH_CALUDE_banana_arrangements_l858_85861

def word_length : ℕ := 6
def a_count : ℕ := 3
def n_count : ℕ := 2
def b_count : ℕ := 1

theorem banana_arrangements : 
  (word_length.factorial) / (a_count.factorial * n_count.factorial * b_count.factorial) = 60 := by
  sorry

end NUMINAMATH_CALUDE_banana_arrangements_l858_85861


namespace NUMINAMATH_CALUDE_point_outside_circle_iff_m_in_range_l858_85887

/-- A circle in the x-y plane defined by the equation x^2 + y^2 + 2x - m = 0 -/
def Circle (m : ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 + 2*p.1 - m = 0}

/-- The point P with coordinates (1,1) -/
def P : ℝ × ℝ := (1, 1)

/-- Predicate to check if a point is outside a circle -/
def IsOutside (p : ℝ × ℝ) (c : Set (ℝ × ℝ)) : Prop :=
  ∀ q ∈ c, (p.1 - q.1)^2 + (p.2 - q.2)^2 > 0

theorem point_outside_circle_iff_m_in_range :
  ∀ m : ℝ, IsOutside P (Circle m) ↔ -1 < m ∧ m < 4 :=
sorry

end NUMINAMATH_CALUDE_point_outside_circle_iff_m_in_range_l858_85887


namespace NUMINAMATH_CALUDE_borrowed_sheets_theorem_l858_85878

/-- Represents a set of notes with sheets and pages -/
structure Notes where
  total_sheets : ℕ
  pages_per_sheet : ℕ
  total_pages : ℕ
  h_pages : total_pages = total_sheets * pages_per_sheet

/-- Represents the state of notes after some sheets are borrowed -/
structure BorrowedNotes where
  original : Notes
  borrowed_sheets : ℕ
  sheets_before : ℕ
  h_valid : sheets_before + borrowed_sheets < original.total_sheets

/-- Calculates the average page number of remaining sheets -/
def average_page_number (bn : BorrowedNotes) : ℚ :=
  let remaining_pages := bn.original.total_pages - bn.borrowed_sheets * bn.original.pages_per_sheet
  let sum_before := bn.sheets_before * (bn.sheets_before * bn.original.pages_per_sheet + 1)
  let first_after := (bn.sheets_before + bn.borrowed_sheets) * bn.original.pages_per_sheet + 1
  let last_after := bn.original.total_pages
  let sum_after := (first_after + last_after) * (last_after - first_after + 1) / 2
  (sum_before + sum_after) / remaining_pages

/-- Theorem stating that if 17 sheets are borrowed from a 35-sheet set of notes,
    the average page number of remaining sheets is 28 -/
theorem borrowed_sheets_theorem (bn : BorrowedNotes)
  (h_total_sheets : bn.original.total_sheets = 35)
  (h_pages_per_sheet : bn.original.pages_per_sheet = 2)
  (h_total_pages : bn.original.total_pages = 70)
  (h_avg : average_page_number bn = 28) :
  bn.borrowed_sheets = 17 := by
  sorry

end NUMINAMATH_CALUDE_borrowed_sheets_theorem_l858_85878


namespace NUMINAMATH_CALUDE_gray_squares_33_l858_85846

/-- The number of squares in the n-th figure of the series -/
def total_squares (n : ℕ) : ℕ := (2 * n - 1) ^ 2

/-- The number of black squares in the n-th figure -/
def black_squares (n : ℕ) : ℕ := n ^ 2

/-- The number of white squares in the n-th figure -/
def white_squares (n : ℕ) : ℕ := (n - 1) ^ 2

/-- The number of gray squares in the n-th figure -/
def gray_squares (n : ℕ) : ℕ := total_squares n - black_squares n - white_squares n

theorem gray_squares_33 : gray_squares 33 = 2112 := by
  sorry

end NUMINAMATH_CALUDE_gray_squares_33_l858_85846


namespace NUMINAMATH_CALUDE_marble_problem_l858_85847

theorem marble_problem : 
  ∀ (x : ℚ), x > 0 →
  let bag1 := x
  let bag2 := 2 * x
  let bag3 := 3 * x
  let green1 := (1 / 2) * bag1
  let green2 := (1 / 3) * bag2
  let green3 := (1 / 4) * bag3
  let total_green := green1 + green2 + green3
  let total_marbles := bag1 + bag2 + bag3
  (total_green / total_marbles) = 23 / 72 :=
by
  sorry

end NUMINAMATH_CALUDE_marble_problem_l858_85847


namespace NUMINAMATH_CALUDE_quadratic_vertex_l858_85870

/-- The quadratic function f(x) = -3x^2 - 2 has its vertex at (0, -2). -/
theorem quadratic_vertex (x : ℝ) :
  let f : ℝ → ℝ := λ x => -3 * x^2 - 2
  (∀ x, f x ≤ f 0) ∧ f 0 = -2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_vertex_l858_85870


namespace NUMINAMATH_CALUDE_shekar_weighted_average_l858_85816

def scores : List Float := [92, 78, 85, 67, 89, 74, 81, 95, 70, 88]
def weights : List Float := [0.20, 0.10, 0.15, 0.08, 0.12, 0.05, 0.07, 0.10, 0.05, 0.08]

def weighted_average (scores : List Float) (weights : List Float) : Float :=
  (List.zip scores weights).map (fun (s, w) => s * w) |> List.sum

theorem shekar_weighted_average :
  weighted_average scores weights = 84.4 := by
  sorry

end NUMINAMATH_CALUDE_shekar_weighted_average_l858_85816


namespace NUMINAMATH_CALUDE_circle_c_equation_l858_85862

/-- A circle C satisfying the given conditions -/
structure CircleC where
  center : ℝ × ℝ
  radius : ℝ
  tangent_y_axis : center.1 = radius
  chord_length : 4 * radius ^ 2 - center.1 ^ 2 = 12
  center_on_line : center.2 = 1/2 * center.1

/-- The equation of the circle C -/
def circle_equation (c : CircleC) (x y : ℝ) : Prop :=
  (x - c.center.1) ^ 2 + (y - c.center.2) ^ 2 = c.radius ^ 2

/-- Theorem stating that the circle C has the equation (x-2)² + (y-1)² = 4 -/
theorem circle_c_equation (c : CircleC) :
  ∀ x y, circle_equation c x y ↔ (x - 2) ^ 2 + (y - 1) ^ 2 = 4 :=
by sorry

end NUMINAMATH_CALUDE_circle_c_equation_l858_85862


namespace NUMINAMATH_CALUDE_range_of_m_l858_85860

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, |x + 1| + |x - 3| ≥ |m - 1|) → m ∈ Set.Icc (-3) 5 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l858_85860


namespace NUMINAMATH_CALUDE_equation_satisfied_at_nine_l858_85826

/-- The sum of an infinite geometric series with first term a and common ratio r. -/
noncomputable def geometricSum (a : ℝ) (r : ℝ) : ℝ := a / (1 - r)

/-- Left-hand side of the equation -/
noncomputable def leftHandSide : ℝ := 
  (geometricSum 1 (1/3)) * (geometricSum 1 (-1/3))

/-- Right-hand side of the equation -/
noncomputable def rightHandSide (y : ℝ) : ℝ := 
  geometricSum 1 (1/y)

/-- The theorem stating that the equation is satisfied when y = 9 -/
theorem equation_satisfied_at_nine : 
  leftHandSide = rightHandSide 9 := by sorry

end NUMINAMATH_CALUDE_equation_satisfied_at_nine_l858_85826


namespace NUMINAMATH_CALUDE_tangent_and_maximum_inequality_and_range_l858_85802

noncomputable def f (a b x : ℝ) : ℝ := a * Real.log x - b * x^2

theorem tangent_and_maximum (a b : ℝ) :
  (f a b 1 = -1/2) →
  (∀ x, x ∈ Set.Icc (1/Real.exp 1) (Real.exp 1) → f a b x ≤ 0) →
  (∃ x, x ∈ Set.Icc (1/Real.exp 1) (Real.exp 1) ∧ f a b x = 0) :=
sorry

theorem inequality_and_range (m : ℝ) :
  (∀ a x, a ∈ Set.Icc 0 (3/2) → x ∈ Set.Ioo 1 (Real.exp 2) →
    f a 0 x ≥ m + x) →
  m ∈ Set.Iic (-Real.exp 2) :=
sorry

end NUMINAMATH_CALUDE_tangent_and_maximum_inequality_and_range_l858_85802


namespace NUMINAMATH_CALUDE_complex_modulus_equation_l858_85873

theorem complex_modulus_equation (m : ℝ) (h1 : m > 0) :
  Complex.abs (5 + m * Complex.I) = 5 * Real.sqrt 26 → m = 25 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_equation_l858_85873


namespace NUMINAMATH_CALUDE_least_xy_value_l858_85891

theorem least_xy_value (x y : ℕ+) (h : (1 : ℚ) / x + (1 : ℚ) / (3 * y) = (1 : ℚ) / 9) :
  (∀ a b : ℕ+, (1 : ℚ) / a + (1 : ℚ) / (3 * b) = (1 : ℚ) / 9 → (a : ℕ) * b ≥ (x : ℕ) * y) ∧
  (x : ℕ) * y = 108 :=
sorry

end NUMINAMATH_CALUDE_least_xy_value_l858_85891


namespace NUMINAMATH_CALUDE_union_A_B_when_a_is_one_A_subset_complement_B_iff_l858_85867

-- Define set A
def A : Set ℝ := {x | (x - 1) / (x - 5) < 0}

-- Define set B
def B (a : ℝ) : Set ℝ := {x | x^2 - 2*a*x + a^2 - 1 < 0}

-- Part 1
theorem union_A_B_when_a_is_one : 
  A ∪ B 1 = {x : ℝ | 0 < x ∧ x < 5} := by sorry

-- Part 2
theorem A_subset_complement_B_iff : 
  ∀ a : ℝ, A ⊆ (Set.univ \ B a) ↔ a ≤ 0 ∨ a ≥ 6 := by sorry

end NUMINAMATH_CALUDE_union_A_B_when_a_is_one_A_subset_complement_B_iff_l858_85867


namespace NUMINAMATH_CALUDE_sum_and_product_membership_l858_85843

def P : Set ℤ := {x | ∃ k, x = 2 * k - 1}
def Q : Set ℤ := {y | ∃ n, y = 2 * n}

theorem sum_and_product_membership (x y : ℤ) (hx : x ∈ P) (hy : y ∈ Q) :
  (x + y) ∈ P ∧ (x * y) ∈ Q := by
  sorry

end NUMINAMATH_CALUDE_sum_and_product_membership_l858_85843


namespace NUMINAMATH_CALUDE_rectangle_dimension_change_l858_85838

theorem rectangle_dimension_change (original_length original_width : ℝ) 
  (h_positive_length : original_length > 0)
  (h_positive_width : original_width > 0) :
  let new_length := 1.4 * original_length
  let new_width := original_width * (1 - 0.2857)
  new_length * new_width = original_length * original_width := by
  sorry

end NUMINAMATH_CALUDE_rectangle_dimension_change_l858_85838


namespace NUMINAMATH_CALUDE_josie_cabinet_unlock_time_l858_85844

/-- Represents the shopping trip details -/
structure ShoppingTrip where
  total_time : ℕ
  shopping_time : ℕ
  cart_wait_time : ℕ
  restocking_wait_time : ℕ
  checkout_wait_time : ℕ

/-- Calculates the time spent waiting for the cabinet to be unlocked -/
def cabinet_unlock_time (trip : ShoppingTrip) : ℕ :=
  trip.total_time - trip.shopping_time - trip.cart_wait_time - trip.restocking_wait_time - trip.checkout_wait_time

/-- Theorem stating that Josie waited 13 minutes for the cabinet to be unlocked -/
theorem josie_cabinet_unlock_time :
  let trip := ShoppingTrip.mk 90 42 3 14 18
  cabinet_unlock_time trip = 13 := by sorry

end NUMINAMATH_CALUDE_josie_cabinet_unlock_time_l858_85844


namespace NUMINAMATH_CALUDE_positive_integer_sum_with_square_twelve_l858_85874

theorem positive_integer_sum_with_square_twelve (M : ℕ+) :
  (M : ℝ)^2 + M = 12 → M = 3 := by
  sorry

end NUMINAMATH_CALUDE_positive_integer_sum_with_square_twelve_l858_85874


namespace NUMINAMATH_CALUDE_min_m_value_l858_85884

/-- The function f(x) = x^2 - 3x --/
def f (x : ℝ) : ℝ := x^2 - 3*x

/-- The interval [-3, 2] --/
def I : Set ℝ := Set.Icc (-3) 2

/-- The theorem statement --/
theorem min_m_value :
  ∃ (m : ℝ), m = 81/4 ∧ 
  (∀ (x₁ x₂ : ℝ), x₁ ∈ I → x₂ ∈ I → |f x₁ - f x₂| ≤ m) ∧
  (∀ (m' : ℝ), (∀ (x₁ x₂ : ℝ), x₁ ∈ I → x₂ ∈ I → |f x₁ - f x₂| ≤ m') → m ≤ m') :=
sorry

end NUMINAMATH_CALUDE_min_m_value_l858_85884


namespace NUMINAMATH_CALUDE_recipe_flour_calculation_l858_85851

theorem recipe_flour_calculation :
  let full_recipe : ℚ := 7 + 3/4
  let one_third_recipe : ℚ := (1/3) * full_recipe
  one_third_recipe = 2 + 7/12 := by sorry

end NUMINAMATH_CALUDE_recipe_flour_calculation_l858_85851


namespace NUMINAMATH_CALUDE_systematic_sample_count_in_range_l858_85849

/-- Systematic sampling function -/
def systematicSample (total : ℕ) (sampleSize : ℕ) (start : ℕ) : List ℕ :=
  List.range sampleSize |>.map (fun i => (start + i * (total / sampleSize)) % total + 1)

/-- Count numbers in a given range -/
def countInRange (list : List ℕ) (low : ℕ) (high : ℕ) : ℕ :=
  list.filter (fun n => low ≤ n && n ≤ high) |>.length

theorem systematic_sample_count_in_range :
  let total := 840
  let sampleSize := 42
  let start := 13
  let sample := systematicSample total sampleSize start
  countInRange sample 490 700 = 11 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sample_count_in_range_l858_85849


namespace NUMINAMATH_CALUDE_unfair_coin_probability_l858_85832

def num_flips : ℕ := 10
def num_heads : ℕ := 3
def prob_heads : ℚ := 1/3
def prob_tails : ℚ := 2/3

theorem unfair_coin_probability : 
  (Nat.choose num_flips num_heads : ℚ) * prob_heads ^ num_heads * prob_tails ^ (num_flips - num_heads) = 15360/59049 := by
  sorry

end NUMINAMATH_CALUDE_unfair_coin_probability_l858_85832


namespace NUMINAMATH_CALUDE_total_schedules_l858_85804

/-- Represents the number of classes to be scheduled -/
def num_classes : ℕ := 4

/-- Represents the number of classes that can be scheduled in the first period -/
def first_period_options : ℕ := 3

/-- Calculates the factorial of a natural number -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- Theorem: The total number of different possible schedules is 18 -/
theorem total_schedules : 
  first_period_options * factorial (num_classes - 1) = 18 := by
  sorry

end NUMINAMATH_CALUDE_total_schedules_l858_85804


namespace NUMINAMATH_CALUDE_staircase_perimeter_l858_85820

/-- A staircase-shaped region with right angles -/
structure StaircaseRegion where
  /-- Number of congruent sides -/
  num_sides : ℕ
  /-- Length of each congruent side -/
  side_length : ℝ
  /-- Area of the region -/
  area : ℝ

/-- Calculates the perimeter of a StaircaseRegion -/
def perimeter (s : StaircaseRegion) : ℝ :=
  sorry

theorem staircase_perimeter (s : StaircaseRegion) 
  (h1 : s.num_sides = 12)
  (h2 : s.side_length = 1)
  (h3 : s.area = 89) :
  perimeter s = 43 := by
  sorry

end NUMINAMATH_CALUDE_staircase_perimeter_l858_85820


namespace NUMINAMATH_CALUDE_travel_time_difference_l858_85896

/-- Proves the equation for the travel time difference between two groups -/
theorem travel_time_difference 
  (x : ℝ) -- walking speed in km/h
  (h1 : x > 0) -- walking speed is positive
  (distance : ℝ) -- distance traveled
  (h2 : distance = 4) -- distance is 4 km
  (time_diff : ℝ) -- time difference in hours
  (h3 : time_diff = 1/3) -- time difference is 1/3 hours
  : 
  distance / x - distance / (2 * x) = time_diff :=
by sorry

end NUMINAMATH_CALUDE_travel_time_difference_l858_85896


namespace NUMINAMATH_CALUDE_max_students_is_18_l858_85836

/-- Represents the structure of Ms. Gregory's class -/
structure ClassStructure where
  boys : ℕ
  girls : ℕ
  science_club : ℕ
  math_club : ℕ

/-- Checks if the given class structure satisfies all conditions -/
def is_valid_structure (c : ClassStructure) : Prop :=
  3 * c.boys = 4 * c.science_club ∧ 
  2 * c.girls = 3 * c.science_club ∧ 
  c.math_club = 2 * c.science_club ∧
  c.boys + c.girls = c.science_club + c.math_club

/-- The maximum number of students in Ms. Gregory's class -/
def max_students : ℕ := 18

/-- Theorem stating that the maximum number of students is 18 -/
theorem max_students_is_18 : 
  ∀ c : ClassStructure, is_valid_structure c → c.boys + c.girls ≤ max_students :=
by
  sorry

#check max_students_is_18

end NUMINAMATH_CALUDE_max_students_is_18_l858_85836


namespace NUMINAMATH_CALUDE_binary_101_is_5_l858_85894

/-- Converts a binary number represented as a list of bits (least significant bit first) to its decimal equivalent. -/
def binary_to_decimal (bits : List Bool) : Nat :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 101₍₂₎ -/
def binary_101 : List Bool := [true, false, true]

/-- Theorem stating that the decimal representation of 101₍₂₎ is 5 -/
theorem binary_101_is_5 : binary_to_decimal binary_101 = 5 := by
  sorry

end NUMINAMATH_CALUDE_binary_101_is_5_l858_85894


namespace NUMINAMATH_CALUDE_interval_for_72_students_8_sample_l858_85839

/-- The interval of segmentation in systematic sampling -/
def interval_of_segmentation (total_students : ℕ) (sample_size : ℕ) : ℕ :=
  total_students / sample_size

/-- Theorem: The interval of segmentation for 72 students and sample size 8 is 9 -/
theorem interval_for_72_students_8_sample : 
  interval_of_segmentation 72 8 = 9 := by
  sorry

end NUMINAMATH_CALUDE_interval_for_72_students_8_sample_l858_85839


namespace NUMINAMATH_CALUDE_angle_properties_l858_85872

/-- Given an angle α with vertex at the origin, initial side on the positive x-axis,
    and terminal side passing through (-3/5, -4/5), prove properties about α and β. -/
theorem angle_properties (α β : Real) : 
  (∃ (P : Real × Real), P.1 = -3/5 ∧ P.2 = -4/5 ∧ 
   Real.cos α = -3/5 ∧ Real.sin α = -4/5) →
  Real.sin (α + π) = 4/5 ∧
  (Real.sin (α + β) = 5/13 → Real.cos β = -56/65 ∨ Real.cos β = 16/65) := by
  sorry

end NUMINAMATH_CALUDE_angle_properties_l858_85872


namespace NUMINAMATH_CALUDE_soft_drink_bottles_l858_85841

theorem soft_drink_bottles (small_bottles : ℕ) : 
  (10000 : ℕ) * (85 : ℕ) / 100 + small_bottles * (88 : ℕ) / 100 = (13780 : ℕ) →
  small_bottles = (6000 : ℕ) :=
by sorry

end NUMINAMATH_CALUDE_soft_drink_bottles_l858_85841


namespace NUMINAMATH_CALUDE_system_solution_l858_85892

/-- Prove that the solution to the system of equations:
    4x - 3y = -10
    6x + 5y = -13
    is (-89/38, 0.21053) -/
theorem system_solution : 
  ∃ (x y : ℝ), 
    (4 * x - 3 * y = -10) ∧ 
    (6 * x + 5 * y = -13) ∧ 
    (x = -89 / 38) ∧ 
    (y = 0.21053) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l858_85892


namespace NUMINAMATH_CALUDE_page_number_added_twice_l858_85865

theorem page_number_added_twice (n : ℕ) (x : ℕ) 
  (h1 : x ≤ n) 
  (h2 : n * (n + 1) / 2 + x = 3050) : 
  x = 47 := by
  sorry

end NUMINAMATH_CALUDE_page_number_added_twice_l858_85865


namespace NUMINAMATH_CALUDE_carols_piggy_bank_l858_85813

/-- Represents the contents of Carol's piggy bank -/
structure PiggyBank where
  nickels : ℕ
  dimes : ℕ

/-- The value of the piggy bank in cents -/
def bankValue (bank : PiggyBank) : ℕ :=
  5 * bank.nickels + 10 * bank.dimes

theorem carols_piggy_bank :
  ∃ (bank : PiggyBank),
    bankValue bank = 455 ∧
    bank.nickels = bank.dimes + 7 ∧
    bank.nickels = 35 := by
  sorry

end NUMINAMATH_CALUDE_carols_piggy_bank_l858_85813


namespace NUMINAMATH_CALUDE_q_div_p_equals_225_l858_85886

/- Define the total number of cards -/
def total_cards : ℕ := 50

/- Define the number of different numbers on the cards -/
def distinct_numbers : ℕ := 10

/- Define the number of cards for each number -/
def cards_per_number : ℕ := 5

/- Define the number of cards drawn -/
def cards_drawn : ℕ := 5

/- Function to calculate binomial coefficient -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/- Probability of drawing all cards with the same number -/
def p : ℚ := (distinct_numbers : ℚ) / choose total_cards cards_drawn

/- Probability of drawing four cards with one number and one card with a different number -/
def q : ℚ := (2250 : ℚ) / choose total_cards cards_drawn

/- Theorem stating the ratio of q to p -/
theorem q_div_p_equals_225 : q / p = 225 := by sorry

end NUMINAMATH_CALUDE_q_div_p_equals_225_l858_85886


namespace NUMINAMATH_CALUDE_mikes_net_spending_l858_85871

/-- The net amount Mike spent at the music store -/
def net_amount (trumpet_cost song_book_price : ℚ) : ℚ :=
  trumpet_cost - song_book_price

/-- Theorem stating that Mike's net spending is $139.32 -/
theorem mikes_net_spending :
  let trumpet_cost : ℚ := 145.16
  let song_book_price : ℚ := 5.84
  net_amount trumpet_cost song_book_price = 139.32 := by
  sorry

end NUMINAMATH_CALUDE_mikes_net_spending_l858_85871


namespace NUMINAMATH_CALUDE_sum_103_odd_numbers_from_63_l858_85857

/-- The sum of the first n odd numbers starting from a given odd number -/
def sumOddNumbers (start : ℕ) (n : ℕ) : ℕ :=
  n * (2 * start + n - 1)

/-- Theorem: The sum of the first 103 odd numbers starting from 63 is 17015 -/
theorem sum_103_odd_numbers_from_63 :
  sumOddNumbers 63 103 = 17015 := by
  sorry

end NUMINAMATH_CALUDE_sum_103_odd_numbers_from_63_l858_85857


namespace NUMINAMATH_CALUDE_zeros_imply_a_range_l858_85805

theorem zeros_imply_a_range (a : ℝ) : 
  (∃ x y, x ∈ (Set.Ioo 0 1) ∧ y ∈ (Set.Ioo 1 2) ∧ 
    x^2 - 2*a*x + 1 = 0 ∧ y^2 - 2*a*y + 1 = 0) → 
  a ∈ (Set.Ioo 1 (5/4)) := by
sorry

end NUMINAMATH_CALUDE_zeros_imply_a_range_l858_85805


namespace NUMINAMATH_CALUDE_line_intercepts_sum_l858_85848

/-- Given a line with equation y - 3 = -3(x - 6), 
    prove that the sum of its x-intercept and y-intercept is 28 -/
theorem line_intercepts_sum (x y : ℝ) : 
  (y - 3 = -3 * (x - 6)) → 
  (∃ x_int y_int : ℝ, 
    (y_int - 3 = -3 * (x_int - 6)) ∧ 
    (0 - 3 = -3 * (x_int - 6)) ∧
    (y_int - 3 = -3 * (0 - 6)) ∧
    (x_int + y_int = 28)) :=
by sorry

end NUMINAMATH_CALUDE_line_intercepts_sum_l858_85848


namespace NUMINAMATH_CALUDE_lateral_surface_area_equilateral_prism_l858_85825

/-- The lateral surface area of a prism with an equilateral triangular base -/
theorem lateral_surface_area_equilateral_prism (a : ℝ) (h : a > 0) :
  let base_side := a
  let base_center_to_vertex := a * Real.sqrt 3 / 3
  let edge_angle := 60 * π / 180
  let edge_length := 2 * base_center_to_vertex / Real.cos edge_angle
  let lateral_perimeter := a + a * Real.sqrt 13 / 2
  lateral_perimeter * edge_length = a^2 * Real.sqrt 3 * (Real.sqrt 13 + 2) / 3 :=
by sorry

end NUMINAMATH_CALUDE_lateral_surface_area_equilateral_prism_l858_85825


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_implies_function_order_l858_85877

def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_inequality_solution_implies_function_order
  (a b c : ℝ)
  (h1 : ∀ x, (x < -2 ∨ x > 4) ↔ a * x^2 + b * x + c < 0) :
  f a b c 5 < f a b c 2 ∧ f a b c 2 < f a b c 1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_implies_function_order_l858_85877

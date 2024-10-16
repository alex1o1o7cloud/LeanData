import Mathlib

namespace NUMINAMATH_CALUDE_min_value_of_sum_of_squares_l601_60157

theorem min_value_of_sum_of_squares (a b : ℝ) : 
  (∃ x : ℝ, x^4 + a*x^3 + 2*x^2 + b*x + 1 = 0) → a^2 + b^2 ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_sum_of_squares_l601_60157


namespace NUMINAMATH_CALUDE_students_interested_in_both_l601_60127

theorem students_interested_in_both (total : ℕ) (music : ℕ) (sports : ℕ) (neither : ℕ) :
  total = 55 →
  music = 35 →
  sports = 45 →
  neither = 4 →
  ∃ both : ℕ, both = 29 ∧ total = music + sports - both + neither :=
by sorry

end NUMINAMATH_CALUDE_students_interested_in_both_l601_60127


namespace NUMINAMATH_CALUDE_complex_expression_evaluation_l601_60197

theorem complex_expression_evaluation : 
  (39/7) / ((8.4 * (6/7) * (6 - ((2.3 + 5/6.25) * 7) / (8 * 0.0125 + 6.9))) - 20.384/1.3) = 15/14 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_evaluation_l601_60197


namespace NUMINAMATH_CALUDE_curve_is_two_rays_and_circle_l601_60150

-- Define the curve equation
def curve_equation (x y : ℝ) : Prop :=
  x * Real.sqrt (2 * x^2 + 2 * y^2 - 3) = 0

-- Define a circle
def is_circle (x y : ℝ) : Prop :=
  x^2 + y^2 = 3/2

-- Define two rays
def is_two_rays (x y : ℝ) : Prop :=
  x = 0 ∧ (y ≥ Real.sqrt 6 / 2 ∨ y ≤ -Real.sqrt 6 / 2)

-- Theorem stating that the curve equation represents two rays and a circle
theorem curve_is_two_rays_and_circle :
  ∀ x y : ℝ, curve_equation x y ↔ (is_circle x y ∨ is_two_rays x y) :=
sorry

end NUMINAMATH_CALUDE_curve_is_two_rays_and_circle_l601_60150


namespace NUMINAMATH_CALUDE_difference_sum_of_T_l601_60109

def T : Finset ℕ := Finset.range 11

def difference_sum (s : Finset ℕ) : ℕ :=
  s.sum (fun i => s.sum (fun j => if i > j then (3^i - 3^j) else 0))

theorem difference_sum_of_T : difference_sum T = 793168 := by
  sorry

end NUMINAMATH_CALUDE_difference_sum_of_T_l601_60109


namespace NUMINAMATH_CALUDE_conditional_probability_B_given_A_l601_60102

-- Define the set of numbers
def S : Finset ℕ := Finset.range 7

-- Define a type for a selection of 5 numbers
def Selection := {s : Finset ℕ // s.card = 5 ∧ s ⊆ S}

-- Define the median of a selection
def median (sel : Selection) : ℚ :=
  sorry

-- Define the average of a selection
def average (sel : Selection) : ℚ :=
  sorry

-- Define event A: median is 4
def eventA (sel : Selection) : Prop :=
  median sel = 4

-- Define event B: average is 4
def eventB (sel : Selection) : Prop :=
  average sel = 4

-- Define the probability measure
noncomputable def P : Set Selection → ℝ :=
  sorry

-- State the theorem
theorem conditional_probability_B_given_A :
  P {sel : Selection | eventB sel ∧ eventA sel} / P {sel : Selection | eventA sel} = 1/3 :=
sorry

end NUMINAMATH_CALUDE_conditional_probability_B_given_A_l601_60102


namespace NUMINAMATH_CALUDE_rectangle_dimension_change_l601_60105

/-- Theorem: Rectangle Dimension Change
  Given a rectangle with length L and breadth B,
  if the length is increased by 10% and the area is increased by 37.5%,
  then the breadth must be increased by 25%.
-/
theorem rectangle_dimension_change
  (L B : ℝ)  -- Original length and breadth
  (L' B' : ℝ) -- New length and breadth
  (h1 : L' = 1.1 * L)  -- Length increased by 10%
  (h2 : L' * B' = 1.375 * (L * B))  -- Area increased by 37.5%
  : B' = 1.25 * B := by
  sorry

end NUMINAMATH_CALUDE_rectangle_dimension_change_l601_60105


namespace NUMINAMATH_CALUDE_consumption_ranking_l601_60118

-- Define the regions
inductive Region
| West
| NonWest
| Russia

-- Define the consumption function
def consumption : Region → ℝ
| Region.West => 21428
| Region.NonWest => 26848.55
| Region.Russia => 302790.13

-- Define the ranking function
def ranking (r : Region) : ℕ :=
  match r with
  | Region.West => 3
  | Region.NonWest => 2
  | Region.Russia => 1

-- Theorem statement
theorem consumption_ranking :
  ∀ r1 r2 : Region, ranking r1 < ranking r2 ↔ consumption r1 > consumption r2 :=
by sorry

end NUMINAMATH_CALUDE_consumption_ranking_l601_60118


namespace NUMINAMATH_CALUDE_odd_even_digit_difference_l601_60179

/-- The upper bound of the range of integers we're considering -/
def upper_bound : ℕ := 8 * 10^20

/-- Counts the number of integers up to n (inclusive) that contain only odd digits -/
def count_odd_digits (n : ℕ) : ℕ := sorry

/-- Counts the number of integers up to n (inclusive) that contain only even digits -/
def count_even_digits (n : ℕ) : ℕ := sorry

/-- The main theorem stating the difference between odd-digit-only and even-digit-only numbers -/
theorem odd_even_digit_difference :
  count_odd_digits upper_bound - count_even_digits upper_bound = (5^21 - 1) / 4 := by sorry

end NUMINAMATH_CALUDE_odd_even_digit_difference_l601_60179


namespace NUMINAMATH_CALUDE_intersection_A_B_l601_60103

def A : Set ℤ := {-3, -1, 2, 6}
def B : Set ℤ := {x | x > 0}

theorem intersection_A_B : A ∩ B = {2, 6} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l601_60103


namespace NUMINAMATH_CALUDE_smallest_common_multiple_lcm_14_10_smallest_number_of_students_l601_60171

theorem smallest_common_multiple (n : ℕ) : n > 0 ∧ 14 ∣ n ∧ 10 ∣ n → n ≥ 70 := by
  sorry

theorem lcm_14_10 : Nat.lcm 14 10 = 70 := by
  sorry

theorem smallest_number_of_students : ∃ (n : ℕ), n > 0 ∧ 14 ∣ n ∧ 10 ∣ n ∧ ∀ (m : ℕ), (m > 0 ∧ 14 ∣ m ∧ 10 ∣ m) → n ≤ m := by
  sorry

end NUMINAMATH_CALUDE_smallest_common_multiple_lcm_14_10_smallest_number_of_students_l601_60171


namespace NUMINAMATH_CALUDE_total_cards_proof_l601_60158

/-- The number of people who have baseball cards -/
def num_people : ℕ := 4

/-- The number of baseball cards each person has -/
def cards_per_person : ℕ := 6

/-- The total number of baseball cards -/
def total_cards : ℕ := num_people * cards_per_person

theorem total_cards_proof : total_cards = 24 := by
  sorry

end NUMINAMATH_CALUDE_total_cards_proof_l601_60158


namespace NUMINAMATH_CALUDE_oblique_projection_preserves_parallelogram_l601_60163

/-- Represents a parallelogram in a 2D plane -/
structure Parallelogram where
  -- Add necessary fields to define a parallelogram
  -- This is a simplified representation
  dummy : Unit

/-- Represents the oblique projection method -/
def obliqueProjection (p : Parallelogram) : Parallelogram :=
  sorry

/-- Theorem stating that the oblique projection of a parallelogram is always a parallelogram -/
theorem oblique_projection_preserves_parallelogram (p : Parallelogram) :
  ∃ (q : Parallelogram), obliqueProjection p = q :=
sorry

end NUMINAMATH_CALUDE_oblique_projection_preserves_parallelogram_l601_60163


namespace NUMINAMATH_CALUDE_final_daisy_count_l601_60156

/-- Represents the number of flowers in Laura's garden -/
structure GardenFlowers where
  daisies : ℕ
  tulips : ℕ

/-- Represents the ratio of daisies to tulips -/
structure FlowerRatio where
  daisy : ℕ
  tulip : ℕ

/-- Theorem stating the final number of daisies after adding tulips while maintaining the ratio -/
theorem final_daisy_count 
  (initial : GardenFlowers) 
  (ratio : FlowerRatio) 
  (added_tulips : ℕ) : 
  (ratio.daisy : ℚ) / (ratio.tulip : ℚ) = (initial.daisies : ℚ) / (initial.tulips : ℚ) →
  initial.tulips = 32 →
  added_tulips = 24 →
  ratio.daisy = 3 →
  ratio.tulip = 4 →
  let final_tulips := initial.tulips + added_tulips
  let final_daisies := (ratio.daisy : ℚ) / (ratio.tulip : ℚ) * final_tulips
  final_daisies = 42 := by
  sorry


end NUMINAMATH_CALUDE_final_daisy_count_l601_60156


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l601_60120

theorem quadratic_inequality_solution_set (m : ℝ) (h : m * (m - 1) < 0) :
  {x : ℝ | x^2 - (m + 1/m) * x + 1 < 0} = {x : ℝ | m < x ∧ x < 1/m} := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l601_60120


namespace NUMINAMATH_CALUDE_disjoint_subset_count_is_three_pow_l601_60180

/-- The number of ways to select two disjoint subsets from a set with n elements -/
def disjointSubsetCount (n : ℕ) : ℕ := 3^n

/-- Theorem: The number of ways to select two disjoint subsets from a set with n elements is 3^n -/
theorem disjoint_subset_count_is_three_pow (n : ℕ) : disjointSubsetCount n = 3^n := by
  sorry

end NUMINAMATH_CALUDE_disjoint_subset_count_is_three_pow_l601_60180


namespace NUMINAMATH_CALUDE_constant_function_proof_l601_60133

theorem constant_function_proof (f : ℤ × ℤ → ℝ) 
  (h1 : ∀ (x y : ℤ), 0 ≤ f (x, y) ∧ f (x, y) ≤ 1)
  (h2 : ∀ (x y : ℤ), f (x, y) = (f (x - 1, y) + f (x, y - 1)) / 2) :
  ∃ (c : ℝ), ∀ (x y : ℤ), f (x, y) = c :=
sorry

end NUMINAMATH_CALUDE_constant_function_proof_l601_60133


namespace NUMINAMATH_CALUDE_reciprocal_sum_pairs_l601_60170

theorem reciprocal_sum_pairs : 
  (Finset.filter 
    (fun p : ℕ × ℕ => 
      p.1 > 0 ∧ p.2 > 0 ∧ (1 : ℚ) / p.1 + (1 : ℚ) / p.2 = (1 : ℚ) / 6)
    (Finset.product (Finset.range 100) (Finset.range 100))).card = 9 :=
by sorry

end NUMINAMATH_CALUDE_reciprocal_sum_pairs_l601_60170


namespace NUMINAMATH_CALUDE_bakery_pie_division_l601_60188

theorem bakery_pie_division (pie_leftover : ℚ) (num_people : ℕ) : 
  pie_leftover = 8 / 9 → num_people = 3 → 
  pie_leftover / num_people = 8 / 27 := by
  sorry

end NUMINAMATH_CALUDE_bakery_pie_division_l601_60188


namespace NUMINAMATH_CALUDE_james_joe_age_ratio_l601_60112

theorem james_joe_age_ratio : 
  ∀ (joe_age james_age : ℕ),
    joe_age = 22 →
    joe_age = james_age + 10 →
    ∃ (k : ℕ), 2 * (joe_age + 8) = k * (james_age + 8) →
    (james_age + 8) / (joe_age + 8) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_james_joe_age_ratio_l601_60112


namespace NUMINAMATH_CALUDE_intersecting_circles_radius_range_l601_60129

/-- Given two intersecting circles, prove that the radius of the first circle falls within a specific range -/
theorem intersecting_circles_radius_range :
  ∀ r : ℝ,
  r > 0 →
  (∃ x y : ℝ, x^2 + y^2 = r^2 ∧ (x+3)^2 + (y-4)^2 = 36) →
  1 < r ∧ r < 11 := by
sorry

end NUMINAMATH_CALUDE_intersecting_circles_radius_range_l601_60129


namespace NUMINAMATH_CALUDE_book_original_price_l601_60113

/-- Proves that given a book sold for Rs 60 with a 20% profit rate, the original price of the book was Rs 50. -/
theorem book_original_price (selling_price : ℝ) (profit_rate : ℝ) : 
  selling_price = 60 → profit_rate = 0.20 → 
  ∃ (original_price : ℝ), original_price = 50 ∧ selling_price = original_price * (1 + profit_rate) :=
by
  sorry

end NUMINAMATH_CALUDE_book_original_price_l601_60113


namespace NUMINAMATH_CALUDE_cousin_name_probability_l601_60195

theorem cousin_name_probability :
  let total_cards : ℕ := 12
  let adrian_cards : ℕ := 7
  let bella_cards : ℕ := 5
  let prob_one_from_each : ℚ := 
    (adrian_cards / total_cards) * (bella_cards / (total_cards - 1)) +
    (bella_cards / total_cards) * (adrian_cards / (total_cards - 1))
  prob_one_from_each = 35 / 66 := by
sorry

end NUMINAMATH_CALUDE_cousin_name_probability_l601_60195


namespace NUMINAMATH_CALUDE_polynomial_factorization_l601_60101

theorem polynomial_factorization (m n : ℝ) :
  (m + n)^2 - 10*(m + n) + 25 = (m + n - 5)^2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l601_60101


namespace NUMINAMATH_CALUDE_pams_bank_account_l601_60137

def initial_balance : ℝ := 400
def withdrawal : ℝ := 250
def current_balance : ℝ := 950

theorem pams_bank_account :
  initial_balance * 3 - withdrawal = current_balance :=
by sorry

end NUMINAMATH_CALUDE_pams_bank_account_l601_60137


namespace NUMINAMATH_CALUDE_sqrt5_diamond_sqrt5_equals_20_l601_60116

-- Define the ¤ operation
def diamond (x y : ℝ) : ℝ := (x + y)^2 - (x - y)^2

-- State the theorem
theorem sqrt5_diamond_sqrt5_equals_20 : diamond (Real.sqrt 5) (Real.sqrt 5) = 20 := by
  sorry

end NUMINAMATH_CALUDE_sqrt5_diamond_sqrt5_equals_20_l601_60116


namespace NUMINAMATH_CALUDE_initial_concentrated_kola_percentage_l601_60135

/-- Proves that the initial percentage of concentrated kola in a solution is 9% -/
theorem initial_concentrated_kola_percentage
  (initial_volume : ℝ)
  (initial_water_percentage : ℝ)
  (added_sugar : ℝ)
  (added_water : ℝ)
  (added_concentrated_kola : ℝ)
  (final_sugar_percentage : ℝ)
  (h1 : initial_volume = 340)
  (h2 : initial_water_percentage = 64)
  (h3 : added_sugar = 3.2)
  (h4 : added_water = 8)
  (h5 : added_concentrated_kola = 6.8)
  (h6 : final_sugar_percentage = 26.536312849162012)
  (h7 : (((100 - initial_water_percentage - 9) * initial_volume / 100 + added_sugar) /
         (initial_volume + added_sugar + added_water + added_concentrated_kola)) * 100 = final_sugar_percentage) :
  9 = 100 - initial_water_percentage - ((initial_volume * initial_water_percentage / 100 + added_water) /
    (initial_volume + added_sugar + added_water + added_concentrated_kola) * 100) :=
by sorry

end NUMINAMATH_CALUDE_initial_concentrated_kola_percentage_l601_60135


namespace NUMINAMATH_CALUDE_angle_DAE_in_special_triangle_l601_60106

-- Define the triangle ABC
def Triangle (A B C : Point) : Prop := sorry

-- Define the angle measure in degrees
def AngleMeasure (A B C : Point) : ℝ := sorry

-- Define the foot of the perpendicular
def PerpendicularFoot (A D : Point) (B C : Point) : Prop := sorry

-- Define the center of the circumscribed circle
def CircumcenterOfTriangle (O A B C : Point) : Prop := sorry

-- Define the diameter of a circle
def DiameterOfCircle (A E O : Point) : Prop := sorry

theorem angle_DAE_in_special_triangle 
  (A B C D E O : Point) 
  (triangle_ABC : Triangle A B C)
  (angle_ACB : AngleMeasure A C B = 40)
  (angle_CBA : AngleMeasure C B A = 60)
  (D_perpendicular : PerpendicularFoot A D B C)
  (O_circumcenter : CircumcenterOfTriangle O A B C)
  (AE_diameter : DiameterOfCircle A E O) :
  AngleMeasure D A E = 20 := by
sorry

end NUMINAMATH_CALUDE_angle_DAE_in_special_triangle_l601_60106


namespace NUMINAMATH_CALUDE_geometric_sequence_a5_l601_60138

/-- A geometric sequence {a_n} where a_3 * a_7 = 64 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_a5 (a : ℕ → ℝ) (h : geometric_sequence a) (h1 : a 3 * a 7 = 64) :
  a 5 = 8 ∨ a 5 = -8 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a5_l601_60138


namespace NUMINAMATH_CALUDE_card_probability_l601_60142

theorem card_probability (diamonds hearts : ℕ) (a : ℕ) :
  diamonds = 3 →
  hearts = 2 →
  (a : ℚ) / (a + diamonds + hearts) = 1 / 2 →
  a = 5 := by
  sorry

end NUMINAMATH_CALUDE_card_probability_l601_60142


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l601_60136

theorem solve_exponential_equation :
  ∃ y : ℝ, (9 : ℝ) ^ y = (3 : ℝ) ^ 12 ∧ y = 6 := by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l601_60136


namespace NUMINAMATH_CALUDE_youngest_son_cookies_value_l601_60154

/-- The number of cookies in a box -/
def total_cookies : ℕ := 54

/-- The number of days the box lasts -/
def days : ℕ := 9

/-- The number of cookies the oldest son gets each day -/
def oldest_son_cookies : ℕ := 4

/-- The number of cookies the youngest son gets each day -/
def youngest_son_cookies : ℕ := (total_cookies - (oldest_son_cookies * days)) / days

theorem youngest_son_cookies_value : youngest_son_cookies = 2 := by
  sorry

end NUMINAMATH_CALUDE_youngest_son_cookies_value_l601_60154


namespace NUMINAMATH_CALUDE_solve_pq_system_l601_60149

theorem solve_pq_system (p q : ℝ) (hp : p > 1) (hq : q > 1) 
  (h1 : 1/p + 1/q = 1) (h2 : p * q = 9) : q = (9 + 3 * Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_pq_system_l601_60149


namespace NUMINAMATH_CALUDE_ghost_mansion_paths_8_2_l601_60184

/-- Calculates the number of ways a ghost can enter and exit a mansion with different windows --/
def ghost_mansion_paths (total_windows : ℕ) (locked_windows : ℕ) : ℕ :=
  let usable_windows := total_windows - locked_windows
  usable_windows * (usable_windows - 1)

/-- Theorem: The number of ways for a ghost to enter and exit a mansion with 8 windows, 2 of which are locked, is 30 --/
theorem ghost_mansion_paths_8_2 :
  ghost_mansion_paths 8 2 = 30 := by
  sorry

#eval ghost_mansion_paths 8 2

end NUMINAMATH_CALUDE_ghost_mansion_paths_8_2_l601_60184


namespace NUMINAMATH_CALUDE_converse_proposition_false_l601_60186

/-- Vectors a and b are collinear -/
def are_collinear (a b : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), a.1 * b.2 = k * a.2 * b.1

/-- The converse of the proposition "If x = 1, then the vectors (-2x, 1) and (-2, x) are collinear" is false -/
theorem converse_proposition_false : ¬ ∀ x : ℝ, 
  are_collinear (-2*x, 1) (-2, x) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_converse_proposition_false_l601_60186


namespace NUMINAMATH_CALUDE_toy_set_pricing_l601_60110

/-- Represents the cost and sales data for Asian Games mascot plush toy sets -/
structure ToySetData where
  cost_price : ℝ
  batch1_quantity : ℕ
  batch1_price : ℝ
  batch2_quantity : ℕ
  batch2_price : ℝ
  total_profit : ℝ
  batch3_quantity : ℕ
  batch3_initial_price : ℝ
  day1_sales : ℕ
  day2_sales : ℕ
  sales_increase_per_reduction : ℝ
  reduction_step : ℝ
  day3_profit : ℝ

/-- Theorem stating the cost price and required price reduction for the toy sets -/
theorem toy_set_pricing (data : ToySetData) :
  data.cost_price = 60 ∧
  (∃ (price_reduction : ℝ), price_reduction = 10 ∧
    (data.batch1_quantity * (data.batch1_price - data.cost_price) +
     data.batch2_quantity * (data.batch2_price - data.cost_price) = data.total_profit) ∧
    (data.day1_sales * (data.batch3_initial_price - data.cost_price) +
     data.day2_sales * (data.batch3_initial_price - data.cost_price) +
     (data.day2_sales + price_reduction / data.reduction_step * data.sales_increase_per_reduction) *
       (data.batch3_initial_price - price_reduction - data.cost_price) = data.day3_profit)) :=
by sorry

end NUMINAMATH_CALUDE_toy_set_pricing_l601_60110


namespace NUMINAMATH_CALUDE_prob_nine_successes_possible_l601_60189

/-- The number of trials -/
def n : ℕ := 10

/-- The success probability -/
def p : ℝ := 0.9

/-- The binomial probability mass function -/
def binomial_pmf (k : ℕ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

/-- Theorem stating that the probability of exactly 9 successes is between 0 and 1 -/
theorem prob_nine_successes_possible :
  0 < binomial_pmf 9 ∧ binomial_pmf 9 < 1 := by
  sorry

end NUMINAMATH_CALUDE_prob_nine_successes_possible_l601_60189


namespace NUMINAMATH_CALUDE_xyz_sum_sqrt_l601_60134

theorem xyz_sum_sqrt (x y z : ℝ) 
  (h1 : y + z = 16) 
  (h2 : z + x = 18) 
  (h3 : x + y = 20) : 
  Real.sqrt (x * y * z * (x + y + z)) = Real.sqrt 18711 := by
  sorry

end NUMINAMATH_CALUDE_xyz_sum_sqrt_l601_60134


namespace NUMINAMATH_CALUDE_integer_root_cubic_equation_l601_60132

theorem integer_root_cubic_equation :
  ∀ x : ℤ, x^3 - 4*x^2 - 11*x + 24 = 0 ↔ x = -3 := by
  sorry

end NUMINAMATH_CALUDE_integer_root_cubic_equation_l601_60132


namespace NUMINAMATH_CALUDE_largest_increase_2006_2007_l601_60173

def students : Fin 5 → ℕ
  | 0 => 80  -- 2003
  | 1 => 88  -- 2004
  | 2 => 94  -- 2005
  | 3 => 106 -- 2006
  | 4 => 130 -- 2007

def percentageIncrease (a b : ℕ) : ℚ :=
  (b - a : ℚ) / a * 100

def largestIncreaseYears : Fin 4 := 3

theorem largest_increase_2006_2007 :
  ∀ i : Fin 4, percentageIncrease (students i) (students (i + 1)) ≤ 
    percentageIncrease (students largestIncreaseYears) (students (largestIncreaseYears + 1)) :=
by sorry

end NUMINAMATH_CALUDE_largest_increase_2006_2007_l601_60173


namespace NUMINAMATH_CALUDE_determine_english_marks_l601_60194

/-- Represents a student's marks in 5 subjects -/
structure StudentMarks where
  english : ℕ
  mathematics : ℕ
  physics : ℕ
  chemistry : ℕ
  biology : ℕ

/-- Calculate the average marks -/
def average (marks : StudentMarks) : ℚ :=
  (marks.english + marks.mathematics + marks.physics + marks.chemistry + marks.biology) / 5

/-- Theorem: Given 4 subject marks and the average, the 5th subject mark is uniquely determined -/
theorem determine_english_marks (marks : StudentMarks) (avg : ℚ) 
    (h1 : marks.mathematics = 65)
    (h2 : marks.physics = 82)
    (h3 : marks.chemistry = 67)
    (h4 : marks.biology = 85)
    (h5 : average marks = avg)
    (h6 : avg = 75) :
  marks.english = 76 := by
  sorry


end NUMINAMATH_CALUDE_determine_english_marks_l601_60194


namespace NUMINAMATH_CALUDE_not_necessarily_right_triangle_l601_60177

theorem not_necessarily_right_triangle (A B C : ℝ) : 
  A + B + C = 180 → A = B → A = 2 * C → ¬ (A = 90 ∨ B = 90 ∨ C = 90) :=
by sorry

end NUMINAMATH_CALUDE_not_necessarily_right_triangle_l601_60177


namespace NUMINAMATH_CALUDE_units_digit_problem_l601_60147

theorem units_digit_problem (n : ℤ) : n = (30 * 31 * 32 * 33 * 34 * 35) / 2500 → n % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_problem_l601_60147


namespace NUMINAMATH_CALUDE_isosceles_triangle_with_60_degree_angle_l601_60196

theorem isosceles_triangle_with_60_degree_angle (α β : ℝ) : 
  α > 0 ∧ β > 0 ∧ -- Angles are positive
  α + β + β = 180 ∧ -- Sum of angles in a triangle
  α = 60 ∧ -- One angle is 60°
  β = β -- Triangle is isosceles with two equal angles
  → α = 60 ∧ β = 60 := by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_with_60_degree_angle_l601_60196


namespace NUMINAMATH_CALUDE_correct_matching_probability_l601_60169

/-- The number of Earthly Branches and zodiac signs -/
def n : ℕ := 12

/-- The number of cards selected from each color -/
def k : ℕ := 3

/-- The probability of correctly matching the selected cards -/
def matching_probability : ℚ := 1 / (n.choose k)

/-- Theorem stating the probability of correctly matching the selected cards -/
theorem correct_matching_probability :
  matching_probability = 1 / 220 :=
by sorry

end NUMINAMATH_CALUDE_correct_matching_probability_l601_60169


namespace NUMINAMATH_CALUDE_equal_balance_after_10_days_l601_60117

/-- Carol's initial borrowing in clams -/
def carol_initial : ℝ := 200

/-- Emily's initial borrowing in clams -/
def emily_initial : ℝ := 250

/-- Carol's daily interest rate -/
def carol_rate : ℝ := 0.15

/-- Emily's daily interest rate -/
def emily_rate : ℝ := 0.10

/-- Number of days after which Carol and Emily owe the same amount -/
def days_equal : ℕ := 10

/-- Carol's balance after t days -/
def carol_balance (t : ℝ) : ℝ := carol_initial * (1 + carol_rate * t)

/-- Emily's balance after t days -/
def emily_balance (t : ℝ) : ℝ := emily_initial * (1 + emily_rate * t)

theorem equal_balance_after_10_days :
  carol_balance days_equal = emily_balance days_equal :=
by sorry

end NUMINAMATH_CALUDE_equal_balance_after_10_days_l601_60117


namespace NUMINAMATH_CALUDE_jayas_rank_from_bottom_l601_60153

/-- Given a class of students, calculate the rank from the bottom based on the rank from the top. -/
def rankFromBottom (totalStudents : ℕ) (rankFromTop : ℕ) : ℕ :=
  totalStudents - rankFromTop + 1

/-- Jaya's rank from the bottom in a class of 53 students where she ranks 5th from the top is 50th. -/
theorem jayas_rank_from_bottom :
  rankFromBottom 53 5 = 50 := by
  sorry

end NUMINAMATH_CALUDE_jayas_rank_from_bottom_l601_60153


namespace NUMINAMATH_CALUDE_initial_overs_calculation_l601_60185

theorem initial_overs_calculation (target : ℝ) (initial_rate : ℝ) (remaining_rate : ℝ) 
  (remaining_overs : ℝ) (h1 : target = 250) (h2 : initial_rate = 4.2) 
  (h3 : remaining_rate = 5.533333333333333) (h4 : remaining_overs = 30) :
  ∃ x : ℝ, x = 20 ∧ initial_rate * x + remaining_rate * remaining_overs = target :=
by
  sorry

end NUMINAMATH_CALUDE_initial_overs_calculation_l601_60185


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l601_60126

theorem pure_imaginary_condition (x : ℝ) : 
  (x^2 - x : ℂ) + (x - 1 : ℂ) * Complex.I = Complex.I * (y : ℝ) → x = 0 :=
by sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l601_60126


namespace NUMINAMATH_CALUDE_normal_vector_for_line_with_angle_pi_div_3_l601_60130

/-- A line in 2D space -/
structure Line2D where
  angle : Real

/-- A vector in 2D space -/
structure Vector2D where
  x : Real
  y : Real

/-- Check if a vector is normal to a line -/
def isNormalVector (l : Line2D) (v : Vector2D) : Prop :=
  v.x * Real.cos l.angle + v.y * Real.sin l.angle = 0

/-- Theorem: (1, -√3/3) is a normal vector of a line with inclination angle π/3 -/
theorem normal_vector_for_line_with_angle_pi_div_3 :
  let l : Line2D := { angle := π / 3 }
  let v : Vector2D := { x := 1, y := -Real.sqrt 3 / 3 }
  isNormalVector l v := by
  sorry

end NUMINAMATH_CALUDE_normal_vector_for_line_with_angle_pi_div_3_l601_60130


namespace NUMINAMATH_CALUDE_set_A_enumeration_l601_60193

def A : Set (ℤ × ℤ) := {p | p.1^2 = p.2 + 1 ∧ |p.1| < 2}

theorem set_A_enumeration : A = {(-1, 0), (0, -1), (1, 0)} := by sorry

end NUMINAMATH_CALUDE_set_A_enumeration_l601_60193


namespace NUMINAMATH_CALUDE_mirror_reflection_of_16_00_l601_60111

/-- Represents a clock time with hours and minutes -/
structure ClockTime where
  hours : Nat
  minutes : Nat
  h_valid : hours < 12
  m_valid : minutes < 60

/-- Represents the reflection of a clock time in a mirror -/
def mirror_reflect (t : ClockTime) : ClockTime :=
  { hours := (12 - t.hours) % 12,
    minutes := t.minutes,
    h_valid := by sorry,
    m_valid := t.m_valid }

/-- The theorem stating that 16:00 reflects to approximately 8:00 in a mirror -/
theorem mirror_reflection_of_16_00 :
  let t : ClockTime := ⟨4, 0, by sorry, by sorry⟩
  let reflected : ClockTime := mirror_reflect t
  reflected.hours = 8 ∧ reflected.minutes = 0 :=
by sorry

end NUMINAMATH_CALUDE_mirror_reflection_of_16_00_l601_60111


namespace NUMINAMATH_CALUDE_staircase_carpet_cost_l601_60123

/-- Represents the dimensions and cost parameters of a staircase -/
structure Staircase where
  num_steps : ℕ
  step_height : ℝ
  step_depth : ℝ
  width : ℝ
  carpet_cost_per_sqm : ℝ

/-- Calculates the cost of carpeting a staircase -/
def carpet_cost (s : Staircase) : ℝ :=
  let total_height := s.num_steps * s.step_height
  let total_depth := s.num_steps * s.step_depth
  let combined_length := total_height + total_depth
  let total_area := combined_length * s.width
  total_area * s.carpet_cost_per_sqm

/-- Theorem: The cost of carpeting the given staircase is 1512 yuan -/
theorem staircase_carpet_cost :
  let s : Staircase := {
    num_steps := 15,
    step_height := 0.16,
    step_depth := 0.26,
    width := 3,
    carpet_cost_per_sqm := 80
  }
  carpet_cost s = 1512 := by
  sorry

end NUMINAMATH_CALUDE_staircase_carpet_cost_l601_60123


namespace NUMINAMATH_CALUDE_bus_interval_theorem_l601_60176

/-- The interval between buses on a circular route -/
def interval (num_buses : ℕ) (total_time : ℕ) : ℕ :=
  total_time / num_buses

/-- The theorem stating the relationship between intervals for 2 and 3 buses -/
theorem bus_interval_theorem (initial_interval : ℕ) :
  initial_interval = 21 →
  interval 2 (2 * initial_interval) = 21 →
  interval 3 (2 * initial_interval) = 14 :=
by
  sorry

#check bus_interval_theorem

end NUMINAMATH_CALUDE_bus_interval_theorem_l601_60176


namespace NUMINAMATH_CALUDE_bart_burning_period_l601_60190

/-- The number of pieces of firewood Bart gets from one tree -/
def pieces_per_tree : ℕ := 75

/-- The number of logs Bart burns per day -/
def logs_per_day : ℕ := 5

/-- The number of trees Bart cuts down for the period -/
def trees_cut : ℕ := 8

/-- The period (in days) that Bart burns the logs -/
def burning_period : ℕ := (pieces_per_tree * trees_cut) / logs_per_day

theorem bart_burning_period :
  burning_period = 120 := by sorry

end NUMINAMATH_CALUDE_bart_burning_period_l601_60190


namespace NUMINAMATH_CALUDE_john_drinks_two_cups_per_day_l601_60124

-- Define the constants
def gallon_to_ounce : ℚ := 128
def cup_to_ounce : ℚ := 8
def days_between_purchases : ℚ := 4
def gallons_per_purchase : ℚ := 1/2

-- Define the function to calculate cups per day
def cups_per_day : ℚ :=
  (gallons_per_purchase * gallon_to_ounce) / (days_between_purchases * cup_to_ounce)

-- Theorem statement
theorem john_drinks_two_cups_per_day :
  cups_per_day = 2 := by sorry

end NUMINAMATH_CALUDE_john_drinks_two_cups_per_day_l601_60124


namespace NUMINAMATH_CALUDE_negation_of_universal_is_existential_l601_60152

-- Define the universe of discourse
variable (U : Type)

-- Define the predicates
variable (Boy : U → Prop)
variable (LovesFootball : U → Prop)

-- State the theorem
theorem negation_of_universal_is_existential :
  (¬ ∀ x, Boy x → LovesFootball x) ↔ (∃ x, Boy x ∧ ¬ LovesFootball x) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_is_existential_l601_60152


namespace NUMINAMATH_CALUDE_tricia_age_l601_60175

-- Define the ages of all people as natural numbers
def Vincent : ℕ := 22
def Rupert : ℕ := Vincent - 2
def Khloe : ℕ := Rupert - 10
def Eugene : ℕ := Khloe * 3
def Yorick : ℕ := Eugene * 2
def Selena : ℕ := Yorick - 5
def Amilia : ℕ := Selena - 3
def Tricia : ℕ := Amilia / 3

-- Theorem to prove Tricia's age
theorem tricia_age : Tricia = 17 := by
  sorry

end NUMINAMATH_CALUDE_tricia_age_l601_60175


namespace NUMINAMATH_CALUDE_isosceles_triangle_l601_60114

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)
  (angle_sum : A + B + C = π)
  (positive_angles : 0 < A ∧ 0 < B ∧ 0 < C)

-- State the theorem
theorem isosceles_triangle (t : Triangle) 
  (h : 2 * Real.sin t.A * Real.cos t.B = Real.sin t.C) : 
  t.A = t.B ∨ t.B = t.C ∨ t.C = t.A :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_l601_60114


namespace NUMINAMATH_CALUDE_mangoes_in_basket_B_mangoes_in_basket_B_is_30_l601_60159

theorem mangoes_in_basket_B (total_baskets : ℕ) (avg_fruits : ℕ) 
  (apples_A : ℕ) (peaches_C : ℕ) (pears_D : ℕ) (bananas_E : ℕ) : ℕ :=
  let total_fruits := total_baskets * avg_fruits
  let accounted_fruits := apples_A + peaches_C + pears_D + bananas_E
  total_fruits - accounted_fruits

#check mangoes_in_basket_B 5 25 15 20 25 35 = 30

theorem mangoes_in_basket_B_is_30 :
  mangoes_in_basket_B 5 25 15 20 25 35 = 30 := by
  sorry

end NUMINAMATH_CALUDE_mangoes_in_basket_B_mangoes_in_basket_B_is_30_l601_60159


namespace NUMINAMATH_CALUDE_f_g_deriv_pos_pos_l601_60172

-- Define f and g as functions from ℝ to ℝ
variable (f g : ℝ → ℝ)

-- Define the conditions
axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom g_even : ∀ x : ℝ, g (-x) = g x
axiom f_deriv_pos_neg : ∀ x : ℝ, x < 0 → deriv f x > 0
axiom g_deriv_neg_neg : ∀ x : ℝ, x < 0 → deriv g x < 0

-- State the theorem
theorem f_g_deriv_pos_pos : 
  ∀ x : ℝ, x > 0 → deriv f x > 0 ∧ deriv g x > 0 := by sorry

end NUMINAMATH_CALUDE_f_g_deriv_pos_pos_l601_60172


namespace NUMINAMATH_CALUDE_cubic_function_b_value_l601_60104

/-- A cubic function f(x) = ax³ + bx² + cx + d with specific properties -/
def cubic_function (a b c d : ℝ) : ℝ → ℝ := fun x ↦ a * x^3 + b * x^2 + c * x + d

theorem cubic_function_b_value (a b c d : ℝ) :
  (cubic_function a b c d (-1) = 0) →
  (cubic_function a b c d 1 = 0) →
  (cubic_function a b c d 0 = 2) →
  b = -2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_b_value_l601_60104


namespace NUMINAMATH_CALUDE_shaded_area_is_120_l601_60198

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle with four vertices -/
structure Rectangle where
  p : Point
  r : Point
  s : Point
  v : Point

/-- Calculates the area of a rectangle -/
def rectangleArea (rect : Rectangle) : ℝ :=
  (rect.r.x - rect.p.x) * (rect.v.y - rect.p.y)

/-- Theorem: The shaded area in the given rectangle is 120 cm² -/
theorem shaded_area_is_120 (rect : Rectangle) 
  (h1 : rect.r.x - rect.p.x = 20) -- PR = 20 cm
  (h2 : rect.v.y - rect.p.y = 12) -- PV = 12 cm
  (u : Point) (t : Point) (q : Point)
  (h3 : u.x = rect.v.x ∧ u.y ≤ rect.v.y ∧ u.y ≥ rect.s.y) -- U is on VS
  (h4 : t.x = rect.v.x ∧ t.y ≤ rect.v.y ∧ t.y ≥ rect.s.y) -- T is on VS
  (h5 : q.y = rect.p.y ∧ q.x ≥ rect.p.x ∧ q.x ≤ rect.r.x) -- Q is on PR
  : rectangleArea rect - (rect.r.x - rect.p.x) * (rect.v.y - rect.p.y) / 2 = 120 :=
sorry

end NUMINAMATH_CALUDE_shaded_area_is_120_l601_60198


namespace NUMINAMATH_CALUDE_vectors_collinear_l601_60128

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define vectors a, b, m, and n
variable (a b m n : V)

-- State the theorem
theorem vectors_collinear (h1 : m = a + b) (h2 : n = 2 • a + 2 • b) (h3 : ¬ Collinear ℝ ({0, a, b} : Set V)) :
  Collinear ℝ ({0, m, n} : Set V) := by
  sorry

end NUMINAMATH_CALUDE_vectors_collinear_l601_60128


namespace NUMINAMATH_CALUDE_hyperbola_equation_l601_60155

-- Define the hyperbola
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

-- Define the parabola
def parabola (x y : ℝ) : Prop :=
  y^2 = 4 * Real.sqrt 7 * x

-- Define the asymptote passing through (2, √3)
def asymptote_through_point (a b : ℝ) : Prop :=
  b / a = Real.sqrt 3 / 2

-- Define the focus on the directrix condition
def focus_on_directrix (c : ℝ) : Prop :=
  c = Real.sqrt 7

-- Theorem statement
theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0)
  (h3 : asymptote_through_point a b)
  (h4 : ∃ c, focus_on_directrix c ∧ a^2 + b^2 = c^2) :
  ∀ x y, hyperbola a b x y ↔ x^2 / 4 - y^2 / 3 = 1 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l601_60155


namespace NUMINAMATH_CALUDE_optimal_garden_is_best_l601_60167

/-- Represents a rectangular garden with one side against a wall --/
structure Garden where
  width : ℝ  -- Width of the garden (perpendicular to the wall)
  length : ℝ  -- Length of the garden (parallel to the wall)

/-- The wall length --/
def wall_length : ℝ := 600

/-- The fence cost per foot --/
def fence_cost_per_foot : ℝ := 6

/-- The total budget for fencing --/
def fence_budget : ℝ := 1800

/-- The minimum required area --/
def min_area : ℝ := 6000

/-- Calculate the area of the garden --/
def area (g : Garden) : ℝ := g.width * g.length

/-- Calculate the perimeter of the garden --/
def perimeter (g : Garden) : ℝ := 2 * g.width + g.length + wall_length

/-- Check if the garden satisfies the budget constraint --/
def satisfies_budget (g : Garden) : Prop :=
  (2 * g.width + g.length) * fence_cost_per_foot ≤ fence_budget

/-- Check if the garden satisfies the area constraint --/
def satisfies_area (g : Garden) : Prop :=
  area g ≥ min_area

/-- The optimal garden dimensions --/
def optimal_garden : Garden :=
  { width := 75, length := 150 }

/-- Theorem stating that the optimal garden maximizes perimeter while satisfying constraints --/
theorem optimal_garden_is_best :
  satisfies_budget optimal_garden ∧
  satisfies_area optimal_garden ∧
  ∀ g : Garden, satisfies_budget g → satisfies_area g →
    perimeter g ≤ perimeter optimal_garden :=
by sorry

end NUMINAMATH_CALUDE_optimal_garden_is_best_l601_60167


namespace NUMINAMATH_CALUDE_unique_root_of_equation_l601_60141

/-- The equation |x| - 4/x = 3|x|/x has exactly one distinct real root. -/
theorem unique_root_of_equation : ∃! x : ℝ, x ≠ 0 ∧ |x| - 4/x = 3*|x|/x := by
  sorry

end NUMINAMATH_CALUDE_unique_root_of_equation_l601_60141


namespace NUMINAMATH_CALUDE_min_value_of_complex_expression_l601_60125

theorem min_value_of_complex_expression (z : ℂ) (h : Complex.abs z = 2) :
  ∃ (min_u : ℝ), min_u = (3/2) * Real.sqrt 3 ∧
  ∀ (u : ℝ), u = Complex.abs (z^2 - z + 1) → u ≥ min_u :=
sorry

end NUMINAMATH_CALUDE_min_value_of_complex_expression_l601_60125


namespace NUMINAMATH_CALUDE_max_product_of_three_l601_60145

def S : Finset Int := {-9, -5, -3, 1, 4, 6, 8}

theorem max_product_of_three (a b c : Int) (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
  a * b * c ≤ 360 ∧ ∃ (x y z : Int), x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x * y * z = 360 :=
by sorry

end NUMINAMATH_CALUDE_max_product_of_three_l601_60145


namespace NUMINAMATH_CALUDE_count_positive_3_to_1400_l601_60160

/-- Represents the operation of flipping signs in three cells -/
def flip_operation (strip : List Bool) (i j k : Nat) : List Bool :=
  sorry

/-- Checks if a number N is positive according to the problem definition -/
def is_positive (N : Nat) : Bool :=
  sorry

/-- Counts the number of positive integers in the range [3, 1400] -/
def count_positive : Nat :=
  (List.range 1398).filter (fun n => is_positive (n + 3)) |>.length

/-- The main theorem stating the count of positive numbers -/
theorem count_positive_3_to_1400 : count_positive = 1396 :=
  sorry

end NUMINAMATH_CALUDE_count_positive_3_to_1400_l601_60160


namespace NUMINAMATH_CALUDE_toothpick_grid_15x12_l601_60140

/-- Represents a rectangular grid of toothpicks -/
structure ToothpickGrid where
  height : Nat
  width : Nat

/-- Calculates the total number of toothpicks in the grid -/
def totalToothpicks (grid : ToothpickGrid) : Nat :=
  (grid.height + 1) * grid.width + (grid.width + 1) * grid.height

/-- Calculates the number of toothpicks in the boundary of the grid -/
def boundaryToothpicks (grid : ToothpickGrid) : Nat :=
  2 * (grid.height + grid.width)

/-- Theorem stating the properties of a 15x12 toothpick grid -/
theorem toothpick_grid_15x12 :
  let grid : ToothpickGrid := ⟨15, 12⟩
  totalToothpicks grid = 387 ∧ boundaryToothpicks grid = 54 := by
  sorry


end NUMINAMATH_CALUDE_toothpick_grid_15x12_l601_60140


namespace NUMINAMATH_CALUDE_triangle_perimeter_impossibility_l601_60178

theorem triangle_perimeter_impossibility (a b x : ℝ) (h1 : a = 24) (h2 : b = 10) :
  (a + b + x = 73) → ¬(a + b > x ∧ a + x > b ∧ b + x > a) :=
by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_impossibility_l601_60178


namespace NUMINAMATH_CALUDE_harrys_book_pages_l601_60187

theorem harrys_book_pages (selenas_pages : ℕ) (harrys_pages : ℕ) : 
  selenas_pages = 400 →
  harrys_pages = selenas_pages / 2 - 20 →
  harrys_pages = 180 :=
by
  sorry

end NUMINAMATH_CALUDE_harrys_book_pages_l601_60187


namespace NUMINAMATH_CALUDE_chris_has_12_marbles_l601_60192

-- Define the number of marbles Chris and Ryan have
def chris_marbles : ℕ := sorry
def ryan_marbles : ℕ := 28

-- Define the total number of marbles in the pile
def total_marbles : ℕ := chris_marbles + ryan_marbles

-- Define the number of marbles remaining after they take their share
def remaining_marbles : ℕ := 20

-- Theorem stating that Chris has 12 marbles
theorem chris_has_12_marbles :
  chris_marbles = 12 :=
by
  sorry


end NUMINAMATH_CALUDE_chris_has_12_marbles_l601_60192


namespace NUMINAMATH_CALUDE_triangle_properties_l601_60174

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem combining all parts of the problem --/
theorem triangle_properties (t : Triangle) (p : ℝ) :
  (Real.sqrt 3 * Real.sin t.B - Real.cos t.B) * (Real.sqrt 3 * Real.sin t.C - Real.cos t.C) = 4 * Real.cos t.B * Real.cos t.C →
  t.A = π / 3 ∧
  (t.a = 2 → 0 < (1/2 * t.b * t.c * Real.sin t.A) ∧ (1/2 * t.b * t.c * Real.sin t.A) ≤ Real.sqrt 3) ∧
  (Real.sin t.B = p * Real.sin t.C → 1/2 < p ∧ p < 2) :=
by sorry


end NUMINAMATH_CALUDE_triangle_properties_l601_60174


namespace NUMINAMATH_CALUDE_fraction_meaningful_l601_60122

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = 4 / (x - 3)) ↔ x ≠ 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_meaningful_l601_60122


namespace NUMINAMATH_CALUDE_one_four_digit_perfect_square_palindrome_l601_60191

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

theorem one_four_digit_perfect_square_palindrome :
  ∃! n : ℕ, is_four_digit n ∧ is_perfect_square n ∧ is_palindrome n :=
sorry

end NUMINAMATH_CALUDE_one_four_digit_perfect_square_palindrome_l601_60191


namespace NUMINAMATH_CALUDE_arithmetic_sequence_product_l601_60107

/-- Given arithmetic sequences a and b satisfying certain conditions, prove a₁b₁ = 4 -/
theorem arithmetic_sequence_product (a b : ℕ → ℝ) : 
  (∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- a is arithmetic
  (∀ n, b (n + 1) - b n = b (n + 2) - b (n + 1)) →  -- b is arithmetic
  a 2 * b 2 = 4 →
  a 3 * b 3 = 8 →
  a 4 * b 4 = 16 →
  a 1 * b 1 = 4 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_product_l601_60107


namespace NUMINAMATH_CALUDE_bus_arrival_probabilities_l601_60148

def prob_bus_A : ℝ := 0.7
def prob_bus_B : ℝ := 0.75

theorem bus_arrival_probabilities :
  (3 * prob_bus_A^2 * (1 - prob_bus_A) = 0.441) ∧
  (1 - (1 - prob_bus_A) * (1 - prob_bus_B) = 0.925) :=
by sorry

end NUMINAMATH_CALUDE_bus_arrival_probabilities_l601_60148


namespace NUMINAMATH_CALUDE_chord_length_perpendicular_bisector_l601_60100

theorem chord_length_perpendicular_bisector (r : ℝ) (h : r = 10) : 
  ∃ (c : ℝ), c = r * Real.sqrt 3 ∧ 
  c = 2 * Real.sqrt (r^2 - (r/2)^2) :=
by sorry

end NUMINAMATH_CALUDE_chord_length_perpendicular_bisector_l601_60100


namespace NUMINAMATH_CALUDE_graph_shift_l601_60166

/-- Given a function f : ℝ → ℝ, prove that the graph of y = f(x + 2) - 1 
    is equivalent to shifting the graph of y = f(x) 2 units left and 1 unit down. -/
theorem graph_shift (f : ℝ → ℝ) (x y : ℝ) :
  y = f (x + 2) - 1 ↔ ∃ x₀ y₀ : ℝ, y₀ = f x₀ ∧ x = x₀ - 2 ∧ y = y₀ - 1 :=
sorry

end NUMINAMATH_CALUDE_graph_shift_l601_60166


namespace NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_l601_60151

/-- The surface area of a sphere circumscribing a rectangular solid -/
theorem circumscribed_sphere_surface_area (a b c : ℝ) (ha : a = 3) (hb : b = 4) (hc : c = 5) :
  let diameter_squared := a^2 + b^2 + c^2
  4 * Real.pi * (diameter_squared / 4) = 50 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_l601_60151


namespace NUMINAMATH_CALUDE_sum_geq_sqrt_three_l601_60164

theorem sum_geq_sqrt_three (a b c : ℝ) 
  (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0)
  (h : a * b + b * c + c * a = 1) : 
  a + b + c ≥ Real.sqrt 3 ∧ 
  (a + b + c = Real.sqrt 3 ↔ a = 1 / Real.sqrt 3 ∧ b = 1 / Real.sqrt 3 ∧ c = 1 / Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_sum_geq_sqrt_three_l601_60164


namespace NUMINAMATH_CALUDE_total_money_proof_l601_60121

/-- Given the money distribution among Cecil, Catherine, and Carmela, 
    prove that their total money is $2800 -/
theorem total_money_proof (cecil_money : ℕ) 
  (h1 : cecil_money = 600)
  (catherine_money : ℕ) 
  (h2 : catherine_money = 2 * cecil_money - 250)
  (carmela_money : ℕ) 
  (h3 : carmela_money = 2 * cecil_money + 50) : 
  cecil_money + catherine_money + carmela_money = 2800 := by
  sorry

end NUMINAMATH_CALUDE_total_money_proof_l601_60121


namespace NUMINAMATH_CALUDE_consecutive_numbers_with_special_properties_l601_60144

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem consecutive_numbers_with_special_properties :
  ∃ (n : ℕ), sum_of_digits n = 8 ∧ (n + 1) % 8 = 0 :=
by
  use 71
  sorry

#eval sum_of_digits 71  -- Should output 8
#eval (72 % 8)  -- Should output 0

end NUMINAMATH_CALUDE_consecutive_numbers_with_special_properties_l601_60144


namespace NUMINAMATH_CALUDE_inequality_proof_l601_60115

theorem inequality_proof (x : ℝ) :
  x > 0 →
  (x * Real.sqrt (12 - x) + Real.sqrt (12 * x - x^3) ≥ 12) ↔ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l601_60115


namespace NUMINAMATH_CALUDE_min_balls_for_target_color_l601_60119

def red_balls : ℕ := 35
def green_balls : ℕ := 25
def yellow_balls : ℕ := 22
def blue_balls : ℕ := 15
def white_balls : ℕ := 12
def black_balls : ℕ := 11

def total_balls : ℕ := red_balls + green_balls + yellow_balls + blue_balls + white_balls + black_balls

def target_color_count : ℕ := 18

theorem min_balls_for_target_color :
  ∃ (n : ℕ), n = 89 ∧
  (∀ (m : ℕ), m < n → ∃ (r g y bl w bk : ℕ),
    r + g + y + bl + w + bk = m ∧
    r ≤ red_balls ∧ g ≤ green_balls ∧ y ≤ yellow_balls ∧
    bl ≤ blue_balls ∧ w ≤ white_balls ∧ bk ≤ black_balls ∧
    r < target_color_count ∧ g < target_color_count ∧ y < target_color_count ∧
    bl < target_color_count ∧ w < target_color_count ∧ bk < target_color_count) ∧
  (∀ (r g y bl w bk : ℕ),
    r + g + y + bl + w + bk = n →
    r ≤ red_balls → g ≤ green_balls → y ≤ yellow_balls →
    bl ≤ blue_balls → w ≤ white_balls → bk ≤ black_balls →
    r ≥ target_color_count ∨ g ≥ target_color_count ∨ y ≥ target_color_count ∨
    bl ≥ target_color_count ∨ w ≥ target_color_count ∨ bk ≥ target_color_count) :=
by sorry

#check min_balls_for_target_color

end NUMINAMATH_CALUDE_min_balls_for_target_color_l601_60119


namespace NUMINAMATH_CALUDE_coprime_lcm_product_l601_60146

theorem coprime_lcm_product {a b : ℕ+} (h_coprime : Nat.Coprime a b) 
  (h_lcm_eq_prod : Nat.lcm a b = a * b) : 
  ∃ (k : ℕ+), a * b = k := by sorry

end NUMINAMATH_CALUDE_coprime_lcm_product_l601_60146


namespace NUMINAMATH_CALUDE_apple_boxes_count_l601_60199

def apples_per_crate : ℕ := 250
def number_of_crates : ℕ := 20
def rotten_apples : ℕ := 320
def apples_per_box : ℕ := 25

theorem apple_boxes_count :
  (apples_per_crate * number_of_crates - rotten_apples) / apples_per_box = 187 := by
  sorry

end NUMINAMATH_CALUDE_apple_boxes_count_l601_60199


namespace NUMINAMATH_CALUDE_min_rotations_is_twelve_l601_60139

/-- The number of elements in the letter sequence -/
def letter_sequence_length : ℕ := 6

/-- The number of elements in the digit sequence -/
def digit_sequence_length : ℕ := 4

/-- The minimum number of rotations needed for both sequences to return to their original form -/
def min_rotations : ℕ := lcm letter_sequence_length digit_sequence_length

theorem min_rotations_is_twelve : min_rotations = 12 := by
  sorry

end NUMINAMATH_CALUDE_min_rotations_is_twelve_l601_60139


namespace NUMINAMATH_CALUDE_bob_has_62_pennies_l601_60183

/-- The number of pennies Alex currently has -/
def alex_pennies : ℕ := sorry

/-- The number of pennies Bob currently has -/
def bob_pennies : ℕ := sorry

/-- If Alex gives Bob two pennies, Bob will have four times as many pennies as Alex has left -/
axiom condition1 : bob_pennies + 2 = 4 * (alex_pennies - 2)

/-- If Bob gives Alex two pennies, Bob will have three times as many pennies as Alex has -/
axiom condition2 : bob_pennies - 2 = 3 * (alex_pennies + 2)

/-- Bob currently has 62 pennies -/
theorem bob_has_62_pennies : bob_pennies = 62 := by sorry

end NUMINAMATH_CALUDE_bob_has_62_pennies_l601_60183


namespace NUMINAMATH_CALUDE_copper_price_calculation_l601_60108

/-- The price of copper per pound in cents -/
def copper_price : ℚ := 65

/-- The price of zinc per pound in cents -/
def zinc_price : ℚ := 30

/-- The weight of brass in pounds -/
def brass_weight : ℚ := 70

/-- The price of brass per pound in cents -/
def brass_price : ℚ := 45

/-- The weight of copper used in pounds -/
def copper_weight : ℚ := 30

/-- The weight of zinc used in pounds -/
def zinc_weight : ℚ := 40

theorem copper_price_calculation : 
  copper_price * copper_weight + zinc_price * zinc_weight = brass_price * brass_weight :=
sorry

end NUMINAMATH_CALUDE_copper_price_calculation_l601_60108


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l601_60165

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  increasing : ∀ n, a n < a (n + 1)
  arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  first_term : a 1 = 2
  geometric : (a 2)^2 = a 1 * a 5

/-- The common difference of the arithmetic sequence is 4 -/
theorem arithmetic_sequence_common_difference (seq : ArithmeticSequence) :
  ∃ d, (∀ n, seq.a (n + 1) - seq.a n = d) ∧ d = 4 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l601_60165


namespace NUMINAMATH_CALUDE_sixPeopleArrangements_l601_60182

/-- The number of ways to arrange n distinct objects. -/
def arrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange six people in a row with Person A and Person B adjacent. -/
def adjacentArrangements : ℕ :=
  arrangements 2 * arrangements 5

/-- The number of ways to arrange six people in a row with Person A and Person B not adjacent. -/
def nonAdjacentArrangements : ℕ :=
  arrangements 4 * arrangements 2

/-- The number of ways to arrange six people in a row with exactly two people between Person A and Person B. -/
def twoPersonsBetweenArrangements : ℕ :=
  arrangements 2 * arrangements 2 * arrangements 3

/-- The number of ways to arrange six people in a row with Person A not at the left end and Person B not at the right end. -/
def notAtEndsArrangements : ℕ :=
  arrangements 6 - 2 * arrangements 5 + arrangements 4

theorem sixPeopleArrangements :
  adjacentArrangements = 240 ∧
  nonAdjacentArrangements = 480 ∧
  twoPersonsBetweenArrangements = 144 ∧
  notAtEndsArrangements = 504 := by
  sorry

end NUMINAMATH_CALUDE_sixPeopleArrangements_l601_60182


namespace NUMINAMATH_CALUDE_greatest_common_multiple_9_15_under_110_l601_60143

theorem greatest_common_multiple_9_15_under_110 : ∃ n : ℕ, 
  (∀ m : ℕ, m < 110 ∧ 9 ∣ m ∧ 15 ∣ m → m ≤ n) ∧
  n < 110 ∧ 9 ∣ n ∧ 15 ∣ n ∧
  n = 90 := by
  sorry

end NUMINAMATH_CALUDE_greatest_common_multiple_9_15_under_110_l601_60143


namespace NUMINAMATH_CALUDE_square_of_102_l601_60181

theorem square_of_102 : 102 * 102 = 10404 := by
  sorry

end NUMINAMATH_CALUDE_square_of_102_l601_60181


namespace NUMINAMATH_CALUDE_hundredths_place_of_seven_twentieths_l601_60131

theorem hundredths_place_of_seven_twentieths (n : ℕ) : 
  (n = 5) ↔ (∃ (a b : ℕ), (7 : ℚ) / 20 = (a + n / 10 + b / 100 : ℚ) ∧ 0 ≤ b ∧ b < 10) :=
by sorry

end NUMINAMATH_CALUDE_hundredths_place_of_seven_twentieths_l601_60131


namespace NUMINAMATH_CALUDE_arithmetic_sequence_difference_l601_60168

/-- An arithmetic sequence with its sum -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum of the first n terms
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_formula : ∀ n, S n = (n : ℝ) * (a 1 + a n) / 2

/-- The common difference of an arithmetic sequence -/
def common_difference (seq : ArithmeticSequence) : ℝ :=
  seq.a 2 - seq.a 1

theorem arithmetic_sequence_difference 
  (seq : ArithmeticSequence) 
  (h1 : seq.a 5 = 8) 
  (h2 : seq.S 3 = 6) : 
  common_difference seq = 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_difference_l601_60168


namespace NUMINAMATH_CALUDE_line_circle_intersection_l601_60162

noncomputable def m : ℝ → ℝ → ℝ → ℝ := sorry

theorem line_circle_intersection (m : ℝ) :
  (∃ A B : ℝ × ℝ,
    (A.1 - m * A.2 + 1 = 0 ∧ (A.1 - 1)^2 + A.2^2 = 4) ∧
    (B.1 - m * B.2 + 1 = 0 ∧ (B.1 - 1)^2 + B.2^2 = 4) ∧
    A ≠ B) →
  (let C : ℝ × ℝ := (1, 0);
   ∃ A B : ℝ × ℝ,
    (A.1 - m * A.2 + 1 = 0 ∧ (A.1 - 1)^2 + A.2^2 = 4) ∧
    (B.1 - m * B.2 + 1 = 0 ∧ (B.1 - 1)^2 + B.2^2 = 4) ∧
    A ≠ B ∧
    abs ((A.1 - C.1) * (B.2 - C.2) - (B.1 - C.1) * (A.2 - C.2)) / 2 = 8/5) →
  m = 2 ∨ m = -2 ∨ m = 1/2 ∨ m = -1/2 :=
sorry

end NUMINAMATH_CALUDE_line_circle_intersection_l601_60162


namespace NUMINAMATH_CALUDE_vector_relations_l601_60161

-- Define the plane vector type
structure PlaneVector where
  x : ℝ
  y : ℝ

-- Define the "›" relation
def vecGreater (a b : PlaneVector) : Prop :=
  a.x > b.x ∨ (a.x = b.x ∧ a.y > b.y)

-- Define vector addition
def vecAdd (a b : PlaneVector) : PlaneVector :=
  ⟨a.x + b.x, a.y + b.y⟩

-- Define dot product
def vecDot (a b : PlaneVector) : ℝ :=
  a.x * b.x + a.y * b.y

-- Theorem statement
theorem vector_relations :
  let e₁ : PlaneVector := ⟨1, 0⟩
  let e₂ : PlaneVector := ⟨0, 1⟩
  let zero : PlaneVector := ⟨0, 0⟩
  
  -- Proposition 1
  (vecGreater e₁ e₂ ∧ vecGreater e₂ zero) ∧
  
  -- Proposition 2
  (∀ a₁ a₂ a₃ : PlaneVector, vecGreater a₁ a₂ → vecGreater a₂ a₃ → vecGreater a₁ a₃) ∧
  
  -- Proposition 3
  (∀ a₁ a₂ a : PlaneVector, vecGreater a₁ a₂ → vecGreater (vecAdd a₁ a) (vecAdd a₂ a)) ∧
  
  -- Proposition 4 (negation)
  ¬(∀ a a₁ a₂ : PlaneVector, vecGreater a zero → vecGreater a₁ a₂ → vecGreater ⟨vecDot a a₁, 0⟩ ⟨vecDot a a₂, 0⟩) :=
by
  sorry


end NUMINAMATH_CALUDE_vector_relations_l601_60161

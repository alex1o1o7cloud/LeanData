import Mathlib

namespace NUMINAMATH_CALUDE_average_apples_per_guest_l3351_335116

/-- Represents the number of apples per serving -/
def apples_per_serving : ℚ := 3/2

/-- Represents the ratio of Red Delicious to Granny Smith apples per serving -/
def apple_ratio : ℚ := 2

/-- Represents the number of guests -/
def num_guests : ℕ := 12

/-- Represents the number of pies -/
def num_pies : ℕ := 3

/-- Represents the number of servings per pie -/
def servings_per_pie : ℕ := 8

/-- Represents the number of cups of apple pieces per Red Delicious apple -/
def red_delicious_cups : ℚ := 1

/-- Represents the number of cups of apple pieces per Granny Smith apple -/
def granny_smith_cups : ℚ := 5/4

/-- Theorem stating that the average number of apples each guest eats is 3 -/
theorem average_apples_per_guest : 
  (num_pies * servings_per_pie * apples_per_serving) / num_guests = 3 := by
  sorry

end NUMINAMATH_CALUDE_average_apples_per_guest_l3351_335116


namespace NUMINAMATH_CALUDE_right_triangle_inequality_l3351_335139

theorem right_triangle_inequality (a b c x : ℝ) : 
  a > 0 → b > 0 → c > 0 → 
  a^2 = b^2 + c^2 → 
  a ≥ b ∧ a ≥ c → 
  (a^x > b^x + c^x ↔ x > 2) :=
sorry

end NUMINAMATH_CALUDE_right_triangle_inequality_l3351_335139


namespace NUMINAMATH_CALUDE_complex_magnitude_product_l3351_335186

theorem complex_magnitude_product : 
  Complex.abs ((3 * Real.sqrt 5 - 5 * Complex.I) * (2 * Real.sqrt 7 + 4 * Complex.I)) = 20 * Real.sqrt 77 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_product_l3351_335186


namespace NUMINAMATH_CALUDE_peanut_distribution_l3351_335153

theorem peanut_distribution (x₁ x₂ x₃ x₄ x₅ : ℕ) : 
  x₁ + x₂ + x₃ + x₄ + x₅ = 100 ∧
  x₁ + x₂ = 52 ∧
  x₂ + x₃ = 43 ∧
  x₃ + x₄ = 34 ∧
  x₄ + x₅ = 30 →
  x₁ = 27 ∧ x₂ = 25 ∧ x₃ = 18 ∧ x₄ = 16 ∧ x₅ = 14 :=
by
  sorry

#check peanut_distribution

end NUMINAMATH_CALUDE_peanut_distribution_l3351_335153


namespace NUMINAMATH_CALUDE_parallel_lines_a_value_l3351_335161

/-- Given two lines l₁ and l₂, prove that a = -1 -/
theorem parallel_lines_a_value (a : ℝ) : 
  (∀ x y : ℝ, 2*x + (a-1)*y + a = 0 ↔ a*x + y + 2 = 0) → -- l₁ and l₂ are parallel
  (2*2 ≠ a^2) →                                         -- Additional condition
  a = -1 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_a_value_l3351_335161


namespace NUMINAMATH_CALUDE_largest_gold_coins_max_gold_coins_l3351_335118

theorem largest_gold_coins (n : ℕ) : n < 120 → n % 15 = 3 → n ≤ 105 := by
  sorry

theorem max_gold_coins : ∃ n : ℕ, n = 105 ∧ n < 120 ∧ n % 15 = 3 ∧ ∀ m : ℕ, m < 120 → m % 15 = 3 → m ≤ n := by
  sorry

end NUMINAMATH_CALUDE_largest_gold_coins_max_gold_coins_l3351_335118


namespace NUMINAMATH_CALUDE_no_permutation_sum_all_nines_l3351_335111

/-- The number of 9's in the sum -/
def num_nines : ℕ := 1111

/-- Function to check if a number is composed of only 9's -/
def is_all_nines (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 10^k - 1 ∧ k = num_nines

/-- Function to check if two numbers are digit permutations of each other -/
def is_permutation (x y : ℕ) : Prop :=
  ∃ σ : Fin (Nat.digits 10 x).length ≃ Fin (Nat.digits 10 y).length,
    ∀ i, (Nat.digits 10 x)[i] = (Nat.digits 10 y)[σ i]

/-- Main theorem statement -/
theorem no_permutation_sum_all_nines :
  ¬∃ (x y : ℕ), is_permutation x y ∧ is_all_nines (x + y) := by
  sorry

end NUMINAMATH_CALUDE_no_permutation_sum_all_nines_l3351_335111


namespace NUMINAMATH_CALUDE_percent_within_one_std_dev_l3351_335103

-- Define a symmetric distribution
structure SymmetricDistribution where
  mean : ℝ
  std_dev : ℝ
  is_symmetric : Bool
  percent_less_than_mean_plus_std : ℝ

-- Theorem statement
theorem percent_within_one_std_dev 
  (dist : SymmetricDistribution) 
  (h1 : dist.is_symmetric = true) 
  (h2 : dist.percent_less_than_mean_plus_std = 92) : 
  (100 - 2 * (100 - dist.percent_less_than_mean_plus_std)) = 84 := by
  sorry

end NUMINAMATH_CALUDE_percent_within_one_std_dev_l3351_335103


namespace NUMINAMATH_CALUDE_fraction_sum_equals_point_four_l3351_335100

theorem fraction_sum_equals_point_four :
  2 / 20 + 3 / 30 + 4 / 40 + 5 / 50 = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_point_four_l3351_335100


namespace NUMINAMATH_CALUDE_max_distance_to_line_l3351_335173

/-- Given m ∈ ℝ, prove that the maximum distance from a point P(x,y) satisfying both
    x + m*y = 0 and m*x - y - 2*m + 4 = 0 to the line (x-1)*cos θ + (y-2)*sin θ = 3 is 3 + √5 -/
theorem max_distance_to_line (m : ℝ) :
  let P : ℝ × ℝ := (x, y) 
  ∃ x y : ℝ, x + m*y = 0 ∧ m*x - y - 2*m + 4 = 0 →
  (∀ θ : ℝ, (x - 1)*Real.cos θ + (y - 2)*Real.sin θ ≤ 3 + Real.sqrt 5) ∧
  (∃ θ : ℝ, (x - 1)*Real.cos θ + (y - 2)*Real.sin θ = 3 + Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_max_distance_to_line_l3351_335173


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l3351_335133

theorem smallest_n_congruence : 
  ∃! (n : ℕ), n > 0 ∧ (3 * n) % 24 = 1410 % 24 ∧ ∀ m : ℕ, m > 0 → (3 * m) % 24 = 1410 % 24 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l3351_335133


namespace NUMINAMATH_CALUDE_cube_root_problem_l3351_335167

theorem cube_root_problem (a : ℕ) : a^3 = 21 * 25 * 45 * 49 → a = 105 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_problem_l3351_335167


namespace NUMINAMATH_CALUDE_modified_circle_radius_l3351_335188

/-- Given a circle with radius r, prove that if its modified area and circumference
    sum to 180π, then r satisfies the equation r² + 2r - 90 = 0 -/
theorem modified_circle_radius (r : ℝ) : 
  (2 * Real.pi * r^2) + (4 * Real.pi * r) = 180 * Real.pi → 
  r^2 + 2*r - 90 = 0 := by
  sorry


end NUMINAMATH_CALUDE_modified_circle_radius_l3351_335188


namespace NUMINAMATH_CALUDE_digit_matching_equality_l3351_335120

theorem digit_matching_equality : ∃ (a b : ℕ), 
  a ≠ b ∧ 
  a ≤ 99 ∧ 
  b ≤ 99 ∧ 
  a + b ≤ 9999 ∧ 
  (a + b)^2 = 100 * a + b :=
sorry

end NUMINAMATH_CALUDE_digit_matching_equality_l3351_335120


namespace NUMINAMATH_CALUDE_identical_car_in_kindergarten_l3351_335159

-- Define the properties of a car
structure Car where
  color : String
  size : String
  hasTrailer : Bool

-- Define the boys and their car collections
def Misha : List Car := [
  { color := "green", size := "small", hasTrailer := false },
  { color := "unknown", size := "small", hasTrailer := false },
  { color := "unknown", size := "unknown", hasTrailer := true }
]

def Vitya : List Car := [
  { color := "unknown", size := "unknown", hasTrailer := false },
  { color := "green", size := "small", hasTrailer := true }
]

def Kolya : List Car := [
  { color := "unknown", size := "big", hasTrailer := false },
  { color := "blue", size := "small", hasTrailer := true }
]

-- Define the theorem
theorem identical_car_in_kindergarten :
  ∃ (c : Car),
    c ∈ Misha ∧ c ∈ Vitya ∧ c ∈ Kolya ∧
    c.color = "green" ∧ c.size = "big" ∧ c.hasTrailer = false :=
by
  sorry

end NUMINAMATH_CALUDE_identical_car_in_kindergarten_l3351_335159


namespace NUMINAMATH_CALUDE_quadratic_equation_real_roots_l3351_335198

theorem quadratic_equation_real_roots (k : ℕ) : 
  (∃ x : ℝ, k * x^2 - 2 * x + 1 = 0) → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_real_roots_l3351_335198


namespace NUMINAMATH_CALUDE_plant_branches_problem_l3351_335138

theorem plant_branches_problem (x : ℕ) : 
  (1 + x + x^2 = 43) → (x = 6) :=
by
  sorry

end NUMINAMATH_CALUDE_plant_branches_problem_l3351_335138


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3351_335148

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 + x - 12 ≥ 0} = {x : ℝ | x ≤ -4 ∨ x ≥ 3} := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3351_335148


namespace NUMINAMATH_CALUDE_birdseed_mix_cost_l3351_335199

theorem birdseed_mix_cost (millet_weight : ℝ) (millet_cost : ℝ) (sunflower_weight : ℝ) (mixture_cost : ℝ) :
  millet_weight = 100 →
  millet_cost = 0.60 →
  sunflower_weight = 25 →
  mixture_cost = 0.70 →
  let total_weight := millet_weight + sunflower_weight
  let total_cost := mixture_cost * total_weight
  let millet_total_cost := millet_weight * millet_cost
  let sunflower_total_cost := total_cost - millet_total_cost
  sunflower_total_cost / sunflower_weight = 1.10 := by
sorry

end NUMINAMATH_CALUDE_birdseed_mix_cost_l3351_335199


namespace NUMINAMATH_CALUDE_paint_mixer_production_time_l3351_335158

/-- A paint mixer's production rate and time to complete a job -/
theorem paint_mixer_production_time 
  (days_for_some_drums : ℕ) 
  (total_drums : ℕ) 
  (total_days : ℕ) 
  (h1 : days_for_some_drums = 3)
  (h2 : total_drums = 360)
  (h3 : total_days = 60) :
  total_days = total_drums / (total_drums / total_days) :=
by sorry

end NUMINAMATH_CALUDE_paint_mixer_production_time_l3351_335158


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l3351_335195

theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  ∃ x y : ℝ, x = -2 ∧ y = 6 ∧ a^(x + 2) + 5 = y :=
sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l3351_335195


namespace NUMINAMATH_CALUDE_grandfather_age_ratio_l3351_335196

/-- Given the current ages of Xiao Hong and her grandfather, 
    prove the ratio of their ages last year -/
theorem grandfather_age_ratio (xiao_hong_age grandfather_age : ℕ) 
  (h1 : xiao_hong_age = 8) 
  (h2 : grandfather_age = 64) : 
  (grandfather_age - 1) / (xiao_hong_age - 1) = 9 := by
  sorry

end NUMINAMATH_CALUDE_grandfather_age_ratio_l3351_335196


namespace NUMINAMATH_CALUDE_bake_sale_fundraiser_l3351_335117

/-- 
Given a bake sale that earned $400 total, prove that the amount kept for ingredients
is $100, when half of the remaining amount plus $10 equals $160.
-/
theorem bake_sale_fundraiser (total_earnings : ℝ) (donation_to_shelter : ℝ) :
  total_earnings = 400 ∧ 
  donation_to_shelter = 160 ∧
  donation_to_shelter = (total_earnings - (total_earnings - donation_to_shelter + 10)) / 2 + 10 →
  total_earnings - donation_to_shelter + 10 = 100 := by
sorry

end NUMINAMATH_CALUDE_bake_sale_fundraiser_l3351_335117


namespace NUMINAMATH_CALUDE_pirate_loot_sum_l3351_335123

def base5ToBase10 (n : List Nat) : Nat :=
  List.foldl (fun acc d => acc * 5 + d) 0 (List.reverse n)

theorem pirate_loot_sum :
  let silver := base5ToBase10 [1, 4, 3, 2]
  let spices := base5ToBase10 [2, 1, 3, 4]
  let silk := base5ToBase10 [3, 0, 2, 1]
  let books := base5ToBase10 [2, 3, 1]
  silver + spices + silk + books = 988 := by
  sorry

end NUMINAMATH_CALUDE_pirate_loot_sum_l3351_335123


namespace NUMINAMATH_CALUDE_max_value_of_function_l3351_335194

theorem max_value_of_function :
  let f : ℝ → ℝ := λ x => (Real.sqrt 3 / 2) * Real.sin (x + Real.pi / 2) + Real.cos (Real.pi / 6 - x)
  ∃ (M : ℝ), M = Real.sqrt 13 / 2 ∧ ∀ (x : ℝ), f x ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_function_l3351_335194


namespace NUMINAMATH_CALUDE_negative_x_exponent_product_l3351_335122

theorem negative_x_exponent_product (x : ℝ) : (-x)^3 * (-x)^4 = -x^7 := by sorry

end NUMINAMATH_CALUDE_negative_x_exponent_product_l3351_335122


namespace NUMINAMATH_CALUDE_trajectory_properties_line_intersection_condition_unique_k_for_dot_product_l3351_335132

noncomputable def trajectory (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + (y - 1)^2) = Real.sqrt 2 * Real.sqrt ((x - 1)^2 + (y - 2)^2)

def line_intersects (k : ℝ) (x y : ℝ) : Prop :=
  y = k * x - 1

theorem trajectory_properties :
  ∃ (center_x center_y radius : ℝ),
    (∀ x y, trajectory x y ↔ (x - center_x)^2 + (y - center_y)^2 = radius^2) ∧
    center_x = 2 ∧ center_y = 3 ∧ radius = 2 :=
sorry

theorem line_intersection_condition (k : ℝ) :
  (∃ x y, trajectory x y ∧ line_intersects k x y) ↔ k > 3/4 :=
sorry

theorem unique_k_for_dot_product :
  ∃! k, k > 3/4 ∧
    ∀ x₁ y₁ x₂ y₂,
      trajectory x₁ y₁ ∧ trajectory x₂ y₂ ∧
      line_intersects k x₁ y₁ ∧ line_intersects k x₂ y₂ →
      x₁ * x₂ + y₁ * y₂ = 11 :=
sorry

end NUMINAMATH_CALUDE_trajectory_properties_line_intersection_condition_unique_k_for_dot_product_l3351_335132


namespace NUMINAMATH_CALUDE_standard_time_proof_l3351_335128

/-- The standard time to complete one workpiece -/
def standard_time : ℝ := 15

/-- The time taken by the first worker after innovation -/
def worker1_time (x : ℝ) : ℝ := x - 5

/-- The time taken by the second worker after innovation -/
def worker2_time (x : ℝ) : ℝ := x - 3

/-- The performance improvement factor -/
def improvement_factor : ℝ := 1.375

theorem standard_time_proof :
  ∃ (x : ℝ),
    x > 0 ∧
    worker1_time x > 0 ∧
    worker2_time x > 0 ∧
    (1 / worker1_time x + 1 / worker2_time x) = (2 / x) * improvement_factor ∧
    x = standard_time :=
by sorry

end NUMINAMATH_CALUDE_standard_time_proof_l3351_335128


namespace NUMINAMATH_CALUDE_rectangle_area_theorem_l3351_335105

theorem rectangle_area_theorem (x : ℝ) : 
  let large_rectangle_area := (2*x + 14) * (2*x + 10)
  let hole_area := (4*x - 6) * (2*x - 4)
  let square_area := (x + 3)^2
  large_rectangle_area - hole_area + square_area = -3*x^2 + 82*x + 125 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_theorem_l3351_335105


namespace NUMINAMATH_CALUDE_fraction_to_decimal_decimal_representation_l3351_335189

theorem fraction_to_decimal : (47 : ℚ) / (2^3 * 5^4) = (94 : ℚ) / 10000 := by
  sorry

theorem decimal_representation : (94 : ℚ) / 10000 = 0.0094 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_decimal_representation_l3351_335189


namespace NUMINAMATH_CALUDE_license_plate_increase_l3351_335183

theorem license_plate_increase : 
  let old_format := 26^2 * 10^3
  let new_format := 26^4 * 10^4
  (new_format / old_format : ℚ) = 2600 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_increase_l3351_335183


namespace NUMINAMATH_CALUDE_orchard_solution_l3351_335141

/-- Represents the number of trees in an orchard -/
structure Orchard where
  peach : ℕ
  apple : ℕ

/-- Conditions for the orchard problem -/
def OrchardConditions (o : Orchard) : Prop :=
  (o.apple = o.peach + 1700) ∧ (o.apple = 3 * o.peach + 200)

/-- Theorem stating the solution to the orchard problem -/
theorem orchard_solution : 
  ∃ o : Orchard, OrchardConditions o ∧ o.peach = 750 ∧ o.apple = 2450 := by
  sorry

end NUMINAMATH_CALUDE_orchard_solution_l3351_335141


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l3351_335178

theorem sqrt_meaningful_range (x : ℝ) :
  (∃ y : ℝ, y ^ 2 = 3 * x - 5) ↔ x ≥ 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l3351_335178


namespace NUMINAMATH_CALUDE_bisection_method_next_interval_l3351_335184

def f (x : ℝ) := x^3 - 2*x - 5

theorem bisection_method_next_interval :
  let a := 2
  let b := 3
  let x₀ := (a + b) / 2
  f a * f x₀ < 0 → ∃ x ∈ Set.Icc a x₀, f x = 0 :=
by sorry

end NUMINAMATH_CALUDE_bisection_method_next_interval_l3351_335184


namespace NUMINAMATH_CALUDE_rectangle_ribbon_length_l3351_335163

/-- The length of ribbon required to form a rectangle -/
def ribbon_length (length width : ℝ) : ℝ := 2 * (length + width)

/-- Theorem: The length of ribbon required to form a rectangle with length 20 feet and width 15 feet is 70 feet -/
theorem rectangle_ribbon_length : 
  ribbon_length 20 15 = 70 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_ribbon_length_l3351_335163


namespace NUMINAMATH_CALUDE_prob_three_same_color_l3351_335106

/-- The number of red marbles in the bag -/
def red_marbles : ℕ := 6

/-- The number of white marbles in the bag -/
def white_marbles : ℕ := 7

/-- The number of blue marbles in the bag -/
def blue_marbles : ℕ := 8

/-- The number of green marbles in the bag -/
def green_marbles : ℕ := 4

/-- The total number of marbles in the bag -/
def total_marbles : ℕ := red_marbles + white_marbles + blue_marbles + green_marbles

/-- The probability of drawing three marbles of the same color without replacement -/
theorem prob_three_same_color : 
  (red_marbles * (red_marbles - 1) * (red_marbles - 2) + 
   white_marbles * (white_marbles - 1) * (white_marbles - 2) +
   blue_marbles * (blue_marbles - 1) * (blue_marbles - 2) +
   green_marbles * (green_marbles - 1) * (green_marbles - 2)) / 
  (total_marbles * (total_marbles - 1) * (total_marbles - 2)) = 1 / 20 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_same_color_l3351_335106


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l3351_335155

theorem min_value_x_plus_y (x y : ℝ) (h1 : 4/x + 9/y = 1) (h2 : x > 0) (h3 : y > 0) : 
  ∀ a b : ℝ, a > 0 → b > 0 → 4/a + 9/b = 1 → x + y ≤ a + b :=
by sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l3351_335155


namespace NUMINAMATH_CALUDE_sin_x_in_terms_of_a_and_b_l3351_335150

theorem sin_x_in_terms_of_a_and_b (a b x : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : 0 < x) (h4 : x < π/2) (h5 : Real.tan x = (3*a*b) / (a^2 - b^2)) : 
  Real.sin x = (3*a*b) / Real.sqrt (a^4 + 7*a^2*b^2 + b^4) := by
  sorry

end NUMINAMATH_CALUDE_sin_x_in_terms_of_a_and_b_l3351_335150


namespace NUMINAMATH_CALUDE_phi_difference_squared_l3351_335179

theorem phi_difference_squared : ∀ Φ φ : ℝ, 
  Φ ≠ φ → 
  Φ^2 - 2*Φ - 1 = 0 → 
  φ^2 - 2*φ - 1 = 0 → 
  (Φ - φ)^2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_phi_difference_squared_l3351_335179


namespace NUMINAMATH_CALUDE_era_burgers_l3351_335109

theorem era_burgers (num_friends : ℕ) (slices_per_burger : ℕ) 
  (friend1_slices : ℕ) (friend2_slices : ℕ) (friend3_slices : ℕ) 
  (friend4_slices : ℕ) (era_slices : ℕ) :
  num_friends = 4 →
  slices_per_burger = 2 →
  friend1_slices = 1 →
  friend2_slices = 2 →
  friend3_slices = 3 →
  friend4_slices = 3 →
  era_slices = 1 →
  (friend1_slices + friend2_slices + friend3_slices + friend4_slices + era_slices) / slices_per_burger = 5 := by
  sorry

end NUMINAMATH_CALUDE_era_burgers_l3351_335109


namespace NUMINAMATH_CALUDE_emmalyn_earnings_l3351_335185

/-- The rate Emmalyn charges per meter for painting fences, in dollars -/
def rate : ℚ := 0.20

/-- The number of fences in the neighborhood -/
def num_fences : ℕ := 50

/-- The length of each fence in meters -/
def fence_length : ℕ := 500

/-- The total amount Emmalyn earned in dollars -/
def total_amount : ℚ := rate * (num_fences * fence_length)

theorem emmalyn_earnings :
  total_amount = 5000 := by sorry

end NUMINAMATH_CALUDE_emmalyn_earnings_l3351_335185


namespace NUMINAMATH_CALUDE_perimeter_of_square_b_l3351_335149

/-- Given a square A with perimeter 40 cm and a square B with area equal to one-third the area of square A, 
    the perimeter of square B is (40√3)/3 cm. -/
theorem perimeter_of_square_b (square_a square_b : Real → Real → Prop) : 
  (∃ side_a, square_a side_a side_a ∧ 4 * side_a = 40) →
  (∃ side_b, square_b side_b side_b ∧ side_b^2 = (side_a^2) / 3) →
  (∃ perimeter_b, perimeter_b = 40 * Real.sqrt 3 / 3) :=
by sorry

end NUMINAMATH_CALUDE_perimeter_of_square_b_l3351_335149


namespace NUMINAMATH_CALUDE_amusement_park_expenses_l3351_335192

theorem amusement_park_expenses (total brought food tshirt left ticket : ℕ) : 
  brought = 75 ∧ 
  food = 13 ∧ 
  tshirt = 23 ∧ 
  left = 9 ∧ 
  brought = food + tshirt + left + ticket → 
  ticket = 30 := by
  sorry

end NUMINAMATH_CALUDE_amusement_park_expenses_l3351_335192


namespace NUMINAMATH_CALUDE_squirrel_acorns_l3351_335126

theorem squirrel_acorns (initial_acorns : ℕ) (winter_months : ℕ) (remaining_per_month : ℕ) : 
  initial_acorns = 210 →
  winter_months = 3 →
  remaining_per_month = 60 →
  initial_acorns - (winter_months * remaining_per_month) = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_squirrel_acorns_l3351_335126


namespace NUMINAMATH_CALUDE_quadratic_inequality_l3351_335175

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * a * x + 4

-- State the theorem
theorem quadratic_inequality (a x₁ x₂ : ℝ) 
  (ha : 0 < a ∧ a < 3) 
  (hx : x₁ < x₂) 
  (hsum : x₁ + x₂ = 0) : 
  f a x₁ < f a x₂ := by
  sorry


end NUMINAMATH_CALUDE_quadratic_inequality_l3351_335175


namespace NUMINAMATH_CALUDE_trig_expression_equals_one_l3351_335191

theorem trig_expression_equals_one : 
  (Real.sin (20 * π / 180) * Real.cos (15 * π / 180) + 
   Real.cos (160 * π / 180) * Real.cos (105 * π / 180)) / 
  (Real.sin (25 * π / 180) * Real.cos (10 * π / 180) + 
   Real.cos (155 * π / 180) * Real.cos (95 * π / 180)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equals_one_l3351_335191


namespace NUMINAMATH_CALUDE_triangle_construction_l3351_335165

-- Define the basic structures
structure Point where
  x : ℝ
  y : ℝ

structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define properties
def isAcute (t : Triangle) : Prop := sorry

def onCircumcircle (p : Point) (t : Triangle) : Prop := sorry

def isAltitude (l : Point → Point → Prop) (t : Triangle) : Prop := sorry

def isAngleBisector (l : Point → Point → Prop) (t : Triangle) : Prop := sorry

def intersectsCircumcircle (l : Point → Point → Prop) (t : Triangle) : Point := sorry

-- Main theorem
theorem triangle_construction (t' : Triangle) (h_acute : isAcute t') :
  ∃ t : Triangle,
    (∀ p : Point, onCircumcircle p t' ↔ onCircumcircle p t) ∧
    isAcute t ∧
    (∀ l, isAltitude l t' → onCircumcircle (intersectsCircumcircle l t') t) ∧
    (∀ l, isAngleBisector l t → onCircumcircle (intersectsCircumcircle l t) t) :=
sorry

end NUMINAMATH_CALUDE_triangle_construction_l3351_335165


namespace NUMINAMATH_CALUDE_line_segment_endpoint_l3351_335127

/-- Given a line segment from (1, 3) to (x, -4) with length 15 and x > 0, prove x = 1 + √176 -/
theorem line_segment_endpoint (x : ℝ) : 
  x > 0 → 
  ((x - 1)^2 + (-4 - 3)^2).sqrt = 15 → 
  x = 1 + Real.sqrt 176 := by
  sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l3351_335127


namespace NUMINAMATH_CALUDE_quiz_probability_correct_l3351_335110

/-- Represents a quiz with one MCQ and two True/False questions -/
structure Quiz where
  mcq_options : Nat
  tf_questions : Nat

/-- Calculates the probability of answering all questions correctly in a quiz -/
def probability_all_correct (q : Quiz) : ℚ :=
  (1 : ℚ) / q.mcq_options * ((1 : ℚ) / 2) ^ q.tf_questions

/-- Theorem: The probability of answering all questions correctly in the given quiz is 1/12 -/
theorem quiz_probability_correct :
  let q := Quiz.mk 3 2
  probability_all_correct q = 1 / 12 := by
  sorry


end NUMINAMATH_CALUDE_quiz_probability_correct_l3351_335110


namespace NUMINAMATH_CALUDE_tangent_parallel_points_l3351_335113

/-- The function f(x) = x³ + x - 2 -/
def f (x : ℝ) : ℝ := x^3 + x - 2

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

theorem tangent_parallel_points :
  {x : ℝ | f' x = 4} = {1, -1} ∧
  f 1 = 0 ∧ f (-1) = -4 := by
  sorry

end NUMINAMATH_CALUDE_tangent_parallel_points_l3351_335113


namespace NUMINAMATH_CALUDE_cos_sin_225_degrees_l3351_335181

theorem cos_sin_225_degrees :
  Real.cos (225 * Real.pi / 180) = -Real.sqrt 2 / 2 ∧
  Real.sin (225 * Real.pi / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_sin_225_degrees_l3351_335181


namespace NUMINAMATH_CALUDE_cube_score_is_40_l3351_335143

/-- Represents the score for a unit cube based on the number of painted faces. -/
def score (painted_faces : Nat) : Int :=
  match painted_faces with
  | 3 => 3
  | 2 => 2
  | 1 => 1
  | 0 => -7
  | _ => 0  -- This case should never occur in our problem

/-- The size of one side of the cube. -/
def cube_size : Nat := 4

/-- The total number of unit cubes in the large cube. -/
def total_cubes : Nat := cube_size ^ 3

/-- The number of corner cubes (with 3 painted faces). -/
def corner_cubes : Nat := 8

/-- The number of edge cubes (with 2 painted faces), excluding corners. -/
def edge_cubes : Nat := 12 * (cube_size - 2)

/-- The number of face cubes (with 1 painted face), excluding edges and corners. -/
def face_cubes : Nat := 6 * (cube_size - 2) ^ 2

/-- The number of internal cubes (with 0 painted faces). -/
def internal_cubes : Nat := (cube_size - 2) ^ 3

theorem cube_score_is_40 :
  (corner_cubes * score 3 +
   edge_cubes * score 2 +
   face_cubes * score 1 +
   internal_cubes * score 0) = 40 ∧
  corner_cubes + edge_cubes + face_cubes + internal_cubes = total_cubes :=
sorry

end NUMINAMATH_CALUDE_cube_score_is_40_l3351_335143


namespace NUMINAMATH_CALUDE_students_just_passed_l3351_335171

theorem students_just_passed (total : ℕ) (first_div_percent : ℚ) (second_div_percent : ℚ)
  (h_total : total = 300)
  (h_first : first_div_percent = 25 / 100)
  (h_second : second_div_percent = 54 / 100)
  (h_no_fail : first_div_percent + second_div_percent < 1) :
  total - (total * first_div_percent).floor - (total * second_div_percent).floor = 63 := by
  sorry

end NUMINAMATH_CALUDE_students_just_passed_l3351_335171


namespace NUMINAMATH_CALUDE_ball_box_theorem_l3351_335169

/-- Represents the state of boxes after a number of steps -/
def BoxState := List Nat

/-- Converts a natural number to its septenary (base 7) representation -/
def toSeptenary (n : Nat) : List Nat :=
  sorry

/-- Simulates the ball-placing process for a given number of steps -/
def simulateSteps (steps : Nat) : BoxState :=
  sorry

/-- Counts the number of non-zero elements in a list -/
def countNonZero (l : List Nat) : Nat :=
  sorry

/-- Sums all elements in a list -/
def sumList (l : List Nat) : Nat :=
  sorry

theorem ball_box_theorem (steps : Nat := 3456) :
  let septenaryRep := toSeptenary steps
  let finalState := simulateSteps steps
  countNonZero finalState = countNonZero septenaryRep ∧
  sumList finalState = sumList septenaryRep :=
by sorry

end NUMINAMATH_CALUDE_ball_box_theorem_l3351_335169


namespace NUMINAMATH_CALUDE_nested_square_root_value_l3351_335115

theorem nested_square_root_value :
  ∃ y : ℝ, y = Real.sqrt (2 + y) → y = 2 := by
  sorry

end NUMINAMATH_CALUDE_nested_square_root_value_l3351_335115


namespace NUMINAMATH_CALUDE_stuffed_toy_dogs_boxes_l3351_335162

theorem stuffed_toy_dogs_boxes (dogs_per_box : ℕ) (total_dogs : ℕ) (h1 : dogs_per_box = 4) (h2 : total_dogs = 28) :
  total_dogs / dogs_per_box = 7 := by
  sorry

end NUMINAMATH_CALUDE_stuffed_toy_dogs_boxes_l3351_335162


namespace NUMINAMATH_CALUDE_quadratic_root_zero_l3351_335176

/-- Given a quadratic equation (a+3)x^2 - 4x + a^2 - 9 = 0 with 0 as a root and a + 3 ≠ 0, prove that a = 3 -/
theorem quadratic_root_zero (a : ℝ) : 
  ((a + 3) * 0^2 - 4 * 0 + a^2 - 9 = 0) → 
  (a + 3 ≠ 0) → 
  (a = 3) := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_zero_l3351_335176


namespace NUMINAMATH_CALUDE_marble_fraction_after_tripling_l3351_335172

theorem marble_fraction_after_tripling (total : ℚ) (h : total > 0) :
  let blue := (2 / 3) * total
  let red := total - blue
  let new_red := 3 * red
  let new_total := blue + new_red
  new_red / new_total = 3 / 5 := by
sorry

end NUMINAMATH_CALUDE_marble_fraction_after_tripling_l3351_335172


namespace NUMINAMATH_CALUDE_tetrahedron_division_possible_l3351_335193

theorem tetrahedron_division_possible (edge_length : ℝ) (target_length : ℝ) : 
  edge_length > 0 → target_length > 0 → target_length < edge_length →
  ∃ n : ℕ, (1/2 : ℝ)^n * edge_length < target_length := by
  sorry

#check tetrahedron_division_possible 1 (1/100)

end NUMINAMATH_CALUDE_tetrahedron_division_possible_l3351_335193


namespace NUMINAMATH_CALUDE_fraction_sum_l3351_335129

theorem fraction_sum (a b : ℚ) (h : a / b = 1 / 3) : (a + b) / b = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_l3351_335129


namespace NUMINAMATH_CALUDE_tangent_line_x_intercept_l3351_335144

-- Define the function f(x) = x³ + 4x + 5
def f (x : ℝ) : ℝ := x^3 + 4*x + 5

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 + 4

theorem tangent_line_x_intercept :
  let slope : ℝ := f' 1
  let y_intercept : ℝ := f 1 - slope * 1
  let x_intercept : ℝ := -y_intercept / slope
  x_intercept = -3/7 := by sorry

end NUMINAMATH_CALUDE_tangent_line_x_intercept_l3351_335144


namespace NUMINAMATH_CALUDE_cubic_sum_problem_l3351_335114

theorem cubic_sum_problem (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) :
  x^3 + y^3 = 65 := by sorry

end NUMINAMATH_CALUDE_cubic_sum_problem_l3351_335114


namespace NUMINAMATH_CALUDE_function_inequality_l3351_335125

theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, deriv f x > f x) : f 2 > Real.exp 2 * f 0 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l3351_335125


namespace NUMINAMATH_CALUDE_select_five_from_eight_l3351_335156

theorem select_five_from_eight (n m : ℕ) (h1 : n = 8) (h2 : m = 5) :
  Nat.choose n m = 56 := by
  sorry

end NUMINAMATH_CALUDE_select_five_from_eight_l3351_335156


namespace NUMINAMATH_CALUDE_total_distance_in_feet_l3351_335154

/-- Conversion factor from miles to feet -/
def miles_to_feet : ℝ := 5280

/-- Conversion factor from yards to feet -/
def yards_to_feet : ℝ := 3

/-- Conversion factor from kilometers to feet -/
def km_to_feet : ℝ := 3280.84

/-- Conversion factor from meters to feet -/
def meters_to_feet : ℝ := 3.28084

/-- Distance walked by Lionel in miles -/
def lionel_distance : ℝ := 4

/-- Distance walked by Esther in yards -/
def esther_distance : ℝ := 975

/-- Distance walked by Niklaus in feet -/
def niklaus_distance : ℝ := 1287

/-- Distance biked by Isabella in kilometers -/
def isabella_distance : ℝ := 18

/-- Distance swam by Sebastian in meters -/
def sebastian_distance : ℝ := 2400

/-- Theorem stating the total combined distance traveled by the friends in feet -/
theorem total_distance_in_feet :
  lionel_distance * miles_to_feet +
  esther_distance * yards_to_feet +
  niklaus_distance +
  isabella_distance * km_to_feet +
  sebastian_distance * meters_to_feet = 89261.136 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_in_feet_l3351_335154


namespace NUMINAMATH_CALUDE_inequality_holds_l3351_335108

theorem inequality_holds (a b c d : ℝ) (h1 : b < a) (h2 : d < c) : a + c > b + d := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_l3351_335108


namespace NUMINAMATH_CALUDE_radical_conjugate_sum_product_l3351_335140

theorem radical_conjugate_sum_product (a b : ℝ) : 
  (a + Real.sqrt b) + (a - Real.sqrt b) = 0 ∧ 
  (a + Real.sqrt b) * (a - Real.sqrt b) = 16 → 
  a + b = -16 := by
sorry

end NUMINAMATH_CALUDE_radical_conjugate_sum_product_l3351_335140


namespace NUMINAMATH_CALUDE_two_people_three_movies_l3351_335131

/-- The number of ways two people can choose tickets from three movies -/
def ticket_choices (num_people : ℕ) (num_movies : ℕ) : ℕ :=
  num_movies ^ num_people

/-- Theorem: Two people choosing from three movies results in 9 different combinations -/
theorem two_people_three_movies :
  ticket_choices 2 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_two_people_three_movies_l3351_335131


namespace NUMINAMATH_CALUDE_sector_max_area_l3351_335177

/-- Given a sector with circumference 8, its area is at most 4 -/
theorem sector_max_area :
  ∀ (r l : ℝ), r > 0 → l > 0 → 2 * r + l = 8 →
  (1 / 2) * l * r ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_sector_max_area_l3351_335177


namespace NUMINAMATH_CALUDE_technician_avg_salary_l3351_335112

def total_workers : ℕ := 24
def avg_salary_all : ℕ := 8000
def num_technicians : ℕ := 8
def avg_salary_non_tech : ℕ := 6000

theorem technician_avg_salary :
  let num_non_tech := total_workers - num_technicians
  let total_salary := avg_salary_all * total_workers
  let total_salary_non_tech := avg_salary_non_tech * num_non_tech
  let total_salary_tech := total_salary - total_salary_non_tech
  total_salary_tech / num_technicians = 12000 := by
  sorry

end NUMINAMATH_CALUDE_technician_avg_salary_l3351_335112


namespace NUMINAMATH_CALUDE_pencil_count_problem_l3351_335190

/-- Given an initial number of pencils, a number of lost pencils, and a number of gained pencils,
    calculate the final number of pencils. -/
def finalPencilCount (initial lost gained : ℕ) : ℕ :=
  initial - lost + gained

/-- Theorem stating that given the specific values in the problem,
    the final pencil count is 2060. -/
theorem pencil_count_problem :
  finalPencilCount 2015 5 50 = 2060 := by
  sorry

end NUMINAMATH_CALUDE_pencil_count_problem_l3351_335190


namespace NUMINAMATH_CALUDE_max_length_theorem_l3351_335151

-- Define the ellipse E
def E (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the circle C
def C (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define a line passing through (0,1)
def Line (k : ℝ) (x y : ℝ) : Prop := y = k * x + 1

-- Define the intersection points
def IntersectionPoints (k : ℝ) :
  (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) :=
  sorry

-- Define the lengths |A₁B₁| and |A₂B₂|
def Length_A1B1 (k : ℝ) : ℝ := sorry
def Length_A2B2 (k : ℝ) : ℝ := sorry

-- The main theorem
theorem max_length_theorem :
  ∃ k : ℝ, Length_A1B1 k = max_length_A1B1 ∧ Length_A2B2 k = 2 * Real.sqrt 30 / 3 :=
sorry

end NUMINAMATH_CALUDE_max_length_theorem_l3351_335151


namespace NUMINAMATH_CALUDE_fraction_equality_l3351_335137

theorem fraction_equality (x : ℝ) : (5 + x) / (7 + x) = (3 + x) / (4 + x) → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3351_335137


namespace NUMINAMATH_CALUDE_driver_net_pay_rate_driver_net_pay_is_25_l3351_335160

/-- Calculates the net rate of pay for a driver given specific conditions -/
theorem driver_net_pay_rate (travel_time : ℝ) (speed : ℝ) (fuel_efficiency : ℝ) 
  (compensation_rate : ℝ) (fuel_cost : ℝ) : ℝ :=
  let total_distance := travel_time * speed
  let fuel_used := total_distance / fuel_efficiency
  let earnings := compensation_rate * total_distance
  let fuel_expense := fuel_cost * fuel_used
  let net_earnings := earnings - fuel_expense
  let net_rate := net_earnings / travel_time
  net_rate

/-- Proves that the driver's net rate of pay is $25 per hour under the given conditions -/
theorem driver_net_pay_is_25 : 
  driver_net_pay_rate 3 50 25 0.60 2.50 = 25 := by
  sorry

end NUMINAMATH_CALUDE_driver_net_pay_rate_driver_net_pay_is_25_l3351_335160


namespace NUMINAMATH_CALUDE_sum_of_large_prime_factors_2310_l3351_335134

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem sum_of_large_prime_factors_2310 : 
  ∃ (factors : List ℕ), 
    (∀ f ∈ factors, is_prime f ∧ f > 5) ∧ 
    (factors.prod = 2310 / (2 * 3 * 5)) ∧
    (factors.sum = 18) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_large_prime_factors_2310_l3351_335134


namespace NUMINAMATH_CALUDE_teachers_percentage_of_boys_l3351_335101

/-- Proves that the percentage of teachers to boys is 20% given the specified conditions -/
theorem teachers_percentage_of_boys (boys girls teachers : ℕ) : 
  (boys : ℚ) / (girls : ℚ) = 3 / 4 →
  girls = 60 →
  boys + girls + teachers = 114 →
  (teachers : ℚ) / (boys : ℚ) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_teachers_percentage_of_boys_l3351_335101


namespace NUMINAMATH_CALUDE_max_tau_minus_n_max_tau_minus_n_achievable_l3351_335121

/-- τ(n) is the number of positive divisors of n -/
def tau (n : ℕ+) : ℕ := sorry

/-- The theorem states that 4τ(n) - n is at most 12 for all positive integers n -/
theorem max_tau_minus_n (n : ℕ+) : 4 * (tau n) - n.val ≤ 12 := by sorry

/-- The theorem states that there exists a positive integer n for which 4τ(n) - n equals 12 -/
theorem max_tau_minus_n_achievable : ∃ n : ℕ+, 4 * (tau n) - n.val = 12 := by sorry

end NUMINAMATH_CALUDE_max_tau_minus_n_max_tau_minus_n_achievable_l3351_335121


namespace NUMINAMATH_CALUDE_minnows_per_prize_bowl_l3351_335180

theorem minnows_per_prize_bowl (total_minnows : ℕ) (total_players : ℕ) (winner_percentage : ℚ) (leftover_minnows : ℕ) :
  total_minnows = 600 →
  total_players = 800 →
  winner_percentage = 15 / 100 →
  leftover_minnows = 240 →
  (total_minnows - leftover_minnows) / (total_players * winner_percentage) = 3 :=
by sorry

end NUMINAMATH_CALUDE_minnows_per_prize_bowl_l3351_335180


namespace NUMINAMATH_CALUDE_van_capacity_l3351_335135

theorem van_capacity (students : ℕ) (adults : ℕ) (vans : ℕ) (h1 : students = 2) (h2 : adults = 6) (h3 : vans = 2) :
  (students + adults) / vans = 4 := by
  sorry

end NUMINAMATH_CALUDE_van_capacity_l3351_335135


namespace NUMINAMATH_CALUDE_magnitude_of_complex_fraction_l3351_335147

theorem magnitude_of_complex_fraction (z : ℂ) : z = (2 + Complex.I) / Complex.I → Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_complex_fraction_l3351_335147


namespace NUMINAMATH_CALUDE_amoeba_count_14_days_l3351_335142

/-- Calculates the number of amoebas after a given number of days -/
def amoeba_count (days : ℕ) : ℕ :=
  if days ≤ 2 then 2^(days - 1)
  else 5 * 2^(days - 3)

/-- The number of amoebas after 14 days is 10240 -/
theorem amoeba_count_14_days : amoeba_count 14 = 10240 := by
  sorry

end NUMINAMATH_CALUDE_amoeba_count_14_days_l3351_335142


namespace NUMINAMATH_CALUDE_triangle_formation_l3351_335152

/-- Two lines in the Cartesian coordinate system -/
structure CartesianLines where
  line1 : ℝ → ℝ
  line2 : ℝ → ℝ
  k : ℝ

/-- Condition for two lines to form a triangle with the x-axis -/
def formsTriangle (lines : CartesianLines) : Prop :=
  lines.k ≠ 0 ∧ lines.k ≠ -1/2

/-- Theorem: The given lines form a triangle with the x-axis if and only if k ≠ -1/2 -/
theorem triangle_formation (lines : CartesianLines) 
  (h1 : lines.line1 = fun x ↦ -0.5 * x - 2)
  (h2 : lines.line2 = fun x ↦ lines.k * x + 3) :
  formsTriangle lines ↔ lines.k ≠ -1/2 :=
sorry

end NUMINAMATH_CALUDE_triangle_formation_l3351_335152


namespace NUMINAMATH_CALUDE_invalid_external_diagonals_l3351_335187

/-- Checks if three numbers can be the lengths of external diagonals of a right regular prism -/
def are_valid_external_diagonals (a b c : ℝ) : Prop :=
  a^2 + b^2 > c^2 ∧ b^2 + c^2 > a^2 ∧ a^2 + c^2 > b^2

/-- Theorem stating that {5, 7, 9} cannot be the external diagonals of a right regular prism -/
theorem invalid_external_diagonals :
  ¬(are_valid_external_diagonals 5 7 9) := by
  sorry

end NUMINAMATH_CALUDE_invalid_external_diagonals_l3351_335187


namespace NUMINAMATH_CALUDE_circle_radius_sqrt34_l3351_335136

/-- Given a circle with center on the x-axis passing through points (0,5) and (2,3),
    prove that its radius is √34. -/
theorem circle_radius_sqrt34 :
  ∀ x : ℝ,
  (x^2 + 5^2 = (x-2)^2 + 3^2) →  -- condition that (x,0) is equidistant from (0,5) and (2,3)
  ∃ r : ℝ,
  r^2 = 34 ∧                    -- r is the radius
  r^2 = x^2 + 5^2               -- distance formula from center to (0,5)
  :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_sqrt34_l3351_335136


namespace NUMINAMATH_CALUDE_stating_max_equations_theorem_l3351_335145

/-- 
Represents the maximum number of equations without real roots 
that the first player can guarantee in a game with n equations.
-/
def max_equations_without_real_roots (n : ℕ) : ℕ :=
  if n % 2 = 0 then 0 else (n + 1) / 2

/-- 
Theorem stating the maximum number of equations without real roots 
that the first player can guarantee in the game.
-/
theorem max_equations_theorem (n : ℕ) :
  max_equations_without_real_roots n = 
    if n % 2 = 0 then 0 else (n + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_stating_max_equations_theorem_l3351_335145


namespace NUMINAMATH_CALUDE_power_multiplication_subtraction_l3351_335157

theorem power_multiplication_subtraction (a : ℝ) : 4 * a * a^3 - a^4 = 3 * a^4 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_subtraction_l3351_335157


namespace NUMINAMATH_CALUDE_correct_calculation_incorrect_calculation_A_incorrect_calculation_B_incorrect_calculation_C_l3351_335104

theorem correct_calculation : 2 * Real.sqrt 3 - Real.sqrt 3 = Real.sqrt 3 :=
by sorry

theorem incorrect_calculation_A : ¬(Real.sqrt 3 + Real.sqrt 2 = Real.sqrt 5) :=
by sorry

theorem incorrect_calculation_B : ¬(Real.sqrt 3 * Real.sqrt 5 = 15) :=
by sorry

theorem incorrect_calculation_C : ¬(Real.sqrt 32 / Real.sqrt 8 = 2 ∨ Real.sqrt 32 / Real.sqrt 8 = -2) :=
by sorry

end NUMINAMATH_CALUDE_correct_calculation_incorrect_calculation_A_incorrect_calculation_B_incorrect_calculation_C_l3351_335104


namespace NUMINAMATH_CALUDE_pet_shop_kittens_l3351_335164

/-- Represents the pet shop inventory and pricing --/
structure PetShop where
  num_puppies : ℕ
  puppy_price : ℕ
  kitten_price : ℕ
  total_value : ℕ

/-- Calculates the number of kittens in the pet shop --/
def num_kittens (shop : PetShop) : ℕ :=
  (shop.total_value - shop.num_puppies * shop.puppy_price) / shop.kitten_price

/-- Theorem stating that the number of kittens in the given pet shop is 4 --/
theorem pet_shop_kittens :
  let shop : PetShop := {
    num_puppies := 2,
    puppy_price := 20,
    kitten_price := 15,
    total_value := 100
  }
  num_kittens shop = 4 := by
  sorry

end NUMINAMATH_CALUDE_pet_shop_kittens_l3351_335164


namespace NUMINAMATH_CALUDE_dog_food_per_dog_l3351_335146

/-- The amount of dog food two dogs eat together per day -/
def total_food : ℝ := 0.25

/-- The number of dogs -/
def num_dogs : ℕ := 2

theorem dog_food_per_dog :
  ∀ (food_per_dog : ℝ),
  (food_per_dog * num_dogs = total_food) →
  (food_per_dog = 0.125) := by
sorry

end NUMINAMATH_CALUDE_dog_food_per_dog_l3351_335146


namespace NUMINAMATH_CALUDE_polynomial_form_l3351_335130

/-- A polynomial with real coefficients -/
def RealPolynomial := ℝ → ℝ

/-- The functional equation that P must satisfy -/
def SatisfiesEquation (P : RealPolynomial) : Prop :=
  ∀ (x : ℝ), x ≠ 0 → P x + P (1/x) = (P (x + 1/x) + P (x - 1/x)) / 2

/-- The form of the polynomial we want to prove -/
def HasRequiredForm (P : RealPolynomial) : Prop :=
  ∃ (a b : ℝ), ∀ (x : ℝ), P x = a * x^4 + b * x^2 + 6 * a

theorem polynomial_form (P : RealPolynomial) :
  SatisfiesEquation P → HasRequiredForm P :=
by sorry

end NUMINAMATH_CALUDE_polynomial_form_l3351_335130


namespace NUMINAMATH_CALUDE_diving_class_capacity_l3351_335182

/-- The number of people that can be accommodated in each diving class -/
def people_per_class : ℕ := 5

/-- The number of weekdays in a week -/
def weekdays : ℕ := 5

/-- The number of weekend days in a week -/
def weekend_days : ℕ := 2

/-- The number of classes per weekday -/
def classes_per_weekday : ℕ := 2

/-- The number of classes per weekend day -/
def classes_per_weekend_day : ℕ := 4

/-- The number of weeks -/
def weeks : ℕ := 3

/-- The total number of people that can take classes in 3 weeks -/
def total_people : ℕ := 270

/-- Theorem stating that the number of people per class is 5 -/
theorem diving_class_capacity :
  people_per_class = 
    total_people / (weeks * (weekdays * classes_per_weekday + weekend_days * classes_per_weekend_day)) :=
by sorry

end NUMINAMATH_CALUDE_diving_class_capacity_l3351_335182


namespace NUMINAMATH_CALUDE_polar_circle_equation_l3351_335174

/-- A circle in a polar coordinate system with radius 1 and center at (1, 0) -/
structure PolarCircle where
  center : ℝ × ℝ := (1, 0)
  radius : ℝ := 1

/-- A point in polar coordinates -/
structure PolarPoint where
  ρ : ℝ
  θ : ℝ

/-- Predicate to check if a point is on the circle -/
def IsOnCircle (c : PolarCircle) (p : PolarPoint) : Prop :=
  p.ρ = 2 * c.radius * Real.cos p.θ

theorem polar_circle_equation (c : PolarCircle) (p : PolarPoint) 
  (h : IsOnCircle c p) : p.ρ = 2 * Real.cos p.θ := by
  sorry

end NUMINAMATH_CALUDE_polar_circle_equation_l3351_335174


namespace NUMINAMATH_CALUDE_consecutive_even_numbers_largest_l3351_335170

theorem consecutive_even_numbers_largest (n : ℕ) : 
  (∀ k : ℕ, k < 7 → ∃ m : ℕ, n + 2*k = 2*m) →  -- 7 consecutive even numbers
  (n + 12 = 3 * n) →                           -- largest is 3 times the smallest
  (n + 12 = 18) :=                             -- largest number is 18
by sorry

end NUMINAMATH_CALUDE_consecutive_even_numbers_largest_l3351_335170


namespace NUMINAMATH_CALUDE_peace_treaty_day_l3351_335107

def day_of_week : Fin 7 → String
| 0 => "Sunday"
| 1 => "Monday"
| 2 => "Tuesday"
| 3 => "Wednesday"
| 4 => "Thursday"
| 5 => "Friday"
| 6 => "Saturday"

def days_between : Nat := 919

theorem peace_treaty_day :
  let start_day : Fin 7 := 4  -- Thursday
  let end_day : Fin 7 := (start_day + days_between) % 7
  day_of_week end_day = "Saturday" := by
  sorry


end NUMINAMATH_CALUDE_peace_treaty_day_l3351_335107


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l3351_335166

theorem arithmetic_mean_problem (x : ℝ) : 
  ((x + 8) + 15 + (2 * x) + 13 + (2 * x + 4) + (3 * x + 5)) / 6 = 30 → x = 13.5 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l3351_335166


namespace NUMINAMATH_CALUDE_arithmetic_sequence_unique_n_l3351_335168

/-- An arithmetic sequence with n terms, where a₁ is the first term and d is the common difference. -/
structure ArithmeticSequence where
  n : ℕ
  a₁ : ℚ
  d : ℚ

/-- The sum of the first k terms of an arithmetic sequence. -/
def sum_first_k (seq : ArithmeticSequence) (k : ℕ) : ℚ :=
  k / 2 * (2 * seq.a₁ + (k - 1) * seq.d)

/-- The sum of the last k terms of an arithmetic sequence. -/
def sum_last_k (seq : ArithmeticSequence) (k : ℕ) : ℚ :=
  k / 2 * (2 * (seq.a₁ + (seq.n - 1) * seq.d) - (k - 1) * seq.d)

/-- The sum of all terms in an arithmetic sequence. -/
def sum_all (seq : ArithmeticSequence) : ℚ :=
  seq.n / 2 * (2 * seq.a₁ + (seq.n - 1) * seq.d)

theorem arithmetic_sequence_unique_n :
  ∀ seq : ArithmeticSequence,
    sum_first_k seq 3 = 34 →
    sum_last_k seq 3 = 146 →
    sum_all seq = 390 →
    seq.n = 11 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_unique_n_l3351_335168


namespace NUMINAMATH_CALUDE_gcd_12012_18018_l3351_335102

theorem gcd_12012_18018 : Nat.gcd 12012 18018 = 6006 := by
  sorry

end NUMINAMATH_CALUDE_gcd_12012_18018_l3351_335102


namespace NUMINAMATH_CALUDE_basketball_lineup_count_l3351_335197

/-- Represents the number of possible line-ups for a basketball team -/
def number_of_lineups (total_players : ℕ) (centers : ℕ) (right_forwards : ℕ) (left_forwards : ℕ) (right_guards : ℕ) (flexible_guards : ℕ) : ℕ :=
  let guard_combinations := flexible_guards * flexible_guards
  guard_combinations * centers * right_forwards * left_forwards

/-- Theorem stating the number of possible line-ups for the given team composition -/
theorem basketball_lineup_count :
  number_of_lineups 10 2 2 2 1 3 = 72 := by
  sorry

end NUMINAMATH_CALUDE_basketball_lineup_count_l3351_335197


namespace NUMINAMATH_CALUDE_power_simplification_l3351_335119

theorem power_simplification : 16^10 * 8^5 / 4^15 = 2^25 := by
  sorry

end NUMINAMATH_CALUDE_power_simplification_l3351_335119


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3351_335124

/-- Given a complex number z = (a+1) - ai where a is real,
    prove that a = -1 is a sufficient but not necessary condition for |z| = 1 -/
theorem sufficient_not_necessary_condition (a : ℝ) :
  let z : ℂ := (a + 1) - a * I
  (a = -1 → Complex.abs z = 1) ∧
  ¬(Complex.abs z = 1 → a = -1) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3351_335124

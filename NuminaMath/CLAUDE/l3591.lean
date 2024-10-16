import Mathlib

namespace NUMINAMATH_CALUDE_batsman_average_theorem_l3591_359193

/-- Represents a batsman's performance -/
structure Batsman where
  innings : ℕ
  totalRuns : ℕ
  lastInningScore : ℕ
  averageIncrease : ℚ

/-- Calculates the new average of a batsman after their latest inning -/
def newAverage (b : Batsman) : ℚ :=
  (b.totalRuns + b.lastInningScore) / b.innings

theorem batsman_average_theorem (b : Batsman) 
  (h1 : b.innings = 17)
  (h2 : b.lastInningScore = 85)
  (h3 : b.averageIncrease = 3)
  (h4 : newAverage b = (b.totalRuns / (b.innings - 1) + b.averageIncrease)) :
  newAverage b = 37 := by
  sorry

end NUMINAMATH_CALUDE_batsman_average_theorem_l3591_359193


namespace NUMINAMATH_CALUDE_max_a_correct_l3591_359173

/-- The inequality x^2 - 4x - a - 1 ≥ 0 has solutions for x ∈ [1, 4] -/
def has_solutions (a : ℝ) : Prop :=
  ∃ x : ℝ, x ∈ Set.Icc 1 4 ∧ x^2 - 4*x - a - 1 ≥ 0

/-- The maximum value of a for which the inequality has solutions -/
def max_a : ℝ := -1

theorem max_a_correct :
  ∀ a : ℝ, has_solutions a ↔ a ≤ max_a :=
by sorry

end NUMINAMATH_CALUDE_max_a_correct_l3591_359173


namespace NUMINAMATH_CALUDE_parallel_vectors_subtraction_l3591_359108

/-- Given vectors a and b in ℝ², where a is parallel to b, prove that 2a - b = (4, -8) -/
theorem parallel_vectors_subtraction (m : ℝ) :
  let a : Fin 2 → ℝ := ![1, -2]
  let b : Fin 2 → ℝ := ![m, 4]
  (∃ (k : ℝ), a = k • b) →
  (2 • a - b) = ![4, -8] := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_subtraction_l3591_359108


namespace NUMINAMATH_CALUDE_road_trip_total_distance_l3591_359152

/-- Represents the road trip with given conditions -/
def RoadTrip (x : ℝ) : Prop :=
  let first_leg := x
  let second_leg := 2 * x
  let third_leg := 40
  let final_leg := 2 * (first_leg + second_leg + third_leg)
  (third_leg = x / 2) ∧
  (first_leg + second_leg + third_leg + final_leg = 840)

/-- Theorem stating the total distance of the road trip -/
theorem road_trip_total_distance : ∃ x : ℝ, RoadTrip x :=
  sorry

end NUMINAMATH_CALUDE_road_trip_total_distance_l3591_359152


namespace NUMINAMATH_CALUDE_red_ball_probability_not_red_ball_probability_l3591_359136

/-- Represents the set of ball colors in the box -/
inductive BallColor
| Red
| White
| Black

/-- Represents the count of balls for each color -/
def ballCount : BallColor → ℕ
| BallColor.Red => 3
| BallColor.White => 5
| BallColor.Black => 7

/-- The total number of balls in the box -/
def totalBalls : ℕ := ballCount BallColor.Red + ballCount BallColor.White + ballCount BallColor.Black

/-- The probability of drawing a ball of a specific color -/
def drawProbability (color : BallColor) : ℚ :=
  ballCount color / totalBalls

theorem red_ball_probability :
  drawProbability BallColor.Red = 1 / 5 := by sorry

theorem not_red_ball_probability :
  1 - drawProbability BallColor.Red = 4 / 5 := by sorry

end NUMINAMATH_CALUDE_red_ball_probability_not_red_ball_probability_l3591_359136


namespace NUMINAMATH_CALUDE_heather_oranges_l3591_359120

def oranges_problem (initial : ℕ) (russell_takes : ℕ) (samantha_takes : ℕ) : Prop :=
  initial - russell_takes - samantha_takes = 13

theorem heather_oranges :
  oranges_problem 60 35 12 := by
  sorry

end NUMINAMATH_CALUDE_heather_oranges_l3591_359120


namespace NUMINAMATH_CALUDE_gcd_lcm_product_90_135_l3591_359165

theorem gcd_lcm_product_90_135 : Nat.gcd 90 135 * Nat.lcm 90 135 = 12150 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_90_135_l3591_359165


namespace NUMINAMATH_CALUDE_midpoint_quadrilateral_area_l3591_359147

/-- A parallelogram in a 2D plane -/
structure Parallelogram where
  area : ℝ

/-- A quadrilateral formed by joining the midpoints of a parallelogram's sides -/
def midpoint_quadrilateral (p : Parallelogram) : Parallelogram :=
  { area := sorry }

/-- The area of the midpoint quadrilateral is 1/4 of the original parallelogram's area -/
theorem midpoint_quadrilateral_area (p : Parallelogram) :
  (midpoint_quadrilateral p).area = p.area / 4 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_quadrilateral_area_l3591_359147


namespace NUMINAMATH_CALUDE_circle_line_intersection_l3591_359155

/-- The necessary and sufficient condition for a circle and a line to have common points -/
theorem circle_line_intersection (k : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 = 1 ∧ y = k*x - 3) ↔ -Real.sqrt 8 ≤ k ∧ k ≤ Real.sqrt 8 := by
  sorry

end NUMINAMATH_CALUDE_circle_line_intersection_l3591_359155


namespace NUMINAMATH_CALUDE_square_of_98_l3591_359128

theorem square_of_98 : (98 : ℕ) ^ 2 = 9604 := by
  sorry

end NUMINAMATH_CALUDE_square_of_98_l3591_359128


namespace NUMINAMATH_CALUDE_farmer_land_ownership_l3591_359130

theorem farmer_land_ownership (total_land : ℝ) : 
  (0.9 * total_land * 0.1 = 90) →
  total_land = 1000 := by
  sorry

end NUMINAMATH_CALUDE_farmer_land_ownership_l3591_359130


namespace NUMINAMATH_CALUDE_pages_to_read_in_third_week_l3591_359191

theorem pages_to_read_in_third_week 
  (total_pages : ℕ) 
  (first_week_fraction : ℚ) 
  (second_week_percent : ℚ) 
  (h1 : total_pages = 600)
  (h2 : first_week_fraction = 1/2)
  (h3 : second_week_percent = 30/100) :
  total_pages - 
  (first_week_fraction * total_pages).floor - 
  (second_week_percent * (total_pages - (first_week_fraction * total_pages).floor)).floor = 210 :=
by
  sorry

end NUMINAMATH_CALUDE_pages_to_read_in_third_week_l3591_359191


namespace NUMINAMATH_CALUDE_max_students_above_average_l3591_359137

theorem max_students_above_average (n : ℕ) (scores : Fin n → ℝ) : 
  n = 80 → (∃ k : ℕ, k ≤ n ∧ k = (Finset.filter (λ i => scores i > (Finset.sum Finset.univ scores) / n) Finset.univ).card) → 
  (∃ k : ℕ, k ≤ n ∧ k = (Finset.filter (λ i => scores i > (Finset.sum Finset.univ scores) / n) Finset.univ).card ∧ k ≤ 79) :=
by sorry

end NUMINAMATH_CALUDE_max_students_above_average_l3591_359137


namespace NUMINAMATH_CALUDE_vector_problem_l3591_359123

theorem vector_problem (a b : Fin 2 → ℝ) (x : ℝ) 
    (h1 : a + b = ![2, x])
    (h2 : a - b = ![-2, 1])
    (h3 : ‖a‖^2 - ‖b‖^2 = -1) : 
  x = 3 := by sorry

end NUMINAMATH_CALUDE_vector_problem_l3591_359123


namespace NUMINAMATH_CALUDE_gcd_490_910_l3591_359171

theorem gcd_490_910 : Nat.gcd 490 910 = 70 := by
  sorry

end NUMINAMATH_CALUDE_gcd_490_910_l3591_359171


namespace NUMINAMATH_CALUDE_stratified_sample_bulbs_l3591_359180

/-- Represents the types of bulbs -/
inductive BulbType
  | W20
  | W40
  | W60

/-- Calculates the number of bulbs of a given type in a sample -/
def sampleSize (totalBulbs : ℕ) (sampleBulbs : ℕ) (ratio : ℕ) (totalRatio : ℕ) : ℕ :=
  (ratio * totalBulbs * sampleBulbs) / (totalRatio * totalBulbs)

theorem stratified_sample_bulbs :
  let totalBulbs : ℕ := 400
  let sampleBulbs : ℕ := 40
  let ratio20W : ℕ := 4
  let ratio40W : ℕ := 3
  let ratio60W : ℕ := 1
  let totalRatio : ℕ := ratio20W + ratio40W + ratio60W
  (sampleSize totalBulbs sampleBulbs ratio20W totalRatio = 20) ∧
  (sampleSize totalBulbs sampleBulbs ratio40W totalRatio = 15) ∧
  (sampleSize totalBulbs sampleBulbs ratio60W totalRatio = 5) :=
by sorry

end NUMINAMATH_CALUDE_stratified_sample_bulbs_l3591_359180


namespace NUMINAMATH_CALUDE_linear_function_not_in_quadrant_I_l3591_359115

/-- A linear function defined by its slope and y-intercept -/
structure LinearFunction where
  slope : ℝ
  yIntercept : ℝ

/-- Defines the four quadrants of the coordinate plane -/
inductive Quadrant
  | I
  | II
  | III
  | IV

/-- Checks if a point (x, y) is in a given quadrant -/
def inQuadrant (x y : ℝ) (q : Quadrant) : Prop :=
  match q with
  | Quadrant.I => x > 0 ∧ y > 0
  | Quadrant.II => x < 0 ∧ y > 0
  | Quadrant.III => x < 0 ∧ y < 0
  | Quadrant.IV => x > 0 ∧ y < 0

/-- Theorem: The graph of y = -2x - 1 does not pass through Quadrant I -/
theorem linear_function_not_in_quadrant_I :
  let f : LinearFunction := { slope := -2, yIntercept := -1 }
  ∀ x y : ℝ, y = f.slope * x + f.yIntercept → ¬(inQuadrant x y Quadrant.I) :=
by
  sorry


end NUMINAMATH_CALUDE_linear_function_not_in_quadrant_I_l3591_359115


namespace NUMINAMATH_CALUDE_total_harvest_l3591_359109

/-- The number of sacks of oranges harvested per day -/
def daily_harvest : ℕ := 83

/-- The number of days of harvest -/
def harvest_days : ℕ := 6

/-- Theorem stating the total number of sacks harvested after 6 days -/
theorem total_harvest : daily_harvest * harvest_days = 498 := by
  sorry

end NUMINAMATH_CALUDE_total_harvest_l3591_359109


namespace NUMINAMATH_CALUDE_rearrangement_theorem_l3591_359189

/-- The number of ways to choose and rearrange 3 people from a group of 7 -/
def rearrangement_count : ℕ := 70

/-- The number of people in the class -/
def class_size : ℕ := 7

/-- The number of people to be rearranged -/
def rearrange_size : ℕ := 3

/-- The number of ways to derange 3 people -/
def derangement_3 : ℕ := 2

theorem rearrangement_theorem : 
  rearrangement_count = derangement_3 * (class_size.choose rearrange_size) := by
  sorry

end NUMINAMATH_CALUDE_rearrangement_theorem_l3591_359189


namespace NUMINAMATH_CALUDE_function_and_value_proof_l3591_359169

noncomputable section

-- Define the function f
def f (A : ℝ) (φ : ℝ) (x : ℝ) : ℝ := A * Real.sin (x + φ)

-- State the theorem
theorem function_and_value_proof 
  (A : ℝ) (φ : ℝ) (α β : ℝ) 
  (h1 : A > 0) 
  (h2 : 0 < φ) (h3 : φ < π) 
  (h4 : ∀ x, f A φ x ≤ 1) 
  (h5 : f A φ (π/3) = 1/2) 
  (h6 : 0 < α) (h7 : α < π/2) 
  (h8 : 0 < β) (h9 : β < π/2) 
  (h10 : f A φ α = 3/5) 
  (h11 : f A φ β = 12/13) :
  (∀ x, f A φ x = Real.cos x) ∧ (f A φ (α - β) = 56/65) := by
  sorry

end

end NUMINAMATH_CALUDE_function_and_value_proof_l3591_359169


namespace NUMINAMATH_CALUDE_fraction_sum_negative_l3591_359148

theorem fraction_sum_negative (a b : ℝ) (h1 : a * b < 0) (h2 : a + b > 0) :
  1 / a + 1 / b < 0 := by
sorry

end NUMINAMATH_CALUDE_fraction_sum_negative_l3591_359148


namespace NUMINAMATH_CALUDE_candy_mixture_cost_per_pound_l3591_359192

/-- Calculates the desired cost per pound of a candy mixture --/
theorem candy_mixture_cost_per_pound 
  (weight_expensive : ℝ) 
  (price_expensive : ℝ) 
  (weight_cheap : ℝ) 
  (price_cheap : ℝ) 
  (h1 : weight_expensive = 20) 
  (h2 : price_expensive = 10) 
  (h3 : weight_cheap = 80) 
  (h4 : price_cheap = 5) : 
  (weight_expensive * price_expensive + weight_cheap * price_cheap) / (weight_expensive + weight_cheap) = 6 := by
  sorry

end NUMINAMATH_CALUDE_candy_mixture_cost_per_pound_l3591_359192


namespace NUMINAMATH_CALUDE_partnership_gain_l3591_359125

/-- Represents the annual gain of a partnership given investments and durations -/
def annual_gain (x : ℝ) : ℝ :=
  let a_investment := x * 12
  let b_investment := 2 * x * 6
  let c_investment := 3 * x * 4
  let total_investment := a_investment + b_investment + c_investment
  let a_share := 6400
  3 * a_share

/-- Theorem stating that the annual gain of the partnership is 19200 -/
theorem partnership_gain : annual_gain x = 19200 :=
sorry

end NUMINAMATH_CALUDE_partnership_gain_l3591_359125


namespace NUMINAMATH_CALUDE_kennel_cat_dog_ratio_l3591_359150

theorem kennel_cat_dog_ratio :
  ∀ (num_dogs num_cats : ℕ),
    num_dogs = 32 →
    num_cats = num_dogs - 8 →
    (num_cats : ℚ) / (num_dogs : ℚ) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_kennel_cat_dog_ratio_l3591_359150


namespace NUMINAMATH_CALUDE_binary_110101101_equals_429_l3591_359122

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_110101101_equals_429 :
  binary_to_decimal [true, false, true, true, false, true, false, true, true] = 429 := by
  sorry

end NUMINAMATH_CALUDE_binary_110101101_equals_429_l3591_359122


namespace NUMINAMATH_CALUDE_smallest_block_size_l3591_359129

/-- Given a rectangular block formed by N congruent 1-cm cubes,
    where 252 cubes are invisible when three faces are viewed,
    the smallest possible value of N is 392. -/
theorem smallest_block_size (N : ℕ) : 
  (∃ l m n : ℕ, 
    l > 0 ∧ m > 0 ∧ n > 0 ∧
    (l - 1) * (m - 1) * (n - 1) = 252 ∧
    N = l * m * n) →
  N ≥ 392 :=
by sorry

end NUMINAMATH_CALUDE_smallest_block_size_l3591_359129


namespace NUMINAMATH_CALUDE_revenue_decrease_percent_l3591_359166

/-- Calculates the decrease percent in revenue when tax is reduced and consumption is increased -/
theorem revenue_decrease_percent 
  (original_tax : ℝ) 
  (original_consumption : ℝ) 
  (tax_reduction_percent : ℝ) 
  (consumption_increase_percent : ℝ) 
  (h1 : tax_reduction_percent = 22) 
  (h2 : consumption_increase_percent = 9) 
  : (1 - (1 - tax_reduction_percent / 100) * (1 + consumption_increase_percent / 100)) * 100 = 15.02 := by
  sorry

end NUMINAMATH_CALUDE_revenue_decrease_percent_l3591_359166


namespace NUMINAMATH_CALUDE_combined_weight_proof_l3591_359131

def combined_weight (mary_weight jamison_weight john_weight peter_weight : ℝ) : ℝ :=
  mary_weight + jamison_weight + john_weight + peter_weight

theorem combined_weight_proof (mary_weight : ℝ) 
  (h1 : mary_weight = 160)
  (h2 : ∃ jamison_weight : ℝ, jamison_weight = mary_weight + 20)
  (h3 : ∃ john_weight : ℝ, john_weight = mary_weight * 1.25)
  (h4 : ∃ peter_weight : ℝ, peter_weight = john_weight * 1.15) :
  ∃ total_weight : ℝ, combined_weight mary_weight 
    (mary_weight + 20) (mary_weight * 1.25) (mary_weight * 1.25 * 1.15) = 770 :=
by
  sorry

end NUMINAMATH_CALUDE_combined_weight_proof_l3591_359131


namespace NUMINAMATH_CALUDE_even_odd_sum_difference_l3591_359100

def sum_even_2_to_40 : ℕ := (20 / 2) * (2 + 40)

def sum_odd_1_to_39 : ℕ := (20 / 2) * (1 + 39)

theorem even_odd_sum_difference : sum_even_2_to_40 - sum_odd_1_to_39 = 20 := by
  sorry

end NUMINAMATH_CALUDE_even_odd_sum_difference_l3591_359100


namespace NUMINAMATH_CALUDE_least_divisor_for_perfect_square_twenty_one_gives_perfect_square_twenty_one_is_least_l3591_359197

def is_perfect_square (x : ℕ) : Prop := ∃ y : ℕ, x = y * y

theorem least_divisor_for_perfect_square : 
  ∀ n : ℕ, n > 0 → is_perfect_square (16800 / n) → n ≥ 21 :=
by sorry

theorem twenty_one_gives_perfect_square : 
  is_perfect_square (16800 / 21) :=
by sorry

theorem twenty_one_is_least :
  ∀ n : ℕ, n > 0 → is_perfect_square (16800 / n) → n = 21 :=
by sorry

end NUMINAMATH_CALUDE_least_divisor_for_perfect_square_twenty_one_gives_perfect_square_twenty_one_is_least_l3591_359197


namespace NUMINAMATH_CALUDE_sum_odd_divisors_300_eq_124_l3591_359172

/-- The sum of all odd divisors of 300 -/
def sum_odd_divisors_300 : ℕ := 124

/-- Theorem: The sum of all odd divisors of 300 is 124 -/
theorem sum_odd_divisors_300_eq_124 : sum_odd_divisors_300 = 124 := by sorry

end NUMINAMATH_CALUDE_sum_odd_divisors_300_eq_124_l3591_359172


namespace NUMINAMATH_CALUDE_average_diff_100_400_50_250_l3591_359168

def average_difference : ℤ → ℤ → ℤ → ℤ → ℤ :=
  fun a b c d => ((b + a) / 2) - ((d + c) / 2)

theorem average_diff_100_400_50_250 :
  average_difference 100 400 50 250 = 100 := by
  sorry

end NUMINAMATH_CALUDE_average_diff_100_400_50_250_l3591_359168


namespace NUMINAMATH_CALUDE_rock_volume_l3591_359199

/-- Calculates the volume of a rock based on the water level rise in a rectangular tank. -/
theorem rock_volume (tank_length tank_width water_rise : ℝ) 
  (h1 : tank_length = 30)
  (h2 : tank_width = 20)
  (h3 : water_rise = 4) :
  tank_length * tank_width * water_rise = 2400 := by
  sorry

#check rock_volume

end NUMINAMATH_CALUDE_rock_volume_l3591_359199


namespace NUMINAMATH_CALUDE_quadratic_root_problem_l3591_359126

theorem quadratic_root_problem (m : ℝ) :
  (1 : ℝ) ^ 2 - 4 * (1 : ℝ) + m + 1 = 0 →
  m = 2 ∧ ∃ (x : ℝ), x ≠ 1 ∧ x ^ 2 - 4 * x + m + 1 = 0 ∧ x = 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_problem_l3591_359126


namespace NUMINAMATH_CALUDE_max_x_value_l3591_359141

theorem max_x_value (x y z : ℝ) 
  (sum_eq : x + y + z = 6) 
  (sum_prod_eq : x*y + x*z + y*z = 10) : 
  x ≤ 2 ∧ ∃ (y z : ℝ), x = 2 ∧ x + y + z = 6 ∧ x*y + x*z + y*z = 10 :=
sorry

end NUMINAMATH_CALUDE_max_x_value_l3591_359141


namespace NUMINAMATH_CALUDE_consecutive_product_square_append_l3591_359177

theorem consecutive_product_square_append (n : ℕ) : ∃ m : ℕ, 100 * (n * (n + 1)) + 25 = m^2 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_product_square_append_l3591_359177


namespace NUMINAMATH_CALUDE_lucinda_jelly_beans_l3591_359196

/-- The number of grape jelly beans Lucinda originally had -/
def original_grape : ℕ := 180

/-- The number of lemon jelly beans Lucinda originally had -/
def original_lemon : ℕ := original_grape / 3

/-- The number of grape jelly beans Lucinda has after gifting -/
def remaining_grape : ℕ := original_grape - 20

/-- The number of lemon jelly beans Lucinda has after gifting -/
def remaining_lemon : ℕ := original_lemon - 20

theorem lucinda_jelly_beans :
  (original_grape = 3 * original_lemon) ∧
  (remaining_grape = 4 * remaining_lemon) →
  original_grape = 180 :=
by sorry

end NUMINAMATH_CALUDE_lucinda_jelly_beans_l3591_359196


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l3591_359101

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_property (a : ℕ → ℝ) 
  (h_geo : geometric_sequence a) 
  (h_prod : a 1 * a 3 * a 11 = 8) : 
  a 2 * a 8 = 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l3591_359101


namespace NUMINAMATH_CALUDE_f_symmetry_l3591_359178

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x^3 - b * x + 1

-- State the theorem
theorem f_symmetry (a b : ℝ) : f a b 2 = -1 → f a b (-2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_f_symmetry_l3591_359178


namespace NUMINAMATH_CALUDE_inscribed_sphere_slant_angle_l3591_359106

/-- A sphere inscribed in a cone with ratio k of tangency circle radius to base radius -/
structure InscribedSphere (k : ℝ) where
  /-- The ratio of the radius of the circle of tangency to the radius of the base of the cone -/
  ratio : k > 0 ∧ k < 1

/-- The cosine of the angle between the slant height and the base of the cone -/
def slant_base_angle_cosine (s : InscribedSphere k) : ℝ := 1 - k

/-- Theorem: The cosine of the angle between the slant height and the base of the cone
    for a sphere inscribed in a cone with ratio k is 1 - k -/
theorem inscribed_sphere_slant_angle (k : ℝ) (s : InscribedSphere k) :
  slant_base_angle_cosine s = 1 - k := by sorry

end NUMINAMATH_CALUDE_inscribed_sphere_slant_angle_l3591_359106


namespace NUMINAMATH_CALUDE_triangle_area_relationship_uncertain_l3591_359157

/-- A triangle with side lengths and area -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  area : ℝ

/-- Proposition: The relationship between areas of two triangles is uncertain -/
theorem triangle_area_relationship_uncertain 
  (ABC : Triangle) (A₁B₁C₁ : Triangle) 
  (h1 : ABC.a > A₁B₁C₁.a) 
  (h2 : ABC.b > A₁B₁C₁.b) 
  (h3 : ABC.c > A₁B₁C₁.c) :
  ¬ (∀ (ABC A₁B₁C₁ : Triangle), 
    ABC.a > A₁B₁C₁.a → ABC.b > A₁B₁C₁.b → ABC.c > A₁B₁C₁.c → 
    (ABC.area > A₁B₁C₁.area ∨ ABC.area < A₁B₁C₁.area ∨ ABC.area = A₁B₁C₁.area)) :=
sorry

end NUMINAMATH_CALUDE_triangle_area_relationship_uncertain_l3591_359157


namespace NUMINAMATH_CALUDE_student_count_problem_l3591_359102

theorem student_count_problem : ∃! n : ℕ, 0 < n ∧ n < 40 ∧ n % 7 = 3 ∧ n % 5 = 1 ∧ n = 31 := by
  sorry

end NUMINAMATH_CALUDE_student_count_problem_l3591_359102


namespace NUMINAMATH_CALUDE_lipstick_cost_l3591_359111

/-- Calculates the cost of each lipstick given the order details -/
theorem lipstick_cost (total_items : ℕ) (num_slippers : ℕ) (slipper_price : ℚ)
  (num_lipsticks : ℕ) (num_hair_colors : ℕ) (hair_color_price : ℚ) (total_paid : ℚ) :
  total_items = num_slippers + num_lipsticks + num_hair_colors →
  total_items = 18 →
  num_slippers = 6 →
  slipper_price = 5/2 →
  num_lipsticks = 4 →
  num_hair_colors = 8 →
  hair_color_price = 3 →
  total_paid = 44 →
  (total_paid - (num_slippers * slipper_price + num_hair_colors * hair_color_price)) / num_lipsticks = 5/4 :=
by sorry

end NUMINAMATH_CALUDE_lipstick_cost_l3591_359111


namespace NUMINAMATH_CALUDE_work_completion_time_l3591_359159

/-- The number of days it takes for worker C to complete the work alone. -/
def c_days : ℚ := 15 / 2

/-- The rate at which worker A completes the work per day. -/
def a_rate : ℚ := 1 / 6

/-- The rate at which worker B completes the work per day. -/
def b_rate : ℚ := 1 / 5

/-- The rate at which worker C completes the work per day. -/
def c_rate : ℚ := 1 / c_days

/-- The combined rate at which workers A, B, and C complete the work per day. -/
def combined_rate : ℚ := a_rate + b_rate + c_rate

theorem work_completion_time :
  a_rate = 1 / 6 →
  b_rate = 1 / 5 →
  combined_rate = 1 / 2 →
  c_days = 15 / 2 := by
  sorry

#eval c_days

end NUMINAMATH_CALUDE_work_completion_time_l3591_359159


namespace NUMINAMATH_CALUDE_initial_sets_count_l3591_359187

/-- The number of letters available (A through J) -/
def n : ℕ := 10

/-- The length of each set of initials -/
def k : ℕ := 3

/-- The number of different three-letter sets of initials possible using letters A through J, with no repetition -/
def num_initial_sets : ℕ := n * (n - 1) * (n - 2)

theorem initial_sets_count : num_initial_sets = 720 := by
  sorry

end NUMINAMATH_CALUDE_initial_sets_count_l3591_359187


namespace NUMINAMATH_CALUDE_junk_items_remaining_l3591_359158

/-- Represents the contents of an attic --/
structure AtticContents where
  total : ℕ
  useful : ℕ
  valuable : ℕ
  junk : ℕ

/-- The initial state of the attic --/
def initial_attic : AtticContents :=
  { total := 100
  , useful := 20
  , valuable := 10
  , junk := 70 }

/-- The number of useful items given away --/
def useful_given_away : ℕ := 4

/-- The number of valuable items sold --/
def valuable_sold : ℕ := 20

/-- The number of useful items remaining after giving some away --/
def useful_remaining : ℕ := 16

/-- Theorem stating the number of junk items remaining in the attic --/
theorem junk_items_remaining (attic : AtticContents) 
  (h1 : attic = initial_attic)
  (h2 : attic.useful = useful_remaining + useful_given_away)
  (h3 : attic.valuable ≤ valuable_sold) : 
  attic.junk - (valuable_sold - attic.valuable) = 60 := by
  sorry


end NUMINAMATH_CALUDE_junk_items_remaining_l3591_359158


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3591_359167

theorem quadratic_equation_solution : 
  ∀ x : ℝ, x^2 - 16 = 0 ↔ x = 4 ∨ x = -4 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3591_359167


namespace NUMINAMATH_CALUDE_nell_gave_jeff_cards_nell_gave_jeff_cards_proof_l3591_359156

/-- Given that Nell initially had 304 baseball cards and now has 276 cards left,
    prove that she gave 28 cards to Jeff. -/
theorem nell_gave_jeff_cards : ℕ → ℕ → ℕ → Prop :=
  fun initial_cards remaining_cards cards_given =>
    initial_cards = 304 →
    remaining_cards = 276 →
    cards_given = initial_cards - remaining_cards →
    cards_given = 28

/-- Proof of the theorem -/
theorem nell_gave_jeff_cards_proof : nell_gave_jeff_cards 304 276 28 := by
  sorry

end NUMINAMATH_CALUDE_nell_gave_jeff_cards_nell_gave_jeff_cards_proof_l3591_359156


namespace NUMINAMATH_CALUDE_factorization_ab_minus_a_l3591_359135

theorem factorization_ab_minus_a (a b : ℝ) : a * b - a = a * (b - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_ab_minus_a_l3591_359135


namespace NUMINAMATH_CALUDE_decimal_places_theorem_l3591_359146

def first_1000_decimal_places (x : ℝ) : List ℕ :=
  sorry

theorem decimal_places_theorem :
  (∀ d ∈ first_1000_decimal_places ((6 + Real.sqrt 35) ^ 1999), d = 9) ∧
  (∀ d ∈ first_1000_decimal_places ((6 + Real.sqrt 37) ^ 1999), d = 0) ∧
  (∀ d ∈ first_1000_decimal_places ((6 + Real.sqrt 37) ^ 2000), d = 9) :=
by sorry

end NUMINAMATH_CALUDE_decimal_places_theorem_l3591_359146


namespace NUMINAMATH_CALUDE_min_k_for_sqrt_inequality_l3591_359119

theorem min_k_for_sqrt_inequality : 
  ∃ k : ℝ, k = Real.sqrt 2 ∧ 
  (∀ x y : ℝ, Real.sqrt x + Real.sqrt y ≤ k * Real.sqrt (x + y)) ∧
  (∀ k' : ℝ, k' < k → 
    ∃ x y : ℝ, Real.sqrt x + Real.sqrt y > k' * Real.sqrt (x + y)) := by
  sorry

end NUMINAMATH_CALUDE_min_k_for_sqrt_inequality_l3591_359119


namespace NUMINAMATH_CALUDE_triangle_is_right_angle_l3591_359140

theorem triangle_is_right_angle (a b c : ℝ) : 
  a = 3 ∧ b = 4 ∧ c^2 - 10*c + 25 = 0 → c^2 = a^2 + b^2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_is_right_angle_l3591_359140


namespace NUMINAMATH_CALUDE_birthday_cookies_l3591_359145

theorem birthday_cookies (friends : ℕ) (packages : ℕ) (cookies_per_package : ℕ) :
  friends = 4 →
  packages = 3 →
  cookies_per_package = 25 →
  (packages * cookies_per_package) / (friends + 1) = 15 :=
by sorry

end NUMINAMATH_CALUDE_birthday_cookies_l3591_359145


namespace NUMINAMATH_CALUDE_rivertown_puzzle_l3591_359151

theorem rivertown_puzzle (p h s c d : ℕ) : 
  p = 4 * h →
  s = 5 * c →
  d = 4 * p →
  ¬ ∃ (h c : ℕ), 99 = 21 * h + 6 * c :=
by sorry

end NUMINAMATH_CALUDE_rivertown_puzzle_l3591_359151


namespace NUMINAMATH_CALUDE_pablos_payment_per_page_l3591_359163

/-- The amount Pablo's mother pays him per page, in dollars. -/
def payment_per_page : ℚ := 1 / 100

/-- The number of pages in each book Pablo reads. -/
def pages_per_book : ℕ := 150

/-- The number of books Pablo read. -/
def books_read : ℕ := 12

/-- The amount Pablo spent on candy, in dollars. -/
def candy_cost : ℕ := 15

/-- The amount Pablo had leftover, in dollars. -/
def leftover : ℕ := 3

theorem pablos_payment_per_page :
  payment_per_page * (pages_per_book * books_read : ℚ) = (candy_cost + leftover : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_pablos_payment_per_page_l3591_359163


namespace NUMINAMATH_CALUDE_comprehensive_survey_appropriate_for_grade9_vision_l3591_359162

/-- Represents the appropriateness of a survey method for a given scenario -/
inductive SurveyAppropriateness
  | Appropriate
  | Inappropriate

/-- Represents different survey methods -/
inductive SurveyMethod
  | Comprehensive
  | Sample

/-- Represents characteristics that can be surveyed -/
inductive Characteristic
  | Vision
  | EquipmentQuality

/-- Represents the size of a group being surveyed -/
inductive GroupSize
  | Large
  | Small

/-- Function to determine if a survey method is appropriate for a given characteristic and group size -/
def is_appropriate (method : SurveyMethod) (char : Characteristic) (size : GroupSize) : SurveyAppropriateness :=
  match method, char, size with
  | SurveyMethod.Comprehensive, Characteristic.Vision, GroupSize.Large => SurveyAppropriateness.Appropriate
  | _, _, _ => SurveyAppropriateness.Inappropriate

theorem comprehensive_survey_appropriate_for_grade9_vision :
  is_appropriate SurveyMethod.Comprehensive Characteristic.Vision GroupSize.Large = SurveyAppropriateness.Appropriate :=
by sorry

end NUMINAMATH_CALUDE_comprehensive_survey_appropriate_for_grade9_vision_l3591_359162


namespace NUMINAMATH_CALUDE_percentage_problem_l3591_359112

theorem percentage_problem (x : ℝ) : 
  (20 / 100 * 40) + (x / 100 * 60) = 23 ↔ x = 25 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3591_359112


namespace NUMINAMATH_CALUDE_expand_difference_of_squares_simplify_fraction_l3591_359181

-- Define a as a real number
variable (a : ℝ)

-- Theorem 1: (a+2)(a-2) = a^2 - 4
theorem expand_difference_of_squares : (a + 2) * (a - 2) = a^2 - 4 := by
  sorry

-- Theorem 2: (a^2-4)/(a+2) + 2 = a
theorem simplify_fraction : (a^2 - 4) / (a + 2) + 2 = a := by
  sorry

end NUMINAMATH_CALUDE_expand_difference_of_squares_simplify_fraction_l3591_359181


namespace NUMINAMATH_CALUDE_intersection_complement_theorem_l3591_359103

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 2, 3}
def B : Set Nat := {3, 4}

theorem intersection_complement_theorem :
  A ∩ (U \ B) = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_theorem_l3591_359103


namespace NUMINAMATH_CALUDE_inequality_proof_l3591_359170

theorem inequality_proof (x y z : ℝ) : 
  x^2 / (x^2 + 2*y*z) + y^2 / (y^2 + 2*z*x) + z^2 / (z^2 + 2*x*y) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3591_359170


namespace NUMINAMATH_CALUDE_circle_center_coordinate_product_l3591_359113

/-- Given a circle with equation x^2 + y^2 = 6x + 10y - 14, 
    the product of its center coordinates is 15 -/
theorem circle_center_coordinate_product : 
  ∀ (h k : ℝ), (∀ x y : ℝ, x^2 + y^2 = 6*x + 10*y - 14 → (x - h)^2 + (y - k)^2 = 20) → 
  h * k = 15 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_coordinate_product_l3591_359113


namespace NUMINAMATH_CALUDE_temperature_difference_l3591_359186

theorem temperature_difference (highest lowest : ℤ) (h1 : highest = -9) (h2 : lowest = -22) :
  highest - lowest = 13 := by
  sorry

end NUMINAMATH_CALUDE_temperature_difference_l3591_359186


namespace NUMINAMATH_CALUDE_age_ratio_l3591_359105

/-- Represents the ages of Sam, Sue, and Kendra -/
structure Ages where
  sam : ℕ
  sue : ℕ
  kendra : ℕ

/-- The conditions of the problem -/
def satisfiesConditions (ages : Ages) : Prop :=
  ages.sam = 2 * ages.sue ∧
  ages.kendra = 18 ∧
  ages.sam + ages.sue + ages.kendra + 9 = 36

/-- The theorem to prove -/
theorem age_ratio (ages : Ages) (h : satisfiesConditions ages) : 
  ages.kendra / ages.sam = 3 := by
  sorry

/-- Auxiliary lemma to help with division -/
lemma div_eq_of_mul_eq {a b c : ℕ} (hb : b ≠ 0) (h : a = b * c) : a / b = c := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_l3591_359105


namespace NUMINAMATH_CALUDE_delta_value_l3591_359194

theorem delta_value : ∃ Δ : ℤ, (5 * (-3) = Δ - 3) → (Δ = -12) := by
  sorry

end NUMINAMATH_CALUDE_delta_value_l3591_359194


namespace NUMINAMATH_CALUDE_prob_non_red_twelve_sided_l3591_359149

/-- Represents a 12-sided die with colored faces -/
structure ColoredDie where
  total_faces : ℕ
  red_faces : ℕ
  yellow_faces : ℕ
  blue_faces : ℕ
  green_faces : ℕ
  face_sum : total_faces = red_faces + yellow_faces + blue_faces + green_faces

/-- The probability of rolling a non-red face on the given die -/
def prob_non_red (d : ColoredDie) : ℚ :=
  (d.total_faces - d.red_faces : ℚ) / d.total_faces

/-- The specific 12-sided die described in the problem -/
def twelve_sided_die : ColoredDie where
  total_faces := 12
  red_faces := 5
  yellow_faces := 4
  blue_faces := 2
  green_faces := 1
  face_sum := by rfl

theorem prob_non_red_twelve_sided : prob_non_red twelve_sided_die = 7/12 := by
  sorry

end NUMINAMATH_CALUDE_prob_non_red_twelve_sided_l3591_359149


namespace NUMINAMATH_CALUDE_red_balloons_count_l3591_359139

/-- Proves that the total number of red balloons after destruction is 40 -/
theorem red_balloons_count (fred_balloons sam_balloons dan_destroyed : ℝ) 
  (h1 : fred_balloons = 10)
  (h2 : sam_balloons = 46)
  (h3 : dan_destroyed = 16) :
  fred_balloons + sam_balloons - dan_destroyed = 40 := by
  sorry

#check red_balloons_count

end NUMINAMATH_CALUDE_red_balloons_count_l3591_359139


namespace NUMINAMATH_CALUDE_initial_student_count_l3591_359184

theorem initial_student_count (initial_avg : ℝ) (new_avg : ℝ) (dropped_score : ℝ) :
  initial_avg = 62.5 →
  new_avg = 62.0 →
  dropped_score = 70 →
  ∃ n : ℕ, n > 0 ∧ 
    (n : ℝ) * initial_avg = ((n - 1) : ℝ) * new_avg + dropped_score ∧
    n = 16 :=
by sorry

end NUMINAMATH_CALUDE_initial_student_count_l3591_359184


namespace NUMINAMATH_CALUDE_reading_time_calculation_l3591_359160

/-- Represents the reading speed in sentences per hour -/
def reading_speed : ℕ := 200

/-- Represents the number of paragraphs per page -/
def paragraphs_per_page : ℕ := 20

/-- Represents the number of sentences per paragraph -/
def sentences_per_paragraph : ℕ := 10

/-- Represents the total number of pages in the book -/
def total_pages : ℕ := 50

/-- Calculates the total reading time in hours -/
def total_reading_time : ℚ :=
  (total_pages * paragraphs_per_page * sentences_per_paragraph) / reading_speed

theorem reading_time_calculation :
  total_reading_time = 50 := by sorry

end NUMINAMATH_CALUDE_reading_time_calculation_l3591_359160


namespace NUMINAMATH_CALUDE_count_with_zero_up_to_3500_l3591_359116

/-- Counts the number of integers from 1 to n that contain the digit 0 in base 10 -/
def count_with_zero (n : ℕ) : ℕ := sorry

/-- The main theorem stating that there are 773 numbers containing 0 up to 3500 -/
theorem count_with_zero_up_to_3500 : count_with_zero 3500 = 773 := by sorry

end NUMINAMATH_CALUDE_count_with_zero_up_to_3500_l3591_359116


namespace NUMINAMATH_CALUDE_compare_cubic_and_mixed_terms_l3591_359154

theorem compare_cubic_and_mixed_terms {a b : ℝ} (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  a^3 + b^3 > a^2 * b + a * b^2 := by
  sorry

end NUMINAMATH_CALUDE_compare_cubic_and_mixed_terms_l3591_359154


namespace NUMINAMATH_CALUDE_spring_mass_for_length_30_l3591_359182

def spring_length (mass : ℝ) : ℝ := 18 + 2 * mass

theorem spring_mass_for_length_30 :
  ∃ (mass : ℝ), spring_length mass = 30 ∧ mass = 6 :=
by sorry

end NUMINAMATH_CALUDE_spring_mass_for_length_30_l3591_359182


namespace NUMINAMATH_CALUDE_calculate_expression_l3591_359179

theorem calculate_expression : (π - 1) ^ 0 + 4 * Real.sin (π / 4) - Real.sqrt 8 + |(-3)| = 4 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l3591_359179


namespace NUMINAMATH_CALUDE_mike_work_hours_l3591_359117

def wash_time : ℕ := 10
def oil_change_time : ℕ := 15
def tire_change_time : ℕ := 30
def paint_time : ℕ := 45
def engine_service_time : ℕ := 60

def cars_washed : ℕ := 9
def cars_oil_changed : ℕ := 6
def tire_sets_changed : ℕ := 2
def cars_painted : ℕ := 4
def engines_serviced : ℕ := 3

def total_minutes : ℕ := 
  wash_time * cars_washed + 
  oil_change_time * cars_oil_changed + 
  tire_change_time * tire_sets_changed + 
  paint_time * cars_painted + 
  engine_service_time * engines_serviced

theorem mike_work_hours : total_minutes / 60 = 10 := by
  sorry

end NUMINAMATH_CALUDE_mike_work_hours_l3591_359117


namespace NUMINAMATH_CALUDE_eggs_from_gertrude_l3591_359164

/-- Represents the number of eggs collected from each chicken -/
structure EggCollection where
  gertrude : ℕ
  blanche : ℕ
  nancy : ℕ
  martha : ℕ

/-- The theorem stating the number of eggs Trevor got from Gertrude -/
theorem eggs_from_gertrude (collection : EggCollection) : 
  collection.blanche = 3 →
  collection.nancy = 2 →
  collection.martha = 2 →
  collection.gertrude + collection.blanche + collection.nancy + collection.martha = 11 →
  collection.gertrude = 4 := by
  sorry

#check eggs_from_gertrude

end NUMINAMATH_CALUDE_eggs_from_gertrude_l3591_359164


namespace NUMINAMATH_CALUDE_divisibility_of_7_power_minus_1_l3591_359183

theorem divisibility_of_7_power_minus_1 : ∃ k : ℤ, 7^51 - 1 = 103 * k := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_7_power_minus_1_l3591_359183


namespace NUMINAMATH_CALUDE_octagon_dissection_and_reassembly_l3591_359132

/-- Represents a regular octagon -/
structure RegularOctagon where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Represents a section of a regular octagon -/
structure OctagonSection where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Checks if two OctagonSections are similar -/
def are_similar (s1 s2 : OctagonSection) : Prop :=
  sorry

/-- Checks if two RegularOctagons are congruent -/
def are_congruent (o1 o2 : RegularOctagon) : Prop :=
  sorry

/-- Represents the dissection of a RegularOctagon into OctagonSections -/
def dissect (o : RegularOctagon) : List OctagonSection :=
  sorry

/-- Represents the reassembly of OctagonSections into RegularOctagons -/
def reassemble (sections : List OctagonSection) : List RegularOctagon :=
  sorry

theorem octagon_dissection_and_reassembly 
  (o : RegularOctagon) : 
  let sections := dissect o
  ∃ (reassembled : List RegularOctagon),
    (reassembled = reassemble sections) ∧ 
    (sections.length = 8) ∧
    (∀ (s1 s2 : OctagonSection), s1 ∈ sections → s2 ∈ sections → are_similar s1 s2) ∧
    (reassembled.length = 8) ∧
    (∀ (o1 o2 : RegularOctagon), o1 ∈ reassembled → o2 ∈ reassembled → are_congruent o1 o2) :=
by
  sorry

end NUMINAMATH_CALUDE_octagon_dissection_and_reassembly_l3591_359132


namespace NUMINAMATH_CALUDE_carolyns_essay_body_sections_l3591_359118

/-- Represents the structure of Carolyn's essay -/
structure EssayStructure where
  intro_length : ℕ
  conclusion_length : ℕ
  body_section_length : ℕ
  total_length : ℕ

/-- Calculates the number of body sections in Carolyn's essay -/
def calculate_body_sections (essay : EssayStructure) : ℕ :=
  let remaining_length := essay.total_length - (essay.intro_length + essay.conclusion_length)
  remaining_length / essay.body_section_length

/-- Theorem stating that Carolyn's essay has 4 body sections -/
theorem carolyns_essay_body_sections :
  let essay := EssayStructure.mk 450 (3 * 450) 800 5000
  calculate_body_sections essay = 4 := by
  sorry

end NUMINAMATH_CALUDE_carolyns_essay_body_sections_l3591_359118


namespace NUMINAMATH_CALUDE_negation_equivalence_l3591_359142

-- Define the lines l₁ and l₂
def l₁ (a x y : ℝ) : Prop := x + a * y + 1 = 0
def l₂ (a x y : ℝ) : Prop := a * x + y + 2 = 0

-- Define when two lines are parallel
def parallel (a : ℝ) : Prop := ∀ x y, l₁ a x y ↔ l₂ a x y

-- State the theorem
theorem negation_equivalence :
  ¬(((a = 1) ∨ (a = -1)) → parallel a) ↔ 
  ((a ≠ 1) ∧ (a ≠ -1)) → ¬(parallel a) :=
sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3591_359142


namespace NUMINAMATH_CALUDE_square_fence_perimeter_l3591_359127

/-- The outer perimeter of a square fence with evenly spaced posts -/
theorem square_fence_perimeter
  (num_posts : ℕ)
  (post_width_inches : ℕ)
  (gap_between_posts_feet : ℕ)
  (h1 : num_posts = 16)
  (h2 : post_width_inches = 6)
  (h3 : gap_between_posts_feet = 4) :
  (4 * (↑num_posts / 4 * (↑post_width_inches / 12 + ↑gap_between_posts_feet) - ↑gap_between_posts_feet)) = 56 :=
by sorry

end NUMINAMATH_CALUDE_square_fence_perimeter_l3591_359127


namespace NUMINAMATH_CALUDE_f_neg_two_eq_neg_two_fifths_l3591_359107

noncomputable def g (x : ℝ) : ℝ := 3 - x^2

noncomputable def f (x : ℝ) : ℝ := 
  if x = 0 then 0 else (3 - (g⁻¹ x)^2) / (g⁻¹ x)^2

theorem f_neg_two_eq_neg_two_fifths : f (-2) = -2/5 := by
  sorry

end NUMINAMATH_CALUDE_f_neg_two_eq_neg_two_fifths_l3591_359107


namespace NUMINAMATH_CALUDE_envelope_length_l3591_359114

/-- Given a rectangular envelope with width 4 inches and area 16 square inches,
    prove that its length is 4 inches. -/
theorem envelope_length (width : ℝ) (area : ℝ) (length : ℝ) : 
  width = 4 → area = 16 → area = width * length → length = 4 := by
  sorry

end NUMINAMATH_CALUDE_envelope_length_l3591_359114


namespace NUMINAMATH_CALUDE_triangle_area_with_median_l3591_359198

/-- Given a triangle with two sides of length 1 and √15, and a median of length 2 to the third side,
    the area of the triangle is √15/2. -/
theorem triangle_area_with_median (a b c : ℝ) (m : ℝ) (h1 : a = 1) (h2 : b = Real.sqrt 15) (h3 : m = 2)
    (hm : m^2 = (2*a^2 + 2*b^2 - c^2) / 4) : (a * b) / 2 = Real.sqrt 15 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_with_median_l3591_359198


namespace NUMINAMATH_CALUDE_probability_perfect_square_three_digit_l3591_359144

/-- A three-digit number is a natural number between 100 and 999, inclusive. -/
def ThreeDigitNumber (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- A perfect square is a natural number that is the square of an integer. -/
def PerfectSquare (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

/-- The count of three-digit numbers that are perfect squares. -/
def CountPerfectSquareThreeDigit : ℕ := 22

/-- The total count of three-digit numbers. -/
def TotalThreeDigitNumbers : ℕ := 900

/-- The probability of a randomly chosen three-digit number being a perfect square is 11/450. -/
theorem probability_perfect_square_three_digit :
  (CountPerfectSquareThreeDigit : ℚ) / (TotalThreeDigitNumbers : ℚ) = 11 / 450 := by
  sorry

end NUMINAMATH_CALUDE_probability_perfect_square_three_digit_l3591_359144


namespace NUMINAMATH_CALUDE_sqrt_of_sqrt_16_over_81_l3591_359121

theorem sqrt_of_sqrt_16_over_81 : Real.sqrt (Real.sqrt (16 / 81)) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_sqrt_16_over_81_l3591_359121


namespace NUMINAMATH_CALUDE_swimmers_second_meeting_time_l3591_359110

theorem swimmers_second_meeting_time
  (pool_length : ℝ)
  (henry_speed : ℝ)
  (george_speed : ℝ)
  (first_meeting_time : ℝ)
  (h1 : pool_length = 100)
  (h2 : george_speed = 2 * henry_speed)
  (h3 : first_meeting_time = 1)
  (h4 : henry_speed * first_meeting_time + george_speed * first_meeting_time = pool_length) :
  let second_meeting_time := 2 * first_meeting_time
  ∃ (distance_henry distance_george : ℝ),
    distance_henry + distance_george = pool_length ∧
    distance_henry = henry_speed * second_meeting_time ∧
    distance_george = george_speed * second_meeting_time :=
by sorry


end NUMINAMATH_CALUDE_swimmers_second_meeting_time_l3591_359110


namespace NUMINAMATH_CALUDE_orthogonal_lines_sweep_l3591_359176

-- Define the circle S
def S (a : ℝ) : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 ≤ a^2}

-- Define the point O outside the circle
def O (x₀ y₀ : ℝ) : ℝ × ℝ := (x₀, y₀)

-- Define a point X inside the circle
def X (c : ℝ) (a : ℝ) : ℝ × ℝ := (c, 0)

-- Define the set of points swept by lines l
def swept_points (a c : ℝ) : Set (ℝ × ℝ) :=
  {p | (c^2 - a^2) * p.1^2 - a^2 * p.2^2 ≤ a^2 * (c^2 - a^2)}

-- State the theorem
theorem orthogonal_lines_sweep (a : ℝ) (x₀ y₀ : ℝ) (h₁ : a > 0) (h₂ : x₀^2 + y₀^2 ≠ a^2) :
  ∀ c, c^2 < a^2 →
    swept_points a c =
    {p | ∃ (X : ℝ × ℝ), X ∈ S a ∧ (p.1 - X.1) * (O x₀ y₀).1 + (p.2 - X.2) * (O x₀ y₀).2 = 0} :=
by sorry

end NUMINAMATH_CALUDE_orthogonal_lines_sweep_l3591_359176


namespace NUMINAMATH_CALUDE_rectangle_area_diagonal_l3591_359143

/-- Given a rectangle with length-to-width ratio of 5:2 and diagonal d, 
    prove that its area A can be expressed as A = kd^2, where k = 10/29 -/
theorem rectangle_area_diagonal (d : ℝ) (h : d > 0) : ∃ (l w : ℝ),
  l > 0 ∧ w > 0 ∧ l / w = 5 / 2 ∧ l ^ 2 + w ^ 2 = d ^ 2 ∧ l * w = (10 / 29) * d ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_diagonal_l3591_359143


namespace NUMINAMATH_CALUDE_perfect_square_condition_l3591_359188

/-- The polynomial in question -/
def P (x m : ℝ) : ℝ := (x-1)*(x+3)*(x-4)*(x-8) + m

/-- The polynomial is a perfect square -/
def is_perfect_square (f : ℝ → ℝ) : Prop :=
  ∃ g : ℝ → ℝ, ∀ x, f x = (g x)^2

theorem perfect_square_condition :
  ∃! m : ℝ, is_perfect_square (P · m) ∧ m = 196 :=
sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l3591_359188


namespace NUMINAMATH_CALUDE_square_side_length_l3591_359161

/-- A square with four identical isosceles triangles on its sides -/
structure SquareWithTriangles where
  /-- Side length of the square -/
  s : ℝ
  /-- Area of one isosceles triangle -/
  triangle_area : ℝ
  /-- The total area of the isosceles triangles equals the area of the remaining region -/
  area_equality : 4 * triangle_area = s^2 - 4 * triangle_area
  /-- The distance between the apexes of two opposite isosceles triangles is 12 -/
  apex_distance : s + 2 * (triangle_area / s) = 12

/-- Theorem: The side length of the square is 24 -/
theorem square_side_length (sq : SquareWithTriangles) : sq.s = 24 :=
sorry

end NUMINAMATH_CALUDE_square_side_length_l3591_359161


namespace NUMINAMATH_CALUDE_john_duck_profit_l3591_359174

/-- Calculates the profit from selling ducks given the following conditions:
  * number_of_ducks: The number of ducks bought and sold
  * cost_per_duck: The cost of each duck when buying
  * weight_per_duck: The weight of each duck in pounds
  * price_per_pound: The selling price per pound of duck
-/
def duck_profit (number_of_ducks : ℕ) (cost_per_duck : ℚ) (weight_per_duck : ℚ) (price_per_pound : ℚ) : ℚ :=
  let total_cost := number_of_ducks * cost_per_duck
  let revenue_per_duck := weight_per_duck * price_per_pound
  let total_revenue := number_of_ducks * revenue_per_duck
  total_revenue - total_cost

/-- Theorem stating that under the given conditions, the profit is $300 -/
theorem john_duck_profit :
  duck_profit 30 10 4 5 = 300 := by
  sorry

end NUMINAMATH_CALUDE_john_duck_profit_l3591_359174


namespace NUMINAMATH_CALUDE_magic_square_theorem_l3591_359133

/-- A type representing a 3x3 grid -/
def Grid := Fin 3 → Fin 3 → ℤ

/-- The set of numbers to be used in the grid -/
def GridNumbers : Finset ℤ := {-3, -2, -1, 0, 1, 2, 3, 4, 5}

/-- The sum of each row, column, and diagonal is equal -/
def is_magic (g : Grid) : Prop :=
  let sum := g 0 0 + g 0 1 + g 0 2
  ∀ i j, (i = j → g i 0 + g i 1 + g i 2 = sum) ∧
         (i = j → g 0 j + g 1 j + g 2 j = sum) ∧
         ((i = 0 ∧ j = 0) → g 0 0 + g 1 1 + g 2 2 = sum) ∧
         ((i = 0 ∧ j = 2) → g 0 2 + g 1 1 + g 2 0 = sum)

/-- The theorem to be proved -/
theorem magic_square_theorem (g : Grid) 
  (h1 : g 0 0 = -2)
  (h2 : g 0 2 = 0)
  (h3 : g 2 2 = 4)
  (h4 : is_magic g)
  (h5 : ∀ i j, g i j ∈ GridNumbers)
  (h6 : ∀ x, x ∈ GridNumbers → ∃! i j, g i j = x) :
  ∃ a b c, g 0 1 = a ∧ g 2 1 = b ∧ g 2 0 = c ∧ a - b - c = 4 :=
sorry

end NUMINAMATH_CALUDE_magic_square_theorem_l3591_359133


namespace NUMINAMATH_CALUDE_train_crossing_time_l3591_359134

/-- Proves that a train with given length and speed takes the specified time to cross an electric pole -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 120 →
  train_speed_kmh = 144 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) →
  crossing_time = 3 := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l3591_359134


namespace NUMINAMATH_CALUDE_not_divisible_by_1000_power_minus_1_l3591_359175

theorem not_divisible_by_1000_power_minus_1 (m : ℕ) :
  ¬(1000^m - 1 ∣ 1978^m - 1) := by
sorry

end NUMINAMATH_CALUDE_not_divisible_by_1000_power_minus_1_l3591_359175


namespace NUMINAMATH_CALUDE_optimal_garden_max_area_l3591_359185

/-- Represents a rectangular garden with given constraints --/
structure Garden where
  length : ℝ
  width : ℝ
  perimeter_constraint : length + width = 200
  length_min : length ≥ 100
  width_min : width ≥ 50
  length_width_diff : length ≥ width + 20

/-- The area of a garden --/
def garden_area (g : Garden) : ℝ := g.length * g.width

/-- The optimal garden dimensions and area --/
def optimal_garden : Garden := {
  length := 120,
  width := 80,
  perimeter_constraint := by sorry,
  length_min := by sorry,
  width_min := by sorry,
  length_width_diff := by sorry
}

/-- Theorem stating that the optimal garden has the maximum area --/
theorem optimal_garden_max_area :
  ∀ g : Garden, garden_area g ≤ garden_area optimal_garden := by sorry

end NUMINAMATH_CALUDE_optimal_garden_max_area_l3591_359185


namespace NUMINAMATH_CALUDE_extremum_point_monotonicity_positive_when_m_leq_2_l3591_359104

-- Define the function f(x)
noncomputable def f (x m : ℝ) : ℝ := Real.exp x - Real.log (x + m)

-- Theorem for the extremum point condition
theorem extremum_point (m : ℝ) : 
  (∃ ε > 0, ∀ x ∈ Set.Ioo (-ε) ε, f x m ≥ f 0 m ∨ f x m ≤ f 0 m) → 
  (deriv (f · m)) 0 = 0 := 
sorry

-- Theorem for monotonicity of f(x)
theorem monotonicity (m : ℝ) : 
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → (f x₁ m < f x₂ m ∨ f x₁ m > f x₂ m) := 
sorry

-- Theorem for f(x) > 0 when m ≤ 2
theorem positive_when_m_leq_2 (x m : ℝ) : 
  m ≤ 2 → f x m > 0 := 
sorry

end NUMINAMATH_CALUDE_extremum_point_monotonicity_positive_when_m_leq_2_l3591_359104


namespace NUMINAMATH_CALUDE_book_pages_proof_l3591_359124

theorem book_pages_proof (x : ℝ) : 
  let day1_remaining := x - (x / 6 + 10)
  let day2_remaining := day1_remaining - (day1_remaining / 3 + 20)
  let day3_remaining := day2_remaining - (day2_remaining / 2 + 25)
  day3_remaining = 120 → x = 552 := by
sorry

end NUMINAMATH_CALUDE_book_pages_proof_l3591_359124


namespace NUMINAMATH_CALUDE_inequality_proof_l3591_359153

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  a^(1/3) * b^(1/3) + c^(1/3) * d^(1/3) ≤ (a+b+c)^(1/3) * (a+c+d)^(1/3) ∧
  (a^(1/3) * b^(1/3) + c^(1/3) * d^(1/3) = (a+b+c)^(1/3) * (a+c+d)^(1/3) ↔
   b = (a/c)*(a+c) ∧ d = (c/a)*(a+c)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3591_359153


namespace NUMINAMATH_CALUDE_three_heads_in_eight_tosses_l3591_359190

/-- The probability of getting exactly k heads in n tosses of a fair coin -/
def probability_k_heads (n k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) / 2^n

/-- The probability of getting exactly 3 heads in 8 tosses of a fair coin is 7/32 -/
theorem three_heads_in_eight_tosses :
  probability_k_heads 8 3 = 7 / 32 := by
  sorry

end NUMINAMATH_CALUDE_three_heads_in_eight_tosses_l3591_359190


namespace NUMINAMATH_CALUDE_water_bills_theorem_l3591_359138

/-- Water pricing structure -/
def water_price (usage : ℕ) : ℚ :=
  if usage ≤ 10 then 0.45 * usage
  else if usage ≤ 20 then 0.45 * 10 + 0.80 * (usage - 10)
  else 0.45 * 10 + 0.80 * 10 + 1.50 * (usage - 20)

/-- Theorem stating the water bills for households A, B, and C -/
theorem water_bills_theorem :
  ∃ (usage_A usage_B usage_C : ℕ),
    usage_A > 20 ∧ 
    10 < usage_B ∧ usage_B ≤ 20 ∧
    usage_C ≤ 10 ∧
    water_price usage_A - water_price usage_B = 7.10 ∧
    water_price usage_B - water_price usage_C = 3.75 ∧
    water_price usage_A = 14 ∧
    water_price usage_B = 6.9 ∧
    water_price usage_C = 3.15 :=
by sorry

end NUMINAMATH_CALUDE_water_bills_theorem_l3591_359138


namespace NUMINAMATH_CALUDE_sum_of_circle_areas_in_5_12_13_triangle_l3591_359195

/-- Represents a circle with a given radius -/
structure Circle where
  radius : ℝ
  radius_pos : radius > 0

/-- Represents a right triangle with circles at its vertices -/
structure TriangleWithCircles where
  side1 : ℝ
  side2 : ℝ
  hypotenuse : ℝ
  circle1 : Circle
  circle2 : Circle
  circle3 : Circle
  is_right_triangle : side1^2 + side2^2 = hypotenuse^2
  circles_tangent : 
    circle1.radius + circle2.radius = side1 ∧
    circle2.radius + circle3.radius = side2 ∧
    circle1.radius + circle3.radius = hypotenuse

/-- The sum of the areas of the circles in a 5-12-13 right triangle with mutually tangent circles at its vertices is 81π -/
theorem sum_of_circle_areas_in_5_12_13_triangle (t : TriangleWithCircles) 
  (h1 : t.side1 = 5) (h2 : t.side2 = 12) (h3 : t.hypotenuse = 13) : 
  π * t.circle1.radius^2 + π * t.circle2.radius^2 + π * t.circle3.radius^2 = 81 * π := by
  sorry

end NUMINAMATH_CALUDE_sum_of_circle_areas_in_5_12_13_triangle_l3591_359195

import Mathlib

namespace NUMINAMATH_CALUDE_product_identity_l2411_241109

theorem product_identity (x y : ℝ) : (x + y^2) * (x - y^2) * (x^2 + y^4) = x^4 - y^8 := by
  sorry

end NUMINAMATH_CALUDE_product_identity_l2411_241109


namespace NUMINAMATH_CALUDE_sum_of_cyclic_equations_l2411_241148

theorem sum_of_cyclic_equations (p q r : ℝ) : 
  p ≠ q ∧ q ≠ r ∧ r ≠ p →
  q = p * (4 - p) →
  r = q * (4 - q) →
  p = r * (4 - r) →
  p + q + r = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cyclic_equations_l2411_241148


namespace NUMINAMATH_CALUDE_sum_of_b_and_c_l2411_241155

theorem sum_of_b_and_c (a b c d : ℝ) 
  (h1 : a + b = 14)
  (h2 : c + d = 3)
  (h3 : a + d = 8) :
  b + c = 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_b_and_c_l2411_241155


namespace NUMINAMATH_CALUDE_smallest_block_volume_l2411_241107

theorem smallest_block_volume (N : ℕ) : 
  (∃ x y z : ℕ, 
    N = x * y * z ∧ 
    (x - 1) * (y - 1) * (z - 1) = 231 ∧
    ∀ a b c : ℕ, a * b * c = N → (a - 1) * (b - 1) * (c - 1) = 231 → 
      x * y * z ≤ a * b * c) → 
  N = 384 := by
sorry

end NUMINAMATH_CALUDE_smallest_block_volume_l2411_241107


namespace NUMINAMATH_CALUDE_sqrt_4_not_plus_minus_2_l2411_241128

theorem sqrt_4_not_plus_minus_2 : ¬(Real.sqrt 4 = 2 ∨ Real.sqrt 4 = -2) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_4_not_plus_minus_2_l2411_241128


namespace NUMINAMATH_CALUDE_wilted_flower_ratio_l2411_241134

theorem wilted_flower_ratio (initial_roses : ℕ) (remaining_flowers : ℕ) :
  initial_roses = 36 →
  remaining_flowers = 12 →
  (initial_roses / 2 - remaining_flowers) / (initial_roses / 2) = 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_wilted_flower_ratio_l2411_241134


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_2s_l2411_241189

-- Define the displacement function
def s (t : ℝ) : ℝ := 3 * t^3 - 2 * t^2 + t + 1

-- Define the velocity function as the derivative of the displacement function
def v (t : ℝ) : ℝ := 9 * t^2 - 4 * t + 1

-- Theorem statement
theorem instantaneous_velocity_at_2s : v 2 = 29 := by
  sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_2s_l2411_241189


namespace NUMINAMATH_CALUDE_initial_mixture_volume_l2411_241185

/-- Proof of initial mixture volume given ratio changes after water addition -/
theorem initial_mixture_volume
  (initial_milk : ℝ)
  (initial_water : ℝ)
  (added_water : ℝ)
  (h1 : initial_milk / initial_water = 4)
  (h2 : added_water = 23)
  (h3 : initial_milk / (initial_water + added_water) = 1.125)
  : initial_milk + initial_water = 45 := by
  sorry

end NUMINAMATH_CALUDE_initial_mixture_volume_l2411_241185


namespace NUMINAMATH_CALUDE_bucket_volume_proof_l2411_241177

/-- The volume of water (in liters) that Tap A runs per minute -/
def tap_a_rate : ℝ := 3

/-- The time (in minutes) it takes Tap B to fill 1/3 of the bucket -/
def tap_b_third_time : ℝ := 20

/-- The time (in minutes) it takes both taps working together to fill the bucket -/
def combined_time : ℝ := 10

/-- The total volume of the bucket in liters -/
def bucket_volume : ℝ := 36

theorem bucket_volume_proof :
  let tap_b_rate := bucket_volume / (3 * tap_b_third_time)
  tap_a_rate + tap_b_rate = bucket_volume / combined_time := by
  sorry

end NUMINAMATH_CALUDE_bucket_volume_proof_l2411_241177


namespace NUMINAMATH_CALUDE_randi_has_more_nickels_l2411_241122

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- Ray's initial amount in cents -/
def ray_initial_amount : ℕ := 175

/-- Amount given to Peter in cents -/
def amount_to_peter : ℕ := 30

/-- Amount given to Randi in cents -/
def amount_to_randi : ℕ := 2 * amount_to_peter

/-- Number of nickels Randi receives -/
def randi_nickels : ℕ := amount_to_randi / nickel_value

/-- Number of nickels Peter receives -/
def peter_nickels : ℕ := amount_to_peter / nickel_value

theorem randi_has_more_nickels : randi_nickels - peter_nickels = 6 := by
  sorry

end NUMINAMATH_CALUDE_randi_has_more_nickels_l2411_241122


namespace NUMINAMATH_CALUDE_point_line_plane_relationship_l2411_241160

-- Define the types for point, line, and plane
variable (Point Line Plane : Type)

-- Define the relationships
variable (lies_on : Point → Line → Prop)
variable (lies_in : Line → Plane → Prop)

-- Define the subset and element relationships
variable (subset : Line → Plane → Prop)
variable (element : Point → Line → Prop)

-- State the theorem
theorem point_line_plane_relationship 
  (A : Point) (a : Line) (α : Plane) 
  (h1 : lies_on A a) 
  (h2 : lies_in a α) :
  element A a ∧ subset a α := by sorry

end NUMINAMATH_CALUDE_point_line_plane_relationship_l2411_241160


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l2411_241188

def A : Set ℝ := {x | x - 1 > 0}
def B : Set ℝ := {x | x^2 - x - 2 > 0}

theorem union_of_A_and_B : A ∪ B = {x | x < -1 ∨ x > 1} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l2411_241188


namespace NUMINAMATH_CALUDE_largest_triangle_perimeter_l2411_241187

theorem largest_triangle_perimeter :
  ∀ x : ℕ,
  x > 0 →
  x < 7 + 9 →
  7 + x > 9 →
  9 + x > 7 →
  ∀ y : ℕ,
  y > 0 →
  y < 7 + 9 →
  7 + y > 9 →
  9 + y > 7 →
  7 + 9 + x ≥ 7 + 9 + y →
  7 + 9 + x = 31 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_triangle_perimeter_l2411_241187


namespace NUMINAMATH_CALUDE_bus_capacity_is_180_l2411_241104

/-- Represents the seating capacity of a double-decker bus -/
def double_decker_bus_capacity : ℕ :=
  let lower_left := 15 * 3
  let lower_right := 12 * 3
  let lower_back := 9
  let upper_left := 20 * 2
  let upper_right := 20 * 2
  let jump_seats := 4 * 1
  let emergency := 6
  lower_left + lower_right + lower_back + upper_left + upper_right + jump_seats + emergency

/-- Theorem stating the total seating capacity of the double-decker bus -/
theorem bus_capacity_is_180 : double_decker_bus_capacity = 180 := by
  sorry

#eval double_decker_bus_capacity

end NUMINAMATH_CALUDE_bus_capacity_is_180_l2411_241104


namespace NUMINAMATH_CALUDE_largest_satisfying_number_l2411_241179

/-- Returns the leading digit of a positive integer -/
def leadingDigit (n : ℕ) : ℕ :=
  if n < 10 then n else leadingDigit (n / 10)

/-- Returns the sum of digits of a positive integer -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

/-- Checks if a number satisfies the condition -/
def satisfiesCondition (n : ℕ) : Prop :=
  n > 0 ∧ n = leadingDigit n * sumOfDigits n

theorem largest_satisfying_number :
  satisfiesCondition 48 ∧ ∀ m : ℕ, m > 48 → ¬satisfiesCondition m :=
sorry

end NUMINAMATH_CALUDE_largest_satisfying_number_l2411_241179


namespace NUMINAMATH_CALUDE_total_non_hot_peppers_l2411_241132

-- Define the types of peppers
inductive PepperType
| Hot
| Sweet
| Mild

-- Define a structure for daily pepper counts
structure DailyPeppers where
  hot : Nat
  sweet : Nat
  mild : Nat

-- Define the week's pepper counts
def weekPeppers : List DailyPeppers := [
  ⟨7, 10, 13⟩,  -- Sunday
  ⟨12, 8, 10⟩,  -- Monday
  ⟨14, 19, 7⟩,  -- Tuesday
  ⟨12, 5, 23⟩,  -- Wednesday
  ⟨5, 20, 5⟩,   -- Thursday
  ⟨18, 15, 12⟩, -- Friday
  ⟨12, 8, 30⟩   -- Saturday
]

-- Function to calculate non-hot peppers for a day
def nonHotPeppers (day : DailyPeppers) : Nat :=
  day.sweet + day.mild

-- Theorem: The sum of non-hot peppers throughout the week is 185
theorem total_non_hot_peppers :
  (weekPeppers.map nonHotPeppers).sum = 185 := by
  sorry


end NUMINAMATH_CALUDE_total_non_hot_peppers_l2411_241132


namespace NUMINAMATH_CALUDE_max_thursday_money_l2411_241121

def tuesday_amount : ℕ := 8

def wednesday_amount : ℕ := 5 * tuesday_amount

def thursday_amount : ℕ := tuesday_amount + 41

theorem max_thursday_money : thursday_amount = 49 := by
  sorry

end NUMINAMATH_CALUDE_max_thursday_money_l2411_241121


namespace NUMINAMATH_CALUDE_small_animal_weight_l2411_241115

def bear_weight_gain (total_weight : ℝ) (berry_fraction : ℝ) (acorn_multiplier : ℝ) (salmon_fraction : ℝ) : ℝ :=
  let berry_weight := total_weight * berry_fraction
  let acorn_weight := berry_weight * acorn_multiplier
  let remaining_weight := total_weight - (berry_weight + acorn_weight)
  let salmon_weight := remaining_weight * salmon_fraction
  total_weight - (berry_weight + acorn_weight + salmon_weight)

theorem small_animal_weight :
  bear_weight_gain 1000 (1/5) 2 (1/2) = 200 := by
  sorry

end NUMINAMATH_CALUDE_small_animal_weight_l2411_241115


namespace NUMINAMATH_CALUDE_mittens_per_box_example_l2411_241159

/-- Given a number of boxes, scarves per box, and total clothing pieces,
    calculate the number of mittens per box. -/
def mittensPerBox (numBoxes : ℕ) (scarvesPerBox : ℕ) (totalClothes : ℕ) : ℕ :=
  let totalScarves := numBoxes * scarvesPerBox
  let totalMittens := totalClothes - totalScarves
  totalMittens / numBoxes

/-- Prove that given 7 boxes, 3 scarves per box, and 49 total clothing pieces,
    there are 4 mittens in each box. -/
theorem mittens_per_box_example :
  mittensPerBox 7 3 49 = 4 := by
  sorry

end NUMINAMATH_CALUDE_mittens_per_box_example_l2411_241159


namespace NUMINAMATH_CALUDE_cube_with_holes_surface_area_l2411_241118

/-- Calculates the total surface area of a cube with square holes on each face. -/
def totalSurfaceArea (cubeEdge : ℝ) (holeEdge : ℝ) (holeDepth : ℝ) : ℝ :=
  let originalSurface := 6 * cubeEdge^2
  let holeArea := 6 * holeEdge^2
  let newSurfaceInHoles := 6 * 4 * holeEdge * holeDepth
  originalSurface - holeArea + newSurfaceInHoles

/-- Theorem: The total surface area of a cube with edge length 4 meters and
    square holes (side 1 meter, depth 1 meter) centered on each face is 114 square meters. -/
theorem cube_with_holes_surface_area :
  totalSurfaceArea 4 1 1 = 114 := by
  sorry

end NUMINAMATH_CALUDE_cube_with_holes_surface_area_l2411_241118


namespace NUMINAMATH_CALUDE_sum_of_possible_x_coordinates_of_A_sum_of_possible_x_coordinates_of_A_is_400_l2411_241126

/-- Given two triangles ABC and ADE with specified areas and coordinates for points B, C, D, and E,
    prove that the sum of all possible x-coordinates of point A is 400. -/
theorem sum_of_possible_x_coordinates_of_A : ℝ → Prop :=
  fun sum_x =>
    ∀ (A B C D E : ℝ × ℝ)
      (area_ABC area_ADE : ℝ),
    B = (0, 0) →
    C = (200, 0) →
    D = (600, 400) →
    E = (610, 410) →
    area_ABC = 3000 →
    area_ADE = 6000 →
    (∃ (x₁ x₂ : ℝ), 
      (A.1 = x₁ ∨ A.1 = x₂) ∧ 
      sum_x = x₁ + x₂) →
    sum_x = 400

/-- Proof of the theorem -/
theorem sum_of_possible_x_coordinates_of_A_is_400 :
  sum_of_possible_x_coordinates_of_A 400 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_possible_x_coordinates_of_A_sum_of_possible_x_coordinates_of_A_is_400_l2411_241126


namespace NUMINAMATH_CALUDE_quadratic_coefficient_sum_l2411_241117

theorem quadratic_coefficient_sum (m n : ℤ) : 
  (∀ x : ℤ, (x + 2) * (x - 1) = x^2 + m*x + n) → m + n = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_sum_l2411_241117


namespace NUMINAMATH_CALUDE_smallest_divisible_by_1_to_10_l2411_241130

theorem smallest_divisible_by_1_to_10 : ∃ n : ℕ, n > 0 ∧ (∀ k : ℕ, 1 ≤ k ∧ k ≤ 10 → k ∣ n) ∧ (∀ m : ℕ, m > 0 ∧ (∀ k : ℕ, 1 ≤ k ∧ k ≤ 10 → k ∣ m) → n ≤ m) ∧ n = 2520 := by
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_1_to_10_l2411_241130


namespace NUMINAMATH_CALUDE_car_expense_difference_l2411_241102

-- Define Alberto's expenses
def alberto_engine : ℝ := 2457
def alberto_transmission : ℝ := 374
def alberto_tires : ℝ := 520
def alberto_discount_rate : ℝ := 0.05

-- Define Samara's expenses
def samara_oil : ℝ := 25
def samara_tires : ℝ := 467
def samara_detailing : ℝ := 79
def samara_stereo : ℝ := 150
def samara_tax_rate : ℝ := 0.07

-- Theorem statement
theorem car_expense_difference : 
  let alberto_total := alberto_engine + alberto_transmission + alberto_tires
  let alberto_discount := alberto_total * alberto_discount_rate
  let alberto_final := alberto_total - alberto_discount
  let samara_total := samara_oil + samara_tires + samara_detailing + samara_stereo
  let samara_tax := samara_total * samara_tax_rate
  let samara_final := samara_total + samara_tax
  alberto_final - samara_final = 2411.98 := by
    sorry

end NUMINAMATH_CALUDE_car_expense_difference_l2411_241102


namespace NUMINAMATH_CALUDE_value_of_b_l2411_241175

theorem value_of_b (a b : ℝ) (h1 : 3 * a + 2 = 2) (h2 : b - 2 * a = 3) : b = 3 := by
  sorry

end NUMINAMATH_CALUDE_value_of_b_l2411_241175


namespace NUMINAMATH_CALUDE_student_average_age_l2411_241123

/-- Given a class of students and a staff member, if including the staff's age
    increases the average age by 1 year, then we can determine the average age of the students. -/
theorem student_average_age
  (num_students : ℕ)
  (staff_age : ℕ)
  (avg_increase : ℝ)
  (h1 : num_students = 32)
  (h2 : staff_age = 49)
  (h3 : avg_increase = 1) :
  (num_students * (staff_age - num_students - 1 : ℝ)) / num_students = 16 := by
  sorry

end NUMINAMATH_CALUDE_student_average_age_l2411_241123


namespace NUMINAMATH_CALUDE_expected_sides_theorem_expected_sides_rectangle_limit_l2411_241124

/-- The expected number of sides of a randomly selected polygon after cuts -/
def expected_sides (n k : ℕ) : ℚ :=
  (n + 4 * k) / (k + 1)

/-- Theorem: The expected number of sides of a randomly selected polygon
    after k cuts, starting with an n-sided polygon, is (n + 4k) / (k + 1) -/
theorem expected_sides_theorem (n k : ℕ) :
  expected_sides n k = (n + 4 * k) / (k + 1) := by
  sorry

/-- Corollary: For a rectangle (n = 4) and large k, the expectation approaches 4 -/
theorem expected_sides_rectangle_limit :
  ∀ ε > 0, ∃ K : ℕ, ∀ k ≥ K, |expected_sides 4 k - 4| < ε := by
  sorry

end NUMINAMATH_CALUDE_expected_sides_theorem_expected_sides_rectangle_limit_l2411_241124


namespace NUMINAMATH_CALUDE_jungkook_has_biggest_number_l2411_241110

def jungkook_number : ℕ := 6 * 3
def yoongi_number : ℕ := 4
def yuna_number : ℕ := 5

theorem jungkook_has_biggest_number :
  jungkook_number > yoongi_number ∧ jungkook_number > yuna_number := by
  sorry

end NUMINAMATH_CALUDE_jungkook_has_biggest_number_l2411_241110


namespace NUMINAMATH_CALUDE_probability_not_pulling_prize_l2411_241133

/-- Given odds of 3:4 for pulling a prize, the probability of not pulling the prize is 4/7 -/
theorem probability_not_pulling_prize (odds_for : ℚ) (odds_against : ℚ) 
  (h_odds : odds_for = 3 ∧ odds_against = 4) :
  (odds_against / (odds_for + odds_against)) = 4/7 := by
sorry

end NUMINAMATH_CALUDE_probability_not_pulling_prize_l2411_241133


namespace NUMINAMATH_CALUDE_pet_walking_problem_l2411_241129

def smallest_common_multiple (a b : ℕ) : ℕ := Nat.lcm a b

theorem pet_walking_problem (gabe_group_size steven_group_size : ℕ) 
  (h1 : gabe_group_size = 2) 
  (h2 : steven_group_size = 10) : 
  smallest_common_multiple gabe_group_size steven_group_size = 20 := by
  sorry

#check pet_walking_problem

end NUMINAMATH_CALUDE_pet_walking_problem_l2411_241129


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2411_241103

theorem imaginary_part_of_complex_fraction (i : ℂ) :
  i * i = -1 →
  (5 * i / (2 - i)).im = 2 :=
by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2411_241103


namespace NUMINAMATH_CALUDE_rent_increase_percentage_l2411_241164

theorem rent_increase_percentage (last_year_earnings : ℝ) : 
  let last_year_rent := 0.25 * last_year_earnings
  let this_year_earnings := 1.45 * last_year_earnings
  let this_year_rent := 0.35 * this_year_earnings
  (this_year_rent / last_year_rent) * 100 = 203 := by
sorry

end NUMINAMATH_CALUDE_rent_increase_percentage_l2411_241164


namespace NUMINAMATH_CALUDE_square_fraction_is_perfect_square_l2411_241170

theorem square_fraction_is_perfect_square (a b k : ℕ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : k > 0) 
  (h4 : (a^2 + b^2 : ℕ) = k * (a * b + 1)) : 
  ∃ (n : ℕ), k = n^2 := by
  sorry

end NUMINAMATH_CALUDE_square_fraction_is_perfect_square_l2411_241170


namespace NUMINAMATH_CALUDE_vacant_seats_l2411_241101

theorem vacant_seats (total_seats : ℕ) (filled_percentage : ℚ) (h1 : total_seats = 600) (h2 : filled_percentage = 1/2) :
  (total_seats : ℚ) * (1 - filled_percentage) = 300 := by
  sorry

end NUMINAMATH_CALUDE_vacant_seats_l2411_241101


namespace NUMINAMATH_CALUDE_system_four_solutions_l2411_241168

theorem system_four_solutions (a : ℝ) (ha : a > 0) :
  ∃! (solutions : Finset (ℝ × ℝ)), 
    solutions.card = 4 ∧
    ∀ (x y : ℝ), (x, y) ∈ solutions ↔ 
      (y = a * x^2 ∧ y^2 + 3 = x^2 + 4*y) :=
sorry

end NUMINAMATH_CALUDE_system_four_solutions_l2411_241168


namespace NUMINAMATH_CALUDE_sin_cos_sum_equals_sqrt3_div_2_l2411_241158

theorem sin_cos_sum_equals_sqrt3_div_2 :
  Real.sin (17 * π / 180) * Real.cos (43 * π / 180) + 
  Real.sin (73 * π / 180) * Real.sin (43 * π / 180) = 
  Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_equals_sqrt3_div_2_l2411_241158


namespace NUMINAMATH_CALUDE_wall_length_given_mirror_area_l2411_241186

/-- Given a square mirror and a rectangular wall, prove the length of the wall
    when the mirror's area is half the wall's area. -/
theorem wall_length_given_mirror_area (mirror_side : ℝ) (wall_width : ℝ) :
  mirror_side = 24 →
  wall_width = 42 →
  (mirror_side ^ 2) * 2 = wall_width * (27.4285714 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_wall_length_given_mirror_area_l2411_241186


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_not_equal_l2411_241136

def A : Set ℝ := {x : ℝ | |x - 2| ≤ 2}
def B : Set ℝ := {y : ℝ | ∃ x : ℝ, y = -x^2 ∧ -1 ≤ x ∧ x ≤ 2}

theorem complement_A_intersect_B_not_equal :
  (Aᶜ ∪ Bᶜ) ≠ Set.univ ∧
  (Aᶜ ∪ Bᶜ) ≠ {x : ℝ | x ≠ 0} ∧
  (Aᶜ ∪ Bᶜ) ≠ {0} :=
by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_not_equal_l2411_241136


namespace NUMINAMATH_CALUDE_quadrilateral_prism_volume_l2411_241191

/-- A quadrilateral prism with specific properties -/
structure QuadrilateralPrism where
  -- The base is a rhombus with apex angle 60°
  base_is_rhombus : Bool
  base_apex_angle : ℝ
  -- The angle between each face and the base is 60°
  face_base_angle : ℝ
  -- There exists a point inside with distance 1 to base and each face
  interior_point_exists : Bool
  -- Volume of the prism
  volume : ℝ

/-- The volume of a quadrilateral prism with specific properties is 8√3 -/
theorem quadrilateral_prism_volume 
  (P : QuadrilateralPrism) 
  (h1 : P.base_is_rhombus = true)
  (h2 : P.base_apex_angle = 60)
  (h3 : P.face_base_angle = 60)
  (h4 : P.interior_point_exists = true) :
  P.volume = 8 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_prism_volume_l2411_241191


namespace NUMINAMATH_CALUDE_intersection_equality_range_l2411_241100

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ 2 * a + 3}
def B : Set ℝ := {x | -2 ≤ x ∧ x ≤ 4}

-- Define the universal set U
def U : Set ℝ := Set.univ

-- Theorem statement
theorem intersection_equality_range (a : ℝ) :
  A a ∩ B = A a ↔ a ∈ Set.Iic (-4) ∪ Set.Icc (-1) (1/2) := by
  sorry

end NUMINAMATH_CALUDE_intersection_equality_range_l2411_241100


namespace NUMINAMATH_CALUDE_total_cash_realized_proof_l2411_241174

/-- Represents a stock with its value and brokerage rate -/
structure Stock where
  value : ℝ
  brokerage_rate : ℝ

/-- Calculates the cash realized for a single stock after brokerage -/
def cash_realized_single (stock : Stock) : ℝ :=
  stock.value * (1 - stock.brokerage_rate)

/-- Calculates the total cash realized for multiple stocks -/
def total_cash_realized (stocks : List Stock) : ℝ :=
  stocks.map cash_realized_single |>.sum

/-- Theorem stating that the total cash realized for the given stocks is 637.818125 -/
theorem total_cash_realized_proof (stockA stockB stockC : Stock)
  (hA : stockA = { value := 120.50, brokerage_rate := 0.0025 })
  (hB : stockB = { value := 210.75, brokerage_rate := 0.005 })
  (hC : stockC = { value := 310.25, brokerage_rate := 0.0075 }) :
  total_cash_realized [stockA, stockB, stockC] = 637.818125 := by
  sorry

end NUMINAMATH_CALUDE_total_cash_realized_proof_l2411_241174


namespace NUMINAMATH_CALUDE_parabola_equation_hyperbola_equation_l2411_241144

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/64 + y^2/16 = 1

-- Define the parabola focus
def parabola_focus : ℝ × ℝ := (-8, 0)

-- Define the hyperbola asymptotes
def hyperbola_asymptote (x y : ℝ) : Prop := y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x

-- Theorem for the parabola equation
theorem parabola_equation : 
  ∃ (x y : ℝ), (x, y) = parabola_focus → y^2 = -32*x := by sorry

-- Theorem for the hyperbola equation
theorem hyperbola_equation :
  (∀ (x y : ℝ), ellipse x y ↔ ellipse (-x) y) → 
  (∀ (x y : ℝ), hyperbola_asymptote x y) →
  ∃ (x y : ℝ), x^2/12 - y^2/36 = 1 := by sorry

end NUMINAMATH_CALUDE_parabola_equation_hyperbola_equation_l2411_241144


namespace NUMINAMATH_CALUDE_bettys_herb_garden_l2411_241142

theorem bettys_herb_garden (basil oregano : ℕ) : 
  oregano = 2 * basil + 2 →
  basil + oregano = 17 →
  basil = 5 := by sorry

end NUMINAMATH_CALUDE_bettys_herb_garden_l2411_241142


namespace NUMINAMATH_CALUDE_fraction_inequality_l2411_241119

theorem fraction_inequality (a b m : ℝ) 
  (h1 : a > b) (h2 : b > 0) (h3 : m > 0) (h4 : a + m > 0) :
  (b + m) / (a + m) > b / a :=
by sorry

end NUMINAMATH_CALUDE_fraction_inequality_l2411_241119


namespace NUMINAMATH_CALUDE_vector_subtraction_l2411_241172

def a : Fin 2 → ℝ := ![-1, 3]
def b : Fin 2 → ℝ := ![2, -1]

theorem vector_subtraction : a - 2 • b = ![-5, 5] := by sorry

end NUMINAMATH_CALUDE_vector_subtraction_l2411_241172


namespace NUMINAMATH_CALUDE_equal_gum_distribution_l2411_241153

/-- Proves that when three people share 99 pieces of gum equally, each person receives 33 pieces. -/
theorem equal_gum_distribution (john_gum : ℕ) (cole_gum : ℕ) (aubrey_gum : ℕ) 
  (h1 : john_gum = 54)
  (h2 : cole_gum = 45)
  (h3 : aubrey_gum = 0)
  (h4 : (john_gum + cole_gum + aubrey_gum) % 3 = 0) :
  (john_gum + cole_gum + aubrey_gum) / 3 = 33 := by
  sorry

end NUMINAMATH_CALUDE_equal_gum_distribution_l2411_241153


namespace NUMINAMATH_CALUDE_fraction_comparison_l2411_241140

theorem fraction_comparison : (1 / (Real.sqrt 5 - 2)) < (1 / (Real.sqrt 6 - Real.sqrt 5)) := by
  sorry

end NUMINAMATH_CALUDE_fraction_comparison_l2411_241140


namespace NUMINAMATH_CALUDE_bead_arrangement_probability_l2411_241192

def num_red : ℕ := 4
def num_white : ℕ := 2
def num_blue : ℕ := 2
def total_beads : ℕ := num_red + num_white + num_blue

def total_arrangements : ℕ := Nat.factorial total_beads / (Nat.factorial num_red * Nat.factorial num_white * Nat.factorial num_blue)

def valid_arrangements : ℕ := 27  -- This is an approximation based on the problem's solution

theorem bead_arrangement_probability :
  (valid_arrangements : ℚ) / total_arrangements = 9 / 140 :=
sorry

end NUMINAMATH_CALUDE_bead_arrangement_probability_l2411_241192


namespace NUMINAMATH_CALUDE_job_completion_time_l2411_241151

/-- Given a job that A and B can complete together in 5 days, and B can complete alone in 10 days,
    prove that A can complete the job alone in 10 days. -/
theorem job_completion_time (rate_A rate_B : ℝ) : 
  rate_A + rate_B = 1 / 5 →  -- A and B together complete the job in 5 days
  rate_B = 1 / 10 →          -- B alone completes the job in 10 days
  rate_A = 1 / 10            -- A alone completes the job in 10 days
:= by sorry

end NUMINAMATH_CALUDE_job_completion_time_l2411_241151


namespace NUMINAMATH_CALUDE_curve_transformation_l2411_241112

/-- Given a scaling transformation and the equation of the transformed curve,
    prove the equation of the original curve. -/
theorem curve_transformation (x y x' y' : ℝ) :
  (x' = 5 * x) →
  (y' = 3 * y) →
  (x'^2 + 4 * y'^2 = 1) →
  (25 * x^2 + 36 * y^2 = 1) := by
  sorry

end NUMINAMATH_CALUDE_curve_transformation_l2411_241112


namespace NUMINAMATH_CALUDE_factory_output_l2411_241197

/-- Calculates the number of batteries manufactured by robots in a given time period. -/
def batteries_manufactured (gather_time min_per_battery : ℕ) (create_time min_per_battery : ℕ) 
  (num_robots : ℕ) (total_time hours : ℕ) : ℕ :=
  let total_time_minutes := total_time * 60
  let time_per_battery := gather_time + create_time
  let batteries_per_robot_per_hour := 60 / time_per_battery
  let batteries_per_hour := num_robots * batteries_per_robot_per_hour
  batteries_per_hour * total_time

/-- The number of batteries manufactured by 10 robots in 5 hours is 200. -/
theorem factory_output : batteries_manufactured 6 9 10 5 = 200 := by
  sorry

end NUMINAMATH_CALUDE_factory_output_l2411_241197


namespace NUMINAMATH_CALUDE_rose_difference_l2411_241157

/-- Given the initial number of roses in a vase, the number of roses thrown away,
    and the final number of roses in the vase, calculate the difference between
    the number of roses thrown away and the number of roses cut from the garden. -/
theorem rose_difference (initial : ℕ) (thrown_away : ℕ) (final : ℕ) :
  initial = 21 → thrown_away = 34 → final = 15 →
  thrown_away - final = 19 := by sorry

end NUMINAMATH_CALUDE_rose_difference_l2411_241157


namespace NUMINAMATH_CALUDE_isosceles_base_angles_equal_l2411_241139

/-- An isosceles triangle is a triangle with two sides of equal length -/
structure IsoscelesTriangle where
  points : Fin 3 → ℝ × ℝ
  isosceles : ∃ i j k, i ≠ j ∧ j ≠ k ∧ k ≠ i ∧ 
    dist (points i) (points j) = dist (points i) (points k)

/-- The base angles of an isosceles triangle are the angles opposite the equal sides -/
def base_angles (t : IsoscelesTriangle) : ℝ × ℝ := sorry

/-- In an isosceles triangle, the two base angles are equal -/
theorem isosceles_base_angles_equal (t : IsoscelesTriangle) : 
  (base_angles t).1 = (base_angles t).2 := by sorry

end NUMINAMATH_CALUDE_isosceles_base_angles_equal_l2411_241139


namespace NUMINAMATH_CALUDE_total_onions_grown_l2411_241111

theorem total_onions_grown (nancy_onions dan_onions mike_onions : ℕ) 
  (h1 : nancy_onions = 2)
  (h2 : dan_onions = 9)
  (h3 : mike_onions = 4) :
  nancy_onions + dan_onions + mike_onions = 15 := by
  sorry

end NUMINAMATH_CALUDE_total_onions_grown_l2411_241111


namespace NUMINAMATH_CALUDE_supply_duration_l2411_241135

/-- Represents the number of pills in one supply -/
def supply : ℕ := 90

/-- Represents the fraction of a pill consumed in one dose -/
def dose : ℚ := 3/4

/-- Represents the number of days between doses -/
def interval : ℕ := 3

/-- Represents the number of days in a month (assumed average) -/
def days_per_month : ℕ := 30

/-- Theorem stating that the given supply lasts 12 months -/
theorem supply_duration :
  (supply : ℚ) * interval / dose / days_per_month = 12 := by
  sorry

end NUMINAMATH_CALUDE_supply_duration_l2411_241135


namespace NUMINAMATH_CALUDE_circle_center_l2411_241176

/-- The center of the circle defined by x^2 + y^2 + 2y = 1 is (0, -1) -/
theorem circle_center (x y : ℝ) : 
  (x^2 + y^2 + 2*y = 1) → (0, -1) = (0, -1) := by sorry

end NUMINAMATH_CALUDE_circle_center_l2411_241176


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_min_reciprocal_sum_attained_l2411_241125

theorem min_reciprocal_sum (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0)
  (h4 : x + y + z = 3) (h5 : y = 2 * x) :
  (1 / x + 1 / y + 1 / z) ≥ 10 / 3 := by
  sorry

theorem min_reciprocal_sum_attained (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0)
  (h4 : x + y + z = 3) (h5 : y = 2 * x) :
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ + y₀ + z₀ = 3 ∧ y₀ = 2 * x₀ ∧
  (1 / x₀ + 1 / y₀ + 1 / z₀) = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_min_reciprocal_sum_attained_l2411_241125


namespace NUMINAMATH_CALUDE_smallest_yummy_number_l2411_241127

/-- Definition of a yummy number -/
def is_yummy (A : ℕ) : Prop :=
  ∃ n : ℕ+, n * (2 * A + n - 1) = 2 * 2023

/-- Theorem stating that 1011 is the smallest yummy number -/
theorem smallest_yummy_number :
  is_yummy 1011 ∧ ∀ A : ℕ, A < 1011 → ¬is_yummy A :=
sorry

end NUMINAMATH_CALUDE_smallest_yummy_number_l2411_241127


namespace NUMINAMATH_CALUDE_digit_sum_of_special_number_l2411_241166

theorem digit_sum_of_special_number : 
  ∀ (x : ℕ) (x' : ℕ) (y : ℕ),
  10000 ≤ x ∧ x < 100000 →  -- x is a five-digit number
  1000 ≤ x' ∧ x' < 10000 →  -- x' is a four-digit number
  0 ≤ y ∧ y < 10 →          -- y is a single digit
  x = 10 * x' + y →         -- x' is x with the ones digit removed
  x + x' = 52713 →          -- given condition
  (x / 10000) + ((x / 1000) % 10) + ((x / 100) % 10) + ((x / 10) % 10) + (x % 10) = 23 :=
by sorry

end NUMINAMATH_CALUDE_digit_sum_of_special_number_l2411_241166


namespace NUMINAMATH_CALUDE_sulfuric_acid_moles_l2411_241183

/-- Represents the chemical reaction Fe + H₂SO₄ → FeSO₄ + H₂ -/
structure ChemicalReaction where
  iron : ℝ
  sulfuricAcid : ℝ
  hydrogen : ℝ

/-- The stoichiometric relationship in the reaction -/
axiom stoichiometry (r : ChemicalReaction) : r.iron = r.sulfuricAcid ∧ r.iron = r.hydrogen

/-- The theorem to prove -/
theorem sulfuric_acid_moles (r : ChemicalReaction) 
  (h1 : r.iron = 2) 
  (h2 : r.hydrogen = 2) : 
  r.sulfuricAcid = 2 := by
  sorry

end NUMINAMATH_CALUDE_sulfuric_acid_moles_l2411_241183


namespace NUMINAMATH_CALUDE_division_problem_l2411_241116

theorem division_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) : 
  dividend = 507 → divisor = 8 → remainder = 19 → 
  dividend = divisor * quotient + remainder →
  quotient = 61 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l2411_241116


namespace NUMINAMATH_CALUDE_arithmetic_sequence_third_term_l2411_241147

/-- Given an arithmetic sequence {a_n} with sum of first n terms S_n = n^2 + n, a_3 = 6 -/
theorem arithmetic_sequence_third_term (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n, S n = n^2 + n) → 
  (∀ n ≥ 2, a n = S n - S (n-1)) → 
  a 3 = 6 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_third_term_l2411_241147


namespace NUMINAMATH_CALUDE_larger_number_proof_l2411_241114

theorem larger_number_proof (a b : ℕ+) (h1 : Nat.gcd a b = 59) 
  (h2 : Nat.lcm a b = 12272) (h3 : 13 ∣ Nat.lcm a b) (h4 : 16 ∣ Nat.lcm a b) :
  max a b = 944 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l2411_241114


namespace NUMINAMATH_CALUDE_shortest_minor_arc_line_l2411_241171

/-- The point M -/
def M : ℝ × ℝ := (1, -2)

/-- The circle C -/
def C (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 9

/-- A line passing through a point -/
def LineThrough (m : ℝ × ℝ) (a b c : ℝ) : Prop :=
  a * m.1 + b * m.2 + c = 0

/-- The theorem stating the equation of the line that divides the circle into two arcs with the shortest minor arc -/
theorem shortest_minor_arc_line :
  ∃ (a b c : ℝ), LineThrough M a b c ∧
  (∀ (x y : ℝ), C x y → (a * x + b * y + c = 0 → 
    ∀ (a' b' c' : ℝ), LineThrough M a' b' c' → 
      (∃ (x' y' : ℝ), C x' y' ∧ a' * x' + b' * y' + c' = 0) → 
        (∃ (x'' y'' : ℝ), C x'' y'' ∧ a * x'' + b * y'' + c = 0 ∧ 
          ∀ (x''' y''' : ℝ), C x''' y''' ∧ a' * x''' + b' * y''' + c' = 0 → 
            (x'' - M.1)^2 + (y'' - M.2)^2 ≤ (x''' - M.1)^2 + (y''' - M.2)^2))) ∧
  a = 1 ∧ b = 2 ∧ c = 3 :=
sorry

end NUMINAMATH_CALUDE_shortest_minor_arc_line_l2411_241171


namespace NUMINAMATH_CALUDE_some_flying_creatures_are_magical_l2411_241198

-- Define our universe
variable (U : Type)

-- Define our predicates
variable (unicorn : U → Prop)
variable (flying : U → Prop)
variable (magical : U → Prop)

-- State the theorem
theorem some_flying_creatures_are_magical :
  (∀ x, unicorn x → flying x) →  -- All unicorns are capable of flying
  (∃ x, magical x ∧ unicorn x) →  -- Some magical creatures are unicorns
  (∃ x, flying x ∧ magical x) :=  -- Some flying creatures are magical creatures
by
  sorry

end NUMINAMATH_CALUDE_some_flying_creatures_are_magical_l2411_241198


namespace NUMINAMATH_CALUDE_money_needed_for_trip_l2411_241146

def trip_cost : ℕ := 5000
def hourly_wage : ℕ := 20
def hours_worked : ℕ := 10
def cookie_price : ℕ := 4
def cookies_sold : ℕ := 24
def lottery_ticket_cost : ℕ := 10
def lottery_winnings : ℕ := 500
def sister_gift : ℕ := 500
def num_sisters : ℕ := 2

theorem money_needed_for_trip :
  trip_cost - (hourly_wage * hours_worked + cookie_price * cookies_sold - lottery_ticket_cost + lottery_winnings + sister_gift * num_sisters) = 3214 := by
  sorry

end NUMINAMATH_CALUDE_money_needed_for_trip_l2411_241146


namespace NUMINAMATH_CALUDE_trailing_zeros_30_factorial_l2411_241184

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25)

/-- Theorem: The number of trailing zeros in 30! is 7 -/
theorem trailing_zeros_30_factorial :
  trailingZeros 30 = 7 := by
  sorry

end NUMINAMATH_CALUDE_trailing_zeros_30_factorial_l2411_241184


namespace NUMINAMATH_CALUDE_simplify_expression_l2411_241162

theorem simplify_expression (m : ℝ) (hm : m ≠ 0) :
  ((m^2 - 3*m + 1) / m + 1) / ((m^2 - 1) / m) = (m - 1) / (m + 1) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2411_241162


namespace NUMINAMATH_CALUDE_reflect_x_minus3_minus5_l2411_241173

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflection of a point across the x-axis -/
def reflect_x (p : Point) : Point :=
  { x := p.x, y := -p.y }

/-- The theorem stating that reflecting P(-3,-5) across the x-axis results in (-3,5) -/
theorem reflect_x_minus3_minus5 :
  let P : Point := { x := -3, y := -5 }
  reflect_x P = { x := -3, y := 5 } := by
  sorry

end NUMINAMATH_CALUDE_reflect_x_minus3_minus5_l2411_241173


namespace NUMINAMATH_CALUDE_function_identity_l2411_241194

theorem function_identity (f : ℕ+ → ℕ+) 
  (h : ∀ n : ℕ+, f (n + 1) > f (f n)) : 
  ∀ n : ℕ+, f n = n := by
sorry

end NUMINAMATH_CALUDE_function_identity_l2411_241194


namespace NUMINAMATH_CALUDE_min_value_expression_l2411_241199

theorem min_value_expression (x y : ℝ) : 5 * x^2 + 4 * y^2 - 8 * x * y + 2 * x + 4 ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l2411_241199


namespace NUMINAMATH_CALUDE_square_area_30cm_l2411_241165

/-- The area of a square with side length 30 centimeters is 900 square centimeters. -/
theorem square_area_30cm (s : ℝ) (h : s = 30) : s * s = 900 := by
  sorry

end NUMINAMATH_CALUDE_square_area_30cm_l2411_241165


namespace NUMINAMATH_CALUDE_max_shapes_in_grid_l2411_241190

/-- The number of rows in the grid -/
def rows : Nat := 8

/-- The number of columns in the grid -/
def columns : Nat := 14

/-- The number of grid points occupied by each shape -/
def points_per_shape : Nat := 8

/-- The total number of grid points in the grid -/
def total_grid_points : Nat := (rows + 1) * (columns + 1)

/-- The maximum number of shapes that can be placed in the grid -/
def max_shapes : Nat := total_grid_points / points_per_shape

theorem max_shapes_in_grid :
  max_shapes = 16 := by sorry

end NUMINAMATH_CALUDE_max_shapes_in_grid_l2411_241190


namespace NUMINAMATH_CALUDE_log_inequality_l2411_241120

theorem log_inequality (a b c : ℝ) : 
  a = Real.log (2/3) → b = Real.log (2/5) → c = Real.log (3/2) → c > a ∧ a > b :=
by sorry

end NUMINAMATH_CALUDE_log_inequality_l2411_241120


namespace NUMINAMATH_CALUDE_pen_probabilities_l2411_241106

/-- The number of pens in the box -/
def total_pens : ℕ := 6

/-- The number of first-class pens -/
def first_class_pens : ℕ := 4

/-- The number of second-class pens -/
def second_class_pens : ℕ := 2

/-- The number of pens drawn -/
def drawn_pens : ℕ := 2

/-- The probability of drawing exactly one first-class pen -/
def prob_one_first_class : ℚ := 8 / 15

/-- The probability of drawing at least one second-class pen -/
def prob_second_class : ℚ := 3 / 5

theorem pen_probabilities :
  (total_pens = first_class_pens + second_class_pens) →
  (prob_one_first_class = (Nat.choose first_class_pens 1 * Nat.choose second_class_pens 1 : ℚ) / Nat.choose total_pens drawn_pens) ∧
  (prob_second_class = 1 - (Nat.choose first_class_pens drawn_pens : ℚ) / Nat.choose total_pens drawn_pens) :=
by sorry

end NUMINAMATH_CALUDE_pen_probabilities_l2411_241106


namespace NUMINAMATH_CALUDE_derivative_f_at_zero_l2411_241137

-- Define the function f
def f (x : ℝ) : ℝ := 25 * x^3 + 13 * x^2 + 2016 * x - 5

-- State the theorem
theorem derivative_f_at_zero : 
  deriv f 0 = 2016 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_zero_l2411_241137


namespace NUMINAMATH_CALUDE_modulus_of_z_l2411_241152

theorem modulus_of_z (z : ℂ) (h : z * Complex.I = 1 + Complex.I) : Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l2411_241152


namespace NUMINAMATH_CALUDE_sqrt_625_div_5_l2411_241167

theorem sqrt_625_div_5 : Real.sqrt 625 / 5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_625_div_5_l2411_241167


namespace NUMINAMATH_CALUDE_johns_shower_duration_johns_shower_theorem_l2411_241138

theorem johns_shower_duration (shower_duration : ℕ) (shower_frequency : ℕ) 
  (water_usage_rate : ℕ) (total_water_usage : ℕ) : ℕ :=
  let water_per_shower := shower_duration * water_usage_rate
  let num_showers := total_water_usage / water_per_shower
  let num_days := num_showers * shower_frequency
  let num_weeks := num_days / 7
  num_weeks

theorem johns_shower_theorem : 
  johns_shower_duration 10 2 2 280 = 4 := by
  sorry

end NUMINAMATH_CALUDE_johns_shower_duration_johns_shower_theorem_l2411_241138


namespace NUMINAMATH_CALUDE_base_10_to_base_7_conversion_l2411_241145

/-- Converts a base 7 number to base 10 --/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- The problem statement --/
theorem base_10_to_base_7_conversion :
  base7ToBase10 [1, 5, 5, 1] = 624 := by
  sorry

end NUMINAMATH_CALUDE_base_10_to_base_7_conversion_l2411_241145


namespace NUMINAMATH_CALUDE_jumping_probabilities_l2411_241150

/-- Probability of an athlete successfully jumping over a 2-meter high bar -/
structure Athlete where
  success_prob : ℝ
  success_prob_nonneg : 0 ≤ success_prob
  success_prob_le_one : success_prob ≤ 1

/-- The problem setup with two athletes A and B -/
def problem_setup (A B : Athlete) : Prop :=
  A.success_prob = 0.7 ∧ B.success_prob = 0.6

/-- The probability that A succeeds on the third attempt -/
def prob_A_third_attempt (A : Athlete) : ℝ :=
  (1 - A.success_prob) * (1 - A.success_prob) * A.success_prob

/-- The probability that at least one of A or B succeeds on the first attempt -/
def prob_at_least_one_first_attempt (A B : Athlete) : ℝ :=
  1 - (1 - A.success_prob) * (1 - B.success_prob)

/-- The probability that A succeeds exactly one more time than B in two attempts for each -/
def prob_A_one_more_than_B (A B : Athlete) : ℝ :=
  2 * A.success_prob * (1 - A.success_prob) * (1 - B.success_prob) * (1 - B.success_prob) +
  A.success_prob * A.success_prob * 2 * B.success_prob * (1 - B.success_prob)

theorem jumping_probabilities (A B : Athlete) 
  (h : problem_setup A B) : 
  prob_A_third_attempt A = 0.063 ∧
  prob_at_least_one_first_attempt A B = 0.88 ∧
  prob_A_one_more_than_B A B = 0.3024 := by
  sorry

end NUMINAMATH_CALUDE_jumping_probabilities_l2411_241150


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2411_241196

theorem complex_fraction_simplification (a : ℝ) (h1 : a ≠ 1) (h2 : a ≠ -1) :
  (1 / (a + 1) - 1 / (a^2 - 1)) / (a / (a - 1) - a) = -1 / (a^2 + a) := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2411_241196


namespace NUMINAMATH_CALUDE_line_increase_l2411_241161

/-- Given a line where an x-increase of 4 corresponds to a y-increase of 10,
    prove that an x-increase of 12 results in a y-increase of 30. -/
theorem line_increase (f : ℝ → ℝ) (h : ∀ x, f (x + 4) - f x = 10) :
  ∀ x, f (x + 12) - f x = 30 := by
  sorry

end NUMINAMATH_CALUDE_line_increase_l2411_241161


namespace NUMINAMATH_CALUDE_max_value_expression_l2411_241131

theorem max_value_expression (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : a + b + c = 1) : 
  (a + 1) * (b + 1) * (c + 1) / (a * b * c + 1) ≤ 16 / 7 := by
  sorry

end NUMINAMATH_CALUDE_max_value_expression_l2411_241131


namespace NUMINAMATH_CALUDE_P_in_fourth_quadrant_l2411_241163

/-- A point in a 2D plane. -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the fourth quadrant. -/
def FourthQuadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- The given point P. -/
def P : Point :=
  { x := 3, y := -2 }

/-- Theorem stating that P is in the fourth quadrant. -/
theorem P_in_fourth_quadrant : FourthQuadrant P := by
  sorry

end NUMINAMATH_CALUDE_P_in_fourth_quadrant_l2411_241163


namespace NUMINAMATH_CALUDE_train_departure_sequences_l2411_241113

theorem train_departure_sequences :
  let total_trains : ℕ := 6
  let trains_per_group : ℕ := 3
  let num_special_trains : ℕ := 2  -- G1 and G2
  let num_regular_trains : ℕ := total_trains - num_special_trains

  -- Number of ways to choose trains for G1's group (excluding G1 itself)
  let group_formations : ℕ := Nat.choose num_regular_trains (trains_per_group - 1)

  -- Number of permutations for each group
  let group_permutations : ℕ := Nat.factorial trains_per_group

  -- Total number of departure sequences
  group_formations * group_permutations * group_permutations = 216 :=
by
  sorry

end NUMINAMATH_CALUDE_train_departure_sequences_l2411_241113


namespace NUMINAMATH_CALUDE_multiples_of_15_between_17_and_158_l2411_241180

theorem multiples_of_15_between_17_and_158 : 
  (Finset.filter (λ x => x % 15 = 0) (Finset.range (158 - 17 + 1))).card = 9 := by
  sorry

end NUMINAMATH_CALUDE_multiples_of_15_between_17_and_158_l2411_241180


namespace NUMINAMATH_CALUDE_fifth_toss_probability_l2411_241154

def coin_flip_probability (n : ℕ) : ℚ :=
  (1 / 2) ^ (n - 1) * (1 / 2)

theorem fifth_toss_probability :
  coin_flip_probability 5 = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fifth_toss_probability_l2411_241154


namespace NUMINAMATH_CALUDE_units_sold_to_A_is_three_l2411_241169

/-- Represents the number of units sold to Customer A in a phone store scenario. -/
def units_sold_to_A (total_phones defective_phones units_sold_to_B units_sold_to_C : ℕ) : ℕ :=
  total_phones - defective_phones - units_sold_to_B - units_sold_to_C

/-- Theorem stating that given the specific conditions of the problem, 
    the number of units sold to Customer A is 3. -/
theorem units_sold_to_A_is_three :
  units_sold_to_A 20 5 5 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_sold_to_A_is_three_l2411_241169


namespace NUMINAMATH_CALUDE_line_intersects_circle_right_angle_l2411_241108

theorem line_intersects_circle_right_angle (k : ℝ) :
  (∃ (P Q : ℝ × ℝ), 
    P.1^2 + P.2^2 = 1 ∧ 
    Q.1^2 + Q.2^2 = 1 ∧ 
    P.2 = k * P.1 + 1 ∧ 
    Q.2 = k * Q.1 + 1 ∧ 
    (P.1 * Q.1 + P.2 * Q.2 = 0)) →
  k = 1 ∨ k = -1 :=
by sorry

end NUMINAMATH_CALUDE_line_intersects_circle_right_angle_l2411_241108


namespace NUMINAMATH_CALUDE_intersection_complement_equals_singleton_zero_l2411_241195

def U : Finset Int := {-1, 0, 1, 2, 3, 4}
def A : Finset Int := {-1, 1, 2, 4}
def B : Finset Int := {-1, 0, 2}

theorem intersection_complement_equals_singleton_zero :
  B ∩ (U \ A) = {0} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equals_singleton_zero_l2411_241195


namespace NUMINAMATH_CALUDE_smallest_square_multiplier_l2411_241178

def y : ℕ := 2^4 * 3^2 * 4^3 * 5^3 * 6^2 * 7^3 * 8^3 * 9^2

theorem smallest_square_multiplier :
  (∀ k : ℕ, k > 0 ∧ k < 350 → ¬ ∃ m : ℕ, k * y = m^2) ∧
  ∃ m : ℕ, 350 * y = m^2 :=
sorry

end NUMINAMATH_CALUDE_smallest_square_multiplier_l2411_241178


namespace NUMINAMATH_CALUDE_transformed_curve_is_circle_l2411_241143

-- Define the initial polar equation
def initial_polar_equation (ρ θ : ℝ) : Prop :=
  ρ^2 = 12 / (3 * (Real.cos θ)^2 + 4 * (Real.sin θ)^2)

-- Define the scaling transformation
def scaling_transformation (x y x' y' : ℝ) : Prop :=
  x' = (1/2) * x ∧ y' = (Real.sqrt 3 / 3) * y

-- Theorem statement
theorem transformed_curve_is_circle :
  ∀ (x y x' y' : ℝ),
  (∃ (ρ θ : ℝ), initial_polar_equation ρ θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) →
  scaling_transformation x y x' y' →
  ∃ (r : ℝ), x'^2 + y'^2 = r^2 :=
sorry

end NUMINAMATH_CALUDE_transformed_curve_is_circle_l2411_241143


namespace NUMINAMATH_CALUDE_base6_to_base10_conversion_l2411_241182

-- Define the base 6 number as a list of digits
def base6_number : List Nat := [5, 4, 3, 2, 1]

-- Define the base of the number system
def base : Nat := 6

-- Function to convert a list of digits in base 6 to base 10
def to_base_10 (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * base^i) 0

-- Theorem statement
theorem base6_to_base10_conversion :
  to_base_10 base6_number base = 7465 := by
  sorry

end NUMINAMATH_CALUDE_base6_to_base10_conversion_l2411_241182


namespace NUMINAMATH_CALUDE_fifteen_factorial_base_nine_zeros_l2411_241193

/-- The number of trailing zeros in n! when written in base b -/
def trailingZeros (n : ℕ) (b : ℕ) : ℕ :=
  sorry

theorem fifteen_factorial_base_nine_zeros :
  trailingZeros 15 9 = 3 :=
sorry

end NUMINAMATH_CALUDE_fifteen_factorial_base_nine_zeros_l2411_241193


namespace NUMINAMATH_CALUDE_reflection_line_equation_l2411_241141

-- Define the triangle vertices and their images
def A : ℝ × ℝ := (3, 2)
def B : ℝ × ℝ := (8, 7)
def C : ℝ × ℝ := (6, -4)
def A' : ℝ × ℝ := (-5, 2)
def B' : ℝ × ℝ := (-10, 7)
def C' : ℝ × ℝ := (-8, -4)

-- Define the reflection line
def L (x : ℝ) : Prop := x = -1

-- Theorem statement
theorem reflection_line_equation :
  (∀ p p', (p = A ∧ p' = A') ∨ (p = B ∧ p' = B') ∨ (p = C ∧ p' = C') →
    p.2 = p'.2 ∧ L ((p.1 + p'.1) / 2)) →
  L (-1) :=
sorry

end NUMINAMATH_CALUDE_reflection_line_equation_l2411_241141


namespace NUMINAMATH_CALUDE_initial_average_height_l2411_241149

theorem initial_average_height (n : ℕ) (wrong_height correct_height actual_average : ℝ) 
  (h1 : n = 35)
  (h2 : wrong_height = 166)
  (h3 : correct_height = 106)
  (h4 : actual_average = 179) :
  (n * actual_average + (wrong_height - correct_height)) / n = 181 :=
by sorry

end NUMINAMATH_CALUDE_initial_average_height_l2411_241149


namespace NUMINAMATH_CALUDE_last_monkey_gets_255_l2411_241156

/-- Represents the process of monkeys dividing apples -/
def monkey_division (n : ℕ) : ℕ → ℕ
| 0 => n
| (k + 1) => 
  let remaining := monkey_division n k
  (remaining - 1) / 5

/-- The number of monkeys -/
def num_monkeys : ℕ := 5

/-- The minimum number of apples needed for the division process -/
def min_apples : ℕ := 5^5 - 4

/-- The amount the last monkey gets -/
def last_monkey_apples : ℕ := monkey_division min_apples (num_monkeys - 1)

theorem last_monkey_gets_255 : last_monkey_apples = 255 := by
  sorry

end NUMINAMATH_CALUDE_last_monkey_gets_255_l2411_241156


namespace NUMINAMATH_CALUDE_bulk_warehouse_case_size_l2411_241105

/-- Proves the number of cans in a bulk warehouse case given pricing information -/
theorem bulk_warehouse_case_size (bulk_case_price : ℚ) (grocery_price : ℚ) (grocery_cans : ℕ) (price_difference : ℚ) : 
  bulk_case_price = 12 →
  grocery_price = 6 →
  grocery_cans = 12 →
  price_difference = 1/4 →
  (bulk_case_price / ((grocery_price / grocery_cans) - price_difference) : ℚ) = 48 :=
by sorry

end NUMINAMATH_CALUDE_bulk_warehouse_case_size_l2411_241105


namespace NUMINAMATH_CALUDE_bank_teller_coin_rolls_l2411_241181

theorem bank_teller_coin_rolls 
  (total_coins : ℕ) 
  (num_tellers : ℕ) 
  (coins_per_roll : ℕ) 
  (h1 : total_coins = 1000) 
  (h2 : num_tellers = 4) 
  (h3 : coins_per_roll = 25) : 
  (total_coins / num_tellers) / coins_per_roll = 10 := by
sorry

end NUMINAMATH_CALUDE_bank_teller_coin_rolls_l2411_241181

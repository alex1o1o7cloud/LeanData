import Mathlib

namespace fraction_inequality_l762_76284

theorem fraction_inequality (a b c x y : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hx : x > 0) (hy : y > 0) : 
  min (x / (a*b + a*c)) (y / (a*c + b*c)) < (x + y) / (a*b + b*c) := by
  sorry

end fraction_inequality_l762_76284


namespace average_increase_calculation_l762_76218

theorem average_increase_calculation (current_matches : ℕ) (current_average : ℚ) (next_match_score : ℕ) : 
  current_matches = 10 →
  current_average = 34 →
  next_match_score = 78 →
  (current_matches + 1) * (current_average + (next_match_score - current_matches * current_average) / (current_matches + 1)) = 
  current_matches * current_average + next_match_score →
  (next_match_score - current_matches * current_average) / (current_matches + 1) = 4 :=
by
  sorry

end average_increase_calculation_l762_76218


namespace right_triangle_altitude_l762_76276

theorem right_triangle_altitude (A h c : ℝ) : 
  A > 0 → c > 0 → h > 0 →
  A = 540 → c = 36 →
  A = (1/2) * c * h →
  h = 30 := by
sorry

end right_triangle_altitude_l762_76276


namespace max_value_expression_l762_76294

theorem max_value_expression (a b c d : ℝ) 
  (ha : -8 ≤ a ∧ a ≤ 8) 
  (hb : -8 ≤ b ∧ b ≤ 8) 
  (hc : -8 ≤ c ∧ c ≤ 8) 
  (hd : -8 ≤ d ∧ d ≤ 8) : 
  (∀ x y z w, -8 ≤ x ∧ x ≤ 8 → -8 ≤ y ∧ y ≤ 8 → -8 ≤ z ∧ z ≤ 8 → -8 ≤ w ∧ w ≤ 8 → 
    x + 2*y + z + 2*w - x*y - y*z - z*w - w*x ≤ 272) ∧ 
  (∃ x y z w, -8 ≤ x ∧ x ≤ 8 ∧ -8 ≤ y ∧ y ≤ 8 ∧ -8 ≤ z ∧ z ≤ 8 ∧ -8 ≤ w ∧ w ≤ 8 ∧
    x + 2*y + z + 2*w - x*y - y*z - z*w - w*x = 272) := by
  sorry

end max_value_expression_l762_76294


namespace next_shared_meeting_l762_76234

/-- Represents the number of days between meetings for each group -/
def drama_club_cycle : ℕ := 3
def choir_cycle : ℕ := 5
def debate_team_cycle : ℕ := 7

/-- Theorem stating that the next shared meeting will occur in 105 days -/
theorem next_shared_meeting :
  ∃ (n : ℕ), n > 0 ∧ 
  n % drama_club_cycle = 0 ∧
  n % choir_cycle = 0 ∧
  n % debate_team_cycle = 0 ∧
  ∀ (m : ℕ), m > 0 ∧ 
    m % drama_club_cycle = 0 ∧
    m % choir_cycle = 0 ∧
    m % debate_team_cycle = 0 →
    n ≤ m :=
by sorry

end next_shared_meeting_l762_76234


namespace luke_stars_made_l762_76271

-- Define the given conditions
def stars_per_jar : ℕ := 85
def bottles_to_fill : ℕ := 4
def additional_stars_needed : ℕ := 307

-- Define the theorem
theorem luke_stars_made : 
  (stars_per_jar * bottles_to_fill) - additional_stars_needed = 33 := by
  sorry

end luke_stars_made_l762_76271


namespace trigonometric_expression_equals_five_fourths_l762_76245

theorem trigonometric_expression_equals_five_fourths :
  Real.sqrt 2 * Real.cos (π / 4) - Real.sin (π / 3) ^ 2 + Real.tan (π / 4) = 5 / 4 := by
  sorry

end trigonometric_expression_equals_five_fourths_l762_76245


namespace rectangle_area_18_l762_76256

/-- A rectangle with base twice the height and area equal to perimeter has area 18 -/
theorem rectangle_area_18 (h : ℝ) (b : ℝ) (area : ℝ) (perimeter : ℝ) : 
  b = 2 * h →                        -- base is twice the height
  area = b * h →                     -- area formula
  perimeter = 2 * (b + h) →          -- perimeter formula
  area = perimeter →                 -- area is numerically equal to perimeter
  area = 18 :=                       -- prove that area is 18
by sorry

end rectangle_area_18_l762_76256


namespace max_value_of_sum_of_square_roots_l762_76274

theorem max_value_of_sum_of_square_roots (a b c : ℝ) : 
  0 ≤ a → 0 ≤ b → 0 ≤ c → a + b + c = 8 → 
  Real.sqrt (3 * a + 1) + Real.sqrt (3 * b + 1) + Real.sqrt (3 * c + 1) ≤ 9 ∧ 
  ∃ a b c, 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ a + b + c = 8 ∧
    Real.sqrt (3 * a + 1) + Real.sqrt (3 * b + 1) + Real.sqrt (3 * c + 1) = 9 :=
by sorry

end max_value_of_sum_of_square_roots_l762_76274


namespace stork_bird_difference_l762_76268

/-- Given initial birds, storks, and additional birds, calculate the difference between storks and total birds -/
theorem stork_bird_difference (initial_birds : ℕ) (storks : ℕ) (additional_birds : ℕ) : 
  initial_birds = 2 → storks = 6 → additional_birds = 3 →
  storks - (initial_birds + additional_birds) = 1 := by
  sorry

end stork_bird_difference_l762_76268


namespace line_plane_intersection_l762_76237

/-- Given a plane α and two intersecting planes that form a line l, 
    prove the direction vector of l and the sine of the angle between l and α. -/
theorem line_plane_intersection (x y z : ℝ) : 
  let α : ℝ → ℝ → ℝ → Prop := λ x y z => x + 2*y - 2*z + 1 = 0
  let plane1 : ℝ → ℝ → ℝ → Prop := λ x y z => x - y + 3 = 0
  let plane2 : ℝ → ℝ → ℝ → Prop := λ x y z => x - 2*z - 1 = 0
  let l : Set (ℝ × ℝ × ℝ) := {p | plane1 p.1 p.2.1 p.2.2 ∧ plane2 p.1 p.2.1 p.2.2}
  let direction_vector : ℝ × ℝ × ℝ := (2, 2, 1)
  let normal_vector : ℝ × ℝ × ℝ := (1, 2, -2)
  let angle_sine : ℝ := 4/9
  (∀ p ∈ l, ∃ t : ℝ, p = (t * direction_vector.1, t * direction_vector.2.1, t * direction_vector.2.2)) ∧
  (|normal_vector.1 * direction_vector.1 + normal_vector.2.1 * direction_vector.2.1 + normal_vector.2.2 * direction_vector.2.2| / 
   (Real.sqrt (normal_vector.1^2 + normal_vector.2.1^2 + normal_vector.2.2^2) * 
    Real.sqrt (direction_vector.1^2 + direction_vector.2.1^2 + direction_vector.2.2^2)) = angle_sine) :=
by sorry

end line_plane_intersection_l762_76237


namespace gcf_of_4140_and_9920_l762_76242

theorem gcf_of_4140_and_9920 : Nat.gcd 4140 9920 = 10 := by
  sorry

end gcf_of_4140_and_9920_l762_76242


namespace lemonade_problem_l762_76200

theorem lemonade_problem (V : ℝ) 
  (h1 : V > 0)
  (h2 : V / 10 = V - 2 * (V / 5))
  (h3 : V / 8 = V - 2 * (V / 5 + V / 20))
  (h4 : V / 3 = V - 2 * (V / 5 + V / 20 + 5 * V / 12)) :
  V / 6 = V - (V / 3) / 2 := by sorry

end lemonade_problem_l762_76200


namespace min_rectangles_to_cover_square_l762_76257

/-- The width of the rectangle. -/
def rectangle_width : ℕ := 3

/-- The height of the rectangle. -/
def rectangle_height : ℕ := 4

/-- The area of a single rectangle. -/
def rectangle_area : ℕ := rectangle_width * rectangle_height

/-- The side length of the square that can be covered exactly by the rectangles. -/
def square_side : ℕ := rectangle_area

/-- The area of the square. -/
def square_area : ℕ := square_side * square_side

/-- The number of rectangles needed to cover the square. -/
def num_rectangles : ℕ := square_area / rectangle_area

theorem min_rectangles_to_cover_square : 
  num_rectangles = 12 ∧ 
  square_area % rectangle_area = 0 ∧
  ∀ n : ℕ, n < square_side → n * n % rectangle_area ≠ 0 :=
sorry

end min_rectangles_to_cover_square_l762_76257


namespace inequality_equivalence_l762_76236

theorem inequality_equivalence (x : ℝ) : (x + 2) / (x - 1) > 3 ↔ 1 < x ∧ x < 5/2 :=
by sorry

end inequality_equivalence_l762_76236


namespace star_value_l762_76233

def star (a b : ℤ) : ℚ := (a : ℚ)⁻¹ + (b : ℚ)⁻¹

theorem star_value (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) (h1 : a + b = 15) (h2 : a * b = 56) :
  star a b = 15 / 56 := by
sorry

end star_value_l762_76233


namespace distance_traveled_l762_76210

/-- Given a speed of 20 km/hr and a travel time of 2.5 hours, prove that the distance traveled is 50 km. -/
theorem distance_traveled (speed : ℝ) (time : ℝ) (distance : ℝ) 
  (h1 : speed = 20)
  (h2 : time = 2.5)
  (h3 : distance = speed * time) : 
  distance = 50 := by
  sorry

end distance_traveled_l762_76210


namespace disinfectant_problem_l762_76287

/-- Represents the price and volume of a disinfectant brand -/
structure DisinfectantBrand where
  price : ℝ
  volume : ℝ

/-- Represents a purchasing plan -/
structure PurchasePlan where
  brand1_bottles : ℕ
  brand2_bottles : ℕ

/-- Calculates the total cost of a purchase plan -/
def totalCost (brand1 brand2 : DisinfectantBrand) (plan : PurchasePlan) : ℝ :=
  brand1.price * plan.brand1_bottles + brand2.price * plan.brand2_bottles

/-- Calculates the total volume of a purchase plan -/
def totalVolume (brand1 brand2 : DisinfectantBrand) (plan : PurchasePlan) : ℝ :=
  brand1.volume * plan.brand1_bottles + brand2.volume * plan.brand2_bottles

/-- Theorem stating the properties of the disinfectant purchase problem -/
theorem disinfectant_problem (brand1 brand2 : DisinfectantBrand) 
  (h1 : brand1.volume = 200)
  (h2 : brand2.volume = 500)
  (h3 : totalCost brand1 brand2 { brand1_bottles := 3, brand2_bottles := 2 } = 80)
  (h4 : totalCost brand1 brand2 { brand1_bottles := 1, brand2_bottles := 4 } = 110)
  (h5 : ∃ (plan : PurchasePlan), totalVolume brand1 brand2 plan = 4000 ∧ 
        plan.brand1_bottles > 0 ∧ plan.brand2_bottles > 0)
  (h6 : ∃ (plan : PurchasePlan), totalCost brand1 brand2 plan = 2500)
  (h7 : (2500 / (1000 * 10 : ℝ)) * (brand1.volume * brand1.price + brand2.volume * brand2.price) = 5000) :
  brand1.price = 10 ∧ brand2.price = 25 ∧ 
  (2500 / (1000 * 10 : ℝ)) * (brand1.volume * brand1.price + brand2.volume * brand2.price) / 1000 = 5 := by
  sorry


end disinfectant_problem_l762_76287


namespace four_propositions_true_l762_76282

theorem four_propositions_true : 
  (∀ x y : ℝ, (x ≠ 0 ∨ y ≠ 0) → x^2 + y^2 ≠ 0) ∧
  (∀ x y : ℝ, x^2 + y^2 ≠ 0 → (x ≠ 0 ∨ y ≠ 0)) ∧
  (¬ ∀ x y : ℝ, (x ≠ 0 ∨ y ≠ 0) → x^2 + y^2 ≠ 0) ∧
  (¬ ∀ x y : ℝ, x^2 + y^2 ≠ 0 → (x ≠ 0 ∨ y ≠ 0)) :=
by sorry

end four_propositions_true_l762_76282


namespace no_inscribed_sphere_after_truncation_l762_76208

/-- A convex polyhedron -/
structure ConvexPolyhedron where
  vertices : Set (ℝ × ℝ × ℝ)
  faces : Set (Set (ℝ × ℝ × ℝ))
  is_convex : Bool
  vertex_count : ℕ
  face_count : ℕ
  vertex_ge_face : vertex_count ≥ face_count

/-- Truncation operation on a convex polyhedron -/
def truncate (P : ConvexPolyhedron) : ConvexPolyhedron :=
  sorry

/-- A sphere -/
structure Sphere where
  center : ℝ × ℝ × ℝ
  radius : ℝ

/-- Predicate to check if a sphere is inscribed in a polyhedron -/
def is_inscribed (S : Sphere) (P : ConvexPolyhedron) : Prop :=
  sorry

/-- Theorem stating that a truncated convex polyhedron cannot have an inscribed sphere -/
theorem no_inscribed_sphere_after_truncation (P : ConvexPolyhedron) :
  ¬ ∃ (S : Sphere), is_inscribed S (truncate P) :=
sorry

end no_inscribed_sphere_after_truncation_l762_76208


namespace sequence_2009th_term_l762_76212

theorem sequence_2009th_term :
  let sequence : ℕ → ℕ := fun n => 2^(n - 1)
  sequence 2009 = 2^2008 := by
  sorry

end sequence_2009th_term_l762_76212


namespace function_max_min_range_l762_76258

open Real

theorem function_max_min_range (m : ℝ) (f : ℝ → ℝ) : 
  (∀ x, f x = m * sin (x + π/4) - Real.sqrt 2 * sin x) → 
  (∃ max min : ℝ, ∀ x ∈ Set.Ioo 0 (7*π/6), f x ≤ max ∧ min ≤ f x) →
  2 < m ∧ m < 3 + Real.sqrt 3 :=
by sorry

end function_max_min_range_l762_76258


namespace rectangle_side_length_l762_76201

/-- Given a rectangle with area 9a^2 - 6ab + 3a and one side length 3a, 
    the other side length is 3a - 2b + 1 -/
theorem rectangle_side_length (a b : ℝ) : 
  let area := 9*a^2 - 6*a*b + 3*a
  let side1 := 3*a
  let side2 := 3*a - 2*b + 1
  area = side1 * side2 := by sorry

end rectangle_side_length_l762_76201


namespace max_factors_theorem_l762_76270

def is_valid_pair (b n : ℕ) : Prop :=
  0 < b ∧ b ≤ 20 ∧ 0 < n ∧ n ≤ 20 ∧ b ≠ n

def num_factors (m : ℕ) : ℕ := (Nat.factors m).length + 1

def max_factors : ℕ := 81

theorem max_factors_theorem :
  ∀ b n : ℕ, is_valid_pair b n →
    num_factors (b^n) ≤ max_factors :=
by sorry

end max_factors_theorem_l762_76270


namespace car_speed_second_hour_l762_76222

/-- Given a car's speed in the first hour and its average speed over two hours,
    calculate its speed in the second hour. -/
theorem car_speed_second_hour 
  (speed_first_hour : ℝ) 
  (average_speed : ℝ) 
  (h1 : speed_first_hour = 20)
  (h2 : average_speed = 40) : 
  ∃ (speed_second_hour : ℝ), speed_second_hour = 60 := by
  sorry

#check car_speed_second_hour

end car_speed_second_hour_l762_76222


namespace sign_up_options_count_l762_76204

/-- The number of students. -/
def num_students : ℕ := 5

/-- The number of teams. -/
def num_teams : ℕ := 3

/-- The total number of sign-up options. -/
def total_options : ℕ := num_teams ^ num_students

/-- 
Theorem: Given 5 students and 3 teams, where each student must choose exactly one team,
the total number of possible sign-up combinations is 3^5.
-/
theorem sign_up_options_count :
  total_options = 243 := by sorry

end sign_up_options_count_l762_76204


namespace nine_digit_divisibility_l762_76291

theorem nine_digit_divisibility (A : ℕ) : 
  A < 10 →
  (457319808 * 10 + A) % 2 = 0 →
  (457319808 * 10 + A) % 5 = 0 →
  (457319808 * 10 + A) % 8 = 0 →
  (457319808 * 10 + A) % 10 = 0 →
  (457319808 * 10 + A) % 16 = 0 →
  A = 0 := by
sorry

end nine_digit_divisibility_l762_76291


namespace gcd_g_x_l762_76206

def g (x : ℤ) : ℤ := (4*x+5)*(5*x+2)*(11*x+8)*(3*x+7)

theorem gcd_g_x (x : ℤ) (h : 2520 ∣ x) : Int.gcd (g x) x = 280 := by
  sorry

end gcd_g_x_l762_76206


namespace profit_calculation_l762_76240

/-- The number of pencils bought by the store owner -/
def total_pencils : ℕ := 2000

/-- The cost price of each pencil in dollars -/
def cost_price : ℚ := 15 / 100

/-- The selling price of each pencil in dollars -/
def selling_price : ℚ := 30 / 100

/-- The desired profit in dollars -/
def desired_profit : ℚ := 150

/-- The number of pencils that must be sold to make the desired profit -/
def pencils_to_sell : ℕ := 1500

theorem profit_calculation :
  (pencils_to_sell : ℚ) * selling_price - (total_pencils : ℚ) * cost_price = desired_profit :=
sorry

end profit_calculation_l762_76240


namespace beaver_leaves_count_l762_76260

theorem beaver_leaves_count :
  ∀ (beaver_dens raccoon_dens : ℕ),
    beaver_dens = raccoon_dens + 3 →
    5 * beaver_dens = 6 * raccoon_dens →
    5 * beaver_dens = 90 :=
by
  sorry

#check beaver_leaves_count

end beaver_leaves_count_l762_76260


namespace sum_divisible_by_three_combinations_l762_76255

/-- The number of integers from 1 to 300 that give remainder 0, 1, or 2 when divided by 3 -/
def count_mod_3 : ℕ := 100

/-- The total number of ways to select 3 numbers from 1 to 300 such that their sum is divisible by 3 -/
def total_combinations : ℕ := 1485100

/-- The number of ways to choose 3 elements from a set of size n -/
def choose (n : ℕ) : ℕ := n * (n - 1) * (n - 2) / 6

theorem sum_divisible_by_three_combinations :
  3 * choose count_mod_3 + count_mod_3^3 = total_combinations :=
sorry

end sum_divisible_by_three_combinations_l762_76255


namespace band_sections_sum_l762_76273

theorem band_sections_sum (total : ℕ) (trumpet_frac trombone_frac clarinet_frac flute_frac : ℚ) : 
  total = 500 →
  trumpet_frac = 1/2 →
  trombone_frac = 3/25 →
  clarinet_frac = 23/100 →
  flute_frac = 2/25 →
  ⌊total * trumpet_frac⌋ + ⌊total * trombone_frac⌋ + ⌊total * clarinet_frac⌋ + ⌊total * flute_frac⌋ = 465 :=
by sorry

end band_sections_sum_l762_76273


namespace gain_amount_calculation_l762_76235

theorem gain_amount_calculation (selling_price : ℝ) (gain_percentage : ℝ) 
  (h1 : selling_price = 110)
  (h2 : gain_percentage = 0.10) : 
  let cost_price := selling_price / (1 + gain_percentage)
  selling_price - cost_price = 10 := by
sorry

end gain_amount_calculation_l762_76235


namespace union_of_M_and_N_l762_76238

def M : Set ℕ := {0, 1, 2}

def N : Set ℕ := {y | ∃ x ∈ M, y = x^2}

theorem union_of_M_and_N : M ∪ N = {0, 1, 2, 4} := by sorry

end union_of_M_and_N_l762_76238


namespace cement_theft_proof_l762_76293

/-- Represents the weight of cement bags in kilograms -/
structure BagWeight where
  small : Nat
  large : Nat

/-- Represents the number of cement bags -/
structure BagCount where
  small : Nat
  large : Nat

/-- Calculates the total weight of cement given bag weights and counts -/
def totalWeight (w : BagWeight) (c : BagCount) : Nat :=
  w.small * c.small + w.large * c.large

/-- Represents the manager's assumption of bag weight -/
def managerAssumedWeight : Nat := 25

theorem cement_theft_proof (w : BagWeight) (c : BagCount) 
  (h1 : w.small = 25)
  (h2 : w.large = 40)
  (h3 : c.small = 2 * c.large)
  (h4 : totalWeight w c - w.large * 60 = managerAssumedWeight * (c.small + c.large)) :
  totalWeight w c - w.large * 60 = 12000 := by
  sorry

end cement_theft_proof_l762_76293


namespace function_property_l762_76246

/-- Given a function f(x) = ax^5 + bx^3 + 3 where f(2023) = 16, prove that f(-2023) = -10 -/
theorem function_property (a b : ℝ) :
  let f : ℝ → ℝ := λ x => a * x^5 + b * x^3 + 3
  f 2023 = 16 → f (-2023) = -10 := by
  sorry

end function_property_l762_76246


namespace m_range_l762_76269

/-- The function f(x) defined in the problem -/
def f (m : ℝ) (x : ℝ) : ℝ := |x^2 - 4| + x^2 + m*x

/-- The condition that f has two distinct zero points in (0, 3) -/
def has_two_distinct_zeros (m : ℝ) : Prop :=
  ∃ x y, 0 < x ∧ x < 3 ∧ 0 < y ∧ y < 3 ∧ x ≠ y ∧ f m x = 0 ∧ f m y = 0

/-- The theorem stating the range of m -/
theorem m_range (m : ℝ) :
  has_two_distinct_zeros m → -14/3 < m ∧ m < -2 :=
sorry

end m_range_l762_76269


namespace optimal_partition_l762_76261

def minimizeAbsoluteErrors (diameters : List ℝ) : 
  ℝ × ℝ × ℝ := sorry

theorem optimal_partition (diameters : List ℝ) 
  (h1 : diameters.length = 120) 
  (h2 : List.Sorted (· ≤ ·) diameters) :
  let (d, a, b) := minimizeAbsoluteErrors diameters
  d = (diameters.get! 59 + diameters.get! 60) / 2 ∧
  a = (diameters.take 60).sum / 60 ∧
  b = (diameters.drop 60).sum / 60 :=
sorry

end optimal_partition_l762_76261


namespace differential_savings_example_l762_76265

/-- Calculates the differential savings when tax rate is reduced -/
def differential_savings (income : ℝ) (old_rate : ℝ) (new_rate : ℝ) : ℝ :=
  income * old_rate - income * new_rate

/-- Proves that the differential savings for a specific case is correct -/
theorem differential_savings_example : 
  differential_savings 34500 0.42 0.28 = 4830 := by
  sorry

end differential_savings_example_l762_76265


namespace x_fifth_plus_inverse_l762_76252

theorem x_fifth_plus_inverse (x : ℝ) (h_pos : x > 0) (h_eq : x^2 + 1/x^2 = 7) :
  x^5 + 1/x^5 = 123 := by
  sorry

end x_fifth_plus_inverse_l762_76252


namespace min_sum_of_product_l762_76226

theorem min_sum_of_product (a b : ℤ) (h : a * b = 144) : 
  ∀ x y : ℤ, x * y = 144 → a + b ≤ x + y ∧ ∃ a₀ b₀ : ℤ, a₀ * b₀ = 144 ∧ a₀ + b₀ = -145 :=
by sorry

end min_sum_of_product_l762_76226


namespace jim_total_cars_l762_76243

/-- The number of model cars Jim has -/
structure ModelCars where
  buicks : ℕ
  fords : ℕ
  chevys : ℕ

/-- Jim's collection of model cars satisfying the given conditions -/
def jim_collection : ModelCars :=
  { buicks := 220,
    fords := 55,
    chevys := 26 }

/-- Theorem stating the total number of model cars Jim has -/
theorem jim_total_cars :
  jim_collection.buicks = 220 ∧
  jim_collection.buicks = 4 * jim_collection.fords ∧
  jim_collection.fords = 2 * jim_collection.chevys + 3 →
  jim_collection.buicks + jim_collection.fords + jim_collection.chevys = 301 := by
  sorry

#eval jim_collection.buicks + jim_collection.fords + jim_collection.chevys

end jim_total_cars_l762_76243


namespace partner_a_contribution_l762_76209

/-- A business partnership where two partners contribute capital for different durations and share profits proportionally. -/
structure BusinessPartnership where
  /-- Duration (in months) that Partner A's capital is used -/
  duration_a : ℕ
  /-- Duration (in months) that Partner B's capital is used -/
  duration_b : ℕ
  /-- Fraction of profit received by Partner B -/
  profit_share_b : ℚ
  /-- Fraction of capital contributed by Partner A -/
  capital_fraction_a : ℚ

/-- Theorem stating that under given conditions, Partner A's capital contribution is 1/4 -/
theorem partner_a_contribution
  (bp : BusinessPartnership)
  (h1 : bp.duration_a = 15)
  (h2 : bp.duration_b = 10)
  (h3 : bp.profit_share_b = 2/3)
  : bp.capital_fraction_a = 1/4 := by
  sorry


end partner_a_contribution_l762_76209


namespace change_difference_is_thirty_percent_l762_76275

-- Define the initial and final percentages
def initial_yes : ℚ := 40 / 100
def initial_no : ℚ := 30 / 100
def initial_undecided : ℚ := 30 / 100
def final_yes : ℚ := 60 / 100
def final_no : ℚ := 20 / 100
def final_undecided : ℚ := 20 / 100

-- Define the minimum and maximum change percentages
def min_change : ℚ := max 0 (final_yes - initial_yes)
def max_change : ℚ := min 1 (initial_no + initial_undecided + abs (final_yes - initial_yes))

-- Theorem statement
theorem change_difference_is_thirty_percent :
  max_change - min_change = 30 / 100 := by
  sorry

end change_difference_is_thirty_percent_l762_76275


namespace bobs_grade_l762_76250

theorem bobs_grade (jenny_grade jason_grade bob_grade : ℕ) : 
  jenny_grade = 95 →
  jason_grade = jenny_grade - 25 →
  bob_grade = jason_grade / 2 →
  bob_grade = 35 := by
sorry

end bobs_grade_l762_76250


namespace shaded_square_fraction_l762_76207

theorem shaded_square_fraction :
  let large_square_side : ℝ := 6
  let small_square_side : ℝ := Real.sqrt 2
  let large_square_area : ℝ := large_square_side ^ 2
  let small_square_area : ℝ := small_square_side ^ 2
  (small_square_area / large_square_area) = (1 : ℝ) / 18 := by sorry

end shaded_square_fraction_l762_76207


namespace power_of_fraction_five_sevenths_fourth_l762_76292

theorem power_of_fraction_five_sevenths_fourth : (5 / 7 : ℚ) ^ 4 = 625 / 2401 := by
  sorry

end power_of_fraction_five_sevenths_fourth_l762_76292


namespace number_of_boys_l762_76295

theorem number_of_boys (total_pupils : ℕ) (number_of_girls : ℕ) 
  (h1 : total_pupils = 929) 
  (h2 : number_of_girls = 542) : 
  total_pupils - number_of_girls = 387 := by
  sorry

end number_of_boys_l762_76295


namespace cab_driver_fifth_day_income_verify_cab_driver_income_l762_76239

/-- Calculates the cab driver's income on the fifth day given the income for the first four days and the average income for five days. -/
theorem cab_driver_fifth_day_income 
  (income_day1 income_day2 income_day3 income_day4 : ℚ) 
  (average_income : ℚ) : ℚ :=
  let total_income := 5 * average_income
  let sum_four_days := income_day1 + income_day2 + income_day3 + income_day4
  total_income - sum_four_days

/-- Verifies that the calculated fifth day income is correct given the specific values from the problem. -/
theorem verify_cab_driver_income : 
  cab_driver_fifth_day_income 300 150 750 400 420 = 500 := by
  sorry

end cab_driver_fifth_day_income_verify_cab_driver_income_l762_76239


namespace shaded_area_proof_l762_76230

/-- Given a grid and two right triangles, prove the area of the smaller triangle -/
theorem shaded_area_proof (grid_width grid_height : ℕ) 
  (large_triangle_base large_triangle_height : ℕ)
  (small_triangle_base small_triangle_height : ℕ) :
  grid_width = 15 →
  grid_height = 5 →
  large_triangle_base = grid_width →
  large_triangle_height = grid_height - 1 →
  small_triangle_base = 12 →
  small_triangle_height = 3 →
  (small_triangle_base * small_triangle_height) / 2 = 18 := by
  sorry


end shaded_area_proof_l762_76230


namespace root_equation_implies_expression_value_l762_76244

theorem root_equation_implies_expression_value (a : ℝ) :
  a^2 - 2*a - 2 = 0 →
  (1 - 1/(a + 1)) / (a^3 / (a^2 + 2*a + 1)) = 1/2 := by
  sorry

end root_equation_implies_expression_value_l762_76244


namespace carlos_cookie_count_l762_76259

/-- Represents the shape of a cookie -/
inductive CookieShape
  | Rectangle
  | Square

/-- Represents a cookie with its shape and area -/
structure Cookie where
  shape : CookieShape
  area : ℝ

/-- Represents a batch of cookies -/
structure CookieBatch where
  shape : CookieShape
  totalArea : ℝ
  count : ℕ

/-- The theorem to be proved -/
theorem carlos_cookie_count 
  (anne_batch : CookieBatch)
  (carlos_batch : CookieBatch)
  (h1 : anne_batch.shape = CookieShape.Rectangle)
  (h2 : carlos_batch.shape = CookieShape.Square)
  (h3 : anne_batch.totalArea = 180)
  (h4 : anne_batch.count = 15)
  (h5 : anne_batch.totalArea = carlos_batch.totalArea) :
  carlos_batch.count = 20 := by
  sorry


end carlos_cookie_count_l762_76259


namespace tim_bodyguard_cost_l762_76211

def bodyguards_cost (num_bodyguards : ℕ) (hourly_rate : ℕ) (hours_per_day : ℕ) (days_per_week : ℕ) : ℕ :=
  num_bodyguards * hourly_rate * hours_per_day * days_per_week

theorem tim_bodyguard_cost :
  bodyguards_cost 2 20 8 7 = 2240 :=
by sorry

end tim_bodyguard_cost_l762_76211


namespace anya_balloons_l762_76225

theorem anya_balloons (total : ℕ) (colors : ℕ) (anya_fraction : ℚ) 
  (h1 : total = 672) 
  (h2 : colors = 4) 
  (h3 : anya_fraction = 1/2) : 
  (total / colors) * anya_fraction = 84 := by
  sorry

end anya_balloons_l762_76225


namespace james_yearly_pages_l762_76297

/-- Calculates the number of pages James writes in a year -/
def pages_per_year (pages_per_letter : ℕ) (num_friends : ℕ) (times_per_week : ℕ) (weeks_per_year : ℕ) : ℕ :=
  pages_per_letter * num_friends * times_per_week * weeks_per_year

/-- Proves that James writes 624 pages in a year -/
theorem james_yearly_pages :
  pages_per_year 3 2 2 52 = 624 := by
  sorry

end james_yearly_pages_l762_76297


namespace bert_grocery_spending_l762_76254

/-- Represents Bert's spending scenario -/
structure BertSpending where
  initial_amount : ℚ
  hardware_fraction : ℚ
  dry_cleaning_amount : ℚ
  final_amount : ℚ

/-- Calculates the fraction spent at the grocery store -/
def grocery_fraction (b : BertSpending) : ℚ :=
  let remaining_before_grocery := b.initial_amount - (b.hardware_fraction * b.initial_amount) - b.dry_cleaning_amount
  let spent_at_grocery := remaining_before_grocery - b.final_amount
  spent_at_grocery / remaining_before_grocery

/-- Theorem stating that Bert spent 1/2 of his remaining money at the grocery store -/
theorem bert_grocery_spending (b : BertSpending) 
  (h1 : b.initial_amount = 52)
  (h2 : b.hardware_fraction = 1/4)
  (h3 : b.dry_cleaning_amount = 9)
  (h4 : b.final_amount = 15) :
  grocery_fraction b = 1/2 := by
  sorry


end bert_grocery_spending_l762_76254


namespace k_value_l762_76279

theorem k_value (a b k : ℝ) 
  (h1 : 4^a = k) 
  (h2 : 9^b = k) 
  (h3 : 1/a + 1/b = 2) : k = 6 := by
sorry

end k_value_l762_76279


namespace domain_f_minus_one_l762_76224

-- Define a real-valued function f
def f : ℝ → ℝ := sorry

-- Define the domain of f(x+1)
def domain_f_plus_one : Set ℝ := Set.Icc (-2) 3

-- Theorem statement
theorem domain_f_minus_one (h : ∀ x, f (x + 1) ∈ domain_f_plus_one ↔ x ∈ domain_f_plus_one) :
  ∀ x, f (x - 1) ∈ Set.Icc 0 5 ↔ x ∈ Set.Icc 0 5 :=
sorry

end domain_f_minus_one_l762_76224


namespace fold_point_sum_l762_76263

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Determines if a line folds one point onto another -/
def folds (l : Line) (p1 p2 : Point) : Prop :=
  let midpoint : Point := ⟨(p1.x + p2.x) / 2, (p1.y + p2.y) / 2⟩
  midpoint.y = l.slope * midpoint.x + l.intercept

/-- The main theorem -/
theorem fold_point_sum (l : Line) :
  folds l ⟨1, 3⟩ ⟨5, 1⟩ →
  folds l ⟨8, 4⟩ ⟨m, n⟩ →
  m + n = 32 / 3 := by
  sorry

end fold_point_sum_l762_76263


namespace cell_division_problem_l762_76283

/-- The number of cells after a given time, starting with one cell -/
def num_cells (division_time : ℕ) (elapsed_time : ℕ) : ℕ :=
  2^(elapsed_time / division_time)

/-- The time between cell divisions in minutes -/
def division_time : ℕ := 30

/-- The total elapsed time in minutes -/
def total_time : ℕ := 4 * 60 + 30

theorem cell_division_problem :
  num_cells division_time total_time = 512 := by
  sorry

end cell_division_problem_l762_76283


namespace remainder_sum_l762_76281

theorem remainder_sum (x : ℤ) (h : x % 6 = 3) : 
  (x^2 % 30) + (x^3 % 11) = 14 := by sorry

end remainder_sum_l762_76281


namespace absolute_value_sum_zero_l762_76217

theorem absolute_value_sum_zero (x y : ℝ) :
  |x - 3| + |y + 2| = 0 → (y - x = -5 ∧ x * y = -6) := by
  sorry

end absolute_value_sum_zero_l762_76217


namespace smallest_perfect_square_with_perfect_square_factors_l762_76285

/-- A function that returns the number of positive integer factors of a natural number -/
def num_factors (n : ℕ) : ℕ := sorry

/-- A function that checks if a natural number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop := sorry

/-- A function that checks if a natural number is a perfect square -/
def is_perfect_square_num (n : ℕ) : Prop := sorry

theorem smallest_perfect_square_with_perfect_square_factors : 
  ∀ n : ℕ, n > 1 → is_perfect_square n → is_perfect_square_num (num_factors n) → n ≥ 36 :=
sorry

end smallest_perfect_square_with_perfect_square_factors_l762_76285


namespace prob_sum_four_twice_l762_76299

/-- A die with 3 sides --/
def ThreeSidedDie : Type := Fin 3

/-- The sum of two dice rolls --/
def diceSum (d1 d2 : ThreeSidedDie) : Nat :=
  d1.val + d2.val + 2

/-- The probability of rolling a sum of 4 with two 3-sided dice --/
def probSumFour : ℚ :=
  3 / 9

/-- The probability of rolling a sum of 4 twice in a row with two 3-sided dice --/
theorem prob_sum_four_twice : probSumFour * probSumFour = 1 / 9 := by
  sorry

end prob_sum_four_twice_l762_76299


namespace dan_picked_nine_apples_l762_76288

/-- The number of apples Benny picked -/
def benny_apples : ℕ := 2

/-- The difference between Dan's and Benny's apple count -/
def difference : ℕ := 7

/-- The number of apples Dan picked -/
def dan_apples : ℕ := benny_apples + difference

theorem dan_picked_nine_apples : dan_apples = 9 := by
  sorry

end dan_picked_nine_apples_l762_76288


namespace total_cost_price_calculation_l762_76248

theorem total_cost_price_calculation (sp1 sp2 sp3 : ℚ) (profit1 loss2 profit3 : ℚ) :
  sp1 = 600 ∧ profit1 = 25/100 ∧
  sp2 = 800 ∧ loss2 = 20/100 ∧
  sp3 = 1000 ∧ profit3 = 30/100 →
  ∃ (cp1 cp2 cp3 : ℚ),
    cp1 = sp1 / (1 + profit1) ∧
    cp2 = sp2 / (1 - loss2) ∧
    cp3 = sp3 / (1 + profit3) ∧
    cp1 + cp2 + cp3 = 2249.23 := by
  sorry

end total_cost_price_calculation_l762_76248


namespace initial_apples_count_l762_76267

def cafeteria_apples (apples_handed_out : ℕ) (apples_per_pie : ℕ) (pies_made : ℕ) : ℕ :=
  apples_handed_out + apples_per_pie * pies_made

theorem initial_apples_count :
  cafeteria_apples 41 5 2 = 51 := by
  sorry

end initial_apples_count_l762_76267


namespace tennis_tournament_result_l762_76213

/-- Represents the number of participants with k points after m rounds in a tournament with 2^n participants. -/
def f (n m k : ℕ) : ℕ := 2^(n - m) * Nat.choose m k

/-- The rules of the tennis tournament. -/
structure TournamentRules where
  participants : ℕ
  victoryPoints : ℕ
  lossPoints : ℕ
  additionalPointRule : Bool
  pairingRule : Bool

/-- The specific tournament in question. -/
def tennisTournament : TournamentRules where
  participants := 256  -- Including two fictitious participants
  victoryPoints := 1
  lossPoints := 0
  additionalPointRule := true
  pairingRule := true

/-- The theorem to be proved. -/
theorem tennis_tournament_result (t : TournamentRules) (h : t = tennisTournament) :
  f 8 8 5 = 56 := by
  sorry

end tennis_tournament_result_l762_76213


namespace largest_prefix_for_two_digit_quotient_l762_76280

def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

theorem largest_prefix_for_two_digit_quotient :
  ∀ n : ℕ, n ≤ 9 →
    (is_two_digit ((n * 100 + 72) / 6) ↔ n ≤ 5) ∧
    (∀ m : ℕ, m ≤ 9 ∧ m > 5 → ¬(is_two_digit ((m * 100 + 72) / 6))) :=
by sorry

end largest_prefix_for_two_digit_quotient_l762_76280


namespace bridget_apples_proof_l762_76253

/-- Represents the number of apples Bridget initially bought -/
def initial_apples : ℕ := 20

/-- Represents the number of apples Bridget ate -/
def apples_eaten : ℕ := 2

/-- Represents the number of apples Bridget gave to Cassie -/
def apples_to_cassie : ℕ := 5

/-- Represents the number of apples Bridget kept for herself -/
def apples_kept : ℕ := 6

theorem bridget_apples_proof :
  let remaining_after_eating := initial_apples - apples_eaten
  let remaining_after_ann := remaining_after_eating - (remaining_after_eating / 3)
  let final_remaining := remaining_after_ann - apples_to_cassie
  final_remaining = apples_kept :=
by sorry

end bridget_apples_proof_l762_76253


namespace star_three_six_eq_seven_l762_76296

/-- The ☆ operation on rational numbers -/
def star (a : ℚ) (x y : ℚ) : ℚ := a^2 * x + a * y + 1

/-- Theorem: If 1 ☆ 2 = 3, then 3 ☆ 6 = 7 -/
theorem star_three_six_eq_seven (a : ℚ) (h : star a 1 2 = 3) : star a 3 6 = 7 := by
  sorry

end star_three_six_eq_seven_l762_76296


namespace negation_of_existence_negation_of_greater_than_100_l762_76223

theorem negation_of_existence (p : ℕ → Prop) : 
  (¬ ∃ n, p n) ↔ (∀ n, ¬ p n) :=
by sorry

theorem negation_of_greater_than_100 : 
  (¬ ∃ n : ℕ, 2^n > 100) ↔ (∀ n : ℕ, 2^n ≤ 100) :=
by sorry

end negation_of_existence_negation_of_greater_than_100_l762_76223


namespace shirt_price_change_l762_76231

theorem shirt_price_change (P : ℝ) (x : ℝ) (h : P > 0) :
  P * (1 + x / 100) * (1 - x / 100) = 0.84 * P → x = 40 := by
  sorry

end shirt_price_change_l762_76231


namespace negation_of_universal_proposition_l762_76202

open Real

theorem negation_of_universal_proposition :
  (¬ ∀ m : ℝ, ∃ x : ℝ, x^2 + x + m = 0) ↔ 
  (∃ m : ℝ, ∀ x : ℝ, x^2 + x + m ≠ 0) :=
by sorry

end negation_of_universal_proposition_l762_76202


namespace product_one_sum_greater_than_reciprocals_l762_76247

theorem product_one_sum_greater_than_reciprocals 
  (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_prod : a * b * c = 1) 
  (h_sum : a + b + c > 1/a + 1/b + 1/c) : 
  (a > 1 ∧ b < 1 ∧ c < 1) ∨ 
  (a < 1 ∧ b > 1 ∧ c < 1) ∨ 
  (a < 1 ∧ b < 1 ∧ c > 1) :=
sorry

end product_one_sum_greater_than_reciprocals_l762_76247


namespace sqrt_expression_equality_l762_76229

theorem sqrt_expression_equality (x : ℝ) (hx : x ≠ 0) :
  Real.sqrt (1 + ((x^6 - x^3) / (3 * x^3))^2) = (x^3 - 1 + Real.sqrt (x^6 - 2*x^3 + 10)) / 3 := by
  sorry

end sqrt_expression_equality_l762_76229


namespace magic_sum_values_l762_76249

/-- Represents a triangle configuration with 6 numbers -/
structure TriangleConfig where
  vertices : Fin 6 → Nat
  distinct : ∀ i j, i ≠ j → vertices i ≠ vertices j
  range : ∀ i, vertices i ∈ ({1, 2, 3, 4, 5, 6} : Set Nat)

/-- The sum of numbers on one side of the triangle -/
def sideSum (config : TriangleConfig) : Nat :=
  config.vertices 0 + config.vertices 1 + config.vertices 2

/-- All sides have the same sum -/
def validConfig (config : TriangleConfig) : Prop :=
  sideSum config = config.vertices 0 + config.vertices 3 + config.vertices 5 ∧
  sideSum config = config.vertices 2 + config.vertices 4 + config.vertices 5

theorem magic_sum_values :
  ∃ (config : TriangleConfig), validConfig config ∧
  sideSum config ∈ ({9, 10, 11, 12} : Set Nat) ∧
  ∀ (otherConfig : TriangleConfig),
    validConfig otherConfig →
    sideSum otherConfig ∈ ({9, 10, 11, 12} : Set Nat) := by
  sorry

end magic_sum_values_l762_76249


namespace fraction_exponent_product_l762_76205

theorem fraction_exponent_product : 
  (8 / 9 : ℚ) ^ 3 * (1 / 3 : ℚ) ^ 3 = 512 / 19683 := by sorry

end fraction_exponent_product_l762_76205


namespace relief_supplies_total_l762_76290

/-- The total amount of relief supplies in tons -/
def total_supplies : ℝ := 644

/-- Team A's daily transport capacity in tons -/
def team_a_capacity : ℝ := 64.4

/-- The percentage by which Team A's capacity exceeds Team B's -/
def capacity_difference_percentage : ℝ := 75

/-- The additional amount Team A has transported when it reaches half the total supplies -/
def additional_transport : ℝ := 138

/-- Theorem stating the total amount of relief supplies -/
theorem relief_supplies_total : 
  ∃ (team_b_capacity : ℝ),
    team_a_capacity = team_b_capacity * (1 + capacity_difference_percentage / 100) ∧
    (total_supplies / 2) - (total_supplies / 2 - additional_transport) = 
      (team_a_capacity - team_b_capacity) * (total_supplies / (2 * team_a_capacity)) ∧
    total_supplies = 644 :=
by sorry

end relief_supplies_total_l762_76290


namespace extracurricular_materials_selection_l762_76241

theorem extracurricular_materials_selection (n : Nat) (k : Nat) (m : Nat) : 
  n = 6 → k = 2 → m = 1 → 
  (Nat.choose n m) * (m * (n - m) * (n - m - 1)) = 120 := by
  sorry

end extracurricular_materials_selection_l762_76241


namespace lincoln_county_houses_l762_76221

/-- The number of houses in Lincoln County after a housing boom -/
def houses_after_boom (original : ℕ) (new_built : ℕ) : ℕ :=
  original + new_built

/-- Theorem: The number of houses in Lincoln County after the housing boom is 118558 -/
theorem lincoln_county_houses :
  houses_after_boom 20817 97741 = 118558 := by
  sorry

end lincoln_county_houses_l762_76221


namespace line_not_in_first_quadrant_l762_76264

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Predicate to check if a line passes through the first quadrant -/
def passesFirstQuadrant (l : Line) : Prop :=
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ l.a * x + l.b * y + l.c = 0

theorem line_not_in_first_quadrant (l : Line) 
  (h1 : ¬passesFirstQuadrant l) 
  (h2 : l.a * l.b > 0) : 
  l.a * l.c ≥ 0 := by
  sorry

end line_not_in_first_quadrant_l762_76264


namespace smallest_N_proof_l762_76262

/-- The smallest natural number N such that N × 999 consists entirely of the digit seven in its decimal representation -/
def smallest_N : ℕ := 778556334111889667445223

/-- Predicate to check if a natural number consists entirely of the digit seven -/
def all_sevens (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 7

theorem smallest_N_proof :
  (smallest_N * 999).digits 10 = List.replicate 27 7 ∧
  ∀ m : ℕ, m < smallest_N → ¬(all_sevens (m * 999)) :=
by sorry

end smallest_N_proof_l762_76262


namespace largest_non_sum_of_composites_l762_76286

def is_composite (n : ℕ) : Prop :=
  ∃ k : ℕ, 1 < k ∧ k < n ∧ n % k = 0

def sum_of_two_composites (n : ℕ) : Prop :=
  ∃ a b : ℕ, is_composite a ∧ is_composite b ∧ a + b = n

theorem largest_non_sum_of_composites :
  (∀ n : ℕ, n > 11 → sum_of_two_composites n) ∧
  ¬sum_of_two_composites 11 :=
sorry

end largest_non_sum_of_composites_l762_76286


namespace product_equality_implies_sum_l762_76219

theorem product_equality_implies_sum (g h : ℚ) : 
  (∀ d : ℚ, (8*d^2 - 5*d + g) * (2*d^2 + h*d - 9) = 16*d^4 + 21*d^3 - 73*d^2 - 41*d + 45) →
  g + h = -82/25 := by
sorry

end product_equality_implies_sum_l762_76219


namespace lindsay_dolls_l762_76266

theorem lindsay_dolls (blonde : ℕ) (brown black red : ℕ) : 
  blonde = 6 →
  brown = 3 * blonde →
  black = brown / 2 →
  red = 2 * black →
  (black + brown + red) - blonde = 39 := by
  sorry

end lindsay_dolls_l762_76266


namespace circle_tangent_line_l762_76220

/-- A circle in polar coordinates with equation ρ = 2cosθ -/
def Circle (ρ θ : ℝ) : Prop := ρ = 2 * Real.cos θ

/-- A line in polar coordinates with equation 3ρcosθ + 4ρsinθ + a = 0 -/
def Line (ρ θ a : ℝ) : Prop := 3 * ρ * Real.cos θ + 4 * ρ * Real.sin θ + a = 0

/-- The circle is tangent to the line -/
def IsTangent (a : ℝ) : Prop :=
  ∃! (ρ θ : ℝ), Circle ρ θ ∧ Line ρ θ a

theorem circle_tangent_line (a : ℝ) :
  IsTangent a ↔ (a = -8 ∨ a = 2) :=
sorry

end circle_tangent_line_l762_76220


namespace fourteenth_root_of_unity_l762_76277

theorem fourteenth_root_of_unity : 
  ∃ n : ℕ, n ≤ 13 ∧ 
  (Complex.tan (π / 7) + Complex.I) / (Complex.tan (π / 7) - Complex.I) = 
  Complex.exp (Complex.I * (2 * ↑n * π / 14)) :=
by sorry

end fourteenth_root_of_unity_l762_76277


namespace digits_of_product_l762_76232

theorem digits_of_product : ∃ n : ℕ, n > 0 ∧ 10^(n-1) ≤ 2^15 * 5^10 * 3 ∧ 2^15 * 5^10 * 3 < 10^n ∧ n = 12 := by
  sorry

end digits_of_product_l762_76232


namespace fifth_degree_polynomial_existence_l762_76272

theorem fifth_degree_polynomial_existence : ∃ (P : ℝ → ℝ),
  (∀ x : ℝ, P x = 0 → x < 0) ∧
  (∀ x : ℝ, (deriv P) x = 0 → x > 0) ∧
  (∃ x : ℝ, P x = 0 ∧ ∀ y : ℝ, y ≠ x → P y ≠ 0) ∧
  (∃ x : ℝ, (deriv P) x = 0 ∧ ∀ y : ℝ, y ≠ x → (deriv P) y ≠ 0) ∧
  (∃ a b c d e f : ℝ, ∀ x : ℝ, P x = a*x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + f) :=
by sorry

end fifth_degree_polynomial_existence_l762_76272


namespace unique_root_of_sin_plus_constant_l762_76227

theorem unique_root_of_sin_plus_constant :
  ∃! x : ℝ, x = Real.sin x + 1993 := by sorry

end unique_root_of_sin_plus_constant_l762_76227


namespace actual_distance_travelled_l762_76298

theorem actual_distance_travelled (speed1 speed2 : ℝ) (extra_distance : ℝ) 
  (h1 : speed1 = 10)
  (h2 : speed2 = 14)
  (h3 : extra_distance = 20)
  (h4 : ∀ d : ℝ, d / speed1 = (d + extra_distance) / speed2) :
  ∃ d : ℝ, d = 50 ∧ d / speed1 = (d + extra_distance) / speed2 := by
sorry

end actual_distance_travelled_l762_76298


namespace max_fraction_value_101_l762_76215

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def max_fraction_value (n : ℕ) : ℚ :=
  (factorial n) / 4

theorem max_fraction_value_101 :
  ∀ (f : ℚ), f = max_fraction_value 101 ∨ f < max_fraction_value 101 :=
sorry

end max_fraction_value_101_l762_76215


namespace pyramid_cross_section_distance_l762_76289

/-- Represents a right hexagonal pyramid -/
structure RightHexagonalPyramid where
  -- Add any necessary fields

/-- Represents a cross-section of the pyramid -/
structure CrossSection where
  area : ℝ
  distance_from_apex : ℝ

theorem pyramid_cross_section_distance
  (pyramid : RightHexagonalPyramid)
  (cs1 cs2 : CrossSection)
  (h : ℝ) :
  cs1.area = 125 * Real.sqrt 3 →
  cs2.area = 500 * Real.sqrt 3 →
  cs2.distance_from_apex - cs1.distance_from_apex = 10 →
  cs2.distance_from_apex = h →
  h = 20 := by
  sorry

#check pyramid_cross_section_distance

end pyramid_cross_section_distance_l762_76289


namespace hyperbola_tangent_line_l762_76214

/-- The equation of a tangent line to a hyperbola -/
theorem hyperbola_tangent_line (a b x₀ y₀ : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : x₀^2 / a^2 - y₀^2 / b^2 = 1) :
  ∃ (x y : ℝ → ℝ), ∀ t, 
    (x t)^2 / a^2 - (y t)^2 / b^2 = 1 ∧ 
    x 0 = x₀ ∧ 
    y 0 = y₀ ∧
    (∀ s, x₀ * (x s) / a^2 - y₀ * (y s) / b^2 = 1) :=
sorry

end hyperbola_tangent_line_l762_76214


namespace cone_surface_area_l762_76278

/-- Given a cone with base radius 3 and lateral surface that unfolds into a sector
    with central angle 2π/3, its surface area is 36π. -/
theorem cone_surface_area (r : ℝ) (θ : ℝ) (S : ℝ) : 
  r = 3 → 
  θ = 2 * Real.pi / 3 →
  S = r * r * Real.pi + r * (r * θ) →
  S = 36 * Real.pi := by
  sorry

end cone_surface_area_l762_76278


namespace felicity_gas_usage_l762_76228

theorem felicity_gas_usage (adhira : ℝ) 
  (h1 : 4 * adhira - 5 + adhira = 30) : 
  4 * adhira - 5 = 23 := by
  sorry

end felicity_gas_usage_l762_76228


namespace markup_percentage_is_40_percent_l762_76216

/-- Proves that the markup percentage on the selling price of a desk is 40% given the specified conditions. -/
theorem markup_percentage_is_40_percent
  (purchase_price : ℝ)
  (selling_price : ℝ)
  (markup : ℝ)
  (gross_profit : ℝ)
  (h1 : purchase_price = 150)
  (h2 : selling_price = purchase_price + markup)
  (h3 : gross_profit = 100)
  (h4 : gross_profit = selling_price - purchase_price) :
  (markup / selling_price) * 100 = 40 := by
  sorry

end markup_percentage_is_40_percent_l762_76216


namespace inverse_proportion_inequality_l762_76203

theorem inverse_proportion_inequality (x₁ x₂ y₁ y₂ : ℝ) :
  x₁ < 0 ∧ 0 < x₂ ∧ y₁ = 3 / x₁ ∧ y₂ = 3 / x₂ → y₁ < 0 ∧ 0 < y₂ := by
  sorry

end inverse_proportion_inequality_l762_76203


namespace solve_for_n_l762_76251

def first_seven_multiples_of_four : List ℕ := [4, 8, 12, 16, 20, 24, 28]

def a : ℚ := (first_seven_multiples_of_four.sum : ℚ) / 7

def b (n : ℕ) : ℚ := 2 * n

theorem solve_for_n (n : ℕ) (h : n > 0) :
  a^2 - (b n)^2 = 0 → n = 8 := by sorry

end solve_for_n_l762_76251

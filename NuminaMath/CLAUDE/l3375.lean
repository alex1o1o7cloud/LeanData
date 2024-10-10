import Mathlib

namespace triangle_radii_product_l3375_337593

theorem triangle_radii_product (a b c : ℝ) (ha : a = 26) (hb : b = 28) (hc : c = 30) :
  let p := (a + b + c) / 2
  let s := Real.sqrt (p * (p - a) * (p - b) * (p - c))
  let r := s / p
  let R := (a * b * c) / (4 * s)
  R * r = 130 := by sorry

end triangle_radii_product_l3375_337593


namespace smallest_divisible_by_9_11_13_l3375_337532

theorem smallest_divisible_by_9_11_13 : ∃ n : ℕ, n > 0 ∧ 
  9 ∣ n ∧ 11 ∣ n ∧ 13 ∣ n ∧ 
  ∀ m : ℕ, m > 0 → 9 ∣ m → 11 ∣ m → 13 ∣ m → n ≤ m :=
by
  use 1287
  sorry

end smallest_divisible_by_9_11_13_l3375_337532


namespace worker_count_l3375_337577

theorem worker_count : ∃ (W : ℕ), 
  (W > 0) ∧ 
  (∃ (C : ℚ), C > 0 ∧ W * C = 300000) ∧
  (W * (C + 50) = 325000) ∧
  W = 500 := by
  sorry

end worker_count_l3375_337577


namespace total_students_correct_l3375_337513

/-- Represents the total number of high school students -/
def total_students : ℕ := 1800

/-- Represents the sample size -/
def sample_size : ℕ := 45

/-- Represents the number of second-year students -/
def second_year_students : ℕ := 600

/-- Represents the number of second-year students selected in the sample -/
def selected_second_year : ℕ := 15

/-- Theorem stating that the total number of students is correct given the sampling information -/
theorem total_students_correct :
  (total_students : ℚ) / sample_size = (second_year_students : ℚ) / selected_second_year :=
sorry

end total_students_correct_l3375_337513


namespace parallel_lines_parallelograms_l3375_337534

/-- The number of ways to choose 2 items from n items -/
def choose_two (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The number of parallelograms formed by intersecting sets of parallel lines -/
def parallelograms_count (set1 : ℕ) (set2 : ℕ) : ℕ :=
  (choose_two set1) * (choose_two set2)

theorem parallel_lines_parallelograms :
  parallelograms_count 3 5 = 30 := by
  sorry

end parallel_lines_parallelograms_l3375_337534


namespace floor_negative_seven_fourths_l3375_337544

theorem floor_negative_seven_fourths : ⌊(-7 : ℚ) / 4⌋ = -2 := by
  sorry

end floor_negative_seven_fourths_l3375_337544


namespace rectangle_area_l3375_337505

-- Define the rectangle ABCD
def rectangle (AB DE : ℝ) : Prop :=
  DE - AB = 9 ∧ DE > AB ∧ AB > 0

-- Define the relationship between areas of trapezoid ABCE and triangle ADE
def area_relation (AB DE : ℝ) : Prop :=
  (AB * DE) / 2 = 5 * ((DE - AB) * AB / 2)

-- Define the relationship between perimeters
def perimeter_relation (AB : ℝ) : Prop :=
  AB * 4/3 = 68

-- Main theorem
theorem rectangle_area (AB DE : ℝ) :
  rectangle AB DE →
  area_relation AB DE →
  perimeter_relation AB →
  AB * DE = 3060 :=
by sorry

end rectangle_area_l3375_337505


namespace room_breadth_calculation_l3375_337511

/-- Given a room with specified dimensions and carpeting costs, calculate its breadth. -/
theorem room_breadth_calculation (room_length : ℝ) (carpet_width : ℝ) (carpet_cost_per_meter : ℝ) (total_cost : ℝ) :
  room_length = 15 →
  carpet_width = 0.75 →
  carpet_cost_per_meter = 0.3 →
  total_cost = 36 →
  (total_cost / carpet_cost_per_meter) * carpet_width / room_length = 6 := by
  sorry

end room_breadth_calculation_l3375_337511


namespace inequality_solution_set_l3375_337531

-- Define the inequality function
def f (m x : ℝ) : ℝ := m * x^2 + (2*m - 1) * x - 2

-- Define the solution set
def solution_set (m : ℝ) : Set ℝ :=
  if m < -1/2 then Set.Ioo (-2) (1/m)
  else if m = -1/2 then ∅
  else if -1/2 < m ∧ m < 0 then Set.Ioo (1/m) (-2)
  else if m = 0 then Set.Ioi (-2)
  else Set.union (Set.Iio (-2)) (Set.Ioi (1/m))

-- Theorem statement
theorem inequality_solution_set (m : ℝ) :
  {x : ℝ | f m x > 0} = solution_set m :=
sorry

end inequality_solution_set_l3375_337531


namespace binomial_16_12_l3375_337550

theorem binomial_16_12 : Nat.choose 16 12 = 1820 := by
  sorry

end binomial_16_12_l3375_337550


namespace bottle_cap_distribution_l3375_337516

/-- Given 18 bottle caps shared among 6 friends, prove that each friend receives 3 bottle caps. -/
theorem bottle_cap_distribution (total_caps : ℕ) (num_friends : ℕ) (caps_per_friend : ℕ) : 
  total_caps = 18 → num_friends = 6 → caps_per_friend = total_caps / num_friends → caps_per_friend = 3 := by
  sorry

end bottle_cap_distribution_l3375_337516


namespace newspaper_delivery_difference_l3375_337587

/-- Calculates the difference in monthly newspaper deliveries between Miranda and Jake -/
def monthly_delivery_difference (jake_weekly : ℕ) (miranda_multiplier : ℕ) (weeks_per_month : ℕ) : ℕ :=
  (jake_weekly * miranda_multiplier - jake_weekly) * weeks_per_month

/-- Proves that the difference in monthly newspaper deliveries between Miranda and Jake is 936 -/
theorem newspaper_delivery_difference :
  monthly_delivery_difference 234 2 4 = 936 := by
  sorry

end newspaper_delivery_difference_l3375_337587


namespace pat_height_l3375_337503

/-- Represents the depth dug on each day in centimeters -/
def depth_day1 : ℝ := 40

/-- Represents the total depth after day 2 in centimeters -/
def depth_day2 : ℝ := 3 * depth_day1

/-- Represents the additional depth dug on day 3 in centimeters -/
def depth_day3 : ℝ := depth_day2 - depth_day1

/-- Represents the distance from the ground surface to Pat's head at the end in centimeters -/
def surface_to_head : ℝ := 50

/-- Theorem stating Pat's height in centimeters -/
theorem pat_height : 
  depth_day2 + depth_day3 - surface_to_head = 150 := by sorry

end pat_height_l3375_337503


namespace truncated_pyramid_volume_division_l3375_337585

/-- Represents a truncated triangular pyramid -/
structure TruncatedPyramid where
  upperBaseArea : ℝ
  lowerBaseArea : ℝ
  height : ℝ
  baseRatio : upperBaseArea / lowerBaseArea = 1 / 4

/-- Represents the volumes of the two parts created by the plane -/
structure DividedVolumes where
  v1 : ℝ
  v2 : ℝ

/-- 
  Given a truncated triangular pyramid where the corresponding sides of the upper and lower 
  bases are in the ratio 1:2, if a plane is drawn through a side of the upper base parallel 
  to the opposite lateral edge, it divides the volume of the truncated pyramid in the ratio 3:4.
-/
theorem truncated_pyramid_volume_division (p : TruncatedPyramid) : 
  ∃ (v : DividedVolumes), v.v1 / v.v2 = 3 / 4 := by
  sorry

end truncated_pyramid_volume_division_l3375_337585


namespace sqrt_simplification_l3375_337551

theorem sqrt_simplification :
  (Real.sqrt 24 - Real.sqrt 2) - (Real.sqrt 8 + Real.sqrt 6) = Real.sqrt 6 - 3 * Real.sqrt 2 := by
  sorry

end sqrt_simplification_l3375_337551


namespace factorization_proof_l3375_337566

theorem factorization_proof (x : ℝ) : 
  (2 * x^3 - 8 * x^2 = 2 * x^2 * (x - 4)) ∧ 
  (x^2 - 14 * x + 49 = (x - 7)^2) := by
  sorry

end factorization_proof_l3375_337566


namespace m_four_sufficient_not_necessary_l3375_337572

/-- Represents a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are perpendicular -/
def are_perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- Define the two lines parameterized by m -/
def line1 (m : ℝ) : Line := ⟨2*m - 4, m + 1, 2⟩
def line2 (m : ℝ) : Line := ⟨m + 1, -m, 3⟩

/-- Main theorem -/
theorem m_four_sufficient_not_necessary :
  (∃ m : ℝ, m ≠ 4 ∧ are_perpendicular (line1 m) (line2 m)) ∧
  are_perpendicular (line1 4) (line2 4) := by
  sorry

end m_four_sufficient_not_necessary_l3375_337572


namespace jerry_cans_time_l3375_337538

def throw_away_cans (total_cans : ℕ) (cans_per_trip : ℕ) (drain_time : ℕ) (walk_time : ℕ) : ℕ :=
  let trips := (total_cans + cans_per_trip - 1) / cans_per_trip
  let drain_total := trips * drain_time
  let walk_total := trips * (2 * walk_time)
  drain_total + walk_total

theorem jerry_cans_time :
  throw_away_cans 35 3 30 10 = 600 := by
  sorry

end jerry_cans_time_l3375_337538


namespace seven_patients_three_doctors_l3375_337580

/-- The number of ways to assign n distinct objects to k distinct categories,
    where each object is assigned to exactly one category and
    each category receives at least one object. -/
def assignments (n k : ℕ) : ℕ :=
  k^n - (k * (k-1)^n - k * (k-1) * (k-2)^n)

/-- There are 7 patients and 3 doctors -/
theorem seven_patients_three_doctors :
  assignments 7 3 = 1806 := by
  sorry

end seven_patients_three_doctors_l3375_337580


namespace camel_height_28_feet_l3375_337539

/-- The height of a camel in feet, given the height of a hare in inches and their relative heights -/
def camel_height_in_feet (hare_height_inches : ℕ) (camel_hare_ratio : ℕ) : ℚ :=
  (hare_height_inches * camel_hare_ratio : ℚ) / 12

/-- Theorem stating the height of a camel in feet given specific measurements -/
theorem camel_height_28_feet :
  camel_height_in_feet 14 24 = 28 := by
  sorry

end camel_height_28_feet_l3375_337539


namespace exists_non_grid_aligned_right_triangle_l3375_337514

/-- A triangle represented by its three vertices -/
structure Triangle where
  a : ℤ × ℤ
  b : ℤ × ℤ
  c : ℤ × ℤ

/-- Check if a triangle is right-angled -/
def is_right_angled (t : Triangle) : Prop :=
  let ab := (t.b.1 - t.a.1, t.b.2 - t.a.2)
  let ac := (t.c.1 - t.a.1, t.c.2 - t.a.2)
  ab.1 * ac.1 + ab.2 * ac.2 = 0

/-- Check if a line segment is aligned with the grid -/
def is_grid_aligned (p1 p2 : ℤ × ℤ) : Prop :=
  p1.1 = p2.1 ∨ p1.2 = p2.2 ∨ (p2.2 - p1.2) * (p2.1 - p1.1) = 0

/-- The main theorem -/
theorem exists_non_grid_aligned_right_triangle :
  ∃ (t : Triangle),
    is_right_angled t ∧
    ¬is_grid_aligned t.a t.b ∧
    ¬is_grid_aligned t.b t.c ∧
    ¬is_grid_aligned t.c t.a :=
  sorry

end exists_non_grid_aligned_right_triangle_l3375_337514


namespace quadratic_trinomial_not_factor_l3375_337570

theorem quadratic_trinomial_not_factor (r : ℕ) (p : Polynomial ℤ) :
  (∀ i, |p.coeff i| < r) →
  p ≠ 0 →
  ¬ (X^2 - r • X - 1 : Polynomial ℤ) ∣ p :=
by sorry

end quadratic_trinomial_not_factor_l3375_337570


namespace nuts_ratio_l3375_337549

/-- Given the following conditions:
  - Sue has 48 nuts
  - Bill has 6 times as many nuts as Harry
  - Bill and Harry have combined 672 nuts
  Prove that the ratio of Harry's nuts to Sue's nuts is 2:1 -/
theorem nuts_ratio (sue_nuts : ℕ) (bill_harry_total : ℕ) :
  sue_nuts = 48 →
  bill_harry_total = 672 →
  ∃ (harry_nuts : ℕ),
    harry_nuts + 6 * harry_nuts = bill_harry_total ∧
    harry_nuts / sue_nuts = 2 :=
by sorry

end nuts_ratio_l3375_337549


namespace solve_system_l3375_337567

theorem solve_system (p q : ℚ) 
  (eq1 : 5 * p + 7 * q = 20) 
  (eq2 : 7 * p + 5 * q = 26) : 
  q = 5 / 12 := by
  sorry

end solve_system_l3375_337567


namespace morning_afternoon_email_difference_l3375_337556

/-- The number of emails Jack received in the afternoon -/
def afternoon_emails : ℕ := 3

/-- The number of emails Jack received in the morning -/
def morning_emails : ℕ := 5

/-- The number of emails Jack received in the evening -/
def evening_emails : ℕ := 16

/-- The theorem states that Jack received 2 more emails in the morning than in the afternoon -/
theorem morning_afternoon_email_difference : morning_emails - afternoon_emails = 2 := by
  sorry

end morning_afternoon_email_difference_l3375_337556


namespace expected_worth_is_one_third_l3375_337504

/-- The probability of getting heads on a coin flip -/
def prob_heads : ℚ := 2/3

/-- The probability of getting tails on a coin flip -/
def prob_tails : ℚ := 1/3

/-- The amount gained on a heads flip -/
def gain_heads : ℚ := 5

/-- The amount lost on a tails flip -/
def loss_tails : ℚ := 9

/-- The expected worth of a coin flip -/
def expected_worth : ℚ := prob_heads * gain_heads - prob_tails * loss_tails

theorem expected_worth_is_one_third : expected_worth = 1/3 := by
  sorry

end expected_worth_is_one_third_l3375_337504


namespace exponent_equality_l3375_337512

theorem exponent_equality 
  (a b c d : ℝ) 
  (x y q z : ℝ) 
  (h1 : a^(2*x) = c^(3*q)) 
  (h2 : a^(2*x) = b) 
  (h3 : c^(2*y) = a^(3*z)) 
  (h4 : c^(2*y) = d) 
  (h5 : a ≠ 0) 
  (h6 : c ≠ 0) : 
  2*x * 3*z = 3*q * 2*y := by
sorry

end exponent_equality_l3375_337512


namespace encrypted_text_is_cipher_of_problem_statement_l3375_337558

/-- Represents a character in the Russian alphabet -/
inductive RussianChar : Type
| vowel : RussianChar
| consonant : RussianChar

/-- Represents a string of Russian characters -/
def RussianString := List RussianChar

/-- The tarabar cipher function -/
def tarabarCipher : RussianString → RussianString := sorry

/-- The first sentence of the problem statement -/
def problemStatement : RussianString := sorry

/-- The given encrypted text -/
def encryptedText : RussianString := sorry

/-- Theorem stating that the encrypted text is a cipher of the problem statement -/
theorem encrypted_text_is_cipher_of_problem_statement :
  tarabarCipher problemStatement = encryptedText := by sorry

end encrypted_text_is_cipher_of_problem_statement_l3375_337558


namespace w_magnitude_bounds_l3375_337529

theorem w_magnitude_bounds (z : ℂ) (h : Complex.abs z = 1) : 
  let w : ℂ := z^4 - z^3 - 3 * z^2 * Complex.I - z + 1
  3 ≤ Complex.abs w ∧ Complex.abs w ≤ 5 := by
sorry

end w_magnitude_bounds_l3375_337529


namespace no_positive_sequence_with_sum_property_l3375_337526

open Real
open Set
open Nat

theorem no_positive_sequence_with_sum_property :
  ¬ (∃ b : ℕ → ℝ, 
    (∀ i : ℕ, i > 0 → b i > 0) ∧ 
    (∀ m : ℕ, m > 0 → (∑' k : ℕ, b (m * k)) = 1 / m)) := by
  sorry

end no_positive_sequence_with_sum_property_l3375_337526


namespace correct_chest_contents_l3375_337575

-- Define the types of coins
inductive CoinType
| Gold
| Silver
| Copper

-- Define the chests
structure Chest where
  label : CoinType
  content : CoinType

-- Define the problem setup
def setup : List Chest := [
  { label := CoinType.Gold, content := CoinType.Silver },
  { label := CoinType.Silver, content := CoinType.Gold },
  { label := CoinType.Gold, content := CoinType.Copper }
]

-- Theorem statement
theorem correct_chest_contents :
  ∀ (chests : List Chest),
  (chests.length = 3) →
  (∃! c, c ∈ chests ∧ c.content = CoinType.Gold) →
  (∃! c, c ∈ chests ∧ c.content = CoinType.Silver) →
  (∃! c, c ∈ chests ∧ c.content = CoinType.Copper) →
  (∀ c ∈ chests, c.label ≠ c.content) →
  (chests = setup) :=
by sorry

end correct_chest_contents_l3375_337575


namespace inverse_sum_reciprocal_l3375_337564

theorem inverse_sum_reciprocal (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (x⁻¹ + y⁻¹ + z⁻¹)⁻¹ = (x * y * z) / (x * z + y * z + x * y) := by
  sorry

end inverse_sum_reciprocal_l3375_337564


namespace timothy_initial_amount_matches_purchases_l3375_337547

/-- The amount of money Timothy had initially -/
def initial_amount : ℕ := 50

/-- The cost of a single t-shirt -/
def tshirt_cost : ℕ := 8

/-- The cost of a single bag -/
def bag_cost : ℕ := 10

/-- The number of t-shirts Timothy bought -/
def tshirts_bought : ℕ := 2

/-- The number of bags Timothy bought -/
def bags_bought : ℕ := 2

/-- The cost of a set of 3 key chains -/
def keychain_set_cost : ℕ := 2

/-- The number of key chains in a set -/
def keychains_per_set : ℕ := 3

/-- The number of key chains Timothy bought -/
def keychains_bought : ℕ := 21

/-- Theorem stating that Timothy's initial amount matches his purchases -/
theorem timothy_initial_amount_matches_purchases :
  initial_amount = 
    tshirts_bought * tshirt_cost + 
    bags_bought * bag_cost + 
    (keychains_bought / keychains_per_set) * keychain_set_cost :=
by
  sorry


end timothy_initial_amount_matches_purchases_l3375_337547


namespace sphere_box_height_l3375_337525

/-- A rectangular box with a large sphere and eight smaller spheres -/
structure SphereBox where
  length : ℝ
  width : ℝ
  height : ℝ
  large_sphere_radius : ℝ
  small_sphere_radius : ℝ
  small_sphere_count : ℕ

/-- Conditions for the sphere arrangement in the box -/
def valid_sphere_arrangement (box : SphereBox) : Prop :=
  box.length = 6 ∧
  box.width = 6 ∧
  box.large_sphere_radius = 3 ∧
  box.small_sphere_radius = 1 ∧
  box.small_sphere_count = 8 ∧
  ∀ (small_sphere : Fin box.small_sphere_count),
    (∃ (side1 side2 side3 : ℝ), side1 + side2 + side3 = box.length + box.width + box.height) ∧
    (box.large_sphere_radius + box.small_sphere_radius = 
     (box.length / 2)^2 + (box.width / 2)^2 + (box.height / 2 - box.small_sphere_radius)^2)

/-- Theorem stating that the height of the box is 8 -/
theorem sphere_box_height (box : SphereBox) 
  (h : valid_sphere_arrangement box) : box.height = 8 := by
  sorry

end sphere_box_height_l3375_337525


namespace price_change_l3375_337588

theorem price_change (x : ℝ) (h : x > 0) : x * (1 - 0.2) * (1 + 0.2) < x := by
  sorry

end price_change_l3375_337588


namespace james_singing_lesson_payment_l3375_337518

/-- Calculates the amount James pays for singing lessons given the specified conditions. -/
def jamesSingingLessonCost (totalLessons : ℕ) (lessonCost : ℕ) (freeLesson : ℕ) (fullPaidLessons : ℕ) : ℕ := 
  let paidLessonsAfterFull := (totalLessons - freeLesson - fullPaidLessons) / 2
  let totalCost := (fullPaidLessons + paidLessonsAfterFull) * lessonCost
  totalCost / 2

/-- Theorem stating that James pays $35 for his singing lessons under the given conditions. -/
theorem james_singing_lesson_payment :
  jamesSingingLessonCost 20 5 1 10 = 35 := by
  sorry

#eval jamesSingingLessonCost 20 5 1 10

end james_singing_lesson_payment_l3375_337518


namespace multiple_of_seven_l3375_337598

theorem multiple_of_seven : (2222^5555 + 5555^2222) % 7 = 0 := by
  sorry

end multiple_of_seven_l3375_337598


namespace men_to_women_percentage_l3375_337562

/-- If the population of women is 50% of the population of men,
    then the population of men is 200% of the population of women. -/
theorem men_to_women_percentage (men women : ℝ) (h : women = 0.5 * men) :
  men / women * 100 = 200 := by
  sorry

end men_to_women_percentage_l3375_337562


namespace congruence_solutions_count_l3375_337510

theorem congruence_solutions_count : ∃ (S : Finset ℕ), 
  (∀ x ∈ S, x < 100 ∧ x > 0 ∧ (x + 13) % 34 = 55 % 34) ∧ 
  (∀ x < 100, x > 0 → (x + 13) % 34 = 55 % 34 → x ∈ S) ∧
  Finset.card S = 3 := by
  sorry

end congruence_solutions_count_l3375_337510


namespace sqrt_expression_equality_l3375_337502

theorem sqrt_expression_equality : 
  2 * Real.sqrt 12 * (3 * Real.sqrt 48 - 4 * Real.sqrt (1/8) - 3 * Real.sqrt 27) = 36 - 4 * Real.sqrt 6 := by
  sorry

end sqrt_expression_equality_l3375_337502


namespace aaron_position_2023_l3375_337545

/-- Represents a point on the 2D plane -/
structure Point where
  x : Int
  y : Int

/-- Represents a direction -/
inductive Direction
  | East
  | North
  | West
  | South

/-- Aaron's movement rules -/
def nextDirection (d : Direction) : Direction :=
  match d with
  | Direction.East => Direction.North
  | Direction.North => Direction.West
  | Direction.West => Direction.South
  | Direction.South => Direction.East

/-- Move one step in the given direction -/
def move (p : Point) (d : Direction) : Point :=
  match d with
  | Direction.East => { x := p.x + 1, y := p.y }
  | Direction.North => { x := p.x, y := p.y + 1 }
  | Direction.West => { x := p.x - 1, y := p.y }
  | Direction.South => { x := p.x, y := p.y - 1 }

/-- Aaron's position after n steps -/
def aaronPosition (n : Nat) : Point :=
  sorry  -- The actual implementation would go here

theorem aaron_position_2023 :
  aaronPosition 2023 = { x := 21, y := -22 } := by
  sorry


end aaron_position_2023_l3375_337545


namespace second_to_tallest_ratio_l3375_337553

/-- The heights of four buildings satisfying certain conditions -/
structure BuildingHeights where
  t : ℝ  -- height of the tallest building
  s : ℝ  -- height of the second tallest building
  u : ℝ  -- height of the third tallest building
  v : ℝ  -- height of the fourth tallest building
  h1 : t = 100  -- the tallest building is 100 feet tall
  h2 : u = s / 2  -- the third tallest is half as tall as the second
  h3 : v = u / 5  -- the fourth is one-fifth as tall as the third
  h4 : t + s + u + v = 180  -- all 4 buildings together are 180 feet tall

/-- The ratio of the second tallest building to the tallest is 1:2 -/
theorem second_to_tallest_ratio (b : BuildingHeights) : b.s / b.t = 1 / 2 := by
  sorry

end second_to_tallest_ratio_l3375_337553


namespace probability_product_greater_than_five_l3375_337533

def S : Finset ℕ := {1, 2, 3, 4, 5}

def pairs : Finset (ℕ × ℕ) := S.product S |>.filter (λ (a, b) => a < b)

def valid_pairs : Finset (ℕ × ℕ) := pairs.filter (λ (a, b) => a * b > 5)

theorem probability_product_greater_than_five :
  (valid_pairs.card : ℚ) / pairs.card = 3 / 5 := by
  sorry

end probability_product_greater_than_five_l3375_337533


namespace midpoint_of_intersection_l3375_337560

-- Define the line
def line (x y : ℝ) : Prop := x - y = 2

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the intersection points
def intersection_points (A B : ℝ × ℝ) : Prop :=
  line A.1 A.2 ∧ parabola A.1 A.2 ∧
  line B.1 B.2 ∧ parabola B.1 B.2 ∧
  A ≠ B

-- Theorem statement
theorem midpoint_of_intersection :
  ∀ A B : ℝ × ℝ, intersection_points A B →
  (A.1 + B.1) / 2 = 4 ∧ (A.2 + B.2) / 2 = 2 := by sorry

end midpoint_of_intersection_l3375_337560


namespace adjacent_probability_l3375_337565

/-- The number of seats in the arrangement -/
def total_seats : ℕ := 9

/-- The number of students to be seated -/
def num_students : ℕ := 8

/-- The number of rows in the seating arrangement -/
def num_rows : ℕ := 3

/-- The number of columns in the seating arrangement -/
def num_columns : ℕ := 3

/-- Calculate the total number of possible seating arrangements -/
def total_arrangements : ℕ := Nat.factorial total_seats

/-- Calculate the number of favorable arrangements where Abby and Bridget are adjacent -/
def favorable_arrangements : ℕ :=
  (num_rows * (num_columns - 1) + num_columns * (num_rows - 1)) * 2 * Nat.factorial (num_students - 1)

/-- The probability that Abby and Bridget are adjacent in the same row or column -/
theorem adjacent_probability :
  (favorable_arrangements : ℚ) / total_arrangements = 5 / 12 := by
  sorry

end adjacent_probability_l3375_337565


namespace triangle_perimeter_l3375_337559

/-- A triangle with two sides of length 2 and 4, and the third side being a solution of x^2 - 6x + 8 = 0 has a perimeter of 10 -/
theorem triangle_perimeter : ∀ a b c : ℝ,
  a = 2 →
  b = 4 →
  c^2 - 6*c + 8 = 0 →
  c > 0 →
  a + b > c →
  b + c > a →
  c + a > b →
  a + b + c = 10 := by
  sorry

end triangle_perimeter_l3375_337559


namespace not_perfect_square_l3375_337552

theorem not_perfect_square (k : ℕ+) : ¬ ∃ (n : ℕ), (16 * k + 8 : ℕ) = n ^ 2 := by
  sorry

end not_perfect_square_l3375_337552


namespace jellybean_count_l3375_337569

/-- The number of jellybeans in a dozen -/
def dozen : ℕ := 12

/-- The number of jellybeans Caleb has -/
def caleb_jellybeans : ℕ := 3 * dozen

/-- The number of jellybeans Sophie has -/
def sophie_jellybeans : ℕ := caleb_jellybeans / 2

/-- The number of jellybeans Max has -/
def max_jellybeans : ℕ := sophie_jellybeans + 2 * dozen

/-- The total number of jellybeans -/
def total_jellybeans : ℕ := caleb_jellybeans + sophie_jellybeans + max_jellybeans

theorem jellybean_count : total_jellybeans = 96 := by sorry

end jellybean_count_l3375_337569


namespace second_polygon_sides_l3375_337540

/-- Given two regular polygons with the same perimeter, where one polygon has 50 sides
    and each of its sides is three times as long as each side of the other polygon,
    prove that the number of sides of the second polygon is 150. -/
theorem second_polygon_sides (s : ℝ) (n : ℕ) : s > 0 →
  50 * (3 * s) = n * s → n = 150 := by
  sorry

end second_polygon_sides_l3375_337540


namespace max_sum_is_fifty_l3375_337522

/-- A hexagonal prism with an added pyramid -/
structure HexagonalPrismWithPyramid where
  /-- Number of faces when pyramid is added to hexagonal face -/
  faces_hex : ℕ
  /-- Number of vertices when pyramid is added to hexagonal face -/
  vertices_hex : ℕ
  /-- Number of edges when pyramid is added to hexagonal face -/
  edges_hex : ℕ
  /-- Number of faces when pyramid is added to rectangular face -/
  faces_rect : ℕ
  /-- Number of vertices when pyramid is added to rectangular face -/
  vertices_rect : ℕ
  /-- Number of edges when pyramid is added to rectangular face -/
  edges_rect : ℕ

/-- The maximum sum of exterior faces, vertices, and edges -/
def max_sum (shape : HexagonalPrismWithPyramid) : ℕ :=
  max (shape.faces_hex + shape.vertices_hex + shape.edges_hex)
      (shape.faces_rect + shape.vertices_rect + shape.edges_rect)

/-- Theorem: The maximum sum of exterior faces, vertices, and edges is 50 -/
theorem max_sum_is_fifty (shape : HexagonalPrismWithPyramid) 
  (h1 : shape.faces_hex = 13)
  (h2 : shape.vertices_hex = 13)
  (h3 : shape.edges_hex = 24)
  (h4 : shape.faces_rect = 11)
  (h5 : shape.vertices_rect = 13)
  (h6 : shape.edges_rect = 22) :
  max_sum shape = 50 := by
  sorry


end max_sum_is_fifty_l3375_337522


namespace original_deck_size_l3375_337574

/-- Represents a deck of playing cards -/
structure Deck where
  total_cards : ℕ

/-- Represents the game setup -/
structure GameSetup where
  original_deck : Deck
  cards_kept_away : ℕ
  cards_in_play : ℕ

/-- The number of cards in a standard deck -/
def standard_deck_size : ℕ := 52

/-- Theorem: The original deck had 52 cards -/
theorem original_deck_size (setup : GameSetup) 
  (h1 : setup.cards_kept_away = 2) 
  (h2 : setup.cards_in_play + setup.cards_kept_away = setup.original_deck.total_cards) : 
  setup.original_deck.total_cards = standard_deck_size := by
  sorry

end original_deck_size_l3375_337574


namespace quadratic_function_theorem_l3375_337517

def QuadraticFunction (a h k : ℝ) : ℝ → ℝ := fun x ↦ a * (x - h)^2 + k

theorem quadratic_function_theorem (a : ℝ) (h k : ℝ) :
  (∀ x, QuadraticFunction a h k x ≤ 2) ∧
  QuadraticFunction a h k 2 = 1 ∧
  QuadraticFunction a h k 4 = 1 →
  h = 3 ∧ k = 2 := by
  sorry

end quadratic_function_theorem_l3375_337517


namespace division_problem_l3375_337509

theorem division_problem (A : ℕ) (h : 59 = 8 * A + 3) : A = 7 := by
  sorry

end division_problem_l3375_337509


namespace multiple_sum_properties_l3375_337557

theorem multiple_sum_properties (x y : ℤ) 
  (hx : ∃ (m : ℤ), x = 6 * m) 
  (hy : ∃ (n : ℤ), y = 12 * n) : 
  (∃ (k : ℤ), x + y = 2 * k) ∧ (∃ (l : ℤ), x + y = 6 * l) := by
  sorry

end multiple_sum_properties_l3375_337557


namespace girls_ran_nine_miles_l3375_337589

/-- The number of laps run by boys -/
def boys_laps : ℕ := 34

/-- The additional laps run by girls compared to boys -/
def additional_girls_laps : ℕ := 20

/-- The fraction of a mile that one lap represents -/
def lap_mile_fraction : ℚ := 1 / 6

/-- The total number of laps run by girls -/
def girls_laps : ℕ := boys_laps + additional_girls_laps

/-- The number of miles run by girls -/
def girls_miles : ℚ := girls_laps * lap_mile_fraction

theorem girls_ran_nine_miles : girls_miles = 9 := by
  sorry

end girls_ran_nine_miles_l3375_337589


namespace hamilton_marching_band_max_members_l3375_337568

theorem hamilton_marching_band_max_members :
  ∀ n : ℕ,
  (∃ k : ℕ, 30 * n = 34 * k + 2) →
  30 * n < 1500 →
  (∀ m : ℕ, (∃ j : ℕ, 30 * m = 34 * j + 2) → 30 * m < 1500 → 30 * m ≤ 30 * n) →
  30 * n = 1260 :=
by sorry

end hamilton_marching_band_max_members_l3375_337568


namespace a_share_l3375_337515

/-- Represents the share of money for each person -/
structure Share where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ

/-- The theorem stating A's share given the conditions -/
theorem a_share (s : Share) 
  (h1 : s.a = 5 * s.d ∧ s.b = 2 * s.d ∧ s.c = 4 * s.d) 
  (h2 : s.c = s.d + 500) : 
  s.a = 2500 := by
  sorry

end a_share_l3375_337515


namespace intersection_equals_B_l3375_337519

-- Define the sets A and B
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def B (p : ℝ) : Set ℝ := {x | p + 1 ≤ x ∧ x ≤ 2*p - 1}

-- State the theorem
theorem intersection_equals_B (p : ℝ) : A ∩ B p = B p ↔ p ≤ 3 := by sorry

end intersection_equals_B_l3375_337519


namespace min_value_squared_sum_l3375_337590

theorem min_value_squared_sum (x y : ℝ) : (x + y)^2 + (x - 1/y)^2 ≥ 2 := by
  sorry

end min_value_squared_sum_l3375_337590


namespace perfect_square_characterization_l3375_337582

theorem perfect_square_characterization (A : ℕ+) :
  (∃ k : ℕ, A = k^2) ↔
  (∀ n : ℕ+, ∃ k : ℕ+, k ≤ n ∧ n ∣ ((A + k)^2 - A)) := by
  sorry

end perfect_square_characterization_l3375_337582


namespace cone_surface_area_l3375_337543

/-- The surface area of a cone formed by rotating a right triangle -/
theorem cone_surface_area (r h l : ℝ) (triangle_condition : r^2 + h^2 = l^2) :
  r = 3 → h = 4 → l = 5 → (π * r * l + π * r^2) = 24 * π := by
  sorry

end cone_surface_area_l3375_337543


namespace f_symmetry_l3375_337520

/-- A cubic function with real coefficients -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x + 1

/-- Theorem: If f(-2) = 0, then f(2) = 2 -/
theorem f_symmetry (a b : ℝ) (h : f a b (-2) = 0) : f a b 2 = 2 := by
  sorry

end f_symmetry_l3375_337520


namespace local_min_implies_b_range_l3375_337542

theorem local_min_implies_b_range (b : ℝ) : 
  (∃ x ∈ Set.Ioo 0 1, IsLocalMin (fun x : ℝ ↦ x^3 - 3*b*x + 3*b) x) → 
  0 < b ∧ b < 1 :=
sorry

end local_min_implies_b_range_l3375_337542


namespace number_of_factors_60_l3375_337524

/-- The number of positive factors of 60 is 12 -/
theorem number_of_factors_60 : Finset.card (Nat.divisors 60) = 12 := by
  sorry

end number_of_factors_60_l3375_337524


namespace delta_phi_equals_negative_one_l3375_337578

def δ (x : ℚ) : ℚ := 5 * x + 6

def φ (x : ℚ) : ℚ := 6 * x + 5

theorem delta_phi_equals_negative_one (x : ℚ) : 
  δ (φ x) = -1 ↔ x = -16/15 := by sorry

end delta_phi_equals_negative_one_l3375_337578


namespace multiplication_of_powers_l3375_337586

theorem multiplication_of_powers (a : ℝ) : 2 * a^2 * a = 2 * a^3 := by
  sorry

end multiplication_of_powers_l3375_337586


namespace task_assignment_ways_l3375_337536

def number_of_students : ℕ := 30
def number_of_tasks : ℕ := 3

def permutations (n : ℕ) (r : ℕ) : ℕ := 
  Nat.factorial n / Nat.factorial (n - r)

theorem task_assignment_ways :
  permutations number_of_students number_of_tasks = 24360 := by
  sorry

end task_assignment_ways_l3375_337536


namespace coefficient_third_term_binomial_expansion_l3375_337573

theorem coefficient_third_term_binomial_expansion :
  let n : ℕ := 3
  let a : ℝ := 2
  let b : ℝ := 1
  let k : ℕ := 2
  (n.choose k) * a^(n - k) * b^k = 6 := by
sorry

end coefficient_third_term_binomial_expansion_l3375_337573


namespace david_pushups_count_l3375_337579

/-- The number of push-ups Zachary did -/
def zachary_pushups : ℕ := 7

/-- The additional number of push-ups David did compared to Zachary -/
def david_extra_pushups : ℕ := 30

/-- The number of push-ups David did -/
def david_pushups : ℕ := zachary_pushups + david_extra_pushups

theorem david_pushups_count : david_pushups = 37 := by sorry

end david_pushups_count_l3375_337579


namespace arithmetic_sequence_second_term_l3375_337594

/-- Given an arithmetic sequence where the sum of the first and third terms is 10,
    prove that the second term is 5. -/
theorem arithmetic_sequence_second_term 
  (a : ℝ) -- First term of the arithmetic sequence
  (d : ℝ) -- Common difference of the arithmetic sequence
  (h : a + (a + 2*d) = 10) -- Sum of first and third terms is 10
  : a + d = 5 := by
  sorry


end arithmetic_sequence_second_term_l3375_337594


namespace sequence_convergence_comparison_l3375_337596

/-- Given sequences (aₙ) and (bₙ) defined by the recurrence relations
    aₙ₊₁ = (aₙ + 1) / 2 and bₙ₊₁ = bₙᵏ, where 0 < k < 1/2 and
    a₀, b₀ ∈ (0, 1), there exists an N such that for all n ≥ N, aₙ < bₙ. -/
theorem sequence_convergence_comparison
  (k : ℝ) (h_k_pos : 0 < k) (h_k_bound : k < 1/2)
  (a₀ b₀ : ℝ) (h_a₀ : 0 < a₀ ∧ a₀ < 1) (h_b₀ : 0 < b₀ ∧ b₀ < 1)
  (a : ℕ → ℝ) (h_a : ∀ n, a (n + 1) = (a n + 1) / 2)
  (b : ℕ → ℝ) (h_b : ∀ n, b (n + 1) = (b n) ^ k)
  (h_a_init : a 0 = a₀) (h_b_init : b 0 = b₀) :
  ∃ N, ∀ n ≥ N, a n < b n :=
sorry

end sequence_convergence_comparison_l3375_337596


namespace inequality_problem_l3375_337508

theorem inequality_problem (x : ℝ) : 
  (x - 1) * |4 - x| < 12 ∧ x - 2 > 0 → 4 < x ∧ x < 8 := by
  sorry

end inequality_problem_l3375_337508


namespace integral_of_special_function_l3375_337535

theorem integral_of_special_function (f : ℝ → ℝ) 
  (h_diff : Differentiable ℝ f) 
  (h_def : ∀ x, f x = x^3 + x^2 * (deriv f 1)) : 
  ∫ x in (0:ℝ)..(2:ℝ), f x = -4 := by
  sorry

end integral_of_special_function_l3375_337535


namespace complement_intersection_theorem_l3375_337507

def U : Set ℕ := {x | x > 0 ∧ x^2 - 9*x + 8 ≤ 0}

def A : Set ℕ := {1, 2, 3}

def B : Set ℕ := {5, 6, 7}

theorem complement_intersection_theorem :
  (U \ A) ∩ (U \ B) = {4, 8} := by sorry

end complement_intersection_theorem_l3375_337507


namespace circle_radius_with_secant_l3375_337599

/-- Represents a circle with an external point and a secant --/
structure CircleWithSecant where
  -- Radius of the circle
  r : ℝ
  -- Distance from external point P to center
  distPC : ℝ
  -- Length of external segment PQ
  lenPQ : ℝ
  -- Length of segment QR
  lenQR : ℝ
  -- Condition: P is outside the circle
  h_outside : distPC > r
  -- Condition: PQ is external segment
  h_external : lenPQ < distPC

/-- The radius of the circle given the specified conditions --/
theorem circle_radius_with_secant (c : CircleWithSecant)
    (h_distPC : c.distPC = 17)
    (h_lenPQ : c.lenPQ = 12)
    (h_lenQR : c.lenQR = 8) :
    c.r = 7 := by
  sorry

end circle_radius_with_secant_l3375_337599


namespace expression_simplification_l3375_337592

theorem expression_simplification (x : ℝ) : 2*x - 3*(2 - x) + 4*(3 + x) - 5*(1 - 2*x) = 19*x + 1 := by
  sorry

end expression_simplification_l3375_337592


namespace sum_of_integers_l3375_337527

theorem sum_of_integers (a b : ℕ+) (h1 : a - b = 14) (h2 : a * b = 120) : a + b = 26 := by
  sorry

end sum_of_integers_l3375_337527


namespace billys_age_l3375_337501

/-- Given the ages of Billy, Joe, and Mary, prove that Billy is 45 years old. -/
theorem billys_age (B J M : ℕ) 
  (h1 : B = 3 * J)           -- Billy's age is three times Joe's age
  (h2 : B + J = 60)          -- The sum of Billy's and Joe's ages is 60
  (h3 : B + M = 90)          -- The sum of Billy's and Mary's ages is 90
  : B = 45 := by
  sorry


end billys_age_l3375_337501


namespace class_average_problem_l3375_337523

/-- Given a class where:
  - 20% of students average 80% on a test
  - 50% of students average X% on a test
  - 30% of students average 40% on a test
  - The overall class average is 58%
  Prove that X = 60 -/
theorem class_average_problem (X : ℝ) : 
  0.2 * 80 + 0.5 * X + 0.3 * 40 = 58 → X = 60 := by
  sorry

end class_average_problem_l3375_337523


namespace largest_d_for_g_range_contains_one_l3375_337563

/-- The quadratic function g(x) defined as 2x^2 - 8x + d -/
def g (d : ℝ) (x : ℝ) : ℝ := 2 * x^2 - 8 * x + d

/-- Theorem stating that the largest value of d such that 1 is in the range of g(x) is 9 -/
theorem largest_d_for_g_range_contains_one :
  (∃ (d : ℝ), ∀ (d' : ℝ), (∃ (x : ℝ), g d' x = 1) → d' ≤ d) ∧
  (∃ (x : ℝ), g 9 x = 1) :=
sorry

end largest_d_for_g_range_contains_one_l3375_337563


namespace h_max_at_72_l3375_337546

/-- The divisor function d(n) -/
def d (n : ℕ+) : ℕ := sorry

/-- The function h(n) = d(n)^2 / n^(1/4) -/
noncomputable def h (n : ℕ+) : ℝ := (d n)^2 / n.val^(1/4 : ℝ)

/-- The theorem stating that h(n) is maximized when n = 72 -/
theorem h_max_at_72 : ∀ n : ℕ+, n ≠ 72 → h n < h 72 := by sorry

end h_max_at_72_l3375_337546


namespace greatest_three_digit_multiple_of_17_l3375_337595

theorem greatest_three_digit_multiple_of_17 : 
  ∀ n : ℕ, n ≤ 999 → n ≥ 100 → n % 17 = 0 → n ≤ 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l3375_337595


namespace workshop_handshakes_l3375_337554

/-- Represents the workshop scenario -/
structure Workshop where
  total_people : Nat
  trainers : Nat
  participants : Nat
  knowledgeable_participants : Nat
  trainers_known_by_knowledgeable : Nat

/-- Calculate the number of handshakes in the workshop -/
def count_handshakes (w : Workshop) : Nat :=
  let unknown_participants := w.participants - w.knowledgeable_participants
  let handshakes_unknown := unknown_participants * (w.total_people - 1)
  let handshakes_knowledgeable := w.knowledgeable_participants * (w.total_people - w.trainers_known_by_knowledgeable - 1)
  handshakes_unknown + handshakes_knowledgeable

/-- The theorem to be proved -/
theorem workshop_handshakes :
  let w : Workshop := {
    total_people := 40,
    trainers := 25,
    participants := 15,
    knowledgeable_participants := 5,
    trainers_known_by_knowledgeable := 10
  }
  count_handshakes w = 540 := by sorry

end workshop_handshakes_l3375_337554


namespace line_equation_l3375_337576

/-- The circle with center (-1, 2) and radius √(5-a) --/
def Circle (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 + 1)^2 + (p.2 - 2)^2 = 5 - a}

/-- The line l --/
def Line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 - p.2 + 1 = 0}

/-- The midpoint of the chord AB --/
def M : ℝ × ℝ := (0, 1)

theorem line_equation (a : ℝ) (h : a < 3) :
  ∃ A B : ℝ × ℝ, A ∈ Circle a ∧ B ∈ Circle a ∧
  A ∈ Line ∧ B ∈ Line ∧
  M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  Line = {p : ℝ × ℝ | p.1 - p.2 + 1 = 0} := by
sorry

end line_equation_l3375_337576


namespace square_side_length_l3375_337584

theorem square_side_length : 
  ∃ (x : ℝ), x > 0 ∧ 4 * x = 2 * (x ^ 2) ∧ x = 2 := by
  sorry

end square_side_length_l3375_337584


namespace factor_value_theorem_l3375_337561

theorem factor_value_theorem (m n : ℚ) : 
  (∀ x : ℚ, (x - 3) * (x + 1) ∣ (3 * x^4 - m * x^2 + n * x - 5)) → 
  |3 * m - 2 * n| = 302 / 3 := by
  sorry

end factor_value_theorem_l3375_337561


namespace factorial16_trailingZeroes_base8_l3375_337530

/-- The number of trailing zeroes in the base 8 representation of 16! -/
def trailingZeroesBase8Factorial16 : ℕ := 5

/-- Theorem stating that the number of trailing zeroes in the base 8 representation of 16! is 5 -/
theorem factorial16_trailingZeroes_base8 :
  trailingZeroesBase8Factorial16 = 5 := by sorry

end factorial16_trailingZeroes_base8_l3375_337530


namespace age_puzzle_l3375_337591

theorem age_puzzle (A : ℕ) (N : ℚ) (h1 : A = 18) (h2 : N * (A + 3) - N * (A - 3) = A) : N = 3 := by
  sorry

end age_puzzle_l3375_337591


namespace children_playing_neither_sport_l3375_337500

theorem children_playing_neither_sport (total : ℕ) (tennis : ℕ) (squash : ℕ) (both : ℕ) 
  (h1 : total = 38)
  (h2 : tennis = 19)
  (h3 : squash = 21)
  (h4 : both = 12) :
  total - (tennis + squash - both) = 10 := by
  sorry

end children_playing_neither_sport_l3375_337500


namespace eighth_grade_girls_l3375_337528

/-- Given the number of boys and girls in eighth grade, proves the number of girls -/
theorem eighth_grade_girls (total : ℕ) (boys girls : ℕ) : 
  total = 68 → 
  boys = 2 * girls - 16 → 
  boys + girls = total → 
  girls = 28 := by
sorry

end eighth_grade_girls_l3375_337528


namespace investment_ratio_is_three_l3375_337548

/-- Represents the investment scenario of three partners A, B, and C --/
structure Investment where
  x : ℝ  -- A's initial investment
  m : ℝ  -- Ratio of C's investment to A's investment
  total_gain : ℝ  -- Total annual gain
  a_share : ℝ  -- A's share of the gain

/-- The ratio of C's investment to A's investment in the given scenario --/
def investment_ratio (inv : Investment) : ℝ :=
  let a_investment := inv.x * 12  -- A's investment for 12 months
  let b_investment := 2 * inv.x * 6  -- B's investment for 6 months
  let c_investment := inv.m * inv.x * 4  -- C's investment for 4 months
  let total_investment := a_investment + b_investment + c_investment
  inv.m

/-- Theorem stating that the investment ratio is 3 given the conditions --/
theorem investment_ratio_is_three (inv : Investment)
  (h1 : inv.total_gain = 15000)
  (h2 : inv.a_share = 5000)
  (h3 : inv.x > 0)
  : investment_ratio inv = 3 := by
  sorry

#check investment_ratio_is_three

end investment_ratio_is_three_l3375_337548


namespace remainder_a_sixth_mod_n_l3375_337583

theorem remainder_a_sixth_mod_n (n : ℕ+) (a : ℤ) (h : a^3 ≡ 1 [ZMOD n]) :
  a^6 ≡ 1 [ZMOD n] := by sorry

end remainder_a_sixth_mod_n_l3375_337583


namespace percentage_problem_l3375_337597

/-- Given a number N and a percentage P, this theorem proves that P is 20%
    when N is 580 and P% of N equals 30% of 120 plus 80. -/
theorem percentage_problem (N : ℝ) (P : ℝ) : 
  N = 580 → 
  (P / 100) * N = (30 / 100) * 120 + 80 → 
  P = 20 := by
sorry

end percentage_problem_l3375_337597


namespace modular_equivalence_problem_l3375_337521

theorem modular_equivalence_problem : ∃ n : ℤ, 0 ≤ n ∧ n < 23 ∧ -315 ≡ n [ZMOD 23] ∧ n = 7 := by
  sorry

end modular_equivalence_problem_l3375_337521


namespace hyperbolas_same_asymptotes_l3375_337581

/-- Given two hyperbolas with equations (x^2/16) - (y^2/25) = 1 and (y^2/49) - (x^2/M) = 1,
    if they have the same asymptotes, then M = 784/25 -/
theorem hyperbolas_same_asymptotes (M : ℝ) :
  (∀ x y : ℝ, x^2/16 - y^2/25 = 1 ↔ y^2/49 - x^2/M = 1) →
  (∀ x y : ℝ, y = (5/4) * x ↔ y = (7/Real.sqrt M) * x) →
  M = 784/25 := by
  sorry

end hyperbolas_same_asymptotes_l3375_337581


namespace coin_problem_l3375_337541

theorem coin_problem :
  ∀ (nickels dimes quarters : ℕ),
    nickels + dimes + quarters = 100 →
    5 * nickels + 10 * dimes + 25 * quarters = 835 →
    ∃ (min_dimes max_dimes : ℕ),
      (∀ d : ℕ, 
        (∃ n q : ℕ, n + d + q = 100 ∧ 5 * n + 10 * d + 25 * q = 835) →
        min_dimes ≤ d ∧ d ≤ max_dimes) ∧
      max_dimes - min_dimes = 64 := by
  sorry

end coin_problem_l3375_337541


namespace jen_work_hours_l3375_337506

/-- 
Given that:
- Jen works 7 hours a week more than Ben
- Jen's work in 4 weeks equals Ben's work in 6 weeks
Prove that Jen works 21 hours per week
-/
theorem jen_work_hours (ben_hours : ℕ) 
  (h1 : ben_hours + 7 = 4 * ben_hours + 28) :
  ben_hours + 7 = 21 := by
  sorry

end jen_work_hours_l3375_337506


namespace fuel_after_600km_distance_with_22L_left_l3375_337537

-- Define the relationship between distance and remaining fuel
def fuel_remaining (s : ℝ) : ℝ := 50 - 0.08 * s

-- Theorem 1: When distance is 600 km, remaining fuel is 2 L
theorem fuel_after_600km : fuel_remaining 600 = 2 := by sorry

-- Theorem 2: When remaining fuel is 22 L, distance traveled is 350 km
theorem distance_with_22L_left : ∃ s : ℝ, fuel_remaining s = 22 ∧ s = 350 := by sorry

end fuel_after_600km_distance_with_22L_left_l3375_337537


namespace quadratic_equation_result_l3375_337555

theorem quadratic_equation_result (m : ℝ) (h : 2 * m^2 + m = -1) : 4 * m^2 + 2 * m + 5 = 3 := by
  sorry

end quadratic_equation_result_l3375_337555


namespace max_rectangles_in_square_l3375_337571

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents a square grid -/
structure Grid where
  size : ℕ

/-- Defines a 4×1 rectangle -/
def fourByOne : Rectangle := { width := 4, height := 1 }

/-- Defines a 6×6 grid -/
def sixBySix : Grid := { size := 6 }

/-- 
  Theorem: The maximum number of 4×1 rectangles that can be placed 
  in a 6×6 square without crossing cell boundaries is 8.
-/
theorem max_rectangles_in_square : 
  ∃ (n : ℕ), n = 8 ∧ 
  (∀ (m : ℕ), m > n → 
    ¬ (∃ (arrangement : List (ℕ × ℕ)), 
      arrangement.length = m ∧
      (∀ (pos : ℕ × ℕ), pos ∈ arrangement → 
        pos.1 + fourByOne.width ≤ sixBySix.size ∧ 
        pos.2 + fourByOne.height ≤ sixBySix.size) ∧
      (∀ (pos1 pos2 : ℕ × ℕ), pos1 ∈ arrangement → pos2 ∈ arrangement → pos1 ≠ pos2 → 
        ¬ (pos1.1 < pos2.1 + fourByOne.width ∧ 
           pos2.1 < pos1.1 + fourByOne.width ∧ 
           pos1.2 < pos2.2 + fourByOne.height ∧ 
           pos2.2 < pos1.2 + fourByOne.height)))) :=
by
  sorry

end max_rectangles_in_square_l3375_337571

import Mathlib

namespace NUMINAMATH_CALUDE_pet_store_hamsters_l1528_152805

theorem pet_store_hamsters (rabbit_count : ℕ) (rabbit_ratio : ℕ) (hamster_ratio : ℕ) : 
  rabbit_count = 18 → 
  rabbit_ratio = 3 → 
  hamster_ratio = 4 → 
  (rabbit_count / rabbit_ratio) * hamster_ratio = 24 := by
sorry

end NUMINAMATH_CALUDE_pet_store_hamsters_l1528_152805


namespace NUMINAMATH_CALUDE_quadratic_root_product_l1528_152806

theorem quadratic_root_product (p q : ℝ) : 
  (∃ x : ℂ, x^2 + p*x + q = 0 ∧ x = 3 - 4*Complex.I) → p*q = -150 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_product_l1528_152806


namespace NUMINAMATH_CALUDE_mark_deck_project_cost_l1528_152812

/-- The total cost of Mark's deck project -/
def deck_project_cost (length width : ℝ) (cost_A cost_B cost_sealant : ℝ) 
  (percent_A : ℝ) (tax_rate : ℝ) : ℝ :=
let total_area := length * width
let area_A := percent_A * total_area
let area_B := (1 - percent_A) * total_area
let cost_materials := cost_A * area_A + cost_B * area_B
let cost_sealant_total := cost_sealant * total_area
let subtotal := cost_materials + cost_sealant_total
subtotal * (1 + tax_rate)

/-- Theorem stating the total cost of Mark's deck project -/
theorem mark_deck_project_cost :
  deck_project_cost 30 40 3 5 1 0.6 0.07 = 6163.20 := by
  sorry


end NUMINAMATH_CALUDE_mark_deck_project_cost_l1528_152812


namespace NUMINAMATH_CALUDE_marys_sheep_ratio_l1528_152823

theorem marys_sheep_ratio (initial : ℕ) (remaining : ℕ) : 
  initial = 400 → remaining = 150 → (initial - remaining * 2) / initial = 1 / 4 := by
  sorry

#check marys_sheep_ratio

end NUMINAMATH_CALUDE_marys_sheep_ratio_l1528_152823


namespace NUMINAMATH_CALUDE_gcd_of_1975_and_2625_l1528_152885

theorem gcd_of_1975_and_2625 : Nat.gcd 1975 2625 = 25 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_1975_and_2625_l1528_152885


namespace NUMINAMATH_CALUDE_g_formula_l1528_152813

noncomputable def g (a : ℝ) : ℝ :=
  let m := Real.exp (Real.log 2 * min a 2)
  let n := Real.exp (Real.log 2 * max (-2) a)
  n - m

theorem g_formula (a : ℝ) (ha : a ≥ 0) :
  g a = if a ≤ 2 then -3 else 1 - Real.exp (Real.log 2 * a) := by
  sorry

end NUMINAMATH_CALUDE_g_formula_l1528_152813


namespace NUMINAMATH_CALUDE_carla_daily_collection_l1528_152816

/-- The number of items Carla needs to collect each day -/
def daily_items (total_leaves total_bugs total_days : ℕ) : ℕ :=
  (total_leaves + total_bugs) / total_days

/-- Proof that Carla needs to collect 5 items per day -/
theorem carla_daily_collection :
  daily_items 30 20 10 = 5 :=
by sorry

end NUMINAMATH_CALUDE_carla_daily_collection_l1528_152816


namespace NUMINAMATH_CALUDE_product_remainder_zero_l1528_152834

theorem product_remainder_zero : 
  (1296 * 1444 * 1700 * 1875) % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_zero_l1528_152834


namespace NUMINAMATH_CALUDE_range_of_f_l1528_152854

def f (x : Int) : Int := (x - 1)^2 + 1

def domain : Set Int := {-1, 0, 1, 2, 3}

theorem range_of_f : 
  {y | ∃ x ∈ domain, f x = y} = {1, 2, 5} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l1528_152854


namespace NUMINAMATH_CALUDE_third_to_second_ratio_l1528_152891

/-- The heights of four buildings satisfy certain conditions -/
structure BuildingHeights where
  h1 : ℝ  -- Height of the tallest building
  h2 : ℝ  -- Height of the second tallest building
  h3 : ℝ  -- Height of the third tallest building
  h4 : ℝ  -- Height of the fourth tallest building
  tallest : h1 = 100
  second_tallest : h2 = h1 / 2
  fourth_tallest : h4 = h3 / 5
  total_height : h1 + h2 + h3 + h4 = 180

/-- The ratio of the third tallest to the second tallest building is 1:2 -/
theorem third_to_second_ratio (b : BuildingHeights) : b.h3 / b.h2 = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_third_to_second_ratio_l1528_152891


namespace NUMINAMATH_CALUDE_vegetarian_eaters_count_family_total_check_l1528_152864

/-- Represents the eating habits distribution in a family -/
structure FamilyEatingHabits where
  total : Nat
  onlyVegetarian : Nat
  onlyNonVegetarian : Nat
  both : Nat
  pescatarian : Nat
  vegan : Nat

/-- Calculates the number of people eating vegetarian food -/
def vegetarianEaters (habits : FamilyEatingHabits) : Nat :=
  habits.onlyVegetarian + habits.both + habits.vegan

/-- The given family's eating habits -/
def familyHabits : FamilyEatingHabits := {
  total := 40
  onlyVegetarian := 16
  onlyNonVegetarian := 12
  both := 8
  pescatarian := 3
  vegan := 1
}

/-- Theorem: The number of vegetarian eaters in the family is 25 -/
theorem vegetarian_eaters_count :
  vegetarianEaters familyHabits = 25 := by
  sorry

/-- Theorem: The sum of all eating habit categories equals the total family members -/
theorem family_total_check :
  familyHabits.onlyVegetarian + familyHabits.onlyNonVegetarian + familyHabits.both +
  familyHabits.pescatarian + familyHabits.vegan = familyHabits.total := by
  sorry

end NUMINAMATH_CALUDE_vegetarian_eaters_count_family_total_check_l1528_152864


namespace NUMINAMATH_CALUDE_visited_both_countries_l1528_152840

theorem visited_both_countries (total : ℕ) (iceland : ℕ) (norway : ℕ) (neither : ℕ) : 
  total = 100 → iceland = 55 → norway = 43 → neither = 63 → 
  (total - neither) = (iceland + norway - (iceland + norway - (total - neither))) := by
  sorry

end NUMINAMATH_CALUDE_visited_both_countries_l1528_152840


namespace NUMINAMATH_CALUDE_zoo_field_trip_l1528_152817

/-- Calculates the number of individuals left at the zoo after a field trip --/
theorem zoo_field_trip (initial_fifth_grade : ℕ) (merged_fifth_grade : ℕ) 
  (initial_chaperones : ℕ) (teachers : ℕ) (third_grade : ℕ) 
  (additional_chaperones : ℕ) (fifth_grade_left : ℕ) (third_grade_left : ℕ) 
  (chaperones_left : ℕ) : 
  initial_fifth_grade = 10 →
  merged_fifth_grade = 12 →
  initial_chaperones = 5 →
  teachers = 2 →
  third_grade = 15 →
  additional_chaperones = 3 →
  fifth_grade_left = 10 →
  third_grade_left = 6 →
  chaperones_left = 2 →
  initial_fifth_grade + merged_fifth_grade + initial_chaperones + teachers + 
    third_grade + additional_chaperones - 
    (fifth_grade_left + third_grade_left + chaperones_left) = 29 := by
  sorry


end NUMINAMATH_CALUDE_zoo_field_trip_l1528_152817


namespace NUMINAMATH_CALUDE_a_range_l1528_152826

theorem a_range (a : ℝ) : 
  (∀ x : ℝ, x ∈ {x | x^2 - 2*x + a > 0} ↔ x^2 - 2*x + a > 0) →
  1 ∉ {x : ℝ | x^2 - 2*x + a > 0} →
  a ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_a_range_l1528_152826


namespace NUMINAMATH_CALUDE_range_of_m_l1528_152877

/-- Given a quadratic inequality and a function with specific domain,
    prove that the range of m is [-1, 0] -/
theorem range_of_m (a : ℝ) (m : ℝ) : 
  (a > 0 ∧ a ≠ 1) →
  (∀ x : ℝ, a * x^2 - a * x - 2 * a^2 > 1 ↔ -a < x ∧ x < 2*a) →
  (∀ x : ℝ, (1/a)^(x^2 + 2*m*x - m) - 1 ≥ 0) →
  m ∈ Set.Icc (-1 : ℝ) 0 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l1528_152877


namespace NUMINAMATH_CALUDE_hall_length_is_18_l1528_152800

/-- Represents the dimensions of a rectangular hall -/
structure HallDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Checks if the hall dimensions satisfy the given conditions -/
def satisfiesConditions (d : HallDimensions) : Prop :=
  d.width = 9 ∧
  2 * (d.length * d.width) = 2 * (d.length * d.height + d.width * d.height) ∧
  d.length * d.width * d.height = 972

theorem hall_length_is_18 :
  ∃ (d : HallDimensions), satisfiesConditions d ∧ d.length = 18 :=
by sorry

end NUMINAMATH_CALUDE_hall_length_is_18_l1528_152800


namespace NUMINAMATH_CALUDE_negative_square_cubed_l1528_152851

theorem negative_square_cubed (a : ℝ) : (-a^2)^3 = -a^6 := by
  sorry

end NUMINAMATH_CALUDE_negative_square_cubed_l1528_152851


namespace NUMINAMATH_CALUDE_berry_problem_l1528_152828

theorem berry_problem (stacy steve skylar : ℕ) : 
  stacy = 3 * steve + 2 →  -- Stacy has 2 more than triple Steve's berries
  steve = skylar / 2 →     -- Steve has half of Skylar's berries
  stacy = 32 →             -- Stacy has 32 berries
  skylar = 20 :=           -- Prove Skylar has 20 berries
by
  sorry


end NUMINAMATH_CALUDE_berry_problem_l1528_152828


namespace NUMINAMATH_CALUDE_beckys_necklace_count_l1528_152870

/-- Calculates the final number of necklaces in Becky's collection -/
def final_necklace_count (initial : ℕ) (broken : ℕ) (new : ℕ) (gifted : ℕ) : ℕ :=
  initial - broken + new - gifted

/-- Theorem stating that Becky's final necklace count is 37 -/
theorem beckys_necklace_count :
  final_necklace_count 50 3 5 15 = 37 := by
  sorry

end NUMINAMATH_CALUDE_beckys_necklace_count_l1528_152870


namespace NUMINAMATH_CALUDE_all_shaded_areas_different_l1528_152841

/-- Represents a square with its division and shaded area -/
structure Square where
  total_divisions : ℕ
  shaded_divisions : ℕ

/-- The three squares in the problem -/
def square_I : Square := { total_divisions := 8, shaded_divisions := 3 }
def square_II : Square := { total_divisions := 9, shaded_divisions := 3 }
def square_III : Square := { total_divisions := 8, shaded_divisions := 4 }

/-- Calculate the shaded fraction of a square -/
def shaded_fraction (s : Square) : ℚ :=
  (s.shaded_divisions : ℚ) / (s.total_divisions : ℚ)

/-- Theorem stating that the shaded areas of all three squares are different -/
theorem all_shaded_areas_different :
  shaded_fraction square_I ≠ shaded_fraction square_II ∧
  shaded_fraction square_I ≠ shaded_fraction square_III ∧
  shaded_fraction square_II ≠ shaded_fraction square_III :=
sorry

end NUMINAMATH_CALUDE_all_shaded_areas_different_l1528_152841


namespace NUMINAMATH_CALUDE_total_eyes_l1528_152811

/-- The total number of eyes given the number of boys, girls, cats, and spiders -/
theorem total_eyes (boys girls cats spiders : ℕ) : 
  boys = 23 → 
  girls = 18 → 
  cats = 10 → 
  spiders = 5 → 
  boys * 2 + girls * 2 + cats * 2 + spiders * 8 = 142 := by
  sorry


end NUMINAMATH_CALUDE_total_eyes_l1528_152811


namespace NUMINAMATH_CALUDE_parallelogram_area_is_37_l1528_152831

/-- The area of a parallelogram formed by two 2D vectors -/
def parallelogramArea (v w : Fin 2 → ℤ) : ℕ :=
  (v 0 * w 1 - v 1 * w 0).natAbs

/-- Vectors v and w -/
def v : Fin 2 → ℤ := ![7, -5]
def w : Fin 2 → ℤ := ![13, -4]

/-- Theorem: The area of the parallelogram formed by v and w is 37 -/
theorem parallelogram_area_is_37 : parallelogramArea v w = 37 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_is_37_l1528_152831


namespace NUMINAMATH_CALUDE_yellow_ball_probability_l1528_152874

-- Define the number of balls of each color
def red_balls : ℕ := 2
def yellow_balls : ℕ := 5
def blue_balls : ℕ := 4

-- Define the total number of balls
def total_balls : ℕ := red_balls + yellow_balls + blue_balls

-- Define the probability of choosing a yellow ball
def prob_yellow : ℚ := yellow_balls / total_balls

-- Theorem statement
theorem yellow_ball_probability : prob_yellow = 5 / 11 := by
  sorry

end NUMINAMATH_CALUDE_yellow_ball_probability_l1528_152874


namespace NUMINAMATH_CALUDE_angles_with_same_terminal_side_l1528_152842

theorem angles_with_same_terminal_side (θ : Real) :
  θ = 150 * Real.pi / 180 →
  {β : Real | ∃ k : ℤ, β = 5 * Real.pi / 6 + 2 * k * Real.pi} =
  {β : Real | ∃ k : ℤ, β = θ + 2 * k * Real.pi} :=
by sorry

end NUMINAMATH_CALUDE_angles_with_same_terminal_side_l1528_152842


namespace NUMINAMATH_CALUDE_sum_of_xyz_is_twelve_l1528_152890

theorem sum_of_xyz_is_twelve (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (hxy : x * y = x + y) (hyz : y * z = 3 * (y + z)) (hzx : z * x = 2 * (z + x)) :
  x + y + z = 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_xyz_is_twelve_l1528_152890


namespace NUMINAMATH_CALUDE_simplify_expression_l1528_152881

theorem simplify_expression (y : ℝ) : (3*y)^3 - (4*y)*(y^2) = 23*y^3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1528_152881


namespace NUMINAMATH_CALUDE_sides_ratio_inscribed_circle_radius_l1528_152889

/-- A right-angled triangle with sides in arithmetic progression -/
structure ArithmeticRightTriangle where
  /-- The common difference of the arithmetic sequence -/
  d : ℝ
  /-- The common difference is positive -/
  d_pos : d > 0
  /-- The shortest side of the triangle -/
  shortest_side : ℝ
  /-- The shortest side is equal to 3d -/
  shortest_side_eq : shortest_side = 3 * d
  /-- The middle side of the triangle -/
  middle_side : ℝ
  /-- The middle side is equal to 4d -/
  middle_side_eq : middle_side = 4 * d
  /-- The longest side of the triangle (hypotenuse) -/
  longest_side : ℝ
  /-- The longest side is equal to 5d -/
  longest_side_eq : longest_side = 5 * d
  /-- The triangle satisfies the Pythagorean theorem -/
  pythagorean : shortest_side^2 + middle_side^2 = longest_side^2

/-- The ratio of sides in an ArithmeticRightTriangle is 3:4:5 -/
theorem sides_ratio (t : ArithmeticRightTriangle) :
  t.shortest_side / t.d = 3 ∧ t.middle_side / t.d = 4 ∧ t.longest_side / t.d = 5 := by
  sorry

/-- The radius of the inscribed circle is equal to the common difference -/
theorem inscribed_circle_radius (t : ArithmeticRightTriangle) :
  let s := (t.shortest_side + t.middle_side + t.longest_side) / 2
  let area := Real.sqrt (s * (s - t.shortest_side) * (s - t.middle_side) * (s - t.longest_side))
  area / s = t.d := by
  sorry

end NUMINAMATH_CALUDE_sides_ratio_inscribed_circle_radius_l1528_152889


namespace NUMINAMATH_CALUDE_greatest_marble_difference_is_six_l1528_152843

/-- Represents a basket of marbles -/
structure Basket where
  color1 : String
  count1 : Nat
  color2 : String
  count2 : Nat

/-- Calculates the absolute difference between two natural numbers -/
def absDiff (a b : Nat) : Nat :=
  if a ≥ b then a - b else b - a

/-- Theorem: The greatest difference between marble counts in any basket is 6 -/
theorem greatest_marble_difference_is_six :
  let basketA : Basket := { color1 := "red", count1 := 4, color2 := "yellow", count2 := 2 }
  let basketB : Basket := { color1 := "green", count1 := 6, color2 := "yellow", count2 := 1 }
  let basketC : Basket := { color1 := "white", count1 := 3, color2 := "yellow", count2 := 9 }
  let diffA := absDiff basketA.count1 basketA.count2
  let diffB := absDiff basketB.count1 basketB.count2
  let diffC := absDiff basketC.count1 basketC.count2
  (max diffA (max diffB diffC)) = 6 := by
  sorry

end NUMINAMATH_CALUDE_greatest_marble_difference_is_six_l1528_152843


namespace NUMINAMATH_CALUDE_m_range_l1528_152860

theorem m_range (m : ℝ) 
  (h1 : |m + 1| ≤ 2)
  (h2 : ¬(¬p))
  (h3 : ¬(p ∧ q))
  (p : Prop)
  (q : Prop) :
  -2 < m ∧ m ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_m_range_l1528_152860


namespace NUMINAMATH_CALUDE_number_cubed_equals_two_to_ninth_l1528_152869

theorem number_cubed_equals_two_to_ninth (number : ℝ) : number ^ 3 = 2 ^ 9 → number = 8 := by
  sorry

end NUMINAMATH_CALUDE_number_cubed_equals_two_to_ninth_l1528_152869


namespace NUMINAMATH_CALUDE_base_conversion_2458_to_base_7_l1528_152846

theorem base_conversion_2458_to_base_7 :
  2458 = 1 * 7^4 + 0 * 7^3 + 1 * 7^2 + 1 * 7^1 + 1 * 7^0 :=
by sorry

end NUMINAMATH_CALUDE_base_conversion_2458_to_base_7_l1528_152846


namespace NUMINAMATH_CALUDE_exists_captivating_number_l1528_152819

/-- A function that checks if a list of digits forms a captivating number -/
def is_captivating (digits : List Nat) : Prop :=
  digits.length = 7 ∧
  digits.toFinset = Finset.range 7 ∧
  ∀ k : Nat, k ∈ Finset.range 7 → 
    (digits.take k).foldl (fun acc d => acc * 10 + d) 0 % (k + 1) = 0

/-- Theorem stating the existence of at least one captivating number -/
theorem exists_captivating_number : ∃ digits : List Nat, is_captivating digits :=
  sorry

end NUMINAMATH_CALUDE_exists_captivating_number_l1528_152819


namespace NUMINAMATH_CALUDE_xy_sum_reciprocals_l1528_152866

theorem xy_sum_reciprocals (x y : ℝ) (θ : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0)
  (h_theta : ∀ (n : ℤ), θ ≠ π / 2 * n)
  (h_eq1 : Real.sin θ / x = Real.cos θ / y)
  (h_eq2 : Real.cos θ ^ 4 / x ^ 4 + Real.sin θ ^ 4 / y ^ 4 = 
           97 * Real.sin (2 * θ) / (x ^ 3 * y + y ^ 3 * x)) :
  x / y + y / x = 4 := by
  sorry

end NUMINAMATH_CALUDE_xy_sum_reciprocals_l1528_152866


namespace NUMINAMATH_CALUDE_same_units_digit_count_l1528_152899

def old_page_numbers := Finset.range 60

theorem same_units_digit_count :
  (old_page_numbers.filter (λ x => x % 10 = (61 - x) % 10)).card = 6 := by
  sorry

end NUMINAMATH_CALUDE_same_units_digit_count_l1528_152899


namespace NUMINAMATH_CALUDE_speed_ratio_and_distance_l1528_152887

/-- Represents a traveler with a constant speed -/
structure Traveler where
  speed : ℝ
  startPosition : ℝ

/-- Represents the problem setup -/
structure TravelProblem where
  A : Traveler
  B : Traveler
  C : Traveler
  distanceAB : ℝ
  timeToCMeetA : ℝ
  timeAMeetsB : ℝ
  BPastMidpoint : ℝ
  CFromA : ℝ

/-- The main theorem that proves the speed ratio and distance -/
theorem speed_ratio_and_distance 
  (p : TravelProblem)
  (h1 : p.A.startPosition = 0)
  (h2 : p.B.startPosition = 0)
  (h3 : p.C.startPosition = p.distanceAB)
  (h4 : p.timeToCMeetA = 20)
  (h5 : p.timeAMeetsB = 10)
  (h6 : p.BPastMidpoint = 105)
  (h7 : p.CFromA = 315)
  : p.A.speed / p.B.speed = 3 ∧ p.distanceAB = 1890 := by
  sorry

#check speed_ratio_and_distance

end NUMINAMATH_CALUDE_speed_ratio_and_distance_l1528_152887


namespace NUMINAMATH_CALUDE_tan_product_thirty_degrees_l1528_152859

theorem tan_product_thirty_degrees :
  let A : Real := 30 * π / 180
  let B : Real := 30 * π / 180
  (1 + Real.tan A) * (1 + Real.tan B) = (4 + 2 * Real.sqrt 3) / 3 := by
sorry

end NUMINAMATH_CALUDE_tan_product_thirty_degrees_l1528_152859


namespace NUMINAMATH_CALUDE_area_at_stage_4_is_1360_l1528_152845

/-- The area of the figure at stage n, given an initial square of side length 4 inches -/
def area_at_stage (n : ℕ) : ℕ :=
  let initial_side := 4
  let rec sum_areas (k : ℕ) (acc : ℕ) : ℕ :=
    if k = 0 then acc
    else sum_areas (k - 1) (acc + (initial_side * 2^(k - 1))^2)
  sum_areas n 0

/-- The theorem stating that the area at stage 4 is 1360 square inches -/
theorem area_at_stage_4_is_1360 : area_at_stage 4 = 1360 := by
  sorry

end NUMINAMATH_CALUDE_area_at_stage_4_is_1360_l1528_152845


namespace NUMINAMATH_CALUDE_f_difference_l1528_152824

-- Define the function f
def f (x : ℝ) : ℝ := x^6 + 3*x^4 - 4*x^3 + x^2 + 2*x

-- State the theorem
theorem f_difference : f 3 - f (-3) = -204 := by
  sorry

end NUMINAMATH_CALUDE_f_difference_l1528_152824


namespace NUMINAMATH_CALUDE_blood_donor_selection_l1528_152825

theorem blood_donor_selection (type_O : Nat) (type_A : Nat) (type_B : Nat) (type_AB : Nat)
  (h1 : type_O = 10)
  (h2 : type_A = 5)
  (h3 : type_B = 8)
  (h4 : type_AB = 3) :
  type_O * type_A * type_B * type_AB = 1200 := by
  sorry

end NUMINAMATH_CALUDE_blood_donor_selection_l1528_152825


namespace NUMINAMATH_CALUDE_simple_interest_rate_l1528_152871

/-- Represents the rate of simple interest per annum -/
def rate : ℚ := 1 / 24

/-- The time period in years -/
def time : ℕ := 12

/-- The ratio of final amount to initial amount -/
def growth_ratio : ℚ := 9 / 6

theorem simple_interest_rate :
  (1 + rate * time) = growth_ratio := by sorry

end NUMINAMATH_CALUDE_simple_interest_rate_l1528_152871


namespace NUMINAMATH_CALUDE_log_equation_roots_range_l1528_152858

-- Define the logarithmic equation
def log_equation (x a : ℝ) : Prop :=
  Real.log (x - 1) + Real.log (3 - x) = Real.log (a - x)

-- Define the condition for two distinct real roots
def has_two_distinct_roots (a : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 1 < x₁ ∧ x₁ < 3 ∧ 1 < x₂ ∧ x₂ < 3 ∧
  log_equation x₁ a ∧ log_equation x₂ a

-- Theorem statement
theorem log_equation_roots_range :
  ∀ a : ℝ, has_two_distinct_roots a ↔ 3 < a ∧ a < 13/4 :=
by sorry

end NUMINAMATH_CALUDE_log_equation_roots_range_l1528_152858


namespace NUMINAMATH_CALUDE_sum_from_simple_interest_and_true_discount_l1528_152836

/-- Given a sum, time, and rate, if the simple interest is 85 and the true discount is 75, then the sum is 637.5 -/
theorem sum_from_simple_interest_and_true_discount
  (P T R : ℝ) -- Sum, Time, and Rate
  (h1 : (P * T * R) / 100 = 85) -- Simple interest equation
  (h2 : (85 * 100) / (100 + R * T) = 75) -- True discount equation
  : P = 637.5 := by
  sorry

end NUMINAMATH_CALUDE_sum_from_simple_interest_and_true_discount_l1528_152836


namespace NUMINAMATH_CALUDE_x_plus_y_value_l1528_152886

theorem x_plus_y_value (x y : Real) 
  (eq1 : x + Real.sin y = 2023)
  (eq2 : x + 2023 * Real.cos y = 2022)
  (y_range : π/2 ≤ y ∧ y ≤ π) :
  x + y = 2022 + π/2 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_value_l1528_152886


namespace NUMINAMATH_CALUDE_largest_sum_and_simplification_l1528_152893

theorem largest_sum_and_simplification : 
  let sums : List ℚ := [1/5 + 1/2, 1/5 + 1/6, 1/5 + 1/4, 1/5 + 1/8, 1/5 + 1/9]
  (∀ x ∈ sums, 1/5 + 1/2 ≥ x) ∧ (1/5 + 1/2 = 7/10) :=
by sorry

end NUMINAMATH_CALUDE_largest_sum_and_simplification_l1528_152893


namespace NUMINAMATH_CALUDE_smallest_yellow_marbles_l1528_152830

theorem smallest_yellow_marbles (n : ℕ) (h1 : n % 10 = 0) (h2 : n ≥ 30) : ∃ (blue red green yellow : ℕ),
  blue = n / 2 ∧
  red = n / 5 ∧
  green = 8 ∧
  yellow = n - (blue + red + green) ∧
  yellow ≥ 1 ∧
  ∀ m : ℕ, m < n → ¬(∃ (b r g y : ℕ),
    b = m / 2 ∧
    r = m / 5 ∧
    g = 8 ∧
    y = m - (b + r + g) ∧
    y ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_smallest_yellow_marbles_l1528_152830


namespace NUMINAMATH_CALUDE_amp_fifteen_amp_l1528_152872

-- Define the & operation
def amp (x : ℝ) : ℝ := 9 - x

-- Define the & prefix operation
def amp_prefix (x : ℝ) : ℝ := x - 9

-- Theorem statement
theorem amp_fifteen_amp : amp_prefix (amp 15) = -15 := by sorry

end NUMINAMATH_CALUDE_amp_fifteen_amp_l1528_152872


namespace NUMINAMATH_CALUDE_rook_placement_on_colored_board_l1528_152803

theorem rook_placement_on_colored_board :
  let n : ℕ := 8  -- number of rooks and rows/columns
  let m : ℕ := 32  -- number of colors
  let total_arrangements : ℕ := n.factorial
  let problematic_arrangements : ℕ := m * (n - 2).factorial
  total_arrangements > problematic_arrangements :=
by sorry

end NUMINAMATH_CALUDE_rook_placement_on_colored_board_l1528_152803


namespace NUMINAMATH_CALUDE_square_of_difference_l1528_152844

theorem square_of_difference (x : ℝ) : (8 - Real.sqrt (x^2 + 64))^2 = x^2 + 128 - 16 * Real.sqrt (x^2 + 64) := by
  sorry

end NUMINAMATH_CALUDE_square_of_difference_l1528_152844


namespace NUMINAMATH_CALUDE_car_down_payment_calculation_l1528_152848

/-- Calculates the down payment for a car purchase given the specified conditions. -/
theorem car_down_payment_calculation
  (car_cost : ℚ)
  (loan_term : ℕ)
  (monthly_payment : ℚ)
  (interest_rate : ℚ)
  (h_car_cost : car_cost = 32000)
  (h_loan_term : loan_term = 48)
  (h_monthly_payment : monthly_payment = 525)
  (h_interest_rate : interest_rate = 5 / 100)
  : ∃ (down_payment : ℚ),
    down_payment = car_cost - (loan_term * monthly_payment + loan_term * (interest_rate * monthly_payment)) ∧
    down_payment = 5540 :=
by sorry

end NUMINAMATH_CALUDE_car_down_payment_calculation_l1528_152848


namespace NUMINAMATH_CALUDE_red_marble_fraction_l1528_152818

theorem red_marble_fraction (total : ℝ) (h : total > 0) :
  let initial_blue := (2/3 : ℝ) * total
  let initial_red := total - initial_blue
  let new_red := 3 * initial_red
  let new_total := initial_blue + new_red
  new_red / new_total = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_red_marble_fraction_l1528_152818


namespace NUMINAMATH_CALUDE_dongwi_festival_cases_l1528_152832

/-- The number of cases in which Dongwi can go to play at the festival. -/
def num_cases (boys_schools : ℕ) (girls_schools : ℕ) : ℕ :=
  boys_schools + girls_schools

/-- Theorem stating that the number of cases for Dongwi to go to play is 7. -/
theorem dongwi_festival_cases :
  let boys_schools := 4
  let girls_schools := 3
  num_cases boys_schools girls_schools = 7 := by
  sorry

end NUMINAMATH_CALUDE_dongwi_festival_cases_l1528_152832


namespace NUMINAMATH_CALUDE_salary_increase_l1528_152802

/-- If a salary increases by 33.33% to $80, prove that the original salary was $60 -/
theorem salary_increase (original : ℝ) (increase_percent : ℝ) (new_salary : ℝ) :
  increase_percent = 33.33 ∧ new_salary = 80 ∧ new_salary = original * (1 + increase_percent / 100) →
  original = 60 := by
  sorry

end NUMINAMATH_CALUDE_salary_increase_l1528_152802


namespace NUMINAMATH_CALUDE_third_height_less_than_30_l1528_152839

/-- Given a triangle with two heights of 12 and 20, prove that the third height is less than 30. -/
theorem third_height_less_than_30 
  (h_a h_b h_c : ℝ) 
  (triangle_exists : ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b)
  (height_a : h_a = 12)
  (height_b : h_b = 20) :
  h_c < 30 := by
sorry

end NUMINAMATH_CALUDE_third_height_less_than_30_l1528_152839


namespace NUMINAMATH_CALUDE_range_of_m_l1528_152856

/-- The range of m that satisfies the given conditions -/
theorem range_of_m (m : ℝ) : m ≥ 9 ↔ 
  (∀ x : ℝ, (|1 - x| > 2 → (x^2 - 2*x + 1 - m^2 > 0))) ∧ 
  (∃ x : ℝ, |1 - x| > 2 ∧ x^2 - 2*x + 1 - m^2 ≤ 0) ∧
  m > 0 :=
by sorry


end NUMINAMATH_CALUDE_range_of_m_l1528_152856


namespace NUMINAMATH_CALUDE_angle_AOF_is_118_l1528_152863

/-- Given a configuration of angles where:
    ∠AOB = ∠BOC
    ∠COD = ∠DOE = ∠EOF
    ∠AOD = 82°
    ∠BOE = 68°
    Prove that ∠AOF = 118° -/
theorem angle_AOF_is_118 (AOB BOC COD DOE EOF AOD BOE : ℝ) : 
  AOB = BOC ∧ 
  COD = DOE ∧ DOE = EOF ∧
  AOD = 82 ∧
  BOE = 68 →
  AOB + BOC + COD + DOE + EOF = 118 := by
  sorry

end NUMINAMATH_CALUDE_angle_AOF_is_118_l1528_152863


namespace NUMINAMATH_CALUDE_arithmetic_sequence_100th_term_l1528_152867

/-- For an arithmetic sequence {a_n} with first term 1 and common difference 3,
    prove that the 100th term is 298. -/
theorem arithmetic_sequence_100th_term :
  ∀ (a : ℕ → ℕ),
  (∀ n, a (n + 1) = a n + 3) →  -- arithmetic sequence with common difference 3
  a 1 = 1 →                    -- first term is 1
  a 100 = 298 := by             -- 100th term is 298
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_100th_term_l1528_152867


namespace NUMINAMATH_CALUDE_exactly_two_solutions_l1528_152876

/-- The number of integer solutions to the given system of equations -/
def num_solutions : ℕ := 2

/-- The system of equations -/
def system (x y z : ℤ) : Prop :=
  x^2 - 4*x*y + 3*y^2 - z^2 = 40 ∧
  -x^2 + 4*y*z + 3*z^2 = 47 ∧
  x^2 + 2*x*y + 9*z^2 = 110

/-- Theorem stating that there are exactly 2 solutions to the system -/
theorem exactly_two_solutions :
  (∃! (solutions : Finset (ℤ × ℤ × ℤ)), solutions.card = num_solutions ∧
    ∀ (x y z : ℤ), (x, y, z) ∈ solutions ↔ system x y z) :=
sorry

end NUMINAMATH_CALUDE_exactly_two_solutions_l1528_152876


namespace NUMINAMATH_CALUDE_max_erasers_purchase_l1528_152847

def pen_cost : ℕ := 3
def pencil_cost : ℕ := 4
def eraser_cost : ℕ := 8
def total_budget : ℕ := 60

def is_valid_purchase (pens pencils erasers : ℕ) : Prop :=
  pens ≥ 1 ∧ pencils ≥ 1 ∧ erasers ≥ 1 ∧
  pens * pen_cost + pencils * pencil_cost + erasers * eraser_cost = total_budget

theorem max_erasers_purchase :
  ∃ (pens pencils : ℕ), is_valid_purchase pens pencils 5 ∧
  ∀ (p n e : ℕ), is_valid_purchase p n e → e ≤ 5 :=
sorry

end NUMINAMATH_CALUDE_max_erasers_purchase_l1528_152847


namespace NUMINAMATH_CALUDE_five_solutions_for_f_f_eq_seven_l1528_152897

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ -2 then x^2 - 6 else x + 5

theorem five_solutions_for_f_f_eq_seven :
  ∃! (s : Finset ℝ), s.card = 5 ∧ ∀ x : ℝ, x ∈ s ↔ f (f x) = 7 :=
sorry

end NUMINAMATH_CALUDE_five_solutions_for_f_f_eq_seven_l1528_152897


namespace NUMINAMATH_CALUDE_tangent_line_b_value_l1528_152809

/-- The curve function -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x + 1

/-- The derivative of the curve function -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + a

theorem tangent_line_b_value :
  ∀ a k b : ℝ,
  f a 2 = 3 →                        -- The curve passes through (2, 3)
  f' a 2 = k →                       -- The slope of the tangent line at x = 2
  3 = k * 2 + b →                    -- The tangent line passes through (2, 3)
  b = -15 := by sorry

end NUMINAMATH_CALUDE_tangent_line_b_value_l1528_152809


namespace NUMINAMATH_CALUDE_defective_engine_fraction_l1528_152873

theorem defective_engine_fraction :
  let total_batches : ℕ := 5
  let engines_per_batch : ℕ := 80
  let non_defective_engines : ℕ := 300
  let total_engines : ℕ := total_batches * engines_per_batch
  let defective_engines : ℕ := total_engines - non_defective_engines
  (defective_engines : ℚ) / total_engines = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_defective_engine_fraction_l1528_152873


namespace NUMINAMATH_CALUDE_bathroom_square_footage_l1528_152882

/-- Calculates the square footage of a bathroom given the number of tiles and tile size. -/
theorem bathroom_square_footage 
  (width_tiles : ℕ) 
  (length_tiles : ℕ) 
  (tile_size_inches : ℕ) 
  (h1 : width_tiles = 10) 
  (h2 : length_tiles = 20) 
  (h3 : tile_size_inches = 6) : 
  (width_tiles * length_tiles * tile_size_inches^2) / 144 = 50 := by
  sorry

#check bathroom_square_footage

end NUMINAMATH_CALUDE_bathroom_square_footage_l1528_152882


namespace NUMINAMATH_CALUDE_base_7_to_decimal_l1528_152880

def to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * base^i) 0

theorem base_7_to_decimal :
  to_decimal [6, 5, 7] 7 = 384 := by
  sorry

end NUMINAMATH_CALUDE_base_7_to_decimal_l1528_152880


namespace NUMINAMATH_CALUDE_line_equation_l1528_152833

/-- Circle P: x^2 + y^2 - 4y = 0 -/
def circleP (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*y = 0

/-- Parabola S: y = x^2 / 8 -/
def parabolaS (x y : ℝ) : Prop :=
  y = x^2 / 8

/-- Line l: y = k*x + b -/
def lineL (k b x y : ℝ) : Prop :=
  y = k*x + b

/-- Center of circle P -/
def centerP : ℝ × ℝ := (0, 2)

/-- Line l passes through the center of circle P -/
def lineThroughCenter (k b : ℝ) : Prop :=
  lineL k b (centerP.1) (centerP.2)

/-- Four intersection points of line l with circle P and parabola S -/
structure IntersectionPoints (k b : ℝ) :=
  (A B C D : ℝ × ℝ)
  (intersectCircleP : circleP A.1 A.2 ∧ circleP B.1 B.2 ∧ circleP C.1 C.2 ∧ circleP D.1 D.2)
  (intersectParabolaS : parabolaS A.1 A.2 ∧ parabolaS B.1 B.2 ∧ parabolaS C.1 C.2 ∧ parabolaS D.1 D.2)
  (onLineL : lineL k b A.1 A.2 ∧ lineL k b B.1 B.2 ∧ lineL k b C.1 C.2 ∧ lineL k b D.1 D.2)
  (leftToRight : A.1 < B.1 ∧ B.1 < C.1 ∧ C.1 < D.1)

/-- Lengths of segments AB, BC, CD form an arithmetic sequence -/
def arithmeticSequence (A B C D : ℝ × ℝ) : Prop :=
  let AB := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  let BC := Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2)
  let CD := Real.sqrt ((D.1 - C.1)^2 + (D.2 - C.2)^2)
  BC - AB = CD - BC

theorem line_equation :
  ∀ k b : ℝ,
  lineThroughCenter k b →
  (∃ pts : IntersectionPoints k b, arithmeticSequence pts.A pts.B pts.C pts.D) →
  (k = -Real.sqrt 2 / 2 ∨ k = Real.sqrt 2 / 2) ∧ b = 2 :=
sorry

end NUMINAMATH_CALUDE_line_equation_l1528_152833


namespace NUMINAMATH_CALUDE_cos_150_degrees_l1528_152808

theorem cos_150_degrees :
  Real.cos (150 * π / 180) = -Real.sqrt 3 / 2 :=
by
  -- Define the cosine subtraction identity
  have cos_subtraction_identity (a b : ℝ) :
    Real.cos (a - b) = Real.cos a * Real.cos b + Real.sin a * Real.sin b :=
    sorry

  -- Express 150° as 180° - 30°
  have h1 : 150 * π / 180 = π - (30 * π / 180) :=
    sorry

  -- Use the cosine subtraction identity
  have h2 : Real.cos (150 * π / 180) =
    Real.cos π * Real.cos (30 * π / 180) + Real.sin π * Real.sin (30 * π / 180) :=
    sorry

  -- Evaluate the expression
  sorry

end NUMINAMATH_CALUDE_cos_150_degrees_l1528_152808


namespace NUMINAMATH_CALUDE_correct_statements_count_l1528_152895

-- Define a structure for sampling statements
structure SamplingStatement :=
  (id : Nat)
  (content : String)
  (isCorrect : Bool)

-- Define the four statements
def statement1 : SamplingStatement :=
  { id := 1
  , content := "When the total number of individuals in a population is not large, it is appropriate to use simple random sampling"
  , isCorrect := true }

def statement2 : SamplingStatement :=
  { id := 2
  , content := "In systematic sampling, after the population is divided evenly, simple random sampling is used in each part"
  , isCorrect := false }

def statement3 : SamplingStatement :=
  { id := 3
  , content := "The lottery activities in department stores are a method of drawing lots"
  , isCorrect := true }

def statement4 : SamplingStatement :=
  { id := 4
  , content := "In systematic sampling, the probability of each individual being selected is equal throughout the entire sampling process (except when exclusions are made)"
  , isCorrect := true }

-- Define the list of all statements
def allStatements : List SamplingStatement := [statement1, statement2, statement3, statement4]

-- Theorem: The number of correct statements is 3
theorem correct_statements_count :
  (allStatements.filter (λ s => s.isCorrect)).length = 3 := by
  sorry


end NUMINAMATH_CALUDE_correct_statements_count_l1528_152895


namespace NUMINAMATH_CALUDE_solution_value_l1528_152861

theorem solution_value (x a : ℝ) (h : x = 3 ∧ 2 * x - 10 = 4 * a) : a = -1 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l1528_152861


namespace NUMINAMATH_CALUDE_two_numbers_problem_l1528_152827

theorem two_numbers_problem (x y : ℝ) : 
  x - y = 11 →
  x^2 + y^2 = 185 →
  (x - y)^2 = 121 →
  ((x = 13 ∧ y = 2) ∨ (x = -5 ∧ y = -16)) := by
sorry

end NUMINAMATH_CALUDE_two_numbers_problem_l1528_152827


namespace NUMINAMATH_CALUDE_line_perp_plane_implies_planes_perp_l1528_152801

/-- A line in 3D space -/
structure Line3D where
  -- Define properties of a line

/-- A plane in 3D space -/
structure Plane3D where
  -- Define properties of a plane

/-- Defines that a line is contained in a plane -/
def line_in_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Defines that a line is perpendicular to a plane -/
def line_perp_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Defines that two planes are perpendicular -/
def planes_perpendicular (p1 p2 : Plane3D) : Prop :=
  sorry

/-- Theorem: If a line contained in a plane is perpendicular to another plane,
    then the two planes are perpendicular -/
theorem line_perp_plane_implies_planes_perp
  (l : Line3D) (α β : Plane3D)
  (h1 : line_in_plane l α)
  (h2 : line_perp_plane l β) :
  planes_perpendicular α β :=
sorry

end NUMINAMATH_CALUDE_line_perp_plane_implies_planes_perp_l1528_152801


namespace NUMINAMATH_CALUDE_river_flow_rate_l1528_152852

/-- Given a river with specified dimensions and flow rate, calculate its flow speed in km/h -/
theorem river_flow_rate (depth : ℝ) (width : ℝ) (flow_volume : ℝ) :
  depth = 2 →
  width = 45 →
  flow_volume = 9000 →
  (flow_volume / (depth * width) / 1000 * 60) = 6 := by
  sorry

end NUMINAMATH_CALUDE_river_flow_rate_l1528_152852


namespace NUMINAMATH_CALUDE_smallest_non_triangle_forming_subtraction_l1528_152814

theorem smallest_non_triangle_forming_subtraction : ∃ (x : ℕ), 
  (x > 0) ∧ 
  (∀ (y : ℕ), y < x → (7 - y) + (24 - y) > (26 - y)) ∧
  ((7 - x) + (24 - x) ≤ (26 - x)) ∧
  x = 5 := by
sorry

end NUMINAMATH_CALUDE_smallest_non_triangle_forming_subtraction_l1528_152814


namespace NUMINAMATH_CALUDE_cashback_is_twelve_percent_l1528_152807

/-- Calculates the cashback percentage given the total cost, rebate, and final cost -/
def cashback_percentage (total_cost rebate final_cost : ℚ) : ℚ :=
  let cost_after_rebate := total_cost - rebate
  let cashback_amount := cost_after_rebate - final_cost
  (cashback_amount / cost_after_rebate) * 100

/-- Theorem stating that the cashback percentage is 12% given the problem conditions -/
theorem cashback_is_twelve_percent :
  cashback_percentage 150 25 110 = 12 := by
  sorry

end NUMINAMATH_CALUDE_cashback_is_twelve_percent_l1528_152807


namespace NUMINAMATH_CALUDE_fresh_grape_water_content_l1528_152822

/-- The percentage of water in raisins -/
def raisin_water_percentage : ℝ := 25

/-- The weight of fresh grapes used -/
def fresh_grape_weight : ℝ := 100

/-- The weight of raisins produced -/
def raisin_weight : ℝ := 20

/-- The percentage of water in fresh grapes -/
def fresh_grape_water_percentage : ℝ := 85

theorem fresh_grape_water_content :
  fresh_grape_water_percentage = 85 :=
sorry

end NUMINAMATH_CALUDE_fresh_grape_water_content_l1528_152822


namespace NUMINAMATH_CALUDE_linear_function_characterization_l1528_152804

/-- A function satisfying the given property for a fixed α -/
def SatisfiesProperty (α : ℝ) (f : ℕ+ → ℝ) : Prop :=
  ∀ (k m : ℕ+), α * m.val ≤ k.val ∧ k.val ≤ (α + 1) * m.val → f (k + m) = f k + f m

/-- The main theorem stating that any function satisfying the property is linear -/
theorem linear_function_characterization (α : ℝ) (hα : α > 0) (f : ℕ+ → ℝ) 
  (hf : SatisfiesProperty α f) : 
  ∃ (D : ℝ), ∀ (n : ℕ+), f n = D * n.val := by
  sorry

end NUMINAMATH_CALUDE_linear_function_characterization_l1528_152804


namespace NUMINAMATH_CALUDE_quadratic_roots_l1528_152865

theorem quadratic_roots (m : ℝ) : 
  ((-5 : ℝ)^2 + m * (-5) - 10 = 0) → ((2 : ℝ)^2 + m * 2 - 10 = 0) := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_l1528_152865


namespace NUMINAMATH_CALUDE_fuel_mixture_theorem_l1528_152853

/-- Represents the state of the fuel tank -/
structure TankState where
  z : Rat  -- Amount of brand Z gasoline
  y : Rat  -- Amount of brand Y gasoline

/-- Fills the tank with brand Z gasoline -/
def fill_z (s : TankState) : TankState :=
  { z := s.z + (1 - s.z - s.y), y := s.y }

/-- Fills the tank with brand Y gasoline -/
def fill_y (s : TankState) : TankState :=
  { z := s.z, y := s.y + (1 - s.z - s.y) }

/-- Removes half of the fuel from the tank -/
def half_empty (s : TankState) : TankState :=
  { z := s.z / 2, y := s.y / 2 }

theorem fuel_mixture_theorem : 
  let s0 : TankState := { z := 0, y := 0 }
  let s1 := fill_z s0
  let s2 := fill_y (TankState.mk (3/4) 0)
  let s3 := fill_z (half_empty s2)
  let s4 := fill_y (half_empty s3)
  s4.z = 7/16 := by sorry

end NUMINAMATH_CALUDE_fuel_mixture_theorem_l1528_152853


namespace NUMINAMATH_CALUDE_vocational_students_form_valid_set_l1528_152835

-- Define the universe of discourse
def Student : Type := String

-- Define the properties
def isDefinite (s : Set Student) : Prop := sorry
def isDistinct (s : Set Student) : Prop := sorry
def isUnordered (s : Set Student) : Prop := sorry

-- Define the sets corresponding to each option
def tallStudents : Set Student := sorry
def vocationalStudents : Set Student := sorry
def goodStudents : Set Student := sorry
def lushTrees : Set Student := sorry

-- Define what makes a valid set
def isValidSet (s : Set Student) : Prop :=
  isDefinite s ∧ isDistinct s ∧ isUnordered s

-- Theorem statement
theorem vocational_students_form_valid_set :
  isValidSet vocationalStudents ∧
  ¬isValidSet tallStudents ∧
  ¬isValidSet goodStudents ∧
  ¬isValidSet lushTrees :=
sorry

end NUMINAMATH_CALUDE_vocational_students_form_valid_set_l1528_152835


namespace NUMINAMATH_CALUDE_install_time_proof_l1528_152821

/-- Calculates the time needed to install remaining windows -/
def time_to_install_remaining (total : ℕ) (installed : ℕ) (time_per_window : ℕ) : ℕ :=
  (total - installed) * time_per_window

/-- Proves that the time to install remaining windows is 48 hours -/
theorem install_time_proof (total : ℕ) (installed : ℕ) (time_per_window : ℕ) 
  (h1 : total = 14)
  (h2 : installed = 8)
  (h3 : time_per_window = 8) :
  time_to_install_remaining total installed time_per_window = 48 := by
  sorry

#eval time_to_install_remaining 14 8 8

end NUMINAMATH_CALUDE_install_time_proof_l1528_152821


namespace NUMINAMATH_CALUDE_force_balance_l1528_152884

/-- A force in 2D space represented by its x and y components -/
structure Force where
  x : ℝ
  y : ℝ

/-- The sum of two forces -/
def Force.add (f g : Force) : Force :=
  ⟨f.x + g.x, f.y + g.y⟩

/-- The negation of a force -/
def Force.neg (f : Force) : Force :=
  ⟨-f.x, -f.y⟩

/-- Given two forces F₁ and F₂, prove that F₃ balances the system -/
theorem force_balance (F₁ F₂ F₃ : Force) 
    (h₁ : F₁ = ⟨1, 1⟩) 
    (h₂ : F₂ = ⟨2, 3⟩) 
    (h₃ : F₃ = ⟨-3, -4⟩) : 
  F₃.add (F₁.add F₂) = ⟨0, 0⟩ := by
  sorry


end NUMINAMATH_CALUDE_force_balance_l1528_152884


namespace NUMINAMATH_CALUDE_shirt_tie_combinations_l1528_152883

/-- The number of possible shirt-and-tie combinations given a set of shirts and ties with restrictions -/
theorem shirt_tie_combinations (total_shirts : ℕ) (total_ties : ℕ) (restricted_shirts : ℕ) (restricted_ties : ℕ) :
  total_shirts = 8 →
  total_ties = 7 →
  restricted_shirts = 3 →
  restricted_ties = 2 →
  total_shirts * total_ties - restricted_shirts * restricted_ties = 50 := by
sorry

end NUMINAMATH_CALUDE_shirt_tie_combinations_l1528_152883


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l1528_152888

-- Problem 1
theorem simplify_expression_1 (x y z : ℝ) :
  (x + y + z)^2 - (x + y - z)^2 = 4*z*(x + y) := by sorry

-- Problem 2
theorem simplify_expression_2 (a b : ℝ) :
  (a + 2*b)^2 - 2*(a + 2*b)*(a - 2*b) + (a - 2*b)^2 = 16*b^2 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l1528_152888


namespace NUMINAMATH_CALUDE_cloth_length_problem_l1528_152815

theorem cloth_length_problem (initial_length : ℕ) : 
  (∃ (remainder1 remainder2 : ℕ),
    initial_length = 32 + remainder1 ∧
    initial_length = 20 + remainder2 ∧
    remainder2 = 3 * remainder1) →
  initial_length = 38 :=
by
  sorry

end NUMINAMATH_CALUDE_cloth_length_problem_l1528_152815


namespace NUMINAMATH_CALUDE_fruits_remaining_l1528_152829

/-- Calculates the number of fruits remaining after picking from multiple trees -/
theorem fruits_remaining
  (num_trees : ℕ)
  (fruits_per_tree : ℕ)
  (fraction_picked : ℚ)
  (h1 : num_trees = 8)
  (h2 : fruits_per_tree = 200)
  (h3 : fraction_picked = 2/5) :
  num_trees * fruits_per_tree - num_trees * (fruits_per_tree * fraction_picked) = 960 :=
by
  sorry

#check fruits_remaining

end NUMINAMATH_CALUDE_fruits_remaining_l1528_152829


namespace NUMINAMATH_CALUDE_quadratic_curve_coefficient_l1528_152878

theorem quadratic_curve_coefficient (p q y1 y2 : ℝ) : 
  (y1 = p + q + 5) →
  (y2 = p - q + 5) →
  (y1 + y2 = 14) →
  p = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_curve_coefficient_l1528_152878


namespace NUMINAMATH_CALUDE_condition_necessity_sufficiency_l1528_152857

theorem condition_necessity_sufficiency : 
  (∀ x : ℝ, (x + 1) * (x^2 + 2) > 0 → (x + 1) * (x + 2) > 0) ∧ 
  (∃ x : ℝ, (x + 1) * (x + 2) > 0 ∧ (x + 1) * (x^2 + 2) ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_condition_necessity_sufficiency_l1528_152857


namespace NUMINAMATH_CALUDE_tax_reduction_l1528_152879

theorem tax_reduction (T C X : ℝ) (h1 : T > 0) (h2 : C > 0) (h3 : X > 0) : 
  (T * (1 - X / 100) * (C * 1.2) = 0.84 * (T * C)) → X = 30 := by
  sorry

end NUMINAMATH_CALUDE_tax_reduction_l1528_152879


namespace NUMINAMATH_CALUDE_xyz_sum_l1528_152850

theorem xyz_sum (x y z : ℕ+) 
  (eq1 : x * y + z = 47)
  (eq2 : y * z + x = 47)
  (eq3 : z * x + y = 47) : 
  x + y + z = 48 := by
sorry

end NUMINAMATH_CALUDE_xyz_sum_l1528_152850


namespace NUMINAMATH_CALUDE_moss_pollen_scientific_notation_l1528_152896

theorem moss_pollen_scientific_notation (d : ℝ) (n : ℤ) :
  d = 0.0000084 →
  d = 8.4 * (10 : ℝ) ^ n →
  n = -6 :=
by
  sorry

end NUMINAMATH_CALUDE_moss_pollen_scientific_notation_l1528_152896


namespace NUMINAMATH_CALUDE_circle_center_correct_l1528_152875

/-- The equation of a circle in the form ax² + bx + cy² + dy + e = 0 --/
structure CircleEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ

/-- The center of a circle --/
structure CircleCenter where
  x : ℝ
  y : ℝ

/-- Function to find the center of a circle given its equation --/
def findCircleCenter (eq : CircleEquation) : CircleCenter :=
  sorry

theorem circle_center_correct :
  let eq := CircleEquation.mk 4 8 4 (-24) 96
  let center := findCircleCenter eq
  center.x = -1 ∧ center.y = 3 := by sorry

end NUMINAMATH_CALUDE_circle_center_correct_l1528_152875


namespace NUMINAMATH_CALUDE_steve_earnings_l1528_152820

/-- Calculates the amount of money an author keeps after selling books and paying an agent. -/
def authorEarnings (totalCopies : ℕ) (advanceCopies : ℕ) (earningsPerCopy : ℚ) (agentPercentage : ℚ) : ℚ :=
  let copiesForEarnings := totalCopies - advanceCopies
  let totalEarnings := copiesForEarnings * earningsPerCopy
  let agentCut := totalEarnings * agentPercentage
  totalEarnings - agentCut

/-- Proves that given the conditions of Steve's book sales, he keeps $1,620,000 after paying his agent. -/
theorem steve_earnings :
  authorEarnings 1000000 100000 2 (1/10) = 1620000 := by
  sorry

end NUMINAMATH_CALUDE_steve_earnings_l1528_152820


namespace NUMINAMATH_CALUDE_tenth_term_of_sequence_l1528_152868

/-- The sequence a_n defined for natural numbers n -/
def a (n : ℕ) : ℚ := (2 * n) / (2 * n + 1)

/-- The 10th term of the sequence equals 20/21 -/
theorem tenth_term_of_sequence : a 10 = 20 / 21 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_of_sequence_l1528_152868


namespace NUMINAMATH_CALUDE_winning_sequence_exists_l1528_152810

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def first_digit (n : ℕ) : ℕ :=
  if n < 10 then n else first_digit (n / 10)

def last_digit (n : ℕ) : ℕ := n % 10

def valid_sequence (seq : List ℕ) : Prop :=
  seq.length > 0 ∧
  ∀ n ∈ seq, is_prime n ∧ n ≤ 100 ∧
  ∀ i < seq.length - 1, last_digit (seq.get ⟨i, by sorry⟩) = first_digit (seq.get ⟨i+1, by sorry⟩) ∧
  ∀ i j, i ≠ j → seq.get ⟨i, by sorry⟩ ≠ seq.get ⟨j, by sorry⟩

theorem winning_sequence_exists :
  ∃ seq : List ℕ, valid_sequence seq ∧ seq.length = 3 ∧
  ∀ p : ℕ, is_prime p → p ≤ 100 → p ∉ seq →
    (seq.length > 0 → first_digit p ≠ last_digit (seq.getLast (by sorry))) :=
sorry

end NUMINAMATH_CALUDE_winning_sequence_exists_l1528_152810


namespace NUMINAMATH_CALUDE_product_equals_120_l1528_152894

theorem product_equals_120 (n : ℕ) (h : n = 3) : 
  (n - 2) * (n - 1) * n * (n + 1) * (n + 2) = 120 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_120_l1528_152894


namespace NUMINAMATH_CALUDE_first_question_percentage_l1528_152892

/-- The percentage of students who answered the first question correctly -/
def first_question_correct : ℝ := sorry

/-- The percentage of students who answered the second question correctly -/
def second_question_correct : ℝ := 35

/-- The percentage of students who answered neither question correctly -/
def neither_correct : ℝ := 20

/-- The percentage of students who answered both questions correctly -/
def both_correct : ℝ := 30

/-- Theorem stating that the percentage of students who answered the first question correctly is 75% -/
theorem first_question_percentage :
  first_question_correct = 75 :=
by sorry

end NUMINAMATH_CALUDE_first_question_percentage_l1528_152892


namespace NUMINAMATH_CALUDE_vectors_not_parallel_l1528_152855

def vector_a : Fin 2 → ℝ := ![2, 0]
def vector_b : Fin 2 → ℝ := ![0, 2]

theorem vectors_not_parallel : ¬ (∃ (k : ℝ), vector_a = k • vector_b) := by
  sorry

end NUMINAMATH_CALUDE_vectors_not_parallel_l1528_152855


namespace NUMINAMATH_CALUDE_jennifer_dogs_count_l1528_152898

/-- The number of dogs Jennifer has -/
def number_of_dogs : ℕ := 2

/-- Time in minutes to groom each dog -/
def grooming_time_per_dog : ℕ := 20

/-- Number of days Jennifer grooms her dogs -/
def grooming_days : ℕ := 30

/-- Total time in hours Jennifer spends grooming in 30 days -/
def total_grooming_time_hours : ℕ := 20

theorem jennifer_dogs_count :
  number_of_dogs * grooming_time_per_dog * grooming_days = total_grooming_time_hours * 60 :=
by sorry

end NUMINAMATH_CALUDE_jennifer_dogs_count_l1528_152898


namespace NUMINAMATH_CALUDE_y_squared_equals_zx_sufficient_not_necessary_l1528_152862

-- Define a function to check if three numbers form an arithmetic sequence
def is_arithmetic_sequence (a b c : ℝ) : Prop :=
  b - a = c - b

-- Define the theorem
theorem y_squared_equals_zx_sufficient_not_necessary 
  (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (is_arithmetic_sequence (Real.log x) (Real.log y) (Real.log z) → y^2 = z*x) ∧
  ¬(y^2 = z*x → is_arithmetic_sequence (Real.log x) (Real.log y) (Real.log z)) :=
by sorry

end NUMINAMATH_CALUDE_y_squared_equals_zx_sufficient_not_necessary_l1528_152862


namespace NUMINAMATH_CALUDE_volunteer_quota_allocation_l1528_152849

theorem volunteer_quota_allocation :
  let n : ℕ := 24  -- Total number of quotas
  let k : ℕ := 3   -- Number of venues
  let total_partitions : ℕ := Nat.choose (n - 1) (k - 1)
  let invalid_partitions : ℕ := (k - 1) * Nat.choose k 2 + 1
  total_partitions - invalid_partitions = 222 := by
  sorry

end NUMINAMATH_CALUDE_volunteer_quota_allocation_l1528_152849


namespace NUMINAMATH_CALUDE_exam_score_theorem_l1528_152838

/-- Represents an examination scoring system and attempts to prove
    the total number of questions attempted given certain conditions. -/
theorem exam_score_theorem (marks_per_correct : ℕ) (marks_per_wrong : ℤ)
                            (total_marks : ℤ) (correct_answers : ℕ) :
  marks_per_correct = 4 →
  marks_per_wrong = -1 →
  total_marks = 140 →
  correct_answers = 40 →
  (correct_answers : ℤ) * marks_per_correct + 
    (total_marks - (correct_answers : ℤ) * marks_per_correct) / marks_per_wrong = 60 := by
  sorry


end NUMINAMATH_CALUDE_exam_score_theorem_l1528_152838


namespace NUMINAMATH_CALUDE_cars_in_parking_lot_l1528_152837

/-- The number of wheels observed in the parking lot -/
def total_wheels : ℕ := 68

/-- The number of wheels on a standard car -/
def wheels_per_car : ℕ := 4

/-- The number of cars in the parking lot -/
def number_of_cars : ℕ := total_wheels / wheels_per_car

theorem cars_in_parking_lot : number_of_cars = 17 := by
  sorry

end NUMINAMATH_CALUDE_cars_in_parking_lot_l1528_152837

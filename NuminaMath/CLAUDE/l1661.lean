import Mathlib

namespace cube_structure_ratio_l1661_166121

/-- A structure formed by joining unit cubes -/
structure CubeStructure where
  num_cubes : ℕ
  central_cube : Bool
  shared_faces : ℕ

/-- Calculate the volume of the cube structure -/
def volume (s : CubeStructure) : ℕ :=
  s.num_cubes

/-- Calculate the surface area of the cube structure -/
def surface_area (s : CubeStructure) : ℕ :=
  (s.num_cubes - 1) * 5

/-- The ratio of volume to surface area -/
def volume_to_surface_ratio (s : CubeStructure) : ℚ :=
  (volume s : ℚ) / (surface_area s : ℚ)

/-- Theorem stating the ratio of volume to surface area for the specific cube structure -/
theorem cube_structure_ratio :
  ∃ (s : CubeStructure),
    s.num_cubes = 8 ∧
    s.central_cube = true ∧
    s.shared_faces = 6 ∧
    volume_to_surface_ratio s = 4 / 15 := by
  sorry

end cube_structure_ratio_l1661_166121


namespace accidental_vs_correct_calculation_l1661_166187

theorem accidental_vs_correct_calculation (x : ℚ) : 
  7 * ((x + 24) / 5) = 70 → (5 * x + 24) / 7 = 22 := by
  sorry

end accidental_vs_correct_calculation_l1661_166187


namespace fraction_equality_l1661_166161

theorem fraction_equality (x y z w k : ℝ) : 
  (9 / (x + y + w) = k / (x + z + w)) ∧ 
  (k / (x + z + w) = 12 / (z - y)) → 
  k = 21 := by
sorry

end fraction_equality_l1661_166161


namespace camping_trip_items_l1661_166138

theorem camping_trip_items (total_items : ℕ) 
  (tent_stakes : ℕ) (drink_mix : ℕ) (water_bottles : ℕ) : 
  total_items = 22 → 
  drink_mix = 3 * tent_stakes → 
  water_bottles = tent_stakes + 2 → 
  total_items = tent_stakes + drink_mix + water_bottles → 
  tent_stakes = 4 := by
sorry

end camping_trip_items_l1661_166138


namespace improper_integral_convergence_l1661_166110

open Real MeasureTheory

/-- The improper integral ∫[a to b] 1/(x-a)^α dx converges if and only if 0 < α < 1, given α > 0 and b > a -/
theorem improper_integral_convergence 
  (a b : ℝ) (α : ℝ) 
  (h1 : α > 0) 
  (h2 : b > a) : 
  (∃ (I : ℝ), ∫ x in a..b, 1 / (x - a) ^ α = I) ↔ 0 < α ∧ α < 1 :=
sorry

end improper_integral_convergence_l1661_166110


namespace income_comparison_l1661_166188

theorem income_comparison (juan tim mart : ℝ) 
  (h1 : tim = 0.6 * juan) 
  (h2 : mart = 0.78 * juan) : 
  (mart - tim) / tim * 100 = 30 := by
  sorry

end income_comparison_l1661_166188


namespace absolute_value_equation_solution_difference_l1661_166116

theorem absolute_value_equation_solution_difference : ∃ (x₁ x₂ : ℝ), 
  (|x₁ - 3| = 15 ∧ |x₂ - 3| = 15 ∧ x₁ ≠ x₂) ∧ |x₁ - x₂| = 30 := by
  sorry

end absolute_value_equation_solution_difference_l1661_166116


namespace largest_angle_is_E_l1661_166114

/-- Represents a hexagon with specific angle properties -/
structure Hexagon where
  /-- Angle A is 100 degrees -/
  angle_A : ℝ
  angle_A_eq : angle_A = 100

  /-- Angle B is 120 degrees -/
  angle_B : ℝ
  angle_B_eq : angle_B = 120

  /-- Angles C and D are equal -/
  angle_C : ℝ
  angle_D : ℝ
  angle_C_eq_D : angle_C = angle_D

  /-- Angle E is 30 degrees more than the average of angles C, D, and F -/
  angle_E : ℝ
  angle_F : ℝ
  angle_E_eq : angle_E = (angle_C + angle_D + angle_F) / 3 + 30

  /-- The sum of all angles in a hexagon is 720 degrees -/
  sum_of_angles : angle_A + angle_B + angle_C + angle_D + angle_E + angle_F = 720

/-- Theorem: The largest angle in the hexagon is 147.5 degrees -/
theorem largest_angle_is_E (h : Hexagon) : h.angle_E = 147.5 := by
  sorry

end largest_angle_is_E_l1661_166114


namespace opposite_numbers_sum_l1661_166175

theorem opposite_numbers_sum (a b : ℝ) : a + b = 0 → 3*a + 3*b + 2 = 2 := by
  sorry

end opposite_numbers_sum_l1661_166175


namespace residue_13_2045_mod_19_l1661_166169

theorem residue_13_2045_mod_19 : (13 ^ 2045 : ℕ) % 19 = 9 := by sorry

end residue_13_2045_mod_19_l1661_166169


namespace cake_sector_angle_l1661_166144

theorem cake_sector_angle (total_sectors : ℕ) (probability : ℚ) : 
  total_sectors = 6 → probability = 1/8 → 
  ∃ (angle : ℚ), angle = 45 ∧ probability = angle / 360 := by
  sorry

end cake_sector_angle_l1661_166144


namespace anna_final_collection_l1661_166111

structure StampCollection :=
  (nature : ℕ)
  (architecture : ℕ)
  (animals : ℕ)

def initial_anna : StampCollection := ⟨10, 15, 12⟩
def initial_alison : StampCollection := ⟨8, 10, 10⟩
def initial_jeff : StampCollection := ⟨12, 9, 10⟩

def transaction1 (anna alison : StampCollection) : StampCollection :=
  ⟨anna.nature + alison.nature / 2, anna.architecture + alison.architecture / 2, anna.animals + alison.animals / 2⟩

def transaction2 (anna : StampCollection) : StampCollection :=
  ⟨anna.nature + 2, anna.architecture, anna.animals - 1⟩

def transaction3 (anna : StampCollection) : StampCollection :=
  ⟨anna.nature, anna.architecture + 3, anna.animals - 5⟩

def transaction4 (anna : StampCollection) : StampCollection :=
  ⟨anna.nature + 7, anna.architecture, anna.animals - 4⟩

def final_anna : StampCollection :=
  transaction4 (transaction3 (transaction2 (transaction1 initial_anna initial_alison)))

theorem anna_final_collection :
  final_anna = ⟨23, 23, 7⟩ := by sorry

end anna_final_collection_l1661_166111


namespace product_xy_l1661_166125

theorem product_xy (x y : ℚ) 
  (h1 : 1/x + 1/y = 4) 
  (h2 : 1/x - 1/y = -6) : 
  x * y = -1/5 := by
sorry

end product_xy_l1661_166125


namespace cars_between_black_and_white_l1661_166135

/-- Given a row of 20 cars, with a black car 16th from the right and a white car 11th from the left,
    the number of cars between the black and white cars is 5. -/
theorem cars_between_black_and_white :
  ∀ (total_cars : ℕ) (black_from_right : ℕ) (white_from_left : ℕ),
    total_cars = 20 →
    black_from_right = 16 →
    white_from_left = 11 →
    white_from_left - (total_cars - black_from_right + 1) - 1 = 5 := by
  sorry

end cars_between_black_and_white_l1661_166135


namespace shoe_multiple_l1661_166165

/-- Given the following conditions:
  - Bonny has 13 pairs of shoes
  - Bonny's shoes are 5 less than a certain multiple of Becky's shoes
  - Bobby has 3 times as many shoes as Becky
  - Bobby has 27 pairs of shoes
  Prove that the multiple of Becky's shoes that is 5 more than Bonny's shoes is 2. -/
theorem shoe_multiple (bonny_shoes : ℕ) (bobby_shoes : ℕ) (becky_shoes : ℕ) (m : ℕ) :
  bonny_shoes = 13 →
  ∃ m, bonny_shoes + 5 = m * becky_shoes →
  bobby_shoes = 3 * becky_shoes →
  bobby_shoes = 27 →
  m = 2 :=
by sorry

end shoe_multiple_l1661_166165


namespace smallest_primer_l1661_166128

/-- A number is primer if it has a prime number of distinct prime factors -/
def isPrimer (n : ℕ) : Prop :=
  Nat.Prime (Finset.card (Nat.factors n).toFinset)

/-- 6 is the smallest primer number -/
theorem smallest_primer : ∀ k : ℕ, k > 0 → k < 6 → ¬ isPrimer k ∧ isPrimer 6 :=
sorry

end smallest_primer_l1661_166128


namespace pie_cost_satisfies_conditions_l1661_166103

/-- The cost of one pie in rubles -/
def pie_cost : ℚ := 20

/-- The total value of Masha's two-ruble coins -/
def two_ruble_coins : ℚ := 4 * pie_cost - 60

/-- The total value of Masha's five-ruble coins -/
def five_ruble_coins : ℚ := 5 * pie_cost - 60

/-- Theorem stating that the pie cost satisfies all given conditions -/
theorem pie_cost_satisfies_conditions :
  (4 * pie_cost = two_ruble_coins + 60) ∧
  (5 * pie_cost = five_ruble_coins + 60) ∧
  (6 * pie_cost = two_ruble_coins + five_ruble_coins + 60) :=
by sorry

#check pie_cost_satisfies_conditions

end pie_cost_satisfies_conditions_l1661_166103


namespace rainy_days_count_l1661_166171

theorem rainy_days_count (n : ℤ) : 
  (∃ (R NR : ℤ),
    R + NR = 7 ∧ 
    n * R + 4 * NR = 26 ∧ 
    4 * NR - n * R = 14 ∧ 
    R ≥ 0 ∧ NR ≥ 0) → 
  (∃ (R : ℤ), R = 2 ∧ 
    (∃ (NR : ℤ), 
      R + NR = 7 ∧ 
      n * R + 4 * NR = 26 ∧ 
      4 * NR - n * R = 14 ∧ 
      R ≥ 0 ∧ NR ≥ 0)) :=
by sorry

end rainy_days_count_l1661_166171


namespace intersection_A_B_l1661_166149

-- Define set A
def A : Set ℝ := {x | x^2 - 4*x + 3 < 0}

-- Define set B
def B : Set ℝ := {x | ∃ y, y = Real.log (x - 2)}

-- Theorem statement
theorem intersection_A_B : A ∩ B = Set.Ioo 2 3 := by
  sorry

end intersection_A_B_l1661_166149


namespace discount_difference_l1661_166176

theorem discount_difference (bill : ℝ) (single_discount : ℝ) (first_discount : ℝ) (second_discount : ℝ) :
  bill = 20000 ∧
  single_discount = 0.3 ∧
  first_discount = 0.25 ∧
  second_discount = 0.05 →
  bill * (1 - first_discount) * (1 - second_discount) - bill * (1 - single_discount) = 250 :=
by sorry

end discount_difference_l1661_166176


namespace roof_ratio_l1661_166141

theorem roof_ratio (length width : ℝ) : 
  length * width = 784 →
  length - width = 42 →
  length / width = 4 :=
by
  sorry

end roof_ratio_l1661_166141


namespace allison_total_supplies_l1661_166131

/-- Represents the number of craft supplies bought by a person -/
structure CraftSupplies where
  glueSticks : ℕ
  constructionPaper : ℕ

/-- The total number of craft supplies -/
def CraftSupplies.total (cs : CraftSupplies) : ℕ :=
  cs.glueSticks + cs.constructionPaper

/-- Given information about Marie's purchases -/
def marie : CraftSupplies :=
  { glueSticks := 15
    constructionPaper := 30 }

/-- Theorem stating the total number of craft supplies Allison bought -/
theorem allison_total_supplies : 
  ∃ (allison : CraftSupplies), 
    (allison.glueSticks = marie.glueSticks + 8) ∧ 
    (allison.constructionPaper * 6 = marie.constructionPaper) ∧ 
    (allison.total = 28) := by
  sorry

end allison_total_supplies_l1661_166131


namespace natural_raisin_cost_l1661_166134

/-- The cost per scoop of golden seedless raisins in dollars -/
def golden_cost : ℚ := 255/100

/-- The number of scoops of golden seedless raisins -/
def golden_scoops : ℕ := 20

/-- The number of scoops of natural seedless raisins -/
def natural_scoops : ℕ := 20

/-- The cost per scoop of the mixture in dollars -/
def mixture_cost : ℚ := 3

/-- The cost per scoop of natural seedless raisins in dollars -/
def natural_cost : ℚ := 345/100

theorem natural_raisin_cost : 
  (golden_cost * golden_scoops + natural_cost * natural_scoops) / (golden_scoops + natural_scoops) = mixture_cost :=
sorry

end natural_raisin_cost_l1661_166134


namespace sum_interior_angles_regular_polygon_l1661_166177

/-- Theorem: Sum of interior angles of a regular polygon with 20-degree exterior angles -/
theorem sum_interior_angles_regular_polygon (n : ℕ) (h1 : n > 2) 
  (h2 : (360 : ℝ) / n = 20) : (n - 2 : ℝ) * 180 = 2880 := by
  sorry

end sum_interior_angles_regular_polygon_l1661_166177


namespace apples_left_l1661_166120

theorem apples_left (initial_apples used_apples : ℕ) 
  (h1 : initial_apples = 43)
  (h2 : used_apples = 41) :
  initial_apples - used_apples = 2 := by
  sorry

end apples_left_l1661_166120


namespace max_value_product_sum_l1661_166162

theorem max_value_product_sum (A M C : ℕ) (h : A + M + C = 15) :
  (∀ a m c : ℕ, a + m + c = 15 → A * M * C + A * M + M * C + C * A ≥ a * m * c + a * m + m * c + c * a) →
  A * M * C + A * M + M * C + C * A = 200 := by
sorry

end max_value_product_sum_l1661_166162


namespace arithmetic_sequence_common_difference_l1661_166117

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h_arithmetic : is_arithmetic_sequence a)
  (h_sum : a 1 + a 7 = -8)
  (h_a2 : a 2 = 2) :
  ∃ d : ℝ, d = -3 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
sorry

end arithmetic_sequence_common_difference_l1661_166117


namespace class_size_is_69_l1661_166112

/-- Represents the number of students in a class with given enrollment data for French and German courses -/
def total_students (french : ℕ) (german : ℕ) (both : ℕ) (neither : ℕ) : ℕ :=
  (french + german - both) + neither

/-- Theorem stating that the total number of students in the class is 69 -/
theorem class_size_is_69 :
  total_students 41 22 9 15 = 69 := by
  sorry

end class_size_is_69_l1661_166112


namespace exponent_multiplication_l1661_166124

theorem exponent_multiplication (a : ℝ) : a^3 * a^2 = a^5 := by
  sorry

end exponent_multiplication_l1661_166124


namespace partnership_gain_l1661_166152

/-- Represents the investment and profit structure of a partnership --/
structure Partnership where
  raman_investment : ℝ
  lakshmi_share : ℝ
  profit_ratio : ℝ → ℝ → ℝ → Prop

/-- Calculates the total annual gain of the partnership --/
def total_annual_gain (p : Partnership) : ℝ :=
  3 * p.lakshmi_share

/-- Theorem stating that the total annual gain of the partnership is 36000 --/
theorem partnership_gain (p : Partnership) 
  (h1 : p.profit_ratio (p.raman_investment * 12) (2 * p.raman_investment * 6) (3 * p.raman_investment * 4))
  (h2 : p.lakshmi_share = 12000) : 
  total_annual_gain p = 36000 := by
  sorry

end partnership_gain_l1661_166152


namespace count_numbers_with_property_l1661_166184

-- Define a two-digit number
def two_digit_number (a b : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9

-- Define the property we're interested in
def has_property (a b : ℕ) : Prop :=
  two_digit_number a b ∧ (10 * a + b - (a + b)) % 10 = 6

-- The theorem to prove
theorem count_numbers_with_property :
  ∃ (S : Finset ℕ), S.card = 10 ∧ 
  (∀ n, n ∈ S ↔ ∃ a b, has_property a b ∧ n = 10 * a + b) :=
sorry

end count_numbers_with_property_l1661_166184


namespace cost_not_proportional_cost_increases_linearly_l1661_166157

/-- Represents the cost of a telegram -/
def telegram_cost (a b n : ℝ) : ℝ := a + b * n

/-- The cost is not proportional to the number of words -/
theorem cost_not_proportional (a b : ℝ) (h : a ≠ 0) :
  ¬∃ k : ℝ, ∀ n : ℝ, telegram_cost a b n = k * n :=
sorry

/-- The cost increases linearly with the number of words -/
theorem cost_increases_linearly (a b : ℝ) (h : b > 0) :
  ∀ n₁ n₂ : ℝ, n₁ < n₂ → telegram_cost a b n₁ < telegram_cost a b n₂ :=
sorry

end cost_not_proportional_cost_increases_linearly_l1661_166157


namespace younger_person_age_l1661_166194

/-- 
Given two persons whose ages differ by 20 years, and 10 years ago the elder was 5 times as old as the younger,
prove that the present age of the younger person is 15 years.
-/
theorem younger_person_age (y e : ℕ) : 
  e = y + 20 → 
  e - 10 = 5 * (y - 10) → 
  y = 15 := by
  sorry

end younger_person_age_l1661_166194


namespace matrix_power_four_l1661_166180

def A : Matrix (Fin 2) (Fin 2) ℤ := !![2, -1; 1, 1]

theorem matrix_power_four :
  A ^ 4 = !![0, -9; 9, -9] := by sorry

end matrix_power_four_l1661_166180


namespace sledding_time_difference_l1661_166191

/-- Given the conditions of Mary and Ann's sledding trip, prove that Ann's trip takes 13 minutes longer than Mary's. -/
theorem sledding_time_difference 
  (mary_hill_length : ℝ) 
  (mary_speed : ℝ) 
  (ann_hill_length : ℝ) 
  (ann_speed : ℝ) 
  (h1 : mary_hill_length = 630)
  (h2 : mary_speed = 90)
  (h3 : ann_hill_length = 800)
  (h4 : ann_speed = 40) :
  ann_hill_length / ann_speed - mary_hill_length / mary_speed = 13 := by
  sorry

end sledding_time_difference_l1661_166191


namespace inequality_solutions_l1661_166181

/-- The solution set of the inequality 2x^2 + x - 3 < 0 -/
def solution_set_1 : Set ℝ := { x | -3/2 < x ∧ x < 1 }

/-- The solution set of the inequality x(9 - x) > 0 -/
def solution_set_2 : Set ℝ := { x | 0 < x ∧ x < 9 }

theorem inequality_solutions :
  (∀ x : ℝ, x ∈ solution_set_1 ↔ 2*x^2 + x - 3 < 0) ∧
  (∀ x : ℝ, x ∈ solution_set_2 ↔ x*(9 - x) > 0) := by
  sorry

end inequality_solutions_l1661_166181


namespace negation_of_existence_proposition_l1661_166143

theorem negation_of_existence_proposition :
  ¬(∃ c : ℝ, c > 0 ∧ ∃ x : ℝ, x^2 - x + c = 0) ↔
  (∀ c : ℝ, c > 0 → ∀ x : ℝ, x^2 - x + c ≠ 0) :=
by sorry

end negation_of_existence_proposition_l1661_166143


namespace valerie_stamps_l1661_166107

/-- The number of stamps Valerie needs for all her envelopes --/
def total_stamps : ℕ :=
  let thank_you_cards := 3
  let water_bill := 1
  let electric_bill := 2
  let internet_bill := 3
  let bills := water_bill + electric_bill + internet_bill
  let rebates := bills + 3
  let job_applications := 2 * rebates
  thank_you_cards + bills + 2 * rebates + job_applications

/-- Theorem stating that Valerie needs 33 stamps in total --/
theorem valerie_stamps : total_stamps = 33 := by
  sorry

end valerie_stamps_l1661_166107


namespace train_stop_time_l1661_166145

/-- Proves that a train with given speeds stops for 20 minutes per hour -/
theorem train_stop_time (speed_without_stops speed_with_stops : ℝ) 
  (h1 : speed_without_stops = 48)
  (h2 : speed_with_stops = 32) :
  (1 - speed_with_stops / speed_without_stops) * 60 = 20 := by
  sorry

#check train_stop_time

end train_stop_time_l1661_166145


namespace slopes_negative_reciprocals_min_area_ANB_l1661_166148

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define points M and N
def M : ℝ × ℝ := (1, 0)
def N : ℝ × ℝ := (-1, 0)

-- Define a line passing through M
def line_through_M (k : ℝ) (x y : ℝ) : Prop := y = k*(x - 1)

-- Define points A and B as intersections of the line and parabola
def point_A (k : ℝ) : ℝ × ℝ := sorry
def point_B (k : ℝ) : ℝ × ℝ := sorry

-- Define slopes of NA and NB
def slope_NA (k : ℝ) : ℝ := sorry
def slope_NB (k : ℝ) : ℝ := sorry

-- Define area of triangle ANB
def area_ANB (k : ℝ) : ℝ := sorry

theorem slopes_negative_reciprocals :
  ∀ k : ℝ, k ≠ 0 → slope_NA k * slope_NB k = -1 :=
sorry

theorem min_area_ANB :
  ∃ min_area : ℝ, min_area = 4 ∧ ∀ k : ℝ, k ≠ 0 → area_ANB k ≥ min_area :=
sorry

end slopes_negative_reciprocals_min_area_ANB_l1661_166148


namespace fixed_points_of_specific_quadratic_min_value_of_ratio_sum_range_of_a_for_always_fixed_point_l1661_166158

-- Define a quadratic function
def quadratic (m n t : ℝ) (x : ℝ) : ℝ := m * x^2 + n * x + t

-- Define a fixed point
def is_fixed_point (m n t : ℝ) (x : ℝ) : Prop := quadratic m n t x = x

theorem fixed_points_of_specific_quadratic :
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ is_fixed_point 1 (-1) (-3) x1 ∧ is_fixed_point 1 (-1) (-3) x2 ∧ x1 = -1 ∧ x2 = 3 := by sorry

theorem min_value_of_ratio_sum :
  ∀ a : ℝ, a > 1 →
  ∃ x1 x2 : ℝ, x1 > 0 ∧ x2 > 0 ∧ x1 ≠ x2 ∧
  is_fixed_point 2 (-(3+a)) (a-1) x1 ∧
  is_fixed_point 2 (-(3+a)) (a-1) x2 →
  (x1 / x2 + x2 / x1 ≥ 8) ∧ (∃ a0 : ℝ, a0 > 1 ∧ ∃ x3 x4 : ℝ, x3 / x4 + x4 / x3 = 8) := by sorry

theorem range_of_a_for_always_fixed_point :
  ∀ a : ℝ, a ≠ 0 →
  (∀ b : ℝ, ∃ x : ℝ, is_fixed_point a (b+1) (b-1) x) ↔
  (a > 0 ∧ a ≤ 1) := by sorry

end fixed_points_of_specific_quadratic_min_value_of_ratio_sum_range_of_a_for_always_fixed_point_l1661_166158


namespace alpha_squared_gt_beta_squared_l1661_166129

theorem alpha_squared_gt_beta_squared 
  (α β : Real) 
  (h1 : α ∈ Set.Icc (-π/2) (π/2)) 
  (h2 : β ∈ Set.Icc (-π/2) (π/2)) 
  (h3 : α * Real.sin α - β * Real.sin β > 0) : 
  α^2 > β^2 :=
sorry

end alpha_squared_gt_beta_squared_l1661_166129


namespace line_intersects_circle_twice_l1661_166130

/-- The circle C with equation x^2 + y^2 - 2x - 6y - 15 = 0 -/
def Circle (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 6*y - 15 = 0

/-- The line l with equation (1+3k)x + (3-2k)y + 4k - 17 = 0 for any real k -/
def Line (k x y : ℝ) : Prop :=
  (1+3*k)*x + (3-2*k)*y + 4*k - 17 = 0

/-- The theorem stating that the line intersects the circle at exactly two points for any real k -/
theorem line_intersects_circle_twice :
  ∀ k : ℝ, ∃! (p1 p2 : ℝ × ℝ), p1 ≠ p2 ∧ 
    Circle p1.1 p1.2 ∧ Circle p2.1 p2.2 ∧
    Line k p1.1 p1.2 ∧ Line k p2.1 p2.2 :=
sorry

end line_intersects_circle_twice_l1661_166130


namespace two_point_questions_count_l1661_166172

/-- A test with two types of questions -/
structure Test where
  total_points : ℕ
  total_questions : ℕ
  two_point_questions : ℕ
  four_point_questions : ℕ

/-- The test satisfies the given conditions -/
def valid_test (t : Test) : Prop :=
  t.total_points = 100 ∧
  t.total_questions = 40 ∧
  t.two_point_questions + t.four_point_questions = t.total_questions ∧
  2 * t.two_point_questions + 4 * t.four_point_questions = t.total_points

theorem two_point_questions_count (t : Test) (h : valid_test t) :
  t.two_point_questions = 30 :=
by sorry

end two_point_questions_count_l1661_166172


namespace expression_evaluation_l1661_166196

theorem expression_evaluation :
  (4^1001 * 9^1002) / (6^1002 * 4^1000) = 3^1002 / 2^1000 := by
  sorry

end expression_evaluation_l1661_166196


namespace first_movie_length_proof_l1661_166173

/-- Represents the length of the first movie in hours -/
def first_movie_length : ℝ := 3.5

/-- Represents the length of the second movie in hours -/
def second_movie_length : ℝ := 1.5

/-- Represents the total available time in hours -/
def total_time : ℝ := 8

/-- Represents the reading rate in words per minute -/
def reading_rate : ℝ := 10

/-- Represents the total number of words read -/
def total_words_read : ℝ := 1800

/-- Proves that given the conditions, the length of the first movie must be 3.5 hours -/
theorem first_movie_length_proof :
  first_movie_length + second_movie_length + (total_words_read / reading_rate / 60) = total_time :=
by sorry

end first_movie_length_proof_l1661_166173


namespace sine_double_angle_special_l1661_166100

theorem sine_double_angle_special (α : Real) : 
  α ∈ Set.Ioo 0 (Real.pi / 2) → 
  Real.cos (α + Real.pi / 6) = 3 / 5 → 
  Real.sin (2 * α + Real.pi / 3) = 24 / 25 := by
  sorry

end sine_double_angle_special_l1661_166100


namespace linda_win_probability_is_two_thirty_first_l1661_166170

/-- Represents the outcome of a coin flip -/
inductive CoinFlip
| Heads
| Tails

/-- Represents a player in the game -/
inductive Player
| Sara
| Peter
| Linda

/-- The game state -/
structure GameState where
  currentPlayer : Player
  saraLastFlip : Option CoinFlip
  
/-- The result of a game round -/
inductive RoundResult
| Continue (newState : GameState)
| SaraWins
| LindaWins

/-- Simulates a single round of the game -/
def playRound (state : GameState) (flip : CoinFlip) : RoundResult := sorry

/-- Calculates the probability of Linda winning given the game rules -/
def lindaWinProbability : ℚ := sorry

/-- Theorem stating that the probability of Linda winning is 2/31 -/
theorem linda_win_probability_is_two_thirty_first :
  lindaWinProbability = 2 / 31 := by sorry

end linda_win_probability_is_two_thirty_first_l1661_166170


namespace square_sum_de_l1661_166119

theorem square_sum_de (a b c d e : ℕ+) 
  (eq1 : (a + 1) * (3 * b * c + 1) = d + 3 * e + 1)
  (eq2 : (b + 1) * (3 * c * a + 1) = 3 * d + e + 13)
  (eq3 : (c + 1) * (3 * a * b + 1) = 4 * (26 - d - e) - 1) :
  d ^ 2 + e ^ 2 = 146 := by
  sorry

end square_sum_de_l1661_166119


namespace lcm_gcd_ratio_240_360_l1661_166195

theorem lcm_gcd_ratio_240_360 : (lcm 240 360) / (gcd 240 360) = 6 := by sorry

end lcm_gcd_ratio_240_360_l1661_166195


namespace three_digit_numbers_divisible_by_17_l1661_166166

theorem three_digit_numbers_divisible_by_17 : 
  (Finset.filter (fun k => 100 ≤ 17 * k ∧ 17 * k ≤ 999) (Finset.range 1000)).card = 53 := by
  sorry

end three_digit_numbers_divisible_by_17_l1661_166166


namespace greatest_number_satisfying_conditions_l1661_166197

/-- A number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m ^ 2

/-- A number is composed of the square of two distinct prime factors -/
def is_product_of_two_distinct_prime_squares (n : ℕ) : Prop :=
  ∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ n = p^2 * q^2

/-- A number has an odd number of positive factors -/
def has_odd_number_of_factors (n : ℕ) : Prop :=
  Odd (Nat.card (Nat.divisors n))

/-- The main theorem -/
theorem greatest_number_satisfying_conditions : 
  (∀ n : ℕ, n < 200 → is_perfect_square n → 
    is_product_of_two_distinct_prime_squares n → 
    has_odd_number_of_factors n → n ≤ 196) ∧ 
  (196 < 200 ∧ is_perfect_square 196 ∧ 
    is_product_of_two_distinct_prime_squares 196 ∧ 
    has_odd_number_of_factors 196) := by
  sorry

end greatest_number_satisfying_conditions_l1661_166197


namespace sqrt_three_plus_two_power_l1661_166183

theorem sqrt_three_plus_two_power : (Real.sqrt 3 + Real.sqrt 2) ^ 2023 * (Real.sqrt 3 - Real.sqrt 2) ^ 2022 = Real.sqrt 3 + Real.sqrt 2 := by
  sorry

end sqrt_three_plus_two_power_l1661_166183


namespace tank_fill_time_l1661_166156

/-- Represents the state of the tank and pipes -/
structure TankSystem where
  pipeA : ℝ  -- Rate at which Pipe A fills the tank (fraction of tank per minute)
  pipeB : ℝ  -- Rate at which Pipe B empties the tank (fraction of tank per minute)
  closeBTime : ℝ  -- Time at which Pipe B is closed (in minutes)

/-- Calculates the time taken to fill the tank given the tank system parameters -/
def timeTakenToFill (system : TankSystem) : ℝ :=
  sorry

/-- Theorem stating that for the given conditions, the tank will be filled in 70 minutes -/
theorem tank_fill_time (system : TankSystem) 
  (hA : system.pipeA = 1 / 8)
  (hB : system.pipeB = 1 / 24)
  (hClose : system.closeBTime = 66) :
  timeTakenToFill system = 70 :=
sorry

end tank_fill_time_l1661_166156


namespace arithmetic_sequence_proof_l1661_166160

-- Define the arithmetic sequence an
def an (n : ℕ) : ℝ := 2 * 3^(n - 1)

-- Define the sequence bn
def bn (n : ℕ) : ℝ := an n - 2 * n

-- Define the sum of the first n terms of bn
def Tn (n : ℕ) : ℝ := 3^n - 1 - n^2 - n

theorem arithmetic_sequence_proof :
  (∀ n : ℕ, n ≥ 1 → an n = 2 * 3^(n - 1)) ∧
  (an 2 = 6) ∧
  (an 1 + an 2 + an 3 = 26) ∧
  (∀ n : ℕ, n ≥ 1 → Tn n = 3^n - 1 - n^2 - n) :=
by sorry

end arithmetic_sequence_proof_l1661_166160


namespace cost_of_dozen_pens_l1661_166101

/-- The cost of one dozen pens given the cost of one pen and the ratio of pen to pencil cost -/
theorem cost_of_dozen_pens 
  (cost_of_one_pen : ℕ) 
  (ratio_pen_to_pencil : ℚ) 
  (h1 : cost_of_one_pen = 65) 
  (h2 : ratio_pen_to_pencil = 5 / 1) : 
  12 * cost_of_one_pen = 780 := by
  sorry

end cost_of_dozen_pens_l1661_166101


namespace sin_minus_cos_with_tan_one_third_l1661_166186

theorem sin_minus_cos_with_tan_one_third 
  (θ : Real) 
  (h1 : θ ∈ Set.Ioo 0 (Real.pi / 2)) 
  (h2 : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 := by
  sorry

end sin_minus_cos_with_tan_one_third_l1661_166186


namespace monotonic_decreasing_interval_l1661_166199

-- Define the function
def f (x : ℝ) : ℝ := 2*x^3 - 6*x^2 - 18*x + 7

-- Define the derivative of the function
def f_derivative (x : ℝ) : ℝ := 6*x^2 - 12*x - 18

-- State the theorem
theorem monotonic_decreasing_interval :
  ∀ x : ℝ, (x > -1 ∧ x < 3) ↔ (f_derivative x < 0) :=
sorry

end monotonic_decreasing_interval_l1661_166199


namespace polynomial_divisibility_implies_specific_coefficients_l1661_166147

theorem polynomial_divisibility_implies_specific_coefficients :
  ∀ (p q : ℝ),
  (∀ x : ℝ, (x + 3) * (x - 2) ∣ (x^5 - x^4 + x^3 - p*x^2 + q*x + 9)) →
  p = -19.5 ∧ q = -55.5 := by
sorry

end polynomial_divisibility_implies_specific_coefficients_l1661_166147


namespace negation_of_existence_negation_of_inequality_l1661_166182

theorem negation_of_existence (p : ℝ → Prop) :
  (¬∃ x, p x) ↔ (∀ x, ¬p x) := by sorry

theorem negation_of_inequality :
  (¬∃ x : ℝ, Real.exp x - x - 1 ≤ 0) ↔ (∀ x : ℝ, Real.exp x - x - 1 > 0) := by sorry

end negation_of_existence_negation_of_inequality_l1661_166182


namespace trash_cans_veterans_park_l1661_166153

/-- The number of trash cans in Veteran's Park after the transfer -/
def final_trash_cans_veterans_park (initial_veterans_park : ℕ) (initial_central_park : ℕ) : ℕ :=
  initial_veterans_park + initial_central_park / 2

/-- Theorem stating the final number of trash cans in Veteran's Park -/
theorem trash_cans_veterans_park :
  ∃ (initial_central_park : ℕ),
    (initial_central_park = 24 / 2 + 8) ∧
    (final_trash_cans_veterans_park 24 initial_central_park = 34) := by
  sorry

#check trash_cans_veterans_park

end trash_cans_veterans_park_l1661_166153


namespace total_distance_walked_l1661_166164

/-- Calculates the total distance walked to various destinations in a school. -/
theorem total_distance_walked (water_fountain_dist : ℕ) (main_office_dist : ℕ) (teacher_lounge_dist : ℕ)
  (water_fountain_trips : ℕ) (main_office_trips : ℕ) (teacher_lounge_trips : ℕ)
  (h1 : water_fountain_dist = 30)
  (h2 : main_office_dist = 50)
  (h3 : teacher_lounge_dist = 35)
  (h4 : water_fountain_trips = 4)
  (h5 : main_office_trips = 2)
  (h6 : teacher_lounge_trips = 3) :
  water_fountain_dist * water_fountain_trips +
  main_office_dist * main_office_trips +
  teacher_lounge_dist * teacher_lounge_trips = 325 := by
  sorry

end total_distance_walked_l1661_166164


namespace normal_commute_time_l1661_166106

/-- A worker's commute scenario -/
structure WorkerCommute where
  normal_speed : ℝ
  normal_distance : ℝ
  normal_time : ℝ
  inclined_speed : ℝ
  inclined_distance : ℝ
  inclined_time : ℝ

/-- The conditions of the worker's commute -/
def commute_conditions (w : WorkerCommute) : Prop :=
  w.inclined_speed = 3 / 4 * w.normal_speed ∧
  w.inclined_distance = 5 / 4 * w.normal_distance ∧
  w.inclined_time = w.normal_time + 20 ∧
  w.normal_distance = w.normal_speed * w.normal_time ∧
  w.inclined_distance = w.inclined_speed * w.inclined_time

/-- The theorem stating that under the given conditions, the normal commute time is 30 minutes -/
theorem normal_commute_time (w : WorkerCommute) 
  (h : commute_conditions w) : w.normal_time = 30 := by
  sorry

end normal_commute_time_l1661_166106


namespace largest_common_value_l1661_166189

/-- The first arithmetic progression -/
def progression1 (n : ℕ) : ℕ := 4 + 5 * n

/-- The second arithmetic progression -/
def progression2 (n : ℕ) : ℕ := 5 + 9 * n

/-- A common term of both progressions -/
def commonTerm (m : ℕ) : ℕ := 14 + 45 * m

theorem largest_common_value :
  (∃ n1 n2 : ℕ, progression1 n1 = 959 ∧ progression2 n2 = 959) ∧ 
  (∀ k : ℕ, k < 1000 → k > 959 → 
    (∀ n1 n2 : ℕ, progression1 n1 ≠ k ∨ progression2 n2 ≠ k)) :=
sorry

end largest_common_value_l1661_166189


namespace power_of_power_l1661_166154

theorem power_of_power (a : ℝ) : (a^5)^2 = a^10 := by
  sorry

end power_of_power_l1661_166154


namespace millet_percentage_in_brand_A_l1661_166150

/-- The percentage of millet in Brand A -/
def millet_in_A : ℝ := 0.4

/-- The percentage of sunflower in Brand A -/
def sunflower_in_A : ℝ := 0.6

/-- The percentage of millet in Brand B -/
def millet_in_B : ℝ := 0.65

/-- The percentage of Brand A in the mix -/
def brand_A_in_mix : ℝ := 0.6

/-- The percentage of Brand B in the mix -/
def brand_B_in_mix : ℝ := 0.4

/-- The percentage of millet in the mix -/
def millet_in_mix : ℝ := 0.5

theorem millet_percentage_in_brand_A :
  millet_in_A * brand_A_in_mix + millet_in_B * brand_B_in_mix = millet_in_mix ∧
  millet_in_A + sunflower_in_A = 1 :=
by sorry

end millet_percentage_in_brand_A_l1661_166150


namespace age_difference_l1661_166146

/-- Given that the total age of A and B is 13 years more than the total age of B and C,
    prove that C is 13 years younger than A. -/
theorem age_difference (A B C : ℕ) (h : A + B = B + C + 13) : A = C + 13 := by
  sorry

end age_difference_l1661_166146


namespace sculpture_and_base_height_l1661_166109

/-- Converts feet to inches -/
def feetToInches (feet : ℕ) : ℕ := feet * 12

/-- Represents the height of an object in feet and inches -/
structure Height where
  feet : ℕ
  inches : ℕ

/-- Converts a Height to total inches -/
def heightToInches (h : Height) : ℕ := feetToInches h.feet + h.inches

/-- Calculates the total height when placing an object on a base -/
def totalHeight (objectHeight : Height) (baseHeight : ℕ) : ℕ :=
  heightToInches objectHeight + baseHeight

theorem sculpture_and_base_height :
  let sculptureHeight : Height := { feet := 2, inches := 10 }
  let baseHeight : ℕ := 4
  totalHeight sculptureHeight baseHeight = 38 := by sorry

end sculpture_and_base_height_l1661_166109


namespace binomial_coefficient_8_4_l1661_166108

theorem binomial_coefficient_8_4 : Nat.choose 8 4 = 70 := by
  sorry

end binomial_coefficient_8_4_l1661_166108


namespace triangle_side_length_range_l1661_166174

theorem triangle_side_length_range (b : ℝ) (B : ℝ) :
  b = 2 →
  B = π / 3 →
  ∃ (a : ℝ), 2 < a ∧ a < 4 * Real.sqrt 3 / 3 :=
by sorry

end triangle_side_length_range_l1661_166174


namespace replacement_theorem_l1661_166118

/-- Calculates the percentage of chemicals in a solution after replacing part of it with a different solution -/
def resulting_solution_percentage (original_percentage : ℝ) (replacement_percentage : ℝ) (replaced_portion : ℝ) : ℝ :=
  let remaining_portion := 1 - replaced_portion
  let original_chemicals := original_percentage * remaining_portion
  let replacement_chemicals := replacement_percentage * replaced_portion
  (original_chemicals + replacement_chemicals) * 100

/-- Theorem stating that replacing half of an 80% solution with a 20% solution results in a 50% solution -/
theorem replacement_theorem :
  resulting_solution_percentage 0.8 0.2 0.5 = 50 := by
  sorry

end replacement_theorem_l1661_166118


namespace slope_angle_range_l1661_166139

noncomputable def slope_angle (α : Real) : Prop :=
  ∃ (x : Real), x ≠ 0 ∧ Real.tan α = (1/2) * (x + 1/x)

theorem slope_angle_range :
  ∀ α, slope_angle α → 
    (α ∈ Set.Icc (π/4) (π/2) ∪ Set.Ioc (π/2) (3*π/4)) := by
  sorry

end slope_angle_range_l1661_166139


namespace intersection_probability_formula_l1661_166192

/-- The number of points evenly spaced around the circle -/
def n : ℕ := 2023

/-- The probability of selecting six distinct points A, B, C, D, E, F from n evenly spaced points 
    on a circle, such that chord AB intersects chord CD but neither intersects chord EF -/
def intersection_probability : ℚ :=
  2 * (Nat.choose (n / 2) 2) / Nat.choose n 6

/-- Theorem stating the probability calculation -/
theorem intersection_probability_formula : 
  intersection_probability = 2 * (Nat.choose (n / 2) 2) / Nat.choose n 6 := by
  sorry

end intersection_probability_formula_l1661_166192


namespace quadratic_function_property_l1661_166104

/-- A quadratic function f(x) = x^2 + bx + c with f(1) = 0 and f(3) = 0 satisfies f(-1) = 8 -/
theorem quadratic_function_property (b c : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = x^2 + b*x + c) 
  (h2 : f 1 = 0) 
  (h3 : f 3 = 0) : 
  f (-1) = 8 := by
  sorry

end quadratic_function_property_l1661_166104


namespace range_of_f_l1661_166132

/-- A monotonically increasing odd function f with f(1) = 2 and f(2) = 3 -/
def f : ℝ → ℝ :=
  sorry

/-- f is monotonically increasing -/
axiom f_increasing (x y : ℝ) : x < y → f x < f y

/-- f is an odd function -/
axiom f_odd (x : ℝ) : f (-x) = -f x

/-- f(1) = 2 -/
axiom f_1 : f 1 = 2

/-- f(2) = 3 -/
axiom f_2 : f 2 = 3

/-- The main theorem -/
theorem range_of_f (x : ℝ) : 
  (-3 < f (x - 3) ∧ f (x - 3) < 2) ↔ (1 < x ∧ x < 4) :=
sorry

end range_of_f_l1661_166132


namespace bird_cost_problem_l1661_166155

/-- The cost of birds in a pet store -/
theorem bird_cost_problem (small_bird_cost big_bird_cost : ℚ) : 
  big_bird_cost = 2 * small_bird_cost →
  5 * big_bird_cost + 3 * small_bird_cost = 5 * small_bird_cost + 3 * big_bird_cost + 20 →
  small_bird_cost = 10 ∧ big_bird_cost = 20 := by
  sorry

end bird_cost_problem_l1661_166155


namespace even_function_alpha_beta_values_l1661_166137

theorem even_function_alpha_beta_values (α β : Real) :
  let f : Real → Real := λ x => 
    if x < 0 then Real.sin (x + α) else Real.cos (x + β)
  (∀ x, f (-x) = f x) →
  α = π / 3 ∧ β = π / 6 :=
by sorry

end even_function_alpha_beta_values_l1661_166137


namespace ball_radius_l1661_166168

theorem ball_radius (hole_diameter : ℝ) (hole_depth : ℝ) (ball_radius : ℝ) : 
  hole_diameter = 30 ∧ hole_depth = 10 → ball_radius = 16.25 := by
  sorry

end ball_radius_l1661_166168


namespace newton_family_mean_age_l1661_166190

theorem newton_family_mean_age :
  let ages : List ℝ := [6, 6, 9, 12]
  let mean := (ages.sum) / (ages.length)
  mean = 8.25 := by
sorry

end newton_family_mean_age_l1661_166190


namespace total_jump_rope_time_l1661_166159

/-- The total jump rope time for four girls given their relative jump times -/
theorem total_jump_rope_time (cindy betsy tina sarah : ℕ) : 
  cindy = 12 →
  betsy = cindy / 2 →
  tina = betsy * 3 →
  sarah = cindy + tina →
  cindy + betsy + tina + sarah = 66 := by
  sorry

end total_jump_rope_time_l1661_166159


namespace max_value_of_f_l1661_166151

/-- Definition of the sum of the first n terms of the geometric sequence -/
def S (n : ℕ) (k : ℝ) : ℝ := 2^(n-1) + k

/-- Definition of the function f -/
def f (k : ℝ) (x : ℝ) : ℝ := x^3 - k*x^2 - 2*x + 1

/-- Theorem stating the maximum value of f -/
theorem max_value_of_f (k : ℝ) : 
  (∃ (n : ℕ), ∀ (m : ℕ), S m k = 2^(m-1) + k) → 
  (∃ (x : ℝ), ∀ (y : ℝ), f k y ≤ f k x) ∧ 
  (∃ (x : ℝ), f k x = 5/2) :=
sorry

end max_value_of_f_l1661_166151


namespace pentagon_3010th_position_l1661_166102

/-- Represents the possible positions of the pentagon --/
inductive PentagonPosition
  | ABCDE
  | EABCD
  | DCBAE
  | EDABC

/-- Represents the operations that can be performed on the pentagon --/
inductive Operation
  | Rotate
  | Reflect

/-- Applies an operation to a pentagon position --/
def applyOperation (pos : PentagonPosition) (op : Operation) : PentagonPosition :=
  match pos, op with
  | PentagonPosition.ABCDE, Operation.Rotate => PentagonPosition.EABCD
  | PentagonPosition.EABCD, Operation.Reflect => PentagonPosition.DCBAE
  | PentagonPosition.DCBAE, Operation.Rotate => PentagonPosition.EDABC
  | PentagonPosition.EDABC, Operation.Reflect => PentagonPosition.ABCDE
  | _, _ => pos  -- Default case to satisfy exhaustiveness

/-- Applies a sequence of alternating rotate and reflect operations --/
def applySequence (n : Nat) : PentagonPosition :=
  match n % 4 with
  | 0 => PentagonPosition.ABCDE
  | 1 => PentagonPosition.EABCD
  | 2 => PentagonPosition.DCBAE
  | _ => PentagonPosition.EDABC

theorem pentagon_3010th_position :
  applySequence 3010 = PentagonPosition.ABCDE :=
sorry


end pentagon_3010th_position_l1661_166102


namespace gcd_1043_2295_l1661_166113

theorem gcd_1043_2295 : Nat.gcd 1043 2295 = 1 := by
  sorry

end gcd_1043_2295_l1661_166113


namespace sum_of_digits_squared_difference_l1661_166198

def x : ℕ := 777777777777777
def y : ℕ := 222222222222223

def digit_sum (n : ℕ) : ℕ :=
  if n = 0 then 0 else n % 10 + digit_sum (n / 10)

theorem sum_of_digits_squared_difference : 
  digit_sum ((x^2 : ℕ) - (y^2 : ℕ)) = 74 := by sorry

end sum_of_digits_squared_difference_l1661_166198


namespace f_neg_one_eq_zero_f_is_even_x_range_l1661_166185

noncomputable section

variable (f : ℝ → ℝ)

-- Define the functional equation
axiom functional_eq : ∀ (x₁ x₂ : ℝ), x₁ ≠ 0 → x₂ ≠ 0 → f (x₁ * x₂) = f x₁ + f x₂

-- Define that f is increasing on (0, +∞)
axiom f_increasing : ∀ (x y : ℝ), 0 < x → x < y → f x < f y

-- Define the inequality condition
axiom f_inequality : ∀ (x : ℝ), f (2 * x - 1) < f x

-- Theorem 1: f(-1) = 0
theorem f_neg_one_eq_zero : f (-1) = 0 := by sorry

-- Theorem 2: f is an even function
theorem f_is_even : ∀ (x : ℝ), f (-x) = f x := by sorry

-- Theorem 3: Range of x
theorem x_range : ∀ (x : ℝ), (1/3 < x ∧ x < 1) ↔ (f (2*x - 1) < f x ∧ ∀ (y z : ℝ), 0 < y → y < z → f y < f z) := by sorry

end f_neg_one_eq_zero_f_is_even_x_range_l1661_166185


namespace find_m_l1661_166178

theorem find_m : ∃ m : ℕ, (1/5 : ℚ)^m * (1/4 : ℚ)^2 = 1/(10^4 : ℚ) ∧ m = 4 := by
  sorry

end find_m_l1661_166178


namespace exam_items_count_l1661_166193

theorem exam_items_count :
  ∀ (total_items : ℕ) (liza_correct : ℕ) (rose_correct : ℕ) (rose_incorrect : ℕ),
    liza_correct = (90 * total_items) / 100 →
    rose_correct = liza_correct + 2 →
    rose_incorrect = 4 →
    total_items = rose_correct + rose_incorrect →
    total_items = 60 := by
  sorry

end exam_items_count_l1661_166193


namespace no_integer_satisfies_conditions_l1661_166167

/-- Sum of digits function -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Theorem stating that no positive integer A satisfies the given conditions -/
theorem no_integer_satisfies_conditions : ¬ ∃ A : ℕ+, 
  (sumOfDigits A = 16) ∧ (sumOfDigits (2 * A) = 17) := by sorry

end no_integer_satisfies_conditions_l1661_166167


namespace six_distinct_one_repeat_probability_l1661_166133

/-- The number of sides on a standard die -/
def num_sides : ℕ := 6

/-- The number of dice rolled -/
def num_dice : ℕ := 7

/-- The probability of rolling seven standard six-sided dice and getting exactly six distinct numbers, with one number repeating once -/
theorem six_distinct_one_repeat_probability : 
  (num_sides.choose 1 * (num_sides - 1).factorial * num_dice.choose 2) / num_sides ^ num_dice = 5 / 186 := by
  sorry

end six_distinct_one_repeat_probability_l1661_166133


namespace smallest_number_with_properties_l1661_166105

def ends_with_6 (n : ℕ) : Prop := n % 10 = 6

def move_6_to_front (n : ℕ) : ℕ :=
  let d := (Nat.log 10 n) + 1
  6 * 10^d + n / 10

theorem smallest_number_with_properties : ℕ := by
  let n := 1538466
  have h1 : ends_with_6 n := by sorry
  have h2 : move_6_to_front n = 4 * n := by sorry
  have h3 : ∀ m < n, ¬(ends_with_6 m ∧ move_6_to_front m = 4 * m) := by sorry
  exact n

end smallest_number_with_properties_l1661_166105


namespace train_passing_time_l1661_166142

/-- Calculates the time for two trains to clear each other --/
theorem train_passing_time (length1 length2 speed1 speed2 : ℝ) 
  (h1 : length1 = 160)
  (h2 : length2 = 280)
  (h3 : speed1 = 42)
  (h4 : speed2 = 30) : 
  (length1 + length2) / ((speed1 + speed2) * (1000 / 3600)) = 22 := by
  sorry

#check train_passing_time

end train_passing_time_l1661_166142


namespace derivative_equals_one_l1661_166140

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 2
  else 2^x

-- Define the derivative of f
noncomputable def f_derivative (x : ℝ) : ℝ :=
  if x > 0 then 1 / (x * Real.log 2)
  else 2^x * Real.log 2

-- Theorem statement
theorem derivative_equals_one (a : ℝ) :
  f_derivative a = 1 ↔ a = 1 / Real.log 2 :=
sorry

end derivative_equals_one_l1661_166140


namespace number_decrease_divide_l1661_166127

theorem number_decrease_divide (x : ℚ) : (x - 4) / 10 = 5 → (x - 5) / 7 = 7 := by
  sorry

end number_decrease_divide_l1661_166127


namespace product_correction_l1661_166122

/-- Reverses the digits of a two-digit number -/
def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

/-- The problem statement -/
theorem product_correction (a b : ℕ) : 
  10 ≤ a ∧ a < 100 →  -- a is a two-digit number
  a > 0 →  -- a is positive
  b > 0 →  -- b is positive
  reverse_digits a * b = 284 →
  a * b = 68 := by
sorry

end product_correction_l1661_166122


namespace cube_color_probability_l1661_166136

def cube_face_colors := Fin 3
def num_faces : Nat := 6

-- Probability of each color
def color_prob : ℚ := 1 / 3

-- Total number of possible color arrangements
def total_arrangements : Nat := 3^num_faces

-- Number of arrangements where all faces are the same color
def all_same_color : Nat := 3

-- Number of arrangements where 5 faces are the same color and 1 is different
def five_same_one_different : Nat := 3 * 6 * 2

-- Number of arrangements where 4 faces are the same color and opposite faces are different
def four_same_opposite_different : Nat := 3 * 3 * 6

-- Total number of suitable arrangements
def suitable_arrangements : Nat := all_same_color + five_same_one_different + four_same_opposite_different

-- Probability of suitable arrangements
def prob_suitable_arrangements : ℚ := suitable_arrangements / total_arrangements

theorem cube_color_probability :
  prob_suitable_arrangements = 31 / 243 :=
sorry

end cube_color_probability_l1661_166136


namespace quadratic_sum_zero_l1661_166179

/-- A quadratic function f(x) = ax^2 + bx + c -/
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_sum_zero 
  (a b c : ℝ) 
  (h1 : quadratic a b c 1 = 0)
  (h2 : quadratic a b c 5 = 0)
  (h3 : ∃ (k : ℝ), ∀ (x : ℝ), quadratic a b c x ≥ 36 ∧ quadratic a b c k = 36) :
  a + b + c = 0 := by
  sorry

end quadratic_sum_zero_l1661_166179


namespace rental_cost_difference_theorem_l1661_166115

/-- Calculates the rental cost difference between a ski boat and a sailboat --/
def rental_cost_difference (
  sailboat_weekday_cost : ℕ)
  (skiboat_weekend_hourly_cost : ℕ)
  (sailboat_fuel_cost_per_hour : ℕ)
  (skiboat_fuel_cost_per_hour : ℕ)
  (rental_hours_per_day : ℕ)
  (rental_days : ℕ)
  (discount_percentage : ℕ) : ℕ :=
  let sailboat_day1_cost := sailboat_weekday_cost + sailboat_fuel_cost_per_hour * rental_hours_per_day
  let sailboat_day2_cost := (sailboat_weekday_cost * (100 - discount_percentage) / 100) + sailboat_fuel_cost_per_hour * rental_hours_per_day
  let sailboat_total_cost := sailboat_day1_cost + sailboat_day2_cost

  let skiboat_day1_cost := skiboat_weekend_hourly_cost * rental_hours_per_day + skiboat_fuel_cost_per_hour * rental_hours_per_day
  let skiboat_day2_cost := (skiboat_weekend_hourly_cost * rental_hours_per_day * (100 - discount_percentage) / 100) + skiboat_fuel_cost_per_hour * rental_hours_per_day
  let skiboat_total_cost := skiboat_day1_cost + skiboat_day2_cost

  skiboat_total_cost - sailboat_total_cost

theorem rental_cost_difference_theorem :
  rental_cost_difference 60 120 10 20 3 2 10 = 630 := by
  sorry

end rental_cost_difference_theorem_l1661_166115


namespace intersection_with_complement_l1661_166126

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 3}
def B : Set ℕ := {2, 3}

theorem intersection_with_complement : A ∩ (U \ B) = {1} := by
  sorry

end intersection_with_complement_l1661_166126


namespace team_handedness_ratio_l1661_166123

/-- A ball team with right-handed and left-handed players -/
structure BallTeam where
  right_handed : ℕ
  left_handed : ℕ

/-- Represents the attendance at practice -/
structure PracticeAttendance (team : BallTeam) where
  present_right : ℕ
  present_left : ℕ
  absent_right : ℕ
  absent_left : ℕ
  total_present : present_right + present_left = team.right_handed + team.left_handed - (absent_right + absent_left)
  all_accounted : present_right + absent_right = team.right_handed
  all_accounted_left : present_left + absent_left = team.left_handed

/-- The theorem representing the problem -/
theorem team_handedness_ratio (team : BallTeam) (attendance : PracticeAttendance team) :
  (2 : ℚ) / 3 * (team.right_handed + team.left_handed) = attendance.absent_right + attendance.absent_left →
  (2 : ℚ) / 3 * (attendance.present_right + attendance.present_left) = attendance.present_left →
  (attendance.absent_right : ℚ) / attendance.absent_left = 14 / 10 →
  (team.right_handed : ℚ) / team.left_handed = 14 / 10 := by
  sorry

end team_handedness_ratio_l1661_166123


namespace remainder_11_pow_101_mod_7_l1661_166163

theorem remainder_11_pow_101_mod_7 : 11^101 % 7 = 2 := by
  sorry

end remainder_11_pow_101_mod_7_l1661_166163

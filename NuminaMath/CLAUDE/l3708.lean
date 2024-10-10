import Mathlib

namespace square_sum_of_special_integers_l3708_370855

theorem square_sum_of_special_integers (x y : ℕ+) 
  (h1 : x * y + x + y = 119)
  (h2 : x^2 * y + x * y^2 = 1870)
  (h3 : x < y) :
  x^2 + y^2 = 986 := by
  sorry

end square_sum_of_special_integers_l3708_370855


namespace chicken_coops_count_l3708_370890

theorem chicken_coops_count (chickens_per_coop : ℕ) (total_chickens : ℕ) 
  (h1 : chickens_per_coop = 60) 
  (h2 : total_chickens = 540) : 
  total_chickens / chickens_per_coop = 9 := by
  sorry

end chicken_coops_count_l3708_370890


namespace perpendicular_line_through_point_l3708_370820

/-- Given a line L1 with equation x - 2y + 3 = 0, prove that the line L2 with equation 2x + y - 1 = 0
    passes through the point (-1, 3) and is perpendicular to L1. -/
theorem perpendicular_line_through_point :
  let L1 : ℝ × ℝ → Prop := fun (x, y) ↦ x - 2*y + 3 = 0
  let L2 : ℝ × ℝ → Prop := fun (x, y) ↦ 2*x + y - 1 = 0
  let point : ℝ × ℝ := (-1, 3)
  (L2 point) ∧ 
  (∀ (p q : ℝ × ℝ), L1 p ∧ L1 q ∧ p ≠ q → 
    let v1 := (p.1 - q.1, p.2 - q.2)
    let v2 := (1, 2)
    v1.1 * v2.1 + v1.2 * v2.2 = 0) :=
by
  sorry

end perpendicular_line_through_point_l3708_370820


namespace train_distance_l3708_370807

/-- Represents the efficiency of a coal-powered train in miles per pound of coal -/
def train_efficiency : ℚ := 5 / 2

/-- Represents the amount of coal remaining in pounds -/
def coal_remaining : ℕ := 160

/-- Calculates the distance a train can travel given its efficiency and remaining coal -/
def distance_traveled (efficiency : ℚ) (coal : ℕ) : ℚ :=
  efficiency * coal

/-- Theorem stating that the train can travel 400 miles before running out of fuel -/
theorem train_distance : distance_traveled train_efficiency coal_remaining = 400 := by
  sorry

end train_distance_l3708_370807


namespace triangle_max_area_l3708_370825

theorem triangle_max_area (a b c : ℝ) :
  a > 0 → b > 0 → c > 0 →
  3 * a * b = 25 - c^2 →
  let angle_C := 60 * π / 180
  let area := (1 / 2) * a * b * Real.sin angle_C
  area ≤ 25 * Real.sqrt 3 / 16 :=
by sorry

end triangle_max_area_l3708_370825


namespace line_intercepts_sum_l3708_370877

/-- Given a line equation 3x + 5y + d = 0, proves that if the sum of x- and y-intercepts is 16, then d = -30 -/
theorem line_intercepts_sum (d : ℝ) : 
  (∃ x y : ℝ, 3 * x + 5 * y + d = 0 ∧ x + y = 16) → d = -30 :=
by sorry

end line_intercepts_sum_l3708_370877


namespace max_x_on_circle_max_x_achieved_l3708_370880

theorem max_x_on_circle (x y : ℝ) (h : x^2 + y^2 = 18*x + 20*y) :
  x ≤ 9 + Real.sqrt 181 :=
by sorry

theorem max_x_achieved (ε : ℝ) (hε : ε > 0) :
  ∃ (x y : ℝ), x^2 + y^2 = 18*x + 20*y ∧ x > 9 + Real.sqrt 181 - ε :=
by sorry

end max_x_on_circle_max_x_achieved_l3708_370880


namespace recycle_388_cans_l3708_370883

/-- Recursively calculate the number of new cans produced from recycling -/
def recycle_cans (initial_cans : ℕ) : ℕ :=
  if initial_cans < 6 then 0
  else
    let new_cans := (2 * initial_cans) / 6
    new_cans + recycle_cans new_cans

/-- The total number of new cans produced from 388 initial cans -/
def total_new_cans : ℕ := recycle_cans 388

/-- Theorem stating that 193 new cans are produced from 388 initial cans -/
theorem recycle_388_cans : total_new_cans = 193 := by
  sorry

end recycle_388_cans_l3708_370883


namespace max_threes_in_selection_l3708_370887

/-- Represents the count of each card type in the selection -/
structure CardSelection :=
  (threes : ℕ)
  (fours : ℕ)
  (fives : ℕ)

/-- The problem constraints -/
def isValidSelection (s : CardSelection) : Prop :=
  s.threes + s.fours + s.fives = 8 ∧
  3 * s.threes + 4 * s.fours + 5 * s.fives = 27 ∧
  s.threes ≤ 10 ∧ s.fours ≤ 10 ∧ s.fives ≤ 10

/-- The theorem statement -/
theorem max_threes_in_selection :
  ∃ (s : CardSelection), isValidSelection s ∧
    (∀ (t : CardSelection), isValidSelection t → t.threes ≤ s.threes) ∧
    s.threes = 6 :=
sorry

end max_threes_in_selection_l3708_370887


namespace product_of_solutions_l3708_370819

theorem product_of_solutions (x : ℝ) : 
  (12 = 2 * x^2 + 4 * x) → 
  (∃ x₁ x₂ : ℝ, (12 = 2 * x₁^2 + 4 * x₁) ∧ (12 = 2 * x₂^2 + 4 * x₂) ∧ (x₁ * x₂ = -6)) :=
by sorry

end product_of_solutions_l3708_370819


namespace cos_sin_sum_equals_zero_l3708_370804

theorem cos_sin_sum_equals_zero :
  Real.cos (5 * Real.pi / 8) * Real.cos (Real.pi / 8) + 
  Real.sin (5 * Real.pi / 8) * Real.sin (Real.pi / 8) = 0 := by
  sorry

end cos_sin_sum_equals_zero_l3708_370804


namespace kevin_born_1984_l3708_370852

/-- The year of the first AMC 8 competition -/
def first_amc8_year : ℕ := 1988

/-- The year Kevin took the AMC 8 -/
def kevins_amc8_year : ℕ := first_amc8_year + 9

/-- Kevin's age when he took the AMC 8 -/
def kevins_age : ℕ := 13

/-- Kevin's birth year -/
def kevins_birth_year : ℕ := kevins_amc8_year - kevins_age

theorem kevin_born_1984 : kevins_birth_year = 1984 := by
  sorry

end kevin_born_1984_l3708_370852


namespace inner_triangle_perimeter_l3708_370800

theorem inner_triangle_perimeter (a : ℝ) (h : a = 8) :
  let outer_leg := a
  let inner_leg := a - 1
  let inner_hypotenuse := inner_leg * Real.sqrt 2
  let inner_perimeter := 2 * inner_leg + inner_hypotenuse
  inner_perimeter = 14 + 7 * Real.sqrt 2 := by
  sorry

end inner_triangle_perimeter_l3708_370800


namespace triangle_angle_proof_l3708_370823

theorem triangle_angle_proof (a b c : ℝ) : 
  a = 60 → b = 40 → a + b + c = 180 → c = 80 :=
by
  sorry

end triangle_angle_proof_l3708_370823


namespace sine_plus_abs_sine_integral_l3708_370861

open Set
open MeasureTheory
open Real

theorem sine_plus_abs_sine_integral : 
  ∫ x in (-π/2)..(π/2), (sin x + |sin x|) = 2 := by sorry

end sine_plus_abs_sine_integral_l3708_370861


namespace hat_cost_calculation_l3708_370815

/-- The price of a wooden toy -/
def wooden_toy_price : ℕ := 20

/-- The number of wooden toys Kendra bought -/
def wooden_toys_bought : ℕ := 2

/-- The number of hats Kendra bought -/
def hats_bought : ℕ := 3

/-- The amount Kendra paid with -/
def amount_paid : ℕ := 100

/-- The change Kendra received -/
def change_received : ℕ := 30

/-- The cost of a hat -/
def hat_cost : ℕ := 10

theorem hat_cost_calculation :
  hat_cost = (amount_paid - change_received - wooden_toy_price * wooden_toys_bought) / hats_bought :=
by sorry

end hat_cost_calculation_l3708_370815


namespace union_of_sets_l3708_370816

open Set

theorem union_of_sets (U A B : Set ℕ) : 
  U = {1, 2, 3, 4, 5, 6} →
  (U \ A) = {1, 2, 4} →
  (U \ B) = {3, 4, 5} →
  A ∪ B = {1, 2, 3, 5, 6} := by
  sorry

end union_of_sets_l3708_370816


namespace car_rental_cost_l3708_370810

/-- Calculates the car rental cost for a vacation given the number of people,
    Airbnb rental cost, and each person's share. -/
theorem car_rental_cost (num_people : ℕ) (airbnb_cost : ℕ) (person_share : ℕ) : 
  num_people = 8 → airbnb_cost = 3200 → person_share = 500 →
  num_people * person_share - airbnb_cost = 800 := by
sorry

end car_rental_cost_l3708_370810


namespace emily_egg_collection_l3708_370841

/-- The number of baskets Emily used --/
def num_baskets : ℕ := 1525

/-- The average number of eggs per basket --/
def eggs_per_basket : ℚ := 37.5

/-- The total number of eggs collected --/
def total_eggs : ℚ := num_baskets * eggs_per_basket

/-- Theorem stating that the total number of eggs is 57,187.5 --/
theorem emily_egg_collection :
  total_eggs = 57187.5 := by sorry

end emily_egg_collection_l3708_370841


namespace problem_statement_l3708_370857

theorem problem_statement (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h1 : a ^ b = b ^ a) (h2 : b = 9 * a) : a = (3 : ℝ) ^ (1/4) := by
  sorry

end problem_statement_l3708_370857


namespace first_group_size_l3708_370836

/-- The number of men in the first group -/
def M : ℕ := sorry

/-- The length of the wall built by the first group -/
def length1 : ℝ := 66

/-- The number of days taken by the first group -/
def days1 : ℕ := 8

/-- The number of men in the second group -/
def men2 : ℕ := 86

/-- The length of the wall built by the second group -/
def length2 : ℝ := 283.8

/-- The number of days taken by the second group -/
def days2 : ℕ := 8

/-- The work done is directly proportional to the number of men and the length of the wall -/
axiom work_proportional : ∀ (men : ℕ) (length : ℝ) (days : ℕ), 
  (men : ℝ) * length / days = (M : ℝ) * length1 / days1

theorem first_group_size : 
  ∃ (m : ℕ), (m : ℝ) ≥ 368.5 ∧ (m : ℝ) < 369.5 ∧ M = m :=
sorry

end first_group_size_l3708_370836


namespace least_integer_satisfying_inequality_l3708_370801

theorem least_integer_satisfying_inequality :
  ∀ x : ℤ, (3 * |x| - 2 > 13) → x ≥ -6 :=
by
  sorry

end least_integer_satisfying_inequality_l3708_370801


namespace arrangements_eq_24_l3708_370834

/-- The number of letter cards -/
def n : ℕ := 6

/-- The number of cards that can be freely arranged -/
def k : ℕ := n - 2

/-- The number of different arrangements of n letter cards where two cards are fixed at the ends -/
def num_arrangements (n : ℕ) : ℕ := Nat.factorial (n - 2)

/-- Theorem stating that the number of arrangements is 24 -/
theorem arrangements_eq_24 : num_arrangements n = 24 := by
  sorry

end arrangements_eq_24_l3708_370834


namespace multiplicative_inverse_123_mod_456_l3708_370837

theorem multiplicative_inverse_123_mod_456 :
  ∃ (x : ℕ), x < 456 ∧ (123 * x) % 456 = 1 :=
by
  use 52
  sorry

end multiplicative_inverse_123_mod_456_l3708_370837


namespace beginner_trig_probability_probability_calculation_l3708_370803

/-- Represents the number of students in each course -/
structure CourseEnrollment where
  BC : ℕ  -- Beginner Calculus
  AC : ℕ  -- Advanced Calculus
  IC : ℕ  -- Intermediate Calculus
  BT : ℕ  -- Beginner Trigonometry
  AT : ℕ  -- Advanced Trigonometry
  IT : ℕ  -- Intermediate Trigonometry

/-- Represents the enrollment conditions for the math department -/
def EnrollmentConditions (e : CourseEnrollment) (total : ℕ) : Prop :=
  e.BC + e.AC + e.IC = (60 * total) / 100 ∧
  e.BT + e.AT + e.IT = (40 * total) / 100 ∧
  e.BC + e.BT = (45 * total) / 100 ∧
  e.AC + e.AT = (35 * total) / 100 ∧
  e.IC + e.IT = (20 * total) / 100 ∧
  e.BC = (125 * e.BT) / 100 ∧
  e.IC + e.AC = (120 * (e.IT + e.AT)) / 100

theorem beginner_trig_probability (e : CourseEnrollment) (total : ℕ) :
  EnrollmentConditions e total → total = 5000 → e.BT = 1000 :=
by sorry

theorem probability_calculation (e : CourseEnrollment) (total : ℕ) :
  EnrollmentConditions e total → total = 5000 → e.BT = 1000 →
  (e.BT : ℚ) / total = 1/5 :=
by sorry

end beginner_trig_probability_probability_calculation_l3708_370803


namespace cos_equation_rational_solution_l3708_370832

theorem cos_equation_rational_solution (a : ℚ) 
  (h1 : 0 < a) (h2 : a < 1) 
  (h3 : Real.cos (3 * Real.pi * a) + 2 * Real.cos (2 * Real.pi * a) = 0) : 
  a = 2/3 := by
sorry

end cos_equation_rational_solution_l3708_370832


namespace point_in_region_t_range_l3708_370856

/-- Given a point (1, t) in the region represented by x - y + 1 > 0, 
    the range of values for t is t < 2 -/
theorem point_in_region_t_range (t : ℝ) : 
  (1 : ℝ) - t + 1 > 0 → t < 2 := by
  sorry

end point_in_region_t_range_l3708_370856


namespace proposition_analysis_l3708_370814

theorem proposition_analysis (a b c : ℝ) : 
  (∀ x y z : ℝ, (x ≤ y → x*z^2 ≤ y*z^2)) ∧ 
  (∃ x y z : ℝ, (x > y ∧ x*z^2 ≤ y*z^2)) ∧
  (∀ x y z : ℝ, (x*z^2 > y*z^2 → x > y)) :=
by sorry

end proposition_analysis_l3708_370814


namespace P_equals_Q_l3708_370842

-- Define a one-to-one, strictly increasing function f: R → R
def f : ℝ → ℝ := sorry

-- Define the properties of f
axiom f_injective : Function.Injective f
axiom f_strictly_increasing : ∀ x y, x < y → f x < f y

-- Define the sets P and Q
def P : Set ℝ := {x | x > f x}
def Q : Set ℝ := {x | x > f (f x)}

-- State the theorem
theorem P_equals_Q : P = Q := by sorry

end P_equals_Q_l3708_370842


namespace range_of_t_l3708_370885

-- Define set A
def A : Set ℝ := {x | (1/4 : ℝ) ≤ 2^x ∧ 2^x ≤ (1/2 : ℝ)}

-- Define set B (parameterized by t)
def B (t : ℝ) : Set ℝ := {x | x^2 - 2*t*x + 1 ≤ 0}

-- Theorem statement
theorem range_of_t (t : ℝ) : 
  (A ∩ B t = A) ↔ t ∈ Set.Iic (-5/4 : ℝ) := by sorry


end range_of_t_l3708_370885


namespace acute_triangle_from_sides_l3708_370813

theorem acute_triangle_from_sides (a b c : ℝ) (ha : a = 5) (hb : b = 6) (hc : c = 7) :
  a + b > c ∧ b + c > a ∧ c + a > b ∧ a^2 + b^2 > c^2 := by
  sorry

end acute_triangle_from_sides_l3708_370813


namespace min_fish_in_aquarium_l3708_370843

/-- Represents the number of fish of each known color in the aquarium -/
structure AquariumFish where
  yellow : ℕ
  blue : ℕ
  green : ℕ

/-- The conditions of the aquarium as described in the problem -/
def aquarium_conditions (fish : AquariumFish) : Prop :=
  fish.yellow = 12 ∧
  fish.blue = fish.yellow / 2 ∧
  fish.green = fish.yellow * 2

/-- The theorem stating the minimum number of fish in the aquarium -/
theorem min_fish_in_aquarium (fish : AquariumFish) 
  (h : aquarium_conditions fish) : 
  fish.yellow + fish.blue + fish.green = 42 := by
  sorry

end min_fish_in_aquarium_l3708_370843


namespace regular_pyramid_lateral_area_l3708_370897

/-- Theorem: The lateral surface area of a regular pyramid equals the area of the base
    divided by the cosine of the dihedral angle between a lateral face and the base. -/
theorem regular_pyramid_lateral_area 
  (n : ℕ) -- number of sides in the base
  (S : ℝ) -- area of one lateral face
  (A : ℝ) -- area of the base
  (α : ℝ) -- dihedral angle between a lateral face and the base
  (h1 : n > 0) -- the pyramid has at least 3 sides
  (h2 : S > 0) -- lateral face area is positive
  (h3 : A > 0) -- base area is positive
  (h4 : 0 < α ∧ α < π / 2) -- dihedral angle is between 0 and π/2
  : n * S = A / Real.cos α := by
  sorry

end regular_pyramid_lateral_area_l3708_370897


namespace michelle_crayons_l3708_370896

theorem michelle_crayons (num_boxes : ℕ) (crayons_per_box : ℕ) (h1 : num_boxes = 7) (h2 : crayons_per_box = 5) :
  num_boxes * crayons_per_box = 35 := by
  sorry

end michelle_crayons_l3708_370896


namespace longest_side_of_triangle_l3708_370881

theorem longest_side_of_triangle (a b c : ℝ) (perimeter : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a / b = 3 / 2 →
  a / c = 2 →
  b / c = 4 / 3 →
  a + b + c = perimeter →
  perimeter = 104 →
  a = 48 := by
sorry

end longest_side_of_triangle_l3708_370881


namespace no_solutions_in_interval_l3708_370840

theorem no_solutions_in_interval (x : Real) : 
  0 ≤ x ∧ x ≤ 2 * Real.pi → 1 / Real.sin x + 1 / Real.cos x ≠ 4 := by
  sorry

end no_solutions_in_interval_l3708_370840


namespace proposition_implication_l3708_370894

theorem proposition_implication (P : ℕ → Prop) 
  (h1 : ∀ k : ℕ, k > 0 → (P k → P (k + 1)))
  (h2 : ¬ P 5) : 
  ¬ P 4 := by
  sorry

end proposition_implication_l3708_370894


namespace unique_quadratic_solution_l3708_370872

/-- Represents a quadratic equation ax^2 + 4x + c = 0 with exactly one solution -/
structure UniqueQuadratic where
  a : ℝ
  c : ℝ
  has_unique_solution : (4^2 - 4*a*c) = 0
  sum_constraint : a + c = 5
  order_constraint : a < c

theorem unique_quadratic_solution (q : UniqueQuadratic) : (q.a, q.c) = (1, 4) := by
  sorry

end unique_quadratic_solution_l3708_370872


namespace smallest_initial_value_l3708_370863

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem smallest_initial_value : 
  ∀ n : ℕ, n ≥ 308 → 
  (is_perfect_square (n - 139) ∧ 
   ∀ m : ℕ, m < n → ¬ is_perfect_square (m - 139)) :=
by sorry

end smallest_initial_value_l3708_370863


namespace number_equation_solution_l3708_370892

theorem number_equation_solution :
  ∀ x : ℝ, 35 + 3 * x^2 = 89 → x = 3 * Real.sqrt 2 ∨ x = -3 * Real.sqrt 2 := by
sorry

end number_equation_solution_l3708_370892


namespace joan_rock_collection_l3708_370858

theorem joan_rock_collection (minerals_today minerals_yesterday gemstones : ℕ) : 
  gemstones = minerals_yesterday / 2 →
  minerals_today = minerals_yesterday + 6 →
  minerals_today = 48 →
  gemstones = 21 := by
sorry

end joan_rock_collection_l3708_370858


namespace inequality_proof_l3708_370873

theorem inequality_proof (n : ℕ+) (x : ℝ) (hx : x > 0) :
  x + (n : ℝ)^(n : ℕ) / x^(n : ℕ) ≥ (n : ℝ) + 1 := by
  sorry

end inequality_proof_l3708_370873


namespace partial_fraction_decomposition_l3708_370874

theorem partial_fraction_decomposition :
  ∃! (A B C : ℚ),
    ∀ (x : ℚ), x ≠ 2 → x ≠ 3 → x ≠ 4 →
      (x^2 - 9) / ((x - 2) * (x - 3) * (x - 4)) =
      A / (x - 2) + B / (x - 3) + C / (x - 4) ∧
      A = -5/2 ∧ B = 0 ∧ C = 7/2 := by
sorry

end partial_fraction_decomposition_l3708_370874


namespace cuboid_volume_l3708_370824

/-- The volume of a cuboid that can be divided into 3 equal cubes, each with edges measuring 6 cm, is 648 cm³. -/
theorem cuboid_volume (cuboid : Real) (cube : Real) :
  (cuboid = 3 * cube) →  -- The cuboid is divided into 3 equal parts
  (cube = 6^3) →         -- Each part is a cube with edges measuring 6 cm
  (cuboid = 648) :=      -- The volume of the original cuboid is 648 cm³
by sorry

end cuboid_volume_l3708_370824


namespace monotonically_increasing_intervals_minimum_value_in_interval_f_properties_l3708_370818

-- Define the function f(x)
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 + 4

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 6 * x^2 - 6 * x

-- Theorem for monotonically increasing intervals
theorem monotonically_increasing_intervals :
  (∀ x < 0, f' x > 0) ∧ (∀ x > 1, f' x > 0) :=
sorry

-- Theorem for minimum value in the interval [-1, 2]
theorem minimum_value_in_interval :
  ∀ x ∈ Set.Icc (-1) 2, f x ≥ f (-1) :=
sorry

-- Main theorem combining both parts
theorem f_properties :
  (∀ x < 0, f' x > 0) ∧ (∀ x > 1, f' x > 0) ∧
  (∀ x ∈ Set.Icc (-1) 2, f x ≥ f (-1)) :=
sorry

end monotonically_increasing_intervals_minimum_value_in_interval_f_properties_l3708_370818


namespace total_books_on_shelves_l3708_370864

theorem total_books_on_shelves (num_shelves : ℕ) (books_per_shelf : ℕ) :
  num_shelves = 625 → books_per_shelf = 28 → num_shelves * books_per_shelf = 22500 := by
  sorry

end total_books_on_shelves_l3708_370864


namespace decagon_triangles_l3708_370867

/-- The number of vertices in a regular decagon -/
def n : ℕ := 10

/-- The number of vertices needed to form a triangle -/
def k : ℕ := 3

/-- The number of triangles that can be formed using the vertices of a regular decagon -/
def num_triangles : ℕ := Nat.choose n k

theorem decagon_triangles : num_triangles = 120 := by
  sorry

end decagon_triangles_l3708_370867


namespace remaining_balance_calculation_l3708_370817

def cost_per_charge : ℚ := 3.5
def number_of_charges : ℕ := 4
def initial_budget : ℚ := 20

theorem remaining_balance_calculation :
  initial_budget - (cost_per_charge * number_of_charges) = 6 := by
  sorry

end remaining_balance_calculation_l3708_370817


namespace smallest_square_area_l3708_370870

/-- The smallest area of a square containing non-overlapping 1x4 and 2x5 rectangles -/
theorem smallest_square_area (r1_width r1_height r2_width r2_height : ℕ) 
  (h1 : r1_width = 1 ∧ r1_height = 4)
  (h2 : r2_width = 2 ∧ r2_height = 5)
  (h_no_overlap : True)  -- Represents the non-overlapping condition
  (h_parallel : True)    -- Represents the parallel sides condition
  : ∃ (s : ℕ), s^2 = 81 ∧ ∀ (t : ℕ), (t ≥ r1_width ∧ t ≥ r1_height ∧ t ≥ r2_width ∧ t ≥ r2_height) → t^2 ≥ s^2 := by
  sorry

#check smallest_square_area

end smallest_square_area_l3708_370870


namespace arithmetic_computation_l3708_370802

theorem arithmetic_computation : 2 + 5 * 3 - 4 + 6 * 2 / 3 = 17 := by sorry

end arithmetic_computation_l3708_370802


namespace complex_fraction_equality_l3708_370848

theorem complex_fraction_equality : (3 - I) / (1 - I) = 2 + I := by
  sorry

end complex_fraction_equality_l3708_370848


namespace absolute_value_minus_self_nonnegative_l3708_370899

theorem absolute_value_minus_self_nonnegative (a : ℝ) : |a| - a ≥ 0 := by
  sorry

end absolute_value_minus_self_nonnegative_l3708_370899


namespace inequality_equivalence_l3708_370868

theorem inequality_equivalence (x : ℝ) : 
  2 / (x + 2) + 4 / (x + 8) ≤ 7 / 3 ↔ -8 < x ∧ x ≤ 4 := by
  sorry

end inequality_equivalence_l3708_370868


namespace tile_count_theorem_l3708_370891

/-- Represents a rectangular floor covered with square tiles. -/
structure TiledRectangle where
  length : ℕ
  width : ℕ
  diagonal_tiles : ℕ

/-- The condition that one side is twice the length of the other. -/
def double_side (rect : TiledRectangle) : Prop :=
  rect.length = 2 * rect.width

/-- The number of tiles on the diagonals. -/
def diagonal_count (rect : TiledRectangle) : ℕ :=
  rect.diagonal_tiles

/-- The total number of tiles covering the floor. -/
def total_tiles (rect : TiledRectangle) : ℕ :=
  rect.length * rect.width

/-- The main theorem stating the problem. -/
theorem tile_count_theorem (rect : TiledRectangle) :
  double_side rect → diagonal_count rect = 49 → total_tiles rect = 50 := by
  sorry


end tile_count_theorem_l3708_370891


namespace simple_interest_rate_calculation_l3708_370822

theorem simple_interest_rate_calculation (P A T : ℝ) (h1 : P = 750) (h2 : A = 900) (h3 : T = 5) :
  let R := (A - P) / (P * T) * 100
  R = 4 := by sorry

end simple_interest_rate_calculation_l3708_370822


namespace decimal_to_fraction_l3708_370850

theorem decimal_to_fraction (x : ℚ) (h : x = 336/100) : 
  ∃ (a b : ℕ), x = a / b ∧ a = 84 ∧ b = 25 := by
  sorry

end decimal_to_fraction_l3708_370850


namespace solution_replacement_concentration_l3708_370811

/-- Calculates the new concentration of a solution after partial replacement -/
def new_concentration (initial_conc : ℝ) (replacement_conc : ℝ) (replaced_fraction : ℝ) : ℝ :=
  (1 - replaced_fraction) * initial_conc + replaced_fraction * replacement_conc

/-- Theorem stating that replacing 7/9 of a 70% solution with a 25% solution results in a 35% solution -/
theorem solution_replacement_concentration :
  new_concentration 0.7 0.25 (7/9) = 0.35 := by
  sorry

end solution_replacement_concentration_l3708_370811


namespace roots_sum_of_squares_l3708_370838

theorem roots_sum_of_squares (m : ℤ) (a b c : ℤ) : 
  (∀ x : ℤ, x^3 - 2023*x + m = 0 ↔ x = a ∨ x = b ∨ x = c) → 
  a^2 + b^2 + c^2 = 4046 := by
  sorry

end roots_sum_of_squares_l3708_370838


namespace smallest_integer_l3708_370821

theorem smallest_integer (a b : ℕ) (ha : a = 60) (h_lcm_gcd : Nat.lcm a b / Nat.gcd a b = 28) :
  ∀ c : ℕ, c > 0 ∧ Nat.lcm a c / Nat.gcd a c = 28 → b ≤ c → b = 105 :=
sorry

end smallest_integer_l3708_370821


namespace log_problem_l3708_370882

theorem log_problem (y : ℝ) (p : ℝ) 
  (h1 : Real.log 5 / Real.log 9 = y)
  (h2 : Real.log 125 / Real.log 3 = p * y) : 
  p = 6 := by sorry

end log_problem_l3708_370882


namespace persimmons_in_box_l3708_370884

/-- Given a box containing apples and persimmons, prove the number of persimmons. -/
theorem persimmons_in_box (apples : ℕ) (persimmons : ℕ) : apples = 3 → persimmons = 2 → persimmons = 2 := by
  sorry

end persimmons_in_box_l3708_370884


namespace min_sum_of_intercepts_l3708_370875

/-- A line with positive x-intercept and y-intercept passing through (1,4) -/
structure InterceptLine where
  a : ℝ
  b : ℝ
  a_pos : a > 0
  b_pos : b > 0
  passes_through : 1 / a + 4 / b = 1

/-- The minimum sum of intercepts for a line passing through (1,4) is 9 -/
theorem min_sum_of_intercepts (l : InterceptLine) : l.a + l.b ≥ 9 := by
  sorry

end min_sum_of_intercepts_l3708_370875


namespace largest_circle_area_in_square_l3708_370839

/-- The area of the largest circle inside a square of side length 70 cm -/
theorem largest_circle_area_in_square : 
  let square_side : ℝ := 70
  let circle_area : ℝ := Real.pi * (square_side / 2)^2
  circle_area = 1225 * Real.pi := by
  sorry

end largest_circle_area_in_square_l3708_370839


namespace racket_sales_total_l3708_370854

/-- The total amount for which rackets were sold, given the average price per pair and the number of pairs sold. -/
theorem racket_sales_total (avg_price : ℝ) (num_pairs : ℕ) (h1 : avg_price = 9.8) (h2 : num_pairs = 70) :
  avg_price * (num_pairs : ℝ) = 686 := by
  sorry

end racket_sales_total_l3708_370854


namespace subset_iff_a_in_range_l3708_370849

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 4*x + 3 < 0}
def B (a : ℝ) : Set ℝ := {x | 2^(1-x) + a ≤ 0 ∧ x^2 - 2*(a+7)*x + 5 ≤ 0}

-- State the theorem
theorem subset_iff_a_in_range :
  ∀ a : ℝ, A ⊆ B a ↔ -4 ≤ a ∧ a ≤ -1 := by sorry

end subset_iff_a_in_range_l3708_370849


namespace distance_to_nearest_town_l3708_370886

theorem distance_to_nearest_town (d : ℝ) :
  (¬ (d ≥ 6)) ∧ (¬ (d ≤ 5)) ∧ (¬ (d ≤ 4)) → 5 < d ∧ d < 6 := by
  sorry

end distance_to_nearest_town_l3708_370886


namespace midpoint_triangle_perimeter_l3708_370888

/-- Given a triangle with perimeter p, the perimeter of the triangle formed by 
    connecting the midpoints of its sides is p/2. -/
theorem midpoint_triangle_perimeter (p : ℝ) (h : p > 0) : 
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = p ∧ 
  (a/2 + b/2 + c/2) = p/2 := by
sorry

end midpoint_triangle_perimeter_l3708_370888


namespace passes_count_l3708_370845

/-- The number of times Griffin and Hailey pass each other during their run -/
def number_of_passes (
  run_time : ℝ)
  (griffin_speed : ℝ)
  (hailey_speed : ℝ)
  (griffin_radius : ℝ)
  (hailey_radius : ℝ) : ℕ :=
  sorry

theorem passes_count :
  number_of_passes 45 260 310 50 45 = 86 :=
sorry

end passes_count_l3708_370845


namespace game_attendance_l3708_370835

theorem game_attendance : ∃ (total : ℕ), 
  (total : ℚ) * (40 / 100) + (total : ℚ) * (34 / 100) + 3 = total ∧ total = 12 := by
  sorry

end game_attendance_l3708_370835


namespace farmer_theorem_l3708_370876

def farmer_problem (initial_tomatoes initial_potatoes remaining_total : ℕ) : ℕ :=
  (initial_tomatoes + initial_potatoes) - remaining_total

theorem farmer_theorem (initial_tomatoes initial_potatoes remaining_total : ℕ) :
  farmer_problem initial_tomatoes initial_potatoes remaining_total =
  (initial_tomatoes + initial_potatoes) - remaining_total :=
by sorry

#eval farmer_problem 175 77 80

end farmer_theorem_l3708_370876


namespace common_root_implies_zero_l3708_370828

theorem common_root_implies_zero (a b : ℝ) : 
  (∃ r : ℝ, r^2 + a*r + b^2 = 0 ∧ r^2 + b*r + a^2 = 0) → 
  ¬(a ≠ 0 ∧ b ≠ 0) := by
sorry

end common_root_implies_zero_l3708_370828


namespace slope_of_line_l3708_370808

/-- The slope of a line given by the equation 4y = -6x + 12 is -3/2 -/
theorem slope_of_line (x y : ℝ) : 4 * y = -6 * x + 12 → (y - 3) / x = -3 / 2 := by
  sorry

end slope_of_line_l3708_370808


namespace quadratic_no_real_roots_l3708_370862

theorem quadratic_no_real_roots : ∀ x : ℝ, x^2 - 4 * Real.sqrt 2 * x + 9 ≠ 0 := by
  sorry

end quadratic_no_real_roots_l3708_370862


namespace other_communities_count_l3708_370893

/-- Given a school with 300 boys, calculate the number of boys belonging to other communities -/
theorem other_communities_count (total : ℕ) (muslim_percent hindu_percent sikh_percent : ℚ) : 
  total = 300 →
  muslim_percent = 44 / 100 →
  hindu_percent = 28 / 100 →
  sikh_percent = 10 / 100 →
  ↑total * (1 - (muslim_percent + hindu_percent + sikh_percent)) = 54 :=
by sorry

end other_communities_count_l3708_370893


namespace vector_equation_solution_l3708_370866

theorem vector_equation_solution :
  let a₁ : ℚ := 181 / 136
  let a₂ : ℚ := 25 / 68
  let v₁ : Fin 2 → ℚ := ![4, -1]
  let v₂ : Fin 2 → ℚ := ![5, 3]
  let result : Fin 2 → ℚ := ![9, 4]
  (a₁ • v₁ + a₂ • v₂) = result := by sorry

end vector_equation_solution_l3708_370866


namespace existence_of_abc_l3708_370869

def S (x : ℕ) : ℕ := (x.digits 10).sum

theorem existence_of_abc : ∃ (a b c : ℕ), 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  S (a + b) < 5 ∧ 
  S (b + c) < 5 ∧ 
  S (c + a) < 5 ∧ 
  S (a + b + c) > 50 := by
  sorry

end existence_of_abc_l3708_370869


namespace solve_fraction_equation_l3708_370859

theorem solve_fraction_equation (x : ℝ) : (1 / 3 - 1 / 4 : ℝ) = 1 / x → x = 12 := by
  sorry

end solve_fraction_equation_l3708_370859


namespace regression_for_related_variables_l3708_370827

/-- A type representing a statistical variable -/
structure StatVariable where
  name : String

/-- A type representing a statistical analysis method -/
inductive AnalysisMethod
  | ErrorAnalysis
  | RegressionAnalysis
  | IndependenceTest

/-- A relation indicating that two variables are related -/
def are_related (v1 v2 : StatVariable) : Prop := sorry

/-- The correct method to analyze related variables -/
def analyze_related_variables (v1 v2 : StatVariable) : AnalysisMethod :=
  AnalysisMethod.RegressionAnalysis

/-- Theorem stating that regression analysis is the correct method for analyzing related variables -/
theorem regression_for_related_variables (height weight : StatVariable) 
    (h : are_related height weight) : 
    analyze_related_variables height weight = AnalysisMethod.RegressionAnalysis := by
  sorry

end regression_for_related_variables_l3708_370827


namespace tangent_ellipse_hyperbola_l3708_370898

/-- Prove that if an ellipse and a hyperbola are tangent, then the parameter m of the hyperbola is 8 -/
theorem tangent_ellipse_hyperbola (x y m : ℝ) :
  (∃ x y, x^2 + 9*y^2 = 9 ∧ x^2 - m*(y+3)^2 = 1) →  -- Ellipse and hyperbola equations
  (∀ x y, x^2 + 9*y^2 = 9 → x^2 - m*(y+3)^2 ≤ 1) →  -- Tangency condition
  (∃ x y, x^2 + 9*y^2 = 9 ∧ x^2 - m*(y+3)^2 = 1) →  -- Point of tangency exists
  m = 8 :=
by sorry

end tangent_ellipse_hyperbola_l3708_370898


namespace quadratic_one_solution_l3708_370829

theorem quadratic_one_solution (p : ℝ) : 
  (∃! x : ℝ, 3 * x^2 - 6 * x + p = 0) ↔ p = 3 := by
  sorry

end quadratic_one_solution_l3708_370829


namespace no_real_intersection_l3708_370847

theorem no_real_intersection : ¬∃ x : ℝ, 3 * x^2 - 6 * x + 5 = 0 := by sorry

end no_real_intersection_l3708_370847


namespace largest_mersenne_prime_under_1000_l3708_370844

def is_mersenne_prime (n : ℕ) : Prop :=
  ∃ p : ℕ, Nat.Prime p ∧ n = 2^p - 1 ∧ Nat.Prime n

theorem largest_mersenne_prime_under_1000 :
  ∀ n : ℕ, is_mersenne_prime n → n < 1000 → n ≤ 127 :=
by sorry

end largest_mersenne_prime_under_1000_l3708_370844


namespace socks_total_is_112_25_l3708_370860

/-- The total number of socks George and Maria have after receiving additional socks -/
def total_socks (george_initial : ℝ) (maria_initial : ℝ) 
                (george_bought : ℝ) (george_from_dad : ℝ) 
                (maria_from_mom : ℝ) (maria_from_aunt : ℝ) : ℝ :=
  (george_initial + george_bought + george_from_dad) + 
  (maria_initial + maria_from_mom + maria_from_aunt)

/-- Theorem stating that the total number of socks is 112.25 -/
theorem socks_total_is_112_25 : 
  total_socks 28.5 24.75 36.25 4.5 15.5 2.75 = 112.25 := by
  sorry

end socks_total_is_112_25_l3708_370860


namespace shaded_areas_sum_l3708_370878

theorem shaded_areas_sum (R : ℝ) (h1 : R > 0) (h2 : π * R^2 = 81 * π) : 
  (π * R^2) / 2 + (π * (R/2)^2) / 2 = 50.625 * π := by sorry

end shaded_areas_sum_l3708_370878


namespace books_sold_on_wednesday_l3708_370889

theorem books_sold_on_wednesday 
  (initial_stock : ℕ) 
  (sold_monday : ℕ) 
  (sold_tuesday : ℕ) 
  (sold_thursday : ℕ) 
  (sold_friday : ℕ) 
  (unsold : ℕ) 
  (h1 : initial_stock = 800)
  (h2 : sold_monday = 60)
  (h3 : sold_tuesday = 10)
  (h4 : sold_thursday = 44)
  (h5 : sold_friday = 66)
  (h6 : unsold = 600) :
  initial_stock - unsold - (sold_monday + sold_tuesday + sold_thursday + sold_friday) = 20 := by
  sorry

end books_sold_on_wednesday_l3708_370889


namespace least_five_digit_congruent_to_six_mod_seventeen_l3708_370895

theorem least_five_digit_congruent_to_six_mod_seventeen :
  ∃ (n : ℕ), 
    (n ≥ 10000 ∧ n < 100000) ∧  -- Five-digit positive integer
    (n % 17 = 6) ∧              -- Congruent to 6 (mod 17)
    (∀ m : ℕ, (m ≥ 10000 ∧ m < 100000) ∧ (m % 17 = 6) → n ≤ m) ∧  -- Least such number
    n = 10002                   -- The number is 10,002
  := by sorry

end least_five_digit_congruent_to_six_mod_seventeen_l3708_370895


namespace angle_property_equivalence_l3708_370806

theorem angle_property_equivalence (θ : Real) (h : 0 ≤ θ ∧ θ ≤ 2 * Real.pi) :
  (∀ x : Real, 0 ≤ x ∧ x ≤ 1 → x^2 * Real.cos θ - x * (1 - x) + 2 * (1 - x)^2 * Real.sin θ > 0) ↔
  (π / 24 < θ ∧ θ < 11 * π / 24) ∨ (25 * π / 24 < θ ∧ θ < 47 * π / 24) :=
by sorry

end angle_property_equivalence_l3708_370806


namespace maurice_previous_rides_l3708_370853

/-- Represents the horseback riding scenario of Maurice and Matt -/
structure RidingScenario where
  maurice_prev_rides : ℕ
  maurice_prev_horses : ℕ
  matt_total_horses : ℕ
  maurice_visit_rides : ℕ
  matt_rides_with_maurice : ℕ
  matt_solo_rides : ℕ
  matt_solo_horses : ℕ

/-- The specific scenario described in the problem -/
def problem_scenario : RidingScenario :=
  { maurice_prev_rides := 0,  -- to be determined
    maurice_prev_horses := 2,
    matt_total_horses := 4,
    maurice_visit_rides := 8,
    matt_rides_with_maurice := 8,
    matt_solo_rides := 16,
    matt_solo_horses := 2 }

/-- Theorem stating the number of Maurice's previous rides -/
theorem maurice_previous_rides (s : RidingScenario) :
  s.maurice_prev_horses = 2 ∧
  s.matt_total_horses = 4 ∧
  s.maurice_visit_rides = 8 ∧
  s.matt_rides_with_maurice = 8 ∧
  s.matt_solo_rides = 16 ∧
  s.matt_solo_horses = 2 ∧
  s.maurice_visit_rides = s.maurice_prev_rides ∧
  (s.matt_rides_with_maurice + s.matt_solo_rides) = 3 * s.maurice_prev_rides →
  s.maurice_prev_rides = 8 := by
  sorry

#check maurice_previous_rides problem_scenario

end maurice_previous_rides_l3708_370853


namespace room_width_calculation_l3708_370826

/-- Given a room with specified length, total paving cost, and paving rate per square meter,
    calculate the width of the room. -/
theorem room_width_calculation (length : ℝ) (total_cost : ℝ) (rate_per_sqm : ℝ) :
  length = 5.5 →
  total_cost = 28875 →
  rate_per_sqm = 1400 →
  (total_cost / rate_per_sqm) / length = 3.75 := by
  sorry

end room_width_calculation_l3708_370826


namespace regular_decagon_area_l3708_370851

theorem regular_decagon_area (s : ℝ) (h : s = Real.sqrt 2) :
  let area := 5 * s^2 * (Real.sqrt (5 + 2 * Real.sqrt 5)) / 4
  area = 3.5 + 4 * Real.sqrt 2 := by
  sorry

end regular_decagon_area_l3708_370851


namespace l_shaped_area_l3708_370812

/-- The area of an L-shaped region formed by subtracting three non-overlapping squares
    from a larger square -/
theorem l_shaped_area (side_length : ℝ) (small_square1 : ℝ) (small_square2 : ℝ) (small_square3 : ℝ)
    (h1 : side_length = 6)
    (h2 : small_square1 = 2)
    (h3 : small_square2 = 4)
    (h4 : small_square3 = 2)
    (h5 : small_square1 + small_square2 + small_square3 ≤ side_length) :
    side_length^2 - (small_square1^2 + small_square2^2 + small_square3^2) = 12 := by
  sorry

end l_shaped_area_l3708_370812


namespace virginia_egg_problem_l3708_370846

/-- Virginia's egg problem -/
theorem virginia_egg_problem (initial_eggs : ℕ) (taken_eggs : ℕ) : 
  initial_eggs = 96 → taken_eggs = 3 → initial_eggs - taken_eggs = 93 := by
sorry

end virginia_egg_problem_l3708_370846


namespace distance_to_point_distance_from_origin_to_point_l3708_370871

theorem distance_to_point : ℝ → ℝ → ℝ
  | x, y => Real.sqrt (x^2 + y^2)

theorem distance_from_origin_to_point : distance_to_point (-8) 15 = 17 := by
  sorry

end distance_to_point_distance_from_origin_to_point_l3708_370871


namespace fishermen_catch_l3708_370805

theorem fishermen_catch (total_fish : ℕ) (carp_ratio : ℚ) (perch_ratio : ℚ) 
  (h_total : total_fish = 80)
  (h_carp : carp_ratio = 5 / 9)
  (h_perch : perch_ratio = 7 / 11) :
  ∃ (first_catch second_catch : ℕ),
    first_catch + second_catch = total_fish ∧
    first_catch = 36 ∧
    second_catch = 44 := by
  sorry

end fishermen_catch_l3708_370805


namespace andrews_grapes_l3708_370831

theorem andrews_grapes (price_grapes : ℕ) (quantity_mangoes : ℕ) (price_mangoes : ℕ) (total_paid : ℕ) :
  price_grapes = 74 →
  quantity_mangoes = 9 →
  price_mangoes = 59 →
  total_paid = 975 →
  ∃ (quantity_grapes : ℕ), 
    quantity_grapes * price_grapes + quantity_mangoes * price_mangoes = total_paid ∧
    quantity_grapes = 6 := by
  sorry

end andrews_grapes_l3708_370831


namespace volume_of_specific_solid_l3708_370833

/-- 
A solid with a square base and extended top edges.
s: side length of the square base
-/
structure ExtendedSolid where
  s : ℝ
  base_square : s > 0
  top_extended : ℝ × ℝ
  vertical_edge : ℝ

/-- The volume of the ExtendedSolid -/
noncomputable def volume (solid : ExtendedSolid) : ℝ :=
  solid.s^2 * solid.s

/-- Theorem: The volume of the specific ExtendedSolid is 128√2 -/
theorem volume_of_specific_solid :
  ∃ (solid : ExtendedSolid),
    solid.s = 4 * Real.sqrt 2 ∧
    solid.top_extended = (3 * solid.s, solid.s) ∧
    solid.vertical_edge = solid.s ∧
    volume solid = 128 * Real.sqrt 2 := by
  sorry

end volume_of_specific_solid_l3708_370833


namespace gcf_of_90_135_225_l3708_370879

theorem gcf_of_90_135_225 : Nat.gcd 90 (Nat.gcd 135 225) = 45 := by
  sorry

end gcf_of_90_135_225_l3708_370879


namespace dave_tickets_l3708_370809

theorem dave_tickets (initial_tickets : ℕ) : 
  (initial_tickets - 2 - 10 = 2) → initial_tickets = 14 := by
  sorry

end dave_tickets_l3708_370809


namespace kendall_change_total_l3708_370830

/-- Represents the value of coins in cents -/
def coin_value (coin : String) : ℕ :=
  match coin with
  | "quarter" => 25
  | "dime" => 10
  | "nickel" => 5
  | _ => 0

/-- Calculates the total value of a given number of coins -/
def coin_total (coin : String) (count : ℕ) : ℕ :=
  (coin_value coin) * count

/-- Theorem stating the total amount of money Kendall has in change -/
theorem kendall_change_total : 
  coin_total "quarter" 10 + coin_total "dime" 12 + coin_total "nickel" 6 = 400 := by
  sorry

end kendall_change_total_l3708_370830


namespace wedding_guests_count_l3708_370865

/-- The number of guests attending the wedding -/
def total_guests : ℕ := 240

/-- The proportion of female guests -/
def female_proportion : ℚ := 3/5

/-- The proportion of female guests from Jay's family -/
def jay_family_proportion : ℚ := 1/2

/-- The number of female guests from Jay's family -/
def jay_family_females : ℕ := 72

theorem wedding_guests_count :
  (jay_family_females : ℚ) = (total_guests : ℚ) * female_proportion * jay_family_proportion :=
by sorry

end wedding_guests_count_l3708_370865

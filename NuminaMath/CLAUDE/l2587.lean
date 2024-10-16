import Mathlib

namespace NUMINAMATH_CALUDE_largest_four_digit_divisible_by_33_l2587_258714

/-- A function that checks if a number is a four-digit number -/
def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

/-- A function that checks if a number is divisible by 33 -/
def divisible_by_33 (n : ℕ) : Prop :=
  n % 33 = 0

/-- Theorem stating that 9999 is the largest four-digit number divisible by 33 -/
theorem largest_four_digit_divisible_by_33 :
  is_four_digit 9999 ∧ 
  divisible_by_33 9999 ∧ 
  ∀ n : ℕ, is_four_digit n → divisible_by_33 n → n ≤ 9999 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_four_digit_divisible_by_33_l2587_258714


namespace NUMINAMATH_CALUDE_boys_to_girls_ratio_l2587_258701

theorem boys_to_girls_ratio (S : ℚ) (G : ℚ) (B : ℚ) : 
  S > 0 → G > 0 → B > 0 →
  S = G + B →
  (1 / 2) * G = (1 / 5) * S →
  B / G = 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_boys_to_girls_ratio_l2587_258701


namespace NUMINAMATH_CALUDE_value_of_a_minus_b_l2587_258796

theorem value_of_a_minus_b (a b : ℝ) 
  (eq1 : 2010 * a + 2014 * b = 2018)
  (eq2 : 2012 * a + 2016 * b = 2020) : 
  a - b = -3 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_minus_b_l2587_258796


namespace NUMINAMATH_CALUDE_decimal_to_binary_21_l2587_258700

theorem decimal_to_binary_21 : 
  (21 : ℕ) = (1 * 2^4 + 0 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0) :=
by sorry

end NUMINAMATH_CALUDE_decimal_to_binary_21_l2587_258700


namespace NUMINAMATH_CALUDE_solution_l2587_258780

-- Define the function f
def f (x : ℝ) : ℝ := (x - 1) * (x - 2) * (x - 3)

-- Define the set M
def M : Set ℝ := {x | f x = 0}

-- Theorem statement
theorem solution : {1, 3} ∪ {2, 3} = M := by sorry

end NUMINAMATH_CALUDE_solution_l2587_258780


namespace NUMINAMATH_CALUDE_path_count_equals_combination_l2587_258705

/-- The width of the grid -/
def grid_width : ℕ := 6

/-- The height of the grid -/
def grid_height : ℕ := 5

/-- The total number of steps required to reach from A to B -/
def total_steps : ℕ := grid_width + grid_height - 2

/-- The number of vertical steps required -/
def vertical_steps : ℕ := grid_height - 1

theorem path_count_equals_combination : 
  (Nat.choose total_steps vertical_steps) = 126 := by sorry

end NUMINAMATH_CALUDE_path_count_equals_combination_l2587_258705


namespace NUMINAMATH_CALUDE_distributive_property_division_l2587_258783

theorem distributive_property_division (a b c : ℝ) (hc : c ≠ 0) :
  (∀ x y z : ℝ, (x + y) * z = x * z + y * z) →
  (a + b) / c = a / c + b / c :=
sorry

end NUMINAMATH_CALUDE_distributive_property_division_l2587_258783


namespace NUMINAMATH_CALUDE_total_paintable_area_is_1624_l2587_258764

/-- The number of bedrooms in Isabella's house -/
def num_bedrooms : ℕ := 4

/-- The length of each bedroom in feet -/
def bedroom_length : ℕ := 15

/-- The width of each bedroom in feet -/
def bedroom_width : ℕ := 12

/-- The height of each bedroom in feet -/
def bedroom_height : ℕ := 9

/-- The area occupied by doorways and windows in each bedroom in square feet -/
def unpaintable_area : ℕ := 80

/-- The total area of walls to be painted in square feet -/
def total_paintable_area : ℕ := 
  num_bedrooms * (
    2 * (bedroom_length * bedroom_height + bedroom_width * bedroom_height) - unpaintable_area
  )

theorem total_paintable_area_is_1624 : total_paintable_area = 1624 := by
  sorry

end NUMINAMATH_CALUDE_total_paintable_area_is_1624_l2587_258764


namespace NUMINAMATH_CALUDE_sasha_floor_problem_l2587_258707

theorem sasha_floor_problem (total_floors : ℕ) :
  (∃ (floors_descended : ℕ),
    floors_descended = total_floors / 3 ∧
    floors_descended + 1 = total_floors - (total_floors / 2)) →
  total_floors + 1 = 7 :=
by sorry

end NUMINAMATH_CALUDE_sasha_floor_problem_l2587_258707


namespace NUMINAMATH_CALUDE_john_squat_difference_l2587_258754

/-- Given John's raw squat weight, the weight added by sleeves, and the percentage added by wraps,
    calculate the difference between the weight added by wraps and sleeves. -/
def weight_difference (raw_squat : ℝ) (sleeve_addition : ℝ) (wrap_percentage : ℝ) : ℝ :=
  raw_squat * wrap_percentage - sleeve_addition

/-- Prove that the difference between the weight added by wraps and sleeves to John's squat is 120 pounds. -/
theorem john_squat_difference :
  weight_difference 600 30 0.25 = 120 := by
  sorry

end NUMINAMATH_CALUDE_john_squat_difference_l2587_258754


namespace NUMINAMATH_CALUDE_water_remaining_l2587_258706

theorem water_remaining (initial : ℚ) (used : ℚ) (remaining : ℚ) : 
  initial = 3 → used = 5/4 → remaining = initial - used → remaining = 7/4 := by
  sorry

end NUMINAMATH_CALUDE_water_remaining_l2587_258706


namespace NUMINAMATH_CALUDE_largest_b_value_l2587_258792

theorem largest_b_value (b : ℝ) (h : (3*b + 6)*(b - 2) = 9*b) : b ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_largest_b_value_l2587_258792


namespace NUMINAMATH_CALUDE_temperature_conversion_l2587_258713

theorem temperature_conversion (k : ℝ) (t : ℝ) : 
  (t = 5 / 9 * (k - 32)) → (k = 167) → (t = 75) := by
  sorry

end NUMINAMATH_CALUDE_temperature_conversion_l2587_258713


namespace NUMINAMATH_CALUDE_difference_of_sums_l2587_258709

def sum_even_up_to (n : ℕ) : ℕ :=
  (n / 2) * (2 + n)

def sum_odd_up_to (n : ℕ) : ℕ :=
  ((n + 1) / 2) ^ 2

theorem difference_of_sums : sum_even_up_to 100 - sum_odd_up_to 29 = 2325 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_sums_l2587_258709


namespace NUMINAMATH_CALUDE_tan_sum_pi_fractions_l2587_258744

theorem tan_sum_pi_fractions : Real.tan (π / 12) + Real.tan (5 * π / 12) = 4 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_pi_fractions_l2587_258744


namespace NUMINAMATH_CALUDE_cuboid_height_calculation_l2587_258703

/-- The surface area of a cuboid given its length, breadth, and height -/
def cuboidSurfaceArea (l b h : ℝ) : ℝ := 2 * (l * b + b * h + h * l)

/-- Theorem: A cuboid with length 4 cm, breadth 6 cm, and surface area 120 cm² has a height of 3.6 cm -/
theorem cuboid_height_calculation (h : ℝ) :
  cuboidSurfaceArea 4 6 h = 120 → h = 3.6 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_height_calculation_l2587_258703


namespace NUMINAMATH_CALUDE_final_position_l2587_258791

/-- Represents the position of the letter F -/
inductive Position
  | PositiveX_PositiveY
  | NegativeX_NegativeY
  | PositiveX_NegativeY
  | NegativeX_PositiveY
  | PositiveXPlusY
  | NegativeXPlusY
  | PositiveXMinusY
  | NegativeXMinusY

/-- Represents the transformations -/
inductive Transformation
  | RotateClockwise (angle : ℝ)
  | ReflectXAxis
  | RotateAroundOrigin (angle : ℝ)

/-- Initial position of F after 90° clockwise rotation -/
def initialPosition : Position := Position.PositiveX_NegativeY

/-- Sequence of transformations -/
def transformations : List Transformation := [
  Transformation.RotateClockwise 45,
  Transformation.ReflectXAxis,
  Transformation.RotateAroundOrigin 180
]

/-- Applies a single transformation to a position -/
def applyTransformation (p : Position) (t : Transformation) : Position :=
  sorry

/-- Applies a sequence of transformations to a position -/
def applyTransformations (p : Position) (ts : List Transformation) : Position :=
  sorry

/-- The final position theorem -/
theorem final_position :
  applyTransformations initialPosition transformations = Position.NegativeXPlusY :=
  sorry

end NUMINAMATH_CALUDE_final_position_l2587_258791


namespace NUMINAMATH_CALUDE_work_completion_time_l2587_258728

theorem work_completion_time 
  (total_work : ℕ) 
  (initial_men : ℕ) 
  (remaining_men : ℕ) 
  (remaining_days : ℕ) :
  initial_men = 100 →
  remaining_men = 50 →
  remaining_days = 40 →
  total_work = remaining_men * remaining_days →
  total_work = initial_men * (total_work / initial_men) →
  total_work / initial_men = 20 :=
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l2587_258728


namespace NUMINAMATH_CALUDE_bell_rings_count_l2587_258771

/-- Represents a school event that causes the bell to ring at its start and end -/
structure SchoolEvent where
  name : String

/-- Represents the school schedule for a day -/
structure SchoolSchedule where
  events : List SchoolEvent

/-- Counts the number of bell rings for a given schedule up to and including a specific event -/
def countBellRings (schedule : SchoolSchedule) (currentEvent : SchoolEvent) : Nat :=
  sorry

/-- Monday's altered schedule -/
def mondaySchedule : SchoolSchedule :=
  { events := [
    { name := "Assembly" },
    { name := "Maths" },
    { name := "History" },
    { name := "Surprise Quiz" },
    { name := "Geography" },
    { name := "Science" },
    { name := "Music" }
  ] }

/-- The current event (Geography class) -/
def currentEvent : SchoolEvent :=
  { name := "Geography" }

theorem bell_rings_count :
  countBellRings mondaySchedule currentEvent = 9 := by
  sorry

end NUMINAMATH_CALUDE_bell_rings_count_l2587_258771


namespace NUMINAMATH_CALUDE_union_A_B_intersection_A_complement_B_l2587_258763

-- Define the universal set U
def U : Set ℝ := {x | x ≤ 4}

-- Define set A
def A : Set ℝ := {x | -2 < x ∧ x < 3}

-- Define set B
def B : Set ℝ := {x | -3 ≤ x ∧ x ≤ 2}

-- Theorem for A ∪ B
theorem union_A_B : A ∪ B = {x | -3 ≤ x ∧ x < 3} := by sorry

-- Theorem for A ∩ (∁U B)
theorem intersection_A_complement_B : A ∩ (U \ B) = {x | 2 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_union_A_B_intersection_A_complement_B_l2587_258763


namespace NUMINAMATH_CALUDE_monomial_polynomial_multiplication_l2587_258762

theorem monomial_polynomial_multiplication :
  ∀ (x y : ℝ), -3 * x * y * (4 * y - 2 * x - 1) = -12 * x * y^2 + 6 * x^2 * y + 3 * x * y := by
  sorry

end NUMINAMATH_CALUDE_monomial_polynomial_multiplication_l2587_258762


namespace NUMINAMATH_CALUDE_arman_sister_age_ratio_l2587_258745

/-- Given Arman and his sister's ages at different points in time, prove the ratio of their current ages -/
theorem arman_sister_age_ratio :
  ∀ (sister_age_4_years_ago : ℕ) (arman_age_4_years_future : ℕ),
    sister_age_4_years_ago = 2 →
    arman_age_4_years_future = 40 →
    (arman_age_4_years_future - 4) / (sister_age_4_years_ago + 4) = 6 :=
by
  sorry


end NUMINAMATH_CALUDE_arman_sister_age_ratio_l2587_258745


namespace NUMINAMATH_CALUDE_first_part_speed_l2587_258738

theorem first_part_speed (total_distance : ℝ) (first_part_distance : ℝ) (second_part_speed : ℝ) (average_speed : ℝ) 
  (h1 : total_distance = 60)
  (h2 : first_part_distance = 12)
  (h3 : second_part_speed = 48)
  (h4 : average_speed = 40)
  (h5 : total_distance = first_part_distance + (total_distance - first_part_distance))
  (h6 : average_speed = total_distance / (first_part_distance / v + (total_distance - first_part_distance) / second_part_speed)) :
  v = 24 := by
  sorry

end NUMINAMATH_CALUDE_first_part_speed_l2587_258738


namespace NUMINAMATH_CALUDE_alice_sales_above_quota_l2587_258789

def alice_quota : ℕ := 2000

def shoe_prices : List (String × ℕ) := [
  ("Adidas", 45),
  ("Nike", 60),
  ("Reeboks", 35),
  ("Puma", 50),
  ("Converse", 40)
]

def sales : List (String × ℕ) := [
  ("Nike", 12),
  ("Adidas", 10),
  ("Reeboks", 15),
  ("Puma", 8),
  ("Converse", 14)
]

def total_sales : ℕ := (sales.map (fun (s : String × ℕ) =>
  match shoe_prices.find? (fun (p : String × ℕ) => p.1 = s.1) with
  | some price => s.2 * price.2
  | none => 0
)).sum

theorem alice_sales_above_quota :
  total_sales - alice_quota = 655 := by sorry

end NUMINAMATH_CALUDE_alice_sales_above_quota_l2587_258789


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_l2587_258712

-- Define the propositions
def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*a*x - a ≥ 0

def q (a : ℝ) : Prop := -1 < a ∧ a < 0

-- Theorem statement
theorem p_necessary_not_sufficient :
  (∀ a : ℝ, q a → p a) ∧ 
  (∃ a : ℝ, p a ∧ ¬q a) :=
sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_l2587_258712


namespace NUMINAMATH_CALUDE_total_money_is_305_l2587_258767

/-- The value of a gold coin in dollars -/
def gold_coin_value : ℕ := 50

/-- The value of a silver coin in dollars -/
def silver_coin_value : ℕ := 25

/-- The number of gold coins -/
def num_gold_coins : ℕ := 3

/-- The number of silver coins -/
def num_silver_coins : ℕ := 5

/-- The amount of cash in dollars -/
def cash : ℕ := 30

/-- The total amount of money in dollars -/
def total_money : ℕ := gold_coin_value * num_gold_coins + silver_coin_value * num_silver_coins + cash

theorem total_money_is_305 : total_money = 305 := by
  sorry

end NUMINAMATH_CALUDE_total_money_is_305_l2587_258767


namespace NUMINAMATH_CALUDE_unique_special_prime_l2587_258757

/-- A prime number that can be written both as a sum of two primes and as a difference of two primes -/
def SpecialPrime (p : ℕ) : Prop :=
  Nat.Prime p ∧
  (∃ q r : ℕ, Nat.Prime q ∧ Nat.Prime r ∧ p = q + r) ∧
  (∃ s t : ℕ, Nat.Prime s ∧ Nat.Prime t ∧ p = s - t)

/-- The theorem stating that 5 is the only prime satisfying the SpecialPrime property -/
theorem unique_special_prime :
  ∀ p : ℕ, SpecialPrime p ↔ p = 5 := by
  sorry

end NUMINAMATH_CALUDE_unique_special_prime_l2587_258757


namespace NUMINAMATH_CALUDE_salary_average_increase_l2587_258732

theorem salary_average_increase 
  (num_employees : ℕ) 
  (avg_salary : ℚ) 
  (manager_salary : ℚ) : 
  num_employees = 24 → 
  avg_salary = 2400 → 
  manager_salary = 4900 → 
  (((num_employees : ℚ) * avg_salary + manager_salary) / ((num_employees : ℚ) + 1)) - avg_salary = 100 := by
  sorry

end NUMINAMATH_CALUDE_salary_average_increase_l2587_258732


namespace NUMINAMATH_CALUDE_carmen_sculpture_height_l2587_258768

/-- Represents a measurement in feet and inches -/
structure FeetInches where
  feet : ℕ
  inches : ℕ
  h_valid : inches < 12

/-- Converts inches to a FeetInches measurement -/
def inchesToFeetInches (totalInches : ℕ) : FeetInches :=
  { feet := totalInches / 12,
    inches := totalInches % 12,
    h_valid := by sorry }

/-- Adds two FeetInches measurements -/
def addFeetInches (a b : FeetInches) : FeetInches :=
  inchesToFeetInches (a.feet * 12 + a.inches + b.feet * 12 + b.inches)

theorem carmen_sculpture_height :
  let rectangular_prism_height : ℕ := 8
  let cylinder_height : ℕ := 15
  let pyramid_height : ℕ := 10
  let base_height : ℕ := 10
  let sculpture_height := rectangular_prism_height + cylinder_height + pyramid_height
  let sculpture_feet_inches := inchesToFeetInches sculpture_height
  let base_feet_inches := inchesToFeetInches base_height
  let combined_height := addFeetInches sculpture_feet_inches base_feet_inches
  combined_height = { feet := 3, inches := 7, h_valid := by sorry } := by sorry

end NUMINAMATH_CALUDE_carmen_sculpture_height_l2587_258768


namespace NUMINAMATH_CALUDE_sean_initial_blocks_l2587_258781

/-- The number of blocks Sean had initially -/
def initial_blocks : ℕ := sorry

/-- The number of blocks eaten by the hippopotamus -/
def blocks_eaten : ℕ := 29

/-- The number of blocks remaining after the hippopotamus ate some -/
def blocks_remaining : ℕ := 26

/-- Theorem stating that Sean initially had 55 blocks -/
theorem sean_initial_blocks : initial_blocks = 55 := by sorry

end NUMINAMATH_CALUDE_sean_initial_blocks_l2587_258781


namespace NUMINAMATH_CALUDE_work_completion_proof_l2587_258736

/-- The number of men initially planned to complete the work -/
def initial_men : ℕ := 38

/-- The number of days it takes the initial group to complete the work -/
def initial_days : ℕ := 10

/-- The number of men sent to another project -/
def men_sent_away : ℕ := 25

/-- The number of days it takes to complete the work after sending men away -/
def new_days : ℕ := 30

/-- The total amount of work in man-days -/
def total_work : ℕ := initial_men * initial_days

theorem work_completion_proof :
  initial_men * initial_days = (initial_men - men_sent_away) * new_days :=
by sorry

end NUMINAMATH_CALUDE_work_completion_proof_l2587_258736


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l2587_258777

/-- Given a line L1 with equation 2x + y - 5 = 0 and a point A(1, 2),
    the line L2 passing through A and perpendicular to L1 has equation x - 2y + 3 = 0 -/
theorem perpendicular_line_equation :
  let L1 : ℝ → ℝ → Prop := fun x y ↦ 2 * x + y - 5 = 0
  let A : ℝ × ℝ := (1, 2)
  let L2 : ℝ → ℝ → Prop := fun x y ↦ x - 2 * y + 3 = 0
  (∀ x y, L2 x y ↔ (y - A.2 = -(1 / (2 : ℝ)) * (x - A.1))) ∧
  (∀ x₁ y₁ x₂ y₂, L1 x₁ y₁ → L1 x₂ y₂ → L2 x₁ y₁ → L2 x₂ y₂ →
    (x₂ - x₁) * (x₂ - x₁) + (y₂ - y₁) * (y₂ - y₁) ≠ 0 →
    ((x₂ - x₁) * (2 : ℝ) + (y₂ - y₁) * (1 : ℝ)) * ((x₂ - x₁) * (1 : ℝ) + (y₂ - y₁) * (-2 : ℝ)) = 0) ∧
  L2 A.1 A.2 :=
by
  sorry


end NUMINAMATH_CALUDE_perpendicular_line_equation_l2587_258777


namespace NUMINAMATH_CALUDE_michael_digging_time_l2587_258766

/-- Given the conditions of Michael's and his father's hole digging, prove that Michael will take 700 hours to dig his hole. -/
theorem michael_digging_time (father_rate : ℝ) (father_time : ℝ) (michael_depth_diff : ℝ) :
  father_rate = 4 →
  father_time = 400 →
  michael_depth_diff = 400 →
  (2 * (father_rate * father_time) - michael_depth_diff) / father_rate = 700 :=
by sorry

end NUMINAMATH_CALUDE_michael_digging_time_l2587_258766


namespace NUMINAMATH_CALUDE_square_area_proof_l2587_258721

theorem square_area_proof (x : ℚ) : 
  (5 * x - 20 = 25 - 2 * x) → 
  ((5 * x - 20)^2 : ℚ) = 7225 / 49 := by
sorry

end NUMINAMATH_CALUDE_square_area_proof_l2587_258721


namespace NUMINAMATH_CALUDE_construction_materials_cost_l2587_258758

/-- The total cost of materials for a construction company. -/
def total_cost (gravel_tons sand_tons cement_tons : Float)
               (gravel_price sand_price cement_price : Float) : Float :=
  gravel_tons * gravel_price + sand_tons * sand_price + cement_tons * cement_price

/-- Theorem stating that the total cost of the given materials is $750.57. -/
theorem construction_materials_cost :
  let gravel_tons : Float := 5.91
  let sand_tons : Float := 8.11
  let cement_tons : Float := 4.35
  let gravel_price : Float := 30.50
  let sand_price : Float := 40.50
  let cement_price : Float := 55.60
  total_cost gravel_tons sand_tons cement_tons gravel_price sand_price cement_price = 750.57 := by
  sorry

end NUMINAMATH_CALUDE_construction_materials_cost_l2587_258758


namespace NUMINAMATH_CALUDE_prob_two_red_two_blue_correct_l2587_258708

/-- The probability of selecting 2 red and 2 blue marbles from a bag -/
def probability_two_red_two_blue : ℚ :=
  let total_marbles : ℕ := 20
  let red_marbles : ℕ := 12
  let blue_marbles : ℕ := 8
  let selected_marbles : ℕ := 4
  616 / 1615

/-- Theorem stating that the probability of selecting 2 red and 2 blue marbles
    from a bag with 12 red and 8 blue marbles, when 4 marbles are selected
    at random without replacement, is equal to 616/1615 -/
theorem prob_two_red_two_blue_correct :
  probability_two_red_two_blue = 616 / 1615 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_red_two_blue_correct_l2587_258708


namespace NUMINAMATH_CALUDE_exists_zero_sum_subset_l2587_258751

/-- Represents a row in the table -/
def Row (n : ℕ) := Fin n → Int

/-- The table with all possible rows of 1 and -1 -/
def OriginalTable (n : ℕ) : Finset (Row n) :=
  sorry

/-- A function that potentially replaces some elements with zero -/
def Corrupt (n : ℕ) : Row n → Row n :=
  sorry

/-- The corrupted table after replacing some elements with zero -/
def CorruptedTable (n : ℕ) : Finset (Row n) :=
  sorry

/-- Sum of a set of rows -/
def RowSum (n : ℕ) (rows : Finset (Row n)) : Row n :=
  sorry

/-- A row of all zeros -/
def ZeroRow (n : ℕ) : Row n :=
  sorry

/-- The main theorem -/
theorem exists_zero_sum_subset (n : ℕ) :
  ∃ (subset : Finset (Row n)), subset ⊆ CorruptedTable n ∧ RowSum n subset = ZeroRow n :=
sorry

end NUMINAMATH_CALUDE_exists_zero_sum_subset_l2587_258751


namespace NUMINAMATH_CALUDE_pirate_treasure_distribution_l2587_258740

/-- Represents the number of coins in the final distribution step -/
def x : ℕ := sorry

/-- Pete's coin distribution pattern -/
def petes_coins (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Polly's final coin count -/
def pollys_coins : ℕ := x

/-- Pete's final coin count -/
def petes_final_coins : ℕ := 3 * x

theorem pirate_treasure_distribution :
  petes_coins x = petes_final_coins ∧
  pollys_coins + petes_final_coins = 20 := by sorry

end NUMINAMATH_CALUDE_pirate_treasure_distribution_l2587_258740


namespace NUMINAMATH_CALUDE_equal_candy_sharing_l2587_258733

/-- Represents the number of candies each person has initially -/
structure CandyDistribution :=
  (mark : ℕ)
  (peter : ℕ)
  (john : ℕ)

/-- Calculates the total number of candies -/
def totalCandies (d : CandyDistribution) : ℕ :=
  d.mark + d.peter + d.john

/-- Calculates the number of candies each person gets after equal sharing -/
def sharedCandies (d : CandyDistribution) : ℕ :=
  totalCandies d / 3

/-- Proves that when Mark (30 candies), Peter (25 candies), and John (35 candies)
    combine their candies and share equally, each person will have 30 candies -/
theorem equal_candy_sharing :
  let d : CandyDistribution := { mark := 30, peter := 25, john := 35 }
  sharedCandies d = 30 := by
  sorry

end NUMINAMATH_CALUDE_equal_candy_sharing_l2587_258733


namespace NUMINAMATH_CALUDE_smallest_product_of_given_numbers_l2587_258773

theorem smallest_product_of_given_numbers : 
  let numbers : List ℕ := [10, 11, 12, 13, 14]
  let smallest := numbers.minimum?
  let next_smallest := numbers.filter (· ≠ smallest.getD 0) |>.minimum?
  smallest.isSome ∧ next_smallest.isSome → 
  smallest.getD 0 * next_smallest.getD 0 = 110 := by
sorry

end NUMINAMATH_CALUDE_smallest_product_of_given_numbers_l2587_258773


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_right_angle_points_l2587_258775

/-- An isosceles trapezoid with the given properties -/
structure IsoscelesTrapezoid where
  a : ℝ  -- length of AB
  c : ℝ  -- length of CD
  h : ℝ  -- perpendicular distance from A to CD
  a_positive : 0 < a
  c_positive : 0 < c
  h_positive : 0 < h
  c_le_a : c ≤ a  -- As CD is parallel to and shorter than AB

/-- The point X on the axis of symmetry -/
def X (t : IsoscelesTrapezoid) := {x : ℝ // 0 ≤ x ∧ x ≤ t.h}

/-- The theorem stating the conditions for X to exist and its distance from AB -/
theorem isosceles_trapezoid_right_angle_points (t : IsoscelesTrapezoid) :
  ∃ (x : X t), (x.val = t.h / 2 - Real.sqrt (t.h^2 - t.a * t.c) / 2 ∨
                x.val = t.h / 2 + Real.sqrt (t.h^2 - t.a * t.c) / 2) ↔
  t.h^2 ≥ t.a * t.c :=
sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_right_angle_points_l2587_258775


namespace NUMINAMATH_CALUDE_hotdog_problem_l2587_258784

theorem hotdog_problem (h1 h2 : ℕ) : 
  h2 = h1 - 25 → 
  h1 + h2 = 125 → 
  h1 = 75 := by
sorry

end NUMINAMATH_CALUDE_hotdog_problem_l2587_258784


namespace NUMINAMATH_CALUDE_functional_equation_solution_l2587_258717

theorem functional_equation_solution (f : ℤ → ℤ) :
  (∀ x y : ℤ, f (f x + y + 1) = x + f y + 1) →
  ((∀ n : ℤ, f n = n) ∨ (∀ n : ℤ, f n = -n - 2)) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l2587_258717


namespace NUMINAMATH_CALUDE_jessica_withdrawal_l2587_258752

theorem jessica_withdrawal (initial_balance : ℝ) (withdrawal : ℝ) : 
  withdrawal = (2 / 5) * initial_balance ∧
  (3 / 5) * initial_balance + (1 / 2) * ((3 / 5) * initial_balance) = 450 →
  withdrawal = 200 := by
sorry

end NUMINAMATH_CALUDE_jessica_withdrawal_l2587_258752


namespace NUMINAMATH_CALUDE_remainder_of_2457634_div_8_l2587_258720

theorem remainder_of_2457634_div_8 : 2457634 % 8 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_2457634_div_8_l2587_258720


namespace NUMINAMATH_CALUDE_turtleneck_discount_theorem_l2587_258711

theorem turtleneck_discount_theorem (C : ℝ) (D : ℝ) : 
  C > 0 →  -- Cost is positive
  (1.50 * C) * (1 - D / 100) = 1.125 * C → -- Equation from profit condition
  D = 25 := by
sorry

end NUMINAMATH_CALUDE_turtleneck_discount_theorem_l2587_258711


namespace NUMINAMATH_CALUDE_blue_tetrahedron_volume_l2587_258790

/-- The volume of the tetrahedron formed by alternating vertices of a cube -/
theorem blue_tetrahedron_volume (cube_side_length : ℝ) (h : cube_side_length = 8) :
  let cube_volume := cube_side_length ^ 3
  let blue_tetrahedron_volume := cube_volume / 3
  blue_tetrahedron_volume = 512 / 3 := by
  sorry

end NUMINAMATH_CALUDE_blue_tetrahedron_volume_l2587_258790


namespace NUMINAMATH_CALUDE_first_group_size_correct_l2587_258722

/-- The number of persons in the first group that can repair a road -/
def first_group_size : ℕ := 78

/-- The number of days the first group takes to repair the road -/
def first_group_days : ℕ := 12

/-- The number of hours per day the first group works -/
def first_group_hours_per_day : ℕ := 5

/-- The number of persons in the second group -/
def second_group_size : ℕ := 30

/-- The number of days the second group takes to repair the road -/
def second_group_days : ℕ := 26

/-- The number of hours per day the second group works -/
def second_group_hours_per_day : ℕ := 6

/-- Theorem stating that the first group size is correct given the conditions -/
theorem first_group_size_correct :
  first_group_size * first_group_days * first_group_hours_per_day =
  second_group_size * second_group_days * second_group_hours_per_day :=
by sorry

end NUMINAMATH_CALUDE_first_group_size_correct_l2587_258722


namespace NUMINAMATH_CALUDE_nth_equation_pattern_l2587_258793

theorem nth_equation_pattern (n : ℕ) : 1 + 6 * n = (3 * n + 1)^2 - 9 * n^2 := by
  sorry

end NUMINAMATH_CALUDE_nth_equation_pattern_l2587_258793


namespace NUMINAMATH_CALUDE_positive_A_value_l2587_258725

-- Define the # relation
def hash (A B : ℝ) : ℝ := A^2 + B^2

-- Theorem statement
theorem positive_A_value (A : ℝ) (h : hash A 7 = 200) : A = Real.sqrt 151 := by
  sorry

end NUMINAMATH_CALUDE_positive_A_value_l2587_258725


namespace NUMINAMATH_CALUDE_prom_ticket_cost_l2587_258795

def total_cost : ℝ := 836
def dinner_cost : ℝ := 120
def tip_percentage : ℝ := 0.30
def limo_cost_per_hour : ℝ := 80
def limo_rental_duration : ℝ := 6
def number_of_tickets : ℝ := 2

theorem prom_ticket_cost :
  let tip_cost := dinner_cost * tip_percentage
  let limo_total_cost := limo_cost_per_hour * limo_rental_duration
  let total_cost_without_tickets := dinner_cost + tip_cost + limo_total_cost
  let ticket_total_cost := total_cost - total_cost_without_tickets
  let ticket_cost := ticket_total_cost / number_of_tickets
  ticket_cost = 100 := by sorry

end NUMINAMATH_CALUDE_prom_ticket_cost_l2587_258795


namespace NUMINAMATH_CALUDE_festival_allowance_rate_l2587_258702

/-- The daily rate for a festival allowance given the number of staff members,
    number of days, and total amount. -/
def daily_rate (staff_members : ℕ) (days : ℕ) (total_amount : ℕ) : ℚ :=
  total_amount / (staff_members * days)

/-- Theorem stating that the daily rate for the festival allowance is 110
    given the problem conditions. -/
theorem festival_allowance_rate : 
  daily_rate 20 30 66000 = 110 := by sorry

end NUMINAMATH_CALUDE_festival_allowance_rate_l2587_258702


namespace NUMINAMATH_CALUDE_largest_prime_divisor_exists_l2587_258719

def base_5_number : ℕ := 2031357

theorem largest_prime_divisor_exists :
  ∃ p : ℕ, Prime p ∧ p ∣ base_5_number ∧ ∀ q : ℕ, Prime q → q ∣ base_5_number → q ≤ p :=
sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_exists_l2587_258719


namespace NUMINAMATH_CALUDE_square_perimeter_l2587_258760

theorem square_perimeter (s : ℝ) (h : s > 0) : 
  (3 * s = 40) → (4 * s = 160 / 3) := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l2587_258760


namespace NUMINAMATH_CALUDE_unique_fraction_property_l2587_258797

theorem unique_fraction_property : ∃! (a b : ℕ), 
  b ≠ 0 ∧ 
  (a : ℚ) / b = (a + 4 : ℚ) / (b + 10) ∧ 
  (a : ℚ) / b = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_unique_fraction_property_l2587_258797


namespace NUMINAMATH_CALUDE_sector_area_l2587_258737

/-- Given a circular sector with an arc length of 2 cm and a central angle of 2 radians,
    prove that the area of the sector is 1 cm². -/
theorem sector_area (arc_length : ℝ) (central_angle : ℝ) (h1 : arc_length = 2) (h2 : central_angle = 2) :
  (1 / 2) * (arc_length / central_angle)^2 * central_angle = 1 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l2587_258737


namespace NUMINAMATH_CALUDE_james_total_toys_l2587_258716

/-- The number of toy cars James buys -/
def toy_cars : ℕ := 20

/-- The number of toy soldiers James buys -/
def toy_soldiers : ℕ := 2 * toy_cars

/-- The total number of toys James buys -/
def total_toys : ℕ := toy_cars + toy_soldiers

theorem james_total_toys : total_toys = 60 := by
  sorry

end NUMINAMATH_CALUDE_james_total_toys_l2587_258716


namespace NUMINAMATH_CALUDE_widget_earnings_proof_l2587_258776

/-- Calculates the earnings per widget given the hourly wage, required widgets per week,
    work hours per week, and total weekly earnings. -/
def earnings_per_widget (hourly_wage : ℚ) (widgets_per_week : ℕ) (hours_per_week : ℕ) (total_earnings : ℚ) : ℚ :=
  (total_earnings - (hourly_wage * hours_per_week)) / widgets_per_week

/-- Proves that the earnings per widget is $0.16 given the specific conditions. -/
theorem widget_earnings_proof :
  let hourly_wage : ℚ := 25/2  -- $12.50
  let widgets_per_week : ℕ := 1250
  let hours_per_week : ℕ := 40
  let total_earnings : ℚ := 700
  earnings_per_widget hourly_wage widgets_per_week hours_per_week total_earnings = 4/25  -- $0.16
  := by sorry

end NUMINAMATH_CALUDE_widget_earnings_proof_l2587_258776


namespace NUMINAMATH_CALUDE_product_relation_l2587_258750

theorem product_relation (x y z : ℝ) (h : x^2 + y^2 = x*y*(z + 1/z)) :
  x = y*z ∨ y = x*z := by sorry

end NUMINAMATH_CALUDE_product_relation_l2587_258750


namespace NUMINAMATH_CALUDE_prime_sum_theorem_l2587_258786

theorem prime_sum_theorem (x y : ℕ) : 
  Nat.Prime x → Nat.Prime y → x < y → x + y = 36 → 4 * x + y = 51 := by sorry

end NUMINAMATH_CALUDE_prime_sum_theorem_l2587_258786


namespace NUMINAMATH_CALUDE_no_solution_to_double_inequality_l2587_258704

theorem no_solution_to_double_inequality :
  ¬ ∃ x : ℝ, (4 * x - 3 < (x + 2)^2) ∧ ((x + 2)^2 < 8 * x - 5) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_to_double_inequality_l2587_258704


namespace NUMINAMATH_CALUDE_derivative_not_in_second_quadrant_l2587_258778

-- Define the quadratic function
def f (x : ℝ) (b c : ℝ) : ℝ := x^2 + b*x + c

-- Define the derivative of f
def f' (x : ℝ) (b : ℝ) : ℝ := 2*x + b

-- Theorem statement
theorem derivative_not_in_second_quadrant (b c : ℝ) :
  (∀ x, f x b c = f (-x + 4) b c) →  -- axis of symmetry is x = 2
  ∀ x y, f' x b = y → ¬(x < 0 ∧ y > 0) :=
by sorry

end NUMINAMATH_CALUDE_derivative_not_in_second_quadrant_l2587_258778


namespace NUMINAMATH_CALUDE_socks_bought_l2587_258787

/-- Given John's sock inventory changes, prove the number of new socks bought. -/
theorem socks_bought (initial : ℕ) (thrown_away : ℕ) (final : ℕ) 
  (h1 : initial = 33)
  (h2 : thrown_away = 19)
  (h3 : final = 27) :
  final - (initial - thrown_away) = 13 := by
  sorry

end NUMINAMATH_CALUDE_socks_bought_l2587_258787


namespace NUMINAMATH_CALUDE_simplify_absolute_value_expression_l2587_258710

noncomputable def f (x : ℝ) : ℝ := |2*x + 1| - |x - 3| + |x - 6|

noncomputable def g (x : ℝ) : ℝ :=
  if x < -1/2 then -2*x + 2
  else if x < 3 then 2*x + 4
  else if x < 6 then 10
  else 2*x - 2

theorem simplify_absolute_value_expression :
  ∀ x : ℝ, f x = g x := by sorry

end NUMINAMATH_CALUDE_simplify_absolute_value_expression_l2587_258710


namespace NUMINAMATH_CALUDE_total_apples_is_36_l2587_258715

/-- The number of apples picked by Mike -/
def mike_apples : ℕ := 7

/-- The number of apples picked by Nancy -/
def nancy_apples : ℕ := 3

/-- The number of apples picked by Keith -/
def keith_apples : ℕ := 6

/-- The number of apples picked by Olivia -/
def olivia_apples : ℕ := 12

/-- The number of apples picked by Thomas -/
def thomas_apples : ℕ := 8

/-- The total number of apples picked -/
def total_apples : ℕ := mike_apples + nancy_apples + keith_apples + olivia_apples + thomas_apples

theorem total_apples_is_36 : total_apples = 36 := by
  sorry

end NUMINAMATH_CALUDE_total_apples_is_36_l2587_258715


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l2587_258727

theorem simplify_and_evaluate (x : ℝ) : 
  (2*x + 1)^2 - (x + 3)*(x - 3) = 30 ↔ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l2587_258727


namespace NUMINAMATH_CALUDE_brownies_needed_l2587_258724

/-- Represents the amount of frosting used for different baked goods -/
structure FrostingUsage where
  layerCake : ℝ
  singleCake : ℝ
  panBrownies : ℝ
  dozenCupcakes : ℝ

/-- Represents the quantities of baked goods Paul needs to prepare -/
structure BakedGoods where
  layerCakes : ℕ
  singleCakes : ℕ
  dozenCupcakes : ℕ

def totalFrostingNeeded : ℝ := 21

theorem brownies_needed (usage : FrostingUsage) (goods : BakedGoods) 
  (h1 : usage.layerCake = 1)
  (h2 : usage.singleCake = 0.5)
  (h3 : usage.panBrownies = 0.5)
  (h4 : usage.dozenCupcakes = 0.5)
  (h5 : goods.layerCakes = 3)
  (h6 : goods.singleCakes = 12)
  (h7 : goods.dozenCupcakes = 6) :
  (totalFrostingNeeded - 
   (goods.layerCakes * usage.layerCake + 
    goods.singleCakes * usage.singleCake + 
    goods.dozenCupcakes * usage.dozenCupcakes)) / usage.panBrownies = 18 := by
  sorry

end NUMINAMATH_CALUDE_brownies_needed_l2587_258724


namespace NUMINAMATH_CALUDE_quadratic_sufficient_not_necessary_l2587_258749

theorem quadratic_sufficient_not_necessary (a b c : ℝ) :
  (∀ x : ℝ, a * x^2 + b * x + c > 0) ↔ 
  ((a > 0 ∧ b^2 - 4*a*c < 0) ∨ 
   ∃ a' b' c' : ℝ, (∀ x : ℝ, a' * x^2 + b' * x + c' > 0) ∧ ¬(a' > 0 ∧ b'^2 - 4*a'*c' < 0)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_sufficient_not_necessary_l2587_258749


namespace NUMINAMATH_CALUDE_duke_of_york_men_percentage_l2587_258746

/-- The percentage of men remaining after two consecutive losses -/
theorem duke_of_york_men_percentage : 
  let initial_men : ℕ := 10000
  let first_loss_rate : ℚ := 1/10
  let second_loss_rate : ℚ := 3/20
  let remaining_men : ℚ := initial_men * (1 - first_loss_rate) * (1 - second_loss_rate)
  let percentage_remaining : ℚ := remaining_men / initial_men * 100
  percentage_remaining = 76.5 := by
  sorry

end NUMINAMATH_CALUDE_duke_of_york_men_percentage_l2587_258746


namespace NUMINAMATH_CALUDE_sugar_for_cake_l2587_258741

/-- Given a cake recipe that requires a total of 0.8 cups of sugar,
    with 0.6 cups used for frosting, prove that the cake itself
    requires 0.2 cups of sugar. -/
theorem sugar_for_cake
  (total_sugar : ℝ)
  (frosting_sugar : ℝ)
  (h1 : total_sugar = 0.8)
  (h2 : frosting_sugar = 0.6) :
  total_sugar - frosting_sugar = 0.2 :=
by sorry

end NUMINAMATH_CALUDE_sugar_for_cake_l2587_258741


namespace NUMINAMATH_CALUDE_complex_product_positive_implies_zero_l2587_258788

theorem complex_product_positive_implies_zero (a : ℝ) :
  (Complex.I * (a - Complex.I)).re > 0 → a = 0 := by sorry

end NUMINAMATH_CALUDE_complex_product_positive_implies_zero_l2587_258788


namespace NUMINAMATH_CALUDE_sum_divisible_by_ten_l2587_258734

theorem sum_divisible_by_ten (n : ℕ) : 
  10 ∣ (n^2 + (n+1)^2 + (n+2)^2 + (n+3)^2) ↔ n % 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_divisible_by_ten_l2587_258734


namespace NUMINAMATH_CALUDE_combinatorial_identity_l2587_258718

theorem combinatorial_identity 
  (n k m : ℕ) 
  (h1 : 1 ≤ k) 
  (h2 : k < m) 
  (h3 : m ≤ n) : 
  (Finset.sum (Finset.range (k + 1)) (λ i => Nat.choose k i * Nat.choose n (m - i))) = 
  Nat.choose (n + k) m := by
  sorry

end NUMINAMATH_CALUDE_combinatorial_identity_l2587_258718


namespace NUMINAMATH_CALUDE_f_decreasing_interval_l2587_258743

-- Define the derivative of f
def f' (x : ℝ) : ℝ := x^2 - 4*x + 3

-- State the theorem
theorem f_decreasing_interval :
  ∀ f : ℝ → ℝ, (∀ x, deriv f x = f' x) →
  ∀ x ∈ Set.Ioo 0 2, deriv (fun y ↦ f (y + 1)) x < 0 :=
by sorry

end NUMINAMATH_CALUDE_f_decreasing_interval_l2587_258743


namespace NUMINAMATH_CALUDE_set_equality_implies_sum_l2587_258769

theorem set_equality_implies_sum (a b : ℝ) : 
  ({a, b/a, 1} : Set ℝ) = ({a^2, a+b, 0} : Set ℝ) → a^2002 + b^2003 = 1 := by
  sorry

end NUMINAMATH_CALUDE_set_equality_implies_sum_l2587_258769


namespace NUMINAMATH_CALUDE_karl_drove_420_miles_l2587_258799

/-- Represents Karl's car and trip details --/
structure KarlsTrip where
  miles_per_gallon : ℝ
  tank_capacity : ℝ
  initial_distance : ℝ
  gas_bought : ℝ
  final_tank_fraction : ℝ

/-- Calculates the total distance driven given the trip details --/
def total_distance (trip : KarlsTrip) : ℝ :=
  trip.initial_distance

/-- Theorem stating that Karl drove 420 miles --/
theorem karl_drove_420_miles :
  let trip := KarlsTrip.mk 30 16 420 10 (3/4)
  total_distance trip = 420 := by sorry

end NUMINAMATH_CALUDE_karl_drove_420_miles_l2587_258799


namespace NUMINAMATH_CALUDE_min_quadrilateral_area_l2587_258785

/-- Definition of the ellipse -/
def ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

/-- The ellipse passes through the point (1, √2/2) -/
axiom point_on_ellipse : ellipse 1 (Real.sqrt 2 / 2)

/-- The point (1,0) is a focus of the ellipse -/
axiom focus_point : ∃ c, c^2 = 1 ∧ c^2 = 2 - 1

/-- Definition of perpendicular lines through (1,0) -/
def perpendicular_lines (m₁ m₂ : ℝ) : Prop := 
  m₁ * m₂ = -1 ∧ m₁ ≠ 0 ∧ m₂ ≠ 0

/-- Definition of the area of the quadrilateral formed by intersection points -/
noncomputable def quadrilateral_area (m₁ m₂ : ℝ) : ℝ := 
  4 * (m₁^2 + 1)^2 / ((m₁^2 + 2) * (2 * m₂^2 + 1))

/-- The main theorem to prove -/
theorem min_quadrilateral_area : 
  ∃ (m₁ m₂ : ℝ), perpendicular_lines m₁ m₂ ∧ 
  (∀ (n₁ n₂ : ℝ), perpendicular_lines n₁ n₂ → 
    quadrilateral_area m₁ m₂ ≤ quadrilateral_area n₁ n₂) ∧
  quadrilateral_area m₁ m₂ = 16/9 :=
sorry

end NUMINAMATH_CALUDE_min_quadrilateral_area_l2587_258785


namespace NUMINAMATH_CALUDE_bridge_dealing_is_systematic_sampling_l2587_258756

/-- Represents the sampling method used in card dealing --/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Other

/-- Represents a deck of cards --/
structure Deck :=
  (size : Nat)
  (shuffled : Bool)

/-- Represents the card dealing process in bridge --/
structure BridgeDealing :=
  (deck : Deck)
  (startingCardRandom : Bool)
  (dealInOrder : Bool)
  (playerHandSize : Nat)

/-- Determines the sampling method used in bridge card dealing --/
def determineSamplingMethod (dealing : BridgeDealing) : SamplingMethod :=
  sorry

/-- Theorem stating that bridge card dealing uses Systematic Sampling --/
theorem bridge_dealing_is_systematic_sampling 
  (dealing : BridgeDealing) 
  (h1 : dealing.deck.size = 52)
  (h2 : dealing.deck.shuffled = true)
  (h3 : dealing.startingCardRandom = true)
  (h4 : dealing.dealInOrder = true)
  (h5 : dealing.playerHandSize = 13) :
  determineSamplingMethod dealing = SamplingMethod.Systematic :=
  sorry

end NUMINAMATH_CALUDE_bridge_dealing_is_systematic_sampling_l2587_258756


namespace NUMINAMATH_CALUDE_min_ones_23x23_l2587_258747

/-- Represents a tiling of a square grid --/
structure Tiling (n : ℕ) :=
  (ones : ℕ)
  (twos : ℕ)
  (threes : ℕ)
  (valid : ones + 4 * twos + 9 * threes = n^2)

/-- The minimum number of 1x1 squares in a valid 23x23 tiling --/
def min_ones : ℕ := 1

theorem min_ones_23x23 :
  ∀ (t : Tiling 23), t.ones ≥ min_ones :=
sorry

end NUMINAMATH_CALUDE_min_ones_23x23_l2587_258747


namespace NUMINAMATH_CALUDE_smallest_r_for_B_subset_C_l2587_258759

open Real Set

-- Define the set A
def A : Set ℝ := {t | 0 < t ∧ t < 2 * π}

-- Define the set B
def B : Set (ℝ × ℝ) := {p | ∃ t ∈ A, p.1 = sin t ∧ p.2 = 2 * sin t * cos t}

-- Define the set C(r)
def C (r : ℝ) : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 ≤ r^2 ∧ r > 0}

-- Theorem statement
theorem smallest_r_for_B_subset_C :
  (∀ r, B ⊆ C r → r ≥ 5/4) ∧ B ⊆ C (5/4) := by sorry

end NUMINAMATH_CALUDE_smallest_r_for_B_subset_C_l2587_258759


namespace NUMINAMATH_CALUDE_bananas_in_basket_e_l2587_258798

/-- Given 5 baskets of fruits with an average of 25 fruits per basket, 
    where basket A contains 15 apples, B has 30 mangoes, C has 20 peaches, 
    D has 25 pears, and E has an unknown number of bananas, 
    prove that basket E contains 35 bananas. -/
theorem bananas_in_basket_e :
  let num_baskets : ℕ := 5
  let avg_fruits_per_basket : ℕ := 25
  let fruits_a : ℕ := 15
  let fruits_b : ℕ := 30
  let fruits_c : ℕ := 20
  let fruits_d : ℕ := 25
  let total_fruits : ℕ := num_baskets * avg_fruits_per_basket
  let fruits_abcd : ℕ := fruits_a + fruits_b + fruits_c + fruits_d
  let fruits_e : ℕ := total_fruits - fruits_abcd
  fruits_e = 35 := by
  sorry

end NUMINAMATH_CALUDE_bananas_in_basket_e_l2587_258798


namespace NUMINAMATH_CALUDE_equation_one_solutions_l2587_258794

theorem equation_one_solutions (x : ℝ) : (x + 2)^2 = 2*x + 4 ↔ x = 0 ∨ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_equation_one_solutions_l2587_258794


namespace NUMINAMATH_CALUDE_bus_rows_theorem_l2587_258772

/-- Represents a school bus with rows of seats split by an aisle -/
structure SchoolBus where
  total_students : ℕ
  students_per_section : ℕ
  sections_per_row : ℕ

/-- Calculates the number of rows in a school bus -/
def num_rows (bus : SchoolBus) : ℕ :=
  (bus.total_students / bus.students_per_section) / bus.sections_per_row

/-- Theorem stating that a bus with 52 students, 2 students per section, and 2 sections per row has 13 rows -/
theorem bus_rows_theorem (bus : SchoolBus) 
  (h1 : bus.total_students = 52)
  (h2 : bus.students_per_section = 2)
  (h3 : bus.sections_per_row = 2) :
  num_rows bus = 13 := by
  sorry

#eval num_rows { total_students := 52, students_per_section := 2, sections_per_row := 2 }

end NUMINAMATH_CALUDE_bus_rows_theorem_l2587_258772


namespace NUMINAMATH_CALUDE_circle_iff_a_eq_neg_one_l2587_258753

/-- Represents a quadratic equation in x and y with parameter a -/
def is_circle (a : ℝ) : Prop :=
  ∃ h k r, ∀ x y : ℝ, 
    a^2 * x^2 + (a + 2) * y^2 + 2*a*x + a = 0 ↔ 
    (x - h)^2 + (y - k)^2 = r^2 ∧ r > 0

/-- The equation represents a circle if and only if a = -1 -/
theorem circle_iff_a_eq_neg_one :
  ∀ a : ℝ, is_circle a ↔ a = -1 := by sorry

end NUMINAMATH_CALUDE_circle_iff_a_eq_neg_one_l2587_258753


namespace NUMINAMATH_CALUDE_garden_ratio_l2587_258729

theorem garden_ratio (area width length : ℝ) : 
  area = 768 →
  width = 16 →
  area = length * width →
  length / width = 3 := by
sorry

end NUMINAMATH_CALUDE_garden_ratio_l2587_258729


namespace NUMINAMATH_CALUDE_inheritance_investment_percentage_l2587_258730

/-- Given an inheritance and investment scenario, prove the unknown investment percentage --/
theorem inheritance_investment_percentage 
  (total_inheritance : ℝ) 
  (known_investment : ℝ) 
  (known_rate : ℝ) 
  (total_interest : ℝ) 
  (h1 : total_inheritance = 4000)
  (h2 : known_investment = 1800)
  (h3 : known_rate = 0.065)
  (h4 : total_interest = 227)
  : ∃ (unknown_rate : ℝ), 
    known_investment * known_rate + (total_inheritance - known_investment) * unknown_rate = total_interest ∧ 
    unknown_rate = 0.05 := by
  sorry


end NUMINAMATH_CALUDE_inheritance_investment_percentage_l2587_258730


namespace NUMINAMATH_CALUDE_total_students_is_880_l2587_258731

/-- The total number of students at the college -/
def total_students : ℕ := 880

/-- The fraction of students enrolled in biology classes -/
def biology_enrollment_rate : ℚ := 35 / 100

/-- The number of students not enrolled in a biology class -/
def students_not_in_biology : ℕ := 572

/-- Theorem stating that the total number of students is 880 -/
theorem total_students_is_880 :
  (1 - biology_enrollment_rate) * total_students = students_not_in_biology :=
sorry

end NUMINAMATH_CALUDE_total_students_is_880_l2587_258731


namespace NUMINAMATH_CALUDE_chandra_reading_pages_l2587_258765

/-- Represents the number of pages in the book -/
def total_pages : ℕ := 900

/-- Represents Chandra's reading speed in seconds per page -/
def chandra_speed : ℕ := 30

/-- Represents Daniel's reading speed in seconds per page -/
def daniel_speed : ℕ := 60

/-- Calculates the number of pages Chandra should read -/
def chandra_pages : ℕ := total_pages * daniel_speed / (chandra_speed + daniel_speed)

theorem chandra_reading_pages :
  chandra_pages = 600 ∧
  chandra_pages * chandra_speed = (total_pages - chandra_pages) * daniel_speed :=
by sorry

end NUMINAMATH_CALUDE_chandra_reading_pages_l2587_258765


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2587_258739

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop := x^2 - 9*x + 18 = 0

-- Define the isosceles triangle
structure IsoscelesTriangle :=
  (base : ℝ)
  (leg : ℝ)
  (base_is_root : quadratic_equation base)
  (leg_is_root : quadratic_equation leg)

-- Theorem statement
theorem isosceles_triangle_perimeter (t : IsoscelesTriangle) : 
  t.base + 2 * t.leg = 15 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2587_258739


namespace NUMINAMATH_CALUDE_f_of_one_equals_two_l2587_258748

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + x

-- State the theorem
theorem f_of_one_equals_two : f 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_of_one_equals_two_l2587_258748


namespace NUMINAMATH_CALUDE_range_of_m_l2587_258761

def p (m : ℝ) : Prop :=
  ∃ (a b : ℝ), a > b ∧ a = 16 - m ∧ b = m - 4 ∧ a > 0 ∧ b > 0

def q (m : ℝ) : Prop :=
  (m - 10)^2 + 3^2 < 13

theorem range_of_m (m : ℝ) :
  (p m ∨ q m) ∧ ¬(p m ∧ q m) →
  (4 < m ∧ m ≤ 8) ∨ (10 ≤ m ∧ m < 12) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l2587_258761


namespace NUMINAMATH_CALUDE_arithmetic_fraction_subtraction_l2587_258782

theorem arithmetic_fraction_subtraction :
  (1 + 3 + 5 + 7) / (2 + 4 + 6 + 8) - (2 + 4 + 6 + 8) / (1 + 3 + 5 + 7) = -9 / 20 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_fraction_subtraction_l2587_258782


namespace NUMINAMATH_CALUDE_largest_prime_sum_under_30_l2587_258779

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def is_sum_of_two_primes (n : ℕ) : Prop :=
  ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ n = p + q

theorem largest_prime_sum_under_30 :
  is_prime 19 ∧
  19 < 30 ∧
  is_sum_of_two_primes 19 ∧
  ∀ n : ℕ, is_prime n → n < 30 → is_sum_of_two_primes n → n ≤ 19 :=
sorry

end NUMINAMATH_CALUDE_largest_prime_sum_under_30_l2587_258779


namespace NUMINAMATH_CALUDE_range_of_x_when_a_is_one_range_of_a_l2587_258755

-- Define the propositions p and q
def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0

def q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0 ∧ x^2 + 3*x - 10 > 0

-- Theorem 1
theorem range_of_x_when_a_is_one (x : ℝ) (h1 : p x 1) (h2 : q x) :
  2 < x ∧ x < 3 := by sorry

-- Theorem 2
theorem range_of_a (a : ℝ) (h : a > 0) 
  (h_suff : ∀ x, ¬(p x a) → ¬(q x))
  (h_not_nec : ∃ x, ¬(q x) ∧ p x a) :
  1 < a ∧ a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_range_of_x_when_a_is_one_range_of_a_l2587_258755


namespace NUMINAMATH_CALUDE_correct_average_l2587_258723

-- Define the number of elements in the set
def n : ℕ := 20

-- Define the initial incorrect average
def incorrect_avg : ℚ := 25.6

-- Define the three pairs of incorrect and correct numbers
def num1 : (ℚ × ℚ) := (57.5, 78.5)
def num2 : (ℚ × ℚ) := (25.25, 35.25)
def num3 : (ℚ × ℚ) := (24.25, 47.5)

-- Define the correct average
def correct_avg : ℚ := 28.3125

-- Theorem statement
theorem correct_average : 
  let incorrect_sum := n * incorrect_avg
  let diff1 := num1.2 - num1.1
  let diff2 := num2.2 - num2.1
  let diff3 := num3.2 - num3.1
  let correct_sum := incorrect_sum + diff1 + diff2 + diff3
  correct_sum / n = correct_avg := by sorry

end NUMINAMATH_CALUDE_correct_average_l2587_258723


namespace NUMINAMATH_CALUDE_wax_calculation_l2587_258735

/-- Given the total required wax and additional wax needed, calculates the amount of wax already possessed. -/
def wax_already_possessed (total_required : ℕ) (additional_needed : ℕ) : ℕ :=
  total_required - additional_needed

/-- Proves that given the specific values in the problem, the wax already possessed is 331 g. -/
theorem wax_calculation :
  let total_required : ℕ := 353
  let additional_needed : ℕ := 22
  wax_already_possessed total_required additional_needed = 331 := by
  sorry

end NUMINAMATH_CALUDE_wax_calculation_l2587_258735


namespace NUMINAMATH_CALUDE_tangency_points_on_sphere_l2587_258774

/-- Represents a sphere in 3D space -/
structure Sphere where
  center : ℝ × ℝ × ℝ
  radius : ℝ

/-- Represents a point in 3D space -/
def Point := ℝ × ℝ × ℝ

/-- Predicate to check if two spheres are tangent -/
def are_tangent (s1 s2 : Sphere) : Prop := sorry

/-- Function to get the tangency point of two spheres -/
def tangency_point (s1 s2 : Sphere) : Point := sorry

/-- Predicate to check if a point lies on a sphere -/
def point_on_sphere (p : Point) (s : Sphere) : Prop := sorry

theorem tangency_points_on_sphere 
  (s1 s2 s3 s4 : Sphere) 
  (h1 : are_tangent s1 s2) (h2 : are_tangent s1 s3) (h3 : are_tangent s1 s4)
  (h4 : are_tangent s2 s3) (h5 : are_tangent s2 s4) (h6 : are_tangent s3 s4) :
  ∃ (s : Sphere), 
    point_on_sphere (tangency_point s1 s2) s ∧
    point_on_sphere (tangency_point s1 s3) s ∧
    point_on_sphere (tangency_point s1 s4) s ∧
    point_on_sphere (tangency_point s2 s3) s ∧
    point_on_sphere (tangency_point s2 s4) s ∧
    point_on_sphere (tangency_point s3 s4) s :=
  sorry

end NUMINAMATH_CALUDE_tangency_points_on_sphere_l2587_258774


namespace NUMINAMATH_CALUDE_sequence_relation_l2587_258726

def x : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 4 * x (n + 1) - x n

def y : ℕ → ℤ
  | 0 => 1
  | 1 => 2
  | (n + 2) => 4 * y (n + 1) - y n

theorem sequence_relation (n : ℕ) : y n ^ 2 = 3 * x n ^ 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_sequence_relation_l2587_258726


namespace NUMINAMATH_CALUDE_rotate_triangle_forms_cone_l2587_258770

/-- A right-angled triangle -/
structure RightTriangle where
  base : ℝ
  height : ℝ
  hypotenuse : ℝ
  right_angle : base^2 + height^2 = hypotenuse^2

/-- A cone -/
structure Cone where
  radius : ℝ
  height : ℝ

/-- The solid formed by rotating a right-angled triangle around one of its right-angle sides -/
def rotateTriangle (t : RightTriangle) : Cone :=
  { radius := t.base, height := t.height }

/-- Theorem: Rotating a right-angled triangle around one of its right-angle sides forms a cone -/
theorem rotate_triangle_forms_cone (t : RightTriangle) :
  ∃ (c : Cone), rotateTriangle t = c :=
sorry

end NUMINAMATH_CALUDE_rotate_triangle_forms_cone_l2587_258770


namespace NUMINAMATH_CALUDE_stream_speed_l2587_258742

/-- The speed of a stream given boat travel times and distances -/
theorem stream_speed (downstream_distance upstream_distance : ℝ) 
  (time : ℝ) (h1 : downstream_distance = 84) (h2 : upstream_distance = 48) 
  (h3 : time = 2) : ∃ (stream_speed : ℝ), stream_speed = 9 ∧ 
  ∃ (boat_speed : ℝ), 
    downstream_distance = (boat_speed + stream_speed) * time ∧
    upstream_distance = (boat_speed - stream_speed) * time :=
by
  sorry


end NUMINAMATH_CALUDE_stream_speed_l2587_258742

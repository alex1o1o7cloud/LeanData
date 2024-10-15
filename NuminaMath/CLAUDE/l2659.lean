import Mathlib

namespace NUMINAMATH_CALUDE_number_of_girls_in_school_l2659_265991

/-- Represents the number of students in a section -/
def SectionSize : ℕ := 24

/-- Represents the total number of boys in the school -/
def TotalBoys : ℕ := 408

/-- Represents the total number of sections -/
def TotalSections : ℕ := 26

/-- Represents the number of sections for boys -/
def BoySections : ℕ := 17

/-- Represents the number of sections for girls -/
def GirlSections : ℕ := 9

/-- Theorem stating the number of girls in the school -/
theorem number_of_girls_in_school : 
  TotalBoys / BoySections = SectionSize ∧ 
  BoySections + GirlSections = TotalSections → 
  GirlSections * SectionSize = 216 :=
by sorry

end NUMINAMATH_CALUDE_number_of_girls_in_school_l2659_265991


namespace NUMINAMATH_CALUDE_abs_five_minus_sqrt_two_l2659_265980

theorem abs_five_minus_sqrt_two : |5 - Real.sqrt 2| = 5 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_five_minus_sqrt_two_l2659_265980


namespace NUMINAMATH_CALUDE_fraction_equality_l2659_265993

theorem fraction_equality (a b c d : ℝ) 
  (h : (a - b) * (c - d) / ((b - c) * (d - a)) = 3 / 4) : 
  (a - c) * (b - d) / ((a - b) * (c - d)) = 1 := by sorry

end NUMINAMATH_CALUDE_fraction_equality_l2659_265993


namespace NUMINAMATH_CALUDE_min_value_of_f_l2659_265944

-- Define the function
def f (x : ℝ) : ℝ := 3 * x^2 + 6 * x + 2

-- State the theorem
theorem min_value_of_f :
  ∃ (m : ℝ), (∀ x, f x ≥ m) ∧ (∃ x₀, f x₀ = m) ∧ m = -1 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2659_265944


namespace NUMINAMATH_CALUDE_chocolate_purchase_l2659_265998

theorem chocolate_purchase (boxes_bought : ℕ) (price_per_box : ℚ) (boxes_given : ℕ) 
  (pieces_per_box : ℕ) (discount_percent : ℚ) : 
  boxes_bought = 12 → 
  price_per_box = 4 → 
  boxes_given = 7 → 
  pieces_per_box = 6 → 
  discount_percent = 15 / 100 → 
  ∃ (amount_paid : ℚ) (pieces_remaining : ℕ), 
    amount_paid = 40.80 ∧ 
    pieces_remaining = 30 :=
by sorry

end NUMINAMATH_CALUDE_chocolate_purchase_l2659_265998


namespace NUMINAMATH_CALUDE_poly_expansions_general_poly_expansion_possible_m_values_l2659_265928

-- Define the polynomial expressions
def poly1 (x : ℝ) := (x + 2) * (x + 3)
def poly2 (x : ℝ) := (x + 2) * (x - 3)
def poly3 (x : ℝ) := (x - 2) * (x + 3)
def poly4 (x : ℝ) := (x - 2) * (x - 3)

-- Define the general polynomial expression
def polyGeneral (x a b : ℝ) := (x + a) * (x + b)

-- Theorem statements
theorem poly_expansions :
  (∀ x : ℝ, poly1 x = x^2 + 5*x + 6) ∧
  (∀ x : ℝ, poly2 x = x^2 - x - 6) ∧
  (∀ x : ℝ, poly3 x = x^2 + x - 6) ∧
  (∀ x : ℝ, poly4 x = x^2 - 5*x + 6) :=
sorry

theorem general_poly_expansion :
  ∀ x a b : ℝ, polyGeneral x a b = x^2 + (a + b)*x + a*b :=
sorry

theorem possible_m_values :
  ∀ a b m : ℤ, (∀ x : ℝ, polyGeneral x (a : ℝ) (b : ℝ) = x^2 + m*x + 5) →
  (m = 6 ∨ m = -6) :=
sorry

end NUMINAMATH_CALUDE_poly_expansions_general_poly_expansion_possible_m_values_l2659_265928


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l2659_265995

/-- The atomic weight of Hydrogen in g/mol -/
def hydrogen_weight : ℝ := 1.01

/-- The atomic weight of Bromine in g/mol -/
def bromine_weight : ℝ := 79.90

/-- The atomic weight of Oxygen in g/mol -/
def oxygen_weight : ℝ := 16.00

/-- The number of Hydrogen atoms in the compound -/
def hydrogen_count : ℕ := 1

/-- The number of Bromine atoms in the compound -/
def bromine_count : ℕ := 1

/-- The number of Oxygen atoms in the compound -/
def oxygen_count : ℕ := 3

/-- The molecular weight of the compound in g/mol -/
def molecular_weight : ℝ :=
  hydrogen_count * hydrogen_weight +
  bromine_count * bromine_weight +
  oxygen_count * oxygen_weight

/-- Theorem stating that the molecular weight of the compound is 128.91 g/mol -/
theorem compound_molecular_weight : molecular_weight = 128.91 := by
  sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l2659_265995


namespace NUMINAMATH_CALUDE_cylinder_volume_from_rectangle_l2659_265948

/-- The volume of a cylinder formed by rotating a rectangle about its longer side -/
theorem cylinder_volume_from_rectangle (length width : ℝ) (length_ge_width : length ≥ width) :
  let radius := length / 2
  let height := width
  let volume := π * radius^2 * height
  length = 20 ∧ width = 10 → volume = 1000 * π := by
  sorry

end NUMINAMATH_CALUDE_cylinder_volume_from_rectangle_l2659_265948


namespace NUMINAMATH_CALUDE_computer_upgrade_cost_l2659_265915

/-- Calculates the total money spent on a computer after replacing a video card -/
def totalSpent (initialCost oldCardSale newCardPrice : ℕ) : ℕ :=
  initialCost + (newCardPrice - oldCardSale)

theorem computer_upgrade_cost :
  totalSpent 1200 300 500 = 1400 := by
  sorry

end NUMINAMATH_CALUDE_computer_upgrade_cost_l2659_265915


namespace NUMINAMATH_CALUDE_min_c_value_l2659_265901

theorem min_c_value (a b c : ℕ+) (h1 : a ≤ b) (h2 : b < c)
  (h3 : ∃! p : ℝ × ℝ, (2 * p.1 + p.2 = 2023) ∧
    (p.2 = |p.1 - a.val| + |p.1 - b.val| + |p.1 - c.val|)) :
  c.val ≥ 2022 ∧ ∃ a b : ℕ+, a ≤ b ∧ b < 2022 ∧
    ∃! p : ℝ × ℝ, (2 * p.1 + p.2 = 2023) ∧
      (p.2 = |p.1 - a.val| + |p.1 - b.val| + |p.1 - 2022|) := by
  sorry

end NUMINAMATH_CALUDE_min_c_value_l2659_265901


namespace NUMINAMATH_CALUDE_students_allowance_l2659_265983

theorem students_allowance (allowance : ℚ) : 
  (allowance > 0) →
  (3 / 5 * allowance + 1 / 3 * (2 / 5 * allowance) + 60 / 100 = allowance) →
  allowance = 225 / 100 := by
sorry

end NUMINAMATH_CALUDE_students_allowance_l2659_265983


namespace NUMINAMATH_CALUDE_rectangle_ratio_l2659_265937

theorem rectangle_ratio (w : ℚ) : 
  w > 0 ∧ 2 * w + 2 * 10 = 30 → w / 10 = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_rectangle_ratio_l2659_265937


namespace NUMINAMATH_CALUDE_biathlon_bicycle_distance_l2659_265924

/-- Given a biathlon with specified conditions, prove the distance of the bicycle race. -/
theorem biathlon_bicycle_distance 
  (total_distance : ℝ) 
  (total_time : ℝ) 
  (run_distance : ℝ) 
  (run_velocity : ℝ) :
  total_distance = 155 →
  total_time = 6 →
  run_distance = 10 →
  run_velocity = 10 →
  total_distance = run_distance + (total_time - run_distance / run_velocity) * 
    ((total_distance - run_distance) / (total_time - run_distance / run_velocity)) →
  total_distance - run_distance = 145 := by
  sorry

#check biathlon_bicycle_distance

end NUMINAMATH_CALUDE_biathlon_bicycle_distance_l2659_265924


namespace NUMINAMATH_CALUDE_power_sum_l2659_265947

theorem power_sum (a m n : ℝ) (h1 : a^m = 2) (h2 : a^n = 3) : a^(m+n) = 6 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_l2659_265947


namespace NUMINAMATH_CALUDE_necessary_and_sufficient_condition_l2659_265920

theorem necessary_and_sufficient_condition (a : ℝ) :
  let f := fun x => x * (x - a) * (x - 2)
  let f' := fun x => 3 * x^2 - 2 * (a + 2) * x + 2 * a
  (0 < a ∧ a < 2) ↔ f' a < 0 := by
  sorry

end NUMINAMATH_CALUDE_necessary_and_sufficient_condition_l2659_265920


namespace NUMINAMATH_CALUDE_half_to_fourth_power_l2659_265943

theorem half_to_fourth_power : (1/2 : ℚ)^4 = 1/16 := by
  sorry

end NUMINAMATH_CALUDE_half_to_fourth_power_l2659_265943


namespace NUMINAMATH_CALUDE_lcm_factor_problem_l2659_265974

theorem lcm_factor_problem (A B : ℕ+) (hcf other_factor : ℕ+) :
  hcf = 23 →
  A = 345 →
  Nat.lcm A B = hcf * other_factor * 15 →
  other_factor = 23 := by
  sorry

end NUMINAMATH_CALUDE_lcm_factor_problem_l2659_265974


namespace NUMINAMATH_CALUDE_total_crayons_l2659_265945

theorem total_crayons (people : ℕ) (crayons_per_person : ℕ) (h1 : people = 3) (h2 : crayons_per_person = 8) :
  people * crayons_per_person = 24 := by
  sorry

end NUMINAMATH_CALUDE_total_crayons_l2659_265945


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l2659_265946

/-- The repeating decimal 0.363636... expressed as a real number -/
def repeating_decimal : ℚ := 0.363636

/-- Theorem stating that the repeating decimal 0.363636... is equal to 4/11 -/
theorem repeating_decimal_equals_fraction : repeating_decimal = 4 / 11 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l2659_265946


namespace NUMINAMATH_CALUDE_square_divisors_count_l2659_265996

-- Define a function to count divisors
def count_divisors (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem square_divisors_count (n : ℕ) : 
  count_divisors n = 4 → count_divisors (n^2) = 7 := by sorry

end NUMINAMATH_CALUDE_square_divisors_count_l2659_265996


namespace NUMINAMATH_CALUDE_minimum_cards_to_turn_l2659_265999

/-- Represents a card with a letter on one side and a number on the other -/
structure Card where
  letter : Char
  number : Nat

/-- Checks if a character is a vowel -/
def isVowel (c : Char) : Bool :=
  c ∈ ['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U']

/-- Checks if a number is even -/
def isEven (n : Nat) : Bool :=
  n % 2 = 0

/-- Checks if a card satisfies the condition: 
    if it has a vowel, it must have an even number -/
def satisfiesCondition (card : Card) : Bool :=
  ¬(isVowel card.letter) || isEven card.number

/-- Represents the set of cards on the table -/
def cardSet : Finset Card := sorry

/-- The number of cards that need to be turned over -/
def cardsToTurn : Nat := sorry

theorem minimum_cards_to_turn : 
  (∀ card ∈ cardSet, satisfiesCondition card) → cardsToTurn = 3 := by
  sorry

end NUMINAMATH_CALUDE_minimum_cards_to_turn_l2659_265999


namespace NUMINAMATH_CALUDE_product_eleven_reciprocal_squares_sum_l2659_265972

theorem product_eleven_reciprocal_squares_sum (a b : ℕ+) :
  a * b = 11 → (1 : ℚ) / a^2 + (1 : ℚ) / b^2 = 122 / 121 := by
  sorry

end NUMINAMATH_CALUDE_product_eleven_reciprocal_squares_sum_l2659_265972


namespace NUMINAMATH_CALUDE_nigella_commission_rate_l2659_265934

/-- Represents a realtor's earnings and house sales --/
structure RealtorSales where
  baseSalary : ℕ
  totalEarnings : ℕ
  houseACost : ℕ
  houseBCost : ℕ
  houseCCost : ℕ

/-- Calculates the commission rate for a realtor given their sales data --/
def commissionRate (sales : RealtorSales) : ℚ :=
  let totalHouseCost := sales.houseACost + sales.houseBCost + sales.houseCCost
  let commission := sales.totalEarnings - sales.baseSalary
  (commission : ℚ) / totalHouseCost

/-- Theorem stating that given the conditions from the problem, the commission rate is 2% --/
theorem nigella_commission_rate :
  let sales : RealtorSales := {
    baseSalary := 3000,
    totalEarnings := 8000,
    houseACost := 60000,
    houseBCost := 3 * 60000,
    houseCCost := 2 * 60000 - 110000
  }
  commissionRate sales = 1/50 := by sorry

end NUMINAMATH_CALUDE_nigella_commission_rate_l2659_265934


namespace NUMINAMATH_CALUDE_line_intersection_area_ratio_l2659_265949

theorem line_intersection_area_ratio (c : ℝ) (h1 : 0 < c) (h2 : c < 6) : 
  let P : ℝ × ℝ := (0, c)
  let Q : ℝ × ℝ := (c, 0)
  let S : ℝ × ℝ := (6, c - 6)
  let area_QRS := (1/2) * (6 - c) * (c - 6)
  let area_QOP := (1/2) * c * c
  area_QRS / area_QOP = 4/25 → c = 30/7 := by
sorry

end NUMINAMATH_CALUDE_line_intersection_area_ratio_l2659_265949


namespace NUMINAMATH_CALUDE_largest_class_size_l2659_265940

/-- Proves that in a school with 5 classes, where each class has 2 students less than the previous class,
    and the total number of students is 115, the largest class has 27 students. -/
theorem largest_class_size (total_students : ℕ) (num_classes : ℕ) (diff : ℕ) :
  total_students = 115 →
  num_classes = 5 →
  diff = 2 →
  ∃ (x : ℕ), x = 27 ∧ 
    (x + (x - diff) + (x - 2*diff) + (x - 3*diff) + (x - 4*diff) = total_students) :=
by sorry

end NUMINAMATH_CALUDE_largest_class_size_l2659_265940


namespace NUMINAMATH_CALUDE_divisibility_condition_l2659_265912

theorem divisibility_condition (n : ℕ+) :
  (∃ m : ℤ, (2^n.val - 1) ∣ (m^2 + 9)) ↔ ∃ x : ℕ, n = 2^x :=
sorry

end NUMINAMATH_CALUDE_divisibility_condition_l2659_265912


namespace NUMINAMATH_CALUDE_point_translation_rotation_l2659_265918

/-- Represents a point in 2D Cartesian coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translates a point horizontally -/
def translate (p : Point) (dx : ℝ) : Point :=
  ⟨p.x + dx, p.y⟩

/-- Rotates a point 90 degrees clockwise around the origin -/
def rotate90Clockwise (p : Point) : Point :=
  ⟨p.y, -p.x⟩

theorem point_translation_rotation (p : Point) :
  p = ⟨-5, 4⟩ →
  (rotate90Clockwise (translate p 8)) = ⟨4, -3⟩ := by
  sorry

end NUMINAMATH_CALUDE_point_translation_rotation_l2659_265918


namespace NUMINAMATH_CALUDE_files_per_folder_l2659_265979

theorem files_per_folder (initial_files : ℕ) (deleted_files : ℕ) (num_folders : ℕ) 
  (h1 : initial_files = 27)
  (h2 : deleted_files = 9)
  (h3 : num_folders = 3)
  (h4 : num_folders > 0) :
  (initial_files - deleted_files) / num_folders = 6 := by
  sorry

end NUMINAMATH_CALUDE_files_per_folder_l2659_265979


namespace NUMINAMATH_CALUDE_cookies_per_bag_l2659_265938

theorem cookies_per_bag (chocolate_chip : ℕ) (oatmeal : ℕ) (baggies : ℕ) :
  chocolate_chip = 13 →
  oatmeal = 41 →
  baggies = 6 →
  (chocolate_chip + oatmeal) / baggies = 9 := by
sorry

end NUMINAMATH_CALUDE_cookies_per_bag_l2659_265938


namespace NUMINAMATH_CALUDE_quadratic_root_relation_l2659_265916

theorem quadratic_root_relation : ∀ x₁ x₂ : ℝ, 
  x₁^2 - 12*x₁ + 5 = 0 → 
  x₂^2 - 12*x₂ + 5 = 0 → 
  x₁ + x₂ - x₁*x₂ = 7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_relation_l2659_265916


namespace NUMINAMATH_CALUDE_coefficient_x4_in_expansion_l2659_265913

theorem coefficient_x4_in_expansion (x : ℝ) : 
  ∃ (a b c d e f : ℝ), (2*x + 1) * (x - 1)^5 = a*x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + f ∧ b = 15 :=
sorry

end NUMINAMATH_CALUDE_coefficient_x4_in_expansion_l2659_265913


namespace NUMINAMATH_CALUDE_steve_nickels_l2659_265977

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The total value of coins in cents -/
def total_value : ℕ := 70

/-- Proves that Steve is holding 2 nickels -/
theorem steve_nickels :
  ∃ (n : ℕ), 
    (n * nickel_value + (n + 4) * dime_value = total_value) ∧
    (n = 2) := by
  sorry

end NUMINAMATH_CALUDE_steve_nickels_l2659_265977


namespace NUMINAMATH_CALUDE_car_speeds_l2659_265903

theorem car_speeds (distance : ℝ) (time_difference : ℝ) (arrival_difference : ℝ) 
  (speed_ratio_small : ℝ) (speed_ratio_large : ℝ) 
  (h1 : distance = 135)
  (h2 : time_difference = 4)
  (h3 : arrival_difference = 1/2)
  (h4 : speed_ratio_small = 5)
  (h5 : speed_ratio_large = 2) :
  ∃ (speed_small : ℝ) (speed_large : ℝ),
    speed_small = 45 ∧ 
    speed_large = 18 ∧
    speed_small / speed_large = speed_ratio_small / speed_ratio_large ∧
    distance / speed_small = distance / speed_large - time_difference - arrival_difference :=
by
  sorry

end NUMINAMATH_CALUDE_car_speeds_l2659_265903


namespace NUMINAMATH_CALUDE_cube_triangle_areas_sum_l2659_265969

/-- Represents a 2x2x2 cube -/
structure Cube :=
  (side_length : ℝ)
  (is_2x2x2 : side_length = 2)

/-- Represents a triangle with vertices from the cube -/
structure CubeTriangle :=
  (vertices : Fin 3 → Fin 8)

/-- The area of a triangle given its side lengths -/
noncomputable def triangle_area (a b c : ℝ) : ℝ := sorry

/-- The sum of areas of all triangles in the cube -/
noncomputable def sum_of_triangle_areas (cube : Cube) : ℝ := sorry

/-- The main theorem -/
theorem cube_triangle_areas_sum (cube : Cube) :
  ∃ (m n p : ℕ), 
    sum_of_triangle_areas cube = m + Real.sqrt n + Real.sqrt p ∧
    m + n + p = 121 := by sorry

end NUMINAMATH_CALUDE_cube_triangle_areas_sum_l2659_265969


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l2659_265927

theorem sufficient_not_necessary :
  (∀ x : ℝ, (x + 1) * (x - 3) < 0 → x > -1) ∧
  (∃ x : ℝ, x > -1 ∧ (x + 1) * (x - 3) ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l2659_265927


namespace NUMINAMATH_CALUDE_student_distribution_l2659_265988

theorem student_distribution (total : ℕ) (schemes : ℕ) : 
  total = 7 → 
  schemes = 108 → 
  (∃ (boys girls : ℕ), 
    boys + girls = total ∧ 
    boys * Nat.choose girls 2 * 6 = schemes ∧
    boys = 3 ∧ 
    girls = 4) :=
by sorry

end NUMINAMATH_CALUDE_student_distribution_l2659_265988


namespace NUMINAMATH_CALUDE_max_sum_of_rolls_l2659_265997

def is_valid_roll_set (rolls : List Nat) : Prop :=
  rolls.length = 24 ∧
  (∀ n : Nat, n ≥ 1 ∧ n ≤ 6 → n ∈ rolls) ∧
  (∀ n : Nat, n ≥ 2 ∧ n ≤ 6 → rolls.count 1 > rolls.count n)

def sum_of_rolls (rolls : List Nat) : Nat :=
  rolls.sum

theorem max_sum_of_rolls :
  ∀ rolls : List Nat,
    is_valid_roll_set rolls →
    sum_of_rolls rolls ≤ 90 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_rolls_l2659_265997


namespace NUMINAMATH_CALUDE_carlotta_performance_length_l2659_265962

/-- Represents the length of Carlotta's final stage performance in minutes -/
def performance_length : ℝ := 6

/-- For every minute of singing, Carlotta spends 3 minutes practicing -/
def practice_ratio : ℝ := 3

/-- For every minute of singing, Carlotta spends 5 minutes throwing tantrums -/
def tantrum_ratio : ℝ := 5

/-- The total combined time of singing, practicing, and throwing tantrums in minutes -/
def total_time : ℝ := 54

theorem carlotta_performance_length :
  performance_length * (1 + practice_ratio + tantrum_ratio) = total_time :=
sorry

end NUMINAMATH_CALUDE_carlotta_performance_length_l2659_265962


namespace NUMINAMATH_CALUDE_pta_spending_ratio_l2659_265917

theorem pta_spending_ratio (initial_amount : ℚ) (spent_on_supplies : ℚ) (amount_left : ℚ) 
  (h1 : initial_amount = 400)
  (h2 : amount_left = 150)
  (h3 : amount_left = initial_amount - spent_on_supplies - (initial_amount - spent_on_supplies) / 2) :
  spent_on_supplies = 100 ∧ spent_on_supplies / initial_amount = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_pta_spending_ratio_l2659_265917


namespace NUMINAMATH_CALUDE_jogger_train_distance_l2659_265941

/-- Calculates the distance a jogger is ahead of a train given their speeds and the time it takes for the train to pass the jogger. -/
theorem jogger_train_distance (jogger_speed : ℝ) (train_speed : ℝ) (train_length : ℝ) (passing_time : ℝ) : 
  jogger_speed = 10 * (5/18) → 
  train_speed = 46 * (5/18) → 
  train_length = 120 → 
  passing_time = 46 → 
  (train_speed - jogger_speed) * passing_time - train_length = 340 := by
  sorry

#check jogger_train_distance

end NUMINAMATH_CALUDE_jogger_train_distance_l2659_265941


namespace NUMINAMATH_CALUDE_smallest_difference_is_six_l2659_265933

/-- The set of available digits --/
def digits : Finset Nat := {0, 1, 2, 6, 9}

/-- A function to create a three-digit number from three digits --/
def makeThreeDigitNumber (x y z : Nat) : Nat := 100 * x + 10 * y + z

/-- A function to create a two-digit number from two digits --/
def makeTwoDigitNumber (u v : Nat) : Nat := 10 * u + v

/-- The theorem statement --/
theorem smallest_difference_is_six :
  ∀ x y z u v : Nat,
    x ∈ digits → y ∈ digits → z ∈ digits → u ∈ digits → v ∈ digits →
    x ≠ y → x ≠ z → x ≠ u → x ≠ v →
    y ≠ z → y ≠ u → y ≠ v →
    z ≠ u → z ≠ v →
    u ≠ v →
    x ≠ 0 → u ≠ 0 →
    makeThreeDigitNumber x y z - makeTwoDigitNumber u v ≥ 6 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_difference_is_six_l2659_265933


namespace NUMINAMATH_CALUDE_largest_divisor_of_difference_of_cubes_squared_l2659_265986

theorem largest_divisor_of_difference_of_cubes_squared (k : ℤ) : 
  ∃ (d : ℤ), d = 16 ∧ 
  d ∣ (((2*k+1)^3)^2 - ((2*k-1)^3)^2) ∧ 
  ∀ (n : ℤ), n > d → ¬(∀ (j : ℤ), n ∣ (((2*j+1)^3)^2 - ((2*j-1)^3)^2)) :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_difference_of_cubes_squared_l2659_265986


namespace NUMINAMATH_CALUDE_square_area_in_circle_l2659_265978

theorem square_area_in_circle (r : ℝ) (h : r = 10) : 
  let s := r * Real.sqrt 2
  let small_square_side := r / Real.sqrt 2
  let center_distance := s / 2
  2 * center_distance^2 = 100 := by sorry

end NUMINAMATH_CALUDE_square_area_in_circle_l2659_265978


namespace NUMINAMATH_CALUDE_max_real_part_sum_l2659_265989

theorem max_real_part_sum (z : Fin 18 → ℂ) (w : Fin 18 → ℂ) : 
  (∀ j : Fin 18, z j ^ 18 = (2 : ℂ) ^ 54) →
  (∀ j : Fin 18, w j = z j ∨ w j = Complex.I * z j ∨ w j = -z j) →
  (∃ w_choice : Fin 18 → ℂ, 
    (∀ j : Fin 18, w_choice j = z j ∨ w_choice j = Complex.I * z j ∨ w_choice j = -z j) ∧
    (Finset.sum Finset.univ (λ j => (w_choice j).re) = 
      8 + 8 * (2 * (1 + Real.sqrt 3 + Real.sqrt 2 + 
        Real.cos (π / 9) + Real.cos (2 * π / 9) + Real.cos (4 * π / 9) + 
        Real.cos (5 * π / 9) + Real.cos (7 * π / 9) + Real.cos (8 * π / 9))))) ∧
  (∀ w_alt : Fin 18 → ℂ, 
    (∀ j : Fin 18, w_alt j = z j ∨ w_alt j = Complex.I * z j ∨ w_alt j = -z j) →
    Finset.sum Finset.univ (λ j => (w_alt j).re) ≤ 
      8 + 8 * (2 * (1 + Real.sqrt 3 + Real.sqrt 2 + 
        Real.cos (π / 9) + Real.cos (2 * π / 9) + Real.cos (4 * π / 9) + 
        Real.cos (5 * π / 9) + Real.cos (7 * π / 9) + Real.cos (8 * π / 9)))) := by
  sorry

end NUMINAMATH_CALUDE_max_real_part_sum_l2659_265989


namespace NUMINAMATH_CALUDE_function_property_l2659_265906

/-- Given a function f(x) = ax² - bx where a and b are positive constants,
    if f(f(1)) = -1 and √(ab) = 3, then a = 1 or a = 2. -/
theorem function_property (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  let f : ℝ → ℝ := λ x => a * x^2 - b * x
  (f (f 1) = -1) ∧ (Real.sqrt (a * b) = 3) → a = 1 ∨ a = 2 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l2659_265906


namespace NUMINAMATH_CALUDE_polynomial_factor_implies_coefficients_l2659_265973

theorem polynomial_factor_implies_coefficients 
  (p q : ℝ) 
  (h : ∃ (a b c : ℝ), px^4 + qx^3 + 40*x^2 - 24*x + 9 = (4*x^2 - 3*x + 2) * (a*x^2 + b*x + c)) :
  p = 12.5 ∧ q = -30.375 := by
sorry

end NUMINAMATH_CALUDE_polynomial_factor_implies_coefficients_l2659_265973


namespace NUMINAMATH_CALUDE_isabella_hair_growth_l2659_265966

/-- Calculates hair growth given initial and final hair lengths -/
def hair_growth (initial_length final_length : ℝ) : ℝ :=
  final_length - initial_length

theorem isabella_hair_growth :
  let initial_length : ℝ := 18
  let final_length : ℝ := 24
  hair_growth initial_length final_length = 6 := by
  sorry

end NUMINAMATH_CALUDE_isabella_hair_growth_l2659_265966


namespace NUMINAMATH_CALUDE_box_number_equation_l2659_265931

theorem box_number_equation (x : ℝ) : 
  (x > 0 ∧ 8 + 7 / x + 3 / 1000 = 8.073) ↔ x = 100 := by
sorry

end NUMINAMATH_CALUDE_box_number_equation_l2659_265931


namespace NUMINAMATH_CALUDE_power_sum_inequality_l2659_265953

theorem power_sum_inequality (k l m : ℕ) :
  2^(k+l) + 2^(k+m) + 2^(l+m) ≤ 2^(k+l+m+1) + 1 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_inequality_l2659_265953


namespace NUMINAMATH_CALUDE_circle_radius_with_chord_l2659_265985

/-- The radius of a circle given specific conditions --/
theorem circle_radius_with_chord (r : ℝ) : 
  (∃ (A B : ℝ × ℝ), 
    -- Line equation
    (A.1 - Real.sqrt 3 * A.2 + 8 = 0) ∧ 
    (B.1 - Real.sqrt 3 * B.2 + 8 = 0) ∧
    -- Circle equation
    (A.1^2 + A.2^2 = r^2) ∧ 
    (B.1^2 + B.2^2 = r^2) ∧
    -- Length of chord AB
    ((A.1 - B.1)^2 + (A.2 - B.2)^2 = 36)) → 
  r = 5 := by
sorry


end NUMINAMATH_CALUDE_circle_radius_with_chord_l2659_265985


namespace NUMINAMATH_CALUDE_distance_proof_l2659_265919

/-- Proves that the distance between two points is 2 km given specific travel conditions -/
theorem distance_proof (T : ℝ) : 
  (4 * (T + 7/60) = 8 * (T - 8/60)) → 
  (4 * (T + 7/60) = 2) := by
  sorry

end NUMINAMATH_CALUDE_distance_proof_l2659_265919


namespace NUMINAMATH_CALUDE_lara_chips_count_l2659_265990

theorem lara_chips_count :
  ∀ (total_chips : ℕ),
  (total_chips / 6 : ℚ) + 34 + 16 = total_chips →
  total_chips = 60 := by
sorry

end NUMINAMATH_CALUDE_lara_chips_count_l2659_265990


namespace NUMINAMATH_CALUDE_haley_magazines_l2659_265958

theorem haley_magazines (boxes : ℕ) (magazines_per_box : ℕ) 
  (h1 : boxes = 7) (h2 : magazines_per_box = 9) : 
  boxes * magazines_per_box = 63 := by
  sorry

end NUMINAMATH_CALUDE_haley_magazines_l2659_265958


namespace NUMINAMATH_CALUDE_mouse_seeds_count_l2659_265963

/-- Represents the number of seeds per burrow for the mouse -/
def mouse_seeds_per_burrow : ℕ := 4

/-- Represents the number of seeds per burrow for the rabbit -/
def rabbit_seeds_per_burrow : ℕ := 7

/-- Represents the difference in number of burrows between mouse and rabbit -/
def burrow_difference : ℕ := 3

theorem mouse_seeds_count (mouse_burrows rabbit_burrows : ℕ) 
  (h1 : mouse_burrows = rabbit_burrows + burrow_difference)
  (h2 : mouse_seeds_per_burrow * mouse_burrows = rabbit_seeds_per_burrow * rabbit_burrows) :
  mouse_seeds_per_burrow * mouse_burrows = 28 := by
  sorry

end NUMINAMATH_CALUDE_mouse_seeds_count_l2659_265963


namespace NUMINAMATH_CALUDE_land_allocation_equations_l2659_265982

/-- Represents the land allocation problem for tea gardens and grain fields. -/
theorem land_allocation_equations (total_area : ℝ) (vegetable_percentage : ℝ) 
  (tea_grain_area : ℝ) (tea_area : ℝ) (grain_area : ℝ) : 
  total_area = 60 ∧ 
  vegetable_percentage = 0.1 ∧ 
  tea_grain_area = total_area - vegetable_percentage * total_area ∧
  tea_area = 2 * grain_area - 3 →
  tea_area + grain_area = 54 ∧ tea_area = 2 * grain_area - 3 :=
by sorry

end NUMINAMATH_CALUDE_land_allocation_equations_l2659_265982


namespace NUMINAMATH_CALUDE_chord_bisected_by_point_4_2_l2659_265975

/-- The equation of an ellipse -/
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 36 + y^2 / 9 = 1

/-- A point is the midpoint of two other points -/
def is_midpoint (x y x1 y1 x2 y2 : ℝ) : Prop :=
  x = (x1 + x2) / 2 ∧ y = (y1 + y2) / 2

/-- A point lies on a line -/
def point_on_line (x y : ℝ) : Prop := x + 2*y - 8 = 0

theorem chord_bisected_by_point_4_2 (x1 y1 x2 y2 : ℝ) :
  is_on_ellipse x1 y1 →
  is_on_ellipse x2 y2 →
  is_midpoint 4 2 x1 y1 x2 y2 →
  point_on_line x1 y1 ∧ point_on_line x2 y2 :=
sorry

end NUMINAMATH_CALUDE_chord_bisected_by_point_4_2_l2659_265975


namespace NUMINAMATH_CALUDE_compound_interest_rate_calculation_l2659_265964

/-- Compound interest rate calculation -/
theorem compound_interest_rate_calculation
  (P : ℝ) (A : ℝ) (t : ℝ) (n : ℝ)
  (h_P : P = 12000)
  (h_A : A = 15200)
  (h_t : t = 7)
  (h_n : n = 1)
  : ∃ r : ℝ, (A = P * (1 + r / n) ^ (n * t)) ∧ (abs (r - 0.0332) < 0.0001) :=
sorry

end NUMINAMATH_CALUDE_compound_interest_rate_calculation_l2659_265964


namespace NUMINAMATH_CALUDE_no_solutions_absolute_value_equation_l2659_265908

theorem no_solutions_absolute_value_equation :
  ¬ ∃ x : ℝ, |x - 2| = |x - 1| + |x - 5| := by
sorry

end NUMINAMATH_CALUDE_no_solutions_absolute_value_equation_l2659_265908


namespace NUMINAMATH_CALUDE_correct_num_cats_l2659_265942

/-- Represents the number of cats on the ship -/
def num_cats : ℕ := 5

/-- Represents the number of sailors on the ship -/
def num_sailors : ℕ := 14 - num_cats

/-- The total number of heads on the ship -/
def total_heads : ℕ := 16

/-- The total number of legs on the ship -/
def total_legs : ℕ := 41

/-- Theorem stating that the number of cats is correct given the conditions -/
theorem correct_num_cats : 
  num_cats + num_sailors + 2 = total_heads ∧ 
  4 * num_cats + 2 * num_sailors + 3 = total_legs :=
by sorry

end NUMINAMATH_CALUDE_correct_num_cats_l2659_265942


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2659_265909

theorem quadratic_equation_roots (m : ℝ) : 
  (∃ x : ℝ, x^2 - 6*x + m = 0 ∧ x = 2) → 
  (∃ y : ℝ, y^2 - 6*y + m = 0 ∧ y = 4) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2659_265909


namespace NUMINAMATH_CALUDE_randy_blocks_theorem_l2659_265950

/-- The number of blocks Randy used for the tower -/
def blocks_used : ℕ := 19

/-- The number of blocks Randy has left -/
def blocks_left : ℕ := 59

/-- The initial number of blocks Randy had -/
def initial_blocks : ℕ := blocks_used + blocks_left

theorem randy_blocks_theorem : initial_blocks = 78 := by
  sorry

end NUMINAMATH_CALUDE_randy_blocks_theorem_l2659_265950


namespace NUMINAMATH_CALUDE_complex_power_150_deg_40_l2659_265932

-- Define DeMoivre's Theorem
axiom deMoivre (θ : ℝ) (n : ℕ) : (Complex.exp (θ * Complex.I)) ^ n = Complex.exp (n * θ * Complex.I)

-- Define the problem
theorem complex_power_150_deg_40 :
  (Complex.exp (150 * π / 180 * Complex.I)) ^ 40 = -1/2 - Complex.I * (Real.sqrt 3 / 2) :=
sorry

end NUMINAMATH_CALUDE_complex_power_150_deg_40_l2659_265932


namespace NUMINAMATH_CALUDE_quadratic_rewrite_ratio_l2659_265959

/-- Given a quadratic expression of the form ak² + bk + d, 
    rewrite it as c(k + p)² + q and return (c, p, q) -/
def rewrite_quadratic (a b d : ℚ) : ℚ × ℚ × ℚ := sorry

theorem quadratic_rewrite_ratio : 
  let (c, p, q) := rewrite_quadratic 8 (-12) 20
  q / p = -62 / 3 := by sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_ratio_l2659_265959


namespace NUMINAMATH_CALUDE_remaining_money_proof_l2659_265965

def salary : ℚ := 190000

def food_fraction : ℚ := 1/5
def rent_fraction : ℚ := 1/10
def clothes_fraction : ℚ := 3/5

def remaining_amount : ℚ := salary * (1 - (food_fraction + rent_fraction + clothes_fraction))

theorem remaining_money_proof :
  remaining_amount = 19000 := by sorry

end NUMINAMATH_CALUDE_remaining_money_proof_l2659_265965


namespace NUMINAMATH_CALUDE_javiers_dogs_l2659_265904

theorem javiers_dogs (total_legs : ℕ) (human_count : ℕ) (human_legs : ℕ) (dog_legs : ℕ) :
  total_legs = 22 →
  human_count = 5 →
  human_legs = 2 →
  dog_legs = 4 →
  (human_count * human_legs + (total_legs - human_count * human_legs) / dog_legs : ℕ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_javiers_dogs_l2659_265904


namespace NUMINAMATH_CALUDE_miami_hurricane_damage_l2659_265956

/-- Calculates the damage amount in Euros given the damage in US dollars and the exchange rate. -/
def damage_in_euros (damage_usd : ℝ) (exchange_rate : ℝ) : ℝ :=
  damage_usd * exchange_rate

/-- Theorem stating that the damage caused by the hurricane in Miami is 40,500,000 Euros. -/
theorem miami_hurricane_damage :
  let damage_usd : ℝ := 45000000
  let exchange_rate : ℝ := 0.9
  damage_in_euros damage_usd exchange_rate = 40500000 := by
  sorry

end NUMINAMATH_CALUDE_miami_hurricane_damage_l2659_265956


namespace NUMINAMATH_CALUDE_characterization_of_k_l2659_265914

/-- The greatest odd divisor of a natural number -/
def greatestOddDivisor (m : ℕ) : ℕ := sorry

/-- The property that n does not divide the greatest odd divisor of k^n + 1 -/
def noDivide (k n : ℕ) : Prop :=
  ¬(n ∣ greatestOddDivisor ((k^n + 1) : ℕ))

/-- The main theorem -/
theorem characterization_of_k (k : ℕ) (h : k ≥ 2) :
  (∃ l : ℕ, l ≥ 2 ∧ k = 2^l - 1) ↔ (∀ n : ℕ, n ≥ 2 → noDivide k n) := by
  sorry

end NUMINAMATH_CALUDE_characterization_of_k_l2659_265914


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2659_265911

theorem quadratic_equation_solution : 
  ∃ x₁ x₂ : ℝ, (x₁^2 - 4 = 0 ∧ x₂^2 - 4 = 0) ∧ x₁ = 2 ∧ x₂ = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2659_265911


namespace NUMINAMATH_CALUDE_polynomial_sum_l2659_265957

variable (x y : ℝ)
variable (P : ℝ → ℝ → ℝ)

theorem polynomial_sum (h : ∀ x y, P x y + (x^2 - y^2) = x^2 + y^2) :
  ∀ x y, P x y = 2 * y^2 := by
sorry

end NUMINAMATH_CALUDE_polynomial_sum_l2659_265957


namespace NUMINAMATH_CALUDE_S_lower_bound_l2659_265939

/-- The least positive integer S(n) such that S(n) ≡ n (mod 2), S(n) ≥ n, 
    and there are no positive integers k, x₁, x₂, ..., xₖ such that 
    n = x₁ + x₂ + ... + xₖ and S(n) = x₁² + x₂² + ... + xₖ² -/
noncomputable def S (n : ℕ) : ℕ := sorry

/-- S(n) grows at least as fast as c * n^(3/2) for some constant c > 0 
    and for all sufficiently large n -/
theorem S_lower_bound :
  ∃ (c : ℝ) (n₀ : ℕ), c > 0 ∧ ∀ n ≥ n₀, (S n : ℝ) ≥ c * n^(3/2) := by sorry

end NUMINAMATH_CALUDE_S_lower_bound_l2659_265939


namespace NUMINAMATH_CALUDE_paityn_blue_hats_l2659_265954

theorem paityn_blue_hats (paityn_red : ℕ) (paityn_blue : ℕ) (zola_red : ℕ) (zola_blue : ℕ) 
  (h1 : paityn_red = 20)
  (h2 : zola_red = (4 : ℕ) * paityn_red / 5)
  (h3 : zola_blue = 2 * paityn_blue)
  (h4 : paityn_red + paityn_blue + zola_red + zola_blue = 2 * 54) :
  paityn_blue = 24 := by
  sorry

end NUMINAMATH_CALUDE_paityn_blue_hats_l2659_265954


namespace NUMINAMATH_CALUDE_intersection_line_equation_l2659_265923

/-- Given two lines l₁ and l₂ in the plane, and a line l passing through their
    intersection point and the origin, prove that l has the equation x - 10y = 0. -/
theorem intersection_line_equation :
  let l₁ : ℝ × ℝ → Prop := λ p => 2 * p.1 + p.2 = 3
  let l₂ : ℝ × ℝ → Prop := λ p => p.1 + 4 * p.2 = 2
  let P : ℝ × ℝ := (10/7, 1/7)  -- Intersection point of l₁ and l₂
  let l : ℝ × ℝ → Prop := λ p => p.1 - 10 * p.2 = 0
  (l₁ P ∧ l₂ P) →  -- P is the intersection of l₁ and l₂
  (l (0, 0)) →     -- l passes through the origin
  (l P) →          -- l passes through P
  ∀ p : ℝ × ℝ, (l₁ p ∧ l₂ p) → l p  -- For any point on both l₁ and l₂, it's also on l
  :=
by sorry

end NUMINAMATH_CALUDE_intersection_line_equation_l2659_265923


namespace NUMINAMATH_CALUDE_not_pythagorean_triple_l2659_265984

/-- Checks if a triple of natural numbers forms a Pythagorean triple --/
def isPythagoreanTriple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

theorem not_pythagorean_triple : 
  ¬(isPythagoreanTriple 15 8 19) ∧ 
  (isPythagoreanTriple 6 8 10) ∧ 
  (isPythagoreanTriple 5 12 13) ∧ 
  (isPythagoreanTriple 3 5 4) := by
  sorry

end NUMINAMATH_CALUDE_not_pythagorean_triple_l2659_265984


namespace NUMINAMATH_CALUDE_triathlete_average_speed_l2659_265968

/-- The average speed of a triathlete for swimming and running events -/
theorem triathlete_average_speed 
  (swim_speed : ℝ) 
  (run_speed : ℝ) 
  (h1 : swim_speed = 1) 
  (h2 : run_speed = 6) : 
  (2 * swim_speed * run_speed) / (swim_speed + run_speed) = 12 / 7 := by
  sorry

end NUMINAMATH_CALUDE_triathlete_average_speed_l2659_265968


namespace NUMINAMATH_CALUDE_inscribed_parallelepiped_volume_l2659_265905

/-- The volume of a rectangular parallelepiped inscribed in a pyramid -/
theorem inscribed_parallelepiped_volume
  (a : ℝ) -- Side length of the square base of the pyramid
  (α β : ℝ) -- Angles α and β as described in the problem
  (h1 : 0 < a)
  (h2 : 0 < α ∧ α < π / 2)
  (h3 : 0 < β ∧ β < π / 2)
  (h4 : α + β < π / 2) :
  ∃ V : ℝ, -- Volume of the parallelepiped
    V = (a^3 * Real.sqrt 2 * Real.sin α * Real.cos α^2 * Real.sin β^3) /
        Real.sin (α + β)^3 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_parallelepiped_volume_l2659_265905


namespace NUMINAMATH_CALUDE_base5_division_proof_l2659_265960

-- Define a function to convert from base 5 to decimal
def base5ToDecimal (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

-- Define a function to convert from decimal to base 5
def decimalToBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
  aux n []

-- Theorem statement
theorem base5_division_proof :
  let dividend := [4, 0, 1, 2]  -- 2104₅ in reverse order
  let divisor := [3, 2]         -- 23₅ in reverse order
  let quotient := [1, 4]        -- 41₅ in reverse order
  (base5ToDecimal dividend) / (base5ToDecimal divisor) = base5ToDecimal quotient :=
by sorry

end NUMINAMATH_CALUDE_base5_division_proof_l2659_265960


namespace NUMINAMATH_CALUDE_cow_hen_problem_l2659_265922

theorem cow_hen_problem (cows hens : ℕ) : 
  4 * cows + 2 * hens = 2 * (cows + hens) + 8 → cows = 4 := by
  sorry

end NUMINAMATH_CALUDE_cow_hen_problem_l2659_265922


namespace NUMINAMATH_CALUDE_cube_diagonal_pairs_l2659_265951

/-- The number of diagonals on the faces of a cube -/
def num_diagonals : ℕ := 12

/-- The total number of pairs of diagonals -/
def total_pairs : ℕ := num_diagonals.choose 2

/-- The number of pairs of diagonals that do not form a 60° angle -/
def non_60_degree_pairs : ℕ := 18

/-- The number of pairs of diagonals that form a 60° angle -/
def pairs_60_degree : ℕ := total_pairs - non_60_degree_pairs

theorem cube_diagonal_pairs :
  pairs_60_degree = 48 := by sorry

end NUMINAMATH_CALUDE_cube_diagonal_pairs_l2659_265951


namespace NUMINAMATH_CALUDE_train_speed_l2659_265971

/-- The speed of a train given its length and time to cross a fixed point. -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 280) (h2 : time = 20) :
  length / time = 14 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l2659_265971


namespace NUMINAMATH_CALUDE_problem_solution_l2659_265925

theorem problem_solution (x y z : ℝ) 
  (h : y^2 + |x - 2023| + Real.sqrt (z - 4) = 6*y - 9) : 
  (y - z)^x = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2659_265925


namespace NUMINAMATH_CALUDE_pet_store_cages_l2659_265929

def number_of_cages (initial_puppies : ℕ) (sold_puppies : ℕ) (puppies_per_cage : ℕ) : ℕ :=
  (initial_puppies - sold_puppies) / puppies_per_cage

theorem pet_store_cages : number_of_cages 18 3 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_cages_l2659_265929


namespace NUMINAMATH_CALUDE_birds_and_storks_on_fence_l2659_265930

theorem birds_and_storks_on_fence (initial_birds initial_storks additional_birds final_total : ℕ) :
  initial_birds = 3 →
  additional_birds = 5 →
  final_total = 10 →
  initial_birds + initial_storks + additional_birds = final_total →
  initial_storks = 2 := by
  sorry

end NUMINAMATH_CALUDE_birds_and_storks_on_fence_l2659_265930


namespace NUMINAMATH_CALUDE_battery_collection_theorem_l2659_265976

/-- Represents the number of batteries collected by students. -/
structure BatteryCollection where
  jiajia : ℕ
  qiqi : ℕ

/-- Represents the state of battery collection before and after the exchange. -/
structure BatteryExchange where
  initial : BatteryCollection
  final : BatteryCollection

/-- Theorem about battery collection and exchange between Jiajia and Qiqi. -/
theorem battery_collection_theorem (m : ℕ) :
  ∃ (exchange : BatteryExchange),
    -- Initial conditions
    exchange.initial.jiajia = m ∧
    exchange.initial.qiqi = 2 * m - 2 ∧
    -- Condition that Qiqi would have twice as many if she collected two more
    exchange.initial.qiqi + 2 = 2 * exchange.initial.jiajia ∧
    -- Final conditions after Qiqi gives two batteries to Jiajia
    exchange.final.jiajia = exchange.initial.jiajia + 2 ∧
    exchange.final.qiqi = exchange.initial.qiqi - 2 ∧
    -- Prove that Qiqi has m - 6 more batteries than Jiajia after the exchange
    exchange.final.qiqi - exchange.final.jiajia = m - 6 :=
by
  sorry

end NUMINAMATH_CALUDE_battery_collection_theorem_l2659_265976


namespace NUMINAMATH_CALUDE_doris_earnings_l2659_265970

/-- Calculates the number of weeks needed to earn a target amount given an hourly rate and weekly work schedule. -/
def weeks_to_earn (hourly_rate : ℕ) (weekday_hours : ℕ) (saturday_hours : ℕ) (target_amount : ℕ) : ℕ :=
  let weekly_hours := weekday_hours * 5 + saturday_hours
  let weekly_earnings := hourly_rate * weekly_hours
  (target_amount + weekly_earnings - 1) / weekly_earnings

/-- Theorem stating that Doris needs 3 weeks to earn at least $1200 given her work schedule. -/
theorem doris_earnings : weeks_to_earn 20 3 5 1200 = 3 := by
  sorry

end NUMINAMATH_CALUDE_doris_earnings_l2659_265970


namespace NUMINAMATH_CALUDE_average_candy_count_l2659_265936

def candy_counts : List Nat := [5, 7, 9, 12, 12, 15, 15, 18, 25]

theorem average_candy_count (num_bags : Nat) (counts : List Nat) 
  (h1 : num_bags = 9)
  (h2 : counts = candy_counts)
  (h3 : counts.length = num_bags) :
  Int.floor ((counts.sum : ℝ) / num_bags + 0.5) = 13 :=
by sorry

end NUMINAMATH_CALUDE_average_candy_count_l2659_265936


namespace NUMINAMATH_CALUDE_largest_common_value_l2659_265952

def arithmetic_progression_1 (n : ℕ) : ℕ := 4 + 5 * n
def arithmetic_progression_2 (n : ℕ) : ℕ := 5 + 8 * n

theorem largest_common_value :
  ∃ (k : ℕ), 
    (∃ (n m : ℕ), arithmetic_progression_1 n = arithmetic_progression_2 m ∧ arithmetic_progression_1 n = k) ∧
    k < 1000 ∧
    (∀ (l : ℕ), l < 1000 → 
      (∃ (p q : ℕ), arithmetic_progression_1 p = arithmetic_progression_2 q ∧ arithmetic_progression_1 p = l) →
      l ≤ k) ∧
    k = 989 :=
by sorry

end NUMINAMATH_CALUDE_largest_common_value_l2659_265952


namespace NUMINAMATH_CALUDE_sqrt_transformation_l2659_265981

theorem sqrt_transformation (n : ℕ) (h : n ≥ 1) : 
  Real.sqrt ((1 : ℝ) / n * ((1 : ℝ) / (n + 1) - (1 : ℝ) / (n + 2))) = 
  (1 : ℝ) / (n + 1) * Real.sqrt ((n + 1 : ℝ) / (n * (n + 2))) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_transformation_l2659_265981


namespace NUMINAMATH_CALUDE_min_value_f_l2659_265992

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 4*x + (2 - a)*Real.log x

-- Define the derivative of f
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 2*x - 4 - (2 - a)/x

-- Theorem statement
theorem min_value_f (a : ℝ) :
  ∃ (min_val : ℝ), ∀ x ∈ Set.Icc (Real.exp 1) (Real.exp 2), f a x ≥ min_val ∧
  (min_val = f a (Real.exp 1) ∨
   min_val = f a (Real.exp 2) ∨
   (∃ y ∈ Set.Ioo (Real.exp 1) (Real.exp 2), min_val = f a y ∧ f_deriv a y = 0)) :=
sorry

end

end NUMINAMATH_CALUDE_min_value_f_l2659_265992


namespace NUMINAMATH_CALUDE_dennis_floors_above_charlie_l2659_265994

/-- The floor number on which Frank lives -/
def frank_floor : ℕ := 16

/-- The floor number on which Charlie lives -/
def charlie_floor : ℕ := frank_floor / 4

/-- The floor number on which Dennis lives -/
def dennis_floor : ℕ := 6

/-- The number of floors Dennis lives above Charlie -/
def floors_above : ℕ := dennis_floor - charlie_floor

theorem dennis_floors_above_charlie : floors_above = 2 := by
  sorry

end NUMINAMATH_CALUDE_dennis_floors_above_charlie_l2659_265994


namespace NUMINAMATH_CALUDE_no_rearranged_powers_of_two_l2659_265921

-- Define a function to check if a number is a power of 2
def isPowerOfTwo (n : ℕ) : Prop := ∃ k : ℕ, n = 2^k

-- Define a function to check if two numbers have the same digits
def haveSameDigits (m n : ℕ) : Prop :=
  ∃ (digits : List ℕ) (perm : List ℕ), 
    digits.length > 0 ∧
    perm.isPerm digits ∧
    m = digits.foldl (fun acc d => acc * 10 + d) 0 ∧
    n = perm.foldl (fun acc d => acc * 10 + d) 0 ∧
    perm.head? ≠ some 0

theorem no_rearranged_powers_of_two :
  ¬∃ (m n : ℕ), m ≠ n ∧ m > 0 ∧ n > 0 ∧ 
  isPowerOfTwo m ∧ isPowerOfTwo n ∧ 
  haveSameDigits m n :=
sorry

end NUMINAMATH_CALUDE_no_rearranged_powers_of_two_l2659_265921


namespace NUMINAMATH_CALUDE_square_of_difference_of_square_roots_l2659_265910

theorem square_of_difference_of_square_roots : 
  (Real.sqrt (5 + 4 * Real.sqrt 3) - Real.sqrt (5 - 4 * Real.sqrt 3))^2 = 10 + 2 * Complex.I * Real.sqrt 23 :=
by sorry

end NUMINAMATH_CALUDE_square_of_difference_of_square_roots_l2659_265910


namespace NUMINAMATH_CALUDE_spiral_grid_third_row_sum_l2659_265926

/-- Represents a position in the grid -/
structure Position :=
  (row : ℕ)
  (col : ℕ)

/-- Represents the spiral grid -/
def SpiralGrid :=
  Position → ℕ

/-- The size of the grid -/
def gridSize : ℕ := 17

/-- The center position of the grid -/
def centerPos : Position :=
  { row := 9, col := 9 }

/-- Creates a spiral grid with the given properties -/
def createSpiralGrid : SpiralGrid :=
  sorry

/-- Checks if a position is in the third row from the top -/
def isInThirdRow (p : Position) : Prop :=
  p.row = 3

/-- Finds the greatest number in the third row -/
def greatestInThirdRow (grid : SpiralGrid) : ℕ :=
  sorry

/-- Finds the least number in the third row -/
def leastInThirdRow (grid : SpiralGrid) : ℕ :=
  sorry

theorem spiral_grid_third_row_sum :
  let grid := createSpiralGrid
  greatestInThirdRow grid + leastInThirdRow grid = 528 := by
  sorry

end NUMINAMATH_CALUDE_spiral_grid_third_row_sum_l2659_265926


namespace NUMINAMATH_CALUDE_students_not_in_chorus_or_band_l2659_265967

theorem students_not_in_chorus_or_band 
  (total : ℕ) (chorus : ℕ) (band : ℕ) (both : ℕ) 
  (h1 : total = 50)
  (h2 : chorus = 18)
  (h3 : band = 26)
  (h4 : both = 2) :
  total - (chorus + band - both) = 8 := by
  sorry

end NUMINAMATH_CALUDE_students_not_in_chorus_or_band_l2659_265967


namespace NUMINAMATH_CALUDE_employee_count_sum_l2659_265955

theorem employee_count_sum : 
  (Finset.sum (Finset.filter (fun s => 200 ≤ s ∧ s ≤ 300 ∧ (s - 1) % 7 = 0) (Finset.range 301)) id) = 3493 :=
by sorry

end NUMINAMATH_CALUDE_employee_count_sum_l2659_265955


namespace NUMINAMATH_CALUDE_unique_number_with_three_prime_factors_l2659_265987

theorem unique_number_with_three_prime_factors (x n : ℕ) : 
  x = 7^n + 1 →
  Odd n →
  (∃ p q : ℕ, Prime p ∧ Prime q ∧ p ≠ q ∧ p ≠ 11 ∧ q ≠ 11 ∧ x = 2 * 11 * p * q) →
  x = 16808 :=
by sorry

end NUMINAMATH_CALUDE_unique_number_with_three_prime_factors_l2659_265987


namespace NUMINAMATH_CALUDE_johns_allowance_l2659_265935

theorem johns_allowance (allowance : ℝ) : 
  (allowance > 0) →
  (2 / 3 * (2 / 5 * allowance) = 1.28) →
  allowance = 4.80 := by
sorry

end NUMINAMATH_CALUDE_johns_allowance_l2659_265935


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l2659_265961

theorem sum_of_roots_quadratic (b : ℝ) (x₁ x₂ : ℝ) : 
  x₁^2 - 2*x₁ + b = 0 → x₂^2 - 2*x₂ + b = 0 → x₁ + x₂ = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l2659_265961


namespace NUMINAMATH_CALUDE_burgers_per_day_l2659_265902

/-- The number of days in June -/
def june_days : ℕ := 30

/-- The cost of each burger in dollars -/
def burger_cost : ℕ := 13

/-- The total amount Alice spent on burgers in June in dollars -/
def total_spent : ℕ := 1560

/-- Alice bought burgers every day in June -/
axiom bought_daily : ∀ d : ℕ, d ≤ june_days → ∃ b : ℕ, b > 0

/-- Theorem: Alice purchased 4 burgers per day in June -/
theorem burgers_per_day : 
  (total_spent / burger_cost) / june_days = 4 := by sorry

end NUMINAMATH_CALUDE_burgers_per_day_l2659_265902


namespace NUMINAMATH_CALUDE_sqrt_x_minus_one_defined_l2659_265900

theorem sqrt_x_minus_one_defined (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 1) ↔ x ≥ 1 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_one_defined_l2659_265900


namespace NUMINAMATH_CALUDE_square_to_rectangle_perimeter_l2659_265907

theorem square_to_rectangle_perimeter (n : ℕ) (a : ℝ) : 
  a > 0 →
  n > 0 →
  ∃ k : ℕ, k > 0 ∧ k < n ∧
  (k : ℝ) * 6 * a = (n - 2 * k : ℝ) * 4 * a ∧
  4 * n * a - (4 * n * a - 40) = 40 →
  4 * n * a = 280 :=
by sorry

end NUMINAMATH_CALUDE_square_to_rectangle_perimeter_l2659_265907

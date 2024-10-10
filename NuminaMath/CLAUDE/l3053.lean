import Mathlib

namespace max_sum_of_square_roots_l3053_305385

theorem max_sum_of_square_roots (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 7) :
  Real.sqrt (3 * x + 2) + Real.sqrt (3 * y + 2) + Real.sqrt (3 * z + 2) ≤ 9 := by
sorry

end max_sum_of_square_roots_l3053_305385


namespace arithmetic_sequence_sum_first_ten_terms_sum_l3053_305335

theorem arithmetic_sequence_sum : ℤ → ℤ → ℕ → ℤ
  | a, l, n => n * (a + l) / 2

theorem first_ten_terms_sum (a l : ℤ) (n : ℕ) (h1 : a = -5) (h2 : l = 40) (h3 : n = 10) :
  arithmetic_sequence_sum a l n = 175 := by
  sorry

end arithmetic_sequence_sum_first_ten_terms_sum_l3053_305335


namespace rectangle_x_value_l3053_305348

/-- A rectangular figure with specified segment lengths -/
structure RectangularFigure where
  top_segment1 : ℝ
  top_segment2 : ℝ
  top_segment3 : ℝ
  bottom_segment1 : ℝ
  bottom_segment2 : ℝ
  bottom_segment3 : ℝ

/-- The property that the total length of top and bottom sides are equal -/
def is_valid_rectangle (r : RectangularFigure) : Prop :=
  r.top_segment1 + r.top_segment2 + r.top_segment3 = r.bottom_segment1 + r.bottom_segment2 + r.bottom_segment3

/-- The theorem stating that X must be 6 for the given rectangular figure -/
theorem rectangle_x_value :
  ∀ (x : ℝ),
  is_valid_rectangle ⟨3, 2, x, 4, 2, 5⟩ → x = 6 := by
  sorry

end rectangle_x_value_l3053_305348


namespace all_groups_have_access_l3053_305378

-- Define the type for house groups
inductive HouseGroup : Type
  | a | b | c | d | e | f | g | h | i | j | k | l | m

-- Define the adjacency relation
def adjacent : HouseGroup → HouseGroup → Prop
  | HouseGroup.a, HouseGroup.b => True
  | HouseGroup.a, HouseGroup.d => True
  | HouseGroup.b, HouseGroup.a => True
  | HouseGroup.b, HouseGroup.c => True
  | HouseGroup.b, HouseGroup.d => True
  | HouseGroup.c, HouseGroup.b => True
  | HouseGroup.d, HouseGroup.a => True
  | HouseGroup.d, HouseGroup.b => True
  | HouseGroup.d, HouseGroup.f => True
  | HouseGroup.d, HouseGroup.e => True
  | HouseGroup.e, HouseGroup.d => True
  | HouseGroup.e, HouseGroup.f => True
  | HouseGroup.e, HouseGroup.j => True
  | HouseGroup.e, HouseGroup.l => True
  | HouseGroup.f, HouseGroup.d => True
  | HouseGroup.f, HouseGroup.e => True
  | HouseGroup.f, HouseGroup.j => True
  | HouseGroup.f, HouseGroup.i => True
  | HouseGroup.f, HouseGroup.g => True
  | HouseGroup.g, HouseGroup.f => True
  | HouseGroup.g, HouseGroup.i => True
  | HouseGroup.g, HouseGroup.h => True
  | HouseGroup.h, HouseGroup.g => True
  | HouseGroup.h, HouseGroup.i => True
  | HouseGroup.i, HouseGroup.j => True
  | HouseGroup.i, HouseGroup.f => True
  | HouseGroup.i, HouseGroup.g => True
  | HouseGroup.i, HouseGroup.h => True
  | HouseGroup.j, HouseGroup.k => True
  | HouseGroup.j, HouseGroup.e => True
  | HouseGroup.j, HouseGroup.f => True
  | HouseGroup.j, HouseGroup.i => True
  | HouseGroup.k, HouseGroup.l => True
  | HouseGroup.k, HouseGroup.j => True
  | HouseGroup.l, HouseGroup.k => True
  | HouseGroup.l, HouseGroup.e => True
  | _, _ => False

-- Define the set of pharmacy locations
def pharmacyLocations : Set HouseGroup :=
  {HouseGroup.b, HouseGroup.i, HouseGroup.l, HouseGroup.m}

-- Define the property of having access to a pharmacy
def hasAccessToPharmacy (g : HouseGroup) : Prop :=
  g ∈ pharmacyLocations ∨ ∃ h ∈ pharmacyLocations, adjacent g h

-- Theorem statement
theorem all_groups_have_access :
  ∀ g : HouseGroup, hasAccessToPharmacy g :=
by sorry

end all_groups_have_access_l3053_305378


namespace train_passing_time_l3053_305391

/-- The length of train A in meters -/
def train_a_length : ℝ := 150

/-- The length of train B in meters -/
def train_b_length : ℝ := 200

/-- The time (in seconds) it takes for a passenger on train A to see train B pass by -/
def time_a_sees_b : ℝ := 10

/-- The time (in seconds) it takes for a passenger on train B to see train A pass by -/
def time_b_sees_a : ℝ := 7.5

theorem train_passing_time :
  (train_b_length / time_a_sees_b) = (train_a_length / time_b_sees_a) :=
by sorry

end train_passing_time_l3053_305391


namespace milk_water_ratio_after_addition_l3053_305331

theorem milk_water_ratio_after_addition 
  (initial_volume : ℝ) 
  (initial_milk_ratio : ℝ) 
  (initial_water_ratio : ℝ) 
  (added_water : ℝ) :
  initial_volume = 45 ∧ 
  initial_milk_ratio = 4 ∧ 
  initial_water_ratio = 1 ∧ 
  added_water = 3 →
  let initial_total_ratio := initial_milk_ratio + initial_water_ratio
  let initial_milk_volume := (initial_milk_ratio / initial_total_ratio) * initial_volume
  let initial_water_volume := (initial_water_ratio / initial_total_ratio) * initial_volume
  let final_milk_volume := initial_milk_volume
  let final_water_volume := initial_water_volume + added_water
  let final_ratio := final_milk_volume / final_water_volume
  final_ratio = 3 := by
sorry

end milk_water_ratio_after_addition_l3053_305331


namespace ethans_rowing_time_l3053_305333

/-- Proves that Ethan's rowing time is 25 minutes given the conditions -/
theorem ethans_rowing_time (total_time : ℕ) (ethan_time : ℕ) :
  total_time = 75 →
  total_time = ethan_time + 2 * ethan_time →
  ethan_time = 25 := by
  sorry

end ethans_rowing_time_l3053_305333


namespace highest_divisible_digit_l3053_305341

theorem highest_divisible_digit : 
  ∃ (a : ℕ), a ≤ 9 ∧ 
  (43752 * 1000 + a * 100 + 539) % 8 = 0 ∧
  (43752 * 1000 + a * 100 + 539) % 9 = 0 ∧
  (43752 * 1000 + a * 100 + 539) % 12 = 0 ∧
  ∀ (b : ℕ), b > a → b ≤ 9 → 
    (43752 * 1000 + b * 100 + 539) % 8 ≠ 0 ∨
    (43752 * 1000 + b * 100 + 539) % 9 ≠ 0 ∨
    (43752 * 1000 + b * 100 + 539) % 12 ≠ 0 :=
by sorry

end highest_divisible_digit_l3053_305341


namespace traditionalist_ratio_in_specific_country_l3053_305323

/-- Represents a country with provinces, progressives, and traditionalists -/
structure Country where
  num_provinces : ℕ
  num_progressives : ℕ
  num_traditionalists_per_province : ℕ

/-- The fraction of the country that is traditionalist -/
def traditionalist_fraction (c : Country) : ℚ :=
  (c.num_traditionalists_per_province * c.num_provinces : ℚ) / 
  (c.num_progressives + c.num_traditionalists_per_province * c.num_provinces : ℚ)

/-- The ratio of traditionalists in one province to total progressives -/
def traditionalist_to_progressive_ratio (c : Country) : ℚ :=
  (c.num_traditionalists_per_province : ℚ) / c.num_progressives

theorem traditionalist_ratio_in_specific_country :
  ∀ c : Country,
    c.num_provinces = 5 →
    traditionalist_fraction c = 3/4 →
    traditionalist_to_progressive_ratio c = 3/5 := by
  sorry

end traditionalist_ratio_in_specific_country_l3053_305323


namespace salary_change_percentage_salary_loss_percentage_l3053_305380

theorem salary_change_percentage (original : ℝ) (original_positive : 0 < original) :
  let decreased := original * (1 - 0.6)
  let increased := decreased * (1 + 0.6)
  increased = original * 0.64 :=
by
  sorry

theorem salary_loss_percentage (original : ℝ) (original_positive : 0 < original) :
  let decreased := original * (1 - 0.6)
  let increased := decreased * (1 + 0.6)
  (original - increased) / original = 0.36 :=
by
  sorry

end salary_change_percentage_salary_loss_percentage_l3053_305380


namespace exponential_function_fixed_point_l3053_305362

theorem exponential_function_fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f := fun (x : ℝ) ↦ a^x + 1
  f 0 = 2 := by
sorry

end exponential_function_fixed_point_l3053_305362


namespace retail_price_calculation_l3053_305309

/-- The retail price of a machine, given wholesale price, discount, and profit margin. -/
theorem retail_price_calculation (wholesale_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) 
  (h1 : wholesale_price = 90)
  (h2 : discount_rate = 0.1)
  (h3 : profit_rate = 0.2) :
  ∃ w : ℝ, w = 120 ∧ 
    (1 - discount_rate) * w = wholesale_price + profit_rate * wholesale_price :=
by sorry

end retail_price_calculation_l3053_305309


namespace second_number_is_sixty_l3053_305324

theorem second_number_is_sixty :
  ∀ (a b : ℝ),
  (a + b + 20 + 60) / 4 = (10 + 70 + 28) / 3 + 4 →
  (a = 60 ∨ b = 60) :=
by
  sorry

end second_number_is_sixty_l3053_305324


namespace one_positive_integer_satisfies_condition_l3053_305329

theorem one_positive_integer_satisfies_condition : 
  ∃! (x : ℕ+), 25 - (5 * x.val) > 15 := by sorry

end one_positive_integer_satisfies_condition_l3053_305329


namespace pet_shop_total_cost_l3053_305352

/-- Represents the cost of purchasing all pets in a pet shop given specific conditions. -/
def total_cost_of_pets (num_puppies num_kittens num_parakeets : ℕ) 
  (parakeet_cost : ℚ) 
  (puppy_parakeet_ratio kitten_parakeet_ratio : ℚ) : ℚ :=
  let puppy_cost := puppy_parakeet_ratio * parakeet_cost
  let kitten_cost := kitten_parakeet_ratio * parakeet_cost
  (num_puppies : ℚ) * puppy_cost + (num_kittens : ℚ) * kitten_cost + (num_parakeets : ℚ) * parakeet_cost

/-- Theorem stating that under given conditions, the total cost of pets is $130. -/
theorem pet_shop_total_cost : 
  total_cost_of_pets 2 2 3 10 3 2 = 130 := by
  sorry

end pet_shop_total_cost_l3053_305352


namespace find_second_number_l3053_305301

theorem find_second_number (x : ℝ) : 
  (20 + 40 + 60) / 3 = ((10 + 28 + x) / 3) + 4 → x = 70 := by
  sorry

end find_second_number_l3053_305301


namespace kindergarten_craft_problem_l3053_305346

theorem kindergarten_craft_problem :
  ∃ (scissors glue_sticks crayons : ℕ),
    scissors + glue_sticks + crayons = 26 ∧
    2 * scissors + 3 * glue_sticks + 4 * crayons = 24 := by
  sorry

end kindergarten_craft_problem_l3053_305346


namespace line_equation_proof_l3053_305336

/-- Given a line in the form ax + by + c = 0, prove it has slope -3 and x-intercept 2 -/
theorem line_equation_proof (a b c : ℝ) (h1 : a = 3) (h2 : b = 1) (h3 : c = -6) : 
  (∀ x y : ℝ, a*x + b*y + c = 0 ↔ y = -3*(x - 2)) := by sorry

end line_equation_proof_l3053_305336


namespace jake_buys_three_packages_l3053_305330

/-- Represents the number of sausage packages Jake buys -/
def num_packages : ℕ := 3

/-- Represents the weight of each sausage package in pounds -/
def package_weight : ℕ := 2

/-- Represents the price per pound of sausages in dollars -/
def price_per_pound : ℕ := 4

/-- Represents the total amount Jake pays in dollars -/
def total_paid : ℕ := 24

/-- Theorem stating that Jake buys 3 packages of sausages -/
theorem jake_buys_three_packages : 
  num_packages * package_weight * price_per_pound = total_paid :=
by sorry

end jake_buys_three_packages_l3053_305330


namespace union_M_complement_N_equals_U_l3053_305357

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set M as the domain of ln(1-x)
def M : Set ℝ := {x | x < 1}

-- Define set N as {x | x²-x < 0}
def N : Set ℝ := {x | x^2 - x < 0}

-- Theorem statement
theorem union_M_complement_N_equals_U : M ∪ (U \ N) = U := by sorry

end union_M_complement_N_equals_U_l3053_305357


namespace sphere_diameter_count_l3053_305386

theorem sphere_diameter_count (total_points : ℕ) (surface_percentage : ℚ) 
  (h1 : total_points = 39)
  (h2 : surface_percentage ≤ 72/100)
  : ∃ (surface_points : ℕ), 
    surface_points ≤ ⌊(surface_percentage * total_points)⌋ ∧ 
    (surface_points.choose 2) = 378 := by
  sorry

end sphere_diameter_count_l3053_305386


namespace box_length_with_cubes_l3053_305300

/-- Given a box with dimensions L × 15 × 6 inches that can be filled entirely
    with 90 identical cubes leaving no space unfilled, prove that the length L
    of the box is 27 inches. -/
theorem box_length_with_cubes (L : ℕ) : 
  (∃ (s : ℕ), L * 15 * 6 = 90 * s^3 ∧ s ∣ 15 ∧ s ∣ 6) → L = 27 := by
  sorry

end box_length_with_cubes_l3053_305300


namespace apple_picking_ratio_l3053_305310

/-- The number of apples Lexie picked -/
def lexie_apples : ℕ := 12

/-- The total number of apples picked by Lexie and Tom -/
def total_apples : ℕ := 36

/-- The number of apples Tom picked -/
def tom_apples : ℕ := total_apples - lexie_apples

/-- The ratio of Tom's apples to Lexie's apples -/
def apple_ratio : ℚ := tom_apples / lexie_apples

theorem apple_picking_ratio :
  apple_ratio = 2 := by sorry

end apple_picking_ratio_l3053_305310


namespace garden_carnations_percentage_l3053_305358

theorem garden_carnations_percentage 
  (total : ℕ) 
  (pink : ℕ) 
  (white : ℕ) 
  (pink_roses : ℕ) 
  (red_carnations : ℕ) 
  (h_pink : pink = 3 * total / 5)
  (h_white : white = total / 5)
  (h_pink_roses : pink_roses = pink / 2)
  (h_red_carnations : red_carnations = (total - pink - white) / 2) :
  (pink - pink_roses + red_carnations + white) * 100 = 60 * total :=
sorry

end garden_carnations_percentage_l3053_305358


namespace seating_arrangement_l3053_305355

/-- The number of ways to arrange n people in n chairs -/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- The number of chairs in the row -/
def total_chairs : ℕ := 7

/-- The number of people to be seated -/
def people_to_seat : ℕ := 5

/-- The number of chairs that must remain empty -/
def empty_chairs : ℕ := 2

theorem seating_arrangement :
  permutations (total_chairs - empty_chairs) = 120 := by
  sorry

end seating_arrangement_l3053_305355


namespace paper_side_length_l3053_305339

theorem paper_side_length (cube_side: ℝ) (num_pieces: ℕ) (paper_side: ℝ)
  (h1: cube_side = 12)
  (h2: num_pieces = 54)
  (h3: (6 * cube_side^2) = (num_pieces * paper_side^2)) :
  paper_side = 4 := by
  sorry

end paper_side_length_l3053_305339


namespace initial_sets_count_l3053_305368

/-- The number of letters available (A through J) -/
def num_letters : ℕ := 10

/-- The number of letters in each set of initials -/
def set_size : ℕ := 3

/-- The number of arrangements for three letters where two are identical -/
def repeated_letter_arrangements : ℕ := 3

/-- The number of different three-letter sets of initials possible using letters A through J, 
    where one letter can appear twice and the third must be different -/
theorem initial_sets_count : 
  num_letters * (num_letters - 1) * repeated_letter_arrangements = 270 := by
  sorry

end initial_sets_count_l3053_305368


namespace equation_is_false_l3053_305325

theorem equation_is_false : 4.58 - (0.45 + 2.58) ≠ 4.58 - 2.58 + 0.45 ∨ 4.58 - (0.45 + 2.58) ≠ 2.45 := by
  sorry

end equation_is_false_l3053_305325


namespace square_difference_630_570_l3053_305313

theorem square_difference_630_570 : 630^2 - 570^2 = 72000 := by
  sorry

end square_difference_630_570_l3053_305313


namespace discriminant_positive_roots_difference_implies_m_values_l3053_305363

-- Define the quadratic equation
def quadratic (m x : ℝ) : ℝ := x^2 + (m + 3) * x + m + 1

-- Define the discriminant of the quadratic equation
def discriminant (m : ℝ) : ℝ := (m + 3)^2 - 4 * (m + 1)

-- Theorem 1: The discriminant is always positive for any real m
theorem discriminant_positive (m : ℝ) : discriminant m > 0 := by sorry

-- Define the roots of the quadratic equation
noncomputable def α (m : ℝ) : ℝ := sorry
noncomputable def β (m : ℝ) : ℝ := sorry

-- Theorem 2: If α - β = 2√2, then m = -3 or m = 1
theorem roots_difference_implies_m_values (m : ℝ) (h : α m - β m = 2 * Real.sqrt 2) : 
  m = -3 ∨ m = 1 := by sorry

end discriminant_positive_roots_difference_implies_m_values_l3053_305363


namespace no_integer_solutions_l3053_305345

theorem no_integer_solutions : ¬∃ (x y : ℤ), 0 < x ∧ x < y ∧ Real.sqrt 4096 = Real.sqrt x + Real.sqrt y + Real.sqrt (2 * x) := by
  sorry

end no_integer_solutions_l3053_305345


namespace exponent_multiplication_l3053_305303

theorem exponent_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end exponent_multiplication_l3053_305303


namespace hyperbola_equation_l3053_305377

/-- The standard equation of a hyperbola with one focus at (2,0) and an asymptote
    with inclination angle of 60° is x^2 - (y^2/3) = 1 -/
theorem hyperbola_equation (C : Set (ℝ × ℝ)) (F : ℝ × ℝ) (θ : ℝ) :
  F = (2, 0) →
  θ = π/3 →
  (∃ (a b : ℝ), ∀ (x y : ℝ),
    (x, y) ∈ C ↔ x^2 / a^2 - y^2 / b^2 = 1 ∧
    b / a = Real.sqrt 3 ∧
    2^2 = a^2 + b^2) →
  (∀ (x y : ℝ), (x, y) ∈ C ↔ x^2 - y^2 / 3 = 1) :=
by sorry


end hyperbola_equation_l3053_305377


namespace extremum_implies_a_in_open_interval_l3053_305394

open Set
open Function
open Real

/-- A function f has exactly one extremum point in an interval (a, b) if there exists
    exactly one point c in (a, b) where f'(c) = 0. -/
def has_exactly_one_extremum (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃! c, a < c ∧ c < b ∧ deriv f c = 0

/-- The cubic function f(x) = x^3 + x^2 - ax - 4 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + x^2 - a*x - 4

theorem extremum_implies_a_in_open_interval :
  ∀ a : ℝ, has_exactly_one_extremum (f a) (-1) 1 → a ∈ Ioo 1 5 :=
sorry

end extremum_implies_a_in_open_interval_l3053_305394


namespace quadratic_inequality_solution_l3053_305392

/-- Given a quadratic inequality ax^2 + bx + 1 > 0 with solution set (-1, 1/3),
    prove that a - b = -1 -/
theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x : ℝ, ax^2 + b*x + 1 > 0 ↔ -1 < x ∧ x < 1/3) → 
  a - b = -1 := by
sorry

end quadratic_inequality_solution_l3053_305392


namespace smallest_four_digit_divisible_by_47_l3053_305381

theorem smallest_four_digit_divisible_by_47 :
  (∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 47 = 0 → 1034 ≤ n) ∧
  1000 ≤ 1034 ∧ 1034 < 10000 ∧ 1034 % 47 = 0 := by
  sorry

end smallest_four_digit_divisible_by_47_l3053_305381


namespace a₁₂_eq_15_l3053_305304

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m
  a₄_eq_1 : a 4 = 1
  a₇_plus_a₉_eq_16 : a 7 + a 9 = 16

/-- The 12th term of the arithmetic sequence is 15 -/
theorem a₁₂_eq_15 (seq : ArithmeticSequence) : seq.a 12 = 15 := by
  sorry

end a₁₂_eq_15_l3053_305304


namespace function_characterization_l3053_305388

/-- A continuous monotonic function satisfying the given inequality is equal to x + 1 -/
theorem function_characterization (f : ℝ → ℝ) 
  (hcont : Continuous f) 
  (hmono : Monotone f) 
  (h0 : f 0 = 1) 
  (hineq : ∀ x y : ℝ, f (x + y) ≥ f x * f y - f (x * y) + 1) :
  ∀ x : ℝ, f x = x + 1 := by
  sorry

end function_characterization_l3053_305388


namespace book_purchase_with_discount_l3053_305361

/-- Calculates the total cost of books with a discount applied -/
theorem book_purchase_with_discount 
  (book_price : ℝ) 
  (quantity : ℕ) 
  (discount_per_book : ℝ) 
  (h1 : book_price = 5) 
  (h2 : quantity = 10) 
  (h3 : discount_per_book = 0.5) : 
  (book_price - discount_per_book) * quantity = 45 := by
sorry

end book_purchase_with_discount_l3053_305361


namespace mary_earnings_l3053_305379

/-- Mary's earnings problem -/
theorem mary_earnings (earnings_per_home : ℕ) (homes_cleaned : ℕ) : 
  earnings_per_home = 46 → homes_cleaned = 6 → earnings_per_home * homes_cleaned = 276 := by
  sorry

end mary_earnings_l3053_305379


namespace problem_stack_total_logs_l3053_305321

/-- Represents a stack of logs -/
structure LogStack where
  bottomRowCount : ℕ
  topRowCount : ℕ
  rowDifference : ℕ

/-- Calculates the total number of logs in the stack -/
def totalLogs (stack : LogStack) : ℕ :=
  sorry

/-- The specific log stack described in the problem -/
def problemStack : LogStack :=
  { bottomRowCount := 20
  , topRowCount := 4
  , rowDifference := 2 }

theorem problem_stack_total_logs :
  totalLogs problemStack = 108 := by
  sorry

end problem_stack_total_logs_l3053_305321


namespace equivalent_operations_l3053_305334

theorem equivalent_operations (x : ℝ) : x * (4/5) / (2/7) = x * (7/5) := by
  sorry

end equivalent_operations_l3053_305334


namespace number_of_refills_l3053_305347

def total_spent : ℕ := 63
def cost_per_refill : ℕ := 21

theorem number_of_refills : total_spent / cost_per_refill = 3 := by
  sorry

end number_of_refills_l3053_305347


namespace parabola_shift_correct_l3053_305390

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The original parabola y = 4x^2 -/
def original_parabola : Parabola := { a := 4, b := 0, c := 0 }

/-- Shifts a parabola horizontally and vertically -/
def shift_parabola (p : Parabola) (h : ℝ) (v : ℝ) : Parabola :=
  { a := p.a
  , b := -2 * p.a * h
  , c := p.a * h^2 + p.c + v }

/-- The resulting parabola after shifting -/
def shifted_parabola : Parabola := shift_parabola original_parabola 9 6

theorem parabola_shift_correct :
  shifted_parabola = { a := 4, b := -72, c := 330 } := by sorry

end parabola_shift_correct_l3053_305390


namespace identity_function_unique_l3053_305397

def C : ℕ := 2022^2022

theorem identity_function_unique :
  ∀ f : ℕ → ℕ,
  (∀ x y : ℕ, x > 0 → y > 0 → 
    ∃ k : ℕ, k > 0 ∧ k ≤ C ∧ f (x + y) = f x + k * f y) →
  f = id :=
by sorry

end identity_function_unique_l3053_305397


namespace distance_to_optimal_shooting_point_l3053_305393

/-- Given a field with width 2b, a goal with width 2a, and a distance c to the sideline,
    prove that the distance x satisfying the conditions is √((b-c)^2 - a^2). -/
theorem distance_to_optimal_shooting_point (b a c x : ℝ) 
  (h1 : b > 0)
  (h2 : a > 0)
  (h3 : c ≥ 0)
  (h4 : c < b)
  (h5 : (b - c)^2 = a^2 + x^2) :
  x = Real.sqrt ((b - c)^2 - a^2) := by
sorry

end distance_to_optimal_shooting_point_l3053_305393


namespace division_problem_l3053_305359

theorem division_problem (x : ℤ) : (64 / x = 4) → x = 16 := by
  sorry

end division_problem_l3053_305359


namespace cookie_count_l3053_305302

theorem cookie_count (cookies_per_bag : ℕ) (num_bags : ℕ) : 
  cookies_per_bag = 41 → num_bags = 53 → cookies_per_bag * num_bags = 2173 := by
  sorry

end cookie_count_l3053_305302


namespace quadratic_inequality_range_l3053_305396

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 2 * a * x + 1 ≥ 0) ↔ (0 ≤ a ∧ a ≤ 1) :=
by sorry

end quadratic_inequality_range_l3053_305396


namespace coffee_consumption_l3053_305306

theorem coffee_consumption (initial_amount : ℝ) (first_fraction : ℝ) (second_fraction : ℝ) (final_amount : ℝ) : 
  initial_amount = 12 →
  first_fraction = 1/4 →
  second_fraction = 1/2 →
  final_amount = 1 →
  initial_amount - (first_fraction * initial_amount + second_fraction * initial_amount + final_amount) = 2 := by
sorry


end coffee_consumption_l3053_305306


namespace quadratic_inequality_solution_sets_l3053_305366

theorem quadratic_inequality_solution_sets (a b : ℝ) :
  (∀ x : ℝ, ax^2 + b*x + 2 > 0 ↔ -1 < x ∧ x < 2) →
  (∀ x : ℝ, 2*x^2 + b*x + a > 0 ↔ x < -1 ∨ x > 1/2) :=
by sorry

end quadratic_inequality_solution_sets_l3053_305366


namespace manny_has_more_ten_bills_l3053_305373

-- Define the number of bills each person has
def mandy_twenty_bills : ℕ := 3
def manny_fifty_bills : ℕ := 2

-- Define the value of each bill type
def twenty_bill_value : ℕ := 20
def fifty_bill_value : ℕ := 50
def ten_bill_value : ℕ := 10

-- Calculate the total value for each person
def mandy_total : ℕ := mandy_twenty_bills * twenty_bill_value
def manny_total : ℕ := manny_fifty_bills * fifty_bill_value

-- Calculate the number of $10 bills each person can get
def mandy_ten_bills : ℕ := mandy_total / ten_bill_value
def manny_ten_bills : ℕ := manny_total / ten_bill_value

-- State the theorem
theorem manny_has_more_ten_bills : manny_ten_bills - mandy_ten_bills = 4 := by
  sorry

end manny_has_more_ten_bills_l3053_305373


namespace melanie_dimes_and_choiceland_coins_l3053_305353

/-- Proves the number of dimes Melanie has and their value in ChoiceLand coins -/
theorem melanie_dimes_and_choiceland_coins 
  (initial_dimes : ℕ) 
  (dad_dimes : ℕ) 
  (mom_dimes : ℕ) 
  (exchange_rate : ℚ) 
  (h1 : initial_dimes = 7)
  (h2 : dad_dimes = 8)
  (h3 : mom_dimes = 4)
  (h4 : exchange_rate = 5/2) : 
  (initial_dimes + dad_dimes + mom_dimes = 19) ∧ 
  ((initial_dimes + dad_dimes + mom_dimes : ℚ) * exchange_rate = 95/2) := by
sorry

end melanie_dimes_and_choiceland_coins_l3053_305353


namespace disc_interaction_conservation_l3053_305312

/-- Represents a disc with radius and angular velocity -/
structure Disc where
  radius : ℝ
  angularVelocity : ℝ

/-- Theorem: Conservation of angular momentum for two interacting discs -/
theorem disc_interaction_conservation
  (d1 d2 : Disc)
  (h_positive_radius : d1.radius > 0 ∧ d2.radius > 0)
  (h_same_material : True)  -- Placeholder for identical material property
  (h_same_thickness : True) -- Placeholder for identical thickness property
  (h_halt : True) -- Placeholder for the condition that both discs come to a halt
  : d1.angularVelocity * d1.radius ^ 3 = d2.angularVelocity * d2.radius ^ 3 := by
  sorry

end disc_interaction_conservation_l3053_305312


namespace stock_price_increase_l3053_305364

theorem stock_price_increase (x : ℝ) : 
  (1 + x / 100) * (1 - 25 / 100) * (1 + 30 / 100) = 117 / 100 → x = 20 := by
  sorry

end stock_price_increase_l3053_305364


namespace total_games_theorem_l3053_305387

/-- The total number of games played by Frankie and Carla -/
def total_games (carla_games frankie_games : ℕ) : ℕ := carla_games + frankie_games

/-- Theorem: Given that Carla won 20 games and Frankie won half as many games as Carla,
    the total number of games played is 30. -/
theorem total_games_theorem :
  ∀ (carla_games frankie_games : ℕ),
    carla_games = 20 →
    frankie_games = carla_games / 2 →
    total_games carla_games frankie_games = 30 := by
  sorry

end total_games_theorem_l3053_305387


namespace bernoulli_inequality_l3053_305367

theorem bernoulli_inequality (c x : ℝ) (p : ℤ) 
  (hc : c > 0) (hp : p > 1) (hx1 : x > -1) (hx2 : x ≠ 0) : 
  (1 + x)^p > 1 + p * x := by
  sorry

end bernoulli_inequality_l3053_305367


namespace power_of_four_equality_l3053_305315

theorem power_of_four_equality (m : ℕ) : 4^m = 4 * 16^3 * 64^2 → m = 13 := by
  sorry

end power_of_four_equality_l3053_305315


namespace triangle_acute_from_inequalities_l3053_305318

theorem triangle_acute_from_inequalities (α β γ : Real) 
  (sum_angles : α + β + γ = Real.pi)
  (ineq1 : Real.sin α > Real.cos β)
  (ineq2 : Real.sin β > Real.cos γ)
  (ineq3 : Real.sin γ > Real.cos α) :
  α < Real.pi / 2 ∧ β < Real.pi / 2 ∧ γ < Real.pi / 2 := by
  sorry

end triangle_acute_from_inequalities_l3053_305318


namespace power_multiplication_calculate_expression_l3053_305338

theorem power_multiplication (a : ℕ) (m n : ℕ) :
  a * (a ^ n) = a ^ (n + 1) :=
by
  sorry

theorem calculate_expression : 
  3000 * (3000 ^ 1500) = 3000 ^ 1501 :=
by
  sorry

end power_multiplication_calculate_expression_l3053_305338


namespace impossible_to_empty_pile_l3053_305314

/-- Represents the state of three piles of stones -/
structure PileState where
  pile1 : ℕ
  pile2 : ℕ
  pile3 : ℕ

/-- Allowed operations on the piles -/
inductive Operation
  | Add : Fin 3 → Operation
  | Remove : Fin 3 → Operation

/-- Applies an operation to a PileState -/
def applyOperation (state : PileState) (op : Operation) : PileState :=
  match op with
  | Operation.Add i => 
      match i with
      | 0 => ⟨state.pile1 + state.pile2 + state.pile3, state.pile2, state.pile3⟩
      | 1 => ⟨state.pile1, state.pile2 + state.pile1 + state.pile3, state.pile3⟩
      | 2 => ⟨state.pile1, state.pile2, state.pile3 + state.pile1 + state.pile2⟩
  | Operation.Remove i =>
      match i with
      | 0 => ⟨state.pile1 - (state.pile2 + state.pile3), state.pile2, state.pile3⟩
      | 1 => ⟨state.pile1, state.pile2 - (state.pile1 + state.pile3), state.pile3⟩
      | 2 => ⟨state.pile1, state.pile2, state.pile3 - (state.pile1 + state.pile2)⟩

/-- Theorem stating that it's impossible to make a pile empty -/
theorem impossible_to_empty_pile (initialState : PileState) 
  (h1 : Odd initialState.pile1) 
  (h2 : Odd initialState.pile2) 
  (h3 : Odd initialState.pile3) :
  ∀ (ops : List Operation), 
    let finalState := ops.foldl applyOperation initialState
    ¬(finalState.pile1 = 0 ∨ finalState.pile2 = 0 ∨ finalState.pile3 = 0) := by
  sorry

end impossible_to_empty_pile_l3053_305314


namespace angle_B_measure_l3053_305383

/-- In a triangle ABC, given that the measures of angles A, B, C form a geometric progression
    and b^2 - a^2 = a*c, prove that the measure of angle B is 2π/7 -/
theorem angle_B_measure (A B C : ℝ) (a b c : ℝ) :
  A > 0 → B > 0 → C > 0 →
  a > 0 → b > 0 → c > 0 →
  A + B + C = π →
  ∃ (q : ℝ), q > 0 ∧ B = q * A ∧ C = q * B →
  b^2 - a^2 = a * c →
  B = 2 * π / 7 := by
sorry

end angle_B_measure_l3053_305383


namespace family_reunion_ratio_l3053_305384

theorem family_reunion_ratio (male_adults female_adults children total_adults total_people : ℕ) : 
  female_adults = male_adults + 50 →
  male_adults = 100 →
  total_adults = male_adults + female_adults →
  total_people = 750 →
  total_people = total_adults + children →
  (children : ℚ) / total_adults = 2 :=
by
  sorry

end family_reunion_ratio_l3053_305384


namespace inequality_implies_a_range_l3053_305322

theorem inequality_implies_a_range (a : ℝ) : 
  (∀ x ∈ Set.Ioo (0 : ℝ) (1/2), x^2 + 2*a*x + 1 ≥ 0) → a ≥ -5/4 := by
sorry

end inequality_implies_a_range_l3053_305322


namespace heavy_operator_daily_rate_l3053_305307

theorem heavy_operator_daily_rate
  (total_workers : ℕ)
  (num_laborers : ℕ)
  (laborer_rate : ℕ)
  (total_payroll : ℕ)
  (h1 : total_workers = 31)
  (h2 : num_laborers = 1)
  (h3 : laborer_rate = 82)
  (h4 : total_payroll = 3952) :
  (total_payroll - num_laborers * laborer_rate) / (total_workers - num_laborers) = 129 := by
sorry

end heavy_operator_daily_rate_l3053_305307


namespace atomic_numbers_descending_l3053_305371

/-- Atomic number of Chlorine -/
def atomic_number_Cl : ℕ := 17

/-- Atomic number of Oxygen -/
def atomic_number_O : ℕ := 8

/-- Atomic number of Lithium -/
def atomic_number_Li : ℕ := 3

/-- Theorem stating that the atomic numbers of Cl, O, and Li are in descending order -/
theorem atomic_numbers_descending :
  atomic_number_Cl > atomic_number_O ∧ atomic_number_O > atomic_number_Li :=
sorry

end atomic_numbers_descending_l3053_305371


namespace glass_pane_area_is_4900_l3053_305399

/-- The area of a square glass pane inside a square frame -/
def glass_pane_area (frame_side_length : ℝ) (frame_width : ℝ) : ℝ :=
  (frame_side_length - 2 * frame_width) ^ 2

/-- Theorem: The area of the square glass pane is 4900 cm² -/
theorem glass_pane_area_is_4900 :
  glass_pane_area 100 15 = 4900 := by
  sorry

end glass_pane_area_is_4900_l3053_305399


namespace chromium_percentage_proof_l3053_305342

/-- The percentage of chromium in the first alloy -/
def chromium_percentage_1 : ℝ := 10

/-- The percentage of chromium in the second alloy -/
def chromium_percentage_2 : ℝ := 8

/-- The weight of the first alloy in kg -/
def weight_1 : ℝ := 15

/-- The weight of the second alloy in kg -/
def weight_2 : ℝ := 35

/-- The percentage of chromium in the new alloy -/
def chromium_percentage_new : ℝ := 8.6

/-- The total weight of the new alloy in kg -/
def total_weight : ℝ := weight_1 + weight_2

theorem chromium_percentage_proof :
  (chromium_percentage_1 / 100) * weight_1 + (chromium_percentage_2 / 100) * weight_2 =
  (chromium_percentage_new / 100) * total_weight :=
by sorry

end chromium_percentage_proof_l3053_305342


namespace first_player_min_score_l3053_305370

/-- Represents a game state with remaining numbers -/
def GameState := List Nat

/-- Removes a list of numbers from the game state -/
def removeNumbers (state : GameState) (toRemove : List Nat) : GameState :=
  state.filter (λ n => n ∉ toRemove)

/-- Calculates the score based on the two remaining numbers -/
def calculateScore (state : GameState) : Nat :=
  if state.length = 2 then
    state.maximum.getD 0 - state.minimum.getD 0
  else
    0

/-- Represents a player's strategy -/
def Strategy := GameState → List Nat

/-- Simulates a game given two strategies -/
def playGame (player1 : Strategy) (player2 : Strategy) : Nat :=
  let initialState : GameState := List.range 101
  let finalState := (List.range 11).foldl
    (λ state round =>
      let state' := removeNumbers state (player1 state)
      removeNumbers state' (player2 state'))
    initialState
  calculateScore finalState

/-- Theorem: The first player can always ensure a score of at least 52 -/
theorem first_player_min_score :
  ∃ (player1 : Strategy), ∀ (player2 : Strategy), playGame player1 player2 ≥ 52 := by
  sorry


end first_player_min_score_l3053_305370


namespace rectangle_fold_trapezoid_l3053_305340

/-- 
Given a rectangle with sides a and b, if folding it along its diagonal 
creates an isosceles trapezoid with three equal sides and the fourth side 
of length 10√3, then a = 15 and b = 5√3.
-/
theorem rectangle_fold_trapezoid (a b : ℝ) 
  (h_rect : a > 0 ∧ b > 0)
  (h_fold : ∃ (x y z : ℝ), x = y ∧ y = z ∧ 
    x^2 + y^2 = a^2 + b^2 ∧ 
    z^2 + (10 * Real.sqrt 3)^2 = a^2 + b^2) : 
  a = 15 ∧ b = 5 * Real.sqrt 3 := by
sorry

end rectangle_fold_trapezoid_l3053_305340


namespace binary_multiplication_l3053_305389

-- Define binary numbers as natural numbers
def binary_1101 : ℕ := 13  -- 1101₂ in decimal
def binary_111 : ℕ := 7    -- 111₂ in decimal

-- Define the expected result
def expected_result : ℕ := 79  -- 1001111₂ in decimal

-- Theorem statement
theorem binary_multiplication :
  binary_1101 * binary_111 = expected_result := by
  sorry

end binary_multiplication_l3053_305389


namespace sell_all_cars_in_five_months_l3053_305337

/-- Calculates the number of months needed to sell all cars -/
def months_to_sell_cars (total_cars : ℕ) (num_salespeople : ℕ) (cars_per_salesperson_per_month : ℕ) : ℕ :=
  total_cars / (num_salespeople * cars_per_salesperson_per_month)

/-- Proves that it takes 5 months to sell all cars under given conditions -/
theorem sell_all_cars_in_five_months : 
  months_to_sell_cars 500 10 10 = 5 := by
  sorry

#eval months_to_sell_cars 500 10 10

end sell_all_cars_in_five_months_l3053_305337


namespace square_triangle_equal_area_l3053_305372

theorem square_triangle_equal_area (square_perimeter : ℝ) (triangle_height : ℝ) (x : ℝ) : 
  square_perimeter = 64 →
  triangle_height = 36 →
  (square_perimeter / 4) ^ 2 = (1 / 2) * x * triangle_height →
  x = 128 / 9 := by
sorry

end square_triangle_equal_area_l3053_305372


namespace bus_speed_problem_l3053_305374

/-- Given two buses traveling in opposite directions, this theorem proves
    the speed of the second bus given the conditions of the problem. -/
theorem bus_speed_problem (east_speed : ℝ) (time : ℝ) (total_distance : ℝ)
  (h1 : east_speed = 55)
  (h2 : time = 4)
  (h3 : total_distance = 460) :
  ∃ west_speed : ℝ, 
    west_speed * time + east_speed * time = total_distance ∧
    west_speed = 60 :=
by sorry

end bus_speed_problem_l3053_305374


namespace weight_of_b_l3053_305398

-- Define variables for weights and heights
variable (W_a W_b W_c : ℚ)
variable (h_a h_b h_c : ℚ)

-- Define the conditions
def condition1 : Prop := (W_a + W_b + W_c) / 3 = 45
def condition2 : Prop := (W_a + W_b) / 2 = 40
def condition3 : Prop := (W_b + W_c) / 2 = 47
def condition4 : Prop := h_a + h_c = 2 * h_b
def condition5 : Prop := ∃ (n : ℤ), W_a + W_b + W_c = 2 * n + 1

-- Theorem statement
theorem weight_of_b 
  (h1 : condition1 W_a W_b W_c)
  (h2 : condition2 W_a W_b)
  (h3 : condition3 W_b W_c)
  (h4 : condition4 h_a h_b h_c)
  (h5 : condition5 W_a W_b W_c) :
  W_b = 39 := by
  sorry

end weight_of_b_l3053_305398


namespace quadratic_equation_two_real_roots_l3053_305311

theorem quadratic_equation_two_real_roots (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ (k + 1) * x^2 - 2 * x + 1 = 0 ∧ (k + 1) * y^2 - 2 * y + 1 = 0) ↔
  (k ≤ 0 ∧ k ≠ -1) :=
sorry

end quadratic_equation_two_real_roots_l3053_305311


namespace sphere_radius_when_area_equals_volume_l3053_305328

theorem sphere_radius_when_area_equals_volume (R : ℝ) : R > 0 →
  (4 * Real.pi * R^2 = (4 / 3) * Real.pi * R^3) → R = 3 := by
  sorry

end sphere_radius_when_area_equals_volume_l3053_305328


namespace product_of_solution_l3053_305305

open Real

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Define the fractional part function
noncomputable def frac (x : ℝ) : ℝ := x - (floor x)

-- State the theorem
theorem product_of_solution (x y : ℝ) 
  (eq1 : (floor x : ℝ) + frac y = 1.7)
  (eq2 : frac x + (floor y : ℝ) = 3.6) :
  x * y = 5.92 := by
  sorry

end product_of_solution_l3053_305305


namespace sum_of_coefficients_is_zero_l3053_305343

-- Define the functions f and g
def f (A B x : ℝ) : ℝ := A * x^2 + B * x + 1
def g (A B x : ℝ) : ℝ := B * x^2 + A * x + 1

-- State the theorem
theorem sum_of_coefficients_is_zero (A B : ℝ) :
  A ≠ B →
  (∀ x, f A B (g A B x) - g A B (f A B x) = x^4 + 5*x^3 + x^2 - 4*x) →
  A + B = 0 := by
sorry

end sum_of_coefficients_is_zero_l3053_305343


namespace simple_sampling_methods_correct_l3053_305360

/-- The set of methods for implementing simple sampling -/
def SimpleSamplingMethods : Set String :=
  {"Lottery method", "Random number table method"}

/-- Theorem stating that the set of methods for implementing simple sampling
    contains exactly the lottery method and random number table method -/
theorem simple_sampling_methods_correct :
  SimpleSamplingMethods = {"Lottery method", "Random number table method"} := by
  sorry

end simple_sampling_methods_correct_l3053_305360


namespace gcd_from_lcm_and_ratio_l3053_305382

theorem gcd_from_lcm_and_ratio (A B : ℕ+) 
  (h_lcm : Nat.lcm A B = 180)
  (h_ratio : A * 6 = B * 5) :
  Nat.gcd A B = 6 := by
  sorry

end gcd_from_lcm_and_ratio_l3053_305382


namespace inequality_proof_l3053_305354

theorem inequality_proof (x₁ x₂ x₃ : ℝ) 
  (h_pos₁ : x₁ > 0) (h_pos₂ : x₂ > 0) (h_pos₃ : x₃ > 0)
  (h_sum : x₁ + x₂ + x₃ = 1) : 
  x₂^2 / x₁ + x₃^2 / x₂ + x₁^2 / x₃ ≥ 1 := by
  sorry

end inequality_proof_l3053_305354


namespace existence_of_special_integers_l3053_305369

theorem existence_of_special_integers : ∃ (a b c : ℤ), 
  (a > 2011) ∧ (b > 2011) ∧ (c > 2011) ∧
  ∃ (n : ℕ), (((a + Real.sqrt b)^c : ℝ) / 10000 - n : ℝ) = 0.20102011 := by
  sorry

end existence_of_special_integers_l3053_305369


namespace total_time_is_four_hours_l3053_305327

def first_movie_length : ℚ := 3/2 -- 1.5 hours
def second_movie_length : ℚ := first_movie_length + 1/2 -- 30 minutes longer
def popcorn_time : ℚ := 1/6 -- 10 minutes in hours
def fries_time : ℚ := 2 * popcorn_time -- twice as long as popcorn time

def total_time : ℚ := first_movie_length + second_movie_length + popcorn_time + fries_time

theorem total_time_is_four_hours : total_time = 4 := by
  sorry

end total_time_is_four_hours_l3053_305327


namespace regular_polygon_interior_angle_sum_l3053_305350

theorem regular_polygon_interior_angle_sum 
  (n : ℕ) 
  (h_exterior : (360 : ℝ) / n = 45) : 
  (n - 2) * 180 = 1080 :=
by sorry

end regular_polygon_interior_angle_sum_l3053_305350


namespace intersection_of_P_and_Q_l3053_305395

def P : Set ℝ := {1, 3, 5, 7}
def Q : Set ℝ := {x | 2 * x - 1 > 5}

theorem intersection_of_P_and_Q : P ∩ Q = {5, 7} := by sorry

end intersection_of_P_and_Q_l3053_305395


namespace range_of_a_minus_b_l3053_305356

theorem range_of_a_minus_b (a b : ℝ) (ha : -1 < a ∧ a < 1) (hb : 1 < b ∧ b < 3) :
  -4 < a - b ∧ a - b < 0 := by
sorry

end range_of_a_minus_b_l3053_305356


namespace sum_of_binary_digits_345_l3053_305320

/-- Returns the binary representation of a natural number as a list of bits -/
def toBinary (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec go (m : ℕ) : List ℕ :=
    if m = 0 then [] else (m % 2) :: go (m / 2)
  go n

/-- Sums the elements of a list of natural numbers -/
def sumList (l : List ℕ) : ℕ :=
  l.foldl (· + ·) 0

theorem sum_of_binary_digits_345 : sumList (toBinary 345) = 5 := by
  sorry

end sum_of_binary_digits_345_l3053_305320


namespace opposite_is_five_l3053_305365

theorem opposite_is_five (x : ℝ) : -x = 5 → x = -5 := by
  sorry

end opposite_is_five_l3053_305365


namespace unique_max_sum_pair_l3053_305332

theorem unique_max_sum_pair :
  ∃! (x y : ℕ), 
    (∃ (k : ℕ), 19 * x + 95 * y = k * k) ∧
    19 * x + 95 * y ≤ 1995 ∧
    (∀ (a b : ℕ), (∃ (m : ℕ), 19 * a + 95 * b = m * m) → 
      19 * a + 95 * b ≤ 1995 → 
      a + b ≤ x + y) :=
by sorry

end unique_max_sum_pair_l3053_305332


namespace root_product_theorem_l3053_305319

theorem root_product_theorem (a b : ℂ) : 
  (a^4 + a^3 - 1 = 0) → 
  (b^4 + b^3 - 1 = 0) → 
  ((a*b)^6 + (a*b)^4 + (a*b)^3 - (a*b)^2 - 1 = 0) := by
  sorry

end root_product_theorem_l3053_305319


namespace range_of_expression_l3053_305326

theorem range_of_expression (α β : Real) 
  (h1 : 0 < α ∧ α < π/2) 
  (h2 : 0 < β ∧ β < π/2) : 
  -π/6 < 2*α - β/3 ∧ 2*α - β/3 < π := by
  sorry

end range_of_expression_l3053_305326


namespace arithmetic_progression_of_primes_l3053_305349

theorem arithmetic_progression_of_primes (a : ℕ → ℕ) (d : ℕ) :
  (∀ i ∈ Finset.range 15, Nat.Prime (a i)) →
  (∀ i ∈ Finset.range 14, a (i + 1) = a i + d) →
  d > 0 →
  a 0 > 15 →
  d > 30000 := by
sorry

end arithmetic_progression_of_primes_l3053_305349


namespace min_value_of_fraction_l3053_305344

theorem min_value_of_fraction (a : ℝ) (h : a > 1) : 
  ∀ x : ℝ, x > 1 → (x^2 - x + 1) / (x - 1) ≥ (a^2 - a + 1) / (a - 1) → 
  (a^2 - a + 1) / (a - 1) = 3 :=
by sorry

end min_value_of_fraction_l3053_305344


namespace group_size_calculation_l3053_305375

theorem group_size_calculation (n : ℕ) : 
  (n * 14 + 34) / (n + 1) = 16 → n = 9 := by
  sorry

end group_size_calculation_l3053_305375


namespace remainder_a_fourth_plus_four_l3053_305376

theorem remainder_a_fourth_plus_four (a : ℤ) (h : ¬ (5 ∣ a)) : (a^4 + 4) % 5 = 0 := by
  sorry

end remainder_a_fourth_plus_four_l3053_305376


namespace composite_product_division_l3053_305316

def first_eight_composites : List Nat := [4, 6, 8, 9, 10, 12, 14, 15]
def next_eight_composites : List Nat := [16, 18, 20, 21, 22, 24, 25, 26]

def product_of_list (l : List Nat) : Nat :=
  l.foldl (· * ·) 1

theorem composite_product_division :
  (product_of_list first_eight_composites) / 
  (product_of_list next_eight_composites) = 1 / 3120 := by
  sorry

end composite_product_division_l3053_305316


namespace find_N_l3053_305308

theorem find_N : ∃ N : ℤ, (10 + 11 + 12) / 3 = (2010 + 2011 + 2012 + N) / 4 → N = -5989 := by
  sorry

end find_N_l3053_305308


namespace total_orders_filled_l3053_305317

/-- Represents the price of a catfish dinner in dollars -/
def catfish_price : ℚ := 6

/-- Represents the price of a popcorn shrimp dinner in dollars -/
def popcorn_shrimp_price : ℚ := 7/2

/-- Represents the total amount collected in dollars -/
def total_collected : ℚ := 267/2

/-- Represents the number of popcorn shrimp dinners sold -/
def popcorn_shrimp_orders : ℕ := 9

/-- Theorem stating that the total number of orders filled is 26 -/
theorem total_orders_filled : ∃ (catfish_orders : ℕ), 
  catfish_price * catfish_orders + popcorn_shrimp_price * popcorn_shrimp_orders = total_collected ∧ 
  catfish_orders + popcorn_shrimp_orders = 26 := by
  sorry

end total_orders_filled_l3053_305317


namespace calculate_expression_l3053_305351

theorem calculate_expression : -3^2 + |(-5)| - 18 * (-1/3)^2 = -6 := by
  sorry

end calculate_expression_l3053_305351

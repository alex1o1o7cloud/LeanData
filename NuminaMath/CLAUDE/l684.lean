import Mathlib

namespace two_digit_factorizations_of_2079_l684_68435

/-- A factorization of a number into two factors -/
structure Factorization :=
  (factor1 : ℕ)
  (factor2 : ℕ)

/-- Check if a number is two-digit (between 10 and 99, inclusive) -/
def isTwoDigit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- Check if a factorization is valid for 2079 with two-digit factors -/
def isValidFactorization (f : Factorization) : Prop :=
  f.factor1 * f.factor2 = 2079 ∧ isTwoDigit f.factor1 ∧ isTwoDigit f.factor2

/-- Two factorizations are considered equal if they have the same factors (in any order) -/
def factorizationEqual (f1 f2 : Factorization) : Prop :=
  (f1.factor1 = f2.factor1 ∧ f1.factor2 = f2.factor2) ∨
  (f1.factor1 = f2.factor2 ∧ f1.factor2 = f2.factor1)

/-- The main theorem: there are exactly 2 unique factorizations of 2079 into two-digit numbers -/
theorem two_digit_factorizations_of_2079 :
  ∃ (f1 f2 : Factorization),
    isValidFactorization f1 ∧
    isValidFactorization f2 ∧
    ¬factorizationEqual f1 f2 ∧
    ∀ (f : Factorization), isValidFactorization f → (factorizationEqual f f1 ∨ factorizationEqual f f2) :=
  sorry

end two_digit_factorizations_of_2079_l684_68435


namespace sum_of_max_marks_is_1300_l684_68478

/-- Given the conditions for three tests (Math, Science, and English) in an examination,
    this theorem proves that the sum of maximum marks for all three tests is 1300. -/
theorem sum_of_max_marks_is_1300 
  (math_pass_percent : ℝ) 
  (math_marks_obtained : ℕ) 
  (math_marks_failed_by : ℕ)
  (science_pass_percent : ℝ) 
  (science_marks_obtained : ℕ) 
  (science_marks_failed_by : ℕ)
  (english_pass_percent : ℝ) 
  (english_marks_obtained : ℕ) 
  (english_marks_failed_by : ℕ)
  (h_math_percent : math_pass_percent = 0.3)
  (h_science_percent : science_pass_percent = 0.5)
  (h_english_percent : english_pass_percent = 0.4)
  (h_math_marks : math_marks_obtained = 80 ∧ math_marks_failed_by = 100)
  (h_science_marks : science_marks_obtained = 120 ∧ science_marks_failed_by = 80)
  (h_english_marks : english_marks_obtained = 60 ∧ english_marks_failed_by = 60) :
  ↑((math_marks_obtained + math_marks_failed_by) / math_pass_percent +
    (science_marks_obtained + science_marks_failed_by) / science_pass_percent +
    (english_marks_obtained + english_marks_failed_by) / english_pass_percent) = 1300 :=
by sorry


end sum_of_max_marks_is_1300_l684_68478


namespace jennifer_discount_is_28_l684_68409

/-- Calculates the discount for whole milk based on the number of cans purchased -/
def whole_milk_discount (cans : ℕ) : ℕ := (cans / 10) * 4

/-- Calculates the discount for almond milk based on the number of cans purchased -/
def almond_milk_discount (cans : ℕ) : ℕ := 
  ((cans / 7) * 3) + ((cans % 7) / 3)

/-- Represents Jennifer's milk purchase and calculates her total discount -/
def jennifer_discount : ℕ :=
  let initial_whole_milk := 40
  let mark_whole_milk := 30
  let mark_skim_milk := 15
  let additional_almond_milk := (mark_whole_milk / 3) * 2
  let additional_whole_milk := (mark_skim_milk / 5) * 4
  let total_whole_milk := initial_whole_milk + additional_whole_milk
  let total_almond_milk := additional_almond_milk
  whole_milk_discount total_whole_milk + almond_milk_discount total_almond_milk

theorem jennifer_discount_is_28 : jennifer_discount = 28 := by
  sorry

#eval jennifer_discount

end jennifer_discount_is_28_l684_68409


namespace max_a_for_quadratic_inequality_l684_68465

theorem max_a_for_quadratic_inequality :
  (∃ (a : ℝ), ∀ (x : ℝ), x^2 - 2*x - a ≥ 0) ∧
  (∀ (a : ℝ), (∀ (x : ℝ), x^2 - 2*x - a ≥ 0) → a ≤ -1) ∧
  (∀ (x : ℝ), x^2 - 2*x - (-1) ≥ 0) :=
by sorry

end max_a_for_quadratic_inequality_l684_68465


namespace jackson_percentage_difference_l684_68479

/-- Represents the count of birds seen by a person -/
structure BirdCount where
  robins : ℕ
  cardinals : ℕ
  blueJays : ℕ
  goldfinches : ℕ
  starlings : ℕ

/-- Calculates the total number of birds seen by a person -/
def totalBirds (count : BirdCount) : ℕ :=
  count.robins + count.cardinals + count.blueJays + count.goldfinches + count.starlings

/-- Calculates the percentage difference from the average -/
def percentageDifference (individual : ℕ) (average : ℚ) : ℚ :=
  ((individual : ℚ) - average) / average * 100

theorem jackson_percentage_difference :
  let gabrielle := BirdCount.mk 7 5 4 3 6
  let chase := BirdCount.mk 4 3 4 2 1
  let maria := BirdCount.mk 5 3 2 4 7
  let jackson := BirdCount.mk 6 2 3 5 2
  let total := totalBirds gabrielle + totalBirds chase + totalBirds maria + totalBirds jackson
  let average : ℚ := (total : ℚ) / 4
  abs (percentageDifference (totalBirds jackson) average - (-7.69)) < 0.01 := by
  sorry

end jackson_percentage_difference_l684_68479


namespace toms_game_sale_l684_68457

/-- Calculates the sale amount of games given initial cost, value increase factor, and sale percentage -/
def gameSaleAmount (initialCost : ℝ) (valueIncreaseFactor : ℝ) (salePercentage : ℝ) : ℝ :=
  initialCost * valueIncreaseFactor * salePercentage

/-- Proves that Tom's game sale amount is $240 given the specified conditions -/
theorem toms_game_sale : gameSaleAmount 200 3 0.4 = 240 := by
  sorry

end toms_game_sale_l684_68457


namespace infinite_triples_sum_of_squares_l684_68447

/-- A number that can be expressed as the sum of one or two squares. -/
def IsSumOfTwoSquares (k : ℤ) : Prop :=
  ∃ a b : ℤ, k = a^2 + b^2

theorem infinite_triples_sum_of_squares (n : ℤ) :
  let N := 2 * n^2 * (n + 1)^2
  IsSumOfTwoSquares N ∧
  IsSumOfTwoSquares (N + 1) ∧
  IsSumOfTwoSquares (N + 2) := by
  sorry


end infinite_triples_sum_of_squares_l684_68447


namespace rectangle_area_difference_l684_68456

/-- The difference in area between two rectangles, where one rectangle's dimensions are 1 cm less
    than the other's in both length and width, is equal to the sum of the larger rectangle's
    length and width, minus 1. -/
theorem rectangle_area_difference (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  x * y - (x - 1) * (y - 1) = x + y - 1 := by
  sorry

end rectangle_area_difference_l684_68456


namespace smallest_number_proof_l684_68499

def smallest_number : ℕ := 271562

theorem smallest_number_proof :
  smallest_number = 271562 ∧
  ∃ k : ℕ, (smallest_number - 18) = k * lcm 14 (lcm 26 28) ∧
  k = 746 ∧
  ∀ y : ℕ, y < smallest_number →
    ¬(∃ m : ℕ, (y - 18) = m * lcm 14 (lcm 26 28) ∧ m = 746) :=
by sorry

end smallest_number_proof_l684_68499


namespace household_size_proof_l684_68490

/-- The number of slices of bread consumed by each member daily. -/
def daily_consumption : ℕ := 5

/-- The number of slices in a loaf of bread. -/
def slices_per_loaf : ℕ := 12

/-- The number of loaves that last for 3 days. -/
def loaves_for_three_days : ℕ := 5

/-- The number of days the loaves last. -/
def days : ℕ := 3

/-- The number of members in the household. -/
def household_members : ℕ := 4

theorem household_size_proof :
  household_members * daily_consumption * days = loaves_for_three_days * slices_per_loaf :=
by sorry

end household_size_proof_l684_68490


namespace pet_ownership_l684_68459

theorem pet_ownership (total_students : ℕ) 
  (dog_owners cat_owners bird_owners fish_only_owners no_pet_owners : ℕ) : 
  total_students = 40 →
  dog_owners = (40 * 5) / 8 →
  cat_owners = 40 / 2 →
  bird_owners = 40 / 4 →
  fish_only_owners = 8 →
  no_pet_owners = 6 →
  ∃ (all_pet_owners : ℕ), all_pet_owners = 6 ∧
    all_pet_owners + fish_only_owners + no_pet_owners ≤ total_students :=
by sorry

end pet_ownership_l684_68459


namespace remainder_approximation_l684_68488

/-- Given two positive real numbers satisfying certain conditions, 
    prove that the remainder of their division is approximately 15. -/
theorem remainder_approximation (L S : ℝ) (hL : L > 0) (hS : S > 0) 
    (h_diff : L - S = 1365)
    (h_approx : |L - 1542.857| < 0.001)
    (h_div : ∃ R : ℝ, R ≥ 0 ∧ L = 8 * S + R) : 
  ∃ R : ℝ, R ≥ 0 ∧ L = 8 * S + R ∧ |R - 15| < 0.1 := by
  sorry

end remainder_approximation_l684_68488


namespace apple_count_difference_l684_68450

/-- The number of green apples initially in the store -/
def initial_green_apples : ℕ := 32

/-- The number of additional red apples compared to green apples initially -/
def red_apple_surplus : ℕ := 200

/-- The number of green apples delivered by the truck -/
def delivered_green_apples : ℕ := 340

/-- The final difference between green and red apples -/
def final_green_red_difference : ℤ := 140

theorem apple_count_difference :
  (initial_green_apples + delivered_green_apples : ℤ) - 
  (initial_green_apples + red_apple_surplus) = 
  final_green_red_difference :=
by sorry

end apple_count_difference_l684_68450


namespace floor_equation_solution_l684_68414

theorem floor_equation_solution (a b : ℝ) : 
  (∀ n : ℕ+, a * ⌊b * n⌋ = b * ⌊a * n⌋) ↔ 
  (a = 0 ∨ b = 0 ∨ (a = b ∧ ∃ m : ℤ, a = m)) := by
sorry

end floor_equation_solution_l684_68414


namespace t_is_perfect_square_l684_68472

theorem t_is_perfect_square (n : ℕ+) (t : ℕ+) (h : t = 2 + 2 * Real.sqrt (1 + 12 * n.val ^ 2)) :
  ∃ (x : ℕ), t = x ^ 2 := by
  sorry

end t_is_perfect_square_l684_68472


namespace lucy_fish_goal_l684_68467

/-- The number of fish Lucy currently has -/
def current_fish : ℕ := 212

/-- The number of additional fish Lucy needs to buy -/
def additional_fish : ℕ := 68

/-- The total number of fish Lucy wants to have -/
def total_fish : ℕ := current_fish + additional_fish

theorem lucy_fish_goal : total_fish = 280 := by
  sorry

end lucy_fish_goal_l684_68467


namespace supremum_of_expression_l684_68431

theorem supremum_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  -1 / (2 * a) - 2 / b ≤ -9 / 2 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + b₀ = 1 ∧ -1 / (2 * a₀) - 2 / b₀ = -9 / 2 :=
sorry

end supremum_of_expression_l684_68431


namespace incircle_area_of_triangle_PF1F2_l684_68407

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 / 24 = 1

-- Define the foci
def F1 : ℝ × ℝ := (-5, 0)
def F2 : ℝ × ℝ := (5, 0)

-- Define a point on the hyperbola in the first quadrant
def P : ℝ × ℝ := sorry

-- Distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem incircle_area_of_triangle_PF1F2 :
  hyperbola P.1 P.2 ∧
  P.1 > 0 ∧ P.2 > 0 ∧
  distance P F1 / distance P F2 = 4 / 3 →
  ∃ (r : ℝ), r^2 * π = 4 * π ∧
  r * (distance P F1 + distance P F2 + distance F1 F2) = distance P F1 * distance P F2 :=
by sorry

end incircle_area_of_triangle_PF1F2_l684_68407


namespace inequality_solution_set_l684_68443

theorem inequality_solution_set (a : ℝ) (h : a < -1) :
  {x : ℝ | (a * x - 1) / (x + 1) < 0} = {x : ℝ | x < -1 ∨ x > 1/a} := by
  sorry

end inequality_solution_set_l684_68443


namespace income_182400_max_income_l684_68415

/-- Represents the income function for the large grain grower --/
def income_function (original_land : ℝ) (original_income_per_mu : ℝ) (additional_land : ℝ) : ℝ :=
  original_land * original_income_per_mu + additional_land * (original_income_per_mu - 2 * additional_land)

/-- Theorem for the total income of 182,400 yuan --/
theorem income_182400 (original_land : ℝ) (original_income_per_mu : ℝ) :
  original_land = 360 ∧ original_income_per_mu = 440 →
  (∃ x : ℝ, (income_function original_land original_income_per_mu x = 182400 ∧ (x = 100 ∨ x = 120))) :=
sorry

/-- Theorem for the maximum total income --/
theorem max_income (original_land : ℝ) (original_income_per_mu : ℝ) :
  original_land = 360 ∧ original_income_per_mu = 440 →
  (∃ x : ℝ, (∀ y : ℝ, income_function original_land original_income_per_mu x ≥ income_function original_land original_income_per_mu y) ∧
             x = 110 ∧
             income_function original_land original_income_per_mu x = 182600) :=
sorry

end income_182400_max_income_l684_68415


namespace circle_area_ratio_l684_68498

theorem circle_area_ratio (R_A R_B : ℝ) (h : R_A > 0 ∧ R_B > 0) : 
  (60 : ℝ) / 360 * (2 * Real.pi * R_A) = (40 : ℝ) / 360 * (2 * Real.pi * R_B) → 
  (R_A^2 * Real.pi) / (R_B^2 * Real.pi) = 9 / 4 := by
sorry

end circle_area_ratio_l684_68498


namespace quadratic_root_sum_product_l684_68452

theorem quadratic_root_sum_product (p q : ℝ) : 
  (∃ x y : ℝ, 3 * x^2 - p * x + q = 0 ∧ 3 * y^2 - p * y + q = 0 ∧ x + y = 8 ∧ x * y = 12) → 
  p + q = 60 := by
sorry

end quadratic_root_sum_product_l684_68452


namespace movie_theatre_revenue_l684_68426

/-- Calculates the total ticket revenue for a movie theatre session -/
theorem movie_theatre_revenue 
  (total_seats : ℕ) 
  (adult_price child_price : ℕ) 
  (num_children : ℕ) 
  (h_full : num_children ≤ total_seats) : 
  let num_adults := total_seats - num_children
  (num_adults * adult_price + num_children * child_price : ℕ) = 1124 :=
by
  sorry

#check movie_theatre_revenue 250 6 4 188

end movie_theatre_revenue_l684_68426


namespace max_volume_is_three_l684_68442

/-- Represents a rectangular solid with given constraints -/
structure RectangularSolid where
  width : ℝ
  length : ℝ
  height : ℝ
  sum_of_edges : width * 4 + length * 4 + height * 4 = 18
  length_width_ratio : length = 2 * width

/-- The volume of a rectangular solid -/
def volume (r : RectangularSolid) : ℝ := r.width * r.length * r.height

/-- Theorem stating that the maximum volume of the rectangular solid is 3 -/
theorem max_volume_is_three :
  ∃ (r : RectangularSolid), volume r = 3 ∧ ∀ (s : RectangularSolid), volume s ≤ 3 := by
  sorry

end max_volume_is_three_l684_68442


namespace square_of_sum_l684_68483

theorem square_of_sum (a b : ℝ) : a^2 + b^2 + 2*a*b = (a + b)^2 := by
  sorry

end square_of_sum_l684_68483


namespace min_value_expression_l684_68463

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 27) :
  x^2 + 6*x*y + 9*y^2 + (3/2)*z^2 ≥ 102 ∧
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ * y₀ * z₀ = 27 ∧
    x₀^2 + 6*x₀*y₀ + 9*y₀^2 + (3/2)*z₀^2 = 102 :=
by sorry

end min_value_expression_l684_68463


namespace polynomial_linear_if_all_powers_l684_68433

/-- A sequence defined by a polynomial recurrence -/
def PolynomialSequence (P : ℕ → ℕ) (n : ℕ) : ℕ → ℕ
  | 0 => n
  | k + 1 => P (PolynomialSequence P n k)

/-- Predicate to check if a number is a perfect power greater than 1 -/
def IsPerfectPower (m : ℕ) : Prop :=
  ∃ (b : ℕ) (k : ℕ), k > 1 ∧ m = k^b

theorem polynomial_linear_if_all_powers (P : ℕ → ℕ) (n : ℕ) :
  (∀ (x y : ℕ), ∃ (a b c : ℤ), P x - P y = a * (x - y) + b * x + c) →
  (∀ (b : ℕ), ∃ (k : ℕ), IsPerfectPower (PolynomialSequence P n k)) →
  ∃ (m q : ℤ), ∀ (x : ℕ), P x = m * x + q :=
sorry

end polynomial_linear_if_all_powers_l684_68433


namespace polynomial_division_problem_l684_68444

theorem polynomial_division_problem (x : ℝ) :
  let quotient := 2 * x + 6
  let divisor := x - 5
  let remainder := 2
  let polynomial := 2 * x^2 - 4 * x - 28
  polynomial = quotient * divisor + remainder := by sorry

end polynomial_division_problem_l684_68444


namespace special_permutations_count_l684_68481

/-- The number of permutations of 5 distinct elements where 2 specific elements are not placed at the ends -/
def special_permutations : ℕ :=
  -- Number of ways to choose 2 positions out of 3 for A and E
  (3 * 2) *
  -- Number of ways to arrange the remaining 3 elements
  (3 * 2 * 1)

theorem special_permutations_count : special_permutations = 36 := by
  sorry

end special_permutations_count_l684_68481


namespace remainder_proof_l684_68413

theorem remainder_proof (g : ℕ) (h : g = 144) :
  (6215 % g = 23) ∧ (7373 % g = 29) ∧
  (∀ d : ℕ, d > g → (6215 % d ≠ 6215 % g ∨ 7373 % d ≠ 7373 % g)) := by
  sorry

end remainder_proof_l684_68413


namespace tourist_group_room_capacity_l684_68458

/-- Given a tourist group and room arrangements, calculate the capacity of small rooms -/
theorem tourist_group_room_capacity
  (total_people : ℕ)
  (large_room_capacity : ℕ)
  (large_rooms_rented : ℕ)
  (h1 : total_people = 26)
  (h2 : large_room_capacity = 3)
  (h3 : large_rooms_rented = 8)
  : ∃ (small_room_capacity : ℕ),
    small_room_capacity > 0 ∧
    small_room_capacity * (total_people - large_room_capacity * large_rooms_rented) = total_people - large_room_capacity * large_rooms_rented ∧
    small_room_capacity = 2 :=
by sorry

end tourist_group_room_capacity_l684_68458


namespace power_product_eq_product_of_powers_l684_68494

theorem power_product_eq_product_of_powers (a b : ℝ) : (a * b)^2 = a^2 * b^2 := by
  sorry

end power_product_eq_product_of_powers_l684_68494


namespace simplify_fraction_l684_68429

theorem simplify_fraction (x y : ℝ) (hx : x = 3) (hy : y = 2) :
  15 * x^2 * y^3 / (9 * x * y^2) = 10 := by
  sorry

end simplify_fraction_l684_68429


namespace multiply_mixed_number_l684_68473

theorem multiply_mixed_number : 7 * (9 + 2/5) = 65 + 4/5 := by
  sorry

end multiply_mixed_number_l684_68473


namespace min_value_sum_reciprocals_l684_68475

theorem min_value_sum_reciprocals (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : x + y + z = 3) :
  1 / (x + 3 * y) + 1 / (y + 3 * z) + 1 / (z + 3 * x) ≥ 1 ∧
  (1 / (x + 3 * y) + 1 / (y + 3 * z) + 1 / (z + 3 * x) = 1 ↔ x = 1 ∧ y = 1 ∧ z = 1) :=
by sorry

end min_value_sum_reciprocals_l684_68475


namespace parallelogram_xy_product_l684_68484

/-- A parallelogram with side lengths given in terms of x and y -/
structure Parallelogram (x y : ℝ) :=
  (ef : ℝ)
  (fg : ℝ)
  (gh : ℝ)
  (he : ℝ)
  (ef_eq : ef = 42)
  (fg_eq : fg = 4 * y^3)
  (gh_eq : gh = 2 * x + 10)
  (he_eq : he = 32)
  (opposite_sides_equal : ef = gh ∧ fg = he)

/-- The product of x and y in the given parallelogram is 32 -/
theorem parallelogram_xy_product (x y : ℝ) (p : Parallelogram x y) :
  x * y = 32 :=
sorry

end parallelogram_xy_product_l684_68484


namespace smallest_lcm_with_gcd_5_l684_68497

theorem smallest_lcm_with_gcd_5 :
  ∃ (m n : ℕ), 
    1000 ≤ m ∧ m < 10000 ∧
    1000 ≤ n ∧ n < 10000 ∧
    Nat.gcd m n = 5 ∧
    Nat.lcm m n = 201000 ∧
    ∀ (p q : ℕ), 
      1000 ≤ p ∧ p < 10000 ∧
      1000 ≤ q ∧ q < 10000 ∧
      Nat.gcd p q = 5 →
      Nat.lcm p q ≥ 201000 :=
by
  sorry

end smallest_lcm_with_gcd_5_l684_68497


namespace jerome_trail_time_l684_68430

/-- The time it takes Jerome to run the trail -/
def jerome_time : ℝ := 6

/-- The time it takes Nero to run the trail -/
def nero_time : ℝ := 3

/-- Jerome's running speed in MPH -/
def jerome_speed : ℝ := 4

/-- Nero's running speed in MPH -/
def nero_speed : ℝ := 8

/-- Theorem stating that Jerome's time to run the trail is 6 hours -/
theorem jerome_trail_time : jerome_time = 6 := by sorry

end jerome_trail_time_l684_68430


namespace negative_two_exponent_division_l684_68419

theorem negative_two_exponent_division : 
  (-2: ℤ) ^ 2014 / (-2 : ℤ) ^ 2013 = -2 := by sorry

end negative_two_exponent_division_l684_68419


namespace ellipse_major_axis_length_l684_68416

/-- The length of the major axis of the ellipse x²/9 + y²/4 = 1 is 6 -/
theorem ellipse_major_axis_length : 
  let ellipse := fun (x y : ℝ) => x^2/9 + y^2/4 = 1
  ∃ (a b : ℝ), a > b ∧ a^2 = 9 ∧ b^2 = 4 ∧ 2*a = 6 :=
by sorry

end ellipse_major_axis_length_l684_68416


namespace inequality_proof_l684_68462

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 / (b^3 * c)) - (a / b^2) ≥ (c / b) - (c^2 / a) ∧
  ((a^2 / (b^3 * c)) - (a / b^2) = (c / b) - (c^2 / a) ↔ a = b * c) :=
by sorry

end inequality_proof_l684_68462


namespace festival_attendance_l684_68489

theorem festival_attendance (total_students : ℕ) (festival_attendees : ℕ) 
  (h1 : total_students = 1500)
  (h2 : festival_attendees = 900) : ℕ :=
by
  let girls : ℕ := sorry
  let boys : ℕ := sorry
  have h3 : girls + boys = total_students := sorry
  have h4 : (3 * girls / 4 : ℚ) + (2 * boys / 3 : ℚ) = festival_attendees := sorry
  have h5 : (3 * girls / 4 : ℕ) = 900 := sorry
  exact 900

#check festival_attendance

end festival_attendance_l684_68489


namespace initial_caterpillars_l684_68468

theorem initial_caterpillars (initial : ℕ) (added : ℕ) (left : ℕ) (remaining : ℕ) : 
  added = 4 → left = 8 → remaining = 10 → initial + added - left = remaining → initial = 14 := by
  sorry

end initial_caterpillars_l684_68468


namespace van_rental_equation_l684_68402

theorem van_rental_equation (x : ℕ+) :
  (180 : ℝ) / x - 180 / (x + 2) = 3 :=
by sorry

end van_rental_equation_l684_68402


namespace count_false_propositions_l684_68434

-- Define the original proposition
def original_prop (a : ℝ) : Prop := a > 1 → a > 2

-- Define the inverse proposition
def inverse_prop (a : ℝ) : Prop := ¬(a > 1) → ¬(a > 2)

-- Define the negation proposition
def negation_prop (a : ℝ) : Prop := ¬(a > 1 → a > 2)

-- Define the converse proposition
def converse_prop (a : ℝ) : Prop := a > 2 → a > 1

-- Count the number of false propositions
def count_false_props : ℕ := 2

-- Theorem statement
theorem count_false_propositions :
  count_false_props = 2 :=
sorry

end count_false_propositions_l684_68434


namespace river_width_calculation_l684_68410

/-- Given a river with depth, flow rate, and volume flow per minute, calculate its width. -/
theorem river_width_calculation (depth : ℝ) (flow_rate_kmph : ℝ) (volume_flow : ℝ) :
  depth = 3 →
  flow_rate_kmph = 2 →
  volume_flow = 3600 →
  (volume_flow / (depth * (flow_rate_kmph * 1000 / 60))) = 36 := by
  sorry

end river_width_calculation_l684_68410


namespace decimal_division_to_percentage_l684_68460

theorem decimal_division_to_percentage : (0.15 / 0.005) * 100 = 3000 := by
  sorry

end decimal_division_to_percentage_l684_68460


namespace triangle_area_isosceles_l684_68492

/-- The area of a triangle with two sides of length 30 and one side of length 40 -/
theorem triangle_area_isosceles (a b c : ℝ) (h1 : a = 30) (h2 : b = 30) (h3 : c = 40) : 
  ∃ area : ℝ, abs (area - Real.sqrt (50 * (50 - a) * (50 - b) * (50 - c))) < 0.01 ∧ 
  446.99 < area ∧ area < 447.01 := by
sorry


end triangle_area_isosceles_l684_68492


namespace marty_stripes_l684_68424

/-- The number of narrow black stripes on Marty the zebra -/
def narrow_black_stripes : ℕ := 8

/-- The number of wide black stripes on Marty the zebra -/
def wide_black_stripes : ℕ := sorry

/-- The number of white stripes on Marty the zebra -/
def white_stripes : ℕ := wide_black_stripes + 7

/-- The total number of black stripes on Marty the zebra -/
def total_black_stripes : ℕ := wide_black_stripes + narrow_black_stripes

theorem marty_stripes : 
  total_black_stripes = white_stripes + 1 → 
  narrow_black_stripes = 8 := by
  sorry

end marty_stripes_l684_68424


namespace calculation_proof_l684_68408

theorem calculation_proof : 101 * 102^2 - 101 * 98^2 = 80800 := by
  sorry

end calculation_proof_l684_68408


namespace cut_tetrahedron_edge_count_l684_68445

/-- Represents a regular tetrahedron with its vertices cut off. -/
structure CutTetrahedron where
  /-- The number of vertices in the original tetrahedron -/
  original_vertices : Nat
  /-- The number of edges in the original tetrahedron -/
  original_edges : Nat
  /-- The number of new edges created by each cut -/
  new_edges_per_cut : Nat
  /-- The cutting planes do not intersect on the solid -/
  non_intersecting_cuts : Prop

/-- The number of edges in the new figure after cutting off each vertex -/
def edge_count (t : CutTetrahedron) : Nat :=
  t.original_edges + t.original_vertices * t.new_edges_per_cut

/-- Theorem stating that a regular tetrahedron with its vertices cut off has 18 edges -/
theorem cut_tetrahedron_edge_count :
  ∀ (t : CutTetrahedron),
    t.original_vertices = 4 →
    t.original_edges = 6 →
    t.new_edges_per_cut = 3 →
    t.non_intersecting_cuts →
    edge_count t = 18 :=
  sorry

end cut_tetrahedron_edge_count_l684_68445


namespace f_g_properties_l684_68476

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := x + a * Real.log x

def g (a b : ℝ) (x : ℝ) : ℝ := f a x + 1/2 * x^2 - b * x

def tangent_perpendicular (a : ℝ) : Prop :=
  (deriv (f a) 1) * (-1/2) = -1

def has_decreasing_interval (a b : ℝ) : Prop :=
  ∃ x y, x < y ∧ ∀ z ∈ Set.Ioo x y, (deriv (g a b) z) < 0

def extreme_points (a b : ℝ) (x₁ x₂ : ℝ) : Prop :=
  x₁ < x₂ ∧ (deriv (g a b) x₁) = 0 ∧ (deriv (g a b) x₂) = 0

theorem f_g_properties (a b : ℝ) (x₁ x₂ : ℝ) 
  (h1 : tangent_perpendicular a)
  (h2 : extreme_points a b x₁ x₂)
  (h3 : b ≥ 7/2) :
  a = 1 ∧ 
  (has_decreasing_interval a b → b > 3) ∧
  (g a b x₁ - g a b x₂ ≥ 15/8 - 2 * Real.log 2) :=
sorry

end f_g_properties_l684_68476


namespace factorization_1_factorization_2_l684_68427

-- Define variables
variable (a b m : ℝ)

-- Theorem for the first factorization
theorem factorization_1 : a^2 * (a - b) - 4 * b^2 * (a - b) = (a - b) * (a - 2*b) * (a + 2*b) := by
  sorry

-- Theorem for the second factorization
theorem factorization_2 : m^2 - 6*m + 9 = (m - 3)^2 := by
  sorry

end factorization_1_factorization_2_l684_68427


namespace inequality_proof_l684_68420

theorem inequality_proof (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_abc : a * b * c = 1) : 
  (1 / (a^3 * (b + c))) + (1 / (b^3 * (c + a))) + (1 / (c^3 * (a + b))) ≥ 3/2 := by
  sorry

end inequality_proof_l684_68420


namespace treasure_hunt_probability_l684_68491

def num_islands : ℕ := 7
def num_treasure_islands : ℕ := 4

def prob_treasure : ℚ := 1/3
def prob_traps : ℚ := 1/6
def prob_neither : ℚ := 1/2

theorem treasure_hunt_probability :
  (Nat.choose num_islands num_treasure_islands : ℚ) *
  prob_treasure ^ num_treasure_islands *
  prob_neither ^ (num_islands - num_treasure_islands) =
  35/648 := by
  sorry

end treasure_hunt_probability_l684_68491


namespace max_value_trig_expression_l684_68486

theorem max_value_trig_expression (x y z : ℝ) :
  (Real.sin (2 * x) + Real.sin y + Real.sin (3 * z)) *
  (Real.cos (2 * x) + Real.cos y + Real.cos (3 * z)) ≤ 9 / 2 := by
sorry

end max_value_trig_expression_l684_68486


namespace sample_data_properties_l684_68439

theorem sample_data_properties (x : Fin 6 → ℝ) (h : ∀ i j : Fin 6, i ≤ j → x i ≤ x j) :
  (let median1 := (x 2 + x 3) / 2
   let median2 := (x 2 + x 3) / 2
   median1 = median2) ∧
  (x 4 - x 1 ≤ x 5 - x 0) :=
by sorry

end sample_data_properties_l684_68439


namespace ten_workers_needed_l684_68446

/-- Represents the project details and worker productivity --/
structure Project where
  total_days : ℕ
  days_passed : ℕ
  work_completed : ℚ
  current_workers : ℕ

/-- Calculates the minimum number of workers needed to complete the project on schedule --/
def min_workers_needed (p : Project) : ℕ :=
  p.current_workers

/-- Theorem stating that for the given project conditions, 10 workers are needed --/
theorem ten_workers_needed (p : Project)
  (h1 : p.total_days = 40)
  (h2 : p.days_passed = 10)
  (h3 : p.work_completed = 1/4)
  (h4 : p.current_workers = 10) :
  min_workers_needed p = 10 := by
  sorry

#eval min_workers_needed {
  total_days := 40,
  days_passed := 10,
  work_completed := 1/4,
  current_workers := 10
}

end ten_workers_needed_l684_68446


namespace sum_of_two_numbers_l684_68425

theorem sum_of_two_numbers (x y : ℤ) : 
  y = 2 * x - 43 →  -- First number is 43 less than twice the second
  max x y = 31 →    -- Larger number is 31
  x + y = 68 :=     -- Sum of the two numbers is 68
by sorry

end sum_of_two_numbers_l684_68425


namespace weeks_to_save_l684_68405

def console_cost : ℕ := 282
def game_cost : ℕ := 75
def initial_savings : ℕ := 42
def weekly_allowance : ℕ := 24

theorem weeks_to_save : ℕ := by
  -- The minimum number of whole weeks required to save enough money
  -- for both the console and the game is 14.
  sorry

end weeks_to_save_l684_68405


namespace composition_of_even_is_even_l684_68469

-- Define a type for real-valued functions
def RealFunction := ℝ → ℝ

-- Define what it means for a function to be even
def IsEven (f : RealFunction) : Prop :=
  ∀ x : ℝ, f (-x) = f x

-- State the theorem
theorem composition_of_even_is_even (g : RealFunction) (h_even : IsEven g) :
  IsEven (g ∘ g) := by
  sorry

end composition_of_even_is_even_l684_68469


namespace boat_speed_l684_68422

/-- The speed of a boat in still water, given its downstream and upstream speeds -/
theorem boat_speed (downstream_speed upstream_speed : ℝ) 
  (h1 : downstream_speed = 10)
  (h2 : upstream_speed = 4) :
  (downstream_speed + upstream_speed) / 2 = 7 := by
  sorry

end boat_speed_l684_68422


namespace pure_imaginary_implies_a_eq_two_second_or_fourth_quadrant_implies_a_range_l684_68438

-- Define the complex number z
def z (a : ℝ) : ℂ := (a^2 - a - 2 : ℝ) + (a^2 - 3*a - 4 : ℝ)*Complex.I

-- Part 1: z is a pure imaginary number implies a = 2
theorem pure_imaginary_implies_a_eq_two :
  ∀ a : ℝ, (z a).re = 0 → (z a).im ≠ 0 → a = 2 := by sorry

-- Part 2: z in second or fourth quadrant implies 2 < a < 4
theorem second_or_fourth_quadrant_implies_a_range :
  ∀ a : ℝ, (z a).re * (z a).im < 0 → 2 < a ∧ a < 4 := by sorry

end pure_imaginary_implies_a_eq_two_second_or_fourth_quadrant_implies_a_range_l684_68438


namespace fraction_simplification_l684_68437

theorem fraction_simplification (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -3) :
  (3*x^2 + x) / ((x - 1) * (x + 3)) + (5 - x) / ((x - 1) * (x + 3)) =
  (3*x^2 + 4) / ((x - 1) * (x + 3)) :=
by sorry

end fraction_simplification_l684_68437


namespace exam_average_l684_68401

theorem exam_average (total_boys : ℕ) (passed_boys : ℕ) (avg_passed : ℕ) (avg_failed : ℕ)
  (h1 : total_boys = 120)
  (h2 : passed_boys = 100)
  (h3 : avg_passed = 39)
  (h4 : avg_failed = 15) :
  (avg_passed * passed_boys + avg_failed * (total_boys - passed_boys)) / total_boys = 35 := by
sorry

end exam_average_l684_68401


namespace subset_ratio_for_ten_elements_l684_68487

theorem subset_ratio_for_ten_elements : 
  let n : ℕ := 10
  let k : ℕ := 3
  let total_subsets : ℕ := 2^n
  let three_element_subsets : ℕ := n.choose k
  (three_element_subsets : ℚ) / total_subsets = 15 / 128 := by
  sorry

end subset_ratio_for_ten_elements_l684_68487


namespace right_triangle_sin_value_l684_68482

theorem right_triangle_sin_value (A B C : ℝ) (h1 : 0 < A ∧ A < π/2) (h2 : 0 < B ∧ B < π/2) :
  B = π/2 → 2 * Real.sin A = 3 * Real.cos A → Real.sin A = 3 * Real.sqrt 13 / 13 := by
  sorry

end right_triangle_sin_value_l684_68482


namespace distance_to_line_mn_l684_68495

/-- The distance from the origin to the line MN, where M is on the hyperbola 2x² - y² = 1
    and N is on the ellipse 4x² + y² = 1, with OM perpendicular to ON. -/
theorem distance_to_line_mn (M N : ℝ × ℝ) : 
  (2 * M.1^2 - M.2^2 = 1) →  -- M is on the hyperbola
  (4 * N.1^2 + N.2^2 = 1) →  -- N is on the ellipse
  (M.1 * N.1 + M.2 * N.2 = 0) →  -- OM ⟂ ON
  let d := Real.sqrt 3 / 3
  ∃ (t : ℝ), t * M.1 + (1 - t) * N.1 = d * (N.2 - M.2) ∧
             t * M.2 + (1 - t) * N.2 = d * (M.1 - N.1) :=
by sorry

end distance_to_line_mn_l684_68495


namespace sqrt_nine_factorial_over_ninety_l684_68423

theorem sqrt_nine_factorial_over_ninety : 
  Real.sqrt (Nat.factorial 9 / 90) = 24 * Real.sqrt 7 := by
  sorry

end sqrt_nine_factorial_over_ninety_l684_68423


namespace alpha_squared_greater_than_beta_squared_l684_68496

theorem alpha_squared_greater_than_beta_squared
  (α β : ℝ)
  (h1 : α ∈ Set.Icc (-π/2) (π/2))
  (h2 : β ∈ Set.Icc (-π/2) (π/2))
  (h3 : α * Real.sin α - β * Real.sin β > 0) :
  α^2 > β^2 := by
  sorry

end alpha_squared_greater_than_beta_squared_l684_68496


namespace correct_group_capacity_l684_68451

/-- The capacity of each group in a systematic sampling -/
def group_capacity (total_students : ℕ) (sample_size : ℕ) : ℕ :=
  (total_students - (total_students % sample_size)) / sample_size

/-- Theorem stating the correct group capacity for the given problem -/
theorem correct_group_capacity :
  group_capacity 5008 200 = 25 := by
  sorry

end correct_group_capacity_l684_68451


namespace farm_animals_l684_68480

/-- The number of chickens on the farm -/
def num_chickens : ℕ := 49

/-- The number of ducks on the farm -/
def num_ducks : ℕ := 37

/-- The number of rabbits on the farm -/
def num_rabbits : ℕ := 21

theorem farm_animals :
  (num_ducks + num_rabbits = num_chickens + 9) →
  num_rabbits = 21 := by
  sorry

end farm_animals_l684_68480


namespace straight_line_angle_l684_68400

/-- 
Given a straight line segment PQ with angle measurements of 90°, x°, and 20° along it,
prove that x = 70°.
-/
theorem straight_line_angle (x : ℝ) : 
  (90 : ℝ) + x + 20 = 180 → x = 70 := by
  sorry

end straight_line_angle_l684_68400


namespace subset_properties_l684_68436

variable {α : Type*}
variable (A B : Set α)

theorem subset_properties (hAB : A ⊆ B) (hA : A.Nonempty) (hB : B.Nonempty) :
  (∀ x, x ∈ A → x ∈ B) ∧
  (∃ x, x ∈ B ∧ x ∉ A) ∧
  (∀ x, x ∉ B → x ∉ A) :=
by sorry

end subset_properties_l684_68436


namespace investment_problem_l684_68403

/-- Proves that the total investment amount is $5,400 given the problem conditions -/
theorem investment_problem (total : ℝ) (amount_at_8_percent : ℝ) (amount_at_10_percent : ℝ)
  (h1 : amount_at_8_percent = 3000)
  (h2 : total = amount_at_8_percent + amount_at_10_percent)
  (h3 : amount_at_8_percent * 0.08 = amount_at_10_percent * 0.10) :
  total = 5400 := by
  sorry

end investment_problem_l684_68403


namespace max_followers_after_three_weeks_l684_68464

def susyInitialFollowers : ℕ := 100
def sarahInitialFollowers : ℕ := 50

def susyWeek1Gain : ℕ := 40
def sarahWeek1Gain : ℕ := 90

def susyTotalFollowers : ℕ := 
  susyInitialFollowers + susyWeek1Gain + (susyWeek1Gain / 2) + (susyWeek1Gain / 4)

def sarahTotalFollowers : ℕ := 
  sarahInitialFollowers + sarahWeek1Gain + (sarahWeek1Gain / 3) + (sarahWeek1Gain / 9)

theorem max_followers_after_three_weeks :
  max susyTotalFollowers sarahTotalFollowers = 180 := by
  sorry

end max_followers_after_three_weeks_l684_68464


namespace fraction_expression_l684_68455

theorem fraction_expression : 
  (3/7 + 5/8) / (5/12 + 2/9) = 531/322 := by
  sorry

end fraction_expression_l684_68455


namespace true_compound_props_l684_68441

def p₁ : Prop := True
def p₂ : Prop := False
def p₃ : Prop := False
def p₄ : Prop := True

def compound_prop_1 : Prop := p₁ ∧ p₄
def compound_prop_2 : Prop := p₁ ∧ p₂
def compound_prop_3 : Prop := ¬p₂ ∨ p₃
def compound_prop_4 : Prop := ¬p₃ ∨ ¬p₄

theorem true_compound_props :
  {compound_prop_1, compound_prop_3, compound_prop_4} = 
  {p : Prop | p = compound_prop_1 ∨ p = compound_prop_2 ∨ p = compound_prop_3 ∨ p = compound_prop_4 ∧ p} :=
by sorry

end true_compound_props_l684_68441


namespace exists_one_one_appended_one_l684_68454

def is_valid_number (n : ℕ) (num : List ℕ) : Prop :=
  num.length = n ∧ ∀ d ∈ num, d = 1 ∨ d = 2 ∨ d = 3

def differs_in_all_positions (n : ℕ) (num1 num2 : List ℕ) : Prop :=
  is_valid_number n num1 ∧ is_valid_number n num2 ∧
  ∀ i, i < n → num1.get ⟨i, by sorry⟩ ≠ num2.get ⟨i, by sorry⟩

def appended_digit (n : ℕ) (num : List ℕ) (d : ℕ) : Prop :=
  is_valid_number n num ∧ (d = 1 ∨ d = 2 ∨ d = 3)

def valid_appending (n : ℕ) (append : List ℕ → ℕ) : Prop :=
  ∀ num1 num2 : List ℕ, differs_in_all_positions n num1 num2 →
    append num1 ≠ append num2

theorem exists_one_one_appended_one (n : ℕ) :
  ∃ (append : List ℕ → ℕ),
    valid_appending n append →
    ∃ (num : List ℕ),
      is_valid_number n num ∧
      (num.count 1 = 1) ∧
      (append num = 1) := by sorry

end exists_one_one_appended_one_l684_68454


namespace bus_seating_capacity_l684_68477

/-- Represents the seating capacity of a bus with specific seat arrangements. -/
structure BusSeating where
  left_seats : Nat
  right_seats : Nat
  people_per_seat : Nat
  back_seat_capacity : Nat

/-- Calculates the total number of people who can sit in the bus. -/
def total_seating_capacity (bus : BusSeating) : Nat :=
  (bus.left_seats + bus.right_seats) * bus.people_per_seat + bus.back_seat_capacity

/-- Theorem stating the total seating capacity of the bus with given conditions. -/
theorem bus_seating_capacity : 
  ∀ (bus : BusSeating), 
    bus.left_seats = 15 → 
    bus.right_seats = bus.left_seats - 3 →
    bus.people_per_seat = 3 →
    bus.back_seat_capacity = 11 →
    total_seating_capacity bus = 92 :=
by
  sorry

end bus_seating_capacity_l684_68477


namespace optimal_selection_l684_68428

/-- Represents a 5x5 matrix of integers -/
def Matrix5x5 : Type := Fin 5 → Fin 5 → ℤ

/-- The given matrix -/
def givenMatrix : Matrix5x5 :=
  λ i j => match i, j with
  | ⟨0, _⟩, ⟨0, _⟩ => 11 | ⟨0, _⟩, ⟨1, _⟩ => 17 | ⟨0, _⟩, ⟨2, _⟩ => 25 | ⟨0, _⟩, ⟨3, _⟩ => 19 | ⟨0, _⟩, ⟨4, _⟩ => 16
  | ⟨1, _⟩, ⟨0, _⟩ => 24 | ⟨1, _⟩, ⟨1, _⟩ => 10 | ⟨1, _⟩, ⟨2, _⟩ => 13 | ⟨1, _⟩, ⟨3, _⟩ => 15 | ⟨1, _⟩, ⟨4, _⟩ => 3
  | ⟨2, _⟩, ⟨0, _⟩ => 12 | ⟨2, _⟩, ⟨1, _⟩ => 5  | ⟨2, _⟩, ⟨2, _⟩ => 14 | ⟨2, _⟩, ⟨3, _⟩ => 2  | ⟨2, _⟩, ⟨4, _⟩ => 18
  | ⟨3, _⟩, ⟨0, _⟩ => 23 | ⟨3, _⟩, ⟨1, _⟩ => 4  | ⟨3, _⟩, ⟨2, _⟩ => 1  | ⟨3, _⟩, ⟨3, _⟩ => 8  | ⟨3, _⟩, ⟨4, _⟩ => 22
  | ⟨4, _⟩, ⟨0, _⟩ => 6  | ⟨4, _⟩, ⟨1, _⟩ => 20 | ⟨4, _⟩, ⟨2, _⟩ => 7  | ⟨4, _⟩, ⟨3, _⟩ => 21 | ⟨4, _⟩, ⟨4, _⟩ => 9
  | _, _ => 0

/-- A selection of 5 elements from the matrix -/
def Selection : Type := Fin 5 → (Fin 5 × Fin 5)

/-- Check if a selection is valid (no two elements in same row or column) -/
def isValidSelection (s : Selection) : Prop :=
  ∀ i j, i ≠ j → (s i).1 ≠ (s j).1 ∧ (s i).2 ≠ (s j).2

/-- The claimed optimal selection -/
def claimedOptimalSelection : Selection :=
  λ i => match i with
  | ⟨0, _⟩ => (⟨0, by norm_num⟩, ⟨2, by norm_num⟩)  -- 25
  | ⟨1, _⟩ => (⟨4, by norm_num⟩, ⟨1, by norm_num⟩)  -- 20
  | ⟨2, _⟩ => (⟨3, by norm_num⟩, ⟨0, by norm_num⟩)  -- 23
  | ⟨3, _⟩ => (⟨2, by norm_num⟩, ⟨4, by norm_num⟩)  -- 18
  | ⟨4, _⟩ => (⟨1, by norm_num⟩, ⟨3, by norm_num⟩)  -- 15

/-- The theorem to prove -/
theorem optimal_selection :
  isValidSelection claimedOptimalSelection ∧
  (∀ s : Selection, isValidSelection s →
    (∃ i, givenMatrix (s i).1 (s i).2 ≤ givenMatrix (claimedOptimalSelection 4).1 (claimedOptimalSelection 4).2)) :=
by sorry

end optimal_selection_l684_68428


namespace work_completion_time_l684_68406

theorem work_completion_time (x : ℝ) : 
  (x > 0) →  -- Ensure x is positive
  (1/x + 1/15 = 1/6) →  -- Combined work rate equals 1/6
  (x = 10) := by
sorry

end work_completion_time_l684_68406


namespace sum_lower_bound_l684_68412

noncomputable def f (x : ℝ) := Real.log x + x^2 + x

theorem sum_lower_bound (x₁ x₂ : ℝ) (h₁ : x₁ > 0) (h₂ : x₂ > 0)
  (h : f x₁ + f x₂ + x₁ * x₂ = 0) :
  x₁ + x₂ ≥ (Real.sqrt 5 - 1) / 2 := by
sorry

end sum_lower_bound_l684_68412


namespace seven_digit_divisibility_l684_68411

theorem seven_digit_divisibility (A B C : Nat) : 
  A < 10 → B < 10 → C < 10 →
  (74 * 100000 + A * 10000 + 52 * 100 + B * 10 + 1) % 3 = 0 →
  (326 * 10000 + A * 1000 + B * 100 + 4 * 10 + C) % 3 = 0 →
  C = 1 := by
sorry

end seven_digit_divisibility_l684_68411


namespace ellipse_properties_l684_68470

-- Define the ellipse
def Ellipse (a b : ℝ) := {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

-- Define the foci
def F₁ : ℝ × ℝ := (-4, 0)
def F₂ : ℝ × ℝ := (4, 0)

-- Define eccentricity
def e : ℝ := 0.8

-- Define dot product of vectors
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define vector from a point to another
def vector_to (p q : ℝ × ℝ) : ℝ × ℝ := (q.1 - p.1, q.2 - p.2)

theorem ellipse_properties :
  ∃ (a b : ℝ),
    -- Standard equation of the ellipse
    (a = 5 ∧ b = 3) ∧
    -- Existence of point P
    ∃ (P : ℝ × ℝ),
      P ∈ Ellipse a b ∧
      dot_product (vector_to F₁ P) (vector_to F₂ P) = 0 ∧
      -- Coordinates of point P
      ((P.1 = 5 * Real.sqrt 7 / 4 ∧ P.2 = 9 / 4) ∨
       (P.1 = -5 * Real.sqrt 7 / 4 ∧ P.2 = 9 / 4) ∨
       (P.1 = 5 * Real.sqrt 7 / 4 ∧ P.2 = -9 / 4) ∨
       (P.1 = -5 * Real.sqrt 7 / 4 ∧ P.2 = -9 / 4)) :=
by
  sorry

end ellipse_properties_l684_68470


namespace diego_martha_can_ratio_l684_68471

theorem diego_martha_can_ratio :
  let martha_cans : ℕ := 90
  let total_needed : ℕ := 150
  let more_needed : ℕ := 5
  let total_collected : ℕ := total_needed - more_needed
  let diego_cans : ℕ := total_collected - martha_cans
  (diego_cans : ℚ) / martha_cans = 11 / 18 := by
  sorry

end diego_martha_can_ratio_l684_68471


namespace f_of_x_minus_one_l684_68485

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x

-- State the theorem
theorem f_of_x_minus_one (x : ℝ) : f (x - 1) = x^2 - 4*x + 3 := by
  sorry

end f_of_x_minus_one_l684_68485


namespace ice_cream_scoops_l684_68404

def ice_cream_problem (single_cone waffle_bowl banana_split double_cone : ℕ) : Prop :=
  single_cone = 1 ∧ 
  banana_split = 3 * single_cone ∧ 
  waffle_bowl = banana_split + 1 ∧
  double_cone = 2 ∧
  single_cone + double_cone + banana_split + waffle_bowl = 10

theorem ice_cream_scoops : 
  ∃ (single_cone waffle_bowl banana_split double_cone : ℕ),
    ice_cream_problem single_cone waffle_bowl banana_split double_cone :=
by
  sorry

end ice_cream_scoops_l684_68404


namespace fifth_term_of_sequence_l684_68421

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) - a n = d

theorem fifth_term_of_sequence (x y : ℚ) (a : ℕ → ℚ) 
  (h_arith : arithmetic_sequence a)
  (h_first : a 0 = x + 2*y)
  (h_second : a 1 = x - 2*y)
  (h_third : a 2 = x + 2*y^2)
  (h_fourth : a 3 = x / (2*y))
  (h_y_nonzero : y ≠ 0) :
  a 4 = -x/6 - 12 := by
  sorry

end fifth_term_of_sequence_l684_68421


namespace power_of_four_equality_l684_68432

theorem power_of_four_equality (m n : ℕ+) (x y : ℝ) 
  (hx : 2^(m : ℕ) = x) (hy : 2^(2*n : ℕ) = y) : 
  4^((m : ℕ) + 2*(n : ℕ)) = x^2 * y^2 := by
  sorry

end power_of_four_equality_l684_68432


namespace pages_revised_once_is_30_l684_68417

/-- Represents the typing service problem --/
structure TypingService where
  totalPages : ℕ
  pagesRevisedTwice : ℕ
  totalCost : ℕ
  firstTypingRate : ℕ
  revisionRate : ℕ

/-- Calculates the number of pages revised once --/
def pagesRevisedOnce (ts : TypingService) : ℕ :=
  ((ts.totalCost - ts.firstTypingRate * ts.totalPages - 
    ts.revisionRate * ts.pagesRevisedTwice * 2) / ts.revisionRate)

/-- Theorem stating the number of pages revised once --/
theorem pages_revised_once_is_30 (ts : TypingService) 
  (h1 : ts.totalPages = 100)
  (h2 : ts.pagesRevisedTwice = 20)
  (h3 : ts.totalCost = 1350)
  (h4 : ts.firstTypingRate = 10)
  (h5 : ts.revisionRate = 5) :
  pagesRevisedOnce ts = 30 := by
  sorry

#eval pagesRevisedOnce {
  totalPages := 100,
  pagesRevisedTwice := 20,
  totalCost := 1350,
  firstTypingRate := 10,
  revisionRate := 5
}

end pages_revised_once_is_30_l684_68417


namespace imaginary_part_of_z_l684_68440

theorem imaginary_part_of_z : Complex.im (((1 : ℂ) - Complex.I) / (2 * Complex.I)) = -1/2 := by
  sorry

end imaginary_part_of_z_l684_68440


namespace function_equation_solution_l684_68493

theorem function_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, (f x + x * y) * f (x - 3 * y) + (f y + x * y) * f (3 * x - y) = (f (x + y))^2) →
  (∀ x : ℝ, f x = 0 ∨ f x = x^2) :=
by sorry

end function_equation_solution_l684_68493


namespace private_pilot_course_cost_l684_68474

/-- The cost of a private pilot course -/
theorem private_pilot_course_cost :
  ∀ (flight_cost ground_cost total_cost : ℕ),
    flight_cost = 950 →
    ground_cost = 325 →
    flight_cost = ground_cost + 625 →
    total_cost = flight_cost + ground_cost →
    total_cost = 1275 := by
  sorry

end private_pilot_course_cost_l684_68474


namespace smallest_positive_period_of_cosine_l684_68418

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.cos (ω * x)

theorem smallest_positive_period_of_cosine 
  (ω : ℝ) 
  (h_ω_pos : ω > 0) 
  (P : ℝ × ℝ) 
  (h_center_symmetry : ∀ x, f ω (2 * P.1 - x) = f ω x) 
  (h_min_distance : ∀ y, abs (P.2 - y) ≥ π) :
  ∃ T > 0, (∀ x, f ω (x + T) = f ω x) ∧ 
  (∀ S, S > 0 → (∀ x, f ω (x + S) = f ω x) → S ≥ T) ∧ 
  T = 2 * π := by
  sorry

end smallest_positive_period_of_cosine_l684_68418


namespace expression_evaluation_l684_68461

theorem expression_evaluation :
  let x : ℚ := -2
  let y : ℚ := 1/2
  (4 * x^2 * y - (6 * x * y - 3 * (4 * x - 2) - x^2 * y) + 1) = -13 := by
sorry

end expression_evaluation_l684_68461


namespace f_neg_five_eq_one_l684_68453

/-- A polynomial function of degree 5 with a constant term of 5 -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^5 + b * x^3 + c * x + 5

/-- Theorem stating that if f(5) = 9, then f(-5) = 1 -/
theorem f_neg_five_eq_one (a b c : ℝ) (h : f a b c 5 = 9) : f a b c (-5) = 1 := by
  sorry

end f_neg_five_eq_one_l684_68453


namespace equal_days_count_l684_68449

/-- Represents the days of the week -/
inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

/-- Returns the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Counts the number of Tuesdays and Thursdays in a 30-day month starting on the given day -/
def countTuesdaysAndThursdays (startDay : DayOfWeek) : Nat × Nat :=
  let rec count (currentDay : DayOfWeek) (daysLeft : Nat) (tuesdays : Nat) (thursdays : Nat) : Nat × Nat :=
    if daysLeft = 0 then
      (tuesdays, thursdays)
    else
      match currentDay with
      | DayOfWeek.Tuesday => count (nextDay currentDay) (daysLeft - 1) (tuesdays + 1) thursdays
      | DayOfWeek.Thursday => count (nextDay currentDay) (daysLeft - 1) tuesdays (thursdays + 1)
      | _ => count (nextDay currentDay) (daysLeft - 1) tuesdays thursdays
  count startDay 30 0 0

/-- Checks if the number of Tuesdays and Thursdays are equal for a given starting day -/
def hasEqualTuesdaysAndThursdays (startDay : DayOfWeek) : Bool :=
  let (tuesdays, thursdays) := countTuesdaysAndThursdays startDay
  tuesdays = thursdays

/-- Counts the number of days that result in equal Tuesdays and Thursdays -/
def countEqualDays : Nat :=
  let days := [DayOfWeek.Sunday, DayOfWeek.Monday, DayOfWeek.Tuesday, DayOfWeek.Wednesday,
               DayOfWeek.Thursday, DayOfWeek.Friday, DayOfWeek.Saturday]
  days.filter hasEqualTuesdaysAndThursdays |>.length

theorem equal_days_count :
  countEqualDays = 4 :=
sorry

end equal_days_count_l684_68449


namespace f_monotonic_implies_a_range_l684_68466

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := -x^2 + 2*(a-1)*x + 2

-- Define monotonicity on an interval
def monotonic_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  (∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y) ∨
  (∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x)

-- Theorem statement
theorem f_monotonic_implies_a_range :
  ∀ a : ℝ, monotonic_on (f a) 2 4 → a ≤ 3 ∨ a ≥ 5 := by
  sorry

end f_monotonic_implies_a_range_l684_68466


namespace profit_and_max_profit_l684_68448

/-- Represents the average daily profit as a function of price reduction --/
def averageDailyProfit (x : ℝ) : ℝ := -2 * x^2 + 60 * x + 800

/-- The price reduction that results in $1200 average daily profit --/
def priceReductionFor1200Profit : ℝ := 20

/-- The price reduction that maximizes average daily profit --/
def priceReductionForMaxProfit : ℝ := 15

/-- The maximum average daily profit --/
def maxAverageDailyProfit : ℝ := 1250

theorem profit_and_max_profit :
  (averageDailyProfit priceReductionFor1200Profit = 1200) ∧
  (∀ x : ℝ, averageDailyProfit x ≤ maxAverageDailyProfit) ∧
  (averageDailyProfit priceReductionForMaxProfit = maxAverageDailyProfit) := by
  sorry


end profit_and_max_profit_l684_68448

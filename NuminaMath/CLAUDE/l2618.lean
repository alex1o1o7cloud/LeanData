import Mathlib

namespace intersection_of_A_and_B_l2618_261885

def A : Set ℝ := {x | -2 < x ∧ x < 4}
def B : Set ℝ := {2, 3, 4, 5}

theorem intersection_of_A_and_B : A ∩ B = {2, 3} := by
  sorry

end intersection_of_A_and_B_l2618_261885


namespace triangle_exterior_angle_theorem_l2618_261811

/-- 
Given a triangle where one side is extended:
- ext_angle is the exterior angle
- int_angle1 is one of the non-adjacent interior angles
- int_angle2 is the other non-adjacent interior angle
-/
theorem triangle_exterior_angle_theorem 
  (ext_angle int_angle1 int_angle2 : ℝ) : 
  ext_angle = 154 ∧ int_angle1 = 58 → int_angle2 = 96 := by
sorry

end triangle_exterior_angle_theorem_l2618_261811


namespace distance_between_x_intercepts_specific_case_l2618_261809

/-- Two lines in a 2D plane -/
structure TwoLines where
  intersection : ℝ × ℝ
  slope1 : ℝ
  slope2 : ℝ

/-- Calculate the distance between x-intercepts of two lines -/
def distance_between_x_intercepts (lines : TwoLines) : ℝ :=
  sorry

/-- The main theorem -/
theorem distance_between_x_intercepts_specific_case :
  let lines : TwoLines := {
    intersection := (12, 20),
    slope1 := 7/2,
    slope2 := -3/2
  }
  distance_between_x_intercepts lines = 800/21 := by sorry

end distance_between_x_intercepts_specific_case_l2618_261809


namespace xy_sum_problem_l2618_261862

theorem xy_sum_problem (x y : ℕ) 
  (pos_x : x > 0) (pos_y : y > 0)
  (bound_x : x < 30) (bound_y : y < 30)
  (eq_condition : x + y + x * y = 119) :
  x + y = 24 ∨ x + y = 20 := by
sorry

end xy_sum_problem_l2618_261862


namespace geometric_arithmetic_ratio_l2618_261888

/-- Given a geometric sequence with positive terms and common ratio q,
    if 3a₁, (1/2)a₃, 2a₂ form an arithmetic sequence, then q = 3 -/
theorem geometric_arithmetic_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n > 0) →  -- all terms are positive
  (∀ n, a (n + 1) = q * a n) →  -- geometric sequence with ratio q
  q > 0 →  -- q is positive
  2 * ((1/2) * a 3) = 3 * a 1 + 2 * a 2 →  -- arithmetic sequence condition
  q = 3 := by
  sorry

end geometric_arithmetic_ratio_l2618_261888


namespace floor_ceiling_product_l2618_261835

theorem floor_ceiling_product : ⌊(3.999 : ℝ)⌋ * ⌈(0.002 : ℝ)⌉ = 3 := by sorry

end floor_ceiling_product_l2618_261835


namespace special_functions_at_zero_l2618_261842

/-- Two non-constant functions satisfying specific addition formulas -/
class SpecialFunctions (f g : ℝ → ℝ) : Prop where
  add_f : ∀ x y, f (x + y) = f x * g y + g x * f y
  add_g : ∀ x y, g (x + y) = g x * g y - f x * f y
  non_constant_f : ∃ x y, f x ≠ f y
  non_constant_g : ∃ x y, g x ≠ g y

/-- The values of f(0) and g(0) for special functions f and g -/
theorem special_functions_at_zero {f g : ℝ → ℝ} [SpecialFunctions f g] :
  f 0 = 0 ∧ g 0 = 1 := by
  sorry

end special_functions_at_zero_l2618_261842


namespace gary_remaining_money_l2618_261867

/-- Calculates the remaining money after a purchase -/
def remaining_money (initial : ℕ) (spent : ℕ) : ℕ :=
  initial - spent

/-- Theorem: Gary's remaining money -/
theorem gary_remaining_money :
  remaining_money 73 55 = 18 := by
  sorry

end gary_remaining_money_l2618_261867


namespace program_attendance_l2618_261832

/-- The total number of people present at the program -/
def total_people (parents pupils teachers staff family_members : ℕ) : ℕ :=
  parents + pupils + teachers + staff + family_members

/-- The number of family members accompanying the pupils -/
def accompanying_family_members (pupils : ℕ) : ℕ :=
  (pupils / 6) * 2

theorem program_attendance : 
  let parents : ℕ := 83
  let pupils : ℕ := 956
  let teachers : ℕ := 154
  let staff : ℕ := 27
  let family_members : ℕ := accompanying_family_members pupils
  total_people parents pupils teachers staff family_members = 1379 := by
sorry

end program_attendance_l2618_261832


namespace min_pieces_for_rearrangement_l2618_261886

/-- Represents a shape made of small squares -/
structure Shape :=
  (squares : Nat)

/-- Represents the goal configuration -/
structure GoalSquare :=
  (side : Nat)

/-- Represents a cutting of the shape into pieces -/
structure Cutting :=
  (num_pieces : Nat)

/-- Predicate to check if a cutting is valid for rearrangement -/
def is_valid_cutting (s : Shape) (g : GoalSquare) (c : Cutting) : Prop :=
  c.num_pieces ≥ 1 ∧ c.num_pieces ≤ s.squares

/-- Predicate to check if a cutting allows rearrangement into the goal square -/
def allows_rearrangement (s : Shape) (g : GoalSquare) (c : Cutting) : Prop :=
  is_valid_cutting s g c ∧ s.squares = g.side * g.side

/-- The main theorem stating the minimum number of pieces required -/
theorem min_pieces_for_rearrangement (s : Shape) (g : GoalSquare) :
  s.squares = 9 → g.side = 3 →
  ∃ (c : Cutting), 
    c.num_pieces = 3 ∧ 
    allows_rearrangement s g c ∧
    ∀ (c' : Cutting), allows_rearrangement s g c' → c'.num_pieces ≥ 3 :=
sorry

end min_pieces_for_rearrangement_l2618_261886


namespace composite_has_at_least_three_factors_l2618_261801

/-- A natural number is composite if it's greater than 1 and not prime -/
def IsComposite (n : ℕ) : Prop :=
  n > 1 ∧ ¬(Nat.Prime n)

/-- The number of factors of a natural number -/
def numFactors (n : ℕ) : ℕ :=
  (Nat.divisors n).card

/-- Theorem: Any composite number has at least 3 factors -/
theorem composite_has_at_least_three_factors (n : ℕ) (h : IsComposite n) :
  numFactors n ≥ 3 :=
sorry

end composite_has_at_least_three_factors_l2618_261801


namespace count_satisfying_integers_l2618_261859

-- Define the function f(n)
def f (n : ℤ) : ℤ := ⌈(99 * n : ℚ) / 100⌉ - ⌊(100 * n : ℚ) / 101⌋

-- State the theorem
theorem count_satisfying_integers :
  (∃ (S : Finset ℤ), (∀ n ∈ S, f n = 1) ∧ S.card = 10100 ∧
    (∀ n : ℤ, f n = 1 → n ∈ S)) :=
sorry

end count_satisfying_integers_l2618_261859


namespace girl_multiplication_problem_l2618_261833

theorem girl_multiplication_problem (incorrect_multiplier : ℕ) (difference : ℕ) (base_number : ℕ) :
  incorrect_multiplier = 34 →
  difference = 1242 →
  base_number = 138 →
  ∃ (correct_multiplier : ℕ), 
    base_number * correct_multiplier = base_number * incorrect_multiplier + difference ∧
    correct_multiplier = 43 :=
by
  sorry

end girl_multiplication_problem_l2618_261833


namespace arithmetic_sequence_first_term_l2618_261806

theorem arithmetic_sequence_first_term
  (a d : ℝ)
  (sum_100 : (100 : ℝ) / 2 * (2 * a + 99 * d) = 1800)
  (sum_51_to_150 : (100 : ℝ) / 2 * (2 * a + 199 * d) = 6300) :
  a = -26.55 :=
by sorry

end arithmetic_sequence_first_term_l2618_261806


namespace divisibility_of_245245_by_35_l2618_261846

theorem divisibility_of_245245_by_35 : 35 ∣ 245245 := by
  sorry

end divisibility_of_245245_by_35_l2618_261846


namespace units_digit_factorial_sum_800_l2618_261873

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def units_digit (n : ℕ) : ℕ := n % 10

def factorial_sum (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem units_digit_factorial_sum_800 :
  units_digit (factorial_sum 800) = 3 := by
sorry

end units_digit_factorial_sum_800_l2618_261873


namespace complement_of_union_M_N_l2618_261840

-- Define the universal set U
def U : Finset Nat := {1, 2, 3, 4, 5, 6}

-- Define set M
def M : Finset Nat := {2, 3, 4}

-- Define set N
def N : Finset Nat := {4, 5}

-- Theorem statement
theorem complement_of_union_M_N :
  (U \ (M ∪ N)) = {1, 6} := by
  sorry

end complement_of_union_M_N_l2618_261840


namespace rhombus_area_in_square_l2618_261819

/-- The area of a rhombus formed by two equilateral triangles in a square -/
theorem rhombus_area_in_square (square_side : ℝ) (h_square_side : square_side = 4) :
  let triangle_height := square_side * (Real.sqrt 3) / 2
  let vertical_overlap := 2 * triangle_height - square_side
  let rhombus_area := (vertical_overlap * square_side) / 2
  rhombus_area = 8 * Real.sqrt 3 - 8 :=
by sorry

end rhombus_area_in_square_l2618_261819


namespace product_of_specific_numbers_l2618_261877

theorem product_of_specific_numbers (x y : ℝ) 
  (h1 : x - y = 6) 
  (h2 : x^3 - y^3 = 198) : 
  x * y = 5 := by
sorry

end product_of_specific_numbers_l2618_261877


namespace at_least_one_less_than_one_l2618_261821

theorem at_least_one_less_than_one (a b c : ℝ) (ha : a < 3) (hb : b < 3) (hc : c < 3) :
  a < 1 ∨ b < 1 ∨ c < 1 := by
  sorry

end at_least_one_less_than_one_l2618_261821


namespace intersect_at_two_points_l2618_261828

/-- The first function representing y = 2x^2 - x + 3 --/
def f (x : ℝ) : ℝ := 2 * x^2 - x + 3

/-- The second function representing y = -x^2 + x + 5 --/
def g (x : ℝ) : ℝ := -x^2 + x + 5

/-- The difference function between f and g --/
def h (x : ℝ) : ℝ := f x - g x

/-- Theorem stating that the graphs of f and g intersect at exactly two distinct points --/
theorem intersect_at_two_points : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ h x₁ = 0 ∧ h x₂ = 0 ∧ ∀ x, h x = 0 → x = x₁ ∨ x = x₂ := by
  sorry

end intersect_at_two_points_l2618_261828


namespace mrs_lee_june_percentage_l2618_261814

/-- Represents the Lee family's income structure -/
structure LeeIncome where
  total : ℝ
  mrs_lee : ℝ
  mr_lee : ℝ
  jack : ℝ
  rest : ℝ

/-- Calculates the total income for June based on May's income and the given changes -/
def june_total (may : LeeIncome) : ℝ :=
  1.2 * may.mrs_lee + 1.1 * may.mr_lee + 0.85 * may.jack + may.rest

/-- Theorem stating that Mrs. Lee's earnings in June are between 0% and 60% of the total income -/
theorem mrs_lee_june_percentage (may : LeeIncome)
  (h1 : may.mrs_lee = 0.5 * may.total)
  (h2 : may.total = may.mrs_lee + may.mr_lee + may.jack + may.rest)
  (h3 : may.total > 0) :
  0 < (1.2 * may.mrs_lee) / (june_total may) ∧ (1.2 * may.mrs_lee) / (june_total may) < 0.6 := by
  sorry

end mrs_lee_june_percentage_l2618_261814


namespace dot_product_zero_nonzero_vectors_l2618_261887

theorem dot_product_zero_nonzero_vectors :
  ∃ (a b : ℝ × ℝ), a ≠ (0, 0) ∧ b ≠ (0, 0) ∧ a.1 * b.1 + a.2 * b.2 = 0 := by
  sorry

end dot_product_zero_nonzero_vectors_l2618_261887


namespace perfect_square_property_l2618_261897

theorem perfect_square_property (x y p : ℕ+) (hp : Nat.Prime p.val) 
  (h : 4 * x.val^2 + 8 * y.val^2 + (2 * x.val - 3 * y.val) * p.val - 12 * x.val * y.val = 0) :
  ∃ (n : ℕ), 4 * y.val + 1 = n^2 := by
  sorry

end perfect_square_property_l2618_261897


namespace asterisk_replacement_l2618_261838

theorem asterisk_replacement : ∃! (x : ℝ), x > 0 ∧ (x / 20) * (x / 80) = 1 := by
  sorry

end asterisk_replacement_l2618_261838


namespace function_passes_through_point_l2618_261853

theorem function_passes_through_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^(x-1) + 2
  f 1 = 3 := by sorry

end function_passes_through_point_l2618_261853


namespace symmetric_points_sum_power_l2618_261870

-- Define a point as a pair of real numbers
def Point := ℝ × ℝ

-- Define symmetry about y-axis
def symmetric_about_y_axis (p q : Point) : Prop :=
  p.1 = -q.1 ∧ p.2 = q.2

theorem symmetric_points_sum_power (a b : ℝ) :
  symmetric_about_y_axis (a, 3) (4, b) →
  (a + b)^2008 = 1 := by
  sorry

end symmetric_points_sum_power_l2618_261870


namespace unique_modular_solution_l2618_261858

theorem unique_modular_solution : ∃! n : ℤ, n ≡ -5678 [ZMOD 10] ∧ 0 ≤ n ∧ n ≤ 9 ∧ n = 2 := by
  sorry

end unique_modular_solution_l2618_261858


namespace perpendicular_vectors_m_value_l2618_261804

/-- Two planar vectors are perpendicular if and only if their dot product is zero -/
def perpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

/-- The theorem states that for given planar vectors a = (-6, 2) and b = (3, m),
    if they are perpendicular, then m = 9 -/
theorem perpendicular_vectors_m_value :
  let a : ℝ × ℝ := (-6, 2)
  let b : ℝ × ℝ := (3, m)
  perpendicular a b → m = 9 := by
sorry

end perpendicular_vectors_m_value_l2618_261804


namespace remainder_of_difference_l2618_261851

theorem remainder_of_difference (a b : ℕ) (ha : a > 0) (hb : b > 0) 
  (ha_mod : a % 6 = 2) (hb_mod : b % 6 = 3) (hab : a > b) : 
  (a - b) % 6 = 5 := by
  sorry

end remainder_of_difference_l2618_261851


namespace focal_lengths_equal_l2618_261860

/-- Focal length of a hyperbola with equation 15y^2 - x^2 = 15 -/
def hyperbola_focal_length : ℝ := 4

/-- Focal length of an ellipse with equation x^2/25 + y^2/9 = 1 -/
def ellipse_focal_length : ℝ := 4

/-- The focal lengths of the given hyperbola and ellipse are equal -/
theorem focal_lengths_equal : hyperbola_focal_length = ellipse_focal_length := by sorry

end focal_lengths_equal_l2618_261860


namespace soup_ratio_l2618_261850

/-- Given the amount of beef bought, unused beef, and vegetables used, 
    calculate the ratio of vegetables to beef used in the soup -/
theorem soup_ratio (beef_bought : ℚ) (unused_beef : ℚ) (vegetables : ℚ) : 
  beef_bought = 4 → unused_beef = 1 → vegetables = 6 →
  vegetables / (beef_bought - unused_beef) = 2 := by sorry

end soup_ratio_l2618_261850


namespace set_intersection_and_complement_l2618_261844

-- Define the sets A and B
def A : Set ℝ := {x | 0 < x ∧ x ≤ 2}
def B : Set ℝ := {x | x < -3 ∨ x > 1}

-- State the theorem
theorem set_intersection_and_complement :
  (A ∩ B = {x | 1 < x ∧ x ≤ 2}) ∧
  ((Aᶜ ∩ Bᶜ) = {x | -3 ≤ x ∧ x ≤ 0}) := by
  sorry

end set_intersection_and_complement_l2618_261844


namespace stating_anoop_join_time_l2618_261826

/-- Represents the number of months in a year -/
def monthsInYear : ℕ := 12

/-- Represents Arjun's investment in rupees -/
def arjunInvestment : ℕ := 20000

/-- Represents Anoop's investment in rupees -/
def anoopInvestment : ℕ := 4000

/-- 
Theorem stating that if Arjun invests for 12 months and Anoop invests for (12 - x) months,
and their profits are divided equally, then Anoop must have joined after 7 months.
-/
theorem anoop_join_time (x : ℕ) : 
  (arjunInvestment * monthsInYear) / (anoopInvestment * (monthsInYear - x)) = 1 → x = 7 := by
  sorry


end stating_anoop_join_time_l2618_261826


namespace correct_ways_to_leave_shop_l2618_261810

/-- The number of different flavors of oreos --/
def num_oreo_flavors : ℕ := 6

/-- The number of different flavors of milk --/
def num_milk_flavors : ℕ := 4

/-- The total number of product types (oreos + milk) --/
def total_product_types : ℕ := num_oreo_flavors + num_milk_flavors

/-- The number of products they leave the shop with --/
def num_products : ℕ := 4

/-- Function to calculate the number of ways Alpha and Beta can leave the shop --/
def ways_to_leave_shop : ℕ := sorry

/-- Theorem stating the correct number of ways to leave the shop --/
theorem correct_ways_to_leave_shop : ways_to_leave_shop = 2546 := by sorry

end correct_ways_to_leave_shop_l2618_261810


namespace children_fed_theorem_l2618_261869

/-- Represents the number of people a meal can feed -/
structure MealCapacity where
  adults : ℕ
  children : ℕ

/-- Calculates the number of children that can be fed with the remaining food -/
def remainingChildrenFed (totalAdults totalChildren consumedAdultMeals : ℕ) (capacity : MealCapacity) : ℕ :=
  let remainingAdultMeals := capacity.adults - consumedAdultMeals
  let childrenPerAdultMeal := capacity.children / capacity.adults
  remainingAdultMeals * childrenPerAdultMeal

/-- Theorem stating that given the conditions, 63 children can be fed with the remaining food -/
theorem children_fed_theorem (totalAdults totalChildren consumedAdultMeals : ℕ) (capacity : MealCapacity) :
  totalAdults = 55 →
  totalChildren = 70 →
  capacity.adults = 70 →
  capacity.children = 90 →
  consumedAdultMeals = 21 →
  remainingChildrenFed totalAdults totalChildren consumedAdultMeals capacity = 63 := by
  sorry

end children_fed_theorem_l2618_261869


namespace night_crew_ratio_l2618_261848

theorem night_crew_ratio (day_workers : ℝ) (night_workers : ℝ) (boxes_per_day_worker : ℝ) 
  (h1 : day_workers > 0)
  (h2 : night_workers > 0)
  (h3 : boxes_per_day_worker > 0)
  (h4 : day_workers * boxes_per_day_worker = 0.7 * (day_workers * boxes_per_day_worker + night_workers * (3/4 * boxes_per_day_worker))) :
  night_workers / day_workers = 4/7 := by
  sorry

end night_crew_ratio_l2618_261848


namespace white_ducks_count_l2618_261864

theorem white_ducks_count (fish_per_white : ℕ) (fish_per_black : ℕ) (fish_per_multi : ℕ)
  (black_ducks : ℕ) (multi_ducks : ℕ) (total_fish : ℕ)
  (h1 : fish_per_white = 5)
  (h2 : fish_per_black = 10)
  (h3 : fish_per_multi = 12)
  (h4 : black_ducks = 7)
  (h5 : multi_ducks = 6)
  (h6 : total_fish = 157) :
  ∃ white_ducks : ℕ, white_ducks * fish_per_white + black_ducks * fish_per_black + multi_ducks * fish_per_multi = total_fish ∧ white_ducks = 3 :=
by
  sorry

end white_ducks_count_l2618_261864


namespace book_loss_percentage_l2618_261841

/-- If the cost price of 8 books equals the selling price of 16 books, then the loss percentage is 50%. -/
theorem book_loss_percentage (C S : ℝ) (h : 8 * C = 16 * S) : 
  (C - S) / C * 100 = 50 := by
  sorry

end book_loss_percentage_l2618_261841


namespace last_two_digits_33_divisible_by_prime_gt_7_l2618_261815

theorem last_two_digits_33_divisible_by_prime_gt_7 (n : ℕ) :
  (∃ k : ℕ, n = 100 * k + 33) →
  ∃ p : ℕ, p > 7 ∧ Prime p ∧ p ∣ n :=
by sorry

end last_two_digits_33_divisible_by_prime_gt_7_l2618_261815


namespace megan_pop_albums_l2618_261881

/-- The number of songs on each album -/
def songs_per_album : ℕ := 7

/-- The number of country albums bought -/
def country_albums : ℕ := 2

/-- The total number of songs bought -/
def total_songs : ℕ := 70

/-- The number of pop albums bought -/
def pop_albums : ℕ := (total_songs - country_albums * songs_per_album) / songs_per_album

theorem megan_pop_albums : pop_albums = 8 := by sorry

end megan_pop_albums_l2618_261881


namespace jersey_profit_calculation_l2618_261802

/-- The amount of money made from each jersey -/
def jersey_profit : ℝ := 165

/-- The number of jerseys sold -/
def jerseys_sold : ℕ := 156

/-- The total money made from selling jerseys -/
def total_jersey_profit : ℝ := jersey_profit * (jerseys_sold : ℝ)

theorem jersey_profit_calculation : total_jersey_profit = 25740 := by
  sorry

end jersey_profit_calculation_l2618_261802


namespace tetrachloromethane_formation_l2618_261879

-- Define the chemical species
structure ChemicalSpecies where
  name : String
  moles : ℝ

-- Define the reaction equation
structure ReactionEquation where
  reactants : List ChemicalSpecies
  products : List ChemicalSpecies

-- Define the problem parameters
def methane : ChemicalSpecies := ⟨"CH4", 1⟩
def chlorine : ChemicalSpecies := ⟨"Cl2", 4⟩
def tetrachloromethane : ChemicalSpecies := ⟨"CCl4", 0⟩ -- Initial amount is 0
def hydrogenChloride : ChemicalSpecies := ⟨"HCl", 0⟩ -- Initial amount is 0

-- Define the balanced reaction equation
def balancedEquation : ReactionEquation :=
  ⟨[methane, chlorine], [tetrachloromethane, hydrogenChloride]⟩

-- Theorem statement
theorem tetrachloromethane_formation
  (reactionEq : ReactionEquation)
  (h1 : reactionEq = balancedEquation)
  (h2 : methane.moles = 1)
  (h3 : chlorine.moles = 4) :
  tetrachloromethane.moles = 1 :=
sorry

end tetrachloromethane_formation_l2618_261879


namespace cos_squared_pi_third_minus_x_l2618_261817

theorem cos_squared_pi_third_minus_x (x : ℝ) (h : Real.sin (x + π/6) = 1/4) :
  Real.cos (π/3 - x) ^ 2 = 1/16 := by
  sorry

end cos_squared_pi_third_minus_x_l2618_261817


namespace pipe_length_is_35_l2618_261822

/-- The length of the pipe in meters -/
def pipe_length : ℝ := 35

/-- The length of Yura's step in meters -/
def step_length : ℝ := 1

/-- The number of steps Yura took against the movement of the tractor -/
def steps_against : ℕ := 20

/-- The number of steps Yura took with the movement of the tractor -/
def steps_with : ℕ := 140

/-- Theorem stating that the pipe length is 35 meters -/
theorem pipe_length_is_35 : 
  ∃ (x : ℝ), 
    (step_length * steps_against : ℝ) = pipe_length - x ∧ 
    (step_length * steps_with : ℝ) = pipe_length + 7 * x ∧
    pipe_length = 35 := by sorry

end pipe_length_is_35_l2618_261822


namespace fir_trees_count_l2618_261803

theorem fir_trees_count : ∃ (n : ℕ), 
  (n ≠ 15) ∧ 
  (n % 11 = 0) ∧ 
  (n < 25) ∧ 
  (n % 22 ≠ 0) ∧
  (n = 11) := by
  sorry

end fir_trees_count_l2618_261803


namespace geometric_series_sum_l2618_261883

theorem geometric_series_sum (a r : ℝ) (n : ℕ) (h1 : r ≠ 1) (h2 : n > 0) :
  let last_term := a * r^(n - 1)
  let series_sum := a * (r^n - 1) / (r - 1)
  (a = 2 ∧ r = 3 ∧ last_term = 4374) → series_sum = 6560 := by
  sorry

end geometric_series_sum_l2618_261883


namespace chord_distance_from_center_l2618_261800

theorem chord_distance_from_center (R : ℝ) (chord_length : ℝ) (h1 : R = 13) (h2 : chord_length = 10) :
  ∃ d : ℝ, d = 12 ∧ d^2 + (chord_length/2)^2 = R^2 :=
by sorry

end chord_distance_from_center_l2618_261800


namespace twelve_switches_four_connections_l2618_261889

/-- The number of connections in a network of switches where each switch connects to a fixed number of others. -/
def connections (n : ℕ) (k : ℕ) : ℕ := n * k / 2

/-- Theorem: In a network of 12 switches, where each switch is directly connected to exactly 4 other switches, the total number of connections is 24. -/
theorem twelve_switches_four_connections :
  connections 12 4 = 24 := by
  sorry

end twelve_switches_four_connections_l2618_261889


namespace quarterly_charge_is_80_l2618_261823

/-- The Kwik-e-Tax Center pricing structure and sales data -/
structure TaxCenter where
  federal_charge : ℕ
  state_charge : ℕ
  federal_sold : ℕ
  state_sold : ℕ
  quarterly_sold : ℕ
  total_revenue : ℕ

/-- The charge for quarterly business taxes -/
def quarterly_charge (tc : TaxCenter) : ℕ :=
  (tc.total_revenue - (tc.federal_charge * tc.federal_sold + tc.state_charge * tc.state_sold)) / tc.quarterly_sold

/-- Theorem stating the charge for quarterly business taxes is $80 -/
theorem quarterly_charge_is_80 (tc : TaxCenter) 
  (h1 : tc.federal_charge = 50)
  (h2 : tc.state_charge = 30)
  (h3 : tc.federal_sold = 60)
  (h4 : tc.state_sold = 20)
  (h5 : tc.quarterly_sold = 10)
  (h6 : tc.total_revenue = 4400) :
  quarterly_charge tc = 80 := by
  sorry

#eval quarterly_charge { federal_charge := 50, state_charge := 30, federal_sold := 60, state_sold := 20, quarterly_sold := 10, total_revenue := 4400 }

end quarterly_charge_is_80_l2618_261823


namespace union_of_A_and_B_l2618_261834

def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 2}
def B : Set ℝ := {x : ℝ | 1 < x ∧ x ≤ 3}

theorem union_of_A_and_B :
  A ∪ B = {x : ℝ | -1 ≤ x ∧ x ≤ 3} := by sorry

end union_of_A_and_B_l2618_261834


namespace light_distance_250_years_l2618_261836

/-- The distance light travels in one year in miles -/
def light_year_distance : ℝ := 5870000000000

/-- The number of years we're calculating for -/
def years : ℝ := 250

/-- The theorem stating the distance light travels in 250 years -/
theorem light_distance_250_years : 
  light_year_distance * years = 1.4675 * (10 : ℝ) ^ 15 := by
  sorry

end light_distance_250_years_l2618_261836


namespace min_perimeter_triangle_l2618_261849

-- Define the triangle
structure Triangle where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  angleA : ℝ
  angleB : ℝ
  angleC : ℝ

-- Define the conditions
def validTriangle (t : Triangle) : Prop :=
  t.angleA = 2 * t.angleB ∧
  t.angleC > Real.pi / 2 ∧
  t.angleA + t.angleB + t.angleC = Real.pi

-- Define the perimeter
def perimeter (t : Triangle) : ℕ :=
  t.a.val + t.b.val + t.c.val

-- Theorem statement
theorem min_perimeter_triangle :
  ∃ (t : Triangle), validTriangle t ∧
    (∀ (t' : Triangle), validTriangle t' → perimeter t ≤ perimeter t') ∧
    perimeter t = 77 :=
sorry

end min_perimeter_triangle_l2618_261849


namespace int_roots_count_l2618_261891

/-- A polynomial of degree 4 with integer coefficients -/
structure IntPoly4 where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ

/-- The number of integer roots of a polynomial, counting multiplicity -/
def num_int_roots (p : IntPoly4) : ℕ := sorry

/-- Theorem stating the possible values for the number of integer roots -/
theorem int_roots_count (p : IntPoly4) : 
  num_int_roots p = 0 ∨ num_int_roots p = 1 ∨ num_int_roots p = 2 ∨ num_int_roots p = 4 :=
sorry

end int_roots_count_l2618_261891


namespace total_spent_equals_621_l2618_261857

/-- The total amount spent by Tate and Peyton on their remaining tickets -/
def total_spent (tate_initial_tickets : ℕ) (tate_initial_price : ℕ) 
  (tate_additional_tickets : ℕ) (tate_additional_price : ℕ)
  (peyton_price : ℕ) : ℕ :=
  let tate_total := tate_initial_tickets * tate_initial_price + 
                    tate_additional_tickets * tate_additional_price
  let peyton_initial_tickets := tate_initial_tickets / 2
  let peyton_remaining_tickets := peyton_initial_tickets - 
                                  (peyton_initial_tickets / 3)
  let peyton_total := peyton_remaining_tickets * peyton_price
  tate_total + peyton_total

/-- Theorem stating the total amount spent by Tate and Peyton -/
theorem total_spent_equals_621 : 
  total_spent 32 14 2 15 13 = 621 := by
  sorry

end total_spent_equals_621_l2618_261857


namespace rectangle_containment_exists_l2618_261818

/-- Represents a rectangle with integer dimensions -/
structure Rectangle where
  width : Nat
  height : Nat

/-- The set of all rectangles with positive integer dimensions -/
def RectangleSet : Set Rectangle :=
  {r : Rectangle | r.width > 0 ∧ r.height > 0}

/-- Predicate to check if one rectangle is contained within another -/
def contains (r1 r2 : Rectangle) : Prop :=
  r1.width ≤ r2.width ∧ r1.height ≤ r2.height

theorem rectangle_containment_exists :
  ∃ r1 r2 : Rectangle, r1 ∈ RectangleSet ∧ r2 ∈ RectangleSet ∧ r1 ≠ r2 ∧ contains r1 r2 := by
  sorry

end rectangle_containment_exists_l2618_261818


namespace jeans_pricing_markup_l2618_261874

theorem jeans_pricing_markup (cost : ℝ) (h : cost > 0) :
  let retailer_price := cost * 1.4
  let customer_price := retailer_price * 1.3
  (customer_price - cost) / cost = 0.82 := by
sorry

end jeans_pricing_markup_l2618_261874


namespace quadratic_inequality_solution_set_l2618_261892

theorem quadratic_inequality_solution_set :
  ∀ x : ℝ, x^2 + x < 0 ↔ -1/2 < x ∧ x < 1/2 := by sorry

end quadratic_inequality_solution_set_l2618_261892


namespace negation_of_existence_negation_of_squared_plus_one_less_than_zero_l2618_261847

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x : ℝ, P x) ↔ (∀ x : ℝ, ¬ P x) := by sorry

theorem negation_of_squared_plus_one_less_than_zero :
  (¬ ∃ x : ℝ, x^2 + 1 < 0) ↔ (∀ x : ℝ, x^2 + 1 ≥ 0) := by sorry

end negation_of_existence_negation_of_squared_plus_one_less_than_zero_l2618_261847


namespace cube_side_length_is_three_l2618_261855

/-- Represents a cube with side length n -/
structure Cube where
  n : ℕ

/-- Calculates the total number of faces of all unit cubes after slicing -/
def totalFaces (c : Cube) : ℕ := 6 * c.n^3

/-- Calculates the number of blue faces (surface area of the original cube) -/
def blueFaces (c : Cube) : ℕ := 6 * c.n^2

/-- Theorem: If one-third of all faces are blue, then the cube's side length is 3 -/
theorem cube_side_length_is_three (c : Cube) :
  3 * blueFaces c = totalFaces c → c.n = 3 := by
  sorry

end cube_side_length_is_three_l2618_261855


namespace part_one_part_two_l2618_261895

-- Define the function f(x) = |x-1|
def f (x : ℝ) : ℝ := |x - 1|

-- Theorem for part I
theorem part_one : ∀ x : ℝ, f x - f (x + 2) < 1 ↔ x > -1/2 := by sorry

-- Theorem for part II
theorem part_two : (∀ x : ℝ, x ∈ Set.Icc 1 2 → x - f (x + 1 - a) ≤ 1) → (a ≤ 1 ∨ a ≥ 3) := by sorry

end part_one_part_two_l2618_261895


namespace sinusoidal_symmetric_center_l2618_261899

/-- Given a sinusoidal function with specific properties, prove that its symmetric center is at (-2π/3, 0) -/
theorem sinusoidal_symmetric_center 
  (f : ℝ → ℝ) 
  (ω φ : ℝ) 
  (h_omega_pos : ω > 0)
  (h_phi_bound : |φ| < π/2)
  (h_f_def : ∀ x, f x = Real.sin (ω * x + φ))
  (h_period : (2 * π) / ω = 4 * π)
  (h_max_at_pi_third : ∀ x, f x ≤ f (π/3)) :
  ∃ (y : ℝ), ∀ (x : ℝ), f (x - (-2*π/3)) = f (-x - (-2*π/3)) ∧ f (-2*π/3) = y :=
sorry

end sinusoidal_symmetric_center_l2618_261899


namespace east_north_not_opposite_forward_backward_opposite_main_theorem_l2618_261898

/-- Represents a direction of movement --/
inductive Direction
  | Forward
  | Backward
  | East
  | North

/-- Represents a quantity with a value and a direction --/
structure Quantity where
  value : ℝ
  direction : Direction

/-- Defines when two quantities are opposite --/
def are_opposite (q1 q2 : Quantity) : Prop :=
  (q1.value = q2.value) ∧
  ((q1.direction = Direction.Forward ∧ q2.direction = Direction.Backward) ∨
   (q1.direction = Direction.Backward ∧ q2.direction = Direction.Forward))

/-- Theorem stating that east and north movements are not opposite --/
theorem east_north_not_opposite :
  ¬(are_opposite
      { value := 10, direction := Direction.East }
      { value := 10, direction := Direction.North }) :=
by
  sorry

/-- Theorem stating that forward and backward movements are opposite --/
theorem forward_backward_opposite :
  are_opposite
    { value := 5, direction := Direction.Forward }
    { value := 5, direction := Direction.Backward } :=
by
  sorry

/-- Main theorem proving that east and north movements are not opposite,
    while forward and backward movements are opposite --/
theorem main_theorem :
  (¬(are_opposite
      { value := 10, direction := Direction.East }
      { value := 10, direction := Direction.North })) ∧
  (are_opposite
    { value := 5, direction := Direction.Forward }
    { value := 5, direction := Direction.Backward }) :=
by
  sorry

end east_north_not_opposite_forward_backward_opposite_main_theorem_l2618_261898


namespace election_outcome_depends_on_radicals_l2618_261816

/-- Represents a political group in the election -/
inductive PoliticalGroup
| Socialist
| Republican
| Radical
| Other

/-- Represents the election models -/
inductive ElectionModel
| A
| B

/-- Represents the election system with four political groups -/
structure ElectionSystem where
  groups : Fin 4 → PoliticalGroup
  groupSize : ℕ
  socialistsPrefB : ℕ
  republicansPrefA : ℕ
  radicalSupport : PoliticalGroup

/-- The outcome of the election -/
def electionOutcome (system : ElectionSystem) : ElectionModel :=
  match system.radicalSupport with
  | PoliticalGroup.Socialist => ElectionModel.B
  | PoliticalGroup.Republican => ElectionModel.A
  | _ => sorry -- This case should not occur in our scenario

/-- Theorem stating that the election outcome depends on radicals' support -/
theorem election_outcome_depends_on_radicals (system : ElectionSystem) 
  (h1 : system.socialistsPrefB = system.republicansPrefA)
  (h2 : system.socialistsPrefB > 0) :
  (∃ (support : PoliticalGroup), 
    electionOutcome {system with radicalSupport := support} = ElectionModel.A) ∧
  (∃ (support : PoliticalGroup), 
    electionOutcome {system with radicalSupport := support} = ElectionModel.B) :=
  sorry


end election_outcome_depends_on_radicals_l2618_261816


namespace factor_expression_l2618_261894

theorem factor_expression (y : ℝ) : 5 * y * (y - 2) + 11 * (y - 2) = (y - 2) * (5 * y + 11) := by
  sorry

end factor_expression_l2618_261894


namespace selection_method1_selection_method2_selection_method3_selection_method4_l2618_261896

/-- The number of ways to select k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The total number of athletes -/
def total_athletes : ℕ := 10

/-- The number of male athletes -/
def male_athletes : ℕ := 6

/-- The number of female athletes -/
def female_athletes : ℕ := 4

/-- The number of athletes to be selected -/
def selected_athletes : ℕ := 5

/-- The number of ways to select 3 males and 2 females -/
theorem selection_method1 : choose male_athletes 3 * choose female_athletes 2 = 120 := sorry

/-- The number of ways to select with at least one captain participating -/
theorem selection_method2 : 2 * choose 8 4 + choose 8 3 = 196 := sorry

/-- The number of ways to select with at least one female athlete -/
theorem selection_method3 : choose total_athletes selected_athletes - choose male_athletes selected_athletes = 246 := sorry

/-- The number of ways to select with both a captain and at least one female athlete -/
theorem selection_method4 : choose 9 4 + choose 8 4 - choose 5 4 = 191 := sorry

end selection_method1_selection_method2_selection_method3_selection_method4_l2618_261896


namespace impossible_sum_110_l2618_261808

def coin_values : List ℕ := [1, 5, 10, 25, 50]

theorem impossible_sum_110 : 
  ¬ ∃ (coins : List ℕ), 
    coins.length = 6 ∧ 
    (∀ c ∈ coins, c ∈ coin_values) ∧ 
    coins.sum = 110 :=
sorry

end impossible_sum_110_l2618_261808


namespace rd_participation_and_optimality_l2618_261875

/-- Represents a firm engaged in R&D -/
structure Firm where
  participates : Bool

/-- Represents the R&D scenario in country A -/
structure RDScenario where
  V : ℝ  -- Value of successful solo development
  α : ℝ  -- Probability of success
  IC : ℝ  -- Investment cost
  firms : Fin 2 → Firm

/-- Expected revenue for a firm when both participate -/
def expectedRevenueBoth (s : RDScenario) : ℝ :=
  s.α * (1 - s.α) * s.V + 0.5 * s.α^2 * s.V

/-- Expected revenue for a firm when only one participates -/
def expectedRevenueOne (s : RDScenario) : ℝ :=
  s.α * s.V

/-- Condition for both firms to participate -/
def bothParticipateCondition (s : RDScenario) : Prop :=
  s.V * s.α * (1 - 0.5 * s.α) ≥ s.IC

/-- Total expected profit when both firms participate -/
def totalProfitBoth (s : RDScenario) : ℝ :=
  2 * (expectedRevenueBoth s - s.IC)

/-- Total expected profit when only one firm participates -/
def totalProfitOne (s : RDScenario) : ℝ :=
  expectedRevenueOne s - s.IC

/-- Theorem stating the conditions for participation and social optimality -/
theorem rd_participation_and_optimality (s : RDScenario) 
    (h_α_pos : 0 < s.α) (h_α_lt_one : s.α < 1) :
  bothParticipateCondition s ↔ 
    expectedRevenueBoth s ≥ s.IC ∧
    (s.V = 16 ∧ s.α = 0.5 ∧ s.IC = 5 → 
      bothParticipateCondition s ∧ totalProfitOne s > totalProfitBoth s) :=
sorry

end rd_participation_and_optimality_l2618_261875


namespace solution_set_of_inequality_l2618_261843

theorem solution_set_of_inequality (x : ℝ) :
  (((x - 2) / (x + 3) > 0) ↔ (x ∈ Set.Iio (-3) ∪ Set.Ioi 2)) :=
by sorry

end solution_set_of_inequality_l2618_261843


namespace replaced_lettuce_cost_is_1_75_l2618_261856

/-- Represents the grocery order with its components -/
structure GroceryOrder where
  originalTotal : ℝ
  tomatoesOld : ℝ
  tomatoesNew : ℝ
  lettuceOld : ℝ
  celeryOld : ℝ
  celeryNew : ℝ
  deliveryAndTip : ℝ
  newTotal : ℝ

/-- The cost of the replaced lettuce given the grocery order details -/
def replacedLettuceCost (order : GroceryOrder) : ℝ :=
  order.lettuceOld + (order.newTotal - order.originalTotal - order.deliveryAndTip) -
  ((order.tomatoesNew - order.tomatoesOld) + (order.celeryNew - order.celeryOld))

/-- Theorem stating that the cost of the replaced lettuce is $1.75 -/
theorem replaced_lettuce_cost_is_1_75 (order : GroceryOrder)
  (h1 : order.originalTotal = 25)
  (h2 : order.tomatoesOld = 0.99)
  (h3 : order.tomatoesNew = 2.20)
  (h4 : order.lettuceOld = 1.00)
  (h5 : order.celeryOld = 1.96)
  (h6 : order.celeryNew = 2.00)
  (h7 : order.deliveryAndTip = 8.00)
  (h8 : order.newTotal = 35) :
  replacedLettuceCost order = 1.75 := by
  sorry

end replaced_lettuce_cost_is_1_75_l2618_261856


namespace kerman_triple_49_64_15_l2618_261831

/-- Definition of a Kerman triple -/
def is_kerman_triple (a b x : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ x > 0 ∧ Real.sqrt (a : ℝ) + Real.sqrt (b : ℝ) = x

/-- Theorem: (49, 64, 15) is a Kerman triple -/
theorem kerman_triple_49_64_15 :
  is_kerman_triple 49 64 15 := by
  sorry

end kerman_triple_49_64_15_l2618_261831


namespace max_distinct_numbers_with_prime_triple_sums_l2618_261871

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- A function that checks if the sum of any three numbers in a list is prime -/
def allTripleSumsPrime (l : List ℕ) : Prop :=
  ∀ a b c : ℕ, a ∈ l → b ∈ l → c ∈ l → a ≠ b → b ≠ c → a ≠ c → isPrime (a + b + c)

/-- The theorem stating that the maximum number of distinct natural numbers
    that can be chosen such that the sum of any three of them is prime is 4 -/
theorem max_distinct_numbers_with_prime_triple_sums :
  (∃ l : List ℕ, l.length = 4 ∧ l.Nodup ∧ allTripleSumsPrime l) ∧
  (∀ l : List ℕ, l.length > 4 → ¬(l.Nodup ∧ allTripleSumsPrime l)) :=
sorry

end max_distinct_numbers_with_prime_triple_sums_l2618_261871


namespace hyperbola_equation_l2618_261827

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1) →
  (∃ (x : ℝ), x = 6 ∧ x^2 = a^2 + b^2) →
  (∃ (x y : ℝ), y = Real.sqrt 3 * x ∧ b / a = Real.sqrt 3) →
  a^2 = 9 ∧ b^2 = 27 :=
by sorry

end hyperbola_equation_l2618_261827


namespace imaginary_part_of_z_l2618_261805

theorem imaginary_part_of_z (z : ℂ) : z - Complex.I = (4 - 2 * Complex.I) / (1 + 2 * Complex.I) → z.im = -1 := by
  sorry

end imaginary_part_of_z_l2618_261805


namespace ellipse_intersection_theorem_l2618_261865

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a line passing through the origin -/
structure Line where
  slope : ℝ

/-- The problem statement -/
theorem ellipse_intersection_theorem (C : Ellipse) (l₁ : Line) :
  -- The ellipse passes through (2, 1)
  (2 / C.a)^2 + (1 / C.b)^2 = 1 →
  -- The eccentricity is √3/2
  (C.a^2 - C.b^2) / C.a^2 = 3/4 →
  -- There exists a point M on x - y + 2√6 = 0 such that MPQ is equilateral
  ∃ (M : ℝ × ℝ), M.1 - M.2 + 2 * Real.sqrt 6 = 0 ∧
    -- (Condition for equilateral triangle, simplified)
    (M.1^2 + M.2^2) = 3 * ((C.a * C.b * l₁.slope / Real.sqrt (C.a^2 * l₁.slope^2 + C.b^2))^2 + 
                           (C.a * C.b / Real.sqrt (C.a^2 * l₁.slope^2 + C.b^2))^2) →
  -- Then l₁ is either y = 0 or y = 2x/7
  l₁.slope = 0 ∨ l₁.slope = 2/7 := by
  sorry

end ellipse_intersection_theorem_l2618_261865


namespace sum_of_squares_reciprocals_l2618_261820

theorem sum_of_squares_reciprocals (a b c : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c)
  (h4 : a / (b - c) + b / (c - a) + c / (a - b) = 1) :
  a / (b - c)^2 + b / (c - a)^2 + c / (a - b)^2 = 0 := by
  sorry

end sum_of_squares_reciprocals_l2618_261820


namespace sequence_problem_l2618_261872

theorem sequence_problem (a b c : ℤ → ℝ) 
  (h_positive : ∀ n, a n > 0 ∧ b n > 0 ∧ c n > 0)
  (h_a : ∀ n, a n ≥ (b (n+1) + c (n-1)) / 2)
  (h_b : ∀ n, b n ≥ (c (n+1) + a (n-1)) / 2)
  (h_c : ∀ n, c n ≥ (a (n+1) + b (n-1)) / 2)
  (h_init : a 0 = 26 ∧ b 0 = 6 ∧ c 0 = 2004) :
  a 2005 = 2004 ∧ b 2005 = 26 ∧ c 2005 = 6 := by
sorry

end sequence_problem_l2618_261872


namespace intersection_of_A_and_B_l2618_261845

def A : Set ℝ := {x : ℝ | x > 0}
def B : Set ℝ := {x : ℝ | x < 1}

theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 0 < x ∧ x < 1} := by sorry

end intersection_of_A_and_B_l2618_261845


namespace student_selection_methods_l2618_261893

def total_students : ℕ := 8
def num_boys : ℕ := 6
def num_girls : ℕ := 2
def students_to_select : ℕ := 4
def boys_to_select : ℕ := 3
def girls_to_select : ℕ := 1

theorem student_selection_methods :
  (Nat.choose num_boys boys_to_select) * (Nat.choose num_girls girls_to_select) = 40 := by
  sorry

end student_selection_methods_l2618_261893


namespace part1_part2_l2618_261868

-- Define the function f
def f (a x : ℝ) : ℝ := x^2 - (a-1)*x + a-2

-- Part 1
theorem part1 (a : ℝ) : 
  (∀ x : ℝ, f a x ≥ -2) ↔ (3 - 2*Real.sqrt 2 ≤ a ∧ a ≤ 3 + 2*Real.sqrt 2) :=
sorry

-- Part 2
theorem part2 (a x : ℝ) :
  (a < 3 → (f a x < 0 ↔ a-2 < x ∧ x < 1)) ∧
  (a = 3 → ¬∃ x, f a x < 0) ∧
  (a > 3 → (f a x < 0 ↔ 1 < x ∧ x < a-2)) :=
sorry

end part1_part2_l2618_261868


namespace share_of_b_l2618_261830

theorem share_of_b (A B C : ℕ) : 
  A = 3 * B → 
  B = C + 25 → 
  A + B + C = 645 → 
  B = 134 := by
sorry

end share_of_b_l2618_261830


namespace school_trip_probabilities_l2618_261854

/-- Represents the setup of a school trip with students and a teacher assigned to cities. -/
structure SchoolTrip where
  numStudents : Nat
  numCities : Nat
  studentsPerCity : Nat

/-- Defines the probability of event A: student a and the teacher go to the same city. -/
def probA (trip : SchoolTrip) : ℚ :=
  1 / trip.numCities

/-- Defines the probability of event B: students a and b go to the same city. -/
def probB (trip : SchoolTrip) : ℚ :=
  1 / (trip.numStudents - 1)

/-- Defines the expected value of ξ, the total number of occurrences of events A and B. -/
def expectedXi (trip : SchoolTrip) : ℚ :=
  8 / 15

/-- Theorem stating the probabilities and expected value for the given school trip scenario. -/
theorem school_trip_probabilities (trip : SchoolTrip) :
  trip.numStudents = 6 ∧ trip.numCities = 3 ∧ trip.studentsPerCity = 2 →
  probA trip = 1/3 ∧ probB trip = 1/5 ∧ expectedXi trip = 8/15 := by
  sorry


end school_trip_probabilities_l2618_261854


namespace zoo_ticket_cost_is_correct_l2618_261813

/-- The cost of a zoo entry ticket per person -/
def zoo_ticket_cost : ℝ := 5

/-- The one-way bus fare per person -/
def bus_fare : ℝ := 1.5

/-- The total amount of money brought -/
def total_amount : ℝ := 40

/-- The amount left after buying tickets and paying for bus fare -/
def amount_left : ℝ := 24

/-- The number of people -/
def num_people : ℕ := 2

theorem zoo_ticket_cost_is_correct : 
  zoo_ticket_cost = (total_amount - amount_left - 2 * num_people * bus_fare) / num_people := by
  sorry

end zoo_ticket_cost_is_correct_l2618_261813


namespace shirt_cost_l2618_261880

theorem shirt_cost (jeans_cost shirt_cost : ℚ) : 
  (3 * jeans_cost + 2 * shirt_cost = 69) →
  (2 * jeans_cost + 3 * shirt_cost = 61) →
  shirt_cost = 9 := by
sorry

end shirt_cost_l2618_261880


namespace four_double_prime_value_l2618_261890

-- Define the prime operation
def prime (q : ℝ) : ℝ := 3 * q - 3

-- State the theorem
theorem four_double_prime_value : prime (prime 4) = 24 := by
  sorry

end four_double_prime_value_l2618_261890


namespace sum_of_digits_of_greatest_prime_divisor_l2618_261839

-- Define the number to factorize
def n : ℕ := 65535

-- Define a function to get the greatest prime divisor
def greatest_prime_divisor (m : ℕ) : ℕ := sorry

-- Define a function to sum the digits of a number
def sum_of_digits (m : ℕ) : ℕ := sorry

-- Theorem statement
theorem sum_of_digits_of_greatest_prime_divisor :
  sum_of_digits (greatest_prime_divisor n) = 14 := by sorry

end sum_of_digits_of_greatest_prime_divisor_l2618_261839


namespace last_digit_of_A_l2618_261861

theorem last_digit_of_A (A : ℕ) : 
  A = (2+1)*(2^2+1)*(2^4+1)*(2^8+1)+1 → 
  A % 10 = (2^16) % 10 := by
sorry

end last_digit_of_A_l2618_261861


namespace prize_buying_l2618_261812

/-- Given the conditions for prize buying, prove the number of pens and notebooks. -/
theorem prize_buying (x y : ℝ) (h1 : 60 * (x + 2*y) = 50 * (x + 3*y)) 
  (total_budget : ℝ) (h2 : total_budget = 60 * (x + 2*y)) : 
  (total_budget / x = 100) ∧ (total_budget / y = 300) := by
  sorry

end prize_buying_l2618_261812


namespace article_price_calculation_l2618_261882

theorem article_price_calculation (p q : ℝ) : 
  let final_price := 1
  let price_after_increase (x : ℝ) := x * (1 + p / 100)
  let price_after_decrease (y : ℝ) := y * (1 - q / 100)
  let original_price := 10000 / (10000 + 100 * (p - q) - p * q)
  price_after_decrease (price_after_increase original_price) = final_price :=
by sorry

end article_price_calculation_l2618_261882


namespace olympic_medal_awards_l2618_261884

/-- The number of ways to award medals in the Olympic 100-meter finals -/
def medal_awards (total_sprinters : ℕ) (american_sprinters : ℕ) (medals : ℕ) : ℕ :=
  let non_american_sprinters := total_sprinters - american_sprinters
  let no_american_medals := non_american_sprinters.descFactorial medals
  let one_american_medal := american_sprinters * medals * (non_american_sprinters.descFactorial (medals - 1))
  no_american_medals + one_american_medal

/-- Theorem stating the number of ways to award medals in the given scenario -/
theorem olympic_medal_awards :
  medal_awards 10 4 3 = 480 :=
by sorry

end olympic_medal_awards_l2618_261884


namespace jack_classic_collection_l2618_261852

/-- The number of books each author has in Jack's classic collection -/
def books_per_author (total_books : ℕ) (num_authors : ℕ) : ℕ :=
  total_books / num_authors

/-- Theorem stating that each author has 33 books in Jack's classic collection -/
theorem jack_classic_collection :
  let total_books : ℕ := 198
  let num_authors : ℕ := 6
  books_per_author total_books num_authors = 33 := by
sorry

end jack_classic_collection_l2618_261852


namespace complex_square_l2618_261863

theorem complex_square (z : ℂ) (i : ℂ) (h1 : z = 5 - 3 * i) (h2 : i^2 = -1) :
  z^2 = 34 - 30 * i := by
  sorry

end complex_square_l2618_261863


namespace smallest_perimeter_consecutive_sides_l2618_261829

theorem smallest_perimeter_consecutive_sides (a b c : ℕ) : 
  a > 2 →
  b = a + 1 →
  c = a + 2 →
  (∀ x y z : ℕ, x > 2 ∧ y = x + 1 ∧ z = x + 2 → a + b + c ≤ x + y + z) →
  a + b + c = 12 :=
sorry

end smallest_perimeter_consecutive_sides_l2618_261829


namespace restaurant_menu_combinations_l2618_261876

theorem restaurant_menu_combinations : 
  (12 * 11) * (12 * 10) = 15840 := by sorry

end restaurant_menu_combinations_l2618_261876


namespace inequality_not_true_l2618_261825

theorem inequality_not_true : Real.sqrt 2 + Real.sqrt 10 ≤ 2 * Real.sqrt 6 := by
  sorry

end inequality_not_true_l2618_261825


namespace inequality_solution_set_l2618_261866

theorem inequality_solution_set (x : ℝ) : 
  abs (2 * x - 1) + abs (2 * x + 1) ≤ 6 ↔ x ∈ Set.Icc (-3/2) (3/2) := by
  sorry

end inequality_solution_set_l2618_261866


namespace min_value_of_x_l2618_261878

theorem min_value_of_x (x : ℝ) : 
  (∀ a : ℝ, a > 0 → x^2 ≤ 1 + a) → 
  x ≥ -1 := by sorry

end min_value_of_x_l2618_261878


namespace nested_fourth_root_solution_l2618_261807

/-- The positive solution to the nested fourth root equation --/
noncomputable def x : ℝ := 3.1412

/-- The left-hand side of the equation --/
noncomputable def lhs (x : ℝ) : ℝ := Real.sqrt (x + Real.sqrt (x + Real.sqrt (x + Real.sqrt x)))

/-- The right-hand side of the equation --/
noncomputable def rhs (x : ℝ) : ℝ := Real.sqrt (x * Real.sqrt (x * Real.sqrt (x * Real.sqrt x)))

/-- Theorem stating that x is the positive solution to the equation --/
theorem nested_fourth_root_solution :
  lhs x = rhs x ∧ x > 0 := by sorry

end nested_fourth_root_solution_l2618_261807


namespace largest_band_members_l2618_261824

theorem largest_band_members : ∃ (m r x : ℕ),
  m < 100 ∧
  r * x + 3 = m ∧
  (r - 3) * (x + 1) = m ∧
  ∀ (m' r' x' : ℕ),
    m' < 100 →
    r' * x' + 3 = m' →
    (r' - 3) * (x' + 1) = m' →
    m' ≤ m ∧
  m = 87 := by
sorry

end largest_band_members_l2618_261824


namespace inequality_holds_iff_m_in_range_l2618_261837

def f (x : ℝ) := x^2 - 1

theorem inequality_holds_iff_m_in_range :
  ∀ m : ℝ, (∀ x ≥ 3, f (x / m) - 4 * m^2 * f x ≤ f (x - 1) + 4 * f m) ↔
    m ≤ -Real.sqrt 2 / 2 ∨ m ≥ Real.sqrt 2 / 2 := by sorry

end inequality_holds_iff_m_in_range_l2618_261837

import Mathlib

namespace NUMINAMATH_CALUDE_circle_equation_l698_69818

theorem circle_equation (x y : ℝ) :
  let center := (3, 4)
  let point := (0, 0)
  let equation := (x - 3)^2 + (y - 4)^2 = 25
  (∀ p, p.1^2 + p.2^2 = (p.1 - center.1)^2 + (p.2 - center.2)^2 → p = point) →
  equation :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l698_69818


namespace NUMINAMATH_CALUDE_sunflower_germination_rate_l698_69831

theorem sunflower_germination_rate 
  (daisy_seeds : ℕ) 
  (sunflower_seeds : ℕ) 
  (daisy_germination_rate : ℚ) 
  (flower_production_rate : ℚ) 
  (total_flowering_plants : ℕ) :
  daisy_seeds = 25 →
  sunflower_seeds = 25 →
  daisy_germination_rate = 3/5 →
  flower_production_rate = 4/5 →
  total_flowering_plants = 28 →
  (daisy_seeds : ℚ) * daisy_germination_rate * flower_production_rate +
  (sunflower_seeds : ℚ) * (4/5) * flower_production_rate = total_flowering_plants →
  (4/5) = 20 / sunflower_seeds :=
by sorry

end NUMINAMATH_CALUDE_sunflower_germination_rate_l698_69831


namespace NUMINAMATH_CALUDE_inequality_solution_l698_69829

theorem inequality_solution (x : ℝ) : 
  3 - 1 / (3 * x + 4) < 5 ↔ x < -11/6 ∨ x > -4/3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l698_69829


namespace NUMINAMATH_CALUDE_sports_equipment_choices_l698_69860

theorem sports_equipment_choices (basketballs volleyballs : ℕ) 
  (hb : basketballs = 5) (hv : volleyballs = 4) : 
  basketballs * volleyballs = 20 := by
  sorry

end NUMINAMATH_CALUDE_sports_equipment_choices_l698_69860


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l698_69814

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_fifth_term
  (a : ℕ → ℤ)
  (h_arithmetic : ArithmeticSequence a)
  (h_21st : a 21 = 26)
  (h_22nd : a 22 = 30) :
  a 5 = -38 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l698_69814


namespace NUMINAMATH_CALUDE_factors_of_36_l698_69837

theorem factors_of_36 : Nat.card (Nat.divisors 36) = 9 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_36_l698_69837


namespace NUMINAMATH_CALUDE_opposite_of_negative_one_half_l698_69809

theorem opposite_of_negative_one_half : 
  ∃ (x : ℚ), x + (-1/2) = 0 ∧ x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_one_half_l698_69809


namespace NUMINAMATH_CALUDE_investment_problem_l698_69866

/-- Calculates the final amount for a simple interest investment -/
def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- Given conditions and proof goal -/
theorem investment_problem (rate : ℝ) :
  simpleInterest 150 rate 6 = 210 →
  simpleInterest 200 rate 3 = 240 := by
  sorry

end NUMINAMATH_CALUDE_investment_problem_l698_69866


namespace NUMINAMATH_CALUDE_hole_depth_proof_l698_69822

/-- The depth of the hole Mat is digging -/
def hole_depth : ℝ := 120

/-- Mat's height in cm -/
def mat_height : ℝ := 90

theorem hole_depth_proof :
  (mat_height = (3/4) * hole_depth) ∧
  (hole_depth - mat_height = mat_height - (1/2) * hole_depth) :=
by sorry

end NUMINAMATH_CALUDE_hole_depth_proof_l698_69822


namespace NUMINAMATH_CALUDE_vector_expression_l698_69807

-- Define vectors a, b, and c
def a : ℝ × ℝ := (1, 1)
def b : ℝ × ℝ := (1, -1)
def c : ℝ × ℝ := (-1, -2)

-- Theorem statement
theorem vector_expression :
  c = (-3/2 : ℝ) • a + (1/2 : ℝ) • b := by sorry

end NUMINAMATH_CALUDE_vector_expression_l698_69807


namespace NUMINAMATH_CALUDE_circle_radius_with_area_four_l698_69892

theorem circle_radius_with_area_four (r : ℝ) :
  r > 0 → π * r^2 = 4 → r = 2 / Real.sqrt π := by sorry

end NUMINAMATH_CALUDE_circle_radius_with_area_four_l698_69892


namespace NUMINAMATH_CALUDE_total_sum_calculation_l698_69865

theorem total_sum_calculation (maggie_share : ℚ) (total_sum : ℚ) : 
  maggie_share = 7500 → 
  maggie_share = (1/8 : ℚ) * total_sum → 
  total_sum = 60000 := by sorry

end NUMINAMATH_CALUDE_total_sum_calculation_l698_69865


namespace NUMINAMATH_CALUDE_kekai_sold_five_shirts_l698_69863

/-- The number of shirts Kekai sold -/
def num_shirts : ℕ := sorry

/-- The number of pants Kekai sold -/
def num_pants : ℕ := 5

/-- The price of each shirt in dollars -/
def shirt_price : ℕ := 1

/-- The price of each pair of pants in dollars -/
def pants_price : ℕ := 3

/-- The amount of money Kekai has left after giving half to his parents -/
def money_left : ℕ := 10

/-- Theorem stating that Kekai sold 5 shirts -/
theorem kekai_sold_five_shirts :
  num_shirts = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_kekai_sold_five_shirts_l698_69863


namespace NUMINAMATH_CALUDE_min_value_of_f_l698_69861

theorem min_value_of_f (x : ℝ) : 
  ∃ (m : ℝ), (∀ y : ℝ, 2 * (Real.cos y)^2 + Real.sin y ≥ m) ∧ 
  (∃ z : ℝ, 2 * (Real.cos z)^2 + Real.sin z = m) ∧ 
  m = -1 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l698_69861


namespace NUMINAMATH_CALUDE_complex_magnitude_power_eight_l698_69857

theorem complex_magnitude_power_eight :
  Complex.abs ((5 / 13 : ℂ) + (12 / 13 : ℂ) * Complex.I) ^ 8 = 1 := by sorry

end NUMINAMATH_CALUDE_complex_magnitude_power_eight_l698_69857


namespace NUMINAMATH_CALUDE_orchid_bushes_total_l698_69886

theorem orchid_bushes_total (current : ℕ) (today : ℕ) (tomorrow : ℕ) :
  current = 47 → today = 37 → tomorrow = 25 →
  current + today + tomorrow = 109 := by
  sorry

end NUMINAMATH_CALUDE_orchid_bushes_total_l698_69886


namespace NUMINAMATH_CALUDE_seats_filled_percentage_l698_69851

/-- The percentage of filled seats in a public show -/
def percentage_filled (total_seats vacant_seats : ℕ) : ℚ :=
  (total_seats - vacant_seats : ℚ) / total_seats * 100

/-- Theorem stating that the percentage of filled seats is 62% -/
theorem seats_filled_percentage (total_seats vacant_seats : ℕ)
  (h1 : total_seats = 600)
  (h2 : vacant_seats = 228) :
  percentage_filled total_seats vacant_seats = 62 := by
  sorry

end NUMINAMATH_CALUDE_seats_filled_percentage_l698_69851


namespace NUMINAMATH_CALUDE_equation_solution_l698_69877

theorem equation_solution :
  ∃ y : ℚ, (5 * y - 2) / (6 * y - 6) = 3 / 4 ↔ y = -5 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l698_69877


namespace NUMINAMATH_CALUDE_composite_has_at_least_three_divisors_l698_69846

-- Define what it means for a number to be composite
def IsComposite (n : ℕ) : Prop :=
  ∃ k : ℕ, 1 < k ∧ k < n ∧ k ∣ n

-- Define the number of divisors function
def NumDivisors (n : ℕ) : ℕ :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

-- Theorem statement
theorem composite_has_at_least_three_divisors (n : ℕ) (h : IsComposite n) :
  NumDivisors n ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_composite_has_at_least_three_divisors_l698_69846


namespace NUMINAMATH_CALUDE_existence_uniqueness_midpoint_l698_69817

/-- Polygonal distance between two points -/
def polygonal_distance (A B : ℝ × ℝ) : ℝ :=
  |A.1 - B.1| + |A.2 - B.2|

/-- Theorem: Existence and uniqueness of point C satisfying given conditions -/
theorem existence_uniqueness_midpoint (A B : ℝ × ℝ) (h : A ≠ B) :
  ∃! C : ℝ × ℝ, 
    polygonal_distance A C + polygonal_distance C B = polygonal_distance A B ∧
    polygonal_distance A C = polygonal_distance C B ∧
    C = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) := by
  sorry

#check existence_uniqueness_midpoint

end NUMINAMATH_CALUDE_existence_uniqueness_midpoint_l698_69817


namespace NUMINAMATH_CALUDE_smallest_two_three_digit_multiples_sum_l698_69821

/-- The smallest positive two-digit number -/
def smallest_two_digit : ℕ := 10

/-- The smallest positive three-digit number -/
def smallest_three_digit : ℕ := 100

/-- The smallest positive two-digit multiple of 5 -/
def c : ℕ := smallest_two_digit

/-- The smallest positive three-digit multiple of 7 -/
def d : ℕ := 
  (smallest_three_digit + 7 - 1) / 7 * 7

theorem smallest_two_three_digit_multiples_sum :
  c + d = 115 := by sorry

end NUMINAMATH_CALUDE_smallest_two_three_digit_multiples_sum_l698_69821


namespace NUMINAMATH_CALUDE_mrs_hilt_marbles_l698_69873

/-- The number of marbles Mrs. Hilt lost -/
def marbles_lost : ℕ := 15

/-- The number of marbles Mrs. Hilt has left -/
def marbles_left : ℕ := 23

/-- The initial number of marbles Mrs. Hilt had -/
def initial_marbles : ℕ := marbles_lost + marbles_left

theorem mrs_hilt_marbles : initial_marbles = 38 := by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_marbles_l698_69873


namespace NUMINAMATH_CALUDE_mans_age_twice_sons_l698_69849

/-- Proves that the number of years until a man's age is twice his son's age is 2 -/
theorem mans_age_twice_sons (
  man_age_difference : ℕ → ℕ → ℕ)
  (son_current_age : ℕ)
  (h1 : man_age_difference son_current_age son_current_age = 34)
  (h2 : son_current_age = 32)
  : ∃ (years : ℕ), years = 2 ∧
    man_age_difference (son_current_age + years) (son_current_age + years) + years =
    2 * (son_current_age + years) :=
by sorry

end NUMINAMATH_CALUDE_mans_age_twice_sons_l698_69849


namespace NUMINAMATH_CALUDE_kenneths_earnings_l698_69842

theorem kenneths_earnings (spent_percentage : ℝ) (remaining_amount : ℝ) (total_earnings : ℝ) : 
  spent_percentage = 10 →
  remaining_amount = 405 →
  (100 - spent_percentage) / 100 * total_earnings = remaining_amount →
  total_earnings = 450 := by
sorry

end NUMINAMATH_CALUDE_kenneths_earnings_l698_69842


namespace NUMINAMATH_CALUDE_marbles_after_2000_steps_l698_69834

/-- Represents the state of baskets with marbles -/
def BasketState := List Nat

/-- Converts a natural number to its base-6 representation -/
def toBase6 (n : Nat) : List Nat :=
  sorry

/-- Sums the digits in a list of natural numbers -/
def sumDigits (digits : List Nat) : Nat :=
  sorry

/-- Simulates the marble placement process for a given number of steps -/
def simulateMarblePlacement (steps : Nat) : BasketState :=
  sorry

/-- Counts the total number of marbles in a given basket state -/
def countMarbles (state : BasketState) : Nat :=
  sorry

/-- Theorem stating that the number of marbles after 2000 steps
    is equal to the sum of digits in the base-6 representation of 2000 -/
theorem marbles_after_2000_steps :
  countMarbles (simulateMarblePlacement 2000) = sumDigits (toBase6 2000) :=
by sorry

end NUMINAMATH_CALUDE_marbles_after_2000_steps_l698_69834


namespace NUMINAMATH_CALUDE_geometric_sequence_middle_term_l698_69896

theorem geometric_sequence_middle_term (y : ℝ) :
  (3^2 : ℝ) < y ∧ y < (3^4 : ℝ) ∧ 
  (y / (3^2 : ℝ)) = ((3^4 : ℝ) / y) →
  y = 27 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_middle_term_l698_69896


namespace NUMINAMATH_CALUDE_valid_n_values_l698_69876

def is_valid_n (f : ℤ → ℤ) (n : ℤ) : Prop :=
  f 1 = -1 ∧ f 4 = 2 ∧ f 8 = 34 ∧ f n = n^2 - 4*n - 18

theorem valid_n_values (f : ℤ → ℤ) (n : ℤ) 
  (h : is_valid_n f n) : n = 3 ∨ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_valid_n_values_l698_69876


namespace NUMINAMATH_CALUDE_five_line_regions_l698_69847

/-- Number of regions formed by n lines in a plane -/
def num_regions (n : ℕ) : ℕ := 1 + n + n.choose 2

/-- A configuration of lines in a plane -/
structure LineConfiguration where
  num_lines : ℕ
  not_parallel : Prop
  not_concurrent : Prop

/-- The number of regions formed by a line configuration -/
def regions_formed (config : LineConfiguration) : ℕ := num_regions config.num_lines

theorem five_line_regions (config : LineConfiguration) :
  config.num_lines = 5 →
  config.not_parallel →
  config.not_concurrent →
  regions_formed config = 16 := by
  sorry

end NUMINAMATH_CALUDE_five_line_regions_l698_69847


namespace NUMINAMATH_CALUDE_arithmetic_sequence_product_sign_change_l698_69878

/-- An arithmetic sequence with its partial sums -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Partial sums
  sum_formula : ∀ n, S n = (n * (a 1 + a n)) / 2
  arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1

/-- The main theorem -/
theorem arithmetic_sequence_product_sign_change 
  (seq : ArithmeticSequence) 
  (h1 : seq.S 7 > seq.S 8) 
  (h2 : seq.S 8 > seq.S 6) :
  (∀ k < 14, seq.S k * seq.S (k + 1) ≥ 0) ∧ 
  (seq.S 14 * seq.S 15 < 0) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_product_sign_change_l698_69878


namespace NUMINAMATH_CALUDE_james_carrot_sticks_l698_69800

theorem james_carrot_sticks (before after total : ℕ) : 
  before = 22 → after = 15 → total = before + after → total = 37 := by sorry

end NUMINAMATH_CALUDE_james_carrot_sticks_l698_69800


namespace NUMINAMATH_CALUDE_solve_for_x_l698_69839

theorem solve_for_x (x y : ℝ) (h1 : x + 3 * y = 33) (h2 : y = 10) : x = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_x_l698_69839


namespace NUMINAMATH_CALUDE_irrational_numbers_have_square_roots_l698_69855

theorem irrational_numbers_have_square_roots : ∃ (x : ℝ), Irrational x ∧ ∃ (y : ℝ), y^2 = x := by
  sorry

end NUMINAMATH_CALUDE_irrational_numbers_have_square_roots_l698_69855


namespace NUMINAMATH_CALUDE_prob_one_head_in_three_flips_l698_69862

/-- The probability of getting exactly one head in three flips of a fair coin is 3/8 -/
theorem prob_one_head_in_three_flips :
  let p : ℝ := 1/2  -- probability of heads for a fair coin
  let n : ℕ := 3    -- number of flips
  let k : ℕ := 1    -- number of heads we want
  (n.choose k) * p^k * (1-p)^(n-k) = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_prob_one_head_in_three_flips_l698_69862


namespace NUMINAMATH_CALUDE_total_jeans_purchased_l698_69867

/-- Represents the number of pairs of Fox jeans purchased -/
def fox_jeans : ℕ := 3

/-- Represents the number of pairs of Pony jeans purchased -/
def pony_jeans : ℕ := 2

/-- Regular price of Fox jeans in dollars -/
def fox_price : ℚ := 15

/-- Regular price of Pony jeans in dollars -/
def pony_price : ℚ := 20

/-- Total discount in dollars -/
def total_discount : ℚ := 9

/-- Sum of discount rates as a percentage -/
def sum_discount_rates : ℚ := 22

/-- Discount rate on Pony jeans as a percentage -/
def pony_discount_rate : ℚ := 18.000000000000014

/-- Theorem stating the total number of jeans purchased -/
theorem total_jeans_purchased : fox_jeans + pony_jeans = 5 := by
  sorry

end NUMINAMATH_CALUDE_total_jeans_purchased_l698_69867


namespace NUMINAMATH_CALUDE_tan_45_degrees_l698_69895

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_45_degrees_l698_69895


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l698_69853

theorem imaginary_part_of_z (z : ℂ) (h : (2 + Complex.I) * z = 2 - 4 * Complex.I) : 
  Complex.im z = -2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l698_69853


namespace NUMINAMATH_CALUDE_science_club_committee_formation_l698_69888

theorem science_club_committee_formation (total_members : ℕ) 
                                         (new_members : ℕ) 
                                         (committee_size : ℕ) 
                                         (h1 : total_members = 20) 
                                         (h2 : new_members = 10) 
                                         (h3 : committee_size = 4) :
  (Nat.choose total_members committee_size) - 
  (Nat.choose new_members committee_size) = 4635 :=
sorry

end NUMINAMATH_CALUDE_science_club_committee_formation_l698_69888


namespace NUMINAMATH_CALUDE_unique_solution_l698_69843

theorem unique_solution (x y z : ℝ) 
  (hx : x > 2) (hy : y > 2) (hz : z > 2)
  (heq : ((x + 3)^2) / (y + z - 3) + ((y + 5)^2) / (z + x - 5) + ((z + 7)^2) / (x + y - 7) = 45) :
  x = 7 ∧ y = 5 ∧ z = 3 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_l698_69843


namespace NUMINAMATH_CALUDE_exact_three_green_probability_l698_69823

def total_marbles : ℕ := 15
def green_marbles : ℕ := 8
def purple_marbles : ℕ := 7
def num_trials : ℕ := 7
def num_green_selected : ℕ := 3

def probability_green : ℚ := green_marbles / total_marbles
def probability_purple : ℚ := purple_marbles / total_marbles

theorem exact_three_green_probability :
  (Nat.choose num_trials num_green_selected : ℚ) *
  (probability_green ^ num_green_selected) *
  (probability_purple ^ (num_trials - num_green_selected)) =
  860818 / 3421867 := by sorry

end NUMINAMATH_CALUDE_exact_three_green_probability_l698_69823


namespace NUMINAMATH_CALUDE_p_and_not_q_is_true_l698_69841

-- Define proposition p
def p : Prop := ∃ x : ℝ, x - 2 > Real.log x

-- Define proposition q
def q : Prop := ∀ x : ℝ, x^2 > 0

-- Theorem to prove
theorem p_and_not_q_is_true : p ∧ ¬q := by
  sorry

end NUMINAMATH_CALUDE_p_and_not_q_is_true_l698_69841


namespace NUMINAMATH_CALUDE_village_assistants_selection_l698_69894

-- Define the total number of college graduates
def total_graduates : ℕ := 10

-- Define the number of people to be selected
def selection_size : ℕ := 3

-- Define a function to calculate the number of ways to select k items from n items
def choose (n k : ℕ) : ℕ := Nat.choose n k

-- Theorem statement
theorem village_assistants_selection :
  choose (total_graduates - 1) selection_size -
  choose (total_graduates - 3) selection_size = 49 := by
  sorry

end NUMINAMATH_CALUDE_village_assistants_selection_l698_69894


namespace NUMINAMATH_CALUDE_sheets_per_student_l698_69836

theorem sheets_per_student (num_classes : ℕ) (students_per_class : ℕ) (total_sheets : ℕ) :
  num_classes = 4 →
  students_per_class = 20 →
  total_sheets = 400 →
  total_sheets / (num_classes * students_per_class) = 5 := by
  sorry

end NUMINAMATH_CALUDE_sheets_per_student_l698_69836


namespace NUMINAMATH_CALUDE_A_P_parity_uniformity_l698_69848

-- Define the set A_P
def A_P : Set ℤ := sorry

-- Define a property for elements of A_P related to positioning in a function or polynomial
def has_positioning_property (n : ℤ) : Prop := sorry

-- Axiom: All elements in A_P have the positioning property
axiom A_P_property : ∀ n ∈ A_P, has_positioning_property n

-- Define parity
def same_parity (a b : ℤ) : Prop := a % 2 = b % 2

-- Theorem: The smallest and largest elements of A_P have the same parity
theorem A_P_parity_uniformity :
  ∀ (min max : ℤ), min ∈ A_P → max ∈ A_P →
  (∀ x ∈ A_P, min ≤ x ∧ x ≤ max) →
  same_parity min max :=
sorry

end NUMINAMATH_CALUDE_A_P_parity_uniformity_l698_69848


namespace NUMINAMATH_CALUDE_trigonometric_identity_l698_69891

theorem trigonometric_identity (α : Real) (h : 3 * Real.sin α + Real.cos α = 0) :
  1 / (Real.cos α ^ 2 + Real.sin (2 * α)) = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l698_69891


namespace NUMINAMATH_CALUDE_sprained_wrist_frosting_time_l698_69819

/-- The time it takes Ann to frost a cake with her sprained wrist -/
def sprained_wrist_time : ℝ := 8

/-- The normal time it takes Ann to frost a cake -/
def normal_time : ℝ := 5

/-- The additional time it takes to frost 10 cakes with a sprained wrist -/
def additional_time : ℝ := 30

theorem sprained_wrist_frosting_time :
  sprained_wrist_time = (10 * normal_time + additional_time) / 10 := by
  sorry

end NUMINAMATH_CALUDE_sprained_wrist_frosting_time_l698_69819


namespace NUMINAMATH_CALUDE_two_times_first_exceeds_three_times_second_l698_69874

theorem two_times_first_exceeds_three_times_second (x y : ℝ) : 
  x + y = 10 → x = 7 → y = 3 → 2 * x - 3 * y = 5 := by
  sorry

end NUMINAMATH_CALUDE_two_times_first_exceeds_three_times_second_l698_69874


namespace NUMINAMATH_CALUDE_expand_cube_105_plus_1_l698_69845

theorem expand_cube_105_plus_1 : 105^3 + 3*(105^2) + 3*105 + 1 = 11856 := by
  sorry

end NUMINAMATH_CALUDE_expand_cube_105_plus_1_l698_69845


namespace NUMINAMATH_CALUDE_complex_number_calculation_l698_69844

/-- Given the complex number i where i^2 = -1, prove that (1+i)(1-i)+(-1+i) = 1+i -/
theorem complex_number_calculation : ∀ i : ℂ, i^2 = -1 → (1+i)*(1-i)+(-1+i) = 1+i := by
  sorry

end NUMINAMATH_CALUDE_complex_number_calculation_l698_69844


namespace NUMINAMATH_CALUDE_movie_ticket_change_l698_69806

/-- Represents the movie ticket formats --/
inductive TicketFormat
  | Regular
  | ThreeD
  | IMAX

/-- Returns the price of a ticket based on its format --/
def ticketPrice (format : TicketFormat) : ℝ :=
  match format with
  | TicketFormat.Regular => 8
  | TicketFormat.ThreeD => 12
  | TicketFormat.IMAX => 15

/-- Calculates the discounted price of a ticket --/
def discountedPrice (price : ℝ) (discount : ℝ) : ℝ :=
  price * (1 - discount)

theorem movie_ticket_change : 
  let format := TicketFormat.ThreeD
  let fullPrice := ticketPrice format
  let discountPercent := 0.25
  let discountedTicket := discountedPrice fullPrice discountPercent
  let totalCost := fullPrice + discountedTicket
  let moneyBrought := 25
  moneyBrought - totalCost = 4 := by sorry


end NUMINAMATH_CALUDE_movie_ticket_change_l698_69806


namespace NUMINAMATH_CALUDE_line_relationship_exclusive_line_relationship_unique_l698_69859

-- Define a line in 3D space
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

-- Define the relationship between two lines
inductive LineRelationship
  | Parallel
  | Skew
  | Intersecting

-- Define a function to determine the relationship between two lines
def determineRelationship (l1 l2 : Line3D) : LineRelationship :=
  sorry

-- Theorem: Two lines must have exactly one of the three relationships
theorem line_relationship_exclusive (l1 l2 : Line3D) :
  (determineRelationship l1 l2 = LineRelationship.Parallel) ∨
  (determineRelationship l1 l2 = LineRelationship.Skew) ∨
  (determineRelationship l1 l2 = LineRelationship.Intersecting) :=
  sorry

-- Theorem: The relationship between two lines is unique
theorem line_relationship_unique (l1 l2 : Line3D) :
  ¬((determineRelationship l1 l2 = LineRelationship.Parallel) ∧
    (determineRelationship l1 l2 = LineRelationship.Skew)) ∧
  ¬((determineRelationship l1 l2 = LineRelationship.Parallel) ∧
    (determineRelationship l1 l2 = LineRelationship.Intersecting)) ∧
  ¬((determineRelationship l1 l2 = LineRelationship.Skew) ∧
    (determineRelationship l1 l2 = LineRelationship.Intersecting)) :=
  sorry

end NUMINAMATH_CALUDE_line_relationship_exclusive_line_relationship_unique_l698_69859


namespace NUMINAMATH_CALUDE_distance_to_hole_is_250_l698_69804

/-- The distance from the starting tee to the hole in a golf game --/
def distance_to_hole (first_hit second_hit beyond_hole : ℕ) : ℕ :=
  first_hit + second_hit - beyond_hole

/-- Theorem stating the distance to the hole given the conditions in the problem --/
theorem distance_to_hole_is_250 :
  let first_hit := 180
  let second_hit := first_hit / 2
  let beyond_hole := 20
  distance_to_hole first_hit second_hit beyond_hole = 250 := by
  sorry

#eval distance_to_hole 180 90 20

end NUMINAMATH_CALUDE_distance_to_hole_is_250_l698_69804


namespace NUMINAMATH_CALUDE_prob_same_heads_value_l698_69801

-- Define the probability of heads for a fair coin
def fair_coin_prob : ℚ := 1/2

-- Define the probability of heads for the biased coin
def biased_coin_prob : ℚ := 5/8

-- Define the number of fair coins
def num_fair_coins : ℕ := 3

-- Define the number of biased coins
def num_biased_coins : ℕ := 1

-- Define the total number of coins
def total_coins : ℕ := num_fair_coins + num_biased_coins

-- Define the function to calculate the probability of getting the same number of heads
def prob_same_heads : ℚ := sorry

-- Theorem statement
theorem prob_same_heads_value : prob_same_heads = 77/225 := by sorry

end NUMINAMATH_CALUDE_prob_same_heads_value_l698_69801


namespace NUMINAMATH_CALUDE_goods_train_length_l698_69883

/-- Calculates the length of a goods train given the speeds of two trains
    traveling in opposite directions and the time taken for the goods train
    to pass a stationary observer in the other train. -/
theorem goods_train_length
  (speed_train : ℝ)
  (speed_goods : ℝ)
  (pass_time : ℝ)
  (h1 : speed_train = 15)
  (h2 : speed_goods = 97)
  (h3 : pass_time = 9)
  : ∃ (length : ℝ), abs (length - 279.99) < 0.01 :=
by
  sorry

end NUMINAMATH_CALUDE_goods_train_length_l698_69883


namespace NUMINAMATH_CALUDE_consecutive_integers_median_l698_69825

theorem consecutive_integers_median (n : ℕ) (sum : ℕ) (h1 : n = 81) (h2 : sum = 9^5) :
  let median := sum / n
  median = 729 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_median_l698_69825


namespace NUMINAMATH_CALUDE_gcd_divisibility_and_multiple_l698_69813

theorem gcd_divisibility_and_multiple (a b n : ℕ) (h : a ≠ 0) :
  let d := Nat.gcd a b
  (n ∣ a ∧ n ∣ b ↔ n ∣ d) ∧
  ∀ c : ℕ, c > 0 → Nat.gcd (a * c) (b * c) = c * Nat.gcd a b :=
by sorry

end NUMINAMATH_CALUDE_gcd_divisibility_and_multiple_l698_69813


namespace NUMINAMATH_CALUDE_splash_width_is_seven_l698_69850

/-- The width of a splash made by a pebble in meters -/
def pebble_splash : ℚ := 1/4

/-- The width of a splash made by a rock in meters -/
def rock_splash : ℚ := 1/2

/-- The width of a splash made by a boulder in meters -/
def boulder_splash : ℚ := 2

/-- The number of pebbles thrown -/
def pebbles_thrown : ℕ := 6

/-- The number of rocks thrown -/
def rocks_thrown : ℕ := 3

/-- The number of boulders thrown -/
def boulders_thrown : ℕ := 2

/-- The total width of splashes made by TreQuan's throws -/
def total_splash_width : ℚ := 
  pebble_splash * pebbles_thrown + 
  rock_splash * rocks_thrown + 
  boulder_splash * boulders_thrown

theorem splash_width_is_seven : total_splash_width = 7 := by
  sorry

end NUMINAMATH_CALUDE_splash_width_is_seven_l698_69850


namespace NUMINAMATH_CALUDE_addition_puzzle_l698_69871

theorem addition_puzzle (x y : ℕ) : 
  x ≠ y →
  x < 10 →
  y < 10 →
  307 + 700 + x = 1010 →
  y - x = 7 :=
by sorry

end NUMINAMATH_CALUDE_addition_puzzle_l698_69871


namespace NUMINAMATH_CALUDE_work_left_after_collaboration_l698_69827

theorem work_left_after_collaboration (days_a days_b collab_days : ℕ) 
  (ha : days_a = 15) (hb : days_b = 20) (hc : collab_days = 3) : 
  1 - (collab_days * (1 / days_a + 1 / days_b)) = 13 / 20 := by
  sorry

end NUMINAMATH_CALUDE_work_left_after_collaboration_l698_69827


namespace NUMINAMATH_CALUDE_inequality_of_distinct_positives_l698_69882

theorem inequality_of_distinct_positives (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) : 
  (b + c - a) / a + (a + c - b) / b + (a + b - c) / c > 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_of_distinct_positives_l698_69882


namespace NUMINAMATH_CALUDE_specific_tetrahedron_volume_l698_69890

/-- A tetrahedron ABCD with specific properties -/
structure Tetrahedron where
  /-- The dihedral angle between faces ABC and BCD in radians -/
  dihedral_angle : ℝ
  /-- The area of triangle ABC -/
  area_ABC : ℝ
  /-- The area of triangle BCD -/
  area_BCD : ℝ
  /-- The length of edge BC -/
  length_BC : ℝ

/-- The volume of the tetrahedron -/
def volume (t : Tetrahedron) : ℝ :=
  sorry

/-- Theorem stating the volume of the specific tetrahedron -/
theorem specific_tetrahedron_volume :
  ∃ t : Tetrahedron,
    t.dihedral_angle = 30 * (π / 180) ∧
    t.area_ABC = 120 ∧
    t.area_BCD = 80 ∧
    t.length_BC = 10 ∧
    volume t = 320 :=
  sorry

end NUMINAMATH_CALUDE_specific_tetrahedron_volume_l698_69890


namespace NUMINAMATH_CALUDE_range_of_k_for_trigonometric_equation_l698_69838

theorem range_of_k_for_trigonometric_equation :
  ∀ k : ℝ, (∃ x : ℝ, x ∈ Set.Icc 0 (π/2) ∧ 
    Real.sqrt 3 * Real.sin (2*x) + Real.cos (2*x) = k + 1) ↔ 
  k ∈ Set.Icc (-2) 1 := by
sorry

end NUMINAMATH_CALUDE_range_of_k_for_trigonometric_equation_l698_69838


namespace NUMINAMATH_CALUDE_quadratic_discriminant_l698_69898

/-- The discriminant of a quadratic equation ax^2 + bx + c -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

theorem quadratic_discriminant :
  discriminant 5 (-6) 1 = 16 := by sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_l698_69898


namespace NUMINAMATH_CALUDE_factor_expression_l698_69872

theorem factor_expression (x : ℝ) : 4*x*(x-5) + 6*(x-5) = (4*x+6)*(x-5) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l698_69872


namespace NUMINAMATH_CALUDE_speed_difference_l698_69820

/-- Two cars traveling in opposite directions -/
structure TwoCars where
  fast_speed : ℝ
  slow_speed : ℝ
  time : ℝ
  distance : ℝ

/-- The conditions of the problem -/
def problem_conditions (cars : TwoCars) : Prop :=
  cars.fast_speed = 55 ∧
  cars.time = 5 ∧
  cars.distance = 500 ∧
  cars.distance = (cars.fast_speed + cars.slow_speed) * cars.time

/-- The theorem to prove -/
theorem speed_difference (cars : TwoCars) 
  (h : problem_conditions cars) : 
  cars.fast_speed - cars.slow_speed = 10 := by
  sorry


end NUMINAMATH_CALUDE_speed_difference_l698_69820


namespace NUMINAMATH_CALUDE_new_average_age_with_teacher_l698_69868

theorem new_average_age_with_teacher 
  (num_students : ℕ) 
  (student_avg_age : ℝ) 
  (teacher_age : ℕ) 
  (h1 : num_students = 50) 
  (h2 : student_avg_age = 14) 
  (h3 : teacher_age = 65) : 
  (num_students * student_avg_age + teacher_age) / (num_students + 1) = 15 := by
sorry

end NUMINAMATH_CALUDE_new_average_age_with_teacher_l698_69868


namespace NUMINAMATH_CALUDE_two_hour_charge_is_174_l698_69869

/-- Represents the pricing model for therapy sessions -/
structure TherapyPricing where
  first_hour : ℕ
  additional_hour : ℕ
  first_hour_premium : first_hour = additional_hour + 40

/-- Calculates the total charge for a given number of hours -/
def total_charge (pricing : TherapyPricing) (hours : ℕ) : ℕ :=
  pricing.first_hour + (hours - 1) * pricing.additional_hour

/-- Theorem stating the correct charge for 2 hours given the conditions -/
theorem two_hour_charge_is_174 (pricing : TherapyPricing) 
  (h1 : total_charge pricing 5 = 375) : 
  total_charge pricing 2 = 174 := by
  sorry

end NUMINAMATH_CALUDE_two_hour_charge_is_174_l698_69869


namespace NUMINAMATH_CALUDE_functional_equation_solution_l698_69889

theorem functional_equation_solution (f : ℤ → ℤ) 
  (h : ∀ x y : ℤ, f (x + y) = f x + f y - 2023) : 
  ∃ c : ℤ, ∀ x : ℤ, f x = c * x + 2023 := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l698_69889


namespace NUMINAMATH_CALUDE_correct_divisor_l698_69840

theorem correct_divisor : ∃ (X : ℕ) (incorrect_divisor correct_divisor : ℕ),
  incorrect_divisor = 87 ∧
  X / incorrect_divisor = 24 ∧
  X / correct_divisor = 58 ∧
  correct_divisor = 36 := by
  sorry

end NUMINAMATH_CALUDE_correct_divisor_l698_69840


namespace NUMINAMATH_CALUDE_faucet_filling_time_l698_69811

/-- Given that four faucets can fill a 120-gallon tub in 5 minutes,
    prove that two faucets can fill a 60-gallon tub in 5 minutes. -/
theorem faucet_filling_time 
  (tub_capacity : ℝ) 
  (filling_time : ℝ) 
  (faucet_count : ℕ) :
  tub_capacity = 120 ∧ 
  filling_time = 5 ∧ 
  faucet_count = 4 →
  ∃ (new_tub_capacity : ℝ) (new_faucet_count : ℕ),
    new_tub_capacity = 60 ∧
    new_faucet_count = 2 ∧
    (new_tub_capacity / new_faucet_count) / (tub_capacity / faucet_count) * filling_time = 5 :=
by sorry

end NUMINAMATH_CALUDE_faucet_filling_time_l698_69811


namespace NUMINAMATH_CALUDE_functional_equation_solution_l698_69856

/-- A function satisfying the given functional equation -/
def FunctionalEquation (g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, g (x * y) + x^2 = x * g y + g x

theorem functional_equation_solution (g : ℝ → ℝ) 
  (h1 : FunctionalEquation g) 
  (h2 : g (-1) = 7) : 
  g (-1001) = 6006013 := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l698_69856


namespace NUMINAMATH_CALUDE_parallel_vectors_k_value_l698_69854

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), t ≠ 0 ∧ a.1 * t = b.1 ∧ a.2 * t = b.2

/-- The problem statement -/
theorem parallel_vectors_k_value :
  let a : ℝ × ℝ := (4, 2)
  let b : ℝ × ℝ := (6, k)
  are_parallel a b → k = 3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_k_value_l698_69854


namespace NUMINAMATH_CALUDE_parabola_expression_l698_69815

theorem parabola_expression (f : ℝ → ℝ) (h1 : f (-3) = 0) (h2 : f 1 = 0) (h3 : f 0 = 2) :
  ∀ x, f x = -2/3 * x^2 - 4/3 * x + 2 :=
by sorry

end NUMINAMATH_CALUDE_parabola_expression_l698_69815


namespace NUMINAMATH_CALUDE_ratio_of_fractions_l698_69808

theorem ratio_of_fractions (P Q : ℤ) : 
  (∀ x : ℝ, x ≠ -7 ∧ x ≠ 0 ∧ x ≠ 6 → 
    P / (x + 7) + Q / (x^2 - 6*x) = (x^2 - x + 15) / (x^3 + x^2 - 42*x)) →
  Q / P = 7 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_fractions_l698_69808


namespace NUMINAMATH_CALUDE_max_diagonal_length_l698_69899

-- Define the quadrilateral PQRS
structure Quadrilateral :=
  (P Q R S : ℝ × ℝ)

-- Define the distance function
def distance (p q : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem max_diagonal_length (PQRS : Quadrilateral) : 
  distance PQRS.P PQRS.Q = 7 →
  distance PQRS.Q PQRS.R = 13 →
  distance PQRS.R PQRS.S = 7 →
  distance PQRS.S PQRS.P = 10 →
  ∃ (pr : ℕ), pr ≤ 19 ∧ 
    distance PQRS.P PQRS.R = pr ∧
    ∀ (x : ℕ), distance PQRS.P PQRS.R = x → x ≤ pr :=
by sorry

end NUMINAMATH_CALUDE_max_diagonal_length_l698_69899


namespace NUMINAMATH_CALUDE_mary_balloon_count_l698_69828

/-- Given that Nancy has 7 black balloons and Mary has 4 times more black balloons than Nancy,
    prove that Mary has 28 black balloons. -/
theorem mary_balloon_count :
  ∀ (nancy_balloons mary_balloons : ℕ),
    nancy_balloons = 7 →
    mary_balloons = 4 * nancy_balloons →
    mary_balloons = 28 :=
by sorry

end NUMINAMATH_CALUDE_mary_balloon_count_l698_69828


namespace NUMINAMATH_CALUDE_numbers_below_nine_and_twenty_four_are_composite_l698_69852

def below_nine (k : ℕ) : ℕ := 4 * k^2 + 5 * k + 1

def below_twenty_four (k : ℕ) : ℕ := 4 * k^2 + 5 * k

theorem numbers_below_nine_and_twenty_four_are_composite :
  (∀ k : ℕ, k ≥ 1 → ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ below_nine k = a * b) ∧
  (∀ k : ℕ, k ≥ 2 → ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ below_twenty_four k = a * b) :=
sorry

end NUMINAMATH_CALUDE_numbers_below_nine_and_twenty_four_are_composite_l698_69852


namespace NUMINAMATH_CALUDE_triangle_tangent_difference_l698_69885

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a² = b² + 4bc sin A and tan A · tan B = 2, then tan B - tan A = -8 -/
theorem triangle_tangent_difference (a b c : ℝ) (A B C : ℝ) : 
  a > 0 → b > 0 → c > 0 → 
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  a^2 = b^2 + 4*b*c*(Real.sin A) →
  Real.tan A * Real.tan B = 2 →
  Real.tan B - Real.tan A = -8 := by
sorry

end NUMINAMATH_CALUDE_triangle_tangent_difference_l698_69885


namespace NUMINAMATH_CALUDE_helga_work_days_l698_69805

/-- Represents Helga's work schedule and output --/
structure HelgaWork where
  articles_per_half_hour : ℕ := 5
  usual_hours_per_day : ℕ := 4
  extra_hours_thursday : ℕ := 2
  extra_hours_friday : ℕ := 3
  total_articles_week : ℕ := 250

/-- Calculates the number of days Helga usually works in a week --/
def usual_work_days (hw : HelgaWork) : ℕ :=
  let articles_per_hour := hw.articles_per_half_hour * 2
  let articles_per_day := articles_per_hour * hw.usual_hours_per_day
  let extra_articles := articles_per_hour * (hw.extra_hours_thursday + hw.extra_hours_friday)
  let usual_articles := hw.total_articles_week - extra_articles
  usual_articles / articles_per_day

theorem helga_work_days (hw : HelgaWork) : usual_work_days hw = 5 := by
  sorry

end NUMINAMATH_CALUDE_helga_work_days_l698_69805


namespace NUMINAMATH_CALUDE_subtraction_of_decimals_l698_69875

theorem subtraction_of_decimals : 5.75 - 1.46 = 4.29 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_decimals_l698_69875


namespace NUMINAMATH_CALUDE_solve_cottage_problem_l698_69897

def cottage_problem (hourly_rate : ℚ) (jack_paid : ℚ) (jill_paid : ℚ) : Prop :=
  let total_paid := jack_paid + jill_paid
  let hours_rented := total_paid / hourly_rate
  hours_rented = 8

theorem solve_cottage_problem :
  cottage_problem 5 20 20 := by sorry

end NUMINAMATH_CALUDE_solve_cottage_problem_l698_69897


namespace NUMINAMATH_CALUDE_multiply_63_37_l698_69824

theorem multiply_63_37 : 63 * 37 = 2331 := by
  sorry

end NUMINAMATH_CALUDE_multiply_63_37_l698_69824


namespace NUMINAMATH_CALUDE_trigonometric_identity_l698_69870

theorem trigonometric_identity (x y : Real) 
  (h : Real.cos (x + y) = 2 / 3) : 
  Real.sin (x - 3 * Real.pi / 10) * Real.cos (y - Real.pi / 5) - 
  Real.sin (x + Real.pi / 5) * Real.cos (y + 3 * Real.pi / 10) = 
  -2 / 3 := by sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l698_69870


namespace NUMINAMATH_CALUDE_hemisphere_surface_area_l698_69893

theorem hemisphere_surface_area (base_area : ℝ) (Q : ℝ) : 
  base_area = 3 →
  Q = (2 * Real.pi * (Real.sqrt (3 / Real.pi))^2) + base_area →
  Q = 9 := by sorry

end NUMINAMATH_CALUDE_hemisphere_surface_area_l698_69893


namespace NUMINAMATH_CALUDE_average_marks_proof_l698_69864

def scores : List ℕ := [76, 65, 82, 67, 55, 89, 74, 63, 78, 71]

theorem average_marks_proof :
  (scores.sum / scores.length : ℚ) = 72 := by
  sorry

end NUMINAMATH_CALUDE_average_marks_proof_l698_69864


namespace NUMINAMATH_CALUDE_certain_number_proof_l698_69826

theorem certain_number_proof : ∃ x : ℝ, x * 9 = 0.45 * 900 ∧ x = 45 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l698_69826


namespace NUMINAMATH_CALUDE_bronze_status_donation_bound_l698_69835

/-- Represents the fundraising status of the school --/
structure FundraisingStatus where
  goal : ℕ
  remaining : ℕ
  bronzeFamilies : ℕ
  silverFamilies : ℕ
  goldFamilies : ℕ

/-- Represents the donation tiers --/
structure DonationTiers where
  bronze : ℕ
  silver : ℕ
  gold : ℕ

/-- The Bronze Status donation is less than or equal to the remaining amount needed --/
theorem bronze_status_donation_bound (status : FundraisingStatus) (tiers : DonationTiers) :
  status.goal = 750 ∧
  status.remaining = 50 ∧
  status.bronzeFamilies = 10 ∧
  status.silverFamilies = 7 ∧
  status.goldFamilies = 1 ∧
  tiers.bronze ≤ tiers.silver ∧
  tiers.silver ≤ tiers.gold →
  tiers.bronze ≤ status.remaining :=
by sorry

end NUMINAMATH_CALUDE_bronze_status_donation_bound_l698_69835


namespace NUMINAMATH_CALUDE_product_of_binary_and_ternary_l698_69880

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

def ternary_to_decimal (t : List ℕ) : ℕ :=
  t.enum.foldl (fun acc (i, digit) => acc + digit * 3^i) 0

theorem product_of_binary_and_ternary :
  let binary := [true, true, false, true]  -- 1011 in binary (least significant bit first)
  let ternary := [2, 0, 1]  -- 102 in ternary (least significant digit first)
  (binary_to_decimal binary) * (ternary_to_decimal ternary) = 121 := by
  sorry

end NUMINAMATH_CALUDE_product_of_binary_and_ternary_l698_69880


namespace NUMINAMATH_CALUDE_exists_same_color_right_triangle_l698_69879

-- Define a color type
inductive Color
| Red
| Blue

-- Define a point type
structure Point where
  x : ℝ
  y : ℝ

-- Define an equilateral triangle
structure EquilateralTriangle where
  a : Point
  b : Point
  c : Point

-- Define a coloring function
def Coloring := Point → Color

-- Define a property for a right-angled triangle
def isRightAngledTriangle (p q r : Point) : Prop := sorry

-- Theorem statement
theorem exists_same_color_right_triangle 
  (triangle : EquilateralTriangle) 
  (coloring : Coloring) : 
  ∃ (p q r : Point), 
    (coloring p = coloring q) ∧ 
    (coloring q = coloring r) ∧ 
    isRightAngledTriangle p q r :=
sorry

end NUMINAMATH_CALUDE_exists_same_color_right_triangle_l698_69879


namespace NUMINAMATH_CALUDE_balloon_count_l698_69858

theorem balloon_count (colors : Nat) (yellow_taken : Nat) : 
  colors = 4 → yellow_taken = 84 → colors * yellow_taken * 2 = 672 := by
  sorry

end NUMINAMATH_CALUDE_balloon_count_l698_69858


namespace NUMINAMATH_CALUDE_total_sum_is_992_l698_69881

/-- Represents the share of money for each person -/
structure Share where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ

/-- The conditions of the money division problem -/
def money_division (s : Share) : Prop :=
  s.b = 0.75 * s.a ∧
  s.c = 0.60 * s.a ∧
  s.d = 0.45 * s.a ∧
  s.e = 0.30 * s.a ∧
  s.e = 96

/-- The theorem stating that the total sum of money is 992 -/
theorem total_sum_is_992 (s : Share) (h : money_division s) : 
  s.a + s.b + s.c + s.d + s.e = 992 := by
  sorry


end NUMINAMATH_CALUDE_total_sum_is_992_l698_69881


namespace NUMINAMATH_CALUDE_expected_worth_is_one_third_l698_69887

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

end NUMINAMATH_CALUDE_expected_worth_is_one_third_l698_69887


namespace NUMINAMATH_CALUDE_orangeade_price_day_two_l698_69816

/-- Represents the price of orangeade per glass on a given day. -/
structure OrangeadePrice where
  day : Nat
  price : ℚ

/-- Represents the amount of ingredients used to make orangeade on a given day. -/
structure OrangeadeIngredients where
  day : Nat
  orange_juice : ℚ
  water : ℚ

/-- Calculates the total volume of orangeade made on a given day. -/
def totalVolume (ingredients : OrangeadeIngredients) : ℚ :=
  ingredients.orange_juice + ingredients.water

/-- Calculates the revenue from selling orangeade on a given day. -/
def revenue (price : OrangeadePrice) (ingredients : OrangeadeIngredients) : ℚ :=
  price.price * totalVolume ingredients

/-- Theorem stating that the price of orangeade on the second day is $0.40 given the conditions. -/
theorem orangeade_price_day_two
  (day1_price : OrangeadePrice)
  (day1_ingredients : OrangeadeIngredients)
  (day2_ingredients : OrangeadeIngredients)
  (h1 : day1_price.day = 1)
  (h2 : day1_price.price = 6/10)
  (h3 : day1_ingredients.day = 1)
  (h4 : day1_ingredients.orange_juice = day1_ingredients.water)
  (h5 : day2_ingredients.day = 2)
  (h6 : day2_ingredients.orange_juice = day1_ingredients.orange_juice)
  (h7 : day2_ingredients.water = 2 * day1_ingredients.water)
  (h8 : revenue day1_price day1_ingredients = revenue { day := 2, price := 4/10 } day2_ingredients) :
  ∃ (day2_price : OrangeadePrice), day2_price.day = 2 ∧ day2_price.price = 4/10 :=
by sorry


end NUMINAMATH_CALUDE_orangeade_price_day_two_l698_69816


namespace NUMINAMATH_CALUDE_sqrt_50_minus_sqrt_48_approx_0_14_l698_69812

theorem sqrt_50_minus_sqrt_48_approx_0_14 (ε : ℝ) (h : ε > 0) :
  ∃ δ > 0, |Real.sqrt 50 - Real.sqrt 48 - 0.14| < δ ∧ δ < ε :=
by sorry

end NUMINAMATH_CALUDE_sqrt_50_minus_sqrt_48_approx_0_14_l698_69812


namespace NUMINAMATH_CALUDE_f_upper_bound_l698_69802

def f_property (f : ℝ → ℝ) : Prop :=
  (∀ x y, 0 ≤ x ∧ 0 ≤ y → f x * f y ≤ y^2 * f (x/2) + x^2 * f (y/2)) ∧
  (∀ x, 0 ≤ x ∧ x ≤ 1 → f x ≤ 2016)

theorem f_upper_bound (f : ℝ → ℝ) (h : f_property f) :
  ∀ x, 0 ≤ x → f x ≤ x^2 := by sorry

end NUMINAMATH_CALUDE_f_upper_bound_l698_69802


namespace NUMINAMATH_CALUDE_baseball_cleats_price_l698_69884

/-- Proves that the price of each pair of baseball cleats is $10 -/
theorem baseball_cleats_price :
  let cards_price : ℝ := 25
  let bat_price : ℝ := 10
  let glove_original_price : ℝ := 30
  let glove_discount_percentage : ℝ := 0.2
  let total_sales : ℝ := 79
  let num_cleats_pairs : ℕ := 2

  let glove_sale_price : ℝ := glove_original_price * (1 - glove_discount_percentage)
  let non_cleats_sales : ℝ := cards_price + bat_price + glove_sale_price
  let cleats_total_price : ℝ := total_sales - non_cleats_sales
  let cleats_pair_price : ℝ := cleats_total_price / num_cleats_pairs

  cleats_pair_price = 10 := by
    sorry

end NUMINAMATH_CALUDE_baseball_cleats_price_l698_69884


namespace NUMINAMATH_CALUDE_large_jar_capacity_l698_69830

/-- Given a shelf of jars with the following properties:
  * There are 100 total jars
  * Small jars hold 3 liters each
  * The total capacity of all jars is 376 liters
  * There are 62 small jars
  This theorem proves that each large jar holds 5 liters. -/
theorem large_jar_capacity (total_jars : ℕ) (small_jar_capacity : ℕ) (total_capacity : ℕ) (small_jars : ℕ)
  (h1 : total_jars = 100)
  (h2 : small_jar_capacity = 3)
  (h3 : total_capacity = 376)
  (h4 : small_jars = 62) :
  (total_capacity - small_jars * small_jar_capacity) / (total_jars - small_jars) = 5 := by
  sorry

end NUMINAMATH_CALUDE_large_jar_capacity_l698_69830


namespace NUMINAMATH_CALUDE_total_jokes_over_two_saturdays_l698_69810

/-- 
Given that Jessy told 11 jokes and Alan told 7 jokes on the first Saturday,
and they both double their jokes on the second Saturday, prove that the
total number of jokes told over both Saturdays is 54.
-/
theorem total_jokes_over_two_saturdays 
  (jessy_first : ℕ) 
  (alan_first : ℕ) 
  (h1 : jessy_first = 11) 
  (h2 : alan_first = 7) : 
  jessy_first + alan_first + 2 * jessy_first + 2 * alan_first = 54 := by
  sorry

end NUMINAMATH_CALUDE_total_jokes_over_two_saturdays_l698_69810


namespace NUMINAMATH_CALUDE_product_of_roots_l698_69833

theorem product_of_roots (x : ℝ) : 
  (25 * x^2 + 60 * x - 350 = 0) → 
  ∃ r₁ r₂ : ℝ, (r₁ * r₂ = -14 ∧ 25 * r₁^2 + 60 * r₁ - 350 = 0 ∧ 25 * r₂^2 + 60 * r₂ - 350 = 0) :=
by sorry

end NUMINAMATH_CALUDE_product_of_roots_l698_69833


namespace NUMINAMATH_CALUDE_trigonometric_equation_solutions_l698_69803

theorem trigonometric_equation_solutions :
  ∀ x : ℝ, x ∈ Set.Icc 0 (2 * Real.pi) →
    (3 * Real.sin x = 1 + Real.cos (2 * x)) ↔ (x = Real.pi / 6 ∨ x = 5 * Real.pi / 6) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solutions_l698_69803


namespace NUMINAMATH_CALUDE_range_of_c_over_a_l698_69832

theorem range_of_c_over_a (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a > b) (h3 : b > c) :
  -2 < c / a ∧ c / a < -1/2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_c_over_a_l698_69832

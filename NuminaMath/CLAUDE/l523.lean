import Mathlib

namespace lcm_gcd_problem_l523_52300

theorem lcm_gcd_problem (a b : ℕ+) : 
  Nat.lcm a b = 4620 → 
  Nat.gcd a b = 21 → 
  a = 210 → 
  b = 462 := by
sorry

end lcm_gcd_problem_l523_52300


namespace arithmetic_sequence_sum_l523_52368

theorem arithmetic_sequence_sum (a₁ aₙ : ℤ) (n : ℕ) (h : n > 0) :
  (a₁ = -4) → (aₙ = 37) → (n = 10) →
  (∃ d : ℚ, ∀ k : ℕ, k < n → a₁ + k * d = aₙ - (n - 1 - k) * d) →
  (n : ℚ) * (a₁ + aₙ) / 2 = 165 :=
by sorry

end arithmetic_sequence_sum_l523_52368


namespace min_throws_for_three_occurrences_l523_52388

/-- The number of sides on each die -/
def sides : ℕ := 6

/-- The number of dice thrown each time -/
def num_dice : ℕ := 4

/-- The minimum possible sum when rolling the dice -/
def min_sum : ℕ := num_dice

/-- The maximum possible sum when rolling the dice -/
def max_sum : ℕ := num_dice * sides

/-- The number of possible different sums -/
def num_sums : ℕ := max_sum - min_sum + 1

/-- The minimum number of throws required -/
def min_throws : ℕ := num_sums * 2 + 1

theorem min_throws_for_three_occurrences :
  min_throws = 43 :=
sorry

end min_throws_for_three_occurrences_l523_52388


namespace ratio_odd_even_divisors_l523_52339

def N : ℕ := 34 * 34 * 63 * 270

def sum_odd_divisors (n : ℕ) : ℕ := sorry

def sum_even_divisors (n : ℕ) : ℕ := sorry

theorem ratio_odd_even_divisors :
  (sum_odd_divisors N : ℚ) / (sum_even_divisors N : ℚ) = 1 / 14 := by sorry

end ratio_odd_even_divisors_l523_52339


namespace expression_calculation_l523_52371

theorem expression_calculation : (75 * 2024 - 25 * 2024) / 2 = 50600 := by
  sorry

end expression_calculation_l523_52371


namespace second_number_in_first_set_l523_52305

theorem second_number_in_first_set (x : ℝ) : 
  (20 + x + 60) / 3 = (10 + 70 + 16) / 3 + 8 ↔ x = 40 := by
  sorry

end second_number_in_first_set_l523_52305


namespace planted_fraction_is_correct_l523_52379

/-- Represents a right triangle with an unplanted square in the corner -/
structure FieldTriangle where
  leg1 : ℝ
  leg2 : ℝ
  square_distance : ℝ

/-- The fraction of the field that is planted -/
def planted_fraction (f : FieldTriangle) : ℚ :=
  367 / 375

theorem planted_fraction_is_correct (f : FieldTriangle) 
  (h1 : f.leg1 = 5)
  (h2 : f.leg2 = 12)
  (h3 : f.square_distance = 4) :
  planted_fraction f = 367 / 375 := by
  sorry

end planted_fraction_is_correct_l523_52379


namespace permutation_5_2_combination_6_3_plus_6_4_l523_52378

-- Define permutation function
def A (n k : ℕ) : ℕ := sorry

-- Define combination function
def C (n k : ℕ) : ℕ := sorry

-- Theorem for A_5^2
theorem permutation_5_2 : A 5 2 = 20 := by sorry

-- Theorem for C_6^3 + C_6^4
theorem combination_6_3_plus_6_4 : C 6 3 + C 6 4 = 35 := by sorry

end permutation_5_2_combination_6_3_plus_6_4_l523_52378


namespace distinct_collections_count_l523_52325

def word : String := "COMPUTATIONS"

def vowels : Finset Char := {'O', 'U', 'A', 'I'}
def consonants : Multiset Char := {'C', 'M', 'P', 'T', 'T', 'S', 'N'}

def vowel_count : Nat := 4
def consonant_count : Nat := 11

def selected_vowels : Nat := 3
def selected_consonants : Nat := 4

theorem distinct_collections_count :
  (Nat.choose vowel_count selected_vowels) *
  (Nat.choose (consonant_count - 1) selected_consonants +
   Nat.choose (consonant_count - 1) (selected_consonants - 1) +
   Nat.choose (consonant_count - 1) (selected_consonants - 2)) = 200 := by
  sorry

end distinct_collections_count_l523_52325


namespace beef_weight_loss_percentage_l523_52358

/-- Given a side of beef weighing 800 pounds before processing and 640 pounds after processing,
    the percentage of weight lost during processing is 20%. -/
theorem beef_weight_loss_percentage (weight_before : ℝ) (weight_after : ℝ) :
  weight_before = 800 ∧ weight_after = 640 →
  (weight_before - weight_after) / weight_before * 100 = 20 := by
  sorry

end beef_weight_loss_percentage_l523_52358


namespace empty_solution_implies_a_geq_half_l523_52345

theorem empty_solution_implies_a_geq_half (a : ℝ) :
  (∀ x : ℝ, a * x^2 + x + a ≥ 0) → a ≥ 1/2 := by
  sorry

end empty_solution_implies_a_geq_half_l523_52345


namespace single_elimination_tournament_games_l523_52375

/-- Calculates the number of games needed in a single-elimination tournament. -/
def gamesNeeded (n : ℕ) : ℕ := n - 1

/-- Theorem: In a single-elimination tournament with 512 players, 511 games are needed to crown the champion. -/
theorem single_elimination_tournament_games :
  gamesNeeded 512 = 511 := by
  sorry

end single_elimination_tournament_games_l523_52375


namespace class_size_from_incorrect_mark_l523_52366

theorem class_size_from_incorrect_mark (original_mark correct_mark : ℚ)
  (h1 : original_mark = 33)
  (h2 : correct_mark = 85)
  (h3 : ∀ (n : ℕ) (A : ℚ), n * (A + 1/2) = n * A + (correct_mark - original_mark)) :
  ∃ (n : ℕ), n = 104 := by
  sorry

end class_size_from_incorrect_mark_l523_52366


namespace y_value_theorem_l523_52394

theorem y_value_theorem (y₁ y₂ y₃ y₄ y₅ y₆ y₇ y₈ : ℝ) 
  (eq1 : y₁ + 4*y₂ + 9*y₃ + 16*y₄ + 25*y₅ + 36*y₆ + 49*y₇ + 64*y₈ = 3)
  (eq2 : 4*y₁ + 9*y₂ + 16*y₃ + 25*y₄ + 36*y₅ + 49*y₆ + 64*y₇ + 81*y₈ = 15)
  (eq3 : 9*y₁ + 16*y₂ + 25*y₃ + 36*y₄ + 49*y₅ + 64*y₆ + 81*y₇ + 100*y₈ = 140) :
  16*y₁ + 25*y₂ + 36*y₃ + 49*y₄ + 64*y₅ + 81*y₆ + 100*y₇ + 121*y₈ = 472 :=
by sorry

end y_value_theorem_l523_52394


namespace simplify_expression_l523_52354

theorem simplify_expression :
  (((Real.sqrt 5 - 2) ^ (Real.sqrt 3 - 2)) / ((Real.sqrt 5 + 2) ^ (Real.sqrt 3 + 2))) = 41 + 20 * Real.sqrt 5 := by
  sorry

end simplify_expression_l523_52354


namespace square_fence_perimeter_l523_52386

-- Define the number of posts
def num_posts : ℕ := 24

-- Define the width of each post in feet
def post_width : ℚ := 1 / 3

-- Define the distance between adjacent posts in feet
def post_spacing : ℕ := 5

-- Define the number of posts per side (excluding corners)
def posts_per_side : ℕ := (num_posts - 4) / 4

-- Define the total number of posts per side (including corners)
def total_posts_per_side : ℕ := posts_per_side + 2

-- Define the number of gaps between posts on one side
def gaps_per_side : ℕ := total_posts_per_side - 1

-- Define the length of one side of the square
def side_length : ℚ := gaps_per_side * post_spacing + total_posts_per_side * post_width

-- Theorem statement
theorem square_fence_perimeter :
  4 * side_length = 129 + 1 / 3 :=
sorry

end square_fence_perimeter_l523_52386


namespace hamburger_combinations_l523_52334

/-- The number of condiments available for hamburgers -/
def num_condiments : ℕ := 9

/-- The number of choices for meat patties -/
def num_patty_choices : ℕ := 4

/-- The number of bread type choices -/
def num_bread_choices : ℕ := 2

/-- The total number of different hamburger combinations -/
def total_combinations : ℕ := 2^num_condiments * num_patty_choices * num_bread_choices

theorem hamburger_combinations :
  total_combinations = 4096 :=
sorry

end hamburger_combinations_l523_52334


namespace two_oak_trees_cut_down_l523_52397

/-- The number of oak trees cut down in the park --/
def oak_trees_cut_down (initial : ℕ) (final : ℕ) : ℕ :=
  initial - final

/-- Theorem: Given the initial and final number of oak trees, prove that 2 trees were cut down --/
theorem two_oak_trees_cut_down :
  oak_trees_cut_down 9 7 = 2 := by
  sorry

end two_oak_trees_cut_down_l523_52397


namespace cos_A_value_projection_BA_on_BC_l523_52391

noncomputable section

variables (A B C : ℝ) (a b c : ℝ)

-- Define the triangle ABC
def triangle_ABC : Prop :=
  2 * (Real.cos ((A - B) / 2))^2 * Real.cos B - Real.sin (A - B) * Real.sin B + Real.cos (A + C) = -3/5

-- Define the side lengths
def side_lengths : Prop :=
  a = 4 * Real.sqrt 2 ∧ b = 5

-- Theorem for part 1
theorem cos_A_value (h : triangle_ABC A B C) : Real.cos A = -3/5 := by sorry

-- Theorem for part 2
theorem projection_BA_on_BC (h1 : triangle_ABC A B C) (h2 : side_lengths a b) :
  ∃ (proj : ℝ), proj = Real.sqrt 2 / 2 ∧ proj = c * Real.cos B := by sorry

end

end cos_A_value_projection_BA_on_BC_l523_52391


namespace largest_prime_divisor_to_test_l523_52321

theorem largest_prime_divisor_to_test (n : ℕ) : 
  1000 ≤ n ∧ n ≤ 1100 → 
  (∀ p : ℕ, p.Prime → p ≤ 31 → n % p ≠ 0) → 
  n.Prime ∨ n = 1 :=
sorry

end largest_prime_divisor_to_test_l523_52321


namespace ellipse_equation_l523_52314

/-- The standard equation of an ellipse with given properties -/
theorem ellipse_equation (b c : ℝ) (h1 : b = 3) (h2 : c = 2) :
  ∃ a : ℝ, a^2 = b^2 + c^2 ∧ 
  (∀ x y : ℝ, (x^2 / b^2) + (y^2 / a^2) = 1 ↔ 
    x^2 / 9 + y^2 / 13 = 1) :=
by sorry

end ellipse_equation_l523_52314


namespace truck_weight_problem_l523_52322

theorem truck_weight_problem (truck_weight trailer_weight : ℝ) : 
  truck_weight + trailer_weight = 7000 →
  trailer_weight = 0.5 * truck_weight - 200 →
  truck_weight = 4800 := by
sorry

end truck_weight_problem_l523_52322


namespace range_of_g_l523_52333

noncomputable def g (x : ℝ) : ℝ := Real.arctan (2 * x) + Real.arctan ((2 - 3 * x) / (2 + 3 * x))

theorem range_of_g :
  Set.range g = {-3 * Real.pi / 4, Real.pi / 4} := by sorry

end range_of_g_l523_52333


namespace brick_width_l523_52374

/-- The surface area of a rectangular prism. -/
def surface_area (l w h : ℝ) : ℝ := 2 * (l * w + l * h + w * h)

/-- Theorem: The width of a brick with given dimensions and surface area. -/
theorem brick_width (l h : ℝ) (sa : ℝ) (hl : l = 10) (hh : h = 3) (hsa : sa = 164) :
  ∃ w : ℝ, w = 4 ∧ surface_area l w h = sa :=
sorry

end brick_width_l523_52374


namespace evaluate_expression_l523_52320

theorem evaluate_expression : -(16 / 2 * 12 - 75 + 4 * (2 * 5) + 25) = -86 := by
  sorry

end evaluate_expression_l523_52320


namespace competition_scores_l523_52387

theorem competition_scores (score24 score46 score12 : ℕ) : 
  score24 + score46 + score12 = 285 →
  ∃ (x : ℕ), score24 - 8 = x ∧ score46 - 12 = x ∧ score12 - 7 = x →
  score24 + score12 = 187 := by
sorry

end competition_scores_l523_52387


namespace range_of_a_l523_52338

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x^2 + a*x + 1 ≥ 0) → -2 ≤ a ∧ a ≤ 2 := by
  sorry

end range_of_a_l523_52338


namespace impossible_grid_arrangement_l523_52336

/-- A type representing the digits 0, 1, and 2 -/
inductive Digit
  | zero
  | one
  | two

/-- A type representing a 100 x 100 grid filled with Digits -/
def Grid := Fin 100 → Fin 100 → Digit

/-- A function to count the number of a specific digit in a 3 x 4 rectangle -/
def countDigitIn3x4Rectangle (g : Grid) (i j : Fin 100) (d : Digit) : ℕ :=
  sorry

/-- A predicate to check if a 3 x 4 rectangle satisfies the condition -/
def isValid3x4Rectangle (g : Grid) (i j : Fin 100) : Prop :=
  countDigitIn3x4Rectangle g i j Digit.zero = 3 ∧
  countDigitIn3x4Rectangle g i j Digit.one = 4 ∧
  countDigitIn3x4Rectangle g i j Digit.two = 5

/-- The main theorem stating that it's impossible to fill the grid satisfying the conditions -/
theorem impossible_grid_arrangement : ¬ ∃ (g : Grid), ∀ (i j : Fin 100), isValid3x4Rectangle g i j := by
  sorry

end impossible_grid_arrangement_l523_52336


namespace keiko_speed_l523_52364

theorem keiko_speed (track_width : ℝ) (time_difference : ℝ) : 
  track_width = 8 → time_difference = 48 → 
  (track_width * π) / time_difference = π / 3 := by
sorry

end keiko_speed_l523_52364


namespace road_construction_equation_l523_52350

theorem road_construction_equation (x : ℝ) : 
  x > 0 →
  (9 : ℝ) / x - 12 / (x + 1) = (1 : ℝ) / 2 ↔
  (9 / x = 12 / (x + 1) + 1 / 2 ∧
   9 = x * (12 / (x + 1) + 1 / 2) ∧
   12 = (x + 1) * (9 / x - 1 / 2)) :=
by sorry


end road_construction_equation_l523_52350


namespace tan_product_from_cosine_sum_l523_52355

theorem tan_product_from_cosine_sum (α β : ℝ) 
  (h : 3 * Real.cos (2 * α + β) + 5 * Real.cos β = 0) : 
  Real.tan (α + β) * Real.tan α = -4 := by
  sorry

end tan_product_from_cosine_sum_l523_52355


namespace positive_expressions_l523_52310

theorem positive_expressions (a b c : ℝ) 
  (ha : 0 < a ∧ a < 2) 
  (hb : -2 < b ∧ b < 0) 
  (hc : 0 < c ∧ c < 3) : 
  0 < b + b^2 ∧ 0 < b + 3*b^2 := by
  sorry

end positive_expressions_l523_52310


namespace simultaneous_completion_time_specific_completion_time_l523_52328

/-- The time taken for two machines to complete an order when working simultaneously, 
    given their individual completion times. -/
theorem simultaneous_completion_time (t1 t2 : ℝ) (h1 : t1 > 0) (h2 : t2 > 0) : 
  (t1 * t2) / (t1 + t2) = (t1 * t2) / ((t1 * t2) * (1 / t1 + 1 / t2)) := by
  sorry

/-- Proof that two machines with completion times of 9 hours and 8 hours respectively
    will take 72/17 hours to complete the order when working simultaneously. -/
theorem specific_completion_time : 
  (9 : ℝ) * 8 / (9 + 8) = 72 / 17 := by
  sorry

end simultaneous_completion_time_specific_completion_time_l523_52328


namespace jacket_cost_calculation_l523_52376

def shorts_cost : ℚ := 13.99
def shirt_cost : ℚ := 12.14
def total_cost : ℚ := 33.56

theorem jacket_cost_calculation :
  ∃ (jacket_cost : ℚ), 
    jacket_cost = total_cost - (shorts_cost + shirt_cost) ∧
    jacket_cost = 7.43 := by
  sorry

end jacket_cost_calculation_l523_52376


namespace transform_minus_four_plus_two_i_l523_52392

/-- Applies a 270° counter-clockwise rotation followed by a scaling of 2 to a complex number -/
def transform (z : ℂ) : ℂ := 2 * (z * Complex.I)

/-- The result of applying the transformation to -4 + 2i -/
theorem transform_minus_four_plus_two_i :
  transform (Complex.ofReal (-4) + Complex.I * Complex.ofReal 2) = Complex.ofReal 4 + Complex.I * Complex.ofReal 8 := by
  sorry

#check transform_minus_four_plus_two_i

end transform_minus_four_plus_two_i_l523_52392


namespace smallest_even_number_sum_1194_l523_52359

/-- Given three consecutive even numbers whose sum is 1194, 
    the smallest of these numbers is 396. -/
theorem smallest_even_number_sum_1194 (x : ℕ) 
  (h1 : x % 2 = 0)  -- x is even
  (h2 : x + (x + 2) + (x + 4) = 1194) : x = 396 := by
  sorry

end smallest_even_number_sum_1194_l523_52359


namespace percent_within_one_sd_is_68_l523_52356

/-- A symmetric distribution with a given percentage below one standard deviation above the mean -/
structure SymmetricDistribution where
  /-- The percentage of the distribution below one standard deviation above the mean -/
  percent_below_one_sd : ℝ
  /-- Assumption that the percentage is 84% -/
  percent_is_84 : percent_below_one_sd = 84

/-- The percentage of a symmetric distribution that lies within one standard deviation of the mean -/
def percent_within_one_sd (d : SymmetricDistribution) : ℝ :=
  2 * d.percent_below_one_sd - 100

theorem percent_within_one_sd_is_68 (d : SymmetricDistribution) :
  percent_within_one_sd d = 68 := by
  sorry

end percent_within_one_sd_is_68_l523_52356


namespace bigger_part_is_thirteen_l523_52370

theorem bigger_part_is_thirteen (x y : ℝ) (h1 : x + y = 24) (h2 : 7 * x + 5 * y = 146) 
  (h3 : x > 0) (h4 : y > 0) : max x y = 13 := by
  sorry

end bigger_part_is_thirteen_l523_52370


namespace cube_root_27_times_fourth_root_16_l523_52381

theorem cube_root_27_times_fourth_root_16 : (27 : ℝ) ^ (1/3) * (16 : ℝ) ^ (1/4) = 6 := by
  sorry

end cube_root_27_times_fourth_root_16_l523_52381


namespace largest_prime_sum_l523_52335

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that returns the digits of a natural number -/
def digits (n : ℕ) : List ℕ := sorry

/-- A function that checks if a list contains exactly the digits 1 to 9 -/
def usesAllDigits (l : List ℕ) : Prop := sorry

theorem largest_prime_sum :
  ∀ (p₁ p₂ p₃ p₄ : ℕ),
    isPrime p₁ ∧ isPrime p₂ ∧ isPrime p₃ ∧ isPrime p₄ →
    usesAllDigits (digits p₁ ++ digits p₂ ++ digits p₃ ++ digits p₄) →
    p₁ + p₂ + p₃ + p₄ ≤ 1798 :=
by
  sorry

end largest_prime_sum_l523_52335


namespace erasers_per_group_l523_52361

theorem erasers_per_group (total_erasers : ℕ) (num_groups : ℕ) (h1 : total_erasers = 270) (h2 : num_groups = 3) :
  total_erasers / num_groups = 90 := by
  sorry

end erasers_per_group_l523_52361


namespace equation_solutions_l523_52324

/-- A parabola that intersects the x-axis at (-1, 0) and (3, 0) -/
structure Parabola where
  m : ℝ
  n : ℝ
  intersect_neg_one : (-1 - m)^2 + n = 0
  intersect_three : (3 - m)^2 + n = 0

/-- The equation to solve -/
def equation (p : Parabola) (x : ℝ) : Prop :=
  (x - 1)^2 + p.m^2 = 2 * p.m * (x - 1) - p.n

/-- The theorem stating the solutions to the equation -/
theorem equation_solutions (p : Parabola) :
  ∃ (x₁ x₂ : ℝ), x₁ = 0 ∧ x₂ = 4 ∧ equation p x₁ ∧ equation p x₂ :=
sorry

end equation_solutions_l523_52324


namespace item_cost_before_tax_reduction_cost_is_1000_l523_52390

theorem item_cost_before_tax_reduction (tax_difference : ℝ) (cost_difference : ℝ) : ℝ :=
  let original_tax_rate := 0.05
  let new_tax_rate := 0.04
  let item_cost := cost_difference / (original_tax_rate - new_tax_rate)
  item_cost

theorem cost_is_1000 :
  item_cost_before_tax_reduction 0.01 10 = 1000 := by
  sorry

end item_cost_before_tax_reduction_cost_is_1000_l523_52390


namespace august_math_problems_l523_52389

def problem (x y z : ℝ) : Prop :=
  let first_answer := x
  let second_answer := 2 * x - y
  let third_answer := 3 * x - z
  let fourth_answer := (x + (2 * x - y) + (3 * x - z)) / 3
  x = 600 ∧
  y > 0 ∧
  z = (x + (2 * x - y)) - 400 ∧
  first_answer + second_answer + third_answer + fourth_answer = 2933.33

theorem august_math_problems :
  ∃ (y z : ℝ), problem 600 y z :=
sorry

end august_math_problems_l523_52389


namespace zeros_between_seven_and_three_l523_52308

theorem zeros_between_seven_and_three : ∀ n : ℕ, 
  (7 * 10^(n + 1) + 3 = 70003) ↔ (n = 4) :=
by sorry

end zeros_between_seven_and_three_l523_52308


namespace estimate_wild_rabbits_l523_52385

theorem estimate_wild_rabbits (initial_marked : ℕ) (recaptured : ℕ) (marked_in_recapture : ℕ) :
  initial_marked = 100 →
  recaptured = 40 →
  marked_in_recapture = 5 →
  (recaptured * initial_marked) / marked_in_recapture = 800 :=
by sorry

end estimate_wild_rabbits_l523_52385


namespace regular_polygon_perimeter_l523_52349

/-- A regular polygon with side length 7 and exterior angle 90 degrees has a perimeter of 28 units. -/
theorem regular_polygon_perimeter (n : ℕ) (side_length : ℝ) (exterior_angle : ℝ) :
  n > 0 ∧ 
  side_length = 7 ∧ 
  exterior_angle = 90 ∧ 
  exterior_angle = 360 / n →
  n * side_length = 28 :=
by sorry

end regular_polygon_perimeter_l523_52349


namespace count_valid_integers_l523_52326

/-- The set of available digits -/
def available_digits : Finset ℕ := {1, 4, 7}

/-- The count of each digit in the available set -/
def digit_count : ℕ → ℕ
  | 1 => 2
  | 4 => 3
  | 7 => 1
  | _ => 0

/-- A valid three-digit integer formed from the available digits -/
structure ValidInteger where
  hundreds : ℕ
  tens : ℕ
  ones : ℕ
  hundreds_in_set : hundreds ∈ available_digits
  tens_in_set : tens ∈ available_digits
  ones_in_set : ones ∈ available_digits
  valid_count : ∀ d ∈ available_digits,
    (if hundreds = d then 1 else 0) +
    (if tens = d then 1 else 0) +
    (if ones = d then 1 else 0) ≤ digit_count d

/-- The set of all valid three-digit integers -/
def valid_integers : Finset ValidInteger := sorry

theorem count_valid_integers :
  Finset.card valid_integers = 31 := by sorry

end count_valid_integers_l523_52326


namespace emma_bank_money_l523_52360

theorem emma_bank_money (X : ℝ) : 
  X > 0 →
  (1/4 : ℝ) * (X - 400) = 400 →
  X = 2000 := by
sorry

end emma_bank_money_l523_52360


namespace consecutive_sum_product_l523_52380

theorem consecutive_sum_product (n : ℕ) (h : n > 100) :
  ∃ (a b c : ℕ), a > 1 ∧ b > 1 ∧ c > 1 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  (3 * (n + 1) = a * b * c ∨ 3 * (n + 2) = a * b * c) :=
by sorry

end consecutive_sum_product_l523_52380


namespace selection_schemes_count_l523_52311

/-- The number of people to choose from -/
def total_people : ℕ := 6

/-- The number of pavilions to visit -/
def pavilions : ℕ := 4

/-- The number of people who cannot visit a specific pavilion -/
def restricted_people : ℕ := 2

/-- Calculates the number of ways to select people for pavilions with restrictions -/
def selection_schemes : ℕ :=
  Nat.descFactorial total_people pavilions - 
  restricted_people * Nat.descFactorial (total_people - 1) (pavilions - 1)

/-- The theorem stating the number of selection schemes -/
theorem selection_schemes_count : selection_schemes = 240 := by sorry

end selection_schemes_count_l523_52311


namespace new_person_age_l523_52315

theorem new_person_age (n : ℕ) (initial_avg : ℝ) (new_avg : ℝ) (new_person_age : ℝ) : 
  n = 9 → 
  initial_avg = 14 → 
  new_avg = 16 → 
  (n * initial_avg + new_person_age) / (n + 1) = new_avg → 
  new_person_age = 34 := by
sorry

end new_person_age_l523_52315


namespace blue_square_area_ratio_l523_52319

/-- Represents a square flag with a symmetric cross -/
structure CrossFlag where
  side : ℝ
  cross_area_ratio : ℝ
  symmetric : Bool

/-- The area of the flag -/
def flag_area (flag : CrossFlag) : ℝ := flag.side ^ 2

/-- The area of the cross -/
def cross_area (flag : CrossFlag) : ℝ := flag.cross_area_ratio * flag_area flag

/-- The theorem stating the relationship between the blue square area and the flag area -/
theorem blue_square_area_ratio (flag : CrossFlag) 
  (h1 : flag.cross_area_ratio = 0.36)
  (h2 : flag.symmetric = true) : 
  (flag.side * 0.2) ^ 2 / flag_area flag = 0.04 := by
  sorry

#check blue_square_area_ratio

end blue_square_area_ratio_l523_52319


namespace graph_not_in_second_quadrant_l523_52365

/-- A linear function y = 3x + k - 2 -/
def f (k : ℝ) (x : ℝ) : ℝ := 3 * x + k - 2

/-- The second quadrant -/
def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- The graph does not pass through the second quadrant -/
def not_in_second_quadrant (k : ℝ) : Prop :=
  ∀ x, ¬(second_quadrant x (f k x))

/-- Theorem: The graph of y = 3x + k - 2 does not pass through the second quadrant
    if and only if k ≤ 2 -/
theorem graph_not_in_second_quadrant (k : ℝ) :
  not_in_second_quadrant k ↔ k ≤ 2 := by
  sorry


end graph_not_in_second_quadrant_l523_52365


namespace tan_x_eq_2_implies_expression_l523_52313

theorem tan_x_eq_2_implies_expression (x : ℝ) (h : Real.tan x = 2) :
  (6 * Real.sin (2 * x) + 2 * Real.cos (2 * x)) / (Real.cos (2 * x) - 3 * Real.sin (2 * x)) = -2/5 := by
  sorry

end tan_x_eq_2_implies_expression_l523_52313


namespace order_divides_exponent_l523_52352

theorem order_divides_exponent (x m d p : ℕ) (hp : Prime p) : 
  (∀ k : ℕ, k > 0 ∧ k < d → x^k % p ≠ 1) →  -- d is the order of x modulo p
  x^d % p = 1 →                             -- definition of order
  x^m % p = 1 →                             -- given condition
  d ∣ m :=                                  -- conclusion: d divides m
sorry

end order_divides_exponent_l523_52352


namespace arithmetic_sequence_ratio_l523_52393

theorem arithmetic_sequence_ratio (a d : ℚ) : 
  let S : ℕ → ℚ := λ n => n / 2 * (2 * a + (n - 1) * d)
  S 15 = 3 * S 8 → a / d = 7 / 3 := by
sorry

end arithmetic_sequence_ratio_l523_52393


namespace problem_solution_l523_52329

theorem problem_solution (n m q q' r r' : ℕ) : 
  n > m ∧ m > 1 ∧
  n = q * m + r ∧ r < m ∧
  n - 1 = q' * m + r' ∧ r' < m ∧
  q + q' = 99 ∧ r + r' = 99 →
  n = 5000 ∧ ∃ k : ℕ, 2 * n = k * k :=
by sorry

end problem_solution_l523_52329


namespace inequalities_always_true_l523_52383

theorem inequalities_always_true (x : ℝ) : 
  (x^2 + 6*x + 10 > 0) ∧ (-x^2 + x - 2 < 0) := by
  sorry

end inequalities_always_true_l523_52383


namespace factory_B_cheaper_for_200_copies_l523_52327

/-- Cost calculation for Factory A -/
def cost_A (x : ℝ) : ℝ := 4.8 * x + 500

/-- Cost calculation for Factory B -/
def cost_B (x : ℝ) : ℝ := 6 * x + 200

/-- Theorem stating that Factory B has lower cost for 200 copies -/
theorem factory_B_cheaper_for_200_copies :
  cost_B 200 < cost_A 200 := by
  sorry

end factory_B_cheaper_for_200_copies_l523_52327


namespace triangle_not_right_angle_l523_52343

theorem triangle_not_right_angle (a b c : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_ratio : b = (4/3) * a ∧ c = (5/3) * a) (h_sum : a + b + c = 180) :
  ¬ (a = 90 ∨ b = 90 ∨ c = 90) :=
sorry

end triangle_not_right_angle_l523_52343


namespace concatenated_number_irrational_l523_52332

/-- The number formed by concatenating the digits of 3^k for k = 1, 2, ... -/
def concatenated_number : ℝ :=
  sorry

/-- Theorem stating that the concatenated number is irrational -/
theorem concatenated_number_irrational : Irrational concatenated_number :=
sorry

end concatenated_number_irrational_l523_52332


namespace problem_solution_l523_52344

def A : Set ℝ := {x | (2*x - 2)/(x + 1) < 1}

def B (a : ℝ) : Set ℝ := {x | x^2 + x + a - a^2 < 0}

theorem problem_solution :
  (∀ x, x ∈ (B 1 ∪ (Set.univ \ A)) ↔ (x < 0 ∨ x ≥ 3)) ∧
  (∀ a, A = B a ↔ a ∈ Set.Iic (-3) ∪ Set.Ici 4) := by sorry

end problem_solution_l523_52344


namespace min_abs_z_plus_i_l523_52340

theorem min_abs_z_plus_i (z : ℂ) (h : Complex.abs (z^2 + 9) = Complex.abs (z * (z + 3*I))) :
  ∃ (w : ℂ), Complex.abs (w + I) = 2 ∧ ∀ (z : ℂ), Complex.abs (z^2 + 9) = Complex.abs (z * (z + 3*I)) → Complex.abs (z + I) ≥ 2 :=
sorry

end min_abs_z_plus_i_l523_52340


namespace grape_problem_l523_52351

theorem grape_problem (x : ℕ) : x > 100 ∧ 
                                x % 3 = 1 ∧ 
                                x % 5 = 2 ∧ 
                                x % 7 = 4 → 
                                x ≤ 172 :=
by sorry

end grape_problem_l523_52351


namespace probability_five_blue_marbles_l523_52362

def total_marbles : ℕ := 15
def blue_marbles : ℕ := 9
def red_marbles : ℕ := 6
def total_draws : ℕ := 8
def blue_draws : ℕ := 5

def prob_blue : ℚ := blue_marbles / total_marbles
def prob_red : ℚ := red_marbles / total_marbles

theorem probability_five_blue_marbles :
  (Nat.choose total_draws blue_draws : ℚ) *
  (prob_blue ^ blue_draws) *
  (prob_red ^ (total_draws - blue_draws)) =
  108864 / 390625 := by sorry

end probability_five_blue_marbles_l523_52362


namespace magnified_cell_size_l523_52399

/-- The diameter of a certain type of cell in meters -/
def cell_diameter : ℝ := 1.56e-6

/-- The magnification factor -/
def magnification : ℝ := 1e6

/-- The magnified size of the cell -/
def magnified_size : ℝ := cell_diameter * magnification

theorem magnified_cell_size :
  magnified_size = 1.56 := by sorry

end magnified_cell_size_l523_52399


namespace cindy_used_stickers_l523_52363

theorem cindy_used_stickers (initial_stickers : ℕ) (cindy_remaining : ℕ) : 
  initial_stickers + 18 = cindy_remaining + 33 → 
  initial_stickers - cindy_remaining = 15 :=
by
  sorry

end cindy_used_stickers_l523_52363


namespace train_bridge_crossing_time_l523_52304

/-- Proves that a train with given length and speed takes the specified time to cross a bridge of given length -/
theorem train_bridge_crossing_time
  (train_length : ℝ)
  (train_speed_kmh : ℝ)
  (bridge_length : ℝ)
  (h1 : train_length = 150)
  (h2 : train_speed_kmh = 45)
  (h3 : bridge_length = 225) :
  (train_length + bridge_length) / (train_speed_kmh * 1000 / 3600) = 30 := by
  sorry

#check train_bridge_crossing_time

end train_bridge_crossing_time_l523_52304


namespace no_negative_sum_of_squares_l523_52316

theorem no_negative_sum_of_squares : ¬∃ (x y : ℝ), x^2 + y^2 < 0 := by
  sorry

end no_negative_sum_of_squares_l523_52316


namespace stock_sale_loss_l523_52357

/-- Calculates the overall loss amount for a stock sale scenario -/
theorem stock_sale_loss (stock_worth : ℝ) (profit_percent : ℝ) (loss_percent : ℝ) 
  (profit_stock_percent : ℝ) (loss_stock_percent : ℝ) :
  stock_worth = 10000 →
  profit_percent = 10 →
  loss_percent = 5 →
  profit_stock_percent = 20 →
  loss_stock_percent = 80 →
  let profit_amount := (profit_stock_percent / 100) * stock_worth * (1 + profit_percent / 100)
  let loss_amount := (loss_stock_percent / 100) * stock_worth * (1 - loss_percent / 100)
  let total_sale := profit_amount + loss_amount
  stock_worth - total_sale = 200 := by sorry

end stock_sale_loss_l523_52357


namespace smallest_solution_of_equation_l523_52301

theorem smallest_solution_of_equation :
  ∃ (x : ℝ), x = -3 ∧ 
  (3 * x) / (x + 3) + (3 * x^2 - 18) / x = 9 ∧
  (∀ y : ℝ, (3 * y) / (y + 3) + (3 * y^2 - 18) / y = 9 → y ≥ x) := by
  sorry

end smallest_solution_of_equation_l523_52301


namespace students_just_passed_l523_52330

theorem students_just_passed (total : ℕ) (first_div_percent : ℚ) (second_div_percent : ℚ) 
  (h_total : total = 300)
  (h_first : first_div_percent = 29/100)
  (h_second : second_div_percent = 54/100)
  (h_no_fail : first_div_percent + second_div_percent < 1) :
  total - (total * first_div_percent).floor - (total * second_div_percent).floor = 51 := by
  sorry

end students_just_passed_l523_52330


namespace matrix_is_own_inverse_l523_52396

def A (a b c : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  !![a, b, c; 2, -1, 0; 0, 0, 1]

theorem matrix_is_own_inverse (a b c : ℝ) :
  A a b c * A a b c = 1 → a = 1 ∧ b = 0 ∧ c = 0 := by
  sorry

end matrix_is_own_inverse_l523_52396


namespace max_value_of_shui_l523_52303

def ChineseDigit := Fin 8

structure Phrase :=
  (jin xin li : ChineseDigit)
  (ke ba shan : ChineseDigit)
  (qiong shui : ChineseDigit)

def is_valid_phrase (p : Phrase) : Prop :=
  p.jin.val + p.xin.val + p.jin.val + p.li.val = 19 ∧
  p.li.val + p.ke.val + p.ba.val + p.shan.val = 19 ∧
  p.shan.val + p.qiong.val + p.shui.val + p.jin.val = 19

def all_different (p : Phrase) : Prop :=
  p.jin ≠ p.xin ∧ p.jin ≠ p.li ∧ p.jin ≠ p.ke ∧ p.jin ≠ p.ba ∧ p.jin ≠ p.shan ∧ p.jin ≠ p.qiong ∧ p.jin ≠ p.shui ∧
  p.xin ≠ p.li ∧ p.xin ≠ p.ke ∧ p.xin ≠ p.ba ∧ p.xin ≠ p.shan ∧ p.xin ≠ p.qiong ∧ p.xin ≠ p.shui ∧
  p.li ≠ p.ke ∧ p.li ≠ p.ba ∧ p.li ≠ p.shan ∧ p.li ≠ p.qiong ∧ p.li ≠ p.shui ∧
  p.ke ≠ p.ba ∧ p.ke ≠ p.shan ∧ p.ke ≠ p.qiong ∧ p.ke ≠ p.shui ∧
  p.ba ≠ p.shan ∧ p.ba ≠ p.qiong ∧ p.ba ≠ p.shui ∧
  p.shan ≠ p.qiong ∧ p.shan ≠ p.shui ∧
  p.qiong ≠ p.shui

theorem max_value_of_shui (p : Phrase) 
  (h1 : is_valid_phrase p)
  (h2 : all_different p)
  (h3 : p.jin.val > p.shan.val ∧ p.shan.val > p.li.val) :
  p.shui.val ≤ 7 := by
  sorry

end max_value_of_shui_l523_52303


namespace walking_speed_problem_l523_52382

/-- The walking speed problem -/
theorem walking_speed_problem 
  (distance_between_homes : ℝ)
  (bob_speed : ℝ)
  (alice_distance : ℝ)
  (time_difference : ℝ)
  (h1 : distance_between_homes = 41)
  (h2 : bob_speed = 4)
  (h3 : alice_distance = 25)
  (h4 : time_difference = 1)
  : ∃ (alice_speed : ℝ), 
    alice_speed = 5 ∧ 
    alice_distance / alice_speed = (distance_between_homes - alice_distance) / bob_speed + time_difference :=
by sorry

end walking_speed_problem_l523_52382


namespace tank_capacity_l523_52347

/-- The capacity of a tank given specific filling and draining rates and a cyclic operation pattern. -/
theorem tank_capacity 
  (fill_rate_A : ℕ) 
  (fill_rate_B : ℕ) 
  (drain_rate_C : ℕ) 
  (total_time : ℕ) 
  (h1 : fill_rate_A = 40)
  (h2 : fill_rate_B = 30)
  (h3 : drain_rate_C = 20)
  (h4 : total_time = 57) :
  fill_rate_A + fill_rate_B - drain_rate_C = 50 →
  (total_time / 3) * (fill_rate_A + fill_rate_B - drain_rate_C) + fill_rate_A = 990 := by
  sorry

#check tank_capacity

end tank_capacity_l523_52347


namespace soup_problem_solution_l523_52307

/-- Represents the number of people a can of soup can feed -/
structure SoupCan where
  adults : ℕ
  children : ℕ

/-- Represents the problem setup -/
structure SoupProblem where
  can : SoupCan
  totalCans : ℕ
  childrenFed : ℕ

/-- Calculates the number of adults that can be fed with the remaining soup -/
def remainingAdults (problem : SoupProblem) : ℕ :=
  let cansUsedForChildren := problem.childrenFed / problem.can.children
  let remainingCans := problem.totalCans - cansUsedForChildren
  remainingCans * problem.can.adults

/-- Theorem stating that given the problem conditions, 12 adults can be fed with the remaining soup -/
theorem soup_problem_solution (problem : SoupProblem) 
  (h1 : problem.can.adults = 4)
  (h2 : problem.can.children = 6)
  (h3 : problem.totalCans = 6)
  (h4 : problem.childrenFed = 18) :
  remainingAdults problem = 12 := by
  sorry

#eval remainingAdults { can := { adults := 4, children := 6 }, totalCans := 6, childrenFed := 18 }

end soup_problem_solution_l523_52307


namespace black_balls_count_l523_52398

theorem black_balls_count (total_balls : ℕ) (white_balls : ℕ → ℕ) (black_balls : ℕ) :
  total_balls = 56 →
  white_balls black_balls = 6 * black_balls →
  total_balls = white_balls black_balls + black_balls →
  black_balls = 8 := by
sorry

end black_balls_count_l523_52398


namespace acute_triangle_in_right_triangle_l523_52318

/-- A triangle represented by its vertices -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Predicate to check if a triangle is acute-angled -/
def IsAcuteAngled (t : Triangle) : Prop := sorry

/-- Function to calculate the area of a triangle -/
def TriangleArea (t : Triangle) : ℝ := sorry

/-- Predicate to check if a triangle is right-angled -/
def IsRightAngled (t : Triangle) : Prop := sorry

/-- Predicate to check if one triangle contains another -/
def Contains (t1 t2 : Triangle) : Prop := sorry

theorem acute_triangle_in_right_triangle :
  ∀ (t : Triangle), IsAcuteAngled t → TriangleArea t = 1 →
  ∃ (r : Triangle), IsRightAngled r ∧ TriangleArea r = Real.sqrt 3 ∧ Contains r t := by
  sorry

end acute_triangle_in_right_triangle_l523_52318


namespace dal_gain_is_104_l523_52302

/-- Calculates the total gain from selling a mixture of dals -/
def calculate_dal_gain (dal_a_kg : ℝ) (dal_a_rate : ℝ) (dal_b_kg : ℝ) (dal_b_rate : ℝ)
                       (dal_c_kg : ℝ) (dal_c_rate : ℝ) (dal_d_kg : ℝ) (dal_d_rate : ℝ)
                       (mixture_rate : ℝ) : ℝ :=
  let total_cost := dal_a_kg * dal_a_rate + dal_b_kg * dal_b_rate +
                    dal_c_kg * dal_c_rate + dal_d_kg * dal_d_rate
  let total_weight := dal_a_kg + dal_b_kg + dal_c_kg + dal_d_kg
  let total_revenue := total_weight * mixture_rate
  total_revenue - total_cost

theorem dal_gain_is_104 :
  calculate_dal_gain 15 14.5 10 13 12 16 8 18 17.5 = 104 := by
  sorry

end dal_gain_is_104_l523_52302


namespace inscribed_circle_radius_l523_52346

theorem inscribed_circle_radius (A p r s : ℝ) : 
  A = 2 * p →  -- Area is twice the perimeter
  A = r * s →  -- Area formula using inradius and semiperimeter
  p = 2 * s →  -- Perimeter is twice the semiperimeter
  r = 4 := by sorry

end inscribed_circle_radius_l523_52346


namespace green_ball_probability_l523_52342

-- Define the containers and their contents
def container_A : Nat × Nat := (3, 7)  -- (red, green)
def container_B : Nat × Nat := (5, 5)
def container_C : Nat × Nat := (5, 5)

-- Define the probability of selecting each container
def container_prob : Rat := 1/3

-- Define the probability of selecting a green ball from each container
def green_prob_A : Rat := container_prob * (container_A.2 / (container_A.1 + container_A.2))
def green_prob_B : Rat := container_prob * (container_B.2 / (container_B.1 + container_B.2))
def green_prob_C : Rat := container_prob * (container_C.2 / (container_C.1 + container_C.2))

-- Theorem: The probability of selecting a green ball is 17/30
theorem green_ball_probability : 
  green_prob_A + green_prob_B + green_prob_C = 17/30 := by
  sorry

end green_ball_probability_l523_52342


namespace f_neg_one_equals_one_l523_52377

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem f_neg_one_equals_one (h : ∀ x, f (x - 1) = x^2 + 1) : f (-1) = 1 := by
  sorry

end f_neg_one_equals_one_l523_52377


namespace prob_three_heads_eight_tosses_l523_52369

/-- The probability of getting exactly k heads in n tosses of a fair coin -/
def prob_k_heads (n k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) / (2 ^ n : ℚ)

/-- Theorem: The probability of getting exactly 3 heads in 8 tosses of a fair coin is 7/32 -/
theorem prob_three_heads_eight_tosses :
  prob_k_heads 8 3 = 7 / 32 := by
  sorry

end prob_three_heads_eight_tosses_l523_52369


namespace customer_satisfaction_probability_l523_52306

/-- The probability that a dissatisfied customer leaves an angry review -/
def prob_dissatisfied_angry : ℝ := 0.80

/-- The probability that a satisfied customer leaves a positive review -/
def prob_satisfied_positive : ℝ := 0.15

/-- The number of angry reviews received -/
def num_angry_reviews : ℕ := 60

/-- The number of positive reviews received -/
def num_positive_reviews : ℕ := 20

/-- The probability that a customer is satisfied -/
def prob_satisfied : ℝ := 0.64

theorem customer_satisfaction_probability :
  prob_satisfied = 0.64 :=
sorry

end customer_satisfaction_probability_l523_52306


namespace similar_triangles_shortest_side_l523_52348

theorem similar_triangles_shortest_side
  (a b c : ℝ)  -- sides of the first triangle
  (k : ℝ)      -- scaling factor
  (h1 : a^2 + b^2 = c^2)  -- Pythagorean theorem for the first triangle
  (h2 : a = 15)           -- given side length of the first triangle
  (h3 : c = 17)           -- hypotenuse of the first triangle
  (h4 : k * c = 68)       -- hypotenuse of the second triangle
  : k * min a b = 32 :=
by sorry

end similar_triangles_shortest_side_l523_52348


namespace train_speed_l523_52317

theorem train_speed (train_length : Real) (crossing_time : Real) (h1 : train_length = 1600) (h2 : crossing_time = 40) :
  (train_length / 1000) / (crossing_time / 3600) = 144 := by
  sorry

end train_speed_l523_52317


namespace eighteen_power_mn_l523_52384

theorem eighteen_power_mn (m n : ℤ) (R S : ℝ) (hR : R = 2^m) (hS : S = 3^n) :
  18^(m+n) = R^n * S^(2*m) := by
  sorry

end eighteen_power_mn_l523_52384


namespace larger_number_proof_l523_52353

theorem larger_number_proof (a b : ℕ+) : 
  (Nat.gcd a b = 20) →
  (Nat.lcm a b = 3640) →
  (13 ∣ Nat.lcm a b) →
  (14 ∣ Nat.lcm a b) →
  max a b = 280 := by
sorry

end larger_number_proof_l523_52353


namespace number_of_girls_in_field_trip_l523_52373

/-- The number of girls in a field trip given the number of students in each van and the total number of boys -/
theorem number_of_girls_in_field_trip (van1 van2 van3 van4 van5 total_boys : ℕ) 
  (h1 : van1 = 24)
  (h2 : van2 = 30)
  (h3 : van3 = 20)
  (h4 : van4 = 36)
  (h5 : van5 = 29)
  (h6 : total_boys = 64) :
  van1 + van2 + van3 + van4 + van5 - total_boys = 75 := by
  sorry

end number_of_girls_in_field_trip_l523_52373


namespace log_equation_solution_l523_52331

theorem log_equation_solution (x : ℝ) (h : x > 0) :
  (Real.log x / Real.log 2) + 1 / (Real.log (x + 1) / Real.log 2) = 1 ↔ x = 1 := by
  sorry

end log_equation_solution_l523_52331


namespace min_socks_for_15_pairs_l523_52367

/-- Represents a box of socks with four different colors. -/
structure SockBox where
  purple : ℕ
  orange : ℕ
  yellow : ℕ
  green : ℕ

/-- The minimum number of socks needed to guarantee at least n pairs. -/
def min_socks_for_pairs (n : ℕ) : ℕ := 2 * n + 3

/-- Theorem stating the minimum number of socks needed for 15 pairs. -/
theorem min_socks_for_15_pairs (box : SockBox) :
  min_socks_for_pairs 15 = 33 :=
sorry

end min_socks_for_15_pairs_l523_52367


namespace quadrilateral_ratio_l523_52323

theorem quadrilateral_ratio (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h1 : a^2 + d^2 - a*d = b^2 + c^2 + b*c) (h2 : a^2 + b^2 = c^2 + d^2) :
  (a*b + c*d) / (a*d + b*c) = Real.sqrt 3 / 2 := by
  sorry

end quadrilateral_ratio_l523_52323


namespace quadratic_root_implies_n_l523_52337

theorem quadratic_root_implies_n (n : ℝ) : 
  (∃ x : ℝ, x^2 - 2*x + n = 0) ∧ (3^2 - 2*3 + n = 0) → n = -3 := by
  sorry

end quadratic_root_implies_n_l523_52337


namespace number_difference_l523_52395

theorem number_difference (a b : ℕ) 
  (sum_eq : a + b = 23540)
  (b_div_16 : b % 16 = 0)
  (b_eq_100a : b = 100 * a) : 
  b - a = 23067 := by sorry

end number_difference_l523_52395


namespace cookies_calculation_l523_52372

/-- The number of people receiving cookies -/
def num_people : ℝ := 6.0

/-- The number of cookies each person should receive -/
def cookies_per_person : ℝ := 24.0

/-- The total number of cookies needed -/
def total_cookies : ℝ := num_people * cookies_per_person

theorem cookies_calculation : total_cookies = 144.0 := by
  sorry

end cookies_calculation_l523_52372


namespace inverse_f_123_l523_52341

noncomputable def f (x : ℝ) : ℝ := 3 * x^3 + 6

theorem inverse_f_123 : f⁻¹ 123 = (39 : ℝ)^(1/3) := by sorry

end inverse_f_123_l523_52341


namespace work_completion_days_l523_52312

/-- Represents the work scenario with initial workers and additional workers joining later. -/
structure WorkScenario where
  initial_workers : ℕ
  additional_workers : ℕ
  days_saved : ℕ

/-- Calculates the original number of days required to complete the work. -/
def original_days (scenario : WorkScenario) : ℕ :=
  2 * scenario.days_saved

/-- Theorem stating that for the given scenario, the original number of days is 6. -/
theorem work_completion_days (scenario : WorkScenario) 
  (h1 : scenario.initial_workers = 10)
  (h2 : scenario.additional_workers = 10)
  (h3 : scenario.days_saved = 3) :
  original_days scenario = 6 := by
  sorry

#eval original_days { initial_workers := 10, additional_workers := 10, days_saved := 3 }

end work_completion_days_l523_52312


namespace probability_equals_three_fourteenths_l523_52309

-- Define the number of red and blue marbles
def red_marbles : ℕ := 15
def blue_marbles : ℕ := 10

-- Define the total number of marbles
def total_marbles : ℕ := red_marbles + blue_marbles

-- Define the number of marbles to be selected
def selected_marbles : ℕ := 4

-- Define the function to calculate combinations
def combination (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- Define the probability of selecting two red and two blue marbles
def probability_two_red_two_blue : ℚ :=
  (6 * combination red_marbles 2 * combination blue_marbles 2) / (combination total_marbles selected_marbles)

-- Theorem statement
theorem probability_equals_three_fourteenths : 
  probability_two_red_two_blue = 3 / 14 := by sorry

end probability_equals_three_fourteenths_l523_52309

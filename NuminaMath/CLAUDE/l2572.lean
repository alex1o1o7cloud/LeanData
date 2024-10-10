import Mathlib

namespace subset_implies_a_equals_one_l2572_257295

def A (a : ℝ) : Set ℝ := {0, -a}
def B (a : ℝ) : Set ℝ := {1, a-2, 2*a-2}

theorem subset_implies_a_equals_one (a : ℝ) : A a ⊆ B a → a = 1 := by
  sorry

end subset_implies_a_equals_one_l2572_257295


namespace condition_sufficient_not_necessary_l2572_257217

-- Define the curve C
def C (a b x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the condition for a and b
def condition (a b : ℝ) : Prop := a = 2 ∧ b = Real.sqrt 2

-- Theorem stating that the condition is sufficient but not necessary
theorem condition_sufficient_not_necessary :
  (∀ a b : ℝ, a * b ≠ 0 →
    (condition a b → C a b (Real.sqrt 2) 1)) ∧
  (∃ a b : ℝ, a * b ≠ 0 ∧ C a b (Real.sqrt 2) 1 ∧ ¬ condition a b) :=
by sorry

end condition_sufficient_not_necessary_l2572_257217


namespace cubic_factorization_l2572_257233

theorem cubic_factorization (m : ℝ) : m^3 - 6*m^2 + 9*m = m*(m-3)^2 := by
  sorry

end cubic_factorization_l2572_257233


namespace regular_polygon_perimeter_regular_polygon_perimeter_proof_l2572_257214

/-- A regular polygon with side length 6 and exterior angle 90 degrees has a perimeter of 24 units. -/
theorem regular_polygon_perimeter : ℝ → ℝ → ℝ → Prop :=
  fun (side_length : ℝ) (exterior_angle : ℝ) (perimeter : ℝ) =>
    side_length = 6 ∧
    exterior_angle = 90 ∧
    perimeter = 24

/-- The theorem statement -/
theorem regular_polygon_perimeter_proof :
  ∃ (side_length exterior_angle perimeter : ℝ),
    regular_polygon_perimeter side_length exterior_angle perimeter :=
by
  sorry

end regular_polygon_perimeter_regular_polygon_perimeter_proof_l2572_257214


namespace product_of_differences_equals_seven_l2572_257213

theorem product_of_differences_equals_seven
  (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ)
  (h₁ : x₁^3 - 3*x₁*y₁^2 = 2016)
  (h₂ : y₁^3 - 3*x₁^2*y₁ = 2000)
  (h₃ : x₂^3 - 3*x₂*y₂^2 = 2016)
  (h₄ : y₂^3 - 3*x₂^2*y₂ = 2000)
  (h₅ : x₃^3 - 3*x₃*y₃^2 = 2016)
  (h₆ : y₃^3 - 3*x₃^2*y₃ = 2000)
  (h₇ : y₁ ≠ 0)
  (h₈ : y₂ ≠ 0)
  (h₉ : y₃ ≠ 0) :
  (1 - x₁/y₁) * (1 - x₂/y₂) * (1 - x₃/y₃) = 7 := by
sorry

end product_of_differences_equals_seven_l2572_257213


namespace inequality_solution_set_l2572_257223

theorem inequality_solution_set (x : ℝ) : 
  (4 * x + 2 < (x - 1)^2 ∧ (x - 1)^2 < 6 * x + 4) ↔ 
  (4 - 2 * Real.sqrt 19 < x ∧ x < 3 + 2 * Real.sqrt 10) :=
by sorry

end inequality_solution_set_l2572_257223


namespace mod_sum_powers_seven_l2572_257265

theorem mod_sum_powers_seven : (9^5 + 4^6 + 5^7) % 7 = 2 := by
  sorry

end mod_sum_powers_seven_l2572_257265


namespace parabola_f_value_l2572_257238

/-- A parabola with equation y = dx² + ex + f, vertex at (-1, 3), and passing through (0, 2) -/
structure Parabola where
  d : ℝ
  e : ℝ
  f : ℝ
  vertex_condition : d * (-1)^2 + e * (-1) + f = 3
  point_condition : d * 0^2 + e * 0 + f = 2

/-- The f-value of the parabola is 2 -/
theorem parabola_f_value (p : Parabola) : p.f = 2 := by
  sorry

end parabola_f_value_l2572_257238


namespace appetizer_cost_per_person_l2572_257250

/-- Calculates the cost per person for a New Year's Eve appetizer --/
theorem appetizer_cost_per_person 
  (num_guests : ℕ) 
  (num_chip_bags : ℕ) 
  (chip_cost : ℚ) 
  (creme_fraiche_cost : ℚ) 
  (salmon_cost : ℚ) 
  (caviar_cost : ℚ) 
  (h1 : num_guests = 12) 
  (h2 : num_chip_bags = 10) 
  (h3 : chip_cost = 1) 
  (h4 : creme_fraiche_cost = 5) 
  (h5 : salmon_cost = 15) 
  (h6 : caviar_cost = 250) :
  (num_chip_bags * chip_cost + creme_fraiche_cost + salmon_cost + caviar_cost) / num_guests = 280 / 12 :=
by sorry

end appetizer_cost_per_person_l2572_257250


namespace persons_age_l2572_257280

theorem persons_age : ∃ (age : ℕ), 
  (5 * (age + 7) - 3 * (age - 7) = age) ∧ 
  (age > 0) := by
  sorry

end persons_age_l2572_257280


namespace circle_diameter_endpoint_l2572_257289

/-- Given a circle with center (4, -2) and one endpoint of a diameter at (1, 5),
    the other endpoint of the diameter is at (7, -9). -/
theorem circle_diameter_endpoint :
  let center : ℝ × ℝ := (4, -2)
  let endpoint1 : ℝ × ℝ := (1, 5)
  let endpoint2 : ℝ × ℝ := (7, -9)
  (endpoint1.1 - center.1 = center.1 - endpoint2.1) ∧
  (endpoint1.2 - center.2 = center.2 - endpoint2.2) :=
by sorry

end circle_diameter_endpoint_l2572_257289


namespace positive_integer_triplet_solution_l2572_257274

theorem positive_integer_triplet_solution (x y z : ℕ+) :
  (x + y)^2 + 3*x + y + 1 = z^2 → x = y ∧ z = 2*x + 1 := by
  sorry

end positive_integer_triplet_solution_l2572_257274


namespace percentage_increase_l2572_257215

theorem percentage_increase (initial : ℝ) (final : ℝ) : 
  initial = 110 → final = 165 → (final - initial) / initial * 100 = 50 := by
  sorry

end percentage_increase_l2572_257215


namespace molecular_weight_N2O5_l2572_257267

/-- Molecular weight calculation for Dinitrogen pentoxide (N2O5) -/
theorem molecular_weight_N2O5 (atomic_weight_N atomic_weight_O : ℝ)
  (h1 : atomic_weight_N = 14.01)
  (h2 : atomic_weight_O = 16.00) :
  2 * atomic_weight_N + 5 * atomic_weight_O = 108.02 := by
  sorry

#check molecular_weight_N2O5

end molecular_weight_N2O5_l2572_257267


namespace four_digit_permutations_l2572_257275

-- Define the multiset
def digit_multiset : Multiset ℕ := {3, 3, 7, 7}

-- Define the function to calculate permutations of a multiset
noncomputable def multiset_permutations (m : Multiset ℕ) : ℕ := sorry

-- Theorem statement
theorem four_digit_permutations :
  multiset_permutations digit_multiset = 6 := by sorry

end four_digit_permutations_l2572_257275


namespace root_of_series_fraction_l2572_257229

theorem root_of_series_fraction (f g : ℕ → ℝ) :
  (∀ k, f k = 8 * k^3) →
  (∀ k, g k = 27 * k^3) →
  (∑' k, f k) / (∑' k, g k) = 8 / 27 →
  ((∑' k, f k) / (∑' k, g k))^(1/3 : ℝ) = 2/3 :=
by sorry

end root_of_series_fraction_l2572_257229


namespace sequence_with_positive_triples_negative_sum_l2572_257249

theorem sequence_with_positive_triples_negative_sum : 
  ∃ (seq : Fin 20 → ℝ), 
    (∀ i : Fin 18, seq i + seq (i + 1) + seq (i + 2) > 0) ∧ 
    (Finset.sum Finset.univ seq < 0) := by
  sorry

end sequence_with_positive_triples_negative_sum_l2572_257249


namespace range_of_a_given_solution_exact_range_of_a_l2572_257254

-- Define the inequality as a function of x and a
def inequality (x a : ℝ) : Prop := 2 * x^2 + a * x - a^2 > 0

-- State the theorem
theorem range_of_a_given_solution : 
  ∀ a : ℝ, inequality 2 a → -2 < a ∧ a < 4 :=
by
  sorry

-- Define the range of a
def range_of_a : Set ℝ := { a : ℝ | -2 < a ∧ a < 4 }

-- State that this is the exact range
theorem exact_range_of_a : 
  ∀ a : ℝ, a ∈ range_of_a ↔ inequality 2 a :=
by
  sorry

end range_of_a_given_solution_exact_range_of_a_l2572_257254


namespace senate_subcommittee_combinations_l2572_257220

theorem senate_subcommittee_combinations (total_republicans : Nat) (total_democrats : Nat) 
  (subcommittee_republicans : Nat) (subcommittee_democrats : Nat) :
  total_republicans = 8 → 
  total_democrats = 6 → 
  subcommittee_republicans = 3 → 
  subcommittee_democrats = 2 → 
  (Nat.choose total_republicans subcommittee_republicans) * 
  (Nat.choose total_democrats subcommittee_democrats) = 840 := by
sorry

end senate_subcommittee_combinations_l2572_257220


namespace farm_animals_count_l2572_257232

/-- Represents a farm with hens, cows, and ducks -/
structure Farm where
  hens : ℕ
  cows : ℕ
  ducks : ℕ

/-- The total number of heads in the farm -/
def total_heads (f : Farm) : ℕ := f.hens + f.cows + f.ducks

/-- The total number of feet in the farm -/
def total_feet (f : Farm) : ℕ := 2 * f.hens + 4 * f.cows + 2 * f.ducks

/-- Theorem stating the number of cows and the sum of hens and ducks in the farm -/
theorem farm_animals_count (f : Farm) 
  (h1 : total_heads f = 72) 
  (h2 : total_feet f = 212) : 
  f.cows = 34 ∧ f.hens + f.ducks = 38 := by
  sorry


end farm_animals_count_l2572_257232


namespace overall_loss_percentage_l2572_257287

/-- Calculate the overall loss percentage for three items given their cost and selling prices -/
theorem overall_loss_percentage 
  (cp_radio cp_tv cp_blender : ℝ) 
  (sp_radio sp_tv sp_blender : ℝ) : 
  let total_cp := cp_radio + cp_tv + cp_blender
  let total_sp := sp_radio + sp_tv + sp_blender
  ((total_cp - total_sp) / total_cp) * 100 = 
    ((4500 + 8000 + 1300) - (3200 + 7500 + 1000)) / (4500 + 8000 + 1300) * 100 := by
  sorry

#eval ((4500 + 8000 + 1300) - (3200 + 7500 + 1000)) / (4500 + 8000 + 1300) * 100

end overall_loss_percentage_l2572_257287


namespace rectangle_perimeter_from_square_l2572_257292

/-- A rectangle with width and length -/
structure Rectangle where
  width : ℝ
  length : ℝ

/-- The perimeter of a rectangle -/
def Rectangle.perimeter (r : Rectangle) : ℝ :=
  2 * (r.width + r.length)

/-- A square formed by 5 identical rectangles -/
structure SquareFromRectangles where
  base_rectangle : Rectangle
  side_length : ℝ
  h_side_length : side_length = 5 * base_rectangle.width

/-- The perimeter of the square formed by rectangles -/
def SquareFromRectangles.perimeter (s : SquareFromRectangles) : ℝ :=
  4 * s.side_length

theorem rectangle_perimeter_from_square (s : SquareFromRectangles) 
    (h_perimeter_diff : s.perimeter = s.base_rectangle.perimeter + 10) :
    s.base_rectangle.perimeter = 15 := by
  sorry

end rectangle_perimeter_from_square_l2572_257292


namespace invalid_vote_percentage_l2572_257243

/-- Proves that the percentage of invalid votes is 15% given the specified conditions --/
theorem invalid_vote_percentage
  (total_votes : ℕ)
  (candidate_a_percentage : ℚ)
  (candidate_a_valid_votes : ℕ)
  (h_total : total_votes = 560000)
  (h_percentage : candidate_a_percentage = 75 / 100)
  (h_valid_votes : candidate_a_valid_votes = 357000) :
  (total_votes - (candidate_a_valid_votes / candidate_a_percentage : ℚ)) / total_votes = 15 / 100 :=
sorry

end invalid_vote_percentage_l2572_257243


namespace no_linear_term_iff_m_eq_two_l2572_257266

/-- The expression (2x-m)(x+1) does not contain a linear term of x if and only if m = 2 -/
theorem no_linear_term_iff_m_eq_two (x m : ℝ) : 
  (2 * x - m) * (x + 1) = 2 * x^2 - m ↔ m = 2 :=
by sorry

end no_linear_term_iff_m_eq_two_l2572_257266


namespace short_trees_calculation_l2572_257268

/-- The number of short trees initially in the park -/
def initial_short_trees : ℕ := 41

/-- The number of short trees to be planted -/
def planted_short_trees : ℕ := 57

/-- The total number of short trees after planting -/
def total_short_trees : ℕ := 98

/-- Theorem stating that the initial number of short trees plus the planted short trees equals the total short trees -/
theorem short_trees_calculation : 
  initial_short_trees + planted_short_trees = total_short_trees :=
by sorry

end short_trees_calculation_l2572_257268


namespace largest_power_dividing_powProduct_l2572_257260

/-- pow(n) is the largest power of the largest prime that divides n -/
def pow (n : ℕ) : ℕ :=
  sorry

/-- The product of pow(n) from 2 to 2023 -/
def powProduct : ℕ :=
  sorry

theorem largest_power_dividing_powProduct : 
  (∀ m : ℕ, 462^m ∣ powProduct → m ≤ 202) ∧ 462^202 ∣ powProduct :=
sorry

end largest_power_dividing_powProduct_l2572_257260


namespace basic_structures_correct_l2572_257200

/-- The set of basic structures of an algorithm -/
def BasicStructures : Set String :=
  {"Sequential structure", "Conditional structure", "Loop structure"}

/-- The correct answer option -/
def CorrectAnswer : Set String :=
  {"Sequential structure", "Conditional structure", "Loop structure"}

/-- Theorem stating that the basic structures of an algorithm are correctly defined -/
theorem basic_structures_correct : BasicStructures = CorrectAnswer := by
  sorry

end basic_structures_correct_l2572_257200


namespace find_x_l2572_257276

-- Define the conditions
def condition1 (x : ℕ) : Prop := 3 * x > 0
def condition2 (x : ℕ) : Prop := x ≥ 10
def condition3 (x : ℕ) : Prop := x > 5

-- Theorem statement
theorem find_x : ∃ (x : ℕ), condition1 x ∧ condition2 x ∧ condition3 x ∧ x = 9 := by
  sorry

end find_x_l2572_257276


namespace container_volume_ratio_l2572_257269

theorem container_volume_ratio : 
  ∀ (v1 v2 : ℝ), v1 > 0 → v2 > 0 → 
  (3/4 : ℝ) * v1 = (5/8 : ℝ) * v2 → 
  v1 / v2 = (5/6 : ℝ) :=
by
  sorry

end container_volume_ratio_l2572_257269


namespace fraction_to_decimal_l2572_257235

theorem fraction_to_decimal : (58 : ℚ) / 125 = (464 : ℚ) / 1000 := by sorry

end fraction_to_decimal_l2572_257235


namespace x_percent_of_x_squared_is_nine_l2572_257205

theorem x_percent_of_x_squared_is_nine (x : ℝ) (h1 : x > 0) (h2 : x / 100 * x^2 = 9) : x = 10 * Real.rpow 3 (1/3) := by
  sorry

end x_percent_of_x_squared_is_nine_l2572_257205


namespace cube_occupation_percentage_l2572_257241

/-- Represents the dimensions of a rectangular box in inches -/
structure BoxDimensions where
  length : ℚ
  width : ℚ
  height : ℚ

/-- Represents the side length of a cube in inches -/
def CubeSideLength : ℚ := 3

/-- The dimensions of the given box -/
def givenBox : BoxDimensions := ⟨6, 5, 10⟩

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (box : BoxDimensions) : ℚ :=
  box.length * box.width * box.height

/-- Calculates the largest dimensions that can be filled with cubes -/
def largestFillableDimensions (box : BoxDimensions) (cubeSize : ℚ) : BoxDimensions :=
  ⟨
    (box.length / cubeSize).floor * cubeSize,
    (box.width / cubeSize).floor * cubeSize,
    (box.height / cubeSize).floor * cubeSize
  ⟩

/-- Calculates the percentage of the box volume occupied by cubes -/
def percentageOccupied (box : BoxDimensions) (cubeSize : ℚ) : ℚ :=
  let fillableBox := largestFillableDimensions box cubeSize
  (boxVolume fillableBox) / (boxVolume box) * 100

theorem cube_occupation_percentage :
  percentageOccupied givenBox CubeSideLength = 54 := by
  sorry

end cube_occupation_percentage_l2572_257241


namespace problem_statement_l2572_257290

theorem problem_statement : (-1 : ℤ) ^ (4 ^ 3) + 2 ^ (3 ^ 2) = 513 := by
  sorry

end problem_statement_l2572_257290


namespace range_of_a_l2572_257270

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 + x - 6 < 0}
def B (a : ℝ) : Set ℝ := {x : ℝ | x > a}

-- Define the sufficient condition
def sufficient_condition (a : ℝ) : Prop := ∀ x, x ∈ A → x ∈ B a

-- Theorem statement
theorem range_of_a :
  ∀ a : ℝ, sufficient_condition a ↔ a ≤ -3 :=
sorry

end range_of_a_l2572_257270


namespace megan_carrots_count_l2572_257230

/-- Calculates the total number of carrots Megan has after picking, throwing out, and picking again. -/
def total_carrots (initial : ℕ) (thrown_out : ℕ) (new_picked : ℕ) : ℕ :=
  initial - thrown_out + new_picked

/-- Proves that Megan's total carrots is 61 given the specific numbers in the problem. -/
theorem megan_carrots_count :
  total_carrots 19 4 46 = 61 := by
  sorry

end megan_carrots_count_l2572_257230


namespace inverse_cube_root_relation_l2572_257201

/-- Given that z varies inversely as ∛w, prove that w = 1 when z = 6, 
    given that z = 3 when w = 8. -/
theorem inverse_cube_root_relation (z w : ℝ) (k : ℝ) 
  (h1 : ∀ w z, z * (w ^ (1/3 : ℝ)) = k)
  (h2 : 3 * (8 ^ (1/3 : ℝ)) = k)
  (h3 : 6 * (w ^ (1/3 : ℝ)) = k) : 
  w = 1 := by sorry

end inverse_cube_root_relation_l2572_257201


namespace triangulation_count_equals_catalan_l2572_257252

/-- The number of ways to triangulate a convex polygon -/
def triangulationCount (n : ℕ) : ℕ := sorry

/-- The n-th Catalan number -/
def catalanNumber (n : ℕ) : ℕ := sorry

/-- Theorem: The number of ways to triangulate a convex (n+2)-gon
    is equal to the (n-1)-th Catalan number -/
theorem triangulation_count_equals_catalan (n : ℕ) :
  triangulationCount (n + 2) = catalanNumber (n - 1) := by sorry

end triangulation_count_equals_catalan_l2572_257252


namespace trivia_game_points_per_question_l2572_257247

theorem trivia_game_points_per_question 
  (first_half_correct : ℕ) 
  (second_half_correct : ℕ) 
  (final_score : ℕ) 
  (h1 : first_half_correct = 5)
  (h2 : second_half_correct = 5)
  (h3 : final_score = 50) :
  final_score / (first_half_correct + second_half_correct) = 5 := by
sorry

end trivia_game_points_per_question_l2572_257247


namespace charles_total_money_l2572_257293

-- Define the value of each coin type in cents
def penny_value : ℕ := 1
def nickel_value : ℕ := 5
def dime_value : ℕ := 10
def quarter_value : ℕ := 25

-- Define the number of coins Charles found on his way to school
def found_pennies : ℕ := 6
def found_nickels : ℕ := 4
def found_dimes : ℕ := 3

-- Define the number of coins Charles already had at home
def home_nickels : ℕ := 3
def home_dimes : ℕ := 2
def home_quarters : ℕ := 1

-- Calculate the total value in cents
def total_cents : ℕ :=
  found_pennies * penny_value +
  (found_nickels + home_nickels) * nickel_value +
  (found_dimes + home_dimes) * dime_value +
  home_quarters * quarter_value

-- Theorem to prove
theorem charles_total_money :
  total_cents = 116 := by sorry

end charles_total_money_l2572_257293


namespace min_value_theorem_l2572_257209

theorem min_value_theorem (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a + b) / c + (b + c) / a + (c + d) / b ≥ 6 ∧
  ((a + b) / c + (b + c) / a + (c + d) / b = 6 ↔ a = b ∧ b = c ∧ c = d) :=
sorry

end min_value_theorem_l2572_257209


namespace largest_satisfying_number_l2572_257221

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

def satisfies_condition (n : ℕ) : Prop :=
  ∀ p : ℕ, is_prime p → p % 2 = 1 → p < n → is_prime (n - p)

theorem largest_satisfying_number :
  satisfies_condition 10 ∧ ∀ n : ℕ, n > 10 → ¬(satisfies_condition n) :=
sorry

end largest_satisfying_number_l2572_257221


namespace parabola_focus_vertex_distance_l2572_257224

/-- A parabola with vertex V and focus F -/
structure Parabola where
  V : ℝ × ℝ  -- Vertex
  F : ℝ × ℝ  -- Focus

/-- A point on a parabola -/
structure ParabolaPoint (p : Parabola) where
  point : ℝ × ℝ
  on_parabola : sorry  -- Condition for the point to be on the parabola

/-- Distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

theorem parabola_focus_vertex_distance 
  (p : Parabola) 
  (A : ParabolaPoint p) 
  (h1 : distance A.point p.F = 18) 
  (h2 : distance A.point p.V = 19) : 
  distance p.F p.V = Real.sqrt 37 := by
  sorry

end parabola_focus_vertex_distance_l2572_257224


namespace pair_probability_after_removal_l2572_257219

def initial_deck_size : ℕ := 52
def cards_per_value : ℕ := 4
def values_count : ℕ := 13
def removed_pairs : ℕ := 2

def remaining_deck_size : ℕ := initial_deck_size - removed_pairs * cards_per_value / 2

def total_ways_to_select_two : ℕ := remaining_deck_size.choose 2

def full_value_count : ℕ := values_count - removed_pairs
def pair_forming_ways_full : ℕ := full_value_count * (cards_per_value.choose 2)
def pair_forming_ways_reduced : ℕ := removed_pairs * ((cards_per_value - 2).choose 2)
def total_pair_forming_ways : ℕ := pair_forming_ways_full + pair_forming_ways_reduced

theorem pair_probability_after_removal : 
  (total_pair_forming_ways : ℚ) / total_ways_to_select_two = 17 / 282 := by sorry

end pair_probability_after_removal_l2572_257219


namespace x_intercept_is_one_l2572_257262

/-- A line passing through two points (x₁, y₁) and (x₂, y₂) -/
structure Line where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ

/-- The x-intercept of a line -/
def x_intercept (l : Line) : ℝ :=
  sorry

/-- The theorem stating that the x-intercept of the given line is 1 -/
theorem x_intercept_is_one :
  let l : Line := { x₁ := 2, y₁ := -2, x₂ := -1, y₂ := 4 }
  x_intercept l = 1 := by
  sorry

end x_intercept_is_one_l2572_257262


namespace square_root_sum_l2572_257259

theorem square_root_sum (y : ℝ) :
  Real.sqrt (64 - y^2) - Real.sqrt (36 - y^2) = 4 →
  Real.sqrt (64 - y^2) + Real.sqrt (36 - y^2) = 7 := by
sorry

end square_root_sum_l2572_257259


namespace combined_box_weight_l2572_257261

def box1_weight : ℕ := 2
def box2_weight : ℕ := 11
def box3_weight : ℕ := 5

theorem combined_box_weight :
  box1_weight + box2_weight + box3_weight = 18 := by
  sorry

end combined_box_weight_l2572_257261


namespace new_average_score_l2572_257202

theorem new_average_score (n : ℕ) (a s : ℚ) (h1 : n = 4) (h2 : a = 78) (h3 : s = 88) :
  (n * a + s) / (n + 1) = 80 := by
  sorry

end new_average_score_l2572_257202


namespace largest_multiple_of_15_under_400_l2572_257218

theorem largest_multiple_of_15_under_400 : ∃ (n : ℕ), n * 15 = 390 ∧ 
  390 < 400 ∧ 
  ∀ (m : ℕ), m * 15 < 400 → m * 15 ≤ 390 := by
  sorry

end largest_multiple_of_15_under_400_l2572_257218


namespace multiples_of_seven_between_15_and_200_l2572_257283

theorem multiples_of_seven_between_15_and_200 : 
  (Finset.filter (fun n => n % 7 = 0 ∧ n > 15 ∧ n < 200) (Finset.range 200)).card = 26 := by
  sorry

end multiples_of_seven_between_15_and_200_l2572_257283


namespace square_side_length_from_voice_range_l2572_257272

/-- The side length of a square ground, given the area of a quarter circle
    representing the range of a trainer's voice from one corner. -/
theorem square_side_length_from_voice_range (r : ℝ) (area : ℝ) 
    (h1 : r = 140)
    (h2 : area = 15393.804002589986)
    (h3 : area = (π * r^2) / 4) : 
  ∃ (s : ℝ), s^2 = r^2 ∧ s = 140 := by
  sorry

end square_side_length_from_voice_range_l2572_257272


namespace y_value_proof_l2572_257271

/-- Given that 150% of x is equal to 75% of y and x = 24, prove that y = 48 -/
theorem y_value_proof (x y : ℝ) (h1 : 1.5 * x = 0.75 * y) (h2 : x = 24) : y = 48 := by
  sorry

end y_value_proof_l2572_257271


namespace square_sum_given_sum_and_product_l2572_257255

theorem square_sum_given_sum_and_product (x y : ℝ) 
  (h1 : x + y = 7) (h2 : x * y = 5) : x^2 + y^2 = 39 := by
  sorry

end square_sum_given_sum_and_product_l2572_257255


namespace z_in_first_quadrant_l2572_257294

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the given equation
def equation (z : ℂ) : Prop := z * (1 - i) = 2 - i

-- Theorem statement
theorem z_in_first_quadrant (z : ℂ) (h : equation z) : 
  z.re > 0 ∧ z.im > 0 := by
  sorry

end z_in_first_quadrant_l2572_257294


namespace game_winner_conditions_l2572_257296

/-- Represents the possible outcomes of the game -/
inductive GameOutcome
  | AWins
  | BWins

/-- Represents the game state -/
structure GameState where
  n : ℕ
  m : ℕ
  currentPlayer : Bool  -- true for A, false for B

/-- Determines the winner of the game given the initial state -/
def determineWinner (initialState : GameState) : GameOutcome :=
  if initialState.n = initialState.m then
    GameOutcome.BWins
  else
    GameOutcome.AWins

/-- Theorem stating the winning conditions for the game -/
theorem game_winner_conditions (n m : ℕ) (hn : n > 1) (hm : m > 1) :
  let initialState := GameState.mk n m true
  determineWinner initialState =
    if n = m then
      GameOutcome.BWins
    else
      GameOutcome.AWins :=
by
  sorry


end game_winner_conditions_l2572_257296


namespace algebraic_expression_value_l2572_257210

theorem algebraic_expression_value (a b : ℝ) (h1 : a - b = 5) (h2 : a * b = -1) :
  (2 * a + 3 * b - 2 * a * b) - (a + 4 * b + a * b) - (3 * a * b + 2 * b - 2 * a) = 21 := by
  sorry

end algebraic_expression_value_l2572_257210


namespace mirror_to_wall_area_ratio_l2572_257216

/-- The ratio of a square mirror's area to a rectangular wall's area --/
theorem mirror_to_wall_area_ratio
  (mirror_side : ℝ)
  (wall_width : ℝ)
  (wall_length : ℝ)
  (h1 : mirror_side = 24)
  (h2 : wall_width = 42)
  (h3 : wall_length = 27.428571428571427)
  : (mirror_side ^ 2) / (wall_width * wall_length) = 0.5 := by
  sorry

end mirror_to_wall_area_ratio_l2572_257216


namespace roman_numeral_calculation_l2572_257291

/-- Roman numeral values -/
def I : ℕ := 1
def V : ℕ := 5
def X : ℕ := 10
def L : ℕ := 50
def C : ℕ := 100
def D : ℕ := 500
def M : ℕ := 1000

/-- The theorem to prove -/
theorem roman_numeral_calculation : 2 * M + 5 * L + 7 * X + 9 * I = 2329 := by
  sorry

end roman_numeral_calculation_l2572_257291


namespace four_students_arrangement_l2572_257257

/-- The number of ways to arrange n students in a line -/
def lineArrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n students in a line with one specific student at either end -/
def arrangementsWithOneAtEnd (n : ℕ) : ℕ := 
  2 * lineArrangements (n - 1)

/-- Theorem: There are 12 ways to arrange 4 students in a line with one specific student at either end -/
theorem four_students_arrangement : arrangementsWithOneAtEnd 4 = 12 := by
  sorry

end four_students_arrangement_l2572_257257


namespace min_sum_abc_l2572_257284

theorem min_sum_abc (a b c : ℕ+) (h : (a : ℚ) / 77 + (b : ℚ) / 91 + (c : ℚ) / 143 = 1) :
  ∃ (a' b' c' : ℕ+), (a' : ℚ) / 77 + (b' : ℚ) / 91 + (c' : ℚ) / 143 = 1 ∧
    (∀ (x y z : ℕ+), (x : ℚ) / 77 + (y : ℚ) / 91 + (z : ℚ) / 143 = 1 → 
      a' + b' + c' ≤ x + y + z) ∧
    a' + b' + c' = 79 :=
by sorry

end min_sum_abc_l2572_257284


namespace max_k_inequality_l2572_257258

theorem max_k_inequality (x y : ℝ) (hx : x ≥ 1) (hy : y ≥ 1) :
  (∃ k : ℝ, ∀ k' : ℝ, 
    (x^2 / (1 + x) + y^2 / (1 + y) + (x - 1) * (y - 1) ≥ k' * x * y) → k' ≤ k) ∧
  (x^2 / (1 + x) + y^2 / (1 + y) + (x - 1) * (y - 1) ≥ ((13 - 5 * Real.sqrt 5) / 2) * x * y) :=
by sorry

end max_k_inequality_l2572_257258


namespace otimes_nested_equality_l2572_257212

/-- Custom binary operation ⊗ -/
def otimes (x y : ℝ) : ℝ := x^3 + 5*x*y - y

theorem otimes_nested_equality (a : ℝ) : otimes a (otimes a a) = 5*a^4 + 24*a^3 - 10*a^2 + a := by
  sorry

end otimes_nested_equality_l2572_257212


namespace parallelogram_base_given_triangle_l2572_257207

/-- Given a triangle and a parallelogram with equal areas and the same height,
    if the base of the triangle is 24 inches, then the base of the parallelogram is 12 inches. -/
theorem parallelogram_base_given_triangle (h : ℝ) (b_p : ℝ) : 
  (1/2 * 24 * h = b_p * h) → b_p = 12 := by
  sorry

end parallelogram_base_given_triangle_l2572_257207


namespace hemisphere_radius_from_cylinder_l2572_257204

/-- The radius of a hemisphere formed from a cylinder of equal volume --/
theorem hemisphere_radius_from_cylinder (r h R : ℝ) : 
  r = 2 * (2 : ℝ)^(1/3) → 
  h = 12 → 
  π * r^2 * h = (2/3) * π * R^3 → 
  R = 2 * 3^(1/3) := by
sorry

end hemisphere_radius_from_cylinder_l2572_257204


namespace weight_estimate_for_178cm_l2572_257263

/-- Regression equation for weight based on height -/
def weight_regression (height : ℝ) : ℝ := 0.72 * height - 58.5

/-- The problem statement -/
theorem weight_estimate_for_178cm :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.5 ∧ |weight_regression 178 - 70| < ε :=
sorry

end weight_estimate_for_178cm_l2572_257263


namespace weight_loss_calculation_l2572_257239

/-- Represents the weight loss calculation problem --/
theorem weight_loss_calculation 
  (current_weight : ℕ) 
  (previous_weight : ℕ) 
  (h1 : current_weight = 27) 
  (h2 : previous_weight = 128) :
  previous_weight - current_weight = 101 := by
  sorry

end weight_loss_calculation_l2572_257239


namespace quadratic_root_implies_v_value_l2572_257277

theorem quadratic_root_implies_v_value : ∀ v : ℝ,
  ((-25 - Real.sqrt 361) / 12 : ℝ) ∈ {x : ℝ | 6 * x^2 + 25 * x + v = 0} →
  v = 11 := by
sorry

end quadratic_root_implies_v_value_l2572_257277


namespace max_value_x_plus_reciprocal_l2572_257237

theorem max_value_x_plus_reciprocal (x : ℝ) (h : 11 = x^2 + 1/x^2) :
  ∃ (y : ℝ), y = x + 1/x ∧ y ≤ Real.sqrt 13 ∧ ∃ (z : ℝ), z = x + 1/x ∧ z = Real.sqrt 13 := by
  sorry

end max_value_x_plus_reciprocal_l2572_257237


namespace brent_nerds_count_l2572_257206

/-- Represents the candy inventory of Brent --/
structure CandyInventory where
  kitKat : ℕ
  hersheyKisses : ℕ
  nerds : ℕ
  lollipops : ℕ
  babyRuths : ℕ
  reeseCups : ℕ

/-- Calculates the total number of candy pieces --/
def totalCandy (inventory : CandyInventory) : ℕ :=
  inventory.kitKat + inventory.hersheyKisses + inventory.nerds + 
  inventory.lollipops + inventory.babyRuths + inventory.reeseCups

/-- Theorem stating that Brent received 8 boxes of Nerds --/
theorem brent_nerds_count : ∃ (inventory : CandyInventory),
  inventory.kitKat = 5 ∧
  inventory.hersheyKisses = 3 * inventory.kitKat ∧
  inventory.lollipops = 11 ∧
  inventory.babyRuths = 10 ∧
  inventory.reeseCups = inventory.babyRuths / 2 ∧
  totalCandy inventory - 5 = 49 ∧
  inventory.nerds = 8 := by
  sorry

end brent_nerds_count_l2572_257206


namespace D_144_l2572_257242

/-- 
D(n) represents the number of ways to write a positive integer n as a product of 
integers greater than 1, where the order of factors matters.
-/
def D (n : ℕ+) : ℕ := sorry

/-- The main theorem stating that D(144) = 41 -/
theorem D_144 : D 144 = 41 := by sorry

end D_144_l2572_257242


namespace max_knights_and_courtiers_l2572_257211

def king_table_size : ℕ := 7
def min_courtiers : ℕ := 12
def max_courtiers : ℕ := 18
def min_knights : ℕ := 10
def max_knights : ℕ := 20

def is_valid_solution (courtiers knights : ℕ) : Prop :=
  min_courtiers ≤ courtiers ∧ courtiers ≤ max_courtiers ∧
  min_knights ≤ knights ∧ knights ≤ max_knights ∧
  (1 : ℚ) / courtiers + (1 : ℚ) / knights = (1 : ℚ) / king_table_size

theorem max_knights_and_courtiers :
  ∃ (max_knights courtiers : ℕ),
    is_valid_solution courtiers max_knights ∧
    ∀ (k : ℕ), is_valid_solution courtiers k → k ≤ max_knights ∧
    max_knights = 14 ∧ courtiers = 14 := by
  sorry

end max_knights_and_courtiers_l2572_257211


namespace computer_cost_l2572_257279

theorem computer_cost (total_budget fridge_cost tv_cost computer_cost : ℕ) : 
  total_budget = 1600 →
  tv_cost = 600 →
  fridge_cost = computer_cost + 500 →
  total_budget = tv_cost + fridge_cost + computer_cost →
  computer_cost = 250 := by
sorry

end computer_cost_l2572_257279


namespace sqrt_one_eighth_same_type_as_sqrt_two_l2572_257297

theorem sqrt_one_eighth_same_type_as_sqrt_two :
  ∃ (q : ℚ), Real.sqrt (1/8 : ℝ) = q * Real.sqrt 2 :=
sorry

end sqrt_one_eighth_same_type_as_sqrt_two_l2572_257297


namespace inscribed_square_arc_length_l2572_257245

/-- Given a square inscribed in a circle with side length 4,
    the arc length intercepted by any side of the square is √2π. -/
theorem inscribed_square_arc_length (s : Real) (r : Real) (arc_length : Real) :
  s = 4 →                        -- Side length of the square is 4
  r = 2 * Real.sqrt 2 →          -- Radius of the circle
  arc_length = Real.sqrt 2 * π → -- Arc length intercepted by any side
  True :=
by sorry

end inscribed_square_arc_length_l2572_257245


namespace red_bottle_caps_l2572_257231

theorem red_bottle_caps (total : ℕ) (green_percentage : ℚ) : 
  total = 125 → green_percentage = 60 / 100 → 
  (total : ℚ) * (1 - green_percentage) = 50 := by
sorry

end red_bottle_caps_l2572_257231


namespace family_photos_l2572_257246

theorem family_photos (total : ℕ) (friends : ℕ) (family : ℕ) 
  (h1 : total = 86) 
  (h2 : friends = 63) 
  (h3 : total = friends + family) : family = 23 := by
  sorry

end family_photos_l2572_257246


namespace competition_score_difference_l2572_257228

def score_60_percent : Real := 0.12
def score_85_percent : Real := 0.20
def score_95_percent : Real := 0.38
def score_105_percent : Real := 1 - (score_60_percent + score_85_percent + score_95_percent)

def mean_score : Real :=
  score_60_percent * 60 + score_85_percent * 85 + score_95_percent * 95 + score_105_percent * 105

def median_score : Real := 95

theorem competition_score_difference : median_score - mean_score = 3.2 := by
  sorry

end competition_score_difference_l2572_257228


namespace logan_snowfall_total_l2572_257286

/-- Represents the snowfall recorded over three days during a snowstorm -/
structure SnowfallRecord where
  wednesday : ℝ
  thursday : ℝ
  friday : ℝ

/-- Calculates the total snowfall from a three-day record -/
def totalSnowfall (record : SnowfallRecord) : ℝ :=
  record.wednesday + record.thursday + record.friday

/-- Theorem stating that Logan's recorded snowfall totals 0.88 cm -/
theorem logan_snowfall_total :
  let record : SnowfallRecord := {
    wednesday := 0.33,
    thursday := 0.33,
    friday := 0.22
  }
  totalSnowfall record = 0.88 := by
  sorry

end logan_snowfall_total_l2572_257286


namespace quadratic_equation_with_root_three_l2572_257299

theorem quadratic_equation_with_root_three :
  ∃ (a b c : ℝ), a ≠ 0 ∧ (∀ x, a * x^2 + b * x + c = 0 ↔ x^2 - 3*x = 0) ∧ (3 : ℝ) ∈ {x : ℝ | a * x^2 + b * x + c = 0} :=
sorry

end quadratic_equation_with_root_three_l2572_257299


namespace chips_in_bag_is_81_l2572_257285

/-- Represents the number of chocolate chips in a bag -/
def chips_in_bag : ℕ := sorry

/-- Represents the number of batches made from one bag of chips -/
def batches_per_bag : ℕ := 3

/-- Represents the number of cookies in each batch -/
def cookies_per_batch : ℕ := 3

/-- Represents the number of chocolate chips in each cookie -/
def chips_per_cookie : ℕ := 9

/-- Theorem stating that the number of chips in a bag is 81 -/
theorem chips_in_bag_is_81 : chips_in_bag = 81 := by sorry

end chips_in_bag_is_81_l2572_257285


namespace negation_of_existence_negation_of_quadratic_inequality_l2572_257253

theorem negation_of_existence (f : ℝ → Prop) : 
  (¬ ∃ x, f x) ↔ (∀ x, ¬ f x) :=
by sorry

theorem negation_of_quadratic_inequality : 
  (¬ ∃ x : ℝ, x^2 - 2*x - 3 < 0) ↔ (∀ x : ℝ, x^2 - 2*x - 3 ≥ 0) :=
by sorry

end negation_of_existence_negation_of_quadratic_inequality_l2572_257253


namespace christmas_sale_pricing_l2572_257226

/-- Represents the discount rate as a fraction -/
def discount_rate : ℚ := 2/5

/-- Calculates the sale price given the original price and discount rate -/
def sale_price (original_price : ℚ) : ℚ := original_price * (1 - discount_rate)

/-- Calculates the original price given the sale price and discount rate -/
def original_price (sale_price : ℚ) : ℚ := sale_price / (1 - discount_rate)

theorem christmas_sale_pricing (a b : ℚ) :
  sale_price a = 3/5 * a ∧ original_price b = 5/3 * b := by
  sorry

end christmas_sale_pricing_l2572_257226


namespace intersection_complement_equality_l2572_257251

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def M : Set Nat := {1, 4}
def N : Set Nat := {1, 3, 5}

theorem intersection_complement_equality : N ∩ (U \ M) = {3, 5} := by
  sorry

end intersection_complement_equality_l2572_257251


namespace smallest_number_l2572_257282

/-- Converts a number from base b to decimal --/
def to_decimal (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * b^i) 0

/-- The number 85 in base 9 --/
def n1 : Nat := to_decimal [5, 8] 9

/-- The number 210 in base 6 --/
def n2 : Nat := to_decimal [0, 1, 2] 6

/-- The number 1000 in base 4 --/
def n3 : Nat := to_decimal [0, 0, 0, 1] 4

/-- The number 111111 in base 2 --/
def n4 : Nat := to_decimal [1, 1, 1, 1, 1, 1] 2

theorem smallest_number : n4 = min n1 (min n2 (min n3 n4)) := by
  sorry

end smallest_number_l2572_257282


namespace power_equality_l2572_257256

theorem power_equality (n m : ℕ) (h1 : 4^n = 3) (h2 : 8^m = 5) : 2^(2*n + 3*m) = 15 := by
  sorry

end power_equality_l2572_257256


namespace job_completion_time_l2572_257208

/-- Represents the number of days needed to complete a job given initial and additional workers -/
def days_to_complete_job (initial_workers : ℕ) (initial_days : ℕ) (total_work_days : ℕ) 
  (days_before_joining : ℕ) (additional_workers : ℕ) : ℕ :=
  let total_workers := initial_workers + additional_workers
  let work_done := initial_workers * days_before_joining
  let remaining_work := total_work_days - work_done
  days_before_joining + (remaining_work + total_workers - 1) / total_workers

/-- Theorem stating that under the given conditions, the job will be completed in 6 days -/
theorem job_completion_time :
  days_to_complete_job 6 8 48 3 4 = 6 := by
  sorry

end job_completion_time_l2572_257208


namespace not_in_first_quadrant_l2572_257236

def linear_function (x : ℝ) : ℝ := -3 * x - 2

theorem not_in_first_quadrant :
  ∀ x y : ℝ, y = linear_function x → ¬(x > 0 ∧ y > 0) := by
  sorry

end not_in_first_quadrant_l2572_257236


namespace sequence_sum_l2572_257227

/-- Given a geometric sequence a, b, c and arithmetic sequences a, x, b and b, y, c, 
    prove that a/x + c/y = 2 -/
theorem sequence_sum (a b c x y : ℝ) 
  (h_geom : b^2 = a*c) 
  (h_arith1 : x = (a + b)/2) 
  (h_arith2 : y = (b + c)/2) : 
  a/x + c/y = 2 := by
  sorry

end sequence_sum_l2572_257227


namespace tangent_circles_m_value_l2572_257225

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 4
def circle2 (x y m : ℝ) : Prop := x^2 + y^2 - 2*m*x + m^2 - 1 = 0

-- Define what it means for two circles to be externally tangent
def externally_tangent (m : ℝ) : Prop :=
  ∃ (x y : ℝ), circle1 x y ∧ circle2 x y m ∧
  ∀ (x' y' : ℝ), circle1 x' y' ∧ circle2 x' y' m → (x = x' ∧ y = y')

-- State the theorem
theorem tangent_circles_m_value (m : ℝ) :
  externally_tangent m → (m = 3 ∨ m = -3) :=
sorry

end tangent_circles_m_value_l2572_257225


namespace monica_reading_plan_l2572_257234

def books_last_year : ℕ := 16
def books_this_year : ℕ := 2 * books_last_year
def books_next_year : ℕ := 69

theorem monica_reading_plan :
  books_next_year - (2 * books_this_year) = 5 := by sorry

end monica_reading_plan_l2572_257234


namespace tangent_line_parallel_implies_a_and_b_l2572_257278

/-- The function f(x) = x³ + ax² + b -/
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b

/-- The derivative of f(x) -/
def f_deriv (a x : ℝ) : ℝ := 3*x^2 + 2*a*x

theorem tangent_line_parallel_implies_a_and_b (a b : ℝ) : 
  f a b 1 = 0 ∧ f_deriv a 1 = -3 → a = -3 ∧ b = 2 := by
  sorry

#check tangent_line_parallel_implies_a_and_b

end tangent_line_parallel_implies_a_and_b_l2572_257278


namespace no_integer_satisfies_conditions_l2572_257222

theorem no_integer_satisfies_conditions : 
  ¬∃ (n : ℤ), ∃ (k : ℤ), 
    n / (25 - n) = k^2 ∧ 
    ∃ (m : ℤ), n = 3 * m :=
by sorry

end no_integer_satisfies_conditions_l2572_257222


namespace average_equals_x_l2572_257288

theorem average_equals_x (x : ℝ) : 
  (2 + 5 + x + 14 + 15) / 5 = x → x = 9 := by
  sorry

end average_equals_x_l2572_257288


namespace simplify_expression_l2572_257248

theorem simplify_expression : ((3 + 4 + 5 + 6) / 2) + ((3 * 6 + 9) / 3) = 18 := by
  sorry

end simplify_expression_l2572_257248


namespace miranda_pillows_l2572_257203

-- Define the constants
def feathers_per_pillow : ℕ := 2
def goose_feathers_per_pound : ℕ := 300
def duck_feathers_per_pound : ℕ := 500
def goose_total_feathers : ℕ := 3600
def duck_total_feathers : ℕ := 4000

-- Theorem statement
theorem miranda_pillows :
  let goose_pounds : ℕ := goose_total_feathers / goose_feathers_per_pound
  let duck_pounds : ℕ := duck_total_feathers / duck_feathers_per_pound
  let total_pounds : ℕ := goose_pounds + duck_pounds
  let pillows : ℕ := total_pounds / feathers_per_pillow
  pillows = 10 := by sorry

end miranda_pillows_l2572_257203


namespace max_distance_between_sine_cosine_curves_l2572_257240

theorem max_distance_between_sine_cosine_curves : ∃ (C : ℝ),
  (∀ (m : ℝ), |2 * Real.sin m - 2 * Real.sqrt 3 * Real.cos m| ≤ C) ∧
  (∃ (m : ℝ), |2 * Real.sin m - 2 * Real.sqrt 3 * Real.cos m| = C) ∧
  C = 4 := by
  sorry

end max_distance_between_sine_cosine_curves_l2572_257240


namespace min_A_over_C_l2572_257273

theorem min_A_over_C (x A C : ℝ) (hx : x > 0) (hA : A > 0) (hC : C > 0)
  (hdefA : x^2 + 1/x^2 = A) (hdefC : x + 1/x = C) :
  ∃ (m : ℝ), m = 2 * Real.sqrt 2 ∧ ∀ y, y = A / C → y ≥ m := by
  sorry

end min_A_over_C_l2572_257273


namespace function_identity_l2572_257264

-- Define the property that f must satisfy
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + f y) + f (x * y) = y * f x + f y + f (f x)

-- State the theorem
theorem function_identity {f : ℝ → ℝ} (h : SatisfiesEquation f) :
  ∀ x : ℝ, f x = x :=
by sorry

end function_identity_l2572_257264


namespace ellipse_foci_distance_l2572_257298

theorem ellipse_foci_distance (a b : ℝ) (ha : a = 8) (hb : b = 3) :
  let c := Real.sqrt (a^2 - b^2)
  2 * c = 2 * Real.sqrt 55 := by sorry

end ellipse_foci_distance_l2572_257298


namespace negation_of_p_negation_of_q_l2572_257244

-- Define the statement p
def p : Prop := ∀ x : ℝ, x > 0 → x^2 - 5*x ≥ -25/4

-- Define the statement q
def q : Prop := ∃ n : ℕ, Even n ∧ n % 3 = 0

-- Theorem for the negation of p
theorem negation_of_p : (¬p) ↔ (∃ x : ℝ, x > 0 ∧ x^2 - 5*x < -25/4) :=
sorry

-- Theorem for the negation of q
theorem negation_of_q : (¬q) ↔ (∀ n : ℕ, Even n → n % 3 ≠ 0) :=
sorry

end negation_of_p_negation_of_q_l2572_257244


namespace polynomial_real_root_iff_b_ge_half_l2572_257281

/-- The polynomial p(x) = x^4 + bx^3 + x^2 + bx - 1 -/
def p (b : ℝ) (x : ℝ) : ℝ := x^4 + b*x^3 + x^2 + b*x - 1

/-- The polynomial p(x) has at least one real root -/
def has_real_root (b : ℝ) : Prop := ∃ x : ℝ, p b x = 0

theorem polynomial_real_root_iff_b_ge_half :
  ∀ b : ℝ, has_real_root b ↔ b ≥ 1/2 := by sorry

end polynomial_real_root_iff_b_ge_half_l2572_257281

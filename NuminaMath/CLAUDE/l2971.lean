import Mathlib

namespace min_distance_to_circle_l2971_297199

theorem min_distance_to_circle (x y : ℝ) : 
  (x - 2)^2 + (y - 1)^2 = 1 → x^2 + y^2 ≥ 6 - 2 * Real.sqrt 5 := by
  sorry

end min_distance_to_circle_l2971_297199


namespace interior_angle_regular_hexagon_l2971_297127

/-- The measure of an interior angle of a regular hexagon -/
def interior_angle_measure_hexagon : ℝ := 120

/-- A regular hexagon has 6 sides -/
def hexagon_sides : ℕ := 6

/-- Theorem: The measure of each interior angle in a regular hexagon is 120° -/
theorem interior_angle_regular_hexagon :
  interior_angle_measure_hexagon = (((hexagon_sides - 2 : ℕ) * 180) / hexagon_sides : ℝ) :=
by sorry

end interior_angle_regular_hexagon_l2971_297127


namespace parallel_lines_length_l2971_297119

-- Define the parallel lines and their lengths
def AB : ℝ := 120
def CD : ℝ := 80
def GH : ℝ := 140

-- Define the property of parallel lines
def parallel (a b c d : ℝ) : Prop := sorry

-- Theorem statement
theorem parallel_lines_length (EF : ℝ) 
  (h1 : parallel AB CD EF GH) : EF = 80 := by
  sorry

end parallel_lines_length_l2971_297119


namespace intersection_circle_regions_l2971_297121

/-- The maximum number of regions in the intersection of n circles -/
def max_regions (n : ℕ) : ℕ :=
  2 * n - 2

/-- Theorem stating the maximum number of regions in the intersection of n circles -/
theorem intersection_circle_regions (n : ℕ) (h : n ≥ 2) :
  max_regions n = 2 * n - 2 := by
  sorry

#check intersection_circle_regions

end intersection_circle_regions_l2971_297121


namespace rectangle_area_diagonal_l2971_297139

theorem rectangle_area_diagonal (d : ℝ) (h : d > 0) : ∃ (l w : ℝ),
  l > 0 ∧ w > 0 ∧ l / w = 5 / 2 ∧ l ^ 2 + w ^ 2 = d ^ 2 ∧ l * w = (10 / 29) * d ^ 2 := by
  sorry

end rectangle_area_diagonal_l2971_297139


namespace opposite_of_eight_l2971_297161

theorem opposite_of_eight : 
  -(8 : ℤ) = -8 := by sorry

end opposite_of_eight_l2971_297161


namespace greatest_integer_b_for_all_real_domain_l2971_297155

theorem greatest_integer_b_for_all_real_domain : ∃ (b : ℤ), 
  (∀ (c : ℤ), c > b → ∃ (x : ℝ), x^2 + c*x + 5 = 0) ∧
  (∀ (x : ℝ), x^2 + b*x + 5 ≠ 0) ∧
  b = 4 := by
sorry

end greatest_integer_b_for_all_real_domain_l2971_297155


namespace walking_rate_ratio_l2971_297153

/-- The ratio of a boy's faster walking rate to his usual walking rate, given his usual time and early arrival time. -/
theorem walking_rate_ratio (usual_time early_time : ℕ) : 
  usual_time = 42 → early_time = 6 → (usual_time : ℚ) / (usual_time - early_time) = 7 / 6 := by
  sorry

end walking_rate_ratio_l2971_297153


namespace power_of_fraction_three_fourths_five_l2971_297187

theorem power_of_fraction_three_fourths_five :
  (3 / 4 : ℚ) ^ 5 = 243 / 1024 := by sorry

end power_of_fraction_three_fourths_five_l2971_297187


namespace frog_jump_probability_l2971_297159

/-- Represents a point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the square garden -/
def Garden :=
  {p : Point | 0 ≤ p.x ∧ p.x ≤ 5 ∧ 0 ≤ p.y ∧ p.y ≤ 5}

/-- Possible jump directions -/
inductive Direction
  | Up
  | Down
  | Left
  | Right

/-- Represents a frog jump -/
def jump (p : Point) (d : Direction) : Point :=
  match d with
  | Direction.Up => ⟨p.x, p.y + 1⟩
  | Direction.Down => ⟨p.x, p.y - 1⟩
  | Direction.Left => ⟨p.x - 1, p.y⟩
  | Direction.Right => ⟨p.x + 1, p.y⟩

/-- Checks if a point is on the vertical sides of the garden -/
def isOnVerticalSide (p : Point) : Prop :=
  (p.x = 0 ∨ p.x = 5) ∧ 0 ≤ p.y ∧ p.y ≤ 5

/-- The probability of ending on a vertical side from a given point -/
noncomputable def probabilityVerticalSide (p : Point) : ℝ := sorry

/-- The theorem to be proved -/
theorem frog_jump_probability :
  probabilityVerticalSide ⟨2, 1⟩ = 13 / 20 := by sorry

end frog_jump_probability_l2971_297159


namespace smallest_value_when_x_is_9_l2971_297129

theorem smallest_value_when_x_is_9 (x : ℝ) (h : x = 9) :
  min (9/x) (min (9/(x+1)) (min (9/(x-2)) (min (9/(6-x)) ((x-2)/9)))) = 9/(x+1) := by
  sorry

end smallest_value_when_x_is_9_l2971_297129


namespace A_power_98_l2971_297133

def A : Matrix (Fin 3) (Fin 3) ℝ := !![0, 0, 0; 0, 0, -1; 0, 1, 0]

theorem A_power_98 : A^98 = !![0, 0, 0; 0, -1, 0; 0, 0, -1] := by
  sorry

end A_power_98_l2971_297133


namespace quadratic_composition_roots_l2971_297108

/-- Given two quadratic trinomials f and g such that f(g(x)) = 0 and g(f(x)) = 0 have no real roots,
    at least one of f(f(x)) = 0 or g(g(x)) = 0 has no real roots. -/
theorem quadratic_composition_roots
  (f g : ℝ → ℝ)
  (hf : ∀ x, ∃ a b c : ℝ, f x = a * x^2 + b * x + c)
  (hg : ∀ x, ∃ d e f : ℝ, g x = d * x^2 + e * x + f)
  (hfg : ¬∃ x, f (g x) = 0)
  (hgf : ¬∃ x, g (f x) = 0) :
  (¬∃ x, f (f x) = 0) ∨ (¬∃ x, g (g x) = 0) :=
sorry

end quadratic_composition_roots_l2971_297108


namespace rectangle_ellipse_perimeter_l2971_297178

/-- Given a rectangle EFGH and an ellipse, prove the perimeter of the rectangle -/
theorem rectangle_ellipse_perimeter (p q c d : ℝ) : 
  p > 0 → q > 0 → c > 0 → d > 0 →
  p * q = 4032 →
  π * c * d = 2016 * π →
  p + q = 2 * c →
  p^2 + q^2 = 4 * (c^2 - d^2) →
  2 * (p + q) = 8 * Real.sqrt 2016 := by
sorry


end rectangle_ellipse_perimeter_l2971_297178


namespace smallest_divisible_by_12_15_18_l2971_297183

theorem smallest_divisible_by_12_15_18 : ∃ n : ℕ+, (∀ m : ℕ+, 12 ∣ m ∧ 15 ∣ m ∧ 18 ∣ m → n ≤ m) ∧ 12 ∣ n ∧ 15 ∣ n ∧ 18 ∣ n := by
  sorry

end smallest_divisible_by_12_15_18_l2971_297183


namespace total_miles_run_l2971_297112

/-- Given that Sam runs 12 miles and Harvey runs 8 miles more than Sam,
    prove that the total distance run by both friends is 32 miles. -/
theorem total_miles_run (sam_miles harvey_miles total_miles : ℕ) : 
  sam_miles = 12 →
  harvey_miles = sam_miles + 8 →
  total_miles = sam_miles + harvey_miles →
  total_miles = 32 := by
sorry

end total_miles_run_l2971_297112


namespace only_set_C_forms_triangle_l2971_297182

/-- Checks if three line segments can form a triangle --/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- The given sets of line segments --/
def set_A : List ℝ := [2, 5, 7]
def set_B : List ℝ := [4, 4, 8]
def set_C : List ℝ := [4, 5, 6]
def set_D : List ℝ := [4, 5, 10]

/-- Theorem stating that only set C can form a triangle --/
theorem only_set_C_forms_triangle :
  ¬(can_form_triangle set_A[0] set_A[1] set_A[2]) ∧
  ¬(can_form_triangle set_B[0] set_B[1] set_B[2]) ∧
  can_form_triangle set_C[0] set_C[1] set_C[2] ∧
  ¬(can_form_triangle set_D[0] set_D[1] set_D[2]) :=
sorry

end only_set_C_forms_triangle_l2971_297182


namespace floor_negative_two_point_eight_l2971_297160

theorem floor_negative_two_point_eight :
  ⌊(-2.8 : ℝ)⌋ = -3 := by sorry

end floor_negative_two_point_eight_l2971_297160


namespace corporation_employees_l2971_297136

theorem corporation_employees (part_time : ℕ) (full_time : ℕ) 
  (h1 : part_time = 2041) 
  (h2 : full_time = 63093) : 
  part_time + full_time = 65134 := by
  sorry

end corporation_employees_l2971_297136


namespace determinant_scaling_l2971_297147

theorem determinant_scaling (a b c d : ℝ) :
  Matrix.det ![![a, b], ![c, d]] = 7 →
  Matrix.det ![![3*a, 3*b], ![3*c, 3*d]] = 63 := by
  sorry

end determinant_scaling_l2971_297147


namespace nine_digit_repeat_gcd_l2971_297149

theorem nine_digit_repeat_gcd : 
  ∃ (d : ℕ), ∀ (n : ℕ), 100 ≤ n → n < 1000 → 
  Nat.gcd d (1001001 * n) = 1001001 ∧
  ∀ (m : ℕ), 100 ≤ m → m < 1000 → Nat.gcd d (1001001 * m) ∣ 1001001 :=
by sorry

end nine_digit_repeat_gcd_l2971_297149


namespace camp_kids_count_camp_kids_count_proof_l2971_297105

theorem camp_kids_count : ℕ → Prop :=
  fun total_kids =>
    let soccer_kids := total_kids / 2
    let morning_soccer_kids := soccer_kids / 4
    let afternoon_soccer_kids := soccer_kids - morning_soccer_kids
    afternoon_soccer_kids = 750 ∧ total_kids = 2000

-- The proof goes here
theorem camp_kids_count_proof : ∃ n : ℕ, camp_kids_count n := by
  sorry

end camp_kids_count_camp_kids_count_proof_l2971_297105


namespace increasing_function_a_range_l2971_297128

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (2*a - 1)*x + 1

-- State the theorem
theorem increasing_function_a_range :
  (∀ x₁ x₂, 2 < x₁ ∧ x₁ < x₂ → f a x₁ < f a x₂) →
  a ∈ Set.Ici (-(3/2)) :=
sorry

end increasing_function_a_range_l2971_297128


namespace arithmetic_sequence_property_l2971_297131

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_property (a : ℕ → ℝ) :
  is_arithmetic_sequence a → a 2 + a 8 = 12 → a 5 = 6 := by
  sorry

end arithmetic_sequence_property_l2971_297131


namespace rectangular_solid_surface_area_l2971_297158

/-- A number is prime or the sum of two consecutive primes -/
def IsPrimeOrSumOfConsecutivePrimes (n : ℕ) : Prop :=
  Nat.Prime n ∨ ∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ q = p + 1 ∧ n = p + q

/-- The theorem statement -/
theorem rectangular_solid_surface_area 
  (a b c : ℕ) 
  (ha : IsPrimeOrSumOfConsecutivePrimes a) 
  (hb : IsPrimeOrSumOfConsecutivePrimes b)
  (hc : IsPrimeOrSumOfConsecutivePrimes c)
  (hv : a * b * c = 399) : 
  2 * (a * b + b * c + c * a) = 422 := by
sorry

end rectangular_solid_surface_area_l2971_297158


namespace triangle_area_l2971_297197

/-- Given a triangle MNP where:
  * MN is the side opposite to the 60° angle
  * MP is the hypotenuse with length 40
  * Angle N is 90°
  Prove that the area of triangle MNP is 200√3 -/
theorem triangle_area (M N P : ℝ × ℝ) : 
  let MN := Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2)
  let MP := Real.sqrt ((M.1 - P.1)^2 + (M.2 - P.2)^2)
  let NP := Real.sqrt ((N.1 - P.1)^2 + (N.2 - P.2)^2)
  (∃ θ : Real, θ = π/3 ∧ MN = MP * Real.sin θ) →  -- MN is opposite to 60° angle
  MP = 40 →  -- MP is the hypotenuse with length 40
  (N.1 - M.1) * (P.1 - M.1) + (N.2 - M.2) * (P.2 - M.2) = 0 →  -- Angle N is 90°
  (1/2) * MN * NP = 200 * Real.sqrt 3 := by
sorry

end triangle_area_l2971_297197


namespace rectangle_area_proof_l2971_297172

/-- Given a rectangle ABCD with area a + 4√3, where the lines joining the centers of 
    circles inscribed in its corners form an equilateral triangle with side length 2, 
    prove that a = 8. -/
theorem rectangle_area_proof (a : ℝ) : 
  let triangle_side_length : ℝ := 2
  let rectangle_width : ℝ := triangle_side_length + triangle_side_length * Real.sqrt 3 / 2
  let rectangle_height : ℝ := 4
  let rectangle_area : ℝ := a + 4 * Real.sqrt 3
  rectangle_area = rectangle_width * rectangle_height → a = 8 :=
by sorry

end rectangle_area_proof_l2971_297172


namespace pants_pricing_l2971_297162

theorem pants_pricing (S P : ℝ) 
  (h1 : S = P + 0.25 * S)
  (h2 : 14 = 0.8 * S - P) :
  P = 210 := by sorry

end pants_pricing_l2971_297162


namespace lcm_of_primes_l2971_297107

theorem lcm_of_primes (p₁ p₂ p₃ p₄ : Nat) (h₁ : p₁ = 97) (h₂ : p₂ = 193) (h₃ : p₃ = 419) (h₄ : p₄ = 673) :
  Nat.lcm p₁ (Nat.lcm p₂ (Nat.lcm p₃ p₄)) = 5280671387 := by
  sorry

end lcm_of_primes_l2971_297107


namespace total_miles_driven_l2971_297138

theorem total_miles_driven (darius_miles julia_miles : ℕ) 
  (h1 : darius_miles = 679) 
  (h2 : julia_miles = 998) : 
  darius_miles + julia_miles = 1677 := by
  sorry

end total_miles_driven_l2971_297138


namespace line_passes_through_fixed_point_l2971_297143

/-- The line equation passing through a fixed point -/
def line_equation (a x y : ℝ) : Prop :=
  a * y = (3 * a - 1) * x - 1

/-- Theorem stating that the line passes through (-1, -3) for all a -/
theorem line_passes_through_fixed_point :
  ∀ (a : ℝ), line_equation a (-1) (-3) :=
by sorry

end line_passes_through_fixed_point_l2971_297143


namespace first_book_pictures_correct_l2971_297148

/-- The number of pictures in the first coloring book -/
def pictures_in_first_book : ℕ := 23

/-- The number of pictures in the second coloring book -/
def pictures_in_second_book : ℕ := 32

/-- The total number of pictures in both coloring books -/
def total_pictures : ℕ := 55

/-- Theorem stating that the number of pictures in the first coloring book is correct -/
theorem first_book_pictures_correct :
  pictures_in_first_book + pictures_in_second_book = total_pictures :=
by sorry

end first_book_pictures_correct_l2971_297148


namespace repeating_decimal_sum_l2971_297198

/-- Represents a repeating decimal with a single digit repeating -/
def RepeatingDecimal (n : ℕ) : ℚ := n / 9

theorem repeating_decimal_sum :
  RepeatingDecimal 6 + RepeatingDecimal 2 - RepeatingDecimal 4 = 4 / 9 := by
  sorry

end repeating_decimal_sum_l2971_297198


namespace money_left_after_spending_l2971_297194

theorem money_left_after_spending (initial_amount spent_on_sweets given_to_friend number_of_friends : ℚ) :
  initial_amount = 8.5 ∧ 
  spent_on_sweets = 1.25 ∧ 
  given_to_friend = 1.2 ∧ 
  number_of_friends = 2 →
  initial_amount - (spent_on_sweets + given_to_friend * number_of_friends) = 4.85 := by
sorry

end money_left_after_spending_l2971_297194


namespace islander_liar_count_l2971_297104

/-- Represents the type of islander: knight or liar -/
inductive IslanderType
| Knight
| Liar

/-- Represents a group of islanders making a statement -/
structure IslanderGroup where
  count : Nat
  statement : Nat

/-- The main theorem to prove -/
theorem islander_liar_count 
  (total_islanders : Nat)
  (group1 group2 group3 : IslanderGroup)
  (h1 : total_islanders = 19)
  (h2 : group1.count = 3 ∧ group1.statement = 3)
  (h3 : group2.count = 6 ∧ group2.statement = 6)
  (h4 : group3.count = 9 ∧ group3.statement = 9)
  (h5 : group1.count + group2.count + group3.count = total_islanders) :
  ∃ (liar_count : Nat), (liar_count = 9 ∨ liar_count = 18 ∨ liar_count = 19) ∧
    (∀ (x : Nat), x ≠ 9 ∧ x ≠ 18 ∧ x ≠ 19 → x ≠ liar_count) :=
by sorry

end islander_liar_count_l2971_297104


namespace prob_at_least_one_box_match_l2971_297191

/-- Represents the probability of a single block matching the previous one -/
def match_probability : ℚ := 1/2

/-- Represents the number of people -/
def num_people : ℕ := 3

/-- Represents the number of boxes -/
def num_boxes : ℕ := 3

/-- Represents the number of colors -/
def num_colors : ℕ := 3

/-- Calculates the probability of all three blocks in a single box being the same color -/
def prob_single_box_match : ℚ := match_probability * match_probability

/-- Calculates the probability of at least one box having all three blocks of the same color -/
theorem prob_at_least_one_box_match : 
  (1 : ℚ) - (1 - prob_single_box_match) ^ num_boxes = 37/64 :=
sorry

end prob_at_least_one_box_match_l2971_297191


namespace rachels_homework_l2971_297177

theorem rachels_homework (math_pages reading_pages : ℕ) : 
  math_pages = 7 → 
  math_pages = reading_pages + 4 → 
  reading_pages = 3 := by
sorry

end rachels_homework_l2971_297177


namespace simplified_expression_l2971_297196

theorem simplified_expression (a : ℝ) (h : a ≠ 1/2) :
  1 - (2 / (1 + (2*a / (1 - 2*a)))) = 4*a - 1 := by
  sorry

end simplified_expression_l2971_297196


namespace alien_resource_conversion_l2971_297186

/-- Converts a base-5 number represented as a list of digits to base 10 -/
def base5ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

theorem alien_resource_conversion :
  base5ToBase10 [3, 6, 2] = 83 := by
  sorry

end alien_resource_conversion_l2971_297186


namespace surviving_trees_count_l2971_297100

/-- Calculates the number of surviving trees after two months given initial conditions --/
theorem surviving_trees_count
  (tree_A_plants tree_B_plants tree_C_plants : ℕ)
  (tree_A_seeds_per_plant tree_B_seeds_per_plant tree_C_seeds_per_plant : ℕ)
  (tree_A_plant_rate tree_B_plant_rate tree_C_plant_rate : ℚ)
  (tree_A_first_month_survival_rate tree_B_first_month_survival_rate tree_C_first_month_survival_rate : ℚ)
  (second_month_survival_rate : ℚ)
  (h1 : tree_A_plants = 25)
  (h2 : tree_B_plants = 20)
  (h3 : tree_C_plants = 10)
  (h4 : tree_A_seeds_per_plant = 1)
  (h5 : tree_B_seeds_per_plant = 2)
  (h6 : tree_C_seeds_per_plant = 3)
  (h7 : tree_A_plant_rate = 3/5)
  (h8 : tree_B_plant_rate = 4/5)
  (h9 : tree_C_plant_rate = 1/2)
  (h10 : tree_A_first_month_survival_rate = 3/4)
  (h11 : tree_B_first_month_survival_rate = 9/10)
  (h12 : tree_C_first_month_survival_rate = 7/10)
  (h13 : second_month_survival_rate = 9/10) :
  ⌊(⌊tree_A_plants * tree_A_seeds_per_plant * tree_A_plant_rate * tree_A_first_month_survival_rate⌋ : ℚ) * second_month_survival_rate⌋ +
  ⌊(⌊tree_B_plants * tree_B_seeds_per_plant * tree_B_plant_rate * tree_B_first_month_survival_rate⌋ : ℚ) * second_month_survival_rate⌋ +
  ⌊(⌊tree_C_plants * tree_C_seeds_per_plant * tree_C_plant_rate * tree_C_first_month_survival_rate⌋ : ℚ) * second_month_survival_rate⌋ = 43 := by
  sorry


end surviving_trees_count_l2971_297100


namespace band_gigs_played_l2971_297157

theorem band_gigs_played (earnings_per_member : ℕ) (num_members : ℕ) (total_earnings : ℕ) : 
  earnings_per_member = 20 →
  num_members = 4 →
  total_earnings = 400 →
  total_earnings / (earnings_per_member * num_members) = 5 := by
sorry

end band_gigs_played_l2971_297157


namespace greatest_prime_factor_factorial_sum_l2971_297174

theorem greatest_prime_factor_factorial_sum : 
  ∃ p : ℕ, p.Prime ∧ p ∣ (Nat.factorial 15 + Nat.factorial 18) ∧ 
  ∀ q : ℕ, q.Prime → q ∣ (Nat.factorial 15 + Nat.factorial 18) → q ≤ p :=
by
  -- The proof would go here
  sorry

#eval 4897 -- This will output the expected result

end greatest_prime_factor_factorial_sum_l2971_297174


namespace candy_store_spending_l2971_297156

def weekly_allowance : ℚ := 4.5

def arcade_spending (allowance : ℚ) : ℚ := (3 / 5) * allowance

def remaining_after_arcade (allowance : ℚ) : ℚ := allowance - arcade_spending allowance

def toy_store_spending (remaining : ℚ) : ℚ := (1 / 3) * remaining

def remaining_after_toy_store (remaining : ℚ) : ℚ := remaining - toy_store_spending remaining

theorem candy_store_spending :
  remaining_after_toy_store (remaining_after_arcade weekly_allowance) = 1.2 := by
  sorry

end candy_store_spending_l2971_297156


namespace smallest_with_16_divisors_l2971_297144

/-- The number of positive integer divisors of n -/
def num_divisors (n : ℕ) : ℕ := (Nat.divisors n).card

/-- n has exactly 16 different positive integer divisors -/
def has_16_divisors (n : ℕ) : Prop := num_divisors n = 16

theorem smallest_with_16_divisors : 
  ∃ (n : ℕ), has_16_divisors n ∧ ∀ (m : ℕ), has_16_divisors m → n ≤ m :=
by sorry

end smallest_with_16_divisors_l2971_297144


namespace landmark_distance_set_l2971_297168

def distance_to_landmark (d : ℝ) : Prop :=
  d > 0 ∧ (d < 7 ∨ d > 7) ∧ (d ≤ 8 ∨ d > 8) ∧ (d ≤ 10 ∨ d > 10)

theorem landmark_distance_set :
  ∀ d : ℝ, distance_to_landmark d ↔ d > 10 :=
sorry

end landmark_distance_set_l2971_297168


namespace triangle_intersection_coord_diff_l2971_297123

/-- Triangle ABC with vertices A(0,10), B(3,-1), C(9,-1) -/
structure Triangle :=
  (A : ℝ × ℝ)
  (B : ℝ × ℝ)
  (C : ℝ × ℝ)

/-- Point R on line AC and S on line BC -/
structure IntersectionPoints (T : Triangle) :=
  (R : ℝ × ℝ)
  (S : ℝ × ℝ)

/-- The area of triangle RSC -/
def areaRSC (T : Triangle) (I : IntersectionPoints T) : ℝ := sorry

/-- The positive difference between x and y coordinates of R -/
def coordDiffR (I : IntersectionPoints T) : ℝ :=
  |I.R.1 - I.R.2|

theorem triangle_intersection_coord_diff 
  (T : Triangle) 
  (hA : T.A = (0, 10)) 
  (hB : T.B = (3, -1)) 
  (hC : T.C = (9, -1)) 
  (I : IntersectionPoints T) 
  (hvert : I.R.1 = I.S.1) -- R and S on the same vertical line
  (harea : areaRSC T I = 20) :
  coordDiffR I = 50/9 := by sorry

end triangle_intersection_coord_diff_l2971_297123


namespace exact_time_solution_l2971_297102

def minute_hand_speed : ℝ := 6
def hour_hand_speed : ℝ := 0.5
def hour_hand_start : ℝ := 270

def clock_problem (t : ℝ) : Prop :=
  0 ≤ t ∧ t < 60 ∧
  |minute_hand_speed * (t + 5) - (hour_hand_start + hour_hand_speed * (t - 2))| = 180

theorem exact_time_solution :
  ∃ t : ℝ, clock_problem t ∧ (21 : ℝ) < t ∧ t < 22 :=
sorry

end exact_time_solution_l2971_297102


namespace bologna_sandwiches_l2971_297190

/-- Given a ratio of cheese, bologna, and peanut butter sandwiches as 1:7:8 and a total of 80 sandwiches,
    prove that the number of bologna sandwiches is 35. -/
theorem bologna_sandwiches (total : ℕ) (cheese : ℕ) (bologna : ℕ) (peanut_butter : ℕ)
  (h_total : total = 80)
  (h_ratio : cheese + bologna + peanut_butter = 16)
  (h_cheese : cheese = 1)
  (h_bologna : bologna = 7)
  (h_peanut_butter : peanut_butter = 8) :
  (total / (cheese + bologna + peanut_butter)) * bologna = 35 :=
by sorry

end bologna_sandwiches_l2971_297190


namespace lines_parallel_when_perpendicular_to_parallel_planes_l2971_297125

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (planeParallel : Plane → Plane → Prop)

-- State the theorem
theorem lines_parallel_when_perpendicular_to_parallel_planes
  (a b : Line) (α β : Plane)
  (h1 : a ≠ b)
  (h2 : α ≠ β)
  (h3 : perpendicular a α)
  (h4 : perpendicular b β)
  (h5 : planeParallel α β) :
  parallel a b :=
sorry

end lines_parallel_when_perpendicular_to_parallel_planes_l2971_297125


namespace yard_width_calculation_l2971_297195

/-- The width of a rectangular yard with a row of trees --/
def yard_width (num_trees : ℕ) (edge_distance : ℝ) (center_distance : ℝ) (end_space : ℝ) : ℝ :=
  let tree_diameter := center_distance - edge_distance
  let total_center_distance := (num_trees - 1) * center_distance
  let total_tree_width := tree_diameter * num_trees
  let total_end_space := 2 * end_space
  total_center_distance + total_tree_width + total_end_space

/-- Theorem stating the width of the yard given the specific conditions --/
theorem yard_width_calculation :
  yard_width 6 12 15 2 = 82 := by
  sorry

end yard_width_calculation_l2971_297195


namespace sara_bouquets_l2971_297134

theorem sara_bouquets (red_flowers yellow_flowers : ℕ) 
  (h1 : red_flowers = 16) 
  (h2 : yellow_flowers = 24) : 
  (Nat.gcd red_flowers yellow_flowers) = 8 := by
  sorry

end sara_bouquets_l2971_297134


namespace proportion_equality_l2971_297113

theorem proportion_equality (a b c d : ℝ) (h : a / b = c / d) :
  (a + c) / c = (b + d) / d := by
  sorry

end proportion_equality_l2971_297113


namespace car_speed_problem_l2971_297124

/-- Proves that the speed in the first hour is 90 km/h given the conditions -/
theorem car_speed_problem (speed_second_hour : ℝ) (average_speed : ℝ) :
  speed_second_hour = 50 →
  average_speed = 70 →
  (speed_first_hour : ℝ) →
  (speed_first_hour + speed_second_hour) / 2 = average_speed →
  speed_first_hour = 90 := by
  sorry

end car_speed_problem_l2971_297124


namespace checkerboard_probability_l2971_297175

/-- The size of one side of the square checkerboard -/
def boardSize : ℕ := 10

/-- The total number of squares on the checkerboard -/
def totalSquares : ℕ := boardSize * boardSize

/-- The number of squares on the perimeter of the checkerboard -/
def perimeterSquares : ℕ := 4 * boardSize - 4

/-- The number of squares not on the perimeter of the checkerboard -/
def innerSquares : ℕ := totalSquares - perimeterSquares

/-- The probability of choosing a square that doesn't touch the outer edge -/
def probabilityInnerSquare : ℚ := innerSquares / totalSquares

theorem checkerboard_probability :
  probabilityInnerSquare = 16 / 25 := by
  sorry

end checkerboard_probability_l2971_297175


namespace difference_of_squares_l2971_297122

theorem difference_of_squares (x y : ℝ) (h1 : x + y = 15) (h2 : x - y = 10) : x^2 - y^2 = 150 := by
  sorry

end difference_of_squares_l2971_297122


namespace egg_problem_solution_l2971_297141

/-- Represents the number of eggs of each type --/
structure EggCounts where
  newLaid : ℕ
  fresh : ℕ
  ordinary : ℕ

/-- Checks if the given egg counts satisfy all problem constraints --/
def satisfiesConstraints (counts : EggCounts) : Prop :=
  counts.newLaid + counts.fresh + counts.ordinary = 100 ∧
  5 * counts.newLaid + counts.fresh + (counts.ordinary / 2) = 100 ∧
  (counts.newLaid = counts.fresh ∨ counts.newLaid = counts.ordinary ∨ counts.fresh = counts.ordinary)

/-- The unique solution to the egg problem --/
def eggSolution : EggCounts :=
  { newLaid := 10, fresh := 10, ordinary := 80 }

/-- Theorem stating that the egg solution is unique and satisfies all constraints --/
theorem egg_problem_solution :
  satisfiesConstraints eggSolution ∧
  ∀ counts : EggCounts, satisfiesConstraints counts → counts = eggSolution := by
  sorry


end egg_problem_solution_l2971_297141


namespace tea_cups_problem_l2971_297179

theorem tea_cups_problem (total_tea : ℕ) (cup_capacity : ℕ) (h1 : total_tea = 1050) (h2 : cup_capacity = 65) :
  (total_tea / cup_capacity : ℕ) = 16 :=
by sorry

end tea_cups_problem_l2971_297179


namespace sufficient_not_necessary_l2971_297146

theorem sufficient_not_necessary (x : ℝ) :
  (x > 1/2 → (1 - 2*x) * (x + 1) < 0) ∧
  ¬(∀ x : ℝ, (1 - 2*x) * (x + 1) < 0 → x > 1/2) :=
sorry

end sufficient_not_necessary_l2971_297146


namespace digit_2015_is_zero_l2971_297116

/-- A sequence formed by arranging all positive integers in increasing order -/
def integer_sequence : ℕ → ℕ := sorry

/-- The nth digit in the integer sequence -/
def nth_digit (n : ℕ) : ℕ := sorry

/-- Theorem: If the 11th digit in the integer sequence is 0, then the 2015th digit is also 0 -/
theorem digit_2015_is_zero (h : nth_digit 11 = 0) : nth_digit 2015 = 0 := by
  sorry

end digit_2015_is_zero_l2971_297116


namespace probability_of_one_in_20_rows_l2971_297192

/-- Represents Pascal's Triangle up to a given number of rows -/
def PascalTriangle (rows : ℕ) : Type := Unit

/-- Counts the total number of elements in the first n rows of Pascal's Triangle -/
def totalElements (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Counts the number of ones in the first n rows of Pascal's Triangle -/
def countOnes (n : ℕ) : ℕ := if n = 0 then 0 else 2 * n - 1

/-- The probability of randomly selecting a 1 from the first n rows of Pascal's Triangle -/
def probabilityOfOne (n : ℕ) : ℚ := countOnes n / totalElements n

theorem probability_of_one_in_20_rows :
  probabilityOfOne 20 = 39 / 210 := by sorry

end probability_of_one_in_20_rows_l2971_297192


namespace pat_calculation_l2971_297137

theorem pat_calculation (x : ℝ) : (x / 8) * 2 - 12 = 40 → x * 8 + 2 * x + 12 > 1000 := by
  sorry

end pat_calculation_l2971_297137


namespace popcorn_tablespoons_needed_l2971_297167

/-- The number of cups of popcorn produced by 2 tablespoons of kernels -/
def cups_per_two_tablespoons : ℕ := 4

/-- The number of cups of popcorn Joanie wants -/
def joanie_cups : ℕ := 3

/-- The number of cups of popcorn Mitchell wants -/
def mitchell_cups : ℕ := 4

/-- The number of cups of popcorn Miles and Davis will split -/
def miles_davis_cups : ℕ := 6

/-- The number of cups of popcorn Cliff wants -/
def cliff_cups : ℕ := 3

/-- The total number of cups of popcorn wanted -/
def total_cups : ℕ := joanie_cups + mitchell_cups + miles_davis_cups + cliff_cups

/-- Theorem stating the number of tablespoons of popcorn kernels needed -/
theorem popcorn_tablespoons_needed : 
  (total_cups / cups_per_two_tablespoons) * 2 = 8 := by
  sorry

end popcorn_tablespoons_needed_l2971_297167


namespace min_distance_complex_circles_l2971_297188

theorem min_distance_complex_circles (z w : ℂ) 
  (hz : Complex.abs (z - (2 - 4 * Complex.I)) = 2)
  (hw : Complex.abs (w - (6 - 5 * Complex.I)) = 4) :
  ∃ (min_dist : ℝ), 
    (∀ z' w' : ℂ, 
      Complex.abs (z' - (2 - 4 * Complex.I)) = 2 → 
      Complex.abs (w' - (6 - 5 * Complex.I)) = 4 → 
      Complex.abs (z' - w') ≥ min_dist) ∧
    Complex.abs (z - w) ≥ min_dist ∧
    min_dist = Real.sqrt 17 - 6 :=
by sorry

end min_distance_complex_circles_l2971_297188


namespace min_overlap_cells_l2971_297189

/-- Given positive integers m and n where m < n, in an n × n board filled with integers from 1 to n^2, 
    if the m largest numbers in each row are colored red and the m largest numbers in each column are colored blue, 
    then the minimum number of cells that are both red and blue is m^2. -/
theorem min_overlap_cells (m n : ℕ) (h1 : 0 < m) (h2 : 0 < n) (h3 : m < n) : 
  (∀ (board : Fin n → Fin n → ℕ), 
    (∀ i j, board i j ∈ Finset.range (n^2 + 1)) →
    (∀ i, ∃ (red : Finset (Fin n)), red.card = m ∧ ∀ j ∈ red, ∀ k ∉ red, board i j ≥ board i k) →
    (∀ j, ∃ (blue : Finset (Fin n)), blue.card = m ∧ ∀ i ∈ blue, ∀ k ∉ blue, board i j ≥ board k j) →
    ∃ (overlap : Finset (Fin n × Fin n)), 
      overlap.card = m^2 ∧ 
      (∀ (i j), (i, j) ∈ overlap ↔ (∃ (red blue : Finset (Fin n)), 
        red.card = m ∧ blue.card = m ∧
        (∀ k ∉ red, board i j ≥ board i k) ∧
        (∀ k ∉ blue, board i j ≥ board k j) ∧
        i ∈ red ∧ j ∈ blue))) :=
by sorry

end min_overlap_cells_l2971_297189


namespace difference_number_and_three_fifths_l2971_297142

theorem difference_number_and_three_fifths (n : ℚ) : n = 140 → n - (3 / 5 * n) = 56 := by
  sorry

end difference_number_and_three_fifths_l2971_297142


namespace exist_same_color_parallel_triangle_l2971_297115

/-- Represents a color --/
inductive Color
  | Red
  | Green
  | Blue

/-- Represents a point in the triangle --/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the coloring of vertices --/
def Coloring := Point → Color

/-- Represents the large equilateral triangle --/
structure LargeTriangle where
  side_length : ℝ
  division_count : ℕ
  coloring : Coloring

/-- Checks if three points form a triangle parallel to the original triangle --/
def is_parallel_triangle (p1 p2 p3 : Point) : Prop := sorry

/-- Main theorem --/
theorem exist_same_color_parallel_triangle (T : LargeTriangle) 
  (h1 : T.division_count = 3000) -- 9000000 small triangles means 3000 divisions per side
  (h2 : T.side_length > 0) :
  ∃ (c : Color) (p1 p2 p3 : Point),
    T.coloring p1 = c ∧
    T.coloring p2 = c ∧
    T.coloring p3 = c ∧
    is_parallel_triangle p1 p2 p3 :=
  sorry

end exist_same_color_parallel_triangle_l2971_297115


namespace angle_calculation_l2971_297164

/-- Given three angles, proves that if angle 1 and angle 2 are complementary, 
    angle 2 and angle 3 are supplementary, and angle 3 equals 18°, 
    then angle 1 equals 108°. -/
theorem angle_calculation (angle1 angle2 angle3 : ℝ) : 
  angle1 + angle2 = 90 →
  angle2 + angle3 = 180 →
  angle3 = 18 →
  angle1 = 108 := by
  sorry

end angle_calculation_l2971_297164


namespace refrigerator_price_calculation_l2971_297181

def refrigerator_purchase_price (labelled_price : ℝ) (discount_rate : ℝ) (additional_costs : ℝ) (selling_price : ℝ) (profit_rate : ℝ) : ℝ :=
  (1 - discount_rate) * labelled_price + additional_costs

theorem refrigerator_price_calculation :
  let labelled_price : ℝ := 18400 / 1.15
  let discount_rate : ℝ := 0.20
  let additional_costs : ℝ := 125 + 250
  let selling_price : ℝ := 18400
  let profit_rate : ℝ := 0.15
  refrigerator_purchase_price labelled_price discount_rate additional_costs selling_price profit_rate = 13175 := by
  sorry

end refrigerator_price_calculation_l2971_297181


namespace lea_notebooks_l2971_297135

/-- The number of notebooks Léa bought -/
def notebooks : ℕ := sorry

/-- The cost of the book Léa bought -/
def book_cost : ℕ := 16

/-- The number of binders Léa bought -/
def num_binders : ℕ := 3

/-- The cost of each binder -/
def binder_cost : ℕ := 2

/-- The cost of each notebook -/
def notebook_cost : ℕ := 1

/-- The total cost of Léa's purchases -/
def total_cost : ℕ := 28

theorem lea_notebooks : 
  notebooks = 6 ∧
  book_cost + num_binders * binder_cost + notebooks * notebook_cost = total_cost :=
sorry

end lea_notebooks_l2971_297135


namespace chess_group_size_l2971_297114

/-- The number of games played when n players each play every other player once -/
def gamesPlayed (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem stating that 14 players result in 91 games when each player plays every other player once -/
theorem chess_group_size :
  ∃ (n : ℕ), n > 0 ∧ gamesPlayed n = 91 ∧ n = 14 := by
  sorry

end chess_group_size_l2971_297114


namespace fraction_product_simplification_l2971_297106

theorem fraction_product_simplification :
  (3 / 4) * (4 / 5) * (5 / 6) * (6 / 7) = 3 / 7 := by
  sorry

end fraction_product_simplification_l2971_297106


namespace apple_cost_price_l2971_297180

theorem apple_cost_price (selling_price : ℚ) (loss_fraction : ℚ) : 
  selling_price = 20 → loss_fraction = 1/6 → 
  ∃ cost_price : ℚ, 
    selling_price = cost_price - (loss_fraction * cost_price) ∧ 
    cost_price = 24 := by
  sorry

end apple_cost_price_l2971_297180


namespace product_purchase_discount_l2971_297165

/-- Proves that if a product is sold for $439.99999999999966 with a 10% profit, 
    and if buying it for x% less and selling at 30% profit would yield $28 more, 
    then x = 10%. -/
theorem product_purchase_discount (x : Real) : 
  (1.1 * (439.99999999999966 / 1.1) = 439.99999999999966) →
  (1.3 * (1 - x/100) * (439.99999999999966 / 1.1) = 439.99999999999966 + 28) →
  x = 10 := by
  sorry

end product_purchase_discount_l2971_297165


namespace base7_multiplication_l2971_297163

/-- Converts a base 7 number (represented as a list of digits) to a natural number. -/
def base7ToNat (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => 7 * acc + d) 0

/-- Converts a natural number to its base 7 representation (as a list of digits). -/
def natToBase7 (n : Nat) : List Nat :=
  if n = 0 then [0]
  else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc
      else aux (m / 7) ((m % 7) :: acc)
    aux n []

/-- The main theorem stating that 356₇ * 4₇ = 21323₇ in base 7. -/
theorem base7_multiplication :
  natToBase7 (base7ToNat [3, 5, 6] * base7ToNat [4]) = [2, 1, 3, 2, 3] := by
  sorry

#eval base7ToNat [3, 5, 6] -- Should output 188
#eval base7ToNat [4] -- Should output 4
#eval natToBase7 (188 * 4) -- Should output [2, 1, 3, 2, 3]

end base7_multiplication_l2971_297163


namespace seventh_person_weight_l2971_297126

def elevator_problem (initial_people : ℕ) (initial_avg_weight : ℚ) (new_avg_weight : ℚ) : ℚ :=
  let total_initial_weight := initial_people * initial_avg_weight
  let total_new_weight := (initial_people + 1) * new_avg_weight
  total_new_weight - total_initial_weight

theorem seventh_person_weight :
  elevator_problem 6 160 151 = 97 := by sorry

end seventh_person_weight_l2971_297126


namespace inequality_range_l2971_297110

theorem inequality_range (m n : ℝ) (hm : m > 0) (hn : n > 0) (h_mn : m + n + 3 = m * n) :
  (∀ x : ℝ, (m + n) * x^2 + 2 * x + m * n - 13 ≥ 0) ↔ 
  (∀ x : ℝ, x ≤ -1 ∨ x ≥ 2/3) :=
by sorry

end inequality_range_l2971_297110


namespace final_display_l2971_297103

def special_key (x : ℚ) : ℚ := 1 / (2 - x)

def iterate_key (n : ℕ) (x : ℚ) : ℚ :=
  match n with
  | 0 => x
  | m + 1 => special_key (iterate_key m x)

theorem final_display : iterate_key 50 3 = 49 / 51 := by
  sorry

end final_display_l2971_297103


namespace explosion_hyperbola_eccentricity_l2971_297101

/-- The eccentricity of a hyperbola formed by an explosion point, given two sentry posts
    1400m apart, a time difference of 3s in hearing the explosion, and a speed of sound of 340m/s -/
theorem explosion_hyperbola_eccentricity
  (distance_between_posts : ℝ)
  (time_difference : ℝ)
  (speed_of_sound : ℝ)
  (h_distance : distance_between_posts = 1400)
  (h_time : time_difference = 3)
  (h_speed : speed_of_sound = 340) :
  let c : ℝ := distance_between_posts / 2
  let a : ℝ := time_difference * speed_of_sound / 2
  c / a = 70 / 51 :=
by sorry

end explosion_hyperbola_eccentricity_l2971_297101


namespace smallest_n_property_ratio_is_sqrt_three_l2971_297185

/-- The smallest positive integer n for which there exist positive real numbers a and b
    such that (a + bi)^n = -(a - bi)^n -/
def smallest_n : ℕ := 4

theorem smallest_n_property (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (Complex.I : ℂ)^smallest_n * (a + b * Complex.I)^smallest_n = -(a - b * Complex.I)^smallest_n :=
sorry

theorem ratio_is_sqrt_three (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h : (Complex.I : ℂ)^smallest_n * (a + b * Complex.I)^smallest_n = -(a - b * Complex.I)^smallest_n) :
  a / b = Real.sqrt 3 :=
sorry

end smallest_n_property_ratio_is_sqrt_three_l2971_297185


namespace tan_60_plus_inverse_sqrt_3_l2971_297154

theorem tan_60_plus_inverse_sqrt_3 :
  let tan_60 := Real.sqrt 3
  tan_60 + (Real.sqrt 3)⁻¹ = (4 * Real.sqrt 3) / 3 := by sorry

end tan_60_plus_inverse_sqrt_3_l2971_297154


namespace f_properties_l2971_297170

noncomputable def f (a x : ℝ) : ℝ := Real.log x - (a + 2) * x + a * x^2

theorem f_properties (a : ℝ) :
  -- Part I: Tangent line equation when a = 0
  (∀ x y : ℝ, f 0 1 = -2 ∧ x + y + 1 = 0 ↔ y = f 0 x ∧ (x - 1) * (f 0 x - f 0 1) = (y - f 0 1) * (x - 1)) ∧
  -- Part II: Monotonicity intervals
  (∀ x : ℝ, 0 < x ∧ x < 1/2 → (∀ h : ℝ, h > 0 → f a (x + h) > f a x)) ∧
  (∀ x : ℝ, x > 1/2 → (∀ h : ℝ, h > 0 → f a (x + h) < f a x)) ∧
  -- Part III: Condition for exactly two zeros
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0 ∧ (∀ x : ℝ, f a x = 0 → x = x₁ ∨ x = x₂) ↔ a < -4 * Real.log 2 - 4) :=
by sorry

end f_properties_l2971_297170


namespace range_of_f_l2971_297111

-- Define the function
def f (x : ℝ) : ℝ := |x + 5| - |x - 3|

-- State the theorem
theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ -3 ≤ y ∧ y ≤ 14 :=
by sorry

end range_of_f_l2971_297111


namespace cats_on_ship_l2971_297120

/-- Represents the number of cats on the ship -/
def num_cats : ℕ := 7

/-- Represents the number of humans on the ship -/
def num_humans : ℕ := 14 - num_cats

/-- The total number of heads on the ship -/
def total_heads : ℕ := 14

/-- The total number of legs on the ship -/
def total_legs : ℕ := 41

theorem cats_on_ship :
  (num_cats + num_humans = total_heads) ∧
  (4 * num_cats + 2 * num_humans - 1 = total_legs) :=
sorry

end cats_on_ship_l2971_297120


namespace geometric_sequence_a7_l2971_297132

/-- A geometric sequence with positive common ratio -/
def GeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  q > 0 ∧ ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_a7 (a : ℕ → ℝ) (q : ℝ) :
  GeometricSequence a q →
  a 4 * a 8 = 2 * (a 5)^2 →
  a 3 = 1 →
  a 7 = 4 := by
  sorry

end geometric_sequence_a7_l2971_297132


namespace binomial_coefficient_even_l2971_297171

theorem binomial_coefficient_even (n : ℕ) (h : Even n) (h2 : n > 0) : 
  Nat.choose n 2 = n * (n - 1) / 2 :=
by sorry

end binomial_coefficient_even_l2971_297171


namespace marta_textbook_problem_l2971_297150

theorem marta_textbook_problem :
  ∀ (sale_books online_books bookstore_books : ℕ)
    (sale_price online_total bookstore_total total_spent : ℚ),
    sale_books = 5 →
    sale_price = 10 →
    online_books = 2 →
    online_total = 40 →
    bookstore_total = 3 * online_total →
    total_spent = 210 →
    sale_books * sale_price + online_total + bookstore_total = total_spent →
    bookstore_books = 2 :=
by
  sorry

end marta_textbook_problem_l2971_297150


namespace women_count_l2971_297151

/-- Represents a company with workers and their retirement plan status -/
structure Company where
  total_workers : ℕ
  workers_without_plan : ℕ
  women_without_plan_ratio : ℚ
  women_with_plan_ratio : ℚ
  total_men : ℕ

/-- Calculates the number of women in the company -/
def number_of_women (c : Company) : ℚ :=
  c.women_without_plan_ratio * c.workers_without_plan +
  c.women_with_plan_ratio * (c.total_workers - c.workers_without_plan)

/-- Theorem stating the number of women in the company -/
theorem women_count (c : Company) 
  (h1 : c.total_workers = 200)
  (h2 : c.workers_without_plan = c.total_workers / 3)
  (h3 : c.women_without_plan_ratio = 2/5)
  (h4 : c.women_with_plan_ratio = 3/5)
  (h5 : c.total_men = 120) :
  ∃ (n : ℕ), n ≤ number_of_women c ∧ number_of_women c < n + 1 ∧ n = 107 := by
  sorry

end women_count_l2971_297151


namespace savings_calculation_l2971_297184

def thomas_monthly_savings : ℕ := 40
def saving_years : ℕ := 6
def months_per_year : ℕ := 12
def joseph_savings_ratio : ℚ := 3 / 5  -- Joseph saves 2/5 less, so he saves 3/5 of what Thomas saves

def total_savings : ℕ := 4608

theorem savings_calculation :
  thomas_monthly_savings * saving_years * months_per_year +
  (thomas_monthly_savings * joseph_savings_ratio).floor * saving_years * months_per_year = total_savings :=
by sorry

end savings_calculation_l2971_297184


namespace cube_root_simplification_l2971_297193

theorem cube_root_simplification : Real.rpow (2^9 * 5^3 * 7^3) (1/3) = 280 := by
  sorry

end cube_root_simplification_l2971_297193


namespace multiply_and_simplify_l2971_297145

theorem multiply_and_simplify (x : ℝ) (h : x ≠ 0) :
  (18 * x^3) * (4 * x^2) * (1 / (2*x)^3) = 9 * x^2 := by
  sorry

end multiply_and_simplify_l2971_297145


namespace mary_balloons_l2971_297130

/-- Given that Nancy has 7 black balloons and Mary has 4 times more black balloons than Nancy,
    prove that Mary has 28 black balloons. -/
theorem mary_balloons (nancy_balloons : ℕ) (mary_multiplier : ℕ) 
    (h1 : nancy_balloons = 7)
    (h2 : mary_multiplier = 4) : 
  nancy_balloons * mary_multiplier = 28 := by
  sorry

end mary_balloons_l2971_297130


namespace basketball_five_bounces_l2971_297109

/-- Calculates the total distance traveled by a basketball dropped from a given height,
    rebounding to a fraction of its previous height, for a given number of bounces. -/
def basketballDistance (initialHeight : ℝ) (reboundFraction : ℝ) (bounces : ℕ) : ℝ :=
  sorry

/-- Theorem stating that a basketball dropped from 80 feet, rebounding to three-quarters
    of its previous height each time, will have traveled 408.125 feet when it hits the
    ground for the fifth time. -/
theorem basketball_five_bounces :
  basketballDistance 80 0.75 5 = 408.125 := by sorry

end basketball_five_bounces_l2971_297109


namespace shaded_area_between_circles_l2971_297173

/-- The area of the shaded region between a circle circumscribing two overlapping circles
    and the two smaller circles. -/
theorem shaded_area_between_circles (r₁ r₂ d R : ℝ) (h₁ : r₁ = 4) (h₂ : r₂ = 5) (h₃ : d = 6)
    (h₄ : R = r₁ + r₂ + d) : π * R^2 - (π * r₁^2 + π * r₂^2) = 184 * π := by
  sorry

#check shaded_area_between_circles

end shaded_area_between_circles_l2971_297173


namespace caesars_meal_cost_proof_l2971_297152

/-- The cost per meal at Caesar's banquet hall -/
def caesars_meal_cost : ℝ := 30

/-- The number of guests attending the prom -/
def num_guests : ℕ := 60

/-- Caesar's room rental fee -/
def caesars_room_fee : ℝ := 800

/-- Venus Hall's room rental fee -/
def venus_room_fee : ℝ := 500

/-- Venus Hall's cost per meal -/
def venus_meal_cost : ℝ := 35

theorem caesars_meal_cost_proof :
  caesars_room_fee + num_guests * caesars_meal_cost =
  venus_room_fee + num_guests * venus_meal_cost :=
by sorry

end caesars_meal_cost_proof_l2971_297152


namespace grid_sum_puzzle_l2971_297117

theorem grid_sum_puzzle :
  ∃ (a b c d e f g : ℤ),
    a = 3 ∧ b = 0 ∧ c = 5 ∧ d = -2 ∧ e = -9 ∧ f = -5 ∧ g = 1 ∧
    a + (-1) + 2 = 4 ∧
    2 + 1 + b = 3 ∧
    c + (-4) + (-3) = -2 ∧
    b - 5 - 4 = e ∧
    f = d - 3 ∧
    g = d + 3 ∧
    -8 = 4 + 3 - 9 - 2 + f + g :=
by sorry

end grid_sum_puzzle_l2971_297117


namespace exam_time_on_type_A_l2971_297166

/-- Represents the time spent on type A problems in an exam -/
def time_on_type_A (total_time : ℚ) (total_questions : ℕ) (type_A_questions : ℕ) : ℚ :=
  let type_B_questions := total_questions - type_A_questions
  let time_ratio := (2 * type_A_questions + type_B_questions) / total_questions
  (total_time * time_ratio * 2 * type_A_questions) / (2 * type_A_questions + type_B_questions)

/-- Theorem stating the time spent on type A problems in the given exam conditions -/
theorem exam_time_on_type_A :
  time_on_type_A (5/2) 200 10 = 100/7 :=
by sorry

end exam_time_on_type_A_l2971_297166


namespace xy_value_l2971_297169

theorem xy_value (x y : ℝ) (h : x^2 + 2*y^2 - 2*x*y + 4*y + 4 = 0) : x^y = 1/4 := by
  sorry

end xy_value_l2971_297169


namespace pages_printed_for_fifty_dollars_l2971_297118

/-- Given the cost of 9 cents for 7 pages, prove that the maximum number of whole pages
    that can be printed for $50 is 3888. -/
theorem pages_printed_for_fifty_dollars (cost_per_seven_pages : ℚ) 
  (h1 : cost_per_seven_pages = 9/100) : 
  ⌊(50 * 100 * 7) / (cost_per_seven_pages * 7)⌋ = 3888 := by
  sorry

end pages_printed_for_fifty_dollars_l2971_297118


namespace perfect_square_trinomial_factorization_l2971_297176

theorem perfect_square_trinomial_factorization (x : ℝ) :
  x^2 - 2*x + 1 = (x - 1)^2 := by
  sorry

end perfect_square_trinomial_factorization_l2971_297176


namespace arcade_tickets_l2971_297140

theorem arcade_tickets (initial_tickets spent_tickets additional_tickets : ℕ) :
  initial_tickets ≥ spent_tickets →
  (initial_tickets - spent_tickets + additional_tickets) = 
    initial_tickets - spent_tickets + additional_tickets :=
by
  sorry

end arcade_tickets_l2971_297140

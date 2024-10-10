import Mathlib

namespace fourth_month_sale_l2961_296140

theorem fourth_month_sale
  (sale1 sale2 sale3 sale5 sale6 : ℕ)
  (average_sale : ℕ)
  (h1 : sale1 = 6435)
  (h2 : sale2 = 6927)
  (h3 : sale3 = 6855)
  (h5 : sale5 = 6562)
  (h6 : sale6 = 6191)
  (h_avg : average_sale = 6700)
  (h_total : average_sale * 6 = sale1 + sale2 + sale3 + sale5 + sale6 + sale4) :
  sale4 = 7230 :=
by
  sorry

end fourth_month_sale_l2961_296140


namespace rectangle_ratio_l2961_296175

/-- A configuration of squares and a rectangle forming a large square -/
structure SquareConfiguration where
  /-- Side length of each small square -/
  s : ℝ
  /-- Side length of the large square -/
  largeSide : ℝ
  /-- Length of the rectangle -/
  rectLength : ℝ
  /-- Width of the rectangle -/
  rectWidth : ℝ
  /-- The large square's side is 3 times the small square's side -/
  large_square : largeSide = 3 * s
  /-- The rectangle's length is 3 times the small square's side -/
  rect_length : rectLength = 3 * s
  /-- The rectangle's width is 2 times the small square's side -/
  rect_width : rectWidth = 2 * s

/-- The ratio of the rectangle's length to its width is 3/2 -/
theorem rectangle_ratio (config : SquareConfiguration) :
  config.rectLength / config.rectWidth = 3 / 2 := by
  sorry

end rectangle_ratio_l2961_296175


namespace max_mineral_worth_l2961_296144

-- Define the mineral types
inductive Mineral
| Sapphire
| Ruby
| Emerald

-- Define the properties of each mineral
def weight (m : Mineral) : Nat :=
  match m with
  | Mineral.Sapphire => 6
  | Mineral.Ruby => 3
  | Mineral.Emerald => 2

def value (m : Mineral) : Nat :=
  match m with
  | Mineral.Sapphire => 18
  | Mineral.Ruby => 9
  | Mineral.Emerald => 4

-- Define the maximum carrying capacity
def maxWeight : Nat := 20

-- Define the minimum available quantity of each mineral
def minQuantity : Nat := 30

-- Define a function to calculate the total weight of a combination of minerals
def totalWeight (s r e : Nat) : Nat :=
  s * weight Mineral.Sapphire + r * weight Mineral.Ruby + e * weight Mineral.Emerald

-- Define a function to calculate the total value of a combination of minerals
def totalValue (s r e : Nat) : Nat :=
  s * value Mineral.Sapphire + r * value Mineral.Ruby + e * value Mineral.Emerald

-- Theorem: The maximum worth of minerals Joe can carry is $58
theorem max_mineral_worth :
  ∃ s r e : Nat,
    s ≤ minQuantity ∧ r ≤ minQuantity ∧ e ≤ minQuantity ∧
    totalWeight s r e ≤ maxWeight ∧
    totalValue s r e = 58 ∧
    ∀ s' r' e' : Nat,
      s' ≤ minQuantity → r' ≤ minQuantity → e' ≤ minQuantity →
      totalWeight s' r' e' ≤ maxWeight →
      totalValue s' r' e' ≤ 58 :=
by sorry

end max_mineral_worth_l2961_296144


namespace range_of_H_l2961_296147

-- Define the function H
def H (x : ℝ) : ℝ := |x + 2| - |x - 3|

-- State the theorem about the range of H
theorem range_of_H :
  Set.range H = Set.Icc 1 5 := by sorry

end range_of_H_l2961_296147


namespace y₁_gt_y₂_l2961_296154

/-- The quadratic function f(x) = x² - 4x - 3 -/
def f (x : ℝ) : ℝ := x^2 - 4*x - 3

/-- Point A on the graph of f -/
def A : ℝ × ℝ := (-1, f (-1))

/-- Point B on the graph of f -/
def B : ℝ × ℝ := (1, f 1)

/-- y₁ coordinate of point A -/
def y₁ : ℝ := A.2

/-- y₂ coordinate of point B -/
def y₂ : ℝ := B.2

theorem y₁_gt_y₂ : y₁ > y₂ := by
  sorry

end y₁_gt_y₂_l2961_296154


namespace cyclists_meeting_time_l2961_296126

/-- The time taken by two cyclists meeting on the road -/
theorem cyclists_meeting_time (x y : ℝ) 
  (h1 : x - 4 = y - 9)  -- Time before meeting is equal for both cyclists
  (h2 : 4 / (y - 9) = (x - 4) / 9)  -- Proportion of speeds based on distances
  : x = 10 ∧ y = 15 := by
  sorry

end cyclists_meeting_time_l2961_296126


namespace acrobats_count_correct_l2961_296187

/-- Represents the number of acrobats in the zoo. -/
def acrobats : ℕ := 5

/-- Represents the number of elephants in the zoo. -/
def elephants : ℕ := sorry

/-- Represents the number of camels in the zoo. -/
def camels : ℕ := sorry

/-- The total number of legs in the zoo. -/
def total_legs : ℕ := 58

/-- The total number of heads in the zoo. -/
def total_heads : ℕ := 17

/-- Theorem stating that the number of acrobats is correct given the conditions. -/
theorem acrobats_count_correct :
  acrobats * 2 + elephants * 4 + camels * 4 = total_legs ∧
  acrobats + elephants + camels = total_heads :=
by sorry

end acrobats_count_correct_l2961_296187


namespace local_max_implies_c_eq_6_l2961_296110

/-- The function f(x) = x(x-c)^2 -/
def f (c : ℝ) (x : ℝ) : ℝ := x * (x - c)^2

/-- f has a local maximum at x = 2 -/
def has_local_max_at_2 (c : ℝ) : Prop :=
  ∃ δ > 0, ∀ x, |x - 2| < δ → f c x ≤ f c 2

theorem local_max_implies_c_eq_6 :
  ∀ c : ℝ, has_local_max_at_2 c → c = 6 := by sorry

end local_max_implies_c_eq_6_l2961_296110


namespace isosceles_triangle_perimeter_l2961_296155

/-- An isosceles triangle with sides a, b, and c, where a and b satisfy |a-2|+(b-5)^2=0 -/
structure IsoscelesTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  isIsosceles : (a = b ∧ c ≠ a) ∨ (b = c ∧ a ≠ b) ∨ (a = c ∧ b ≠ a)
  satisfiesEquation : |a - 2| + (b - 5)^2 = 0

/-- The perimeter of an isosceles triangle is 12 if it satisfies the given condition -/
theorem isosceles_triangle_perimeter (t : IsoscelesTriangle) : t.a + t.b + t.c = 12 := by
  sorry


end isosceles_triangle_perimeter_l2961_296155


namespace prob_open_door_third_attempt_l2961_296158

/-- Probability of opening a door on the third attempt given 5 keys with only one correct key -/
theorem prob_open_door_third_attempt (total_keys : ℕ) (correct_keys : ℕ) (attempt : ℕ) :
  total_keys = 5 →
  correct_keys = 1 →
  attempt = 3 →
  (1 : ℚ) / total_keys = 1 / 5 := by
  sorry

end prob_open_door_third_attempt_l2961_296158


namespace bridge_length_calculation_l2961_296145

/-- Calculates the length of a bridge given train parameters --/
theorem bridge_length_calculation (train_length : ℝ) (train_speed_kmh : ℝ) (time_to_pass : ℝ) : 
  train_length = 410 ∧ 
  train_speed_kmh = 45 ∧ 
  time_to_pass = 44 → 
  (train_speed_kmh * 1000 / 3600) * time_to_pass - train_length = 140 := by sorry

end bridge_length_calculation_l2961_296145


namespace perfect_square_trinomial_condition_l2961_296138

/-- A trinomial ax^2 + bx + c is a perfect square if there exists r such that ax^2 + bx + c = (rx + s)^2 for all x -/
def IsPerfectSquareTrinomial (a b c : ℝ) : Prop :=
  ∃ r s : ℝ, ∀ x : ℝ, a * x^2 + b * x + c = (r * x + s)^2

/-- If x^2 + kx + 81 is a perfect square trinomial, then k = 18 or k = -18 -/
theorem perfect_square_trinomial_condition (k : ℝ) :
  IsPerfectSquareTrinomial 1 k 81 → k = 18 ∨ k = -18 := by
  sorry

end perfect_square_trinomial_condition_l2961_296138


namespace correct_scientific_notation_l2961_296109

/-- Scientific notation representation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coeff_range : 1 ≤ coefficient ∧ coefficient < 10

/-- Check if a ScientificNotation represents a given number -/
def represents (sn : ScientificNotation) (n : ℝ) : Prop :=
  sn.coefficient * (10 : ℝ) ^ sn.exponent = n

/-- The number we want to represent in scientific notation -/
def target_number : ℝ := 2034000

/-- The proposed scientific notation representation -/
def proposed_representation : ScientificNotation := {
  coefficient := 2.034
  exponent := 6
  coeff_range := by sorry
}

/-- Theorem stating that the proposed representation is correct -/
theorem correct_scientific_notation :
  represents proposed_representation target_number :=
by sorry

end correct_scientific_notation_l2961_296109


namespace trees_in_yard_l2961_296139

/-- The number of trees planted along a yard. -/
def number_of_trees (yard_length : ℕ) (tree_distance : ℕ) : ℕ :=
  (yard_length / tree_distance) + 1

/-- Theorem: Given a yard 441 metres long with trees planted at equal distances,
    one tree at each end, and 21 metres between consecutive trees,
    there are 22 trees planted along the yard. -/
theorem trees_in_yard :
  let yard_length : ℕ := 441
  let tree_distance : ℕ := 21
  number_of_trees yard_length tree_distance = 22 := by
  sorry

end trees_in_yard_l2961_296139


namespace min_value_of_u_l2961_296104

theorem min_value_of_u (a b : ℝ) (h : 3*a^2 - 10*a*b + 8*b^2 + 5*a - 10*b = 0) :
  ∃ (u_min : ℝ), u_min = -34 ∧ ∀ (u : ℝ), u = 9*a^2 + 72*b + 2 → u ≥ u_min :=
by sorry

end min_value_of_u_l2961_296104


namespace power_function_inequality_l2961_296172

-- Define the power function
def f (x : ℝ) : ℝ := x^(4/5)

-- State the theorem
theorem power_function_inequality (x₁ x₂ : ℝ) (h : 0 < x₁ ∧ x₁ < x₂) : 
  f ((x₁ + x₂)/2) > (f x₁ + f x₂)/2 := by
  sorry

end power_function_inequality_l2961_296172


namespace ceiling_examples_ceiling_equals_two_m_range_equation_solutions_l2961_296116

-- Definition of the ceiling function for rational numbers
def ceiling (x : ℚ) : ℤ := Int.ceil x

-- Theorem 1: Calculating specific ceiling values
theorem ceiling_examples : ceiling (4.7) = 5 ∧ ceiling (-5.3) = -5 := by sorry

-- Theorem 2: Relationship when ceiling equals 2
theorem ceiling_equals_two (a : ℚ) : ceiling a = 2 ↔ 1 < a ∧ a ≤ 2 := by sorry

-- Theorem 3: Range of m satisfying the given condition
theorem m_range (m : ℚ) : ceiling (-2*m + 7) = -3 ↔ 5 ≤ m ∧ m < 5.5 := by sorry

-- Theorem 4: Solutions to the equation
theorem equation_solutions (n : ℚ) : ceiling (4.5*n - 2.5) = 3*n + 1 ↔ n = 2 ∨ n = 7/3 := by sorry

end ceiling_examples_ceiling_equals_two_m_range_equation_solutions_l2961_296116


namespace point_in_region_l2961_296176

def region (x y : ℝ) : Prop := 2 * x + y - 6 < 0

theorem point_in_region :
  region 0 1 ∧ ¬region 5 0 ∧ ¬region 0 7 ∧ ¬region 2 3 := by
  sorry

end point_in_region_l2961_296176


namespace streetlight_problem_l2961_296162

/-- The number of ways to select k non-adjacent items from a sequence of n items,
    excluding the first and last items. -/
def non_adjacent_selections (n k : ℕ) : ℕ :=
  Nat.choose (n - k - 1) k

/-- The problem statement -/
theorem streetlight_problem :
  non_adjacent_selections 12 3 = Nat.choose 7 3 := by
  sorry

end streetlight_problem_l2961_296162


namespace flower_bee_butterfly_difference_l2961_296183

theorem flower_bee_butterfly_difference (flowers bees butterflies : ℕ) 
  (h1 : flowers = 12) 
  (h2 : bees = 7) 
  (h3 : butterflies = 4) : 
  (flowers - bees) - butterflies = 1 := by
  sorry

end flower_bee_butterfly_difference_l2961_296183


namespace quadratic_equation_solution_l2961_296192

theorem quadratic_equation_solution : ∃ x : ℝ, 3 * x^2 - 6 * x + 3 = 0 ∧ x = 1 := by sorry

end quadratic_equation_solution_l2961_296192


namespace inequality_solution_set_l2961_296124

theorem inequality_solution_set :
  {x : ℝ | (x + 1) * (2 - x) < 0} = {x : ℝ | x > 2 ∨ x < -1} := by
  sorry

end inequality_solution_set_l2961_296124


namespace popcorn_package_solution_l2961_296152

/-- Represents a package of popcorn buckets -/
structure Package where
  buckets : ℕ
  cost : ℚ

/-- Proves that buying 48 packages of Package B satisfies all conditions -/
theorem popcorn_package_solution :
  let package_b : Package := ⟨9, 8⟩
  let num_packages : ℕ := 48
  let total_buckets : ℕ := num_packages * package_b.buckets
  let total_cost : ℚ := num_packages * package_b.cost
  (total_buckets ≥ 426) ∧ 
  (total_cost ≤ 400) ∧ 
  (num_packages ≤ 60) :=
by
  sorry


end popcorn_package_solution_l2961_296152


namespace problem_solution_l2961_296186

theorem problem_solution (a b : ℚ) (h1 : 7 * a + 3 * b = 0) (h2 : a = 2 * b - 3) :
  5 * b - 4 * a = 141 / 17 := by
  sorry

end problem_solution_l2961_296186


namespace complex_square_plus_self_l2961_296198

theorem complex_square_plus_self (z : ℂ) :
  z = -1/2 + (Complex.I * Real.sqrt 3) / 2 →
  z^2 + z = -1 := by
sorry

end complex_square_plus_self_l2961_296198


namespace range_of_g_l2961_296120

noncomputable def g (x : ℝ) : ℝ := Real.arctan x + Real.arctan ((2 - x) / (2 + x))

theorem range_of_g :
  Set.range g = {y | -π/2 ≤ y ∧ y ≤ Real.arctan 2} :=
sorry

end range_of_g_l2961_296120


namespace count_four_digit_with_seven_l2961_296146

/-- A four-digit positive integer with 7 as the thousands digit -/
def FourDigitWithSeven : Type := { n : ℕ // 7000 ≤ n ∧ n ≤ 7999 }

/-- The count of four-digit positive integers with 7 as the thousands digit -/
def CountFourDigitWithSeven : ℕ := Finset.card (Finset.filter (λ n => 7000 ≤ n ∧ n ≤ 7999) (Finset.range 10000))

theorem count_four_digit_with_seven :
  CountFourDigitWithSeven = 1000 := by
  sorry

end count_four_digit_with_seven_l2961_296146


namespace solution_set_equality_l2961_296171

theorem solution_set_equality : {x : ℤ | (3*x - 1)*(x + 3) = 0} = {-3} := by
  sorry

end solution_set_equality_l2961_296171


namespace square_of_97_l2961_296188

theorem square_of_97 : 97^2 = 9409 := by
  sorry

end square_of_97_l2961_296188


namespace sequence_values_bound_l2961_296167

theorem sequence_values_bound (f : ℝ → ℝ) (h : ∀ x : ℝ, f (x + 2) = -f x) :
  let a : ℕ → ℝ := λ n => f n
  ∃ S : Finset ℝ, S.card ≤ 4 ∧ ∀ n : ℕ, a n ∈ S :=
by
  sorry

end sequence_values_bound_l2961_296167


namespace octahedron_sum_theorem_l2961_296105

/-- Represents an octahedron with numbered faces -/
structure NumberedOctahedron where
  lowest_number : ℕ
  face_count : ℕ
  is_consecutive : Bool
  opposite_faces_diff : ℕ

/-- The sum of numbers on an octahedron with the given properties -/
def octahedron_sum (o : NumberedOctahedron) : ℕ :=
  8 * o.lowest_number + 28

/-- Theorem stating the sum of numbers on the octahedron -/
theorem octahedron_sum_theorem (o : NumberedOctahedron) :
  o.face_count = 8 ∧ 
  o.is_consecutive = true ∧ 
  o.opposite_faces_diff = 2 →
  octahedron_sum o = 8 * o.lowest_number + 28 :=
by
  sorry

#check octahedron_sum_theorem

end octahedron_sum_theorem_l2961_296105


namespace equilateral_triangle_point_distances_l2961_296129

theorem equilateral_triangle_point_distances 
  (h x y z : ℝ) 
  (h_pos : h > 0)
  (inside_triangle : x > 0 ∧ y > 0 ∧ z > 0)
  (height_sum : h = x + y + z)
  (triangle_inequality : x + y > z ∧ y + z > x ∧ z + x > y) :
  x < h/2 ∧ y < h/2 ∧ z < h/2 :=
by sorry

end equilateral_triangle_point_distances_l2961_296129


namespace evergreen_elementary_grade6_l2961_296115

theorem evergreen_elementary_grade6 (total : ℕ) (grade4 : ℕ) (grade5 : ℕ) 
  (h1 : total = 100)
  (h2 : grade4 = 30)
  (h3 : grade5 = 35) :
  total - grade4 - grade5 = 35 := by
  sorry

end evergreen_elementary_grade6_l2961_296115


namespace x_minus_y_values_l2961_296135

theorem x_minus_y_values (x y : ℝ) 
  (hx : |x| = 4) 
  (hy : |y| = 2) 
  (hxy : |x + y| = x + y) : 
  x - y = 2 ∨ x - y = 6 := by
sorry

end x_minus_y_values_l2961_296135


namespace age_difference_proof_l2961_296195

theorem age_difference_proof (ann_age susan_age : ℕ) : 
  ann_age > susan_age →
  ann_age + susan_age = 27 →
  susan_age = 11 →
  ann_age - susan_age = 5 := by
sorry

end age_difference_proof_l2961_296195


namespace tax_difference_is_twenty_cents_l2961_296185

/-- The price of the item before tax -/
def price : ℝ := 40

/-- The first tax rate as a percentage -/
def tax_rate1 : ℝ := 7.25

/-- The second tax rate as a percentage -/
def tax_rate2 : ℝ := 6.75

/-- Theorem stating the difference between the two tax amounts -/
theorem tax_difference_is_twenty_cents :
  (price * (tax_rate1 / 100)) - (price * (tax_rate2 / 100)) = 0.20 := by
  sorry

end tax_difference_is_twenty_cents_l2961_296185


namespace first_player_always_wins_l2961_296127

/-- Represents a position on the rectangular table -/
structure Position :=
  (x : ℤ) (y : ℤ)

/-- Represents the state of the game -/
structure GameState :=
  (table : Set Position)
  (occupied : Set Position)
  (currentPlayer : Bool)

/-- The winning strategy for the first player -/
def firstPlayerStrategy (initialState : GameState) : Prop :=
  ∃ (strategy : GameState → Position),
    ∀ (state : GameState),
      state.currentPlayer = true →
      strategy state ∉ state.occupied →
      strategy state ∈ state.table

/-- The main theorem stating that the first player always has a winning strategy -/
theorem first_player_always_wins :
  ∀ (initialState : GameState),
    initialState.occupied = ∅ →
    initialState.table.Nonempty →
    ∃ (center : Position), center ∈ initialState.table →
      firstPlayerStrategy initialState :=
sorry

end first_player_always_wins_l2961_296127


namespace disrespectful_quadratic_max_root_sum_l2961_296107

/-- A quadratic polynomial with real coefficients and leading coefficient 1 -/
def QuadraticPolynomial (b c : ℝ) := fun (x : ℝ) ↦ x^2 + b*x + c

/-- The condition for a polynomial to be "disrespectful" -/
def isDisrespectful (p : ℝ → ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ : ℝ), (∀ x : ℝ, p (p x) = 0 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃)

/-- The sum of roots of a quadratic polynomial -/
def sumOfRoots (b c : ℝ) : ℝ := -b

theorem disrespectful_quadratic_max_root_sum (b c : ℝ) :
  let p := QuadraticPolynomial b c
  isDisrespectful p ∧ 
  (∀ b' c' : ℝ, isDisrespectful (QuadraticPolynomial b' c') → sumOfRoots b' c' ≤ sumOfRoots b c) →
  p 1 = 5/16 := by
  sorry

end disrespectful_quadratic_max_root_sum_l2961_296107


namespace equilateral_triangle_25_division_equilateral_triangle_5_equal_parts_l2961_296134

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  side_length : ℝ
  side_length_pos : side_length > 0

/-- Represents a division of an equilateral triangle -/
structure TriangleDivision where
  original : EquilateralTriangle
  num_divisions : ℕ
  num_divisions_pos : num_divisions > 0

/-- Theorem: An equilateral triangle can be divided into 25 smaller equilateral triangles -/
theorem equilateral_triangle_25_division (t : EquilateralTriangle) :
  ∃ (d : TriangleDivision), d.original = t ∧ d.num_divisions = 25 :=
sorry

/-- Represents a grouping of the divided triangles -/
structure TriangleGrouping where
  division : TriangleDivision
  num_groups : ℕ
  num_groups_pos : num_groups > 0
  triangles_per_group : ℕ
  triangles_per_group_pos : triangles_per_group > 0
  valid_grouping : division.num_divisions = num_groups * triangles_per_group

/-- Theorem: The 25 smaller triangles can be grouped into 5 equal parts -/
theorem equilateral_triangle_5_equal_parts (t : EquilateralTriangle) :
  ∃ (g : TriangleGrouping), g.division.original = t ∧ g.num_groups = 5 ∧ g.triangles_per_group = 5 :=
sorry

end equilateral_triangle_25_division_equilateral_triangle_5_equal_parts_l2961_296134


namespace power_zero_equals_one_l2961_296141

theorem power_zero_equals_one (x : ℝ) (hx : x ≠ 0) : x ^ 0 = 1 := by
  sorry

end power_zero_equals_one_l2961_296141


namespace walk_distance_before_rest_l2961_296181

theorem walk_distance_before_rest 
  (total_distance : ℝ) 
  (distance_after_rest : ℝ) 
  (h1 : total_distance = 1) 
  (h2 : distance_after_rest = 0.25) : 
  total_distance - distance_after_rest = 0.75 := by
sorry

end walk_distance_before_rest_l2961_296181


namespace inverse_trig_sum_zero_l2961_296169

theorem inverse_trig_sum_zero : 
  Real.arctan (Real.sqrt 3 / 3) + Real.arcsin (-1/2) + Real.arccos 1 = 0 := by
  sorry

end inverse_trig_sum_zero_l2961_296169


namespace johns_journey_distance_l2961_296142

/-- Calculates the total distance traveled by John given his journey conditions -/
def total_distance (
  initial_driving_speed : ℝ)
  (initial_driving_time : ℝ)
  (second_driving_speed : ℝ)
  (second_driving_time : ℝ)
  (biking_speed : ℝ)
  (biking_time : ℝ)
  (walking_speed : ℝ)
  (walking_time : ℝ) : ℝ :=
  initial_driving_speed * initial_driving_time +
  second_driving_speed * second_driving_time +
  biking_speed * biking_time +
  walking_speed * walking_time

/-- Theorem stating that John's total travel distance is 179 miles -/
theorem johns_journey_distance : 
  total_distance 55 2 45 1 15 1.5 3 0.5 = 179 := by
  sorry

end johns_journey_distance_l2961_296142


namespace oscar_class_count_l2961_296137

/-- The number of questions per student on the final exam -/
def questions_per_student : ℕ := 10

/-- The number of students per class -/
def students_per_class : ℕ := 35

/-- The total number of questions to review -/
def total_questions : ℕ := 1750

/-- The number of classes Professor Oscar has -/
def number_of_classes : ℕ := total_questions / (questions_per_student * students_per_class)

theorem oscar_class_count :
  number_of_classes = 5 := by
  sorry

end oscar_class_count_l2961_296137


namespace os_value_l2961_296117

/-- Square with center and points on its diagonals -/
structure SquareWithPoints where
  /-- Side length of the square -/
  a : ℝ
  /-- Center of the square -/
  O : ℝ × ℝ
  /-- Point P on OA -/
  P : ℝ × ℝ
  /-- Point Q on OB -/
  Q : ℝ × ℝ
  /-- Point R on OC -/
  R : ℝ × ℝ
  /-- Point S on OD -/
  S : ℝ × ℝ
  /-- A is a vertex of the square -/
  A : ℝ × ℝ
  /-- B is a vertex of the square -/
  B : ℝ × ℝ
  /-- C is a vertex of the square -/
  C : ℝ × ℝ
  /-- D is a vertex of the square -/
  D : ℝ × ℝ
  /-- O is the center of the square ABCD -/
  h_center : O = (0, 0)
  /-- ABCD is a square with side length 2a -/
  h_square : A = (-a, a) ∧ B = (a, a) ∧ C = (a, -a) ∧ D = (-a, -a)
  /-- P is on OA with OP = 3 -/
  h_P : P = (-3*a/Real.sqrt 2, 3*a/Real.sqrt 2) ∧ Real.sqrt ((P.1 - O.1)^2 + (P.2 - O.2)^2) = 3
  /-- Q is on OB with OQ = 5 -/
  h_Q : Q = (5*a/Real.sqrt 2, 5*a/Real.sqrt 2) ∧ Real.sqrt ((Q.1 - O.1)^2 + (Q.2 - O.2)^2) = 5
  /-- R is on OC with OR = 4 -/
  h_R : R = (4*a/Real.sqrt 2, -4*a/Real.sqrt 2) ∧ Real.sqrt ((R.1 - O.1)^2 + (R.2 - O.2)^2) = 4
  /-- S is on OD -/
  h_S : ∃ x : ℝ, S = (-x*a/Real.sqrt 2, -x*a/Real.sqrt 2)
  /-- X is the intersection of AB and PQ -/
  h_X : ∃ X : ℝ × ℝ, X.2 = a ∧ X.2 = (1/4)*X.1 + 15*a/(4*Real.sqrt 2)
  /-- Y is the intersection of BC and QR -/
  h_Y : ∃ Y : ℝ × ℝ, Y.1 = a ∧ Y.2 = 9*Y.1 - 40*a/Real.sqrt 2
  /-- Z is the intersection of CD and RS -/
  h_Z : ∃ Z : ℝ × ℝ, Z.2 = -a ∧ Z.2 + 4*a/Real.sqrt 2 = (4*a - S.1)/(-(4*a + S.1)) * (Z.1 - 4*a/Real.sqrt 2)
  /-- X, Y, and Z are collinear -/
  h_collinear : ∀ X Y Z : ℝ × ℝ, 
    (X.2 = a ∧ X.2 = (1/4)*X.1 + 15*a/(4*Real.sqrt 2)) →
    (Y.1 = a ∧ Y.2 = 9*Y.1 - 40*a/Real.sqrt 2) →
    (Z.2 = -a ∧ Z.2 + 4*a/Real.sqrt 2 = (4*a - S.1)/(-(4*a + S.1)) * (Z.1 - 4*a/Real.sqrt 2)) →
    (Y.2 - X.2)*(Z.1 - X.1) = (Z.2 - X.2)*(Y.1 - X.1)

/-- The main theorem: OS = 60/23 -/
theorem os_value (sq : SquareWithPoints) : 
  Real.sqrt ((sq.S.1 - sq.O.1)^2 + (sq.S.2 - sq.O.2)^2) = 60/23 := by
  sorry

end os_value_l2961_296117


namespace roses_distribution_l2961_296177

def distribute_roses (initial : ℕ) (stolen : ℕ) (recipients : ℕ) : ℕ :=
  (initial - stolen) / recipients

theorem roses_distribution (initial : ℕ) (stolen : ℕ) (recipients : ℕ)
  (h1 : initial = 40)
  (h2 : stolen = 4)
  (h3 : recipients = 9)
  : distribute_roses initial stolen recipients = 4 := by
  sorry

end roses_distribution_l2961_296177


namespace courses_difference_count_l2961_296179

/-- The number of available courses -/
def total_courses : ℕ := 4

/-- The number of courses each person chooses -/
def courses_per_person : ℕ := 2

/-- The number of ways to choose courses with at least one difference -/
def ways_with_difference : ℕ := 30

/-- Theorem stating that the number of ways with at least one course different is 30 -/
theorem courses_difference_count :
  (total_courses.choose courses_per_person * courses_per_person.choose courses_per_person) +
  (total_courses.choose 1 * (total_courses - 1).choose 1 * (total_courses - 2).choose 1) =
  ways_with_difference :=
sorry

end courses_difference_count_l2961_296179


namespace equality_condition_l2961_296193

theorem equality_condition (x y : ℝ) : 
  (x - 9)^2 + (y - 10)^2 + (x - y)^2 = 1/3 → x = 28/3 ∧ y = 29/3 := by
  sorry

end equality_condition_l2961_296193


namespace opposite_reciprocal_equation_l2961_296111

theorem opposite_reciprocal_equation (a b c d : ℝ) 
  (h1 : a + b = 0)  -- a and b are opposites
  (h2 : c * d = 1)  -- c and d are reciprocals
  : (a + b)^2 - 3*(c*d)^4 = -3 := by
  sorry

end opposite_reciprocal_equation_l2961_296111


namespace pie_eating_contest_l2961_296133

theorem pie_eating_contest (a b c : ℚ) 
  (ha : a = 5/6) (hb : b = 2/3) (hc : c = 3/4) : 
  (max a (max b c) - min a (min b c)) = 1/6 :=
sorry

end pie_eating_contest_l2961_296133


namespace scooter_repair_cost_l2961_296113

/-- 
Given a scooter purchase, we define the following:
purchase_price: The initial cost of the scooter
selling_price: The price at which the scooter was sold
gain_percent: The percentage gain on the sale
repair_cost: The amount spent on repairs

We prove that the repair cost satisfies the equation relating these variables.
-/
theorem scooter_repair_cost 
  (purchase_price : ℝ) 
  (selling_price : ℝ) 
  (gain_percent : ℝ) 
  (repair_cost : ℝ) 
  (h1 : purchase_price = 4400)
  (h2 : selling_price = 5800)
  (h3 : gain_percent = 0.1154) :
  selling_price = (purchase_price + repair_cost) * (1 + gain_percent) :=
by sorry

end scooter_repair_cost_l2961_296113


namespace intersection_complement_equality_l2961_296197

def U : Set Nat := {1, 2, 3, 4, 5}
def M : Set Nat := {1, 4}
def N : Set Nat := {1, 3, 5}

theorem intersection_complement_equality : N ∩ (U \ M) = {3, 5} := by
  sorry

end intersection_complement_equality_l2961_296197


namespace shepherd_sheep_equations_correct_l2961_296184

/-- Represents the number of sheep each shepherd has -/
structure ShepherdSheep where
  a : ℤ  -- number of sheep A has
  b : ℤ  -- number of sheep B has

/-- Checks if the given system of equations satisfies the conditions of the problem -/
def satisfies_conditions (s : ShepherdSheep) : Prop :=
  (s.a + 9 = 2 * (s.b - 9)) ∧ (s.b + 9 = s.a - 9)

/-- Theorem stating that the system of equations correctly represents the problem -/
theorem shepherd_sheep_equations_correct :
  ∃ (s : ShepherdSheep), satisfies_conditions s :=
sorry

end shepherd_sheep_equations_correct_l2961_296184


namespace arithmetic_geometric_ratio_l2961_296190

/-- An arithmetic sequence with common difference d ≠ 0 -/
def ArithmeticSequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  d ≠ 0 ∧ ∀ n, a (n + 1) = a n + d

/-- Three terms of a sequence form a geometric sequence -/
def FormGeometricSequence (a b c : ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ b = a * q ∧ c = b * q

theorem arithmetic_geometric_ratio 
  (a : ℕ → ℝ) (d : ℝ) 
  (h_arith : ArithmeticSequence a d)
  (h_geom : FormGeometricSequence (a 2) (a 3) (a 6)) :
  ∃ q : ℝ, q = 3 ∧ FormGeometricSequence (a 2) (a 3) (a 6) := by
  sorry

end arithmetic_geometric_ratio_l2961_296190


namespace simplify_fraction_l2961_296149

theorem simplify_fraction (b : ℚ) (h : b = 2) : 15 * b^4 / (45 * b^3) = 2/3 := by
  sorry

end simplify_fraction_l2961_296149


namespace combinatorial_identities_l2961_296108

-- Define combinatorial choice function
def C (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- Define permutation function
def A (n m : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial (n - m))

theorem combinatorial_identities :
  (3 * C 8 3 - 2 * C 5 2 = 148) ∧
  (∀ n m : ℕ, n ≥ m → m ≥ 2 → A n m = n * A (n-1) (m-1)) :=
by sorry

end combinatorial_identities_l2961_296108


namespace unique_n_satisfying_conditions_l2961_296130

theorem unique_n_satisfying_conditions : ∃! (n : ℕ), n ≥ 1 ∧
  ∃ (a b : ℕ+), 
    (∀ (p : ℕ), Prime p → ¬(p^3 ∣ (a.val^2 + b.val + 3))) ∧
    ((a.val * b.val + 3 * b.val + 8) : ℚ) / (a.val^2 + b.val + 3) = n ∧
    n = 2 := by
  sorry

end unique_n_satisfying_conditions_l2961_296130


namespace concurrent_circles_and_collinearity_l2961_296182

-- Define the basic structures
structure Point := (x y : ℝ)

structure Triangle :=
  (A B C : Point)

structure Circle :=
  (center : Point) (radius : ℝ)

-- Define the given conditions
def D (triangle : Triangle) : Point := sorry
def E (triangle : Triangle) : Point := sorry
def F (triangle : Triangle) : Point := sorry

-- Define the circles
def circleAEF (triangle : Triangle) : Circle := sorry
def circleBFD (triangle : Triangle) : Circle := sorry
def circleCDE (triangle : Triangle) : Circle := sorry

-- Define concurrency
def areConcurrent (c1 c2 c3 : Circle) : Prop := sorry

-- Define collinearity
def areCollinear (p1 p2 p3 : Point) : Prop := sorry

-- Define if a point lies on a circle
def liesOnCircle (p : Point) (c : Circle) : Prop := sorry

-- Define the circumcircle of a triangle
def circumcircle (triangle : Triangle) : Circle := sorry

-- The theorem to prove
theorem concurrent_circles_and_collinearity 
  (triangle : Triangle) : 
  areConcurrent (circleAEF triangle) (circleBFD triangle) (circleCDE triangle) ∧ 
  (∃ M : Point, 
    liesOnCircle M (circleAEF triangle) ∧ 
    liesOnCircle M (circleBFD triangle) ∧ 
    liesOnCircle M (circleCDE triangle) ∧
    (liesOnCircle M (circumcircle triangle) ↔ 
      areCollinear (D triangle) (E triangle) (F triangle))) := by
  sorry

end concurrent_circles_and_collinearity_l2961_296182


namespace largest_fraction_l2961_296159

theorem largest_fraction :
  let fractions := [2/5, 3/7, 4/9, 5/11, 6/13]
  ∀ x ∈ fractions, (6/13 : ℚ) ≥ x := by
sorry

end largest_fraction_l2961_296159


namespace card_arrangement_probability_l2961_296106

theorem card_arrangement_probability : 
  let total_arrangements : ℕ := 24
  let favorable_arrangements : ℕ := 2
  let probability : ℚ := favorable_arrangements / total_arrangements
  probability = 1 / 12 := by sorry

end card_arrangement_probability_l2961_296106


namespace rhombus_side_length_l2961_296170

/-- A rhombus with area K and one diagonal three times the length of the other has side length √(5K/3) -/
theorem rhombus_side_length (K : ℝ) (K_pos : K > 0) : 
  ∃ (d₁ d₂ s : ℝ), d₁ > 0 ∧ d₂ > 0 ∧ s > 0 ∧ 
  d₂ = 3 * d₁ ∧ 
  K = (1/2) * d₁ * d₂ ∧
  s^2 = (d₁/2)^2 + (d₂/2)^2 ∧
  s = Real.sqrt ((5 * K) / 3) :=
by sorry

end rhombus_side_length_l2961_296170


namespace jack_letters_difference_l2961_296166

theorem jack_letters_difference (morning_emails morning_letters afternoon_emails afternoon_letters : ℕ) :
  morning_emails = 6 →
  morning_letters = 8 →
  afternoon_emails = 2 →
  afternoon_letters = 7 →
  morning_letters - afternoon_letters = 1 := by
  sorry

end jack_letters_difference_l2961_296166


namespace factor_expression_l2961_296168

theorem factor_expression (b : ℝ) : 49 * b^2 + 98 * b = 49 * b * (b + 2) := by
  sorry

end factor_expression_l2961_296168


namespace purchasing_plan_comparison_pricing_strategy_comparison_l2961_296180

-- Purchasing plans comparison
theorem purchasing_plan_comparison 
  (a b : ℝ) (m n : ℝ) (h1 : a ≠ b) (h2 : a > 0) (h3 : b > 0) (h4 : m > 0) (h5 : n > 0) :
  (2 * a * b) / (a + b) < (a + b) / 2 := by
sorry

-- Pricing strategies comparison
theorem pricing_strategy_comparison 
  (p q : ℝ) (h : p ≠ q) :
  100 * (1 + p) * (1 + q) < 100 * (1 + (p + q) / 2)^2 := by
sorry

end purchasing_plan_comparison_pricing_strategy_comparison_l2961_296180


namespace right_triangle_area_l2961_296102

theorem right_triangle_area (a b : ℝ) (h1 : a^2 - 7*a + 12 = 0) (h2 : b^2 - 7*b + 12 = 0) (h3 : a ≠ b) :
  ∃ (area : ℝ), (area = 6 ∨ area = (3 * Real.sqrt 7) / 2) ∧
  ((area = a * b / 2) ∨ (area = a * Real.sqrt (b^2 - a^2) / 2) ∨ (area = b * Real.sqrt (a^2 - b^2) / 2)) :=
by sorry

end right_triangle_area_l2961_296102


namespace inequality_proof_l2961_296160

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (1 + a / b) * (1 + b / c) * (1 + c / a) ≥ 2 * (1 + (a + b + c) / Real.rpow (a * b * c) (1/3)) := by
  sorry

end inequality_proof_l2961_296160


namespace modulus_of_z_l2961_296143

/-- The modulus of the complex number z = 1 / (i - 1) is equal to √2/2 -/
theorem modulus_of_z (z : ℂ) : z = 1 / (Complex.I - 1) → Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end modulus_of_z_l2961_296143


namespace cubic_polynomial_satisfies_conditions_l2961_296100

def q (x : ℚ) : ℚ := -20/93 * x^3 - 110/93 * x^2 - 372/93 * x - 525/93

theorem cubic_polynomial_satisfies_conditions :
  q 1 = -11 ∧ q 2 = -15 ∧ q 3 = -25 ∧ q 5 = -65 := by
  sorry

end cubic_polynomial_satisfies_conditions_l2961_296100


namespace equation_solution_l2961_296128

theorem equation_solution : ∃ x : ℚ, (4 * x - 2) / (5 * x - 5) = 3 / 4 ∧ x = -7 := by
  sorry

end equation_solution_l2961_296128


namespace workshop_efficiency_l2961_296119

theorem workshop_efficiency (x : ℝ) : 
  (1500 / x - 1500 / (2.5 * x) = 18) → x = 50 :=
by
  sorry

end workshop_efficiency_l2961_296119


namespace investment_sum_l2961_296194

theorem investment_sum (raghu_investment : ℕ) : 
  raghu_investment = 2100 →
  let trishul_investment := raghu_investment - raghu_investment / 10
  let vishal_investment := trishul_investment + trishul_investment / 10
  raghu_investment + trishul_investment + vishal_investment = 6069 := by
  sorry

end investment_sum_l2961_296194


namespace expression_nonpositive_l2961_296161

theorem expression_nonpositive (x : ℝ) : (6 * x - 1) / 4 - 2 * x ≤ 0 ↔ x ≥ -1/2 := by
  sorry

end expression_nonpositive_l2961_296161


namespace fraction_powers_equality_l2961_296153

theorem fraction_powers_equality : (0.4 ^ 4) / (0.04 ^ 3) = 400 := by
  sorry

end fraction_powers_equality_l2961_296153


namespace quadratic_solution_difference_squared_l2961_296114

theorem quadratic_solution_difference_squared : 
  ∀ a b : ℝ, (2 * a^2 - 7 * a + 3 = 0) → 
             (2 * b^2 - 7 * b + 3 = 0) → 
             (a ≠ b) →
             (a - b)^2 = 25/4 := by
  sorry

end quadratic_solution_difference_squared_l2961_296114


namespace spend_representation_l2961_296150

-- Define a type for monetary transactions
inductive Transaction
| receive (amount : ℤ)
| spend (amount : ℤ)

-- Define a function to represent transactions
def represent : Transaction → ℤ
| Transaction.receive amount => amount
| Transaction.spend amount => -amount

-- Theorem statement
theorem spend_representation (amount : ℤ) :
  represent (Transaction.receive amount) = amount →
  represent (Transaction.spend amount) = -amount :=
by
  sorry

end spend_representation_l2961_296150


namespace Al2O3_weight_and_H2_volume_l2961_296163

/-- Molar mass of Aluminum in g/mol -/
def molar_mass_Al : ℝ := 26.98

/-- Molar mass of Oxygen in g/mol -/
def molar_mass_O : ℝ := 16.00

/-- Volume occupied by 1 mole of gas at STP in liters -/
def molar_volume_STP : ℝ := 22.4

/-- Molar mass of Al2O3 in g/mol -/
def molar_mass_Al2O3 : ℝ := 2 * molar_mass_Al + 3 * molar_mass_O

/-- Number of moles of Al2O3 -/
def moles_Al2O3 : ℝ := 6

/-- Theorem stating the weight of Al2O3 and volume of H2 produced -/
theorem Al2O3_weight_and_H2_volume :
  (moles_Al2O3 * molar_mass_Al2O3 = 611.76) ∧
  (moles_Al2O3 * 3 * molar_volume_STP = 403.2) := by
  sorry

end Al2O3_weight_and_H2_volume_l2961_296163


namespace coefficient_of_term_l2961_296178

theorem coefficient_of_term (x y : ℝ) : 
  ∃ (c : ℝ), -π * x * y^3 / 5 = c * x * y^3 ∧ c = -π / 5 := by
  sorry

end coefficient_of_term_l2961_296178


namespace total_winter_clothing_l2961_296174

/-- Represents the contents of a box of winter clothing -/
structure BoxContents where
  scarves : ℕ
  mittens : ℕ
  hats : ℕ

/-- Calculates the total number of items in a box -/
def totalItemsInBox (box : BoxContents) : ℕ :=
  box.scarves + box.mittens + box.hats

/-- The contents of the four boxes -/
def box1 : BoxContents := { scarves := 3, mittens := 5, hats := 2 }
def box2 : BoxContents := { scarves := 4, mittens := 3, hats := 1 }
def box3 : BoxContents := { scarves := 2, mittens := 6, hats := 3 }
def box4 : BoxContents := { scarves := 1, mittens := 7, hats := 2 }

/-- Theorem stating that the total number of winter clothing items is 39 -/
theorem total_winter_clothing : 
  totalItemsInBox box1 + totalItemsInBox box2 + totalItemsInBox box3 + totalItemsInBox box4 = 39 := by
  sorry

end total_winter_clothing_l2961_296174


namespace six_digit_divisible_by_7_8_9_l2961_296148

theorem six_digit_divisible_by_7_8_9 :
  ∃ (n₁ n₂ : ℕ),
    523000 ≤ n₁ ∧ n₁ < 524000 ∧
    523000 ≤ n₂ ∧ n₂ < 524000 ∧
    n₁ ≠ n₂ ∧
    n₁ % 7 = 0 ∧ n₁ % 8 = 0 ∧ n₁ % 9 = 0 ∧
    n₂ % 7 = 0 ∧ n₂ % 8 = 0 ∧ n₂ % 9 = 0 ∧
    n₁ = 523152 ∧ n₂ = 523656 :=
by sorry

end six_digit_divisible_by_7_8_9_l2961_296148


namespace cereal_box_servings_l2961_296123

theorem cereal_box_servings (total_cups : ℕ) (serving_size : ℕ) (h1 : total_cups = 18) (h2 : serving_size = 2) :
  total_cups / serving_size = 9 := by
  sorry

end cereal_box_servings_l2961_296123


namespace correct_dispersion_measure_l2961_296118

-- Define a type for measures of data dispersion
structure DisperesionMeasure where
  makeFullUseOfData : Bool
  useMultipleNumericalValues : Bool
  smallerValueForLargerDispersion : Bool

-- Define a function to check if a dispersion measure is correct
def isCorrectMeasure (m : DisperesionMeasure) : Prop :=
  m.makeFullUseOfData ∧ m.useMultipleNumericalValues ∧ ¬m.smallerValueForLargerDispersion

-- Theorem: The correct dispersion measure makes full use of data and uses multiple numerical values
theorem correct_dispersion_measure :
  ∃ (m : DisperesionMeasure), isCorrectMeasure m ∧
    m.makeFullUseOfData = true ∧
    m.useMultipleNumericalValues = true :=
  sorry


end correct_dispersion_measure_l2961_296118


namespace angle_inequality_equivalence_l2961_296131

theorem angle_inequality_equivalence (θ : Real) : 
  (∀ x : Real, 0 ≤ x ∧ x ≤ 1 → 
    x^3 * Real.sin θ + x^2 * Real.cos θ - x * (1 - x) + (1 - x)^2 * Real.sin θ > 0) ↔ 
  (π / 12 < θ ∧ θ < 5 * π / 12) :=
by sorry

end angle_inequality_equivalence_l2961_296131


namespace second_reduction_percentage_l2961_296112

theorem second_reduction_percentage (P : ℝ) (R : ℝ) (h1 : P > 0) :
  (1 - R / 100) * (0.75 * P) = 0.375 * P →
  R = 50 := by
sorry

end second_reduction_percentage_l2961_296112


namespace jellybean_count_l2961_296156

theorem jellybean_count (black green orange : ℕ) 
  (green_count : green = black + 2)
  (orange_count : orange = green - 1)
  (total_count : black + green + orange = 27) :
  black = 8 := by
  sorry

end jellybean_count_l2961_296156


namespace second_cat_blue_eyes_l2961_296136

/-- The number of blue-eyed kittens the first cat has -/
def first_cat_blue : ℕ := 3

/-- The number of brown-eyed kittens the first cat has -/
def first_cat_brown : ℕ := 7

/-- The number of brown-eyed kittens the second cat has -/
def second_cat_brown : ℕ := 6

/-- The percentage of kittens with blue eyes -/
def blue_eye_percentage : ℚ := 35 / 100

/-- The number of blue-eyed kittens the second cat has -/
def second_cat_blue : ℕ := 4

theorem second_cat_blue_eyes :
  (first_cat_blue + second_cat_blue : ℚ) / 
  (first_cat_blue + first_cat_brown + second_cat_blue + second_cat_brown) = 
  blue_eye_percentage := by
  sorry

#check second_cat_blue_eyes

end second_cat_blue_eyes_l2961_296136


namespace purchase_payment_possible_l2961_296122

theorem purchase_payment_possible :
  ∃ (x y : ℕ), x ≤ 15 ∧ y ≤ 15 ∧ 3 * x - 5 * y = 19 :=
sorry

end purchase_payment_possible_l2961_296122


namespace days_to_empty_tube_l2961_296189

-- Define the volume of the gel tube in mL
def tube_volume : ℝ := 128

-- Define the daily usage of gel in mL
def daily_usage : ℝ := 4

-- Theorem statement
theorem days_to_empty_tube : 
  (tube_volume / daily_usage : ℝ) = 32 := by
  sorry

end days_to_empty_tube_l2961_296189


namespace number_of_children_selected_l2961_296191

def total_boys : ℕ := 5
def total_girls : ℕ := 5
def prob_three_boys_three_girls : ℚ := 100 / 210

theorem number_of_children_selected (n : ℕ) : 
  (total_boys = 5 ∧ total_girls = 5 ∧ 
   prob_three_boys_three_girls = 100 / (Nat.choose (total_boys + total_girls) n)) → 
  n = 6 := by
  sorry

end number_of_children_selected_l2961_296191


namespace circle_op_five_three_l2961_296157

-- Define the operation ∘
def circle_op (a b : ℕ) : ℕ := 4*a + 6*b + 1

-- State the theorem
theorem circle_op_five_three : circle_op 5 3 = 39 := by sorry

end circle_op_five_three_l2961_296157


namespace book_sale_earnings_l2961_296173

/-- Calculates the total earnings from a book sale --/
theorem book_sale_earnings (total_books : ℕ) (price_high : ℚ) (price_low : ℚ) : 
  total_books = 10 ∧ 
  price_high = 5/2 ∧ 
  price_low = 2 → 
  (2/5 * total_books : ℚ) * price_high + (3/5 * total_books : ℚ) * price_low = 22 := by
  sorry

#check book_sale_earnings

end book_sale_earnings_l2961_296173


namespace minimum_value_and_range_l2961_296199

def f (x a : ℝ) : ℝ := |x - 4| + |x - a|

theorem minimum_value_and_range (a : ℝ) :
  (∀ x, f x a ≥ 3) ∧ (∃ x, f x a = 3) →
  ((a = 1 ∨ a = 7) ∧
   (a = 1 → ∀ x, f x a ≤ 5 ↔ 0 ≤ x ∧ x ≤ 5) ∧
   (a = 7 → ∀ x, f x a ≤ 5 ↔ 3 ≤ x ∧ x ≤ 8)) :=
by sorry

end minimum_value_and_range_l2961_296199


namespace divisible_by_6_up_to_88_eq_l2961_296164

def divisible_by_6_up_to_88 : Set ℕ :=
  {n | 1 < n ∧ n ≤ 88 ∧ n % 6 = 0}

theorem divisible_by_6_up_to_88_eq :
  divisible_by_6_up_to_88 = {6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84} := by
  sorry

end divisible_by_6_up_to_88_eq_l2961_296164


namespace loom_weaving_time_l2961_296121

/-- The rate at which the loom weaves cloth in meters per second -/
def weaving_rate : ℝ := 0.128

/-- The time it takes to weave 15 meters of cloth in seconds -/
def time_for_15_meters : ℝ := 117.1875

/-- The amount of cloth woven in 15 meters -/
def cloth_amount : ℝ := 15

theorem loom_weaving_time (C : ℝ) :
  C ≥ 0 →
  weaving_rate > 0 →
  time_for_15_meters * weaving_rate = cloth_amount →
  C / weaving_rate = (C : ℝ) / 0.128 := by
  sorry

end loom_weaving_time_l2961_296121


namespace cubic_equation_integer_solution_l2961_296132

theorem cubic_equation_integer_solution :
  ∃! (x : ℤ), 2 * x^3 + 5 * x^2 - 9 * x - 18 = 0 :=
by
  -- The proof goes here
  sorry

end cubic_equation_integer_solution_l2961_296132


namespace cookie_pie_ratio_is_seven_fourths_l2961_296125

/-- The ratio of students preferring cookies to those preferring pie -/
def cookie_pie_ratio (total_students : ℕ) (cookie_preference : ℕ) (pie_preference : ℕ) : ℚ :=
  cookie_preference / pie_preference

theorem cookie_pie_ratio_is_seven_fourths :
  cookie_pie_ratio 800 280 160 = 7/4 := by
  sorry

end cookie_pie_ratio_is_seven_fourths_l2961_296125


namespace a_zero_sufficient_for_P_range_of_a_when_only_one_true_l2961_296196

-- Define the propositions P and Q as functions of a
def P (a : ℝ) : Prop := ∀ x : ℝ, a * x^2 + a * x + 1 > 0
def Q (a : ℝ) : Prop := ∃ x : ℝ, x^2 - x + a = 0

-- Theorem 1: a = 0 is a sufficient condition for P
theorem a_zero_sufficient_for_P : ∀ a : ℝ, a = 0 → P a := by sorry

-- Theorem 2: The range of a when only one of P and Q is true
theorem range_of_a_when_only_one_true : 
  {a : ℝ | (P a ∧ ¬Q a) ∨ (¬P a ∧ Q a)} = 
  {a : ℝ | a < 0 ∨ (1/4 < a ∧ a < 4)} := by sorry

end a_zero_sufficient_for_P_range_of_a_when_only_one_true_l2961_296196


namespace cubic_equation_solution_l2961_296101

theorem cubic_equation_solution :
  ∃ y : ℝ, y > 0 ∧ 6 * y^(1/3) - 3 * (y / y^(2/3)) = 12 + 2 * y^(1/3) ∧ y = 1728 := by
  sorry

end cubic_equation_solution_l2961_296101


namespace solution_inequality1_solution_inequality2_l2961_296165

-- Define the inequalities
def inequality1 (x : ℝ) : Prop := 2 * x^2 + x - 3 < 0
def inequality2 (x : ℝ) : Prop := x * (9 - x) > 0

-- Define the solution sets
def solution_set1 : Set ℝ := {x | -3/2 < x ∧ x < 1}
def solution_set2 : Set ℝ := {x | 0 < x ∧ x < 9}

-- Theorem statements
theorem solution_inequality1 : {x : ℝ | inequality1 x} = solution_set1 := by sorry

theorem solution_inequality2 : {x : ℝ | inequality2 x} = solution_set2 := by sorry

end solution_inequality1_solution_inequality2_l2961_296165


namespace even_pairs_ge_odd_pairs_l2961_296151

/-- A sequence of zeros and ones -/
def BinarySequence := List Bool

/-- Count pairs (1,0) with even number of digits between them -/
def countEvenPairs (seq : BinarySequence) : ℕ := sorry

/-- Count pairs (1,0) with odd number of digits between them -/
def countOddPairs (seq : BinarySequence) : ℕ := sorry

/-- Theorem: In any binary sequence, the number of (1,0) pairs with even digits between
    is greater than or equal to the number of (1,0) pairs with odd digits between -/
theorem even_pairs_ge_odd_pairs (seq : BinarySequence) :
  countEvenPairs seq ≥ countOddPairs seq := by
  sorry

end even_pairs_ge_odd_pairs_l2961_296151


namespace last_two_digits_of_factorial_sum_factorial_ends_with_zeros_l2961_296103

def factorial_sum_last_two_digits : ℕ := 46

theorem last_two_digits_of_factorial_sum :
  let sum := List.sum (List.map Nat.factorial (List.range 25 |>.map (fun i => 4 * i + 3)))
  (sum % 100 = factorial_sum_last_two_digits) :=
by
  sorry

theorem factorial_ends_with_zeros (n : ℕ) (h : n ≥ 10) :
  ∃ k : ℕ, Nat.factorial n = 100 * k :=
by
  sorry

end last_two_digits_of_factorial_sum_factorial_ends_with_zeros_l2961_296103

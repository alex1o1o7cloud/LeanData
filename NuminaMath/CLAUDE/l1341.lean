import Mathlib

namespace fraction_sum_l1341_134169

theorem fraction_sum (x y : ℚ) (h : x / y = 5 / 2) : (x + y) / y = 7 / 2 := by
  sorry

end fraction_sum_l1341_134169


namespace billy_crayons_l1341_134158

theorem billy_crayons (jane_crayons : ℝ) (total_crayons : ℕ) 
  (h1 : jane_crayons = 52.0) 
  (h2 : total_crayons = 114) : 
  ↑total_crayons - jane_crayons = 62 := by
  sorry

end billy_crayons_l1341_134158


namespace percussion_probability_l1341_134178

def total_sounds : ℕ := 6
def percussion_sounds : ℕ := 3

theorem percussion_probability :
  (percussion_sounds.choose 2 : ℚ) / (total_sounds.choose 2) = 1 / 5 := by
  sorry

end percussion_probability_l1341_134178


namespace alex_growth_rate_l1341_134198

/-- Alex's growth rate problem -/
theorem alex_growth_rate :
  let required_height : ℚ := 54
  let current_height : ℚ := 48
  let growth_rate_upside_down : ℚ := 1 / 12
  let hours_upside_down_per_month : ℚ := 2
  let months_in_year : ℕ := 12
  let height_difference := required_height - current_height
  let growth_from_hanging := growth_rate_upside_down * hours_upside_down_per_month * months_in_year
  let natural_growth := height_difference - growth_from_hanging
  natural_growth / months_in_year = 1 / 3 := by
sorry


end alex_growth_rate_l1341_134198


namespace part1_part2_l1341_134156

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + 2*|x + 1|

-- Part 1
theorem part1 : 
  {x : ℝ | f 2 x > 4} = {x : ℝ | x < -4/3 ∨ x > 0} := by sorry

-- Part 2
theorem part2 : 
  ({x : ℝ | f a x < 3*x + 4} = {x : ℝ | x > 2}) → a = 6 := by sorry

end part1_part2_l1341_134156


namespace sin_neg_pi_half_l1341_134157

theorem sin_neg_pi_half : Real.sin (-π / 2) = -1 := by sorry

end sin_neg_pi_half_l1341_134157


namespace tom_reading_pages_l1341_134120

theorem tom_reading_pages (total_hours : ℕ) (total_days : ℕ) (pages_per_hour : ℕ) (target_days : ℕ) :
  total_hours = 10 →
  total_days = 5 →
  pages_per_hour = 50 →
  target_days = 7 →
  (total_hours / total_days) * pages_per_hour * target_days = 700 :=
by
  sorry

end tom_reading_pages_l1341_134120


namespace similar_triangles_leg_length_l1341_134168

/-- Two similar right triangles with legs 12 and 9 in the first triangle, 
    and y and 6 in the second triangle, have y equal to 8 -/
theorem similar_triangles_leg_length : ∀ y : ℝ,
  (12 : ℝ) / y = 9 / 6 → y = 8 := by sorry

end similar_triangles_leg_length_l1341_134168


namespace additional_amount_needed_l1341_134189

def fundraiser_goal : ℕ := 750
def bronze_donation : ℕ := 25
def silver_donation : ℕ := 50
def gold_donation : ℕ := 100
def bronze_count : ℕ := 10
def silver_count : ℕ := 7
def gold_count : ℕ := 1

theorem additional_amount_needed : 
  fundraiser_goal - (bronze_donation * bronze_count + silver_donation * silver_count + gold_donation * gold_count) = 50 := by
  sorry

end additional_amount_needed_l1341_134189


namespace min_teachers_for_given_problem_l1341_134192

/-- Represents the number of teachers for each subject -/
structure SubjectTeachers where
  english : Nat
  history : Nat
  geography : Nat

/-- The minimum number of teachers required given the subject teachers -/
def minTeachersRequired (s : SubjectTeachers) : Nat :=
  sorry

/-- Theorem stating the minimum number of teachers required for the given problem -/
theorem min_teachers_for_given_problem :
  let s : SubjectTeachers := ⟨9, 7, 6⟩
  minTeachersRequired s = 13 := by
  sorry

end min_teachers_for_given_problem_l1341_134192


namespace sqrt_sum_squares_equals_sum_l1341_134193

theorem sqrt_sum_squares_equals_sum (a b c : ℝ) :
  Real.sqrt (a^2 + b^2 + c^2) = a + b + c ↔ a*b + b*c + c*a = 0 ∧ a + b + c ≥ 0 := by
  sorry

end sqrt_sum_squares_equals_sum_l1341_134193


namespace cos_2x_quadratic_equation_l1341_134181

theorem cos_2x_quadratic_equation (a b c : ℝ) :
  ∃ (f : ℝ → ℝ), 
    (∀ x, a * (Real.cos x)^2 + b * Real.cos x + c = 0) →
    (∀ x, f (Real.cos (2 * x)) = 0) ∧
    (∃ p q r : ℝ, ∀ y, f y = p * y^2 + q * y + r ∧
      p = a^2 ∧
      q = 2 * (a^2 + 2 * a * c - b^2) ∧
      r = (a^2 + 2 * c)^2 - 2 * b^2) :=
by sorry

end cos_2x_quadratic_equation_l1341_134181


namespace intersection_complement_equality_l1341_134106

def U : Finset Nat := {0, 1, 2, 3, 4}
def A : Finset Nat := {0, 1, 3}
def B : Finset Nat := {2, 3}

theorem intersection_complement_equality : A ∩ (U \ B) = {0, 1} := by
  sorry

end intersection_complement_equality_l1341_134106


namespace initial_strawberries_l1341_134162

/-- The number of strawberries Paul picked -/
def picked : ℕ := 78

/-- The total number of strawberries Paul had after picking more -/
def total : ℕ := 120

/-- The initial number of strawberries in Paul's basket -/
def initial : ℕ := total - picked

theorem initial_strawberries : initial + picked = total := by
  sorry

end initial_strawberries_l1341_134162


namespace delta_theta_solution_l1341_134137

theorem delta_theta_solution :
  ∃ (Δ Θ : ℤ), 4 * 3 = Δ - 5 + Θ ∧ Θ = 14 ∧ Δ = 3 := by
  sorry

end delta_theta_solution_l1341_134137


namespace partition_fifth_power_l1341_134112

/-- Number of partitions of a 1 × n rectangle into 1 × 1 squares and broken dominoes -/
def F (n : ℕ) : ℕ :=
  sorry

/-- A broken domino consists of two 1 × 1 squares separated by four squares -/
def is_broken_domino (tile : List (ℕ × ℕ)) : Prop :=
  tile.length = 2 ∧ ∃ i : ℕ, tile = [(i, 1), (i + 5, 1)]

/-- A valid tiling of a 1 × n rectangle -/
def valid_tiling (n : ℕ) (tiling : List (List (ℕ × ℕ))) : Prop :=
  (tiling.join.map Prod.fst).toFinset = Finset.range n ∧
  ∀ tile ∈ tiling, tile.length = 1 ∨ is_broken_domino tile

theorem partition_fifth_power (n : ℕ) :
  (F (5 * n) : ℕ) = (F n) ^ 5 :=
sorry

end partition_fifth_power_l1341_134112


namespace frank_fence_length_l1341_134159

/-- Given a rectangular yard with one side of 40 feet and an area of 320 square feet,
    the perimeter minus one side is 56 feet. -/
theorem frank_fence_length :
  ∀ (length width : ℝ),
    length = 40 →
    length * width = 320 →
    2 * width + length = 56 :=
by
  sorry

end frank_fence_length_l1341_134159


namespace marble_selection_ways_l1341_134134

def total_marbles : ℕ := 9
def marbles_to_choose : ℕ := 4
def remaining_marbles : ℕ := total_marbles - 1
def remaining_to_choose : ℕ := marbles_to_choose - 1

theorem marble_selection_ways :
  (remaining_marbles.choose remaining_to_choose) = 56 := by
  sorry

end marble_selection_ways_l1341_134134


namespace no_solution_factorial_power_l1341_134165

theorem no_solution_factorial_power (n k : ℕ) (hn : n > 5) (hk : k > 0) :
  (Nat.factorial (n - 1) + 1 ≠ n ^ k) := by
  sorry

end no_solution_factorial_power_l1341_134165


namespace intersection_point_theorem_l1341_134133

/-- A plane in 3D space -/
structure Plane

/-- A line in 3D space -/
structure Line

/-- A point in 3D space -/
structure Point

/-- The intersection of two planes is a line -/
def plane_intersection (p1 p2 : Plane) : Line :=
  sorry

/-- A point lies on a line -/
def point_on_line (p : Point) (l : Line) : Prop :=
  sorry

/-- A point lies on a plane -/
def point_on_plane (p : Point) (pl : Plane) : Prop :=
  sorry

theorem intersection_point_theorem 
  (α β γ : Plane) 
  (M : Point) :
  let a := plane_intersection α β
  let b := plane_intersection α γ
  let c := plane_intersection β γ
  (point_on_line M a ∧ point_on_line M b) → 
  point_on_line M c :=
by
  sorry

end intersection_point_theorem_l1341_134133


namespace remaining_average_of_prime_numbers_l1341_134117

theorem remaining_average_of_prime_numbers 
  (total_count : Nat) 
  (subset_count : Nat) 
  (total_average : ℚ) 
  (subset_average : ℚ) 
  (h1 : total_count = 20) 
  (h2 : subset_count = 10) 
  (h3 : total_average = 95) 
  (h4 : subset_average = 85) : 
  (total_count * total_average - subset_count * subset_average) / (total_count - subset_count) = 105 := by
sorry

end remaining_average_of_prime_numbers_l1341_134117


namespace greatest_distance_between_circle_centers_l1341_134123

theorem greatest_distance_between_circle_centers
  (rectangle_width : ℝ)
  (rectangle_height : ℝ)
  (circle_diameter : ℝ)
  (h1 : rectangle_width = 20)
  (h2 : rectangle_height = 15)
  (h3 : circle_diameter = 8)
  (h4 : circle_diameter ≤ rectangle_width)
  (h5 : circle_diameter ≤ rectangle_height) :
  let max_horizontal_distance := rectangle_width - circle_diameter
  let max_vertical_distance := rectangle_height - circle_diameter
  Real.sqrt (max_horizontal_distance ^ 2 + max_vertical_distance ^ 2) = Real.sqrt 193 :=
by sorry

end greatest_distance_between_circle_centers_l1341_134123


namespace angle_BXY_is_30_degrees_l1341_134136

-- Define the points and angles
variable (A B C D X Y E : Point)
variable (angle_AXE angle_CYX angle_BXY : ℝ)

-- Define the parallel lines condition
variable (h1 : Parallel (Line.mk A B) (Line.mk C D))

-- Define the angle relationship
variable (h2 : angle_AXE = 4 * angle_CYX - 90)

-- Define the equality of alternate interior angles
variable (h3 : angle_AXE = angle_CYX)

-- Define the relationship between BXY and AXE due to parallel lines
variable (h4 : angle_BXY = angle_AXE)

-- State the theorem
theorem angle_BXY_is_30_degrees :
  angle_BXY = 30 := by sorry

end angle_BXY_is_30_degrees_l1341_134136


namespace percentage_of_boats_eaten_by_fish_l1341_134184

theorem percentage_of_boats_eaten_by_fish 
  (initial_boats : ℕ) 
  (shot_boats : ℕ) 
  (remaining_boats : ℕ) 
  (h1 : initial_boats = 30) 
  (h2 : shot_boats = 2) 
  (h3 : remaining_boats = 22) : 
  (initial_boats - shot_boats - remaining_boats) / initial_boats * 100 = 20 := by
  sorry

end percentage_of_boats_eaten_by_fish_l1341_134184


namespace student_council_committees_l1341_134146

theorem student_council_committees (n : ℕ) : 
  (n.choose 2 = 15) → (n.choose 3 = 20) :=
by sorry

end student_council_committees_l1341_134146


namespace negation_equivalence_l1341_134126

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x > 0 ∧ x^2 - x ≤ 0) ↔ (∀ x : ℝ, x > 0 → x^2 - x > 0) :=
by sorry

end negation_equivalence_l1341_134126


namespace dartboard_angle_measure_l1341_134147

/-- The measure of the central angle of a region on a circular dartboard, given its probability -/
theorem dartboard_angle_measure (p : ℝ) (h : p = 1 / 8) : 
  p * 360 = 45 := by sorry

end dartboard_angle_measure_l1341_134147


namespace g_15_equals_281_l1341_134188

/-- The function g defined for all natural numbers -/
def g (n : ℕ) : ℕ := n^2 + n + 41

/-- Theorem stating that g(15) equals 281 -/
theorem g_15_equals_281 : g 15 = 281 := by
  sorry

end g_15_equals_281_l1341_134188


namespace solution_set_part_i_value_of_a_l1341_134116

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + 3 * x

-- Part I
theorem solution_set_part_i (a : ℝ) (h : a = 2) :
  {x : ℝ | f a x ≥ 3 * x + 2} = {x : ℝ | x ≥ 4 ∨ x ≤ 0} :=
sorry

-- Part II
theorem value_of_a (a : ℝ) (h : a > 0) :
  ({x : ℝ | f a x ≤ 0} = {x : ℝ | x ≤ -1}) → a = 2 :=
sorry

end solution_set_part_i_value_of_a_l1341_134116


namespace color_selection_ways_l1341_134144

def total_colors : ℕ := 10
def colors_to_choose : ℕ := 3
def remaining_colors : ℕ := total_colors - 1  -- Subtracting blue

theorem color_selection_ways :
  (total_colors.choose colors_to_choose) - (remaining_colors.choose (colors_to_choose - 1)) =
  remaining_colors.choose (colors_to_choose - 1) := by
  sorry

end color_selection_ways_l1341_134144


namespace inequality_solution_set_l1341_134195

theorem inequality_solution_set (x : ℝ) :
  (1 / (x + 2) + 4 / (x + 4) ≥ 1) ↔ (x > -2 ∧ x ≤ 1) :=
by sorry

end inequality_solution_set_l1341_134195


namespace william_bottle_caps_l1341_134186

/-- Given that William initially had 2 bottle caps and now has 43 bottle caps in total,
    prove that he bought 41 bottle caps. -/
theorem william_bottle_caps :
  let initial_caps : ℕ := 2
  let total_caps : ℕ := 43
  let bought_caps : ℕ := total_caps - initial_caps
  bought_caps = 41 := by sorry

end william_bottle_caps_l1341_134186


namespace exists_special_sequence_l1341_134132

/-- A sequence of natural numbers with the property that all natural numbers
    appear exactly once as differences between its members. -/
def special_sequence : Set ℕ → Prop :=
  λ S => (∀ n : ℕ, ∃! (a b : ℕ), a ∈ S ∧ b ∈ S ∧ a > b ∧ a - b = n) ∧
         (∀ a : ℕ, a ∈ S → ∃ b : ℕ, b > a ∧ b ∈ S)

/-- Theorem stating the existence of a special sequence of natural numbers. -/
theorem exists_special_sequence : ∃ S : Set ℕ, special_sequence S :=
  sorry


end exists_special_sequence_l1341_134132


namespace smallest_x_properties_l1341_134101

/-- The smallest integer with 18 positive factors that is divisible by both 18 and 24 -/
def smallest_x : ℕ := 288

/-- The number of positive factors of smallest_x -/
def factor_count : ℕ := 18

theorem smallest_x_properties :
  (∃ (factors : Finset ℕ), factors.card = factor_count ∧ 
    ∀ d ∈ factors, d ∣ smallest_x) ∧
  18 ∣ smallest_x ∧
  24 ∣ smallest_x ∧
  ∀ y : ℕ, y < smallest_x →
    ¬(∃ (factors : Finset ℕ), factors.card = factor_count ∧
      ∀ d ∈ factors, d ∣ y ∧ 18 ∣ y ∧ 24 ∣ y) :=
by
  sorry

#eval smallest_x

end smallest_x_properties_l1341_134101


namespace train_length_l1341_134103

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmph : ℝ) (time_seconds : ℝ) : 
  speed_kmph = 360 → time_seconds = 5 → speed_kmph * (5 / 18) * time_seconds = 500 := by
  sorry

#check train_length

end train_length_l1341_134103


namespace simplify_expression_l1341_134149

theorem simplify_expression (x : ℝ) :
  Real.sqrt (x^6 + 3*x^4 + 2*x^2) = |x| * Real.sqrt ((x^2 + 1) * (x^2 + 2)) := by
  sorry

end simplify_expression_l1341_134149


namespace imaginary_part_of_2i_plus_1_l1341_134124

theorem imaginary_part_of_2i_plus_1 :
  let z : ℂ := 2 * Complex.I * (1 + Complex.I)
  (z.im : ℝ) = 2 := by sorry

end imaginary_part_of_2i_plus_1_l1341_134124


namespace map_distance_to_actual_distance_l1341_134135

/-- Given a map scale and a distance on the map, calculate the actual distance -/
theorem map_distance_to_actual_distance 
  (scale : ℚ) 
  (map_distance : ℚ) 
  (h_scale : scale = 1 / 10000) 
  (h_map_distance : map_distance = 16) : 
  let actual_distance := map_distance / scale
  actual_distance = 1600 := by sorry

end map_distance_to_actual_distance_l1341_134135


namespace problem_statement_l1341_134155

-- Define the base conversion function
def baseToDecimal (digits : List Nat) (base : Nat) : Nat :=
  digits.foldl (fun acc d => acc * base + d) 0

-- Define the problem statement
theorem problem_statement (A B : Nat) (h1 : A = 2 * B) 
  (h2 : baseToDecimal [2, 2, 4] B + baseToDecimal [5, 5] A = baseToDecimal [1, 3, 4] (A + B)) :
  A + B = 9 := by
  sorry

end problem_statement_l1341_134155


namespace sin_inequality_l1341_134119

theorem sin_inequality (n : ℕ) (hn : n > 0) :
  Real.sin (1 / n) + Real.sin (2 / n) > (3 / n) * Real.cos (1 / n) := by
  sorry

end sin_inequality_l1341_134119


namespace dirt_pile_volume_decomposition_l1341_134108

/-- Represents the dimensions of a rectangular storage bin -/
structure BinDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents the parameters of a dirt pile around the storage bin -/
structure DirtPileParams where
  slantDistance : ℝ

/-- Calculates the volume of the dirt pile around a storage bin -/
def dirtPileVolume (bin : BinDimensions) (pile : DirtPileParams) : ℝ :=
  sorry

theorem dirt_pile_volume_decomposition (bin : BinDimensions) (pile : DirtPileParams) :
  bin.length = 10 ∧ bin.width = 12 ∧ bin.height = 3 ∧ pile.slantDistance = 4 →
  ∃ (m n : ℕ), dirtPileVolume bin pile = m + n * Real.pi ∧ m + n = 280 :=
sorry

end dirt_pile_volume_decomposition_l1341_134108


namespace quadratic_roots_imply_d_value_l1341_134153

theorem quadratic_roots_imply_d_value (d : ℝ) : 
  (∀ x : ℝ, 2 * x^2 + 9 * x + d = 0 ↔ x = (-9 + Real.sqrt 17) / 4 ∨ x = (-9 - Real.sqrt 17) / 4) →
  d = 8 := by
sorry

end quadratic_roots_imply_d_value_l1341_134153


namespace fair_hair_percentage_l1341_134102

/-- Given a company where 10% of all employees are women with fair hair,
    and 40% of fair-haired employees are women,
    prove that 25% of all employees have fair hair. -/
theorem fair_hair_percentage
  (total_employees : ℝ)
  (women_fair_hair_percentage : ℝ)
  (women_among_fair_hair_percentage : ℝ)
  (h1 : women_fair_hair_percentage = 0.1)
  (h2 : women_among_fair_hair_percentage = 0.4)
  : (women_fair_hair_percentage / women_among_fair_hair_percentage) * 100 = 25 := by
  sorry

end fair_hair_percentage_l1341_134102


namespace cement_mixture_percentage_l1341_134174

/-- Calculates the percentage of cement in the second mixture for concrete production --/
theorem cement_mixture_percentage 
  (total_concrete : Real) 
  (final_cement_percentage : Real)
  (first_mixture_percentage : Real)
  (second_mixture_amount : Real) :
  let total_cement := total_concrete * final_cement_percentage / 100
  let first_mixture_amount := total_concrete - second_mixture_amount
  let first_mixture_cement := first_mixture_amount * first_mixture_percentage / 100
  let second_mixture_cement := total_cement - first_mixture_cement
  second_mixture_cement / second_mixture_amount * 100 = 80 :=
by
  sorry

#check cement_mixture_percentage 10 62 20 7

end cement_mixture_percentage_l1341_134174


namespace exam_score_problem_l1341_134121

theorem exam_score_problem (total_questions : ℕ) (correct_score : ℤ) (wrong_score : ℤ) (total_score : ℤ) :
  total_questions = 50 →
  correct_score = 4 →
  wrong_score = -1 →
  total_score = 130 →
  ∃ (correct_answers : ℕ),
    correct_answers ≤ total_questions ∧
    correct_score * correct_answers + wrong_score * (total_questions - correct_answers) = total_score ∧
    correct_answers = 36 := by
  sorry

end exam_score_problem_l1341_134121


namespace triangle_area_formula_right_angle_l1341_134173

theorem triangle_area_formula_right_angle (a b : ℝ) (h : a > 0 ∧ b > 0) :
  (1/2) * (a * b) / Real.sin (π/2) = (1/2) * a * b := by
  sorry

end triangle_area_formula_right_angle_l1341_134173


namespace marias_additional_cupcakes_l1341_134179

/-- Given that Maria initially made 19 cupcakes, sold 5, and ended up with 24 cupcakes,
    prove that she made 10 additional cupcakes. -/
theorem marias_additional_cupcakes :
  let initial_cupcakes : ℕ := 19
  let sold_cupcakes : ℕ := 5
  let final_cupcakes : ℕ := 24
  let additional_cupcakes := final_cupcakes - (initial_cupcakes - sold_cupcakes)
  additional_cupcakes = 10 := by sorry

end marias_additional_cupcakes_l1341_134179


namespace inscribed_circle_inequality_l1341_134138

variable (a b c u v w : ℝ)

-- a, b, c are positive real numbers representing side lengths of a triangle
variable (ha : a > 0) (hb : b > 0) (hc : c > 0)

-- u, v, w are positive real numbers representing distances from incenter to opposite vertices
variable (hu : u > 0) (hv : v > 0) (hw : w > 0)

-- Triangle inequality
variable (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b)

theorem inscribed_circle_inequality :
  (a + b + c) * (1/u + 1/v + 1/w) ≤ 3 * (a/u + b/v + c/w) := by
  sorry

end inscribed_circle_inequality_l1341_134138


namespace sum_of_reciprocal_squares_l1341_134107

-- Define the cubic polynomial
def cubic_poly (x : ℝ) : ℝ := x^3 - 7*x^2 + 3*x + 4

-- Define the roots of the polynomial
axiom a : ℝ
axiom b : ℝ
axiom c : ℝ

-- State that a, b, c are roots of the polynomial
axiom a_root : cubic_poly a = 0
axiom b_root : cubic_poly b = 0
axiom c_root : cubic_poly c = 0

-- State that a, b, c are distinct
axiom roots_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c

-- Theorem to prove
theorem sum_of_reciprocal_squares : 
  1/a^2 + 1/b^2 + 1/c^2 = 65/16 := by sorry

end sum_of_reciprocal_squares_l1341_134107


namespace arithmetic_mean_of_sequence_l1341_134170

def integer_sequence : List Int := List.range 10 |>.map (λ x => x - 5)

theorem arithmetic_mean_of_sequence (seq : List Int := integer_sequence) :
  seq.sum / seq.length = 0 := by
  sorry

end arithmetic_mean_of_sequence_l1341_134170


namespace tank_capacity_l1341_134140

/-- Represents a cylindrical tank with a given capacity and current fill level. -/
structure CylindricalTank where
  capacity : ℝ
  fill_percentage : ℝ
  current_volume : ℝ

/-- 
Theorem: Given a cylindrical tank that contains 60 liters of water when it is 40% full, 
the total capacity of the tank when it is completely full is 150 liters.
-/
theorem tank_capacity (tank : CylindricalTank) 
  (h1 : tank.fill_percentage = 0.4)
  (h2 : tank.current_volume = 60) : 
  tank.capacity = 150 := by
  sorry

#check tank_capacity

end tank_capacity_l1341_134140


namespace product_equals_square_l1341_134115

theorem product_equals_square : 500 * 3986 * 0.3986 * 20 = (3986 : ℝ)^2 := by
  sorry

end product_equals_square_l1341_134115


namespace complement_union_theorem_l1341_134185

def U : Set ℤ := {x | 1 ≤ x ∧ x ≤ 6}
def A : Set ℤ := {1, 3, 4}
def B : Set ℤ := {2, 4}

theorem complement_union_theorem :
  (U \ A) ∪ B = {2, 4, 5, 6} := by sorry

end complement_union_theorem_l1341_134185


namespace sum_of_coordinates_A_l1341_134180

/-- Given three points A, B, and C in a plane, where C divides AB in a 1:2 ratio,
    and the coordinates of B and C are known, prove that the sum of A's coordinates is 9. -/
theorem sum_of_coordinates_A (A B C : ℝ × ℝ) : 
  (C.1 - A.1) / (B.1 - A.1) = 1/3 →
  (B.1 - C.1) / (B.1 - A.1) = 2/3 →
  B = (2, 8) →
  C = (5, 2) →
  A.1 + A.2 = 9 := by
sorry

end sum_of_coordinates_A_l1341_134180


namespace travis_apple_sale_price_l1341_134177

/-- Calculates the price per box of apples given the total number of apples,
    apples per box, and desired total revenue. -/
def price_per_box (total_apples : ℕ) (apples_per_box : ℕ) (total_revenue : ℕ) : ℚ :=
  (total_revenue : ℚ) / ((total_apples / apples_per_box) : ℚ)

/-- Proves that given Travis's conditions, he must sell each box for $35. -/
theorem travis_apple_sale_price :
  price_per_box 10000 50 7000 = 35 := by
  sorry

end travis_apple_sale_price_l1341_134177


namespace perfect_square_divisibility_l1341_134187

theorem perfect_square_divisibility (a p q : ℕ+) (h1 : ∃ k : ℕ+, a = k ^ 2) 
  (h2 : a = p * q) (h3 : (2021 : ℕ) ∣ p ^ 3 + q ^ 3 + p ^ 2 * q + p * q ^ 2) :
  (2021 : ℕ) ∣ Nat.sqrt a.val := by sorry

end perfect_square_divisibility_l1341_134187


namespace cube_side_area_l1341_134143

/-- Given a cube with volume 125 cubic decimeters, 
    prove that the surface area of one side is 2500 square centimeters. -/
theorem cube_side_area (volume : ℝ) (side_length : ℝ) : 
  volume = 125 →
  side_length^3 = volume →
  (side_length * 10)^2 = 2500 := by
sorry

end cube_side_area_l1341_134143


namespace operation_equality_l1341_134105

-- Define a custom type for the allowed operations
inductive Operation
  | Add
  | Sub
  | Mul
  | Div

-- Define a function to apply the operation
def applyOp (op : Operation) (a b : ℚ) : ℚ :=
  match op with
  | Operation.Add => a + b
  | Operation.Sub => a - b
  | Operation.Mul => a * b
  | Operation.Div => a / b

-- State the theorem
theorem operation_equality (star mul : Operation) :
  (applyOp star 20 5) / (applyOp mul 15 5) = 1 →
  (applyOp star 8 4) / (applyOp mul 10 2) = 1/5 := by
  sorry

end operation_equality_l1341_134105


namespace fraction_simplification_l1341_134191

theorem fraction_simplification :
  let x := (1/2 - 1/3) / (3/7 + 1/9)
  x * (1/4) = 21/272 := by sorry

end fraction_simplification_l1341_134191


namespace circle_tangent_to_lines_l1341_134130

/-- A circle with center (0, k) is tangent to lines y = x, y = -x, y = 10, and y = -4x. -/
theorem circle_tangent_to_lines (k : ℝ) (h : k > 10) :
  let r := 10 * Real.sqrt 34 * (Real.sqrt 2 / (Real.sqrt 2 - 1 / Real.sqrt 17)) - 10 * Real.sqrt 2
  ∃ (circle : Set (ℝ × ℝ)),
    (∀ (x y : ℝ), (x, y) ∈ circle ↔ (x^2 + (y - k)^2 = r^2)) ∧
    (∃ (x₁ y₁ : ℝ), (x₁, y₁) ∈ circle ∧ y₁ = x₁) ∧
    (∃ (x₂ y₂ : ℝ), (x₂, y₂) ∈ circle ∧ y₂ = -x₂) ∧
    (∃ (x₃ y₃ : ℝ), (x₃, y₃) ∈ circle ∧ y₃ = 10) ∧
    (∃ (x₄ y₄ : ℝ), (x₄, y₄) ∈ circle ∧ y₄ = -4*x₄) :=
by sorry

end circle_tangent_to_lines_l1341_134130


namespace vector_addition_scalar_multiplication_l1341_134125

/-- Given two 2D vectors a and b, prove that a + 3b equals the specified result. -/
theorem vector_addition_scalar_multiplication 
  (a b : ℝ × ℝ) 
  (ha : a = (2, 3)) 
  (hb : b = (-1, 5)) : 
  a + 3 • b = (-1, 18) := by
  sorry

end vector_addition_scalar_multiplication_l1341_134125


namespace prob_two_boys_from_three_boys_one_girl_l1341_134104

/-- The probability of selecting 2 boys from a group of 3 boys and 1 girl is 1/2 -/
theorem prob_two_boys_from_three_boys_one_girl :
  let total_students : ℕ := 4
  let num_boys : ℕ := 3
  let num_girls : ℕ := 1
  let students_to_select : ℕ := 2
  (Nat.choose num_boys students_to_select : ℚ) / (Nat.choose total_students students_to_select) = 1 / 2 :=
by sorry

end prob_two_boys_from_three_boys_one_girl_l1341_134104


namespace last_three_average_l1341_134176

theorem last_three_average (numbers : List ℝ) : 
  numbers.length = 6 →
  numbers.sum / 6 = 60 →
  (numbers.take 3).sum / 3 = 55 →
  (numbers.drop 3).sum = 195 →
  (numbers.drop 3).sum / 3 = 65 := by
sorry

end last_three_average_l1341_134176


namespace solution_set_inequalities_l1341_134114

theorem solution_set_inequalities :
  {x : ℝ | x - 2 > 1 ∧ x < 4} = {x : ℝ | 3 < x ∧ x < 4} := by sorry

end solution_set_inequalities_l1341_134114


namespace special_triangle_properties_l1341_134113

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Given conditions for the triangle -/
def SpecialTriangle (t : Triangle) : Prop :=
  t.a * Real.cos t.C + t.c * Real.cos t.A = 4 * t.b * Real.cos t.B ∧
  t.b = 2 * Real.sqrt 19 ∧
  (1 / 2) * t.a * t.b * Real.sin t.C = 6 * Real.sqrt 15

theorem special_triangle_properties (t : Triangle) (h : SpecialTriangle t) :
  Real.cos t.B = 1 / 4 ∧ t.a + t.b + t.c = 14 + 2 * Real.sqrt 19 := by
  sorry

end special_triangle_properties_l1341_134113


namespace weightlifting_ratio_l1341_134166

theorem weightlifting_ratio (total weight_first weight_second : ℕ) 
  (h1 : total = weight_first + weight_second)
  (h2 : weight_first = 700)
  (h3 : 2 * weight_first = weight_second + 300)
  (h4 : total = 1800) : 
  weight_first * 11 = weight_second * 7 := by
  sorry

end weightlifting_ratio_l1341_134166


namespace secretary_project_hours_l1341_134161

/-- Proves that given three secretaries whose work times are in the ratio of 2:3:5 and who worked a combined total of 80 hours, the secretary who worked the longest spent 40 hours on the project. -/
theorem secretary_project_hours (t1 t2 t3 : ℝ) : 
  t1 + t2 + t3 = 80 ∧ 
  t2 = (3/2) * t1 ∧ 
  t3 = (5/2) * t1 → 
  t3 = 40 := by
sorry

end secretary_project_hours_l1341_134161


namespace ellipse_condition_l1341_134145

/-- The equation of the curve -/
def curve_equation (x y k : ℝ) : Prop :=
  x^2 + 2*y^2 - 6*x + 24*y = k

/-- The condition for a non-degenerate ellipse -/
def is_non_degenerate_ellipse (k : ℝ) : Prop :=
  ∃ (a b c d e : ℝ), a > 0 ∧ b > 0 ∧
    ∀ (x y : ℝ), curve_equation x y k ↔ ((x - c)^2 / a + (y - d)^2 / b = e)

/-- The theorem stating the condition for the curve to be a non-degenerate ellipse -/
theorem ellipse_condition :
  ∀ k : ℝ, is_non_degenerate_ellipse k ↔ k > -81 :=
sorry

end ellipse_condition_l1341_134145


namespace cornelia_area_is_17_over_6_l1341_134129

/-- Represents an equiangular octagon with alternating side lengths -/
structure EquiangularOctagon where
  side1 : ℝ
  side2 : ℝ

/-- Represents a self-intersecting octagon formed by connecting alternate vertices of an equiangular octagon -/
structure SelfIntersectingOctagon where
  base : EquiangularOctagon

/-- The area enclosed by a self-intersecting octagon -/
def enclosed_area (octagon : SelfIntersectingOctagon) : ℝ := sorry

/-- The theorem stating that the area enclosed by CORNELIA is 17/6 -/
theorem cornelia_area_is_17_over_6 (caroline : EquiangularOctagon) 
  (cornelia : SelfIntersectingOctagon) (h1 : caroline.side1 = Real.sqrt 2) 
  (h2 : caroline.side2 = 1) (h3 : cornelia.base = caroline) : 
  enclosed_area cornelia = 17 / 6 := by sorry

end cornelia_area_is_17_over_6_l1341_134129


namespace trigonometric_equation_solution_l1341_134194

theorem trigonometric_equation_solution (z : ℝ) : 
  (Real.sin (3 * z) + Real.sin z ^ 3 = (3 * Real.sqrt 3 / 4) * Real.sin (2 * z)) ↔ 
  (∃ k : ℤ, z = k * Real.pi) ∨ 
  (∃ n : ℤ, z = Real.pi / 2 * (2 * n + 1)) ∨ 
  (∃ l : ℤ, z = Real.pi / 6 + 2 * Real.pi * l ∨ z = -Real.pi / 6 + 2 * Real.pi * l) :=
by sorry

end trigonometric_equation_solution_l1341_134194


namespace sum_13_eq_26_l1341_134167

/-- An arithmetic sequence. -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

/-- The sum of the first n terms of an arithmetic sequence. -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  (List.range n).map seq.a |>.sum

theorem sum_13_eq_26 (seq : ArithmeticSequence) 
    (h : seq.a 3 + seq.a 7 + seq.a 11 = 6) : 
  sum_n seq 13 = 26 := by
  sorry

end sum_13_eq_26_l1341_134167


namespace function_identification_l1341_134152

-- Define the exponential function
noncomputable def exp (x : ℝ) : ℝ := Real.exp x

-- Define the property of being symmetric about the y-axis
def symmetric_about_y_axis (f g : ℝ → ℝ) : Prop :=
  ∀ x, f x = g (-x)

-- Define the property of being a translation to the right by 1 unit
def translated_right_by_one (f g : ℝ → ℝ) : Prop :=
  ∀ x, f x = g (x - 1)

-- State the theorem
theorem function_identification
  (f g : ℝ → ℝ)
  (h1 : translated_right_by_one f g)
  (h2 : symmetric_about_y_axis g exp) :
  ∀ x, f x = exp (-x - 1) :=
by sorry

end function_identification_l1341_134152


namespace mrs_sheridan_initial_fish_l1341_134111

/-- The number of fish Mrs. Sheridan received from her sister -/
def fish_from_sister : ℕ := 47

/-- The total number of fish Mrs. Sheridan has after receiving fish from her sister -/
def total_fish : ℕ := 69

/-- The initial number of fish Mrs. Sheridan had -/
def initial_fish : ℕ := total_fish - fish_from_sister

theorem mrs_sheridan_initial_fish :
  initial_fish = 22 :=
sorry

end mrs_sheridan_initial_fish_l1341_134111


namespace range_of_f_l1341_134109

-- Define the function f
def f (x : ℝ) : ℝ := |x + 3| - |x - 5|

-- State the theorem about the range of f
theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ -8 ≤ y ∧ y ≤ 8 := by sorry

end range_of_f_l1341_134109


namespace range_of_even_quadratic_function_l1341_134160

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 2

-- Define the property of being an even function
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Define the domain of the function
def domain (a : ℝ) : Set ℝ := Set.Icc (1 + a) 2

-- State the theorem
theorem range_of_even_quadratic_function (a b : ℝ) :
  is_even (f a b) ∧ (∀ x ∈ domain a, f a b x ∈ Set.Icc (-10) 2) →
  Set.range (f a b) = Set.Icc (-10) 2 :=
sorry

end range_of_even_quadratic_function_l1341_134160


namespace sum_of_fractions_inequality_l1341_134199

theorem sum_of_fractions_inequality (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  a / (b + c) + b / (c + d) + c / (d + a) + d / (a + b) ≥ 2 := by sorry

end sum_of_fractions_inequality_l1341_134199


namespace jamie_hourly_rate_l1341_134196

/-- Represents Jamie's flyer delivery job -/
structure FlyerJob where
  days_per_week : ℕ
  hours_per_day : ℕ
  total_weeks : ℕ
  total_earnings : ℕ

/-- Calculates the hourly rate given a flyer delivery job -/
def hourly_rate (job : FlyerJob) : ℚ :=
  job.total_earnings / (job.days_per_week * job.hours_per_day * job.total_weeks)

/-- Theorem stating that Jamie's hourly rate is $10 -/
theorem jamie_hourly_rate :
  let job : FlyerJob := {
    days_per_week := 2,
    hours_per_day := 3,
    total_weeks := 6,
    total_earnings := 360
  }
  hourly_rate job = 10 := by sorry

end jamie_hourly_rate_l1341_134196


namespace ad_time_theorem_l1341_134148

/-- Calculates the total advertisement time in a week given the ad duration and cycle time -/
def total_ad_time_per_week (ad_duration : ℚ) (cycle_time : ℚ) : ℚ :=
  let ads_per_hour : ℚ := 60 / cycle_time
  let ad_time_per_hour : ℚ := ads_per_hour * ad_duration
  let hours_per_week : ℚ := 24 * 7
  ad_time_per_hour * hours_per_week

/-- Converts minutes to hours and minutes -/
def minutes_to_hours_and_minutes (total_minutes : ℚ) : ℚ × ℚ :=
  let hours : ℚ := total_minutes / 60
  let minutes : ℚ := total_minutes % 60
  (hours.floor, minutes)

theorem ad_time_theorem :
  let ad_duration : ℚ := 3/2  -- 1.5 minutes
  let cycle_time : ℚ := 20    -- 20 minutes (including ad duration)
  let total_minutes : ℚ := total_ad_time_per_week ad_duration cycle_time
  let (hours, minutes) := minutes_to_hours_and_minutes total_minutes
  hours = 12 ∧ minutes = 36 := by
  sorry


end ad_time_theorem_l1341_134148


namespace cylinder_to_sphere_l1341_134131

/-- Given a cylinder with base radius 4 and lateral area 16π/3,
    prove its volume and the radius of an equivalent sphere -/
theorem cylinder_to_sphere (r : ℝ) (L : ℝ) (h : ℝ) (V : ℝ) (R : ℝ) :
  r = 4 →
  L = 16 / 3 * Real.pi →
  L = 2 * Real.pi * r * h →
  V = Real.pi * r^2 * h →
  V = 4 / 3 * Real.pi * R^3 →
  V = 32 / 3 * Real.pi ∧ R = 2 := by
  sorry

end cylinder_to_sphere_l1341_134131


namespace first_day_exceeding_2000_l1341_134151

def algae_population (n : ℕ) : ℕ := 5 * 3^n

theorem first_day_exceeding_2000 :
  ∃ n : ℕ, n > 0 ∧ algae_population n > 2000 ∧ ∀ m : ℕ, m < n → algae_population m ≤ 2000 :=
by
  use 7
  sorry

end first_day_exceeding_2000_l1341_134151


namespace nested_expression_evaluation_l1341_134154

theorem nested_expression_evaluation : (3 * (3 * (3 * (3 * (3 * (3 + 2) + 2) + 2) + 2) + 2) + 2) = 1457 := by
  sorry

end nested_expression_evaluation_l1341_134154


namespace absolute_value_inequality_l1341_134110

theorem absolute_value_inequality (x : ℝ) :
  2 ≤ |x - 3| ∧ |x - 3| ≤ 8 ↔ (-5 ≤ x ∧ x ≤ 1) ∨ (5 ≤ x ∧ x ≤ 11) := by
  sorry

end absolute_value_inequality_l1341_134110


namespace floor_sum_example_l1341_134127

theorem floor_sum_example : ⌊(12.7 : ℝ)⌋ + ⌊(-12.7 : ℝ)⌋ = -1 := by
  sorry

end floor_sum_example_l1341_134127


namespace converse_inequality_abs_l1341_134118

theorem converse_inequality_abs (x y : ℝ) : x > |y| → x > y := by
  sorry

end converse_inequality_abs_l1341_134118


namespace max_segments_on_unit_disc_l1341_134182

/-- The maximum number of segments with lengths greater than 1 determined by n points on a unit disc -/
def maxSegments (n : ℕ) : ℚ :=
  2 * n^2 / 5

/-- Theorem stating the maximum number of segments with lengths greater than 1 -/
theorem max_segments_on_unit_disc (n : ℕ) (h : n ≥ 2) :
  maxSegments n = (2 * n^2 : ℚ) / 5 :=
by sorry

end max_segments_on_unit_disc_l1341_134182


namespace product_greater_than_sum_l1341_134197

theorem product_greater_than_sum {a b : ℝ} (ha : a > 2) (hb : b > 2) : a * b > a + b := by
  sorry

end product_greater_than_sum_l1341_134197


namespace a_equals_three_iff_parallel_l1341_134171

def line1 (a : ℝ) (x y : ℝ) : Prop := x + a * y + 2 = 0
def line2 (a : ℝ) (x y : ℝ) : Prop := (a - 2) * x + 3 * y + 6 * a = 0

def parallel (a : ℝ) : Prop := ∀ (x y : ℝ), line1 a x y ↔ ∃ (k : ℝ), line2 a (x + k) (y + k)

theorem a_equals_three_iff_parallel :
  ∀ (a : ℝ), a = 3 ↔ parallel a :=
sorry

end a_equals_three_iff_parallel_l1341_134171


namespace intersection_nonempty_implies_a_geq_neg_one_l1341_134150

def M : Set ℝ := {x | -1 ≤ x ∧ x < 2}
def N (a : ℝ) : Set ℝ := {x | x ≤ a}

theorem intersection_nonempty_implies_a_geq_neg_one (a : ℝ) :
  (M ∩ N a).Nonempty → a ≥ -1 := by
  sorry

end intersection_nonempty_implies_a_geq_neg_one_l1341_134150


namespace average_weight_equation_indeterminate_section_b_size_l1341_134142

theorem average_weight_equation (x : ℕ) : (36 * 30) + (x * 30) = (36 + x) * 30 := by
  sorry

theorem indeterminate_section_b_size : 
  ∀ (x : ℕ), (36 * 30) + (x * 30) = (36 + x) * 30 := by
  sorry

end average_weight_equation_indeterminate_section_b_size_l1341_134142


namespace intersection_implies_a_value_l1341_134172

theorem intersection_implies_a_value (a : ℝ) : 
  let A : Set ℝ := {-1, a^2 + 1, a^2 - 3}
  let B : Set ℝ := {-4, a - 1, a + 1}
  A ∩ B = {-2} → a = -1 := by
sorry

end intersection_implies_a_value_l1341_134172


namespace solution_set_for_a_3_min_value_and_range_l1341_134128

-- Define the functions f and g
def f (a x : ℝ) : ℝ := |2*x - a| + a
def g (x : ℝ) : ℝ := |2*x - 1|

-- Part 1
theorem solution_set_for_a_3 :
  {x : ℝ | f 3 x ≤ 6} = {x : ℝ | 0 ≤ x ∧ x ≤ 3} :=
sorry

-- Part 2
theorem min_value_and_range :
  (∀ x : ℝ, f a x + g x ≥ 3) ↔ a ∈ Set.Ici 2 :=
sorry

#check solution_set_for_a_3
#check min_value_and_range

end solution_set_for_a_3_min_value_and_range_l1341_134128


namespace product_of_three_numbers_l1341_134100

theorem product_of_three_numbers (x y z : ℚ) : 
  x + y + z = 190 ∧ 
  8 * x = y - 7 ∧ 
  8 * x = z + 11 ∧
  x ≤ y ∧ 
  x ≤ z →
  x * y * z = (97 * 215 * 161) / 108 := by
sorry

end product_of_three_numbers_l1341_134100


namespace average_weight_decrease_l1341_134190

theorem average_weight_decrease (n : ℕ) (initial_avg : ℝ) (new_weight : ℝ) :
  n = 30 →
  initial_avg = 102 →
  new_weight = 40 →
  let total_weight := n * initial_avg
  let new_total_weight := total_weight + new_weight
  let new_avg := new_total_weight / (n + 1)
  initial_avg - new_avg = 2 := by
sorry

end average_weight_decrease_l1341_134190


namespace speedboat_drift_time_l1341_134139

/-- The time taken for a speedboat to drift along a river --/
theorem speedboat_drift_time 
  (L : ℝ) -- Total length of the river
  (v : ℝ) -- Speed of the speedboat in still water
  (u : ℝ) -- Speed of the water flow when reservoir is discharging
  (h1 : v = L / 150) -- Speed of boat in still water
  (h2 : v + u = L / 60) -- Speed of boat with water flow
  (h3 : u > 0) -- Water flow is positive
  : (L / 3) / u = 100 / 3 := by
  sorry

end speedboat_drift_time_l1341_134139


namespace client_ladder_cost_l1341_134122

/-- The total cost for a set of ladders given the number of ladders, rungs per ladder, and cost per rung -/
def total_cost (num_ladders : ℕ) (rungs_per_ladder : ℕ) (cost_per_rung : ℕ) : ℕ :=
  num_ladders * rungs_per_ladder * cost_per_rung

/-- The theorem stating the total cost for the client's ladder order -/
theorem client_ladder_cost :
  let cost_per_rung := 2
  let cost_first_set := total_cost 10 50 cost_per_rung
  let cost_second_set := total_cost 20 60 cost_per_rung
  cost_first_set + cost_second_set = 3400 := by sorry

end client_ladder_cost_l1341_134122


namespace inequality_solution_set_l1341_134163

theorem inequality_solution_set (x : ℝ) : 5 * x + 1 ≥ 3 * x - 5 ↔ x ≥ -3 := by
  sorry

end inequality_solution_set_l1341_134163


namespace final_selling_price_theorem_l1341_134141

/-- The final selling price of a batch of computers -/
def final_selling_price (a : ℝ) : ℝ :=
  a * (1 + 0.2) * (1 - 0.09)

/-- Theorem stating the final selling price calculation -/
theorem final_selling_price_theorem (a : ℝ) :
  final_selling_price a = a * (1 + 0.2) * (1 - 0.09) :=
by sorry

end final_selling_price_theorem_l1341_134141


namespace min_x_plus_y_l1341_134183

theorem min_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2 * y = x * y) :
  x + y ≥ 3 + 2 * Real.sqrt 2 := by
  sorry

end min_x_plus_y_l1341_134183


namespace students_not_enrolled_l1341_134175

theorem students_not_enrolled (total : ℕ) (french : ℕ) (german : ℕ) (both : ℕ) 
  (h1 : total = 94) 
  (h2 : french = 41) 
  (h3 : german = 22) 
  (h4 : both = 9) : 
  total - (french + german - both) = 40 := by
  sorry

end students_not_enrolled_l1341_134175


namespace distance_product_l1341_134164

theorem distance_product (b₁ b₂ : ℝ) : 
  (∀ b : ℝ, (3*b - 5)^2 + (b - 3)^2 = 39 → b = b₁ ∨ b = b₂) →
  (3*b₁ - 5)^2 + (b₁ - 3)^2 = 39 →
  (3*b₂ - 5)^2 + (b₂ - 3)^2 = 39 →
  b₁ * b₂ = -(9/16) := by
sorry

end distance_product_l1341_134164

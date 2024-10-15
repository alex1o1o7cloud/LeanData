import Mathlib

namespace NUMINAMATH_CALUDE_brick_width_calculation_l3451_345161

theorem brick_width_calculation (courtyard_length : ℝ) (courtyard_width : ℝ)
  (brick_length : ℝ) (total_bricks : ℕ) :
  courtyard_length = 18 →
  courtyard_width = 12 →
  brick_length = 0.12 →
  total_bricks = 30000 →
  ∃ (brick_width : ℝ),
    brick_width = 0.06 ∧
    courtyard_length * courtyard_width * 100 * 100 = total_bricks * brick_length * brick_width * 10000 :=
by sorry

end NUMINAMATH_CALUDE_brick_width_calculation_l3451_345161


namespace NUMINAMATH_CALUDE_jason_balloon_count_l3451_345180

/-- Calculates the final number of balloons Jason has after a series of changes. -/
def final_balloon_count (initial_violet : ℕ) (initial_red : ℕ) 
  (violet_given : ℕ) (red_given : ℕ) (violet_acquired : ℕ) : ℕ :=
  let remaining_violet := initial_violet - violet_given + violet_acquired
  let remaining_red := (initial_red - red_given) * 3
  remaining_violet + remaining_red

/-- Proves that Jason ends up with 35 balloons given the initial quantities and changes. -/
theorem jason_balloon_count : 
  final_balloon_count 15 12 3 5 2 = 35 := by
  sorry

end NUMINAMATH_CALUDE_jason_balloon_count_l3451_345180


namespace NUMINAMATH_CALUDE_number_of_basic_events_l3451_345103

/-- The number of ways to choose 2 items from a set of 3 items -/
def choose_two_from_three : ℕ := 3

/-- The set of interest groups -/
def interest_groups : Finset String := {"Mathematics", "Computer Science", "Model Aviation"}

/-- Xiao Ming must join exactly two groups -/
def join_two_groups (groups : Finset String) : Finset (Finset String) :=
  groups.powerset.filter (fun s => s.card = 2)

theorem number_of_basic_events :
  (join_two_groups interest_groups).card = choose_two_from_three := by sorry

end NUMINAMATH_CALUDE_number_of_basic_events_l3451_345103


namespace NUMINAMATH_CALUDE_max_word_ratio_bound_l3451_345191

/-- Represents a crossword on an n × n grid. -/
structure Crossword (n : ℕ) where
  cells : Set (Fin n × Fin n)
  nonempty : cells.Nonempty

/-- The number of words in a crossword. -/
def num_words (n : ℕ) (c : Crossword n) : ℕ := sorry

/-- The minimum number of words needed to cover a crossword. -/
def min_cover_words (n : ℕ) (c : Crossword n) : ℕ := sorry

/-- Theorem: The maximum ratio of words to minimum cover words is 1 + n/2 -/
theorem max_word_ratio_bound {n : ℕ} (hn : n ≥ 2) (c : Crossword n) :
  (num_words n c : ℚ) / (min_cover_words n c) ≤ 1 + n / 2 := by
  sorry

end NUMINAMATH_CALUDE_max_word_ratio_bound_l3451_345191


namespace NUMINAMATH_CALUDE_mike_speaker_cost_l3451_345165

/-- The amount Mike spent on speakers -/
def speaker_cost (total_cost new_tire_cost : ℚ) : ℚ :=
  total_cost - new_tire_cost

/-- Theorem: Mike spent $118.54 on speakers -/
theorem mike_speaker_cost : 
  speaker_cost 224.87 106.33 = 118.54 := by sorry

end NUMINAMATH_CALUDE_mike_speaker_cost_l3451_345165


namespace NUMINAMATH_CALUDE_point_on_line_value_l3451_345116

/-- A point lies on a line if it satisfies the line's equation -/
def PointOnLine (x₁ y₁ x₂ y₂ x y : ℝ) : Prop :=
  (y - y₁) * (x₂ - x₁) = (y₂ - y₁) * (x - x₁)

theorem point_on_line_value :
  ∀ x : ℝ, PointOnLine 1 4 4 1 x 8 → x = -3 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_value_l3451_345116


namespace NUMINAMATH_CALUDE_greatest_multiple_of_four_under_sqrt_400_l3451_345134

theorem greatest_multiple_of_four_under_sqrt_400 :
  ∀ x : ℕ, 
    x > 0 → 
    (∃ k : ℕ, x = 4 * k) → 
    x^2 < 400 → 
    x ≤ 16 ∧ 
    (∀ y : ℕ, y > 0 → (∃ m : ℕ, y = 4 * m) → y^2 < 400 → y ≤ x) :=
by sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_four_under_sqrt_400_l3451_345134


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3451_345141

theorem quadratic_inequality_solution_set (a : ℝ) (h : a > 0) :
  let solution_set := {x : ℝ | a * x^2 - (a + 2) * x + 2 ≥ 0}
  (a = 2 → solution_set = Set.univ) ∧
  (0 < a ∧ a < 2 → solution_set = Set.Iic 1 ∪ Set.Ici (2 / a)) ∧
  (a > 2 → solution_set = Set.Iic (2 / a) ∪ Set.Ici 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3451_345141


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3451_345120

-- Define set A
def A : Set ℝ := {x | ∃ y, y = Real.log (2 * x - x^2)}

-- Define set B
def B : Set ℝ := {y | ∃ x > 0, y = 2^x}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = Set.Ioo 1 2 := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3451_345120


namespace NUMINAMATH_CALUDE_f_properties_l3451_345195

open Real

noncomputable def f (x : ℝ) : ℝ := x * log x - x

theorem f_properties :
  let f := f
  (∀ x₁ x₂, 1 < x₁ ∧ x₁ < x₂ → f x₁ < f x₂) ∧
  (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 → f x₁ > f x₂) ∧
  (∀ x, x > 0 → f x ≥ -1) ∧
  f 1 = -1 :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l3451_345195


namespace NUMINAMATH_CALUDE_average_towel_price_l3451_345102

def towel_price_problem (price1 price2 price3 : ℕ) (quantity1 quantity2 quantity3 : ℕ) : Prop :=
  let total_cost := price1 * quantity1 + price2 * quantity2 + price3 * quantity3
  let total_quantity := quantity1 + quantity2 + quantity3
  (total_cost : ℚ) / total_quantity = 205

theorem average_towel_price :
  towel_price_problem 100 150 500 3 5 2 := by
  sorry

end NUMINAMATH_CALUDE_average_towel_price_l3451_345102


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l3451_345122

theorem expression_simplification_and_evaluation :
  let x : ℝ := Real.sqrt 3 - 2
  ((x - 1) / (x - 2) + (2 * x - 8) / (x^2 - 4)) / (x + 5) = Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l3451_345122


namespace NUMINAMATH_CALUDE_solve_equation_l3451_345108

theorem solve_equation (n : ℤ) : n + (n + 1) + (n + 2) + (n + 3) = 26 → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3451_345108


namespace NUMINAMATH_CALUDE_arithmetic_mean_fractions_l3451_345176

theorem arithmetic_mean_fractions (x a : ℝ) (hx : x ≠ 0) :
  (1 / 2) * ((x + 2 * a) / x + (x - 3 * a) / x) = 1 - a / (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_fractions_l3451_345176


namespace NUMINAMATH_CALUDE_trig_identity_l3451_345199

theorem trig_identity (α : Real) (h : Real.sin α + Real.cos α = 1/5) :
  (Real.sin α - Real.cos α)^2 = 49/25 ∧ Real.sin α^3 + Real.cos α^3 = 37/125 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l3451_345199


namespace NUMINAMATH_CALUDE_larger_number_proof_l3451_345173

theorem larger_number_proof (x y : ℝ) (sum_eq : x + y = 30) (diff_eq : x - y = 4) : 
  max x y = 17 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l3451_345173


namespace NUMINAMATH_CALUDE_square_sum_equals_36_l3451_345197

theorem square_sum_equals_36 (x y z w : ℝ) 
  (eq1 : x^2 / (2^2 - 1^2) + y^2 / (2^2 - 3^2) + z^2 / (2^2 - 5^2) + w^2 / (2^2 - 7^2) = 1)
  (eq2 : x^2 / (4^2 - 1^2) + y^2 / (4^2 - 3^2) + z^2 / (4^2 - 5^2) + w^2 / (4^2 - 7^2) = 1)
  (eq3 : x^2 / (6^2 - 1^2) + y^2 / (6^2 - 3^2) + z^2 / (6^2 - 5^2) + w^2 / (6^2 - 7^2) = 1)
  (eq4 : x^2 / (8^2 - 1^2) + y^2 / (8^2 - 3^2) + z^2 / (8^2 - 5^2) + w^2 / (8^2 - 7^2) = 1) :
  x^2 + y^2 + z^2 + w^2 = 36 := by
sorry


end NUMINAMATH_CALUDE_square_sum_equals_36_l3451_345197


namespace NUMINAMATH_CALUDE_high_school_math_club_payment_l3451_345189

theorem high_school_math_club_payment (B : ℕ) : 
  B < 10 → (∃ k : ℤ, 200 + 10 * B + 5 = 13 * k) → B = 1 :=
by sorry

end NUMINAMATH_CALUDE_high_school_math_club_payment_l3451_345189


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l3451_345125

theorem solve_exponential_equation :
  ∃ y : ℝ, (3 : ℝ) ^ (y + 3) = 81 ^ y ∧ y = 1 := by sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l3451_345125


namespace NUMINAMATH_CALUDE_fold_paper_sum_l3451_345121

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Checks if two points are symmetric about a given line -/
def areSymmetric (p1 p2 : Point) (l : Line) : Prop :=
  -- Definition of symmetry about a line
  sorry

/-- Finds the fold line given two pairs of symmetric points -/
def findFoldLine (p1 p2 p3 p4 : Point) : Line :=
  -- Definition to find the fold line
  sorry

/-- Main theorem -/
theorem fold_paper_sum (m n : ℝ) :
  let p1 : Point := ⟨0, 2⟩
  let p2 : Point := ⟨4, 0⟩
  let p3 : Point := ⟨9, 5⟩
  let p4 : Point := ⟨m, n⟩
  let foldLine := findFoldLine p1 p2 p3 p4
  areSymmetric p1 p2 foldLine ∧ areSymmetric p3 p4 foldLine →
  m + n = 10 := by
  sorry

end NUMINAMATH_CALUDE_fold_paper_sum_l3451_345121


namespace NUMINAMATH_CALUDE_cube_root_fraction_equivalence_l3451_345166

theorem cube_root_fraction_equivalence :
  let x : ℝ := 12.75
  let y : ℚ := 51 / 4
  x = y →
  (6 / x) ^ (1/3 : ℝ) = 2 / (17 ^ (1/3 : ℝ)) :=
by
  sorry

end NUMINAMATH_CALUDE_cube_root_fraction_equivalence_l3451_345166


namespace NUMINAMATH_CALUDE_suv_highway_mpg_l3451_345114

/-- The average miles per gallon (mpg) on the highway for an SUV -/
def highway_mpg : ℝ := 12.2

/-- The maximum distance in miles that the SUV can travel on 25 gallons of gasoline -/
def max_distance : ℝ := 305

/-- The amount of gasoline in gallons used to calculate the maximum distance -/
def gasoline_amount : ℝ := 25

theorem suv_highway_mpg :
  highway_mpg = max_distance / gasoline_amount :=
by sorry

end NUMINAMATH_CALUDE_suv_highway_mpg_l3451_345114


namespace NUMINAMATH_CALUDE_factor_implies_b_value_l3451_345196

theorem factor_implies_b_value (a b : ℤ) :
  (∃ c : ℤ, ∀ x : ℝ, (x^2 - 2*x - 1) * (c*x - 1) = a*x^3 + b*x^2 + 1) →
  b = -3 := by
  sorry

end NUMINAMATH_CALUDE_factor_implies_b_value_l3451_345196


namespace NUMINAMATH_CALUDE_distance_between_points_l3451_345175

/-- The distance between points A and B -/
def distance : ℝ := sorry

/-- The speed of the first pedestrian -/
def speed1 : ℝ := sorry

/-- The speed of the second pedestrian -/
def speed2 : ℝ := sorry

theorem distance_between_points (h1 : distance / (2 * speed1) = 15 / speed2)
                                (h2 : 24 / speed1 = distance / (2 * speed2))
                                (h3 : distance / speed1 = distance / speed2) :
  distance = 40 := by sorry

end NUMINAMATH_CALUDE_distance_between_points_l3451_345175


namespace NUMINAMATH_CALUDE_tangent_circles_radius_l3451_345130

/-- Two circles are externally tangent if the distance between their centers
    equals the sum of their radii -/
def externally_tangent (r1 r2 d : ℝ) : Prop := d = r1 + r2

theorem tangent_circles_radius (d r1 r2 : ℝ) :
  d = 8 → r1 = 3 → externally_tangent r1 r2 d → r2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_tangent_circles_radius_l3451_345130


namespace NUMINAMATH_CALUDE_cycling_speeds_l3451_345181

/-- Represents the cycling speeds of four people -/
structure CyclingGroup where
  henry_speed : ℝ
  liz_speed : ℝ
  jack_speed : ℝ
  tara_speed : ℝ

/-- The cycling group satisfies the given conditions -/
def satisfies_conditions (g : CyclingGroup) : Prop :=
  g.henry_speed = 5 ∧
  g.liz_speed = 3/4 * g.henry_speed ∧
  g.jack_speed = 6/5 * g.liz_speed ∧
  g.tara_speed = 9/8 * g.jack_speed

/-- Theorem stating the cycling speeds of Jack and Tara -/
theorem cycling_speeds (g : CyclingGroup) 
  (h : satisfies_conditions g) : 
  g.jack_speed = 4.5 ∧ g.tara_speed = 5.0625 := by
  sorry

#check cycling_speeds

end NUMINAMATH_CALUDE_cycling_speeds_l3451_345181


namespace NUMINAMATH_CALUDE_ants_meet_after_11_laps_l3451_345185

/-- The number of laps on the small circle before the ants meet again -/
def num_laps_to_meet (large_radius small_radius : ℕ) : ℕ :=
  Nat.lcm large_radius small_radius / small_radius

theorem ants_meet_after_11_laps :
  num_laps_to_meet 33 9 = 11 := by sorry

end NUMINAMATH_CALUDE_ants_meet_after_11_laps_l3451_345185


namespace NUMINAMATH_CALUDE_count_multiples_of_four_l3451_345118

theorem count_multiples_of_four : ∃ (n : ℕ), n = (Finset.filter (fun x => x % 4 = 0 ∧ x > 300 ∧ x < 700) (Finset.range 700)).card ∧ n = 99 := by
  sorry

end NUMINAMATH_CALUDE_count_multiples_of_four_l3451_345118


namespace NUMINAMATH_CALUDE_five_cubes_volume_l3451_345151

/-- The volume of a cube with edge length s -/
def cubeVolume (s : ℝ) : ℝ := s ^ 3

/-- The total volume of n cubes, each with edge length s -/
def totalVolume (n : ℕ) (s : ℝ) : ℝ := n * cubeVolume s

/-- Theorem: The total volume of five cubes with edge length 6 feet is 1080 cubic feet -/
theorem five_cubes_volume : totalVolume 5 6 = 1080 := by
  sorry

end NUMINAMATH_CALUDE_five_cubes_volume_l3451_345151


namespace NUMINAMATH_CALUDE_quadratic_one_solution_sum_l3451_345158

theorem quadratic_one_solution_sum (a : ℝ) : 
  let f : ℝ → ℝ := λ x => 9*x^2 + a*x + 12*x + 16
  let discriminant := (a + 12)^2 - 4*9*16
  (∃! x, f x = 0) → 
  (∃ a₁ a₂, discriminant = 0 ∧ a = a₁ ∨ a = a₂ ∧ a₁ + a₂ = -24) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_sum_l3451_345158


namespace NUMINAMATH_CALUDE_quadratic_point_m_value_l3451_345170

theorem quadratic_point_m_value (a m : ℝ) : 
  a > 0 → 
  m ≠ 0 → 
  3 = -a * m^2 + 2 * a * m + 3 → 
  m = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_point_m_value_l3451_345170


namespace NUMINAMATH_CALUDE_leftover_space_is_one_l3451_345100

-- Define the wall length
def wall_length : ℝ := 15

-- Define the desk length
def desk_length : ℝ := 2

-- Define the bookcase length
def bookcase_length : ℝ := 1.5

-- Define the function to calculate the space left over
def space_left_over (n : ℕ) : ℝ :=
  wall_length - (n * desk_length + n * bookcase_length)

-- Theorem statement
theorem leftover_space_is_one :
  ∃ n : ℕ, n > 0 ∧ 
    space_left_over n = 1 ∧
    ∀ m : ℕ, m > n → space_left_over m < 1 :=
  sorry

end NUMINAMATH_CALUDE_leftover_space_is_one_l3451_345100


namespace NUMINAMATH_CALUDE_solution_to_equation_l3451_345133

theorem solution_to_equation :
  ∃! (x y : ℝ), x ≠ 0 ∧ y ≠ 0 ∧ (5 * x)^10 = (10 * y)^5 - 25 * x ∧ x = 1/5 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_equation_l3451_345133


namespace NUMINAMATH_CALUDE_expression_evaluation_l3451_345143

theorem expression_evaluation (x y : ℝ) (h1 : x > y) (h2 : y > 0) :
  (x^(y+1) * y^(x-1)) / (y^y * x^x) = x^(y-x+1) * y^(x-y-1) := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3451_345143


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3451_345105

theorem min_value_reciprocal_sum (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hsum : x + y + z = 1) : 
  1/x + 4/y + 9/z ≥ 36 := by
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3451_345105


namespace NUMINAMATH_CALUDE_translated_function_eq_l3451_345115

-- Define the original function
def f (x : ℝ) : ℝ := 3 * x^2 - 6 * x - 1

-- Define the translated function
def g (x : ℝ) : ℝ := f (x + 1) + 3

-- Theorem stating that the translated function is equal to 3x^2 - 1
theorem translated_function_eq (x : ℝ) : g x = 3 * x^2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_translated_function_eq_l3451_345115


namespace NUMINAMATH_CALUDE_sequence_sum_equals_63_l3451_345111

theorem sequence_sum_equals_63 : 
  (Finset.range 9).sum (fun i => (i + 4) * (1 - 1 / (i + 2))) = 63 := by sorry

end NUMINAMATH_CALUDE_sequence_sum_equals_63_l3451_345111


namespace NUMINAMATH_CALUDE_album_difference_l3451_345109

/-- Represents the number of albums each person has -/
structure AlbumCounts where
  adele : ℕ
  bridget : ℕ
  katrina : ℕ
  miriam : ℕ

/-- The conditions of the problem -/
def problem_conditions (counts : AlbumCounts) : Prop :=
  counts.miriam = 5 * counts.katrina ∧
  counts.katrina = 6 * counts.bridget ∧
  counts.bridget < counts.adele ∧
  counts.adele + counts.bridget + counts.katrina + counts.miriam = 585 ∧
  counts.adele = 30

/-- The theorem to be proved -/
theorem album_difference (counts : AlbumCounts) 
  (h : problem_conditions counts) : 
  counts.adele - counts.bridget = 15 := by
  sorry

end NUMINAMATH_CALUDE_album_difference_l3451_345109


namespace NUMINAMATH_CALUDE_smallest_irrational_distance_points_theorem_l3451_345157

/-- The smallest number of points in ℝⁿ such that every point of ℝⁿ is an irrational distance from at least one of the points -/
def smallest_irrational_distance_points (n : ℕ) : ℕ :=
  if n = 1 then 2 else 3

/-- Theorem stating the smallest number of points in ℝⁿ such that every point of ℝⁿ is an irrational distance from at least one of the points -/
theorem smallest_irrational_distance_points_theorem (n : ℕ) (hn : n > 0) :
  smallest_irrational_distance_points n = if n = 1 then 2 else 3 :=
sorry

end NUMINAMATH_CALUDE_smallest_irrational_distance_points_theorem_l3451_345157


namespace NUMINAMATH_CALUDE_gem_purchase_theorem_l3451_345142

/-- Proves that given the conditions of gem purchasing and bonuses, 
    the amount spent to obtain 30,000 gems is $250. -/
theorem gem_purchase_theorem (gems_per_dollar : ℕ) (bonus_rate : ℚ) (final_gems : ℕ) : 
  gems_per_dollar = 100 →
  bonus_rate = 1/5 →
  final_gems = 30000 →
  (final_gems : ℚ) / (gems_per_dollar : ℚ) / (1 + bonus_rate) = 250 := by
  sorry

end NUMINAMATH_CALUDE_gem_purchase_theorem_l3451_345142


namespace NUMINAMATH_CALUDE_union_of_S_and_T_l3451_345169

def S : Set ℕ := {1, 3, 5}
def T : Set ℕ := {3, 6}

theorem union_of_S_and_T : S ∪ T = {1, 3, 5, 6} := by sorry

end NUMINAMATH_CALUDE_union_of_S_and_T_l3451_345169


namespace NUMINAMATH_CALUDE_green_blue_difference_l3451_345152

/-- Represents the colors of disks in the bag -/
inductive DiskColor
  | Blue
  | Yellow
  | Green

/-- Represents the bag of disks -/
structure DiskBag where
  total : ℕ
  blue : ℕ
  yellow : ℕ
  green : ℕ
  color_sum : blue + yellow + green = total
  ratio : blue * 18 = total * 3 ∧ yellow * 18 = total * 7 ∧ green * 18 = total * 8

theorem green_blue_difference (bag : DiskBag) (h : bag.total = 144) :
  bag.green - bag.blue = 40 := by
  sorry

end NUMINAMATH_CALUDE_green_blue_difference_l3451_345152


namespace NUMINAMATH_CALUDE_restaurant_peppers_total_weight_l3451_345112

theorem restaurant_peppers_total_weight 
  (green_peppers : ℝ) 
  (red_peppers : ℝ) 
  (h1 : green_peppers = 0.3333333333333333) 
  (h2 : red_peppers = 0.3333333333333333) : 
  green_peppers + red_peppers = 0.6666666666666666 := by
sorry

end NUMINAMATH_CALUDE_restaurant_peppers_total_weight_l3451_345112


namespace NUMINAMATH_CALUDE_geometric_sequence_n_l3451_345163

theorem geometric_sequence_n (a₁ q aₙ : ℚ) (n : ℕ) : 
  a₁ = 1/2 → q = 1/2 → aₙ = 1/32 → aₙ = a₁ * q^(n-1) → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_n_l3451_345163


namespace NUMINAMATH_CALUDE_valid_string_count_l3451_345179

/-- A run is a set of consecutive identical letters in a string. -/
def Run := ℕ

/-- A valid string is a 10-letter string composed of A's and B's where no more than 3 consecutive letters are the same. -/
def ValidString := Fin 10 → Bool

/-- The number of runs in a valid string is between 4 and 10, inclusive. -/
def ValidRunCount (n : ℕ) : Prop := 4 ≤ n ∧ n ≤ 10

/-- The generating function for a single run is x + x^2 + x^3. -/
def SingleRunGeneratingFunction (x : ℝ) : ℝ := x + x^2 + x^3

/-- The coefficient of x^(10-n) in the expansion of ((1-x^3)^n) / ((1-x)^n). -/
def Coefficient (n : ℕ) : ℕ := sorry

/-- The total number of valid strings. -/
def TotalValidStrings : ℕ := 2 * (Coefficient 4 + Coefficient 5 + Coefficient 6 + Coefficient 7 + Coefficient 8 + Coefficient 9 + Coefficient 10)

theorem valid_string_count : TotalValidStrings = 548 := by sorry

end NUMINAMATH_CALUDE_valid_string_count_l3451_345179


namespace NUMINAMATH_CALUDE_square_between_endpoints_l3451_345194

theorem square_between_endpoints (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (h_cd : c * d = 1) :
  Real.sqrt ((a + c) * (b + d)) ≥ Real.sqrt (a * b) + 1 := by
  sorry

end NUMINAMATH_CALUDE_square_between_endpoints_l3451_345194


namespace NUMINAMATH_CALUDE_minimal_sum_roots_and_qtilde_value_l3451_345124

/-- Represents a quadratic polynomial q(x) = x^2 - (a+b)x + ab -/
def QuadPoly (a b : ℝ) (x : ℝ) : ℝ :=
  x^2 - (a + b) * x + a * b

/-- The condition that q(q(x)) = 0 has exactly three real solutions -/
def HasThreeSolutions (a b : ℝ) : Prop :=
  ∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    QuadPoly a b (QuadPoly a b x) = 0 ∧
    QuadPoly a b (QuadPoly a b y) = 0 ∧
    QuadPoly a b (QuadPoly a b z) = 0 ∧
    ∀ w : ℝ, QuadPoly a b (QuadPoly a b w) = 0 → w = x ∨ w = y ∨ w = z

/-- The sum of roots of q(x) = 0 -/
def SumOfRoots (a b : ℝ) : ℝ := a + b

/-- The polynomial ̃q(x) = x^2 + 2x + 1 -/
def QTilde (x : ℝ) : ℝ := x^2 + 2 * x + 1

theorem minimal_sum_roots_and_qtilde_value :
  ∀ a b : ℝ,
  HasThreeSolutions a b →
  (∀ c d : ℝ, HasThreeSolutions c d → SumOfRoots a b ≤ SumOfRoots c d) →
  (QuadPoly a b = QTilde) ∧ QTilde 2 = 9 := by sorry

end NUMINAMATH_CALUDE_minimal_sum_roots_and_qtilde_value_l3451_345124


namespace NUMINAMATH_CALUDE_orange_distribution_l3451_345106

theorem orange_distribution (oranges_per_child : ℕ) (total_oranges : ℕ) (num_children : ℕ) : 
  oranges_per_child = 3 → 
  total_oranges = 12 → 
  num_children * oranges_per_child = total_oranges →
  num_children = 4 := by
sorry

end NUMINAMATH_CALUDE_orange_distribution_l3451_345106


namespace NUMINAMATH_CALUDE_quadratic_point_ordering_l3451_345128

/-- A quadratic function f(x) = (x+1)² + 1 -/
def f (x : ℝ) : ℝ := (x + 1)^2 + 1

/-- Point A on the graph of f -/
def A : ℝ × ℝ := (-3, f (-3))

/-- Point B on the graph of f -/
def B : ℝ × ℝ := (0, f 0)

/-- Point C on the graph of f -/
def C : ℝ × ℝ := (2, f 2)

theorem quadratic_point_ordering :
  B.2 < A.2 ∧ A.2 < C.2 := by sorry

end NUMINAMATH_CALUDE_quadratic_point_ordering_l3451_345128


namespace NUMINAMATH_CALUDE_min_distance_between_curves_l3451_345119

/-- The minimum distance between a point on y = (1/2)e^x and a point on y = ln(2x) -/
theorem min_distance_between_curves : ∃ (min_dist : ℝ),
  min_dist = Real.sqrt 2 * (1 - Real.log 2) ∧
  ∀ (x₁ x₂ : ℝ),
    let p := (x₁, (1/2) * Real.exp x₁)
    let q := (x₂, Real.log (2 * x₂))
    Real.sqrt ((x₁ - x₂)^2 + ((1/2) * Real.exp x₁ - Real.log (2 * x₂))^2) ≥ min_dist :=
by sorry

end NUMINAMATH_CALUDE_min_distance_between_curves_l3451_345119


namespace NUMINAMATH_CALUDE_trivia_team_score_l3451_345182

/-- Represents a trivia team with their scores -/
structure TriviaTeam where
  totalMembers : Nat
  absentMembers : Nat
  scores : List Nat

/-- Calculates the total score of a trivia team -/
def totalScore (team : TriviaTeam) : Nat :=
  team.scores.sum

/-- Theorem: The trivia team's total score is 26 points -/
theorem trivia_team_score : 
  ∀ (team : TriviaTeam), 
    team.totalMembers = 8 → 
    team.absentMembers = 3 → 
    team.scores = [4, 6, 8, 8] → 
    totalScore team = 26 := by
  sorry

end NUMINAMATH_CALUDE_trivia_team_score_l3451_345182


namespace NUMINAMATH_CALUDE_min_value_complex_expression_l3451_345188

/-- Given a complex number z where |z - 3 + 2i| = 3, 
    the minimum value of |z + 1 - i|^2 + |z - 7 + 3i|^2 is 86. -/
theorem min_value_complex_expression (z : ℂ) 
  (h : Complex.abs (z - (3 - 2*Complex.I)) = 3) : 
  (Complex.abs (z + (1 - Complex.I)))^2 + (Complex.abs (z - (7 - 3*Complex.I)))^2 ≥ 86 ∧ 
  ∃ w : ℂ, Complex.abs (w - (3 - 2*Complex.I)) = 3 ∧ 
    (Complex.abs (w + (1 - Complex.I)))^2 + (Complex.abs (w - (7 - 3*Complex.I)))^2 = 86 :=
by sorry

end NUMINAMATH_CALUDE_min_value_complex_expression_l3451_345188


namespace NUMINAMATH_CALUDE_alternating_sequence_sum_l3451_345162

def alternating_sequence (first last step : ℕ) : List ℤ :=
  let n := (first - last) / step + 1
  List.range n |> List.map (λ i => first - i * step) |> List.map (λ x => if x % (2 * step) = 0 then x else -x)

theorem alternating_sequence_sum (first last step : ℕ) :
  first > last ∧ step > 0 ∧ (first - last) % step = 0 →
  List.sum (alternating_sequence first last step) = 520 :=
by
  sorry

#eval List.sum (alternating_sequence 1050 20 20)

end NUMINAMATH_CALUDE_alternating_sequence_sum_l3451_345162


namespace NUMINAMATH_CALUDE_candy_box_original_price_l3451_345139

/-- Given a candy box with an original price, which after a 25% increase becomes 10 pounds,
    prove that the original price was 8 pounds. -/
theorem candy_box_original_price (original_price : ℝ) : 
  (original_price * 1.25 = 10) → original_price = 8 := by
  sorry

end NUMINAMATH_CALUDE_candy_box_original_price_l3451_345139


namespace NUMINAMATH_CALUDE_factorization_problem_1_l3451_345132

theorem factorization_problem_1 (a x : ℝ) : 3*a*x^2 - 6*a*x + 3*a = 3*a*(x-1)^2 := by sorry

end NUMINAMATH_CALUDE_factorization_problem_1_l3451_345132


namespace NUMINAMATH_CALUDE_thirty_day_month_equal_tuesdays_thursdays_l3451_345136

/-- Represents the days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Counts the number of occurrences of a specific day in a 30-day month starting from a given day -/
def countDayInMonth (startDay : DayOfWeek) (dayToCount : DayOfWeek) : Nat :=
  sorry

/-- Checks if a 30-day month starting from a given day has equal Tuesdays and Thursdays -/
def hasEqualTuesdaysThursdays (startDay : DayOfWeek) : Bool :=
  countDayInMonth startDay DayOfWeek.Tuesday = countDayInMonth startDay DayOfWeek.Thursday

/-- Counts the number of possible start days for a 30-day month with equal Tuesdays and Thursdays -/
def countValidStartDays : Nat :=
  sorry

theorem thirty_day_month_equal_tuesdays_thursdays :
  countValidStartDays = 4 :=
sorry

end NUMINAMATH_CALUDE_thirty_day_month_equal_tuesdays_thursdays_l3451_345136


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3451_345187

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 - a*x + a ≥ 0) → a ∈ Set.Icc 0 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3451_345187


namespace NUMINAMATH_CALUDE_min_value_of_z3_l3451_345123

open Complex

theorem min_value_of_z3 (z₁ z₂ z₃ : ℂ) 
  (h1 : ∃ (a : ℝ), z₁ / z₂ = Complex.I * a)
  (h2 : abs z₁ = 1)
  (h3 : abs z₂ = 1)
  (h4 : abs (z₁ + z₂ + z₃) = 1) :
  abs z₃ ≥ Real.sqrt 2 - 1 := by
sorry

end NUMINAMATH_CALUDE_min_value_of_z3_l3451_345123


namespace NUMINAMATH_CALUDE_paint_usage_correct_l3451_345154

/-- Represents the amount of paint used for a canvas size -/
structure PaintUsage where
  red : ℕ
  blue : ℕ
  yellow : ℕ
  green : ℕ

/-- Calculates the total paint used for a given canvas size and count -/
def totalPaintUsed (usage : PaintUsage) (count : ℕ) : PaintUsage :=
  { red := usage.red * count
  , blue := usage.blue * count
  , yellow := usage.yellow * count
  , green := usage.green * count
  }

/-- Adds two PaintUsage structures -/
def addPaintUsage (a b : PaintUsage) : PaintUsage :=
  { red := a.red + b.red
  , blue := a.blue + b.blue
  , yellow := a.yellow + b.yellow
  , green := a.green + b.green
  }

theorem paint_usage_correct : 
  let extraLarge : PaintUsage := { red := 5, blue := 3, yellow := 2, green := 1 }
  let large : PaintUsage := { red := 4, blue := 2, yellow := 3, green := 1 }
  let medium : PaintUsage := { red := 3, blue := 1, yellow := 2, green := 1 }
  let small : PaintUsage := { red := 1, blue := 1, yellow := 1, green := 1 }
  
  let totalUsage := addPaintUsage
    (addPaintUsage
      (addPaintUsage
        (totalPaintUsed extraLarge 3)
        (totalPaintUsed large 5))
      (totalPaintUsed medium 6))
    (totalPaintUsed small 8)

  totalUsage.red = 61 ∧
  totalUsage.blue = 33 ∧
  totalUsage.yellow = 41 ∧
  totalUsage.green = 22 :=
by sorry


end NUMINAMATH_CALUDE_paint_usage_correct_l3451_345154


namespace NUMINAMATH_CALUDE_solution_set_of_equation_l3451_345198

def is_solution (x : ℝ) : Prop :=
  -2*x > 0 ∧ 3 - x^2 > 0 ∧ -2*x = 3 - x^2

theorem solution_set_of_equation : 
  {x : ℝ | is_solution x} = {-1} := by sorry

end NUMINAMATH_CALUDE_solution_set_of_equation_l3451_345198


namespace NUMINAMATH_CALUDE_sum_of_squares_l3451_345149

theorem sum_of_squares (m n : ℝ) (h1 : m + n = 7) (h2 : m * n = 3) : m^2 + n^2 = 43 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l3451_345149


namespace NUMINAMATH_CALUDE_namjoon_has_14_pencils_l3451_345107

/-- Represents the number of pencils in a dozen -/
def dozen : ℕ := 12

/-- Represents the number of dozens Taehyung bought -/
def bought_dozens : ℕ := 2

/-- Represents the total number of pencils Taehyung bought -/
def total_pencils : ℕ := bought_dozens * dozen

/-- Represents the number of pencils Taehyung has -/
def taehyung_pencils : ℕ := total_pencils / 2

/-- Represents the number of pencils Namjoon has -/
def namjoon_pencils : ℕ := taehyung_pencils + 4

theorem namjoon_has_14_pencils : namjoon_pencils = 14 := by
  sorry

end NUMINAMATH_CALUDE_namjoon_has_14_pencils_l3451_345107


namespace NUMINAMATH_CALUDE_largest_positive_integer_solution_l3451_345177

theorem largest_positive_integer_solution :
  ∀ x : ℕ+, 2 * (x + 1) ≥ 5 * x - 3 ↔ x ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_largest_positive_integer_solution_l3451_345177


namespace NUMINAMATH_CALUDE_only_36_is_perfect_square_l3451_345150

theorem only_36_is_perfect_square : 
  (∃ n : ℤ, n * n = 36) ∧ 
  (∀ m : ℤ, m * m ≠ 32) ∧ 
  (∀ m : ℤ, m * m ≠ 33) ∧ 
  (∀ m : ℤ, m * m ≠ 34) ∧ 
  (∀ m : ℤ, m * m ≠ 35) :=
by sorry

end NUMINAMATH_CALUDE_only_36_is_perfect_square_l3451_345150


namespace NUMINAMATH_CALUDE_fiftieth_term_of_sequence_l3451_345156

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

theorem fiftieth_term_of_sequence :
  let a₁ := 3
  let d := 6
  let n := 50
  arithmetic_sequence a₁ d n = 297 := by sorry

end NUMINAMATH_CALUDE_fiftieth_term_of_sequence_l3451_345156


namespace NUMINAMATH_CALUDE_degree_of_g_l3451_345153

-- Define the polynomial f(x)
def f (x : ℝ) : ℝ := -9 * x^4 + 2 * x^3 - 7 * x + 8

-- State the theorem
theorem degree_of_g (g : ℝ → ℝ) :
  (∃ (a b : ℝ), ∀ x, f x + g x = a * x + b) →  -- degree of f(x) + g(x) is 1
  (∃ (a b c d e : ℝ), a ≠ 0 ∧ ∀ x, g x = a * x^4 + b * x^3 + c * x^2 + d * x + e) :=  -- g(x) is a polynomial of degree 4
by sorry

end NUMINAMATH_CALUDE_degree_of_g_l3451_345153


namespace NUMINAMATH_CALUDE_exists_irrational_between_3_and_4_l3451_345148

theorem exists_irrational_between_3_and_4 : ∃ x : ℝ, Irrational x ∧ 3 < x ∧ x < 4 := by
  sorry

end NUMINAMATH_CALUDE_exists_irrational_between_3_and_4_l3451_345148


namespace NUMINAMATH_CALUDE_bills_milk_problem_l3451_345190

/-- Represents the problem of determining the amount of milk Bill got from his cow --/
theorem bills_milk_problem (M : ℝ) : 
  M > 0 ∧ 
  (M / 16) * 5 + (M / 8) * 6 + (M / 2) * 3 = 41 → 
  M = 16 :=
by sorry

end NUMINAMATH_CALUDE_bills_milk_problem_l3451_345190


namespace NUMINAMATH_CALUDE_checkout_speed_ratio_l3451_345192

/-- Represents the problem of determining the ratio of cashier checkout speed to the rate of increase in waiting people. -/
theorem checkout_speed_ratio
  (n : ℕ)  -- Initial number of people in line
  (y : ℝ)  -- Rate at which number of people waiting increases (people per minute)
  (x : ℝ)  -- Cashier's checkout speed (people per minute)
  (h1 : 20 * 2 * x = 20 * y + n)  -- Equation for 2 counters open for 20 minutes
  (h2 : 12 * 3 * x = 12 * y + n)  -- Equation for 3 counters open for 12 minutes
  : x = 2 * y :=
sorry

end NUMINAMATH_CALUDE_checkout_speed_ratio_l3451_345192


namespace NUMINAMATH_CALUDE_second_amount_equals_600_l3451_345146

/-- Calculate simple interest -/
def simple_interest (principal rate time : ℝ) : ℝ := principal * rate * time

/-- The problem statement -/
theorem second_amount_equals_600 :
  ∃ (P : ℝ),
    simple_interest 100 0.05 48 = simple_interest P 0.10 4 ∧
    P = 600 := by
  sorry

end NUMINAMATH_CALUDE_second_amount_equals_600_l3451_345146


namespace NUMINAMATH_CALUDE_distance_between_points_l3451_345140

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (1, 3)
  let p2 : ℝ × ℝ := (6, 7)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = Real.sqrt 41 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l3451_345140


namespace NUMINAMATH_CALUDE_triangle_excircle_relation_l3451_345144

/-- Given a triangle ABC with sides a, b, c and excircle radii r_a, r_b, r_c opposite to vertices A, B, C respectively, 
    the sum of the squares of each side divided by the product of its opposite excircle radius and the sum of the other two radii equals 2. -/
theorem triangle_excircle_relation (a b c r_a r_b r_c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ r_a > 0 ∧ r_b > 0 ∧ r_c > 0) :
  a^2 / (r_a * (r_b + r_c)) + b^2 / (r_b * (r_c + r_a)) + c^2 / (r_c * (r_a + r_b)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_excircle_relation_l3451_345144


namespace NUMINAMATH_CALUDE_prime_arithmetic_seq_large_diff_l3451_345147

/-- A sequence of 15 different positive prime numbers in arithmetic progression -/
structure PrimeArithmeticSequence where
  terms : Fin 15 → ℕ
  is_prime : ∀ i, Nat.Prime (terms i)
  is_arithmetic : ∀ i j k, i.val + k.val = j.val → terms i + terms k = 2 * terms j
  is_distinct : ∀ i j, i ≠ j → terms i ≠ terms j

/-- The common difference of an arithmetic sequence -/
def common_difference (seq : PrimeArithmeticSequence) : ℕ :=
  seq.terms 1 - seq.terms 0

/-- Theorem: The common difference of a sequence of 15 different positive primes
    in arithmetic progression is greater than 30000 -/
theorem prime_arithmetic_seq_large_diff (seq : PrimeArithmeticSequence) :
  common_difference seq > 30000 := by
  sorry

end NUMINAMATH_CALUDE_prime_arithmetic_seq_large_diff_l3451_345147


namespace NUMINAMATH_CALUDE_four_solutions_l3451_345193

-- Define the piecewise function f(x)
noncomputable def f (k : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then Real.log x else k * x + 2

-- Theorem statement
theorem four_solutions (k : ℝ) (h : k > 0) :
  ∃! (s : Finset ℝ), s.card = 4 ∧ ∀ x ∈ s, |f k x| = 1 :=
sorry

end NUMINAMATH_CALUDE_four_solutions_l3451_345193


namespace NUMINAMATH_CALUDE_ram_pairs_sold_l3451_345138

/-- Represents the sales and earnings of a hardware store for a week. -/
structure StoreSales where
  graphics_cards : Nat
  hard_drives : Nat
  cpus : Nat
  ram_pairs : Nat
  graphics_card_price : Nat
  hard_drive_price : Nat
  cpu_price : Nat
  ram_pair_price : Nat
  total_earnings : Nat

/-- Calculates the total earnings from a given StoreSales. -/
def calculate_earnings (sales : StoreSales) : Nat :=
  sales.graphics_cards * sales.graphics_card_price +
  sales.hard_drives * sales.hard_drive_price +
  sales.cpus * sales.cpu_price +
  sales.ram_pairs * sales.ram_pair_price

/-- Theorem stating that the number of RAM pairs sold is 4. -/
theorem ram_pairs_sold (sales : StoreSales) :
  sales.graphics_cards = 10 →
  sales.hard_drives = 14 →
  sales.cpus = 8 →
  sales.graphics_card_price = 600 →
  sales.hard_drive_price = 80 →
  sales.cpu_price = 200 →
  sales.ram_pair_price = 60 →
  sales.total_earnings = 8960 →
  calculate_earnings sales = sales.total_earnings →
  sales.ram_pairs = 4 := by
  sorry


end NUMINAMATH_CALUDE_ram_pairs_sold_l3451_345138


namespace NUMINAMATH_CALUDE_jen_triple_flips_l3451_345126

/-- Represents the number of flips in a specific type of flip. -/
def flips_per_type (flip_type : String) : ℕ :=
  if flip_type = "double" then 2 else 3

/-- Represents the total number of flips performed by a gymnast. -/
def total_flips (completed_flips : ℕ) (flip_type : String) : ℕ :=
  completed_flips * flips_per_type flip_type

theorem jen_triple_flips (tyler_double_flips : ℕ) (h1 : tyler_double_flips = 12) :
  let tyler_total_flips := total_flips tyler_double_flips "double"
  let jen_total_flips := 2 * tyler_total_flips
  jen_total_flips / flips_per_type "triple" = 16 := by
  sorry

end NUMINAMATH_CALUDE_jen_triple_flips_l3451_345126


namespace NUMINAMATH_CALUDE_same_type_quadratic_root_l3451_345145

theorem same_type_quadratic_root (a : ℝ) : 
  (∃ (k : ℝ), k^2 = 12 ∧ k^2 = 2*a - 5) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_same_type_quadratic_root_l3451_345145


namespace NUMINAMATH_CALUDE_perfect_square_condition_l3451_345117

theorem perfect_square_condition (x y : ℕ) :
  ∃ (n : ℕ), (x + y)^2 + 3*x + y + 1 = n^2 ↔ x = y :=
sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l3451_345117


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l3451_345168

theorem sum_of_three_numbers (x y z : ℝ) 
  (sum_xy : x + y = 29)
  (sum_yz : y + z = 46)
  (sum_zx : z + x = 53) :
  x + y + z = 64 := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l3451_345168


namespace NUMINAMATH_CALUDE_inequality_preservation_l3451_345178

theorem inequality_preservation (a b : ℝ) (h : a < b) : -2 + 2*a < -2 + 2*b := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l3451_345178


namespace NUMINAMATH_CALUDE_video_game_lives_l3451_345174

theorem video_game_lives (initial_lives lost_lives gained_lives : ℕ) 
  (h1 : initial_lives = 43)
  (h2 : lost_lives = 14)
  (h3 : gained_lives = 27) :
  initial_lives - lost_lives + gained_lives = 56 :=
by sorry

end NUMINAMATH_CALUDE_video_game_lives_l3451_345174


namespace NUMINAMATH_CALUDE_units_digit_of_result_l3451_345164

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The result of the calculation -/
def result : ℕ := 7 * 18 * 1978 - 7^4

theorem units_digit_of_result : unitsDigit result = 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_result_l3451_345164


namespace NUMINAMATH_CALUDE_john_initial_plays_l3451_345131

/-- The number of acts in each play -/
def acts_per_play : ℕ := 5

/-- The number of wigs John wears per act -/
def wigs_per_act : ℕ := 2

/-- The cost of each wig in dollars -/
def cost_per_wig : ℕ := 5

/-- The selling price of each wig from the dropped play in dollars -/
def selling_price_per_wig : ℕ := 4

/-- The total amount John spent in dollars -/
def total_spent : ℕ := 110

/-- The number of plays John was initially performing in -/
def initial_plays : ℕ := 3

theorem john_initial_plays :
  initial_plays * (acts_per_play * wigs_per_act * cost_per_wig) -
  (acts_per_play * wigs_per_act * selling_price_per_wig) = total_spent :=
by sorry

end NUMINAMATH_CALUDE_john_initial_plays_l3451_345131


namespace NUMINAMATH_CALUDE_sum_coordinates_of_D_l3451_345159

/-- Given that N(6,2) is the midpoint of line segment CD and C(10,-2), 
    prove that the sum of coordinates of D is 8 -/
theorem sum_coordinates_of_D (N C D : ℝ × ℝ) : 
  N = (6, 2) → 
  C = (10, -2) → 
  N = ((C.1 + D.1) / 2, (C.2 + D.2) / 2) → 
  D.1 + D.2 = 8 := by
sorry

end NUMINAMATH_CALUDE_sum_coordinates_of_D_l3451_345159


namespace NUMINAMATH_CALUDE_expression_evaluation_l3451_345104

theorem expression_evaluation :
  let x : ℝ := -2
  let y : ℝ := 1
  let z : ℝ := 1
  let w : ℝ := 3
  (x^2 * y^2 * z^2) - (x^2 * y * z^2) + (y / w) * Real.sin (x * z) = -(1/3) * Real.sin 2 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3451_345104


namespace NUMINAMATH_CALUDE_binary_arithmetic_l3451_345129

-- Define binary numbers as natural numbers
def bin_10110 : ℕ := 22  -- 10110 in binary is 22 in decimal
def bin_1011 : ℕ := 11   -- 1011 in binary is 11 in decimal
def bin_11100 : ℕ := 28  -- 11100 in binary is 28 in decimal
def bin_11101 : ℕ := 29  -- 11101 in binary is 29 in decimal
def bin_100010 : ℕ := 34 -- 100010 in binary is 34 in decimal

-- Define a function to convert a natural number to its binary representation
def to_binary (n : ℕ) : List ℕ := sorry

-- Theorem statement
theorem binary_arithmetic :
  to_binary (bin_10110 + bin_1011 - bin_11100 + bin_11101) = to_binary bin_100010 :=
sorry

end NUMINAMATH_CALUDE_binary_arithmetic_l3451_345129


namespace NUMINAMATH_CALUDE_card_distribution_l3451_345172

theorem card_distribution (total_cards : Nat) (num_players : Nat) 
  (h1 : total_cards = 57) (h2 : num_players = 4) :
  ∃ (cards_per_player : Nat) (unassigned_cards : Nat),
    cards_per_player * num_players + unassigned_cards = total_cards ∧
    cards_per_player = 14 ∧
    unassigned_cards = 1 := by
  sorry

end NUMINAMATH_CALUDE_card_distribution_l3451_345172


namespace NUMINAMATH_CALUDE_line_perpendicular_to_plane_l3451_345110

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Line → Prop)
variable (perpendicularToPlane : Line → Plane → Prop)
variable (intersect : Line → Line → Prop)

-- State the theorem
theorem line_perpendicular_to_plane 
  (l m n : Line) (α : Plane) 
  (h1 : intersect l m)
  (h2 : parallel l α)
  (h3 : parallel m α)
  (h4 : perpendicular n l)
  (h5 : perpendicular n m) :
  perpendicularToPlane n α :=
sorry

end NUMINAMATH_CALUDE_line_perpendicular_to_plane_l3451_345110


namespace NUMINAMATH_CALUDE_complex_fraction_real_implies_zero_l3451_345155

theorem complex_fraction_real_implies_zero (a : ℝ) : 
  (Complex.I : ℂ) * (Complex.I : ℂ) = -1 →
  (((a : ℂ) + 2 * Complex.I) / ((a : ℂ) - 2 * Complex.I)).im = 0 →
  a = 0 := by sorry

end NUMINAMATH_CALUDE_complex_fraction_real_implies_zero_l3451_345155


namespace NUMINAMATH_CALUDE_square_plaza_area_l3451_345167

/-- The area of a square plaza with side length 5 × 10^2 m is 2.5 × 10^5 m^2. -/
theorem square_plaza_area :
  let side_length : ℝ := 5 * 10^2
  let area : ℝ := side_length^2
  area = 2.5 * 10^5 := by sorry

end NUMINAMATH_CALUDE_square_plaza_area_l3451_345167


namespace NUMINAMATH_CALUDE_alla_boris_meeting_l3451_345135

/-- Represents the meeting point of Alla and Boris along a straight alley with lampposts -/
def meeting_point (total_lampposts : ℕ) (alla_position : ℕ) (boris_position : ℕ) : ℕ :=
  alla_position + (total_lampposts - alla_position - boris_position + 1) / 2

/-- Theorem stating that Alla and Boris meet at lamppost 163 under the given conditions -/
theorem alla_boris_meeting :
  let total_lampposts : ℕ := 400
  let alla_start : ℕ := 1
  let boris_start : ℕ := total_lampposts
  let alla_position : ℕ := 55
  let boris_position : ℕ := 321
  meeting_point total_lampposts alla_position boris_position = 163 :=
by sorry

end NUMINAMATH_CALUDE_alla_boris_meeting_l3451_345135


namespace NUMINAMATH_CALUDE_unique_solution_characterization_l3451_345113

/-- The set of real numbers a for which the system has a unique solution -/
def UniqueSystemSolutionSet : Set ℝ :=
  {a | a < -5 ∨ a > -1}

/-- The system of equations -/
def SystemEquations (x y a : ℝ) : Prop :=
  x = 4 * Real.sqrt y + a ∧ y^2 - x^2 + 3*y - 5*x - 4 = 0

theorem unique_solution_characterization (a : ℝ) :
  (∃! p : ℝ × ℝ, SystemEquations p.1 p.2 a) ↔ a ∈ UniqueSystemSolutionSet :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_characterization_l3451_345113


namespace NUMINAMATH_CALUDE_sergeant_travel_distance_l3451_345171

/-- Proves that given an infantry column of length 1 km, if the infantry walks 4/3 km
    during the time it takes for someone to travel from the end to the beginning of
    the column and back at twice the speed of the infantry, then the total distance
    traveled by that person is 8/3 km. -/
theorem sergeant_travel_distance
  (column_length : ℝ)
  (infantry_distance : ℝ)
  (sergeant_speed_ratio : ℝ)
  (h1 : column_length = 1)
  (h2 : infantry_distance = 4/3)
  (h3 : sergeant_speed_ratio = 2) :
  2 * infantry_distance = 8/3 := by
  sorry

#check sergeant_travel_distance

end NUMINAMATH_CALUDE_sergeant_travel_distance_l3451_345171


namespace NUMINAMATH_CALUDE_pictures_in_first_album_l3451_345101

theorem pictures_in_first_album (total_pictures : ℕ) (albums : ℕ) (pictures_per_album : ℕ) :
  total_pictures = 35 →
  albums = 3 →
  pictures_per_album = 7 →
  total_pictures - (albums * pictures_per_album) = 14 := by
  sorry

end NUMINAMATH_CALUDE_pictures_in_first_album_l3451_345101


namespace NUMINAMATH_CALUDE_mixture_weight_l3451_345160

/-- Calculates the weight of a mixture of two brands of vegetable ghee -/
theorem mixture_weight 
  (weight_a : ℝ) 
  (weight_b : ℝ) 
  (ratio_a : ℝ) 
  (ratio_b : ℝ) 
  (total_volume : ℝ) 
  (h1 : weight_a = 900) 
  (h2 : weight_b = 850) 
  (h3 : ratio_a = 3) 
  (h4 : ratio_b = 2) 
  (h5 : total_volume = 4) : 
  (weight_a * (ratio_a / (ratio_a + ratio_b)) * total_volume + 
   weight_b * (ratio_b / (ratio_a + ratio_b)) * total_volume) / 1000 = 3.52 := by
sorry

end NUMINAMATH_CALUDE_mixture_weight_l3451_345160


namespace NUMINAMATH_CALUDE_taxi_fare_80_miles_l3451_345184

/-- Calculates the taxi fare for a given distance -/
def taxiFare (distance : ℝ) : ℝ :=
  sorry

theorem taxi_fare_80_miles : 
  -- Given conditions
  (taxiFare 60 = 150) →  -- 60-mile ride costs $150
  (∀ d, taxiFare d = 20 + (taxiFare d - 20) * d / 60) →  -- Flat rate of $20 and proportional charge
  -- Conclusion
  (taxiFare 80 = 193) :=
by
  sorry

end NUMINAMATH_CALUDE_taxi_fare_80_miles_l3451_345184


namespace NUMINAMATH_CALUDE_pythagorean_triple_identification_l3451_345183

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

theorem pythagorean_triple_identification :
  is_pythagorean_triple 5 12 13 ∧
  ¬is_pythagorean_triple 8 12 15 ∧
  is_pythagorean_triple 8 15 17 ∧
  is_pythagorean_triple 9 40 41 :=
by sorry

end NUMINAMATH_CALUDE_pythagorean_triple_identification_l3451_345183


namespace NUMINAMATH_CALUDE_sin_2x_equals_cos_2x_minus_pi_over_4_l3451_345137

theorem sin_2x_equals_cos_2x_minus_pi_over_4 (x : ℝ) : 
  Real.sin (2 * x) = Real.cos (2 * (x - π / 4)) := by
  sorry

end NUMINAMATH_CALUDE_sin_2x_equals_cos_2x_minus_pi_over_4_l3451_345137


namespace NUMINAMATH_CALUDE_new_persons_weight_l3451_345186

theorem new_persons_weight (original_count : ℕ) (weight_increase : ℝ) 
  (replaced_weight1 : ℝ) (replaced_weight2 : ℝ) : 
  original_count = 20 →
  weight_increase = 5 →
  replaced_weight1 = 58 →
  replaced_weight2 = 64 →
  (original_count : ℝ) * weight_increase + replaced_weight1 + replaced_weight2 = 222 :=
by sorry

end NUMINAMATH_CALUDE_new_persons_weight_l3451_345186


namespace NUMINAMATH_CALUDE_reciprocal_roots_quadratic_equation_l3451_345127

theorem reciprocal_roots_quadratic_equation :
  ∀ (α β : ℝ),
  (α^2 - 7*α - 1 = 0) →
  (β^2 - 7*β - 1 = 0) →
  (α + β = 7) →
  (α * β = -1) →
  ((1/α)^2 + 7*(1/α) - 1 = 0) ∧
  ((1/β)^2 + 7*(1/β) - 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_reciprocal_roots_quadratic_equation_l3451_345127

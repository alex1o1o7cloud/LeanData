import Mathlib

namespace store_discount_l518_51815

theorem store_discount (original_price : ℝ) (original_price_pos : original_price > 0) :
  let first_discount := 0.4
  let second_discount := 0.1
  let claimed_discount := 0.5
  let price_after_first_discount := original_price * (1 - first_discount)
  let final_price := price_after_first_discount * (1 - second_discount)
  let actual_discount := 1 - (final_price / original_price)
  actual_discount = 0.46 ∧ claimed_discount - actual_discount = 0.04 := by
  sorry

end store_discount_l518_51815


namespace arithmetic_progression_quartic_l518_51877

theorem arithmetic_progression_quartic (q : ℝ) : 
  (∃ (a d : ℝ), ∀ (x : ℝ), x^4 - 40*x^2 + q = 0 ↔ 
    (x = a - 3*d/2 ∨ x = a - d/2 ∨ x = a + d/2 ∨ x = a + 3*d/2)) → 
  q = 144 := by
sorry


end arithmetic_progression_quartic_l518_51877


namespace unique_pairs_satisfying_W_l518_51851

def W (x : ℕ) : ℤ := x^4 - 3*x^3 + 5*x^2 - 9*x

theorem unique_pairs_satisfying_W :
  ∀ a b : ℕ, a ≠ b ∧ a > 0 ∧ b > 0 ∧ W a = W b ↔ (a = 1 ∧ b = 2) ∨ (a = 2 ∧ b = 1) :=
by sorry

end unique_pairs_satisfying_W_l518_51851


namespace bert_pencil_usage_l518_51808

/-- The number of days it takes to use up a pencil given the total words per pencil and words per puzzle -/
def days_to_use_pencil (total_words_per_pencil : ℕ) (words_per_puzzle : ℕ) : ℕ :=
  total_words_per_pencil / words_per_puzzle

/-- Theorem stating that it takes Bert 14 days to use up a pencil -/
theorem bert_pencil_usage : days_to_use_pencil 1050 75 = 14 := by
  sorry

#eval days_to_use_pencil 1050 75

end bert_pencil_usage_l518_51808


namespace product_x_z_l518_51858

-- Define the parallelogram EFGH
structure Parallelogram :=
  (E F G H : ℝ × ℝ)
  (is_parallelogram : True)  -- This is a placeholder for the parallelogram property

-- Define the lengths of the sides
def side_length (A B : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem product_x_z (EFGH : Parallelogram) 
  (EF_length : side_length EFGH.E EFGH.F = 52)
  (FG_length : ∃ z, side_length EFGH.F EFGH.G = 4 * z^2 + 4)
  (GH_length : ∃ x, side_length EFGH.G EFGH.H = 5 * x + 6)
  (HE_length : side_length EFGH.H EFGH.E = 16) :
  ∃ x z, x * z = 46 * Real.sqrt 3 / 5 :=
sorry

end product_x_z_l518_51858


namespace lesser_solution_quadratic_l518_51820

theorem lesser_solution_quadratic (x : ℝ) : 
  x^2 + 10*x - 24 = 0 ∧ (∀ y : ℝ, y^2 + 10*y - 24 = 0 → x ≤ y) → x = -12 := by
  sorry

end lesser_solution_quadratic_l518_51820


namespace coin_set_existence_l518_51853

def is_valid_coin_set (weights : List Nat) : Prop :=
  ∀ k, k ∈ weights → 
    ∃ (A B : List Nat), 
      A ∪ B = weights.erase k ∧ 
      A.sum = B.sum

theorem coin_set_existence (n : Nat) : 
  (∃ weights : List Nat, 
    weights.length = n ∧ 
    weights.Nodup ∧
    is_valid_coin_set weights) ↔ 
  (Odd n ∧ n ≥ 7) :=
sorry

end coin_set_existence_l518_51853


namespace range_of_t_l518_51825

theorem range_of_t (a b c t : ℝ) 
  (eq1 : 6 * a = 2 * b - 6)
  (eq2 : 6 * a = 3 * c)
  (cond1 : b ≥ 0)
  (cond2 : c ≤ 2)
  (def_t : t = 2 * a + b - c) :
  0 ≤ t ∧ t ≤ 6 := by
  sorry

end range_of_t_l518_51825


namespace andreas_living_room_area_l518_51844

/-- The area of Andrea's living room floor, given a carpet covering 20% of it --/
theorem andreas_living_room_area 
  (carpet_length : ℝ) 
  (carpet_width : ℝ) 
  (carpet_coverage_percent : ℝ) 
  (h1 : carpet_length = 4)
  (h2 : carpet_width = 9)
  (h3 : carpet_coverage_percent = 20) : 
  carpet_length * carpet_width / (carpet_coverage_percent / 100) = 180 := by
sorry

end andreas_living_room_area_l518_51844


namespace rectangular_to_spherical_conversion_l518_51866

def rectangular_to_spherical (x y z : ℝ) : ℝ × ℝ × ℝ := sorry

theorem rectangular_to_spherical_conversion :
  let (ρ, θ, φ) := rectangular_to_spherical (4 * Real.sqrt 2) (-4) 4
  ρ = 8 ∧ θ = 7 * Real.pi / 4 ∧ φ = Real.pi / 3 ∧
  ρ > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧ 0 ≤ φ ∧ φ ≤ Real.pi :=
by sorry

end rectangular_to_spherical_conversion_l518_51866


namespace product_of_consecutive_integers_120_l518_51847

theorem product_of_consecutive_integers_120 : 
  ∃ (a b c d e : ℤ), 
    (b = a + 1) ∧ 
    (a * b = 120) ∧ 
    (d = c + 1) ∧ 
    (e = d + 1) ∧ 
    (c * d * e = 120) ∧ 
    (a + b + c + d + e = 36) :=
by sorry

end product_of_consecutive_integers_120_l518_51847


namespace union_of_positive_and_square_ge_self_is_reals_l518_51811

open Set

theorem union_of_positive_and_square_ge_self_is_reals :
  let M : Set ℝ := {x | x > 0}
  let N : Set ℝ := {x | x^2 ≥ x}
  M ∪ N = univ := by sorry

end union_of_positive_and_square_ge_self_is_reals_l518_51811


namespace black_friday_sales_l518_51846

/-- Calculates the number of televisions sold after a given number of years,
    given an initial sale and yearly increase. -/
def televisionsSold (initialSale : ℕ) (yearlyIncrease : ℕ) (years : ℕ) : ℕ :=
  initialSale + yearlyIncrease * years

/-- Theorem stating that given an initial sale of 327 televisions and
    an increase of 50 televisions per year, the number of televisions
    sold after 3 years will be 477. -/
theorem black_friday_sales : televisionsSold 327 50 3 = 477 := by
  sorry

end black_friday_sales_l518_51846


namespace high_school_students_l518_51887

theorem high_school_students (total : ℕ) (ratio : ℕ) (mia zoe : ℕ) : 
  total = 2500 →
  ratio = 4 →
  mia = ratio * zoe →
  mia + zoe = total →
  mia = 2000 := by
sorry

end high_school_students_l518_51887


namespace P_intersect_Q_equals_Q_l518_51881

-- Define the sets P and Q
def P : Set ℝ := {y | ∃ x : ℝ, y = x^2}
def Q : Set ℝ := {y | ∃ x : ℝ, y = x^2 + 1}

-- Theorem statement
theorem P_intersect_Q_equals_Q : P ∩ Q = Q := by
  sorry

end P_intersect_Q_equals_Q_l518_51881


namespace complex_roots_on_circle_l518_51839

theorem complex_roots_on_circle : 
  ∀ z : ℂ, (z + 2)^5 = 64 * z^5 → Complex.abs (z + 2/15) = 2/15 := by
  sorry

end complex_roots_on_circle_l518_51839


namespace pattern_solution_l518_51888

theorem pattern_solution (n : ℕ+) (a b : ℕ+) :
  (∀ k : ℕ+, Real.sqrt (k + k / (k^2 - 1)) = k * Real.sqrt (k / (k^2 - 1))) →
  (Real.sqrt (8 + b / a) = 8 * Real.sqrt (b / a)) →
  a = 63 ∧ b = 8 := by sorry

end pattern_solution_l518_51888


namespace distance_between_points_l518_51824

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (2, 6)
  let p2 : ℝ × ℝ := (5, 2)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = 5 := by
sorry

end distance_between_points_l518_51824


namespace arithmetic_sequence_sum_l518_51891

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  ArithmeticSequence a →
  (a 1 - a 4 - a 8 - a 12 + a 15 = 2) →
  (a 3 + a 13 = -4) := by
  sorry

end arithmetic_sequence_sum_l518_51891


namespace minkowski_sum_properties_l518_51896

/-- A convex polygon with perimeter and area -/
structure ConvexPolygon where
  perimeter : ℝ
  area : ℝ

/-- The Minkowski sum of a convex polygon and a circle -/
def minkowskiSum (K : ConvexPolygon) (r : ℝ) : Set (ℝ × ℝ) := sorry

/-- The length of the curve resulting from the Minkowski sum -/
def curveLength (K : ConvexPolygon) (r : ℝ) : ℝ := sorry

/-- The area of the figure bounded by the Minkowski sum -/
def boundedArea (K : ConvexPolygon) (r : ℝ) : ℝ := sorry

/-- Main theorem about the Minkowski sum of a convex polygon and a circle -/
theorem minkowski_sum_properties (K : ConvexPolygon) (r : ℝ) :
  (curveLength K r = K.perimeter + 2 * Real.pi * r) ∧
  (boundedArea K r = K.area + K.perimeter * r + Real.pi * r^2) := by
  sorry

end minkowski_sum_properties_l518_51896


namespace shirt_pricing_l518_51862

theorem shirt_pricing (total_shirts : ℕ) (first_shirt_price second_shirt_price : ℚ) 
  (remaining_shirts : ℕ) (min_avg_remaining : ℚ) :
  total_shirts = 6 →
  first_shirt_price = 40 →
  second_shirt_price = 50 →
  remaining_shirts = 4 →
  min_avg_remaining = 52.5 →
  (first_shirt_price + second_shirt_price + remaining_shirts * min_avg_remaining) / total_shirts = 50 := by
  sorry

end shirt_pricing_l518_51862


namespace largest_constant_inequality_l518_51850

theorem largest_constant_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x / Real.sqrt (y + z) + y / Real.sqrt (z + x) + z / Real.sqrt (x + y) ≤ (Real.sqrt 6 / 2) * Real.sqrt (x + y + z) ∧
  ∀ k : ℝ, (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 →
    a / Real.sqrt (b + c) + b / Real.sqrt (c + a) + c / Real.sqrt (a + b) ≤ k * Real.sqrt (a + b + c)) →
  k ≤ Real.sqrt 6 / 2 :=
by sorry

end largest_constant_inequality_l518_51850


namespace decimal_25_equals_base5_100_l518_51864

/-- Converts a natural number to its base 5 representation --/
def toBaseFive (n : ℕ) : List ℕ :=
  if n < 5 then [n]
  else (n % 5) :: toBaseFive (n / 5)

/-- Theorem: The decimal number 25 is equivalent to 100₅ in base 5 --/
theorem decimal_25_equals_base5_100 : toBaseFive 25 = [0, 0, 1] := by
  sorry

end decimal_25_equals_base5_100_l518_51864


namespace inequality_problem_l518_51807

theorem inequality_problem (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a^2 + b^2 + 2 ≥ 2*a + 2*b) ∧ 
  (Real.log (a*b + 1) ≥ 0) ∧ 
  (Real.sqrt (a + b) ≥ 2) ∧ 
  ¬(∀ (x y : ℝ), x > 0 → y > 0 → x^3 + y^3 ≥ 2*x*y^2) :=
by sorry

end inequality_problem_l518_51807


namespace bus_problem_l518_51848

/-- The number of children initially on the bus -/
def initial_children : ℕ := 5

/-- The number of children who got off the bus -/
def children_off : ℕ := 63

/-- The number of children who got on the bus -/
def children_on : ℕ := children_off + 9

/-- The number of children on the bus after the changes -/
def final_children : ℕ := 14

theorem bus_problem :
  initial_children - children_off + children_on = final_children :=
by sorry

end bus_problem_l518_51848


namespace sun_division_l518_51806

/-- Represents the share distribution among x, y, and z -/
structure ShareDistribution where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The problem statement -/
theorem sun_division (s : ShareDistribution) :
  s.x = 1 →                -- For each rupee x gets
  s.y = 0.45 →             -- y gets 45 paisa (0.45 rupees)
  s.z = 0.5 →              -- z gets 50 paisa (0.5 rupees)
  s.y * (1 / 0.45) = 45 →  -- The share of y is Rs. 45
  s.x * (1 / 0.45) + s.y * (1 / 0.45) + s.z * (1 / 0.45) = 195 := by
  sorry

#check sun_division

end sun_division_l518_51806


namespace solubility_product_scientific_notation_l518_51840

theorem solubility_product_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 0.0000000028 = a * (10 : ℝ) ^ n :=
by sorry

end solubility_product_scientific_notation_l518_51840


namespace range_of_a_l518_51800

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, |x - a| + |x - 1| ≤ 3) → -2 ≤ a ∧ a ≤ 4 := by
  sorry

end range_of_a_l518_51800


namespace no_professors_are_student_council_members_l518_51861

-- Define the universe
variable (U : Type)

-- Define the predicates
variable (Professor : U → Prop)
variable (StudentCouncilMember : U → Prop)
variable (Wise : U → Prop)

-- State the theorem
theorem no_professors_are_student_council_members
  (h1 : ∀ x, Professor x → Wise x)
  (h2 : ∀ x, StudentCouncilMember x → ¬Wise x) :
  ∀ x, Professor x → ¬StudentCouncilMember x :=
by sorry

end no_professors_are_student_council_members_l518_51861


namespace orange_pyramid_count_l518_51860

/-- Calculates the number of oranges in a pyramid layer given its width and length -/
def layer_oranges (width : ℕ) (length : ℕ) : ℕ := width * length

/-- Calculates the total number of oranges in a pyramid stack -/
def total_oranges (base_width : ℕ) (base_length : ℕ) : ℕ :=
  let base := layer_oranges base_width base_length
  let layer2 := layer_oranges (base_width - 1) (base_length - 1)
  let layer3 := layer_oranges (base_width - 2) (base_length - 2)
  let layer4 := layer_oranges (base_width - 3) (base_length - 3)
  let layer5 := layer_oranges (base_width - 4) (base_length - 4)
  let layer6 := layer_oranges (base_width - 5) (base_length - 5)
  let layer7 := layer_oranges (base_width - 6) (base_length - 6)
  base + layer2 + layer3 + layer4 + layer5 + layer6 + layer7 + 1

theorem orange_pyramid_count :
  total_oranges 7 10 = 225 := by
  sorry

end orange_pyramid_count_l518_51860


namespace smallest_integer_l518_51802

theorem smallest_integer (a b : ℕ+) (ha : a = 60) (h : Nat.lcm a b / Nat.gcd a b = 44) :
  ∃ (m : ℕ+), ∀ (n : ℕ+), (Nat.lcm a n / Nat.gcd a n = 44) → m ≤ n ∧ m = 165 :=
sorry

end smallest_integer_l518_51802


namespace family_egg_count_l518_51829

/-- Calculates the final number of eggs a family has after various events --/
theorem family_egg_count (initial_eggs : ℚ) 
                          (mother_used : ℚ) 
                          (father_used : ℚ) 
                          (chicken1_laid : ℚ) 
                          (chicken2_laid : ℚ) 
                          (chicken3_laid : ℚ) 
                          (chicken4_laid : ℚ) 
                          (oldest_child_took : ℚ) 
                          (youngest_child_broke : ℚ) : 
  initial_eggs = 25 ∧ 
  mother_used = 7.5 ∧ 
  father_used = 2.5 ∧ 
  chicken1_laid = 2.5 ∧ 
  chicken2_laid = 3 ∧ 
  chicken3_laid = 4.5 ∧ 
  chicken4_laid = 1 ∧ 
  oldest_child_took = 1.5 ∧ 
  youngest_child_broke = 0.5 → 
  initial_eggs - (mother_used + father_used) + 
  (chicken1_laid + chicken2_laid + chicken3_laid + chicken4_laid) - 
  (oldest_child_took + youngest_child_broke) = 24 := by
  sorry


end family_egg_count_l518_51829


namespace angle_C_measure_l518_51828

theorem angle_C_measure (A B C : ℝ) (h1 : 4 * Real.sin A + 2 * Real.cos B = 4)
  (h2 : (1/2) * Real.sin B + Real.cos A = Real.sqrt 3 / 2) : C = π / 6 := by
  sorry

end angle_C_measure_l518_51828


namespace geometric_sequence_common_ratio_l518_51831

def geometric_sequence (a : ℕ → ℝ) := ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a) 
  (h_a2 : a 2 = 2) 
  (h_a5 : a 5 = 1/4) : 
  ∃ q : ℝ, q = 1/2 ∧ ∀ n : ℕ, a (n + 1) = a n * q :=
sorry

end geometric_sequence_common_ratio_l518_51831


namespace three_boys_ages_exist_l518_51832

theorem three_boys_ages_exist : ∃ (A B C : ℝ), 
  A + B + C = 29.5 ∧ 
  C = 11.3 ∧ 
  (A = 2 * B ∨ B = 2 * C ∨ A = 2 * C) ∧
  A > 0 ∧ B > 0 ∧ C > 0 := by
  sorry

end three_boys_ages_exist_l518_51832


namespace max_red_dragons_l518_51819

-- Define the dragon colors
inductive DragonColor
| Red
| Green
| Blue

-- Define the structure of a dragon
structure Dragon where
  color : DragonColor
  heads : Fin 3 → Bool  -- Each head is either truthful (true) or lying (false)

-- Define the statements made by each head
def headStatements (d : Dragon) (left right : DragonColor) : Prop :=
  (d.heads 0 = (left = DragonColor.Green)) ∧
  (d.heads 1 = (right = DragonColor.Blue)) ∧
  (d.heads 2 = (left ≠ DragonColor.Red ∧ right ≠ DragonColor.Red))

-- Define the condition that at least one head tells the truth
def atLeastOneTruthful (d : Dragon) : Prop :=
  ∃ i : Fin 3, d.heads i = true

-- Define the arrangement of dragons around the table
def validArrangement (arrangement : Fin 530 → Dragon) : Prop :=
  ∀ i : Fin 530,
    let left := arrangement ((i.val - 1 + 530) % 530)
    let right := arrangement ((i.val + 1) % 530)
    headStatements (arrangement i) left.color right.color ∧
    atLeastOneTruthful (arrangement i)

-- The main theorem
theorem max_red_dragons :
  ∀ arrangement : Fin 530 → Dragon,
    validArrangement arrangement →
    (∃ n : Nat, n ≤ 176 ∧ (∀ i : Fin 530, (arrangement i).color = DragonColor.Red → i.val < n)) :=
sorry

end max_red_dragons_l518_51819


namespace highway_traffic_l518_51812

/-- The number of vehicles involved in accidents per 100 million vehicles -/
def accident_rate : ℚ := 75

/-- The total number of vehicles involved in accidents -/
def total_accidents : ℕ := 4500

/-- The number of vehicles (in millions) that traveled on the highway -/
def total_vehicles : ℕ := 6000

theorem highway_traffic :
  (accident_rate / 100000000) * (total_vehicles * 1000000) = total_accidents :=
sorry

end highway_traffic_l518_51812


namespace distinct_sums_count_l518_51823

def bag_X : Finset ℕ := {2, 5, 7}
def bag_Y : Finset ℕ := {1, 4, 8}

theorem distinct_sums_count : 
  Finset.card ((bag_X.product bag_Y).image (fun p => p.1 + p.2)) = 8 := by
  sorry

end distinct_sums_count_l518_51823


namespace expression_result_l518_51841

theorem expression_result : (3.242 * 14) / 100 = 0.45388 := by
  sorry

end expression_result_l518_51841


namespace trivia_team_total_score_l518_51821

def trivia_team_points : Prop :=
  let total_members : ℕ := 12
  let absent_members : ℕ := 4
  let present_members : ℕ := total_members - absent_members
  let scores : List ℕ := [8, 12, 9, 5, 10, 7, 14, 11]
  scores.length = present_members ∧ scores.sum = 76

theorem trivia_team_total_score : trivia_team_points := by
  sorry

end trivia_team_total_score_l518_51821


namespace point_movement_to_y_axis_l518_51873

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define the problem statement
theorem point_movement_to_y_axis (a : ℝ) :
  let P : Point := ⟨a + 1, a⟩
  let P₁ : Point := ⟨P.x + 3, P.y⟩
  P₁.x = 0 → P = ⟨-3, -4⟩ := by
  sorry

end point_movement_to_y_axis_l518_51873


namespace arithmetic_sequence_terms_l518_51859

/-- The number of terms between 400 and 600 in an arithmetic sequence -/
theorem arithmetic_sequence_terms (a₁ : ℝ) (d : ℝ) (n : ℕ) :
  a₁ = 110 →
  d = 6 →
  (∃ k₁ k₂ : ℕ, 
    a₁ + (k₁ - 1) * d ≥ 400 ∧
    a₁ + (k₁ - 1) * d < a₁ + k₁ * d ∧
    a₁ + (k₂ - 1) * d ≤ 600 ∧
    a₁ + k₂ * d > 600 ∧
    k₂ - k₁ + 1 = 33) :=
by
  sorry


end arithmetic_sequence_terms_l518_51859


namespace systematic_sampling_theorem_l518_51836

/-- Represents a systematic sampling setup -/
structure SystematicSampling where
  total_students : ℕ
  sample_size : ℕ
  group_size : ℕ
  first_group_number : ℕ

/-- The theorem for systematic sampling -/
theorem systematic_sampling_theorem (s : SystematicSampling)
  (h1 : s.total_students = 300)
  (h2 : s.sample_size = 20)
  (h3 : s.group_size = s.total_students / s.sample_size)
  (h4 : s.first_group_number < s.group_size)
  (h5 : 231 = s.first_group_number + 15 * s.group_size) :
  s.first_group_number = 6 := by
  sorry

#check systematic_sampling_theorem

end systematic_sampling_theorem_l518_51836


namespace solve_for_t_l518_51865

theorem solve_for_t (s t : ℤ) (eq1 : 9 * s + 5 * t = 108) (eq2 : s = t - 2) : t = 9 := by
  sorry

end solve_for_t_l518_51865


namespace consecutive_numbers_sum_l518_51813

theorem consecutive_numbers_sum (x : ℕ) : 
  x * (x + 1) = 12650 → x + (x + 1) = 225 := by sorry

end consecutive_numbers_sum_l518_51813


namespace soldiers_divisible_by_six_l518_51878

theorem soldiers_divisible_by_six (b : ℕ+) : 
  ∃ k : ℕ, b + 3 * b ^ 2 + 2 * b ^ 3 = 6 * k := by
  sorry

end soldiers_divisible_by_six_l518_51878


namespace cd_cost_l518_51889

/-- Given that two identical CDs cost $24, prove that seven CDs cost $84. -/
theorem cd_cost (cost_of_two : ℕ) (h : cost_of_two = 24) : 7 * (cost_of_two / 2) = 84 := by
  sorry

end cd_cost_l518_51889


namespace pole_height_l518_51894

theorem pole_height (cable_ground_distance : ℝ) (person_distance : ℝ) (person_height : ℝ)
  (h1 : cable_ground_distance = 5)
  (h2 : person_distance = 4)
  (h3 : person_height = 3) :
  let pole_height := cable_ground_distance * person_height / (cable_ground_distance - person_distance)
  pole_height = 15 := by sorry

end pole_height_l518_51894


namespace equilateral_triangle_complex_l518_51803

/-- Given complex numbers a, b, c forming an equilateral triangle with side length 24
    and |a + b + c| = 48, prove that |ab + ac + bc| = 768. -/
theorem equilateral_triangle_complex (a b c : ℂ) :
  (∃ (w : ℂ), w^3 = 1 ∧ w ≠ 1 ∧ b - a = 24 * w ∧ c - a = 24 * w^2) →
  Complex.abs (a + b + c) = 48 →
  Complex.abs (a * b + a * c + b * c) = 768 := by
sorry

end equilateral_triangle_complex_l518_51803


namespace fixed_point_exponential_function_l518_51809

theorem fixed_point_exponential_function (a : ℝ) (ha : a > 0) (ha_neq_one : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^(x + 3) - 2
  f (-3) = -1 := by sorry

end fixed_point_exponential_function_l518_51809


namespace oprah_car_giveaway_l518_51827

theorem oprah_car_giveaway (initial_cars final_cars years : ℕ) 
  (h1 : initial_cars = 3500)
  (h2 : final_cars = 500)
  (h3 : years = 60) :
  (initial_cars - final_cars) / years = 50 :=
by sorry

end oprah_car_giveaway_l518_51827


namespace fraction_addition_l518_51884

theorem fraction_addition (d : ℝ) : (6 + 5 * d) / 11 + 3 = (39 + 5 * d) / 11 := by
  sorry

end fraction_addition_l518_51884


namespace cafe_latte_cost_correct_l518_51892

/-- Represents the cost of a cafe latte -/
def cafe_latte_cost : ℝ := 1.50

/-- Represents the cost of a cappuccino -/
def cappuccino_cost : ℝ := 2

/-- Represents the cost of an iced tea -/
def iced_tea_cost : ℝ := 3

/-- Represents the cost of an espresso -/
def espresso_cost : ℝ := 1

/-- Represents the number of cappuccinos Sandy ordered -/
def num_cappuccinos : ℕ := 3

/-- Represents the number of iced teas Sandy ordered -/
def num_iced_teas : ℕ := 2

/-- Represents the number of cafe lattes Sandy ordered -/
def num_lattes : ℕ := 2

/-- Represents the number of espressos Sandy ordered -/
def num_espressos : ℕ := 2

/-- Represents the amount Sandy paid -/
def amount_paid : ℝ := 20

/-- Represents the change Sandy received -/
def change_received : ℝ := 3

theorem cafe_latte_cost_correct :
  cafe_latte_cost * num_lattes +
  cappuccino_cost * num_cappuccinos +
  iced_tea_cost * num_iced_teas +
  espresso_cost * num_espressos =
  amount_paid - change_received :=
by sorry

end cafe_latte_cost_correct_l518_51892


namespace turtleneck_sweater_profit_l518_51867

/-- Represents the pricing strategy and profit calculation for a store selling turtleneck sweaters -/
theorem turtleneck_sweater_profit (C : ℝ) : 
  let initial_markup := 0.20
  let new_year_markup := 0.25
  let february_discount := 0.08
  let SP1 := C * (1 + initial_markup)
  let SP2 := SP1 * (1 + new_year_markup)
  let SPF := SP2 * (1 - february_discount)
  let profit := SPF - C
  profit = 0.38 * C :=
by sorry

end turtleneck_sweater_profit_l518_51867


namespace fair_hair_percentage_l518_51837

/-- Given that 10% of employees are women with fair hair and 40% of fair-haired employees
    are women, prove that 25% of employees have fair hair. -/
theorem fair_hair_percentage
  (total_employees : ℕ)
  (women_fair_hair_percentage : ℚ)
  (women_percentage_of_fair_hair : ℚ)
  (h1 : women_fair_hair_percentage = 1 / 10)
  (h2 : women_percentage_of_fair_hair = 2 / 5)
  : (total_employees : ℚ) * 1 / 4 = (total_employees : ℚ) * women_fair_hair_percentage / women_percentage_of_fair_hair :=
by sorry

end fair_hair_percentage_l518_51837


namespace hyperbola_eccentricity_sqrt_two_l518_51869

/-- A hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  positive_a : 0 < a
  positive_b : 0 < b

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola a b) : ℝ := sorry

/-- A point on the plane -/
structure Point (α : Type*) where
  x : α
  y : α

/-- The area of a triangle given three points -/
def triangle_area (A B C : Point ℝ) : ℝ := sorry

/-- Theorem: The eccentricity of a hyperbola with specific properties is √2 -/
theorem hyperbola_eccentricity_sqrt_two 
  (a b c : ℝ) (h : Hyperbola a b) 
  (M N : Point ℝ) (A : Point ℝ) :
  (∃ F₁ F₂ : Point ℝ, 
    -- M and N are on the asymptote
    -- MF₁NF₂ is a rectangle
    -- A is a vertex of the hyperbola
    triangle_area A M N = (1/2) * c^2) →
  eccentricity h = Real.sqrt 2 := by sorry

end hyperbola_eccentricity_sqrt_two_l518_51869


namespace stratified_sampling_result_l518_51898

/-- Represents a department in the unit -/
inductive Department
| A
| B
| C

/-- The number of employees in each department -/
def employeeCount (d : Department) : ℕ :=
  match d with
  | .A => 27
  | .B => 63
  | .C => 81

/-- The number of people drawn from department B -/
def drawnFromB : ℕ := 7

/-- The number of people drawn from a department in stratified sampling -/
def peopleDrawn (d : Department) : ℚ :=
  (employeeCount d : ℚ) * (drawnFromB : ℚ) / (employeeCount .B : ℚ)

/-- The total number of people drawn from all departments -/
def totalDrawn : ℚ :=
  peopleDrawn .A + peopleDrawn .B + peopleDrawn .C

theorem stratified_sampling_result :
  totalDrawn = 23 := by sorry

end stratified_sampling_result_l518_51898


namespace age_puzzle_l518_51804

theorem age_puzzle (A : ℕ) (h : A = 32) : ∃ N : ℚ, N * (A + 4) - 4 * (A - 4) = A ∧ N = 4 := by
  sorry

end age_puzzle_l518_51804


namespace jeff_shelter_cats_l518_51895

/-- Calculates the number of cats in Jeff's shelter after a series of events -/
def cats_in_shelter (initial : ℕ) (monday_found : ℕ) (tuesday_found : ℕ) (wednesday_adopted : ℕ) : ℕ :=
  initial + monday_found + tuesday_found - wednesday_adopted

/-- Theorem stating the number of cats in Jeff's shelter after the given events -/
theorem jeff_shelter_cats : cats_in_shelter 20 2 1 6 = 17 := by
  sorry

#eval cats_in_shelter 20 2 1 6

end jeff_shelter_cats_l518_51895


namespace smallest_sum_x_y_l518_51883

theorem smallest_sum_x_y (x y : ℕ+) 
  (h1 : (2010 : ℚ) / 2011 < (x : ℚ) / y)
  (h2 : (x : ℚ) / y < (2011 : ℚ) / 2012) :
  ∀ (a b : ℕ+), 
    ((2010 : ℚ) / 2011 < (a : ℚ) / b ∧ (a : ℚ) / b < (2011 : ℚ) / 2012) →
    (x + y : ℕ) ≤ (a + b : ℕ) ∧
    (x + y : ℕ) = 8044 :=
by sorry

end smallest_sum_x_y_l518_51883


namespace average_problem_l518_51871

theorem average_problem (a b c d P : ℝ) :
  (a + b + c + d) / 4 = 8 →
  (a + b + c + d + P) / 5 = P →
  P = 8 := by
sorry

end average_problem_l518_51871


namespace boat_speed_in_still_water_l518_51854

/-- Proves that the speed of a boat in still water is 20 km/hr -/
theorem boat_speed_in_still_water : 
  ∀ (x : ℝ), 
    (5 : ℝ) = 5 → -- Rate of current is 5 km/hr
    ((x + 5) * (21 / 60) = (35 / 4 : ℝ)) → -- Distance travelled downstream in 21 minutes is 8.75 km
    x = 20 := by
  sorry

end boat_speed_in_still_water_l518_51854


namespace average_marks_second_class_l518_51875

theorem average_marks_second_class 
  (students1 : ℕ) 
  (students2 : ℕ) 
  (avg1 : ℝ) 
  (avg_combined : ℝ) :
  students1 = 25 →
  students2 = 40 →
  avg1 = 50 →
  avg_combined = 59.23076923076923 →
  let total_students := students1 + students2
  let avg2 := (avg_combined * total_students - avg1 * students1) / students2
  avg2 = 65 := by sorry

end average_marks_second_class_l518_51875


namespace city_tax_solution_l518_51849

/-- Represents the tax system of a city --/
structure CityTax where
  residents : ℕ
  taxPerResident : ℕ

/-- The conditions of the tax system --/
def taxConditions (ct : CityTax) : Prop :=
  (ct.residents + 3000) * (ct.taxPerResident - 10) = ct.residents * ct.taxPerResident ∧
  (ct.residents - 1000) * (ct.taxPerResident + 10) = ct.residents * ct.taxPerResident

/-- The theorem stating the solution to the problem --/
theorem city_tax_solution (ct : CityTax) (h : taxConditions ct) :
  ct.residents = 3000 ∧ ct.taxPerResident = 20 ∧ ct.residents * ct.taxPerResident = 60000 := by
  sorry


end city_tax_solution_l518_51849


namespace eight_books_distribution_l518_51835

/-- The number of ways to distribute indistinguishable books between two locations --/
def distribute_books (total : ℕ) : ℕ := 
  if total ≥ 2 then total - 1 else 0

/-- Theorem: Distributing 8 indistinguishable books between two locations, 
    with at least one book in each location, results in 7 different ways --/
theorem eight_books_distribution : distribute_books 8 = 7 := by
  sorry

end eight_books_distribution_l518_51835


namespace negation_of_universal_statement_l518_51899

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^2 + 2 > 0) ↔ (∃ x : ℝ, x^2 + 2 ≤ 0) := by sorry

end negation_of_universal_statement_l518_51899


namespace michael_needs_eleven_more_l518_51897

/-- Given Michael's current money and the total cost of items he wants to buy,
    calculate the additional money he needs. -/
def additional_money_needed (current_money total_cost : ℕ) : ℕ :=
  if total_cost > current_money then total_cost - current_money else 0

/-- Theorem stating that Michael needs $11 more to buy all items. -/
theorem michael_needs_eleven_more :
  let current_money : ℕ := 50
  let cake_cost : ℕ := 20
  let bouquet_cost : ℕ := 36
  let balloons_cost : ℕ := 5
  let total_cost : ℕ := cake_cost + bouquet_cost + balloons_cost
  additional_money_needed current_money total_cost = 11 := by
  sorry

end michael_needs_eleven_more_l518_51897


namespace coin_flip_probability_l518_51818

/-- The probability of getting exactly k successes in n trials --/
def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k : ℚ) * p^k * (1 - p)^(n - k)

/-- The probability of getting exactly 4 heads in 10 flips of a coin with 3/7 probability of heads --/
theorem coin_flip_probability : 
  binomial_probability 10 4 (3/7) = 69874560 / 282576201 := by
  sorry

end coin_flip_probability_l518_51818


namespace quadrilateral_diagonal_segment_length_l518_51872

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a quadrilateral -/
structure Quadrilateral :=
  (P Q R S : Point)

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Calculates the area of a triangle given three points -/
def triangleArea (p1 p2 p3 : Point) : ℝ := sorry

/-- Determines if a quadrilateral is convex -/
def isConvex (quad : Quadrilateral) : Prop := sorry

/-- Finds the intersection point of two line segments -/
def intersection (p1 p2 p3 p4 : Point) : Point := sorry

theorem quadrilateral_diagonal_segment_length 
  (PQRS : Quadrilateral) 
  (T : Point) :
  isConvex PQRS →
  distance PQRS.P PQRS.Q = 15 →
  distance PQRS.R PQRS.S = 20 →
  distance PQRS.P PQRS.R = 25 →
  T = intersection PQRS.P PQRS.R PQRS.Q PQRS.S →
  triangleArea PQRS.P T PQRS.R = triangleArea PQRS.Q T PQRS.S →
  distance PQRS.P T = 75 / 7 := by
  sorry

end quadrilateral_diagonal_segment_length_l518_51872


namespace roses_given_l518_51857

-- Define the total number of students
def total_students : ℕ := 28

-- Define the relationship between flowers
def flower_relationship (daffodils roses tulips : ℕ) : Prop :=
  roses = 4 * daffodils ∧ tulips = 10 * roses

-- Define the total number of flowers given
def total_flowers (boys girls : ℕ) : ℕ := boys * girls

-- Define the constraint that the total number of students is the sum of boys and girls
def student_constraint (boys girls : ℕ) : Prop :=
  boys + girls = total_students

-- Theorem statement
theorem roses_given (boys girls daffodils roses tulips : ℕ) :
  student_constraint boys girls →
  flower_relationship daffodils roses tulips →
  total_flowers boys girls = daffodils + roses + tulips →
  roses = 16 := by
  sorry

end roses_given_l518_51857


namespace jans_cable_sections_l518_51830

theorem jans_cable_sections (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hxy : y < x) :
  z = (51 * x) / (160 * y) → z = ((51 : ℕ) / 160) * (x / y) := by
sorry

end jans_cable_sections_l518_51830


namespace int_poly5_root_count_l518_51826

/-- A polynomial of degree 5 with integer coefficients -/
structure IntPoly5 where
  b : ℤ
  c : ℤ
  d : ℤ
  e : ℤ
  f : ℤ

/-- The set of possible numbers of integer roots for an IntPoly5 -/
def possibleRootCounts : Set ℕ := {0, 1, 2, 4, 5}

/-- The number of integer roots (counting multiplicity) of an IntPoly5 -/
def numIntegerRoots (p : IntPoly5) : ℕ := sorry

/-- Theorem stating that the number of integer roots of an IntPoly5 is in the set of possible root counts -/
theorem int_poly5_root_count (p : IntPoly5) : numIntegerRoots p ∈ possibleRootCounts := by sorry

end int_poly5_root_count_l518_51826


namespace quadratic_factorization_l518_51893

/-- A quadratic expression in x and y with a parameter k -/
def quadratic (x y : ℝ) (k : ℝ) : ℝ := 2 * x^2 - 6 * y^2 + x * y + k * x + 6

/-- Predicate to check if an expression is a product of two linear factors -/
def is_product_of_linear_factors (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (a b c d e g : ℝ), ∀ x y, f x y = (a * x + b * y + c) * (d * x + e * y + g)

/-- Theorem stating that if the quadratic expression is factorizable, then k = 7 or k = -7 -/
theorem quadratic_factorization (k : ℝ) :
  is_product_of_linear_factors (quadratic · · k) → k = 7 ∨ k = -7 :=
sorry

end quadratic_factorization_l518_51893


namespace absolute_value_of_c_l518_51801

theorem absolute_value_of_c (a b c : ℤ) : 
  a * (3 + Complex.I)^4 + b * (3 + Complex.I)^3 + c * (3 + Complex.I)^2 + b * (3 + Complex.I) + a = 0 →
  Int.gcd a (Int.gcd b c) = 1 →
  |c| = 1106 := by
  sorry

end absolute_value_of_c_l518_51801


namespace part_one_part_two_l518_51834

-- Define the universe set U
def U : Set ℕ := {0, 1, 2, 3}

-- Define the set A as a function of m
def A (m : ℝ) : Set ℕ := {x ∈ U | x^2 + m*x = 0}

-- Part 1
theorem part_one (m : ℝ) : (Aᶜ m = {1, 2}) → m = -3 := by sorry

-- Part 2
theorem part_two (m : ℝ) : (∃! x, x ∈ A m) → m = 0 := by sorry

end part_one_part_two_l518_51834


namespace words_per_page_smaller_type_l518_51845

/-- Calculates words per page in smaller type given article details -/
def wordsPerPageSmallerType (totalWords : ℕ) (totalPages : ℕ) (smallerTypePages : ℕ) (wordsPerPageLargerType : ℕ) : ℕ :=
  let largerTypePages := totalPages - smallerTypePages
  let wordsInLargerType := largerTypePages * wordsPerPageLargerType
  let wordsInSmallerType := totalWords - wordsInLargerType
  wordsInSmallerType / smallerTypePages

/-- Proves that words per page in smaller type is 2400 for given article details -/
theorem words_per_page_smaller_type :
  wordsPerPageSmallerType 48000 21 17 1800 = 2400 := by
  sorry

end words_per_page_smaller_type_l518_51845


namespace k_range_for_single_extremum_l518_51876

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := Real.exp x / x + k * (Real.log x - x)

theorem k_range_for_single_extremum (k : ℝ) :
  (∀ x > 0, x ≠ 1 → (deriv (f k)) x ≠ 0) →
  (deriv (f k)) 1 = 0 →
  k ≤ Real.exp 1 :=
sorry

end k_range_for_single_extremum_l518_51876


namespace fraction_to_decimal_l518_51810

theorem fraction_to_decimal : (47 : ℚ) / (2^3 * 5^4) = 0.0094 := by sorry

end fraction_to_decimal_l518_51810


namespace quadratic_equation_solution_l518_51879

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x ↦ x^2 - 2*x
  ∃ x₁ x₂ : ℝ, x₁ = 0 ∧ x₂ = 2 ∧ (∀ x : ℝ, f x = 0 ↔ x = x₁ ∨ x = x₂) :=
by sorry

end quadratic_equation_solution_l518_51879


namespace triangle_midpoint_x_sum_l518_51855

theorem triangle_midpoint_x_sum (a b c : ℝ) (S : ℝ) : 
  a + b + c = S → 
  (a + b) / 2 + (b + c) / 2 + (c + a) / 2 = S :=
by sorry

end triangle_midpoint_x_sum_l518_51855


namespace cost_per_chicken_problem_l518_51805

/-- Given a total number of birds, a fraction of ducks, and the total cost to feed chickens,
    calculate the cost per chicken. -/
def cost_per_chicken (total_birds : ℕ) (duck_fraction : ℚ) (total_cost : ℚ) : ℚ :=
  let chicken_fraction : ℚ := 1 - duck_fraction
  let num_chickens : ℚ := chicken_fraction * total_birds
  total_cost / num_chickens

theorem cost_per_chicken_problem :
  cost_per_chicken 15 (1/3) 20 = 2 := by
  sorry

end cost_per_chicken_problem_l518_51805


namespace triangle_property_l518_51870

-- Define a triangle ABC
structure Triangle :=
  (A B C : Real)  -- angles
  (a b c : Real)  -- sides

-- Define the theorem
theorem triangle_property (t : Triangle) 
  (h : t.a^2 + t.b^2 = 2018 * t.c^2) :
  (2 * Real.sin t.A * Real.sin t.B * Real.cos t.C) / (1 - Real.cos t.C ^ 2) = 2017 := by
  sorry

end triangle_property_l518_51870


namespace square_of_sum_l518_51842

theorem square_of_sum (x y k m : ℝ) (h1 : x * y = k) (h2 : x^2 + y^2 = m) :
  (x + y)^2 = m + 2*k := by
  sorry

end square_of_sum_l518_51842


namespace diagonal_cut_square_area_l518_51817

/-- A square cut along its diagonal with specific translations of one half -/
structure DiagonalCutSquare where
  /-- The side length of the square -/
  side : ℝ
  /-- The first translation distance -/
  trans1 : ℝ
  /-- The second translation distance -/
  trans2 : ℝ
  /-- The first translation is 3 units -/
  h_trans1 : trans1 = 3
  /-- The second translation is 5 units -/
  h_trans2 : trans2 = 5
  /-- The overlapping areas after each translation are equal -/
  h_equal_overlap : trans1 * trans2 = trans2 * (side - trans1 - trans2)

/-- The theorem stating that under the given conditions, the square's area is 121 -/
theorem diagonal_cut_square_area (s : DiagonalCutSquare) : s.side^2 = 121 := by
  sorry

end diagonal_cut_square_area_l518_51817


namespace two_digit_addition_puzzle_l518_51874

theorem two_digit_addition_puzzle :
  ∀ (A B : ℕ),
    A ≠ B →
    A < 10 →
    B < 10 →
    10 * A + B + 25 = 10 * B + 3 →
    B = 8 :=
by sorry

end two_digit_addition_puzzle_l518_51874


namespace defective_product_scenarios_l518_51816

theorem defective_product_scenarios 
  (total_products : Nat) 
  (defective_products : Nat) 
  (good_products : Nat) 
  (h1 : total_products = 10)
  (h2 : defective_products = 4)
  (h3 : good_products = 6)
  (h4 : total_products = defective_products + good_products) :
  (Nat.choose good_products 1) * (Nat.choose defective_products 1) * (Nat.factorial 4) = 
  (number_of_scenarios : Nat) := by
  sorry

end defective_product_scenarios_l518_51816


namespace a_6_value_l518_51868

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem a_6_value (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 4 = 1 ∧ a 8 = 3) ∨ (a 4 = 3 ∧ a 8 = 1) →
  a 6 = Real.sqrt 3 := by
  sorry

end a_6_value_l518_51868


namespace complex_expression_equals_negative_i_l518_51863

theorem complex_expression_equals_negative_i :
  let i : ℂ := Complex.I
  (1 + 2*i) * i^3 + 2*i^2 = -i :=
by sorry

end complex_expression_equals_negative_i_l518_51863


namespace original_ghee_quantity_l518_51856

/-- Represents the composition of ghee -/
structure GheeComposition where
  pure : Rat
  vanaspati : Rat
  impurities : Rat

/-- The original ghee composition -/
def originalComposition : GheeComposition :=
  { pure := 40/100, vanaspati := 30/100, impurities := 30/100 }

/-- The desired final ghee composition -/
def desiredComposition : GheeComposition :=
  { pure := 45/100, vanaspati := 25/100, impurities := 30/100 }

/-- The amount of pure ghee added (in kg) -/
def addedPureGhee : Rat := 20

/-- Theorem stating the original quantity of blended ghee -/
theorem original_ghee_quantity : 
  ∃ (x : Rat), 
    (originalComposition.pure * x + addedPureGhee = desiredComposition.pure * (x + addedPureGhee)) ∧
    (originalComposition.vanaspati * x = desiredComposition.vanaspati * (x + addedPureGhee)) ∧
    x = 220 := by
  sorry

end original_ghee_quantity_l518_51856


namespace fraction_sum_equals_point_three_l518_51838

theorem fraction_sum_equals_point_three :
  5 / 50 + 4 / 40 + 6 / 60 = 0.3 := by
  sorry

end fraction_sum_equals_point_three_l518_51838


namespace discount_savings_difference_l518_51880

def shoe_price : ℕ := 50
def discount_a_percent : ℕ := 40
def discount_b_amount : ℕ := 15

def cost_with_discount_a : ℕ := shoe_price + (shoe_price - (shoe_price * discount_a_percent / 100))
def cost_with_discount_b : ℕ := shoe_price + (shoe_price - discount_b_amount)

theorem discount_savings_difference : 
  cost_with_discount_b - cost_with_discount_a = 5 := by
  sorry

end discount_savings_difference_l518_51880


namespace number_of_proper_subsets_of_P_l518_51882

def M : Finset ℤ := {-1, 1, 2, 3, 4, 5}
def N : Finset ℤ := {1, 2, 4}
def P : Finset ℤ := M ∩ N

theorem number_of_proper_subsets_of_P : (Finset.powerset P).card - 1 = 7 := by
  sorry

end number_of_proper_subsets_of_P_l518_51882


namespace log_sum_equals_four_l518_51833

theorem log_sum_equals_four : Real.log 64 / Real.log 8 + Real.log 81 / Real.log 9 = 4 := by
  sorry

end log_sum_equals_four_l518_51833


namespace towel_shrinkage_l518_51885

/-- Given a rectangular towel that loses 20% of its length and has a total area
    decrease of 27.999999999999993%, the percentage decrease in breadth is 10%. -/
theorem towel_shrinkage (L B : ℝ) (L' B' : ℝ) : 
  L' = 0.8 * L →
  L' * B' = 0.72 * (L * B) →
  B' = 0.9 * B :=
by sorry

end towel_shrinkage_l518_51885


namespace x_less_than_y_l518_51843

theorem x_less_than_y : 123456789 * 123456786 < 123456788 * 123456787 := by
  sorry

end x_less_than_y_l518_51843


namespace floor_abs_neq_abs_floor_exists_floor_diff_lt_floor_eq_implies_diff_lt_one_floor_inequality_solution_set_l518_51890

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

-- Proposition A (negation)
theorem floor_abs_neq_abs_floor : ∃ x : ℝ, floor (|x|) ≠ |floor x| :=
sorry

-- Proposition B
theorem exists_floor_diff_lt : ∃ x y : ℝ, floor (x - y) < floor x - floor y :=
sorry

-- Proposition C
theorem floor_eq_implies_diff_lt_one :
  ∀ x y : ℝ, floor x = floor y → x - y < 1 :=
sorry

-- Proposition D
theorem floor_inequality_solution_set :
  {x : ℝ | 2 * (floor x)^2 - floor x - 3 ≥ 0} = {x : ℝ | x < 0 ∨ x ≥ 2} :=
sorry

end floor_abs_neq_abs_floor_exists_floor_diff_lt_floor_eq_implies_diff_lt_one_floor_inequality_solution_set_l518_51890


namespace reciprocal_sum_and_product_l518_51852

theorem reciprocal_sum_and_product (x y : ℝ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : x + y = 12) (h4 : x * y = 32) : 
  1 / x + 1 / y = 3 / 8 ∧ 1 / (x * y) = 1 / 32 := by
  sorry

end reciprocal_sum_and_product_l518_51852


namespace congress_room_arrangement_l518_51886

/-- A type representing delegates -/
def Delegate : Type := ℕ

/-- A relation representing the ability to communicate directly -/
def CanCommunicate : Delegate → Delegate → Prop := sorry

/-- The total number of delegates -/
def totalDelegates : ℕ := 1000

theorem congress_room_arrangement 
  (delegates : Finset Delegate) 
  (h_count : delegates.card = totalDelegates)
  (h_communication : ∀ (a b c : Delegate), a ∈ delegates → b ∈ delegates → c ∈ delegates → 
    (CanCommunicate a b ∨ CanCommunicate b c ∨ CanCommunicate a c)) :
  ∃ (pairs : List (Delegate × Delegate)), 
    (∀ (pair : Delegate × Delegate), pair ∈ pairs → CanCommunicate pair.1 pair.2) ∧ 
    (pairs.length = totalDelegates / 2) ∧
    (∀ (d : Delegate), d ∈ delegates ↔ (∃ (pair : Delegate × Delegate), pair ∈ pairs ∧ (d = pair.1 ∨ d = pair.2))) :=
sorry

end congress_room_arrangement_l518_51886


namespace average_difference_l518_51822

def num_students : ℕ := 120
def num_teachers : ℕ := 5
def class_sizes : List ℕ := [40, 30, 20, 15, 15]

def t : ℚ := (num_students : ℚ) / num_teachers

def s : ℚ := (List.sum (List.map (λ x => x * x) class_sizes) : ℚ) / num_students

theorem average_difference : t - s = -3.92 := by sorry

end average_difference_l518_51822


namespace elevator_weight_problem_l518_51814

/-- Given the initial conditions and new averages after each person enters,
    prove that the weights of X, Y, and Z are 195 lbs, 141 lbs, and 126 lbs respectively. -/
theorem elevator_weight_problem (initial_people : Nat) (initial_avg : ℝ)
    (avg_after_X : ℝ) (avg_after_Y : ℝ) (avg_after_Z : ℝ)
    (h1 : initial_people = 6)
    (h2 : initial_avg = 160)
    (h3 : avg_after_X = 165)
    (h4 : avg_after_Y = 162)
    (h5 : avg_after_Z = 158) :
    ∃ (X Y Z : ℝ),
      X = 195 ∧
      Y = 141 ∧
      Z = 126 ∧
      (initial_people : ℝ) * initial_avg + X = (initial_people + 1 : ℝ) * avg_after_X ∧
      ((initial_people + 1 : ℝ) * avg_after_X + Y = (initial_people + 2 : ℝ) * avg_after_Y) ∧
      ((initial_people + 2 : ℝ) * avg_after_Y + Z = (initial_people + 3 : ℝ) * avg_after_Z) :=
by sorry

end elevator_weight_problem_l518_51814

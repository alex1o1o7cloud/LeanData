import Mathlib

namespace NUMINAMATH_CALUDE_question_ratio_l4040_404024

/-- Represents the number of questions submitted by each person -/
structure QuestionSubmission where
  rajat : ℕ
  vikas : ℕ
  abhishek : ℕ

/-- The total number of questions submitted -/
def total_questions : ℕ := 24

/-- Theorem stating the ratio of questions submitted -/
theorem question_ratio (qs : QuestionSubmission) 
  (h1 : qs.rajat + qs.vikas + qs.abhishek = total_questions)
  (h2 : qs.vikas = 6) :
  ∃ (r a : ℕ), r = qs.rajat ∧ a = qs.abhishek ∧ r + a = 18 :=
by sorry

end NUMINAMATH_CALUDE_question_ratio_l4040_404024


namespace NUMINAMATH_CALUDE_sequoia_maple_height_difference_l4040_404052

/-- Represents the height of a tree in feet and quarters of a foot -/
structure TreeHeight where
  feet : ℕ
  quarters : Fin 4

/-- Converts a TreeHeight to a rational number -/
def treeHeightToRational (h : TreeHeight) : ℚ :=
  h.feet + h.quarters.val / 4

/-- The height of the maple tree -/
def mapleHeight : TreeHeight := ⟨13, 3⟩

/-- The height of the sequoia -/
def sequoiaHeight : TreeHeight := ⟨20, 2⟩

theorem sequoia_maple_height_difference :
  treeHeightToRational sequoiaHeight - treeHeightToRational mapleHeight = 27 / 4 := by
  sorry

#eval treeHeightToRational sequoiaHeight - treeHeightToRational mapleHeight

end NUMINAMATH_CALUDE_sequoia_maple_height_difference_l4040_404052


namespace NUMINAMATH_CALUDE_prob_both_heads_is_one_fourth_l4040_404044

/-- A coin of uniform density -/
structure Coin :=
  (side : Bool)

/-- The sample space of tossing two coins -/
def TwoCoins := Coin × Coin

/-- The event where both coins land heads up -/
def BothHeads (outcome : TwoCoins) : Prop :=
  outcome.1.side ∧ outcome.2.side

/-- The probability measure on the sample space -/
axiom prob : Set TwoCoins → ℝ

/-- The probability measure satisfies basic properties -/
axiom prob_nonneg : ∀ A : Set TwoCoins, 0 ≤ prob A
axiom prob_le_one : ∀ A : Set TwoCoins, prob A ≤ 1
axiom prob_additive : ∀ A B : Set TwoCoins, A ∩ B = ∅ → prob (A ∪ B) = prob A + prob B

/-- The probability of each outcome is equal due to uniform density -/
axiom prob_uniform : ∀ x y : TwoCoins, prob {x} = prob {y}

theorem prob_both_heads_is_one_fourth :
  prob {x : TwoCoins | BothHeads x} = 1/4 := by
  sorry

#check prob_both_heads_is_one_fourth

end NUMINAMATH_CALUDE_prob_both_heads_is_one_fourth_l4040_404044


namespace NUMINAMATH_CALUDE_closest_fraction_to_two_thirds_l4040_404083

theorem closest_fraction_to_two_thirds :
  let fractions : List ℚ := [4/7, 9/14, 20/31, 61/95, 73/110]
  let target : ℚ := 2/3
  let differences := fractions.map (fun x => |x - target|)
  differences.minimum? = some |73/110 - 2/3| :=
by sorry

end NUMINAMATH_CALUDE_closest_fraction_to_two_thirds_l4040_404083


namespace NUMINAMATH_CALUDE_titu_andreescu_inequality_l4040_404080

theorem titu_andreescu_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^5 - a^2 + 3) * (b^5 - b^2 + 3) * (c^5 - c^2 + 3) ≥ (a + b + c)^3 := by
  sorry

end NUMINAMATH_CALUDE_titu_andreescu_inequality_l4040_404080


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l4040_404066

/-- Expresses the sum of three repeating decimals as a rational number -/
theorem repeating_decimal_sum : 
  (2 : ℚ) / 9 + (3 : ℚ) / 99 + (4 : ℚ) / 9999 = 283 / 11111 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l4040_404066


namespace NUMINAMATH_CALUDE_finitely_many_odd_divisors_l4040_404036

theorem finitely_many_odd_divisors (k : ℕ+) :
  (∃ c : ℕ, k + 1 = 2^c) ↔
  (∃ S : Finset ℕ, ∀ n : ℕ, n % 2 = 1 → (n ∣ k^n + 1) → n ∈ S) :=
by sorry

end NUMINAMATH_CALUDE_finitely_many_odd_divisors_l4040_404036


namespace NUMINAMATH_CALUDE_perpendicular_tangents_ratio_l4040_404062

/-- Given a line ax - by - 2 = 0 and a curve y = x³ with perpendicular tangents at point P(1,1),
    the value of b/a is -3. -/
theorem perpendicular_tangents_ratio (a b : ℝ) : 
  (∃ (x y : ℝ), a * x - b * y - 2 = 0 ∧ y = x^3) →  -- Line and curve equations
  (∃ (x y : ℝ), x = 1 ∧ y = 1 ∧ 
    (∀ (t : ℝ), a * t - b * (t^3) - 2 = 0 ↔ a * (x - t) + b * (y - t^3) = 0)) →  -- Perpendicular tangents at P(1,1)
  b / a = -3 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_tangents_ratio_l4040_404062


namespace NUMINAMATH_CALUDE_optimal_meeting_time_l4040_404038

/-- The optimal meeting time for a pedestrian and cyclist on a circular path -/
theorem optimal_meeting_time 
  (pedestrian_speed : ℝ) 
  (cyclist_speed : ℝ) 
  (path_length : ℝ) 
  (walked_distance : ℝ) 
  (remaining_distance : ℝ) 
  (h1 : pedestrian_speed = 6.5)
  (h2 : cyclist_speed = 20)
  (h3 : path_length = 4 * Real.pi)
  (h4 : walked_distance = 6.5)
  (h5 : remaining_distance = 4 * Real.pi - 6.5)
  (h6 : walked_distance = pedestrian_speed * 1) -- 1 hour of walking
  : ∃ (t : ℝ), t = (155 - 28 * Real.pi) / 172 ∧ 
    t = min (remaining_distance / (pedestrian_speed + cyclist_speed))
            ((path_length - walked_distance) / (pedestrian_speed + cyclist_speed)) := by
  sorry

end NUMINAMATH_CALUDE_optimal_meeting_time_l4040_404038


namespace NUMINAMATH_CALUDE_scientific_notation_proof_l4040_404067

theorem scientific_notation_proof : 
  284000000 = 2.84 * (10 ^ 8) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_proof_l4040_404067


namespace NUMINAMATH_CALUDE_probability_log_integer_l4040_404023

def S : Set ℕ := {n | ∃ k : ℕ, 1 ≤ k ∧ k ≤ 20 ∧ n = 3^k}

def is_valid_pair (a b : ℕ) : Prop :=
  a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ ∃ k : ℕ, b = a^k

def total_pairs : ℕ := Nat.choose 20 2

def valid_pairs : ℕ := 48

theorem probability_log_integer :
  (valid_pairs : ℚ) / total_pairs = 24 / 95 := by sorry

end NUMINAMATH_CALUDE_probability_log_integer_l4040_404023


namespace NUMINAMATH_CALUDE_ball_bounce_distance_l4040_404057

/-- The sum of an infinite geometric series with first term h and common ratio 0.8 is equal to 5h -/
theorem ball_bounce_distance (h : ℝ) (h_pos : h > 0) : 
  (∑' n, h * (0.8 ^ n)) = 5 * h := by sorry

end NUMINAMATH_CALUDE_ball_bounce_distance_l4040_404057


namespace NUMINAMATH_CALUDE_max_wins_l4040_404035

/-- Given the ratio of Chloe's wins to Max's wins and Chloe's total wins,
    calculate Max's wins. -/
theorem max_wins (chloe_wins : ℕ) (chloe_ratio : ℕ) (max_ratio : ℕ) 
    (h1 : chloe_wins = 24)
    (h2 : chloe_ratio = 8)
    (h3 : max_ratio = 3) :
  chloe_wins * max_ratio / chloe_ratio = 9 := by
  sorry

#check max_wins

end NUMINAMATH_CALUDE_max_wins_l4040_404035


namespace NUMINAMATH_CALUDE_det_special_matrix_is_zero_l4040_404039

open Real Matrix

theorem det_special_matrix_is_zero (θ φ : ℝ) : 
  det !![0, cos θ, sin θ; -cos θ, 0, cos φ; -sin θ, -cos φ, 0] = 0 := by
  sorry

end NUMINAMATH_CALUDE_det_special_matrix_is_zero_l4040_404039


namespace NUMINAMATH_CALUDE_rectangle_existence_l4040_404027

theorem rectangle_existence : ∃ (x y : ℝ), 
  (2 * (x + y) = 2 * (2 + 1) * 2) ∧ 
  (x * y = 2 * 1 * 2) ∧ 
  (x > 0) ∧ (y > 0) ∧
  (x = 3 + Real.sqrt 5) ∧ (y = 3 - Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_existence_l4040_404027


namespace NUMINAMATH_CALUDE_option_b_more_cost_effective_l4040_404086

/-- Cost function for Option A -/
def cost_a (x : ℝ) : ℝ := 60 + 18 * x

/-- Cost function for Option B -/
def cost_b (x : ℝ) : ℝ := 150 + 15 * x

/-- Theorem stating that Option B is more cost-effective for 40 kg of blueberries -/
theorem option_b_more_cost_effective :
  cost_b 40 < cost_a 40 := by sorry

end NUMINAMATH_CALUDE_option_b_more_cost_effective_l4040_404086


namespace NUMINAMATH_CALUDE_range_of_a_l4040_404013

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, a * x^2 + a * x + 1 < 0) → a ∈ Set.Iio 0 ∪ Set.Ioi 4 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l4040_404013


namespace NUMINAMATH_CALUDE_upper_limit_correct_l4040_404096

def digit_product (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) * digit_product (n / 10)

def upper_limit : ℕ := 7533

theorem upper_limit_correct :
  ∀ h : ℕ, h > 0 ∧ digit_product h = 210 → h < upper_limit :=
by sorry

end NUMINAMATH_CALUDE_upper_limit_correct_l4040_404096


namespace NUMINAMATH_CALUDE_marks_score_l4040_404017

theorem marks_score (highest_score : ℕ) (score_range : ℕ) (marks_score : ℕ) :
  highest_score = 98 →
  score_range = 75 →
  marks_score = 2 * (highest_score - score_range) →
  marks_score = 46 :=
by
  sorry

end NUMINAMATH_CALUDE_marks_score_l4040_404017


namespace NUMINAMATH_CALUDE_inequality_solution_set_l4040_404012

theorem inequality_solution_set (x : ℝ) : 
  (2 * x^2 / (x + 2) ≥ 3 / (x - 2) + 6 / 4) ↔ 
  (x < -2 ∨ (-2 < x ∧ x < (1 - Real.sqrt 129) / 8) ∨ 
   (2 < x ∧ x < 3) ∨ 
   ((1 + Real.sqrt 129) / 8 < x)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l4040_404012


namespace NUMINAMATH_CALUDE_weight_of_replaced_person_l4040_404093

theorem weight_of_replaced_person 
  (n : ℕ) 
  (original_total : ℝ) 
  (new_weight : ℝ) 
  (average_increase : ℝ) 
  (h1 : n = 10)
  (h2 : new_weight = 75)
  (h3 : average_increase = 3)
  : 
  (original_total + new_weight - (original_total / n + average_increase * n)) = 45 :=
by sorry

end NUMINAMATH_CALUDE_weight_of_replaced_person_l4040_404093


namespace NUMINAMATH_CALUDE_triangle_covering_polygon_l4040_404053

-- Define the types for points and polygons
variable (Point : Type) [NormedAddCommGroup Point] [InnerProductSpace ℝ Point]
variable (Polygon : Type)
variable (Triangle : Type)

-- Define the properties and relations
variable (covers : Triangle → Polygon → Prop)
variable (congruent : Triangle → Triangle → Prop)
variable (has_parallel_side : Triangle → Polygon → Prop)

-- State the theorem
theorem triangle_covering_polygon
  (ABC : Triangle) (M : Polygon) 
  (h_covers : covers ABC M) :
  ∃ (DEF : Triangle), 
    congruent DEF ABC ∧ 
    covers DEF M ∧ 
    has_parallel_side DEF M :=
sorry

end NUMINAMATH_CALUDE_triangle_covering_polygon_l4040_404053


namespace NUMINAMATH_CALUDE_sophie_widget_production_l4040_404073

/-- Sophie's widget production problem -/
theorem sophie_widget_production 
  (w t : ℕ) -- w: widgets per hour, t: hours worked on Wednesday
  (h1 : w = 3 * t) -- condition that w = 3t
  : w * t - (w + 5) * (t - 3) = 4 * t + 15 := by
  sorry

end NUMINAMATH_CALUDE_sophie_widget_production_l4040_404073


namespace NUMINAMATH_CALUDE_coffee_stock_problem_l4040_404061

/-- Proves that given the conditions of the coffee stock problem, 
    the percentage of the initial stock that was decaffeinated is 20%. -/
theorem coffee_stock_problem (initial_stock : ℝ) (additional_purchase : ℝ) 
  (decaf_percent_new : ℝ) (total_decaf_percent : ℝ) :
  initial_stock = 400 →
  additional_purchase = 100 →
  decaf_percent_new = 60 →
  total_decaf_percent = 28.000000000000004 →
  (initial_stock * (20 / 100) + additional_purchase * (decaf_percent_new / 100)) / 
  (initial_stock + additional_purchase) * 100 = total_decaf_percent :=
by
  sorry

end NUMINAMATH_CALUDE_coffee_stock_problem_l4040_404061


namespace NUMINAMATH_CALUDE_shaded_area_is_correct_l4040_404074

/-- A square and a right triangle with equal height -/
structure GeometricSetup where
  /-- Height of both the square and the triangle -/
  height : ℝ
  /-- Base length of both the square and the triangle -/
  base : ℝ
  /-- The lower right vertex of the square and lower left vertex of the triangle -/
  intersection : ℝ × ℝ
  /-- Assertion that the height equals the base -/
  height_eq_base : height = base
  /-- Assertion that the intersection point is at (15, 0) -/
  intersection_is_fifteen : intersection = (15, 0)
  /-- Assertion that the base length is 15 -/
  base_is_fifteen : base = 15

/-- The area of the shaded region -/
def shaded_area (setup : GeometricSetup) : ℝ := 168.75

/-- Theorem stating that the shaded area is 168.75 square units -/
theorem shaded_area_is_correct (setup : GeometricSetup) : 
  shaded_area setup = 168.75 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_is_correct_l4040_404074


namespace NUMINAMATH_CALUDE_problem_solution_l4040_404045

theorem problem_solution :
  ∀ (a b c d : ℝ),
    1000 * a = 85^2 - 15^2 →
    5 * a + 2 * b = 41 →
    (-3)^2 + 6 * (-3) + c = 0 →
    d^2 = (5 - c)^2 + (4 - 1)^2 →
    a = 7 ∧ b = 3 ∧ c = 9 ∧ d = 5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l4040_404045


namespace NUMINAMATH_CALUDE_gcd_lcm_888_1147_l4040_404046

theorem gcd_lcm_888_1147 : 
  (Nat.gcd 888 1147 = 37) ∧ (Nat.lcm 888 1147 = 27528) := by sorry

end NUMINAMATH_CALUDE_gcd_lcm_888_1147_l4040_404046


namespace NUMINAMATH_CALUDE_max_min_product_l4040_404001

theorem max_min_product (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a + b + c = 12) (h_prod_sum : a * b + b * c + c * a = 35) :
  ∃ (m : ℝ), m = min (a * b) (min (b * c) (c * a)) ∧ m ≤ 3 ∧
  ∀ (m' : ℝ), m' = min (a * b) (min (b * c) (c * a)) → m' ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_max_min_product_l4040_404001


namespace NUMINAMATH_CALUDE_robin_gum_pieces_l4040_404082

/-- The number of gum packages Robin has -/
def num_packages : ℕ := 135

/-- The number of pieces in each package of gum -/
def pieces_per_package : ℕ := 46

/-- The total number of gum pieces Robin has -/
def total_pieces : ℕ := num_packages * pieces_per_package

theorem robin_gum_pieces : total_pieces = 6210 := by
  sorry

end NUMINAMATH_CALUDE_robin_gum_pieces_l4040_404082


namespace NUMINAMATH_CALUDE_vector_properties_l4040_404043

open Real

/-- Given vectors satisfying certain conditions, prove parallelism and angle between vectors -/
theorem vector_properties (a b c : ℝ × ℝ) : 
  (3 • a - 2 • b = (2, 6)) → 
  (a + 2 • b = (6, 2)) → 
  (c = (1, 1)) → 
  (∃ (k : ℝ), a = k • c) ∧ 
  (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_vector_properties_l4040_404043


namespace NUMINAMATH_CALUDE_number_of_boys_l4040_404065

theorem number_of_boys (total : ℕ) (boys : ℕ) : 
  total = 900 →
  (total - boys : ℚ) = (boys : ℚ) * (total : ℚ) / 100 →
  boys = 90 := by
sorry

end NUMINAMATH_CALUDE_number_of_boys_l4040_404065


namespace NUMINAMATH_CALUDE_midpoint_value_part1_midpoint_value_part2_l4040_404028

-- Definition of midpoint value
def is_midpoint_value (a b : ℝ) : Prop := a^2 - b > 0

-- Part 1
theorem midpoint_value_part1 : 
  is_midpoint_value 4 3 ∧ ∀ x, x^2 - 8*x + 3 = 0 ↔ x^2 - 2*4*x + 3 = 0 :=
sorry

-- Part 2
theorem midpoint_value_part2 (m n : ℝ) : 
  (is_midpoint_value 3 n ∧ 
   ∀ x, x^2 - m*x + n = 0 ↔ x^2 - 2*3*x + n = 0 ∧
   (n^2 - m*n + n = 0)) →
  (n = 0 ∨ n = 5) :=
sorry

end NUMINAMATH_CALUDE_midpoint_value_part1_midpoint_value_part2_l4040_404028


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l4040_404008

theorem least_addition_for_divisibility : 
  ∃ (n : ℕ), n = 256 ∧ 
  (∀ m : ℕ, (1019 + m) % 25 = 0 ∧ (1019 + m) % 17 = 0 → m ≥ n) ∧
  (1019 + n) % 25 = 0 ∧ (1019 + n) % 17 = 0 :=
by sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l4040_404008


namespace NUMINAMATH_CALUDE_weight_difference_proof_l4040_404049

/-- Proves the difference between the average weight of two departing students and Joe's weight --/
theorem weight_difference_proof 
  (n : ℕ) -- number of students in the original group
  (initial_avg : ℝ) -- initial average weight
  (joe_weight : ℝ) -- Joe's weight
  (new_avg : ℝ) -- new average weight after Joe joins
  (final_avg : ℝ) -- final average weight after two students leave
  (h1 : initial_avg = 30)
  (h2 : joe_weight = 43)
  (h3 : new_avg = initial_avg + 1)
  (h4 : final_avg = initial_avg)
  (h5 : (n * initial_avg + joe_weight) / (n + 1) = new_avg)
  (h6 : ((n + 1) * new_avg - 2 * final_avg) / (n - 1) = final_avg) :
  (((n + 1) * new_avg - n * final_avg) / 2) - joe_weight = -6.5 := by
  sorry

end NUMINAMATH_CALUDE_weight_difference_proof_l4040_404049


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l4040_404030

/-- Given a geometric sequence {a_n} with a₁ = 3 and a₁ + a₃ + a₅ = 21, 
    prove that a₃ + a₅ + a₇ = 42 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) : 
  (∀ n, a (n + 1) = a n * q) →  -- Geometric sequence definition
  a 1 = 3 →                     -- First term condition
  a 1 + a 3 + a 5 = 21 →        -- Sum of odd terms condition
  a 3 + a 5 + a 7 = 42 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l4040_404030


namespace NUMINAMATH_CALUDE_platform_length_l4040_404094

/-- The length of a platform given a train's speed, length, and crossing time -/
theorem platform_length (train_speed : ℝ) (train_length : ℝ) (crossing_time : ℝ) :
  train_speed = 72 →
  train_length = 230.0384 →
  crossing_time = 24 →
  (train_speed * 1000 / 3600) * crossing_time - train_length = 249.9616 := by
  sorry

end NUMINAMATH_CALUDE_platform_length_l4040_404094


namespace NUMINAMATH_CALUDE_inequality_proof_l4040_404070

def M : Set ℝ := {x | -2 < |x - 1| - |x + 2| ∧ |x - 1| - |x + 2| < 0}

theorem inequality_proof (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) :
  (|1/3 * a + 1/6 * b| < 1/4) ∧ (|1 - 4*a*b| > 2 * |a - b|) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4040_404070


namespace NUMINAMATH_CALUDE_max_distance_AB_l4040_404051

-- Define the ellipse C
def C (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define the point M
def M : ℝ × ℝ := (2, 0)

-- Define the line passing through M and intersecting C at A and B
def line_through_M (k : ℝ) (x : ℝ) : ℝ := k * (x - 2)

-- Define the condition for A and B being on both the line and the ellipse
def A_B_on_line_and_C (k x y : ℝ) : Prop :=
  C x y ∧ y = line_through_M k x

-- Define the vector addition condition
def vector_addition_condition (xA yA xB yB xP yP t : ℝ) : Prop :=
  xA + xB = t * xP ∧ yA + yB = t * yP

-- Main theorem
theorem max_distance_AB :
  ∀ (k xA yA xB yB xP yP t : ℝ),
    A_B_on_line_and_C k xA yA →
    A_B_on_line_and_C k xB yB →
    C xP yP →
    vector_addition_condition xA yA xB yB xP yP t →
    2 * Real.sqrt 6 / 3 < t →
    t < 2 →
    ∃ (max_dist : ℝ), max_dist = 2 * Real.sqrt 5 / 3 ∧
      ((xA - xB)^2 + (yA - yB)^2)^(1/2 : ℝ) ≤ max_dist :=
by sorry

end NUMINAMATH_CALUDE_max_distance_AB_l4040_404051


namespace NUMINAMATH_CALUDE_probability_multiple_3_or_5_l4040_404010

def is_multiple_of_3_or_5 (n : ℕ) : Bool :=
  n % 3 = 0 || n % 5 = 0

def count_multiples (max : ℕ) : ℕ :=
  (List.range max).filter is_multiple_of_3_or_5 |>.length

theorem probability_multiple_3_or_5 :
  (count_multiples 20 : ℚ) / 20 = 9 / 20 := by
  sorry

end NUMINAMATH_CALUDE_probability_multiple_3_or_5_l4040_404010


namespace NUMINAMATH_CALUDE_min_value_theorem_l4040_404075

theorem min_value_theorem (a b c : ℝ) (h1 : c > 0) (h2 : a ≠ 0) (h3 : b ≠ 0)
  (h4 : 4 * a^2 - 2 * a * b + 4 * b^2 - c = 0)
  (h5 : ∀ x y : ℝ, x ≠ 0 → y ≠ 0 → 4 * x^2 - 2 * x * y + 4 * y^2 - c = 0 →
    (2 * x + y)^2 ≤ (2 * a + b)^2) :
  ∀ x y z : ℝ, x ≠ 0 → y ≠ 0 → z > 0 →
    4 * x^2 - 2 * x * y + 4 * y^2 - z = 0 →
    3 / a - 4 / b + 5 / c ≤ 3 / x - 4 / y + 5 / z :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l4040_404075


namespace NUMINAMATH_CALUDE_log_a_equals_three_l4040_404085

theorem log_a_equals_three (a : ℝ) (h1 : a > 0) (h2 : a^(2/3) = 4/9) : 
  Real.log a / Real.log (2/3) = 3 := by
  sorry

end NUMINAMATH_CALUDE_log_a_equals_three_l4040_404085


namespace NUMINAMATH_CALUDE_expression_equals_five_halves_l4040_404037

theorem expression_equals_five_halves :
  Real.sqrt 12 - 2 * Real.cos (π / 6) + |Real.sqrt 3 - 2| + 2^(-1 : ℤ) = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_five_halves_l4040_404037


namespace NUMINAMATH_CALUDE_science_club_membership_l4040_404018

theorem science_club_membership (total : ℕ) (chem : ℕ) (bio : ℕ) (both : ℕ) 
  (h1 : total = 80)
  (h2 : chem = 48)
  (h3 : bio = 40)
  (h4 : both = 25) :
  total - (chem + bio - both) = 17 := by
  sorry

end NUMINAMATH_CALUDE_science_club_membership_l4040_404018


namespace NUMINAMATH_CALUDE_average_wage_is_21_l4040_404015

def male_workers : ℕ := 20
def female_workers : ℕ := 15
def child_workers : ℕ := 5

def male_wage : ℕ := 25
def female_wage : ℕ := 20
def child_wage : ℕ := 8

def total_workers : ℕ := male_workers + female_workers + child_workers

def total_wages : ℕ := male_workers * male_wage + female_workers * female_wage + child_workers * child_wage

theorem average_wage_is_21 : total_wages / total_workers = 21 := by
  sorry

end NUMINAMATH_CALUDE_average_wage_is_21_l4040_404015


namespace NUMINAMATH_CALUDE_tan_alpha_three_implies_expression_equals_two_l4040_404076

theorem tan_alpha_three_implies_expression_equals_two (α : Real) 
  (h : Real.tan α = 3) : 
  (Real.sin (α - π) + Real.cos (π - α)) / 
  (Real.sin (π / 2 - α) + Real.cos (π / 2 + α)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_three_implies_expression_equals_two_l4040_404076


namespace NUMINAMATH_CALUDE_custom_op_nested_l4040_404060

/-- Custom binary operation ⊗ -/
def custom_op (x y : ℝ) : ℝ := x^3 - y^2 + x

/-- Theorem stating the result of k ⊗ (k ⊗ k) -/
theorem custom_op_nested (k : ℝ) : custom_op k (custom_op k k) = -k^6 + 2*k^5 - 3*k^4 + 3*k^3 - k^2 + 2*k := by
  sorry

end NUMINAMATH_CALUDE_custom_op_nested_l4040_404060


namespace NUMINAMATH_CALUDE_unit_cost_decrease_l4040_404022

/-- Regression equation for unit product cost -/
def regression_equation (x : ℝ) : ℝ := 356 - 1.5 * x

/-- Theorem stating the relationship between output and unit product cost -/
theorem unit_cost_decrease (x : ℝ) :
  regression_equation (x + 1) = regression_equation x - 1.5 := by
  sorry

end NUMINAMATH_CALUDE_unit_cost_decrease_l4040_404022


namespace NUMINAMATH_CALUDE_smallest_perimeter_isosceles_triangle_l4040_404019

/-- Triangle with positive integer side lengths --/
structure IsoscelesTriangle where
  pq : ℕ+
  qr : ℕ+

/-- Angle bisector intersection point --/
structure AngleBisectorIntersection where
  qj : ℝ

/-- Theorem statement for the smallest perimeter of the isosceles triangle --/
theorem smallest_perimeter_isosceles_triangle
  (t : IsoscelesTriangle)
  (j : AngleBisectorIntersection)
  (h1 : j.qj = 10) :
  2 * (t.pq + t.qr) ≥ 416 ∧
  ∃ (t' : IsoscelesTriangle), 2 * (t'.pq + t'.qr) = 416 := by
  sorry

end NUMINAMATH_CALUDE_smallest_perimeter_isosceles_triangle_l4040_404019


namespace NUMINAMATH_CALUDE_shelf_filling_problem_l4040_404088

/-- Represents the shelf filling problem with biology and geography books -/
theorem shelf_filling_problem 
  (B G P Q K : ℕ) 
  (h_distinct : B ≠ G ∧ B ≠ P ∧ B ≠ Q ∧ B ≠ K ∧ 
                G ≠ P ∧ G ≠ Q ∧ G ≠ K ∧ 
                P ≠ Q ∧ P ≠ K ∧ 
                Q ≠ K)
  (h_positive : B > 0 ∧ G > 0 ∧ P > 0 ∧ Q > 0 ∧ K > 0)
  (h_fill1 : ∃ (a : ℚ), a > 0 ∧ B * a + G * (2 * a) = K * a)
  (h_fill2 : ∃ (a : ℚ), a > 0 ∧ P * a + Q * (2 * a) = K * a) :
  K = B + 2 * G :=
sorry

end NUMINAMATH_CALUDE_shelf_filling_problem_l4040_404088


namespace NUMINAMATH_CALUDE_deal_or_no_deal_probability_l4040_404055

/-- The total number of boxes in the game -/
def total_boxes : ℕ := 30

/-- The number of boxes containing at least $50,000 -/
def high_value_boxes : ℕ := 9

/-- The target probability (50%) expressed as a fraction -/
def target_probability : ℚ := 1/2

/-- The minimum number of boxes that need to be eliminated -/
def boxes_to_eliminate : ℕ := 12

theorem deal_or_no_deal_probability :
  boxes_to_eliminate = total_boxes - 2 * high_value_boxes :=
by sorry

end NUMINAMATH_CALUDE_deal_or_no_deal_probability_l4040_404055


namespace NUMINAMATH_CALUDE_second_quarter_profit_l4040_404078

def annual_profit : ℕ := 8000
def first_quarter_profit : ℕ := 1500
def third_quarter_profit : ℕ := 3000
def fourth_quarter_profit : ℕ := 2000

theorem second_quarter_profit :
  annual_profit - (first_quarter_profit + third_quarter_profit + fourth_quarter_profit) = 1500 :=
by sorry

end NUMINAMATH_CALUDE_second_quarter_profit_l4040_404078


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l4040_404031

theorem complex_magnitude_problem (z : ℂ) (h : (1 - Complex.I) * z = 1 + Complex.I) : 
  Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l4040_404031


namespace NUMINAMATH_CALUDE_modular_inverse_57_mod_59_l4040_404087

theorem modular_inverse_57_mod_59 : ∃ x : ℕ, x < 59 ∧ (57 * x) % 59 = 1 :=
by
  use 29
  sorry

end NUMINAMATH_CALUDE_modular_inverse_57_mod_59_l4040_404087


namespace NUMINAMATH_CALUDE_geometric_series_product_l4040_404029

theorem geometric_series_product (x : ℝ) : x = 4 ↔ 
  (∑' n, (1/2)^n) * (∑' n, (-1/2)^n) = ∑' n, (1/x)^n :=
by sorry

end NUMINAMATH_CALUDE_geometric_series_product_l4040_404029


namespace NUMINAMATH_CALUDE_quarters_percentage_correct_l4040_404007

/-- Calculates the percentage of total value in quarters given the number of dimes, quarters, and half-dollars. -/
def percentInQuarters (dimes quarters halfDollars : ℕ) : ℚ :=
  let dimeValue := 10 * dimes
  let quarterValue := 25 * quarters
  let halfDollarValue := 50 * halfDollars
  let totalValue := dimeValue + quarterValue + halfDollarValue
  (quarterValue : ℚ) / totalValue * 100

/-- Theorem stating that the percentage of total value in quarters is approximately 45.45% -/
theorem quarters_percentage_correct :
  ∃ ε > 0, abs (percentInQuarters 40 30 10 - 45.45) < ε :=
sorry

end NUMINAMATH_CALUDE_quarters_percentage_correct_l4040_404007


namespace NUMINAMATH_CALUDE_square_area_increase_l4040_404079

theorem square_area_increase (s : ℝ) (h : s > 0) : 
  let new_side := 1.25 * s
  let original_area := s^2
  let new_area := new_side^2
  (new_area - original_area) / original_area = 0.5625 := by
sorry

end NUMINAMATH_CALUDE_square_area_increase_l4040_404079


namespace NUMINAMATH_CALUDE_max_subset_size_2021_l4040_404063

/-- Given a natural number N, returns the maximum size of a subset A of {1, ..., N}
    such that any two numbers in A are neither coprime nor have a divisibility relationship. -/
def maxSubsetSize (N : ℕ) : ℕ :=
  sorry

/-- The maximum subset size for N = 2021 is 505. -/
theorem max_subset_size_2021 : maxSubsetSize 2021 = 505 := by
  sorry

end NUMINAMATH_CALUDE_max_subset_size_2021_l4040_404063


namespace NUMINAMATH_CALUDE_geometric_series_problem_l4040_404050

theorem geometric_series_problem (b₁ q : ℝ) (h_decrease : |q| < 1) : 
  (b₁ / (1 - q^2) = 2 + b₁ * q / (1 - q^2)) →
  (b₁^2 / (1 - q^4) - b₁^2 * q^2 / (1 - q^4) = 36/5) →
  (b₁ = 3 ∧ q = 1/2) := by
sorry

end NUMINAMATH_CALUDE_geometric_series_problem_l4040_404050


namespace NUMINAMATH_CALUDE_min_value_of_f_inequality_abc_l4040_404048

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| + |x - 5|

-- Theorem for the minimum value of f
theorem min_value_of_f : ∃ m : ℝ, ∀ x : ℝ, f x ≥ m ∧ ∃ y : ℝ, f y = m :=
sorry

-- Theorem for the inequality
theorem inequality_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq_6 : a + b + c = 6) : 
  a^2 + b^2 + c^2 ≥ 12 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_inequality_abc_l4040_404048


namespace NUMINAMATH_CALUDE_undefined_values_expression_undefined_l4040_404069

theorem undefined_values (x : ℝ) : 
  (2 * x^2 - 8 * x - 42 = 0) ↔ (x = 7 ∨ x = -3) :=
by sorry

theorem expression_undefined (x : ℝ) :
  ¬ (∃ y : ℝ, y = (3 * x^2 - 1) / (2 * x^2 - 8 * x - 42)) ↔ (x = 7 ∨ x = -3) :=
by sorry

end NUMINAMATH_CALUDE_undefined_values_expression_undefined_l4040_404069


namespace NUMINAMATH_CALUDE_largest_among_abcd_l4040_404033

theorem largest_among_abcd (a b c d : ℝ) 
  (h : a - 1 = b + 2 ∧ a - 1 = c - 3 ∧ a - 1 = d + 4) : 
  c ≥ a ∧ c ≥ b ∧ c ≥ d := by
sorry

end NUMINAMATH_CALUDE_largest_among_abcd_l4040_404033


namespace NUMINAMATH_CALUDE_base_conversion_3050_l4040_404040

def base_10_to_base_8 (n : ℕ) : ℕ :=
  5000 + 700 + 50 + 2

theorem base_conversion_3050 :
  base_10_to_base_8 3050 = 5752 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_3050_l4040_404040


namespace NUMINAMATH_CALUDE_fruit_bag_probabilities_l4040_404005

theorem fruit_bag_probabilities (apples oranges : ℕ) (h1 : apples = 7) (h2 : oranges = 1) :
  let total := apples + oranges
  (apples : ℚ) / total = 7 / 8 ∧ (oranges : ℚ) / total = 1 / 8 := by
sorry


end NUMINAMATH_CALUDE_fruit_bag_probabilities_l4040_404005


namespace NUMINAMATH_CALUDE_circumscribed_circle_radius_right_triangle_l4040_404072

/-- The radius of the circumscribed circle of a right triangle with sides 10, 8, and 6 is 5 -/
theorem circumscribed_circle_radius_right_triangle : 
  ∀ (a b c r : ℝ), 
  a = 10 → b = 8 → c = 6 → 
  a^2 = b^2 + c^2 → 
  r = a / 2 → 
  r = 5 := by sorry

end NUMINAMATH_CALUDE_circumscribed_circle_radius_right_triangle_l4040_404072


namespace NUMINAMATH_CALUDE_greatest_sum_consecutive_integers_l4040_404097

theorem greatest_sum_consecutive_integers (n : ℕ) : 
  (n * (n + 1) < 500) → 
  (∀ m : ℕ, m * (m + 1) < 500 → m + (m + 1) ≤ n + (n + 1)) →
  n + (n + 1) = 43 := by
  sorry

end NUMINAMATH_CALUDE_greatest_sum_consecutive_integers_l4040_404097


namespace NUMINAMATH_CALUDE_limit_P_div_B_l4040_404099

/-- The number of ways to make n cents using quarters, dimes, nickels, and pennies -/
def P (n : ℕ) : ℕ := sorry

/-- The number of ways to make n cents using dollar bills, quarters, dimes, and nickels -/
def B (n : ℕ) : ℕ := sorry

/-- The value of a penny in cents -/
def penny : ℕ := 1

/-- The value of a nickel in cents -/
def nickel : ℕ := 5

/-- The value of a dime in cents -/
def dime : ℕ := 10

/-- The value of a quarter in cents -/
def quarter : ℕ := 25

/-- The value of a dollar bill in cents -/
def dollar : ℕ := 100

/-- The limit of P_n / B_n as n approaches infinity -/
theorem limit_P_div_B :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |((P n : ℝ) / (B n : ℝ)) - (1 / 20)| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_P_div_B_l4040_404099


namespace NUMINAMATH_CALUDE_figure_reassemble_to_square_l4040_404095

/-- Represents a figure on a graph paper --/
structure GraphFigure where
  area : ℝ
  triangles : ℕ

/-- Represents a square --/
structure Square where
  side : ℝ

/-- Function to check if a figure can be reassembled into a square --/
def can_reassemble_to_square (figure : GraphFigure) (square : Square) : Prop :=
  figure.area = square.side ^ 2 ∧ figure.triangles = 5

/-- Theorem stating that the given figure can be reassembled into a square --/
theorem figure_reassemble_to_square :
  ∃ (figure : GraphFigure) (square : Square),
    figure.area = 20 ∧ can_reassemble_to_square figure square :=
by sorry

end NUMINAMATH_CALUDE_figure_reassemble_to_square_l4040_404095


namespace NUMINAMATH_CALUDE_inheritance_tax_problem_l4040_404089

theorem inheritance_tax_problem (x : ℝ) : 
  (0.25 * x + 0.15 * (x - 0.25 * x) = 12000) → 
  (round x : ℤ) = 33097 := by
  sorry

end NUMINAMATH_CALUDE_inheritance_tax_problem_l4040_404089


namespace NUMINAMATH_CALUDE_percentage_of_percentage_l4040_404047

theorem percentage_of_percentage (total : ℝ) (percentage1 : ℝ) (amount : ℝ) (percentage2 : ℝ) :
  total = 500 →
  percentage1 = 50 →
  amount = 25 →
  percentage2 = 10 →
  (amount / (percentage1 / 100 * total)) * 100 = percentage2 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_percentage_l4040_404047


namespace NUMINAMATH_CALUDE_football_angles_l4040_404002

-- Define the football structure
structure Football :=
  (edge_length : ℝ)
  (pentagon_sides : ℕ)
  (hexagon_sides : ℕ)
  (pentagons_per_hexagon : ℕ)

-- Define the angles between faces
def angle_between_hexagons (f : Football) : ℝ := sorry
def angle_between_hexagon_and_pentagon (f : Football) : ℝ := sorry

-- Theorem statement
theorem football_angles 
  (f : Football) 
  (h1 : f.edge_length = 1)
  (h2 : f.pentagon_sides = 5)
  (h3 : f.hexagon_sides = 6)
  (h4 : f.pentagons_per_hexagon = 5) :
  ∃ (α β : ℝ), 
    α = angle_between_hexagons f ∧
    β = angle_between_hexagon_and_pentagon f ∧
    ∃ (t1 t2 : ℝ → ℝ), 
      (t1 = Real.tan) ∧ 
      (t2 = Real.tan) ∧
      (t1 α = (Real.sqrt (3 * 3 - 2 * 2)) / 2) ∧
      (t2 β = (Real.sqrt (5 - 2 * Real.sqrt 5)) / (3 - Real.sqrt 5)) :=
sorry

end NUMINAMATH_CALUDE_football_angles_l4040_404002


namespace NUMINAMATH_CALUDE_average_of_a_and_b_l4040_404098

theorem average_of_a_and_b (a b : ℝ) : 
  (4 + 6 + 8 + a + b) / 5 = 20 → (a + b) / 2 = 41 := by
sorry

end NUMINAMATH_CALUDE_average_of_a_and_b_l4040_404098


namespace NUMINAMATH_CALUDE_complex_equation_sum_l4040_404042

theorem complex_equation_sum (a b : ℝ) :
  (a - 2 * Complex.I) * Complex.I = b - Complex.I →
  a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l4040_404042


namespace NUMINAMATH_CALUDE_xy_max_and_x_plus_y_min_l4040_404016

theorem xy_max_and_x_plus_y_min (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x * y + x + 2 * y = 6) :
  (∀ a b : ℝ, a > 0 → b > 0 → a * b + a + 2 * b = 6 → x * y ≥ a * b) ∧
  (∀ a b : ℝ, a > 0 → b > 0 → a * b + a + 2 * b = 6 → x + y ≤ a + b) ∧
  (x * y = 2 ∨ x + y = 4 * Real.sqrt 2 - 3) :=
sorry

end NUMINAMATH_CALUDE_xy_max_and_x_plus_y_min_l4040_404016


namespace NUMINAMATH_CALUDE_kozlov_inequality_l4040_404003

theorem kozlov_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a * b + b * c + c * a = 1) :
  Real.sqrt (a + 1 / a) + Real.sqrt (b + 1 / b) + Real.sqrt (c + 1 / c) ≥ 2 * (Real.sqrt a + Real.sqrt b + Real.sqrt c) := by
  sorry

end NUMINAMATH_CALUDE_kozlov_inequality_l4040_404003


namespace NUMINAMATH_CALUDE_one_meter_per_minute_not_implies_uniform_speed_l4040_404077

/-- A snail's movement over time -/
structure SnailMovement where
  /-- The distance traveled by the snail in meters -/
  distance : ℝ → ℝ
  /-- The property that the snail travels 1 meter every minute -/
  travels_one_meter_per_minute : ∀ t : ℝ, distance (t + 1) - distance t = 1

/-- Definition of uniform speed -/
def UniformSpeed (s : SnailMovement) : Prop :=
  ∃ v : ℝ, ∀ t₁ t₂ : ℝ, s.distance t₂ - s.distance t₁ = v * (t₂ - t₁)

/-- Theorem stating that traveling 1 meter per minute does not imply uniform speed -/
theorem one_meter_per_minute_not_implies_uniform_speed :
  ¬(∀ s : SnailMovement, UniformSpeed s) :=
sorry

end NUMINAMATH_CALUDE_one_meter_per_minute_not_implies_uniform_speed_l4040_404077


namespace NUMINAMATH_CALUDE_total_score_is_248_l4040_404056

/-- Calculates the total score across 4 subjects given 3 scores and the 4th as their average -/
def totalScoreAcross4Subjects (geography math english : ℕ) : ℕ :=
  let history := (geography + math + english) / 3
  geography + math + english + history

/-- Proves that given the specific scores, the total across 4 subjects is 248 -/
theorem total_score_is_248 :
  totalScoreAcross4Subjects 50 70 66 = 248 := by
  sorry

#eval totalScoreAcross4Subjects 50 70 66

end NUMINAMATH_CALUDE_total_score_is_248_l4040_404056


namespace NUMINAMATH_CALUDE_owen_burger_spending_l4040_404068

/-- The number of days in June -/
def days_in_june : ℕ := 30

/-- The number of burgers Owen buys per day -/
def burgers_per_day : ℕ := 2

/-- The cost of each burger in dollars -/
def cost_per_burger : ℕ := 12

/-- Theorem: Owen's total spending on burgers in June is $720 -/
theorem owen_burger_spending :
  days_in_june * burgers_per_day * cost_per_burger = 720 := by
  sorry


end NUMINAMATH_CALUDE_owen_burger_spending_l4040_404068


namespace NUMINAMATH_CALUDE_sum_of_squares_divisible_by_seven_l4040_404004

theorem sum_of_squares_divisible_by_seven (x y : ℤ) : 
  (7 ∣ x^2 + y^2) → (7 ∣ x) ∧ (7 ∣ y) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_divisible_by_seven_l4040_404004


namespace NUMINAMATH_CALUDE_sum_of_roots_l4040_404081

theorem sum_of_roots (x : ℝ) : 
  (∃ y z : ℝ, (3*x + 4)*(x - 5) + (3*x + 4)*(x - 7) = 0 ∧ x = y ∨ x = z) → 
  (∃ y z : ℝ, (3*x + 4)*(x - 5) + (3*x + 4)*(x - 7) = 0 ∧ x = y ∨ x = z ∧ y + z = 14/3) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_l4040_404081


namespace NUMINAMATH_CALUDE_domino_distribution_l4040_404058

theorem domino_distribution (total_dominoes : ℕ) (num_players : ℕ) 
  (h1 : total_dominoes = 28) (h2 : num_players = 4) :
  total_dominoes / num_players = 7 := by
  sorry

end NUMINAMATH_CALUDE_domino_distribution_l4040_404058


namespace NUMINAMATH_CALUDE_chocolate_bar_count_l4040_404009

/-- The number of chocolate bars in a box, given the cost per bar and the sales amount when all but 3 bars are sold. -/
def number_of_bars (cost_per_bar : ℕ) (sales_amount : ℕ) : ℕ :=
  (sales_amount + 3 * cost_per_bar) / cost_per_bar

theorem chocolate_bar_count : number_of_bars 3 18 = 9 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bar_count_l4040_404009


namespace NUMINAMATH_CALUDE_base_k_representation_of_fraction_l4040_404020

theorem base_k_representation_of_fraction (k : ℕ) (h : k = 18) :
  let series_sum := (1 / k + 6 / k^2) / (1 - 1 / k^2)
  series_sum = 8 / 63 := by
  sorry

end NUMINAMATH_CALUDE_base_k_representation_of_fraction_l4040_404020


namespace NUMINAMATH_CALUDE_symmetric_angles_sum_l4040_404092

theorem symmetric_angles_sum (α β : Real) : 
  0 < α ∧ α < 2 * Real.pi ∧ 
  0 < β ∧ β < 2 * Real.pi ∧ 
  α = 2 * Real.pi - β → 
  α + β = 2 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_symmetric_angles_sum_l4040_404092


namespace NUMINAMATH_CALUDE_smallest_positive_integer_ending_in_3_divisible_by_11_l4040_404090

theorem smallest_positive_integer_ending_in_3_divisible_by_11 : 
  ∃ n : ℕ, n > 0 ∧ n % 10 = 3 ∧ n % 11 = 0 ∧ 
  ∀ m : ℕ, m > 0 → m % 10 = 3 → m % 11 = 0 → m ≥ n :=
by
  use 33
  sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_ending_in_3_divisible_by_11_l4040_404090


namespace NUMINAMATH_CALUDE_correct_articles_for_categories_l4040_404032

-- Define a type for grammatical articles
inductive Article
  | Indefinite -- represents "a/an"
  | Definite   -- represents "the"
  | None       -- represents no article (used for plural nouns)

-- Define a function to determine the correct article for a category
def correctArticle (isFirstCategory : Bool) (isPlural : Bool) : Article :=
  if isFirstCategory then
    Article.Indefinite
  else if isPlural then
    Article.None
  else
    Article.Definite

-- Theorem statement
theorem correct_articles_for_categories :
  ∀ (isFirstCategory : Bool) (isPlural : Bool),
    (isFirstCategory ∧ ¬isPlural) →
    (¬isFirstCategory ∧ isPlural) →
    (correctArticle isFirstCategory isPlural = Article.Indefinite ∧
     correctArticle (¬isFirstCategory) isPlural = Article.None) :=
by
  sorry


end NUMINAMATH_CALUDE_correct_articles_for_categories_l4040_404032


namespace NUMINAMATH_CALUDE_initial_oak_trees_l4040_404091

theorem initial_oak_trees (final_trees : ℕ) (cut_trees : ℕ) : final_trees = 7 → cut_trees = 2 → final_trees + cut_trees = 9 := by
  sorry

end NUMINAMATH_CALUDE_initial_oak_trees_l4040_404091


namespace NUMINAMATH_CALUDE_equation_solution_l4040_404071

theorem equation_solution : ∃! x : ℝ, (32 : ℝ) ^ (x - 2) / (8 : ℝ) ^ (x - 2) = (512 : ℝ) ^ (3 * x) ∧ x = -4/25 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l4040_404071


namespace NUMINAMATH_CALUDE_factorization_2x_squared_minus_8_l4040_404084

theorem factorization_2x_squared_minus_8 (x : ℝ) : 2 * x^2 - 8 = 2 * (x + 2) * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_2x_squared_minus_8_l4040_404084


namespace NUMINAMATH_CALUDE_inverse_variation_sqrt_l4040_404014

/-- Given that y varies inversely as √x and y = 3 when x = 4, prove that y = √2 when x = 18 -/
theorem inverse_variation_sqrt (y : ℝ → ℝ) (k : ℝ) :
  (∀ x, y x * Real.sqrt x = k) →  -- y varies inversely as √x
  y 4 = 3 →                      -- y = 3 when x = 4
  y 18 = Real.sqrt 2 :=          -- y = √2 when x = 18
by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_sqrt_l4040_404014


namespace NUMINAMATH_CALUDE_inequality_solution_range_l4040_404041

theorem inequality_solution_range (k : ℝ) : 
  (k ≠ 0 ∧ k^2 * 1^2 - 6*k*1 + 8 ≥ 0) →
  k ∈ (Set.Ioi 4 : Set ℝ) ∪ (Set.Icc 0 2 : Set ℝ) ∪ (Set.Iio 0 : Set ℝ) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l4040_404041


namespace NUMINAMATH_CALUDE_roots_of_equation_l4040_404034

theorem roots_of_equation : 
  let f : ℝ → ℝ := λ x => x * (x - 3)^2 * (5 + x)
  {x : ℝ | f x = 0} = {0, 3, -5} := by
sorry

end NUMINAMATH_CALUDE_roots_of_equation_l4040_404034


namespace NUMINAMATH_CALUDE_time_conversion_not_100_l4040_404025

/-- Represents the conversion rate between adjacent time units -/
def time_conversion_rate : ℕ := 60

/-- The set of standard time units -/
inductive TimeUnit
| Hour
| Minute
| Second

theorem time_conversion_not_100 : time_conversion_rate ≠ 100 := by
  sorry

end NUMINAMATH_CALUDE_time_conversion_not_100_l4040_404025


namespace NUMINAMATH_CALUDE_hyperbola_equation_l4040_404021

/-- The standard equation of a hyperbola passing through a given point with a given eccentricity -/
theorem hyperbola_equation (x y : ℝ) (e : ℝ) (h1 : x = 3) (h2 : y = -Real.sqrt 2) (h3 : e = Real.sqrt 5 / 2) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
  (x^2 / a^2 - y^2 / b^2 = 1) ∧
  (e = Real.sqrt (a^2 + b^2) / a) ∧
  (a = 1 ∧ b = 1/2) := by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l4040_404021


namespace NUMINAMATH_CALUDE_original_number_is_ten_l4040_404064

theorem original_number_is_ten : 
  ∃ x : ℝ, (2 * x + 5 = x / 2 + 20) ∧ (x = 10) := by
  sorry

end NUMINAMATH_CALUDE_original_number_is_ten_l4040_404064


namespace NUMINAMATH_CALUDE_lucy_groceries_l4040_404006

/-- The number of packs of groceries Lucy bought -/
def total_groceries (cookies cake chocolate : ℕ) : ℕ :=
  cookies + cake + chocolate

/-- Theorem stating that Lucy bought 42 packs of groceries in total -/
theorem lucy_groceries : total_groceries 4 22 16 = 42 := by
  sorry

end NUMINAMATH_CALUDE_lucy_groceries_l4040_404006


namespace NUMINAMATH_CALUDE_solve_equation_l4040_404000

theorem solve_equation (x : ℤ) (h : 9873 + x = 13200) : x = 3327 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l4040_404000


namespace NUMINAMATH_CALUDE_final_elevation_proof_l4040_404059

def calculate_final_elevation (initial_elevation : ℝ) 
                               (rate1 rate2 rate3 : ℝ) 
                               (time1 time2 time3 : ℝ) : ℝ :=
  initial_elevation - (rate1 * time1 + rate2 * time2 + rate3 * time3)

theorem final_elevation_proof (initial_elevation : ℝ) 
                              (rate1 rate2 rate3 : ℝ) 
                              (time1 time2 time3 : ℝ) :
  calculate_final_elevation initial_elevation rate1 rate2 rate3 time1 time2 time3 =
  initial_elevation - (rate1 * time1 + rate2 * time2 + rate3 * time3) :=
by
  sorry

#eval calculate_final_elevation 400 10 15 12 5 3 6

end NUMINAMATH_CALUDE_final_elevation_proof_l4040_404059


namespace NUMINAMATH_CALUDE_solution_fraction_proof_l4040_404026

def initial_amount : ℚ := 2

def first_day_usage (amount : ℚ) : ℚ := (1 / 4) * amount

def second_day_usage (amount : ℚ) : ℚ := (1 / 2) * amount

def remaining_after_two_days (initial : ℚ) : ℚ :=
  initial - first_day_usage initial - second_day_usage (initial - first_day_usage initial)

theorem solution_fraction_proof :
  remaining_after_two_days initial_amount / initial_amount = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_solution_fraction_proof_l4040_404026


namespace NUMINAMATH_CALUDE_unique_sequence_existence_l4040_404054

theorem unique_sequence_existence :
  ∃! a : ℕ → ℝ,
    (∀ n, a n > 0) ∧
    a 0 = 1 ∧
    (∀ n : ℕ, a (n + 1) = a (n - 1) - a n) :=
by sorry

end NUMINAMATH_CALUDE_unique_sequence_existence_l4040_404054


namespace NUMINAMATH_CALUDE_no_solution_exists_l4040_404011

def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def number_with_B (B : ℕ) : ℕ := 12345670 + B

theorem no_solution_exists :
  ¬ ∃ B : ℕ, is_digit B ∧ 
    (number_with_B B).mod 2 = 0 ∧
    (number_with_B B).mod 5 = 0 ∧
    (number_with_B B).mod 11 = 0 :=
sorry

end NUMINAMATH_CALUDE_no_solution_exists_l4040_404011

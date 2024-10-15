import Mathlib

namespace NUMINAMATH_CALUDE_solution_for_k_3_solution_for_k_neg_2_solution_for_k_lt_neg_2_solution_for_k_between_neg_2_and_0_l696_69657

-- Define the inequality
def inequality (k : ℝ) (x : ℝ) : Prop :=
  k * x^2 + (k - 2) * x - 2 < 0

-- Theorem for k = 3
theorem solution_for_k_3 :
  ∀ x : ℝ, inequality 3 x ↔ -1 < x ∧ x < 2/3 :=
sorry

-- Theorems for k < 0
theorem solution_for_k_neg_2 :
  ∀ x : ℝ, inequality (-2) x ↔ x ≠ -1 :=
sorry

theorem solution_for_k_lt_neg_2 :
  ∀ k x : ℝ, k < -2 → (inequality k x ↔ x < -1 ∨ x > 2/k) :=
sorry

theorem solution_for_k_between_neg_2_and_0 :
  ∀ k x : ℝ, -2 < k ∧ k < 0 → (inequality k x ↔ x > -1 ∨ x < 2/k) :=
sorry

end NUMINAMATH_CALUDE_solution_for_k_3_solution_for_k_neg_2_solution_for_k_lt_neg_2_solution_for_k_between_neg_2_and_0_l696_69657


namespace NUMINAMATH_CALUDE_train_crossing_time_train_crossing_platform_time_l696_69632

/-- Calculates the time required for a train to cross a platform --/
theorem train_crossing_time 
  (train_speed_kmph : ℝ) 
  (man_crossing_time : ℝ) 
  (platform_length : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  let train_length := train_speed_mps * man_crossing_time
  let total_distance := train_length + platform_length
  total_distance / train_speed_mps

/-- Proves that the train takes 30 seconds to cross the platform --/
theorem train_crossing_platform_time : 
  train_crossing_time 72 19 220 = 30 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_train_crossing_platform_time_l696_69632


namespace NUMINAMATH_CALUDE_coin_triangle_proof_l696_69644

def triangle_sum (n : ℕ) : ℕ := n * (n + 1) / 2

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem coin_triangle_proof (N : ℕ) (h : triangle_sum N = 2016) :
  sum_of_digits N = 9 := by
  sorry

end NUMINAMATH_CALUDE_coin_triangle_proof_l696_69644


namespace NUMINAMATH_CALUDE_triple_overlap_area_is_six_l696_69631

/-- Represents a rectangular carpet with width and height in meters -/
structure Carpet where
  width : ℝ
  height : ℝ

/-- Represents the hall and the carpets placed in it -/
structure CarpetLayout where
  hallWidth : ℝ
  hallHeight : ℝ
  carpet1 : Carpet
  carpet2 : Carpet
  carpet3 : Carpet

/-- Calculates the area covered by all three carpets in the given layout -/
def tripleOverlapArea (layout : CarpetLayout) : ℝ :=
  sorry

/-- Theorem stating that the area covered by all three carpets is 6 square meters -/
theorem triple_overlap_area_is_six (layout : CarpetLayout) 
  (h1 : layout.hallWidth = 10 ∧ layout.hallHeight = 10)
  (h2 : layout.carpet1.width = 6 ∧ layout.carpet1.height = 8)
  (h3 : layout.carpet2.width = 6 ∧ layout.carpet2.height = 6)
  (h4 : layout.carpet3.width = 5 ∧ layout.carpet3.height = 7) :
  tripleOverlapArea layout = 6 :=
sorry

end NUMINAMATH_CALUDE_triple_overlap_area_is_six_l696_69631


namespace NUMINAMATH_CALUDE_bowling_ball_weight_is_correct_l696_69667

/-- The weight of a single bowling ball in pounds -/
def bowling_ball_weight : ℝ := 18.75

/-- The weight of a single canoe in pounds -/
def canoe_weight : ℝ := 30

/-- Theorem stating that the weight of one bowling ball is 18.75 pounds -/
theorem bowling_ball_weight_is_correct :
  (8 * bowling_ball_weight = 5 * canoe_weight) ∧
  (4 * canoe_weight = 120) →
  bowling_ball_weight = 18.75 :=
by
  sorry

end NUMINAMATH_CALUDE_bowling_ball_weight_is_correct_l696_69667


namespace NUMINAMATH_CALUDE_farm_animals_l696_69662

theorem farm_animals (total_legs : ℕ) (chicken_count : ℕ) : 
  total_legs = 38 → chicken_count = 5 → ∃ (sheep_count : ℕ), 
    chicken_count + sheep_count = 12 ∧ 
    2 * chicken_count + 4 * sheep_count = total_legs :=
by
  sorry

end NUMINAMATH_CALUDE_farm_animals_l696_69662


namespace NUMINAMATH_CALUDE_hyperbola_line_intersection_l696_69656

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2/9 - y^2 = 1

/-- The line equation -/
def line (x y : ℝ) : Prop := y = (1/3)*(x + 1)

/-- The number of intersection points -/
def intersection_count : ℕ := 1

/-- Theorem stating that the number of intersection points between the hyperbola and the line is 1 -/
theorem hyperbola_line_intersection :
  ∃! n : ℕ, n = intersection_count ∧ 
  (∃ (x y : ℝ), hyperbola x y ∧ line x y) ∧
  (∀ (x₁ y₁ x₂ y₂ : ℝ), hyperbola x₁ y₁ ∧ line x₁ y₁ ∧ hyperbola x₂ y₂ ∧ line x₂ y₂ → x₁ = x₂ ∧ y₁ = y₂) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_line_intersection_l696_69656


namespace NUMINAMATH_CALUDE_zoo_animal_count_l696_69664

/-- Calculates the total number of animals in a zoo with specific enclosure arrangements --/
def total_animals_in_zoo : ℕ :=
  let tiger_enclosures := 4
  let tigers_per_enclosure := 4
  let zebra_enclosures := (tiger_enclosures / 2) * 3
  let zebras_per_enclosure := 10
  let elephant_giraffe_pattern_repetitions := 4
  let elephants_per_enclosure := 3
  let giraffes_per_enclosure := 2
  let rhino_enclosures := 5
  let rhinos_per_enclosure := 1
  let chimpanzee_enclosures := rhino_enclosures * 2
  let chimpanzees_per_enclosure := 8

  let total_tigers := tiger_enclosures * tigers_per_enclosure
  let total_zebras := zebra_enclosures * zebras_per_enclosure
  let total_elephants := elephant_giraffe_pattern_repetitions * elephants_per_enclosure
  let total_giraffes := elephant_giraffe_pattern_repetitions * 2 * giraffes_per_enclosure
  let total_rhinos := rhino_enclosures * rhinos_per_enclosure
  let total_chimpanzees := chimpanzee_enclosures * chimpanzees_per_enclosure

  total_tigers + total_zebras + total_elephants + total_giraffes + total_rhinos + total_chimpanzees

theorem zoo_animal_count : total_animals_in_zoo = 189 := by
  sorry

end NUMINAMATH_CALUDE_zoo_animal_count_l696_69664


namespace NUMINAMATH_CALUDE_proportion_solution_l696_69629

theorem proportion_solution (x : ℚ) : (2 : ℚ) / 5 = (4 : ℚ) / 3 / x → x = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_proportion_solution_l696_69629


namespace NUMINAMATH_CALUDE_set_operation_result_l696_69663

def A : Set ℕ := {1, 3, 4, 5}
def B : Set ℕ := {2, 4, 6}
def C : Set ℕ := {0, 1, 2, 3, 4}

theorem set_operation_result : (A ∪ B) ∩ C = {1, 2, 3, 4} := by sorry

end NUMINAMATH_CALUDE_set_operation_result_l696_69663


namespace NUMINAMATH_CALUDE_sine_increasing_omega_range_l696_69666

/-- Given that y = sin(ωx) is increasing on the interval [-π/3, π/3], 
    the range of values for ω is (0, 3/2]. -/
theorem sine_increasing_omega_range (ω : ℝ) : 
  (∀ x ∈ Set.Icc (-π/3) (π/3), 
    Monotone (fun x => Real.sin (ω * x))) → 
  ω ∈ Set.Ioo 0 (3/2) :=
sorry

end NUMINAMATH_CALUDE_sine_increasing_omega_range_l696_69666


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l696_69607

theorem necessary_not_sufficient_condition (a : ℝ) :
  (∀ a, a^2 > 2*a → (a > 2 ∨ a < 0)) ∧
  (∃ a, a > 2 ∧ a^2 ≤ 2*a) :=
sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l696_69607


namespace NUMINAMATH_CALUDE_parallelogram_vertex_product_l696_69646

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A parallelogram defined by its four vertices -/
structure Parallelogram where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Check if two points are diagonally opposite in a parallelogram -/
def diagonallyOpposite (p : Parallelogram) (p1 p2 : Point) : Prop :=
  (p1 = p.A ∧ p2 = p.C) ∨ (p1 = p.B ∧ p2 = p.D) ∨ (p1 = p.C ∧ p2 = p.A) ∨ (p1 = p.D ∧ p2 = p.B)

/-- The main theorem -/
theorem parallelogram_vertex_product (p : Parallelogram) :
  p.A = Point.mk (-1) 3 →
  p.B = Point.mk 2 (-1) →
  p.D = Point.mk 7 6 →
  diagonallyOpposite p p.A p.D →
  p.C.x * p.C.y = 40 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_vertex_product_l696_69646


namespace NUMINAMATH_CALUDE_fifteen_plus_neg_twentythree_l696_69679

-- Define the operation for adding a positive and negative rational number
def add_pos_neg (a b : ℚ) : ℚ := -(b - a)

-- State the theorem
theorem fifteen_plus_neg_twentythree :
  15 + (-23) = add_pos_neg 15 23 :=
by sorry

end NUMINAMATH_CALUDE_fifteen_plus_neg_twentythree_l696_69679


namespace NUMINAMATH_CALUDE_divisibility_problem_l696_69697

theorem divisibility_problem (a b : ℕ) 
  (h1 : b ∣ (5 * a - 1))
  (h2 : b ∣ (a - 10))
  (h3 : ¬(b ∣ (3 * a + 5))) :
  b = 49 := by
sorry

end NUMINAMATH_CALUDE_divisibility_problem_l696_69697


namespace NUMINAMATH_CALUDE_polynomial_value_equality_l696_69606

theorem polynomial_value_equality (x : ℝ) : 
  x^2 - (5/2)*x = 6 → 2*x^2 - 5*x + 6 = 18 := by
sorry

end NUMINAMATH_CALUDE_polynomial_value_equality_l696_69606


namespace NUMINAMATH_CALUDE_larger_cross_section_distance_l696_69614

/-- Represents a right triangular pyramid -/
structure RightTriangularPyramid where
  /-- The height of the pyramid -/
  height : ℝ
  /-- The area of the base of the pyramid -/
  baseArea : ℝ

/-- Represents a cross section of the pyramid -/
structure CrossSection where
  /-- The distance from the apex to the cross section -/
  distanceFromApex : ℝ
  /-- The area of the cross section -/
  area : ℝ

/-- 
Theorem: In a right triangular pyramid, if two cross sections parallel to the base 
have areas of 144√3 and 324√3 square cm, and are 6 cm apart, 
then the larger cross section is 18 cm from the apex.
-/
theorem larger_cross_section_distance (pyramid : RightTriangularPyramid) 
  (section1 section2 : CrossSection) : 
  section1.area = 144 * Real.sqrt 3 →
  section2.area = 324 * Real.sqrt 3 →
  |section1.distanceFromApex - section2.distanceFromApex| = 6 →
  max section1.distanceFromApex section2.distanceFromApex = 18 := by
  sorry

#check larger_cross_section_distance

end NUMINAMATH_CALUDE_larger_cross_section_distance_l696_69614


namespace NUMINAMATH_CALUDE_exponent_multiplication_l696_69658

theorem exponent_multiplication (x : ℝ) (a b : ℕ) : x^a * x^b = x^(a + b) := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l696_69658


namespace NUMINAMATH_CALUDE_negation_of_implication_l696_69620

theorem negation_of_implication (x : ℝ) :
  ¬(x^2 = 1 → x = 1 ∨ x = -1) ↔ (x^2 ≠ 1 → x ≠ 1 ∧ x ≠ -1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l696_69620


namespace NUMINAMATH_CALUDE_kolya_best_strategy_method1_most_advantageous_method2_3_least_advantageous_l696_69693

/-- Represents the number of nuts Kolya gets in each method -/
structure KolyaNuts (n : ℕ) where
  method1 : ℕ
  method2 : ℕ
  method3 : ℕ

/-- The theorem stating the most and least advantageous methods for Kolya -/
theorem kolya_best_strategy (n : ℕ) (h : n ≥ 2) :
  ∃ (k : KolyaNuts n),
    k.method1 ≥ n + 1 ∧
    k.method2 ≤ n ∧
    k.method3 ≤ n :=
by sorry

/-- Helper function to determine the most advantageous method -/
def most_advantageous (k : KolyaNuts n) : ℕ :=
  max k.method1 (max k.method2 k.method3)

/-- Helper function to determine the least advantageous method -/
def least_advantageous (k : KolyaNuts n) : ℕ :=
  min k.method1 (min k.method2 k.method3)

/-- Theorem stating that method 1 is the most advantageous for Kolya -/
theorem method1_most_advantageous (n : ℕ) (h : n ≥ 2) :
  ∃ (k : KolyaNuts n), most_advantageous k = k.method1 :=
by sorry

/-- Theorem stating that methods 2 and 3 are the least advantageous for Kolya -/
theorem method2_3_least_advantageous (n : ℕ) (h : n ≥ 2) :
  ∃ (k : KolyaNuts n), least_advantageous k = k.method2 ∧ least_advantageous k = k.method3 :=
by sorry

end NUMINAMATH_CALUDE_kolya_best_strategy_method1_most_advantageous_method2_3_least_advantageous_l696_69693


namespace NUMINAMATH_CALUDE_circular_seating_arrangement_l696_69610

/-- The number of people at the table -/
def total_people : ℕ := 8

/-- The number of people who must sit together -/
def must_sit_together : ℕ := 2

/-- The number of people available to sit next to the fixed person -/
def available_neighbors : ℕ := total_people - must_sit_together - 1

/-- The number of neighbors to choose -/
def neighbors_to_choose : ℕ := 2

theorem circular_seating_arrangement :
  Nat.choose available_neighbors neighbors_to_choose = 10 := by
  sorry

end NUMINAMATH_CALUDE_circular_seating_arrangement_l696_69610


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_53_l696_69612

theorem smallest_four_digit_divisible_by_53 :
  ∀ n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ n % 53 = 0 → n ≥ 1007 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_53_l696_69612


namespace NUMINAMATH_CALUDE_representable_set_l696_69603

def representable (k : ℕ) : Prop :=
  ∃ x y z : ℕ+, k = (x + y + z)^2 / (x * y * z)

theorem representable_set : 
  {k : ℕ | representable k} = {1, 2, 3, 4, 5, 6, 8, 9} :=
by sorry

end NUMINAMATH_CALUDE_representable_set_l696_69603


namespace NUMINAMATH_CALUDE_intersection_of_sets_l696_69692

theorem intersection_of_sets : 
  let M : Set Int := {-1, 0}
  let N : Set Int := {0, 1}
  M ∩ N = {0} := by sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l696_69692


namespace NUMINAMATH_CALUDE_airplane_capacity_theorem_l696_69676

/-- The total luggage weight an airplane can hold -/
def airplane_luggage_capacity 
  (num_people : ℕ) 
  (bags_per_person : ℕ) 
  (bag_weight : ℕ) 
  (additional_bags : ℕ) : ℕ :=
  (num_people * bags_per_person * bag_weight) + (additional_bags * bag_weight)

/-- Theorem stating the total luggage weight the airplane can hold -/
theorem airplane_capacity_theorem :
  airplane_luggage_capacity 6 5 50 90 = 6000 := by
  sorry

end NUMINAMATH_CALUDE_airplane_capacity_theorem_l696_69676


namespace NUMINAMATH_CALUDE_paint_left_calculation_paint_problem_solution_l696_69618

/-- Given the total amount of paint needed and the amount of paint to buy,
    calculate the amount of paint left from the previous project. -/
theorem paint_left_calculation (total_paint : ℕ) (paint_to_buy : ℕ) :
  total_paint ≥ paint_to_buy →
  total_paint - paint_to_buy = total_paint - paint_to_buy :=
by
  sorry

/-- The specific problem instance -/
def paint_problem : ℕ × ℕ := (333, 176)

/-- The solution to the specific problem instance -/
theorem paint_problem_solution :
  let (total_paint, paint_to_buy) := paint_problem
  total_paint - paint_to_buy = 157 :=
by
  sorry

end NUMINAMATH_CALUDE_paint_left_calculation_paint_problem_solution_l696_69618


namespace NUMINAMATH_CALUDE_same_color_ratio_property_l696_69681

/-- A coloring of natural numbers using 2017 colors -/
def Coloring := ℕ → Fin 2017

/-- The theorem stating that for any coloring of natural numbers using 2017 colors,
    there exist two natural numbers of the same color with a specific ratio property -/
theorem same_color_ratio_property (c : Coloring) :
  ∃ (a b : ℕ), a ≠ 0 ∧ b ≠ 0 ∧ c a = c b ∧ 
  ∃ (k : ℕ), k ≠ 0 ∧ b = k * a ∧ 2016 ∣ k := by
  sorry

end NUMINAMATH_CALUDE_same_color_ratio_property_l696_69681


namespace NUMINAMATH_CALUDE_absolute_value_and_roots_l696_69649

theorem absolute_value_and_roots : |-3| + (Real.sqrt 2 - 1)^0 - (Real.sqrt 3)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_and_roots_l696_69649


namespace NUMINAMATH_CALUDE_area_under_curve_l696_69669

-- Define the curve
def f (x : ℝ) : ℝ := x^2 + 1

-- Define the bounds of integration
def a : ℝ := 0
def b : ℝ := 1

-- State the theorem
theorem area_under_curve : ∫ x in a..b, f x = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_area_under_curve_l696_69669


namespace NUMINAMATH_CALUDE_square_of_integer_l696_69655

theorem square_of_integer (x y : ℤ) (h : x + y = 10^18) :
  (x^2 * y^2) + ((x^2 + y^2) * (x + y)^2) = (x*y + x^2 + y^2)^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_integer_l696_69655


namespace NUMINAMATH_CALUDE_range_of_a_solution_set_l696_69675

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 1| + |x - 2|

-- Theorem for part I
theorem range_of_a (a : ℝ) :
  (∃ x, f x < 2 * a - 1) ↔ a > 2 :=
sorry

-- Theorem for part II
theorem solution_set :
  {x : ℝ | f x ≥ x^2 - 2*x} = {x : ℝ | -1 ≤ x ∧ x ≤ 2 + Real.sqrt 3} :=
sorry

end NUMINAMATH_CALUDE_range_of_a_solution_set_l696_69675


namespace NUMINAMATH_CALUDE_circle_radius_decrease_l696_69665

theorem circle_radius_decrease (r : ℝ) (h : r > 0) :
  let original_area := π * r^2
  let new_area := 0.58 * original_area
  let new_radius := Real.sqrt (new_area / π)
  (r - new_radius) / r = 1 - Real.sqrt 0.58 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_decrease_l696_69665


namespace NUMINAMATH_CALUDE_balloons_bought_at_park_l696_69638

theorem balloons_bought_at_park (allan_initial : ℕ) (jake_initial : ℕ) (jake_bought : ℕ) :
  allan_initial = 6 →
  jake_initial = 2 →
  allan_initial = jake_initial + jake_bought + 1 →
  jake_bought = 3 := by
sorry

end NUMINAMATH_CALUDE_balloons_bought_at_park_l696_69638


namespace NUMINAMATH_CALUDE_family_gathering_handshakes_count_l696_69601

/-- Represents the number of unique handshakes at a family gathering with twins and triplets -/
def familyGatheringHandshakes : ℕ :=
  let twin_sets := 12
  let triplet_sets := 5
  let twins := twin_sets * 2
  let triplets := triplet_sets * 3
  let first_twin_sets := 4
  let first_twins := first_twin_sets * 2

  -- Handshakes among twins
  let twin_handshakes := (twins * (twins - 2)) / 2

  -- Handshakes among triplets
  let triplet_handshakes := (triplets * (triplets - 3)) / 2

  -- Handshakes between first 4 sets of twins and triplets
  let first_twin_triplet_handshakes := first_twins * (triplets / 3)

  twin_handshakes + triplet_handshakes + first_twin_triplet_handshakes

/-- The total number of unique handshakes at the family gathering is 394 -/
theorem family_gathering_handshakes_count :
  familyGatheringHandshakes = 394 := by
  sorry

end NUMINAMATH_CALUDE_family_gathering_handshakes_count_l696_69601


namespace NUMINAMATH_CALUDE_sum_of_digits_theorem_l696_69635

def decimal_digit_sum (n : ℕ) : ℕ := sorry

theorem sum_of_digits_theorem : 
  decimal_digit_sum (2^2007 * 5^2005 * 7) = 10 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_theorem_l696_69635


namespace NUMINAMATH_CALUDE_orange_count_correct_l696_69685

/-- The number of oranges in the box -/
def num_oranges : ℕ := 24

/-- The initial number of kiwis in the box -/
def initial_kiwis : ℕ := 30

/-- The number of kiwis added to the box -/
def added_kiwis : ℕ := 26

/-- The percentage of oranges after adding kiwis -/
def orange_percentage : ℚ := 30 / 100

theorem orange_count_correct :
  (orange_percentage * (num_oranges + initial_kiwis + added_kiwis) : ℚ) = num_oranges := by
  sorry

end NUMINAMATH_CALUDE_orange_count_correct_l696_69685


namespace NUMINAMATH_CALUDE_binary_to_quaternary_conversion_l696_69699

/-- Converts a binary (base 2) number to its decimal (base 10) representation -/
def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

/-- Converts a decimal (base 10) number to its quaternary (base 4) representation -/
def decimal_to_quaternary (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
    aux n []

theorem binary_to_quaternary_conversion :
  decimal_to_quaternary (binary_to_decimal [false, true, true, true, false, true, true, false, true]) = [1, 1, 2, 3, 2] := by
  sorry

end NUMINAMATH_CALUDE_binary_to_quaternary_conversion_l696_69699


namespace NUMINAMATH_CALUDE_frank_second_half_correct_l696_69694

/-- Represents the number of questions Frank answered correctly in the first half -/
def first_half_correct : ℕ := 3

/-- Represents the points awarded for each correct answer -/
def points_per_question : ℕ := 3

/-- Represents Frank's final score -/
def final_score : ℕ := 15

/-- Calculates the number of questions Frank answered correctly in the second half -/
def second_half_correct : ℕ :=
  (final_score - first_half_correct * points_per_question) / points_per_question

theorem frank_second_half_correct :
  second_half_correct = 2 := by sorry

end NUMINAMATH_CALUDE_frank_second_half_correct_l696_69694


namespace NUMINAMATH_CALUDE_alien_home_planet_abductees_l696_69650

def total_abducted : ℕ := 1000
def return_percentage : ℚ := 528 / 1000
def to_zog : ℕ := 135
def to_xelbor : ℕ := 88
def to_qyruis : ℕ := 45

theorem alien_home_planet_abductees :
  total_abducted - 
  (↑(total_abducted) * return_percentage).floor - 
  to_zog - 
  to_xelbor - 
  to_qyruis = 204 := by
  sorry

end NUMINAMATH_CALUDE_alien_home_planet_abductees_l696_69650


namespace NUMINAMATH_CALUDE_canoe_trip_average_speed_l696_69604

/-- Proves that the average distance per day for the remaining days of a canoe trip is 32 km/day -/
theorem canoe_trip_average_speed
  (total_distance : ℝ)
  (total_days : ℕ)
  (completed_days : ℕ)
  (completed_fraction : ℚ)
  (h1 : total_distance = 168)
  (h2 : total_days = 6)
  (h3 : completed_days = 3)
  (h4 : completed_fraction = 3/7)
  : (total_distance - completed_fraction * total_distance) / (total_days - completed_days : ℝ) = 32 := by
  sorry

#check canoe_trip_average_speed

end NUMINAMATH_CALUDE_canoe_trip_average_speed_l696_69604


namespace NUMINAMATH_CALUDE_lcm_problem_l696_69613

theorem lcm_problem (a b c : ℕ+) (h1 : a = 15) (h2 : b = 25) (h3 : Nat.lcm (Nat.lcm a b) c = 525) : c = 7 := by
  sorry

end NUMINAMATH_CALUDE_lcm_problem_l696_69613


namespace NUMINAMATH_CALUDE_expression_simplification_l696_69641

theorem expression_simplification (x y : ℝ) :
  7 * y - 3 * x + 8 + 2 * y^2 - x + 12 = 2 * y^2 + 7 * y - 4 * x + 20 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l696_69641


namespace NUMINAMATH_CALUDE_triangle_area_problem_l696_69634

theorem triangle_area_problem (x : ℝ) (h1 : x > 0) : 
  (1/2 : ℝ) * x * (3*x) = 96 → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_problem_l696_69634


namespace NUMINAMATH_CALUDE_tailor_trim_problem_l696_69695

theorem tailor_trim_problem (original_side : ℝ) (trimmed_other_side : ℝ) (remaining_area : ℝ) 
  (h1 : original_side = 18)
  (h2 : trimmed_other_side = 3)
  (h3 : remaining_area = 120) :
  ∃ x : ℝ, x = 10 ∧ (original_side - x) * (original_side - trimmed_other_side) = remaining_area :=
by
  sorry

end NUMINAMATH_CALUDE_tailor_trim_problem_l696_69695


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l696_69668

theorem simplify_and_evaluate (a : ℝ) (h : a = 3) :
  (1 + 1 / (a + 1)) / ((a^2 - 4) / (2 * a + 2)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l696_69668


namespace NUMINAMATH_CALUDE_derivatives_correct_l696_69674

-- Function 1
def f₁ (x : ℝ) : ℝ := 3 * x^3 - 4 * x

-- Function 2
def f₂ (x : ℝ) : ℝ := (2 * x - 1) * (3 * x + 2)

-- Function 3
def f₃ (x : ℝ) : ℝ := x^2 * (x^3 - 4)

theorem derivatives_correct :
  (∀ x, deriv f₁ x = 9 * x^2 - 4) ∧
  (∀ x, deriv f₂ x = 12 * x + 1) ∧
  (∀ x, deriv f₃ x = 5 * x^4 - 8 * x) := by
  sorry

end NUMINAMATH_CALUDE_derivatives_correct_l696_69674


namespace NUMINAMATH_CALUDE_unit_digit_of_2_power_2024_l696_69617

theorem unit_digit_of_2_power_2024 (unit_digit : ℕ → ℕ) (h : ∀ n : ℕ, unit_digit (2^n) = unit_digit (2^(n % 4))) : unit_digit (2^2024) = 6 := by
  sorry

end NUMINAMATH_CALUDE_unit_digit_of_2_power_2024_l696_69617


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l696_69671

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  x^4 = (x^2 + 3*x - 4) * q + (-51*x + 52) :=
sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l696_69671


namespace NUMINAMATH_CALUDE_shortest_tangent_length_l696_69645

-- Define the circles
def C1 (x y : ℝ) : Prop := (x - 12)^2 + (y - 3)^2 = 25
def C2 (x y : ℝ) : Prop := (x + 9)^2 + (y + 4)^2 = 49

-- Define the centers and radii
def center1 : ℝ × ℝ := (12, 3)
def center2 : ℝ × ℝ := (-9, -4)
def radius1 : ℝ := 5
def radius2 : ℝ := 7

-- Theorem statement
theorem shortest_tangent_length :
  ∃ (R S : ℝ × ℝ),
    C1 R.1 R.2 ∧ C2 S.1 S.2 ∧
    (∀ (P Q : ℝ × ℝ), C1 P.1 P.2 → C2 Q.1 Q.2 →
      Real.sqrt ((R.1 - S.1)^2 + (R.2 - S.2)^2) ≤ Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)) ∧
    Real.sqrt ((R.1 - S.1)^2 + (R.2 - S.2)^2) = 70 :=
by sorry


end NUMINAMATH_CALUDE_shortest_tangent_length_l696_69645


namespace NUMINAMATH_CALUDE_parallel_lines_slope_l696_69624

theorem parallel_lines_slope (a : ℝ) : 
  (∃ (x y : ℝ), a * x - 5 * y - 9 = 0 ∧ 2 * x - 3 * y - 10 = 0) →
  a = 10/3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_slope_l696_69624


namespace NUMINAMATH_CALUDE_tan_alpha_sqrt_three_l696_69677

theorem tan_alpha_sqrt_three (α : Real) 
  (h1 : α ∈ Set.Ioo 0 (π / 2)) 
  (h2 : Real.sin α ^ 2 + Real.cos (2 * α) = 1 / 4) : 
  Real.tan α = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_sqrt_three_l696_69677


namespace NUMINAMATH_CALUDE_square_of_binomial_constant_l696_69611

/-- If 16x^2 + 40x + b is the square of a binomial, then b = 25 -/
theorem square_of_binomial_constant (b : ℝ) : 
  (∃ (p q : ℝ), ∀ x, 16 * x^2 + 40 * x + b = (p * x + q)^2) → b = 25 := by
  sorry

end NUMINAMATH_CALUDE_square_of_binomial_constant_l696_69611


namespace NUMINAMATH_CALUDE_min_cars_with_racing_stripes_l696_69602

theorem min_cars_with_racing_stripes 
  (total_cars : ℕ) 
  (cars_without_ac : ℕ) 
  (max_ac_no_stripes : ℕ) 
  (h1 : total_cars = 100)
  (h2 : cars_without_ac = 37)
  (h3 : max_ac_no_stripes = 59) :
  ∃ (min_cars_with_stripes : ℕ), 
    min_cars_with_stripes = 4 ∧ 
    min_cars_with_stripes ≤ total_cars - cars_without_ac ∧
    min_cars_with_stripes = total_cars - cars_without_ac - max_ac_no_stripes :=
by
  sorry

#check min_cars_with_racing_stripes

end NUMINAMATH_CALUDE_min_cars_with_racing_stripes_l696_69602


namespace NUMINAMATH_CALUDE_student_number_problem_l696_69630

theorem student_number_problem (x : ℝ) : 2 * x - 140 = 102 → x = 121 := by
  sorry

end NUMINAMATH_CALUDE_student_number_problem_l696_69630


namespace NUMINAMATH_CALUDE_min_max_cubic_linear_exists_y_min_max_zero_min_max_value_is_zero_l696_69670

theorem min_max_cubic_linear (y : ℝ) : 
  (∃ (x : ℝ), 0 ≤ x ∧ x ≤ 1 ∧ |x^3 - x*y| = 0) ∨ 
  (∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → |x^3 - x*y| > 0) :=
sorry

theorem exists_y_min_max_zero : 
  ∃ (y : ℝ), ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → |x^3 - x*y| ≤ 0 :=
sorry

theorem min_max_value_is_zero : 
  ∃ (y : ℝ), (∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → |x^3 - x*y| ≤ 0) ∧ 
  (∀ (y' : ℝ), (∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → |x^3 - x*y'| ≤ 0) → y' = y) :=
sorry

end NUMINAMATH_CALUDE_min_max_cubic_linear_exists_y_min_max_zero_min_max_value_is_zero_l696_69670


namespace NUMINAMATH_CALUDE_quadratic_rewrite_ratio_l696_69609

theorem quadratic_rewrite_ratio (b c : ℝ) :
  (∀ x, x^2 + 1300*x + 1300 = (x + b)^2 + c) →
  c / b = -648 := by
sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_ratio_l696_69609


namespace NUMINAMATH_CALUDE_domain_of_composite_function_l696_69682

-- Define the function f with domain (-1, 0)
def f : ℝ → ℝ := sorry

-- Define the composite function g(x) = f(2x+1)
def g (x : ℝ) : ℝ := f (2 * x + 1)

-- Theorem statement
theorem domain_of_composite_function :
  (∀ x, f x ≠ 0 → -1 < x ∧ x < 0) →
  (∀ x, g x ≠ 0 → -1 < x ∧ x < -1/2) :=
sorry

end NUMINAMATH_CALUDE_domain_of_composite_function_l696_69682


namespace NUMINAMATH_CALUDE_joaozinho_meeting_day_l696_69605

-- Define the days of the week
inductive Day : Type
  | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday

-- Define a function to determine if Joãozinho lies on a given day
def lies_on_day (d : Day) : Prop :=
  d = Day.Tuesday ∨ d = Day.Thursday ∨ d = Day.Saturday

-- Define a function to get the next day
def next_day (d : Day) : Day :=
  match d with
  | Day.Monday => Day.Tuesday
  | Day.Tuesday => Day.Wednesday
  | Day.Wednesday => Day.Thursday
  | Day.Thursday => Day.Friday
  | Day.Friday => Day.Saturday
  | Day.Saturday => Day.Sunday
  | Day.Sunday => Day.Monday

-- Theorem statement
theorem joaozinho_meeting_day :
  ∀ (meeting_day : Day),
    (lies_on_day meeting_day →
      (meeting_day ≠ Day.Saturday ∧
       next_day meeting_day ≠ Day.Wednesday)) →
    meeting_day = Day.Thursday :=
by
  sorry


end NUMINAMATH_CALUDE_joaozinho_meeting_day_l696_69605


namespace NUMINAMATH_CALUDE_largest_pot_cost_is_correct_l696_69627

/-- Represents the cost of flower pots and their properties -/
structure FlowerPots where
  num_pots : ℕ
  total_cost_after_discount : ℚ
  discount_per_pot : ℚ
  price_difference : ℚ

/-- Calculates the cost of the largest pot before discount -/
def largest_pot_cost (fp : FlowerPots) : ℚ :=
  let total_discount := fp.num_pots * fp.discount_per_pot
  let total_cost_before_discount := fp.total_cost_after_discount + total_discount
  let smallest_pot_cost := (total_cost_before_discount - (fp.num_pots - 1) * fp.num_pots / 2 * fp.price_difference) / fp.num_pots
  smallest_pot_cost + (fp.num_pots - 1) * fp.price_difference

/-- Theorem stating that the cost of the largest pot before discount is $1.85 -/
theorem largest_pot_cost_is_correct (fp : FlowerPots) 
  (h1 : fp.num_pots = 6)
  (h2 : fp.total_cost_after_discount = 33/4)  -- $8.25 as a fraction
  (h3 : fp.discount_per_pot = 1/10)           -- $0.10 as a fraction
  (h4 : fp.price_difference = 3/20)           -- $0.15 as a fraction
  : largest_pot_cost fp = 37/20 := by         -- $1.85 as a fraction
  sorry

#eval largest_pot_cost {num_pots := 6, total_cost_after_discount := 33/4, discount_per_pot := 1/10, price_difference := 3/20}

end NUMINAMATH_CALUDE_largest_pot_cost_is_correct_l696_69627


namespace NUMINAMATH_CALUDE_remainder_31_pow_31_plus_31_mod_32_l696_69660

theorem remainder_31_pow_31_plus_31_mod_32 : (31^31 + 31) % 32 = 30 := by
  sorry

end NUMINAMATH_CALUDE_remainder_31_pow_31_plus_31_mod_32_l696_69660


namespace NUMINAMATH_CALUDE_conic_touches_square_l696_69640

/-- The conic equation derived from the differential equation -/
def conic (h : ℝ) (x y : ℝ) : Prop :=
  y^2 + 2*h*x*y + x^2 = 9*(1 - h^2)

/-- The square with sides touching the conic -/
def square (x y : ℝ) : Prop :=
  (x = 3 ∨ x = -3 ∨ y = 3 ∨ y = -3) ∧ (abs x ≤ 3 ∧ abs y ≤ 3)

/-- The theorem stating that the conic touches the sides of the square -/
theorem conic_touches_square (h : ℝ) (h_bounds : 0 ≤ h ∧ h ≤ 1) :
  ∃ (x y : ℝ), conic h x y ∧ square x y :=
sorry

end NUMINAMATH_CALUDE_conic_touches_square_l696_69640


namespace NUMINAMATH_CALUDE_range_of_a_l696_69687

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, Real.sin x ^ 2 + Real.cos x + a = 0) → 
  a ∈ Set.Icc (-5/4 : ℝ) 1 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l696_69687


namespace NUMINAMATH_CALUDE_average_of_six_numbers_l696_69696

theorem average_of_six_numbers (a b c d e f : ℝ) :
  (a + b + c + d + e + f) / 6 = 4.60 →
  (c + d) / 2 = 3.8 →
  (e + f) / 2 = 6.6 →
  (a + b) / 2 = 3.4 :=
by sorry

end NUMINAMATH_CALUDE_average_of_six_numbers_l696_69696


namespace NUMINAMATH_CALUDE_remaining_water_l696_69637

/-- 
Given an initial amount of water and an amount used, 
calculate the remaining amount of water.
-/
theorem remaining_water (initial : ℚ) (used : ℚ) (remaining : ℚ) :
  initial = 4 →
  used = 9/4 →
  remaining = initial - used →
  remaining = 7/4 := by
  sorry

end NUMINAMATH_CALUDE_remaining_water_l696_69637


namespace NUMINAMATH_CALUDE_sunshine_orchard_pumpkins_l696_69643

def moonglow_pumpkins : ℕ := 14

def sunshine_pumpkins : ℕ := 3 * moonglow_pumpkins + 12

theorem sunshine_orchard_pumpkins : sunshine_pumpkins = 54 := by
  sorry

end NUMINAMATH_CALUDE_sunshine_orchard_pumpkins_l696_69643


namespace NUMINAMATH_CALUDE_probability_factor_less_than_8_of_90_l696_69639

def positive_factors (n : ℕ) : Finset ℕ :=
  (Finset.range n).filter (λ x => x > 0 ∧ n % x = 0)

def factors_less_than (n k : ℕ) : Finset ℕ :=
  (positive_factors n).filter (λ x => x < k)

theorem probability_factor_less_than_8_of_90 :
  (factors_less_than 90 8).card / (positive_factors 90).card = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_probability_factor_less_than_8_of_90_l696_69639


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l696_69689

theorem fractional_equation_solution :
  ∃ x : ℚ, (1 - x) / (x - 2) - 1 = 2 / (2 - x) ∧ x = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l696_69689


namespace NUMINAMATH_CALUDE_even_function_derivative_is_odd_l696_69623

theorem even_function_derivative_is_odd 
  (f : ℝ → ℝ) 
  (g : ℝ → ℝ) 
  (h_even : ∀ x, f (-x) = f x) 
  (h_deriv : ∀ x, HasDerivAt f (g x) x) : 
  ∀ x, g (-x) = -g x := by sorry

end NUMINAMATH_CALUDE_even_function_derivative_is_odd_l696_69623


namespace NUMINAMATH_CALUDE_programming_contest_grouping_l696_69647

/-- The number of programmers in the contest -/
def num_programmers : ℕ := 2008

/-- The number of rounds needed -/
def num_rounds : ℕ := 11

/-- A function that represents the grouping of programmers in each round -/
def grouping (round : ℕ) (programmer : ℕ) : Bool :=
  sorry

theorem programming_contest_grouping :
  (∀ (p1 p2 : ℕ), p1 < num_programmers → p2 < num_programmers → p1 ≠ p2 →
    ∃ (r : ℕ), r < num_rounds ∧ grouping r p1 ≠ grouping r p2) ∧
  (∀ (n : ℕ), n < num_rounds →
    ∃ (p1 p2 : ℕ), p1 < num_programmers ∧ p2 < num_programmers ∧ p1 ≠ p2 ∧
      ∀ (r : ℕ), r < n → grouping r p1 = grouping r p2) :=
sorry

end NUMINAMATH_CALUDE_programming_contest_grouping_l696_69647


namespace NUMINAMATH_CALUDE_expand_product_l696_69683

theorem expand_product (x : ℝ) : 2 * (x + 2) * (x + 3) * (x + 4) = 2 * x^3 + 18 * x^2 + 52 * x + 48 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l696_69683


namespace NUMINAMATH_CALUDE_school_students_l696_69622

def total_students (n : ℕ) (largest_class : ℕ) (diff : ℕ) : ℕ :=
  (n * (2 * largest_class - (n - 1) * diff)) / 2

theorem school_students :
  total_students 5 24 2 = 100 := by
  sorry

end NUMINAMATH_CALUDE_school_students_l696_69622


namespace NUMINAMATH_CALUDE_temperature_difference_l696_69648

theorem temperature_difference (highest lowest : Int) 
  (h1 : highest = 2) 
  (h2 : lowest = -8) : 
  highest - lowest = 10 := by
  sorry

end NUMINAMATH_CALUDE_temperature_difference_l696_69648


namespace NUMINAMATH_CALUDE_derivative_of_sin_cubed_inverse_l696_69686

noncomputable def f (x : ℝ) : ℝ := Real.sin (1 / x) ^ 3

theorem derivative_of_sin_cubed_inverse (x : ℝ) (hx : x ≠ 0) :
  deriv f x = -3 / x^2 * Real.sin (1 / x)^2 * Real.cos (1 / x) := by
  sorry

end NUMINAMATH_CALUDE_derivative_of_sin_cubed_inverse_l696_69686


namespace NUMINAMATH_CALUDE_increasing_linear_function_l696_69698

def linearFunction (k b : ℝ) (x : ℝ) : ℝ := k * x + b

theorem increasing_linear_function (k b : ℝ) (h : k > 0) :
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → linearFunction k b x₁ < linearFunction k b x₂ :=
by
  sorry

end NUMINAMATH_CALUDE_increasing_linear_function_l696_69698


namespace NUMINAMATH_CALUDE_season_length_l696_69600

/-- The number of games in the entire season -/
def total_games : ℕ := 20

/-- Donovan Mitchell's current average points per game -/
def current_average : ℕ := 26

/-- Number of games played so far -/
def games_played : ℕ := 15

/-- Donovan Mitchell's goal average for the entire season -/
def goal_average : ℕ := 30

/-- Required average for remaining games to reach the goal -/
def required_average : ℕ := 42

/-- Theorem stating that the total number of games is 20 -/
theorem season_length : 
  current_average * games_played + required_average * (total_games - games_played) = 
  goal_average * total_games := by sorry

end NUMINAMATH_CALUDE_season_length_l696_69600


namespace NUMINAMATH_CALUDE_min_value_xy_minus_2x_l696_69653

theorem min_value_xy_minus_2x (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : y * Real.log x + y * Real.log y = Real.exp x) :
  ∃ (m : ℝ), m = 2 - 2 * Real.log 2 ∧ ∀ (x' y' : ℝ), x' > 0 → y' > 0 → 
  y' * Real.log x' + y' * Real.log y' = Real.exp x' → x' * y' - 2 * x' ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_xy_minus_2x_l696_69653


namespace NUMINAMATH_CALUDE_problem_statement_l696_69628

theorem problem_statement :
  (∃ m : ℝ, |m| + m = 0 ∧ m ≥ 0) ∧
  (∃ a b : ℝ, |a - b| = b - a ∧ b ≤ a) ∧
  (∀ a b : ℝ, a^5 + b^5 = 0 → a + b = 0) ∧
  (∃ a b : ℝ, a + b = 0 ∧ a / b ≠ -1) ∧
  (∀ a b c : ℚ, |a| / a + |b| / b + |c| / c = 1 → |a * b * c| / (a * b * c) = -1) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l696_69628


namespace NUMINAMATH_CALUDE_prank_combinations_count_l696_69633

/-- The number of choices for each day of the week-long prank --/
def prank_choices : List Nat := [1, 2, 3, 4, 2]

/-- The total number of combinations for the week-long prank --/
def total_combinations : Nat := prank_choices.prod

/-- Theorem stating that the total number of combinations is 48 --/
theorem prank_combinations_count :
  total_combinations = 48 := by sorry

end NUMINAMATH_CALUDE_prank_combinations_count_l696_69633


namespace NUMINAMATH_CALUDE_car_speed_proof_l696_69684

/-- Proves that a car's speed is 60 km/h if it takes 12 seconds longer to travel 1 km compared to 75 km/h -/
theorem car_speed_proof (v : ℝ) : v > 0 → (1 / v * 3600 = 1 / 75 * 3600 + 12) ↔ v = 60 :=
by sorry

end NUMINAMATH_CALUDE_car_speed_proof_l696_69684


namespace NUMINAMATH_CALUDE_sqrt_5_greater_than_2_l696_69654

theorem sqrt_5_greater_than_2 : Real.sqrt 5 > 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_5_greater_than_2_l696_69654


namespace NUMINAMATH_CALUDE_inverse_proposition_reciprocals_l696_69651

/-- The inverse proposition of "If ab = 1, then a and b are reciprocals" -/
theorem inverse_proposition_reciprocals (a b : ℝ) :
  (a ≠ 0 ∧ b ≠ 0 ∧ a * b = 1 → a = 1 / b ∧ b = 1 / a) →
  (a = 1 / b ∧ b = 1 / a → a * b = 1) :=
sorry

end NUMINAMATH_CALUDE_inverse_proposition_reciprocals_l696_69651


namespace NUMINAMATH_CALUDE_four_zeros_when_a_positive_l696_69673

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then a * x + 1 else Real.log x / Real.log 3

def F (a : ℝ) (x : ℝ) : ℝ :=
  f a (f a x) + 1

theorem four_zeros_when_a_positive (a : ℝ) (h : a > 0) :
  ∃ (x₁ x₂ x₃ x₄ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
    F a x₁ = 0 ∧ F a x₂ = 0 ∧ F a x₃ = 0 ∧ F a x₄ = 0 ∧
    ∀ x, F a x = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄ :=
sorry

end

end NUMINAMATH_CALUDE_four_zeros_when_a_positive_l696_69673


namespace NUMINAMATH_CALUDE_unique_positive_number_l696_69672

theorem unique_positive_number : ∃! x : ℝ, x > 0 ∧ x^2 + x = 210 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_positive_number_l696_69672


namespace NUMINAMATH_CALUDE_plant_sales_net_profit_l696_69688

-- Define the costs
def basil_seed_cost : ℚ := 2
def mint_seed_cost : ℚ := 3
def zinnia_seed_cost : ℚ := 7
def potting_soil_cost : ℚ := 15

-- Define the number of plants per packet
def basil_plants_per_packet : ℕ := 20
def mint_plants_per_packet : ℕ := 15
def zinnia_plants_per_packet : ℕ := 10

-- Define the germination rates
def basil_germination_rate : ℚ := 4/5
def mint_germination_rate : ℚ := 3/4
def zinnia_germination_rate : ℚ := 7/10

-- Define the selling prices
def healthy_basil_price : ℚ := 5
def small_basil_price : ℚ := 3
def healthy_mint_price : ℚ := 6
def small_mint_price : ℚ := 4
def healthy_zinnia_price : ℚ := 10
def small_zinnia_price : ℚ := 7

-- Define the number of plants sold
def healthy_basil_sold : ℕ := 12
def small_basil_sold : ℕ := 8
def healthy_mint_sold : ℕ := 10
def small_mint_sold : ℕ := 4
def healthy_zinnia_sold : ℕ := 5
def small_zinnia_sold : ℕ := 2

-- Define the total cost
def total_cost : ℚ := basil_seed_cost + mint_seed_cost + zinnia_seed_cost + potting_soil_cost

-- Define the total revenue
def total_revenue : ℚ := 
  healthy_basil_price * healthy_basil_sold +
  small_basil_price * small_basil_sold +
  healthy_mint_price * healthy_mint_sold +
  small_mint_price * small_mint_sold +
  healthy_zinnia_price * healthy_zinnia_sold +
  small_zinnia_price * small_zinnia_sold

-- Define the net profit
def net_profit : ℚ := total_revenue - total_cost

-- Theorem to prove
theorem plant_sales_net_profit : net_profit = 197 := by sorry

end NUMINAMATH_CALUDE_plant_sales_net_profit_l696_69688


namespace NUMINAMATH_CALUDE_university_size_l696_69619

/-- Represents the total number of students in a university --/
def total_students (sample_size : ℕ) (other_grades_sample : ℕ) (other_grades_total : ℕ) : ℕ :=
  (other_grades_total * sample_size) / other_grades_sample

/-- Theorem stating the total number of students in the university --/
theorem university_size :
  let sample_size : ℕ := 500
  let freshmen_sample : ℕ := 200
  let sophomore_sample : ℕ := 100
  let other_grades_sample : ℕ := sample_size - freshmen_sample - sophomore_sample
  let other_grades_total : ℕ := 3000
  total_students sample_size other_grades_sample other_grades_total = 7500 := by
  sorry

end NUMINAMATH_CALUDE_university_size_l696_69619


namespace NUMINAMATH_CALUDE_lemonade_stand_profit_l696_69608

/-- Calculates the net profit from a lemonade stand --/
theorem lemonade_stand_profit
  (glasses_per_gallon : ℕ)
  (cost_per_gallon : ℚ)
  (gallons_made : ℕ)
  (price_per_glass : ℚ)
  (glasses_drunk : ℕ)
  (glasses_unsold : ℕ)
  (h1 : glasses_per_gallon = 16)
  (h2 : cost_per_gallon = 7/2)
  (h3 : gallons_made = 2)
  (h4 : price_per_glass = 1)
  (h5 : glasses_drunk = 5)
  (h6 : glasses_unsold = 6) :
  (gallons_made * glasses_per_gallon - glasses_drunk - glasses_unsold) * price_per_glass -
  (gallons_made * cost_per_gallon) = 14 :=
by sorry

end NUMINAMATH_CALUDE_lemonade_stand_profit_l696_69608


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l696_69680

theorem fraction_to_decimal : (63 : ℚ) / (2^3 * 5^4) = 0.0126 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l696_69680


namespace NUMINAMATH_CALUDE_prism_surface_area_l696_69636

theorem prism_surface_area (R : ℝ) (h : ℝ) :
  R > 0 →
  (R / 2)^2 + 3 = R^2 →
  2 + h^2 = 4 * R^2 →
  2 + 4 * h = 4 * Real.sqrt 14 + 2 :=
by sorry

end NUMINAMATH_CALUDE_prism_surface_area_l696_69636


namespace NUMINAMATH_CALUDE_exactly_one_approve_probability_l696_69659

def p_approve : ℝ := 0.7

def p_exactly_one_approve : ℝ :=
  3 * p_approve * (1 - p_approve) * (1 - p_approve)

theorem exactly_one_approve_probability :
  p_exactly_one_approve = 0.189 := by
  sorry

end NUMINAMATH_CALUDE_exactly_one_approve_probability_l696_69659


namespace NUMINAMATH_CALUDE_brooklyn_annual_donation_l696_69678

/-- Brooklyn's monthly donation in dollars -/
def monthly_donation : ℕ := 1453

/-- Number of months in a year -/
def months_in_year : ℕ := 12

/-- Brooklyn's total donation in a year -/
def annual_donation : ℕ := monthly_donation * months_in_year

theorem brooklyn_annual_donation : annual_donation = 17436 := by
  sorry

end NUMINAMATH_CALUDE_brooklyn_annual_donation_l696_69678


namespace NUMINAMATH_CALUDE_money_sum_l696_69626

theorem money_sum (a b : ℕ) (h1 : (1 : ℚ) / 3 * a = (1 : ℚ) / 4 * b) (h2 : b = 484) : a + b = 847 := by
  sorry

end NUMINAMATH_CALUDE_money_sum_l696_69626


namespace NUMINAMATH_CALUDE_tangent_circle_equation_l696_69661

/-- A circle passing through two points and tangent to a line -/
structure TangentCircle where
  center : ℝ × ℝ
  radius : ℝ
  passes_through_A : (center.1 - 1)^2 + (center.2 - 2)^2 = radius^2
  passes_through_B : (center.1 - 1)^2 + (center.2 - 10)^2 = radius^2
  tangent_to_line : |center.1 - 2*center.2 - 1| / Real.sqrt 5 = radius

/-- The theorem stating that a circle passing through (1, 2) and (1, 10) and 
    tangent to x - 2y - 1 = 0 must have one of two specific equations -/
theorem tangent_circle_equation : 
  ∀ c : TangentCircle, 
    ((c.center.1 = 3 ∧ c.center.2 = 6 ∧ c.radius^2 = 20) ∨
     (c.center.1 = -7 ∧ c.center.2 = 6 ∧ c.radius^2 = 80)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_circle_equation_l696_69661


namespace NUMINAMATH_CALUDE_winner_takes_eight_l696_69625

/-- Represents the game state and rules --/
structure Game where
  winner_candies : ℕ
  loser_candies : ℕ
  total_rounds : ℕ
  tim_wins : ℕ
  nick_total : ℕ
  tim_total : ℕ

/-- The game satisfies the given conditions --/
def valid_game (g : Game) : Prop :=
  g.winner_candies > g.loser_candies ∧
  g.loser_candies > 0 ∧
  g.tim_wins = 2 ∧
  g.nick_total = 30 ∧
  g.tim_total = 25 ∧
  g.total_rounds * (g.winner_candies + g.loser_candies) = g.nick_total + g.tim_total

/-- The theorem to be proved --/
theorem winner_takes_eight (g : Game) (h : valid_game g) : g.winner_candies = 8 := by
  sorry

end NUMINAMATH_CALUDE_winner_takes_eight_l696_69625


namespace NUMINAMATH_CALUDE_scaled_triangle_area_is_32_l696_69615

/-- The area of a triangle with vertices at (0,0), (-3, 7), and (-7, 3), scaled by a factor of 2 -/
def scaledTriangleArea : ℝ := 32

/-- The scaling factor -/
def scalingFactor : ℝ := 2

/-- The coordinates of the triangle vertices -/
def triangleVertices : List (ℝ × ℝ) := [(0, 0), (-3, 7), (-7, 3)]

/-- Theorem: The area of the scaled triangle is 32 square units -/
theorem scaled_triangle_area_is_32 :
  scaledTriangleArea = 32 :=
by sorry

end NUMINAMATH_CALUDE_scaled_triangle_area_is_32_l696_69615


namespace NUMINAMATH_CALUDE_apples_to_pears_ratio_l696_69690

/-- Represents the contents of a shopping cart --/
structure ShoppingCart where
  apples : ℕ
  oranges : ℕ
  pears : ℕ
  bananas : ℕ
  peaches : ℕ

/-- Defines the relationships between fruit quantities in the shopping cart --/
def validCart (cart : ShoppingCart) : Prop :=
  cart.oranges = 2 * cart.apples ∧
  cart.pears = 5 * cart.oranges ∧
  cart.bananas = 3 * cart.pears ∧
  cart.peaches = cart.bananas / 2

/-- Theorem stating that apples are 1/10 of pears in a valid shopping cart --/
theorem apples_to_pears_ratio (cart : ShoppingCart) (h : validCart cart) :
  cart.apples = cart.pears / 10 := by
  sorry


end NUMINAMATH_CALUDE_apples_to_pears_ratio_l696_69690


namespace NUMINAMATH_CALUDE_apple_bags_count_l696_69621

/-- The number of bags of apples loaded onto a lorry -/
def number_of_bags (empty_weight loaded_weight bag_weight : ℕ) : ℕ :=
  (loaded_weight - empty_weight) / bag_weight

/-- Theorem stating that the number of bags of apples is 20 -/
theorem apple_bags_count : 
  let empty_weight : ℕ := 500
  let loaded_weight : ℕ := 1700
  let bag_weight : ℕ := 60
  number_of_bags empty_weight loaded_weight bag_weight = 20 := by
  sorry

end NUMINAMATH_CALUDE_apple_bags_count_l696_69621


namespace NUMINAMATH_CALUDE_min_value_expression_l696_69652

theorem min_value_expression (a b c : ℝ) (h1 : c > b) (h2 : b > a) (h3 : c ≠ 0) :
  ((a + b)^2 + (b + c)^2 + (c - a)^2) / c^2 ≥ 2 ∧
  ∃ a' b' c', c' > b' ∧ b' > a' ∧ c' ≠ 0 ∧
    ((a' + b')^2 + (b' + c')^2 + (c' - a')^2) / c'^2 = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l696_69652


namespace NUMINAMATH_CALUDE_lcm_gcd_product_8_16_l696_69616

theorem lcm_gcd_product_8_16 : Nat.lcm 8 16 * Nat.gcd 8 16 = 128 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_product_8_16_l696_69616


namespace NUMINAMATH_CALUDE_cookies_per_box_type3_is_16_l696_69642

/-- The number of cookies in each box of the third type -/
def cookies_per_box_type3 (
  cookies_per_box_type1 : ℕ)
  (cookies_per_box_type2 : ℕ)
  (boxes_sold_type1 : ℕ)
  (boxes_sold_type2 : ℕ)
  (boxes_sold_type3 : ℕ)
  (total_cookies_sold : ℕ) : ℕ :=
  (total_cookies_sold - (cookies_per_box_type1 * boxes_sold_type1 + cookies_per_box_type2 * boxes_sold_type2)) / boxes_sold_type3

theorem cookies_per_box_type3_is_16 :
  cookies_per_box_type3 12 20 50 80 70 3320 = 16 := by
  sorry

end NUMINAMATH_CALUDE_cookies_per_box_type3_is_16_l696_69642


namespace NUMINAMATH_CALUDE_not_parabola_l696_69691

-- Define the equation
def equation (α : Real) (x y : Real) : Prop :=
  x^2 * Real.sin α + y^2 * Real.cos α = 1

-- Theorem statement
theorem not_parabola (α : Real) (h : α ∈ Set.Icc 0 Real.pi) :
  ¬∃ (a b c : Real), ∀ (x y : Real),
    equation α x y ↔ y = a*x^2 + b*x + c :=
sorry

end NUMINAMATH_CALUDE_not_parabola_l696_69691

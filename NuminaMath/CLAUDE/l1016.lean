import Mathlib

namespace NUMINAMATH_CALUDE_fourth_root_equation_solutions_l1016_101631

theorem fourth_root_equation_solutions :
  let f : ℝ → ℝ := λ x => (x ^ (1/4 : ℝ)) - 15 / (8 - x ^ (1/4 : ℝ))
  {x : ℝ | f x = 0} = {625, 81} := by
sorry

end NUMINAMATH_CALUDE_fourth_root_equation_solutions_l1016_101631


namespace NUMINAMATH_CALUDE_counterexample_exists_l1016_101699

theorem counterexample_exists : ∃ a : ℝ, a > -2 ∧ ¬(a^2 > 4) :=
  ⟨0, by
    constructor
    · -- Prove 0 > -2
      sorry
    · -- Prove ¬(0^2 > 4)
      sorry⟩

#check counterexample_exists

end NUMINAMATH_CALUDE_counterexample_exists_l1016_101699


namespace NUMINAMATH_CALUDE_smallest_multiplier_for_perfect_square_l1016_101604

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem smallest_multiplier_for_perfect_square : 
  (∀ k : ℕ, k > 0 ∧ k < 7 → ¬ is_perfect_square (1008 * k)) ∧ 
  is_perfect_square (1008 * 7) := by
  sorry

#check smallest_multiplier_for_perfect_square

end NUMINAMATH_CALUDE_smallest_multiplier_for_perfect_square_l1016_101604


namespace NUMINAMATH_CALUDE_corn_growth_ratio_l1016_101627

theorem corn_growth_ratio :
  ∀ (growth_week1 growth_week2 growth_week3 total_height : ℝ),
    growth_week1 = 2 →
    growth_week2 = 2 * growth_week1 →
    total_height = 22 →
    total_height = growth_week1 + growth_week2 + growth_week3 →
    growth_week3 / growth_week2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_corn_growth_ratio_l1016_101627


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l1016_101611

-- Define the universal set U
def U : Set (ℝ × ℝ) := Set.univ

-- Define set M
def M : Set (ℝ × ℝ) := {p | (p.2 + 2) / (p.1 - 2) = 1}

-- Define set N
def N : Set (ℝ × ℝ) := {p | p.2 ≠ p.1 - 4}

-- Statement to prove
theorem complement_intersection_theorem :
  (U \ M) ∩ (U \ N) = {(2, -2)} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l1016_101611


namespace NUMINAMATH_CALUDE_inequalities_not_equivalent_l1016_101693

theorem inequalities_not_equivalent : 
  ¬(∀ x : ℝ, (Real.sqrt (x - 1) < Real.sqrt (2 - x)) ↔ (x - 1 < 2 - x)) := by
sorry

end NUMINAMATH_CALUDE_inequalities_not_equivalent_l1016_101693


namespace NUMINAMATH_CALUDE_base_twelve_equality_l1016_101674

/-- Given a base b, this function converts a number in base b to decimal --/
def toDecimal (digits : List Nat) (b : Nat) : Nat :=
  digits.reverse.enum.foldl (fun acc (i, d) => acc + d * b^i) 0

/-- The proposition states that in base b, 35₍ᵦ₎² equals 1331₍ᵦ₎, and b equals 12 --/
theorem base_twelve_equality : ∃ b : Nat, 
  b > 1 ∧ 
  (toDecimal [3, 5] b)^2 = toDecimal [1, 3, 3, 1] b ∧ 
  b = 12 := by
  sorry

end NUMINAMATH_CALUDE_base_twelve_equality_l1016_101674


namespace NUMINAMATH_CALUDE_five_topping_pizzas_l1016_101658

theorem five_topping_pizzas (n : Nat) (k : Nat) (h1 : n = 8) (h2 : k = 5) :
  Nat.choose n k = 56 := by
  sorry

end NUMINAMATH_CALUDE_five_topping_pizzas_l1016_101658


namespace NUMINAMATH_CALUDE_ten_thousand_one_divides_repeat_digit_number_l1016_101617

/-- An 8-digit positive integer with the first four digits repeated -/
def RepeatDigitNumber (a b c d : Nat) : Nat :=
  a * 10000000 + b * 1000000 + c * 100000 + d * 10000 +
  a * 1000 + b * 100 + c * 10 + d

/-- Theorem: 10001 is a factor of any 8-digit number with repeated first four digits -/
theorem ten_thousand_one_divides_repeat_digit_number 
  (a b c d : Nat) (ha : a > 0) (hb : b < 10) (hc : c < 10) (hd : d < 10) :
  10001 ∣ RepeatDigitNumber a b c d := by
  sorry

end NUMINAMATH_CALUDE_ten_thousand_one_divides_repeat_digit_number_l1016_101617


namespace NUMINAMATH_CALUDE_special_set_bounds_l1016_101650

/-- A set of points in 3D space satisfying the given conditions -/
def SpecialSet (n : ℕ) (S : Set (ℝ × ℝ × ℝ)) : Prop :=
  (n > 0) ∧ 
  (∀ (planes : Finset (Set (ℝ × ℝ × ℝ))), planes.card = n → 
    ∃ (p : ℝ × ℝ × ℝ), p ∈ S ∧ ∀ (plane : Set (ℝ × ℝ × ℝ)), plane ∈ planes → p ∉ plane) ∧
  (∀ (X : ℝ × ℝ × ℝ), X ∈ S → 
    ∃ (planes : Finset (Set (ℝ × ℝ × ℝ))), planes.card = n ∧ 
      ∀ (Y : ℝ × ℝ × ℝ), Y ∈ S \ {X} → ∃ (plane : Set (ℝ × ℝ × ℝ)), plane ∈ planes ∧ Y ∈ plane)

theorem special_set_bounds (n : ℕ) (S : Set (ℝ × ℝ × ℝ)) (h : SpecialSet n S) :
  (3 * n + 1 : ℕ) ≤ S.ncard ∧ S.ncard ≤ Nat.choose (n + 3) 3 := by
  sorry

end NUMINAMATH_CALUDE_special_set_bounds_l1016_101650


namespace NUMINAMATH_CALUDE_total_pencils_distributed_l1016_101668

/-- 
Given a teacher who distributes pencils equally among students, 
this theorem proves that the total number of pencils distributed 
is equal to the product of the number of students and the number 
of pencils each student receives.
-/
theorem total_pencils_distributed (num_students : ℕ) (pencils_per_student : ℕ) 
  (h1 : num_students = 12) 
  (h2 : pencils_per_student = 3) : 
  num_students * pencils_per_student = 36 := by
  sorry

#check total_pencils_distributed

end NUMINAMATH_CALUDE_total_pencils_distributed_l1016_101668


namespace NUMINAMATH_CALUDE_quadratic_equivalences_l1016_101602

theorem quadratic_equivalences (x : ℝ) : 
  (((x ≠ 1 ∧ x ≠ 2) → x^2 - 3*x + 2 ≠ 0) ∧
   ((x^2 - 3*x + 2 = 0) → (x = 1 ∨ x = 2)) ∧
   ((x = 1 ∨ x = 2) → x^2 - 3*x + 2 = 0)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equivalences_l1016_101602


namespace NUMINAMATH_CALUDE_barbed_wire_rate_l1016_101639

/-- Given a square field with area 3136 sq m and a total cost of 932.40 Rs for drawing barbed wire
    around it, leaving two 1 m wide gates, the rate of drawing barbed wire per meter is 4.2 Rs/m. -/
theorem barbed_wire_rate (area : ℝ) (total_cost : ℝ) (gate_width : ℝ) (num_gates : ℕ) :
  area = 3136 →
  total_cost = 932.40 →
  gate_width = 1 →
  num_gates = 2 →
  (total_cost / (4 * Real.sqrt area - num_gates * gate_width) : ℝ) = 4.2 := by
  sorry

end NUMINAMATH_CALUDE_barbed_wire_rate_l1016_101639


namespace NUMINAMATH_CALUDE_solution_set_of_equation_l1016_101676

theorem solution_set_of_equation (x : ℝ) : 
  (Real.sin (2 * x) - π * Real.sin x) * Real.sqrt (11 * x^2 - x^4 - 10) = 0 ↔ 
  x ∈ ({-Real.sqrt 10, -π, -1, 1, π, Real.sqrt 10} : Set ℝ) := by
sorry

end NUMINAMATH_CALUDE_solution_set_of_equation_l1016_101676


namespace NUMINAMATH_CALUDE_mei_oranges_l1016_101677

theorem mei_oranges (peaches pears oranges baskets : ℕ) : 
  peaches = 9 →
  pears = 18 →
  baskets > 0 →
  peaches % baskets = 0 →
  pears % baskets = 0 →
  oranges % baskets = 0 →
  baskets = 3 →
  oranges = 9 :=
by sorry

end NUMINAMATH_CALUDE_mei_oranges_l1016_101677


namespace NUMINAMATH_CALUDE_lynn_ogen_interest_l1016_101647

/-- Calculates the total annual interest for Lynn Ogen's investments -/
theorem lynn_ogen_interest (x : ℝ) (h1 : x - 100 = 400) :
  0.09 * x + 0.07 * (x - 100) = 73 := by
  sorry

end NUMINAMATH_CALUDE_lynn_ogen_interest_l1016_101647


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_six_l1016_101673

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

def is_divisible_by (n m : ℕ) : Prop := m ∣ n

def last_digit (n : ℕ) : ℕ := n % 10

def sum_of_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  List.sum digits

theorem smallest_four_digit_divisible_by_six :
  ∀ n : ℕ, is_four_digit n →
    (is_divisible_by n 6 → n ≥ 1002) ∧
    (is_divisible_by 1002 6) ∧
    is_four_digit 1002 :=
sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_six_l1016_101673


namespace NUMINAMATH_CALUDE_colbert_treehouse_ratio_l1016_101653

/-- Proves that the ratio of planks from Colbert's parents to the total number of planks is 1:2 -/
theorem colbert_treehouse_ratio :
  let total_planks : ℕ := 200
  let storage_planks : ℕ := total_planks / 4
  let friends_planks : ℕ := 20
  let store_planks : ℕ := 30
  let parents_planks : ℕ := total_planks - (storage_planks + friends_planks + store_planks)
  (parents_planks : ℚ) / total_planks = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_colbert_treehouse_ratio_l1016_101653


namespace NUMINAMATH_CALUDE_rectangle_area_l1016_101613

theorem rectangle_area (L B : ℝ) (h1 : L - B = 23) (h2 : 2 * L + 2 * B = 166) : L * B = 1590 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l1016_101613


namespace NUMINAMATH_CALUDE_journey_time_comparison_l1016_101665

/-- Represents the speed of walking -/
def walking_speed : ℝ := 1

/-- Represents the speed of cycling -/
def cycling_speed : ℝ := 2 * walking_speed

/-- Represents the speed of the bus -/
def bus_speed : ℝ := 5 * cycling_speed

/-- Represents half the total journey distance -/
def half_journey : ℝ := 1

theorem journey_time_comparison : 
  (half_journey / bus_speed + half_journey / walking_speed) > (2 * half_journey) / cycling_speed :=
sorry

end NUMINAMATH_CALUDE_journey_time_comparison_l1016_101665


namespace NUMINAMATH_CALUDE_two_tails_in_seven_flips_l1016_101612

def unfair_coin_flip (n : ℕ) (k : ℕ) (p : ℚ) : ℚ :=
  (Nat.choose n k : ℚ) * p^k * (1 - p)^(n - k)

theorem two_tails_in_seven_flips :
  unfair_coin_flip 7 2 (3/4) = 189/16384 := by
  sorry

end NUMINAMATH_CALUDE_two_tails_in_seven_flips_l1016_101612


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l1016_101633

theorem absolute_value_inequality (x : ℝ) : 
  |x^2 - 5*x + 6| < x^2 - 4 ↔ x > 2 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l1016_101633


namespace NUMINAMATH_CALUDE_one_and_half_times_product_of_digits_l1016_101606

/-- Function to calculate the product of digits of a natural number -/
def productOfDigits (n : ℕ) : ℕ := sorry

/-- Theorem stating that 48 and 0 are the only natural numbers that are 1.5 times the product of their digits -/
theorem one_and_half_times_product_of_digits :
  ∀ (A : ℕ), A = (3 / 2 : ℚ) * (productOfDigits A) ↔ A = 48 ∨ A = 0 := by sorry

end NUMINAMATH_CALUDE_one_and_half_times_product_of_digits_l1016_101606


namespace NUMINAMATH_CALUDE_quadratic_always_positive_l1016_101638

theorem quadratic_always_positive (a : ℝ) :
  (∀ x : ℝ, a * x^2 - 2 * a * x + 3 > 0) → 0 ≤ a ∧ a < 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_l1016_101638


namespace NUMINAMATH_CALUDE_field_dimension_solution_l1016_101629

/-- Represents the dimensions of a rectangular field -/
structure FieldDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangular field -/
def fieldArea (d : FieldDimensions) : ℝ := d.length * d.width

/-- Theorem: For a rectangular field with dimensions (3m + 4) and (m - 3),
    if the area is 80 square units, then m = 19/3 -/
theorem field_dimension_solution (m : ℝ) :
  let d := FieldDimensions.mk (3 * m + 4) (m - 3)
  fieldArea d = 80 → m = 19/3 := by
  sorry


end NUMINAMATH_CALUDE_field_dimension_solution_l1016_101629


namespace NUMINAMATH_CALUDE_sqrt_two_minus_sqrt_eight_l1016_101615

theorem sqrt_two_minus_sqrt_eight : Real.sqrt 2 - Real.sqrt 8 = -Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_minus_sqrt_eight_l1016_101615


namespace NUMINAMATH_CALUDE_bookshop_inventory_l1016_101687

/-- Calculates the final number of books in a bookshop after weekend sales and a new shipment --/
theorem bookshop_inventory (initial_inventory : ℕ) (saturday_in_store : ℕ) (saturday_online : ℕ) (sunday_in_store_multiplier : ℕ) (sunday_online_increase : ℕ) (new_shipment : ℕ) : 
  initial_inventory = 743 →
  saturday_in_store = 37 →
  saturday_online = 128 →
  sunday_in_store_multiplier = 2 →
  sunday_online_increase = 34 →
  new_shipment = 160 →
  initial_inventory - 
    (saturday_in_store + saturday_online + 
     sunday_in_store_multiplier * saturday_in_store + 
     (saturday_online + sunday_online_increase)) + 
  new_shipment = 502 := by
sorry

end NUMINAMATH_CALUDE_bookshop_inventory_l1016_101687


namespace NUMINAMATH_CALUDE_two_distinct_roots_range_l1016_101698

theorem two_distinct_roots_range (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 - (m+2)*x - m + 1 = 0 ∧ y^2 - (m+2)*y - m + 1 = 0) ↔
  m < -8 ∨ m > 0 := by
sorry

end NUMINAMATH_CALUDE_two_distinct_roots_range_l1016_101698


namespace NUMINAMATH_CALUDE_ellipse_perpendicular_bisector_x_intersection_l1016_101667

def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

def perpendicular_bisector_intersects_x_axis (A B P : ℝ × ℝ) : Prop :=
  ∃ (m : ℝ), (P.2 = 0) ∧ 
  (P.1 - (A.1 + B.1)/2) = m * ((A.2 + B.2)/2) ∧
  (B.2 - A.2) * (P.1 - (A.1 + B.1)/2) = (A.1 - B.1) * ((A.2 + B.2)/2)

theorem ellipse_perpendicular_bisector_x_intersection
  (a b : ℝ) (h_ab : a > b ∧ b > 0) (A B P : ℝ × ℝ) :
  ellipse a b A.1 A.2 →
  ellipse a b B.1 B.2 →
  perpendicular_bisector_intersects_x_axis A B P →
  -((a^2 - b^2)/a) < P.1 ∧ P.1 < (a^2 - b^2)/a :=
by sorry

end NUMINAMATH_CALUDE_ellipse_perpendicular_bisector_x_intersection_l1016_101667


namespace NUMINAMATH_CALUDE_probability_next_queen_after_first_l1016_101680

/-- Represents a standard deck of 54 playing cards -/
def StandardDeck : ℕ := 54

/-- Number of queens in a standard deck -/
def QueenCount : ℕ := 4

/-- Probability of drawing a queen after the first queen -/
def ProbabilityNextQueenAfterFirst : ℚ := 2 / 27

/-- Theorem stating the probability of drawing a queen after the first queen -/
theorem probability_next_queen_after_first :
  ProbabilityNextQueenAfterFirst = QueenCount / StandardDeck :=
by
  sorry


end NUMINAMATH_CALUDE_probability_next_queen_after_first_l1016_101680


namespace NUMINAMATH_CALUDE_two_million_times_three_million_l1016_101601

theorem two_million_times_three_million : 
  (2 * 1000000) * (3 * 1000000) = 6 * 1000000000000 := by
  sorry

end NUMINAMATH_CALUDE_two_million_times_three_million_l1016_101601


namespace NUMINAMATH_CALUDE_expand_expression_l1016_101691

theorem expand_expression (x : ℝ) : (17*x + 18 - 3*x^2) * (4*x) = -12*x^3 + 68*x^2 + 72*x := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l1016_101691


namespace NUMINAMATH_CALUDE_cos_ninety_degrees_l1016_101628

theorem cos_ninety_degrees : Real.cos (π / 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_ninety_degrees_l1016_101628


namespace NUMINAMATH_CALUDE_sugar_amount_l1016_101651

/-- Represents the amounts of ingredients in pounds -/
structure Ingredients where
  sugar : ℝ
  flour : ℝ
  baking_soda : ℝ

/-- The ratios and conditions given in the problem -/
def satisfies_conditions (i : Ingredients) : Prop :=
  i.sugar / i.flour = 3 / 8 ∧
  i.flour / i.baking_soda = 10 ∧
  i.flour / (i.baking_soda + 60) = 8

/-- The theorem stating that under the given conditions, the amount of sugar is 900 pounds -/
theorem sugar_amount (i : Ingredients) :
  satisfies_conditions i → i.sugar = 900 := by
  sorry

end NUMINAMATH_CALUDE_sugar_amount_l1016_101651


namespace NUMINAMATH_CALUDE_c_value_theorem_l1016_101637

theorem c_value_theorem : ∃ c : ℝ, 
  (∀ x : ℝ, x * (3 * x + 1) < c ↔ -7/3 < x ∧ x < 2) ∧ c = 14 := by
  sorry

end NUMINAMATH_CALUDE_c_value_theorem_l1016_101637


namespace NUMINAMATH_CALUDE_point_not_on_transformed_plane_l1016_101605

/-- The similarity transformation coefficient -/
def k : ℚ := 5/2

/-- The original plane equation: x + y - 2z + 2 = 0 -/
def plane_a (x y z : ℚ) : Prop := x + y - 2*z + 2 = 0

/-- The transformed plane equation: x + y - 2z + 5 = 0 -/
def plane_a_transformed (x y z : ℚ) : Prop := x + y - 2*z + 5 = 0

/-- Point A -/
def point_A : ℚ × ℚ × ℚ := (2, -3, 1)

/-- Theorem: Point A does not belong to the image of plane a after similarity transformation -/
theorem point_not_on_transformed_plane :
  ¬ plane_a_transformed point_A.1 point_A.2.1 point_A.2.2 :=
by sorry

end NUMINAMATH_CALUDE_point_not_on_transformed_plane_l1016_101605


namespace NUMINAMATH_CALUDE_glued_polyhedron_edge_length_l1016_101648

/-- A polyhedron formed by gluing a square-based pyramid to a regular tetrahedron -/
structure GluedPolyhedron where
  -- Square-based pyramid
  pyramid_edge_length : ℝ
  pyramid_edge_count : ℕ
  -- Regular tetrahedron
  tetrahedron_edge_length : ℝ
  tetrahedron_edge_count : ℕ
  -- Gluing properties
  glued_edges : ℕ
  merged_edges : ℕ
  -- Conditions
  pyramid_square_base : pyramid_edge_count = 8
  all_edges_length_2 : pyramid_edge_length = 2 ∧ tetrahedron_edge_length = 2
  tetrahedron_regular : tetrahedron_edge_count = 6
  glued_face_edges : glued_edges = 3
  merged_parallel_edges : merged_edges = 2

/-- The total edge length of the glued polyhedron -/
def totalEdgeLength (p : GluedPolyhedron) : ℝ :=
  (p.pyramid_edge_count + p.tetrahedron_edge_count - p.glued_edges - p.merged_edges) * p.pyramid_edge_length

/-- Theorem stating that the total edge length of the glued polyhedron is 18 -/
theorem glued_polyhedron_edge_length (p : GluedPolyhedron) : totalEdgeLength p = 18 := by
  sorry

end NUMINAMATH_CALUDE_glued_polyhedron_edge_length_l1016_101648


namespace NUMINAMATH_CALUDE_cubic_equation_root_l1016_101672

theorem cubic_equation_root : ∃ x : ℝ, x^3 + 6*x^2 + 12*x + 35 = 0 :=
  by
    use -5
    -- Proof goes here
    sorry

end NUMINAMATH_CALUDE_cubic_equation_root_l1016_101672


namespace NUMINAMATH_CALUDE_fraction_simplification_l1016_101619

theorem fraction_simplification (x y : ℚ) (hx : x = 4) (hy : y = 5) :
  (1 / y + 1 / x) / (1 / x) = 9 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1016_101619


namespace NUMINAMATH_CALUDE_sand_pouring_problem_l1016_101655

/-- Represents the fraction of sand remaining after n pourings -/
def remaining_sand (n : ℕ) : ℚ :=
  2 / (n + 2)

/-- The number of pourings required to reach exactly 1/5 of the original sand -/
def required_pourings : ℕ := 8

theorem sand_pouring_problem :
  remaining_sand required_pourings = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_sand_pouring_problem_l1016_101655


namespace NUMINAMATH_CALUDE_scientific_notation_317000_l1016_101643

theorem scientific_notation_317000 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 317000 = a * (10 : ℝ) ^ n ∧ a = 3.17 ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_317000_l1016_101643


namespace NUMINAMATH_CALUDE_vincent_sticker_packs_l1016_101632

theorem vincent_sticker_packs (yesterday_packs today_extra_packs : ℕ) :
  yesterday_packs = 15 →
  today_extra_packs = 10 →
  yesterday_packs + (yesterday_packs + today_extra_packs) = 40 := by
  sorry

end NUMINAMATH_CALUDE_vincent_sticker_packs_l1016_101632


namespace NUMINAMATH_CALUDE_school_math_survey_l1016_101646

theorem school_math_survey (total : ℝ) (math_likers : ℝ) (olympiad_participants : ℝ)
  (h1 : math_likers ≤ total)
  (h2 : olympiad_participants = math_likers + 0.1 * (total - math_likers))
  (h3 : olympiad_participants = 0.46 * total) :
  math_likers = 0.4 * total :=
by sorry

end NUMINAMATH_CALUDE_school_math_survey_l1016_101646


namespace NUMINAMATH_CALUDE_leading_coefficient_of_p_l1016_101656

/-- The polynomial in question -/
def p (x : ℝ) : ℝ := -2*(x^5 - x^4 + 2*x^3) + 6*(x^5 + x^2 - 1) - 5*(3*x^5 + x^3 + 4)

/-- The leading coefficient of a polynomial -/
def leadingCoefficient (p : ℝ → ℝ) : ℝ :=
  sorry  -- Definition of leading coefficient

theorem leading_coefficient_of_p :
  leadingCoefficient p = -11 := by
  sorry

end NUMINAMATH_CALUDE_leading_coefficient_of_p_l1016_101656


namespace NUMINAMATH_CALUDE_min_value_w_l1016_101654

theorem min_value_w (x y : ℝ) : 
  3 * x^2 + 3 * y^2 + 9 * x - 6 * y + 30 ≥ 20.25 ∧ 
  ∃ (a b : ℝ), 3 * a^2 + 3 * b^2 + 9 * a - 6 * b + 30 = 20.25 :=
by sorry

end NUMINAMATH_CALUDE_min_value_w_l1016_101654


namespace NUMINAMATH_CALUDE_right_triangle_leg_length_l1016_101642

theorem right_triangle_leg_length
  (hypotenuse : ℝ)
  (leg1 : ℝ)
  (h1 : hypotenuse = 15)
  (h2 : leg1 = 9)
  (h3 : hypotenuse^2 = leg1^2 + leg2^2) :
  leg2 = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_leg_length_l1016_101642


namespace NUMINAMATH_CALUDE_parabola_intersection_sum_of_squares_l1016_101690

theorem parabola_intersection_sum_of_squares (k : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, x^2 + k*x + 4 = 0 ↔ x = x₁ ∨ x = x₂) →
  x₁ ≠ x₂ →
  x₁^2 + x₂^2 > 8 :=
by sorry

end NUMINAMATH_CALUDE_parabola_intersection_sum_of_squares_l1016_101690


namespace NUMINAMATH_CALUDE_probability_of_event_l1016_101670

noncomputable def x : ℝ := sorry

-- x is uniformly distributed between 200 and 300
axiom x_range : 200 ≤ x ∧ x < 300

-- Floor of square root of 2x is 25
axiom floor_sqrt_2x : ⌊Real.sqrt (2 * x)⌋ = 25

-- Define the event that floor of square root of x is 17
def event : Prop := ⌊Real.sqrt x⌋ = 17

-- Define the probability measure
noncomputable def P : Set ℝ → ℝ := sorry

-- Theorem statement
theorem probability_of_event :
  P {y : ℝ | 200 ≤ y ∧ y < 300 ∧ ⌊Real.sqrt (2 * y)⌋ = 25 ∧ ⌊Real.sqrt y⌋ = 17} / 
  P {y : ℝ | 200 ≤ y ∧ y < 300} = 23 / 200 := by sorry

end NUMINAMATH_CALUDE_probability_of_event_l1016_101670


namespace NUMINAMATH_CALUDE_negation_existence_real_l1016_101663

theorem negation_existence_real : 
  (¬ ∃ x : ℝ, x > 1) ↔ (∀ x : ℝ, x ≤ 1) := by sorry

end NUMINAMATH_CALUDE_negation_existence_real_l1016_101663


namespace NUMINAMATH_CALUDE_log_equation_solution_l1016_101684

theorem log_equation_solution (x : ℝ) (h : x > 0) :
  (Real.log x^3 / Real.log 3) + (Real.log x / Real.log (1/3)) = 8 → x = 81 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l1016_101684


namespace NUMINAMATH_CALUDE_poultry_farm_daily_loss_l1016_101621

/-- Calculates the daily loss of guinea fowls in a poultry farm scenario --/
theorem poultry_farm_daily_loss (initial_chickens initial_turkeys initial_guinea_fowls : ℕ)
  (daily_chicken_loss daily_turkey_loss : ℕ) (total_birds_after_week : ℕ) :
  initial_chickens = 300 →
  initial_turkeys = 200 →
  initial_guinea_fowls = 80 →
  daily_chicken_loss = 20 →
  daily_turkey_loss = 8 →
  total_birds_after_week = 349 →
  ∃ (daily_guinea_fowl_loss : ℕ),
    daily_guinea_fowl_loss = 5 ∧
    total_birds_after_week = 
      initial_chickens - 7 * daily_chicken_loss +
      initial_turkeys - 7 * daily_turkey_loss +
      initial_guinea_fowls - 7 * daily_guinea_fowl_loss :=
by
  sorry


end NUMINAMATH_CALUDE_poultry_farm_daily_loss_l1016_101621


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1016_101692

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h_arithmetic : is_arithmetic_sequence a)
  (h_sum : a 1 + a 7 = -2)
  (h_a3 : a 3 = 2) :
  ∃ d : ℝ, (∀ n : ℕ, a (n + 1) = a n + d) ∧ d = -3 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1016_101692


namespace NUMINAMATH_CALUDE_minimize_sum_of_distances_l1016_101645

/-- Given points A, B, and C in a 2D plane, where:
    A has coordinates (4, 6)
    B has coordinates (3, 0)
    C has coordinates (k, 0)
    This theorem states that the value of k that minimizes
    the sum of distances AC + BC is 3. -/
theorem minimize_sum_of_distances :
  let A : ℝ × ℝ := (4, 6)
  let B : ℝ × ℝ := (3, 0)
  let C : ℝ → ℝ × ℝ := λ k => (k, 0)
  let distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  let total_distance (k : ℝ) : ℝ := distance A (C k) + distance B (C k)
  ∃ k₀ : ℝ, k₀ = 3 ∧ ∀ k : ℝ, total_distance k₀ ≤ total_distance k :=
by sorry

end NUMINAMATH_CALUDE_minimize_sum_of_distances_l1016_101645


namespace NUMINAMATH_CALUDE_regular_hexagon_vector_relation_l1016_101683

-- Define a regular hexagon
structure RegularHexagon (V : Type*) [AddCommGroup V] [Module ℝ V] :=
  (A B C D E F : V)
  (is_regular : sorry)  -- This would typically include conditions that define a regular hexagon

-- Theorem statement
theorem regular_hexagon_vector_relation 
  {V : Type*} [AddCommGroup V] [Module ℝ V] 
  (hex : RegularHexagon V) 
  (a b : V) 
  (h1 : hex.B - hex.A = a) 
  (h2 : hex.E - hex.A = b) : 
  hex.C - hex.B = (1/2 : ℝ) • a + (1/2 : ℝ) • b := by
  sorry

end NUMINAMATH_CALUDE_regular_hexagon_vector_relation_l1016_101683


namespace NUMINAMATH_CALUDE_rachel_second_level_treasures_l1016_101610

/-- Represents the video game scoring system and Rachel's performance --/
structure GameScore where
  points_per_treasure : ℕ
  treasures_first_level : ℕ
  total_score : ℕ

/-- Calculates the number of treasures found on the second level --/
def treasures_second_level (game : GameScore) : ℕ :=
  (game.total_score - game.points_per_treasure * game.treasures_first_level) / game.points_per_treasure

/-- Theorem stating that Rachel found 2 treasures on the second level --/
theorem rachel_second_level_treasures :
  let game : GameScore := {
    points_per_treasure := 9,
    treasures_first_level := 5,
    total_score := 63
  }
  treasures_second_level game = 2 := by
  sorry

end NUMINAMATH_CALUDE_rachel_second_level_treasures_l1016_101610


namespace NUMINAMATH_CALUDE_distance_to_other_focus_l1016_101609

/-- The distance from a point on an ellipse to the other focus -/
theorem distance_to_other_focus (x y : ℝ) :
  x^2 / 9 + y^2 / 4 = 1 →  -- P is on the ellipse
  ∃ (f₁ f₂ : ℝ × ℝ),  -- existence of two foci
    (∀ (p : ℝ × ℝ), x^2 / 9 + y^2 / 4 = 1 →
      Real.sqrt ((p.1 - f₁.1)^2 + (p.2 - f₁.2)^2) +
      Real.sqrt ((p.1 - f₂.1)^2 + (p.2 - f₂.2)^2) = 6) →  -- definition of ellipse
    Real.sqrt ((x - f₁.1)^2 + (y - f₁.2)^2) = 1 →  -- distance to one focus is 1
    Real.sqrt ((x - f₂.1)^2 + (y - f₂.2)^2) = 5  -- distance to other focus is 5
    := by sorry

end NUMINAMATH_CALUDE_distance_to_other_focus_l1016_101609


namespace NUMINAMATH_CALUDE_beads_per_necklace_l1016_101679

theorem beads_per_necklace (total_beads : ℕ) (num_necklaces : ℕ) 
  (h1 : total_beads = 308) (h2 : num_necklaces = 11) :
  total_beads / num_necklaces = 28 := by
  sorry

end NUMINAMATH_CALUDE_beads_per_necklace_l1016_101679


namespace NUMINAMATH_CALUDE_units_digit_of_4_pow_10_l1016_101641

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The statement that the units digit of 4^10 is 6 -/
theorem units_digit_of_4_pow_10 : unitsDigit (4^10) = 6 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_4_pow_10_l1016_101641


namespace NUMINAMATH_CALUDE_point_on_unit_circle_l1016_101666

theorem point_on_unit_circle (t : ℝ) :
  let x := (t^3 - 1) / (t^3 + 1)
  let y := (2*t^3) / (t^3 + 1)
  x^2 + y^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_point_on_unit_circle_l1016_101666


namespace NUMINAMATH_CALUDE_alien_year_conversion_l1016_101618

/-- Converts a base-8 number to base-10 --/
def base8ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

/-- The problem statement --/
theorem alien_year_conversion :
  base8ToBase10 [2, 6, 3] = 242 := by
  sorry

end NUMINAMATH_CALUDE_alien_year_conversion_l1016_101618


namespace NUMINAMATH_CALUDE_sequence_fifth_term_is_fifteen_l1016_101681

theorem sequence_fifth_term_is_fifteen (a : ℕ → ℝ) :
  (∀ n : ℕ, n ≠ 0 → a n / n = n - 2) →
  a 5 = 15 := by
sorry

end NUMINAMATH_CALUDE_sequence_fifth_term_is_fifteen_l1016_101681


namespace NUMINAMATH_CALUDE_inequality_theta_range_l1016_101664

theorem inequality_theta_range (θ : Real) : 
  (∀ x ∈ Set.Icc 0 1, x^2 * Real.cos θ - x * (1 - x) + (1 - x)^2 * Real.sin θ > 0) ↔ 
  ∃ k : ℤ, θ ∈ Set.Ioo (2 * k * Real.pi + Real.pi / 12) (2 * k * Real.pi + 5 * Real.pi / 12) :=
by sorry

end NUMINAMATH_CALUDE_inequality_theta_range_l1016_101664


namespace NUMINAMATH_CALUDE_promotion_b_saves_more_l1016_101661

/-- The cost of a single shirt in dollars -/
def shirtCost : ℝ := 40

/-- The cost of two shirts under Promotion A -/
def promotionACost : ℝ := shirtCost + (shirtCost * 0.75)

/-- The cost of two shirts under Promotion B -/
def promotionBCost : ℝ := shirtCost + (shirtCost - 15)

/-- Theorem stating that Promotion B costs $5 less than Promotion A -/
theorem promotion_b_saves_more :
  promotionACost - promotionBCost = 5 := by
  sorry

end NUMINAMATH_CALUDE_promotion_b_saves_more_l1016_101661


namespace NUMINAMATH_CALUDE_coopers_age_l1016_101685

theorem coopers_age (cooper dante maria : ℕ) : 
  cooper + dante + maria = 31 →
  dante = 2 * cooper →
  maria = dante + 1 →
  cooper = 6 := by
sorry

end NUMINAMATH_CALUDE_coopers_age_l1016_101685


namespace NUMINAMATH_CALUDE_triangle_area_product_l1016_101616

theorem triangle_area_product (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : (1/2) * (12/a) * (12/b) = 12) : a * b = 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_product_l1016_101616


namespace NUMINAMATH_CALUDE_voldemort_lunch_calories_l1016_101675

/-- Calculates the calories consumed for lunch given the daily calorie limit,
    calories from dinner items, breakfast, and remaining calories. -/
def calories_for_lunch (daily_limit : ℕ) (cake : ℕ) (chips : ℕ) (coke : ℕ)
                       (breakfast : ℕ) (remaining : ℕ) : ℕ :=
  daily_limit - (cake + chips + coke + breakfast + remaining)

/-- Proves that Voldemort consumed 780 calories for lunch. -/
theorem voldemort_lunch_calories :
  calories_for_lunch 2500 110 310 215 560 525 = 780 := by
  sorry

end NUMINAMATH_CALUDE_voldemort_lunch_calories_l1016_101675


namespace NUMINAMATH_CALUDE_max_abs_z_value_l1016_101608

theorem max_abs_z_value (a b c z : ℂ) 
  (h1 : Complex.abs a = Complex.abs b) 
  (h2 : Complex.abs a = 2 * Complex.abs c)
  (h3 : Complex.abs a > 0)
  (h4 : 2 * a * z^2 + b * z + c * z = 0) : 
  Complex.abs z ≤ 3/4 := by
  sorry

end NUMINAMATH_CALUDE_max_abs_z_value_l1016_101608


namespace NUMINAMATH_CALUDE_interior_perimeter_is_155_l1016_101614

/-- Triangle PQR with parallel lines forming interior triangle --/
structure TriangleWithParallels where
  /-- Side length PQ --/
  pq : ℝ
  /-- Side length QR --/
  qr : ℝ
  /-- Side length PR --/
  pr : ℝ
  /-- Length of intersection of m_P with triangle interior --/
  m_p : ℝ
  /-- Length of intersection of m_Q with triangle interior --/
  m_q : ℝ
  /-- Length of intersection of m_R with triangle interior --/
  m_r : ℝ
  /-- m_P is parallel to QR --/
  m_p_parallel_qr : True
  /-- m_Q is parallel to RP --/
  m_q_parallel_rp : True
  /-- m_R is parallel to PQ --/
  m_r_parallel_pq : True

/-- The perimeter of the interior triangle formed by parallel lines --/
def interiorPerimeter (t : TriangleWithParallels) : ℝ :=
  t.m_p + t.m_q + t.m_r

/-- Theorem: The perimeter of the interior triangle is 155 --/
theorem interior_perimeter_is_155 (t : TriangleWithParallels) 
  (h1 : t.pq = 160) (h2 : t.qr = 300) (h3 : t.pr = 240)
  (h4 : t.m_p = 75) (h5 : t.m_q = 60) (h6 : t.m_r = 20) :
  interiorPerimeter t = 155 := by
  sorry

end NUMINAMATH_CALUDE_interior_perimeter_is_155_l1016_101614


namespace NUMINAMATH_CALUDE_simplify_expression_l1016_101669

theorem simplify_expression (y : ℝ) : 5*y + 7*y - 3*y = 9*y := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1016_101669


namespace NUMINAMATH_CALUDE_final_pen_count_l1016_101657

def pen_collection (initial : ℕ) (mike_gives : ℕ) (cindy_multiplier : ℕ) (sharon_takes : ℕ) : ℕ :=
  ((initial + mike_gives) * cindy_multiplier) - sharon_takes

theorem final_pen_count : pen_collection 5 20 2 10 = 40 := by
  sorry

end NUMINAMATH_CALUDE_final_pen_count_l1016_101657


namespace NUMINAMATH_CALUDE_right_triangle_max_ratio_l1016_101688

theorem right_triangle_max_ratio (k l a b c : ℝ) (hk : k > 0) (hl : l > 0) : 
  (k * a)^2 + (l * b)^2 = c^2 → (k * a + l * b) / c ≤ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_max_ratio_l1016_101688


namespace NUMINAMATH_CALUDE_smaller_number_problem_l1016_101623

theorem smaller_number_problem (x y : ℤ) : 
  y = 2 * x - 3 →  -- One number is 3 less than twice another
  x + y = 39 →     -- The sum of the two numbers is 39
  x = 14           -- The smaller number is 14
  := by sorry

end NUMINAMATH_CALUDE_smaller_number_problem_l1016_101623


namespace NUMINAMATH_CALUDE_unique_prime_product_perfect_power_l1016_101696

/-- The nth prime number -/
def nthPrime (n : ℕ) : ℕ := sorry

/-- The product of the first k prime numbers -/
def primeProduct (k : ℕ) : ℕ := sorry

/-- A number is a perfect power if it can be expressed as a^n where a > 1 and n > 1 -/
def isPerfectPower (m : ℕ) : Prop := sorry

theorem unique_prime_product_perfect_power :
  ∀ k : ℕ, (k ≠ 0 ∧ isPerfectPower (primeProduct k - 1)) ↔ k = 1 := by sorry

end NUMINAMATH_CALUDE_unique_prime_product_perfect_power_l1016_101696


namespace NUMINAMATH_CALUDE_min_colors_is_23_l1016_101603

/-- Represents a coloring arrangement for 8 boxes with 6 balls each -/
structure ColorArrangement where
  n : ℕ  -- Number of colors
  boxes : Fin 8 → Finset (Fin n)
  all_boxes_size_six : ∀ i, (boxes i).card = 6
  no_duplicate_colors : ∀ i j, i ≠ j → (boxes i ∩ boxes j).card ≤ 1

/-- The minimum number of colors needed for a valid ColorArrangement -/
def min_colors : ℕ := 23

/-- Theorem stating that 23 is the minimum number of colors needed -/
theorem min_colors_is_23 :
  (∃ arrangement : ColorArrangement, arrangement.n = min_colors) ∧
  (∀ arrangement : ColorArrangement, arrangement.n ≥ min_colors) :=
sorry

end NUMINAMATH_CALUDE_min_colors_is_23_l1016_101603


namespace NUMINAMATH_CALUDE_repeating_block_length_l1016_101671

/-- The number of digits in the smallest repeating block of the decimal expansion of 4/7 -/
def smallest_repeating_block_length : ℕ := 6

/-- The fraction we're considering -/
def fraction : ℚ := 4/7

theorem repeating_block_length :
  smallest_repeating_block_length = 6 ∧ 
  ∃ (n : ℕ) (d : ℕ+), fraction = n / d ∧ 
  smallest_repeating_block_length ≤ d - 1 :=
sorry

end NUMINAMATH_CALUDE_repeating_block_length_l1016_101671


namespace NUMINAMATH_CALUDE_problem_statement_l1016_101640

theorem problem_statement (x y z : ℝ) 
  (sum_eq : x + y + z = 12) 
  (sum_sq_eq : x^2 + y^2 + z^2 = 54) : 
  (9 ≤ x*y ∧ x*y ≤ 25) ∧ 
  (9 ≤ y*z ∧ y*z ≤ 25) ∧ 
  (9 ≤ z*x ∧ z*x ≤ 25) ∧
  ((x ≤ 3 ∧ (y ≥ 5 ∨ z ≥ 5)) ∨ 
   (y ≤ 3 ∧ (x ≥ 5 ∨ z ≥ 5)) ∨ 
   (z ≤ 3 ∧ (x ≥ 5 ∨ y ≥ 5))) := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l1016_101640


namespace NUMINAMATH_CALUDE_sum_of_largest_and_smallest_prime_factors_of_1320_l1016_101622

theorem sum_of_largest_and_smallest_prime_factors_of_1320 :
  ∃ (smallest largest : Nat),
    smallest.Prime ∧
    largest.Prime ∧
    smallest ∣ 1320 ∧
    largest ∣ 1320 ∧
    (∀ p : Nat, p.Prime → p ∣ 1320 → p ≥ smallest) ∧
    (∀ p : Nat, p.Prime → p ∣ 1320 → p ≤ largest) ∧
    smallest + largest = 13 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_largest_and_smallest_prime_factors_of_1320_l1016_101622


namespace NUMINAMATH_CALUDE_rectangle_decomposition_theorem_l1016_101644

/-- A function that checks if a rectangle can be decomposed into n and m+n congruent squares -/
def has_unique_decomposition (m : ℕ+) : Prop :=
  ∃! n : ℕ+, ∃ a b : ℕ+, a^2 - b^2 = n ∧ a^2 - b^2 = m + n

/-- A function that checks if a number is an odd prime -/
def is_odd_prime (p : ℕ+) : Prop :=
  Nat.Prime p.val ∧ p.val % 2 = 1

/-- The main theorem stating the equivalence of the two conditions -/
theorem rectangle_decomposition_theorem (m : ℕ+) :
  has_unique_decomposition m ↔ 
  (∃ p : ℕ+, is_odd_prime p ∧ (m = p ∨ m = 2 * p ∨ m = 4 * p)) :=
sorry

end NUMINAMATH_CALUDE_rectangle_decomposition_theorem_l1016_101644


namespace NUMINAMATH_CALUDE_system_solution_characterization_l1016_101660

/-- The system of equations has either a unique solution or infinitely many solutions when m ≠ -1 -/
theorem system_solution_characterization (m : ℝ) (hm : m ≠ -1) :
  (∃! x y : ℝ, m * x + y = m + 1 ∧ x + m * y = 2 * m) ∨
  (∃ f g : ℝ → ℝ, ∀ t : ℝ, m * (f t) + (g t) = m + 1 ∧ (f t) + m * (g t) = 2 * m) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_characterization_l1016_101660


namespace NUMINAMATH_CALUDE_isosceles_triangle_rational_trig_l1016_101649

/-- An isosceles triangle with integer base and height has rational sine and cosine of vertex angle -/
theorem isosceles_triangle_rational_trig (BC AD : ℤ) (h : BC > 0 ∧ AD > 0) : 
  ∃ (sinA cosA : ℚ), 
    sinA = Real.sin (Real.arccos ((BC * BC - 2 * AD * AD) / (BC * BC + 2 * AD * AD))) ∧
    cosA = (BC * BC - 2 * AD * AD) / (BC * BC + 2 * AD * AD) := by
  sorry

#check isosceles_triangle_rational_trig

end NUMINAMATH_CALUDE_isosceles_triangle_rational_trig_l1016_101649


namespace NUMINAMATH_CALUDE_cos_72_minus_cos_144_l1016_101694

theorem cos_72_minus_cos_144 : Real.cos (72 * π / 180) - Real.cos (144 * π / 180) = 1.117962 := by
  sorry

end NUMINAMATH_CALUDE_cos_72_minus_cos_144_l1016_101694


namespace NUMINAMATH_CALUDE_inequality_system_solution_range_l1016_101625

theorem inequality_system_solution_range (a : ℝ) : 
  (∃ x : ℝ, ax < 1 ∧ x - a < 0) → a ∈ Set.Ici (-1) :=
by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_range_l1016_101625


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_ratio_l1016_101600

/-- For a geometric sequence with common ratio 2, the ratio of the sum of the first 3 terms to the first term is 7 -/
theorem geometric_sequence_sum_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n, a (n + 1) = 2 * a n) →  -- Geometric sequence with common ratio 2
  (∀ n, S n = (a 1) * (1 - 2^n) / (1 - 2)) →  -- Sum formula for geometric sequence
  S 3 / a 1 = 7 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_ratio_l1016_101600


namespace NUMINAMATH_CALUDE_sum_zero_iff_fractions_sum_neg_two_l1016_101634

theorem sum_zero_iff_fractions_sum_neg_two (x y : ℝ) (h : x * y ≠ 0) :
  x + y = 0 ↔ x / y + y / x = -2 := by
  sorry

end NUMINAMATH_CALUDE_sum_zero_iff_fractions_sum_neg_two_l1016_101634


namespace NUMINAMATH_CALUDE_height_comparison_l1016_101659

theorem height_comparison (a b : ℝ) (h : a = b * (1 - 0.25)) :
  b = a * (1 + 1/3) :=
by sorry

end NUMINAMATH_CALUDE_height_comparison_l1016_101659


namespace NUMINAMATH_CALUDE_degree_of_product_l1016_101620

-- Define polynomials h and j
variable (h j : Polynomial ℝ)

-- Define the degrees of h and j
variable (deg_h : Polynomial.degree h = 3)
variable (deg_j : Polynomial.degree j = 5)

-- Theorem statement
theorem degree_of_product :
  Polynomial.degree (h.comp (Polynomial.X ^ 4) * j.comp (Polynomial.X ^ 3)) = 27 := by
  sorry

end NUMINAMATH_CALUDE_degree_of_product_l1016_101620


namespace NUMINAMATH_CALUDE_series_sum_l1016_101630

theorem series_sum : 
  let a : ℕ → ℚ := fun n => (4*n + 3) / ((4*n + 1)^2 * (4*n + 5)^2)
  ∑' n, a n = 1/200 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_l1016_101630


namespace NUMINAMATH_CALUDE_quadratic_always_positive_l1016_101626

theorem quadratic_always_positive (k : ℝ) : 
  (∀ x : ℝ, x^2 + k*x + 1 > 0) ↔ -2 < k ∧ k < 2 := by sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_l1016_101626


namespace NUMINAMATH_CALUDE_clock_angle_at_2_30_l1016_101695

/-- The number of hours on a standard analog clock -/
def clock_hours : ℕ := 12

/-- The number of degrees in a full rotation -/
def full_rotation : ℕ := 360

/-- The time in hours (including fractional part) -/
def time : ℚ := 2.5

/-- Calculates the angle of the hour hand from the 12 o'clock position -/
def hour_hand_angle (t : ℚ) : ℚ := (t * full_rotation) / clock_hours

/-- Calculates the angle of the minute hand from the 12 o'clock position -/
def minute_hand_angle (t : ℚ) : ℚ := ((t - t.floor) * full_rotation)

/-- Calculates the absolute difference between two angles -/
def angle_difference (a b : ℚ) : ℚ := min (abs (a - b)) (full_rotation - abs (a - b))

theorem clock_angle_at_2_30 :
  angle_difference (hour_hand_angle time) (minute_hand_angle time) = 105 := by
  sorry

end NUMINAMATH_CALUDE_clock_angle_at_2_30_l1016_101695


namespace NUMINAMATH_CALUDE_triangular_pyramid_angle_l1016_101686

/-- Represents a triangular pyramid with specific properties -/
structure TriangularPyramid where
  -- The length of the hypotenuse of the base triangle
  c : ℝ
  -- The volume of the pyramid
  V : ℝ
  -- All lateral edges form the same angle with the base plane
  lateral_angle_uniform : True
  -- This angle is equal to one of the acute angles of the right triangle in the base
  angle_matches_base : True
  -- Ensure c and V are positive
  c_pos : c > 0
  V_pos : V > 0

/-- 
Theorem: In a triangular pyramid where all lateral edges form the same angle α 
with the base plane, and this angle is equal to one of the acute angles of the 
right triangle in the base, if the hypotenuse of the base triangle is c and the 
volume of the pyramid is V, then α = arcsin(√(12V/c³)).
-/
theorem triangular_pyramid_angle (p : TriangularPyramid) : 
  ∃ α : ℝ, α = Real.arcsin (Real.sqrt (12 * p.V / p.c^3)) := by
  sorry

end NUMINAMATH_CALUDE_triangular_pyramid_angle_l1016_101686


namespace NUMINAMATH_CALUDE_area_of_triangle_abc_is_150_over_7_l1016_101635

/-- Given a circle with center O and radius r, and points A and B on a line passing through O,
    this function calculates the area of triangle ABC, where C is the intersection of tangents
    drawn from A and B to the circle. -/
def triangle_area_from_tangents (r OA AB : ℝ) : ℝ :=
  sorry

/-- Theorem stating that for a circle with radius 12 and points A and B such that OA = 15 and AB = 5,
    the area of triangle ABC formed by the intersection of tangents is 150/7. -/
theorem area_of_triangle_abc_is_150_over_7 :
  triangle_area_from_tangents 12 15 5 = 150 / 7 := by
  sorry

end NUMINAMATH_CALUDE_area_of_triangle_abc_is_150_over_7_l1016_101635


namespace NUMINAMATH_CALUDE_school_supplies_purchase_l1016_101689

-- Define the cost of one unit of type A and one unit of type B
def cost_A : ℝ := 15
def cost_B : ℝ := 25

-- Define the total number of units to be purchased
def total_units : ℕ := 100

-- Define the maximum total cost
def max_total_cost : ℝ := 2000

-- Theorem to prove
theorem school_supplies_purchase :
  -- Condition 1: The sum of costs of one unit of each type is $40
  cost_A + cost_B = 40 →
  -- Condition 2: The number of units of type A that can be purchased with $90 
  -- is the same as the number of units of type B that can be purchased with $150
  90 / cost_A = 150 / cost_B →
  -- Condition 3: The total cost should not exceed $2000
  ∀ y : ℕ, y ≤ total_units → cost_A * y + cost_B * (total_units - y) ≤ max_total_cost →
  -- Conclusion: The minimum number of units of type A to be purchased is 50
  (∀ z : ℕ, z < 50 → cost_A * z + cost_B * (total_units - z) > max_total_cost) ∧
  cost_A * 50 + cost_B * (total_units - 50) ≤ max_total_cost :=
by sorry


end NUMINAMATH_CALUDE_school_supplies_purchase_l1016_101689


namespace NUMINAMATH_CALUDE_prob_ace_then_king_standard_deck_l1016_101662

/-- A standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (ace_count : ℕ)
  (king_count : ℕ)

/-- The probability of drawing an Ace then a King from a standard deck -/
def prob_ace_then_king (d : Deck) : ℚ :=
  (d.ace_count : ℚ) / d.total_cards * (d.king_count : ℚ) / (d.total_cards - 1)

/-- A standard 52-card deck -/
def standard_deck : Deck :=
  { total_cards := 52
  , ace_count := 4
  , king_count := 4 }

/-- Theorem: The probability of drawing an Ace then a King from a standard 52-card deck is 4/663 -/
theorem prob_ace_then_king_standard_deck :
  prob_ace_then_king standard_deck = 4 / 663 := by
  sorry

end NUMINAMATH_CALUDE_prob_ace_then_king_standard_deck_l1016_101662


namespace NUMINAMATH_CALUDE_six_people_round_table_one_reserved_l1016_101636

/-- The number of ways to arrange people around a round table --/
def roundTableArrangements (n : ℕ) (reserved : ℕ) : ℕ :=
  Nat.factorial (n - reserved)

/-- Theorem: 6 people around a round table with 1 reserved seat --/
theorem six_people_round_table_one_reserved :
  roundTableArrangements 6 1 = 120 := by
  sorry

end NUMINAMATH_CALUDE_six_people_round_table_one_reserved_l1016_101636


namespace NUMINAMATH_CALUDE_daves_hourly_wage_l1016_101607

/-- Dave's hourly wage calculation --/
theorem daves_hourly_wage (monday_hours tuesday_hours total_amount : ℕ) 
  (h1 : monday_hours = 6)
  (h2 : tuesday_hours = 2)
  (h3 : total_amount = 48) :
  total_amount / (monday_hours + tuesday_hours) = 6 := by
  sorry

end NUMINAMATH_CALUDE_daves_hourly_wage_l1016_101607


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_squares_l1016_101652

theorem arithmetic_geometric_mean_squares (a b : ℝ) 
  (h_arithmetic : (a + b) / 2 = 20)
  (h_geometric : Real.sqrt (a * b) = 10) : 
  a^2 + b^2 = 1400 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_squares_l1016_101652


namespace NUMINAMATH_CALUDE_smallest_prime_factor_of_1739_l1016_101682

theorem smallest_prime_factor_of_1739 : Nat.Prime 1739 := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_of_1739_l1016_101682


namespace NUMINAMATH_CALUDE_function_satisfying_conditions_l1016_101678

theorem function_satisfying_conditions (f : ℝ → ℝ) 
  (h1 : ∀ x, f (f x * f (1 - x)) = f x) 
  (h2 : ∀ x, f (f x) = 1 - f x) : 
  ∀ x, f x = (1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_function_satisfying_conditions_l1016_101678


namespace NUMINAMATH_CALUDE_circle_equation_l1016_101697

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A circle in 2D space -/
structure Circle where
  center : Point2D
  radius : ℝ

/-- Function to check if a point is on a circle -/
def isOnCircle (p : Point2D) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

/-- Function to check if two points are symmetric with respect to y = x -/
def isSymmetricYEqX (p1 p2 : Point2D) : Prop :=
  p1.x = p2.y ∧ p1.y = p2.x

/-- Theorem: Given a circle C with radius 1 and center symmetric to (1, 0) 
    with respect to the line y = x, its standard equation is x^2 + (y - 1)^2 = 1 -/
theorem circle_equation (C : Circle) 
    (h1 : C.radius = 1)
    (h2 : isSymmetricYEqX C.center ⟨1, 0⟩) : 
    ∀ (p : Point2D), isOnCircle p C ↔ p.x^2 + (p.y - 1)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_l1016_101697


namespace NUMINAMATH_CALUDE_is_focus_of_hyperbola_l1016_101624

/-- The equation of the hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 - 3*y^2 + 6*x - 12*y - 8 = 0

/-- The focus point -/
def focus : ℝ × ℝ := (-1, -2)

/-- Theorem stating that the given point is a focus of the hyperbola -/
theorem is_focus_of_hyperbola : 
  ∃ (c : ℝ), c > 0 ∧ 
  ∀ (x y : ℝ), hyperbola_equation x y → 
    (x + 1)^2 + (y + 2)^2 - ((x + 5)^2 + (y + 2)^2) = 4*c := by
  sorry

end NUMINAMATH_CALUDE_is_focus_of_hyperbola_l1016_101624

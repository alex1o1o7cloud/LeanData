import Mathlib

namespace NUMINAMATH_CALUDE_square_root_of_sixteen_l424_42441

theorem square_root_of_sixteen : 
  ∃ (x : ℝ), x^2 = 16 ∧ (x = 4 ∨ x = -4) :=
sorry

end NUMINAMATH_CALUDE_square_root_of_sixteen_l424_42441


namespace NUMINAMATH_CALUDE_cos_450_degrees_l424_42437

theorem cos_450_degrees (h1 : ∀ x, Real.cos (x + 2 * Real.pi) = Real.cos x)
                         (h2 : Real.cos (Real.pi / 2) = 0) : 
  Real.cos (5 * Real.pi / 2) = 0 := by
sorry

end NUMINAMATH_CALUDE_cos_450_degrees_l424_42437


namespace NUMINAMATH_CALUDE_tv_set_selection_count_l424_42476

def total_sets : ℕ := 9
def type_a_sets : ℕ := 4
def type_b_sets : ℕ := 5
def sets_to_select : ℕ := 3

theorem tv_set_selection_count :
  (Nat.choose total_sets sets_to_select) -
  (Nat.choose type_a_sets sets_to_select) -
  (Nat.choose type_b_sets sets_to_select) = 70 := by
  sorry

end NUMINAMATH_CALUDE_tv_set_selection_count_l424_42476


namespace NUMINAMATH_CALUDE_runner_stops_at_d_l424_42494

/-- Represents the quarters of the circular track -/
inductive Quarter : Type
  | A : Quarter
  | B : Quarter
  | C : Quarter
  | D : Quarter

/-- Represents a point on the circular track -/
structure TrackPoint where
  position : ℝ  -- position in feet from the start point
  quarter : Quarter

/-- The circular track -/
structure Track where
  circumference : ℝ
  start_point : TrackPoint

/-- Calculates the final position after running a given distance -/
def final_position (track : Track) (distance : ℝ) : TrackPoint :=
  sorry

theorem runner_stops_at_d (track : Track) (distance : ℝ) :
  track.circumference = 100 →
  distance = 10000 →
  track.start_point.quarter = Quarter.A →
  (final_position track distance).quarter = Quarter.D :=
sorry

end NUMINAMATH_CALUDE_runner_stops_at_d_l424_42494


namespace NUMINAMATH_CALUDE_carrots_theorem_l424_42489

/-- The number of carrots Sandy grew -/
def sandy_carrots : ℕ := 8

/-- The number of carrots Mary grew -/
def mary_carrots : ℕ := 6

/-- The total number of carrots grown by Sandy and Mary -/
def total_carrots : ℕ := sandy_carrots + mary_carrots

theorem carrots_theorem : total_carrots = 14 := by
  sorry

end NUMINAMATH_CALUDE_carrots_theorem_l424_42489


namespace NUMINAMATH_CALUDE_beckett_olaf_age_difference_l424_42401

/-- Given the ages of four people satisfying certain conditions, prove that Beckett is 8 years younger than Olaf. -/
theorem beckett_olaf_age_difference :
  ∀ (beckett_age olaf_age shannen_age jack_age : ℕ),
    beckett_age = 12 →
    ∃ (x : ℕ), beckett_age + x = olaf_age →
    shannen_age + 2 = olaf_age →
    jack_age = 2 * shannen_age + 5 →
    beckett_age + olaf_age + shannen_age + jack_age = 71 →
    x = 8 := by
  sorry

end NUMINAMATH_CALUDE_beckett_olaf_age_difference_l424_42401


namespace NUMINAMATH_CALUDE_smallest_b_for_quadratic_inequality_l424_42402

theorem smallest_b_for_quadratic_inequality :
  ∀ b : ℝ, b^2 - 16*b + 63 ≤ 0 → b ≥ 7 :=
by sorry

end NUMINAMATH_CALUDE_smallest_b_for_quadratic_inequality_l424_42402


namespace NUMINAMATH_CALUDE_intersection_complement_equal_l424_42446

def U : Set Int := Set.univ

def A : Set Int := {-2, -1, 1, 2}

def B : Set Int := {1, 2}

theorem intersection_complement_equal : A ∩ (Set.compl B) = {-2, -1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equal_l424_42446


namespace NUMINAMATH_CALUDE_officers_count_l424_42413

/-- The number of ways to choose 5 distinct officers from a group of 12 people -/
def choose_officers : ℕ := 12 * 11 * 10 * 9 * 8

/-- Theorem stating that the number of ways to choose 5 distinct officers 
    from a group of 12 people is 95040 -/
theorem officers_count : choose_officers = 95040 := by
  sorry

end NUMINAMATH_CALUDE_officers_count_l424_42413


namespace NUMINAMATH_CALUDE_randys_house_blocks_l424_42466

/-- Given Randy's block building scenario, prove the number of blocks used for the house. -/
theorem randys_house_blocks (total : ℕ) (tower : ℕ) (difference : ℕ) (house : ℕ) : 
  total = 90 → tower = 63 → difference = 26 → house = tower + difference → house = 89 := by
  sorry

end NUMINAMATH_CALUDE_randys_house_blocks_l424_42466


namespace NUMINAMATH_CALUDE_sarah_pencils_count_l424_42484

/-- The number of pencils Sarah buys on Monday -/
def monday_pencils : ℕ := 20

/-- The number of pencils Sarah buys on Tuesday -/
def tuesday_pencils : ℕ := 18

/-- The number of pencils Sarah buys on Wednesday -/
def wednesday_pencils : ℕ := 3 * tuesday_pencils

/-- The total number of pencils Sarah has -/
def total_pencils : ℕ := monday_pencils + tuesday_pencils + wednesday_pencils

theorem sarah_pencils_count : total_pencils = 92 := by
  sorry

end NUMINAMATH_CALUDE_sarah_pencils_count_l424_42484


namespace NUMINAMATH_CALUDE_distribute_four_into_two_l424_42467

/-- The number of ways to distribute n distinguishable objects into k distinguishable containers -/
def distribute (n k : ℕ) : ℕ := k^n

/-- Theorem: There are 16 ways to distribute 4 distinguishable balls into 2 distinguishable boxes -/
theorem distribute_four_into_two : distribute 4 2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_distribute_four_into_two_l424_42467


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l424_42481

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 12) :
  let side := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
  4 * side = 4 * Real.sqrt 61 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l424_42481


namespace NUMINAMATH_CALUDE_sum_triangle_quadrilateral_sides_l424_42405

/-- A triangle is a shape with 3 sides -/
def Triangle : Nat := 3

/-- A quadrilateral is a shape with 4 sides -/
def Quadrilateral : Nat := 4

/-- The sum of the sides of a triangle and a quadrilateral is 7 -/
theorem sum_triangle_quadrilateral_sides : Triangle + Quadrilateral = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_triangle_quadrilateral_sides_l424_42405


namespace NUMINAMATH_CALUDE_outermost_ring_count_9x9_l424_42469

/-- Represents a square grid with alternating circles and rhombuses -/
structure AlternatingGrid (n : ℕ) where
  size : ℕ
  size_pos : size > 0
  is_square : ∃ k : ℕ, size = k * k

/-- The number of elements in the outermost ring of an AlternatingGrid -/
def outermost_ring_count (grid : AlternatingGrid n) : ℕ :=
  4 * (grid.size - 1)

/-- Theorem: The number of elements in the outermost ring of a 9x9 AlternatingGrid is 81 -/
theorem outermost_ring_count_9x9 :
  ∀ (grid : AlternatingGrid 9), grid.size = 9 → outermost_ring_count grid = 81 :=
by
  sorry


end NUMINAMATH_CALUDE_outermost_ring_count_9x9_l424_42469


namespace NUMINAMATH_CALUDE_original_area_l424_42480

/-- In an oblique dimetric projection, given a regular triangle as the intuitive diagram -/
structure ObliqueTriangle where
  /-- Side length of the intuitive diagram -/
  side_length : ℝ
  /-- Area ratio of original to intuitive -/
  area_ratio : ℝ
  /-- Side length is positive -/
  side_length_pos : 0 < side_length
  /-- Area ratio is positive -/
  area_ratio_pos : 0 < area_ratio

/-- Theorem: Area of the original figure in oblique dimetric projection -/
theorem original_area (t : ObliqueTriangle) (h1 : t.side_length = 2) (h2 : t.area_ratio = 2 * Real.sqrt 2) :
  ∃ (area : ℝ), area = 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_original_area_l424_42480


namespace NUMINAMATH_CALUDE_min_bullseyes_for_victory_l424_42432

/-- Represents the archery competition scenario -/
structure ArcheryCompetition where
  total_shots : ℕ := 120
  halfway_point : ℕ := 60
  alex_lead_at_half : ℕ := 60
  bullseye_score : ℕ := 10
  alex_min_score : ℕ := 3

/-- Theorem stating the minimum number of bullseyes Alex needs to guarantee victory -/
theorem min_bullseyes_for_victory (comp : ArcheryCompetition) :
  ∃ n : ℕ, n = 52 ∧
  (∀ m : ℕ, -- m represents Alex's current score
    (comp.alex_lead_at_half + m = comp.halfway_point * comp.alex_min_score) →
    (m + n * comp.bullseye_score + (comp.halfway_point - n) * comp.alex_min_score >
     m - comp.alex_lead_at_half + comp.halfway_point * comp.bullseye_score) ∧
    (∀ k : ℕ, k < n →
      ∃ p : ℕ, p ≤ m - comp.alex_lead_at_half + comp.halfway_point * comp.bullseye_score ∧
      p ≥ m + k * comp.bullseye_score + (comp.halfway_point - k) * comp.alex_min_score)) :=
sorry

end NUMINAMATH_CALUDE_min_bullseyes_for_victory_l424_42432


namespace NUMINAMATH_CALUDE_detect_non_conforming_probability_l424_42472

/-- The number of cans in a box -/
def total_cans : ℕ := 5

/-- The number of non-conforming cans in the box -/
def non_conforming_cans : ℕ := 2

/-- The number of cans selected for testing -/
def selected_cans : ℕ := 2

/-- The probability of detecting at least one non-conforming product -/
def probability_detect : ℚ := 7 / 10

theorem detect_non_conforming_probability :
  probability_detect = (Nat.choose non_conforming_cans 1 * Nat.choose (total_cans - non_conforming_cans) 1 + 
                        Nat.choose non_conforming_cans 2) / 
                       Nat.choose total_cans selected_cans :=
by sorry

end NUMINAMATH_CALUDE_detect_non_conforming_probability_l424_42472


namespace NUMINAMATH_CALUDE_sum_congruence_l424_42454

theorem sum_congruence : (1 + 22 + 333 + 4444 + 55555 + 666666 + 7777777 + 88888888) % 9 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_congruence_l424_42454


namespace NUMINAMATH_CALUDE_expression_evaluation_l424_42460

theorem expression_evaluation : 
  3 + Real.sqrt 3 + (1 / (3 + Real.sqrt 3)) + (1 / (Real.sqrt 3 - 3)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l424_42460


namespace NUMINAMATH_CALUDE_sum_xyz_equality_l424_42448

theorem sum_xyz_equality (x y z : ℝ) 
  (h1 : x^2 + y^2 + z^2 = 1) 
  (h2 : x + 2*y + 3*z = Real.sqrt 14) : 
  x + y + z = (3 * Real.sqrt 14) / 7 := by
sorry

end NUMINAMATH_CALUDE_sum_xyz_equality_l424_42448


namespace NUMINAMATH_CALUDE_relationship_abc_l424_42445

theorem relationship_abc :
  let a : ℤ := -2 * 3^2
  let b : ℤ := (-2 * 3)^2
  let c : ℤ := -(2 * 3)^2
  b > a ∧ a > c := by sorry

end NUMINAMATH_CALUDE_relationship_abc_l424_42445


namespace NUMINAMATH_CALUDE_dice_sides_proof_l424_42412

theorem dice_sides_proof (n : ℕ) (h : n ≥ 3) :
  (3 / n^2 : ℚ)^2 = 1/9 → n = 3 :=
by sorry

end NUMINAMATH_CALUDE_dice_sides_proof_l424_42412


namespace NUMINAMATH_CALUDE_book_sale_pricing_l424_42458

theorem book_sale_pricing (total_books : ℕ) (higher_price lower_price total_earnings : ℚ) :
  total_books = 10 →
  lower_price = 2 →
  total_earnings = 22 →
  (2 / 5 : ℚ) * total_books * higher_price + (3 / 5 : ℚ) * total_books * lower_price = total_earnings →
  higher_price = (5 / 2 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_book_sale_pricing_l424_42458


namespace NUMINAMATH_CALUDE_student_group_problem_first_group_size_l424_42408

theorem student_group_problem (x : ℕ) : 
  x * x + (x + 5) * (x + 5) = 13000 → x = 78 := by sorry

theorem first_group_size (x : ℕ) :
  x * x + (x + 5) * (x + 5) = 13000 → x + 5 = 83 := by sorry

end NUMINAMATH_CALUDE_student_group_problem_first_group_size_l424_42408


namespace NUMINAMATH_CALUDE_odd_guess_probability_l424_42420

theorem odd_guess_probability (n : ℕ) (hn : n = 2002) :
  (n - n / 3 : ℚ) / n > 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_odd_guess_probability_l424_42420


namespace NUMINAMATH_CALUDE_test_score_for_three_hours_l424_42464

/-- A model for a test score based on preparation time. -/
structure TestScore where
  maxPoints : ℝ
  scoreFunction : ℝ → ℝ
  knownScore : ℝ
  knownTime : ℝ

/-- Theorem: Given the conditions, prove that 3 hours of preparation results in a score of 202.5 -/
theorem test_score_for_three_hours 
  (test : TestScore)
  (h1 : test.maxPoints = 150)
  (h2 : ∀ t, test.scoreFunction t = (test.knownScore / test.knownTime^2) * t^2)
  (h3 : test.knownScore = 90)
  (h4 : test.knownTime = 2) :
  test.scoreFunction 3 = 202.5 := by
  sorry


end NUMINAMATH_CALUDE_test_score_for_three_hours_l424_42464


namespace NUMINAMATH_CALUDE_two_distinct_roots_l424_42410

-- Define the function representing the equation
def f (x p : ℝ) : ℝ := x^2 - 2*|x| - p

-- State the theorem
theorem two_distinct_roots (p : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ f x p = 0 ∧ f y p = 0) ↔ p > -1 :=
sorry

end NUMINAMATH_CALUDE_two_distinct_roots_l424_42410


namespace NUMINAMATH_CALUDE_arccos_gt_arctan_iff_l424_42457

/-- The theorem states that for all real numbers x, arccos x is greater than arctan x
    if and only if x is in the interval [-1, 1/√3), given that arccos x is defined for x in [-1,1]. -/
theorem arccos_gt_arctan_iff (x : ℝ) : 
  Real.arccos x > Real.arctan x ↔ x ∈ Set.Icc (-1 : ℝ) (1 / Real.sqrt 3) ∧ x ≠ 1 / Real.sqrt 3 := by
  sorry

/-- This definition ensures that arccos is only defined on [-1, 1] -/
def arccos_domain (x : ℝ) : Prop := x ∈ Set.Icc (-1 : ℝ) 1

end NUMINAMATH_CALUDE_arccos_gt_arctan_iff_l424_42457


namespace NUMINAMATH_CALUDE_length_AE_l424_42487

/-- Represents a point on a line -/
structure Point where
  x : ℝ

/-- Calculates the distance between two points -/
def distance (p q : Point) : ℝ := abs (p.x - q.x)

/-- Theorem: Length of AE given specific conditions -/
theorem length_AE (a b c d e : Point) 
  (consecutive : a.x < b.x ∧ b.x < c.x ∧ c.x < d.x ∧ d.x < e.x)
  (bc_eq_3cd : distance b c = 3 * distance c d)
  (de_eq_7 : distance d e = 7)
  (ab_eq_5 : distance a b = 5)
  (ac_eq_11 : distance a c = 11) :
  distance a e = 18 := by
  sorry

end NUMINAMATH_CALUDE_length_AE_l424_42487


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_positive_n_for_unique_solution_l424_42497

/-- For a quadratic equation ax^2 + bx + c = 0, its discriminant is b^2 - 4ac --/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- A quadratic equation has exactly one solution if and only if its discriminant is zero --/
def has_exactly_one_solution (a b c : ℝ) : Prop :=
  discriminant a b c = 0

theorem unique_solution_quadratic (n : ℝ) :
  has_exactly_one_solution 4 n 16 ↔ n = 16 ∨ n = -16 :=
sorry

theorem positive_n_for_unique_solution :
  ∃ n : ℝ, n > 0 ∧ has_exactly_one_solution 4 n 16 ∧ n = 16 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_positive_n_for_unique_solution_l424_42497


namespace NUMINAMATH_CALUDE_equation_solution_l424_42499

theorem equation_solution : ∃ x : ℝ, (3 / (x - 1) = 5 + 3 * x / (1 - x)) ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l424_42499


namespace NUMINAMATH_CALUDE_combined_area_rhombus_circle_l424_42406

/-- The combined area of a rhombus and a circle -/
theorem combined_area_rhombus_circle (d1 d2 r : ℝ) (h1 : d1 = 40) (h2 : d2 = 30) (h3 : r = 10) :
  (d1 * d2 / 2) + (π * r^2) = 600 + 100 * π := by
  sorry

end NUMINAMATH_CALUDE_combined_area_rhombus_circle_l424_42406


namespace NUMINAMATH_CALUDE_binomial_12_3_l424_42434

theorem binomial_12_3 : Nat.choose 12 3 = 220 := by
  sorry

end NUMINAMATH_CALUDE_binomial_12_3_l424_42434


namespace NUMINAMATH_CALUDE_parallelogram_area_l424_42491

/-- The area of a parallelogram with one angle of 150 degrees and two consecutive sides of lengths 10 and 20 is 100√3. -/
theorem parallelogram_area (a b : ℝ) (θ : ℝ) (h1 : a = 10) (h2 : b = 20) (h3 : θ = 150 * π / 180) :
  a * b * Real.sin (π - θ) = 100 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l424_42491


namespace NUMINAMATH_CALUDE_smallest_n_for_rectangle_l424_42479

/-- A function that checks if it's possible to form a rectangle with given pieces --/
def can_form_rectangle (pieces : List Nat) : Prop :=
  ∃ (w h : Nat), w * 2 + h * 2 = pieces.sum ∧ w > 0 ∧ h > 0

/-- The main theorem stating that 102 is the smallest N that satisfies the conditions --/
theorem smallest_n_for_rectangle : 
  (∀ n < 102, ¬∃ (pieces : List Nat), 
    pieces.length = n ∧ 
    pieces.sum = 200 ∧ 
    (∀ p ∈ pieces, p > 0) ∧
    can_form_rectangle pieces) ∧
  (∃ (pieces : List Nat), 
    pieces.length = 102 ∧ 
    pieces.sum = 200 ∧ 
    (∀ p ∈ pieces, p > 0) ∧
    can_form_rectangle pieces) :=
by sorry

#check smallest_n_for_rectangle

end NUMINAMATH_CALUDE_smallest_n_for_rectangle_l424_42479


namespace NUMINAMATH_CALUDE_four_digit_number_with_specific_factors_l424_42443

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def count_prime_factors (n : ℕ) : ℕ := sorry
def count_non_prime_factors (n : ℕ) : ℕ := sorry
def count_total_factors (n : ℕ) : ℕ := sorry

theorem four_digit_number_with_specific_factors :
  ∃ (n : ℕ), is_four_digit n ∧ 
             count_prime_factors n = 3 ∧ 
             count_non_prime_factors n = 39 ∧ 
             count_total_factors n = 42 :=
sorry

end NUMINAMATH_CALUDE_four_digit_number_with_specific_factors_l424_42443


namespace NUMINAMATH_CALUDE_annie_brownies_l424_42444

def brownies_problem (total : ℕ) : Prop :=
  let after_admin : ℕ := total / 2
  let after_carl : ℕ := after_admin / 2
  let after_simon : ℕ := after_carl - 2
  (after_simon = 3) ∧ (total > 0)

theorem annie_brownies : ∃ (total : ℕ), brownies_problem total ∧ total = 20 := by
  sorry

end NUMINAMATH_CALUDE_annie_brownies_l424_42444


namespace NUMINAMATH_CALUDE_correct_quotient_proof_l424_42425

theorem correct_quotient_proof (N : ℕ) : 
  N % 21 = 0 →  -- remainder is 0 when divided by 21
  N / 12 = 56 → -- quotient is 56 when divided by 12
  N / 21 = 32   -- correct quotient when divided by 21
:= by sorry

end NUMINAMATH_CALUDE_correct_quotient_proof_l424_42425


namespace NUMINAMATH_CALUDE_intersection_distance_l424_42427

/-- The distance between the points of intersection of three lines -/
theorem intersection_distance (x₁ x₂ : ℝ) (y : ℝ) : 
  x₁ = 1975 / 3 ∧ 
  x₂ = 1981 / 3 ∧ 
  y = 1975 ∧ 
  (3 * x₁ = y) ∧ 
  (3 * x₂ - 6 = y) →
  Real.sqrt ((x₂ - x₁)^2 + (y - y)^2) = 2 := by
  sorry

#check intersection_distance

end NUMINAMATH_CALUDE_intersection_distance_l424_42427


namespace NUMINAMATH_CALUDE_correct_operation_l424_42493

theorem correct_operation (a b : ℝ) : 3 * a^2 * b - 3 * b * a^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l424_42493


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l424_42470

/-- 
Given an arithmetic sequence {a_n} where the first three terms are a-1, a-1, and 2a+3,
prove that the general term formula is a_n = 2n-3.
-/
theorem arithmetic_sequence_general_term 
  (a_n : ℕ → ℝ) 
  (a : ℝ) 
  (h1 : a_n 1 = a - 1) 
  (h2 : a_n 2 = a - 1) 
  (h3 : a_n 3 = 2*a + 3) 
  (h_arithmetic : ∀ n : ℕ, a_n (n + 1) - a_n n = a_n (n + 2) - a_n (n + 1)) :
  ∀ n : ℕ, a_n n = 2*n - 3 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l424_42470


namespace NUMINAMATH_CALUDE_larger_number_proof_l424_42433

theorem larger_number_proof (L S : ℕ) (hL : L > S) :
  L - S = 1365 → L = 6 * S + 20 → L = 1634 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l424_42433


namespace NUMINAMATH_CALUDE_min_baking_time_three_cakes_l424_42452

/-- Represents a cake that needs to be baked on both sides -/
structure Cake where
  side1_baked : Bool
  side2_baked : Bool

/-- Represents a baking pan that can hold up to two cakes -/
structure Pan where
  capacity : Nat
  current_cakes : List Cake

/-- The time it takes to bake one side of a cake -/
def bake_time : Nat := 1

/-- The function to calculate the minimum baking time for all cakes -/
def min_baking_time (cakes : List Cake) (pan : Pan) : Nat :=
  sorry

/-- Theorem stating that the minimum baking time for three cakes is 3 minutes -/
theorem min_baking_time_three_cakes :
  let cakes := [Cake.mk false false, Cake.mk false false, Cake.mk false false]
  let pan := Pan.mk 2 []
  min_baking_time cakes pan = 3 := by
  sorry

end NUMINAMATH_CALUDE_min_baking_time_three_cakes_l424_42452


namespace NUMINAMATH_CALUDE_quadratic_roots_problem_l424_42430

theorem quadratic_roots_problem (a : ℝ) (x₁ x₂ : ℝ) : 
  (∃ (x : ℝ), x^2 + a*x + 6 = 0) ∧  -- equation has real roots
  (x₁ ≠ x₂) ∧  -- roots are distinct
  (x₁^2 + a*x₁ + 6 = 0) ∧  -- x₁ is a root
  (x₂^2 + a*x₂ + 6 = 0) ∧  -- x₂ is a root
  (x₁ - 72 / (25 * x₂^3) = x₂ - 72 / (25 * x₁^3)) -- given condition
  → 
  a = 9 ∨ a = -9 :=
by sorry


end NUMINAMATH_CALUDE_quadratic_roots_problem_l424_42430


namespace NUMINAMATH_CALUDE_unique_solution_club_l424_42478

/-- The ♣ operation -/
def club (A B : ℝ) : ℝ := 3 * A + 2 * B + 5

/-- Theorem stating that 21 is the unique solution to A ♣ 7 = 82 -/
theorem unique_solution_club : ∃! A : ℝ, club A 7 = 82 ∧ A = 21 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_club_l424_42478


namespace NUMINAMATH_CALUDE_unique_integer_solution_l424_42414

theorem unique_integer_solution : ∃! (d e f : ℕ+), 
  let x : ℝ := Real.sqrt ((Real.sqrt 77 / 2) + (5 / 2))
  x^100 = 3*x^98 + 18*x^96 + 13*x^94 - x^50 + d*x^46 + e*x^44 + f*x^40 ∧ 
  d + e + f = 86 := by sorry

end NUMINAMATH_CALUDE_unique_integer_solution_l424_42414


namespace NUMINAMATH_CALUDE_twenty_students_no_math_l424_42498

/-- Represents a class of students with information about their subject choices. -/
structure ClassInfo where
  total : ℕ
  no_science : ℕ
  no_either : ℕ
  both : ℕ

/-- Calculates the number of students who didn't opt for math. -/
def students_no_math (info : ClassInfo) : ℕ :=
  info.total - info.both - (info.no_science - info.no_either)

/-- Theorem stating that in a specific class, 20 students didn't opt for math. -/
theorem twenty_students_no_math :
  let info : ClassInfo := {
    total := 40,
    no_science := 15,
    no_either := 2,
    both := 7
  }
  students_no_math info = 20 := by
  sorry


end NUMINAMATH_CALUDE_twenty_students_no_math_l424_42498


namespace NUMINAMATH_CALUDE_tangent_line_at_zero_derivative_monotone_increasing_f_superadditive_l424_42403

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.log (x + 1)

theorem tangent_line_at_zero (x : ℝ) :
  (deriv f) 0 = 1 :=
sorry

theorem derivative_monotone_increasing :
  StrictMonoOn (deriv f) (Set.Icc 0 2) :=
sorry

theorem f_superadditive {s t : ℝ} (hs : s > 0) (ht : t > 0) :
  f (s + t) > f s + f t :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_zero_derivative_monotone_increasing_f_superadditive_l424_42403


namespace NUMINAMATH_CALUDE_smallest_whole_number_above_sum_l424_42422

theorem smallest_whole_number_above_sum : ∃ n : ℕ, 
  (n : ℝ) > (4 + 1/2 : ℝ) + (6 + 1/3 : ℝ) + (8 + 1/4 : ℝ) + (10 + 1/5 : ℝ) ∧ 
  ∀ m : ℕ, (m : ℝ) > (4 + 1/2 : ℝ) + (6 + 1/3 : ℝ) + (8 + 1/4 : ℝ) + (10 + 1/5 : ℝ) → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_whole_number_above_sum_l424_42422


namespace NUMINAMATH_CALUDE_yard_length_26_trees_l424_42440

/-- The length of a yard with equally spaced trees -/
def yardLength (n : ℕ) (d : ℝ) : ℝ := (n - 1 : ℝ) * d

/-- Theorem: The length of a yard with 26 equally spaced trees, 
    one at each end, and 12 meters between consecutive trees, is 300 meters. -/
theorem yard_length_26_trees : yardLength 26 12 = 300 := by
  sorry

end NUMINAMATH_CALUDE_yard_length_26_trees_l424_42440


namespace NUMINAMATH_CALUDE_blood_cell_count_l424_42418

theorem blood_cell_count (sample1 sample2 : ℕ) 
  (h1 : sample1 = 4221) 
  (h2 : sample2 = 3120) : 
  sample1 + sample2 = 7341 := by
sorry

end NUMINAMATH_CALUDE_blood_cell_count_l424_42418


namespace NUMINAMATH_CALUDE_square_root_of_sixteen_l424_42485

theorem square_root_of_sixteen : Real.sqrt 16 = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_sixteen_l424_42485


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_equation_l424_42453

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^2 + b * x + c
  (∃ x₁ x₂, f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ ≠ x₂) →
  (∃ s, s = x₁ + x₂ ∧ s = -b / a) :=
sorry

theorem sum_of_roots_specific_equation :
  let f : ℝ → ℝ := λ x => x^2 + 2010*x - (2011 + 18*x)
  (∃ x₁ x₂, f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ ≠ x₂) →
  (∃ s, s = x₁ + x₂ ∧ s = -1992) :=
sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_equation_l424_42453


namespace NUMINAMATH_CALUDE_largest_solution_of_equation_l424_42475

theorem largest_solution_of_equation :
  ∃ (x : ℚ), x = -10/9 ∧ 
  5*(9*x^2 + 9*x + 10) = x*(9*x - 40) ∧
  ∀ (y : ℚ), 5*(9*y^2 + 9*y + 10) = y*(9*y - 40) → y ≤ x :=
by sorry

end NUMINAMATH_CALUDE_largest_solution_of_equation_l424_42475


namespace NUMINAMATH_CALUDE_division_problem_l424_42492

theorem division_problem (R Q D : ℕ) : 
  D = 3 * Q ∧ 
  D = 3 * R + 3 ∧ 
  113 = D * Q + R → 
  R = 5 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l424_42492


namespace NUMINAMATH_CALUDE_range_of_m_l424_42417

-- Define propositions P and Q as functions of m
def P (m : ℝ) : Prop := ∃ x y : ℝ, x^2 / m^2 + y^2 / (2*m + 8) = 1 ∧ 
  ∃ c : ℝ, c > 0 ∧ x^2 / m^2 - y^2 / (2*m + 8) = c

def Q (m : ℝ) : Prop := ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
  x₁^2 + (2*m - 3)*x₁ + 1/4 = 0 ∧ 
  x₂^2 + (2*m - 3)*x₂ + 1/4 = 0

-- Define the theorem
theorem range_of_m : 
  (∀ m : ℝ, (P m ∨ Q m) ∧ ¬(P m ∧ Q m)) → 
  (∀ m : ℝ, m ≤ -4 ∨ (-2 ≤ m ∧ m < 1) ∨ (2 < m ∧ m ≤ 4)) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l424_42417


namespace NUMINAMATH_CALUDE_power_of_power_l424_42421

theorem power_of_power (a : ℝ) : (a^2)^3 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l424_42421


namespace NUMINAMATH_CALUDE_g_neg_one_eq_zero_l424_42416

/-- The function g(x) as defined in the problem -/
def g (s : ℝ) (x : ℝ) : ℝ := 3 * x^4 - 2 * x^3 + 4 * x^2 - 5 * x + s

/-- Theorem stating that g(-1) = 0 when s = -14 -/
theorem g_neg_one_eq_zero :
  g (-14) (-1) = 0 := by sorry

end NUMINAMATH_CALUDE_g_neg_one_eq_zero_l424_42416


namespace NUMINAMATH_CALUDE_asterisk_replacement_l424_42409

theorem asterisk_replacement : ∃! (x : ℝ), x > 0 ∧ (x / 21) * (x / 189) = 1 := by sorry

end NUMINAMATH_CALUDE_asterisk_replacement_l424_42409


namespace NUMINAMATH_CALUDE_janets_employees_work_hours_l424_42438

/-- Represents the problem of calculating work hours for Janet's employees --/
theorem janets_employees_work_hours :
  let warehouse_workers : ℕ := 4
  let managers : ℕ := 2
  let warehouse_wage : ℚ := 15
  let manager_wage : ℚ := 20
  let fica_tax_rate : ℚ := (1 / 10 : ℚ)
  let days_per_month : ℕ := 25
  let total_monthly_cost : ℚ := 22000

  ∃ (hours_per_day : ℚ),
    (warehouse_workers * warehouse_wage * hours_per_day * days_per_month +
     managers * manager_wage * hours_per_day * days_per_month) * (1 + fica_tax_rate) = total_monthly_cost ∧
    hours_per_day = 8 := by
  sorry

end NUMINAMATH_CALUDE_janets_employees_work_hours_l424_42438


namespace NUMINAMATH_CALUDE_roots_distinct_and_sum_integer_l424_42465

/-- Given that a, b, c are roots of x^3 - x^2 - x - 1 = 0, prove they are distinct and
    that (a^1982 - b^1982)/(a - b) + (b^1982 - c^1982)/(b - c) + (c^1982 - a^1982)/(c - a) is an integer -/
theorem roots_distinct_and_sum_integer (a b c : ℂ) : 
  (a^3 - a^2 - a - 1 = 0) → 
  (b^3 - b^2 - b - 1 = 0) → 
  (c^3 - c^2 - c - 1 = 0) → 
  (a ≠ b ∧ b ≠ c ∧ c ≠ a) ∧ 
  (∃ k : ℤ, (a^1982 - b^1982)/(a - b) + (b^1982 - c^1982)/(b - c) + (c^1982 - a^1982)/(c - a) = k) := by
  sorry

end NUMINAMATH_CALUDE_roots_distinct_and_sum_integer_l424_42465


namespace NUMINAMATH_CALUDE_volume_after_density_change_l424_42463

/-- Given a substance with initial density and a density change factor, 
    calculate the new volume of a specified mass. -/
theorem volume_after_density_change 
  (initial_mass : ℝ) 
  (initial_volume : ℝ) 
  (density_change_factor : ℝ) 
  (mass_to_calculate : ℝ) 
  (h1 : initial_mass > 0)
  (h2 : initial_volume > 0)
  (h3 : density_change_factor > 0)
  (h4 : mass_to_calculate > 0)
  (h5 : initial_mass = 500)
  (h6 : initial_volume = 1)
  (h7 : density_change_factor = 1.25)
  (h8 : mass_to_calculate = 0.001) : 
  (mass_to_calculate / (initial_mass / initial_volume * density_change_factor)) * 1000000 = 1.6 := by
  sorry

#check volume_after_density_change

end NUMINAMATH_CALUDE_volume_after_density_change_l424_42463


namespace NUMINAMATH_CALUDE_student_allocation_arrangements_l424_42490

theorem student_allocation_arrangements : 
  let n : ℕ := 4  -- number of students
  let m : ℕ := 3  -- number of locations
  let arrangements := {f : Fin n → Fin m | ∀ i : Fin m, ∃ j : Fin n, f j = i}
  Fintype.card arrangements = 36 := by
sorry

end NUMINAMATH_CALUDE_student_allocation_arrangements_l424_42490


namespace NUMINAMATH_CALUDE_jilin_coldest_l424_42495

structure City where
  name : String
  temperature : Int

def beijing : City := { name := "Beijing", temperature := -5 }
def shanghai : City := { name := "Shanghai", temperature := 6 }
def shenzhen : City := { name := "Shenzhen", temperature := 19 }
def jilin : City := { name := "Jilin", temperature := -22 }

def cities : List City := [beijing, shanghai, shenzhen, jilin]

theorem jilin_coldest : 
  ∀ c ∈ cities, jilin.temperature ≤ c.temperature :=
by sorry

end NUMINAMATH_CALUDE_jilin_coldest_l424_42495


namespace NUMINAMATH_CALUDE_probability_spade_heart_spade_l424_42482

/-- A standard deck of cards. -/
def StandardDeck : ℕ := 52

/-- The number of cards of each suit in a standard deck. -/
def CardsPerSuit : ℕ := 13

/-- The probability of drawing ♠, ♥, ♠ in sequence from a standard deck. -/
def ProbabilitySpadeHeartSpade : ℚ :=
  (CardsPerSuit : ℚ) / StandardDeck *
  (CardsPerSuit : ℚ) / (StandardDeck - 1) *
  (CardsPerSuit - 1 : ℚ) / (StandardDeck - 2)

theorem probability_spade_heart_spade :
  ProbabilitySpadeHeartSpade = 78 / 5100 := by
  sorry

end NUMINAMATH_CALUDE_probability_spade_heart_spade_l424_42482


namespace NUMINAMATH_CALUDE_complex_fraction_imaginary_l424_42496

theorem complex_fraction_imaginary (a : ℝ) : 
  (∃ (k : ℝ), (2 + I) / (a - I) = k * I) → a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_imaginary_l424_42496


namespace NUMINAMATH_CALUDE_cos_240_degrees_l424_42439

theorem cos_240_degrees : Real.cos (240 * π / 180) = -(1/2) := by
  sorry

end NUMINAMATH_CALUDE_cos_240_degrees_l424_42439


namespace NUMINAMATH_CALUDE_units_digit_of_product_l424_42429

theorem units_digit_of_product (a b c : ℕ) : 
  (2^1501 * 5^1502 * 11^1503) % 10 = 0 :=
by sorry

end NUMINAMATH_CALUDE_units_digit_of_product_l424_42429


namespace NUMINAMATH_CALUDE_not_always_perfect_square_l424_42461

theorem not_always_perfect_square (d : ℕ) (h1 : d > 0) (h2 : d ≠ 2) (h3 : d ≠ 5) (h4 : d ≠ 13) :
  ∃ (a b : ℕ), a ∈ ({2, 5, 13, d} : Set ℕ) ∧ b ∈ ({2, 5, 13, d} : Set ℕ) ∧ a ≠ b ∧
  ¬∃ (k : ℕ), a * b - 1 = k * k :=
by sorry

end NUMINAMATH_CALUDE_not_always_perfect_square_l424_42461


namespace NUMINAMATH_CALUDE_workout_difference_l424_42442

/-- Represents Oliver's workout schedule over four days -/
structure WorkoutSchedule where
  monday : ℝ
  tuesday : ℝ
  wednesday : ℝ
  thursday : ℝ

/-- Checks if a workout schedule satisfies the given conditions -/
def is_valid_schedule (s : WorkoutSchedule) : Prop :=
  s.monday = 4 ∧
  s.tuesday < s.monday ∧
  s.wednesday = 2 * s.monday ∧
  s.thursday = 2 * s.tuesday ∧
  s.monday + s.tuesday + s.wednesday + s.thursday = 18

/-- Theorem stating that for any valid workout schedule, 
    the difference between Monday's and Tuesday's workout time is 2 hours -/
theorem workout_difference (s : WorkoutSchedule) 
  (h : is_valid_schedule s) : s.monday - s.tuesday = 2 := by
  sorry

end NUMINAMATH_CALUDE_workout_difference_l424_42442


namespace NUMINAMATH_CALUDE_inequality_proof_l424_42449

theorem inequality_proof (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c)
  (h : a^4 + b^4 + c^4 ≤ 2*(a^2*b^2 + b^2*c^2 + c^2*a^2)) :
  (a ≤ b + c ∧ b ≤ c + a ∧ c ≤ a + b) ∧
  (a^2 + b^2 + c^2 ≤ 2*(a*b + b*c + c*a)) ∧
  ¬(∀ x y z : ℝ, x^2 + y^2 + z^2 ≤ 2*(x*y + y*z + z*x) →
    x^4 + y^4 + z^4 ≤ 2*(x^2*y^2 + y^2*z^2 + z^2*x^2)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l424_42449


namespace NUMINAMATH_CALUDE_monotonic_increasing_interval_of_f_l424_42471

/-- The function f(x) = -x^2 + 2x - 2 -/
def f (x : ℝ) : ℝ := -x^2 + 2*x - 2

/-- The monotonic increasing interval of f(x) = -x^2 + 2x - 2 is (-∞, 1) -/
theorem monotonic_increasing_interval_of_f :
  ∀ x y : ℝ, x < y → y ≤ 1 → f x < f y :=
sorry

end NUMINAMATH_CALUDE_monotonic_increasing_interval_of_f_l424_42471


namespace NUMINAMATH_CALUDE_complex_number_real_minus_imag_l424_42400

theorem complex_number_real_minus_imag : 
  let z : ℂ := 5 / (-3 - Complex.I)
  let a : ℝ := z.re
  let b : ℝ := z.im
  a - b = -2 := by sorry

end NUMINAMATH_CALUDE_complex_number_real_minus_imag_l424_42400


namespace NUMINAMATH_CALUDE_copper_wire_length_greater_than_225_l424_42436

/-- Represents the properties of a copper wire -/
structure CopperWire where
  density : Real
  volume : Real
  diagonal : Real

/-- Theorem: The length of a copper wire with given properties is greater than 225 meters -/
theorem copper_wire_length_greater_than_225 (wire : CopperWire)
  (h1 : wire.density = 8900)
  (h2 : wire.volume = 0.5e-3)
  (h3 : wire.diagonal = 2e-3) :
  let cross_section_area := (wire.diagonal / Real.sqrt 2) ^ 2
  let length := wire.volume / cross_section_area
  length > 225 := by
  sorry

#check copper_wire_length_greater_than_225

end NUMINAMATH_CALUDE_copper_wire_length_greater_than_225_l424_42436


namespace NUMINAMATH_CALUDE_min_value_theorem_f4_range_theorem_m_range_theorem_l424_42407

-- Define the functions f and g
def f (a b x : ℝ) : ℝ := a * x + b
def g (x : ℝ) : ℝ := x^2 - 2*x - 8

-- Theorem 1
theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hf2 : f a b 2 = 1) :
  (1 / a + 2 / b) ≥ 8 :=
sorry

-- Theorem 2
theorem f4_range_theorem (a b : ℝ) (h : ∀ x ∈ Set.Icc 1 2, 0 ≤ f a b x ∧ f a b x ≤ 1) :
  -2 ≤ f a b 4 ∧ f a b 4 ≤ 3 :=
sorry

-- Theorem 3
theorem m_range_theorem (m : ℝ) :
  (∀ x > 2, g x ≥ (m + 2) * x - m - 15) → m ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_f4_range_theorem_m_range_theorem_l424_42407


namespace NUMINAMATH_CALUDE_prop_q_not_necessary_nor_sufficient_l424_42473

/-- Proposition P: The solution sets of two quadratic inequalities are the same -/
def PropP (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) : Prop :=
  {x : ℝ | a₁ * x^2 + b₁ * x + c₁ > 0} = {x : ℝ | a₂ * x^2 + b₂ * x + c₂ > 0}

/-- Proposition Q: The coefficients of two quadratic expressions are proportional -/
def PropQ (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ a₁ = k * a₂ ∧ b₁ = k * b₂ ∧ c₁ = k * c₂

theorem prop_q_not_necessary_nor_sufficient :
  ¬(∀ a₁ b₁ c₁ a₂ b₂ c₂ : ℝ, PropQ a₁ b₁ c₁ a₂ b₂ c₂ → PropP a₁ b₁ c₁ a₂ b₂ c₂) ∧
  ¬(∀ a₁ b₁ c₁ a₂ b₂ c₂ : ℝ, PropP a₁ b₁ c₁ a₂ b₂ c₂ → PropQ a₁ b₁ c₁ a₂ b₂ c₂) :=
sorry

end NUMINAMATH_CALUDE_prop_q_not_necessary_nor_sufficient_l424_42473


namespace NUMINAMATH_CALUDE_expression_evaluation_l424_42483

theorem expression_evaluation :
  let x : ℕ := 3
  let y : ℕ := 4
  5 * x^(y+1) + 6 * y^(x+1) = 2751 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l424_42483


namespace NUMINAMATH_CALUDE_sweettarts_distribution_l424_42423

theorem sweettarts_distribution (total_sweettarts : ℕ) (num_friends : ℕ) (sweettarts_per_friend : ℕ) :
  total_sweettarts = 15 →
  num_friends = 3 →
  total_sweettarts = num_friends * sweettarts_per_friend →
  sweettarts_per_friend = 5 := by
  sorry

end NUMINAMATH_CALUDE_sweettarts_distribution_l424_42423


namespace NUMINAMATH_CALUDE_geometric_series_sum_l424_42468

theorem geometric_series_sum : 
  let a : ℝ := 1
  let r : ℝ := -3
  let n : ℕ := 7
  let S := (a * (r^n - 1)) / (r - 1)
  ((-3)^6 = 729) → ((-3)^7 = -2187) → S = 547 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l424_42468


namespace NUMINAMATH_CALUDE_unique_line_through_points_l424_42456

-- Define a type for points in a plane
axiom Point : Type

-- Define a type for straight lines
axiom Line : Type

-- Define a relation for a point being on a line
axiom on_line : Point → Line → Prop

-- Axiom: For any two distinct points, there exists a line passing through both points
axiom line_through_points (P Q : Point) (h : P ≠ Q) : ∃ L : Line, on_line P L ∧ on_line Q L

-- Theorem: There is a unique straight line passing through any two distinct points
theorem unique_line_through_points (P Q : Point) (h : P ≠ Q) : 
  ∃! L : Line, on_line P L ∧ on_line Q L :=
sorry

end NUMINAMATH_CALUDE_unique_line_through_points_l424_42456


namespace NUMINAMATH_CALUDE_trig_expression_simplification_l424_42451

theorem trig_expression_simplification :
  let left_numerator := Real.sin (15 * π / 180) + Real.sin (30 * π / 180) + 
                        Real.sin (45 * π / 180) + Real.sin (60 * π / 180) + 
                        Real.sin (75 * π / 180)
  let denominator := Real.cos (10 * π / 180) * Real.cos (20 * π / 180) * 
                     Real.cos (30 * π / 180) * 2
  let right_numerator := 2 * Real.sqrt 2 * Real.cos (22.5 * π / 180) * 
                         Real.cos (7.5 * π / 180)
  left_numerator / denominator = right_numerator / denominator := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_simplification_l424_42451


namespace NUMINAMATH_CALUDE_min_value_theorem_equality_condition_l424_42415

theorem min_value_theorem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x^2 + y^2 + z^2 = 1) : 
  (x^5 / (y^2 + z^2 - y*z)) + (y^5 / (z^2 + x^2 - z*x)) + (z^5 / (x^2 + y^2 - x*y)) ≥ Real.sqrt 3 / 3 :=
by sorry

theorem equality_condition (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x^2 + y^2 + z^2 = 1) : 
  (x^5 / (y^2 + z^2 - y*z)) + (y^5 / (z^2 + x^2 - z*x)) + (z^5 / (x^2 + y^2 - x*y)) = Real.sqrt 3 / 3 ↔ 
  x = 1 / Real.sqrt 3 ∧ y = 1 / Real.sqrt 3 ∧ z = 1 / Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_equality_condition_l424_42415


namespace NUMINAMATH_CALUDE_chair_cost_l424_42450

theorem chair_cost (total_spent : ℕ) (num_chairs : ℕ) (cost_per_chair : ℚ)
  (h1 : total_spent = 180)
  (h2 : num_chairs = 12)
  (h3 : (cost_per_chair : ℚ) * (num_chairs : ℚ) = total_spent) :
  cost_per_chair = 15 := by
  sorry

end NUMINAMATH_CALUDE_chair_cost_l424_42450


namespace NUMINAMATH_CALUDE_quadratic_root_theorem_l424_42419

-- Define the quadratic equation
def quadratic (x k : ℝ) : ℝ := x^2 + 2*x + 3 - k

-- Define the condition for distinct real roots
def has_distinct_real_roots (k : ℝ) : Prop :=
  ∃ α β : ℝ, α ≠ β ∧ quadratic α k = 0 ∧ quadratic β k = 0

-- Define the relationship between k and the roots
def root_relationship (k α β : ℝ) : Prop :=
  k^2 = α * β + 3 * k

-- Theorem statement
theorem quadratic_root_theorem (k : ℝ) :
  has_distinct_real_roots k ∧ (∃ α β : ℝ, root_relationship k α β) → k = 3 :=
sorry

end NUMINAMATH_CALUDE_quadratic_root_theorem_l424_42419


namespace NUMINAMATH_CALUDE_fraction_problem_l424_42404

theorem fraction_problem :
  let x : ℚ := 4
  let y : ℚ := 15
  (y = x^2 - 1) ∧
  ((x + 2) / (y + 2) > 1/4) ∧
  ((x - 3) / (y - 3) = 1/12) := by
sorry

end NUMINAMATH_CALUDE_fraction_problem_l424_42404


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l424_42424

theorem quadratic_no_real_roots
  (p q a b c : ℝ)
  (pos_p : p > 0)
  (pos_q : q > 0)
  (pos_a : a > 0)
  (pos_b : b > 0)
  (pos_c : c > 0)
  (p_neq_q : p ≠ q)
  (geom_seq : a^2 = p * q)
  (arith_seq_1 : 2 * b = p + c)
  (arith_seq_2 : 2 * c = b + q) :
  (2 * a)^2 - 4 * b * c < 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l424_42424


namespace NUMINAMATH_CALUDE_rotated_semicircle_area_l424_42428

/-- The area of a figure formed by rotating a semicircle around one of its ends by 45 degrees -/
theorem rotated_semicircle_area (R : ℝ) (h : R > 0) :
  let α : Real := π / 4  -- 45 degrees in radians
  let semicircle_area := π * R^2 / 2
  let rotated_area := (2 * R)^2 * α / 2
  rotated_area = semicircle_area :=
by sorry

end NUMINAMATH_CALUDE_rotated_semicircle_area_l424_42428


namespace NUMINAMATH_CALUDE_crayon_purchase_l424_42426

def half_dozen : ℕ := 6

theorem crayon_purchase (total_cost : ℕ) (cost_per_crayon : ℕ) (half_dozens : ℕ) : 
  total_cost = 48 ∧ 
  cost_per_crayon = 2 ∧ 
  total_cost = half_dozens * half_dozen * cost_per_crayon →
  half_dozens = 4 := by
sorry

end NUMINAMATH_CALUDE_crayon_purchase_l424_42426


namespace NUMINAMATH_CALUDE_calculate_expression_l424_42431

theorem calculate_expression : (-3)^2 / 4 * (1/4) = 9/16 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l424_42431


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l424_42462

theorem triangle_angle_measure (X Y Z : ℝ) : 
  Y = 30 → Z = 3 * Y → X + Y + Z = 180 → X = 60 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l424_42462


namespace NUMINAMATH_CALUDE_profit_share_difference_example_l424_42455

/-- Given a total profit and a ratio of division between two parties, 
    calculate the difference between their shares. -/
def profit_share_difference (total_profit : ℚ) (ratio_x ratio_y : ℚ) : ℚ :=
  let total_ratio := ratio_x + ratio_y
  let share_x := (ratio_x / total_ratio) * total_profit
  let share_y := (ratio_y / total_ratio) * total_profit
  share_x - share_y

/-- Theorem stating that for a total profit of 500 and a ratio of 1/2 : 1/3, 
    the difference in profit shares is 100. -/
theorem profit_share_difference_example : 
  profit_share_difference 500 (1/2) (1/3) = 100 := by
  sorry

#eval profit_share_difference 500 (1/2) (1/3)

end NUMINAMATH_CALUDE_profit_share_difference_example_l424_42455


namespace NUMINAMATH_CALUDE_some_number_equation_l424_42435

/-- Given the equation x - 8 / 7 * 5 + 10 = 13.285714285714286, prove that x = 9 -/
theorem some_number_equation (x : ℝ) : x - 8 / 7 * 5 + 10 = 13.285714285714286 → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_some_number_equation_l424_42435


namespace NUMINAMATH_CALUDE_sequence_formula_l424_42474

def sequence_sum (a : ℕ → ℝ) (n : ℕ) : ℝ := 3 + 2 * a n

theorem sequence_formula (a : ℕ → ℝ) (h : ∀ n, sequence_sum a n = 3 + 2 * a n) :
  ∀ n, a n = -3 * 2^(n - 1) := by
  sorry

end NUMINAMATH_CALUDE_sequence_formula_l424_42474


namespace NUMINAMATH_CALUDE_division_problem_l424_42459

theorem division_problem : (62976 : ℕ) / 512 = 123 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l424_42459


namespace NUMINAMATH_CALUDE_integer_product_condition_l424_42477

theorem integer_product_condition (a : ℝ) : 
  (∀ n : ℕ, ∃ m : ℤ, a * n * (n + 2) * (n + 4) = m) ↔ 
  (∃ k : ℤ, a = k / 3) :=
sorry

end NUMINAMATH_CALUDE_integer_product_condition_l424_42477


namespace NUMINAMATH_CALUDE_gcd_of_three_numbers_l424_42411

theorem gcd_of_three_numbers : Nat.gcd 9125 (Nat.gcd 4257 2349) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_three_numbers_l424_42411


namespace NUMINAMATH_CALUDE_cos_pi_minus_2alpha_l424_42486

theorem cos_pi_minus_2alpha (α : ℝ) (h : Real.cos (π / 2 - α) = Real.sqrt 2 / 3) : 
  Real.cos (π - 2 * α) = -5 / 9 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_minus_2alpha_l424_42486


namespace NUMINAMATH_CALUDE_polygon_interior_angle_sum_induction_base_l424_42488

/-- A polygon is a closed plane figure with at least 3 sides. -/
structure Polygon where
  sides : ℕ
  sides_ge_3 : sides ≥ 3

/-- The base case for the polygon interior angle sum theorem. -/
def polygon_interior_angle_sum_base_case : ℕ := 3

/-- Theorem: The base case for mathematical induction in the polygon interior angle sum theorem is n=3. -/
theorem polygon_interior_angle_sum_induction_base :
  polygon_interior_angle_sum_base_case = 3 :=
by sorry

end NUMINAMATH_CALUDE_polygon_interior_angle_sum_induction_base_l424_42488


namespace NUMINAMATH_CALUDE_inequality_properties_l424_42447

theorem inequality_properties (a b : ℝ) (h : (1 / a) < (1 / b) ∧ (1 / b) < 0) :
  (a + b < a * b) ∧ (abs a < abs b) ∧ (b / a + a / b > 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_properties_l424_42447

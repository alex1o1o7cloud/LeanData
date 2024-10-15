import Mathlib

namespace NUMINAMATH_CALUDE_translated_quadratic_vertex_l3076_307662

/-- The vertex of a quadratic function translated to the right by 3 units -/
theorem translated_quadratic_vertex (f g : ℝ → ℝ) (h : ℝ) :
  (∀ x, f x = 2 * (x - 1)^2 - 3) →
  (∀ x, g x = 2 * (x - 4)^2 - 3) →
  (∀ x, g x = f (x - 3)) →
  h = 4 →
  (∀ x, g x ≥ g h) →
  g h = -3 :=
by sorry

end NUMINAMATH_CALUDE_translated_quadratic_vertex_l3076_307662


namespace NUMINAMATH_CALUDE_quadrant_I_solution_l3076_307697

theorem quadrant_I_solution (c : ℝ) :
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x - 2*y = 5 ∧ c*x + 3*y = 2) ↔ -3/2 < c ∧ c < 2/5 :=
sorry

end NUMINAMATH_CALUDE_quadrant_I_solution_l3076_307697


namespace NUMINAMATH_CALUDE_reasonable_reasoning_types_l3076_307629

/-- Represents different types of reasoning --/
inductive ReasoningType
  | Analogy
  | Inductive
  | Deductive

/-- Determines if a reasoning type is considered reasonable --/
def is_reasonable (r : ReasoningType) : Prop :=
  match r with
  | ReasoningType.Analogy => true
  | ReasoningType.Inductive => true
  | ReasoningType.Deductive => false

/-- Theorem stating which reasoning types are reasonable --/
theorem reasonable_reasoning_types :
  (is_reasonable ReasoningType.Analogy) ∧
  (is_reasonable ReasoningType.Inductive) ∧
  ¬(is_reasonable ReasoningType.Deductive) :=
by sorry


end NUMINAMATH_CALUDE_reasonable_reasoning_types_l3076_307629


namespace NUMINAMATH_CALUDE_polynomial_coefficient_bound_l3076_307681

-- Define the polynomial function
def p (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

-- State the theorem
theorem polynomial_coefficient_bound (a b c d : ℝ) :
  (∀ x : ℝ, |x| < 1 → |p a b c d x| ≤ 1) →
  |a| + |b| + |c| + |d| ≤ 7 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_bound_l3076_307681


namespace NUMINAMATH_CALUDE_hawks_score_l3076_307628

/-- The number of touchdowns scored by the Hawks -/
def num_touchdowns : ℕ := 3

/-- The number of points for each touchdown -/
def points_per_touchdown : ℕ := 7

/-- The total number of points scored by the Hawks -/
def total_points : ℕ := num_touchdowns * points_per_touchdown

/-- Theorem stating that the total points scored by the Hawks is 21 -/
theorem hawks_score :
  total_points = 21 := by sorry

end NUMINAMATH_CALUDE_hawks_score_l3076_307628


namespace NUMINAMATH_CALUDE_difference_is_64_l3076_307696

/-- Defines the sequence a_n based on the given recurrence relation -/
def a : ℕ → ℕ → ℕ
  | n, x => if n = 0 then x
            else if x % 2 = 0 then a (n-1) (x / 2)
            else a (n-1) (3 * x + 1)

/-- Returns all possible values of a_1 given a_7 = 2 -/
def possible_a1 : List ℕ :=
  (List.range 1000).filter (λ x => a 6 x = 2)

/-- Calculates the maximum sum of the first 7 terms -/
def max_sum : ℕ :=
  (possible_a1.map (λ x => List.sum (List.map (a · x) (List.range 7)))).maximum?
    |>.getD 0

/-- Calculates the sum of all possible values of a_1 -/
def sum_possible_a1 : ℕ :=
  List.sum possible_a1

/-- The main theorem to be proved -/
theorem difference_is_64 : max_sum - sum_possible_a1 = 64 := by
  sorry

end NUMINAMATH_CALUDE_difference_is_64_l3076_307696


namespace NUMINAMATH_CALUDE_cubic_three_zeros_a_range_l3076_307634

/-- A function f(x) = x^3 - 3x + a has three distinct zeros -/
def has_three_distinct_zeros (a : ℝ) : Prop :=
  ∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    x^3 - 3*x + a = 0 ∧
    y^3 - 3*y + a = 0 ∧
    z^3 - 3*z + a = 0

/-- If f(x) = x^3 - 3x + a has three distinct zeros, then a is in the open interval (-2, 2) -/
theorem cubic_three_zeros_a_range :
  ∀ a : ℝ, has_three_distinct_zeros a → -2 < a ∧ a < 2 :=
by sorry

end NUMINAMATH_CALUDE_cubic_three_zeros_a_range_l3076_307634


namespace NUMINAMATH_CALUDE_inequality_solution_count_l3076_307698

theorem inequality_solution_count : 
  (∃ (S : Finset ℕ), 
    (∀ n ∈ S, (n : ℝ) + 6 * ((n : ℝ) - 1) * ((n : ℝ) - 15) < 0) ∧ 
    (∀ n : ℕ, (n : ℝ) + 6 * ((n : ℝ) - 1) * ((n : ℝ) - 15) < 0 → n ∈ S) ∧
    Finset.card S = 13) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_count_l3076_307698


namespace NUMINAMATH_CALUDE_courses_choice_theorem_l3076_307645

/-- The number of courses available -/
def total_courses : ℕ := 4

/-- The number of courses each person chooses -/
def courses_per_person : ℕ := 2

/-- The number of ways to choose courses with at least one difference -/
def ways_with_difference : ℕ := 30

/-- Theorem stating the number of ways to choose courses with at least one difference -/
theorem courses_choice_theorem : 
  (Nat.choose total_courses courses_per_person) * 
  (Nat.choose total_courses courses_per_person) - 
  (Nat.choose total_courses courses_per_person) = ways_with_difference :=
by sorry

end NUMINAMATH_CALUDE_courses_choice_theorem_l3076_307645


namespace NUMINAMATH_CALUDE_athletes_game_count_l3076_307623

theorem athletes_game_count (malik_yards josiah_yards darnell_yards total_yards : ℕ) 
  (h1 : malik_yards = 18)
  (h2 : josiah_yards = 22)
  (h3 : darnell_yards = 11)
  (h4 : total_yards = 204) :
  ∃ n : ℕ, n * (malik_yards + josiah_yards + darnell_yards) = total_yards ∧ n = 4 := by
sorry

end NUMINAMATH_CALUDE_athletes_game_count_l3076_307623


namespace NUMINAMATH_CALUDE_rectangular_prism_sum_l3076_307684

/-- A rectangular prism is a three-dimensional shape with 6 rectangular faces. -/
structure RectangularPrism where
  -- We don't need to define specific dimensions, as they don't affect the result

/-- The number of edges in a rectangular prism -/
def num_edges (rp : RectangularPrism) : ℕ := 12

/-- The number of vertices in a rectangular prism -/
def num_vertices (rp : RectangularPrism) : ℕ := 8

/-- The number of faces in a rectangular prism -/
def num_faces (rp : RectangularPrism) : ℕ := 6

/-- Theorem: The sum of edges, vertices, and faces of any rectangular prism is 26 -/
theorem rectangular_prism_sum (rp : RectangularPrism) :
  num_edges rp + num_vertices rp + num_faces rp = 26 := by
  sorry


end NUMINAMATH_CALUDE_rectangular_prism_sum_l3076_307684


namespace NUMINAMATH_CALUDE_arithmetic_sequence_product_inequality_l3076_307618

/-- An arithmetic sequence of 8 terms with positive values and non-zero common difference -/
structure ArithmeticSequence8 where
  a : Fin 8 → ℝ
  positive : ∀ i, a i > 0
  is_arithmetic : ∃ d ≠ 0, ∀ i j, a j - a i = (j - i : ℝ) * d

theorem arithmetic_sequence_product_inequality (seq : ArithmeticSequence8) :
  seq.a 0 * seq.a 7 < seq.a 3 * seq.a 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_product_inequality_l3076_307618


namespace NUMINAMATH_CALUDE_cuboid_volume_calculation_l3076_307638

def cuboid_volume (length width height : ℝ) : ℝ := length * width * height

theorem cuboid_volume_calculation :
  let length : ℝ := 6
  let width : ℝ := 5
  let height : ℝ := 6
  cuboid_volume length width height = 180 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_volume_calculation_l3076_307638


namespace NUMINAMATH_CALUDE_painted_cubes_multiple_of_unpainted_l3076_307649

theorem painted_cubes_multiple_of_unpainted (n : ℕ) : ∃ n, n > 0 ∧ (n + 2)^3 > 10 ∧ n^3 ∣ ((n + 2)^3 - n^3) := by
  sorry

end NUMINAMATH_CALUDE_painted_cubes_multiple_of_unpainted_l3076_307649


namespace NUMINAMATH_CALUDE_cameron_total_questions_l3076_307642

def usual_questions_per_tourist : ℕ := 2

def group_size_1 : ℕ := 6
def group_size_2 : ℕ := 11
def group_size_3 : ℕ := 8
def group_size_4 : ℕ := 7

def inquisitive_tourist_multiplier : ℕ := 3

theorem cameron_total_questions :
  let group_1_questions := group_size_1 * usual_questions_per_tourist
  let group_2_questions := group_size_2 * usual_questions_per_tourist
  let group_3_questions := (group_size_3 - 1) * usual_questions_per_tourist +
                           usual_questions_per_tourist * inquisitive_tourist_multiplier
  let group_4_questions := group_size_4 * usual_questions_per_tourist
  group_1_questions + group_2_questions + group_3_questions + group_4_questions = 68 := by
  sorry

end NUMINAMATH_CALUDE_cameron_total_questions_l3076_307642


namespace NUMINAMATH_CALUDE_min_value_of_e_l3076_307689

def e (x : ℝ) (C : ℝ) : ℝ := (x - 1) * (x - 3) * (x - 4) * (x - 6) + C

theorem min_value_of_e (C : ℝ) : 
  (C = -0.5625) ↔ (∀ x : ℝ, e x C ≥ 1 ∧ ∃ x₀ : ℝ, e x₀ C = 1) :=
sorry

end NUMINAMATH_CALUDE_min_value_of_e_l3076_307689


namespace NUMINAMATH_CALUDE_product_from_lcm_gcd_l3076_307640

theorem product_from_lcm_gcd (a b : ℕ+) 
  (h1 : Nat.lcm a b = 24) 
  (h2 : Nat.gcd a b = 8) : 
  a * b = 192 := by
  sorry

end NUMINAMATH_CALUDE_product_from_lcm_gcd_l3076_307640


namespace NUMINAMATH_CALUDE_equation_solution_l3076_307604

theorem equation_solution (x : ℚ) (h : x ≠ 3) : (x + 5) / (x - 3) = 4 ↔ x = 17 / 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3076_307604


namespace NUMINAMATH_CALUDE_cos_45_degrees_l3076_307665

theorem cos_45_degrees : Real.cos (π / 4) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_45_degrees_l3076_307665


namespace NUMINAMATH_CALUDE_pet_store_combinations_l3076_307671

def num_puppies : ℕ := 12
def num_kittens : ℕ := 10
def num_hamsters : ℕ := 8
def num_rabbits : ℕ := 5
def num_people : ℕ := 4

theorem pet_store_combinations : 
  (num_puppies * num_kittens * num_hamsters * num_rabbits) * Nat.factorial num_people = 115200 :=
by sorry

end NUMINAMATH_CALUDE_pet_store_combinations_l3076_307671


namespace NUMINAMATH_CALUDE_train_crossing_time_l3076_307677

/-- Calculates the time taken for a train to cross a signal pole -/
theorem train_crossing_time 
  (train_length : ℝ) 
  (platform_length : ℝ) 
  (platform_crossing_time : ℝ) 
  (h1 : train_length = 300) 
  (h2 : platform_length = 250) 
  (h3 : platform_crossing_time = 33) :
  (train_length / ((train_length + platform_length) / platform_crossing_time)) = 18 :=
by sorry

end NUMINAMATH_CALUDE_train_crossing_time_l3076_307677


namespace NUMINAMATH_CALUDE_line_equation_point_slope_l3076_307680

/-- Theorem: Equation of a line with given slope passing through a point -/
theorem line_equation_point_slope (k x₀ y₀ : ℝ) :
  ∀ x y : ℝ, (y - y₀ = k * (x - x₀)) ↔ (y = k * x + (y₀ - k * x₀)) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_point_slope_l3076_307680


namespace NUMINAMATH_CALUDE_fish_problem_l3076_307624

/-- The number of fish Ken and Kendra brought home -/
def total_fish_brought_home (ken_caught : ℕ) (ken_released : ℕ) (kendra_caught : ℕ) : ℕ :=
  (ken_caught - ken_released) + kendra_caught

/-- Theorem stating the total number of fish brought home by Ken and Kendra -/
theorem fish_problem :
  ∀ (ken_caught : ℕ) (kendra_caught : ℕ),
    ken_caught = 2 * kendra_caught →
    kendra_caught = 30 →
    total_fish_brought_home ken_caught 3 kendra_caught = 87 := by
  sorry


end NUMINAMATH_CALUDE_fish_problem_l3076_307624


namespace NUMINAMATH_CALUDE_expected_black_pairs_modified_deck_l3076_307627

/-- A deck of cards -/
structure Deck :=
  (total : ℕ)
  (black : ℕ)
  (red : ℕ)
  (h_total : total = black + red)

/-- The expected number of pairs of adjacent black cards in a circular deal -/
def expected_black_pairs (d : Deck) : ℚ :=
  (d.black : ℚ) * (d.black - 1) / (d.total - 1)

/-- The main theorem -/
theorem expected_black_pairs_modified_deck :
  ∃ (d : Deck), d.total = 60 ∧ d.black = 30 ∧ d.red = 30 ∧ expected_black_pairs d = 870 / 59 := by
  sorry

end NUMINAMATH_CALUDE_expected_black_pairs_modified_deck_l3076_307627


namespace NUMINAMATH_CALUDE_percentage_calculation_l3076_307641

theorem percentage_calculation (total : ℝ) (difference : ℝ) : 
  total = 6000 ∧ difference = 693 → 
  ∃ P : ℝ, (1/10 * total) - (P/100 * total) = difference ∧ P = 1.55 := by
sorry

end NUMINAMATH_CALUDE_percentage_calculation_l3076_307641


namespace NUMINAMATH_CALUDE_log_difference_l3076_307639

theorem log_difference (a b c d : ℕ+) 
  (h1 : (Real.log b) / (Real.log a) = 3/2)
  (h2 : (Real.log d) / (Real.log c) = 5/4)
  (h3 : a - c = 9) :
  b - d = 93 := by
  sorry

end NUMINAMATH_CALUDE_log_difference_l3076_307639


namespace NUMINAMATH_CALUDE_binary_number_divisibility_l3076_307685

theorem binary_number_divisibility : ∃ k : ℕ, 2^139 + 2^105 + 2^15 + 2^13 = 136 * k := by
  sorry

end NUMINAMATH_CALUDE_binary_number_divisibility_l3076_307685


namespace NUMINAMATH_CALUDE_lcm_gcf_ratio_280_476_l3076_307619

theorem lcm_gcf_ratio_280_476 : Nat.lcm 280 476 / Nat.gcd 280 476 = 170 := by sorry

end NUMINAMATH_CALUDE_lcm_gcf_ratio_280_476_l3076_307619


namespace NUMINAMATH_CALUDE_circle_tangent_to_parallel_lines_l3076_307664

/-- A circle is tangent to two parallel lines and its center lies on a third line. -/
theorem circle_tangent_to_parallel_lines (x y : ℝ) :
  (3 * x - 4 * y = 12 ∨ 3 * x - 4 * y = -48) ∧ 
  (x - 2 * y = 0) →
  x = -18 ∧ y = -9 := by
  sorry

end NUMINAMATH_CALUDE_circle_tangent_to_parallel_lines_l3076_307664


namespace NUMINAMATH_CALUDE_machine_value_after_two_years_l3076_307610

/-- Calculates the value of a machine after a given number of years, 
    given its initial value and yearly depreciation rate. -/
def machine_value (initial_value : ℝ) (depreciation_rate : ℝ) (years : ℕ) : ℝ :=
  initial_value * (1 - depreciation_rate) ^ years

/-- Theorem stating that a machine purchased for $8,000 with a 10% yearly depreciation rate
    will have a value of $6,480 after two years. -/
theorem machine_value_after_two_years :
  machine_value 8000 0.1 2 = 6480 := by
  sorry

#eval machine_value 8000 0.1 2

end NUMINAMATH_CALUDE_machine_value_after_two_years_l3076_307610


namespace NUMINAMATH_CALUDE_like_terms_exponents_l3076_307617

theorem like_terms_exponents (m n : ℕ) : 
  (∀ x y : ℝ, ∃ k : ℝ, 2 * x^(n+2) * y^3 = k * (-3 * x^3 * y^(2*m-1))) → 
  (m = 2 ∧ n = 1) := by
sorry

end NUMINAMATH_CALUDE_like_terms_exponents_l3076_307617


namespace NUMINAMATH_CALUDE_tangent_curve_a_value_l3076_307666

noncomputable def curve (a : ℝ) (x : ℝ) : ℝ := Real.log (x + a)

noncomputable def tangent_line (x : ℝ) : ℝ := x + 2

theorem tangent_curve_a_value (a : ℝ) :
  (∃ x₀ : ℝ, curve a x₀ = tangent_line x₀ ∧
    (∀ x : ℝ, x ≠ x₀ → curve a x ≠ tangent_line x) ∧
    (deriv (curve a) x₀ = deriv tangent_line x₀)) →
  a = 3 := by sorry

end NUMINAMATH_CALUDE_tangent_curve_a_value_l3076_307666


namespace NUMINAMATH_CALUDE_outfits_count_l3076_307636

/-- The number of different outfits that can be created given a set of clothing items. -/
def number_of_outfits (shirts : Nat) (pants : Nat) (ties : Nat) (shoes : Nat) : Nat :=
  shirts * pants * (ties + 1) * shoes

/-- Theorem stating that the number of outfits is 240 given the specific clothing items. -/
theorem outfits_count :
  number_of_outfits 5 4 5 2 = 240 := by
  sorry

end NUMINAMATH_CALUDE_outfits_count_l3076_307636


namespace NUMINAMATH_CALUDE_system_solution_l3076_307656

theorem system_solution (x y k : ℝ) : 
  x - y = k + 2 →
  x + 3*y = k →
  x + y = 2 →
  k = 1 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l3076_307656


namespace NUMINAMATH_CALUDE_expression_factorization_l3076_307646

theorem expression_factorization (x : ℝ) : 
  (7 * x^6 + 36 * x^4 - 8) - (3 * x^6 - 4 * x^4 + 6) = 2 * (2 * x^6 + 20 * x^4 - 7) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l3076_307646


namespace NUMINAMATH_CALUDE_smallest_n_for_integer_S_l3076_307692

/-- Definition of S_n as the sum of reciprocals of non-zero digits from 1 to 2·10^n -/
def S (n : ℕ) : ℚ :=
  sorry

/-- Theorem stating that 32 is the smallest positive integer n for which S_n is an integer -/
theorem smallest_n_for_integer_S :
  ∀ k : ℕ, k > 0 → k < 32 → ¬ (S k).isInt ∧ (S 32).isInt := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_integer_S_l3076_307692


namespace NUMINAMATH_CALUDE_distance_to_point_distance_from_origin_to_point_l3076_307675

theorem distance_to_point : ℝ × ℝ → ℝ
  | (x, y) => Real.sqrt (x^2 + y^2)

theorem distance_from_origin_to_point :
  distance_to_point (12, -5) = 13 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_point_distance_from_origin_to_point_l3076_307675


namespace NUMINAMATH_CALUDE_abs_sum_gt_abs_prod_plus_one_implies_prod_zero_l3076_307686

theorem abs_sum_gt_abs_prod_plus_one_implies_prod_zero (a b : ℤ) : 
  |a + b| > |1 + a * b| → a * b = 0 := by
  sorry

end NUMINAMATH_CALUDE_abs_sum_gt_abs_prod_plus_one_implies_prod_zero_l3076_307686


namespace NUMINAMATH_CALUDE_cab_journey_time_l3076_307603

/-- The usual time for a cab to cover a journey -/
def usual_time : ℝ → Prop :=
  λ T => (6 / 5 * T = T + 15) ∧ (T = 75)

theorem cab_journey_time :
  ∃ T : ℝ, usual_time T :=
sorry

end NUMINAMATH_CALUDE_cab_journey_time_l3076_307603


namespace NUMINAMATH_CALUDE_least_positive_integer_multiple_53_l3076_307607

theorem least_positive_integer_multiple_53 :
  ∃ (x : ℕ), x > 0 ∧ 
  (∀ (y : ℕ), 0 < y ∧ y < x → ¬(53 ∣ (2*y)^2 + 2*47*(2*y) + 47^2)) ∧
  (53 ∣ (2*x)^2 + 2*47*(2*x) + 47^2) ∧
  x = 6 := by
sorry

end NUMINAMATH_CALUDE_least_positive_integer_multiple_53_l3076_307607


namespace NUMINAMATH_CALUDE_sum_of_multiples_is_even_l3076_307658

theorem sum_of_multiples_is_even (a b : ℤ) 
  (ha : ∃ k : ℤ, a = 6 * k) 
  (hb : ∃ m : ℤ, b = 8 * m) : 
  ∃ n : ℤ, a + b = 2 * n :=
sorry

end NUMINAMATH_CALUDE_sum_of_multiples_is_even_l3076_307658


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3076_307682

def M : Set ℤ := {-1, 0, 1, 2, 3}
def N : Set ℤ := {-2, 0}

theorem intersection_of_M_and_N : M ∩ N = {0} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3076_307682


namespace NUMINAMATH_CALUDE_equation_solution_l3076_307694

theorem equation_solution : ∃ x : ℚ, 9 - 3 / (x / 3) + 3 = 3 :=
by
  use 1
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3076_307694


namespace NUMINAMATH_CALUDE_preceding_binary_l3076_307695

def binary_to_decimal (b : List Bool) : ℕ :=
  b.foldr (fun bit acc => 2 * acc + if bit then 1 else 0) 0

def decimal_to_binary (n : ℕ) : List Bool :=
  if n = 0 then
    [false]
  else
    let rec aux (m : ℕ) : List Bool :=
      if m = 0 then
        []
      else
        (m % 2 = 1) :: aux (m / 2)
    aux n |>.reverse

theorem preceding_binary (N : List Bool) :
  N = [true, true, true, false, false] →
  decimal_to_binary (binary_to_decimal N - 1) = [true, true, false, true, true] := by
  sorry

end NUMINAMATH_CALUDE_preceding_binary_l3076_307695


namespace NUMINAMATH_CALUDE_problem_statement_l3076_307668

theorem problem_statement : (-0.125)^2007 * (-8)^2008 = -8 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3076_307668


namespace NUMINAMATH_CALUDE_cone_height_equal_cylinder_l3076_307650

/-- Given a cylinder M with base radius 2 and height 6, and a cone N with base diameter
    equal to its slant height, if their volumes are equal, then the height of cone N is 6. -/
theorem cone_height_equal_cylinder (r : ℝ) :
  let cylinder_volume := π * 2^2 * 6
  let cone_base_radius := r
  let cone_height := Real.sqrt 3 * r
  let cone_volume := (1/3) * π * cone_base_radius^2 * cone_height
  cylinder_volume = cone_volume →
  cone_height = 6 := by
sorry

end NUMINAMATH_CALUDE_cone_height_equal_cylinder_l3076_307650


namespace NUMINAMATH_CALUDE_abs_eq_neg_iff_nonpos_l3076_307678

theorem abs_eq_neg_iff_nonpos (a : ℝ) : |a| = -a ↔ a ≤ 0 := by sorry

end NUMINAMATH_CALUDE_abs_eq_neg_iff_nonpos_l3076_307678


namespace NUMINAMATH_CALUDE_range_of_a_l3076_307612

-- Define the functions f and g
def f (a x : ℝ) : ℝ := x^2 - x - a - 2
def g (a x : ℝ) : ℝ := x^2 - (a+1)*x - 2

-- Define the theorem
theorem range_of_a (a : ℝ) 
  (x₁ x₂ x₃ x₄ : ℝ) 
  (h₁ : f a x₁ = 0)
  (h₂ : f a x₂ = 0)
  (h₃ : g a x₃ = 0)
  (h₄ : g a x₄ = 0)
  (h₅ : x₃ < x₁ ∧ x₁ < x₄ ∧ x₄ < x₂) :
  -2 < a ∧ a < 0 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l3076_307612


namespace NUMINAMATH_CALUDE_simplify_expression_l3076_307620

theorem simplify_expression (x : ℝ) (h : x ≠ 2) :
  2 - (2 * (1 - (3 - (2 / (2 - x))))) = 6 - 4 / (2 - x) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3076_307620


namespace NUMINAMATH_CALUDE_ping_pong_balls_count_l3076_307630

/-- The number of ping-pong balls in the gym storage -/
def ping_pong_balls : ℕ :=
  let total_balls : ℕ := 240
  let baseball_boxes : ℕ := 35
  let baseballs_per_box : ℕ := 4
  let tennis_ball_boxes : ℕ := 6
  let tennis_balls_per_box : ℕ := 3
  let baseballs : ℕ := baseball_boxes * baseballs_per_box
  let tennis_balls : ℕ := tennis_ball_boxes * tennis_balls_per_box
  total_balls - (baseballs + tennis_balls)

theorem ping_pong_balls_count : ping_pong_balls = 82 := by
  sorry

end NUMINAMATH_CALUDE_ping_pong_balls_count_l3076_307630


namespace NUMINAMATH_CALUDE_transform_OAB_l3076_307632

/-- Transformation from xy-plane to uv-plane -/
def transform (x y : ℝ) : ℝ × ℝ := (x^2 - y^2, x * y)

/-- Triangle OAB in xy-plane -/
def triangle_OAB : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ x ∧ p = (x, y)}

/-- Image of triangle OAB in uv-plane -/
def image_OAB : Set (ℝ × ℝ) :=
  {q | ∃ p ∈ triangle_OAB, q = transform p.1 p.2}

theorem transform_OAB :
  (0, 0) ∈ image_OAB ∧ (1, 0) ∈ image_OAB ∧ (0, 1) ∈ image_OAB :=
sorry

end NUMINAMATH_CALUDE_transform_OAB_l3076_307632


namespace NUMINAMATH_CALUDE_special_function_value_l3076_307667

/-- A function satisfying the given property -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ m n : ℝ, f (m + n^2) = f m + 2 * (f n)^2

theorem special_function_value (f : ℝ → ℝ) 
  (h1 : special_function f) 
  (h2 : f 1 ≠ 0) : 
  f 2014 = 1007 := by
  sorry

end NUMINAMATH_CALUDE_special_function_value_l3076_307667


namespace NUMINAMATH_CALUDE_exists_perpendicular_line_l3076_307693

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define a relation for a line being in a plane
variable (in_plane : Line → Plane → Prop)

-- Define a relation for perpendicularity between lines
variable (perpendicular : Line → Line → Prop)

-- Theorem statement
theorem exists_perpendicular_line (a : Line) (α : Plane) :
  ∃ l : Line, in_plane l α ∧ perpendicular l a :=
sorry

end NUMINAMATH_CALUDE_exists_perpendicular_line_l3076_307693


namespace NUMINAMATH_CALUDE_total_squares_16x16_board_l3076_307654

/-- The size of the chess board -/
def boardSize : Nat := 16

/-- The total number of squares on a square chess board of given size -/
def totalSquares (n : Nat) : Nat :=
  (n * (n + 1) * (2 * n + 1)) / 6

/-- An irregular shape on the chess board -/
structure IrregularShape where
  size : Nat
  isNonRectangular : Bool

/-- Theorem stating the total number of squares on a 16x16 chess board -/
theorem total_squares_16x16_board (shapes : List IrregularShape) 
  (h1 : ∀ s ∈ shapes, s.size ≥ 4)
  (h2 : ∀ s ∈ shapes, s.isNonRectangular = true) :
  totalSquares boardSize = 1496 := by
  sorry

#eval totalSquares boardSize

end NUMINAMATH_CALUDE_total_squares_16x16_board_l3076_307654


namespace NUMINAMATH_CALUDE_even_function_properties_l3076_307635

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x ≤ f y

def decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f y ≤ f x

theorem even_function_properties (f : ℝ → ℝ) 
  (h_even : is_even f)
  (h_incr : increasing_on f 0 7)
  (h_f7 : f 7 = 6) :
  decreasing_on f (-7) 0 ∧ ∀ x, -7 ≤ x → x ≤ 7 → f x ≤ 6 :=
sorry

end NUMINAMATH_CALUDE_even_function_properties_l3076_307635


namespace NUMINAMATH_CALUDE_games_given_to_friend_l3076_307660

theorem games_given_to_friend (initial_games : ℕ) (remaining_games : ℕ) 
  (h1 : initial_games = 9) 
  (h2 : remaining_games = 5) : 
  initial_games - remaining_games = 4 := by
  sorry

end NUMINAMATH_CALUDE_games_given_to_friend_l3076_307660


namespace NUMINAMATH_CALUDE_total_curve_length_is_6pi_l3076_307625

/-- Regular tetrahedron with edge length 4 -/
structure RegularTetrahedron :=
  (edge_length : ℝ)
  (is_regular : edge_length = 4)

/-- Point on the surface of the tetrahedron -/
structure SurfacePoint (t : RegularTetrahedron) :=
  (distance_from_vertex : ℝ)
  (on_surface : distance_from_vertex = 3)

/-- Total length of curve segments -/
def total_curve_length (t : RegularTetrahedron) (p : SurfacePoint t) : ℝ := sorry

/-- Theorem: The total length of curve segments is 6π -/
theorem total_curve_length_is_6pi (t : RegularTetrahedron) (p : SurfacePoint t) :
  total_curve_length t p = 6 * Real.pi :=
sorry

end NUMINAMATH_CALUDE_total_curve_length_is_6pi_l3076_307625


namespace NUMINAMATH_CALUDE_roots_of_equation_l3076_307643

theorem roots_of_equation (x : ℝ) : 
  x * (x - 1) = 0 ↔ x = 0 ∨ x = 1 := by sorry

end NUMINAMATH_CALUDE_roots_of_equation_l3076_307643


namespace NUMINAMATH_CALUDE_square_minus_product_plus_square_l3076_307601

theorem square_minus_product_plus_square : 6^2 - 4*5 + 4^2 = 32 := by
  sorry

end NUMINAMATH_CALUDE_square_minus_product_plus_square_l3076_307601


namespace NUMINAMATH_CALUDE_binomial_expansion_theorem_l3076_307637

theorem binomial_expansion_theorem (n k : ℕ) (a b : ℝ) : 
  n ≥ 2 → 
  k > 0 → 
  a * b ≠ 0 → 
  a = (k + 1) * b → 
  (n.choose 1 * (k * b)^(n - 1) * (-b) + n.choose 2 * (k * b)^(n - 2) * (-b)^2 = k * b^n * k^(n - 2)) → 
  n = 2 * k + 2 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_theorem_l3076_307637


namespace NUMINAMATH_CALUDE_sine_transformation_l3076_307670

theorem sine_transformation (x : ℝ) :
  let f (t : ℝ) := Real.sin t
  let g (t : ℝ) := f (t - (2/3) * Real.pi)
  let h (t : ℝ) := g (t / 3)
  h x = Real.sin (3 * x - (2/3) * Real.pi) := by
sorry

end NUMINAMATH_CALUDE_sine_transformation_l3076_307670


namespace NUMINAMATH_CALUDE_total_population_is_56000_l3076_307600

/-- The total population of Boise, Seattle, and Lake View -/
def total_population (boise seattle lakeview : ℕ) : ℕ :=
  boise + seattle + lakeview

/-- Theorem: The total population of the three cities is 56000 -/
theorem total_population_is_56000 :
  ∃ (boise seattle lakeview : ℕ),
    boise = (3 * seattle) / 5 ∧
    lakeview = seattle + 4000 ∧
    lakeview = 24000 ∧
    total_population boise seattle lakeview = 56000 := by
  sorry

end NUMINAMATH_CALUDE_total_population_is_56000_l3076_307600


namespace NUMINAMATH_CALUDE_sandy_correct_sums_l3076_307644

theorem sandy_correct_sums 
  (total_sums : ℕ) 
  (total_marks : ℤ) 
  (marks_per_correct : ℕ) 
  (marks_per_incorrect : ℕ) 
  (h1 : total_sums = 30)
  (h2 : total_marks = 65)
  (h3 : marks_per_correct = 3)
  (h4 : marks_per_incorrect = 2) :
  ∃ (correct_sums : ℕ), 
    correct_sums * marks_per_correct - (total_sums - correct_sums) * marks_per_incorrect = total_marks ∧ 
    correct_sums = 25 :=
by sorry

end NUMINAMATH_CALUDE_sandy_correct_sums_l3076_307644


namespace NUMINAMATH_CALUDE_solve_chips_problem_l3076_307616

def chips_problem (total father_chips brother_chips : ℕ) : Prop :=
  total = 800 ∧ father_chips = 268 ∧ brother_chips = 182 →
  total - (father_chips + brother_chips) = 350

theorem solve_chips_problem :
  ∀ (total father_chips brother_chips : ℕ),
    chips_problem total father_chips brother_chips :=
by
  sorry

end NUMINAMATH_CALUDE_solve_chips_problem_l3076_307616


namespace NUMINAMATH_CALUDE_solve_equation_l3076_307615

theorem solve_equation : ∃ x : ℚ, (3 * x + 15 = (1/3) * (7 * x + 42)) ∧ (x = -3/2) := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3076_307615


namespace NUMINAMATH_CALUDE_prob_heart_king_spade_l3076_307690

/-- Represents a standard deck of 52 cards -/
def standardDeck : ℕ := 52

/-- Number of hearts in a standard deck -/
def numHearts : ℕ := 13

/-- Number of kings in a standard deck -/
def numKings : ℕ := 4

/-- Number of spades in a standard deck -/
def numSpades : ℕ := 13

/-- Probability of drawing a heart, then a king, then a spade from a standard 52-card deck without replacement -/
theorem prob_heart_king_spade : 
  (numHearts : ℚ) / standardDeck * 
  numKings / (standardDeck - 1) * 
  numSpades / (standardDeck - 2) = 13 / 2550 := by sorry

end NUMINAMATH_CALUDE_prob_heart_king_spade_l3076_307690


namespace NUMINAMATH_CALUDE_f_value_at_half_l3076_307609

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The domain of f is [a-1, 2a] -/
def HasDomain (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f x ≠ 0 → a - 1 ≤ x ∧ x ≤ 2 * a

/-- The function f(x) = ax² + bx + 3a + b -/
def f (a b : ℝ) (x : ℝ) : ℝ :=
  a * x^2 + b * x + 3 * a + b

theorem f_value_at_half (a b : ℝ) :
  IsEven (f a b) → HasDomain (f a b) a → f a b (1/2) = 13/12 := by
  sorry

end NUMINAMATH_CALUDE_f_value_at_half_l3076_307609


namespace NUMINAMATH_CALUDE_exam_passing_marks_l3076_307605

theorem exam_passing_marks (T : ℝ) (P : ℝ) : 
  (0.3 * T = P - 60) →
  (0.4 * T + 10 = P) →
  (0.5 * T - 5 = P + 40) →
  P = 210 := by
  sorry

end NUMINAMATH_CALUDE_exam_passing_marks_l3076_307605


namespace NUMINAMATH_CALUDE_irrational_numbers_count_l3076_307648

theorem irrational_numbers_count : ∃! (s : Finset ℝ), 
  (∀ x ∈ s, Irrational x ∧ ∃ k : ℤ, (x + 1) / (x^2 - 3*x + 3) = k) ∧ 
  Finset.card s = 2 := by
sorry

end NUMINAMATH_CALUDE_irrational_numbers_count_l3076_307648


namespace NUMINAMATH_CALUDE_equidistant_complex_function_l3076_307655

/-- A complex function f(z) = (a+bi)z with the property that f(z) is equidistant
    from z and 3z for all complex z, and |a+bi| = 5, implies b^2 = 21 -/
theorem equidistant_complex_function (a b : ℝ) : 
  (∀ z : ℂ, ‖(a + b * Complex.I) * z - z‖ = ‖(a + b * Complex.I) * z - 3 * z‖) →
  Complex.abs (a + b * Complex.I) = 5 →
  b^2 = 21 := by
sorry

end NUMINAMATH_CALUDE_equidistant_complex_function_l3076_307655


namespace NUMINAMATH_CALUDE_greatest_piece_length_l3076_307614

theorem greatest_piece_length (rope1 rope2 rope3 : ℕ) 
  (h1 : rope1 = 28) (h2 : rope2 = 45) (h3 : rope3 = 63) : 
  Nat.gcd rope1 (Nat.gcd rope2 rope3) = 7 := by
  sorry

end NUMINAMATH_CALUDE_greatest_piece_length_l3076_307614


namespace NUMINAMATH_CALUDE_circle_tangent_to_parabola_directrix_l3076_307608

/-- Given a circle and a parabola, if the circle is tangent to the directrix of the parabola,
    then the parameter p of the parabola is 2. -/
theorem circle_tangent_to_parabola_directrix (x y : ℝ) (p : ℝ) :
  (x^2 + y^2 - 6*x - 7 = 0) →  -- Circle equation
  (p > 0) →                   -- p is positive
  (∃ (y : ℝ), y^2 = 2*p*x) →  -- Parabola equation
  (∃ (x₀ : ℝ), ∀ (x y : ℝ), x^2 + y^2 - 6*x - 7 = 0 → |x - x₀| ≥ p/2) →  -- Circle is tangent to directrix
  p = 2 := by
sorry

end NUMINAMATH_CALUDE_circle_tangent_to_parabola_directrix_l3076_307608


namespace NUMINAMATH_CALUDE_no_solution_exists_l3076_307613

theorem no_solution_exists : ¬∃ (a b : ℕ+), 
  (a * b + 90 = 24 * Nat.lcm a b + 15 * Nat.gcd a b) ∧ 
  (Nat.gcd a b = 3) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l3076_307613


namespace NUMINAMATH_CALUDE_trig_identity_l3076_307674

theorem trig_identity (α : ℝ) : 
  Real.sin α ^ 2 + Real.cos (π/6 - α) ^ 2 - Real.sin α * Real.cos (π/6 - α) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l3076_307674


namespace NUMINAMATH_CALUDE_sin_20_cos_10_minus_cos_160_cos_80_l3076_307659

theorem sin_20_cos_10_minus_cos_160_cos_80 :
  Real.sin (20 * π / 180) * Real.cos (10 * π / 180) -
  Real.cos (160 * π / 180) * Real.cos (80 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_20_cos_10_minus_cos_160_cos_80_l3076_307659


namespace NUMINAMATH_CALUDE_union_equality_iff_range_l3076_307679

def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}

def B (m : ℝ) : Set ℝ := {x | x^2 - (2*m + 1)*x + 2*m < 0}

theorem union_equality_iff_range (m : ℝ) : A ∪ B m = A ↔ -1/2 ≤ m ∧ m ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_union_equality_iff_range_l3076_307679


namespace NUMINAMATH_CALUDE_prob_X_or_Y_or_Z_wins_l3076_307663

-- Define the probabilities
def prob_X : ℚ := 1/4
def prob_Y : ℚ := 1/8
def prob_Z : ℚ := 1/12

-- Define the total number of cars
def total_cars : ℕ := 15

-- Theorem statement
theorem prob_X_or_Y_or_Z_wins : 
  prob_X + prob_Y + prob_Z = 11/24 := by sorry

end NUMINAMATH_CALUDE_prob_X_or_Y_or_Z_wins_l3076_307663


namespace NUMINAMATH_CALUDE_ratio_of_sum_and_difference_l3076_307669

theorem ratio_of_sum_and_difference (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) 
  (h : a + b = 5 * (a - b)) : a / b = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_sum_and_difference_l3076_307669


namespace NUMINAMATH_CALUDE_mathborough_rainfall_2006_l3076_307683

/-- The total rainfall in Mathborough for 2006 given the average monthly rainfall in 2005 and the increase in 2006 -/
theorem mathborough_rainfall_2006 
  (avg_2005 : ℝ) 
  (increase_2006 : ℝ) 
  (h1 : avg_2005 = 40) 
  (h2 : increase_2006 = 3) : 
  (avg_2005 + increase_2006) * 12 = 516 := by
  sorry

#check mathborough_rainfall_2006

end NUMINAMATH_CALUDE_mathborough_rainfall_2006_l3076_307683


namespace NUMINAMATH_CALUDE_right_triangle_sin_A_l3076_307626

/-- In a right triangle ABC with ∠C = 90°, where sides opposite to angles A, B, and C are a, b, and c respectively, sin A = a/c -/
theorem right_triangle_sin_A (A B C : ℝ) (a b c : ℝ) 
  (h_right : A + B + C = Real.pi)
  (h_C : C = Real.pi / 2)
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_pythagorean : a^2 + b^2 = c^2) :
  Real.sin A = a / c := by sorry

end NUMINAMATH_CALUDE_right_triangle_sin_A_l3076_307626


namespace NUMINAMATH_CALUDE_largest_solution_is_25_l3076_307661

theorem largest_solution_is_25 :
  ∃ (x : ℝ), (x^2 + x - 1 + |x^2 - (x - 1)|) / 2 = 35*x - 250 ∧
  x = 25 ∧
  ∀ (y : ℝ), (y^2 + y - 1 + |y^2 - (y - 1)|) / 2 = 35*y - 250 → y ≤ 25 :=
by sorry

end NUMINAMATH_CALUDE_largest_solution_is_25_l3076_307661


namespace NUMINAMATH_CALUDE_balls_total_weight_l3076_307633

/-- Represents the total weight of five colored metal balls -/
def total_weight (blue brown green red yellow : ℝ) : ℝ :=
  blue + brown + green + red + yellow

/-- Theorem stating the total weight of the balls -/
theorem balls_total_weight :
  ∃ (blue brown green red yellow : ℝ),
    blue = 6 ∧
    brown = 3.12 ∧
    green = 4.25 ∧
    red = 2 * green ∧
    yellow = red - 1.5 ∧
    total_weight blue brown green red yellow = 28.87 := by
  sorry

end NUMINAMATH_CALUDE_balls_total_weight_l3076_307633


namespace NUMINAMATH_CALUDE_even_increasing_function_solution_set_l3076_307672

-- Define the function f
def f (a b x : ℝ) : ℝ := (x - 2) * (a * x + b)

-- State the theorem
theorem even_increasing_function_solution_set
  (a b : ℝ)
  (h_even : ∀ x, f a b x = f a b (-x))
  (h_increasing : ∀ x y, 0 < x → x < y → f a b x < f a b y)
  : {x : ℝ | f a b (2 - x) > 0} = {x : ℝ | x < 0 ∨ x > 4} :=
by sorry

end NUMINAMATH_CALUDE_even_increasing_function_solution_set_l3076_307672


namespace NUMINAMATH_CALUDE_broken_eggs_count_l3076_307611

/-- Given a total of 24 eggs, where some are broken, some are cracked, and some are perfect,
    prove that the number of broken eggs is 3 under the following conditions:
    1. The number of cracked eggs is twice the number of broken eggs
    2. The difference between perfect and cracked eggs is 9 -/
theorem broken_eggs_count (broken : ℕ) (cracked : ℕ) (perfect : ℕ) : 
  perfect + cracked + broken = 24 →
  cracked = 2 * broken →
  perfect - cracked = 9 →
  broken = 3 := by sorry

end NUMINAMATH_CALUDE_broken_eggs_count_l3076_307611


namespace NUMINAMATH_CALUDE_greatest_common_divisor_under_60_l3076_307647

theorem greatest_common_divisor_under_60 : ∃ (n : ℕ), 
  n < 60 ∧ 
  n ∣ 546 ∧ 
  n ∣ 108 ∧ 
  (∀ m : ℕ, m < 60 → m ∣ 546 → m ∣ 108 → m ≤ n) ∧
  n = 42 := by
sorry

end NUMINAMATH_CALUDE_greatest_common_divisor_under_60_l3076_307647


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a2_l3076_307691

/-- An arithmetic sequence with a_1 = 1 and a_{n+2} - a_n = 3 has a_2 = 5/2 -/
theorem arithmetic_sequence_a2 (a : ℕ → ℚ) :
  a 1 = 1 →
  (∀ n : ℕ, a (n + 2) - a n = 3) →
  (∀ n : ℕ, ∃ d : ℚ, a (n + 1) = a n + d) →
  a 2 = 5/2 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a2_l3076_307691


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l3076_307687

/-- The lateral surface area of a cone with base radius 3 and height 4 is 15π. -/
theorem cone_lateral_surface_area :
  ∀ (r h : ℝ) (lateral_area : ℝ),
    r = 3 →
    h = 4 →
    lateral_area = π * r * (Real.sqrt (r^2 + h^2)) →
    lateral_area = 15 * π :=
by sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l3076_307687


namespace NUMINAMATH_CALUDE_smallest_product_l3076_307676

def digits : List ℕ := [5, 6, 7, 8]

def valid_arrangement (a b c d : ℕ) : Prop :=
  a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

def product (a b c d : ℕ) : ℕ := (10 * a + b) * (10 * c + d)

theorem smallest_product :
  ∀ a b c d : ℕ, valid_arrangement a b c d →
    product a b c d ≥ 4368 :=
sorry

end NUMINAMATH_CALUDE_smallest_product_l3076_307676


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3076_307673

theorem sqrt_equation_solution (x : ℝ) : 
  Real.sqrt (9 - 2 * x) = 5 → x = -8 := by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3076_307673


namespace NUMINAMATH_CALUDE_num_finches_is_four_l3076_307699

-- Define the constants based on the problem conditions
def parakeet_consumption : ℕ := 2 -- grams per day
def parrot_consumption : ℕ := 14 -- grams per day
def finch_consumption : ℕ := parakeet_consumption / 2 -- grams per day
def num_parakeets : ℕ := 3
def num_parrots : ℕ := 2
def total_birdseed : ℕ := 266 -- grams for a week
def days_in_week : ℕ := 7

-- Theorem to prove
theorem num_finches_is_four :
  ∃ (num_finches : ℕ),
    num_finches = 4 ∧
    total_birdseed = (num_parakeets * parakeet_consumption + 
                      num_parrots * parrot_consumption + 
                      num_finches * finch_consumption) * days_in_week :=
by
  sorry


end NUMINAMATH_CALUDE_num_finches_is_four_l3076_307699


namespace NUMINAMATH_CALUDE_gcd_problem_l3076_307653

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = 2 * k * 1171) :
  Int.gcd (3 * b^2 + 17 * b + 91) (b + 11) = 11 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l3076_307653


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3076_307651

theorem sqrt_equation_solution :
  ∀ x : ℚ, (x > 2) → (Real.sqrt (7 * x) / Real.sqrt (2 * (x - 2)) = 3) → x = 36 / 11 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3076_307651


namespace NUMINAMATH_CALUDE_weight_of_CCl4_l3076_307652

/-- The molar mass of Carbon in g/mol -/
def molar_mass_C : ℝ := 12.01

/-- The molar mass of Chlorine in g/mol -/
def molar_mass_Cl : ℝ := 35.45

/-- The number of Carbon atoms in a CCl4 molecule -/
def num_C_atoms : ℕ := 1

/-- The number of Chlorine atoms in a CCl4 molecule -/
def num_Cl_atoms : ℕ := 4

/-- The number of moles of CCl4 -/
def num_moles : ℝ := 8

/-- Theorem: The weight of 8 moles of CCl4 is 1230.48 grams -/
theorem weight_of_CCl4 : 
  let molar_mass_CCl4 := molar_mass_C * num_C_atoms + molar_mass_Cl * num_Cl_atoms
  num_moles * molar_mass_CCl4 = 1230.48 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_CCl4_l3076_307652


namespace NUMINAMATH_CALUDE_quadratic_form_inequality_l3076_307602

theorem quadratic_form_inequality (a b c d : ℝ) (h : a * d - b * c = 1) :
  a^2 + b^2 + c^2 + d^2 + a * c + b * d > 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_inequality_l3076_307602


namespace NUMINAMATH_CALUDE_ellipse_focal_distance_l3076_307657

/-- Given an ellipse with equation x²/(8-m) + y²/(m-2) = 1, 
    where the major axis is on the y-axis and the focal distance is 4,
    prove that the value of m is 7. -/
theorem ellipse_focal_distance (m : ℝ) : 
  (∀ x y : ℝ, x^2 / (8 - m) + y^2 / (m - 2) = 1) →  -- Ellipse equation
  (8 - m < m - 2) →                                -- Major axis on y-axis
  (m - 2 - (8 - m) = 4) →                          -- Focal distance is 4
  m = 7 := by
sorry

end NUMINAMATH_CALUDE_ellipse_focal_distance_l3076_307657


namespace NUMINAMATH_CALUDE_equation_D_is_quadratic_l3076_307631

/-- Definition of a quadratic equation in terms of x -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation x^2 - 5x = 0 -/
def equation_D (x : ℝ) : ℝ := x^2 - 5*x

/-- Theorem: equation_D is a quadratic equation -/
theorem equation_D_is_quadratic : is_quadratic_equation equation_D := by
  sorry

end NUMINAMATH_CALUDE_equation_D_is_quadratic_l3076_307631


namespace NUMINAMATH_CALUDE_matrix_power_difference_l3076_307688

def B : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; 0, 2]

theorem matrix_power_difference :
  B^20 - 3 • B^19 = !![0, 4 * 2^19; 0, -2^19] := by sorry

end NUMINAMATH_CALUDE_matrix_power_difference_l3076_307688


namespace NUMINAMATH_CALUDE_no_integer_solutions_l3076_307606

theorem no_integer_solutions : ¬∃ (x y : ℤ), x^2 - 4*y^2 = 2011 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l3076_307606


namespace NUMINAMATH_CALUDE_system_solution_existence_l3076_307621

theorem system_solution_existence (a b : ℝ) :
  (∃ x y : ℝ, x^2 + y^2 + x*y = a ∧ x^2 - y^2 = b) ↔ 
  -2*a ≤ Real.sqrt 3 * b ∧ Real.sqrt 3 * b ≤ 2*a :=
by sorry

end NUMINAMATH_CALUDE_system_solution_existence_l3076_307621


namespace NUMINAMATH_CALUDE_abc_inequality_l3076_307622

theorem abc_inequality (a b c : ℝ) (ha : 0 ≤ a ∧ a ≤ 2) (hb : 0 ≤ b ∧ b ≤ 2) (hc : 0 ≤ c ∧ c ≤ 2) :
  (a - b) * (b - c) * (a - c) ≤ 2 ∧
  ((a - b) * (b - c) * (a - c) = 2 ↔ 
    ((a = 2 ∧ b = 1 ∧ c = 0) ∨ (a = 1 ∧ b = 0 ∧ c = 2) ∨ (a = 0 ∧ b = 2 ∧ c = 1))) :=
by sorry


end NUMINAMATH_CALUDE_abc_inequality_l3076_307622

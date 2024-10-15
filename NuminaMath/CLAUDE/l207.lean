import Mathlib

namespace NUMINAMATH_CALUDE_product_from_lcm_gcd_l207_20768

theorem product_from_lcm_gcd (a b : ℕ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : Nat.lcm a b = 60) (h4 : Nat.gcd a b = 5) : a * b = 300 := by
  sorry

end NUMINAMATH_CALUDE_product_from_lcm_gcd_l207_20768


namespace NUMINAMATH_CALUDE_opposite_of_negative_two_thirds_l207_20778

theorem opposite_of_negative_two_thirds :
  -(-(2/3 : ℚ)) = 2/3 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_two_thirds_l207_20778


namespace NUMINAMATH_CALUDE_no_solution_for_absolute_value_equation_l207_20752

theorem no_solution_for_absolute_value_equation :
  ¬ ∃ x : ℝ, |x - 4| = x^2 + 6*x + 8 := by
sorry

end NUMINAMATH_CALUDE_no_solution_for_absolute_value_equation_l207_20752


namespace NUMINAMATH_CALUDE_max_value_of_d_l207_20703

theorem max_value_of_d (a b c d : ℝ) 
  (sum_eq : a + b + c + d = 10)
  (sum_products_eq : a*b + a*c + a*d + b*c + b*d + c*d = 20) :
  d ≤ (5 + Real.sqrt 105) / 2 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_d_l207_20703


namespace NUMINAMATH_CALUDE_stirling_second_kind_l207_20784

/-- Stirling number of the second kind -/
def S (n k : ℕ) : ℚ :=
  sorry

/-- Main theorem for Stirling numbers of the second kind -/
theorem stirling_second_kind (n : ℕ) (h : n ≥ 2) :
  (∀ k, k ≥ 2 → S n k = k * S (n-1) k + S (n-1) (k-1)) ∧
  S n 1 = 1 ∧
  S n 2 = 2^(n-1) - 1 ∧
  S n 3 = (1/6) * 3^n - (1/2) * 2^n + 1/2 ∧
  S n 4 = (1/24) * 4^n - (1/6) * 3^n + (1/4) * 2^n - 1/6 :=
by
  sorry

end NUMINAMATH_CALUDE_stirling_second_kind_l207_20784


namespace NUMINAMATH_CALUDE_three_digit_number_operation_l207_20755

theorem three_digit_number_operation (a b c : ℕ) : 
  a ≥ 1 ∧ a ≤ 9 ∧ b ≥ 0 ∧ b ≤ 9 ∧ c ≥ 0 ∧ c ≤ 9 ∧ a = c + 3 →
  (4 * (100 * a + 10 * b + c) - (100 * c + 10 * b + a)) % 10 = 7 := by
sorry

end NUMINAMATH_CALUDE_three_digit_number_operation_l207_20755


namespace NUMINAMATH_CALUDE_angle_with_supplement_four_times_complement_l207_20717

theorem angle_with_supplement_four_times_complement (x : ℝ) : 
  (180 - x = 4 * (90 - x)) → x = 60 := by
  sorry

end NUMINAMATH_CALUDE_angle_with_supplement_four_times_complement_l207_20717


namespace NUMINAMATH_CALUDE_screening_methods_count_l207_20786

/-- The number of units showing the documentary -/
def num_units : ℕ := 4

/-- The number of different screening methods -/
def screening_methods : ℕ := num_units ^ num_units

/-- Theorem stating that the number of different screening methods
    is equal to 4^4 when there are 4 units each showing the film once -/
theorem screening_methods_count :
  screening_methods = 4^4 :=
by sorry

end NUMINAMATH_CALUDE_screening_methods_count_l207_20786


namespace NUMINAMATH_CALUDE_gumball_price_l207_20746

/-- Given that Melanie sells 4 gumballs for a total of 32 cents, prove that each gumball costs 8 cents. -/
theorem gumball_price (num_gumballs : ℕ) (total_cents : ℕ) (price_per_gumball : ℕ) 
  (h1 : num_gumballs = 4)
  (h2 : total_cents = 32)
  (h3 : price_per_gumball * num_gumballs = total_cents) :
  price_per_gumball = 8 := by
  sorry

end NUMINAMATH_CALUDE_gumball_price_l207_20746


namespace NUMINAMATH_CALUDE_triangles_on_circle_l207_20719

theorem triangles_on_circle (n : ℕ) (h : n = 15) : 
  (Nat.choose n 3) = 455 := by sorry

end NUMINAMATH_CALUDE_triangles_on_circle_l207_20719


namespace NUMINAMATH_CALUDE_division_value_proof_l207_20793

theorem division_value_proof (x : ℝ) : (9 / x) * 12 = 18 → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_division_value_proof_l207_20793


namespace NUMINAMATH_CALUDE_one_tree_baskets_l207_20799

/-- The number of apples that can fit in one basket -/
def apples_per_basket : ℕ := 15

/-- The number of apples produced by 10 trees -/
def apples_from_ten_trees : ℕ := 3000

/-- The number of trees -/
def number_of_trees : ℕ := 10

/-- Theorem: One apple tree can fill 20 baskets -/
theorem one_tree_baskets : 
  (apples_from_ten_trees / number_of_trees) / apples_per_basket = 20 := by
  sorry

end NUMINAMATH_CALUDE_one_tree_baskets_l207_20799


namespace NUMINAMATH_CALUDE_turtle_position_and_distance_l207_20767

def turtle_movements : List Int := [-8, 7, -3, 9, -6, -4, 10]

theorem turtle_position_and_distance :
  (List.sum turtle_movements = 5) ∧
  (List.sum (List.map Int.natAbs turtle_movements) = 47) := by
  sorry

end NUMINAMATH_CALUDE_turtle_position_and_distance_l207_20767


namespace NUMINAMATH_CALUDE_rose_cost_is_six_l207_20723

/-- The cost of each rose when buying in bulk -/
def rose_cost (dozen : ℕ) (discount_percent : ℚ) (total_paid : ℚ) : ℚ :=
  total_paid / (discount_percent / 100) / (dozen * 12)

/-- Theorem: The cost of each rose is $6 -/
theorem rose_cost_is_six :
  rose_cost 5 80 288 = 6 := by
  sorry

end NUMINAMATH_CALUDE_rose_cost_is_six_l207_20723


namespace NUMINAMATH_CALUDE_min_exponent_sum_l207_20707

/-- Given a positive integer A that can be factorized as A = 2^α × 3^β × 5^γ,
    where α, β, and γ are natural numbers, and satisfying the conditions:
    - A/2 is a perfect square
    - A/3 is a perfect cube
    - A/5 is a perfect fifth power
    The minimum value of α + β + γ is 31. -/
theorem min_exponent_sum (A : ℕ+) (α β γ : ℕ) 
  (h_factorization : (A : ℕ) = 2^α * 3^β * 5^γ)
  (h_half_square : ∃ k : ℕ, 2 * k^2 = A)
  (h_third_cube : ∃ k : ℕ, 3 * k^3 = A)
  (h_fifth_power : ∃ k : ℕ, 5 * k^5 = A) :
  α + β + γ ≥ 31 :=
sorry

end NUMINAMATH_CALUDE_min_exponent_sum_l207_20707


namespace NUMINAMATH_CALUDE_dalton_savings_proof_l207_20773

/-- The amount of money Dalton saved from his allowance -/
def dalton_savings : ℕ := sorry

/-- The cost of all items Dalton wants to buy -/
def total_cost : ℕ := 23

/-- The amount Dalton's uncle gave him -/
def uncle_contribution : ℕ := 13

/-- The additional amount Dalton needs -/
def additional_needed : ℕ := 4

theorem dalton_savings_proof :
  dalton_savings = total_cost - uncle_contribution - additional_needed :=
by sorry

end NUMINAMATH_CALUDE_dalton_savings_proof_l207_20773


namespace NUMINAMATH_CALUDE_average_sitting_time_l207_20730

def num_students : ℕ := 6
def num_seats : ℕ := 4
def travel_time_hours : ℕ := 3
def travel_time_minutes : ℕ := 12

theorem average_sitting_time :
  let total_minutes : ℕ := travel_time_hours * 60 + travel_time_minutes
  let total_sitting_time : ℕ := num_seats * total_minutes
  let avg_sitting_time : ℕ := total_sitting_time / num_students
  avg_sitting_time = 128 := by
sorry

end NUMINAMATH_CALUDE_average_sitting_time_l207_20730


namespace NUMINAMATH_CALUDE_quartic_equation_roots_l207_20741

theorem quartic_equation_roots (a b : ℝ) :
  let x : ℝ → Prop := λ x => x^4 - 2*a*x^2 + b^2 = 0
  ∃ (ε₁ ε₂ : {r : ℝ // r = 1 ∨ r = -1}),
    x (ε₁ * (Real.sqrt ((a + b)/2) + ε₂ * Real.sqrt ((a - b)/2))) :=
by
  sorry

end NUMINAMATH_CALUDE_quartic_equation_roots_l207_20741


namespace NUMINAMATH_CALUDE_quadratic_inequality_l207_20756

/-- Represents a quadratic function of the form f(x) = -2ax^2 + ax - 4 where a > 0 -/
def QuadraticFunction (a : ℝ) : ℝ → ℝ := 
  fun x ↦ -2 * a * x^2 + a * x - 4

theorem quadratic_inequality (a : ℝ) (ha : a > 0) :
  let f := QuadraticFunction a
  f 2 < f (-1) ∧ f (-1) < f 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l207_20756


namespace NUMINAMATH_CALUDE_sequence_not_ap_or_gp_l207_20791

-- Define the sequence
def a : ℕ → ℕ
  | n => if n % 2 = 0 then ((n / 2) + 1)^2 else (n / 2 + 1) * (n / 2 + 2)

-- State the theorem
theorem sequence_not_ap_or_gp :
  -- The sequence is increasing
  (∀ n : ℕ, a n < a (n + 1)) ∧
  -- Each even-indexed term is the arithmetic mean of its neighbors
  (∀ n : ℕ, a (2 * n) = (a (2 * n - 1) + a (2 * n + 1)) / 2) ∧
  -- Each odd-indexed term is the geometric mean of its neighbors
  (∀ n : ℕ, n > 0 → a (2 * n - 1) = Int.sqrt (a (2 * n - 2) * a (2 * n))) ∧
  -- The sequence never becomes an arithmetic progression
  (∀ k : ℕ, ∃ n : ℕ, n ≥ k ∧ a (n + 2) - a (n + 1) ≠ a (n + 1) - a n) ∧
  -- The sequence never becomes a geometric progression
  (∀ k : ℕ, ∃ n : ℕ, n ≥ k ∧ a (n + 2) * a n ≠ (a (n + 1))^2) :=
by sorry

end NUMINAMATH_CALUDE_sequence_not_ap_or_gp_l207_20791


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l207_20779

theorem arithmetic_mean_of_fractions (x a b : ℝ) (hx : x ≠ 0) (hb : b ≠ 0) :
  (((x + a + b) / x + (x - a - b) / x) / 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l207_20779


namespace NUMINAMATH_CALUDE_total_jellybeans_needed_l207_20747

/-- The number of jellybeans needed to fill a large glass -/
def large_glass_beans : ℕ := 50

/-- The number of jellybeans needed to fill a small glass -/
def small_glass_beans : ℕ := large_glass_beans / 2

/-- The number of large glasses -/
def num_large_glasses : ℕ := 5

/-- The number of small glasses -/
def num_small_glasses : ℕ := 3

/-- Theorem: The total number of jellybeans needed to fill all glasses is 325 -/
theorem total_jellybeans_needed : 
  num_large_glasses * large_glass_beans + num_small_glasses * small_glass_beans = 325 := by
  sorry

end NUMINAMATH_CALUDE_total_jellybeans_needed_l207_20747


namespace NUMINAMATH_CALUDE_empty_set_subset_of_all_l207_20713

theorem empty_set_subset_of_all (A : Set α) : ∅ ⊆ A := by
  sorry

end NUMINAMATH_CALUDE_empty_set_subset_of_all_l207_20713


namespace NUMINAMATH_CALUDE_two_red_balls_in_bag_l207_20775

/-- Represents the contents of a bag of balls -/
structure BagOfBalls where
  redBalls : ℕ
  yellowBalls : ℕ

/-- The probability of selecting a yellow ball given another yellow ball was selected -/
def probYellowGivenYellow (bag : BagOfBalls) : ℚ :=
  (bag.yellowBalls - 1) / (bag.redBalls + bag.yellowBalls - 1)

theorem two_red_balls_in_bag :
  ∀ (bag : BagOfBalls),
    bag.yellowBalls = 3 →
    probYellowGivenYellow bag = 1/2 →
    bag.redBalls = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_red_balls_in_bag_l207_20775


namespace NUMINAMATH_CALUDE_probability_log_integer_l207_20751

def S : Set ℕ := {n | ∃ k : ℕ, 1 ≤ k ∧ k ≤ 15 ∧ n = 3^k}

def is_valid_pair (a b : ℕ) : Prop :=
  a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ ∃ k : ℕ, b = a^k

def total_pairs : ℕ := Nat.choose 15 2

def valid_pairs : ℕ := 30

theorem probability_log_integer :
  (valid_pairs : ℚ) / total_pairs = 2 / 7 := by sorry

end NUMINAMATH_CALUDE_probability_log_integer_l207_20751


namespace NUMINAMATH_CALUDE_largest_number_l207_20788

theorem largest_number : 
  let a := 0.965
  let b := 0.9687
  let c := 0.9618
  let d := 0.955
  b > a ∧ b > c ∧ b > d := by
  sorry

end NUMINAMATH_CALUDE_largest_number_l207_20788


namespace NUMINAMATH_CALUDE_log_product_equals_two_l207_20796

theorem log_product_equals_two (y : ℝ) (h : y > 0) : 
  (Real.log y / Real.log 3) * (Real.log 9 / Real.log y) = 2 → y = 9 := by
  sorry

end NUMINAMATH_CALUDE_log_product_equals_two_l207_20796


namespace NUMINAMATH_CALUDE_tetrahedron_cross_section_perimeter_bounds_l207_20753

/-- A regular tetrahedron -/
structure RegularTetrahedron where
  edge_length : ℝ
  edge_length_pos : edge_length > 0

/-- A quadrilateral cross-section of a regular tetrahedron -/
structure TetrahedronCrossSection (t : RegularTetrahedron) where
  perimeter : ℝ
  is_quadrilateral : True  -- This is a placeholder for the quadrilateral property

/-- The perimeter of a quadrilateral cross-section of a regular tetrahedron 
    is between 2a and 3a, where a is the edge length of the tetrahedron -/
theorem tetrahedron_cross_section_perimeter_bounds 
  (t : RegularTetrahedron) (c : TetrahedronCrossSection t) : 
  2 * t.edge_length ≤ c.perimeter ∧ c.perimeter ≤ 3 * t.edge_length :=
sorry

end NUMINAMATH_CALUDE_tetrahedron_cross_section_perimeter_bounds_l207_20753


namespace NUMINAMATH_CALUDE_teacher_selection_problem_l207_20721

/-- The number of ways to select k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

theorem teacher_selection_problem (male female total selected : ℕ) 
  (h1 : male = 3)
  (h2 : female = 6)
  (h3 : total = male + female)
  (h4 : selected = 5) :
  choose total selected - choose female selected = 120 := by sorry

end NUMINAMATH_CALUDE_teacher_selection_problem_l207_20721


namespace NUMINAMATH_CALUDE_greater_number_problem_l207_20776

theorem greater_number_problem (x y : ℝ) (h1 : x > y) (h2 : x > 0) (h3 : y > 0) 
  (h4 : x * y = 2048) (h5 : (x + y) - (x - y) = 64) : x = 64 := by
  sorry

end NUMINAMATH_CALUDE_greater_number_problem_l207_20776


namespace NUMINAMATH_CALUDE_part_one_part_two_l207_20726

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| + |x - a|

-- Part 1
theorem part_one : 
  {x : ℝ | f 1 x ≥ 2} = {x : ℝ | x ≤ 0 ∨ x ≥ 2} := by sorry

-- Part 2
theorem part_two : 
  ∀ a : ℝ, a > 1 → (∀ x : ℝ, f a x + |x - 1| ≥ 2) ↔ a ≥ 3 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l207_20726


namespace NUMINAMATH_CALUDE_complex_fraction_equals_25_l207_20724

theorem complex_fraction_equals_25 :
  ((5 / 2) / (1 / 2) * (5 / 2)) / ((5 / 2) * (1 / 2) / (5 / 2)) = 25 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equals_25_l207_20724


namespace NUMINAMATH_CALUDE_circle_intersection_l207_20708

def circle1 (x y : ℝ) : Prop := (x - 2)^2 + (y - 10)^2 = 50

def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*(x - y) - 18 = 0

theorem circle_intersection :
  ∃ (x1 y1 x2 y2 : ℝ),
    (circle1 x1 y1 ∧ circle2 x1 y1) ∧
    (circle1 x2 y2 ∧ circle2 x2 y2) ∧
    (x1 = 3 ∧ y1 = 3) ∧
    (x2 = -3 ∧ y2 = 5) :=
  sorry

end NUMINAMATH_CALUDE_circle_intersection_l207_20708


namespace NUMINAMATH_CALUDE_group_size_is_factor_l207_20771

def num_cows : ℕ := 24
def num_sheep : ℕ := 7
def num_goats : ℕ := 113

def total_animals : ℕ := num_cows + num_sheep + num_goats

theorem group_size_is_factor :
  ∀ (group_size : ℕ), 
    group_size > 1 ∧ 
    group_size < total_animals ∧ 
    total_animals % group_size = 0 →
    ∃ (num_groups : ℕ), num_groups * group_size = total_animals :=
by sorry

end NUMINAMATH_CALUDE_group_size_is_factor_l207_20771


namespace NUMINAMATH_CALUDE_positive_integers_sum_product_l207_20794

theorem positive_integers_sum_product (P Q : ℕ+) (h : P + Q + P * Q = 90) : P + Q = 18 := by
  sorry

end NUMINAMATH_CALUDE_positive_integers_sum_product_l207_20794


namespace NUMINAMATH_CALUDE_cubic_equation_q_expression_l207_20785

theorem cubic_equation_q_expression (a b q r : ℝ) (h1 : b ≠ 0) :
  (∃ (x : ℂ), x^3 + q*x + r = 0 ∧ (x = a + b*I ∨ x = a - b*I)) →
  q = b^2 - 3*a^2 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_q_expression_l207_20785


namespace NUMINAMATH_CALUDE_vector_parallel_if_negative_l207_20729

variable {n : Type*} [NormedAddCommGroup n] [InnerProductSpace ℝ n]

def parallel (a b : n) : Prop := ∃ (k : ℝ), a = k • b

theorem vector_parallel_if_negative (a b : n) : a = -b → parallel a b := by
  sorry

end NUMINAMATH_CALUDE_vector_parallel_if_negative_l207_20729


namespace NUMINAMATH_CALUDE_line_slope_l207_20740

theorem line_slope (x y : ℝ) :
  (x / 2 + y / 3 = 1) → (∃ m b : ℝ, y = m * x + b ∧ m = -3/2) :=
by
  sorry

end NUMINAMATH_CALUDE_line_slope_l207_20740


namespace NUMINAMATH_CALUDE_max_diff_color_pairs_l207_20718

/-- Represents a grid of black and white cells -/
structure Grid :=
  (size : Nat)
  (black_cells : Fin size → Fin size → Bool)

/-- The number of black cells in a given row -/
def row_black_count (g : Grid) (row : Fin g.size) : Nat :=
  (List.range g.size).count (λ col ↦ g.black_cells row col)

/-- The number of black cells in a given column -/
def col_black_count (g : Grid) (col : Fin g.size) : Nat :=
  (List.range g.size).count (λ row ↦ g.black_cells row col)

/-- The number of pairs of adjacent differently colored cells -/
def diff_color_pairs (g : Grid) : Nat :=
  sorry

/-- The theorem statement -/
theorem max_diff_color_pairs :
  ∃ (g : Grid),
    g.size = 100 ∧
    (∀ col₁ col₂ : Fin g.size, col_black_count g col₁ = col_black_count g col₂) ∧
    (∀ row₁ row₂ : Fin g.size, row₁ ≠ row₂ → row_black_count g row₁ ≠ row_black_count g row₂) ∧
    (∀ g' : Grid,
      g'.size = 100 →
      (∀ col₁ col₂ : Fin g'.size, col_black_count g' col₁ = col_black_count g' col₂) →
      (∀ row₁ row₂ : Fin g'.size, row₁ ≠ row₂ → row_black_count g' row₁ ≠ row_black_count g' row₂) →
      diff_color_pairs g' ≤ diff_color_pairs g) ∧
    diff_color_pairs g = 14601 :=
  sorry

end NUMINAMATH_CALUDE_max_diff_color_pairs_l207_20718


namespace NUMINAMATH_CALUDE_cube_volume_surface_area_l207_20706

theorem cube_volume_surface_area (x : ℝ) : 
  (∃ (s : ℝ), s > 0 ∧ s^3 = 3*x ∧ 6*s^2 = x) → x = 5832 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_surface_area_l207_20706


namespace NUMINAMATH_CALUDE_expansion_coefficient_x_fifth_l207_20701

theorem expansion_coefficient_x_fifth (x : ℝ) :
  ∃ (aₙ a₁ a₂ a₃ a₄ a₅ : ℝ),
    x^5 = aₙ + a₁*(x-1) + a₂*(x-1)^2 + a₃*(x-1)^3 + a₄*(x-1)^4 + a₅*(x-1)^5 ∧
    a₄ = 5 := by
  sorry

end NUMINAMATH_CALUDE_expansion_coefficient_x_fifth_l207_20701


namespace NUMINAMATH_CALUDE_bill_donuts_l207_20722

theorem bill_donuts (total : ℕ) (secretary_takes : ℕ) (final : ℕ) : 
  total = 50 →
  secretary_takes = 4 →
  final = 22 →
  final * 2 = total - secretary_takes - (total - secretary_takes - final * 2) :=
by sorry

end NUMINAMATH_CALUDE_bill_donuts_l207_20722


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l207_20760

theorem problem_1 : (-1/12 - 1/16 + 3/4 - 1/6) * (-48) = -21 := by sorry

theorem problem_2 : -99*(8/9) * 8 = -799*(1/9) := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l207_20760


namespace NUMINAMATH_CALUDE_simplify_expression_l207_20789

theorem simplify_expression (x : ℝ) : (3*x - 4)*(x + 8) - (x + 6)*(3*x - 2) = 4*x - 20 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l207_20789


namespace NUMINAMATH_CALUDE_max_sum_in_S_l207_20732

/-- The set of ordered pairs of integers (x,y) satisfying x^2 + y^2 = 50 -/
def S : Set (ℤ × ℤ) := {p | p.1^2 + p.2^2 = 50}

/-- The theorem stating that the maximum sum of x+y for (x,y) in S is 10 -/
theorem max_sum_in_S : (⨆ p ∈ S, (p.1 + p.2 : ℤ)) = 10 := by sorry

end NUMINAMATH_CALUDE_max_sum_in_S_l207_20732


namespace NUMINAMATH_CALUDE_price_per_kg_correct_l207_20745

/-- The price per kilogram of rooster -/
def price_per_kg : ℝ := 0.5

/-- The weight of the first rooster in kilograms -/
def weight1 : ℝ := 30

/-- The weight of the second rooster in kilograms -/
def weight2 : ℝ := 40

/-- The total earnings from selling both roosters -/
def total_earnings : ℝ := 35

/-- Theorem stating that the price per kilogram is correct -/
theorem price_per_kg_correct : 
  price_per_kg * (weight1 + weight2) = total_earnings := by sorry

end NUMINAMATH_CALUDE_price_per_kg_correct_l207_20745


namespace NUMINAMATH_CALUDE_ratio_sum_over_y_l207_20744

theorem ratio_sum_over_y (x y : ℚ) (h : x / y = 1 / 2) : (x + y) / y = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_sum_over_y_l207_20744


namespace NUMINAMATH_CALUDE_class_average_l207_20704

theorem class_average (total_students : ℕ) (high_scorers : ℕ) (zero_scorers : ℕ) (high_score : ℝ) (rest_average : ℝ) :
  total_students = 25 →
  high_scorers = 3 →
  zero_scorers = 3 →
  high_score = 95 →
  rest_average = 45 →
  let rest_students := total_students - high_scorers - zero_scorers
  let total_marks := high_scorers * high_score + zero_scorers * 0 + rest_students * rest_average
  total_marks / total_students = 45.6 := by
sorry

end NUMINAMATH_CALUDE_class_average_l207_20704


namespace NUMINAMATH_CALUDE_chord_intersection_triangles_l207_20774

/-- The number of points on the circle -/
def n : ℕ := 10

/-- The number of chords is the number of ways to choose 2 points from n points -/
def num_chords : ℕ := n.choose 2

/-- The number of intersection points is the number of ways to choose 4 points from n points -/
def num_intersections : ℕ := n.choose 4

/-- The number of triangles is the number of ways to choose 3 intersection points -/
def num_triangles : ℕ := num_intersections.choose 3

/-- Theorem stating the number of triangles formed by chord intersections -/
theorem chord_intersection_triangles :
  num_triangles = 1524180 :=
sorry

end NUMINAMATH_CALUDE_chord_intersection_triangles_l207_20774


namespace NUMINAMATH_CALUDE_hippopotamus_cards_l207_20728

theorem hippopotamus_cards (initial_cards remaining_cards : ℕ) : 
  initial_cards = 72 → remaining_cards = 11 → initial_cards - remaining_cards = 61 := by
  sorry

end NUMINAMATH_CALUDE_hippopotamus_cards_l207_20728


namespace NUMINAMATH_CALUDE_ellipse_minor_axis_length_l207_20750

/-- An ellipse with axes parallel to the coordinate axes passing through 
    the given points has a minor axis length of 4. -/
theorem ellipse_minor_axis_length : 
  ∀ (e : Set (ℝ × ℝ)),
  (∀ (x y : ℝ), (x, y) ∈ e ↔ ((x - 3/2)^2 / 3^2) + ((y - 1)^2 / b^2) = 1) →
  (0, 0) ∈ e →
  (0, 2) ∈ e →
  (3, 0) ∈ e →
  (3, 2) ∈ e →
  (3/2, 3) ∈ e →
  ∃ (b : ℝ), b = 2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_minor_axis_length_l207_20750


namespace NUMINAMATH_CALUDE_final_numbers_l207_20743

def process (n : ℕ) : Set ℕ :=
  {m : ℕ | ∃ (i j : ℕ), i ≤ n ∧ j ≤ n ∧ m = i * j}

theorem final_numbers (n : ℕ) :
  process n = {m : ℕ | ∃ (k : ℕ), k ≤ n ∧ m = k^2} :=
by sorry

#check final_numbers 2009

end NUMINAMATH_CALUDE_final_numbers_l207_20743


namespace NUMINAMATH_CALUDE_passenger_speed_on_train_l207_20765

/-- The speed of a passenger relative to the railway track when moving on a train -/
def passenger_speed_relative_to_track (train_speed passenger_speed : ℝ) : ℝ × ℝ :=
  (train_speed + passenger_speed, |train_speed - passenger_speed|)

/-- Theorem: The speed of a passenger relative to the railway track
    when the train moves at 60 km/h and the passenger moves at 3 km/h relative to the train -/
theorem passenger_speed_on_train :
  let train_speed := 60
  let passenger_speed := 3
  passenger_speed_relative_to_track train_speed passenger_speed = (63, 57) := by
  sorry

end NUMINAMATH_CALUDE_passenger_speed_on_train_l207_20765


namespace NUMINAMATH_CALUDE_cubic_fraction_equals_fifteen_l207_20764

theorem cubic_fraction_equals_fifteen :
  let a : ℤ := 7
  let b : ℤ := 5
  let c : ℤ := 3
  (a^3 + b^3 + c^3) / (a^2 - a*b + b^2 - b*c + c^2) = 15 := by
  sorry

end NUMINAMATH_CALUDE_cubic_fraction_equals_fifteen_l207_20764


namespace NUMINAMATH_CALUDE_binomial_variance_calculation_l207_20763

/-- A random variable following a binomial distribution B(n, p) -/
structure BinomialVariable where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- Expected value of a binomial variable -/
def expectedValue (ξ : BinomialVariable) : ℝ := ξ.n * ξ.p

/-- Variance of a binomial variable -/
def variance (ξ : BinomialVariable) : ℝ := ξ.n * ξ.p * (1 - ξ.p)

theorem binomial_variance_calculation (ξ : BinomialVariable) 
  (h_n : ξ.n = 36) 
  (h_exp : expectedValue ξ = 12) : 
  variance ξ = 8 := by
  sorry

end NUMINAMATH_CALUDE_binomial_variance_calculation_l207_20763


namespace NUMINAMATH_CALUDE_residue_of_11_pow_2021_mod_19_l207_20749

theorem residue_of_11_pow_2021_mod_19 :
  (11 : ℤ) ^ 2021 ≡ 17 [ZMOD 19] := by
  sorry

end NUMINAMATH_CALUDE_residue_of_11_pow_2021_mod_19_l207_20749


namespace NUMINAMATH_CALUDE_power_equation_solution_l207_20762

theorem power_equation_solution :
  ∃ x : ℝ, (5 : ℝ)^(x + 2) = 625 ∧ x = 2 := by sorry

end NUMINAMATH_CALUDE_power_equation_solution_l207_20762


namespace NUMINAMATH_CALUDE_max_value_sqrt_x_1_minus_9x_l207_20710

theorem max_value_sqrt_x_1_minus_9x (x : ℝ) (h1 : 0 < x) (h2 : x < 1/9) :
  ∃ (max_val : ℝ), max_val = 1/6 ∧ ∀ y, 0 < y ∧ y < 1/9 → Real.sqrt (y * (1 - 9*y)) ≤ max_val :=
by sorry

end NUMINAMATH_CALUDE_max_value_sqrt_x_1_minus_9x_l207_20710


namespace NUMINAMATH_CALUDE_quadratic_root_property_l207_20715

theorem quadratic_root_property (m : ℝ) (x₁ x₂ : ℝ) : 
  (2 * x₁^2 + 4 * x₁ + m = 0) →
  (2 * x₂^2 + 4 * x₂ + m = 0) →
  (x₁^2 + x₂^2 + 2*x₁*x₂ - x₁^2*x₂^2 = 0) →
  m = -4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_property_l207_20715


namespace NUMINAMATH_CALUDE_power_fraction_equality_l207_20709

theorem power_fraction_equality : (2^8 : ℚ) / (8^2 : ℚ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_power_fraction_equality_l207_20709


namespace NUMINAMATH_CALUDE_smallest_largest_multiples_l207_20700

theorem smallest_largest_multiples :
  ∃ (smallest largest : ℕ),
    (smallest ≥ 10 ∧ smallest < 100) ∧
    (largest ≥ 100 ∧ largest < 1000) ∧
    (∀ n : ℕ, n ≥ 10 ∧ n < 100 ∧ 2 ∣ n ∧ 3 ∣ n ∧ 5 ∣ n → n ≥ smallest) ∧
    (∀ n : ℕ, n ≥ 100 ∧ n < 1000 ∧ 2 ∣ n ∧ 3 ∣ n ∧ 5 ∣ n → n ≤ largest) ∧
    2 ∣ smallest ∧ 3 ∣ smallest ∧ 5 ∣ smallest ∧
    2 ∣ largest ∧ 3 ∣ largest ∧ 5 ∣ largest ∧
    smallest = 30 ∧ largest = 990 := by
  sorry

end NUMINAMATH_CALUDE_smallest_largest_multiples_l207_20700


namespace NUMINAMATH_CALUDE_hexagon_area_l207_20731

/-- A regular hexagon with vertices P and R -/
structure RegularHexagon where
  P : ℝ × ℝ
  R : ℝ × ℝ

/-- The area of a regular hexagon -/
def area (h : RegularHexagon) : ℝ := sorry

/-- Theorem: The area of a regular hexagon with P at (0,0) and R at (10,2) is 156√3 -/
theorem hexagon_area :
  let h : RegularHexagon := { P := (0, 0), R := (10, 2) }
  area h = 156 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_hexagon_area_l207_20731


namespace NUMINAMATH_CALUDE_mod_equivalence_unique_solution_l207_20736

theorem mod_equivalence_unique_solution : 
  ∃! n : ℤ, 0 ≤ n ∧ n ≤ 15 ∧ n ≡ 15827 [ZMOD 16] ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_mod_equivalence_unique_solution_l207_20736


namespace NUMINAMATH_CALUDE_smallest_n_all_digits_odd_l207_20770

/-- Function to check if all digits of a natural number are odd -/
def allDigitsOdd (n : ℕ) : Prop := sorry

/-- The smallest integer greater than 1 such that all digits of 9997n are odd -/
def smallestN : ℕ := 3335

theorem smallest_n_all_digits_odd :
  smallestN > 1 ∧
  allDigitsOdd (9997 * smallestN) ∧
  ∀ m : ℕ, m > 1 → m < smallestN → ¬(allDigitsOdd (9997 * m)) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_all_digits_odd_l207_20770


namespace NUMINAMATH_CALUDE_set_A_equals_roster_l207_20782

def A : Set ℤ := {x | ∃ (n : ℕ+), 6 / (5 - x) = n}

theorem set_A_equals_roster : A = {-1, 2, 3, 4} := by sorry

end NUMINAMATH_CALUDE_set_A_equals_roster_l207_20782


namespace NUMINAMATH_CALUDE_secret_codes_count_l207_20787

/-- The number of colors available for the secret code -/
def num_colors : ℕ := 7

/-- The number of slots in the secret code -/
def num_slots : ℕ := 4

/-- The number of possible secret codes -/
def num_codes : ℕ := num_colors ^ num_slots

/-- Theorem: The number of possible secret codes is 2401 -/
theorem secret_codes_count : num_codes = 2401 := by
  sorry

end NUMINAMATH_CALUDE_secret_codes_count_l207_20787


namespace NUMINAMATH_CALUDE_no_rational_solution_for_odd_coeff_quadratic_l207_20712

theorem no_rational_solution_for_odd_coeff_quadratic 
  (a b c : ℤ) (ha : Odd a) (hb : Odd b) (hc : Odd c) :
  ∀ x : ℚ, a * x^2 + b * x + c ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_rational_solution_for_odd_coeff_quadratic_l207_20712


namespace NUMINAMATH_CALUDE_frequency_of_score_range_l207_20742

theorem frequency_of_score_range (total_students : ℕ) (high_scorers : ℕ) 
  (h1 : total_students = 50) (h2 : high_scorers = 10) : 
  (high_scorers : ℚ) / total_students = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_frequency_of_score_range_l207_20742


namespace NUMINAMATH_CALUDE_kickball_students_total_l207_20734

theorem kickball_students_total (wednesday : ℕ) (fewer_thursday : ℕ) : 
  wednesday = 37 → fewer_thursday = 9 → 
  wednesday + (wednesday - fewer_thursday) = 65 := by
  sorry

end NUMINAMATH_CALUDE_kickball_students_total_l207_20734


namespace NUMINAMATH_CALUDE_two_digit_numbers_problem_l207_20720

def F (p q : ℕ) : ℚ :=
  let p1 := p / 10
  let p2 := p % 10
  let q1 := q / 10
  let q2 := q % 10
  let sum := (1000 * p1 + 100 * q1 + 10 * q2 + p2) + (1000 * q1 + 100 * p1 + 10 * p2 + q2)
  (sum : ℚ) / 11

theorem two_digit_numbers_problem (m n : ℕ) 
  (hm : m ≤ 9) (hn : 1 ≤ n ∧ n ≤ 9) :
  let a := 10 + m
  let b := 10 * n + 5
  150 * F a 18 + F b 26 = 32761 →
  m + n = 12 ∨ m + n = 11 ∨ m + n = 10 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_numbers_problem_l207_20720


namespace NUMINAMATH_CALUDE_next_perfect_square_with_two_twos_l207_20735

/-- A number begins with two 2s if its first two digits are 2 when written in base 10. -/
def begins_with_two_twos (n : ℕ) : Prop :=
  n ≥ 220 ∧ n < 230

/-- The sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

/-- A perfect square is a natural number that is the square of another natural number. -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem next_perfect_square_with_two_twos : 
  (∀ n : ℕ, is_perfect_square n ∧ begins_with_two_twos n ∧ n < 2500 → n ≤ 225) ∧
  is_perfect_square 2500 ∧
  begins_with_two_twos 2500 ∧
  sum_of_digits 2500 = 7 :=
sorry

end NUMINAMATH_CALUDE_next_perfect_square_with_two_twos_l207_20735


namespace NUMINAMATH_CALUDE_inequality_system_implies_a_leq_3_l207_20705

theorem inequality_system_implies_a_leq_3 :
  (∀ x : ℝ, (4 * (x - 1) > 3 * x - 1 ∧ 5 * x > 3 * x + 2 * a) ↔ x > 3) →
  a ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_implies_a_leq_3_l207_20705


namespace NUMINAMATH_CALUDE_integral_reciprocal_x_l207_20797

theorem integral_reciprocal_x : ∫ x in (1:ℝ)..2, (1:ℝ) / x = Real.log 2 := by sorry

end NUMINAMATH_CALUDE_integral_reciprocal_x_l207_20797


namespace NUMINAMATH_CALUDE_sum_of_squares_in_ratio_l207_20781

theorem sum_of_squares_in_ratio (a b c : ℚ) : 
  (a : ℚ) + b + c = 9 →
  b = 2 * a →
  c = 4 * a →
  a^2 + b^2 + c^2 = 1701 / 49 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_in_ratio_l207_20781


namespace NUMINAMATH_CALUDE_gcd_condition_equivalence_l207_20711

theorem gcd_condition_equivalence (m n : ℕ+) :
  (∀ (x y : ℕ+), x ∣ m → y ∣ n → Nat.gcd (x + y) (m * n) > 1) ↔ Nat.gcd m n > 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_condition_equivalence_l207_20711


namespace NUMINAMATH_CALUDE_complex_power_24_l207_20748

theorem complex_power_24 : (((1 - Complex.I) / Real.sqrt 2) ^ 24 : ℂ) = 1 := by sorry

end NUMINAMATH_CALUDE_complex_power_24_l207_20748


namespace NUMINAMATH_CALUDE_least_six_digit_multiple_l207_20766

theorem least_six_digit_multiple : ∃ n : ℕ, 
  (n ≥ 100000 ∧ n < 1000000) ∧ 
  (12 ∣ n) ∧ (15 ∣ n) ∧ (18 ∣ n) ∧ (23 ∣ n) ∧ (29 ∣ n) ∧
  (∀ m : ℕ, m ≥ 100000 ∧ m < n → ¬((12 ∣ m) ∧ (15 ∣ m) ∧ (18 ∣ m) ∧ (23 ∣ m) ∧ (29 ∣ m))) :=
by
  use 120060
  sorry

end NUMINAMATH_CALUDE_least_six_digit_multiple_l207_20766


namespace NUMINAMATH_CALUDE_correct_train_sequence_l207_20733

-- Define the steps as an enumeration
inductive TrainStep
  | BuyTicket
  | WaitInWaitingRoom
  | CheckTicketAtGate
  | BoardTrain

def correct_sequence : List TrainStep :=
  [TrainStep.BuyTicket, TrainStep.WaitInWaitingRoom, TrainStep.CheckTicketAtGate, TrainStep.BoardTrain]

-- Define a function to check if a given sequence is correct
def is_correct_sequence (sequence : List TrainStep) : Prop :=
  sequence = correct_sequence

-- Theorem stating that the given sequence is correct
theorem correct_train_sequence : 
  is_correct_sequence [TrainStep.BuyTicket, TrainStep.WaitInWaitingRoom, TrainStep.CheckTicketAtGate, TrainStep.BoardTrain] :=
by sorry


end NUMINAMATH_CALUDE_correct_train_sequence_l207_20733


namespace NUMINAMATH_CALUDE_min_value_of_expression_l207_20725

theorem min_value_of_expression (x y : ℝ) : (x^3*y - 1)^2 + (x + y)^2 ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l207_20725


namespace NUMINAMATH_CALUDE_no_numbers_divisible_by_all_l207_20792

theorem no_numbers_divisible_by_all : ∀ n : ℕ, 1 ≤ n ∧ n ≤ 1000 →
  ¬(2 ∣ n ∧ 3 ∣ n ∧ 4 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n ∧ 11 ∣ n) := by
  sorry

end NUMINAMATH_CALUDE_no_numbers_divisible_by_all_l207_20792


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l207_20777

theorem least_addition_for_divisibility (n m : ℕ) (h : n = 1056 ∧ m = 27) :
  ∃ x : ℕ, (n + x) % m = 0 ∧ ∀ y : ℕ, y < x → (n + y) % m ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l207_20777


namespace NUMINAMATH_CALUDE_christopher_stroll_l207_20716

theorem christopher_stroll (speed : ℝ) (time : ℝ) (distance : ℝ) : 
  speed = 4 → time = 1.25 → distance = speed * time → distance = 5 := by
  sorry

end NUMINAMATH_CALUDE_christopher_stroll_l207_20716


namespace NUMINAMATH_CALUDE_machine_value_after_two_years_l207_20769

-- Define the initial purchase price
def initialPrice : ℝ := 8000

-- Define the depreciation rate (20% = 0.20)
def depreciationRate : ℝ := 0.20

-- Define the time period in years
def timePeriod : ℕ := 2

-- Function to calculate the market value after a given number of years
def marketValue (years : ℕ) : ℝ :=
  initialPrice * (1 - depreciationRate) ^ years

-- Theorem statement
theorem machine_value_after_two_years :
  marketValue timePeriod = 5120 := by
  sorry


end NUMINAMATH_CALUDE_machine_value_after_two_years_l207_20769


namespace NUMINAMATH_CALUDE_bank_profit_maximization_l207_20759

/-- The bank's profit maximization problem -/
theorem bank_profit_maximization
  (k : ℝ) (h_k_pos : k > 0) :
  let deposit_amount (x : ℝ) := k * x
  let profit (x : ℝ) := 0.048 * deposit_amount x - x * deposit_amount x
  ∃ (x_max : ℝ), x_max ∈ Set.Ioo 0 0.048 ∧
    ∀ (x : ℝ), x ∈ Set.Ioo 0 0.048 → profit x ≤ profit x_max ∧
    x_max = 0.024 :=
by sorry

end NUMINAMATH_CALUDE_bank_profit_maximization_l207_20759


namespace NUMINAMATH_CALUDE_no_max_min_value_l207_20780

/-- The function f(x) = x³ - (3/2)x² + 1 has neither a maximum value nor a minimum value -/
theorem no_max_min_value (f : ℝ → ℝ) (h : ∀ x, f x = x^3 - (3/2)*x^2 + 1) :
  (¬ ∃ y, ∀ x, f x ≤ f y) ∧ (¬ ∃ y, ∀ x, f x ≥ f y) := by
  sorry

end NUMINAMATH_CALUDE_no_max_min_value_l207_20780


namespace NUMINAMATH_CALUDE_punch_mixture_theorem_l207_20739

/-- Given a 2-liter mixture that is 15% fruit juice, adding 0.125 liters of pure fruit juice
    results in a new mixture that is 20% fruit juice. -/
theorem punch_mixture_theorem :
  let initial_volume : ℝ := 2
  let initial_concentration : ℝ := 0.15
  let added_juice : ℝ := 0.125
  let target_concentration : ℝ := 0.20
  let final_volume : ℝ := initial_volume + added_juice
  let final_juice_amount : ℝ := initial_volume * initial_concentration + added_juice
  final_juice_amount / final_volume = target_concentration := by
sorry


end NUMINAMATH_CALUDE_punch_mixture_theorem_l207_20739


namespace NUMINAMATH_CALUDE_xy_plus_2y_value_l207_20783

theorem xy_plus_2y_value (x y : ℝ) (h : x * (x + y) = x^2 + 12) : x*y + 2*y = 12 := by
  sorry

end NUMINAMATH_CALUDE_xy_plus_2y_value_l207_20783


namespace NUMINAMATH_CALUDE_alice_bob_sum_l207_20772

theorem alice_bob_sum : ∀ (a b : ℕ),
  1 ≤ a ∧ a ≤ 50 ∧                     -- Alice's number is between 1 and 50
  1 ≤ b ∧ b ≤ 50 ∧                     -- Bob's number is between 1 and 50
  a ≠ b ∧                              -- Numbers are drawn without replacement
  a ≠ 1 ∧ a ≠ 50 ∧                     -- Alice can't tell who has the larger number
  b > a ∧                              -- Bob knows he has the larger number
  ∃ (d : ℕ), d > 1 ∧ d < b ∧ d ∣ b ∧   -- Bob's number is composite
  ∃ (k : ℕ), 50 * b + a = k * k →      -- 50 * Bob's number + Alice's number is a perfect square
  a + b = 29 := by
sorry

end NUMINAMATH_CALUDE_alice_bob_sum_l207_20772


namespace NUMINAMATH_CALUDE_angle_bisector_sum_geq_nine_times_inradius_l207_20795

/-- A triangle with an incircle -/
structure TriangleWithIncircle where
  /-- The radius of the incircle -/
  r : ℝ
  /-- The first angle bisector -/
  f_a : ℝ
  /-- The second angle bisector -/
  f_b : ℝ
  /-- The third angle bisector -/
  f_c : ℝ
  /-- Assumption that r is positive -/
  r_pos : r > 0
  /-- Assumption that angle bisectors are positive -/
  f_a_pos : f_a > 0
  f_b_pos : f_b > 0
  f_c_pos : f_c > 0

/-- The sum of angle bisectors is greater than or equal to 9 times the incircle radius -/
theorem angle_bisector_sum_geq_nine_times_inradius (t : TriangleWithIncircle) :
  t.f_a + t.f_b + t.f_c ≥ 9 * t.r :=
sorry

end NUMINAMATH_CALUDE_angle_bisector_sum_geq_nine_times_inradius_l207_20795


namespace NUMINAMATH_CALUDE_y_decreasing_order_l207_20737

-- Define the linear function
def f (x : ℝ) (b : ℝ) : ℝ := -2 * x + b

-- Define the theorem
theorem y_decreasing_order (b : ℝ) (y₁ y₂ y₃ : ℝ) 
  (h₁ : f (-2) b = y₁)
  (h₂ : f (-1) b = y₂)
  (h₃ : f 1 b = y₃) :
  y₁ > y₂ ∧ y₂ > y₃ := by
  sorry

end NUMINAMATH_CALUDE_y_decreasing_order_l207_20737


namespace NUMINAMATH_CALUDE_good_numbers_characterization_l207_20754

/-- A natural number is good if every natural divisor of n, when increased by 1, is a divisor of n+1 -/
def IsGood (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∣ n → (d + 1) ∣ (n + 1)

/-- Characterization of good numbers -/
theorem good_numbers_characterization (n : ℕ) :
  IsGood n ↔ n = 1 ∨ (Nat.Prime n ∧ n % 2 = 1) :=
sorry

end NUMINAMATH_CALUDE_good_numbers_characterization_l207_20754


namespace NUMINAMATH_CALUDE_certified_mail_delivery_l207_20714

/-- The total number of pieces of certified mail delivered by Johann and his friends -/
def total_mail (friends_mail : ℕ) (johann_mail : ℕ) : ℕ :=
  2 * friends_mail + johann_mail

/-- Theorem stating the total number of pieces of certified mail to be delivered -/
theorem certified_mail_delivery :
  let friends_mail := 41
  let johann_mail := 98
  total_mail friends_mail johann_mail = 180 := by
sorry

end NUMINAMATH_CALUDE_certified_mail_delivery_l207_20714


namespace NUMINAMATH_CALUDE_sphere_volume_ratio_l207_20758

theorem sphere_volume_ratio (r₁ r₂ : ℝ) (h : r₁ > 0 ∧ r₂ > 0) :
  (4 * Real.pi * r₁^2) / (4 * Real.pi * r₂^2) = 4 / 9 →
  ((4 / 3) * Real.pi * r₁^3) / ((4 / 3) * Real.pi * r₂^3) = 8 / 27 := by
sorry

end NUMINAMATH_CALUDE_sphere_volume_ratio_l207_20758


namespace NUMINAMATH_CALUDE_trig_problem_l207_20798

theorem trig_problem (α β : Real) 
  (h_acute_α : 0 < α ∧ α < Real.pi/2)
  (h_acute_β : 0 < β ∧ β < Real.pi/2)
  (h_sin_α : Real.sin α = 3/5)
  (h_tan_diff : Real.tan (α - β) = -1/3) :
  (Real.sin (α - β) = -Real.sqrt 10 / 10) ∧ 
  (Real.cos β = 9 * Real.sqrt 10 / 50) := by
sorry

end NUMINAMATH_CALUDE_trig_problem_l207_20798


namespace NUMINAMATH_CALUDE_exists_sequence_to_target_state_l207_20761

-- Define the state of the urn
structure UrnState :=
  (white : ℕ)
  (black : ℕ)

-- Define the replacement rules
inductive ReplacementRule
| Rule1 -- 3 black → 1 black + 2 white
| Rule2 -- 2 black + 1 white → 3 black
| Rule3 -- 1 black + 2 white → 2 white
| Rule4 -- 3 white → 2 white + 1 black

-- Define a function to apply a rule to an urn state
def applyRule (state : UrnState) (rule : ReplacementRule) : UrnState :=
  match rule with
  | ReplacementRule.Rule1 => 
      if state.black ≥ 3 then UrnState.mk (state.white + 2) (state.black - 2) else state
  | ReplacementRule.Rule2 => 
      if state.black ≥ 2 ∧ state.white ≥ 1 then UrnState.mk (state.white - 1) (state.black + 1) else state
  | ReplacementRule.Rule3 => 
      if state.black ≥ 1 ∧ state.white ≥ 2 then UrnState.mk state.white (state.black - 1) else state
  | ReplacementRule.Rule4 => 
      if state.white ≥ 3 then UrnState.mk (state.white - 1) (state.black + 1) else state

-- Define the initial state
def initialState : UrnState := UrnState.mk 50 50

-- Define the target state
def targetState : UrnState := UrnState.mk 2 0

-- Theorem to prove
theorem exists_sequence_to_target_state : 
  ∃ (sequence : List ReplacementRule), 
    (sequence.foldl applyRule initialState) = targetState :=
sorry

end NUMINAMATH_CALUDE_exists_sequence_to_target_state_l207_20761


namespace NUMINAMATH_CALUDE_marble_prob_diff_l207_20757

/-- The number of red marbles in the box -/
def red_marbles : ℕ := 1200

/-- The number of black marbles in the box -/
def black_marbles : ℕ := 800

/-- The total number of marbles in the box -/
def total_marbles : ℕ := red_marbles + black_marbles

/-- The probability of drawing two marbles of the same color -/
def prob_same_color : ℚ :=
  (Nat.choose red_marbles 2 + Nat.choose black_marbles 2) / Nat.choose total_marbles 2

/-- The probability of drawing two marbles of different colors -/
def prob_diff_color : ℚ :=
  (red_marbles * black_marbles) / Nat.choose total_marbles 2

/-- Theorem stating the absolute difference between the probabilities -/
theorem marble_prob_diff :
  |prob_same_color - prob_diff_color| = 7900 / 199900 := by
  sorry


end NUMINAMATH_CALUDE_marble_prob_diff_l207_20757


namespace NUMINAMATH_CALUDE_cake_comparison_l207_20727

theorem cake_comparison : (1 : ℚ) / 3 > (1 : ℚ) / 4 ∧ (1 : ℚ) / 3 > (1 : ℚ) / 5 := by
  sorry

end NUMINAMATH_CALUDE_cake_comparison_l207_20727


namespace NUMINAMATH_CALUDE_triangle_side_length_l207_20738

theorem triangle_side_length (a b : ℝ) (A B C : ℝ) :
  b = 2 →
  B = 30 * (π / 180) →
  C = 135 * (π / 180) →
  A + B + C = π →
  a / Real.sin A = b / Real.sin B →
  a = Real.sqrt 6 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l207_20738


namespace NUMINAMATH_CALUDE_max_value_ratio_l207_20702

theorem max_value_ratio (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + y + z)^2 / (x^2 + y^2 + z^2) ≤ 3 ∧
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ (a + b + c)^2 / (a^2 + b^2 + c^2) = 3 :=
by sorry

end NUMINAMATH_CALUDE_max_value_ratio_l207_20702


namespace NUMINAMATH_CALUDE_average_pencils_is_111_75_l207_20790

def anna_pencils : ℕ := 50

def harry_pencils : ℕ := 2 * anna_pencils - 19

def lucy_pencils : ℕ := 3 * anna_pencils - 13

def david_pencils : ℕ := 4 * anna_pencils - 21

def total_pencils : ℕ := anna_pencils + harry_pencils + lucy_pencils + david_pencils

def average_pencils : ℚ := total_pencils / 4

theorem average_pencils_is_111_75 : average_pencils = 111.75 := by
  sorry

end NUMINAMATH_CALUDE_average_pencils_is_111_75_l207_20790

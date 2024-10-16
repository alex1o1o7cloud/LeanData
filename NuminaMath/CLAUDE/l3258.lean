import Mathlib

namespace NUMINAMATH_CALUDE_newscast_advertising_time_l3258_325861

theorem newscast_advertising_time (total_time national_news international_news sports weather : ℕ)
  (h_total : total_time = 30)
  (h_national : national_news = 12)
  (h_international : international_news = 5)
  (h_sports : sports = 5)
  (h_weather : weather = 2) :
  total_time - (national_news + international_news + sports + weather) = 6 := by
  sorry

end NUMINAMATH_CALUDE_newscast_advertising_time_l3258_325861


namespace NUMINAMATH_CALUDE_symmetry_sum_l3258_325866

/-- Two points are symmetric with respect to the y-axis if their x-coordinates are opposite
    and their y-coordinates are equal -/
def symmetric_wrt_y_axis (p q : ℝ × ℝ) : Prop :=
  p.1 = -q.1 ∧ p.2 = q.2

theorem symmetry_sum (a b : ℝ) :
  symmetric_wrt_y_axis (a, -3) (4, b) → a + b = -7 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_sum_l3258_325866


namespace NUMINAMATH_CALUDE_hagrid_divisible_by_three_l3258_325875

def HAGRID (H A G R I D : ℕ) : ℕ := 100000*H + 10000*A + 1000*G + 100*R + 10*I + D

theorem hagrid_divisible_by_three 
  (H A G R I D : ℕ) 
  (h_distinct : H ≠ A ∧ H ≠ G ∧ H ≠ R ∧ H ≠ I ∧ H ≠ D ∧ 
                A ≠ G ∧ A ≠ R ∧ A ≠ I ∧ A ≠ D ∧ 
                G ≠ R ∧ G ≠ I ∧ G ≠ D ∧ 
                R ≠ I ∧ R ≠ D ∧ 
                I ≠ D)
  (h_range : H < 10 ∧ A < 10 ∧ G < 10 ∧ R < 10 ∧ I < 10 ∧ D < 10) : 
  3 ∣ (HAGRID H A G R I D * H * A * G * R * I * D) :=
sorry

end NUMINAMATH_CALUDE_hagrid_divisible_by_three_l3258_325875


namespace NUMINAMATH_CALUDE_magazine_budget_cut_percentage_l3258_325805

def original_budget : ℚ := 940
def new_budget : ℚ := 752

theorem magazine_budget_cut_percentage : 
  (original_budget - new_budget) / original_budget * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_magazine_budget_cut_percentage_l3258_325805


namespace NUMINAMATH_CALUDE_infinitely_many_even_floor_alpha_n_squared_l3258_325834

theorem infinitely_many_even_floor_alpha_n_squared (α : ℝ) (hα : α > 0) :
  ∃ S : Set ℕ, Set.Infinite S ∧ ∀ n ∈ S, Even ⌊α * n^2⌋ := by sorry

end NUMINAMATH_CALUDE_infinitely_many_even_floor_alpha_n_squared_l3258_325834


namespace NUMINAMATH_CALUDE_expression_value_at_three_l3258_325829

theorem expression_value_at_three :
  let x : ℝ := 3
  (x^3 - 2*x^2 - 21*x + 36) / (x - 6) = 6 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_at_three_l3258_325829


namespace NUMINAMATH_CALUDE_galaxy_first_chinese_supercomputer_l3258_325891

/-- Represents a supercomputer -/
structure Supercomputer where
  name : String
  country : String
  performance : ℕ  -- calculations per second
  year_introduced : ℕ
  month_introduced : ℕ

/-- The Galaxy supercomputer -/
def galaxy : Supercomputer :=
  { name := "Galaxy"
  , country := "China"
  , performance := 100000000  -- 100 million
  , year_introduced := 1983
  , month_introduced := 12 }

/-- Predicate to check if a supercomputer meets the criteria -/
def meets_criteria (sc : Supercomputer) : Prop :=
  sc.country = "China" ∧
  sc.performance ≥ 100000000 ∧
  sc.year_introduced = 1983 ∧
  sc.month_introduced = 12

/-- Theorem stating that Galaxy was China's first supercomputer meeting the criteria -/
theorem galaxy_first_chinese_supercomputer :
  meets_criteria galaxy ∧
  ∀ (sc : Supercomputer), meets_criteria sc → sc.name = galaxy.name :=
by sorry


end NUMINAMATH_CALUDE_galaxy_first_chinese_supercomputer_l3258_325891


namespace NUMINAMATH_CALUDE_smallest_prime_after_six_nonprimes_l3258_325826

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

def consecutive_nonprimes (start : ℕ) : Prop :=
  ∀ k : ℕ, k < 6 → ¬(is_prime (start + k))

theorem smallest_prime_after_six_nonprimes :
  ∀ p : ℕ, is_prime p →
    (∃ start : ℕ, consecutive_nonprimes start ∧ start + 6 < p) →
    p ≥ 127 :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_after_six_nonprimes_l3258_325826


namespace NUMINAMATH_CALUDE_least_positive_linear_combination_l3258_325821

theorem least_positive_linear_combination (x y z : ℤ) : 
  ∃ (a b c : ℤ), 24*a + 20*b + 12*c = 4 ∧ 
  (∀ (x y z : ℤ), 24*x + 20*y + 12*z = 0 ∨ |24*x + 20*y + 12*z| ≥ 4) :=
by sorry

end NUMINAMATH_CALUDE_least_positive_linear_combination_l3258_325821


namespace NUMINAMATH_CALUDE_fraction_division_l3258_325839

theorem fraction_division (x : ℝ) (hx : x ≠ 0) :
  (3 / 8) / (5 * x / 12) = 9 / (10 * x) := by
  sorry

end NUMINAMATH_CALUDE_fraction_division_l3258_325839


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3258_325857

theorem min_value_reciprocal_sum (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : Real.log (a + b) = 0) :
  (∀ x y : ℝ, x > 0 → y > 0 → Real.log (x + y) = 0 → 1/x + 4/y ≥ 1/a + 4/b) ∧ 
  1/a + 4/b = 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3258_325857


namespace NUMINAMATH_CALUDE_inequality_cube_l3258_325889

theorem inequality_cube (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > 0) :
  (a - c)^3 > (b - c)^3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_cube_l3258_325889


namespace NUMINAMATH_CALUDE_proposition_falsity_l3258_325820

theorem proposition_falsity (P : ℕ → Prop) 
  (h1 : ∀ k : ℕ, k > 0 → (P k → P (k + 1)))
  (h2 : ¬ P 5) : 
  ¬ P 4 := by
sorry

end NUMINAMATH_CALUDE_proposition_falsity_l3258_325820


namespace NUMINAMATH_CALUDE_flash_interval_l3258_325853

/-- Proves that the time between each flash is 6 seconds, given that a light flashes 450 times in ¾ of an hour. -/
theorem flash_interval (flashes : ℕ) (time : ℚ) (h1 : flashes = 450) (h2 : time = 3/4) :
  (time * 3600) / flashes = 6 := by
  sorry

end NUMINAMATH_CALUDE_flash_interval_l3258_325853


namespace NUMINAMATH_CALUDE_correct_propositions_l3258_325819

-- Define the propositions
def vertical_angles_equal : Prop := True
def complementary_angles_of_equal_angles_equal : Prop := True
def corresponding_angles_equal : Prop := False
def parallel_transitivity : Prop := True
def parallel_sides_equal_or_supplementary : Prop := True
def inverse_proportion_inequality : Prop := False
def inequality_squared : Prop := False
def irrational_numbers_not_representable : Prop := False

-- Theorem statement
theorem correct_propositions :
  vertical_angles_equal ∧
  complementary_angles_of_equal_angles_equal ∧
  parallel_transitivity ∧
  parallel_sides_equal_or_supplementary ∧
  ¬corresponding_angles_equal ∧
  ¬inverse_proportion_inequality ∧
  ¬inequality_squared ∧
  ¬irrational_numbers_not_representable :=
by sorry

end NUMINAMATH_CALUDE_correct_propositions_l3258_325819


namespace NUMINAMATH_CALUDE_largest_integer_less_than_80_remainder_3_mod_5_l3258_325864

theorem largest_integer_less_than_80_remainder_3_mod_5 : ∃ n : ℕ, 
  (n < 80 ∧ n % 5 = 3 ∧ ∀ m : ℕ, m < 80 ∧ m % 5 = 3 → m ≤ n) ∧ n = 78 := by
  sorry

end NUMINAMATH_CALUDE_largest_integer_less_than_80_remainder_3_mod_5_l3258_325864


namespace NUMINAMATH_CALUDE_paths_on_specific_grid_l3258_325825

/-- The number of paths on a rectangular grid from (0,0) to (m,n) moving only right or up -/
def grid_paths (m n : ℕ) : ℕ := Nat.choose (m + n) n

/-- The specific grid dimensions -/
def grid_width : ℕ := 7
def grid_height : ℕ := 3

theorem paths_on_specific_grid :
  grid_paths grid_width grid_height = 120 := by
  sorry

end NUMINAMATH_CALUDE_paths_on_specific_grid_l3258_325825


namespace NUMINAMATH_CALUDE_other_root_of_complex_quadratic_l3258_325808

theorem other_root_of_complex_quadratic (z : ℂ) :
  z^2 = -75 + 40*I ∧ (5 + 7*I)^2 = -75 + 40*I →
  (-5 - 7*I)^2 = -75 + 40*I :=
by sorry

end NUMINAMATH_CALUDE_other_root_of_complex_quadratic_l3258_325808


namespace NUMINAMATH_CALUDE_all_a_equal_one_l3258_325803

def cyclic_index (i : ℕ) : ℕ :=
  match i % 100 with
  | 0 => 100
  | n => n

theorem all_a_equal_one (a : ℕ → ℝ) 
  (h_ineq : ∀ i, a (cyclic_index i) - 4 * a (cyclic_index (i + 1)) + 3 * a (cyclic_index (i + 2)) ≥ 0)
  (h_a1 : a 1 = 1) :
  ∀ i, a i = 1 := by
sorry

end NUMINAMATH_CALUDE_all_a_equal_one_l3258_325803


namespace NUMINAMATH_CALUDE_alphabet_composition_l3258_325843

theorem alphabet_composition (total : ℕ) (both : ℕ) (line_only : ℕ) (dot_only : ℕ) : 
  total = 40 →
  both = 8 →
  line_only = 24 →
  total = both + line_only + dot_only →
  dot_only = 8 := by
sorry

end NUMINAMATH_CALUDE_alphabet_composition_l3258_325843


namespace NUMINAMATH_CALUDE_oscar_swag_bag_scarf_cost_l3258_325882

/-- The cost of each designer scarf in the Oscar swag bag -/
def scarf_cost (total_value earring_cost iphone_cost num_earrings num_scarves : ℕ) : ℕ :=
  (total_value - (num_earrings * earring_cost + iphone_cost)) / num_scarves

/-- Theorem: The cost of each designer scarf in the Oscar swag bag is $1,500 -/
theorem oscar_swag_bag_scarf_cost :
  scarf_cost 20000 6000 2000 2 4 = 1500 := by
  sorry

end NUMINAMATH_CALUDE_oscar_swag_bag_scarf_cost_l3258_325882


namespace NUMINAMATH_CALUDE_sequence_properties_l3258_325877

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

def S (b : ℕ → ℝ) : ℕ → ℝ
  | 0 => 0
  | n + 1 => S b n + b (n + 1)

def T (c : ℕ → ℝ) : ℕ → ℝ
  | 0 => 0
  | n + 1 => T c n + c (n + 1)

theorem sequence_properties (a b c : ℕ → ℝ) :
  arithmetic_sequence a
  ∧ a 5 = 14
  ∧ a 7 = 20
  ∧ b 1 = 2/3
  ∧ (∀ n : ℕ, n ≥ 2 → 3 * S b n = S b (n-1) + 2)
  ∧ (∀ n : ℕ, c n = a n * b n)
  →
  (∀ n : ℕ, a n = 3*n - 1)
  ∧ (∀ n : ℕ, b n = 2 * (1/3)^n)
  ∧ (∀ n : ℕ, n ≥ 1 → T c n < 7/2)
  ∧ (∀ m : ℝ, (∀ n : ℕ, n ≥ 1 → T c n < m) → m ≥ 7/2) :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l3258_325877


namespace NUMINAMATH_CALUDE_nancy_bathroom_flooring_l3258_325823

/-- Represents the dimensions of a rectangular area -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.length * r.width

/-- The central area of Nancy's bathroom -/
def central_area : Rectangle := { length := 10, width := 10 }

/-- The hallway area of Nancy's bathroom -/
def hallway : Rectangle := { length := 6, width := 4 }

/-- The total area of hardwood flooring in Nancy's bathroom -/
def total_flooring_area : ℝ := area central_area + area hallway

theorem nancy_bathroom_flooring :
  total_flooring_area = 124 := by sorry

end NUMINAMATH_CALUDE_nancy_bathroom_flooring_l3258_325823


namespace NUMINAMATH_CALUDE_digit_sum_power_equality_l3258_325844

-- Define the sum of digits function
def S (m : ℕ) : ℕ := sorry

-- Define the set of solutions
def solution_set : Set (ℕ × ℕ) :=
  {p | ∃ (b : ℕ), p = (1, b + 1)} ∪ {(3, 2), (9, 1)}

-- State the theorem
theorem digit_sum_power_equality :
  ∀ a b : ℕ, a > 0 → b > 0 →
  (S (a^(b+1)) = a^b ↔ (a, b) ∈ solution_set) := by sorry

end NUMINAMATH_CALUDE_digit_sum_power_equality_l3258_325844


namespace NUMINAMATH_CALUDE_max_value_of_t_l3258_325878

theorem max_value_of_t (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  min x (y / (x^2 + y^2)) ≤ 1 / Real.sqrt 2 ∧
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ min x (y / (x^2 + y^2)) = 1 / Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_t_l3258_325878


namespace NUMINAMATH_CALUDE_no_real_roots_iff_m_gt_one_l3258_325856

-- Define the quadratic equation
def quadratic (x m : ℝ) : ℝ := x^2 - 2*x + m

-- Theorem statement
theorem no_real_roots_iff_m_gt_one (m : ℝ) :
  (∀ x : ℝ, quadratic x m ≠ 0) ↔ m > 1 :=
by sorry

end NUMINAMATH_CALUDE_no_real_roots_iff_m_gt_one_l3258_325856


namespace NUMINAMATH_CALUDE_water_added_to_container_l3258_325830

/-- Proves that the amount of water added to a container with a capacity of 40 liters,
    initially 40% full, to make it 3/4 full, is 14 liters. -/
theorem water_added_to_container (capacity : ℝ) (initial_percentage : ℝ) (final_fraction : ℝ) :
  capacity = 40 →
  initial_percentage = 0.4 →
  final_fraction = 3/4 →
  (final_fraction * capacity) - (initial_percentage * capacity) = 14 := by
  sorry

end NUMINAMATH_CALUDE_water_added_to_container_l3258_325830


namespace NUMINAMATH_CALUDE_slow_dancers_count_l3258_325896

theorem slow_dancers_count (total_kids : ℕ) (non_slow_dancers : ℕ) : 
  total_kids = 140 → 
  non_slow_dancers = 10 → 
  (total_kids / 4 : ℕ) - non_slow_dancers = 25 := by
  sorry

end NUMINAMATH_CALUDE_slow_dancers_count_l3258_325896


namespace NUMINAMATH_CALUDE_perimeter_plus_area_sum_l3258_325840

/-- A parallelogram with integer coordinates -/
structure IntegerParallelogram where
  v1 : ℤ × ℤ
  v2 : ℤ × ℤ
  v3 : ℤ × ℤ
  v4 : ℤ × ℤ

/-- The specific parallelogram from the problem -/
def specificParallelogram : IntegerParallelogram :=
  { v1 := (2, 3)
    v2 := (5, 7)
    v3 := (11, 7)
    v4 := (8, 3) }

/-- Calculate the perimeter of the parallelogram -/
def perimeter (p : IntegerParallelogram) : ℝ :=
  sorry

/-- Calculate the area of the parallelogram -/
def area (p : IntegerParallelogram) : ℝ :=
  sorry

/-- The main theorem to prove -/
theorem perimeter_plus_area_sum (p : IntegerParallelogram) :
  p = specificParallelogram → perimeter p + area p = 46 :=
sorry

end NUMINAMATH_CALUDE_perimeter_plus_area_sum_l3258_325840


namespace NUMINAMATH_CALUDE_complex_real_part_l3258_325865

theorem complex_real_part (z : ℂ) (h : (z^2 + z).im = 0) : z.re = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_complex_real_part_l3258_325865


namespace NUMINAMATH_CALUDE_kaleb_candy_count_l3258_325813

/-- The number of candies Kaleb can buy with his arcade tickets -/
def candies_kaleb_can_buy (whack_a_mole_tickets : ℕ) (skee_ball_tickets : ℕ) (candy_cost : ℕ) : ℕ :=
  (whack_a_mole_tickets + skee_ball_tickets) / candy_cost

/-- Proof that Kaleb can buy 3 candies with his arcade tickets -/
theorem kaleb_candy_count : candies_kaleb_can_buy 8 7 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_kaleb_candy_count_l3258_325813


namespace NUMINAMATH_CALUDE_min_distance_to_origin_l3258_325802

-- Define the circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 2*y + 1 = 0
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 6*y + 9 = 0

-- Define the condition |PA| = |PB|
def equal_chords (x y : ℝ) : Prop := 
  ∃ (xa ya xb yb : ℝ), C₁ xa ya ∧ C₂ xb yb ∧ 
  (x - xa)^2 + (y - ya)^2 = (x - xb)^2 + (y - yb)^2

-- Theorem statement
theorem min_distance_to_origin : 
  ∀ (x y : ℝ), equal_chords x y → 
  ∃ (x' y' : ℝ), equal_chords x' y' ∧ 
  ∀ (x'' y'' : ℝ), equal_chords x'' y'' → 
  (x'^2 + y'^2 : ℝ) ≤ x''^2 + y''^2 ∧
  (x'^2 + y'^2 : ℝ) = (4/5)^2 := by
  sorry

end NUMINAMATH_CALUDE_min_distance_to_origin_l3258_325802


namespace NUMINAMATH_CALUDE_dot_product_of_vectors_l3258_325867

theorem dot_product_of_vectors (a b : ℝ × ℝ) 
  (h1 : a + b = (1, -3))
  (h2 : a - b = (3, 7)) :
  a • b = -12 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_of_vectors_l3258_325867


namespace NUMINAMATH_CALUDE_exists_marked_points_with_distance_l3258_325852

/-- Represents a marked point on the segment -/
structure MarkedPoint where
  position : ℚ
  deriving Repr

/-- The process of marking points on a segment of length 3^n -/
def markPoints (n : ℕ) : List MarkedPoint :=
  sorry

/-- Theorem stating the existence of two marked points with distance k -/
theorem exists_marked_points_with_distance (n : ℕ) (k : ℕ) 
  (h : 1 ≤ k ∧ k ≤ 3^n) : 
  ∃ (p q : MarkedPoint), p ∈ markPoints n ∧ q ∈ markPoints n ∧ 
    |p.position - q.position| = k :=
  sorry

end NUMINAMATH_CALUDE_exists_marked_points_with_distance_l3258_325852


namespace NUMINAMATH_CALUDE_last_four_digits_of_special_N_l3258_325854

/-- Given a positive integer N where N and N^2 end with the same five non-zero digits in base 10,
    prove that the last four digits of N are 2999. -/
theorem last_four_digits_of_special_N (N : ℕ) : 
  (N > 0) →
  (∃ a b c d e : ℕ, a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 ∧ 
    N % 100000 = 10000 * a + 1000 * b + 100 * c + 10 * d + e ∧
    (N^2) % 100000 = 10000 * a + 1000 * b + 100 * c + 10 * d + e) →
  N % 10000 = 2999 :=
by sorry

end NUMINAMATH_CALUDE_last_four_digits_of_special_N_l3258_325854


namespace NUMINAMATH_CALUDE_emilys_flowers_l3258_325824

theorem emilys_flowers (flower_cost : ℕ) (total_spent : ℕ) : 
  flower_cost = 3 →
  total_spent = 12 →
  ∃ (roses daisies : ℕ), 
    roses = daisies ∧ 
    roses + daisies = total_spent / flower_cost :=
by
  sorry

end NUMINAMATH_CALUDE_emilys_flowers_l3258_325824


namespace NUMINAMATH_CALUDE_grains_in_cup_is_480_l3258_325876

/-- Represents the number of grains of rice in one cup -/
def grains_in_cup : ℕ :=
  let half_cup_tablespoons : ℕ := 8
  let teaspoons_per_tablespoon : ℕ := 3
  let grains_per_teaspoon : ℕ := 10
  2 * (half_cup_tablespoons * teaspoons_per_tablespoon * grains_per_teaspoon)

/-- Theorem stating that there are 480 grains of rice in one cup -/
theorem grains_in_cup_is_480 : grains_in_cup = 480 := by
  sorry

end NUMINAMATH_CALUDE_grains_in_cup_is_480_l3258_325876


namespace NUMINAMATH_CALUDE_special_function_property_l3258_325815

-- Define a monotonic function f from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the monotonicity of f
def Monotonic (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y

-- Define the special property of f
def SpecialProperty (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∀ x₁ x₂, f (x * x₁ + x * x₂) = f x + f x₁ + f x₂

theorem special_function_property 
  (h_monotonic : Monotonic f) 
  (h_exists : ∃ x, SpecialProperty f x) :
  (f 1 + f 0 = 0) ∧ (∃ x, SpecialProperty f x ∧ x = 1) :=
by sorry

end NUMINAMATH_CALUDE_special_function_property_l3258_325815


namespace NUMINAMATH_CALUDE_student_average_greater_than_true_average_l3258_325801

theorem student_average_greater_than_true_average 
  (a b c : ℝ) (h : a < b ∧ b < c) : (a + b + c) / 2 > (a + b + c) / 3 := by
  sorry

end NUMINAMATH_CALUDE_student_average_greater_than_true_average_l3258_325801


namespace NUMINAMATH_CALUDE_sum_of_products_l3258_325849

theorem sum_of_products (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x^2 + x*y + y^2 = 27)
  (eq2 : y^2 + y*z + z^2 = 25)
  (eq3 : z^2 + x*z + x^2 = 52) :
  x*y + y*z + x*z = 30 := by
sorry

end NUMINAMATH_CALUDE_sum_of_products_l3258_325849


namespace NUMINAMATH_CALUDE_cockroach_search_l3258_325832

/-- The cockroach's search problem -/
theorem cockroach_search (D : ℝ) (h : D > 0) :
  ∃ (path : ℕ → ℝ × ℝ),
    (∀ n, dist (path n) (path (n+1)) ≤ 1) ∧
    (∀ n, dist (path (n+1)) (D, 0) < dist (path n) (D, 0) ∨
          dist (path (n+1)) (D, 0) = dist (path n) (D, 0)) ∧
    (∃ n, path n = (D, 0)) ∧
    (∃ n, path n = (D, 0) ∧ n ≤ ⌊(3/2 * D + 7)⌋) :=
sorry


end NUMINAMATH_CALUDE_cockroach_search_l3258_325832


namespace NUMINAMATH_CALUDE_penguin_colony_growth_l3258_325863

/-- Represents the penguin colony growth over three years -/
structure PenguinColony where
  initial_size : ℕ
  first_year_growth : ℕ → ℕ
  second_year_growth : ℕ → ℕ
  third_year_gain : ℕ
  current_size : ℕ
  fish_per_penguin : ℚ
  initial_fish_caught : ℕ

/-- Theorem stating the number of penguins gained in the third year -/
theorem penguin_colony_growth (colony : PenguinColony) : colony.third_year_gain = 129 :=
  by
  have h1 : colony.initial_size = 158 := by sorry
  have h2 : colony.first_year_growth colony.initial_size = 2 * colony.initial_size := by sorry
  have h3 : colony.second_year_growth (colony.first_year_growth colony.initial_size) = 
            3 * (colony.first_year_growth colony.initial_size) := by sorry
  have h4 : colony.current_size = 1077 := by sorry
  have h5 : colony.fish_per_penguin = 3/2 := by sorry
  have h6 : colony.initial_fish_caught = 237 := by sorry
  have h7 : colony.initial_size * colony.fish_per_penguin = colony.initial_fish_caught := by sorry
  sorry

end NUMINAMATH_CALUDE_penguin_colony_growth_l3258_325863


namespace NUMINAMATH_CALUDE_sum_of_smallest_and_largest_prime_l3258_325885

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def primes_between_1_and_50 : Set ℕ := {n : ℕ | 1 < n ∧ n ≤ 50 ∧ is_prime n}

theorem sum_of_smallest_and_largest_prime :
  ∃ (p q : ℕ), p ∈ primes_between_1_and_50 ∧ q ∈ primes_between_1_and_50 ∧
  (∀ r ∈ primes_between_1_and_50, p ≤ r) ∧
  (∀ r ∈ primes_between_1_and_50, r ≤ q) ∧
  p + q = 49 :=
sorry

end NUMINAMATH_CALUDE_sum_of_smallest_and_largest_prime_l3258_325885


namespace NUMINAMATH_CALUDE_line_intersects_circle_l3258_325884

/-- Given a point outside a circle, prove that a specific line intersects the circle -/
theorem line_intersects_circle (x₀ y₀ a : ℝ) (h₁ : a > 0) (h₂ : x₀^2 + y₀^2 > a^2) :
  ∃ (x y : ℝ), x^2 + y^2 = a^2 ∧ x₀*x + y₀*y = a^2 := by
  sorry

end NUMINAMATH_CALUDE_line_intersects_circle_l3258_325884


namespace NUMINAMATH_CALUDE_circle_properties_l3258_325890

/-- A circle passing through two points with its center on a line -/
def circle_equation (x y : ℝ) : Prop :=
  (x - 1/2)^2 + (y - 1)^2 = 5/4

theorem circle_properties :
  (circle_equation 1 0) ∧
  (circle_equation 0 2) ∧
  (∃ (a : ℝ), circle_equation a (2*a)) :=
sorry

end NUMINAMATH_CALUDE_circle_properties_l3258_325890


namespace NUMINAMATH_CALUDE_sum_of_five_integers_l3258_325888

theorem sum_of_five_integers (C y M A : ℕ) : 
  C > 0 → y > 0 → M > 0 → A > 0 →
  C ≠ y → C ≠ M → C ≠ A → y ≠ M → y ≠ A → M ≠ A →
  C + y + M + M + A = 11 →
  M = 1 := by
sorry

end NUMINAMATH_CALUDE_sum_of_five_integers_l3258_325888


namespace NUMINAMATH_CALUDE_city_rentals_per_mile_rate_l3258_325887

/-- Represents the daily rental rate in dollars -/
def daily_rate_sunshine : ℝ := 17.99

/-- Represents the per-mile rate for Sunshine Car Rentals in dollars -/
def per_mile_rate_sunshine : ℝ := 0.18

/-- Represents the daily rental rate for City Rentals in dollars -/
def daily_rate_city : ℝ := 18.95

/-- Represents the number of miles driven -/
def miles_driven : ℝ := 48

/-- Represents the unknown per-mile rate for City Rentals -/
def per_mile_rate_city : ℝ := 0.16

theorem city_rentals_per_mile_rate :
  daily_rate_sunshine + per_mile_rate_sunshine * miles_driven =
  daily_rate_city + per_mile_rate_city * miles_driven :=
by sorry

#check city_rentals_per_mile_rate

end NUMINAMATH_CALUDE_city_rentals_per_mile_rate_l3258_325887


namespace NUMINAMATH_CALUDE_ellipse_min_value_l3258_325835

/-- For an ellipse with semi-major axis a, semi-minor axis b, and eccentricity e,
    prove that the minimum value of (a² + 1) / b is 4√3 / 3 when e = 1/2. -/
theorem ellipse_min_value (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : (a^2 - b^2) / a^2 = 1/4) :
  (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1) →
  (a^2 + 1) / b ≥ 4 * Real.sqrt 3 / 3 :=
sorry

end NUMINAMATH_CALUDE_ellipse_min_value_l3258_325835


namespace NUMINAMATH_CALUDE_faye_candy_problem_l3258_325804

theorem faye_candy_problem (initial : ℕ) (received : ℕ) (final : ℕ) (eaten : ℕ) : 
  initial = 47 → received = 40 → final = 62 → 
  initial - eaten + received = final → 
  eaten = 25 := by sorry

end NUMINAMATH_CALUDE_faye_candy_problem_l3258_325804


namespace NUMINAMATH_CALUDE_min_moves_to_guarantee_coin_find_l3258_325862

/-- Represents the game state with thimbles and a hidden coin. -/
structure ThimbleGame where
  numThimbles : Nat
  numFlipPerMove : Nat

/-- Represents a strategy for playing the game. -/
structure Strategy where
  numMoves : Nat

/-- Determines if a strategy is guaranteed to find the coin. -/
def isGuaranteedStrategy (game : ThimbleGame) (strategy : Strategy) : Prop :=
  ∀ (coinPosition : Nat), coinPosition < game.numThimbles → 
    ∃ (move : Nat), move < strategy.numMoves ∧ 
      (∃ (flippedThimble : Nat), flippedThimble < game.numFlipPerMove ∧ 
        (coinPosition + move) % game.numThimbles = flippedThimble)

/-- The main theorem stating the minimum number of moves required. -/
theorem min_moves_to_guarantee_coin_find (game : ThimbleGame) 
    (h1 : game.numThimbles = 100) (h2 : game.numFlipPerMove = 4) : 
    ∃ (strategy : Strategy), 
      isGuaranteedStrategy game strategy ∧ 
      strategy.numMoves = 33 ∧
      (∀ (otherStrategy : Strategy), 
        isGuaranteedStrategy game otherStrategy → 
        otherStrategy.numMoves ≥ 33) :=
  sorry

end NUMINAMATH_CALUDE_min_moves_to_guarantee_coin_find_l3258_325862


namespace NUMINAMATH_CALUDE_triangle_ABC_properties_l3258_325836

noncomputable def triangle_ABC (a b c A B C : ℝ) : Prop :=
  2 * c = Real.sqrt 3 * a + 2 * b * Real.cos A ∧
  c = 1 ∧
  1/2 * a * c * Real.sin B = Real.sqrt 3 / 2

theorem triangle_ABC_properties (a b c A B C : ℝ) 
  (h : triangle_ABC a b c A B C) : 
  B = π / 6 ∧ b = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_ABC_properties_l3258_325836


namespace NUMINAMATH_CALUDE_angle_sum_at_point_l3258_325873

theorem angle_sum_at_point (x y : ℝ) : 
  3 * x + 6 * x + (x + y) + 4 * y = 360 → x = 0 ∧ y = 72 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_at_point_l3258_325873


namespace NUMINAMATH_CALUDE_foreign_language_speakers_l3258_325883

theorem foreign_language_speakers (total_students : ℕ) (male_students : ℕ) (female_students : ℕ) :
  male_students = female_students →
  (3 : ℚ) / 5 * male_students + (2 : ℚ) / 3 * female_students = (19 : ℚ) / 30 * (male_students + female_students) :=
by sorry

end NUMINAMATH_CALUDE_foreign_language_speakers_l3258_325883


namespace NUMINAMATH_CALUDE_chloe_trivia_score_l3258_325833

/-- Chloe's trivia game score calculation -/
theorem chloe_trivia_score (first_round : ℕ) (last_round_loss : ℕ) (total_points : ℕ) 
  (h1 : first_round = 40)
  (h2 : last_round_loss = 4)
  (h3 : total_points = 86) :
  ∃ second_round : ℕ, second_round = 50 ∧ 
    first_round + second_round - last_round_loss = total_points :=
by sorry

end NUMINAMATH_CALUDE_chloe_trivia_score_l3258_325833


namespace NUMINAMATH_CALUDE_oblique_triangular_prism_surface_area_l3258_325847

/-- The total surface area of an oblique triangular prism -/
theorem oblique_triangular_prism_surface_area
  (a l : ℝ)
  (h_a_pos : 0 < a)
  (h_l_pos : 0 < l) :
  let lateral_surface_area := 3 * a * l
  let base_area := a^2 * Real.sqrt 3 / 2
  let total_surface_area := lateral_surface_area + 2 * base_area
  total_surface_area = 3 * a * l + a^2 * Real.sqrt 3 :=
by sorry


end NUMINAMATH_CALUDE_oblique_triangular_prism_surface_area_l3258_325847


namespace NUMINAMATH_CALUDE_ribbon_solution_l3258_325869

def ribbon_problem (total : ℝ) : Prop :=
  let remaining_after_first := total / 2
  let remaining_after_second := remaining_after_first * 2 / 3
  let remaining_after_third := remaining_after_second / 2
  remaining_after_third = 250

theorem ribbon_solution :
  ribbon_problem 1500 := by sorry

end NUMINAMATH_CALUDE_ribbon_solution_l3258_325869


namespace NUMINAMATH_CALUDE_inscribed_square_area_l3258_325871

theorem inscribed_square_area (x y : ℝ) (h1 : x = 18) (h2 : y = 30) :
  let s := Real.sqrt ((x * y) / (x + y))
  s ^ 2 = 540 := by sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l3258_325871


namespace NUMINAMATH_CALUDE_delta_sum_bound_l3258_325895

/-- The greatest odd divisor of a positive integer -/
def greatest_odd_divisor (n : ℕ+) : ℕ+ :=
  sorry

/-- The sum of δ(n)/n from 1 to x -/
def delta_sum (x : ℕ+) : ℚ :=
  sorry

/-- Theorem: For any positive integer x, |∑(n=1 to x) [δ(n)/n] - (2/3)x| < 1 -/
theorem delta_sum_bound (x : ℕ+) :
  |delta_sum x - (2/3 : ℚ) * x.val| < 1 :=
sorry

end NUMINAMATH_CALUDE_delta_sum_bound_l3258_325895


namespace NUMINAMATH_CALUDE_prism_with_21_edges_has_9_faces_l3258_325822

/-- The number of faces in a prism with a given number of edges -/
def prism_faces (edges : ℕ) : ℕ :=
  2 + edges / 3

/-- Theorem: A prism with 21 edges has 9 faces -/
theorem prism_with_21_edges_has_9_faces :
  prism_faces 21 = 9 := by
  sorry

end NUMINAMATH_CALUDE_prism_with_21_edges_has_9_faces_l3258_325822


namespace NUMINAMATH_CALUDE_range_of_f_l3258_325811

-- Define the function
def f (x : ℝ) : ℝ := -x^2 - 6*x - 5

-- State the theorem
theorem range_of_f :
  Set.range f = Set.Iic 4 :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l3258_325811


namespace NUMINAMATH_CALUDE_dividingChordLength_l3258_325807

/-- A hexagon inscribed in a circle with alternating side lengths -/
structure AlternatingHexagon where
  /-- The length of three consecutive sides -/
  shortSide : ℝ
  /-- The length of the other three consecutive sides -/
  longSide : ℝ
  /-- The short sides are indeed shorter than the long sides -/
  shortLessThanLong : shortSide < longSide

/-- The chord dividing the hexagon into two trapezoids -/
def dividingChord (h : AlternatingHexagon) : ℝ := sorry

theorem dividingChordLength (h : AlternatingHexagon) 
  (h_short : h.shortSide = 4)
  (h_long : h.longSide = 6) :
  dividingChord h = 480 / 49 := by
  sorry

end NUMINAMATH_CALUDE_dividingChordLength_l3258_325807


namespace NUMINAMATH_CALUDE_sallys_initial_cards_l3258_325831

/-- Proves that Sally's initial number of cards was 27 given the problem conditions -/
theorem sallys_initial_cards : 
  ∀ x : ℕ, 
  (x + 41 + 20 = 88) → 
  x = 27 := by
  sorry

end NUMINAMATH_CALUDE_sallys_initial_cards_l3258_325831


namespace NUMINAMATH_CALUDE_no_base_for_131_square_l3258_325810

theorem no_base_for_131_square (b : ℕ) : b > 3 → ¬∃ (n : ℕ), b^2 + 3*b + 1 = n^2 := by
  sorry

end NUMINAMATH_CALUDE_no_base_for_131_square_l3258_325810


namespace NUMINAMATH_CALUDE_min_sum_squares_on_parabola_l3258_325855

/-- The parabola equation y² = 4x -/
def parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- The line passing through P(4, 0) and (x, y) -/
def line_through_P (x y : ℝ) : Prop := ∃ k : ℝ, y = k * (x - 4)

/-- Theorem: The minimum value of y₁² + y₂² is 32 for points on the parabola
    intersected by a line through P(4, 0) -/
theorem min_sum_squares_on_parabola (x₁ y₁ x₂ y₂ : ℝ) :
  parabola x₁ y₁ →
  parabola x₂ y₂ →
  line_through_P x₁ y₁ →
  line_through_P x₂ y₂ →
  y₁^2 + y₂^2 ≥ 32 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squares_on_parabola_l3258_325855


namespace NUMINAMATH_CALUDE_markov_equation_solution_l3258_325851

/-- Markov equation -/
def markov_equation (x y z : ℕ+) : Prop :=
  x^2 + y^2 + z^2 = 3*x*y*z

/-- Definition of coprime positive integers -/
def coprime (a b : ℕ+) : Prop :=
  Nat.gcd a.val b.val = 1

/-- Definition of sum of squares of two coprime integers -/
def sum_of_coprime_squares (a : ℕ+) : Prop :=
  ∃ (p q : ℕ+), coprime p q ∧ a = p^2 + q^2

/-- Main theorem -/
theorem markov_equation_solution :
  ∀ (a b c : ℕ+), markov_equation a b c →
    (coprime a b ∧ coprime b c ∧ coprime a c) ∧
    (a ≠ 1 → sum_of_coprime_squares a) :=
sorry

end NUMINAMATH_CALUDE_markov_equation_solution_l3258_325851


namespace NUMINAMATH_CALUDE_solve_system_l3258_325838

theorem solve_system (x y : ℝ) 
  (eq1 : 3 * x - 2 * y = 18) 
  (eq2 : x + y = 20) : 
  y = 8.4 := by sorry

end NUMINAMATH_CALUDE_solve_system_l3258_325838


namespace NUMINAMATH_CALUDE_hyperbolas_same_asymptotes_l3258_325845

/-- Given two hyperbolas with equations (x^2/16) - (y^2/25) = 1 and (y^2/49) - (x^2/M) = 1,
    if they have the same asymptotes, then M = 784/25 -/
theorem hyperbolas_same_asymptotes (M : ℝ) :
  (∀ x y : ℝ, x^2/16 - y^2/25 = 1 ↔ y^2/49 - x^2/M = 1) →
  (∀ x y : ℝ, y = (5/4) * x ↔ y = (7/Real.sqrt M) * x) →
  M = 784/25 := by
  sorry

end NUMINAMATH_CALUDE_hyperbolas_same_asymptotes_l3258_325845


namespace NUMINAMATH_CALUDE_minimum_value_theorem_l3258_325812

-- Define the line equation
def line_equation (m n x y : ℝ) : Prop := m * x + n * y + 2 = 0

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := (x + 3)^2 + (y + 1)^2 = 1

-- Define the chord length condition
def chord_length_condition (m n : ℝ) : Prop := 
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    line_equation m n x₁ y₁ ∧ 
    line_equation m n x₂ y₂ ∧ 
    circle_equation x₁ y₁ ∧ 
    circle_equation x₂ y₂ ∧ 
    (x₂ - x₁)^2 + (y₂ - y₁)^2 = 4

theorem minimum_value_theorem (m n : ℝ) 
  (hm : m > 0) (hn : n > 0) 
  (h_chord : chord_length_condition m n) : 
  (∀ m' n' : ℝ, m' > 0 → n' > 0 → chord_length_condition m' n' → 1/m' + 3/n' ≥ 1/m + 3/n) → 
  1/m + 3/n = 6 := by
sorry

end NUMINAMATH_CALUDE_minimum_value_theorem_l3258_325812


namespace NUMINAMATH_CALUDE_opposite_of_negative_three_l3258_325870

theorem opposite_of_negative_three : 
  ∃ y : ℤ, ((-3 : ℤ) + y = 0) ∧ y = 3 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_three_l3258_325870


namespace NUMINAMATH_CALUDE_double_add_five_minus_half_l3258_325837

theorem double_add_five_minus_half (x : ℝ) (h : x = 4) : 2 * x + 5 - x / 2 = 11 := by
  sorry

end NUMINAMATH_CALUDE_double_add_five_minus_half_l3258_325837


namespace NUMINAMATH_CALUDE_q_completes_in_four_days_l3258_325898

-- Define the work as a positive real number
variable (W : ℝ) (hW : W > 0)

-- Define the time taken by p and q together
def combined_time : ℝ := 20

-- Define the time p worked alone
def p_alone_time : ℝ := 4

-- Define the total time of work
def total_time : ℝ := 10

-- Define q's time to complete the work alone
def q_alone_time : ℝ := 4

-- Theorem statement
theorem q_completes_in_four_days :
  (W / combined_time + W / q_alone_time) * (total_time - p_alone_time) = W * (1 - p_alone_time / combined_time) :=
sorry

end NUMINAMATH_CALUDE_q_completes_in_four_days_l3258_325898


namespace NUMINAMATH_CALUDE_derivative_sin_minus_cos_at_pi_l3258_325868

open Real

theorem derivative_sin_minus_cos_at_pi :
  let f : ℝ → ℝ := fun x ↦ sin x - cos x
  deriv f π = -1 := by
  sorry

end NUMINAMATH_CALUDE_derivative_sin_minus_cos_at_pi_l3258_325868


namespace NUMINAMATH_CALUDE_complex_fraction_evaluation_l3258_325817

theorem complex_fraction_evaluation : 
  1 / (3 + 1 / (3 + 1 / (3 - 1 / (3 + 1 / (2 * (3 + 2 / 5)))))) = 968/3191 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_evaluation_l3258_325817


namespace NUMINAMATH_CALUDE_parallel_lines_angle_measure_l3258_325827

/-- Given two parallel lines intersected by a transversal, 
    if one angle is (x+40)° and the other is (3x-40)°, 
    then the first angle measures 85°. -/
theorem parallel_lines_angle_measure :
  ∀ (x : ℝ) (α β : ℝ),
  α = x + 40 →
  β = 3*x - 40 →
  α + β = 180 →
  α = 85 := by
sorry

end NUMINAMATH_CALUDE_parallel_lines_angle_measure_l3258_325827


namespace NUMINAMATH_CALUDE_min_sum_of_coefficients_l3258_325880

theorem min_sum_of_coefficients (b c : ℕ+) 
  (h1 : ∃ (x y : ℝ), x ≠ y ∧ 2 * x^2 + b * x + c = 0 ∧ 2 * y^2 + b * y + c = 0)
  (h2 : ∃ (x y : ℝ), x - y = 30 ∧ 2 * x^2 + b * x + c = 0 ∧ 2 * y^2 + b * y + c = 0) :
  (∀ (b' c' : ℕ+), 
    (∃ (x y : ℝ), x ≠ y ∧ 2 * x^2 + b' * x + c' = 0 ∧ 2 * y^2 + b' * y + c' = 0) →
    (∃ (x y : ℝ), x - y = 30 ∧ 2 * x^2 + b' * x + c' = 0 ∧ 2 * y^2 + b' * y + c' = 0) →
    b'.val + c'.val ≥ b.val + c.val) →
  b.val + c.val = 126 := by
sorry

end NUMINAMATH_CALUDE_min_sum_of_coefficients_l3258_325880


namespace NUMINAMATH_CALUDE_quadratic_roots_to_coefficients_l3258_325842

theorem quadratic_roots_to_coefficients (b c : ℝ) : 
  (∀ x : ℝ, x^2 + b*x + c = 0 ↔ x = 1 ∨ x = -2) → 
  b = 1 ∧ c = -2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_to_coefficients_l3258_325842


namespace NUMINAMATH_CALUDE_tire_usage_l3258_325850

/-- Proves that each tire is used for 32,000 miles given the conditions of the problem -/
theorem tire_usage (total_miles : ℕ) (total_tires : ℕ) (tires_in_use : ℕ) 
  (h1 : total_miles = 40000)
  (h2 : total_tires = 5)
  (h3 : tires_in_use = 4)
  (h4 : tires_in_use < total_tires) :
  (total_miles * tires_in_use) / total_tires = 32000 := by
  sorry

end NUMINAMATH_CALUDE_tire_usage_l3258_325850


namespace NUMINAMATH_CALUDE_square_difference_l3258_325841

theorem square_difference (a b : ℝ) (h1 : a + b = 20) (h2 : a - b = 4) : a^2 - b^2 = 80 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l3258_325841


namespace NUMINAMATH_CALUDE_melanie_total_dimes_l3258_325800

def initial_dimes : ℕ := 7
def dimes_from_dad : ℕ := 8
def dimes_from_mom : ℕ := 4

theorem melanie_total_dimes : 
  initial_dimes + dimes_from_dad + dimes_from_mom = 19 := by
  sorry

end NUMINAMATH_CALUDE_melanie_total_dimes_l3258_325800


namespace NUMINAMATH_CALUDE_dinner_pizzas_count_l3258_325894

/-- The number of pizzas served during lunch -/
def lunch_pizzas : ℕ := 9

/-- The total number of pizzas served today -/
def total_pizzas : ℕ := 15

/-- The number of pizzas served during dinner -/
def dinner_pizzas : ℕ := total_pizzas - lunch_pizzas

theorem dinner_pizzas_count : dinner_pizzas = 6 := by
  sorry

end NUMINAMATH_CALUDE_dinner_pizzas_count_l3258_325894


namespace NUMINAMATH_CALUDE_factorial_sum_remainder_20_l3258_325818

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem factorial_sum_remainder_20 :
  (factorial 1 + factorial 2 + factorial 3 + factorial 4) % 20 = 13 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_remainder_20_l3258_325818


namespace NUMINAMATH_CALUDE_colin_skipping_speed_l3258_325858

theorem colin_skipping_speed (bruce_speed tony_speed brandon_speed colin_speed : ℝ) :
  bruce_speed = 1 →
  tony_speed = 2 * bruce_speed →
  brandon_speed = (1/3) * tony_speed →
  colin_speed = 6 * brandon_speed →
  colin_speed = 4 := by
sorry

end NUMINAMATH_CALUDE_colin_skipping_speed_l3258_325858


namespace NUMINAMATH_CALUDE_pizza_toppings_combinations_l3258_325860

theorem pizza_toppings_combinations (n : ℕ) (h : n = 8) : 
  n + (n.choose 2) + (n.choose 3) = 92 := by
  sorry

end NUMINAMATH_CALUDE_pizza_toppings_combinations_l3258_325860


namespace NUMINAMATH_CALUDE_stating_max_areas_formula_l3258_325874

/-- Represents a circular disk divided by radii and secant lines -/
structure DividedDisk where
  n : ℕ
  radii_count : ℕ := 3 * n
  secant_lines : ℕ := 2
  h_positive : n > 0

/-- 
Calculates the maximum number of non-overlapping areas in a divided disk 
-/
def max_areas (disk : DividedDisk) : ℕ := 4 * disk.n + 1

/-- 
Theorem stating that the maximum number of non-overlapping areas 
in a divided disk is 4n + 1 
-/
theorem max_areas_formula (disk : DividedDisk) : 
  max_areas disk = 4 * disk.n + 1 := by sorry

end NUMINAMATH_CALUDE_stating_max_areas_formula_l3258_325874


namespace NUMINAMATH_CALUDE_intersecting_quadratic_properties_l3258_325881

/-- A quadratic function that intersects both coordinate axes at three points -/
structure IntersectingQuadratic where
  b : ℝ
  intersects_axes : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ -x₁^2 - 2*x₁ + b = 0 ∧ -x₂^2 - 2*x₂ + b = 0
  intersects_y : b ≠ 0

/-- The range of possible values for b -/
def valid_b_range (q : IntersectingQuadratic) : Prop :=
  q.b > -1 ∧ q.b ≠ 0

/-- The equation of the circle passing through the three intersection points -/
def circle_equation (q : IntersectingQuadratic) (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x + (1 - q.b)*y - q.b = 0

theorem intersecting_quadratic_properties (q : IntersectingQuadratic) :
  valid_b_range q ∧
  ∀ (x y : ℝ), circle_equation q x y ↔ 
    (x = 0 ∧ y = q.b) ∨ 
    (y = 0 ∧ -x^2 - 2*x + q.b = 0) :=
sorry

end NUMINAMATH_CALUDE_intersecting_quadratic_properties_l3258_325881


namespace NUMINAMATH_CALUDE_perfect_square_characterization_l3258_325846

theorem perfect_square_characterization (A : ℕ+) :
  (∃ k : ℕ, A = k^2) ↔
  (∀ n : ℕ+, ∃ k : ℕ+, k ≤ n ∧ n ∣ ((A + k)^2 - A)) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_characterization_l3258_325846


namespace NUMINAMATH_CALUDE_symmetry_condition_implies_symmetric_about_one_l3258_325806

/-- A function f: ℝ → ℝ is symmetric about x = 1 if f(1 + x) = f(1 - x) for all x ∈ ℝ -/
def SymmetricAboutOne (f : ℝ → ℝ) : Prop :=
  ∀ x, f (1 + x) = f (1 - x)

/-- Main theorem: If f(x) - f(2 - x) = 0 for all x, then f is symmetric about x = 1 -/
theorem symmetry_condition_implies_symmetric_about_one (f : ℝ → ℝ) 
    (h : ∀ x, f x - f (2 - x) = 0) : SymmetricAboutOne f := by
  sorry

#check symmetry_condition_implies_symmetric_about_one

end NUMINAMATH_CALUDE_symmetry_condition_implies_symmetric_about_one_l3258_325806


namespace NUMINAMATH_CALUDE_conclusion_1_conclusion_2_conclusion_3_l3258_325859

-- Define the quadratic function
def f (b : ℝ) (x : ℝ) : ℝ := x^2 - 2*b*x + 3

-- Theorem 1
theorem conclusion_1 (b : ℝ) :
  (∀ m : ℝ, m*(m - 2*b) ≥ 1 - 2*b) → b = 1 := by sorry

-- Theorem 2
theorem conclusion_2 (b : ℝ) :
  ∃ h k : ℝ, (∀ x : ℝ, f b x ≥ f b h) ∧ k = f b h ∧ k = -h^2 + 3 := by sorry

-- Theorem 3
theorem conclusion_3 (b : ℝ) :
  (∀ x : ℝ, -1 ≤ x → x ≤ 5 → f b x ≤ f b (-1)) →
  (∃ m₁ m₂ p : ℝ, m₁ ≠ m₂ ∧ f b m₁ = p ∧ f b m₂ = p) →
  ∃ m₁ m₂ : ℝ, m₁ + m₂ > 4 := by sorry

end NUMINAMATH_CALUDE_conclusion_1_conclusion_2_conclusion_3_l3258_325859


namespace NUMINAMATH_CALUDE_store_distance_l3258_325816

def walking_speed : ℝ := 2
def running_speed : ℝ := 10
def average_time_minutes : ℝ := 56

theorem store_distance : 
  ∃ (distance : ℝ),
    (distance / walking_speed + distance / running_speed + distance / running_speed) / 3 = average_time_minutes / 60 ∧
    distance = 4 := by
  sorry

end NUMINAMATH_CALUDE_store_distance_l3258_325816


namespace NUMINAMATH_CALUDE_joana_shopping_problem_l3258_325892

theorem joana_shopping_problem :
  ∃! (b c : ℕ), 15 * b + 17 * c = 143 :=
by sorry

end NUMINAMATH_CALUDE_joana_shopping_problem_l3258_325892


namespace NUMINAMATH_CALUDE_apartment_cost_splitting_l3258_325814

/-- The number of people splitting the cost of the new apartment -/
def number_of_people : ℕ := 3

/-- John's two brothers -/
def johns_brothers : ℕ := 2

theorem apartment_cost_splitting :
  number_of_people = johns_brothers + 1 := by
  sorry

end NUMINAMATH_CALUDE_apartment_cost_splitting_l3258_325814


namespace NUMINAMATH_CALUDE_green_face_box_dimensions_l3258_325872

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Checks if the given dimensions satisfy the green face condition -/
def satisfiesGreenFaceCondition (dim : BoxDimensions) : Prop :=
  3 * ((dim.a - 2) * (dim.b - 2) * (dim.c - 2)) = dim.a * dim.b * dim.c

/-- List of valid box dimensions -/
def validDimensions : List BoxDimensions := [
  ⟨7, 30, 4⟩, ⟨8, 18, 4⟩, ⟨9, 14, 4⟩, ⟨10, 12, 4⟩,
  ⟨5, 27, 5⟩, ⟨6, 12, 5⟩, ⟨7, 9, 5⟩, ⟨6, 8, 6⟩
]

theorem green_face_box_dimensions :
  ∀ dim : BoxDimensions,
    satisfiesGreenFaceCondition dim ↔ dim ∈ validDimensions :=
by sorry

end NUMINAMATH_CALUDE_green_face_box_dimensions_l3258_325872


namespace NUMINAMATH_CALUDE_tan_greater_than_cubic_l3258_325848

theorem tan_greater_than_cubic (x : ℝ) (h1 : 0 < x) (h2 : x < π / 2) :
  Real.tan x > x + (1 / 3) * x^3 := by
  sorry

end NUMINAMATH_CALUDE_tan_greater_than_cubic_l3258_325848


namespace NUMINAMATH_CALUDE_trigonometric_identity_l3258_325886

theorem trigonometric_identity : 
  Real.sin (20 * π / 180) * Real.cos (10 * π / 180) - 
  Real.cos (160 * π / 180) * Real.sin (10 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l3258_325886


namespace NUMINAMATH_CALUDE_three_digit_congruence_count_l3258_325899

theorem three_digit_congruence_count : 
  (Finset.filter (fun x => 100 ≤ x ∧ x < 1000 ∧ (2895 * x + 547) % 17 = 1613 % 17) 
    (Finset.range 1000)).card = 53 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_congruence_count_l3258_325899


namespace NUMINAMATH_CALUDE_n_equals_fourteen_l3258_325809

def first_seven_multiples_of_seven : List ℕ := [7, 14, 21, 28, 35, 42, 49]

def a : ℚ := (first_seven_multiples_of_seven.sum : ℚ) / 7

def first_three_multiples (n : ℕ) : List ℕ := [n, 2*n, 3*n]

def b (n : ℕ) : ℕ := (first_three_multiples n).nthLe 1 sorry

theorem n_equals_fourteen (n : ℕ) (h : a^2 - (b n : ℚ)^2 = 0) : n = 14 := by
  sorry

end NUMINAMATH_CALUDE_n_equals_fourteen_l3258_325809


namespace NUMINAMATH_CALUDE_not_polite_and_power_of_two_polite_or_power_of_two_l3258_325897

/-- A number is polite if it can be written as the sum of consecutive integers from m to n, where m < n. -/
def IsPolite (N : ℕ) : Prop :=
  ∃ m n : ℕ, m < n ∧ N = (n * (n + 1) - m * (m - 1)) / 2

/-- A number is a power of two if it can be written as 2^ℓ for some non-negative integer ℓ. -/
def IsPowerOfTwo (N : ℕ) : Prop :=
  ∃ ℓ : ℕ, N = 2^ℓ

/-- No number is both polite and a power of two. -/
theorem not_polite_and_power_of_two (N : ℕ) : ¬(IsPolite N ∧ IsPowerOfTwo N) := by
  sorry

/-- Every positive integer is either polite or a power of two. -/
theorem polite_or_power_of_two (N : ℕ) : N > 0 → IsPolite N ∨ IsPowerOfTwo N := by
  sorry

end NUMINAMATH_CALUDE_not_polite_and_power_of_two_polite_or_power_of_two_l3258_325897


namespace NUMINAMATH_CALUDE_function_not_in_third_quadrant_l3258_325893

theorem function_not_in_third_quadrant
  (a b : ℝ) (ha : 0 < a) (ha' : a < 1) (hb : b > -1) :
  ¬∃ (x y : ℝ), x < 0 ∧ y < 0 ∧ y = a^x + b :=
by sorry

end NUMINAMATH_CALUDE_function_not_in_third_quadrant_l3258_325893


namespace NUMINAMATH_CALUDE_intersection_equality_l3258_325879

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p | p.1^2 + 2*p.1 + p.2^2 ≤ 0}
def N (a : ℝ) : Set (ℝ × ℝ) := {p | p.2 ≥ p.1 + a}

-- State the theorem
theorem intersection_equality (a : ℝ) : M ∩ N a = M ↔ a ≤ 1 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_equality_l3258_325879


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3258_325828

theorem quadratic_equation_solution (x : ℝ) 
  (eq : 2 * x^2 = 9 * x - 4) 
  (neq : x ≠ 4) : 
  2 * x = 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3258_325828

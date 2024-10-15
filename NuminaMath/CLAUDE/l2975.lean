import Mathlib

namespace NUMINAMATH_CALUDE_simple_interest_months_l2975_297546

/-- Simple interest calculation -/
theorem simple_interest_months (principal : ℝ) (rate : ℝ) (interest : ℝ) : 
  principal = 10000 →
  rate = 0.08 →
  interest = 800 →
  (interest / (principal * rate)) * 12 = 12 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_months_l2975_297546


namespace NUMINAMATH_CALUDE_bridget_weight_l2975_297577

/-- Given that Martha weighs 2 pounds and Bridget is 37 pounds heavier than Martha,
    prove that Bridget weighs 39 pounds. -/
theorem bridget_weight (martha_weight : ℕ) (weight_difference : ℕ) :
  martha_weight = 2 →
  weight_difference = 37 →
  martha_weight + weight_difference = 39 :=
by sorry

end NUMINAMATH_CALUDE_bridget_weight_l2975_297577


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2975_297522

/-- An isosceles triangle with side lengths a and b satisfying a certain equation has perimeter 10 -/
theorem isosceles_triangle_perimeter (a b : ℝ) : 
  a > 0 → b > 0 → -- Positive side lengths
  (∃ c : ℝ, c > 0 ∧ a + a + c = b + b) → -- Isosceles triangle condition
  2 * Real.sqrt (3 * a - 6) + 3 * Real.sqrt (2 - a) = b - 4 → -- Given equation
  a + a + b = 10 := by -- Perimeter is 10
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2975_297522


namespace NUMINAMATH_CALUDE_intersection_A_B_l2975_297535

-- Define the sets A and B
def A : Set ℝ := {x | |x - 1| > 2}
def B : Set ℝ := {x | x * (x - 5) < 0}

-- State the theorem
theorem intersection_A_B :
  A ∩ B = {x : ℝ | 3 < x ∧ x < 5} :=
by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l2975_297535


namespace NUMINAMATH_CALUDE_computer_contracts_probability_l2975_297505

theorem computer_contracts_probability 
  (p_hardware : ℝ) 
  (p_not_software : ℝ) 
  (p_at_least_one : ℝ) 
  (h1 : p_hardware = 4/5) 
  (h2 : p_not_software = 3/5) 
  (h3 : p_at_least_one = 5/6) : 
  p_hardware + (1 - p_not_software) - p_at_least_one = 11/30 :=
by sorry

end NUMINAMATH_CALUDE_computer_contracts_probability_l2975_297505


namespace NUMINAMATH_CALUDE_count_satisfying_integers_l2975_297513

def satisfies_conditions (n : ℤ) : Prop :=
  (n + 5) * (n - 5) * (n - 15) < 0 ∧ n > 7

theorem count_satisfying_integers :
  ∃ (S : Finset ℤ), (∀ n ∈ S, satisfies_conditions n) ∧ 
                    (∀ n, satisfies_conditions n → n ∈ S) ∧
                    S.card = 7 := by
  sorry

end NUMINAMATH_CALUDE_count_satisfying_integers_l2975_297513


namespace NUMINAMATH_CALUDE_parabola_focus_l2975_297578

/-- The parabola equation -/
def parabola_equation (x y : ℝ) : Prop := x^2 = -4*y

/-- The focus of the parabola -/
def focus : ℝ × ℝ := (0, -1)

/-- Theorem: The focus of the parabola x^2 = -4y is (0, -1) -/
theorem parabola_focus :
  ∀ x y : ℝ, parabola_equation x y → (x, y) = focus :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_l2975_297578


namespace NUMINAMATH_CALUDE_three_divisors_iff_prime_square_l2975_297598

/-- A positive integer has exactly three distinct divisors if and only if it is the square of a prime number. -/
theorem three_divisors_iff_prime_square (n : ℕ) :
  (∃! (s : Finset ℕ), s.card = 3 ∧ ∀ d ∈ s, d ∣ n) ↔ ∃ p : ℕ, Nat.Prime p ∧ n = p^2 :=
sorry

end NUMINAMATH_CALUDE_three_divisors_iff_prime_square_l2975_297598


namespace NUMINAMATH_CALUDE_max_value_theorem_l2975_297582

theorem max_value_theorem (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a^2 + b^2 + c^2 ≠ 0) :
  (|a + 2*b + 3*c| / Real.sqrt (a^2 + b^2 + c^2)) ≤ Real.sqrt 2 ∧
  ∃ (a' b' c' : ℝ), a' + b' + c' = 0 ∧ a'^2 + b'^2 + c'^2 ≠ 0 ∧
    |a' + 2*b' + 3*c'| / Real.sqrt (a'^2 + b'^2 + c'^2) = Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l2975_297582


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l2975_297592

/-- Given an ellipse defined by the equation 4x² + y² = 16, 
    its major axis has length 8. -/
theorem ellipse_major_axis_length :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
  (∀ x y : ℝ, 4 * x^2 + y^2 = 16 ↔ x^2 / a^2 + y^2 / b^2 = 1) ∧
  2 * max a b = 8 :=
sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l2975_297592


namespace NUMINAMATH_CALUDE_parabola_vertex_on_x_axis_l2975_297596

/-- A parabola with equation y = x^2 - 8x + m has its vertex on the x-axis if and only if m = 16 -/
theorem parabola_vertex_on_x_axis (m : ℝ) :
  (∃ x : ℝ, x^2 - 8*x + m = 0 ∧ 
   ∀ t : ℝ, t^2 - 8*t + m ≥ 0) ↔ 
  m = 16 :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_on_x_axis_l2975_297596


namespace NUMINAMATH_CALUDE_running_reduction_is_five_l2975_297529

/-- Carly's running distances over four weeks -/
def running_distances : Fin 4 → ℚ
  | 0 => 2                        -- Week 1: 2 miles
  | 1 => 2 * 2 + 3                -- Week 2: twice as long as week 1 plus 3 extra miles
  | 2 => (2 * 2 + 3) * (9/7)      -- Week 3: 9/7 as much as week 2
  | 3 => 4                        -- Week 4: 4 miles due to injury

/-- The reduction in Carly's running distance when she was injured -/
def running_reduction : ℚ :=
  running_distances 2 - running_distances 3

theorem running_reduction_is_five :
  running_reduction = 5 := by sorry

end NUMINAMATH_CALUDE_running_reduction_is_five_l2975_297529


namespace NUMINAMATH_CALUDE_quadrilateral_area_is_18_l2975_297502

/-- Represents a 2D point with x and y coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the area of a quadrilateral given its four vertices -/
def quadrilateralArea (p1 p2 p3 p4 : Point) : ℝ :=
  0.5 * abs ((p1.x * p2.y + p2.x * p3.y + p3.x * p4.y + p4.x * p1.y) -
             (p1.y * p2.x + p2.y * p3.x + p3.y * p4.x + p4.y * p1.x))

/-- Theorem: The area of the quadrilateral with vertices at (0,0), (4,0), (6,3), and (4,6) is 18 -/
theorem quadrilateral_area_is_18 :
  let p1 : Point := ⟨0, 0⟩
  let p2 : Point := ⟨4, 0⟩
  let p3 : Point := ⟨6, 3⟩
  let p4 : Point := ⟨4, 6⟩
  quadrilateralArea p1 p2 p3 p4 = 18 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_is_18_l2975_297502


namespace NUMINAMATH_CALUDE_typing_speed_ratio_l2975_297500

theorem typing_speed_ratio (T M : ℝ) (h1 : T > 0) (h2 : M > 0) 
  (h3 : T + M = 12) (h4 : T + 1.25 * M = 14) : M / T = 2 := by
  sorry

end NUMINAMATH_CALUDE_typing_speed_ratio_l2975_297500


namespace NUMINAMATH_CALUDE_fourth_score_calculation_l2975_297523

theorem fourth_score_calculation (s1 s2 s3 s4 : ℕ) (h1 : s1 = 65) (h2 : s2 = 67) (h3 : s3 = 76)
  (h_average : (s1 + s2 + s3 + s4) / 4 = 75) : s4 = 92 := by
  sorry

end NUMINAMATH_CALUDE_fourth_score_calculation_l2975_297523


namespace NUMINAMATH_CALUDE_number_of_children_l2975_297576

theorem number_of_children (crayons_per_child : ℕ) (total_crayons : ℕ) (h1 : crayons_per_child = 3) (h2 : total_crayons = 18) :
  total_crayons / crayons_per_child = 6 := by
  sorry

end NUMINAMATH_CALUDE_number_of_children_l2975_297576


namespace NUMINAMATH_CALUDE_prime_sum_theorem_l2975_297568

theorem prime_sum_theorem (p q : ℕ) : 
  Nat.Prime p → 
  Nat.Prime q → 
  Nat.Prime (7 * p + q) → 
  Nat.Prime (2 * q + 11) → 
  p^q + q^p = 17 := by
sorry

end NUMINAMATH_CALUDE_prime_sum_theorem_l2975_297568


namespace NUMINAMATH_CALUDE_base_85_modulo_17_l2975_297531

theorem base_85_modulo_17 (b : ℕ) : 
  0 ≤ b ∧ b ≤ 16 → (352936524 : ℕ) ≡ b [MOD 17] ↔ b = 4 := by
  sorry

end NUMINAMATH_CALUDE_base_85_modulo_17_l2975_297531


namespace NUMINAMATH_CALUDE_a_gt_b_sufficient_not_necessary_for_a_sq_gt_b_sq_l2975_297550

theorem a_gt_b_sufficient_not_necessary_for_a_sq_gt_b_sq :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  (∀ (x y : ℝ), x > 0 ∧ y > 0 → (x > y → x^2 > y^2)) ∧
  (a^2 > b^2 ∧ a ≤ b) := by
  sorry

end NUMINAMATH_CALUDE_a_gt_b_sufficient_not_necessary_for_a_sq_gt_b_sq_l2975_297550


namespace NUMINAMATH_CALUDE_lara_cookies_count_l2975_297597

/-- Calculates the total number of cookies baked by Lara --/
def total_cookies (
  num_trays : ℕ
  ) (
  large_rows_per_tray : ℕ
  ) (
  medium_rows_per_tray : ℕ
  ) (
  small_rows_per_tray : ℕ
  ) (
  large_cookies_per_row : ℕ
  ) (
  medium_cookies_per_row : ℕ
  ) (
  small_cookies_per_row : ℕ
  ) (
  extra_large_cookies : ℕ
  ) : ℕ :=
  (large_rows_per_tray * large_cookies_per_row * num_trays + extra_large_cookies) +
  (medium_rows_per_tray * medium_cookies_per_row * num_trays) +
  (small_rows_per_tray * small_cookies_per_row * num_trays)

theorem lara_cookies_count :
  total_cookies 4 5 4 6 6 7 8 6 = 430 := by
  sorry

end NUMINAMATH_CALUDE_lara_cookies_count_l2975_297597


namespace NUMINAMATH_CALUDE_sphere_surface_area_from_cube_l2975_297545

theorem sphere_surface_area_from_cube (cube_edge_length : ℝ) (sphere_radius : ℝ) : 
  cube_edge_length = 2 →
  sphere_radius^2 = 3 →
  4 * Real.pi * sphere_radius^2 = 12 * Real.pi :=
by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_from_cube_l2975_297545


namespace NUMINAMATH_CALUDE_integer_solution_l2975_297537

theorem integer_solution (x : ℤ) : x + 8 > 9 ∧ -3*x > -15 → x = 2 ∨ x = 3 ∨ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_integer_solution_l2975_297537


namespace NUMINAMATH_CALUDE_exists_sixth_root_of_3_30_sixth_root_of_3_30_correct_l2975_297584

theorem exists_sixth_root_of_3_30 : ∃ n : ℕ, n^6 = 3^30 :=
by
  -- The proof would go here
  sorry

def sixth_root_of_3_30 : ℕ :=
  -- The definition of the actual value would go here
  -- We're not providing the implementation as per the instructions
  sorry

-- This theorem ensures that our defined value actually satisfies the property
theorem sixth_root_of_3_30_correct : (sixth_root_of_3_30)^6 = 3^30 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_exists_sixth_root_of_3_30_sixth_root_of_3_30_correct_l2975_297584


namespace NUMINAMATH_CALUDE_special_pair_characterization_l2975_297533

/-- A pair of integers is special if it is of the form (n, n-1) or (n-1, n) for some positive integer n. -/
def IsSpecialPair (p : ℤ × ℤ) : Prop :=
  ∃ n : ℤ, n > 0 ∧ (p = (n, n - 1) ∨ p = (n - 1, n))

/-- The sum of two pairs -/
def PairSum (p q : ℤ × ℤ) : ℤ × ℤ :=
  (p.1 + q.1, p.2 + q.2)

/-- A pair can be expressed as a sum of special pairs -/
def CanExpressAsSumOfSpecialPairs (p : ℤ × ℤ) : Prop :=
  ∃ (k : ℕ) (specialPairs : Fin k → ℤ × ℤ),
    k ≥ 2 ∧
    (∀ i, IsSpecialPair (specialPairs i)) ∧
    (∀ i j, i ≠ j → specialPairs i ≠ specialPairs j) ∧
    p = Finset.sum Finset.univ (λ i => specialPairs i)

theorem special_pair_characterization (n m : ℤ) 
    (h_positive : n > 0 ∧ m > 0)
    (h_not_special : ¬IsSpecialPair (n, m)) :
    CanExpressAsSumOfSpecialPairs (n, m) ↔ n + m ≥ (n - m)^2 := by
  sorry

end NUMINAMATH_CALUDE_special_pair_characterization_l2975_297533


namespace NUMINAMATH_CALUDE_wire_around_square_field_l2975_297512

theorem wire_around_square_field (area : ℝ) (wire_length : ℝ) : 
  area = 69696 → wire_length = 15840 → 
  (wire_length / (4 * Real.sqrt area)) = 15 := by
  sorry

end NUMINAMATH_CALUDE_wire_around_square_field_l2975_297512


namespace NUMINAMATH_CALUDE_two_sector_area_l2975_297586

theorem two_sector_area (r : ℝ) (h : r = 15) : 
  2 * (45 / 360) * (π * r^2) = 56.25 * π := by
  sorry

end NUMINAMATH_CALUDE_two_sector_area_l2975_297586


namespace NUMINAMATH_CALUDE_second_rate_is_five_percent_l2975_297553

def total_sum : ℚ := 2678
def second_part : ℚ := 1648
def first_part : ℚ := total_sum - second_part
def first_rate : ℚ := 3 / 100
def first_duration : ℚ := 8
def second_duration : ℚ := 3

def first_interest : ℚ := first_part * first_rate * first_duration

theorem second_rate_is_five_percent : 
  ∃ (second_rate : ℚ), 
    second_rate * 100 = 5 ∧ 
    first_interest = second_part * second_rate * second_duration :=
sorry

end NUMINAMATH_CALUDE_second_rate_is_five_percent_l2975_297553


namespace NUMINAMATH_CALUDE_revenue_growth_equation_l2975_297560

theorem revenue_growth_equation (x : ℝ) : 
  let january_revenue : ℝ := 900000
  let total_revenue : ℝ := 1440000
  90000 * (1 + x) + 90000 * (1 + x)^2 = total_revenue - january_revenue :=
by sorry

end NUMINAMATH_CALUDE_revenue_growth_equation_l2975_297560


namespace NUMINAMATH_CALUDE_problem_solving_probability_l2975_297565

theorem problem_solving_probability (p1 p2 p3 : ℝ) 
  (h1 : p1 = 1/5) (h2 : p2 = 1/3) (h3 : p3 = 1/4) :
  1 - (1 - p1) * (1 - p2) * (1 - p3) = 3/5 := by
sorry

end NUMINAMATH_CALUDE_problem_solving_probability_l2975_297565


namespace NUMINAMATH_CALUDE_number_of_keepers_l2975_297593

/-- Represents the number of feet for each animal type --/
def animalFeet : Nat → Nat
| 0 => 2  -- hen
| 1 => 4  -- goat
| 2 => 4  -- camel
| 3 => 8  -- spider
| 4 => 8  -- octopus
| _ => 0

/-- Represents the count of each animal type --/
def animalCount : Nat → Nat
| 0 => 50  -- hens
| 1 => 45  -- goats
| 2 => 8   -- camels
| 3 => 12  -- spiders
| 4 => 6   -- octopuses
| _ => 0

/-- Calculates the total number of animal feet --/
def totalAnimalFeet : Nat :=
  List.range 5
    |> List.map (fun i => animalFeet i * animalCount i)
    |> List.sum

/-- Calculates the total number of animal heads --/
def totalAnimalHeads : Nat :=
  List.range 5
    |> List.map animalCount
    |> List.sum

/-- Theorem stating the number of keepers in the caravan --/
theorem number_of_keepers :
  ∃ k : Nat,
    k = 39 ∧
    totalAnimalFeet + (2 * k - 2) = totalAnimalHeads + k + 372 :=
by
  sorry


end NUMINAMATH_CALUDE_number_of_keepers_l2975_297593


namespace NUMINAMATH_CALUDE_probability_of_common_books_l2975_297527

def total_books : ℕ := 12
def books_to_choose : ℕ := 6
def common_books : ℕ := 3

theorem probability_of_common_books :
  (Nat.choose total_books common_books * 
   Nat.choose (total_books - common_books) (books_to_choose - common_books) * 
   Nat.choose (total_books - common_books) (books_to_choose - common_books)) / 
  (Nat.choose total_books books_to_choose * Nat.choose total_books books_to_choose) = 220 / 1215 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_common_books_l2975_297527


namespace NUMINAMATH_CALUDE_frank_candy_total_l2975_297508

/-- Given that Frank puts 11 pieces of candy in each bag and makes 2 bags,
    prove that the total number of candy pieces is 22. -/
theorem frank_candy_total (pieces_per_bag : ℕ) (num_bags : ℕ) 
    (h1 : pieces_per_bag = 11) (h2 : num_bags = 2) : 
    pieces_per_bag * num_bags = 22 := by
  sorry

end NUMINAMATH_CALUDE_frank_candy_total_l2975_297508


namespace NUMINAMATH_CALUDE_base_3_8_digit_difference_l2975_297589

/-- The number of digits in the base-b representation of a positive integer n -/
def numDigits (n : ℕ) (b : ℕ) : ℕ :=
  if n = 0 then 1 else Nat.log b n + 1

/-- The theorem stating the difference in the number of digits between base-3 and base-8 representations of 2035 -/
theorem base_3_8_digit_difference :
  numDigits 2035 3 - numDigits 2035 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_base_3_8_digit_difference_l2975_297589


namespace NUMINAMATH_CALUDE_min_triangles_to_cover_l2975_297549

/-- The minimum number of small equilateral triangles needed to cover a large equilateral triangle -/
theorem min_triangles_to_cover (small_side : ℝ) (large_side : ℝ) : 
  small_side = 2 →
  large_side = 16 →
  (large_side / small_side) ^ 2 = 64 :=
by sorry

end NUMINAMATH_CALUDE_min_triangles_to_cover_l2975_297549


namespace NUMINAMATH_CALUDE_joseph_total_distance_l2975_297517

/-- The total distance Joseph ran over 3 days, given he ran 900 meters each day. -/
def total_distance (distance_per_day : ℕ) (days : ℕ) : ℕ :=
  distance_per_day * days

/-- Theorem stating that Joseph ran 2700 meters in total. -/
theorem joseph_total_distance :
  total_distance 900 3 = 2700 := by
  sorry

end NUMINAMATH_CALUDE_joseph_total_distance_l2975_297517


namespace NUMINAMATH_CALUDE_triangle_side_length_l2975_297564

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a = √3, b = 3, and B = 2A, then c = 2√3 -/
theorem triangle_side_length (A B C : Real) (a b c : Real) :
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Angles are positive
  A + B + C = π ∧          -- Sum of angles in a triangle
  a = Real.sqrt 3 ∧        -- Given: a = √3
  b = 3 ∧                  -- Given: b = 3
  B = 2 * A →              -- Given: B = 2A
  c = 2 * Real.sqrt 3 :=   -- Conclusion: c = 2√3
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2975_297564


namespace NUMINAMATH_CALUDE_min_cost_for_equal_distribution_l2975_297536

def tangerines_needed (initial : ℕ) (people : ℕ) : ℕ :=
  (people - initial % people) % people

def cost_of_additional_tangerines (initial : ℕ) (people : ℕ) (price : ℕ) : ℕ :=
  tangerines_needed initial people * price

theorem min_cost_for_equal_distribution (initial : ℕ) (people : ℕ) (price : ℕ) 
  (h1 : initial = 98) (h2 : people = 12) (h3 : price = 450) :
  cost_of_additional_tangerines initial people price = 4500 := by
  sorry

end NUMINAMATH_CALUDE_min_cost_for_equal_distribution_l2975_297536


namespace NUMINAMATH_CALUDE_cosine_two_local_minima_l2975_297514

/-- A function f(x) = cos(ωx) has exactly two local minimum points in [0, π/2] iff 6 ≤ ω < 10 -/
theorem cosine_two_local_minima (ω : ℝ) (h : ω > 0) :
  (∃! (n : ℕ), n = 2 ∧ ∀ (x : ℝ), x ∈ Set.Icc 0 (π / 2) →
    (∃ (ε : ℝ), ε > 0 ∧ ∀ (y : ℝ), y ∈ Set.Ioo (x - ε) (x + ε) →
      Real.cos (ω * y) ≥ Real.cos (ω * x))) ↔
  6 ≤ ω ∧ ω < 10 :=
sorry

end NUMINAMATH_CALUDE_cosine_two_local_minima_l2975_297514


namespace NUMINAMATH_CALUDE_f_is_cone_bottomed_g_is_not_cone_bottomed_h_max_cone_bottomed_constant_l2975_297573

-- Definition of a "cone-bottomed" function
def is_cone_bottomed (f : ℝ → ℝ) : Prop :=
  ∃ M : ℝ, M > 0 ∧ ∀ x : ℝ, |f x| ≥ M * |x|

-- Specific functions
def f (x : ℝ) : ℝ := 2 * x
def g (x : ℝ) : ℝ := x^3
def h (x : ℝ) : ℝ := x^2 + 1

-- Theorems to prove
theorem f_is_cone_bottomed : is_cone_bottomed f := sorry

theorem g_is_not_cone_bottomed : ¬ is_cone_bottomed g := sorry

theorem h_max_cone_bottomed_constant :
  ∀ M : ℝ, (is_cone_bottomed h ∧ ∀ N : ℝ, is_cone_bottomed h → N ≤ M) → M = 2 := sorry

end NUMINAMATH_CALUDE_f_is_cone_bottomed_g_is_not_cone_bottomed_h_max_cone_bottomed_constant_l2975_297573


namespace NUMINAMATH_CALUDE_road_trip_time_calculation_l2975_297557

/-- Calculates the total time for a road trip given the specified conditions -/
theorem road_trip_time_calculation (distance : ℝ) (speed : ℝ) (break_interval : ℝ) (break_duration : ℝ) (hotel_search_time : ℝ) : 
  distance = 2790 →
  speed = 62 →
  break_interval = 5 →
  break_duration = 0.5 →
  hotel_search_time = 0.5 →
  (distance / speed + 
   (⌊distance / speed / break_interval⌋ - 1) * break_duration + 
   hotel_search_time) = 49.5 := by
  sorry

#check road_trip_time_calculation

end NUMINAMATH_CALUDE_road_trip_time_calculation_l2975_297557


namespace NUMINAMATH_CALUDE_triangle_ratio_theorem_l2975_297559

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_ratio_theorem (abc : Triangle) :
  abc.a = 5 →
  Real.cos abc.B = 4/5 →
  (1/2) * abc.a * abc.c * Real.sin abc.B = 12 →
  (abc.a + abc.c) / (Real.sin abc.A + Real.sin abc.C) = 25/3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_ratio_theorem_l2975_297559


namespace NUMINAMATH_CALUDE_zahra_kimmie_ratio_l2975_297538

def kimmie_earnings : ℚ := 450
def total_savings : ℚ := 375

theorem zahra_kimmie_ratio (zahra_earnings : ℚ) 
  (h1 : zahra_earnings < kimmie_earnings)
  (h2 : total_savings = (1/2) * kimmie_earnings + (1/2) * zahra_earnings) :
  zahra_earnings / kimmie_earnings = 2/3 := by
sorry

end NUMINAMATH_CALUDE_zahra_kimmie_ratio_l2975_297538


namespace NUMINAMATH_CALUDE_proposition_is_false_l2975_297562

theorem proposition_is_false : ¬(∀ x : ℤ, x ∈ ({1, -1, 0} : Set ℤ) → 2*x + 1 > 0) := by
  sorry

end NUMINAMATH_CALUDE_proposition_is_false_l2975_297562


namespace NUMINAMATH_CALUDE_no_natural_squares_diff_2018_l2975_297580

theorem no_natural_squares_diff_2018 : ¬∃ (a b : ℕ), a^2 - b^2 = 2018 := by
  sorry

end NUMINAMATH_CALUDE_no_natural_squares_diff_2018_l2975_297580


namespace NUMINAMATH_CALUDE_distance_between_vertices_l2975_297595

-- Define the two parabolas
def parabola1 (x : ℝ) : ℝ := x^2 - 6*x + 13
def parabola2 (x : ℝ) : ℝ := x^2 + 2*x + 4

-- Define the vertex of a parabola
def vertex (f : ℝ → ℝ) : ℝ × ℝ := sorry

-- Define the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

theorem distance_between_vertices : 
  distance (vertex parabola1) (vertex parabola2) = Real.sqrt 17 := by sorry

end NUMINAMATH_CALUDE_distance_between_vertices_l2975_297595


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l2975_297571

/-- Given simple interest, principal, and time, calculate the interest rate in paise per rupee per month -/
theorem interest_rate_calculation (simple_interest principal time : ℚ) 
  (h1 : simple_interest = 4.8)
  (h2 : principal = 8)
  (h3 : time = 12) :
  (simple_interest / (principal * time)) * 100 = 5 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l2975_297571


namespace NUMINAMATH_CALUDE_simplify_2A_minus_3B_value_2A_minus_3B_specific_value_2A_minus_3B_independent_l2975_297558

-- Define A and B as functions of x and y
def A (x y : ℝ) : ℝ := -x + 2*y - 4*x*y
def B (x y : ℝ) : ℝ := -3*x - y + x*y

-- Theorem 1: Simplification of 2A - 3B
theorem simplify_2A_minus_3B (x y : ℝ) :
  2 * A x y - 3 * B x y = 7*x + 7*y - 11*x*y := by sorry

-- Theorem 2: Value of 2A - 3B under specific conditions
theorem value_2A_minus_3B_specific (x y : ℝ) 
  (h1 : x + y = 6/7) (h2 : x * y = -2) :
  2 * A x y - 3 * B x y = 28 := by sorry

-- Theorem 3: Value of 2A - 3B when independent of y
theorem value_2A_minus_3B_independent (x : ℝ) 
  (h : ∀ y : ℝ, 2 * A x y - 3 * B x y = 2 * A x 0 - 3 * B x 0) :
  2 * A x 0 - 3 * B x 0 = 49/11 := by sorry

end NUMINAMATH_CALUDE_simplify_2A_minus_3B_value_2A_minus_3B_specific_value_2A_minus_3B_independent_l2975_297558


namespace NUMINAMATH_CALUDE_max_value_quadratic_l2975_297548

theorem max_value_quadratic (r : ℝ) : 
  -3 * r^2 + 30 * r + 8 ≤ 83 ∧ ∃ r : ℝ, -3 * r^2 + 30 * r + 8 = 83 := by
  sorry

end NUMINAMATH_CALUDE_max_value_quadratic_l2975_297548


namespace NUMINAMATH_CALUDE_min_value_of_reciprocal_sum_l2975_297539

theorem min_value_of_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 3) :
  1/a + 4/b ≥ 3 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + b₀ = 3 ∧ 1/a₀ + 4/b₀ = 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_reciprocal_sum_l2975_297539


namespace NUMINAMATH_CALUDE_pfd_product_theorem_l2975_297543

/-- Partial fraction decomposition coefficients -/
structure PFDCoefficients where
  A : ℚ
  B : ℚ
  C : ℚ

/-- The partial fraction decomposition of (x^2 - 25) / ((x - 1)(x + 3)(x - 4)) -/
def partial_fraction_decomposition : (ℚ → ℚ) → PFDCoefficients → Prop :=
  λ f coeffs =>
    ∀ x, x ≠ 1 ∧ x ≠ -3 ∧ x ≠ 4 →
      f x = coeffs.A / (x - 1) + coeffs.B / (x + 3) + coeffs.C / (x - 4)

/-- The original rational function -/
def original_function (x : ℚ) : ℚ :=
  (x^2 - 25) / ((x - 1) * (x + 3) * (x - 4))

theorem pfd_product_theorem :
  ∃ coeffs : PFDCoefficients,
    partial_fraction_decomposition original_function coeffs ∧
    coeffs.A * coeffs.B * coeffs.C = 24/49 := by
  sorry

end NUMINAMATH_CALUDE_pfd_product_theorem_l2975_297543


namespace NUMINAMATH_CALUDE_sanchez_problem_l2975_297591

theorem sanchez_problem (x y : ℕ+) : x - y = 3 → x * y = 56 → x + y = 17 := by sorry

end NUMINAMATH_CALUDE_sanchez_problem_l2975_297591


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l2975_297504

theorem partial_fraction_decomposition :
  let f (x : ℝ) := (2 * x^2 + 5 * x - 3) / (x^2 - x - 42)
  let g (x : ℝ) := (11/13) / (x - 7) + (15/13) / (x + 6)
  ∀ x : ℝ, x ≠ 7 → x ≠ -6 → f x = g x :=
by sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l2975_297504


namespace NUMINAMATH_CALUDE_perfect_square_condition_l2975_297503

theorem perfect_square_condition (n : ℕ) : 
  (∃ (a : ℕ), 2^n + 3 = a^2) ↔ n = 0 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l2975_297503


namespace NUMINAMATH_CALUDE_system_two_solutions_l2975_297520

/-- The system of equations has exactly two solutions if and only if a = 49 or a = 289 -/
theorem system_two_solutions (a : ℝ) : 
  (∃! x y : ℝ, |x + y + 8| + |x - y + 8| = 16 ∧ (|x| - 8)^2 + (|y| - 15)^2 = a) ↔ 
  (a = 49 ∨ a = 289) :=
sorry

end NUMINAMATH_CALUDE_system_two_solutions_l2975_297520


namespace NUMINAMATH_CALUDE_uranus_appearance_time_l2975_297532

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  inv : minutes < 60

/-- Adds minutes to a given time -/
def addMinutes (t : Time) (m : ℕ) : Time :=
  let totalMinutes := t.minutes + m
  let newHours := t.hours + totalMinutes / 60
  let newMinutes := totalMinutes % 60
  ⟨newHours % 24, newMinutes, by sorry⟩

/-- Calculates the difference in minutes between two times -/
def minutesBetween (t1 t2 : Time) : ℕ :=
  (t2.hours - t1.hours) * 60 + (t2.minutes - t1.minutes)

theorem uranus_appearance_time
  (marsDisappearance : Time)
  (jupiterDelay : ℕ)
  (uranusDelay : ℕ)
  (h1 : marsDisappearance = ⟨0, 10, by sorry⟩)  -- 12:10 AM
  (h2 : jupiterDelay = 161)  -- 2 hours and 41 minutes
  (h3 : uranusDelay = 196)  -- 3 hours and 16 minutes
  : minutesBetween ⟨6, 0, by sorry⟩ (addMinutes (addMinutes marsDisappearance jupiterDelay) uranusDelay) = 7 :=
by sorry

end NUMINAMATH_CALUDE_uranus_appearance_time_l2975_297532


namespace NUMINAMATH_CALUDE_min_absolute_sum_l2975_297554

theorem min_absolute_sum (x : ℝ) : 
  |x + 1| + |x + 3| + |x + 6| ≥ 5 ∧ ∃ y : ℝ, |y + 1| + |y + 3| + |y + 6| = 5 :=
by sorry

end NUMINAMATH_CALUDE_min_absolute_sum_l2975_297554


namespace NUMINAMATH_CALUDE_trig_identity_l2975_297528

theorem trig_identity : (2 * Real.cos (10 * π / 180) - Real.sin (20 * π / 180)) / Real.sin (70 * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l2975_297528


namespace NUMINAMATH_CALUDE_fans_with_all_items_l2975_297555

/-- The number of fans in the stadium -/
def total_fans : ℕ := 5000

/-- The interval for t-shirt vouchers -/
def t_shirt_interval : ℕ := 60

/-- The interval for cap vouchers -/
def cap_interval : ℕ := 45

/-- The interval for water bottle vouchers -/
def water_bottle_interval : ℕ := 40

/-- Theorem: The number of fans receiving all three items is equal to the floor of total_fans divided by the LCM of the three intervals -/
theorem fans_with_all_items (total_fans t_shirt_interval cap_interval water_bottle_interval : ℕ) :
  (total_fans / Nat.lcm (Nat.lcm t_shirt_interval cap_interval) water_bottle_interval : ℕ) = 13 :=
sorry

end NUMINAMATH_CALUDE_fans_with_all_items_l2975_297555


namespace NUMINAMATH_CALUDE_f_properties_l2975_297572

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then x + 5 else -2 * x + 8

theorem f_properties :
  (f 2 = 4) ∧
  (f (f (-1)) = 0) ∧
  (∀ x, f x ≥ 4 ↔ -1 ≤ x ∧ x ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l2975_297572


namespace NUMINAMATH_CALUDE_prob_A_or_B_l2975_297563

/- Given probabilities -/
def P_A : ℝ := 0.4
def P_B : ℝ := 0.65
def P_A_and_B : ℝ := 0.25

/- Theorem to prove -/
theorem prob_A_or_B : P_A + P_B - P_A_and_B = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_prob_A_or_B_l2975_297563


namespace NUMINAMATH_CALUDE_smallest_possible_abs_z_l2975_297547

theorem smallest_possible_abs_z (z : ℂ) (h : Complex.abs (z - 8) + Complex.abs (z - Complex.I * 7) = 17) :
  ∃ (w : ℂ), Complex.abs w ≤ Complex.abs z ∧ Complex.abs (w - 8) + Complex.abs (w - Complex.I * 7) = 17 ∧ Complex.abs w = 7 / Real.sqrt 113 :=
sorry

end NUMINAMATH_CALUDE_smallest_possible_abs_z_l2975_297547


namespace NUMINAMATH_CALUDE_dima_speed_ratio_l2975_297530

/-- Represents the time it takes Dima to walk from home to school -/
def walk_time : ℝ := 24

/-- Represents the time it takes Dima to run from home to school -/
def run_time : ℝ := 12

/-- Represents the time remaining before the school bell rings when Dima realizes he forgot his phone -/
def time_remaining : ℝ := 15

/-- States that Dima walks halfway to school before realizing he forgot his phone -/
axiom halfway_condition : walk_time / 2 = time_remaining - 3

/-- States that if Dima runs back home and then to school, he'll be 3 minutes late -/
axiom run_condition : run_time / 2 + run_time = time_remaining + 3

/-- States that if Dima runs back home and then walks to school, he'll be 15 minutes late -/
axiom run_walk_condition : run_time / 2 + walk_time = time_remaining + 15

/-- Theorem stating that Dima's running speed is twice his walking speed -/
theorem dima_speed_ratio : walk_time / run_time = 2 := by sorry

end NUMINAMATH_CALUDE_dima_speed_ratio_l2975_297530


namespace NUMINAMATH_CALUDE_weight_of_b_l2975_297561

theorem weight_of_b (a b c : ℝ) 
  (h1 : (a + b + c) / 3 = 45)
  (h2 : (a + b) / 2 = 40)
  (h3 : (b + c) / 2 = 45) :
  b = 35 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_b_l2975_297561


namespace NUMINAMATH_CALUDE_caps_collection_total_l2975_297511

theorem caps_collection_total (A B C : ℕ) : 
  A = (B + C) / 2 →
  B = (A + C) / 3 →
  C = 150 →
  A + B + C = 360 := by
sorry

end NUMINAMATH_CALUDE_caps_collection_total_l2975_297511


namespace NUMINAMATH_CALUDE_ice_cream_sales_theorem_l2975_297575

def ice_cream_sales (monday tuesday : ℕ) : Prop :=
  ∃ (wednesday thursday total : ℕ),
    wednesday = 2 * tuesday ∧
    thursday = (3 * wednesday) / 2 ∧
    total = monday + tuesday + wednesday + thursday ∧
    total = 82000

theorem ice_cream_sales_theorem :
  ice_cream_sales 10000 12000 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_sales_theorem_l2975_297575


namespace NUMINAMATH_CALUDE_smallest_y_for_perfect_cube_l2975_297566

def x : ℕ := 7 * 24 * 48

def is_perfect_cube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3

theorem smallest_y_for_perfect_cube :
  ∃! y : ℕ, y > 0 ∧ is_perfect_cube (x * y) ∧ ∀ z : ℕ, z > 0 ∧ z < y → ¬is_perfect_cube (x * z) :=
by sorry

end NUMINAMATH_CALUDE_smallest_y_for_perfect_cube_l2975_297566


namespace NUMINAMATH_CALUDE_triangle_height_l2975_297587

theorem triangle_height (area : ℝ) (base : ℝ) (height : ℝ)
  (h_area : area = 24)
  (h_base : base = 8)
  (h_triangle_area : area = (base * height) / 2) :
  height = 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_height_l2975_297587


namespace NUMINAMATH_CALUDE_abs_inequality_solution_set_l2975_297506

theorem abs_inequality_solution_set (x : ℝ) : 
  |3*x + 1| - |x - 1| < 0 ↔ -1 < x ∧ x < 1 := by sorry

end NUMINAMATH_CALUDE_abs_inequality_solution_set_l2975_297506


namespace NUMINAMATH_CALUDE_exchange_calculation_l2975_297510

/-- Exchange rate between lire and dollars -/
def exchange_rate : ℚ := 2500 / 2

/-- Amount of dollars to be exchanged -/
def dollars_to_exchange : ℚ := 5

/-- Function to calculate lire received for a given amount of dollars -/
def lire_received (dollars : ℚ) : ℚ := dollars * exchange_rate

theorem exchange_calculation :
  lire_received dollars_to_exchange = 6250 := by
  sorry

end NUMINAMATH_CALUDE_exchange_calculation_l2975_297510


namespace NUMINAMATH_CALUDE_video_streaming_cost_theorem_l2975_297521

/-- Calculates the total cost for one person's share of a video streaming subscription over a year -/
theorem video_streaming_cost_theorem 
  (monthly_cost : ℝ) 
  (num_people_sharing : ℕ) 
  (months_in_year : ℕ) 
  (h1 : monthly_cost = 14) 
  (h2 : num_people_sharing = 2) 
  (h3 : months_in_year = 12) :
  (monthly_cost / num_people_sharing) * months_in_year = 84 := by
  sorry

end NUMINAMATH_CALUDE_video_streaming_cost_theorem_l2975_297521


namespace NUMINAMATH_CALUDE_percentage_relation_l2975_297590

theorem percentage_relation (a b c : ℝ) 
  (h1 : c = 0.14 * a) 
  (h2 : c = 0.40 * b) : 
  b = 0.35 * a := by
sorry

end NUMINAMATH_CALUDE_percentage_relation_l2975_297590


namespace NUMINAMATH_CALUDE_cinema_uses_systematic_sampling_l2975_297526

/-- Represents a sampling method --/
inductive SamplingMethod
| Lottery
| Stratified
| RandomNumberTable
| Systematic

/-- Represents a cinema with rows and seats per row --/
structure Cinema where
  rows : Nat
  seatsPerRow : Nat

/-- Represents a selection rule for seats --/
structure SelectionRule where
  endDigit : Nat

/-- Determines if a sampling method is systematic based on cinema layout and selection rule --/
def isSystematicSampling (c : Cinema) (r : SelectionRule) : Prop :=
  r.endDigit = c.seatsPerRow % 10 ∧ c.seatsPerRow % 10 ≠ 0

/-- Theorem stating that the given cinema scenario uses systematic sampling --/
theorem cinema_uses_systematic_sampling (c : Cinema) (r : SelectionRule) :
  c.rows = 50 → c.seatsPerRow = 30 → r.endDigit = 8 →
  isSystematicSampling c r ∧ SamplingMethod.Systematic = SamplingMethod.Systematic := by
  sorry

end NUMINAMATH_CALUDE_cinema_uses_systematic_sampling_l2975_297526


namespace NUMINAMATH_CALUDE_alice_spending_percentage_l2975_297551

theorem alice_spending_percentage (alice_initial bob_initial alice_final : ℝ) :
  bob_initial = 0.9 * alice_initial →
  alice_final = 0.9 * bob_initial →
  (alice_initial - alice_final) / alice_initial = 0.19 := by
  sorry

end NUMINAMATH_CALUDE_alice_spending_percentage_l2975_297551


namespace NUMINAMATH_CALUDE_max_angle_on_perp_bisector_l2975_297599

-- Define the circle
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the angle between three points
def angle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- Define the perpendicular bisector of a line segment
def perpBisector (p1 p2 : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

theorem max_angle_on_perp_bisector 
  (O A : ℝ × ℝ) (r : ℝ) 
  (h_circle : Circle O r)
  (h_interior : A ∈ interior (Circle O r))
  (h_different : A ≠ O) :
  ∃ P : ℝ × ℝ, P ∈ Circle O r ∧ 
    P ∈ perpBisector O A ∧
    ∀ Q : ℝ × ℝ, Q ∈ Circle O r → angle O P A ≥ angle O Q A :=
sorry

end NUMINAMATH_CALUDE_max_angle_on_perp_bisector_l2975_297599


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2975_297594

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x > 0}
def B : Set ℝ := {x : ℝ | x < 4}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 0 < x ∧ x < 4} :=
sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2975_297594


namespace NUMINAMATH_CALUDE_eccentricity_for_one_and_nine_l2975_297570

/-- The eccentricity of a curve given two positive numbers -/
def eccentricity_of_curve (x y : ℝ) : Set ℝ :=
  let a := (x + y) / 2
  let b := Real.sqrt (x * y)
  let e₁ := Real.sqrt (a - b) / Real.sqrt a
  let e₂ := Real.sqrt (a + b) / Real.sqrt a
  {e₁, e₂}

/-- Theorem: The eccentricity of the curve for numbers 1 and 9 -/
theorem eccentricity_for_one_and_nine :
  eccentricity_of_curve 1 9 = {Real.sqrt 10 / 5, 2 * Real.sqrt 10 / 5} :=
by sorry

end NUMINAMATH_CALUDE_eccentricity_for_one_and_nine_l2975_297570


namespace NUMINAMATH_CALUDE_stock_price_after_two_years_l2975_297583

/-- The stock price after two years of changes -/
theorem stock_price_after_two_years 
  (initial_price : ℝ) 
  (first_year_increase : ℝ) 
  (second_year_decrease : ℝ) 
  (h1 : initial_price = 120)
  (h2 : first_year_increase = 1.2)
  (h3 : second_year_decrease = 0.3) :
  initial_price * (1 + first_year_increase) * (1 - second_year_decrease) = 184.8 := by
  sorry

end NUMINAMATH_CALUDE_stock_price_after_two_years_l2975_297583


namespace NUMINAMATH_CALUDE_sin_plus_cos_equals_sqrt_a_plus_one_l2975_297540

theorem sin_plus_cos_equals_sqrt_a_plus_one (θ : Real) (a : Real) 
  (h1 : 0 < θ ∧ θ < π / 2) -- θ is an acute angle
  (h2 : Real.sin (2 * θ) = a) : -- sin 2θ = a
  Real.sin θ + Real.cos θ = Real.sqrt (a + 1) := by sorry

end NUMINAMATH_CALUDE_sin_plus_cos_equals_sqrt_a_plus_one_l2975_297540


namespace NUMINAMATH_CALUDE_solution_set_implies_sum_l2975_297524

/-- If the solution set of (x-a)(x-b) < 0 is (-1,2), then a+b = 1 -/
theorem solution_set_implies_sum (a b : ℝ) : 
  (∀ x, (x-a)*(x-b) < 0 ↔ -1 < x ∧ x < 2) → a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_implies_sum_l2975_297524


namespace NUMINAMATH_CALUDE_cube_sum_equals_four_l2975_297541

theorem cube_sum_equals_four (x y : ℝ) 
  (h1 : x + y = 1) 
  (h2 : x^2 + y^2 = 3) : 
  x^3 + y^3 = 4 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_equals_four_l2975_297541


namespace NUMINAMATH_CALUDE_sin_48_greater_cos_48_l2975_297556

theorem sin_48_greater_cos_48 : Real.sin (48 * π / 180) > Real.cos (48 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_sin_48_greater_cos_48_l2975_297556


namespace NUMINAMATH_CALUDE_triangle_side_length_l2975_297574

/-- In a triangle ABC, given that tan B = √3, AB = 3, and the area is (3√3)/2, prove that AC = √7 -/
theorem triangle_side_length (B : Real) (C : Real) (tanB : Real.tan B = Real.sqrt 3) 
  (AB : Real) (hAB : AB = 3) (area : Real) (harea : area = (3 * Real.sqrt 3) / 2) : 
  Real.sqrt ((AB^2) + (2^2) - 2 * AB * 2 * Real.cos B) = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2975_297574


namespace NUMINAMATH_CALUDE_service_fee_is_24_percent_l2975_297518

/-- Calculates the service fee percentage given the cost of food, tip, and total amount spent. -/
def service_fee_percentage (food_cost tip total_spent : ℚ) : ℚ :=
  ((total_spent - food_cost - tip) / food_cost) * 100

/-- Theorem stating that the service fee percentage is 24% given the problem conditions. -/
theorem service_fee_is_24_percent :
  let food_cost : ℚ := 50
  let tip : ℚ := 5
  let total_spent : ℚ := 61
  service_fee_percentage food_cost tip total_spent = 24 := by
  sorry

end NUMINAMATH_CALUDE_service_fee_is_24_percent_l2975_297518


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l2975_297525

theorem quadratic_equal_roots (x : ℝ) : 
  (∃ r : ℝ, (x^2 + 2*x + 1 = 0) ↔ (x = r ∧ x = r)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l2975_297525


namespace NUMINAMATH_CALUDE_logarithm_sum_simplification_l2975_297588

theorem logarithm_sum_simplification :
  1 / (Real.log 2 / Real.log 7 + 1) +
  1 / (Real.log 3 / Real.log 11 + 1) +
  1 / (Real.log 5 / Real.log 13 + 1) = 3 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_sum_simplification_l2975_297588


namespace NUMINAMATH_CALUDE_charlene_necklaces_l2975_297515

theorem charlene_necklaces (sold : ℕ) (given_away : ℕ) (left : ℕ) 
  (h1 : sold = 16) (h2 : given_away = 18) (h3 : left = 26) :
  sold + given_away + left = 60 := by
  sorry

end NUMINAMATH_CALUDE_charlene_necklaces_l2975_297515


namespace NUMINAMATH_CALUDE_sum_of_decimals_l2975_297569

/-- The sum of 0.2, 0.03, 0.004, 0.0005, and 0.00006 is equal to 5864/25000 -/
theorem sum_of_decimals : 
  (0.2 : ℚ) + 0.03 + 0.004 + 0.0005 + 0.00006 = 5864 / 25000 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_decimals_l2975_297569


namespace NUMINAMATH_CALUDE_line_y_intercept_l2975_297542

/-- Given a line with equation 3x - y + 6 = 0, prove that its y-intercept is 6 -/
theorem line_y_intercept (x y : ℝ) (h : 3 * x - y + 6 = 0) : y = 6 ↔ x = 0 :=
sorry

end NUMINAMATH_CALUDE_line_y_intercept_l2975_297542


namespace NUMINAMATH_CALUDE_sqrt_neg_four_squared_plus_cube_root_neg_eight_equals_two_l2975_297534

theorem sqrt_neg_four_squared_plus_cube_root_neg_eight_equals_two :
  Real.sqrt ((-4)^2) + ((-8 : ℝ) ^ (1/3 : ℝ)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_neg_four_squared_plus_cube_root_neg_eight_equals_two_l2975_297534


namespace NUMINAMATH_CALUDE_square_sheet_area_l2975_297507

theorem square_sheet_area (x : ℝ) : 
  x > 0 → x * (x - 3) = 40 → x^2 = 64 := by
  sorry

end NUMINAMATH_CALUDE_square_sheet_area_l2975_297507


namespace NUMINAMATH_CALUDE_triangle_properties_l2975_297509

/-- Represents an acute triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure AcuteTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2
  law_of_sines : a / Real.sin A = b / Real.sin B
  law_of_cosines : c^2 = a^2 + b^2 - 2*a*b*Real.cos C

/-- Given conditions for the specific triangle -/
def special_triangle (t : AcuteTriangle) : Prop :=
  t.a = 2 * t.b * Real.sin t.A ∧ t.a = 3 * Real.sqrt 3 ∧ t.c = 5

theorem triangle_properties (t : AcuteTriangle) (h : special_triangle t) : 
  t.B = π/6 ∧ t.b = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l2975_297509


namespace NUMINAMATH_CALUDE_product_digits_count_l2975_297585

theorem product_digits_count : ∃ n : ℕ, 
  (1002000000000000000 * 999999999999999999 : ℕ) ≥ 10^37 ∧ 
  (1002000000000000000 * 999999999999999999 : ℕ) < 10^38 :=
by sorry

end NUMINAMATH_CALUDE_product_digits_count_l2975_297585


namespace NUMINAMATH_CALUDE_m_gt_neg_one_sufficient_not_necessary_l2975_297516

/-- Represents the equation of a conic section in the form (x^2 / a) - (y^2 / b) = 1 --/
structure ConicSection where
  a : ℝ
  b : ℝ

/-- Defines when a ConicSection represents a hyperbola --/
def is_hyperbola (c : ConicSection) : Prop :=
  c.a > 0 ∧ c.b > 0

/-- The conic section defined by the given equation --/
def conic_equation (m : ℝ) : ConicSection :=
  { a := 2 + m, b := 1 + m }

/-- The theorem to be proved --/
theorem m_gt_neg_one_sufficient_not_necessary :
  (∀ m : ℝ, m > -1 → is_hyperbola (conic_equation m)) ∧
  ¬(∀ m : ℝ, is_hyperbola (conic_equation m) → m > -1) :=
sorry

end NUMINAMATH_CALUDE_m_gt_neg_one_sufficient_not_necessary_l2975_297516


namespace NUMINAMATH_CALUDE_original_fraction_l2975_297519

theorem original_fraction (x y : ℚ) : 
  (x * (1 + 12/100)) / (y * (1 - 2/100)) = 6/7 → x/y = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_original_fraction_l2975_297519


namespace NUMINAMATH_CALUDE_dow_jones_problem_l2975_297544

/-- The Dow Jones Industrial Average problem -/
theorem dow_jones_problem (end_value : ℝ) (percent_fall : ℝ) :
  end_value = 8722 →
  percent_fall = 2 →
  (1 - percent_fall / 100) * 8900 = end_value :=
by
  sorry

end NUMINAMATH_CALUDE_dow_jones_problem_l2975_297544


namespace NUMINAMATH_CALUDE_first_day_price_is_four_l2975_297579

/-- Represents the pen sales scenario over three days -/
structure PenSales where
  day1_price : ℝ
  day1_quantity : ℝ

/-- The revenue is the same for all three days -/
def same_revenue (s : PenSales) : Prop :=
  s.day1_price * s.day1_quantity = 
  (s.day1_price - 1) * (s.day1_quantity + 100) ∧
  s.day1_price * s.day1_quantity = 
  (s.day1_price + 2) * (s.day1_quantity - 100)

/-- The price on the first day is 4 yuan -/
theorem first_day_price_is_four :
  ∃ (s : PenSales), same_revenue s ∧ s.day1_price = 4 := by
  sorry

end NUMINAMATH_CALUDE_first_day_price_is_four_l2975_297579


namespace NUMINAMATH_CALUDE_feet_per_mile_l2975_297552

/-- Proves that if an object travels 200 feet in 2 seconds with a speed of 68.18181818181819 miles per hour, then there are 5280 feet in one mile. -/
theorem feet_per_mile (distance : ℝ) (time : ℝ) (speed : ℝ) (feet_per_mile : ℝ) :
  distance = 200 →
  time = 2 →
  speed = 68.18181818181819 →
  distance / time = speed * feet_per_mile / 3600 →
  feet_per_mile = 5280 := by
  sorry

end NUMINAMATH_CALUDE_feet_per_mile_l2975_297552


namespace NUMINAMATH_CALUDE_no_geometric_triple_in_arithmetic_sequence_l2975_297581

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

-- Define the property of containing 1 and √2
def contains_one_and_sqrt_two (a : ℕ → ℝ) : Prop :=
  ∃ k l : ℕ, a k = 1 ∧ a l = Real.sqrt 2

-- Define a geometric sequence of three terms
def geometric_sequence (x y z : ℝ) : Prop :=
  y^2 = x * z

-- Main theorem
theorem no_geometric_triple_in_arithmetic_sequence 
  (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a) 
  (h2 : contains_one_and_sqrt_two a) : 
  ¬∃ i j k : ℕ, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ geometric_sequence (a i) (a j) (a k) :=
sorry

end NUMINAMATH_CALUDE_no_geometric_triple_in_arithmetic_sequence_l2975_297581


namespace NUMINAMATH_CALUDE_total_precious_stones_l2975_297501

/-- The number of precious stones in agate -/
def agate_stones : ℕ := 30

/-- The number of precious stones in olivine -/
def olivine_stones : ℕ := agate_stones + 5

/-- The number of precious stones in diamond -/
def diamond_stones : ℕ := olivine_stones + 11

/-- The total number of precious stones in agate, olivine, and diamond -/
def total_stones : ℕ := agate_stones + olivine_stones + diamond_stones

theorem total_precious_stones : total_stones = 111 := by
  sorry

end NUMINAMATH_CALUDE_total_precious_stones_l2975_297501


namespace NUMINAMATH_CALUDE_proportion_equality_l2975_297567

-- Define variables a and b
variable (a b : ℝ)

-- Define the given condition
def condition : Prop := 2 * a = 5 * b

-- State the theorem to be proved
theorem proportion_equality (h : condition a b) : a / 5 = b / 2 := by
  sorry

end NUMINAMATH_CALUDE_proportion_equality_l2975_297567

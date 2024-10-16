import Mathlib

namespace NUMINAMATH_CALUDE_apples_to_sell_for_target_profit_l3337_333738

/-- Represents the number of apples bought in one transaction -/
def apples_bought : ℕ := 4

/-- Represents the cost in cents for buying apples_bought apples -/
def buying_cost : ℕ := 15

/-- Represents the number of apples sold in one transaction -/
def apples_sold : ℕ := 7

/-- Represents the revenue in cents from selling apples_sold apples -/
def selling_revenue : ℕ := 35

/-- Represents the target profit in cents -/
def target_profit : ℕ := 140

/-- Theorem stating that 112 apples need to be sold to achieve the target profit -/
theorem apples_to_sell_for_target_profit :
  (selling_revenue * 112 / apples_sold) - (buying_cost * 112 / apples_bought) = target_profit := by
  sorry

end NUMINAMATH_CALUDE_apples_to_sell_for_target_profit_l3337_333738


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l3337_333764

theorem necessary_not_sufficient_condition (a b c : ℝ) :
  (∀ a b c : ℝ, a * c^2 < b * c^2 → a < b) ∧
  (∃ a b c : ℝ, a < b ∧ ¬(a * c^2 < b * c^2)) :=
sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l3337_333764


namespace NUMINAMATH_CALUDE_selling_price_equation_l3337_333739

/-- Represents the selling price of pants in a store -/
def selling_price (X : ℝ) : ℝ :=
  let initial_price := X
  let discount_rate := 0.1
  let bulk_discount := 5
  let markup_rate := 0.25
  let discounted_price := initial_price * (1 - discount_rate)
  let final_purchase_cost := discounted_price - bulk_discount
  let marked_up_price := final_purchase_cost * (1 + markup_rate)
  marked_up_price

/-- Theorem stating the relationship between initial purchase price and selling price -/
theorem selling_price_equation (X : ℝ) :
  selling_price X = 1.125 * X - 6.25 := by
  sorry

end NUMINAMATH_CALUDE_selling_price_equation_l3337_333739


namespace NUMINAMATH_CALUDE_some_number_equation_l3337_333769

theorem some_number_equation (y : ℝ) : 
  ∃ (n : ℝ), n * (1 + y) + 17 = n * (-1 + y) - 21 ∧ n = -19 := by
  sorry

end NUMINAMATH_CALUDE_some_number_equation_l3337_333769


namespace NUMINAMATH_CALUDE_charlie_crayon_count_l3337_333762

/-- The number of crayons each person has -/
structure CrayonCounts where
  billie : ℕ
  bobbie : ℕ
  lizzie : ℕ
  charlie : ℕ

/-- The conditions of the crayon problem -/
def crayon_problem (c : CrayonCounts) : Prop :=
  c.billie = 18 ∧
  c.bobbie = 3 * c.billie ∧
  c.lizzie = c.bobbie / 2 ∧
  c.charlie = 2 * c.lizzie

theorem charlie_crayon_count (c : CrayonCounts) (h : crayon_problem c) : c.charlie = 54 := by
  sorry

end NUMINAMATH_CALUDE_charlie_crayon_count_l3337_333762


namespace NUMINAMATH_CALUDE_freds_dark_blue_marbles_l3337_333783

/-- Proves that the number of dark blue marbles is 6 given the conditions of Fred's marble collection. -/
theorem freds_dark_blue_marbles :
  let total_marbles : ℕ := 63
  let red_marbles : ℕ := 38
  let green_marbles : ℕ := red_marbles / 2
  let dark_blue_marbles : ℕ := total_marbles - red_marbles - green_marbles
  dark_blue_marbles = 6 := by
  sorry

end NUMINAMATH_CALUDE_freds_dark_blue_marbles_l3337_333783


namespace NUMINAMATH_CALUDE_gcd_problem_l3337_333714

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = 2142 * k) : 
  Nat.gcd (Int.natAbs (b^2 + 11*b + 28)) (Int.natAbs (b + 6)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l3337_333714


namespace NUMINAMATH_CALUDE_race_distance_proof_l3337_333741

/-- The distance Alex and Max ran together at the beginning of the race -/
def initial_even_distance : ℕ := 540

/-- The total race distance in feet -/
def total_race_distance : ℕ := 5000

/-- The distance left for Max to catch up to Alex at the end -/
def remaining_distance : ℕ := 3890

/-- The sum of relative position changes between Alex and Max -/
def relative_position_changes : ℕ := 570

theorem race_distance_proof :
  initial_even_distance + relative_position_changes = total_race_distance - remaining_distance :=
by sorry

end NUMINAMATH_CALUDE_race_distance_proof_l3337_333741


namespace NUMINAMATH_CALUDE_common_chord_intersection_l3337_333754

-- Define a type for points in a plane
variable (Point : Type)

-- Define a type for circles
variable (Circle : Type)

-- Function to check if a point is on a circle
variable (on_circle : Point → Circle → Prop)

-- Function to check if two circles intersect
variable (intersect : Circle → Circle → Prop)

-- Function to create a circle passing through two points
variable (circle_through : Point → Point → Circle)

-- Function to find the common chord of two circles
variable (common_chord : Circle → Circle → Set Point)

-- Theorem statement
theorem common_chord_intersection
  (A B C D : Point)
  (h : ∀ (c1 c2 : Circle), on_circle A c1 → on_circle B c1 → 
                           on_circle C c2 → on_circle D c2 → 
                           intersect c1 c2) :
  ∃ (P : Point), ∀ (c1 c2 : Circle),
    on_circle A c1 → on_circle B c1 →
    on_circle C c2 → on_circle D c2 →
    P ∈ common_chord c1 c2 :=
sorry

end NUMINAMATH_CALUDE_common_chord_intersection_l3337_333754


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l3337_333799

theorem expression_simplification_and_evaluation (a : ℕ) 
  (h1 : 2 * a + 1 < 3 * a + 3) 
  (h2 : 2 / 3 * (a - 1) ≤ 1 / 2 * (a + 1 / 3)) 
  (h3 : a ≠ 0) 
  (h4 : a ≠ 1) 
  (h5 : a ≠ 2) : 
  ∃ (result : ℕ), 
    ((a + 1 - (4 * a - 5) / (a - 1)) / (1 / a - 1 / (a^2 - a)) = a * (a - 2)) ∧ 
    (result = a * (a - 2)) ∧ 
    (result = 3 ∨ result = 8 ∨ result = 15) :=
sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l3337_333799


namespace NUMINAMATH_CALUDE_order_inequality_l3337_333735

theorem order_inequality (x a b : ℝ) (h1 : x < a) (h2 : a < b) (h3 : b < 0) :
  x^2 > a*x ∧ a*x > a*b ∧ a*b > a^2 := by
  sorry

end NUMINAMATH_CALUDE_order_inequality_l3337_333735


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_of_squares_l3337_333792

theorem consecutive_integers_sum_of_squares (b : ℤ) : 
  (b - 1) * b * (b + 1) = 12 * (3 * b) + b^2 → 
  (b - 1)^2 + b^2 + (b + 1)^2 = 149 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_of_squares_l3337_333792


namespace NUMINAMATH_CALUDE_problem_solution_l3337_333748

def f (x : ℝ) : ℝ := 2 * x - 4
def g (x : ℝ) : ℝ := -x + 4

theorem problem_solution :
  (f 1 = -2 ∧ g 1 = 3) ∧
  (∀ x, f x * g x = -2 * x^2 + 12 * x - 16) ∧
  (Set.Icc 2 4 = {x | f x * g x = 0}) ∧
  (∀ x y, x < 3 ∧ y < 3 ∧ x < y → f x * g x < f y * g y) ∧
  (∀ x y, x > 3 ∧ y > 3 ∧ x < y → f x * g x > f y * g y) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3337_333748


namespace NUMINAMATH_CALUDE_wire_service_reporters_l3337_333703

theorem wire_service_reporters (total_reporters : ℝ) 
  (local_politics_reporters : ℝ) (politics_reporters : ℝ) :
  local_politics_reporters = 0.12 * total_reporters →
  local_politics_reporters = 0.6 * politics_reporters →
  total_reporters - politics_reporters = 0.8 * total_reporters :=
by
  sorry

end NUMINAMATH_CALUDE_wire_service_reporters_l3337_333703


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3337_333790

theorem geometric_sequence_ratio (a : ℕ → ℝ) (h_pos : ∀ n, a n > 0) 
  (h_geom : ∃ q : ℝ, q > 0 ∧ ∀ n, a (n + 1) = a n * q) 
  (h_arith : 2 * a 1 - a 2 = a 2 - a 3 / 2) :
  (a 2017 + a 2016) / (a 2015 + a 2014) = 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3337_333790


namespace NUMINAMATH_CALUDE_cube_plus_reciprocal_cube_l3337_333755

theorem cube_plus_reciprocal_cube (r : ℝ) (h : (r + 1/r)^2 = 5) :
  r^3 + 1/r^3 = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_cube_plus_reciprocal_cube_l3337_333755


namespace NUMINAMATH_CALUDE_larger_solution_quadratic_equation_l3337_333776

theorem larger_solution_quadratic_equation :
  ∃ (x y : ℝ), x ≠ y ∧ 
  x^2 - 13*x + 36 = 0 ∧
  y^2 - 13*y + 36 = 0 ∧
  (∀ z : ℝ, z^2 - 13*z + 36 = 0 → z = x ∨ z = y) ∧
  max x y = 9 := by
sorry

end NUMINAMATH_CALUDE_larger_solution_quadratic_equation_l3337_333776


namespace NUMINAMATH_CALUDE_product_75_360_trailing_zeros_l3337_333753

/-- The number of trailing zeros in a natural number -/
def trailingZeros (n : ℕ) : ℕ := sorry

/-- Theorem: The product of 75 and 360 has 3 trailing zeros -/
theorem product_75_360_trailing_zeros :
  trailingZeros (75 * 360) = 3 := by sorry

end NUMINAMATH_CALUDE_product_75_360_trailing_zeros_l3337_333753


namespace NUMINAMATH_CALUDE_max_intersected_edges_is_twelve_l3337_333720

/-- A regular 10-sided prism -/
structure RegularDecagonalPrism where
  -- We don't need to define the internal structure,
  -- as the problem doesn't require specific properties beyond it being a regular 10-sided prism

/-- A plane in 3D space -/
structure Plane where
  -- We don't need to define the internal structure of a plane

/-- The number of edges a plane intersects with a prism -/
def intersected_edges (prism : RegularDecagonalPrism) (plane : Plane) : ℕ :=
  sorry -- Definition not provided, as it's not explicitly given in the problem

/-- The maximum number of edges that can be intersected by any plane -/
def max_intersected_edges (prism : RegularDecagonalPrism) : ℕ :=
  sorry -- Definition not provided, as it's not explicitly given in the problem

/-- Theorem: The maximum number of edges of a regular 10-sided prism 
    that can be intersected by a plane is 12 -/
theorem max_intersected_edges_is_twelve (prism : RegularDecagonalPrism) :
  max_intersected_edges prism = 12 := by
  sorry

-- The proof is omitted as per instructions

end NUMINAMATH_CALUDE_max_intersected_edges_is_twelve_l3337_333720


namespace NUMINAMATH_CALUDE_largest_number_hcf_lcm_l3337_333775

theorem largest_number_hcf_lcm (a b : ℕ+) : 
  (Nat.gcd a b = 62) →
  (∃ (x y : ℕ+), x = 11 ∧ y = 12 ∧ Nat.lcm a b = 62 * x * y) →
  max a b = 744 := by
sorry

end NUMINAMATH_CALUDE_largest_number_hcf_lcm_l3337_333775


namespace NUMINAMATH_CALUDE_no_such_function_exists_l3337_333705

theorem no_such_function_exists : 
  ∀ f : ℝ → ℝ, ∃ x y : ℝ, (f x + f y) / 2 < f ((x + y) / 2) + |x - y| := by
  sorry

end NUMINAMATH_CALUDE_no_such_function_exists_l3337_333705


namespace NUMINAMATH_CALUDE_largest_three_digit_product_l3337_333782

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop := sorry

theorem largest_three_digit_product : ∃ (m x y : ℕ),
  100 ≤ m ∧ m < 1000 ∧
  isPrime x ∧ isPrime y ∧ isPrime (10 * x - y) ∧
  x < 10 ∧ y < 10 ∧ x ≠ y ∧
  m = x * y * (10 * x - y) ∧
  ∀ (m' x' y' : ℕ),
    100 ≤ m' ∧ m' < 1000 →
    isPrime x' ∧ isPrime y' ∧ isPrime (10 * x' - y') →
    x' < 10 ∧ y' < 10 ∧ x' ≠ y' →
    m' = x' * y' * (10 * x' - y') →
    m' ≤ m ∧
  m = 705 := by
  sorry

end NUMINAMATH_CALUDE_largest_three_digit_product_l3337_333782


namespace NUMINAMATH_CALUDE_problem_one_problem_two_l3337_333747

-- Problem 1
theorem problem_one : (-1/3)⁻¹ + (Real.pi - 3.14)^0 = -2 := by sorry

-- Problem 2
theorem problem_two : ∀ x : ℝ, (2*x - 3)^2 - 2*x*(2*x - 6) = 9 := by sorry

end NUMINAMATH_CALUDE_problem_one_problem_two_l3337_333747


namespace NUMINAMATH_CALUDE_ball_box_problem_l3337_333774

/-- The number of ways to put n different balls into m different boxes -/
def ways_to_put_balls (n m : ℕ) : ℕ := m^n

/-- The number of ways to put n different balls into m different boxes with exactly k boxes left empty -/
def ways_with_empty_boxes (n m k : ℕ) : ℕ := sorry

theorem ball_box_problem :
  (ways_to_put_balls 4 4 = 256) ∧
  (ways_with_empty_boxes 4 4 1 = 144) ∧
  (ways_with_empty_boxes 4 4 2 = 84) := by sorry

end NUMINAMATH_CALUDE_ball_box_problem_l3337_333774


namespace NUMINAMATH_CALUDE_min_sum_xy_l3337_333760

theorem min_sum_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 9/y = 1) : x + y ≥ 16 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_xy_l3337_333760


namespace NUMINAMATH_CALUDE_g_equation_holds_l3337_333751

-- Define the polynomial g(x)
noncomputable def g (x : ℝ) : ℝ := -4*x^5 + 4*x^3 - 5*x^2 + 2*x + 4

-- State the theorem
theorem g_equation_holds (x : ℝ) : 4*x^5 + 3*x^3 - 2*x + g x = 7*x^3 - 5*x^2 + 4 := by
  sorry

end NUMINAMATH_CALUDE_g_equation_holds_l3337_333751


namespace NUMINAMATH_CALUDE_simple_interest_rate_calculation_l3337_333780

theorem simple_interest_rate_calculation (P A T : ℝ) (h1 : P = 750) (h2 : A = 1050) (h3 : T = 5) :
  let SI := A - P
  let R := (SI * 100) / (P * T)
  R = 8 := by sorry

end NUMINAMATH_CALUDE_simple_interest_rate_calculation_l3337_333780


namespace NUMINAMATH_CALUDE_stickers_per_page_l3337_333744

theorem stickers_per_page (total_stickers : ℕ) (total_pages : ℕ) 
  (h1 : total_stickers = 220) 
  (h2 : total_pages = 22) 
  (h3 : total_stickers > 0) 
  (h4 : total_pages > 0) : 
  total_stickers / total_pages = 10 :=
sorry

end NUMINAMATH_CALUDE_stickers_per_page_l3337_333744


namespace NUMINAMATH_CALUDE_average_speed_inequality_l3337_333770

theorem average_speed_inequality (a b v : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hab : a < b) 
  (hv : v = (2 * a * b) / (a + b)) : 
  a < v ∧ v < Real.sqrt (a * b) := by
  sorry

end NUMINAMATH_CALUDE_average_speed_inequality_l3337_333770


namespace NUMINAMATH_CALUDE_distinct_differences_sequence_length_l3337_333731

def is_valid_n (n : ℕ) : Prop :=
  ∃ k : ℕ+, (n = 4 * k ∨ n = 4 * k - 1)

theorem distinct_differences_sequence_length {n : ℕ} (h_n : n ≥ 3) :
  (∃ (a : ℕ → ℝ), (∀ i j : Fin n, i ≠ j → |a i - a (i + 1)| ≠ |a j - a (j + 1)|)) →
  is_valid_n n :=
sorry

end NUMINAMATH_CALUDE_distinct_differences_sequence_length_l3337_333731


namespace NUMINAMATH_CALUDE_symmetric_circle_l3337_333702

/-- Given a point P(a, b) symmetric to line l with symmetric point P'(b + 1, a - 1),
    and a circle C with equation x^2 + y^2 - 6x - 2y = 0,
    prove that the equation of the circle C' symmetric to C with respect to line l
    is (x - 2)^2 + (y - 2)^2 = 10 -/
theorem symmetric_circle (a b : ℝ) :
  let P : ℝ × ℝ := (a, b)
  let P' : ℝ × ℝ := (b + 1, a - 1)
  let C (x y : ℝ) := x^2 + y^2 - 6*x - 2*y = 0
  let C' (x y : ℝ) := (x - 2)^2 + (y - 2)^2 = 10
  (∀ x y, C x y ↔ C' y x) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_circle_l3337_333702


namespace NUMINAMATH_CALUDE_smallest_divisible_by_15_and_24_l3337_333778

theorem smallest_divisible_by_15_and_24 : ∃ n : ℕ, (n > 0 ∧ n % 15 = 0 ∧ n % 24 = 0 ∧ ∀ m : ℕ, (m > 0 ∧ m % 15 = 0 ∧ m % 24 = 0) → n ≤ m) ∧ n = 120 := by
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_15_and_24_l3337_333778


namespace NUMINAMATH_CALUDE_company_problem_solution_l3337_333791

def company_problem (total_employees : ℕ) 
                    (clerical_fraction technical_fraction managerial_fraction : ℚ)
                    (clerical_reduction technical_reduction managerial_reduction : ℚ) : ℚ :=
  let initial_clerical := (clerical_fraction * total_employees : ℚ)
  let initial_technical := (technical_fraction * total_employees : ℚ)
  let initial_managerial := (managerial_fraction * total_employees : ℚ)
  
  let remaining_clerical := initial_clerical * (1 - clerical_reduction)
  let remaining_technical := initial_technical * (1 - technical_reduction)
  let remaining_managerial := initial_managerial * (1 - managerial_reduction)
  
  let total_remaining := remaining_clerical + remaining_technical + remaining_managerial
  
  remaining_clerical / total_remaining

theorem company_problem_solution :
  let result := company_problem 5000 (1/5) (2/5) (2/5) (1/3) (1/4) (1/5)
  ∃ (ε : ℚ), abs (result - 177/1000) < ε ∧ ε < 1/1000 :=
by
  sorry

end NUMINAMATH_CALUDE_company_problem_solution_l3337_333791


namespace NUMINAMATH_CALUDE_equation_solution_l3337_333734

theorem equation_solution : 
  ∃ x : ℚ, (1 / 6 + 7 / x = 15 / x + 1 / 15 + 2) ∧ (x = -80 / 19) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3337_333734


namespace NUMINAMATH_CALUDE_gcd_a_squared_plus_9a_plus_24_and_a_plus_4_l3337_333726

theorem gcd_a_squared_plus_9a_plus_24_and_a_plus_4 (a : ℤ) (h : ∃ k : ℤ, a = 1428 * k) :
  Nat.gcd (Int.natAbs (a^2 + 9*a + 24)) (Int.natAbs (a + 4)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_gcd_a_squared_plus_9a_plus_24_and_a_plus_4_l3337_333726


namespace NUMINAMATH_CALUDE_subtraction_of_decimals_l3337_333742

theorem subtraction_of_decimals : 3.75 - 2.18 = 1.57 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_decimals_l3337_333742


namespace NUMINAMATH_CALUDE_sally_lost_cards_l3337_333725

def pokemon_cards_lost (initial : ℕ) (received : ℕ) (current : ℕ) : ℕ :=
  initial + received - current

theorem sally_lost_cards (initial : ℕ) (received : ℕ) (current : ℕ)
  (h1 : initial = 27)
  (h2 : received = 41)
  (h3 : current = 48) :
  pokemon_cards_lost initial received current = 20 := by
  sorry

end NUMINAMATH_CALUDE_sally_lost_cards_l3337_333725


namespace NUMINAMATH_CALUDE_statement_b_statement_c_not_statement_a_not_statement_d_l3337_333757

-- Statement B
theorem statement_b (a b : ℝ) : a > b → a - 1 > b - 2 := by sorry

-- Statement C
theorem statement_c (a b c : ℝ) (h : c ≠ 0) : a / c^2 > b / c^2 → a > b := by sorry

-- Disproof of Statement A
theorem not_statement_a : ¬ (∀ a b c : ℝ, a > b → a * c^2 > b * c^2) := by sorry

-- Disproof of Statement D
theorem not_statement_d : ¬ (∀ a b : ℝ, a > b → a^2 > b^2) := by sorry

end NUMINAMATH_CALUDE_statement_b_statement_c_not_statement_a_not_statement_d_l3337_333757


namespace NUMINAMATH_CALUDE_hike_length_is_48_l3337_333767

/-- Represents the length of a multi-day hike --/
structure HikeLength where
  day1 : ℝ
  day2 : ℝ
  day3 : ℝ
  day4 : ℝ
  day5 : ℝ

/-- The conditions of the hike as described in the problem --/
def hike_conditions (h : HikeLength) : Prop :=
  h.day1 + h.day2 + h.day3 = 34 ∧
  (h.day2 + h.day3) / 2 = 12 ∧
  h.day3 + h.day4 + h.day5 = 40 ∧
  h.day1 + h.day3 + h.day5 = 38 ∧
  h.day4 = 14

/-- The theorem stating that given the conditions, the total length of the trail is 48 miles --/
theorem hike_length_is_48 (h : HikeLength) (hc : hike_conditions h) :
  h.day1 + h.day2 + h.day3 + h.day4 + h.day5 = 48 := by
  sorry


end NUMINAMATH_CALUDE_hike_length_is_48_l3337_333767


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3337_333772

/-- Geometric sequence with common ratio greater than 1 -/
def GeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  q > 1 ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) 
  (h_geo : GeometricSequence a q)
  (h_sum : a 1 + a 4 = 9)
  (h_prod : a 2 * a 3 = 8) :
  (a 2015 + a 2016) / (a 2013 + a 2014) = 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3337_333772


namespace NUMINAMATH_CALUDE_extremum_condition_l3337_333740

/-- A function f: ℝ → ℝ has an extremum at x₀ if f(x₀) is either a maximum or minimum value of f. -/
def HasExtremumAt (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∀ x, f x ≤ f x₀ ∨ ∀ x, f x ≥ f x₀

/-- Theorem: For a differentiable function f: ℝ → ℝ, f'(x₀) = 0 is a necessary but not sufficient
    condition for f(x₀) to be an extremum of f(x). -/
theorem extremum_condition (f : ℝ → ℝ) (hf : Differentiable ℝ f) :
  (∀ x₀ : ℝ, HasExtremumAt f x₀ → deriv f x₀ = 0) ∧
  ¬(∀ x₀ : ℝ, deriv f x₀ = 0 → HasExtremumAt f x₀) :=
sorry

end NUMINAMATH_CALUDE_extremum_condition_l3337_333740


namespace NUMINAMATH_CALUDE_marcos_strawberries_weight_l3337_333732

/-- Given the initial total weight of strawberries collected by Marco and his dad,
    the additional weight of strawberries found by dad, and dad's final weight of strawberries,
    prove that Marco's strawberries weigh 6 pounds. -/
theorem marcos_strawberries_weight
  (initial_total : ℕ)
  (dads_additional : ℕ)
  (dads_final : ℕ)
  (h1 : initial_total = 22)
  (h2 : dads_additional = 30)
  (h3 : dads_final = 16) :
  initial_total - dads_final = 6 :=
by sorry

end NUMINAMATH_CALUDE_marcos_strawberries_weight_l3337_333732


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3337_333723

theorem inequality_solution_set (x : ℝ) : x + 8 < 4*x - 1 ↔ x > 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3337_333723


namespace NUMINAMATH_CALUDE_anna_money_left_l3337_333707

def original_amount : ℚ := 32
def spent_fraction : ℚ := 1/4

theorem anna_money_left : 
  (1 - spent_fraction) * original_amount = 24 := by
  sorry

end NUMINAMATH_CALUDE_anna_money_left_l3337_333707


namespace NUMINAMATH_CALUDE_function_inequality_l3337_333768

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the condition f''(x) < f(x) for all x ∈ ℝ
variable (h : ∀ x : ℝ, (deriv (deriv f)) x < f x)

-- State the theorem
theorem function_inequality (f : ℝ → ℝ) (h : ∀ x : ℝ, (deriv (deriv f)) x < f x) :
  f 2 < Real.exp 2 * f 0 ∧ f 2001 < Real.exp 2001 * f 0 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l3337_333768


namespace NUMINAMATH_CALUDE_basketball_distribution_l3337_333793

theorem basketball_distribution (total_basketballs : ℕ) (basketballs_per_class : ℕ) (num_classes : ℕ) : 
  total_basketballs = 54 → 
  basketballs_per_class = 7 → 
  total_basketballs = num_classes * basketballs_per_class →
  num_classes = 7 := by
sorry

end NUMINAMATH_CALUDE_basketball_distribution_l3337_333793


namespace NUMINAMATH_CALUDE_a_range_l3337_333715

-- Define the propositions p and q
def p (x : ℝ) : Prop := |x - 3/4| ≤ 1/4
def q (x a : ℝ) : Prop := (x - a) * (x - a - 1) ≤ 0

-- Define the range of x for p
def p_range (x : ℝ) : Prop := 1/2 ≤ x ∧ x ≤ 1

-- Define the range of x for q
def q_range (x a : ℝ) : Prop := a ≤ x ∧ x ≤ a + 1

-- Define the sufficient but not necessary condition
def sufficient_not_necessary (a : ℝ) : Prop :=
  (∀ x, p_range x → q_range x a) ∧
  ¬(∀ x, q_range x a → p_range x)

-- State the theorem
theorem a_range :
  ∀ a : ℝ, sufficient_not_necessary a ↔ (0 ≤ a ∧ a ≤ 1/2) :=
sorry

end NUMINAMATH_CALUDE_a_range_l3337_333715


namespace NUMINAMATH_CALUDE_wendy_albums_l3337_333795

theorem wendy_albums (total_pictures : ℕ) (pictures_in_one_album : ℕ) (pictures_per_album : ℕ) 
  (h1 : total_pictures = 45)
  (h2 : pictures_in_one_album = 27)
  (h3 : pictures_per_album = 2) :
  (total_pictures - pictures_in_one_album) / pictures_per_album = 9 := by
  sorry

end NUMINAMATH_CALUDE_wendy_albums_l3337_333795


namespace NUMINAMATH_CALUDE_clothing_sale_theorem_l3337_333749

/-- The marked price of an item of clothing --/
def marked_price : ℝ := 300

/-- The loss per item when sold at 40% of marked price --/
def loss_at_40_percent : ℝ := 30

/-- The profit per item when sold at 70% of marked price --/
def profit_at_70_percent : ℝ := 60

/-- The maximum discount percentage that can be offered without incurring a loss --/
def max_discount_percent : ℝ := 50

theorem clothing_sale_theorem :
  (0.4 * marked_price - loss_at_40_percent = 0.7 * marked_price + profit_at_70_percent) ∧
  (max_discount_percent / 100 * marked_price = 0.4 * marked_price + loss_at_40_percent) := by
  sorry

end NUMINAMATH_CALUDE_clothing_sale_theorem_l3337_333749


namespace NUMINAMATH_CALUDE_rectangle_area_l3337_333771

theorem rectangle_area (x y : ℝ) 
  (h1 : (x + 3.5) * (y - 1.5) = x * y)
  (h2 : (x - 3.5) * (y + 2.5) = x * y)
  (h3 : 2 * (x + 3.5) + 2 * (y - 3.5) = 2 * x + 2 * y) :
  x * y = 196 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_l3337_333771


namespace NUMINAMATH_CALUDE_tangent_equation_solution_l3337_333746

theorem tangent_equation_solution (x : Real) :
  (Real.tan x * Real.tan (20 * π / 180) + 
   Real.tan (20 * π / 180) * Real.tan (40 * π / 180) + 
   Real.tan (40 * π / 180) * Real.tan x = 1) ↔
  (∃ k : ℤ, x = (30 + 180 * k) * π / 180) :=
by sorry

end NUMINAMATH_CALUDE_tangent_equation_solution_l3337_333746


namespace NUMINAMATH_CALUDE_curve_circle_intersection_l3337_333773

-- Define the curve
def curve (x y : ℝ) : Prop := y = x^2 - 4*x + 3

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + (y - 2)^2 = 5

-- Define the line
def line (x y : ℝ) (m : ℝ) : Prop := x + y + m = 0

-- Define perpendicularity of OA and OB
def perpendicular (xA yA xB yB : ℝ) : Prop := xA * xB + yA * yB = 0

-- Main theorem
theorem curve_circle_intersection (m : ℝ) :
  ∃ (x1 y1 x2 y2 : ℝ),
    curve 0 3 ∧ curve 1 0 ∧ curve 3 0 ∧  -- Curve intersects axes at (0,3), (1,0), and (3,0)
    circle_C 0 3 ∧ circle_C 1 0 ∧ circle_C 3 0 ∧  -- These points lie on circle C
    circle_C x1 y1 ∧ circle_C x2 y2 ∧  -- A and B lie on circle C
    line x1 y1 m ∧ line x2 y2 m ∧  -- A and B lie on the line
    perpendicular x1 y1 x2 y2  -- OA is perpendicular to OB
  →
    (m = -1 ∨ m = -3) :=
sorry

end NUMINAMATH_CALUDE_curve_circle_intersection_l3337_333773


namespace NUMINAMATH_CALUDE_curve_self_intersection_l3337_333737

theorem curve_self_intersection :
  ∃ (a b : ℝ), a ≠ b ∧
    a^2 - 4 = b^2 - 4 ∧
    a^3 - 6*a + 4 = b^3 - 6*b + 4 ∧
    (a^2 - 4 = 2 ∧ a^3 - 6*a + 4 = 4) :=
sorry

end NUMINAMATH_CALUDE_curve_self_intersection_l3337_333737


namespace NUMINAMATH_CALUDE_michelle_candy_sugar_l3337_333745

/-- The total grams of sugar in Michelle's candy purchase -/
def total_sugar (num_bars : ℕ) (sugar_per_bar : ℕ) (lollipop_sugar : ℕ) : ℕ :=
  num_bars * sugar_per_bar + lollipop_sugar

/-- Theorem: The total sugar in Michelle's candy purchase is 177 grams -/
theorem michelle_candy_sugar :
  total_sugar 14 10 37 = 177 := by sorry

end NUMINAMATH_CALUDE_michelle_candy_sugar_l3337_333745


namespace NUMINAMATH_CALUDE_range_of_a_l3337_333716

-- Define the conditions
def p (x : ℝ) : Prop := |x - 2| < 3
def q (x a : ℝ) : Prop := 0 < x ∧ x < a

-- State the theorem
theorem range_of_a :
  (∀ x a : ℝ, q x a → p x) ∧ 
  (∃ x a : ℝ, p x ∧ ¬(q x a)) →
  ∀ a : ℝ, (∃ x : ℝ, q x a) ↔ (0 < a ∧ a ≤ 5) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3337_333716


namespace NUMINAMATH_CALUDE_number_of_dimes_l3337_333765

-- Define the total number of coins
def total_coins : ℕ := 50

-- Define the total value in cents
def total_value : ℕ := 430

-- Define the value of a dime in cents
def dime_value : ℕ := 10

-- Define the value of a nickel in cents
def nickel_value : ℕ := 5

-- Theorem statement
theorem number_of_dimes :
  ∃ (d n : ℕ), d + n = total_coins ∧
               d * dime_value + n * nickel_value = total_value ∧
               d = 36 := by
  sorry

end NUMINAMATH_CALUDE_number_of_dimes_l3337_333765


namespace NUMINAMATH_CALUDE_all_players_odd_sum_probability_l3337_333743

def number_of_tiles : ℕ := 15
def number_of_players : ℕ := 5
def tiles_per_player : ℕ := 3

def probability_all_odd_sum : ℚ :=
  480 / 19019

theorem all_players_odd_sum_probability :
  (number_of_tiles = 15) →
  (number_of_players = 5) →
  (tiles_per_player = 3) →
  probability_all_odd_sum = 480 / 19019 :=
by sorry

end NUMINAMATH_CALUDE_all_players_odd_sum_probability_l3337_333743


namespace NUMINAMATH_CALUDE_only_zero_solution_l3337_333710

theorem only_zero_solution (n : ℕ) : 
  (∃ k : ℤ, (30 * n + 2) = k * (12 * n + 1)) ↔ n = 0 :=
by sorry

end NUMINAMATH_CALUDE_only_zero_solution_l3337_333710


namespace NUMINAMATH_CALUDE_distance_difference_l3337_333719

/-- The width of a street in Simplifiedtown -/
def street_width : ℝ := 30

/-- The length of one side of a square block in Simplifiedtown -/
def block_side_length : ℝ := 400

/-- The distance Sarah runs from the block's inner edge -/
def sarah_distance : ℝ := 400

/-- The distance Maude runs from the block's inner edge -/
def maude_distance : ℝ := block_side_length + street_width

/-- The theorem stating the difference in distance run by Maude and Sarah -/
theorem distance_difference :
  4 * maude_distance - 4 * sarah_distance = 120 :=
sorry

end NUMINAMATH_CALUDE_distance_difference_l3337_333719


namespace NUMINAMATH_CALUDE_function_properties_l3337_333728

/-- Given a function f(x) = ax - bx^2 where a and b are positive real numbers,
    this theorem states two properties:
    1. If f(x) ≤ 1 for all real x, then a ≤ 2√b.
    2. When b > 1, for x in [0, 1], |f(x)| ≤ 1 if and only if b - 1 ≤ a ≤ 2√b. -/
theorem function_properties (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let f := fun x : ℝ => a * x - b * x^2
  (∀ x, f x ≤ 1) → a ≤ 2 * Real.sqrt b ∧
  (b > 1 → (∀ x ∈ Set.Icc 0 1, |f x| ≤ 1) ↔ b - 1 ≤ a ∧ a ≤ 2 * Real.sqrt b) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l3337_333728


namespace NUMINAMATH_CALUDE_base10_512_to_base5_l3337_333766

/-- Converts a base-10 number to its base-5 representation -/
def toBase5 (n : ℕ) : List ℕ :=
  sorry

theorem base10_512_to_base5 :
  toBase5 512 = [4, 0, 2, 2] :=
sorry

end NUMINAMATH_CALUDE_base10_512_to_base5_l3337_333766


namespace NUMINAMATH_CALUDE_max_area_rectangle_with_perimeter_52_l3337_333722

/-- The maximum area of a rectangle with a perimeter of 52 centimeters is 169 square centimeters. -/
theorem max_area_rectangle_with_perimeter_52 :
  ∀ (length width : ℝ),
  length > 0 → width > 0 →
  2 * (length + width) = 52 →
  length * width ≤ 169 := by
sorry

end NUMINAMATH_CALUDE_max_area_rectangle_with_perimeter_52_l3337_333722


namespace NUMINAMATH_CALUDE_triangle_properties_l3337_333787

/-- Given a triangle ABC with angle C = π/4 and the relation 2sin²A - 1 = sin²B,
    prove that tan B = 2 and if side b = 1, the area of the triangle is 3/8 -/
theorem triangle_properties (A B C : Real) (a b c : Real) :
  C = π/4 →
  2 * Real.sin A ^ 2 - 1 = Real.sin B ^ 2 →
  Real.tan B = 2 ∧
  (b = 1 → (1/2) * a * b * Real.sin C = 3/8) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l3337_333787


namespace NUMINAMATH_CALUDE_proportion_equality_l3337_333788

theorem proportion_equality (x : ℝ) : (0.75 / x = 3 / 8) → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_proportion_equality_l3337_333788


namespace NUMINAMATH_CALUDE_unique_solution_equation_l3337_333798

theorem unique_solution_equation : ∃! x : ℝ, (28 + 48 / x) * x = 1980 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_equation_l3337_333798


namespace NUMINAMATH_CALUDE_tangent_lines_to_circle_l3337_333709

/-- A line in 2D space represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A circle in 2D space represented by (x - h)² + (y - k)² = r² -/
structure Circle where
  h : ℝ
  k : ℝ
  r : ℝ

/-- Check if a point (x, y) lies on a line -/
def Line.contains (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y + l.c = 0

/-- Check if a line is tangent to a circle -/
def isTangent (l : Line) (c : Circle) : Prop :=
  ∃ (x y : ℝ), l.contains x y ∧ (x - c.h)^2 + (y - c.k)^2 = c.r^2 ∧
  ∀ (x' y' : ℝ), l.contains x' y' → (x' - c.h)^2 + (y' - c.k)^2 ≥ c.r^2

theorem tangent_lines_to_circle (c : Circle) (p : ℝ × ℝ) :
  c.h = 0 ∧ c.k = 0 ∧ c.r = 3 ∧ p = (3, 1) →
  ∃ (l1 l2 : Line),
    (l1.a = 4 ∧ l1.b = 3 ∧ l1.c = -15) ∧
    (l2.a = 1 ∧ l2.b = 0 ∧ l2.c = -3) ∧
    l1.contains p.1 p.2 ∧
    l2.contains p.1 p.2 ∧
    isTangent l1 c ∧
    isTangent l2 c ∧
    ∀ (l : Line), l.contains p.1 p.2 ∧ isTangent l c → l = l1 ∨ l = l2 :=
by sorry

end NUMINAMATH_CALUDE_tangent_lines_to_circle_l3337_333709


namespace NUMINAMATH_CALUDE_operation_equations_l3337_333796

theorem operation_equations :
  (37.3 / (1/2) = 74 + 3/5) ∧
  (33/40 * 10/11 = 0.75) ∧
  (0.45 - 1/20 = 2/5) ∧
  (0.375 + 1/40 = 0.4) := by
sorry

end NUMINAMATH_CALUDE_operation_equations_l3337_333796


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l3337_333786

theorem quadratic_expression_value (x y : ℝ) 
  (eq1 : 3 * x + 2 * y = 7) 
  (eq2 : 2 * x + 3 * y = 8) : 
  13 * x^2 + 22 * x * y + 13 * y^2 = 113 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l3337_333786


namespace NUMINAMATH_CALUDE_smallest_prime_after_six_nonprimes_l3337_333752

/-- A function that returns true if a number is prime, false otherwise -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

/-- A function that returns true if there are six consecutive nonprime numbers before n -/
def sixConsecutiveNonprimes (n : ℕ) : Prop :=
  ∀ k : ℕ, n - 6 ≤ k → k < n → ¬(isPrime k)

/-- Theorem stating that 89 is the smallest prime number after six consecutive nonprimes -/
theorem smallest_prime_after_six_nonprimes :
  isPrime 89 ∧ sixConsecutiveNonprimes 89 ∧
  ∀ m : ℕ, m < 89 → ¬(isPrime m ∧ sixConsecutiveNonprimes m) :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_after_six_nonprimes_l3337_333752


namespace NUMINAMATH_CALUDE_constant_term_is_135_l3337_333701

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the constant term in the expansion
def constant_term (x : ℝ) : ℝ :=
  binomial 6 2 * 3^2

-- Theorem statement
theorem constant_term_is_135 :
  constant_term = 135 := by sorry

end NUMINAMATH_CALUDE_constant_term_is_135_l3337_333701


namespace NUMINAMATH_CALUDE_f_is_even_l3337_333724

-- Define g as an even function
def g_even (g : ℝ → ℝ) : Prop := ∀ x, g x = g (-x)

-- Define f in terms of g
def f (g : ℝ → ℝ) (x : ℝ) : ℝ := |g (x^4)|

-- Theorem statement
theorem f_is_even (g : ℝ → ℝ) (h : g_even g) : ∀ x, f g x = f g (-x) := by
  sorry

end NUMINAMATH_CALUDE_f_is_even_l3337_333724


namespace NUMINAMATH_CALUDE_online_price_theorem_l3337_333727

/-- The price that the buyer observes online for a product sold by a distributor through an online store -/
theorem online_price_theorem (cost : ℝ) (commission_rate : ℝ) (profit_rate : ℝ) 
  (h_cost : cost = 19)
  (h_commission : commission_rate = 0.2)
  (h_profit : profit_rate = 0.2) :
  let distributor_price := cost * (1 + profit_rate)
  let online_price := distributor_price / (1 - commission_rate)
  online_price = 28.5 := by
sorry

end NUMINAMATH_CALUDE_online_price_theorem_l3337_333727


namespace NUMINAMATH_CALUDE_aunt_wang_lilies_l3337_333789

theorem aunt_wang_lilies (rose_cost lily_cost roses_bought total_spent : ℕ) : 
  rose_cost = 5 →
  lily_cost = 9 →
  roses_bought = 2 →
  total_spent = 55 →
  (total_spent - rose_cost * roses_bought) / lily_cost = 5 := by
  sorry

end NUMINAMATH_CALUDE_aunt_wang_lilies_l3337_333789


namespace NUMINAMATH_CALUDE_power_of_product_l3337_333758

theorem power_of_product (a : ℝ) : (2 * a^2)^3 = 8 * a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l3337_333758


namespace NUMINAMATH_CALUDE_smallest_n_divisible_by_24_and_1024_l3337_333759

theorem smallest_n_divisible_by_24_and_1024 :
  ∃ n : ℕ+, (∀ m : ℕ+, m < n → (¬(24 ∣ m^2) ∨ ¬(1024 ∣ m^3))) ∧
  (24 ∣ n^2) ∧ (1024 ∣ n^3) ∧ n = 48 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_divisible_by_24_and_1024_l3337_333759


namespace NUMINAMATH_CALUDE_cistern_width_is_four_l3337_333781

/-- Represents the dimensions and properties of a cistern --/
structure Cistern where
  length : ℝ
  width : ℝ
  depth : ℝ
  wetSurfaceArea : ℝ

/-- Calculates the total wet surface area of a cistern --/
def totalWetSurfaceArea (c : Cistern) : ℝ :=
  c.length * c.width + 2 * c.length * c.depth + 2 * c.width * c.depth

/-- Theorem stating that a cistern with given dimensions has a width of 4 meters --/
theorem cistern_width_is_four :
  ∃ (c : Cistern),
    c.length = 6 ∧
    c.depth = 1.25 ∧
    c.wetSurfaceArea = 49 ∧
    totalWetSurfaceArea c = c.wetSurfaceArea ∧
    c.width = 4 := by
  sorry


end NUMINAMATH_CALUDE_cistern_width_is_four_l3337_333781


namespace NUMINAMATH_CALUDE_imaginary_sum_zero_l3337_333718

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem imaginary_sum_zero :
  i^13 + i^18 + i^23 + i^28 + i^33 + i^38 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_imaginary_sum_zero_l3337_333718


namespace NUMINAMATH_CALUDE_cycle_transactions_result_l3337_333777

/-- Calculates the final amount after three cycle transactions -/
def final_amount (initial_cost : ℝ) (loss1 gain2 gain3 : ℝ) : ℝ :=
  let selling_price1 := initial_cost * (1 - loss1)
  let selling_price2 := selling_price1 * (1 + gain2)
  selling_price2 * (1 + gain3)

/-- Theorem stating the final amount after three cycle transactions -/
theorem cycle_transactions_result :
  final_amount 1600 0.12 0.15 0.20 = 1943.04 := by
  sorry

#eval final_amount 1600 0.12 0.15 0.20

end NUMINAMATH_CALUDE_cycle_transactions_result_l3337_333777


namespace NUMINAMATH_CALUDE_brothers_savings_l3337_333779

def isabelle_ticket_cost : ℕ := 20
def brother_ticket_cost : ℕ := 10
def number_of_brothers : ℕ := 2
def isabelle_savings : ℕ := 5
def work_weeks : ℕ := 10
def weekly_earnings : ℕ := 3

def total_ticket_cost : ℕ := isabelle_ticket_cost + number_of_brothers * brother_ticket_cost

def isabelle_total_earnings : ℕ := isabelle_savings + work_weeks * weekly_earnings

theorem brothers_savings : 
  total_ticket_cost - isabelle_total_earnings = 5 := by sorry

end NUMINAMATH_CALUDE_brothers_savings_l3337_333779


namespace NUMINAMATH_CALUDE_cross_area_is_two_l3337_333704

/-- Represents a point in a 2D grid --/
structure GridPoint where
  x : ℚ
  y : ℚ

/-- Represents a triangle in the grid --/
structure Triangle where
  v1 : GridPoint
  v2 : GridPoint
  v3 : GridPoint

/-- The center point of the 4x4 grid --/
def gridCenter : GridPoint := { x := 2, y := 2 }

/-- A function to create a midpoint on the grid edge --/
def gridEdgeMidpoint (x y : ℚ) : GridPoint := { x := x, y := y }

/-- The four triangles forming the cross shape --/
def crossTriangles : List Triangle := [
  { v1 := gridCenter, v2 := gridEdgeMidpoint 0 2, v3 := gridEdgeMidpoint 2 0 },
  { v1 := gridCenter, v2 := gridEdgeMidpoint 2 4, v3 := gridEdgeMidpoint 4 2 },
  { v1 := gridCenter, v2 := gridEdgeMidpoint 0 2, v3 := gridEdgeMidpoint 2 4 },
  { v1 := gridCenter, v2 := gridEdgeMidpoint 2 0, v3 := gridEdgeMidpoint 4 2 }
]

/-- Calculate the area of a single triangle --/
def triangleArea (t : Triangle) : ℚ := 0.5

/-- Calculate the total area of the cross shape --/
def crossArea : ℚ := (crossTriangles.map triangleArea).sum

/-- The theorem stating that the area of the cross shape is 2 --/
theorem cross_area_is_two : crossArea = 2 := by sorry

end NUMINAMATH_CALUDE_cross_area_is_two_l3337_333704


namespace NUMINAMATH_CALUDE_sum_with_radical_conjugate_l3337_333729

theorem sum_with_radical_conjugate : 
  let x : ℝ := 10 - Real.sqrt 2018
  let y : ℝ := 10 + Real.sqrt 2018  -- Definition of radical conjugate
  x + y = 20 := by
sorry

end NUMINAMATH_CALUDE_sum_with_radical_conjugate_l3337_333729


namespace NUMINAMATH_CALUDE_games_purchased_l3337_333784

theorem games_purchased (total_income : ℕ) (expense : ℕ) (game_cost : ℕ) :
  total_income = 69 →
  expense = 24 →
  game_cost = 5 →
  (total_income - expense) / game_cost = 9 :=
by sorry

end NUMINAMATH_CALUDE_games_purchased_l3337_333784


namespace NUMINAMATH_CALUDE_sum_A_B_equals_negative_five_halves_l3337_333761

theorem sum_A_B_equals_negative_five_halves (A B : ℚ) :
  (∀ x : ℚ, x ≠ 4 ∧ x ≠ 5 →
    (B * x - 15) / (x^2 - 9*x + 20) = A / (x - 4) + 4 / (x - 5)) →
  A + B = -5/2 := by
sorry

end NUMINAMATH_CALUDE_sum_A_B_equals_negative_five_halves_l3337_333761


namespace NUMINAMATH_CALUDE_candidates_per_state_l3337_333721

theorem candidates_per_state : 
  ∀ (x : ℕ), 
    (x * 6 / 100 : ℚ) + 80 = (x * 7 / 100 : ℚ) → 
    x = 8000 := by
  sorry

end NUMINAMATH_CALUDE_candidates_per_state_l3337_333721


namespace NUMINAMATH_CALUDE_sugar_consumption_problem_l3337_333785

theorem sugar_consumption_problem (initial_price : ℝ) (initial_consumption : ℝ) :
  let price_increase_factor := 1.32
  let expenditure_increase_factor := 1.10
  let new_consumption := 25

  (price_increase_factor * initial_price * new_consumption = 
   expenditure_increase_factor * initial_price * initial_consumption) →
  initial_consumption = 75 := by
sorry

end NUMINAMATH_CALUDE_sugar_consumption_problem_l3337_333785


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l3337_333708

theorem system_of_equations_solution :
  ∃ (x y : ℚ), (3 * x - 4 * y = -6) ∧ (6 * x - 5 * y = 9) ∧ (x = 22 / 3) ∧ (y = 7) := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l3337_333708


namespace NUMINAMATH_CALUDE_civilisation_meaning_l3337_333794

/-- The meaning of a word -/
def word_meaning (word : String) : String :=
  sorry

/-- Theorem: The meaning of "civilisation (n.)" is "civilization" -/
theorem civilisation_meaning : word_meaning "civilisation (n.)" = "civilization" :=
  sorry

end NUMINAMATH_CALUDE_civilisation_meaning_l3337_333794


namespace NUMINAMATH_CALUDE_aladdin_gold_bars_l3337_333797

theorem aladdin_gold_bars (x : ℕ) : 
  (x + 1023000) / 1024 ≤ x := by sorry

end NUMINAMATH_CALUDE_aladdin_gold_bars_l3337_333797


namespace NUMINAMATH_CALUDE_number_divided_by_0_025_equals_40_l3337_333700

theorem number_divided_by_0_025_equals_40 (x : ℝ) : x / 0.025 = 40 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_number_divided_by_0_025_equals_40_l3337_333700


namespace NUMINAMATH_CALUDE_gcd_lcm_product_24_40_l3337_333733

theorem gcd_lcm_product_24_40 : Nat.gcd 24 40 * Nat.lcm 24 40 = 960 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_24_40_l3337_333733


namespace NUMINAMATH_CALUDE_trap_existence_for_specific_feeders_l3337_333717

/-- A sequence of real numbers. -/
def Sequence := ℕ → ℝ

/-- A feeder is an interval that contains infinitely many terms of the sequence. -/
def IsFeeder (s : Sequence) (a b : ℝ) : Prop :=
  ∀ n : ℕ, ∃ m ≥ n, a ≤ s m ∧ s m ≤ b

/-- A trap is an interval that contains all but finitely many terms of the sequence. -/
def IsTrap (s : Sequence) (a b : ℝ) : Prop :=
  ∃ N : ℕ, ∀ n ≥ N, a ≤ s n ∧ s n ≤ b

/-- Main theorem about traps in sequences with specific feeders. -/
theorem trap_existence_for_specific_feeders (s : Sequence) 
  (h1 : IsFeeder s 0 1) (h2 : IsFeeder s 9 10) : 
  (¬ ∃ a : ℝ, IsTrap s a (a + 1)) ∧
  (∃ a : ℝ, IsTrap s a (a + 9)) := by sorry


end NUMINAMATH_CALUDE_trap_existence_for_specific_feeders_l3337_333717


namespace NUMINAMATH_CALUDE_product_of_numbers_l3337_333736

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 20) (h2 : x^2 + y^2 = 200) : x * y = 100 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l3337_333736


namespace NUMINAMATH_CALUDE_black_balloons_count_l3337_333763

/-- Given the number of gold balloons, calculate the number of black balloons -/
def black_balloons (gold : ℕ) (total : ℕ) : ℕ :=
  total - (gold + 2 * gold)

/-- Theorem: There are 150 black balloons given the problem conditions -/
theorem black_balloons_count : black_balloons 141 573 = 150 := by
  sorry

end NUMINAMATH_CALUDE_black_balloons_count_l3337_333763


namespace NUMINAMATH_CALUDE_sin_alpha_for_given_point_l3337_333730

theorem sin_alpha_for_given_point : ∀ α : Real,
  let x : Real := -2
  let y : Real := 2 * Real.sqrt 3
  let r : Real := Real.sqrt (x^2 + y^2)
  (∃ A : ℝ × ℝ, A = (x, y) ∧ A.1 = r * Real.cos α ∧ A.2 = r * Real.sin α) →
  Real.sin α = Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_sin_alpha_for_given_point_l3337_333730


namespace NUMINAMATH_CALUDE_parallel_transitive_l3337_333750

-- Define the concept of straight lines
variable (Line : Type)

-- Define the parallel relationship between lines
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem parallel_transitive (a b c : Line) :
  parallel a b → parallel b c → parallel a c := by
  sorry

end NUMINAMATH_CALUDE_parallel_transitive_l3337_333750


namespace NUMINAMATH_CALUDE_tables_left_l3337_333711

theorem tables_left (original_tables : ℝ) (customers_per_table : ℝ) (current_customers : ℕ) :
  original_tables = 44.0 →
  customers_per_table = 8.0 →
  current_customers = 256 →
  original_tables - (current_customers : ℝ) / customers_per_table = 12.0 := by
  sorry

end NUMINAMATH_CALUDE_tables_left_l3337_333711


namespace NUMINAMATH_CALUDE_function_shift_and_overlap_l3337_333712

theorem function_shift_and_overlap (f : ℝ → ℝ) :
  (∀ x, f (x - π / 12) = Real.cos (π / 2 - 2 * x)) →
  (∀ x, f x = Real.sin (2 * x - π / 6)) :=
by sorry

end NUMINAMATH_CALUDE_function_shift_and_overlap_l3337_333712


namespace NUMINAMATH_CALUDE_pyramid_volume_transformation_l3337_333713

/-- Represents a pyramid with a triangular base -/
structure Pyramid where
  volume : ℝ
  base_a : ℝ
  base_b : ℝ
  base_c : ℝ
  height : ℝ

/-- Transforms a pyramid according to the given conditions -/
def transform_pyramid (p : Pyramid) : Pyramid :=
  { volume := 0,  -- We'll prove this is 12 * p.volume
    base_a := 2 * p.base_a,
    base_b := 2 * p.base_b,
    base_c := 3 * p.base_c,
    height := 3 * p.height }

theorem pyramid_volume_transformation (p : Pyramid) :
  (transform_pyramid p).volume = 12 * p.volume := by
  sorry

#check pyramid_volume_transformation

end NUMINAMATH_CALUDE_pyramid_volume_transformation_l3337_333713


namespace NUMINAMATH_CALUDE_smallest_integer_inequality_l3337_333756

theorem smallest_integer_inequality : ∀ x : ℤ, x + 5 < 3*x - 9 → x ≥ 8 ∧ 8 + 5 < 3*8 - 9 := by sorry

end NUMINAMATH_CALUDE_smallest_integer_inequality_l3337_333756


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_squared_l3337_333706

/-- Theorem: Sum of reciprocals squared --/
theorem sum_of_reciprocals_squared :
  let a := Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 15
  let b := -Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 15
  let c := Real.sqrt 3 - Real.sqrt 5 + Real.sqrt 15
  let d := -Real.sqrt 3 - Real.sqrt 5 + Real.sqrt 15
  (1/a + 1/b + 1/c + 1/d)^2 = 960/3481 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_squared_l3337_333706

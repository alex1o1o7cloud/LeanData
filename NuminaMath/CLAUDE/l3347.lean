import Mathlib

namespace NUMINAMATH_CALUDE_ratio_of_a_to_b_l3347_334723

theorem ratio_of_a_to_b (A B : ℝ) (hA : A ≠ 0) (hB : B ≠ 0) 
  (h : (2 / 3) * A = (3 / 4) * B) : A / B = 9 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_a_to_b_l3347_334723


namespace NUMINAMATH_CALUDE_R_is_converse_negation_of_P_l3347_334713

-- Define the proposition P
def P : Prop := ∀ x y : ℝ, x + y = 0 → (x = -y ∧ y = -x)

-- Define the negation of P (Q)
def Q : Prop := ¬P

-- Define the inverse of Q (R)
def R : Prop := ∀ x y : ℝ, ¬(x = -y ∧ y = -x) → x + y ≠ 0

-- Theorem stating that R is the converse negation of P
theorem R_is_converse_negation_of_P : R = (∀ x y : ℝ, ¬(x = -y ∧ y = -x) → x + y ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_R_is_converse_negation_of_P_l3347_334713


namespace NUMINAMATH_CALUDE_larrys_coincidence_l3347_334708

theorem larrys_coincidence (a b c d e : ℝ) 
  (ha : a = 5) (hb : b = 3) (hc : c = 6) (hd : d = 4) :
  a - b + c + d - e = a - (b - (c + (d - e))) :=
by sorry

end NUMINAMATH_CALUDE_larrys_coincidence_l3347_334708


namespace NUMINAMATH_CALUDE_prob_second_day_restaurant_A_l3347_334720

/-- Represents the restaurants in the Olympic Village -/
inductive Restaurant
| A  -- Smart restaurant
| B  -- Manual restaurant

/-- The probability of choosing restaurant A on the second day -/
def prob_second_day_A (first_day_choice : Restaurant) : ℝ :=
  match first_day_choice with
  | Restaurant.A => 0.6
  | Restaurant.B => 0.5

/-- The probability of choosing a restaurant on the first day -/
def prob_first_day (r : Restaurant) : ℝ := 0.5

/-- The theorem stating the probability of going to restaurant A on the second day -/
theorem prob_second_day_restaurant_A :
  (prob_first_day Restaurant.A * prob_second_day_A Restaurant.A +
   prob_first_day Restaurant.B * prob_second_day_A Restaurant.B) = 0.55 := by
  sorry


end NUMINAMATH_CALUDE_prob_second_day_restaurant_A_l3347_334720


namespace NUMINAMATH_CALUDE_base4_to_decimal_example_l3347_334754

/-- Converts a base-4 number to decimal --/
def base4ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (4 ^ i)) 0

/-- The base-4 representation of the number --/
def base4Number : List Nat := [2, 1, 0, 0, 3]

/-- Theorem: The base-4 number 30012₍₄₎ is equal to 774 in decimal notation --/
theorem base4_to_decimal_example : base4ToDecimal base4Number = 774 := by
  sorry

end NUMINAMATH_CALUDE_base4_to_decimal_example_l3347_334754


namespace NUMINAMATH_CALUDE_two_stage_discount_l3347_334738

/-- Calculate the actual discount and difference from claimed discount in a two-stage discount scenario -/
theorem two_stage_discount (initial_discount additional_discount claimed_discount : ℝ) :
  initial_discount = 0.4 →
  additional_discount = 0.25 →
  claimed_discount = 0.6 →
  let remaining_after_initial := 1 - initial_discount
  let remaining_after_additional := remaining_after_initial * (1 - additional_discount)
  let actual_discount := 1 - remaining_after_additional
  actual_discount = 0.55 ∧ claimed_discount - actual_discount = 0.05 := by
  sorry

end NUMINAMATH_CALUDE_two_stage_discount_l3347_334738


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l3347_334745

def A : Set ℝ := {x | x < 1}
def B : Set ℝ := {-1, 0, 2}

theorem complement_A_intersect_B : 
  (Set.univ \ A) ∩ B = {2} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l3347_334745


namespace NUMINAMATH_CALUDE_variance_scaling_l3347_334729

/-- Given a list of 8 real numbers, compute its variance -/
def variance (xs : List ℝ) : ℝ := sorry

theorem variance_scaling (xs : List ℝ) (h : variance xs = 3) :
  variance (xs.map (· * 2)) = 12 := by sorry

end NUMINAMATH_CALUDE_variance_scaling_l3347_334729


namespace NUMINAMATH_CALUDE_cube_vertex_distance_to_plane_l3347_334799

/-- Given a cube with side length 15 and three vertices adjacent to vertex A
    at heights 15, 17, and 18 above a plane, the distance from vertex A to the plane is 28/3 -/
theorem cube_vertex_distance_to_plane :
  ∀ (a b c d : ℝ),
  a^2 + b^2 + c^2 = 1 →
  15 * a + d = 15 →
  15 * b + d = 17 →
  15 * c + d = 18 →
  d = 28 / 3 :=
by sorry

end NUMINAMATH_CALUDE_cube_vertex_distance_to_plane_l3347_334799


namespace NUMINAMATH_CALUDE_tangent_line_triangle_area_l3347_334785

/-- A line tangent to both y = x² and y = -1/x -/
structure TangentLine where
  -- The slope of the tangent line
  m : ℝ
  -- The y-intercept of the tangent line
  b : ℝ
  -- The x-coordinate of the point of tangency on y = x²
  x₁ : ℝ
  -- The x-coordinate of the point of tangency on y = -1/x
  x₂ : ℝ
  -- Condition: The line is tangent to y = x² at (x₁, x₁²)
  h₁ : m * x₁ + b = x₁^2
  -- Condition: The slope at the point of tangency on y = x² is correct
  h₂ : m = 2 * x₁
  -- Condition: The line is tangent to y = -1/x at (x₂, -1/x₂)
  h₃ : m * x₂ + b = -1 / x₂
  -- Condition: The slope at the point of tangency on y = -1/x is correct
  h₄ : m = 1 / x₂^2

/-- The area of the triangle formed by a tangent line and the coordinate axes is 2 -/
theorem tangent_line_triangle_area (l : TangentLine) : 
  (1 / 2) * (1 / l.m) * (-l.b) = 2 := by sorry

end NUMINAMATH_CALUDE_tangent_line_triangle_area_l3347_334785


namespace NUMINAMATH_CALUDE_product_of_two_numbers_with_sum_100_l3347_334756

theorem product_of_two_numbers_with_sum_100 (a : ℝ) : 
  let b := 100 - a
  (a + b = 100) → (a * b = a * (100 - a)) := by
  sorry

end NUMINAMATH_CALUDE_product_of_two_numbers_with_sum_100_l3347_334756


namespace NUMINAMATH_CALUDE_smallest_number_l3347_334750

-- Define the numbers in their respective bases
def num_base3 : ℕ := 1 * 3^3 + 0 * 3^2 + 0 * 3^1 + 2 * 3^0
def num_base6 : ℕ := 2 * 6^2 + 1 * 6^1 + 0 * 6^0
def num_base4 : ℕ := 1 * 4^3 + 0 * 4^2 + 0 * 4^1 + 0 * 4^0
def num_base2 : ℕ := 1 * 2^5 + 1 * 2^4 + 1 * 2^3 + 1 * 2^2 + 1 * 2^1 + 1 * 2^0

-- Theorem statement
theorem smallest_number :
  num_base2 < num_base3 ∧ num_base2 < num_base6 ∧ num_base2 < num_base4 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_l3347_334750


namespace NUMINAMATH_CALUDE_triangle_angles_l3347_334787

open Real

theorem triangle_angles (a b c : ℝ) (A B C : ℝ) : 
  let m : ℝ × ℝ := (sqrt 3, -1)
  let n : ℝ × ℝ := (cos A, sin A)
  (m.1 * n.1 + m.2 * n.2 = 0) →  -- m ⊥ n
  (a * cos B + b * cos A = c * sin C) →  -- given condition
  (a > 0 ∧ b > 0 ∧ c > 0) →  -- positive side lengths
  (A > 0 ∧ B > 0 ∧ C > 0) →  -- positive angles
  (A + B + C = π) →  -- sum of angles in a triangle
  (A = π/3 ∧ B = π/6) :=
by sorry

end NUMINAMATH_CALUDE_triangle_angles_l3347_334787


namespace NUMINAMATH_CALUDE_equation_solution_l3347_334700

theorem equation_solution (y : ℝ) : 
  Real.sqrt (4 + Real.sqrt (3 * y - 2)) = Real.sqrt 12 → y = 22 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l3347_334700


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l3347_334770

-- Problem 1
theorem problem_1 : (1) - 6 - 13 + (-24) = -43 := by sorry

-- Problem 2
theorem problem_2 : (-6) / (3/7) * (-7) = 98 := by sorry

-- Problem 3
theorem problem_3 : (2/3 - 1/12 - 1/15) * (-60) = -31 := by sorry

-- Problem 4
theorem problem_4 : -1^4 - 1/6 * (2 - (-3)^2) = 1/6 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l3347_334770


namespace NUMINAMATH_CALUDE_fancy_sandwich_cost_l3347_334727

/-- The cost of a fancy ham and cheese sandwich given Teresa's shopping list and total spent --/
theorem fancy_sandwich_cost (num_sandwiches : ℕ) (salami_cost brie_cost olive_price_per_pound feta_price_per_pound bread_cost total_spent : ℚ) 
  (olive_weight feta_weight : ℚ) : 
  num_sandwiches = 2 ∧ 
  salami_cost = 4 ∧ 
  brie_cost = 3 * salami_cost ∧ 
  olive_price_per_pound = 10 ∧ 
  olive_weight = 1/4 ∧ 
  feta_price_per_pound = 8 ∧ 
  feta_weight = 1/2 ∧ 
  bread_cost = 2 ∧ 
  total_spent = 40 → 
  (total_spent - (salami_cost + brie_cost + olive_price_per_pound * olive_weight + 
    feta_price_per_pound * feta_weight + bread_cost)) / num_sandwiches = 7.75 := by
  sorry

end NUMINAMATH_CALUDE_fancy_sandwich_cost_l3347_334727


namespace NUMINAMATH_CALUDE_two_from_four_one_from_pair_l3347_334726

/-- The number of ways to select 2 students from a group of 4, where exactly one is chosen from a specific pair --/
theorem two_from_four_one_from_pair : ℕ := by
  sorry

end NUMINAMATH_CALUDE_two_from_four_one_from_pair_l3347_334726


namespace NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l3347_334742

theorem rectangular_to_polar_conversion :
  let x : ℝ := 8
  let y : ℝ := 3 * Real.sqrt 3
  let r : ℝ := Real.sqrt (x^2 + y^2)
  let θ : ℝ := Real.arctan (y / x)
  r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧ r = Real.sqrt 91 ∧ θ = Real.arctan (3 * Real.sqrt 3 / 8) :=
by sorry

end NUMINAMATH_CALUDE_rectangular_to_polar_conversion_l3347_334742


namespace NUMINAMATH_CALUDE_trig_identity_l3347_334746

theorem trig_identity (α : Real) 
  (h : 3 * Real.sin α + Real.cos α = 0) : 
  1 / (Real.cos α ^ 2 + 2 * Real.sin α * Real.cos α) = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l3347_334746


namespace NUMINAMATH_CALUDE_tan_function_property_l3347_334760

theorem tan_function_property (a : ℝ) : 
  let f : ℝ → ℝ := λ x => 1 + Real.tan x
  f a = 3 → f (-a) = -1 := by
sorry

end NUMINAMATH_CALUDE_tan_function_property_l3347_334760


namespace NUMINAMATH_CALUDE_personal_planners_count_l3347_334718

/-- The cost of a spiral notebook in dollars -/
def spiral_notebook_cost : ℝ := 15

/-- The cost of a personal planner in dollars -/
def personal_planner_cost : ℝ := 10

/-- The number of spiral notebooks bought -/
def num_spiral_notebooks : ℕ := 4

/-- The discount rate applied to the total cost -/
def discount_rate : ℝ := 0.2

/-- The total discounted cost in dollars -/
def total_discounted_cost : ℝ := 112

/-- The number of personal planners bought -/
def num_personal_planners : ℕ := 8

theorem personal_planners_count :
  ∃ (x : ℕ),
    (1 - discount_rate) * (spiral_notebook_cost * num_spiral_notebooks + personal_planner_cost * x) = total_discounted_cost ∧
    x = num_personal_planners :=
by sorry

end NUMINAMATH_CALUDE_personal_planners_count_l3347_334718


namespace NUMINAMATH_CALUDE_max_value_of_S_l3347_334722

theorem max_value_of_S (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  let S := min x (min (y + 1/x) (1/y))
  ∃ (max_S : ℝ), max_S = Real.sqrt 2 ∧
    (∀ x' y' : ℝ, x' > 0 → y' > 0 → 
      min x' (min (y' + 1/x') (1/y')) ≤ max_S) ∧
    S = max_S ↔ x = Real.sqrt 2 ∧ y = Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_S_l3347_334722


namespace NUMINAMATH_CALUDE_not_perfect_square_l3347_334717

theorem not_perfect_square : 
  (∃ x : ℕ, 6^3024 = x^2) ∧
  (∀ y : ℕ, 7^3025 ≠ y^2) ∧
  (∃ z : ℕ, 8^3026 = z^2) ∧
  (∃ w : ℕ, 9^3027 = w^2) ∧
  (∃ v : ℕ, 10^3028 = v^2) := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_l3347_334717


namespace NUMINAMATH_CALUDE_starting_lineup_combinations_l3347_334711

/-- The number of ways to choose k elements from a set of n elements --/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The total number of players on the team --/
def totalPlayers : ℕ := 15

/-- The number of All-Star players --/
def allStars : ℕ := 3

/-- The size of the starting lineup --/
def lineupSize : ℕ := 5

theorem starting_lineup_combinations :
  choose (totalPlayers - allStars) (lineupSize - allStars) = 66 := by
  sorry

end NUMINAMATH_CALUDE_starting_lineup_combinations_l3347_334711


namespace NUMINAMATH_CALUDE_line_ellipse_intersection_length_l3347_334721

theorem line_ellipse_intersection_length : ∃ A B : ℝ × ℝ,
  (∀ x y : ℝ, y = x - 1 → x^2 / 4 + y^2 / 3 = 1 → (x, y) = A ∨ (x, y) = B) →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 24 / 7 :=
sorry

end NUMINAMATH_CALUDE_line_ellipse_intersection_length_l3347_334721


namespace NUMINAMATH_CALUDE_y_min_at_a_or_b_l3347_334790

/-- The function y in terms of x, a, and b -/
def y (x a b : ℝ) : ℝ := (x - a)^3 + (x - b)^3

/-- Theorem stating that the minimum of y occurs at either a or b -/
theorem y_min_at_a_or_b (a b : ℝ) :
  ∃ (x : ℝ), (∀ (z : ℝ), y z a b ≥ y x a b) ∧ (x = a ∨ x = b) := by
  sorry

end NUMINAMATH_CALUDE_y_min_at_a_or_b_l3347_334790


namespace NUMINAMATH_CALUDE_decimal_equals_fraction_l3347_334752

/-- The decimal representation of the number we're considering -/
def decimal : ℚ := 0.73264264264

/-- The denominator of the fraction we're looking for -/
def denominator : ℕ := 999900

/-- The numerator of the fraction we're looking for -/
def numerator : ℕ := 732635316

/-- Theorem stating that our decimal is equal to the fraction numerator/denominator -/
theorem decimal_equals_fraction : decimal = (numerator : ℚ) / denominator := by
  sorry

end NUMINAMATH_CALUDE_decimal_equals_fraction_l3347_334752


namespace NUMINAMATH_CALUDE_mod_twelve_power_six_l3347_334716

theorem mod_twelve_power_six (n : ℕ) : 12^6 ≡ n [ZMOD 9] → 0 ≤ n → n < 9 → n = 0 := by
  sorry

end NUMINAMATH_CALUDE_mod_twelve_power_six_l3347_334716


namespace NUMINAMATH_CALUDE_expression_value_l3347_334766

theorem expression_value : 3 * ((18 + 7)^2 - (7^2 + 18^2)) = 756 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3347_334766


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l3347_334725

theorem cubic_roots_sum (a b c : ℝ) : 
  (a^3 - 3*a - 1 = 0) → 
  (b^3 - 3*b - 1 = 0) → 
  (c^3 - 3*c - 1 = 0) → 
  a*(b - c)^2 + b*(c - a)^2 + c*(a - b)^2 = -9 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l3347_334725


namespace NUMINAMATH_CALUDE_inequality_proof_l3347_334715

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  1 / (a^3 * (b + c)) + 1 / (b^3 * (a + c)) + 1 / (c^3 * (a + b)) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3347_334715


namespace NUMINAMATH_CALUDE_horner_method_v3_l3347_334769

def f (x : ℝ) : ℝ := 3*x^5 - 2*x^4 + 2*x^3 - 4*x^2 - 7

def horner_v3 (a : ℝ → ℝ) (x : ℝ) : ℝ :=
  let v0 := 3
  let v1 := v0 * x - 2
  let v2 := v1 * x + 2
  v2 * x - 4

theorem horner_method_v3 :
  horner_v3 f 2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_v3_l3347_334769


namespace NUMINAMATH_CALUDE_line_segment_endpoint_l3347_334751

/-- Given a line segment from (3, 7) to (-9, y) with length 15 and y > 0, prove y = 16 -/
theorem line_segment_endpoint (y : ℝ) : 
  (((3 : ℝ) - (-9))^2 + (y - 7)^2 = 15^2) → y > 0 → y = 16 :=
by sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l3347_334751


namespace NUMINAMATH_CALUDE_purely_imaginary_iff_a_equals_one_l3347_334767

theorem purely_imaginary_iff_a_equals_one (a : ℝ) :
  let z : ℂ := a^2 - 1 + (a + 1) * I
  (z.re = 0 ∧ z.im ≠ 0) ↔ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_iff_a_equals_one_l3347_334767


namespace NUMINAMATH_CALUDE_sum_of_ages_l3347_334714

/-- Viggo's age when his brother was 2 years old -/
def viggos_age_when_brother_was_2 (brothers_age_when_2 : ℕ) : ℕ :=
  10 + 2 * brothers_age_when_2

/-- The current age of Viggo's younger brother -/
def brothers_current_age : ℕ := 10

/-- Viggo's current age -/
def viggos_current_age : ℕ :=
  brothers_current_age + (viggos_age_when_brother_was_2 2 - 2)

theorem sum_of_ages : 
  viggos_current_age + brothers_current_age = 32 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ages_l3347_334714


namespace NUMINAMATH_CALUDE_number_of_grey_stones_l3347_334782

/-- Given a collection of stones with specific properties, prove the number of grey stones. -/
theorem number_of_grey_stones 
  (total_stones : ℕ) 
  (white_stones : ℕ) 
  (green_stones : ℕ) 
  (h1 : total_stones = 100)
  (h2 : white_stones = 60)
  (h3 : green_stones = 60)
  (h4 : white_stones > total_stones - white_stones)
  (h5 : (white_stones : ℚ) / (total_stones - white_stones) = (grey_stones : ℚ) / green_stones) :
  grey_stones = 90 :=
by
  sorry

#check number_of_grey_stones

end NUMINAMATH_CALUDE_number_of_grey_stones_l3347_334782


namespace NUMINAMATH_CALUDE_bad_carrots_count_bad_carrots_problem_l3347_334792

theorem bad_carrots_count (nancy_carrots : ℕ) (mom_carrots : ℕ) (good_carrots : ℕ) : ℕ :=
  let total_carrots := nancy_carrots + mom_carrots
  total_carrots - good_carrots

theorem bad_carrots_problem :
  bad_carrots_count 38 47 71 = 14 := by
  sorry

end NUMINAMATH_CALUDE_bad_carrots_count_bad_carrots_problem_l3347_334792


namespace NUMINAMATH_CALUDE_quadratic_always_positive_l3347_334779

theorem quadratic_always_positive : ∀ x : ℝ, 3 * x^2 - 2 * x + 1 > 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_l3347_334779


namespace NUMINAMATH_CALUDE_polynomial_root_equivalence_l3347_334704

theorem polynomial_root_equivalence : ∀ r : ℝ, 
  r^2 - 2*r - 1 = 0 → r^5 - 29*r - 12 = 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_root_equivalence_l3347_334704


namespace NUMINAMATH_CALUDE_hannah_ran_9km_on_monday_l3347_334712

/-- The distance Hannah ran on Wednesday in meters -/
def wednesday_distance : ℕ := 4816

/-- The distance Hannah ran on Friday in meters -/
def friday_distance : ℕ := 2095

/-- The additional distance Hannah ran on Monday compared to Wednesday and Friday combined, in meters -/
def monday_additional_distance : ℕ := 2089

/-- The number of meters in a kilometer -/
def meters_per_kilometer : ℕ := 1000

/-- Theorem stating that Hannah ran 9 kilometers on Monday -/
theorem hannah_ran_9km_on_monday : 
  (wednesday_distance + friday_distance + monday_additional_distance) / meters_per_kilometer = 9 := by
  sorry

end NUMINAMATH_CALUDE_hannah_ran_9km_on_monday_l3347_334712


namespace NUMINAMATH_CALUDE_dave_tray_capacity_l3347_334768

theorem dave_tray_capacity (trays_table1 trays_table2 num_trips : ℕ) 
  (h1 : trays_table1 = 17)
  (h2 : trays_table2 = 55)
  (h3 : num_trips = 8) :
  (trays_table1 + trays_table2) / num_trips = 9 := by
  sorry

#check dave_tray_capacity

end NUMINAMATH_CALUDE_dave_tray_capacity_l3347_334768


namespace NUMINAMATH_CALUDE_other_factor_is_five_l3347_334778

def w : ℕ := 120

theorem other_factor_is_five :
  ∀ (product : ℕ),
  (∃ (k : ℕ), product = 936 * w * k) →
  (∃ (m : ℕ), product = 2^5 * 3^3 * m) →
  (∀ (x : ℕ), x < w → ¬(∃ (y : ℕ), 936 * x * y = product)) →
  (∃ (n : ℕ), product = 936 * w * 5 * n) :=
by sorry

end NUMINAMATH_CALUDE_other_factor_is_five_l3347_334778


namespace NUMINAMATH_CALUDE_expression_equals_zero_l3347_334703

theorem expression_equals_zero (x : ℝ) : 
  (((x^2 - 2*x + 1) / (x^2 - 1)) / ((x - 1) / (x^2 + x))) - x = 0 :=
by sorry

end NUMINAMATH_CALUDE_expression_equals_zero_l3347_334703


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3347_334772

theorem sufficient_not_necessary (a b : ℝ) : 
  (a^2 + b^2 = 0 → a * b = 0) ∧ 
  ∃ a b : ℝ, a * b = 0 ∧ a^2 + b^2 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3347_334772


namespace NUMINAMATH_CALUDE_projection_of_a_onto_b_l3347_334755

def a : Fin 2 → ℝ := ![3, 4]
def b : Fin 2 → ℝ := ![1, 2]

theorem projection_of_a_onto_b :
  let proj := (((a 0) * (b 0) + (a 1) * (b 1)) / ((b 0)^2 + (b 1)^2)) • b
  proj 0 = 11/5 ∧ proj 1 = 22/5 := by
  sorry

end NUMINAMATH_CALUDE_projection_of_a_onto_b_l3347_334755


namespace NUMINAMATH_CALUDE_mansion_rooms_less_than_55_l3347_334764

/-- Represents the number of rooms with a specific type of bouquet -/
structure BouquetRooms where
  roses : ℕ
  carnations : ℕ
  chrysanthemums : ℕ

/-- Represents the number of rooms with combinations of bouquets -/
structure OverlapRooms where
  carnations_chrysanthemums : ℕ
  chrysanthemums_roses : ℕ
  carnations_roses : ℕ

theorem mansion_rooms_less_than_55 (b : BouquetRooms) (o : OverlapRooms) 
    (h1 : b.roses = 30)
    (h2 : b.carnations = 20)
    (h3 : b.chrysanthemums = 10)
    (h4 : o.carnations_chrysanthemums = 2)
    (h5 : o.chrysanthemums_roses = 3)
    (h6 : o.carnations_roses = 4) :
    b.roses + b.carnations + b.chrysanthemums - 
    o.carnations_chrysanthemums - o.chrysanthemums_roses - o.carnations_roses < 55 := by
  sorry


end NUMINAMATH_CALUDE_mansion_rooms_less_than_55_l3347_334764


namespace NUMINAMATH_CALUDE_solution_set_part1_range_of_a_l3347_334705

-- Define the function f
def f (a x : ℝ) : ℝ := |x - a| + |x + 3|

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f 1 x ≥ 6} = {x : ℝ | x ≤ -4 ∨ x ≥ 2} :=
sorry

-- Part 2
theorem range_of_a :
  {a : ℝ | ∀ x, f a x > -a} = {a : ℝ | a > -3/2} :=
sorry

end NUMINAMATH_CALUDE_solution_set_part1_range_of_a_l3347_334705


namespace NUMINAMATH_CALUDE_candy_distribution_l3347_334748

/-- Represents the number of candies eaten by each person -/
structure CandyEaten where
  andrey : ℕ
  boris : ℕ
  denis : ℕ

/-- Represents the relative eating rates of the three people -/
structure EatingRates where
  andrey_boris : ℚ  -- Ratio of Andrey's rate to Boris's rate
  andrey_denis : ℚ  -- Ratio of Andrey's rate to Denis's rate

/-- Given the eating rates and total candies eaten, calculate how many candies each person ate -/
def calculate_candy_eaten (rates : EatingRates) (total : ℕ) : CandyEaten :=
  sorry

/-- The main theorem to prove -/
theorem candy_distribution (rates : EatingRates) (total : ℕ) :
  rates.andrey_boris = 4/3 →
  rates.andrey_denis = 6/7 →
  total = 70 →
  let result := calculate_candy_eaten rates total
  result.andrey = 24 ∧ result.boris = 18 ∧ result.denis = 28 :=
sorry

end NUMINAMATH_CALUDE_candy_distribution_l3347_334748


namespace NUMINAMATH_CALUDE_f_properties_implications_l3347_334773

/-- A function satisfying the given properties -/
def f_properties (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = f x) ∧  -- even function
  (∀ x, f x = f (2 - x)) ∧  -- symmetric about x = 1
  (∀ x₁ x₂, x₁ ∈ Set.Icc 0 (1/2) → x₂ ∈ Set.Icc 0 (1/2) → f (x₁ + x₂) = f x₁ * f x₂) ∧
  f 1 = 2

theorem f_properties_implications {f : ℝ → ℝ} (hf : f_properties f) :
  f (1/2) = Real.sqrt 2 ∧ f (1/4) = Real.sqrt (Real.sqrt 2) ∧ ∀ x, f x = f (x + 2) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_implications_l3347_334773


namespace NUMINAMATH_CALUDE_hyperbola_equation_special_case_l3347_334798

/-- A hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  a_pos : 0 < a
  b_pos : 0 < b

/-- The equation of a hyperbola -/
def hyperbola_equation (h : Hyperbola a b) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

/-- The distance from the foci to the asymptotes of a hyperbola -/
def foci_to_asymptote_distance (h : Hyperbola a b) : ℝ :=
  b

/-- The length of the real axis of a hyperbola -/
def real_axis_length (h : Hyperbola a b) : ℝ :=
  2 * a

/-- Theorem: If the distance from the foci to the asymptotes equals the length of the real axis
    and the point (2,2) lies on the hyperbola, then the equation of the hyperbola is x^2/3 - y^2/12 = 1 -/
theorem hyperbola_equation_special_case (h : Hyperbola a b) :
  foci_to_asymptote_distance h = real_axis_length h →
  hyperbola_equation h 2 2 →
  ∀ x y, hyperbola_equation h x y ↔ x^2 / 3 - y^2 / 12 = 1 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_special_case_l3347_334798


namespace NUMINAMATH_CALUDE_range_sum_of_bounds_l3347_334765

open Set Real

noncomputable def h (x : ℝ) : ℝ := 3 / (3 + 9 * x^2)

theorem range_sum_of_bounds (a b : ℝ) :
  (∀ y ∈ Set.range h, a < y ∧ y ≤ b) ∧
  (∀ ε > 0, ∃ y ∈ Set.range h, y < a + ε) ∧
  (b ∈ Set.range h) →
  a + b = 1 := by sorry

end NUMINAMATH_CALUDE_range_sum_of_bounds_l3347_334765


namespace NUMINAMATH_CALUDE_wechat_group_size_l3347_334701

theorem wechat_group_size :
  ∃ (n : ℕ), n > 0 ∧ n * (n - 1) / 2 = 72 ∧ n = 9 := by
  sorry

end NUMINAMATH_CALUDE_wechat_group_size_l3347_334701


namespace NUMINAMATH_CALUDE_num_closed_lockers_l3347_334733

/-- The number of lockers and students -/
def n : ℕ := 100

/-- A locker is open if and only if its number is a perfect square -/
def is_open (k : ℕ) : Prop := ∃ m : ℕ, k = m^2

/-- The number of perfect squares less than or equal to n -/
def num_perfect_squares (n : ℕ) : ℕ := (n.sqrt : ℕ)

/-- The main theorem: The number of closed lockers is equal to
    the total number of lockers minus the number of perfect squares -/
theorem num_closed_lockers : 
  n - (num_perfect_squares n) = 90 := by sorry

end NUMINAMATH_CALUDE_num_closed_lockers_l3347_334733


namespace NUMINAMATH_CALUDE_ship_passengers_l3347_334719

theorem ship_passengers : ∀ (P : ℕ),
  (P / 12 : ℚ) + (P / 4 : ℚ) + (P / 9 : ℚ) + (P / 6 : ℚ) + 42 = P →
  P = 108 := by
  sorry

end NUMINAMATH_CALUDE_ship_passengers_l3347_334719


namespace NUMINAMATH_CALUDE_adams_earnings_l3347_334789

/-- Adam's lawn mowing earnings problem -/
theorem adams_earnings (earnings_per_lawn : ℕ) (total_lawns : ℕ) (forgotten_lawns : ℕ) :
  earnings_per_lawn = 9 →
  total_lawns = 12 →
  forgotten_lawns = 8 →
  (total_lawns - forgotten_lawns) * earnings_per_lawn = 36 :=
by sorry

end NUMINAMATH_CALUDE_adams_earnings_l3347_334789


namespace NUMINAMATH_CALUDE_solution_implies_relationship_l3347_334732

theorem solution_implies_relationship (a b c : ℝ) :
  (a * (-3) + c * (-2) = 1) →
  (c * (-3) - b * (-2) = 2) →
  9 * a + 4 * b = 1 := by
sorry

end NUMINAMATH_CALUDE_solution_implies_relationship_l3347_334732


namespace NUMINAMATH_CALUDE_dimes_in_tip_jar_l3347_334774

def nickel_value : ℚ := 0.05
def dime_value : ℚ := 0.10
def half_dollar_value : ℚ := 0.50

def shining_nickels : ℕ := 3
def shining_dimes : ℕ := 13
def tip_jar_half_dollars : ℕ := 9

def total_amount : ℚ := 6.65

theorem dimes_in_tip_jar :
  ∃ (tip_jar_dimes : ℕ),
    (shining_nickels * nickel_value + shining_dimes * dime_value +
     tip_jar_dimes * dime_value + tip_jar_half_dollars * half_dollar_value = total_amount) ∧
    tip_jar_dimes = 7 :=
by sorry

end NUMINAMATH_CALUDE_dimes_in_tip_jar_l3347_334774


namespace NUMINAMATH_CALUDE_probability_standard_bulb_l3347_334724

/-- Probability of a bulb being from factory 1 -/
def p_factory1 : ℝ := 0.45

/-- Probability of a bulb being from factory 2 -/
def p_factory2 : ℝ := 0.40

/-- Probability of a bulb being from factory 3 -/
def p_factory3 : ℝ := 0.15

/-- Probability of a bulb from factory 1 being standard -/
def p_standard_factory1 : ℝ := 0.70

/-- Probability of a bulb from factory 2 being standard -/
def p_standard_factory2 : ℝ := 0.80

/-- Probability of a bulb from factory 3 being standard -/
def p_standard_factory3 : ℝ := 0.81

/-- The probability of purchasing a standard bulb from the store -/
theorem probability_standard_bulb :
  p_factory1 * p_standard_factory1 + 
  p_factory2 * p_standard_factory2 + 
  p_factory3 * p_standard_factory3 = 0.7565 := by
  sorry

end NUMINAMATH_CALUDE_probability_standard_bulb_l3347_334724


namespace NUMINAMATH_CALUDE_bulb_selection_problem_l3347_334783

theorem bulb_selection_problem (total_bulbs : ℕ) (defective_bulbs : ℕ) (probability : ℚ) :
  total_bulbs = 10 →
  defective_bulbs = 4 →
  probability = 1 / 15 →
  ∃ n : ℕ, (((total_bulbs - defective_bulbs : ℚ) / total_bulbs) ^ n = probability) ∧ n = 5 :=
by sorry

end NUMINAMATH_CALUDE_bulb_selection_problem_l3347_334783


namespace NUMINAMATH_CALUDE_roger_has_more_candy_l3347_334793

-- Define the number of bags for Sandra and Roger
def sandra_bags : ℕ := 2
def roger_bags : ℕ := 2

-- Define the number of candies in each of Sandra's bags
def sandra_candy_per_bag : ℕ := 6

-- Define the number of candies in Roger's bags
def roger_candy_bag1 : ℕ := 11
def roger_candy_bag2 : ℕ := 3

-- Calculate the total number of candies for Sandra and Roger
def sandra_total : ℕ := sandra_bags * sandra_candy_per_bag
def roger_total : ℕ := roger_candy_bag1 + roger_candy_bag2

-- State the theorem
theorem roger_has_more_candy : roger_total = sandra_total + 2 := by
  sorry

end NUMINAMATH_CALUDE_roger_has_more_candy_l3347_334793


namespace NUMINAMATH_CALUDE_household_survey_l3347_334759

/-- Households survey problem -/
theorem household_survey (neither : ℕ) (only_w : ℕ) (both : ℕ) 
  (h1 : neither = 80)
  (h2 : only_w = 60)
  (h3 : both = 40) :
  neither + only_w + 3 * both + both = 300 := by
  sorry

end NUMINAMATH_CALUDE_household_survey_l3347_334759


namespace NUMINAMATH_CALUDE_combined_weight_is_6600_l3347_334795

/-- The weight of the elephant in tons -/
def elephant_weight_tons : ℝ := 3

/-- The weight of a ton in pounds -/
def pounds_per_ton : ℝ := 2000

/-- The percentage of the elephant's weight that the donkey weighs less -/
def donkey_weight_percentage : ℝ := 0.9

/-- The combined weight of the elephant and donkey in pounds -/
def combined_weight_pounds : ℝ :=
  elephant_weight_tons * pounds_per_ton +
  elephant_weight_tons * pounds_per_ton * (1 - donkey_weight_percentage)

theorem combined_weight_is_6600 :
  combined_weight_pounds = 6600 :=
sorry

end NUMINAMATH_CALUDE_combined_weight_is_6600_l3347_334795


namespace NUMINAMATH_CALUDE_smallest_multiple_37_3_mod_97_l3347_334788

theorem smallest_multiple_37_3_mod_97 : ∃ n : ℕ, 
  n > 0 ∧ 
  37 ∣ n ∧ 
  n % 97 = 3 ∧ 
  ∀ m : ℕ, m > 0 → 37 ∣ m → m % 97 = 3 → n ≤ m :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_37_3_mod_97_l3347_334788


namespace NUMINAMATH_CALUDE_power_inequality_l3347_334741

theorem power_inequality (n : ℕ) (hn : n > 3) : n^(n+1) > (n+1)^n := by
  sorry

end NUMINAMATH_CALUDE_power_inequality_l3347_334741


namespace NUMINAMATH_CALUDE_divisibility_by_five_l3347_334763

theorem divisibility_by_five (x y : ℤ) :
  (∃ k : ℤ, x^2 - 2*x*y + 2*y^2 = 5*k ∨ x^2 + 2*x*y + 2*y^2 = 5*k) ↔ 
  (∃ a b : ℤ, x = 5*a ∧ y = 5*b) ∨ 
  (∀ k : ℤ, x ≠ 5*k ∧ y ≠ 5*k) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_by_five_l3347_334763


namespace NUMINAMATH_CALUDE_jackson_decorations_given_l3347_334781

/-- Given that Mrs. Jackson has 4 boxes of Christmas decorations with 15 decorations in each box
    and she used 35 decorations, prove that she gave 25 decorations to her neighbor. -/
theorem jackson_decorations_given (boxes : ℕ) (decorations_per_box : ℕ) (used_decorations : ℕ)
    (h1 : boxes = 4)
    (h2 : decorations_per_box = 15)
    (h3 : used_decorations = 35) :
    boxes * decorations_per_box - used_decorations = 25 := by
  sorry

end NUMINAMATH_CALUDE_jackson_decorations_given_l3347_334781


namespace NUMINAMATH_CALUDE_problem_solution_l3347_334710

-- Define a function to check if a number is square-free
def is_square_free (n : ℕ) : Prop :=
  ∀ (p : ℕ), Nat.Prime p → (p * p ∣ n) → p = 1

-- Define the condition for the problem
def satisfies_condition (p : ℕ) : Prop :=
  Nat.Prime p ∧ p ≥ 3 ∧
  ∀ (q : ℕ), Nat.Prime q → q < p →
    is_square_free (p - p / q * q)

-- State the theorem
theorem problem_solution :
  {p : ℕ | satisfies_condition p} = {3, 5, 7, 13} :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l3347_334710


namespace NUMINAMATH_CALUDE_sarah_car_robots_l3347_334797

/-- Prove that Sarah has 125 car robots given the conditions of the problem -/
theorem sarah_car_robots :
  ∀ (tom michael bob sarah : ℕ),
  tom = 15 →
  michael = 2 * tom →
  bob = 8 * michael →
  sarah = (bob / 2) + 5 →
  sarah = 125 := by
sorry

end NUMINAMATH_CALUDE_sarah_car_robots_l3347_334797


namespace NUMINAMATH_CALUDE_basketball_team_selection_l3347_334731

def total_players : ℕ := 18
def quadruplets : ℕ := 4
def players_to_choose : ℕ := 8
def max_quadruplets : ℕ := 2

theorem basketball_team_selection :
  (Nat.choose total_players players_to_choose) -
  (Nat.choose quadruplets 3 * Nat.choose (total_players - quadruplets) (players_to_choose - 3) +
   Nat.choose quadruplets 4 * Nat.choose (total_players - quadruplets) (players_to_choose - 4)) = 34749 := by
  sorry

end NUMINAMATH_CALUDE_basketball_team_selection_l3347_334731


namespace NUMINAMATH_CALUDE_parabola_passes_through_points_parabola_general_form_l3347_334737

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2 - 2*x - 3

-- Theorem statement
theorem parabola_passes_through_points :
  (parabola (-1) = 0) ∧ (parabola 3 = 0) :=
by
  sorry

-- Verify the general form
theorem parabola_general_form (x : ℝ) :
  ∃ (b c : ℝ), parabola x = x^2 - b*x + c :=
by
  sorry

end NUMINAMATH_CALUDE_parabola_passes_through_points_parabola_general_form_l3347_334737


namespace NUMINAMATH_CALUDE_range_of_a_l3347_334786

-- Define the propositions p and q
def p (x a : ℝ) : Prop := |x - a| < 3
def q (x : ℝ) : Prop := x^2 - 2*x - 3 < 0

-- Define the theorem
theorem range_of_a :
  (∀ x, q x → p x a) ∧ 
  (∃ x, p x a ∧ ¬q x) →
  0 ≤ a ∧ a ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3347_334786


namespace NUMINAMATH_CALUDE_solve_equation_l3347_334707

theorem solve_equation (x y : ℝ) (h1 : y = (x^2 - 9) / (x - 3)) 
  (h2 : y = 3*x) (h3 : x ≠ 3) : x = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3347_334707


namespace NUMINAMATH_CALUDE_smallest_cut_length_l3347_334794

theorem smallest_cut_length (x : ℕ) : x > 0 ∧ x ≤ 12 ∧ (12 - x) + (20 - x) ≤ (24 - x) →
  x ≥ 8 ∧ ∀ y : ℕ, y > 0 ∧ y < x → (12 - y) + (20 - y) > (24 - y) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_cut_length_l3347_334794


namespace NUMINAMATH_CALUDE_clothing_price_theorem_l3347_334744

/-- The price per item of clothing in yuan -/
def price : ℝ := 110

/-- The number of items sold in the first scenario -/
def quantity1 : ℕ := 10

/-- The number of items sold in the second scenario -/
def quantity2 : ℕ := 11

/-- The discount percentage in the first scenario -/
def discount_percent : ℝ := 0.08

/-- The discount amount in yuan for the second scenario -/
def discount_amount : ℝ := 30

theorem clothing_price_theorem :
  quantity1 * (price * (1 - discount_percent)) = quantity2 * (price - discount_amount) :=
sorry

end NUMINAMATH_CALUDE_clothing_price_theorem_l3347_334744


namespace NUMINAMATH_CALUDE_component_service_life_probability_l3347_334730

theorem component_service_life_probability 
  (P_exceed_1_year : ℝ) 
  (P_exceed_2_years : ℝ) 
  (h1 : P_exceed_1_year = 0.6) 
  (h2 : P_exceed_2_years = 0.3) :
  (P_exceed_2_years / P_exceed_1_year) = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_component_service_life_probability_l3347_334730


namespace NUMINAMATH_CALUDE_wheel_probability_l3347_334771

theorem wheel_probability (p_A p_B p_C p_D : ℚ) : 
  p_A = 1/4 →
  p_B = 1/3 →
  p_C = 1/6 →
  p_A + p_B + p_C + p_D = 1 →
  p_D = 1/4 := by
sorry

end NUMINAMATH_CALUDE_wheel_probability_l3347_334771


namespace NUMINAMATH_CALUDE_min_disks_required_l3347_334734

/-- Represents the number of files of each size --/
structure FileDistribution :=
  (large : Nat)  -- 0.9 MB files
  (medium : Nat) -- 0.8 MB files
  (small : Nat)  -- 0.5 MB files

/-- Represents the problem setup --/
def diskProblem : FileDistribution :=
  { large := 5
    medium := 15
    small := 20 }

/-- Disk capacity in MB --/
def diskCapacity : Rat := 2

/-- File sizes in MB --/
def largeFileSize : Rat := 9/10
def mediumFileSize : Rat := 4/5
def smallFileSize : Rat := 1/2

/-- The theorem stating the minimum number of disks required --/
theorem min_disks_required (fd : FileDistribution) 
  (h1 : fd.large + fd.medium + fd.small = 40)
  (h2 : fd = diskProblem) :
  ∃ (n : Nat), n = 18 ∧ 
  (∀ (m : Nat), m < n → 
    m * diskCapacity < 
    fd.large * largeFileSize + fd.medium * mediumFileSize + fd.small * smallFileSize) :=
  sorry

end NUMINAMATH_CALUDE_min_disks_required_l3347_334734


namespace NUMINAMATH_CALUDE_range_of_a_for_max_and_min_l3347_334706

/-- The cubic function with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 3*a*x^2 + 3*(a+2)*x + 1

/-- The derivative of f with respect to x -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 6*a*x + 3*(a+2)

/-- The discriminant of the quadratic equation f'(x) = 0 -/
def discriminant (a : ℝ) : ℝ := 36*a^2 - 36*(a+2)

/-- The condition for f to have both maximum and minimum -/
def has_max_and_min (a : ℝ) : Prop := discriminant a > 0

theorem range_of_a_for_max_and_min :
  ∀ a : ℝ, has_max_and_min a ↔ (a < -1 ∨ a > 2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_for_max_and_min_l3347_334706


namespace NUMINAMATH_CALUDE_congruence_problem_l3347_334775

theorem congruence_problem : ∃! n : ℤ, 0 ≤ n ∧ n ≤ 6 ∧ n ≡ 123456 [ZMOD 7] := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l3347_334775


namespace NUMINAMATH_CALUDE_quadratic_inequality_solutions_l3347_334702

theorem quadratic_inequality_solutions (d : ℝ) :
  d > 0 →
  (∃ x : ℝ, x^2 - 8*x + d < 0) ↔ d < 16 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solutions_l3347_334702


namespace NUMINAMATH_CALUDE_second_number_value_l3347_334709

theorem second_number_value (x y z : ℝ) 
  (sum_eq : x + y + z = 120)
  (ratio_xy : x / y = 3 / 4)
  (ratio_yz : y / z = 4 / 7) :
  y = 240 / 7 := by
sorry

end NUMINAMATH_CALUDE_second_number_value_l3347_334709


namespace NUMINAMATH_CALUDE_complement_A_in_U_l3347_334777

def U : Set ℕ := {x | 1 < x ∧ x < 5}
def A : Set ℕ := {2, 3}

theorem complement_A_in_U : (U \ A) = {4} := by sorry

end NUMINAMATH_CALUDE_complement_A_in_U_l3347_334777


namespace NUMINAMATH_CALUDE_negation_of_proposition_l3347_334753

theorem negation_of_proposition (A B : Set α) :
  ¬(∀ x, x ∈ A ∩ B → x ∈ A ∨ x ∈ B) ↔ (∃ x, x ∉ A ∩ B ∧ x ∉ A ∧ x ∉ B) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l3347_334753


namespace NUMINAMATH_CALUDE_v_3_equals_262_l3347_334762

/-- The polynomial f(x) -/
def f (x : ℝ) : ℝ := 7*x^7 + 6*x^6 + 5*x^5 + 4*x^4 + 3*x^3 + 2*x^2 + x

/-- The value of x -/
def x : ℝ := 3

/-- The value of v_3 using Horner's method for the first three terms -/
def v_3 : ℝ := ((7*x + 6)*x + 5)*x + 4

/-- Theorem stating that v_3 equals 262 -/
theorem v_3_equals_262 : v_3 = 262 := by
  sorry

end NUMINAMATH_CALUDE_v_3_equals_262_l3347_334762


namespace NUMINAMATH_CALUDE_red_balloons_count_l3347_334735

theorem red_balloons_count (total : ℕ) (green : ℕ) (h1 : total = 17) (h2 : green = 9) :
  total - green = 8 := by
  sorry

end NUMINAMATH_CALUDE_red_balloons_count_l3347_334735


namespace NUMINAMATH_CALUDE_four_primes_sum_l3347_334736

theorem four_primes_sum (p₁ p₂ p₃ p₄ : ℕ) : 
  Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ ∧ 
  p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₃ ≠ p₄ ∧
  (p₁ * p₂ * p₃ * p₄ ∣ 16^4 + 16^2 + 1) ∧
  p₁ + p₂ + p₃ + p₄ = 264 := by
  sorry

end NUMINAMATH_CALUDE_four_primes_sum_l3347_334736


namespace NUMINAMATH_CALUDE_swimming_speed_in_still_water_l3347_334739

/-- Proves that a person's swimming speed in still water is 4 km/h given the conditions -/
theorem swimming_speed_in_still_water 
  (water_speed : ℝ) 
  (distance : ℝ) 
  (time : ℝ) 
  (h1 : water_speed = 2)
  (h2 : distance = 8)
  (h3 : time = 4)
  (h4 : (swimming_speed - water_speed) * time = distance) :
  swimming_speed = 4 :=
by
  sorry

#check swimming_speed_in_still_water

end NUMINAMATH_CALUDE_swimming_speed_in_still_water_l3347_334739


namespace NUMINAMATH_CALUDE_dogwood_trees_planted_tomorrow_l3347_334784

/-- The number of dogwood trees planted tomorrow to reach the desired total -/
def trees_planted_tomorrow (initial_trees : ℕ) (planted_today : ℕ) (final_total : ℕ) : ℕ :=
  final_total - (initial_trees + planted_today)

/-- Theorem stating the number of trees planted tomorrow -/
theorem dogwood_trees_planted_tomorrow :
  trees_planted_tomorrow 39 41 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_dogwood_trees_planted_tomorrow_l3347_334784


namespace NUMINAMATH_CALUDE_fraction_addition_l3347_334758

theorem fraction_addition (d : ℝ) : (5 + 2 * d) / 8 + 3 = (29 + 2 * d) / 8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l3347_334758


namespace NUMINAMATH_CALUDE_cyclist_speed_l3347_334776

theorem cyclist_speed (speed : ℝ) : 
  (speed ≥ 0) →                           -- Non-negative speed
  (5 * speed + 5 * speed = 50) →          -- Total distance after 5 hours
  (speed = 5) :=                          -- Speed of each cyclist
by sorry

end NUMINAMATH_CALUDE_cyclist_speed_l3347_334776


namespace NUMINAMATH_CALUDE_greatest_two_digit_multiple_of_17_l3347_334728

theorem greatest_two_digit_multiple_of_17 : 
  ∀ n : ℕ, n ≤ 99 → n ≥ 10 → n % 17 = 0 → n ≤ 85 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_two_digit_multiple_of_17_l3347_334728


namespace NUMINAMATH_CALUDE_exp_gt_one_plus_x_l3347_334791

theorem exp_gt_one_plus_x : ∀ x : ℝ, x > 0 → Real.exp x > 1 + x := by sorry

end NUMINAMATH_CALUDE_exp_gt_one_plus_x_l3347_334791


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l3347_334780

theorem equal_roots_quadratic (k B : ℝ) : 
  k = 1 → (∃ x : ℝ, 2 * k * x^2 + B * x + 2 = 0 ∧ 
    ∀ y : ℝ, 2 * k * y^2 + B * y + 2 = 0 → y = x) → 
  B = 4 ∨ B = -4 := by
  sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l3347_334780


namespace NUMINAMATH_CALUDE_quilt_remaining_squares_l3347_334743

/-- Calculates the number of remaining squares to sew in a quilt -/
theorem quilt_remaining_squares (squares_per_side : ℕ) (sewn_percentage : ℚ) : 
  squares_per_side = 16 → sewn_percentage = 1/4 → 
  (2 * squares_per_side : ℚ) * (1 - sewn_percentage) = 24 := by
  sorry


end NUMINAMATH_CALUDE_quilt_remaining_squares_l3347_334743


namespace NUMINAMATH_CALUDE_ratio_change_l3347_334740

theorem ratio_change (x y : ℤ) (n : ℤ) : 
  y = 72 → x / y = 1 / 4 → (x + n) / y = 1 / 3 → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_ratio_change_l3347_334740


namespace NUMINAMATH_CALUDE_arccos_sum_equals_arcsin_l3347_334747

theorem arccos_sum_equals_arcsin (x : ℝ) : 
  Real.arccos x + Real.arccos (1 - x) = Real.arcsin x →
  (x = 0 ∨ x = 1 ∨ x = (1 + Real.sqrt 5) / 2) := by
sorry

end NUMINAMATH_CALUDE_arccos_sum_equals_arcsin_l3347_334747


namespace NUMINAMATH_CALUDE_angle_ratio_not_right_triangle_l3347_334757

/-- Triangle ABC with angles A, B, and C in the ratio 3:4:5 is not necessarily a right triangle -/
theorem angle_ratio_not_right_triangle (A B C : ℝ) : 
  A / B = 3 / 4 ∧ B / C = 4 / 5 ∧ A + B + C = π → 
  ¬ (A = π / 2 ∨ B = π / 2 ∨ C = π / 2) :=
by sorry

end NUMINAMATH_CALUDE_angle_ratio_not_right_triangle_l3347_334757


namespace NUMINAMATH_CALUDE_composite_product_quotient_l3347_334761

def first_five_composites : List ℕ := [21, 22, 24, 25, 26]
def next_five_composites : List ℕ := [27, 28, 30, 32, 33]

def product_list (l : List ℕ) : ℕ := l.foldl (·*·) 1

theorem composite_product_quotient :
  (product_list first_five_composites : ℚ) / (product_list next_five_composites) = 1 / 1964 := by
  sorry

end NUMINAMATH_CALUDE_composite_product_quotient_l3347_334761


namespace NUMINAMATH_CALUDE_sum_of_exponents_15_factorial_l3347_334749

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def largest_perfect_square_divisor (n : ℕ) : ℕ :=
  sorry

def prime_factors (n : ℕ) : List ℕ :=
  sorry

def exponents_of_prime_factors (n : ℕ) : List ℕ :=
  sorry

theorem sum_of_exponents_15_factorial :
  let n := factorial 15
  let largest_square := largest_perfect_square_divisor n
  let square_root := Nat.sqrt largest_square
  let exponents := exponents_of_prime_factors square_root
  List.sum exponents = 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_exponents_15_factorial_l3347_334749


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l3347_334796

/-- For a regular polygon with an exterior angle of 36°, the number of sides is 10. -/
theorem regular_polygon_sides (n : ℕ) (exterior_angle : ℝ) : 
  exterior_angle = 36 → n * exterior_angle = 360 → n = 10 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l3347_334796

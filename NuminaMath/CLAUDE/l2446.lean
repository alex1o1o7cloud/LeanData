import Mathlib

namespace f_intersects_positive_y_axis_l2446_244693

-- Define the function f(x) = x + 1
def f (x : ℝ) : ℝ := x + 1

-- Theorem stating that f intersects the y-axis at a point with positive y-coordinate
theorem f_intersects_positive_y_axis : ∃ (y : ℝ), y > 0 ∧ f 0 = y := by
  sorry

end f_intersects_positive_y_axis_l2446_244693


namespace tiles_needed_for_room_l2446_244619

/-- Proves that the number of 3-inch by 5-inch tiles needed to cover a 10-foot by 15-foot room is 1440 -/
theorem tiles_needed_for_room : 
  let room_length : ℚ := 10
  let room_width : ℚ := 15
  let tile_length : ℚ := 3 / 12  -- 3 inches in feet
  let tile_width : ℚ := 5 / 12   -- 5 inches in feet
  let room_area := room_length * room_width
  let tile_area := tile_length * tile_width
  let tiles_needed := room_area / tile_area
  ⌈tiles_needed⌉ = 1440 := by sorry

end tiles_needed_for_room_l2446_244619


namespace ratio_of_segments_l2446_244624

/-- Given five consecutive points on a straight line, prove the ratio of two segments --/
theorem ratio_of_segments (a b c d e : ℝ) : 
  (b < c) ∧ (c < d) ∧  -- Consecutive points
  (e - d = 8) ∧        -- de = 8
  (b - a = 5) ∧        -- ab = 5
  (c - a = 11) ∧       -- ac = 11
  (e - a = 21)         -- ae = 21
  → (c - b) / (d - c) = 3 / 1 := by
sorry

end ratio_of_segments_l2446_244624


namespace largest_prime_with_2023_digits_p_is_prime_p_has_2023_digits_smallest_k_divisible_by_30_l2446_244675

/-- The largest prime with 2023 digits -/
def p : ℕ := sorry

theorem largest_prime_with_2023_digits (q : ℕ) (h : q > p) : ¬ Prime q := sorry

theorem p_is_prime : Prime p := sorry

theorem p_has_2023_digits : (Nat.digits 10 p).length = 2023 := sorry

theorem smallest_k_divisible_by_30 : 
  ∃ k : ℕ, k > 0 ∧ 30 ∣ (p^3 - k) ∧ ∀ m : ℕ, 0 < m ∧ m < k → ¬(30 ∣ (p^3 - m)) :=
sorry

end largest_prime_with_2023_digits_p_is_prime_p_has_2023_digits_smallest_k_divisible_by_30_l2446_244675


namespace integral_of_f_minus_x_l2446_244656

/-- Given a function f: ℝ → ℝ such that f'(x) = 2x + 1 for all x ∈ ℝ,
    prove that the definite integral of f(-x) from -1 to 3 equals 14/3. -/
theorem integral_of_f_minus_x (f : ℝ → ℝ) (h : ∀ x, deriv f x = 2 * x + 1) :
  ∫ x in (-1)..(3), f (-x) = 14/3 := by
  sorry

end integral_of_f_minus_x_l2446_244656


namespace johns_house_nails_l2446_244686

/-- The number of nails needed for a house wall -/
def total_nails (large_planks small_planks large_nails small_nails : ℕ) : ℕ :=
  large_nails + small_nails

/-- Theorem stating the total number of nails needed for John's house wall -/
theorem johns_house_nails :
  total_nails 12 10 15 5 = 20 := by
  sorry

end johns_house_nails_l2446_244686


namespace binomial_square_simplification_l2446_244646

theorem binomial_square_simplification (m n p : ℝ) :
  ¬(∃ a b, (-m - n) * (m + n) = a^2 - b^2) ∧
  (∃ a b, (-m - n) * (-m + n) = a^2 - b^2) ∧
  (∃ a b, (m * n + p) * (m * n - p) = a^2 - b^2) ∧
  (∃ a b, (0.3 * m - n) * (-n - 0.3 * m) = a^2 - b^2) :=
by sorry

end binomial_square_simplification_l2446_244646


namespace john_running_speed_equation_l2446_244669

theorem john_running_speed_equation :
  ∃ (x : ℝ), x > 0 ∧ 6.6 * x^2 - 31.6 * x - 16 = 0 := by
  sorry

end john_running_speed_equation_l2446_244669


namespace last_three_digits_of_5_to_15000_l2446_244650

theorem last_three_digits_of_5_to_15000 (h : 5^500 ≡ 1 [ZMOD 1000]) :
  5^15000 ≡ 1 [ZMOD 1000] := by
  sorry

end last_three_digits_of_5_to_15000_l2446_244650


namespace prime_pairs_dividing_sum_of_powers_l2446_244651

theorem prime_pairs_dividing_sum_of_powers (p q : Nat) : 
  Prime p → Prime q → (p * q ∣ 3^p + 3^q) ↔ 
  ((p = 2 ∧ q = 3) ∨ (p = 3 ∧ q = 2) ∨ (p = 3 ∧ q = 3) ∨ 
   (p = 3 ∧ q = 5) ∨ (p = 5 ∧ q = 3)) := by
  sorry

end prime_pairs_dividing_sum_of_powers_l2446_244651


namespace parabola_min_value_l2446_244680

theorem parabola_min_value (x y : ℝ) : 
  y^2 = 4*x → (∀ x' y' : ℝ, y'^2 = 4*x' → 1/2 * y'^2 + x'^2 + 3 ≥ 1/2 * y^2 + x^2 + 3) → 
  1/2 * y^2 + x^2 + 3 = 3 := by
sorry

end parabola_min_value_l2446_244680


namespace bird_migration_difference_l2446_244623

theorem bird_migration_difference (migrating_families : ℕ) (remaining_families : ℕ)
  (avg_birds_migrating : ℕ) (avg_birds_remaining : ℕ)
  (h1 : migrating_families = 86)
  (h2 : remaining_families = 45)
  (h3 : avg_birds_migrating = 12)
  (h4 : avg_birds_remaining = 8) :
  migrating_families * avg_birds_migrating - remaining_families * avg_birds_remaining = 672 := by
  sorry

end bird_migration_difference_l2446_244623


namespace smallest_b_in_arithmetic_sequence_l2446_244697

theorem smallest_b_in_arithmetic_sequence (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →  -- positive terms
  ∃ d : ℝ, a = b - d ∧ c = b + d →  -- arithmetic sequence
  a * b * c = 216 →  -- product condition
  b ≥ 6 ∧ (∀ x : ℝ, x > 0 ∧ (∃ y z : ℝ, y > 0 ∧ z > 0 ∧ 
    (∃ e : ℝ, y = x - e ∧ z = x + e) ∧ 
    y * x * z = 216) → x ≥ 6) :=
by sorry

end smallest_b_in_arithmetic_sequence_l2446_244697


namespace expression_simplification_and_evaluation_l2446_244673

/-- The original expression as a function of x and square -/
def original_expr (x : ℝ) (square : ℝ) : ℝ :=
  (3 - 2*x^2 - 5*x) - (square*x^2 + 3*x - 4)

/-- The simplified expression as a function of x and square -/
def simplified_expr (x : ℝ) (square : ℝ) : ℝ :=
  (-2 - square)*x^2 - 8*x + 7

theorem expression_simplification_and_evaluation :
  ∀ (x : ℝ) (square : ℝ),
  /- 1. The simplified form is correct -/
  original_expr x square = simplified_expr x square ∧
  /- 2. When x=-2 and square=-2, the expression evaluates to -17 -/
  original_expr (-2) (-2) = -17 ∧
  /- 3. The value of square that eliminates the quadratic term is -2 -/
  ∃ (square : ℝ), (-2 - square) = 0 ∧ square = -2 := by
  sorry

end expression_simplification_and_evaluation_l2446_244673


namespace proposition_equivalence_l2446_244612

-- Define the concept of angles
def Angle : Type := ℝ

-- Define what it means for two angles to be equal
def equal_angles (a b : Angle) : Prop := a = b

-- Define what it means for two angles to be vertical angles
def vertical_angles (a b : Angle) : Prop := sorry

-- State the original proposition
def original_proposition : Prop :=
  ∀ a b : Angle, equal_angles a b → vertical_angles a b

-- State the conditional form
def conditional_form : Prop :=
  ∀ a b : Angle, vertical_angles a b → equal_angles a b

-- Theorem stating the equivalence of the two forms
theorem proposition_equivalence : original_proposition ↔ conditional_form :=
  sorry

end proposition_equivalence_l2446_244612


namespace arithmetic_equality_l2446_244643

theorem arithmetic_equality : 3889 + 12.808 - 47.80600000000004 = 3854.002 := by
  sorry

end arithmetic_equality_l2446_244643


namespace greatest_solution_of_equation_l2446_244658

theorem greatest_solution_of_equation (x : Real) : 
  x ∈ Set.Icc 0 (10 * Real.pi) →
  |2 * Real.sin x - 1| + |2 * Real.cos (2 * x) - 1| = 0 →
  x ≤ 61 * Real.pi / 6 ∧ 
  ∃ y ∈ Set.Icc 0 (10 * Real.pi), 
    |2 * Real.sin y - 1| + |2 * Real.cos (2 * y) - 1| = 0 ∧
    y = 61 * Real.pi / 6 :=
by sorry

end greatest_solution_of_equation_l2446_244658


namespace percentage_relationship_l2446_244611

theorem percentage_relationship (x y : ℝ) (c : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h1 : x = 2.5 * y) (h2 : 2 * y = c / 100 * x) : c = 80 := by
  sorry

end percentage_relationship_l2446_244611


namespace barbells_bought_l2446_244672

theorem barbells_bought (amount_given : ℕ) (change_received : ℕ) (cost_per_barbell : ℕ) : 
  amount_given = 850 → change_received = 40 → cost_per_barbell = 270 → 
  (amount_given - change_received) / cost_per_barbell = 3 :=
by sorry

end barbells_bought_l2446_244672


namespace log_equation_solution_l2446_244699

theorem log_equation_solution (b x : ℝ) (hb_pos : b > 0) (hb_neq_one : b ≠ 1) (hx_neq_one : x ≠ 1) :
  (Real.log x) / (Real.log (b^3)) + (Real.log b) / (Real.log (x^3)) = 3 →
  x = Real.exp ((9 + Real.sqrt 77) * Real.log b / 2) ∨
  x = Real.exp ((9 - Real.sqrt 77) * Real.log b / 2) := by
sorry

end log_equation_solution_l2446_244699


namespace sean_net_profit_l2446_244663

/-- Represents the pricing tiers for patches --/
inductive PricingTier
  | small
  | medium
  | large
  | xlarge

/-- Calculates the price per patch based on the pricing tier --/
def price_per_patch (tier : PricingTier) : ℚ :=
  match tier with
  | .small => 12
  | .medium => 11.5
  | .large => 11
  | .xlarge => 10.5

/-- Represents a sale of patches --/
structure Sale :=
  (quantity : ℕ)
  (customers : ℕ)
  (tier : PricingTier)

/-- Calculates the total cost for ordering patches --/
def total_cost (patches : ℕ) : ℚ :=
  let units := (patches + 99) / 100  -- Round up to nearest 100
  1.25 * patches + 20 * units

/-- Calculates the revenue from a sale --/
def sale_revenue (sale : Sale) : ℚ :=
  sale.quantity * sale.customers * price_per_patch sale.tier

/-- Calculates the total revenue from all sales --/
def total_revenue (sales : List Sale) : ℚ :=
  sales.map sale_revenue |> List.sum

/-- The main theorem stating Sean's net profit --/
theorem sean_net_profit (sales : List Sale) 
  (h_sales : sales = [
    {quantity := 15, customers := 5, tier := .small},
    {quantity := 50, customers := 2, tier := .medium},
    {quantity := 25, customers := 1, tier := .large}
  ]) : 
  total_revenue sales - total_cost (sales.map (λ s => s.quantity * s.customers) |> List.sum) = 2035 := by
  sorry


end sean_net_profit_l2446_244663


namespace set_A_equals_zero_to_three_l2446_244664

def A : Set ℤ := {x | x^2 - 3*x - 4 < 0}

theorem set_A_equals_zero_to_three : A = {0, 1, 2, 3} := by
  sorry

end set_A_equals_zero_to_three_l2446_244664


namespace triangulated_rectangle_has_36_triangles_l2446_244690

/-- Represents a rectangle divided into triangles -/
structure TriangulatedRectangle where
  smallest_triangles : ℕ
  has_isosceles_triangles : Bool
  has_large_right_triangles : Bool

/-- Counts the total number of triangles in a triangulated rectangle -/
def count_triangles (rect : TriangulatedRectangle) : ℕ :=
  sorry

/-- Theorem: A rectangle divided into 16 smallest right triangles contains 36 total triangles -/
theorem triangulated_rectangle_has_36_triangles :
  ∀ (rect : TriangulatedRectangle),
    rect.smallest_triangles = 16 →
    rect.has_isosceles_triangles = true →
    rect.has_large_right_triangles = true →
    count_triangles rect = 36 :=
  sorry

end triangulated_rectangle_has_36_triangles_l2446_244690


namespace simplify_complex_fraction_l2446_244640

theorem simplify_complex_fraction :
  (1 / ((2 / (Real.sqrt 5 + 2)) + (3 / (Real.sqrt 7 - 2)))) =
  ((2 * Real.sqrt 5 + Real.sqrt 7 + 2) / (23 + 4 * Real.sqrt 35)) := by
  sorry

end simplify_complex_fraction_l2446_244640


namespace winnie_lollipops_l2446_244617

theorem winnie_lollipops (total_lollipops : ℕ) (num_friends : ℕ) (h1 : total_lollipops = 400) (h2 : num_friends = 13) :
  total_lollipops - (num_friends * (total_lollipops / num_friends)) = 10 := by
  sorry

end winnie_lollipops_l2446_244617


namespace gayle_bicycle_ride_l2446_244667

/-- Gayle's bicycle ride problem -/
theorem gayle_bicycle_ride 
  (sunny_speed : ℝ) 
  (rainy_speed : ℝ) 
  (total_distance : ℝ) 
  (total_time : ℝ) 
  (h1 : sunny_speed = 40)
  (h2 : rainy_speed = 25)
  (h3 : total_distance = 20)
  (h4 : total_time = 50/60) -- Convert 50 minutes to hours
  : ∃ (rainy_time : ℝ), 
    rainy_time = 32/60 ∧ -- Convert 32 minutes to hours
    rainy_time * rainy_speed + (total_time - rainy_time) * sunny_speed = total_distance :=
by
  sorry


end gayle_bicycle_ride_l2446_244667


namespace tan_plus_cot_l2446_244631

theorem tan_plus_cot (α : ℝ) (h : Real.sin (2 * α) = 3 / 4) :
  Real.tan α + (Real.tan α)⁻¹ = 8 / 3 := by
  sorry

end tan_plus_cot_l2446_244631


namespace sine_cosine_difference_equals_half_l2446_244648

theorem sine_cosine_difference_equals_half : 
  Real.sin (43 * π / 180) * Real.cos (13 * π / 180) - 
  Real.cos (43 * π / 180) * Real.sin (13 * π / 180) = 1 / 2 := by
  sorry

end sine_cosine_difference_equals_half_l2446_244648


namespace intersection_and_union_when_a_is_5_intersection_equals_b_iff_a_in_range_l2446_244676

def A : Set ℝ := {x | 1 < x - 1 ∧ x - 1 < 3}
def B (a : ℝ) : Set ℝ := {x | (x - 3) * (x - a) < 0}

theorem intersection_and_union_when_a_is_5 :
  (A ∩ B 5 = {x | 3 < x ∧ x < 4}) ∧
  (A ∪ B 5 = {x | 2 < x ∧ x < 5}) := by sorry

theorem intersection_equals_b_iff_a_in_range :
  ∀ a : ℝ, A ∩ B a = B a ↔ 2 ≤ a ∧ a ≤ 4 := by sorry

end intersection_and_union_when_a_is_5_intersection_equals_b_iff_a_in_range_l2446_244676


namespace seventh_root_of_137858491849_l2446_244662

theorem seventh_root_of_137858491849 : 
  (137858491849 : ℝ) ^ (1/7 : ℝ) = 11 := by sorry

end seventh_root_of_137858491849_l2446_244662


namespace min_value_and_inequality_solution_l2446_244600

theorem min_value_and_inequality_solution :
  ∃ m : ℝ,
    (∀ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 →
      (1 / a^3 + 1 / b^3 + 1 / c^3 + 27 * a * b * c) ≥ m) ∧
    (∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧
      (1 / a^3 + 1 / b^3 + 1 / c^3 + 27 * a * b * c) = m) ∧
    m = 18 ∧
    (∀ x : ℝ, |x + 1| - 2 * x < m ↔ x > -19/3) :=
by sorry

end min_value_and_inequality_solution_l2446_244600


namespace pauls_cousin_score_l2446_244684

theorem pauls_cousin_score (paul_score : ℕ) (total_score : ℕ) 
  (h1 : paul_score = 3103)
  (h2 : total_score = 5816) :
  total_score - paul_score = 2713 :=
by sorry

end pauls_cousin_score_l2446_244684


namespace yanni_found_money_l2446_244657

/-- The amount of money Yanni found at the mall -/
def money_found (initial_money mother_gave toy_cost money_left : ℚ) : ℚ :=
  (toy_cost + money_left) - (initial_money + mother_gave)

/-- Theorem stating how much money Yanni found at the mall -/
theorem yanni_found_money : 
  money_found 0.85 0.40 1.60 0.15 = 0.50 := by sorry

end yanni_found_money_l2446_244657


namespace pet_shop_dogs_l2446_244642

theorem pet_shop_dogs (dogs cats bunnies : ℕ) : 
  dogs + cats + bunnies > 0 →
  dogs * 7 = cats * 4 →
  dogs * 9 = bunnies * 4 →
  dogs + bunnies = 364 →
  dogs = 112 := by
sorry

end pet_shop_dogs_l2446_244642


namespace phone_repair_cost_l2446_244689

theorem phone_repair_cost (laptop_cost computer_cost : ℕ) 
  (phone_repairs laptop_repairs computer_repairs : ℕ) (total_earnings : ℕ) :
  laptop_cost = 15 →
  computer_cost = 18 →
  phone_repairs = 5 →
  laptop_repairs = 2 →
  computer_repairs = 2 →
  total_earnings = 121 →
  ∃ (phone_cost : ℕ), 
    phone_cost * phone_repairs + 
    laptop_cost * laptop_repairs + 
    computer_cost * computer_repairs = total_earnings ∧
    phone_cost = 11 :=
by sorry

end phone_repair_cost_l2446_244689


namespace p_iff_a_in_range_exactly_one_true_iff_a_in_range_l2446_244692

-- Define the propositions and conditions
def p (a : ℝ) : Prop := ∃ x : ℝ, -1 < x ∧ x < 1 ∧ x^2 - (a+2)*x + 2*a = 0

def q (m : ℝ) : Prop := ∃ x₁ x₂ : ℝ, 
  x₁^2 - 2*m*x₁ - 3 = 0 ∧ 
  x₂^2 - 2*m*x₂ - 3 = 0 ∧ 
  x₁ ≠ x₂

def inequality_holds (a m : ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, q m → a^2 - 3*a ≥ |x₁ - x₂|

-- State the theorems
theorem p_iff_a_in_range (a : ℝ) : 
  p a ↔ -1 < a ∧ a < 1 :=
sorry

theorem exactly_one_true_iff_a_in_range (a : ℝ) : 
  (p a ∧ ∀ m : ℝ, -1 ≤ m ∧ m ≤ 1 → ¬(q m ∧ inequality_holds a m)) ∨
  (¬p a ∧ ∃ m : ℝ, -1 ≤ m ∧ m ≤ 1 ∧ q m ∧ inequality_holds a m)
  ↔ 
  a < 1 ∨ a ≥ 4 :=
sorry

end p_iff_a_in_range_exactly_one_true_iff_a_in_range_l2446_244692


namespace smallest_number_in_sequence_l2446_244694

theorem smallest_number_in_sequence (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- positive integers
  (a + b + c) / 3 = 30 →  -- arithmetic mean is 30
  b = 28 →  -- median is 28
  c = b + 6 →  -- largest number is 6 more than median
  a ≤ b ∧ a ≤ c →  -- a is the smallest number
  a = 28 :=
by sorry

end smallest_number_in_sequence_l2446_244694


namespace square_park_fencing_cost_l2446_244616

/-- The total cost of fencing a square-shaped park -/
theorem square_park_fencing_cost (cost_per_side : ℕ) (h : cost_per_side = 72) : 
  cost_per_side * 4 = 288 := by
  sorry

#check square_park_fencing_cost

end square_park_fencing_cost_l2446_244616


namespace square_root_division_problem_l2446_244681

theorem square_root_division_problem : ∃ x : ℝ, (Real.sqrt 5184) / x = 4 ∧ x = 18 := by
  sorry

end square_root_division_problem_l2446_244681


namespace tangent_line_distance_l2446_244659

/-- The curve function -/
def f (x : ℝ) : ℝ := -x^3 + 2*x

/-- The derivative of the curve function -/
def f' (x : ℝ) : ℝ := -3*x^2 + 2

/-- The x-coordinate of the tangent point -/
def x₀ : ℝ := -1

/-- The y-coordinate of the tangent point -/
def y₀ : ℝ := f x₀

/-- The slope of the tangent line -/
def m : ℝ := f' x₀

/-- The point we're measuring the distance from -/
def P : ℝ × ℝ := (3, 2)

theorem tangent_line_distance :
  let A : ℝ := 1
  let B : ℝ := 1
  let C : ℝ := -(m * x₀ - y₀)
  (A * P.1 + B * P.2 + C) / Real.sqrt (A^2 + B^2) = 7 * Real.sqrt 2 / 2 := by
  sorry

end tangent_line_distance_l2446_244659


namespace tangent_slope_implies_a_range_l2446_244635

open Real

theorem tangent_slope_implies_a_range (a : ℝ) :
  (∀ x ∈ Set.Ioo 1 2, (a / x - 2 * (x - 1)) > 1) →
  a ≥ 6 := by
  sorry

end tangent_slope_implies_a_range_l2446_244635


namespace yellow_balls_count_l2446_244683

def total_balls : ℕ := 1500

def red_balls : ℕ := (2 * total_balls) / 7

def remaining_after_red : ℕ := total_balls - red_balls

def blue_balls : ℕ := (3 * remaining_after_red) / 11

def remaining_after_blue : ℕ := remaining_after_red - blue_balls

def green_balls : ℕ := remaining_after_blue / 5

def remaining_after_green : ℕ := remaining_after_blue - green_balls

def orange_balls : ℕ := remaining_after_green / 8

def yellow_balls : ℕ := remaining_after_green - orange_balls

theorem yellow_balls_count : yellow_balls = 546 := by
  sorry

end yellow_balls_count_l2446_244683


namespace unique_solution_to_equation_l2446_244627

theorem unique_solution_to_equation :
  ∃! x : ℝ, x > 0 ∧ x^(Real.log x / Real.log 5) = x^2 / 25 := by
  sorry

end unique_solution_to_equation_l2446_244627


namespace polynomial_degree_l2446_244637

/-- The degree of the polynomial 3 + 7x^2 + (1/2)x^5 - 10x + 11 is 5 -/
theorem polynomial_degree : 
  let p : Polynomial ℚ := 3 + 7 * X^2 + (1/2) * X^5 - 10 * X + 11
  Polynomial.degree p = 5 := by sorry

end polynomial_degree_l2446_244637


namespace fibonacci_periodicity_l2446_244603

-- Define p-arithmetic system
class PArithmetic (p : ℕ) where
  sqrt5_extractable : ∃ x, x^2 = 5
  fermat_little : ∀ a : ℤ, a ≠ 0 → a^(p-1) ≡ 1 [ZMOD p]

-- Define Fibonacci sequence
def fibonacci (v₀ v₁ : ℤ) : ℕ → ℤ
| 0 => v₀
| 1 => v₁
| (n+2) => fibonacci v₀ v₁ n + fibonacci v₀ v₁ (n+1)

-- Theorem statement
theorem fibonacci_periodicity {p : ℕ} [PArithmetic p] (v₀ v₁ : ℤ) :
  ∀ k : ℕ, fibonacci v₀ v₁ (k + p - 1) = fibonacci v₀ v₁ k :=
sorry

end fibonacci_periodicity_l2446_244603


namespace max_non_overlapping_ge_min_covering_l2446_244625

/-- A polygon in a 2D plane -/
structure Polygon where
  -- Add necessary fields for a polygon

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if a circle's center is inside a polygon -/
def Circle.centerInside (c : Circle) (p : Polygon) : Prop :=
  sorry

/-- Checks if two circles are non-overlapping -/
def Circle.nonOverlapping (c1 c2 : Circle) : Prop :=
  sorry

/-- Checks if a set of circles covers a polygon -/
def covers (circles : Set Circle) (p : Polygon) : Prop :=
  sorry

/-- The maximum number of non-overlapping circles of diameter 1 with centers inside the polygon -/
def maxNonOverlappingCircles (p : Polygon) : ℕ :=
  sorry

/-- The minimum number of circles of radius 1 that can cover the polygon -/
def minCoveringCircles (p : Polygon) : ℕ :=
  sorry

/-- Theorem: The maximum number of non-overlapping circles of diameter 1 with centers inside a polygon
    is greater than or equal to the minimum number of circles of radius 1 needed to cover the polygon -/
theorem max_non_overlapping_ge_min_covering (p : Polygon) :
  maxNonOverlappingCircles p ≥ minCoveringCircles p :=
sorry

end max_non_overlapping_ge_min_covering_l2446_244625


namespace two_tangents_from_three_zero_l2446_244682

/-- The curve y = x^2 - 2x -/
def curve (x : ℝ) : ℝ := x^2 - 2*x

/-- Condition for two tangents to exist from a point (a, b) to the curve -/
def two_tangents_condition (a b : ℝ) : Prop :=
  a^2 - 2*a - b > 0

/-- Theorem stating that (3, 0) satisfies the two tangents condition -/
theorem two_tangents_from_three_zero :
  two_tangents_condition 3 0 := by
  sorry

end two_tangents_from_three_zero_l2446_244682


namespace triangle_side_lengths_l2446_244622

theorem triangle_side_lengths 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h_area : (1/2) * a * b * Real.sin C = 10 * Real.sqrt 3)
  (h_sum : a + b = 13)
  (h_angle : C = π/3) :
  ((a = 5 ∧ b = 8 ∧ c = 7) ∨ (a = 8 ∧ b = 5 ∧ c = 7)) :=
by sorry

end triangle_side_lengths_l2446_244622


namespace system_is_linear_l2446_244639

/-- A linear equation in two variables -/
structure LinearEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A system of two linear equations -/
structure LinearSystem where
  eq1 : LinearEquation
  eq2 : LinearEquation

/-- The specific system of equations we want to prove is linear -/
def system : LinearSystem := {
  eq1 := { a := 1, b := 0, c := 1 }  -- Represents x = 1
  eq2 := { a := 3, b := -2, c := 6 } -- Represents 3x - 2y = 6
}

/-- Theorem stating that our system is indeed a system of two linear equations -/
theorem system_is_linear : ∃ (s : LinearSystem), s = system := by sorry

end system_is_linear_l2446_244639


namespace sum_of_three_numbers_l2446_244695

theorem sum_of_three_numbers (a b c : ℤ) 
  (h1 : a + b = 31) 
  (h2 : b + c = 47) 
  (h3 : c + a = 54) : 
  a + b + c = 66 := by
sorry

end sum_of_three_numbers_l2446_244695


namespace parallelism_sufficiency_not_necessity_l2446_244677

/-- Two lines are parallel if their slopes are equal -/
def are_parallel (m₁ n₁ c₁ m₂ n₂ c₂ : ℝ) : Prop :=
  m₁ * n₂ = m₂ * n₁ ∧ m₁ ≠ 0 ∧ m₂ ≠ 0

/-- The condition for parallelism of the given lines -/
def parallelism_condition (a : ℝ) : Prop :=
  are_parallel 2 a (-2) (a + 1) 1 (-a)

theorem parallelism_sufficiency_not_necessity :
  (∀ a : ℝ, a = 1 → parallelism_condition a) ∧
  ¬(∀ a : ℝ, parallelism_condition a → a = 1) :=
by sorry

end parallelism_sufficiency_not_necessity_l2446_244677


namespace shopkeeper_cloth_cost_price_l2446_244649

/-- Given a shopkeeper who sells cloth at a loss, calculate the cost price per metre. -/
theorem shopkeeper_cloth_cost_price
  (total_metres : ℕ)
  (selling_price : ℕ)
  (loss_per_metre : ℕ)
  (h1 : total_metres = 500)
  (h2 : selling_price = 18000)
  (h3 : loss_per_metre = 5) :
  (selling_price + total_metres * loss_per_metre) / total_metres = 41 := by
sorry

end shopkeeper_cloth_cost_price_l2446_244649


namespace prime_condition_l2446_244613

theorem prime_condition (p : ℕ) : 
  Nat.Prime p ∧ Nat.Prime (p^4 - 3*p^2 + 9) → p = 2 := by
  sorry

end prime_condition_l2446_244613


namespace triangle_abc_theorem_l2446_244654

theorem triangle_abc_theorem (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- a, b, c are sides opposite to angles A, B, C
  a > 0 ∧ b > 0 ∧ c > 0 →
  -- Given equation
  Real.cos (2 * A) + 2 * Real.sin (π + B) ^ 2 + 2 * Real.cos (π / 2 + C) ^ 2 - 1 = 2 * Real.sin B * Real.sin C →
  -- Given side lengths
  b = 4 ∧ c = 5 →
  -- Conclusions
  A = π / 3 ∧ Real.sin B = 2 * Real.sqrt 7 / 7 := by
  sorry

end triangle_abc_theorem_l2446_244654


namespace largest_digit_sum_l2446_244607

theorem largest_digit_sum (a b c z : ℕ) : 
  (a < 10 ∧ b < 10 ∧ c < 10) →  -- a, b, c are digits
  (100 * a + 10 * b + c = 1000 / z) →  -- 0.abc = 1/z
  (0 < z ∧ z ≤ 12) →  -- 0 < z ≤ 12
  (∃ (x y w : ℕ), x + y + w ≤ 8 ∧ 
    (100 * x + 10 * y + w = 1000 / z) ∧ 
    (x < 10 ∧ y < 10 ∧ w < 10)) →
  a + b + c ≤ 8 :=
sorry

end largest_digit_sum_l2446_244607


namespace quadratic_two_roots_l2446_244628

theorem quadratic_two_roots 
  (a b c α : ℝ) 
  (f : ℝ → ℝ) 
  (h_quad : ∀ x, f x = a * x^2 + b * x + c) 
  (h_exists : a * f α < 0) : 
  ∃ x₁ x₂, x₁ < α ∧ α < x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ ≠ x₂ := by
sorry

end quadratic_two_roots_l2446_244628


namespace sum_of_roots_l2446_244634

theorem sum_of_roots (a b : ℝ) : 
  a * (a - 4) = 12 → b * (b - 4) = 12 → a ≠ b → a + b = 4 := by
sorry

end sum_of_roots_l2446_244634


namespace triangle_is_right_angle_l2446_244674

theorem triangle_is_right_angle (u : ℝ) 
  (h1 : 0 < 3*u - 2) 
  (h2 : 0 < 3*u + 2) 
  (h3 : 0 < 6*u) : 
  (3*u - 2) + (3*u + 2) = 6*u := by
  sorry

end triangle_is_right_angle_l2446_244674


namespace prob_exactly_one_correct_l2446_244679

variable (p₁ p₂ : ℝ)

-- A and B independently solve the same problem
axiom prob_A : 0 ≤ p₁ ∧ p₁ ≤ 1
axiom prob_B : 0 ≤ p₂ ∧ p₂ ≤ 1

-- The probability that exactly one person solves the problem
def prob_exactly_one : ℝ := p₁ * (1 - p₂) + p₂ * (1 - p₁)

-- Theorem stating that the probability of exactly one person solving is correct
theorem prob_exactly_one_correct :
  prob_exactly_one p₁ p₂ = p₁ * (1 - p₂) + p₂ * (1 - p₁) :=
by sorry

end prob_exactly_one_correct_l2446_244679


namespace max_value_x_plus_2y_l2446_244688

theorem max_value_x_plus_2y (x y : ℝ) (h : 3 * (x^2 + y^2) = x + y) : 
  ∃ (max : ℝ), max = 3 ∧ ∀ (a b : ℝ), 3 * (a^2 + b^2) = a + b → a + 2*b ≤ max :=
sorry

end max_value_x_plus_2y_l2446_244688


namespace average_rate_of_change_average_rate_of_change_on_interval_l2446_244630

def f (x : ℝ) : ℝ := 2 * x + 1

theorem average_rate_of_change (a b : ℝ) (h : a < b) :
  (f b - f a) / (b - a) = 2 :=
by sorry

theorem average_rate_of_change_on_interval :
  (f 2 - f 1) / (2 - 1) = 2 :=
by sorry

end average_rate_of_change_average_rate_of_change_on_interval_l2446_244630


namespace frustum_slant_height_l2446_244678

theorem frustum_slant_height (r₁ r₂ : ℝ) (h : r₁ = 2 ∧ r₂ = 5) :
  let l := (π * (r₁^2 + r₂^2)) / (π * (r₁ + r₂))
  l = 29 / 7 := by
  sorry

end frustum_slant_height_l2446_244678


namespace parabola_translation_l2446_244666

-- Define the original and transformed parabolas
def original_parabola (x : ℝ) : ℝ := x^2
def transformed_parabola (x : ℝ) : ℝ := x^2 - 5

-- Define the translation
def translation (y : ℝ) : ℝ := y - 5

-- Theorem statement
theorem parabola_translation :
  ∀ x : ℝ, transformed_parabola x = translation (original_parabola x) := by
  sorry

end parabola_translation_l2446_244666


namespace divisor_and_equation_l2446_244636

theorem divisor_and_equation (k : ℕ) : 
  (∃ n : ℕ, n * (18^k) = 1) → (6^k - k^6 = 1 ↔ k = 0) := by
sorry

end divisor_and_equation_l2446_244636


namespace parabola_line_intersection_l2446_244647

/-- Given a parabola and a line passing through its focus, 
    prove the value of p when the triangle area is 4 -/
theorem parabola_line_intersection (p : ℝ) : 
  let parabola := fun (x y : ℝ) => x^2 = 2*p*y
  let focus := (0, p/2)
  let line := fun (x y : ℝ) => y = Real.sqrt 3 * x + p/2
  let origin := (0, 0)
  let triangle_area (A B : ℝ × ℝ) := 
    abs ((A.1 - origin.1) * (B.2 - origin.2) - (B.1 - origin.1) * (A.2 - origin.2)) / 2
  ∃ (A B : ℝ × ℝ), 
    parabola A.1 A.2 ∧ 
    parabola B.1 B.2 ∧ 
    line A.1 A.2 ∧ 
    line B.1 B.2 ∧ 
    triangle_area A B = 4 →
    p = 2 * Real.sqrt 2 ∨ p = -2 * Real.sqrt 2 := by
  sorry

end parabola_line_intersection_l2446_244647


namespace cube_volume_from_face_perimeter_l2446_244602

theorem cube_volume_from_face_perimeter (face_perimeter : ℝ) (h : face_perimeter = 28) :
  let side_length := face_perimeter / 4
  side_length ^ 3 = 343 := by sorry

end cube_volume_from_face_perimeter_l2446_244602


namespace company_stores_l2446_244618

theorem company_stores (total_uniforms : ℕ) (uniforms_per_store : ℕ) (h1 : total_uniforms = 121) (h2 : uniforms_per_store = 4) :
  (total_uniforms / uniforms_per_store : ℕ) = 30 :=
by sorry

end company_stores_l2446_244618


namespace distributor_cost_distributor_cost_proof_l2446_244610

/-- The cost of an item for a distributor given online store commission, desired profit, and observed price. -/
theorem distributor_cost (commission_rate : ℝ) (profit_rate : ℝ) (observed_price : ℝ) : ℝ :=
  let selling_price := observed_price / (1 - commission_rate)
  let cost := selling_price / (1 + profit_rate)
  cost

/-- Proof that the distributor's cost is $28.125 given the specified conditions. -/
theorem distributor_cost_proof :
  distributor_cost 0.2 0.2 27 = 28.125 :=
by sorry

end distributor_cost_distributor_cost_proof_l2446_244610


namespace perpendicular_lines_b_value_l2446_244691

theorem perpendicular_lines_b_value (b : ℝ) : 
  let v1 : Fin 2 → ℝ := ![4, -1]
  let v2 : Fin 2 → ℝ := ![b, 8]
  (∀ i, v1 i * v2 i = 0) → b = 2 := by
sorry

end perpendicular_lines_b_value_l2446_244691


namespace min_value_of_fraction_sum_l2446_244644

theorem min_value_of_fraction_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (2 / x + 1 / y) ≥ 2 * Real.sqrt 2 := by
  sorry

end min_value_of_fraction_sum_l2446_244644


namespace S_is_open_line_segment_l2446_244620

-- Define the set of points satisfying the conditions
def S : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 + p.2 = 1 ∧ p.1^2 + p.2^2 < 25}

-- Theorem statement
theorem S_is_open_line_segment :
  ∃ (a b : ℝ × ℝ), a ≠ b ∧
    S = {p : ℝ × ℝ | ∃ (t : ℝ), 0 < t ∧ t < 1 ∧ p = (1 - t) • a + t • b} :=
sorry

end S_is_open_line_segment_l2446_244620


namespace expression_evaluation_l2446_244605

theorem expression_evaluation :
  let a : ℕ := 3
  let b : ℕ := 2
  let c : ℕ := 1
  ((a^2 + b*c) + (a*b + c))^2 - ((a^2 + b*c) - (a*b + c))^2 = 308 := by
  sorry

end expression_evaluation_l2446_244605


namespace assign_four_providers_from_twentyfive_l2446_244653

/-- The number of ways to assign different service providers to children -/
def assignProviders (totalProviders : ℕ) (children : ℕ) : ℕ :=
  (List.range children).foldl (fun acc i => acc * (totalProviders - i)) 1

/-- Theorem: Assigning 4 different service providers to 4 children from 25 providers -/
theorem assign_four_providers_from_twentyfive :
  assignProviders 25 4 = 303600 := by
  sorry

end assign_four_providers_from_twentyfive_l2446_244653


namespace rectangle_area_is_72_l2446_244665

-- Define the radius of the circles
def circle_radius : ℝ := 3

-- Define the rectangle
structure Rectangle where
  length : ℝ
  width : ℝ

-- Define the property that the circles touch each other and the rectangle sides
def circles_touch_rectangle_and_each_other (r : Rectangle) : Prop :=
  r.length = 4 * circle_radius ∧ r.width = 2 * circle_radius

-- Theorem statement
theorem rectangle_area_is_72 (r : Rectangle) 
  (h : circles_touch_rectangle_and_each_other r) : r.length * r.width = 72 := by
  sorry

end rectangle_area_is_72_l2446_244665


namespace max_rooks_300x300_l2446_244670

/-- Represents a chessboard with side length n -/
structure Chessboard (n : ℕ) where
  size : n > 0

/-- Represents a valid rook placement on a chessboard -/
structure RookPlacement (n : ℕ) where
  board : Chessboard n
  rooks : Finset (ℕ × ℕ)
  valid : ∀ (r1 r2 : ℕ × ℕ), r1 ∈ rooks → r2 ∈ rooks → r1 ≠ r2 →
    (r1.1 = r2.1 ∨ r1.2 = r2.2) → 
    (∀ r3 ∈ rooks, r3 ≠ r1 ∧ r3 ≠ r2 → r3.1 ≠ r1.1 ∧ r3.2 ≠ r1.2 ∧ r3.1 ≠ r2.1 ∧ r3.2 ≠ r2.2)

/-- The maximum number of rooks that can be placed on a 300x300 chessboard
    such that each rook attacks no more than one other rook is 400 -/
theorem max_rooks_300x300 : 
  ∀ (p : RookPlacement 300), Finset.card p.rooks ≤ 400 ∧ 
  ∃ (p' : RookPlacement 300), Finset.card p'.rooks = 400 := by
  sorry

end max_rooks_300x300_l2446_244670


namespace cube_painting_cost_l2446_244632

/-- The cost to paint a cube given paint cost, coverage, and cube dimensions -/
theorem cube_painting_cost 
  (paint_cost : ℝ) 
  (paint_coverage : ℝ) 
  (cube_side : ℝ) 
  (h1 : paint_cost = 40) 
  (h2 : paint_coverage = 20) 
  (h3 : cube_side = 10) : 
  paint_cost * (6 * cube_side^2 / paint_coverage) = 1200 := by
  sorry

#check cube_painting_cost

end cube_painting_cost_l2446_244632


namespace julia_tag_game_l2446_244668

/-- The number of kids Julia played with on Monday -/
def monday_kids : ℕ := 16

/-- The difference in the number of kids Julia played with on Monday compared to Tuesday -/
def difference : ℕ := 12

/-- The number of kids Julia played with on Tuesday -/
def tuesday_kids : ℕ := monday_kids - difference

theorem julia_tag_game :
  tuesday_kids = 4 :=
sorry

end julia_tag_game_l2446_244668


namespace min_disks_needed_l2446_244652

def total_files : ℕ := 45
def disk_capacity : ℚ := 1.44

def file_size_1 : ℚ := 0.9
def file_count_1 : ℕ := 5

def file_size_2 : ℚ := 0.6
def file_count_2 : ℕ := 15

def file_size_3 : ℚ := 0.5
def file_count_3 : ℕ := total_files - file_count_1 - file_count_2

theorem min_disks_needed : 
  ∃ (n : ℕ), n = 20 ∧ 
  (∀ m : ℕ, m < n → 
    m * disk_capacity < 
      file_count_1 * file_size_1 + 
      file_count_2 * file_size_2 + 
      file_count_3 * file_size_3) ∧
  n * disk_capacity ≥ 
    file_count_1 * file_size_1 + 
    file_count_2 * file_size_2 + 
    file_count_3 * file_size_3 :=
by sorry

end min_disks_needed_l2446_244652


namespace largest_root_is_six_l2446_244698

/-- Polynomial P(x) = x^6 - 15x^5 + 74x^4 - 130x^3 + ax^2 + bx -/
def P (a b : ℝ) (x : ℝ) : ℝ := x^6 - 15*x^5 + 74*x^4 - 130*x^3 + a*x^2 + b*x

/-- Line L(x) = cx - 24 -/
def L (c : ℝ) (x : ℝ) : ℝ := c*x - 24

/-- The difference between P(x) and L(x) -/
def D (a b c : ℝ) (x : ℝ) : ℝ := P a b x - L c x

theorem largest_root_is_six (a b c : ℝ) : 
  (∃ p q : ℝ, (∀ x : ℝ, D a b c x = (x - p)^3 * (x - q)^2)) →
  (∀ x : ℝ, D a b c x ≥ 0) →
  (∃ x₁ x₂ x₃ : ℝ, D a b c x₁ = 0 ∧ D a b c x₂ = 0 ∧ D a b c x₃ = 0 ∧ x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃) →
  (∃ x : ℝ, D a b c x = 0 ∧ x = 6 ∧ ∀ y : ℝ, D a b c y = 0 → y ≤ x) :=
sorry

end largest_root_is_six_l2446_244698


namespace mayoral_election_votes_l2446_244601

theorem mayoral_election_votes (z : ℕ) (hz : z = 25000) : ∃ x y : ℕ,
  y = z - (2 * z / 5) ∧
  x = y + (y / 2) ∧
  x = 22500 :=
by
  sorry

end mayoral_election_votes_l2446_244601


namespace torus_division_theorem_l2446_244606

/-- Represents a torus surface -/
structure TorusSurface where
  -- Add necessary fields here

/-- Represents a path on the torus surface -/
structure PathOnTorus where
  -- Add necessary fields here

/-- Represents the outer equator of the torus -/
def outerEquator : PathOnTorus :=
  sorry

/-- Represents a helical line on the torus -/
def helicalLine : PathOnTorus :=
  sorry

/-- Counts the number of regions a torus surface is divided into when cut along given paths -/
def countRegions (surface : TorusSurface) (path1 path2 : PathOnTorus) : ℕ :=
  sorry

/-- Theorem stating that cutting a torus along its outer equator and a helical line divides it into 3 parts -/
theorem torus_division_theorem (surface : TorusSurface) :
  countRegions surface outerEquator helicalLine = 3 :=
sorry

end torus_division_theorem_l2446_244606


namespace marbles_left_l2446_244645

def marbles_in_box : ℕ := 50
def white_marbles : ℕ := 20

def red_blue_marbles : ℕ := marbles_in_box - white_marbles
def blue_marbles : ℕ := red_blue_marbles / 2
def red_marbles : ℕ := red_blue_marbles / 2

def marbles_removed : ℕ := 2 * (white_marbles - blue_marbles)

theorem marbles_left : marbles_in_box - marbles_removed = 40 := by
  sorry

end marbles_left_l2446_244645


namespace intersection_digit_l2446_244615

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

def third_digit (n : ℕ) : ℕ := (n / 100) % 10

def four_digit_power_of_2 (m : ℕ) : Prop :=
  ∃ k, is_four_digit (2^k) ∧ m = third_digit (2^k)

def four_digit_power_of_5 (n : ℕ) : Prop :=
  ∃ k, is_four_digit (5^k) ∧ n = third_digit (5^k)

theorem intersection_digit :
  ∃! d, four_digit_power_of_2 d ∧ four_digit_power_of_5 d :=
sorry

end intersection_digit_l2446_244615


namespace four_fish_guarantee_l2446_244685

/-- A coloring of vertices of a regular polygon -/
def Coloring (n : ℕ) := Fin n → Bool

/-- The number of perpendicular segments with same-colored endpoints for a given diagonal -/
def sameColorSegments (n : ℕ) (c : Coloring n) (d : Fin n) : ℕ :=
  sorry

theorem four_fish_guarantee (c : Coloring 20) :
  ∃ d : Fin 20, sameColorSegments 20 c d ≥ 4 := by
  sorry

end four_fish_guarantee_l2446_244685


namespace arithmetic_sequence_common_difference_l2446_244608

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℚ)
  (h_arith : ArithmeticSequence a)
  (h_cond1 : a 7 - 2 * a 4 = -1)
  (h_cond2 : a 3 = 0) :
  ∃ d : ℚ, (∀ n : ℕ, a (n + 1) = a n + d) ∧ d = -1/2 :=
sorry

end arithmetic_sequence_common_difference_l2446_244608


namespace harry_beach_collection_l2446_244655

/-- The number of items Harry has left after his walk on the beach -/
def items_left (sea_stars seashells snails lost : ℕ) : ℕ :=
  sea_stars + seashells + snails - lost

/-- Theorem stating that Harry has 59 items left after his walk -/
theorem harry_beach_collection : items_left 34 21 29 25 = 59 := by
  sorry

end harry_beach_collection_l2446_244655


namespace bumper_car_line_theorem_l2446_244660

/-- The number of people initially in line for bumper cars -/
def initial_people : ℕ := 9

/-- The number of people who left the line -/
def people_left : ℕ := 6

/-- The number of people who joined the line -/
def people_joined : ℕ := 3

/-- The final number of people in line -/
def final_people : ℕ := 6

/-- Theorem stating that the initial number of people satisfies the given conditions -/
theorem bumper_car_line_theorem :
  initial_people - people_left + people_joined = final_people :=
by sorry

end bumper_car_line_theorem_l2446_244660


namespace right_triangle_sin_q_l2446_244629

/-- In a right triangle PQR with angle R = 90° and 3sin Q = 4cos Q, sin Q = 4/5 -/
theorem right_triangle_sin_q (Q : Real) (h1 : 3 * Real.sin Q = 4 * Real.cos Q) : Real.sin Q = 4/5 := by
  sorry

end right_triangle_sin_q_l2446_244629


namespace circle_radius_from_area_l2446_244671

theorem circle_radius_from_area (A : ℝ) (r : ℝ) (h : A = 121 * Real.pi) :
  A = Real.pi * r^2 → r = 11 := by
  sorry

end circle_radius_from_area_l2446_244671


namespace equation_solution_l2446_244641

theorem equation_solution (x : ℝ) (hx : x ≠ 0) :
  (9 / x^2) - (6 / x) + 1 = 0 → 2 / x = 2 / 3 := by
  sorry

end equation_solution_l2446_244641


namespace sum_exterior_angles_dodecagon_l2446_244626

/-- The number of sides in a dodecagon -/
def dodecagon_sides : ℕ := 12

/-- The sum of exterior angles of any polygon in degrees -/
def sum_exterior_angles : ℝ := 360

/-- Theorem: The sum of the exterior angles of a regular dodecagon is 360° -/
theorem sum_exterior_angles_dodecagon :
  sum_exterior_angles = 360 := by sorry

end sum_exterior_angles_dodecagon_l2446_244626


namespace least_number_to_add_for_divisibility_l2446_244609

theorem least_number_to_add_for_divisibility (n m : ℕ) (h : n = 1076 ∧ m = 23) :
  ∃ x : ℕ, (n + x) % m = 0 ∧ ∀ y : ℕ, y < x → (n + y) % m ≠ 0 ∧ x = 5 := by
  sorry

end least_number_to_add_for_divisibility_l2446_244609


namespace angle_expression_proof_l2446_244614

theorem angle_expression_proof (α : Real) (h : Real.tan α = 2) :
  (Real.cos (α - π) - 2 * Real.cos (π / 2 + α)) / (Real.sin (α - 3 * π / 2) - Real.sin α) = -3 :=
by sorry

end angle_expression_proof_l2446_244614


namespace inverse_of_inverse_f_l2446_244638

-- Define the original function f
def f (x : ℝ) : ℝ := 2 * x + 3

-- Define the inverse of f^(-1)(x+1)
def g (x : ℝ) : ℝ := 2 * x + 2

-- Theorem statement
theorem inverse_of_inverse_f (x : ℝ) : 
  g (f⁻¹ (x + 1)) = x ∧ f⁻¹ (g x + 1) = x := by
  sorry


end inverse_of_inverse_f_l2446_244638


namespace equal_area_rectangles_l2446_244696

/-- Given two rectangles with equal area, where one rectangle measures 5 inches by 24 inches
    and the other is 8 inches long, prove that the width of the second rectangle is 15 inches. -/
theorem equal_area_rectangles (carol_length carol_width jordan_length jordan_width : ℝ) :
  carol_length = 5 →
  carol_width = 24 →
  jordan_length = 8 →
  carol_length * carol_width = jordan_length * jordan_width →
  jordan_width = 15 := by
  sorry

end equal_area_rectangles_l2446_244696


namespace quadratic_roots_product_l2446_244621

theorem quadratic_roots_product (n : ℝ) (c d : ℝ) :
  c^2 - n*c + 4 = 0 →
  d^2 - n*d + 4 = 0 →
  ∃ (s : ℝ), (c + 1/d)^2 - s*(c + 1/d) + 25/4 = 0 ∧
             (d + 1/c)^2 - s*(d + 1/c) + 25/4 = 0 :=
by sorry

end quadratic_roots_product_l2446_244621


namespace marbles_distribution_l2446_244604

theorem marbles_distribution (total_marbles : ℕ) (kept_marbles : ℕ) (best_friends : ℕ) (marbles_per_best_friend : ℕ) (neighborhood_friends : ℕ) :
  total_marbles = 1125 →
  kept_marbles = 100 →
  best_friends = 2 →
  marbles_per_best_friend = 50 →
  neighborhood_friends = 7 →
  (total_marbles - kept_marbles - best_friends * marbles_per_best_friend) / neighborhood_friends = 132 :=
by sorry

end marbles_distribution_l2446_244604


namespace inequality_proof_l2446_244633

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : x + y + z ≥ 1/x + 1/y + 1/z) :
  x/y + y/z + z/x ≥ 1/(x*y) + 1/(y*z) + 1/(z*x) := by
  sorry

end inequality_proof_l2446_244633


namespace last_three_digits_of_7_to_120_l2446_244687

theorem last_three_digits_of_7_to_120 : 7^120 % 1000 = 681 := by
  sorry

end last_three_digits_of_7_to_120_l2446_244687


namespace two_cos_thirty_degrees_equals_sqrt_three_l2446_244661

theorem two_cos_thirty_degrees_equals_sqrt_three :
  2 * Real.cos (30 * π / 180) = Real.sqrt 3 := by
  sorry

end two_cos_thirty_degrees_equals_sqrt_three_l2446_244661

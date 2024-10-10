import Mathlib

namespace percentage_increase_l786_78632

theorem percentage_increase (initial : ℝ) (final : ℝ) : 
  initial = 100 → final = 110 → (final - initial) / initial * 100 = 10 := by
sorry

end percentage_increase_l786_78632


namespace taxi_charge_per_segment_l786_78637

/-- Calculates the additional charge per 2/5 of a mile for a taxi service -/
theorem taxi_charge_per_segment (initial_fee : ℚ) (total_distance : ℚ) (total_charge : ℚ) :
  initial_fee = 2.25 →
  total_distance = 3.6 →
  total_charge = 3.60 →
  (total_charge - initial_fee) / (total_distance / (2/5)) = 0.15 := by
  sorry

end taxi_charge_per_segment_l786_78637


namespace min_socks_for_pairs_l786_78660

/-- Represents the number of colors of socks in the drawer -/
def num_colors : ℕ := 4

/-- Represents the number of pairs we want to guarantee -/
def required_pairs : ℕ := 10

/-- Theorem: The minimum number of socks to guarantee the required pairs -/
theorem min_socks_for_pairs :
  ∀ (sock_counts : Fin num_colors → ℕ),
  (∀ i, sock_counts i > 0) →
  ∃ (n : ℕ),
    n = num_colors + 2 * required_pairs ∧
    ∀ (m : ℕ), m ≥ n →
      ∀ (selection : Fin m → Fin num_colors),
      ∃ (pairs : Fin required_pairs → Fin m × Fin m),
        ∀ i, 
          (pairs i).1 < (pairs i).2 ∧
          selection (pairs i).1 = selection (pairs i).2 ∧
          ∀ j, i ≠ j → 
            ({(pairs i).1, (pairs i).2} : Set (Fin m)) ∩ {(pairs j).1, (pairs j).2} = ∅ :=
by
  sorry

end min_socks_for_pairs_l786_78660


namespace linear_function_difference_l786_78666

-- Define the properties of the linear function g
def g_properties (g : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, ∃ a b : ℝ, g x = a * x + b) ∧ 
  (∀ d : ℝ, g (d + 2) - g d = 4)

-- State the theorem
theorem linear_function_difference 
  (g : ℝ → ℝ) 
  (h : g_properties g) : 
  g 4 - g 8 = -8 := by
  sorry

end linear_function_difference_l786_78666


namespace order_of_logarithms_and_root_l786_78612

theorem order_of_logarithms_and_root (a b c : ℝ) : 
  a = 2 * Real.log 0.99 →
  b = Real.log 0.98 →
  c = Real.sqrt 0.96 - 1 →
  a > b ∧ b > c := by
  sorry

end order_of_logarithms_and_root_l786_78612


namespace arithmetic_sequence_count_l786_78672

theorem arithmetic_sequence_count : 
  ∀ (a d last : ℕ) (n : ℕ),
    a = 2 →
    d = 4 →
    last = 2018 →
    last = a + (n - 1) * d →
    n = 505 := by
  sorry

end arithmetic_sequence_count_l786_78672


namespace geometric_sequence_sum_l786_78670

theorem geometric_sequence_sum (n : ℕ) (a r : ℚ) (h1 : a = 1/3) (h2 : r = 1/3) :
  (a * (1 - r^n) / (1 - r) = 80/243) → n = 3 := by
  sorry

end geometric_sequence_sum_l786_78670


namespace company_sugar_usage_l786_78626

/-- The amount of sugar (in grams) used by a chocolate company in two minutes -/
def sugar_used_in_two_minutes (sugar_per_bar : ℝ) (bars_per_minute : ℝ) : ℝ :=
  2 * (sugar_per_bar * bars_per_minute)

/-- Theorem stating that the company uses 108 grams of sugar in two minutes -/
theorem company_sugar_usage :
  sugar_used_in_two_minutes 1.5 36 = 108 := by
  sorry

end company_sugar_usage_l786_78626


namespace cosine_sine_identity_l786_78611

theorem cosine_sine_identity : 
  Real.cos (35 * π / 180) * Real.cos (25 * π / 180) - 
  Real.sin (145 * π / 180) * Real.cos (65 * π / 180) = 1/2 := by
  sorry

end cosine_sine_identity_l786_78611


namespace even_painted_faces_count_l786_78696

/-- Represents a rectangular block with given dimensions -/
structure Block where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Counts the number of cubes with an even number of painted faces in a block -/
def countEvenPaintedFaces (b : Block) : ℕ :=
  sorry

/-- The main theorem stating that a 6x3x2 block has 16 cubes with even number of painted faces -/
theorem even_painted_faces_count : 
  let b : Block := { length := 6, width := 3, height := 2 }
  countEvenPaintedFaces b = 16 := by
  sorry

end even_painted_faces_count_l786_78696


namespace jar_balls_count_l786_78645

theorem jar_balls_count (initial_blue : ℕ) (removed : ℕ) (prob : ℚ) :
  initial_blue = 6 →
  removed = 3 →
  prob = 1/5 →
  (initial_blue - removed : ℚ) / ((initial_blue - removed : ℚ) + (18 - initial_blue : ℚ)) = prob →
  18 = initial_blue + (18 - initial_blue) :=
by sorry

end jar_balls_count_l786_78645


namespace ten_customers_miss_sunday_paper_l786_78616

/-- Represents Kyle's newspaper delivery route -/
structure NewspaperRoute where
  totalHouses : ℕ
  dailyDeliveries : ℕ
  sundayOnlyDeliveries : ℕ
  weeklyTotalDeliveries : ℕ

/-- Calculates the number of customers who do not get the Sunday paper -/
def customersMissingSundayPaper (route : NewspaperRoute) : ℕ :=
  route.totalHouses - (route.totalHouses - (route.weeklyTotalDeliveries - 6 * route.totalHouses - route.sundayOnlyDeliveries))

/-- Theorem stating that 10 customers do not get the Sunday paper -/
theorem ten_customers_miss_sunday_paper (route : NewspaperRoute) 
  (h1 : route.totalHouses = 100)
  (h2 : route.dailyDeliveries = 100)
  (h3 : route.sundayOnlyDeliveries = 30)
  (h4 : route.weeklyTotalDeliveries = 720) :
  customersMissingSundayPaper route = 10 := by
  sorry

#eval customersMissingSundayPaper { totalHouses := 100, dailyDeliveries := 100, sundayOnlyDeliveries := 30, weeklyTotalDeliveries := 720 }

end ten_customers_miss_sunday_paper_l786_78616


namespace area_YPW_is_8_l786_78676

/-- Represents a rectangle XYZW with given length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Represents a point P that divides the diagonal XW of a rectangle -/
structure DiagonalPoint where
  ratio_XP : ℝ
  ratio_PW : ℝ

/-- Calculates the area of triangle YPW in the given rectangle with the given diagonal point -/
def area_YPW (rect : Rectangle) (p : DiagonalPoint) : ℝ :=
  sorry

/-- Theorem stating that for a rectangle with length 8 and width 6, 
    if P divides XW in ratio 2:1, then area of YPW is 8 -/
theorem area_YPW_is_8 (rect : Rectangle) (p : DiagonalPoint) :
  rect.length = 8 →
  rect.width = 6 →
  p.ratio_XP = 2 →
  p.ratio_PW = 1 →
  area_YPW rect p = 8 := by
  sorry

end area_YPW_is_8_l786_78676


namespace reappearance_is_lcm_reappearance_is_twenty_l786_78669

/-- The cycle length of the letter sequence -/
def letter_cycle_length : ℕ := 5

/-- The cycle length of the digit sequence -/
def digit_cycle_length : ℕ := 4

/-- The line number where the original sequences reappear together -/
def reappearance_line : ℕ := 20

/-- Theorem stating that the reappearance line is the LCM of the cycle lengths -/
theorem reappearance_is_lcm :
  reappearance_line = Nat.lcm letter_cycle_length digit_cycle_length := by
  sorry

/-- Theorem stating that the reappearance line is 20 -/
theorem reappearance_is_twenty : reappearance_line = 20 := by
  sorry

end reappearance_is_lcm_reappearance_is_twenty_l786_78669


namespace stratified_sampling_male_count_l786_78682

theorem stratified_sampling_male_count 
  (total_employees : ℕ) 
  (female_employees : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_employees = 120) 
  (h2 : female_employees = 72) 
  (h3 : sample_size = 15) :
  (total_employees - female_employees) * sample_size / total_employees = 6 := by
sorry

end stratified_sampling_male_count_l786_78682


namespace angle_sum_in_triangle_l786_78635

theorem angle_sum_in_triangle (A B C : ℝ) : 
  A + B + C = 180 →
  A + B = 150 →
  C = 30 := by
  sorry

end angle_sum_in_triangle_l786_78635


namespace combined_mean_of_two_sets_mean_of_fifteen_numbers_l786_78687

theorem combined_mean_of_two_sets (set1_count : ℕ) (set1_mean : ℚ) 
                                  (set2_count : ℕ) (set2_mean : ℚ) : ℚ :=
  let total_count := set1_count + set2_count
  let combined_sum := set1_count * set1_mean + set2_count * set2_mean
  combined_sum / total_count

theorem mean_of_fifteen_numbers : 
  combined_mean_of_two_sets 7 15 8 22 = 281 / 15 := by
  sorry

end combined_mean_of_two_sets_mean_of_fifteen_numbers_l786_78687


namespace simplify_expressions_l786_78610

theorem simplify_expressions :
  (1 + (-0.5) = 0.5) ∧
  (2 - 10.1 = -10.1) ∧
  (3 + 7 = 10) ∧
  (4 - (-20) = 24) ∧
  (5 + |-(2/3)| = 17/3) ∧
  (6 - |-(4/5)| = 26/5) ∧
  (7 + (-(-10)) = 17) ∧
  (8 - (-(-20/7)) = -12/7) := by
  sorry

end simplify_expressions_l786_78610


namespace fraction_product_simplification_l786_78642

/-- The product of fractions from 10/5 to 2520/2515 -/
def fraction_product : ℕ → ℚ
  | 0 => 2 -- 10/5
  | n + 1 => fraction_product n * ((5 * (n + 2)) / (5 * (n + 1)))

theorem fraction_product_simplification :
  fraction_product 502 = 504 := by
  sorry

end fraction_product_simplification_l786_78642


namespace coordinates_of_B_l786_78680

/-- Given a line segment AB parallel to the y-axis, with A(1, -2) and AB = 8,
    the coordinates of B are either (1, -10) or (1, 6). -/
theorem coordinates_of_B (A B : ℝ × ℝ) : 
  A = (1, -2) →
  (B.1 = A.1) →
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 8 →
  (B = (1, -10) ∨ B = (1, 6)) :=
by sorry

end coordinates_of_B_l786_78680


namespace geometric_series_sum_l786_78688

theorem geometric_series_sum : 
  let a : ℚ := 1/3
  let r : ℚ := -1/4
  let n : ℕ := 6
  let series_sum : ℚ := a * (1 - r^n) / (1 - r)
  series_sum = 4095/30720 := by
  sorry

end geometric_series_sum_l786_78688


namespace max_product_decomposition_l786_78608

theorem max_product_decomposition (a : ℝ) (ha : a > 0) :
  ∀ x y : ℝ, x ≥ 0 → y ≥ 0 → x + y = a →
  x * y ≤ (a / 2) * (a / 2) ∧
  (x * y = (a / 2) * (a / 2) ↔ x = a / 2 ∧ y = a / 2) :=
by sorry

end max_product_decomposition_l786_78608


namespace fifteenth_even_multiple_of_3_l786_78602

/-- The nth positive even integer that is a multiple of 3 -/
def evenMultipleOf3 (n : ℕ) : ℕ := 6 * n

/-- The 15th positive even integer that is a multiple of 3 is 90 -/
theorem fifteenth_even_multiple_of_3 : evenMultipleOf3 15 = 90 := by
  sorry

end fifteenth_even_multiple_of_3_l786_78602


namespace min_distance_complex_l786_78613

theorem min_distance_complex (z : ℂ) (h : Complex.abs (z + 2 - 2*I) = 1) :
  ∃ (min_val : ℝ), min_val = 2 ∧ ∀ w : ℂ, Complex.abs (w + 2 - 2*I) = 1 → Complex.abs (w - 1 - 2*I) ≥ min_val :=
by sorry

end min_distance_complex_l786_78613


namespace average_age_combined_l786_78699

theorem average_age_combined (num_students : ℕ) (num_parents : ℕ) 
  (avg_age_students : ℚ) (avg_age_parents : ℚ) :
  num_students = 40 →
  num_parents = 60 →
  avg_age_students = 12 →
  avg_age_parents = 40 →
  ((num_students : ℚ) * avg_age_students + (num_parents : ℚ) * avg_age_parents) / 
    ((num_students : ℚ) + (num_parents : ℚ)) = 28.8 := by
  sorry

end average_age_combined_l786_78699


namespace x_is_25_percent_greater_than_88_l786_78694

theorem x_is_25_percent_greater_than_88 (x : ℝ) : 
  x = 88 * (1 + 0.25) → x = 110 := by
  sorry

end x_is_25_percent_greater_than_88_l786_78694


namespace seeds_sowed_l786_78625

/-- Proves that the number of buckets of seeds sowed is 2.75 -/
theorem seeds_sowed (initial : ℝ) (final : ℝ) (h1 : initial = 8.75) (h2 : final = 6) :
  initial - final = 2.75 := by
  sorry

end seeds_sowed_l786_78625


namespace armband_break_even_l786_78650

/-- The cost of an individual ticket in dollars -/
def individual_ticket_cost : ℚ := 3/4

/-- The cost of an armband in dollars -/
def armband_cost : ℚ := 15

/-- The number of rides at which the armband cost equals the individual ticket cost -/
def break_even_rides : ℕ := 20

theorem armband_break_even :
  (individual_ticket_cost * break_even_rides : ℚ) = armband_cost :=
sorry

end armband_break_even_l786_78650


namespace star_calculation_l786_78678

def star (x y : ℝ) : ℝ := (x + y) * (x - y)

theorem star_calculation : star 2 (star 3 (star 1 2)) = 4 := by
  sorry

end star_calculation_l786_78678


namespace print_shop_cost_difference_l786_78606

/-- The cost difference between two print shops for a given number of copies -/
def cost_difference (price_x price_y : ℚ) (num_copies : ℕ) : ℚ :=
  (price_y - price_x) * num_copies

/-- Theorem stating the cost difference between print shops Y and X for 40 copies -/
theorem print_shop_cost_difference :
  cost_difference (120/100) (170/100) 40 = 20 := by
  sorry

end print_shop_cost_difference_l786_78606


namespace circle_tangent_ratio_l786_78673

-- Define the types for points and circles
variable (Point Circle : Type)

-- Define the basic geometric relations
variable (on_circle : Point → Circle → Prop)
variable (inside_circle : Circle → Circle → Prop)
variable (concentric : Circle → Circle → Prop)
variable (tangent_to : Point → Point → Circle → Prop)
variable (intersects : Point → Point → Circle → Point → Prop)
variable (midpoint : Point → Point → Point → Prop)
variable (line_through : Point → Point → Point → Prop)
variable (perp_bisector : Point → Point → Point → Point → Prop)
variable (ratio : Point → Point → Point → ℚ → Prop)

-- State the theorem
theorem circle_tangent_ratio 
  (Γ₁ Γ₂ : Circle) 
  (A B C D E F M : Point) :
  concentric Γ₁ Γ₂ →
  inside_circle Γ₂ Γ₁ →
  on_circle A Γ₁ →
  on_circle B Γ₂ →
  tangent_to A B Γ₂ →
  intersects A B Γ₁ C →
  midpoint D A B →
  line_through A E F →
  on_circle E Γ₂ →
  on_circle F Γ₂ →
  perp_bisector D E M B →
  perp_bisector C F M B →
  ratio A M C (3/2) :=
by sorry

end circle_tangent_ratio_l786_78673


namespace line_segment_endpoint_l786_78661

/-- Given a line segment with midpoint (3, -3) and one endpoint (7, 4),
    prove that the other endpoint is (-1, -10). -/
theorem line_segment_endpoint
  (midpoint : ℝ × ℝ)
  (endpoint1 : ℝ × ℝ)
  (h_midpoint : midpoint = (3, -3))
  (h_endpoint1 : endpoint1 = (7, 4)) :
  ∃ (endpoint2 : ℝ × ℝ),
    endpoint2 = (-1, -10) ∧
    midpoint = (
      (endpoint1.1 + endpoint2.1) / 2,
      (endpoint1.2 + endpoint2.2) / 2
    ) :=
by sorry

end line_segment_endpoint_l786_78661


namespace student_weights_l786_78624

/-- Theorem: Total and average weight of students
Given 10 students with a base weight and weight deviations, 
prove the total weight and average weight. -/
theorem student_weights (base_weight : ℝ) (weight_deviations : List ℝ) : 
  base_weight = 50 ∧ 
  weight_deviations = [2, 3, -7.5, -3, 5, -8, 3.5, 4.5, 8, -1.5] →
  (List.sum weight_deviations + 10 * base_weight = 509) ∧
  ((List.sum weight_deviations + 10 * base_weight) / 10 = 50.9) := by
  sorry

#check student_weights

end student_weights_l786_78624


namespace cubic_roots_sum_cubes_l786_78674

theorem cubic_roots_sum_cubes (a b c : ℝ) : 
  (2 * a^3 - 3 * a^2 + 165 * a - 4 = 0) →
  (2 * b^3 - 3 * b^2 + 165 * b - 4 = 0) →
  (2 * c^3 - 3 * c^2 + 165 * c - 4 = 0) →
  (a + b - 1)^3 + (b + c - 1)^3 + (c + a - 1)^3 = 117 := by
sorry

end cubic_roots_sum_cubes_l786_78674


namespace min_value_2a_plus_b_l786_78605

theorem min_value_2a_plus_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (heq : 3 * a + b = a^2 + a * b) :
  ∀ x y : ℝ, x > 0 → y > 0 → 3 * x + y = x^2 + x * y → 2 * x + y ≥ 3 + 2 * Real.sqrt 2 :=
sorry

end min_value_2a_plus_b_l786_78605


namespace sandwich_cost_l786_78655

theorem sandwich_cost (num_sandwiches num_drinks drink_cost total_cost : ℕ) 
  (h1 : num_sandwiches = 3)
  (h2 : num_drinks = 2)
  (h3 : drink_cost = 4)
  (h4 : total_cost = 26) :
  ∃ (sandwich_cost : ℕ), 
    num_sandwiches * sandwich_cost + num_drinks * drink_cost = total_cost ∧ 
    sandwich_cost = 6 := by
  sorry

end sandwich_cost_l786_78655


namespace quadratic_root_proof_l786_78658

theorem quadratic_root_proof (v : ℝ) : 
  v = 7 → (5 * (((-21 - Real.sqrt 301) / 10) ^ 2) + 21 * ((-21 - Real.sqrt 301) / 10) + v = 0) :=
by sorry

end quadratic_root_proof_l786_78658


namespace arithmetic_sequence_sum_l786_78681

/-- Given two arithmetic sequences {an} and {bn} with the specified conditions, 
    prove that a5 + b5 = 35 -/
theorem arithmetic_sequence_sum (a b : ℕ → ℕ) 
  (h1 : ∀ n, a (n + 1) - a n = a 2 - a 1)  -- a is an arithmetic sequence
  (h2 : ∀ n, b (n + 1) - b n = b 2 - b 1)  -- b is an arithmetic sequence
  (h3 : a 1 + b 1 = 7)                     -- first condition
  (h4 : a 3 + b 3 = 21)                    -- second condition
  : a 5 + b 5 = 35 := by
  sorry

end arithmetic_sequence_sum_l786_78681


namespace arithmetic_sequence_sum_l786_78654

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem statement -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 1 + a 3 = 2) →
  (a 3 + a 5 = 4) →
  (a 5 + a 7 = 6) :=
by
  sorry

end arithmetic_sequence_sum_l786_78654


namespace ferry_distance_ratio_l786_78649

/-- The ratio of the distance covered by ferry Q to the distance covered by ferry P -/
theorem ferry_distance_ratio :
  let speed_p : ℝ := 8
  let time_p : ℝ := 3
  let speed_q : ℝ := speed_p + 1
  let time_q : ℝ := time_p + 5
  let distance_p : ℝ := speed_p * time_p
  let distance_q : ℝ := speed_q * time_q
  (distance_q / distance_p : ℝ) = 3 := by
  sorry

end ferry_distance_ratio_l786_78649


namespace square_plus_reciprocal_squared_l786_78690

theorem square_plus_reciprocal_squared (x : ℝ) (h : x ≠ 0) :
  x^2 + 1/x^2 = 2 → x^4 + 1/x^4 = 2 :=
by
  sorry

end square_plus_reciprocal_squared_l786_78690


namespace permutations_of_six_objects_l786_78659

theorem permutations_of_six_objects : Nat.factorial 6 = 720 := by
  sorry

end permutations_of_six_objects_l786_78659


namespace oak_trees_after_planting_l786_78634

/-- The number of oak trees in the park after planting -/
def total_oak_trees (initial : ℕ) (new : ℕ) : ℕ :=
  initial + new

/-- Theorem: The total number of oak trees after planting is 11 -/
theorem oak_trees_after_planting :
  total_oak_trees 9 2 = 11 := by
  sorry

end oak_trees_after_planting_l786_78634


namespace system_solution_l786_78636

theorem system_solution (x y : ℝ) (m n : ℤ) : 
  (4 * (Real.cos x)^2 * (Real.sin (x/6))^2 + 4 * Real.sin (x/6) - 4 * (Real.sin x)^2 * Real.sin (x/6) + 1 = 0 ∧
   Real.sin (x/4) = Real.sqrt (Real.cos y)) ↔ 
  ((x = 11 * Real.pi + 24 * Real.pi * ↑m ∧ (y = Real.pi/3 + 2 * Real.pi * ↑n ∨ y = -Real.pi/3 + 2 * Real.pi * ↑n)) ∨
   (x = -5 * Real.pi + 24 * Real.pi * ↑m ∧ (y = Real.pi/3 + 2 * Real.pi * ↑n ∨ y = -Real.pi/3 + 2 * Real.pi * ↑n))) :=
by sorry

end system_solution_l786_78636


namespace cost_difference_l786_78665

-- Define the monthly costs
def rental_cost : ℕ := 20
def new_car_cost : ℕ := 30

-- Define the number of months in a year
def months_in_year : ℕ := 12

-- Define the total costs for a year
def total_rental_cost : ℕ := rental_cost * months_in_year
def total_new_car_cost : ℕ := new_car_cost * months_in_year

-- Theorem statement
theorem cost_difference :
  total_new_car_cost - total_rental_cost = 120 :=
by sorry

end cost_difference_l786_78665


namespace at_least_one_geq_two_l786_78677

theorem at_least_one_geq_two (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + 1/b ≥ 2) ∨ (b + 1/c ≥ 2) ∨ (c + 1/a ≥ 2) := by
  sorry

end at_least_one_geq_two_l786_78677


namespace catherine_stationery_l786_78630

theorem catherine_stationery (initial_pens initial_pencils pens_given pencils_given remaining_pens remaining_pencils : ℕ) :
  initial_pens = initial_pencils →
  pens_given = 36 →
  pencils_given = 16 →
  remaining_pens = 36 →
  remaining_pencils = 28 →
  initial_pens - pens_given = remaining_pens →
  initial_pencils - pencils_given = remaining_pencils →
  initial_pens = 72 ∧ initial_pencils = 72 := by
sorry

end catherine_stationery_l786_78630


namespace janes_apple_baskets_l786_78691

theorem janes_apple_baskets :
  ∀ (total_apples : ℕ) (apples_taken : ℕ) (apples_left : ℕ),
    total_apples = 64 →
    apples_taken = 3 →
    apples_left = 13 →
    ∃ (num_baskets : ℕ),
      num_baskets * (apples_left + apples_taken) = total_apples ∧
      num_baskets = 4 :=
by
  sorry

end janes_apple_baskets_l786_78691


namespace fort_men_count_l786_78603

/-- Represents the initial number of men in the fort -/
def initial_men : ℕ := 150

/-- Represents the number of days the initial provision would last -/
def initial_days : ℕ := 45

/-- Represents the number of days after which some men left -/
def days_before_leaving : ℕ := 10

/-- Represents the number of men who left the fort -/
def men_who_left : ℕ := 25

/-- Represents the number of days the remaining food lasted -/
def remaining_days : ℕ := 42

/-- Theorem stating that given the conditions, the initial number of men in the fort was 150 -/
theorem fort_men_count :
  initial_men * (initial_days - days_before_leaving) = 
  (initial_men - men_who_left) * remaining_days :=
by sorry

end fort_men_count_l786_78603


namespace circle_y_axis_intersection_length_l786_78693

/-- A circle passes through points A(1, 3), B(4, 2), and C(1, -7). 
    The segment MN is formed by the intersection of this circle with the y-axis. -/
theorem circle_y_axis_intersection_length :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    let circle := {(x, y) : ℝ × ℝ | (x - center.1)^2 + (y - center.2)^2 = radius^2}
    (1, 3) ∈ circle ∧ (4, 2) ∈ circle ∧ (1, -7) ∈ circle →
    let y_intersections := {y : ℝ | (0, y) ∈ circle}
    ∃ (m n : ℝ), m ∈ y_intersections ∧ n ∈ y_intersections ∧ m ≠ n ∧ 
    |m - n| = 4 * Real.sqrt 6 :=
by sorry

end circle_y_axis_intersection_length_l786_78693


namespace inequality_solution_l786_78639

theorem inequality_solution (x : ℝ) : 
  (x * (x - 1)) / ((x - 5)^2) ≥ 15 ↔ 
  (x ≤ 4.09 ∨ x ≥ 6.56) ∧ x ≠ 5 :=
by sorry

end inequality_solution_l786_78639


namespace parallelogram_height_l786_78651

theorem parallelogram_height (area base height : ℝ) : 
  area = 120 ∧ base = 12 ∧ area = base * height → height = 10 := by
  sorry

end parallelogram_height_l786_78651


namespace no_solution_fractional_equation_l786_78683

theorem no_solution_fractional_equation :
  ¬∃ (x : ℝ), (x - 2) / (2 * x - 1) + 1 = 3 / (2 - 4 * x) :=
by sorry

end no_solution_fractional_equation_l786_78683


namespace base_6_addition_l786_78657

/-- Converts a base-6 number to base-10 --/
def to_base_10 (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (6 ^ i)) 0

/-- Converts a base-10 number to base-6 --/
def to_base_6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 6) ((m % 6) :: acc)
  aux n []

theorem base_6_addition :
  to_base_6 (to_base_10 [4, 2, 5, 3] + to_base_10 [2, 4, 4, 2]) = [0, 1, 4, 0, 1] := by
  sorry

end base_6_addition_l786_78657


namespace factorization_of_quadratic_l786_78675

theorem factorization_of_quadratic (a : ℝ) : a^2 + 2*a = a*(a + 2) := by
  sorry

end factorization_of_quadratic_l786_78675


namespace proportional_function_expression_l786_78643

/-- Given a proportional function y = kx (k ≠ 0), if y = 6 when x = 4, 
    then the function can be expressed as y = (3/2)x -/
theorem proportional_function_expression (k : ℝ) (h1 : k ≠ 0) : 
  (∀ x y, y = k * x) → (6 : ℝ) = k * 4 → 
  ∀ x y, y = k * x ↔ y = (3/2) * x := by
  sorry

end proportional_function_expression_l786_78643


namespace dinner_arrangements_l786_78685

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- There are 5 people in the group -/
def total_people : ℕ := 5

/-- The number of people who cook -/
def cooks : ℕ := 2

theorem dinner_arrangements :
  choose total_people cooks = 10 := by
  sorry

end dinner_arrangements_l786_78685


namespace largest_prime_divisor_l786_78646

-- Define the number in base 5
def base_5_number : Nat := 200220220

-- Define the function to convert from base 5 to base 10
def base_5_to_10 (n : Nat) : Nat :=
  let digits := n.digits 5
  (List.range digits.length).foldl (fun acc i => acc + digits[i]! * (5^i)) 0

-- Define the number in base 10
def number : Nat := base_5_to_10 base_5_number

-- Statement to prove
theorem largest_prime_divisor :
  ∃ (p : Nat), Nat.Prime p ∧ p ∣ number ∧ ∀ (q : Nat), Nat.Prime q → q ∣ number → q ≤ p :=
by sorry

end largest_prime_divisor_l786_78646


namespace positive_root_range_l786_78692

theorem positive_root_range : ∃ x : ℝ, x^2 - 2*x - 1 = 0 ∧ x > 0 ∧ 2 < x ∧ x < 3 := by
  sorry

end positive_root_range_l786_78692


namespace rosie_pies_l786_78698

/-- The number of apples required to make one pie -/
def apples_per_pie : ℕ := 5

/-- The total number of apples Rosie has -/
def total_apples : ℕ := 32

/-- The maximum number of whole pies that can be made -/
def max_pies : ℕ := total_apples / apples_per_pie

theorem rosie_pies :
  max_pies = 6 :=
sorry

end rosie_pies_l786_78698


namespace equal_angles_not_always_vertical_l786_78640

-- Define the concept of an angle
def Angle : Type := ℝ

-- Define the concept of vertical angles
def are_vertical_angles (a b : Angle) : Prop := sorry

-- Define the equality of angles
def angle_equal (a b : Angle) : Prop := a = b

-- Theorem stating that equal angles are not necessarily vertical angles
theorem equal_angles_not_always_vertical :
  ∃ (a b : Angle), angle_equal a b ∧ ¬(are_vertical_angles a b) := by
  sorry

end equal_angles_not_always_vertical_l786_78640


namespace pen_notebook_difference_l786_78604

theorem pen_notebook_difference (notebooks pens : ℕ) : 
  notebooks = 30 →
  notebooks + pens = 110 →
  pens > notebooks →
  pens - notebooks = 50 := by
sorry

end pen_notebook_difference_l786_78604


namespace divisors_540_multiple_of_two_l786_78644

/-- The number of positive divisors of 540 that are multiples of 2 -/
def divisors_multiple_of_two (n : ℕ) : ℕ :=
  (Finset.filter (λ d => d > 0 ∧ n % d = 0 ∧ d % 2 = 0) (Finset.range (n + 1))).card

/-- Theorem stating that the number of positive divisors of 540 that are multiples of 2 is 16 -/
theorem divisors_540_multiple_of_two :
  divisors_multiple_of_two 540 = 16 := by
  sorry

end divisors_540_multiple_of_two_l786_78644


namespace constant_path_mapping_l786_78686

/-- Given two segments AB and A'B' with their respective midpoints D and D', 
    prove that for any point P on AB with distance x from D, 
    and its associated point P' on A'B' with distance y from D', x + y = 6.5 -/
theorem constant_path_mapping (AB A'B' : ℝ) (D D' x y : ℝ) : 
  AB = 5 →
  A'B' = 8 →
  D = AB / 2 →
  D' = A'B' / 2 →
  x + y + D + D' = AB + A'B' →
  x + y = 6.5 := by
  sorry

end constant_path_mapping_l786_78686


namespace fraction_problem_l786_78648

theorem fraction_problem (x : ℚ) : 
  (5 / 6 : ℚ) * 576 = x * 576 + 300 → x = 5 / 16 := by
  sorry

end fraction_problem_l786_78648


namespace quadratic_residue_mod_prime_l786_78615

theorem quadratic_residue_mod_prime (p : Nat) (h_prime : Nat.Prime p) (h_odd : p % 2 = 1) :
  (∃ a : Int, (a ^ 2) % p = (p - 1) % p) ↔ p % 4 = 1 := by
  sorry

end quadratic_residue_mod_prime_l786_78615


namespace difference_c_minus_a_l786_78629

theorem difference_c_minus_a (a b c d k : ℝ) : 
  (a + b) / 2 = 45 →
  (b + c) / 2 = 50 →
  (a + c + d) / 3 = 60 →
  a^2 + b^2 + c^2 + d^2 = k →
  c - a = 10 := by
sorry

end difference_c_minus_a_l786_78629


namespace min_value_of_expression_min_value_attained_l786_78695

theorem min_value_of_expression (x : ℝ) : 
  (15 - x) * (13 - x) * (15 + x) * (13 + x) ≥ -784 :=
by sorry

theorem min_value_attained : 
  ∃ x : ℝ, (15 - x) * (13 - x) * (15 + x) * (13 + x) = -784 :=
by sorry

end min_value_of_expression_min_value_attained_l786_78695


namespace inequality_solution_set_l786_78638

theorem inequality_solution_set :
  {x : ℝ | (1 : ℝ) / x < (1 : ℝ) / 2} = {x : ℝ | x < 0 ∨ x > 2} := by sorry

end inequality_solution_set_l786_78638


namespace parallelepiped_volume_l786_78607

/-- The volume of a rectangular parallelepiped with given diagonal, angle, and base perimeter. -/
theorem parallelepiped_volume (l P α : ℝ) (hl : l > 0) (hP : P > 0) (hα : 0 < α ∧ α < π / 2) :
  ∃ V : ℝ, V = (l * (P^2 - 4 * l^2 * Real.sin α ^ 2) * Real.cos α) / 8 ∧
    V > 0 ∧
    ∀ (x y h : ℝ),
      x > 0 → y > 0 → h > 0 →
      x + y = P / 2 →
      x^2 + y^2 = l^2 * Real.sin α ^ 2 →
      h = l * Real.cos α →
      V = x * y * h :=
by
  sorry

end parallelepiped_volume_l786_78607


namespace pinecrest_academy_ratio_l786_78667

theorem pinecrest_academy_ratio (j s : ℕ) (h1 : 3 * s = 6 * j) : s / j = 1 / 2 := by
  sorry

#check pinecrest_academy_ratio

end pinecrest_academy_ratio_l786_78667


namespace nested_root_simplification_l786_78617

theorem nested_root_simplification :
  (81 * Real.sqrt (27 * Real.sqrt 9)) ^ (1/4) = 3 * 9 ^ (1/4) := by
  sorry

end nested_root_simplification_l786_78617


namespace intersection_M_N_l786_78619

open Set

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - 4 > 0}
def N : Set ℝ := {x | x < 0}

-- State the theorem
theorem intersection_M_N : M ∩ N = {x | x < -2} := by sorry

end intersection_M_N_l786_78619


namespace difference_largest_smallest_n_l786_78628

-- Define a convex n-gon
def ConvexNGon (n : ℕ) := n ≥ 3

-- Define an odd prime number
def OddPrime (p : ℕ) := Nat.Prime p ∧ p % 2 = 1

-- Define the condition that all interior angles are odd primes
def AllAnglesOddPrime (n : ℕ) (angles : Fin n → ℕ) :=
  ∀ i, OddPrime (angles i)

-- Define the sum of interior angles of an n-gon
def InteriorAngleSum (n : ℕ) : ℕ := 180 * (n - 2)

-- Define the condition that the sum of angles equals the interior angle sum
def AnglesSumToInteriorSum (n : ℕ) (angles : Fin n → ℕ) :=
  (Finset.univ.sum angles) = InteriorAngleSum n

-- Main theorem
theorem difference_largest_smallest_n :
  ∃ (n_min n_max : ℕ),
    (ConvexNGon n_min ∧
     ∃ angles_min, AllAnglesOddPrime n_min angles_min ∧ AnglesSumToInteriorSum n_min angles_min) ∧
    (ConvexNGon n_max ∧
     ∃ angles_max, AllAnglesOddPrime n_max angles_max ∧ AnglesSumToInteriorSum n_max angles_max) ∧
    (∀ n, ConvexNGon n → 
      (∃ angles, AllAnglesOddPrime n angles ∧ AnglesSumToInteriorSum n angles) →
      n_min ≤ n ∧ n ≤ n_max) ∧
    n_max - n_min = 356 :=
sorry

end difference_largest_smallest_n_l786_78628


namespace at_least_one_product_leq_one_l786_78641

theorem at_least_one_product_leq_one (x y z : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0) (h_sum : x + y + z = 3) : 
  x * (x + y - z) ≤ 1 ∨ y * (y + z - x) ≤ 1 ∨ z * (z + x - y) ≤ 1 := by
  sorry

end at_least_one_product_leq_one_l786_78641


namespace angela_insect_count_l786_78609

theorem angela_insect_count (dean_insects jacob_insects angela_insects : ℕ) : 
  dean_insects = 30 →
  jacob_insects = 5 * dean_insects →
  angela_insects = jacob_insects / 2 →
  angela_insects = 75 := by
  sorry

end angela_insect_count_l786_78609


namespace sqrt_ceil_floor_sum_l786_78652

theorem sqrt_ceil_floor_sum : 
  ⌈Real.sqrt 50⌉ + ⌈Real.sqrt 150⌉ + ⌊Real.sqrt 350⌋ = 39 := by
  sorry

end sqrt_ceil_floor_sum_l786_78652


namespace cab_driver_income_l786_78614

/-- Given a cab driver's income for 5 days, prove that the income on the third day is $450 -/
theorem cab_driver_income (income : Fin 5 → ℕ) 
  (day1 : income 0 = 600)
  (day2 : income 1 = 250)
  (day4 : income 3 = 400)
  (day5 : income 4 = 800)
  (avg_income : (income 0 + income 1 + income 2 + income 3 + income 4) / 5 = 500) :
  income 2 = 450 := by
  sorry

end cab_driver_income_l786_78614


namespace curve_to_line_equation_l786_78662

theorem curve_to_line_equation (t : ℝ) (x y : ℝ) 
  (h1 : x = 3 * t + 5) (h2 : y = 5 * t - 3) : 
  y = (5 * x - 34) / 3 := by
  sorry

end curve_to_line_equation_l786_78662


namespace f_and_g_odd_and_increasing_l786_78663

-- Define the functions
def f (x : ℝ) := x * |x|
def g (x : ℝ) := x^3

-- Define what it means for a function to be odd
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define what it means for a function to be increasing
def is_increasing (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f x < f y

-- Theorem statement
theorem f_and_g_odd_and_increasing :
  (is_odd f ∧ is_increasing f) ∧ (is_odd g ∧ is_increasing g) :=
sorry

end f_and_g_odd_and_increasing_l786_78663


namespace min_people_to_ask_for_hat_color_l786_78622

/-- Represents the minimum number of people to ask to ensure a majority of truthful answers -/
def min_people_to_ask (knights : ℕ) (civilians : ℕ) : ℕ :=
  civilians + (civilians + 1)

/-- Theorem stating the minimum number of people to ask in the given scenario -/
theorem min_people_to_ask_for_hat_color (knights : ℕ) (civilians : ℕ) 
  (h1 : knights = 50) (h2 : civilians = 15) :
  min_people_to_ask knights civilians = 31 := by
  sorry

#eval min_people_to_ask 50 15

end min_people_to_ask_for_hat_color_l786_78622


namespace unique_k_for_pythagorean_like_equation_l786_78671

theorem unique_k_for_pythagorean_like_equation :
  ∃! k : ℕ+, ∃ a b : ℕ+, a^2 + b^2 = k * a * b := by sorry

end unique_k_for_pythagorean_like_equation_l786_78671


namespace deepak_present_age_l786_78601

-- Define the ages as natural numbers
variable (R D : ℕ)

-- Define the conditions
def ratio_condition : Prop := 4 * D = 3 * R
def future_age_condition : Prop := R + 6 = 26

-- Theorem statement
theorem deepak_present_age 
  (h1 : ratio_condition R D) 
  (h2 : future_age_condition R) : 
  D = 15 := by sorry

end deepak_present_age_l786_78601


namespace marble_arrangements_mod_1000_l786_78689

/-- The number of blue marbles --/
def blue_marbles : ℕ := 6

/-- The maximum number of yellow marbles that maintains the balance --/
def yellow_marbles : ℕ := 18

/-- The total number of marbles --/
def total_marbles : ℕ := blue_marbles + yellow_marbles

/-- The number of different arrangements --/
def arrangements : ℕ := Nat.choose total_marbles blue_marbles

theorem marble_arrangements_mod_1000 :
  arrangements % 1000 = 564 := by sorry

end marble_arrangements_mod_1000_l786_78689


namespace rectangle_area_increase_rectangle_area_percentage_increase_l786_78653

theorem rectangle_area_increase (l w : ℝ) (hl : l > 0) (hw : w > 0) : 
  (1.3 * l) * (1.2 * w) = 1.56 * (l * w) := by
  sorry

theorem rectangle_area_percentage_increase :
  (1.56 - 1) * 100 = 56 := by
  sorry

end rectangle_area_increase_rectangle_area_percentage_increase_l786_78653


namespace two_books_different_genres_l786_78656

theorem two_books_different_genres (n : ℕ) (h : n = 4) : 
  (n.choose 2) * n * n = 96 :=
by sorry

end two_books_different_genres_l786_78656


namespace rachel_cookies_l786_78684

theorem rachel_cookies (mona jasmine rachel : ℕ) : 
  mona = 20 →
  jasmine = mona - 5 →
  rachel > jasmine →
  mona + jasmine + rachel = 60 →
  rachel = 25 := by
sorry

end rachel_cookies_l786_78684


namespace rahul_twice_mary_age_l786_78620

/-- Proves that Rahul will be twice as old as Mary after 20 years -/
theorem rahul_twice_mary_age : ∀ (x : ℕ),
  let mary_age : ℕ := 10
  let rahul_age : ℕ := mary_age + 30
  x = 20 ↔ rahul_age + x = 2 * (mary_age + x) :=
by sorry

end rahul_twice_mary_age_l786_78620


namespace rsa_factorization_l786_78647

theorem rsa_factorization :
  ∃ (p q : ℕ), 
    400000001 = p * q ∧ 
    Nat.Prime p ∧ 
    Nat.Prime q ∧ 
    p = 20201 ∧ 
    q = 19801 := by
  sorry

end rsa_factorization_l786_78647


namespace range_of_a_l786_78664

def P (a : ℝ) : Set ℝ := {x | a - 4 < x ∧ x < a + 4}
def Q : Set ℝ := {x | x^2 - 4*x + 3 < 0}

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Q, x ∈ P a) → -1 < a ∧ a < 5 :=
by sorry

end range_of_a_l786_78664


namespace coin_placement_coloring_l786_78600

theorem coin_placement_coloring (n : ℕ) (h1 : 1 < n) (h2 : n < 2010) :
  (∃ (coloring : Fin 2010 → Fin n) (initial_positions : Fin n → Fin 2010),
    ∀ (t : ℕ) (i j : Fin n),
      i ≠ j →
      coloring ((initial_positions i + t) % 2010) ≠
      coloring ((initial_positions j + t) % 2010)) ↔
  2010 % n = 0 :=
sorry

end coin_placement_coloring_l786_78600


namespace supplement_congruence_l786_78618

/-- Two angles are congruent if they have the same measure -/
def congruent_angles (α β : Real) : Prop := α = β

/-- The supplement of an angle is another angle that, when added to it, equals 180° -/
def supplement (α : Real) : Real := 180 - α

theorem supplement_congruence (α β : Real) :
  congruent_angles (supplement α) (supplement β) → congruent_angles α β := by
  sorry

end supplement_congruence_l786_78618


namespace triangle_third_side_length_l786_78621

theorem triangle_third_side_length
  (a b c : ℕ)
  (h1 : a = 2)
  (h2 : b = 5)
  (h3 : Odd c)
  (h4 : a + b > c)
  (h5 : b + c > a)
  (h6 : c + a > b) :
  c = 5 := by
sorry

end triangle_third_side_length_l786_78621


namespace inequality_proof_l786_78668

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b = 3 + b - a) : (3 / b) + (1 / a) ≥ 3 := by
  sorry

end inequality_proof_l786_78668


namespace inequality_equivalence_l786_78623

theorem inequality_equivalence (x : ℝ) : 
  (x ≠ 5) → ((x^2 + 2*x + 1) / ((x-5)^2) ≥ 15 ↔ 
    ((76 - 3*Real.sqrt 60) / 14 ≤ x ∧ x < 5) ∨ 
    (5 < x ∧ x ≤ (76 + 3*Real.sqrt 60) / 14)) := by
  sorry

end inequality_equivalence_l786_78623


namespace sum_of_a_and_a1_is_nine_l786_78633

theorem sum_of_a_and_a1_is_nine (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ : ℝ) :
  (∀ x : ℝ, (x + 1)^2 + (x + 1)^11 = a + a₁*(x + 2) + a₂*(x + 2)^2 + a₃*(x + 2)^3 + 
   a₄*(x + 2)^4 + a₅*(x + 2)^5 + a₆*(x + 2)^6 + a₇*(x + 2)^7 + a₈*(x + 2)^8 + 
   a₉*(x + 2)^9 + a₁₀*(x + 2)^10 + a₁₁*(x + 2)^11) →
  a + a₁ = 9 := by
sorry

end sum_of_a_and_a1_is_nine_l786_78633


namespace expand_expression_l786_78697

theorem expand_expression (x y z : ℝ) : 
  (2*x + 5) * (3*y + 15 + 4*z) = 6*x*y + 30*x + 8*x*z + 15*y + 20*z + 75 := by
  sorry

end expand_expression_l786_78697


namespace cloth_cost_price_l786_78627

/-- Proves that the cost price of one metre of cloth is 66.25,
    given the selling conditions of a cloth trader. -/
theorem cloth_cost_price
  (meters_sold : ℕ)
  (selling_price : ℚ)
  (profit_per_meter : ℚ)
  (h_meters : meters_sold = 80)
  (h_price : selling_price = 6900)
  (h_profit : profit_per_meter = 20) :
  (selling_price - meters_sold * profit_per_meter) / meters_sold = 66.25 := by
  sorry

end cloth_cost_price_l786_78627


namespace root_difference_range_l786_78679

noncomputable section

variables (a b c d : ℝ) (x₁ x₂ : ℝ)

def g (x : ℝ) := a * x^3 + b * x^2 + c * x + d

def f (x : ℝ) := 3 * a * x^2 + 2 * b * x + c

theorem root_difference_range (ha : a ≠ 0) 
  (h_sum : a + b + c = 0) 
  (h_prod : f 0 * f 1 > 0) 
  (h_roots : f x₁ = 0 ∧ f x₂ = 0) :
  ∃ (l u : ℝ), l = Real.sqrt 3 / 3 ∧ u = 2 / 3 ∧ 
  l ≤ |x₁ - x₂| ∧ |x₁ - x₂| < u :=
sorry

end root_difference_range_l786_78679


namespace triangle_inequality_l786_78631

theorem triangle_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b) :
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0 := by
  sorry

end triangle_inequality_l786_78631

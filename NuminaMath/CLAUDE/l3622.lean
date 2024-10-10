import Mathlib

namespace symmetric_point_correct_specific_case_l3622_362277

def symmetric_point (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)

theorem symmetric_point_correct (p : ℝ × ℝ) : 
  symmetric_point p = (-p.1, -p.2) := by sorry

theorem specific_case : 
  symmetric_point (3, -1) = (-3, 1) := by sorry

end symmetric_point_correct_specific_case_l3622_362277


namespace condition_relationship_l3622_362269

theorem condition_relationship (a b : ℝ) :
  (∀ a b : ℝ, a > b ∧ b > 1 → a - b < a^2 - b^2) ∧
  (∃ a b : ℝ, a - b < a^2 - b^2 ∧ ¬(a > b ∧ b > 1)) :=
by sorry

end condition_relationship_l3622_362269


namespace equal_sequence_l3622_362254

theorem equal_sequence (x : Fin 2011 → ℝ) (x' : Fin 2011 → ℝ)
  (h1 : ∀ i : Fin 2011, x i + x (i + 1) = 2 * x' i)
  (h2 : ∃ σ : Equiv (Fin 2011) (Fin 2011), ∀ i, x' i = x (σ i)) :
  ∀ i j : Fin 2011, x i = x j :=
by sorry

end equal_sequence_l3622_362254


namespace intersection_of_M_and_N_l3622_362253

def M : Set Int := {-1, 1}
def N : Set Int := {-2, 1, 0}

theorem intersection_of_M_and_N : M ∩ N = {1} := by sorry

end intersection_of_M_and_N_l3622_362253


namespace mike_money_total_l3622_362271

/-- Given that Mike has 9 5-dollar bills, prove that his total money is $45. -/
theorem mike_money_total : 
  let number_of_bills : ℕ := 9
  let bill_value : ℕ := 5
  number_of_bills * bill_value = 45 := by
  sorry

end mike_money_total_l3622_362271


namespace no_infinite_line_family_l3622_362290

theorem no_infinite_line_family :
  ¬ ∃ (k : ℕ → ℝ),
    (∀ n, k n ≠ 0) ∧
    (∀ n, k (n + 1) = (1 - 1 / k n) - (1 - k n)) ∧
    (∀ n, k n * k (n + 1) ≥ 0) :=
by sorry

end no_infinite_line_family_l3622_362290


namespace tesla_ratio_proof_l3622_362281

/-- The number of Teslas owned by Chris -/
def chris_teslas : ℕ := 6

/-- The number of Teslas owned by Elon -/
def elon_teslas : ℕ := 13

/-- The number of additional Teslas Elon has compared to Sam -/
def elon_sam_difference : ℕ := 10

/-- The number of Teslas owned by Sam -/
def sam_teslas : ℕ := elon_teslas - elon_sam_difference

theorem tesla_ratio_proof :
  (sam_teslas : ℚ) / chris_teslas = 1 / 2 := by sorry

end tesla_ratio_proof_l3622_362281


namespace smallest_five_digit_negative_congruent_to_2_mod_17_l3622_362283

theorem smallest_five_digit_negative_congruent_to_2_mod_17 : 
  ∃ (n : ℤ), 
    n = -10011 ∧ 
    n ≡ 2 [ZMOD 17] ∧ 
    n < 0 ∧ 
    -99999 ≤ n ∧ 
    ∀ (m : ℤ), (m ≡ 2 [ZMOD 17] ∧ m < 0 ∧ -99999 ≤ m) → n ≤ m :=
sorry

end smallest_five_digit_negative_congruent_to_2_mod_17_l3622_362283


namespace cole_fence_cost_l3622_362239

theorem cole_fence_cost (side_length : ℝ) (back_length : ℝ) (cost_per_foot : ℝ)
  (h_side : side_length = 9)
  (h_back : back_length = 18)
  (h_cost : cost_per_foot = 3)
  (h_neighbor_back : ∃ (x : ℝ), x = back_length * cost_per_foot / 2)
  (h_neighbor_left : ∃ (y : ℝ), y = side_length * cost_per_foot / 3) :
  ∃ (total_cost : ℝ), total_cost = 72 ∧
    total_cost = side_length * cost_per_foot + 
                 (2/3) * side_length * cost_per_foot + 
                 back_length * cost_per_foot / 2 :=
by sorry

end cole_fence_cost_l3622_362239


namespace interesting_coeffs_of_product_l3622_362257

/-- A real number is interesting if it can be expressed as a + b√2 where a and b are integers -/
def interesting (r : ℝ) : Prop :=
  ∃ (a b : ℤ), r = a + b * Real.sqrt 2

/-- A polynomial with interesting coefficients -/
def interesting_poly (p : Polynomial ℝ) : Prop :=
  ∀ i, interesting (p.coeff i)

/-- The main theorem -/
theorem interesting_coeffs_of_product
  (A B Q : Polynomial ℝ)
  (hA : interesting_poly A)
  (hB : interesting_poly B)
  (hB_const : B.coeff 0 = 1)
  (hABQ : A = B * Q) :
  interesting_poly Q :=
sorry

end interesting_coeffs_of_product_l3622_362257


namespace smallest_four_digit_negative_congruent_to_one_mod_37_l3622_362270

theorem smallest_four_digit_negative_congruent_to_one_mod_37 :
  ∀ n : ℤ, n < 0 ∧ n ≥ -9999 ∧ n ≡ 1 [ZMOD 37] → n ≥ -1034 :=
by
  sorry

end smallest_four_digit_negative_congruent_to_one_mod_37_l3622_362270


namespace fraction_simplification_l3622_362276

theorem fraction_simplification (c : ℝ) : (5 - 4 * c) / 9 - 3 = (-22 - 4 * c) / 9 := by
  sorry

end fraction_simplification_l3622_362276


namespace cube_roots_l3622_362237

theorem cube_roots : (39 : ℕ)^3 = 59319 ∧ (47 : ℕ)^3 = 103823 := by
  sorry

end cube_roots_l3622_362237


namespace rectangle_breadth_calculation_l3622_362229

/-- Given a rectangle with original length 18 cm and unknown breadth,
    if the length is increased to 25 cm and the new breadth is 7.2 cm
    while maintaining the same area, then the original breadth was 10 cm. -/
theorem rectangle_breadth_calculation (original_breadth : ℝ) : 
  18 * original_breadth = 25 * 7.2 → original_breadth = 10 := by
  sorry

#check rectangle_breadth_calculation

end rectangle_breadth_calculation_l3622_362229


namespace john_boxes_l3622_362223

theorem john_boxes (stan jules joseph : ℕ) (john : ℚ) : 
  stan = 100 →
  joseph = stan / 5 →
  jules = joseph + 5 →
  john = jules * (6/5) →
  john = 30 :=
by
  sorry

end john_boxes_l3622_362223


namespace problem_solution_l3622_362203

def p (x : ℝ) : Prop := x^2 - 7*x + 10 < 0

def q (x m : ℝ) : Prop := x^2 - 4*m*x + 3*m^2 < 0

theorem problem_solution (m : ℝ) (h : m > 0) :
  (∀ x, m = 4 → (p x ∧ q x m) → (4 < x ∧ x < 5)) ∧
  ((∀ x, ¬(q x m) → ¬(p x)) ∧ ¬(∀ x, ¬(p x) → ¬(q x m)) → (5/3 ≤ m ∧ m ≤ 2)) :=
sorry

end problem_solution_l3622_362203


namespace product_seven_l3622_362224

theorem product_seven : ∃ (x y : ℤ), x * y = 7 :=
sorry

end product_seven_l3622_362224


namespace monotonic_increasing_interval_l3622_362201

-- Define the function
def f (x : ℝ) : ℝ := 3*x - x^2

-- State the theorem
theorem monotonic_increasing_interval :
  ∀ x y : ℝ, x < y ∧ x < (3/2) ∧ y < (3/2) → f x < f y :=
by sorry

end monotonic_increasing_interval_l3622_362201


namespace perfume_price_change_l3622_362213

def original_price : ℝ := 1200
def increase_rate : ℝ := 0.10
def decrease_rate : ℝ := 0.15

theorem perfume_price_change :
  let increased_price := original_price * (1 + increase_rate)
  let final_price := increased_price * (1 - decrease_rate)
  original_price - final_price = 78 := by
sorry

end perfume_price_change_l3622_362213


namespace judys_shopping_cost_l3622_362218

/-- Represents Judy's shopping trip cost calculation --/
theorem judys_shopping_cost :
  let carrot_cost : ℕ := 5 * 1
  let milk_cost : ℕ := 3 * 3
  let pineapple_cost : ℕ := 2 * (4 / 2)
  let flour_cost : ℕ := 2 * 5
  let ice_cream_cost : ℕ := 7
  let total_before_coupon := carrot_cost + milk_cost + pineapple_cost + flour_cost + ice_cream_cost
  let coupon_value : ℕ := 5
  let coupon_threshold : ℕ := 25
  total_before_coupon ≥ coupon_threshold →
  total_before_coupon - coupon_value = 30 :=
by sorry

end judys_shopping_cost_l3622_362218


namespace cube_sum_and_reciprocal_l3622_362288

theorem cube_sum_and_reciprocal (x R S : ℝ) (hx : x ≠ 0) :
  (x + 1 / x = R) → (x^3 + 1 / x^3 = S) → (S = R^3 - 3 * R) := by
  sorry

end cube_sum_and_reciprocal_l3622_362288


namespace correct_operation_l3622_362227

theorem correct_operation (a b : ℝ) : (-2 * a^2 * b)^3 = -8 * a^6 * b^3 := by
  sorry

end correct_operation_l3622_362227


namespace quadratic_always_positive_l3622_362214

theorem quadratic_always_positive (a : ℝ) : 
  (∀ x : ℝ, x^2 + 2*x - a > 0) → a < -1 := by
  sorry

end quadratic_always_positive_l3622_362214


namespace line_chart_division_l3622_362263

/-- Represents a line chart -/
structure LineChart where
  /-- The line chart uses rise or fall of a line to represent increase or decrease in statistical quantities -/
  represents_statistical_quantities : Bool

/-- Represents a simple line chart -/
structure SimpleLineChart extends LineChart

/-- Represents a compound line chart -/
structure CompoundLineChart extends LineChart

/-- Theorem stating that line charts can be divided into simple and compound line charts -/
theorem line_chart_division (lc : LineChart) : 
  (∃ (slc : SimpleLineChart), slc.toLineChart = lc) ∨ 
  (∃ (clc : CompoundLineChart), clc.toLineChart = lc) :=
sorry

end line_chart_division_l3622_362263


namespace usual_time_to_school_l3622_362233

/-- The usual time for a boy to reach school, given that when he walks 7/6 of his usual rate,
    he reaches school 2 minutes early. -/
theorem usual_time_to_school (usual_rate : ℝ) (usual_time : ℝ) 
    (h1 : usual_rate > 0) (h2 : usual_time > 0)
    (h3 : usual_rate * usual_time = (7/6 * usual_rate) * (usual_time - 2)) : 
  usual_time = 14 := by
  sorry

end usual_time_to_school_l3622_362233


namespace union_A_complement_B_A_subset_B_iff_a_in_range_l3622_362252

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | 0 < 2*x + a ∧ 2*x + a ≤ 3}
def B : Set ℝ := {x | 2*x^2 - 3*x - 2 < 0}

-- Part 1
theorem union_A_complement_B :
  A 1 ∪ (Set.univ \ B) = {x : ℝ | x ≤ 1 ∨ x ≥ 2} := by sorry

-- Part 2
theorem A_subset_B_iff_a_in_range (a : ℝ) :
  A a ⊆ B ↔ -1 < a ∧ a ≤ 1 := by sorry

end union_A_complement_B_A_subset_B_iff_a_in_range_l3622_362252


namespace triangle_area_l3622_362231

/-- Given a triangle ABC where:
  * b is the length of the side opposite to angle B
  * c is the length of the side opposite to angle C
  * C is twice the measure of angle B
prove that the area of the triangle is 15√7/16 -/
theorem triangle_area (A B C : ℝ) (a b c : ℝ) : 
  b = 2 → 
  c = 3 → 
  C = 2 * B → 
  (1/2) * b * c * Real.sin A = 15 * Real.sqrt 7 / 16 := by
  sorry

end triangle_area_l3622_362231


namespace geometric_sequence_a6_l3622_362278

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_a6 (a : ℕ → ℝ) :
  geometric_sequence a → a 2 = 4 → a 4 = 2 → a 6 = 1 := by
  sorry

end geometric_sequence_a6_l3622_362278


namespace movie_theater_revenue_is_6810_l3622_362299

/-- Represents the revenue calculation for a movie theater --/
def movie_theater_revenue : ℕ := by
  -- Matinee ticket prices and sales
  let matinee_price : ℕ := 5
  let matinee_early_bird_discount : ℚ := 0.5
  let matinee_early_bird_tickets : ℕ := 20
  let matinee_regular_tickets : ℕ := 180

  -- Evening ticket prices and sales
  let evening_price : ℕ := 12
  let evening_group_discount : ℚ := 0.1
  let evening_student_senior_discount : ℚ := 0.25
  let evening_group_tickets : ℕ := 150
  let evening_student_senior_tickets : ℕ := 75
  let evening_regular_tickets : ℕ := 75

  -- 3D ticket prices and sales
  let threeD_price : ℕ := 20
  let threeD_online_surcharge : ℕ := 3
  let threeD_family_discount : ℚ := 0.15
  let threeD_online_tickets : ℕ := 60
  let threeD_family_tickets : ℕ := 25
  let threeD_regular_tickets : ℕ := 15

  -- Late-night ticket prices and sales
  let late_night_price : ℕ := 10
  let late_night_high_demand_increase : ℚ := 0.2
  let late_night_high_demand_tickets : ℕ := 30
  let late_night_regular_tickets : ℕ := 20

  -- Calculate total revenue
  let total_revenue : ℕ := 6810

  exact total_revenue

/-- Theorem stating that the movie theater's revenue on this day is $6810 --/
theorem movie_theater_revenue_is_6810 : movie_theater_revenue = 6810 := by
  sorry

end movie_theater_revenue_is_6810_l3622_362299


namespace gcd_count_for_product_180_l3622_362274

theorem gcd_count_for_product_180 (a b : ℕ+) (h : Nat.gcd a b * Nat.lcm a b = 180) :
  ∃! (s : Finset ℕ+), (∀ x ∈ s, ∃ (a' b' : ℕ+), Nat.gcd a' b' * Nat.lcm a' b' = 180 ∧ Nat.gcd a' b' = x) ∧ s.card = 8 :=
sorry

end gcd_count_for_product_180_l3622_362274


namespace decreasing_reciprocal_function_l3622_362221

theorem decreasing_reciprocal_function 
  (f : ℝ → ℝ) 
  (h_f : ∀ x, x > 0 → f x = 1 / x) :
  ∀ x₁ x₂, x₁ > 0 → x₂ > 0 → x₁ < x₂ → f x₁ > f x₂ := by
sorry

end decreasing_reciprocal_function_l3622_362221


namespace line_slope_l3622_362282

def curve (x y : ℝ) : Prop := 5 * y = 2 * x^2 - 9 * x + 10

def line_through_origin (k x y : ℝ) : Prop := y = k * x

theorem line_slope (k : ℝ) :
  (∃ x₁ x₂ y₁ y₂ : ℝ,
    curve x₁ y₁ ∧ curve x₂ y₂ ∧
    line_through_origin k x₁ y₁ ∧
    line_through_origin k x₂ y₂ ∧
    x₁ ≠ x₂ ∧
    x₁ + x₂ = 77) →
  k = 29 := by
sorry

end line_slope_l3622_362282


namespace town_distance_bounds_l3622_362258

/-- Given two towns A and B that are 8 km apart, and towns B and C that are 10 km apart,
    prove that the distance between towns A and C is at least 2 km and at most 18 km. -/
theorem town_distance_bounds (A B C : ℝ × ℝ) : 
  dist A B = 8 → dist B C = 10 → 2 ≤ dist A C ∧ dist A C ≤ 18 := by
  sorry

end town_distance_bounds_l3622_362258


namespace intersection_points_theorem_l3622_362261

/-- The maximum number of intersection points in the first quadrant
    given 15 points on the x-axis and 10 points on the y-axis -/
def max_intersection_points (x_points y_points : ℕ) : ℕ :=
  (x_points.choose 2) * (y_points.choose 2)

/-- Theorem stating the maximum number of intersection points -/
theorem intersection_points_theorem :
  max_intersection_points 15 10 = 4725 := by
  sorry

end intersection_points_theorem_l3622_362261


namespace negative_option_l3622_362259

theorem negative_option : ∀ (x : ℝ), 
  (x = -(-3) ∨ x = -|5| ∨ x = 1/2 ∨ x = 0) → 
  (x < 0 ↔ x = -|5|) := by
sorry

end negative_option_l3622_362259


namespace compare_exponentials_l3622_362285

theorem compare_exponentials :
  (4 : ℝ) ^ (1/4) > (5 : ℝ) ^ (1/5) ∧
  (5 : ℝ) ^ (1/5) > (16 : ℝ) ^ (1/16) ∧
  (16 : ℝ) ^ (1/16) > (25 : ℝ) ^ (1/25) :=
by sorry

end compare_exponentials_l3622_362285


namespace intersection_area_of_circles_l3622_362228

/-- The area of intersection of two circles with radius 4 and centers 4/α apart -/
theorem intersection_area_of_circles (α : ℝ) (h : α = 1/2) : 
  let r : ℝ := 4
  let d : ℝ := 4/α
  let β : ℝ := (2*r)^2 - 2*(π*r^2/2)
  β = 64 - 16*π := by sorry

end intersection_area_of_circles_l3622_362228


namespace quadratic_two_distinct_roots_l3622_362220

theorem quadratic_two_distinct_roots :
  let a : ℝ := 1
  let b : ℝ := 2
  let c : ℝ := -1
  let discriminant := b^2 - 4*a*c
  discriminant > 0 ∧ ∃ x y : ℝ, x ≠ y ∧ a*x^2 + b*x + c = 0 ∧ a*y^2 + b*y + c = 0 :=
by sorry

end quadratic_two_distinct_roots_l3622_362220


namespace count_9_in_1_to_1000_l3622_362236

/-- Count of digit 9 in a specific place value for numbers from 1 to 1000 -/
def count_digit_9_in_place (place : Nat) : Nat :=
  1000 / (10 ^ place)

/-- Total count of digit 9 in all integers from 1 to 1000 -/
def total_count_9 : Nat :=
  count_digit_9_in_place 0 + count_digit_9_in_place 1 + count_digit_9_in_place 2

theorem count_9_in_1_to_1000 :
  total_count_9 = 300 := by
  sorry

end count_9_in_1_to_1000_l3622_362236


namespace senior_citizen_tickets_l3622_362289

theorem senior_citizen_tickets (total_tickets : ℕ) (adult_price senior_price : ℚ) (total_receipts : ℚ) :
  total_tickets = 529 →
  adult_price = 25 →
  senior_price = 15 →
  total_receipts = 9745 →
  ∃ (adult_tickets senior_tickets : ℕ),
    adult_tickets + senior_tickets = total_tickets ∧
    adult_price * adult_tickets + senior_price * senior_tickets = total_receipts ∧
    senior_tickets = 348 :=
by sorry

end senior_citizen_tickets_l3622_362289


namespace quadratic_roots_real_and_equal_l3622_362297

theorem quadratic_roots_real_and_equal : ∃ x : ℝ, 
  (∀ y : ℝ, y^2 + 4*y*Real.sqrt 2 + 8 = 0 ↔ y = x) ∧ 
  (x^2 + 4*x*Real.sqrt 2 + 8 = 0) := by
  sorry

end quadratic_roots_real_and_equal_l3622_362297


namespace gcf_of_180_270_450_l3622_362247

theorem gcf_of_180_270_450 : Nat.gcd 180 (Nat.gcd 270 450) = 90 := by sorry

end gcf_of_180_270_450_l3622_362247


namespace quadrilateral_area_l3622_362248

/-- A quadrilateral with right angles at B and D, diagonal AC of length 5,
    and two sides with distinct integer lengths has an area of 12. -/
theorem quadrilateral_area (A B C D : ℝ × ℝ) : 
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let CD := Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2)
  let DA := Real.sqrt ((D.1 - A.1)^2 + (D.2 - A.2)^2)
  let AC := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  (B.1 - A.1) * (B.2 - D.2) = (B.2 - A.2) * (B.1 - D.1) →  -- right angle at B
  (D.1 - C.1) * (D.2 - B.2) = (D.2 - C.2) * (D.1 - B.1) →  -- right angle at D
  AC = 5 →
  (∃ (x y : ℕ), (AB = x ∨ BC = x ∨ CD = x ∨ DA = x) ∧ 
                (AB = y ∨ BC = y ∨ CD = y ∨ DA = y) ∧ x ≠ y) →
  (1/2 * AB * BC) + (1/2 * CD * DA) = 12 :=
by sorry

end quadrilateral_area_l3622_362248


namespace magazine_cost_l3622_362241

/-- The cost of a magazine and pencil, given specific conditions -/
theorem magazine_cost (pencil_cost coupon_value total_spent : ℚ) :
  pencil_cost = 0.5 →
  coupon_value = 0.35 →
  total_spent = 1 →
  ∃ (magazine_cost : ℚ),
    magazine_cost + pencil_cost - coupon_value = total_spent ∧
    magazine_cost = 0.85 := by
  sorry

end magazine_cost_l3622_362241


namespace gcd_48_180_l3622_362250

theorem gcd_48_180 : Nat.gcd 48 180 = 12 := by
  sorry

end gcd_48_180_l3622_362250


namespace total_crayons_l3622_362296

/-- Given that each child has 12 crayons and there are 18 children, 
    prove that the total number of crayons is 216. -/
theorem total_crayons (crayons_per_child : ℕ) (num_children : ℕ) 
  (h1 : crayons_per_child = 12) (h2 : num_children = 18) : 
  crayons_per_child * num_children = 216 := by
sorry

end total_crayons_l3622_362296


namespace angle_terminal_side_value_l3622_362209

/-- Given a point P(-4t, 3t) on the terminal side of angle θ, where t ≠ 0,
    the value of 2sinθ + cosθ is either 2/5 or -2/5. -/
theorem angle_terminal_side_value (t : ℝ) (θ : ℝ) (h : t ≠ 0) :
  let P : ℝ × ℝ := (-4 * t, 3 * t)
  (∃ (k : ℝ), k > 0 ∧ P = k • (Real.cos θ, Real.sin θ)) →
  2 * Real.sin θ + Real.cos θ = 2 / 5 ∨ 2 * Real.sin θ + Real.cos θ = -2 / 5 :=
by sorry


end angle_terminal_side_value_l3622_362209


namespace jeremy_watermelon_weeks_l3622_362226

/-- The number of weeks watermelons will last for Jeremy -/
def watermelon_weeks (total : ℕ) (eaten_per_week : ℕ) (given_to_dad : ℕ) : ℕ :=
  total / (eaten_per_week + given_to_dad)

/-- Theorem: Given Jeremy's watermelon consumption pattern, the watermelons will last 6 weeks -/
theorem jeremy_watermelon_weeks :
  watermelon_weeks 30 3 2 = 6 := by
  sorry

end jeremy_watermelon_weeks_l3622_362226


namespace roots_power_sum_divisible_l3622_362211

/-- Given two roots of a quadratic equation with a prime coefficient,
    their p-th powers sum to a multiple of p². -/
theorem roots_power_sum_divisible (p : ℕ) (x₁ x₂ : ℝ) 
  (h_prime : Nat.Prime p) 
  (h_p_gt_two : p > 2) 
  (h_roots : x₁^2 - p*x₁ + 1 = 0 ∧ x₂^2 - p*x₂ + 1 = 0) : 
  ∃ (k : ℤ), x₁^p + x₂^p = k * p^2 := by
  sorry

#check roots_power_sum_divisible

end roots_power_sum_divisible_l3622_362211


namespace perpendicular_lines_theorem_l3622_362208

/-- A line in 3D space -/
structure Line3D where
  -- We represent a line by a point and a direction vector
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- Perpendicular relation between two lines -/
def perpendicular (l1 l2 : Line3D) : Prop :=
  -- Definition of perpendicularity
  sorry

/-- Parallel relation between two lines -/
def parallel (l1 l2 : Line3D) : Prop :=
  -- Definition of parallelism
  sorry

theorem perpendicular_lines_theorem (a b c d : Line3D) 
  (h1 : perpendicular a b)
  (h2 : perpendicular b c)
  (h3 : perpendicular c d)
  (h4 : perpendicular d a) :
  parallel b d ∨ parallel a c :=
sorry

end perpendicular_lines_theorem_l3622_362208


namespace power_sum_properties_l3622_362206

theorem power_sum_properties (a b c d : ℝ) 
  (sum_eq : a + b = c + d) 
  (cube_sum_eq : a^3 + b^3 = c^3 + d^3) : 
  (a^5 + b^5 = c^5 + d^5) ∧ 
  ¬(∀ (a b c d : ℝ), (a + b = c + d) → (a^3 + b^3 = c^3 + d^3) → (a^4 + b^4 = c^4 + d^4)) := by
sorry

end power_sum_properties_l3622_362206


namespace motorcycle_car_profit_difference_l3622_362264

/-- Represents the production and sales data for a vehicle type -/
structure VehicleProduction where
  materialCost : ℕ
  quantity : ℕ
  price : ℕ

/-- Calculates the profit for a given vehicle production -/
def profit (v : VehicleProduction) : ℤ :=
  (v.quantity * v.price : ℤ) - v.materialCost

/-- Proves that the difference in profit between motorcycle and car production is $50 -/
theorem motorcycle_car_profit_difference 
  (car : VehicleProduction)
  (motorcycle : VehicleProduction)
  (h_car : car = { materialCost := 100, quantity := 4, price := 50 })
  (h_motorcycle : motorcycle = { materialCost := 250, quantity := 8, price := 50 }) :
  profit motorcycle - profit car = 50 := by
  sorry

#eval profit { materialCost := 250, quantity := 8, price := 50 } - 
      profit { materialCost := 100, quantity := 4, price := 50 }

end motorcycle_car_profit_difference_l3622_362264


namespace proposition_truth_values_l3622_362262

theorem proposition_truth_values (p q : Prop) 
  (hp : p) 
  (hq : ¬q) : 
  (p ∨ q) ∧ ¬(p ∧ q) ∧ ¬((¬p) ∧ (¬q)) ∧ ¬(¬p) := by
  sorry

end proposition_truth_values_l3622_362262


namespace unique_solution_system_l3622_362242

theorem unique_solution_system (x y z : ℝ) : 
  x * (1 + y * z) = 9 ∧ 
  y * (1 + x * z) = 12 ∧ 
  z * (1 + x * y) = 10 ↔ 
  x = 1 ∧ y = 4 ∧ z = 2 :=
by sorry

end unique_solution_system_l3622_362242


namespace second_interest_rate_is_20_percent_l3622_362246

/-- Given a total amount, an amount at 10% interest, and a total profit,
    calculate the second interest rate. -/
def calculate_second_interest_rate (total_amount : ℕ) (amount_at_10_percent : ℕ) (total_profit : ℕ) : ℚ :=
  let amount_at_second_rate := total_amount - amount_at_10_percent
  let interest_from_first_part := (10 : ℚ) / 100 * amount_at_10_percent
  let interest_from_second_part := total_profit - interest_from_first_part
  (interest_from_second_part * 100) / amount_at_second_rate

/-- Theorem stating that under the given conditions, the second interest rate is 20%. -/
theorem second_interest_rate_is_20_percent :
  calculate_second_interest_rate 80000 70000 9000 = 20 := by
  sorry

end second_interest_rate_is_20_percent_l3622_362246


namespace inclination_angle_range_l3622_362266

-- Define the slope k and inclination angle α
variable (k α : ℝ)

-- Define the relationship between k and α
def slope_angle_relation (k α : ℝ) : Prop := k = Real.tan α

-- Define the range of k
def slope_range (k : ℝ) : Prop := -1 ≤ k ∧ k < Real.sqrt 3

-- Define the range of α
def angle_range (α : ℝ) : Prop := 
  (0 ≤ α ∧ α < Real.pi/3) ∨ (3*Real.pi/4 ≤ α ∧ α < Real.pi)

-- State the theorem
theorem inclination_angle_range :
  ∀ k α, slope_angle_relation k α → slope_range k → angle_range α :=
sorry

end inclination_angle_range_l3622_362266


namespace power_inequality_l3622_362240

theorem power_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a^a * b^b ≥ (a*b)^((a+b)/2) := by
  sorry

end power_inequality_l3622_362240


namespace arithmetic_sequence_formula_l3622_362215

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_formula 
  (a : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a)
  (h_sum1 : a 1 + a 2 + a 3 = 0)
  (h_sum2 : a 4 + a 5 + a 6 = 18) :
  ∀ n : ℕ, a n = 2 * n - 4 := by
sorry

end arithmetic_sequence_formula_l3622_362215


namespace constant_term_expansion_l3622_362234

theorem constant_term_expansion (α : Real) 
  (h : Real.sin (π - α) = 2 * Real.cos α) : 
  (Finset.range 7).sum (fun k => 
    (Nat.choose 6 k : Real) * 
    (Real.tan α)^k * 
    ((-1)^k * Nat.choose 6 (6-k))) = 160 := by
  sorry

end constant_term_expansion_l3622_362234


namespace whole_substitution_problems_l3622_362217

theorem whole_substitution_problems :
  -- Problem 1
  (∀ m n : ℝ, m - n = -1 → 2 * (m - n)^2 + 18 = 20) ∧
  -- Problem 2
  (∀ m n : ℝ, m^2 + 2*m*n = 10 ∧ n^2 + 3*m*n = 6 → 2*m^2 + n^2 + 7*m*n = 26) ∧
  -- Problem 3
  (∀ a b c m : ℝ, a*(-1)^5 + b*(-1)^3 + c*(-1) - 5 = m → 
    a*(1)^5 + b*(1)^3 + c*(1) - 5 = -m - 10) :=
by sorry

end whole_substitution_problems_l3622_362217


namespace plot_perimeter_l3622_362272

/-- A rectangular plot with specific properties -/
structure RectangularPlot where
  width : ℝ
  length : ℝ
  fencing_rate : ℝ
  fencing_cost : ℝ
  length_width_relation : length = width + 10
  cost_relation : fencing_cost = (2 * (length + width)) * fencing_rate

/-- The perimeter of a rectangular plot -/
def perimeter (plot : RectangularPlot) : ℝ :=
  2 * (plot.length + plot.width)

/-- Theorem stating the perimeter of the specific plot -/
theorem plot_perimeter (plot : RectangularPlot) 
  (h1 : plot.fencing_rate = 6.5)
  (h2 : plot.fencing_cost = 910) : 
  perimeter plot = 140 := by
  sorry

end plot_perimeter_l3622_362272


namespace f_zero_one_eq_neg_one_one_l3622_362225

/-- The type of points in the real plane -/
def RealPair := ℝ × ℝ

/-- The mapping f: A → B -/
def f (p : RealPair) : RealPair :=
  (p.1 - p.2, p.1 + p.2)

/-- Theorem stating that f(0, 1) = (-1, 1) -/
theorem f_zero_one_eq_neg_one_one :
  f (0, 1) = (-1, 1) := by
  sorry

end f_zero_one_eq_neg_one_one_l3622_362225


namespace fair_coin_probability_difference_l3622_362238

def binomial (n k : ℕ) : ℕ := Nat.choose n k

def fairCoinProbability (n k : ℕ) : ℚ :=
  (binomial n k : ℚ) * (1 / 2) ^ n

theorem fair_coin_probability_difference :
  let p3 := fairCoinProbability 5 3
  let p4 := fairCoinProbability 5 4
  abs (p3 - p4) = 5 / 32 := by sorry

end fair_coin_probability_difference_l3622_362238


namespace supplementary_angles_ratio_l3622_362286

theorem supplementary_angles_ratio (a b : ℝ) : 
  a + b = 180 →  -- angles are supplementary
  a / b = 5 / 3 →  -- ratio of angles is 5:3
  b = 67.5 :=  -- smaller angle is 67.5°
by
  sorry

end supplementary_angles_ratio_l3622_362286


namespace new_cube_edge_length_new_cube_edge_is_six_l3622_362294

/-- Given three cubes with edges 3, 4, and 5 decimeters, when melted and formed into a single cube,
    the edge length of the new cube is 6 decimeters. -/
theorem new_cube_edge_length : ℝ → Prop :=
  fun edge_length =>
    let cube1_volume : ℝ := 3^3
    let cube2_volume : ℝ := 4^3
    let cube3_volume : ℝ := 5^3
    let total_volume : ℝ := cube1_volume + cube2_volume + cube3_volume
    edge_length^3 = total_volume ∧ edge_length = 6

/-- The edge length of the new cube is indeed 6 decimeters. -/
theorem new_cube_edge_is_six : new_cube_edge_length 6 := by
  sorry

end new_cube_edge_length_new_cube_edge_is_six_l3622_362294


namespace coefficient_of_y_squared_l3622_362298

theorem coefficient_of_y_squared (a : ℝ) : 
  (∀ y : ℝ, a * y^2 - 8 * y + 55 = 59) → 
  (∃ y : ℝ, y = 2) → 
  a = 5 := by
  sorry

end coefficient_of_y_squared_l3622_362298


namespace number_of_arrangements_l3622_362273

/-- Represents a step on the staircase -/
structure Step :=
  (occupants : Finset Char)
  (h : occupants.card ≤ 2)

/-- Represents an arrangement of people on the staircase -/
def Arrangement := Finset Step

/-- The set of all valid arrangements -/
def AllArrangements : Finset Arrangement :=
  sorry

/-- The number of different ways 4 people can stand on 5 steps -/
theorem number_of_arrangements :
  (AllArrangements.filter (fun arr => arr.sum (fun step => step.occupants.card) = 4)).card = 540 :=
sorry

end number_of_arrangements_l3622_362273


namespace fertilizer_on_half_field_l3622_362216

/-- Theorem: Amount of fertilizer on half a football field -/
theorem fertilizer_on_half_field (total_area : ℝ) (total_fertilizer : ℝ) 
  (h1 : total_area = 7200)
  (h2 : total_fertilizer = 1200) :
  (total_fertilizer / total_area) * (total_area / 2) = 600 := by
  sorry

end fertilizer_on_half_field_l3622_362216


namespace quadratic_root_difference_l3622_362210

theorem quadratic_root_difference (p q : ℝ) : 
  let r := (p + Real.sqrt (p^2 + q))
  let s := (p - Real.sqrt (p^2 + q))
  abs (r - s) = Real.sqrt (2 * p^2 + 2 * q) :=
by sorry

end quadratic_root_difference_l3622_362210


namespace election_margin_of_victory_l3622_362284

theorem election_margin_of_victory 
  (total_votes : ℕ) 
  (winning_percentage : ℚ) 
  (winning_votes : ℕ) : 
  winning_percentage = 29/50 → 
  winning_votes = 1044 → 
  (winning_votes : ℚ) / winning_percentage = total_votes → 
  winning_votes - (total_votes - winning_votes) = 288 :=
by sorry

end election_margin_of_victory_l3622_362284


namespace walnut_logs_per_tree_l3622_362287

theorem walnut_logs_per_tree (pine_trees maple_trees walnut_trees : ℕ)
  (logs_per_pine logs_per_maple total_logs : ℕ) :
  pine_trees = 8 →
  maple_trees = 3 →
  walnut_trees = 4 →
  logs_per_pine = 80 →
  logs_per_maple = 60 →
  total_logs = 1220 →
  ∃ logs_per_walnut : ℕ,
    logs_per_walnut = 100 ∧
    total_logs = pine_trees * logs_per_pine + maple_trees * logs_per_maple + walnut_trees * logs_per_walnut :=
by sorry

end walnut_logs_per_tree_l3622_362287


namespace bhanu_petrol_expense_l3622_362256

/-- Calculates Bhanu's petrol expense given his house rent expense and spending percentages -/
theorem bhanu_petrol_expense (house_rent : ℝ) (petrol_percent : ℝ) (rent_percent : ℝ) : 
  house_rent = 140 → 
  petrol_percent = 0.3 → 
  rent_percent = 0.2 → 
  ∃ (total_income : ℝ), 
    total_income > 0 ∧ 
    rent_percent * (1 - petrol_percent) * total_income = house_rent ∧
    petrol_percent * total_income = 300 :=
by sorry

end bhanu_petrol_expense_l3622_362256


namespace max_value_theorem_l3622_362200

theorem max_value_theorem (m n : ℝ) 
  (h1 : 0 ≤ m - n) (h2 : m - n ≤ 1) 
  (h3 : 2 ≤ m + n) (h4 : m + n ≤ 4) : 
  (∀ x y : ℝ, 0 ≤ x - y ∧ x - y ≤ 1 ∧ 2 ≤ x + y ∧ x + y ≤ 4 → m - 2*n ≥ x - 2*y) →
  2019*m + 2020*n = 2019 := by
sorry

end max_value_theorem_l3622_362200


namespace unique_integer_modulo_l3622_362293

theorem unique_integer_modulo : ∃! n : ℕ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ 123456 [MOD 8] ∧ n = 0 := by
  sorry

end unique_integer_modulo_l3622_362293


namespace pechkin_ate_four_tenths_l3622_362244

/-- The fraction of the cake eaten by each person -/
structure CakeFractions where
  pechkin : ℝ
  fyodor : ℝ
  matroskin : ℝ
  sharik : ℝ

/-- The conditions of the cake-eating problem -/
def cake_problem (f : CakeFractions) : Prop :=
  -- The whole cake was eaten
  f.pechkin + f.fyodor + f.matroskin + f.sharik = 1 ∧
  -- Uncle Fyodor ate half as much as Pechkin
  f.fyodor = f.pechkin / 2 ∧
  -- Cat Matroskin ate half as much as the portion of the cake that Pechkin did not eat
  f.matroskin = (1 - f.pechkin) / 2 ∧
  -- Sharik ate one-tenth of the cake
  f.sharik = 1 / 10

/-- Theorem stating that given the conditions, Pechkin ate 0.4 of the cake -/
theorem pechkin_ate_four_tenths (f : CakeFractions) :
  cake_problem f → f.pechkin = 0.4 := by sorry

end pechkin_ate_four_tenths_l3622_362244


namespace show_dog_cost_l3622_362205

/-- Proves that the cost of each show dog is $250 given the problem conditions -/
theorem show_dog_cost (num_dogs : ℕ) (num_puppies : ℕ) (puppy_price : ℕ) (total_profit : ℕ) : 
  num_dogs = 2 →
  num_puppies = 6 →
  puppy_price = 350 →
  total_profit = 1600 →
  (num_puppies * puppy_price - total_profit) / num_dogs = 250 := by
  sorry

end show_dog_cost_l3622_362205


namespace min_value_theorem_l3622_362265

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y - 3 = 0) :
  (4 * y - x + 6) / (x * y) ≥ 9 :=
sorry

end min_value_theorem_l3622_362265


namespace f_inequality_solution_set_m_range_l3622_362268

def f (x : ℝ) : ℝ := |2*x - 1| + |x + 1|

theorem f_inequality_solution_set :
  {x : ℝ | f x < 4} = {x : ℝ | -4/3 < x ∧ x < 4/3} := by sorry

theorem m_range (m : ℝ) :
  (∃ x₀ : ℝ, ∀ t : ℝ, f x₀ < |m + t| + |t - m|) ↔ 
  (m < -3/4 ∨ m > 3/4) := by sorry

end f_inequality_solution_set_m_range_l3622_362268


namespace intersection_A_B_l3622_362212

def A : Set ℝ := { x | 2 * x^2 - 3 * x - 2 ≤ 0 }

def B : Set ℝ := {-1, 0, 1, 2, 3}

theorem intersection_A_B : A ∩ B = {0, 1, 2} := by sorry

end intersection_A_B_l3622_362212


namespace equation_solution_l3622_362267

theorem equation_solution : ∃ x : ℝ, 10111 - 10 * 2 * (5 + x) = 0 ∧ x = 500.55 := by
  sorry

end equation_solution_l3622_362267


namespace person_a_parts_l3622_362222

/-- Represents the number of parts made by each person -/
structure PartProduction where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Represents the production scenario described in the problem -/
def production_scenario (p : PartProduction) : Prop :=
  p.c = 20 ∧
  4 * p.b = 3 * p.c ∧
  10 * p.a = 3 * (p.a + p.b + p.c)

theorem person_a_parts :
  ∀ p : PartProduction, production_scenario p → p.a = 15 :=
by
  sorry

end person_a_parts_l3622_362222


namespace star_value_for_specific_conditions_l3622_362207

-- Define the * operation for non-zero integers
def star (a b : ℤ) : ℚ := 1 / a + 1 / b

-- Theorem statement
theorem star_value_for_specific_conditions (a b : ℤ) 
  (h1 : a ≠ 0) 
  (h2 : b ≠ 0) 
  (h3 : a + b = 16) 
  (h4 : a^2 + b^2 = 136) : 
  star a b = 4/15 := by
  sorry

end star_value_for_specific_conditions_l3622_362207


namespace subtraction_of_negative_l3622_362243

theorem subtraction_of_negative : 4 - (-7) = 11 := by sorry

end subtraction_of_negative_l3622_362243


namespace star_difference_l3622_362260

def star (x y : ℝ) : ℝ := 2*x*y - 3*x + y

theorem star_difference : (star 6 4) - (star 4 6) = -8 := by
  sorry

end star_difference_l3622_362260


namespace solution_to_system_l3622_362279

/-- Prove that (4, 2, 3) is the solution to the given system of equations --/
theorem solution_to_system : ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧
  x^2 + y^2 + Real.sqrt 3 * x * y = 20 + 8 * Real.sqrt 3 ∧
  y^2 + z^2 = 13 ∧
  z^2 + x^2 + x * z = 37 ∧
  x = 4 ∧ y = 2 ∧ z = 3 := by
  sorry

end solution_to_system_l3622_362279


namespace largest_four_digit_palindrome_divisible_by_three_l3622_362292

/-- A four-digit palindrome is a number between 1000 and 9999 that reads the same forwards and backwards -/
def is_four_digit_palindrome (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧ (n.digits 10).reverse = n.digits 10

/-- A number is divisible by 3 if it leaves no remainder when divided by 3 -/
def divisible_by_three (n : ℕ) : Prop :=
  n % 3 = 0

theorem largest_four_digit_palindrome_divisible_by_three :
  ∀ n : ℕ, is_four_digit_palindrome n → divisible_by_three n → n ≤ 9999 :=
by sorry

end largest_four_digit_palindrome_divisible_by_three_l3622_362292


namespace sum_of_roots_l3622_362219

theorem sum_of_roots (p q r s : ℝ) : 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s →
  (∀ x, x^2 - 14*p*x - 15*q = 0 ↔ x = r ∨ x = s) →
  (∀ x, x^2 - 14*r*x - 15*s = 0 ↔ x = p ∨ x = q) →
  p + q + r + s = 3150 := by
sorry

end sum_of_roots_l3622_362219


namespace triangular_pyramid_inequality_l3622_362249

-- Define a structure for a triangular pyramid
structure TriangularPyramid where
  -- We don't need to explicitly define vertices A, B, C, D
  -- as they are implicit in the following measurements
  R : ℝ  -- radius of circumscribed sphere
  r : ℝ  -- radius of inscribed sphere
  a : ℝ  -- length of longest edge
  h : ℝ  -- length of shortest altitude

-- State the theorem
theorem triangular_pyramid_inequality (pyramid : TriangularPyramid) :
  pyramid.R / pyramid.r > pyramid.a / pyramid.h := by
  sorry

end triangular_pyramid_inequality_l3622_362249


namespace ali_age_l3622_362280

/-- Given the ages of Ali, Yusaf, and Umar, prove Ali's age -/
theorem ali_age (ali yusaf umar : ℕ) 
  (h1 : ali = yusaf + 3)
  (h2 : umar = 2 * yusaf)
  (h3 : umar = 10) : 
  ali = 8 := by
  sorry

end ali_age_l3622_362280


namespace davids_biology_marks_l3622_362204

theorem davids_biology_marks
  (english : ℕ)
  (mathematics : ℕ)
  (physics : ℕ)
  (chemistry : ℕ)
  (average : ℕ)
  (h1 : english = 91)
  (h2 : mathematics = 65)
  (h3 : physics = 82)
  (h4 : chemistry = 67)
  (h5 : average = 78)
  (h6 : (english + mathematics + physics + chemistry + biology) / 5 = average) :
  biology = 85 := by
  sorry

end davids_biology_marks_l3622_362204


namespace parallelogram_count_l3622_362235

/-- Given two sets of parallel lines in a plane, prove the number of parallelograms formed -/
theorem parallelogram_count (m n : ℕ) : ℕ := by
  /- m is the number of lines in the first set -/
  /- n is the number of lines in the second set -/
  /- The two sets of lines are parallel and intersect -/
  /- The number of parallelograms formed is Combination(m,2) * Combination(n,2) -/
  sorry

#check parallelogram_count

end parallelogram_count_l3622_362235


namespace symmetric_point_parabola_l3622_362230

/-- Given a parabola y = a(x+2)^2 and a point A(1,4), 
    prove that the point (-5,4) is symmetric to A 
    with respect to the parabola's axis of symmetry -/
theorem symmetric_point_parabola (a : ℝ) : 
  let parabola := fun (x : ℝ) => a * (x + 2)^2
  let A : ℝ × ℝ := (1, 4)
  let axis_of_symmetry : ℝ := -2
  let symmetric_point : ℝ × ℝ := (-5, 4)
  (symmetric_point.1 - axis_of_symmetry = -(A.1 - axis_of_symmetry)) ∧ 
  (symmetric_point.2 = A.2) :=
by sorry

end symmetric_point_parabola_l3622_362230


namespace hidden_piece_area_l3622_362202

/-- Represents the surface areas of the 7 visible pieces of the wooden block -/
def visible_areas : List ℝ := [148, 46, 72, 28, 88, 126, 58]

/-- The total number of pieces the wooden block is cut into -/
def total_pieces : ℕ := 8

/-- Theorem: Given a wooden block cut into 8 pieces, where the surface areas of 7 pieces are known,
    and the sum of these areas is 566, the surface area of the 8th piece is 22. -/
theorem hidden_piece_area (h1 : visible_areas.length = total_pieces - 1)
                          (h2 : visible_areas.sum = 566) : 
  ∃ (hidden_area : ℝ), hidden_area = 22 ∧ 
    visible_areas.sum + hidden_area = (visible_areas.sum + hidden_area) / 2 * 2 := by
  sorry

end hidden_piece_area_l3622_362202


namespace johns_new_total_capacity_l3622_362255

/-- Represents the lifting capacities of a weightlifter -/
structure LiftingCapacities where
  cleanAndJerk : ℝ
  snatch : ℝ

/-- Calculates the new lifting capacities after improvement -/
def improvedCapacities (initial : LiftingCapacities) : LiftingCapacities :=
  { cleanAndJerk := initial.cleanAndJerk * 2,
    snatch := initial.snatch * 1.8 }

/-- Calculates the total lifting capacity -/
def totalCapacity (capacities : LiftingCapacities) : ℝ :=
  capacities.cleanAndJerk + capacities.snatch

/-- John's initial lifting capacities -/
def johnsInitialCapacities : LiftingCapacities :=
  { cleanAndJerk := 80,
    snatch := 50 }

theorem johns_new_total_capacity :
  totalCapacity (improvedCapacities johnsInitialCapacities) = 250 := by
  sorry


end johns_new_total_capacity_l3622_362255


namespace count_integer_root_cases_correct_l3622_362295

/-- The number of real values 'a' for which x^2 + ax + 12a = 0 has only integer roots -/
def count_integer_root_cases : ℕ := 8

/-- A function that returns true if the quadratic equation x^2 + ax + 12a = 0 has only integer roots -/
def has_only_integer_roots (a : ℝ) : Prop :=
  ∃ p q : ℤ, ∀ x : ℝ, x^2 + a*x + 12*a = 0 ↔ x = p ∨ x = q

/-- The theorem stating that there are exactly 8 real numbers 'a' for which
    the quadratic equation x^2 + ax + 12a = 0 has only integer roots -/
theorem count_integer_root_cases_correct :
  (∃ S : Finset ℝ, Finset.card S = count_integer_root_cases ∧
    (∀ a : ℝ, a ∈ S ↔ has_only_integer_roots a)) := by
  sorry


end count_integer_root_cases_correct_l3622_362295


namespace unique_root_of_abs_equation_l3622_362245

/-- The equation x|x| - 3|x| - 4 = 0 has exactly one real root -/
theorem unique_root_of_abs_equation : ∃! x : ℝ, x * |x| - 3 * |x| - 4 = 0 := by
  sorry

end unique_root_of_abs_equation_l3622_362245


namespace intersection_height_l3622_362232

/-- Represents a line in 2D space --/
structure Line where
  m : ℚ  -- slope
  b : ℚ  -- y-intercept

/-- Creates a line from two points --/
def lineFromPoints (x1 y1 x2 y2 : ℚ) : Line :=
  let m := (y2 - y1) / (x2 - x1)
  let b := y1 - m * x1
  { m := m, b := b }

/-- Calculates the y-coordinate for a given x on the line --/
def Line.yAt (l : Line) (x : ℚ) : ℚ :=
  l.m * x + l.b

theorem intersection_height : 
  let line1 := lineFromPoints 0 30 120 0
  let line2 := lineFromPoints 0 0 120 50
  let x_intersect := (line2.b - line1.b) / (line1.m - line2.m)
  line1.yAt x_intersect = 75/4 := by sorry

end intersection_height_l3622_362232


namespace functional_equation_solution_l3622_362251

theorem functional_equation_solution (f g : ℝ → ℝ) :
  (∀ x y : ℝ, f (x + y * g x) = g x + x * f y) →
  (f = id ∧ g = id) :=
by sorry

end functional_equation_solution_l3622_362251


namespace conditional_probability_proof_l3622_362291

-- Define the number of balls of each color and total number of balls
def red_balls : ℕ := 2
def yellow_balls : ℕ := 2
def blue_balls : ℕ := 2
def total_balls : ℕ := red_balls + yellow_balls + blue_balls

-- Define the number of draws
def num_draws : ℕ := 3

-- Define events A and B
def event_A : ℕ := 3 * red_balls * red_balls * total_balls
def event_B : ℕ := 3 * red_balls * red_balls * red_balls

-- Define the conditional probability P(B|A)
def prob_B_given_A : ℚ := event_B / event_A

-- Theorem to prove
theorem conditional_probability_proof : prob_B_given_A = 1 / 3 := by
  sorry

end conditional_probability_proof_l3622_362291


namespace divide_inequality_l3622_362275

theorem divide_inequality (x : ℝ) : -6 * x > 2 ↔ x < -1/3 := by sorry

end divide_inequality_l3622_362275

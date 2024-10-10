import Mathlib

namespace amanda_savings_l2750_275048

/-- The cost of a single lighter at the gas station in dollars -/
def gas_station_price : ℚ := 175 / 100

/-- The cost of a pack of 12 lighters online in dollars -/
def online_pack_price : ℚ := 5

/-- The number of lighters in each online pack -/
def lighters_per_pack : ℕ := 12

/-- The number of lighters Amanda wants to buy -/
def lighters_to_buy : ℕ := 24

/-- The savings Amanda would have by buying online instead of at the gas station -/
theorem amanda_savings : 
  (lighters_to_buy : ℚ) * gas_station_price - 
  (lighters_to_buy / lighters_per_pack : ℚ) * online_pack_price = 32 := by
  sorry

end amanda_savings_l2750_275048


namespace remainder_theorem_polynomial_remainder_l2750_275047

def f (x : ℝ) : ℝ := x^5 - 8*x^4 + 12*x^3 + 20*x^2 - 19*x - 24

theorem remainder_theorem (f : ℝ → ℝ) (a : ℝ) :
  ∃ q : ℝ → ℝ, ∀ x, f x = (x - a) * q x + f a := sorry

theorem polynomial_remainder :
  ∃ q : ℝ → ℝ, ∀ x, f x = (x - 5) * q x + 1012 := by
  sorry

end remainder_theorem_polynomial_remainder_l2750_275047


namespace difference_of_squares_l2750_275029

theorem difference_of_squares : 435^2 - 365^2 = 56000 := by
  sorry

end difference_of_squares_l2750_275029


namespace trapezoid_intersection_distances_l2750_275099

/-- Given a trapezoid ABCD with legs AB and CD, and bases AD and BC where AD > BC,
    this theorem proves the distances from the intersection point M of the extended legs
    to the vertices of the trapezoid. -/
theorem trapezoid_intersection_distances
  (AB CD AD BC : ℝ) -- Lengths of sides
  (h_AD_gt_BC : AD > BC) -- Condition: AD > BC
  : ∃ (BM AM CM DM : ℝ),
    BM = (AB * BC) / (AD - BC) ∧
    AM = (AB * AD) / (AD - BC) ∧
    CM = (CD * BC) / (AD - BC) ∧
    DM = (CD * AD) / (AD - BC) := by
  sorry

end trapezoid_intersection_distances_l2750_275099


namespace sons_age_next_year_l2750_275080

/-- Given a father who is 35 years old and whose age is five times that of his son,
    prove that the son's age next year will be 8 years. -/
theorem sons_age_next_year (father_age : ℕ) (son_age : ℕ) : 
  father_age = 35 → father_age = 5 * son_age → son_age + 1 = 8 := by
  sorry

end sons_age_next_year_l2750_275080


namespace point_on_line_implies_a_equals_negative_eight_l2750_275046

/-- A point (a, 0) lies on the line y = x + 8 -/
def point_on_line (a : ℝ) : Prop :=
  0 = a + 8

/-- Theorem: If (a, 0) lies on the line y = x + 8, then a = -8 -/
theorem point_on_line_implies_a_equals_negative_eight (a : ℝ) :
  point_on_line a → a = -8 := by
  sorry

end point_on_line_implies_a_equals_negative_eight_l2750_275046


namespace container_capacity_l2750_275002

theorem container_capacity (container_volume : ℝ) (num_containers : ℕ) : 
  (8 : ℝ) = 0.2 * container_volume → 
  num_containers = 40 → 
  num_containers * container_volume = 1600 := by
sorry

end container_capacity_l2750_275002


namespace thomson_savings_l2750_275035

def incentive : ℚ := 240

def food_fraction : ℚ := 1/3
def clothes_fraction : ℚ := 1/5
def savings_fraction : ℚ := 3/4

def food_expense : ℚ := food_fraction * incentive
def clothes_expense : ℚ := clothes_fraction * incentive
def total_expense : ℚ := food_expense + clothes_expense
def remaining : ℚ := incentive - total_expense
def savings : ℚ := savings_fraction * remaining

theorem thomson_savings : savings = 84 := by
  sorry

end thomson_savings_l2750_275035


namespace derivative_f_at_zero_l2750_275039

/-- The factorial function -/
def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

/-- The function f(x) = x(x+1)(x+2)...(x+n) -/
def f (n : ℕ) (x : ℝ) : ℝ := (List.range (n + 1)).foldl (fun acc i => acc * (x + i)) x

/-- Theorem: The derivative of f(x) at x = 0 is equal to n! -/
theorem derivative_f_at_zero (n : ℕ) : 
  deriv (f n) 0 = factorial n := by sorry

end derivative_f_at_zero_l2750_275039


namespace complex_expression_equality_l2750_275041

theorem complex_expression_equality : 
  let M := (Real.sqrt (Real.sqrt 7 + 3) + Real.sqrt (Real.sqrt 7 - 3)) / Real.sqrt (Real.sqrt 7 + 2) - Real.sqrt (5 - 2 * Real.sqrt 6)
  M = (1 + Real.sqrt 3 + Real.sqrt 5 + 3 * Real.sqrt 2) / 3 := by
  sorry

end complex_expression_equality_l2750_275041


namespace tan_theta_in_terms_of_x_l2750_275073

theorem tan_theta_in_terms_of_x (θ : Real) (x : Real) 
  (h_acute : 0 < θ ∧ θ < π / 2)
  (h_x : x > 1)
  (h_sin : Real.sin (θ / 2) = Real.sqrt ((x + 1) / (2 * x))) :
  Real.tan θ = -Real.sqrt (x^2 - 1) := by
  sorry

end tan_theta_in_terms_of_x_l2750_275073


namespace bread_in_pond_l2750_275005

theorem bread_in_pond (total_bread : ℕ) (duck1_bread : ℕ) (duck2_bread : ℕ) (duck3_bread : ℕ) 
  (h1 : total_bread = 100)
  (h2 : duck1_bread = total_bread / 2)
  (h3 : duck2_bread = 13)
  (h4 : duck3_bread = 7) :
  total_bread - (duck1_bread + duck2_bread + duck3_bread) = 30 := by
  sorry

end bread_in_pond_l2750_275005


namespace solution_set_of_equation_l2750_275089

theorem solution_set_of_equation : 
  ∃ (S : Set ℂ), S = {6, 2, 4 + 2*I, 4 - 2*I} ∧ 
  ∀ x : ℂ, (x - 2)^4 + (x - 6)^4 = 272 ↔ x ∈ S :=
sorry

end solution_set_of_equation_l2750_275089


namespace fraction_ratio_l2750_275070

theorem fraction_ratio (N : ℝ) (h1 : (1/3) * (2/5) * N = 14) (h2 : 0.4 * N = 168) :
  14 / ((1/3) * (2/5) * N) = 1 := by
  sorry

end fraction_ratio_l2750_275070


namespace simplify_expression_l2750_275025

theorem simplify_expression (x : ℝ) : 
  2*x - 3*(2-x) + (1/2)*(3-2*x) - 5*(2+3*x) = -11*x - 15.5 := by
sorry

end simplify_expression_l2750_275025


namespace apples_in_baskets_l2750_275077

theorem apples_in_baskets (total_apples : ℕ) (num_baskets : ℕ) (removed_apples : ℕ) 
  (h1 : total_apples = 64)
  (h2 : num_baskets = 4)
  (h3 : removed_apples = 3)
  : (total_apples / num_baskets) - removed_apples = 13 := by
  sorry

end apples_in_baskets_l2750_275077


namespace largest_non_prime_sequence_l2750_275090

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem largest_non_prime_sequence :
  ∃ (start : ℕ),
    (∀ i ∈ Finset.range 7, 
      let n := start + i
      10 ≤ n ∧ n < 40 ∧ ¬(is_prime n)) ∧
    (∀ j ≥ start + 7, 
      ¬(∀ i ∈ Finset.range 7, 
        let n := j + i
        10 ≤ n ∧ n < 40 ∧ ¬(is_prime n))) →
  start + 6 = 32 :=
sorry

end largest_non_prime_sequence_l2750_275090


namespace sufficient_but_not_necessary_l2750_275069

theorem sufficient_but_not_necessary
  (a : ℝ) (ha_pos : a > 0) (ha_neq_one : a ≠ 1)
  (f : ℝ → ℝ) (hf : f = λ x => a^x)
  (g : ℝ → ℝ) (hg : g = λ x => (2-a)*x^3) :
  (∀ x y : ℝ, x < y → f x > f y) →
  (∀ x y : ℝ, x < y → g x < g y) ∧
  ¬(∀ x y : ℝ, x < y → g x < g y → ∀ x y : ℝ, x < y → f x > f y) :=
by sorry

end sufficient_but_not_necessary_l2750_275069


namespace tangent_parallel_points_l2750_275036

def f (x : ℝ) : ℝ := x^3 + x - 2

theorem tangent_parallel_points :
  ∀ x y : ℝ, (f x = y ∧ (3 * x^2 + 1 = 4)) ↔ ((x = 1 ∧ y = 0) ∨ (x = -1 ∧ y = -4)) := by
  sorry

end tangent_parallel_points_l2750_275036


namespace partial_fraction_sum_zero_l2750_275082

theorem partial_fraction_sum_zero (x : ℝ) (A B C D E F : ℝ) :
  (1 : ℝ) / (x * (x + 1) * (x + 2) * (x + 3) * (x + 4) * (x + 5)) =
  A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 4) + F / (x + 5) →
  A + B + C + D + E + F = 0 := by
sorry

end partial_fraction_sum_zero_l2750_275082


namespace sum_of_ages_in_five_years_l2750_275060

/-- Given that Linda's current age is 13 and she is 3 more than 2 times Jane's age,
    prove that the sum of their ages in five years will be 28. -/
theorem sum_of_ages_in_five_years (jane_age : ℕ) (linda_age : ℕ) : 
  linda_age = 13 → linda_age = 2 * jane_age + 3 → 
  linda_age + 5 + (jane_age + 5) = 28 := by
sorry

end sum_of_ages_in_five_years_l2750_275060


namespace infinitely_many_pairs_exist_l2750_275027

/-- Definition of triangular numbers -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Theorem stating the existence of infinitely many pairs (a, b) satisfying the property -/
theorem infinitely_many_pairs_exist :
  ∃ f : ℕ → ℕ × ℕ, ∀ k : ℕ,
    let (a, b) := f k
    ∀ n : ℕ, (∃ m : ℕ, a * triangular_number n + b = triangular_number m) ↔
              (∃ l : ℕ, triangular_number n = triangular_number l) :=
by sorry

end infinitely_many_pairs_exist_l2750_275027


namespace johnson_family_seating_l2750_275028

/-- The number of ways to arrange 5 boys and 4 girls in a row of 9 chairs such that at least 2 boys are next to each other -/
def seating_arrangements (num_boys : ℕ) (num_girls : ℕ) : ℕ :=
  Nat.factorial (num_boys + num_girls) - 2 * (Nat.factorial num_boys * Nat.factorial num_girls)

/-- Theorem stating that the number of seating arrangements for 5 boys and 4 girls with at least 2 boys next to each other is 357120 -/
theorem johnson_family_seating :
  seating_arrangements 5 4 = 357120 := by
  sorry

end johnson_family_seating_l2750_275028


namespace at_most_one_perfect_square_l2750_275079

theorem at_most_one_perfect_square (a : ℕ → ℤ) 
  (h : ∀ n, a (n + 1) = a n ^ 3 + 1999) : 
  ∃! k, ∃ m : ℤ, a k = m ^ 2 :=
sorry

end at_most_one_perfect_square_l2750_275079


namespace factorization_x_squared_minus_nine_l2750_275038

theorem factorization_x_squared_minus_nine (x : ℝ) : x^2 - 9 = (x - 3) * (x + 3) := by
  sorry

end factorization_x_squared_minus_nine_l2750_275038


namespace parabola_unique_intersection_l2750_275009

/-- A parabola defined by x = -4y^2 - 6y + 10 -/
def parabola (y : ℝ) : ℝ := -4 * y^2 - 6 * y + 10

/-- The condition for a vertical line x = m to intersect the parabola at exactly one point -/
def unique_intersection (m : ℝ) : Prop :=
  ∃! y, parabola y = m

theorem parabola_unique_intersection :
  ∀ m : ℝ, unique_intersection m → m = 49 / 4 := by sorry

end parabola_unique_intersection_l2750_275009


namespace product_xyz_l2750_275067

theorem product_xyz (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x * (y + z) = 198)
  (eq2 : y * (z + x) = 216)
  (eq3 : z * (x + y) = 234) :
  x * y * z = 1080 := by
sorry

end product_xyz_l2750_275067


namespace honey_tax_calculation_l2750_275076

/-- Represents the tax per pound of honey -/
def tax_per_pound : ℝ := 1

theorem honey_tax_calculation 
  (bulk_price : ℝ) 
  (minimum_spend : ℝ) 
  (total_paid : ℝ) 
  (excess_pounds : ℝ) 
  (h1 : bulk_price = 5)
  (h2 : minimum_spend = 40)
  (h3 : total_paid = 240)
  (h4 : excess_pounds = 32)
  : tax_per_pound = 1 := by
  sorry

#check honey_tax_calculation

end honey_tax_calculation_l2750_275076


namespace trigonometric_identity_l2750_275063

theorem trigonometric_identity (x y : ℝ) : 
  Real.sin (x + y) * Real.sin x + Real.cos (x + y) * Real.cos x = Real.cos y := by
  sorry

end trigonometric_identity_l2750_275063


namespace senior_mean_score_l2750_275081

theorem senior_mean_score (total_students : ℕ) (overall_mean : ℝ) 
  (senior_total_score : ℝ) :
  total_students = 200 →
  overall_mean = 80 →
  senior_total_score = 7200 →
  ∃ (num_seniors num_non_seniors : ℕ) (senior_mean non_senior_mean : ℝ),
    num_non_seniors = (5 / 4 : ℝ) * num_seniors ∧
    senior_mean = (6 / 5 : ℝ) * non_senior_mean ∧
    num_seniors + num_non_seniors = total_students ∧
    (num_seniors * senior_mean + num_non_seniors * non_senior_mean) / total_students = overall_mean ∧
    num_seniors * senior_mean = senior_total_score ∧
    senior_mean = 80.9 := by
  sorry

end senior_mean_score_l2750_275081


namespace store_goods_values_l2750_275026

/-- Given a store with two grades of goods, prove the initial values of the goods. -/
theorem store_goods_values (x y : ℝ) (a b : ℝ) (h1 : x + y = 450)
  (h2 : y / b * (a + b) = 400) (h3 : x / a * (a + b) = 480) :
  x = 300 ∧ y = 150 := by
  sorry


end store_goods_values_l2750_275026


namespace frog_arrangement_count_l2750_275004

def frog_arrangements (n : ℕ) (g r : ℕ) (b : ℕ) : Prop :=
  n = g + r + b ∧
  g = 3 ∧
  r = 3 ∧
  b = 1

theorem frog_arrangement_count :
  ∀ (n g r b : ℕ),
    frog_arrangements n g r b →
    (n - 1) * 2 * (g.factorial * r.factorial) = 504 :=
by sorry

end frog_arrangement_count_l2750_275004


namespace ellipse_max_value_l2750_275066

theorem ellipse_max_value (x y : ℝ) : 
  x^2 + 4*y^2 = 4 → 
  ∃ (M : ℝ), M = 7 ∧ ∀ (a b : ℝ), a^2 + 4*b^2 = 4 → (3/4)*a^2 + 2*a - b^2 ≤ M :=
by sorry

end ellipse_max_value_l2750_275066


namespace quadratic_inequality_l2750_275031

theorem quadratic_inequality (x : ℝ) : -3 * x^2 + 5 * x + 4 < 0 ↔ -4/3 < x ∧ x < 1 := by
  sorry

end quadratic_inequality_l2750_275031


namespace jay_change_calculation_l2750_275042

/-- Calculates the change Jay received after purchasing items with a discount --/
theorem jay_change_calculation (book pen ruler notebook pencil_case : ℚ)
  (h_book : book = 25)
  (h_pen : pen = 4)
  (h_ruler : ruler = 1)
  (h_notebook : notebook = 8)
  (h_pencil_case : pencil_case = 6)
  (discount_rate : ℚ)
  (h_discount : discount_rate = 0.1)
  (paid_amount : ℚ)
  (h_paid : paid_amount = 100) :
  let total_before_discount := book + pen + ruler + notebook + pencil_case
  let discount_amount := discount_rate * total_before_discount
  let total_after_discount := total_before_discount - discount_amount
  paid_amount - total_after_discount = 60.4 := by
sorry

end jay_change_calculation_l2750_275042


namespace faces_after_fifth_step_l2750_275059

/-- Represents the number of vertices at step n -/
def V : ℕ → ℕ
| 0 => 8
| n + 1 => 3 * V n

/-- Represents the number of faces at step n -/
def F : ℕ → ℕ
| 0 => 6
| n + 1 => F n + V n

/-- Theorem stating that the number of faces after the fifth step is 974 -/
theorem faces_after_fifth_step : F 5 = 974 := by
  sorry

end faces_after_fifth_step_l2750_275059


namespace decimal_89_equals_base5_324_l2750_275054

/-- Converts a natural number to its base-5 representation -/
def toBase5 (n : ℕ) : List ℕ :=
  if n < 5 then [n]
  else (n % 5) :: toBase5 (n / 5)

theorem decimal_89_equals_base5_324 : toBase5 89 = [4, 2, 3] := by
  sorry

end decimal_89_equals_base5_324_l2750_275054


namespace open_box_volume_calculation_l2750_275068

/-- Given a rectangular sheet and squares cut from corners, calculates the volume of the resulting open box. -/
def openBoxVolume (sheetLength sheetWidth squareSide : ℝ) : ℝ :=
  (sheetLength - 2 * squareSide) * (sheetWidth - 2 * squareSide) * squareSide

/-- Theorem: The volume of the open box formed from a 48m x 36m sheet with 5m squares cut from corners is 9880 m³. -/
theorem open_box_volume_calculation :
  openBoxVolume 48 36 5 = 9880 := by
  sorry

#eval openBoxVolume 48 36 5

end open_box_volume_calculation_l2750_275068


namespace max_students_distribution_l2750_275032

theorem max_students_distribution (num_pens num_pencils : ℕ) :
  let max_students := Nat.gcd num_pens num_pencils
  ∃ (pens_per_student pencils_per_student : ℕ),
    num_pens = max_students * pens_per_student ∧
    num_pencils = max_students * pencils_per_student ∧
    ∀ (n : ℕ),
      (∃ (p q : ℕ), num_pens = n * p ∧ num_pencils = n * q) →
      n ≤ max_students :=
by sorry

end max_students_distribution_l2750_275032


namespace no_common_solution_l2750_275071

theorem no_common_solution : ¬∃ (x y : ℝ), x^2 + y^2 = 25 ∧ x^2 + 3*y = 45 := by
  sorry

end no_common_solution_l2750_275071


namespace power_of_two_equality_l2750_275015

theorem power_of_two_equality (m : ℤ) : 
  2^1999 - 2^1998 - 2^1997 + 2^1996 - 2^1995 = m * 2^1995 → m = 5 := by
  sorry

end power_of_two_equality_l2750_275015


namespace perpendicular_transitivity_l2750_275017

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines and planes
variable (perp : Line → Plane → Prop)

-- State the theorem
theorem perpendicular_transitivity
  (m n : Line) (α β : Plane)
  (hm : m ≠ n) (hαβ : α ≠ β)
  (hmβ : perp m β) (hnβ : perp n β) (hnα : perp n α) :
  perp m α :=
sorry

end perpendicular_transitivity_l2750_275017


namespace lactate_bicarbonate_reaction_in_extracellular_fluid_l2750_275098

-- Define the extracellular fluid
structure ExtracellularFluid where
  is_liquid_environment : Bool

-- Define a biochemical reaction
structure BiochemicalReaction where
  occurs_in_extracellular_fluid : Bool

-- Define the specific reaction
def lactate_bicarbonate_reaction : BiochemicalReaction where
  occurs_in_extracellular_fluid := true

-- Theorem statement
theorem lactate_bicarbonate_reaction_in_extracellular_fluid 
  (ecf : ExtracellularFluid) 
  (h : ecf.is_liquid_environment = true) : 
  lactate_bicarbonate_reaction.occurs_in_extracellular_fluid = true := by
  sorry

end lactate_bicarbonate_reaction_in_extracellular_fluid_l2750_275098


namespace arithmetic_sequence_sum_l2750_275074

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  (∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m) →  -- arithmetic sequence condition
  a 5 + a 8 = 5 →                                   -- given condition
  a 2 + a 11 = 5 :=                                 -- conclusion to prove
by
  sorry

end arithmetic_sequence_sum_l2750_275074


namespace trout_weight_fishing_scenario_l2750_275034

/-- Calculates the weight of trout caught given the fishing conditions -/
theorem trout_weight (num_campers : ℕ) (fish_per_camper : ℕ) 
                     (num_bass : ℕ) (bass_weight : ℕ) 
                     (num_salmon : ℕ) (salmon_weight : ℕ) : ℕ :=
  let total_fish_needed := num_campers * fish_per_camper
  let total_bass_weight := num_bass * bass_weight
  let total_salmon_weight := num_salmon * salmon_weight
  total_fish_needed - (total_bass_weight + total_salmon_weight)

/-- The specific fishing scenario described in the problem -/
theorem fishing_scenario : trout_weight 22 2 6 2 2 12 = 8 := by
  sorry

end trout_weight_fishing_scenario_l2750_275034


namespace rectangle_problem_l2750_275058

/-- Given three rectangles with equal areas and integer sides, where one side is 31,
    the length of a side perpendicular to the side of length 31 is 992. -/
theorem rectangle_problem (a b : ℕ) (h1 : a > 0) (h2 : b > 0) : 
  (a * 31 = b * (992 : ℕ)) ∧ (∃ k l : ℕ, k * l = 31 * (k + l) ∧ k = 992) := by
  sorry

end rectangle_problem_l2750_275058


namespace seven_eighths_of_48_l2750_275011

theorem seven_eighths_of_48 : (7 / 8 : ℚ) * 48 = 42 := by
  sorry

end seven_eighths_of_48_l2750_275011


namespace part_a_part_b_l2750_275085

def solution_set_a : Set (ℤ × ℤ) := {(6, -21), (-13, -2), (4, 15), (23, -4), (7, -12), (-4, -1), (3, 6), (14, -5), (8, -9), (-1, 0), (2, 3), (11, -6)}

def equation_set_a : Set (ℤ × ℤ) := {(x, y) | x * y + 3 * x - 5 * y = -3}

theorem part_a : equation_set_a = solution_set_a := by sorry

def solution_set_b : Set (ℤ × ℤ) := {(4, 2)}

def equation_set_b : Set (ℤ × ℤ) := {(x, y) | x - y = x / y}

theorem part_b : equation_set_b = solution_set_b := by sorry

end part_a_part_b_l2750_275085


namespace cube_sphere_volume_l2750_275016

theorem cube_sphere_volume (cube_surface_area : ℝ) (h_surface_area : cube_surface_area = 18) :
  let cube_edge := Real.sqrt (cube_surface_area / 6)
  let sphere_radius := (Real.sqrt 3 * cube_edge) / 2
  let sphere_volume := (4 / 3) * Real.pi * sphere_radius ^ 3
  sphere_volume = 9 * Real.pi / 2 := by
  sorry

end cube_sphere_volume_l2750_275016


namespace correct_probability_distribution_l2750_275050

/-- Represents the number of students -/
def num_students : ℕ := 4

/-- Represents the number of cookie types -/
def num_cookie_types : ℕ := 3

/-- Represents the total number of cookies -/
def total_cookies : ℕ := num_students * num_cookie_types

/-- Represents the number of cookies of each type -/
def cookies_per_type : ℕ := num_students

/-- Calculates the probability of each student receiving one cookie of each type -/
def probability_all_students_correct_distribution : ℚ :=
  144 / 3850

/-- Theorem stating that the calculated probability is correct -/
theorem correct_probability_distribution :
  probability_all_students_correct_distribution = 144 / 3850 := by
  sorry

end correct_probability_distribution_l2750_275050


namespace fixed_point_of_f_l2750_275010

-- Define the set of valid 'a' values
def ValidA : Set ℝ := { x | (0 < x ∧ x < 1) ∨ (1 < x) }

-- Define the logarithm function
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Define our function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log a x + 2

-- State the theorem
theorem fixed_point_of_f (a : ℝ) (h : a ∈ ValidA) : f a 1 = 2 := by
  sorry

end fixed_point_of_f_l2750_275010


namespace find_a_and_b_l2750_275053

-- Define the sets A and B
def A : Set ℝ := {x | x^3 + 3*x^2 + 2*x > 0}
def B (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b ≤ 0}

-- State the theorem
theorem find_a_and_b :
  ∃ (a b : ℝ),
    (A ∩ B a b = {x | 0 < x ∧ x ≤ 2}) ∧
    (A ∪ B a b = {x | x > -2}) ∧
    a = -1 ∧
    b = -2 :=
by sorry

end find_a_and_b_l2750_275053


namespace triangle_geometric_sequence_cosine_l2750_275037

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that if a, b, c form a geometric sequence and c = 2a, then cos B = 1/√2 -/
theorem triangle_geometric_sequence_cosine (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Ensure positive side lengths
  A > 0 ∧ B > 0 ∧ C > 0 →  -- Ensure positive angles
  A + B + C = π →  -- Sum of angles in a triangle
  c^2 = a^2 + b^2 - 2*a*b*Real.cos C →  -- Law of cosines
  (∃ r : ℝ, b = a*r ∧ c = b*r) →  -- Geometric sequence condition
  c = 2*a →  -- Given condition
  Real.cos B = 1 / Real.sqrt 2 := by
sorry

end triangle_geometric_sequence_cosine_l2750_275037


namespace infinite_series_sum_l2750_275044

/-- The sum of the infinite series ∑(n=1 to ∞) (3n - 2) / (n(n + 1)(n + 3)) is equal to 7/12. -/
theorem infinite_series_sum : 
  ∑' (n : ℕ), (3 * n - 2 : ℝ) / (n * (n + 1) * (n + 3)) = 7 / 12 := by
  sorry

end infinite_series_sum_l2750_275044


namespace square_sum_inequality_l2750_275033

theorem square_sum_inequality (a b x y : ℝ) : (a^2 + b^2) * (x^2 + y^2) ≥ (a*x + b*y)^2 := by
  sorry

end square_sum_inequality_l2750_275033


namespace student_count_l2750_275083

theorem student_count : 
  ∃ n₁ n₂ : ℕ, n₁ ≠ n₂ ∧ 
  (∀ n : ℕ, (70 < n ∧ n < 130 ∧ 
             n % 4 = 2 ∧ 
             n % 5 = 2 ∧ 
             n % 6 = 2) ↔ (n = n₁ ∨ n = n₂)) ∧
  n₁ = 92 ∧ n₂ = 122 :=
by sorry

end student_count_l2750_275083


namespace triangle_with_angle_ratio_1_2_3_is_right_triangle_l2750_275096

theorem triangle_with_angle_ratio_1_2_3_is_right_triangle (a b c : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  a + b + c = 180 →
  b = 2 * a →
  c = 3 * a →
  c = 90 :=
sorry

end triangle_with_angle_ratio_1_2_3_is_right_triangle_l2750_275096


namespace arithmetic_calculation_l2750_275022

theorem arithmetic_calculation : -16 - (-12) - 24 + 18 = -10 := by
  sorry

end arithmetic_calculation_l2750_275022


namespace square_area_increase_l2750_275092

theorem square_area_increase (s : ℝ) (h : s > 0) :
  let new_side := 1.15 * s
  let original_area := s^2
  let new_area := new_side^2
  (new_area - original_area) / original_area = 0.3225 := by
  sorry

end square_area_increase_l2750_275092


namespace quadratic_root_range_l2750_275012

theorem quadratic_root_range (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ < -1 ∧ x₂ > 1 ∧ 
   x₁^2 + (m-1)*x₁ + m^2 - 2 = 0 ∧ 
   x₂^2 + (m-1)*x₂ + m^2 - 2 = 0) → 
  0 < m ∧ m < 1 := by
sorry

end quadratic_root_range_l2750_275012


namespace star_equation_solution_l2750_275052

/-- Define the ⋆ operation -/
def star (a b : ℝ) : ℝ := 3 * a - 2 * b^2

/-- Theorem: If a ⋆ 4 = 17, then a = 49/3 -/
theorem star_equation_solution (a : ℝ) (h : star a 4 = 17) : a = 49/3 := by
  sorry

end star_equation_solution_l2750_275052


namespace correct_requirements_l2750_275019

/-- A cricket game scenario -/
structure CricketGame where
  totalOvers : ℕ
  firstInningOvers : ℕ
  runsScored : ℕ
  wicketsLost : ℕ
  runRate : ℚ
  targetScore : ℕ

/-- Calculate the required run rate and partnership score -/
def calculateRequirements (game : CricketGame) : ℚ × ℕ :=
  let remainingOvers := game.totalOvers - game.firstInningOvers
  let remainingRuns := game.targetScore - game.runsScored
  let requiredRunRate := remainingRuns / remainingOvers
  let requiredPartnership := remainingRuns
  (requiredRunRate, requiredPartnership)

/-- Theorem stating the correct calculation of requirements -/
theorem correct_requirements (game : CricketGame) 
    (h1 : game.totalOvers = 50)
    (h2 : game.firstInningOvers = 10)
    (h3 : game.runsScored = 32)
    (h4 : game.wicketsLost = 3)
    (h5 : game.runRate = 32/10)
    (h6 : game.targetScore = 282) :
    calculateRequirements game = (25/4, 250) := by
  sorry

#eval calculateRequirements {
  totalOvers := 50,
  firstInningOvers := 10,
  runsScored := 32,
  wicketsLost := 3,
  runRate := 32/10,
  targetScore := 282
}

end correct_requirements_l2750_275019


namespace vendor_throw_away_percent_l2750_275013

-- Define the initial number of apples (100 for simplicity)
def initial_apples : ℝ := 100

-- Define the percentage of apples sold on the first day
def first_day_sale_percent : ℝ := 30

-- Define the percentage of apples sold on the second day
def second_day_sale_percent : ℝ := 50

-- Define the total percentage of apples thrown away
def total_thrown_away_percent : ℝ := 42

-- Define the percentage of remaining apples thrown away on the first day
def first_day_throw_away_percent : ℝ := 20

theorem vendor_throw_away_percent :
  let remaining_after_first_sale := initial_apples * (1 - first_day_sale_percent / 100)
  let remaining_after_first_throw := remaining_after_first_sale * (1 - first_day_throw_away_percent / 100)
  let sold_second_day := remaining_after_first_throw * (second_day_sale_percent / 100)
  let thrown_away_second_day := remaining_after_first_throw - sold_second_day
  let total_thrown_away := (remaining_after_first_sale - remaining_after_first_throw) + thrown_away_second_day
  total_thrown_away = initial_apples * (total_thrown_away_percent / 100) :=
by sorry

end vendor_throw_away_percent_l2750_275013


namespace equation_has_root_in_interval_l2750_275095

theorem equation_has_root_in_interval (t : ℝ) (h : t ∈ ({6, 7, 8, 9} : Set ℝ)) :
  ∃ x : ℝ, x ∈ Set.Icc 1 2 ∧ x^4 - t*x + 1/t = 0 := by
  sorry

#check equation_has_root_in_interval

end equation_has_root_in_interval_l2750_275095


namespace ratio_constraint_l2750_275006

theorem ratio_constraint (a b : ℝ) (h1 : 0 ≤ a) (h2 : a < b) 
  (h3 : ∀ x : ℝ, a + b * Real.cos x + (b / (2 * Real.sqrt 2)) * Real.cos (2 * x) ≥ 0) :
  (b + a) / (b - a) = 3 + 2 * Real.sqrt 2 := by
  sorry

end ratio_constraint_l2750_275006


namespace xiao_ying_final_grade_l2750_275055

/-- Represents a student's physical education grade components and scores -/
structure PhysEdGrade where
  regular_activity_weight : Real
  theory_test_weight : Real
  skills_test_weight : Real
  regular_activity_score : Real
  theory_test_score : Real
  skills_test_score : Real

/-- Calculates the final physical education grade -/
def calculate_final_grade (grade : PhysEdGrade) : Real :=
  grade.regular_activity_weight * grade.regular_activity_score +
  grade.theory_test_weight * grade.theory_test_score +
  grade.skills_test_weight * grade.skills_test_score

/-- Xiao Ying's physical education grade components and scores -/
def xiao_ying_grade : PhysEdGrade :=
  { regular_activity_weight := 0.3
    theory_test_weight := 0.2
    skills_test_weight := 0.5
    regular_activity_score := 90
    theory_test_score := 80
    skills_test_score := 94 }

/-- Theorem: Xiao Ying's final physical education grade is 90 points -/
theorem xiao_ying_final_grade :
  calculate_final_grade xiao_ying_grade = 90 := by
  sorry

end xiao_ying_final_grade_l2750_275055


namespace smallest_even_five_digit_number_has_eight_in_tens_place_l2750_275086

-- Define a type for digits
inductive Digit : Type
  | one : Digit
  | three : Digit
  | five : Digit
  | six : Digit
  | eight : Digit

-- Define a function to convert Digit to Nat
def digitToNat : Digit → Nat
  | Digit.one => 1
  | Digit.three => 3
  | Digit.five => 5
  | Digit.six => 6
  | Digit.eight => 8

-- Define a function to check if a number is even
def isEven (n : Nat) : Bool :=
  n % 2 == 0

-- Define a function to construct a five-digit number from Digits
def makeNumber (a b c d e : Digit) : Nat :=
  10000 * (digitToNat a) + 1000 * (digitToNat b) + 100 * (digitToNat c) + 10 * (digitToNat d) + (digitToNat e)

-- Define the theorem
theorem smallest_even_five_digit_number_has_eight_in_tens_place :
  ∀ (a b c d e : Digit),
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
    c ≠ d ∧ c ≠ e ∧
    d ≠ e →
    isEven (makeNumber a b c d e) →
    (∀ (x y z w v : Digit),
      x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ x ≠ v ∧
      y ≠ z ∧ y ≠ w ∧ y ≠ v ∧
      z ≠ w ∧ z ≠ v ∧
      w ≠ v →
      isEven (makeNumber x y z w v) →
      makeNumber a b c d e ≤ makeNumber x y z w v) →
    d = Digit.eight :=
  sorry

end smallest_even_five_digit_number_has_eight_in_tens_place_l2750_275086


namespace chord_length_l2750_275072

/-- The length of the chord formed by the intersection of a line and a circle -/
theorem chord_length (l : Real → Real × Real) (C₁ : Real → Real × Real) : 
  (∀ t, l t = (1 + 3/5 * t, 4/5 * t)) →
  (∀ θ, C₁ θ = (Real.cos θ, Real.sin θ)) →
  (∃ A B, A ≠ B ∧ (∃ t₁ t₂, l t₁ = A ∧ l t₂ = B) ∧ (∃ θ₁ θ₂, C₁ θ₁ = A ∧ C₁ θ₂ = B)) →
  ∃ A B, A ≠ B ∧ (∃ t₁ t₂, l t₁ = A ∧ l t₂ = B) ∧ (∃ θ₁ θ₂, C₁ θ₁ = A ∧ C₁ θ₂ = B) ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 6/5 :=
by sorry

end chord_length_l2750_275072


namespace quadratic_inequality_no_solution_l2750_275030

theorem quadratic_inequality_no_solution :
  ¬∃ x : ℝ, x^2 - 2*x + 3 < 0 := by
  sorry

end quadratic_inequality_no_solution_l2750_275030


namespace incenter_distance_l2750_275064

/-- Represents a triangle with vertices P, Q, and R -/
structure Triangle (P Q R : ℝ × ℝ) : Prop where
  isosceles : dist P Q = dist P R
  pq_length : dist P Q = 17
  qr_length : dist Q R = 16

/-- Represents the incenter of a triangle -/
def incenter (t : Triangle P Q R) : ℝ × ℝ := sorry

/-- Represents the incircle of a triangle -/
def incircle (t : Triangle P Q R) : Set (ℝ × ℝ) := sorry

/-- Represents a point where the incircle touches a side of the triangle -/
def touchPoint (t : Triangle P Q R) (side : (ℝ × ℝ) × (ℝ × ℝ)) : ℝ × ℝ := sorry

theorem incenter_distance (P Q R : ℝ × ℝ) (t : Triangle P Q R) :
  let J := incenter t
  let C := touchPoint t (Q, R)
  dist C J = Real.sqrt 87.04 := by sorry

end incenter_distance_l2750_275064


namespace max_area_inscribed_quadrilateral_l2750_275065

/-- The maximum area of an inscribed quadrilateral within a circle -/
def max_area_circle (r : ℝ) : ℝ := 2 * r^2

/-- The equation of an ellipse -/
def is_ellipse (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

/-- The maximum area of an inscribed quadrilateral within an ellipse -/
def max_area_ellipse (a b : ℝ) : ℝ := 2 * a * b

theorem max_area_inscribed_quadrilateral 
  (r a b : ℝ) 
  (hr : r > 0) 
  (hab : a > b) 
  (hb : b > 0) : 
  max_area_ellipse a b = 2 * a * b :=
sorry

end max_area_inscribed_quadrilateral_l2750_275065


namespace johns_candy_cost_l2750_275094

/-- The amount John pays for candy bars after sharing the cost with Dave -/
def johnsPay (totalBars : ℕ) (daveBars : ℕ) (originalPrice : ℚ) (discountRate : ℚ) : ℚ :=
  let discountedPrice := originalPrice * (1 - discountRate)
  let totalCost := totalBars * discountedPrice
  let johnBars := totalBars - daveBars
  johnBars * discountedPrice

/-- Theorem stating that John pays $11.20 for his share of the candy bars -/
theorem johns_candy_cost :
  johnsPay 20 6 1 (20 / 100) = 11.2 := by
  sorry

end johns_candy_cost_l2750_275094


namespace fgh_supermarket_difference_l2750_275087

/-- The number of FGH supermarkets in the US and Canada -/
structure FGHSupermarkets where
  total : ℕ
  us : ℕ
  canada : ℕ

/-- The conditions for FGH supermarkets -/
def validFGHSupermarkets (s : FGHSupermarkets) : Prop :=
  s.total = 60 ∧
  s.us + s.canada = s.total ∧
  s.us = 37 ∧
  s.us > s.canada

/-- Theorem: The difference between FGH supermarkets in the US and Canada is 14 -/
theorem fgh_supermarket_difference (s : FGHSupermarkets) 
  (h : validFGHSupermarkets s) : s.us - s.canada = 14 := by
  sorry

end fgh_supermarket_difference_l2750_275087


namespace prime_cube_plus_two_l2750_275024

theorem prime_cube_plus_two (m : ℕ) : 
  Prime m → Prime (m^2 + 2) → m = 3 ∧ Prime (m^3 + 2) :=
by sorry

end prime_cube_plus_two_l2750_275024


namespace product_inequality_l2750_275088

theorem product_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (a + 1/a) * (b + 1/b) ≥ 25/4 := by sorry

end product_inequality_l2750_275088


namespace problem_I_problem_II_l2750_275023

theorem problem_I (α : Real) (h : α = π / 6) : 
  (2 * Real.sin (π + α) * Real.cos (π - α) - Real.cos (π + α)) / 
  (1 + Real.sin α ^ 2 + Real.sin (π - α) - Real.cos (π + α) ^ 2) = Real.sqrt 3 := by
  sorry

theorem problem_II (α : Real) (h : Real.tan α / (Real.tan α - 6) = -1) : 
  (2 * Real.cos α - 3 * Real.sin α) / (3 * Real.cos α + 4 * Real.sin α) = -7 / 15 := by
  sorry

end problem_I_problem_II_l2750_275023


namespace prob_odd_sum_coin_dice_prob_odd_sum_coin_dice_is_seven_sixteenths_l2750_275062

def coin_toss : Type := Bool
def die_roll : Type := Fin 6

def is_head (c : coin_toss) : Prop := c = true
def is_tail (c : coin_toss) : Prop := c = false

def sum_is_odd (rolls : List ℕ) : Prop := (rolls.sum % 2 = 1)

def prob_head : ℚ := 1/2
def prob_tail : ℚ := 1/2

def prob_odd_sum_two_dice : ℚ := 1/2

theorem prob_odd_sum_coin_dice : ℚ :=
  let p_0_head := prob_tail^3
  let p_1_head := 3 * prob_head * prob_tail^2
  let p_2_head := 3 * prob_head^2 * prob_tail
  let p_3_head := prob_head^3

  let p_odd_0_dice := 0
  let p_odd_2_dice := prob_odd_sum_two_dice
  let p_odd_4_dice := 1/2
  let p_odd_6_dice := 1/2

  p_0_head * p_odd_0_dice +
  p_1_head * p_odd_2_dice +
  p_2_head * p_odd_4_dice +
  p_3_head * p_odd_6_dice

theorem prob_odd_sum_coin_dice_is_seven_sixteenths :
  prob_odd_sum_coin_dice = 7/16 := by sorry

end prob_odd_sum_coin_dice_prob_odd_sum_coin_dice_is_seven_sixteenths_l2750_275062


namespace preimage_of_two_three_l2750_275049

/-- Given a mapping f : ℝ × ℝ → ℝ × ℝ defined by f(x, y) = (x+y, x-y),
    prove that f(5/2, -1/2) = (2, 3) -/
theorem preimage_of_two_three (f : ℝ × ℝ → ℝ × ℝ) 
    (h : ∀ x y : ℝ, f (x, y) = (x + y, x - y)) : 
    f (5/2, -1/2) = (2, 3) := by
  sorry

end preimage_of_two_three_l2750_275049


namespace kendra_toy_purchase_l2750_275001

/-- The price of a wooden toy -/
def toy_price : ℕ := 20

/-- The price of a hat -/
def hat_price : ℕ := 10

/-- The number of hats Kendra bought -/
def hats_bought : ℕ := 3

/-- The amount of money Kendra started with -/
def initial_money : ℕ := 100

/-- The amount of change Kendra received -/
def change_received : ℕ := 30

/-- The number of wooden toys Kendra bought -/
def toys_bought : ℕ := 2

theorem kendra_toy_purchase :
  toy_price * toys_bought + hat_price * hats_bought = initial_money - change_received :=
by sorry

end kendra_toy_purchase_l2750_275001


namespace count_satisfying_pairs_l2750_275056

def satisfies_inequalities (a b : ℤ) : Prop :=
  (a^2 + b^2 < 25) ∧ (a^2 + b^2 < 10*a) ∧ (a^2 + b^2 < 10*b)

theorem count_satisfying_pairs :
  ∃! (s : Finset (ℤ × ℤ)), 
    (∀ (p : ℤ × ℤ), p ∈ s ↔ satisfies_inequalities p.1 p.2) ∧
    s.card = 8 := by
  sorry

end count_satisfying_pairs_l2750_275056


namespace sum_of_divisors_450_prime_factors_and_gcd_l2750_275045

def sumOfDivisors (n : ℕ) : ℕ := sorry

def numberOfDistinctPrimeFactors (n : ℕ) : ℕ := sorry

theorem sum_of_divisors_450_prime_factors_and_gcd :
  let s := sumOfDivisors 450
  numberOfDistinctPrimeFactors s = 3 ∧ Nat.gcd s 450 = 3 := by sorry

end sum_of_divisors_450_prime_factors_and_gcd_l2750_275045


namespace rat_value_l2750_275020

/-- Represents the value of a letter based on its position in the alphabet -/
def letter_value (c : Char) : ℕ :=
  match c with
  | 'a' => 1
  | 'b' => 2
  | 'c' => 3
  | 'd' => 4
  | 'e' => 5
  | 'f' => 6
  | 'g' => 7
  | 'h' => 8
  | 'i' => 9
  | 'j' => 10
  | 'k' => 11
  | 'l' => 12
  | 'm' => 13
  | 'n' => 14
  | 'o' => 15
  | 'p' => 16
  | 'q' => 17
  | 'r' => 18
  | 's' => 19
  | 't' => 20
  | 'u' => 21
  | 'v' => 22
  | 'w' => 23
  | 'x' => 24
  | 'y' => 25
  | 'z' => 26
  | _ => 0

/-- Calculates the number value of a word -/
def word_value (w : String) : ℕ :=
  (w.toList.map letter_value).sum * w.length

/-- Theorem: The number value of the word "rat" is 117 -/
theorem rat_value : word_value "rat" = 117 := by
  sorry

end rat_value_l2750_275020


namespace claire_crafting_time_l2750_275093

/-- Represents Claire's daily schedule --/
structure ClairesSchedule where
  clean : ℝ
  cook : ℝ
  errands : ℝ
  craft : ℝ
  tailor : ℝ

/-- Conditions for Claire's schedule --/
def validSchedule (s : ClairesSchedule) : Prop :=
  s.clean = 2 * s.cook ∧
  s.errands = s.cook - 1 ∧
  s.craft = s.tailor ∧
  s.clean + s.cook + s.errands + s.craft + s.tailor = 16 ∧
  s.craft + s.tailor = 9

/-- Theorem stating that in a valid schedule, Claire spends 4.5 hours crafting --/
theorem claire_crafting_time (s : ClairesSchedule) (h : validSchedule s) : s.craft = 4.5 := by
  sorry


end claire_crafting_time_l2750_275093


namespace train_passing_time_l2750_275051

/-- The time taken for a train to pass a stationary point -/
theorem train_passing_time (length : ℝ) (speed_kmh : ℝ) : 
  length = 280 → speed_kmh = 36 → 
  (length / (speed_kmh * 1000 / 3600)) = 28 := by
  sorry

end train_passing_time_l2750_275051


namespace complex_fraction_real_l2750_275018

theorem complex_fraction_real (a : ℝ) : 
  (((a : ℂ) + Complex.I) / (1 + Complex.I)).im = 0 → a = 1 := by
  sorry

end complex_fraction_real_l2750_275018


namespace scores_relative_to_average_l2750_275061

def scores : List ℤ := [95, 86, 90, 87, 92]
def average : ℚ := 90

theorem scores_relative_to_average :
  let relative_scores := scores.map (λ s => s - average)
  relative_scores = [5, -4, 0, -3, 2] := by
  sorry

end scores_relative_to_average_l2750_275061


namespace quadratic_equation_roots_l2750_275091

theorem quadratic_equation_roots (c : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, x^2 + 2*x + c = 0 ↔ x = x₁ ∨ x = x₂) →
  x₁^2 + x₂^2 = c^2 - 2*c →
  c = -2 ∧ x₁ = -1 + Real.sqrt 3 ∧ x₂ = -1 - Real.sqrt 3 :=
by sorry

end quadratic_equation_roots_l2750_275091


namespace mountain_valley_trail_length_l2750_275057

/-- Represents the length of the Mountain Valley Trail hike --/
def MountainValleyTrail : Type := { trail : ℕ // trail > 0 }

/-- Represents the daily hike distances --/
def DailyHikes : Type := Fin 5 → ℕ

theorem mountain_valley_trail_length 
  (hikes : DailyHikes) 
  (day1_2 : hikes 0 + hikes 1 = 30)
  (day2_4_avg : (hikes 1 + hikes 3) / 2 = 15)
  (day3_4_5 : hikes 2 + hikes 3 + hikes 4 = 45)
  (day1_3 : hikes 0 + hikes 2 = 33) :
  ∃ (trail : MountainValleyTrail), (hikes 0 + hikes 1 + hikes 2 + hikes 3 + hikes 4 : ℕ) = trail.val ∧ trail.val = 75 := by
  sorry

end mountain_valley_trail_length_l2750_275057


namespace student_count_l2750_275000

theorem student_count (total_erasers total_pencils leftover_erasers leftover_pencils : ℕ)
  (h1 : total_erasers = 49)
  (h2 : total_pencils = 66)
  (h3 : leftover_erasers = 4)
  (h4 : leftover_pencils = 6) :
  ∃ (students : ℕ),
    students > 0 ∧
    (total_erasers - leftover_erasers) % students = 0 ∧
    (total_pencils - leftover_pencils) % students = 0 ∧
    students = 15 := by
  sorry

end student_count_l2750_275000


namespace naval_formation_arrangements_l2750_275075

/-- The number of ways to arrange 2 submarines one in front of the other -/
def submarine_arrangements : ℕ := 2

/-- The number of ways to arrange 6 ships in two groups of 3 -/
def ship_arrangements : ℕ := 720

/-- The number of invalid arrangements where all ships on one side are of the same type -/
def invalid_arrangements : ℕ := 2 * 2

/-- The total number of valid arrangements -/
def total_arrangements : ℕ := submarine_arrangements * (ship_arrangements - invalid_arrangements)

theorem naval_formation_arrangements : total_arrangements = 1296 := by
  sorry

end naval_formation_arrangements_l2750_275075


namespace fishing_competition_duration_l2750_275003

theorem fishing_competition_duration 
  (jackson_daily : ℕ) 
  (jonah_daily : ℕ) 
  (george_daily : ℕ) 
  (total_catch : ℕ) 
  (h1 : jackson_daily = 6)
  (h2 : jonah_daily = 4)
  (h3 : george_daily = 8)
  (h4 : total_catch = 90) :
  ∃ (days : ℕ), days * (jackson_daily + jonah_daily + george_daily) = total_catch ∧ days = 5 := by
  sorry

end fishing_competition_duration_l2750_275003


namespace triangle_angle_calculation_l2750_275007

theorem triangle_angle_calculation (a b c : ℝ) (A B C : ℝ) :
  a = 4 * Real.sqrt 3 →
  c = 12 →
  C = π / 3 →
  0 < a ∧ 0 < b ∧ 0 < c →
  A + B + C = π →
  a / Real.sin A = b / Real.sin B →
  b / Real.sin B = c / Real.sin C →
  A = π / 6 := by
sorry

end triangle_angle_calculation_l2750_275007


namespace correct_polynomial_result_l2750_275040

/-- Given a polynomial P, prove that if subtracting P from a^2 - 5a + 7 results in 2a^2 - 3a + 5,
    then adding P to 2a^2 - 3a + 5 yields 5a^2 - 11a + 17. -/
theorem correct_polynomial_result (P : Polynomial ℚ) : 
  (a^2 - 5*a + 7 : Polynomial ℚ) - P = 2*a^2 - 3*a + 5 →
  P + (2*a^2 - 3*a + 5 : Polynomial ℚ) = 5*a^2 - 11*a + 17 := by
  sorry

end correct_polynomial_result_l2750_275040


namespace exists_field_trip_with_frequent_participants_l2750_275097

/-- Represents a field trip -/
structure FieldTrip where
  participants : Finset (Fin 20)
  at_least_four : participants.card ≥ 4

/-- Represents the collection of all field trips -/
structure FieldTrips where
  trips : Finset FieldTrip
  nonempty : trips.Nonempty

theorem exists_field_trip_with_frequent_participants (ft : FieldTrips) :
  ∃ (trip : FieldTrip), trip ∈ ft.trips ∧
    ∀ (student : Fin 20), student ∈ trip.participants →
      (ft.trips.filter (λ t : FieldTrip => student ∈ t.participants)).card ≥ ft.trips.card / 17 :=
sorry

end exists_field_trip_with_frequent_participants_l2750_275097


namespace justin_reading_theorem_l2750_275084

/-- Calculates the total number of pages Justin reads in a week -/
def totalPagesRead (firstDayPages : ℕ) (remainingDays : ℕ) : ℕ :=
  firstDayPages + remainingDays * (2 * firstDayPages)

/-- Proves that Justin reads 130 pages in a week -/
theorem justin_reading_theorem :
  totalPagesRead 10 6 = 130 :=
by sorry

end justin_reading_theorem_l2750_275084


namespace johns_height_l2750_275014

/-- Given the heights of various people and their relationships, prove John's height. -/
theorem johns_height (carl becky amy helen angela tom mary john : ℝ) 
  (h1 : carl = 120)
  (h2 : becky = 2 * carl)
  (h3 : amy = 1.2 * becky)
  (h4 : helen = amy + 3)
  (h5 : angela = helen + 4)
  (h6 : tom = angela - 70)
  (h7 : mary = 2 * tom)
  (h8 : john = 1.5 * mary) : 
  john = 675 := by sorry

end johns_height_l2750_275014


namespace back_lot_filled_fraction_l2750_275078

/-- Proves that the fraction of the back parking lot filled is 1/2 -/
theorem back_lot_filled_fraction :
  let front_spaces : ℕ := 52
  let back_spaces : ℕ := 38
  let total_spaces : ℕ := front_spaces + back_spaces
  let parked_cars : ℕ := 39
  let available_spaces : ℕ := 32
  let filled_back_spaces : ℕ := total_spaces - parked_cars - available_spaces
  (filled_back_spaces : ℚ) / back_spaces = 1 / 2 := by sorry

end back_lot_filled_fraction_l2750_275078


namespace combined_wave_amplitude_l2750_275021

noncomputable def y₁ (t : ℝ) : ℝ := 3 * Real.sqrt 2 * Real.sin (100 * Real.pi * t)
noncomputable def y₂ (t : ℝ) : ℝ := 3 * Real.sin (100 * Real.pi * t - Real.pi / 4)
noncomputable def y (t : ℝ) : ℝ := y₁ t + y₂ t

theorem combined_wave_amplitude :
  ∃ (A : ℝ) (φ : ℝ), ∀ t, y t = A * Real.sin (100 * Real.pi * t + φ) ∧ A = 3 * Real.sqrt 5 :=
sorry

end combined_wave_amplitude_l2750_275021


namespace sum_of_squares_zero_iff_all_zero_l2750_275043

theorem sum_of_squares_zero_iff_all_zero (a b c : ℝ) :
  a^2 + b^2 + c^2 = 0 ↔ a = 0 ∧ b = 0 ∧ c = 0 := by
  sorry

end sum_of_squares_zero_iff_all_zero_l2750_275043


namespace article_word_limit_l2750_275008

/-- Calculates the word limit for an article given specific font and page constraints. -/
theorem article_word_limit 
  (total_pages : ℕ) 
  (large_font_pages : ℕ) 
  (large_font_words_per_page : ℕ) 
  (small_font_words_per_page : ℕ) 
  (h1 : total_pages = 21)
  (h2 : large_font_pages = 4)
  (h3 : large_font_words_per_page = 1800)
  (h4 : small_font_words_per_page = 2400) :
  large_font_pages * large_font_words_per_page + 
  (total_pages - large_font_pages) * small_font_words_per_page = 48000 :=
by sorry

end article_word_limit_l2750_275008

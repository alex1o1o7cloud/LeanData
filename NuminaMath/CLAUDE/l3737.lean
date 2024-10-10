import Mathlib

namespace subsets_containing_neither_A_nor_B_l3737_373770

variable (X : Finset ℕ)
variable (A B : Finset ℕ)

theorem subsets_containing_neither_A_nor_B :
  X.card = 10 →
  A ⊆ X →
  B ⊆ X →
  A.card = 3 →
  B.card = 4 →
  Disjoint A B →
  (X.powerset.filter (λ S => ¬(A ⊆ S) ∧ ¬(B ⊆ S))).card = 840 :=
by sorry

end subsets_containing_neither_A_nor_B_l3737_373770


namespace banana_pies_count_l3737_373757

def total_pies : ℕ := 30
def ratio_sum : ℕ := 2 + 5 + 3

theorem banana_pies_count :
  let banana_ratio : ℕ := 3
  (banana_ratio * total_pies) / ratio_sum = 9 :=
by sorry

end banana_pies_count_l3737_373757


namespace pencil_count_l3737_373785

theorem pencil_count (num_pens : ℕ) (max_students : ℕ) (num_pencils : ℕ) : 
  num_pens = 640 →
  max_students = 40 →
  num_pens % max_students = 0 →
  num_pencils % max_students = 0 →
  ∃ k : ℕ, num_pencils = 40 * k :=
by
  sorry

end pencil_count_l3737_373785


namespace symmetric_point_coords_l3737_373762

/-- A point in a 2D plane represented by its x and y coordinates. -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Defines symmetry with respect to the y-axis. -/
def symmetricYAxis (a b : Point2D) : Prop :=
  b.x = -a.x ∧ b.y = a.y

/-- Theorem: If point B is symmetric to point A(2, -1) with respect to the y-axis,
    then the coordinates of point B are (-2, -1). -/
theorem symmetric_point_coords :
  let a : Point2D := ⟨2, -1⟩
  let b : Point2D := ⟨-2, -1⟩
  symmetricYAxis a b → b = ⟨-2, -1⟩ := by
  sorry

end symmetric_point_coords_l3737_373762


namespace speed_limit_exceeders_l3737_373765

/-- Represents the percentage of motorists who exceed the speed limit -/
def exceed_limit_percent : ℝ := sorry

/-- Represents the percentage of all motorists who receive speeding tickets -/
def receive_ticket_percent : ℝ := 10

/-- Represents the percentage of speed limit exceeders who do not receive tickets -/
def no_ticket_percent : ℝ := 30

theorem speed_limit_exceeders :
  exceed_limit_percent = 14 :=
by
  have h1 : receive_ticket_percent = exceed_limit_percent * (100 - no_ticket_percent) / 100 :=
    sorry
  sorry

end speed_limit_exceeders_l3737_373765


namespace sin_double_alpha_l3737_373778

theorem sin_double_alpha (α : Real) (h : Real.cos (π / 4 - α) = 4 / 5) : 
  Real.sin (2 * α) = 7 / 25 := by
  sorry

end sin_double_alpha_l3737_373778


namespace final_crayons_count_l3737_373787

def initial_crayons : ℝ := 7.5
def mary_took : ℝ := 3.2
def mark_took : ℝ := 0.5
def jane_took : ℝ := 1.3
def mary_returned : ℝ := 0.7
def sarah_added : ℝ := 3.5
def tom_added : ℝ := 2.8
def alice_took : ℝ := 1.5

theorem final_crayons_count :
  initial_crayons - mary_took - mark_took - jane_took + mary_returned + sarah_added + tom_added - alice_took = 8 := by
  sorry

end final_crayons_count_l3737_373787


namespace stock_price_calculation_l3737_373727

/-- Calculates the price of a stock given the income, dividend rate, and investment amount. -/
theorem stock_price_calculation (income : ℝ) (dividend_rate : ℝ) (investment : ℝ) :
  income = 450 →
  dividend_rate = 0.1 →
  investment = 4860 →
  (investment / (income / dividend_rate)) * 100 = 108 := by
  sorry

end stock_price_calculation_l3737_373727


namespace initial_fliers_count_l3737_373753

theorem initial_fliers_count (morning_fraction : ℚ) (afternoon_fraction : ℚ) (remaining_fliers : ℕ) : 
  morning_fraction = 1/5 →
  afternoon_fraction = 1/4 →
  remaining_fliers = 1500 →
  ∃ initial_fliers : ℕ, 
    initial_fliers = 2500 ∧
    (1 - morning_fraction) * (1 - afternoon_fraction) * initial_fliers = remaining_fliers :=
by
  sorry

end initial_fliers_count_l3737_373753


namespace geometric_sequence_property_l3737_373792

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_property (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 4 ^ 2 + 3 * a 4 + 1 = 0) →
  (a 12 ^ 2 + 3 * a 12 + 1 = 0) →
  a 8 = -1 := by
  sorry

end geometric_sequence_property_l3737_373792


namespace polynomial_product_sum_l3737_373742

theorem polynomial_product_sum (g h k : ℤ) : 
  (∀ d : ℤ, (5*d^2 + 4*d + g) * (4*d^2 + h*d - 5) = 20*d^4 + 11*d^3 - 9*d^2 + k*d - 20) →
  g + h + k = -16 := by
sorry

end polynomial_product_sum_l3737_373742


namespace min_reciprocal_sum_l3737_373739

theorem min_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  (1 / a + 1 / b) ≥ 2 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + b₀ = 2 ∧ 1 / a₀ + 1 / b₀ = 2 := by
  sorry

end min_reciprocal_sum_l3737_373739


namespace cupboard_cost_price_l3737_373777

theorem cupboard_cost_price (selling_price selling_price_increased : ℝ) 
  (h1 : selling_price = 0.84 * 5625)
  (h2 : selling_price_increased = 1.16 * 5625)
  (h3 : selling_price_increased - selling_price = 1800) : 
  5625 = 5625 := by sorry

end cupboard_cost_price_l3737_373777


namespace periodic_even_function_extension_l3737_373741

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem periodic_even_function_extension
  (f : ℝ → ℝ)
  (h_periodic : is_periodic f 2)
  (h_even : is_even f)
  (h_defined : ∀ x ∈ Set.Icc 2 3, f x = -2 * (x - 3)^2 + 4) :
  ∀ x ∈ Set.Icc 0 2, f x = -2 * (x - 1)^2 + 4 :=
sorry

end periodic_even_function_extension_l3737_373741


namespace inverse_g_sum_l3737_373764

noncomputable def g (x : ℝ) : ℝ :=
  if x ≤ 2 then 3 - x else 3 * x - x^2

theorem inverse_g_sum : ∃ (f : ℝ → ℝ), Function.LeftInverse f g ∧ Function.RightInverse f g ∧ f (-4) + f 0 + f 4 = 6 := by
  sorry

end inverse_g_sum_l3737_373764


namespace largest_three_digit_sum_l3737_373734

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- The sum of YXX + YX + ZY given X, Y, and Z -/
def sum (X Y Z : Digit) : ℕ :=
  111 * Y.val + 12 * X.val + 10 * Z.val

/-- Predicate to check if three digits are distinct -/
def distinct (X Y Z : Digit) : Prop :=
  X ≠ Y ∧ Y ≠ Z ∧ X ≠ Z

theorem largest_three_digit_sum :
  ∃ (X Y Z : Digit), distinct X Y Z ∧ 
    sum X Y Z ≤ 999 ∧
    ∀ (A B C : Digit), distinct A B C → sum A B C ≤ sum X Y Z :=
by
  sorry

end largest_three_digit_sum_l3737_373734


namespace half_angle_quadrant_l3737_373750

/-- An angle is in the third quadrant if it's between 180° and 270° (modulo 360°) -/
def is_in_third_quadrant (α : Real) : Prop :=
  ∃ k : ℤ, k * 360 + 180 < α ∧ α < k * 360 + 270

/-- An angle is in the second quadrant if it's between 90° and 180° (modulo 360°) -/
def is_in_second_quadrant (α : Real) : Prop :=
  ∃ n : ℤ, n * 360 + 90 < α ∧ α < n * 360 + 180

/-- An angle is in the fourth quadrant if it's between 270° and 360° (modulo 360°) -/
def is_in_fourth_quadrant (α : Real) : Prop :=
  ∃ n : ℤ, n * 360 + 270 < α ∧ α < n * 360 + 360

theorem half_angle_quadrant (α : Real) :
  is_in_third_quadrant α → is_in_second_quadrant (α/2) ∨ is_in_fourth_quadrant (α/2) := by
  sorry

end half_angle_quadrant_l3737_373750


namespace new_average_after_increase_l3737_373700

theorem new_average_after_increase (numbers : List ℝ) (h1 : numbers.length = 8) 
  (h2 : numbers.sum / numbers.length = 8) : 
  let new_numbers := numbers.map (λ x => if numbers.indexOf x < 5 then x + 4 else x)
  new_numbers.sum / new_numbers.length = 10.5 := by
sorry

end new_average_after_increase_l3737_373700


namespace prime_4n_2n_1_implies_n_power_of_3_l3737_373704

-- Define a function to check if a number is prime
def isPrime (p : ℕ) : Prop := p > 1 ∧ ∀ m : ℕ, m > 1 → m < p → ¬(p % m = 0)

-- Define a function to check if a number is a power of 3
def isPowerOf3 (n : ℕ) : Prop := ∃ k : ℕ, n = 3^k

-- Theorem statement
theorem prime_4n_2n_1_implies_n_power_of_3 (n : ℕ) :
  n > 0 → isPrime (4^n + 2^n + 1) → isPowerOf3 n :=
by sorry

end prime_4n_2n_1_implies_n_power_of_3_l3737_373704


namespace parabola_no_intersection_l3737_373763

/-- A parabola is defined by the equation y = -x^2 - 6x + m -/
def parabola (x m : ℝ) : ℝ := -x^2 - 6*x + m

/-- The parabola does not intersect the x-axis if it has no real roots -/
def no_intersection (m : ℝ) : Prop :=
  ∀ x : ℝ, parabola x m ≠ 0

/-- If the parabola does not intersect the x-axis, then m < -9 -/
theorem parabola_no_intersection (m : ℝ) :
  no_intersection m → m < -9 := by
  sorry

end parabola_no_intersection_l3737_373763


namespace total_reptiles_count_l3737_373784

/-- The number of swamps in the sanctuary -/
def num_swamps : ℕ := 4

/-- The number of reptiles in each swamp -/
def reptiles_per_swamp : ℕ := 356

/-- The total number of reptiles in all swamp areas -/
def total_reptiles : ℕ := num_swamps * reptiles_per_swamp

theorem total_reptiles_count : total_reptiles = 1424 := by
  sorry

end total_reptiles_count_l3737_373784


namespace football_inventory_solution_l3737_373776

/-- Represents the football inventory problem -/
structure FootballInventory where
  total_footballs : ℕ
  total_cost : ℕ
  football_a_purchase : ℕ
  football_a_marked : ℕ
  football_b_purchase : ℕ
  football_b_marked : ℕ
  football_a_discount : ℚ
  football_b_discount : ℚ

/-- The specific football inventory problem instance -/
def problem : FootballInventory :=
  { total_footballs := 200
  , total_cost := 14400
  , football_a_purchase := 80
  , football_a_marked := 120
  , football_b_purchase := 60
  , football_b_marked := 90
  , football_a_discount := 1/5
  , football_b_discount := 1/10
  }

/-- Theorem stating the solution to the football inventory problem -/
theorem football_inventory_solution (p : FootballInventory) 
  (h1 : p = problem) : 
  ∃ (a b profit : ℕ), 
    a + b = p.total_footballs ∧ 
    a * p.football_a_purchase + b * p.football_b_purchase = p.total_cost ∧
    a = 120 ∧ 
    b = 80 ∧
    profit = a * (p.football_a_marked * (1 - p.football_a_discount) - p.football_a_purchase) + 
             b * (p.football_b_marked * (1 - p.football_b_discount) - p.football_b_purchase) ∧
    profit = 3600 :=
by
  sorry

end football_inventory_solution_l3737_373776


namespace triangle_properties_l3737_373731

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given conditions for the triangle -/
def TriangleConditions (t : Triangle) : Prop :=
  t.a / Real.cos t.A = t.c / (2 - Real.cos t.C) ∧
  t.b = 4 ∧
  t.c = 3 ∧
  (1/2) * t.a * t.b * Real.sin t.C = 3

theorem triangle_properties (t : Triangle) (h : TriangleConditions t) :
  t.a = 2 ∧ 3 * Real.sin t.C + 4 * Real.cos t.C = 5 := by
  sorry

end triangle_properties_l3737_373731


namespace xyz_bounds_l3737_373736

-- Define the problem
theorem xyz_bounds (x y z a : ℝ) (ha : a > 0) 
  (h1 : x + y + z = a) (h2 : x^2 + y^2 + z^2 = a^2 / 2) :
  (0 ≤ x ∧ x ≤ 2*a/3) ∧ (0 ≤ y ∧ y ≤ 2*a/3) ∧ (0 ≤ z ∧ z ≤ 2*a/3) := by
  sorry

end xyz_bounds_l3737_373736


namespace sam_total_spending_l3737_373747

/-- The value of a penny in dollars -/
def penny_value : ℚ := 1 / 100

/-- The value of a nickel in dollars -/
def nickel_value : ℚ := 5 / 100

/-- The value of a dime in dollars -/
def dime_value : ℚ := 1 / 10

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 25 / 100

/-- Calculate the total value of coins -/
def coin_value (pennies nickels dimes quarters : ℕ) : ℚ :=
  (pennies : ℚ) * penny_value + (nickels : ℚ) * nickel_value +
  (dimes : ℚ) * dime_value + (quarters : ℚ) * quarter_value

/-- Sam's spending for each day of the week -/
def monday_spending : ℚ := coin_value 5 3 0 0
def tuesday_spending : ℚ := coin_value 0 0 8 4
def wednesday_spending : ℚ := coin_value 0 7 10 2
def thursday_spending : ℚ := coin_value 20 15 12 6
def friday_spending : ℚ := coin_value 45 20 25 10

/-- The total amount Sam spent during the week -/
def total_spending : ℚ :=
  monday_spending + tuesday_spending + wednesday_spending + thursday_spending + friday_spending

/-- Theorem: Sam spent $14.05 in total during the week -/
theorem sam_total_spending : total_spending = 1405 / 100 := by
  sorry


end sam_total_spending_l3737_373747


namespace exists_real_cube_less_than_one_no_rational_square_root_of_two_not_all_natural_cube_greater_than_square_all_real_square_plus_one_positive_l3737_373702

-- Statement 1
theorem exists_real_cube_less_than_one : ∃ x : ℝ, x^3 < 1 := by sorry

-- Statement 2
theorem no_rational_square_root_of_two : ¬ ∃ x : ℚ, x^2 = 2 := by sorry

-- Statement 3
theorem not_all_natural_cube_greater_than_square : 
  ¬ ∀ x : ℕ, x^3 > x^2 := by sorry

-- Statement 4
theorem all_real_square_plus_one_positive : 
  ∀ x : ℝ, x^2 + 1 > 0 := by sorry

end exists_real_cube_less_than_one_no_rational_square_root_of_two_not_all_natural_cube_greater_than_square_all_real_square_plus_one_positive_l3737_373702


namespace triangular_pyramid_volume_l3737_373781

/-- The volume of a triangular pyramid formed by intersecting a right prism with a plane --/
theorem triangular_pyramid_volume 
  (a α β φ : ℝ) 
  (ha : a > 0)
  (hα : 0 < α ∧ α < π)
  (hβ : 0 < β ∧ β < π)
  (hαβ : α + β < π)
  (hφ : 0 < φ ∧ φ < π/2) :
  ∃ V : ℝ, V = (a^3 * Real.sin α^2 * Real.sin β^2 * Real.tan φ) / (6 * Real.sin (α + β)^2) :=
by sorry

end triangular_pyramid_volume_l3737_373781


namespace meeting_arrangements_presidency_meeting_arrangements_l3737_373793

/-- Represents a school in the club --/
structure School :=
  (members : Nat)

/-- Represents the club --/
structure Club :=
  (schools : Finset School)
  (total_members : Nat)

/-- Represents a meeting arrangement --/
structure MeetingArrangement :=
  (host : School)
  (host_representatives : Nat)
  (other_representatives : Nat)

/-- The number of ways to choose k items from n items --/
def choose (n k : Nat) : Nat :=
  Nat.choose n k

/-- Theorem: Number of possible meeting arrangements --/
theorem meeting_arrangements (club : Club) (arrangement : MeetingArrangement) : Nat :=
  let num_schools := Finset.card club.schools
  let host_choices := choose num_schools 1
  let host_rep_choices := choose arrangement.host.members arrangement.host_representatives
  let other_rep_choices := (choose arrangement.host.members arrangement.other_representatives) ^ (num_schools - 1)
  host_choices * host_rep_choices * other_rep_choices

/-- Main theorem: Prove the number of possible arrangements is 40,000 --/
theorem presidency_meeting_arrangements :
  ∀ (club : Club) (arrangement : MeetingArrangement),
    Finset.card club.schools = 4 →
    (∀ s ∈ club.schools, s.members = 5) →
    club.total_members = 20 →
    arrangement.host_representatives = 3 →
    arrangement.other_representatives = 2 →
    meeting_arrangements club arrangement = 40000 :=
sorry

end meeting_arrangements_presidency_meeting_arrangements_l3737_373793


namespace power_sum_geq_product_l3737_373745

theorem power_sum_geq_product (a b c : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c)
  (sum_one : a + b + c = 1) : 
  a^4 + b^4 + c^4 ≥ a*b*c :=
by sorry

end power_sum_geq_product_l3737_373745


namespace product_sum_equals_30_l3737_373751

theorem product_sum_equals_30 (a b c d : ℝ) 
  (eq1 : a + b + c = 5)
  (eq2 : a + b + d = 3)
  (eq3 : a + c + d = 8)
  (eq4 : b + c + d = 17) : 
  a * b + c * d = 30 := by
  sorry

end product_sum_equals_30_l3737_373751


namespace parallel_line_slope_intercept_l3737_373714

/-- The slope-intercept form of a line parallel to 4x + y - 2 = 0 and passing through (3, 2) -/
theorem parallel_line_slope_intercept :
  ∃ (m b : ℝ), 
    (∀ (x y : ℝ), 4 * x + y - 2 = 0 → y = -4 * x + b) ∧ 
    (2 = m * 3 + b) ∧
    (∀ (x y : ℝ), y = m * x + b ↔ y = -4 * x + 14) :=
by sorry

end parallel_line_slope_intercept_l3737_373714


namespace animath_interns_pigeonhole_l3737_373755

theorem animath_interns_pigeonhole (n : ℕ) (h : n > 0) :
  ∃ (i j : Fin n) (k : ℕ), i ≠ j ∧
  (∃ (f : Fin n → ℕ), (∀ x, f x < n - 1) ∧ f i = k ∧ f j = k) :=
sorry

end animath_interns_pigeonhole_l3737_373755


namespace triangular_fence_perimeter_l3737_373701

/-- Calculates the perimeter of a triangular fence with evenly spaced posts -/
theorem triangular_fence_perimeter
  (num_posts : ℕ)
  (post_width : ℝ)
  (post_spacing : ℝ)
  (h_num_posts : num_posts = 18)
  (h_post_width : post_width = 0.5)
  (h_post_spacing : post_spacing = 4)
  (h_divisible : num_posts % 3 = 0) :
  let posts_per_side := num_posts / 3
  let side_length := (posts_per_side - 1) * post_spacing + posts_per_side * post_width
  3 * side_length = 69 := by sorry

end triangular_fence_perimeter_l3737_373701


namespace marble_202_is_white_l3737_373722

/-- Represents the colors of marbles -/
inductive Color
  | Gray
  | White
  | Black
  | Red

/-- Returns the color of the nth marble in the repeating pattern -/
def marbleColor (n : ℕ) : Color :=
  match n % 15 with
  | 0 | 1 | 2 | 3 | 4 | 5 => Color.Gray
  | 6 | 7 | 8 => Color.White
  | 9 | 10 | 11 | 12 => Color.Black
  | _ => Color.Red

theorem marble_202_is_white :
  marbleColor 202 = Color.White := by
  sorry

end marble_202_is_white_l3737_373722


namespace traffic_survey_l3737_373711

theorem traffic_survey (N : ℕ) 
  (drivers_A : ℕ) (sample_A : ℕ) (sample_B : ℕ) (sample_C : ℕ) (sample_D : ℕ) : 
  drivers_A = 96 →
  sample_A = 12 →
  sample_B = 21 →
  sample_C = 25 →
  sample_D = 43 →
  N = (sample_A + sample_B + sample_C + sample_D) * drivers_A / sample_A →
  N = 808 := by
sorry

end traffic_survey_l3737_373711


namespace trailing_zeros_80_factorial_l3737_373774

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ := sorry

theorem trailing_zeros_80_factorial :
  trailingZeros 73 = 16 → trailingZeros 80 = 18 := by sorry

end trailing_zeros_80_factorial_l3737_373774


namespace no_rectangular_prism_with_diagonals_7_8_11_l3737_373723

theorem no_rectangular_prism_with_diagonals_7_8_11 :
  ¬ ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    ({7^2, 8^2, 11^2} : Finset ℝ) = {a^2 + b^2, b^2 + c^2, a^2 + c^2} :=
by sorry

end no_rectangular_prism_with_diagonals_7_8_11_l3737_373723


namespace megan_popsicle_consumption_l3737_373768

/-- The number of Popsicles Megan consumes in a given time period -/
def popsicles_consumed (minutes_per_popsicle : ℕ) (total_minutes : ℕ) : ℕ :=
  total_minutes / minutes_per_popsicle

theorem megan_popsicle_consumption :
  popsicles_consumed 18 (5 * 60 + 36) = 18 := by
  sorry

end megan_popsicle_consumption_l3737_373768


namespace parabola_intersects_x_axis_l3737_373743

-- Define the parabola
def parabola (x m : ℝ) : ℝ := x^2 + 2*x + m - 1

-- Theorem statement
theorem parabola_intersects_x_axis (m : ℝ) :
  (∃ x : ℝ, parabola x m = 0) ↔ m ≤ 2 := by sorry

end parabola_intersects_x_axis_l3737_373743


namespace line_circle_separation_l3737_373721

/-- Given a point P(x₀, y₀) inside a circle C: x² + y² = r², 
    the line xx₀ + yy₀ = r² is separated from the circle C. -/
theorem line_circle_separation 
  (x₀ y₀ r : ℝ) 
  (h_inside : x₀^2 + y₀^2 < r^2) : 
  let d := r^2 / Real.sqrt (x₀^2 + y₀^2)
  d > r := by
sorry

end line_circle_separation_l3737_373721


namespace pascal_all_even_rows_l3737_373772

/-- Returns true if a row in Pascal's triangle consists of all even numbers except for the 1s at each end -/
def isAllEvenExceptEnds (row : ℕ) : Bool := sorry

/-- Counts the number of rows in Pascal's triangle from row 2 to row 30 (inclusive) that consist of all even numbers except for the 1s at each end -/
def countAllEvenRows : ℕ := sorry

theorem pascal_all_even_rows : countAllEvenRows = 4 := by sorry

end pascal_all_even_rows_l3737_373772


namespace imaginary_unit_sum_l3737_373726

-- Define the complex number i
def i : ℂ := Complex.I

-- Theorem statement
theorem imaginary_unit_sum : i + i^3 = 0 := by sorry

end imaginary_unit_sum_l3737_373726


namespace system_equation_solution_l3737_373783

theorem system_equation_solution :
  ∀ (x₁ x₂ x₃ x₄ x₅ : ℝ),
  2*x₁ + x₂ + x₃ + x₄ + x₅ = 6 →
  x₁ + 2*x₂ + x₃ + x₄ + x₅ = 12 →
  x₁ + x₂ + 2*x₃ + x₄ + x₅ = 24 →
  x₁ + x₂ + x₃ + 2*x₄ + x₅ = 48 →
  x₁ + x₂ + x₃ + x₄ + 2*x₅ = 96 →
  3*x₄ + 2*x₅ = 181 :=
by
  sorry

end system_equation_solution_l3737_373783


namespace egg_leftover_proof_l3737_373737

/-- The number of eggs left over when selling a given number of eggs in cartons of 10 -/
def leftover_eggs (total_eggs : ℕ) : ℕ :=
  total_eggs % 10

theorem egg_leftover_proof (john_eggs maria_eggs nikhil_eggs : ℕ) 
  (h1 : john_eggs = 45)
  (h2 : maria_eggs = 38)
  (h3 : nikhil_eggs = 29) :
  leftover_eggs (john_eggs + maria_eggs + nikhil_eggs) = 2 := by
  sorry

end egg_leftover_proof_l3737_373737


namespace symmetric_line_equation_l3737_373709

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The line y = x -/
def bisector_line : Line := { a := 1, b := -1, c := 0 }

/-- Checks if a line is the angle bisector of two other lines -/
def is_angle_bisector (bisector : Line) (l1 : Line) (l2 : Line) : Prop := sorry

/-- Theorem: If the bisector of the angle between lines l₁ and l₂ is y = x,
    and the equation of l₁ is ax + by + c = 0 (ab > 0),
    then the equation of l₂ is bx + ay + c = 0 -/
theorem symmetric_line_equation (l1 : Line) (l2 : Line) 
    (h1 : is_angle_bisector bisector_line l1 l2)
    (h2 : l1.a * l1.b > 0) : 
  l2.a = l1.b ∧ l2.b = l1.a ∧ l2.c = l1.c := by
  sorry

end symmetric_line_equation_l3737_373709


namespace stones_for_hall_l3737_373799

/-- Calculates the number of stones required to pave a hall -/
def stones_required (hall_length hall_width stone_length stone_width : ℚ) : ℕ :=
  let hall_area := hall_length * hall_width * 100
  let stone_area := stone_length * stone_width
  (hall_area / stone_area).num.natAbs

/-- Theorem stating that 9000 stones are required to pave the given hall -/
theorem stones_for_hall : stones_required 72 30 4 6 = 9000 := by
  sorry

end stones_for_hall_l3737_373799


namespace tangent_parallel_implies_a_equals_one_l3737_373738

-- Define the curve
def curve (a : ℝ) (x : ℝ) : ℝ := a * x^2

-- Define the tangent line
def tangent_line (a : ℝ) (x : ℝ) : ℝ := 2 * a * (x - 1) + a

-- Define the given line
def given_line (x : ℝ) : ℝ := 2 * x - 6

theorem tangent_parallel_implies_a_equals_one (a : ℝ) :
  (∀ x : ℝ, tangent_line a x = given_line x) →
  a = 1 :=
by sorry

end tangent_parallel_implies_a_equals_one_l3737_373738


namespace power_two_plus_one_div_by_three_l3737_373725

theorem power_two_plus_one_div_by_three (n : ℕ) : 
  3 ∣ (2^n + 1) ↔ Odd n := by sorry

end power_two_plus_one_div_by_three_l3737_373725


namespace polygon_with_135_degree_angles_is_octagon_l3737_373712

theorem polygon_with_135_degree_angles_is_octagon :
  ∀ (n : ℕ) (interior_angle : ℝ),
    n ≥ 3 →
    interior_angle = 135 →
    (n - 2) * 180 / n = interior_angle →
    n = 8 :=
by
  sorry

end polygon_with_135_degree_angles_is_octagon_l3737_373712


namespace a_plus_b_value_m_range_l3737_373730

-- Define the sets A and B
def A (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b ≤ 0}
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*m*x + m^2 - 4 < 0}

-- Theorem 1
theorem a_plus_b_value :
  ∀ a b : ℝ, A a b = {x | -1 ≤ x ∧ x ≤ 4} → a + b = -7 :=
sorry

-- Theorem 2
theorem m_range (a b : ℝ) :
  A a b = {x | -1 ≤ x ∧ x ≤ 4} →
  (∀ x : ℝ, x ∈ A a b → x ∉ B m) →
  m ≤ -3 ∨ m ≥ 6 :=
sorry

end a_plus_b_value_m_range_l3737_373730


namespace abie_chips_count_l3737_373797

theorem abie_chips_count (initial bags_given bags_bought : ℕ) 
  (h1 : initial = 20)
  (h2 : bags_given = 4)
  (h3 : bags_bought = 6) : 
  initial - bags_given + bags_bought = 22 := by
  sorry

end abie_chips_count_l3737_373797


namespace anton_card_difference_l3737_373748

/-- Given that Anton has three times as many cards as Heike, Ann has the same number of cards as Heike, 
    and Ann has 60 cards, prove that Anton has 120 more cards than Ann. -/
theorem anton_card_difference (heike_cards : ℕ) (ann_cards : ℕ) (anton_cards : ℕ) 
    (h1 : anton_cards = 3 * heike_cards)
    (h2 : ann_cards = heike_cards)
    (h3 : ann_cards = 60) : 
  anton_cards - ann_cards = 120 := by
  sorry

end anton_card_difference_l3737_373748


namespace system_solution_l3737_373754

theorem system_solution (x y a : ℝ) : 
  (4 * x + y = a ∧ 2 * x + 5 * y = 3 * a ∧ x = 2) → a = 18 := by
  sorry

end system_solution_l3737_373754


namespace derivative_sin_cos_plus_one_l3737_373713

theorem derivative_sin_cos_plus_one (x : ℝ) : 
  let f : ℝ → ℝ := λ x => Real.sin x * (Real.cos x + 1)
  (deriv f) x = Real.cos (2 * x) + Real.cos x := by
sorry

end derivative_sin_cos_plus_one_l3737_373713


namespace onion_chop_time_is_four_l3737_373740

/-- Represents the time in minutes for Bill's omelet preparation tasks -/
structure OmeletPrep where
  pepper_chop_time : ℕ
  cheese_grate_time : ℕ
  assemble_cook_time : ℕ
  total_peppers : ℕ
  total_onions : ℕ
  total_omelets : ℕ
  total_prep_time : ℕ

/-- Calculates the time to chop an onion given the omelet preparation details -/
def time_to_chop_onion (prep : OmeletPrep) : ℕ :=
  let pepper_time := prep.pepper_chop_time * prep.total_peppers
  let cheese_time := prep.cheese_grate_time * prep.total_omelets
  let cook_time := prep.assemble_cook_time * prep.total_omelets
  let remaining_time := prep.total_prep_time - (pepper_time + cheese_time + cook_time)
  remaining_time / prep.total_onions

/-- Theorem stating that it takes 4 minutes to chop an onion given the specific conditions -/
theorem onion_chop_time_is_four : 
  let prep : OmeletPrep := {
    pepper_chop_time := 3,
    cheese_grate_time := 1,
    assemble_cook_time := 5,
    total_peppers := 4,
    total_onions := 2,
    total_omelets := 5,
    total_prep_time := 50
  }
  time_to_chop_onion prep = 4 := by
  sorry

end onion_chop_time_is_four_l3737_373740


namespace fifi_green_hangers_l3737_373767

/-- The number of green hangers in Fifi's closet -/
def green_hangers : ℕ := 4

/-- The number of pink hangers in Fifi's closet -/
def pink_hangers : ℕ := 7

/-- The number of blue hangers in Fifi's closet -/
def blue_hangers : ℕ := green_hangers - 1

/-- The number of yellow hangers in Fifi's closet -/
def yellow_hangers : ℕ := blue_hangers - 1

/-- The total number of hangers in Fifi's closet -/
def total_hangers : ℕ := 16

theorem fifi_green_hangers :
  pink_hangers + green_hangers + blue_hangers + yellow_hangers = total_hangers :=
by sorry

end fifi_green_hangers_l3737_373767


namespace combined_weight_of_boxes_l3737_373790

theorem combined_weight_of_boxes (box1 box2 box3 : ℕ) 
  (h1 : box1 = 2) 
  (h2 : box2 = 11) 
  (h3 : box3 = 5) : 
  box1 + box2 + box3 = 18 := by
  sorry

end combined_weight_of_boxes_l3737_373790


namespace inverse_sum_reciprocal_l3737_373733

theorem inverse_sum_reciprocal (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (a⁻¹ + b⁻¹ + c⁻¹)⁻¹ = (a * b * c) / (a * b + a * c + b * c) := by
  sorry

end inverse_sum_reciprocal_l3737_373733


namespace seven_solutions_condition_l3737_373758

-- Define the function f
def f (x : ℝ) : ℝ := |x^2 - 1| - 1

-- State the theorem
theorem seven_solutions_condition (b c : ℝ) :
  (∃! (s : Finset ℝ), s.card = 7 ∧ ∀ x ∈ s, f x ^ 2 - b * f x + c = 0) ↔ 
  (c ≤ 0 ∧ 0 < b ∧ b < 1) :=
sorry

end seven_solutions_condition_l3737_373758


namespace stick_cutting_l3737_373760

theorem stick_cutting (short_length long_length : ℝ) : 
  short_length > 0 →
  long_length = short_length + 12 →
  short_length + long_length = 20 →
  (long_length / short_length : ℝ) = 4 := by
sorry

end stick_cutting_l3737_373760


namespace parabola_equation_l3737_373786

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space using the general form ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a parabola in its general form ax^2 + bxy + cy^2 + dx + ey + f = 0 -/
structure Parabola where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  e : ℤ
  f : ℤ

/-- Function to calculate the greatest common divisor of six integers -/
def gcd6 (a b c d e f : ℤ) : ℤ := sorry

/-- Theorem stating the equation of the parabola with given focus and directrix -/
theorem parabola_equation (focus : Point) (directrix : Line) : 
  focus.x = 2 ∧ focus.y = 4 ∧ 
  directrix.a = 4 ∧ directrix.b = 5 ∧ directrix.c = 20 → 
  ∃ (p : Parabola), 
    p.a = 25 ∧ p.b = -40 ∧ p.c = 16 ∧ p.d = 0 ∧ p.e = 0 ∧ p.f = 0 ∧ 
    p.a > 0 ∧ 
    gcd6 (abs p.a) (abs p.b) (abs p.c) (abs p.d) (abs p.e) (abs p.f) = 1 := by
  sorry

end parabola_equation_l3737_373786


namespace function_positivity_condition_l3737_373788

theorem function_positivity_condition (m : ℝ) : 
  (∀ x : ℝ, max (2*m*x^2 - 2*(4-m)*x + 1) (m*x) > 0) ↔ (0 < m ∧ m < 8) :=
sorry

end function_positivity_condition_l3737_373788


namespace hotel_rate_problem_l3737_373735

-- Define the flat rate for the first night and the nightly rate for additional nights
variable (f : ℝ) -- Flat rate for the first night
variable (n : ℝ) -- Nightly rate for additional nights

-- Define Alice's stay
def alice_stay : ℝ := f + 4 * n

-- Define Bob's stay
def bob_stay : ℝ := f + 9 * n

-- State the theorem
theorem hotel_rate_problem (h1 : alice_stay = 245) (h2 : bob_stay = 470) : f = 65 := by
  sorry

end hotel_rate_problem_l3737_373735


namespace max_elevation_l3737_373717

/-- The elevation function of a particle projected vertically upward -/
def s (t : ℝ) : ℝ := 200 * t - 20 * t^2

/-- The maximum elevation reached by the particle -/
theorem max_elevation : ∃ (t : ℝ), ∀ (t' : ℝ), s t' ≤ s t ∧ s t = 500 := by
  sorry

end max_elevation_l3737_373717


namespace isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l3737_373795

/-- An isosceles triangle with base 10 and equal sides 7 has perimeter 24 -/
theorem isosceles_triangle_perimeter : ℝ → ℝ → ℝ → Prop :=
  fun base side perimeter =>
    base = 10 ∧ side = 7 ∧ perimeter = base + 2 * side → perimeter = 24

-- The proof is omitted
theorem isosceles_triangle_perimeter_proof : isosceles_triangle_perimeter 10 7 24 := by sorry

end isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l3737_373795


namespace equation_infinite_solutions_l3737_373749

theorem equation_infinite_solutions (a b : ℝ) : 
  b = 1 → 
  (∀ x : ℝ, a * (3 * x - 2) + b * (2 * x - 3) = 8 * x - 7) → 
  a = 2 := by
sorry

end equation_infinite_solutions_l3737_373749


namespace price_change_theorem_l3737_373708

theorem price_change_theorem (p : ℝ) : 
  (1 + p / 100) * (1 - p / 200) = 1 + p / 300 → p = 100 / 3 :=
by sorry

end price_change_theorem_l3737_373708


namespace rowing_speed_contradiction_l3737_373706

theorem rowing_speed_contradiction (man_rate : ℝ) (with_stream : ℝ) (against_stream : ℝ) :
  man_rate = 6 →
  with_stream = 20 →
  with_stream = man_rate + (with_stream - man_rate) →
  against_stream = man_rate - (with_stream - man_rate) →
  against_stream < 0 :=
by sorry

#check rowing_speed_contradiction

end rowing_speed_contradiction_l3737_373706


namespace f_is_convex_f_range_a_l3737_373761

/-- Definition of a convex function -/
def IsConvex (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, f ((x₁ + x₂) / 2) ≤ (f x₁ + f x₂) / 2

/-- The quadratic function f(x) = ax^2 + x -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + x

/-- Theorem: f is convex when a > 0 -/
theorem f_is_convex (a : ℝ) (ha : a > 0) : IsConvex (f a) := by sorry

/-- Theorem: Range of a when |f(x)| ≤ 1 for x ∈ [0,1] -/
theorem f_range_a (a : ℝ) (ha : a ≠ 0) :
  (∀ x : ℝ, x ∈ Set.Icc 0 1 → |f a x| ≤ 1) ↔ a ∈ Set.Icc (-2) 0 := by sorry

end f_is_convex_f_range_a_l3737_373761


namespace charitable_distribution_boy_amount_l3737_373728

def charitable_distribution (initial_pennies : ℕ) 
  (farmer_pennies : ℕ) (beggar_pennies : ℕ) (boy_pennies : ℕ) : Prop :=
  initial_pennies = 42 ∧
  farmer_pennies = initial_pennies / 2 + 1 ∧
  beggar_pennies = (initial_pennies - farmer_pennies) / 2 + 2 ∧
  boy_pennies = initial_pennies - farmer_pennies - beggar_pennies - 1

theorem charitable_distribution_boy_amount :
  ∀ (initial_pennies farmer_pennies beggar_pennies boy_pennies : ℕ),
  charitable_distribution initial_pennies farmer_pennies beggar_pennies boy_pennies →
  boy_pennies = 7 :=
by sorry

end charitable_distribution_boy_amount_l3737_373728


namespace carreys_fixed_amount_is_20_l3737_373705

/-- The fixed amount Carrey paid for the car rental -/
def carreys_fixed_amount : ℝ := 20

/-- The rate per kilometer for Carrey's rental -/
def carreys_rate_per_km : ℝ := 0.25

/-- The fixed amount Samuel paid for the car rental -/
def samuels_fixed_amount : ℝ := 24

/-- The rate per kilometer for Samuel's rental -/
def samuels_rate_per_km : ℝ := 0.16

/-- The number of kilometers driven by both Carrey and Samuel -/
def kilometers_driven : ℝ := 44.44444444444444

theorem carreys_fixed_amount_is_20 :
  carreys_fixed_amount + carreys_rate_per_km * kilometers_driven =
  samuels_fixed_amount + samuels_rate_per_km * kilometers_driven :=
sorry

end carreys_fixed_amount_is_20_l3737_373705


namespace expression_simplification_l3737_373775

theorem expression_simplification (x y : ℝ) 
  (h : y = Real.sqrt (x - 3) + Real.sqrt (6 - 2*x) + 2) : 
  Real.sqrt (2*x) * Real.sqrt (x/y) * (Real.sqrt (y/x) + Real.sqrt (1/y)) = 
    Real.sqrt 6 + (3 * Real.sqrt 2) / 2 := by
  sorry

end expression_simplification_l3737_373775


namespace log_equation_holds_l3737_373759

theorem log_equation_holds (x : ℝ) (h1 : x > 0) (h2 : x ≠ 1) :
  (Real.log x / Real.log 2) * (Real.log 7 / Real.log x) + Real.log 7 / Real.log 10 = Real.log 7 / Real.log 2 :=
by sorry

end log_equation_holds_l3737_373759


namespace congruence_sufficient_not_necessary_for_equal_area_l3737_373771

-- Define the property of two triangles being congruent
def are_congruent (t1 t2 : Triangle) : Prop := sorry

-- Define the property of two triangles having equal area
def have_equal_area (t1 t2 : Triangle) : Prop := sorry

-- Theorem stating that congruence is sufficient but not necessary for equal area
theorem congruence_sufficient_not_necessary_for_equal_area :
  (∀ t1 t2 : Triangle, are_congruent t1 t2 → have_equal_area t1 t2) ∧
  (∃ t1 t2 : Triangle, have_equal_area t1 t2 ∧ ¬are_congruent t1 t2) := by sorry

end congruence_sufficient_not_necessary_for_equal_area_l3737_373771


namespace square_area_increase_l3737_373718

theorem square_area_increase (s : ℝ) (h : s > 0) :
  let new_side := 1.35 * s
  let original_area := s^2
  let new_area := new_side^2
  (new_area - original_area) / original_area = 0.8225 := by
  sorry

end square_area_increase_l3737_373718


namespace perfect_square_primes_l3737_373715

theorem perfect_square_primes (p : ℕ) : 
  Nat.Prime p ∧ ∃ (n : ℕ), (2^(p+1) - 4) / p = n^2 ↔ p = 3 ∨ p = 7 := by
sorry

end perfect_square_primes_l3737_373715


namespace division_property_l3737_373769

theorem division_property (n : ℕ) : 
  (n / 5 = 248) ∧ (n % 5 = 4) → (n / 9 + n % 9 = 140) := by
  sorry

end division_property_l3737_373769


namespace fraction_problem_l3737_373789

theorem fraction_problem (p q x y : ℚ) :
  p / q = 4 / 5 →
  x / y + (2 * q - p) / (2 * q + p) = 3 →
  x / y = 18 / 7 := by
  sorry

end fraction_problem_l3737_373789


namespace wendy_shoes_left_l3737_373773

theorem wendy_shoes_left (total : ℕ) (given_away : ℕ) (h1 : total = 33) (h2 : given_away = 14) :
  total - given_away = 19 := by
  sorry

end wendy_shoes_left_l3737_373773


namespace min_cost_butter_l3737_373720

/-- The cost of a 16 oz package of butter -/
def cost_16oz : ℝ := 7

/-- The cost of an 8 oz package of butter -/
def cost_8oz : ℝ := 4

/-- The cost of a 4 oz package of butter before discount -/
def cost_4oz : ℝ := 2

/-- The discount rate applied to 4 oz packages -/
def discount_rate : ℝ := 0.5

/-- The total amount of butter needed in ounces -/
def butter_needed : ℝ := 16

/-- Theorem stating that the minimum cost of purchasing 16 oz of butter is $6.0 -/
theorem min_cost_butter : 
  min cost_16oz (cost_8oz + 2 * (cost_4oz * (1 - discount_rate))) = 6 := by sorry

end min_cost_butter_l3737_373720


namespace inequalities_hold_l3737_373782

theorem inequalities_hold (a b c x y z : ℝ) 
  (h1 : x^2 < a^2) (h2 : y^2 < b^2) (h3 : z^2 < c^2) :
  (x^2 + y^2 + z^2 < a^2 + b^2 + c^2) ∧ 
  (x^3 + y^3 + z^3 < a^3 + b^3 + c^3) := by
  sorry

end inequalities_hold_l3737_373782


namespace parallel_postulate_l3737_373798

-- Define a structure for points and lines in a 2D Euclidean plane
structure EuclideanPlane where
  Point : Type
  Line : Type
  on_line : Point → Line → Prop
  parallel : Line → Line → Prop

-- State the theorem
theorem parallel_postulate (plane : EuclideanPlane) 
  (l : plane.Line) (p : plane.Point) (h : ¬ plane.on_line p l) :
  ∃! m : plane.Line, plane.on_line p m ∧ plane.parallel m l :=
sorry

end parallel_postulate_l3737_373798


namespace julies_work_hours_l3737_373796

theorem julies_work_hours (hourly_rate : ℝ) (days_per_week : ℕ) (monthly_salary : ℝ) :
  hourly_rate = 5 →
  days_per_week = 6 →
  monthly_salary = 920 →
  (monthly_salary / hourly_rate) / (days_per_week * 4 - 1) = 8 := by
  sorry

end julies_work_hours_l3737_373796


namespace fraction_equality_l3737_373752

theorem fraction_equality (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : (4*a - b) / (a + 4*b) = 3) : 
  (a - 4*b) / (4*a + b) = 9 / 53 := by
  sorry

end fraction_equality_l3737_373752


namespace solution_to_equation_l3737_373716

theorem solution_to_equation : 
  ∃ (x₁ x₂ : ℝ), 
    (x₁ = 1.5 + Real.sqrt 1.5 ∧ x₂ = 1.5 - Real.sqrt 1.5) ∧ 
    (∀ x : ℝ, x^4 + (3 - x)^4 = 130 ↔ (x = x₁ ∨ x = x₂)) := by
  sorry

end solution_to_equation_l3737_373716


namespace butterfly_distribution_theorem_l3737_373779

/-- Represents the movement rules for butterflies on a cube --/
structure ButterflyMovement where
  adjacent : ℕ  -- Number of butterflies moving to each adjacent vertex
  opposite : ℕ  -- Number of butterflies moving to the opposite vertex
  flyaway : ℕ   -- Number of butterflies flying away

/-- Represents the state of butterflies on a cube --/
structure CubeState where
  vertices : Fin 8 → ℕ  -- Number of butterflies at each vertex

/-- Defines the condition for equal distribution of butterflies --/
def is_equally_distributed (state : CubeState) : Prop :=
  ∀ i j : Fin 8, state.vertices i = state.vertices j

/-- Defines the evolution of the cube state according to movement rules --/
def evolve (initial : CubeState) (rules : ButterflyMovement) : ℕ → CubeState
  | 0 => initial
  | n+1 => sorry  -- Implementation of evolution step

/-- Main theorem: N must be a multiple of 45 for equal distribution --/
theorem butterfly_distribution_theorem 
  (N : ℕ) 
  (initial : CubeState) 
  (rules : ButterflyMovement) 
  (h_initial : ∃ v : Fin 8, initial.vertices v = N ∧ ∀ w : Fin 8, w ≠ v → initial.vertices w = 0)
  (h_rules : rules.adjacent = 3 ∧ rules.opposite = 1 ∧ rules.flyaway = 1) :
  (∃ t : ℕ, is_equally_distributed (evolve initial rules t)) ↔ ∃ k : ℕ, N = 45 * k :=
sorry

end butterfly_distribution_theorem_l3737_373779


namespace toys_production_time_l3737_373766

theorem toys_production_time (goal : ℕ) (rate : ℕ) (days_worked : ℕ) (days_left : ℕ) : 
  goal = 1000 → 
  rate = 100 → 
  days_worked = 6 → 
  rate * days_worked + rate * days_left = goal → 
  days_left = 4 := by
sorry

end toys_production_time_l3737_373766


namespace stratified_sampling_size_l3737_373746

theorem stratified_sampling_size (total_population : ℕ) (stratum_size : ℕ) (stratum_sample : ℕ) (h1 : total_population = 3600) (h2 : stratum_size = 1000) (h3 : stratum_sample = 25) : 
  (stratum_size : ℚ) / total_population * (total_sample : ℚ) = stratum_sample → total_sample = 90 :=
by
  sorry

end stratified_sampling_size_l3737_373746


namespace probability_point_near_vertex_l3737_373791

/-- The probability of a randomly selected point from a square being within a certain distance from a vertex -/
theorem probability_point_near_vertex (side_length : ℝ) (distance : ℝ) : 
  side_length > 0 → distance > 0 → distance ≤ side_length →
  (π * distance^2) / (4 * side_length^2) = π / 16 ↔ side_length = 4 ∧ distance = 2 :=
by sorry

end probability_point_near_vertex_l3737_373791


namespace hyperbola_eccentricity_l3737_373744

/-- Given a hyperbola with the following properties:
    - Equation: x²/a² - y²/b² = 1
    - a > 0, b > 0
    - Focal distance is 8
    - Left vertex A is at (-a, 0)
    - Point B is at (0, b)
    - Right focus F is at (4, 0)
    - Dot product of BA and BF equals 2a
    The eccentricity of the hyperbola is 2 -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let A : ℝ × ℝ := (-a, 0)
  let B : ℝ × ℝ := (0, b)
  let F : ℝ × ℝ := (4, 0)
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 → 
    (x - (-a))^2 + y^2 = (x - 4)^2 + y^2) →
  (B.1 - A.1) * (F.1 - B.1) + (B.2 - A.2) * (F.2 - B.2) = 2 * a →
  4 / a = 2 := by
sorry

end hyperbola_eccentricity_l3737_373744


namespace distribution_of_slots_l3737_373729

theorem distribution_of_slots (n : ℕ) (k : ℕ) :
  n = 6 →
  k = 3 →
  (Nat.choose (n - 1) (k - 1) : ℕ) = 10 :=
by sorry

end distribution_of_slots_l3737_373729


namespace solution_set_correct_l3737_373780

/-- The solution set of the system of equations y² = x and y = x -/
def solution_set : Set (ℝ × ℝ) := {(1, 1), (0, 0)}

/-- The system of equations y² = x and y = x -/
def system_equations (p : ℝ × ℝ) : Prop :=
  p.2 ^ 2 = p.1 ∧ p.2 = p.1

theorem solution_set_correct :
  ∀ p : ℝ × ℝ, p ∈ solution_set ↔ system_equations p := by
  sorry

end solution_set_correct_l3737_373780


namespace total_lost_or_given_equals_sum_l3737_373756

/-- Represents the number of crayons in various states --/
structure CrayonCounts where
  given_to_friends : ℕ
  lost : ℕ
  total_lost_or_given : ℕ

/-- Theorem stating that the total number of crayons lost or given away
    is equal to the sum of crayons given to friends and crayons lost --/
theorem total_lost_or_given_equals_sum (c : CrayonCounts)
  (h1 : c.given_to_friends = 52)
  (h2 : c.lost = 535)
  (h3 : c.total_lost_or_given = 587) :
  c.total_lost_or_given = c.given_to_friends + c.lost := by
  sorry

#check total_lost_or_given_equals_sum

end total_lost_or_given_equals_sum_l3737_373756


namespace subtract_negative_six_a_l3737_373719

theorem subtract_negative_six_a (a : ℝ) : (4 * a^2 - 3 * a + 7) - (-6 * a) = 4 * a^2 - 9 * a + 7 := by
  sorry

end subtract_negative_six_a_l3737_373719


namespace original_milk_cost_is_three_l3737_373732

/-- The original cost of a gallon of whole milk -/
def original_milk_cost : ℝ := 3

/-- The current price of a gallon of whole milk -/
def current_milk_price : ℝ := 2

/-- The discount on a box of cereal -/
def cereal_discount : ℝ := 1

/-- The total savings from buying 3 gallons of milk and 5 boxes of cereal -/
def total_savings : ℝ := 8

/-- Theorem stating that the original cost of a gallon of whole milk is $3 -/
theorem original_milk_cost_is_three :
  original_milk_cost = 3 ∧
  current_milk_price = 2 ∧
  cereal_discount = 1 ∧
  total_savings = 8 ∧
  3 * (original_milk_cost - current_milk_price) + 5 * cereal_discount = total_savings :=
by sorry

end original_milk_cost_is_three_l3737_373732


namespace surface_area_is_39_l3737_373724

/-- Represents the structure made of unit cubes -/
structure CubeStructure where
  total_cubes : Nat
  pyramid_base : Nat
  extension_height : Nat

/-- Calculates the exposed surface area of the cube structure -/
def exposed_surface_area (s : CubeStructure) : Nat :=
  sorry

/-- The theorem stating that the exposed surface area of the given structure is 39 square meters -/
theorem surface_area_is_39 (s : CubeStructure) 
  (h1 : s.total_cubes = 18)
  (h2 : s.pyramid_base = 3)
  (h3 : s.extension_height = 4) : 
  exposed_surface_area s = 39 :=
sorry

end surface_area_is_39_l3737_373724


namespace rhombus_side_length_l3737_373707

/-- A rhombus with diagonals in ratio 1:2 and shorter diagonal 4 cm has side length 2√5 cm -/
theorem rhombus_side_length (d1 d2 side : ℝ) : 
  d1 > 0 → -- shorter diagonal is positive
  d2 = 2 * d1 → -- ratio of diagonals is 1:2
  d1 = 4 → -- shorter diagonal is 4 cm
  side^2 = (d1/2)^2 + (d2/2)^2 → -- Pythagorean theorem for half-diagonals
  side = 2 * Real.sqrt 5 := by
  sorry

end rhombus_side_length_l3737_373707


namespace area_under_curve_l3737_373703

-- Define the curve
def f (x : ℝ) : ℝ := x^3

-- Define the bounds
def a : ℝ := 0
def b : ℝ := 2

-- State the theorem
theorem area_under_curve : 
  (∫ x in a..b, f x) = 4 := by sorry

end area_under_curve_l3737_373703


namespace quarters_found_l3737_373710

def dime_value : ℚ := 0.1
def nickel_value : ℚ := 0.05
def penny_value : ℚ := 0.01
def quarter_value : ℚ := 0.25

def num_dimes : ℕ := 3
def num_nickels : ℕ := 4
def num_pennies : ℕ := 200
def total_amount : ℚ := 5

theorem quarters_found :
  ∃ (num_quarters : ℕ),
    (num_quarters : ℚ) * quarter_value +
    (num_dimes : ℚ) * dime_value +
    (num_nickels : ℚ) * nickel_value +
    (num_pennies : ℚ) * penny_value = total_amount ∧
    num_quarters = 10 :=
by sorry

end quarters_found_l3737_373710


namespace abs_diff_lt_abs_one_minus_prod_l3737_373794

theorem abs_diff_lt_abs_one_minus_prod {x y : ℝ} (hx : |x| < 1) (hy : |y| < 1) :
  |x - y| < |1 - x * y| := by
  sorry

end abs_diff_lt_abs_one_minus_prod_l3737_373794

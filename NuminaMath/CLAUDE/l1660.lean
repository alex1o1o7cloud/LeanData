import Mathlib

namespace cubic_equation_solution_l1660_166003

theorem cubic_equation_solution (x : ℝ) (h : x^3 + 1/x^3 = -52) : x + 1/x = -4 := by
  sorry

end cubic_equation_solution_l1660_166003


namespace fraction_to_decimal_l1660_166066

theorem fraction_to_decimal : (47 : ℚ) / (2^3 * 5^4) = 0.5875 := by
  sorry

end fraction_to_decimal_l1660_166066


namespace product_of_square_roots_l1660_166033

theorem product_of_square_roots (x y z : ℝ) :
  x = 75 → y = 48 → z = 12 → Real.sqrt x * Real.sqrt y * Real.sqrt z = 120 * Real.sqrt 3 := by
  sorry

end product_of_square_roots_l1660_166033


namespace expression_simplification_l1660_166024

theorem expression_simplification (a b : ℤ) (h : b = a + 1) (ha : a = 2015) :
  (a^4 - 3*a^3*b + 3*a^2*b^2 - b^4 + a) / (a*b) = -(a-1)^2 / a^3 := by
  sorry

end expression_simplification_l1660_166024


namespace boats_left_l1660_166040

def total_boats : ℕ := 30
def fish_eaten_percentage : ℚ := 1/5
def boats_shot : ℕ := 2

theorem boats_left : 
  total_boats - (total_boats * fish_eaten_percentage).floor - boats_shot = 22 := by
sorry

end boats_left_l1660_166040


namespace protective_clothing_equation_l1660_166046

/-- Represents the equation for the protective clothing production problem -/
theorem protective_clothing_equation (x : ℝ) (h : x > 0) :
  let total_sets := 1000
  let increase_rate := 0.2
  let days_ahead := 2
  let original_days := total_sets / x
  let actual_days := total_sets / (x * (1 + increase_rate))
  original_days - actual_days = days_ahead :=
by sorry

end protective_clothing_equation_l1660_166046


namespace inequality_solution_and_sqrt2_l1660_166013

-- Define the inequality
def inequality (x : ℝ) : Prop := (5/2 * x - 1) > 3 * x

-- Define the solution set
def solution_set : Set ℝ := {x | x < -2}

-- Theorem statement
theorem inequality_solution_and_sqrt2 :
  (∀ x, inequality x ↔ x ∈ solution_set) ∧
  ¬ inequality (-Real.sqrt 2) := by
  sorry

end inequality_solution_and_sqrt2_l1660_166013


namespace music_tool_cost_l1660_166082

/-- Calculates the cost of a music tool given the total spent and costs of other items --/
theorem music_tool_cost (total_spent flute_cost songbook_cost : ℚ) :
  total_spent = 158.35 ∧ flute_cost = 142.46 ∧ songbook_cost = 7 →
  total_spent - (flute_cost + songbook_cost) = 8.89 := by
sorry

end music_tool_cost_l1660_166082


namespace inequality_solution_condition_l1660_166099

theorem inequality_solution_condition (a : ℝ) :
  (∃ x : ℝ, x ≥ a ∧ |x - a| + |2*x + 1| ≤ 2*a + x) ↔ a ≥ 1 := by
  sorry

end inequality_solution_condition_l1660_166099


namespace min_value_trigonometric_expression_l1660_166012

theorem min_value_trigonometric_expression (x₁ x₂ x₃ x₄ : ℝ) 
  (h_positive : x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₄ > 0) 
  (h_sum : x₁ + x₂ + x₃ + x₄ = π) : 
  (2 * Real.sin x₁ ^ 2 + 1 / Real.sin x₁ ^ 2) * 
  (2 * Real.sin x₂ ^ 2 + 1 / Real.sin x₂ ^ 2) * 
  (2 * Real.sin x₃ ^ 2 + 1 / Real.sin x₃ ^ 2) * 
  (2 * Real.sin x₄ ^ 2 + 1 / Real.sin x₄ ^ 2) ≥ 81 :=
by sorry

end min_value_trigonometric_expression_l1660_166012


namespace collinear_vectors_m_value_l1660_166025

/-- Two vectors are collinear in opposite directions if one is a negative scalar multiple of the other -/
def collinear_opposite (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k < 0 ∧ a.1 = k * b.1 ∧ a.2 = k * b.2

theorem collinear_vectors_m_value (m : ℝ) :
  let a : ℝ × ℝ := (m, -4)
  let b : ℝ × ℝ := (-1, m + 3)
  collinear_opposite a b → m = 1 := by
sorry

end collinear_vectors_m_value_l1660_166025


namespace triangle_sine_inequality_l1660_166021

theorem triangle_sine_inequality (A B C : Real) : 
  A > 0 → B > 0 → C > 0 → A + B + C = π →
  1 / Real.sin (A / 2) + 1 / Real.sin (B / 2) + 1 / Real.sin (C / 2) ≥ 6 := by
sorry

end triangle_sine_inequality_l1660_166021


namespace positive_difference_problem_l1660_166067

def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0))

theorem positive_difference_problem (x : ℕ) 
  (h1 : (45 + x) / 2 = 50) 
  (h2 : is_prime x) : 
  Int.natAbs (x - 45) = 8 := by
sorry

end positive_difference_problem_l1660_166067


namespace root_value_theorem_l1660_166077

theorem root_value_theorem (m : ℝ) (h : 2 * m^2 - 7 * m + 1 = 0) :
  m * (2 * m - 7) + 5 = 4 := by
  sorry

end root_value_theorem_l1660_166077


namespace no_positive_integer_solution_l1660_166035

theorem no_positive_integer_solution :
  ¬ ∃ (a b c d : ℕ+), (a^2 + b^2 = c^2 - d^2) ∧ (a * b = c * d) := by
  sorry

end no_positive_integer_solution_l1660_166035


namespace cole_fence_cost_is_225_l1660_166087

/-- Calculates the total cost for Cole's fence installation given the backyard dimensions,
    fencing costs, and neighbor contributions. -/
def cole_fence_cost (side_length : ℝ) (back_length : ℝ) (side_cost : ℝ) (back_cost : ℝ)
                    (back_neighbor_contribution : ℝ) (left_neighbor_contribution : ℝ)
                    (installation_fee : ℝ) : ℝ :=
  let total_fencing_cost := 2 * side_length * side_cost + back_length * back_cost
  let neighbor_contributions := back_neighbor_contribution + left_neighbor_contribution
  total_fencing_cost - neighbor_contributions + installation_fee

theorem cole_fence_cost_is_225 :
  cole_fence_cost 15 30 4 5 75 20 50 = 225 := by
  sorry

end cole_fence_cost_is_225_l1660_166087


namespace sqrt_difference_equality_l1660_166075

theorem sqrt_difference_equality : Real.sqrt (49 + 121) - Real.sqrt (36 - 9) = Real.sqrt 170 - 3 * Real.sqrt 3 := by
  sorry

end sqrt_difference_equality_l1660_166075


namespace simple_interest_rate_l1660_166051

/-- Calculates the simple interest rate given loan amounts, durations, and total interest received. -/
theorem simple_interest_rate 
  (loan_b loan_c : ℕ) 
  (duration_b duration_c : ℕ) 
  (total_interest : ℕ) : 
  loan_b = 5000 → 
  loan_c = 3000 → 
  duration_b = 2 → 
  duration_c = 4 → 
  total_interest = 1540 → 
  ∃ (rate : ℚ), 
    rate = 7 ∧ 
    (loan_b * duration_b * rate + loan_c * duration_c * rate) / 100 = total_interest :=
by sorry

end simple_interest_rate_l1660_166051


namespace ellipse_equation_1_ellipse_equation_2_l1660_166090

-- Define the ellipse type
structure Ellipse where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the standard equation of an ellipse
def standard_equation (e : Ellipse) (x y : ℝ) : Prop :=
  y^2 / e.a^2 + x^2 / e.b^2 = 1

-- Define the focal length
def focal_length (e : Ellipse) : ℝ := 2 * e.c

-- Define the sum of distances from a point on the ellipse to the two focal points
def sum_of_distances (e : Ellipse) : ℝ := 2 * e.a

-- Theorem 1
theorem ellipse_equation_1 :
  ∃ (e : Ellipse),
    focal_length e = 4 ∧
    standard_equation e 3 2 ∧
    e.a = 4 ∧ e.b^2 = 12 :=
sorry

-- Theorem 2
theorem ellipse_equation_2 :
  ∃ (e : Ellipse),
    focal_length e = 10 ∧
    sum_of_distances e = 26 ∧
    ((e.a = 13 ∧ e.b = 12) ∨ (e.a = 12 ∧ e.b = 13)) :=
sorry

end ellipse_equation_1_ellipse_equation_2_l1660_166090


namespace total_carrots_l1660_166085

theorem total_carrots (sally_carrots fred_carrots : ℕ) 
  (h1 : sally_carrots = 6) 
  (h2 : fred_carrots = 4) : 
  sally_carrots + fred_carrots = 10 := by
sorry

end total_carrots_l1660_166085


namespace clock_centers_distance_l1660_166019

/-- Two identically accurate clocks with hour hands -/
structure Clock where
  center : ℝ × ℝ
  hand_length : ℝ

/-- The configuration of two clocks -/
structure ClockPair where
  clock1 : Clock
  clock2 : Clock
  m : ℝ  -- Minimum distance between hour hand ends
  M : ℝ  -- Maximum distance between hour hand ends

/-- The theorem stating the distance between clock centers -/
theorem clock_centers_distance (cp : ClockPair) :
  let (x1, y1) := cp.clock1.center
  let (x2, y2) := cp.clock2.center
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2) = (cp.M + cp.m) / 2 := by
  sorry

end clock_centers_distance_l1660_166019


namespace intersection_range_l1660_166045

-- Define the semicircle
def semicircle (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 2)^2 = 4 ∧ y ≥ 2

-- Define the line
def line (x y k : ℝ) : Prop :=
  y = k * (x - 1) + 5

-- Define the condition for two distinct intersection points
def has_two_intersections (k : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), x₁ ≠ x₂ ∧ 
    semicircle x₁ y₁ ∧ semicircle x₂ y₂ ∧
    line x₁ y₁ k ∧ line x₂ y₂ k

-- Theorem statement
theorem intersection_range :
  ∀ k : ℝ, has_two_intersections k ↔ 
    (k ∈ Set.Icc (-3/2) (-Real.sqrt 5/2) ∨ 
     k ∈ Set.Ioc (Real.sqrt 5/2) (3/2)) :=
sorry

end intersection_range_l1660_166045


namespace reverse_digit_integers_l1660_166098

theorem reverse_digit_integers (q r : ℕ) : 
  (q ≥ 10 ∧ q < 100) →  -- q is a two-digit positive integer
  (r ≥ 10 ∧ r < 100) →  -- r is a two-digit positive integer
  (q.div 10 = r.mod 10 ∧ q.mod 10 = r.div 10) →  -- q and r have the same digits in reverse order
  (q > r → q - r < 20) →  -- positive difference is less than 20
  (r > q → r - q < 20) →  -- positive difference is less than 20
  (∀ a b : ℕ, (a ≥ 10 ∧ a < 100) → (b ≥ 10 ∧ b < 100) → 
    (a.div 10 = b.mod 10 ∧ a.mod 10 = b.div 10) → (a - b ≤ 18)) →  -- greatest possible difference is 18
  (q.div 10 = q.mod 10 + 2) →  -- tens digit is 2 more than units digit for q
  (r.div 10 + 2 = r.mod 10) -- tens digit is 2 more than units digit for r (reverse of q)
  := by sorry

end reverse_digit_integers_l1660_166098


namespace class_savings_l1660_166073

/-- Calculates the total amount saved by a class for a field trip over a given period. -/
theorem class_savings (num_students : ℕ) (contribution : ℕ) (num_weeks : ℕ) :
  num_students = 30 →
  contribution = 2 →
  num_weeks = 8 →
  num_students * contribution * num_weeks = 480 := by
  sorry

#check class_savings

end class_savings_l1660_166073


namespace imaginary_part_of_z_l1660_166053

theorem imaginary_part_of_z (z : ℂ) : z = (1 - I) / (1 + 3*I) → z.im = -2/5 := by
  sorry

end imaginary_part_of_z_l1660_166053


namespace cube_root_of_110592_l1660_166091

theorem cube_root_of_110592 :
  ∃ (n : ℕ), n^3 = 110592 ∧ n = 48 :=
by
  -- Define the number
  let number : ℕ := 110592

  -- Define the conditions
  have h1 : 10^3 = 1000 := by sorry
  have h2 : 100^3 = 1000000 := by sorry
  have h3 : 1000 < number ∧ number < 1000000 := by sorry
  have h4 : number % 10 = 2 := by sorry
  have h5 : ∀ (m : ℕ), m % 10 = 8 → (m^3) % 10 = 2 := by sorry
  have h6 : 4^3 = 64 := by sorry
  have h7 : 5^3 = 125 := by sorry
  have h8 : 64 < 110 ∧ 110 < 125 := by sorry

  -- Prove the theorem
  sorry

end cube_root_of_110592_l1660_166091


namespace third_largest_number_l1660_166022

/-- Given five numbers in a specific ratio with a known product, 
    this theorem proves the value of the third largest number. -/
theorem third_largest_number 
  (a b c d e : ℝ) 
  (ratio : a / 2.3 = b / 3.7 ∧ a / 2.3 = c / 5.5 ∧ a / 2.3 = d / 7.1 ∧ a / 2.3 = e / 8.9) 
  (product : a * b * c * d * e = 900000) : 
  ∃ (ε : ℝ), abs (c - 14.85) < ε ∧ ε > 0 := by
  sorry

end third_largest_number_l1660_166022


namespace roots_equation_value_l1660_166097

theorem roots_equation_value (α β : ℝ) : 
  α^2 - α - 1 = 0 → β^2 - β - 1 = 0 → α^4 + 3*β = 5 := by sorry

end roots_equation_value_l1660_166097


namespace expression_nonnegative_l1660_166027

theorem expression_nonnegative (a b c d e : ℝ) : 
  (a-b)*(a-c)*(a-d)*(a-e) + (b-a)*(b-c)*(b-d)*(b-e) + (c-a)*(c-b)*(c-d)*(c-e) +
  (d-a)*(d-b)*(d-c)*(d-e) + (e-a)*(e-b)*(e-c)*(e-d) ≥ 0 := by
  sorry

end expression_nonnegative_l1660_166027


namespace product_divisible_by_1419_l1660_166038

theorem product_divisible_by_1419 : ∃ k : ℕ, 86 * 87 * 88 = 1419 * k := by
  sorry

end product_divisible_by_1419_l1660_166038


namespace one_sixth_of_x_l1660_166074

theorem one_sixth_of_x (x : ℝ) (h : x / 3 = 4) : x / 6 = 2 := by
  sorry

end one_sixth_of_x_l1660_166074


namespace trains_crossing_time_l1660_166089

/-- The time taken for two trains to cross each other -/
theorem trains_crossing_time (length1 length2 speed1 speed2 : ℝ) 
  (h1 : length1 = 200) 
  (h2 : length2 = 160) 
  (h3 : speed1 = 68 * 1000 / 3600) 
  (h4 : speed2 = 40 * 1000 / 3600) : 
  (length1 + length2) / (speed1 + speed2) = 12 :=
by sorry

end trains_crossing_time_l1660_166089


namespace right_triangle_with_60_degree_angle_l1660_166043

theorem right_triangle_with_60_degree_angle (α β : ℝ) : 
  α = 60 → -- One acute angle is 60°
  α + β + 90 = 180 → -- Sum of angles in a triangle is 180°
  β = 30 := by -- The other acute angle is 30°
sorry

end right_triangle_with_60_degree_angle_l1660_166043


namespace floor_equation_solution_l1660_166056

theorem floor_equation_solution (x : ℝ) : 
  (⌊⌊3 * x⌋ + 1/2⌋ = ⌊x + 4⌋) ↔ (5/3 ≤ x ∧ x < 7/3) :=
sorry

end floor_equation_solution_l1660_166056


namespace remainder_928927_div_6_l1660_166044

theorem remainder_928927_div_6 : 928927 % 6 = 1 := by
  sorry

end remainder_928927_div_6_l1660_166044


namespace orange_harvest_after_six_days_l1660_166079

/-- The number of sacks of oranges harvested after a given number of days. -/
def oranges_harvested (daily_rate : ℕ) (days : ℕ) : ℕ :=
  daily_rate * days

/-- Theorem stating that given a daily harvest rate of 83 sacks per day,
    the total number of sacks harvested after 6 days is equal to 498. -/
theorem orange_harvest_after_six_days :
  oranges_harvested 83 6 = 498 := by
  sorry

end orange_harvest_after_six_days_l1660_166079


namespace bake_sale_profit_split_l1660_166006

/-- The number of dozens of cookies John makes -/
def dozens : ℕ := 6

/-- The number of cookies in a dozen -/
def cookies_per_dozen : ℕ := 12

/-- The selling price of each cookie in dollars -/
def selling_price : ℚ := 3/2

/-- The cost to make each cookie in dollars -/
def cost_per_cookie : ℚ := 1/4

/-- The amount each charity receives in dollars -/
def charity_amount : ℚ := 45

/-- The number of charities John splits the profit between -/
def num_charities : ℕ := 2

theorem bake_sale_profit_split :
  (dozens * cookies_per_dozen * selling_price - dozens * cookies_per_dozen * cost_per_cookie) / charity_amount = num_charities := by
  sorry

end bake_sale_profit_split_l1660_166006


namespace comparison_of_expressions_l1660_166059

theorem comparison_of_expressions (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a^2 / b + b^2 / a ≥ a + b := by
  sorry

end comparison_of_expressions_l1660_166059


namespace min_max_quadratic_form_l1660_166080

theorem min_max_quadratic_form (x y : ℝ) (h : 9*x^2 + 12*x*y + 4*y^2 = 1) :
  let f := fun (x y : ℝ) => 3*x^2 + 4*x*y + 2*y^2
  ∃ (m M : ℝ), (∀ a b : ℝ, m ≤ f a b ∧ f a b ≤ M) ∧ m = 1 ∧ M = 1 := by
  sorry

end min_max_quadratic_form_l1660_166080


namespace tobys_money_sharing_l1660_166076

theorem tobys_money_sharing (initial_amount : ℚ) (brothers : ℕ) (share_fraction : ℚ) :
  initial_amount = 343 →
  brothers = 2 →
  share_fraction = 1/7 →
  initial_amount - (brothers * (share_fraction * initial_amount)) = 245 := by
  sorry

end tobys_money_sharing_l1660_166076


namespace triangle_perimeter_l1660_166050

theorem triangle_perimeter (a b c A B C : ℝ) : 
  (c * Real.cos B + b * Real.cos C = 2 * a * Real.cos A) →
  (a = 2) →
  (1/2 * b * c * Real.sin A = Real.sqrt 3) →
  (a + b + c = 6) := by
sorry

end triangle_perimeter_l1660_166050


namespace ice_cube_distribution_l1660_166060

/-- Given a total of 30 ice cubes and 6 cups, prove that each cup should contain 5 ice cubes when divided equally. -/
theorem ice_cube_distribution (total_ice_cubes : ℕ) (num_cups : ℕ) (ice_per_cup : ℕ) :
  total_ice_cubes = 30 →
  num_cups = 6 →
  ice_per_cup = total_ice_cubes / num_cups →
  ice_per_cup = 5 := by
  sorry

end ice_cube_distribution_l1660_166060


namespace red_hair_count_example_l1660_166020

/-- Given a class with a hair color ratio and total number of students,
    calculate the number of students with red hair. -/
def red_hair_count (red blonde black total : ℕ) : ℕ :=
  (red * total) / (red + blonde + black)

/-- Theorem: In a class of 48 students with a hair color ratio of 3 : 6 : 7
    (red : blonde : black), the number of students with red hair is 9. -/
theorem red_hair_count_example : red_hair_count 3 6 7 48 = 9 := by
  sorry

end red_hair_count_example_l1660_166020


namespace right_triangle_hypotenuse_l1660_166004

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 6 → b = 8 → c^2 = a^2 + b^2 → c = 10 := by
  sorry

end right_triangle_hypotenuse_l1660_166004


namespace tangent_sum_l1660_166088

theorem tangent_sum (x y : ℝ) 
  (h1 : (Real.sin x / Real.cos y) + (Real.sin y / Real.cos x) = 1)
  (h2 : (Real.cos x / Real.sin y) + (Real.cos y / Real.sin x) = 6) :
  (Real.tan x / Real.tan y) + (Real.tan y / Real.tan x) = 124/13 := by
  sorry

end tangent_sum_l1660_166088


namespace three_tribes_at_campfire_l1660_166096

/-- Represents a native at the campfire -/
structure Native where
  tribe : ℕ

/-- Represents the circle of natives around the campfire -/
def Campfire := Vector Native 7

/-- Check if a native tells the truth to their left neighbor -/
def tellsTruth (c : Campfire) (i : Fin 7) : Prop :=
  (c.get i).tribe = (c.get ((i + 1) % 7)).tribe →
    (∀ j : Fin 7, j ≠ i ∧ j ≠ ((i + 1) % 7) → (c.get j).tribe ≠ (c.get i).tribe)

/-- The main theorem: there are exactly 3 tribes represented at the campfire -/
theorem three_tribes_at_campfire (c : Campfire) 
  (h : ∀ i : Fin 7, tellsTruth c i) :
  ∃! n : ℕ, n = 3 ∧ (∀ t : ℕ, (∃ i : Fin 7, (c.get i).tribe = t) → t ≤ n) :=
sorry

end three_tribes_at_campfire_l1660_166096


namespace min_value_reciprocal_sum_l1660_166023

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = 2) :
  1/x + 2/y ≥ 9/2 := by
  sorry

end min_value_reciprocal_sum_l1660_166023


namespace sum_of_last_two_digits_of_8_pow_2003_l1660_166055

theorem sum_of_last_two_digits_of_8_pow_2003 : 
  ∃ (n : ℕ), 8^2003 ≡ n [ZMOD 100] ∧ (n / 10 % 10 + n % 10 = 5) :=
sorry

end sum_of_last_two_digits_of_8_pow_2003_l1660_166055


namespace angle_difference_range_l1660_166036

theorem angle_difference_range (α β : Real) (h1 : -π < α) (h2 : α < β) (h3 : β < π) :
  -2*π < α - β ∧ α - β < 0 :=
by sorry

end angle_difference_range_l1660_166036


namespace parallel_vector_proof_l1660_166095

/-- Given a planar vector b parallel to a = (2, 1) with magnitude 2√5, prove b is either (4, 2) or (-4, -2) -/
theorem parallel_vector_proof (b : ℝ × ℝ) : 
  (∃ k : ℝ, b = (2*k, k)) → -- b is parallel to (2, 1)
  (b.1^2 + b.2^2 = 20) →    -- |b| = 2√5
  (b = (4, 2) ∨ b = (-4, -2)) := by
sorry

end parallel_vector_proof_l1660_166095


namespace lines_perpendicular_to_plane_are_parallel_l1660_166049

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- Define the theorem
theorem lines_perpendicular_to_plane_are_parallel
  (m n : Line) (α : Plane)
  (h1 : m ≠ n)
  (h2 : perpendicular m α)
  (h3 : perpendicular n α) :
  parallel m n :=
sorry

end lines_perpendicular_to_plane_are_parallel_l1660_166049


namespace horse_track_distance_l1660_166058

/-- The distance covered by a horse running one turn around a square-shaped track -/
def track_distance (side_length : ℝ) : ℝ := 4 * side_length

/-- Theorem: The distance covered by a horse running one turn around a square-shaped track
    with sides of length 40 meters is equal to 160 meters -/
theorem horse_track_distance :
  track_distance 40 = 160 := by
  sorry

end horse_track_distance_l1660_166058


namespace cone_unfolded_side_view_is_sector_l1660_166030

/-- A shape with one curved side and two straight sides -/
structure ConeUnfoldedSideView where
  curved_side : ℕ
  straight_sides : ℕ
  h_curved : curved_side = 1
  h_straight : straight_sides = 2

/-- Definition of a sector -/
def is_sector (shape : ConeUnfoldedSideView) : Prop :=
  shape.curved_side = 1 ∧ shape.straight_sides = 2

/-- Theorem: The unfolded side view of a cone is a sector -/
theorem cone_unfolded_side_view_is_sector (shape : ConeUnfoldedSideView) :
  is_sector shape :=
by sorry

end cone_unfolded_side_view_is_sector_l1660_166030


namespace quadratic_maximum_l1660_166009

theorem quadratic_maximum (r : ℝ) : 
  -7 * r^2 + 50 * r - 20 ≤ 5 ∧ ∃ r, -7 * r^2 + 50 * r - 20 = 5 :=
by sorry

end quadratic_maximum_l1660_166009


namespace sqrt_difference_equals_seven_twelfths_l1660_166037

theorem sqrt_difference_equals_seven_twelfths :
  Real.sqrt (16 / 9) - Real.sqrt (9 / 16) = 7 / 12 := by
  sorry

end sqrt_difference_equals_seven_twelfths_l1660_166037


namespace min_value_theorem_l1660_166084

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  b / a + 1 / b ≥ 3 ∧ (b / a + 1 / b = 3 ↔ a = 1 / 2) := by
  sorry

end min_value_theorem_l1660_166084


namespace at_operation_example_l1660_166081

def at_operation (x y : ℤ) : ℤ := x * y - 2 * x + 3 * y

theorem at_operation_example : (at_operation 8 5) - (at_operation 5 8) = -15 := by
  sorry

end at_operation_example_l1660_166081


namespace roberto_outfits_l1660_166029

/-- The number of different outfits Roberto can create -/
def number_of_outfits (trousers shirts jackets hats : ℕ) : ℕ :=
  trousers * shirts * jackets * hats

/-- Theorem stating the number of outfits Roberto can create -/
theorem roberto_outfits :
  number_of_outfits 5 8 4 2 = 320 := by
  sorry

end roberto_outfits_l1660_166029


namespace seated_students_count_l1660_166061

/-- Given a school meeting with teachers and students, calculate the number of seated students. -/
theorem seated_students_count 
  (total_attendees : ℕ) 
  (seated_teachers : ℕ) 
  (standing_students : ℕ) 
  (h1 : total_attendees = 355) 
  (h2 : seated_teachers = 30) 
  (h3 : standing_students = 25) : 
  total_attendees = seated_teachers + standing_students + 300 :=
by sorry

end seated_students_count_l1660_166061


namespace at_least_one_quadratic_has_root_l1660_166057

theorem at_least_one_quadratic_has_root (a b c d : ℝ) (h : a * c = 2 * b + 2 * d) :
  (a^2 - 4*b ≥ 0) ∨ (c^2 - 4*d ≥ 0) := by sorry

end at_least_one_quadratic_has_root_l1660_166057


namespace triangle_angle_proof_l1660_166069

theorem triangle_angle_proof (A B C : ℝ) (m n : ℝ × ℝ) :
  A + B + C = π →
  m = (Real.sqrt 3 * Real.sin A, Real.sin B) →
  n = (Real.cos B, Real.sqrt 3 * Real.cos A) →
  m.1 * n.1 + m.2 * n.2 = 1 + Real.cos (A + B) →
  C = 2 * π / 3 := by
  sorry

end triangle_angle_proof_l1660_166069


namespace benny_piggy_bank_l1660_166064

theorem benny_piggy_bank (january_amount february_amount total_amount : ℕ) 
  (h1 : january_amount = 19)
  (h2 : february_amount = january_amount)
  (h3 : total_amount = 46) : 
  total_amount - (january_amount + february_amount) = 8 := by
  sorry

end benny_piggy_bank_l1660_166064


namespace motorboat_travel_theorem_l1660_166070

noncomputable def motorboat_travel_fraction (S : ℝ) (v : ℝ) : Set ℝ :=
  let u₁ := (2 / 3) * v
  let u₂ := (1 / 3) * v
  let V_m₁ := 2 * v + u₁
  let V_m₂ := 2 * v + u₂
  let V_b₁ := 3 * v - u₁
  let V_b₂ := 3 * v - u₂
  let t₁ := S / (5 * v)
  let d := (56 / 225) * S
  { x | x = (V_m₁ * t₁ + d) / S ∨ x = (V_m₂ * t₁ + d) / S }

theorem motorboat_travel_theorem (S : ℝ) (v : ℝ) (h_S : S > 0) (h_v : v > 0) :
  motorboat_travel_fraction S v = {161 / 225, 176 / 225} := by
  sorry

end motorboat_travel_theorem_l1660_166070


namespace weight_of_b_l1660_166007

theorem weight_of_b (a b c : ℝ) 
  (h1 : (a + b + c) / 3 = 43) 
  (h2 : (a + b) / 2 = 40) 
  (h3 : (b + c) / 2 = 43) : 
  b = 37 := by
  sorry

end weight_of_b_l1660_166007


namespace inscribed_circle_radius_when_area_equals_perimeter_l1660_166034

/-- A triangle with an inscribed circle -/
structure Triangle :=
  (area : ℝ)
  (perimeter : ℝ)
  (inradius : ℝ)

/-- The theorem stating that if a triangle's area equals its perimeter, 
    then the radius of its inscribed circle is 2 -/
theorem inscribed_circle_radius_when_area_equals_perimeter 
  (t : Triangle) 
  (h : t.area = t.perimeter) : 
  t.inradius = 2 :=
sorry

end inscribed_circle_radius_when_area_equals_perimeter_l1660_166034


namespace no_very_convex_function_l1660_166078

theorem no_very_convex_function :
  ∀ f : ℝ → ℝ, ∃ x y : ℝ, (f x + f y) / 2 < f ((x + y) / 2) + |x - y| :=
by sorry

end no_very_convex_function_l1660_166078


namespace add_fractions_l1660_166026

theorem add_fractions : (2 : ℚ) / 5 + (3 : ℚ) / 8 = (31 : ℚ) / 40 := by
  sorry

end add_fractions_l1660_166026


namespace expected_imbalance_six_teams_l1660_166042

/-- Represents a baseball league with n teams -/
structure BaseballLeague (n : ℕ) where
  teams : Fin n → Unit

/-- Represents the schedule of games in the league -/
def Schedule (n : ℕ) := Fin n → Fin n → Bool

/-- Calculates the imbalance (minimum number of undefeated teams) for a given schedule -/
def imbalance (n : ℕ) (schedule : Schedule n) : ℕ := sorry

/-- The expected value of the imbalance for a league with n teams -/
def expectedImbalance (n : ℕ) : ℚ := sorry

/-- Theorem: The expected imbalance for a 6-team league is 5055 / 2^15 -/
theorem expected_imbalance_six_teams :
  expectedImbalance 6 = 5055 / 2^15 := by sorry

end expected_imbalance_six_teams_l1660_166042


namespace inequality_proof_l1660_166094

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum_squares : a^2 + b^2 + c^2 = 1) : 
  (a^5 + b^5)/(a*b*(a+b)) + (b^5 + c^5)/(b*c*(b+c)) + (c^5 + a^5)/(c*a*(c+a)) ≥ 3*(a*b + b*c + c*a) - 2 ∧
  (a^5 + b^5)/(a*b*(a+b)) + (b^5 + c^5)/(b*c*(b+c)) + (c^5 + a^5)/(c*a*(c+a)) ≥ 6 - 5*(a*b + b*c + c*a) :=
by sorry

end inequality_proof_l1660_166094


namespace equation_one_integral_root_l1660_166071

theorem equation_one_integral_root :
  ∃! x : ℤ, x - 9 / (x - 2) = 5 - 9 / (x - 2) :=
by
  sorry

end equation_one_integral_root_l1660_166071


namespace blueberries_count_l1660_166093

/-- Represents the number of blueberries in each blue box -/
def blueberries : ℕ := sorry

/-- Represents the number of strawberries in each red box -/
def strawberries : ℕ := sorry

/-- The increase in total berries when replacing a blue box with a red box -/
def berry_increase : ℕ := 10

/-- The increase in the difference between strawberries and blueberries when replacing a blue box with a red box -/
def difference_increase : ℕ := 50

theorem blueberries_count : 
  (strawberries - blueberries = berry_increase) ∧ 
  (strawberries = difference_increase) → 
  blueberries = 40 := by sorry

end blueberries_count_l1660_166093


namespace gem_stone_necklaces_count_l1660_166031

/-- The number of gem stone necklaces sold by Faye -/
def gem_stone_necklaces : ℕ := 7

/-- The number of bead necklaces sold by Faye -/
def bead_necklaces : ℕ := 3

/-- The price of each necklace in dollars -/
def necklace_price : ℕ := 7

/-- The total earnings from all necklaces in dollars -/
def total_earnings : ℕ := 70

/-- Theorem stating that the number of gem stone necklaces sold is 7 -/
theorem gem_stone_necklaces_count :
  gem_stone_necklaces = (total_earnings - bead_necklaces * necklace_price) / necklace_price :=
by sorry

end gem_stone_necklaces_count_l1660_166031


namespace backpacks_sold_to_dept_store_l1660_166008

def total_backpacks : ℕ := 48
def total_cost : ℕ := 576
def swap_meet_sold : ℕ := 17
def swap_meet_price : ℕ := 18
def dept_store_price : ℕ := 25
def remainder_price : ℕ := 22
def total_profit : ℕ := 442

theorem backpacks_sold_to_dept_store :
  ∃ x : ℕ, 
    x * dept_store_price + 
    swap_meet_sold * swap_meet_price + 
    (total_backpacks - swap_meet_sold - x) * remainder_price - 
    total_cost = total_profit ∧
    x = 10 := by
  sorry

end backpacks_sold_to_dept_store_l1660_166008


namespace prime_power_divisibility_l1660_166014

theorem prime_power_divisibility : 
  (∃ p : ℕ, p ≥ 7 ∧ Nat.Prime p ∧ (p^4 - 1) % 48 = 0) ∧ 
  (∃ q : ℕ, q ≥ 7 ∧ Nat.Prime q ∧ (q^4 - 1) % 48 ≠ 0) := by
  sorry

end prime_power_divisibility_l1660_166014


namespace alex_has_largest_final_answer_l1660_166092

def maria_operation (x : ℕ) : ℕ := ((x - 2) * 3) + 4

def alex_operation (x : ℕ) : ℕ := ((x * 3) - 3) + 4

def lee_operation (x : ℕ) : ℕ := ((x - 2) + 4) * 3

theorem alex_has_largest_final_answer :
  let maria_start := 12
  let alex_start := 15
  let lee_start := 13
  let maria_final := maria_operation maria_start
  let alex_final := alex_operation alex_start
  let lee_final := lee_operation lee_start
  alex_final > maria_final ∧ alex_final > lee_final :=
by sorry

end alex_has_largest_final_answer_l1660_166092


namespace complex_fraction_simplification_l1660_166047

theorem complex_fraction_simplification :
  1 + 2 / (1 + 2 / (2 * 2)) = 7 / 3 := by
  sorry

end complex_fraction_simplification_l1660_166047


namespace train_passing_platform_l1660_166028

/-- A train passes a platform -/
theorem train_passing_platform 
  (train_length : ℝ) 
  (tree_crossing_time : ℝ) 
  (platform_length : ℝ) 
  (h1 : train_length = 600) 
  (h2 : tree_crossing_time = 60) 
  (h3 : platform_length = 450) : 
  (train_length + platform_length) / (train_length / tree_crossing_time) = 105 := by
  sorry

end train_passing_platform_l1660_166028


namespace total_time_equals_sum_l1660_166002

/-- The total time Porche initially had for homework -/
def total_time : ℕ := 180

/-- Time required for math homework -/
def math_time : ℕ := 45

/-- Time required for English homework -/
def english_time : ℕ := 30

/-- Time required for science homework -/
def science_time : ℕ := 50

/-- Time required for history homework -/
def history_time : ℕ := 25

/-- Time left for the special project -/
def project_time : ℕ := 30

/-- Theorem stating that the total time is the sum of all homework times -/
theorem total_time_equals_sum :
  total_time = math_time + english_time + science_time + history_time + project_time := by
  sorry

end total_time_equals_sum_l1660_166002


namespace scout_saturday_hours_scout_saturday_hours_is_four_l1660_166018

/-- Scout's delivery earnings over a weekend --/
theorem scout_saturday_hours : ℕ :=
  let base_pay : ℕ := 10  -- Base pay per hour in dollars
  let tip_per_customer : ℕ := 5  -- Tip per customer in dollars
  let saturday_customers : ℕ := 5  -- Number of customers on Saturday
  let sunday_hours : ℕ := 5  -- Hours worked on Sunday
  let sunday_customers : ℕ := 8  -- Number of customers on Sunday
  let total_earnings : ℕ := 155  -- Total earnings for the weekend in dollars

  let saturday_hours : ℕ := 
    (total_earnings - 
     (base_pay * sunday_hours + tip_per_customer * sunday_customers + 
      tip_per_customer * saturday_customers)) / base_pay

  saturday_hours

/-- Proof that Scout worked 4 hours on Saturday --/
theorem scout_saturday_hours_is_four : scout_saturday_hours = 4 := by
  sorry

end scout_saturday_hours_scout_saturday_hours_is_four_l1660_166018


namespace solve_equation_l1660_166041

theorem solve_equation (x : ℝ) : (x^3 * 6^2) / 432 = 144 → x = 12 := by
  sorry

end solve_equation_l1660_166041


namespace solve_for_y_l1660_166005

theorem solve_for_y (x y : ℝ) (h1 : x^2 + 2 = y - 4) (h2 : x = -3) : y = 15 := by
  sorry

end solve_for_y_l1660_166005


namespace pet_store_cats_l1660_166039

theorem pet_store_cats (initial_siamese : ℕ) (cats_sold : ℕ) (cats_left : ℕ) :
  initial_siamese = 12 →
  cats_sold = 20 →
  cats_left = 12 →
  ∃ initial_house : ℕ, initial_house = 20 ∧ initial_siamese + initial_house = cats_sold + cats_left :=
by sorry

end pet_store_cats_l1660_166039


namespace circle_locus_l1660_166062

/-- The locus of the center of a circle passing through (-2, 0) and tangent to x = 2 -/
theorem circle_locus (x₀ y₀ : ℝ) : 
  (∃ (r : ℝ), r > 0 ∧ 
    (x₀ + 2)^2 + y₀^2 = r^2 ∧ 
    |x₀ - 2| = r) →
  y₀^2 = -8 * x₀ :=
by sorry

end circle_locus_l1660_166062


namespace cos_2x_value_l1660_166048

theorem cos_2x_value (x : Real) (h : Real.sin (π / 2 + x) = 3 / 5) : 
  Real.cos (2 * x) = -7 / 25 := by
  sorry

end cos_2x_value_l1660_166048


namespace root_in_interval_l1660_166011

def f (x : ℝ) := x^3 - 3*x + 1

theorem root_in_interval :
  ∃ x : ℝ, 1 < x ∧ x < 2 ∧ f x = 0 :=
by sorry

end root_in_interval_l1660_166011


namespace quadratic_inequality_always_negative_l1660_166063

theorem quadratic_inequality_always_negative : ∀ x : ℝ, -6 * x^2 + 2 * x - 4 < 0 := by
  sorry

end quadratic_inequality_always_negative_l1660_166063


namespace one_distinct_computable_value_l1660_166054

/-- Represents a valid parenthesization of the expression 3^3^3^3 --/
inductive Parenthesization
| Original : Parenthesization
| Left : Parenthesization
| Middle : Parenthesization
| Right : Parenthesization
| DoubleLeft : Parenthesization
| DoubleRight : Parenthesization

/-- Evaluates a given parenthesization to a natural number if computable --/
def evaluate : Parenthesization → Option ℕ
| Parenthesization.Original => none
| Parenthesization.Left => none
| Parenthesization.Middle => some 19683
| Parenthesization.Right => none
| Parenthesization.DoubleLeft => none
| Parenthesization.DoubleRight => none

/-- The number of distinct, computable values when changing the order of exponentiation in 3^3^3^3 --/
def distinctComputableValues : ℕ :=
  (List.map evaluate [Parenthesization.Left, Parenthesization.Middle, Parenthesization.Right,
                      Parenthesization.DoubleLeft, Parenthesization.DoubleRight]).filterMap id |>.eraseDups |>.length

theorem one_distinct_computable_value : distinctComputableValues = 1 := by
  sorry

end one_distinct_computable_value_l1660_166054


namespace min_value_xyz_l1660_166016

theorem min_value_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 8) :
  x + 2 * y + 4 * z ≥ 12 := by
  sorry

end min_value_xyz_l1660_166016


namespace matching_pair_probability_l1660_166052

def black_socks : ℕ := 12
def blue_socks : ℕ := 10
def total_socks : ℕ := black_socks + blue_socks

def matching_pairs : ℕ := (black_socks * (black_socks - 1)) / 2 + (blue_socks * (blue_socks - 1)) / 2
def total_combinations : ℕ := (total_socks * (total_socks - 1)) / 2

theorem matching_pair_probability :
  (matching_pairs : ℚ) / total_combinations = 111 / 231 := by sorry

end matching_pair_probability_l1660_166052


namespace power_product_equality_l1660_166032

theorem power_product_equality : (-0.125)^2021 * 8^2022 = -8 := by sorry

end power_product_equality_l1660_166032


namespace john_emu_pens_l1660_166072

/-- The number of pens for emus that John has -/
def num_pens : ℕ := sorry

/-- The number of emus per pen -/
def emus_per_pen : ℕ := 6

/-- The ratio of female emus to total emus -/
def female_ratio : ℚ := 1/2

/-- The number of eggs laid by each female emu per day -/
def eggs_per_female_per_day : ℕ := 1

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The total number of eggs collected in a week -/
def total_eggs_per_week : ℕ := 84

theorem john_emu_pens : 
  num_pens = 4 := by sorry

end john_emu_pens_l1660_166072


namespace ratio_problem_l1660_166001

theorem ratio_problem (a b c : ℝ) (h1 : b / a = 3) (h2 : c / b = 2) : 
  (a + b) / (b + c) = 4 / 9 := by
  sorry

end ratio_problem_l1660_166001


namespace factor_expression_l1660_166083

theorem factor_expression (x y : ℝ) : 3 * x^2 - 3 * y^2 = 3 * (x + y) * (x - y) := by
  sorry

end factor_expression_l1660_166083


namespace calculate_principal_l1660_166068

/-- Given simple interest, rate, and time, calculate the principal sum -/
theorem calculate_principal (simple_interest rate time : ℝ) : 
  simple_interest = 16065 * rate * time / 100 →
  rate = 5 →
  time = 5 →
  simple_interest = 4016.25 := by
  sorry

#check calculate_principal

end calculate_principal_l1660_166068


namespace expected_malfunctioning_computers_correct_l1660_166000

/-- The expected number of malfunctioning computers -/
def expected_malfunctioning_computers (a b : ℝ) : ℝ := a + b

/-- Theorem: The expected number of malfunctioning computers is a + b -/
theorem expected_malfunctioning_computers_correct (a b : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) : 
  expected_malfunctioning_computers a b = a + b := by
  sorry

end expected_malfunctioning_computers_correct_l1660_166000


namespace range_of_a_l1660_166015

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x > 0 → a < x + 4 / x) → 
  a < 4 := by
sorry

end range_of_a_l1660_166015


namespace tetrahedron_volume_l1660_166065

/-- The volume of a tetrahedron with vertices on the positive coordinate axes -/
theorem tetrahedron_volume (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a^2 + b^2 = 25) (h5 : b^2 + c^2 = 36) (h6 : c^2 + a^2 = 49) :
  (1 / 6 : ℝ) * a * b * c = Real.sqrt 95 := by
sorry

end tetrahedron_volume_l1660_166065


namespace circle_equation_correct_l1660_166086

-- Define the center and radius of the circle
def center : ℝ × ℝ := (1, -2)
def radius : ℝ := 3

-- Define the equation of the circle
def circle_equation (x y : ℝ) : Prop :=
  (x - center.1)^2 + (y - center.2)^2 = radius^2

-- Theorem to prove
theorem circle_equation_correct :
  ∀ x y : ℝ, circle_equation x y ↔ (x - 1)^2 + (y + 2)^2 = 9 :=
by sorry

end circle_equation_correct_l1660_166086


namespace equation_consequences_l1660_166017

theorem equation_consequences (x y : ℝ) (h : x^2 + y^2 - x*y = 1) :
  (-2 : ℝ) ≤ x + y ∧ x + y ≤ 2 ∧ 2/3 ≤ x^2 + y^2 ∧ x^2 + y^2 ≤ 2 := by
  sorry

end equation_consequences_l1660_166017


namespace units_digit_of_7_to_5_l1660_166010

theorem units_digit_of_7_to_5 : 7^5 % 10 = 7 := by
  sorry

end units_digit_of_7_to_5_l1660_166010

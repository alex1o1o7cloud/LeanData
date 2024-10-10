import Mathlib

namespace equation_solution_l3810_381093

theorem equation_solution : 
  ∃! x : ℚ, (x + 10) / (x - 4) = (x - 3) / (x + 6) ∧ x = -48 / 23 := by
  sorry

end equation_solution_l3810_381093


namespace intersection_and_union_of_sets_l3810_381056

theorem intersection_and_union_of_sets (x : ℝ) 
  (A : Set ℝ) (B : Set ℝ)
  (hA : A = {-3, x^2, x+1})
  (hB : B = {x-3, 2*x-1, x^2+1})
  (hIntersection : A ∩ B = {-3}) :
  x = -1 ∧ A ∪ B = {-4, -3, 0, 1, 2} := by
  sorry

end intersection_and_union_of_sets_l3810_381056


namespace factor_implies_m_value_l3810_381044

theorem factor_implies_m_value (x y m : ℝ) : 
  (∃ k : ℝ, (1 - 2*x + y) * k = 4*x*y - 4*x^2 - y^2 - m) → m = -1 := by
  sorry

end factor_implies_m_value_l3810_381044


namespace at_least_one_passes_l3810_381039

def exam_pool : ℕ := 10
def A_correct : ℕ := 6
def B_correct : ℕ := 8
def test_questions : ℕ := 3
def passing_threshold : ℕ := 2

def prob_A_pass : ℚ := (Nat.choose A_correct 2 * Nat.choose (exam_pool - A_correct) 1 + Nat.choose A_correct 3) / Nat.choose exam_pool test_questions

def prob_B_pass : ℚ := (Nat.choose B_correct 2 * Nat.choose (exam_pool - B_correct) 1 + Nat.choose B_correct 3) / Nat.choose exam_pool test_questions

theorem at_least_one_passes : 
  1 - (1 - prob_A_pass) * (1 - prob_B_pass) = 44 / 45 := by sorry

end at_least_one_passes_l3810_381039


namespace count_rectangular_subsets_5x5_l3810_381094

/-- The number of ways to select a rectangular subset in a 5x5 grid -/
def rectangular_subsets_5x5 : ℕ := 225

/-- A proof that there are 225 ways to select a rectangular subset in a 5x5 grid -/
theorem count_rectangular_subsets_5x5 : rectangular_subsets_5x5 = 225 := by
  sorry

end count_rectangular_subsets_5x5_l3810_381094


namespace triangle_theorem_l3810_381021

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem states that if b cos C + c cos B = 2a cos A and AB · AC = √3 in a triangle ABC,
    then the measure of angle A is π/3 and the area of the triangle is 3/2. -/
theorem triangle_theorem (t : Triangle) 
  (h1 : t.b * Real.cos t.C + t.c * Real.cos t.B = 2 * t.a * Real.cos t.A)
  (h2 : t.a * t.c * Real.cos t.A = Real.sqrt 3) : 
  t.A = π / 3 ∧ (1 / 2 * t.a * t.c * Real.sin t.A = 3 / 2) := by
  sorry

end triangle_theorem_l3810_381021


namespace integral_x_plus_inverse_x_l3810_381080

theorem integral_x_plus_inverse_x : ∫ x in (1 : ℝ)..2, (x + 1/x) = 3/2 + Real.log 2 := by
  sorry

end integral_x_plus_inverse_x_l3810_381080


namespace first_load_pieces_l3810_381034

/-- The number of pieces of clothing in the first load -/
def first_load (total : ℕ) (num_small_loads : ℕ) (pieces_per_small_load : ℕ) : ℕ :=
  total - (num_small_loads * pieces_per_small_load)

/-- Theorem stating that the number of pieces of clothing in the first load is 17 -/
theorem first_load_pieces : first_load 47 5 6 = 17 := by
  sorry

end first_load_pieces_l3810_381034


namespace three_tangent_lines_l3810_381018

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/4 = 1

-- Define a line passing through (1, 0)
def line_through_P (m : ℝ) (x y : ℝ) : Prop := y = m * (x - 1)

-- Define the condition for a line to intersect the hyperbola at only one point
def single_intersection (m : ℝ) : Prop :=
  ∃! (x y : ℝ), hyperbola x y ∧ line_through_P m x y

-- Main theorem
theorem three_tangent_lines :
  ∃ (m₁ m₂ m₃ : ℝ), 
    single_intersection m₁ ∧ 
    single_intersection m₂ ∧ 
    single_intersection m₃ ∧
    m₁ ≠ m₂ ∧ m₁ ≠ m₃ ∧ m₂ ≠ m₃ ∧
    ∀ (m : ℝ), single_intersection m → m = m₁ ∨ m = m₂ ∨ m = m₃ :=
sorry

end three_tangent_lines_l3810_381018


namespace product_zero_implies_factor_zero_unit_circle_sum_one_implies_diff_sqrt_three_l3810_381040

variables (z₁ z₂ : ℂ)

-- Statement B
theorem product_zero_implies_factor_zero : z₁ * z₂ = 0 → z₁ = 0 ∨ z₂ = 0 := by sorry

-- Statement D
theorem unit_circle_sum_one_implies_diff_sqrt_three : 
  Complex.abs z₁ = 1 → Complex.abs z₂ = 1 → z₁ + z₂ = 1 → Complex.abs (z₁ - z₂) = Real.sqrt 3 := by sorry

end product_zero_implies_factor_zero_unit_circle_sum_one_implies_diff_sqrt_three_l3810_381040


namespace ellipse_tangent_circle_radius_l3810_381041

/-- Given an ellipse with semi-major axis a and semi-minor axis b,
    and a circle centered at one of its foci and tangent to the ellipse,
    prove that the radius of the circle is √((a^2 - b^2)/2). -/
theorem ellipse_tangent_circle_radius 
  (a b : ℝ) 
  (h_a : a = 6) 
  (h_b : b = 3) : 
  let c := Real.sqrt (a^2 - b^2)
  let r := Real.sqrt ((a^2 - b^2)/2)
  ∀ x y : ℝ,
  (x^2 / a^2 + y^2 / b^2 = 1) →
  ((x - c)^2 + y^2 = r^2) →
  r = Real.sqrt 6 :=
by sorry


end ellipse_tangent_circle_radius_l3810_381041


namespace bottle_caps_problem_l3810_381004

theorem bottle_caps_problem (sammy janine billie : ℕ) 
  (h1 : sammy = 8)
  (h2 : sammy = janine + 2)
  (h3 : janine = 3 * billie) :
  billie = 2 := by
  sorry

end bottle_caps_problem_l3810_381004


namespace book_arrangement_count_l3810_381016

/-- The number of ways to arrange two types of indistinguishable objects in a row -/
def arrangement_count (n m : ℕ) : ℕ :=
  Nat.choose (n + m) n

/-- Theorem: Arranging 4 copies of one book and 5 copies of another book yields 126 possibilities -/
theorem book_arrangement_count :
  arrangement_count 4 5 = 126 := by
  sorry

end book_arrangement_count_l3810_381016


namespace largest_even_number_with_sum_20_l3810_381069

/-- A function that returns the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- A function that checks if all digits in a natural number are different -/
def has_different_digits (n : ℕ) : Prop := sorry

/-- The theorem stating that 86420 is the largest even number with all different digits whose digits add up to 20 -/
theorem largest_even_number_with_sum_20 : 
  ∀ n : ℕ, 
    n % 2 = 0 ∧ 
    has_different_digits n ∧ 
    sum_of_digits n = 20 → 
    n ≤ 86420 := by sorry

end largest_even_number_with_sum_20_l3810_381069


namespace triangle_angle_calculation_l3810_381073

theorem triangle_angle_calculation (a b c : ℝ) (A B C : ℝ) :
  a = 3 * Real.sqrt 2 →
  c = 3 →
  C = π / 6 →
  A = π / 4 ∨ A = 3 * π / 4 :=
by sorry

end triangle_angle_calculation_l3810_381073


namespace spinner_probability_l3810_381048

theorem spinner_probability (p_A p_B p_C p_D : ℚ) : 
  p_A = 1/2 →
  p_B = 1/8 →
  p_C = p_D →
  p_A + p_B + p_C + p_D = 1 →
  p_C = 3/16 := by
sorry

end spinner_probability_l3810_381048


namespace balloon_ratio_is_seven_l3810_381022

-- Define the number of balloons for Dan and Tim
def dans_balloons : ℕ := 29
def tims_balloons : ℕ := 203

-- Define the ratio of Tim's balloons to Dan's balloons
def balloon_ratio : ℚ := tims_balloons / dans_balloons

-- Theorem stating that the ratio is 7
theorem balloon_ratio_is_seven : balloon_ratio = 7 := by
  sorry

end balloon_ratio_is_seven_l3810_381022


namespace max_product_sum_l3810_381075

theorem max_product_sum (X Y Z : ℕ) (sum_constraint : X + Y + Z = 15) :
  (∀ X' Y' Z' : ℕ, X' + Y' + Z' = 15 → 
    X' * Y' * Z' + X' * Y' + Y' * Z' + Z' * X' ≤ X * Y * Z + X * Y + Y * Z + Z * X) →
  X * Y * Z + X * Y + Y * Z + Z * X = 200 := by
  sorry

end max_product_sum_l3810_381075


namespace no_zero_term_l3810_381087

/-- An arithmetic progression is defined by its first term and common difference -/
structure ArithmeticProgression where
  a : ℝ  -- first term
  d : ℝ  -- common difference

/-- The nth term of an arithmetic progression -/
def nthTerm (ap : ArithmeticProgression) (n : ℕ) : ℝ :=
  ap.a + (n - 1 : ℝ) * ap.d

/-- The condition given in the problem -/
def satisfiesCondition (ap : ArithmeticProgression) : Prop :=
  nthTerm ap 5 + nthTerm ap 21 = nthTerm ap 8 + nthTerm ap 15 + nthTerm ap 13

/-- The main theorem -/
theorem no_zero_term (ap : ArithmeticProgression) 
    (h : satisfiesCondition ap) : 
    ¬∃ (n : ℕ), n > 0 ∧ nthTerm ap n = 0 :=
  sorry

end no_zero_term_l3810_381087


namespace maci_blue_pens_l3810_381096

/-- The number of blue pens Maci needs -/
def num_blue_pens : ℕ := sorry

/-- The number of red pens Maci needs -/
def num_red_pens : ℕ := 15

/-- The cost of a blue pen in cents -/
def blue_pen_cost : ℕ := 10

/-- The cost of a red pen in cents -/
def red_pen_cost : ℕ := 2 * blue_pen_cost

/-- The total cost of all pens in cents -/
def total_cost : ℕ := 400

theorem maci_blue_pens :
  num_blue_pens * blue_pen_cost + num_red_pens * red_pen_cost = total_cost ∧
  num_blue_pens = 10 :=
sorry

end maci_blue_pens_l3810_381096


namespace unique_solution_l3810_381005

def f (d : ℝ) (x : ℝ) : ℝ := 4 * x^3 - d * x

def g (a b c : ℝ) (x : ℝ) : ℝ := 4 * x^3 + a * x^2 + b * x + c

theorem unique_solution :
  ∃! (a b c d : ℝ),
    (∀ x ∈ Set.Icc (-1 : ℝ) 1, |f d x| ≤ 1) ∧
    (∀ x ∈ Set.Icc (-1 : ℝ) 1, |g a b c x| ≤ 1) ∧
    a = 0 ∧ b = -3 ∧ c = 0 ∧ d = 3 :=
by sorry

end unique_solution_l3810_381005


namespace triangle_area_equation_l3810_381068

theorem triangle_area_equation : ∃! (x : ℝ), x > 3 ∧ (1/2 : ℝ) * (x - 3) * (3*x + 7) = 12*x - 9 := by
  sorry

end triangle_area_equation_l3810_381068


namespace rectangle_locus_l3810_381055

/-- Given a rectangle with length l and width w, and a fixed number b,
    this theorem states that the locus of all points P(x, y) in the plane of the rectangle
    such that the sum of the squares of the distances from P to the four vertices
    of the rectangle equals b is a circle if and only if b > l^2 + w^2. -/
theorem rectangle_locus (l w b : ℝ) :
  (∃ (c : ℝ × ℝ) (r : ℝ),
    ∀ (x y : ℝ),
      (x - 0)^2 + (y - 0)^2 + (x - l)^2 + (y - 0)^2 +
      (x - l)^2 + (y - w)^2 + (x - 0)^2 + (y - w)^2 = b ↔
      (x - c.1)^2 + (y - c.2)^2 = r^2) ↔
  b > l^2 + w^2 := by sorry

end rectangle_locus_l3810_381055


namespace sum_of_cubes_l3810_381010

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 := by
  sorry

end sum_of_cubes_l3810_381010


namespace min_chicken_hits_l3810_381037

def ring_toss (chicken monkey dog : ℕ) : Prop :=
  chicken * 9 + monkey * 5 + dog * 2 = 61 ∧
  chicken + monkey + dog = 10 ∧
  chicken ≥ 1 ∧ monkey ≥ 1 ∧ dog ≥ 1

theorem min_chicken_hits :
  ∀ chicken monkey dog : ℕ,
    ring_toss chicken monkey dog →
    chicken ≥ 5 :=
by
  sorry

end min_chicken_hits_l3810_381037


namespace gain_percentage_calculation_l3810_381060

def cost_price : ℝ := 180
def selling_price : ℝ := 216

theorem gain_percentage_calculation : 
  let gain_percentage := (selling_price / cost_price - 1) * 100
  gain_percentage = 20 := by sorry

end gain_percentage_calculation_l3810_381060


namespace specific_arrangement_probability_l3810_381074

def num_red_lamps : ℕ := 4
def num_blue_lamps : ℕ := 4
def num_lamps_on : ℕ := 4

def probability_specific_arrangement : ℚ := 3/49

theorem specific_arrangement_probability :
  let total_lamps := num_red_lamps + num_blue_lamps
  let total_arrangements := (total_lamps.choose num_red_lamps) * (total_lamps.choose num_lamps_on)
  let favorable_outcomes := (total_lamps - 2).choose (num_red_lamps - 1) * (total_lamps - 2).choose (num_lamps_on - 1)
  (favorable_outcomes : ℚ) / total_arrangements = probability_specific_arrangement :=
sorry

end specific_arrangement_probability_l3810_381074


namespace min_value_trig_expression_l3810_381062

theorem min_value_trig_expression (α β : Real) :
  ∃ (min : Real),
    (∀ (α' β' : Real), (3 * Real.cos α' + 4 * Real.sin β' - 7)^2 + (3 * Real.sin α' + 4 * Real.cos β' - 12)^2 ≥ min) ∧
    ((3 * Real.cos α + 4 * Real.sin β - 7)^2 + (3 * Real.sin α + 4 * Real.cos β - 12)^2 = min) ∧
    min = 36 := by
  sorry

end min_value_trig_expression_l3810_381062


namespace son_age_l3810_381020

theorem son_age (father_age son_age : ℕ) : 
  father_age = son_age + 24 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 22 := by
sorry

end son_age_l3810_381020


namespace gcd_lcm_sum_18_30_45_l3810_381000

/-- The sum of the greatest common factor and least common multiple of 18, 30, and 45 is 93 -/
theorem gcd_lcm_sum_18_30_45 : 
  (Nat.gcd 18 (Nat.gcd 30 45) + Nat.lcm 18 (Nat.lcm 30 45)) = 93 := by
  sorry

end gcd_lcm_sum_18_30_45_l3810_381000


namespace outfits_count_l3810_381019

/-- The number of possible outfits given the number of shirts, pants, and shoes. -/
def number_of_outfits (shirts : ℕ) (pants : ℕ) (shoes : ℕ) : ℕ :=
  shirts * pants * shoes

/-- Theorem stating that the number of outfits from 4 shirts, 5 pants, and 2 shoes is 40. -/
theorem outfits_count : number_of_outfits 4 5 2 = 40 := by
  sorry

end outfits_count_l3810_381019


namespace two_zeros_neither_necessary_nor_sufficient_l3810_381064

open Real

-- Define the function f and its derivative f'
variable (f : ℝ → ℝ) (f' : ℝ → ℝ)

-- Define the property of f' having exactly two zeros in (0, 2)
def has_two_zeros (f' : ℝ → ℝ) : Prop :=
  ∃ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 2 ∧ f' x₁ = 0 ∧ f' x₂ = 0 ∧
  ∀ x, 0 < x ∧ x < 2 ∧ f' x = 0 → x = x₁ ∨ x = x₂

-- Define the property of f having exactly two extreme points in (0, 2)
def has_two_extreme_points (f : ℝ → ℝ) (f' : ℝ → ℝ) : Prop :=
  ∃ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 2 ∧ f' x₁ = 0 ∧ f' x₂ = 0 ∧
  (∀ x, 0 < x ∧ x < x₁ → f' x ≠ 0) ∧
  (∀ x, x₁ < x ∧ x < x₂ → f' x ≠ 0) ∧
  (∀ x, x₂ < x ∧ x < 2 → f' x ≠ 0)

-- Theorem stating that has_two_zeros is neither necessary nor sufficient for has_two_extreme_points
theorem two_zeros_neither_necessary_nor_sufficient :
  ¬(∀ f f', has_two_zeros f' → has_two_extreme_points f f') ∧
  ¬(∀ f f', has_two_extreme_points f f' → has_two_zeros f') :=
sorry

end two_zeros_neither_necessary_nor_sufficient_l3810_381064


namespace perfect_squares_problem_l3810_381098

theorem perfect_squares_problem (m n a b c d : ℕ) :
  2000 + 100 * a + 10 * b + 9 = n^2 →
  2000 + 100 * c + 10 * d + 9 = m^2 →
  m > n →
  10 ≤ 10 * a + b →
  10 * a + b ≤ 99 →
  10 ≤ 10 * c + d →
  10 * c + d ≤ 99 →
  m + n = 100 ∧ (10 * a + b) + (10 * c + d) = 100 := by
  sorry

end perfect_squares_problem_l3810_381098


namespace count_integers_in_range_l3810_381053

theorem count_integers_in_range : 
  (Finset.filter (fun x => 30 < x^2 + 8*x + 16 ∧ x^2 + 8*x + 16 < 60) (Finset.range 100)).card = 2 := by
  sorry

end count_integers_in_range_l3810_381053


namespace choir_members_count_l3810_381099

theorem choir_members_count : ∃! n : ℕ, 200 ≤ n ∧ n ≤ 300 ∧ n % 10 = 6 ∧ n % 11 = 6 := by
  sorry

end choir_members_count_l3810_381099


namespace boat_distance_downstream_l3810_381095

/-- Calculates the distance traveled downstream by a boat -/
theorem boat_distance_downstream 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (time : ℝ) 
  (h1 : boat_speed = 22) 
  (h2 : stream_speed = 5) 
  (h3 : time = 5) : 
  boat_speed + stream_speed * time = 135 := by
  sorry

#check boat_distance_downstream

end boat_distance_downstream_l3810_381095


namespace quadratic_symmetry_l3810_381084

/-- A quadratic function with axis of symmetry at x = 6 and p(0) = -3 -/
def p (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_symmetry (a b c : ℝ) :
  (∀ x, p a b c (6 + x) = p a b c (6 - x)) →  -- axis of symmetry at x = 6
  p a b c 0 = -3 →                           -- p(0) = -3
  p a b c 12 = -3 :=                         -- p(12) = -3
by
  sorry

end quadratic_symmetry_l3810_381084


namespace security_system_probability_l3810_381082

theorem security_system_probability (p : ℝ) : 
  (1/8 : ℝ) * (1 - p) + (7/8 : ℝ) * p = 9/40 → p = 2/15 := by
  sorry

end security_system_probability_l3810_381082


namespace expression_value_l3810_381088

theorem expression_value (x y : ℤ) (hx : x = 3) (hy : y = 2) :
  3 * x - 4 * y + 2 * y = 5 := by
  sorry

end expression_value_l3810_381088


namespace slow_clock_catch_up_l3810_381083

/-- Represents the number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- Represents how many minutes the clock is slow per hour -/
def slow_rate : ℕ := 4

/-- Represents the current time on the slow clock in minutes past 11:00 -/
def current_slow_time : ℕ := 46

/-- Represents the target time on the slow clock in minutes past 11:00 -/
def target_slow_time : ℕ := 60

/-- Theorem stating that it takes 15 minutes of correct time for the slow clock to reach 12:00 -/
theorem slow_clock_catch_up :
  (target_slow_time - current_slow_time) * minutes_per_hour / (minutes_per_hour - slow_rate) = 15 := by
  sorry

end slow_clock_catch_up_l3810_381083


namespace perpendicular_lines_direction_vectors_l3810_381050

theorem perpendicular_lines_direction_vectors (b : ℝ) :
  let v1 : Fin 2 → ℝ := ![- 5, 11]
  let v2 : Fin 2 → ℝ := ![b, 3]
  (∀ i : Fin 2, (v1 • v2) = 0) → b = 33 / 5 := by
  sorry

end perpendicular_lines_direction_vectors_l3810_381050


namespace simplify_algebraic_expression_l3810_381026

theorem simplify_algebraic_expression (a b : ℝ) (h : a ≠ b) :
  (a^3 - b^3) / (a * b) - (a * b^2 - b^3) / (a * b - a^3) = 2 * a * (a - b) / b :=
sorry

end simplify_algebraic_expression_l3810_381026


namespace ring_arrangement_count_l3810_381024

/-- The number of ways to arrange 6 rings out of 10 on 5 fingers -/
def ring_arrangements : ℕ := sorry

/-- The number of ways to choose 6 rings out of 10 -/
def choose_rings : ℕ := sorry

/-- The number of ways to order 6 rings -/
def order_rings : ℕ := sorry

/-- The number of ways to distribute 6 rings among 5 fingers -/
def distribute_rings : ℕ := sorry

theorem ring_arrangement_count :
  ring_arrangements = choose_rings * order_rings * distribute_rings ∧
  ring_arrangements = 31752000 :=
sorry

end ring_arrangement_count_l3810_381024


namespace product_of_primes_in_final_positions_l3810_381066

-- Define the colors
inductive Color
| Red
| Yellow
| Green
| Blue

-- Define the positions in a 2x2 grid
inductive Position
| TopLeft
| TopRight
| BottomLeft
| BottomRight

-- Define the transformation function
def transform (c : Color) : Position → Position
| Position.TopLeft => 
    match c with
    | Color.Red => Position.TopRight
    | Color.Yellow => Position.TopRight
    | Color.Green => Position.BottomLeft
    | Color.Blue => Position.BottomRight
| Position.TopRight => 
    match c with
    | Color.Red => Position.TopRight
    | Color.Yellow => Position.TopRight
    | Color.Green => Position.BottomRight
    | Color.Blue => Position.BottomRight
| Position.BottomLeft => 
    match c with
    | Color.Red => Position.TopLeft
    | Color.Yellow => Position.TopLeft
    | Color.Green => Position.BottomLeft
    | Color.Blue => Position.BottomLeft
| Position.BottomRight => 
    match c with
    | Color.Red => Position.TopLeft
    | Color.Yellow => Position.TopLeft
    | Color.Green => Position.BottomRight
    | Color.Blue => Position.BottomRight

-- Define the numbers in Figure 4
def figure4 (p : Position) : Nat :=
  match p with
  | Position.TopLeft => 6
  | Position.TopRight => 7
  | Position.BottomLeft => 5
  | Position.BottomRight => 8

-- Define primality
def isPrime (n : Nat) : Prop := n > 1 ∧ ∀ m : Nat, m > 1 → m < n → ¬(n % m = 0)

-- Theorem statement
theorem product_of_primes_in_final_positions : 
  let finalRedPosition := transform Color.Red (transform Color.Red Position.TopLeft)
  let finalYellowPosition := transform Color.Yellow (transform Color.Yellow Position.TopRight)
  (isPrime (figure4 finalRedPosition) ∧ isPrime (figure4 finalYellowPosition)) →
  figure4 finalRedPosition * figure4 finalYellowPosition = 55 := by
  sorry


end product_of_primes_in_final_positions_l3810_381066


namespace watch_cost_price_l3810_381013

/-- The cost price of a watch given specific selling conditions -/
theorem watch_cost_price (C : ℚ) : 
  (0.9 * C = C - 0.1 * C) →  -- Selling price at 10% loss
  (1.04 * C = C + 0.04 * C) →  -- Selling price at 4% gain
  (1.04 * C - 0.9 * C = 200) →  -- Difference between selling prices
  C = 10000 / 7 := by
sorry

end watch_cost_price_l3810_381013


namespace vector_combination_l3810_381001

/-- Given three points A, B, and C in a plane, prove that the coordinates of 1/2 * AC - 1/4 * BC are (-3, 6) -/
theorem vector_combination (A B C : ℝ × ℝ) (h1 : A = (2, -4)) (h2 : B = (0, 6)) (h3 : C = (-8, 10)) :
  (1 / 2 : ℝ) • (C - A) - (1 / 4 : ℝ) • (C - B) = (-3, 6) := by
  sorry

end vector_combination_l3810_381001


namespace tangent_slope_circle_l3810_381023

/-- Slope of the line tangent to a circle -/
theorem tangent_slope_circle (center : ℝ × ℝ) (point : ℝ × ℝ) : 
  center = (3, 2) → point = (5, 5) → 
  (let radius_slope := (point.2 - center.2) / (point.1 - center.1);
   -1 / radius_slope) = -2/3 := by sorry

end tangent_slope_circle_l3810_381023


namespace sum_of_x_and_y_l3810_381051

theorem sum_of_x_and_y (x y : ℝ) : 
  |x - 2*y - 3| + (y - 2*x)^2 = 0 → x + y = -3 := by
  sorry

end sum_of_x_and_y_l3810_381051


namespace lcm_of_25_35_50_l3810_381029

theorem lcm_of_25_35_50 : Nat.lcm 25 (Nat.lcm 35 50) = 350 := by
  sorry

end lcm_of_25_35_50_l3810_381029


namespace quadratic_equation_solution_l3810_381017

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  x₁^2 - 4*x₁ - 5 = 0 ∧ 
  x₂^2 - 4*x₂ - 5 = 0 ∧ 
  x₁ = 5 ∧ 
  x₂ = -1 := by
  sorry

end quadratic_equation_solution_l3810_381017


namespace min_bilingual_students_l3810_381047

theorem min_bilingual_students (total : ℕ) (hindi : ℕ) (english : ℕ) 
  (h_total : total = 40)
  (h_hindi : hindi = 30)
  (h_english : english = 20) :
  ∃ (both : ℕ), both ≥ hindi + english - total ∧ 
    ∀ (x : ℕ), x ≥ hindi + english - total → x ≥ both :=
by sorry

end min_bilingual_students_l3810_381047


namespace shift_function_unit_shift_l3810_381059

/-- A function satisfying specific inequalities for shifts of 24 and 77 -/
def ShiftFunction (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f (x + 24) ≤ f x + 24) ∧ (∀ x : ℝ, f (x + 77) ≥ f x + 77)

/-- Theorem stating that a ShiftFunction satisfies f(x+1) = f(x)+1 for all real x -/
theorem shift_function_unit_shift (f : ℝ → ℝ) (hf : ShiftFunction f) :
  ∀ x : ℝ, f (x + 1) = f x + 1 := by
  sorry

end shift_function_unit_shift_l3810_381059


namespace polka_dot_price_is_67_l3810_381065

def checkered_price : ℝ := 75
def plain_price : ℝ := 45
def striped_price : ℝ := 63
def total_price : ℝ := 250

def checkered_per_yard : ℝ := 7.5
def plain_per_yard : ℝ := 6
def striped_per_yard : ℝ := 9
def polka_dot_per_yard : ℝ := 4.5

def discount_rate : ℝ := 0.1
def discount_threshold : ℝ := 10

def polka_dot_price : ℝ := total_price - (checkered_price + plain_price + striped_price)

theorem polka_dot_price_is_67 : polka_dot_price = 67 := by
  sorry

end polka_dot_price_is_67_l3810_381065


namespace point_on_number_line_l3810_381061

/-- Given points P, Q, and R on a number line, where Q is halfway between P and R,
    P is at -6, and Q is at -1, prove that R is at 4. -/
theorem point_on_number_line (P Q R : ℝ) : 
  Q = (P + R) / 2 → P = -6 → Q = -1 → R = 4 := by
  sorry

end point_on_number_line_l3810_381061


namespace quarter_circles_sum_limit_l3810_381071

/-- The sum of the lengths of quarter-circles approaches a value between the diameter and semi-circumference -/
theorem quarter_circles_sum_limit (D : ℝ) (h : D > 0) :
  ∃ (L : ℝ), (∀ ε > 0, ∃ N, ∀ n ≥ N, |2 * n * (π * D / (8 * n)) - L| < ε) ∧
             D < L ∧ L < π * D / 2 := by
  sorry

end quarter_circles_sum_limit_l3810_381071


namespace square_roots_theorem_l3810_381012

theorem square_roots_theorem (a : ℝ) :
  (3 - a) ^ 2 = (2 * a + 1) ^ 2 → (3 - a) ^ 2 = 49 := by
  sorry

end square_roots_theorem_l3810_381012


namespace factorization_equality_l3810_381036

theorem factorization_equality (x y : ℝ) : 
  4 * (x - y + 1) + y * (y - 2 * x) = (y - 2) * (y - 2 - 2 * x) := by
  sorry

end factorization_equality_l3810_381036


namespace provider_choice_count_l3810_381003

/-- The total number of service providers --/
def total_providers : ℕ := 25

/-- The number of providers available to the youngest child --/
def restricted_providers : ℕ := 15

/-- The number of children --/
def num_children : ℕ := 4

/-- The number of ways to choose service providers for the children --/
def choose_providers : ℕ := total_providers * (total_providers - 1) * (total_providers - 2) * restricted_providers

theorem provider_choice_count :
  choose_providers = 207000 :=
sorry

end provider_choice_count_l3810_381003


namespace square_division_theorem_l3810_381002

theorem square_division_theorem :
  ∃ (s : ℝ) (a b : ℝ) (n m : ℕ),
    s > 0 ∧ a > 0 ∧ b > 0 ∧
    b / a ≤ 1.25 ∧
    n + m = 40 ∧
    s * s = n * a * a + m * b * b :=
by sorry

end square_division_theorem_l3810_381002


namespace expensive_gimbap_count_l3810_381078

def basic_gimbap : ℕ := 2000
def tuna_gimbap : ℕ := 3500
def red_pepper_gimbap : ℕ := 3000
def beef_gimbap : ℕ := 4000
def rice_gimbap : ℕ := 3500

def gimbap_prices : List ℕ := [basic_gimbap, tuna_gimbap, red_pepper_gimbap, beef_gimbap, rice_gimbap]

def count_expensive_gimbap (prices : List ℕ) : ℕ :=
  (prices.filter (λ price => price ≥ 3500)).length

theorem expensive_gimbap_count : count_expensive_gimbap gimbap_prices = 3 := by
  sorry

end expensive_gimbap_count_l3810_381078


namespace prob_two_non_defective_pens_l3810_381032

/-- The probability of selecting two non-defective pens from a box of 8 pens, where 2 are defective -/
theorem prob_two_non_defective_pens (total_pens : ℕ) (defective_pens : ℕ) (selected_pens : ℕ) :
  total_pens = 8 →
  defective_pens = 2 →
  selected_pens = 2 →
  (total_pens - defective_pens : ℚ) / total_pens *
  ((total_pens - defective_pens - 1 : ℚ) / (total_pens - 1)) = 15 / 28 :=
by sorry

end prob_two_non_defective_pens_l3810_381032


namespace quadratic_roots_at_minimum_l3810_381043

/-- Given a quadratic function y = ax² + bx + c with a ≠ 0 and its lowest point at (1, -1),
    the roots of ax² + bx + c = -1 are both equal to 1. -/
theorem quadratic_roots_at_minimum (a b c : ℝ) (ha : a ≠ 0) :
  (∀ x, a * x^2 + b * x + c ≥ a * 1^2 + b * 1 + c) →
  (a * 1^2 + b * 1 + c = -1) →
  (∀ x, a * x^2 + b * x + c = -1 → x = 1) :=
by sorry

end quadratic_roots_at_minimum_l3810_381043


namespace max_sum_cubes_max_sum_cubes_achieved_l3810_381006

theorem max_sum_cubes (a b c d e : ℝ) (h : a^2 + b^2 + c^2 + d^2 + e^2 = 5) :
  a^3 + b^3 + c^3 + d^3 + e^3 ≤ 5 * Real.sqrt 5 :=
by sorry

theorem max_sum_cubes_achieved (h : ∃ a b c d e : ℝ, a^2 + b^2 + c^2 + d^2 + e^2 = 5 ∧ a^3 + b^3 + c^3 + d^3 + e^3 = 5 * Real.sqrt 5) :
  ∃ a b c d e : ℝ, a^2 + b^2 + c^2 + d^2 + e^2 = 5 ∧ a^3 + b^3 + c^3 + d^3 + e^3 = 5 * Real.sqrt 5 :=
by sorry

end max_sum_cubes_max_sum_cubes_achieved_l3810_381006


namespace expression_evaluation_l3810_381072

theorem expression_evaluation :
  let x : ℚ := 2
  let y : ℚ := -1/3
  (4*x^2 - 2*x*y + y^2) - 3*(x^2 - x*y + 5*y^2) = 16/9 := by
  sorry

end expression_evaluation_l3810_381072


namespace tan_75_deg_l3810_381076

/-- Proves that tan 75° = 2 + √3 given tan 60° and tan 15° -/
theorem tan_75_deg (tan_60_deg : Real.tan (60 * π / 180) = Real.sqrt 3)
                   (tan_15_deg : Real.tan (15 * π / 180) = 2 - Real.sqrt 3) :
  Real.tan (75 * π / 180) = 2 + Real.sqrt 3 := by
  sorry

end tan_75_deg_l3810_381076


namespace labourerPayCorrect_l3810_381067

/-- Calculates the total amount received by a labourer given the engagement conditions and absence -/
def labourerPay (totalDays : ℕ) (payRate : ℚ) (fineRate : ℚ) (absentDays : ℕ) : ℚ :=
  let workedDays := totalDays - absentDays
  let totalEarned := (workedDays : ℚ) * payRate
  let totalFine := (absentDays : ℚ) * fineRate
  totalEarned - totalFine

/-- The labourer's pay calculation is correct for the given conditions -/
theorem labourerPayCorrect :
  labourerPay 25 2 0.5 5 = 37.5 := by
  sorry

#eval labourerPay 25 2 0.5 5

end labourerPayCorrect_l3810_381067


namespace xy_less_18_implies_x_less_2_or_y_less_9_l3810_381038

theorem xy_less_18_implies_x_less_2_or_y_less_9 :
  ∀ x y : ℝ, x * y < 18 → x < 2 ∨ y < 9 := by
  sorry

end xy_less_18_implies_x_less_2_or_y_less_9_l3810_381038


namespace expression_evaluation_l3810_381007

theorem expression_evaluation (x : ℝ) (hx : x ≠ 0) :
  Real.sqrt (1 + ((x^6 - 2) / (3 * x^3))^2) = 
  (Real.sqrt ((x^6 + 4) * (x^6 + 1))) / (3 * x^3) := by
  sorry

end expression_evaluation_l3810_381007


namespace park_orchid_bushes_after_planting_l3810_381079

/-- The number of orchid bushes in the park after planting -/
def total_orchid_bushes (current : ℕ) (newly_planted : ℕ) : ℕ :=
  current + newly_planted

/-- Theorem: The park will have 35 orchid bushes after planting -/
theorem park_orchid_bushes_after_planting :
  total_orchid_bushes 22 13 = 35 := by
  sorry

end park_orchid_bushes_after_planting_l3810_381079


namespace total_students_at_competition_l3810_381045

/-- The number of students from each school at a science fair competition --/
structure SchoolAttendance where
  quantum : ℕ
  schrodinger : ℕ
  einstein : ℕ
  newton : ℕ
  galileo : ℕ
  pascal : ℕ
  faraday : ℕ

/-- The conditions of the science fair competition --/
def scienceFairConditions (s : SchoolAttendance) : Prop :=
  s.quantum = 90 ∧
  s.schrodinger = (2 * s.quantum) / 3 ∧
  s.einstein = (4 * s.schrodinger) / 9 ∧
  s.newton = (5 * s.einstein) / 12 ∧
  s.galileo = (11 * s.newton) / 20 ∧
  s.pascal = (13 * s.galileo) / 50 ∧
  s.faraday = 4 * (s.quantum + s.schrodinger + s.einstein + s.newton + s.galileo + s.pascal)

/-- The theorem stating the total number of students at the competition --/
theorem total_students_at_competition (s : SchoolAttendance) 
  (h : scienceFairConditions s) : 
  s.quantum + s.schrodinger + s.einstein + s.newton + s.galileo + s.pascal + s.faraday = 980 := by
  sorry

end total_students_at_competition_l3810_381045


namespace circle_equation_l3810_381033

-- Define the circle
def Circle (center : ℝ × ℝ) (radius : ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the points M and N
def M : ℝ × ℝ := (-2, 2)
def N : ℝ × ℝ := (-1, -1)

-- Define the line equation x - y - 1 = 0
def LineEquation (p : ℝ × ℝ) : Prop := p.1 - p.2 - 1 = 0

-- Theorem statement
theorem circle_equation :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    LineEquation center ∧
    M ∈ Circle center radius ∧
    N ∈ Circle center radius ∧
    center = (3, 2) ∧
    radius = 5 :=
  sorry

end circle_equation_l3810_381033


namespace area_of_JKLMNO_l3810_381054

/-- Represents a polygon with 6 vertices -/
structure Hexagon :=
  (J K L M N O : ℝ × ℝ)

/-- Represents a point in 2D space -/
def Point := ℝ × ℝ

/-- Calculate the area of a rectangle given its width and height -/
def rectangleArea (width height : ℝ) : ℝ := width * height

/-- The given polygon JKLMNO -/
def polygon : Hexagon := sorry

/-- The intersection point P -/
def P : Point := sorry

/-- Theorem: The area of polygon JKLMNO is 62 square units -/
theorem area_of_JKLMNO : 
  let JK : ℝ := 8
  let KL : ℝ := 10
  let OP : ℝ := 6
  let PM : ℝ := 3
  let area_JKLMNP := rectangleArea JK KL
  let area_PMNO := rectangleArea PM OP
  area_JKLMNP - area_PMNO = 62 := by sorry

end area_of_JKLMNO_l3810_381054


namespace runners_in_picture_probability_l3810_381090

/-- Represents a runner on a circular track -/
structure Runner where
  lapTime : ℝ
  direction : Bool  -- True for counterclockwise, False for clockwise

/-- Represents the track and photograph setup -/
structure TrackSetup where
  rachelLapTime : ℝ
  robertLapTime : ℝ
  totalTime : ℝ
  photographerPosition : ℝ
  pictureWidth : ℝ

/-- Calculates the probability of both runners being in the picture -/
def probabilityBothInPicture (setup : TrackSetup) : ℝ :=
  sorry  -- Proof omitted

theorem runners_in_picture_probability (setup : TrackSetup) 
  (h1 : setup.rachelLapTime = 75)
  (h2 : setup.robertLapTime = 100)
  (h3 : setup.totalTime = 12 * 60)
  (h4 : setup.photographerPosition = 1/3)
  (h5 : setup.pictureWidth = 1/5) :
  probabilityBothInPicture setup = 4/15 := by
  sorry

#check runners_in_picture_probability

end runners_in_picture_probability_l3810_381090


namespace cube_volume_from_surface_area_l3810_381025

theorem cube_volume_from_surface_area :
  ∀ (side : ℝ), 
    side > 0 →
    6 * side^2 = 486 →
    side^3 = 729 :=
by
  sorry

end cube_volume_from_surface_area_l3810_381025


namespace multiples_of_5_or_7_not_35_l3810_381089

def count_multiples (n : ℕ) (d : ℕ) : ℕ := (n / d : ℕ)

theorem multiples_of_5_or_7_not_35 : 
  (count_multiples 3000 5) + (count_multiples 3000 7) - (count_multiples 3000 35) = 943 := by
  sorry

end multiples_of_5_or_7_not_35_l3810_381089


namespace pizza_order_count_l3810_381042

theorem pizza_order_count (slices_per_pizza : ℕ) (total_slices : ℕ) (h1 : slices_per_pizza = 8) (h2 : total_slices = 168) :
  total_slices / slices_per_pizza = 21 := by
  sorry

end pizza_order_count_l3810_381042


namespace blue_to_yellow_ratio_l3810_381009

/-- Represents the number of fish of each color in the aquarium -/
structure FishCount where
  yellow : ℕ
  blue : ℕ
  green : ℕ
  other : ℕ

/-- The conditions of the aquarium -/
def aquariumConditions (f : FishCount) : Prop :=
  f.yellow = 12 ∧
  f.green = 2 * f.yellow ∧
  f.yellow + f.blue + f.green + f.other = 42

/-- The theorem stating the ratio of blue to yellow fish -/
theorem blue_to_yellow_ratio (f : FishCount) 
  (h : aquariumConditions f) : 
  f.blue * 2 = f.yellow := by sorry

end blue_to_yellow_ratio_l3810_381009


namespace integer_pair_conditions_l3810_381058

theorem integer_pair_conditions (a b : ℕ+) : 
  (∃ k : ℕ, a^3 = k * b^2) ∧ 
  (∃ m : ℕ, b - 1 = m * (a - 1)) → 
  (a = b) ∨ (b = 1) := by
sorry

end integer_pair_conditions_l3810_381058


namespace quadratic_equation_solution_l3810_381027

theorem quadratic_equation_solution (a b : ℕ+) :
  (∃ x : ℝ, x^2 + 14*x = 24 ∧ x > 0 ∧ x = Real.sqrt a - b) →
  a + b = 80 := by
sorry

end quadratic_equation_solution_l3810_381027


namespace factorization_problem_l3810_381070

theorem factorization_problem (C D : ℤ) :
  (∀ y : ℝ, 15 * y^2 - 76 * y + 48 = (C * y - 16) * (D * y - 3)) →
  C * D + C = 20 := by
  sorry

end factorization_problem_l3810_381070


namespace regina_farm_correct_l3810_381011

/-- Represents the farm animals and their selling prices -/
structure Farm where
  cows : ℕ
  pigs : ℕ
  cow_price : ℕ
  pig_price : ℕ

/-- Regina's farm satisfying the given conditions -/
def regina_farm : Farm where
  cows := 20  -- We'll prove this is correct
  pigs := 80  -- Four times the number of cows
  cow_price := 800
  pig_price := 400

/-- The total sale value of all animals on the farm -/
def total_sale_value (f : Farm) : ℕ :=
  f.cows * f.cow_price + f.pigs * f.pig_price

theorem regina_farm_correct :
  regina_farm.pigs = 4 * regina_farm.cows ∧
  total_sale_value regina_farm = 48000 := by
  sorry

#eval regina_farm.cows  -- Should output 20

end regina_farm_correct_l3810_381011


namespace extended_segment_endpoint_l3810_381086

/-- Given a segment with endpoints A(-3, 5) and B(9, -1) extended through B to point C,
    where BC = 1/2 * AB, prove that the coordinates of C are (15, -4). -/
theorem extended_segment_endpoint (A B C : ℝ × ℝ) : 
  A = (-3, 5) →
  B = (9, -1) →
  C - B = (1/2 : ℝ) • (B - A) →
  C = (15, -4) := by
  sorry

end extended_segment_endpoint_l3810_381086


namespace sqrt_calculation_l3810_381015

theorem sqrt_calculation : 
  Real.sqrt 48 / Real.sqrt 3 + Real.sqrt (1/2) * Real.sqrt 12 - Real.sqrt 24 = 4 - Real.sqrt 6 := by
  sorry

end sqrt_calculation_l3810_381015


namespace distinct_convex_polygons_l3810_381057

/-- Represents a triangle with side lengths --/
structure Triangle :=
  (side1 side2 side3 : ℝ)

/-- Represents a convex polygon --/
structure ConvexPolygon :=
  (vertices : List (ℝ × ℝ))

/-- Checks if a polygon is convex --/
def isConvex (p : ConvexPolygon) : Prop :=
  sorry

/-- Counts the number of distinct convex polygons that can be formed --/
def countConvexPolygons (triangles : List Triangle) : ℕ :=
  sorry

/-- The main theorem --/
theorem distinct_convex_polygons :
  let triangles : List Triangle := [
    ⟨3, 3, 3⟩, ⟨3, 3, 3⟩,  -- Two equilateral triangles
    ⟨3, 4, 5⟩, ⟨3, 4, 5⟩   -- Two scalene triangles
  ]
  countConvexPolygons triangles = 16 := by
  sorry

end distinct_convex_polygons_l3810_381057


namespace fraction_inequality_l3810_381097

theorem fraction_inequality (x : ℝ) : (x + 4) / (x^2 + 4*x + 13) ≥ 0 ↔ x ≥ -4 := by
  sorry

end fraction_inequality_l3810_381097


namespace inequality_proof_l3810_381081

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum_squares : a^2 + b^2 + c^2 = 1) : 
  Real.sqrt (1/a - a) + Real.sqrt (1/b - b) + Real.sqrt (1/c - c) ≥ 
  Real.sqrt (2*a) + Real.sqrt (2*b) + Real.sqrt (2*c) := by
  sorry

end inequality_proof_l3810_381081


namespace linear_function_unique_l3810_381031

/-- A function f: ℝ → ℝ is increasing if for all x, y ∈ ℝ, x < y implies f(x) < f(y) -/
def Increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

theorem linear_function_unique
  (f : ℝ → ℝ)
  (h1 : ∀ x, f (f x) = 4 * x + 6)
  (h2 : Increasing f) :
  ∀ x, f x = 2 * x + 2 :=
sorry

end linear_function_unique_l3810_381031


namespace divisibility_pairs_l3810_381028

def satisfies_condition (a b : ℕ) : Prop :=
  (a + 1) % b = 0 ∧ (b + 1) % a = 0

theorem divisibility_pairs :
  ∀ a b : ℕ, satisfies_condition a b ↔ ((a = 1 ∧ b = 1) ∨ (a = 1 ∧ b = 2) ∨ (a = 2 ∧ b = 3)) :=
by sorry

end divisibility_pairs_l3810_381028


namespace quadratic_reducible_conditions_l3810_381052

def is_quadratic_or_reducible (a b : ℚ) : Prop :=
  ∃ (p q r : ℚ), ∀ x : ℚ, x ≠ 1 ∧ x ≠ 2 ∧ x ≠ 3 ∧ x ≠ 4 ∧ x ≠ 5 →
    (a / (1 - x) - 2 / (2 - x) + 3 / (3 - x) - 4 / (4 - x) + b / (5 - x) = 0) ↔
    (p * x^2 + q * x + r = 0)

theorem quadratic_reducible_conditions :
  ∀ a b : ℚ, is_quadratic_or_reducible a b ↔
    ((a, b) = (1, 2) ∨
     (a, b) = (13/48, 178/48) ∨
     (a, b) = (9/14, 5/2) ∨
     (a, b) = (1/2, 5/2) ∨
     (a, b) = (0, 0)) := by sorry

end quadratic_reducible_conditions_l3810_381052


namespace set_union_problem_l3810_381035

theorem set_union_problem (a b : ℕ) : 
  let A : Set ℕ := {5, 2^a}
  let B : Set ℕ := {a, b}
  A ∩ B = {8} →
  A ∪ B = {3, 5, 8} := by
sorry

end set_union_problem_l3810_381035


namespace polynomial_factorization_l3810_381046

theorem polynomial_factorization (x : ℤ) : 
  (x^3 - x^2 + 2*x - 1) * (x^3 - x - 1) = x^6 - x^5 + x^4 - x^3 - x^2 - x + 1 := by
  sorry

end polynomial_factorization_l3810_381046


namespace sin_30_degrees_l3810_381091

/-- Sine of 30 degrees is 1/2 -/
theorem sin_30_degrees : Real.sin (π / 6) = 1 / 2 := by
  sorry

end sin_30_degrees_l3810_381091


namespace half_angle_in_second_quadrant_l3810_381014

open Real

/-- An angle is in the third quadrant if it's between π and 3π/2 -/
def in_third_quadrant (θ : ℝ) : Prop := π < θ ∧ θ < 3*π/2

/-- An angle is in the second quadrant if it's between π/2 and π -/
def in_second_quadrant (θ : ℝ) : Prop := π/2 < θ ∧ θ < π

theorem half_angle_in_second_quadrant (θ : ℝ) 
  (h1 : in_third_quadrant θ) 
  (h2 : |cos θ| = -cos (θ/2)) : 
  in_second_quadrant (θ/2) := by
  sorry

end half_angle_in_second_quadrant_l3810_381014


namespace multiple_of_number_l3810_381092

theorem multiple_of_number (n : ℝ) (h : n = 6) : ∃ k : ℝ, 3 * n - 6 = k * n ∧ k = 2 := by
  sorry

end multiple_of_number_l3810_381092


namespace expected_value_fair_12_sided_die_l3810_381077

def fair_12_sided_die : Finset ℕ := Finset.range 12

theorem expected_value_fair_12_sided_die : 
  (fair_12_sided_die.sum (λ x => (x + 1) * (1 : ℚ)) / 12) = (13 : ℚ) / 2 := by
  sorry

end expected_value_fair_12_sided_die_l3810_381077


namespace not_prime_sum_products_l3810_381049

theorem not_prime_sum_products (a b c d : ℤ) 
  (h1 : a > b) (h2 : b > c) (h3 : c > d) (h4 : d > 0) 
  (h5 : a * c + b * d = (b + d + a - c) * (b + d - a + c)) : 
  ¬ Prime (a * b + c * d) := by
sorry

end not_prime_sum_products_l3810_381049


namespace vehicle_value_last_year_l3810_381063

theorem vehicle_value_last_year 
  (value_this_year : ℝ) 
  (value_ratio : ℝ) 
  (h1 : value_this_year = 16000)
  (h2 : value_ratio = 0.8)
  (h3 : value_this_year = value_ratio * value_last_year) :
  value_last_year = 20000 :=
by
  sorry

end vehicle_value_last_year_l3810_381063


namespace power_function_quadrant_propositions_l3810_381030

-- Define a power function
def is_power_function (f : ℝ → ℝ) : Prop := 
  ∃ a b : ℝ, ∀ x : ℝ, f x = a * x ^ b

-- Define the property of not passing through the fourth quadrant
def not_in_fourth_quadrant (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f x = y → ¬(x > 0 ∧ y < 0)

-- The main theorem
theorem power_function_quadrant_propositions :
  let P : (ℝ → ℝ) → Prop := λ f => is_power_function f → not_in_fourth_quadrant f
  let contrapositive : (ℝ → ℝ) → Prop := λ f => ¬(not_in_fourth_quadrant f) → ¬(is_power_function f)
  let converse : (ℝ → ℝ) → Prop := λ f => not_in_fourth_quadrant f → is_power_function f
  let inverse : (ℝ → ℝ) → Prop := λ f => ¬(is_power_function f) → ¬(not_in_fourth_quadrant f)
  (∀ f : ℝ → ℝ, P f) ∧
  (∀ f : ℝ → ℝ, contrapositive f) ∧
  ¬(∀ f : ℝ → ℝ, converse f) ∧
  ¬(∀ f : ℝ → ℝ, inverse f) :=
by sorry

end power_function_quadrant_propositions_l3810_381030


namespace marj_wallet_remaining_l3810_381008

/-- Calculates the remaining money in Marj's wallet after expenses --/
def remaining_money (initial_usd : ℚ) (initial_euro : ℚ) (initial_pound : ℚ) 
  (euro_to_usd : ℚ) (pound_to_usd : ℚ) (cake_cost : ℚ) (gift_cost : ℚ) (donation : ℚ) : ℚ :=
  initial_usd + initial_euro * euro_to_usd + initial_pound * pound_to_usd - cake_cost - gift_cost - donation

/-- Theorem stating that Marj will have $64.40 left in her wallet after expenses --/
theorem marj_wallet_remaining : 
  remaining_money 81.5 10 5 1.18 1.32 17.5 12.7 5.3 = 64.4 := by
  sorry

end marj_wallet_remaining_l3810_381008


namespace smallest_three_digit_palindrome_non_palindromic_product_l3810_381085

/-- A function that checks if a number is a three-digit palindrome -/
def isThreeDigitPalindrome (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ (n / 100 = n % 10)

/-- A function that checks if a number is a five-digit palindrome -/
def isFiveDigitPalindrome (n : ℕ) : Prop :=
  10000 ≤ n ∧ n ≤ 99999 ∧ (n / 10000 = n % 10) ∧ ((n / 1000) % 10 = (n / 10) % 10)

/-- The main theorem stating that 707 is the smallest three-digit palindrome
    whose product with 103 is not a five-digit palindrome -/
theorem smallest_three_digit_palindrome_non_palindromic_product :
  (∀ n : ℕ, isThreeDigitPalindrome n ∧ n < 707 → isFiveDigitPalindrome (n * 103)) ∧
  isThreeDigitPalindrome 707 ∧
  ¬isFiveDigitPalindrome (707 * 103) :=
sorry

end smallest_three_digit_palindrome_non_palindromic_product_l3810_381085

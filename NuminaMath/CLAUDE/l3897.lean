import Mathlib

namespace jorge_age_proof_l3897_389761

/-- Jorge's age in 2005 -/
def jorge_age_2005 : ℕ := 16

/-- Simon's age in 2010 -/
def simon_age_2010 : ℕ := 45

/-- Age difference between Simon and Jorge -/
def age_difference : ℕ := 24

/-- Years between 2005 and 2010 -/
def years_difference : ℕ := 5

theorem jorge_age_proof :
  jorge_age_2005 = simon_age_2010 - years_difference - age_difference :=
by sorry

end jorge_age_proof_l3897_389761


namespace y₁_less_than_y₂_l3897_389721

/-- The parabola equation -/
def parabola (x : ℝ) : ℝ := -x^2 - 4*x + 1

/-- y₁ is the y-coordinate of the point (-3, y₁) on the parabola -/
def y₁ : ℝ := parabola (-3)

/-- y₂ is the y-coordinate of the point (-2, y₂) on the parabola -/
def y₂ : ℝ := parabola (-2)

/-- Theorem stating that y₁ < y₂ for the given parabola and points -/
theorem y₁_less_than_y₂ : y₁ < y₂ := by
  sorry

end y₁_less_than_y₂_l3897_389721


namespace modulus_of_complex_difference_l3897_389743

theorem modulus_of_complex_difference (z : ℂ) : z = -1 - Complex.I → Complex.abs (2 - z) = Real.sqrt 10 := by
  sorry

end modulus_of_complex_difference_l3897_389743


namespace large_square_perimeter_l3897_389775

-- Define the original square's perimeter
def original_perimeter : ℝ := 56

-- Define the number of parts the original square is divided into
def division_parts : ℕ := 4

-- Define the number of small squares used to form the large square
def small_squares : ℕ := 441

-- Theorem statement
theorem large_square_perimeter (original_perimeter : ℝ) (division_parts : ℕ) (small_squares : ℕ) :
  original_perimeter = 56 ∧ 
  division_parts = 4 ∧ 
  small_squares = 441 →
  (original_perimeter / (4 * Real.sqrt (small_squares : ℝ))) * 
  (4 * Real.sqrt (small_squares : ℝ)) = 588 := by
  sorry


end large_square_perimeter_l3897_389775


namespace right_triangle_ratio_minimum_l3897_389702

theorem right_triangle_ratio_minimum (a b c : ℝ) (h_right : a^2 + b^2 = c^2) (h_pos : c > 0) :
  (a^2 + b) / c^2 ≥ 1 ∧ ∃ (a₀ b₀ c₀ : ℝ), a₀^2 + b₀^2 = c₀^2 ∧ c₀ > 0 ∧ (a₀^2 + b₀) / c₀^2 = 1 :=
sorry

end right_triangle_ratio_minimum_l3897_389702


namespace train_length_l3897_389795

/-- Given a train that passes a pole in 11 seconds and a 120 m long platform in 22 seconds, 
    its length is 120 meters. -/
theorem train_length (pole_time : ℝ) (platform_time : ℝ) (platform_length : ℝ) 
    (h1 : pole_time = 11)
    (h2 : platform_time = 22)
    (h3 : platform_length = 120) : ℝ :=
  by sorry

#check train_length

end train_length_l3897_389795


namespace smallest_next_divisor_after_493_l3897_389725

theorem smallest_next_divisor_after_493 (n : ℕ) : 
  1000 ≤ n ∧ n < 10000 ∧  -- n is a 4-digit number
  Even n ∧                -- n is even
  n % 493 = 0 →           -- 493 is a divisor of n
  (∃ (d : ℕ), d > 493 ∧ n % d = 0 ∧ d ≤ 510 ∧ 
    ∀ (k : ℕ), 493 < k ∧ k < d → n % k ≠ 0) :=
by sorry

end smallest_next_divisor_after_493_l3897_389725


namespace total_rainfall_equals_1368_l3897_389700

def average_monthly_rainfall (year : ℕ) : ℝ :=
  35 + 3 * (year - 2010)

def yearly_rainfall (year : ℕ) : ℝ :=
  12 * average_monthly_rainfall year

def total_rainfall_2010_to_2012 : ℝ :=
  yearly_rainfall 2010 + yearly_rainfall 2011 + yearly_rainfall 2012

theorem total_rainfall_equals_1368 :
  total_rainfall_2010_to_2012 = 1368 := by sorry

end total_rainfall_equals_1368_l3897_389700


namespace no_fixed_points_implies_a_range_l3897_389786

/-- A quadratic function f(x) = x^2 + ax + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 1

/-- The property of having no fixed points -/
def has_no_fixed_points (a : ℝ) : Prop :=
  ∀ x : ℝ, f a x ≠ x

/-- The main theorem -/
theorem no_fixed_points_implies_a_range (a : ℝ) :
  has_no_fixed_points a → -1 < a ∧ a < 3 :=
sorry

end no_fixed_points_implies_a_range_l3897_389786


namespace area_of_triangle_qpo_l3897_389705

/-- Represents a parallelogram ABCD with specific properties -/
structure SpecialParallelogram where
  -- The area of the parallelogram
  area : ℝ
  -- DP bisects BC
  dp_bisects_bc : Bool
  -- CQ bisects AD
  cq_bisects_ad : Bool
  -- DP divides triangle BCD into regions of area k/4 and 3k/4
  dp_divides_bcd : Bool

/-- Theorem stating the area of triangle QPO in the special parallelogram -/
theorem area_of_triangle_qpo (ABCD : SpecialParallelogram) :
  let k := ABCD.area
  let area_qpo := (9 : ℝ) / 8 * k
  ABCD.dp_bisects_bc ∧ ABCD.cq_bisects_ad ∧ ABCD.dp_divides_bcd →
  area_qpo = (9 : ℝ) / 8 * k :=
by
  sorry


end area_of_triangle_qpo_l3897_389705


namespace reflect_L_shape_is_mirrored_l3897_389747

/-- Represents a 2D shape -/
structure Shape :=
  (points : Set (ℝ × ℝ))

/-- Represents a vertical line -/
structure VerticalLine :=
  (x : ℝ)

/-- Defines an L-like shape -/
def LLikeShape : Shape :=
  sorry

/-- Defines a mirrored L-like shape -/
def MirroredLLikeShape : Shape :=
  sorry

/-- Reflects a point across a vertical line -/
def reflectPoint (p : ℝ × ℝ) (line : VerticalLine) : ℝ × ℝ :=
  (2 * line.x - p.1, p.2)

/-- Reflects a shape across a vertical line -/
def reflectShape (s : Shape) (line : VerticalLine) : Shape :=
  ⟨s.points.image (λ p => reflectPoint p line)⟩

/-- Theorem: Reflecting an L-like shape across a vertical line results in a mirrored L-like shape -/
theorem reflect_L_shape_is_mirrored (line : VerticalLine) :
  reflectShape LLikeShape line = MirroredLLikeShape :=
sorry

end reflect_L_shape_is_mirrored_l3897_389747


namespace cos_45_degrees_l3897_389715

theorem cos_45_degrees : Real.cos (π / 4) = Real.sqrt 2 / 2 := by
  sorry

end cos_45_degrees_l3897_389715


namespace product_bounds_l3897_389765

theorem product_bounds (x y z : Real) 
  (h1 : x ≥ y) (h2 : y ≥ z) (h3 : z ≥ π/12) 
  (h4 : x + y + z = π/2) : 
  1/8 ≤ Real.cos x * Real.sin y * Real.cos z ∧ 
  Real.cos x * Real.sin y * Real.cos z ≤ (2 + Real.sqrt 3) / 8 := by
  sorry

end product_bounds_l3897_389765


namespace continued_fraction_evaluation_l3897_389756

theorem continued_fraction_evaluation :
  2 + 3 / (4 + 5 / (6 + 7/8)) = 137/52 := by
  sorry

end continued_fraction_evaluation_l3897_389756


namespace g_values_l3897_389791

-- Define the function g
def g (x : ℝ) : ℝ := -2 * x^2 - 3 * x + 1

-- State the theorem
theorem g_values : g (-1) = 2 ∧ g (-2) = -1 := by
  sorry

end g_values_l3897_389791


namespace oil_leak_during_repair_l3897_389713

/-- Represents the oil leak scenario -/
structure OilLeak where
  initial_leak : ℝ
  initial_time : ℝ
  repair_time : ℝ
  rate_reduction : ℝ
  total_leak : ℝ

/-- Calculates the amount of oil leaked during repair -/
def leak_during_repair (scenario : OilLeak) : ℝ :=
  let initial_rate := scenario.initial_leak / scenario.initial_time
  let reduced_rate := initial_rate * scenario.rate_reduction
  scenario.total_leak - scenario.initial_leak

/-- Theorem stating the amount of oil leaked during repair -/
theorem oil_leak_during_repair :
  let scenario : OilLeak := {
    initial_leak := 2475,
    initial_time := 7,
    repair_time := 5,
    rate_reduction := 0.75,
    total_leak := 6206
  }
  leak_during_repair scenario = 3731 := by sorry

end oil_leak_during_repair_l3897_389713


namespace optimal_racket_purchase_l3897_389780

/-- The optimal purchasing plan for badminton rackets -/
theorem optimal_racket_purchase 
  (total_cost : ℕ) 
  (num_pairs : ℕ) 
  (price_diff : ℕ) 
  (discount_a : ℚ) 
  (discount_b : ℕ) 
  (max_cost : ℕ) 
  (min_a : ℕ) :
  total_cost = num_pairs * (price_a + price_b) ∧
  price_b = price_a - price_diff ∧
  new_price_a = price_a * discount_a ∧
  new_price_b = price_b - discount_b ∧
  (∀ m : ℕ, m ≥ min_a → m ≤ 50 → 
    new_price_a * m + new_price_b * (50 - m) ≤ max_cost) →
  optimal_m = 38 ∧ 
  optimal_cost = new_price_a * optimal_m + new_price_b * (50 - optimal_m) ∧
  (∀ m : ℕ, m ≥ min_a → m ≤ 50 → 
    new_price_a * m + new_price_b * (50 - m) ≥ optimal_cost) :=
by
  sorry

#check optimal_racket_purchase 1300 20 15 (4/5) 4 1500 38

end optimal_racket_purchase_l3897_389780


namespace triangle_perimeter_bound_l3897_389745

theorem triangle_perimeter_bound (a b s : ℝ) : 
  a = 7 → b = 23 → (a + b > s ∧ a + s > b ∧ b + s > a) → 
  ∃ (n : ℕ), n = 60 ∧ ∀ (p : ℝ), p = a + b + s → (n : ℝ) > p ∧ 
  ∀ (m : ℕ), (m : ℝ) > p → m ≥ n :=
sorry

end triangle_perimeter_bound_l3897_389745


namespace sum_of_powers_and_reciprocals_is_integer_l3897_389716

theorem sum_of_powers_and_reciprocals_is_integer
  (x : ℝ)
  (h : ∃ (k : ℤ), x + 1 / x = k)
  (n : ℕ)
  : ∃ (m : ℤ), x^n + 1 / x^n = m :=
by
  sorry

end sum_of_powers_and_reciprocals_is_integer_l3897_389716


namespace intersection_of_M_and_N_l3897_389746

-- Define the sets M and N
def M : Set ℝ := {x | x > -1}
def N : Set ℝ := {x | x^2 + 2*x < 0}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {x : ℝ | -1 < x ∧ x < 0} := by
  sorry

end intersection_of_M_and_N_l3897_389746


namespace S_is_circle_l3897_389744

-- Define the set of complex numbers satisfying |z-3|=1
def S : Set ℂ := {z : ℂ | Complex.abs (z - 3) = 1}

-- Theorem statement
theorem S_is_circle : 
  ∃ (center : ℂ) (radius : ℝ), S = {z : ℂ | Complex.abs (z - center) = radius} ∧ radius > 0 :=
by sorry

end S_is_circle_l3897_389744


namespace triangle_inequality_l3897_389734

theorem triangle_inequality (a : ℝ) : a ≥ 1 →
  (3 * a + (a + 1) ≥ 2) ∧
  (3 * (a - 1) + 2 * a ≥ 2) ∧
  (3 * 1 + 3 ≥ 2) := by
  sorry

end triangle_inequality_l3897_389734


namespace mr_green_potato_yield_l3897_389729

/-- Represents the dimensions of a rectangular garden in steps -/
structure GardenDimensions where
  length : ℕ
  width : ℕ

/-- Calculates the expected potato yield from a garden -/
def expected_potato_yield (garden : GardenDimensions) (step_length : ℕ) (yield_per_sqft : ℚ) : ℚ :=
  (garden.length * step_length * (garden.width * step_length) : ℚ) * yield_per_sqft

/-- Theorem stating the expected potato yield for Mr. Green's garden -/
theorem mr_green_potato_yield :
  let garden := GardenDimensions.mk 15 20
  let step_length := 2
  let yield_per_sqft := 1/2
  expected_potato_yield garden step_length yield_per_sqft = 600 := by
  sorry

end mr_green_potato_yield_l3897_389729


namespace expression_simplification_and_ratio_l3897_389794

theorem expression_simplification_and_ratio :
  let expr := (6 * m + 4 * n + 12) / 4
  let a := 3/2
  let b := 1
  let c := 3
  expr = a * m + b * n + c ∧ (a + b + c) / c = 11/6 :=
by sorry

end expression_simplification_and_ratio_l3897_389794


namespace circle_circumference_irrational_l3897_389788

/-- The circumference of a circle with rational radius is irrational -/
theorem circle_circumference_irrational (r : ℚ) : 
  Irrational (2 * Real.pi * (r : ℝ)) :=
sorry

end circle_circumference_irrational_l3897_389788


namespace unseen_sum_is_21_l3897_389711

/-- Represents a standard six-sided die -/
structure Die :=
  (faces : Fin 6 → Nat)
  (opposite_sum : ∀ (i : Fin 6), faces i + faces (5 - i) = 7)

/-- The sum of the numbers on the unseen faces of two dice -/
def unseen_sum (d1 d2 : Die) (v1 v2 v3 : Fin 6) (w1 w2 w3 : Fin 6) : Nat :=
  (7 - d1.faces v1) + (7 - d1.faces v2) + (7 - d1.faces v3) +
  (7 - d2.faces w1) + (7 - d2.faces w2) + (7 - d2.faces w3)

theorem unseen_sum_is_21 (d1 d2 : Die) :
  d1.faces 0 = 6 → d1.faces 1 = 2 → d1.faces 2 = 3 →
  d2.faces 0 = 1 → d2.faces 1 = 4 → d2.faces 2 = 5 →
  unseen_sum d1 d2 0 1 2 0 1 2 = 21 :=
by
  sorry

end unseen_sum_is_21_l3897_389711


namespace combination_sum_l3897_389724

theorem combination_sum (n : ℕ) : 
  (5 : ℚ) / 2 ≤ n ∧ n ≤ 3 → Nat.choose (2*n) (10 - 2*n) + Nat.choose (3 + n) (2*n) = 16 := by
  sorry

end combination_sum_l3897_389724


namespace haley_concert_spending_l3897_389792

/-- Calculates the total cost of concert tickets based on a pricing structure -/
def calculate_total_cost (initial_price : ℕ) (discounted_price : ℕ) (initial_quantity : ℕ) (discounted_quantity : ℕ) : ℕ :=
  initial_price * initial_quantity + discounted_price * discounted_quantity

/-- Proves that Haley's total spending on concert tickets is $27 -/
theorem haley_concert_spending :
  let initial_price : ℕ := 4
  let discounted_price : ℕ := 3
  let initial_quantity : ℕ := 3
  let discounted_quantity : ℕ := 5
  calculate_total_cost initial_price discounted_price initial_quantity discounted_quantity = 27 :=
by
  sorry

#eval calculate_total_cost 4 3 3 5

end haley_concert_spending_l3897_389792


namespace circle_intersection_and_tangent_lines_l3897_389766

-- Define the circles and point
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 4*y - 4 = 0
def circle_C1 (x y : ℝ) : Prop := (x - 3)^2 + (y - 1)^2 = 4
def point_P : ℝ × ℝ := (3, 1)

-- Define the intersection of two circles
def circles_intersect (C1 C2 : (ℝ → ℝ → Prop)) : Prop :=
  ∃ x y, C1 x y ∧ C2 x y

-- Define a tangent line to a circle passing through a point
def is_tangent_line (a b c : ℝ) (C : ℝ → ℝ → Prop) (P : ℝ × ℝ) : Prop :=
  (∀ x y, C x y → a*x + b*y + c ≠ 0) ∧
  (∃ x y, C x y ∧ a*x + b*y + c = 0) ∧
  a*(P.1) + b*(P.2) + c = 0

-- Theorem statement
theorem circle_intersection_and_tangent_lines :
  (circles_intersect circle_C circle_C1) ∧
  (is_tangent_line 0 1 (-1) circle_C point_P) ∧
  (is_tangent_line 12 5 (-41) circle_C point_P) :=
sorry

end circle_intersection_and_tangent_lines_l3897_389766


namespace total_monthly_bill_wfh_l3897_389701

-- Define the original monthly bill
def original_bill : ℝ := 60

-- Define the percentage increase
def percentage_increase : ℝ := 0.45

-- Define the additional cost for faster internet
def faster_internet_cost : ℝ := 25

-- Define the additional cost for cloud storage
def cloud_storage_cost : ℝ := 15

-- Theorem to prove the total monthly bill working from home
theorem total_monthly_bill_wfh :
  original_bill * (1 + percentage_increase) + faster_internet_cost + cloud_storage_cost = 127 := by
  sorry

end total_monthly_bill_wfh_l3897_389701


namespace ounces_per_pound_l3897_389797

-- Define constants
def pounds_per_ton : ℝ := 2500
def gunny_bag_capacity_tons : ℝ := 13
def num_packets : ℝ := 2000
def packet_weight_pounds : ℝ := 16
def packet_weight_ounces : ℝ := 4

-- Define the theorem
theorem ounces_per_pound : ∃ (x : ℝ), 
  (gunny_bag_capacity_tons * pounds_per_ton = num_packets * (packet_weight_pounds + packet_weight_ounces / x)) → 
  x = 16 := by
  sorry

end ounces_per_pound_l3897_389797


namespace negation_of_all_odd_double_even_l3897_389751

def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1
def is_even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

def A : Set ℤ := {n : ℤ | is_odd n}
def B : Set ℤ := {n : ℤ | is_even n}

theorem negation_of_all_odd_double_even :
  (¬ ∀ x ∈ A, (2 * x) ∈ B) ↔ (∃ x ∈ A, (2 * x) ∉ B) :=
sorry

end negation_of_all_odd_double_even_l3897_389751


namespace caravan_spaces_l3897_389708

theorem caravan_spaces (total_spaces : ℕ) (caravans_parked : ℕ) (spaces_left : ℕ) 
  (h1 : total_spaces = 30)
  (h2 : caravans_parked = 3)
  (h3 : spaces_left = 24)
  (h4 : total_spaces = caravans_parked * (total_spaces - spaces_left) + spaces_left) :
  total_spaces - spaces_left = 2 := by
  sorry

end caravan_spaces_l3897_389708


namespace total_fish_in_lake_l3897_389709

/-- Represents the number of fish per white duck -/
def fish_per_white_duck : ℕ := 5

/-- Represents the number of fish per black duck -/
def fish_per_black_duck : ℕ := 10

/-- Represents the number of fish per multicolor duck -/
def fish_per_multicolor_duck : ℕ := 12

/-- Represents the number of white ducks -/
def white_ducks : ℕ := 3

/-- Represents the number of black ducks -/
def black_ducks : ℕ := 7

/-- Represents the number of multicolor ducks -/
def multicolor_ducks : ℕ := 6

/-- Theorem stating that the total number of fish in the lake is 157 -/
theorem total_fish_in_lake : 
  white_ducks * fish_per_white_duck + 
  black_ducks * fish_per_black_duck + 
  multicolor_ducks * fish_per_multicolor_duck = 157 := by
  sorry

end total_fish_in_lake_l3897_389709


namespace probability_of_mathematics_letter_l3897_389717

theorem probability_of_mathematics_letter :
  let total_letters : ℕ := 26
  let unique_letters : ℕ := 8
  let probability : ℚ := unique_letters / total_letters
  probability = 4 / 13 := by
sorry

end probability_of_mathematics_letter_l3897_389717


namespace systematic_sampling_probability_l3897_389785

theorem systematic_sampling_probability 
  (total_students : ℕ) 
  (selected_students : ℕ) 
  (h1 : total_students = 52) 
  (h2 : selected_students = 10) :
  (1 - 2 / total_students) * (selected_students / (total_students - 2)) = 5 / 26 := by
  sorry

end systematic_sampling_probability_l3897_389785


namespace parabola_point_coordinate_l3897_389710

/-- Theorem: For a point on the parabola x^2 = 4y with focus (0, 1),
    if its distance from the focus is 5, then its x-coordinate is ±4. -/
theorem parabola_point_coordinate (x y : ℝ) : 
  x^2 = 4*y →  -- Point (x, y) is on the parabola
  (x - 0)^2 + (y - 1)^2 = 5^2 →  -- Distance from (x, y) to focus (0, 1) is 5
  x = 4 ∨ x = -4 :=
by sorry

end parabola_point_coordinate_l3897_389710


namespace set_A_representation_l3897_389771

def A : Set (ℝ × ℝ) := {(x, y) | 3 * x + y = 11 ∧ x - y = 1}

theorem set_A_representation : A = {(3, 2)} := by sorry

end set_A_representation_l3897_389771


namespace initial_bacteria_count_l3897_389764

/-- The number of seconds between each doubling of bacteria -/
def doubling_period : ℕ := 30

/-- The total time elapsed in seconds -/
def total_time : ℕ := 150

/-- The number of bacteria after the total time has elapsed -/
def final_bacteria_count : ℕ := 20480

/-- The number of doubling periods that have occurred -/
def num_doubling_periods : ℕ := total_time / doubling_period

theorem initial_bacteria_count : 
  ∃ (initial_count : ℕ), initial_count * (2^num_doubling_periods) = final_bacteria_count ∧ 
                          initial_count = 640 := by sorry

end initial_bacteria_count_l3897_389764


namespace ellipse_parabola_problem_l3897_389778

/-- Given an ellipse and a parabola with specific properties, prove the equation of the ellipse,
    the coordinates of a point, and the range of a certain expression. -/
theorem ellipse_parabola_problem (a b p : ℝ) (F : ℝ × ℝ) :
  a > b ∧ b > 0 ∧ p > 0 ∧  -- Conditions on a, b, and p
  (∃ (x y : ℝ), x^2/a^2 + y^2/b^2 = 1 ∧ y^2 = 2*p*x) ∧  -- C₁ and C₂ have a common point
  (F.1 - F.2 + 1)^2 / 2 = 2 ∧  -- Distance from F to x - y + 1 = 0 is √2
  (∃ (x y : ℝ), x^2/a^2 + y^2/b^2 = 1 ∧ y^2 = 2*p*x ∧ (x - 3/2)^2 + y^2 = 6) →  -- Common chord length is 2√6
  (a^2 = 9 ∧ b^2 = 8 ∧ F = (1, 0)) ∧  -- Equation of C₁ and coordinates of F
  (∀ k : ℝ, k ≠ 0 → 
    1/6 < (21*k^2 + 8)/(48*(k^2 + 1)) ∧ (21*k^2 + 8)/(48*(k^2 + 1)) ≤ 7/16) -- Range of 1/|AB| + 1/|CD|
  := by sorry

end ellipse_parabola_problem_l3897_389778


namespace multiplication_simplification_l3897_389781

theorem multiplication_simplification : 9 * (1 / 13) * 26 = 18 := by
  sorry

end multiplication_simplification_l3897_389781


namespace xy_problem_l3897_389731

theorem xy_problem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y = 25) (h4 : x / y = 36) : y = 5/6 := by
  sorry

end xy_problem_l3897_389731


namespace total_phone_cost_l3897_389730

def phone_cost : ℝ := 1000
def monthly_contract : ℝ := 200
def case_cost_percentage : ℝ := 0.20
def headphones_cost_ratio : ℝ := 0.5
def months_in_year : ℕ := 12

def total_cost : ℝ :=
  phone_cost +
  (monthly_contract * months_in_year) +
  (phone_cost * case_cost_percentage) +
  (phone_cost * case_cost_percentage * headphones_cost_ratio)

theorem total_phone_cost : total_cost = 3700 := by
  sorry

end total_phone_cost_l3897_389730


namespace no_divisible_by_four_exists_l3897_389793

theorem no_divisible_by_four_exists : 
  ¬ ∃ (B : ℕ), B < 10 ∧ (8000000 + 100000 * B + 4000 + 635 + 1) % 4 = 0 := by
sorry

end no_divisible_by_four_exists_l3897_389793


namespace max_distance_from_unit_circle_to_point_l3897_389749

theorem max_distance_from_unit_circle_to_point (z : ℂ) :
  Complex.abs z = 1 →
  (∀ w : ℂ, Complex.abs w = 1 → Complex.abs (w - (3 + 4*I)) ≤ Complex.abs (z - (3 + 4*I))) →
  Complex.abs (z - (3 + 4*I)) = 6 :=
by sorry

end max_distance_from_unit_circle_to_point_l3897_389749


namespace line_passes_through_intersections_l3897_389799

/-- First circle equation -/
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 3*x - y = 0

/-- Second circle equation -/
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + y = 0

/-- Line equation -/
def line (x y : ℝ) : Prop := x - 2*y = 0

/-- Theorem stating that the line passes through the intersection points of the circles -/
theorem line_passes_through_intersections :
  ∀ x y : ℝ, circle1 x y ∧ circle2 x y → line x y :=
by sorry

end line_passes_through_intersections_l3897_389799


namespace composite_increasing_pos_l3897_389753

/-- An odd function that is positive and increasing for negative x -/
def OddPositiveIncreasingNeg (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x < 0, 0 < f x) ∧
  (∀ x y, x < y ∧ y < 0 → f x < f y)

/-- The composite function f[f(x)] is increasing for positive x -/
theorem composite_increasing_pos 
  (f : ℝ → ℝ) 
  (h : OddPositiveIncreasingNeg f) : 
  ∀ x y, 0 < x ∧ x < y → f (f x) < f (f y) := by
sorry

end composite_increasing_pos_l3897_389753


namespace first_day_of_month_l3897_389703

-- Define days of the week
inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

-- Function to get the next day
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

-- Function to advance a day by n days
def advanceDay (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => advanceDay (nextDay d) n

-- Theorem statement
theorem first_day_of_month (d : DayOfWeek) :
  advanceDay d 23 = DayOfWeek.Tuesday → d = DayOfWeek.Sunday :=
by sorry


end first_day_of_month_l3897_389703


namespace harry_galleons_l3897_389735

theorem harry_galleons (H He R : ℕ) : 
  (H + He = 12) →
  (H + R = 120) →
  (∃ k : ℕ, H + He + R = 7 * k) →
  (H + He + R ≥ H) →
  (H + He + R ≥ He) →
  (H + He + R ≥ R) →
  (H > 0) →
  (H = 6) := by
sorry

end harry_galleons_l3897_389735


namespace colonization_combinations_l3897_389728

/-- The number of habitable planets --/
def total_planets : ℕ := 18

/-- The number of Earth-like planets --/
def earth_like_planets : ℕ := 9

/-- The number of Mars-like planets --/
def mars_like_planets : ℕ := 9

/-- The resource units required to colonize an Earth-like planet --/
def earth_like_resource : ℕ := 3

/-- The resource units required to colonize a Mars-like planet --/
def mars_like_resource : ℕ := 2

/-- The total resource units available for colonization --/
def total_resources : ℕ := 27

/-- The number of Earth-like planets that can be colonized --/
def colonized_earth_like : ℕ := 7

/-- The number of Mars-like planets that can be colonized --/
def colonized_mars_like : ℕ := 3

theorem colonization_combinations : 
  (Nat.choose earth_like_planets colonized_earth_like) * 
  (Nat.choose mars_like_planets colonized_mars_like) = 3024 :=
by sorry

end colonization_combinations_l3897_389728


namespace percentage_problem_l3897_389767

theorem percentage_problem (percentage : ℝ) : 
  (percentage * 100 - 20 = 60) → percentage = 0.8 := by
  sorry

end percentage_problem_l3897_389767


namespace kelvin_frog_paths_l3897_389704

/-- Represents a position in the coordinate plane -/
structure Position :=
  (x : ℕ) (y : ℕ)

/-- Represents a move that Kelvin can make -/
inductive Move
  | Walk : Move
  | Jump : Move

/-- Defines the possible moves Kelvin can make from a given position -/
def possibleMoves (pos : Position) : List Position :=
  [
    {x := pos.x, y := pos.y + 1},     -- Walk up
    {x := pos.x + 1, y := pos.y},     -- Walk right
    {x := pos.x + 1, y := pos.y + 1}, -- Walk diagonally
    {x := pos.x, y := pos.y + 2},     -- Jump up
    {x := pos.x + 2, y := pos.y},     -- Jump right
    {x := pos.x + 1, y := pos.y + 1}  -- Jump diagonally
  ]

/-- Counts the number of ways to reach the target position from the start position -/
def countWays (start : Position) (target : Position) : ℕ :=
  sorry

theorem kelvin_frog_paths : countWays {x := 0, y := 0} {x := 6, y := 8} = 1831830 := by
  sorry

end kelvin_frog_paths_l3897_389704


namespace graph_below_line_l3897_389750

noncomputable def f (x : ℝ) := x * Real.log x - x^2 - 1

theorem graph_below_line (x : ℝ) (h : x > 0) : Real.log x - Real.exp x + 1 < 0 := by
  sorry

end graph_below_line_l3897_389750


namespace deck_size_proof_l3897_389787

theorem deck_size_proof (r b : ℕ) : 
  (r : ℚ) / (r + b) = 2 / 5 →
  (r : ℚ) / (r + b + 6) = 1 / 3 →
  r + b = 30 := by
  sorry

end deck_size_proof_l3897_389787


namespace point_condition_l3897_389726

-- Define the unit circle ω in the xy-plane
def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the condition for right angles
def right_angle (A B C P : Point3D) : Prop :=
  (A.x - P.x) * (B.x - P.x) + (A.y - P.y) * (B.y - P.y) + (A.z - P.z) * (B.z - P.z) = 0

-- Main theorem
theorem point_condition (P : Point3D) (h_not_xy : P.z ≠ 0) :
  (∃ A B C : Point3D, 
    unit_circle A.x A.y ∧ 
    unit_circle B.x B.y ∧ 
    unit_circle C.x C.y ∧ 
    right_angle A B P C ∧ 
    right_angle A C P B ∧ 
    right_angle B C P A) →
  P.x^2 + P.y^2 + 2*P.z^2 = 1 := by
  sorry

end point_condition_l3897_389726


namespace complex_fraction_simplification_l3897_389748

theorem complex_fraction_simplification :
  (5 + 7 * Complex.I) / (2 - 3 * Complex.I) = 31/13 + 29/13 * Complex.I := by
  sorry

end complex_fraction_simplification_l3897_389748


namespace turtles_on_happy_island_l3897_389707

theorem turtles_on_happy_island :
  let lonely_island_turtles : ℕ := 25
  let happy_island_turtles : ℕ := 2 * lonely_island_turtles + 10
  happy_island_turtles = 60 :=
by sorry

end turtles_on_happy_island_l3897_389707


namespace exists_function_satisfying_conditions_l3897_389754

-- Define the properties of the function f
def satisfies_conditions (f : ℝ → ℝ) : Prop :=
  (∀ x₁ x₂ : ℝ, x₁ + x₂ = 0 → f x₁ + f x₂ = 0) ∧ 
  (∀ x t : ℝ, t > 0 → f (x + t) > f x)

-- State the theorem
theorem exists_function_satisfying_conditions : 
  ∃ f : ℝ → ℝ, satisfies_conditions f ∧ f = fun x ↦ x^3 := by
  sorry

end exists_function_satisfying_conditions_l3897_389754


namespace polynomial_sum_of_coefficients_l3897_389783

def g (a b c d : ℝ) (x : ℂ) : ℂ := x^4 + a*x^3 + b*x^2 + c*x + d

theorem polynomial_sum_of_coefficients 
  (a b c d : ℝ) 
  (h1 : g a b c d (3*Complex.I) = 0)
  (h2 : g a b c d (3 + Complex.I) = 0) :
  a + b + c + d = 49 := by
  sorry

end polynomial_sum_of_coefficients_l3897_389783


namespace only_taller_students_not_set_l3897_389777

-- Define the options
inductive SetOption
  | PrimesUpTo20
  | RootsOfEquation
  | TallerStudents
  | AllSquares

-- Define a predicate for well-defined sets
def is_well_defined_set (option : SetOption) : Prop :=
  match option with
  | SetOption.PrimesUpTo20 => true
  | SetOption.RootsOfEquation => true
  | SetOption.TallerStudents => false
  | SetOption.AllSquares => true

-- Theorem statement
theorem only_taller_students_not_set :
  ∀ (option : SetOption),
    ¬(is_well_defined_set option) ↔ option = SetOption.TallerStudents :=
by sorry

end only_taller_students_not_set_l3897_389777


namespace gcd_of_135_and_81_l3897_389784

theorem gcd_of_135_and_81 : Nat.gcd 135 81 = 27 := by
  sorry

end gcd_of_135_and_81_l3897_389784


namespace power_zero_eq_one_iff_nonzero_l3897_389779

theorem power_zero_eq_one_iff_nonzero (a : ℝ) : a ^ 0 = 1 ↔ a ≠ 0 := by sorry

end power_zero_eq_one_iff_nonzero_l3897_389779


namespace min_concerts_is_14_l3897_389755

/-- Represents a schedule of concerts --/
structure Schedule where
  numSingers : Nat
  singersPerConcert : Nat
  numConcerts : Nat
  pairsPerformTogether : Nat

/-- Checks if a schedule is valid --/
def isValidSchedule (s : Schedule) : Prop :=
  s.numSingers = 8 ∧
  s.singersPerConcert = 4 ∧
  s.numConcerts * (s.singersPerConcert.choose 2) = s.numSingers.choose 2 * s.pairsPerformTogether

/-- Theorem: The minimum number of concerts is 14 --/
theorem min_concerts_is_14 :
  ∀ s : Schedule, isValidSchedule s → s.numConcerts ≥ 14 :=
by sorry

end min_concerts_is_14_l3897_389755


namespace chocolate_probability_theorem_l3897_389714

/- Define the type for a box of chocolates -/
structure ChocolateBox where
  white : ℕ
  total : ℕ
  h_total : total > 0

/- Define the probability of drawing a white chocolate from a box -/
def prob (box : ChocolateBox) : ℚ :=
  box.white / box.total

/- Define the combined box of chocolates -/
def combinedBox (box1 box2 : ChocolateBox) : ChocolateBox where
  white := box1.white + box2.white
  total := box1.total + box2.total
  h_total := by
    simp [gt_iff_lt, add_pos_iff]
    exact Or.inl box1.h_total

/- Theorem statement -/
theorem chocolate_probability_theorem (box1 box2 : ChocolateBox) :
  ∃ (box1' box2' : ChocolateBox),
    (prob (combinedBox box1' box2') = 7 / 12) ∧
    (prob (combinedBox box1 box2) = 11 / 19) ∧
    (prob (combinedBox box1 box2) > min (prob box1) (prob box2)) :=
  sorry

end chocolate_probability_theorem_l3897_389714


namespace number_added_to_23_l3897_389774

theorem number_added_to_23 : ∃! x : ℝ, 23 + x = 34 := by
  sorry

end number_added_to_23_l3897_389774


namespace parabola_ellipse_intersection_l3897_389752

/-- Represents the equations mx + ny² = 0 and mx² + ny² = 1 for m > 0 and n > 0 -/
def represents_parabola_ellipse_intersection (m n : ℝ) : Prop :=
  m > 0 ∧ n > 0 ∧ m < n ∧
  ∃ (x y : ℝ), 
    (m * x + n * y^2 = 0) ∧
    (m * x^2 + n * y^2 = 1) ∧
    (-1 < x ∧ x < 0)

/-- Theorem stating that the equations represent a parabola opening to the left intersecting an ellipse -/
theorem parabola_ellipse_intersection :
  ∀ (m n : ℝ), represents_parabola_ellipse_intersection m n →
  ∃ (x y : ℝ), 
    (y^2 = -m/n * x) ∧  -- Parabola equation
    (x^2/(1/m) + y^2/(1/n) = 1) ∧  -- Ellipse equation
    (-1 < x ∧ x < 0) :=
by sorry


end parabola_ellipse_intersection_l3897_389752


namespace largest_power_of_two_dividing_N_l3897_389722

/-- The number of vertices in the graph -/
def num_vertices : ℕ := 8

/-- The number of ordered pairs to examine for each permutation -/
def pairs_per_permutation : ℕ := num_vertices * (num_vertices - 1)

/-- The total number of examinations in Sophia's algorithm -/
def N : ℕ := (Nat.factorial num_vertices) * pairs_per_permutation

/-- The theorem stating that the largest power of two dividing N is 10 -/
theorem largest_power_of_two_dividing_N :
  ∃ k : ℕ, (2^10 : ℕ) ∣ N ∧ ¬(2^(k+1) : ℕ) ∣ N ∧ k = 10 := by
  sorry

end largest_power_of_two_dividing_N_l3897_389722


namespace sphere_radius_in_truncated_cone_l3897_389706

/-- The radius of a sphere tangent to the bases and lateral surface of a truncated cone --/
theorem sphere_radius_in_truncated_cone (R r : ℝ) (hR : R = 24) (hr : r = 5) :
  ∃ (radius : ℝ), radius > 0 ∧ radius^2 = (R - r)^2 / 2 :=
by sorry

end sphere_radius_in_truncated_cone_l3897_389706


namespace arithmetic_sequence_sum_property_l3897_389796

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  s : ℕ → ℝ  -- The sum function
  sum_def : ∀ n, s n = (n : ℝ) * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2
  arith_def : ∀ n, a (n + 1) - a n = a 2 - a 1

/-- Theorem: If s_30 = s_60 for an arithmetic sequence, then s_90 = 0 -/
theorem arithmetic_sequence_sum_property (seq : ArithmeticSequence) 
  (h : seq.s 30 = seq.s 60) : seq.s 90 = 0 := by
  sorry

end arithmetic_sequence_sum_property_l3897_389796


namespace x_to_neg_y_equals_half_l3897_389727

theorem x_to_neg_y_equals_half (x y : ℝ) (h : Real.sqrt (x + y - 3) = -(x - 2*y)^2) : 
  x^(-y) = (1/2 : ℝ) := by
sorry

end x_to_neg_y_equals_half_l3897_389727


namespace quadratic_polynomial_k_value_l3897_389740

/-- A polynomial is quadratic if its degree is exactly 2 -/
def IsQuadratic (p : Polynomial ℝ) : Prop :=
  p.degree = 2

theorem quadratic_polynomial_k_value :
  ∀ k : ℝ,
    IsQuadratic (Polynomial.monomial 3 (k - 2) + Polynomial.monomial 2 k + Polynomial.monomial 1 (-2) + Polynomial.monomial 0 (-6))
    → k = 2 := by
  sorry

end quadratic_polynomial_k_value_l3897_389740


namespace burning_time_3x5_grid_l3897_389720

/-- Represents a rectangular grid of toothpicks -/
structure ToothpickGrid :=
  (rows : ℕ)
  (cols : ℕ)
  (toothpicks : ℕ)

/-- Represents the burning properties of toothpicks -/
structure BurningProperties :=
  (burn_time : ℕ)  -- Time for one toothpick to burn completely
  (spread_speed : ℕ)  -- Speed at which fire spreads (assumed constant)

/-- Calculates the total burning time for a toothpick grid -/
def total_burning_time (grid : ToothpickGrid) (props : BurningProperties) : ℕ :=
  sorry

/-- Theorem stating the burning time for the specific problem -/
theorem burning_time_3x5_grid :
  ∀ (grid : ToothpickGrid) (props : BurningProperties),
    grid.rows = 3 ∧ 
    grid.cols = 5 ∧ 
    grid.toothpicks = 38 ∧
    props.burn_time = 10 ∧
    props.spread_speed = 1 →
    total_burning_time grid props = 50 :=
by
  sorry

end burning_time_3x5_grid_l3897_389720


namespace island_puzzle_l3897_389772

-- Define the possible types for natives
inductive NativeType
  | Knight
  | Liar

-- Define a function to represent the truthfulness of a statement based on the native type
def isTruthful (t : NativeType) (s : Prop) : Prop :=
  match t with
  | NativeType.Knight => s
  | NativeType.Liar => ¬s

-- Define the statement made by A
def statementA (typeA typeB : NativeType) : Prop :=
  typeA = NativeType.Liar ∨ typeB = NativeType.Liar

-- Theorem stating that A is a knight and B is a liar
theorem island_puzzle :
  ∃ (typeA typeB : NativeType),
    isTruthful typeA (statementA typeA typeB) ∧
    typeA = NativeType.Knight ∧
    typeB = NativeType.Liar :=
  sorry

end island_puzzle_l3897_389772


namespace some_number_value_l3897_389723

theorem some_number_value (x : ℝ) :
  64 + 5 * 12 / (180 / x) = 65 → x = 3 := by
  sorry

end some_number_value_l3897_389723


namespace simplify_rational_function_l3897_389768

theorem simplify_rational_function (x : ℝ) (h : x ≠ -1) :
  (x + 1) / (x^2 + 2*x + 1) = 1 / (x + 1) := by
  sorry

end simplify_rational_function_l3897_389768


namespace bicycle_selling_prices_l3897_389773

def calculate_selling_price (purchase_price : ℕ) (loss_percentage : ℕ) : ℕ :=
  purchase_price - (purchase_price * loss_percentage / 100)

def bicycle1_purchase_price : ℕ := 1800
def bicycle1_loss_percentage : ℕ := 25

def bicycle2_purchase_price : ℕ := 2700
def bicycle2_loss_percentage : ℕ := 15

def bicycle3_purchase_price : ℕ := 2200
def bicycle3_loss_percentage : ℕ := 20

theorem bicycle_selling_prices :
  (calculate_selling_price bicycle1_purchase_price bicycle1_loss_percentage = 1350) ∧
  (calculate_selling_price bicycle2_purchase_price bicycle2_loss_percentage = 2295) ∧
  (calculate_selling_price bicycle3_purchase_price bicycle3_loss_percentage = 1760) :=
by sorry

end bicycle_selling_prices_l3897_389773


namespace ahmed_hassan_apple_ratio_l3897_389770

/-- Ahmed's orchard has 8 orange trees and an unknown number of apple trees. -/
def ahmed_orange_trees : ℕ := 8

/-- Hassan's orchard has 1 apple tree. -/
def hassan_apple_trees : ℕ := 1

/-- Hassan's orchard has 2 orange trees. -/
def hassan_orange_trees : ℕ := 2

/-- The difference in total trees between Ahmed's and Hassan's orchards. -/
def tree_difference : ℕ := 9

/-- Ahmed's apple trees -/
def ahmed_apple_trees : ℕ := ahmed_orange_trees + tree_difference - (hassan_apple_trees + hassan_orange_trees)

theorem ahmed_hassan_apple_ratio :
  ahmed_apple_trees = 4 * hassan_apple_trees := by
  sorry

end ahmed_hassan_apple_ratio_l3897_389770


namespace pascal_row_20_sum_l3897_389757

/-- Binomial coefficient -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- The sum of the third, fourth, and fifth elements in Row 20 of Pascal's triangle -/
def pascalSum : ℕ := binomial 20 2 + binomial 20 3 + binomial 20 4

/-- Theorem stating that the sum of the third, fourth, and fifth elements 
    in Row 20 of Pascal's triangle is equal to 6175 -/
theorem pascal_row_20_sum : pascalSum = 6175 := by
  sorry

end pascal_row_20_sum_l3897_389757


namespace tower_height_property_l3897_389760

/-- The height of a tower that appears under twice the angle from 18 meters as it does from 48 meters -/
def tower_height : ℝ := 24

/-- The distance from which the tower appears under twice the angle -/
def closer_distance : ℝ := 18

/-- The distance from which the tower appears under the original angle -/
def farther_distance : ℝ := 48

/-- The angle at which the tower appears from the farther distance -/
noncomputable def base_angle (h : ℝ) : ℝ := Real.arctan (h / farther_distance)

/-- The theorem stating the property of the tower's height -/
theorem tower_height_property : 
  base_angle (2 * tower_height) = 2 * base_angle tower_height := by sorry

end tower_height_property_l3897_389760


namespace sqrt_equation_solutions_l3897_389789

theorem sqrt_equation_solutions :
  {x : ℝ | x ≥ 2 ∧ Real.sqrt (x + 5 - 6 * Real.sqrt (x - 2)) + Real.sqrt (x + 10 - 8 * Real.sqrt (x - 2)) = 2} =
  {8.25, 22.25} := by
  sorry

end sqrt_equation_solutions_l3897_389789


namespace tetrahedron_bug_return_probability_l3897_389763

/-- Probability of returning to the starting vertex after n steps in a regular tetrahedron -/
def return_probability (n : ℕ) : ℚ :=
  match n with
  | 0 => 1
  | n + 1 => (1 - return_probability n) / 3

/-- The probability of returning to the starting vertex after 8 steps is 547/2187 -/
theorem tetrahedron_bug_return_probability :
  return_probability 8 = 547 / 2187 := by
  sorry

#eval return_probability 8

end tetrahedron_bug_return_probability_l3897_389763


namespace square_puzzle_l3897_389790

theorem square_puzzle (n : ℕ) 
  (h1 : n^2 + 20 = (n + 1)^2 - 9) : n = 14 ∧ n^2 + 20 = 216 := by
  sorry

#check square_puzzle

end square_puzzle_l3897_389790


namespace hexagon_CF_length_l3897_389798

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hexagon with specific properties -/
structure Hexagon where
  A : Point
  B : Point
  C : Point
  D : Point
  E : Point
  F : Point
  square_side : ℝ
  other_side : ℝ
  h_square : square_side = 20
  h_other : other_side = 23
  h_square_ABDE : A.x = 0 ∧ A.y = 0 ∧ B.x = square_side ∧ B.y = 0 ∧
                  D.x = square_side ∧ D.y = square_side ∧ E.x = 0 ∧ E.y = square_side
  h_parallel : C.x = B.x ∧ F.x = A.x
  h_BC : (C.x - B.x)^2 + (C.y - B.y)^2 = other_side^2
  h_CD : (D.x - C.x)^2 + (D.y - C.y)^2 = other_side^2
  h_EF : (F.x - E.x)^2 + (F.y - E.y)^2 = other_side^2
  h_FA : (A.x - F.x)^2 + (A.y - F.y)^2 = other_side^2

/-- The theorem to be proved -/
theorem hexagon_CF_length (h : Hexagon) :
  ∃ n : ℕ, n = 28 ∧ n = ⌊Real.sqrt ((h.C.x - h.F.x)^2 + (h.C.y - h.F.y)^2)⌋ := by
  sorry

end hexagon_CF_length_l3897_389798


namespace correct_dispatch_plans_l3897_389741

/-- The number of teachers available for selection -/
def total_teachers : ℕ := 8

/-- The number of teachers to be selected -/
def selected_teachers : ℕ := 4

/-- The number of remote areas -/
def remote_areas : ℕ := 4

/-- Function to calculate the number of ways to select teachers -/
def select_teachers : ℕ :=
  let with_a_c := Nat.choose (total_teachers - 3) (selected_teachers - 2)
  let without_a_c := Nat.choose (total_teachers - 2) selected_teachers
  with_a_c + without_a_c

/-- Function to calculate the number of ways to arrange teachers in areas -/
def arrange_teachers : ℕ := Nat.factorial selected_teachers

/-- The total number of different dispatch plans -/
def total_dispatch_plans : ℕ := select_teachers * arrange_teachers

theorem correct_dispatch_plans : total_dispatch_plans = 600 := by
  sorry

end correct_dispatch_plans_l3897_389741


namespace shelby_rain_time_l3897_389739

/-- Represents Shelby's driving scenario -/
structure DrivingScenario where
  speed_sunny : ℝ  -- Speed when not raining (miles per hour)
  speed_rainy : ℝ  -- Speed when raining (miles per hour)
  total_time : ℝ   -- Total journey time (minutes)
  stop_time : ℝ    -- Total stop time (minutes)
  total_distance : ℝ -- Total distance covered (miles)

/-- Calculates the time driven in rain given a DrivingScenario -/
def time_in_rain (scenario : DrivingScenario) : ℝ :=
  sorry

/-- Theorem stating that given Shelby's driving conditions, she drove 48 minutes in the rain -/
theorem shelby_rain_time (scenario : DrivingScenario) 
  (h1 : scenario.speed_sunny = 40)
  (h2 : scenario.speed_rainy = 25)
  (h3 : scenario.total_time = 75)
  (h4 : scenario.stop_time = 15)
  (h5 : scenario.total_distance = 28) :
  time_in_rain scenario = 48 := by
  sorry

end shelby_rain_time_l3897_389739


namespace complex_modulus_two_thirds_plus_three_i_l3897_389738

theorem complex_modulus_two_thirds_plus_three_i :
  Complex.abs (Complex.ofReal (2/3) + Complex.I * 3) = Real.sqrt 85 / 3 := by
  sorry

end complex_modulus_two_thirds_plus_three_i_l3897_389738


namespace absolute_difference_41st_terms_l3897_389769

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + d * (n - 1)

theorem absolute_difference_41st_terms :
  let A := arithmetic_sequence 50 6
  let B := arithmetic_sequence 100 (-15)
  abs (A 41 - B 41) = 790 := by sorry

end absolute_difference_41st_terms_l3897_389769


namespace last_problem_number_l3897_389732

theorem last_problem_number (start : ℕ) (problems_solved : ℕ) : 
  start = 80 → problems_solved = 46 → start + problems_solved - 1 = 125 := by
sorry

end last_problem_number_l3897_389732


namespace recreational_area_diameter_l3897_389762

/-- The diameter of the outer boundary of a circular recreational area -/
def outer_boundary_diameter (pond_diameter : ℝ) (flowerbed_width : ℝ) (jogging_path_width : ℝ) : ℝ :=
  pond_diameter + 2 * (flowerbed_width + jogging_path_width)

/-- Theorem: The diameter of the outer boundary of the circular recreational area is 64 feet -/
theorem recreational_area_diameter : 
  outer_boundary_diameter 20 10 12 = 64 := by sorry

end recreational_area_diameter_l3897_389762


namespace part1_part2_l3897_389782

def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*m*x + m^2 - 1 ≤ 0}

def p (x : ℝ) : Prop := x ∈ A
def q (m : ℝ) (x : ℝ) : Prop := x ∈ B m

theorem part1 (m : ℝ) (h : ∀ x, q m x → p x) (h' : ∃ x, p x ∧ ¬q m x) :
  0 ≤ m ∧ m ≤ 1 := by sorry

theorem part2 (m : ℝ) (h : ∀ x ∈ A, x^2 + m ≥ 4 + 3*x) :
  m ≥ 25/4 := by sorry

end part1_part2_l3897_389782


namespace inequality_solution_l3897_389758

theorem inequality_solution (x y : ℝ) :
  (4 * Real.sin x - Real.sqrt (Real.cos y) - Real.sqrt (Real.cos y - 16 * (Real.cos x)^2 + 12) ≥ 2) ↔
  (∃ (n k : ℤ), x = ((-1)^n * π / 6 + 2 * n * π) ∧ y = (π / 2 + k * π)) :=
by sorry

end inequality_solution_l3897_389758


namespace trivia_team_points_l3897_389736

/-- Calculates the total points scored by a trivia team given the total number of members,
    the number of absent members, and the points scored by each attending member. -/
def total_points (total_members : ℕ) (absent_members : ℕ) (points_per_member : ℕ) : ℕ :=
  (total_members - absent_members) * points_per_member

/-- Proves that a trivia team with 15 total members, 6 absent members, and 3 points per
    attending member scores a total of 27 points. -/
theorem trivia_team_points :
  total_points 15 6 3 = 27 := by
  sorry

end trivia_team_points_l3897_389736


namespace five_T_three_l3897_389712

-- Define the operation T
def T (a b : ℤ) : ℤ := 4*a + 6*b

-- Theorem to prove
theorem five_T_three : T 5 3 = 38 := by
  sorry

end five_T_three_l3897_389712


namespace greatest_rational_root_quadratic_l3897_389742

theorem greatest_rational_root_quadratic (a b c : ℕ) (ha : a ≤ 100) (hb : b ≤ 100) (hc : c ≤ 100) (ha_pos : a > 0) :
  ∀ (p q : ℤ), q ≠ 0 → a * (p / q)^2 + b * (p / q) + c = 0 →
  (p : ℚ) / q ≤ (-1 : ℚ) / 99 :=
sorry

end greatest_rational_root_quadratic_l3897_389742


namespace triangle_inequality_with_additional_segment_l3897_389719

theorem triangle_inequality_with_additional_segment
  (a b c d : ℝ)
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b)
  (h_d_positive : d > 0)
  (a₁ : ℝ) (h_a₁ : a₁ = min a d)
  (b₁ : ℝ) (h_b₁ : b₁ = min b d)
  (c₁ : ℝ) (h_c₁ : c₁ = min c d) :
  a₁ + b₁ > c₁ ∧ b₁ + c₁ > a₁ ∧ c₁ + a₁ > b₁ :=
by sorry

end triangle_inequality_with_additional_segment_l3897_389719


namespace exists_valid_solution_l3897_389733

def mother_charge : ℝ := 6.50
def child_charge_per_year : ℝ := 0.65
def total_bill : ℝ := 13.00

def is_valid_solution (twin_age youngest_age : ℕ) : Prop :=
  twin_age > youngest_age ∧
  mother_charge + child_charge_per_year * (2 * twin_age + youngest_age) = total_bill

theorem exists_valid_solution :
  ∃ (twin_age youngest_age : ℕ), is_valid_solution twin_age youngest_age ∧ (youngest_age = 2 ∨ youngest_age = 4) :=
sorry

end exists_valid_solution_l3897_389733


namespace imaginary_unit_sum_l3897_389759

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem imaginary_unit_sum : i + i^2 + i^3 = -1 := by sorry

end imaginary_unit_sum_l3897_389759


namespace not_perfect_square_with_1234_divisors_l3897_389718

/-- A natural number with exactly 1234 divisors is not a perfect square. -/
theorem not_perfect_square_with_1234_divisors (n : ℕ) : 
  (∃ (d : Finset ℕ), d = {x | x ∣ n} ∧ d.card = 1234) → ¬∃ (m : ℕ), n = m^2 := by
  sorry

end not_perfect_square_with_1234_divisors_l3897_389718


namespace ducks_remaining_theorem_l3897_389776

def ducks_remaining (initial : ℕ) : ℕ :=
  let after_first := initial - (initial / 4)
  let after_second := after_first - (after_first / 6)
  after_second - (after_second * 3 / 10)

theorem ducks_remaining_theorem :
  ducks_remaining 320 = 140 := by
  sorry

end ducks_remaining_theorem_l3897_389776


namespace parcel_cost_formula_l3897_389737

/-- The cost function for sending a parcel post package -/
def parcel_cost (P : ℕ) : ℕ :=
  20 + 5 * (P - 1)

theorem parcel_cost_formula (P : ℕ) (h : P ≥ 2) :
  parcel_cost P = 20 + 5 * (P - 1) :=
by sorry

end parcel_cost_formula_l3897_389737

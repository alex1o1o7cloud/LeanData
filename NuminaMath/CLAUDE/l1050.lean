import Mathlib

namespace system_solution_unique_l1050_105094

theorem system_solution_unique :
  ∃! (x y : ℝ), x > 0 ∧ y > 0 ∧
    x^4 + y^4 - x^2*y^2 = 13 ∧
    x^2 - y^2 + 2*x*y = 1 ∧
    x = 1 ∧ y = 2 := by
  sorry

end system_solution_unique_l1050_105094


namespace card_flip_game_l1050_105073

theorem card_flip_game (n k : ℕ) (hn : Odd n) (hk : Even k) (hkn : k < n) :
  ∀ (t : ℕ), ∃ (i : ℕ), i < n ∧ Even (t * k / n + (if i < t * k % n then 1 else 0)) := by
  sorry

end card_flip_game_l1050_105073


namespace simplify_fraction_l1050_105070

theorem simplify_fraction (a : ℝ) (h1 : a ≠ 1) (h2 : a ≠ -1) :
  (2 / (a^2 - 1)) * (1 / (a - 1)) = 2 / ((a - 1)^2 * (a + 1)) := by
  sorry

end simplify_fraction_l1050_105070


namespace babysitting_hourly_rate_l1050_105052

/-- Calculates the hourly rate for babysitting given total expenses, hours worked, and leftover money -/
def calculate_hourly_rate (total_expenses : ℕ) (hours_worked : ℕ) (leftover : ℕ) : ℚ :=
  (total_expenses + leftover) / hours_worked

/-- Theorem: Given the problem conditions, the hourly rate for babysitting is $8 -/
theorem babysitting_hourly_rate :
  let total_expenses := 65
  let hours_worked := 9
  let leftover := 7
  calculate_hourly_rate total_expenses hours_worked leftover = 8 := by
sorry

end babysitting_hourly_rate_l1050_105052


namespace fifth_rollercoaster_speed_l1050_105081

/-- Theorem: Given 5 rollercoasters with specific speeds and average, prove the speed of the fifth rollercoaster -/
theorem fifth_rollercoaster_speed 
  (v₁ v₂ v₃ v₄ v₅ : ℝ) 
  (h1 : v₁ = 50)
  (h2 : v₂ = 62)
  (h3 : v₃ = 73)
  (h4 : v₄ = 70)
  (h_avg : (v₁ + v₂ + v₃ + v₄ + v₅) / 5 = 59) :
  v₅ = 40 := by
  sorry

end fifth_rollercoaster_speed_l1050_105081


namespace complex_power_195_deg_60_l1050_105005

theorem complex_power_195_deg_60 :
  (Complex.exp (195 * π / 180 * Complex.I)) ^ 60 = (1 / 2 : ℂ) - Complex.I * (Real.sqrt 3 / 2) := by
  sorry

end complex_power_195_deg_60_l1050_105005


namespace inequality_proof_l1050_105097

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  (1 / (a^3 * (b + c))) + (1 / (b^3 * (c + a))) + (1 / (c^3 * (a + b))) ≥ 3/2 := by
  sorry

end inequality_proof_l1050_105097


namespace integer_solution_proof_l1050_105031

theorem integer_solution_proof (a b c : ℤ) :
  a + b + c = 24 →
  a^2 + b^2 + c^2 = 210 →
  a * b * c = 440 →
  ({a, b, c} : Set ℤ) = {5, 8, 11} := by
  sorry

end integer_solution_proof_l1050_105031


namespace some_ounce_size_is_eight_l1050_105051

/-- The size of the some-ounce glasses -/
def some_ounce_size : ℕ := sorry

/-- The total amount of water available -/
def total_water : ℕ := 122

/-- The number of 5-ounce glasses filled -/
def five_ounce_glasses : ℕ := 6

/-- The number of some-ounce glasses filled -/
def some_ounce_glasses : ℕ := 4

/-- The number of 4-ounce glasses that can be filled with remaining water -/
def four_ounce_glasses : ℕ := 15

/-- Theorem stating that the size of the some-ounce glasses is 8 ounces -/
theorem some_ounce_size_is_eight :
  some_ounce_size = 8 ∧
  total_water = 
    five_ounce_glasses * 5 + 
    some_ounce_glasses * some_ounce_size + 
    four_ounce_glasses * 4 :=
by sorry

end some_ounce_size_is_eight_l1050_105051


namespace hanks_pancakes_l1050_105082

/-- The number of pancakes in a big stack -/
def big_stack : ℕ := 5

/-- The number of pancakes in a short stack -/
def short_stack : ℕ := 3

/-- The number of customers who ordered short stack pancakes -/
def short_stack_orders : ℕ := 9

/-- The number of customers who ordered big stack pancakes -/
def big_stack_orders : ℕ := 6

/-- The total number of pancakes Hank needs to make -/
def total_pancakes : ℕ := short_stack_orders * short_stack + big_stack_orders * big_stack

theorem hanks_pancakes : total_pancakes = 57 := by
  sorry

end hanks_pancakes_l1050_105082


namespace solve_flower_problem_l1050_105089

def flower_problem (minyoung_flowers : ℕ) (ratio : ℕ) : Prop :=
  let yoojung_flowers := minyoung_flowers / ratio
  minyoung_flowers + yoojung_flowers = 30

theorem solve_flower_problem :
  flower_problem 24 4 := by
  sorry

end solve_flower_problem_l1050_105089


namespace contrapositive_equivalence_l1050_105035

theorem contrapositive_equivalence :
  (∀ x y : ℝ, x^2 + y^2 = 0 → x = 0 ∧ y = 0) ↔
  (∀ x y : ℝ, ¬(x = 0 ∧ y = 0) → x^2 + y^2 ≠ 0) :=
by sorry

end contrapositive_equivalence_l1050_105035


namespace intersection_implies_range_l1050_105090

def A (a : ℝ) : Set ℝ := {x | |x - a| < 2}

def B : Set ℝ := {x | (2*x - 1) / (x + 2) < 1}

theorem intersection_implies_range (a : ℝ) : A a ∩ B = A a → a ∈ Set.Icc 0 1 := by
  sorry

end intersection_implies_range_l1050_105090


namespace min_omega_for_50_maxima_l1050_105091

theorem min_omega_for_50_maxima (ω : ℝ) : ω > 0 → (∀ x ∈ Set.Icc 0 1, ∃ y, y = Real.sin (ω * x)) →
  (∃ (maxima : Finset ℝ), maxima.card ≥ 50 ∧ 
    ∀ t ∈ maxima, t ∈ Set.Icc 0 1 ∧ 
    (∀ h ∈ Set.Icc 0 1, Real.sin (ω * t) ≥ Real.sin (ω * h))) →
  ω ≥ 197 * Real.pi / 2 :=
sorry

end min_omega_for_50_maxima_l1050_105091


namespace exists_special_number_l1050_105048

/-- A function that returns the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- A function that returns the number of digits in a natural number -/
def num_digits (n : ℕ) : ℕ := sorry

/-- A function that checks if all digits of a natural number are non-zero -/
def all_digits_nonzero (n : ℕ) : Prop := sorry

/-- Theorem: There exists a 1000-digit natural number with all non-zero digits that is divisible by the sum of its digits -/
theorem exists_special_number : 
  ∃ n : ℕ, 
    num_digits n = 1000 ∧ 
    all_digits_nonzero n ∧ 
    n % sum_of_digits n = 0 :=
sorry

end exists_special_number_l1050_105048


namespace partnership_contribution_l1050_105025

theorem partnership_contribution 
  (a_capital : ℕ) 
  (a_time : ℕ) 
  (b_time : ℕ) 
  (total_profit : ℕ) 
  (a_profit : ℕ) 
  (h1 : a_capital = 5000)
  (h2 : a_time = 8)
  (h3 : b_time = 5)
  (h4 : total_profit = 8400)
  (h5 : a_profit = 4800) :
  ∃ b_capital : ℕ, 
    (a_capital * a_time : ℚ) / ((a_capital * a_time + b_capital * b_time) : ℚ) = 
    (a_profit : ℚ) / (total_profit : ℚ) ∧ 
    b_capital = 6000 := by
  sorry

end partnership_contribution_l1050_105025


namespace prob_different_colors_for_given_box_l1050_105078

/-- A box containing balls of two colors -/
structure Box where
  small_balls : ℕ
  black_balls : ℕ

/-- The probability of drawing two balls of different colors -/
def prob_different_colors (b : Box) : ℚ :=
  let total_balls := b.small_balls + b.black_balls
  let different_color_combinations := b.small_balls * b.black_balls
  let total_combinations := (total_balls * (total_balls - 1)) / 2
  different_color_combinations / total_combinations

/-- The theorem stating the probability of drawing two balls of different colors -/
theorem prob_different_colors_for_given_box :
  prob_different_colors { small_balls := 3, black_balls := 1 } = 1/2 := by
  sorry


end prob_different_colors_for_given_box_l1050_105078


namespace plot_perimeter_is_220_l1050_105092

/-- Represents a rectangular plot with the given conditions -/
structure RectangularPlot where
  width : ℝ
  length : ℝ
  fencingCostPerMeter : ℝ
  totalFencingCost : ℝ
  lengthWidthRelation : length = width + 10
  fencingCostRelation : fencingCostPerMeter * (2 * (length + width)) = totalFencingCost

/-- The perimeter of the rectangular plot -/
def perimeter (plot : RectangularPlot) : ℝ :=
  2 * (plot.length + plot.width)

/-- Theorem stating that the perimeter of the plot is 220 meters -/
theorem plot_perimeter_is_220 (plot : RectangularPlot) 
    (h1 : plot.fencingCostPerMeter = 6.5)
    (h2 : plot.totalFencingCost = 1430) : 
  perimeter plot = 220 := by
  sorry

end plot_perimeter_is_220_l1050_105092


namespace hyperbola_eccentricity_theorem_l1050_105033

/-- Hyperbola type -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- Point type -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point is on the asymptote of a hyperbola -/
def onAsymptote (h : Hyperbola) (p : Point) : Prop :=
  p.y = (h.b / h.a) * p.x ∨ p.y = -(h.b / h.a) * p.x

/-- Calculates the angle between three points -/
noncomputable def angle (p1 p2 p3 : Point) : ℝ := sorry

/-- Calculates the eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := sorry

/-- Main theorem -/
theorem hyperbola_eccentricity_theorem (h : Hyperbola) (a1 a2 b : Point) :
  a1.x = -h.a ∧ a1.y = 0 ∧  -- A₁ is left vertex
  a2.x = h.a ∧ a2.y = 0 ∧   -- A₂ is right vertex
  onAsymptote h b ∧         -- B is on asymptote
  angle a1 b a2 = π/3 →     -- ∠A₁BA₂ = 60°
  eccentricity h = Real.sqrt 21 / 3 := by
  sorry

end hyperbola_eccentricity_theorem_l1050_105033


namespace possible_values_of_x_l1050_105053

theorem possible_values_of_x (x : ℕ) 
  (h1 : ∃ k : ℕ, x = 9 * k)
  (h2 : x^2 > 225)
  (h3 : x < 30) :
  x = 18 ∨ x = 27 :=
sorry

end possible_values_of_x_l1050_105053


namespace theater_ticket_contradiction_l1050_105069

theorem theater_ticket_contradiction :
  ∀ (adult_price child_price : ℚ) 
    (total_tickets adult_tickets : ℕ) 
    (total_receipts : ℚ),
  adult_price = 12 →
  total_tickets = 130 →
  adult_tickets = 90 →
  total_receipts = 840 →
  ¬(adult_price * adult_tickets + 
    child_price * (total_tickets - adult_tickets) = 
    total_receipts) :=
by
  sorry

#check theater_ticket_contradiction

end theater_ticket_contradiction_l1050_105069


namespace strawberry_picking_l1050_105064

/-- The number of baskets Lilibeth fills -/
def baskets : ℕ := 6

/-- The number of strawberries each basket holds -/
def strawberries_per_basket : ℕ := 50

/-- The number of Lilibeth's friends who pick the same amount as her -/
def friends : ℕ := 3

/-- The total number of strawberries picked by Lilibeth and her friends -/
def total_strawberries : ℕ := (friends + 1) * (baskets * strawberries_per_basket)

theorem strawberry_picking :
  total_strawberries = 1200 := by
  sorry

end strawberry_picking_l1050_105064


namespace fraction_product_cube_specific_fraction_product_l1050_105008

theorem fraction_product_cube (a b c d : ℚ) :
  (a / b)^3 * (c / d)^3 = ((a * c) / (b * d))^3 :=
by sorry

theorem specific_fraction_product :
  (8 / 9 : ℚ)^3 * (3 / 5 : ℚ)^3 = 512 / 3375 :=
by sorry

end fraction_product_cube_specific_fraction_product_l1050_105008


namespace jakes_weight_l1050_105084

theorem jakes_weight (jake_weight sister_weight : ℝ) 
  (h1 : jake_weight - 15 = 2 * sister_weight)
  (h2 : jake_weight + sister_weight = 132) : 
  jake_weight = 93 := by
sorry

end jakes_weight_l1050_105084


namespace sum_longest_altitudes_is_21_l1050_105000

/-- A triangle with sides 9, 12, and 15 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a = 9
  hb : b = 12
  hc : c = 15
  right_angle : a^2 + b^2 = c^2

/-- The sum of the lengths of the two longest altitudes in the right triangle -/
def sum_longest_altitudes (t : RightTriangle) : ℝ := t.a + t.b

/-- Theorem stating that the sum of the two longest altitudes is 21 -/
theorem sum_longest_altitudes_is_21 (t : RightTriangle) :
  sum_longest_altitudes t = 21 := by sorry

end sum_longest_altitudes_is_21_l1050_105000


namespace imaginary_part_of_z_l1050_105019

theorem imaginary_part_of_z (z : ℂ) (h : z + z * Complex.I = 2) : 
  z.im = -1 := by sorry

end imaginary_part_of_z_l1050_105019


namespace geometric_sequence_problem_l1050_105041

/-- Given a geometric sequence {aₙ}, prove that if a₅ - a₁ = 15 and a₄ - a₂ = 6,
    then (a₃ = 4 and q = 2) or (a₃ = -4 and q = 1/2) -/
theorem geometric_sequence_problem (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- Definition of geometric sequence
  a 5 - a 1 = 15 →              -- First given condition
  a 4 - a 2 = 6 →               -- Second given condition
  ((a 3 = 4 ∧ q = 2) ∨ (a 3 = -4 ∧ q = 1/2)) :=
by sorry

end geometric_sequence_problem_l1050_105041


namespace factorization_equality_l1050_105036

theorem factorization_equality (a x y : ℝ) : a * x^2 + 2 * a * x * y + a * y^2 = a * (x + y)^2 := by
  sorry

end factorization_equality_l1050_105036


namespace solution_set_implies_a_value_l1050_105015

theorem solution_set_implies_a_value (a : ℝ) :
  (∀ x : ℝ, |a * x + 2| < 4 ↔ -1 < x ∧ x < 3) →
  a = -2 :=
by sorry

end solution_set_implies_a_value_l1050_105015


namespace expression_value_l1050_105027

theorem expression_value (a b c : ℝ) (ha : a ≠ 3) (hb : b ≠ 4) (hc : c ≠ 5) :
  (a - 3) / (5 - c) * (b - 4) / (3 - a) * (c - 5) / (4 - b) = -1 := by
  sorry

end expression_value_l1050_105027


namespace p_sufficient_not_necessary_l1050_105007

-- Define the propositions p and q
def p (x : ℝ) : Prop := x = 1
def q (x : ℝ) : Prop := x - 1 = Real.sqrt (x - 1)

-- Theorem stating that p is a sufficient but not necessary condition for q
theorem p_sufficient_not_necessary :
  (∀ x : ℝ, p x → q x) ∧ (∃ x : ℝ, q x ∧ ¬p x) := by sorry

end p_sufficient_not_necessary_l1050_105007


namespace total_cost_after_increase_l1050_105098

def price_increase : ℚ := 15 / 100

def original_orange_price : ℚ := 40
def original_mango_price : ℚ := 50

def new_orange_price : ℚ := original_orange_price * (1 + price_increase)
def new_mango_price : ℚ := original_mango_price * (1 + price_increase)

def total_cost : ℚ := 10 * new_orange_price + 10 * new_mango_price

theorem total_cost_after_increase :
  total_cost = 1035 := by sorry

end total_cost_after_increase_l1050_105098


namespace lost_pages_problem_l1050_105047

/-- Calculates the number of lost pages of stickers -/
def lost_pages (stickers_per_page : ℕ) (initial_pages : ℕ) (remaining_stickers : ℕ) : ℕ :=
  (stickers_per_page * initial_pages - remaining_stickers) / stickers_per_page

theorem lost_pages_problem :
  let stickers_per_page : ℕ := 20
  let initial_pages : ℕ := 12
  let remaining_stickers : ℕ := 220
  lost_pages stickers_per_page initial_pages remaining_stickers = 1 := by
  sorry

end lost_pages_problem_l1050_105047


namespace equation_equivalence_l1050_105012

theorem equation_equivalence :
  ∀ x : ℝ, x^2 - 2*x - 2 = 0 ↔ (x - 1)^2 = 3 :=
by sorry

end equation_equivalence_l1050_105012


namespace fencing_rate_proof_l1050_105038

/-- Given a rectangular plot with the following properties:
    - The length is 10 meters more than the width
    - The perimeter is 340 meters
    - The total cost of fencing is 2210 Rs
    Prove that the rate per meter for fencing is 6.5 Rs -/
theorem fencing_rate_proof (width : ℝ) (length : ℝ) (perimeter : ℝ) (total_cost : ℝ) :
  length = width + 10 →
  perimeter = 340 →
  perimeter = 2 * (length + width) →
  total_cost = 2210 →
  total_cost / perimeter = 6.5 := by
  sorry

end fencing_rate_proof_l1050_105038


namespace sabrina_cookies_l1050_105039

/-- The number of cookies Sabrina had at the start -/
def initial_cookies : ℕ := 20

/-- The number of cookies Sabrina gave to her brother -/
def cookies_to_brother : ℕ := 10

/-- The number of cookies Sabrina's mother gave her -/
def cookies_from_mother : ℕ := cookies_to_brother / 2

/-- The fraction of cookies Sabrina gave to her sister -/
def fraction_to_sister : ℚ := 2 / 3

/-- The number of cookies Sabrina has left -/
def remaining_cookies : ℕ := 5

theorem sabrina_cookies :
  initial_cookies = cookies_to_brother + 
    (initial_cookies - cookies_to_brother + cookies_from_mother) * (1 - fraction_to_sister) :=
by sorry

end sabrina_cookies_l1050_105039


namespace fraction_simplification_l1050_105042

theorem fraction_simplification (x : ℝ) (h : x ≠ -3) : 
  (x^2 - 9) / (x^2 + 6*x + 9) - (2*x + 1) / (2*x + 6) = -7 / (2*x + 6) := by
  sorry

end fraction_simplification_l1050_105042


namespace equation_solution_l1050_105017

theorem equation_solution : ∃! x : ℝ, 2 * x + 1 = x - 1 ∧ x = -2 := by sorry

end equation_solution_l1050_105017


namespace point_D_in_fourth_quadrant_l1050_105010

/-- Definition of a point in the Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the fourth quadrant -/
def fourth_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- The given point D -/
def point_D : Point :=
  { x := 6, y := -7 }

/-- Theorem: point D is in the fourth quadrant -/
theorem point_D_in_fourth_quadrant : fourth_quadrant point_D := by
  sorry

end point_D_in_fourth_quadrant_l1050_105010


namespace fixed_point_of_exponential_function_l1050_105011

theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (ha_ne_one : a ≠ 1) :
  let f := λ x : ℝ => a^(x - 2) + 2
  f 2 = 3 := by sorry

end fixed_point_of_exponential_function_l1050_105011


namespace park_diameter_l1050_105002

/-- Given a circular park with a central pond, vegetable garden, and jogging path,
    this theorem proves that the diameter of the outer boundary is 64 feet. -/
theorem park_diameter (pond_diameter vegetable_width jogging_width : ℝ) 
  (h1 : pond_diameter = 20)
  (h2 : vegetable_width = 12)
  (h3 : jogging_width = 10) :
  2 * (pond_diameter / 2 + vegetable_width + jogging_width) = 64 := by
  sorry

end park_diameter_l1050_105002


namespace freshman_class_size_l1050_105044

theorem freshman_class_size :
  ∃ n : ℕ, n < 500 ∧ n % 25 = 24 ∧ n % 19 = 11 ∧
  ∀ m : ℕ, m < n → (m % 25 ≠ 24 ∨ m % 19 ≠ 11) :=
by sorry

end freshman_class_size_l1050_105044


namespace trig_identity_l1050_105026

theorem trig_identity (α β : Real) 
  (h : (Real.cos α)^6 / (Real.cos β)^3 + (Real.sin α)^6 / (Real.sin β)^3 = 1) :
  (Real.sin β)^6 / (Real.sin α)^3 + (Real.cos β)^6 / (Real.cos α)^3 = 1 := by
  sorry

end trig_identity_l1050_105026


namespace inequality_implications_l1050_105028

theorem inequality_implications (a b : ℝ) (h : a > b) :
  (a + 2 > b + 2) ∧
  (-a < -b) ∧
  (2 * a > 2 * b) ∧
  ∃ c : ℝ, ¬(a * c^2 > b * c^2) :=
by sorry

end inequality_implications_l1050_105028


namespace equal_intercept_line_equation_l1050_105049

/-- A line with equal x and y intercepts passing through a given point. -/
structure EqualInterceptLine where
  -- The x-coordinate of the point the line passes through
  x : ℝ
  -- The y-coordinate of the point the line passes through
  y : ℝ
  -- The common intercept value
  a : ℝ
  -- The line passes through the point (x, y)
  point_on_line : x / a + y / a = 1

/-- The equation of a line with equal x and y intercepts passing through (2,1) is x + y - 3 = 0 -/
theorem equal_intercept_line_equation :
  ∀ (l : EqualInterceptLine), l.x = 2 ∧ l.y = 1 → (λ x y => x + y - 3 = 0) = (λ x y => x / l.a + y / l.a = 1) :=
by sorry

end equal_intercept_line_equation_l1050_105049


namespace distinct_students_count_l1050_105014

/-- The number of distinct students taking the math contest at Euclid Middle School -/
def distinct_students : ℕ := by
  -- Define the number of students in each class
  let gauss_class : ℕ := 12
  let euler_class : ℕ := 10
  let fibonnaci_class : ℕ := 7
  
  -- Define the number of students counted twice
  let double_counted : ℕ := 1
  
  -- Calculate the total number of distinct students
  exact gauss_class + euler_class + fibonnaci_class - double_counted

/-- Theorem stating that the number of distinct students taking the contest is 28 -/
theorem distinct_students_count : distinct_students = 28 := by
  sorry

end distinct_students_count_l1050_105014


namespace intersection_A_complement_B_l1050_105079

open Set

/-- The universal set U -/
def U : Set Nat := {1, 2, 3, 4, 5, 6, 7}

/-- Set A -/
def A : Set Nat := {3, 4, 5}

/-- Set B -/
def B : Set Nat := {1, 3, 6}

/-- Theorem stating that the intersection of A and the complement of B in U equals {4, 5} -/
theorem intersection_A_complement_B : A ∩ (U \ B) = {4, 5} := by
  sorry

end intersection_A_complement_B_l1050_105079


namespace log_6_6_log_2_8_log_equation_l1050_105037

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- Theorem statements
theorem log_6_6 : log 6 6 = 1 := by sorry

theorem log_2_8 : log 2 8 = 3 := by sorry

theorem log_equation (m : ℝ) : log 2 (m - 2) = 4 → m = 18 := by sorry

end log_6_6_log_2_8_log_equation_l1050_105037


namespace book_arrangement_theorem_l1050_105067

/-- The number of ways to arrange books of different languages on a shelf. -/
def arrange_books (arabic : ℕ) (german : ℕ) (spanish : ℕ) : ℕ :=
  (Nat.factorial 3) * (Nat.factorial arabic) * (Nat.factorial german) * (Nat.factorial spanish)

/-- Theorem: The number of ways to arrange 10 books (2 Arabic, 4 German, 4 Spanish) on a shelf,
    keeping books of the same language together, is equal to 6912. -/
theorem book_arrangement_theorem :
  arrange_books 2 4 4 = 6912 := by
  sorry

end book_arrangement_theorem_l1050_105067


namespace even_periodic_function_l1050_105075

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def is_period_two (f : ℝ → ℝ) : Prop := ∀ x, f x = f (x + 2)

theorem even_periodic_function 
  (f : ℝ → ℝ) 
  (h1 : is_even f) 
  (h2 : is_period_two f) 
  (h3 : ∀ x ∈ Set.Icc 2 3, f x = x) :
  ∀ x ∈ Set.Icc (-1) 0, f x = 2 - x :=
sorry

end even_periodic_function_l1050_105075


namespace problem_proof_l1050_105072

theorem problem_proof : (5 * 12) / (180 / 3) + 61 = 62 := by
  sorry

end problem_proof_l1050_105072


namespace phase_shift_sin_5x_minus_pi_half_l1050_105087

/-- The phase shift of the function y = sin(5x - π/2) is π/10 to the right or -π/10 to the left -/
theorem phase_shift_sin_5x_minus_pi_half :
  let f : ℝ → ℝ := λ x => Real.sin (5 * x - π / 2)
  ∃ φ : ℝ, (φ = π / 10 ∨ φ = -π / 10) ∧
    ∀ x : ℝ, f x = Real.sin (5 * (x - φ)) :=
by sorry

end phase_shift_sin_5x_minus_pi_half_l1050_105087


namespace money_left_l1050_105074

def initial_amount : ℚ := 200.50
def sweets_cost : ℚ := 35.25
def stickers_cost : ℚ := 10.75
def friend_gift : ℚ := 25.20
def num_friends : ℕ := 4
def charity_donation : ℚ := 15.30

theorem money_left : 
  initial_amount - (sweets_cost + stickers_cost + friend_gift * num_friends + charity_donation) = 38.40 := by
  sorry

end money_left_l1050_105074


namespace average_waiting_time_for_first_bite_l1050_105030

/-- The average waiting time for the first bite in a fishing scenario --/
theorem average_waiting_time_for_first_bite 
  (time_interval : ℝ) 
  (first_rod_bites : ℝ) 
  (second_rod_bites : ℝ) 
  (total_bites : ℝ) 
  (h1 : time_interval = 6)
  (h2 : first_rod_bites = 3)
  (h3 : second_rod_bites = 2)
  (h4 : total_bites = first_rod_bites + second_rod_bites) :
  (time_interval / total_bites) = 1.2 := by
  sorry

end average_waiting_time_for_first_bite_l1050_105030


namespace eggs_per_basket_l1050_105001

theorem eggs_per_basket (total_eggs : ℕ) (num_baskets : ℕ) 
  (h1 : total_eggs = 8484) (h2 : num_baskets = 303) :
  total_eggs / num_baskets = 28 := by
  sorry

end eggs_per_basket_l1050_105001


namespace medical_team_probability_l1050_105086

theorem medical_team_probability (male_doctors female_doctors team_size : ℕ) 
  (h1 : male_doctors = 6)
  (h2 : female_doctors = 3)
  (h3 : team_size = 5) : 
  (1 - (Nat.choose male_doctors team_size : ℚ) / (Nat.choose (male_doctors + female_doctors) team_size)) = 60/63 := by
  sorry

end medical_team_probability_l1050_105086


namespace square_perimeter_problem_l1050_105060

/-- Given two squares with perimeters 20 and 28, prove that a third square with side length
    equal to the positive difference of the side lengths of the first two squares has a perimeter of 8. -/
theorem square_perimeter_problem (square_I square_II square_III : ℝ → ℝ) :
  (∀ s, square_I s = 4 * s) →
  (∀ s, square_II s = 4 * s) →
  (∀ s, square_III s = 4 * s) →
  (∃ s_I, square_I s_I = 20) →
  (∃ s_II, square_II s_II = 28) →
  (∃ s_III, s_III = |s_I - s_II| ∧ square_III s_III = 8) :=
by sorry

end square_perimeter_problem_l1050_105060


namespace complex_fraction_simplification_l1050_105057

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (3 - 4*i) / (1 + i) = -1/2 - 7/2*i :=
by sorry

end complex_fraction_simplification_l1050_105057


namespace square_of_two_power_minus_twice_l1050_105085

theorem square_of_two_power_minus_twice (N : ℕ+) :
  (∃ k : ℕ, 2^N.val - 2 * N.val = k^2) ↔ N = 1 ∨ N = 2 := by
  sorry

end square_of_two_power_minus_twice_l1050_105085


namespace income_increase_theorem_l1050_105065

/-- Represents the student's income sources and total income -/
structure StudentIncome where
  scholarship : ℝ
  partTimeJob : ℝ
  parentalSupport : ℝ
  totalIncome : ℝ

/-- Theorem stating the relationship between income sources and total income increase -/
theorem income_increase_theorem (income : StudentIncome) 
  (h1 : income.scholarship + income.partTimeJob + income.parentalSupport = income.totalIncome)
  (h2 : 2 * income.scholarship + income.partTimeJob + income.parentalSupport = 1.05 * income.totalIncome)
  (h3 : income.scholarship + 2 * income.partTimeJob + income.parentalSupport = 1.15 * income.totalIncome) :
  income.scholarship + income.partTimeJob + 2 * income.parentalSupport = 1.8 * income.totalIncome := by
  sorry


end income_increase_theorem_l1050_105065


namespace simplify_fraction_product_l1050_105099

theorem simplify_fraction_product : 8 * (15 / 9) * (-21 / 35) = -8 / 3 := by
  sorry

end simplify_fraction_product_l1050_105099


namespace art_piece_original_price_l1050_105043

/-- Proves that the original purchase price of an art piece is $4000 -/
theorem art_piece_original_price :
  ∀ (original_price future_price : ℝ),
  future_price = 3 * original_price →
  future_price - original_price = 8000 →
  original_price = 4000 := by
sorry

end art_piece_original_price_l1050_105043


namespace girl_transfer_problem_l1050_105021

/-- Represents the number of girls in each group before and after transfers -/
structure GirlCounts where
  initial_B : ℕ
  initial_A : ℕ
  initial_C : ℕ
  final : ℕ

/-- Represents the number of girls transferred between groups -/
structure GirlTransfers where
  from_A : ℕ
  from_B : ℕ
  from_C : ℕ

/-- The theorem statement for the girl transfer problem -/
theorem girl_transfer_problem (g : GirlCounts) (t : GirlTransfers) : 
  g.initial_A = g.initial_B + 4 →
  g.initial_B = g.initial_C + 1 →
  t.from_C = 2 →
  g.final = g.initial_A - t.from_A + t.from_C →
  g.final = g.initial_B - t.from_B + t.from_A →
  g.final = g.initial_C - t.from_C + t.from_B →
  t.from_A = 5 ∧ t.from_B = 4 := by
  sorry


end girl_transfer_problem_l1050_105021


namespace small_circle_radius_l1050_105050

/-- Given a large circle with radius 10 meters containing seven smaller congruent circles
    that fit exactly along its diameter, prove that the radius of each smaller circle is 10/7 meters. -/
theorem small_circle_radius (R : ℝ) (n : ℕ) (r : ℝ) : 
  R = 10 → n = 7 → 2 * R = n * (2 * r) → r = 10 / 7 := by sorry

end small_circle_radius_l1050_105050


namespace carnival_friends_l1050_105004

theorem carnival_friends (total_tickets : ℕ) (tickets_per_person : ℕ) (h1 : total_tickets = 234) (h2 : tickets_per_person = 39) :
  total_tickets / tickets_per_person = 6 := by
  sorry

end carnival_friends_l1050_105004


namespace is_vertex_of_parabola_l1050_105046

/-- The parabola equation -/
def f (x : ℝ) : ℝ := -4 * x^2 - 16 * x - 20

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (-2, -4)

/-- Theorem stating that the given point is the vertex of the parabola -/
theorem is_vertex_of_parabola :
  ∀ x : ℝ, f x ≤ f (vertex.1) ∧ f (vertex.1) = vertex.2 :=
sorry

end is_vertex_of_parabola_l1050_105046


namespace combinatorial_number_identity_l1050_105003

theorem combinatorial_number_identity (n r : ℕ) (h1 : n > r) (h2 : r ≥ 1) :
  Nat.choose n r = (n / r) * Nat.choose (n - 1) (r - 1) := by
  sorry

end combinatorial_number_identity_l1050_105003


namespace subtract_point_five_from_forty_seven_point_two_l1050_105096

theorem subtract_point_five_from_forty_seven_point_two : 47.2 - 0.5 = 46.7 := by
  sorry

end subtract_point_five_from_forty_seven_point_two_l1050_105096


namespace total_amount_proof_l1050_105024

/-- Given that r has two-thirds of the total amount and r has Rs. 2400,
    prove that the total amount p, q, and r have among themselves is Rs. 3600. -/
theorem total_amount_proof (r p q : ℕ) (h1 : r = 2400) (h2 : r * 3 = (p + q + r) * 2) :
  p + q + r = 3600 := by
  sorry

end total_amount_proof_l1050_105024


namespace plane_perpendicularity_l1050_105032

-- Define the type for planes
variable (Plane : Type)

-- Define the relations for parallel and perpendicular planes
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem plane_perpendicularity 
  (α β γ : Plane) 
  (h1 : parallel α β) 
  (h2 : perpendicular β γ) : 
  perpendicular α γ :=
sorry

end plane_perpendicularity_l1050_105032


namespace angle_trig_sum_l1050_105093

/-- Given an angle α with vertex at the origin, initial side on the positive x-axis,
    and a point P on its terminal side, prove that 2sinα + cosα = -2/5 -/
theorem angle_trig_sum (α : Real) (m : Real) (h1 : m < 0) :
  let P : Prod Real Real := (-4 * m, 3 * m)
  2 * Real.sin α + Real.cos α = -2/5 := by
  sorry

end angle_trig_sum_l1050_105093


namespace rectangle_area_l1050_105055

/-- Given a wire of length 32 cm bent into a rectangle with a length-to-width ratio of 5:3,
    the area of the resulting rectangle is 60 cm². -/
theorem rectangle_area (wire_length : ℝ) (length : ℝ) (width : ℝ) : 
  wire_length = 32 →
  length / width = 5 / 3 →
  2 * (length + width) = wire_length →
  length * width = 60 := by
sorry

end rectangle_area_l1050_105055


namespace andrews_to_jeffreys_steps_ratio_l1050_105062

theorem andrews_to_jeffreys_steps_ratio : 
  ∃ (a b : ℕ), a ≠ 0 ∧ b ≠ 0 ∧ 150 * b = 200 * a ∧ a = 3 ∧ b = 4 := by
  sorry

end andrews_to_jeffreys_steps_ratio_l1050_105062


namespace x_current_age_l1050_105058

/-- Proves that X's current age is 45 years given the provided conditions -/
theorem x_current_age (x y : ℕ) : 
  (x - 3 = 2 * (y - 3)) →  -- X's age was double Y's age three years ago
  (x + y + 14 = 83) →      -- Seven years from now, the sum of their ages will be 83
  x = 45 :=                -- X's current age is 45
by
  sorry

#check x_current_age

end x_current_age_l1050_105058


namespace survey_problem_l1050_105061

theorem survey_problem (A B C : ℝ) 
  (h_A : A = 50)
  (h_B : B = 30)
  (h_C : C = 20)
  (h_union : A + B + C - 17 = 78) 
  (h_multiple : 17 ≤ A + B + C - 78) :
  A + B + C - 78 = 5 := by
sorry

end survey_problem_l1050_105061


namespace total_doors_is_3600_l1050_105029

/-- Calculates the number of doors needed for a building with uniform floor plans -/
def doorsForUniformBuilding (floors : ℕ) (apartmentsPerFloor : ℕ) (doorsPerApartment : ℕ) : ℕ :=
  floors * apartmentsPerFloor * doorsPerApartment

/-- Calculates the number of doors needed for a building with alternating floor plans -/
def doorsForAlternatingBuilding (floors : ℕ) (oddApartments : ℕ) (evenApartments : ℕ) (doorsPerApartment : ℕ) : ℕ :=
  ((floors + 1) / 2 * oddApartments + (floors / 2) * evenApartments) * doorsPerApartment

/-- The total number of doors needed for all four buildings -/
def totalDoors : ℕ :=
  doorsForUniformBuilding 15 5 8 +
  doorsForUniformBuilding 25 6 10 +
  doorsForAlternatingBuilding 20 7 5 9 +
  doorsForAlternatingBuilding 10 8 4 7

theorem total_doors_is_3600 : totalDoors = 3600 := by
  sorry

end total_doors_is_3600_l1050_105029


namespace max_value_constrained_product_l1050_105016

theorem max_value_constrained_product (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 5*x + 3*y < 75) :
  x * y * (75 - 5*x - 3*y) ≤ 3125/3 :=
sorry

end max_value_constrained_product_l1050_105016


namespace inequality_proof_l1050_105018

theorem inequality_proof (a b c d : ℝ) 
  (non_neg_a : a ≥ 0) (non_neg_b : b ≥ 0) (non_neg_c : c ≥ 0) (non_neg_d : d ≥ 0)
  (sum_eq_four : a + b + c + d = 4) : 
  (a*b + c*d) * (a*c + b*d) * (a*d + b*c) ≤ 8 := by
sorry

end inequality_proof_l1050_105018


namespace quadratic_system_solution_l1050_105013

theorem quadratic_system_solution :
  ∃ (x₁ y₁ x₂ y₂ : ℚ),
    (x₁ = 2/9 ∧ y₁ = 35/117) ∧
    (x₂ = -1 ∧ y₂ = -5/26) ∧
    (∀ x y : ℚ, 9*x^2 + 8*x - 2 = 0 ∧ 27*x^2 + 26*y + 8*x - 14 = 0 →
      (x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂)) :=
by sorry

end quadratic_system_solution_l1050_105013


namespace paris_travel_distance_l1050_105034

theorem paris_travel_distance (total_distance train_distance bus_distance cab_distance : ℝ) : 
  total_distance = 500 ∧
  bus_distance = train_distance / 2 ∧
  cab_distance = bus_distance / 3 ∧
  total_distance = train_distance + bus_distance + cab_distance →
  train_distance = 300 := by
sorry

end paris_travel_distance_l1050_105034


namespace hotel_guests_count_l1050_105006

/-- The number of guests attending the Oates reunion -/
def oates_attendees : ℕ := 50

/-- The number of guests attending the Hall reunion -/
def hall_attendees : ℕ := 62

/-- The number of guests attending both reunions -/
def both_attendees : ℕ := 12

/-- The total number of guests at the hotel -/
def total_guests : ℕ := (oates_attendees - both_attendees) + (hall_attendees - both_attendees) + both_attendees

theorem hotel_guests_count :
  total_guests = 100 := by sorry

end hotel_guests_count_l1050_105006


namespace jimin_calculation_l1050_105080

theorem jimin_calculation (x : ℤ) (h : 20 - x = 60) : 34 * x = -1360 := by
  sorry

end jimin_calculation_l1050_105080


namespace collinear_points_x_value_l1050_105045

/-- Three points are collinear if the slope between any two pairs of points is equal -/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℚ) : Prop :=
  (y₂ - y₁) * (x₃ - x₂) = (y₃ - y₂) * (x₂ - x₁)

theorem collinear_points_x_value :
  ∀ x : ℚ, collinear 2 7 10 x 25 (-2) → x = 89 / 23 :=
by
  sorry

#check collinear_points_x_value

end collinear_points_x_value_l1050_105045


namespace homework_problem_count_l1050_105088

theorem homework_problem_count (math_pages : ℕ) (reading_pages : ℕ) (problems_per_page : ℕ) : 
  math_pages = 2 → reading_pages = 4 → problems_per_page = 5 →
  (math_pages + reading_pages) * problems_per_page = 30 := by
sorry

end homework_problem_count_l1050_105088


namespace bird_families_to_africa_l1050_105020

theorem bird_families_to_africa (total : ℕ) (to_asia : ℕ) (remaining : ℕ) : 
  total = 85 → to_asia = 37 → remaining = 25 → total - to_asia - remaining = 23 :=
by sorry

end bird_families_to_africa_l1050_105020


namespace geometric_sequence_special_term_l1050_105095

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The 14th term of a geometric sequence -/
def a_14 (a : ℕ → ℝ) : ℝ := a 14

/-- The 4th term of a geometric sequence -/
def a_4 (a : ℕ → ℝ) : ℝ := a 4

/-- The 24th term of a geometric sequence -/
def a_24 (a : ℕ → ℝ) : ℝ := a 24

/-- Theorem: In a geometric sequence, if a_4 and a_24 are roots of 3x^2 - 2014x + 9 = 0, then a_14 = √3 -/
theorem geometric_sequence_special_term (a : ℕ → ℝ) :
  geometric_sequence a →
  (3 * (a_4 a)^2 - 2014 * (a_4 a) + 9 = 0) →
  (3 * (a_24 a)^2 - 2014 * (a_24 a) + 9 = 0) →
  a_14 a = Real.sqrt 3 := by
  sorry

end geometric_sequence_special_term_l1050_105095


namespace blue_notes_under_red_l1050_105054

theorem blue_notes_under_red (red_rows : Nat) (red_per_row : Nat) (additional_blue : Nat) (total_notes : Nat) : Nat :=
  let total_red := red_rows * red_per_row
  let total_blue := total_notes - total_red
  let blue_under_red := (total_blue - additional_blue) / total_red
  blue_under_red

#check blue_notes_under_red 5 6 10 100

end blue_notes_under_red_l1050_105054


namespace rectangular_box_surface_area_l1050_105076

theorem rectangular_box_surface_area 
  (a b c : ℝ) 
  (edge_sum : 4 * a + 4 * b + 4 * c = 160) 
  (diagonal : a^2 + b^2 + c^2 = 25^2) : 
  2 * (a * b + b * c + c * a) = 975 := by
  sorry

end rectangular_box_surface_area_l1050_105076


namespace arithmetic_computation_l1050_105071

theorem arithmetic_computation : (-9 * 5) - (-7 * -2) + (11 * -4) = -103 := by
  sorry

end arithmetic_computation_l1050_105071


namespace ken_kept_twenty_pencils_l1050_105068

/-- The number of pencils Ken initially had -/
def initial_pencils : ℕ := 50

/-- The number of pencils Ken gave to Manny -/
def pencils_to_manny : ℕ := 10

/-- The number of additional pencils Ken gave to Nilo compared to Manny -/
def additional_pencils_to_nilo : ℕ := 10

/-- The number of pencils Ken kept -/
def pencils_kept : ℕ := initial_pencils - (pencils_to_manny + (pencils_to_manny + additional_pencils_to_nilo))

theorem ken_kept_twenty_pencils : pencils_kept = 20 := by sorry

end ken_kept_twenty_pencils_l1050_105068


namespace spherical_coordinates_of_point_A_l1050_105066

theorem spherical_coordinates_of_point_A :
  let x : ℝ := (3 * Real.sqrt 3) / 2
  let y : ℝ := 9 / 2
  let z : ℝ := 3
  let r : ℝ := Real.sqrt (x^2 + y^2 + z^2)
  let θ : ℝ := Real.arctan (y / x)
  let φ : ℝ := Real.arccos (z / r)
  (r = 6) ∧ (θ = π / 3) ∧ (φ = π / 3) := by sorry

end spherical_coordinates_of_point_A_l1050_105066


namespace children_on_bus_after_stop_l1050_105023

/-- The number of children on a bus after a stop, given the initial number,
    the number who got on, and the relationship between those who got on and off. -/
theorem children_on_bus_after_stop
  (initial : ℕ)
  (got_on : ℕ)
  (h1 : initial = 28)
  (h2 : got_on = 82)
  (h3 : ∃ (got_off : ℕ), got_on = got_off + 2) :
  initial + got_on - (got_on - 2) = 28 := by
  sorry


end children_on_bus_after_stop_l1050_105023


namespace computer_pricing_l1050_105009

/-- 
Given a computer's selling price and profit percentage, 
calculate the new selling price for a different profit percentage.
-/
theorem computer_pricing (initial_price : ℝ) (initial_profit_percent : ℝ) 
  (new_profit_percent : ℝ) (new_price : ℝ) :
  initial_price = (1 + initial_profit_percent / 100) * (initial_price / (1 + initial_profit_percent / 100)) →
  new_price = (1 + new_profit_percent / 100) * (initial_price / (1 + initial_profit_percent / 100)) →
  initial_price = 2240 →
  initial_profit_percent = 40 →
  new_profit_percent = 50 →
  new_price = 2400 :=
by sorry

end computer_pricing_l1050_105009


namespace complex_magnitude_l1050_105040

theorem complex_magnitude (z : ℂ) (h : z = Complex.mk 2 (-1)) : Complex.abs z = Real.sqrt 5 := by
  sorry

end complex_magnitude_l1050_105040


namespace sum_squares_possible_values_l1050_105063

/-- Given a positive real number A, prove that for any y in the open interval (0, A^2),
    there exists a sequence of positive real numbers {x_j} such that the sum of x_j equals A
    and the sum of x_j^2 equals y. -/
theorem sum_squares_possible_values (A : ℝ) (hA : A > 0) (y : ℝ) (hy1 : y > 0) (hy2 : y < A^2) :
  ∃ (x : ℕ → ℝ), (∀ j, x j > 0) ∧
    (∑' j, x j) = A ∧
    (∑' j, (x j)^2) = y :=
sorry

end sum_squares_possible_values_l1050_105063


namespace rectangle_ratio_l1050_105059

/-- Proves that for a rectangle with width 5 inches and area 50 square inches, 
    the ratio of its length to its width is 2. -/
theorem rectangle_ratio : 
  ∀ (length width : ℝ), 
    width = 5 → 
    length * width = 50 → 
    length / width = 2 := by
  sorry

end rectangle_ratio_l1050_105059


namespace max_polygon_area_l1050_105022

/-- A point with integer coordinates satisfying the given conditions -/
structure ValidPoint where
  x : ℕ+
  y : ℕ+
  cond1 : x ∣ (2 * y + 1)
  cond2 : y ∣ (2 * x + 1)

/-- The set of all valid points -/
def ValidPoints : Set ValidPoint := {p : ValidPoint | True}

/-- The area of a polygon formed by a set of points -/
noncomputable def polygonArea (points : Set ValidPoint) : ℝ := sorry

/-- The maximum area of a polygon formed by valid points -/
theorem max_polygon_area :
  ∃ (points : Set ValidPoint), points ⊆ ValidPoints ∧ polygonArea points = 20 ∧
    ∀ (otherPoints : Set ValidPoint), otherPoints ⊆ ValidPoints →
      polygonArea otherPoints ≤ 20 := by sorry

end max_polygon_area_l1050_105022


namespace non_shaded_perimeter_l1050_105056

/-- Calculates the perimeter of the non-shaded region in a composite figure --/
theorem non_shaded_perimeter (outer_length outer_width side_length side_width shaded_area : ℝ) : 
  outer_length = 10 →
  outer_width = 8 →
  side_length = 2 →
  side_width = 4 →
  shaded_area = 78 →
  let total_area := outer_length * outer_width + side_length * side_width
  let non_shaded_area := total_area - shaded_area
  let non_shaded_width := side_length
  let non_shaded_length := non_shaded_area / non_shaded_width
  2 * (non_shaded_length + non_shaded_width) = 14 :=
by sorry


end non_shaded_perimeter_l1050_105056


namespace additional_amount_for_free_shipping_l1050_105083

-- Define the book prices and discount
def book1_price : ℝ := 13.00
def book2_price : ℝ := 15.00
def book3_price : ℝ := 10.00
def book4_price : ℝ := 10.00
def discount_rate : ℝ := 0.25
def free_shipping_threshold : ℝ := 50.00

-- Calculate the discounted prices for books 1 and 2
def discounted_book1_price : ℝ := book1_price * (1 - discount_rate)
def discounted_book2_price : ℝ := book2_price * (1 - discount_rate)

-- Calculate the total cost of all four books
def total_cost : ℝ := discounted_book1_price + discounted_book2_price + book3_price + book4_price

-- Define the additional amount needed for free shipping
def additional_amount : ℝ := free_shipping_threshold - total_cost

-- Theorem to prove
theorem additional_amount_for_free_shipping :
  additional_amount = 9.00 := by sorry

end additional_amount_for_free_shipping_l1050_105083


namespace probability_both_selected_l1050_105077

theorem probability_both_selected (prob_ram : ℚ) (prob_ravi : ℚ) 
  (h1 : prob_ram = 4/7) (h2 : prob_ravi = 1/5) : 
  prob_ram * prob_ravi = 4/35 := by
  sorry

end probability_both_selected_l1050_105077

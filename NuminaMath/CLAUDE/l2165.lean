import Mathlib

namespace smallest_n_terminating_with_2_l2165_216557

def is_terminating_decimal (n : ℕ+) : Prop :=
  ∃ a b : ℕ, n = 2^a * 5^b

def contains_digit_2 (n : ℕ) : Prop :=
  ∃ k m : ℕ, n = 10 * k + 2 + 10 * m

theorem smallest_n_terminating_with_2 :
  ∃ n : ℕ+, 
    is_terminating_decimal n ∧ 
    contains_digit_2 n.val ∧ 
    (∀ m : ℕ+, m < n → ¬(is_terminating_decimal m ∧ contains_digit_2 m.val)) ∧
    n = 2 :=
  sorry

end smallest_n_terminating_with_2_l2165_216557


namespace x1_value_l2165_216562

theorem x1_value (x1 x2 x3 : ℝ) 
  (h1 : 0 ≤ x3 ∧ x3 ≤ x2 ∧ x2 ≤ x1 ∧ x1 ≤ 1) 
  (h2 : (1 - x1)^2 + (x1 - x2)^2 + (x2 - x3)^2 + x3^2 = 1/2) : 
  x1 = 2/3 := by
sorry

end x1_value_l2165_216562


namespace min_groups_for_athletes_l2165_216507

theorem min_groups_for_athletes (total_athletes : ℕ) (max_group_size : ℕ) (h1 : total_athletes = 30) (h2 : max_group_size = 12) : 
  ∃ (num_groups : ℕ), 
    num_groups ≥ 1 ∧ 
    num_groups ≤ total_athletes ∧
    ∃ (group_size : ℕ), 
      group_size > 0 ∧
      group_size ≤ max_group_size ∧
      total_athletes = num_groups * group_size ∧
      ∀ (n : ℕ), n < num_groups → 
        ¬∃ (g : ℕ), g > 0 ∧ g ≤ max_group_size ∧ total_athletes = n * g :=
by
  sorry

end min_groups_for_athletes_l2165_216507


namespace f_minus_five_eq_zero_l2165_216587

open Function

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f x = -f (-x)

theorem f_minus_five_eq_zero
  (f : ℝ → ℝ)
  (h1 : is_even (fun x ↦ f (1 - 2*x)))
  (h2 : is_odd (fun x ↦ f (x - 1))) :
  f (-5) = 0 := by
  sorry

end f_minus_five_eq_zero_l2165_216587


namespace cubic_polynomial_r_value_l2165_216551

/-- A cubic polynomial with integer coefficients -/
structure CubicPolynomial where
  p : Int
  q : Int
  r : Int

/-- The property that all roots of a cubic polynomial are negative integers -/
def hasAllNegativeIntegerRoots (g : CubicPolynomial) : Prop := sorry

/-- Theorem: For a cubic polynomial g(x) = x^3 + px^2 + qx + r with all roots being negative integers
    and p + q + r = 100, the value of r must be 0 -/
theorem cubic_polynomial_r_value (g : CubicPolynomial)
    (h1 : hasAllNegativeIntegerRoots g)
    (h2 : g.p + g.q + g.r = 100) :
    g.r = 0 := by sorry

end cubic_polynomial_r_value_l2165_216551


namespace not_sufficient_not_necessary_l2165_216503

theorem not_sufficient_not_necessary (a b : ℝ) : 
  (a ≠ 5 ∧ b ≠ -5) ↔ (a + b ≠ 0) → False :=
by sorry

end not_sufficient_not_necessary_l2165_216503


namespace no_linear_term_implies_m_equals_negative_three_l2165_216584

theorem no_linear_term_implies_m_equals_negative_three (m : ℝ) :
  (∀ x : ℝ, ∃ a b : ℝ, (x + m) * (x + 3) = a * x^2 + b) →
  m = -3 :=
sorry

end no_linear_term_implies_m_equals_negative_three_l2165_216584


namespace three_face_painted_count_l2165_216527

/-- Represents a cuboid made of small cubes -/
structure Cuboid where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents the state of the cuboid after modifications -/
structure ModifiedCuboid extends Cuboid where
  removed_cubes : ℕ
  surface_painted : Bool

/-- Counts the number of small cubes with three painted faces -/
def count_three_face_painted (c : ModifiedCuboid) : ℕ :=
  sorry

/-- The main theorem to prove -/
theorem three_face_painted_count 
  (c : ModifiedCuboid) 
  (h1 : c.length = 12 ∧ c.width = 3 ∧ c.height = 6)
  (h2 : c.removed_cubes = 3)
  (h3 : c.surface_painted = true) :
  count_three_face_painted c = 8 :=
sorry

end three_face_painted_count_l2165_216527


namespace equation_solution_l2165_216520

theorem equation_solution : ∃! y : ℝ, 4 + 2.3 * y = 1.7 * y - 20 :=
by
  use -40
  constructor
  · -- Prove that y = -40 satisfies the equation
    sorry
  · -- Prove uniqueness
    sorry

end equation_solution_l2165_216520


namespace min_cars_theorem_l2165_216538

/-- Calculates the minimum number of cars needed for a family where each car must rest one day a week and all adults want to drive daily. -/
def min_cars_needed (num_adults : ℕ) : ℕ :=
  if num_adults ≤ 6 then
    num_adults + 1
  else
    (num_adults * 7 + 5) / 6

theorem min_cars_theorem (num_adults : ℕ) :
  (num_adults = 5 → min_cars_needed num_adults = 6) ∧
  (num_adults = 8 → min_cars_needed num_adults = 10) :=
by sorry

#eval min_cars_needed 5  -- Should output 6
#eval min_cars_needed 8  -- Should output 10

end min_cars_theorem_l2165_216538


namespace bunny_burrow_exits_l2165_216525

-- Define the rate at which a bunny comes out of its burrow
def bunny_rate : ℕ := 3

-- Define the number of bunnies
def num_bunnies : ℕ := 20

-- Define the time period in hours
def time_period : ℕ := 10

-- Define the number of minutes in an hour
def minutes_per_hour : ℕ := 60

-- Theorem statement
theorem bunny_burrow_exits :
  bunny_rate * minutes_per_hour * time_period * num_bunnies = 36000 := by
  sorry

end bunny_burrow_exits_l2165_216525


namespace area_between_graphs_l2165_216561

-- Define the functions
def f (x : ℝ) : ℝ := |2 * x| - 3
def g (x : ℝ) : ℝ := |x|

-- Define the area enclosed by the graphs
def enclosed_area : ℝ := 9

-- Theorem statement
theorem area_between_graphs :
  (∃ (a b : ℝ), a < b ∧
    (∀ x ∈ Set.Icc a b, f x ≠ g x) ∧
    (∀ x ∈ Set.Ioi b, f x = g x) ∧
    (∀ x ∈ Set.Iio a, f x = g x)) →
  (∫ (x : ℝ) in Set.Icc (-3) 3, |f x - g x|) = enclosed_area :=
sorry

end area_between_graphs_l2165_216561


namespace negation_equivalence_l2165_216544

variable (m : ℤ)

theorem negation_equivalence :
  (¬ ∃ x : ℤ, x^2 + x + m < 0) ↔ (∀ x : ℤ, x^2 + x + m ≥ 0) :=
by sorry

end negation_equivalence_l2165_216544


namespace sturgeons_caught_l2165_216569

/-- Given the total number of fishes caught and the number of pikes and herrings,
    prove that the number of sturgeons caught is 40. -/
theorem sturgeons_caught (total_fish : ℕ) (pikes : ℕ) (herrings : ℕ) 
    (h1 : total_fish = 145)
    (h2 : pikes = 30)
    (h3 : herrings = 75) :
    total_fish - (pikes + herrings) = 40 := by
  sorry

end sturgeons_caught_l2165_216569


namespace orange_apple_ratio_l2165_216535

/-- Represents the contents of a shopping cart with apples, oranges, and pears. -/
structure ShoppingCart where
  apples : ℕ
  oranges : ℕ
  pears : ℕ

/-- Checks if the shopping cart satisfies the given conditions. -/
def satisfiesConditions (cart : ShoppingCart) : Prop :=
  cart.pears = 4 * cart.oranges ∧
  cart.apples = (1 / 12 : ℚ) * cart.pears

/-- The main theorem stating the relationship between oranges and apples. -/
theorem orange_apple_ratio (cart : ShoppingCart) 
  (h : satisfiesConditions cart) (h_nonzero : cart.apples > 0) : 
  cart.oranges = 3 * cart.apples := by
  sorry


end orange_apple_ratio_l2165_216535


namespace shift_repeating_segment_2011th_digit_6_l2165_216523

/-- Represents a repeating decimal with an initial non-repeating part and a repeating segment. -/
structure RepeatingDecimal where
  initial : ℚ
  repeating : List ℕ

/-- Shifts the repeating segment of a repeating decimal. -/
def shiftRepeatingSegment (d : RepeatingDecimal) (n : ℕ) : RepeatingDecimal :=
  sorry

/-- Gets the nth digit after the decimal point in a repeating decimal. -/
def nthDigitAfterDecimal (d : RepeatingDecimal) (n : ℕ) : ℕ :=
  sorry

/-- The main theorem about shifting the repeating segment. -/
theorem shift_repeating_segment_2011th_digit_6 (d : RepeatingDecimal) :
  d.initial = 0.1 ∧ d.repeating = [2, 3, 4, 5, 6, 7, 8] →
  ∃ (k : ℕ), 
    let d' := shiftRepeatingSegment d k
    nthDigitAfterDecimal d' 2011 = 6 ∧
    d'.initial = 0.1 ∧ d'.repeating = [2, 3, 4, 5, 6, 7, 8] :=
  sorry

end shift_repeating_segment_2011th_digit_6_l2165_216523


namespace polygon_triangulation_l2165_216598

/-- A color type with three possible values -/
inductive Color
  | one
  | two
  | three

/-- A vertex of a polygon -/
structure Vertex where
  color : Color

/-- A convex polygon -/
structure ConvexPolygon where
  vertices : List Vertex
  convex : Bool
  all_colors_present : Bool
  no_adjacent_same_color : Bool

/-- A triangle with three vertices -/
structure Triangle where
  v1 : Vertex
  v2 : Vertex
  v3 : Vertex

/-- A triangulation of a polygon -/
structure Triangulation where
  triangles : List Triangle

/-- The main theorem -/
theorem polygon_triangulation (p : ConvexPolygon) :
  p.convex ∧ p.all_colors_present ∧ p.no_adjacent_same_color →
  ∃ (t : Triangulation), ∀ (triangle : Triangle), triangle ∈ t.triangles →
    triangle.v1.color ≠ triangle.v2.color ∧
    triangle.v2.color ≠ triangle.v3.color ∧
    triangle.v3.color ≠ triangle.v1.color :=
sorry

end polygon_triangulation_l2165_216598


namespace hcf_of_36_and_84_l2165_216501

theorem hcf_of_36_and_84 : Nat.gcd 36 84 = 12 := by
  sorry

end hcf_of_36_and_84_l2165_216501


namespace zoe_gre_exam_month_l2165_216509

-- Define the months as an enumeration
inductive Month
  | January | February | March | April | May | June | July | August | September | October | November | December

-- Define a function to add months
def addMonths (start : Month) (n : Nat) : Month :=
  match n with
  | 0 => start
  | Nat.succ m => addMonths (match start with
    | Month.January => Month.February
    | Month.February => Month.March
    | Month.March => Month.April
    | Month.April => Month.May
    | Month.May => Month.June
    | Month.June => Month.July
    | Month.July => Month.August
    | Month.August => Month.September
    | Month.September => Month.October
    | Month.October => Month.November
    | Month.November => Month.December
    | Month.December => Month.January
  ) m

-- Theorem statement
theorem zoe_gre_exam_month :
  addMonths Month.April 2 = Month.June :=
by sorry

end zoe_gre_exam_month_l2165_216509


namespace four_distinct_roots_implies_c_magnitude_l2165_216576

/-- The polynomial Q(x) -/
def Q (c x : ℂ) : ℂ := (x^2 - 2*x + 3) * (x^2 - c*x + 6) * (x^2 - 4*x + 12)

/-- The theorem statement -/
theorem four_distinct_roots_implies_c_magnitude (c : ℂ) :
  (∃ (s : Finset ℂ), s.card = 4 ∧ (∀ x ∈ s, Q c x = 0) ∧
   (∀ x : ℂ, Q c x = 0 → x ∈ s)) →
  Complex.abs c = Real.sqrt 11 := by
  sorry

end four_distinct_roots_implies_c_magnitude_l2165_216576


namespace three_circles_middle_radius_l2165_216534

/-- Configuration of three circles with two common tangent lines -/
structure ThreeCirclesConfig where
  r_large : ℝ  -- radius of the largest circle
  r_small : ℝ  -- radius of the smallest circle
  r_middle : ℝ  -- radius of the middle circle
  tangent_lines : ℕ  -- number of common tangent lines

/-- Theorem: In a configuration of three circles with two common tangent lines,
    if the radius of the largest circle is 18 and the radius of the smallest circle is 8,
    then the radius of the middle circle is 12. -/
theorem three_circles_middle_radius 
  (config : ThreeCirclesConfig) 
  (h1 : config.r_large = 18) 
  (h2 : config.r_small = 8) 
  (h3 : config.tangent_lines = 2) : 
  config.r_middle = 12 := by
  sorry

end three_circles_middle_radius_l2165_216534


namespace double_inequality_solution_l2165_216568

theorem double_inequality_solution (x : ℝ) : 
  -1 < (x^2 - 16*x + 24) / (x^2 - 4*x + 8) ∧ 
  (x^2 - 16*x + 24) / (x^2 - 4*x + 8) < 1 ↔ 
  (3/2 < x ∧ x < 4) ∨ (8 < x) := by
  sorry

end double_inequality_solution_l2165_216568


namespace f_difference_l2165_216530

/-- The function f(x) = x^4 + 2x^3 + 3x^2 + 7x -/
def f (x : ℝ) : ℝ := x^4 + 2*x^3 + 3*x^2 + 7*x

/-- Theorem: f(3) - f(-3) = 150 -/
theorem f_difference : f 3 - f (-3) = 150 := by sorry

end f_difference_l2165_216530


namespace coffee_mixture_cost_l2165_216585

/-- Proves that the cost of the second coffee brand is $116.67 per kg -/
theorem coffee_mixture_cost (brand1_cost brand2_cost mixture_price profit_rate : ℝ)
  (h1 : brand1_cost = 200)
  (h2 : mixture_price = 177)
  (h3 : profit_rate = 0.18)
  (h4 : (2 * brand1_cost + 3 * brand2_cost) / 5 * (1 + profit_rate) = mixture_price) :
  brand2_cost = 116.67 := by
  sorry

end coffee_mixture_cost_l2165_216585


namespace largest_visible_sum_l2165_216554

/-- Represents a standard die with opposite faces summing to 7 -/
structure Die where
  faces : Fin 6 → Nat
  opposite_sum : ∀ i : Fin 3, faces i + faces (i + 3) = 7

/-- Represents a 3x3x3 cube assembled from 27 dice -/
structure Cube where
  dice : Fin 3 → Fin 3 → Fin 3 → Die

/-- Calculates the sum of visible values on the 6 faces of the cube -/
def visibleSum (c : Cube) : Nat :=
  sorry

/-- States that the largest possible sum of visible values is 288 -/
theorem largest_visible_sum (c : Cube) : 
  visibleSum c ≤ 288 ∧ ∃ c' : Cube, visibleSum c' = 288 :=
sorry

end largest_visible_sum_l2165_216554


namespace pension_program_participation_rate_l2165_216555

structure Shift where
  members : ℕ
  participation_rate : ℚ

def company_x : List Shift := [
  { members := 60, participation_rate := 1/5 },
  { members := 50, participation_rate := 2/5 },
  { members := 40, participation_rate := 1/10 }
]

theorem pension_program_participation_rate :
  let total_workers := (company_x.map (λ s => s.members)).sum
  let participating_workers := (company_x.map (λ s => (s.members : ℚ) * s.participation_rate)).sum
  participating_workers / total_workers = 6/25 := by
sorry

end pension_program_participation_rate_l2165_216555


namespace c_share_is_27_l2165_216563

/-- Represents the rent share calculation for a pasture -/
structure PastureRent where
  a_oxen : ℕ
  a_months : ℕ
  b_oxen : ℕ
  b_months : ℕ
  c_oxen : ℕ
  c_months : ℕ
  total_rent : ℕ

/-- Calculates the share of rent for person C -/
def calculate_c_share (pr : PastureRent) : ℚ :=
  let total_ox_months := pr.a_oxen * pr.a_months + pr.b_oxen * pr.b_months + pr.c_oxen * pr.c_months
  let rent_per_ox_month := pr.total_rent / total_ox_months
  (pr.c_oxen * pr.c_months * rent_per_ox_month : ℚ)

/-- Theorem stating that C's share of rent is 27 Rs -/
theorem c_share_is_27 (pr : PastureRent) 
  (h1 : pr.a_oxen = 10) (h2 : pr.a_months = 7)
  (h3 : pr.b_oxen = 12) (h4 : pr.b_months = 5)
  (h5 : pr.c_oxen = 15) (h6 : pr.c_months = 3)
  (h7 : pr.total_rent = 105) : 
  calculate_c_share pr = 27 := by
  sorry


end c_share_is_27_l2165_216563


namespace extremum_implies_a_equals_12_l2165_216592

/-- The function f(x) = a * ln(x) + x^2 - 10x has an extremum at x = 3 -/
def has_extremum_at_3 (a : ℝ) : Prop :=
  let f := fun (x : ℝ) => a * Real.log x + x^2 - 10*x
  ∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), 0 < |x - 3| ∧ |x - 3| < ε → 
    (f x - f 3) * (x - 3) ≤ 0

/-- Given that f(x) = a * ln(x) + x^2 - 10x has an extremum at x = 3, prove that a = 12 -/
theorem extremum_implies_a_equals_12 : 
  has_extremum_at_3 a → a = 12 := by sorry

end extremum_implies_a_equals_12_l2165_216592


namespace smallest_next_divisor_after_221_l2165_216567

theorem smallest_next_divisor_after_221 (n : ℕ) :
  (n ≥ 1000 ∧ n ≤ 9999) →  -- n is a 4-digit number
  Even n →                 -- n is even
  221 ∣ n →                -- 221 is a divisor of n
  ∃ (d : ℕ), d ∣ n ∧ d > 221 ∧ d ≤ 238 ∧ ∀ (x : ℕ), x ∣ n → x > 221 → x ≥ d :=
by sorry

end smallest_next_divisor_after_221_l2165_216567


namespace unique_solution_system_l2165_216594

theorem unique_solution_system (x y z : ℝ) : 
  x ≠ 0 → y ≠ 0 → z ≠ 0 →
  (1 / x + 1 / y + 1 / z = 3) →
  (1 / (x * y) + 1 / (y * z) + 1 / (z * x) = 3) →
  (1 / (x * y * z) = 1) →
  (x = 1 ∧ y = 1 ∧ z = 1) := by
  sorry

end unique_solution_system_l2165_216594


namespace inequality_for_positive_integers_l2165_216536

theorem inequality_for_positive_integers (n : ℕ) (h : n > 0) : 2 * n - 1 < (n + 1)^2 := by
  sorry

end inequality_for_positive_integers_l2165_216536


namespace quiz_probability_l2165_216513

theorem quiz_probability : 
  let n : ℕ := 6  -- number of questions
  let m : ℕ := 6  -- number of possible answers per question
  let p : ℚ := 1 - (m - 1 : ℚ) / m  -- probability of getting one question right
  1 - (1 - p) ^ n = 31031 / 46656 :=
by sorry

end quiz_probability_l2165_216513


namespace rental_company_fixed_amount_l2165_216517

/-- The fixed amount charged by the first rental company -/
def F : ℝ := 41.95

/-- The per-mile rate charged by the first rental company -/
def rate1 : ℝ := 0.29

/-- The fixed amount charged by City Rentals -/
def fixed2 : ℝ := 38.95

/-- The per-mile rate charged by City Rentals -/
def rate2 : ℝ := 0.31

/-- The number of miles driven -/
def miles : ℝ := 150.0

theorem rental_company_fixed_amount :
  F + rate1 * miles = fixed2 + rate2 * miles :=
sorry

end rental_company_fixed_amount_l2165_216517


namespace parabola_vertex_l2165_216542

/-- Given a parabola y = -x^2 + ax + b ≤ 0 with roots at x = -4 and x = 6,
    prove that its vertex is at (1, 25). -/
theorem parabola_vertex (a b : ℝ) :
  (∀ x, -x^2 + a*x + b ≤ 0 ↔ x ∈ Set.Ici 6 ∪ Set.Iic (-4)) →
  ∃ k, -1^2 + a*1 + b = k ∧ ∀ x, -x^2 + a*x + b ≤ k :=
by sorry

end parabola_vertex_l2165_216542


namespace number_equation_solution_l2165_216547

theorem number_equation_solution : 
  ∃ n : ℝ, (n * n - 30158 * 30158) / (n - 30158) = 100000 ∧ n = 69842 := by
  sorry

end number_equation_solution_l2165_216547


namespace deepak_age_l2165_216526

theorem deepak_age (rahul_age deepak_age : ℕ) : 
  (rahul_age : ℚ) / deepak_age = 4 / 3 →
  rahul_age + 2 = 26 →
  deepak_age = 18 := by
sorry

end deepak_age_l2165_216526


namespace butterfly_collection_l2165_216578

theorem butterfly_collection (total : ℕ) (blue : ℕ) : 
  total = 19 → 
  blue = 6 → 
  ∃ (yellow : ℕ), blue = 2 * yellow → 
  ∃ (black : ℕ), black = total - (blue + yellow) ∧ black = 10 := by
  sorry

end butterfly_collection_l2165_216578


namespace bacteria_population_growth_l2165_216564

def bacteria_count (n : ℕ) : ℕ :=
  if n % 2 = 0 then
    2^(n/2 + 1)
  else
    2^((n-1)/2 + 1)

theorem bacteria_population_growth (n : ℕ) :
  bacteria_count n = if n % 2 = 0 then 2^(n/2 + 1) else 2^((n-1)/2 + 1) :=
by sorry

end bacteria_population_growth_l2165_216564


namespace equation_property_l2165_216515

theorem equation_property (a b : ℝ) : 3 * a = 3 * b → a = b := by
  sorry

end equation_property_l2165_216515


namespace midpoint_sum_invariant_l2165_216591

/-- Represents a polygon with n vertices in the Cartesian plane -/
structure Polygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- Creates a new polygon from the midpoints of the sides of the given polygon -/
def midpointPolygon {n : ℕ} (p : Polygon n) : Polygon n := sorry

/-- Sums the x-coordinates of a polygon's vertices -/
def sumXCoordinates {n : ℕ} (p : Polygon n) : ℝ := sorry

theorem midpoint_sum_invariant (p₁ : Polygon 200) 
  (h : sumXCoordinates p₁ = 4018) :
  let p₂ := midpointPolygon p₁
  let p₃ := midpointPolygon p₂
  let p₄ := midpointPolygon p₃
  sumXCoordinates p₄ = 4018 := by sorry

end midpoint_sum_invariant_l2165_216591


namespace probability_two_black_cards_l2165_216553

theorem probability_two_black_cards (total_cards : ℕ) (black_cards : ℕ) 
  (h1 : total_cards = 52) 
  (h2 : black_cards = 26) :
  (black_cards * (black_cards - 1)) / (total_cards * (total_cards - 1)) = 25 / 102 := by
  sorry

end probability_two_black_cards_l2165_216553


namespace min_value_sum_reciprocals_l2165_216548

theorem min_value_sum_reciprocals (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) 
  (sum_eq_3 : x + y + z = 3) : 
  1 / (x + 3*y) + 1 / (y + 3*z) + 1 / (z + 3*x) ≥ 3/4 := by
  sorry

end min_value_sum_reciprocals_l2165_216548


namespace max_y_over_x_l2165_216540

theorem max_y_over_x (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_eq : 1 / x + 2 * y = 3) :
  y / x ≤ 9 / 8 ∧ ∃ (x₀ y₀ : ℝ), 0 < x₀ ∧ 0 < y₀ ∧ 1 / x₀ + 2 * y₀ = 3 ∧ y₀ / x₀ = 9 / 8 :=
sorry

end max_y_over_x_l2165_216540


namespace birds_on_fence_l2165_216545

theorem birds_on_fence (initial_birds : ℕ) : 
  initial_birds + 8 = 20 → initial_birds = 12 := by
  sorry

end birds_on_fence_l2165_216545


namespace sqrt_less_than_5x_iff_l2165_216528

theorem sqrt_less_than_5x_iff (x : ℝ) (h : x > 0) :
  Real.sqrt x < 5 * x ↔ x > 1 / 25 := by sorry

end sqrt_less_than_5x_iff_l2165_216528


namespace solve_exponential_equation_l2165_216583

theorem solve_exponential_equation :
  ∃ n : ℕ, 8^n * 8^n * 8^n = 64^3 ∧ n = 2 := by sorry

end solve_exponential_equation_l2165_216583


namespace sochi_price_decrease_in_euros_l2165_216580

/-- Represents the price decrease in Sochi apartments in euros -/
def sochi_price_decrease_euros : ℝ := 32.5

/-- The price decrease of Moscow apartments in rubles -/
def moscow_price_decrease_rubles : ℝ := 20

/-- The price decrease of Moscow apartments in euros -/
def moscow_price_decrease_euros : ℝ := 40

/-- The price decrease of Sochi apartments in rubles -/
def sochi_price_decrease_rubles : ℝ := 10

theorem sochi_price_decrease_in_euros :
  let initial_price_rubles : ℝ := 100  -- Arbitrary initial price
  let initial_price_euros : ℝ := 100   -- Arbitrary initial price
  let moscow_new_price_rubles : ℝ := initial_price_rubles * (1 - moscow_price_decrease_rubles / 100)
  let moscow_new_price_euros : ℝ := initial_price_euros * (1 - moscow_price_decrease_euros / 100)
  let sochi_new_price_rubles : ℝ := initial_price_rubles * (1 - sochi_price_decrease_rubles / 100)
  let exchange_rate : ℝ := moscow_new_price_rubles / moscow_new_price_euros
  let sochi_new_price_euros : ℝ := sochi_new_price_rubles / exchange_rate
  (initial_price_euros - sochi_new_price_euros) / initial_price_euros * 100 = sochi_price_decrease_euros :=
by sorry

end sochi_price_decrease_in_euros_l2165_216580


namespace unique_perfect_square_divisor_l2165_216541

theorem unique_perfect_square_divisor : ∃! (n : ℕ), n > 0 ∧ ∃ (k : ℕ), (n^3 - 1989) / n = k^2 := by
  sorry

end unique_perfect_square_divisor_l2165_216541


namespace work_hours_per_day_l2165_216533

theorem work_hours_per_day 
  (total_hours : ℝ) 
  (total_days : ℝ) 
  (h1 : total_hours = 8.0) 
  (h2 : total_days = 4.0) 
  (h3 : total_days > 0) : 
  total_hours / total_days = 2.0 := by
sorry

end work_hours_per_day_l2165_216533


namespace overlap_number_l2165_216581

theorem overlap_number (numbers : List ℝ) : 
  numbers.length = 9 ∧ 
  (numbers.take 5).sum / 5 = 7 ∧ 
  (numbers.drop 4).sum / 5 = 10 ∧ 
  numbers.sum / 9 = 74 / 9 → 
  ∃ x ∈ numbers, x = 11 ∧ x ∈ numbers.take 5 ∧ x ∈ numbers.drop 4 := by
sorry

end overlap_number_l2165_216581


namespace sqrt_two_sqrt_three_minus_three_equals_fourth_root_difference_l2165_216500

theorem sqrt_two_sqrt_three_minus_three_equals_fourth_root_difference : 
  Real.sqrt (2 * Real.sqrt 3 - 3) = (27/4)^(1/4) - (3/4)^(1/4) := by
  sorry

end sqrt_two_sqrt_three_minus_three_equals_fourth_root_difference_l2165_216500


namespace mario_garden_after_two_weeks_l2165_216504

/-- Calculates the number of flowers on a plant after a given number of weeks -/
def flowers_after_weeks (initial : ℕ) (growth_rate : ℕ) (weeks : ℕ) : ℕ :=
  initial + growth_rate * weeks

/-- Calculates the number of flowers on a plant that doubles each week -/
def flowers_doubling (initial : ℕ) (weeks : ℕ) : ℕ :=
  initial * (2^weeks)

/-- Represents Mario's garden and calculates the total number of blossoms -/
def mario_garden (weeks : ℕ) : ℕ :=
  let hibiscus1 := flowers_after_weeks 2 3 weeks
  let hibiscus2 := flowers_after_weeks 4 4 weeks
  let hibiscus3 := flowers_after_weeks 16 5 weeks
  let rose1 := flowers_after_weeks 3 2 weeks
  let rose2 := flowers_after_weeks 5 3 weeks
  let sunflower := flowers_doubling 6 weeks
  hibiscus1 + hibiscus2 + hibiscus3 + rose1 + rose2 + sunflower

theorem mario_garden_after_two_weeks :
  mario_garden 2 = 88 := by
  sorry

end mario_garden_after_two_weeks_l2165_216504


namespace negative_seven_minus_seven_l2165_216514

theorem negative_seven_minus_seven : (-7) - 7 = -14 := by
  sorry

end negative_seven_minus_seven_l2165_216514


namespace max_sum_of_four_digits_l2165_216521

def is_valid_digit (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 9

theorem max_sum_of_four_digits :
  ∀ A B C D : ℕ,
    is_valid_digit A → is_valid_digit B → is_valid_digit C → is_valid_digit D →
    A ≠ B → A ≠ C → A ≠ D → B ≠ C → B ≠ D → C ≠ D →
    (A + B) + (C + D) ≤ 30 :=
by sorry

end max_sum_of_four_digits_l2165_216521


namespace largest_inscribed_circle_circumference_l2165_216516

/-- The circumference of the largest circle inscribed in a square -/
theorem largest_inscribed_circle_circumference (s : ℝ) (h : s = 12) :
  2 * s * Real.pi = 24 * Real.pi := by sorry

end largest_inscribed_circle_circumference_l2165_216516


namespace max_t_and_solution_set_l2165_216529

open Real

noncomputable def f (x : ℝ) := 9 / (sin x)^2 + 4 / (cos x)^2

theorem max_t_and_solution_set :
  (∃ (t : ℝ), ∀ (x : ℝ), 0 < x ∧ x < π/2 → f x ≥ t) ∧
  (∀ (x : ℝ), 0 < x ∧ x < π/2 → f x ≥ 25) ∧
  (∀ (t : ℝ), (∀ (x : ℝ), 0 < x ∧ x < π/2 → f x ≥ t) → t ≤ 25) ∧
  ({x : ℝ | |x + 5| + |2*x - 1| ≤ 6} = {x : ℝ | 0 ≤ x ∧ x ≤ 2/3}) := by
  sorry

end max_t_and_solution_set_l2165_216529


namespace rain_period_end_time_l2165_216539

def start_time : ℕ := 8  -- 8 am
def rain_duration : ℕ := 4
def no_rain_duration : ℕ := 5

def total_duration : ℕ := rain_duration + no_rain_duration

def end_time : ℕ := start_time + total_duration

theorem rain_period_end_time :
  end_time = 17  -- 5 pm in 24-hour format
:= by sorry

end rain_period_end_time_l2165_216539


namespace water_added_to_tank_l2165_216506

theorem water_added_to_tank (tank_capacity : ℚ) 
  (h1 : tank_capacity = 56)
  (initial_fraction : ℚ) (final_fraction : ℚ)
  (h2 : initial_fraction = 3/4)
  (h3 : final_fraction = 7/8) :
  final_fraction * tank_capacity - initial_fraction * tank_capacity = 7 := by
  sorry

end water_added_to_tank_l2165_216506


namespace systematic_sampling_theorem_l2165_216572

/-- Represents a systematic sampling setup -/
structure SystematicSampling where
  total_students : ℕ
  num_groups : ℕ
  interval : ℕ
  first_number : ℕ

/-- Calculates the number drawn from a given group -/
def number_from_group (s : SystematicSampling) (group : ℕ) : ℕ :=
  s.first_number + s.interval * (group - 1)

theorem systematic_sampling_theorem (s : SystematicSampling) 
  (h1 : s.total_students = 160)
  (h2 : s.num_groups = 20)
  (h3 : s.interval = 8)
  (h4 : number_from_group s 16 = 123) :
  number_from_group s 2 = 11 := by
  sorry

#check systematic_sampling_theorem

end systematic_sampling_theorem_l2165_216572


namespace reflection_across_x_axis_l2165_216546

/-- A point in a 2D Cartesian coordinate system -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Reflection of a point across the x-axis -/
def reflect_x (p : Point2D) : Point2D :=
  { x := p.x, y := -p.y }

theorem reflection_across_x_axis :
  let P : Point2D := { x := -2, y := 3 }
  reflect_x P = { x := -2, y := -3 } := by sorry

end reflection_across_x_axis_l2165_216546


namespace probability_opposite_corner_is_one_third_l2165_216596

/-- Represents a cube with its properties -/
structure Cube where
  vertices : Fin 8
  faces : Fin 6

/-- Represents the ant's position on the cube -/
inductive Position
  | Corner : Fin 8 → Position

/-- Represents a single move of the ant -/
def Move : Type := Position → Position

/-- The probability of the ant ending at the diagonally opposite corner after two moves -/
def probability_opposite_corner (c : Cube) : ℚ :=
  1/3

/-- Theorem stating that the probability of ending at the diagonally opposite corner is 1/3 -/
theorem probability_opposite_corner_is_one_third (c : Cube) :
  probability_opposite_corner c = 1/3 := by sorry

end probability_opposite_corner_is_one_third_l2165_216596


namespace four_digit_perfect_square_with_equal_digit_pairs_l2165_216549

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def has_two_pairs_of_equal_digits (n : ℕ) : Prop :=
  ∃ a b : ℕ, a < 10 ∧ b < 10 ∧ n = 1000 * a + 100 * a + 10 * b + b

theorem four_digit_perfect_square_with_equal_digit_pairs :
  is_four_digit 7744 ∧ is_perfect_square 7744 ∧ has_two_pairs_of_equal_digits 7744 :=
by sorry

end four_digit_perfect_square_with_equal_digit_pairs_l2165_216549


namespace sequence_sum_l2165_216531

theorem sequence_sum : 
  let seq := [3, 15, 27, 53, 65, 17, 29, 41, 71, 83]
  List.sum seq = 404 := by
sorry

end sequence_sum_l2165_216531


namespace mary_initial_marbles_l2165_216565

/-- The number of yellow marbles Mary gave to Joan -/
def marbles_given : ℕ := 3

/-- The number of yellow marbles Mary has left -/
def marbles_left : ℕ := 6

/-- The initial number of yellow marbles Mary had -/
def initial_marbles : ℕ := marbles_given + marbles_left

theorem mary_initial_marbles : initial_marbles = 9 := by
  sorry

end mary_initial_marbles_l2165_216565


namespace factors_with_more_than_three_factors_l2165_216522

def number_to_factor := 2550

-- Function to count factors of a number
def count_factors (n : ℕ) : ℕ := sorry

-- Function to count numbers with more than 3 factors
def count_numbers_with_more_than_three_factors (n : ℕ) : ℕ := sorry

theorem factors_with_more_than_three_factors :
  count_numbers_with_more_than_three_factors number_to_factor = 9 := by sorry

end factors_with_more_than_three_factors_l2165_216522


namespace line_perp_plane_implies_perp_line_l2165_216589

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between two lines
variable (perpendicular_lines : Line → Line → Prop)

-- Define the "contained in" relation for a line in a plane
variable (contained_in : Line → Plane → Prop)

theorem line_perp_plane_implies_perp_line 
  (l m : Line) (α : Plane) :
  perpendicular_line_plane l α → contained_in m α → perpendicular_lines l m :=
sorry

end line_perp_plane_implies_perp_line_l2165_216589


namespace min_perimeter_triangle_l2165_216511

theorem min_perimeter_triangle (a b c : ℕ) (h_integer : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_cosA : Real.cos (Real.arccos ((b^2 + c^2 - a^2) / (2 * b * c))) = 11/16)
  (h_cosB : Real.cos (Real.arccos ((a^2 + c^2 - b^2) / (2 * a * c))) = 7/8)
  (h_cosC : Real.cos (Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))) = -1/4)
  (h_triangle : a + b > c ∧ b + c > a ∧ a + c > b) : 
  a + b + c ≥ 9 := by
  sorry

#check min_perimeter_triangle

end min_perimeter_triangle_l2165_216511


namespace calculate_expression_l2165_216505

theorem calculate_expression : 8 * (5 + 2/5) - 3 = 40.2 := by
  sorry

end calculate_expression_l2165_216505


namespace rainfall_problem_l2165_216560

/-- Rainfall problem --/
theorem rainfall_problem (monday_rain tuesday_rain wednesday_rain thursday_rain friday_rain : ℝ)
  (h_monday : monday_rain = 3)
  (h_tuesday : tuesday_rain = 2 * monday_rain)
  (h_wednesday : wednesday_rain = 0)
  (h_friday : friday_rain = monday_rain + tuesday_rain + wednesday_rain + thursday_rain)
  (h_average : (monday_rain + tuesday_rain + wednesday_rain + thursday_rain + friday_rain) / 7 = 4) :
  thursday_rain = 5 := by
sorry

end rainfall_problem_l2165_216560


namespace lcm_36_150_l2165_216577

theorem lcm_36_150 : Nat.lcm 36 150 = 900 := by
  sorry

end lcm_36_150_l2165_216577


namespace notebooks_left_l2165_216550

theorem notebooks_left (total : ℕ) (h1 : total = 28) : 
  total - (total / 4 + total * 3 / 7) = 9 := by
  sorry

end notebooks_left_l2165_216550


namespace rectangle_dimension_change_l2165_216566

theorem rectangle_dimension_change (L B : ℝ) (x : ℝ) (h_positive : L > 0 ∧ B > 0) :
  (1.20 * L) * (B * (1 - x / 100)) = 1.04 * (L * B) → x = 40 / 3 := by
sorry

end rectangle_dimension_change_l2165_216566


namespace girls_equal_barefoot_children_l2165_216552

/-- Given a lawn with boys and girls, some of whom are barefoot and some wearing shoes,
    prove that the number of girls equals the number of barefoot children
    when the number of barefoot boys equals the number of girls with shoes. -/
theorem girls_equal_barefoot_children
  (num_barefoot_boys : ℕ)
  (num_girls_with_shoes : ℕ)
  (num_barefoot_girls : ℕ)
  (h : num_barefoot_boys = num_girls_with_shoes) :
  num_girls_with_shoes + num_barefoot_girls = num_barefoot_boys + num_barefoot_girls :=
by sorry

end girls_equal_barefoot_children_l2165_216552


namespace car_speed_is_45_l2165_216559

/-- Represents the scenario of a car and motorcyclist journey --/
structure Journey where
  distance : ℝ  -- Distance from A to B in km
  moto_speed : ℝ  -- Motorcyclist's speed in km/h
  delay : ℝ  -- Delay before motorcyclist starts in hours
  car_speed : ℝ  -- Car's speed in km/h (to be proven)

/-- Theorem stating that under given conditions, the car's speed is 45 km/h --/
theorem car_speed_is_45 (j : Journey) 
  (h1 : j.distance = 82.5)
  (h2 : j.moto_speed = 60)
  (h3 : j.delay = 1/3)
  (h4 : ∃ t : ℝ, 
    t > 0 ∧ 
    j.car_speed * (t + j.delay) = j.moto_speed * t ∧ 
    (j.distance - j.moto_speed * t) / j.car_speed = t / 2) :
  j.car_speed = 45 := by
sorry


end car_speed_is_45_l2165_216559


namespace internally_tangent_circles_distance_l2165_216543

/-- Two circles are internally tangent if the distance between their centers
    is equal to the absolute difference of their radii -/
def internally_tangent (r₁ r₂ d : ℝ) : Prop :=
  d = |r₁ - r₂|

theorem internally_tangent_circles_distance
  (r₁ r₂ d : ℝ)
  (h₁ : r₁ = 3)
  (h₂ : r₂ = 6)
  (h₃ : internally_tangent r₁ r₂ d) :
  d = 3 :=
sorry

end internally_tangent_circles_distance_l2165_216543


namespace additive_inverse_problem_l2165_216556

theorem additive_inverse_problem (m : ℤ) : (m + 1) + (-2) = 0 → m = 1 := by
  sorry

end additive_inverse_problem_l2165_216556


namespace number_solution_l2165_216579

theorem number_solution : ∃ x : ℝ, 2 * x - 2.6 * 4 = 10 ∧ x = 10.2 := by
  sorry

end number_solution_l2165_216579


namespace david_pushups_count_l2165_216597

/-- The number of push-ups Zachary did -/
def zachary_pushups : ℕ := 59

/-- The difference between David's and Zachary's push-ups -/
def david_extra_pushups : ℕ := 19

/-- The number of push-ups David did -/
def david_pushups : ℕ := zachary_pushups + david_extra_pushups

theorem david_pushups_count : david_pushups = 78 := by
  sorry

end david_pushups_count_l2165_216597


namespace proportional_function_decreasing_l2165_216588

theorem proportional_function_decreasing (x₁ x₂ : ℝ) (h : x₁ < x₂) : -2 * x₁ > -2 * x₂ := by
  sorry

end proportional_function_decreasing_l2165_216588


namespace least_prime_factor_of_9_5_plus_9_4_l2165_216508

theorem least_prime_factor_of_9_5_plus_9_4 :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ (9^5 + 9^4) ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ (9^5 + 9^4) → p ≤ q :=
by sorry

end least_prime_factor_of_9_5_plus_9_4_l2165_216508


namespace tangent_line_property_l2165_216510

/-- Given a line tangent to ln x and e^x, prove that 1/x₁ - 2/(x₂-1) = 1 --/
theorem tangent_line_property (x₁ x₂ : ℝ) (h₁ : x₁ > 0) (h₂ : x₂ > 0) : 
  (∃ (m b : ℝ), 
    (∀ x, m * x + b = (1 / x₁) * x + Real.log x₁ - 1) ∧
    (∀ x, m * x + b = Real.exp x₂ * x - Real.exp x₂ * (x₂ - 1))) →
  1 / x₁ - 2 / (x₂ - 1) = 1 := by
  sorry


end tangent_line_property_l2165_216510


namespace unique_solution_l2165_216537

/-- Represents the ages of the grandchildren --/
structure GrandchildrenAges where
  martinka : ℕ
  tomasek : ℕ
  jaromir : ℕ
  kacka : ℕ
  ida : ℕ
  verka : ℕ

/-- The conditions given in the problem --/
def satisfiesConditions (ages : GrandchildrenAges) : Prop :=
  ages.martinka = ages.tomasek + 8 ∧
  ages.verka = ages.ida + 7 ∧
  ages.martinka = ages.jaromir + 1 ∧
  ages.kacka = ages.tomasek + 11 ∧
  ages.jaromir = ages.ida + 4 ∧
  ages.tomasek + ages.jaromir = 13

/-- The theorem stating that there is a unique solution satisfying all conditions --/
theorem unique_solution : ∃! ages : GrandchildrenAges, satisfiesConditions ages ∧
  ages.martinka = 11 ∧
  ages.tomasek = 3 ∧
  ages.jaromir = 10 ∧
  ages.kacka = 14 ∧
  ages.ida = 6 ∧
  ages.verka = 13 :=
sorry

end unique_solution_l2165_216537


namespace fano_plane_properties_l2165_216575

/-- A point in the Fano plane. -/
inductive Point
| P1 | P2 | P3 | P4 | P5 | P6 | P7

/-- A line in the Fano plane. -/
inductive Line
| L1 | L2 | L3 | L4 | L5 | L6 | L7

/-- The incidence relation between points and lines in the Fano plane. -/
def incidence : Point → Line → Prop
| Point.P1, Line.L1 => True
| Point.P1, Line.L2 => True
| Point.P1, Line.L3 => True
| Point.P2, Line.L1 => True
| Point.P2, Line.L4 => True
| Point.P2, Line.L5 => True
| Point.P3, Line.L1 => True
| Point.P3, Line.L6 => True
| Point.P3, Line.L7 => True
| Point.P4, Line.L2 => True
| Point.P4, Line.L4 => True
| Point.P4, Line.L6 => True
| Point.P5, Line.L2 => True
| Point.P5, Line.L5 => True
| Point.P5, Line.L7 => True
| Point.P6, Line.L3 => True
| Point.P6, Line.L4 => True
| Point.P6, Line.L7 => True
| Point.P7, Line.L3 => True
| Point.P7, Line.L5 => True
| Point.P7, Line.L6 => True
| _, _ => False

/-- The theorem stating that the Fano plane satisfies the required properties. -/
theorem fano_plane_properties :
  (∀ l : Line, ∃! (p1 p2 p3 : Point), p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3 ∧
    incidence p1 l ∧ incidence p2 l ∧ incidence p3 l ∧
    (∀ p : Point, incidence p l → p = p1 ∨ p = p2 ∨ p = p3)) ∧
  (∀ p : Point, ∃! (l1 l2 l3 : Line), l1 ≠ l2 ∧ l1 ≠ l3 ∧ l2 ≠ l3 ∧
    incidence p l1 ∧ incidence p l2 ∧ incidence p l3 ∧
    (∀ l : Line, incidence p l → l = l1 ∨ l = l2 ∨ l = l3)) :=
by sorry

end fano_plane_properties_l2165_216575


namespace equation_condition_l2165_216573

theorem equation_condition (a b c : ℤ) : 
  a * (a - b) + b * (b - c) + c * (c - a) = 2 → (a > b ∧ b = c) :=
by sorry

end equation_condition_l2165_216573


namespace sequence_properties_l2165_216599

def a : ℕ → ℕ
  | 0 => 1
  | n + 1 => (7 * a n + Nat.sqrt (45 * (a n)^2 - 36)) / 2

theorem sequence_properties :
  (∀ n : ℕ, a n > 0) ∧
  (∀ n : ℕ, ∃ k : ℕ, a n * a (n + 1) - 1 = k^2) := by
  sorry

end sequence_properties_l2165_216599


namespace monotone_decreasing_implies_m_eq_6_l2165_216502

/-- A function f is monotonically decreasing on an interval (a, b) if for all x, y in (a, b),
    x < y implies f(x) > f(y) -/
def MonotonicallyDecreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x > f y

/-- The function f(x) = x^3 - mx^2 + 2m^2 - 5 -/
def f (m : ℝ) (x : ℝ) : ℝ := x^3 - m*x^2 + 2*m^2 - 5

theorem monotone_decreasing_implies_m_eq_6 :
  ∀ m : ℝ, MonotonicallyDecreasing (f m) (-9) 0 → m = 6 := by
  sorry

end monotone_decreasing_implies_m_eq_6_l2165_216502


namespace square_area_ratio_l2165_216586

/-- Given three squares A, B, and C with side lengths x, 3x, and 2x respectively,
    prove that the ratio of the area of Square A to the combined area of Square B and Square C is 1/13 -/
theorem square_area_ratio (x : ℝ) (hx : x > 0) : 
  (x^2) / ((3*x)^2 + (2*x)^2) = 1 / 13 := by
sorry

end square_area_ratio_l2165_216586


namespace sum_of_prime_factor_exponents_l2165_216558

/-- The sum of exponents in the given expression of prime factors -/
def sum_of_exponents : ℕ :=
  9 + 5 + 7 + 4 + 6 + 3 + 5 + 2

/-- The theorem states that the sum of exponents in the given expression equals 41 -/
theorem sum_of_prime_factor_exponents : sum_of_exponents = 41 := by
  sorry

end sum_of_prime_factor_exponents_l2165_216558


namespace mean_equality_implies_z_l2165_216518

theorem mean_equality_implies_z (z : ℚ) : 
  (4 + 16 + 20) / 3 = (2 * 4 + z) / 2 → z = 56 / 3 := by
  sorry

end mean_equality_implies_z_l2165_216518


namespace route_distance_l2165_216590

theorem route_distance (time_Q : ℝ) (time_Y : ℝ) (speed_ratio : ℝ) :
  time_Q = 2 →
  time_Y = 4/3 →
  speed_ratio = 3/2 →
  ∃ (distance : ℝ) (speed_Q : ℝ),
    distance = speed_Q * time_Q ∧
    distance = (speed_ratio * speed_Q) * time_Y ∧
    distance = 3/2 := by
  sorry

end route_distance_l2165_216590


namespace rattlesnake_tail_difference_l2165_216595

/-- The number of tail segments in an Eastern rattlesnake -/
def eastern_segments : ℕ := 6

/-- The number of tail segments in a Western rattlesnake -/
def western_segments : ℕ := 8

/-- The percentage difference in tail size between Eastern and Western rattlesnakes,
    expressed as a percentage of the Western rattlesnake's tail size -/
def percentage_difference : ℚ :=
  (western_segments - eastern_segments : ℚ) / western_segments * 100

/-- Theorem stating that the percentage difference in tail size between
    Eastern and Western rattlesnakes is 25% -/
theorem rattlesnake_tail_difference :
  percentage_difference = 25 := by sorry

end rattlesnake_tail_difference_l2165_216595


namespace field_ratio_l2165_216512

/-- Proves that a rectangular field with perimeter 360 meters and width 75 meters has a length-to-width ratio of 7:5 -/
theorem field_ratio (perimeter width : ℝ) (h_perimeter : perimeter = 360) (h_width : width = 75) :
  let length := (perimeter - 2 * width) / 2
  (length / width) = 7 / 5 := by
  sorry

end field_ratio_l2165_216512


namespace former_apartment_size_l2165_216570

-- Define the given constants
def former_rent_per_sqft : ℝ := 2
def new_apartment_rent : ℝ := 2800
def yearly_savings : ℝ := 1200

-- Define the theorem
theorem former_apartment_size :
  ∃ (size : ℝ),
    size * former_rent_per_sqft = new_apartment_rent / 2 + yearly_savings / 12 ∧
    size = 750 :=
by sorry

end former_apartment_size_l2165_216570


namespace algebraic_expression_evaluation_l2165_216524

theorem algebraic_expression_evaluation :
  ∀ (a b : ℝ), 
  (2 * a * (-1)^3 - 3 * b * (-1) + 8 = 18) → 
  (9 * b - 6 * a + 2 = 32) := by
sorry

end algebraic_expression_evaluation_l2165_216524


namespace range_of_t_l2165_216593

-- Define a monotonically decreasing function
def MonoDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

-- State the theorem
theorem range_of_t (f : ℝ → ℝ) (h1 : MonoDecreasing f) :
  {t : ℝ | f (t^2) - f t < 0} = {t : ℝ | t < 0 ∨ t > 1} := by
  sorry

end range_of_t_l2165_216593


namespace inequality_region_is_triangle_l2165_216519

/-- The region described by a system of inequalities -/
def InequalityRegion (x y : ℝ) : Prop :=
  x + y - 1 ≤ 0 ∧ -x + y - 1 ≤ 0 ∧ y ≥ -1

/-- The triangle with vertices (0, 1), (2, -1), and (-2, -1) -/
def Triangle (x y : ℝ) : Prop :=
  (x = 0 ∧ y = 1) ∨ (x = 2 ∧ y = -1) ∨ (x = -2 ∧ y = -1) ∨
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ 
    ((x = 2*t - 2 ∧ y = -1) ∨
     (x = 2*t ∧ y = -t) ∨
     (x = -2*t ∧ y = t)))

theorem inequality_region_is_triangle :
  ∀ x y : ℝ, InequalityRegion x y ↔ Triangle x y :=
by sorry

end inequality_region_is_triangle_l2165_216519


namespace chewing_gums_count_l2165_216574

/-- Given the total number of treats, chocolate bars, and candies, prove the number of chewing gums. -/
theorem chewing_gums_count 
  (total_treats : ℕ) 
  (chocolate_bars : ℕ) 
  (candies : ℕ) 
  (h1 : total_treats = 155) 
  (h2 : chocolate_bars = 55) 
  (h3 : candies = 40) : 
  total_treats - (chocolate_bars + candies) = 60 := by
  sorry

#check chewing_gums_count

end chewing_gums_count_l2165_216574


namespace pentagonal_prism_lateral_angle_l2165_216571

/-- A pentagonal prism is a three-dimensional geometric shape with two congruent pentagonal bases 
    and five rectangular lateral faces. --/
structure PentagonalPrism where
  base : Pentagon
  height : ℝ
  height_pos : height > 0

/-- The angle φ is the angle between adjacent edges in a lateral face of the pentagonal prism. --/
def lateral_face_angle (prism : PentagonalPrism) : ℝ := sorry

/-- Theorem: In a pentagonal prism, the angle φ between adjacent edges in a lateral face must be 90°. --/
theorem pentagonal_prism_lateral_angle (prism : PentagonalPrism) : 
  lateral_face_angle prism = 90 := by sorry

end pentagonal_prism_lateral_angle_l2165_216571


namespace imaginary_part_of_3_minus_2i_l2165_216582

theorem imaginary_part_of_3_minus_2i :
  Complex.im (3 - 2 * Complex.I) = -2 := by
  sorry

end imaginary_part_of_3_minus_2i_l2165_216582


namespace no_solution_for_equal_ratios_l2165_216532

theorem no_solution_for_equal_ratios :
  ¬∃ (x : ℝ), (4 + x) / (5 + x) = (1 + x) / (2 + x) := by
sorry

end no_solution_for_equal_ratios_l2165_216532

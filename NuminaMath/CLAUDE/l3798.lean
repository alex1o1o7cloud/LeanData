import Mathlib

namespace min_difference_theorem_l3798_379831

noncomputable def f (x : ℝ) : ℝ := Real.exp x

noncomputable def g (x : ℝ) : ℝ := Real.log (x / 2) + 1 / 2

theorem min_difference_theorem (m : ℝ) (hm : m > 0) :
  ∃ (a b : ℝ), f a = m ∧ f b = m ∧
  ∀ (a' b' : ℝ), f a' = m → f b' = m → b - a ≤ b' - a' ∧ b - a = 2 + Real.log 2 :=
sorry

end min_difference_theorem_l3798_379831


namespace greatest_display_groups_l3798_379866

theorem greatest_display_groups (plates spoons glasses bowls : ℕ) 
  (h_plates : plates = 3219)
  (h_spoons : spoons = 5641)
  (h_glasses : glasses = 1509)
  (h_bowls : bowls = 2387) :
  Nat.gcd (Nat.gcd (Nat.gcd plates spoons) glasses) bowls = 1 := by
  sorry

end greatest_display_groups_l3798_379866


namespace field_division_proof_l3798_379803

theorem field_division_proof (total_area smaller_area larger_area certain_value : ℝ) : 
  total_area = 900 ∧ 
  smaller_area = 405 ∧ 
  larger_area = total_area - smaller_area ∧ 
  larger_area - smaller_area = (1 / 5) * certain_value →
  certain_value = 450 := by
sorry

end field_division_proof_l3798_379803


namespace divisibility_implication_l3798_379806

theorem divisibility_implication (k n : ℤ) :
  (13 ∣ (k + 4*n)) → (13 ∣ (10*k + n)) := by
  sorry

end divisibility_implication_l3798_379806


namespace hyperbola_equation_l3798_379848

/-- Given an ellipse and a hyperbola with the same foci, if one asymptote of the hyperbola
    is y = √2 x, then the equation of the hyperbola is 2y^2 - 4x^2 = 1 -/
theorem hyperbola_equation (x y : ℝ) :
  (∃ (a b : ℝ), (4 * x^2 + y^2 = 1) ∧ 
   (∃ (m : ℝ), 0 < m ∧ m < 3/4 ∧ 
     y^2 / m - x^2 / ((3/4) - m) = 1) ∧
   (∃ (k : ℝ), y = k * x ∧ k^2 = 2)) →
  2 * y^2 - 4 * x^2 = 1 := by
sorry

end hyperbola_equation_l3798_379848


namespace spider_trade_l3798_379858

/-- The number of spiders Pugsley and Wednesday trade --/
theorem spider_trade (P W x : ℕ) : 
  P = 4 →  -- Pugsley's initial number of spiders
  W + x = 9 * (P - x) →  -- First scenario equation
  P + 6 = W - 6 →  -- Second scenario equation
  x = 2  -- Number of spiders Pugsley gives to Wednesday
:= by sorry

end spider_trade_l3798_379858


namespace winter_clothing_mittens_per_box_l3798_379871

theorem winter_clothing_mittens_per_box 
  (num_boxes : ℕ) 
  (scarves_per_box : ℕ) 
  (total_pieces : ℕ) 
  (h1 : num_boxes = 3)
  (h2 : scarves_per_box = 3)
  (h3 : total_pieces = 21) :
  (total_pieces - num_boxes * scarves_per_box) / num_boxes = 4 := by
sorry

end winter_clothing_mittens_per_box_l3798_379871


namespace x_less_than_one_necessary_not_sufficient_for_x_squared_less_than_one_l3798_379838

theorem x_less_than_one_necessary_not_sufficient_for_x_squared_less_than_one :
  ∀ x : ℝ, (x^2 < 1 → x < 1) ∧ ¬(x < 1 → x^2 < 1) := by sorry

end x_less_than_one_necessary_not_sufficient_for_x_squared_less_than_one_l3798_379838


namespace initial_red_marbles_l3798_379827

theorem initial_red_marbles (r g : ℕ) : 
  r * 3 = g * 5 → 
  (r - 20) * 5 = (g + 40) * 1 → 
  r = 317 := by
sorry

end initial_red_marbles_l3798_379827


namespace constant_product_of_distances_l3798_379816

/-- Hyperbola type representing x^2 - y^2/4 = 1 -/
structure Hyperbola where
  x : ℝ
  y : ℝ
  eq : x^2 - y^2/4 = 1

/-- Line type representing a line passing through a point on the hyperbola -/
structure Line (h : Hyperbola) where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept
  passes_through : m * h.x + b = h.y

/-- Intersection point of a line with an asymptote -/
structure AsymptoteIntersection (h : Hyperbola) (l : Line h) where
  x : ℝ
  y : ℝ
  on_asymptote : y = 2*x ∨ y = -2*x
  on_line : y = l.m * x + l.b

/-- Theorem: Product of distances from origin to asymptote intersections is constant -/
theorem constant_product_of_distances (h : Hyperbola) (l : Line h) 
  (a b : AsymptoteIntersection h l) 
  (midpoint : h.x = (a.x + b.x)/2 ∧ h.y = (a.y + b.y)/2) :
  (a.x^2 + a.y^2) * (b.x^2 + b.y^2) = 25 := by sorry

end constant_product_of_distances_l3798_379816


namespace blue_pill_cost_correct_l3798_379863

/-- The cost of a blue pill in dollars -/
def blue_pill_cost : ℝ := 23.50

/-- The cost of a red pill in dollars -/
def red_pill_cost : ℝ := blue_pill_cost - 2

/-- The number of days of medication -/
def days : ℕ := 21

/-- The total cost of medication for the entire period -/
def total_cost : ℝ := 945

theorem blue_pill_cost_correct :
  blue_pill_cost * days + red_pill_cost * days = total_cost :=
by sorry

end blue_pill_cost_correct_l3798_379863


namespace root_value_theorem_l3798_379845

theorem root_value_theorem (m : ℝ) : 
  (2 * m^2 - 3 * m - 1 = 0) → (3 * m * (2 * m - 3) - 1 = 2) := by
  sorry

end root_value_theorem_l3798_379845


namespace valid_intersection_numbers_l3798_379829

/-- A circle with arcs that intersect each other. -/
structure CircleWithArcs where
  num_arcs : ℕ
  intersections_per_arc : ℕ

/-- Predicate to check if a number is not a multiple of 8. -/
def not_multiple_of_eight (n : ℕ) : Prop :=
  n % 8 ≠ 0

/-- Theorem stating the conditions for valid intersection numbers in a circle with 100 arcs. -/
theorem valid_intersection_numbers (circle : CircleWithArcs) :
    circle.num_arcs = 100 →
    1 ≤ circle.intersections_per_arc ∧
    circle.intersections_per_arc ≤ 99 ∧
    not_multiple_of_eight (circle.intersections_per_arc + 1) :=
by sorry

end valid_intersection_numbers_l3798_379829


namespace parabola_intersection_point_l3798_379825

theorem parabola_intersection_point (n m : ℕ) (x₀ y₀ : ℝ) 
  (hn : n ≥ 2) 
  (hm : m > 0) 
  (h1 : y₀^2 = n * x₀ - 1) 
  (h2 : y₀ = x₀) : 
  ∃ k : ℕ, k ≥ 2 ∧ (x₀^m)^2 = k * (x₀^m) - 1 := by
sorry

end parabola_intersection_point_l3798_379825


namespace officer_election_ways_l3798_379800

def club_size : ℕ := 12
def num_officers : ℕ := 5

theorem officer_election_ways :
  (club_size * (club_size - 1) * (club_size - 2) * (club_size - 3) * (club_size - 4) : ℕ) = 95040 := by
  sorry

end officer_election_ways_l3798_379800


namespace f_strictly_increasing_l3798_379814

-- Define the function f
def f (x : ℝ) : ℝ := x^2 * (2 - x)

-- Theorem statement
theorem f_strictly_increasing : 
  ∀ x y, 0 < x ∧ x < y ∧ y < 4/3 → f x < f y := by
  sorry

end f_strictly_increasing_l3798_379814


namespace rosy_fish_count_l3798_379801

/-- The number of fish Lilly has -/
def lillys_fish : ℕ := 10

/-- The total number of fish Lilly and Rosy have together -/
def total_fish : ℕ := 19

/-- The number of fish Rosy has -/
def rosys_fish : ℕ := total_fish - lillys_fish

theorem rosy_fish_count : rosys_fish = 9 := by
  sorry

end rosy_fish_count_l3798_379801


namespace first_digit_base_7_of_528_l3798_379873

/-- The first digit of the base 7 representation of a natural number -/
def first_digit_base_7 (n : ℕ) : ℕ :=
  if n = 0 then 0
  else
    let k := (Nat.log n 7).succ
    (n / 7^(k-1)) % 7

/-- Theorem: The first digit of the base 7 representation of 528 is 1 -/
theorem first_digit_base_7_of_528 :
  first_digit_base_7 528 = 1 := by sorry

end first_digit_base_7_of_528_l3798_379873


namespace lines_perpendicular_when_a_is_neg_six_l3798_379820

/-- Given two lines l₁ and l₂ defined by their equations, prove that they are perpendicular when a = -6 -/
theorem lines_perpendicular_when_a_is_neg_six (a : ℝ) :
  a = -6 →
  let l₁ := {(x, y) : ℝ × ℝ | a * x + (1 - a) * y - 3 = 0}
  let l₂ := {(x, y) : ℝ × ℝ | (a - 1) * x + 2 * (a + 3) * y - 2 = 0}
  let m₁ := a / (1 - a)
  let m₂ := (a - 1) / (2 * (a + 3))
  m₁ * m₂ = -1 := by
  sorry

end lines_perpendicular_when_a_is_neg_six_l3798_379820


namespace unique_valid_stamp_set_l3798_379893

/-- Given stamps of denominations 7, n, and n+1 cents, 
    110 cents is the greatest postage that cannot be formed -/
def is_valid_stamp_set (n : ℕ) : Prop :=
  ∀ m : ℕ, m > 110 → ∃ (a b c : ℕ), m = 7 * a + n * b + (n + 1) * c ∧
  ¬∃ (a b c : ℕ), 110 = 7 * a + n * b + (n + 1) * c

theorem unique_valid_stamp_set :
  ∃! n : ℕ, n > 0 ∧ is_valid_stamp_set n :=
by sorry

end unique_valid_stamp_set_l3798_379893


namespace sum_of_seventh_powers_l3798_379889

theorem sum_of_seventh_powers (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (hsum : x + y + z = 0) (hprod : x*y + x*z + y*z ≠ 0) :
  (x^7 + y^7 + z^7) / (x*y*z * (x*y + x*z + y*z)) = -7 := by
  sorry

end sum_of_seventh_powers_l3798_379889


namespace election_votes_count_l3798_379864

theorem election_votes_count :
  ∀ (total_votes : ℕ) (losing_candidate_votes winning_candidate_votes : ℕ),
    losing_candidate_votes = (35 * total_votes) / 100 →
    winning_candidate_votes = losing_candidate_votes + 2370 →
    losing_candidate_votes + winning_candidate_votes = total_votes →
    total_votes = 7900 :=
by
  sorry

end election_votes_count_l3798_379864


namespace divisors_of_18m_squared_l3798_379823

/-- A function that returns the number of positive divisors of a natural number -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- A function that checks if a natural number is even -/
def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

theorem divisors_of_18m_squared (m : ℕ) 
  (h1 : is_even m) 
  (h2 : num_divisors m = 9) : 
  num_divisors (18 * m^2) = 54 := by sorry

end divisors_of_18m_squared_l3798_379823


namespace parabola_x_intercepts_l3798_379870

/-- The number of x-intercepts for the parabola x = -3y^2 + 2y + 3 -/
theorem parabola_x_intercepts : ∃! x : ℝ, ∃ y : ℝ, x = -3 * y^2 + 2 * y + 3 ∧ y = 0 := by
  sorry

end parabola_x_intercepts_l3798_379870


namespace inequality_solution_set_l3798_379811

-- Define the function f
def f (b c x : ℝ) : ℝ := x^2 - b*x + c

-- Define the theorem
theorem inequality_solution_set 
  (b c : ℝ) 
  (hb : b > 0) 
  (hc : c > 0) 
  (x₁ x₂ : ℝ) 
  (h_zeros : f b c x₁ = 0 ∧ f b c x₂ = 0) 
  (h_progression : (∃ r : ℝ, x₁ = -1 * r ∧ x₂ = -1 / r) ∨ 
                   (∃ d : ℝ, x₁ = -1 - d ∧ x₂ = -1 + d)) :
  {x : ℝ | (x - b) / (x - c) ≤ 0} = Set.Ioo 1 (5/2) ∪ {5/2} := by
  sorry

end inequality_solution_set_l3798_379811


namespace sqrt_sum_equals_twelve_l3798_379832

theorem sqrt_sum_equals_twelve : 
  Real.sqrt ((5 - 3 * Real.sqrt 2)^2) + Real.sqrt ((5 + 3 * Real.sqrt 2)^2) + 2 = 12 := by
  sorry

end sqrt_sum_equals_twelve_l3798_379832


namespace electric_car_charging_cost_l3798_379835

/-- Calculates the total cost of charging an electric car -/
def total_charging_cost (charges_per_week : ℕ) (num_weeks : ℕ) (cost_per_charge : ℚ) : ℚ :=
  (charges_per_week * num_weeks : ℕ) * cost_per_charge

/-- Proves that the total cost of charging an electric car under given conditions is $121.68 -/
theorem electric_car_charging_cost :
  total_charging_cost 3 52 (78/100) = 12168/100 := by
  sorry

end electric_car_charging_cost_l3798_379835


namespace intersection_A_B_l3798_379826

def A : Set ℝ := {x | x ≤ 2*x + 1 ∧ 2*x + 1 ≤ 5}
def B : Set ℝ := {x | 0 < x ∧ x ≤ 3}

theorem intersection_A_B : A ∩ B = {x : ℝ | 0 < x ∧ x ≤ 2} := by sorry

end intersection_A_B_l3798_379826


namespace function_property_l3798_379847

/-- Given a function f(x) = (x^2 + ax + b)(e^x - e), where a and b are real numbers,
    and f(x) ≥ 0 for all x > 0, then a ≥ -1. -/
theorem function_property (a b : ℝ) :
  (∀ x > 0, (x^2 + a*x + b) * (Real.exp x - Real.exp 1) ≥ 0) →
  a ≥ -1 :=
by sorry

end function_property_l3798_379847


namespace base_8_2453_equals_1323_l3798_379833

def base_8_to_10 (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (8 ^ i)) 0

theorem base_8_2453_equals_1323 :
  base_8_to_10 [3, 5, 4, 2] = 1323 := by
  sorry

end base_8_2453_equals_1323_l3798_379833


namespace probability_at_least_one_woman_l3798_379884

/-- The probability of selecting at least one woman when choosing three people
    at random from a group of eight men and four women -/
theorem probability_at_least_one_woman (men : ℕ) (women : ℕ) : 
  men = 8 → women = 4 → 
  (1 - (Nat.choose men 3 : ℚ) / (Nat.choose (men + women) 3 : ℚ)) = 41 / 55 := by
  sorry

end probability_at_least_one_woman_l3798_379884


namespace sum_2012_terms_equals_negative_2012_l3798_379887

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

def sum_arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem sum_2012_terms_equals_negative_2012 :
  let a₁ : ℤ := -2012
  let d : ℤ := 2
  let n : ℕ := 2012
  sum_arithmetic_sequence a₁ d n = -2012 := by sorry

end sum_2012_terms_equals_negative_2012_l3798_379887


namespace tangent_line_equation_f_positive_when_a_is_one_minimum_value_when_a_is_e_squared_l3798_379812

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := 1 - (a * x^2) / Real.exp x

-- State the theorems to be proved
theorem tangent_line_equation (a : ℝ) :
  (∃ k, ∀ x, k * x + (f a 1 - k) = f a x + (deriv (f a)) 1 * (x - 1)) →
  ∃ k, k = 1 ∧ ∀ x, x + 1 = f a x + (deriv (f a)) 1 * (x - 1) :=
sorry

theorem f_positive_when_a_is_one :
  ∀ x > 0, f 1 x > 0 :=
sorry

theorem minimum_value_when_a_is_e_squared :
  (∃ x, f (Real.exp 2) x = -3) ∧ (∀ x, f (Real.exp 2) x ≥ -3) :=
sorry

end tangent_line_equation_f_positive_when_a_is_one_minimum_value_when_a_is_e_squared_l3798_379812


namespace adjacent_triangles_toothpicks_l3798_379856

/-- Calculates the number of toothpicks needed for an equilateral triangle -/
def toothpicks_for_triangle (base : ℕ) : ℕ :=
  3 * (base * (base + 1) / 2) / 2

/-- The number of toothpicks needed for two adjacent equilateral triangles -/
def total_toothpicks (large_base small_base : ℕ) : ℕ :=
  toothpicks_for_triangle large_base + toothpicks_for_triangle small_base - small_base

theorem adjacent_triangles_toothpicks :
  total_toothpicks 100 50 = 9462 :=
sorry

end adjacent_triangles_toothpicks_l3798_379856


namespace even_function_implies_a_zero_l3798_379817

/-- A function f is even if f(x) = f(-x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

/-- The function f(x) = x^2 - |x + a| -/
def f (a : ℝ) : ℝ → ℝ := λ x ↦ x^2 - |x + a|

/-- If f(x) = x^2 - |x + a| is an even function, then a = 0 -/
theorem even_function_implies_a_zero (a : ℝ) : IsEven (f a) → a = 0 := by
  sorry

end even_function_implies_a_zero_l3798_379817


namespace smallest_four_digit_numbers_l3798_379828

def is_valid (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n < 10000 ∧
  n % 2 = 1 ∧ n % 3 = 1 ∧ n % 4 = 1 ∧ n % 5 = 1 ∧ n % 6 = 1

theorem smallest_four_digit_numbers :
  let valid_numbers := [1021, 1081, 1141, 1201]
  (∀ n ∈ valid_numbers, is_valid n) ∧
  (∀ m, is_valid m → m ≥ 1021) ∧
  (∀ n ∈ valid_numbers, ∀ m, is_valid m ∧ m < n → m ∈ valid_numbers) :=
by sorry

end smallest_four_digit_numbers_l3798_379828


namespace intersection_of_A_and_B_l3798_379834

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | 2 < x ∧ x < 4}
def B : Set ℝ := {x : ℝ | (x - 1) * (x - 3) < 0}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 2 < x ∧ x < 3} := by sorry

end intersection_of_A_and_B_l3798_379834


namespace point_d_coordinates_l3798_379891

/-- Given two points P and Q in the plane, and a point D on the line segment PQ such that
    PD = 2DQ, prove that D has specific coordinates. -/
theorem point_d_coordinates (P Q D : ℝ × ℝ) : 
  P = (-3, -2) →
  Q = (5, 10) →
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ D = (1 - t) • P + t • Q) →
  (P.1 - D.1)^2 + (P.2 - D.2)^2 = 4 * ((Q.1 - D.1)^2 + (Q.2 - D.2)^2) →
  D = (3, 7) := by
sorry


end point_d_coordinates_l3798_379891


namespace circle_problem_l3798_379822

-- Define the equation of the general circle
def general_circle (k : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*k*x - (2*k+6)*y - 2*k - 31 = 0

-- Define the specific circle E
def circle_E (x y : ℝ) : Prop :=
  (x+2)^2 + (y-1)^2 = 32

-- Theorem statement
theorem circle_problem :
  (∀ k : ℝ, general_circle k (-6) 5 ∧ general_circle k 2 (-3)) ∧
  (circle_E (-6) 5 ∧ circle_E 2 (-3)) ∧
  (∀ P : ℝ × ℝ, ¬circle_E P.1 P.2 →
    ∃ A B : ℝ × ℝ,
      circle_E A.1 A.2 ∧
      circle_E B.1 B.2 ∧
      (∀ X : ℝ × ℝ, circle_E X.1 X.2 →
        (P.1 - A.1) * (X.1 - A.1) + (P.2 - A.2) * (X.2 - A.2) = 0 ∧
        (P.1 - B.1) * (X.1 - B.1) + (P.2 - B.2) * (X.2 - B.2) = 0) ∧
      ((P.1 - A.1) * (P.1 - B.1) + (P.2 - A.2) * (P.2 - B.2) ≥ 64 * Real.sqrt 2 - 96)) :=
sorry

end circle_problem_l3798_379822


namespace total_canoes_by_april_l3798_379876

def canoes_built (month : Nat) : Nat :=
  match month with
  | 0 => 4  -- February
  | n + 1 => 3 * canoes_built n

theorem total_canoes_by_april : 
  (canoes_built 0) + (canoes_built 1) + (canoes_built 2) = 52 := by
  sorry

end total_canoes_by_april_l3798_379876


namespace team_b_mean_tasks_l3798_379837

/-- Represents the office with two teams -/
structure Office :=
  (total_members : ℕ)
  (team_a_members : ℕ)
  (team_b_members : ℕ)
  (team_a_mean_tasks : ℝ)
  (team_b_mean_tasks : ℝ)

/-- The conditions of the office as described in the problem -/
def office_conditions (o : Office) : Prop :=
  o.total_members = 260 ∧
  o.team_a_members = (13 * o.team_b_members) / 10 ∧
  o.team_a_mean_tasks = 80 ∧
  o.team_b_mean_tasks = (6 * o.team_a_mean_tasks) / 5

/-- The theorem stating that under the given conditions, Team B's mean tasks is 96 -/
theorem team_b_mean_tasks (o : Office) (h : office_conditions o) : 
  o.team_b_mean_tasks = 96 := by
  sorry


end team_b_mean_tasks_l3798_379837


namespace parallel_lines_m_values_l3798_379875

/-- Given two lines l₁ and l₂ with equations 3x + my - 1 = 0 and (m+2)x - (m-2)y + 2 = 0 respectively,
    if l₁ is parallel to l₂, then m = -6 or m = 1. -/
theorem parallel_lines_m_values (m : ℝ) :
  let l₁ := {(x, y) : ℝ × ℝ | 3 * x + m * y - 1 = 0}
  let l₂ := {(x, y) : ℝ × ℝ | (m + 2) * x - (m - 2) * y + 2 = 0}
  (∀ (a b c d : ℝ), a * (m + 2) = 3 * c ∧ b * (m - 2) = -m * d → (a, b) = (c, d)) →
  m = -6 ∨ m = 1 := by
  sorry


end parallel_lines_m_values_l3798_379875


namespace sqrt_inequality_solution_set_l3798_379865

theorem sqrt_inequality_solution_set (x : ℝ) :
  (x^3 - 8) / x ≥ 0 →
  (Real.sqrt ((x^3 - 8) / x) > x - 2 ↔ x ∈ Set.Ioi 2 ∪ Set.Iio 0) :=
by sorry

end sqrt_inequality_solution_set_l3798_379865


namespace hyperbola_eccentricity_l3798_379896

/-- The eccentricity of a hyperbola with equation x²/4 - y² = 1 is √5/2 -/
theorem hyperbola_eccentricity : 
  let hyperbola := {(x, y) : ℝ × ℝ | x^2 / 4 - y^2 = 1}
  let a : ℝ := 2  -- semi-major axis
  let b : ℝ := 1  -- semi-minor axis
  let c : ℝ := Real.sqrt (a^2 + b^2)  -- focal distance
  let e : ℝ := c / a  -- eccentricity
  e = Real.sqrt 5 / 2 := by sorry

end hyperbola_eccentricity_l3798_379896


namespace system_solution_l3798_379897

/-- Given a system of two linear equations in two variables,
    prove that the solution satisfies both equations. -/
theorem system_solution (x y : ℚ) : 
  x = 14 ∧ y = 29/5 →
  -x + 5*y = 15 ∧ 4*x - 10*y = -2 := by sorry

end system_solution_l3798_379897


namespace jim_out_of_pocket_l3798_379861

def out_of_pocket (first_ring_cost second_ring_cost first_ring_sale_price : ℕ) : ℕ :=
  first_ring_cost + second_ring_cost - first_ring_sale_price

theorem jim_out_of_pocket :
  let first_ring_cost : ℕ := 10000
  let second_ring_cost : ℕ := 2 * first_ring_cost
  let first_ring_sale_price : ℕ := first_ring_cost / 2
  out_of_pocket first_ring_cost second_ring_cost first_ring_sale_price = 25000 := by
  sorry

end jim_out_of_pocket_l3798_379861


namespace light_reflection_l3798_379894

-- Define the incident light ray
def incident_ray (x y : ℝ) : Prop := x - 2*y + 3 = 0

-- Define the reflection line
def reflection_line (x y : ℝ) : Prop := y = x

-- Define the reflected light ray
def reflected_ray (x y : ℝ) : Prop := 2*x - y - 3 = 0

-- Theorem statement
theorem light_reflection 
  (x y : ℝ) 
  (h_incident : incident_ray x y) 
  (h_reflection : reflection_line x y) : 
  reflected_ray x y :=
sorry

end light_reflection_l3798_379894


namespace cube_of_square_of_third_smallest_prime_l3798_379855

def third_smallest_prime : ℕ := sorry

theorem cube_of_square_of_third_smallest_prime :
  (third_smallest_prime ^ 2) ^ 3 = 15625 := by sorry

end cube_of_square_of_third_smallest_prime_l3798_379855


namespace iron_percentage_in_alloy_l3798_379877

/-- The percentage of alloy in the ore -/
def alloy_percentage : ℝ := 0.25

/-- The total amount of ore in kg -/
def total_ore : ℝ := 266.6666666666667

/-- The amount of pure iron obtained in kg -/
def pure_iron : ℝ := 60

/-- The percentage of iron in the alloy -/
def iron_percentage : ℝ := 0.9

theorem iron_percentage_in_alloy :
  alloy_percentage * total_ore * iron_percentage = pure_iron :=
sorry

end iron_percentage_in_alloy_l3798_379877


namespace sum_of_squares_of_divisors_1800_l3798_379840

def sumOfSquaresOfDivisors (n : ℕ) : ℕ := sorry

theorem sum_of_squares_of_divisors_1800 :
  sumOfSquaresOfDivisors 1800 = 5035485 :=
by
  sorry

end sum_of_squares_of_divisors_1800_l3798_379840


namespace rational_numbers_equivalence_l3798_379844

-- Define the set of integers
def Integers : Set ℚ := {q : ℚ | ∃ (n : ℤ), q = n}

-- Define the set of fractions
def Fractions : Set ℚ := {q : ℚ | ∃ (a b : ℤ), b ≠ 0 ∧ q = a / b}

-- Theorem statement
theorem rational_numbers_equivalence :
  Set.univ = Integers ∪ Fractions :=
sorry

end rational_numbers_equivalence_l3798_379844


namespace fruit_combination_count_l3798_379860

/-- The number of combinations when choosing k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of fruit types available -/
def fruit_types : ℕ := 4

/-- The number of fruits to be chosen -/
def fruits_to_choose : ℕ := 3

/-- Theorem: The number of combinations when choosing 3 fruits from 4 types is 4 -/
theorem fruit_combination_count : choose fruit_types fruits_to_choose = 4 := by
  sorry

end fruit_combination_count_l3798_379860


namespace prime_factor_congruence_l3798_379898

theorem prime_factor_congruence (p : ℕ) (h_prime : Prime p) :
  ∃ q : ℕ, Prime q ∧ q ∣ (p^p - 1) ∧ q ≡ 1 [ZMOD p] := by
  sorry

end prime_factor_congruence_l3798_379898


namespace quadratic_m_value_l3798_379882

/-- A quadratic function with specific properties -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  min_value : ℝ
  min_x : ℝ
  point_zero : ℝ
  point_five : ℝ

/-- The properties of the given quadratic function -/
def given_quadratic : QuadraticFunction where
  a := 0
  b := 0
  c := 0
  min_value := -10
  min_x := -1
  point_zero := 8
  point_five := 0  -- This is the m we want to prove

/-- The theorem stating the value of m -/
theorem quadratic_m_value (f : QuadraticFunction) (h1 : f.min_value = -10) 
    (h2 : f.min_x = -1) (h3 : f.point_zero = 8) : f.point_five = 638 := by
  sorry

end quadratic_m_value_l3798_379882


namespace jumping_contest_l3798_379899

/-- The jumping contest problem -/
theorem jumping_contest (grasshopper_jump mouse_jump : ℕ) 
  (h1 : grasshopper_jump = 25)
  (h2 : mouse_jump = 31) : 
  (grasshopper_jump + 32) - mouse_jump = 26 := by
  sorry


end jumping_contest_l3798_379899


namespace arithmetic_sequence_condition_l3798_379878

/-- For an arithmetic sequence with first term a₁ and common difference d,
    the condition 2a₁ + 11d > 0 is sufficient but not necessary for 2a₁ + 11d ≥ 0 -/
theorem arithmetic_sequence_condition (a₁ d : ℝ) :
  (∃ x y : ℝ, (x > y) ∧ (x ≥ 0) ∧ (y < 0)) ∧
  (2 * a₁ + 11 * d > 0 → 2 * a₁ + 11 * d ≥ 0) ∧
  ¬(2 * a₁ + 11 * d ≥ 0 → 2 * a₁ + 11 * d > 0) :=
sorry

end arithmetic_sequence_condition_l3798_379878


namespace sphere_surface_area_from_rectangular_solid_l3798_379819

/-- Given a rectangular solid with adjacent face areas of 2, 3, and 6, 
    and all vertices lying on a sphere, the surface area of this sphere is 14π. -/
theorem sphere_surface_area_from_rectangular_solid (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  a * b = 6 →
  b * c = 2 →
  a * c = 3 →
  (∃ (r : ℝ), r > 0 ∧ a^2 + b^2 + c^2 = (2*r)^2) →
  4 * π * ((a^2 + b^2 + c^2) / 4) = 14 * π :=
by sorry

end sphere_surface_area_from_rectangular_solid_l3798_379819


namespace sin_1320_degrees_l3798_379805

theorem sin_1320_degrees : Real.sin (1320 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_1320_degrees_l3798_379805


namespace system_solution_l3798_379813

theorem system_solution (m : ℝ) : 
  (∃ x y : ℝ, x + y = 3*m ∧ x - y = 5*m ∧ 2*x + 3*y = 10) → m = 2 := by
  sorry

end system_solution_l3798_379813


namespace BAB_better_than_ABA_l3798_379886

/-- Represents a wrestler's opponent -/
inductive Opponent
| A  -- Andrei
| B  -- Boris

/-- Represents a schedule of three matches -/
def Schedule := List Opponent

/-- The probability of Vladimir losing to a given opponent -/
def losing_probability (o : Opponent) : ℝ :=
  match o with
  | Opponent.A => 0.4
  | Opponent.B => 0.3

/-- The probability of Vladimir winning against a given opponent -/
def winning_probability (o : Opponent) : ℝ :=
  1 - losing_probability o

/-- Calculates the probability of Vladimir qualifying given a schedule -/
def qualifying_probability (s : Schedule) : ℝ :=
  match s with
  | [o1, o2, o3] =>
    winning_probability o1 * losing_probability o2 * winning_probability o3 +
    winning_probability o1 * winning_probability o2 +
    losing_probability o1 * winning_probability o2 * winning_probability o3
  | _ => 0  -- Invalid schedule

def ABA : Schedule := [Opponent.A, Opponent.B, Opponent.A]
def BAB : Schedule := [Opponent.B, Opponent.A, Opponent.B]

theorem BAB_better_than_ABA :
  qualifying_probability BAB > qualifying_probability ABA :=
sorry

end BAB_better_than_ABA_l3798_379886


namespace middle_part_of_proportional_division_l3798_379804

theorem middle_part_of_proportional_division (total : ℚ) (a b c : ℚ) 
  (h_total : total = 120)
  (h_prop : ∃ (x : ℚ), a = x ∧ b = (1/2) * x ∧ c = (1/4) * x)
  (h_sum : a + b + c = total) :
  b = 34 + 2/7 := by
  sorry

end middle_part_of_proportional_division_l3798_379804


namespace dwayne_class_a_count_l3798_379869

/-- Proves that given the conditions from Mrs. Carter's and Mr. Dwayne's classes,
    the number of students who received an 'A' in Mr. Dwayne's class is 12. -/
theorem dwayne_class_a_count :
  let carter_total : ℕ := 20
  let carter_a_count : ℕ := 8
  let dwayne_total : ℕ := 30
  let ratio : ℚ := carter_a_count / carter_total
  ∃ (dwayne_a_count : ℕ), 
    (dwayne_a_count : ℚ) / dwayne_total = ratio ∧ 
    dwayne_a_count = 12 :=
by sorry

end dwayne_class_a_count_l3798_379869


namespace bobs_speed_l3798_379815

theorem bobs_speed (initial_time : ℝ) (construction_time : ℝ) (construction_speed : ℝ) (total_distance : ℝ) :
  initial_time = 1.5 →
  construction_time = 2 →
  construction_speed = 45 →
  total_distance = 180 →
  ∃ (initial_speed : ℝ),
    initial_speed * initial_time + construction_speed * construction_time = total_distance ∧
    initial_speed = 60 :=
by sorry

end bobs_speed_l3798_379815


namespace prism_dimensions_l3798_379883

/-- Represents the dimensions of a rectangular prism -/
structure PrismDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Checks if the given dimensions satisfy the conditions of the problem -/
def satisfiesConditions (d : PrismDimensions) : Prop :=
  -- One edge is five times longer than another
  (d.length = 5 * d.width ∨ d.width = 5 * d.length ∨ d.length = 5 * d.height ∨
   d.height = 5 * d.length ∨ d.width = 5 * d.height ∨ d.height = 5 * d.width) ∧
  -- Increasing height by 2 increases volume by 90
  d.length * d.width * 2 = 90 ∧
  -- Changing height to half of (height + 2) makes volume three-fifths of original
  (d.height + 2) / 2 = 3 / 5 * d.height

/-- The theorem stating the only possible dimensions for the rectangular prism -/
theorem prism_dimensions :
  ∀ d : PrismDimensions,
    satisfiesConditions d →
    (d = ⟨0.9, 50, 10⟩ ∨ d = ⟨50, 0.9, 10⟩ ∨
     d = ⟨2, 22.5, 10⟩ ∨ d = ⟨22.5, 2, 10⟩ ∨
     d = ⟨3, 15, 10⟩ ∨ d = ⟨15, 3, 10⟩) :=
by
  sorry

end prism_dimensions_l3798_379883


namespace consecutive_integer_sum_l3798_379881

theorem consecutive_integer_sum (n : ℕ) :
  (∃ k : ℤ, (k - 2) + (k - 1) + k + (k + 1) + (k + 2) = n) ∧
  (¬ ∃ m : ℤ, (m - 1) + m + (m + 1) + (m + 2) = n) :=
by
  sorry

#check consecutive_integer_sum 225

end consecutive_integer_sum_l3798_379881


namespace remaining_boys_average_weight_l3798_379849

/-- Proves that given a class of 30 boys where 22 boys have an average weight of 50.25 kg
    and the average weight of all boys is 48.89 kg, the average weight of the remaining boys is 45.15 kg. -/
theorem remaining_boys_average_weight :
  let total_boys : ℕ := 30
  let known_boys : ℕ := 22
  let known_boys_avg_weight : ℝ := 50.25
  let all_boys_avg_weight : ℝ := 48.89
  let remaining_boys : ℕ := total_boys - known_boys
  let remaining_boys_avg_weight : ℝ := (total_boys * all_boys_avg_weight - known_boys * known_boys_avg_weight) / remaining_boys
  remaining_boys_avg_weight = 45.15 := by
sorry

end remaining_boys_average_weight_l3798_379849


namespace least_x_squared_divisible_by_240_l3798_379854

theorem least_x_squared_divisible_by_240 :
  ∀ x : ℕ, x > 0 → x^2 % 240 = 0 → x ≥ 60 :=
by
  sorry

end least_x_squared_divisible_by_240_l3798_379854


namespace max_value_sum_of_roots_l3798_379802

theorem max_value_sum_of_roots (x y z : ℝ) 
  (sum_eq : x + y + z = 3)
  (x_ge : x ≥ -1)
  (y_ge : y ≥ -2/3)
  (z_ge : z ≥ -2) :
  (∀ a b c : ℝ, a + b + c = 3 → a ≥ -1 → b ≥ -2/3 → c ≥ -2 →
    Real.sqrt (3*a + 3) + Real.sqrt (3*b + 2) + Real.sqrt (3*c + 6) ≤ 
    Real.sqrt (3*x + 3) + Real.sqrt (3*y + 2) + Real.sqrt (3*z + 6)) ∧
  Real.sqrt (3*x + 3) + Real.sqrt (3*y + 2) + Real.sqrt (3*z + 6) = 2 * Real.sqrt 15 :=
sorry

end max_value_sum_of_roots_l3798_379802


namespace tshirt_cost_l3798_379843

/-- The Razorback t-shirt Shop problem -/
theorem tshirt_cost (total_sales : ℝ) (num_shirts : ℕ) (cost_per_shirt : ℝ)
  (h1 : total_sales = 720)
  (h2 : num_shirts = 45)
  (h3 : cost_per_shirt = total_sales / num_shirts) :
  cost_per_shirt = 16 := by
  sorry

end tshirt_cost_l3798_379843


namespace binomial_expansion_example_l3798_379841

theorem binomial_expansion_example : 
  (0.5 : ℝ)^3 + 3 * (0.5 : ℝ)^2 * (-1.5) + 3 * (0.5 : ℝ) * (-1.5)^2 + (-1.5)^3 = -1 := by
  sorry

end binomial_expansion_example_l3798_379841


namespace certain_number_proof_l3798_379850

theorem certain_number_proof : ∃ n : ℕ, n * 240 = 1038 * 40 ∧ n = 173 := by
  sorry

end certain_number_proof_l3798_379850


namespace right_triangle_product_divisible_by_30_l3798_379867

theorem right_triangle_product_divisible_by_30 (a b c : ℤ) :
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →
  (30 : ℤ) ∣ (a * b * c) :=
sorry

end right_triangle_product_divisible_by_30_l3798_379867


namespace intersection_and_union_range_of_m_l3798_379824

-- Define the sets A, B, and C
def A : Set ℝ := {x | x ≤ -3 ∨ x ≥ 2}
def B : Set ℝ := {x | 1 < x ∧ x < 5}
def C (m : ℝ) : Set ℝ := {x | m - 1 ≤ x ∧ x ≤ 2*m}

-- Theorem for part (Ⅰ)
theorem intersection_and_union :
  (A ∩ B = {x | 2 ≤ x ∧ x < 5}) ∧
  ((Aᶜ ∪ B) = {x | -3 < x ∧ x < 5}) := by sorry

-- Theorem for part (Ⅱ)
theorem range_of_m (m : ℝ) :
  (B ∩ C m = C m) → (m < -1 ∨ (2 < m ∧ m < 5/2)) := by sorry

end intersection_and_union_range_of_m_l3798_379824


namespace negation_of_existence_square_gt_power_negation_l3798_379852

theorem negation_of_existence (p : ℕ → Prop) :
  (¬∃ n : ℕ, n > 1 ∧ p n) ↔ (∀ n : ℕ, n > 1 → ¬(p n)) := by sorry

theorem square_gt_power_negation :
  (¬∃ n : ℕ, n > 1 ∧ n^2 > 2^n) ↔ (∀ n : ℕ, n > 1 → n^2 ≤ 2^n) := by sorry

end negation_of_existence_square_gt_power_negation_l3798_379852


namespace village_foods_sales_l3798_379857

/-- Represents the pricing structure for lettuce -/
structure LettucePricing where
  first : Float
  second : Float
  additional : Float

/-- Represents the pricing structure for tomatoes -/
structure TomatoPricing where
  firstTwo : Float
  nextTwo : Float
  additional : Float

/-- Calculates the total sales per month for Village Foods -/
def totalSalesPerMonth (
  customersPerMonth : Nat
) (
  lettucePerCustomer : Nat
) (
  tomatoesPerCustomer : Nat
) (
  lettucePricing : LettucePricing
) (
  tomatoPricing : TomatoPricing
) (
  discountThreshold : Float
) (
  discountRate : Float
) : Float :=
  sorry

/-- Theorem stating that the total sales per month is $2350 -/
theorem village_foods_sales :
  totalSalesPerMonth
    500  -- customers per month
    2    -- lettuce per customer
    4    -- tomatoes per customer
    { first := 1.50, second := 1.00, additional := 0.75 }  -- lettuce pricing
    { firstTwo := 0.60, nextTwo := 0.50, additional := 0.40 }  -- tomato pricing
    10.00  -- discount threshold
    0.10   -- discount rate
  = 2350.00 :=
by sorry

end village_foods_sales_l3798_379857


namespace age_difference_l3798_379842

/-- Given that the total age of A and B is 16 years more than the total age of B and C,
    prove that C is 16 years younger than A. -/
theorem age_difference (A B C : ℕ) (h : A + B = B + C + 16) : A = C + 16 := by
  sorry

end age_difference_l3798_379842


namespace circle_tangent_to_parabola_directrix_l3798_379836

/-- The value of m for which the circle x^2 + y^2 + mx - 1/4 = 0 is tangent to the directrix of the parabola y^2 = 4x -/
theorem circle_tangent_to_parabola_directrix (x y m : ℝ) : 
  (∃ x y, x^2 + y^2 + m*x - 1/4 = 0) → -- Circle equation
  (∃ x y, y^2 = 4*x) → -- Parabola equation
  (∃ x y, x^2 + y^2 + m*x - 1/4 = 0 ∧ x = -1) → -- Circle is tangent to directrix (x = -1)
  m = 3/4 := by
sorry

end circle_tangent_to_parabola_directrix_l3798_379836


namespace intersection_line_of_circles_l3798_379807

/-- Given two circles in the plane, this theorem proves that the equation of the line
    passing through their intersection points has a specific form. -/
theorem intersection_line_of_circles (x y : ℝ) : 
  (x^2 + y^2 = 10) ∧ ((x-1)^2 + (y-3)^2 = 10) → x + 3*y - 5 = 0 :=
by sorry

end intersection_line_of_circles_l3798_379807


namespace absolute_value_equation_solution_l3798_379862

theorem absolute_value_equation_solution :
  ∀ x : ℝ, (|2*x - 6| = 3*x + 5) ↔ (x = 1/5) :=
by sorry

end absolute_value_equation_solution_l3798_379862


namespace stating_call_ratio_theorem_l3798_379808

/-- Represents the ratio of calls processed by team members -/
structure CallRatio where
  team_a : ℚ
  team_b : ℚ

/-- Represents the distribution of calls and agents between two teams -/
structure CallCenter where
  agent_ratio : ℚ  -- Ratio of team A agents to team B agents
  team_b_calls : ℚ -- Fraction of total calls processed by team B

/-- 
Given a call center with specified agent ratio and call distribution,
calculates the ratio of calls processed by each member of team A to team B
-/
def calculate_call_ratio (cc : CallCenter) : CallRatio :=
  { team_a := 7,
    team_b := 5 }

/-- 
Theorem stating that for a call center where team A has 5/8 as many agents as team B,
and team B processes 8/15 of the calls, the ratio of calls processed per agent
of team A to team B is 7:5
-/
theorem call_ratio_theorem (cc : CallCenter) 
  (h1 : cc.agent_ratio = 5 / 8)
  (h2 : cc.team_b_calls = 8 / 15) :
  calculate_call_ratio cc = { team_a := 7, team_b := 5 } := by
  sorry

end stating_call_ratio_theorem_l3798_379808


namespace parrot_count_theorem_l3798_379846

/-- Represents the types of parrots -/
inductive ParrotType
  | Green
  | Yellow
  | Mottled

/-- Represents the behavior of parrots -/
def ParrotBehavior : ParrotType → Bool → Bool
  | ParrotType.Green, _ => true
  | ParrotType.Yellow, _ => false
  | ParrotType.Mottled, b => b

theorem parrot_count_theorem 
  (total_parrots : Nat)
  (green_count : Nat)
  (yellow_count : Nat)
  (mottled_count : Nat)
  (h_total : total_parrots = 100)
  (h_sum : green_count + yellow_count + mottled_count = total_parrots)
  (h_first_statement : green_count + (mottled_count / 2) = 50)
  (h_second_statement : yellow_count + (mottled_count / 2) = 50)
  : yellow_count = green_count :=
by sorry

end parrot_count_theorem_l3798_379846


namespace combined_weight_l3798_379859

theorem combined_weight (person baby nurse : ℝ)
  (h1 : person + baby = 78)
  (h2 : nurse + baby = 69)
  (h3 : person + nurse = 137) :
  person + nurse + baby = 142 :=
by sorry

end combined_weight_l3798_379859


namespace wire_cutting_problem_l3798_379872

/-- The length of wire pieces that satisfies the given conditions -/
def wire_piece_length : ℕ := 83

theorem wire_cutting_problem (initial_length second_length : ℕ) 
  (h1 : initial_length = 1000)
  (h2 : second_length = 1070)
  (h3 : 12 * wire_piece_length ≤ initial_length)
  (h4 : 12 * wire_piece_length ≤ second_length)
  (h5 : ∀ x : ℕ, x > wire_piece_length → 12 * x > second_length) :
  wire_piece_length = 83 := by
  sorry

end wire_cutting_problem_l3798_379872


namespace rationalize_sqrt_five_twelfths_l3798_379890

theorem rationalize_sqrt_five_twelfths : 
  Real.sqrt (5 / 12) = Real.sqrt 15 / 6 := by sorry

end rationalize_sqrt_five_twelfths_l3798_379890


namespace smallest_n_for_inequality_l3798_379818

theorem smallest_n_for_inequality : 
  ∃ n : ℕ, n > 0 ∧ (∀ k : ℕ, k > 0 → (1 : ℚ) / k - (1 : ℚ) / (k + 1) < (1 : ℚ) / 15 → k ≥ n) ∧ 
  ((1 : ℚ) / n - (1 : ℚ) / (n + 1) < (1 : ℚ) / 15) ∧ n = 4 :=
sorry

end smallest_n_for_inequality_l3798_379818


namespace six_students_arrangement_l3798_379892

/-- The number of ways to arrange n elements -/
def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

/-- The number of ways to arrange n elements, excluding arrangements where two specific elements are adjacent -/
def arrangementWithoutAdjacent (n : ℕ) : ℕ :=
  factorial n - (factorial (n - 1) * 2)

theorem six_students_arrangement :
  arrangementWithoutAdjacent 6 = 480 := by
  sorry

end six_students_arrangement_l3798_379892


namespace first_term_to_common_diff_ratio_l3798_379809

/-- An arithmetic progression with a specific property -/
structure ArithmeticProgression where
  a : ℝ  -- First term
  d : ℝ  -- Common difference
  sum_15_eq_3sum_5 : (15 * a + 105 * d) = 3 * (5 * a + 10 * d)

/-- The ratio of the first term to the common difference is 5:1 -/
theorem first_term_to_common_diff_ratio 
  (ap : ArithmeticProgression) : ap.a / ap.d = 5 := by
  sorry

end first_term_to_common_diff_ratio_l3798_379809


namespace journey_distance_l3798_379839

theorem journey_distance (total_time : ℝ) (speed1 : ℝ) (speed2 : ℝ) 
  (h1 : total_time = 20)
  (h2 : speed1 = 21)
  (h3 : speed2 = 24) : 
  ∃ (distance : ℝ), distance = 448 ∧ 
    total_time = (distance / 2) / speed1 + (distance / 2) / speed2 :=
by
  sorry

end journey_distance_l3798_379839


namespace after_school_program_l3798_379821

/-- Represents the number of students in each class combination --/
structure ClassCombinations where
  drawing_only : ℕ
  chess_only : ℕ
  music_only : ℕ
  drawing_chess : ℕ
  drawing_music : ℕ
  chess_music : ℕ
  all_three : ℕ

/-- The after-school program problem --/
theorem after_school_program 
  (total_students : ℕ) 
  (drawing_students : ℕ) 
  (chess_students : ℕ) 
  (music_students : ℕ) 
  (multi_class_students : ℕ) 
  (h1 : total_students = 30)
  (h2 : drawing_students = 15)
  (h3 : chess_students = 17)
  (h4 : music_students = 12)
  (h5 : multi_class_students = 14)
  : ∃ (c : ClassCombinations), 
    c.drawing_only + c.chess_only + c.music_only + 
    c.drawing_chess + c.drawing_music + c.chess_music + c.all_three = total_students ∧
    c.drawing_only + c.drawing_chess + c.drawing_music + c.all_three = drawing_students ∧
    c.chess_only + c.drawing_chess + c.chess_music + c.all_three = chess_students ∧
    c.music_only + c.drawing_music + c.chess_music + c.all_three = music_students ∧
    c.drawing_chess + c.drawing_music + c.chess_music + c.all_three = multi_class_students ∧
    c.all_three = 2 := by
  sorry

end after_school_program_l3798_379821


namespace joan_football_games_l3798_379810

/-- The number of football games Joan attended this year -/
def games_this_year : ℕ := 4

/-- The number of football games Joan attended last year -/
def games_last_year : ℕ := 9

/-- The total number of football games Joan attended -/
def total_games : ℕ := games_this_year + games_last_year

theorem joan_football_games : total_games = 13 := by
  sorry

end joan_football_games_l3798_379810


namespace number_problem_l3798_379880

theorem number_problem (x : ℕ) (h1 : x + 3927 = 13800) : x = 9873 := by
  sorry

end number_problem_l3798_379880


namespace arithmetic_sequence_problem_l3798_379888

theorem arithmetic_sequence_problem (a : ℕ → ℚ) (S : ℕ → ℚ) : 
  (∀ n, a (n + 1) - a n = 1) →  -- arithmetic sequence with common difference 1
  (∀ n, S n = n * a 1 + n * (n - 1) / 2) →  -- sum formula for arithmetic sequence
  (S 8 = 4 * S 4) →  -- given condition
  a 10 = 19 / 2 := by
sorry

end arithmetic_sequence_problem_l3798_379888


namespace segment_length_after_reflection_l3798_379868

-- Define the points
def Z : ℝ × ℝ := (-5, 3)
def Z' : ℝ × ℝ := (5, 3)

-- Define the reflection over y-axis
def reflect_over_y_axis (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

-- Theorem statement
theorem segment_length_after_reflection :
  Z' = reflect_over_y_axis Z ∧ 
  Real.sqrt ((Z'.1 - Z.1)^2 + (Z'.2 - Z.2)^2) = 10 := by
  sorry

end segment_length_after_reflection_l3798_379868


namespace triangle_isosceles_from_equation_l3798_379851

/-- A triangle with sides a, b, and c is isosceles if it satisfies the equation a^2 - bc = a(b - c) -/
theorem triangle_isosceles_from_equation (a b c : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) (h_eq : a^2 - b*c = a*(b - c)) : 
  a = b ∨ b = c ∨ c = a := by
  sorry

end triangle_isosceles_from_equation_l3798_379851


namespace f_properties_l3798_379853

noncomputable def f (x : ℝ) : ℝ := Real.sin x * (2 * Real.sqrt 3 * Real.cos x - Real.sin x) + 1

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
    (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T')) ∧
  (∀ (x y : ℝ), -π/4 ≤ x ∧ x < y ∧ y ≤ π/6 → f x < f y) ∧
  (∀ (x y : ℝ), π/6 ≤ x ∧ x < y ∧ y ≤ π/4 → f x > f y) := by
  sorry

end f_properties_l3798_379853


namespace cookie_batch_size_l3798_379830

theorem cookie_batch_size 
  (num_batches : ℕ) 
  (num_people : ℕ) 
  (cookies_per_person : ℕ) 
  (h1 : num_batches = 4)
  (h2 : num_people = 16)
  (h3 : cookies_per_person = 6) :
  (num_people * cookies_per_person) / num_batches / 12 = 2 := by
sorry

end cookie_batch_size_l3798_379830


namespace no_perfect_square_solution_l3798_379874

theorem no_perfect_square_solution (n : ℕ) : ¬∃ (m : ℕ), n^5 - 5*n^3 + 4*n + 7 = m^2 := by
  sorry

end no_perfect_square_solution_l3798_379874


namespace arithmetic_sequence_sum_l3798_379879

def arithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmeticSequence a →
  a 1 = 2 →
  a 2 + a 5 = 13 →
  a 5 + a 6 + a 7 = 33 := by
  sorry

end arithmetic_sequence_sum_l3798_379879


namespace minimum_bailing_rate_l3798_379895

/-- The minimum bailing rate problem -/
theorem minimum_bailing_rate
  (distance : ℝ)
  (water_entry_rate : ℝ)
  (max_water_capacity : ℝ)
  (rowing_speed : ℝ)
  (h1 : distance = 2)
  (h2 : water_entry_rate = 8)
  (h3 : max_water_capacity = 50)
  (h4 : rowing_speed = 2)
  : ∃ (bailing_rate : ℝ),
    bailing_rate = 8 ∧
    (∀ r : ℝ, r < 8 →
      (distance / rowing_speed) * (water_entry_rate - r) > max_water_capacity) ∧
    (distance / rowing_speed) * (water_entry_rate - bailing_rate) ≤ max_water_capacity :=
by sorry

end minimum_bailing_rate_l3798_379895


namespace yellow_ball_count_l3798_379885

/-- Given a bag with red and yellow balls, if the probability of drawing a red ball is 0.2,
    then the number of yellow balls is 20. -/
theorem yellow_ball_count (red_balls : ℕ) (yellow_balls : ℕ) :
  red_balls = 5 →
  (red_balls : ℚ) / ((red_balls : ℚ) + (yellow_balls : ℚ)) = 1/5 →
  yellow_balls = 20 := by
  sorry


end yellow_ball_count_l3798_379885

import Mathlib

namespace sarahs_waist_cm_l1345_134532

-- Define the conversion factor from inches to centimeters
def inches_to_cm : ℝ := 2.54

-- Define Sarah's waist size in inches
def sarahs_waist_inches : ℝ := 27

-- Theorem to prove Sarah's waist size in centimeters
theorem sarahs_waist_cm : 
  ∃ (waist_cm : ℝ), abs (waist_cm - (sarahs_waist_inches * inches_to_cm)) < 0.05 :=
by
  sorry

end sarahs_waist_cm_l1345_134532


namespace divisible_by_twelve_l1345_134577

theorem divisible_by_twelve (n : ℤ) (h : n > 1) : ∃ k : ℤ, n^4 - n^2 = 12 * k := by
  sorry

end divisible_by_twelve_l1345_134577


namespace collinear_sufficient_not_necessary_for_coplanar_l1345_134553

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Four points in 3D space -/
structure FourPoints where
  p1 : Point3D
  p2 : Point3D
  p3 : Point3D
  p4 : Point3D

/-- Predicate for three points being collinear -/
def threeCollinear (fp : FourPoints) : Prop :=
  sorry

/-- Predicate for four points being coplanar -/
def fourCoplanar (fp : FourPoints) : Prop :=
  sorry

/-- Theorem stating that three collinear points is a sufficient but not necessary condition for four coplanar points -/
theorem collinear_sufficient_not_necessary_for_coplanar :
  (∀ fp : FourPoints, threeCollinear fp → fourCoplanar fp) ∧
  (∃ fp : FourPoints, fourCoplanar fp ∧ ¬threeCollinear fp) :=
sorry

end collinear_sufficient_not_necessary_for_coplanar_l1345_134553


namespace inequality_proof_l1345_134582

theorem inequality_proof (a b c : ℝ) : 
  a * b + b * c + c * a + max (|a - b|) (max (|b - c|) (|c - a|)) ≤ 1 + (1/3) * (a + b + c)^2 := by
  sorry

end inequality_proof_l1345_134582


namespace optimal_racket_purchase_l1345_134523

/-- Represents the purchase and selling prices of rackets -/
structure RacketPrices where
  tableTennisBuy : ℝ
  tableTennisSell : ℝ
  badmintonBuy : ℝ
  badmintonSell : ℝ

/-- Represents the quantity of rackets to purchase -/
structure RacketQuantities where
  tableTennis : ℝ
  badminton : ℝ

/-- Calculates the profit given prices and quantities -/
def calculateProfit (prices : RacketPrices) (quantities : RacketQuantities) : ℝ :=
  (prices.tableTennisSell - prices.tableTennisBuy) * quantities.tableTennis +
  (prices.badmintonSell - prices.badmintonBuy) * quantities.badminton

/-- The main theorem stating the optimal solution -/
theorem optimal_racket_purchase
  (prices : RacketPrices)
  (h1 : 2 * prices.tableTennisBuy + prices.badmintonBuy = 120)
  (h2 : 4 * prices.tableTennisBuy + 3 * prices.badmintonBuy = 270)
  (h3 : prices.tableTennisSell = 55)
  (h4 : prices.badmintonSell = 50)
  : ∃ (quantities : RacketQuantities),
    quantities.tableTennis + quantities.badminton = 300 ∧
    quantities.tableTennis ≥ (1/3) * quantities.badminton ∧
    prices.tableTennisBuy = 45 ∧
    prices.badmintonBuy = 30 ∧
    quantities.tableTennis = 75 ∧
    quantities.badminton = 225 ∧
    calculateProfit prices quantities = 5250 ∧
    ∀ (other : RacketQuantities),
      other.tableTennis + other.badminton = 300 →
      other.tableTennis ≥ (1/3) * other.badminton →
      calculateProfit prices quantities ≥ calculateProfit prices other := by
  sorry

end optimal_racket_purchase_l1345_134523


namespace expression_equivalence_l1345_134531

theorem expression_equivalence :
  (5 + 2) * (5^2 + 2^2) * (5^4 + 2^4) * (5^8 + 2^8) * (5^16 + 2^16) * (5^32 + 2^32) * (5^64 + 2^64) = 5^128 - 2^128 := by
  sorry

end expression_equivalence_l1345_134531


namespace triangle_area_proof_l1345_134508

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that the area of the triangle is √7/4 under the given conditions. -/
theorem triangle_area_proof (A B C : Real) (a b c : Real) :
  sinA = 2 * sinB →
  c = Real.sqrt 2 →
  cosC = 3 / 4 →
  (1 / 2) * a * b * sinC = Real.sqrt 7 / 4 := by
  sorry

end triangle_area_proof_l1345_134508


namespace smallest_positive_quadratic_form_l1345_134519

def quadratic_form (x y : ℤ) : ℤ := 20 * x^2 + 80 * x * y + 95 * y^2

theorem smallest_positive_quadratic_form :
  (∃ x y : ℤ, quadratic_form x y = 67) ∧
  (∀ n : ℕ, n > 0 → n < 67 → ∀ x y : ℤ, quadratic_form x y ≠ n) :=
sorry

end smallest_positive_quadratic_form_l1345_134519


namespace complex_power_sum_l1345_134520

theorem complex_power_sum : ∃ (i : ℂ), i^2 = -1 ∧ (1 - i)^2016 + (1 + i)^2016 = 2^1009 := by
  sorry

end complex_power_sum_l1345_134520


namespace sum_three_digit_integers_mod_1000_l1345_134571

def sum_three_digit_integers : ℕ :=
  (45 * 100 * 100) + (45 * 100 * 10) + (45 * 100)

theorem sum_three_digit_integers_mod_1000 :
  sum_three_digit_integers % 1000 = 500 := by sorry

end sum_three_digit_integers_mod_1000_l1345_134571


namespace perpendicular_lines_m_values_l1345_134533

theorem perpendicular_lines_m_values (m : ℝ) : 
  (∃ (x y : ℝ), mx + 2*y + 1 = 0 ∧ x - m^2*y + 1/2 = 0) →
  (m * 1 + 2 * (-m^2) = 0) →
  (m = 0 ∨ m = 1/2) :=
sorry

end perpendicular_lines_m_values_l1345_134533


namespace derivative_zero_necessary_not_sufficient_l1345_134512

-- Define a real-valued function
variable (f : ℝ → ℝ)

-- Define the property of being an extreme value point
def is_extreme_point (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∀ x, f x ≤ f x₀ ∨ f x ≥ f x₀

-- Define the theorem
theorem derivative_zero_necessary_not_sufficient :
  (∀ x₀ : ℝ, is_extreme_point f x₀ → (deriv f) x₀ = 0) ∧
  (∃ x₀ : ℝ, (deriv f) x₀ = 0 ∧ ¬(is_extreme_point f x₀)) :=
sorry

end derivative_zero_necessary_not_sufficient_l1345_134512


namespace min_value_of_sum_of_squares_l1345_134526

theorem min_value_of_sum_of_squares (x y z : ℝ) (h : 2*x + 3*y + 4*z = 1) :
  ∃ (m : ℝ), (∀ a b c : ℝ, 2*a + 3*b + 4*c = 1 → a^2 + b^2 + c^2 ≥ m) ∧
             (x^2 + y^2 + z^2 = m) ∧
             (m = 1/29) :=
by sorry

end min_value_of_sum_of_squares_l1345_134526


namespace solve_system_l1345_134517

theorem solve_system (p q : ℚ) 
  (eq1 : 5 * p + 6 * q = 10) 
  (eq2 : 6 * p + 5 * q = 17) : 
  p = 52 / 11 := by
sorry

end solve_system_l1345_134517


namespace intersection_of_A_and_B_l1345_134503

def A : Set (ℝ × ℝ) := {p | p.2 = p.1 + 1}
def B : Set (ℝ × ℝ) := {p | p.2 = 4 - 2*p.1}

theorem intersection_of_A_and_B :
  A ∩ B = {(1, 2)} := by sorry

end intersection_of_A_and_B_l1345_134503


namespace total_tiles_count_l1345_134568

def room_length : ℕ := 18
def room_width : ℕ := 15
def border_tile_size : ℕ := 2
def border_width : ℕ := 2
def inner_tile_size : ℕ := 3

def border_tiles : ℕ := 
  2 * (room_length / border_tile_size + room_width / border_tile_size) + 4

def inner_area : ℕ := (room_length - 2 * border_width) * (room_width - 2 * border_width)
def inner_tiles : ℕ := inner_area / (inner_tile_size * inner_tile_size)

theorem total_tiles_count :
  border_tiles + inner_tiles = 45 := by sorry

end total_tiles_count_l1345_134568


namespace range_of_m_l1345_134562

-- Define set A
def A : Set ℝ := {x : ℝ | (x + 1) * (x - 6) ≤ 0}

-- Define set B
def B (m : ℝ) : Set ℝ := {x : ℝ | m - 1 ≤ x ∧ x ≤ 2 * m + 1}

-- Theorem statement
theorem range_of_m (m : ℝ) : 
  (A ∩ B m = B m) ↔ (m < -2 ∨ (0 ≤ m ∧ m ≤ 5/2)) :=
sorry

end range_of_m_l1345_134562


namespace expression_evaluation_l1345_134591

theorem expression_evaluation :
  let expr := 3 * 15 + 20 / 4 + 1
  let max_expr := 3 * (15 + 20 / 4 + 1)
  let min_expr := (3 * 15 + 20) / (4 + 1)
  (expr = 51) ∧ 
  (max_expr = 63) ∧ 
  (min_expr = 13) ∧
  (∀ x : ℤ, (∃ e : ℤ → ℤ, e expr = x) → (x ≤ max_expr ∧ x ≥ min_expr)) :=
by sorry

end expression_evaluation_l1345_134591


namespace min_value_trig_expression_min_value_trig_expression_achievable_l1345_134507

theorem min_value_trig_expression (α β : ℝ) :
  (3 * Real.cos α + 4 * Real.sin β - 5)^2 + (3 * Real.sin α + 4 * Real.cos β - 12)^2 ≥ 36 :=
by sorry

theorem min_value_trig_expression_achievable :
  ∃ (α β : ℝ), (3 * Real.cos α + 4 * Real.sin β - 5)^2 + (3 * Real.sin α + 4 * Real.cos β - 12)^2 = 36 :=
by sorry

end min_value_trig_expression_min_value_trig_expression_achievable_l1345_134507


namespace oil_in_barrels_l1345_134578

theorem oil_in_barrels (barrel_a barrel_b : ℚ) : 
  barrel_a = 3/4 → 
  barrel_b = barrel_a + 1/10 → 
  barrel_a + barrel_b = 8/5 := by
sorry

end oil_in_barrels_l1345_134578


namespace negation_equivalence_l1345_134570

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (student : U → Prop)
variable (shares_truth : U → Prop)

-- Define the original statement
def every_student_shares_truth : Prop :=
  ∀ x, student x → shares_truth x

-- Define the negation
def negation_statement : Prop :=
  ∃ x, student x ∧ ¬(shares_truth x)

-- Theorem to prove
theorem negation_equivalence :
  ¬(every_student_shares_truth U student shares_truth) ↔ negation_statement U student shares_truth :=
by sorry

end negation_equivalence_l1345_134570


namespace highway_vehicles_l1345_134574

theorem highway_vehicles (total : ℕ) (trucks : ℕ) (cars : ℕ) 
  (h1 : total = 300)
  (h2 : cars = 2 * trucks)
  (h3 : total = cars + trucks) :
  trucks = 100 := by
  sorry

end highway_vehicles_l1345_134574


namespace reflect_d_twice_l1345_134567

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflects a point over the y-axis -/
def reflectOverYAxis (p : Point) : Point :=
  { x := -p.x, y := p.y }

/-- Reflects a point over the line y = -x -/
def reflectOverYEqualNegX (p : Point) : Point :=
  { x := -p.y, y := -p.x }

/-- The main theorem stating that reflecting point D(5,1) over y-axis and then over y=-x results in D''(-1,5) -/
theorem reflect_d_twice :
  let d : Point := { x := 5, y := 1 }
  let d' := reflectOverYAxis d
  let d'' := reflectOverYEqualNegX d'
  d''.x = -1 ∧ d''.y = 5 := by sorry

end reflect_d_twice_l1345_134567


namespace no_solution_quadratic_inequality_l1345_134557

theorem no_solution_quadratic_inequality (x : ℝ) : 
  (5 * x^2 + 6 * x + 8 < 0) ∧ (abs x > 2) → False :=
by
  sorry


end no_solution_quadratic_inequality_l1345_134557


namespace decreasing_function_implies_a_less_than_one_l1345_134592

/-- A function f: ℝ → ℝ is decreasing if for all x y, x < y implies f x > f y -/
def Decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

/-- The function f(x) = (a-1)x + 1 -/
def f (a : ℝ) : ℝ → ℝ := λ x ↦ (a - 1) * x + 1

theorem decreasing_function_implies_a_less_than_one (a : ℝ) :
  Decreasing (f a) → a < 1 := by
  sorry

end decreasing_function_implies_a_less_than_one_l1345_134592


namespace intersection_complement_equality_l1345_134513

-- Define set A
def A : Set ℝ := {x | 1 < x ∧ x < 4}

-- Define set B
def B : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

-- Theorem statement
theorem intersection_complement_equality : A ∩ (Set.univ \ B) = Set.Ioo 3 4 := by
  sorry

end intersection_complement_equality_l1345_134513


namespace coin_flip_problem_l1345_134583

theorem coin_flip_problem (p_heads : ℝ) (p_event : ℝ) (n : ℕ) :
  p_heads = 1/2 →
  p_event = 0.03125 →
  p_event = p_heads * (1 - p_heads)^4 →
  n = 5 :=
by
  sorry

end coin_flip_problem_l1345_134583


namespace three_power_fraction_equals_41_40_l1345_134527

theorem three_power_fraction_equals_41_40 :
  (3^1008 + 3^1004) / (3^1008 - 3^1004) = 41/40 := by
  sorry

end three_power_fraction_equals_41_40_l1345_134527


namespace arithmetic_sequence_sum_l1345_134587

/-- Given an arithmetic sequence {aₙ} where Sₙ denotes the sum of its first n terms,
    if a₄ + a₆ + a₈ = 15, then S₁₁ = 55. -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence condition
  (∀ n : ℕ, S n = (n : ℝ) * (a 1 + a n) / 2) →          -- sum formula
  a 4 + a 6 + a 8 = 15 →                                -- given condition
  S 11 = 55 :=
by sorry

end arithmetic_sequence_sum_l1345_134587


namespace min_value_of_absolute_sum_l1345_134511

theorem min_value_of_absolute_sum (x : ℚ) : 
  ∀ x : ℚ, |3 - x| + |x - 2| + |-1 + x| ≥ 2 ∧ 
  ∃ x : ℚ, |3 - x| + |x - 2| + |-1 + x| = 2 :=
by sorry

end min_value_of_absolute_sum_l1345_134511


namespace probability_of_mixed_selection_l1345_134596

theorem probability_of_mixed_selection (n_boys n_girls n_select : ℕ) :
  n_boys = 5 →
  n_girls = 2 →
  n_select = 3 →
  (Nat.choose (n_boys + n_girls) n_select - Nat.choose n_boys n_select - Nat.choose n_girls n_select) / Nat.choose (n_boys + n_girls) n_select = 3 / 5 := by
  sorry

end probability_of_mixed_selection_l1345_134596


namespace greatest_b_value_l1345_134579

theorem greatest_b_value (b : ℝ) : 
  (∀ x : ℝ, -x^2 + 8*x - 15 ≥ 0 → x ≤ 5) ∧ (-5^2 + 8*5 - 15 ≥ 0) := by
  sorry

end greatest_b_value_l1345_134579


namespace least_N_congruence_l1345_134501

/-- Sum of digits in base 3 representation -/
def f (n : ℕ) : ℕ := sorry

/-- Sum of digits in base 8 representation of f(n) -/
def g (n : ℕ) : ℕ := sorry

/-- The least value of n such that g(n) ≥ 10 -/
def N : ℕ := sorry

theorem least_N_congruence : N ≡ 862 [MOD 1000] := by sorry

end least_N_congruence_l1345_134501


namespace noah_lights_on_time_l1345_134537

def bedroom_wattage : ℝ := 6
def office_wattage : ℝ := 3 * bedroom_wattage
def living_room_wattage : ℝ := 4 * bedroom_wattage
def total_energy_used : ℝ := 96

def total_wattage_per_hour : ℝ := bedroom_wattage + office_wattage + living_room_wattage

theorem noah_lights_on_time :
  total_energy_used / total_wattage_per_hour = 2 := by sorry

end noah_lights_on_time_l1345_134537


namespace range_of_a_l1345_134556

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, Real.exp x - a * Real.log (a * x - a) + a > 0) →
  a > 0 →
  0 < a ∧ a < Real.exp 2 :=
by sorry

end range_of_a_l1345_134556


namespace remaining_travel_distance_l1345_134559

theorem remaining_travel_distance 
  (total_distance : ℕ)
  (amoli_speed : ℕ)
  (amoli_time : ℕ)
  (anayet_speed : ℕ)
  (anayet_time : ℕ)
  (h1 : total_distance = 369)
  (h2 : amoli_speed = 42)
  (h3 : amoli_time = 3)
  (h4 : anayet_speed = 61)
  (h5 : anayet_time = 2) :
  total_distance - (amoli_speed * amoli_time + anayet_speed * anayet_time) = 121 :=
by sorry

end remaining_travel_distance_l1345_134559


namespace min_value_of_expression_l1345_134552

theorem min_value_of_expression (a b c : ℝ) 
  (sum_condition : a + b + c = -1)
  (product_condition : a * b * c ≤ -3) :
  (a * b + 1) / (a + b) + (b * c + 1) / (b + c) + (c * a + 1) / (c + a) ≥ 3 := by
  sorry

end min_value_of_expression_l1345_134552


namespace gcd_lcm_product_l1345_134500

theorem gcd_lcm_product (a b : ℕ+) (h : Nat.gcd a b * Nat.lcm a b = 252) :
  (∃ s : Finset ℕ+, s.card = 4 ∧ ∀ x : ℕ+, x ∈ s ↔ ∃ a b : ℕ+, Nat.gcd a b = x ∧ Nat.gcd a b * Nat.lcm a b = 252) :=
sorry

end gcd_lcm_product_l1345_134500


namespace min_zeros_in_special_set_l1345_134597

theorem min_zeros_in_special_set (n : ℕ) (a : Fin n → ℝ) 
  (h : n = 2011)
  (sum_property : ∀ i j k : Fin n, ∃ l : Fin n, a i + a j + a k = a l) :
  (Finset.filter (fun i => a i = 0) Finset.univ).card ≥ 2009 :=
sorry

end min_zeros_in_special_set_l1345_134597


namespace fourth_week_miles_l1345_134563

-- Define the number of weeks
def num_weeks : ℕ := 4

-- Define the number of days walked per week
def days_per_week : ℕ := 6

-- Define the miles walked per day for each week
def miles_per_day (week : ℕ) : ℕ :=
  if week < 4 then week else 0  -- The 4th week is unknown, so we set it to 0 initially

-- Define the total miles walked
def total_miles : ℕ := 60

-- Theorem to prove
theorem fourth_week_miles :
  ∃ (x : ℕ), 
    (miles_per_day 1 * days_per_week +
     miles_per_day 2 * days_per_week +
     miles_per_day 3 * days_per_week +
     x * days_per_week = total_miles) ∧
    x = 4 := by
  sorry

end fourth_week_miles_l1345_134563


namespace simplified_expression_equals_half_l1345_134502

theorem simplified_expression_equals_half :
  let x : ℚ := 1/3
  let y : ℚ := -1/2
  (2*x + 3*y)^2 - (2*x + y)*(2*x - y) = 1/2 :=
by sorry

end simplified_expression_equals_half_l1345_134502


namespace triangle_area_implies_q_value_l1345_134518

/-- Given a triangle DEF with vertices D(3, 15), E(15, 0), and F(0, q),
    if the area of the triangle is 30, then q = 12.5 -/
theorem triangle_area_implies_q_value :
  ∀ q : ℝ,
  let D : ℝ × ℝ := (3, 15)
  let E : ℝ × ℝ := (15, 0)
  let F : ℝ × ℝ := (0, q)
  let triangle_area := abs ((3 * q + 15 * q - 45) / 2)
  triangle_area = 30 → q = 12.5 := by
  sorry

end triangle_area_implies_q_value_l1345_134518


namespace smallest_non_special_number_l1345_134536

def is_triangular (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * (k + 1) / 2

def is_prime_power (n : ℕ) : Prop :=
  ∃ p k : ℕ, Nat.Prime p ∧ n = p ^ k

def is_prime_plus_one (n : ℕ) : Prop :=
  ∃ p : ℕ, Nat.Prime p ∧ n = p + 1

def is_product_of_distinct_primes (n : ℕ) : Prop :=
  ∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ n = p * q

theorem smallest_non_special_number : 
  (∀ n < 40, is_triangular n ∨ is_prime_power n ∨ is_prime_plus_one n ∨ is_product_of_distinct_primes n) ∧
  ¬(is_triangular 40 ∨ is_prime_power 40 ∨ is_prime_plus_one 40 ∨ is_product_of_distinct_primes 40) :=
sorry

end smallest_non_special_number_l1345_134536


namespace problem_solution_l1345_134564

theorem problem_solution (x y : ℝ) 
  (h1 : x = 151)
  (h2 : x^3 * y - 4 * x^2 * y + 4 * x * y = 342200) : 
  y = 342200 / 3354151 := by sorry

end problem_solution_l1345_134564


namespace circle_square_area_ratio_l1345_134573

theorem circle_square_area_ratio (r : ℝ) (h : r > 0) :
  let inner_square_side := 3 * r
  let outer_circle_radius := inner_square_side * Real.sqrt 2 / 2
  let outer_square_side := 2 * outer_circle_radius
  (π * r^2) / (outer_square_side^2) = π / 18 := by
  sorry

end circle_square_area_ratio_l1345_134573


namespace triangle_value_l1345_134550

theorem triangle_value (triangle r : ℝ) 
  (h1 : triangle + r = 72)
  (h2 : (triangle + r) + r = 117) : 
  triangle = 27 := by
sorry

end triangle_value_l1345_134550


namespace missing_fibonacci_term_l1345_134561

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci n + fibonacci (n + 1)

theorem missing_fibonacci_term : ∃ x : ℕ, 
  fibonacci 0 = 1 ∧ 
  fibonacci 1 = 1 ∧ 
  fibonacci 2 = 2 ∧ 
  fibonacci 3 = 3 ∧ 
  fibonacci 4 = 5 ∧ 
  fibonacci 5 = x ∧ 
  fibonacci 6 = 13 ∧ 
  x = 8 := by
  sorry

end missing_fibonacci_term_l1345_134561


namespace unique_number_exists_l1345_134515

def is_valid_number (n : ℕ) : Prop :=
  ∃ (a b c d e : ℕ),
    n = a * 10000 + b * 1000 + c * 100 + d * 10 + e ∧
    a ∈ ({1, 2, 3, 4, 5} : Set ℕ) ∧
    b ∈ ({1, 2, 3, 4, 5} : Set ℕ) ∧
    c ∈ ({1, 2, 3, 4, 5} : Set ℕ) ∧
    d ∈ ({1, 2, 3, 4, 5} : Set ℕ) ∧
    e ∈ ({1, 2, 3, 4, 5} : Set ℕ) ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
    c ≠ d ∧ c ≠ e ∧
    d ≠ e ∧
    (a * 100 + b * 10 + c) % 4 = 0 ∧
    (b * 100 + c * 10 + d) % 5 = 0 ∧
    (c * 100 + d * 10 + e) % 3 = 0

theorem unique_number_exists : ∃! n : ℕ, is_valid_number n :=
sorry

end unique_number_exists_l1345_134515


namespace cube_surface_area_increase_l1345_134505

theorem cube_surface_area_increase (L : ℝ) (h : L > 0) :
  let original_surface_area := 6 * L^2
  let new_edge_length := 1.4 * L
  let new_surface_area := 6 * new_edge_length^2
  (new_surface_area - original_surface_area) / original_surface_area = 0.96 := by
sorry

end cube_surface_area_increase_l1345_134505


namespace complex_power_2019_l1345_134590

theorem complex_power_2019 (i : ℂ) (h : i^2 = -1) : i^2019 = -i := by
  sorry

end complex_power_2019_l1345_134590


namespace slope_of_AA_l1345_134509

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a triangle in 2D space
structure Triangle where
  A : Point2D
  B : Point2D
  C : Point2D

-- Define the transformation (shift right by 2 and reflect across y=x)
def transform (p : Point2D) : Point2D :=
  { x := p.y, y := p.x + 2 }

-- Theorem statement
theorem slope_of_AA'_is_one (t : Triangle)
  (h1 : t.A.x ≥ 0 ∧ t.A.y ≥ 0)  -- A is in first quadrant
  (h2 : t.B.x ≥ 0 ∧ t.B.y ≥ 0)  -- B is in first quadrant
  (h3 : t.C.x ≥ 0 ∧ t.C.y ≥ 0)  -- C is in first quadrant
  (h4 : t.A.x + 2 ≥ 0 ∧ t.A.y ≥ 0)  -- A+2 is in first quadrant
  (h5 : t.B.x + 2 ≥ 0 ∧ t.B.y ≥ 0)  -- B+2 is in first quadrant
  (h6 : t.C.x + 2 ≥ 0 ∧ t.C.y ≥ 0)  -- C+2 is in first quadrant
  (h7 : t.A.x ≠ t.A.y)  -- A not on y=x
  (h8 : t.B.x ≠ t.B.y)  -- B not on y=x
  (h9 : t.C.x ≠ t.C.y)  -- C not on y=x
  : (transform t.A).y - t.A.y = (transform t.A).x - t.A.x :=
by sorry

end slope_of_AA_l1345_134509


namespace infinite_solutions_exist_l1345_134560

theorem infinite_solutions_exist :
  ∃ m : ℕ+, ∀ n : ℕ, ∃ a b c : ℕ+,
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (1 : ℚ) / a + (1 : ℚ) / b + (1 : ℚ) / c + (1 : ℚ) / (a * b * c) = m / (a + b + c) :=
sorry

end infinite_solutions_exist_l1345_134560


namespace p_necessary_not_sufficient_for_q_l1345_134535

-- Define the basic structures
structure Line :=
  (id : ℕ)

structure Plane :=
  (id : ℕ)

-- Define the perpendicular relationships
def perpendicularToCountlessLines (l : Line) (p : Plane) : Prop :=
  sorry

def perpendicularToPlane (l : Line) : Plane → Prop :=
  sorry

-- Define the conditions p and q
def p (a : Line) (α : Plane) : Prop :=
  perpendicularToCountlessLines a α

def q (a : Line) (α : Plane) : Prop :=
  perpendicularToPlane a α

-- Theorem statement
theorem p_necessary_not_sufficient_for_q :
  (∀ (a : Line) (α : Plane), q a α → p a α) ∧
  (∃ (a : Line) (α : Plane), p a α ∧ ¬(q a α)) :=
sorry

end p_necessary_not_sufficient_for_q_l1345_134535


namespace symmetric_circle_equation_l1345_134525

/-- Given a circle and a line of symmetry, this theorem proves the equation of the symmetric circle. -/
theorem symmetric_circle_equation (x y : ℝ) :
  (x^2 + y^2 + 2*x - 2*y + 1 = 0) →  -- Given circle equation
  (x - y = 0) →                      -- Line of symmetry
  (x^2 + y^2 - 2*x + 2*y + 1 = 0)    -- Symmetric circle equation
:= by sorry

end symmetric_circle_equation_l1345_134525


namespace problem_solving_probability_l1345_134575

theorem problem_solving_probability (xavier_prob yvonne_prob zelda_prob : ℚ)
  (hx : xavier_prob = 1 / 4)
  (hy : yvonne_prob = 1 / 3)
  (hz : zelda_prob = 5 / 8) :
  xavier_prob * yvonne_prob * (1 - zelda_prob) = 1 / 32 := by
  sorry

end problem_solving_probability_l1345_134575


namespace power_mod_eight_l1345_134510

theorem power_mod_eight : 3^23 ≡ 3 [MOD 8] := by sorry

end power_mod_eight_l1345_134510


namespace company_signs_used_l1345_134598

/-- The number of signs in the special sign language --/
def total_signs : ℕ := 124

/-- The number of unused signs --/
def unused_signs : ℕ := 2

/-- The number of additional area codes if all signs were used --/
def additional_codes : ℕ := 488

/-- The number of signs in each area code --/
def signs_per_code : ℕ := 2

/-- The number of signs used fully by the company --/
def signs_used : ℕ := total_signs - unused_signs

theorem company_signs_used : signs_used = 120 := by
  sorry

end company_signs_used_l1345_134598


namespace cube_sum_and_reciprocal_l1345_134522

theorem cube_sum_and_reciprocal (x : ℝ) (h : x + 1/x = -3) : x^3 + 1/x^3 = -18 := by
  sorry

end cube_sum_and_reciprocal_l1345_134522


namespace cross_symmetry_l1345_134555

/-- Represents a square in the cross shape --/
inductive Square
| TopLeft
| TopRight
| Center
| BottomLeft
| BottomRight

/-- Represents a cross shape made of 5 squares --/
def CrossShape := Square → Square

/-- Defines the diagonal reflection operation --/
def diagonalReflection (c : CrossShape) : CrossShape :=
  fun s => match s with
  | Square.TopLeft => c Square.BottomRight
  | Square.TopRight => c Square.BottomLeft
  | Square.Center => c Square.Center
  | Square.BottomLeft => c Square.TopRight
  | Square.BottomRight => c Square.TopLeft

/-- Theorem: A cross shape is symmetric with respect to diagonal reflection
    if and only if it satisfies the specified swap conditions --/
theorem cross_symmetry (c : CrossShape) :
  (∀ s : Square, diagonalReflection c s = c s) ↔
  (c Square.TopRight = Square.BottomLeft ∧
   c Square.BottomLeft = Square.TopRight ∧
   c Square.TopLeft = Square.BottomRight ∧
   c Square.BottomRight = Square.TopLeft ∧
   c Square.Center = Square.Center) :=
by sorry


end cross_symmetry_l1345_134555


namespace complement_of_at_least_two_defective_l1345_134554

/-- Represents the number of products in the sample -/
def sample_size : ℕ := 10

/-- Represents the event of having at least two defective products -/
def event_A (defective : ℕ) : Prop := defective ≥ 2

/-- Represents the complementary event of A -/
def complement_A (defective : ℕ) : Prop := defective ≤ 1

/-- Theorem stating that the complement of event A is "at most one defective product" -/
theorem complement_of_at_least_two_defective :
  ∀ (defective : ℕ), defective ≤ sample_size →
    (¬ event_A defective) ↔ complement_A defective := by
  sorry

end complement_of_at_least_two_defective_l1345_134554


namespace monic_quartic_specific_values_l1345_134594

-- Define a monic quartic polynomial
def is_monic_quartic (f : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, ∀ x, f x = x^4 + a*x^3 + b*x^2 + c*x + d

-- State the theorem
theorem monic_quartic_specific_values (f : ℝ → ℝ) 
  (h_monic : is_monic_quartic f)
  (h1 : f (-2) = -4)
  (h2 : f 1 = -1)
  (h3 : f (-4) = -16)
  (h4 : f 5 = -25) :
  f 0 = 40 := by sorry

end monic_quartic_specific_values_l1345_134594


namespace integer_area_iff_specific_lengths_l1345_134589

/-- A right triangle with a circumscribed circle -/
structure RightTriangleWithCircle where
  AB : ℝ  -- Length of side AB
  BC : ℝ  -- Length of side BC (diameter of the circle)
  h : AB > 0
  d : BC > 0
  right_angle : AB * BC = AB^2  -- Condition for right angle and tangency

/-- The area of the triangle is an integer -/
def has_integer_area (t : RightTriangleWithCircle) : Prop :=
  ∃ n : ℕ, (1/2) * t.AB * t.BC = n

/-- The main theorem -/
theorem integer_area_iff_specific_lengths (t : RightTriangleWithCircle) :
  has_integer_area t ↔ t.AB ∈ ({4, 8, 12} : Set ℝ) :=
sorry

end integer_area_iff_specific_lengths_l1345_134589


namespace function_equivalence_l1345_134551

open Real

theorem function_equivalence (x : ℝ) :
  2 * (cos x)^2 - Real.sqrt 3 * sin (2 * x) = 2 * sin (2 * (x + 5 * π / 12)) + 1 := by
  sorry

end function_equivalence_l1345_134551


namespace initial_people_at_table_l1345_134541

theorem initial_people_at_table (initial : ℕ) 
  (h1 : initial ≥ 6)
  (h2 : initial - 6 + 5 = 10) : initial = 11 := by
  sorry

end initial_people_at_table_l1345_134541


namespace g_zero_at_three_l1345_134566

def g (x s : ℝ) : ℝ := 3 * x^4 + 2 * x^3 - x^2 - 4 * x + s

theorem g_zero_at_three (s : ℝ) : g 3 s = 0 ↔ s = -276 := by sorry

end g_zero_at_three_l1345_134566


namespace average_weight_calculation_l1345_134585

theorem average_weight_calculation (total_boys : ℕ) (group1_boys : ℕ) (group2_boys : ℕ)
  (group2_avg_weight : ℝ) (total_avg_weight : ℝ) :
  total_boys = group1_boys + group2_boys →
  group2_boys = 8 →
  group2_avg_weight = 45.15 →
  total_avg_weight = 48.55 →
  let group1_avg_weight := (total_boys * total_avg_weight - group2_boys * group2_avg_weight) / group1_boys
  group1_avg_weight = 50.25 := by
sorry

end average_weight_calculation_l1345_134585


namespace multiples_of_12_around_negative_150_l1345_134580

theorem multiples_of_12_around_negative_150 :
  ∀ n m : ℤ,
  (∀ k : ℤ, 12 * k < -150 → k ≤ n) →
  (∀ j : ℤ, 12 * j > -150 → m ≤ j) →
  12 * n = -156 ∧ 12 * m = -144 :=
by
  sorry

end multiples_of_12_around_negative_150_l1345_134580


namespace rectangle_diagonal_l1345_134524

/-- Given a rectangle with perimeter 72 meters and length-to-width ratio of 5:2,
    its diagonal length is 194/7 meters. -/
theorem rectangle_diagonal (length width : ℝ) : 
  2 * (length + width) = 72 →
  length / width = 5 / 2 →
  Real.sqrt (length^2 + width^2) = 194 / 7 := by
sorry

end rectangle_diagonal_l1345_134524


namespace rectangular_plot_poles_l1345_134534

/-- Calculates the number of fence poles needed for a rectangular plot -/
def fence_poles (length width pole_distance : ℕ) : ℕ :=
  (2 * (length + width)) / pole_distance

/-- Theorem: A 60m by 50m rectangular plot with poles 5m apart needs 44 poles -/
theorem rectangular_plot_poles :
  fence_poles 60 50 5 = 44 := by
  sorry

end rectangular_plot_poles_l1345_134534


namespace infinitely_many_pairs_with_roots_product_one_l1345_134542

theorem infinitely_many_pairs_with_roots_product_one :
  ∀ n : ℕ, n > 2 →
  ∃ a b : ℤ,
    (∃ x y : ℝ, x ≠ y ∧ x * y = 1 ∧
      x^2019 = a * x + b ∧ y^2019 = a * y + b) ∧
    (∀ m : ℕ, m > 2 → m ≠ n →
      ∃ c d : ℤ, c ≠ a ∨ d ≠ b ∧
        (∃ u v : ℝ, u ≠ v ∧ u * v = 1 ∧
          u^2019 = c * u + d ∧ v^2019 = c * v + d)) :=
by sorry

end infinitely_many_pairs_with_roots_product_one_l1345_134542


namespace no_valid_sequence_for_arrangement_D_l1345_134581

/-- Represents a cell in the 2x4 grid -/
inductive Cell
| topLeft | topMidLeft | topMidRight | topRight
| bottomLeft | bottomMidLeft | bottomMidRight | bottomRight

/-- Checks if two cells are adjacent (share a common vertex) -/
def adjacent (c1 c2 : Cell) : Prop :=
  match c1, c2 with
  | Cell.topLeft, Cell.topMidLeft | Cell.topLeft, Cell.bottomLeft | Cell.topLeft, Cell.bottomMidLeft => True
  | Cell.topMidLeft, Cell.topLeft | Cell.topMidLeft, Cell.topMidRight | Cell.topMidLeft, Cell.bottomLeft | Cell.topMidLeft, Cell.bottomMidLeft | Cell.topMidLeft, Cell.bottomMidRight => True
  | Cell.topMidRight, Cell.topMidLeft | Cell.topMidRight, Cell.topRight | Cell.topMidRight, Cell.bottomMidLeft | Cell.topMidRight, Cell.bottomMidRight | Cell.topMidRight, Cell.bottomRight => True
  | Cell.topRight, Cell.topMidRight | Cell.topRight, Cell.bottomMidRight | Cell.topRight, Cell.bottomRight => True
  | Cell.bottomLeft, Cell.topLeft | Cell.bottomLeft, Cell.topMidLeft | Cell.bottomLeft, Cell.bottomMidLeft => True
  | Cell.bottomMidLeft, Cell.topLeft | Cell.bottomMidLeft, Cell.topMidLeft | Cell.bottomMidLeft, Cell.topMidRight | Cell.bottomMidLeft, Cell.bottomLeft | Cell.bottomMidLeft, Cell.bottomMidRight => True
  | Cell.bottomMidRight, Cell.topMidLeft | Cell.bottomMidRight, Cell.topMidRight | Cell.bottomMidRight, Cell.topRight | Cell.bottomMidRight, Cell.bottomMidLeft | Cell.bottomMidRight, Cell.bottomRight => True
  | Cell.bottomRight, Cell.topMidRight | Cell.bottomRight, Cell.topRight | Cell.bottomRight, Cell.bottomMidRight => True
  | _, _ => False

/-- Represents a sequence of cell selections -/
def CellSequence := List Cell

/-- Checks if a cell sequence is valid according to the rules -/
def validSequence (seq : CellSequence) : Prop :=
  match seq with
  | [] => True
  | [_] => True
  | c1 :: c2 :: rest => adjacent c1 c2 ∧ validSequence (c2 :: rest)

/-- Represents the arrangement D -/
def arrangementD : List Cell :=
  [Cell.topLeft, Cell.topMidLeft, Cell.topMidRight, Cell.topRight,
   Cell.bottomLeft, Cell.bottomMidRight, Cell.bottomMidLeft, Cell.bottomRight]

/-- Theorem stating that no valid sequence can produce arrangement D -/
theorem no_valid_sequence_for_arrangement_D :
  ¬∃ (seq : CellSequence), validSequence seq ∧ seq.map (λ c => c) = arrangementD := by
  sorry


end no_valid_sequence_for_arrangement_D_l1345_134581


namespace plumber_pipe_cost_l1345_134545

/-- The total cost of copper and plastic pipe given specific quantities and prices -/
theorem plumber_pipe_cost (copper_length : ℕ) (plastic_length : ℕ) 
  (copper_price : ℕ) (plastic_price : ℕ) : 
  copper_length = 10 → 
  plastic_length = 15 → 
  copper_price = 5 → 
  plastic_price = 3 → 
  copper_length * copper_price + plastic_length * plastic_price = 95 := by
  sorry

#check plumber_pipe_cost

end plumber_pipe_cost_l1345_134545


namespace regular_octagon_interior_angle_is_135_l1345_134595

/-- The measure of one interior angle of a regular octagon in degrees -/
def regular_octagon_interior_angle : ℝ := 135

/-- Theorem: The measure of one interior angle of a regular octagon is 135 degrees -/
theorem regular_octagon_interior_angle_is_135 :
  regular_octagon_interior_angle = 135 := by
  sorry

end regular_octagon_interior_angle_is_135_l1345_134595


namespace ice_harvest_theorem_l1345_134539

/-- Represents a team that harvests ice blocks -/
inductive Team
| A
| B
| C

/-- Represents the proportion of total ice harvested by each team -/
def teamProportion (t : Team) : ℝ :=
  match t with
  | Team.A => 0.3
  | Team.B => 0.3
  | Team.C => 0.4

/-- Represents the utilization rate of ice harvested by each team -/
def utilizationRate (t : Team) : ℝ :=
  match t with
  | Team.A => 0.8
  | Team.B => 0.75
  | Team.C => 0.6

/-- The number of random draws -/
def numDraws : ℕ := 3

/-- Theorem stating the expectation of Team C's blocks being selected and the probability of a usable block being from Team B -/
theorem ice_harvest_theorem :
  let p := teamProportion Team.C
  let expectation := p * numDraws
  let probUsableB := (teamProportion Team.B * utilizationRate Team.B) /
    (teamProportion Team.A * utilizationRate Team.A +
     teamProportion Team.B * utilizationRate Team.B +
     teamProportion Team.C * utilizationRate Team.C)
  expectation = 6/5 ∧ probUsableB = 15/47 := by
  sorry


end ice_harvest_theorem_l1345_134539


namespace log_comparison_l1345_134516

theorem log_comparison : Real.log 4 / Real.log 3 > Real.log 5 / Real.log 4 := by sorry

end log_comparison_l1345_134516


namespace inverse_square_function_l1345_134558

/-- A function that varies inversely as the square of its input -/
noncomputable def f (y : ℝ) : ℝ := 
  9 / (y * y)

/-- Theorem stating that if f(y) = 1 for some y and f(2) = 2.25, then f(3) = 1 -/
theorem inverse_square_function (h1 : ∃ y, f y = 1) (h2 : f 2 = 2.25) : f 3 = 1 := by
  sorry

end inverse_square_function_l1345_134558


namespace cricket_team_throwers_l1345_134514

/-- Represents a cricket team with throwers and non-throwers -/
structure CricketTeam where
  total_players : ℕ
  throwers : ℕ
  right_handed : ℕ
  left_handed : ℕ

/-- Conditions for the cricket team problem -/
def valid_cricket_team (team : CricketTeam) : Prop :=
  team.total_players = 67 ∧
  team.throwers + team.right_handed + team.left_handed = team.total_players ∧
  team.right_handed + team.throwers = 57 ∧
  3 * team.left_handed = 2 * team.right_handed

theorem cricket_team_throwers (team : CricketTeam) 
  (h : valid_cricket_team team) : team.throwers = 37 := by
  sorry

end cricket_team_throwers_l1345_134514


namespace largest_number_is_312_base_4_l1345_134576

def base_to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.reverse.enum.foldl (fun acc (i, d) => acc + d * base ^ i) 0

theorem largest_number_is_312_base_4 :
  let binary := [1, 1, 1, 1, 1]
  let ternary := [1, 2, 2, 1]
  let quaternary := [3, 1, 2]
  let octal := [5, 6]
  
  (base_to_decimal quaternary 4) = 54 ∧
  (base_to_decimal quaternary 4) > (base_to_decimal binary 2) ∧
  (base_to_decimal quaternary 4) > (base_to_decimal ternary 3) ∧
  (base_to_decimal quaternary 4) > (base_to_decimal octal 8) :=
by
  sorry

end largest_number_is_312_base_4_l1345_134576


namespace midpoint_coordinate_product_l1345_134528

/-- The product of the coordinates of the midpoint of a line segment
    with endpoints (5, -2) and (-3, 6) is equal to 2. -/
theorem midpoint_coordinate_product : 
  let x₁ : ℝ := 5
  let y₁ : ℝ := -2
  let x₂ : ℝ := -3
  let y₂ : ℝ := 6
  let midpoint_x : ℝ := (x₁ + x₂) / 2
  let midpoint_y : ℝ := (y₁ + y₂) / 2
  midpoint_x * midpoint_y = 2 := by
  sorry

end midpoint_coordinate_product_l1345_134528


namespace price_increase_percentage_l1345_134530

theorem price_increase_percentage (old_price new_price : ℝ) (h1 : old_price = 300) (h2 : new_price = 330) :
  (new_price - old_price) / old_price * 100 = 10 := by
  sorry

end price_increase_percentage_l1345_134530


namespace nancy_math_problems_l1345_134569

/-- The number of math problems Nancy had to solve -/
def math_problems : ℝ := 17.0

/-- The number of spelling problems Nancy had to solve -/
def spelling_problems : ℝ := 15.0

/-- The number of problems Nancy can finish in an hour -/
def problems_per_hour : ℝ := 8.0

/-- The number of hours it took Nancy to finish all problems -/
def total_hours : ℝ := 4.0

/-- Theorem stating that the number of math problems Nancy had is 17.0 -/
theorem nancy_math_problems : 
  math_problems = 
    problems_per_hour * total_hours - spelling_problems :=
by sorry

end nancy_math_problems_l1345_134569


namespace expression_equals_14_l1345_134506

theorem expression_equals_14 (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h_sum : x + y + z = 0) (h_prod : x*y + x*z + y*z ≠ 0) :
  (x^7 + y^7 + z^7) / (x*y*z*(x^2 + y^2 + z^2)) = 14 := by
  sorry

end expression_equals_14_l1345_134506


namespace cubic_expansion_sum_l1345_134599

theorem cubic_expansion_sum (a a₁ a₂ a₃ : ℝ) 
  (h : ∀ x : ℝ, x^3 = a + a₁*(x-2) + a₂*(x-2)^2 + a₃*(x-2)^3) : 
  a₁ + a₂ + a₃ = 19 := by
sorry

end cubic_expansion_sum_l1345_134599


namespace strawberry_supply_theorem_l1345_134572

/-- Represents the weekly strawberry requirements for each bakery -/
structure BakeryRequirements where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Calculates the total number of sacks needed for all bakeries over a given period -/
def totalSacks (req : BakeryRequirements) (weeks : ℕ) : ℕ :=
  (req.first + req.second + req.third) * weeks

/-- The problem statement -/
theorem strawberry_supply_theorem (req : BakeryRequirements) (weeks : ℕ) 
    (h1 : req.first = 2)
    (h2 : req.second = 4)
    (h3 : req.third = 12)
    (h4 : weeks = 4) :
  totalSacks req weeks = 72 := by
  sorry

#check strawberry_supply_theorem

end strawberry_supply_theorem_l1345_134572


namespace probability_same_length_is_33_105_l1345_134547

/-- The number of sides in a regular hexagon -/
def num_sides : ℕ := 6

/-- The number of diagonals in a regular hexagon -/
def num_diagonals : ℕ := 9

/-- The number of longer diagonals in a regular hexagon -/
def num_longer_diagonals : ℕ := 6

/-- The number of shorter diagonals in a regular hexagon -/
def num_shorter_diagonals : ℕ := 3

/-- The set of all sides and diagonals of a regular hexagon -/
def S : Finset ℕ := Finset.range (num_sides + num_diagonals)

/-- The probability of selecting two segments of the same length from S -/
def probability_same_length : ℚ :=
  (Nat.choose num_sides 2 + Nat.choose num_longer_diagonals 2 + Nat.choose num_shorter_diagonals 2) /
  Nat.choose S.card 2

theorem probability_same_length_is_33_105 :
  probability_same_length = 33 / 105 := by sorry

end probability_same_length_is_33_105_l1345_134547


namespace problem_solution_l1345_134543

def f (a x : ℝ) : ℝ := 3 * |x - a| + |3 * x + 1|

def g (x : ℝ) : ℝ := |4 * x - 1| - |x + 2|

theorem problem_solution :
  (∀ x : ℝ, g x < 6 ↔ -7/5 < x ∧ x < 3) ∧
  (∃ x₁ x₂ : ℝ, f a x₁ = -g x₂) → -13/12 ≤ a ∧ a ≤ 5/12 :=
sorry

end problem_solution_l1345_134543


namespace last_digit_of_one_over_two_to_ten_l1345_134586

theorem last_digit_of_one_over_two_to_ten (n : ℕ) : 
  n = 10 → (1 : ℚ) / (2^n : ℚ) * 10^n % 10 = 5 := by
  sorry

end last_digit_of_one_over_two_to_ten_l1345_134586


namespace solution_to_equation_l1345_134588

theorem solution_to_equation :
  ∃! (x y : ℝ), x^2 + (1 - y)^2 + (x - y)^2 = 1/3 ∧ x = 1/3 ∧ y = 2/3 := by
  sorry

end solution_to_equation_l1345_134588


namespace joyce_gave_three_oranges_l1345_134593

/-- The number of oranges Joyce gave to Clarence -/
def oranges_from_joyce (initial_oranges final_oranges : ℕ) : ℕ :=
  final_oranges - initial_oranges

theorem joyce_gave_three_oranges :
  oranges_from_joyce 5 8 = 3 := by
  sorry

end joyce_gave_three_oranges_l1345_134593


namespace raft_problem_l1345_134565

/-- The number of people who can fit on a raft under specific conditions -/
def raft_capacity (capacity_without_jackets : ℕ) (capacity_reduction : ℕ) (people_needing_jackets : ℕ) : ℕ :=
  let capacity_with_jackets := capacity_without_jackets - capacity_reduction
  min capacity_with_jackets (people_needing_jackets + (capacity_with_jackets - people_needing_jackets))

/-- Theorem stating that under the given conditions, 14 people can fit on the raft -/
theorem raft_problem : raft_capacity 21 7 8 = 14 := by
  sorry

end raft_problem_l1345_134565


namespace sum_bottle_caps_l1345_134529

/-- The number of bottle caps for each child -/
def bottle_caps : Fin 9 → ℕ
  | ⟨0, _⟩ => 5
  | ⟨1, _⟩ => 8
  | ⟨2, _⟩ => 12
  | ⟨3, _⟩ => 7
  | ⟨4, _⟩ => 9
  | ⟨5, _⟩ => 10
  | ⟨6, _⟩ => 15
  | ⟨7, _⟩ => 4
  | ⟨8, _⟩ => 11
  | ⟨n+9, h⟩ => absurd h (Nat.not_lt_of_ge (Nat.le_add_left 9 n))

/-- The theorem stating that the sum of bottle caps is 81 -/
theorem sum_bottle_caps : (Finset.univ.sum bottle_caps) = 81 := by
  sorry

end sum_bottle_caps_l1345_134529


namespace correct_precipitation_forecast_interpretation_l1345_134521

/-- Represents the possible interpretations of a precipitation forecast --/
inductive PrecipitationForecastInterpretation
  | RainDuration
  | AreaCoverage
  | Probability
  | NoMeaningfulForecast

/-- Represents a precipitation forecast --/
structure PrecipitationForecast where
  probability : ℝ
  interpretation : PrecipitationForecastInterpretation

/-- Asserts that a given interpretation is correct for a precipitation forecast --/
def is_correct_interpretation (forecast : PrecipitationForecast) : Prop :=
  forecast.interpretation = PrecipitationForecastInterpretation.Probability

/-- Theorem: Given an 80% precipitation forecast, the correct interpretation is that there's an 80% chance of rain --/
theorem correct_precipitation_forecast_interpretation 
  (forecast : PrecipitationForecast) 
  (h : forecast.probability = 0.8) :
  is_correct_interpretation forecast :=
sorry

end correct_precipitation_forecast_interpretation_l1345_134521


namespace multiple_of_one_third_equals_two_ninths_l1345_134544

theorem multiple_of_one_third_equals_two_ninths :
  ∃ x : ℚ, x * (1/3 : ℚ) = 2/9 ∧ x = 2/3 := by sorry

end multiple_of_one_third_equals_two_ninths_l1345_134544


namespace quadratic_roots_range_l1345_134540

theorem quadratic_roots_range (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + m*x₁ + (m + 3) = 0 ∧ x₂^2 + m*x₂ + (m + 3) = 0) ↔
  m < -2 ∨ m > 6 :=
sorry

end quadratic_roots_range_l1345_134540


namespace unique_equal_sums_l1345_134546

def arithmetic_sum (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * a₁ + (n - 1 : ℚ) * d)

theorem unique_equal_sums : ∃! (n : ℕ), n > 0 ∧ 
  arithmetic_sum 3 7 n = arithmetic_sum 5 3 n := by sorry

end unique_equal_sums_l1345_134546


namespace min_even_integers_l1345_134584

theorem min_even_integers (a b c d e f : ℤ) : 
  a + b = 30 →
  a + b + c + d = 50 →
  a + b + c + d + e + f = 70 →
  ∃ (x y z w u v : ℤ), 
    x + y = 30 ∧
    x + y + z + w = 50 ∧
    x + y + z + w + u + v = 70 ∧
    Even x ∧ Even y ∧ Even z ∧ Even w ∧ Even u ∧ Even v :=
by sorry

end min_even_integers_l1345_134584


namespace square_roots_problem_l1345_134549

theorem square_roots_problem (a : ℝ) : 
  (a + 3) ^ 2 = (2 * a - 6) ^ 2 → (a + 3) ^ 2 = 16 := by
  sorry

end square_roots_problem_l1345_134549


namespace stone_slab_length_l1345_134538

/-- Given a total floor area covered by equal-sized square stone slabs,
    calculate the length of each slab in centimeters. -/
theorem stone_slab_length
  (total_area : ℝ)
  (num_slabs : ℕ)
  (h_area : total_area = 67.5)
  (h_num : num_slabs = 30)
  : ∃ (slab_length : ℝ),
    slab_length = 150 ∧
    slab_length^2 * num_slabs = total_area * 10000 := by
  sorry

#check stone_slab_length

end stone_slab_length_l1345_134538


namespace largest_number_of_cubic_roots_l1345_134548

theorem largest_number_of_cubic_roots (p q r : ℝ) 
  (sum_eq : p + q + r = 3)
  (sum_prod_eq : p * q + p * r + q * r = -6)
  (prod_eq : p * q * r = -8) :
  max p (max q r) = (1 + Real.sqrt 17) / 2 := by
  sorry

end largest_number_of_cubic_roots_l1345_134548


namespace meal_with_tip_l1345_134504

/-- Calculates the total amount spent on a meal including tip -/
theorem meal_with_tip (lunch_cost : ℝ) (tip_percentage : ℝ) : 
  lunch_cost = 50.50 → tip_percentage = 20 → lunch_cost * (1 + tip_percentage / 100) = 60.60 := by
  sorry

end meal_with_tip_l1345_134504

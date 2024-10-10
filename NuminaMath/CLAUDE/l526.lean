import Mathlib

namespace rectangle_ratio_l526_52665

theorem rectangle_ratio (area : ℝ) (length : ℝ) (breadth : ℝ) :
  area = 6075 →
  length = 135 →
  area = length * breadth →
  length / breadth = 3 := by
sorry

end rectangle_ratio_l526_52665


namespace intersection_implies_k_range_l526_52689

/-- The line equation kx - y - k - 1 = 0 intersects the line segment MN,
    where M(2,1) and N(3,2) are the endpoints of the segment. -/
def intersects_segment (k : ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧
    k * (2 + t) - (1 + t) - k - 1 = 0

/-- The theorem states that if the line intersects the segment MN,
    then k is in the range [3/2, 2]. -/
theorem intersection_implies_k_range :
  ∀ k : ℝ, intersects_segment k → 3/2 ≤ k ∧ k ≤ 2 :=
by sorry

end intersection_implies_k_range_l526_52689


namespace factorization_equality_l526_52673

theorem factorization_equality (a : ℝ) : (a + 1) * (a + 2) + 1/4 = (a + 3/2)^2 := by
  sorry

end factorization_equality_l526_52673


namespace sum_first_six_primes_mod_seventh_l526_52623

def first_six_primes : List Nat := [2, 3, 5, 7, 11, 13]
def seventh_prime : Nat := 17

theorem sum_first_six_primes_mod_seventh : (first_six_primes.sum % seventh_prime) = 7 := by
  sorry

end sum_first_six_primes_mod_seventh_l526_52623


namespace function_equation_1_bijective_function_equation_2_neither_function_equation_3_neither_function_equation_4_neither_l526_52651

-- 1. f(x+f(y))=2f(x)+y is bijective
theorem function_equation_1_bijective (f : ℝ → ℝ) :
  (∀ x y, f (x + f y) = 2 * f x + y) → Function.Bijective f :=
sorry

-- 2. f(f(x))=0 is neither injective nor surjective
theorem function_equation_2_neither (f : ℝ → ℝ) :
  (∀ x, f (f x) = 0) → ¬(Function.Injective f ∨ Function.Surjective f) :=
sorry

-- 3. f(f(x))=sin(x) is neither injective nor surjective
theorem function_equation_3_neither (f : ℝ → ℝ) :
  (∀ x, f (f x) = Real.sin x) → ¬(Function.Injective f ∨ Function.Surjective f) :=
sorry

-- 4. f(x+y)=f(x)f(y) is neither injective nor surjective
theorem function_equation_4_neither (f : ℝ → ℝ) :
  (∀ x y, f (x + y) = f x * f y) → ¬(Function.Injective f ∨ Function.Surjective f) :=
sorry

end function_equation_1_bijective_function_equation_2_neither_function_equation_3_neither_function_equation_4_neither_l526_52651


namespace largest_reciprocal_l526_52682

def numbers : List ℚ := [1/4, 3/7, 2, 10, 2023]

def reciprocal (x : ℚ) : ℚ := 1 / x

theorem largest_reciprocal :
  ∀ n ∈ numbers, reciprocal (1/4) ≥ reciprocal n :=
by sorry

end largest_reciprocal_l526_52682


namespace investment_value_after_six_weeks_l526_52687

/-- Calculates the final investment value after six weeks of changes and compound interest --/
def calculate_investment (initial_investment : ℝ) (week1_gain : ℝ) (week1_add : ℝ)
  (week2_gain : ℝ) (week2_withdraw : ℝ) (week3_loss : ℝ) (week4_gain : ℝ) (week4_add : ℝ)
  (week5_gain : ℝ) (week6_loss : ℝ) (week6_withdraw : ℝ) (weekly_interest : ℝ) : ℝ :=
  let week1 := (initial_investment * (1 + week1_gain) * (1 + weekly_interest)) + week1_add
  let week2 := (week1 * (1 + week2_gain) * (1 + weekly_interest)) - week2_withdraw
  let week3 := week2 * (1 - week3_loss) * (1 + weekly_interest)
  let week4 := (week3 * (1 + week4_gain) * (1 + weekly_interest)) + week4_add
  let week5 := week4 * (1 + week5_gain) * (1 + weekly_interest)
  let week6 := (week5 * (1 - week6_loss) * (1 + weekly_interest)) - week6_withdraw
  week6

/-- The final investment value after six weeks is approximately $819.74 --/
theorem investment_value_after_six_weeks :
  ∃ ε > 0, |calculate_investment 400 0.25 200 0.50 150 0.10 0.20 100 0.05 0.15 250 0.02 - 819.74| < ε :=
sorry

end investment_value_after_six_weeks_l526_52687


namespace line_parallel_perpendicular_implies_planes_perpendicular_l526_52600

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (perpendicularPlanes : Plane → Plane → Prop)

-- State the theorem
theorem line_parallel_perpendicular_implies_planes_perpendicular
  (l : Line) (α β : Plane) :
  parallel l α → perpendicular l β → perpendicularPlanes α β :=
by sorry

end line_parallel_perpendicular_implies_planes_perpendicular_l526_52600


namespace perfect_apples_l526_52675

/-- Given a batch of apples, calculate the number of perfect apples -/
theorem perfect_apples (total : ℕ) (too_small : ℚ) (not_ripe : ℚ) 
  (h1 : total = 30) 
  (h2 : too_small = 1/6) 
  (h3 : not_ripe = 1/3) : 
  ↑total * (1 - (too_small + not_ripe)) = 15 := by
  sorry

end perfect_apples_l526_52675


namespace taras_rowing_speed_l526_52635

/-- Tara's rowing problem -/
theorem taras_rowing_speed 
  (downstream_distance : ℝ) 
  (upstream_distance : ℝ) 
  (time : ℝ) 
  (current_speed : ℝ) 
  (h1 : downstream_distance = 20) 
  (h2 : upstream_distance = 4) 
  (h3 : time = 2) 
  (h4 : current_speed = 2) :
  ∃ v : ℝ, 
    v + current_speed = downstream_distance / time ∧ 
    v - current_speed = upstream_distance / time ∧ 
    v = 8 := by
sorry

end taras_rowing_speed_l526_52635


namespace delta_max_success_ratio_l526_52614

/-- Represents a participant's score in a math challenge --/
structure Score where
  points_scored : ℚ
  points_attempted : ℚ

/-- Calculates the success ratio of a score --/
def successRatio (s : Score) : ℚ := s.points_scored / s.points_attempted

/-- Represents the scores of a participant over two days --/
structure TwoDayScore where
  day1 : Score
  day2 : Score

/-- Calculates the overall success ratio for a two-day score --/
def overallSuccessRatio (s : TwoDayScore) : ℚ :=
  (s.day1.points_scored + s.day2.points_scored) / (s.day1.points_attempted + s.day2.points_attempted)

/-- Gamma's score for each day --/
def gammaScore : Score := { points_scored := 180, points_attempted := 300 }

/-- Delta's maximum possible two-day score --/
def deltaMaxScore : TwoDayScore := {
  day1 := { points_scored := 179, points_attempted := 299 },
  day2 := { points_scored := 180, points_attempted := 301 }
}

theorem delta_max_success_ratio :
  (∀ s : TwoDayScore,
    s.day1.points_attempted + s.day2.points_attempted = 600 ∧
    successRatio s.day1 < successRatio gammaScore ∧
    successRatio s.day2 < successRatio gammaScore) →
  (∀ s : TwoDayScore,
    s.day1.points_attempted + s.day2.points_attempted = 600 ∧
    successRatio s.day1 < successRatio gammaScore ∧
    successRatio s.day2 < successRatio gammaScore →
    overallSuccessRatio s ≤ overallSuccessRatio deltaMaxScore) :=
by sorry

end delta_max_success_ratio_l526_52614


namespace set_inclusion_equivalence_l526_52638

-- Define set A
def A (a : ℝ) : Set ℝ := { x | a - 1 ≤ x ∧ x ≤ a + 2 }

-- Define set B
def B : Set ℝ := { x | |x - 4| < 1 }

-- Theorem statement
theorem set_inclusion_equivalence (a : ℝ) : A a ⊇ B ↔ 3 ≤ a ∧ a ≤ 4 := by
  sorry

end set_inclusion_equivalence_l526_52638


namespace capital_calculation_l526_52661

/-- Calculates the capital of a business partner who joined later --/
def calculate_capital (x_capital y_capital : ℕ) (z_profit total_profit : ℕ) (z_join_month : ℕ) : ℕ :=
  let x_share := x_capital * 12
  let y_share := y_capital * 12
  let z_months := 12 - z_join_month
  let total_ratio := x_share + y_share
  ((z_profit * total_ratio) / (total_profit - z_profit)) / z_months

theorem capital_calculation (x_capital y_capital : ℕ) (z_profit total_profit : ℕ) (z_join_month : ℕ) :
  x_capital = 20000 →
  y_capital = 25000 →
  z_profit = 14000 →
  total_profit = 50000 →
  z_join_month = 5 →
  calculate_capital x_capital y_capital z_profit total_profit z_join_month = 30000 := by
  sorry

#eval calculate_capital 20000 25000 14000 50000 5

end capital_calculation_l526_52661


namespace min_value_of_sum_of_roots_l526_52628

theorem min_value_of_sum_of_roots (x : ℝ) :
  Real.sqrt (x^2 + 4*x + 5) + Real.sqrt (x^2 - 8*x + 25) ≥ 2 * Real.sqrt 13 := by
  sorry

end min_value_of_sum_of_roots_l526_52628


namespace and_sufficient_not_necessary_for_or_l526_52690

theorem and_sufficient_not_necessary_for_or (p q : Prop) :
  (∀ (p q : Prop), p ∧ q → p ∨ q) ∧
  (∃ (p q : Prop), p ∨ q ∧ ¬(p ∧ q)) :=
sorry

end and_sufficient_not_necessary_for_or_l526_52690


namespace appetizer_cost_is_six_l526_52671

/-- The cost of dinner for a group, including main meals, appetizers, tip, and rush order fee. --/
def dinner_cost (main_meal_cost : ℝ) (num_people : ℕ) (num_appetizers : ℕ) (appetizer_cost : ℝ) (tip_rate : ℝ) (rush_fee : ℝ) : ℝ :=
  let subtotal := main_meal_cost * num_people + appetizer_cost * num_appetizers
  subtotal + tip_rate * subtotal + rush_fee

/-- Theorem stating that the appetizer cost is $6.00 given the specified conditions. --/
theorem appetizer_cost_is_six :
  ∃ (appetizer_cost : ℝ),
    dinner_cost 12 4 2 appetizer_cost 0.2 5 = 77 ∧
    appetizer_cost = 6 := by
  sorry

end appetizer_cost_is_six_l526_52671


namespace contrapositive_equivalence_l526_52672

theorem contrapositive_equivalence :
  (∀ x : ℝ, x^2 = 1 → (x = 1 ∨ x = -1)) ↔
  (∀ x : ℝ, (x ≠ 1 ∧ x ≠ -1) → x^2 ≠ 1) :=
by sorry

end contrapositive_equivalence_l526_52672


namespace cyclic_inequality_l526_52668

theorem cyclic_inequality (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  a^4 * b + b^4 * c + c^4 * d + d^4 * a ≥ a * b * c * d * (a + b + c + d) := by
  sorry

end cyclic_inequality_l526_52668


namespace complement_A_intersect_B_l526_52615

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x < 0}
def B : Set ℝ := {-2, -1, 0, 1, 2}

theorem complement_A_intersect_B :
  (Set.compl A) ∩ B = {0, 1, 2} := by sorry

end complement_A_intersect_B_l526_52615


namespace only_vertical_angles_true_l526_52696

-- Define the propositions
def vertical_angles_equal : Prop := ∀ (α β : ℝ), α = β → α = β
def corresponding_angles_equal : Prop := ∀ (α β : ℝ), α = β
def product_one_implies_one : Prop := ∀ (a b : ℝ), a * b = 1 → a = 1 ∨ b = 1
def square_root_of_four : Prop := ∀ (x : ℝ), x^2 = 4 → x = 2

-- Theorem stating that only vertical_angles_equal is true
theorem only_vertical_angles_true : 
  vertical_angles_equal ∧ 
  ¬corresponding_angles_equal ∧ 
  ¬product_one_implies_one ∧ 
  ¬square_root_of_four :=
sorry

end only_vertical_angles_true_l526_52696


namespace digits_difference_in_base_d_l526_52601

/-- Given two digits A and B in base d > 7, such that AB + AA = 172 in base d, prove A - B = 5 in base d -/
theorem digits_difference_in_base_d (d A B : ℕ) : 
  d > 7 →
  A < d →
  B < d →
  (A * d + B) + (A * d + A) = 1 * d^2 + 7 * d + 2 →
  A - B = 5 := by
sorry

end digits_difference_in_base_d_l526_52601


namespace parabola_shift_l526_52632

/-- The equation of a parabola after horizontal and vertical shifts -/
def shifted_parabola (a b c : ℝ) (x y : ℝ) : Prop :=
  y = a * (x - b)^2 + c

theorem parabola_shift :
  ∀ (x y : ℝ),
  (y = 3 * x^2) →  -- Original parabola
  (shifted_parabola 3 1 (-2) x y)  -- Shifted parabola
  := by sorry

end parabola_shift_l526_52632


namespace gcd_linear_combination_l526_52603

theorem gcd_linear_combination (a b : ℤ) : 
  Int.gcd (5*a + 3*b) (13*a + 8*b) = Int.gcd a b := by sorry

end gcd_linear_combination_l526_52603


namespace expression_simplification_find_k_value_l526_52642

-- Problem 1: Simplify the expression
theorem expression_simplification (x : ℝ) :
  (2*x + 1)^2 - (2*x + 1)*(2*x - 1) + (x + 1)*(x - 3) = x^2 + 2*x - 1 :=
by sorry

-- Problem 2: Find the value of k
theorem find_k_value (x y k : ℝ) 
  (eq1 : x + y = 1)
  (eq2 : k*x + (k-1)*y = 7)
  (eq3 : 3*x - 2*y = 5) :
  k = 33/5 :=
by sorry

end expression_simplification_find_k_value_l526_52642


namespace total_items_eq_1900_l526_52694

/-- The number of rows of pencils and crayons. -/
def num_rows : ℕ := 19

/-- The number of pencils in each row. -/
def pencils_per_row : ℕ := 57

/-- The number of crayons in each row. -/
def crayons_per_row : ℕ := 43

/-- The total number of pencils and crayons. -/
def total_items : ℕ := num_rows * (pencils_per_row + crayons_per_row)

theorem total_items_eq_1900 : total_items = 1900 := by
  sorry

end total_items_eq_1900_l526_52694


namespace base_3_312_property_l526_52606

def base_3_representation (n : ℕ) : List ℕ :=
  sorry

def count_digit (l : List ℕ) (d : ℕ) : ℕ :=
  sorry

theorem base_3_312_property :
  let base_3_312 := base_3_representation 312
  let x := count_digit base_3_312 0
  let y := count_digit base_3_312 1
  let z := count_digit base_3_312 2
  z - y + x = 2 := by sorry

end base_3_312_property_l526_52606


namespace mangoes_in_basket_l526_52618

/-- The number of mangoes in a basket of fruits -/
def mangoes_count (total_fruits : ℕ) (pears : ℕ) (pawpaws : ℕ) (lemons : ℕ) : ℕ :=
  total_fruits - (pears + pawpaws + lemons + lemons)

theorem mangoes_in_basket :
  mangoes_count 58 10 12 9 = 18 :=
by sorry

end mangoes_in_basket_l526_52618


namespace allison_june_uploads_l526_52616

/-- Calculates the total number of video hours uploaded by Allison in June -/
def total_video_hours (initial_rate : ℕ) (days_in_june : ℕ) (initial_period : ℕ) : ℕ :=
  let doubled_rate := 2 * initial_rate
  let remaining_period := days_in_june - initial_period
  initial_rate * initial_period + doubled_rate * remaining_period

/-- Theorem stating that Allison's total uploaded video hours in June is 450 -/
theorem allison_june_uploads :
  total_video_hours 10 30 15 = 450 := by
  sorry

end allison_june_uploads_l526_52616


namespace expand_binomials_l526_52612

theorem expand_binomials (x : ℝ) : (3 * x + 4) * (2 * x - 6) = 6 * x^2 - 10 * x - 24 := by
  sorry

end expand_binomials_l526_52612


namespace range_of_m_l526_52656

theorem range_of_m (x y m : ℝ) : 
  x > 0 → y > 0 → x + y = 3 → 
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 3 → 
    (4 / (x + 1)) + (16 / y) > m^2 - 3*m + 5) → 
  -1 < m ∧ m < 4 := by
sorry

end range_of_m_l526_52656


namespace minimum_cost_is_2200_l526_52677

/-- Represents the transportation problem for washing machines -/
structure TransportationProblem where
  totalWashingMachines : ℕ
  typeATrucks : ℕ
  typeBTrucks : ℕ
  typeACapacity : ℕ
  typeBCapacity : ℕ
  typeACost : ℕ
  typeBCost : ℕ

/-- Calculates the minimum transportation cost for the given problem -/
def minimumTransportationCost (p : TransportationProblem) : ℕ :=
  sorry

/-- The main theorem stating that the minimum transportation cost is 2200 yuan -/
theorem minimum_cost_is_2200 :
  let p : TransportationProblem := {
    totalWashingMachines := 100,
    typeATrucks := 4,
    typeBTrucks := 8,
    typeACapacity := 20,
    typeBCapacity := 10,
    typeACost := 400,
    typeBCost := 300
  }
  minimumTransportationCost p = 2200 := by
  sorry

end minimum_cost_is_2200_l526_52677


namespace amount_increase_l526_52609

theorem amount_increase (initial_amount : ℚ) : 
  (initial_amount * (9/8) * (9/8) = 4050) → initial_amount = 3200 := by
  sorry

end amount_increase_l526_52609


namespace rotation_equivalence_l526_52666

def clockwise_rotation : ℝ := 480
def counterclockwise_rotation : ℝ := 240

theorem rotation_equivalence :
  ∀ y : ℝ,
  y < 360 →
  (clockwise_rotation % 360 = (360 - y) % 360) →
  y = counterclockwise_rotation :=
by
  sorry

end rotation_equivalence_l526_52666


namespace largest_two_digit_remainder_two_l526_52698

theorem largest_two_digit_remainder_two : ∃ n : ℕ, 
  (n ≥ 10 ∧ n ≤ 99) ∧ 
  n % 13 = 2 ∧ 
  (∀ m : ℕ, (m ≥ 10 ∧ m ≤ 99) ∧ m % 13 = 2 → m ≤ n) ∧
  n = 93 := by
sorry

end largest_two_digit_remainder_two_l526_52698


namespace least_multiple_ending_zero_l526_52649

theorem least_multiple_ending_zero : ∃ n : ℕ, 
  (∀ k : ℕ, k ≤ 10 → k > 0 → n % k = 0) ∧ 
  (n % 10 = 0) ∧
  (∀ m : ℕ, m < n → (∃ k : ℕ, k ≤ 10 ∧ k > 0 ∧ m % k ≠ 0) ∨ m % 10 ≠ 0) ∧
  n = 2520 :=
by sorry

end least_multiple_ending_zero_l526_52649


namespace car_pricing_problem_l526_52676

theorem car_pricing_problem (X : ℝ) (A : ℝ) : 
  X > 0 →
  0.8 * X * (1 + A / 100) = 1.2 * X →
  A = 50 := by
sorry

end car_pricing_problem_l526_52676


namespace max_altitude_product_l526_52678

/-- 
Given a triangle ABC with base AB = 1 and altitude from C of length h,
this theorem states the maximum product of the three altitudes and the
triangle configuration that achieves it.
-/
theorem max_altitude_product (h : ℝ) (h_pos : h > 0) :
  let max_product := if h ≤ 1/2 then h^2 else h^3 / (h^2 + 1/4)
  let optimal_triangle := if h ≤ 1/2 then "right triangle at C" else "isosceles triangle with AC = BC"
  ∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 →
    (a * b * h ≤ max_product ∧
    (a * b * h = max_product ↔ 
      (h ≤ 1/2 ∧ c^2 = a^2 + b^2) ∨ 
      (h > 1/2 ∧ a = b))) :=
by sorry


end max_altitude_product_l526_52678


namespace plane_graph_is_bipartite_plane_regions_two_colorable_l526_52680

/-- A graph representing regions formed by lines dividing a plane -/
structure PlaneGraph where
  V : Type* -- Vertices (regions)
  E : V → V → Prop -- Edges (neighboring regions)

/-- Definition of a bipartite graph -/
def IsBipartite (G : PlaneGraph) : Prop :=
  ∃ (A B : Set G.V), (∀ v, v ∈ A ∨ v ∈ B) ∧ 
    (∀ u v, G.E u v → (u ∈ A ∧ v ∈ B) ∨ (u ∈ B ∧ v ∈ A))

/-- Theorem: The graph representing regions formed by lines dividing a plane is bipartite -/
theorem plane_graph_is_bipartite (G : PlaneGraph) : IsBipartite G := by
  sorry

/-- Corollary: Regions formed by lines dividing a plane can be colored with two colors -/
theorem plane_regions_two_colorable (G : PlaneGraph) : 
  ∃ (color : G.V → Bool), ∀ u v, G.E u v → color u ≠ color v := by
  sorry

end plane_graph_is_bipartite_plane_regions_two_colorable_l526_52680


namespace peach_basket_problem_l526_52622

theorem peach_basket_problem (baskets : Nat) (total_peaches : Nat) (green_excess : Nat) :
  baskets = 2 →
  green_excess = 2 →
  total_peaches = 12 →
  ∃ red_peaches : Nat, red_peaches * baskets + (red_peaches + green_excess) * baskets = total_peaches ∧ red_peaches = 2 :=
by
  sorry

end peach_basket_problem_l526_52622


namespace problem_solution_l526_52681

theorem problem_solution (a b : ℝ) (h1 : a + b = 8) (h2 : a - b = 4) : 
  a^2 - b^2 = 32 ∧ a * b = 12 := by
sorry

end problem_solution_l526_52681


namespace carol_extra_chore_earnings_l526_52659

/-- Proves that given the conditions, Carol earns $1.50 per extra chore -/
theorem carol_extra_chore_earnings
  (weekly_allowance : ℚ)
  (num_weeks : ℕ)
  (total_amount : ℚ)
  (avg_extra_chores : ℚ)
  (h1 : weekly_allowance = 20)
  (h2 : num_weeks = 10)
  (h3 : total_amount = 425)
  (h4 : avg_extra_chores = 15) :
  (total_amount - weekly_allowance * num_weeks) / (avg_extra_chores * num_weeks) = 3/2 :=
sorry

end carol_extra_chore_earnings_l526_52659


namespace sqrt_3_times_sqrt_12_l526_52602

theorem sqrt_3_times_sqrt_12 : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_3_times_sqrt_12_l526_52602


namespace nine_a_value_l526_52692

theorem nine_a_value (a b : ℚ) (eq1 : 8 * a + 3 * b = 0) (eq2 : a = b - 3) : 9 * a = -81 / 11 := by
  sorry

end nine_a_value_l526_52692


namespace square_2007_position_l526_52625

-- Define the possible square positions
inductive SquarePosition
  | ABCD
  | DCBA

-- Define the transformations
def rotate180 (pos : SquarePosition) : SquarePosition :=
  match pos with
  | SquarePosition.ABCD => SquarePosition.DCBA
  | SquarePosition.DCBA => SquarePosition.ABCD

def reflectHorizontal (pos : SquarePosition) : SquarePosition := pos

-- Define the sequence of transformations
def transformSquare (n : Nat) : SquarePosition :=
  if n % 2 = 1 then
    rotate180 SquarePosition.ABCD
  else
    reflectHorizontal (rotate180 SquarePosition.ABCD)

-- State the theorem
theorem square_2007_position :
  transformSquare 2007 = SquarePosition.DCBA := by sorry

end square_2007_position_l526_52625


namespace age_difference_l526_52660

theorem age_difference (A B C : ℕ) (h : C = A - 13) : A + B - (B + C) = 13 := by
  sorry

end age_difference_l526_52660


namespace quadratic_equation_roots_l526_52647

theorem quadratic_equation_roots (m : ℝ) : 
  (∃ z₁ z₂ : ℂ, z₁^2 + 5*z₁ + m = 0 ∧ z₂^2 + 5*z₂ + m = 0 ∧ Complex.abs (z₁ - z₂) = 3) → 
  (m = 4 ∨ m = 17/2) :=
by sorry

end quadratic_equation_roots_l526_52647


namespace solution_set_inequality_l526_52633

theorem solution_set_inequality (x : ℝ) :
  (Set.Iio (1/3 : ℝ)) = {x | Real.sqrt (x^2 - 2*x + 1) > 2*x} := by
  sorry

end solution_set_inequality_l526_52633


namespace box_width_proof_l526_52662

/-- Proves that the width of a box with given dimensions and constraints is 18 cm -/
theorem box_width_proof (length height : ℝ) (cube_volume min_cubes : ℕ) :
  length = 7 →
  height = 3 →
  cube_volume = 9 →
  min_cubes = 42 →
  ∃ width : ℝ,
    width * length * height = min_cubes * cube_volume ∧
    width = 18 := by
  sorry

end box_width_proof_l526_52662


namespace complex_addition_proof_l526_52691

theorem complex_addition_proof : ∃ z : ℂ, 2 * (5 - 3*I) + z = 4 + 11*I :=
by
  use -6 + 17*I
  sorry

end complex_addition_proof_l526_52691


namespace smallest_sum_of_two_distinct_primes_above_70_l526_52697

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem smallest_sum_of_two_distinct_primes_above_70 :
  ∃ (p q : ℕ), 
    is_prime p ∧ 
    is_prime q ∧ 
    p > 70 ∧ 
    q > 70 ∧ 
    p ≠ q ∧ 
    p + q = 144 ∧ 
    (∀ (r s : ℕ), is_prime r → is_prime s → r > 70 → s > 70 → r ≠ s → r + s ≥ 144) :=
by sorry

end smallest_sum_of_two_distinct_primes_above_70_l526_52697


namespace min_value_expression_l526_52617

theorem min_value_expression (a d b c : ℝ) 
  (ha : a ≥ 0) (hd : d ≥ 0) (hb : b > 0) (hc : c > 0) (h_sum : b + c ≥ a + d) :
  ∃ (x y z w : ℝ), x ≥ 0 ∧ y > 0 ∧ z > 0 ∧ w ≥ 0 ∧ y + z ≥ x + w ∧
    ∀ (a' d' b' c' : ℝ), a' ≥ 0 → d' ≥ 0 → b' > 0 → c' > 0 → b' + c' ≥ a' + d' →
      (b' / (c' + d')) + (c' / (a' + b')) ≥ (y / (z + w)) + (z / (x + y)) ∧
      (y / (z + w)) + (z / (x + y)) = Real.sqrt 2 - 1 / 2 :=
by
  sorry


end min_value_expression_l526_52617


namespace onion_weight_problem_l526_52685

/-- Given 40 onions weighing 7.68 kg, and 35 of these onions having an average weight of 190 grams,
    the average weight of the remaining 5 onions is 206 grams. -/
theorem onion_weight_problem (total_weight : Real) (remaining_avg : Real) :
  total_weight = 7.68 →
  remaining_avg = 190 →
  (total_weight * 1000 - 35 * remaining_avg) / 5 = 206 := by
sorry

end onion_weight_problem_l526_52685


namespace principal_calculation_l526_52631

/-- Proves that given specific conditions, the principal amount is 900 --/
theorem principal_calculation (interest_rate : ℚ) (time : ℚ) (final_amount : ℚ) :
  interest_rate = 5 / 100 →
  time = 12 / 5 →
  final_amount = 1008 →
  final_amount = (1 + interest_rate * time) * 900 :=
by sorry

end principal_calculation_l526_52631


namespace cards_given_to_jeff_l526_52605

/-- The number of cards Nell initially had -/
def initial_cards : ℕ := 304

/-- The number of cards Nell has left -/
def remaining_cards : ℕ := 276

/-- The number of cards Nell gave to Jeff -/
def cards_given : ℕ := initial_cards - remaining_cards

theorem cards_given_to_jeff : cards_given = 28 := by
  sorry

end cards_given_to_jeff_l526_52605


namespace complex_angle_pi_third_l526_52629

theorem complex_angle_pi_third (z : ℂ) : 
  z = 1 + Complex.I * Real.sqrt 3 → 
  ∃ (r : ℝ), z = r * Complex.exp (Complex.I * (Real.pi / 3)) :=
by sorry

end complex_angle_pi_third_l526_52629


namespace least_sum_m_n_l526_52610

theorem least_sum_m_n (m n : ℕ+) 
  (h1 : Nat.gcd (m + n) 330 = 1)
  (h2 : ∃ k : ℕ, m^m.val = k * n^n.val)
  (h3 : ¬ ∃ k : ℕ, m = k * n) :
  ∀ p q : ℕ+, 
    (Nat.gcd (p + q) 330 = 1) → 
    (∃ k : ℕ, p^p.val = k * q^q.val) → 
    (¬ ∃ k : ℕ, p = k * q) → 
    m + n ≤ p + q :=
by
  sorry

end least_sum_m_n_l526_52610


namespace select_shoes_count_l526_52657

/-- The number of ways to select 4 shoes from 4 pairs of different shoes,
    with at least 2 shoes forming a pair -/
def select_shoes : ℕ :=
  Nat.choose 8 4 - 16

theorem select_shoes_count : select_shoes = 54 := by
  sorry

end select_shoes_count_l526_52657


namespace michaels_truck_rental_cost_l526_52653

/-- Calculates the total cost of renting a truck -/
def truckRentalCost (rentalFee : ℚ) (chargePerMile : ℚ) (milesDriven : ℕ) : ℚ :=
  rentalFee + chargePerMile * milesDriven

/-- Proves that the total cost for Michael's truck rental is $95.74 -/
theorem michaels_truck_rental_cost :
  truckRentalCost 20.99 0.25 299 = 95.74 := by
  sorry

#eval truckRentalCost 20.99 0.25 299

end michaels_truck_rental_cost_l526_52653


namespace selection_problem_l526_52630

theorem selection_problem (n_teachers : ℕ) (n_students : ℕ) : n_teachers = 4 → n_students = 5 →
  (Nat.choose n_teachers 1 * Nat.choose n_students 2 + 
   Nat.choose n_teachers 2 * Nat.choose n_students 1) = 70 := by
  sorry

end selection_problem_l526_52630


namespace regular_octagon_exterior_angle_regular_octagon_exterior_angle_is_45_l526_52613

/-- The measure of an exterior angle in a regular octagon is 45 degrees. -/
theorem regular_octagon_exterior_angle : ℝ :=
  let n : ℕ := 8  -- number of sides in an octagon
  let interior_angle_sum : ℝ := (n - 2) * 180
  let interior_angle : ℝ := interior_angle_sum / n
  let exterior_angle : ℝ := 180 - interior_angle
  exterior_angle

/-- The measure of an exterior angle in a regular octagon is 45 degrees. -/
theorem regular_octagon_exterior_angle_is_45 :
  regular_octagon_exterior_angle = 45 := by
  sorry

end regular_octagon_exterior_angle_regular_octagon_exterior_angle_is_45_l526_52613


namespace intersecting_rectangles_area_l526_52624

/-- The total shaded area of two intersecting rectangles -/
theorem intersecting_rectangles_area (rect1_width rect1_height rect2_width rect2_height overlap_width overlap_height : ℕ) 
  (h1 : rect1_width = 4 ∧ rect1_height = 12)
  (h2 : rect2_width = 5 ∧ rect2_height = 7)
  (h3 : overlap_width = 4 ∧ overlap_height = 5) :
  rect1_width * rect1_height + rect2_width * rect2_height - overlap_width * overlap_height = 63 :=
by sorry

end intersecting_rectangles_area_l526_52624


namespace divisible_by_2_3_5_7_under_300_l526_52626

theorem divisible_by_2_3_5_7_under_300 : 
  ∃! n : ℕ, n > 0 ∧ n < 300 ∧ 2 ∣ n ∧ 3 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n :=
by sorry

end divisible_by_2_3_5_7_under_300_l526_52626


namespace complex_expression_simplification_l526_52619

theorem complex_expression_simplification :
  ∀ (i : ℂ), i^2 = -1 → 7 * (4 - 2*i) + 4*i * (6 - 3*i) = 40 + 10*i := by
  sorry

end complex_expression_simplification_l526_52619


namespace diagonal_length_after_triangle_removal_l526_52634

/-- The diagonal length of a quadrilateral formed by removing two equal-area right triangles from opposite corners of a square --/
theorem diagonal_length_after_triangle_removal (s : ℝ) (A : ℝ) (h1 : s = 20) (h2 : A = 50) :
  let x := Real.sqrt (2 * A)
  let diagonal := Real.sqrt ((s - x)^2 + (s - x)^2)
  diagonal = 10 * Real.sqrt 2 := by
  sorry

#check diagonal_length_after_triangle_removal

end diagonal_length_after_triangle_removal_l526_52634


namespace min_value_theorem_l526_52652

theorem min_value_theorem (a m n : ℝ) (ha : a > 0) (ha_ne_one : a ≠ 1) (hm : m > 0) (hn : n > 0) 
  (h_fixed_point : a^(2 - 2) = 1) 
  (h_linear : m * 2 + 4 * n = 1) : 
  1 / m + 2 / n ≥ 18 := by
sorry

end min_value_theorem_l526_52652


namespace function_inequality_l526_52644

theorem function_inequality (f : ℝ → ℝ) (a b : ℝ) 
  (h_diff : Differentiable ℝ f) 
  (h_ab : a > b ∧ b > 1) 
  (h_deriv : ∀ x, (x - 1) * deriv f x ≥ 0) : 
  f a + f b ≥ 2 * f 1 := by
  sorry

end function_inequality_l526_52644


namespace sin_transformations_l526_52648

open Real

theorem sin_transformations (x : ℝ) :
  (∀ x, sin (2 * (x - π/6)) = sin (2*x - π/3)) ∧
  (∀ x, sin (2 * (x - π/3)) = sin (2*x - π/3)) ∧
  (∀ x, sin (2 * (x + 5*π/6)) = sin (2*x - π/3)) :=
by sorry

end sin_transformations_l526_52648


namespace total_feet_count_l526_52620

/-- Given a total of 50 animals with 30 hens, prove that the total number of feet is 140. -/
theorem total_feet_count (total_animals : ℕ) (num_hens : ℕ) (hen_feet : ℕ) (cow_feet : ℕ) : 
  total_animals = 50 → 
  num_hens = 30 → 
  hen_feet = 2 → 
  cow_feet = 4 → 
  num_hens * hen_feet + (total_animals - num_hens) * cow_feet = 140 := by
sorry

end total_feet_count_l526_52620


namespace ratio_problem_l526_52667

theorem ratio_problem (p q n : ℝ) (h1 : p / q = 5 / n) (h2 : 2 * p + q = 14) : n = 1 := by
  sorry

end ratio_problem_l526_52667


namespace concentric_circles_ratio_l526_52645

theorem concentric_circles_ratio (r R : ℝ) (h1 : R = 10) 
  (h2 : π * R^2 = 2 * (π * R^2 - π * r^2)) : r = 5 * Real.sqrt 2 := by
  sorry

end concentric_circles_ratio_l526_52645


namespace coordinates_in_new_basis_l526_52683

open LinearAlgebra

variable {𝕜 : Type*} [Field 𝕜]
variable {E : Type*} [AddCommGroup E] [Module 𝕜 E]

/-- Given a vector space E over a field 𝕜, and two bases e and e' of E, 
    prove that the coordinates of a vector x in the new basis e' are {0, 1, -1} -/
theorem coordinates_in_new_basis 
  (e : Basis (Fin 3) 𝕜 E) 
  (e' : Basis (Fin 3) 𝕜 E) 
  (x : E) :
  (∀ i : Fin 3, e' i = 
    if i = 0 then e 0 + 2 • (e 2)
    else if i = 1 then e 1 + e 2
    else -(e 0) - (e 1) - 2 • (e 2)) →
  (x = e 0 + 2 • (e 1) + 3 • (e 2)) →
  (∃ a b c : 𝕜, x = a • (e' 0) + b • (e' 1) + c • (e' 2) ∧ a = 0 ∧ b = 1 ∧ c = -1) :=
by sorry

end coordinates_in_new_basis_l526_52683


namespace parabola_shift_l526_52693

/-- Represents a parabola in the form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally and vertically -/
def shift (p : Parabola) (h : ℝ) (v : ℝ) : Parabola :=
  { a := p.a
    b := -2 * p.a * h + p.b
    c := p.a * h^2 - p.b * h + p.c + v }

theorem parabola_shift :
  let original := Parabola.mk 5 0 0
  let shifted := shift original 2 3
  shifted = Parabola.mk 5 (-20) 23 := by
  sorry

end parabola_shift_l526_52693


namespace third_number_is_five_l526_52699

def hcf (a b c : ℕ) : ℕ := sorry

def lcm (a b c : ℕ) : ℕ := sorry

theorem third_number_is_five (a b c : ℕ) 
  (ha : a = 30)
  (hb : b = 75)
  (hhcf : hcf a b c = 15)
  (hlcm : lcm a b c = 750) :
  c = 5 := by sorry

end third_number_is_five_l526_52699


namespace cafe_tables_needed_l526_52639

def base5ToDecimal (n : Nat) : Nat :=
  (n / 100) * 25 + ((n / 10) % 10) * 5 + (n % 10)

def customersPerTable : Nat := 3

def cafeCapacity : Nat := 123

theorem cafe_tables_needed :
  let decimalCapacity := base5ToDecimal cafeCapacity
  ⌈(decimalCapacity : ℚ) / customersPerTable⌉ = 13 := by
  sorry

end cafe_tables_needed_l526_52639


namespace rectangle_area_l526_52658

/-- Given a rectangle with width m centimeters and length 1 centimeter more than twice its width,
    its area is equal to 2m^2 + m square centimeters. -/
theorem rectangle_area (m : ℝ) : 
  let width := m
  let length := 2 * m + 1
  width * length = 2 * m^2 + m := by sorry

end rectangle_area_l526_52658


namespace unique_triple_l526_52674

theorem unique_triple (a b c : ℕ) : 
  a > 1 → b > 1 → c > 1 → 
  (bc + 1) % a = 0 → 
  (ac + 1) % b = 0 → 
  (ab + 1) % c = 0 → 
  a = 2 ∧ b = 3 ∧ c = 7 := by
sorry

end unique_triple_l526_52674


namespace road_repair_equation_l526_52655

theorem road_repair_equation (x : ℝ) (h : x > 0) : 
  (150 / x - 150 / (x + 5) = 5) ↔ 
  (∃ (original_days actual_days : ℝ), 
    original_days > 0 ∧ 
    actual_days > 0 ∧ 
    original_days = 150 / x ∧ 
    actual_days = 150 / (x + 5) ∧ 
    original_days - actual_days = 5) :=
by sorry

end road_repair_equation_l526_52655


namespace max_product_constrained_max_product_constrained_achieved_l526_52641

theorem max_product_constrained (a b : ℝ) : 
  a > 0 → b > 0 → a + 2*b = 2 → ab ≤ 1/2 := by
  sorry

theorem max_product_constrained_achieved (a b : ℝ) : 
  ∃ a b, a > 0 ∧ b > 0 ∧ a + 2*b = 2 ∧ ab = 1/2 := by
  sorry

end max_product_constrained_max_product_constrained_achieved_l526_52641


namespace quadratic_inequality_range_l526_52664

theorem quadratic_inequality_range (m : ℝ) : 
  (∀ x : ℝ, x > 0 ∧ x ≤ 1 → x^2 - 4*x ≥ m) → m ≤ -3 := by
  sorry

end quadratic_inequality_range_l526_52664


namespace first_number_value_l526_52695

theorem first_number_value (a b c : ℕ) : 
  a + b + c = 500 → 
  (b = 200 ∨ c = 200 ∨ a = 200) → 
  b = 2 * c → 
  c = 100 → 
  a = 200 := by
sorry

end first_number_value_l526_52695


namespace calculation_proof_l526_52607

theorem calculation_proof : -2^2 - Real.sqrt 9 + (-5)^2 * (2/5) = 3 := by
  sorry

end calculation_proof_l526_52607


namespace muffin_cost_l526_52646

theorem muffin_cost (num_muffins : ℕ) (juice_cost total_cost : ℚ) : 
  num_muffins = 3 → 
  juice_cost = 29/20 → 
  total_cost = 37/10 → 
  (total_cost - juice_cost) / num_muffins = 3/4 := by
  sorry

end muffin_cost_l526_52646


namespace factor_tree_problem_l526_52621

/-- Factor tree problem -/
theorem factor_tree_problem (F G Y Z X : ℕ) : 
  F = 2 * 5 →
  G = 7 * 3 →
  Y = 7 * F →
  Z = 11 * G →
  X = Y * Z →
  X = 16170 := by
  sorry

end factor_tree_problem_l526_52621


namespace sequence_properties_l526_52654

def a (n : ℕ) : ℚ := (2/3)^(n-1) * ((2/3)^(n-1) - 1)

theorem sequence_properties :
  (∀ n : ℕ, a n ≤ a 1) ∧
  (∀ n : ℕ, a n ≥ a 3) ∧
  (∀ n : ℕ, n ≥ 3 → a n > a (n+1)) ∧
  (a 1 = 0) ∧
  (a 3 = -20/81) := by
  sorry

end sequence_properties_l526_52654


namespace secret_codes_count_l526_52686

/-- The number of colors available in the game -/
def num_colors : ℕ := 8

/-- The number of slots in the game -/
def num_slots : ℕ := 5

/-- The total number of possible secret codes -/
def total_codes : ℕ := num_colors ^ num_slots

/-- Theorem stating that the total number of possible secret codes is 32768 -/
theorem secret_codes_count : total_codes = 32768 := by
  sorry

end secret_codes_count_l526_52686


namespace unique_valid_number_l526_52669

def is_valid_number (n : ℕ) : Prop :=
  ∃ (p q r s t u : ℕ),
    0 ≤ p ∧ p ≤ 9 ∧
    0 ≤ q ∧ q ≤ 9 ∧
    0 ≤ r ∧ r ≤ 9 ∧
    0 ≤ s ∧ s ≤ 9 ∧
    0 ≤ t ∧ t ≤ 9 ∧
    0 ≤ u ∧ u ≤ 9 ∧
    n = p * 10^7 + q * 10^6 + 7 * 10^5 + 8 * 10^4 + r * 10^3 + s * 10^2 + t * 10 + u ∧
    n % 17 = 0 ∧
    n % 19 = 0 ∧
    p + q + r + s = t + u

theorem unique_valid_number :
  ∃! n, is_valid_number n :=
sorry

end unique_valid_number_l526_52669


namespace pond_water_after_evaporation_l526_52643

/-- Calculates the remaining water in a pond after evaporation --/
def remaining_water (initial_amount : ℝ) (evaporation_rate : ℝ) (days : ℝ) : ℝ :=
  initial_amount - evaporation_rate * days

/-- Theorem: The pond contains 205 gallons after 45 days --/
theorem pond_water_after_evaporation :
  remaining_water 250 1 45 = 205 := by
  sorry

end pond_water_after_evaporation_l526_52643


namespace rabbits_count_l526_52679

/-- Represents the number of rabbits and peacocks in a zoo. -/
structure ZooAnimals where
  rabbits : ℕ
  peacocks : ℕ

/-- The total number of heads in the zoo is 60. -/
def total_heads (zoo : ZooAnimals) : Prop :=
  zoo.rabbits + zoo.peacocks = 60

/-- The total number of legs in the zoo is 192. -/
def total_legs (zoo : ZooAnimals) : Prop :=
  4 * zoo.rabbits + 2 * zoo.peacocks = 192

/-- Theorem stating that given the conditions, the number of rabbits is 36. -/
theorem rabbits_count (zoo : ZooAnimals) 
  (h1 : total_heads zoo) (h2 : total_legs zoo) : zoo.rabbits = 36 := by
  sorry

end rabbits_count_l526_52679


namespace point_movement_l526_52640

/-- 
Given a point P on a number line that is moved 4 units to the right and then 7 units to the left,
if its final position is 9, then its original position was 12.
-/
theorem point_movement (P : ℝ) : 
  (P + 4 - 7 = 9) → P = 12 := by
sorry

end point_movement_l526_52640


namespace box_volume_increase_l526_52670

/-- 
Given a rectangular box with dimensions l, w, and h satisfying:
1. Volume is 5400 cubic inches
2. Surface area is 1920 square inches
3. Sum of edge lengths is 240 inches
Prove that increasing each dimension by 2 inches results in a volume of 7568 cubic inches
-/
theorem box_volume_increase (l w h : ℝ) 
  (hvolume : l * w * h = 5400)
  (harea : 2 * (l * w + w * h + h * l) = 1920)
  (hedge : 4 * (l + w + h) = 240) :
  (l + 2) * (w + 2) * (h + 2) = 7568 := by
  sorry

end box_volume_increase_l526_52670


namespace triangle_side_length_l526_52663

/-- Given a triangle XYZ with side lengths and median, prove the length of XZ -/
theorem triangle_side_length (XY YZ XM : ℝ) (h1 : XY = 7) (h2 : YZ = 10) (h3 : XM = 5) :
  ∃ (XZ : ℝ), XZ = Real.sqrt 51 :=
by sorry

end triangle_side_length_l526_52663


namespace max_areas_is_9n_l526_52627

/-- Represents a circular disk divided by radii and secant lines -/
structure DividedDisk where
  n : ℕ
  radii : Fin (3 * n)
  secant_lines : Fin 2

/-- The maximum number of non-overlapping areas in a divided disk -/
def max_areas (disk : DividedDisk) : ℕ :=
  9 * disk.n

/-- Theorem stating that the maximum number of non-overlapping areas is 9n -/
theorem max_areas_is_9n (disk : DividedDisk) :
  max_areas disk = 9 * disk.n :=
by sorry

end max_areas_is_9n_l526_52627


namespace counterexample_exists_l526_52688

theorem counterexample_exists : ∃ (n : ℕ), n ≥ 2 ∧ 
  ∃ (k : ℕ), (2^(2^n) % (2^n - 1) = k) ∧ ¬∃ (m : ℕ), k = 4^m :=
by sorry

end counterexample_exists_l526_52688


namespace max_reciprocal_sum_l526_52684

/-- Given a quadratic polynomial x^2 - sx + q with roots a and b, 
    where a + b = a^2 + b^2 = a^3 + b^3 = ... = a^2008 + b^2008,
    the maximum value of 1/a^2009 + 1/b^2009 is 2. -/
theorem max_reciprocal_sum (s q a b : ℝ) : 
  (∀ n : ℕ, n ≥ 1 ∧ n ≤ 2008 → a^n + b^n = a + b) →
  a * b = q →
  a + b = s →
  x^2 - s*x + q = (x - a) * (x - b) →
  (∃ M : ℝ, ∀ s' q' a' b' : ℝ, 
    (∀ n : ℕ, n ≥ 1 ∧ n ≤ 2008 → a'^n + b'^n = a' + b') →
    a' * b' = q' →
    a' + b' = s' →
    x^2 - s'*x + q' = (x - a') * (x - b') →
    1 / a'^2009 + 1 / b'^2009 ≤ M) ∧
  1 / a^2009 + 1 / b^2009 = M →
  M = 2 := by
sorry

end max_reciprocal_sum_l526_52684


namespace probability_estimate_l526_52611

def is_hit (n : ℕ) : Bool :=
  n ≥ 3 ∧ n ≤ 9

def group_has_three_hits (group : List ℕ) : Bool :=
  (group.filter is_hit).length ≥ 3

def count_successful_groups (groups : List (List ℕ)) : ℕ :=
  (groups.filter group_has_three_hits).length

theorem probability_estimate (groups : List (List ℕ)) 
  (h1 : groups.length = 20) 
  (h2 : ∀ g ∈ groups, g.length = 4) 
  (h3 : ∀ g ∈ groups, ∀ n ∈ g, n ≥ 0 ∧ n ≤ 9) : 
  (count_successful_groups groups : ℚ) / groups.length = 0.6 := by
  sorry

end probability_estimate_l526_52611


namespace randy_store_spending_l526_52636

/-- Proves that Randy spends $2 per store trip -/
theorem randy_store_spending (initial_amount : ℕ) (final_amount : ℕ) (trips_per_month : ℕ) (months_per_year : ℕ) :
  initial_amount = 200 →
  final_amount = 104 →
  trips_per_month = 4 →
  months_per_year = 12 →
  (initial_amount - final_amount) / (trips_per_month * months_per_year) = 2 := by
  sorry

end randy_store_spending_l526_52636


namespace triangle_inequality_l526_52608

theorem triangle_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ¬(a + b > c ∧ b + c > a ∧ c + a > b) ↔ min a (min b c) + (a + b + c - max a (max b c) - min a (min b c)) ≤ max a (max b c) := by
  sorry

end triangle_inequality_l526_52608


namespace condition_equivalence_l526_52650

theorem condition_equivalence (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a^2 + b^2 ≥ 2*a*b) ↔ (a/b + b/a ≥ 2) :=
by sorry

end condition_equivalence_l526_52650


namespace sin_plus_cos_for_point_l526_52604

/-- Theorem: If the terminal side of angle α passes through point P(-4,3), then sin α + cos α = -1/5 -/
theorem sin_plus_cos_for_point (α : Real) : 
  (∃ (x y : Real), x = -4 ∧ y = 3 ∧ Real.cos α = x / Real.sqrt (x^2 + y^2) ∧ Real.sin α = y / Real.sqrt (x^2 + y^2)) → 
  Real.sin α + Real.cos α = -1/5 := by
  sorry

end sin_plus_cos_for_point_l526_52604


namespace sum_black_eq_sum_white_l526_52637

/-- Represents a frame in a multiplication table -/
structure Frame (m n : ℕ) :=
  (is_odd_m : Odd m)
  (is_odd_n : Odd n)

/-- The sum of numbers in black squares of the frame -/
def sum_black (f : Frame m n) : ℕ := sorry

/-- The sum of numbers in white squares of the frame -/
def sum_white (f : Frame m n) : ℕ := sorry

/-- Theorem stating that the sum of numbers in black squares equals the sum of numbers in white squares -/
theorem sum_black_eq_sum_white (m n : ℕ) (f : Frame m n) :
  sum_black f = sum_white f := by sorry

end sum_black_eq_sum_white_l526_52637

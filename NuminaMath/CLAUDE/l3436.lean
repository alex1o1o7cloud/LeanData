import Mathlib

namespace perpendicular_chords_diameter_l3436_343665

theorem perpendicular_chords_diameter (r : ℝ) (a b c d : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 →
  a + b = 7 →
  c + d = 8 →
  (a * b = r^2) ∧ (c * d = r^2) →
  2 * r = Real.sqrt 65 :=
by sorry

end perpendicular_chords_diameter_l3436_343665


namespace improved_milk_production_l3436_343628

/-- Given initial milk production parameters and an efficiency increase,
    calculate the new milk production for a different number of cows and days. -/
theorem improved_milk_production
  (a b c d e : ℝ)
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0)
  (h_initial : b / (a * c) = initial_rate)
  (h_efficiency_increase : new_rate = initial_rate * 1.2) :
  new_rate * d * e = (1.2 * b * d * e) / (a * c) :=
sorry

end improved_milk_production_l3436_343628


namespace tom_purchases_amount_l3436_343679

/-- Calculates the amount available for purchases given hourly rate, work hours, and savings rate. -/
def amountAvailableForPurchases (hourlyRate : ℚ) (workHours : ℕ) (savingsRate : ℚ) : ℚ :=
  let totalEarnings := hourlyRate * workHours
  let savingsAmount := savingsRate * totalEarnings
  totalEarnings - savingsAmount

/-- Proves that Tom's amount available for purchases is $181.35 -/
theorem tom_purchases_amount :
  let hourlyRate : ℚ := 13/2  -- $6.50
  let workHours : ℕ := 31
  let savingsRate : ℚ := 1/10  -- 10%
  amountAvailableForPurchases hourlyRate workHours savingsRate = 36270/200  -- $181.35
  := by sorry

end tom_purchases_amount_l3436_343679


namespace student_pairs_l3436_343655

theorem student_pairs (n : ℕ) (h : n = 12) : Nat.choose n 2 = 66 := by
  sorry

end student_pairs_l3436_343655


namespace arithmetic_progression_sum_dependency_l3436_343613

/-- Given an arithmetic progression with first term a and common difference d,
    s₁, s₂, and s₄ are the sums of n, 2n, and 4n terms respectively.
    R is defined as s₄ - s₂ - s₁. -/
theorem arithmetic_progression_sum_dependency
  (n : ℕ) (a d : ℝ) 
  (s₁ : ℝ := n * (2 * a + (n - 1) * d) / 2)
  (s₂ : ℝ := 2 * n * (2 * a + (2 * n - 1) * d) / 2)
  (s₄ : ℝ := 4 * n * (2 * a + (4 * n - 1) * d) / 2)
  (R : ℝ := s₄ - s₂ - s₁) :
  R = 6 * d * n^2 := by sorry

end arithmetic_progression_sum_dependency_l3436_343613


namespace cody_tickets_l3436_343659

/-- Calculates the final number of tickets Cody has after winning, spending, and winning again. -/
def final_tickets (initial : ℕ) (spent : ℕ) (won_later : ℕ) : ℕ :=
  initial - spent + won_later

/-- Theorem stating that Cody ends up with 30 tickets given the problem conditions. -/
theorem cody_tickets : final_tickets 49 25 6 = 30 := by
  sorry

end cody_tickets_l3436_343659


namespace subset_intersection_condition_l3436_343607

theorem subset_intersection_condition (M N : Set α) (h_nonempty : M.Nonempty) (h_subset : M ⊆ N) :
  (∀ a, a ∈ M ∩ N → (a ∈ M ∨ a ∈ N)) ∧
  ¬(∀ a, (a ∈ M ∨ a ∈ N) → a ∈ M ∩ N) :=
by sorry

end subset_intersection_condition_l3436_343607


namespace remainder_problem_l3436_343675

theorem remainder_problem : 123456789012 % 252 = 144 := by
  sorry

end remainder_problem_l3436_343675


namespace smallest_r_is_pi_over_two_l3436_343646

theorem smallest_r_is_pi_over_two :
  ∃ (r : ℝ) (f g : ℝ → ℝ), r > 0 ∧
    Differentiable ℝ f ∧ Differentiable ℝ g ∧
    f 0 > 0 ∧
    g 0 = 0 ∧
    (∀ x, |deriv f x| ≤ |g x|) ∧
    (∀ x, |deriv g x| ≤ |f x|) ∧
    f r = 0 ∧
    (∀ r' > 0, (∃ f' g' : ℝ → ℝ,
      Differentiable ℝ f' ∧ Differentiable ℝ g' ∧
      f' 0 > 0 ∧
      g' 0 = 0 ∧
      (∀ x, |deriv f' x| ≤ |g' x|) ∧
      (∀ x, |deriv g' x| ≤ |f' x|) ∧
      f' r' = 0) → r' ≥ r) ∧
    r = π / 2 := by
  sorry

end smallest_r_is_pi_over_two_l3436_343646


namespace wilsborough_savings_l3436_343661

/-- Mrs. Wilsborough's concert ticket purchase problem -/
theorem wilsborough_savings (vip_price regular_price : ℕ) 
  (vip_count regular_count leftover : ℕ) :
  vip_price = 100 →
  regular_price = 50 →
  vip_count = 2 →
  regular_count = 3 →
  leftover = 150 →
  vip_count * vip_price + regular_count * regular_price + leftover = 500 :=
by sorry

end wilsborough_savings_l3436_343661


namespace A_divisibility_l3436_343601

/-- Definition of A_l for a prime p > 3 -/
def A (p : ℕ) (l : ℕ) : ℕ :=
  sorry

/-- Theorem statement -/
theorem A_divisibility (p : ℕ) (h_prime : Nat.Prime p) (h_p_gt_3 : p > 3) :
  (∀ l, 1 ≤ l ∧ l ≤ p - 2 → p ∣ A p l) ∧
  (∀ l, 1 < l ∧ l < p ∧ Odd l → p^2 ∣ A p l) :=
by sorry

end A_divisibility_l3436_343601


namespace ring_arrangements_l3436_343608

theorem ring_arrangements (n k f : ℕ) (h1 : n = 10) (h2 : k = 7) (h3 : f = 5) :
  let m := (n.choose k) * k.factorial * ((k + f - 1).choose (f - 1))
  (m / 100000000 : ℕ) = 199 :=
by sorry

end ring_arrangements_l3436_343608


namespace pascal_triangle_symmetry_and_sum_l3436_343619

def pascal_triangle (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

theorem pascal_triangle_symmetry_and_sum (n : ℕ) :
  pascal_triangle 48 46 = pascal_triangle 48 2 ∧
  pascal_triangle 48 46 + pascal_triangle 48 2 = 2256 := by
  sorry

end pascal_triangle_symmetry_and_sum_l3436_343619


namespace simultaneous_ring_time_l3436_343632

def library_period : ℕ := 18
def hospital_period : ℕ := 24
def community_center_period : ℕ := 30

def next_simultaneous_ring (t₁ t₂ t₃ : ℕ) : ℕ :=
  Nat.lcm (Nat.lcm t₁ t₂) t₃

theorem simultaneous_ring_time :
  next_simultaneous_ring library_period hospital_period community_center_period = 360 :=
by sorry

end simultaneous_ring_time_l3436_343632


namespace prime_between_squares_l3436_343652

theorem prime_between_squares : ∃! p : ℕ, 
  Nat.Prime p ∧ 
  ∃ x : ℕ, p = x^2 + 5 ∧ p + 9 = (x + 1)^2 := by
sorry

end prime_between_squares_l3436_343652


namespace probability_at_least_one_correct_l3436_343687

theorem probability_at_least_one_correct (n : ℕ) (k : ℕ) :
  n > 0 → k > 0 →
  let p := 1 - (1 - 1 / n) ^ k
  p = 11529 / 15625 ↔ n = 5 ∧ k = 6 :=
sorry

end probability_at_least_one_correct_l3436_343687


namespace system_solution_l3436_343697

theorem system_solution :
  ∃ (x y : ℚ), 
    (4 * x - 3 * y = -8) ∧
    (5 * x + 9 * y = -18) ∧
    (x = -14/3) ∧
    (y = -32/9) := by
  sorry

end system_solution_l3436_343697


namespace guys_with_bullets_l3436_343641

theorem guys_with_bullets (n : ℕ) (h : n > 0) : 
  (∀ (guy : Fin n), 25 - 4 = (n * 25 - n * 4) / n) → n ≥ 1 :=
by sorry

end guys_with_bullets_l3436_343641


namespace vessel_capacity_proof_l3436_343686

/-- Proves that the capacity of the first vessel is 2 liters given the problem conditions -/
theorem vessel_capacity_proof (
  first_vessel_alcohol_percentage : ℝ)
  (second_vessel_capacity : ℝ)
  (second_vessel_alcohol_percentage : ℝ)
  (total_liquid_poured : ℝ)
  (new_vessel_capacity : ℝ)
  (new_mixture_alcohol_percentage : ℝ)
  (h1 : first_vessel_alcohol_percentage = 0.20)
  (h2 : second_vessel_capacity = 6)
  (h3 : second_vessel_alcohol_percentage = 0.55)
  (h4 : total_liquid_poured = 8)
  (h5 : new_vessel_capacity = 10)
  (h6 : new_mixture_alcohol_percentage = 0.37)
  : ∃ (first_vessel_capacity : ℝ),
    first_vessel_capacity = 2 ∧
    first_vessel_capacity * first_vessel_alcohol_percentage +
    second_vessel_capacity * second_vessel_alcohol_percentage =
    new_vessel_capacity * new_mixture_alcohol_percentage :=
by sorry

end vessel_capacity_proof_l3436_343686


namespace skew_perpendicular_plane_skew_parallel_perpendicular_l3436_343666

-- Define the basic types
variable (Point Line Plane : Type)

-- Define the relationships
variable (skew : Line → Line → Prop)
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (perpendicularLines : Line → Line → Prop)

-- Theorem 1
theorem skew_perpendicular_plane 
  (a b : Line) (α : Plane) 
  (h1 : skew a b) 
  (h2 : perpendicular a α) : 
  ¬ perpendicular b α := by sorry

-- Theorem 2
theorem skew_parallel_perpendicular 
  (a b l : Line) (α : Plane) 
  (h1 : skew a b) 
  (h2 : parallel a α) 
  (h3 : parallel b α) 
  (h4 : perpendicular l α) : 
  perpendicularLines l a ∧ perpendicularLines l b := by sorry

end skew_perpendicular_plane_skew_parallel_perpendicular_l3436_343666


namespace isosceles_triangle_third_vertex_y_coordinate_l3436_343604

/-- 
Given an isosceles triangle with:
- Base vertices at (3, 5) and (13, 5)
- Two equal sides of length 10 units
- Third vertex in the first quadrant

Prove that the y-coordinate of the third vertex is 5 + 5√3
-/
theorem isosceles_triangle_third_vertex_y_coordinate :
  ∀ (x y : ℝ),
  x > 0 →  -- First quadrant condition for x
  y > 5 →  -- First quadrant condition for y
  (x - 3)^2 + (y - 5)^2 = 100 →  -- Distance from (3, 5) is 10
  (x - 13)^2 + (y - 5)^2 = 100 →  -- Distance from (13, 5) is 10
  y = 5 + 5 * Real.sqrt 3 :=
by sorry

end isosceles_triangle_third_vertex_y_coordinate_l3436_343604


namespace modulo_congruence_l3436_343692

theorem modulo_congruence : ∃! n : ℕ, 0 ≤ n ∧ n ≤ 6 ∧ n ≡ 100000 [ZMOD 7] := by
  sorry

end modulo_congruence_l3436_343692


namespace count_squarish_numbers_l3436_343698

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def is_two_digit_perfect_square (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧ is_perfect_square n

def is_squarish (n : ℕ) : Prop :=
  100000 ≤ n ∧ n < 1000000 ∧
  is_perfect_square n ∧
  n % 16 = 0 ∧
  (∀ d, d ∈ n.digits 10 → d ≠ 0) ∧
  is_two_digit_perfect_square (n / 10000) ∧
  is_two_digit_perfect_square ((n / 100) % 100) ∧
  is_two_digit_perfect_square (n % 100)

theorem count_squarish_numbers :
  ∃! (s : Finset ℕ), (∀ n ∈ s, is_squarish n) ∧ s.card = 3 :=
sorry

end count_squarish_numbers_l3436_343698


namespace park_diameter_l3436_343623

/-- Given a circular park with concentric rings, calculate the diameter of the outer boundary. -/
theorem park_diameter (statue_width garden_width path_width fountain_diameter : ℝ) : 
  statue_width = 2 ∧ 
  garden_width = 10 ∧ 
  path_width = 8 ∧ 
  fountain_diameter = 12 → 
  2 * (fountain_diameter / 2 + statue_width + garden_width + path_width) = 52 := by
sorry

end park_diameter_l3436_343623


namespace absolute_value_equation_l3436_343672

theorem absolute_value_equation (x y : ℝ) :
  |2*x - Real.sqrt y| = 2*x + Real.sqrt y → y = 0 := by
  sorry

end absolute_value_equation_l3436_343672


namespace range_of_a_l3436_343610

def p (x : ℝ) : Prop := 2 * x^2 - 3 * x + 1 ≤ 0

def q (x a : ℝ) : Prop := x^2 - (2*a+1)*x + a*(a+1) ≤ 0

theorem range_of_a :
  (∀ x, ¬(p x) → ¬(q x a)) ∧
  (∃ x, ¬(p x) ∧ (q x a)) →
  0 ≤ a ∧ a ≤ 1/2 :=
sorry

end range_of_a_l3436_343610


namespace gaskets_sold_l3436_343622

/-- Calculates the total cost of gasket packages --/
def totalCost (packages : ℕ) : ℚ :=
  if packages ≤ 10 then
    25 * packages
  else
    250 + 20 * (packages - 10)

/-- Proves that 65 packages of gaskets were sold given the conditions --/
theorem gaskets_sold : ∃ (packages : ℕ), packages > 10 ∧ totalCost packages = 1340 := by
  sorry

#eval totalCost 65

end gaskets_sold_l3436_343622


namespace square_difference_division_problem_solution_l3436_343658

theorem square_difference_division (a b : ℕ) (h : a > b) :
  (a^2 - b^2) / (a - b) = a + b :=
by sorry

theorem problem_solution :
  (275^2 - 245^2) / 30 = 520 :=
by sorry

end square_difference_division_problem_solution_l3436_343658


namespace sin_thirty_degrees_l3436_343645

theorem sin_thirty_degrees : Real.sin (π / 6) = 1 / 2 := by
  sorry

end sin_thirty_degrees_l3436_343645


namespace s_one_eq_one_l3436_343667

/-- s(n) is a function that returns the n-digit number formed by attaching
    the first n perfect squares in order. -/
def s (n : ℕ) : ℕ := sorry

/-- Theorem: s(1) equals 1 -/
theorem s_one_eq_one : s 1 = 1 := by sorry

end s_one_eq_one_l3436_343667


namespace john_movie_count_l3436_343637

/-- The number of movies John has -/
def num_movies : ℕ := 100

/-- The trade-in value of each VHS in dollars -/
def vhs_value : ℕ := 2

/-- The cost of each DVD in dollars -/
def dvd_cost : ℕ := 10

/-- The total cost to replace all movies in dollars -/
def total_cost : ℕ := 800

theorem john_movie_count :
  (dvd_cost * num_movies) - (vhs_value * num_movies) = total_cost :=
by sorry

end john_movie_count_l3436_343637


namespace sqrt_7_simplest_l3436_343664

def is_simplest_quadratic_radical (x : ℝ) : Prop :=
  ∃ (y : ℝ), x = Real.sqrt y ∧ 
  (∀ (z : ℕ), z > 1 → ¬(∃ (w : ℝ), y = z * w ^ 2)) ∧
  y ≠ 1

theorem sqrt_7_simplest : 
  is_simplest_quadratic_radical (Real.sqrt 7) ∧
  ¬(is_simplest_quadratic_radical (Real.sqrt 4)) ∧
  ¬(is_simplest_quadratic_radical (Real.sqrt (1/4))) ∧
  ¬(is_simplest_quadratic_radical (Real.sqrt 27)) :=
sorry

end sqrt_7_simplest_l3436_343664


namespace complement_of_A_wrt_U_l3436_343653

def U : Set ℕ := {3, 4, 5, 6}
def A : Set ℕ := {3, 5}

theorem complement_of_A_wrt_U : 
  {x ∈ U | x ∉ A} = {4, 6} := by sorry

end complement_of_A_wrt_U_l3436_343653


namespace power_of_one_third_l3436_343620

theorem power_of_one_third (a b : ℕ) : 
  (2^a = 8 ∧ 5^b = 25) → (1/3 : ℚ)^(b - a) = 3 := by
  sorry

end power_of_one_third_l3436_343620


namespace min_value_inequality_l3436_343695

def f (k : ℝ) (x : ℝ) : ℝ := |x - 1| + |x + k|

theorem min_value_inequality (k : ℝ) (a b c : ℝ) 
  (h1 : k > 0)
  (h2 : ∀ x, f k x ≥ 3)
  (h3 : ∃ x, f k x = 3)
  (h4 : a + b + c = k) :
  a^2 + b^2 + c^2 ≥ 4/3 := by
sorry

end min_value_inequality_l3436_343695


namespace min_k_for_inequality_l3436_343639

theorem min_k_for_inequality (x y : ℝ) : 
  x * (x - 1) ≤ y * (1 - y) → 
  (∃ k : ℝ, (∀ x y : ℝ, x * (x - 1) ≤ y * (1 - y) → x^2 + y^2 ≤ k) ∧ 
   (∀ k' : ℝ, k' < k → ∃ x y : ℝ, x * (x - 1) ≤ y * (1 - y) ∧ x^2 + y^2 > k')) ∧
  (∀ k : ℝ, (∀ x y : ℝ, x * (x - 1) ≤ y * (1 - y) → x^2 + y^2 ≤ k) → k ≥ 2) :=
sorry

end min_k_for_inequality_l3436_343639


namespace M_in_fourth_quadrant_l3436_343618

/-- A point in the Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the fourth quadrant -/
def is_in_fourth_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- The given point M -/
def M : Point :=
  { x := 2, y := -5 }

/-- Theorem stating that M is in the fourth quadrant -/
theorem M_in_fourth_quadrant : is_in_fourth_quadrant M := by
  sorry


end M_in_fourth_quadrant_l3436_343618


namespace quadratic_form_k_value_l3436_343676

theorem quadratic_form_k_value (a h k : ℚ) : 
  (∀ x, x^2 - 7*x = a*(x - h)^2 + k) → k = -49/4 := by
  sorry

end quadratic_form_k_value_l3436_343676


namespace line_equation_equiv_l3436_343660

/-- The line equation in vector form -/
def line_equation (x y : ℝ) : Prop :=
  (3 : ℝ) * (x - 2) + (-4 : ℝ) * (y - 8) = 0

/-- The line equation in slope-intercept form -/
def slope_intercept_form (x y : ℝ) : Prop :=
  y = (3/4) * x + (13/2)

/-- Theorem stating the equivalence of the two forms -/
theorem line_equation_equiv :
  ∀ x y : ℝ, line_equation x y ↔ slope_intercept_form x y :=
sorry

end line_equation_equiv_l3436_343660


namespace triangle_properties_l3436_343699

/-- Theorem about properties of an acute triangle ABC --/
theorem triangle_properties 
  (A B C : Real) -- Angles of the triangle
  (a b c : Real) -- Sides of the triangle opposite to A, B, C respectively
  (h_acute : A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = π) -- Triangle is acute
  (h_sine : Real.sqrt 3 * a = 2 * c * Real.sin A) -- Given condition
  (h_side : a = 2) -- Given side length
  (h_area : (1/2) * a * c * Real.sin B = (3 * Real.sqrt 3) / 2) -- Given area
  : C = π/3 ∧ c = Real.sqrt 7 := by
  sorry


end triangle_properties_l3436_343699


namespace geometric_sequence_ratio_l3436_343693

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

/-- The arithmetic sequence condition -/
def ArithmeticCondition (a : ℕ → ℝ) : Prop :=
  2 * ((1 / 2) * a 3) = 3 * a 1 + 2 * a 2

theorem geometric_sequence_ratio (a : ℕ → ℝ) :
  GeometricSequence a →
  ArithmeticCondition a →
  (a 20 + a 19) / (a 18 + a 17) = 9 := by
  sorry

end geometric_sequence_ratio_l3436_343693


namespace tangent_line_problem_l3436_343694

theorem tangent_line_problem (a : ℝ) :
  (∃ (m : ℝ), 
    (∀ x y : ℝ, y = x^3 → (y - 0 = m * (x - 1) → (∀ t : ℝ, t ≠ x → t^3 > m * (t - 1)))) ∧
    (∀ x y : ℝ, y = a * x^2 + (15/4) * x - 9 → (y - 0 = m * (x - 1) → 
      (∀ t : ℝ, t ≠ x → a * t^2 + (15/4) * t - 9 ≠ m * (t - 1))))) →
  a = -1 ∨ a = -25/64 := by
sorry

end tangent_line_problem_l3436_343694


namespace sine_special_angle_l3436_343611

theorem sine_special_angle (α : Real) 
  (h1 : α ∈ Set.Ioo (π / 2) π) 
  (h2 : Real.sin (-α - π) = Real.sqrt 5 / 5) : 
  Real.sin (α - 3 * π / 2) = -(2 * Real.sqrt 5) / 5 := by
  sorry

end sine_special_angle_l3436_343611


namespace magnitude_of_z_l3436_343602

theorem magnitude_of_z (z : ℂ) (h : z * Complex.I = 1 - Complex.I) : Complex.abs z = Real.sqrt 2 := by
  sorry

end magnitude_of_z_l3436_343602


namespace aquaflow_pump_solution_l3436_343631

/-- Represents the Aquaflow system pumping problem -/
def AquaflowPump (initial_rate : ℝ) (increased_rate : ℝ) (target_volume : ℝ) : Prop :=
  let initial_time := 30 -- minutes
  let initial_volume := initial_rate * (initial_time / 60)
  let remaining_volume := target_volume - initial_volume
  let increased_time := (remaining_volume / increased_rate) * 60
  initial_time + increased_time = 75

/-- Theorem stating the solution to the Aquaflow pumping problem -/
theorem aquaflow_pump_solution :
  AquaflowPump 360 480 540 := by
  sorry

end aquaflow_pump_solution_l3436_343631


namespace power_three_mod_thirteen_l3436_343647

theorem power_three_mod_thirteen : 3^39 % 13 = 1 := by
  sorry

end power_three_mod_thirteen_l3436_343647


namespace expression_equals_100_l3436_343642

theorem expression_equals_100 : (50 - (2050 - 250)) + (2050 - (250 - 50)) = 100 := by
  sorry

end expression_equals_100_l3436_343642


namespace brianna_book_purchase_l3436_343612

theorem brianna_book_purchase (total_money : ℚ) (total_books : ℚ) :
  total_money > 0 ∧ total_books > 0 →
  (1 / 4 : ℚ) * total_money = (1 / 2 : ℚ) * total_books →
  total_money - 2 * ((1 / 4 : ℚ) * total_money) = (1 / 2 : ℚ) * total_money :=
by sorry

end brianna_book_purchase_l3436_343612


namespace square_sum_from_means_l3436_343600

theorem square_sum_from_means (a b : ℝ) 
  (h_arithmetic : (a + b) / 2 = 20)
  (h_geometric : Real.sqrt (a * b) = Real.sqrt 96) :
  a^2 + b^2 = 1408 := by
  sorry

end square_sum_from_means_l3436_343600


namespace ping_pong_dominating_subset_l3436_343656

/-- Represents a ping-pong match result between two players -/
inductive MatchResult
  | Win
  | Loss

/-- Represents a team of ping-pong players -/
def Team := Fin 1000

/-- Represents the result of all matches between two teams -/
def MatchResults := Team → Team → MatchResult

theorem ping_pong_dominating_subset (results : MatchResults) :
  ∃ (dominating_team : Bool) (subset : Finset Team),
    subset.card ≤ 10 ∧
    ∀ (opponent : Team),
      ∃ (player : Team),
        player ∈ subset ∧
        ((dominating_team = true  ∧ results player opponent = MatchResult.Win) ∨
         (dominating_team = false ∧ results opponent player = MatchResult.Loss)) :=
sorry

end ping_pong_dominating_subset_l3436_343656


namespace marbles_left_l3436_343643

theorem marbles_left (initial_marbles given_marbles : ℝ) :
  initial_marbles = 9.0 →
  given_marbles = 3.0 →
  initial_marbles - given_marbles = 6.0 := by
sorry

end marbles_left_l3436_343643


namespace intersection_complement_equality_l3436_343671

universe u

def I : Set ℕ := {0, 1, 2, 3}
def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {0, 2, 3}

theorem intersection_complement_equality :
  M ∩ (I \ N) = {1} := by sorry

end intersection_complement_equality_l3436_343671


namespace equation_solutions_l3436_343633

theorem equation_solutions : 
  (∃ (x₁ x₂ : ℝ), x₁ = 2 + Real.sqrt 3 ∧ x₂ = 2 - Real.sqrt 3 ∧ 
    x₁^2 - 4*x₁ + 1 = 0 ∧ x₂^2 - 4*x₂ + 1 = 0) ∧
  (∃ (x₃ x₄ : ℝ), x₃ = 2/5 ∧ x₄ = -5/3 ∧ 
    5*x₃ - 2 = (2 - 5*x₃)*(3*x₃ + 4) ∧ 5*x₄ - 2 = (2 - 5*x₄)*(3*x₄ + 4)) :=
by sorry


end equation_solutions_l3436_343633


namespace derivative_f_l3436_343636

noncomputable def f (x : ℝ) : ℝ := (3/2) * Real.log (Real.tanh (x/2)) + Real.cosh x - Real.cosh x / (2 * Real.sinh x ^ 2)

theorem derivative_f (x : ℝ) : 
  deriv f x = Real.cosh x ^ 4 / Real.sinh x ^ 3 :=
by sorry

end derivative_f_l3436_343636


namespace tan_equation_solution_l3436_343680

theorem tan_equation_solution (x : ℝ) : 
  x = 30 * Real.pi / 180 → 
  Real.tan (3 * x) * Real.tan (5 * x) = Real.tan (7 * x) * Real.tan (9 * x) :=
by
  sorry

end tan_equation_solution_l3436_343680


namespace two_element_subsets_of_three_element_set_l3436_343678

theorem two_element_subsets_of_three_element_set :
  let S : Finset Int := {-1, 0, 2}
  (Finset.filter (fun M => M.card = 2) (S.powerset)).card = 3 := by
  sorry

end two_element_subsets_of_three_element_set_l3436_343678


namespace performance_arrangements_l3436_343683

/-- The number of performances of each type -/
def num_singing : ℕ := 2
def num_dance : ℕ := 3
def num_variety : ℕ := 3

/-- The total number of performances -/
def total_performances : ℕ := num_singing + num_dance + num_variety

/-- Number of ways to arrange performances with singing at beginning and end -/
def arrangement_singing_ends : ℕ := 1440

/-- Number of ways to arrange performances with non-adjacent singing -/
def arrangement_non_adjacent_singing : ℕ := 30240

/-- Number of ways to arrange performances with adjacent singing and non-adjacent dance -/
def arrangement_adjacent_singing_non_adjacent_dance : ℕ := 2880

theorem performance_arrangements :
  (total_performances = 8) →
  (arrangement_singing_ends = 1440) ∧
  (arrangement_non_adjacent_singing = 30240) ∧
  (arrangement_adjacent_singing_non_adjacent_dance = 2880) :=
by sorry

end performance_arrangements_l3436_343683


namespace farmer_problem_solution_l3436_343673

/-- A farmer sells ducks and chickens and buys a wheelbarrow -/
def FarmerProblem (duck_price chicken_price : ℕ) (duck_sold chicken_sold : ℕ) (wheelbarrow_profit : ℕ) :=
  let total_earnings := duck_price * duck_sold + chicken_price * chicken_sold
  let wheelbarrow_cost := wheelbarrow_profit / 2
  (wheelbarrow_cost : ℚ) / total_earnings = 1 / 2

theorem farmer_problem_solution :
  FarmerProblem 10 8 2 5 60 := by sorry

end farmer_problem_solution_l3436_343673


namespace weekend_sleep_calculation_l3436_343644

/-- Calculates the number of hours slept during weekends per day, given the total weekly sleep and weekday sleep hours. -/
def weekend_sleep_hours (total_weekly_sleep : ℕ) (weekday_sleep : ℕ) : ℚ :=
  ((total_weekly_sleep - (weekday_sleep * 5)) : ℚ) / 2

/-- Theorem stating that given 51 hours of total weekly sleep and 7 hours of sleep each weekday, 
    the number of hours slept each day during weekends is 8. -/
theorem weekend_sleep_calculation :
  weekend_sleep_hours 51 7 = 8 := by
  sorry

end weekend_sleep_calculation_l3436_343644


namespace triangle_hypotenuse_length_l3436_343629

-- Define the triangle and points
def Triangle (P Q R : ℝ × ℝ) : Prop := sorry

def RightTriangle (P Q R : ℝ × ℝ) : Prop := 
  Triangle P Q R ∧ sorry -- Add condition for right angle

def PointOnLine (P Q M : ℝ × ℝ) : Prop := sorry

-- Define the ratio condition
def RatioCondition (P M Q : ℝ × ℝ) : Prop := 
  ∃ (k : ℝ), k = 1/3 ∧ sorry -- Add condition for PM:MQ = 1:3

-- Define the distance function
def distance (A B : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem triangle_hypotenuse_length 
  (P Q R M N : ℝ × ℝ) 
  (h1 : RightTriangle P Q R) 
  (h2 : PointOnLine P Q M) 
  (h3 : PointOnLine P R N) 
  (h4 : RatioCondition P M Q) 
  (h5 : RatioCondition P N R) 
  (h6 : distance Q N = 20) 
  (h7 : distance M R = 36) : 
  distance Q R = 2 * Real.sqrt 399 := by
  sorry

end triangle_hypotenuse_length_l3436_343629


namespace ellipse_minor_axis_length_l3436_343688

/-- Represents a point in 2D plane -/
structure Point where
  x : ℚ
  y : ℚ

/-- Represents an ellipse -/
structure Ellipse where
  center : Point
  semimajor : ℚ
  semiminor : ℚ

/-- Check if a point lies on the ellipse -/
def pointOnEllipse (p : Point) (e : Ellipse) : Prop :=
  (p.x - e.center.x)^2 / e.semimajor^2 + (p.y - e.center.y)^2 / e.semiminor^2 = 1

/-- The six given points -/
def points : List Point := [
  ⟨-5/2, 2⟩, ⟨0, 0⟩, ⟨0, 3⟩, ⟨4, 0⟩, ⟨4, 3⟩, ⟨2, 4⟩
]

/-- The ellipse passing through the points -/
def ellipse : Ellipse := ⟨⟨2, 3/2⟩, 2, 5/2⟩

theorem ellipse_minor_axis_length :
  (∀ (p1 p2 p3 : Point), p1 ∈ points → p2 ∈ points → p3 ∈ points → 
    p1 ≠ p2 → p2 ≠ p3 → p1 ≠ p3 → ¬ (∃ (m b : ℚ), p1.y = m * p1.x + b ∧ p2.y = m * p2.x + b ∧ p3.y = m * p3.x + b)) →
  (∀ p : Point, p ∈ points → pointOnEllipse p ellipse) →
  ellipse.semiminor * 2 = 5 := by
  sorry

end ellipse_minor_axis_length_l3436_343688


namespace polygon_sides_l3436_343689

theorem polygon_sides (sum_interior_angles : ℝ) : sum_interior_angles = 540 → ∃ n : ℕ, n = 5 ∧ (n - 2) * 180 = sum_interior_angles := by
  sorry

end polygon_sides_l3436_343689


namespace range_of_a_l3436_343630

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 12, x^2 - a ≥ 0) ∨ 
  (∃ x₀ : ℝ, x₀^2 + (a-1)*x₀ + 1 < 0) →
  ¬((∀ x ∈ Set.Icc 1 12, x^2 - a ≥ 0) ∧ 
    (∃ x₀ : ℝ, x₀^2 + (a-1)*x₀ + 1 < 0)) →
  (-1 ≤ a ∧ a ≤ 1) ∨ a > 3 :=
by sorry


end range_of_a_l3436_343630


namespace least_possible_area_of_square_l3436_343670

/-- Represents the measurement of a square's side length to the nearest centimeter. -/
def MeasuredSideLength : ℝ := 4

/-- The minimum possible actual side length given the measured side length. -/
def MinActualSideLength : ℝ := MeasuredSideLength - 0.5

/-- Calculates the area of a square given its side length. -/
def SquareArea (sideLength : ℝ) : ℝ := sideLength * sideLength

/-- The least possible actual area of the square. -/
def LeastPossibleArea : ℝ := SquareArea MinActualSideLength

theorem least_possible_area_of_square :
  LeastPossibleArea = 12.25 := by
  sorry

end least_possible_area_of_square_l3436_343670


namespace find_d_l3436_343621

-- Define the functions f and g
def f (c : ℝ) (x : ℝ) : ℝ := 5 * x + c
def g (c : ℝ) (x : ℝ) : ℝ := c * x - 3

-- State the theorem
theorem find_d (c : ℝ) :
  (∃ d : ℝ, ∀ x : ℝ, f c (g c x) = 15 * x + d) →
  (∃ d : ℝ, ∀ x : ℝ, f c (g c x) = 15 * x + d ∧ d = -12) :=
by sorry

end find_d_l3436_343621


namespace grid_value_theorem_l3436_343685

/-- Represents a 5x5 grid of integers -/
def Grid := Fin 5 → Fin 5 → ℤ

/-- Checks if a sequence of 5 integers forms an arithmetic progression -/
def isArithmeticSequence (seq : Fin 5 → ℤ) : Prop :=
  ∀ i j k : Fin 5, i < j → j < k → seq j - seq i = seq k - seq j

/-- Checks if all rows and columns of a grid form arithmetic sequences -/
def isValidGrid (g : Grid) : Prop :=
  (∀ row : Fin 5, isArithmeticSequence (λ col => g row col)) ∧
  (∀ col : Fin 5, isArithmeticSequence (λ row => g row col))

theorem grid_value_theorem (g : Grid) :
  isValidGrid g →
  g 1 1 = 74 →
  g 2 4 = 186 →
  g 3 2 = 103 →
  g 4 0 = 0 →
  g 0 3 = 142 := by
  sorry

#check grid_value_theorem

end grid_value_theorem_l3436_343685


namespace cheryl_mm_theorem_l3436_343606

def cheryl_mm_problem (initial : ℕ) (eaten_lunch : ℕ) (eaten_dinner : ℕ) (remaining : ℕ) : Prop :=
  initial - eaten_lunch - eaten_dinner - remaining = 18

theorem cheryl_mm_theorem :
  cheryl_mm_problem 40 7 5 10 := by sorry

end cheryl_mm_theorem_l3436_343606


namespace quadratic_equation_roots_property_l3436_343696

theorem quadratic_equation_roots_property : ∃ (p q : ℝ),
  p + q = 7 ∧
  |p - q| = 9 ∧
  ∀ x, x^2 - 7*x - 8 = 0 ↔ (x = p ∨ x = q) := by
  sorry

end quadratic_equation_roots_property_l3436_343696


namespace investment_principal_l3436_343635

/-- Proves that an investment with a 9% simple annual interest rate yielding $231 monthly interest has a principal of $30,800 --/
theorem investment_principal (monthly_interest : ℝ) (annual_rate : ℝ) :
  monthly_interest = 231 →
  annual_rate = 0.09 →
  (monthly_interest / (annual_rate / 12)) = 30800 := by
  sorry

end investment_principal_l3436_343635


namespace tenth_term_of_sequence_l3436_343616

def arithmetic_sequence (a : ℚ) (d : ℚ) (n : ℕ) : ℚ := a + (n - 1 : ℚ) * d

theorem tenth_term_of_sequence (a : ℚ) (b : ℚ) :
  a = 3/4 → b = 1 → arithmetic_sequence a (b - a) 10 = 3 := by
  sorry

end tenth_term_of_sequence_l3436_343616


namespace solution_in_quadrant_IV_l3436_343624

/-- Given a system of equations x + 2y = 4 and kx - y = 1, where k is a constant,
    the solution (x, y) is in Quadrant IV if and only if -1/2 < k < 2 -/
theorem solution_in_quadrant_IV (k : ℝ) : 
  (∃ x y : ℝ, x + 2*y = 4 ∧ k*x - y = 1 ∧ x > 0 ∧ y < 0) ↔ -1/2 < k ∧ k < 2 := by
  sorry

end solution_in_quadrant_IV_l3436_343624


namespace jack_john_vote_difference_l3436_343674

/-- Calculates the number of votes Jack received more than John in an election with given conditions. -/
theorem jack_john_vote_difference :
  let total_votes : ℕ := 1150
  let john_votes : ℕ := 150
  let remaining_votes : ℕ := total_votes - john_votes
  let james_votes : ℕ := (7 * remaining_votes) / 10
  let jacob_votes : ℕ := (3 * (john_votes + james_votes)) / 10
  let joey_votes : ℕ := ((125 * jacob_votes) + 50) / 100
  let jack_votes : ℕ := (95 * joey_votes) / 100
  jack_votes - john_votes = 153 := by sorry

end jack_john_vote_difference_l3436_343674


namespace andy_candy_problem_l3436_343684

/-- The number of teachers who gave Andy candy canes -/
def num_teachers : ℕ := sorry

/-- The number of candy canes Andy gets from his parents -/
def candy_from_parents : ℕ := 2

/-- The number of candy canes Andy gets from each teacher -/
def candy_per_teacher : ℕ := 3

/-- The fraction of candy canes Andy buys compared to what he was given -/
def buy_fraction : ℚ := 1 / 7

/-- The number of candy canes that cause one cavity -/
def candy_per_cavity : ℕ := 4

/-- The total number of cavities Andy gets -/
def total_cavities : ℕ := 16

theorem andy_candy_problem :
  let total_candy := candy_from_parents + num_teachers * candy_per_teacher
  let bought_candy := (total_candy : ℚ) * buy_fraction
  (↑total_candy + bought_candy) / candy_per_cavity = total_cavities ↔ num_teachers = 18 := by
  sorry

end andy_candy_problem_l3436_343684


namespace ratio_w_to_y_l3436_343677

theorem ratio_w_to_y (w x y z : ℝ) 
  (hw : w / x = 4 / 3)
  (hy : y / z = 3 / 2)
  (hz : z / x = 1 / 6) :
  w / y = 16 / 3 := by
sorry

end ratio_w_to_y_l3436_343677


namespace terrell_weight_lifting_l3436_343627

/-- The number of times Terrell lifts the original weights -/
def original_lifts : ℕ := 15

/-- The weight of each original weight in pounds -/
def original_weight : ℕ := 25

/-- The weight of each new weight in pounds -/
def new_weight : ℕ := 10

/-- The number of weights Terrell lifts each time -/
def num_weights : ℕ := 2

/-- The total weight Terrell lifts with the original weights -/
def total_original_weight : ℕ := num_weights * original_weight * original_lifts

/-- The number of times Terrell needs to lift the new weights to match the original total weight -/
def new_lifts : ℚ := total_original_weight / (num_weights * new_weight)

theorem terrell_weight_lifting :
  new_lifts = 37.5 := by sorry

end terrell_weight_lifting_l3436_343627


namespace fraction_to_decimal_l3436_343691

theorem fraction_to_decimal : (45 : ℚ) / (2^3 * 5^4) = (9 : ℚ) / 1000 := by
  sorry

end fraction_to_decimal_l3436_343691


namespace quadratic_two_real_roots_l3436_343640

/-- 
Given a quadratic equation (a-1)x^2 - 4x - 1 = 0, where 'a' is a parameter,
this theorem states the conditions on 'a' for the equation to have two real roots.
-/
theorem quadratic_two_real_roots (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ (a - 1) * x^2 - 4*x - 1 = 0 ∧ (a - 1) * y^2 - 4*y - 1 = 0) ↔ 
  (a ≥ -3 ∧ a ≠ 1) :=
sorry

end quadratic_two_real_roots_l3436_343640


namespace four_propositions_true_l3436_343650

theorem four_propositions_true (x y : ℝ) : 
  (((x = 0 ∧ y = 0) → (x^2 + y^2 ≠ 0)) ∧                   -- Original
   ((x^2 + y^2 ≠ 0) → (x = 0 ∧ y = 0)) ∧                   -- Converse
   (¬(x = 0 ∧ y = 0) → ¬(x^2 + y^2 ≠ 0)) ∧                 -- Inverse
   (¬(x^2 + y^2 ≠ 0) → ¬(x = 0 ∧ y = 0)))                  -- Contrapositive
  := by sorry

end four_propositions_true_l3436_343650


namespace arithmetic_sequence_ratio_l3436_343615

/-- Given an arithmetic sequence with first four terms a, x, b, 2x, prove that a/b = 1/3 -/
theorem arithmetic_sequence_ratio (a x b : ℝ) :
  (x - a = b - x) ∧ (b - x = 2 * x - b) → a / b = 1 / 3 := by
  sorry

end arithmetic_sequence_ratio_l3436_343615


namespace proposition_logic_l3436_343690

theorem proposition_logic (p q : Prop) (hp : p = (2 + 2 = 5)) (hq : q = (3 > 2)) :
  (p ∨ q) ∧ ¬(¬q) := by sorry

end proposition_logic_l3436_343690


namespace f_deriv_negative_one_eq_negative_two_l3436_343605

-- Define the function f
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^4 + b * x^2 + c

-- Define the derivative of f
def f_deriv (a b : ℝ) (x : ℝ) : ℝ := 4 * a * x^3 + 2 * b * x

-- State the theorem
theorem f_deriv_negative_one_eq_negative_two 
  (a b c : ℝ) (h : f_deriv a b 1 = 2) : f_deriv a b (-1) = -2 := by
  sorry

end f_deriv_negative_one_eq_negative_two_l3436_343605


namespace not_always_both_false_l3436_343669

theorem not_always_both_false (p q : Prop) : 
  ¬(p ∧ q) → (¬p ∧ ¬q) → False :=
sorry

end not_always_both_false_l3436_343669


namespace average_age_problem_l3436_343654

theorem average_age_problem (a c : ℝ) : 
  (a + c) / 2 = 32 →
  ((a + c) + 23) / 3 = 29 :=
by
  sorry

end average_age_problem_l3436_343654


namespace evaluate_expression_l3436_343603

theorem evaluate_expression : 3 * 307 + 4 * 307 + 2 * 307 + 307^2 = 97012 := by
  sorry

end evaluate_expression_l3436_343603


namespace invalid_external_diagonals_l3436_343657

def is_valid_external_diagonals (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a^2 + b^2 > c^2 ∧
  a^2 + c^2 > b^2 ∧
  b^2 + c^2 > a^2

theorem invalid_external_diagonals :
  ¬ (is_valid_external_diagonals 5 6 9) :=
by sorry

end invalid_external_diagonals_l3436_343657


namespace find_g_x_l3436_343648

/-- Given that 4x^4 - 6x^2 + 2 + g(x) = 7x^3 - 3x^2 + 4x - 1 for all x,
    prove that g(x) = -4x^4 + 7x^3 + 3x^2 + 4x - 3 -/
theorem find_g_x (g : ℝ → ℝ) :
  (∀ x, 4 * x^4 - 6 * x^2 + 2 + g x = 7 * x^3 - 3 * x^2 + 4 * x - 1) →
  (∀ x, g x = -4 * x^4 + 7 * x^3 + 3 * x^2 + 4 * x - 3) := by
  sorry

end find_g_x_l3436_343648


namespace parabola_tangent_to_line_l3436_343662

/-- 
Given a parabola y = ax^2 + 6 that is tangent to the line y = 2x - 3,
prove that the value of the constant a is 1/9.
-/
theorem parabola_tangent_to_line (a : ℝ) : 
  (∃ x : ℝ, a * x^2 + 6 = 2 * x - 3 ∧ 
   ∀ y : ℝ, y ≠ x → a * y^2 + 6 ≠ 2 * y - 3) →
  a = 1 / 9 := by
sorry

end parabola_tangent_to_line_l3436_343662


namespace smallest_n_congruence_l3436_343649

theorem smallest_n_congruence : ∃! n : ℕ, (∀ a ∈ Finset.range 9, n % (a + 2) = (a + 1)) ∧ 
  (∀ m : ℕ, m < n → ∃ a ∈ Finset.range 9, m % (a + 2) ≠ (a + 1)) ∧ n = 2519 := by
  sorry

end smallest_n_congruence_l3436_343649


namespace tank_filling_time_l3436_343682

/-- Proves that the first pipe takes 5 hours to fill the tank alone given the conditions of the problem -/
theorem tank_filling_time (T : ℝ) 
  (h1 : T > 0)  -- Ensuring T is positive
  (h2 : 1/T + 1/4 - 1/20 = 1/2.5) : T = 5 := by
  sorry


end tank_filling_time_l3436_343682


namespace number_of_girls_l3436_343681

theorem number_of_girls (total_pupils : ℕ) (boys : ℕ) (teachers : ℕ) 
  (h1 : total_pupils = 626)
  (h2 : boys = 318)
  (h3 : teachers = 36) :
  total_pupils - boys - teachers = 272 := by
  sorry

end number_of_girls_l3436_343681


namespace set_operations_l3436_343651

open Set

-- Define the sets A and B
def A : Set ℝ := {x | -1 ≤ x ∧ x < 4}
def B : Set ℝ := {x | 3 * x - 7 ≤ 8 - 2 * x}

-- State the theorem
theorem set_operations :
  (B = {x : ℝ | x ≤ 3}) ∧
  (A ∪ B = {x : ℝ | x < 4}) ∧
  ((Aᶜ) ∩ B = {x : ℝ | x < -1}) := by
  sorry

end set_operations_l3436_343651


namespace inscribed_circle_radius_l3436_343617

/-- An isosceles right triangle with legs of length 8 units -/
structure IsoscelesRightTriangle where
  /-- The length of each leg -/
  leg_length : ℝ
  /-- The leg length is 8 units -/
  leg_is_eight : leg_length = 8

/-- The inscribed circle of the isosceles right triangle -/
def inscribed_circle (t : IsoscelesRightTriangle) : ℝ := sorry

/-- Theorem: The radius of the inscribed circle in an isosceles right triangle
    with legs of length 8 units is 8 - 4√2 -/
theorem inscribed_circle_radius (t : IsoscelesRightTriangle) :
  inscribed_circle t = 8 - 4 * Real.sqrt 2 := by sorry

end inscribed_circle_radius_l3436_343617


namespace total_cookies_l3436_343625

theorem total_cookies (cookies_per_bag : ℕ) (number_of_bags : ℕ) : 
  cookies_per_bag = 41 → number_of_bags = 53 → cookies_per_bag * number_of_bags = 2173 := by
  sorry

end total_cookies_l3436_343625


namespace conditional_probability_b_given_a_and_c_l3436_343638

-- Define the sample space and probability measure
variable (Ω : Type) [MeasurableSpace Ω]
variable (P : Measure Ω)

-- Define events as measurable sets
variable (a b c : Set Ω)

-- Define probabilities
variable (pa pb pc pab pac pbc pabc : ℝ)

-- State the theorem
theorem conditional_probability_b_given_a_and_c
  (h_pa : P a = pa)
  (h_pb : P b = pb)
  (h_pc : P c = pc)
  (h_pab : P (a ∩ b) = pab)
  (h_pac : P (a ∩ c) = pac)
  (h_pbc : P (b ∩ c) = pbc)
  (h_pabc : P (a ∩ b ∩ c) = pabc)
  (h_pa_val : pa = 5/23)
  (h_pb_val : pb = 7/23)
  (h_pc_val : pc = 1/23)
  (h_pab_val : pab = 2/23)
  (h_pac_val : pac = 1/23)
  (h_pbc_val : pbc = 1/23)
  (h_pabc_val : pabc = 1/23)
  : P (b ∩ (a ∩ c)) / P (a ∩ c) = 1 :=
sorry

end conditional_probability_b_given_a_and_c_l3436_343638


namespace count_divisible_numbers_main_result_l3436_343626

theorem count_divisible_numbers (n : ℕ) (m : ℕ) : 
  (Finset.filter (fun k => (k^2 - 1) % m = 0) (Finset.range (n + 1))).card = 4 * (n / m) :=
by
  sorry

theorem main_result : 
  (Finset.filter (fun k => (k^2 - 1) % 485 = 0) (Finset.range 485001)).card = 4000 :=
by
  sorry

end count_divisible_numbers_main_result_l3436_343626


namespace a5_greater_than_b5_l3436_343609

-- Define the geometric sequence a_n
def geometric_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a₁ * q ^ (n - 1)

-- Define the arithmetic sequence b_n
def arithmetic_sequence (b₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  b₁ + (n - 1) * d

theorem a5_greater_than_b5 
  (a₁ b₁ q d : ℝ)
  (h1 : a₁ = b₁)
  (h2 : a₁ > 0)
  (h3 : geometric_sequence a₁ q 3 = arithmetic_sequence b₁ d 3)
  (h4 : a₁ ≠ geometric_sequence a₁ q 3) :
  geometric_sequence a₁ q 5 > arithmetic_sequence b₁ d 5 := by
  sorry

end a5_greater_than_b5_l3436_343609


namespace total_savings_theorem_l3436_343668

/-- Represents the savings of a child in various currencies and denominations -/
structure Savings where
  pennies : ℕ
  nickels : ℕ
  dimes : ℕ
  quarters : ℕ
  one_dollar_bills : ℕ
  five_dollar_bills : ℕ
  two_dollar_canadian_coins : ℕ
  one_dollar_canadian_coins : ℕ
  five_dollar_canadian_bills : ℕ
  one_pound_uk_coins : ℕ

/-- Conversion rates for different currencies -/
structure ConversionRates where
  british_pound_to_usd : ℚ
  canadian_dollar_to_usd : ℚ

/-- Calculates the total savings in US dollars -/
def calculate_total_savings (teagan_savings : Savings) (rex_savings : Savings) (toni_savings : Savings) (rates : ConversionRates) : ℚ :=
  sorry

/-- Theorem stating the total savings of the three kids -/
theorem total_savings_theorem (teagan_savings rex_savings toni_savings : Savings) (rates : ConversionRates) :
  teagan_savings.pennies = 200 ∧
  teagan_savings.one_dollar_bills = 15 ∧
  teagan_savings.two_dollar_canadian_coins = 13 ∧
  rex_savings.nickels = 100 ∧
  rex_savings.quarters = 45 ∧
  rex_savings.one_pound_uk_coins = 8 ∧
  rex_savings.one_dollar_canadian_coins = 20 ∧
  toni_savings.dimes = 330 ∧
  toni_savings.five_dollar_bills = 12 ∧
  toni_savings.five_dollar_canadian_bills = 7 ∧
  rates.british_pound_to_usd = 138/100 ∧
  rates.canadian_dollar_to_usd = 76/100 →
  calculate_total_savings teagan_savings rex_savings toni_savings rates = 19885/100 := by
  sorry

end total_savings_theorem_l3436_343668


namespace cricketer_average_score_l3436_343614

theorem cricketer_average_score 
  (total_matches : ℕ) 
  (matches_with_known_average : ℕ) 
  (known_average : ℝ) 
  (total_average : ℝ) 
  (h1 : total_matches = 25)
  (h2 : matches_with_known_average = 15)
  (h3 : known_average = 70)
  (h4 : total_average = 66) :
  let remaining_matches := total_matches - matches_with_known_average
  (total_matches * total_average - matches_with_known_average * known_average) / remaining_matches = 60 := by
  sorry

end cricketer_average_score_l3436_343614


namespace pig_bacon_profit_l3436_343663

def average_pig_bacon : ℝ := 20
def average_type_a_bacon : ℝ := 12
def average_type_b_bacon : ℝ := 8
def type_a_price : ℝ := 6
def type_b_price : ℝ := 4
def this_pig_size_ratio : ℝ := 0.5
def this_pig_type_a_ratio : ℝ := 0.75
def this_pig_type_b_ratio : ℝ := 0.25
def type_a_cost : ℝ := 1.5
def type_b_cost : ℝ := 0.8

theorem pig_bacon_profit : 
  let this_pig_bacon := average_pig_bacon * this_pig_size_ratio
  let this_pig_type_a := this_pig_bacon * this_pig_type_a_ratio
  let this_pig_type_b := this_pig_bacon * this_pig_type_b_ratio
  let revenue := this_pig_type_a * type_a_price + this_pig_type_b * type_b_price
  let cost := this_pig_type_a * type_a_cost + this_pig_type_b * type_b_cost
  revenue - cost = 41.75 := by
sorry

end pig_bacon_profit_l3436_343663


namespace final_amounts_l3436_343634

/-- Represents a person with their current amount of money -/
structure Person where
  name : String
  amount : ℚ

/-- Represents the state of all persons involved in the transactions -/
structure State where
  michael : Person
  thomas : Person
  emily : Person

/-- Performs the series of transactions described in the problem -/
def performTransactions (initial : State) : State :=
  let s1 := { initial with
    michael := { initial.michael with amount := initial.michael.amount * (1 - 0.3) },
    thomas := { initial.thomas with amount := initial.thomas.amount + initial.michael.amount * 0.3 }
  }
  let s2 := { s1 with
    thomas := { s1.thomas with amount := s1.thomas.amount * (1 - 0.25) },
    emily := { s1.emily with amount := s1.emily.amount + s1.thomas.amount * 0.25 }
  }
  let s3 := { s2 with
    emily := { s2.emily with amount := (s2.emily.amount - 10) / 2 },
    michael := { s2.michael with amount := s2.michael.amount + (s2.emily.amount - 10) / 2 }
  }
  s3

/-- The main theorem stating the final amounts after transactions -/
theorem final_amounts (initial : State)
  (h_michael : initial.michael.amount = 42)
  (h_thomas : initial.thomas.amount = 17)
  (h_emily : initial.emily.amount = 30) :
  let final := performTransactions initial
  final.michael.amount = 43.1 ∧
  final.thomas.amount = 22.2 ∧
  final.emily.amount = 13.7 := by
  sorry


end final_amounts_l3436_343634

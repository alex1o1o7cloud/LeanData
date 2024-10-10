import Mathlib

namespace complex_expression_simplification_l589_58949

theorem complex_expression_simplification :
  3 / Real.sqrt 3 - (Real.sqrt 3)^2 - Real.sqrt 27 + |Real.sqrt 3 - 2| = -1 - 3 * Real.sqrt 3 := by
  sorry

end complex_expression_simplification_l589_58949


namespace race_head_start_l589_58947

/-- Proves the head start distance in a race with given conditions -/
theorem race_head_start 
  (race_distance : ℝ) 
  (speed_ratio : ℝ) 
  (win_margin : ℝ) 
  (h1 : race_distance = 600)
  (h2 : speed_ratio = 5/4)
  (h3 : win_margin = 200) :
  ∃ (head_start : ℝ), 
    head_start = 100 ∧ 
    (race_distance - head_start) / speed_ratio = (race_distance - win_margin) / 1 :=
by sorry

end race_head_start_l589_58947


namespace fraction_equality_l589_58922

theorem fraction_equality (q r s t : ℚ) 
  (h1 : q / r = 9)
  (h2 : s / r = 6)
  (h3 : s / t = 1 / 2) :
  t / q = 4 / 3 := by
sorry

end fraction_equality_l589_58922


namespace jerrys_age_l589_58985

/-- Given that Mickey's age is 5 years more than 200% of Jerry's age,
    and Mickey is 21 years old, Jerry's age is 8 years. -/
theorem jerrys_age (mickey_age jerry_age : ℕ) : 
  mickey_age = 2 * jerry_age + 5 →
  mickey_age = 21 →
  jerry_age = 8 := by
sorry

end jerrys_age_l589_58985


namespace optimal_production_solution_l589_58976

/-- Represents the production problem with given parameters -/
structure ProductionProblem where
  total_units : ℕ
  workers : ℕ
  a_per_unit : ℕ
  b_per_unit : ℕ
  c_per_unit : ℕ
  a_per_worker : ℕ
  b_per_worker : ℕ
  c_per_worker : ℕ

/-- Calculates the completion time for a given worker distribution -/
def completion_time (prob : ProductionProblem) (x k : ℕ) : ℚ :=
  max (prob.a_per_unit * prob.total_units / (prob.a_per_worker * x : ℚ))
    (max (prob.b_per_unit * prob.total_units / (prob.b_per_worker * k * x : ℚ))
         (prob.c_per_unit * prob.total_units / (prob.c_per_worker * (prob.workers - (1 + k) * x) : ℚ)))

/-- The main theorem stating the optimal solution -/
theorem optimal_production_solution (prob : ProductionProblem) 
    (h_prob : prob.total_units = 3000 ∧ prob.workers = 200 ∧ 
              prob.a_per_unit = 2 ∧ prob.b_per_unit = 2 ∧ prob.c_per_unit = 1 ∧
              prob.a_per_worker = 6 ∧ prob.b_per_worker = 3 ∧ prob.c_per_worker = 2) :
    ∃ (x : ℕ), x > 0 ∧ x < prob.workers ∧ 
    completion_time prob x 2 = 250 / 11 ∧
    ∀ (y k : ℕ), y > 0 → y < prob.workers → k > 0 → 
    completion_time prob y k ≥ 250 / 11 := by
  sorry

end optimal_production_solution_l589_58976


namespace pump_problem_l589_58918

theorem pump_problem (x y : ℝ) 
  (h1 : x / 4 + y / 12 = 11)  -- Four pumps fill first tanker and 1/3 of second in 11 hours
  (h2 : x / 3 + y / 4 = 18)   -- Three pumps fill first tanker, one fills 1/4 of second in 18 hours
  : y / 3 = 8 :=              -- Three pumps fill second tanker in 8 hours
by sorry

end pump_problem_l589_58918


namespace pencils_found_l589_58958

theorem pencils_found (initial bought final misplaced broken : ℕ) : 
  initial = 20 →
  bought = 2 →
  final = 16 →
  misplaced = 7 →
  broken = 3 →
  final = initial - misplaced - broken + bought + (final - (initial - misplaced - broken + bought)) →
  final - (initial - misplaced - broken + bought) = 4 :=
by sorry

end pencils_found_l589_58958


namespace vasya_initial_larger_l589_58936

/-- Represents the initial investments and profit rates for Vasya and Petya --/
structure InvestmentScenario where
  vasya_initial : ℝ
  petya_initial : ℝ
  vasya_rate : ℝ
  petya_rate : ℝ
  exchange_rate_increase : ℝ

/-- Calculates the profit for a given initial investment and rate --/
def profit (initial : ℝ) (rate : ℝ) : ℝ := initial * rate

/-- Calculates Petya's effective rate considering exchange rate increase --/
def petya_effective_rate (petya_rate : ℝ) (exchange_rate_increase : ℝ) : ℝ :=
  1 + petya_rate + exchange_rate_increase + petya_rate * exchange_rate_increase

/-- Theorem stating that Vasya's initial investment is larger given equal profits --/
theorem vasya_initial_larger (scenario : InvestmentScenario) 
  (h1 : scenario.vasya_rate = 0.20)
  (h2 : scenario.petya_rate = 0.10)
  (h3 : scenario.exchange_rate_increase = 0.095)
  (h4 : profit scenario.vasya_initial scenario.vasya_rate = 
        profit scenario.petya_initial (petya_effective_rate scenario.petya_rate scenario.exchange_rate_increase)) :
  scenario.vasya_initial > scenario.petya_initial := by
  sorry


end vasya_initial_larger_l589_58936


namespace cosine_sum_special_case_l589_58952

theorem cosine_sum_special_case : 
  Real.cos (π/12) * Real.cos (π/6) - Real.sin (π/12) * Real.sin (π/6) = Real.sqrt 2 / 2 := by
  sorry

end cosine_sum_special_case_l589_58952


namespace sum_of_three_numbers_l589_58941

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 222) 
  (h2 : a*b + b*c + c*a = 131) : 
  a + b + c = 22 := by sorry

end sum_of_three_numbers_l589_58941


namespace triangle_problem_l589_58902

theorem triangle_problem (AB BC : ℝ) (θ : ℝ) (h t : ℝ) 
  (hyp1 : AB = 7)
  (hyp2 : BC = 25)
  (hyp3 : 100 * Real.sin θ = t)
  (hyp4 : h = AB * Real.sin θ) :
  t = 96 ∧ h = 168 / 25 := by
  sorry


end triangle_problem_l589_58902


namespace quadratic_roots_relation_l589_58961

theorem quadratic_roots_relation (b c : ℝ) : 
  (∃ r s : ℝ, 2 * r^2 - 4 * r - 10 = 0 ∧ 2 * s^2 - 4 * s - 10 = 0 ∧
   ∀ x : ℝ, x^2 + b * x + c = 0 ↔ (x = r - 3 ∨ x = s - 3)) →
  c = -2 := by
sorry

end quadratic_roots_relation_l589_58961


namespace souvenir_cost_problem_l589_58905

theorem souvenir_cost_problem (total_souvenirs : ℕ) (total_cost : ℚ) 
  (cheap_souvenirs : ℕ) (cheap_cost : ℚ) (expensive_souvenirs : ℕ) :
  total_souvenirs = 1000 →
  total_cost = 220 →
  cheap_souvenirs = 400 →
  cheap_cost = 1/4 →
  expensive_souvenirs = total_souvenirs - cheap_souvenirs →
  (total_cost - cheap_souvenirs * cheap_cost) / expensive_souvenirs = 1/5 := by
  sorry

end souvenir_cost_problem_l589_58905


namespace factors_of_539_l589_58903

theorem factors_of_539 : 
  ∃ (p q : Nat), p.Prime ∧ q.Prime ∧ p * q = 539 ∧ p = 13 ∧ q = 41 := by
  sorry

end factors_of_539_l589_58903


namespace four_heads_in_five_tosses_l589_58907

def n : ℕ := 5
def k : ℕ := 4
def p : ℚ := 1/2

def binomial_coefficient (n k : ℕ) : ℕ := 
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (binomial_coefficient n k : ℚ) * p^k * (1 - p)^(n - k)

theorem four_heads_in_five_tosses : 
  binomial_probability n k p = 5/32 := by sorry

end four_heads_in_five_tosses_l589_58907


namespace sum_and_divide_l589_58929

theorem sum_and_divide : (40 + 5) / 5 = 9 := by
  sorry

end sum_and_divide_l589_58929


namespace base_salary_minimum_l589_58979

/-- The base salary of Tom's new sales job -/
def base_salary : ℝ := 45000

/-- The salary of Tom's previous job -/
def previous_salary : ℝ := 75000

/-- The commission percentage on each sale -/
def commission_percentage : ℝ := 0.15

/-- The price of each sale -/
def sale_price : ℝ := 750

/-- The minimum number of sales required to not lose money -/
def min_sales : ℝ := 266.67

theorem base_salary_minimum : 
  base_salary + min_sales * (commission_percentage * sale_price) ≥ previous_salary :=
sorry

end base_salary_minimum_l589_58979


namespace cindys_math_operation_l589_58965

theorem cindys_math_operation (x : ℝ) : (x - 12) / 2 = 64 → (x - 6) / 4 = 33.5 := by
  sorry

end cindys_math_operation_l589_58965


namespace inequality_holds_iff_c_equals_one_l589_58991

theorem inequality_holds_iff_c_equals_one (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (∃ c : ℝ, c > 0 ∧ ∀ x y z : ℝ, x > 0 → y > 0 → z > 0 →
    (x^3 * y + y^3 * z + z^3 * x) / (x + y + z) + 4 * c / (x * y * z) ≥ 2 * c + 2) ↔
  (∃ c : ℝ, c = 1) :=
by sorry

end inequality_holds_iff_c_equals_one_l589_58991


namespace parabola_focus_l589_58917

/-- A parabola is defined by the equation x^2 = -8y -/
def parabola (x y : ℝ) : Prop := x^2 = -8*y

/-- The focus of a parabola is a point on its axis of symmetry -/
def is_focus (x y : ℝ) (p : ℝ → ℝ → Prop) : Prop :=
  ∀ (u v : ℝ), p u v → (x = 0 ∧ y = -2)

/-- Theorem: The focus of the parabola x^2 = -8y is located at (0, -2) -/
theorem parabola_focus :
  is_focus 0 (-2) parabola :=
sorry

end parabola_focus_l589_58917


namespace f_plus_g_positive_implies_m_bound_l589_58993

/-- The base of the natural logarithm -/
noncomputable def e : ℝ := Real.exp 1

/-- The function f(x) = e^x / x -/
noncomputable def f (x : ℝ) : ℝ := (Real.exp x) / x

/-- The function g(x) = mx -/
def g (m : ℝ) (x : ℝ) : ℝ := m * x

/-- Theorem stating that if f(x) + g(x) > 0 for all x > 0, then m > -e^2/4 -/
theorem f_plus_g_positive_implies_m_bound (m : ℝ) :
  (∀ x : ℝ, x > 0 → f x + g m x > 0) →
  m > -(e^2 / 4) := by
  sorry

end f_plus_g_positive_implies_m_bound_l589_58993


namespace sector_arc_length_l589_58911

theorem sector_arc_length (θ : Real) (r : Real) (h1 : θ = 90) (h2 : r = 6) :
  (θ / 360) * (2 * Real.pi * r) = 3 * Real.pi :=
sorry

end sector_arc_length_l589_58911


namespace parallel_line_theorem_perpendicular_line_theorem_l589_58983

-- Define the line l
def line_l (x y : ℝ) : Prop := 3 * x + 4 * y + 1 = 0

-- Define point A
def point_A : ℝ × ℝ := (1, 2)

-- Define line l1
def line_l1 (x y : ℝ) : Prop := 3 * x + 4 * y - 11 = 0

-- Define line l2
def line_l2 (x y : ℝ) : Prop := 4 * x - 3 * y + 2 = 0

-- Theorem for parallel line l1
theorem parallel_line_theorem :
  (∀ x y : ℝ, line_l1 x y ↔ 3 * x + 4 * y - 11 = 0) ∧
  (line_l1 (point_A.1) (point_A.2)) ∧
  (∃ k : ℝ, k ≠ 0 ∧ ∀ x y : ℝ, line_l x y ↔ line_l1 (k * x) (k * y)) :=
sorry

-- Theorem for perpendicular line l2
theorem perpendicular_line_theorem :
  (∀ x y : ℝ, line_l2 x y ↔ 4 * x - 3 * y + 2 = 0) ∧
  (line_l2 (point_A.1) (point_A.2)) ∧
  (∀ x1 y1 x2 y2 : ℝ, line_l x1 y1 → line_l x2 y2 →
    3 * (x2 - x1) + 4 * (y2 - y1) = 0 →
    4 * (x2 - x1) - 3 * (y2 - y1) = 0) :=
sorry

end parallel_line_theorem_perpendicular_line_theorem_l589_58983


namespace percent_problem_l589_58995

theorem percent_problem (x : ℝ) (h : 0.4 * x = 160) : 0.5 * x = 200 := by
  sorry

end percent_problem_l589_58995


namespace complex_fraction_simplification_l589_58910

theorem complex_fraction_simplification :
  (2 - I) / (2 + I) = 3/5 - 4/5 * I :=
by sorry

end complex_fraction_simplification_l589_58910


namespace even_periodic_increasing_function_inequality_l589_58957

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def has_period_two (f : ℝ → ℝ) : Prop := ∀ x, f (x + 2) = f x

def increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x < f y

theorem even_periodic_increasing_function_inequality (f : ℝ → ℝ)
  (h_even : is_even f)
  (h_period : has_period_two f)
  (h_increasing : increasing_on f (-1) 0) :
  f 3 < f (Real.sqrt 2) ∧ f (Real.sqrt 2) < f 2 :=
sorry

end even_periodic_increasing_function_inequality_l589_58957


namespace geometric_sequence_sum_l589_58994

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 1 + a 2 = 20) →
  (a 3 + a 4 = 40) →
  (a 5 + a 6 = 80) :=
by sorry

end geometric_sequence_sum_l589_58994


namespace series_sum_l589_58934

/-- The sum of the infinite series ∑(k=1 to ∞) k/4^k is equal to 4/9 -/
theorem series_sum : ∑' k, k / (4 : ℝ) ^ k = 4 / 9 := by
  sorry

end series_sum_l589_58934


namespace farmer_max_profit_l589_58942

/-- Represents the farmer's problem of maximizing profit given land and budget constraints -/
theorem farmer_max_profit (total_land : ℝ) (rice_yield peanut_yield : ℝ) 
  (rice_cost peanut_cost : ℝ) (rice_price peanut_price : ℝ) (budget : ℝ) :
  total_land = 2 →
  rice_yield = 6000 →
  peanut_yield = 1500 →
  rice_cost = 3600 →
  peanut_cost = 1200 →
  rice_price = 3 →
  peanut_price = 5 →
  budget = 6000 →
  ∃ (rice_area peanut_area : ℝ),
    rice_area = 1.5 ∧
    peanut_area = 0.5 ∧
    rice_area + peanut_area ≤ total_land ∧
    rice_cost * rice_area + peanut_cost * peanut_area ≤ budget ∧
    ∀ (x y : ℝ),
      x + y ≤ total_land →
      rice_cost * x + peanut_cost * y ≤ budget →
      (rice_price * rice_yield - rice_cost) * x + (peanut_price * peanut_yield - peanut_cost) * y ≤
      (rice_price * rice_yield - rice_cost) * rice_area + (peanut_price * peanut_yield - peanut_cost) * peanut_area :=
by
  sorry


end farmer_max_profit_l589_58942


namespace iron_bucket_area_l589_58944

/-- The area of iron sheet needed for a rectangular bucket -/
def bucket_area (length width height : ℝ) : ℝ :=
  length * width + 2 * (length * height + width * height)

/-- Theorem: The area of iron sheet needed for the specified bucket is 1.24 square meters -/
theorem iron_bucket_area :
  let length : ℝ := 0.4
  let width : ℝ := 0.3
  let height : ℝ := 0.8
  bucket_area length width height = 1.24 := by
  sorry


end iron_bucket_area_l589_58944


namespace union_covers_reals_a_equals_complement_b_l589_58925

open Set Real

-- Define the sets A and B
def A (m : ℝ) : Set ℝ := {x | -m ≤ x - 2 ∧ x - 2 ≤ m}
def B : Set ℝ := {x | x ≤ -2 ∨ x ≥ 4}

-- Part 1: A ∪ B = ℝ iff m ≥ 4
theorem union_covers_reals (m : ℝ) : A m ∪ B = univ ↔ m ≥ 4 := by sorry

-- Part 2: A = ℝ\B iff 0 < m < 2
theorem a_equals_complement_b (m : ℝ) : A m = Bᶜ ↔ 0 < m ∧ m < 2 := by sorry

end union_covers_reals_a_equals_complement_b_l589_58925


namespace find_x_l589_58974

theorem find_x (p q r x : ℝ) 
  (h1 : (p + q + r) / 3 = 4) 
  (h2 : (p + q + r + x) / 4 = 5) : 
  x = 8 := by sorry

end find_x_l589_58974


namespace square_root_of_four_l589_58926

theorem square_root_of_four : ∀ x : ℝ, x^2 = 4 ↔ x = 2 ∨ x = -2 := by sorry

end square_root_of_four_l589_58926


namespace trigonometric_product_equality_l589_58990

theorem trigonometric_product_equality : 
  3.420 * Real.sin (10 * π / 180) * Real.sin (20 * π / 180) * Real.sin (30 * π / 180) * 
  Real.sin (40 * π / 180) * Real.sin (50 * π / 180) * Real.sin (60 * π / 180) * 
  Real.sin (70 * π / 180) * Real.sin (80 * π / 180) = 3 / 256 := by
  sorry

end trigonometric_product_equality_l589_58990


namespace train_stoppage_time_l589_58906

/-- Given a train with speeds excluding and including stoppages, 
    calculate the number of minutes the train stops per hour. -/
theorem train_stoppage_time (speed_without_stops speed_with_stops : ℝ) : 
  speed_without_stops = 48 → speed_with_stops = 36 → 
  (speed_without_stops - speed_with_stops) / speed_without_stops * 60 = 15 := by
  sorry

#check train_stoppage_time

end train_stoppage_time_l589_58906


namespace park_short_bushes_after_planting_l589_58931

/-- The number of short bushes in a park after planting new ones. -/
def total_short_bushes (initial : ℕ) (planted : ℕ) : ℕ :=
  initial + planted

/-- Theorem stating that the total number of short bushes after planting is 57. -/
theorem park_short_bushes_after_planting :
  total_short_bushes 37 20 = 57 := by
  sorry

end park_short_bushes_after_planting_l589_58931


namespace polynomial_parity_l589_58966

/-- Represents a polynomial with integer coefficients -/
def IntPolynomial := List Int

/-- Multiplies two polynomials -/
def polyMult (p q : IntPolynomial) : IntPolynomial := sorry

/-- Checks if all elements in a list are even -/
def allEven (l : List Int) : Prop := ∀ x ∈ l, Even x

/-- Checks if all elements in a list are multiples of 4 -/
def allMultiplesOf4 (l : List Int) : Prop := ∀ x ∈ l, ∃ k, x = 4 * k

/-- Checks if at least one element in a list is odd -/
def hasOdd (l : List Int) : Prop := ∃ x ∈ l, Odd x

theorem polynomial_parity (P Q : IntPolynomial) :
  (allEven (polyMult P Q)) ∧ ¬(allMultiplesOf4 (polyMult P Q)) →
  ((allEven P ∧ hasOdd Q) ∨ (allEven Q ∧ hasOdd P)) := by
  sorry

end polynomial_parity_l589_58966


namespace stamp_collection_theorem_l589_58971

def stamp_collection_value (total_stamps : ℕ) (sample_stamps : ℕ) (sample_value : ℕ) (bonus_per_set : ℕ) : ℕ :=
  let stamp_value : ℕ := sample_value / sample_stamps
  let total_value : ℕ := total_stamps * stamp_value
  let complete_sets : ℕ := total_stamps / sample_stamps
  let bonus : ℕ := complete_sets * bonus_per_set
  total_value + bonus

theorem stamp_collection_theorem :
  stamp_collection_value 21 7 28 5 = 99 := by
  sorry

end stamp_collection_theorem_l589_58971


namespace sandbox_ratio_l589_58986

/-- A rectangular sandbox with specific dimensions. -/
structure Sandbox where
  width : ℝ
  length : ℝ
  perimeter : ℝ
  length_multiple : ℝ
  width_eq : width = 5
  perimeter_eq : perimeter = 30
  length_eq : length = length_multiple * width

/-- The ratio of length to width for a sandbox with given properties is 2:1. -/
theorem sandbox_ratio (s : Sandbox) : s.length / s.width = 2 := by
  sorry


end sandbox_ratio_l589_58986


namespace henry_initial_games_count_l589_58950

/-- The number of games Henry had initially -/
def henry_initial_games : ℕ := 33

/-- The number of games Neil had initially -/
def neil_initial_games : ℕ := 2

/-- The number of games Henry gave to Neil -/
def games_given : ℕ := 5

theorem henry_initial_games_count : 
  henry_initial_games = 33 :=
by
  have h1 : henry_initial_games - games_given = 4 * (neil_initial_games + games_given) :=
    sorry
  sorry

#check henry_initial_games_count

end henry_initial_games_count_l589_58950


namespace adam_initial_amount_l589_58964

/-- The cost of the airplane in dollars -/
def airplane_cost : ℚ := 4.28

/-- The change Adam receives after buying the airplane in dollars -/
def change_received : ℚ := 0.72

/-- Adam's initial amount of money in dollars -/
def initial_amount : ℚ := airplane_cost + change_received

theorem adam_initial_amount :
  initial_amount = 5 :=
by sorry

end adam_initial_amount_l589_58964


namespace warden_citations_l589_58968

theorem warden_citations (total : ℕ) (littering off_leash parking : ℕ) : 
  total = 24 ∧ 
  littering = off_leash ∧ 
  parking = 2 * (littering + off_leash) ∧ 
  total = littering + off_leash + parking →
  littering = 4 := by
sorry

end warden_citations_l589_58968


namespace circuit_equation_l589_58998

/-- Given voltage and impedance, prove the current satisfies the equation V = IZ -/
theorem circuit_equation (V Z I : ℂ) (hV : V = 2 + 3*I) (hZ : Z = 2 - I) : 
  V = I * Z ↔ I = (1 : ℝ)/5 + (8 : ℝ)/5 * I :=
sorry

end circuit_equation_l589_58998


namespace f_min_max_l589_58954

-- Define the function
def f (x : ℝ) : ℝ := -x^3 + 3*x + 2

-- State the theorem
theorem f_min_max :
  (∃ x₁ : ℝ, f x₁ = -1 ∧ ∀ x : ℝ, f x ≥ -1) ∧
  (∃ x₂ : ℝ, f x₂ = 3 ∧ ∀ x : ℝ, f x ≤ 3) := by
  sorry

end f_min_max_l589_58954


namespace mean_score_is_74_9_l589_58924

structure ScoreDistribution where
  score : ℕ
  num_students : ℕ

def total_students : ℕ := 100

def score_data : List ScoreDistribution := [
  ⟨100, 10⟩,
  ⟨90, 15⟩,
  ⟨80, 20⟩,
  ⟨70, 30⟩,
  ⟨60, 20⟩,
  ⟨50, 4⟩,
  ⟨40, 1⟩
]

def sum_scores : ℕ := (score_data.map (λ x => x.score * x.num_students)).sum

theorem mean_score_is_74_9 : 
  (sum_scores : ℚ) / total_students = 749 / 10 := by
  sorry

end mean_score_is_74_9_l589_58924


namespace cone_volume_with_inscribed_square_l589_58967

/-- The volume of a cone with a square inscribed in its base --/
theorem cone_volume_with_inscribed_square (a α : ℝ) (h_a : a > 0) (h_α : 0 < α ∧ α < π) :
  let r := a * Real.sqrt 2 / 2
  let h := a * Real.sqrt (Real.cos α) / (2 * Real.sin (α/2) ^ 2)
  π * r^2 * h / 3 = π * a^3 * Real.sqrt (Real.cos α) / (12 * Real.sin (α/2) ^ 2) :=
by sorry

end cone_volume_with_inscribed_square_l589_58967


namespace problem_solving_distribution_l589_58939

theorem problem_solving_distribution (x y z : ℕ) : 
  x + y + z = 100 →  -- Total problems
  x + 2*y + 3*z = 180 →  -- Sum of problems solved by each person
  x - z = 20  -- Difference between difficult and easy problems
:= by sorry

end problem_solving_distribution_l589_58939


namespace john_puppy_profit_l589_58999

/-- Calculates the profit from selling puppies given the initial conditions --/
def puppy_profit (initial_puppies : ℕ) (sale_price : ℕ) (stud_fee : ℕ) : ℕ :=
  let remaining_after_giving_away := initial_puppies / 2
  let remaining_after_keeping_one := remaining_after_giving_away - 1
  let total_sales := remaining_after_keeping_one * sale_price
  total_sales - stud_fee

/-- Proves that John's profit from selling puppies is $1500 --/
theorem john_puppy_profit : puppy_profit 8 600 300 = 1500 := by
  sorry

end john_puppy_profit_l589_58999


namespace line_points_property_l589_58930

theorem line_points_property (x₁ x₂ x₃ y₁ y₂ y₃ : ℝ) :
  y₁ = -2 * x₁ + 3 →
  y₂ = -2 * x₂ + 3 →
  y₃ = -2 * x₃ + 3 →
  x₁ < x₂ →
  x₂ < x₃ →
  x₂ * x₃ < 0 →
  y₁ * y₂ > 0 := by
  sorry

end line_points_property_l589_58930


namespace banana_arrangements_l589_58959

/-- The number of distinct arrangements of letters in a word -/
def distinctArrangements (totalLetters : ℕ) (repetitions : List ℕ) : ℕ :=
  Nat.factorial totalLetters / (repetitions.map Nat.factorial).prod

/-- Proof that the number of distinct arrangements of "BANANA" is 60 -/
theorem banana_arrangements :
  distinctArrangements 6 [3, 2, 1] = 60 := by
  sorry

end banana_arrangements_l589_58959


namespace least_multiplier_for_perfect_square_l589_58933

def original_number : ℕ := 2^5 * 3^6 * 4^3 * 5^3 * 6^7

theorem least_multiplier_for_perfect_square :
  ∀ n : ℕ, n > 0 → (∃ m : ℕ, (original_number * n) = m^2) →
  15 ≤ n :=
by sorry

end least_multiplier_for_perfect_square_l589_58933


namespace banana_split_difference_l589_58963

/-- The number of ice cream scoops in Oli's banana split -/
def oli_scoops : ℕ := 4

/-- The number of ice cream scoops in Victoria's banana split -/
def victoria_scoops : ℕ := 2 * oli_scoops + oli_scoops

/-- The number of ice cream scoops in Brian's banana split -/
def brian_scoops : ℕ := oli_scoops + 3

/-- The total difference in scoops of ice cream between Oli's, Victoria's, and Brian's banana splits -/
def total_difference : ℕ := 
  (victoria_scoops - oli_scoops) + (brian_scoops - oli_scoops) + (victoria_scoops - brian_scoops)

theorem banana_split_difference : total_difference = 16 := by
  sorry

end banana_split_difference_l589_58963


namespace weight_of_A_l589_58937

theorem weight_of_A (A B C D : ℝ) : 
  (A + B + C) / 3 = 84 →
  (A + B + C + D) / 4 = 80 →
  (B + C + D + (D + 8)) / 4 = 79 →
  A = 80 :=
by sorry

end weight_of_A_l589_58937


namespace number_equation_solution_l589_58938

theorem number_equation_solution :
  ∃ x : ℝ, 5.4 * x + 0.6 = 108.45000000000003 ∧ x = 19.97222222222222 :=
by sorry

end number_equation_solution_l589_58938


namespace vector_equation_solution_l589_58980

theorem vector_equation_solution :
  ∃ (a b : ℚ),
    (2 : ℚ) * a + (-2 : ℚ) * b = 10 ∧
    (3 : ℚ) * a + (5 : ℚ) * b = -8 ∧
    a = 17/8 ∧ b = -23/8 := by
  sorry

end vector_equation_solution_l589_58980


namespace sum_of_abc_l589_58923

theorem sum_of_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a * b = 36) (hac : a * c = 72) (hbc : b * c = 108) :
  a + b + c = 11 * Real.sqrt 6 := by
sorry

end sum_of_abc_l589_58923


namespace time_after_2700_minutes_l589_58984

-- Define a custom type for time
structure Time where
  hours : Nat
  minutes : Nat
  deriving Repr

-- Define a function to add minutes to a given time
def addMinutes (t : Time) (m : Nat) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + m
  let newHours := (totalMinutes / 60) % 24
  let newMinutes := totalMinutes % 60
  { hours := newHours, minutes := newMinutes }

-- Define the starting time (6:00 a.m.)
def startTime : Time := { hours := 6, minutes := 0 }

-- Define the number of minutes to add
def minutesToAdd : Nat := 2700

-- Define the expected end time (3:00 a.m. the next day)
def expectedEndTime : Time := { hours := 3, minutes := 0 }

-- Theorem statement
theorem time_after_2700_minutes :
  addMinutes startTime minutesToAdd = expectedEndTime := by
  sorry

end time_after_2700_minutes_l589_58984


namespace postage_cost_theorem_l589_58916

/-- The floor function, representing the greatest integer less than or equal to x -/
def floor (x : ℝ) : ℤ := sorry

/-- The cost in cents for mailing a letter weighing W ounces -/
def postageCost (W : ℝ) : ℤ := sorry

theorem postage_cost_theorem (W : ℝ) : 
  postageCost W = -6 * floor (-W) :=
sorry

end postage_cost_theorem_l589_58916


namespace mixed_number_multiplication_l589_58960

theorem mixed_number_multiplication (a b c d e f : ℚ) :
  a + b / c = -3 ∧ b / c = 3 / 4 ∧ d / e = 5 / 7 →
  (a + b / c) * (d / e) = (a - b / c) * (d / e) := by
  sorry

end mixed_number_multiplication_l589_58960


namespace sum_surface_areas_of_cut_cube_l589_58940

/-- The sum of surface areas of cuboids resulting from cutting a unit cube -/
theorem sum_surface_areas_of_cut_cube : 
  let n : ℕ := 4  -- number of divisions per side
  let num_cuboids : ℕ := n^3
  let side_length : ℚ := 1 / n
  let surface_area_one_cuboid : ℚ := 6 * side_length^2
  surface_area_one_cuboid * num_cuboids = 24 := by sorry

end sum_surface_areas_of_cut_cube_l589_58940


namespace periodic_decimal_to_fraction_l589_58962

theorem periodic_decimal_to_fraction :
  (0.02 : ℚ) = 2 / 99 →
  (2.06 : ℚ) = 68 / 33 := by
sorry

end periodic_decimal_to_fraction_l589_58962


namespace simplify_expression_l589_58972

theorem simplify_expression (r : ℝ) : 100*r - 48*r + 10 = 52*r + 10 := by
  sorry

end simplify_expression_l589_58972


namespace seating_arrangements_count_l589_58988

def front_seats : Nat := 4
def back_seats : Nat := 5
def people_to_seat : Nat := 2

def is_adjacent (row1 row2 seat1 seat2 : Nat) : Bool :=
  (row1 = row2 ∧ seat2 = seat1 + 1) ∨
  (row1 = 1 ∧ row2 = 2 ∧ (seat1 = seat2 ∨ seat1 + 1 = seat2))

def count_seating_arrangements : Nat :=
  let total_seats := front_seats + back_seats
  (total_seats.choose people_to_seat) -
  (front_seats - 1 + back_seats - 1 + front_seats)

theorem seating_arrangements_count :
  count_seating_arrangements = 58 := by
  sorry

end seating_arrangements_count_l589_58988


namespace rare_coin_value_l589_58973

/-- Given a collection of rare coins where 4 coins are worth 16 dollars, 
    prove that 20 coins of the same type are worth 80 dollars. -/
theorem rare_coin_value (total_coins : ℕ) (sample_coins : ℕ) (sample_value : ℚ) :
  total_coins = 20 →
  sample_coins = 4 →
  sample_value = 16 →
  (total_coins : ℚ) * (sample_value / sample_coins) = 80 :=
by sorry

end rare_coin_value_l589_58973


namespace alice_number_sum_l589_58935

/-- Represents the process of subtracting the smallest prime divisor from a number -/
def subtractSmallestPrimeDivisor (n : ℕ) : ℕ := sorry

/-- Returns true if the number is prime, false otherwise -/
def isPrime (n : ℕ) : Prop := sorry

theorem alice_number_sum : 
  ∀ n : ℕ, 
  n > 0 → 
  (isPrime (n.iterate subtractSmallestPrimeDivisor 2022)) → 
  (n = 4046 ∨ n = 4047) ∧ 
  (4046 + 4047 = 8093) := 
sorry

end alice_number_sum_l589_58935


namespace rhombus_area_l589_58912

/-- The area of a rhombus with diagonals of 6cm and 8cm is 24cm². -/
theorem rhombus_area (d1 d2 : ℝ) (h1 : d1 = 6) (h2 : d2 = 8) :
  (1 / 2) * d1 * d2 = 24 := by
  sorry

end rhombus_area_l589_58912


namespace negation_of_universal_quantifier_l589_58997

theorem negation_of_universal_quantifier (x : ℝ) :
  (¬ ∀ m : ℝ, m ∈ Set.Icc 0 1 → x + 1 / x ≥ 2^m) ↔
  (∃ m : ℝ, m ∈ Set.Icc 0 1 ∧ x + 1 / x < 2^m) :=
by sorry

end negation_of_universal_quantifier_l589_58997


namespace line_plane_perpendicular_parallel_l589_58951

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem line_plane_perpendicular_parallel 
  (l m : Line) (α : Plane) : 
  perpendicular l α → parallel l m → perpendicular m α :=
sorry

end line_plane_perpendicular_parallel_l589_58951


namespace figure_area_is_74_l589_58975

/-- Represents the dimensions of the composite rectangular figure -/
structure FigureDimensions where
  height : ℕ
  width1 : ℕ
  width2 : ℕ
  width3 : ℕ
  height2 : ℕ
  height3 : ℕ

/-- Calculates the area of the composite rectangular figure -/
def calculateArea (d : FigureDimensions) : ℕ :=
  d.height * d.width1 + 
  (d.height - d.height2) * d.width2 +
  d.height2 * d.width2 +
  (d.height - d.height3) * d.width3

/-- Theorem stating that the area of the figure with given dimensions is 74 square units -/
theorem figure_area_is_74 (d : FigureDimensions) 
  (h1 : d.height = 7)
  (h2 : d.width1 = 6)
  (h3 : d.width2 = 4)
  (h4 : d.width3 = 5)
  (h5 : d.height2 = 2)
  (h6 : d.height3 = 6) :
  calculateArea d = 74 := by
  sorry

#eval calculateArea { height := 7, width1 := 6, width2 := 4, width3 := 5, height2 := 2, height3 := 6 }

end figure_area_is_74_l589_58975


namespace parallel_vectors_imply_x_l589_58914

/-- Two vectors in ℝ² are parallel if and only if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_imply_x (x : ℝ) :
  let a : ℝ × ℝ := (1, 2*x + 1)
  let b : ℝ × ℝ := (2, 3)
  parallel a b → x = 1/4 := by sorry

end parallel_vectors_imply_x_l589_58914


namespace parking_lot_theorem_l589_58970

/-- A multi-story parking lot with equal-sized levels -/
structure ParkingLot where
  total_spaces : ℕ
  num_levels : ℕ
  cars_on_one_level : ℕ

/-- Calculates the number of additional cars that can fit on one level -/
def additional_cars (p : ParkingLot) : ℕ :=
  (p.total_spaces / p.num_levels) - p.cars_on_one_level

theorem parking_lot_theorem (p : ParkingLot) 
  (h1 : p.total_spaces = 425)
  (h2 : p.num_levels = 5)
  (h3 : p.cars_on_one_level = 23) :
  additional_cars p = 62 := by
  sorry

#eval additional_cars { total_spaces := 425, num_levels := 5, cars_on_one_level := 23 }

end parking_lot_theorem_l589_58970


namespace y_value_l589_58982

theorem y_value (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 16) : y = 4 := by
  sorry

end y_value_l589_58982


namespace complex_magnitude_range_l589_58927

theorem complex_magnitude_range (z : ℂ) (h : Complex.abs z = 1) :
  4 * Real.sqrt 2 ≤ Complex.abs ((z + 1) + Complex.I * (7 - z)) ∧
  Complex.abs ((z + 1) + Complex.I * (7 - z)) ≤ 6 * Real.sqrt 2 :=
by sorry

end complex_magnitude_range_l589_58927


namespace f_has_minimum_l589_58956

def f (x : ℝ) := |2*x + 1| - |x - 4|

theorem f_has_minimum : ∃ m : ℝ, (∀ x : ℝ, f x ≥ m) ∧ (∃ x : ℝ, f x = m) := by
  sorry

end f_has_minimum_l589_58956


namespace wire_length_proof_l589_58943

theorem wire_length_proof (side_length : ℝ) (total_area : ℝ) (original_length : ℝ) : 
  side_length = 2 →
  total_area = 92 →
  original_length = (total_area / (side_length ^ 2)) * (4 * side_length) →
  original_length = 184 := by
  sorry

#check wire_length_proof

end wire_length_proof_l589_58943


namespace mary_has_29_nickels_l589_58977

/-- Calculates the total number of nickels Mary has after receiving gifts and doing chores. -/
def marys_nickels (initial : ℕ) (from_dad : ℕ) (mom_multiplier : ℕ) (from_chores : ℕ) : ℕ :=
  initial + from_dad + (mom_multiplier * from_dad) + from_chores

/-- Theorem stating that Mary has 29 nickels after all transactions. -/
theorem mary_has_29_nickels : 
  marys_nickels 7 5 3 2 = 29 := by
  sorry

end mary_has_29_nickels_l589_58977


namespace shirt_sale_tax_percentage_l589_58989

theorem shirt_sale_tax_percentage : 
  let num_fandoms : ℕ := 4
  let shirts_per_fandom : ℕ := 5
  let original_price : ℚ := 15
  let discount_percentage : ℚ := 20 / 100
  let total_paid : ℚ := 264

  let discounted_price : ℚ := original_price * (1 - discount_percentage)
  let total_shirts : ℕ := num_fandoms * shirts_per_fandom
  let total_cost_before_tax : ℚ := discounted_price * total_shirts
  let tax_amount : ℚ := total_paid - total_cost_before_tax
  let tax_percentage : ℚ := tax_amount / total_cost_before_tax * 100

  tax_percentage = 10 := by sorry

end shirt_sale_tax_percentage_l589_58989


namespace solution_set_f_leq_0_range_of_m_l589_58953

-- Define the function f
def f (x : ℝ) : ℝ := |x - 2| - |2*x + 1|

-- Theorem for the solution set of f(x) ≤ 0
theorem solution_set_f_leq_0 :
  {x : ℝ | f x ≤ 0} = {x : ℝ | x ≥ 1/3 ∨ x ≤ -3} :=
sorry

-- Theorem for the range of m
theorem range_of_m :
  {m : ℝ | ∀ x, f x - 2*m^2 ≤ 4*m} = {m : ℝ | m ≤ -5/2 ∨ m ≥ 1/2} :=
sorry

end solution_set_f_leq_0_range_of_m_l589_58953


namespace initial_daily_consumption_l589_58921

/-- Proves that the initial daily consumption per soldier is 3 kg -/
theorem initial_daily_consumption (initial_soldiers : ℕ) (initial_days : ℕ) 
  (new_soldiers : ℕ) (new_days : ℕ) (new_consumption : ℚ) : 
  initial_soldiers = 1200 →
  initial_days = 30 →
  new_soldiers = 528 →
  new_days = 25 →
  new_consumption = 5/2 →
  (initial_soldiers * initial_days * (3 : ℚ) = 
   (initial_soldiers + new_soldiers) * new_days * new_consumption) := by
  sorry

end initial_daily_consumption_l589_58921


namespace sum_of_roots_quadratic_l589_58969

theorem sum_of_roots_quadratic (x : ℝ) : (x + 3) * (x - 4) = 20 → ∃ y : ℝ, (y + 3) * (y - 4) = 20 ∧ x + y = 1 := by
  sorry

end sum_of_roots_quadratic_l589_58969


namespace kayla_total_items_l589_58987

/-- Represents the number of items bought by a person -/
structure Items :=
  (chocolate_bars : ℕ)
  (soda_cans : ℕ)

/-- The total number of items -/
def Items.total (i : Items) : ℕ := i.chocolate_bars + i.soda_cans

/-- Theresa bought twice the number of items as Kayla -/
def twice (kayla : Items) (theresa : Items) : Prop :=
  theresa.chocolate_bars = 2 * kayla.chocolate_bars ∧
  theresa.soda_cans = 2 * kayla.soda_cans

theorem kayla_total_items 
  (kayla theresa : Items)
  (h1 : twice kayla theresa)
  (h2 : theresa.chocolate_bars = 12)
  (h3 : theresa.soda_cans = 18) :
  kayla.total = 15 :=
by sorry

end kayla_total_items_l589_58987


namespace hyperbola_asymptote_slopes_l589_58900

/-- The hyperbola equation -/
def hyperbola_eq (x y : ℝ) : Prop :=
  (y - 1)^2 / 16 - (x + 2)^2 / 25 = 4

/-- The slopes of the asymptotes -/
def asymptote_slopes : Set ℝ := {0.8, -0.8}

/-- Theorem stating that the slopes of the asymptotes of the given hyperbola are ±0.8 -/
theorem hyperbola_asymptote_slopes :
  ∀ (x y : ℝ), hyperbola_eq x y → (∃ (m : ℝ), m ∈ asymptote_slopes ∧ 
    ∃ (b : ℝ), y = m * x + b) :=
sorry

end hyperbola_asymptote_slopes_l589_58900


namespace line_parabola_intersection_l589_58920

/-- The line x = my + 1 intersects the parabola y² = x at two distinct points for any real m -/
theorem line_parabola_intersection (m : ℝ) : 
  ∃ (y₁ y₂ : ℝ), y₁ ≠ y₂ ∧ 
  (y₁^2 = m * y₁ + 1) ∧ 
  (y₂^2 = m * y₂ + 1) := by
sorry

end line_parabola_intersection_l589_58920


namespace power_boat_travel_time_l589_58904

/-- The time it takes for the power boat to travel from A to B -/
def travel_time_AB : ℝ := 4

/-- The distance between dock A and dock B in km -/
def distance_AB : ℝ := 20

/-- The original speed of the river current -/
def river_speed : ℝ := sorry

/-- The speed of the power boat relative to the river -/
def boat_speed : ℝ := sorry

/-- The total time of the journey in hours -/
def total_time : ℝ := 12

theorem power_boat_travel_time :
  let increased_river_speed := 1.5 * river_speed
  let downstream_speed := boat_speed + river_speed
  let upstream_speed := boat_speed - increased_river_speed
  distance_AB / downstream_speed = travel_time_AB ∧
  distance_AB + upstream_speed * (total_time - travel_time_AB) = river_speed * total_time :=
by sorry

end power_boat_travel_time_l589_58904


namespace count_off_ones_l589_58932

theorem count_off_ones (n : ℕ) (h : n = 1994) : 
  (n / (Nat.lcm 3 4) : ℕ) = 166 := by
  sorry

end count_off_ones_l589_58932


namespace quadratic_one_root_l589_58955

/-- If the quadratic x^2 + 6mx + 2m has exactly one real root, then m = 2/9 -/
theorem quadratic_one_root (m : ℝ) : 
  (∃! x, x^2 + 6*m*x + 2*m = 0) → m = 2/9 := by
  sorry

end quadratic_one_root_l589_58955


namespace impossible_score_l589_58901

/-- Represents the score of a quiz -/
structure QuizScore where
  correct : ℕ
  unanswered : ℕ
  incorrect : ℕ
  total_questions : ℕ
  score : ℤ

/-- The quiz scoring system -/
def quiz_score (qs : QuizScore) : Prop :=
  qs.correct + qs.unanswered + qs.incorrect = qs.total_questions ∧
  qs.score = 5 * qs.correct + 2 * qs.unanswered - qs.incorrect

theorem impossible_score : 
  ∀ qs : QuizScore, 
  qs.total_questions = 25 → 
  quiz_score qs → 
  qs.score ≠ 127 := by
sorry

end impossible_score_l589_58901


namespace circle_through_points_l589_58945

/-- The general equation of a circle -/
def CircleEquation (x y D E F : ℝ) : Prop :=
  x^2 + y^2 + D*x + E*y + F = 0

/-- The specific circle equation we want to prove -/
def SpecificCircle (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 6*y = 0

/-- Theorem stating that the specific circle equation passes through the given points -/
theorem circle_through_points :
  (∀ D E F : ℝ, CircleEquation 0 0 D E F → CircleEquation 4 0 D E F → CircleEquation (-1) 1 D E F
    → ∀ x y : ℝ, CircleEquation x y D E F ↔ SpecificCircle x y) := by
  sorry

end circle_through_points_l589_58945


namespace grid_tiling_condition_l589_58981

/-- Represents a tile type that can cover a 2x2 or larger area of a grid -/
structure Tile :=
  (width : ℕ)
  (height : ℕ)
  (valid : width ≥ 2 ∧ height ≥ 2)

/-- Represents the set of 6 available tile types -/
def TileSet : Set Tile := sorry

/-- Predicate to check if a grid can be tiled with the given tile set -/
def canBeTiled (m n : ℕ) (tiles : Set Tile) : Prop := sorry

/-- Main theorem: A rectangular grid can be tiled iff 4 divides m or n, and neither is 1 -/
theorem grid_tiling_condition (m n : ℕ) :
  canBeTiled m n TileSet ↔ (4 ∣ m ∨ 4 ∣ n) ∧ m ≠ 1 ∧ n ≠ 1 :=
sorry

end grid_tiling_condition_l589_58981


namespace carbonic_acid_weight_l589_58913

/-- Atomic weight of Hydrogen in g/mol -/
def atomic_weight_H : ℝ := 1.008

/-- Atomic weight of Carbon in g/mol -/
def atomic_weight_C : ℝ := 12.011

/-- Atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 15.999

/-- Number of Hydrogen atoms in Carbonic acid -/
def num_H : ℕ := 2

/-- Number of Carbon atoms in Carbonic acid -/
def num_C : ℕ := 1

/-- Number of Oxygen atoms in Carbonic acid -/
def num_O : ℕ := 3

/-- Number of moles of Carbonic acid -/
def num_moles : ℝ := 8

/-- Molecular weight of Carbonic acid in g/mol -/
def molecular_weight_H2CO3 : ℝ := 
  num_H * atomic_weight_H + num_C * atomic_weight_C + num_O * atomic_weight_O

/-- Total weight of given moles of Carbonic acid in grams -/
def total_weight : ℝ := num_moles * molecular_weight_H2CO3

theorem carbonic_acid_weight : total_weight = 496.192 := by
  sorry

end carbonic_acid_weight_l589_58913


namespace average_age_proof_l589_58928

/-- Given three people a, b, and c, prove that if their average age is 25 years
    and b's age is 17 years, then the average age of a and c is 29 years. -/
theorem average_age_proof (a b c : ℕ) : 
  (a + b + c) / 3 = 25 → b = 17 → (a + c) / 2 = 29 := by
  sorry

end average_age_proof_l589_58928


namespace work_completion_time_l589_58908

/-- The time required for x to complete a work given the combined time of x and y, and the time for y alone. -/
theorem work_completion_time (combined_time y_time : ℝ) (h1 : combined_time > 0) (h2 : y_time > 0) :
  let x_rate := 1 / combined_time - 1 / y_time
  x_rate > 0 → 1 / x_rate = y_time := by
  sorry

end work_completion_time_l589_58908


namespace simplify_expression_l589_58915

theorem simplify_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^3 - b^3 = a - b) :
  a/b - b/a + 1/(a*b) = -1 + 2/(a*b) := by
  sorry

end simplify_expression_l589_58915


namespace tangent_line_to_circle_l589_58996

theorem tangent_line_to_circle (r : ℝ) (h1 : r > 0) : 
  (∃ (x y : ℝ), x + y = 2*r ∧ (x - 1)^2 + (y - 1)^2 = r^2 ∧ 
   ∀ (x' y' : ℝ), x' + y' = 2*r → (x' - 1)^2 + (y' - 1)^2 ≥ r^2) →
  r = 2 + Real.sqrt 2 := by
sorry

end tangent_line_to_circle_l589_58996


namespace candidate_a_votes_l589_58909

def total_votes : ℕ := 560000
def invalid_percentage : ℚ := 15 / 100
def candidate_a_percentage : ℚ := 80 / 100

theorem candidate_a_votes : 
  (1 - invalid_percentage) * candidate_a_percentage * total_votes = 380800 := by
  sorry

end candidate_a_votes_l589_58909


namespace find_y_l589_58978

def rotation_equivalence (y : ℝ) : Prop :=
  (480 % 360 : ℝ) = (360 - y) % 360 ∧ y < 360

theorem find_y : ∃ y : ℝ, rotation_equivalence y ∧ y = 240 := by
  sorry

end find_y_l589_58978


namespace shape_triangle_area_ratio_l589_58948

/-- A shape with a certain area -/
structure Shape where
  area : ℝ
  area_pos : area > 0

/-- A triangle with a certain area -/
structure Triangle where
  area : ℝ
  area_pos : area > 0

/-- The theorem stating the relationship between the areas of a shape and a triangle -/
theorem shape_triangle_area_ratio 
  (s : Shape) 
  (t : Triangle) 
  (h : s.area / t.area = 2) : 
  s.area = 2 * t.area := by
  sorry

end shape_triangle_area_ratio_l589_58948


namespace welders_left_correct_l589_58919

/-- The number of welders who left after the first day -/
def welders_who_left : ℕ := 9

/-- The initial number of welders -/
def initial_welders : ℕ := 12

/-- The number of days to complete the order with all welders -/
def initial_days : ℕ := 3

/-- The number of additional days needed after some welders left -/
def additional_days : ℕ := 8

theorem welders_left_correct :
  ∃ (r : ℝ), r > 0 ∧
  initial_welders * r * initial_days = (initial_welders - welders_who_left) * r * (1 + additional_days) :=
by sorry

end welders_left_correct_l589_58919


namespace min_value_theorem_l589_58992

/-- Given x > 0 and y > 0 satisfying ln(xy)^y = e^x, 
    the minimum value of x^2y - ln x - x is 1 -/
theorem min_value_theorem (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) 
  (h : Real.log (x * y) ^ y = Real.exp x) : 
  ∃ (z : ℝ), z = 1 ∧ ∀ (w : ℝ), x^2 * y - Real.log x - x ≥ w → z ≤ w :=
sorry

end min_value_theorem_l589_58992


namespace largest_inscribed_triangle_l589_58946

-- Define a convex polygon
def ConvexPolygon (M : Set (ℝ × ℝ)) : Prop := sorry

-- Define an inscribed triangle in a polygon
def InscribedTriangle (T : Set (ℝ × ℝ)) (M : Set (ℝ × ℝ)) : Prop := sorry

-- Define the area of a triangle
def TriangleArea (T : Set (ℝ × ℝ)) : ℝ := sorry

-- Define a triangle formed by three vertices of a polygon
def VertexTriangle (T : Set (ℝ × ℝ)) (M : Set (ℝ × ℝ)) : Prop := sorry

theorem largest_inscribed_triangle (M : Set (ℝ × ℝ)) (h : ConvexPolygon M) :
  ∃ (T : Set (ℝ × ℝ)), VertexTriangle T M ∧
    ∀ (S : Set (ℝ × ℝ)), InscribedTriangle S M → TriangleArea S ≤ TriangleArea T :=
sorry

end largest_inscribed_triangle_l589_58946

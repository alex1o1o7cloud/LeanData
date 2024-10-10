import Mathlib

namespace quadratic_vertex_form_l3608_360858

theorem quadratic_vertex_form (x : ℝ) : 
  ∃ (a k h : ℝ), 3 * x^2 + 9 * x + 20 = a * (x - h)^2 + k ∧ h = -3/2 := by
  sorry

end quadratic_vertex_form_l3608_360858


namespace papers_printed_proof_l3608_360804

theorem papers_printed_proof :
  let presses1 : ℕ := 40
  let presses2 : ℕ := 30
  let time1 : ℝ := 12
  let time2 : ℝ := 15.999999999999998
  let rate : ℝ := (presses2 * time2) / (presses1 * time1)
  presses1 * rate * time1 = 40 := by
  sorry

end papers_printed_proof_l3608_360804


namespace book_cost_price_l3608_360895

def cost_price : ℝ → Prop := λ c => 
  (c * 1.1 + 90 = c * 1.15) ∧ 
  (c > 0)

theorem book_cost_price : ∃ c, cost_price c ∧ c = 1800 := by
  sorry

end book_cost_price_l3608_360895


namespace largest_mersenne_prime_under_500_l3608_360851

def is_mersenne_prime (n : ℕ) : Prop :=
  ∃ p : ℕ, Prime p ∧ n = 2^p - 1 ∧ Prime n

theorem largest_mersenne_prime_under_500 : 
  (∀ n : ℕ, n < 500 → is_mersenne_prime n → n ≤ 127) ∧ 
  is_mersenne_prime 127 :=
sorry

end largest_mersenne_prime_under_500_l3608_360851


namespace min_value_ab_l3608_360839

theorem min_value_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a * b = a + 9 * b + 7) :
  a * b ≥ 49 := by
  sorry

end min_value_ab_l3608_360839


namespace puppy_and_food_cost_l3608_360886

/-- Calculates the total cost of a puppy and food for a given number of weeks -/
def totalCost (puppyCost : ℚ) (foodPerDay : ℚ) (daysSupply : ℕ) (cupPerBag : ℚ) (bagCost : ℚ) : ℚ :=
  let totalDays : ℕ := daysSupply
  let totalFood : ℚ := (totalDays : ℚ) * foodPerDay
  let bagsNeeded : ℚ := totalFood / cupPerBag
  let foodCost : ℚ := bagsNeeded * bagCost
  puppyCost + foodCost

/-- Theorem stating that the total cost of a puppy and food for 3 weeks is $14 -/
theorem puppy_and_food_cost :
  totalCost 10 (1/3) 21 (7/2) 2 = 14 := by
  sorry

end puppy_and_food_cost_l3608_360886


namespace max_glass_height_l3608_360859

/-- The maximum height of a truncated cone-shaped glass that can roll around a circular table without reaching the edge -/
theorem max_glass_height (table_diameter : Real) (glass_bottom_diameter : Real) (glass_top_diameter : Real)
  (h_table : table_diameter = 160)
  (h_glass_bottom : glass_bottom_diameter = 5)
  (h_glass_top : glass_top_diameter = 6.5) :
  ∃ (max_height : Real), 
    (∀ (h : Real), h > 0 ∧ h < max_height → 
      ∃ (x y : Real), x^2 + y^2 < (table_diameter/2)^2 ∧ 
        ((h * glass_bottom_diameter/2) / (glass_top_diameter/2 - glass_bottom_diameter/2))^2 + h^2 = 
        ((y - x) * (glass_top_diameter/2 - glass_bottom_diameter/2) / h)^2) ∧
    max_height < (3/13) * Real.sqrt 6389.4375 := by
  sorry

end max_glass_height_l3608_360859


namespace bubble_gum_cost_l3608_360847

theorem bubble_gum_cost (total_cost : ℕ) (total_pieces : ℕ) (cost_per_piece : ℕ) : 
  total_cost = 2448 → 
  total_pieces = 136 → 
  total_cost = total_pieces * cost_per_piece → 
  cost_per_piece = 18 := by
  sorry

end bubble_gum_cost_l3608_360847


namespace two_group_subcommittee_count_l3608_360809

theorem two_group_subcommittee_count :
  let total_people : ℕ := 8
  let group_a_size : ℕ := 5
  let group_b_size : ℕ := 3
  let subcommittee_size : ℕ := 2
  group_a_size + group_b_size = total_people →
  group_a_size * group_b_size = 15
  := by sorry

end two_group_subcommittee_count_l3608_360809


namespace cube_preserves_order_l3608_360808

theorem cube_preserves_order (a b : ℝ) : a > b → a^3 > b^3 := by
  sorry

end cube_preserves_order_l3608_360808


namespace alpha_beta_not_perfect_square_l3608_360853

/-- A polynomial of degree 4 with roots 0, αβ, βγ, and γα -/
def f (α β γ : ℕ) (x : ℤ) : ℤ := x * (x - α * β) * (x - β * γ) * (x - γ * α)

/-- Theorem: Given positive integers α, β, γ, and an integer s such that
    f(-1) = f(s)², αβ is not a perfect square. -/
theorem alpha_beta_not_perfect_square (α β γ : ℕ) (s : ℤ) 
    (hα : α > 0) (hβ : β > 0) (hγ : γ > 0)
    (h_eq : f α β γ (-1) = (f α β γ s)^2) :
    ¬ ∃ (k : ℕ), α * β = k^2 := by
  sorry

end alpha_beta_not_perfect_square_l3608_360853


namespace modulo_problem_l3608_360841

theorem modulo_problem (m : ℕ) : 
  (65 * 76 * 87 ≡ m [ZMOD 25]) → 
  (0 ≤ m ∧ m < 25) → 
  m = 5 := by
  sorry

end modulo_problem_l3608_360841


namespace intersection_P_Q_l3608_360840

-- Define the sets P and Q
def P : Set ℝ := {x | Real.log x > 0}
def Q : Set ℝ := {x | -1 < x ∧ x < 2}

-- Define the open interval (1, 2)
def open_interval_one_two : Set ℝ := {x | 1 < x ∧ x < 2}

-- State the theorem
theorem intersection_P_Q : P ∩ Q = open_interval_one_two := by sorry

end intersection_P_Q_l3608_360840


namespace solution_to_system_of_equations_l3608_360887

theorem solution_to_system_of_equations :
  ∃ (x y : ℝ),
    (1 / x + 1 / (2 * y) = (x^2 + 3 * y^2) * (3 * x^2 + y^2)) ∧
    (1 / x - 1 / (2 * y) = 2 * (y^4 - x^4)) ∧
    (x = (3^(1/5) + 1) / 2) ∧
    (y = (3^(1/5) - 1) / 2) := by
  sorry

end solution_to_system_of_equations_l3608_360887


namespace age_ratio_future_l3608_360892

/-- Given Alan's current age a and Bella's current age b, prove that the number of years
    until their age ratio is 3:2 is 7, given the conditions on their past ages. -/
theorem age_ratio_future (a b : ℕ) (h1 : a - 3 = 2 * (b - 3)) (h2 : a - 8 = 3 * (b - 8)) :
  ∃ x : ℕ, x = 7 ∧ (a + x) * 2 = (b + x) * 3 :=
sorry

end age_ratio_future_l3608_360892


namespace first_digit_base_9_of_628_l3608_360832

/-- The first digit of the base 9 representation of a number -/
def first_digit_base_9 (n : ℕ) : ℕ :=
  if n < 9 then n else first_digit_base_9 (n / 9)

/-- The number in base 10 -/
def number : ℕ := 628

theorem first_digit_base_9_of_628 :
  first_digit_base_9 number = 7 := by
  sorry

end first_digit_base_9_of_628_l3608_360832


namespace hilton_lost_marbles_l3608_360825

/-- Proves the number of marbles Hilton lost given the initial and final conditions -/
theorem hilton_lost_marbles (initial : ℕ) (found : ℕ) (final : ℕ) : 
  initial = 26 → found = 6 → final = 42 → 
  ∃ (lost : ℕ), lost = 10 ∧ final = initial + found - lost + 2 * lost := by
  sorry

end hilton_lost_marbles_l3608_360825


namespace simplify_sqrt_expression_l3608_360865

theorem simplify_sqrt_expression :
  Real.sqrt 7 - Real.sqrt 28 + Real.sqrt 63 = 2 * Real.sqrt 7 := by
  sorry

end simplify_sqrt_expression_l3608_360865


namespace andrew_work_days_l3608_360879

/-- Given that Andrew worked 2.5 hours each day and a total of 7.5 hours,
    prove that he spent 3 days working on the report. -/
theorem andrew_work_days (hours_per_day : ℝ) (total_hours : ℝ) 
    (h1 : hours_per_day = 2.5)
    (h2 : total_hours = 7.5) :
    total_hours / hours_per_day = 3 := by
  sorry

end andrew_work_days_l3608_360879


namespace triangle_angle_measure_l3608_360846

theorem triangle_angle_measure (A B C : ℝ) (a b c : ℝ) : 
  (∃ (p q : ℝ × ℝ), p = (1, -Real.sqrt 3) ∧ q = (Real.cos B, Real.sin B) ∧ p.1 * q.2 = p.2 * q.1) →
  b * Real.cos C + c * Real.cos B = 2 * a * Real.sin A →
  A + B + C = Real.pi →
  C = Real.pi / 6 := by
sorry

end triangle_angle_measure_l3608_360846


namespace constant_term_binomial_expansion_l3608_360869

theorem constant_term_binomial_expansion :
  let n : ℕ := 8
  let a : ℚ := 1
  let b : ℚ := -2/3
  let r : ℕ := 6
  (Nat.choose n r) * a^(n-r) * b^r = 1792 := by sorry

end constant_term_binomial_expansion_l3608_360869


namespace sufficient_condition_implies_conditional_l3608_360815

-- Define p and q as propositions
variable (p q : Prop)

-- Define what it means for p to be a sufficient condition for q
def is_sufficient_condition (p q : Prop) : Prop :=
  p → q

-- Theorem statement
theorem sufficient_condition_implies_conditional 
  (h : is_sufficient_condition p q) : (p → q) = True :=
sorry

end sufficient_condition_implies_conditional_l3608_360815


namespace tom_batteries_total_l3608_360878

/-- The total number of batteries Tom used is 19, given the number of batteries used for each category. -/
theorem tom_batteries_total (flashlight_batteries : ℕ) (toy_batteries : ℕ) (controller_batteries : ℕ)
  (h1 : flashlight_batteries = 2)
  (h2 : toy_batteries = 15)
  (h3 : controller_batteries = 2) :
  flashlight_batteries + toy_batteries + controller_batteries = 19 := by
  sorry

end tom_batteries_total_l3608_360878


namespace parallel_vectors_x_value_l3608_360836

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

/-- Given vectors a and b, if they are parallel, then x = -4 -/
theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (1, -2)
  let b : ℝ × ℝ := (2, x)
  are_parallel a b → x = -4 := by
  sorry

end parallel_vectors_x_value_l3608_360836


namespace sum_of_common_ratios_is_three_l3608_360893

/-- Given two geometric sequences with different common ratios s and t, 
    both starting with term m, if m s^2 - m t^2 = 3(m s - m t), then s + t = 3 -/
theorem sum_of_common_ratios_is_three 
  (m : ℝ) (s t : ℝ) (h_diff : s ≠ t) (h_m_nonzero : m ≠ 0) 
  (h_eq : m * s^2 - m * t^2 = 3 * (m * s - m * t)) : 
  s + t = 3 := by
sorry

end sum_of_common_ratios_is_three_l3608_360893


namespace number_of_dogs_l3608_360857

theorem number_of_dogs (total_legs : ℕ) (num_humans : ℕ) (human_legs : ℕ) (dog_legs : ℕ)
  (h1 : total_legs = 24)
  (h2 : num_humans = 2)
  (h3 : human_legs = 2)
  (h4 : dog_legs = 4) :
  (total_legs - num_humans * human_legs) / dog_legs = 5 := by
  sorry

end number_of_dogs_l3608_360857


namespace solution_set_of_inequality_l3608_360845

/-- Given a function f : ℝ → ℝ with f(0) = 1 and f'(x) > f(x) for all x,
    the set of x where f(x) > e^x is (0, +∞) -/
theorem solution_set_of_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
    (h0 : f 0 = 1) (h1 : ∀ x, deriv f x > f x) :
    {x : ℝ | f x > Real.exp x} = Set.Ioi 0 := by
  sorry

end solution_set_of_inequality_l3608_360845


namespace triangle_properties_l3608_360820

-- Define the triangle
def triangle_ABC (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

-- State the theorem
theorem triangle_properties :
  ∀ (a b c : ℝ),
  triangle_ABC a b c →
  a = 2 →
  c = 3 →
  Real.cos (Real.arccos (1/4)) = 1/4 →
  b = Real.sqrt 10 ∧
  Real.sin (Real.arccos ((a^2 + b^2 - c^2) / (2*a*b))) = (3 * Real.sqrt 6) / 8 := by
  sorry


end triangle_properties_l3608_360820


namespace hyperbola_equation_l3608_360856

/-- A hyperbola in the Cartesian coordinate system -/
structure Hyperbola where
  /-- The equation of the hyperbola in the form (x^2 / a^2) - (y^2 / b^2) = 1 -/
  equation : ℝ → ℝ → Prop

/-- A parabola in the Cartesian coordinate system -/
structure Parabola where
  /-- The equation of the parabola -/
  equation : ℝ → ℝ → Prop

/-- The focus of a parabola -/
def parabola_focus (p : Parabola) : ℝ × ℝ := sorry

/-- The right focus of a hyperbola -/
def hyperbola_right_focus (h : Hyperbola) : ℝ × ℝ := sorry

/-- Theorem: Given a hyperbola C with its center at the origin, passing through (1, 0),
    and its right focus coinciding with the focus of y^2 = 8x, 
    the standard equation of C is x^2 - y^2/3 = 1 -/
theorem hyperbola_equation 
  (C : Hyperbola)
  (center_origin : C.equation 0 0)
  (passes_through_1_0 : C.equation 1 0)
  (p : Parabola)
  (p_eq : p.equation = fun x y ↦ y^2 = 8*x)
  (focus_coincide : hyperbola_right_focus C = parabola_focus p) :
  C.equation = fun x y ↦ x^2 - y^2/3 = 1 := by
  sorry

end hyperbola_equation_l3608_360856


namespace annual_profit_calculation_l3608_360850

theorem annual_profit_calculation (second_half_profit first_half_profit total_profit : ℕ) :
  second_half_profit = 442500 →
  first_half_profit = second_half_profit + 2750000 →
  total_profit = first_half_profit + second_half_profit →
  total_profit = 3635000 := by
  sorry

end annual_profit_calculation_l3608_360850


namespace ball_pricing_theorem_l3608_360888

/-- Represents the price and quantity of basketballs and volleyballs -/
structure BallPrices where
  basketball_price : ℕ
  volleyball_price : ℕ
  basketball_quantity : ℕ
  volleyball_quantity : ℕ

/-- Conditions of the ball purchasing problem -/
def ball_conditions (prices : BallPrices) : Prop :=
  prices.basketball_quantity + prices.volleyball_quantity = 20 ∧
  2 * prices.basketball_price + 3 * prices.volleyball_price = 190 ∧
  3 * prices.basketball_price = 5 * prices.volleyball_price

/-- Cost calculation for a given quantity of basketballs and volleyballs -/
def total_cost (prices : BallPrices) (b_qty : ℕ) (v_qty : ℕ) : ℕ :=
  b_qty * prices.basketball_price + v_qty * prices.volleyball_price

/-- Theorem stating the correct prices and most cost-effective plan -/
theorem ball_pricing_theorem (prices : BallPrices) :
  ball_conditions prices →
  prices.basketball_price = 50 ∧
  prices.volleyball_price = 30 ∧
  (∀ b v, b + v = 20 → b ≥ 8 → total_cost prices b v ≤ 800 →
    total_cost prices 8 12 ≤ total_cost prices b v) :=
sorry

end ball_pricing_theorem_l3608_360888


namespace fraction_irreducible_l3608_360885

theorem fraction_irreducible (n : ℤ) : Int.gcd (3 * n + 10) (4 * n + 13) = 1 := by
  sorry

end fraction_irreducible_l3608_360885


namespace two_balls_picked_l3608_360827

/-- Represents the number of balls of each color in the bag -/
structure BagContents where
  red : Nat
  blue : Nat
  green : Nat

/-- Calculates the total number of balls in the bag -/
def totalBalls (bag : BagContents) : Nat :=
  bag.red + bag.blue + bag.green

/-- Calculates the probability of picking two red balls -/
def probTwoRed (bag : BagContents) (picked : Nat) : Rat :=
  if picked ≠ 2 then 0
  else
    let total := totalBalls bag
    (bag.red : Rat) / total * ((bag.red - 1) : Rat) / (total - 1)

theorem two_balls_picked (bag : BagContents) (picked : Nat) :
  bag.red = 4 → bag.blue = 3 → bag.green = 2 →
  probTwoRed bag picked = 1/6 →
  picked = 2 := by
  sorry

end two_balls_picked_l3608_360827


namespace quadratic_root_equivalence_l3608_360855

theorem quadratic_root_equivalence (a : ℝ) : 
  (∃ k : ℝ, k > 0 ∧ Real.sqrt 12 = k * Real.sqrt 3) ∧ 
  (∃ m : ℝ, m > 0 ∧ 5 * Real.sqrt (a + 1) = m * Real.sqrt 3) →
  a = 2 := by
  sorry

end quadratic_root_equivalence_l3608_360855


namespace quadratic_transformation_impossibility_l3608_360897

/-- Represents a quadratic trinomial ax^2 + bx + c -/
structure QuadraticTrinomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the discriminant of a quadratic trinomial -/
def discriminant (f : QuadraticTrinomial) : ℝ :=
  f.b^2 - 4*f.a*f.c

/-- Represents the allowed operations on quadratic trinomials -/
inductive QuadraticOperation
  | op1 : QuadraticOperation  -- f(x) → x^2 f(1 + 1/x)
  | op2 : QuadraticOperation  -- f(x) → (x-1)^2 f(1/(x-1))

/-- Applies a quadratic operation to a quadratic trinomial -/
def applyOperation (f : QuadraticTrinomial) (op : QuadraticOperation) : QuadraticTrinomial :=
  match op with
  | QuadraticOperation.op1 => QuadraticTrinomial.mk f.a (2*f.a + f.b) (f.a + f.b + f.c)
  | QuadraticOperation.op2 => QuadraticTrinomial.mk f.c (f.b - 2*f.c) (f.a - f.b + f.c)

/-- Theorem stating that it's impossible to transform x^2 + 4x + 3 into x^2 + 10x + 9
    using only the allowed operations -/
theorem quadratic_transformation_impossibility :
  ∀ (ops : List QuadraticOperation),
  let f := QuadraticTrinomial.mk 1 4 3
  let g := QuadraticTrinomial.mk 1 10 9
  let result := ops.foldl applyOperation f
  result ≠ g := by
  sorry

end quadratic_transformation_impossibility_l3608_360897


namespace tons_approximation_l3608_360805

/-- Two real numbers are approximately equal if their absolute difference is less than 0.5 -/
def approximately_equal (x y : ℝ) : Prop := |x - y| < 0.5

/-- 1 ton is defined as 1000 kilograms -/
def ton : ℝ := 1000

theorem tons_approximation : approximately_equal (29.6 * ton) (30 * ton) := by sorry

end tons_approximation_l3608_360805


namespace crazy_silly_school_books_l3608_360881

theorem crazy_silly_school_books (books_read books_unread : ℕ) 
  (h1 : books_read = 13) 
  (h2 : books_unread = 8) : 
  books_read + books_unread = 21 := by
  sorry

end crazy_silly_school_books_l3608_360881


namespace root_square_condition_l3608_360803

theorem root_square_condition (a : ℚ) : 
  (∃ x y : ℚ, x^2 - (15/4)*x + a^3 = 0 ∧ y^2 - (15/4)*y + a^3 = 0 ∧ x = y^2) ↔ 
  (a = 3/2 ∨ a = -5/2) := by
sorry

end root_square_condition_l3608_360803


namespace percentage_problem_l3608_360848

/-- Given that 15% of 40 is greater than y% of 16 by 2, prove that y = 25 -/
theorem percentage_problem (y : ℝ) : 
  (0.15 * 40 = y / 100 * 16 + 2) → y = 25 := by
  sorry

end percentage_problem_l3608_360848


namespace square_difference_division_l3608_360816

theorem square_difference_division : (196^2 - 169^2) / 27 = 365 := by
  sorry

end square_difference_division_l3608_360816


namespace square_sum_value_l3608_360867

theorem square_sum_value (x y : ℝ) (h1 : x - y = 10) (h2 : x * y = 9) : x^2 + y^2 = 118 := by
  sorry

end square_sum_value_l3608_360867


namespace parallelogram_height_l3608_360880

/-- The height of a parallelogram with given area and base -/
theorem parallelogram_height (area base height : ℝ) 
  (h_area : area = 416)
  (h_base : base = 26)
  (h_formula : area = base * height) : 
  height = 16 := by
  sorry

end parallelogram_height_l3608_360880


namespace range_of_a_l3608_360826

-- Define the sets M and N
def M : Set ℝ := {x | |x - 1| ≤ 1}
def N (a : ℝ) : Set ℝ := {x | (x - a) * (x - a - 3) ≤ 0}

-- Define the theorem
theorem range_of_a :
  ∀ a : ℝ, 
  (∀ x : ℝ, x ∈ M → x ∈ N a) ∧ 
  (∃ x : ℝ, x ∈ N a ∧ x ∉ M) → 
  a ∈ Set.Icc (-1 : ℝ) 0 := by
  sorry

end range_of_a_l3608_360826


namespace cost_shop1_calculation_l3608_360801

-- Define the problem parameters
def books_shop1 : ℕ := 65
def books_shop2 : ℕ := 35
def cost_shop2 : ℕ := 2000
def avg_price : ℕ := 85

-- Theorem to prove
theorem cost_shop1_calculation :
  let total_books : ℕ := books_shop1 + books_shop2
  let total_cost : ℕ := total_books * avg_price
  let cost_shop1 : ℕ := total_cost - cost_shop2
  cost_shop1 = 6500 := by sorry

end cost_shop1_calculation_l3608_360801


namespace crayons_per_box_l3608_360877

theorem crayons_per_box (total_crayons : Float) (total_boxes : Float) 
  (h1 : total_crayons = 7.0)
  (h2 : total_boxes = 1.4) :
  total_crayons / total_boxes = 5 := by
  sorry

end crayons_per_box_l3608_360877


namespace animal_count_animal_group_count_l3608_360828

theorem animal_count (total_horses : ℕ) (cow_cow_diff : ℕ) : ℕ :=
  let total_animals := 2 * (total_horses + cow_cow_diff)
  total_animals

theorem animal_group_count : animal_count 75 10 = 170 := by
  sorry

end animal_count_animal_group_count_l3608_360828


namespace intersection_S_T_l3608_360844

def S : Set ℝ := {x | |x| < 5}
def T : Set ℝ := {x | (x+7)*(x-3) < 0}

theorem intersection_S_T : S ∩ T = {x : ℝ | -5 < x ∧ x < 3} := by sorry

end intersection_S_T_l3608_360844


namespace smores_theorem_l3608_360864

def smores_problem (graham_crackers : ℕ) (marshmallows : ℕ) : ℕ :=
  let smores_from_graham := graham_crackers / 2
  smores_from_graham - marshmallows

theorem smores_theorem (graham_crackers marshmallows : ℕ) :
  graham_crackers = 48 →
  marshmallows = 6 →
  smores_problem graham_crackers marshmallows = 18 :=
by sorry

end smores_theorem_l3608_360864


namespace liquid_depth_inverted_cone_l3608_360884

/-- Represents a right circular cone. -/
structure Cone where
  height : ℝ
  baseRadius : ℝ

/-- Represents the liquid in the cone. -/
structure Liquid where
  depthPointDown : ℝ
  depthPointUp : ℝ

/-- Theorem stating the relationship between cone dimensions, liquid depth, and the expression m - n∛p. -/
theorem liquid_depth_inverted_cone (c : Cone) (l : Liquid) 
  (h_height : c.height = 12)
  (h_radius : c.baseRadius = 5)
  (h_depth_down : l.depthPointDown = 9)
  (h_p_cube_free : ∀ (q : ℕ), q > 1 → ¬(q ^ 3 ∣ 37)) :
  ∃ (m n : ℕ), m = 12 ∧ n = 3 ∧ l.depthPointUp = m - n * (37 : ℝ) ^ (1/3 : ℝ) := by
  sorry

end liquid_depth_inverted_cone_l3608_360884


namespace profit_margin_relation_l3608_360871

theorem profit_margin_relation (S C : ℝ) (n : ℝ) (h1 : S > 0) (h2 : C > 0) (h3 : n > 0) : 
  ((1 / 3 : ℝ) * S = (1 / n : ℝ) * C) → n = 2 := by
  sorry

end profit_margin_relation_l3608_360871


namespace straw_purchase_solution_l3608_360891

/-- Represents the cost and quantity of straws --/
structure StrawPurchase where
  costA : ℚ  -- Cost per pack of type A straws
  costB : ℚ  -- Cost per pack of type B straws
  maxA : ℕ   -- Maximum number of type A straws that can be purchased

/-- Verifies if the given costs satisfy the purchase scenarios --/
def satisfiesPurchaseScenarios (sp : StrawPurchase) : Prop :=
  12 * sp.costA + 15 * sp.costB = 171 ∧
  24 * sp.costA + 28 * sp.costB = 332

/-- Checks if the maximum number of type A straws satisfies the constraints --/
def satisfiesConstraints (sp : StrawPurchase) : Prop :=
  sp.maxA ≤ 100 ∧
  sp.costA * sp.maxA + sp.costB * (100 - sp.maxA) ≤ 600 ∧
  ∀ m : ℕ, m > sp.maxA → sp.costA * m + sp.costB * (100 - m) > 600

/-- Theorem stating the solution to the straw purchase problem --/
theorem straw_purchase_solution :
  ∃ sp : StrawPurchase,
    sp.costA = 8 ∧ sp.costB = 5 ∧ sp.maxA = 33 ∧
    satisfiesPurchaseScenarios sp ∧
    satisfiesConstraints sp := by
  sorry

end straw_purchase_solution_l3608_360891


namespace calculation_proof_l3608_360876

theorem calculation_proof :
  (1) * (-3)^2 - (-1)^3 - (-2) - |(-12)| = 0 ∧
  -2^2 * 3 * (-3/2) / (2/3) - 4 * (-3/2)^2 = 18 := by sorry

end calculation_proof_l3608_360876


namespace not_div_sum_if_div_sum_squares_l3608_360870

theorem not_div_sum_if_div_sum_squares (a b : ℤ) : 
  7 ∣ (a^2 + b^2 + 1) → ¬(7 ∣ (a + b)) := by
  sorry

end not_div_sum_if_div_sum_squares_l3608_360870


namespace diagonals_from_vertex_is_six_l3608_360817

/-- A polygon with internal angles of 140 degrees -/
structure Polygon140 where
  /-- The number of sides in the polygon -/
  sides : ℕ
  /-- The internal angle of the polygon is 140 degrees -/
  internal_angle : sides * 140 = (sides - 2) * 180

/-- The number of diagonals from a single vertex in a Polygon140 -/
def diagonals_from_vertex (p : Polygon140) : ℕ :=
  p.sides - 3

/-- Theorem: The number of diagonals from a vertex in a Polygon140 is 6 -/
theorem diagonals_from_vertex_is_six (p : Polygon140) :
  diagonals_from_vertex p = 6 := by
  sorry

end diagonals_from_vertex_is_six_l3608_360817


namespace angela_january_sleep_l3608_360802

/-- The number of hours Angela slept per night in December -/
def december_sleep_hours : ℝ := 6.5

/-- The number of days in December -/
def december_days : ℕ := 31

/-- The additional hours Angela slept in January compared to December -/
def january_additional_sleep : ℝ := 62

/-- The number of days in January -/
def january_days : ℕ := 31

/-- Calculates the total hours Angela slept in December -/
def december_total_sleep : ℝ := december_sleep_hours * december_days

/-- Calculates the total hours Angela slept in January -/
def january_total_sleep : ℝ := december_total_sleep + january_additional_sleep

/-- Theorem stating that Angela slept 8.5 hours per night in January -/
theorem angela_january_sleep :
  january_total_sleep / january_days = 8.5 := by
  sorry

end angela_january_sleep_l3608_360802


namespace three_cones_theorem_l3608_360899

/-- Represents a cone with apex A -/
structure Cone where
  apexAngle : ℝ

/-- Represents a plane -/
structure Plane where

/-- Checks if a cone touches a plane -/
def touchesPlane (c : Cone) (p : Plane) : Prop :=
  sorry

/-- Checks if two cones are identical -/
def areIdentical (c1 c2 : Cone) : Prop :=
  c1.apexAngle = c2.apexAngle

/-- Checks if cones lie on the same side of a plane -/
def onSameSide (c1 c2 c3 : Cone) (p : Plane) : Prop :=
  sorry

theorem three_cones_theorem (c1 c2 c3 : Cone) (p : Plane) :
  areIdentical c1 c2 →
  c3.apexAngle = π / 2 →
  touchesPlane c1 p →
  touchesPlane c2 p →
  touchesPlane c3 p →
  onSameSide c1 c2 c3 p →
  c1.apexAngle = 2 * Real.arctan (4 / 5) :=
sorry

end three_cones_theorem_l3608_360899


namespace roots_and_d_values_l3608_360898

-- Define the polynomial p(x)
def p (c d x : ℝ) : ℝ := x^3 + c*x + d

-- Define the polynomial q(x)
def q (c d x : ℝ) : ℝ := x^3 + c*x + d + 144

-- State the theorem
theorem roots_and_d_values (u v c d : ℝ) :
  (p c d u = 0) ∧ (p c d v = 0) ∧ 
  (q c d (u + 3) = 0) ∧ (q c d (v - 2) = 0) →
  (d = 84 ∨ d = -15) := by
  sorry


end roots_and_d_values_l3608_360898


namespace zhang_san_correct_probability_l3608_360819

theorem zhang_san_correct_probability :
  let total_questions : ℕ := 4
  let questions_with_ideas : ℕ := 3
  let questions_unclear : ℕ := 1
  let prob_correct_with_idea : ℚ := 3/4
  let prob_correct_when_unclear : ℚ := 1/4
  let prob_selecting_question_with_idea : ℚ := questions_with_ideas / total_questions
  let prob_selecting_question_unclear : ℚ := questions_unclear / total_questions

  prob_selecting_question_with_idea * prob_correct_with_idea +
  prob_selecting_question_unclear * prob_correct_when_unclear = 5/8 :=
by sorry

end zhang_san_correct_probability_l3608_360819


namespace arithmetic_geometric_mean_inequality_l3608_360811

theorem arithmetic_geometric_mean_inequality (a b : ℝ) : 
  (a^2 + b^2) / 2 ≥ ((a + b) / 2)^2 := by sorry

end arithmetic_geometric_mean_inequality_l3608_360811


namespace simplify_fraction_product_l3608_360854

theorem simplify_fraction_product : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := by
  sorry

end simplify_fraction_product_l3608_360854


namespace parabola_vertex_l3608_360852

/-- Given a quadratic function f(x) = -x^2 + cx + d where c and d are real numbers,
    and the solution to f(x) ≤ 0 is (-∞, -4] ∪ [6, ∞),
    prove that the vertex of the parabola is (1, 25). -/
theorem parabola_vertex (c d : ℝ) :
  (∀ x, -x^2 + c*x + d ≤ 0 ↔ x ≤ -4 ∨ x ≥ 6) →
  ∃ (vertex : ℝ × ℝ), vertex = (1, 25) ∧
    ∀ x, -x^2 + c*x + d ≤ -(x - vertex.1)^2 + vertex.2 :=
sorry

end parabola_vertex_l3608_360852


namespace quadratic_inequality_solution_range_l3608_360873

theorem quadratic_inequality_solution_range (c : ℝ) :
  (c > 0) →
  (∃ x : ℝ, x^2 - 10*x + c < 0) ↔ (c < 25) :=
by sorry

end quadratic_inequality_solution_range_l3608_360873


namespace product_minus_difference_l3608_360821

theorem product_minus_difference (x y : ℝ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : x + y = 6) (h4 : x / y = 6) : x * y - (x - y) = 6 / 49 := by
  sorry

end product_minus_difference_l3608_360821


namespace escalator_time_to_cover_l3608_360889

/-- Proves that a person walking on a moving escalator takes 10 seconds to cover its length -/
theorem escalator_time_to_cover (escalator_speed : ℝ) (person_speed : ℝ) (escalator_length : ℝ) :
  escalator_speed = 15 →
  person_speed = 3 →
  escalator_length = 180 →
  escalator_length / (escalator_speed + person_speed) = 10 := by
  sorry

end escalator_time_to_cover_l3608_360889


namespace p_money_theorem_l3608_360833

theorem p_money_theorem (p q r : ℚ) : 
  (p = (1/6 * p + 1/6 * p) + 32) → p = 48 := by
  sorry

end p_money_theorem_l3608_360833


namespace fractional_equation_solution_range_l3608_360823

theorem fractional_equation_solution_range (m x : ℝ) : 
  (m / (2 * x - 1) + 3 = 0) → 
  (x > 0) → 
  (m < 3 ∧ m ≠ 0) := by
sorry

end fractional_equation_solution_range_l3608_360823


namespace sine_cosine_inequality_l3608_360849

theorem sine_cosine_inequality (x : Real) (h : Real.sin x + Real.cos x ≤ 0) :
  Real.sin x ^ 1993 + Real.cos x ^ 1993 ≤ 0 := by
  sorry

end sine_cosine_inequality_l3608_360849


namespace arithmetic_sequence_ratio_l3608_360800

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ+ → ℚ

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ+) : ℚ :=
  (Finset.range n.val).sum (λ i => seq.a ⟨i + 1, Nat.succ_pos i⟩)

theorem arithmetic_sequence_ratio (a b : ArithmeticSequence) :
  (∀ n : ℕ+, (sum_n a n) / (sum_n b n) = (38 * n + 14) / (2 * n + 1)) →
  (a.a 6) / (b.a 7) = 16 := by
  sorry

end arithmetic_sequence_ratio_l3608_360800


namespace pancake_milk_calculation_l3608_360835

/-- Given the ratio of pancakes to quarts of milk for 18 pancakes,
    and the conversion rate of quarts to pints,
    prove that the number of pints needed for 9 pancakes is 3. -/
theorem pancake_milk_calculation (pancakes_18 : ℕ) (quarts_18 : ℚ) (pints_per_quart : ℚ) :
  pancakes_18 = 18 →
  quarts_18 = 3 →
  pints_per_quart = 2 →
  (9 : ℚ) * quarts_18 * pints_per_quart / pancakes_18 = 3 := by
  sorry

end pancake_milk_calculation_l3608_360835


namespace intersection_of_M_and_N_l3608_360896

def M : Set ℝ := {x | x^2 - 3*x = 0}
def N : Set ℝ := {x | x > -1}

theorem intersection_of_M_and_N : M ∩ N = {0, 3} := by sorry

end intersection_of_M_and_N_l3608_360896


namespace matrix_equation_properties_l3608_360806

open Matrix ComplexConjugate

variable {n : ℕ}

def conjugate_transpose (A : Matrix (Fin n) (Fin n) ℂ) : Matrix (Fin n) (Fin n) ℂ :=
  star A

theorem matrix_equation_properties
  (α : ℂ)
  (A : Matrix (Fin n) (Fin n) ℂ)
  (h_alpha : α ≠ 0)
  (h_A : A ≠ 0)
  (h_eq : A ^ 2 + (conjugate_transpose A) ^ 2 = α • (A * conjugate_transpose A)) :
  α.im = 0 ∧ Complex.abs α ≤ 2 ∧ A * conjugate_transpose A = conjugate_transpose A * A := by
  sorry

end matrix_equation_properties_l3608_360806


namespace parallel_lines_k_value_l3608_360812

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y : ℝ, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- The value of k for which the lines y = 5x - 3 and y = (3k)x + 7 are parallel -/
theorem parallel_lines_k_value :
  (∀ x y : ℝ, y = 5 * x - 3 ↔ y = (3 * k) * x + 7) → k = 5 / 3 :=
by sorry

end parallel_lines_k_value_l3608_360812


namespace smallest_x_plus_y_l3608_360838

theorem smallest_x_plus_y : ∃ (x y : ℕ), 
  x > 0 ∧ y > 0 ∧ x < y ∧
  (100 : ℚ) + (x : ℚ) / y = 2 * ((100 : ℚ) * x / y) ∧
  ∀ (a b : ℕ), a > 0 → b > 0 → a < b → 
    (100 : ℚ) + (a : ℚ) / b = 2 * ((100 : ℚ) * a / b) →
    x + y ≤ a + b ∧
  x + y = 299 :=
by sorry

end smallest_x_plus_y_l3608_360838


namespace x_axis_condition_l3608_360813

/-- A line in the 2D plane represented by the equation Ax + By + C = 0 -/
structure Line where
  A : ℝ
  B : ℝ
  C : ℝ

/-- The x-axis is a line where y = 0 for all x -/
def is_x_axis (l : Line) : Prop :=
  ∀ x y : ℝ, l.A * x + l.B * y + l.C = 0 ↔ y = 0

/-- If a line is the x-axis, then B ≠ 0 and A = C = 0 -/
theorem x_axis_condition (l : Line) :
  is_x_axis l → l.B ≠ 0 ∧ l.A = 0 ∧ l.C = 0 := by
  sorry

end x_axis_condition_l3608_360813


namespace jacob_has_six_marshmallows_l3608_360894

/-- Calculates the number of marshmallows Jacob currently has -/
def jacobs_marshmallows (graham_crackers : ℕ) (more_marshmallows_needed : ℕ) : ℕ :=
  (graham_crackers / 2) - more_marshmallows_needed

/-- Proves that Jacob has 6 marshmallows given the problem conditions -/
theorem jacob_has_six_marshmallows :
  jacobs_marshmallows 48 18 = 6 := by
  sorry

end jacob_has_six_marshmallows_l3608_360894


namespace girls_in_school_l3608_360863

/-- The number of girls in a school after new students join -/
def total_girls (initial_girls new_girls : ℕ) : ℕ :=
  initial_girls + new_girls

/-- Theorem stating that the total number of girls after new students joined is 1414 -/
theorem girls_in_school (initial_girls new_girls : ℕ) 
  (h1 : initial_girls = 732) 
  (h2 : new_girls = 682) : 
  total_girls initial_girls new_girls = 1414 := by
  sorry

end girls_in_school_l3608_360863


namespace line_through_two_points_l3608_360810

/-- Given a line with equation x = 4y + 5 passing through points (m, n) and (m + 2, n + p),
    prove that p = 1/2 -/
theorem line_through_two_points (m n p : ℝ) : 
  (m = 4 * n + 5) ∧ (m + 2 = 4 * (n + p) + 5) → p = 1/2 := by
  sorry

end line_through_two_points_l3608_360810


namespace intersection_of_A_and_B_l3608_360866

-- Define the sets A and B
def A : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = 3^p.1}
def B : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = 2^(-p.1)}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {(0, 1)} := by
  sorry

end intersection_of_A_and_B_l3608_360866


namespace function_comparison_l3608_360868

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
variable (h_even : ∀ x, f x = f (-x))
variable (h_decreasing : ∀ x y, x < y → y < 0 → f x > f y)

-- Define the theorem
theorem function_comparison (x₁ x₂ : ℝ) 
  (h₁ : x₁ < 0) (h₂ : x₁ + x₂ > 0) : f x₁ < f x₂ := by
  sorry

end function_comparison_l3608_360868


namespace felix_brother_lifting_capacity_l3608_360831

/-- Given information about Felix and his brother's weights and lifting capacities,
    prove how much Felix's brother can lift off the ground. -/
theorem felix_brother_lifting_capacity
  (felix_lift_ratio : ℝ)
  (felix_lift_weight : ℝ)
  (brother_weight_ratio : ℝ)
  (brother_lift_ratio : ℝ)
  (h1 : felix_lift_ratio = 1.5)
  (h2 : felix_lift_weight = 150)
  (h3 : brother_weight_ratio = 2)
  (h4 : brother_lift_ratio = 3) :
  felix_lift_weight * brother_weight_ratio * brother_lift_ratio / felix_lift_ratio = 600 :=
by sorry

end felix_brother_lifting_capacity_l3608_360831


namespace absolute_difference_of_factors_l3608_360843

theorem absolute_difference_of_factors (m n : ℝ) 
  (h1 : m * n = 6) 
  (h2 : m + n = 7) : 
  |m - n| = 5 := by
  sorry

end absolute_difference_of_factors_l3608_360843


namespace max_value_constraint_l3608_360824

theorem max_value_constraint (x y z : ℝ) (h : 9 * x^2 + 4 * y^2 + 25 * z^2 = 1) :
  ∃ (M : ℝ), M = Real.sqrt 298 ∧ (8 * x + 3 * y + 15 * z ≤ M) ∧
  ∃ (x₀ y₀ z₀ : ℝ), 9 * x₀^2 + 4 * y₀^2 + 25 * z₀^2 = 1 ∧ 8 * x₀ + 3 * y₀ + 15 * z₀ = M :=
by sorry

end max_value_constraint_l3608_360824


namespace zookeeper_fish_count_l3608_360872

theorem zookeeper_fish_count (penguins_fed : ℕ) (total_penguins : ℕ) (penguins_to_feed : ℕ) :
  penguins_fed = 19 →
  total_penguins = 36 →
  penguins_to_feed = 17 →
  penguins_fed + penguins_to_feed = total_penguins :=
by sorry

end zookeeper_fish_count_l3608_360872


namespace sibling_count_l3608_360890

theorem sibling_count (boys girls : ℕ) : 
  boys = 1 ∧ 
  boys - 1 = 0 ∧ 
  girls - 1 = boys → 
  boys + girls = 3 := by
sorry

end sibling_count_l3608_360890


namespace negation_of_proposition_l3608_360860

def proposition (x : ℕ) : Prop := (1/2:ℝ)^x ≤ 1/2

theorem negation_of_proposition :
  (¬ ∀ (x : ℕ), x > 0 → proposition x) ↔ (∃ (x : ℕ), x > 0 ∧ (1/2:ℝ)^x > 1/2) :=
sorry

end negation_of_proposition_l3608_360860


namespace inscribed_quadrilateral_tangent_point_property_l3608_360818

/-- A quadrilateral inscribed in a circle with an inscribed circle inside it. -/
structure InscribedQuadrilateral where
  /-- The lengths of the four sides of the quadrilateral -/
  sides : Fin 4 → ℝ
  /-- The quadrilateral is inscribed in a circle -/
  inscribed_in_circle : Bool
  /-- There is a circle inscribed in the quadrilateral -/
  has_inscribed_circle : Bool

/-- The point of tangency divides a side into two segments -/
def tangent_point_division (q : InscribedQuadrilateral) : ℝ × ℝ := sorry

/-- The theorem stating the property of the inscribed quadrilateral -/
theorem inscribed_quadrilateral_tangent_point_property 
  (q : InscribedQuadrilateral)
  (h1 : q.sides 0 = 65)
  (h2 : q.sides 1 = 95)
  (h3 : q.sides 2 = 125)
  (h4 : q.sides 3 = 105)
  (h5 : q.inscribed_in_circle = true)
  (h6 : q.has_inscribed_circle = true) :
  let (x, y) := tangent_point_division q
  |x - y| = 14 := by sorry

end inscribed_quadrilateral_tangent_point_property_l3608_360818


namespace r₂_bound_l3608_360829

/-- The function f(x) = x² - r₂x + r₃ -/
def f (r₂ r₃ : ℝ) (x : ℝ) : ℝ := x^2 - r₂ * x + r₃

/-- The sequence g_n defined recursively -/
def g (r₂ r₃ : ℝ) : ℕ → ℝ
  | 0 => 0
  | n + 1 => f r₂ r₃ (g r₂ r₃ n)

/-- The property that g₂ᵢ < g₂ᵢ₊₁ and g₂ᵢ₊₁ > g₂ᵢ₊₂ for 0 ≤ i ≤ 2011 -/
def alternating_property (r₂ r₃ : ℝ) : Prop :=
  ∀ i, 0 ≤ i ∧ i ≤ 2011 → g r₂ r₃ (2*i) < g r₂ r₃ (2*i + 1) ∧ g r₂ r₃ (2*i + 1) > g r₂ r₃ (2*i + 2)

/-- The property that there exists j such that gᵢ₊₁ > gᵢ for all i > j -/
def eventually_increasing (r₂ r₃ : ℝ) : Prop :=
  ∃ j : ℕ, ∀ i, i > j → g r₂ r₃ (i + 1) > g r₂ r₃ i

/-- The property that the sequence is unbounded -/
def unbounded (r₂ r₃ : ℝ) : Prop :=
  ∀ M : ℝ, ∃ N : ℕ, g r₂ r₃ N > M

theorem r₂_bound (r₂ r₃ : ℝ) 
  (h₁ : alternating_property r₂ r₃)
  (h₂ : eventually_increasing r₂ r₃)
  (h₃ : unbounded r₂ r₃) :
  |r₂| > 2 ∧ ∀ ε > 0, ∃ r₂' r₃', 
    alternating_property r₂' r₃' ∧ 
    eventually_increasing r₂' r₃' ∧ 
    unbounded r₂' r₃' ∧ 
    |r₂'| < 2 + ε :=
sorry

end r₂_bound_l3608_360829


namespace probability_of_selection_l3608_360882

def multiple_choice_count : ℕ := 12
def fill_in_blank_count : ℕ := 4
def open_ended_count : ℕ := 6
def total_questions : ℕ := multiple_choice_count + fill_in_blank_count + open_ended_count
def selection_count : ℕ := 3

theorem probability_of_selection (multiple_choice_count fill_in_blank_count open_ended_count total_questions selection_count : ℕ) 
  (h1 : multiple_choice_count = 12)
  (h2 : fill_in_blank_count = 4)
  (h3 : open_ended_count = 6)
  (h4 : total_questions = multiple_choice_count + fill_in_blank_count + open_ended_count)
  (h5 : selection_count = 3) :
  (Nat.choose multiple_choice_count 1 * Nat.choose open_ended_count 2 +
   Nat.choose multiple_choice_count 2 * Nat.choose open_ended_count 1 +
   Nat.choose multiple_choice_count 1 * Nat.choose open_ended_count 1 * Nat.choose fill_in_blank_count 1) /
  (Nat.choose total_questions selection_count - Nat.choose (fill_in_blank_count + open_ended_count) selection_count) = 43 / 71 := by
  sorry

#check probability_of_selection

end probability_of_selection_l3608_360882


namespace absolute_value_inequality_implies_m_equals_negative_four_l3608_360830

theorem absolute_value_inequality_implies_m_equals_negative_four (m : ℝ) :
  (∀ x : ℝ, |2*x - m| ≤ |3*x + 6|) → m = -4 := by
  sorry

end absolute_value_inequality_implies_m_equals_negative_four_l3608_360830


namespace derivative_problems_l3608_360807

open Real

theorem derivative_problems :
  (∀ x : ℝ, deriv (λ x => (2*x^2 + 3)*(3*x - 1)) x = 18*x^2 - 4*x + 9) ∧
  (∀ x : ℝ, deriv (λ x => x * exp x + 2*x + 1) x = exp x + x * exp x + 2) := by
  sorry

end derivative_problems_l3608_360807


namespace min_value_sum_squares_l3608_360842

theorem min_value_sum_squares (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : a/b + b/c + c/a + b/a + c/b + a/c = 9) : 
  ∀ x y z : ℝ, x > 0 → y > 0 → z > 0 → 
  x/y + y/z + z/x + y/x + z/y + x/z = 9 → 
  (a/b + b/c + c/a)^2 + (b/a + c/b + a/c)^2 ≤ (x/y + y/z + z/x)^2 + (y/x + z/y + x/z)^2 :=
by sorry

end min_value_sum_squares_l3608_360842


namespace correct_answer_l3608_360822

theorem correct_answer (x : ℝ) (h : 3 * x = 90) : x - 30 = 0 := by
  sorry

end correct_answer_l3608_360822


namespace repair_cost_calculation_l3608_360814

/-- Calculates the total cost of car repair given the following parameters:
  * rate1: hourly rate of the first mechanic
  * hours1: hours worked per day by the first mechanic
  * days1: number of days worked by the first mechanic
  * rate2: hourly rate of the second mechanic
  * hours2: hours worked per day by the second mechanic
  * days2: number of days worked by the second mechanic
  * parts_cost: cost of parts used in the repair
-/
def total_repair_cost (rate1 hours1 days1 rate2 hours2 days2 parts_cost : ℕ) : ℕ :=
  rate1 * hours1 * days1 + rate2 * hours2 * days2 + parts_cost

/-- Theorem stating that the total repair cost for the given scenario is $14,420 -/
theorem repair_cost_calculation :
  total_repair_cost 60 8 14 75 6 10 3200 = 14420 := by
  sorry

#eval total_repair_cost 60 8 14 75 6 10 3200

end repair_cost_calculation_l3608_360814


namespace f_lower_bound_and_equality_l3608_360862

def f (x a b : ℝ) : ℝ := |x + a| + |x - b|

theorem f_lower_bound_and_equality (a b : ℝ) 
  (h : (1 / (2 * a)) + (2 / b) = 1) :
  (∀ x, f x a b ≥ 9/2) ∧
  (∃ x, f x a b = 9/2 → a = 3/2 ∧ b = 3) := by
sorry

end f_lower_bound_and_equality_l3608_360862


namespace divisibility_proof_l3608_360834

theorem divisibility_proof (a : ℤ) (n : ℕ) : 
  ∃ k : ℤ, (a + 1)^(2*n + 1) + a^(n + 2) = k * (a^2 + a + 1) := by
  sorry

end divisibility_proof_l3608_360834


namespace five_by_five_uncoverable_l3608_360875

/-- Represents a game board -/
structure Board :=
  (rows : Nat)
  (cols : Nat)
  (black_squares : Nat)
  (white_squares : Nat)

/-- Represents a domino placement on the board -/
def DominoPlacement := List (Nat × Nat)

/-- Check if a board can be covered by dominoes -/
def can_be_covered (b : Board) (p : DominoPlacement) : Prop :=
  (b.rows * b.cols = 2 * p.length) ∧ 
  (b.black_squares = b.white_squares)

/-- The 5x5 board with specific color pattern -/
def board_5x5 : Board :=
  { rows := 5
  , cols := 5
  , black_squares := 9   -- central 3x3 section
  , white_squares := 16  -- border
  }

/-- Theorem stating that the 5x5 board cannot be covered -/
theorem five_by_five_uncoverable : 
  ∀ p : DominoPlacement, ¬(can_be_covered board_5x5 p) :=
sorry

end five_by_five_uncoverable_l3608_360875


namespace bridge_length_bridge_length_proof_l3608_360883

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  let total_distance := train_speed_ms * crossing_time
  total_distance - train_length

/-- Proof of the bridge length given specific conditions -/
theorem bridge_length_proof :
  bridge_length 150 45 30 = 225 := by
  sorry

end bridge_length_bridge_length_proof_l3608_360883


namespace greatest_prime_factor_of_125_l3608_360874

theorem greatest_prime_factor_of_125 : ∃ p : ℕ, 
  Nat.Prime p ∧ p ∣ 125 ∧ ∀ q : ℕ, Nat.Prime q → q ∣ 125 → q ≤ p :=
by sorry

end greatest_prime_factor_of_125_l3608_360874


namespace unique_consecutive_set_sum_18_l3608_360837

/-- A set of consecutive positive integers -/
def ConsecutiveSet (a n : ℕ) : Set ℕ := {x | ∃ k, 0 ≤ k ∧ k < n ∧ x = a + k}

/-- The sum of a set of consecutive positive integers -/
def ConsecutiveSetSum (a n : ℕ) : ℕ := n * (2 * a + n - 1) / 2

/-- Main theorem: There is exactly one set of consecutive positive integers with sum 18 -/
theorem unique_consecutive_set_sum_18 :
  ∃! p : ℕ × ℕ, p.2 ≥ 2 ∧ ConsecutiveSetSum p.1 p.2 = 18 :=
sorry

end unique_consecutive_set_sum_18_l3608_360837


namespace seven_digit_number_product_l3608_360861

theorem seven_digit_number_product : ∃ (x y : ℕ), 
  (1000000 ≤ x ∧ x < 10000000) ∧ 
  (1000000 ≤ y ∧ y < 10000000) ∧ 
  (10^7 * x + y = 3 * x * y) ∧ 
  (x = 1666667 ∧ y = 3333334) := by
  sorry

end seven_digit_number_product_l3608_360861

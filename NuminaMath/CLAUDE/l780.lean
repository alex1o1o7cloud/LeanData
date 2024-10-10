import Mathlib

namespace triangle_area_l780_78011

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x) * Real.cos (ω * x) - 2 * Real.sqrt 3 * (Real.cos (ω * x))^2 + Real.sqrt 3

theorem triangle_area (ω : ℝ) (A B C : ℝ) (a b c : ℝ) :
  ω > 0 →
  (∀ x : ℝ, f ω (x + π / (2 * ω)) = f ω x) →
  0 < C ∧ C < π / 2 →
  f 1 C = Real.sqrt 3 →
  c = 3 →
  Real.sin B = 2 * Real.sin A →
  (1 / 2 : ℝ) * a * b * Real.sin C = 3 * Real.sqrt 3 / 2 :=
by sorry

end triangle_area_l780_78011


namespace prob_at_least_two_white_correct_l780_78071

/-- The probability of drawing at least two white balls in three draws from a bag 
    containing 2 red balls and 4 white balls, with replacement -/
def prob_at_least_two_white : ℚ := 20 / 27

/-- The total number of balls in the bag -/
def total_balls : ℕ := 6

/-- The number of white balls in the bag -/
def white_balls : ℕ := 4

/-- The number of draws -/
def num_draws : ℕ := 3

theorem prob_at_least_two_white_correct : 
  prob_at_least_two_white = 
    (Nat.choose num_draws 2 * (white_balls / total_balls)^2 * ((total_balls - white_balls) / total_balls)) +
    (white_balls / total_balls)^num_draws :=
sorry

end prob_at_least_two_white_correct_l780_78071


namespace pizza_slices_l780_78098

theorem pizza_slices (num_pizzas : ℕ) (slices_per_pizza : ℕ) 
  (h1 : num_pizzas = 17) 
  (h2 : slices_per_pizza = 4) : 
  num_pizzas * slices_per_pizza = 68 := by
  sorry

end pizza_slices_l780_78098


namespace root_sum_absolute_value_l780_78040

theorem root_sum_absolute_value (m : ℤ) (p q r : ℤ) : 
  (∀ x : ℤ, x^3 - 2023*x + m = 0 ↔ x = p ∨ x = q ∨ x = r) →
  |p| + |q| + |r| = 106 := by
sorry

end root_sum_absolute_value_l780_78040


namespace equation_solutions_l780_78060

theorem equation_solutions :
  (∃ x : ℚ, 3 * x - (x - 1) = 7 ∧ x = 3) ∧
  (∃ x : ℚ, (2 * x - 1) / 3 - (x - 3) / 6 = 1 ∧ x = 5 / 3) := by
  sorry

end equation_solutions_l780_78060


namespace quadratic_roots_properties_l780_78047

theorem quadratic_roots_properties (x₁ x₂ : ℝ) :
  x₁^2 - 2*x₁ - 1 = 0 ∧ x₂^2 - 2*x₂ - 1 = 0 →
  (x₁ + x₂) * (x₁ * x₂) = -2 ∧ (x₁ - x₂)^2 = 8 := by
  sorry

end quadratic_roots_properties_l780_78047


namespace two_numbers_difference_l780_78027

theorem two_numbers_difference (x y : ℤ) 
  (sum_eq : x + y = 40)
  (triple_minus_double : 3 * max x y - 2 * min x y = 8) :
  |x - y| = 4 := by sorry

end two_numbers_difference_l780_78027


namespace circle_tangent_slope_range_l780_78082

theorem circle_tangent_slope_range (x y : ℝ) (h : x^2 + y^2 = 1) :
  ∃ (k : ℝ), k = 3/4 ∧ ∀ (z : ℝ), z ≥ k → ∃ (a b : ℝ), a^2 + b^2 = 1 ∧ z = (b + 2) / (a + 1) :=
sorry

end circle_tangent_slope_range_l780_78082


namespace function_behavior_l780_78061

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 6*x + 10

-- Define the derivative of the function
def f' (x : ℝ) : ℝ := 2*x - 6

-- Theorem statement
theorem function_behavior :
  ∃ c ∈ Set.Ioo 2 4, 
    (∀ x ∈ Set.Ioo 2 c, (f' x < 0)) ∧ 
    (∀ x ∈ Set.Ioo c 4, (f' x > 0)) :=
sorry


end function_behavior_l780_78061


namespace family_ages_l780_78059

theorem family_ages (man son daughter : ℕ) : 
  man = son + 46 →
  man + 2 = 2 * (son + 2) →
  daughter = son - 4 →
  son + daughter = 84 := by
sorry

end family_ages_l780_78059


namespace five_people_six_chairs_l780_78041

/-- The number of ways to arrange n people in m chairs -/
def arrange (n : ℕ) (m : ℕ) : ℕ := sorry

/-- There are 5 people and 6 chairs -/
def num_people : ℕ := 5
def num_chairs : ℕ := 6

theorem five_people_six_chairs :
  arrange num_people num_chairs = 720 := by sorry

end five_people_six_chairs_l780_78041


namespace divisibility_theorem_l780_78093

theorem divisibility_theorem (a b c : ℕ) 
  (h1 : b ∣ a^3) 
  (h2 : c ∣ b^3) 
  (h3 : a ∣ c^3) : 
  a * b * c ∣ (a + b + c)^13 := by
  sorry

end divisibility_theorem_l780_78093


namespace problem_solution_l780_78086

noncomputable def problem (e₁ e₂ OA OB : ℝ × ℝ) (C : ℝ × ℝ) : Prop :=
  let A := OA
  let B := OB
  -- e₁ and e₂ are unit vectors in the direction of x-axis and y-axis
  e₁ = (1, 0) ∧ e₂ = (0, 1) ∧
  -- OA = e₁ + e₂
  OA = (e₁.1 + e₂.1, e₁.2 + e₂.2) ∧
  -- OB = 5e₁ + 3e₂
  OB = (5 * e₁.1 + 3 * e₂.1, 5 * e₁.2 + 3 * e₂.2) ∧
  -- AB ⟂ AC
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0 ∧
  -- |AB| = |AC|
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = (C.1 - A.1)^2 + (C.2 - A.2)^2 →
  OB.1 * C.1 + OB.2 * C.2 = 6 ∨ OB.1 * C.1 + OB.2 * C.2 = 10

theorem problem_solution :
  ∀ (e₁ e₂ OA OB : ℝ × ℝ) (C : ℝ × ℝ), problem e₁ e₂ OA OB C :=
sorry

end problem_solution_l780_78086


namespace range_of_a_l780_78057

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ a then x^2 - 2 else x + 2

noncomputable def g (a : ℝ) (x : ℝ) : ℝ :=
  f a (Real.log x + 1/x) - a

theorem range_of_a (a : ℝ) :
  (∃ x > 0, g a x = 0) → a ∈ Set.Icc (-1) 2 ∪ Set.Ici 3 :=
sorry

end range_of_a_l780_78057


namespace triangle_area_triangle_area_proof_l780_78054

/-- The area of the triangle formed by y = 3x - 6, y = -4x + 24, and the y-axis -/
theorem triangle_area : ℝ → Prop :=
  λ area : ℝ =>
    let line1 : ℝ → ℝ := λ x => 3 * x - 6
    let line2 : ℝ → ℝ := λ x => -4 * x + 24
    let y_axis : ℝ → ℝ := λ x => 0
    let intersection_x : ℝ := 30 / 7
    let intersection_y : ℝ := line1 intersection_x
    let y_intercept1 : ℝ := line1 0
    let y_intercept2 : ℝ := line2 0
    area = 450 / 7 ∧
    area = (1 / 2) * (y_intercept2 - y_intercept1) * intersection_x

/-- Proof of the triangle area theorem -/
theorem triangle_area_proof : triangle_area (450 / 7) := by
  sorry

end triangle_area_triangle_area_proof_l780_78054


namespace henry_seed_growth_l780_78048

/-- Given that Henry starts with 5 seeds and triples his seeds each day, 
    this theorem proves that it takes 6 days to exceed 500 seeds. -/
theorem henry_seed_growth (n : ℕ) : n > 0 ∧ 5 * 3^(n-1) > 500 ↔ n ≥ 6 :=
sorry

end henry_seed_growth_l780_78048


namespace favorite_number_ratio_l780_78084

theorem favorite_number_ratio :
  ∀ (misty_number glory_number : ℕ),
    glory_number = 450 →
    misty_number + glory_number = 600 →
    glory_number / misty_number = 3 :=
by
  sorry

end favorite_number_ratio_l780_78084


namespace unique_solution_l780_78044

theorem unique_solution : ∀ a b : ℕ+,
  (¬ (7 ∣ (a * b * (a + b)))) →
  ((7^7) ∣ ((a + b)^7 - a^7 - b^7)) →
  (a = 18 ∧ b = 1) := by
  sorry

end unique_solution_l780_78044


namespace smallest_sum_with_conditions_l780_78087

theorem smallest_sum_with_conditions (a b : ℕ+) 
  (h1 : Nat.gcd (a + b) 330 = 1)
  (h2 : ∃ k : ℕ, a^(a:ℕ) = k * b^(b:ℕ))
  (h3 : ¬∃ m : ℕ, a = m * b) :
  ∀ (x y : ℕ+), 
    (Nat.gcd (x + y) 330 = 1) → 
    (∃ k : ℕ, x^(x:ℕ) = k * y^(y:ℕ)) → 
    (¬∃ m : ℕ, x = m * y) → 
    (a + b : ℕ) ≤ (x + y : ℕ) ∧ 
    (a + b : ℕ) = 507 := by
  sorry

end smallest_sum_with_conditions_l780_78087


namespace max_profit_min_sales_for_profit_l780_78032

-- Define the cost per unit
def cost : ℝ := 20

-- Define the relationship between price and sales volume
def sales_volume (x : ℝ) : ℝ := -10 * x + 500

-- Define the profit function
def profit (x : ℝ) : ℝ := (x - cost) * sales_volume x

-- Define the price constraints
def price_constraint (x : ℝ) : Prop := 25 ≤ x ∧ x ≤ 38

-- Theorem 1: Maximum profit occurs at x = 35 and is equal to 2250
theorem max_profit :
  ∃ (x : ℝ), price_constraint x ∧
  profit x = 2250 ∧
  ∀ (y : ℝ), price_constraint y → profit y ≤ profit x :=
sorry

-- Theorem 2: At price 38, selling 120 units yields a profit of at least 2000
theorem min_sales_for_profit :
  sales_volume 38 ≥ 120 ∧ profit 38 ≥ 2000 :=
sorry

end max_profit_min_sales_for_profit_l780_78032


namespace no_integer_roots_l780_78037

/-- A polynomial with integer coefficients -/
def IntPolynomial := ℕ → ℤ

/-- Evaluation of a polynomial at a point -/
def eval (P : IntPolynomial) (x : ℤ) : ℤ :=
  sorry

/-- A number is odd if it's not divisible by 2 -/
def IsOdd (n : ℤ) : Prop := n % 2 ≠ 0

theorem no_integer_roots (P : IntPolynomial) 
  (h0 : IsOdd (eval P 0)) 
  (h1 : IsOdd (eval P 1)) : 
  ∀ (n : ℤ), eval P n ≠ 0 := by
  sorry

end no_integer_roots_l780_78037


namespace envelope_width_l780_78080

/-- Given a rectangular envelope with an area of 36 square inches and a height of 6 inches,
    prove that its width is 6 inches. -/
theorem envelope_width (area : ℝ) (height : ℝ) (width : ℝ) 
    (h1 : area = 36) 
    (h2 : height = 6) 
    (h3 : area = width * height) : 
  width = 6 := by
  sorry

end envelope_width_l780_78080


namespace dog_distance_proof_l780_78012

/-- The distance the dog runs when Ivan travels from work to home -/
def dog_distance (total_distance : ℝ) : ℝ :=
  2 * total_distance

theorem dog_distance_proof (total_distance : ℝ) (h1 : total_distance = 6) :
  dog_distance total_distance = 12 :=
by
  sorry

#check dog_distance_proof

end dog_distance_proof_l780_78012


namespace rectangle_width_l780_78010

/-- Given a rectangle where the length is 3 times the width and the area is 108 square inches,
    prove that the width is 6 inches. -/
theorem rectangle_width (w : ℝ) (h1 : w > 0) (h2 : 3 * w * w = 108) : w = 6 := by
  sorry

end rectangle_width_l780_78010


namespace buckingham_palace_visitors_l780_78033

theorem buckingham_palace_visitors 
  (total_visitors : ℕ) 
  (previous_day_visitors : ℕ) 
  (today_visitors : ℕ) 
  (h1 : total_visitors = 949) 
  (h2 : previous_day_visitors = 703) 
  (h3 : today_visitors > 0) 
  (h4 : total_visitors = previous_day_visitors + today_visitors) : 
  today_visitors = 246 := by
sorry

end buckingham_palace_visitors_l780_78033


namespace select_parts_with_first_class_l780_78083

theorem select_parts_with_first_class (total : Nat) (first_class : Nat) (second_class : Nat) (select : Nat) :
  total = first_class + second_class →
  first_class = 5 →
  second_class = 3 →
  select = 3 →
  (Nat.choose total select) - (Nat.choose second_class select) = 55 := by
  sorry

end select_parts_with_first_class_l780_78083


namespace lino_shell_collection_l780_78095

/-- The number of shells Lino picked up in the morning -/
def morning_shells : ℕ := 292

/-- The number of shells Lino picked up in the afternoon -/
def afternoon_shells : ℕ := 324

/-- The total number of shells Lino picked up -/
def total_shells : ℕ := morning_shells + afternoon_shells

/-- Theorem stating that the total number of shells Lino picked up is 616 -/
theorem lino_shell_collection : total_shells = 616 := by
  sorry

end lino_shell_collection_l780_78095


namespace vector_sum_parallel_l780_78043

/-- Two vectors are parallel if their components are proportional -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem vector_sum_parallel (x : ℝ) :
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (x, -2)
  parallel a b → a.1 + b.1 = -2 ∧ a.2 + b.2 = -1 := by
  sorry

end vector_sum_parallel_l780_78043


namespace switching_strategy_wins_more_than_half_l780_78064

structure ThreeBoxGame where
  boxes : Fin 3 → Bool  -- True if box contains prize, False if empty
  prize_exists : ∃ i, boxes i = true
  two_empty : ∃ i j, i ≠ j ∧ boxes i = false ∧ boxes j = false

def initial_choice (game : ThreeBoxGame) : Fin 3 :=
  sorry

def host_opens (game : ThreeBoxGame) (choice : Fin 3) : Fin 3 :=
  sorry

def switch (initial : Fin 3) (opened : Fin 3) : Fin 3 :=
  sorry

def probability_of_winning_by_switching (game : ThreeBoxGame) : ℝ :=
  sorry

theorem switching_strategy_wins_more_than_half :
  ∀ game : ThreeBoxGame, probability_of_winning_by_switching game > 1/2 :=
sorry

end switching_strategy_wins_more_than_half_l780_78064


namespace three_heads_probability_l780_78079

/-- A fair coin has a probability of 1/2 for heads on a single flip -/
def fair_coin_prob : ℚ := 1/2

/-- The probability of getting three heads in three flips of a fair coin -/
def three_heads_prob : ℚ := fair_coin_prob * fair_coin_prob * fair_coin_prob

/-- Theorem: The probability of getting three heads in three flips of a fair coin is 1/8 -/
theorem three_heads_probability : three_heads_prob = 1/8 := by
  sorry

end three_heads_probability_l780_78079


namespace sock_selection_problem_l780_78099

theorem sock_selection_problem :
  let total_socks : ℕ := 7
  let socks_to_choose : ℕ := 4
  let number_of_ways : ℕ := Nat.choose total_socks socks_to_choose
  number_of_ways = 35 := by
sorry

end sock_selection_problem_l780_78099


namespace children_on_bus_after_stop_l780_78063

theorem children_on_bus_after_stop (initial : ℕ) (got_on : ℕ) (got_off : ℕ) :
  initial = 22 → got_on = 40 → got_off = 60 →
  initial + got_on - got_off = 2 := by
  sorry

end children_on_bus_after_stop_l780_78063


namespace constrained_optimization_l780_78013

theorem constrained_optimization (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0)
  (h1 : 3*x + 5*y + 7*z = 10) (h2 : x + 2*y + 5*z = 6) :
  let w := 2*x - 3*y + 4*z
  ∃ (max_w : ℝ), (∀ x' y' z' : ℝ, x' ≥ 0 → y' ≥ 0 → z' ≥ 0 →
    3*x' + 5*y' + 7*z' = 10 → x' + 2*y' + 5*z' = 6 →
    2*x' - 3*y' + 4*z' ≤ max_w) ∧
  max_w = 3 ∧ w ≤ 6 :=
sorry

end constrained_optimization_l780_78013


namespace new_boarders_count_new_boarders_joined_school_l780_78014

theorem new_boarders_count (initial_boarders : ℕ) (initial_ratio_boarders : ℕ) (initial_ratio_day : ℕ)
                            (new_ratio_boarders : ℕ) (new_ratio_day : ℕ) : ℕ :=
  let initial_day_students := initial_boarders * initial_ratio_day / initial_ratio_boarders
  let new_boarders := initial_day_students * new_ratio_boarders / new_ratio_day - initial_boarders
  new_boarders

theorem new_boarders_joined_school :
  new_boarders_count 60 2 5 1 2 = 15 := by
  sorry

end new_boarders_count_new_boarders_joined_school_l780_78014


namespace simple_interest_rate_for_doubling_l780_78025

/-- 
Given a sum of money that doubles itself in 10 years at simple interest,
prove that the rate percent per annum is 10%.
-/
theorem simple_interest_rate_for_doubling (P : ℝ) (h : P > 0) : 
  ∃ (R : ℝ), R > 0 ∧ R ≤ 100 ∧ P + (P * R * 10) / 100 = 2 * P ∧ R = 10 := by
  sorry

end simple_interest_rate_for_doubling_l780_78025


namespace point_outside_circle_l780_78036

/-- A circle with a given diameter -/
structure Circle where
  diameter : ℝ
  diameter_pos : diameter > 0

/-- A point with a given distance from the center of a circle -/
structure Point (c : Circle) where
  distance_from_center : ℝ
  distance_pos : distance_from_center > 0

/-- Definition of a point being outside a circle -/
def is_outside (c : Circle) (p : Point c) : Prop :=
  p.distance_from_center > c.diameter / 2

theorem point_outside_circle (c : Circle) (p : Point c) 
  (h_diam : c.diameter = 10) 
  (h_dist : p.distance_from_center = 6) : 
  is_outside c p := by
  sorry

end point_outside_circle_l780_78036


namespace odd_function_derivative_l780_78008

theorem odd_function_derivative (f : ℝ → ℝ) (x₀ : ℝ) (k : ℝ) :
  (∀ x, f (-x) = -f x) →
  Differentiable ℝ f →
  deriv f (-x₀) = k →
  k ≠ 0 →
  deriv f x₀ = k :=
by sorry

end odd_function_derivative_l780_78008


namespace third_term_of_arithmetic_sequence_l780_78021

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℕ → ℝ
  | 0 => a₁
  | k + 1 => a₁ + k * d

theorem third_term_of_arithmetic_sequence :
  let a₁ : ℝ := 11
  let a₆ : ℝ := 39
  let n : ℕ := 6
  let d : ℝ := (a₆ - a₁) / (n - 1)
  arithmetic_sequence a₁ d n 2 = 22.2 := by
  sorry

end third_term_of_arithmetic_sequence_l780_78021


namespace salary_calculation_l780_78053

/-- Given a series of salary changes and a final salary, calculate the original salary --/
theorem salary_calculation (S : ℝ) : 
  S * 1.12 * 0.93 * 1.15 * 0.90 = 5204.21 → S = 5504.00 := by
  sorry

end salary_calculation_l780_78053


namespace quadratic_square_completion_l780_78028

theorem quadratic_square_completion (p q : ℤ) : 
  (∀ x : ℝ, x^2 - 6*x + 3 = 0 ↔ (x + p)^2 = q) → p + q = 3 :=
by sorry

end quadratic_square_completion_l780_78028


namespace vertically_opposite_angles_equal_l780_78005

-- Define a type for angles
def Angle : Type := ℝ

-- Define a function to represent vertically opposite angles
def verticallyOpposite (α β : Angle) : Prop := sorry

-- Theorem: Vertically opposite angles are equal
theorem vertically_opposite_angles_equal (α β : Angle) :
  verticallyOpposite α β → α = β :=
sorry

end vertically_opposite_angles_equal_l780_78005


namespace min_bushes_for_zucchinis_l780_78065

/-- The number of containers of blueberries per bush -/
def containers_per_bush : ℕ := 10

/-- The number of containers of blueberries needed to trade for one zucchini -/
def containers_per_zucchini : ℚ := 4 / 3

/-- The number of zucchinis Natalie wants to obtain -/
def target_zucchinis : ℕ := 72

/-- The minimum number of bushes needed to obtain at least the target number of zucchinis -/
def min_bushes_needed : ℕ :=
  (target_zucchinis * containers_per_zucchini / containers_per_bush).ceil.toNat

theorem min_bushes_for_zucchinis :
  min_bushes_needed = 10 := by sorry

end min_bushes_for_zucchinis_l780_78065


namespace initial_kittens_initial_kittens_is_18_l780_78046

/-- The number of kittens Tim gave to Jessica -/
def kittens_to_jessica : ℕ := 3

/-- The number of kittens Tim gave to Sara -/
def kittens_to_sara : ℕ := 6

/-- The number of kittens Tim has left -/
def kittens_left : ℕ := 9

/-- Theorem stating the initial number of kittens Tim had -/
theorem initial_kittens : ℕ :=
  kittens_to_jessica + kittens_to_sara + kittens_left

/-- Proof that the initial number of kittens is 18 -/
theorem initial_kittens_is_18 : initial_kittens = 18 := by
  sorry

end initial_kittens_initial_kittens_is_18_l780_78046


namespace librarian_took_books_oliver_book_problem_l780_78091

theorem librarian_took_books (total_books : ℕ) (books_per_shelf : ℕ) (shelves_needed : ℕ) : ℕ :=
  let remaining_books := shelves_needed * books_per_shelf
  total_books - remaining_books

theorem oliver_book_problem :
  librarian_took_books 46 4 9 = 10 := by
  sorry

end librarian_took_books_oliver_book_problem_l780_78091


namespace set_intersection_complement_l780_78016

/-- Given sets A and B, if the intersection of the complement of A and B equals B,
    then m is less than or equal to -11 or greater than or equal to 3. -/
theorem set_intersection_complement (m : ℝ) : 
  let A : Set ℝ := {x | -2 < x ∧ x < 3}
  let B : Set ℝ := {x | m < x ∧ x < m + 9}
  (Aᶜ ∩ B = B) → (m ≤ -11 ∨ m ≥ 3) :=
by
  sorry


end set_intersection_complement_l780_78016


namespace exactly_three_imply_l780_78034

open Classical

variables (p q r : Prop)

def statement1 : Prop := ¬p ∧ q ∧ ¬r
def statement2 : Prop := p ∧ ¬q ∧ ¬r
def statement3 : Prop := ¬p ∧ ¬q ∧ r
def statement4 : Prop := p ∧ q ∧ ¬r

def implication : Prop := (¬p → ¬q) → ¬r

theorem exactly_three_imply :
  ∃! (n : Nat), n = 3 ∧
  (n = (if statement1 p q r → implication p q r then 1 else 0) +
       (if statement2 p q r → implication p q r then 1 else 0) +
       (if statement3 p q r → implication p q r then 1 else 0) +
       (if statement4 p q r → implication p q r then 1 else 0)) :=
by sorry

end exactly_three_imply_l780_78034


namespace largest_choir_size_l780_78035

theorem largest_choir_size : 
  ∃ (n : ℕ), 
    (∃ (k : ℕ), n = k^2 + 11) ∧ 
    (∃ (m : ℕ), n = m * (m + 5)) ∧
    (∀ (x : ℕ), 
      ((∃ (k : ℕ), x = k^2 + 11) ∧ 
       (∃ (m : ℕ), x = m * (m + 5))) → 
      x ≤ n) ∧
    n = 325 :=
by sorry

end largest_choir_size_l780_78035


namespace g_negative_three_l780_78094

-- Define the function g
def g (d e f : ℝ) (x : ℝ) : ℝ := d * x^5 + e * x^3 + f * x + 6

-- State the theorem
theorem g_negative_three (d e f : ℝ) : g d e f 3 = -9 → g d e f (-3) = 21 := by
  sorry

end g_negative_three_l780_78094


namespace sum_of_variables_l780_78077

theorem sum_of_variables (x y z : ℝ) 
  (eq1 : y + z = 10 - 2*x)
  (eq2 : x + z = -12 - 4*y)
  (eq3 : x + y = 5 - 2*z) :
  2*x + 2*y + 2*z = 3 := by
sorry

end sum_of_variables_l780_78077


namespace opposite_of_negative_six_l780_78031

-- Define the concept of opposite
def opposite (a : ℝ) : ℝ := -a

-- State the theorem
theorem opposite_of_negative_six :
  opposite (-6) = 6 := by
  sorry

end opposite_of_negative_six_l780_78031


namespace line_y_intercept_l780_78049

/-- A line with slope 3 and x-intercept (-4, 0) has y-intercept (0, 12) -/
theorem line_y_intercept (slope : ℝ) (x_intercept : ℝ × ℝ) :
  slope = 3 ∧ x_intercept = (-4, 0) →
  ∃ (y : ℝ), (∀ (x : ℝ), y = slope * x + (slope * x_intercept.1 + x_intercept.2)) ∧
              y = slope * 0 + (slope * x_intercept.1 + x_intercept.2) ∧
              (0, y) = (0, 12) :=
by sorry


end line_y_intercept_l780_78049


namespace divisibility_of_p_and_q_l780_78000

def ones (n : ℕ) : ℕ := (10^n - 1) / 9

def p (n : ℕ) : ℕ := ones n * (10^(3*n) + 9*10^(2*n) + 8*10^n + 7)

def q (n : ℕ) : ℕ := ones (n+1) * (10^(3*(n+1)) + 9*10^(2*(n+1)) + 8*10^(n+1) + 7)

theorem divisibility_of_p_and_q (n : ℕ) (h : 1987 ∣ ones n) : 
  1987 ∣ p n ∧ 1987 ∣ q n := by
  sorry

end divisibility_of_p_and_q_l780_78000


namespace sector_angle_values_l780_78018

-- Define a sector
structure Sector where
  radius : ℝ
  centralAngle : ℝ

-- Define the perimeter and area of a sector
def perimeter (s : Sector) : ℝ := 2 * s.radius + s.radius * s.centralAngle
def area (s : Sector) : ℝ := 0.5 * s.radius * s.radius * s.centralAngle

-- Theorem statement
theorem sector_angle_values :
  ∃ s : Sector, perimeter s = 6 ∧ area s = 2 ∧ (s.centralAngle = 1 ∨ s.centralAngle = 4) :=
sorry

end sector_angle_values_l780_78018


namespace binary_operation_proof_l780_78075

/-- Convert a binary number (represented as a list of bits) to a natural number -/
def binary_to_nat (bits : List Bool) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

/-- The first binary number 11001₂ -/
def num1 : List Bool := [true, true, false, false, true]

/-- The second binary number 1101₂ -/
def num2 : List Bool := [true, true, false, true]

/-- The third binary number 101₂ -/
def num3 : List Bool := [true, false, true]

/-- The result 100111010₂ -/
def result : List Bool := [true, false, false, true, true, true, false, true, false]

/-- Theorem stating that (11001₂ * 1101₂) - 101₂ = 100111010₂ -/
theorem binary_operation_proof :
  (binary_to_nat num1 * binary_to_nat num2) - binary_to_nat num3 = binary_to_nat result := by
  sorry

end binary_operation_proof_l780_78075


namespace cart_distance_theorem_l780_78092

/-- Represents a cart with two wheels -/
structure Cart where
  front_wheel_circumference : ℝ
  back_wheel_circumference : ℝ

/-- Calculates the distance traveled by the cart -/
def distance_traveled (c : Cart) (back_revolutions : ℝ) : ℝ :=
  back_revolutions * c.back_wheel_circumference

/-- Theorem stating the distance traveled by the cart -/
theorem cart_distance_theorem (c : Cart) 
    (h1 : c.front_wheel_circumference = 30)
    (h2 : c.back_wheel_circumference = 32)
    (h3 : ∃ (r : ℝ), r * c.back_wheel_circumference = (r + 5) * c.front_wheel_circumference) :
  ∃ (r : ℝ), distance_traveled c r = 2400 := by
  sorry

#check cart_distance_theorem

end cart_distance_theorem_l780_78092


namespace parallelogram_perimeter_example_l780_78062

/-- A parallelogram with side lengths a and b -/
structure Parallelogram where
  a : ℝ
  b : ℝ

/-- The perimeter of a parallelogram -/
def perimeter (p : Parallelogram) : ℝ := 2 * (p.a + p.b)

theorem parallelogram_perimeter_example : 
  let p : Parallelogram := { a := 10, b := 7 }
  perimeter p = 34 := by
  sorry

end parallelogram_perimeter_example_l780_78062


namespace option_c_is_experimental_l780_78042

-- Define a type for survey methods
inductive SurveyMethod
| Direct
| Experimental
| SecondaryData

-- Define a type for survey options
inductive SurveyOption
| A
| B
| C
| D

-- Define a function that assigns a survey method to each option
def survey_method (option : SurveyOption) : SurveyMethod :=
  match option with
  | SurveyOption.A => SurveyMethod.Direct
  | SurveyOption.B => SurveyMethod.Direct
  | SurveyOption.C => SurveyMethod.Experimental
  | SurveyOption.D => SurveyMethod.SecondaryData

-- Define the experimental method suitability
def is_suitable_for_experimental (method : SurveyMethod) : Prop :=
  method = SurveyMethod.Experimental

-- Theorem: Option C is the only one suitable for the experimental method
theorem option_c_is_experimental :
  ∀ (option : SurveyOption),
    is_suitable_for_experimental (survey_method option) ↔ option = SurveyOption.C :=
by
  sorry

#check option_c_is_experimental

end option_c_is_experimental_l780_78042


namespace teachers_in_school_l780_78017

/-- Calculates the number of teachers required in a school --/
def teachers_required (total_students : ℕ) (lessons_per_student : ℕ) (lessons_per_teacher : ℕ) (students_per_class : ℕ) : ℕ :=
  (total_students * lessons_per_student) / (students_per_class * lessons_per_teacher)

/-- Theorem stating that 50 teachers are required given the specific conditions --/
theorem teachers_in_school : 
  teachers_required 1200 5 4 30 = 50 := by
  sorry

#eval teachers_required 1200 5 4 30

end teachers_in_school_l780_78017


namespace expected_digits_icosahedral_die_l780_78073

def icosahedral_die := Finset.range 20

theorem expected_digits_icosahedral_die :
  let digits_function := fun n => if n < 10 then 1 else 2
  let expected_value := (icosahedral_die.sum fun i => digits_function (i + 1)) / icosahedral_die.card
  expected_value = 31 / 20 := by
sorry

end expected_digits_icosahedral_die_l780_78073


namespace fraction_decimal_digits_l780_78067

/-- The fraction we're considering -/
def fraction : ℚ := 987654321 / (2^30 * 5^2 * 3)

/-- The minimum number of digits to the right of the decimal point -/
def min_decimal_digits : ℕ := 30

/-- Theorem stating that the minimum number of digits to the right of the decimal point
    needed to express the fraction as a decimal is equal to min_decimal_digits -/
theorem fraction_decimal_digits :
  (∀ n : ℕ, n < min_decimal_digits → ∃ m : ℕ, fraction * 10^n ≠ m) ∧
  (∃ m : ℕ, fraction * 10^min_decimal_digits = m) :=
sorry

end fraction_decimal_digits_l780_78067


namespace point_product_l780_78081

theorem point_product (y₁ y₂ : ℝ) : 
  ((-4 - 7)^2 + (y₁ - 3)^2 = 13^2) →
  ((-4 - 7)^2 + (y₂ - 3)^2 = 13^2) →
  y₁ ≠ y₂ →
  y₁ * y₂ = -39 := by
sorry

end point_product_l780_78081


namespace largest_x_abs_equation_l780_78088

theorem largest_x_abs_equation : ∃ (x : ℝ), x = 7 ∧ |x + 3| = 10 ∧ ∀ y : ℝ, |y + 3| = 10 → y ≤ x := by
  sorry

end largest_x_abs_equation_l780_78088


namespace train_passing_jogger_time_l780_78022

/-- Calculates the time it takes for a train to pass a jogger given their speeds and initial positions -/
theorem train_passing_jogger_time
  (jogger_speed : ℝ)
  (train_speed : ℝ)
  (train_length : ℝ)
  (initial_distance : ℝ)
  (h1 : jogger_speed = 10)
  (h2 : train_speed = 46)
  (h3 : train_length = 120)
  (h4 : initial_distance = 340)
  : (initial_distance + train_length) / (train_speed - jogger_speed) * (3600 / 1000) = 46 := by
  sorry

#check train_passing_jogger_time

end train_passing_jogger_time_l780_78022


namespace root_equivalence_l780_78097

theorem root_equivalence (α : ℂ) : 
  α^2 - 2*α - 2 = 0 → α^5 - 44*α^3 - 32*α^2 - 2 = 0 := by
sorry

end root_equivalence_l780_78097


namespace gcd_division_remainder_l780_78030

theorem gcd_division_remainder (a b : ℕ) (h1 : a > b) (h2 : ∃ q r : ℕ, a = b * q + r ∧ 0 < r ∧ r < b) :
  Nat.gcd a b = Nat.gcd b (a % b) :=
by sorry

end gcd_division_remainder_l780_78030


namespace kennel_arrangement_count_l780_78066

/-- The number of chickens in the kennel -/
def num_chickens : Nat := 4

/-- The number of dogs in the kennel -/
def num_dogs : Nat := 3

/-- The number of cats in the kennel -/
def num_cats : Nat := 5

/-- The total number of animals in the kennel -/
def total_animals : Nat := num_chickens + num_dogs + num_cats

/-- The number of ways to arrange animals within their groups -/
def intra_group_arrangements : Nat := (Nat.factorial num_chickens) * (Nat.factorial num_dogs) * (Nat.factorial num_cats)

/-- The number of valid group orders (chickens-dogs-cats and chickens-cats-dogs) -/
def valid_group_orders : Nat := 2

/-- The total number of ways to arrange the animals -/
def total_arrangements : Nat := valid_group_orders * intra_group_arrangements

theorem kennel_arrangement_count :
  total_arrangements = 34560 :=
sorry

end kennel_arrangement_count_l780_78066


namespace jeff_average_skips_l780_78096

def jeff_skips (sam_skips : ℕ) (rounds : ℕ) : List ℕ :=
  let round1 := sam_skips - 1
  let round2 := sam_skips - 3
  let round3 := sam_skips + 4
  let round4 := sam_skips / 2
  let round5 := round4 + (sam_skips - round4 + 2)
  [round1, round2, round3, round4, round5]

def average_skips (skips : List ℕ) : ℚ :=
  (skips.sum : ℚ) / skips.length

theorem jeff_average_skips (sam_skips : ℕ) (rounds : ℕ) :
  sam_skips = 16 ∧ rounds = 5 →
  average_skips (jeff_skips sam_skips rounds) = 74/5 :=
by sorry

end jeff_average_skips_l780_78096


namespace number_difference_l780_78085

theorem number_difference (a b : ℕ) 
  (sum_eq : a + b = 21800)
  (b_div_100 : 100 ∣ b)
  (a_eq_b_div_100 : a = b / 100) :
  b - a = 21384 := by sorry

end number_difference_l780_78085


namespace exists_homogeneous_polynomial_for_irreducible_lattice_points_l780_78055

-- Define an irreducible lattice point
def irreducible_lattice_point (p : ℤ × ℤ) : Prop :=
  Int.gcd p.1 p.2 = 1

-- Define a homogeneous polynomial with integer coefficients
def homogeneous_polynomial (f : ℤ → ℤ → ℤ) (d : ℕ) : Prop :=
  ∀ (c : ℤ) (x y : ℤ), f (c * x) (c * y) = c^d * f x y

-- The main theorem
theorem exists_homogeneous_polynomial_for_irreducible_lattice_points 
  (S : Finset (ℤ × ℤ)) (h : ∀ p ∈ S, irreducible_lattice_point p) :
  ∃ (f : ℤ → ℤ → ℤ) (d : ℕ), 
    d ≥ 1 ∧ 
    homogeneous_polynomial f d ∧ 
    (∀ p ∈ S, f p.1 p.2 = 1) := by
  sorry


end exists_homogeneous_polynomial_for_irreducible_lattice_points_l780_78055


namespace isosceles_right_triangle_hypotenuse_l780_78058

theorem isosceles_right_triangle_hypotenuse (square_side : ℝ) (triangle_leg : ℝ) : 
  square_side = 2 →
  4 * (1/2 * triangle_leg^2) = square_side^2 →
  triangle_leg^2 + triangle_leg^2 = 4 := by
  sorry

end isosceles_right_triangle_hypotenuse_l780_78058


namespace sum_of_positive_factors_36_l780_78070

-- Define the sum of positive factors function
def sumOfPositiveFactors (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem sum_of_positive_factors_36 : sumOfPositiveFactors 36 = 91 := by
  sorry

end sum_of_positive_factors_36_l780_78070


namespace count_non_dividing_eq_29_l780_78020

/-- g(n) is the product of proper positive integer divisors of n -/
def g (n : ℕ) : ℕ := sorry

/-- count_non_dividing counts the number of integers n between 2 and 100 (inclusive) 
    for which n does not divide g(n) -/
def count_non_dividing : ℕ := sorry

/-- Theorem stating that the count of integers n between 2 and 100 (inclusive) 
    for which n does not divide g(n) is equal to 29 -/
theorem count_non_dividing_eq_29 : count_non_dividing = 29 := by sorry

end count_non_dividing_eq_29_l780_78020


namespace diophantine_equation_solution_l780_78074

theorem diophantine_equation_solution :
  ∀ x y z : ℤ, 2*x^2 + 2*x^2*z^2 + z^2 + 7*y^2 - 42*y + 33 = 0 ↔
  (x = 1 ∧ y = 5 ∧ z = 0) ∨
  (x = -1 ∧ y = 5 ∧ z = 0) ∨
  (x = 1 ∧ y = 1 ∧ z = 0) ∨
  (x = -1 ∧ y = 1 ∧ z = 0) :=
by sorry

end diophantine_equation_solution_l780_78074


namespace circle_tangent_to_ellipse_l780_78045

/-- Two circles of radius s are externally tangent to each other and internally tangent to the ellipse x^2 + 4y^2 = 8. The radius s of the circles is √(3/2). -/
theorem circle_tangent_to_ellipse (s : ℝ) : 
  (∃ (x y : ℝ), x^2 + 4*y^2 = 8 ∧ (x - s)^2 + y^2 = s^2) → s = Real.sqrt (3/2) := by
  sorry

end circle_tangent_to_ellipse_l780_78045


namespace point_on_x_axis_l780_78072

/-- Given a point P with coordinates (4, 2a+10), prove that if P lies on the x-axis, then a = -5 -/
theorem point_on_x_axis (a : ℝ) : 
  let P : ℝ × ℝ := (4, 2*a + 10)
  (P.2 = 0) → a = -5 := by
  sorry

end point_on_x_axis_l780_78072


namespace heath_carrot_planting_l780_78026

/-- Given the conditions of Heath's carrot planting, prove the number of plants in each row. -/
theorem heath_carrot_planting 
  (total_rows : ℕ) 
  (planting_time : ℕ) 
  (planting_rate : ℕ) 
  (h1 : total_rows = 400)
  (h2 : planting_time = 20)
  (h3 : planting_rate = 6000) :
  (planting_time * planting_rate) / total_rows = 300 := by
  sorry

end heath_carrot_planting_l780_78026


namespace geometric_sequence_sum_l780_78023

/-- A geometric sequence with positive terms. -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n, a (n + 1) = a n * r ∧ a n > 0

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25 →
  a 3 + a 5 = 5 := by
sorry

end geometric_sequence_sum_l780_78023


namespace geralds_initial_notebooks_l780_78024

theorem geralds_initial_notebooks (jack_initial gerald_initial jack_remaining paula_given mike_given : ℕ) : 
  jack_initial = gerald_initial + 13 →
  jack_initial = jack_remaining + paula_given + mike_given →
  jack_remaining = 10 →
  paula_given = 5 →
  mike_given = 6 →
  gerald_initial = 8 := by
sorry

end geralds_initial_notebooks_l780_78024


namespace nearest_whole_number_to_24567_4999997_l780_78050

theorem nearest_whole_number_to_24567_4999997 :
  let x : ℝ := 24567.4999997
  ∃ (n : ℤ), ∀ (m : ℤ), |x - n| ≤ |x - m| ∧ n = 24567 :=
sorry

end nearest_whole_number_to_24567_4999997_l780_78050


namespace cards_lost_ratio_l780_78052

/-- Represents the number of cards Phil buys each week -/
def cards_per_week : ℕ := 20

/-- Represents the number of weeks in a year -/
def weeks_in_year : ℕ := 52

/-- Represents the number of cards Phil has left after the fire -/
def cards_left : ℕ := 520

/-- Theorem stating that the ratio of cards lost to total cards before the fire is 1:2 -/
theorem cards_lost_ratio :
  let total_cards := cards_per_week * weeks_in_year
  let lost_cards := total_cards - cards_left
  (lost_cards : ℚ) / total_cards = 1 / 2 := by sorry

end cards_lost_ratio_l780_78052


namespace french_exam_min_words_to_learn_l780_78007

/-- The minimum number of words to learn for a 90% score on a French vocabulary exam -/
theorem french_exam_min_words_to_learn :
  ∀ (total_words : ℕ) (guess_success_rate : ℚ) (target_score : ℚ),
    total_words = 800 →
    guess_success_rate = 1/10 →
    target_score = 9/10 →
    ∃ (words_to_learn : ℕ),
      words_to_learn ≥ 712 ∧
      (words_to_learn : ℚ) / total_words +
        guess_success_rate * ((total_words : ℚ) - words_to_learn) / total_words ≥ target_score :=
by sorry

end french_exam_min_words_to_learn_l780_78007


namespace quadratic_zero_discriminant_l780_78019

/-- The quadratic equation 5x^2 - 10x√3 + k = 0 has zero discriminant if and only if k = 15 -/
theorem quadratic_zero_discriminant (k : ℝ) :
  (∀ x : ℝ, 5 * x^2 - 10 * x * Real.sqrt 3 + k = 0) →
  ((-10 * Real.sqrt 3)^2 - 4 * 5 * k = 0) ↔
  k = 15 := by
sorry

end quadratic_zero_discriminant_l780_78019


namespace range_of_m_for_increasing_f_l780_78002

/-- A quadratic function f(x) = 4x^2 - mx + 5 that is increasing on [-2, +∞) -/
def f (m : ℝ) (x : ℝ) : ℝ := 4 * x^2 - m * x + 5

/-- The property that f is increasing on [-2, +∞) -/
def is_increasing_on_interval (m : ℝ) : Prop :=
  ∀ x y, x ≥ -2 → y ≥ -2 → x < y → f m x < f m y

/-- The theorem stating the range of m for which f is increasing on [-2, +∞) -/
theorem range_of_m_for_increasing_f :
  ∀ m : ℝ, is_increasing_on_interval m → m ≤ -16 :=
sorry

end range_of_m_for_increasing_f_l780_78002


namespace sum_of_roots_l780_78006

theorem sum_of_roots (a b c d : ℝ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  (∀ x : ℝ, x^2 - 14*a*x + 15*b = 0 ↔ x = c ∨ x = d) →
  (∀ x : ℝ, x^2 - 14*c*x - 15*d = 0 ↔ x = a ∨ x = b) →
  a + b + c + d = 3150 := by
sorry

end sum_of_roots_l780_78006


namespace arithmetic_progression_rth_term_l780_78056

/-- The sum of the first n terms of an arithmetic progression -/
def S (n : ℕ) : ℝ := 3 * n^2 + 4 * n + 5

/-- The r-th term of the arithmetic progression -/
def a (r : ℕ) : ℝ := 6 * r + 1

theorem arithmetic_progression_rth_term (r : ℕ) : 
  a r = S r - S (r - 1) :=
sorry

end arithmetic_progression_rth_term_l780_78056


namespace intersection_of_A_and_B_l780_78051

-- Define set A
def A : Set ℝ := {y | ∃ x, y = Real.exp x}

-- Define set B
def B : Set ℝ := {x | x^2 - x - 6 ≤ 0}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = Set.Ioo 0 3 := by sorry

end intersection_of_A_and_B_l780_78051


namespace quadratic_roots_condition_l780_78038

theorem quadratic_roots_condition (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ m * x^2 + 2 * x - 1 = 0 ∧ m * y^2 + 2 * y - 1 = 0) ↔ 
  (m > -1 ∧ m ≠ 0) :=
by sorry

end quadratic_roots_condition_l780_78038


namespace point_on_k_graph_l780_78076

-- Define the functions f and k
variable (f : ℝ → ℝ)
variable (k : ℝ → ℝ)

-- State the theorem
theorem point_on_k_graph (h1 : f 4 = 8) (h2 : ∀ x, k x = (f x)^3) :
  ∃ x y : ℝ, k x = y ∧ x + y = 516 := by
sorry

end point_on_k_graph_l780_78076


namespace smallest_t_for_no_h_route_l780_78078

/-- Represents a chessboard --/
structure Chessboard :=
  (size : Nat)
  (h_size : size = 8)

/-- Represents a Horse's move --/
structure HorseMove :=
  (horizontal : Nat)
  (vertical : Nat)

/-- Represents an H Route --/
def HRoute (board : Chessboard) (move : HorseMove) : Prop :=
  ∃ (path : List (Nat × Nat)), 
    path.length = board.size * board.size ∧
    ∀ (pos : Nat × Nat), pos ∈ path → 
      pos.1 ≤ board.size ∧ pos.2 ≤ board.size

/-- The main theorem --/
theorem smallest_t_for_no_h_route : 
  ∀ (board : Chessboard),
    ∀ (t : Nat),
      t > 0 →
      (∀ (start : Nat × Nat), 
        start.1 ≤ board.size ∧ start.2 ≤ board.size →
        ¬ HRoute board ⟨t, t+1⟩) →
      t = 2 :=
sorry

end smallest_t_for_no_h_route_l780_78078


namespace kaleb_boxes_correct_l780_78069

/-- The number of boxes Kaleb bought initially -/
def initial_boxes : ℕ := 9

/-- The number of boxes Kaleb gave to his little brother -/
def given_boxes : ℕ := 5

/-- The number of pieces in each box -/
def pieces_per_box : ℕ := 6

/-- The number of pieces Kaleb still has -/
def remaining_pieces : ℕ := 54

/-- Theorem stating that the initial number of boxes is correct -/
theorem kaleb_boxes_correct :
  initial_boxes * pieces_per_box = remaining_pieces + given_boxes * pieces_per_box :=
by sorry

end kaleb_boxes_correct_l780_78069


namespace sin_45_degrees_l780_78004

theorem sin_45_degrees :
  let r : ℝ := 1  -- radius of the unit circle
  let θ : ℝ := Real.pi / 4  -- 45° in radians
  let Q : ℝ × ℝ := (r * Real.cos θ, r * Real.sin θ)  -- point on the circle at 45°
  let E : ℝ × ℝ := (Q.1, 0)  -- foot of the perpendicular from Q to x-axis
  Real.sin θ = 1 / Real.sqrt 2 :=
by sorry

end sin_45_degrees_l780_78004


namespace g10_diamonds_l780_78015

/-- Number of diamonds in figure G_n -/
def num_diamonds (n : ℕ) : ℕ :=
  if n = 1 then 2
  else if n = 2 then 10
  else 2 + 2 * n^2 + 2 * n - 4

/-- The sequence of figures G_n satisfies the given properties -/
axiom sequence_property (n : ℕ) (h : n ≥ 3) :
  num_diamonds n = num_diamonds (n - 1) + 4 * (n + 1)

/-- G_1 has 2 diamonds -/
axiom g1_diamonds : num_diamonds 1 = 2

/-- G_2 has 10 diamonds -/
axiom g2_diamonds : num_diamonds 2 = 10

/-- Theorem: G_10 has 218 diamonds -/
theorem g10_diamonds : num_diamonds 10 = 218 := by sorry

end g10_diamonds_l780_78015


namespace pool_depths_l780_78089

/-- Pool depths problem -/
theorem pool_depths (john sarah susan mike : ℕ) : 
  john = 2 * sarah + 5 →  -- John's pool is 5 feet deeper than 2 times Sarah's pool
  john = 15 →  -- John's pool is 15 feet deep
  susan = john + sarah - 3 →  -- Susan's pool is 3 feet shallower than the sum of John's and Sarah's pool depths
  mike = john + sarah + susan + 4 →  -- Mike's pool is 4 feet deeper than the combined depth of John's, Sarah's, and Susan's pools
  sarah = 5 ∧ susan = 17 ∧ mike = 41 := by
  sorry

end pool_depths_l780_78089


namespace polynomial_constant_term_product_l780_78009

variable (p q r : ℝ[X])

theorem polynomial_constant_term_product 
  (h1 : r = p * q)
  (h2 : p.coeff 0 = 6)
  (h3 : r.coeff 0 = -18) :
  q.eval 0 = -3 := by
  sorry

end polynomial_constant_term_product_l780_78009


namespace no_integer_solutions_for_20122012_l780_78039

theorem no_integer_solutions_for_20122012 :
  ¬∃ (a b c : ℤ), a^2 + b^2 + c^2 = 20122012 := by
  sorry

end no_integer_solutions_for_20122012_l780_78039


namespace production_days_l780_78003

theorem production_days (n : ℕ) 
  (h1 : (50 : ℝ) * n = n * 50)
  (h2 : (50 : ℝ) * n + 115 = (n + 1) * 55) : n = 12 := by
  sorry

end production_days_l780_78003


namespace travel_options_count_l780_78001

/-- The number of flights from A to B in one day -/
def num_flights : ℕ := 3

/-- The number of trains from A to B in one day -/
def num_trains : ℕ := 2

/-- The total number of ways to travel from A to B in one day -/
def total_ways : ℕ := num_flights + num_trains

theorem travel_options_count : total_ways = 5 := by sorry

end travel_options_count_l780_78001


namespace excess_donation_l780_78029

/-- Trader's profit calculation -/
def trader_profit : ℝ := 1200

/-- Allocation percentage for next shipment -/
def allocation_percentage : ℝ := 0.60

/-- Family donation amount -/
def family_donation : ℝ := 250

/-- Friends donation calculation -/
def friends_donation : ℝ := family_donation * 1.20

/-- Local association donation calculation -/
def local_association_donation : ℝ := (family_donation + friends_donation) * 1.5

/-- Total donations received -/
def total_donations : ℝ := family_donation + friends_donation + local_association_donation

/-- Allocated amount for next shipment -/
def allocated_amount : ℝ := trader_profit * allocation_percentage

/-- Theorem: The difference between total donations and allocated amount is $655 -/
theorem excess_donation : total_donations - allocated_amount = 655 := by sorry

end excess_donation_l780_78029


namespace tunnel_length_tunnel_length_proof_l780_78090

/-- Calculates the length of a tunnel given train and time information -/
theorem tunnel_length (train_length : ℝ) (train_speed : ℝ) (exit_time : ℝ) : ℝ :=
  let tunnel_length := train_speed * exit_time / 60 - train_length
  2

theorem tunnel_length_proof :
  tunnel_length 1 60 3 = 2 := by sorry

end tunnel_length_tunnel_length_proof_l780_78090


namespace sqrt_equation_solution_l780_78068

theorem sqrt_equation_solution (y : ℝ) : 
  (Real.sqrt 1.21) / (Real.sqrt y) + (Real.sqrt 1.44) / (Real.sqrt 0.49) = 2.9365079365079367 → 
  y = 0.81 := by
sorry

end sqrt_equation_solution_l780_78068

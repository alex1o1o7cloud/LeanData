import Mathlib

namespace remainder_of_b_mod_29_l766_76694

theorem remainder_of_b_mod_29 :
  let b := (((13⁻¹ : ZMod 29) + (17⁻¹ : ZMod 29) + (19⁻¹ : ZMod 29))⁻¹ : ZMod 29)
  b = 2 := by sorry

end remainder_of_b_mod_29_l766_76694


namespace quadratic_real_roots_condition_l766_76623

theorem quadratic_real_roots_condition (m : ℝ) : 
  (∃ x : ℝ, x^2 + 2*x + m = 0) → m ≤ 1 := by
  sorry

end quadratic_real_roots_condition_l766_76623


namespace tv_show_cost_per_episode_l766_76697

/-- Given a TV show season with the following properties:
  * The season has 22 episodes
  * The total cost of the season is $35,200
  * The second half of the season costs 120% more per episode than the first half
  Prove that the cost per episode for the first half of the season is $1,000. -/
theorem tv_show_cost_per_episode 
  (total_episodes : ℕ) 
  (total_cost : ℚ) 
  (second_half_increase : ℚ) :
  total_episodes = 22 →
  total_cost = 35200 →
  second_half_increase = 1.2 →
  let first_half_cost := total_cost / (total_episodes / 2 * (1 + 1 + second_half_increase))
  first_half_cost = 1000 := by
sorry

end tv_show_cost_per_episode_l766_76697


namespace scooter_repair_percentage_l766_76672

theorem scooter_repair_percentage (profit_percentage : ℝ) (profit_amount : ℝ) (repair_cost : ℝ) :
  profit_percentage = 0.2 →
  profit_amount = 1100 →
  repair_cost = 500 →
  (repair_cost / (profit_amount / profit_percentage)) * 100 = 500 / 5500 * 100 := by
sorry

end scooter_repair_percentage_l766_76672


namespace apples_per_box_l766_76615

/-- Proves that the number of apples per box is 50 given the total number of apples,
    the desired amount to take home, and the price per box. -/
theorem apples_per_box
  (total_apples : ℕ)
  (take_home_amount : ℕ)
  (price_per_box : ℕ)
  (h1 : total_apples = 10000)
  (h2 : take_home_amount = 7000)
  (h3 : price_per_box = 35) :
  total_apples / (take_home_amount / price_per_box) = 50 := by
  sorry

#check apples_per_box

end apples_per_box_l766_76615


namespace fibonacci_product_theorem_l766_76607

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- The property we want to prove -/
def satisfies_property (m n : ℕ) : Prop :=
  m > 0 ∧ n > 0 ∧ fib m * fib n = m * n

/-- The theorem statement -/
theorem fibonacci_product_theorem :
  ∀ m n : ℕ, satisfies_property m n ↔ (m = 1 ∧ n = 1) ∨ (m = 5 ∧ n = 5) := by
  sorry

end fibonacci_product_theorem_l766_76607


namespace sand_weight_difference_l766_76669

def box_weight : ℕ := 250
def box_filled_weight : ℕ := 1780
def bucket_weight : ℕ := 460
def bucket_filled_weight : ℕ := 2250

theorem sand_weight_difference :
  (bucket_filled_weight - bucket_weight) - (box_filled_weight - box_weight) = 260 :=
by sorry

end sand_weight_difference_l766_76669


namespace percent_of_a_l766_76614

theorem percent_of_a (a b c : ℝ) (h1 : b = 0.5 * a) (h2 : c = 0.5 * b) :
  c = 0.25 * a := by
  sorry

end percent_of_a_l766_76614


namespace derivative_of_square_root_l766_76682

theorem derivative_of_square_root (x : ℝ) (h : x > 0) :
  deriv (fun x => Real.sqrt x) x = 1 / (2 * Real.sqrt x) := by
sorry

end derivative_of_square_root_l766_76682


namespace farmer_tomato_rows_l766_76632

/-- The number of tomato plants in each row -/
def plants_per_row : ℕ := 10

/-- The number of tomatoes yielded by each plant -/
def tomatoes_per_plant : ℕ := 20

/-- The total number of tomatoes harvested by the farmer -/
def total_tomatoes : ℕ := 6000

/-- The number of rows of tomatoes planted by the farmer -/
def rows_of_tomatoes : ℕ := total_tomatoes / (plants_per_row * tomatoes_per_plant)

theorem farmer_tomato_rows : rows_of_tomatoes = 30 := by
  sorry

end farmer_tomato_rows_l766_76632


namespace monotonic_range_a_l766_76692

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + a*x^2 - x - 1

-- State the theorem
theorem monotonic_range_a :
  (∀ a : ℝ, ∀ x : ℝ, Monotone (f a)) ↔ a ∈ Set.Icc (-Real.sqrt 3) (Real.sqrt 3) :=
sorry

end monotonic_range_a_l766_76692


namespace hyperbola_standard_equation_l766_76622

def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 25 - y^2 / 9 = 1 ∨ y^2 / 25 - x^2 / 9 = 1

theorem hyperbola_standard_equation
  (center_origin : ℝ × ℝ)
  (real_axis_length : ℝ)
  (imaginary_axis_length : ℝ)
  (h1 : center_origin = (0, 0))
  (h2 : real_axis_length = 10)
  (h3 : imaginary_axis_length = 6) :
  ∀ x y : ℝ, hyperbola_equation x y := by
sorry

end hyperbola_standard_equation_l766_76622


namespace max_a_value_l766_76675

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * Real.log x - a * x^2 + 3

theorem max_a_value (a : ℝ) :
  (∃ m n : ℝ, m ∈ Set.Icc 1 5 ∧ n ∈ Set.Icc 1 5 ∧ n - m ≥ 2 ∧ f a m = f a n) →
  a ≤ Real.log 3 / 4 :=
sorry

end max_a_value_l766_76675


namespace quadratic_coefficient_not_one_l766_76642

/-- A quadratic equation in x is of the form px^2 + qx + r = 0 where p ≠ 0 -/
def is_quadratic_equation (p q r : ℝ) : Prop := p ≠ 0

theorem quadratic_coefficient_not_one (a : ℝ) :
  is_quadratic_equation (a - 1) (-1) 7 → a ≠ 1 := by
  sorry

end quadratic_coefficient_not_one_l766_76642


namespace fraction_chain_l766_76656

theorem fraction_chain (a b c d e : ℝ) 
  (h1 : a / b = 5)
  (h2 : b / c = 1 / 2)
  (h3 : c / d = 3)
  (h4 : d / e = 1 / 4)
  : e / a = 8 / 15 := by
  sorry

end fraction_chain_l766_76656


namespace triathlon_completion_time_l766_76665

/-- A triathlon participant's speeds and completion time -/
theorem triathlon_completion_time 
  (swim_dist : ℝ) 
  (cycle_dist : ℝ) 
  (run_dist : ℝ) 
  (swim_speed : ℝ) 
  (h1 : swim_dist = 1.5) 
  (h2 : cycle_dist = 40) 
  (h3 : run_dist = 10) 
  (h4 : swim_speed > 0) 
  (h5 : swim_speed * 5 * 2.5 * (swim_dist / swim_speed + run_dist / (5 * swim_speed)) = 
        cycle_dist + swim_speed * 5 * 2.5 * 6) : 
  swim_dist / swim_speed + cycle_dist / (swim_speed * 5 * 2.5) + run_dist / (swim_speed * 5) = 134 :=
by sorry

end triathlon_completion_time_l766_76665


namespace find_divisor_l766_76648

theorem find_divisor (dividend quotient remainder : ℕ) 
  (h1 : dividend = 12401)
  (h2 : quotient = 76)
  (h3 : remainder = 13)
  (h4 : dividend = quotient * 163 + remainder) :
  163 = dividend / quotient :=
by sorry

end find_divisor_l766_76648


namespace total_bathing_suits_l766_76626

theorem total_bathing_suits (men_suits : ℕ) (women_suits : ℕ)
  (h1 : men_suits = 14797)
  (h2 : women_suits = 4969) :
  men_suits + women_suits = 19766 := by
  sorry

end total_bathing_suits_l766_76626


namespace square_divisibility_l766_76647

theorem square_divisibility (n : ℕ+) (h : ∀ m : ℕ+, m ∣ n → m ≤ 12) : 144 ∣ n^2 := by
  sorry

end square_divisibility_l766_76647


namespace sum_of_digits_l766_76657

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def is_single_digit (n : ℕ) : Prop := 0 ≤ n ∧ n ≤ 9

def divisible_by_11 (n : ℕ) : Prop := ∃ k : ℕ, n = 11 * k

theorem sum_of_digits (a b : ℕ) : 
  is_single_digit a → 
  is_single_digit b → 
  is_three_digit (700 + 10 * a + 1) →
  is_three_digit (100 * b + 60 + 5) →
  (700 + 10 * a + 1) + 184 = (100 * b + 60 + 5) →
  divisible_by_11 (100 * b + 60 + 5) →
  a + b = 9 := by
sorry

end sum_of_digits_l766_76657


namespace vector_perpendicular_l766_76645

def a : ℝ × ℝ := (2, 0)
def b : ℝ × ℝ := (1, 1)

theorem vector_perpendicular : (a.1 - b.1) * b.1 + (a.2 - b.2) * b.2 = 0 := by
  sorry

end vector_perpendicular_l766_76645


namespace binomial_expansion_property_l766_76605

theorem binomial_expansion_property (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (2*x + Real.sqrt 3)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) →
  (a₀ + a₂ + a₄)^2 - (a₁ + a₃)^2 = 1 := by
sorry

end binomial_expansion_property_l766_76605


namespace product_of_factors_for_six_factor_number_l766_76661

def has_six_factors (x : ℕ) : Prop :=
  (Finset.filter (· ∣ x) (Finset.range (x + 1))).card = 6

def product_of_factors (x : ℕ) : ℕ :=
  (Finset.filter (· ∣ x) (Finset.range (x + 1))).prod id

theorem product_of_factors_for_six_factor_number (x : ℕ) 
  (h1 : x > 1) (h2 : has_six_factors x) : 
  product_of_factors x = x^3 := by
  sorry

end product_of_factors_for_six_factor_number_l766_76661


namespace smallest_positive_e_l766_76684

def is_root (p : ℝ → ℝ) (r : ℝ) : Prop := p r = 0

theorem smallest_positive_e (a b c d e : ℤ) : 
  let p := fun (x : ℝ) => (a : ℝ) * x^4 + (b : ℝ) * x^3 + (c : ℝ) * x^2 + (d : ℝ) * x + (e : ℝ)
  (is_root p (-3) ∧ is_root p 6 ∧ is_root p 10 ∧ is_root p (-1/2)) →
  (e > 0) →
  (∀ e' : ℤ, e' > 0 → 
    let p' := fun (x : ℝ) => (a : ℝ) * x^4 + (b : ℝ) * x^3 + (c : ℝ) * x^2 + (d : ℝ) * x + (e' : ℝ)
    (is_root p' (-3) ∧ is_root p' 6 ∧ is_root p' 10 ∧ is_root p' (-1/2)) → e' ≥ e) →
  e = 180 := by
sorry

end smallest_positive_e_l766_76684


namespace backpacking_roles_l766_76638

theorem backpacking_roles (n : ℕ) (h : n = 10) : 
  (n.choose 2) * ((n - 2).choose 1) = 360 := by
  sorry

end backpacking_roles_l766_76638


namespace tangent_line_a_range_l766_76688

/-- The line equation ax + y - 2 = 0 -/
def line_equation (a x y : ℝ) : Prop := a * x + y - 2 = 0

/-- The first circle equation (x-1)² + y² = 1 -/
def circle1_equation (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

/-- The second circle equation x² + (y-1)² = 1/4 -/
def circle2_equation (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 1/4

/-- The line is tangent to both circles -/
def is_tangent_to_both_circles (a : ℝ) : Prop :=
  ∃ x y, line_equation a x y ∧ 
         ((circle1_equation x y ∧ ¬∃ x' y', x' ≠ x ∧ y' ≠ y ∧ line_equation a x' y' ∧ circle1_equation x' y') ∨
          (circle2_equation x y ∧ ¬∃ x' y', x' ≠ x ∧ y' ≠ y ∧ line_equation a x' y' ∧ circle2_equation x' y'))

theorem tangent_line_a_range :
  ∀ a : ℝ, is_tangent_to_both_circles a ↔ -Real.sqrt 3 < a ∧ a < 3/4 :=
sorry

end tangent_line_a_range_l766_76688


namespace rowing_current_velocity_l766_76673

/-- Proves that the velocity of the current is 1 kmph given the conditions of the rowing problem. -/
theorem rowing_current_velocity 
  (still_water_speed : ℝ) 
  (distance : ℝ) 
  (total_time : ℝ) 
  (h1 : still_water_speed = 5)
  (h2 : distance = 2.4)
  (h3 : total_time = 1) :
  ∃ v : ℝ, v = 1 ∧ total_time = distance / (still_water_speed + v) + distance / (still_water_speed - v) :=
by sorry

end rowing_current_velocity_l766_76673


namespace jenny_calculation_l766_76650

theorem jenny_calculation (x : ℤ) (h : x - 26 = -14) : x + 26 = 38 := by
  sorry

end jenny_calculation_l766_76650


namespace complex_equation_solution_l766_76606

theorem complex_equation_solution (z : ℂ) : z / (1 - I) = 3 + 2*I → z = 5 - I := by
  sorry

end complex_equation_solution_l766_76606


namespace delivery_problem_l766_76600

theorem delivery_problem (total : ℕ) (cider : ℕ) (beer : ℕ) 
  (h_total : total = 180)
  (h_cider : cider = 40)
  (h_beer : beer = 80) :
  let mixture := total - (cider + beer)
  (cider / 2 + beer / 2 + mixture / 2) = 90 := by
  sorry

end delivery_problem_l766_76600


namespace point_c_in_second_quadrant_l766_76644

/-- Represents a point in a 2D coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point is in the second quadrant -/
def isInSecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Given points -/
def pointA : Point := ⟨5, 3⟩
def pointB : Point := ⟨5, -3⟩
def pointC : Point := ⟨-5, 3⟩
def pointD : Point := ⟨-5, -3⟩

/-- Theorem: Point C is the only point in the second quadrant -/
theorem point_c_in_second_quadrant :
  isInSecondQuadrant pointC ∧
  ¬isInSecondQuadrant pointA ∧
  ¬isInSecondQuadrant pointB ∧
  ¬isInSecondQuadrant pointD :=
by sorry

end point_c_in_second_quadrant_l766_76644


namespace longest_segment_in_quarter_circle_l766_76624

theorem longest_segment_in_quarter_circle (d : ℝ) (h : d = 16) :
  let r := d / 2
  let chord_length := r * Real.sqrt 2
  chord_length ^ 2 = 128 := by sorry

end longest_segment_in_quarter_circle_l766_76624


namespace cost_difference_l766_76640

def ice_cream_quantity : ℕ := 100
def yoghurt_quantity : ℕ := 35
def cheese_quantity : ℕ := 50
def milk_quantity : ℕ := 20

def ice_cream_price : ℚ := 12
def yoghurt_price : ℚ := 3
def cheese_price : ℚ := 8
def milk_price : ℚ := 4

def ice_cream_discount : ℚ := 0.05
def yoghurt_tax : ℚ := 0.08
def cheese_discount : ℚ := 0.10

def returned_ice_cream : ℕ := 10
def returned_cheese : ℕ := 5

def adjusted_ice_cream_cost : ℚ :=
  (ice_cream_quantity * ice_cream_price) * (1 - ice_cream_discount) -
  (returned_ice_cream * ice_cream_price)

def adjusted_yoghurt_cost : ℚ :=
  (yoghurt_quantity * yoghurt_price) * (1 + yoghurt_tax)

def adjusted_cheese_cost : ℚ :=
  (cheese_quantity * cheese_price) * (1 - cheese_discount) -
  (returned_cheese * cheese_price)

def adjusted_milk_cost : ℚ :=
  milk_quantity * milk_price

theorem cost_difference :
  adjusted_ice_cream_cost + adjusted_cheese_cost -
  (adjusted_yoghurt_cost + adjusted_milk_cost) = 1146.60 := by
  sorry

end cost_difference_l766_76640


namespace linear_equation_solution_l766_76620

theorem linear_equation_solution (a b : ℝ) : 
  (a * (-2) - 3 * b * 3 = 5) → 
  (a * 4 - 3 * b * 1 = 5) → 
  a + b = 0 := by
sorry

end linear_equation_solution_l766_76620


namespace coefficient_x_term_expansion_l766_76628

theorem coefficient_x_term_expansion (x : ℝ) : 
  (∃ a b c d e : ℝ, (1 + x) * (2 - x)^4 = a*x^4 + b*x^3 + c*x^2 + d*x + e) → 
  (∃ a b c d e : ℝ, (1 + x) * (2 - x)^4 = a*x^4 + b*x^3 + c*x^2 + (-16)*x + e) :=
by sorry

end coefficient_x_term_expansion_l766_76628


namespace systematic_sampling_l766_76618

theorem systematic_sampling (total : Nat) (sample_size : Nat) (drawn : Nat) : 
  total = 800 → 
  sample_size = 50 → 
  drawn = 7 → 
  ∃ (selected : Nat), 
    selected = drawn + 2 * (total / sample_size) ∧ 
    33 ≤ selected ∧ 
    selected ≤ 48 := by
  sorry

end systematic_sampling_l766_76618


namespace square_min_rotation_angle_l766_76660

/-- The minimum rotation angle for a square to coincide with its original position -/
def min_rotation_angle_square : ℝ := 90

/-- A square has rotational symmetry of order 4 -/
def rotational_symmetry_order_square : ℕ := 4

theorem square_min_rotation_angle :
  min_rotation_angle_square = 360 / rotational_symmetry_order_square :=
by sorry

end square_min_rotation_angle_l766_76660


namespace unique_prime_square_product_l766_76689

theorem unique_prime_square_product (a b c : ℕ) : 
  (Nat.Prime (a^2 + 1)) ∧ 
  (Nat.Prime (b^2 + 1)) ∧ 
  ((a^2 + 1) * (b^2 + 1) = c^2 + 1) →
  a = 2 ∧ b = 1 ∧ c = 3 :=
by sorry

end unique_prime_square_product_l766_76689


namespace nonagon_diagonal_intersection_probability_l766_76696

/-- A regular nonagon is a 9-sided polygon with all sides equal and all angles equal. -/
def RegularNonagon : Type := Unit

/-- A diagonal of a regular nonagon is a line segment connecting two non-adjacent vertices. -/
def Diagonal (n : RegularNonagon) : Type := Unit

/-- Two diagonals intersect if they have a point in common inside the nonagon. -/
def Intersect (n : RegularNonagon) (d1 d2 : Diagonal n) : Prop := sorry

/-- The probability of an event is the number of favorable outcomes divided by the total number of possible outcomes. -/
def Probability (event : Prop) : ℚ := sorry

theorem nonagon_diagonal_intersection_probability (n : RegularNonagon) :
  Probability (∃ (d1 d2 : Diagonal n), Intersect n d1 d2) = 14 / 39 := by sorry

end nonagon_diagonal_intersection_probability_l766_76696


namespace max_area_equilateral_triangle_in_rectangle_l766_76659

/-- The maximum area of an equilateral triangle inscribed in a 12x5 rectangle --/
theorem max_area_equilateral_triangle_in_rectangle :
  ∃ (A : ℝ),
    A = (25 : ℝ) * Real.sqrt 3 / 3 ∧
    ∀ (s : ℝ),
      s > 0 →
      s ≤ 12 →
      s * Real.sqrt 3 / 2 ≤ 5 →
      (Real.sqrt 3 / 4) * s^2 ≤ A :=
by sorry

end max_area_equilateral_triangle_in_rectangle_l766_76659


namespace expand_polynomial_l766_76685

theorem expand_polynomial (x : ℝ) : (x + 3) * (4 * x^2 - 5 * x - 6) = 4 * x^3 + 7 * x^2 - 21 * x - 18 := by
  sorry

end expand_polynomial_l766_76685


namespace even_m_permutation_exists_l766_76603

/-- A permutation of numbers from 1 to m -/
def Permutation (m : ℕ) := { f : ℕ → ℕ // Function.Bijective f ∧ ∀ i, i ≤ m → f i ≤ m }

/-- Partial sums of a permutation -/
def PartialSums (m : ℕ) (p : Permutation m) : ℕ → ℕ
  | 0 => 0
  | n + 1 => PartialSums m p n + p.val (n + 1)

/-- Different remainders property -/
def DifferentRemainders (m : ℕ) (p : Permutation m) : Prop :=
  ∀ i j, i ≤ m → j ≤ m → i ≠ j → PartialSums m p i % m ≠ PartialSums m p j % m

theorem even_m_permutation_exists (m : ℕ) (h : m > 1) (he : Even m) :
  ∃ p : Permutation m, DifferentRemainders m p := by
  sorry

end even_m_permutation_exists_l766_76603


namespace compound_interest_time_calculation_l766_76652

/-- Proves that the time t satisfies the compound interest equation for the given problem --/
theorem compound_interest_time_calculation 
  (initial_investment : ℝ) 
  (annual_rate : ℝ) 
  (compounding_frequency : ℝ) 
  (final_amount : ℝ) 
  (h1 : initial_investment = 600)
  (h2 : annual_rate = 0.10)
  (h3 : compounding_frequency = 2)
  (h4 : final_amount = 661.5) :
  ∃ t : ℝ, final_amount = initial_investment * (1 + annual_rate / compounding_frequency) ^ (compounding_frequency * t) :=
sorry

end compound_interest_time_calculation_l766_76652


namespace trig_expression_value_quadratic_equation_solutions_quadratic_root_property_l766_76653

-- Part 1
theorem trig_expression_value : 
  2 * Real.tan (60 * π / 180) * Real.cos (30 * π / 180) - Real.sin (45 * π / 180) ^ 2 = 5/2 := by
sorry

-- Part 2
theorem quadratic_equation_solutions :
  let f : ℝ → ℝ := λ x => 2 * (x + 2)^2 - 3 * (x + 2)
  ∃ x₁ x₂ : ℝ, x₁ = -2 ∧ x₂ = -1/2 ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ 
  ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ := by
sorry

-- Part 3
theorem quadratic_root_property :
  ∀ m : ℝ, m^2 - 5*m - 2 = 0 → 2*m^2 - 10*m + 2023 = 2027 := by
sorry

end trig_expression_value_quadratic_equation_solutions_quadratic_root_property_l766_76653


namespace min_value_x_plus_sqrt_x2_y2_l766_76608

theorem min_value_x_plus_sqrt_x2_y2 (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 2) :
  ∃ (min : ℝ), min = 8/5 ∧ ∀ (z : ℝ), z > 0 → 2 * z + (2 - 2 * z) = 2 →
    x + Real.sqrt (x^2 + y^2) ≥ min ∧ z + Real.sqrt (z^2 + (2 - 2 * z)^2) ≥ min :=
by sorry

end min_value_x_plus_sqrt_x2_y2_l766_76608


namespace complex_fraction_sum_l766_76664

theorem complex_fraction_sum : 
  let U := 1 / (4 - Real.sqrt 15) - 1 / (Real.sqrt 15 - Real.sqrt 14) + 
           1 / (Real.sqrt 14 - 3) - 1 / (3 - Real.sqrt 12) + 
           1 / (Real.sqrt 12 - Real.sqrt 11)
  U = 10 + Real.sqrt 11 := by
  sorry

end complex_fraction_sum_l766_76664


namespace fraction_sum_squared_l766_76690

theorem fraction_sum_squared (x y z m n p : ℝ) 
  (h1 : x/m + y/n + z/p = 1)
  (h2 : m/x + n/y + p/z = 0) :
  x^2/m^2 + y^2/n^2 + z^2/p^2 = 1 := by
  sorry

end fraction_sum_squared_l766_76690


namespace coloring_books_per_shelf_l766_76679

theorem coloring_books_per_shelf 
  (initial_stock : ℕ) 
  (sold : ℕ) 
  (shelves : ℕ) 
  (h1 : initial_stock = 87) 
  (h2 : sold = 33) 
  (h3 : shelves = 9) 
  (h4 : shelves > 0) : 
  (initial_stock - sold) / shelves = 6 := by
sorry

end coloring_books_per_shelf_l766_76679


namespace min_transport_cost_l766_76609

/-- Represents the transportation problem between two villages and two destinations -/
structure TransportProblem where
  villageA_supply : ℝ
  villageB_supply : ℝ
  destX_demand : ℝ
  destY_demand : ℝ
  costA_to_X : ℝ
  costA_to_Y : ℝ
  costB_to_X : ℝ
  costB_to_Y : ℝ

/-- Calculates the total transportation cost given the amount transported from A to X -/
def totalCost (p : TransportProblem) (x : ℝ) : ℝ :=
  p.costA_to_X * x + p.costA_to_Y * (p.villageA_supply - x) +
  p.costB_to_X * (p.destX_demand - x) + p.costB_to_Y * (x - (p.villageA_supply + p.villageB_supply - p.destX_demand - p.destY_demand))

/-- The specific problem instance -/
def vegetableProblem : TransportProblem :=
  { villageA_supply := 80
  , villageB_supply := 60
  , destX_demand := 65
  , destY_demand := 75
  , costA_to_X := 50
  , costA_to_Y := 30
  , costB_to_X := 60
  , costB_to_Y := 45 }

/-- Theorem stating that the minimum transportation cost for the vegetable problem is 6100 -/
theorem min_transport_cost :
  ∃ x, x ≥ 0 ∧ x ≤ vegetableProblem.villageA_supply ∧
       x ≤ vegetableProblem.destX_demand ∧
       x ≥ (vegetableProblem.villageA_supply + vegetableProblem.villageB_supply - vegetableProblem.destX_demand - vegetableProblem.destY_demand) ∧
       totalCost vegetableProblem x = 6100 ∧
       ∀ y, y ≥ 0 → y ≤ vegetableProblem.villageA_supply →
             y ≤ vegetableProblem.destX_demand →
             y ≥ (vegetableProblem.villageA_supply + vegetableProblem.villageB_supply - vegetableProblem.destX_demand - vegetableProblem.destY_demand) →
             totalCost vegetableProblem x ≤ totalCost vegetableProblem y :=
by sorry


end min_transport_cost_l766_76609


namespace symmetry_sine_cosine_function_l766_76604

/-- Given a function f(x) = a*sin(x) + b*cos(x) where ab ≠ 0, 
    if the graph of f(x) is symmetric about x = π/6 and f(x₀) = 8/5 * a, 
    then sin(2x₀ + π/6) = 7/25 -/
theorem symmetry_sine_cosine_function 
  (a b x₀ : ℝ) 
  (h1 : a * b ≠ 0) 
  (f : ℝ → ℝ) 
  (h2 : ∀ x, f x = a * Real.sin x + b * Real.cos x) 
  (h3 : ∀ x, f (π/3 - x) = f (π/3 + x)) 
  (h4 : f x₀ = 8/5 * a) : 
  Real.sin (2*x₀ + π/6) = 7/25 := by
  sorry

end symmetry_sine_cosine_function_l766_76604


namespace inequality_system_solution_l766_76658

theorem inequality_system_solution (x : ℝ) : 
  (1 - (2*x - 1)/2 > (3*x - 1)/4 ∧ 2 - 3*x ≤ 4 - x) ↔ -1 ≤ x ∧ x < 1 := by
  sorry

end inequality_system_solution_l766_76658


namespace supplementary_angle_theorem_l766_76631

theorem supplementary_angle_theorem (x : ℝ) : 
  x > 0 ∧ x < 180 ∧ x = 3 * (180 - x) → x = 135 := by
  sorry

end supplementary_angle_theorem_l766_76631


namespace expression_equality_l766_76619

theorem expression_equality : 
  (Real.log 5) ^ 0 + (9 / 4) ^ (1 / 2) + Real.sqrt ((1 - Real.sqrt 2) ^ 2) - 2 ^ (Real.log 2 / Real.log 4) = 3 / 2 := by
  sorry

end expression_equality_l766_76619


namespace bank_deposit_time_calculation_l766_76613

/-- Proves that given two equal deposits at the same interest rate, 
    if the difference in interest is known, we can determine the time for the first deposit. -/
theorem bank_deposit_time_calculation 
  (deposit : ℝ) 
  (rate : ℝ) 
  (time_second : ℝ) 
  (interest_diff : ℝ) 
  (h1 : deposit = 640)
  (h2 : rate = 0.15)
  (h3 : time_second = 5)
  (h4 : interest_diff = 144) :
  ∃ (time_first : ℝ), 
    deposit * rate * time_second - deposit * rate * time_first = interest_diff ∧ 
    time_first = 3.5 := by
  sorry


end bank_deposit_time_calculation_l766_76613


namespace arithmetic_matrix_properties_l766_76681

/-- Represents a matrix with the given properties -/
def ArithmeticMatrix (n : ℕ) (d : ℕ → ℝ) : Prop :=
  n ≥ 3 ∧
  ∀ m k, m ≤ n → k ≤ n → 
    (∃ a : ℕ → ℕ → ℝ, 
      a m k = 1 + (k - 1) * d m ∧
      (∀ i, i ≤ n → a i 1 = 1) ∧
      (∀ i j, i ≤ n → j < n → a i (j + 1) - a i j = d i) ∧
      (∀ i j, i < n → j ≤ n → a (i + 1) j - a i j = a (i + 1) 1 - a i 1))

/-- The main theorem -/
theorem arithmetic_matrix_properties {n : ℕ} {d : ℕ → ℝ} 
  (h : ArithmeticMatrix n d) :
  (∃ c : ℝ, d 2 - d 1 = d 3 - d 2) ∧
  (∀ m, 3 ≤ m → m ≤ n → d m = (2 - m) * d 1 + (m - 1) * d 2) := by
  sorry

end arithmetic_matrix_properties_l766_76681


namespace max_value_base_conversion_l766_76636

theorem max_value_base_conversion (n A B C : ℕ) : 
  n > 0 →
  n = 64 * A + 8 * B + C →
  n = 81 * C + 9 * B + A →
  C % 2 = 0 →
  A ≤ 7 →
  B ≤ 7 →
  C ≤ 7 →
  n ≤ 64 :=
by sorry

end max_value_base_conversion_l766_76636


namespace breath_holding_improvement_l766_76654

/-- Calculates the final breath-holding time after three weeks of practice --/
def final_breath_holding_time (initial_time : ℝ) : ℝ :=
  let after_first_week := initial_time * 2
  let after_second_week := after_first_week * 2
  after_second_week * 1.5

/-- Theorem stating that given an initial breath-holding time of 10 seconds,
    the final time after three weeks of practice is 60 seconds --/
theorem breath_holding_improvement :
  final_breath_holding_time 10 = 60 := by
  sorry

#eval final_breath_holding_time 10

end breath_holding_improvement_l766_76654


namespace arithmetic_evaluation_l766_76612

theorem arithmetic_evaluation : (7 + 5 + 3) / 3 - 2 / 3 = 13 / 3 := by
  sorry

end arithmetic_evaluation_l766_76612


namespace number_puzzle_solution_l766_76662

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem number_puzzle_solution :
  ∀ (A B C : ℕ),
  (sum_of_digits A = B) →
  (sum_of_digits B = C) →
  (A + B + C = 60) →
  (A = 44 ∨ A = 50 ∨ A = 47) :=
by sorry

end number_puzzle_solution_l766_76662


namespace expression_value_l766_76676

theorem expression_value : 
  (1 - 2/7) / (0.25 + 3 * (1/4)) + (2 * 0.3) / (1.3 - 0.4) = 29/21 := by
  sorry

end expression_value_l766_76676


namespace expression_bounds_l766_76698

theorem expression_bounds (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 1) :
  14/27 ≤ x^3 + 2*y^2 + (10/3)*z ∧ x^3 + 2*y^2 + (10/3)*z ≤ 10/3 := by
sorry

end expression_bounds_l766_76698


namespace q_is_zero_l766_76635

/-- A cubic polynomial with roots at -2, 0, and 2, passing through (1, -3) -/
def g (p q r s : ℝ) (x : ℝ) : ℝ := p * x^3 + q * x^2 + r * x + s

theorem q_is_zero (p q r s : ℝ) :
  (∀ x, x = -2 ∨ x = 0 ∨ x = 2 → g p q r s x = 0) →
  g p q r s 1 = -3 →
  q = 0 :=
sorry

end q_is_zero_l766_76635


namespace min_product_reciprocal_sum_l766_76629

theorem min_product_reciprocal_sum (a b : ℕ+) (h : (a : ℚ)⁻¹ + (3 * b : ℚ)⁻¹ = 1/9) : 
  (∀ c d : ℕ+, (c : ℚ)⁻¹ + (3 * d : ℚ)⁻¹ = 1/9 → c * d ≥ a * b) ∧ a * b = 108 := by
  sorry

end min_product_reciprocal_sum_l766_76629


namespace prob_double_is_one_seventh_l766_76671

/-- The number of integers in the modified domino set -/
def n : ℕ := 13

/-- The total number of domino pairings in the set -/
def total_pairings : ℕ := n * (n + 1) / 2

/-- The number of doubles in the set -/
def num_doubles : ℕ := n

/-- The probability of selecting a double from the modified domino set -/
def prob_double : ℚ := num_doubles / total_pairings

theorem prob_double_is_one_seventh : prob_double = 1 / 7 := by
  sorry

end prob_double_is_one_seventh_l766_76671


namespace dollar_function_iteration_l766_76610

-- Define the dollar function
def dollar (N : ℝ) : ℝ := 0.3 * N + 2

-- State the theorem
theorem dollar_function_iteration : dollar (dollar (dollar 60)) = 4.4 := by
  sorry

end dollar_function_iteration_l766_76610


namespace coupon_probability_l766_76695

def total_coupons : ℕ := 17
def semyon_coupons : ℕ := 9
def temyon_missing : ℕ := 6

theorem coupon_probability : 
  (Nat.choose temyon_missing temyon_missing * Nat.choose (total_coupons - temyon_missing) (semyon_coupons - temyon_missing)) / 
  Nat.choose total_coupons semyon_coupons = 3 / 442 := by
  sorry

end coupon_probability_l766_76695


namespace negation_of_existence_proposition_l766_76655

theorem negation_of_existence_proposition :
  (¬∃ x : ℝ, x > 0 ∧ Real.sin x > 2^x - 1) ↔ (∀ x : ℝ, x > 0 → Real.sin x ≤ 2^x - 1) :=
by sorry

end negation_of_existence_proposition_l766_76655


namespace outfits_from_five_shirts_three_pants_l766_76674

/-- The number of outfits that can be made from a given number of shirts and pants -/
def number_of_outfits (shirts : ℕ) (pants : ℕ) : ℕ := shirts * pants

/-- Theorem: Given 5 shirts and 3 pairs of pants, the number of outfits is 15 -/
theorem outfits_from_five_shirts_three_pants : 
  number_of_outfits 5 3 = 15 := by
  sorry

end outfits_from_five_shirts_three_pants_l766_76674


namespace solve_problem_l766_76637

def problem (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧
  (x + y + 9 + 10 + 11) / 5 = 10 ∧
  ((x - 10)^2 + (y - 10)^2 + (9 - 10)^2 + (10 - 10)^2 + (11 - 10)^2) / 5 = 2

theorem solve_problem (x y : ℝ) (h : problem x y) : |x - y| = 4 :=
by sorry

end solve_problem_l766_76637


namespace polynomial_simplification_l766_76686

theorem polynomial_simplification (y : ℝ) : 
  (3*y - 2) * (5*y^11 + 3*y^10 + 5*y^9 + 3*y^8 + 5*y^7) = 
  15*y^12 - y^11 + 9*y^10 - y^9 + 9*y^8 - 10*y^7 := by
sorry

end polynomial_simplification_l766_76686


namespace production_rate_equation_l766_76617

theorem production_rate_equation (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_diff : x = y + 4) :
  (100 / x = 80 / y) ↔ 
  (∃ (rate_A rate_B : ℝ), 
    rate_A = x ∧ 
    rate_B = y ∧ 
    rate_A > rate_B ∧ 
    rate_A - rate_B = 4 ∧
    (100 / rate_A) = (80 / rate_B)) :=
by sorry

end production_rate_equation_l766_76617


namespace max_students_distribution_l766_76651

theorem max_students_distribution (pens toys books : ℕ) 
  (h_pens : pens = 451) 
  (h_toys : toys = 410) 
  (h_books : books = 325) : 
  (∃ (students : ℕ), students > 0 ∧ 
    pens % students = 0 ∧ 
    toys % students = 0 ∧ 
    books % students = 0) →
  (∀ (n : ℕ), n > 1 → 
    (pens % n ≠ 0 ∨ toys % n ≠ 0 ∨ books % n ≠ 0)) :=
by sorry

end max_students_distribution_l766_76651


namespace parallel_vectors_x_value_l766_76663

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  ∀ x : ℝ, are_parallel (2, x) (1, 2) → x = 4 := by
  sorry

end parallel_vectors_x_value_l766_76663


namespace oak_trees_after_five_days_l766_76649

/-- Calculates the final number of oak trees in the park after 5 days -/
def final_oak_trees (initial : ℕ) (plant_rate_1 plant_rate_2 remove_rate_1 remove_rate_2 : ℕ) : ℕ :=
  let net_change_1 := (plant_rate_1 - remove_rate_1) * 2
  let net_change_2 := (plant_rate_2 - remove_rate_1)
  let net_change_3 := (plant_rate_2 - remove_rate_2) * 2
  initial + net_change_1 + net_change_2 + net_change_3

/-- Theorem stating that given the initial number of oak trees and planting/removal rates, 
    the final number of oak trees after 5 days will be 15 -/
theorem oak_trees_after_five_days :
  final_oak_trees 5 3 4 2 1 = 15 := by
  sorry

end oak_trees_after_five_days_l766_76649


namespace propositions_correctness_l766_76639

-- Define the property of being even
def IsEven (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

-- Define the property of being divisible by 2
def DivisibleBy2 (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

-- Define the inequality from proposition ③
def Inequality (a x : ℝ) : Prop := (a^2 - 4) * x^2 + (a + 2) * x - 1 ≥ 0

theorem propositions_correctness :
  -- Proposition ②
  (¬ ∀ n : ℤ, DivisibleBy2 n → IsEven n) ↔ (∃ n : ℤ, DivisibleBy2 n ∧ ¬IsEven n)
  ∧
  -- Proposition ③
  ∃ a : ℝ, (¬ (|a| ≤ 1)) ∧ (∀ x : ℝ, ¬Inequality a x) :=
by sorry

end propositions_correctness_l766_76639


namespace line_segment_length_l766_76699

/-- Given points A, B, C, and D on a line in that order, prove that CD = 3 cm -/
theorem line_segment_length (A B C D : ℝ) : 
  (A < B) → (B < C) → (C < D) →  -- Points are in order on the line
  (B - A = 2) →                  -- AB = 2 cm
  (C - A = 5) →                  -- AC = 5 cm
  (D - B = 6) →                  -- BD = 6 cm
  (D - C = 3) :=                 -- CD = 3 cm (to be proved)
by sorry

end line_segment_length_l766_76699


namespace distance_to_x_axis_for_point_p_l766_76677

/-- The distance from a point to the x-axis in a Cartesian coordinate system --/
def distanceToXAxis (x y : ℝ) : ℝ := |y|

/-- Theorem: The distance from point P(3, -2) to the x-axis is 2 --/
theorem distance_to_x_axis_for_point_p :
  distanceToXAxis 3 (-2) = 2 := by sorry

end distance_to_x_axis_for_point_p_l766_76677


namespace complex_modulus_l766_76691

theorem complex_modulus (z : ℂ) (h : (1 + Complex.I) * z = -3 + Complex.I) : Complex.abs z = Real.sqrt 5 := by
  sorry

end complex_modulus_l766_76691


namespace round_trip_average_speed_l766_76627

/-- The average speed of a round trip journey given outbound and inbound speeds -/
theorem round_trip_average_speed 
  (outbound_speed inbound_speed : ℝ) 
  (outbound_speed_pos : outbound_speed > 0)
  (inbound_speed_pos : inbound_speed > 0)
  (h_outbound : outbound_speed = 44)
  (h_inbound : inbound_speed = 36) :
  2 * outbound_speed * inbound_speed / (outbound_speed + inbound_speed) = 39.6 := by
  sorry

#check round_trip_average_speed

end round_trip_average_speed_l766_76627


namespace circle_properties_l766_76670

-- Define the circle family
def circle_family (t : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*(t+3)*x - 2*t*y + t^2 + 4*t + 8 = 0

-- Define the line l
def line_l (x y : ℝ) : Prop := x + y - 3 = 0

-- Define the tangent line m
def line_m (x y : ℝ) : Prop := y = -1 ∧ x = 2

theorem circle_properties :
  -- Part 1: Centers lie on y = x - 3
  (∀ t : ℝ, t ≠ -1 → ∃ x y : ℝ, circle_family t x y ∧ y = x - 3) ∧
  -- Part 2: Maximum chord length is 2√2
  (∃ max_length : ℝ, max_length = 2 * Real.sqrt 2 ∧
    ∀ t : ℝ, t ≠ -1 →
      ∀ x₁ y₁ x₂ y₂ : ℝ,
        circle_family t x₁ y₁ ∧ line_l x₁ y₁ ∧
        circle_family t x₂ y₂ ∧ line_l x₂ y₂ →
        Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) ≤ max_length) ∧
  -- Part 3: Line m is tangent to all circles
  (∀ t : ℝ, t ≠ -1 →
    ∃ x y : ℝ, circle_family t x y ∧ line_m x y ∧
    ∀ x' y' : ℝ, circle_family t x' y' →
      (x' - x)^2 + (y' - y)^2 ≥ 0 ∧
      ((x' - x)^2 + (y' - y)^2 = 0 → x' = x ∧ y' = y)) :=
by
  sorry

end circle_properties_l766_76670


namespace quadratic_range_on_unit_interval_l766_76668

/-- The range of a quadratic function on a closed interval --/
theorem quadratic_range_on_unit_interval
  (a b c : ℝ) (ha : a < 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c
  ∃ (min max : ℝ), min = c ∧ max = -b^2 / (4 * a) + c ∧
    Set.Icc min max = Set.Icc 0 1 ∩ Set.range f :=
by sorry

end quadratic_range_on_unit_interval_l766_76668


namespace train_speed_problem_l766_76616

/-- Proves that given the conditions of the train problem, the speed of Train A is 43 miles per hour. -/
theorem train_speed_problem (speed_B : ℝ) (headstart : ℝ) (overtake_distance : ℝ) 
  (h1 : speed_B = 45)
  (h2 : headstart = 2)
  (h3 : overtake_distance = 180) :
  ∃ (speed_A : ℝ) (overtake_time : ℝ), 
    speed_A = 43 ∧ 
    speed_A * (headstart + overtake_time) = overtake_distance ∧
    speed_B * overtake_time = overtake_distance :=
by
  sorry


end train_speed_problem_l766_76616


namespace logarithm_equations_l766_76641

-- Define the common logarithm (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the logarithm with arbitrary base
noncomputable def log (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem logarithm_equations :
  (lg 4 + lg 500 - lg 2 = 3) ∧
  ((27 : ℝ)^(1/3) + (log 3 2) * (log 2 3) = 4) :=
by sorry

end logarithm_equations_l766_76641


namespace smallest_n_for_equal_cost_l766_76667

theorem smallest_n_for_equal_cost : ∃ n : ℕ+, 
  (∀ m : ℕ+, m < n → ¬∃ (r g b : ℕ+), 18 * r = 21 * g ∧ 21 * g = 25 * b ∧ 25 * b = 24 * m) ∧
  (∃ (r g b : ℕ+), 18 * r = 21 * g ∧ 21 * g = 25 * b ∧ 25 * b = 24 * n) ∧
  n = 132 :=
by sorry

end smallest_n_for_equal_cost_l766_76667


namespace ratio_from_S1_ratio_from_S1_S2_ratio_from_S2_l766_76601

/-- Represents a trapezoid with diagonals intersecting at a point -/
structure Trapezoid where
  S : ℝ  -- Area of the trapezoid
  S1 : ℝ  -- Area of triangle OBC
  S2 : ℝ  -- Area of triangle OCD
  S3 : ℝ  -- Area of triangle ODA
  S4 : ℝ  -- Area of triangle AOB
  AD : ℝ  -- Length of side AD
  BC : ℝ  -- Length of side BC

/-- There exists a function that determines AD/BC given S1/S -/
theorem ratio_from_S1 (t : Trapezoid) : 
  ∃ f : ℝ → ℝ, t.AD / t.BC = f (t.S1 / t.S) :=
sorry

/-- There exists a function that determines AD/BC given (S1+S2)/S -/
theorem ratio_from_S1_S2 (t : Trapezoid) : 
  ∃ f : ℝ → ℝ, t.AD / t.BC = f ((t.S1 + t.S2) / t.S) :=
sorry

/-- There exists a function that determines AD/BC given S2/S -/
theorem ratio_from_S2 (t : Trapezoid) : 
  ∃ f : ℝ → ℝ, t.AD / t.BC = f (t.S2 / t.S) :=
sorry

end ratio_from_S1_ratio_from_S1_S2_ratio_from_S2_l766_76601


namespace nested_fraction_evaluation_l766_76687

theorem nested_fraction_evaluation : 
  2 + 1 / (3 + 1 / (2 + 2)) = 30 / 13 := by
  sorry

end nested_fraction_evaluation_l766_76687


namespace age_difference_l766_76634

def arun_age : ℕ := 60

def gokul_age (a : ℕ) : ℕ := (a - 6) / 18

def madan_age (g : ℕ) : ℕ := g + 5

theorem age_difference : 
  madan_age (gokul_age arun_age) - gokul_age arun_age = 5 := by
  sorry

end age_difference_l766_76634


namespace tangent_sqrt_two_implications_l766_76630

theorem tangent_sqrt_two_implications (θ : Real) (h : Real.tan θ = Real.sqrt 2) :
  ((Real.cos θ + Real.sin θ) / (Real.cos θ - Real.sin θ) = -3 - 2 * Real.sqrt 2) ∧
  (Real.sin θ ^ 2 - Real.sin θ * Real.cos θ + 2 * Real.cos θ ^ 2 = (4 - Real.sqrt 2) / 3) := by
  sorry

end tangent_sqrt_two_implications_l766_76630


namespace min_sqrt_equality_l766_76611

theorem min_sqrt_equality (x y z : ℝ) : 
  x ≥ 1 ∧ y ≥ 1 ∧ z ≥ 1 →
  (min (Real.sqrt (x + x*y*z)) (min (Real.sqrt (y + x*y*z)) (Real.sqrt (z + x*y*z))) = 
   Real.sqrt (x - 1) + Real.sqrt (y - 1) + Real.sqrt (z - 1)) ↔
  ∃ t : ℝ, t > 0 ∧ 
    x = 1 + (t / (t^2 + 1))^2 ∧ 
    y = 1 + 1 / t^2 ∧ 
    z = 1 + t^2 :=
by sorry

end min_sqrt_equality_l766_76611


namespace wholesale_price_correct_l766_76646

/-- The retail price of the machine -/
def retail_price : ℝ := 167.99999999999997

/-- The discount rate applied to the retail price -/
def discount_rate : ℝ := 0.10

/-- The profit rate as a percentage of the wholesale price -/
def profit_rate : ℝ := 0.20

/-- The wholesale price of the machine -/
def wholesale_price : ℝ := 126.00

/-- Theorem stating that the given wholesale price is correct -/
theorem wholesale_price_correct : 
  wholesale_price = (retail_price * (1 - discount_rate)) / (1 + profit_rate) :=
sorry

end wholesale_price_correct_l766_76646


namespace gcd_210_162_l766_76633

theorem gcd_210_162 : Nat.gcd 210 162 = 6 := by
  sorry

end gcd_210_162_l766_76633


namespace period_of_cos_3x_l766_76680

/-- The period of cos(3x) is 2π/3 -/
theorem period_of_cos_3x :
  let f : ℝ → ℝ := λ x ↦ Real.cos (3 * x)
  ∃ T : ℝ, T > 0 ∧ ∀ x, f (x + T) = f x ∧ ∀ S, 0 < S ∧ S < T → ∃ x, f (x + S) ≠ f x :=
by sorry

end period_of_cos_3x_l766_76680


namespace gcf_98_140_245_l766_76621

theorem gcf_98_140_245 : Nat.gcd 98 (Nat.gcd 140 245) = 7 := by
  sorry

end gcf_98_140_245_l766_76621


namespace arrange_four_math_four_history_l766_76602

/-- The number of ways to arrange books on a shelf --/
def arrange_books (n_math : ℕ) (n_history : ℕ) : ℕ :=
  if n_math ≥ 2 then
    n_math * (n_math - 1) * (n_math + n_history - 2).factorial
  else
    0

/-- Theorem: Arranging 4 math books and 4 history books with math books on both ends --/
theorem arrange_four_math_four_history :
  arrange_books 4 4 = 8640 := by
  sorry

end arrange_four_math_four_history_l766_76602


namespace simplify_sqrt_expression_l766_76666

theorem simplify_sqrt_expression :
  (2 * Real.sqrt 10) / (Real.sqrt 4 + Real.sqrt 3 + Real.sqrt 5) =
  (4 * Real.sqrt 10 - 15 * Real.sqrt 2) / 11 := by
sorry

end simplify_sqrt_expression_l766_76666


namespace product_mod_seventeen_l766_76625

theorem product_mod_seventeen :
  (1234 * 1235 * 1236 * 1237 * 1238) % 17 = 9 := by
  sorry

end product_mod_seventeen_l766_76625


namespace f_increasing_l766_76678

-- Define the function f(x) = x³ + x + 1
def f (x : ℝ) : ℝ := x^3 + x + 1

-- Theorem statement
theorem f_increasing : ∀ x y : ℝ, x < y → f x < f y := by
  sorry

end f_increasing_l766_76678


namespace max_volume_cube_l766_76693

/-- A rectangular solid with length l, width w, and height h -/
structure RectangularSolid where
  l : ℝ
  w : ℝ
  h : ℝ
  l_pos : 0 < l
  w_pos : 0 < w
  h_pos : 0 < h

/-- The surface area of a rectangular solid -/
def surfaceArea (r : RectangularSolid) : ℝ :=
  2 * (r.l * r.w + r.l * r.h + r.w * r.h)

/-- The volume of a rectangular solid -/
def volume (r : RectangularSolid) : ℝ :=
  r.l * r.w * r.h

/-- Theorem: Among all rectangular solids with a fixed surface area S,
    the cube has the maximum volume, and this maximum volume is (S/6)^(3/2) -/
theorem max_volume_cube (S : ℝ) (h_pos : 0 < S) :
  ∃ (max_vol : ℝ),
    (∀ (r : RectangularSolid), surfaceArea r = S → volume r ≤ max_vol) ∧
    (∃ (cube : RectangularSolid), surfaceArea cube = S ∧ volume cube = max_vol) ∧
    max_vol = (S / 6) ^ (3/2) :=
  sorry

end max_volume_cube_l766_76693


namespace polar_to_rectangular_l766_76643

/-- Conversion from polar to rectangular coordinates -/
theorem polar_to_rectangular (r : ℝ) (θ : ℝ) :
  let (x, y) := (r * Real.cos θ, r * Real.sin θ)
  (x, y) = (5 / 2, -5 * Real.sqrt 3 / 2) ↔ r = 5 ∧ θ = 5 * Real.pi / 3 := by
  sorry

end polar_to_rectangular_l766_76643


namespace band_member_earnings_l766_76683

theorem band_member_earnings (attendees : ℕ) (ticket_price : ℝ) (band_share : ℝ) (band_members : ℕ) : 
  attendees = 500 → 
  ticket_price = 30 → 
  band_share = 0.7 → 
  band_members = 4 → 
  (attendees * ticket_price * band_share) / band_members = 2625 := by
sorry

end band_member_earnings_l766_76683
